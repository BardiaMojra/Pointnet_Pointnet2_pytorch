#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-
"""
build_dataset.py -- DLO segmentation dataset from raw L515 frames and the v034 ground truth.

Per episode of the eval_qm split (fit + hold-out; eqm_split.build(), read only, never saved):
N_FRAMES clean frames (with nodes and crop on disk) spread evenly over the episode. Per frame:
raw pcd (camera frame) -> robot frame with the episode's own lee_H/lee_offset through the
pipeline's transform_xyz_rgb -> fixed workspace box -> label 1 where a point is
  within eps of the frame's node polyline (dlo_nodes_pcd, pole end first), not past either end
  AND within eps of a dlo_raw_crop_pcd point,       eps = rope radius + LABEL_PAD_M.
The crop alone is not a label: it still holds the pole-top assembly and some gripper points.
Crop-only and polyline-only counts are kept per frame for comparison.
Hold-out episodes also give N_FLAGGED FAIL/WARN frames for qualitative renders.
Transform check per frame: NN distance from every crop point to the transformed raw points.

Writes (all under data/, git-ignored):
  frames/<batch>/<ep>/<ep>_f<frame>.npz   xyz float32 (robot frame, m), rgb uint8, label uint8
  flagged/<ep>_f<frame>.npz               same, for FAIL/WARN hold-out frames
  records/<ep>.json                       per-frame stats of one finished episode
Output goes only into an empty dir (general_rules §4): --clean true deletes this script's previous build.
  index.csv, meta.json                    every frame; box, label radii, split, summary numbers
  ../split_dlo.json                       the fit/hold-out episode lists used (tracked)

  dgx: docker run --rm --user 1000:1000 -e HOME=/home/smerx -v /home/smerx:/home/smerx \
         -v /media/smerx:/media/smerx:rslave -w ~/git/pointnetpp dlo_melodic \
         python3.8 dlo/build_dataset.py [--max_eps N] [--jobs 6]
Platforms: u18_a64 (dlo_melodic container on the dgx).
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool

# --- Defaults ---
SESSION_DIR = os.path.expanduser("~/bags/test_session_001")
GT_DIR = os.path.join(SESSION_DIR, "groundtruth")
DEV_DIR = os.path.expanduser("~/git/dlo_data_001/dev")
OUT_VER = "v034"            # ground-truth version read (_out_v034)
N_FRAMES = 30               # clean frames per episode, evenly spread
N_FLAGGED = 12              # FAIL/WARN hold-out frames for renders (3 per object)
LABEL_PAD_M = 0.003         # label radius = rope radius + this
BOX_MARGIN_M = 0.10         # workspace box = union of config crop limits + this margin
D004_DIAM_M = 0.01027       # objects.txt has no d004 diameter_m_meas; thickness-study value (its comment)
JOBS = 6                    # worker processes (another agent shares the host's CPUs)
MAX_EPS = 0                 # 0 = every episode; >0 = first N per batch (smoke runs)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "data")

PLATFORMS = ("u18_a64",)


def platform_id():
    """u<Ubuntu major>_<x86|a64>, as ~/git/bashrc/platform.sh computes it."""
    osr = {}
    try:
        with open("/etc/os-release") as f:
            for line in f:
                k, _, v = line.strip().partition("=")
                osr[k] = v.strip('"')
    except OSError:
        pass
    os_id = "u" if osr.get("ID") == "ubuntu" else osr.get("ID", "linux")
    arch = {"x86_64": "x86", "aarch64": "a64"}.get(os.uname()[4], os.uname()[4])
    return "%s%s_%s" % (os_id, osr.get("VERSION_ID", "").split(".")[0], arch)


if platform_id() not in PLATFORMS:
    sys.stderr.write("refusing to run %s: written for %s, but %s is %s\n" % (
        os.path.basename(__file__), " ".join(PLATFORMS), os.uname()[1], platform_id()))
    sys.exit(3)

sys.path[:0] = [DEV_DIR, os.path.join(DEV_DIR, "studies"), os.path.join(DEV_DIR, "eval_qm")]
# one BLAS thread per worker: JOBS processes, not JOBS x 20 threads, on a shared host
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np                                    # noqa: E402
from scipy.spatial import cKDTree                     # noqa: E402
import dlo_perception_v034 as P                       # noqa: E402  raw loading + camera->robot transform
import drive_pool                                     # noqa: E402
import eqm_split                                      # noqa: E402  the one train/hold-out split
from dlo_study_lib import discovery as D              # noqa: E402
from dlo_study_lib import objects as O                # noqa: E402


class _Calib(object):
    """The two Config fields transform_xyz_rgb() reads, from the episode's own config.json."""
    def __init__(self, cfg):
        self.lee_H = np.asarray(cfg["lee_H"], dtype=float)
        self.lee_offset = np.asarray(cfg["lee_offset"], dtype=float)


def label_radii():
    """{dlo: (label radius m, diameter m, source)}; radius = diameter_m_meas/2 + LABEL_PAD_M."""
    objs = O.load()
    out = {}
    for dlo in ("d001", "d002", "d003", "d004"):
        d, src = objs.get(dlo, {}).get("diameter_m_meas"), "objects.txt diameter_m_meas"
        if d is None:
            d, src = D004_DIAM_M, "D004_DIAM_M (thickness study; objects.txt blank)"
        out[dlo] = (d / 2.0 + LABEL_PAD_M, d, src)
    return out


def workspace_box(episodes):
    """[xmin, xmax, ymin, ymax, zmin, zmax]: union over every episode config of the background
    box (far wall pole-anchored as bg_far_y: <= pole_set_crop_y_max + dlo_backswing_m), + margin."""
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for batch, ep in episodes:
        cfg = D.load_config(os.path.join(GT_DIR, batch, ep, "_out_" + OUT_VER))
        y_far = max(cfg["rm_bg_crop_y_max"], cfg["pole_set_crop_y_max"] + cfg["dlo_backswing_m"])
        lo = np.minimum(lo, [cfg["rm_bg_crop_x_min"], cfg["rm_bg_crop_y_min"], cfg["rm_bg_crop_z_min"]])
        hi = np.maximum(hi, [cfg["rm_bg_crop_x_max"], y_far, cfg["rm_bg_crop_z_max"]])
    lo, hi = lo - BOX_MARGIN_M, hi + BOX_MARGIN_M
    return [float(v) for pair in zip(lo, hi) for v in pair]


def resolve_raw(mounts, batch, ep):
    """First drive in drive_pool priority order holding raw_data/<batch>/<ep>/pcd, or None."""
    for _name, m in mounts:
        p = os.path.join(m, "bags", drive_pool.SESSION_NAME, "raw_data", batch, ep, "pcd")
        if os.path.isdir(p):
            return p
    return None


def in_box(xyz, box):
    return ((xyz[:, 0] > box[0]) & (xyz[:, 0] < box[1]) & (xyz[:, 1] > box[2]) &
            (xyz[:, 1] < box[3]) & (xyz[:, 2] > box[4]) & (xyz[:, 2] < box[5]))


def polyline_mask(xyz, nodes, eps):
    """Points within eps of the node polyline, minus those past either end (nearest segment is
    the first/last one and the projection falls before node 0 / after the last node)."""
    mask = np.zeros(len(xyz), dtype=bool)
    near = np.all((xyz > nodes.min(0) - eps) & (xyz < nodes.max(0) + eps), axis=1)
    p = xyz[near]
    if len(nodes) < 2 or len(p) == 0:
        return mask
    a, ab = nodes[:-1], np.diff(nodes, axis=0)
    ap = p[:, None, :] - a[None]
    t = np.einsum("nsk,sk->ns", ap, ab) / np.maximum((ab ** 2).sum(1), 1e-12)
    d2 = ((ap - np.clip(t, 0, 1)[..., None] * ab[None]) ** 2).sum(-1)
    k = d2.argmin(1)
    tk = t[np.arange(len(p)), k]
    past_end = ((k == 0) & (tk < 0)) | ((k == len(ab) - 1) & (tk > 1))
    mask[np.where(near)[0]] = (d2[np.arange(len(p)), k] <= eps ** 2) & ~past_end
    return mask


def build_frame(od, raw_dir, raw_name, frame, calib, box, eps_m, out_path):
    """Load, transform, crop, label and save one frame; return its stats row."""
    crop_path = D.frame_file(od, "dlo_raw_crop_pcd", frame, "pcd")
    nodes_path = D.frame_file(od, "dlo_nodes_pcd", frame, "pcd")
    row = {"frame": frame, "npz": out_path, "has_crop": crop_path is not None, "has_nodes": nodes_path is not None}
    x = P.transform_xyz_rgb(P.load_point_cloud_with_rgb(os.path.join(raw_dir, raw_name)), calib)
    x = x[np.isfinite(x).all(axis=1)]
    xb = x[in_box(x, box)]
    label = np.zeros(len(xb), dtype=np.uint8)
    row.update(n_raw=len(x), n_box=len(xb), n_crop=0, crop_out_box=0, n_dlo_crop=0, n_dlo_poly=0,
               nn_med_mm=np.nan, nn_max_mm=np.nan, label_src="none")
    near_crop = np.zeros(len(xb), dtype=bool)
    if crop_path is not None:
        # raw file and crop must be the same capture: same seq/rostime suffix after the prefix
        if os.path.basename(crop_path).split("_", 1)[1] != raw_name.split("_", 1)[1]:
            raise ValueError("frame %d: crop %s does not match raw %s" % (frame, crop_path, raw_name))
        crop = P.load_point_cloud_with_rgb(crop_path)[:, :3]
        d_nn, _ = cKDTree(x[:, :3]).query(crop)
        d_lab, _ = cKDTree(crop).query(xb[:, :3], distance_upper_bound=eps_m)
        near_crop = d_lab <= eps_m
        label = near_crop.astype(np.uint8)
        row.update(n_crop=len(crop), crop_out_box=int((~in_box(crop, box)).sum()), n_dlo_crop=int(near_crop.sum()),
                   nn_med_mm=1e3 * float(np.median(d_nn)), nn_max_mm=1e3 * float(d_nn.max()), label_src="crop")
    if nodes_path is not None:
        near_poly = polyline_mask(xb[:, :3], P.load_point_cloud_with_rgb(nodes_path)[:, :3], eps_m)
        row["n_dlo_poly"] = int(near_poly.sum())
        if crop_path is not None:
            label = (near_poly & near_crop).astype(np.uint8)
            row["label_src"] = "poly&crop"
    row["n_dlo"] = int(label.sum())
    rgb = np.clip(np.round(xb[:, 3:6] * 255.0), 0, 255).astype(np.uint8)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, xyz=xb[:, :3].astype(np.float32), rgb=rgb, label=label)
    return row


def episode_job(task):
    """All N_FRAMES clean frames of one episode (plus its flagged candidates if hold-out)."""
    batch, ep, split, box, eps_m, raw_dir, out_dir = task
    rec_path = os.path.join(out_dir, "records", ep + ".json")
    if os.path.isfile(rec_path):
        with open(rec_path) as f:
            return json.load(f)
    t0 = time.time()
    od = os.path.join(GT_DIR, batch, ep, "_out_" + OUT_VER)
    fl = D.read_frame_log(od)
    fno = fl.frame_no().astype(int)
    status = np.array([s.strip().upper() for s in fl.text("status")])
    clean = fl.clean & ~fl.has_warn & (status == "PASS")
    on_disk = set(D.frames_on_disk(od, "dlo_nodes_pcd")) & set(D.frames_on_disk(od, "dlo_raw_crop_pcd"))
    with open(os.path.join(od, "eps_data", "pcd_fnames.txt")) as f:
        fnames = f.read().split()
    calib = _Calib(D.load_config(od))
    cand = np.array([f for f in fno[clean] if f in on_disk], dtype=int)
    pick = cand[np.unique(np.round(np.linspace(0, len(cand) - 1, N_FRAMES)).astype(int))]
    rows = []
    for fr in pick:
        out_path = os.path.join(out_dir, "frames", batch, ep, "%s_f%06d.npz" % (ep, fr))
        r = build_frame(od, raw_dir, fnames[fr - 1], int(fr), calib, box, eps_m, out_path)
        r.update(batch=batch, ep=ep, split=split, kind="clean", status="PASS", tags="")
        rows.append(r)
    flagged = []
    if split == "holdout":
        for i in np.where(np.isin(status, ("FAIL", "WARN")))[0]:
            tags = "|".join(fl.error_tags[i] + fl.warn_tags[i])
            flagged.append([int(fno[i]), str(status[i]), tags])
    rec = {"batch": batch, "ep": ep, "split": split, "rows": rows, "flagged_cand": flagged,
           "n_frames": int(fl.n_rows), "n_clean": int(clean.sum()), "n_cand": int(len(cand)), "sec": time.time() - t0}
    os.makedirs(os.path.dirname(rec_path), exist_ok=True)
    with open(rec_path + ".tmp", "w") as f:
        json.dump(rec, f, default=float)
    os.replace(rec_path + ".tmp", rec_path)
    return rec


def pick_flagged(recs, n):
    """n/4 flagged hold-out frames per object, FAIL first, stable hash order."""
    by_obj = {}
    for rec in recs:
        for fr, st, tags in rec["flagged_cand"]:
            key = hashlib.sha1(("%s_%d" % (rec["ep"], fr)).encode()).hexdigest()
            by_obj.setdefault(rec["batch"][:4], []).append((st != "FAIL", key, rec, fr, st, tags))
    out = []
    for dlo in sorted(by_obj):
        cands = sorted(by_obj[dlo], key=lambda c: (c[0], c[1]))
        n_fail = min(2, sum(1 for c in cands if not c[0]))
        warns = [c for c in cands if c[0]]
        out += cands[:n_fail] + warns[:n // 4 - n_fail]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--jobs", type=int, default=JOBS)
    ap.add_argument("--max_eps", type=int, default=MAX_EPS)
    ap.add_argument("--out_dir", default=OUT_DIR)
    ap.add_argument("--clean", default="false", help="true: delete this script's previous output first "
                    "(general_rules §4; deleting an old build needs smerx's approval)")
    args = ap.parse_args()
    # general_rules §4: output only into an empty dir. data/ also holds data/seg7 (build_dataset7.py), so this
    # script owns, checks and cleans only its own entries there
    own = [os.path.join(args.out_dir, n) for n in ("frames", "flagged", "records", "index.csv", "meta.json")]
    if any(os.path.exists(p) for p in own):
        if args.clean != "true":
            sys.exit("refusing to write into %s: a previous build is there (general_rules §4); pass --clean true to "
                     "delete it first" % args.out_dir)
        import shutil
        for p in own:
            if os.path.isdir(p):
                shutil.rmtree(p)
            elif os.path.exists(p):
                os.remove(p)
        print("deleted the previous build in %s (--clean true)" % args.out_dir, flush=True)

    split = eqm_split.build(existing=eqm_split.load())      # returned dict only; never saved
    episodes = []
    for batch, v in split["batches"].items():
        for part in ("fit", "holdout"):
            names = v[part][:args.max_eps] if args.max_eps else v[part]
            episodes += [(batch, ep, part) for ep in names]
    split_rec = {"source": "eqm_split.build(existing=eqm_split.load()), %s" % time.strftime("%Y-%m-%d %H:%M"),
                 "batches": {b: {"fit": v["fit"], "holdout": v["holdout"], "source": v["source"]}
                             for b, v in split["batches"].items()}}
    with open(os.path.join(HERE, "split_dlo.json"), "w") as f:
        json.dump(split_rec, f, indent=1)

    radii = label_radii()
    box = workspace_box([(b, e) for b, e, _ in episodes])
    mounts = drive_pool.all_mounts()
    tasks = []
    for batch, ep, part in episodes:
        raw_dir = resolve_raw(mounts, batch, ep)
        if raw_dir is None:
            raise IOError("no raw_data for %s/%s on any drive: %s" % (batch, ep, mounts))
        tasks.append((batch, ep, part, box, radii[batch[:4]][0], raw_dir, args.out_dir))
    print("episodes %d (fit %d, holdout %d), box %s, jobs %d" % (
        len(tasks), sum(t[2] == "fit" for t in tasks), sum(t[2] == "holdout" for t in tasks),
        np.round(box, 3).tolist(), args.jobs), flush=True)

    t0, recs = time.time(), []
    with Pool(args.jobs) as pool:
        for i, rec in enumerate(pool.imap_unordered(episode_job, tasks, chunksize=1)):
            recs.append(rec)
            if (i + 1) % 20 == 0 or i + 1 == len(tasks):
                print("  %d/%d episodes, %.0f s" % (i + 1, len(tasks), time.time() - t0), flush=True)

    # flagged hold-out frames for the qualitative renders (step 4)
    rows = [r for rec in recs for r in rec["rows"]]
    for _w, _k, rec, fr, st, tags in pick_flagged([r for r in recs if r["split"] == "holdout"], N_FLAGGED):
        od = os.path.join(GT_DIR, rec["batch"], rec["ep"], "_out_" + OUT_VER)
        with open(os.path.join(od, "eps_data", "pcd_fnames.txt")) as f:
            fnames = f.read().split()
        out_path = os.path.join(args.out_dir, "flagged", "%s_f%06d.npz" % (rec["ep"], fr))
        raw_dir = resolve_raw(mounts, rec["batch"], rec["ep"])
        r = build_frame(od, raw_dir, fnames[fr - 1], fr, _Calib(D.load_config(od)), box,
                        radii[rec["batch"][:4]][0], out_path)
        r.update(batch=rec["batch"], ep=rec["ep"], split="holdout", kind="flagged", status=st, tags=tags)
        rows.append(r)

    cols = ["batch", "ep", "split", "kind", "frame", "status", "tags", "label_src", "has_crop", "has_nodes",
            "n_raw", "n_box", "n_dlo", "n_dlo_crop", "n_dlo_poly", "n_crop", "crop_out_box", "nn_med_mm",
            "nn_max_mm", "npz"]
    rows.sort(key=lambda r: (r["kind"], r["batch"], r["ep"], r["frame"]))
    with open(os.path.join(args.out_dir, "index.csv"), "w") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([os.path.relpath(r[c], args.out_dir) if c == "npz" else r[c] for c in cols])

    # summary numbers for REPORT.md
    clean_rows = [r for r in rows if r["kind"] == "clean"]
    per_obj = {}
    for dlo in sorted(radii):
        rr = [r for r in clean_rows if r["batch"].startswith(dlo)]
        nb = max(1, sum(r["n_box"] for r in rr))
        per_obj[dlo] = {"frames": len(rr), "dlo_frac": sum(r["n_dlo"] for r in rr) / nb,
                        "dlo_frac_crop_only": sum(r["n_dlo_crop"] for r in rr) / nb,
                        "dlo_frac_poly_only": sum(r["n_dlo_poly"] for r in rr) / nb,
                        "n_box_mean": float(np.mean([r["n_box"] for r in rr])),
                        "n_dlo_mean": float(np.mean([r["n_dlo"] for r in rr])),
                        "n_dlo_crop_mean": float(np.mean([r["n_dlo_crop"] for r in rr])),
                        "label_eps_m": radii[dlo][0], "diam_m": radii[dlo][1], "diam_src": radii[dlo][2]}
    ep_nn = {}
    for r in clean_rows:
        ep_nn.setdefault(r["ep"], []).append(r["nn_med_mm"])
    ep_med = np.array([np.median(v) for v in ep_nn.values()])
    size_mb = sum(os.path.getsize(os.path.join(dp, fn)) for dp, _d, fns in os.walk(args.out_dir)
                  for fn in fns if fn.endswith(".npz")) / 1e6
    meta = {"gt_version": OUT_VER, "label_rule": "near node polyline (ends cut) AND near crop, eps = r + pad",
            "box": box, "box_margin_m": BOX_MARGIN_M, "label_pad_m": LABEL_PAD_M,
            "n_frames_per_ep": N_FRAMES, "per_object": per_obj,
            "split_frames": {s: sum(r["split"] == s for r in clean_rows) for s in ("fit", "holdout")},
            "split_episodes": {s: sum(rec["split"] == s for rec in recs) for s in ("fit", "holdout")},
            "eps_short_of_n_frames": sum(len(rec["rows"]) < N_FRAMES for rec in recs),
            "flagged_frames": sum(r["kind"] == "flagged" for r in rows),
            "transform_check_mm": {"episodes": len(ep_med), "median_of_ep_medians": float(np.median(ep_med)),
                                   "max_ep_median": float(np.max(ep_med)),
                                   "max_single_point": float(np.nanmax([r["nn_max_mm"] for r in clean_rows]))},
            "crop_points_outside_box": int(sum(r["crop_out_box"] for r in clean_rows)),
            "npz_total_mb": size_mb, "build_sec": time.time() - t0}
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
