#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-
"""
build_dataset7.py -- 7-class scene-segmentation dataset (run r02) from raw L515 frames and v034 GT.

Classes, labelled from geometry and pipeline regions (colour only where the pipeline's own region
test uses it: the red finger and the red cloth):
  0 other / background
  1 rope body          within eps of the v034 node path (ends cut), not class 2 or 5
  2 rope gripper end   rope points in the last END_M (arc length) of the node path, gripper side
  3 gripper            non-rope points in the FK gripper capsule (gripper_fk_capsule) or red points
                       near the FK marker (remove_gripper_red_finger: the red right finger)
  4 pole and mount     non-rope points in the fitted shaft capsule (fit_pole_shaft + remove_pole_shaft;
                       an episode whose own fit fails gets the batch's median slopes, radius and z range,
                       placed at its own pole marker minus the batch's median marker-to-axis offset)
                       or in v035's pole-top static voxels (static_voxels) inside MOUNT_BOX around
                       the frame's pole marker (the static-voxel region also holds the backdrop), or
                       in the shaft capsule's axis extended from the fit's bottom down to the table
                       plane (the fit starts at z 0.16 m; v035 also removes that base structure)
  5 rope pole end      rope points in the first END_M of the node path, pole side
  6 table / red cloth  red points in the pole-bottom box (remove_set_polbott, red-only) or points
                       within TABLE_BAND_M of the episode's table plane
Precedence: rope (1, 2, 5) > gripper (3) > table/cloth (6) > pole (4) > other (0).
eps = rope radius + 3 mm (build_dataset.label_radii). Pipeline functions are imported read only
(dlo_perception_v034, v035) and applied with an index column appended, so each mask is exactly the
set of points that function drops; nothing is written outside data/seg7/.

Frames: N_FRAMES clean frames per episode (fit and hold-out, eqm_split as build_dataset.py), evenly
spread; a frame failing the crop guard is replaced by the nearest passing clean frame. Crop guard =
share of v034 crop points > CROP_FAR_M from the node path must be <= CROP_FAR_FRAC, measured over
  --guard_mode v034       the whole crop (as specified; rejects ~99% of v034 clean frames, since the
                          v034 crop still holds the pole-top assembly)
  --guard_mode explained  crop points outside the pole / gripper / table regions above (what v035
                          has already removed when its own guard runs); the r02 default
Hold-out also gets the 12 flagged frames of data/index.csv. Per episode, once: shaft fit, static
voxels and table plane from STATIC_FRAMES raw frames spread over the episode.
Writes data/seg7/{frames,flagged,records}/, data/seg7/index.csv, data/seg7/meta.json (git-ignored).
  dgx: docker run --rm --user 1000:1000 -e HOME=/home/smerx -v /home/smerx:/home/smerx \
         -v /media/smerx:/media/smerx:rslave -w ~/git/pointnetpp dlo_melodic \
         python3.8 dlo/build_dataset7.py [--max_eps N] [--jobs 6] [--guard_mode v034|explained]
Platforms: u18_a64 (dlo_melodic container on the dgx).
"""
import argparse
import contextlib
import csv
import glob
import io
import json
import os
import sys
import time
from multiprocessing import Pool

# --- Defaults ---
N_FRAMES = 30                # clean frames per episode
END_M = 0.05                 # tied end length (arc length along the node path) for classes 2 and 5
CROP_FAR_M = 0.05            # crop guard: a crop point this far from the node path is not rope
CROP_FAR_FRAC = 0.20         # ... more than this share of such points: frame excluded
GUARD_MODE = "explained"     # r02 default (smerx 2026-09-14); "v034" = whole crop (see docstring)
TABLE_NEAR_Z = (0.10, 0.03)  # table_planes.csv flags planes within 0.03 m of z = 0.10 m
GUARD_SEARCH = 15            # candidate frames tried either side of a rejected one
MOUNT_BOX = (-0.08, 0.08, -0.03, 0.12, -0.10, 0.10)   # dx, dy, dz around the pole marker: pole mount region
TABLE_BAND_M = 0.015         # table surface = within this of the episode's table plane
TABLE_SEARCH_Z = (-0.10, 0.15)   # table plane = densest 1 cm z-bin in this range (in-box points)
STATIC_FRAMES = 20           # raw frames per episode for the static map and the table plane
JOBS = 6
MAX_EPS = 0                  # 0 = every episode; >0 = first N per batch and part (smoke runs)
# v035 Config defaults for static_voxels() (dlo_perception_v035.py:452-453, 617-627); v034 configs lack them
V035_POLE_TOP = dict(pole_top_static_box_pad=[0.10, 0.35, 0.02, 0.15, 0.61, 0.10], pole_top_static_voxel_m=0.008,
                     pole_top_static_frac=0.5, pole_top_static_frames=STATIC_FRAMES, pole_top_static_dilate=1,
                     pole_top_static_low_z=0.16, pole_top_static_low_frac=0.2)
CLASSES = ("other", "rope body", "rope gripper end", "gripper", "pole and mount", "rope pole end", "table / red cloth")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "data", "seg7")

PLATFORMS = ("u18_a64",)
sys.path.insert(0, HERE)
import build_dataset as B     # noqa: E402  its own guard (u18_a64), split, raw loading, radii, box, drive pool
if B.platform_id() not in PLATFORMS:
    sys.exit("refusing to run %s: written for %s, this is %s" % (os.path.basename(__file__), PLATFORMS, B.platform_id()))
import numpy as np            # noqa: E402
with contextlib.redirect_stdout(io.StringIO()):
    import dlo_perception_v035 as M    # noqa: E402  static_voxels, pole_top_box, _points_to_polyline (read only)
P, D = B.P, B.D


def quiet():
    return contextlib.redirect_stdout(io.StringIO())


class _NS(object):
    """Attribute bag: an episode's config.json (numeric lists as arrays), a frame's df, eps_data."""
    pass


def load_cfg(od, raw_dir):
    c = _NS()
    for k, v in D.load_config(od).items():
        if isinstance(v, list):
            try:
                v = np.asarray(v, dtype=float)
            except (TypeError, ValueError):
                pass
        setattr(c, k, v)
    c.in_pcd_dir = raw_dir
    for k, v in V035_POLE_TOP.items():
        setattr(c, k, np.asarray(v, dtype=float) if isinstance(v, list) else v)
    return c


def frame_df(od, frame):
    """The frame's logged poses (frame_data/<frame>_*.json) as the df fields the region functions read."""
    df = _NS()
    df.lee_pose_rob, df.pole_pose, df.lee_lrbox, df.pole_prbox = None, np.full(3, np.nan), None, None
    hits = sorted(glob.glob(os.path.join(od, "frame_data", "%06d_*.json" % frame)))
    if hits:
        with open(hits[0]) as f:
            fd = json.load(f)
        arr = lambda v: np.asarray(v, dtype=float) if v is not None else None
        df.lee_pose_rob = arr(fd.get("lee_pose_rob"))
        pp = arr(fd.get("pole_pose"))
        df.pole_pose = pp[:3] if pp is not None and pp.size >= 3 else np.full(3, np.nan)
        df.lee_lrbox, df.pole_prbox = arr(fd.get("lee_lrbox")), arr(fd.get("pole_prbox"))
    return df


def hsv_mask(rgb01, lo, hi):
    hsv = P.get_rgb2hsv_set(rgb01)
    return np.all((hsv > np.asarray(lo) / [359, 99, 99]) & (hsv < np.asarray(hi) / [359, 99, 99]), axis=1)


def is_red(rgb01, cfg):
    return hsv_mask(rgb01, cfg.hsv_red_low_min, cfg.hsv_red_low_max) | hsv_mask(rgb01, cfg.hsv_red_high_min, cfg.hsv_red_high_max)


def in_box6(xyz, b):
    if b is None:
        return np.zeros(len(xyz), bool)
    b = np.asarray(b, dtype=float)
    return np.all((xyz > b[[0, 2, 4]]) & (xyz < b[[1, 3, 5]]), axis=1) if b.size == 6 else np.zeros(len(xyz), bool)


def rope_arclen(xyz, nodes, eps):
    """(mask within eps of the node path and not past either end, arc length at the projection, path length)."""
    mask, s = np.zeros(len(xyz), dtype=bool), np.full(len(xyz), np.nan)
    a, ab = nodes[:-1], np.diff(nodes, axis=0)
    seg = np.linalg.norm(ab, axis=1)
    L = float(seg.sum())
    near = np.all((xyz > nodes.min(0) - eps) & (xyz < nodes.max(0) + eps), axis=1)
    p = xyz[near]
    if len(nodes) < 2 or len(p) == 0:
        return mask, s, L
    ap = p[:, None, :] - a[None]
    t = np.einsum("nsk,sk->ns", ap, ab) / np.maximum(seg ** 2, 1e-12)
    d2 = ((ap - np.clip(t, 0, 1)[..., None] * ab[None]) ** 2).sum(-1)
    k = d2.argmin(1)
    r = np.arange(len(p))
    tk = t[r, k]
    past_end = ((k == 0) & (tk < 0)) | ((k == len(ab) - 1) & (tk > 1))
    ok = (d2[r, k] <= eps ** 2) & ~past_end
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    w = np.where(near)[0]
    mask[w] = ok
    s[w] = cum[k] + np.clip(tk, 0, 1) * seg[k]
    return mask, s, L


def removed_by(fn, x6, *args):
    """Mask of the points a pipeline removal function drops (index column appended, output rows mapped back)."""
    x7 = np.hstack([x6, np.arange(len(x6), dtype=float)[:, None]])
    with quiet():
        kept = fn(x7, *args)
    m = np.ones(len(x6), dtype=bool)
    m[kept[:, 6].astype(int)] = False
    return m


def region_masks(x6, df, cfg, eps, static, table_z):
    """Non-rope region masks for xyz + rgb01 rows: FK capsule, red finger, red cloth, table band,
    shaft capsule, static mount voxels."""
    xyz, n = x6[:, :3], len(x6)
    z = lambda: np.zeros(n, bool)
    caps = P.gripper_fk_capsule(df, cfg) if df.lee_pose_rob is not None and df.lee_pose_rob.size == 7 else None
    m = {"caps": P._points_in_capsule_mask(xyz, *caps) if caps is not None else z(),
         "red_finger": removed_by(P.remove_gripper_red_finger, x6, df, cfg) if caps is not None else z(),
         "cloth": removed_by(P.remove_set_polbott, x6, cfg),
         "band": np.abs(xyz[:, 2] - table_z) <= TABLE_BAND_M if table_z is not None else z(),
         "shaft": removed_by(P.remove_pole_shaft, x6, df, cfg, eps) if eps.pole_shaft is not None else z(),
         "shaft_base": z(), "stat": z()}
    sh = eps.pole_shaft
    if sh is not None and table_z is not None and table_z < sh["z_lo"]:
        # the shaft fit starts at z 0.16 m: its axis, extended down to the table plane, same radius + margin
        axis = lambda zz: np.array([sh["wx"][0] + sh["wx"][1] * zz, sh["wy"][0] + sh["wy"][1] * zz, zz])
        m["shaft_base"] = P._points_in_capsule_mask(xyz, axis(table_z), axis(sh["z_lo"]),
                                                    sh["radius"] + cfg.pole_shaft_excl_margin_m)
    if static:
        pc = df.pole_pose if np.isfinite(df.pole_pose).all() else np.array(
            [(cfg.pole_set_crop_x_min + cfg.pole_set_crop_x_max) / 2, (cfg.pole_set_crop_y_min + cfg.pole_set_crop_y_max) / 2,
             (cfg.pole_set_crop_z_min + cfg.pole_set_crop_z_max) / 2])
        w = np.where(in_box6(xyz, M.pole_top_box(cfg)) & in_box6(xyz - pc, MOUNT_BOX))[0]
        keys = np.floor(xyz[w] / cfg.pole_top_static_voxel_m).astype(np.int64)
        m["stat"][w] = np.fromiter((tuple(k) in static for k in keys), dtype=bool, count=len(w))
    return m


def crop_far_shares(od, frame, cfg, eps, static, table_z):
    """(far share of the whole v034 crop, far share of the crop points outside the pole/gripper/table regions)."""
    c = D.frame_file(od, "dlo_raw_crop_pcd", frame, "pcd")
    k = D.frame_file(od, "dlo_nodes_pcd", frame, "pcd")
    if c is None or k is None:
        return np.nan, np.nan
    with quiet():
        crop, nodes = P.load_point_cloud_with_rgb(c), P.load_point_cloud_with_rgb(k)[:, :3]
    if len(crop) == 0 or len(nodes) < 2:
        return np.nan, np.nan
    far = M._points_to_polyline(crop[:, :3], nodes) > CROP_FAR_M
    rm = region_masks(crop, frame_df(od, frame), cfg, eps, static, table_z)
    explained = rm["caps"] | rm["red_finger"] | rm["cloth"] | rm["band"] | rm["shaft"] | rm["shaft_base"] | rm["stat"]
    return float(far.mean()), float((far & ~explained).mean())


def episode_maps(cfg, fnames, box, shaft=None):
    """Shaft (the pre-pass fit or its batch fallback, see main), pole-top static voxels (v035
    static_voxels) and table plane z."""
    eps = _NS()
    eps.pcd_fnames, eps.pole_shaft, eps.pole_pose_calib = fnames, shaft, None
    n = len(fnames)
    clouds = []
    for i in np.unique(np.linspace(0, n - 1, min(STATIC_FRAMES, n)).astype(int)):
        with quiet():
            clouds.append(P.transform_xyz_rgb(P.load_point_cloud_with_rgb(os.path.join(cfg.in_pcd_dir, fnames[i])), cfg)[:, :3])
    with quiet():
        static = M.static_voxels(clouds, cfg)
    z = np.concatenate([c[B.in_box(c, box), 2] for c in clouds])
    z = z[(z > TABLE_SEARCH_Z[0]) & (z < TABLE_SEARCH_Z[1])]
    table_z = None
    if len(z) > 500 * len(clouds):
        h, e = np.histogram(z, bins=np.arange(TABLE_SEARCH_Z[0], TABLE_SEARCH_Z[1] + 1e-9, 0.01))
        c0 = e[np.argmax(h)] + 0.005
        table_z = float(np.median(z[np.abs(z - c0) <= 0.01]))
    return eps, static, table_z


def shaft_job(task):
    """Pre-pass: v034 fit_pole_shaft() for one episode -> (batch, ep, fit or None)."""
    batch, ep, raw_dir = task
    od = os.path.join(B.GT_DIR, batch, ep, "_out_" + B.OUT_VER)
    with open(os.path.join(od, "eps_data", "pcd_fnames.txt")) as f:
        fnames = f.read().split()
    eps = _NS()
    eps.pcd_fnames, eps.pole_shaft, eps.pole_pose_calib = fnames, None, None
    with quiet():
        fit = P.fit_pole_shaft(eps, load_cfg(od, raw_dir))
    if fit is not None:
        fit["src"] = "own fit"
    return batch, ep, fit


def pole_marker(batch, ep):
    """Median pole marker position of the episode (v034 eps_data/pole_poses.txt), or None."""
    p = os.path.join(B.GT_DIR, batch, ep, "_out_" + B.OUT_VER, "eps_data", "pole_poses.txt")
    try:
        a = np.loadtxt(p, delimiter=",", ndmin=2)
    except (OSError, ValueError):
        return None
    a = a[np.isfinite(a).all(axis=1)] if a.size else a
    return np.median(a[:, :3], axis=0) if len(a) else None


def batch_fallback(fits, markers):
    """Shaft for episodes whose own fit fails: the batch's median slopes, radius and z range, with the
    axis placed at the episode's own pole marker minus the batch's median marker-to-axis offset (the pole
    can shift between episodes and its marker moves with it). Returns ({ep: fit}, {batch: spread})."""
    out, spread = {}, {}
    axis_at = lambda f, z: np.array([f["wx"][0] + f["wx"][1] * z, f["wy"][0] + f["wy"][1] * z])
    for b in sorted(set(bb for bb, _ in fits)):
        own = [(e, f) for (bb, e), f in fits.items() if bb == b and f is not None]
        if not own:
            continue
        med = lambda k, i=None: float(np.median([f[k] if i is None else f[k][i] for _, f in own]))
        offs = np.array([markers[e][:2] - axis_at(f, markers[e][2]) for e, f in own if markers.get(e) is not None])
        v = np.median(offs, axis=0) if len(offs) else None
        base = {"wx": [med("wx", 0), med("wx", 1)], "wy": [med("wy", 0), med("wy", 1)], "z_lo": med("z_lo"),
                "z_hi": med("z_hi"), "radius": med("radius"), "lean": med("lean"), "slices": 0, "frames": 0}
        for (bb, e), f in fits.items():
            if bb != b or f is not None:
                continue
            m = markers.get(e)
            if m is None or v is None:
                out[e] = dict(base, src="batch median of %d fits" % len(own))
            else:
                a = m[:2] - v
                out[e] = dict(base, wx=[float(a[0] - base["wx"][1] * m[2]), base["wx"][1]],
                              wy=[float(a[1] - base["wy"][1] * m[2]), base["wy"][1]],
                              src="pole marker - batch median offset (%d fits)" % len(own))
        ax = np.array([axis_at(f, 0.30) for _, f in own])
        spread[b] = {"own_fits": len(own), "axis_xy_std_at_z0.30_m": np.round(ax.std(0), 4).tolist(),
                     "radius_p05_p95_m": np.round(np.percentile([f["radius"] for _, f in own], [5, 95]), 4).tolist(),
                     "marker_to_axis_offset_xy_m": None if v is None else np.round(v, 4).tolist(),
                     "offset_deviation_p50_p95_m": None if v is None else
                     np.round(np.percentile(np.linalg.norm(offs - v, axis=1), [50, 95]), 4).tolist()}
    return out, spread


def build_frame7(od, raw_dir, raw_name, frame, cfg, eps, static, table_z, box, eps_m, out_path):
    """Load, transform, crop to the box, label 7 classes, save; return the stats row."""
    crop_path = D.frame_file(od, "dlo_raw_crop_pcd", frame, "pcd")
    nodes_path = D.frame_file(od, "dlo_nodes_pcd", frame, "pcd")
    if crop_path is not None and os.path.basename(crop_path).split("_", 1)[1] != raw_name.split("_", 1)[1]:
        raise ValueError("frame %d: crop %s does not match raw %s" % (frame, crop_path, raw_name))
    with quiet():
        x = P.transform_xyz_rgb(P.load_point_cloud_with_rgb(os.path.join(raw_dir, raw_name)), cfg)
    x = x[np.isfinite(x).all(axis=1)]
    xb = x[B.in_box(x, box)]
    xyz, rgb = xb[:, :3], xb[:, 3:6]
    df = frame_df(od, frame)
    n = len(xb)
    row = {"frame": frame, "npz": out_path, "has_nodes": nodes_path is not None, "n_box": n, "table_z": table_z}

    rope, s, L = np.zeros(n, bool), np.full(n, np.nan), np.nan
    if nodes_path is not None:
        with quiet():
            nodes = P.load_point_cloud_with_rgb(nodes_path)[:, :3]
        rope, s, L = rope_arclen(xyz, nodes, eps_m)
    pole_end = rope & (s <= END_M)
    grip_end = rope & (s >= L - END_M) & ~pole_end
    rm = region_masks(xb, df, cfg, eps, static, table_z)
    grip = ~rope & (rm["caps"] | rm["red_finger"])
    table = ~rope & ~grip & (rm["cloth"] | rm["band"])
    pole = ~rope & ~grip & ~table & (rm["shaft"] | rm["shaft_base"] | rm["stat"])

    lab = np.zeros(n, dtype=np.uint8)
    lab[rope] = 1
    lab[grip_end] = 2
    lab[pole_end] = 5
    lab[grip] = 3
    lab[table] = 6
    lab[pole] = 4
    for c in range(7):
        row["n_c%d" % c] = int((lab == c).sum())
    # sanity checks: where the geometric end classes sit, and what colour confirms
    row["c2_in_caps_or_lrbox"] = int(((lab == 2) & (rm["caps"] | in_box6(xyz, df.lee_lrbox))).sum())
    row["c5_in_prbox"] = int(((lab == 5) & in_box6(xyz, df.pole_prbox)).sum())
    row["c2_green"] = int(hsv_mask(rgb[lab == 2], cfg.hsv_green_min, cfg.hsv_green_max).sum()) if (lab == 2).any() else 0
    row["c5_red"] = int(is_red(rgb[lab == 5], cfg).sum()) if (lab == 5).any() else 0
    row["c4_blue"] = int(hsv_mask(rgb[lab == 4], cfg.hsv_blue_min, cfg.hsv_blue_max).sum()) if (lab == 4).any() else 0
    row["c3_red_finger"] = int((grip & rm["red_finger"] & ~rm["caps"]).sum())
    row["c6_cloth_red"] = int((table & rm["cloth"]).sum())
    row["path_len_m"] = L
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, xyz=xyz.astype(np.float32),
                        rgb=np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8), label=lab)
    return row


def episode_job(task):
    batch, ep, split, box, eps_m, raw_dir, out_dir, flagged, guard_mode, shaft = task
    rec_path = os.path.join(out_dir, "records", ep + ".json")
    if os.path.isfile(rec_path):
        with open(rec_path) as f:
            return json.load(f)
    t0 = time.time()
    od = os.path.join(B.GT_DIR, batch, ep, "_out_" + B.OUT_VER)
    fl = D.read_frame_log(od)
    fno = fl.frame_no().astype(int)
    status = np.array([s.strip().upper() for s in fl.text("status")])
    clean = fl.clean & ~fl.has_warn & (status == "PASS")
    on_disk = set(D.frames_on_disk(od, "dlo_nodes_pcd")) & set(D.frames_on_disk(od, "dlo_raw_crop_pcd"))
    with open(os.path.join(od, "eps_data", "pcd_fnames.txt")) as f:
        fnames = f.read().split()
    cfg = load_cfg(od, raw_dir)
    eps, static, table_z = episode_maps(cfg, fnames, box, shaft)
    cand = [int(f) for f in fno[clean] if f in on_disk]
    targets = np.unique(np.round(np.linspace(0, len(cand) - 1, N_FRAMES)).astype(int))
    used, picks, n_checked, n_fail, n_repl, n_unfilled = set(), [], 0, 0, 0, 0
    n_fail_mode = {"v034": 0, "explained": 0}      # the same checked frames judged by both guard modes
    for ti in targets:
        found = None
        for off in [0] + [o for k in range(1, GUARD_SEARCH + 1) for o in (k, -k)]:
            j = ti + off
            if j < 0 or j >= len(cand) or cand[j] in used:
                continue
            far_all, far_unexpl = crop_far_shares(od, cand[j], cfg, eps, static, table_z)
            far = far_all if guard_mode == "v034" else far_unexpl
            n_checked += 1
            n_fail_mode["v034"] += int(not (np.isfinite(far_all) and far_all <= CROP_FAR_FRAC))
            n_fail_mode["explained"] += int(not (np.isfinite(far_unexpl) and far_unexpl <= CROP_FAR_FRAC))
            if np.isfinite(far) and far <= CROP_FAR_FRAC:
                found = (cand[j], far_all, far_unexpl, off != 0)
                break
            n_fail += 1
        if found is None:
            n_unfilled += 1
            continue
        used.add(found[0])
        picks.append(found)
        n_repl += int(found[3])
    rows = []
    for fr, far_all, far_unexpl, repl in picks:
        out_path = os.path.join(out_dir, "frames", batch, ep, "%s_f%06d.npz" % (ep, fr))
        r = build_frame7(od, raw_dir, fnames[fr - 1], fr, cfg, eps, static, table_z, box, eps_m, out_path)
        r.update(batch=batch, ep=ep, split=split, kind="clean", status="PASS", tags="", crop_far=far_all,
                 crop_far_unexpl=far_unexpl, guard_repl=repl)
        rows.append(r)
    for fr, st, tags in flagged:
        out_path = os.path.join(out_dir, "flagged", "%s_f%06d.npz" % (ep, fr))
        r = build_frame7(od, raw_dir, fnames[fr - 1], fr, cfg, eps, static, table_z, box, eps_m, out_path)
        far_all, far_unexpl = crop_far_shares(od, fr, cfg, eps, static, table_z)
        r.update(batch=batch, ep=ep, split=split, kind="flagged", status=st, tags=tags, crop_far=far_all,
                 crop_far_unexpl=far_unexpl, guard_repl=False)
        rows.append(r)
    sh = eps.pole_shaft
    rec = {"batch": batch, "ep": ep, "split": split, "rows": rows, "n_cand": len(cand), "n_guard_checked": n_checked,
           "n_guard_fail": n_fail, "n_guard_fail_v034": n_fail_mode["v034"], "n_guard_fail_explained": n_fail_mode["explained"],
           "n_guard_replaced": n_repl, "n_unfilled": n_unfilled, "table_z": table_z,
           "n_static_voxels": len(static) if static else 0, "guard_mode": guard_mode,
           "shaft": None if sh is None else {k: sh[k] for k in ("wx", "wy", "radius", "z_lo", "z_hi", "lean", "slices", "src")},
           "sec": time.time() - t0}
    os.makedirs(os.path.dirname(rec_path), exist_ok=True)
    with open(rec_path + ".tmp", "w") as f:
        json.dump(rec, f, default=float)
    os.replace(rec_path + ".tmp", rec_path)
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--jobs", type=int, default=JOBS)
    ap.add_argument("--max_eps", type=int, default=MAX_EPS)
    ap.add_argument("--guard_mode", default=GUARD_MODE, choices=("v034", "explained"))
    ap.add_argument("--out_dir", default=OUT_DIR)
    ap.add_argument("--clean", default="false", help="true: delete a non-empty --out_dir first (general_rules §4: "
                    "output only into an empty dir, so no dir mixes runs; deleting an old build needs smerx's approval)")
    args = ap.parse_args()
    if os.path.isdir(args.out_dir) and os.listdir(args.out_dir):
        if args.clean != "true":
            sys.exit("refusing to write into non-empty %s (general_rules §4): pass --clean true to delete it first"
                     % args.out_dir)
        import shutil
        shutil.rmtree(args.out_dir)
        print("deleted old %s (--clean true)" % args.out_dir, flush=True)
    with quiet():
        split = B.eqm_split.build(existing=B.eqm_split.load())     # returned dict only; never saved to eval/
    # the split this build used, tracked next to the code (eval_qm can re-save split.json later)
    with open(os.path.join(HERE, "split_seg7.json"), "w") as f:
        json.dump({"source": "eqm_split.build(existing=eqm_split.load()), %s" % time.strftime("%Y-%m-%d %H:%M"),
                   "batches": {b: {"fit": v["fit"], "holdout": v["holdout"], "source": v["source"]}
                               for b, v in split["batches"].items()}}, f, indent=1)
    meta1 = json.load(open(os.path.join(HERE, "data", "meta.json")))
    box = meta1["box"]                                            # same workspace box as r01
    radii = B.label_radii()
    flagged = {}
    for r in csv.DictReader(open(os.path.join(HERE, "data", "index.csv"))):
        if r["kind"] == "flagged":
            flagged.setdefault(r["ep"], []).append((int(r["frame"]), r["status"], r["tags"]))
    episodes = []
    for batch, v in split["batches"].items():
        for part in ("fit", "holdout"):
            names = v[part][:args.max_eps] if args.max_eps else v[part]
            episodes += [(batch, ep, part) for ep in names]
    mounts = B.drive_pool.all_mounts()
    raw = {e: B.resolve_raw(mounts, b, e) for b, e, _ in episodes}
    t0 = time.time()
    with Pool(args.jobs) as pool:                  # pre-pass: every episode's own shaft fit
        fits = {(b, e): f for b, e, f in pool.imap_unordered(shaft_job, [(b, e, raw[e]) for b, e, _ in episodes])}
    markers = {e: pole_marker(b, e) for b, e, _ in episodes}
    fallback, spread = batch_fallback(fits, markers)
    shaft = {e: fits[(b, e)] if fits[(b, e)] is not None else fallback.get(e) for b, e, _ in episodes}
    print("shaft pre-pass %.0f s: own fits %d/%d, batch fallback %d, none %d" % (
        time.time() - t0, sum(f is not None for f in fits.values()), len(fits),
        sum(fits[(b, e)] is None and shaft[e] is not None for b, e, _ in episodes),
        sum(shaft[e] is None for _, e, _ in episodes)), flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "shaft_fits.json"), "w") as f:
        json.dump({"per_episode": {e: shaft[e] for _, e, _ in episodes}, "batch_fallback": fallback,
                   "batch_spread": spread}, f, indent=1, default=float)
    tasks = [(b, e, p, box, radii[b[:4]][0], raw[e], args.out_dir, flagged.get(e, []), args.guard_mode, shaft[e])
             for b, e, p in episodes]
    print("episodes %d (fit %d, holdout %d), guard %s, jobs %d" % (
        len(tasks), sum(t[2] == "fit" for t in tasks), sum(t[2] == "holdout" for t in tasks), args.guard_mode, args.jobs), flush=True)
    t0, recs = time.time(), []
    with Pool(args.jobs) as pool:
        for i, rec in enumerate(pool.imap_unordered(episode_job, tasks, chunksize=1)):
            recs.append(rec)
            if (i + 1) % 20 == 0 or i + 1 == len(tasks):
                print("  %d/%d episodes, %.0f s" % (i + 1, len(tasks), time.time() - t0), flush=True)

    rows = sorted((r for rec in recs for r in rec["rows"]), key=lambda r: (r["kind"], r["batch"], r["ep"], r["frame"]))
    cols = ["batch", "ep", "split", "kind", "frame", "status", "tags", "crop_far", "crop_far_unexpl", "guard_repl", "table_z",
            "path_len_m", "n_box"] + ["n_c%d" % c for c in range(7)] + [
            "c2_in_caps_or_lrbox", "c5_in_prbox", "c2_green", "c5_red", "c4_blue", "c3_red_finger", "c6_cloth_red",
            "has_nodes", "npz"]
    with open(os.path.join(args.out_dir, "index.csv"), "w") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([os.path.relpath(r[c], args.out_dir) if c == "npz" else r[c] for c in cols])

    # one row per episode: its table plane, static map, shaft fit and guard counts
    near = lambda z: z is not None and abs(z - TABLE_NEAR_Z[0]) <= TABLE_NEAR_Z[1]
    with open(os.path.join(args.out_dir, "table_planes.csv"), "w") as f:
        w = csv.writer(f)
        w.writerow(["batch", "ep", "split", "table_z_m", "near_0.10_m", "n_static_voxels", "shaft_fit", "shaft_src",
                    "shaft_radius_m", "shaft_z_lo_m", "shaft_z_hi_m", "frames", "guard_checked", "guard_fail_explained",
                    "guard_fail_v034", "guard_replaced"])
        for rec in sorted(recs, key=lambda r: (r["batch"], r["ep"])):
            sh = rec["shaft"] or {}
            w.writerow([rec["batch"], rec["ep"], rec["split"], "" if rec["table_z"] is None else round(rec["table_z"], 4),
                        near(rec["table_z"]), rec["n_static_voxels"], sh.get("src") == "own fit", sh.get("src", "none"),
                        round(sh.get("radius", np.nan), 4),
                        round(sh.get("z_lo", np.nan), 3), round(sh.get("z_hi", np.nan), 3),
                        sum(x["kind"] == "clean" for x in rec["rows"]), rec["n_guard_checked"],
                        rec["n_guard_fail_explained"], rec["n_guard_fail_v034"], rec["n_guard_replaced"]])

    clean = [r for r in rows if r["kind"] == "clean"]

    def shares(rr):
        tot = np.array([sum(r["n_c%d" % c] for r in rr) for c in range(7)], dtype=float)
        return {CLASSES[c]: float(tot[c] / max(1.0, tot.sum())) for c in range(7)}

    fit = [r for r in clean if r["split"] == "fit"]
    f_fit = np.array(list(shares(fit).values())) if fit else np.full(7, np.nan)
    tz = [rec["table_z"] for rec in recs if rec["table_z"] is not None]
    csum = lambda k: sum(r[k] for r in clean)
    q = lambda v: np.percentile(v, [5, 50, 95]).round(3).tolist() if len(v) else None
    meta = {"gt_version": B.OUT_VER, "classes": list(CLASSES), "box": box, "end_m": END_M, "mount_box": MOUNT_BOX,
            "table_band_m": TABLE_BAND_M,
            "crop_guard": {"mode": args.guard_mode, "far_m": CROP_FAR_M, "far_frac": CROP_FAR_FRAC,
                           "checked": sum(r["n_guard_checked"] for r in recs), "failed": sum(r["n_guard_fail"] for r in recs),
                           "failed_if_v034_mode": sum(r["n_guard_fail_v034"] for r in recs),
                           "failed_if_explained_mode": sum(r["n_guard_fail_explained"] for r in recs),
                           "explained_regions": "FK gripper capsule | red points within gripper_red_excl_radius of the "
                                                "FK marker | red points in the pole-bottom box | table band +-%.3f m | "
                                                "pole-shaft capsule | shaft axis extended to the table plane | static "
                                                "voxels in the mount box %s m around the pole marker" % (TABLE_BAND_M, MOUNT_BOX),
                           "replaced": sum(r["n_guard_replaced"] for r in recs), "unfilled": sum(r["n_unfilled"] for r in recs),
                           "kept_far_whole_crop_p05_p50_p95": q([r["crop_far"] for r in clean]),
                           "kept_far_unexplained_p05_p50_p95": q([r["crop_far_unexpl"] for r in clean])},
            "episodes": {s: sum(r["split"] == s for r in recs) for s in ("fit", "holdout")},
            "episodes_without_frames": sum(len([x for x in r["rows"] if x["kind"] == "clean"]) == 0 for r in recs),
            "frames": {s: sum(r["split"] == s for r in clean) for s in ("fit", "holdout")},
            "flagged_frames": sum(r["kind"] == "flagged" for r in rows),
            "class_share_all_clean": shares(clean),
            "class_share_by_object": {d: shares([r for r in clean if r["batch"].startswith(d)]) for d in ("d001", "d002", "d003", "d004")},
            "class_weight_sqrt_inv_freq_fit": (np.sqrt(np.nanmax(f_fit) / np.maximum(f_fit, 1e-9))).round(3).tolist(),
            "shaft_own_fit": sum((rec["shaft"] or {}).get("src") == "own fit" for rec in recs),
            "shaft_fallback": sum((rec["shaft"] or {}).get("src", "own fit") != "own fit" for rec in recs),
            "shaft_none": sum(rec["shaft"] is None for rec in recs), "shaft_batch_spread": spread,
            "static_voxels_median": float(np.median([r["n_static_voxels"] for r in recs])),
            "table_z": {"episodes_with_plane": len(tz), "median": float(np.median(tz)) if tz else None,
                        "p05_p95": np.percentile(tz, [5, 95]).round(3).tolist() if tz else None,
                        "near_0.10_m": sorted(r["ep"] for r in recs if near(r["table_z"]))},
            "sanity": {"c2_in_fk_capsule_or_lrbox": csum("c2_in_caps_or_lrbox") / max(1, csum("n_c2")),
                       "c5_in_pole_prbox": csum("c5_in_prbox") / max(1, csum("n_c5")),
                       "c2_green": csum("c2_green") / max(1, csum("n_c2")), "c5_red": csum("c5_red") / max(1, csum("n_c5")),
                       "c4_blue": csum("c4_blue") / max(1, csum("n_c4")),
                       "c3_red_finger_outside_capsule": csum("c3_red_finger") / max(1, csum("n_c3")),
                       "c6_red_cloth": csum("c6_cloth_red") / max(1, csum("n_c6"))},
            "build_sec": time.time() - t0}
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
