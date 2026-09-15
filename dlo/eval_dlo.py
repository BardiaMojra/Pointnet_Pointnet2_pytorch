#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_dlo.py -- hold-out evaluation of a pn2_dlo checkpoint on dlo/data.

Per frame: fixed-seed random N-point subsample of the in-box cloud -> model -> the sampled points'
P(DLO) spread to every in-box point by 1-NN -> DLO IoU, precision, recall on all in-box points
(pooled over points per object, speed and overall; also on the sampled points only).
Two operating points: argmax (P >= 0.5) and the threshold with the best IoU on the training
validation frames (fit-split episodes, sha1 % val_mod == 0; hold-out never used to pick it).
Timing on the GB10, B=1 after warmup: model only, and end to end from an in-box cloud on the host
(copy to GPU, subsample, model, 1-NN spread, copy back).
Renders to figs/ at the calibrated threshold: every flagged (FAIL/WARN) hold-out frame, and the
lowest-IoU clean frame per object.
Writes logs/<run>/eval_holdout<tag>.json and eval_frames<tag>.csv.
  dgx: docker run --rm --gpus all --ipc host --user 1000:1000 -v /home/smerx:/home/smerx \
         -w ~/git/pointnetpp dlo_torch python3 dlo/eval_dlo.py --run r01
Platforms: u24_a64 (dlo_torch container on the dgx).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import platform_guard                                                        # noqa: E402
platform_guard.require(("u24_a64",), __file__)

import argparse                                                              # noqa: E402
import csv                                                                   # noqa: E402
import json                                                                  # noqa: E402
import time                                                                  # noqa: E402

import numpy as np                                                           # noqa: E402
import torch                                                                 # noqa: E402

import dlo_data                                                              # noqa: E402
import pn2_dlo                                                               # noqa: E402
import viz_dlo                                                               # noqa: E402

# --- Defaults ---
HERE = os.path.dirname(os.path.abspath(__file__))
RUN = "r01"
CKPT = "best.pth"
N_POINTS = 0               # 0 = the checkpoint's training n_points
SEED = 7
WARMUP = 10                # hold-out frames run before timing starts
NN_CHUNK = 8192
RENDER = "true"
THR_GRID = np.round(np.arange(0.30, 0.951, 0.05), 2)   # candidate P(DLO) thresholds (includes 0.5)


def counts(pred, lab):
    return np.array([np.sum(pred & lab), np.sum(pred & ~lab), np.sum(~pred & lab)], dtype=np.int64)


def prf(c):
    tp, fp, fn = [int(v) for v in c]
    return {"iou": tp / max(1, tp + fp + fn), "prec": tp / max(1, tp + fp), "rec": tp / max(1, tp + fn)}


def predict(model, xyz, rgb, center, n_points, seed, amp):
    """(P(DLO) on all points, P(DLO) on sampled points, sample idx, model ms, end-to-end ms)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(xyz), n_points, replace=len(xyz) < n_points)
    torch.manual_seed(seed)                     # random centroids in pn2_dlo are drawn with torch.rand
    x_np = dlo_data.to_input(xyz, rgb, center)
    torch.cuda.synchronize(); t0 = time.time()
    X = torch.from_numpy(x_np).cuda()
    xs = X[:, torch.from_numpy(idx).cuda()].unsqueeze(0)
    torch.cuda.synchronize(); t1 = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        prob_s = model(xs)[0][:, 1].exp()
    torch.cuda.synchronize(); t2 = time.time()
    a, s = X[:3].T.contiguous(), xs[0, :3].T.contiguous()
    nn_idx = torch.cat([torch.cdist(a[i:i + NN_CHUNK], s).argmin(1) for i in range(0, a.shape[0], NN_CHUNK)])
    prob = prob_s[nn_idx].cpu().numpy()
    torch.cuda.synchronize(); t3 = time.time()
    return prob, prob_s.cpu().numpy(), idx, 1e3 * (t2 - t1), 1e3 * (t3 - t0)


def val_rows(cfg):
    """The training run's validation frames (same rule as train_dlo.py)."""
    fit = dlo_data.read_index(split="fit")
    out = []
    for ep in sorted(set(r["ep"] for r in fit if dlo_data.is_val(r["ep"], cfg["val_mod"]))):
        rr = [r for r in fit if r["ep"] == ep]
        out += [rr[i] for i in np.unique(np.linspace(0, len(rr) - 1, cfg["val_frames_per_ep"]).astype(int))]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default=RUN)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--n_points", type=int, default=N_POINTS)
    ap.add_argument("--render", default=RENDER)
    ap.add_argument("--max_frames", type=int, default=0, help="0 = every hold-out clean frame; >0 = evenly strided subset (smoke)")
    ap.add_argument("--tag", default="", help="suffix for the output file names (smoke runs)")
    args = ap.parse_args()
    ck = torch.load(os.path.join(HERE, "checkpoints", args.run, args.ckpt), map_location="cuda", weights_only=False)
    cfg = ck["cfg"]
    pn2_dlo.SAMPLING["random_min"] = cfg["random_min"]
    amp = cfg["amp"] == "true"
    n_points = args.n_points or cfg["n_points"]
    model = pn2_dlo.get_model().cuda()
    model.load_state_dict(ck["model"])
    model.eval()
    torch.backends.cudnn.benchmark = True
    meta = dlo_data.load_meta()
    center, box = dlo_data.box_center(meta["box"]), meta["box"]
    log_dir = os.path.join(HERE, "logs", args.run)
    fig_dir = os.path.join(HERE, "figs")

    # 1. threshold on the validation frames (fit split), IoU over all in-box points
    vr = val_rows(cfg)
    c_thr = np.zeros((len(THR_GRID), 3), dtype=np.int64)
    for i, r in enumerate(vr):
        xyz, rgb, lab = dlo_data.load_frame(dlo_data.DATA_DIR, r)
        prob = predict(model, xyz, rgb, center, n_points, 100000 + i, amp)[0]
        lab = lab.astype(bool)
        for k, t in enumerate(THR_GRID):
            c_thr[k] += counts(prob >= t, lab)
    val_iou = [prf(c)["iou"] for c in c_thr]
    thr = float(THR_GRID[int(np.argmax(val_iou))])
    print("val frames %d: IoU at 0.5 = %.4f, best threshold %.2f -> IoU %.4f" % (
        len(vr), val_iou[list(THR_GRID).index(0.5)], thr, max(val_iou)), flush=True)

    # 2. hold-out clean frames (+ flagged frames for renders)
    rows = dlo_data.read_index(split="holdout", kind="clean")
    if args.max_frames:
        rows = rows[::max(1, len(rows) // args.max_frames)]
    flagged = dlo_data.read_index(split="holdout", kind="flagged")
    groups = {"argmax": {}, "cal": {}}
    frames, tm, tt = [], [], []
    c_sampled = np.zeros(3, dtype=np.int64)
    for i, r in enumerate(rows + flagged):
        xyz, rgb, lab = dlo_data.load_frame(dlo_data.DATA_DIR, r)
        lab = lab.astype(bool)
        prob, prob_s, idx, ms_model, ms_total = predict(model, xyz, rgb, center, n_points, SEED + i, amp)
        pred = prob >= thr
        c, c05 = counts(pred, lab), counts(prob >= 0.5, lab)
        m, m05 = prf(c), prf(c05)
        f = dict(batch=r["batch"], ep=r["ep"], frame=r["frame"], kind=r["kind"], status=r["status"],
                 tags=r["tags"], label_src=r["label_src"], n_box=len(xyz), n_dlo=int(lab.sum()), n_pred=int(pred.sum()),
                 tp=int(c[0]), fp=int(c[1]), fn=int(c[2]), iou=m["iou"], prec=m["prec"], rec=m["rec"],
                 iou_argmax=m05["iou"], model_ms=ms_model, total_ms=ms_total)
        frames.append(f)
        if r["kind"] == "clean":
            if i >= WARMUP:
                tm.append(ms_model); tt.append(ms_total)
            c_sampled += counts(prob_s >= thr, lab[idx])
            for g in ("all", "obj:" + r["batch"][:4], "speed:" + r["batch"][5:], "batch:" + r["batch"]):
                for key, cc in (("argmax", c05), ("cal", c)):
                    groups[key][g] = groups[key].get(g, np.zeros(3, dtype=np.int64)) + cc
        elif args.render == "true":
            # before = what v034 saved as this frame's DLO cloud; after = model; then model vs label
            has = r["label_src"] != "none"
            crop = viz_dlo.v034_crop_mask(xyz, r["batch"], r["ep"], r["frame"])
            rows_fig = []
            if crop is not None:
                rows_fig.append(("before: v034 dlo_raw_crop_pcd", [("v034 crop", crop, viz_dlo.COL["crop"])]))
            rows_fig.append(("after: PointNet++ (P>=%.2f)" % thr, [("pred DLO", pred, viz_dlo.COL["label"])]))
            if has:
                rows_fig.append(("after vs label (%s)" % r["label_src"], viz_dlo.overlay_sets(label=lab, pred=pred)))
            viz_dlo.render(xyz, rgb, rows_fig, "%s f%d | %s %s | IoU vs pipeline %.3f" % (
                r["ep"], r["frame"], r["status"], r["tags"], m["iou"]),
                os.path.join(fig_dir, "flagged_%s_f%06d.png" % (r["ep"], r["frame"])), box)
        if (i + 1) % 500 == 0:
            print("  %d/%d frames, running IoU %.4f" % (i + 1, len(rows) + len(flagged), prf(groups["cal"]["all"])["iou"]),
                  flush=True)

    clean = [f for f in frames if f["kind"] == "clean"]
    fi = np.array([f["iou"] for f in clean])
    out = {"run": args.run, "ckpt": args.ckpt, "epoch": ck["epoch"], "val_iou_ckpt": ck.get("val_iou"),
           "n_points": n_points, "frames": len(clean), "episodes": len(set(f["ep"] for f in clean)),
           "threshold": {"cal": thr, "grid": THR_GRID.tolist(), "val_iou": val_iou, "val_frames": len(vr)},
           "pooled_argmax": {g: dict(prf(c), tp=int(c[0]), fp=int(c[1]), fn=int(c[2])) for g, c in sorted(groups["argmax"].items())},
           "pooled_cal": {g: dict(prf(c), tp=int(c[0]), fp=int(c[1]), fn=int(c[2])) for g, c in sorted(groups["cal"].items())},
           "sampled_points_only_cal": prf(c_sampled),
           "frame_iou_cal": {"mean": float(fi.mean()), "median": float(np.median(fi)), "p05": float(np.percentile(fi, 5)),
                             "frac_below_0.5": float(np.mean(fi < 0.5)), "frac_below_0.8": float(np.mean(fi < 0.8))},
           "timing_ms": {"model_median": float(np.median(tm)), "model_p90": float(np.percentile(tm, 90)),
                         "total_median": float(np.median(tt)), "total_p90": float(np.percentile(tt, 90)),
                         "n_box_median": float(np.median([f["n_box"] for f in clean])), "gpu": torch.cuda.get_device_name()},
           "flagged": [{k: f[k] for k in ("ep", "frame", "status", "tags", "label_src", "n_dlo", "n_pred", "iou", "prec", "rec")}
                       for f in frames if f["kind"] == "flagged"]}
    with open(os.path.join(log_dir, "eval_holdout%s.json" % args.tag), "w") as fh:
        json.dump(out, fh, indent=1)
    with open(os.path.join(log_dir, "eval_frames%s.csv" % args.tag), "w") as fh:
        w = csv.DictWriter(fh, fieldnames=list(frames[0].keys()))
        w.writeheader()
        w.writerows(frames)

    if args.render == "true":
        for dlo in ("d001", "d002", "d003", "d004"):
            f = min((f for f in clean if f["batch"].startswith(dlo)), key=lambda f: f["iou"])
            i = next(j for j, r in enumerate(rows) if r["ep"] == f["ep"] and r["frame"] == f["frame"])
            xyz, rgb, lab = dlo_data.load_frame(dlo_data.DATA_DIR, rows[i])
            pred = predict(model, xyz, rgb, center, n_points, SEED + i, amp)[0] >= thr
            viz_dlo.render(xyz, rgb, [("model (P>=%.2f) vs label" % thr, viz_dlo.overlay_sets(label=lab.astype(bool), pred=pred))],
                           "worst hold-out %s: %s f%d | IoU %.3f P %.3f R %.3f" % (
                               dlo, f["ep"], f["frame"], f["iou"], f["prec"], f["rec"]),
                           os.path.join(fig_dir, "worst_%s_%s_f%06d.png" % (dlo, f["ep"], f["frame"])), box)
    print(json.dumps({k: out[k] for k in ("threshold", "sampled_points_only_cal", "frame_iou_cal", "timing_ms")}, indent=1))
    for key in ("pooled_argmax", "pooled_cal"):
        print(key)
        for g, v in out[key].items():
            print("  %-16s IoU %.4f P %.4f R %.4f" % (g, v["iou"], v["prec"], v["rec"]))


if __name__ == "__main__":
    main()
