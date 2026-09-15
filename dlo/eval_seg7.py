#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_seg7.py -- hold-out evaluation of a multi-class pn2_dlo checkpoint (r02: 7 classes) on dlo/data/seg7.

Per hold-out clean frame: fixed-seed random N-point subsample of the in-box cloud -> model argmax -> the
sampled points' classes spread to every in-box point by 1-NN -> confusion matrix over all in-box points,
pooled per object, speed, batch and overall -> per-class IoU, precision, recall and mean IoU; "rope" =
classes 1+2+5 merged, for comparison with r01's DLO IoU. Timing on the GB10, B=1 after warmup: model only,
and end to end from an in-box cloud on the host (copy, subsample, model, 1-NN spread, copy back).
Renders: per object the median and the lowest mean-IoU hold-out frame, and every flagged (FAIL/WARN)
frame, label vs prediction; charts: training curves, per-class metrics, confusion matrix. All to
dlo/figs/ and, captioned, to dlo_data_001/dev/dev_figs/v036-pn2-<run>_NN_<what>.png (viz_seg7).
Writes logs/<run>/eval_holdout<tag>.json and eval_frames<tag>.csv.
  dgx: docker run --rm --gpus all --ipc host --user 1000:1000 -v /home/smerx:/home/smerx \
         -w ~/git/pointnetpp dlo_torch python3 dlo/eval_seg7.py --run r02
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
import viz_dlo as V                                                          # noqa: E402
import viz_seg7 as S                                                         # noqa: E402

# --- Defaults ---
HERE = os.path.dirname(os.path.abspath(__file__))
RUN = "r02"
CKPT = "best.pth"
DATA = "data/seg7"
SEED = 7
WARMUP = 10                # hold-out frames run before timing starts
NN_CHUNK = 8192
ROPE = (1, 2, 5)           # rope classes, merged for the r01 comparison
RENDER = "true"


def predict(model, xyz, rgb, center, n_points, seed, amp):
    """(class on all points, class on sampled points, sample idx, model ms, end-to-end ms)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(xyz), n_points, replace=len(xyz) < n_points)
    torch.manual_seed(seed)                     # random centroids in pn2_dlo are drawn with torch.rand
    x_np = dlo_data.to_input(xyz, rgb, center)
    torch.cuda.synchronize(); t0 = time.time()
    X = torch.from_numpy(x_np).cuda()
    xs = X[:, torch.from_numpy(idx).cuda()].unsqueeze(0)
    torch.cuda.synchronize(); t1 = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        pred_s = model(xs)[0].argmax(-1)
    torch.cuda.synchronize(); t2 = time.time()
    a, s = X[:3].T.contiguous(), xs[0, :3].T.contiguous()
    nn_idx = torch.cat([torch.cdist(a[i:i + NN_CHUNK], s).argmin(1) for i in range(0, a.shape[0], NN_CHUNK)])
    pred = pred_s[nn_idx].cpu().numpy().astype(np.uint8)
    torch.cuda.synchronize(); t3 = time.time()
    return pred, pred_s.cpu().numpy().astype(np.uint8), idx, 1e3 * (t2 - t1), 1e3 * (t3 - t0)


def conf_np(lab, pred, k):
    return np.bincount(lab.astype(np.int64) * k + pred.astype(np.int64), minlength=k * k).reshape(k, k)


def metrics(c):
    c = np.asarray(c, dtype=float)
    tp = np.diag(c)
    iou = tp / np.maximum(1, c.sum(0) + c.sum(1) - tp)
    prec, rec = tp / np.maximum(1, c.sum(0)), tp / np.maximum(1, c.sum(1))
    r = np.zeros(len(c), bool)
    r[list(ROPE)] = True
    rtp, rfp, rfn = c[np.ix_(r, r)].sum(), c[np.ix_(~r, r)].sum(), c[np.ix_(r, ~r)].sum()
    present = (c.sum(0) + c.sum(1)) > 0
    return {"iou": iou.tolist(), "prec": prec.tolist(), "rec": rec.tolist(), "miou": float(iou.mean()),
            "miou_present": float(iou[present].mean()) if present.any() else float("nan"),
            "rope_iou": float(rtp / max(1, rtp + rfp + rfn)), "rope_prec": float(rtp / max(1, rtp + rfp)),
            "rope_rec": float(rtp / max(1, rtp + rfn)), "points": int(c.sum())}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default=RUN)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--render", default=RENDER)
    ap.add_argument("--max_frames", type=int, default=0, help="0 = every hold-out clean frame; >0 = strided subset (smoke)")
    ap.add_argument("--tag", default="", help="suffix for output names; non-empty = smoke run, nothing published")
    args = ap.parse_args()
    ck = torch.load(os.path.join(HERE, "checkpoints", args.run, args.ckpt), map_location="cuda", weights_only=False)
    cfg = ck["cfg"]
    K = cfg["classes"]
    pn2_dlo.SAMPLING["random_min"] = cfg["random_min"]
    model = pn2_dlo.get_model(num_classes=K).cuda()
    model.load_state_dict(ck["model"])
    model.eval()
    torch.backends.cudnn.benchmark = True
    data_dir = os.path.join(HERE, args.data)
    meta = dlo_data.load_meta(data_dir)
    names = meta["classes"]
    center, box = dlo_data.box_center(meta["box"]), meta["box"]
    log_dir = os.path.join(HERE, "logs", args.run)
    publish = args.render == "true" and not args.tag

    rows = dlo_data.read_index(data_dir, split="holdout", kind="clean")
    if args.max_frames:
        rows = rows[::max(1, len(rows) // args.max_frames)]
    flagged = dlo_data.read_index(data_dir, split="holdout", kind="flagged")
    groups, frames, tm, tt = {}, [], [], []
    c_sampled = np.zeros((K, K), dtype=np.int64)
    for i, r in enumerate(rows + flagged):
        xyz, rgb, lab = dlo_data.load_frame(data_dir, r)
        pred, pred_s, idx, ms_model, ms_total = predict(model, xyz, rgb, center, cfg["n_points"], SEED + i, True)
        c = conf_np(lab, pred, K)
        m = metrics(c)
        f = dict(i=i, batch=r["batch"], ep=r["ep"], frame=r["frame"], kind=r["kind"], status=r["status"], tags=r["tags"],
                 n_box=len(xyz), miou=m["miou_present"], rope_iou=m["rope_iou"], model_ms=ms_model, total_ms=ms_total,
                 **{"iou_c%d" % k: m["iou"][k] for k in range(K)}, **{"n_lab_c%d" % k: int((lab == k).sum()) for k in range(K)})
        frames.append(f)
        if r["kind"] == "clean":
            if i >= WARMUP:
                tm.append(ms_model); tt.append(ms_total)
            c_sampled += conf_np(lab[idx], pred_s, K)
            for g in ("all", "obj:" + r["batch"][:4], "speed:" + r["batch"][5:], "batch:" + r["batch"]):
                groups[g] = groups.get(g, np.zeros((K, K), dtype=np.int64)) + c
        if (i + 1) % 500 == 0:
            print("  %d/%d frames, running mIoU %.4f" % (i + 1, len(rows) + len(flagged), metrics(groups["all"])["miou"]),
                  flush=True)

    clean = [f for f in frames if f["kind"] == "clean"]
    fm = np.array([f["miou"] for f in clean])
    ca = groups["all"]
    out = {"run": args.run, "ckpt": args.ckpt, "epoch": ck["epoch"], "val_miou_ckpt": ck.get("val_iou"),
           "val_iou_per_class_ckpt": ck.get("val_iou_per_class"), "classes": names, "n_points": cfg["n_points"],
           "frames": len(clean), "episodes": len(set(f["ep"] for f in clean)),
           "pooled": {g: metrics(c) for g, c in sorted(groups.items())},
           "confusion_all": ca.tolist(),
           "confusion_all_row_pct": (100.0 * ca / np.maximum(1, ca.sum(1, keepdims=True))).round(2).tolist(),
           "sampled_points_only": metrics(c_sampled),
           "frame_miou_present": {"mean": float(fm.mean()), "median": float(np.median(fm)), "p05": float(np.percentile(fm, 5)),
                                  "min": float(fm.min())},
           "timing_ms": {"model_median": float(np.median(tm)), "model_p90": float(np.percentile(tm, 90)),
                         "total_median": float(np.median(tt)), "total_p90": float(np.percentile(tt, 90)),
                         "n_box_median": float(np.median([f["n_box"] for f in clean])), "gpu": torch.cuda.get_device_name()},
           "flagged": [{k: f[k] for k in ("ep", "frame", "status", "tags", "miou", "rope_iou")} for f in frames
                       if f["kind"] == "flagged"]}
    with open(os.path.join(log_dir, "eval_holdout%s.json" % args.tag), "w") as fh:
        json.dump(out, fh, indent=1)
    with open(os.path.join(log_dir, "eval_frames%s.csv" % args.tag), "w") as fh:
        w = csv.DictWriter(fh, fieldnames=list(frames[0].keys()))
        w.writeheader()
        w.writerows(frames)

    if args.render == "true":
        pubs = []
        by_i = {f["i"]: r for f, r in zip(frames, rows + flagged)}
        for dlo in ("d001", "d002", "d003", "d004"):
            ff = sorted((f for f in clean if f["batch"].startswith(dlo)), key=lambda f: f["miou"])
            if not ff:
                continue
            for which, f in (("median", ff[len(ff) // 2]), ("lowest", ff[0])):
                r = by_i[f["i"]]
                xyz, rgb, lab = dlo_data.load_frame(data_dir, r)
                pred = predict(model, xyz, rgb, center, cfg["n_points"], SEED + f["i"], True)[0]
                what = "holdout_%s_%s_miou_%s_f%06d" % (dlo, which, f["ep"], f["frame"])
                src = S.render_pair(xyz.astype(np.float64), lab, pred, "%s frame %d | frame mIoU %.3f, rope IoU %.3f" % (
                    f["ep"], f["frame"], f["miou"], f["rope_iou"]), os.path.join(V.FIG_DIR, "%s_%s%s.png" % (
                        args.run, what, args.tag)), box)
                if publish:
                    body = ("Hold-out frame of %s with the %s frame mIoU of that object's %d hold-out clean frames (run %s). "
                            "Upper row: training-style label (geometry rules of the r02 label build, from the v034 ground "
                            "truth); lower row: model prediction. Per-class IoU in this frame: %s." % (
                                dlo, which, len(ff), args.run, ", ".join("%s %.2f" % (names[k], f["iou_c%d" % k])
                                                                          for k in range(K) if f["n_lab_c%d" % k])))
                    pubs.append(S.publish(src, args.run, 6, what, "v036 PointNet++ %s hold-out, %s frame of %s: %s "
                                          "frame %d, label vs prediction" % (args.run, which, dlo, f["ep"], f["frame"]), body))
        for f in (f for f in frames if f["kind"] == "flagged"):
            r = by_i[f["i"]]
            xyz, rgb, lab = dlo_data.load_frame(data_dir, r)
            pred = predict(model, xyz, rgb, center, cfg["n_points"], SEED + f["i"], True)[0]
            what = "flagged_%s_f%06d" % (f["ep"], f["frame"])
            src = S.render_pair(xyz.astype(np.float64), lab, pred, "%s frame %d | %s %s | mIoU vs label %.3f" % (
                f["ep"], f["frame"], f["status"], f["tags"], f["miou"]), os.path.join(V.FIG_DIR, "%s_%s%s.png" % (
                    args.run, what, args.tag)), box, label_note="label (v034-derived)")
            if publish:
                body = ("Frame the v034 pipeline flagged (%s, tags: %s), from a hold-out episode (run %s). Upper row: "
                        "labels from the same geometry rules, which rely on v034's node path and can be wrong on flagged "
                        "frames; lower row: model prediction. mIoU against those labels %.3f, rope (classes 1+2+5) IoU "
                        "%.3f." % (f["status"], f["tags"].replace("|", ", ") or "none", args.run, f["miou"], f["rope_iou"]))
                pubs.append(S.publish(src, args.run, 7, what, "v036 PointNet++ %s on a v034-flagged frame: %s frame %d "
                                      "(%s), label vs prediction" % (args.run, f["ep"], f["frame"], f["status"]), body))
        if publish:
            pubs += [S.chart_metrics(out, args.run), S.chart_confusion(out, args.run), S.chart_curves(args.run)]
        print("\n".join(pubs))
    al = out["pooled"]["all"]
    print("mIoU %.4f | rope IoU %.4f P %.4f R %.4f | timing %s" % (al["miou"], al["rope_iou"], al["rope_prec"],
                                                                     al["rope_rec"], out["timing_ms"]))
    for g, m in out["pooled"].items():
        if not g.startswith("batch:"):
            print("  %-12s mIoU %.4f  IoU %s" % (g, m["miou"], np.round(m["iou"], 3).tolist()))


if __name__ == "__main__":
    main()
