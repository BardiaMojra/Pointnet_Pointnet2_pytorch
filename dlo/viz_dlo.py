#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
viz_dlo.py -- figures for dlo/: point-cloud renders, training curves, hold-out metrics.

Renders: top (x-y) and side (y-z) views of one frame's workspace points, robot frame; background
grey by luminance, overlays on top: label magenta; v034 crop orange; prediction vs label = TP green,
FP red, FN blue. Charts: reference dataviz palette slots 1-3 (blue, orange, aqua), neutral grey for
context series. PNG at 300 dpi, opaque points.
  python3 dlo/viz_dlo.py labels       # one fit clean frame per object -> figs/labels_*.png
  python3 dlo/viz_dlo.py curves r01   # logs/r01/train_log.csv -> figs/train_r01.png
  python3 dlo/viz_dlo.py metrics r01  # logs/r01/eval_holdout.json -> figs/metrics_r01.png
  python3 dlo/viz_dlo.py labels7      # 7-class labels, one frame per object -> dev_figs/v036-pn2-r02_01_*.png
  python3 dlo/viz_dlo.py shares7      # 7-class point shares by object -> dev_figs/v036-pn2-r02_02_*.png
Published figures (dev_figs) carry a title band and a caption band that spells out every abbreviation.
Platforms: u24_a64 (dlo_torch container on the dgx).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import platform_guard                                                        # noqa: E402
platform_guard.require(("u24_a64",), __file__)

import glob                                                                  # noqa: E402
import hashlib                                                               # noqa: E402
import json                                                                  # noqa: E402

import matplotlib                                                            # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                              # noqa: E402
import numpy as np                                                           # noqa: E402

# --- Defaults ---
HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figs")
GT_DIR = os.path.expanduser("~/bags/test_session_001/groundtruth")
GT_VER = "v034"
DPI = 300
COL = {"label": "#c51b7d", "crop": "#eb6834", "tp": "#1a9850", "fp": "#d73027", "fn": "#2c7bb6"}
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")          # dataviz reference categorical slots 1-3 (light)
INK, INK2, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e4e3de", "#fcfcfb"
VIEWS = (("top (x-y)", 0, 1), ("side (y-z)", 1, 2))
OBJ_NAME = {"d001": "d001 black rope", "d002": "d002 white cord", "d003": "d003 tan rope", "d004": "d004 cat5e x3"}


def _grey(rgb):
    lum = (0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]) / 255.0
    g = 0.55 + 0.35 * lum
    return np.stack([g, g, g], axis=1)


def overlay_sets(label=None, pred=None):
    """[(name, mask, colour)] drawn over the grey background."""
    if pred is None:
        return [("label (DLO)", label.astype(bool), COL["label"])]
    lab = np.zeros_like(pred, dtype=bool) if label is None else label.astype(bool)
    pr = pred.astype(bool)
    return [("TP", pr & lab, COL["tp"]), ("FP", pr & ~lab, COL["fp"]), ("FN", ~pr & lab, COL["fn"])]


def read_pcd_xyz(path):
    """xyz of an uncompressed binary/ascii PCD with float x y z (+ rgb) fields, as the pipeline writes crops."""
    with open(path, "rb") as f:
        hdr = {}
        while True:
            line = f.readline().decode("ascii", "replace").strip()
            k, _, v = line.partition(" ")
            hdr[k] = v.split()
            if k == "DATA":
                break
        fields, n = hdr["FIELDS"], int(hdr["POINTS"][0])
        if hdr["DATA"][0] == "binary":
            data = np.frombuffer(f.read(4 * len(fields) * n), dtype=np.float32).reshape(n, len(fields))
        elif hdr["DATA"][0] == "ascii":
            data = np.loadtxt(f, dtype=np.float64).reshape(n, len(fields))
        else:
            raise ValueError("%s: DATA %s not supported" % (path, hdr["DATA"][0]))
    return data[:, [fields.index(c) for c in "xyz"]].astype(np.float64)


def v034_crop_mask(xyz, batch, ep, frame, tol_m=1e-4):
    """In-box points that are the v034 dlo_raw_crop_pcd points of this frame (exact subset, NN < tol), or None."""
    from scipy.spatial import cKDTree
    hits = sorted(glob.glob(os.path.join(GT_DIR, batch, ep, "_out_" + GT_VER, "dlo_raw_crop_pcd", "%06d_*.pcd" % frame)))
    if not hits:
        return None
    d, _ = cKDTree(read_pcd_xyz(hits[0])).query(xyz, distance_upper_bound=tol_m)
    return d <= tol_m


def render(xyz, rgb, rows, title, out_png, box=None):
    """rows: [(row title, [(name, mask, colour), ...])]; one figure row per entry, two views each."""
    fig, axes = plt.subplots(len(rows), 2, figsize=(9.0, 3.9 * len(rows)), squeeze=False,
                             gridspec_kw={"width_ratios": [1.0, 1.25]})
    grey = _grey(rgb)
    for ri, (rtitle, sets) in enumerate(rows):
        for ci, (vname, a, b) in enumerate(VIEWS):
            ax = axes[ri, ci]
            ax.scatter(xyz[:, a], xyz[:, b], s=0.15, c=grey, marker=".", linewidths=0, rasterized=True)
            for name, mask, colour in sets:
                if mask.any():
                    ax.scatter(xyz[mask, a], xyz[mask, b], s=0.8, c=colour, marker=".", linewidths=0,
                               rasterized=True, label="%s (%d)" % (name, int(mask.sum())))
            ax.set_aspect("equal")
            if box is not None:
                ax.set_xlim(box[2 * a], box[2 * a + 1]); ax.set_ylim(box[2 * b], box[2 * b + 1])
            ax.set_xlabel("xyz"[a] + " (m)"); ax.set_ylabel("xyz"[b] + " (m)")
            ax.set_title("%s | %s" % (rtitle, vname), fontsize=9)
            if ci == 1:
                ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7, markerscale=8, frameon=False)
    import textwrap
    fig.suptitle("\n".join(textwrap.wrap(title, 95, break_on_hyphens=False)), fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)
    return out_png


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=7)


def main_labels():
    import dlo_data
    meta = dlo_data.load_meta()
    rows = dlo_data.read_index(split="fit")
    outs = []
    for dlo, batch in (("d001", "d001_v050"), ("d002", "d002_v070"), ("d003", "d003_v100"), ("d004", "d004_v070")):
        eps = sorted(set(r["ep"] for r in rows if r["batch"] == batch),
                     key=lambda e: hashlib.sha1(e.encode()).hexdigest())
        rr = [r for r in rows if r["ep"] == eps[0]]
        r = rr[len(rr) // 2]
        xyz, rgb, lab = dlo_data.load_frame(dlo_data.DATA_DIR, r)
        title = "%s frame %d | %d box pts, %d DLO (%.1f%%), eps %.1f mm" % (
            r["ep"], r["frame"], len(xyz), lab.sum(), 100.0 * lab.mean(), 1e3 * meta["per_object"][dlo]["label_eps_m"])
        outs.append(render(xyz, rgb, [("label", overlay_sets(label=lab))], title,
                           os.path.join(FIG_DIR, "labels_%s_%s_f%06d.png" % (dlo, r["ep"], r["frame"])), meta["box"]))
    print("\n".join(outs))


def main_curves(run):
    """logs/<run>/train_log.csv -> figs/train_<run>.png: loss and DLO IoU, train (running) vs val (per epoch)."""
    import csv
    with open(os.path.join(HERE, "logs", run, "train_log.csv")) as f:
        rows = list(csv.DictReader(f))
    tr = [r for r in rows if r["kind"] == "train"]
    va = [r for r in rows if r["kind"] == "val"]
    steps_ep = max(int(r["gstep"]) for r in va) / max(1, len(va))
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    for ax, key, name in ((axes[0], "loss", "weighted NLL loss"), (axes[1], "iou", "DLO IoU (argmax, sampled points)")):
        _style(ax)
        ax.plot([int(r["gstep"]) / steps_ep for r in tr], [float(r[key]) for r in tr], lw=1.0, color="#a3a29b",
                label="train, running mean per 100 steps")
        ax.plot([int(r["gstep"]) / steps_ep for r in va], [float(r[key]) for r in va], "o-", ms=5, lw=2.0,
                color=SERIES[0], mec=SURFACE, mew=1.0, label="val, 320 frames of 32 fit-split episodes")
        ax.set_xlabel("epochs completed", color=INK2, fontsize=8); ax.set_title(name, fontsize=9, color=INK)
    axes[0].set_yscale("log")
    ticks = [t for t in (0.01, 0.015, 0.02, 0.03, 0.05, 0.1, 0.2, 0.4) if t < 1.5 * max(float(r["loss"]) for r in tr)]
    axes[0].set_yticks(ticks)
    axes[0].set_yticklabels(["%g" % t for t in ticks])
    axes[0].minorticks_off()
    axes[1].set_ylim(0.7, 1.0)
    best = max(va, key=lambda r: float(r["iou"]))
    axes[1].annotate("best %.3f (after epoch %d)" % (float(best["iou"]), int(best["epoch"]) + 1),
                     (int(best["gstep"]) / steps_ep, float(best["iou"])), xytext=(0, 10), textcoords="offset points",
                     ha="center", fontsize=7, color=INK)
    axes[1].legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "train_%s.png" % run)
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(out, dpi=DPI, facecolor="white")
    print(out)


def main_metrics(run, tag=""):
    """logs/<run>/eval_holdout.json -> figs/metrics_<run>.png: dot plot of hold-out IoU/precision/recall
    (calibrated threshold) per object and speed, argmax IoU for reference, and val IoU vs threshold."""
    with open(os.path.join(HERE, "logs", run, "eval_holdout%s.json" % tag)) as f:
        ev = json.load(f)
    cal, arg = ev["pooled_cal"], ev["pooled_argmax"]
    groups = [("all", "all hold-out")] + [("obj:" + k, v) for k, v in OBJ_NAME.items()] + \
             [("speed:" + s, "speed " + s) for s in ("v050", "v070", "v100")]
    groups = [(g, n) for g, n in groups if g in cal]
    y = np.arange(len(groups))[::-1].astype(float)
    y[1:] -= 0.5                                                    # gap after "all"
    y[1 + 4:] -= 0.5                                                # gap between objects and speeds
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9), gridspec_kw={"width_ratios": [1.6, 1.0]})
    ax = axes[0]
    _style(ax)
    ax.grid(axis="y", visible=False)
    series = (("iou", "IoU", SERIES[0], "o"), ("prec", "precision", SERIES[1], "s"), ("rec", "recall", SERIES[2], "^"))
    for key, name, colour, mk in series:
        ax.plot([cal[g][key] for g, _ in groups], y, mk, ms=7, color=colour, mec=SURFACE, mew=1.0, ls="none",
                label="%s (P >= %.2f)" % (name, ev["threshold"]["cal"]), zorder=3)
    ax.plot([arg[g]["iou"] for g, _ in groups], y, "o", ms=7, mfc="none", mec="#8c8b84", mew=1.2, ls="none",
            label="IoU at argmax (P >= 0.50)", zorder=2)
    for (g, _), yy in zip(groups, y):
        ax.annotate("%.3f" % cal[g]["iou"], (cal[g]["iou"], yy), xytext=(-6, 6), textcoords="offset points",
                    ha="right", fontsize=6.5, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([n for _, n in groups], fontsize=7.5, color=INK)
    lo = min(min(cal[g][k] for g, _ in groups for k in ("iou", "prec", "rec")), min(arg[g]["iou"] for g, _ in groups))
    ax.set_xlim(np.floor((lo - 0.02) * 20) / 20, 1.005)
    ax.set_xlabel("pooled over all in-box points of %d hold-out clean frames (%d episodes)" % (ev["frames"], ev["episodes"]),
                  fontsize=7.5, color=INK2)
    ax.set_title("hold-out DLO segmentation (axis starts at %.2f)" % ax.get_xlim()[0], fontsize=9, color=INK)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.16), ncol=2)
    ax2 = axes[1]
    _style(ax2)
    th = ev["threshold"]
    ax2.plot(th["grid"], th["val_iou"], "o-", ms=5, lw=2.0, color=SERIES[0], mec=SURFACE, mew=1.0)
    k = th["grid"].index(th["cal"])
    ax2.annotate("chosen %.2f\nval IoU %.3f" % (th["cal"], th["val_iou"][k]), (th["cal"], th["val_iou"][k]),
                 xytext=(0, -140), textcoords="offset points", ha="center", fontsize=7, color=INK,
                 arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    ax2.set_xlabel("P(DLO) threshold", fontsize=7.5, color=INK2)
    ax2.set_title("threshold pick on %d val frames (fit split)" % th["val_frames"], fontsize=9, color=INK)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "metrics_%s%s.png" % (run, tag))
    fig.savefig(out, dpi=DPI, facecolor="white")
    print(out)


# ---- 7-class scene segmentation (run r02) ----
DEV_FIGS = os.path.expanduser("~/git/dlo_data_001/dev/dev_figs")
CLASS7 = ("other / background", "rope body", "rope gripper end", "gripper", "pole and mount", "rope pole end",
          "table / red cloth")
# other = light neutral; classes 1-6 = dataviz reference categorical slots 1-6, in order
CLASS7_COL = ("#c9c8c2", "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
CLASS7_DRAW = (0, 6, 4, 3, 1, 2, 5)          # small rope-end classes drawn last (on top)
TERMS7 = ("Terms: d001-d004 = the four ropes (d001 black rope, d002 white cord, d003 tan rope, d004 bundle of three "
          "cat5e cables); v050/v070/v100 = gripper speed setting; tNNN = test (episode) number; fNNN = frame number; "
          "v034 = ground-truth pipeline version the labels come from; node path = the ordered rope centreline nodes "
          "v034 saved per frame (pole end first); eps = rope radius + 3 mm; FK = forward kinematics (gripper pose from "
          "the robot's joint angles); robot frame = the robot base coordinate system (m); top view = x-y, "
          "side view = y-z.")


def band(src_png, out_png, title, body):
    """Title band + image (1:1 pixels) + caption band; same layout as dlo_data_001/dev/dev_figs/make_dev_figs.py
    _band() (u18 only, so mirrored here)."""
    import textwrap
    img = plt.imread(src_png)
    h, w = img.shape[:2]
    dpi = 100.0
    t_px, b_px = float(np.clip(w / 55.0, 24, 84)), float(np.clip(w / 95.0, 17, 52))
    pad = 0.6 * b_px
    chars = int((w - 2 * pad) / (0.56 * b_px))              # 0.56: room for upper-case tag strings
    t_lines = textwrap.wrap(title, int((w - 2 * pad) / (0.58 * t_px)))
    b_lines = []
    for para in body.split("\n"):
        b_lines += textwrap.wrap(para, chars) or [""]
    head, foot = len(t_lines) * t_px * 1.35 + 2 * pad, len(b_lines) * b_px * 1.42 + 2 * pad
    H = h + head + foot
    fig = plt.figure(figsize=(w / dpi, H / dpi), dpi=dpi, facecolor="white")
    ax = fig.add_axes([0, foot / H, 1, h / H])
    ax.imshow(img, interpolation="none")
    ax.set_axis_off()
    fig.text(pad / w, 1 - pad / H, "\n".join(t_lines), va="top", ha="left", fontsize=t_px * 72 / dpi,
             fontweight="bold", color=INK, linespacing=1.2)
    fig.text(pad / w, (foot - pad) / H, "\n".join(b_lines), va="top", ha="left", fontsize=b_px * 72 / dpi,
             color=INK, linespacing=1.3)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=dpi, facecolor="white")
    plt.close(fig)
    return out_png


def render7(xyz, rows, title, out_png, box=None):
    """rows: [(row title, per-point class array 0-6)]; top and side view per row, points coloured by class."""
    import textwrap
    fig, axes = plt.subplots(len(rows), 2, figsize=(9.5, 3.9 * len(rows)), squeeze=False,
                             gridspec_kw={"width_ratios": [1.0, 1.25]})
    for ri, (rtitle, lab) in enumerate(rows):
        for ci, (vname, a, b) in enumerate(VIEWS):
            ax = axes[ri, ci]
            for c in CLASS7_DRAW:
                m = lab == c
                if m.any():
                    ax.scatter(xyz[m, a], xyz[m, b], s=0.15 if c == 0 else 0.7, c=CLASS7_COL[c], marker=".",
                               linewidths=0, rasterized=True, label="%d %s (%d pts)" % (c, CLASS7[c], int(m.sum())))
            ax.set_aspect("equal")
            if box is not None:
                ax.set_xlim(box[2 * a], box[2 * a + 1]); ax.set_ylim(box[2 * b], box[2 * b + 1])
            ax.set_xlabel("xyz"[a] + " (m)"); ax.set_ylabel("xyz"[b] + " (m)")
            ax.set_title("%s | %s" % (rtitle, vname), fontsize=9)
            if ci == 1:
                hs, ls = ax.get_legend_handles_labels()
                order = sorted(range(len(ls)), key=lambda i: int(ls[i].split()[0]))
                ax.legend([hs[i] for i in order], [ls[i] for i in order], loc="upper left", bbox_to_anchor=(1.01, 1.0),
                          fontsize=7, markerscale=8, frameon=False)
    fig.suptitle("\n".join(textwrap.wrap(title, 95)), fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)
    return out_png


def _seg7():
    import csv
    d = os.path.join(HERE, "data", "seg7")
    with open(os.path.join(d, "meta.json")) as f:
        meta = json.load(f)
    with open(os.path.join(d, "index.csv")) as f:
        rows = list(csv.DictReader(f))
    return d, meta, rows


def main_labels7(tag="r02"):
    """One clean frame per object (first episode by sha1 among those with frames, middle frame) -> dlo/figs and
    dev_figs/v036-pn2-<tag>_01_labels_<obj>_<ep>_f<frame>.png."""
    d, meta, rows = _seg7()
    g = meta["crop_guard"]
    for dlo in ("d001", "d002", "d003", "d004"):
        rr = [r for r in rows if r["kind"] == "clean" and r["batch"].startswith(dlo)]
        ep = sorted(set(r["ep"] for r in rr), key=lambda e: hashlib.sha1(e.encode()).hexdigest())[0]
        re = [r for r in rr if r["ep"] == ep]
        r = re[len(re) // 2]
        z = np.load(os.path.join(d, r["npz"]))
        lab, xyz = z["label"], z["xyz"].astype(np.float64)
        src = render7(xyz, [("7-class label", lab)], "%s frame %s | %d in-box points" % (ep, r["frame"], len(xyz)),
                      os.path.join(FIG_DIR, "labels7_%s_%s_f%06d.png" % (dlo, ep, int(r["frame"]))), meta["box"])
        cnt = ", ".join("%s %.1f%%" % (CLASS7[c], 100.0 * np.mean(lab == c)) for c in range(7))
        body = ("Training labels for the 7-class PointNet++ scene segmentation (run r02), one clean frame of %s: "
                "all points of the fixed workspace box in the robot frame, coloured by class. Shares in this frame: %s.\n"
                "Rules, geometry first: rope = within eps of the v034 node path, ends cut; rope pole end / rope gripper "
                "end = the first / last %.0f cm of that path (arc length); gripper = non-rope points in the FK gripper "
                "capsule, or red points near the FK marker (the red finger); pole and mount = non-rope points in the "
                "fitted pole-shaft capsule or its axis extended down to the table plane (161 d001 episodes without "
                "their own fit use the batch's median shaft placed at their own pole marker), or in static pole-top "
                "voxels inside the mount box around the pole marker; "
                "table / red cloth = red points in the pole-bottom box, or points within %.1f cm of the episode's table "
                "plane; other = everything else. Colour enters only through the pipeline's own red tests. Precedence: "
                "rope > gripper > table > pole > other. Frame chosen after the crop guard (mode '%s': a frame whose v034 "
                "crop has more than %.0f%% of its points over %.0f cm from the node path is replaced).\n%s" % (
                    dlo, cnt, 100 * meta["end_m"], 100 * meta["table_band_m"], g["mode"], 100 * g["far_frac"],
                    100 * g["far_m"], TERMS7))
        print(band(src, os.path.join(DEV_FIGS, "v036-pn2-%s_01_labels_%s_%s_f%06d.png" % (tag, dlo, ep, int(r["frame"]))),
                   "v036 PointNet++ r02 training labels, 7 classes: %s, %s frame %s (top and side view)" % (
                       dlo, ep, r["frame"]), body))


def main_shares7(tag="r02"):
    """Per-class point shares by object (dot plot, log axis) -> dev_figs/v036-pn2-<tag>_02_class_point_shares_by_object.png."""
    d, meta, rows = _seg7()
    objs = (("all", "all clean frames"), ("d001", "d001 black rope"), ("d002", "d002 white cord"),
            ("d003", "d003 tan rope"), ("d004", "d004 cat5e x3"))
    sh = {"all": meta["class_share_all_clean"]}
    sh.update(meta["class_share_by_object"])
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    _style(ax)
    y = np.arange(7)[::-1]
    for (k, name), mk, colour in zip(objs, ("o", "s", "^", "D", "v"), (INK,) + SERIES + ("#e87ba4",)):
        vals = [100.0 * sh[k][c] for c in meta["classes"]]
        if k == "all":
            ax.plot(vals, y, mk, ms=9, mfc="none", mec=INK, mew=1.4, ls="none", label=name, zorder=3)
        else:
            ax.plot(vals, y, mk, ms=6, color=colour, mec=SURFACE, mew=1.0, ls="none", label=name, zorder=2)
    for c, yy in enumerate(y):
        v = 100 * sh["all"][meta["classes"][c]]
        ax.annotate("%.2f%%" % v, (v, yy), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=6.5, color=INK)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(["%d %s" % (c, CLASS7[c]) for c in range(7)], fontsize=7.5, color=INK)
    ax.set_xlabel("share of in-box points (%, log axis)", fontsize=8, color=INK2)
    ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    src = os.path.join(FIG_DIR, "class_shares7.png")
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(src, dpi=DPI, facecolor="white")
    plt.close(fig)
    g = meta["crop_guard"]
    body = ("Share of the points of each class, pooled over the clean frames of the r02 label build (%d fit + %d hold-out "
            "frames from %d + %d episodes): all objects (open black circle, value printed) and per object. Training "
            "class weights = sqrt(largest share / class share): %s. Crop guard mode '%s': %d of %d checked frames failed "
            "and were replaced by their nearest passing neighbour.\n%s" % (
                meta["frames"]["fit"], meta["frames"]["holdout"], meta["episodes"]["fit"], meta["episodes"]["holdout"],
                ", ".join("%s %.2f" % (CLASS7[c], w) for c, w in enumerate(meta["class_weight_sqrt_inv_freq_fit"])),
                g["mode"], g["failed"], g["checked"], TERMS7))
    print(band(src, os.path.join(DEV_FIGS, "v036-pn2-%s_02_class_point_shares_by_object.png" % tag),
               "v036 PointNet++ r02: point share of each of the 7 classes, by object", body))


if __name__ == "__main__":
    cmd = sys.argv[1:2]
    if cmd == ["labels"]:
        main_labels()
    elif cmd == ["curves"] and len(sys.argv) > 2:
        main_curves(sys.argv[2])
    elif cmd == ["metrics"] and len(sys.argv) > 2:
        main_metrics(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "")
    elif cmd == ["labels7"]:
        main_labels7()
    elif cmd == ["shares7"]:
        main_shares7()
    else:
        sys.exit(__doc__)
