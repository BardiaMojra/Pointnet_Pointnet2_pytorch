#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
viz_seg7.py -- figures for the multi-class runs (r02: 7 classes): training curves, per-class hold-out
metrics, confusion matrix, label/prediction renders; each saved to dlo/figs/ and, with a title band
and a caption band that spells out every abbreviation, to dlo_data_001/dev/dev_figs/ as
v036-pn2-<run>_NN_<what>.png (viz_dlo.band). Colours: classes as viz_dlo.CLASS7_COL; metric series
= dataviz reference slots 1-3; confusion = one-hue blue ramp (dataviz reference sequential steps).
  python3 dlo/viz_seg7.py curves r02   # logs/r02/train_log.csv -> ..._03_training_curves.png
Platforms: u24_a64 (dlo_torch container on the dgx).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import viz_dlo as V                                                          # noqa: E402  platform guard, Agg, palette, band()

import csv                                                                   # noqa: E402
import matplotlib.pyplot as plt                                              # noqa: E402
import numpy as np                                                           # noqa: E402
from matplotlib.colors import LinearSegmentedColormap                        # noqa: E402

# --- Defaults ---
SEQ_BLUE = ("#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
TERMS_METRIC = ("IoU = intersection over union (true positives / (true + false positives + false negatives)); "
                "P = precision; R = recall; mIoU = mean IoU over the 7 classes; argmax = class with the highest "
                "predicted probability; 1-NN = nearest sampled point; hold-out = episodes never used in training; "
                "val = validation frames from training-split episodes held back from training.")


def fig_name(run, nn, what):
    return os.path.join(V.DEV_FIGS, "v036-pn2-%s_%02d_%s.png" % (run, nn, what))


def publish(src, run, nn, what, title, body):
    return V.band(src, fig_name(run, nn, what), title, body + "\n" + V.TERMS7 + " " + TERMS_METRIC)


def chart_curves(run):
    """Loss, mIoU (train running, val) and per-class val IoU per epoch."""
    with open(os.path.join(V.HERE, "logs", run, "train_log.csv")) as f:
        rows = list(csv.DictReader(f))
    tr, va = [r for r in rows if r["kind"] == "train"], [r for r in rows if r["kind"] == "val"]
    k = sum(1 for c in rows[0] if c.startswith("iou_c"))
    spe = max(int(r["gstep"]) for r in va) / max(1, len(va))
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.6))
    ax = axes[0]
    V._style(ax)
    ax.plot([int(r["gstep"]) / spe for r in tr], [float(r["loss"]) for r in tr], lw=1.0, color="#a3a29b", label="train")
    ax.plot([int(r["gstep"]) / spe for r in va], [float(r["loss"]) for r in va], "o-", ms=5, lw=2.0, color=V.SERIES[0],
            mec=V.SURFACE, label="val")
    lv = [float(r["loss"]) for r in tr + va]
    if max(lv) / max(1e-9, min(lv)) > 10:
        from matplotlib.ticker import FuncFormatter, LogLocator
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "%g" % v))
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.set_title("weighted NLL loss", fontsize=9, color=V.INK)
    ax = axes[1]
    V._style(ax)
    ax.plot([int(r["gstep"]) / spe for r in tr], [float(r["iou"]) for r in tr], lw=1.0, color="#a3a29b",
            label="train, running mean per 100 steps")
    ax.plot([int(r["gstep"]) / spe for r in va], [float(r["iou"]) for r in va], "o-", ms=5, lw=2.0, color=V.SERIES[0],
            mec=V.SURFACE, label="val")
    b = max(va, key=lambda r: float(r["iou"]))
    ax.annotate("best %.3f (after epoch %d)" % (float(b["iou"]), int(b["epoch"]) + 1), (int(b["gstep"]) / spe, float(b["iou"])),
                xytext=(0, -16), textcoords="offset points", ha="center", fontsize=7, color=V.INK)
    ax.set_title("mIoU (argmax, sampled points)", fontsize=9, color=V.INK)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax = axes[2]
    V._style(ax)
    for c in range(k):
        ax.plot([int(r["gstep"]) / spe for r in va], [float(r["iou_c%d" % c]) for r in va], "o-", ms=3.5, lw=1.6,
                color=V.CLASS7_COL[c] if c else "#8c8b84", label="%d %s" % (c, V.CLASS7[c]))
    ax.set_title("val IoU per class", fontsize=9, color=V.INK)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    for ax in axes:
        ax.set_xlabel("epochs completed", fontsize=8, color=V.INK2)
    fig.tight_layout()
    src = os.path.join(V.FIG_DIR, "train_%s.png" % run)
    fig.savefig(src, dpi=V.DPI, facecolor="white")
    plt.close(fig)
    import json
    with open(os.path.join(V.HERE, "logs", run, "config.json")) as f:
        cfg = json.load(f)
    body = ("Training of the 7-class PointNet++ scene segmentation (run %s) on the GB10 GPU: weighted NLL (negative "
            "log-likelihood) loss, mIoU and per-class IoU. Training: %d frames from %d training-split episodes, the "
            "train curves are running means over 100 steps of %d frames x %d points; val = %d frames from %d other "
            "training-split episodes held back from training, evaluated after each epoch; best.pth is the epoch with "
            "the highest val mIoU." % (run, cfg["n_train"], cfg["train_eps"], cfg["batch"], cfg["n_points"],
                                       cfg["n_val"], cfg["val_eps"]))
    return publish(src, run, 3, "training_curves", "v036 PointNet++ %s: training curves, 7 classes" % run, body)


def chart_metrics(ev, run):
    """Left: hold-out IoU / P / R per class (all objects); right: IoU per class and object."""
    names = ev["classes"]
    k = len(names)
    al = ev["pooled"]["all"]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.0), gridspec_kw={"width_ratios": [1.0, 1.0]})
    y = np.arange(k)[::-1].astype(float)
    ax = axes[0]
    V._style(ax)
    ax.grid(axis="y", visible=False)
    for key, name, colour, mk in (("iou", "IoU", V.SERIES[0], "o"), ("prec", "P (precision)", V.SERIES[1], "s"),
                                  ("rec", "R (recall)", V.SERIES[2], "^")):
        ax.plot(al[key], y, mk, ms=7, color=colour, mec=V.SURFACE, mew=1.0, ls="none", label=name, zorder=3)
    for c, yy in enumerate(y):
        ax.annotate("%.3f" % al["iou"][c], (al["iou"][c], yy), xytext=(0, 7), textcoords="offset points", ha="center",
                    fontsize=6.5, color=V.INK)
    ax.set_yticks(y)
    ax.set_yticklabels(["%d %s" % (c, names[c]) for c in range(k)], fontsize=7.5, color=V.INK)
    lo = min(min(al[m]) for m in ("iou", "prec", "rec"))
    ax.set_xlim(max(0.0, np.floor((lo - 0.03) * 20) / 20), 1.005)
    ax.set_title("all hold-out frames: mIoU %.3f, rope (1+2+5) IoU %.3f" % (al["miou"], al["rope_iou"]), fontsize=9,
                 color=V.INK)
    ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.1), ncol=3)
    ax = axes[1]
    V._style(ax)
    ax.grid(axis="y", visible=False)
    objs = (("obj:d001", "d001 black rope", "s"), ("obj:d002", "d002 white cord", "^"), ("obj:d003", "d003 tan rope", "D"),
            ("obj:d004", "d004 cat5e x3", "v"))
    for (g, name, mk), colour in zip(objs, V.SERIES + ("#eda100",)):
        if g in ev["pooled"]:
            ax.plot(ev["pooled"][g]["iou"], y, mk, ms=6, color=colour, mec=V.SURFACE, mew=1.0, ls="none", label=name)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    lo = min(min(ev["pooled"][g]["iou"]) for g, _, _ in objs if g in ev["pooled"])
    ax.set_xlim(max(0.0, np.floor((lo - 0.03) * 20) / 20), 1.005)
    ax.set_title("IoU per class and object", fontsize=9, color=V.INK)
    ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.1), ncol=4)
    for ax in axes:
        ax.set_xlabel("pooled over all in-box points (axis starts at %.2f)" % ax.get_xlim()[0], fontsize=7.5, color=V.INK2)
    fig.tight_layout()
    src = os.path.join(V.FIG_DIR, "metrics_%s.png" % run)
    fig.savefig(src, dpi=V.DPI, facecolor="white")
    plt.close(fig)
    body = ("Hold-out evaluation of run %s on %d clean frames from %d episodes never used in training: each frame's "
            "16,384 sampled points are classified by argmax, then every in-box point takes the class of its 1-NN "
            "sampled point; counts are pooled over all points. Left: IoU, P and R per class over all objects. Right: "
            "IoU per class for each object." % (run, ev["frames"], ev["episodes"]))
    return publish(src, run, 4, "holdout_iou_precision_recall_per_class", "v036 PointNet++ %s: hold-out IoU, "
                   "precision and recall per class" % run, body)


def chart_confusion(ev, run):
    """Row-normalised confusion matrix (label rows, prediction columns), all hold-out points."""
    names = ev["classes"]
    c = np.array(ev["confusion_all"], dtype=float)
    rn = c / np.maximum(1, c.sum(1, keepdims=True))
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    ax.imshow(rn, cmap=cmap, vmin=0, vmax=1)
    k = len(names)
    for i in range(k):
        for j in range(k):
            ax.text(j, i, "%.1f" % (100 * rn[i, j]) if rn[i, j] >= 0.0005 else "0", ha="center", va="center", fontsize=7,
                    color="white" if rn[i, j] > 0.55 else V.INK)
    lbl = ["%d %s" % (i, names[i]) for i in range(k)]
    ax.set_xticks(range(k)); ax.set_xticklabels(lbl, rotation=35, ha="right", fontsize=7, color=V.INK)
    ax.set_yticks(range(k)); ax.set_yticklabels(lbl, fontsize=7, color=V.INK)
    ax.set_xlabel("predicted class", fontsize=8, color=V.INK2)
    ax.set_ylabel("label class (row sums to 100%)", fontsize=8, color=V.INK2)
    ax.set_title("share of each label class's points predicted as each class (%)", fontsize=9, color=V.INK)
    fig.tight_layout()
    src = os.path.join(V.FIG_DIR, "confusion_%s.png" % run)
    fig.savefig(src, dpi=V.DPI, facecolor="white")
    plt.close(fig)
    body = ("Confusion matrix of run %s over all in-box points of %d hold-out clean frames: row = label class, "
            "column = predicted class, each row normalised to 100%% (so the diagonal is each class's recall). Darker "
            "blue = larger share." % (run, ev["frames"]))
    return publish(src, run, 5, "holdout_confusion_matrix", "v036 PointNet++ %s: hold-out confusion matrix, 7 classes"
                   % run, body)


def render_pair(xyz, lab, pred, title, out_png, box, label_note="label"):
    return V.render7(xyz, [(label_note, lab), ("prediction", pred)], title, out_png, box)


if __name__ == "__main__":
    if sys.argv[1:2] == ["curves"] and len(sys.argv) > 2:
        print(chart_curves(sys.argv[2]))
    else:
        sys.exit(__doc__)
