#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_dlo.py -- train pn2_dlo on the FIT split of a dlo/ dataset (clean frames); log loss and IoU.

--classes 2 --data data       r01: DLO vs not (index columns n_dlo / n_box)
--classes 7 --data data/seg7  r02: 7-class scene segmentation (index columns n_c0..n_c6)
A validation subset of fit episodes (sha1(ep) % VAL_MOD == 0) is held back from training and picks
best.pth (DLO IoU for 2 classes, mean IoU over all classes otherwise); hold-out episodes are never
read here. Loss: NLL with class weights (max_freq / freq) ** W_POW from the training frames' class
point counts (for 2 classes the same weights as r01). AdamW, warmup + cosine LR, bf16 AMP (distances
stay fp32, see pn2_dlo.sqdist32).
Writes checkpoints/<run>/{best,last}.pth and logs/<run>/{config.json, train_log.csv}.
  dgx: docker run --rm --gpus all --ipc host --user 1000:1000 -v /home/smerx:/home/smerx \
         -w ~/git/pointnetpp dlo_torch python3 dlo/train_dlo.py --run r02 --classes 7 --data data/seg7
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
import math                                                                  # noqa: E402
import time                                                                  # noqa: E402

import numpy as np                                                           # noqa: E402
import torch                                                                 # noqa: E402
import torch.nn.functional as F                                              # noqa: E402
from torch.utils.data import DataLoader                                      # noqa: E402

import dlo_data                                                              # noqa: E402
import pn2_dlo                                                               # noqa: E402

# --- Defaults ---
HERE = os.path.dirname(os.path.abspath(__file__))
RUN = "r01"
DATA = "data"              # dataset dir under dlo/
CLASSES = 2
EPOCHS = 10
BATCH = 8
N_POINTS = 16384
LR = 1e-3
WD = 1e-4
WARMUP = 300               # linear warmup steps, then cosine to 0
WORKERS = 4
VAL_MOD = 20               # fit episodes with sha1(ep) % VAL_MOD == 0 = validation subset
VAL_FRAMES_PER_EP = 10
W_POW = 0.5                # class weight = (max_freq / freq) ** W_POW
RANDOM_MIN = 512           # pn2_dlo.SAMPLING["random_min"]
AMP = "true"
LOG_EVERY = 100
MAX_STEPS = 0              # >0: stop each epoch after this many steps (smoke runs)
RESUME = "false"


def confusion(pred, y, k):
    """[k, k] confusion counts on the GPU, rows = label, columns = prediction."""
    return torch.bincount((y.reshape(-1) * k + pred.reshape(-1)), minlength=k * k).reshape(k, k)


def scores(conf):
    """(summary iou, prec, rec, per-class iou): class 1 for 2 classes, means over classes otherwise."""
    c = conf.double().cpu().numpy()
    tp = np.diag(c)
    iou = tp / np.maximum(1, c.sum(0) + c.sum(1) - tp)
    prec, rec = tp / np.maximum(1, c.sum(0)), tp / np.maximum(1, c.sum(1))
    if len(c) == 2:
        return iou[1], prec[1], rec[1], iou
    return iou.mean(), prec.mean(), rec.mean(), iou


def class_freq(rows, k):
    if k == 2 and "n_dlo" in rows[0]:
        f = sum(r["n_dlo"] for r in rows) / sum(r["n_box"] for r in rows)
        return np.array([1 - f, f])
    n = np.array([sum(r["n_c%d" % c] for r in rows) for c in range(k)], dtype=float)
    return n / n.sum()


def evaluate(model, loader, weight, amp, k):
    model.eval()
    conf, loss_sum, n = torch.zeros(k, k, dtype=torch.long, device="cuda"), 0.0, 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        for x, y in loader:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            logp = model(x)
            loss_sum += float(F.nll_loss(logp.reshape(-1, k), y.reshape(-1), weight=weight)) * len(x)
            n += len(x)
            conf += confusion(logp.argmax(-1), y, k)
    model.train()
    return loss_sum / max(1, n), scores(conf)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for k, v in (("run", RUN), ("data", DATA), ("classes", CLASSES), ("epochs", EPOCHS), ("batch", BATCH),
                 ("n_points", N_POINTS), ("lr", LR), ("wd", WD), ("warmup", WARMUP), ("workers", WORKERS),
                 ("val_mod", VAL_MOD), ("val_frames_per_ep", VAL_FRAMES_PER_EP), ("w_pow", W_POW),
                 ("random_min", RANDOM_MIN), ("amp", AMP), ("log_every", LOG_EVERY), ("max_steps", MAX_STEPS),
                 ("resume", RESUME)):
        ap.add_argument("--" + k, type=type(v), default=v)
    args = ap.parse_args()
    amp, K = args.amp == "true", args.classes
    pn2_dlo.SAMPLING["random_min"] = args.random_min
    torch.backends.cudnn.benchmark = True
    data_dir = os.path.join(HERE, args.data)

    ck_dir = os.path.join(HERE, "checkpoints", args.run)
    log_dir = os.path.join(HERE, "logs", args.run)
    os.makedirs(ck_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    meta = dlo_data.load_meta(data_dir)
    center = dlo_data.box_center(meta["box"])
    fit = dlo_data.read_index(data_dir, split="fit")
    train_rows = [r for r in fit if not dlo_data.is_val(r["ep"], args.val_mod)]
    val_rows = []
    for ep in sorted(set(r["ep"] for r in fit if dlo_data.is_val(r["ep"], args.val_mod))):
        rr = [r for r in fit if r["ep"] == ep]
        val_rows += [rr[i] for i in np.unique(np.linspace(0, len(rr) - 1, args.val_frames_per_ep).astype(int))]
    freq = class_freq(train_rows, K)
    weight = torch.tensor((freq.max() / freq) ** args.w_pow, dtype=torch.float32, device="cuda")
    cfg = dict(vars(args), box=meta["box"], center=center.tolist(), class_freq_train=freq.tolist(),
               dlo_frac_train=float(freq[1]) if K == 2 else None, class_names=meta.get("classes"),
               class_weight=weight.tolist(), sa_cfg=pn2_dlo.SA_CFG, n_train=len(train_rows), n_val=len(val_rows),
               train_eps=len(set(r["ep"] for r in train_rows)), val_eps=len(set(r["ep"] for r in val_rows)))
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    print("%d classes | train %d frames / %d eps, val %d frames / %d eps | freq %s | weight %s" % (
        K, len(train_rows), cfg["train_eps"], len(val_rows), cfg["val_eps"], np.round(freq, 4).tolist(),
        np.round(weight.tolist(), 3).tolist()), flush=True)

    tl = DataLoader(dlo_data.DLOFrames(train_rows, center, args.n_points, augment=True, data_dir=data_dir),
                    batch_size=args.batch, shuffle=True, num_workers=args.workers, drop_last=True, pin_memory=True,
                    persistent_workers=True)
    vl = DataLoader(dlo_data.DLOFrames(val_rows, center, args.n_points, seed=12345, data_dir=data_dir),
                    batch_size=args.batch, num_workers=args.workers, pin_memory=True)
    model = pn2_dlo.get_model(num_classes=K).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_ep = min(len(tl), args.max_steps) if args.max_steps else len(tl)
    total = steps_ep * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup) *
                                              0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    start_ep, best, gstep = 0, -1.0, 0
    log_path = os.path.join(log_dir, "train_log.csv")
    if args.resume == "true" and os.path.isfile(os.path.join(ck_dir, "last.pth")):
        ck = torch.load(os.path.join(ck_dir, "last.pth"), map_location="cuda")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start_ep, best, gstep = ck["epoch"] + 1, ck["best"], ck["gstep"]
        print("resumed after epoch %d (best val IoU %.4f)" % (ck["epoch"], best), flush=True)
    else:
        with open(log_path, "w") as f:
            csv.writer(f).writerow(["kind", "epoch", "gstep", "lr", "loss", "iou", "prec", "rec", "s_per_step"] +
                                   ["iou_c%d" % c for c in range(K)])

    model.train()
    for ep in range(start_ep, args.epochs):
        conf, loss_sum, n_log, t0 = torch.zeros(K, K, dtype=torch.long, device="cuda"), 0.0, 0, time.time()
        for step, (x, y) in enumerate(tl):
            if step >= steps_ep:
                break
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logp = model(x)
            loss = F.nll_loss(logp.reshape(-1, K), y.reshape(-1), weight=weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            gstep += 1
            loss_sum += loss.item(); n_log += 1
            conf += confusion(logp.detach().argmax(-1), y, K)
            if gstep % args.log_every == 0 or step + 1 == steps_ep:
                iou, p, r, ious = scores(conf)
                sps = (time.time() - t0) / n_log
                row = ["train", ep, gstep, sched.get_last_lr()[0], loss_sum / n_log, iou, p, r, sps] + list(ious)
                with open(log_path, "a") as f:
                    csv.writer(f).writerow(row)
                print("ep %d step %d/%d loss %.4f iou %.4f p %.3f r %.3f lr %.2e %.3f s/step" % (
                    ep, step + 1, steps_ep, row[4], iou, p, r, row[3], sps), flush=True)
                conf, loss_sum, n_log, t0 = torch.zeros(K, K, dtype=torch.long, device="cuda"), 0.0, 0, time.time()
        tv = time.time()
        vloss, (iou, p, r, ious) = evaluate(model, vl, weight, amp, K)
        with open(log_path, "a") as f:
            csv.writer(f).writerow(["val", ep, gstep, sched.get_last_lr()[0], vloss, iou, p, r, ""] + list(ious))
        state = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                 "epoch": ep, "gstep": gstep, "best": max(best, iou), "cfg": cfg, "val_iou": iou,
                 "val_iou_per_class": list(map(float, ious))}
        torch.save(state, os.path.join(ck_dir, "last.pth"))
        if iou > best:
            best = iou
            torch.save(state, os.path.join(ck_dir, "best.pth"))
        print("== ep %d val loss %.4f iou %.4f p %.3f r %.3f (best %.4f) per-class %s %.0f s" % (
            ep, vloss, iou, p, r, best, np.round(ious, 3).tolist(), time.time() - tv), flush=True)


if __name__ == "__main__":
    main()
