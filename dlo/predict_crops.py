#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_crops.py -- model crops: the rope points a pn2_dlo checkpoint finds in each listed frame's raw
workspace box, for replaying segmentation on them (dlo_data_001 v035 harness).

Input: --frames CSV with columns episode, frame (other columns ignored). Per frame, as eval_seg7.py:
raw pcd (drive_pool) -> robot frame with the episode's own lee_H / lee_offset (_out_v034/eps_data/config.json)
-> the model's workspace box -> fixed-seed random N-point subsample -> model -> class probabilities spread to
every in-box point by 1-NN -> rope = argmax in {1, 2, 5} (7-class model) or in {1} (2-class model);
--threshold t instead keeps points with P(rope) >= t. Raw loading is open3d, checked identical (xyz and rgb,
0 difference) to the pipeline's load_point_cloud_with_rgb + transform_xyz_rgb on dataset frames.
Output under --out_dir:
  <episode>/<frame:06d>.npz  xyzrgb (N, 6) float32 (robot frame, m; rgb in [0, 1] as the pipeline's xyzrgb
                             arrays, i.e. as dlo_raw_crop_pcd loads), prob (N,) float32 = P(rope) per point
  manifest.csv               episode, frame, n_raw, n_pts (in-box points), n_rope, ms (GPU ms per frame), ms_load, status
  predict_log.json           model, sha256, box, counts, throughput
Frames are batched (--batch) on the GB10; raw loading runs in --workers DataLoader processes.
  dgx: docker run --rm --gpus all --ipc host --user 1000:1000 -v /home/smerx:/home/smerx \
         -v /media/smerx:/media/smerx:rslave -w ~/git/pointnetpp dlo_torch \
         python3 dlo/predict_crops.py --frames <frames.csv> [--ckpt dlo/checkpoints/r02/best.pth] [--out_dir DIR]
Platforms: u24_a64 (dlo_torch container on the dgx).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import platform_guard                                                        # noqa: E402
platform_guard.require(("u24_a64",), __file__)

import argparse                                                              # noqa: E402
import csv                                                                   # noqa: E402
import glob                                                                  # noqa: E402
import hashlib                                                               # noqa: E402
import json                                                                  # noqa: E402
import time                                                                  # noqa: E402
import zlib                                                                  # noqa: E402

import numpy as np                                                           # noqa: E402
import open3d as o3d                                                         # noqa: E402
import torch                                                                 # noqa: E402
from torch.utils.data import DataLoader, Dataset                             # noqa: E402

import dlo_data                                                              # noqa: E402
import pn2_dlo                                                               # noqa: E402

# --- Defaults ---
HERE = os.path.dirname(os.path.abspath(__file__))
DEV_DIR = os.path.expanduser("~/git/dlo_data_001/dev")
GT_DIR = os.path.expanduser("~/bags/test_session_001/groundtruth")
GT_VER = "v034"            # config.json (lee_H, lee_offset) and pcd_fnames.txt from _out_<GT_VER>, else the newest
CKPT = os.path.join(HERE, "checkpoints", "r02", "best.pth")
OUT_DIR = os.path.join(DEV_DIR, "studies", "gt_ablation_v035", "_scratch", "pn2crops", "r02")
BATCH = 8
WORKERS = 4
NN_CHUNK = 8192
LOG_EVERY = 200
ROPE = {7: (1, 2, 5), 2: (1,)}

sys.path.insert(0, DEV_DIR)
import drive_pool                                                            # noqa: E402  stdlib only


def out_dir_of(ep):
    """The episode's _out_<GT_VER> dir with a config.json, else its newest _out_* that has one."""
    ed = os.path.join(GT_DIR, ep.split("_", 1)[1], ep)
    pref = os.path.join(ed, "_out_" + GT_VER)
    if os.path.isfile(os.path.join(pref, "eps_data", "config.json")):
        return pref
    c = sorted(d for d in glob.glob(os.path.join(ed, "_out_v*")) if os.path.isfile(os.path.join(d, "eps_data", "config.json")))
    return c[-1] if c else None


class RawFrames(Dataset):
    """(episode, frame) -> in-box robot-frame xyz + rgb of the raw frame (open3d load, pipeline transform)."""

    def __init__(self, items, box):
        self.items, self.box = items, np.asarray(box, dtype=float)
        self.mounts, self.cache = None, {}

    def episode(self, ep):
        if ep not in self.cache:
            if self.mounts is None:
                self.mounts = drive_pool.all_mounts()
            od = out_dir_of(ep)
            if od is None:
                self.cache[ep] = None
            else:
                with open(os.path.join(od, "eps_data", "config.json")) as f:
                    cfg = json.load(f)
                with open(os.path.join(od, "eps_data", "pcd_fnames.txt")) as f:
                    fnames = f.read().split()
                raw = None
                for _n, m in self.mounts:
                    p = os.path.join(m, "bags", drive_pool.SESSION_NAME, "raw_data", ep.split("_", 1)[1], ep, "pcd")
                    if os.path.isdir(p):
                        raw = p
                        break
                self.cache[ep] = (np.asarray(cfg["lee_H"], float), np.asarray(cfg["lee_offset"], float), fnames, raw)
        return self.cache[ep]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ep, fr = self.items[i]
        t0 = time.time()
        out = {"ep": ep, "frame": fr, "xyz": np.zeros((0, 3), np.float32), "rgb": np.zeros((0, 3), np.float32),
               "n_raw": 0, "status": "ok"}
        e = self.episode(ep)
        if e is None or e[3] is None or not (1 <= fr <= len(e[2])):
            out["status"] = "missing " + ("gt config" if e is None else "raw_data" if e[3] is None else "frame")
        else:
            H, off, fnames, raw = e
            pc = o3d.io.read_point_cloud(os.path.join(raw, fnames[fr - 1]), remove_nan_points=False,
                                         remove_infinite_points=False)
            xyz, rgb = np.asarray(pc.points), np.asarray(pc.colors)
            xt = (np.hstack([xyz, np.ones((len(xyz), 1))]) @ H.T)[:, :3] + off       # pipeline transform_xyz_rgb
            b = self.box
            m = (np.isfinite(xt).all(axis=1) & (xt[:, 0] > b[0]) & (xt[:, 0] < b[1]) & (xt[:, 1] > b[2]) &
                 (xt[:, 1] < b[3]) & (xt[:, 2] > b[4]) & (xt[:, 2] < b[5]))
            out.update(xyz=xt[m].astype(np.float32), rgb=(rgb[m] if len(rgb) else np.full((m.sum(), 3), 0.5)).astype(np.float32),
                       n_raw=len(xyz))
        out["ms_load"] = 1e3 * (time.time() - t0)
        return out


def read_frames(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    key = "episode" if "episode" in rows[0] else "ep"
    seen, items = set(), []
    for r in rows:
        it = (r[key].strip(), int(r["frame"]))
        if it not in seen:
            seen.add(it)
            items.append(it)
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--frames", required=True)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--out_dir", default=OUT_DIR)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--threshold", type=float, default=None, help="keep P(rope) >= t instead of argmax in the rope classes")
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--clean", default="false", help="true: delete a non-empty --out_dir first (general_rules §4: "
                    "output only into an empty dir; deleting an old run needs smerx's approval)")
    args = ap.parse_args()
    if os.path.isdir(args.out_dir) and os.listdir(args.out_dir):
        if args.clean != "true":
            sys.exit("refusing to write into non-empty %s (general_rules §4): pass --clean true to delete it first"
                     % args.out_dir)
        import shutil
        shutil.rmtree(args.out_dir)
        print("deleted old %s (--clean true)" % args.out_dir, flush=True)

    ck = torch.load(args.ckpt, map_location="cuda", weights_only=False)
    cfg = ck["cfg"]
    K = int(cfg.get("classes", 2))
    rope = list(ROPE[K])
    pn2_dlo.SAMPLING["random_min"] = cfg["random_min"]
    model = pn2_dlo.get_model(num_classes=K).cuda()
    model.load_state_dict(ck["model"])
    model.eval()
    torch.backends.cudnn.benchmark = True
    box, center, n_points = cfg["box"], np.asarray(cfg["center"]), int(cfg["n_points"])
    with open(args.ckpt, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()

    items = read_frames(args.frames)
    if args.max_frames:
        items = items[:args.max_frames]
    os.makedirs(args.out_dir, exist_ok=True)
    dl = DataLoader(RawFrames(items, box), batch_size=args.batch, num_workers=args.workers, collate_fn=lambda b: b)
    man, t_start, n_done, gpu_ms = [], time.time(), 0, []
    print("%d frames | model %s (%d classes, epoch %d) | rope classes %s | %s" % (
        len(items), args.ckpt, K, ck["epoch"], rope, "P(rope) >= %.2f" % args.threshold if args.threshold else "argmax"),
        flush=True)
    for batch in dl:
        live = [b for b in batch if len(b["xyz"])]
        if live:
            torch.cuda.synchronize(); t0 = time.time()
            xs, idxs = [], []
            for b in live:
                seed = zlib.crc32(("%s_%d" % (b["ep"], b["frame"])).encode())
                idx = np.random.default_rng(seed).choice(len(b["xyz"]), n_points, replace=len(b["xyz"]) < n_points)
                idxs.append(idx)
                xs.append(dlo_data.to_input(b["xyz"][idx].astype(np.float64), np.round(b["rgb"][idx] * 255.0), center))
            torch.manual_seed(zlib.crc32(("%s_%d" % (live[0]["ep"], live[0]["frame"])).encode()))
            X = torch.from_numpy(np.stack(xs)).cuda()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.get("amp", "true") == "true"):
                logp = model(X).float()                                      # [B, N, K]
            p_rope_s = logp.exp()[:, :, rope].sum(-1)                        # [B, N]
            is_rope_s = torch.isin(logp.argmax(-1), torch.tensor(rope, device="cuda"))
            res = []
            for j, b in enumerate(live):
                a = torch.from_numpy(b["xyz"]).cuda()
                s = X[j, :3].T.contiguous() + torch.tensor(center, dtype=torch.float32, device="cuda")
                nn = torch.cat([torch.cdist(a[i:i + NN_CHUNK], s).argmin(1) for i in range(0, len(a), NN_CHUNK)])
                prob = p_rope_s[j][nn]
                keep = (prob >= args.threshold) if args.threshold is not None else is_rope_s[j][nn]
                res.append((prob.cpu().numpy(), keep.cpu().numpy()))
            torch.cuda.synchronize()
            ms = 1e3 * (time.time() - t0) / len(live)
            gpu_ms.append(ms)
            for b, (prob, keep) in zip(live, res):
                d = os.path.join(args.out_dir, b["ep"])
                os.makedirs(d, exist_ok=True)
                np.savez_compressed(os.path.join(d, "%06d.npz" % b["frame"]),
                                    xyzrgb=np.hstack([b["xyz"][keep], b["rgb"][keep]]).astype(np.float32),
                                    prob=prob[keep].astype(np.float32))
                man.append([b["ep"], b["frame"], b["n_raw"], len(b["xyz"]), int(keep.sum()), round(ms, 1),
                            round(b["ms_load"], 1), b["status"]])
        for b in batch:
            if not len(b["xyz"]):
                man.append([b["ep"], b["frame"], b["n_raw"], 0, 0, "", round(b["ms_load"], 1),
                            b["status"] if b["status"] != "ok" else "empty box"])
        n_done += len(batch)
        if n_done % LOG_EVERY < len(batch) or n_done == len(items):
            el = time.time() - t_start
            print("  %d/%d frames, %.1f frames/s, GPU %.1f ms/frame (median)" % (
                n_done, len(items), n_done / el, np.median(gpu_ms) if gpu_ms else float("nan")), flush=True)

    with open(os.path.join(args.out_dir, "manifest.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "frame", "n_raw", "n_pts", "n_rope", "ms", "ms_load", "status"])
        w.writerows(sorted(man, key=lambda r: (r[0], r[1])))
    el = time.time() - t_start
    ok = [r for r in man if r[7] == "ok"]
    log = {"ckpt": os.path.abspath(args.ckpt), "sha256": sha, "classes": K, "epoch": ck["epoch"], "rope_classes": rope,
           "rule": "P(rope) >= %.2f" % args.threshold if args.threshold is not None else "argmax in rope classes",
           "box": box, "n_points": n_points, "gt_config_from": "_out_" + GT_VER + " (else newest)", "frames_file":
           os.path.abspath(args.frames), "frames": len(items), "ok": len(ok), "not_ok": len(man) - len(ok),
           "rope_pts_median": float(np.median([r[4] for r in ok])) if ok else None,
           "wall_s": el, "frames_per_s": len(items) / el,
           "gpu_ms_per_frame_median": float(np.median(gpu_ms)) if gpu_ms else None,
           "load_ms_per_frame_median": float(np.median([r[6] for r in man])) if man else None,
           "batch": args.batch, "workers": args.workers, "gpu": torch.cuda.get_device_name(),
           "created": time.strftime("%Y-%m-%d %H:%M")}
    with open(os.path.join(args.out_dir, "predict_log.json"), "w") as f:
        json.dump(log, f, indent=1)
    print(json.dumps({k: log[k] for k in ("frames", "ok", "not_ok", "rope_pts_median", "wall_s", "frames_per_s",
                                          "gpu_ms_per_frame_median", "load_ms_per_frame_median")}))


if __name__ == "__main__":
    main()
