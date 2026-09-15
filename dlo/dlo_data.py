# -*- coding: utf-8 -*-
"""
dlo_data.py -- reading dlo/data (index.csv, meta.json, per-frame npz) for training and evaluation.
Model input per point: xyz minus the workspace-box centre (m) + rgb / 255, as [6, N] float32.
Platforms: u24_a64 (dlo_torch container on the dgx).
"""
import csv
import hashlib
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
INT_COLS = ("frame", "n_raw", "n_box", "n_dlo", "n_dlo_crop", "n_dlo_poly", "n_crop", "crop_out_box")


def load_meta(data_dir=DATA_DIR):
    with open(os.path.join(data_dir, "meta.json")) as f:
        return json.load(f)


def box_center(box):
    return np.array([(box[0] + box[1]) / 2, (box[2] + box[3]) / 2, (box[4] + box[5]) / 2])


def read_index(data_dir=DATA_DIR, split=None, kind="clean"):
    """Rows of index.csv (ints cast) for one split ('fit'/'holdout'/None) and kind."""
    with open(os.path.join(data_dir, "index.csv")) as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        if (split and r["split"] != split) or (kind and r["kind"] != kind):
            continue
        for c in INT_COLS + tuple("n_c%d" % k for k in range(16)):   # r01 index and data/seg7 index
            if r.get(c, "") != "":
                r[c] = int(r[c])
        out.append(r)
    return out


def is_val(ep, mod):
    """Validation subset of the FIT episodes: stable sha1(ep) % mod == 0."""
    return int(hashlib.sha1(ep.encode()).hexdigest(), 16) % mod == 0


def load_frame(data_dir, row):
    d = np.load(os.path.join(data_dir, row["npz"]))
    return d["xyz"], d["rgb"], d["label"]


def to_input(xyz, rgb, center):
    """[n, 3] float + [n, 3] uint8 -> [6, n] float32 model input."""
    return np.concatenate([xyz - center, rgb.astype(np.float32) / 255.0], axis=1).astype(np.float32).T


class DLOFrames(Dataset):
    """One random n_points subsample per frame per epoch; seed fixes it (validation)."""

    def __init__(self, rows, center, n_points, augment=False, seed=None, data_dir=DATA_DIR):
        self.rows, self.center, self.n, self.augment, self.seed, self.dir = rows, center, n_points, augment, seed, data_dir

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        xyz, rgb, lab = load_frame(self.dir, self.rows[i])
        rng = np.random.default_rng(None if self.seed is None else self.seed + i)
        idx = rng.choice(len(xyz), self.n, replace=len(xyz) < self.n)
        xyz, rgb, lab = xyz[idx].astype(np.float64), rgb[idx], lab[idx]
        if self.augment:
            # small rigid + scale + colour jitter; the camera and robot frame are fixed on this rig
            a = np.deg2rad(rng.uniform(-5, 5))
            rot = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
            xyz = (xyz - self.center) @ rot.T * rng.uniform(0.97, 1.03) + rng.uniform(-0.03, 0.03, 3) + self.center
            xyz = xyz + rng.normal(0, 0.001, xyz.shape)
            rgbf = np.clip(rgb.astype(np.float32) * rng.uniform(0.8, 1.2) + rng.uniform(-12, 12, 3), 0, 255)
            rgb = rgbf
        x = to_input(xyz, rgb, self.center)
        return torch.from_numpy(x), torch.from_numpy(lab.astype(np.int64))
