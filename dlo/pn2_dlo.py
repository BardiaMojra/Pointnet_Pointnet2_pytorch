#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pn2_dlo.py -- PointNet++ (SSG) semantic segmentation, DLO vs background, xyz + rgb input.

Adapted from models/pointnet2_sem_seg.py (upstream files are not modified):
  - 6 input channels: xyz (robot frame, m, minus the workspace-box centre) + rgb in [0, 1]
  - radii sized for a 1-2 cm rope in a ~1.1 x 1.6 x 1.3 m workspace at N = 16384 points
  - fp1 also gets the raw per-point features (upstream passes None), so the finest level sees colour
  - ball query and 3-NN interpolation use topk instead of a full sort, distances always in fp32
  - levels with npoint >= SAMPLING["random_min"] pick centroids at random instead of FPS (speed)
Benchmark: python3 dlo/pn2_dlo.py [B N random_min amp]
Platforms: u24_a64 (dlo_torch container on the dgx; the host itself has no torch).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import platform_guard                                                        # noqa: E402
platform_guard.require(("u24_a64",), __file__)

import torch                                                                 # noqa: E402
import torch.nn as nn                                                        # noqa: E402
import torch.nn.functional as F                                              # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import models.pointnet2_utils as U                                           # noqa: E402
from models.pointnet2_utils import (PointNetSetAbstraction, PointNetFeaturePropagation,  # noqa: E402
                                    index_points, square_distance)

# --- Defaults ---
# (npoint, radius m, nsample, mlp) per set-abstraction level
SA_CFG = ((2048, 0.03, 32, (32, 32, 64)),
          (512, 0.08, 32, (64, 64, 128)),
          (128, 0.20, 32, (128, 128, 256)),
          (32, 0.50, 32, (256, 256, 512)))
IN_FEAT = 6
SAMPLING = {"random_min": 512}   # read at call time; train/eval store it in the checkpoint config

_fps = U.farthest_point_sample


def sqdist32(src, dst):
    """Upstream square_distance, forced to fp32 outside autocast (mm-scale radii need it)."""
    with torch.autocast("cuda", enabled=False):
        return square_distance(src.float(), dst.float())


def sample_centroids(xyz, npoint):
    """Random centroids for large levels (the input is already a random subsample), FPS otherwise."""
    if npoint >= SAMPLING["random_min"]:
        return torch.rand(xyz.shape[0], xyz.shape[1], device=xyz.device).argsort(dim=1)[:, :npoint]
    return _fps(xyz, npoint)


def query_ball_point_topk(radius, nsample, xyz, new_xyz):
    """nsample nearest points within radius (padded with the nearest one); [B, S, nsample]."""
    d, idx = torch.topk(sqdist32(new_xyz, xyz), nsample, dim=-1, largest=False, sorted=True)
    return torch.where(d > radius ** 2, idx[:, :, :1].expand_as(idx), idx)


# upstream sample_and_group() looks both up at call time; this process uses the versions above
U.query_ball_point = query_ball_point_topk
U.farthest_point_sample = sample_centroids


class FeaturePropagationTopk(PointNetFeaturePropagation):
    """Upstream feature propagation with topk(3) in place of a full sort."""

    def forward(self, xyz1, xyz2, points1, points2):
        xyz1, xyz2, points2 = xyz1.permute(0, 2, 1), xyz2.permute(0, 2, 1), points2.permute(0, 2, 1)
        n = xyz1.shape[1]
        if xyz2.shape[1] == 1:
            interp = points2.repeat(1, n, 1)
        else:
            d, idx = torch.topk(sqdist32(xyz1, xyz2), 3, dim=-1, largest=False)
            w = 1.0 / (d.clamp_min(0) + 1e-8)
            w = w / w.sum(dim=-1, keepdim=True)
            interp = torch.sum(index_points(points2, idx) * w.unsqueeze(-1), dim=2)
        new = interp if points1 is None else torch.cat([points1.permute(0, 2, 1).to(interp.dtype), interp], dim=-1)
        new = new.permute(0, 2, 1)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new = F.relu(bn(conv(new)))
        return new


class get_model(nn.Module):
    def __init__(self, num_classes=2, in_feat=IN_FEAT, sa_cfg=SA_CFG):
        super(get_model, self).__init__()
        chans, last = [], in_feat
        self.sa = nn.ModuleList()
        for npoint, radius, nsample, mlp in sa_cfg:
            self.sa.append(PointNetSetAbstraction(npoint, radius, nsample, last + 3, list(mlp), False))
            last = mlp[-1]
            chans.append(last)
        # decoder: each fp merges level k features with the upsampled level k+1 features
        self.fp = nn.ModuleList([
            FeaturePropagationTopk(chans[3] + chans[2], [256, 256]),
            FeaturePropagationTopk(256 + chans[1], [256, 256]),
            FeaturePropagationTopk(256 + chans[0], [256, 128]),
            FeaturePropagationTopk(128 + in_feat, [128, 128, 128])])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, x):
        """x: [B, 6, N] -> per-point log-probabilities [B, N, num_classes] (fp32)."""
        xyz, feats = [x[:, :3, :]], [x]
        for sa in self.sa:
            l_xyz, l_pts = sa(xyz[-1], feats[-1])
            xyz.append(l_xyz)
            feats.append(l_pts)
        up = feats[4]
        for k, fp in zip((3, 2, 1, 0), self.fp):
            up = fp(xyz[k], xyz[k + 1], feats[k], up)
        y = self.conv2(self.drop1(F.relu(self.bn1(self.conv1(up)))))
        return F.log_softmax(y.float(), dim=1).permute(0, 2, 1)


if __name__ == "__main__":
    # smoke benchmark: train step and B=1 inference time
    import time
    torch.backends.cudnn.benchmark = True
    a = sys.argv[1:] + [None] * 4
    B, N = int(a[0] or 8), int(a[1] or 16384)
    SAMPLING["random_min"] = int(a[2] or 512)
    amp = (a[3] or "true") == "true"
    dev = torch.device("cuda")
    m = get_model().to(dev)
    print("params %.2fM | B=%d N=%d random_min=%d amp=%s" % (
        sum(p.numel() for p in m.parameters()) / 1e6, B, N, SAMPLING["random_min"], amp))
    x = torch.rand(B, 6, N, device=dev)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    ts = []
    for i in range(8):
        torch.cuda.synchronize(); t = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logp = m(x)
        loss = F.nll_loss(logp.reshape(-1, 2), torch.randint(0, 2, (B * N,), device=dev))
        opt.zero_grad(); loss.backward(); opt.step()
        torch.cuda.synchronize(); ts.append(time.time() - t)
    print("train step: %.3f s (median of last 5)" % sorted(ts[-5:])[2])
    m.eval()
    ts = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        for i in range(10):
            torch.cuda.synchronize(); t = time.time(); m(x[:1]); torch.cuda.synchronize()
            ts.append(time.time() - t)
    print("infer B=1: %.1f ms (median of last 7) | max mem %.1f GB" % (
        1e3 * sorted(ts[-7:])[3], torch.cuda.max_memory_allocated() / 1e9))
