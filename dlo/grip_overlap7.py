#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-
"""
grip_overlap7.py -- preview of the r03 gripper class where the rope's gripper end meets the left finger.

Per frame (default: the N_PER_OBJ clean frames of data/seg7 with the most rope-gripper-end points per rope,
different episodes): the 7-class labels under the r02 gripper rule (--grip_mode fk: FK capsule + red finger)
and under the r03 rule (--grip_mode body: whole gripper at the detected marker), same frame, same node path.
Writes data/grip_overlap/<ep>_f<frame>.npz (xyz, rgb, lab_fk, lab_body, marker) and stats.json (per frame:
class 2 / 3 counts under both rules, what each r03 gripper rule contributes, rope-end points inside the
gripper region kept as rope). Renders and captions: viz_seg7.py grip_overlap (dlo_torch).
Output only into an empty dir (general_rules §4): --clean true deletes the previous preview first.
  dgx: docker run --rm --user 1000:1000 -e HOME=/home/smerx -v /home/smerx:/home/smerx \
         -v /media/smerx:/media/smerx:rslave -w ~/git/pointnetpp dlo_melodic python3.8 dlo/grip_overlap7.py
Platforms: u18_a64 (dlo_melodic container on the dgx).
"""
import argparse
import csv
import json
import os
import shutil
import sys

# --- Defaults ---
N_PER_OBJ = 2
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "data", "grip_overlap")

sys.path.insert(0, HERE)
import build_dataset7 as S7   # noqa: E402  platform guard (u18_a64), label code
import numpy as np            # noqa: E402

B, D = S7.B, S7.D


def pick_frames(n):
    rows = [r for r in csv.DictReader(open(os.path.join(HERE, "data", "seg7", "index.csv"))) if r["kind"] == "clean"]
    out = []
    for dlo in ("d001", "d002", "d003", "d004"):
        rr = sorted((r for r in rows if r["batch"].startswith(dlo)), key=lambda r: -int(r["n_c2"]))
        eps = []
        for r in rr:
            if r["ep"] not in eps:
                eps.append(r["ep"])
                out.append((r["batch"], r["ep"], int(r["frame"])))
            if len(eps) == n:
                break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n_per_obj", type=int, default=N_PER_OBJ)
    ap.add_argument("--frames", default="", help="CSV with episode, frame columns instead of the default pick (any "
                    "v034 frame, e.g. flagged ones; labels then come from that frame's own v034 node path)")
    ap.add_argument("--out_dir", default=OUT_DIR)
    ap.add_argument("--clean", default="false")
    args = ap.parse_args()
    if os.path.isdir(args.out_dir) and os.listdir(args.out_dir):
        if args.clean != "true":
            sys.exit("refusing to write into non-empty %s (general_rules §4): pass --clean true" % args.out_dir)
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir)
    meta = json.load(open(os.path.join(HERE, "data", "meta.json")))
    radii = B.label_radii()
    mounts = B.drive_pool.all_mounts()
    stats = []
    if args.frames:
        with open(args.frames) as f:
            todo = [(r["episode"].split("_", 1)[1], r["episode"], int(r["frame"])) for r in csv.DictReader(f)]
    else:
        todo = pick_frames(args.n_per_obj)
    for batch, ep, fr in todo:
        od = os.path.join(B.GT_DIR, batch, ep, "_out_" + B.OUT_VER)
        raw_dir = B.resolve_raw(mounts, batch, ep)
        cfg = S7.load_cfg(od, raw_dir)
        fl = D.read_frame_log(od)
        lee = S7.episode_lee(fl)
        with open(os.path.join(od, "eps_data", "pcd_fnames.txt")) as f:
            fnames = f.read().split()
        shaft = S7.shaft_job((batch, ep, raw_dir))[2]
        S7.GRIP_MODE = "body"
        eps, static, table_z, grip_static = S7.episode_maps(cfg, fnames, meta["box"], shaft, od, lee)
        labs, rows = {}, {}
        for mode in ("fk", "body"):
            S7.GRIP_MODE = mode
            p = os.path.join(args.out_dir, "_tmp_%s.npz" % mode)
            rows[mode] = S7.build_frame7(od, raw_dir, fnames[fr - 1], fr, cfg, eps, static, table_z, meta["box"],
                                         radii[batch[:4]][0], p, grip_static, lee)
            z = np.load(p)
            labs[mode] = z["label"]
            xyz, rgb = z["xyz"], z["rgb"]
            os.remove(p)
        np.savez_compressed(os.path.join(args.out_dir, "%s_f%06d.npz" % (ep, fr)), xyz=xyz, rgb=rgb,
                            lab_fk=labs["fk"], lab_body=labs["body"], marker=np.asarray(lee.get(fr), dtype=np.float32))
        keys = ("n_c2", "n_c3", "c3_col", "c3_static", "c3_static_only", "c3_red_finger", "c2_in_grip_region")
        st = {"batch": batch, "ep": ep, "frame": fr, "marker": rows["body"]["marker"],
              "n_grip_static_voxels": len(grip_static) if grip_static else 0,
              "fk": {k: rows["fk"][k] for k in keys}, "body": {k: rows["body"][k] for k in keys},
              "rope_unchanged": bool(np.array_equal(np.isin(labs["fk"], (1, 2, 5)), np.isin(labs["body"], (1, 2, 5))))}
        stats.append(st)
        print(json.dumps(st), flush=True)
    with open(os.path.join(args.out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=1)


if __name__ == "__main__":
    main()
