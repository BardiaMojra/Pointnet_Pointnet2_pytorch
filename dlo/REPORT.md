# dlo/ REPORT: PointNet++ DLO extraction from raw L515 frames (v036 groundwork)

Run r01, 2026-09-14. Code: `build_dataset.py`, `pn2_dlo.py`, `train_dlo.py`, `eval_dlo.py`, `viz_dlo.py` (plan in `PLAN.md`).

## 1. Dataset (dlo_melodic, 400 s at 6 workers, 9.9 GB npz, git-ignored)
| Item | Value |
|---|---|
| Ground truth | `_out_v034`, d001-d004 × v050/v070/v100, tests 1-100, 930 episodes |
| Split | `eqm_split.build()`, read only (confirmed by smerx): fit 744 / hold-out 186 episodes; recorded in `split_dlo.json` |
| Frames | 30 clean frames per episode, evenly spread: fit 22,320 (train 21,360 + val 960 from 32 fit episodes, 320 used), hold-out 5,580, plus 12 flagged hold-out frames |
| Clean frame | status PASS, no error tag, no warn tag, not recovered |
| Workspace box (robot frame, m) | x [-0.10, 1.00], y [0.30, 1.80], z [-0.08, 1.10]: union of all `rm_bg_crop_*` boxes (pole-anchored far wall) + 0.10 m; points per frame mean 24k-31k |
| Transform check (crop point to nearest transformed raw point) | worst episode median 3.8e-5 mm, worst single point 7.3e-5 mm (float32 rounding); 0 crop points outside the box |

A point is labelled DLO if it is within eps of the frame's node polyline (`dlo_nodes_pcd`, not past either end) and also within eps of a `dlo_raw_crop_pcd` point. eps = rope radius + 3 mm.

| Object | Diameter (mm) | eps (mm) | DLO fraction, crop only (old) | DLO fraction, polyline and crop (used) |
|---|---|---|---|---|
| d001 black rope | 15.1 | 10.6 | 11.85% | 6.12% |
| d002 white cord | 9.6 | 7.8 | 9.41% | 3.49% |
| d003 tan rope | 18.8 | 12.4 | 11.44% | 5.66% |
| d004 cat5e ×3 | 10.27* | 8.1 | 7.57% | 3.10% |

\* No measured value in `objects.txt`; thickness-study estimate used.

What the crop-only rule labelled that is not rope:
- the pole-top assembly past node 0: about 20-36% of crop points
- points past the gripper end: about 15% of the gripper-zone points
- in t001_d001_v050/v070, 20-44% of each PASS-frame crop is non-rope down to table height

Figures: `figs/labels_d00{1..4}_*.png`.

## 2. Model (`pn2_dlo.py`, adapted from `models/pointnet2_sem_seg.py`)
- 2 classes; input is xyz relative to the box centre plus rgb; N = 16,384 random points per frame; 0.97 M parameters.
- Set-abstraction levels (centroids / radius): 2048/0.03 m, 512/0.08 m, 128/0.20 m, 32/0.50 m.
- The finest propagation layer also receives the raw point features.
- Neighbour search uses topk with fp32 distances.
- Levels with 512 or more centroids choose them at random, not by farthest-point sampling. Benchmark: 0.21 s per training step and 32 ms inference, against 0.44 s and 255 ms with farthest-point sampling everywhere.
- Loss: weighted NLL, DLO class weight 4.37 = sqrt((1-f)/f) with f = 5.0%. AdamW, learning rate 1e-3 with warmup and cosine decay, batch 8, bf16 mixed precision.
- Augmentation: rotation about z of ±5°, scale ±3%, shift ±3 cm, 1 mm jitter, colour jitter.

## 3. Training (GB10, 12 epochs, 0.23 s/step, about 2.2 h)
- Best validation IoU 0.928 after epoch 11 (`checkpoints/r01/best.pth`); epoch 12 gave 0.927.
- CPU use about 1.2 cores. Figure: `figs/train_r01.png`.

## 4. Hold-out evaluation (5,580 clean frames, 186 episodes)
Per frame: the model predicts on 16,384 sampled points, then each remaining in-box point takes the class of its nearest sampled point. Metrics are pooled over all in-box points. The 0.80 threshold was chosen on validation frames from the fit split. Figure: `figs/metrics_r01.png`.

| Group | IoU | Precision | Recall | IoU at argmax |
|---|---|---|---|---|
| **all** | **0.931** | 0.954 | 0.975 | 0.917 |
| d001 black rope | 0.926 | 0.953 | 0.970 | 0.911 |
| d002 white cord | 0.904 | 0.938 | 0.961 | 0.884 |
| d003 tan rope | 0.953 | 0.966 | 0.986 | 0.944 |
| d004 cat5e ×3 | 0.885 | 0.917 | 0.963 | 0.868 |
| v050 | 0.935 | 0.957 | 0.976 | 0.922 |
| v070 | 0.934 | 0.956 | 0.976 | 0.920 |
| v100 | 0.924 | 0.949 | 0.973 | 0.909 |

- **Per-frame IoU:** median 0.932, mean 0.925, 5th percentile 0.854, 0.66% of frames below 0.8, none below 0.5. On the sampled points only, IoU is 0.936.
- **Per object, median / 5th percentile:** d001 0.930/0.884, d002 0.907/0.833, d003 0.957/0.913, d004 0.903/0.796.
- **Weakest cells:** d002_v100 0.878 and d004_v050 0.878.
- **Size effect:** IoU rises with the number of rope points in the frame, from a median of 0.905 in the lowest quartile to 0.955 in the highest.

Timing on the GB10 (B=1, clean GPU, median / p90):

| Stage | ms |
|---|---|
| Model forward | 37.4 / 42.7 |
| On GPU, from in-box cloud (copy, sample, model, nearest-neighbour spread, copy back) | 92.0 / 111.8 |
| CPU: load raw pcd | 20.9 / 45.1 |
| CPU: camera-to-robot transform + box crop | 8.4 / 13.0 |

Flagged frames (FAIL/WARN, IoU measured against the pipeline's own label, which may be wrong). Renders `figs/flagged_*.png` show three rows: before (v034 crop, orange), after (model), after vs label.

| Frame | Status | IoU | Note |
|---|---|---|---|
| t043_d003_v100 f231 | FAIL cone widened | 0.947 | model drops the pole assembly the crop kept |
| t011_d003_v070 f38 | FAIL | 0.935 | |
| t043_d001_v100 f573 | FAIL | 0.927 | |
| t049_d001_v100 f687 | WARN lee sparse | 0.926 | |
| t079_d002_v050 f322 | FAIL | 0.909 | |
| t043_d002_v100 f1008 | FAIL | 0.835 | |
| t010_d004_v050 f976 | WARN | 0.821 | |
| t092_d002_v050 f1168 | WARN recovered | 0.777 | |
| t010_d001_v100 f524 | FAIL | 0.750 | |
| t002_d003_v070 f941 | WARN recovered | 0.740 | model marks rope near the pole that the label misses |
| t004_d004_v050 f210 | FAIL | 0.639 | label polyline misses a loop |
| t004_d004_v050 f695 | FAIL | 0.504 | most "false positives" are on the loop the label misses |

In all 12 flagged frames the model still finds the rope. Worst clean frame per object: `figs/worst_d00{1..4}_*.png`.

## 5. Limitations
- Labels come from the pipeline's own polyline. They inherit its end cuts and anything it missed, and eps is a fixed radius (d004's is estimated).
- Errors concentrate at the rope ends (tape, pole top, gripper) and at flying depth pixels on the thin, bright d002 cord.
- d004 has only 30 episodes, and only 6 are hold-out.
- The model is tied to this rig: absolute robot-frame coordinates and the fixed box. It still needs each episode's calibration (`lee_H`) at inference.
- The threshold and metrics are pooled over points. No node or topology metric yet.
- Timing does not include writing results or ROS I/O.
- The palette validator could not run (no node on the dgx); the charts use the reference palette's slots unchanged.

## 6. Next steps
- Replace the brute-force nearest-neighbour step (about 55 ms) with a sampled-point KD-tree or voxel hash, or run the model on all in-box points in one pass.
- Down-weight or crop the end zones in the loss; add a boundary/Dice term; train on 2-3 random subsamples per frame.
- Relabel FAIL/WARN frames with the model to find pipeline gaps (the t004_d004 loop, the t001_d001 crops), then use this as the v036 extraction step.
- Node segmentation (smerx's idea) can start from these per-point probabilities.

---

## r02: 7-class scene segmentation (2026-09-15)

Weights at `checkpoints/r02/best.pth` (the last epoch, 12). Figures in `dlo_data_001/dev/dev_figs/v036-pn2-r02_NN_*.png` (27, each with a caption band).

### Labels (`build_dataset7.py` → `data/seg7/`, 9.4 GB, 485 s)
Labels come from geometry and the pipeline's own regions; colour enters only through the pipeline's red tests. Precedence: rope > gripper > table/cloth > pole > other. eps = rope radius + 3 mm.

| # | Class | Rule |
|---|---|---|
| 0 | other | everything else |
| 1 | rope body | within eps of the v034 node path, ends cut; not 2 or 5 |
| 2 | rope gripper end | rope points in the last 5 cm (arc length) of the node path |
| 3 | gripper | non-rope points in the forward-kinematics (FK) gripper capsule, or red points within 8 cm of the FK marker (the red finger) |
| 4 | pole and mount | non-rope points in any of three regions: the fitted shaft capsule; its axis extended down to the table plane; static v035 voxels inside the mount box (x -8..8, y -3..12, z -10..10 cm around the pole marker). 161 d001 episodes without their own shaft fit use the batch-median shaft placed at their own pole marker. |
| 5 | rope pole end | rope points in the first 5 cm of the node path |
| 6 | table / red cloth | red points in the pole-bottom box, or points within ±1.5 cm of the episode's table plane |

- **Data:**
  - *Frames.* Same eqm_split: 744 fit / 186 hold-out episodes. 30 clean frames per episode: 22,308 fit and 5,579 hold-out (13 slots unfilled). Plus 12 flagged frames, 11 of them in hold-out episodes.
  - *Split.* eval_qm re-saved `split.json` on 2026-09-14 20:44; against r01's split, t008_d004_v050 is now hold-out and t010_d004_v050 fit. r02 follows the current file, recorded in `split_seg7.json`.
  - *Crop guard.* "Explained" mode: a frame is dropped if more than 20% of its crop points lie over 5 cm from the node path, counting only crop points outside all six regions above (gripper capsule, red finger, red cloth, table band, shaft and its base, mount-box voxels). Of 28,921 checked frames, 1,034 fail in explained mode (3.6%) against 20,402 in whole-crop mode (70.5%). 244 frames were replaced by a neighbour. In the kept frames, the far share is 27% of the whole crop (median) but only 0.2% of the unexplained points (p95 4.9%).
  - *Table plane.* Estimated per episode; median z 0.015 m. 198 episodes sit near 0.10 m: d002_v050 (2), d002_v070 (2), d002_v100 (52), d003_v050 (1), d003_v070 (23), d003_v100 (88), and all 30 d004. Listed in `data/seg7/table_planes.csv`.
  - *d001 shaft fallback.* v034's shaft fit fails in 161 of 300 d001 episodes; there the batch-median shaft is placed at the episode's own pole marker (6-14 mm off at the median, 22-34 mm at p95, checked on episodes with their own fit). d001 pole share: 11% without the fallback, 18.3% with it.
- **Class shares:**

| Class | All | d001 | d002 | d003 | d004 | Training weight |
|---|---|---|---|---|---|---|
| 0 other | 53.4% | 56.9 | 51.9 | 52.5 | 49.5 | 1.00 |
| 1 rope body | 3.84% | 4.71 | 2.53 | 4.44 | 2.30 | 3.74 |
| 2 rope gripper end | 0.59% | 0.71 | 0.48 | 0.61 | 0.41 | 9.53 |
| 3 gripper | 0.97% | 0.84 | 0.86 | 1.12 | 1.34 | 7.43 |
| 4 pole and mount | 17.5% | 19.6 | 19.1 | 14.8 | 15.1 | 1.75 |
| 5 rope pole end | 0.59% | 0.70 | 0.49 | 0.60 | 0.39 | 9.53 |
| 6 table / red cloth | 23.1% | 16.6 | 24.6 | 25.9 | 31.0 | 1.52 |

- **Sanity checks:**
  - 99.9% of class-2 points lie in the FK capsule or the gripper-end box, and 70% of them are green.
  - 96.9% of class-5 points lie in the pole-end box.
  - 75% of class 6 is red cloth.

### Training (`train_dlo.py --classes 7 --data data/seg7`)
- Same PointNet++ as r01, 7 outputs, 16,384 points per frame, batch 8, 12 epochs, AdamW at 1e-3 with a cosine schedule, bf16. Class weights are sqrt(largest share / class share).
- 21,318 training frames from 711 episodes; validation 330 frames from 33 held-back fit episodes. About 2.3 h on the GB10.
- Best validation mIoU 0.855 after epoch 12. Per class: other 0.96, rope body 0.96, gripper end 0.78, gripper 0.64, pole 0.97, pole end 0.72, table 0.96.

### Hold-out evaluation (`eval_seg7.py`, 5,579 clean frames, 186 episodes, points pooled)

| Class | IoU | P | R |
|---|---|---|---|
| 0 other | 0.959 | 0.990 | 0.968 |
| 1 rope body | 0.960 | 0.977 | 0.982 |
| 2 rope gripper end | 0.744 | 0.786 | 0.933 |
| 3 gripper | 0.599 | 0.655 | 0.875 |
| 4 pole and mount | 0.967 | 0.982 | 0.985 |
| 5 rope pole end | 0.702 | 0.718 | 0.969 |
| 6 table / red cloth | 0.946 | 0.962 | 0.982 |
| **mIoU** | **0.840** | | |
| rope (1+2+5 merged) | 0.916 | 0.927 | 0.988 |

| Group | mIoU | Rope IoU |
|---|---|---|
| d001 | 0.836 | 0.910 |
| d002 | 0.823 | 0.882 |
| d003 | 0.858 | 0.944 |
| d004 | 0.813 | 0.862 |
| v050 | 0.848 | 0.921 |
| v070 | 0.844 | 0.919 |
| v100 | 0.827 | 0.908 |

- **Confusions (row %, so the diagonal is recall):**
  - gripper → rope gripper end 8.3%
  - rope gripper end → gripper 5.7%
  - gripper → other 3.3%
  - rope pole end → pole 1.9%
  - other → table 1.6%, and table → other 1.7%
- **Frame mIoU:** median 0.842, 5th percentile 0.759, minimum 0.576.
- **Comparison with r01:** rope IoU 0.916 against r01's 0.917 at argmax (0.931 at r01's 0.80 threshold). The label rule differs, and 2 episodes differ in the split.
- **Timing on the GB10, idle GPU, B=1:** model median 39 ms (p90 55); end to end, including the spread to all in-box points, median 99 ms (p90 123). Raw load plus transform plus crop adds about 30 ms on the CPU.

### Flagged hold-out frames (labels come from v034's node path and can be wrong here)

| Frame | Status | mIoU | Rope IoU |
|---|---|---|---|
| t011_d003_v070 f38 | FAIL | 0.821 | 0.956 |
| t043_d003_v100 f231 | FAIL | 0.802 | 0.928 |
| t043_d001_v100 f573 | FAIL | 0.752 | 0.906 |
| t049_d001_v100 f687 | WARN | 0.798 | 0.890 |
| t079_d002_v050 f322 | FAIL | 0.836 | 0.888 |
| t043_d002_v100 f1008 | FAIL | 0.702 | 0.830 |
| t002_d003_v070 f941 | WARN | 0.624 | 0.750 |
| t010_d001_v100 f524 | FAIL | 0.683 | 0.736 |
| t092_d002_v050 f1168 | WARN | 0.685 | 0.722 |
| t004_d004_v050 f210 | FAIL | 0.706 | 0.604 |
| t004_d004_v050 f695 | FAIL | 0.694 | 0.449 |

On t004_d004_v050 f695 the model finds the whole rope, including a loop the v034 node path misses (839 rope-body points against 317 in the label). The low rope IoU there is a label error.

### Crops for the harness replay (`predict_crops.py`, dlo_torch)
- 2,077 it22_tune frames (68 episodes) to `dlo_data_001/dev/studies/gt_ablation_v035/_scratch/pn2crops/r02/<episode>/<frame:06d>.npz` (`xyzrgb` float32 in the robot frame, `prob` = P(rope)); rope = top class in {1, 2, 5}. None empty; median 1,253 rope points per frame (min 508); 11.6 frames/s.
- Split of those frames: fit 1,140 control + 589 target, hold-out 220 control + 128 target; 89 are exact r02 training or validation frames, so fit and hold-out are reported separately.

### Figures (`dev_figs/v036-pn2-r02_*`)
- **01** label renders, one per object.
- **02** class shares.
- **03** training curves.
- **04** per-class IoU, precision and recall.
- **05** confusion matrix.
- **06** hold-out renders: median and lowest frame per object.
- **07** the 11 flagged frames.

### Limitations
- **Gripper label.** It is the small FK capsule only, so the real gripper body is labelled "other" and gripper precision is 0.66.
- **Rope-end classes.** They are a fixed 5 cm of arc length, so the boundaries with the rope body, gripper and pole are ambiguous.
- **Inherited errors.** Labels inherit v034 node-path errors on hard frames.
- **Approximate geometry.** d001 shafts use the fallback in 161 episodes, and the table plane is one per episode.
- **Timing.** The spread from sampled points to all in-box points takes about 60 of the 99 ms.

### Next
- r03 labels from the improved node paths.
- Gripper class from the whole gripper body instead of the FK capsule.
- Crop replay results from the harness decide whether a crop-from-model loop is worth continuing.
- Faster spread step, and per-class thresholds.
