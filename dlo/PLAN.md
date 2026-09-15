# dlo/: PointNet++ DLO extraction from raw L515 frames (groundwork for v036)

Goal: per-point DLO / not-DLO segmentation of the raw depth cloud, in the robot frame, trained on
the v034 ground truth. Upstream files stay untouched; all work lives in this folder.

## How it fits
raw pcd (camera frame, drive) -> `transform_xyz_rgb` with the episode's own `lee_H`/`lee_offset`
(`eps_data/config.json`) -> fixed workspace box (robot frame) -> label = within eps = r+3 mm of the
frame's node polyline (`dlo_nodes_pcd`, ends cut at node 0 / last node) AND within eps of a
`dlo_raw_crop_pcd` point -> npz -> PointNet++ (xyz+rgb, 2 classes) -> hold-out metrics.
The crop alone is not a label: it still holds the pole-top assembly and some gripper points
(coordinator correction, 2026-09-14; crop-only counts kept per frame for comparison).

## Steps
| # | What | Where | Env |
|---|---|---|---|
| 1 | dataset: 30 clean frames per episode, evenly spread; flagged (FAIL/WARN) hold-out frames for renders | `build_dataset.py` -> `data/` | `dlo_melodic` (py3.8, reuses `dlo_perception_v034`) |
| 2 | model: `pointnet2_sem_seg` adapted (xyz+rgb in, raw features into fp1, topk ball query), 2 classes, N=16384 random points, sqrt inverse-frequency class weights; random centroids for levels >= 512 pts + bf16 AMP (benchmark: 0.21 s/step B=8, 32 ms/frame vs 0.44 s, 255 ms all-FPS) | `pn2_dlo.py` | `dlo_torch` |
| 3 | train on the fit split, log loss + DLO IoU per epoch | `train_dlo.py` -> `checkpoints/`, `logs/` | `dlo_torch`, GB10 |
| 4 | hold-out eval: IoU/precision/recall per object and speed, ms/frame; renders of flagged frames | `eval_dlo.py` -> `logs/`, `figs/` | `dlo_torch` |
| 5 | `REPORT.md` | here | |

## Decisions
- Split: `eval_qm/eqm_split.build()` (the code behind `eval/_eval_qm/split.json`), read only, never
  saved; tests 1-100, d000 excluded. Hold-out episodes never enter training. Record: `split_dlo.json`.
- Clean frame: status PASS, no error tag, no warn tag, not recovered.
- Label radius r: `diameter_m_meas/2` from `objects.txt`; d004 has none, uses the thickness-study
  10.27 mm (objects.txt comment), flagged in the report.
- Workspace box: union of every episode's `rm_bg_crop_*` box (far wall pole-anchored, as
  `bg_far_y`) and `dlo_cbox` defaults, plus a margin; checked to contain every label point.
- Transform check per episode: median NN distance, crop points -> transformed raw points.
- CPU: dataset build with 6 workers, dataloader 4 workers (another agent shares this host).

## r02: 7-class scene segmentation (smerx, 2026-09-14)
r01 kept as `models/pn2_dlo_2class_r01.pth` + `models/model_card_2class_r01.json` (checkpoints/r01 untouched).

| # | Class | Rule (geometry first; colour only via the pipeline's own red tests) |
|---|---|---|
| 0 | other | everything else |
| 1 | rope body | within eps of the v034 node path, ends cut; not 2 or 5 |
| 2 | rope gripper end | rope points in the last 5 cm (arc length) of the node path |
| 3 | gripper | non-rope points in `gripper_fk_capsule`, or red near the FK marker (`remove_gripper_red_finger`) |
| 4 | pole and mount | non-rope points in the `fit_pole_shaft` capsule (`remove_pole_shaft`) or its axis extended down to the table plane, or v035 `static_voxels` inside a mount box around the pole marker |
| 5 | rope pole end | rope points in the first 5 cm of the node path |
| 6 | table / red cloth | red points in the pole-bottom box (`remove_set_polbott`), or within 1.5 cm of the episode's table plane |

- Precedence rope > gripper > table > pole > other; pipeline functions imported read only (v034, v035),
  applied with an index column so each mask is exactly what the function drops.
- Data: same eqm_split, 30 clean frames per episode, crop guard with nearest-neighbour replacement;
  `build_dataset7.py` -> `data/seg7/` (+ `table_planes.csv`, `shaft_fits.json`).
- Crop guard (smerx 2026-09-14): "explained" mode is the r02 default: the far share (crop points > 5 cm
  from the node path), counted over crop points outside the gripper capsule / red finger / red cloth /
  table band / shaft capsule and its base / mount-box static voxels, must be <= 0.20. The whole-crop
  mode first specified fails ~70% of checked frames: the v034 crop still holds the pole assembly.
- Pole: static voxels only inside a mount box (x -8..8, y -3..12, z -10..10 cm) around the pole marker;
  the shaft axis is extended down to the table plane; the 161 d001 episodes whose own shaft fit fails
  get the batch's median shaft placed at their own pole marker minus the batch's median
  marker-to-axis offset (checked on own-fit episodes: deviation p50 6-14 mm, p95 22-34 mm).
- Table: per-episode plane (densest 1 cm z-bin below 0.15 m) +-1.5 cm; 198 episodes near z 0.10 m.
- Train/eval: `train_dlo.py --classes 7 --data data/seg7` (sqrt inverse-frequency weights), run r02;
  `eval_seg7.py`: per-class IoU/P/R, confusion matrix, timing; `viz_seg7.py`: figures to
  `dlo_data_001/dev/dev_figs/v036-pn2-r02_NN_*.png` with title and caption bands.
