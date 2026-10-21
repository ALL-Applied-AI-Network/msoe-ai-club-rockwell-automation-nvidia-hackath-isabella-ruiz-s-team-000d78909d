# Vial inspection: training and evaluation (Group 2)

ResNet18 classifier for the five position labels, trained on digital-twin crops only, scored recall-first.

## Setup

```bash
pip install -r requirements.txt
python tests/test_core.py        # checks splitting, metrics and the gallery (no GPU needed)
```

On Brev, install the CUDA build of PyTorch that matches the machine (see pytorch.org), then the rest of `requirements.txt`.

## Data layout

Every set is a folder with one sub-folder per label, or a CSV list with columns `path,label`:

```
twin/train/pass/G_14_pass_0003.png
twin/train/missing_cap/...
twin/train/crooked_cap/...
twin/train/damaged_vial/...
twin/train/no_vial/...
```

Ask Group 1 to put the fixture and position in each file name (`G_14_...`, `S_3_...`). The reports use them, and they let us find position-specific failures later. Missing class folders are fine while the twin is still catching up.

## Order of work

**1. Prepare the real data (do this once, now).** Real images are only for checking the model, never for training.

The lab's `cap_seal` folder has full camera frames but no crops or labels. Turn it into labelled crops:

```bash
# a) cut the 30 good-fixture positions from every useful frame
python cut_real_crops.py --src path/to/cap_seal --out real_crops
```

This skips frames where the robot arm covers the fixture, and frames where nothing changed since the last kept one (the line sits still for long stretches). It writes:
- `real_crops/crops/`: crops named like `v5_cap_seal_..._G_14.png`, where `v5` is the video;
- `real_crops/crops_index.csv`: one row per crop, with an empty `label` column;
- `real_crops/review/`: one contact sheet per video, so you can spot bad frames.

Crop centres come from `real_positions.json`. These are the 30 G positions in the 640 × 480 image, computed from the twin's calibrated camera and checked against the real frames. The scrap fixture is left out for now, because the twin's scrap positions don't line up with the real tray.

```bash
# b) fill the label column (pass, missing_cap, crooked_cap, damaged_vial, no_vial; blank = skip)
#    in Excel or from the lab's labels, then sort the crops into label folders
python apply_labels.py --index real_crops/crops_index.csv --out real_labeled

# c) freeze the split, keeping each video on one side
python split_real.py --src real_labeled --out splits --val-frac 0.4 --group-regex "^(v\d+)_"
```

The `suggestion` column (`cap_seen` / `no_cap_seen`) is only a hint from the colour at the crop centre. Look at every crop before labelling it: an open vial and an empty hole both show `no_cap_seen`.

`split_real.py` refuses to overwrite `real_test.csv`. Don't evaluate on it until week 3. With only 5 videos, check the printed table: each defect class should appear in both lists. If not, rerun with a different `--seed`, but only before anyone has looked at results.

**2. Train.**

```bash
python train.py --train twin/train --val twin/val --real-val splits/real_val.csv --out runs/baseline
```

Each epoch prints the false-alarm rate at the threshold that catches every defect. `best.pt` is the epoch with the fewest false alarms on the real validation list, and it stores that threshold, frozen for the test set.

**3. Evaluate.**

```bash
python evaluate.py --checkpoint runs/baseline/best.pt --data splits/real_val.csv --out runs/baseline/eval_real_val --threshold auto
python evaluate.py --checkpoint runs/baseline/best.pt --data twin/val --out runs/baseline/eval_twin_val
```

Open `report.md` and `gallery.html`. The gallery shows every missed defect, the false alarms, open-vial vs empty-hole mix-ups and the closest calls, each with a Grad-CAM heatmap of what pushed it toward "defect". If the heat sits on glare, the fixture or a neighbour, tell Group 1 what to vary in the twin.

**4. Final test (week 3, once).**

```bash
python evaluate.py --checkpoint runs/final/best.pt --data splits/real_test.csv --out runs/final/eval_TEST
```

Uses the frozen threshold from the checkpoint. Don't tune anything after looking at it.

## How the decision works

- **Defect score** = P(missing_cap) + P(crooked_cap) + P(damaged_vial). A crop is flagged when the score is at or above the threshold. The top-scoring class is reported as the reason.
- **no_vial is not a defect here.** Whether an empty position is an error depends on the fill pattern, which Group 3 handles.
- **Headline number:** false-alarm rate on pass crops at the threshold that catches every known defect. Always report it with the counts, since the real set has few defects.

## Knobs worth trying (change one at a time)

| Flag | Default | Try |
| --- | --- | --- |
| `--defect-weight` | 2.0 | 1 to 4; higher pushes recall, costs false alarms |
| `--epochs` | 15 | more if twin_val loss is still falling |
| `--lr` | 3e-4 | 1e-4 if training is unstable |
| `--img-size` | 224 | larger if caps are small in the crop |

Augmentation (in `model.py`) is deliberately mild: small shifts so the label never drifts to a neighbouring vial, plus brightness and contrast jitter for lighting.

## Files

| File | What it does | Needs PyTorch |
| --- | --- | --- |
| `labels.py` | class names, folder and CSV loading, position parsing | no |
| `metrics.py` | defect score, catch-all threshold, all reported numbers | no |
| `cut_real_crops.py` | real frames → position crops, skipping arm-covered and repeated frames | no (needs OpenCV) |
| `real_positions.json` | pixel centre of each G position in the real camera image | — |
| `apply_labels.py` | sorts labelled crops into one folder per label | no |
| `split_real.py` | one-time real val/test split | no |
| `gallery.py` | HTML gallery of the cases that matter | no |
| `model.py` | ResNet18, transforms, dataset, prediction | yes |
| `train.py` | training loop and checkpoint selection | yes |
| `evaluate.py` | report, metrics, predictions, gallery | yes |
| `gradcam.py` | heatmaps for the defect score | yes |
