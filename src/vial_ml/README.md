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

**1. Freeze the real split (do this once, now).** Real images are never trained on.

```bash
python split_real.py --src real_crops/ --out splits/ --val-frac 0.4 --group-regex "^(.+?)_[GS]_?\d+"
```

`--group-regex` pulls the source-image id from each file name so that crops from one camera image never land in both lists. Adjust it to the real file names, or drop it to split by class only. The script refuses to overwrite `real_test.csv`. Don't evaluate on it until week 3.

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
| `split_real.py` | one-time real val/test split | no |
| `gallery.py` | HTML gallery of the cases that matter | no |
| `model.py` | ResNet18, transforms, dataset, prediction | yes |
| `train.py` | training loop and checkpoint selection | yes |
| `evaluate.py` | report, metrics, predictions, gallery | yes |
| `gradcam.py` | heatmaps for the defect score | yes |
