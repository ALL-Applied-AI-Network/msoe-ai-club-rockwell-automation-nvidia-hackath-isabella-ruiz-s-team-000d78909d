"""Evaluate a trained checkpoint on a set of crops.

Writes to --out:
    report.md          missed defects, false alarms, per-class table, confusion matrix
    metrics.json       the same numbers, machine-readable
    predictions.csv    one row per crop: probabilities, defect score, flagged or not
    gallery.html       missed defects, false alarms, open-vs-empty confusions, close calls,
                       each with its Grad-CAM heatmap

Threshold (--threshold):
    checkpoint  the frozen threshold saved by train.py (default; use this for the test set)
    auto        the catch-all threshold on THIS set (validation only)
    <number>    a fixed value

Usage:
    python evaluate.py --checkpoint runs/baseline/best.pt --data splits/real_val.csv --out runs/baseline/eval_real_val
    python evaluate.py --checkpoint runs/baseline/best.pt --data splits/real_test.csv --out runs/baseline/eval_TEST
"""
import argparse
import csv
import json
import os

from gallery import pick_cases, write_html
from labels import CLASSES, load_items, parse_position
from metrics import evaluate, public, report_markdown
from model import load_checkpoint, pick_device, predict


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True, help="folder of class sub-folders, or a CSV list")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", default="checkpoint")
    ap.add_argument("--no-gradcam", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = pick_device()
    model, ckpt = load_checkpoint(args.checkpoint, device)
    size = ckpt["img_size"]

    if args.threshold == "checkpoint":
        thr = ckpt.get("threshold")
        if thr is None:
            raise SystemExit("Checkpoint has no saved threshold; pass --threshold auto or a number.")
    elif args.threshold == "auto":
        thr = None
    else:
        thr = float(args.threshold)

    items = load_items(args.data)
    probs = predict(model, items, size, device, workers=args.workers)
    res = evaluate([lab for _, lab in items], probs, threshold=thr)

    title = f"{os.path.basename(args.data.rstrip('/'))} · epoch {ckpt.get('epoch')}"
    with open(os.path.join(args.out, "report.md"), "w") as f:
        f.write(report_markdown(res, title=title))
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump({"checkpoint": os.path.abspath(args.checkpoint), "data": os.path.abspath(args.data),
                   **public(res)}, f, indent=2)
    with open(os.path.join(args.out, "predictions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "fixture", "position", "true", "pred", *[f"p_{c}" for c in CLASSES],
                    "defect_score", "flagged"])
        for i, (path, lab) in enumerate(items):
            fixture, pos = parse_position(path)
            w.writerow([path, fixture or "", pos or "", CLASSES[lab], CLASSES[res["_pred"][i]],
                        *[f"{p:.5f}" for p in probs[i]], f"{res['_scores'][i]:.5f}", int(res["_flagged"][i])])

    sections = pick_cases(items, res)
    cams = {}
    if not args.no_gradcam:
        from gradcam import save_cams
        cam_dir = os.path.join(args.out, "gradcam")
        os.makedirs(cam_dir, exist_ok=True)
        wanted = sorted({i for idxs in sections.values() for i in idxs})
        dests = [os.path.join(cam_dir, f"{i:05d}.png") for i in wanted]
        save_cams(model, [items[i][0] for i in wanted], size, device, dests)
        cams = dict(zip(wanted, dests))
    write_html(args.out, items, res, probs, sections, cams, title=title)

    d, fa = res["defects"], res["false_alarms"]
    print(f"missed defects {d['missed']}/{d['support']}  false alarms {fa['count']}/{fa['pass_support']}"
          f"  threshold {res['threshold']} ({res['threshold_source']})")
    print(f"report: {os.path.join(args.out, 'report.md')}  gallery: {os.path.join(args.out, 'gallery.html')}")


if __name__ == "__main__":
    main()
