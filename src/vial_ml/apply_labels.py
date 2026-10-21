"""Sort labelled real crops into one folder per label, ready for split_real.py.

Fill the `label` column of crops_index.csv (from cut_real_crops.py) with one of
    pass, missing_cap, crooked_cap, damaged_vial, no_vial
Leave a row blank to skip it (unsure, blurry, half-covered). Then:

    python apply_labels.py --index real_crops/crops_index.csv --out real_labeled
    python split_real.py --src real_labeled --out splits --group-regex "^(v\\d+)_"

Crops are copied, not moved, so you can relabel and rerun (delete --out first).
"""
import argparse
import csv
import os
import shutil
import sys

from labels import CLASSES


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True, help="crops_index.csv with the label column filled in")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    crop_dir = os.path.join(os.path.dirname(os.path.abspath(args.index)), "crops")
    if os.path.exists(args.out) and any(os.scandir(args.out)):
        sys.exit(f"{args.out} is not empty; delete it first so old labels don't linger")

    rows = list(csv.DictReader(open(args.index, newline="", encoding="utf-8-sig")))
    counts = {c: 0 for c in CLASSES}
    blank, bad = 0, []
    for i, r in enumerate(rows, start=2):
        lab = (r.get("label") or "").strip().lower().replace(" ", "_")
        if not lab:
            blank += 1
            continue
        if lab not in counts:
            bad.append(f"line {i}: '{r['label']}' ({r['crop']})")
            continue
        src = os.path.join(crop_dir, r["crop"])
        if not os.path.exists(src):
            bad.append(f"line {i}: crop file missing ({r['crop']})")
            continue
        os.makedirs(os.path.join(args.out, lab), exist_ok=True)
        shutil.copy2(src, os.path.join(args.out, lab, r["crop"]))
        counts[lab] += 1

    if bad:
        print(f"{len(bad)} rows have problems (fix them and rerun):")
        for b in bad[:20]:
            print("  " + b)
    print(f"copied {sum(counts.values())} crops; {blank} rows left blank")
    for c in CLASSES:
        print(f"  {c:<14}{counts[c]:>6}")
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
