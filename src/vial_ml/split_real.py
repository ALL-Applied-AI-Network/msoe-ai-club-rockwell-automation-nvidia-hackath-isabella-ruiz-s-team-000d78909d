"""Split the real sample crops ONCE into a validation list and a frozen test list.

Real images are never used for training. The validation list is for choosing
checkpoints and the threshold; the test list is opened once, in week 3.

Crops from the same camera image overlap (neighbouring vials appear in each
other's crops), so split by scene when you can: pass --group-regex with one
capture group that pulls the source-image id out of the file name, e.g.
    --group-regex "^(.+?)_[GS]_?\\d+"   for names like  img0042_G_14.png

Usage:
    python split_real.py --src real_crops/ --out splits/ --val-frac 0.4
"""
import argparse
import os
import random
import re
import sys
from collections import defaultdict

from labels import CLASSES, class_counts, scan_folder, write_list


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="folder with one sub-folder per label")
    ap.add_argument("--out", required=True, help="where to write real_val.csv and real_test.csv")
    ap.add_argument("--val-frac", type=float, default=0.4, help="share of scenes/crops for validation")
    ap.add_argument("--group-regex", default=None, help="regex whose group 1 is the source-image id")
    ap.add_argument("--seed", type=int, default=12)
    args = ap.parse_args()

    test_path = os.path.join(args.out, "real_test.csv")
    if os.path.exists(test_path):
        sys.exit(f"{test_path} already exists. The test split is frozen; delete it only if the team agrees.")

    items = scan_folder(args.src)
    rng = random.Random(args.seed)

    if args.group_regex:
        rx = re.compile(args.group_regex)
        groups = defaultdict(list)
        for path, idx in items:
            m = rx.search(os.path.basename(path))
            if not m:
                sys.exit(f"--group-regex did not match {os.path.basename(path)}")
            groups[m.group(1)].append((path, idx))
        keys = sorted(groups)
        rng.shuffle(keys)
        n_val = max(1, round(len(keys) * args.val_frac))
        val = [it for k in keys[:n_val] for it in groups[k]]
        test = [it for k in keys[n_val:] for it in groups[k]]
        print(f"Split {len(keys)} scenes: {n_val} validation, {len(keys) - n_val} test")
    else:
        # Stratified by class so rare defects land in both lists.
        by_class = defaultdict(list)
        for it in items:
            by_class[it[1]].append(it)
        val, test = [], []
        for idx in sorted(by_class):
            group = by_class[idx]
            rng.shuffle(group)
            n_val = round(len(group) * args.val_frac)
            if len(group) >= 2:
                n_val = min(max(n_val, 1), len(group) - 1)
            val += group[:n_val]
            test += group[n_val:]
        print("Split by class (no scene grouping; neighbouring crops may leak between lists)")

    os.makedirs(args.out, exist_ok=True)
    write_list(os.path.join(args.out, "real_val.csv"), val)
    write_list(test_path, test)

    vc, tc = class_counts(val), class_counts(test)
    print(f"\n{'label':<14}{'val':>6}{'test':>6}")
    for c in CLASSES:
        print(f"{c:<14}{vc[c]:>6}{tc[c]:>6}")
    for c in ("missing_cap", "crooked_cap", "damaged_vial"):
        if tc[c] < 5:
            print(f"note: only {tc[c]} {c} crops in test; report counts next to every recall figure")


if __name__ == "__main__":
    main()
