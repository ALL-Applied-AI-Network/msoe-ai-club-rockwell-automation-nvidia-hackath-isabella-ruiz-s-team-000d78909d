"""Fine-tune an ImageNet-pretrained ResNet18 on digital-twin crops.

Training data is twin images only. Real crops are used only to pick the best
epoch and the threshold (--real-val), never for gradient updates.

Usage:
    python train.py --train twin/train --val twin/val --real-val splits/real_val.csv --out runs/baseline

--train / --val take a folder with one sub-folder per label, or a CSV list
(path,label). Without --val, 15% of --train is held out at random.
"""
import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from labels import CLASSES, DEFECT_IDX, class_counts, load_items
from metrics import evaluate, public
from model import CropDataset, build_model, pick_device, predict, train_transform


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def class_weights(items, defect_weight):
    """Inverse-frequency weights, with defect classes boosted to push recall."""
    counts = np.array([class_counts(items)[c] for c in CLASSES], dtype=float)
    present = counts > 0
    w = np.ones(len(CLASSES))
    w[present] = counts[present].sum() / (present.sum() * counts[present])
    w[DEFECT_IDX] *= defect_weight
    return torch.tensor(w / w[present].mean(), dtype=torch.float32)


def mean_loss(model, items, size, device, criterion, workers):
    probs = predict(model, items, size, device, workers=workers)
    y = torch.tensor([lab for _, lab in items])
    p = torch.tensor(probs).clamp_min(1e-8)
    # Weighted NLL of the softmax output = the same weighted cross-entropy.
    loss = nn.functional.nll_loss(p.log(), y, weight=criterion.weight.cpu())
    return float(loss), probs, y.numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True, action="append",
                    help="twin training set (folder or CSV); repeat to combine several")
    ap.add_argument("--init", default=None,
                    help="start from this checkpoint (e.g. runs/twin_v4/best.pt) instead of ImageNet weights")
    ap.add_argument("--val", default=None, help="twin validation set")
    ap.add_argument("--real-val", default=None, help="real validation list from split_real.py (never trained on)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--defect-weight", type=float, default=2.0, help="extra loss weight on defect classes")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=12)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = pick_device()

    train_items = [it for src in args.train for it in load_items(src)]
    if args.val:
        val_items = load_items(args.val)
    else:
        rng = random.Random(args.seed)
        shuffled = train_items[:]
        rng.shuffle(shuffled)
        n_val = max(1, int(0.15 * len(shuffled)))
        val_items, train_items = shuffled[:n_val], shuffled[n_val:]
    real_items = load_items(args.real_val) if args.real_val else None
    select_items, select_name = (real_items, "real_val") if real_items else (val_items, "twin_val")

    print(f"device: {device}")
    for name, items in (("train", train_items), ("twin_val", val_items), ("real_val", real_items)):
        if items:
            print(f"{name:<9} {len(items):>6} crops  {class_counts(items)}")
    print(f"picking the best epoch on: {select_name}\n")

    loader = DataLoader(CropDataset(train_items, train_transform(args.img_size)), batch_size=args.batch_size,
                        shuffle=True, num_workers=args.workers, drop_last=len(train_items) > args.batch_size)
    model = build_model(pretrained=not args.no_pretrained and not args.init)
    if args.init:
        ckpt = torch.load(args.init, map_location="cpu", weights_only=False)
        if ckpt.get("classes") != CLASSES:
            raise SystemExit(f"{args.init} has classes {ckpt.get('classes')}, expected {CLASSES}")
        model.load_state_dict(ckpt["model"])
        print(f"starting from {args.init} (epoch {ckpt.get('epoch')})")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_items, args.defect_weight).to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    history_path = os.path.join(args.out, "history.csv")
    with open(history_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "twin_val_loss", "twin_val_fa_rate",
                                "select_fa_rate", "select_threshold", "seconds"])

    best_key, best = None, None
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total, seen = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            seen += len(y)
        sched.step()

        val_loss, val_probs, val_y = mean_loss(model, val_items, args.img_size, device, criterion, args.workers)
        val_res = evaluate(val_y, val_probs)
        if select_items is val_items:
            sel_res, sel_loss = val_res, val_loss
        else:
            sel_loss, sel_probs, sel_y = mean_loss(model, select_items, args.img_size, device, criterion, args.workers)
            sel_res = evaluate(sel_y, sel_probs)

        # Lower is better: false alarms at the catch-all threshold, then loss.
        fa = sel_res["false_alarms"]["rate"]
        key = (fa if fa is not None else 1.0, sel_loss)
        secs = time.time() - t0
        print(f"epoch {epoch:>2}  train {total / seen:.4f}  twin_val {val_loss:.4f}  "
              f"{select_name} false alarms {fa if fa is None else round(100 * fa, 1)}% "
              f"@ thr {sel_res['threshold']}  ({secs:.0f}s)")
        with open(history_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, total / seen, val_loss, val_res["false_alarms"]["rate"],
                                    fa, sel_res["threshold"], round(secs, 1)])

        if best_key is None or key < best_key:
            best_key = key
            best = {
                "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "classes": CLASSES,
                "img_size": args.img_size,
                "epoch": epoch,
                # Frozen threshold for the test set: catch-all on the selection set.
                "threshold": sel_res["threshold"],
                "threshold_set": select_name,
                "args": vars(args),
            }
            torch.save(best, os.path.join(args.out, "best.pt"))
            with open(os.path.join(args.out, "best_metrics.json"), "w") as f:
                json.dump({"epoch": epoch, select_name: public(sel_res), "twin_val": public(val_res)}, f, indent=2)

    torch.save({**best, "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "epoch": args.epochs, "note": "last epoch; threshold copied from best"},
               os.path.join(args.out, "last.pt"))
    print(f"\nbest epoch {best['epoch']}, threshold {best['threshold']} (set on {select_name})")
    print(f"saved {os.path.join(args.out, 'best.pt')}")


if __name__ == "__main__":
    main()
