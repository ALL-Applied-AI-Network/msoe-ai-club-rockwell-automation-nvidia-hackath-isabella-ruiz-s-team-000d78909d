"""Cut good-fixture (G) position crops from the real overhead-camera frames.

Input: the cap_seal folder from the lab (images/ + image_index.csv).
Output (in --out):
    crops/v5_cap_seal_20260923_201527_141712_G_14.png   one crop per position per kept frame
    crops_index.csv     one row per crop: frame, video, time, position, a suggestion, and an empty
                        `label` column for you to fill (or to fill from the lab's labels)
    frames.csv          every frame and why it was kept or skipped
    review/video_<n>.jpg  contact sheets of the fixture in every kept frame, to spot bad frames

Frames are skipped when:
    - the robot arm covers the fixture (a large yellow area over it),
    - nothing changed at any position since the last kept frame of the same video
      (the line sits still for long stretches; --keep-duplicates turns this off).

Crop centres come from real_positions.json: the 30 G positions in the 640x480 image,
computed from the digital twin's calibrated camera and checked against these frames.
The scrap fixture (S) is left out because the twin's S positions don't line up.

The video number starts each crop name, so split_real.py can keep each video on one side:
    python split_real.py --src real_labeled --out splits --group-regex "^(v\\d+)_"

Usage:
    python cut_real_crops.py --src "C:\\...\\cap_seal" --out real_crops
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("This script needs OpenCV: pip install opencv-python")

HERE = os.path.dirname(os.path.abspath(__file__))


def load_positions(path):
    data = json.load(open(path))
    names = sorted(data["positions"])
    return names, np.array([data["positions"][n] for n in names], dtype=float)


def yellow_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (20, 120, 120), (38, 255, 255))
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def largest_blob(mask):
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    return int(stats[1:, cv2.CC_STAT_AREA].max()) if n > 1 else 0


def cut(img, cx, cy, size):
    x0, y0 = int(round(cx - size / 2)), int(round(cy - size / 2))
    pad = max(0, -x0, -y0, x0 + size - img.shape[1], y0 + size - img.shape[0])
    if pad:
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        x0, y0 = x0 + pad, y0 + pad
    return img[y0:y0 + size, x0:x0 + size]


def window_diffs(gray_a, gray_b, pts, r=12):
    """Mean absolute difference around each position (r px half-width)."""
    out = []
    for u, v in pts:
        x0, y0 = int(u) - r, int(v) - r
        a = gray_a[y0:y0 + 2 * r, x0:x0 + 2 * r].astype(np.float32)
        b = gray_b[y0:y0 + 2 * r, x0:x0 + 2 * r].astype(np.float32)
        out.append(float(np.abs(a - b).mean()))
    return np.array(out)


def suggest(mask, u, v, r=7):
    """A hint only: is there yellow (a cap) at the position centre? Never a label."""
    patch = mask[int(v) - r:int(v) + r + 1, int(u) - r:int(u) + r + 1]
    frac = float((patch > 0).mean()) if patch.size else 0.0
    return ("cap_seen" if frac > 0.3 else "no_cap_seen"), round(frac, 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="the cap_seal folder (with images/ and image_index.csv)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--positions", default=os.path.join(HERE, "real_positions.json"))
    ap.add_argument("--crop-px", type=int, default=64)
    ap.add_argument("--arm-area", type=int, default=2500,
                    help="yellow blob size (px) over the fixture that counts as the robot arm")
    ap.add_argument("--change", type=float, default=10.0,
                    help="a frame is new if any position changed by more than this (mean grey level)")
    ap.add_argument("--keep-duplicates", action="store_true")
    args = ap.parse_args()

    index_path = os.path.join(args.src, "image_index.csv")
    img_dir = os.path.join(args.src, "images")
    if not os.path.isfile(index_path) or not os.path.isdir(img_dir):
        sys.exit(f"Expected images/ and image_index.csv inside {args.src}")
    names, pts = load_positions(args.positions)
    m = args.crop_px // 2 + 4
    rx0, ry0 = (pts.min(0) - m).astype(int)
    rx1, ry1 = (pts.max(0) + m).astype(int)

    frames = sorted(csv.DictReader(open(index_path, newline="")), key=lambda r: r["timestamp"])
    crop_dir = os.path.join(args.out, "crops")
    review_dir = os.path.join(args.out, "review")
    os.makedirs(crop_dir, exist_ok=True)
    os.makedirs(review_dir, exist_ok=True)

    crop_rows, frame_rows = [], []
    last_kept = {}            # video -> grey image of last kept frame
    thumbs = {}               # video -> list of (thumbnail, caption)
    for i, fr in enumerate(frames):
        name, video = fr["filename"], fr["video_id"]
        img = cv2.imread(os.path.join(img_dir, name))
        if img is None:
            frame_rows.append([name, video, fr["timestamp"], "skipped", "unreadable", ""])
            continue
        if img.shape[:2] != (480, 640):
            frame_rows.append([name, video, fr["timestamp"], "skipped", f"size {img.shape[1]}x{img.shape[0]}", ""])
            continue
        mask = yellow_mask(img)
        arm = largest_blob(mask[ry0:ry1, rx0:rx1])
        if arm > args.arm_area:
            frame_rows.append([name, video, fr["timestamp"], "skipped", "robot arm over fixture", arm])
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if not args.keep_duplicates and video in last_kept:
            d = window_diffs(gray, last_kept[video], pts)
            if d.max() < args.change:
                frame_rows.append([name, video, fr["timestamp"], "skipped", "no change since last kept frame",
                                   round(float(d.max()), 1)])
                continue
        last_kept[video] = gray
        frame_rows.append([name, video, fr["timestamp"], "kept", "", arm])

        stem = os.path.splitext(name)[0]
        for pname, (u, v) in zip(names, pts):
            fixture, pos = pname[0], int(pname[1:])
            crop_name = f"v{video}_{stem}_{fixture}_{pos:02d}.png"
            cv2.imwrite(os.path.join(crop_dir, crop_name), cut(img, u, v, args.crop_px))
            hint, frac = suggest(mask, u, v)
            crop_rows.append([crop_name, name, video, fr["timestamp"], fixture, pos, hint, frac, ""])

        t = img[ry0:ry1, rx0:rx1].copy()
        for u, v in pts:
            cv2.circle(t, (int(u - rx0), int(v - ry0)), 3, (0, 0, 255), 1)
        thumbs.setdefault(video, []).append((t, fr["timestamp"][11:19]))
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(frames)} frames read", flush=True)

    with open(os.path.join(args.out, "crops_index.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["crop", "frame", "video_id", "timestamp", "fixture", "position", "suggestion", "yellow_at_centre",
                    "label"])
        w.writerows(crop_rows)
    with open(os.path.join(args.out, "frames.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "video_id", "timestamp", "status", "reason", "value"])
        w.writerows(frame_rows)

    for video, items in thumbs.items():
        h, wd = items[0][0].shape[:2]
        cols = 8
        rows = (len(items) + cols - 1) // cols
        sheet = np.full((rows * (h + 16), cols * wd, 3), 255, np.uint8)
        for k, (t, cap) in enumerate(items):
            r, c = divmod(k, cols)
            sheet[r * (h + 16) + 16:(r + 1) * (h + 16), c * wd:(c + 1) * wd] = t
            cv2.putText(sheet, f"{k} {cap}", (c * wd + 3, r * (h + 16) + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 0, 0), 1)
        cv2.imwrite(os.path.join(review_dir, f"video_{video}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])

    kept = sum(1 for r in frame_rows if r[3] == "kept")
    reasons = {}
    for r in frame_rows:
        if r[3] == "skipped":
            reasons[r[4]] = reasons.get(r[4], 0) + 1
    print(f"\nframes: {len(frames)} read, {kept} kept, skipped: {reasons}")
    by_video = {}
    for r in frame_rows:
        if r[3] == "kept":
            by_video[r[1]] = by_video.get(r[1], 0) + 1
    print(f"kept frames by video: {dict(sorted(by_video.items()))}")
    print(f"crops written: {len(crop_rows)} in {crop_dir}")
    print("Next: open review/ to check the kept frames, then fill the `label` column of crops_index.csv "
          "and run apply_labels.py.")


if __name__ == "__main__":
    main()
