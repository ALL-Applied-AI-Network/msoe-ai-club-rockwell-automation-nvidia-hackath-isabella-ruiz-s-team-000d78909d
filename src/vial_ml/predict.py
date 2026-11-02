"""Inspect whole 640x480 station images: 30 good-fixture positions in, a label and defect score per position out.

Use it four ways:
    python predict.py --model runs/mixed_v1/best.pt --image photo.png              # one photo
    python predict.py --model runs/mixed_v1/best.pt --folder station_oct30          # every photo in a folder
    python predict.py --model runs/mixed_v1/best.pt --watch incoming --show         # live: new photos as they appear
    python predict.py --model runs/mixed_v1/best.pt --camera 0 --show               # live: straight from a camera

For each image it writes (in --out, default live_out/):
    <name>_result.jpg   the photo with a box per position: green = pass, red = defect flagged, grey = no_vial
    <name>_result.json  every position's label, defect score and flag, plus timing
and one line per image to results.csv. --show also opens a window (press q to quit).

A position is FLAGGED when its defect score (P(missing)+P(crooked)+P(damaged)) is at or above the
checkpoint's threshold, the same catch-all rule used in evaluation. --threshold overrides it.

For Aiden and Nolan (backend), import and call:
    from predict import Inspector
    insp = Inspector("best.pt")
    result = insp.inspect(bgr_image)     # numpy array from cv2.imread, 640x480
    result["positions"]                  # list of 30 dicts: position, label, defect_score, flagged, probs
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("This script needs OpenCV: pip install opencv-python")
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cut_real_crops import cut, largest_blob, load_positions, yellow_mask  # noqa: E402
from labels import CLASSES, DEFECT_IDX  # noqa: E402
from model import eval_transform, load_checkpoint, pick_device  # noqa: E402

COLOURS = {"pass": (60, 200, 60), "no_vial": (160, 160, 160), "flag": (40, 40, 230)}


class Inspector:
    def __init__(self, model_path, positions=os.path.join(HERE, "real_positions.json"), crop_px=64,
                 threshold=None, arm_area=2500, device=None):
        self.device = device or pick_device()
        self.model, ckpt = load_checkpoint(model_path, self.device)
        self.transform = eval_transform(ckpt.get("img_size", 224))
        self.threshold = threshold if threshold is not None else ckpt.get("threshold")
        if self.threshold is None:
            self.threshold = 0.5
            print("checkpoint has no threshold; using 0.5 (pass --threshold to set one)")
        self.names, self.pts = load_positions(positions)
        self.crop_px, self.arm_area = crop_px, arm_area
        m = crop_px // 2 + 4
        self.region = (*(self.pts.min(0) - m).astype(int), *(self.pts.max(0) + m).astype(int))

    @torch.no_grad()
    def predict_probs(self, bgr, pts=None):
        """Softmax probabilities (30 x 5) for the crops at pts (default: the calibrated positions)."""
        pts = self.pts if pts is None else pts
        batch = torch.stack([
            self.transform(Image.fromarray(cv2.cvtColor(cut(bgr, u, v, self.crop_px), cv2.COLOR_BGR2RGB)))
            for u, v in pts]).to(self.device)
        return torch.softmax(self.model(batch), dim=1).cpu().numpy()

    @torch.no_grad()
    def inspect(self, bgr):
        t0 = time.time()
        if bgr is None or bgr.ndim != 3:
            raise ValueError("expected a colour image")
        if bgr.shape[:2] != (480, 640):
            bgr = cv2.resize(bgr, (640, 480))   # positions are calibrated for 640x480
        x0, y0, x1, y1 = self.region
        arm = largest_blob(yellow_mask(bgr)[y0:y1, x0:x1])
        if arm > self.arm_area:
            return {"status": "skipped", "reason": "robot arm over fixture", "positions": [],
                    "seconds": round(time.time() - t0, 3)}
        probs = self.predict_probs(bgr)
        out = []
        for name, p in zip(self.names, probs):
            score = float(p[DEFECT_IDX].sum())
            out.append({"position": name, "label": CLASSES[int(p.argmax())], "defect_score": round(score, 4),
                        "flagged": score >= self.threshold,
                        "probs": {c: round(float(v), 4) for c, v in zip(CLASSES, p)}})
        return {"status": "ok", "threshold": self.threshold, "positions": out,
                "flagged": [r["position"] for r in out if r["flagged"]],
                "seconds": round(time.time() - t0, 3)}

    def draw(self, bgr, result):
        img = cv2.resize(bgr, (640, 480)) if bgr.shape[:2] != (480, 640) else bgr.copy()
        if result["status"] != "ok":
            cv2.putText(img, result["reason"], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            return img
        h = self.crop_px // 2 - 6
        for r, (u, v) in zip(result["positions"], self.pts):
            col = COLOURS["flag"] if r["flagged"] else COLOURS.get(r["label"], COLOURS["pass"])
            cv2.rectangle(img, (int(u - h), int(v - h)), (int(u + h), int(v + h)), col, 2 if r["flagged"] else 1)
            if r["flagged"]:
                cv2.putText(img, r["label"].split("_")[0], (int(u - h), int(v - h) - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
        n = len(result["flagged"])
        cv2.putText(img, f"{n} flagged  ({result['seconds'] * 1000:.0f} ms)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOURS["flag"] if n else COLOURS["pass"], 2)
        return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image")
    src.add_argument("--folder")
    src.add_argument("--watch", help="folder to watch for new photos (live)")
    src.add_argument("--camera", type=int, help="camera index for cv2.VideoCapture (live)")
    ap.add_argument("--out", default="live_out")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--show", action="store_true", help="open a window with the result")
    ap.add_argument("--every", type=float, default=1.0, help="seconds between checks in --watch/--camera mode")
    args = ap.parse_args()

    insp = Inspector(args.model, threshold=args.threshold)
    print(f"device {insp.device}, threshold {insp.threshold:.4f}")
    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "results.csv")
    new_log = not os.path.exists(log_path)
    log = open(log_path, "a", newline="")
    lw = csv.writer(log)
    if new_log:
        lw.writerow(["image", "status", "flagged", "n_flagged", "seconds"])

    def handle(name, bgr):
        res = insp.inspect(bgr)
        res["image"] = name
        stem = os.path.splitext(os.path.basename(name))[0]
        vis = insp.draw(bgr, res)
        cv2.imwrite(os.path.join(args.out, stem + "_result.jpg"), vis)
        json.dump(res, open(os.path.join(args.out, stem + "_result.json"), "w"), indent=1)
        lw.writerow([name, res["status"], " ".join(res.get("flagged", [])), len(res.get("flagged", [])),
                     res["seconds"]])
        log.flush()
        print(f"{name}: {res['status']}, flagged {res.get('flagged', [])} ({res['seconds'] * 1000:.0f} ms)",
              flush=True)
        if args.show:
            cv2.imshow("vial inspection", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt

    def is_photo(f):
        return f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))

    try:
        if args.image:
            handle(args.image, cv2.imread(args.image))
            if args.show:
                cv2.waitKey(0)
        elif args.folder:
            for f in sorted(filter(is_photo, os.listdir(args.folder))):
                handle(f, cv2.imread(os.path.join(args.folder, f)))
        elif args.watch:
            seen = set()
            print(f"watching {args.watch} for new photos (Ctrl+C to stop)")
            while True:
                for f in sorted(filter(is_photo, os.listdir(args.watch))):
                    if f not in seen:
                        seen.add(f)
                        time.sleep(0.2)   # let the camera software finish writing the file
                        img = cv2.imread(os.path.join(args.watch, f))
                        if img is not None:
                            handle(f, img)
                time.sleep(args.every)
        else:
            cap = cv2.VideoCapture(args.camera)
            if not cap.isOpened():
                sys.exit(f"could not open camera {args.camera}")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            i = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    sys.exit("camera stopped returning frames")
                handle(f"cam_{time.strftime('%Y%m%d_%H%M%S')}_{i:05d}.png", frame)
                i += 1
                time.sleep(args.every)
    except KeyboardInterrupt:
        print("stopped")
    log.close()


if __name__ == "__main__":
    main()
