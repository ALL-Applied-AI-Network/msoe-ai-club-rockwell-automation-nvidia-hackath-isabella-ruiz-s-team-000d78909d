"""The inspection pipeline behind the app: image checks, fixture cropping, model, fill pattern, verdict.

    photo (any size) -> resize to 640x480
      -> robot-arm check          (skip the frame if the arm covers the fixture)
      -> image quality            (brightness and sharpness of the 30 crop windows)
      -> fixture alignment        (median offset of the yellow caps from their calibrated spots;
                                   crops follow the fixture if it has shifted)
      -> model                    (30 crops -> 5 class probabilities each)
      -> catch-all rule           (defect score >= threshold -> flagged)
      -> fill pattern             (vials are placed G01, G02, ... in order; an empty hole inside
                                   the fill count is a missed placement)
      -> verdict                  PASS / REJECT / CHECK

Nothing here imports torch, so the app also runs in demo mode on a laptop without the model.
"""
import os
import sys

import numpy as np
import cv2

VIAL_ML = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, VIAL_ML)
from cut_real_crops import largest_blob, load_positions, yellow_mask  # noqa: E402

CLASSES = ["pass", "missing_cap", "crooked_cap", "damaged_vial", "no_vial"]
DEFECT_IDX = [1, 2, 3]
POSITIONS_JSON = os.path.join(VIAL_ML, "real_positions.json")

# Measured on the 89 real cap_seal frames (see README "Image checks"); outside these the
# photo doesn't look like the station camera, so results are marked CHECK.
QUALITY_LIMITS = {"brightness": (120.0, 200.0), "sharpness_min": 1500.0}
# Real frames: brightness 145-177, sharpness 2700-3400 (a 1 px blur drops it to ~350).
# Caps normally sit ~4 px above the calibrated crop centres (median over 51 real frames); alignment
# measures movement relative to that, so an unmoved fixture gets the same crops as in training.
NORMAL_CAP_OFFSET = np.array([-0.1, -4.3])


class Pipeline:
    def __init__(self, predictor, positions=POSITIONS_JSON, crop_px=64, arm_area=2500,
                 align=True, min_shift=6.0, max_shift=15.0):
        self.predictor = predictor
        self.names, self.pts = load_positions(positions)
        self.crop_px, self.arm_area = crop_px, arm_area
        self.align, self.min_shift, self.max_shift = align, min_shift, max_shift
        m = crop_px // 2 + 4
        self.region = (*(self.pts.min(0) - m).astype(int), *(self.pts.max(0) + m).astype(int))

    # ---------- image processing ----------
    @staticmethod
    def normalise(bgr):
        if bgr is None or bgr.ndim != 3:
            raise ValueError("not a colour image")
        if bgr.shape[:2] != (480, 640):
            bgr = cv2.resize(bgr, (640, 480), interpolation=cv2.INTER_AREA)
        return bgr

    def arm_area_px(self, mask):
        x0, y0, x1, y1 = self.region
        return largest_blob(mask[y0:y1, x0:x1])

    def quality(self, bgr):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        h = self.crop_px // 2
        bright, sharp = [], []
        for u, v in self.pts:
            p = gray[max(0, int(v) - h):int(v) + h, max(0, int(u) - h):int(u) + h]
            bright.append(float(p.mean()))
            sharp.append(float(cv2.Laplacian(p, cv2.CV_64F).var()))
        b, s = float(np.median(bright)), float(np.median(sharp))
        warnings = []
        lo, hi = QUALITY_LIMITS["brightness"]
        if b < lo:
            warnings.append(f"image darker than the station camera normally gives (brightness {b:.0f})")
        if b > hi:
            warnings.append(f"image brighter than the station camera normally gives (brightness {b:.0f})")
        if s < QUALITY_LIMITS["sharpness_min"]:
            warnings.append(f"image looks blurred (sharpness {s:.0f})")
        return {"brightness": round(b, 1), "sharpness": round(s, 1), "warnings": warnings}

    def fixture_shift(self, mask, search=18, rounds=4):
        """Median offset (dx, dy) of the yellow caps from where they normally sit, or None if
        fewer than 5 caps are visible. Windows re-centre on the estimate each round, so shifts
        larger than the window are still measured."""
        d = np.zeros(2)
        used = 0
        for _ in range(rounds):
            offs = []
            for u, v in self.pts + NORMAL_CAP_OFFSET + d:
                x0, y0 = int(round(u)) - search, int(round(v)) - search
                win = mask[max(0, y0):y0 + 2 * search + 1, max(0, x0):x0 + 2 * search + 1]
                ys, xs = np.nonzero(win)
                if len(xs) >= 40:
                    offs.append((xs.mean() + max(0, x0) - u, ys.mean() + max(0, y0) - v))
            if len(offs) < 5:
                return None
            step = np.median(np.array(offs), axis=0)
            d, used = d + step, len(offs)
            if np.hypot(*step) < 0.5:
                break
        return float(d[0]), float(d[1]), used

    # ---------- decision logic ----------
    @staticmethod
    def decide(probs, threshold, fill_count=None):
        """Per-position status, fill pattern and verdict from the model's probabilities.

        status: pass | defect | empty | missed (empty inside the fill pattern) | unexpected
        (a vial past a manually entered fill count).
        """
        rows = []
        for i, p in enumerate(probs):
            score = float(p[DEFECT_IDX].sum())
            top = CLASSES[int(np.argmax(p))]
            flagged = score >= threshold
            if flagged:
                status = "defect"
                kind = CLASSES[DEFECT_IDX[int(np.argmax(p[DEFECT_IDX]))]]
            elif top == "no_vial":
                status, kind = "empty", "no_vial"
            else:
                status, kind = "pass", "pass"
            rows.append({"position": f"G{i + 1:02d}", "index": i + 1, "status": status, "label": kind,
                         "top_class": top, "defect_score": round(score, 4),
                         "probs": {c: round(float(v), 4) for c, v in zip(CLASSES, p)}})

        occupied = [r["index"] for r in rows if r["status"] != "empty"]
        auto = max(occupied) if occupied else 0
        manual = fill_count is not None
        fill = int(fill_count) if manual else auto
        for r in rows:
            if r["index"] <= fill and r["status"] == "empty":
                r["status"] = "missed"
            elif manual and r["index"] > fill and r["status"] != "empty":
                r["status"] = "unexpected"

        n = {k: sum(r["status"] == k for r in rows) for k in ("pass", "defect", "empty", "missed", "unexpected")}
        if n["defect"] or n["missed"]:
            verdict = "REJECT"
        elif n["unexpected"]:
            verdict = "CHECK"
        else:
            verdict = "PASS"
        return {"positions": rows, "counts": n, "verdict": verdict,
                "fill": {"count": fill, "mode": "manual" if manual else "auto", "auto_count": auto}}

    def run(self, bgr, threshold=None, fill_count=None):
        bgr = self.normalise(bgr)
        thr = self.predictor.threshold if threshold is None else float(threshold)
        mask = yellow_mask(bgr)
        arm = self.arm_area_px(mask)
        quality = self.quality(bgr)
        base = {"threshold": thr, "quality": quality, "model": self.predictor.name,
                "demo": getattr(self.predictor, "demo", False)}
        if arm > self.arm_area:
            return {**base, "verdict": "SKIPPED", "reason": "robot arm over the fixture", "positions": [],
                    "counts": {}, "fill": None, "alignment": None, "probs": None}

        pts, alignment = self.pts, {"applied": False, "dx": 0.0, "dy": 0.0, "caps_used": 0}
        if self.align:
            s = self.fixture_shift(mask)
            if s:
                dx, dy, used = s
                alignment.update(dx=round(dx, 1), dy=round(dy, 1), caps_used=used)
                dist = float(np.hypot(dx, dy))
                if dist > self.max_shift:
                    quality["warnings"].append(f"fixture looks moved by {dist:.0f} px; check the camera and tray")
                elif dist >= self.min_shift:
                    pts = self.pts + np.array([dx, dy])
                    alignment["applied"] = True
        probs = np.asarray(self.predictor.predict_probs(bgr, pts))
        out = {**base, **self.decide(probs, thr, fill_count), "alignment": alignment,
               "probs": probs.round(5).tolist()}
        if quality["warnings"] and out["verdict"] == "PASS":
            out["verdict"] = "CHECK"
        out["crop_centres"] = [[round(float(u), 1), round(float(v), 1)] for u, v in pts]
        return out

    def rescore(self, result, threshold, fill_count=None):
        """Re-apply the threshold and fill count to a stored result without re-running the model."""
        if not result.get("probs"):
            return result
        out = {**result, **self.decide(np.array(result["probs"]), float(threshold), fill_count),
               "threshold": float(threshold)}
        if result["quality"]["warnings"] and out["verdict"] == "PASS":
            out["verdict"] = "CHECK"
        return out

    def annotate(self, bgr, result):
        img = self.normalise(bgr).copy()
        colours = {"pass": (70, 190, 70), "empty": (150, 150, 150), "defect": (40, 40, 225),
                   "missed": (0, 140, 255), "unexpected": (220, 120, 0)}
        if result["verdict"] == "SKIPPED":
            cv2.putText(img, "SKIPPED: " + result["reason"], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 200, 255), 2)
            return img
        h = 15   # smaller than the crop so neighbouring boxes don't overlap
        for r, (u, v) in zip(result["positions"], result["crop_centres"]):
            c = colours[r["status"]]
            bad = r["status"] in ("defect", "missed")
            cv2.rectangle(img, (int(u - h), int(v - h)), (int(u + h), int(v + h)), c, 2 if bad else 1)
            if r["status"] != "pass":
                cv2.putText(img, r["position"], (int(u - h), int(v - h) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, c, 1)
        cv2.putText(img, result["verdict"], (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (40, 40, 225) if result["verdict"] == "REJECT" else (70, 190, 70), 2)
        return img


class ModelPredictor:
    """The trained ResNet18 via predict.Inspector (needs torch)."""
    demo = False

    def __init__(self, model_path, threshold=None):
        from predict import Inspector
        self.insp = Inspector(model_path, threshold=threshold)
        self.threshold = float(self.insp.threshold)
        self.name = os.path.relpath(model_path)

    def predict_probs(self, bgr, pts):
        return self.insp.predict_probs(bgr, pts)


class OrderOnlyPredictor:
    """No cap classifier: order tracking alone decides (empty / open / capped from the empty-tray
    comparison). Single-photo results are not meaningful in this mode; use Order tracking."""
    demo = False
    order_only = True
    name = "Order tracking only (no cap classifier)"
    threshold = 1.01   # never flags in single-photo view

    def predict_probs(self, bgr, pts):
        p = np.zeros((len(pts), 5))
        p[:, 0] = 1.0
        return p


class DemoPredictor:
    """NOT the model. A colour rule (yellow cap at the centre or not) that produces fake
    probabilities, so the interface can be built and demonstrated without torch or best.pt."""
    demo = True
    name = "DEMO (colour rule, not the model)"
    threshold = 0.5

    def predict_probs(self, bgr, pts):
        mask = yellow_mask(bgr)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        out = []
        for u, v in pts:
            yel = float((mask[int(v) - 7:int(v) + 8, int(u) - 7:int(u) + 8] > 0).mean())
            dark = float((gray[int(v) - 7:int(v) + 8, int(u) - 7:int(u) + 8] < 90).mean())
            if yel > 0.3:
                p = [0.9, 0.03, 0.04, 0.02, 0.01]
            elif dark > 0.3:
                p = [0.05, 0.1, 0.02, 0.03, 0.8]
            else:
                p = [0.1, 0.7, 0.05, 0.05, 0.1]
            out.append(p)
        return np.array(out)
