"""Placement-by-placement inspection: watch the good fixture fill and check each placement as it happens.

At the start of an order the tray is empty; that photo becomes the reference for all 30 holes.
Every later photo (with the robot arm out of view) is compared with it, hole by hole:

    change from the empty reference          what it means
    ---------------------------------        -----------------------------------
    small   (< occupied_min)                 still empty
    medium  (occupied_min .. capped_min)     a vial with an OPEN mouth (missing cap)
    large   (>= capped_min)                  a capped vial -> the model checks the cap/vial

Because each hole is compared with itself, lighting, plate colour and glare cancel out, so an
empty hole cannot be confused with a transparent open vial (the hardest case for a single-photo model).

The robot fills G01..G05 (rightmost column, top to bottom), then the next column left, so:
    new vial at the expected position      -> PLACED, then inspected
    new vial further along                 -> the skipped positions are MISSED placements
    expected vial never appears after a    -> MISSED PICK
      "robot placed" signal
    an inspected position changes again    -> DISTURBED (cap moved, vial knocked) or REMOVED
    "order complete" with holes left       -> MISSING final placements

Thresholds were measured on the real validation positions only (empty <= 11.7, open 37-43,
capped 64-86 with a 10 px window); the held-out test positions were not used.
"""
import re
import time
from datetime import datetime

import cv2
import numpy as np

from cut_real_crops import yellow_mask

_TS = re.compile(r"(\d{8})_(\d{6})_(\d{6})")


def photo_time(name):
    """Capture time from a station file name like cap_seal_20260923_200828_882680.jpg (epoch seconds),
    or None when the name carries no timestamp."""
    m = _TS.search(name or "")
    if not m:
        return None
    try:
        return datetime.strptime("".join(m.groups()), "%Y%m%d%H%M%S%f").timestamp()
    except ValueError:
        return None


def fmt_time(ts):
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S") if ts else ""


CLASSES = ["pass", "missing_cap", "crooked_cap", "damaged_vial", "no_vial"]

DEFAULTS = {
    "window": 10,          # half-size (px) of the patch compared at each hole
    "occupied_min": 20.0,  # change from the empty tray that means "a vial is here"
    "capped_min": 55.0,    # change that means "a capped vial" (below it: open mouth)
    "disturb_min": 20.0,   # change of an inspected position since it was accepted
    "cap_defect_thr": 0.5, # model: P(crooked)+P(damaged) among the vial classes, to flag a capped vial
    "stable_max": 8.0,
    # The twin-only model calls most REAL good vials crooked/damaged (P(pass) is near 0 on real photos),
    # so by default its opinion only asks for a check by eye. Set True once a model is validated on real data.
    "model_can_reject": False,
    "open_max_yellow": 0.35,  # open vials on the validation positions: <= 0.17 cap colour at the centre     # a hole counts as settled when it changed less than this since the last photo
}


def gray(bgr):
    return cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (3, 3), 0).astype(np.float32)


def patch_change(a, b, pts, r):
    """Mean absolute difference per hole, after matching each patch's mean brightness."""
    out = np.empty(len(pts))
    for i, (u, v) in enumerate(pts):
        x = a[int(v) - r:int(v) + r, int(u) - r:int(u) + r]
        y = b[int(v) - r:int(v) + r, int(u) - r:int(u) + r]
        out[i] = np.abs((x - x.mean()) - (y - y.mean())).mean()
    return out


class OrderTracker:
    def __init__(self, names, pts, settings=None):
        self.names, self.pts = names, np.asarray(pts, float)
        self.s = {**DEFAULTS, **(settings or {})}
        self.reset()

    # ---------- state ----------
    def reset(self):
        self.active = False
        self.ref = None
        self.ref_name = None
        self.state = ["pending"] * 30      # pending | ok | check | open | flagged | missed | disturbed | removed | missing
        self.detail = [{} for _ in range(30)]
        self.accepted = [None] * 30        # grey image at the moment each position was inspected
        self.next = 1                      # next position the robot should fill
        self.events = []
        self.awaiting_placement = None     # set by a "robot placed" signal
        self.complete = False
        self.frames = 0
        self.prev = None                   # previous photo (grey), for the "settled" check
        self.t0 = None                     # capture time of the empty-tray reference
        self.cur_ts = None                 # capture time of the photo being processed
        self.ref_bgr = None                # colour reference, for the order record
        self.first_seen = [None] * 30      # first photo time a vial appeared in each hole
        self.crops = [None] * 30           # colour crop of each vial when it was inspected
        self.event_crops = {}              # (position, kind) -> colour crop at the moment of a later event

    def log(self, kind, pos, text, frame=None):
        self.events.append({"t": fmt_time(self.cur_ts) or time.strftime("%H:%M:%S"), "ts": self.cur_ts, "kind": kind,
                            "position": f"G{pos:02d}" if pos else "", "text": text, "frame": frame})

    def crop(self, bgr, i, r=30):
        u, v = self.pts[i]
        y0, x0 = max(0, int(v) - r), max(0, int(u) - r)
        return bgr[y0:int(v) + r, x0:int(u) + r].copy()

    def start(self, bgr, name, ts=None):
        """Begin an order with this photo of the empty tray as the reference."""
        self.reset()
        self.cur_ts = self.t0 = ts or photo_time(name) or time.time()
        self.ref_bgr = bgr.copy()
        g = gray(bgr)
        caps = self._caps_visible(bgr)
        self.ref, self.ref_name, self.active, self.prev = g, name, True, g
        self.log("start", None, f"Order started. Empty-tray reference: {name}", name)
        if caps:
            self.log("warn", None, f"The reference may not be empty: caps seen at "
                                   f"{', '.join('G%02d' % p for p in caps)}. Start again with an empty tray.", name)
        return self.summary()

    def _caps_visible(self, bgr):
        m = yellow_mask(bgr)
        out = []
        for i, (u, v) in enumerate(self.pts):
            p = m[int(v - 4.3) - 5:int(v - 4.3) + 6, int(u) - 5:int(u) + 6]
            if p.size and (p > 0).mean() > 0.5:
                out.append(i + 1)
        return out

    # ---------- per frame ----------
    def update(self, bgr, name, probs=None, ts=None):
        """Process one arm-free photo. probs: the model's 30x5 probabilities for this photo (optional).
        ts: capture time (epoch s); default from the file name, else now."""
        if not self.active:
            return None
        self.frames += 1
        self.cur_ts = ts or photo_time(name) or time.time()
        g = gray(bgr)
        r = self.s["window"]
        change = patch_change(g, self.ref, self.pts, r)
        occupied = change >= self.s["occupied_min"]
        # Only act on holes that look the same as in the previous photo: the arm, its cable and a
        # vial still settling all move between photos, a placed vial doesn't.
        settled = patch_change(g, self.prev, self.pts, r) < self.s["stable_max"]
        self._unsettled = [n for n, ok in zip(self.names, settled) if not ok]
        self.prev = g
        # placement time = the first photo in which a vial shows up in that hole
        for i in range(30):
            if self.state[i] in ("pending", "missed", "missing"):
                if occupied[i] and self.first_seen[i] is None:
                    self.first_seen[i] = self.cur_ts
                elif not occupied[i]:
                    self.first_seen[i] = None

        # 1. positions already inspected: did anything move or disappear?
        for i in range(30):
            if self.state[i] in ("ok", "check", "open", "flagged") and self.accepted[i] is not None and settled[i]:
                if not occupied[i]:
                    self.state[i] = "removed"
                    self.event_crops[(i, "removed")] = self.crop(bgr, i)
                    self.detail[i]["event_at"] = self.cur_ts
                    self.log("defect", i + 1, "vial gone after it was placed", name)
                    continue
                d = patch_change(g, self.accepted[i], self.pts[i:i + 1], r)[0]
                if d >= self.s["disturb_min"]:
                    self.state[i] = "disturbed"
                    self.detail[i]["disturb_change"] = round(float(d), 1)
                    self.detail[i]["event_at"] = self.cur_ts
                    self.event_crops[(i, "disturbed")] = self.crop(bgr, i)
                    self.log("defect", i + 1, f"changed after inspection (cap moved or vial knocked; change {d:.0f})",
                             name)

        # 2. new vials, in fill order
        new = [i + 1 for i in range(30)
               if occupied[i] and settled[i] and self.state[i] in ("pending", "missed", "missing")]
        for pos in sorted(new):
            i = pos - 1
            if self.state[i] in ("missed", "missing"):
                self.log("info", pos, "late placement into a hole that had been skipped", name)
            elif pos > self.next:
                for skipped in range(self.next, pos):
                    # only if that hole really looks empty (an open vial can flicker between photos)
                    if self.state[skipped - 1] == "pending" and not occupied[skipped - 1]:
                        self.state[skipped - 1] = "missed"
                        self.log("defect", skipped, f"skipped: the robot placed G{pos:02d} next", name)
            self._inspect(i, change[i], g, name, probs, self._yellow_at(bgr, i))
            self.crops[i] = self.crop(bgr, i)
            self.next = max(self.next, pos + 1)
            self.awaiting_placement = None

        # 3. a placement was signalled but nothing arrived
        if self.awaiting_placement and not new and settled[self.awaiting_placement - 1]:
            pos = self.awaiting_placement
            self.state[pos - 1] = "missed"
            self.log("defect", pos, "missed pick: the robot reported a placement but the hole is still empty", name)
            self.awaiting_placement = None
            self.next = max(self.next, pos + 1)

        if all(s != "pending" for s in self.state) and not self.complete:
            self.complete = True
            self.log("info", None, "all 30 positions filled", name)
        return self.summary(change)

    def _yellow_at(self, bgr, i):
        """Share of cap-coloured pixels at the hole's centre (where a seated cap sits)."""
        u, v = self.pts[i]
        p = yellow_mask(bgr)[int(v - 4.3) - 5:int(v - 4.3) + 6, int(u) - 5:int(u) + 6]
        return float((p > 0).mean()) if p.size else 0.0

    def _inspect(self, i, c, g, name, probs, yellow):
        pos = i + 1
        self.accepted[i] = g
        det = {"change": round(float(c), 1), "cap_colour_at_centre": round(yellow, 2), "frame": name,
               "placed_at": self.first_seen[i] or self.cur_ts, "inspected_at": self.cur_ts}
        # Open mouth: a smaller change than a capped vial AND no cap colour at the centre.
        if c < self.s["capped_min"] and yellow < self.s["open_max_yellow"]:
            self.state[i] = "open"
            det["label"] = "missing_cap"
            self.log("defect", pos, f"placed with an OPEN mouth (missing cap; change {c:.0f})", name)
        else:
            label, score = "pass", None
            if probs is not None:
                p = np.asarray(probs[i], float)[:4]          # ignore no_vial: we know a vial is here
                p = p / max(p.sum(), 1e-9)
                score = float(p[2] + p[3])
                det["model"] = {c_: round(float(v), 4) for c_, v in zip(CLASSES[:4], p)}
                det["cap_defect_score"] = round(score, 4)
                if score >= self.s["cap_defect_thr"]:
                    label = CLASSES[2 + int(np.argmax(p[2:4]))]
            det["label"] = label
            if label == "pass":
                self.state[i] = "ok"
                self.log("ok", pos, "placed, cap present" + (f" (model cap-defect score {score:.2f})"
                                                            if score is not None else ""), name)
            elif self.s["model_can_reject"]:
                self.state[i] = "flagged"
                self.log("defect", pos, f"placed, model flags {label.replace('_', ' ')} (score {score:.2f})", name)
            else:
                self.state[i] = "check"
                self.log("warn", pos, f"placed, cap present; model suggests {label.replace('_', ' ')} "
                                      f"(score {score:.2f}, not validated on real photos): check by eye", name)
        self.detail[i] = det

    # ---------- signals ----------
    def robot_placed(self):
        """The robot reports it just placed a vial (at self.next). Checked on the next photo."""
        if self.active and self.next <= 30:
            self.awaiting_placement = self.next
            self.log("signal", self.next, "robot reports a placement", None)
        return self.summary()

    def order_complete(self, count=None):
        """Order-complete signal: everything up to `count` (default 30) should be filled."""
        count = 30 if count is None else int(count)
        for i in range(count):
            if self.state[i] == "pending":
                self.state[i] = "missing"
                self.log("defect", i + 1, "missing at order complete", None)
        self.complete = True
        self.log("info", None, f"order complete signal ({count} vials expected)", None)
        return self.summary()

    # ---------- analytics ----------
    def cycle_stats(self):
        """Time between consecutive placements (seconds), in the order vials arrived."""
        placed = sorted((d["placed_at"], i + 1) for i, d in enumerate(self.detail) if d.get("placed_at"))
        rows, prev = [], self.t0
        for t, pos in placed:
            rows.append({"position": f"G{pos:02d}", "placed_at": t, "time": fmt_time(t),
                         "gap_s": round(t - prev, 1) if prev else None})
            prev = t
        gaps = [r["gap_s"] for r in rows[1:] if r["gap_s"] is not None]   # the first gap includes setup
        if not gaps:
            return {"placements": rows, "median_s": None, "slow": []}
        med = float(np.median(gaps))
        slow_cut = max(2 * med, med + 60)
        for r in rows[1:]:
            r["slow"] = r["gap_s"] is not None and r["gap_s"] > slow_cut
        span = placed[-1][0] - placed[0][0]
        return {"placements": rows, "median_s": round(med, 1), "mean_s": round(float(np.mean(gaps)), 1),
                "slow_cut_s": round(slow_cut, 1), "slow": [r["position"] for r in rows if r.get("slow")],
                "first_to_last_s": round(span, 1),
                "per_hour": round(3600 * len(gaps) / span, 1) if span > 0 else None}

    # ---------- output ----------
    def summary(self, change=None):
        bad = [s for s in self.state if s in ("open", "flagged", "missed", "disturbed", "removed", "missing")]
        filled = sum(s not in ("pending", "missed", "missing") for s in self.state)
        if not self.active:
            verdict = "IDLE"
        elif bad:
            verdict = "REJECT"
        elif "check" in self.state:
            verdict = "CHECK"
        elif self.complete:
            verdict = "PASS"
        else:
            verdict = "IN PROGRESS"
        return {
            "active": self.active, "verdict": verdict, "reference": self.ref_name, "frames": self.frames,
            "next": f"G{self.next:02d}" if self.next <= 30 else None, "filled": filled,
            "complete": self.complete, "awaiting_placement": self.awaiting_placement,
            "positions": [{"position": n, "state": s, **d} for n, s, d in zip(self.names, self.state, self.detail)],
            "change": None if change is None else [round(float(c), 1) for c in change],
            "unsettled": self._unsettled if change is not None else [],
            "events": self.events[-200:], "settings": self.s,
            "started_at": self.t0, "cycle": self.cycle_stats() if self.active else None,
        }
