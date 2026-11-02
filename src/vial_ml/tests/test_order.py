"""Order tracker on real cap_seal frames, with events created by editing the photos
(the real sequences contain no skipped holes, missed picks or knocked caps).
Needs the cap_seal images: set CAP_SEAL to the folder with image_index.csv."""
import collections
import csv
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "app"))
sys.path.insert(0, ROOT)
from inspection import Pipeline, DemoPredictor, yellow_mask  # noqa: E402
from order import OrderTracker  # noqa: E402

CAP_SEAL = os.environ.get("CAP_SEAL", "/mnt/user-data/uploads/cap_seal")
P = Pipeline(DemoPredictor())


def video5():
    idx = list(csv.DictReader(open(os.path.join(CAP_SEAL, "image_index.csv"))))
    have = set(os.listdir(os.path.join(CAP_SEAL, "images")))
    fs = sorted((r for r in idx if r["video_id"] == "5" and r["filename"] in have), key=lambda r: r["timestamp"])
    out = []
    for r in fs:
        im = cv2.imread(os.path.join(CAP_SEAL, "images", r["filename"]))
        if P.arm_area_px(yellow_mask(im)) <= 2500:
            out.append((r["filename"], im))
    return out


def paste(dst, src, pos, r=22):
    u, v = P.pts[pos - 1]
    y0, x0 = int(v) - r, int(u) - r
    dst = dst.copy()
    dst[y0:y0 + 2 * r, x0:x0 + 2 * r] = src[y0:y0 + 2 * r, x0:x0 + 2 * r]
    return dst


def run(frames, signal_at=None, complete=False):
    t = OrderTracker(P.names, P.pts)
    t.start(frames[0][1], frames[0][0])
    for k, (name, im) in enumerate(frames[1:], 1):
        if signal_at == k:
            t.robot_placed()
        t.update(im, name)
    if complete:
        t.order_complete()
    return t.summary()


def events(s, kind="defect"):
    return [(e["position"], e["text"].split(":")[0].split(" (")[0]) for e in s["events"] if e["kind"] == kind]


def test_real_order_matches_labels():
    s = run(video5())
    assert events(s) == [("G17", "placed with an OPEN mouth"), ("G20", "placed with an OPEN mouth")]
    assert s["filled"] == 22


def test_skipped_hole():
    fr = video5()
    ref = fr[0][1]
    # erase G10 from every photo: the robot "never placed" it
    fr = [(n, paste(im, ref, 10)) for n, im in fr]
    s = run(fr)
    assert ("G10", "skipped") in events(s)


def test_vial_removed_and_disturbed():
    fr = video5()
    ref, last = fr[0][1], fr[-1][1]
    # after the order, G05 disappears and G06's vial is pushed 6 px sideways (a knocked vial)
    im = paste(last, ref, 5)
    u, v = P.pts[5]
    box = (slice(int(v) - 22, int(v) + 22), slice(int(u) - 22, int(u) + 22))
    im[box] = np.roll(last[box], 6, axis=1)
    s = run(fr + [("knock1.jpg", im), ("knock2.jpg", im.copy())])   # seen twice -> settled
    ev = events(s)
    assert ("G05", "vial gone after it was placed") in ev
    assert any(p == "G06" and t.startswith("changed after inspection") for p, t in ev)


def test_missed_pick_signal():
    fr = video5()
    # after G22 the robot "reports a placement" but the next photos are unchanged
    fr = fr + [("same1.jpg", fr[-1][1].copy()), ("same2.jpg", fr[-1][1].copy())]
    s = run(fr, signal_at=len(fr) - 2)
    assert ("G23", "missed pick") in events(s)


def test_order_complete_marks_missing():
    s = run(video5(), complete=True)
    assert s["verdict"] == "REJECT" and sum(p["state"] == "missing" for p in s["positions"]) == 8




def test_model_opinion_is_only_a_check_by_default():
    fr = video5()
    t = OrderTracker(P.names, P.pts)
    t.start(fr[0][1], fr[0][0])
    crooked = np.tile([0.05, 0.05, 0.85, 0.05, 0.0], (30, 1))   # a model that calls every vial crooked
    for name, im in fr[1:]:
        t.update(im, name, crooked)
    s = t.summary()
    states = [p["state"] for p in s["positions"]]
    assert states.count("check") == 20 and states.count("flagged") == 0 and states.count("open") == 2
    assert s["verdict"] == "REJECT"          # still rejected, but only for the two real open vials

if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
