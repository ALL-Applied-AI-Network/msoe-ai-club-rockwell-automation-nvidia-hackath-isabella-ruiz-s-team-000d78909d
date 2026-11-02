"""Decision logic of the inspection app."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app"))
from inspection import Pipeline  # noqa: E402

PASS = [0.9, 0.03, 0.03, 0.02, 0.02]
EMPTY = [0.02, 0.03, 0.02, 0.01, 0.92]
MISSING = [0.05, 0.9, 0.02, 0.01, 0.02]


def probs(rows):
    return np.array(rows)


def test_full_good_tray_passes():
    r = Pipeline.decide(probs([PASS] * 30), 0.5)
    assert r["verdict"] == "PASS" and r["fill"]["count"] == 30


def test_partial_fill_auto():
    r = Pipeline.decide(probs([PASS] * 12 + [EMPTY] * 18), 0.5)
    assert r["verdict"] == "PASS" and r["fill"]["count"] == 12 and r["counts"]["empty"] == 18


def test_hole_inside_fill_is_missed():
    rows = [PASS] * 12 + [EMPTY] * 18
    rows[4] = EMPTY
    r = Pipeline.decide(probs(rows), 0.5)
    assert r["verdict"] == "REJECT" and r["positions"][4]["status"] == "missed"


def test_defect_and_threshold():
    rows = [PASS] * 30
    rows[7] = MISSING
    assert Pipeline.decide(probs(rows), 0.5)["positions"][7]["label"] == "missing_cap"
    assert Pipeline.decide(probs(rows), 0.95)["verdict"] == "PASS"


def test_manual_fill():
    rows = [PASS] * 12 + [EMPTY] * 18
    r = Pipeline.decide(probs(rows), 0.5, fill_count=10)
    assert r["verdict"] == "CHECK" and r["counts"]["unexpected"] == 2
    r = Pipeline.decide(probs(rows), 0.5, fill_count=15)
    assert r["verdict"] == "REJECT" and r["counts"]["missed"] == 3


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
