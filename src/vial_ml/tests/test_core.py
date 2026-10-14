"""Checks for the parts that don't need PyTorch. Run: python tests/test_core.py"""
import csv
import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from gallery import pick_cases, write_html  # noqa: E402
from labels import CLASSES, CLASS_TO_IDX, load_items, parse_position  # noqa: E402
from metrics import catch_all_threshold, defect_scores, evaluate, public, report_markdown  # noqa: E402

P, MC, CC, DV, NV = (CLASS_TO_IDX[c] for c in CLASSES)


def onehot_ish(label, conf):
    p = np.full(5, (1 - conf) / 4)
    p[label] = conf
    return p


def test_scores_and_threshold():
    probs = np.array([[0.7, 0.1, 0.1, 0.05, 0.05], [0.1, 0.2, 0.3, 0.3, 0.1]])
    assert np.allclose(defect_scores(probs), [0.25, 0.8])
    assert catch_all_threshold([P, CC], defect_scores(probs)) == 0.8
    assert catch_all_threshold([P, NV], defect_scores(probs)) is None


def test_evaluate_auto_catches_every_defect():
    y = [P, P, P, P, MC, CC, DV, NV]
    probs = np.array([
        onehot_ish(P, 0.95), onehot_ish(P, 0.9), onehot_ish(P, 0.6), onehot_ish(P, 0.97),
        onehot_ish(MC, 0.9), onehot_ish(P, 0.5),  # crooked cap the model thinks is pass: hard case
        onehot_ish(DV, 0.8), onehot_ish(NV, 0.9),
    ])
    r = evaluate(y, probs)
    assert r["defects"]["missed"] == 0 and r["defects"]["support"] == 3
    assert r["hardest_defect"]["class"] == "crooked_cap"
    # pass crop with conf 0.6 has defect score 0.3 < hard crooked score 0.375 → not flagged
    assert r["false_alarms"]["count"] == 0, r["false_alarms"]
    assert r["per_class"]["crooked_cap"]["recall"] == 0.0  # top class wrong, still caught by threshold
    assert sum(map(sum, r["confusion_matrix"]["rows_true_cols_pred"])) == len(y)


def test_frozen_threshold_reports_escapes_and_alarms():
    y = [P, P, MC, NV, MC]
    probs = np.array([
        onehot_ish(P, 0.4),           # defect score 0.45 → false alarm at 0.4
        onehot_ish(P, 0.9),
        onehot_ish(P, 0.8),           # missing cap scored 0.15 → escape
        onehot_ish(MC, 0.7),          # empty hole called open vial
        onehot_ish(NV, 0.8),          # open vial called empty hole → escape
    ])
    r = evaluate(y, probs, threshold=0.4)
    assert r["threshold_source"] == "frozen"
    assert r["defects"]["missed"] == 2
    assert r["defects"]["by_class"]["missing_cap"]["missed"] == 2
    assert r["false_alarms"]["count"] == 1 and r["false_alarms"]["pass_support"] == 2
    m = r["missing_cap_vs_no_vial"]
    assert m["missing_cap_predicted_no_vial"] == 1 and m["no_vial_predicted_missing_cap"] == 1
    md = report_markdown(r)
    assert "Missed defects: 2 of 2" in md
    import json
    json.dumps(public(r))  # must be JSON-safe


def test_parse_position():
    assert parse_position("/x/G_14_crooked_cap_0003.png") == ("G", 14)
    assert parse_position("img0042_S-3.png") == ("S", 3)
    assert parse_position("G07.jpg") == ("G", 7)
    assert parse_position("crop_0001.png") == (None, None)


def make_images(root, n_per_class, scenes=4):
    for c in CLASSES:
        os.makedirs(os.path.join(root, c), exist_ok=True)
        for k in range(n_per_class):
            Image.new("RGB", (64, 64), (k * 20 % 255, 80, 120)).save(
                os.path.join(root, c, f"scene{k % scenes:02d}_G_{k + 1}_{c}.png"))


def test_split_real_and_gallery():
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "real")
        make_images(src, 6)
        out = os.path.join(tmp, "splits")
        cmd = [sys.executable, os.path.join(ROOT, "split_real.py"), "--src", src, "--out", out,
               "--group-regex", r"^(scene\d+)_"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0, res.stderr
        val, test = load_items(os.path.join(out, "real_val.csv")), load_items(os.path.join(out, "real_test.csv"))
        assert len(val) + len(test) == 30
        scene = lambda p: os.path.basename(p).split("_")[0]  # noqa: E731
        assert not ({scene(p) for p, _ in val} & {scene(p) for p, _ in test}), "scene leaked across splits"
        # Frozen: a second run must refuse.
        assert subprocess.run(cmd, capture_output=True, text=True).returncode != 0

        # Gallery on fake predictions for the test list.
        rng = np.random.default_rng(0)
        probs = rng.dirichlet(np.ones(5), size=len(test))
        r = evaluate([lab for _, lab in test], probs, threshold=0.5)
        sections = pick_cases(test, r)
        path = write_html(tmp, test, r, probs, sections)
        html_text = open(path).read()
        assert "Missed defects (escapes)" in html_text and "<img" in html_text or r["defects"]["missed"] == 0

        # Plain stratified split also works.
        out2 = os.path.join(tmp, "splits2")
        res = subprocess.run([sys.executable, os.path.join(ROOT, "split_real.py"), "--src", src, "--out", out2],
                             capture_output=True, text=True)
        assert res.returncode == 0, res.stderr
        with open(os.path.join(out2, "real_test.csv")) as f:
            labels = {row["label"] for row in csv.DictReader(f)}
        assert labels == set(CLASSES)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} passed")
