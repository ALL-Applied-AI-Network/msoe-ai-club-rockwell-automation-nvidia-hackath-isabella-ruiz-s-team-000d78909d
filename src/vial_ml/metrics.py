"""Recall-first metrics for the vial classifier.

Decision rule: a crop is flagged as a defect when its defect score
    P(missing_cap) + P(crooked_cap) + P(damaged_vial)
is at or above a threshold. no_vial is not a defect here; whether an empty
position is an error is decided by the fill-pattern logic.

The headline number is the false-alarm rate (pass crops flagged) at the
threshold that catches every known defect.
"""
import numpy as np

from labels import CLASSES, DEFECT_IDX, NO_VIAL_IDX, PASS_IDX, CLASS_TO_IDX


def defect_scores(probs):
    probs = np.asarray(probs, dtype=float)
    return probs[:, DEFECT_IDX].sum(axis=1)


def catch_all_threshold(y_true, scores):
    """Highest threshold that still flags every defect crop (None if no defects)."""
    y_true = np.asarray(y_true)
    is_def = np.isin(y_true, DEFECT_IDX)
    if not is_def.any():
        return None
    return float(np.asarray(scores)[is_def].min())


def evaluate(y_true, probs, threshold=None):
    """All numbers for one set of crops.

    threshold=None picks the catch-all threshold on this set (use for
    validation). Pass a frozen float for the test set.
    """
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs, dtype=float)
    n = len(y_true)
    k = len(CLASSES)
    pred = probs.argmax(axis=1)
    scores = defect_scores(probs)

    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, pred):
        cm[t, p] += 1

    per_class = {}
    for i, c in enumerate(CLASSES):
        support = int(cm[i].sum())
        predicted = int(cm[:, i].sum())
        tp = int(cm[i, i])
        per_class[c] = {
            "support": support,
            "precision": tp / predicted if predicted else None,
            "recall": tp / support if support else None,
        }

    auto = threshold is None
    thr = catch_all_threshold(y_true, scores) if auto else float(threshold)

    is_def = np.isin(y_true, DEFECT_IDX)
    is_pass = y_true == PASS_IDX
    flagged = scores >= thr if thr is not None else np.zeros(n, dtype=bool)

    missed_by_class = {}
    for c in ("missing_cap", "crooked_cap", "damaged_vial"):
        m = y_true == CLASS_TO_IDX[c]
        missed_by_class[c] = {
            "support": int(m.sum()),
            "missed": int((m & ~flagged).sum()),
            "recall": float((m & flagged).sum() / m.sum()) if m.any() else None,
        }

    n_pass = int(is_pass.sum())
    false_alarms = int((is_pass & flagged).sum())

    # The pair the lab warns about: open vial vs empty hole.
    mc, nv = CLASS_TO_IDX["missing_cap"], NO_VIAL_IDX

    # Which defect crop set the threshold (the hardest one to catch).
    hardest = None
    if auto and thr is not None:
        idx = np.where(is_def)[0]
        j = int(idx[np.argmin(scores[idx])])
        hardest = {"index": j, "class": CLASSES[y_true[j]], "score": float(scores[j])}

    return {
        "n": n,
        "accuracy": float((pred == y_true).mean()) if n else None,
        "threshold": thr,
        "threshold_source": "catch-all on this set" if auto else "frozen",
        "hardest_defect": hardest,
        "defects": {
            "support": int(is_def.sum()),
            "missed": int((is_def & ~flagged).sum()),
            "recall": float((is_def & flagged).sum() / is_def.sum()) if is_def.any() else None,
            "by_class": missed_by_class,
        },
        "false_alarms": {
            "pass_support": n_pass,
            "count": false_alarms,
            "rate": false_alarms / n_pass if n_pass else None,
        },
        "missing_cap_vs_no_vial": {
            "missing_cap_predicted_no_vial": int(cm[mc, nv]),
            "no_vial_predicted_missing_cap": int(cm[nv, mc]),
        },
        "per_class": per_class,
        "confusion_matrix": {"labels": CLASSES, "rows_true_cols_pred": cm.tolist()},
        "_flagged": flagged,
        "_pred": pred,
        "_scores": scores,
    }


def public(result):
    """Drop the per-crop arrays so the result can be saved as JSON."""
    return {k: v for k, v in result.items() if not k.startswith("_")}


def _pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def report_markdown(result, title="Evaluation"):
    r = result
    lines = [f"# {title}", ""]
    d, fa = r["defects"], r["false_alarms"]
    thr = "n/a" if r["threshold"] is None else f"{r['threshold']:.4f}"
    lines.append(
        f"**Missed defects: {d['missed']} of {d['support']}** · "
        f"**False alarms: {fa['count']} of {fa['pass_support']} pass crops ({_pct(fa['rate'])})** · "
        f"threshold {thr} ({r['threshold_source']})"
    )
    if r["hardest_defect"]:
        h = r["hardest_defect"]
        lines.append("")
        lines.append(f"The threshold was set by the hardest defect: a {h['class']} crop scoring {h['score']:.4f}.")
    lines += ["", "## Missed defects by class", "", "| Class | Crops | Missed | Recall |", "| --- | --- | --- | --- |"]
    for c, v in d["by_class"].items():
        lines.append(f"| {c} | {v['support']} | {v['missed']} | {_pct(v['recall'])} |")

    lines += ["", "## Per-class results (top predicted class)", "", "| Class | Crops | Precision | Recall |", "| --- | --- | --- | --- |"]
    for c, v in r["per_class"].items():
        lines.append(f"| {c} | {v['support']} | {_pct(v['precision'])} | {_pct(v['recall'])} |")

    m = r["missing_cap_vs_no_vial"]
    lines += [
        "",
        "## Open vial vs empty hole",
        "",
        f"- missing_cap predicted as no_vial: {m['missing_cap_predicted_no_vial']}",
        f"- no_vial predicted as missing_cap: {m['no_vial_predicted_missing_cap']}",
        "",
        "## Confusion matrix (rows = true, columns = predicted)",
        "",
        "| true \\ pred | " + " | ".join(CLASSES) + " |",
        "| --- | " + " | ".join("---" for _ in CLASSES) + " |",
    ]
    for c, row in zip(CLASSES, r["confusion_matrix"]["rows_true_cols_pred"]):
        lines.append(f"| {c} | " + " | ".join(str(x) for x in row) + " |")
    lines += ["", f"Overall accuracy (for reference only): {_pct(r['accuracy'])}", ""]
    return "\n".join(lines)
