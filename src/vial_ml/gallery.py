"""Image galleries for the cases that matter. No PyTorch needed."""
import html
import os

from labels import CLASSES, CLASS_TO_IDX, NO_VIAL_IDX, PASS_IDX, parse_position


def pick_cases(items, result, max_false_alarms=40, max_correct=12):
    """Indices for each gallery section, most important first."""
    y = [lab for _, lab in items]
    pred, scores, flagged = result["_pred"], result["_scores"], result["_flagged"]
    defect = {CLASS_TO_IDX[c] for c in ("missing_cap", "crooked_cap", "damaged_vial")}
    mc = CLASS_TO_IDX["missing_cap"]

    missed = sorted((i for i in range(len(y)) if y[i] in defect and not flagged[i]), key=lambda i: scores[i])
    false_alarms = sorted((i for i in range(len(y)) if y[i] == PASS_IDX and flagged[i]),
                          key=lambda i: -scores[i])[:max_false_alarms]
    open_vs_empty = [i for i in range(len(y))
                     if (y[i] == mc and pred[i] == NO_VIAL_IDX) or (y[i] == NO_VIAL_IDX and pred[i] == mc)]
    caught = sorted((i for i in range(len(y)) if y[i] in defect and flagged[i]), key=lambda i: scores[i])[:max_correct]
    return {
        "Missed defects (escapes)": missed,
        "False alarms (pass crops flagged), highest score first": false_alarms,
        "Open vial vs empty hole confusions": open_vs_empty,
        "Caught defects, lowest score first (closest calls)": caught,
    }


def write_html(out_dir, items, result, probs, sections, cams=None, title="Evaluation gallery"):
    """cams: optional {item index: path to Grad-CAM image}."""
    cams = cams or {}
    thr = result["threshold"]

    def card(i):
        path, lab = items[i]
        fixture, pos = parse_position(path)
        where = f"{fixture} {pos}" if fixture else os.path.basename(path)
        rel = os.path.relpath(os.path.abspath(path), os.path.abspath(out_dir))
        top = sorted(range(len(CLASSES)), key=lambda k: -probs[i][k])[:2]
        top_txt = ", ".join(f"{CLASSES[k]} {probs[i][k]:.2f}" for k in top)
        cam = ""
        if i in cams:
            cam_rel = os.path.relpath(os.path.abspath(cams[i]), os.path.abspath(out_dir))
            cam = f'<img src="{html.escape(cam_rel)}" alt="Grad-CAM">'
        return (
            f'<figure><div class="pics"><img src="{html.escape(rel)}" alt="crop">{cam}</div>'
            f"<figcaption><b>{html.escape(where)}</b><br>true <b>{CLASSES[lab]}</b> · "
            f"pred {CLASSES[result['_pred'][i]]}<br>defect score {result['_scores'][i]:.3f}<br>"
            f'<span class="q">{html.escape(top_txt)}</span><br>'
            f'<span class="q">{html.escape(os.path.basename(path))}</span></figcaption></figure>'
        )

    parts = [
        "<!doctype html><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<title>{html.escape(title)}</title>",
        "<style>body{font:14px system-ui,sans-serif;margin:16px;color:#1d1d1f;background:#fff}"
        "h2{margin-top:28px}.grid{display:flex;flex-wrap:wrap;gap:12px}"
        "figure{margin:0;border:1px solid #ddd;border-radius:8px;padding:8px;width:296px}"
        ".pics{display:flex;gap:4px}.pics img{width:144px;height:144px;object-fit:contain;background:#f4f4f4}"
        "figcaption{margin-top:6px;line-height:1.4}.q{color:#666;font-size:12px}"
        "@media(prefers-color-scheme:dark){body{background:#151515;color:#eee}figure{border-color:#333}"
        ".pics img{background:#222}.q{color:#aaa}}</style>",
        f"<h1>{html.escape(title)}</h1>",
        f"<p>Threshold on defect score: <b>{'n/a' if thr is None else f'{thr:.4f}'}</b> "
        f"({html.escape(result['threshold_source'])}). Missed defects: <b>{result['defects']['missed']}</b> of "
        f"{result['defects']['support']}. False alarms: <b>{result['false_alarms']['count']}</b> of "
        f"{result['false_alarms']['pass_support']} pass crops. Right-hand image = Grad-CAM for the defect score.</p>",
    ]
    for name, idxs in sections.items():
        parts.append(f"<h2>{html.escape(name)} ({len(idxs)})</h2>")
        parts.append('<div class="grid">' + "".join(card(i) for i in idxs) + "</div>" if idxs else "<p>None.</p>")
    path = os.path.join(out_dir, "gallery.html")
    with open(path, "w") as f:
        f.write("\n".join(parts))
    return path
