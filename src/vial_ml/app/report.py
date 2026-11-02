"""Order inspection record: one self-contained HTML page per order (print it or save it as PDF).

Holds, for the order being tracked:
  - verdict, timing summary and the organizers' tray findings,
  - the fixture map with a photo of every vial at the moment it was placed,
  - before/after evidence for every problem (empty-tray crop next to the crop at the event),
  - cycle time between placements, with unusually slow placements marked,
  - the full position table and event log, and the settings used.
"""
import base64
import html
from datetime import datetime

import cv2

STATE_TEXT = {
    "ok": "Placed OK", "check": "Check by eye", "open": "Open mouth (missing cap)", "flagged": "Flagged by model",
    "missed": "Skipped / missed pick", "missing": "Missing at completion", "disturbed": "Changed after inspection",
    "removed": "Vial gone", "pending": "Not filled",
}
STATE_COLOUR = {
    "ok": "#2E7D32", "check": "#1F5FBF", "open": "#C0392B", "flagged": "#C0392B", "disturbed": "#C0392B",
    "removed": "#C0392B", "missed": "#D35400", "missing": "#D35400", "pending": "#9AA0A6",
}
PROBLEM = ("open", "flagged", "disturbed", "removed", "missed", "missing")


def _img(bgr, size=120):
    if bgr is None or bgr.size == 0:
        return ""
    im = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else ""


def _t(ts, with_date=False):
    if not ts:
        return "—"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S" if with_date else "%H:%M:%S")


def _dur(s):
    if s is None:
        return "—"
    s = int(round(s))
    return f"{s // 60} min {s % 60:02d} s" if s >= 60 else f"{s} s"


def _chart(cycle):
    rows = [r for r in cycle["placements"][1:] if r["gap_s"] is not None]
    if not rows:
        return "<p class='muted'>Not enough placements yet for cycle times.</p>"
    W, H, pad_l, pad_b, pad_t = 900, 260, 56, 44, 16
    top = max(max(r["gap_s"] for r in rows), cycle.get("slow_cut_s") or 0) * 1.1
    bw = (W - pad_l - 10) / len(rows)
    y = lambda v: pad_t + (H - pad_t - pad_b) * (1 - v / top)  # noqa: E731
    parts = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Seconds between placements">']
    for k in range(0, int(top) + 1, max(30, int(top / 5) // 30 * 30 or 30)):
        parts.append(f'<line x1="{pad_l}" x2="{W}" y1="{y(k):.1f}" y2="{y(k):.1f}" stroke="#E3E3E3"/>'
                     f'<text x="{pad_l - 8}" y="{y(k) + 4:.1f}" text-anchor="end" font-size="12" fill="#666">{k}s</text>')
    for j, r in enumerate(rows):
        x = pad_l + j * bw
        col = "#D35400" if r.get("slow") else "#4A6FA5"
        parts.append(f'<rect x="{x + 2:.1f}" y="{y(r["gap_s"]):.1f}" width="{bw - 4:.1f}" '
                     f'height="{H - pad_b - y(r["gap_s"]):.1f}" fill="{col}"><title>{r["position"]}: '
                     f'{r["gap_s"]:.0f} s after the previous vial</title></rect>'
                     f'<text x="{x + bw / 2:.1f}" y="{H - pad_b + 16}" text-anchor="middle" font-size="11" '
                     f'fill="#444">{r["position"][1:]}</text>')
    if cycle.get("median_s"):
        my = y(cycle["median_s"])
        parts.append(f'<line x1="{pad_l}" x2="{W}" y1="{my:.1f}" y2="{my:.1f}" stroke="#1A1A1A" '
                     f'stroke-dasharray="5 4"/><text x="{pad_l + 6}" y="{my - 6:.1f}" font-size="13" font-weight="700" '
                     f'fill="#1A1A1A" stroke="#FFFFFF" stroke-width="4" paint-order="stroke">'
                     f'median {cycle["median_s"]:.0f} s</text>')
    parts.append(f'<text x="{(W + pad_l) / 2}" y="{H - 6}" text-anchor="middle" font-size="12" fill="#666">'
                 f'position (in the order placed)</text></svg>')
    return "".join(parts)


def build(tracker, findings=None, meta=None):
    meta = meta or {}
    s = tracker.summary()
    cyc = s["cycle"] or {"placements": []}
    st, det = tracker.state, tracker.detail
    n_bad = sum(x in PROBLEM for x in st)
    n_check = st.count("check")
    last_ts = max([e["ts"] for e in s["events"] if e.get("ts")] or [tracker.t0 or 0])
    verdict_col = {"REJECT": "#C0392B", "PASS": "#2E7D32", "CHECK": "#1F5FBF"}.get(s["verdict"], "#1A1A1A")
    esc = html.escape

    # fixture map in camera orientation: G01-G05 is the rightmost column, top to bottom
    cells = []
    for row in range(5):
        for col in range(6):
            i = (5 - col) * 5 + row
            state = st[i]
            img = _img(tracker.crops[i]) if tracker.crops[i] is not None else _img(tracker.crop(tracker.ref_bgr, i))
            placed = det[i].get("placed_at")
            cells.append(
                f'<div class="cell" style="border-color:{STATE_COLOUR[state]}">'
                f'<img src="{img}" alt="G{i + 1:02d}"><div class="cap"><b>G{i + 1:02d}</b> '
                f'<span style="color:{STATE_COLOUR[state]}">{STATE_TEXT[state]}</span><br>'
                f'<span class="muted">{_t(placed) if placed else ""}</span></div></div>')

    # evidence for every problem
    ev = []
    for i, state in enumerate(st):
        if state not in PROBLEM:
            continue
        before = _img(tracker.crop(tracker.ref_bgr, i), 150)
        after_crop = tracker.event_crops.get((i, state))
        if after_crop is None:
            after_crop = tracker.crops[i]
        after = _img(after_crop, 150) if after_crop is not None else ""
        when = det[i].get("event_at") or det[i].get("placed_at")
        why = [e["text"] for e in s["events"] if e["position"] == f"G{i + 1:02d}" and e["kind"] == "defect"]
        ev.append(
            f'<div class="ev"><div class="pair"><figure><img src="{before}" alt=""><figcaption>Empty tray'
            f'</figcaption></figure>' + (f'<figure><img src="{after}" alt=""><figcaption>{_t(when)}'
                                         f'</figcaption></figure>' if after else
                                         '<figure class="none"><figcaption>no vial</figcaption></figure>') +
            f'</div><div><h3>G{i + 1:02d} · <span style="color:{STATE_COLOUR[state]}">{STATE_TEXT[state]}</span>'
            f'</h3><p>{esc("; ".join(why) or STATE_TEXT[state])}</p>'
            f'<p class="muted">Change from empty tray: {det[i].get("change", "—")} · cap colour at centre: '
            f'{det[i].get("cap_colour_at_centre", "—")} · photo: {esc(str(det[i].get("frame", "—")))}</p></div></div>')

    # findings (organizers' rules.py)
    find_rows = ""
    if findings:
        yn = lambda v: "yes" if v is True else "no" if v is False else "not established"  # noqa: E731
        names = {"complete_order": "Complete good order", "missing_vial": "Missing vial (gap in the fill)",
                 "bad_misclassified_as_good": "Defective vial in the good fixture"}
        find_rows = "".join(f"<tr><td>{names.get(k, k)}</td><td><b>{yn(v)}</b></td></tr>"
                            for k, v in findings["findings"].items())

    # position table
    gaps = {r["position"]: r for r in cyc["placements"]}
    table = []
    for i, state in enumerate(st):
        d, name = det[i], f"G{i + 1:02d}"
        g = gaps.get(name, {})
        model = d.get("cap_defect_score")
        gap = "" if g.get("gap_s") is None else f"{g['gap_s']:.0f} s"
        table.append(
            f'<tr><td><b>{name}</b></td><td style="color:{STATE_COLOUR[state]}">{STATE_TEXT[state]}</td>'
            f'<td>{_t(d.get("placed_at"))}</td><td class="r">{gap}'
            f'{" ⚠ slow" if g.get("slow") else ""}</td><td class="r">{d.get("change", "")}</td>'
            f'<td class="r">{d.get("cap_colour_at_centre", "")}</td><td class="r">{"" if model is None else model}</td>'
            f'<td class="small">{esc(str(d.get("frame", "")))}</td></tr>')

    events = "".join(f'<tr><td>{e["t"]}</td><td>{e["position"]}</td><td>{esc(e["kind"])}</td>'
                     f'<td>{esc(e["text"])}</td></tr>' for e in s["events"])
    cfg = tracker.s
    slow = ", ".join(cyc.get("slow") or []) or "none"

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Order record {esc(str(s['reference']))}</title>
<style>
body{{margin:0;background:#fff;color:#1A1A1A;font:14px/1.45 Arial,Helvetica,sans-serif}}
main{{max-width:1000px;margin:0 auto;padding:28px 20px 60px}}
h1{{font-size:24px;margin:0 0 4px}} h2{{font-size:17px;margin:30px 0 10px;padding-bottom:6px;border-bottom:2px solid #1A1A1A}}
h3{{font-size:15px;margin:0 0 4px}} p{{margin:0 0 6px;overflow-wrap:anywhere}} .muted{{color:#666}} .small{{font-size:11px;color:#666;word-break:break-all}}
.top{{display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap;align-items:flex-start}}
.verdict{{font-size:30px;font-weight:700;color:{verdict_col};border:3px solid {verdict_col};padding:4px 14px}}
.kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:16px}}
.kpi{{border:1px solid #DDD;padding:10px 12px}} .kpi b{{display:block;font-size:20px}}
.map{{display:grid;grid-template-columns:repeat(6,1fr);gap:6px;max-width:760px}}
.scroll{{overflow-x:auto}}
.cell{{border:3px solid #CCC;background:#FAFAFA}} .cell img{{width:100%;display:block;aspect-ratio:1}}
.cap{{padding:4px 6px;font-size:11px;line-height:1.3}}
.ev{{display:flex;gap:16px;padding:12px 0;border-bottom:1px solid #E3E3E3}}
.pair{{display:flex;gap:8px}} figure{{margin:0;text-align:center;font-size:11px;color:#666}}
figure img{{width:110px;height:110px;display:block}} figure.none{{width:110px;height:110px;background:#F3F3F3;display:grid;place-items:center}}
table{{width:100%;border-collapse:collapse;font-size:13px}} th,td{{text-align:left;padding:5px 6px;border-bottom:1px solid #E3E3E3;vertical-align:top}}
th{{background:#F5F5F5}} .r{{text-align:right}}
.print{{border:1px solid #1A1A1A;background:#fff;padding:6px 12px;cursor:pointer;font:inherit}}
@media (max-width:700px){{.kpis{{grid-template-columns:repeat(2,1fr)}} .ev{{flex-direction:column}}}}
@media print{{.print{{display:none}} main{{padding:0}} h2{{break-after:avoid}} .ev,.cell{{break-inside:avoid}}}}
</style></head><body><main>
<div class="top"><div><h1>Order inspection record</h1>
<p>CSI final station · good fixture G01–G30 · placement-by-placement inspection</p>
<p class="muted">Empty-tray reference: {esc(str(s['reference']))}<br>Generated {_t(datetime.now().timestamp(), True)} ·
{esc(meta.get('model', ''))}</p></div>
<div style="text-align:right"><div class="verdict">{s['verdict']}</div>
<p style="margin-top:8px"><button class="print" onclick="window.print()">Print / save as PDF</button></p></div></div>

<div class="kpis">
<div class="kpi">Placed<b>{s['filled']} / 30</b></div>
<div class="kpi">Problems<b style="color:{'#C0392B' if n_bad else '#2E7D32'}">{n_bad}</b>{f'<span class="muted">{n_check} to check by eye</span>' if n_check else ''}</div>
<div class="kpi">Order time<b>{_t(tracker.t0)}–{_t(last_ts)}</b><span class="muted">{_dur(last_ts - tracker.t0 if tracker.t0 else None)}</span></div>
<div class="kpi">Cycle time (median)<b>{_dur(cyc.get('median_s'))}</b><span class="muted">{cyc.get('per_hour') or '—'} placements / hour</span></div>
</div>

{f'<h2>Tray findings (organizers’ rules)</h2><table>{find_rows}</table>' if find_rows else ''}

<h2>Problems with photo evidence</h2>
{''.join(ev) or '<p>No problems found.</p>'}

<h2>Fixture at the end of the record</h2>
<p class="muted">As seen by the camera; each photo is the vial when it was inspected (empty holes: the empty tray).</p>
<div class="map">{''.join(cells)}</div>

<h2>Cycle time between placements</h2>
<p class="muted">Seconds from one vial to the next; the first placement is left out because it includes order setup.
Slow = more than {_dur(cyc.get('slow_cut_s'))} (twice the median, at least a minute over it). Slow placements: <b>{slow}</b>.</p>
{_chart(cyc) if cyc['placements'] else ''}

<h2>All positions</h2>
<div class="scroll"><table><tr><th>Pos</th><th>Status</th><th>Placed</th><th class="r">Gap</th><th class="r">Change</th><th class="r">Cap colour</th><th class="r">Model</th><th>Photo</th></tr>
{''.join(table)}</table></div>

<h2>Event log</h2>
<div class="scroll"><table><tr><th>Time</th><th>Pos</th><th>Kind</th><th>Event</th></tr>{events}</table></div>

<h2>Settings</h2>
<p class="muted">Empty below {cfg['occupied_min']} · open vial below {cfg['capped_min']} with cap colour under {cfg['open_max_yellow']} ·
moved after inspection at {cfg['disturb_min']} · settled when under {cfg['stable_max']} between photos · window {cfg['window']} px ·
model can reject: {cfg['model_can_reject']} · {s['frames']} photos compared.</p>
</main></body></html>"""
