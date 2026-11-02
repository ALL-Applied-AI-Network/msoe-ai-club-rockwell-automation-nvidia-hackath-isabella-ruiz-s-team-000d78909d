"""Tray-level findings in the organizers' format (organizer_rules.py is rules.py from their labelling tool).

Each finding is True, False or None (not established), exactly as the organizers define them.
"""
from organizer_rules import summarize

# order-tracking state -> (presence, class). None means "not established".
ORDER_SLOT = {
    "ok": (True, "pass"),
    "check": (True, None),          # cap present but the model is unsure: not established
    "open": (True, "missing_cap"),
    "flagged": (True, None),        # replaced below by the model's label
    "disturbed": (True, None),      # changed after inspection: needs a look
    "pending": (False, "no_vial"),
    "missed": (False, "no_vial"),
    "missing": (False, "no_vial"),
    "removed": (False, "no_vial"),
}


def from_photo(result):
    """Findings from a single-photo result (the model's label for every position)."""
    if not result.get("positions"):
        return None
    slots = []
    for p in result["positions"]:
        label = p["label"] if p["status"] in ("defect",) else ("no_vial" if p["status"] in ("empty", "missed") else "pass")
        slots.append({"position": int(p["position"][1:]), "presence": label != "no_vial", "class": label,
                      "review_status": "model"})
    fc = (result.get("fill") or {})
    return summarize("good", slots, expected_through=fc.get("count") if fc.get("mode") == "manual" else None)


def from_order(order, expected_through=None):
    """Findings from the order tracker's state."""
    if not order or not order.get("active"):
        return None
    slots = []
    for p in order["positions"]:
        presence, cls = ORDER_SLOT[p["state"]]
        if p["state"] == "flagged":
            cls = p.get("label")
        slots.append({"position": int(p["position"][1:]), "presence": presence, "class": cls,
                      "review_status": "model"})
    if expected_through is None and order.get("complete") and any(p["state"] == "missing" for p in order["positions"]):
        expected_through = 30
    return summarize("good", slots, expected_through=expected_through)
