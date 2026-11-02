import csv
import ntpath
import os
import re
import sys

CLASSES = ["pass", "missing_cap", "crooked_cap", "damaged_vial", "no_vial"]
DEFECT_CLASSES = ["missing_cap", "crooked_cap", "damaged_vial"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
DEFECT_IDX = [CLASS_TO_IDX[c] for c in DEFECT_CLASSES]
PASS_IDX = CLASS_TO_IDX["pass"]
NO_VIAL_IDX = CLASS_TO_IDX["no_vial"]

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

_POS_RE = re.compile(r"(?<![A-Za-z])(?P<fixture>[GS])[_-]?(?P<pos>\d{1,2})(?!\d)")


def scan_folder(root):
    """Return [(path, class_idx)] from a folder with one sub-folder per class.

    Missing class folders are fine (a twin drop may not have every class yet).
    Unknown folders are reported and skipped, so a typo doesn't go unnoticed.
    """
    if not os.path.isdir(root):
        sys.exit(f"Not a folder: {root}")
    items = []
    for name in sorted(os.listdir(root)):
        sub = os.path.join(root, name)
        if not os.path.isdir(sub):
            continue
        if name not in CLASS_TO_IDX:
            print(f"warning: skipping unknown class folder '{name}' in {root}", file=sys.stderr)
            continue
        for dirpath, _, files in os.walk(sub):
            for f in sorted(files):
                if os.path.splitext(f)[1].lower() in IMG_EXTS:
                    items.append((os.path.join(dirpath, f), CLASS_TO_IDX[name]))
    if not items:
        sys.exit(f"No images found under {root} (expected sub-folders named {CLASSES})")
    return items


def _resolve(path, label, base):
    """Find a listed crop even if the list was made on another computer.

    Lists store paths relative to the CSV, with forward slashes. Older lists
    (or ones moved from Windows to Linux) may hold a full path that doesn't
    exist here; then look for the same file name in the usual label folders
    next to the CSV: ../real_labeled/<label>/ or ./<label>/.
    """
    p = path.replace("\\", "/")
    candidates = [p if os.path.isabs(p) else os.path.join(base, p)]
    name = ntpath.basename(path)
    candidates += [os.path.join(base, "..", "real_labeled", label, name), os.path.join(base, label, name)]
    for c in candidates:
        if os.path.exists(c):
            return os.path.normpath(c)
    return os.path.normpath(candidates[0])   # reported as missing when opened


def read_list(csv_path):
    """Read a crop list CSV with columns path,label."""
    items = []
    base = os.path.dirname(os.path.abspath(csv_path))
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            label = row["label"].strip()
            if label not in CLASS_TO_IDX:
                sys.exit(f"Unknown label '{label}' in {csv_path}")
            items.append((_resolve(row["path"], label, base), CLASS_TO_IDX[label]))
    missing = [p for p, _ in items if not os.path.exists(p)]
    if missing:
        sys.exit(f"{len(missing)} crops listed in {csv_path} were not found, e.g. {missing[0]}\n"
                 f"Copy the real_labeled folder next to the splits folder (same parent) and retry.")
    return items


def write_list(csv_path, items):
    """Paths are stored relative to the CSV, with forward slashes, so the list works on any computer."""
    base = os.path.dirname(os.path.abspath(csv_path))
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label"])
        for path, idx in items:
            w.writerow([os.path.relpath(os.path.abspath(path), base).replace(os.sep, "/"), CLASSES[idx]])


def load_items(source):
    """A folder of class sub-folders, or a CSV list."""
    if source.lower().endswith(".csv"):
        return read_list(source)
    return scan_folder(source)


def parse_position(path):
    """(fixture, position) from a file name like G_14_pass_0003.png, else (None, None)."""
    m = _POS_RE.search(os.path.basename(path))
    if not m:
        return None, None
    return m.group("fixture"), int(m.group("pos"))


def class_counts(items):
    counts = {c: 0 for c in CLASSES}
    for _, idx in items:
        counts[CLASSES[idx]] += 1
    return counts
