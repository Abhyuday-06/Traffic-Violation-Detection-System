"""
prepare_helmet_2class.py
========================
Fixes the Helmet dataset for clean 2-class (Helmet / No-Helmet) detection:

  1. Converts polygon annotations (>5 values per line) → bounding boxes
     by computing the min/max envelope of the polygon vertices.
  2. Drops all class-2 (person) annotations, which are inconsistently
     labelled across 78 % of the training images.
  3. Drops empty label files that would confuse training.
  4. Writes the fixed labels to  Helmet_fixed/{split}/labels/
  5. Creates directory junctions (no file copying) for the images so
     YOLO can resolve label paths automatically.
  6. Writes  Helmet_fixed/data.yaml  with nc=2.

Run once before training:
    python prepare_helmet_2class.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

# ── paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "Helmet"
DST_DIR = PROJECT_ROOT / "Helmet_fixed"
SPLITS = ("train", "valid", "test")

# ── class remapping ──────────────────────────────────────────────────────────
# Original:  0=Helmet  1=No-Helmet  2=person (drop)
KEEP_CLASSES = {0, 1}          # class IDs to keep (no remapping needed)
NEW_NC = 2
NEW_NAMES = ["Helmet", "No-Helmet"]


# ── helpers ──────────────────────────────────────────────────────────────────

def polygon_to_bbox(values: list[float]) -> tuple[float, float, float, float]:
    """Convert flat polygon [x1,y1,x2,y2,...] to YOLO bbox [cx,cy,w,h]."""
    xs = values[0::2]
    ys = values[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    w = x_max - x_min
    h = y_max - y_min
    return cx, cy, w, h


def fix_label_file(src_path: Path) -> list[str]:
    """
    Return fixed annotation lines for a single label file.
    Returns an empty list for files that end up with no valid annotations.
    """
    fixed: list[str] = []
    raw = src_path.read_text(encoding="utf-8").strip()
    if not raw:
        return fixed

    for line in raw.splitlines():
        parts = line.strip().split()
        if not parts:
            continue

        cls = int(parts[0])
        if cls not in KEEP_CLASSES:
            continue                    # drop person (class 2)

        coords = list(map(float, parts[1:]))

        if len(coords) == 4:
            # standard bbox: [cx, cy, w, h] — keep as-is
            cx, cy, w, h = coords
        elif len(coords) > 4 and len(coords) % 2 == 0:
            # polygon: convert to bbox envelope
            cx, cy, w, h = polygon_to_bbox(coords)
        else:
            # malformed line — skip
            print(f"  SKIP malformed line in {src_path.name}: {line[:60]}")
            continue

        # clamp to [0, 1] to guard against out-of-bounds polygon vertices
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        w  = max(0.0, min(1.0, w))
        h  = max(0.0, min(1.0, h))

        fixed.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return fixed


def copy_images(src_image_dir: Path, dst_image_dir: Path) -> int:
    """Copy image files so YOLO resolves labels from the correct directory."""
    import shutil
    dst_image_dir.mkdir(parents=True, exist_ok=True)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [p for p in src_image_dir.iterdir() if p.suffix.lower() in image_extensions]
    for img in images:
        dst = dst_image_dir / img.name
        if not dst.exists():
            shutil.copy2(img, dst)
    print(f"  copied {len(images)} images -> {dst_image_dir}")
    return len(images)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    stats: dict[str, dict] = {}

    for split in SPLITS:
        src_label_dir = SRC_DIR / split / "labels"
        src_image_dir = SRC_DIR / split / "images"
        dst_label_dir = DST_DIR / split / "labels"
        dst_image_dir = DST_DIR / split / "images"   # will be a junction

        if not src_label_dir.exists():
            print(f"[{split}] label dir not found, skipping: {src_label_dir}")
            continue
        if not src_image_dir.exists():
            print(f"[{split}] image dir not found, skipping: {src_image_dir}")
            continue

        dst_label_dir.mkdir(parents=True, exist_ok=True)

        # ── copy images ─────────────────────────────────────────────────────────
        print(f"\n[{split}] Copying images (this may take a moment) ...")
        copy_images(src_image_dir, dst_image_dir)

        # ── process labels ──────────────────────────────────────────────────
        label_files = sorted(src_label_dir.glob("*.txt"))
        total = len(label_files)
        kept = 0
        empty_dropped = 0
        poly_converted = 0
        person_dropped_lines = 0

        print(f"[{split}] Processing {total} label files …")

        for src_lf in label_files:
            # quick per-file stats
            raw_lines = [l for l in src_lf.read_text(encoding="utf-8").splitlines() if l.strip()]
            before_person = sum(1 for l in raw_lines if l.strip() and int(l.split()[0]) == 2)
            before_poly   = sum(1 for l in raw_lines if l.strip() and len(l.split()) > 5)
            person_dropped_lines += before_person
            poly_converted       += before_poly

            fixed_lines = fix_label_file(src_lf)

            dst_lf = dst_label_dir / src_lf.name
            if fixed_lines:
                dst_lf.write_text("\n".join(fixed_lines) + "\n", encoding="utf-8")
                kept += 1
            else:
                # write empty file so YOLO doesn't complain about missing labels
                dst_lf.write_text("", encoding="utf-8")
                empty_dropped += 1

        stats[split] = {
            "total_files":        total,
            "files_with_annots":  kept,
            "files_now_empty":    empty_dropped,
            "poly_converted":     poly_converted,
            "person_lines_dropped": person_dropped_lines,
        }

    # ── write data.yaml ─────────────────────────────────────────────────────
    dst_yaml = DST_DIR / "data.yaml"
    yaml_data = {
        "path":  str(DST_DIR),
        "train": "train/images",
        "val":   "valid/images",
        "test":  "test/images",
        "nc":    NEW_NC,
        "names": NEW_NAMES,
    }
    with dst_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_data, f, sort_keys=False, allow_unicode=True)
    print(f"\nWrote {dst_yaml}")

    # ── summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DATASET FIX SUMMARY")
    print("=" * 60)
    for split, s in stats.items():
        print(f"\n  [{split}]")
        print(f"    Total label files   : {s['total_files']}")
        print(f"    Files with annots   : {s['files_with_annots']}")
        print(f"    Files now empty     : {s['files_now_empty']}")
        print(f"    Polygon->BBox fixed  : {s['poly_converted']}")
        print(f"    Person lines dropped: {s['person_lines_dropped']}")
    print("\n  Ready to train on: Helmet_fixed/data.yaml")
    print("  Classes: 0=Helmet  1=No-Helmet  (nc=2)")
    print("=" * 60)


if __name__ == "__main__":
    if DST_DIR.exists():
        print(f"Helmet_fixed/ already exists. Delete it first to re-run.")
        sys.exit(0)
    main()
