"""Augment the training split with transformed OBB samples.

Uses Albumentations with keypoint mode to transform the four OBB corner
points alongside the image.  After transformation, coordinates are
**clamped to [0, 1]** to prevent out-of-bounds values that would crash
YOLO validation.  A configurable retry budget (``max_attempts``) absorbs
stochastic transform failures without silent data loss.

Augmentation probability is weighted towards user-specified focus classes
(e.g. mars, milky_way) to counteract class imbalance.
"""

from __future__ import annotations

import argparse
import random
import uuid
from pathlib import Path

import albumentations as A
import cv2

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Canonical class order — must match data.yaml / classes.txt (alphabetical).
CLASS_NAMES: list[str] = [
    "bounty",
    "galaxy",
    "galaxy_caramel",
    "hand",
    "maltesers",
    "mars",
    "milky_way",
    "snickers",
    "twix",
]


# ---------------------------------------------------------------------------
# Label I/O
# ---------------------------------------------------------------------------

def _read_obb(path: Path) -> tuple[list[list[float]], list[int]]:
    """Read YOLO-OBB label file (class x1 y1 x2 y2 x3 y3 x4 y4)."""
    if not path.is_file():
        return [], []
    obbs: list[list[float]] = []
    classes: list[int] = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.split()
        if len(parts) >= 9:
            classes.append(int(parts[0]))
            obbs.append([float(x) for x in parts[1:9]])
    return obbs, classes


def _write_obb(path: Path, obbs: list[list[float]], classes: list[int]) -> None:
    """Write YOLO-OBB label file."""
    lines = [f"{cls} {' '.join(f'{c:.6f}' for c in coords)}"
             for cls, coords in zip(classes, obbs, strict=True)]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


# ---------------------------------------------------------------------------
# Transform pipeline
# ---------------------------------------------------------------------------

def _build_transform() -> A.Compose:
    """Heavy rotation + colour jitter suited for reflective candy wrappers."""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.SafeRotate(limit=180, border_mode=cv2.BORDER_CONSTANT, p=0.8),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.5),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


# ---------------------------------------------------------------------------
# Single-image augmentation
# ---------------------------------------------------------------------------

def _clamp(v: float) -> float:
    """Clamp normalised coordinate to [0, 1]."""
    return max(0.0, min(1.0, v))


def _augment_one(
    image_path: Path,
    label_path: Path,
    out_img: Path,
    out_lbl: Path,
    transform: A.Compose,
) -> bool:
    """Apply one random augmentation.  Returns False on any failure."""
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        return False
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    obbs, classes = _read_obb(label_path)
    if not obbs:
        return False

    # Flatten normalised OBB corners → absolute keypoints
    kps = []
    for obb in obbs:
        for i in range(0, 8, 2):
            kps.append((obb[i] * w, obb[i + 1] * h))

    try:
        out = transform(image=rgb, keypoints=kps)
    except Exception:
        return False

    new_kps = out["keypoints"]
    if not new_kps or len(new_kps) != len(kps):
        return False

    # Write transformed image
    cv2.imwrite(str(out_img), cv2.cvtColor(out["image"], cv2.COLOR_RGB2BGR))

    # Reconstruct OBBs with clamped coordinates
    new_obbs: list[list[float]] = []
    for i in range(0, len(new_kps), 4):
        corners = new_kps[i:i + 4]
        new_obbs.append([
            v for px, py in corners
            for v in (_clamp(px / w), _clamp(py / h))
        ])

    _write_obb(out_lbl, new_obbs, classes)
    return True


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def _count_images(directory: Path) -> int:
    return sum(1 for p in directory.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES)


def run(
    dataset_root: Path,
    target_train: int,
    seed: int,
    focus_classes: list[str],
    focus_multiplier: int,
) -> None:
    img_dir = dataset_root / "images" / "train"
    lbl_dir = dataset_root / "labels" / "train"
    if not img_dir.is_dir() or not lbl_dir.is_dir():
        raise FileNotFoundError(f"Expected {img_dir} and {lbl_dir}")

    focus_ids = [CLASS_NAMES.index(c) for c in focus_classes if c in CLASS_NAMES]
    print(f"Focus classes: {focus_classes} (IDs {focus_ids}), weight ×{focus_multiplier}")

    sources = sorted(
        p for p in img_dir.iterdir()
        if p.suffix.lower() in _IMAGE_SUFFIXES and (lbl_dir / f"{p.stem}.txt").is_file()
    )
    if not sources:
        raise RuntimeError("No labelled training images found.")

    # Weight source images by presence of focus classes
    weights: list[int] = []
    for src in sources:
        _, cls_ids = _read_obb(lbl_dir / f"{src.stem}.txt")
        weights.append(focus_multiplier if any(c in focus_ids for c in cls_ids) else 1)

    rng = random.Random(seed)
    start = _count_images(img_dir)
    need = max(0, target_train - start)
    transform = _build_transform()

    made = 0
    attempts = 0
    max_attempts = need * 30

    while made < need and attempts < max_attempts:
        attempts += 1
        src = rng.choices(sources, weights=weights, k=1)[0]
        uid = uuid.uuid4().hex[:8]
        stem = f"aug_obb_{uid}_{src.stem}"
        if _augment_one(
            src,
            lbl_dir / f"{src.stem}.txt",
            img_dir / f"{stem}{src.suffix.lower()}",
            lbl_dir / f"{stem}.txt",
            transform,
        ):
            made += 1
            if made % 20 == 0 or made == need:
                print(f"  {made}/{need} augmentations generated")

    if attempts >= max_attempts and made < need:
        print(f"WARNING: hit max_attempts ({max_attempts}), only {made}/{need} generated.")

    print(f"Training images: {start} → {_count_images(img_dir)}  (target ≥ {target_train})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Augment OBB training data.")
    ap.add_argument("--dataset", type=Path, default=Path("candy_dataset"))
    ap.add_argument("--target", type=int, default=200, help="Minimum training images.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--focus-classes", nargs="+", default=["mars", "milky_way"])
    ap.add_argument("--focus-multiplier", type=int, default=5)
    args = ap.parse_args()
    run(args.dataset.resolve(), args.target, args.seed, args.focus_classes, args.focus_multiplier)


if __name__ == "__main__":
    main()
