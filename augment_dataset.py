"""
Erweitert den Trainings-Split um augmentierte Bilder (README: Advanced, Ziel ≥ 200 Bilder im Train).

Verwendet Albumentations mit YOLO-normalisierten Boxen; schreibt neue Paare in
candy_dataset/images/train und candy_dataset/labels/train (Präfix ``aug_``).
"""

from __future__ import annotations

import argparse
import random
import uuid
from pathlib import Path

import albumentations as A
import cv2

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def read_yolo_labels(path: Path) -> tuple[list[list[float]], list[int]]:
    if not path.is_file():
        return [], []
    bboxes: list[list[float]] = []
    classes: list[int] = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        bbox = [float(x) for x in parts[1:5]]
        bboxes.append(bbox)
        classes.append(cls)
    return bboxes, classes


def write_yolo_labels(path: Path, bboxes: list[list[float]], classes: list[int]) -> None:
    lines = []
    for cls, bb in zip(classes, bboxes, strict=True):
        lines.append(f"{cls} {bb[0]:.10f} {bb[1]:.10f} {bb[2]:.10f} {bb[3]:.10f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def build_transform() -> A.Compose:
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.4),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.3),
    )


def augment_one(
    image_path: Path,
    label_path: Path,
    out_img: Path,
    out_lbl: Path,
    transform: A.Compose,
) -> bool:
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        return False
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    bboxes, classes = read_yolo_labels(label_path)
    if not bboxes:
        return False
    try:
        out = transform(image=image, bboxes=bboxes, class_labels=classes)
    except Exception:
        return False
    new_boxes = out["bboxes"]
    new_cls = out["class_labels"]
    if not new_boxes:
        return False
    out_bgr = cv2.cvtColor(out["image"], cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_img), out_bgr)
    bb = [[float(x) for x in r] for r in new_boxes]
    cc = [int(c) for c in new_cls]
    write_yolo_labels(out_lbl, bb, cc)
    return True


def count_train_images(train_dir: Path) -> int:
    return sum(1 for p in train_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS and p.is_file())


def run(
    dataset_root: Path,
    target_train: int,
    seed: int,
    max_attempts_factor: int,
) -> None:
    img_train = dataset_root / "images" / "train"
    lbl_train = dataset_root / "labels" / "train"
    if not img_train.is_dir() or not lbl_train.is_dir():
        raise FileNotFoundError(f"Erwarte {img_train} und {lbl_train}")

    sources = [
        p
        for p in sorted(img_train.iterdir())
        if p.suffix.lower() in IMAGE_EXTS and (lbl_train / f"{p.stem}.txt").is_file()
    ]
    if not sources:
        raise RuntimeError("Keine Trainingsbilder mit Label gefunden.")

    rng = random.Random(seed)
    start = count_train_images(img_train)
    if start >= target_train:
        print(f"Train hat bereits {start} Bilder (Ziel {target_train}). Nichts zu tun.")
        return

    need = target_train - start
    transform = build_transform()
    attempts = 0
    max_attempts = need * max_attempts_factor
    made = 0

    while made < need and attempts < max_attempts:
        attempts += 1
        src = rng.choice(sources)
        lbl = lbl_train / f"{src.stem}.txt"
        uid = uuid.uuid4().hex[:10]
        stem = f"aug_{uid}_{src.stem}"
        out_img = img_train / f"{stem}{src.suffix.lower()}"
        out_lbl = lbl_train / f"{stem}.txt"
        if augment_one(src, lbl, out_img, out_lbl, transform):
            made += 1
            if made % 20 == 0 or made == need:
                print(f"... {made}/{need} neue Augmentationen")

    end = count_train_images(img_train)
    print(f"Fertig: Trainingsbilder {start} -> {end} (Ziel war >= {target_train}).")


def main() -> None:
    p = argparse.ArgumentParser(description="Trainingsdaten augmentieren (YOLO-Format).")
    p.add_argument(
        "--dataset",
        type=Path,
        default=Path("candy_dataset"),
        help="Dataset-Root mit images/train, labels/train (Standard: candy_dataset).",
    )
    p.add_argument(
        "--target",
        type=int,
        default=200,
        help="Mindestanzahl Bilder im Train-Ordner (README Advanced: ≥ 200).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max-attempts-factor",
        type=int,
        default=30,
        help="Max. Versuche = fehlende × Faktor (falls Transformationen ausfallen).",
    )
    args = p.parse_args()
    root = args.dataset.resolve()
    run(root, args.target, args.seed, args.max_attempts_factor)


if __name__ == "__main__":
    main()
