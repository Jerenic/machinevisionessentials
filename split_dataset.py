"""
Teilt einen Label-Studio-YOLO-Export in train/val/test (README-Struktur unter candy_dataset/).

Standard-Split: 80 % train, 10 % val, 10 % test (im README steht fälschlich „80 % test“).
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def find_latest_export(candy_root: Path) -> Path:
    projects = [p for p in candy_root.iterdir() if p.is_dir() and p.name.startswith("project-")]
    if not projects:
        raise FileNotFoundError(f"Kein Unterordner project-* unter {candy_root}")
    return max(projects, key=lambda p: p.stat().st_mtime)


def load_class_names(classes_file: Path) -> list[str]:
    lines = classes_file.read_text(encoding="utf-8").strip().splitlines()
    return [ln.strip() for ln in lines if ln.strip()]


def collect_pairs(src_images: Path, src_labels: Path) -> list[tuple[Path, Path | None]]:
    pairs: list[tuple[Path, Path | None]] = []
    for img in sorted(src_images.iterdir()):
        if not img.is_file() or img.suffix.lower() not in IMAGE_EXTS:
            continue
        label = src_labels / f"{img.stem}.txt"
        if not label.is_file():
            print(f"Warnung: kein Label zu {img.name}, übersprungen.", file=sys.stderr)
            continue
        pairs.append((img, label))
    return pairs


def split_indices(n: int, train_r: float, val_r: float, test_r: float, seed: int) -> tuple[list[int], list[int], list[int]]:
    s = train_r + val_r + test_r
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"Anteile müssen 1.0 ergeben, ist {s}")
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    # Ganzzahlen so wählen, dass Summe == n (Rest landet in test)
    n_train = int(n * train_r)
    n_val = int(n * val_r)
    n_test = n - n_train - n_val
    train_i = idx[:n_train]
    val_i = idx[n_train : n_train + n_val]
    test_i = idx[n_train + n_val :]
    return train_i, val_i, test_i


def copy_split(
    pairs: list[tuple[Path, Path]],
    indices: list[int],
    out_images: Path,
    out_labels: Path,
) -> None:
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    for i in indices:
        img, lbl = pairs[i]
        shutil.copy2(img, out_images / img.name)
        shutil.copy2(lbl, out_labels / lbl.name)


def write_data_yaml(out_root: Path, names: list[str]) -> None:
    lines = [
        "# Ultralytics YOLO: train/val/test relativ zum Ordner dieser YAML (candy_dataset/).",
        "# Kein path:-Eintrag: Root ist das Verzeichnis von data.yaml (siehe Ultralytics-Doku).",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(names)}",
        "names:",
    ]
    for i, n in enumerate(names):
        safe = n.replace("'", "''")
        lines.append(f"  {i}: {safe}")
    (out_root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    source: Path,
    out_root: Path,
    train_r: float,
    val_r: float,
    test_r: float,
    seed: int,
) -> None:
    src_images = source / "images"
    src_labels = source / "labels"
    classes_file = source / "classes.txt"
    if not src_images.is_dir() or not src_labels.is_dir():
        raise FileNotFoundError(f"Erwarte {source}/images und {source}/labels")
    names = load_class_names(classes_file) if classes_file.is_file() else []

    pairs = collect_pairs(src_images, src_labels)
    if not pairs:
        raise RuntimeError("Keine gültigen Bild-Label-Paare gefunden.")
    n = len(pairs)
    train_i, val_i, test_i = split_indices(n, train_r, val_r, test_r, seed)

    for sub in ("train", "val", "test"):
        (out_root / "images" / sub).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / sub).mkdir(parents=True, exist_ok=True)

    copy_split(pairs, train_i, out_root / "images" / "train", out_root / "labels" / "train")
    copy_split(pairs, val_i, out_root / "images" / "val", out_root / "labels" / "val")
    copy_split(pairs, test_i, out_root / "images" / "test", out_root / "labels" / "test")

    if names:
        write_data_yaml(out_root, names)
    else:
        print("Hinweis: keine classes.txt – data.yaml nicht geschrieben.", file=sys.stderr)

    print(f"Quelle: {source}")
    print(f"Ziel:   {out_root}")
    print(f"Train: {len(train_i)}, Val: {len(val_i)}, Test: {len(test_i)} (gesamt {n})")


def main() -> None:
    root = Path(__file__).resolve().parent
    default_candy = root / "candy_dataset"

    p = argparse.ArgumentParser(description="YOLO-Export nach train/val/test aufteilen.")
    p.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Ordner mit images/, labels/, classes.txt (Standard: neuester project-* unter candy_dataset/).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=default_candy,
        help="Ausgabe-Root mit images/{train,val,test}, labels/{...}, data.yaml (Standard: candy_dataset/).",
    )
    p.add_argument("--train", type=float, default=0.8, help="Anteil Train (Standard: 0.8)")
    p.add_argument("--val", type=float, default=0.1, help="Anteil Val (Standard: 0.1)")
    p.add_argument("--test", type=float, default=0.1, help="Anteil Test (Standard: 0.1)")
    p.add_argument("--seed", type=int, default=42, help="Zufallssamen für den Split")
    args = p.parse_args()

    source = args.source
    if source is None:
        source = find_latest_export(args.out if args.out.is_dir() else default_candy)
    else:
        source = source.resolve()

    out_root = args.out.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    run(source, out_root, args.train, args.val, args.test, args.seed)


if __name__ == "__main__":
    main()
