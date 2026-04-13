"""README Step 3: YOLO-Training mit Ultralytics (Detect)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO


def default_workers() -> int:
    # Unter Windows spawnen viele Worker jeweils eigene Prozesse (torch/cv2/DLLs) und treffen
    # leicht auf WinError 1455 (Auslagerungsdatei) oder OOM — 0 = Laden im Hauptthread, stabiler.
    return 0 if sys.platform == "win32" else 8


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO-Objekterkennung trainieren (Ultralytics).")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("candy_dataset/data.yaml"),
        help="Pfad zur data.yaml",
    )
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="Startgewichte (README: u. a. yolo11n, yolo26n — je nach Ultralytics-Version).",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=f"DataLoader-Worker (Standard: {default_workers()} auf Windows, sonst 8). Bei Speicherfehlern 0 wählen.",
    )
    parser.add_argument("--device", default="", help="z. B. cpu, 0, 0,1 (leer = Auto)")
    # project + name -> z. B. runs/detect/candy_train (ohne doppeltes runs/detect/)
    parser.add_argument("--project", type=Path, default=Path("runs"))
    parser.add_argument("--name", default="detect/candy_train")
    args = parser.parse_args()

    data_path = args.data.resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"data.yaml nicht gefunden: {data_path}")

    model = YOLO(args.model)
    workers = args.workers if args.workers is not None else default_workers()

    train_kw: dict = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": workers,
        "project": str(args.project.resolve()),
        "name": args.name,
    }
    if args.device:
        train_kw["device"] = args.device

    model.train(**train_kw)


if __name__ == "__main__":
    main()
