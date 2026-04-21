"""Train a YOLO-OBB model via Ultralytics on the candy dataset.

Wraps ``ultralytics.YOLO.train()`` with CLI arguments for reproducible
experiment tracking.  Supports early stopping (``--patience``) and
selectable optimiser (``--optimizer``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO


def _default_workers() -> int:
    """Return 0 on Windows to avoid WinError 1455 / OOM in spawned workers."""
    return 0 if sys.platform == "win32" else 8


def main() -> None:
    ap = argparse.ArgumentParser(description="YOLO-OBB training (Ultralytics).")
    ap.add_argument("--data", type=Path, default=Path("candy_dataset/data.yaml"),
                    help="Path to data.yaml (must reference OBB labels).")
    ap.add_argument("--model", default="yolo11n-obb.pt",
                    help="Pre-trained OBB weights to start from.")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=50,
                    help="Early-stopping patience (epochs w/o val improvement; 0 = off).")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--optimizer", default="AdamW",
                    help="Optimiser name accepted by Ultralytics (AdamW, SGD, Adam, auto).")
    ap.add_argument("--workers", type=int, default=None,
                    help="DataLoader workers (default: 0 on Windows, 8 elsewhere).")
    ap.add_argument("--device", default="", help="Device string, e.g. cpu, 0, 0,1.")
    ap.add_argument("--project", type=Path, default=Path("runs"))
    ap.add_argument("--name", default="obb/candy_train")
    args = ap.parse_args()

    data_path = args.data.resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"data.yaml not found: {data_path}")

    model = YOLO(args.model)
    workers = args.workers if args.workers is not None else _default_workers()

    train_kw: dict = {
        "data": str(data_path),
        "epochs": args.epochs,
        "patience": args.patience,
        "optimizer": args.optimizer,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": workers,
        "project": str(args.project.resolve()),
        "name": args.name,
        "task": "obb",
    }
    if args.device:
        train_kw["device"] = args.device

    model.train(**train_kw)


if __name__ == "__main__":
    main()
