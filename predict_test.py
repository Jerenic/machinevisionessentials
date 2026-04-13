"""README Step 4: Inferenz auf dem Testsplit (Ausgabe wie in README mit Bounding Boxes)."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO auf Testbildern ausführen und speichern.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Pfad zu best.pt (Standard: neuestes unter runs/**/weights/best.pt)",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("candy_dataset/images/test"),
        help="Ordner mit Testbildern",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--project", type=Path, default=Path("runs"))
    parser.add_argument("--name", default="detect/predict_test")
    parser.add_argument("--device", default="", help="cpu oder GPU-Index")
    args = parser.parse_args()

    w = args.weights
    if w is None:
        root = Path("runs").resolve()
        candidates = sorted(root.rglob("weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise FileNotFoundError(
                "Kein best.pt unter runs/ gefunden. Zuerst train_yolo.py ausführen oder --weights setzen."
            )
        w = candidates[0]
        print(f"Verwende Gewichte: {w}")
    w = w.resolve()
    if not w.is_file():
        raise FileNotFoundError(
            f"Gewichte nicht gefunden: {w}\nZuerst train_yolo.py ausführen oder --weights setzen."
        )
    src = args.source.resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"Test-Ordner fehlt: {src}")

    model = YOLO(str(w))
    pred_kw: dict = {
        "source": str(src),
        "imgsz": args.imgsz,
        "save": True,
        "project": str(args.project.resolve()),
        "name": args.name,
        "exist_ok": True,
    }
    if args.device:
        pred_kw["device"] = args.device

    model.predict(**pred_kw)
    out_dir = Path(args.project) / Path(args.name)
    print(f"Fertig. Ausgaben unter: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
