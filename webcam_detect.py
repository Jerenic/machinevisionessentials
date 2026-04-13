"""README Step 5: Webcam + YOLO (Echtzeit). Expert: zusätzliche Klassen brauchen Anpassung im Modell."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def _default_weights() -> str:
    root = Path("runs").resolve()
    candidates = sorted(root.rglob("weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return "runs/detect/candy_train/weights/best.pt"
    w = candidates[0]
    print(f"Verwende Gewichte: {w}")
    return str(w)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live-Detektion mit Webcam (OpenCV + YOLO).")
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="Pfad zu best.pt (leer = neuester Lauf unter runs/)",
    )
    parser.add_argument("--camera", type=int, default=0, help="Videoquelle (0 = Standard-Webcam)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="", help="cpu oder GPU-Index")
    args = parser.parse_args()

    weights = args.weights if args.weights.strip() else _default_weights()
    model = YOLO(weights)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Webcam konnte nicht geöffnet werden.")

    pred_kw: dict = {"imgsz": args.imgsz, "verbose": False}
    if args.device:
        pred_kw["device"] = args.device

    print("Live-Modus — Taste 'q' zum Beenden.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame, **pred_kw)
        annotated = results[0].plot()
        cv2.imshow("Candy YOLO", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
