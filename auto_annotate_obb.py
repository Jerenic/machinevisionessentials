"""Auto-annotate unlabelled images with a trained YOLO-OBB model.

Runs batch inference, saves visualised predictions for manual review,
and exports a Label Studio-compatible JSON with OBB annotations
(``rectanglelabels`` + ``rotation``).

Float32 → native-float conversion is applied explicitly to avoid
``json.dump`` failures with NumPy scalars.
"""

from __future__ import annotations

import argparse
import json
import math
import uuid
from pathlib import Path

import cv2  # noqa: F401 – imported for side-effects (image codec registration)
from ultralytics import YOLO

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

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _obb_to_ls(
    cx: float, cy: float, w: float, h: float, angle_rad: float,
    img_w: int, img_h: int,
) -> dict:
    """Convert YOLO-OBB centre/size/angle to Label Studio percentage coords."""
    return {
        "x": (cx - w / 2) / img_w * 100.0,
        "y": (cy - h / 2) / img_h * 100.0,
        "width": w / img_w * 100.0,
        "height": h / img_h * 100.0,
        "rotation": math.degrees(angle_rad),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Auto-annotate images with YOLO-OBB.")
    ap.add_argument("--model", required=True, help="Path to trained OBB weights (.pt).")
    ap.add_argument("--images", type=Path, default=Path("mydata/Candies"),
                    help="Folder with unlabelled images.")
    ap.add_argument("--output", type=Path, default=Path("auto_annotations.json"),
                    help="Output JSON for Label Studio import.")
    ap.add_argument("--output-img-dir", type=Path, default=Path("auto_annotated_images"),
                    help="Folder for visualised predictions.")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    ap.add_argument("--local-files-root", type=Path, default=Path("mydata"),
                    help="LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT for relative paths.")
    args = ap.parse_args()

    model = YOLO(args.model)

    img_dir = args.images.resolve()
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")

    out_img_dir = args.output_img_dir.resolve()
    out_img_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
    )
    print(f"Found {len(image_paths)} images. Starting inference …")

    results = model.predict(
        source=[str(p) for p in image_paths],
        conf=args.conf,
        save=True,
        project=out_img_dir.parent.name,
        name=out_img_dir.name,
        exist_ok=True,
    )

    ls_tasks: list[dict] = []

    for img_path, res in zip(image_paths, results, strict=True):
        if res.obb is None or len(res.obb) == 0:
            continue

        img_h, img_w = res.orig_shape
        boxes = res.obb.xywhr.cpu().numpy()   # (N, 5): cx, cy, w, h, angle
        classes = res.obb.cls.cpu().numpy()    # (N,)
        scores = res.obb.conf.cpu().numpy()    # (N,)

        annotations: list[dict] = []
        for box, cls_id, score in zip(boxes, classes, scores, strict=True):
            cx, cy, w, h, angle_rad = (float(v) for v in box)
            cls_idx = int(cls_id)
            cls_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f"class_{cls_idx}"

            value = _obb_to_ls(cx, cy, w, h, angle_rad, img_w, img_h)
            value["rectanglelabels"] = [cls_name]

            annotations.append({
                "original_width": int(img_w),
                "original_height": int(img_h),
                "image_rotation": 0,
                "value": {k: float(v) if isinstance(v, float) else v for k, v in value.items()},
                "id": uuid.uuid4().hex[:10],
                "from_name": "bbox",
                "to_name": "image",
                "type": "rectanglelabels",
                "origin": "manual",
                "score": float(score),
            })

        # Build relative URL for Label Studio local storage
        try:
            rel = img_path.absolute().relative_to(args.local_files_root.resolve())
        except ValueError:
            raise ValueError(
                f"{img_path.absolute()} is outside local-files-root "
                f"{args.local_files_root.resolve()}. Label Studio requires relative paths."
            )

        ls_tasks.append({
            "data": {"image": f"/data/local-files/?d={rel.as_posix()}"},
            "predictions": [{"model_version": "auto-yolo-v1", "result": annotations}],
        })

    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(ls_tasks, fh, indent=2)

    print(f"Label Studio JSON written: {args.output}")
    print(f"Visual predictions saved:  {out_img_dir.absolute()}")


if __name__ == "__main__":
    main()
