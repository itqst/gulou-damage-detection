"""
Train additional YOLO comparison models on the run9 universal dataset.

These runs are supplementary experiments for model-version comparison. They do
not modify the existing single-case or universal YOLOv8n models.
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN9 = ROOT / "run9"
COMPARE = RUN9 / "model_compare"
DATA_YAML = RUN9 / "datasets" / "universal" / "data.yaml"
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))

from ultralytics import YOLO  # noqa: E402


MODELS = {
    "yolov9t": "yolov9t.pt",
    "yolov10n": "yolov10n.pt",
    "yolo11n": "yolo11n.pt",
}


def safe_reset_dir(path: Path) -> None:
    resolved = path.resolve()
    allowed = COMPARE.resolve()
    if allowed not in resolved.parents and resolved != allowed:
        raise RuntimeError(f"Refusing to reset outside run9/model_compare: {resolved}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def train_one(key: str, overwrite: bool) -> None:
    model_name = MODELS[key]
    project_dir = COMPARE / key / "model"
    run_dir = project_dir / "train"

    if not DATA_YAML.exists():
        raise FileNotFoundError(DATA_YAML)
    if run_dir.exists() and not overwrite:
        print(f"[skip] {key}: {run_dir} already exists")
        return
    if overwrite:
        safe_reset_dir(project_dir)

    print(f"[train] {key}")
    print(f"  data: {DATA_YAML}")
    print(f"  base: {model_name}")
    print(f"  out : {run_dir}")
    print("  config: epochs=150, batch=8, imgsz=640, patience=30, device=0")

    model = YOLO(model_name)
    model.train(
        data=str(DATA_YAML),
        epochs=150,
        imgsz=640,
        batch=8,
        augment=True,
        patience=30,
        workers=0,
        device=0,
        project=str(project_dir),
        name="train",
        exist_ok=True,
        val=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=MODELS.keys(), default=list(MODELS.keys()))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    COMPARE.mkdir(parents=True, exist_ok=True)
    for key in args.models:
        train_one(key, args.overwrite)


if __name__ == "__main__":
    main()
