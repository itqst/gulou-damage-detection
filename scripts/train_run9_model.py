"""
Train one final run9 model with the shared report configuration.

Run this script sequentially for zengchong, chaoli, congjiang, and universal.
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN9 = ROOT / "run9"
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))

from ultralytics import YOLO  # noqa: E402


CASES = {
    "zengchong": RUN9 / "datasets" / "zengchong" / "data.yaml",
    "chaoli": RUN9 / "datasets" / "chaoli" / "data.yaml",
    "congjiang": RUN9 / "datasets" / "congjiang" / "data.yaml",
    "universal": RUN9 / "datasets" / "universal" / "data.yaml",
}


def safe_reset_dir(path: Path) -> None:
    resolved = path.resolve()
    allowed = RUN9.resolve()
    if allowed not in resolved.parents and resolved != allowed:
        raise RuntimeError(f"Refusing to reset outside run9: {resolved}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES.keys(), required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data_yaml = CASES[args.case]
    base_model = ROOT / "yolov8n.pt"
    project_dir = RUN9 / args.case / "model"
    run_dir = project_dir / "train"

    if not data_yaml.exists():
        raise FileNotFoundError(data_yaml)
    if not base_model.exists():
        raise FileNotFoundError(base_model)
    if run_dir.exists() and not args.overwrite:
        raise FileExistsError(f"{run_dir} exists. Use --overwrite for a clean rerun.")
    if args.overwrite:
        safe_reset_dir(project_dir)

    print(f"Final run9 training case: {args.case}")
    print(f"Data: {data_yaml}")
    print(f"Base model: {base_model}")
    print(f"Output: {run_dir}")
    print("Shared config: yolov8n.pt, epochs=150, batch=8, imgsz=640, patience=30")

    model = YOLO(str(base_model))
    model.train(
        data=str(data_yaml),
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


if __name__ == "__main__":
    main()
