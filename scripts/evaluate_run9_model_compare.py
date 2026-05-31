"""
Evaluate run9 model-version comparison runs on universal and per-case tests.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN9 = ROOT / "run9"
COMPARE = RUN9 / "model_compare"
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))

from ultralytics import YOLO  # noqa: E402


MODELS = {
    "yolov8n": RUN9 / "universal" / "model" / "train" / "weights" / "best.pt",
    "yolov9t": COMPARE / "yolov9t" / "model" / "train" / "weights" / "best.pt",
    "yolov10n": COMPARE / "yolov10n" / "model" / "train" / "weights" / "best.pt",
    "yolo11n": COMPARE / "yolo11n" / "model" / "train" / "weights" / "best.pt",
}

DATASETS = {
    "universal": RUN9 / "datasets" / "universal" / "data.yaml",
    "zengchong": RUN9 / "datasets" / "zengchong" / "data.yaml",
    "chaoli": RUN9 / "datasets" / "chaoli" / "data.yaml",
    "congjiang": RUN9 / "datasets" / "congjiang" / "data.yaml",
}


def metrics_to_dict(model_key: str, dataset_key: str, model_path: Path, data_yaml: Path, metrics) -> dict:
    box = metrics.box
    return {
        "model": model_key,
        "dataset": dataset_key,
        "model_path": str(model_path),
        "data": str(data_yaml),
        "precision": float(box.mp),
        "recall": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
        "fitness": float(metrics.fitness),
        "save_dir": str(metrics.save_dir),
        "speed": getattr(metrics, "speed", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=MODELS.keys(), default=list(MODELS.keys()))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS.keys(), default=list(DATASETS.keys()))
    args = parser.parse_args()

    project = COMPARE / "evaluation"
    project.mkdir(parents=True, exist_ok=True)
    summaries = []

    for model_key in args.models:
        model_path = MODELS[model_key]
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        model = YOLO(str(model_path))
        for dataset_key in args.datasets:
            data_yaml = DATASETS[dataset_key]
            if not data_yaml.exists():
                raise FileNotFoundError(data_yaml)
            name = f"{model_key}_on_{dataset_key}_test"
            print(f"[eval] {name}")
            metrics = model.val(
                data=str(data_yaml),
                split="test",
                imgsz=640,
                batch=8,
                device=0,
                workers=0,
                project=str(project),
                name=name,
                exist_ok=True,
                plots=True,
            )
            summary = metrics_to_dict(model_key, dataset_key, model_path, data_yaml, metrics)
            summaries.append(summary)
            out_json = project / name / "metrics_summary.json"
            out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(summary, ensure_ascii=False, indent=2))

    summary_path = project / "all_model_compare_metrics.json"
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {summary_path}")


if __name__ == "__main__":
    main()
