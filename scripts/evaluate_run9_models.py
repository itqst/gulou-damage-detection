"""
Evaluate final run9 models on held-out test sets.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN9 = ROOT / "run9"
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))

from ultralytics import YOLO  # noqa: E402


JOBS = [
    {
        "name": "zengchong_single_on_zengchong_test",
        "model": RUN9 / "zengchong" / "model" / "train" / "weights" / "best.pt",
        "data": RUN9 / "datasets" / "zengchong" / "data.yaml",
        "model_case": "zengchong",
        "data_case": "zengchong",
        "kind": "single",
    },
    {
        "name": "chaoli_single_on_chaoli_test",
        "model": RUN9 / "chaoli" / "model" / "train" / "weights" / "best.pt",
        "data": RUN9 / "datasets" / "chaoli" / "data.yaml",
        "model_case": "chaoli",
        "data_case": "chaoli",
        "kind": "single",
    },
    {
        "name": "congjiang_single_on_congjiang_test",
        "model": RUN9 / "congjiang" / "model" / "train" / "weights" / "best.pt",
        "data": RUN9 / "datasets" / "congjiang" / "data.yaml",
        "model_case": "congjiang",
        "data_case": "congjiang",
        "kind": "single",
    },
    {
        "name": "universal_on_universal_test",
        "model": RUN9 / "universal" / "model" / "train" / "weights" / "best.pt",
        "data": RUN9 / "datasets" / "universal" / "data.yaml",
        "model_case": "universal",
        "data_case": "universal",
        "kind": "universal",
    },
    {
        "name": "universal_on_zengchong_test",
        "model": RUN9 / "universal" / "model" / "train" / "weights" / "best.pt",
        "data": RUN9 / "datasets" / "zengchong" / "data.yaml",
        "model_case": "universal",
        "data_case": "zengchong",
        "kind": "universal_case",
    },
    {
        "name": "universal_on_chaoli_test",
        "model": RUN9 / "universal" / "model" / "train" / "weights" / "best.pt",
        "data": RUN9 / "datasets" / "chaoli" / "data.yaml",
        "model_case": "universal",
        "data_case": "chaoli",
        "kind": "universal_case",
    },
    {
        "name": "universal_on_congjiang_test",
        "model": RUN9 / "universal" / "model" / "train" / "weights" / "best.pt",
        "data": RUN9 / "datasets" / "congjiang" / "data.yaml",
        "model_case": "universal",
        "data_case": "congjiang",
        "kind": "universal_case",
    },
]


def metrics_to_dict(job: dict, metrics) -> dict:
    box = metrics.box
    result = {
        "name": job["name"],
        "model_case": job["model_case"],
        "data_case": job["data_case"],
        "kind": job["kind"],
        "model": str(job["model"]),
        "data": str(job["data"]),
        "precision": float(box.mp),
        "recall": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
        "fitness": float(metrics.fitness),
        "save_dir": str(metrics.save_dir),
    }
    if hasattr(metrics, "speed"):
        result["speed"] = metrics.speed
    return result


def main() -> None:
    project = RUN9 / "evaluation"
    project.mkdir(parents=True, exist_ok=True)
    summaries = []

    for job in JOBS:
        if not Path(job["model"]).exists():
            raise FileNotFoundError(job["model"])
        if not Path(job["data"]).exists():
            raise FileNotFoundError(job["data"])
        print(f"Evaluating: {job['name']}")
        model = YOLO(str(job["model"]))
        metrics = model.val(
            data=str(job["data"]),
            split="test",
            imgsz=640,
            batch=8,
            device=0,
            workers=0,
            project=str(project),
            name=job["name"],
            exist_ok=True,
            plots=True,
        )
        summary = metrics_to_dict(job, metrics)
        summaries.append(summary)
        out_json = project / job["name"] / "metrics_summary.json"
        out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    summary_path = project / "all_metrics_summary.json"
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"All evaluation summaries written to: {summary_path}")


if __name__ == "__main__":
    main()
