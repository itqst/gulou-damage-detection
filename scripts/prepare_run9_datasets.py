"""
Prepare isolated datasets for the final run9 experiments.

All final training jobs read from run9/datasets only. This prevents accidental
changes to the working experiment folders from affecting report-ready runs.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN9 = ROOT / "run9"
DATASETS = RUN9 / "datasets"


SOURCE_DATASETS = {
    "zengchong": ROOT / "gulou_project",
    "chaoli": ROOT / "experiments" / "dataset_gulou_02_chaoli_with_neg",
    "congjiang": ROOT / "experiments" / "dataset_gulou_03_congjiang_with_neg",
}


def safe_reset_dir(path: Path) -> None:
    resolved = path.resolve()
    allowed = RUN9.resolve()
    if allowed not in resolved.parents and resolved != allowed:
        raise RuntimeError(f"Refusing to reset outside run9: {resolved}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_split_dataset(source: Path, target: Path, prefix: str | None = None) -> dict:
    summary = {
        "images": 0,
        "positive_images": 0,
        "negative_images": 0,
        "boxes": 0,
        "splits": {},
    }
    for split in ["train", "val", "test"]:
        src_img_dir = source / split / "images"
        src_lbl_dir = source / split / "labels"
        dst_img_dir = target / split / "images"
        dst_lbl_dir = target / split / "labels"
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        split_summary = {
            "images": 0,
            "positive_images": 0,
            "negative_images": 0,
            "boxes": 0,
        }
        for image_path in sorted(src_img_dir.glob("*")):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            label_path = src_lbl_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                raise FileNotFoundError(label_path)

            out_stem = f"{prefix}__{image_path.stem}" if prefix else image_path.stem
            out_img = dst_img_dir / f"{out_stem}{image_path.suffix.lower()}"
            out_lbl = dst_lbl_dir / f"{out_stem}.txt"
            shutil.copy2(image_path, out_img)
            shutil.copy2(label_path, out_lbl)

            lines = [line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            box_count = len(lines)
            split_summary["images"] += 1
            split_summary["boxes"] += box_count
            if box_count:
                split_summary["positive_images"] += 1
            else:
                split_summary["negative_images"] += 1

        summary["splits"][split] = split_summary
        for key in ["images", "positive_images", "negative_images", "boxes"]:
            summary[key] += split_summary[key]
    summary["positive_ratio"] = round(summary["positive_images"] / summary["images"], 6)
    return summary


def write_yaml(target: Path) -> None:
    yaml_text = f"""path: {target.as_posix()}
train: train/images
val: val/images
test: test/images

names:
  0: damage
"""
    (target / "data.yaml").write_text(yaml_text, encoding="utf-8")


def prepare_single_datasets() -> dict:
    report = {}
    for case_id, source in SOURCE_DATASETS.items():
        target = DATASETS / case_id
        safe_reset_dir(target)
        summary = copy_split_dataset(source, target)
        write_yaml(target)
        (target / "dataset_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        report[case_id] = summary
    return report


def prepare_universal_dataset() -> dict:
    target = DATASETS / "universal"
    safe_reset_dir(target)
    summary = {
        "images": 0,
        "positive_images": 0,
        "negative_images": 0,
        "boxes": 0,
        "by_case": {},
        "splits": {
            "train": {"images": 0, "positive_images": 0, "negative_images": 0, "boxes": 0},
            "val": {"images": 0, "positive_images": 0, "negative_images": 0, "boxes": 0},
            "test": {"images": 0, "positive_images": 0, "negative_images": 0, "boxes": 0},
        },
    }
    for case_id in ["zengchong", "chaoli", "congjiang"]:
        case_source = DATASETS / case_id
        case_summary = copy_split_dataset(case_source, target, prefix=case_id)
        summary["by_case"][case_id] = case_summary
        for key in ["images", "positive_images", "negative_images", "boxes"]:
            summary[key] += case_summary[key]
        for split in ["train", "val", "test"]:
            for key in ["images", "positive_images", "negative_images", "boxes"]:
                summary["splits"][split][key] += case_summary["splits"][split][key]
    summary["positive_ratio"] = round(summary["positive_images"] / summary["images"], 6)
    write_yaml(target)
    (target / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    RUN9.mkdir(parents=True, exist_ok=True)
    (RUN9 / "logs").mkdir(parents=True, exist_ok=True)
    (RUN9 / "docs").mkdir(parents=True, exist_ok=True)

    report = prepare_single_datasets()
    report["universal"] = prepare_universal_dataset()
    (DATASETS / "all_dataset_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
