#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import requests


def run(command: list[str], cwd: Path) -> None:
    print("$", " ".join(command))
    subprocess.run(command, cwd=str(cwd), check=True)


def split_training_csv(source: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    rows = list(csv.DictReader(source.open("r", encoding="utf-8")))
    if len(rows) < 30:
        raise ValueError(f"Need at least 30 rows for train/val/test split, got {len(rows)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_end = int(len(rows) * 0.70)
    val_end = int(len(rows) * 0.85)
    splits = {
        "train": rows[:train_end],
        "val": rows[train_end:val_end],
        "test": rows[val_end:],
    }

    paths: dict[str, Path] = {}
    for name, split_rows in splits.items():
        path = output_dir / f"{name}.csv"
        paths[name] = path
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(split_rows)
    return paths["train"], paths["val"], paths["test"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrain -> evaluate -> register/promote pipeline.")
    parser.add_argument("--workspace-dir", default="artifacts/mlops-pipeline")
    parser.add_argument("--source-training-csv", default="artifacts/test-data/synthetic_training_transactions.csv")
    parser.add_argument("--deployed-dir", default="serving/runtime/deployed")
    parser.add_argument("--archive-dir", default="serving/runtime/archive")
    parser.add_argument("--data-quality-post-url")
    parser.add_argument("--reload-url")
    parser.add_argument("--synthetic-bootstrap", action="store_true")
    parser.add_argument("--min-top3-accuracy", type=float, default=0.70)
    parser.add_argument("--min-macro-f1", type=float, default=0.55)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    workspace = (repo / args.workspace_dir).resolve()
    source_training_csv = (repo / args.source_training_csv).resolve()
    deployed_dir = (repo / args.deployed_dir).resolve()
    archive_dir = (repo / args.archive_dir).resolve()
    challenger_dir = workspace / "challenger"

    if args.synthetic_bootstrap or not source_training_csv.exists():
        run(
            [
                sys.executable,
                "scripts/generate_actual_bank_import.py",
                "--rows",
                "1200",
                "--training-output",
                str(source_training_csv),
            ],
            repo,
        )

    dataset_dir = workspace / "datasets"
    train_csv, val_csv, test_csv = split_training_csv(source_training_csv, dataset_dir)

    quality_args = [
        sys.executable,
        "data/data_quality_check.py",
        "training-set",
        "--train",
        str(train_csv),
        "--eval",
        str(val_csv),
        "--output-json",
        str(workspace / "quality" / "training_set.json"),
        "--fail-on-error",
    ]
    if args.data_quality_post_url:
        quality_args.extend(["--post-url", args.data_quality_post_url])
    run(quality_args, repo)

    shutil.rmtree(challenger_dir, ignore_errors=True)
    run(
        [
            sys.executable,
            "training/train_model.py",
            "--train-dataset",
            str(train_csv),
            "--val-dataset",
            str(val_csv),
            "--test-dataset",
            str(test_csv),
            "--output-dir",
            str(challenger_dir),
            "--run-name",
            "automated-retrain",
        ],
        repo,
    )

    run(
        [
            sys.executable,
            "scripts/export_model_variants.py",
            "--model-dir",
            str(challenger_dir),
            "--sample-dataset",
            str(test_csv),
        ],
        repo,
    )

    run(
        [
            sys.executable,
            "scripts/promote_model.py",
            "--challenger-dir",
            str(challenger_dir),
            "--deployed-dir",
            str(deployed_dir),
            "--archive-dir",
            str(archive_dir),
            "--min-top3-accuracy",
            str(args.min_top3_accuracy),
            "--min-macro-f1",
            str(args.min_macro_f1),
        ],
        repo,
    )

    decision_path = archive_dir / "last_decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    reload_result = None
    if args.reload_url and decision.get("promoted"):
        response = requests.post(args.reload_url, timeout=10)
        response.raise_for_status()
        reload_result = response.json()
    print(json.dumps({"decision": decision, "reload": reload_result}, indent=2))


if __name__ == "__main__":
    main()
