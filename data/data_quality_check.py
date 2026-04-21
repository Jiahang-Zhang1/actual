from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests


VALID_CATEGORIES = {
    "Food & Dining",
    "Transportation",
    "Shopping & Retail",
    "Entertainment & Recreation",
    "Healthcare & Medical",
    "Healthcare",
    "Utilities & Services",
    "Bills & Utilities",
    "Financial Services",
    "Income",
    "Travel",
    "Other",
    "Charity & Donations",
    "Government & Legal",
    "Transfer",
    "Groceries",
}


def read_frame(path: str | Path) -> pd.DataFrame:
    data_path = Path(path)
    if data_path.suffix == ".csv":
        return pd.read_csv(data_path)
    if data_path.suffix in {".json", ".jsonl"}:
        lines = [
            json.loads(line)
            for line in data_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return pd.DataFrame(lines)
    return pd.read_parquet(data_path)


def result(stage: str, passed: bool, metrics: dict[str, Any], issues: list[str]) -> dict[str, Any]:
    checked_at = datetime.now(timezone.utc).isoformat()
    return {
        "stage": stage,
        "check_type": stage,
        "passed": passed,
        "issue_count": len(issues),
        "issues": issues,
        "metrics": metrics,
        "checked_at": checked_at,
        "timestamp": checked_at,
    }


def check_ingestion_quality(filepath: str | Path) -> dict[str, Any]:
    print(f"\n=== Ingestion Quality Check: {filepath} ===")
    df = read_frame(filepath)
    issues: list[str] = []
    required_cols = ["transaction_description", "category", "country", "currency"]

    missing_cols = [column for column in required_cols if column not in df.columns]
    if missing_cols:
        issues.append(f"missing columns: {missing_cols}")
    else:
        print("[PASS] All required columns present")

    null_counts: dict[str, int] = {}
    for column in required_cols:
        if column in df.columns:
            null_count = int(df[column].isna().sum())
            null_counts[column] = null_count
            if null_count:
                issues.append(f"{column} has {null_count} null values")
            else:
                print(f"[PASS] No nulls in '{column}'")

    # The data branch tightened this gate to make ingestion monitoring useful
    # for retraining jobs rather than only tiny smoke-test fixtures.
    if len(df) < 1000:
        issues.append(f"too few rows for ingestion monitoring: {len(df)}")
    else:
        print(f"[PASS] Row count: {len(df)}")

    invalid_categories: list[str] = []
    if "category" in df.columns:
        invalid_categories = sorted(
            str(value)
            for value in df.loc[~df["category"].isin(VALID_CATEGORIES), "category"]
            .dropna()
            .unique()
        )
        if invalid_categories:
            print(f"[WARN] Unknown categories found: {invalid_categories[:5]}")
            issues.append(f"unknown categories: {invalid_categories[:10]}")
        else:
            print("[PASS] All categories valid")

    metrics = {
        "row_count": int(len(df)),
        "required_columns": len(missing_cols) == 0,
        "missing_column_count": len(missing_cols),
        "null_counts": null_counts,
        "unknown_categories": invalid_categories[:5],
        "invalid_category_count": len(invalid_categories),
        "description_null_rate": float(df.get("transaction_description", pd.Series(dtype=float)).isna().mean())
        if "transaction_description" in df.columns
        else 1.0,
    }
    return result("ingestion", not issues, metrics, issues)


def check_training_set_quality(train_path: str | Path, eval_path: str | Path | None = None) -> dict[str, Any]:
    print("\n=== Training Set Quality Check ===")
    train_df = read_frame(train_path)
    eval_df = read_frame(eval_path) if eval_path else pd.DataFrame()
    issues: list[str] = []

    required_cols = {"transaction_description", "category"}
    dataframes = {"train": train_df}
    if eval_path:
        dataframes["eval"] = eval_df
    for name, df in dataframes.items():
        missing = sorted(required_cols - set(df.columns))
        if missing:
            issues.append(f"{name} missing columns: {missing}")

    overlap_ratio = 0.0
    min_label_ratio = 0.0
    train_ratio = 0.0
    if "category" in train_df.columns:
        label_distribution = train_df["category"].value_counts(normalize=True)
        min_label_ratio = float(label_distribution.min()) if not label_distribution.empty else 0.0
        if min_label_ratio < 0.01:
            issues.append(f"severe class imbalance, min label ratio={min_label_ratio:.3f}")
        else:
            print(f"[PASS] Label distribution acceptable, min ratio: {min_label_ratio:.3f}")

    if eval_path and not issues:
        key_columns = [
            column
            for column in ["transaction_description", "category", "country", "transaction_date", "amount"]
            if column in train_df.columns and column in eval_df.columns
        ]
        # Repeated merchants are normal in bank transactions, so the leakage
        # check uses exact transaction-like keys when date/amount are present.
        train_keys = set(tuple(row) for row in train_df[key_columns].astype(str).itertuples(index=False, name=None))
        eval_keys = set(tuple(row) for row in eval_df[key_columns].astype(str).itertuples(index=False, name=None))
        overlap_ratio = len(train_keys.intersection(eval_keys)) / max(len(eval_keys), 1)
        if overlap_ratio > 0.60:
            issues.append(f"possible leakage, train/eval overlap={overlap_ratio:.3f}")
        else:
            print(f"[PASS] Overlap ratio acceptable: {overlap_ratio:.3f}")

        total = len(train_df) + len(eval_df)
        train_ratio = len(train_df) / max(total, 1)
        if train_ratio < 0.70 or train_ratio > 0.90:
            issues.append(f"unexpected train/eval split ratio={train_ratio:.3f}")
        else:
            print(f"[PASS] Train/eval split: {train_ratio:.3f}/{1 - train_ratio:.3f}")
    elif not eval_path:
        train_ratio = 1.0

    metrics = {
        "train_row_count": int(len(train_df)),
        "eval_row_count": int(len(eval_df)),
        "overlap_ratio": round(float(overlap_ratio), 6),
        "min_label_ratio": round(float(min_label_ratio), 6),
        "train_ratio": round(float(train_ratio), 6),
    }
    return result("training_set", not issues, metrics, issues)


def check_inference_drift(reference_path: str | Path, inference_path: str | Path) -> dict[str, Any]:
    print("\n=== Inference Drift Check ===")
    reference_df = read_frame(reference_path)
    inference_df = read_frame(inference_path)
    issues: list[str] = []

    drift_score = 0.0
    for column in ["country", "currency"]:
        if column not in reference_df.columns or column not in inference_df.columns:
            issues.append(f"{column} missing for drift check")
            continue

        reference_dist = reference_df[column].fillna("unknown").astype(str).value_counts(normalize=True)
        inference_dist = inference_df[column].fillna("unknown").astype(str).value_counts(normalize=True)
        print(f"[INFO] Reference {column}: {dict(reference_dist.head(3))}")
        print(f"[INFO] Inference {column}: {dict(inference_dist.head(3))}")
        values = set(reference_dist.index).union(set(inference_dist.index))
        column_drift = sum(abs(float(reference_dist.get(v, 0.0)) - float(inference_dist.get(v, 0.0))) for v in values) / 2.0
        drift_score = max(drift_score, column_drift)
        if column_drift > 0.20:
            issues.append(f"{column} drift score={column_drift:.3f}")

    metrics = {
        "reference_row_count": int(len(reference_df)),
        "inference_row_count": int(len(inference_df)),
        "drift_score": round(float(drift_score), 6),
    }
    return result("online_drift", not issues, metrics, issues)


def write_json(path: str | Path | None, payload: dict[str, Any]) -> None:
    if not path:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_prometheus(path: str | Path | None, payload: dict[str, Any]) -> None:
    if not path:
        return
    stage = payload["stage"]
    lines = [
        "# HELP actual_data_quality_pass Whether the latest data quality check passed.",
        "# TYPE actual_data_quality_pass gauge",
        f'actual_data_quality_pass{{stage="{stage}"}} {1 if payload["passed"] else 0}',
        "# HELP actual_data_quality_issue_count Number of issues found by the latest data quality check.",
        "# TYPE actual_data_quality_issue_count gauge",
        f'actual_data_quality_issue_count{{stage="{stage}"}} {payload["issue_count"]}',
    ]
    for key, value in payload.get("metrics", {}).items():
        if isinstance(value, (int, float, bool)):
            lines.append(f'actual_data_quality_metric{{stage="{stage}",metric="{key}"}} {int(value) if isinstance(value, bool) else value}')
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def post_result(url: str | None, payload: dict[str, Any]) -> None:
    if not url:
        return
    response = requests.post(url, json=payload, timeout=10)
    response.raise_for_status()
    print(f"[INFO] Posted results to {url}: {response.status_code}")


def parse_args() -> argparse.Namespace:
    # Support the data branch's legacy no-argument behavior while keeping the
    # serving branch's explicit subcommands for Kubernetes CronJobs.
    if len(sys.argv) == 1:
        sys.argv.append("all")

    parser = argparse.ArgumentParser(description="Run Actual ML data quality checks.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingestion")
    p_ingest.add_argument("--input")

    p_training = sub.add_parser("training-set", aliases=["training_set"])
    p_training.add_argument("--train", "--input", dest="train_path")
    p_training.add_argument("--eval", "--eval-input", dest="eval_path")

    p_drift = sub.add_parser("online-drift", aliases=["online_drift"])
    p_drift.add_argument("--reference")
    p_drift.add_argument("--inference", "--input", dest="inference")

    p_all = sub.add_parser("all")
    p_all.add_argument("--data-path", default=os.getenv("DATA_PATH", "/home/cc"))

    for command in [p_ingest, p_training, p_drift, p_all]:
        command.add_argument("--output-json")
        command.add_argument("--output-prom")
        command.add_argument("--post-url")
        command.add_argument("--fail-on-error", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(os.getenv("DATA_PATH", "/home/cc"))

    if args.command == "ingestion":
        payload = check_ingestion_quality(args.input or data_path / "transactions_clean_v1.parquet")
    elif args.command in {"training-set", "training_set"}:
        payload = check_training_set_quality(
            args.train_path or data_path / "train_data.csv",
            args.eval_path or data_path / "eval_data.csv",
        )
    elif args.command in {"online-drift", "online_drift"}:
        payload = check_inference_drift(
            args.reference or data_path / "transactions_clean_v1.parquet",
            args.inference or data_path / "online_features_output.csv",
        )
    elif args.command == "all":
        checks = [
            check_ingestion_quality(data_path / "transactions_clean_v1.parquet"),
            check_training_set_quality(data_path / "train_data.csv", data_path / "eval_data.csv"),
            check_inference_drift(
                data_path / "transactions_clean_v1.parquet",
                data_path / "online_features_output.csv",
            ),
        ]
        payload = result(
            "all",
            all(check["passed"] for check in checks),
            {check["stage"]: check["metrics"] for check in checks},
            [issue for check in checks for issue in check["issues"]],
        )
        payload["checks"] = checks
    else:
        raise ValueError(args.command)

    write_json(args.output_json, payload)
    write_prometheus(args.output_prom, payload)
    if args.command == "all" and args.post_url:
        for check in payload["checks"]:
            post_result(args.post_url, check)
    else:
        post_result(args.post_url, payload)
    print(json.dumps(payload, indent=2))

    if args.fail_on_error and not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
