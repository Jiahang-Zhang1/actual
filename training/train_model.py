import argparse
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer


def read_dataset(path: str) -> pd.DataFrame:
    dataset_path = Path(path)
    if dataset_path.suffix == ".csv":
        return pd.read_csv(dataset_path)
    return pd.read_parquet(dataset_path)


def amount_bucket(value) -> str:
    if value is None or value == "":
        return "unknown"
    try:
        amount = abs(float(value))
    except (TypeError, ValueError):
        return "unknown"
    if amount < 20:
        return "micro"
    if amount < 100:
        return "small"
    if amount < 500:
        return "medium"
    return "large"


def weekday_token(value) -> str:
    if not value:
        return "unknown"
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return "unknown"
    return parsed.day_name().lower()


def build_feature_text(row: pd.Series) -> str:
    parts = [
        str(row.get("transaction_description", "")).strip().lower(),
        f"country={str(row.get('country', 'unknown')).strip().lower() or 'unknown'}",
        f"currency={str(row.get('currency', 'unknown')).strip().lower() or 'unknown'}",
        f"amount_bucket={amount_bucket(row.get('amount'))}",
        f"weekday={weekday_token(row.get('transaction_date'))}",
    ]

    account_id = str(row.get("account_id", "") or "").strip().lower()
    if account_id:
        parts.append(f"account={account_id}")

    imported_description = str(row.get("imported_description", "") or "").strip().lower()
    if imported_description:
        parts.append(f"imported={imported_description}")

    notes = str(row.get("notes", "") or "").strip().lower()
    if notes:
        parts.append(f"notes={notes}")

    return " ".join(part for part in parts if part)


def build_feature_series(df: pd.DataFrame):
    return df.apply(build_feature_text, axis=1)


def build_pipeline(max_word_features: int, max_char_features: int, c_value: float) -> Pipeline:
    return Pipeline(
        steps=[
            ("feature_text", FunctionTransformer(build_feature_series, validate=False)),
            (
                "vectorizer",
                FeatureUnion(
                    transformer_list=[
                        (
                            "word",
                            TfidfVectorizer(
                                analyzer="word",
                                ngram_range=(1, 2),
                                max_features=max_word_features,
                                min_df=1,
                                sublinear_tf=True,
                            ),
                        ),
                        (
                            "char",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                ngram_range=(3, 5),
                                max_features=max_char_features,
                                min_df=1,
                                sublinear_tf=True,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    C=c_value,
                    class_weight="balanced",
                    n_jobs=1,
                ),
            ),
        ]
    )


def top_k_accuracy(probabilities, labels, classes, k: int) -> float:
    label_to_index = {label: idx for idx, label in enumerate(classes)}
    total = 0
    hits = 0
    for probs, label in zip(probabilities, labels):
        total += 1
        target_idx = label_to_index[label]
        top_k = probs.argsort()[-k:][::-1]
        if target_idx in top_k:
            hits += 1
    return hits / total if total else 0.0


def evaluate_model(model, df: pd.DataFrame) -> dict:
    probabilities = model.predict_proba(df)
    predictions = model.predict(df)
    labels = df["category"].astype(str)
    classes = list(model.named_steps["clf"].classes_)
    return {
        "row_count": int(len(df)),
        "top1_accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "top3_accuracy": float(top_k_accuracy(probabilities, labels, classes, k=3)),
    }


def maybe_start_mlflow(run_name: str):
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("actual-smart-transaction-categorization")
    return mlflow.start_run(run_name=run_name, log_system_metrics=True)


def main():
    parser = argparse.ArgumentParser(description="Train the Actual smart transaction categorizer.")
    parser.add_argument("--train-dataset", required=True)
    parser.add_argument("--val-dataset", required=True)
    parser.add_argument("--test-dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--external-dataset", help="Optional CSV/parquet dataset used for pretraining or warm-start supervision.")
    parser.add_argument("--run-name", default="smart-transaction-categorizer")
    parser.add_argument("--max-word-features", type=int, default=25000)
    parser.add_argument("--max-char-features", type=int, default=25000)
    parser.add_argument("--c-value", type=float, default=2.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = read_dataset(args.train_dataset)
    val_df = read_dataset(args.val_dataset)
    test_df = read_dataset(args.test_dataset)

    if args.external_dataset:
        external_df = read_dataset(args.external_dataset)
        train_df = pd.concat([external_df, train_df], ignore_index=True, sort=False)
    else:
        external_df = None

    required_columns = {"transaction_description", "category"}
    for name, df in {"train": train_df, "val": val_df, "test": test_df}.items():
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"{name} dataset is missing columns: {sorted(missing)}")

    run = maybe_start_mlflow(args.run_name)
    with run:
        mlflow.log_params(
            {
                "train_rows": len(train_df),
                "val_rows": len(val_df),
                "test_rows": len(test_df),
                "external_rows": 0 if external_df is None else len(external_df),
                "max_word_features": args.max_word_features,
                "max_char_features": args.max_char_features,
                "c_value": args.c_value,
                "python_version": platform.python_version(),
                "platform": platform.platform(),
            }
        )

        model = build_pipeline(
            max_word_features=args.max_word_features,
            max_char_features=args.max_char_features,
            c_value=args.c_value,
        )
        model.fit(train_df, train_df["category"].astype(str))

        val_metrics = evaluate_model(model, val_df)
        test_metrics = evaluate_model(model, test_df)
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items() if k != "row_count"})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items() if k != "row_count"})

        model_version = datetime.now(timezone.utc).strftime("v%Y%m%d%H%M%S")

        joblib.dump(model, output_dir / "model.joblib")
        metadata = {
            "model_version": model_version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metrics": {
                "validation": val_metrics,
                "test": test_metrics,
            },
            "classes": list(model.named_steps["clf"].classes_),
        }
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        (output_dir / "metrics.json").write_text(json.dumps(test_metrics, indent=2))

        mlflow.log_artifacts(str(output_dir), artifact_path="serving_bundle")
        mlflow.sklearn.log_model(sk_model=model, artifact_path="sk_model")

        print(json.dumps({"model_version": model_version, "test_metrics": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
