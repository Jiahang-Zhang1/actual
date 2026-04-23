import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from serving.app.postprocess import apply_confidence_policy
from serving.app.taxonomy import canonicalize_category, taxonomy_manifest


def read_dataset(path: str) -> pd.DataFrame:
    dataset_path = Path(path)
    if dataset_path.suffix == ".csv":
        return pd.read_csv(dataset_path)
    return pd.read_parquet(dataset_path)


def prepare_labeled_dataset(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    if df.empty:
        raise ValueError(f"{dataset_name} dataset is empty.")
    if "category" not in df.columns:
        return df

    prepared = df.copy()
    prepared["category_raw"] = prepared["category"].fillna("").astype(str).str.strip()
    prepared["category"] = prepared["category"].map(canonicalize_category)
    unsupported = prepared["category"].isna()
    if unsupported.any():
        unknown_labels = sorted(
            value
            for value in prepared.loc[unsupported, "category_raw"].dropna().unique().tolist()
            if value
        )
        raise ValueError(
            f"{dataset_name} dataset contains unsupported labels: {unknown_labels[:10]}"
        )
    return prepared


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


def clean_text(value) -> str:
    return str(value or "").strip()


def choose_description(row: pd.Series) -> str:
    for column in [
        "transaction_description",
        "transaction_description_clean",
        "merchant_text",
        "imported_description",
        "notes",
    ]:
        cleaned = clean_text(row.get(column))
        if cleaned:
            return cleaned
    return ""


def description_source(row: pd.Series) -> str:
    for column in [
        "transaction_description",
        "transaction_description_clean",
        "merchant_text",
        "imported_description",
        "notes",
    ]:
        if clean_text(row.get(column)):
            return column
    return "derived"


def build_feature_text(row: pd.Series) -> str:
    description = choose_description(row) or "manual entry"
    parts = [
        description.lower(),
        f"description_source={description_source(row)}",
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


def _string_column(df: pd.DataFrame, column: str, default: str) -> pd.Series:
    if column not in df:
        return pd.Series([default] * len(df), index=df.index, dtype=str)
    return df[column].fillna(default).astype(str)


def build_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Keep the trained artifact ONNX-convertible by doing feature text assembly
    # before the sklearn pipeline. The online serving adapter applies the same
    # contract before calling joblib/ONNX variants.
    return pd.DataFrame(
        {
            "transaction_description": df.apply(build_feature_text, axis=1),
            "country": _string_column(df, "country", "US"),
            "currency": _string_column(df, "currency", "USD"),
        }
    )


def build_classifier(model_family: str, c_value: float):
    if model_family == "logreg":
        return LogisticRegression(
            max_iter=1000,
            C=c_value,
            class_weight="balanced",
            n_jobs=1,
        )
    if model_family == "linear_svm":
        return LinearSVC(
            max_iter=3000,
            C=c_value,
            class_weight="balanced",
            random_state=42,
        )
    if model_family == "sgd_log":
        return SGDClassifier(
            loss="log_loss",
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )
    raise ValueError(f"Unsupported model family: {model_family}")


def build_pipeline(
    max_word_features: int,
    max_char_features: int,
    c_value: float,
    model_family: str,
) -> Pipeline:
    # max_char_features is retained for CLI compatibility; the deployed
    # training artifact uses ONNX-friendly word TF-IDF plus categorical inputs.
    return Pipeline(
        steps=[
            (
                "preprocessor",
                ColumnTransformer(
                    transformers=[
                        (
                            "transaction_description_word_tfidf",
                            TfidfVectorizer(
                                analyzer="word",
                                ngram_range=(1, 2),
                                max_features=max_word_features,
                                min_df=1,
                                sublinear_tf=True,
                            ),
                            "transaction_description",
                        ),
                        (
                            "categorical",
                            OneHotEncoder(handle_unknown="ignore"),
                            ["country", "currency"],
                        ),
                    ],
                ),
            ),
            ("clf", build_classifier(model_family, c_value)),
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


def _softmax_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim == 1:
        matrix = np.stack([-matrix, matrix], axis=1)
    matrix = matrix - np.max(matrix, axis=1, keepdims=True)
    exp_matrix = np.exp(matrix)
    denom = np.sum(exp_matrix, axis=1, keepdims=True)
    denom[denom == 0.0] = 1.0
    return exp_matrix / denom


def model_score_matrix(model, frame: pd.DataFrame) -> np.ndarray:
    # Candidate families expose either calibrated probabilities or raw margins.
    # Convert raw margins to comparable probabilities so Top-K gates are stable.
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(frame), dtype=float)
    if hasattr(model, "decision_function"):
        return _softmax_rows(np.asarray(model.decision_function(frame), dtype=float))
    clf = model.named_steps["clf"]
    transformed = model.named_steps["preprocessor"].transform(frame)
    if hasattr(clf, "predict_proba"):
        return np.asarray(clf.predict_proba(transformed), dtype=float)
    if hasattr(clf, "decision_function"):
        return _softmax_rows(np.asarray(clf.decision_function(transformed), dtype=float))
    raise RuntimeError("Candidate model exposes neither predict_proba nor decision_function.")


def multiclass_nll(probabilities: np.ndarray, labels: pd.Series, classes: list[str]) -> float:
    label_to_index = {label: idx for idx, label in enumerate(classes)}
    target_indices = np.array([label_to_index[str(label)] for label in labels], dtype=int)
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-9, 1.0)
    row_indices = np.arange(len(target_indices))
    return float(-np.mean(np.log(clipped[row_indices, target_indices])))


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: pd.Series,
    classes: list[str],
    bins: int = 10,
) -> float:
    matrix = np.asarray(probabilities, dtype=float)
    if len(matrix) == 0:
        return 0.0

    label_to_index = {label: idx for idx, label in enumerate(classes)}
    target_indices = np.array([label_to_index[str(label)] for label in labels], dtype=int)
    pred_indices = np.argmax(matrix, axis=1)
    confidences = np.max(matrix, axis=1)
    correct = (pred_indices == target_indices).astype(float)

    ece = 0.0
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences >= lower) & (
            confidences <= upper if np.isclose(upper, 1.0) else confidences < upper
        )
        if not np.any(mask):
            continue
        bin_accuracy = float(np.mean(correct[mask]))
        bin_confidence = float(np.mean(confidences[mask]))
        ece += abs(bin_accuracy - bin_confidence) * (np.sum(mask) / len(matrix))
    return float(ece)


def multiclass_brier_score(probabilities: np.ndarray, labels: pd.Series, classes: list[str]) -> float:
    matrix = np.asarray(probabilities, dtype=float)
    if len(matrix) == 0:
        return 0.0
    label_to_index = {label: idx for idx, label in enumerate(classes)}
    truth = np.zeros_like(matrix)
    for row_index, label in enumerate(labels):
        truth[row_index, label_to_index[str(label)]] = 1.0
    return float(np.mean(np.sum((matrix - truth) ** 2, axis=1)))


def fit_temperature(probabilities: np.ndarray, labels: pd.Series, classes: list[str]) -> tuple[float, dict]:
    candidate_temperatures = np.concatenate(
        [
            np.linspace(0.7, 2.5, 37),
            np.linspace(2.5, 5.0, 11),
        ]
    )

    best_temperature = 1.0
    best_nll = multiclass_nll(probabilities, labels, classes)
    for temperature in candidate_temperatures:
        candidate = apply_confidence_policy(
            probabilities[0],
            classes,
            description="calibration placeholder",
            description_source="transaction_description",
            metadata={"confidence_policy": {"temperature": float(temperature)}},
        )
        del candidate
        scaled = np.vstack(
            [
                apply_confidence_policy(
                    row,
                    classes,
                    description="calibration placeholder",
                    description_source="transaction_description",
                    metadata={"confidence_policy": {"temperature": float(temperature)}},
                )
                for row in probabilities
            ]
        )
        nll = multiclass_nll(scaled, labels, classes)
        if nll < best_nll:
            best_nll = nll
            best_temperature = float(temperature)

    before = {
        "nll": multiclass_nll(probabilities, labels, classes),
        "ece": expected_calibration_error(probabilities, labels, classes),
        "brier": multiclass_brier_score(probabilities, labels, classes),
    }
    after_matrix = np.vstack(
        [
            apply_confidence_policy(
                row,
                classes,
                description="calibration placeholder",
                description_source="transaction_description",
                metadata={"confidence_policy": {"temperature": best_temperature}},
            )
            for row in probabilities
        ]
    )
    after = {
        "nll": multiclass_nll(after_matrix, labels, classes),
        "ece": expected_calibration_error(after_matrix, labels, classes),
        "brier": multiclass_brier_score(after_matrix, labels, classes),
    }
    return best_temperature, {"before": before, "after": after}


def default_register_model_name() -> str | None:
    configured = os.environ.get("MLFLOW_REGISTER_MODEL_NAME")
    if configured:
        return configured
    if os.environ.get("MLFLOW_TRACKING_URI"):
        return "actual-smart-transaction-categorizer"
    return None


def evaluate_model(model, df: pd.DataFrame) -> dict:
    frame = build_model_frame(df)
    probabilities = model_score_matrix(model, frame)
    predictions = model.predict(frame)
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


def register_model_if_configured(
    run_id: str,
    model_name: str | None,
    metadata: dict,
) -> dict:
    if not model_name:
        return {}

    # Register the candidate in MLflow so promotion/rollback can point to an
    # auditable model version instead of only a copied file on disk.
    model_uri = f"runs:/{run_id}/sk_model"
    registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)
    client = MlflowClient()
    client.set_registered_model_alias(
        name=model_name,
        alias="candidate",
        version=str(registered_model.version),
    )
    client.set_model_version_tag(
        name=model_name,
        version=registered_model.version,
        key="actual.role",
        value="candidate",
    )
    client.set_model_version_tag(
        name=model_name,
        version=registered_model.version,
        key="actual.model_version",
        value=str(metadata["model_version"]),
    )
    if metadata.get("model_family"):
        client.set_model_version_tag(
            name=model_name,
            version=registered_model.version,
            key="actual.model_family",
            value=str(metadata["model_family"]),
        )
    for split_name, split_metrics in metadata["metrics"].items():
        for metric_name, value in split_metrics.items():
            client.set_model_version_tag(
                name=model_name,
                version=registered_model.version,
                key=f"actual.{split_name}.{metric_name}",
                value=str(value),
            )
    if metadata.get("taxonomy", {}).get("taxonomy_version"):
        client.set_model_version_tag(
            name=model_name,
            version=registered_model.version,
            key="actual.taxonomy_version",
            value=str(metadata["taxonomy"]["taxonomy_version"]),
        )
    if metadata.get("confidence_policy", {}).get("temperature"):
        client.set_model_version_tag(
            name=model_name,
            version=registered_model.version,
            key="actual.temperature",
            value=str(metadata["confidence_policy"]["temperature"]),
        )
    return {
        "mlflow_run_id": run_id,
        "mlflow_model_name": model_name,
        "mlflow_model_version": str(registered_model.version),
        "mlflow_model_uri": model_uri,
        "mlflow_candidate_alias": "candidate",
    }


def main():
    parser = argparse.ArgumentParser(description="Train the Actual smart transaction categorizer.")
    parser.add_argument("--train-dataset", required=True)
    parser.add_argument("--val-dataset", required=True)
    parser.add_argument("--test-dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--external-dataset", help="Optional CSV/parquet dataset used for pretraining or warm-start supervision.")
    parser.add_argument("--run-name", default="smart-transaction-categorizer")
    parser.add_argument(
        "--register-model-name",
        default=default_register_model_name(),
        help="Optional MLflow registered model name for challenger registration.",
    )
    parser.add_argument("--max-word-features", type=int, default=25000)
    parser.add_argument("--max-char-features", type=int, default=25000)
    parser.add_argument("--c-value", type=float, default=2.0)
    parser.add_argument(
        "--model-families",
        default="logreg,linear_svm,sgd_log",
        help="Comma-separated candidate families. Supported: logreg, linear_svm, sgd_log.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = read_dataset(args.train_dataset)
    val_df = read_dataset(args.val_dataset)
    test_df = read_dataset(args.test_dataset)
    train_df = prepare_labeled_dataset(train_df, "train")
    val_df = prepare_labeled_dataset(val_df, "validation")
    test_df = prepare_labeled_dataset(test_df, "test")

    if args.external_dataset:
        external_df = read_dataset(args.external_dataset)
        external_df = prepare_labeled_dataset(external_df, "external")
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

        train_frame = build_model_frame(train_df)
        families = [item.strip() for item in args.model_families.split(",") if item.strip()]
        candidate_records = []
        best_candidate = None

        for family in families:
            candidate_model = build_pipeline(
                max_word_features=args.max_word_features,
                max_char_features=args.max_char_features,
                c_value=args.c_value,
                model_family=family,
            )
            candidate_model.fit(train_frame, train_df["category"].astype(str))
            candidate_val_metrics = evaluate_model(candidate_model, val_df)
            candidate_test_metrics = evaluate_model(candidate_model, test_df)
            record = {
                "model_family": family,
                "validation": candidate_val_metrics,
                "test": candidate_test_metrics,
            }
            candidate_records.append(record)
            mlflow.log_metrics(
                {
                    f"{family}_val_{key}": value
                    for key, value in candidate_val_metrics.items()
                    if key != "row_count"
                }
            )
            mlflow.log_metrics(
                {
                    f"{family}_test_{key}": value
                    for key, value in candidate_test_metrics.items()
                    if key != "row_count"
                }
            )
            score = (
                candidate_val_metrics["top3_accuracy"],
                candidate_val_metrics["macro_f1"],
                candidate_val_metrics["top1_accuracy"],
            )
            if best_candidate is None or score > best_candidate["score"]:
                best_candidate = {
                    "score": score,
                    "model": candidate_model,
                    "record": record,
                }

        if best_candidate is None:
            raise RuntimeError("No model candidates were trained.")

        model = best_candidate["model"]
        selected_record = best_candidate["record"]
        selected_family = selected_record["model_family"]
        val_metrics = selected_record["validation"]
        test_metrics = selected_record["test"]
        classes = list(model.named_steps["clf"].classes_)
        validation_probabilities = model_score_matrix(model, build_model_frame(val_df))
        temperature, calibration_summary = fit_temperature(
            validation_probabilities,
            val_df["category"].astype(str),
            classes,
        )
        confidence_policy = {
            "method": "temperature_keyword_fallback",
            "temperature": round(float(temperature), 6),
            "keyword_fallback": {
                "enabled": True,
                "blend_weight": 0.35,
                "max_primary_confidence": 0.58,
                "allowed_sources": [
                    "transaction_description",
                    "transaction_description_clean",
                    "merchant_text",
                    "imported_description",
                    "notes",
                    "derived",
                ],
            },
            "validation_calibration": calibration_summary,
        }
        mlflow.log_params(
            {
                "candidate_model_families": ",".join(families),
                "selected_model_family": selected_family,
                "model_selection_policy": "max validation top3_accuracy, macro_f1, top1_accuracy",
                "confidence_policy_method": confidence_policy["method"],
                "confidence_temperature": confidence_policy["temperature"],
            }
        )
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items() if k != "row_count"})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items() if k != "row_count"})
        mlflow.log_metrics(
            {
                "calibration_val_nll_before": calibration_summary["before"]["nll"],
                "calibration_val_nll_after": calibration_summary["after"]["nll"],
                "calibration_val_ece_before": calibration_summary["before"]["ece"],
                "calibration_val_ece_after": calibration_summary["after"]["ece"],
                "calibration_val_brier_before": calibration_summary["before"]["brier"],
                "calibration_val_brier_after": calibration_summary["after"]["brier"],
            }
        )
        mlflow.log_dict(candidate_records, artifact_file="candidate_models.json")

        model_version = datetime.now(timezone.utc).strftime("v%Y%m%d%H%M%S")

        joblib.dump(model, output_dir / "model.joblib")
        metadata = {
            "model_version": model_version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_family": selected_family,
            "candidate_models": candidate_records,
            "model_selection_policy": "max validation top3_accuracy, macro_f1, top1_accuracy",
            "taxonomy": taxonomy_manifest(),
            "confidence_policy": confidence_policy,
            "metrics": {
                "validation": val_metrics,
                "test": test_metrics,
            },
            "classes": classes,
        }
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        (output_dir / "metrics.json").write_text(json.dumps(test_metrics, indent=2))
        (output_dir / "label_taxonomy.json").write_text(
            json.dumps(metadata["taxonomy"], indent=2),
            encoding="utf-8",
        )
        (output_dir / "register_result.json").write_text(json.dumps({}, indent=2), encoding="utf-8")

        mlflow.log_artifacts(str(output_dir), artifact_path="serving_bundle")
        mlflow.sklearn.log_model(sk_model=model, artifact_path="sk_model")
        registry_metadata = register_model_if_configured(
            run.info.run_id,
            args.register_model_name,
            metadata,
        )
        if registry_metadata:
            metadata.update(registry_metadata)
            (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
            (output_dir / "register_result.json").write_text(
                json.dumps(registry_metadata, indent=2),
                encoding="utf-8",
            )
            mlflow.log_dict(metadata, artifact_file="serving_bundle/metadata.json")
            mlflow.log_dict(registry_metadata, artifact_file="serving_bundle/register_result.json")

        print(json.dumps({"model_version": model_version, "test_metrics": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
