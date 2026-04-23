import argparse
import json
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

import train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained challenger model and emit detailed metrics artifacts."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--val-dataset", required=True)
    parser.add_argument("--test-dataset", required=True)
    parser.add_argument(
        "--output-dir",
        help="Optional output directory for metrics artifacts. Defaults to --model-dir.",
    )
    parser.add_argument(
        "--label-mapping-file",
        help="Optional JSON file mapping raw labels to aligned deployment labels.",
    )
    parser.add_argument("--high-confidence-threshold", type=float, default=0.70)
    return parser.parse_args()


def load_label_mapping(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    mapping_path = Path(path)
    raw_mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    return {str(key): str(value) for key, value in raw_mapping.items()}


def map_labels(labels: Iterable[str], mapping: dict[str, str]) -> list[str]:
    return [mapping.get(str(label), str(label)) for label in labels]


def safe_macro_f1(true_labels: list[str], pred_labels: list[str]) -> float:
    if not true_labels:
        return 0.0
    return float(f1_score(true_labels, pred_labels, average="macro", zero_division=0))


def top_k_accuracy_from_indices(
    topk_indices: np.ndarray,
    true_labels: list[str],
    classes: list[str],
) -> float:
    if len(true_labels) == 0:
        return 0.0
    label_to_index = {label: idx for idx, label in enumerate(classes)}
    hits = 0
    for label, row_indices in zip(true_labels, topk_indices):
        target_idx = label_to_index.get(label)
        if target_idx is not None and target_idx in set(map(int, row_indices)):
            hits += 1
    return hits / len(true_labels)


def high_confidence_metrics(
    true_labels: list[str],
    pred_labels: list[str],
    confidences: list[float],
    threshold: float,
) -> dict:
    if not true_labels:
        return {
            "high_confidence_precision": 0.0,
            "high_confidence_coverage": 0.0,
            "high_confidence_count": 0,
        }

    selected = [
        (truth, pred)
        for truth, pred, confidence in zip(true_labels, pred_labels, confidences)
        if confidence >= threshold
    ]
    if not selected:
        return {
            "high_confidence_precision": 0.0,
            "high_confidence_coverage": 0.0,
            "high_confidence_count": 0,
        }

    correct = sum(1 for truth, pred in selected if truth == pred)
    return {
        "high_confidence_precision": correct / len(selected),
        "high_confidence_coverage": len(selected) / len(true_labels),
        "high_confidence_count": len(selected),
    }


def per_class_report(true_labels: list[str], pred_labels: list[str]) -> dict[str, dict]:
    if not true_labels:
        return {}
    report = classification_report(
        true_labels,
        pred_labels,
        output_dict=True,
        zero_division=0,
    )
    excluded = {"accuracy", "macro avg", "weighted avg"}
    return {
        label: {
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1_score": float(metrics["f1-score"]),
            "support": int(metrics["support"]),
        }
        for label, metrics in report.items()
        if label not in excluded
    }


def amount_bucket_series(df: pd.DataFrame) -> pd.Series:
    if "amount" not in df:
        return pd.Series(["unknown"] * len(df), index=df.index, dtype=str)
    return df["amount"].apply(train_model.amount_bucket)


def string_flag(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df:
        return pd.Series([False] * len(df), index=df.index, dtype=bool)
    return df[column].fillna("").astype(str).str.strip().ne("")


def topk_lists(
    probabilities: np.ndarray,
    classes: list[str],
    limit: int = 3,
) -> tuple[np.ndarray, list[list[str]], list[list[float]]]:
    capped_limit = min(limit, len(classes))
    indices = np.argsort(probabilities, axis=1)[:, -capped_limit:][:, ::-1]
    labels = [[classes[int(idx)] for idx in row] for row in indices]
    scores = [[float(probabilities[row_idx][int(idx)]) for idx in row] for row_idx, row in enumerate(indices)]
    return indices, labels, scores


def split_predictions_dataframe(
    df: pd.DataFrame,
    true_labels: list[str],
    pred_labels: list[str],
    confidences: list[float],
    top3_labels: list[list[str]],
    top3_scores: list[list[float]],
    mapping: dict[str, str],
) -> pd.DataFrame:
    records = []
    for row_index, (_, row) in enumerate(df.reset_index(drop=True).iterrows()):
        labels = top3_labels[row_index]
        scores = top3_scores[row_index]
        record = {
            "true_label": true_labels[row_index],
            "predicted_label": pred_labels[row_index],
            "confidence": confidences[row_index],
            "mapped_true_label": mapping.get(true_labels[row_index], true_labels[row_index]),
            "mapped_predicted_label": mapping.get(pred_labels[row_index], pred_labels[row_index]),
        }
        for rank in range(3):
            record[f"top{rank + 1}_label"] = labels[rank] if rank < len(labels) else None
            record[f"top{rank + 1}_score"] = scores[rank] if rank < len(scores) else None
        for column in [
            "transaction_description",
            "country",
            "currency",
            "amount",
            "transaction_date",
            "notes",
            "imported_description",
            "account_id",
            "category",
        ]:
            if column in row:
                record[column] = row[column]
        record["description_source"] = train_model.description_source(row)
        record["amount_bucket"] = train_model.amount_bucket(row.get("amount"))
        records.append(record)
    return pd.DataFrame.from_records(records)


def compute_split_metrics(
    df: pd.DataFrame,
    model,
    mapping: dict[str, str],
    high_confidence_threshold: float,
) -> tuple[dict, dict[str, dict], pd.DataFrame]:
    if df.empty:
        empty_metrics = {
            "row_count": 0,
            "top1_accuracy": 0.0,
            "top3_accuracy": 0.0,
            "macro_f1": 0.0,
            "mapped_top1_accuracy": 0.0,
            "mapped_top3_accuracy": 0.0,
            "mapped_macro_f1": 0.0,
            "high_confidence_precision": 0.0,
            "high_confidence_coverage": 0.0,
            "high_confidence_count": 0,
        }
        return empty_metrics, {}, pd.DataFrame()

    frame = train_model.build_model_frame(df)
    probabilities = train_model.model_score_matrix(model, frame)
    pred_labels = list(model.predict(frame))
    true_labels = df["category"].astype(str).tolist()
    classes = list(model.named_steps["clf"].classes_)
    top3_indices, top3_labels, top3_scores = topk_lists(probabilities, classes, limit=3)
    confidences = [scores[0] if scores else 0.0 for scores in top3_scores]

    mapped_true_labels = map_labels(true_labels, mapping)
    mapped_pred_labels = map_labels(pred_labels, mapping)
    mapped_top3_labels = [map_labels(labels, mapping) for labels in top3_labels]
    mapped_top3_accuracy = 0.0
    if mapped_true_labels:
        mapped_hits = sum(
            1 for truth, labels in zip(mapped_true_labels, mapped_top3_labels) if truth in labels
        )
        mapped_top3_accuracy = mapped_hits / len(mapped_true_labels)

    metrics = {
        "row_count": int(len(df)),
        "top1_accuracy": float(accuracy_score(true_labels, pred_labels)),
        "top3_accuracy": float(top_k_accuracy_from_indices(top3_indices, true_labels, classes)),
        "macro_f1": safe_macro_f1(true_labels, pred_labels),
        "mapped_top1_accuracy": float(accuracy_score(mapped_true_labels, mapped_pred_labels)),
        "mapped_top3_accuracy": float(mapped_top3_accuracy),
        "mapped_macro_f1": safe_macro_f1(mapped_true_labels, mapped_pred_labels),
    }
    metrics.update(
        high_confidence_metrics(
            true_labels,
            pred_labels,
            confidences,
            high_confidence_threshold,
        )
    )

    per_class = per_class_report(true_labels, pred_labels)
    predictions = split_predictions_dataframe(
        df=df,
        true_labels=true_labels,
        pred_labels=pred_labels,
        confidences=confidences,
        top3_labels=top3_labels,
        top3_scores=top3_scores,
        mapping=mapping,
    )
    return metrics, per_class, predictions


def compute_slice_metrics(predictions: pd.DataFrame) -> dict[str, dict]:
    if predictions.empty:
        return {}

    slices: dict[str, dict] = {}
    mask_map = {
        "manual_sparse": predictions["description_source"].eq("derived")
        | predictions["description_source"].eq("notes"),
        "notes_backed": predictions["description_source"].eq("notes"),
        "imported_backed": predictions["description_source"].eq("imported_description"),
        "full_description": predictions["description_source"].eq("transaction_description"),
    }

    for source in sorted(predictions["description_source"].dropna().astype(str).unique()):
        mask_map[f"description_source:{source}"] = predictions["description_source"].eq(source)
    for bucket in sorted(predictions["amount_bucket"].dropna().astype(str).unique()):
        mask_map[f"amount_bucket:{bucket}"] = predictions["amount_bucket"].eq(bucket)

    for name, mask in mask_map.items():
        subset = predictions[mask].copy()
        if subset.empty:
            continue
        true_labels = subset["true_label"].astype(str).tolist()
        pred_labels = subset["predicted_label"].astype(str).tolist()
        top3_hits = subset.apply(
            lambda row: row["true_label"]
            in [row.get("top1_label"), row.get("top2_label"), row.get("top3_label")],
            axis=1,
        )
        slices[name] = {
            "row_count": int(len(subset)),
            "top1_accuracy": float(accuracy_score(true_labels, pred_labels)),
            "top3_accuracy": float(top3_hits.mean()),
            "macro_f1": safe_macro_f1(true_labels, pred_labels),
            "avg_confidence": float(subset["confidence"].mean()),
        }
    return slices


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else model_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model = joblib.load(model_dir / "model.joblib")
    val_df = train_model.read_dataset(args.val_dataset)
    test_df = train_model.read_dataset(args.test_dataset)
    mapping = load_label_mapping(args.label_mapping_file)

    split_summaries = {}
    per_class_summaries = {}
    slice_summaries = {}

    for split_name, split_df in {"validation": val_df, "test": test_df}.items():
        metrics, per_class, predictions = compute_split_metrics(
            split_df,
            model,
            mapping,
            args.high_confidence_threshold,
        )
        predictions["split"] = split_name
        split_summaries[split_name] = metrics
        per_class_summaries[split_name] = per_class
        slice_summaries[split_name] = compute_slice_metrics(predictions)
        predictions.to_csv(output_dir / f"{split_name}_predictions.csv", index=False)

    test_metrics = dict(split_summaries["test"])
    test_metrics["validation"] = split_summaries["validation"]
    test_metrics["test"] = split_summaries["test"]
    test_metrics["label_mapping_file"] = args.label_mapping_file
    test_metrics["high_confidence_threshold"] = args.high_confidence_threshold

    (output_dir / "metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    (output_dir / "per_class_metrics.json").write_text(
        json.dumps(per_class_summaries, indent=2),
        encoding="utf-8",
    )
    (output_dir / "slice_metrics.json").write_text(
        json.dumps(slice_summaries, indent=2),
        encoding="utf-8",
    )

    print(json.dumps({"metrics": test_metrics, "per_class_metrics": per_class_summaries}, indent=2))


if __name__ == "__main__":
    main()
