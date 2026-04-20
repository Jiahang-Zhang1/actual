#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SERVING_ROOT = REPO_ROOT / "serving"
sys.path.insert(0, str(SERVING_ROOT))

from app.backends.onnx_backend import OnnxBackend
from app.backends.sklearn_backend import SklearnBackend
from app.feature_adapter import build_feature_frame
from app.schemas import PredictRequest
from tools.prepare_artifacts import export_onnx, file_size_mb, quantize_dynamic_model


def read_dataset(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export joblib, ONNX, and dynamic-quantized ONNX variants and select the best serving artifact."
    )
    parser.add_argument("--model-dir", required=True, help="Directory containing model.joblib and metadata.json.")
    parser.add_argument("--sample-dataset", help="Optional CSV/parquet sample used for parity and latency checks.")
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--trials", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--min-label-match", type=float, default=0.99)
    parser.add_argument("--min-top3-match", type=float, default=0.95)
    return parser.parse_args()


def default_requests() -> list[PredictRequest]:
    return [
        PredictRequest(transaction_description="Starbucks Store 1458 New York NY", country="US", currency="USD"),
        PredictRequest(transaction_description="Amazon Marketplace PMTS", country="US", currency="USD"),
        PredictRequest(transaction_description="Payroll Direct Deposit", country="US", currency="USD"),
        PredictRequest(transaction_description="Con Edison Electric", country="US", currency="USD"),
        PredictRequest(transaction_description="IRS Tax Payment", country="US", currency="USD"),
    ]


def sample_requests(sample_dataset: str | None, max_samples: int) -> list[PredictRequest]:
    if not sample_dataset:
        return default_requests()
    df = read_dataset(Path(sample_dataset)).head(max_samples)
    requests: list[PredictRequest] = []
    for _, row in df.iterrows():
        requests.append(
            PredictRequest(
                transaction_description=str(row.get("transaction_description", "")),
                country=str(row.get("country", "US") or "US"),
                currency=str(row.get("currency", "USD") or "USD"),
                amount=float(row["amount"]) if "amount" in row and pd.notna(row["amount"]) else None,
                transaction_date=str(row.get("transaction_date", "") or "") or None,
                notes=str(row.get("notes", "") or "") or None,
                imported_description=str(row.get("imported_description", "") or "") or None,
            )
        )
    return requests or default_requests()


def topk_indices(probabilities: np.ndarray, k: int = 3) -> np.ndarray:
    return np.argsort(probabilities, axis=1)[:, -k:][:, ::-1]


def benchmark_backend(
    name: str,
    backend,
    frame: pd.DataFrame,
    baseline_labels: list[str],
    baseline_top3: np.ndarray,
    artifact_path: Path,
    trials: int,
    warmup: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        backend.predict(frame)

    latencies_ms: list[float] = []
    output = None
    for _ in range(trials):
        started = time.perf_counter()
        output = backend.predict(frame)
        latencies_ms.append((time.perf_counter() - started) * 1000.0)

    assert output is not None
    top3 = topk_indices(output.probabilities, k=min(3, len(output.classes)))
    label_matches = [left == right for left, right in zip(baseline_labels, output.labels)]
    top3_matches = [
        set(map(int, left)) == set(map(int, right))
        for left, right in zip(baseline_top3, top3)
    ]
    return {
        "status": "ok",
        "variant": name,
        "backend_kind": backend.kind if name != "onnx_dynamic_quant" else "onnx_dynamic_quant",
        "path": artifact_path.name,
        "artifact_size_mb": file_size_mb(artifact_path),
        "p50_latency_ms": round(float(np.percentile(latencies_ms, 50)), 4),
        "p95_latency_ms": round(float(np.percentile(latencies_ms, 95)), 4),
        "throughput_items_per_second": round(
            float(len(frame) * trials / max(sum(latencies_ms) / 1000.0, 1e-9)),
            4,
        ),
        "label_match_rate": round(sum(label_matches) / len(label_matches), 4),
        "top3_match_rate": round(sum(top3_matches) / len(top3_matches), 4),
    }


def failed_variant(name: str, path: Path, exc: Exception) -> dict[str, Any]:
    return {
        "status": "failed",
        "variant": name,
        "backend_kind": name,
        "path": path.name,
        "artifact_size_mb": file_size_mb(path),
        "error": str(exc),
        "label_match_rate": 0.0,
        "top3_match_rate": 0.0,
        "p95_latency_ms": None,
    }


def choose_variant(variants: dict[str, dict[str, Any]], min_label_match: float, min_top3_match: float) -> str:
    # Selection is conservative: only variants that match sklearn predictions
    # can win, then we choose the lowest p95 latency and use file size as a tie-breaker.
    eligible = [
        row
        for row in variants.values()
        if row.get("status") == "ok"
        and float(row.get("label_match_rate", 0.0)) >= min_label_match
        and float(row.get("top3_match_rate", 0.0)) >= min_top3_match
        and row.get("p95_latency_ms") is not None
    ]
    if not eligible:
        return "baseline"
    winner = sorted(
        eligible,
        key=lambda row: (
            float(row["p95_latency_ms"]),
            float(row.get("artifact_size_mb") or 1e9),
            0 if row["variant"] == "onnx_dynamic_quant" else 1,
        ),
    )[0]
    return str(winner["variant"])


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir).resolve()
    source_path = model_dir / "model.joblib"
    onnx_path = model_dir / "model.onnx"
    quant_path = model_dir / "model.dynamic_quant.onnx"
    if not source_path.exists():
        raise FileNotFoundError(f"Missing trained sklearn artifact: {source_path}")

    requests = sample_requests(args.sample_dataset, args.max_samples)
    frame = build_feature_frame(requests)
    baseline = SklearnBackend(str(source_path))
    baseline_output = baseline.predict(frame)
    baseline_top3 = topk_indices(baseline_output.probabilities, k=min(3, len(baseline_output.classes)))

    variants: dict[str, dict[str, Any]] = {
        "baseline": benchmark_backend(
            "baseline",
            baseline,
            frame,
            baseline_output.labels,
            baseline_top3,
            source_path,
            args.trials,
            args.warmup,
        )
    }

    try:
        export_onnx(source_path, onnx_path, frame.head(1))
        variants["onnx"] = benchmark_backend(
            "onnx",
            OnnxBackend(str(onnx_path), str(source_path)),
            frame,
            baseline_output.labels,
            baseline_top3,
            onnx_path,
            args.trials,
            args.warmup,
        )
    except Exception as exc:
        variants["onnx"] = failed_variant("onnx", onnx_path, exc)

    try:
        if not onnx_path.exists():
            raise FileNotFoundError("ONNX export did not produce model.onnx")
        quantize_dynamic_model(onnx_path, quant_path)
        variants["onnx_dynamic_quant"] = benchmark_backend(
            "onnx_dynamic_quant",
            OnnxBackend(str(quant_path), str(source_path)),
            frame,
            baseline_output.labels,
            baseline_top3,
            quant_path,
            args.trials,
            args.warmup,
        )
    except Exception as exc:
        variants["onnx_dynamic_quant"] = failed_variant("onnx_dynamic_quant", quant_path, exc)

    metadata_path = model_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    selected_variant = choose_variant(variants, args.min_label_match, args.min_top3_match)
    selection = {
        "model_version": metadata.get("model_version", "unknown"),
        "selected_variant": selected_variant,
        "selection_policy": {
            "must_match_sklearn_label_rate": args.min_label_match,
            "must_match_sklearn_top3_rate": args.min_top3_match,
            "rank_by": ["p95_latency_ms", "artifact_size_mb", "prefer_dynamic_quantized_tie_break"],
        },
        "paths": {
            "baseline": source_path.name,
            "onnx": onnx_path.name,
            "onnx_dynamic_quant": quant_path.name,
        },
        "variants": variants,
    }
    (model_dir / "selected_model.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
