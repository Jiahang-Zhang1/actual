from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.backends.onnx_backend import OnnxBackend
from app.backends.sklearn_backend import SklearnBackend, load_sklearn_model
from app.config import get_settings
from app.feature_adapter import build_feature_frame
from app.schemas import PredictRequest
from tools.common import git_sha, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ONNX artifacts from the source sklearn model.")
    parser.add_argument("--force", action="store_true", help="Recreate optimized artifacts even if they already exist.")
    return parser.parse_args()


def ensure_directories(root: Path) -> None:
    (root / "models/optimized").mkdir(parents=True, exist_ok=True)
    (root / "artifacts/examples").mkdir(parents=True, exist_ok=True)
    (root / "results/raw").mkdir(parents=True, exist_ok=True)
    (root / "results/summary").mkdir(parents=True, exist_ok=True)


def _sanitize_for_onnx_inplace(obj) -> None:
    from sklearn.compose import ColumnTransformer
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.pipeline import FeatureUnion, Pipeline

    if isinstance(obj, (CountVectorizer, TfidfVectorizer)):
        obj.strip_accents = None
        return

    if isinstance(obj, Pipeline):
        for _, step in obj.steps:
            _sanitize_for_onnx_inplace(step)
        return

    if isinstance(obj, ColumnTransformer):
        for _, trans, _ in obj.transformers:
            if trans not in ("drop", "passthrough"):
                _sanitize_for_onnx_inplace(trans)
        if hasattr(obj, "transformers_"):
            for _, trans, _ in obj.transformers_:
                if trans not in ("drop", "passthrough"):
                    _sanitize_for_onnx_inplace(trans)
        return

    if isinstance(obj, FeatureUnion):
        for _, trans in obj.transformer_list:
            _sanitize_for_onnx_inplace(trans)
        if hasattr(obj, "transformer_list_"):
            for _, trans in obj.transformer_list_:
                _sanitize_for_onnx_inplace(trans)
        return


def export_onnx(source_model_path: Path, target_path: Path, sample_frame: pd.DataFrame) -> None:
    from skl2onnx import to_onnx

    model = load_sklearn_model(str(source_model_path))
    _sanitize_for_onnx_inplace(model)
    clf = getattr(model, "named_steps", {}).get("clf")
    options = None
    if clf is not None:
        options = {id(clf): {"raw_scores": True}}

    onx = to_onnx(model, sample_frame, target_opset=17, options=options)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(onx.SerializeToString())


def quantize_dynamic_model(source_onnx_path: Path, target_quant_path: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    target_quant_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        str(source_onnx_path),
        str(target_quant_path),
        weight_type=QuantType.QInt8,
    )


def file_size_mb(path: Path) -> float | None:
    if not path.exists():
        return None
    return round(path.stat().st_size / (1024 ** 2), 4)


def main() -> None:
    args = parse_args()
    settings = get_settings()
    root = Path(__file__).resolve().parents[1]
    ensure_directories(root)

    source_model_path = Path(settings.source_model_path)
    if not source_model_path.exists():
        raise FileNotFoundError(
            f"Missing source model at {source_model_path}. "
            "This serving bundle expects the training artifact under models/source/."
        )

    onnx_path = root / "models/optimized/v2_tfidf_linearsvc_model.onnx"
    quant_path = root / "models/optimized/v2_tfidf_linearsvc_model.dynamic_quant.onnx"

    sample_request = PredictRequest(
        transaction_description="STARBUCKS STORE 1458 NEW YORK NY",
        country="US",
        currency="USD",
    )
    sample_frame = build_feature_frame([sample_request])

    if args.force or not onnx_path.exists():
        export_onnx(source_model_path, onnx_path, sample_frame)

    if args.force or not quant_path.exists():
        quantize_dynamic_model(onnx_path, quant_path)

    baseline_backend = SklearnBackend(str(source_model_path))
    onnx_backend = OnnxBackend(str(onnx_path), str(source_model_path))
    quant_backend = OnnxBackend(str(quant_path), str(source_model_path))

    baseline_out = baseline_backend.predict(sample_frame)
    onnx_out = onnx_backend.predict(sample_frame)
    quant_out = quant_backend.predict(sample_frame)

    order = np.argsort(baseline_out.probabilities[0])[::-1][: settings.top_k]
    output_sample = {
        "predicted_category_id": baseline_out.labels[0],
        "confidence": round(float(baseline_out.probabilities[0][order[0]]), 6),
        "top_categories": [
            {
                "category_id": baseline_out.classes[idx],
                "score": round(float(baseline_out.probabilities[0][idx]), 6),
            }
            for idx in order
        ],
        "model_version": settings.model_version,
    }

    write_json(
        root / "artifacts/examples/input_sample.json",
        {
            "transaction_description": "STARBUCKS STORE 1458 NEW YORK NY",
            "country": "US",
            "currency": "USD",
        },
    )
    write_json(
        root / "artifacts/examples/batch_input_sample.json",
        {
            "items": [
                {
                    "transaction_description": "STARBUCKS STORE 1458 NEW YORK NY",
                    "country": "US",
                    "currency": "USD",
                },
                {
                    "transaction_description": "AMAZON MARKETPLACE PMTS",
                    "country": "US",
                    "currency": "USD",
                },
            ]
        },
    )
    write_json(root / "artifacts/examples/output_sample.json", output_sample)

    manifest = {
        "model_version": settings.model_version,
        "code_version": git_sha(),
        "source_model_path": str(source_model_path),
        "optimized_model_paths": {
            "onnx": str(onnx_path),
            "onnx_dynamic_quant": str(quant_path),
        },
        "sizes_mb": {
            "source": file_size_mb(source_model_path),
            "onnx": file_size_mb(onnx_path),
            "onnx_dynamic_quant": file_size_mb(quant_path),
        },
        "validation_sample": {
            "baseline_label": baseline_out.labels[0],
            "onnx_label": onnx_out.labels[0],
            "onnx_dynamic_quant_label": quant_out.labels[0],
            "baseline_top1_confidence": round(float(np.max(baseline_out.probabilities[0])), 6),
            "onnx_top1_confidence": round(float(np.max(onnx_out.probabilities[0])), 6),
            "onnx_dynamic_quant_top1_confidence": round(float(np.max(quant_out.probabilities[0])), 6),
        },
        "classes": baseline_backend.classes,
    }
    write_json(root / "models/manifest.json", manifest)
    print("Prepared artifacts:")
    print((root / "models/manifest.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
