from __future__ import annotations

import json

from app.config import _resolve_bundle_selection


def test_manifest_selection_is_used_when_no_promoted_bundle(tmp_path):
    models_dir = tmp_path / "models"
    optimized_dir = models_dir / "optimized"
    source_dir = models_dir / "source"
    runtime_dir = tmp_path / "runtime" / "deployed"
    optimized_dir.mkdir(parents=True)
    source_dir.mkdir(parents=True)
    runtime_dir.mkdir(parents=True)
    dynamic_model = optimized_dir / "model.dynamic_quant.onnx"
    source_model = source_dir / "model.joblib"
    dynamic_model.write_text("onnx", encoding="utf-8")
    source_model.write_text("joblib", encoding="utf-8")
    (models_dir / "manifest.json").write_text(
        json.dumps(
            {
                "model_version": "v-manifest",
                "source_model_path": str(source_model),
                "optimized_model_paths": {
                    "onnx_dynamic_quant": str(dynamic_model),
                },
            },
        ),
        encoding="utf-8",
    )

    backend_kind, model_path, source_model_path, model_version = _resolve_bundle_selection(
        "onnx_dynamic_quant",
        str(dynamic_model),
        str(source_model),
        "fallback-version",
        str(runtime_dir),
    )

    assert backend_kind == "onnx_dynamic_quant"
    assert model_path == str(dynamic_model)
    assert source_model_path == str(source_model)
    assert model_version == "v-manifest"


def test_promoted_bundle_overrides_manifest_selection(tmp_path):
    models_dir = tmp_path / "models"
    optimized_dir = models_dir / "optimized"
    source_dir = models_dir / "source"
    runtime_dir = tmp_path / "runtime" / "deployed"
    optimized_dir.mkdir(parents=True)
    source_dir.mkdir(parents=True)
    runtime_dir.mkdir(parents=True)
    dynamic_model = optimized_dir / "model.dynamic_quant.onnx"
    source_model = source_dir / "model.joblib"
    dynamic_model.write_text("onnx", encoding="utf-8")
    source_model.write_text("joblib", encoding="utf-8")
    (models_dir / "manifest.json").write_text(
        json.dumps(
            {
                "model_version": "v-manifest",
                "source_model_path": str(source_model),
                "optimized_model_paths": {
                    "onnx_dynamic_quant": str(dynamic_model),
                },
            },
        ),
        encoding="utf-8",
    )
    (runtime_dir / "model.onnx").write_text("promoted", encoding="utf-8")
    (runtime_dir / "model.joblib").write_text("promoted-source", encoding="utf-8")
    (runtime_dir / "selected_model.json").write_text(
        json.dumps(
            {
                "model_version": "v-selected",
                "selected_variant": "onnx",
                "paths": {
                    "baseline": "model.joblib",
                    "onnx": "model.onnx",
                },
            },
        ),
        encoding="utf-8",
    )
    (runtime_dir / "metadata.json").write_text(
        json.dumps({"model_version": "v-promoted"}),
        encoding="utf-8",
    )

    backend_kind, model_path, source_model_path, model_version = _resolve_bundle_selection(
        "auto",
        str(dynamic_model),
        str(source_model),
        "fallback-version",
        str(runtime_dir),
    )

    assert backend_kind == "onnx"
    assert model_path == str(runtime_dir / "model.onnx")
    assert source_model_path == str(runtime_dir / "model.joblib")
    assert model_version == "v-promoted"
