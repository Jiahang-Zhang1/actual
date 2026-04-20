from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from app.backends.base import ModelBackend
from app.backends.onnx_backend import OnnxBackend
from app.backends.sklearn_backend import SklearnBackend
from app.config import get_settings
from app.feature_adapter import build_feature_frame
from app.schemas import PredictRequest

_model_bundle_signature: tuple[int, ...] | None = None


def _current_model_bundle_signature() -> tuple[int, ...]:
    settings = get_settings()
    bundle_dir = Path(settings.model_bundle_dir)
    tracked_files = (
        "selected_model.json",
        "metadata.json",
        "model.joblib",
        "model.onnx",
        "model.dynamic_quant.onnx",
    )
    return tuple(
        (bundle_dir / filename).stat().st_mtime_ns
        if (bundle_dir / filename).exists()
        else 0
        for filename in tracked_files
    )


@lru_cache(maxsize=1)
def get_backend() -> ModelBackend:
    settings = get_settings()
    if settings.backend_kind == "baseline":
        return SklearnBackend(settings.model_path)
    if settings.backend_kind in {"onnx", "onnx_dynamic_quant"}:
        return OnnxBackend(settings.model_path, settings.source_model_path)
    raise ValueError(f"Unsupported BACKEND_KIND={settings.backend_kind}")


def warmup_backend() -> None:
    backend = get_backend()
    frame = build_feature_frame(
        [
            PredictRequest(
                transaction_description="STARBUCKS STORE 1458 NEW YORK NY",
                country="US",
                currency="USD",
            )
        ]
    )
    backend.predict(frame)


def reload_backend() -> None:
    global _model_bundle_signature
    get_settings.cache_clear()
    get_backend.cache_clear()
    warmup_backend()
    _model_bundle_signature = _current_model_bundle_signature()


def refresh_backend_if_model_changed() -> None:
    global _model_bundle_signature
    signature = _current_model_bundle_signature()
    if _model_bundle_signature is None:
        _model_bundle_signature = signature
        return
    if signature != _model_bundle_signature:
        # Each worker process checks the shared deployed bundle before serving
        # requests, so promotion/rollback reaches all local workers and K8s pods.
        reload_backend()
