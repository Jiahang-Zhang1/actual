from __future__ import annotations

from functools import lru_cache

from app.backends.base import ModelBackend
from app.backends.onnx_backend import OnnxBackend
from app.backends.sklearn_backend import SklearnBackend
from app.config import get_settings
from app.feature_adapter import build_feature_frame
from app.schemas import PredictRequest


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
