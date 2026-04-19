from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _git_sha_fallback() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return os.getenv("CODE_VERSION", "unknown").strip() or "unknown"


def default_model_path(backend_kind: str) -> str:
    if backend_kind == "baseline":
        return "/workspace/models/source/v2_tfidf_linearsvc_model.joblib"
    if backend_kind == "onnx":
        return "/workspace/models/optimized/v2_tfidf_linearsvc_model.onnx"
    if backend_kind == "onnx_dynamic_quant":
        return "/workspace/models/optimized/v2_tfidf_linearsvc_model.dynamic_quant.onnx"
    raise ValueError(f"Unsupported BACKEND_KIND={backend_kind}")


@dataclass(frozen=True)
class Settings:
    backend_kind: str
    model_path: str
    source_model_path: str
    model_version: str
    code_version: str
    top_k: int
    service_host: str
    service_port: int
    web_concurrency: int
    log_level: str
    runtime_dir: str
    rollout_context: str
    monitor_window_minutes: int
    promotion_min_requests: int
    promotion_min_feedback: int
    promotion_max_p95_ms: float
    promotion_max_error_rate: float
    promotion_min_top1_acceptance: float
    promotion_min_top3_acceptance: float
    rollback_min_requests: int
    rollback_min_feedback: int
    rollback_max_p95_ms: float
    rollback_max_error_rate: float
    rollback_min_top1_acceptance: float
    rollback_min_top3_acceptance: float


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    backend_kind = os.getenv("BACKEND_KIND", "onnx_dynamic_quant").strip()
    model_path = os.getenv("MODEL_PATH", "").strip() or default_model_path(backend_kind)
    return Settings(
        backend_kind=backend_kind,
        model_path=model_path,
        source_model_path=os.getenv(
            "SOURCE_MODEL_PATH",
            "/workspace/models/source/v2_tfidf_linearsvc_model.joblib",
        ).strip(),
        model_version=os.getenv("MODEL_VERSION", "v2_tfidf_linearsvc").strip(),
        code_version=os.getenv("CODE_VERSION", "").strip() or _git_sha_fallback(),
        top_k=int(os.getenv("TOP_K", "3")),
        service_host=os.getenv("SERVICE_HOST", "0.0.0.0").strip(),
        service_port=int(os.getenv("SERVICE_PORT", "8000")),
        web_concurrency=int(os.getenv("WEB_CONCURRENCY", "1")),
        log_level=os.getenv("LOG_LEVEL", "info").strip(),
        runtime_dir=os.getenv("RUNTIME_DIR", "/workspace/runtime").strip(),
        rollout_context=os.getenv("ROLLOUT_CONTEXT", "production").strip(),
        monitor_window_minutes=int(os.getenv("MONITOR_WINDOW_MINUTES", "60")),
        promotion_min_requests=int(os.getenv("PROMOTION_MIN_REQUESTS", "100")),
        promotion_min_feedback=int(os.getenv("PROMOTION_MIN_FEEDBACK", "20")),
        promotion_max_p95_ms=float(os.getenv("PROMOTION_MAX_P95_MS", "100")),
        promotion_max_error_rate=float(os.getenv("PROMOTION_MAX_ERROR_RATE", "0.01")),
        promotion_min_top1_acceptance=float(os.getenv("PROMOTION_MIN_TOP1_ACCEPTANCE", "0.60")),
        promotion_min_top3_acceptance=float(os.getenv("PROMOTION_MIN_TOP3_ACCEPTANCE", "0.80")),
        rollback_min_requests=int(os.getenv("ROLLBACK_MIN_REQUESTS", "20")),
        rollback_min_feedback=int(os.getenv("ROLLBACK_MIN_FEEDBACK", "10")),
        rollback_max_p95_ms=float(os.getenv("ROLLBACK_MAX_P95_MS", "250")),
        rollback_max_error_rate=float(os.getenv("ROLLBACK_MAX_ERROR_RATE", "0.02")),
        rollback_min_top1_acceptance=float(os.getenv("ROLLBACK_MIN_TOP1_ACCEPTANCE", "0.45")),
        rollback_min_top3_acceptance=float(os.getenv("ROLLBACK_MIN_TOP3_ACCEPTANCE", "0.70")),
    )
