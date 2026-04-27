from __future__ import annotations

import os
import json
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


def _resolve_path_from_manifest(manifest_path: Path, raw_path: object, fallback_path: str) -> str:
    if not raw_path:
        return fallback_path
    path = Path(str(raw_path))
    if path.is_absolute():
        return str(path)
    return str((manifest_path.parent / path).resolve())


def _manifest_candidates(
    fallback_model_path: str,
    fallback_source_model_path: str,
    model_bundle_dir: str,
) -> list[Path]:
    candidates = [
        Path(model_bundle_dir) / "manifest.json",
        Path(fallback_model_path).parent.parent / "manifest.json",
        Path(fallback_source_model_path).parent.parent / "manifest.json",
    ]
    seen: set[Path] = set()
    unique_candidates: list[Path] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique_candidates.append(candidate)
    return unique_candidates


def _resolve_manifest_selection(
    requested_backend_kind: str,
    fallback_model_path: str,
    fallback_source_model_path: str,
    fallback_model_version: str,
    model_bundle_dir: str,
) -> tuple[str, str, str, str] | None:
    for manifest_path in _manifest_candidates(
        fallback_model_path,
        fallback_source_model_path,
        model_bundle_dir,
    ):
        if not manifest_path.exists():
            continue

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        backend_kind = "onnx_dynamic_quant" if requested_backend_kind == "auto" else requested_backend_kind
        optimized_paths = manifest.get("optimized_model_paths") or {}
        source_model_path = _resolve_path_from_manifest(
            manifest_path,
            manifest.get("source_model_path"),
            fallback_source_model_path,
        )
        model_path = source_model_path
        if backend_kind != "baseline":
            model_path = _resolve_path_from_manifest(
                manifest_path,
                optimized_paths.get(backend_kind),
                fallback_model_path,
            )
        model_version = str(manifest.get("model_version") or fallback_model_version)
        return backend_kind, model_path, source_model_path, model_version

    return None


def _resolve_bundle_selection(
    requested_backend_kind: str,
    fallback_model_path: str,
    fallback_source_model_path: str,
    fallback_model_version: str,
    model_bundle_dir: str,
) -> tuple[str, str, str, str]:
    bundle_dir = Path(model_bundle_dir)
    selection_path = bundle_dir / "selected_model.json"
    metadata_path = bundle_dir / "metadata.json"
    if not selection_path.exists():
        manifest_selection = _resolve_manifest_selection(
            requested_backend_kind,
            fallback_model_path,
            fallback_source_model_path,
            fallback_model_version,
            model_bundle_dir,
        )
        if manifest_selection is not None:
            return manifest_selection
        backend_kind = "onnx_dynamic_quant" if requested_backend_kind == "auto" else requested_backend_kind
        return backend_kind, fallback_model_path, fallback_source_model_path, fallback_model_version

    # A promoted model bundle includes all variants plus selected_model.json.
    # Serving reads it at startup/reload so promotion can change both model
    # version and artifact variant without rebuilding the container.
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    selected_variant = str(selection.get("selected_variant") or "baseline")
    backend_kind = selected_variant if requested_backend_kind == "auto" else requested_backend_kind
    paths = selection.get("paths", {})
    variant_path = paths.get(backend_kind) or paths.get(selected_variant) or "model.joblib"
    source_path = paths.get("baseline") or "model.joblib"
    model_version = str(metadata.get("model_version") or selection.get("model_version") or fallback_model_version)
    return (
        backend_kind,
        str((bundle_dir / variant_path).resolve()),
        str((bundle_dir / source_path).resolve()),
        model_version,
    )


@dataclass(frozen=True)
class Settings:
    backend_kind: str
    model_path: str
    source_model_path: str
    model_bundle_dir: str
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
    requested_backend_kind = os.getenv("BACKEND_KIND", "onnx_dynamic_quant").strip()
    default_backend_kind = "onnx_dynamic_quant" if requested_backend_kind == "auto" else requested_backend_kind
    fallback_model_path = os.getenv("MODEL_PATH", "").strip() or default_model_path(default_backend_kind)
    fallback_source_model_path = os.getenv(
        "SOURCE_MODEL_PATH",
        "/workspace/models/source/v2_tfidf_linearsvc_model.joblib",
    ).strip()
    fallback_model_version = os.getenv("MODEL_VERSION", "v2_tfidf_linearsvc").strip()
    model_bundle_dir = os.getenv("MODEL_BUNDLE_DIR", "/workspace/runtime/deployed").strip()
    backend_kind, model_path, source_model_path, model_version = _resolve_bundle_selection(
        requested_backend_kind,
        fallback_model_path,
        fallback_source_model_path,
        fallback_model_version,
        model_bundle_dir,
    )
    return Settings(
        backend_kind=backend_kind,
        model_path=model_path,
        source_model_path=source_model_path,
        model_bundle_dir=model_bundle_dir,
        model_version=model_version,
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
