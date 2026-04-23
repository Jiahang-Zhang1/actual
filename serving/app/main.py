from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from app.config import get_settings
from app.feature_adapter import build_description, build_feature_frame, description_source
from app.postprocess import apply_confidence_policy, load_bundle_metadata
from app.runtime import (
    get_backend,
    refresh_backend_if_model_changed,
    reload_backend,
    warmup_backend,
)
from app.schemas import (
    BatchPredictRequest,
    CategoryScore,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    PredictRequest,
    PredictBatchResponse,
    PredictResponse,
    VersionResponse,
)
from app.telemetry import hardware_string

settings = get_settings()
app = FastAPI(
    title="ActualBudget Smart Transaction Categorization API",
    version=settings.code_version,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
Instrumentator().instrument(app).expose(app)

logger = logging.getLogger(__name__)

RUNTIME_DIR = Path(settings.runtime_dir)
FEEDBACK_LOG = RUNTIME_DIR / "feedback_events.jsonl"
REQUEST_LOG = RUNTIME_DIR / "request_events.jsonl"
PREDICTION_LOG = RUNTIME_DIR / "prediction_events.jsonl"
DATA_QUALITY_LOG = RUNTIME_DIR / "data_quality_events.jsonl"

prediction_confidence = Histogram(
    "prediction_confidence",
    "Model prediction confidence",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

predicted_class_total = Counter(
    "predicted_class_total",
    "Predicted class counts",
    ["category_id"],
)

feedback_total = Counter(
    "feedback_total",
    "User feedback counts",
    ["selected_category_id"],
)

feedback_match_total = Counter(
    "feedback_match_total",
    "Count where user picked model top1",
)

feedback_top3_match_total = Counter(
    "feedback_top3_match_total",
    "Count where user picked one of the model top-k candidates",
)

live_http_requests_total = Counter(
    "live_http_requests_total",
    "Live request counts for online monitoring and alerting.",
    ["handler", "status"],
)

live_http_request_duration_seconds = Histogram(
    "live_http_request_duration_seconds",
    "Live request latency for online monitoring and alerting.",
    ["handler"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

data_quality_pass = Gauge(
    "actual_data_quality_pass",
    "Whether the latest data quality check passed.",
    ["stage"],
)

data_quality_issue_count = Gauge(
    "actual_data_quality_issue_count",
    "Number of issues from the latest data quality check.",
    ["stage"],
)

data_quality_metric = Gauge(
    "actual_data_quality_metric",
    "Numeric metrics emitted by data quality checks.",
    ["stage", "metric"],
)

synthetic_request_total = Counter(
    "synthetic_request_total",
    "Synthetic requests excluded from online monitoring windows.",
    ["path", "source"],
)

synthetic_prediction_total = Counter(
    "synthetic_prediction_total",
    "Synthetic prediction items excluded from online monitoring windows.",
    ["source"],
)

synthetic_feedback_total = Counter(
    "synthetic_feedback_total",
    "Synthetic feedback events excluded from online monitoring windows.",
    ["source"],
)


def _current_settings():
    global settings
    refresh_backend_if_model_changed()
    settings = get_settings()
    return settings


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def _synthetic_source(request: Request) -> str | None:
    source = request.headers.get("x-actual-traffic-source", "").strip().lower()
    if not source:
        synthetic_header = request.headers.get("x-actual-synthetic-traffic", "").strip().lower()
        if synthetic_header in {"1", "true", "yes", "on"}:
            return "synthetic"
        return None
    if source in {"user", "production", "live"}:
        return None
    return source


def _ensure_runtime_dir() -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    _ensure_runtime_dir()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _safe_append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    try:
        _append_jsonl(path, row)
    except Exception:
        logger.exception("Failed to append JSONL event to %s", path)


def _iter_recent_events(path: Path, window_minutes: int):
    if not path.exists():
        return
    cutoff = datetime.now(UTC) - timedelta(minutes=window_minutes)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = _parse_ts(event.get("ts"))
            if ts is None or ts >= cutoff:
                yield event


def _request_summary(window_minutes: int) -> Dict[str, Any]:
    events = list(_iter_recent_events(REQUEST_LOG, window_minutes))
    if not events:
        return {
            "request_count": 0,
            "item_count": 0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "p99_latency_ms": 0.0,
            "error_rate": 0.0,
        }

    latencies = np.array([float(e.get("latency_ms", 0.0)) for e in events], dtype=float)
    item_count = int(sum(int(e.get("item_count", 1)) for e in events))
    errors = sum(1 for e in events if int(e.get("status_code", 500)) >= 400)
    total = len(events)
    return {
        "request_count": total,
        "item_count": item_count,
        "p50_latency_ms": round(float(np.percentile(latencies, 50)), 4),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 4),
        "p99_latency_ms": round(float(np.percentile(latencies, 99)), 4),
        "error_rate": round(errors / total, 4),
    }


def _prediction_summary(window_minutes: int) -> Dict[str, Any]:
    events = list(_iter_recent_events(PREDICTION_LOG, window_minutes))
    if not events:
        return {
            "prediction_count": 0,
            "avg_confidence": 0.0,
            "p10_confidence": 0.0,
            "predicted_category_counts": {},
        }

    confidences = np.array([float(e.get("confidence", 0.0)) for e in events], dtype=float)
    counts: Dict[str, int] = {}
    for event in events:
        category_id = str(event.get("predicted_category_id", "unknown"))
        counts[category_id] = counts.get(category_id, 0) + 1
    return {
        "prediction_count": len(events),
        "avg_confidence": round(float(np.mean(confidences)), 4),
        "p10_confidence": round(float(np.percentile(confidences, 10)), 4),
        "predicted_category_counts": counts,
    }


def _feedback_summary(window_minutes: int) -> Dict[str, Any]:
    events = list(_iter_recent_events(FEEDBACK_LOG, window_minutes))
    if not events:
        return {
            "feedback_count": 0,
            "top1_acceptance": 0.0,
            "top3_acceptance": 0.0,
            "selected_category_counts": {},
        }

    top1 = 0
    top3 = 0
    selected: Dict[str, int] = {}
    for event in events:
        applied = str(event.get("applied_category_id", ""))
        predicted = str(event.get("predicted_category_id", ""))
        candidates = [str(x) for x in event.get("candidate_category_ids", []) if x]
        if not candidates and predicted:
            candidates = [predicted]
        selected[applied] = selected.get(applied, 0) + 1
        if applied and applied == predicted:
            top1 += 1
        if applied and applied in candidates:
            top3 += 1
    total = len(events)
    return {
        "feedback_count": total,
        "top1_acceptance": round(top1 / total, 4),
        "top3_acceptance": round(top3 / total, 4),
        "selected_category_counts": selected,
    }


def _monitor_summary() -> Dict[str, Any]:
    current_settings = _current_settings()
    window_minutes = current_settings.monitor_window_minutes
    summary = {
        "backend_kind": current_settings.backend_kind,
        "model_version": current_settings.model_version,
        "code_version": current_settings.code_version,
        "rollout_context": current_settings.rollout_context,
        "window_minutes": window_minutes,
    }
    summary.update(_request_summary(window_minutes))
    summary.update(_prediction_summary(window_minutes))
    summary.update(_feedback_summary(window_minutes))
    summary["data_quality"] = _data_quality_summary(window_minutes)
    return summary


def _record_data_quality(payload: Dict[str, Any]) -> Dict[str, Any]:
    stage = str(payload.get("stage", "unknown"))
    passed = bool(payload.get("passed", False))
    issue_count = int(payload.get("issue_count", len(payload.get("issues", []))))
    metrics = payload.get("metrics", {})
    event = {
        "ts": _utc_now_iso(),
        "stage": stage,
        "passed": passed,
        "issue_count": issue_count,
        "issues": payload.get("issues", []),
        "metrics": metrics if isinstance(metrics, dict) else {},
    }

    data_quality_pass.labels(stage=stage).set(1 if passed else 0)
    data_quality_issue_count.labels(stage=stage).set(issue_count)
    if isinstance(metrics, dict):
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                data_quality_metric.labels(stage=stage, metric=str(key)).set(float(value))

    _safe_append_jsonl(DATA_QUALITY_LOG, event)
    return event


def _data_quality_summary(window_minutes: int) -> Dict[str, Any]:
    latest_by_stage: Dict[str, Dict[str, Any]] = {}
    for event in _iter_recent_events(DATA_QUALITY_LOG, window_minutes):
        stage = str(event.get("stage", "unknown"))
        latest_by_stage[stage] = event
    return latest_by_stage


def _monitor_decision(summary: Dict[str, Any]) -> Dict[str, Any]:
    current_settings = _current_settings()
    reasons: List[str] = []
    action = "hold"
    thresholds = {
        "promotion": {
            "min_requests": current_settings.promotion_min_requests,
            "min_feedback": current_settings.promotion_min_feedback,
            "max_p95_ms": current_settings.promotion_max_p95_ms,
            "max_error_rate": current_settings.promotion_max_error_rate,
            "min_top1_acceptance": current_settings.promotion_min_top1_acceptance,
            "min_top3_acceptance": current_settings.promotion_min_top3_acceptance,
        },
        "rollback": {
            "min_requests": current_settings.rollback_min_requests,
            "min_feedback": current_settings.rollback_min_feedback,
            "max_p95_ms": current_settings.rollback_max_p95_ms,
            "max_error_rate": current_settings.rollback_max_error_rate,
            "min_top1_acceptance": current_settings.rollback_min_top1_acceptance,
            "min_top3_acceptance": current_settings.rollback_min_top3_acceptance,
        },
    }

    if current_settings.rollout_context == "candidate":
        # Candidate context can only propose promotion after meeting quality and
        # stability thresholds with enough traffic and feedback.
        if summary["request_count"] < current_settings.promotion_min_requests:
            reasons.append("candidate sample size too small for promotion")
        if summary["feedback_count"] < current_settings.promotion_min_feedback:
            reasons.append("candidate feedback volume too small for promotion")
        if summary["p95_latency_ms"] > current_settings.promotion_max_p95_ms:
            reasons.append("candidate p95 latency above promotion threshold")
        if summary["error_rate"] > current_settings.promotion_max_error_rate:
            reasons.append("candidate error rate above promotion threshold")
        if summary["feedback_count"] >= current_settings.promotion_min_feedback:
            if summary["top1_acceptance"] < current_settings.promotion_min_top1_acceptance:
                reasons.append("candidate top1 acceptance below promotion threshold")
            if summary["top3_acceptance"] < current_settings.promotion_min_top3_acceptance:
                reasons.append("candidate top3 acceptance below promotion threshold")
        if not reasons:
            action = "promote_candidate"
            reasons.append("candidate met latency, error, and feedback thresholds")
    else:
        # Production context is conservative: trigger rollback when user quality
        # or operational SLOs fall below configured safety limits.
        if summary["request_count"] >= current_settings.rollback_min_requests:
            if summary["p95_latency_ms"] > current_settings.rollback_max_p95_ms:
                reasons.append("production p95 latency above rollback threshold")
            if summary["error_rate"] > current_settings.rollback_max_error_rate:
                reasons.append("production error rate above rollback threshold")
        else:
            reasons.append("production request volume too low for rollback decision")
        if summary["feedback_count"] >= current_settings.rollback_min_feedback:
            if summary["top1_acceptance"] < current_settings.rollback_min_top1_acceptance:
                reasons.append("production top1 acceptance below rollback threshold")
            if summary["top3_acceptance"] < current_settings.rollback_min_top3_acceptance:
                reasons.append("production top3 acceptance below rollback threshold")
        elif summary["request_count"] >= current_settings.rollback_min_requests:
            reasons.append("production feedback volume too low to evaluate acceptance")
        if any("rollback threshold" in r for r in reasons):
            action = "rollback_active"
    return {
        "recommended_action": action,
        "reasons": reasons,
        "thresholds": thresholds,
        "summary": summary,
    }


def _response_from_row(
    probabilities: np.ndarray,
    classes: List[str],
    *,
    record_observability: bool = True,
) -> PredictResponse:
    current_settings = _current_settings()
    ordered_idx = np.argsort(probabilities)[::-1][: current_settings.top_k]
    top_categories = [
        CategoryScore(category_id=str(classes[idx]), score=round(float(probabilities[idx]), 6))
        for idx in ordered_idx
    ]

    predicted_label = top_categories[0].category_id
    confidence = top_categories[0].score
    if record_observability:
        prediction_confidence.observe(confidence)
        predicted_class_total.labels(category_id=str(predicted_label)).inc()

    return PredictResponse(
        predicted_category_id=str(predicted_label),
        confidence=confidence,
        top_categories=top_categories,
        model_version=current_settings.model_version,
    )


def _predict_many(
    items: list[PredictRequest],
    *,
    record_observability: bool = True,
    synthetic_source: str | None = None,
) -> list[PredictResponse]:
    if not items:
        raise ValueError("items must not be empty")

    _current_settings()
    metadata = load_bundle_metadata(_current_settings().model_bundle_dir)
    backend = get_backend()
    frame = build_feature_frame(items)
    output = backend.predict(frame)
    responses = [
        _response_from_row(
            apply_confidence_policy(
                output.probabilities[idx],
                output.classes,
                description=build_description(items[idx]),
                description_source=description_source(items[idx]),
                amount=items[idx].amount,
                account_id=items[idx].account_id,
                metadata=metadata,
            ),
            output.classes,
            record_observability=record_observability,
        )
        for idx in range(len(items))
    ]
    if synthetic_source:
        synthetic_prediction_total.labels(source=synthetic_source).inc(len(responses))
    if record_observability:
        for response in responses:
            _safe_append_jsonl(
                PREDICTION_LOG,
                {
                    "ts": _utc_now_iso(),
                    "predicted_category_id": response.predicted_category_id,
                    "confidence": response.confidence,
                    "candidate_category_ids": [item.category_id for item in response.top_categories],
                    "model_version": response.model_version,
                },
            )
    return responses


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    synthetic_source = _synthetic_source(request)
    request.state.synthetic_source = synthetic_source
    request.state.record_observability = synthetic_source is None
    if synthetic_source:
        synthetic_request_total.labels(path=request.url.path, source=synthetic_source).inc()
    start = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        if request.url.path in {"/predict", "/predict_batch"} and getattr(
            request.state, "record_observability", True
        ):
            handler = request.url.path
            latency_ms = round((time.perf_counter() - start) * 1000.0, 4)
            status_label = f"{max(status_code, 500) // 100}xx"
            live_http_requests_total.labels(handler=handler, status=status_label).inc()
            live_http_request_duration_seconds.labels(handler=handler).observe(latency_ms / 1000.0)
            _safe_append_jsonl(
                REQUEST_LOG,
                {
                    "ts": _utc_now_iso(),
                    "path": handler,
                    "status_code": status_code,
                    "latency_ms": latency_ms,
                    "item_count": int(getattr(request.state, "item_count", 1)),
                },
            )


@app.on_event("startup")
def _startup() -> None:
    _ensure_runtime_dir()
    warmup_backend()


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    current_settings = _current_settings()
    return HealthResponse(
        status="ok",
        ready=True,
        backend_kind=current_settings.backend_kind,
        model_version=current_settings.model_version,
        code_version=current_settings.code_version,
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return healthz()


@app.get("/readyz", response_model=HealthResponse)
def readyz() -> HealthResponse:
    try:
        warmup_backend()
        current_settings = _current_settings()
        return HealthResponse(
            status="ready",
            ready=True,
            backend_kind=current_settings.backend_kind,
            model_version=current_settings.model_version,
            code_version=current_settings.code_version,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/versionz", response_model=VersionResponse)
def versionz() -> VersionResponse:
    current_settings = _current_settings()
    backend = get_backend()
    return VersionResponse(
        backend_kind=current_settings.backend_kind,
        model_version=current_settings.model_version,
        code_version=current_settings.code_version,
        model_path=current_settings.model_path,
        source_model_path=current_settings.source_model_path,
        providers=backend.providers(),
        hardware=hardware_string(),
    )


@app.post("/admin/reload-model")
def admin_reload_model() -> Dict[str, Any]:
    # Promotion and rollback jobs call this after updating the deployed model
    # artifact so serving can switch versions without manual pod restarts.
    global settings
    reload_backend()
    settings = get_settings()
    return {"status": "ok", "reloaded": True, "model_version": settings.model_version}


@app.get("/monitor/summary")
def monitor_summary() -> Dict[str, Any]:
    return _monitor_summary()


@app.get("/monitor/decision")
def monitor_decision() -> Dict[str, Any]:
    summary = _monitor_summary()
    return _monitor_decision(summary)


@app.post("/monitor/data-quality")
def monitor_data_quality(payload: Dict[str, Any]) -> Dict[str, Any]:
    # Data jobs post stage-level quality results here so serving owns one
    # Prometheus scrape target for model, feedback, rollout, and data health.
    return {"status": "ok", "saved": True, "event": _record_data_quality(payload)}


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(request: FeedbackRequest, raw_request: Request) -> FeedbackResponse:
    synthetic_source = getattr(raw_request.state, "synthetic_source", None)
    if synthetic_source:
        synthetic_feedback_total.labels(source=synthetic_source).inc()
        return FeedbackResponse(status="ok", saved=False)

    event = {
        "ts": _utc_now_iso(),
        "transaction_id": request.transaction_id,
        "model_version": request.model_version,
        "predicted_category_id": request.predicted_category_id,
        "applied_category_id": request.applied_category_id,
        "confidence": request.confidence,
        "candidate_category_ids": request.candidate_category_ids,
    }
    _safe_append_jsonl(FEEDBACK_LOG, event)
    feedback_total.labels(selected_category_id=request.applied_category_id).inc()
    if request.applied_category_id == request.predicted_category_id:
        feedback_match_total.inc()
    candidates = request.candidate_category_ids or [request.predicted_category_id]
    if request.applied_category_id in candidates:
        feedback_top3_match_total.inc()
    return FeedbackResponse(status="ok", saved=True)


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, raw_request: Request) -> PredictResponse:
    raw_request.state.item_count = 1
    try:
        return _predict_many(
            [body],
            record_observability=getattr(raw_request.state, "record_observability", True),
            synthetic_source=getattr(raw_request.state, "synthetic_source", None),
        )[0]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict_batch", response_model=PredictBatchResponse)
def predict_batch(body: BatchPredictRequest, raw_request: Request) -> PredictBatchResponse:
    raw_request.state.item_count = max(1, len(body.items))
    try:
        return PredictBatchResponse(
            items=_predict_many(
                body.items,
                record_observability=getattr(raw_request.state, "record_observability", True),
                synthetic_source=getattr(raw_request.state, "synthetic_source", None),
            )
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
