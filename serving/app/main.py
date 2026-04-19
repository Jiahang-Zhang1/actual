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
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from app.config import get_settings
from app.feature_adapter import build_feature_frame, choose_description
from app.runtime import get_backend, warmup_backend
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


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


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
    window_minutes = settings.monitor_window_minutes
    summary = {
        "backend_kind": settings.backend_kind,
        "model_version": settings.model_version,
        "code_version": settings.code_version,
        "rollout_context": settings.rollout_context,
        "window_minutes": window_minutes,
    }
    summary.update(_request_summary(window_minutes))
    summary.update(_prediction_summary(window_minutes))
    summary.update(_feedback_summary(window_minutes))
    return summary


def _monitor_decision(summary: Dict[str, Any]) -> Dict[str, Any]:
    reasons: List[str] = []
    action = "hold"
    thresholds = {
        "promotion": {
            "min_requests": settings.promotion_min_requests,
            "min_feedback": settings.promotion_min_feedback,
            "max_p95_ms": settings.promotion_max_p95_ms,
            "max_error_rate": settings.promotion_max_error_rate,
            "min_top1_acceptance": settings.promotion_min_top1_acceptance,
            "min_top3_acceptance": settings.promotion_min_top3_acceptance,
        },
        "rollback": {
            "min_requests": settings.rollback_min_requests,
            "min_feedback": settings.rollback_min_feedback,
            "max_p95_ms": settings.rollback_max_p95_ms,
            "max_error_rate": settings.rollback_max_error_rate,
            "min_top1_acceptance": settings.rollback_min_top1_acceptance,
            "min_top3_acceptance": settings.rollback_min_top3_acceptance,
        },
    }

    if settings.rollout_context == "candidate":
        if summary["request_count"] < settings.promotion_min_requests:
            reasons.append("candidate sample size too small for promotion")
        if summary["feedback_count"] < settings.promotion_min_feedback:
            reasons.append("candidate feedback volume too small for promotion")
        if summary["p95_latency_ms"] > settings.promotion_max_p95_ms:
            reasons.append("candidate p95 latency above promotion threshold")
        if summary["error_rate"] > settings.promotion_max_error_rate:
            reasons.append("candidate error rate above promotion threshold")
        if summary["feedback_count"] >= settings.promotion_min_feedback:
            if summary["top1_acceptance"] < settings.promotion_min_top1_acceptance:
                reasons.append("candidate top1 acceptance below promotion threshold")
            if summary["top3_acceptance"] < settings.promotion_min_top3_acceptance:
                reasons.append("candidate top3 acceptance below promotion threshold")
        if not reasons:
            action = "promote_candidate"
            reasons.append("candidate met latency, error, and feedback thresholds")
    else:
        if summary["request_count"] >= settings.rollback_min_requests:
            if summary["p95_latency_ms"] > settings.rollback_max_p95_ms:
                reasons.append("production p95 latency above rollback threshold")
            if summary["error_rate"] > settings.rollback_max_error_rate:
                reasons.append("production error rate above rollback threshold")
        else:
            reasons.append("production request volume too low for rollback decision")
        if summary["feedback_count"] >= settings.rollback_min_feedback:
            if summary["top1_acceptance"] < settings.rollback_min_top1_acceptance:
                reasons.append("production top1 acceptance below rollback threshold")
            if summary["top3_acceptance"] < settings.rollback_min_top3_acceptance:
                reasons.append("production top3 acceptance below rollback threshold")
        elif summary["request_count"] >= settings.rollback_min_requests:
            reasons.append("production feedback volume too low to evaluate acceptance")
        if any("rollback threshold" in r for r in reasons):
            action = "rollback_active"
    return {
        "recommended_action": action,
        "reasons": reasons,
        "thresholds": thresholds,
        "summary": summary,
    }


def _response_from_row(label: str, probabilities: np.ndarray, classes: List[str]) -> PredictResponse:
    ordered_idx = np.argsort(probabilities)[::-1][: settings.top_k]
    top_categories = [
        CategoryScore(category_id=str(classes[idx]), score=round(float(probabilities[idx]), 6))
        for idx in ordered_idx
    ]

    predicted_idx = ordered_idx[0]
    predicted_label = label if label in classes else top_categories[0].category_id
    if predicted_label in classes:
        predicted_idx = classes.index(predicted_label)

    confidence = round(float(probabilities[predicted_idx]), 6)
    prediction_confidence.observe(confidence)
    predicted_class_total.labels(category_id=str(predicted_label)).inc()

    return PredictResponse(
        predicted_category_id=str(predicted_label),
        confidence=confidence,
        top_categories=top_categories,
        model_version=settings.model_version,
    )


def _predict_many(items: list[PredictRequest]) -> list[PredictResponse]:
    if not items:
        raise ValueError("items must not be empty")
    if any(not choose_description(item) for item in items):
        raise ValueError("transaction_description, transaction_description_clean, or merchant_text is required")

    backend = get_backend()
    frame = build_feature_frame(items)
    output = backend.predict(frame)
    responses = [
        _response_from_row(output.labels[idx], output.probabilities[idx], output.classes)
        for idx in range(len(items))
    ]
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
    start = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        if request.url.path in {"/predict", "/predict_batch"}:
            latency_ms = round((time.perf_counter() - start) * 1000.0, 4)
            _safe_append_jsonl(
                REQUEST_LOG,
                {
                    "ts": _utc_now_iso(),
                    "path": request.url.path,
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
    return HealthResponse(
        status="ok",
        ready=True,
        backend_kind=settings.backend_kind,
        model_version=settings.model_version,
        code_version=settings.code_version,
    )


@app.get("/readyz", response_model=HealthResponse)
def readyz() -> HealthResponse:
    try:
        warmup_backend()
        return HealthResponse(
            status="ready",
            ready=True,
            backend_kind=settings.backend_kind,
            model_version=settings.model_version,
            code_version=settings.code_version,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/versionz", response_model=VersionResponse)
def versionz() -> VersionResponse:
    backend = get_backend()
    return VersionResponse(
        backend_kind=settings.backend_kind,
        model_version=settings.model_version,
        code_version=settings.code_version,
        model_path=settings.model_path,
        source_model_path=settings.source_model_path,
        providers=backend.providers(),
        hardware=hardware_string(),
    )


@app.get("/monitor/summary")
def monitor_summary() -> Dict[str, Any]:
    return _monitor_summary()


@app.get("/monitor/decision")
def monitor_decision() -> Dict[str, Any]:
    summary = _monitor_summary()
    return _monitor_decision(summary)


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(request: FeedbackRequest) -> FeedbackResponse:
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
        return _predict_many([body])[0]
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
        return PredictBatchResponse(items=_predict_many(body.items))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
