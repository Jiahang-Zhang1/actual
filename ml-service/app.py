import json
import os
from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app


MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/models/current"))

REQUEST_COUNT = Counter(
    "actual_ml_predict_requests_total",
    "Total number of prediction requests handled by the ML service.",
)
REQUEST_ERRORS = Counter(
    "actual_ml_predict_errors_total",
    "Total number of failed prediction requests handled by the ML service.",
)
REQUEST_LATENCY = Histogram(
    "actual_ml_predict_latency_seconds",
    "Prediction latency for the ML service.",
)
PREDICTION_CONFIDENCE = Histogram(
    "actual_ml_prediction_confidence",
    "Distribution of top-1 model confidence.",
    buckets=(0.0, 0.25, 0.5, 0.65, 0.8, 0.9, 1.0),
)
PREDICTED_CATEGORY = Counter(
    "actual_ml_predicted_category_total",
    "Predicted top-1 categories returned by the ML service.",
    ["category_id"],
)
MODEL_LOADED = Gauge(
    "actual_ml_model_loaded",
    "Whether a model artifact is currently loaded.",
)


class PredictRequest(BaseModel):
    transaction_id: Optional[str] = None
    transaction_description: str = Field(
        validation_alias=AliasChoices("transaction_description", "transactionDescription")
    )
    country: Optional[str] = None
    currency: Optional[str] = None
    amount: Optional[float] = None
    transaction_date: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("transaction_date", "transactionDate"),
    )
    account_id: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("account_id", "accountId"),
    )
    notes: Optional[str] = None
    imported_description: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("imported_description", "importedDescription"),
    )

    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class ModelBundle:
    def __init__(self, pipeline: Any, metadata: dict[str, Any]):
        self.pipeline = pipeline
        self.metadata = metadata
        self.model_version = metadata.get("model_version", "unknown")

    def predict(self, payload: PredictRequest) -> dict[str, Any]:
        input_frame = pd.DataFrame(
            [
                {
                    "transaction_description": payload.transaction_description,
                    "country": payload.country or "unknown",
                    "currency": payload.currency or "unknown",
                    "amount": payload.amount,
                    "transaction_date": payload.transaction_date,
                    "account_id": payload.account_id,
                    "notes": payload.notes,
                    "imported_description": payload.imported_description,
                }
            ]
        )

        if not hasattr(self.pipeline, "predict_proba"):
            raise RuntimeError("Loaded model does not support predict_proba().")

        probabilities = self.pipeline.predict_proba(input_frame)[0]
        classes = list(self.pipeline.named_steps["clf"].classes_)
        scored = sorted(
            zip(classes, probabilities),
            key=lambda item: item[1],
            reverse=True,
        )
        top_categories = [
            {"category_id": category_id, "score": float(score)}
            for category_id, score in scored[:3]
        ]
        predicted_category_id, confidence = scored[0]
        return {
            "predicted_category_id": predicted_category_id,
            "confidence": float(confidence),
            "top_categories": top_categories,
            "model_version": self.model_version,
        }


model_bundle: Optional[ModelBundle] = None
app = FastAPI(title="Actual Smart Transaction Categorizer", version="1.0.0")
app.mount("/metrics", make_asgi_app())


def load_model_from_disk() -> Optional[ModelBundle]:
    model_path = MODEL_DIR / "model.joblib"
    metadata_path = MODEL_DIR / "metadata.json"
    if not model_path.exists():
        return None

    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())

    pipeline = joblib.load(model_path)
    return ModelBundle(pipeline=pipeline, metadata=metadata)


@app.on_event("startup")
def startup_event():
    global model_bundle
    model_bundle = load_model_from_disk()
    MODEL_LOADED.set(1 if model_bundle is not None else 0)


@app.get("/health")
def health():
    return {
        "status": "ok" if model_bundle is not None else "no-model",
        "model_version": None if model_bundle is None else model_bundle.model_version,
    }


@app.post("/predict")
def predict(payload: PredictRequest):
    global model_bundle
    REQUEST_COUNT.inc()

    if model_bundle is None:
        model_bundle = load_model_from_disk()
        MODEL_LOADED.set(1 if model_bundle is not None else 0)

    if model_bundle is None:
        REQUEST_ERRORS.inc()
        raise HTTPException(status_code=503, detail="Model artifact is not loaded.")

    if not payload.transaction_description.strip():
        REQUEST_ERRORS.inc()
        raise HTTPException(status_code=400, detail="transaction_description is required.")

    with REQUEST_LATENCY.time():
        try:
            result = model_bundle.predict(payload)
        except Exception as exc:  # pragma: no cover - defensive path
            REQUEST_ERRORS.inc()
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    PREDICTION_CONFIDENCE.observe(result["confidence"])
    PREDICTED_CATEGORY.labels(result["predicted_category_id"]).inc()
    return result
