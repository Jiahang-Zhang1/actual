from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class _Model(BaseModel):
    model_config = ConfigDict(protected_namespaces=(), extra="ignore")


class PredictRequest(_Model):
    transaction_description: Optional[str] = Field(default=None, min_length=1)
    transaction_description_clean: Optional[str] = None
    merchant_text: Optional[str] = None

    country: Optional[str] = "US"
    currency: Optional[str] = "USD"

    amount: Optional[float] = None
    transaction_date: Optional[str] = None
    account_type: Optional[str] = None
    description_length: Optional[int] = None


class BatchPredictRequest(_Model):
    items: List[PredictRequest] = Field(default_factory=list)


class CategoryScore(_Model):
    category_id: str
    score: float


class PredictResponse(_Model):
    predicted_category_id: str
    confidence: float
    top_categories: List[CategoryScore]
    model_version: str


class PredictBatchResponse(_Model):
    items: List[PredictResponse]


class HealthResponse(_Model):
    status: str
    ready: bool
    backend_kind: str
    model_version: str
    code_version: str


class VersionResponse(_Model):
    backend_kind: str
    model_version: str
    code_version: str
    model_path: str
    source_model_path: str
    providers: List[str]
    hardware: str


class FeedbackRequest(_Model):
    transaction_id: str
    model_version: str
    predicted_category_id: str
    applied_category_id: str
    confidence: float | None = None
    candidate_category_ids: List[str] = Field(default_factory=list)


class FeedbackResponse(_Model):
    status: str
    saved: bool
