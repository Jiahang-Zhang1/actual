from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class _Model(BaseModel):
    model_config = ConfigDict(protected_namespaces=(), extra="ignore")


class PredictRequest(_Model):
    model_config = ConfigDict(
        protected_namespaces=(),
        extra="ignore",
        json_schema_extra={
            "examples": [
                {
                    "transaction_description": "STARBUCKS STORE 1458 NEW YORK NY",
                    "country": "US",
                    "currency": "USD",
                    "amount": 5.75,
                    "notes": "coffee",
                },
                {
                    "notes": "monthly phone bill",
                    "currency": "USD",
                    "amount": 78.44,
                },
                {
                    "account_id": "checking-account",
                    "currency": "USD",
                    "amount": 24.99,
                },
            ]
        },
    )

    transaction_description: Optional[str] = None
    transaction_description_clean: Optional[str] = None
    merchant_text: Optional[str] = None

    country: Optional[str] = "US"
    currency: Optional[str] = "USD"

    amount: Optional[float] = None
    transaction_date: Optional[str] = None
    account_type: Optional[str] = None
    account_id: Optional[str] = None
    imported_description: Optional[str] = None
    notes: Optional[str] = None
    description_length: Optional[int] = None


class BatchPredictRequest(_Model):
    model_config = ConfigDict(
        protected_namespaces=(),
        extra="ignore",
        json_schema_extra={
            "examples": [
                {
                    "items": [
                        {
                            "transaction_description": "STARBUCKS STORE 1458 NEW YORK NY",
                            "country": "US",
                            "currency": "USD",
                        },
                        {
                            "notes": "uber ride home",
                            "currency": "USD",
                            "amount": 18.2,
                        },
                    ]
                }
            ]
        },
    )

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
