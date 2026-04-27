from __future__ import annotations

from typing import Any, List, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator


class _Model(BaseModel):
    model_config = ConfigDict(protected_namespaces=(), extra="ignore")


class PredictRequest(_Model):
    model_config = ConfigDict(
        protected_namespaces=(),
        extra="ignore",
        populate_by_name=True,
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

    transaction_description: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("transaction_description", "transactionDescription"),
    )
    transaction_description_clean: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "transaction_description_clean",
            "transactionDescriptionClean",
        ),
    )
    merchant_text: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("merchant_text", "merchantText"),
    )

    country: Optional[str] = "US"
    currency: Optional[str] = "USD"

    amount: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices("amount", "transaction_amount", "transactionAmount"),
    )
    transaction_date: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("transaction_date", "transactionDate"),
    )
    account_type: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("account_type", "accountType"),
    )
    account_id: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("account_id", "accountId", "account"),
    )
    imported_description: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("imported_description", "importedDescription"),
    )
    notes: Optional[str] = None
    description_length: Optional[int] = Field(
        default=None,
        validation_alias=AliasChoices("description_length", "descriptionLength"),
    )

    @field_validator(
        "transaction_description",
        "transaction_description_clean",
        "merchant_text",
        "country",
        "currency",
        "transaction_date",
        "account_type",
        "account_id",
        "imported_description",
        "notes",
        mode="before",
    )
    @classmethod
    def _coerce_optional_text(cls, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).replace("\n", " ").strip()
        return text or None

    @field_validator("amount", mode="before")
    @classmethod
    def _coerce_amount(cls, value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, str):
            cleaned = value.strip().replace(",", "")
            if cleaned.startswith("$"):
                cleaned = cleaned[1:].strip()
            if cleaned.startswith("(") and cleaned.endswith(")"):
                cleaned = f"-{cleaned[1:-1]}"
            if not cleaned:
                return None
            value = cleaned
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @field_validator("description_length", mode="before")
    @classmethod
    def _coerce_description_length(cls, value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None


class BatchPredictRequest(_Model):
    model_config = ConfigDict(
        protected_namespaces=(),
        extra="ignore",
        populate_by_name=True,
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

    items: List[PredictRequest] = Field(
        default_factory=list,
        validation_alias=AliasChoices("items", "transactions", "records"),
    )

    @field_validator("items", mode="before")
    @classmethod
    def _coerce_items(cls, value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, dict):
            return [value]
        return value


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
