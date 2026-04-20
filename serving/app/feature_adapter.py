from __future__ import annotations

from typing import Iterable

import pandas as pd

from app.schemas import PredictRequest


def choose_description(item: PredictRequest) -> str:
    return (
        item.transaction_description
        or item.transaction_description_clean
        or item.merchant_text
        or ""
    ).strip()


def _amount_bucket(value: float | None) -> str:
    if value is None:
        return "unknown"
    amount = abs(float(value))
    if amount < 20:
        return "micro"
    if amount < 100:
        return "small"
    if amount < 500:
        return "medium"
    return "large"


def _weekday_token(value: str | None) -> str:
    if not value:
        return "unknown"
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return "unknown"
    return str(parsed.day_name()).lower()


def build_serving_feature_text(item: PredictRequest) -> str:
    # This mirrors training.build_model_frame so joblib, ONNX, and quantized
    # ONNX artifacts all receive the same production feature contract.
    parts = [
        choose_description(item).lower(),
        f"country={(item.country or 'US').lower()}",
        f"currency={(item.currency or 'USD').lower()}",
        f"amount_bucket={_amount_bucket(item.amount)}",
        f"weekday={_weekday_token(item.transaction_date)}",
    ]
    if item.account_id:
        parts.append(f"account={item.account_id.lower()}")
    if item.imported_description:
        parts.append(f"imported={item.imported_description.lower()}")
    if item.notes:
        parts.append(f"notes={item.notes.lower()}")
    return " ".join(part for part in parts if part)


def build_feature_frame(items: Iterable[PredictRequest]) -> pd.DataFrame:
    rows = []
    for item in items:
        rows.append(
            {
                "transaction_description": build_serving_feature_text(item),
                "country": item.country or "US",
                "currency": item.currency or "USD",
            }
        )
    return pd.DataFrame(rows, columns=["transaction_description", "country", "currency"])


def dataframe_to_onnx_inputs(frame: pd.DataFrame) -> dict:
    inputs = {}
    for column in ["transaction_description", "country", "currency"]:
        values = frame[column].fillna("").astype(str).to_numpy().reshape(-1, 1)
        inputs[column] = values
    return inputs
