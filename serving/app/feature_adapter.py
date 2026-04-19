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


def build_feature_frame(items: Iterable[PredictRequest]) -> pd.DataFrame:
    rows = []
    for item in items:
        rows.append(
            {
                "transaction_description": choose_description(item),
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
