from __future__ import annotations

import pandas as pd

from data.data_generator import build_payload, sparse_payload
from data.data_quality_check import check_ingestion_quality, check_training_set_quality
from data.online_features import compute_features, format_for_serving


def test_online_features_preserve_manual_notes_fallback():
    features = compute_features(
        {
            "notes": "monthly internet bill",
            "amount": "78.44",
            "currency": "USD",
        }
    )

    assert features["transaction_description"] == "monthly internet bill"
    assert features["amount_bucket"] == "small"
    assert format_for_serving(features)["notes"] == "monthly internet bill"


def test_online_features_builds_sparse_account_amount_description():
    features = compute_features(
        {
            "account_id": "checking-account",
            "amount": 24.99,
            "currency": "USD",
        }
    )

    assert features["transaction_description"] == "account checking-account USD amount 24.99"
    assert (
        format_for_serving(features)["transaction_description"]
        == features["transaction_description"]
    )


def test_online_features_accepts_payee_alias_for_manual_entry():
    features = compute_features(
        {
            "payee": "LYFT RIDE HOME",
            "payment": "18.20",
            "currency": "USD",
        }
    )

    serving_payload = format_for_serving(features)
    assert serving_payload["transaction_description"] == "LYFT RIDE HOME"
    assert serving_payload["merchant_text"] == "LYFT RIDE HOME"
    assert serving_payload["amount"] == "18.20"


def test_data_generator_can_emit_sparse_manual_payloads():
    payloads = {
        variant: sparse_payload(variant)
        for variant in [
            "notes_only",
            "amount_only",
            "amount_only_positive_expense",
            "account_amount_only",
            "account_positive_expense",
            "payee_no_notes",
            "payee_positive_expense",
            "payee_amount_conflict",
            "payee_notes_conflict",
            "income_text_positive",
            "camel_case_actual_payload",
            "invalid_amount_string",
            "empty_payload",
        ]
    }

    assert payloads["empty_payload"] == {}
    assert "notes" in payloads["notes_only"]
    assert "amount" in payloads["amount_only"]
    assert payloads["amount_only_positive_expense"]["amount"] > 0
    assert "transaction_description" not in payloads["amount_only"]
    assert "account_id" in payloads["account_amount_only"]
    assert payloads["account_positive_expense"]["amount"] > 0
    assert payloads["payee_no_notes"]["notes"] == ""
    assert payloads["payee_positive_expense"]["amount"] > 0
    assert payloads["payee_notes_conflict"]["transaction_description"] == "LYFT RIDE"
    assert "PAY" in payloads["income_text_positive"]["transaction_description"]
    assert "transactionDescription" in payloads["camel_case_actual_payload"]
    assert payloads["invalid_amount_string"]["amount"] == "not-a-number"
    assert "transaction_description" in build_payload(sparse_rate=0.0)


def test_ingestion_quality_accepts_canonical_taxonomy_aliases(tmp_path):
    path = tmp_path / "transactions.csv"
    pd.DataFrame(
        [
            {
                "transaction_description": f"verizon bill {index}",
                "category": "Bills & Utilities",
                "country": "US",
                "currency": "USD",
            }
            for index in range(1000)
        ]
    ).to_csv(path, index=False)

    payload = check_ingestion_quality(path)

    assert payload["passed"]
    assert payload["metrics"]["alias_category_count"] == 1
    assert payload["metrics"]["normalized_category_distribution"]["Utilities & Services"] == 1000


def test_training_quality_rejects_unsupported_labels(tmp_path):
    train_path = tmp_path / "train.csv"
    pd.DataFrame(
        [
            {"transaction_description": f"known {index}", "category": "Food"}
            for index in range(50)
        ]
        + [
            {"transaction_description": "unsupported", "category": "Other"}
        ]
    ).to_csv(train_path, index=False)

    payload = check_training_set_quality(train_path)

    assert not payload["passed"]
    assert payload["metrics"]["unsupported_label_count"] == 1
