from app.feature_adapter import build_feature_frame, description_source
from app.schemas import PredictRequest


def test_build_feature_frame_uses_expected_columns():
    frame = build_feature_frame(
        [
            PredictRequest(
                transaction_description="STARBUCKS STORE 1458 NEW YORK NY",
                country="US",
                currency="USD",
            )
        ]
    )
    assert list(frame.columns) == ["transaction_description", "country", "currency"]
    assert frame.iloc[0]["country"] == "US"
    assert frame.iloc[0]["currency"] == "USD"


def test_build_feature_frame_falls_back_to_merchant_text():
    frame = build_feature_frame([PredictRequest(merchant_text="NETFLIX.COM", country="US", currency="USD")])
    description = frame.iloc[0]["transaction_description"]
    assert description.startswith("netflix.com")
    assert "country=us" in description
    assert "currency=usd" in description


def test_build_feature_frame_accepts_clean_description():
    frame = build_feature_frame(
        [
            PredictRequest(
                transaction_description_clean="starbucks store 1458",
                country="US",
                currency="USD",
            )
        ]
    )
    description = frame.iloc[0]["transaction_description"]
    assert description.startswith("starbucks store 1458")
    assert "amount_bucket=unknown" in description


def test_build_feature_frame_uses_notes_for_sparse_manual_entry():
    frame = build_feature_frame(
        [
            PredictRequest(
                notes="monthly phone bill",
                currency="USD",
                amount=78.44,
            )
        ]
    )
    description = frame.iloc[0]["transaction_description"]
    assert description.startswith("monthly phone bill")
    assert "description_source=notes" in description
    assert "amount_bucket=small" in description


def test_build_feature_frame_falls_back_to_manual_entry_when_text_is_missing():
    frame = build_feature_frame(
        [
            PredictRequest(
                account_id="checking-account",
                currency="USD",
                amount=24.99,
            )
        ]
    )
    description = frame.iloc[0]["transaction_description"]
    assert description.startswith("manual entry")
    assert "description_source=derived" in description
    assert "amount_bucket=small" in description


def test_generated_account_amount_fallback_is_treated_as_derived():
    request = PredictRequest(
        transaction_description="account checking-account USD amount 24.99",
        account_id="checking-account",
        currency="USD",
        amount=24.99,
    )

    assert description_source(request) == "derived"
    frame = build_feature_frame([request])
    description = frame.iloc[0]["transaction_description"]
    assert description.startswith("account checking-account usd amount 24.99")
    assert "description_source=derived" in description
