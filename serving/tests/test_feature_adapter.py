from app.feature_adapter import build_feature_frame
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
    assert frame.iloc[0]["transaction_description"] == "NETFLIX.COM"


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
    assert frame.iloc[0]["transaction_description"] == "starbucks store 1458"
