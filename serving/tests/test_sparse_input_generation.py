import sys
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.generate_sparse_input_matrix import (  # noqa: E402
    VARIANT_METADATA,
    VARIANT_NAMES,
    build_balanced_rows,
    build_sparse_matrix,
    variant_payload,
)
from serving.app.taxonomy import CANONICAL_CATEGORIES  # noqa: E402


class DummyBackendOutput:
    classes = ["Food & Dining", "Shopping & Retail", "Entertainment & Recreation"]

    def __init__(self, size: int):
        self.labels = ["Food & Dining"] * size
        self.probabilities = [[0.8, 0.1, 0.1] for _ in range(size)]


class DummyBackend:
    def predict(self, frame):
        return DummyBackendOutput(len(frame))

    def providers(self):
        return ["dummy"]


def test_build_balanced_rows_covers_each_canonical_category():
    rows = build_balanced_rows(rows_per_category=1, seed=9183)
    categories = sorted({row["category"] for row in rows})
    assert categories == sorted(CANONICAL_CATEGORIES)


def test_sparse_matrix_contains_all_variants_for_each_seed_row():
    rows_per_category = 1
    cases = build_sparse_matrix(rows_per_category=rows_per_category, seed=9183)
    assert len(cases) == len(CANONICAL_CATEGORIES) * len(VARIANT_NAMES)
    first_case_variants = {case["variant"] for case in cases[: len(VARIANT_NAMES)]}
    assert first_case_variants == set(VARIANT_NAMES)
    assert {case["blank_style"] for case in cases} == {
        metadata["blank_style"] for metadata in VARIANT_METADATA.values()
    }


def test_sparse_variants_include_whitespace_null_and_empty_shapes():
    row = build_balanced_rows(rows_per_category=1, seed=9183)[0]

    whitespace_all_text_amount = variant_payload("whitespace_all_text_amount", row)
    assert whitespace_all_text_amount["transaction_description"] == " "
    assert whitespace_all_text_amount["notes"] == "   "

    null_description_imported_fallback = variant_payload("null_description_imported_fallback", row)
    assert "transaction_description" in null_description_imported_fallback
    assert null_description_imported_fallback["transaction_description"] is None
    assert null_description_imported_fallback["imported_description"] == row["imported_description"]

    empty_payload = variant_payload("empty_payload", row)
    assert empty_payload == {}


def test_sparse_matrix_cases_satisfy_predict_and_predict_batch_contract(monkeypatch, tmp_path):
    monkeypatch.setattr("app.main.get_backend", lambda: DummyBackend())
    monkeypatch.setattr("app.main.RUNTIME_DIR", tmp_path)
    monkeypatch.setattr("app.main.FEEDBACK_LOG", tmp_path / "feedback_events.jsonl")
    monkeypatch.setattr("app.main.REQUEST_LOG", tmp_path / "request_events.jsonl")
    monkeypatch.setattr("app.main.PREDICTION_LOG", tmp_path / "prediction_events.jsonl")

    client = TestClient(app)
    headers = {"X-Actual-Traffic-Source": "sparse-matrix-test"}
    cases = build_sparse_matrix(rows_per_category=1, seed=9183)

    for case in cases:
        response = client.post("/predict", headers=headers, json=case["payload"])
        assert response.status_code == 200, case["case_id"]
        payload = response.json()
        assert isinstance(payload["predicted_category_id"], str), case["case_id"]
        assert payload["predicted_category_id"], case["case_id"]
        assert payload["confidence"] == payload["top_categories"][0]["score"]

    batch_response = client.post(
        "/predict_batch",
        headers=headers,
        json={"items": [case["payload"] for case in cases]},
    )
    assert batch_response.status_code == 200
    batch_payload = batch_response.json()
    assert len(batch_payload["items"]) == len(cases)
    assert all(
        item["confidence"] == item["top_categories"][0]["score"]
        for item in batch_payload["items"]
    )
