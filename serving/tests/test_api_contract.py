from fastapi.testclient import TestClient

from app.main import app


class DummyBackendOutput:
    classes = ["Food & Dining", "Shopping & Retail", "Transportation"]

    def __init__(self, size: int = 1):
        self.labels = ["Food & Dining"] * size
        self.probabilities = [[0.8, 0.1, 0.1] for _ in range(size)]


class DummyBackend:
    def predict(self, frame):
        return DummyBackendOutput(len(frame))

    def providers(self):
        return ["dummy"]


def test_predict_endpoint_contract(monkeypatch):
    monkeypatch.setattr("app.main.get_backend", lambda: DummyBackend())
    client = TestClient(app)
    response = client.post(
        "/predict",
        json={
            "transaction_description": "STARBUCKS STORE 1458 NEW YORK NY",
            "country": "US",
            "currency": "USD",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["predicted_category_id"]
    assert len(payload["top_categories"]) == 3


def test_predict_endpoint_contract_accepts_sparse_manual_entry(monkeypatch, tmp_path):
    monkeypatch.setattr("app.main.get_backend", lambda: DummyBackend())
    monkeypatch.setattr("app.main.RUNTIME_DIR", tmp_path)
    monkeypatch.setattr("app.main.REQUEST_LOG", tmp_path / "request_events.jsonl")
    monkeypatch.setattr("app.main.PREDICTION_LOG", tmp_path / "prediction_events.jsonl")
    client = TestClient(app)
    response = client.post(
        "/predict",
        headers={"X-Actual-Traffic-Source": "contract-test"},
        json={
            "account_id": "checking-account",
            "currency": "USD",
            "amount": 24.99,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload["predicted_category_id"], str)
    assert payload["predicted_category_id"]
    assert payload["confidence"] == payload["top_categories"][0]["score"]
    assert len(payload["top_categories"]) == 3


def test_predict_endpoint_contract_accepts_loose_manual_payload(monkeypatch, tmp_path):
    monkeypatch.setattr("app.main.get_backend", lambda: DummyBackend())
    monkeypatch.setattr("app.main.RUNTIME_DIR", tmp_path)
    monkeypatch.setattr("app.main.REQUEST_LOG", tmp_path / "request_events.jsonl")
    monkeypatch.setattr("app.main.PREDICTION_LOG", tmp_path / "prediction_events.jsonl")

    client = TestClient(app)
    response = client.post(
        "/predict",
        headers={"X-Actual-Traffic-Source": "contract-test"},
        json={
            "foo": "bar",
            "memo": "teacher typed this by hand",
            "amount": "",
            "descriptionLength": "",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["predicted_category_id"]
    assert payload["confidence"] == payload["top_categories"][0]["score"]


def test_predict_endpoint_contract_accepts_camel_case_fields(monkeypatch, tmp_path):
    monkeypatch.setattr("app.main.get_backend", lambda: DummyBackend())
    monkeypatch.setattr("app.main.RUNTIME_DIR", tmp_path)
    monkeypatch.setattr("app.main.REQUEST_LOG", tmp_path / "request_events.jsonl")
    monkeypatch.setattr("app.main.PREDICTION_LOG", tmp_path / "prediction_events.jsonl")

    client = TestClient(app)
    response = client.post(
        "/predict",
        headers={"X-Actual-Traffic-Source": "contract-test"},
        json={
            "merchantText": "LYFT RIDE",
            "accountId": "credit-card",
            "transactionAmount": "$18.20",
            "transactionDate": "2026-04-27",
            "currency": "",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["predicted_category_id"] == "Transportation"


def test_predict_endpoint_contract_keeps_positive_lyft_manual_entry_transport(monkeypatch, tmp_path):
    monkeypatch.setattr("app.main.get_backend", lambda: DummyBackend())
    monkeypatch.setattr("app.main.RUNTIME_DIR", tmp_path)
    monkeypatch.setattr("app.main.REQUEST_LOG", tmp_path / "request_events.jsonl")
    monkeypatch.setattr("app.main.PREDICTION_LOG", tmp_path / "prediction_events.jsonl")

    client = TestClient(app)
    response = client.post(
        "/predict",
        headers={"X-Actual-Traffic-Source": "contract-test"},
        json={
            "payee": "LYFT RIDE HOME",
            "payment": "18.20",
            "memo": "",
            "currency": "USD",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["predicted_category_id"] == "Transportation"
    assert payload["confidence"] <= 0.8


def test_predict_endpoint_contract_accepts_payee_and_memo_aliases(monkeypatch, tmp_path):
    monkeypatch.setattr("app.main.get_backend", lambda: DummyBackend())
    monkeypatch.setattr("app.main.RUNTIME_DIR", tmp_path)
    monkeypatch.setattr("app.main.REQUEST_LOG", tmp_path / "request_events.jsonl")
    monkeypatch.setattr("app.main.PREDICTION_LOG", tmp_path / "prediction_events.jsonl")

    client = TestClient(app)
    response = client.post(
        "/predict",
        headers={"X-Actual-Traffic-Source": "contract-test"},
        json={
            "payee": "STARBUCKS NO NOTES",
            "memo": "",
            "payment": "7.50",
            "extra_professor_field": "ignored",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["predicted_category_id"] == "Food & Dining"
    assert payload["confidence"] == payload["top_categories"][0]["score"]


def test_predict_batch_endpoint_contract_with_online_features_shape(monkeypatch):
    monkeypatch.setattr("app.main.get_backend", lambda: DummyBackend())
    client = TestClient(app)
    response = client.post(
        "/predict_batch",
        json={
            "items": [
                {
                    "transaction_description": "STARBUCKS STORE 1458 NEW YORK NY",
                    "transaction_description_clean": "starbucks store 1458 new york ny",
                    "country": "US",
                    "currency": "USD",
                    "description_length": 6,
                }
            ]
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert "items" in payload
    assert payload["items"][0]["predicted_category_id"] == "Food & Dining"


def test_predict_batch_endpoint_contract_accepts_transactions_alias(monkeypatch, tmp_path):
    monkeypatch.setattr("app.main.get_backend", lambda: DummyBackend())
    monkeypatch.setattr("app.main.RUNTIME_DIR", tmp_path)
    monkeypatch.setattr("app.main.REQUEST_LOG", tmp_path / "request_events.jsonl")
    monkeypatch.setattr("app.main.PREDICTION_LOG", tmp_path / "prediction_events.jsonl")

    client = TestClient(app)
    response = client.post(
        "/predict_batch",
        headers={"X-Actual-Traffic-Source": "contract-test"},
        json={
            "transactions": [
                {"amount": "", "foo": "bar"},
                {"merchantText": "SHAKE SHACK", "transactionAmount": "12.34"},
            ]
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["items"]) == 2
    assert all(item["predicted_category_id"] for item in payload["items"])


def test_feedback_and_monitor_summary(monkeypatch, tmp_path):
    monkeypatch.setattr("app.main.get_backend", lambda: DummyBackend())
    monkeypatch.setattr("app.main.RUNTIME_DIR", tmp_path)
    monkeypatch.setattr("app.main.FEEDBACK_LOG", tmp_path / "feedback_events.jsonl")
    monkeypatch.setattr("app.main.REQUEST_LOG", tmp_path / "request_events.jsonl")
    monkeypatch.setattr("app.main.PREDICTION_LOG", tmp_path / "prediction_events.jsonl")
    client = TestClient(app)

    client.post(
        "/feedback",
        json={
            "transaction_id": "tx-1",
            "model_version": "v1",
            "predicted_category_id": "Food & Dining",
            "applied_category_id": "Food & Dining",
            "confidence": 0.8,
            "candidate_category_ids": ["Food & Dining", "Shopping & Retail"],
        },
    )
    summary = client.get("/monitor/summary")
    assert summary.status_code == 200
    payload = summary.json()
    assert payload["feedback_count"] == 1
    assert payload["top1_acceptance"] == 1.0
    assert payload["top3_acceptance"] == 1.0


def test_synthetic_traffic_is_excluded_from_monitor_summary(monkeypatch, tmp_path):
    monkeypatch.setattr("app.main.get_backend", lambda: DummyBackend())
    monkeypatch.setattr("app.main.RUNTIME_DIR", tmp_path)
    monkeypatch.setattr("app.main.FEEDBACK_LOG", tmp_path / "feedback_events.jsonl")
    monkeypatch.setattr("app.main.REQUEST_LOG", tmp_path / "request_events.jsonl")
    monkeypatch.setattr("app.main.PREDICTION_LOG", tmp_path / "prediction_events.jsonl")
    client = TestClient(app)
    headers = {"X-Actual-Traffic-Source": "benchmark"}

    predict = client.post(
        "/predict",
        headers=headers,
        json={
            "transaction_description": "STARBUCKS STORE 1458 NEW YORK NY",
            "country": "US",
            "currency": "USD",
        },
    )
    assert predict.status_code == 200

    feedback = client.post(
        "/feedback",
        headers=headers,
        json={
            "transaction_id": "tx-synthetic",
            "model_version": "v1",
            "predicted_category_id": "Food & Dining",
            "applied_category_id": "Food & Dining",
            "confidence": 0.8,
            "candidate_category_ids": ["Food & Dining", "Shopping & Retail"],
        },
    )
    assert feedback.status_code == 200
    assert feedback.json()["saved"] is False

    summary = client.get("/monitor/summary")
    assert summary.status_code == 200
    payload = summary.json()
    assert payload["request_count"] == 0
    assert payload["prediction_count"] == 0
    assert payload["feedback_count"] == 0
