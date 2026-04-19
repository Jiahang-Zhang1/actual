from fastapi.testclient import TestClient

from app.main import app


class DummyBackendOutput:
    labels = ["Food & Dining"]
    probabilities = [[0.8, 0.1, 0.1]]
    classes = ["Food & Dining", "Shopping & Retail", "Entertainment & Recreation"]


class DummyBackend:
    def predict(self, frame):
        return DummyBackendOutput()

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
    assert payload["predicted_category_id"] == "Food & Dining"
    assert len(payload["top_categories"]) == 3


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
