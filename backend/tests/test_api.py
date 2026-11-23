import pytest
from fastapi.testclient import TestClient

from app import main
from app.services.otp import otp_service

class FakeModel:
    def predict_proba(self, rows):
        out = []
        for row in rows:
            # index 3 is new receiver; keep test outcomes stable while still
            # exercising the real scoring blend and hard-block rule.
            p = 0.01 if row[0] < 1000 else (0.50 if row[0] < 50000 else 0.99)
            out.append([1-p, p])
        return out

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main.config, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(main, "ensure_model", lambda: None)
    monkeypatch.setattr(main, "model_data", {"model": FakeModel(), "model_name": "Test model",
        "features": main.config.FEATURES, "metrics": {"roc_auc": 1.0}})
    receiver_events, user_events = main.receiver_events, main.user_events
    receiver_events.clear(); user_events.clear(); otp_service.items.clear()
    with TestClient(main.app) as c:
        yield c

def request(user="alice", amount=500, signals=None):
    return {"user_id": user, "receiver_id": "merchant@bank", "receiver_name": "Merchant", "amount": amount,
            "hour": 13, "signals": signals or {}}

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_allow_decision(client):
    response = client.post("/api/v1/payments/check", json=request())
    assert response.status_code == 200
    assert response.json()["decision"] == "ALLOW"

def test_verify_decision(client):
    response = client.post("/api/v1/payments/check", json=request("bob", 3000, {"vpn": True}))
    assert response.status_code == 200
    assert response.json()["decision"] == "VERIFY"

def test_block_decision(client):
    response = client.post("/api/v1/payments/check", json=request("carol", 60000, {"new_device": True, "emulator": True}))
    assert response.status_code == 200
    assert response.json()["risk_score"] == 100
    assert response.json()["decision"] == "BLOCK"

def test_otp_attempt_limit(client):
    result = client.post("/api/v1/payments/check", json=request("dave", 3000, {"vpn": True})).json()
    txid = result["transaction_id"]
    sent = client.post("/api/v1/otp/send", json={"user_id": "dave", "transaction_id": txid})
    assert sent.status_code == 200
    wrong_code = "000000" if sent.json()["otp"] != "000000" else "000001"
    for _ in range(3):
        response = client.post("/api/v1/otp/verify", json={"user_id": "dave", "transaction_id": txid, "otp": wrong_code})
        assert response.status_code == 400
    last = client.post("/api/v1/otp/verify", json={"user_id": "dave", "transaction_id": txid, "otp": sent.json()["otp"]})
    assert last.status_code == 429
