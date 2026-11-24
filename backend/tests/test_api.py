import pytest
from fastapi.testclient import TestClient

from app import main
from app.services.otp import otp_service

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main.config, "DB_PATH", tmp_path / "test.db")
    main.ensure_model()
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
    response = client.post("/api/v1/payments/check", json=request("bob", 25000))
    assert response.status_code == 200
    result = response.json()
    assert result["decision"] == "VERIFY"
    assert 40 <= result["risk_score"] <= 69
    assert result["engine"] == f"FastAPI + {main.model_data['model_name']}"
    assert "The payment matches your usual pattern" not in result["reasons"]

def test_block_decision(client):
    response = client.post("/api/v1/payments/check", json=request("carol", 60000, {"new_device": True, "emulator": True}))
    assert response.status_code == 200
    assert response.json()["risk_score"] == 100
    assert response.json()["decision"] == "BLOCK"
    assert "The payment matches your usual pattern" not in response.json()["reasons"]

def test_otp_attempt_limit(client):
    result = client.post("/api/v1/payments/check", json=request("dave", 10000, {"vpn": True})).json()
    txid = result["transaction_id"]
    sent = client.post("/api/v1/otp/send", json={"user_id": "dave", "transaction_id": txid})
    assert sent.status_code == 200
    wrong_code = "000000" if sent.json()["otp"] != "000000" else "000001"
    for _ in range(3):
        response = client.post("/api/v1/otp/verify", json={"user_id": "dave", "transaction_id": txid, "otp": wrong_code})
        assert response.status_code == 400
    last = client.post("/api/v1/otp/verify", json={"user_id": "dave", "transaction_id": txid, "otp": sent.json()["otp"]})
    assert last.status_code == 429

def test_verify_sweep_real_model(client):
    # Each row is an unseen receiver/user. The actual fitted estimator is loaded
    # by the fixture and each profile must remain in the verification band.
    for amount in (10_000, 25_000, 50_000):
        result = client.post("/api/v1/payments/check", json=request(f"sweep_{amount}", amount)).json()
        assert result["decision"] == "VERIFY", (amount, result)
        assert 40 <= result["risk_score"] <= 69
    for amount in (1_000, 5_000, 25_000, 50_000):
        user = f"vpn_{amount}"
        payload = request(user, 500)
        payload["receiver_id"] = f"receiver-{amount}@bank"
        assert client.post("/api/v1/payments/check", json=payload).status_code == 200
        payload = request(user, amount, {"vpn": True})
        payload["receiver_id"] = f"receiver-{amount}@bank"
        result = client.post("/api/v1/payments/check", json=payload).json()
        assert result["decision"] == "VERIFY", (amount, result)
        assert 40 <= result["risk_score"] <= 69
