from __future__ import annotations
import json
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import joblib
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from . import config
from .schemas import PaymentCheck, OTPSend, OTPVerify, PaymentConfirm, Feedback, Investigate
from .services.otp import otp_service
from .services.scoring import score_payment
from .services.agent import investigate

model_data = None
receiver_events: dict[str, list[tuple[float, str]]] = {}
user_events: dict[str, list[float]] = {}

def connect():
    conn = sqlite3.connect(config.DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with connect() as db:
        db.execute("""CREATE TABLE IF NOT EXISTS transactions (
            id TEXT PRIMARY KEY, user_id TEXT NOT NULL, receiver_id TEXT NOT NULL, receiver_name TEXT NOT NULL,
            amount REAL NOT NULL, merchant_domain TEXT, risk_score REAL NOT NULL, decision TEXT NOT NULL,
            reasons TEXT NOT NULL, next_step TEXT NOT NULL, created_at TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
            feedback TEXT, features TEXT NOT NULL)""")
        db.execute("CREATE INDEX IF NOT EXISTS idx_tx_user_time ON transactions(user_id, created_at)")
        db.execute("""CREATE TABLE IF NOT EXISTS user_baselines (
            user_id TEXT PRIMARY KEY, total REAL NOT NULL DEFAULT 0, txn_count INTEGER NOT NULL DEFAULT 0)""")
        db.execute("CREATE TABLE IF NOT EXISTS known_receivers (user_id TEXT, receiver_id TEXT, PRIMARY KEY(user_id, receiver_id))")

def ensure_model():
    global model_data
    if not config.MODEL_PATH.exists():
        import subprocess, sys
        subprocess.run([sys.executable, str(config.BASE_DIR / "ml" / "train.py")], cwd=config.BASE_DIR, check=True)
    model_data = joblib.load(config.MODEL_PATH)

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    ensure_model()
    yield

app = FastAPI(title="SafePay AI Backend", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=config.ALLOWED_ORIGINS, allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

def db_stats(user_id: str, receiver_id: str):
    with connect() as db:
        baseline = db.execute("SELECT total, txn_count FROM user_baselines WHERE user_id=?", (user_id,)).fetchone()
        known = db.execute("SELECT 1 FROM known_receivers WHERE user_id=? AND receiver_id=?", (user_id, receiver_id)).fetchone()
    avg = baseline["total"] / baseline["txn_count"] if baseline and baseline["txn_count"] else 1000.0
    now = time.time()
    events = [(t, u) for t, u in receiver_events.get(receiver_id, []) if now-t <= 600]
    receiver_events[receiver_id] = events
    user_events[user_id] = [t for t in user_events.get(user_id, []) if now-t <= 3600]
    return {"average": avg, "receivers": {receiver_id} if known else set(),
            "receiver_count": len({u for _, u in events}), "user_count": len(user_events[user_id])}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model_data is not None}

@app.get("/api/v1/model/info")
def model_info():
    if model_data is None: raise HTTPException(503, "Model not loaded")
    return {"best_model": model_data["model_name"], "metrics": model_data["metrics"], "training_source": model_data.get("training_source")}

@app.post("/api/v1/payments/check")
def check_payment(req: PaymentCheck):
    if model_data is None: raise HTTPException(503, "Model not loaded")
    stats = db_stats(req.user_id, req.receiver_id)
    try:
        risk, decision, reasons, step, features, probability, rules = score_payment(req, model_data, stats)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    txid = str(uuid.uuid4())
    created = datetime.now(timezone.utc).isoformat()
    with connect() as db:
        db.execute("INSERT INTO transactions (id,user_id,receiver_id,receiver_name,amount,merchant_domain,risk_score,decision,reasons,next_step,created_at,features) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                   (txid, req.user_id, req.receiver_id, req.receiver_name, req.amount, req.merchant_domain, risk, decision, json.dumps(reasons), step, created, json.dumps(features)))
        db.execute("INSERT OR IGNORE INTO known_receivers(user_id,receiver_id) VALUES (?,?)", (req.user_id, req.receiver_id))
    now = time.time()
    receiver_events.setdefault(req.receiver_id, []).append((now, req.user_id))
    user_events.setdefault(req.user_id, []).append(now)
    return {"transaction_id": txid, "risk_score": risk, "decision": decision, "reasons": reasons,
            "next_step": step, "engine": f"FastAPI + {model_data['model_name']}"}

@app.post("/api/v1/otp/send")
def send_otp(req: OTPSend):
    with connect() as db:
        tx = db.execute("SELECT user_id,decision FROM transactions WHERE id=?", (req.transaction_id,)).fetchone()
    if not tx or tx["user_id"] != req.user_id: raise HTTPException(404, "Transaction not found")
    if tx["decision"] != "VERIFY": raise HTTPException(400, "OTP is only needed for VERIFY decisions")
    code = otp_service.send(req.user_id, req.transaction_id)
    # Demo backend has no SMS provider; expose the code so the existing frontend can complete its OTP flow.
    return {"sent": True, "expires_in_seconds": config.OTP_TTL_SECONDS, "otp": code}

@app.post("/api/v1/otp/verify")
def verify_otp(req: OTPVerify):
    ok, message = otp_service.verify(req.user_id, req.transaction_id, req.otp)
    if not ok: raise HTTPException(429 if message.startswith("Maximum") else 400, message)
    return {"verified": True, "message": message}

@app.post("/api/v1/payments/confirm")
def confirm_payment(req: PaymentConfirm):
    with connect() as db:
        tx = db.execute("SELECT * FROM transactions WHERE id=?", (req.transaction_id,)).fetchone()
        if not tx: raise HTTPException(404, "Transaction not found")
        if tx["decision"] == "BLOCK": raise HTTPException(403, "Payment is blocked")
        if tx["decision"] == "VERIFY":
            otp = otp_service.items.get((tx["user_id"], req.transaction_id))
            if not otp or not otp.get("verified"): raise HTTPException(403, "Verify OTP before confirming")
        db.execute("UPDATE transactions SET status='confirmed' WHERE id=?", (req.transaction_id,))
        db.execute("INSERT INTO user_baselines(user_id,total,txn_count) VALUES (?,?,1) ON CONFLICT(user_id) DO UPDATE SET total=total+excluded.total, txn_count=txn_count+1", (tx["user_id"], tx["amount"]))
    return {"transaction_id": req.transaction_id, "status": "confirmed"}

@app.get("/api/v1/transactions")
def transactions(user_id: str = Query(...)):
    with connect() as db:
        rows = db.execute("SELECT id,user_id,receiver_id,receiver_name,amount,merchant_domain,risk_score,decision,reasons,next_step,created_at,status,feedback FROM transactions WHERE user_id=? ORDER BY created_at DESC LIMIT 100", (user_id,)).fetchall()
    return [{**dict(row), "reasons": json.loads(row["reasons"])} for row in rows]

@app.post("/api/v1/transactions/{transaction_id}/feedback")
def feedback(transaction_id: str, req: Feedback):
    with connect() as db:
        cur = db.execute("UPDATE transactions SET feedback=? WHERE id=?", (req.label, transaction_id))
        if cur.rowcount == 0: raise HTTPException(404, "Transaction not found")
    return {"transaction_id": transaction_id, "feedback": req.label}

@app.post("/api/v1/agent/investigate")
def agent_investigate(req: Investigate):
    with connect() as db:
        tx = db.execute("SELECT * FROM transactions WHERE id=?", (req.transaction_id,)).fetchone()
        baseline = db.execute("SELECT total,txn_count FROM user_baselines WHERE user_id=?", (tx["user_id"],)).fetchone() if tx else None
    if not tx: raise HTTPException(404, "Transaction not found")
    features = json.loads(tx["features"])
    context = {"receiver_count": features["receiver_txn_count_10m"], "domain_risk": features["domain_risk"],
               "user_count": features["user_txn_count_1h"], "average": baseline["total"]/baseline["txn_count"] if baseline and baseline["txn_count"] else 1000}
    return investigate(dict(tx), context)
