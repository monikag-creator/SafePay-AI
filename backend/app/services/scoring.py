from __future__ import annotations
import re
from datetime import datetime, timezone
from ..config import BLOCK_MIN, ML_WEIGHT, RULES_WEIGHT, VERIFY_MIN
from ..schemas import PaymentCheck
from ml.features import make_features
from ml.explain import explain

RECEIVER_RE = re.compile(r"^[a-zA-Z0-9.\-_]{2,}@[a-zA-Z]{2,}$")

def decide(score: float) -> str:
    if score >= BLOCK_MIN:
        return "BLOCK"
    if score >= VERIFY_MIN:
        return "VERIFY"
    return "ALLOW"

def score_payment(req: PaymentCheck, model_data: dict, stats: dict):
    if not RECEIVER_RE.fullmatch(req.receiver_id):
        raise ValueError("receiver_id must be a valid UPI-style ID, such as name@bank")
    hour = req.hour if req.hour is not None else datetime.now(timezone.utc).hour
    row = make_features(req.amount, stats["average"], hour, req.receiver_id not in stats["receivers"],
        req.signals.new_device, req.signals.vpn, req.signals.emulator, req.signals.odd_location,
        stats["receiver_count"], stats["user_count"], req.merchant_domain)
    vector = [[row[k] for k in model_data["features"]]]
    probability = float(model_data["model"].predict_proba(vector)[0][1])
    rules = 0.0
    if req.signals.vpn: rules += .12
    if req.signals.new_device: rules += .17
    if req.signals.emulator: rules += .25
    if req.signals.odd_location: rules += .15
    if row["amount_vs_user_avg"] >= 5: rules += .18
    if row["domain_risk"] >= .5: rules += .20
    if stats["receiver_count"] >= 3: rules += .25
    risk = min(100.0, max(0.0, 100 * (ML_WEIGHT * probability + RULES_WEIGHT * min(rules, 1))))
    hard_block = req.signals.emulator and req.signals.new_device and req.amount > 50000
    if hard_block: risk = 100.0
    decision = decide(risk)
    reasons = explain(model_data["model"], row, model_data["features"])
    if req.signals.vpn: reasons.insert(0, "A VPN was detected")
    if req.signals.emulator and "An emulator was detected" not in reasons: reasons.insert(0, "An emulator was detected")
    if row["domain_risk"] >= .5: reasons.insert(0, "The merchant domain looks suspicious")
    if stats["receiver_count"] >= 3: reasons.insert(0, "This receiver has received payments from 3 or more senders in 10 minutes")
    reasons = list(dict.fromkeys(reasons))[:5]
    step = {"ALLOW": "Payment is safe to continue", "VERIFY": "Verify with the one-time password", "BLOCK": "Payment blocked for your protection"}[decision]
    return round(risk, 2), decision, reasons, step, row, probability, rules
