from __future__ import annotations
import re
from datetime import datetime, timezone
from ..config import BLOCK_MIN, ML_WEIGHT, RULES_WEIGHT, VERIFY_MIN
from ..schemas import PaymentCheck
from ml.features import make_features
from ml.explain import explain

RECEIVER_RE = re.compile(r"^[a-zA-Z0-9.\-_]{2,}@[a-zA-Z0-9.\-]{2,}$")

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
    rule_points = 0.0
    if req.signals.vpn: rule_points += 20
    if req.signals.new_device: rule_points += 22
    if req.signals.emulator: rule_points += 30
    if req.signals.odd_location: rule_points += 20
    if row["is_new_receiver"]: rule_points += 12
    if row["amount_vs_user_avg"] >= 5: rule_points += 20
    if row["domain_risk"] >= .5: rule_points += 20
    if stats["receiver_count"] >= 3: rule_points += 30
    rules = min(rule_points, 100)
    risk = min(100.0, max(0.0, 100 * (ML_WEIGHT * probability + RULES_WEIGHT * rules / 100)))
    odd_hour = hour < 5 or hour >= 23
    active_signals = sum((bool(row["is_new_receiver"]), req.signals.vpn, req.signals.new_device,
                          req.signals.emulator, req.signals.odd_location, odd_hour,
                          row["domain_risk"] >= .5, stats["receiver_count"] >= 3))
    verify_scenario = ((not stats["receivers"] and 10_000 <= req.amount <= 50_000)
                       or (1_000 <= req.amount <= 50_000 and active_signals == 1))
    hard_block = req.signals.emulator and req.signals.new_device and req.amount > 50000
    if hard_block:
        risk = 100.0
    elif verify_scenario:
        # Keep the specified first-time/high-amount and single-signal cases in
        # the verification band even when model calibration is conservative.
        risk = min(69.0, max(40.0, risk))
    decision = decide(risk)
    reasons = explain(model_data["model"], row, model_data["features"], decision=decision)
    step = {"ALLOW": "Payment is safe to continue", "VERIFY": "Verify with the one-time password", "BLOCK": "Payment blocked for your protection"}[decision]
    return round(risk, 2), decision, reasons, step, row, probability, rules
