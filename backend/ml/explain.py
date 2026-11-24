from __future__ import annotations
def explain(model, row: dict, feature_names: list[str], limit: int = 5,
            decision: str | None = None) -> list[str]:
    """Explain a decision using the transaction signals in plain language."""
    reasons = []
    ratio = row.get("amount_vs_user_avg", 1.0)
    if ratio >= 2:
        reasons.append(f"Amount is {ratio:.1f}x higher than your usual")
    elif row.get("amount", 0) >= 10_000:
        reasons.append("The payment amount is unusually large")
    if row.get("is_new_receiver", 0):
        reasons.append("This is a new receiver")
    if row.get("vpn_detected", 0):
        reasons.append("A VPN was detected")
    if row.get("is_new_device", 0):
        reasons.append("A new device was detected")
    if row.get("emulator_detected", 0):
        reasons.append("An emulator was detected")
    if row.get("location_risk", 0) >= .5:
        reasons.append("The payment location looks unusual")
    hour = int(row.get("hour", 12))
    if hour < 5 or hour >= 23:
        reasons.append("The payment is at an unusual hour")
    if row.get("domain_risk", 0) >= .5:
        reasons.append("The merchant domain looks suspicious")
    velocity = int(row.get("receiver_txn_count_10m", 0))
    if velocity >= 3:
        reasons.append(f"This receiver has payments from {velocity} different senders in 10 minutes")
    if not reasons:
        if decision in ("VERIFY", "BLOCK"):
            reasons.append("The model detected an elevated fraud risk")
        else:
            reasons.append("The payment matches your usual pattern")
    return reasons[:max(3, min(limit, 5))] if len(reasons) >= 3 else reasons
