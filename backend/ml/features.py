from __future__ import annotations
import re

SUSPICIOUS_TLDS = {"xyz", "top", "click"}
KEYWORDS = ("crypto", "deal", "free", "lucky", "bonus")

def domain_risk(domain: str | None) -> float:
    if not domain:
        return 0.0
    d = domain.lower().strip().split("://")[-1].split("/")[0].split(":")[0].rstrip(".")
    if not d:
        return 0.0
    labels = d.split(".")
    risk = 0.0
    if labels[-1] in SUSPICIOUS_TLDS:
        risk += 0.55
    if any(k in d for k in KEYWORDS):
        risk += 0.35
    if d.count("-") >= 2:
        risk += 0.25
    return min(1.0, risk)

def make_features(amount: float, average: float, hour: int, is_new_receiver: bool,
                  is_new_device: bool, vpn: bool, emulator: bool, odd_location: bool,
                  receiver_count: int, user_count: int, domain: str | None) -> dict[str, float]:
    return {"amount": float(amount), "amount_vs_user_avg": float(amount / max(average, 1.0)),
            "hour": float(hour), "is_new_receiver": float(is_new_receiver), "is_new_device": float(is_new_device),
            "vpn_detected": float(vpn), "emulator_detected": float(emulator),
            "location_risk": 0.85 if odd_location else 0.05, "receiver_txn_count_10m": float(receiver_count),
            "user_txn_count_1h": float(user_count), "domain_risk": domain_risk(domain)}

def synthetic_rows(n: int = 50_000, seed: int = 42):
    """Build realistic, reproducible UPI-like samples with exactly ~3% fraud."""
    import numpy as np
    rng = np.random.default_rng(seed)
    fraud = np.zeros(n, dtype=int)
    fraud[rng.choice(n, size=round(n * .03), replace=False)] = 1
    amount = np.clip(rng.lognormal(6.0 + fraud * 1.0, 1.0 + fraud * .35, n), 10, 200_000)
    usual = np.clip(rng.lognormal(6, .8, n), 50, 50_000)
    ratio = amount / usual
    hour = rng.integers(0, 24, n)
    new_receiver = rng.binomial(1, .15 + fraud * .5, n)
    new_device = rng.binomial(1, .04 + fraud * .45, n)
    vpn = rng.binomial(1, .02 + fraud * .35, n)
    emulator = rng.binomial(1, .01 + fraud * .30, n)
    location = np.clip(rng.beta(1.5, 8, n) + fraud * rng.beta(3, 2, n), 0, 1)
    receiver_count = rng.poisson(1.2 + fraud * 3, n)
    user_count = rng.poisson(1.5 + fraud * 2, n)
    domain = np.clip(rng.beta(1, 15, n) + fraud * rng.beta(2.5, 2, n), 0, 1)
    X = np.column_stack([amount, ratio, hour, new_receiver, new_device, vpn, emulator, location, receiver_count, user_count, domain])
    return X, fraud
