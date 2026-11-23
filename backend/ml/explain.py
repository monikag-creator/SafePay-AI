from __future__ import annotations
import numpy as np

LABELS = {
    "amount": "The payment amount is unusually large",
    "amount_vs_user_avg": "Amount is {value:.1f}x higher than your usual",
    "hour": "The payment is being made at an unusual hour",
    "is_new_receiver": "This is a new receiver",
    "is_new_device": "A new device was detected",
    "vpn_detected": "A VPN was detected",
    "emulator_detected": "An emulator was detected",
    "location_risk": "The location looks unusual",
    "receiver_txn_count_10m": "This receiver has received many recent payments",
    "user_txn_count_1h": "There have been many recent payments from your account",
    "domain_risk": "The merchant domain looks suspicious",
}

def explain(model, row: dict, feature_names: list[str], limit: int = 5) -> list[str]:
    X = np.array([[row[f] for f in feature_names]], dtype=float)
    values = None
    try:
        import shap
        estimator = model.steps[-1][1] if hasattr(model, "steps") else model
        data = X
        if hasattr(model, "steps"):
            for _, step in model.steps[:-1]:
                data = step.transform(data)
        try:
            sv = shap.TreeExplainer(estimator)(data)
            arr = np.asarray(sv.values)
            values = arr[0, :, 1] if arr.ndim == 3 else arr.reshape(-1)
        except Exception:
            sv = shap.LinearExplainer(estimator, data)(data)
            values = np.asarray(sv.values).reshape(-1)
    except Exception:
        pass
    # A stable, local fallback keeps explanations available if optional SHAP runtime
    # support is incompatible with a particular sklearn/XGBoost release.
    if values is None or len(values) != len(feature_names):
        values = np.array([max(0, row[k] - base) for k, base in zip(feature_names,
            [5000, 1, 8, 0, 0, 0, 0, .1, 2, 3, .1])], dtype=float)
    order = np.argsort(np.abs(values))[::-1]
    reasons = []
    for idx in order:
        key = feature_names[int(idx)]
        val = row[key]
        if abs(values[idx]) < 1e-8 or (key not in ("amount", "amount_vs_user_avg") and val <= 0.05):
            continue
        template = LABELS[key]
        reason = template.format(value=val) if "{value" in template else template
        if reason not in reasons:
            reasons.append(reason)
        if len(reasons) == limit:
            break
    return reasons or ["The payment matches your usual pattern"]
