from __future__ import annotations
import sys
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score,
                             confusion_matrix, classification_report)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from features import synthetic_rows

ROOT = Path(__file__).resolve().parent
FEATURE_NAMES = ["amount", "amount_vs_user_avg", "hour", "is_new_receiver", "is_new_device", "vpn_detected", "emulator_detected", "location_risk", "receiver_txn_count_10m", "user_txn_count_1h", "domain_risk"]

def load_data():
    paths = list((ROOT / "data").glob("*.csv")) if (ROOT / "data").exists() else []
    if paths:
        import pandas as pd
        df = pd.read_csv(paths[0])
        label = next((c for c in ("is_fraud", "fraud", "isFraud") if c in df.columns), None)
        if label:
            mapping = {"amount": "amount", "amount_vs_user_avg": "amount_vs_user_avg", "hour": "hour", "is_new_receiver": "is_new_receiver", "is_new_device": "is_new_device", "vpn_detected": "vpn_detected", "emulator_detected": "emulator_detected", "location_risk": "location_risk", "receiver_txn_count_10m": "receiver_txn_count_10m", "user_txn_count_1h": "user_txn_count_1h", "domain_risk": "domain_risk"}
            if all(c in df for c in FEATURE_NAMES):
                return df[FEATURE_NAMES].fillna(0).to_numpy(), df[label].astype(int).to_numpy(), paths[0].name
            # PaySim schema fallback: map amount/hour, synthesize unavailable context signals.
            if "amount" in df and label == "isFraud":
                n = len(df)
                X = np.zeros((n, len(FEATURE_NAMES)))
                X[:, 0] = df.amount.fillna(0)
                X[:, 1] = 1
                X[:, 2] = df.get("step", 12).to_numpy() % 24 if hasattr(df.get("step", 12), "to_numpy") else 12
                X[:, 3:] = 0.05
                return X, df[label].astype(int).to_numpy(), paths[0].name
    X, y = synthetic_rows()
    return X, y, "synthetic (50,000 transactions)"

def main():
    X, y, source = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    pos_weight = max(1.0, float((y_train == 0).sum() / max(1, (y_train == 1).sum())))
    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)),
        "Random Forest": RandomForestClassifier(n_estimators=120, max_depth=14, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=42),
        "XGBoost": None,
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(n_estimators=140, max_depth=5, learning_rate=.08, subsample=.85,
             colsample_bytree=.9, scale_pos_weight=pos_weight, eval_metric="logloss", n_jobs=-1, random_state=42)
    except ImportError:
        # Keep training usable in constrained environments; XGBoost is pinned in requirements.
        models.pop("XGBoost")
        models["Gradient Boosting (fallback)"] = HistGradientBoostingClassifier(max_iter=140, learning_rate=.08, max_leaf_nodes=25, class_weight="balanced", random_state=42)
    best, best_name, best_auc = None, None, -1
    scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, prob)
        scores[name] = auc
        if auc > best_auc:
            best, best_name, best_auc = model, name, auc
    pred = best.predict(X_test)
    cm = confusion_matrix(y_test, pred).tolist()
    metrics = {"precision": float(precision_score(y_test, pred, zero_division=0)), "recall": float(recall_score(y_test, pred, zero_division=0)), "f1": float(f1_score(y_test, pred, zero_division=0)), "roc_auc": float(best_auc), "confusion_matrix": cm}
    payload = {"model": best, "model_name": best_name, "features": FEATURE_NAMES, "metrics": metrics, "training_source": source, "candidate_auc": scores}
    joblib.dump(payload, ROOT / "model.joblib")
    (ROOT / "model_report.md").write_text(f"# SafePay AI model report\n\nTraining data: {source}. Rows: {len(y):,}; fraud: {int(y.sum()):,} ({y.mean():.2%}).\n\nBest model: **{best_name}**\n\n| Metric | Value |\n|---|---:|\n| Precision | {metrics['precision']:.4f} |\n| Recall | {metrics['recall']:.4f} |\n| F1 | {metrics['f1']:.4f} |\n| ROC-AUC | {metrics['roc_auc']:.4f} |\n\nConfusion matrix (rows actual [genuine, fraud], columns predicted [genuine, fraud]): `{cm}`\n\nCandidate ROC-AUC: `{scores}`\n", encoding="utf-8")
    print(f"Best model: {best_name}; ROC-AUC={best_auc:.4f}")

if __name__ == "__main__":
    main()
