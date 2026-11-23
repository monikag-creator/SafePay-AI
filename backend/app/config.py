import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = Path(os.getenv("DATABASE_PATH", str(BASE_DIR / "safepay.db")))
MODEL_PATH = BASE_DIR / "ml" / "model.joblib"
REPORT_PATH = BASE_DIR / "ml" / "model_report.md"
FEATURES = ["amount", "amount_vs_user_avg", "hour", "is_new_receiver", "is_new_device", "vpn_detected", "emulator_detected", "location_risk", "receiver_txn_count_10m", "user_txn_count_1h", "domain_risk"]
ALLOW_MAX = 39
VERIFY_MIN = 40
BLOCK_MIN = 70
OTP_TTL_SECONDS = 300
OTP_MAX_ATTEMPTS = 3
ALLOWED_ORIGINS = [x.strip() for x in os.getenv("ALLOWED_ORIGINS", "https://safepay-ai-fintech-a-5eid.bolt.host,http://localhost:5173").split(",") if x.strip()]
RULES_WEIGHT = 0.15
ML_WEIGHT = 0.85
