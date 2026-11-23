# SafePay AI backend

FastAPI backend for the existing Bolt frontend. It trains a fraud scoring model, blends its probability with transparent risk rules, and records payments and confirmed user baselines in SQLite.

## Local run

Requires Python 3.11. Create and activate a virtual environment from the `backend` directory, then run:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python ml/train.py
uvicorn app.main:app --reload
```

The model is also trained automatically at startup if `ml/model.joblib` is missing. Open `http://127.0.0.1:8000/docs` for interactive API docs. To allow another frontend origin, set `ALLOWED_ORIGINS` to a comma-separated list. The default allows the Bolt app and `http://localhost:5173`.

SQLite lives at `safepay.db` by default; set `DATABASE_PATH` to choose another file. OTP codes are returned in the `/otp/send` response because this MVP has no SMS provider. In-memory OTPs and velocity counters reset when the process restarts.

## Decision examples

Replace `BASE` with `http://127.0.0.1:8000`. Each example submits one payment check.

ALLOW:

```bash
curl -X POST "$BASE/api/v1/payments/check" -H "Content-Type: application/json" -d '{"user_id":"alice","receiver_id":"shop@bank","receiver_name":"Local Shop","amount":350,"hour":13,"signals":{}}'
```

VERIFY:

```bash
curl -X POST "$BASE/api/v1/payments/check" -H "Content-Type: application/json" -d '{"user_id":"bob","receiver_id":"newperson@upi","receiver_name":"New Person","amount":25000,"hour":23,"signals":{"vpn":true,"new_device":true,"odd_location":true}}'
```

BLOCK (hard rule: emulator + new device + amount above 50,000):

```bash
curl -X POST "$BASE/api/v1/payments/check" -H "Content-Type: application/json" -d '{"user_id":"carol","receiver_id":"unknown@bank","receiver_name":"Unknown","amount":75000,"hour":2,"signals":{"new_device":true,"emulator":true}}'
```

## Render deployment

1. Push this repository to GitHub with the `backend/` directory at its root-level location shown here.
2. In Render, choose **New + → Blueprint** and select the repository; Render reads `backend/render.yaml` (or create a Web Service manually with root directory `backend`).
3. The build installs pinned requirements and trains the model. The start command launches Uvicorn. Free instances may sleep when idle; the first request after sleep can take longer.
4. Set `ALLOWED_ORIGINS` in the service environment to the Bolt URL and any additional comma-separated frontend origins.
5. Copy the deployed `https://...onrender.com` URL into the Bolt frontend's backend/API base URL setting.

SQLite and in-memory velocity/OTP state are suitable for a simple single-instance demo. Render's free filesystem is ephemeral, so transaction history and the trained artifact can reset on redeploy; the build regenerates the artifact. Use persistent storage for durable history.

## Model

`ml/train.py` uses the first CSV in `ml/data/` if it contains the expected feature columns and a fraud label; PaySim's standard `isFraud`/`amount` fields are mapped with unavailable contextual features filled using neutral values. Otherwise it generates 50,000 reproducible UPI-like transactions with 3% fraud. Logistic Regression, Random Forest, and XGBoost are compared on a stratified holdout; the best ROC-AUC model is saved to `ml/model.joblib`, with metrics in `ml/model_report.md`. SHAP explanations are attempted per prediction; a local feature-based fallback is used if SHAP cannot initialize for the installed model runtime.

Run the quick API checks from this directory with `python -m pytest -q`.
