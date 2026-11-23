import secrets
import time
from threading import Lock
from ..config import OTP_MAX_ATTEMPTS, OTP_TTL_SECONDS

class OTPService:
    def __init__(self):
        self.items = {}
        self.lock = Lock()

    def send(self, user_id: str, transaction_id: str) -> str:
        code = f"{secrets.randbelow(1_000_000):06d}"
        with self.lock:
            self.items[(user_id, transaction_id)] = {"code": code, "expires": time.time() + OTP_TTL_SECONDS, "attempts": 0, "verified": False}
        return code

    def verify(self, user_id: str, transaction_id: str, code: str) -> tuple[bool, str]:
        with self.lock:
            item = self.items.get((user_id, transaction_id))
            if not item or time.time() > item["expires"]:
                return False, "OTP missing or expired"
            if item["attempts"] >= OTP_MAX_ATTEMPTS:
                return False, "Maximum OTP attempts exceeded"
            item["attempts"] += 1
            if secrets.compare_digest(item["code"], code):
                item["verified"] = True
                return True, "OTP verified"
            return False, "Invalid OTP"

otp_service = OTPService()
