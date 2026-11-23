from typing import Literal
from pydantic import BaseModel, Field

class Signals(BaseModel):
    vpn: bool = False
    new_device: bool = False
    emulator: bool = False
    odd_location: bool = False

class PaymentCheck(BaseModel):
    user_id: str = Field(min_length=1, max_length=120)
    receiver_id: str = Field(min_length=1, max_length=255)
    receiver_name: str = Field(min_length=1, max_length=200)
    amount: float = Field(gt=0, le=10_000_000)
    merchant_domain: str | None = None
    hour: int | None = Field(default=None, ge=0, le=23)
    signals: Signals = Field(default_factory=Signals)

class OTPSend(BaseModel):
    user_id: str
    transaction_id: str

class OTPVerify(OTPSend):
    otp: str = Field(pattern=r"^\d{6}$")

class PaymentConfirm(BaseModel):
    transaction_id: str

class Feedback(BaseModel):
    label: Literal["fraud", "me"]

class Investigate(BaseModel):
    transaction_id: str
