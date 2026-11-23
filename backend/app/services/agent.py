def investigate(transaction: dict, context: dict) -> dict:
    steps = [
        {"tool": "receiver_history", "result": f"{context['receiver_count']} receiver payments in recent window"},
        {"tool": "domain_risk", "result": f"Risk {context['domain_risk']:.2f}"},
        {"tool": "velocity", "result": f"{context['user_count']} payments by user in recent window"},
        {"tool": "user_baseline", "result": f"Typical amount {context['average']:.2f}"},
    ]
    score = transaction["risk_score"]
    verdict = "high risk" if score >= 70 else "needs verification" if score >= 40 else "likely legitimate"
    actions = ["Block payment and contact the user"] if score >= 70 else ["Require OTP verification"] if score >= 40 else ["Allow payment"]
    return {"steps": steps, "verdict": verdict, "confidence": round(max(score, 100-score)/100, 2), "actions": actions,
            "report": f"Transaction {transaction['id']} scored {score:.1f}/100 ({verdict})."}
