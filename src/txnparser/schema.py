"""Output schema, prompt and output parsing shared by data generation, training, inference and evaluation."""

from __future__ import annotations

import json
import re
from typing import Any

CHANNELS = ["UPI", "IMPS", "NEFT", "RTGS", "CARD", "ATM", "NACH", "CHEQUE", "BANK"]

CATEGORIES = [
    "food_dining",
    "groceries",
    "shopping",
    "travel",
    "fuel",
    "utilities",
    "telecom",
    "entertainment",
    "healthcare",
    "education",
    "rent",
    "salary",
    "emi_loan",
    "investment",
    "insurance",
    "tax",
    "transfer",
    "cash_withdrawal",
    "bank_charges",
    "interest",
    "refund",
]

FIELDS = [
    "direction",
    "amount",
    "channel",
    "counterparty",
    "category",
    "reference",
    "account_last4",
    "balance",
    "date",
]

SYSTEM_PROMPT = (
    "You convert Indian bank SMS alerts and bank statement lines into JSON. "
    "Return only one JSON object with exactly these keys:\n"
    '  "direction": "debit" or "credit"\n'
    '  "amount": number in rupees, or null if not stated\n'
    f'  "channel": one of {", ".join(CHANNELS)} (BANK = interest, charges or other bank entries)\n'
    '  "counterparty": the merchant, person, employer or institution on the other side, in Title Case, '
    "or null if there is none (interest, charges, ATM)\n"
    f'  "category": one of {", ".join(CATEGORIES)}\n'
    '  "reference": the UPI ref / UTR / RRN / cheque number exactly as written, or null\n'
    '  "account_last4": last 4 digits of the customer\'s own account or card, or null\n'
    '  "balance": available balance after the transaction, or null\n'
    '  "date": transaction date as YYYY-MM-DD, or null'
)

_JSON_OBJ = re.compile(r"\{.*\}", re.S)


def user_prompt(text: str) -> str:
    return f"Transaction:\n{text}"


def messages(text: str, label: dict | None = None, system: str = SYSTEM_PROMPT) -> list[dict]:
    """Chat messages for one example; include the label as the assistant turn when training."""
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user_prompt(text)}]
    if label is not None:
        msgs.append({"role": "assistant", "content": to_json(label)})
    return msgs


def to_json(label: dict) -> str:
    return json.dumps({k: label.get(k) for k in FIELDS}, ensure_ascii=False)


def parse_output(raw: str) -> dict[str, Any] | None:
    """Parse a model completion into a dict. Returns None if no valid JSON object is found."""
    if raw is None:
        return None
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.S)
    for candidate in (s, *(m.group(0) for m in [_JSON_OBJ.search(s)] if m)):
        try:
            obj = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(obj, dict):
            return obj
    return None
