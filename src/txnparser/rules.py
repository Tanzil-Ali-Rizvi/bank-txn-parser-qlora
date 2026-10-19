"""Hand-written regex + dictionary parser: the traditional baseline the fine-tuned model has to beat.

It knows the merchant names that appear in the training set (as a lookup table) and the common keywords of
Indian bank messages. Like most rule-based parsers, it has no way to handle a merchant or a format it has not
been told about.

Usage:  python -m txnparser.rules --data data --out results/preds
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

from .evaluate import load_jsonl
from .generate import plain_upper, slug, title

MONEY = r"(?:Rs\.?|INR|₹)\s?([\d,]+(?:\.\d+)?)"
NUM = r"[-+]?[\d,]+(?:\.\d+)?"
DATE_FORMATS = [
    "%d-%m-%y",
    "%d%b%y",
    "%d-%b-%Y",
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%d %b %Y",
    "%d.%m.%y",
    "%d/%m/%y",
    "%d-%b-%y",
]
DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2}|\d{2}[-/.]\d{2}[-/.]\d{2,4}|\d{2}[-\s]?[A-Za-z]{3}[-\s]?\d{2,4})\b")


class RuleParser:
    def __init__(self, train_rows: list[dict]):
        # merchant lookup built from the training labels only
        self.merchants: dict[str, tuple[str, str]] = {}
        for r in train_rows:
            lab = r["label"]
            name = lab["counterparty"]
            if name and r["meta"]["kind"] in {"upi_merchant", "card", "nach", "neft_tax", "upi_refund", "card_refund"}:
                cat = lab["category"] if lab["category"] != "refund" else self.merchants.get(name, (None, None))[1]
                if cat:
                    self.merchants[name] = (plain_upper(name), cat)

    # ------------------------------------------------------------------ fields
    def parse(self, text: str) -> dict:
        t = text
        up = t.upper()
        lines = t.splitlines()
        is_csv = len(lines) == 2 and lines[0].lower().startswith("txn date")
        row = lines[-1] if is_csv else t

        # direction
        debit_kw = r"\b(DEBITED|DEBIT|SPENT|SENT|WITHDRAWN|WITHDRAWAL OF|PAID|TO TRANSFER|/DR/|\bDR\b|NWD|ATW|PURCHASE|RECOVERED|DEDUCTED)\b"
        credit_kw = r"\b(CREDITED|RECEIVED|BY TRANSFER|/CR/|\bCR\b|REFUND|REVERSAL|DEPOSITED|DEP)\b"
        direction = None
        m_dr = re.search(r"WITHDRAWAL: ([\d,.]+)", up)
        if m_dr:
            direction = "debit"
        elif "WITHDRAWAL:  |" in up or re.search(r"WITHDRAWAL: \|", up):
            direction = "credit"
        elif is_csv:
            parts = re.findall(r'(?:"[^"]*"|[^,]*)', row)
            parts = [p for p in parts if p != ""]
            direction = "debit" if len(parts) >= 3 and parts[2].strip() else "credit"
        elif re.search(r",\s*-[\d,]", t) or re.search(r"\s-[\d,]+(?:\.\d+)?\s*$", t):
            direction = "debit"
        elif re.search(r",\s*\+[\d,]", t):
            direction = "credit"
        elif re.search(credit_kw, up) and not re.search(r"DEBITED|SPENT|SENT|WITHDRAWN", up):
            direction = "credit"
        elif re.search(debit_kw, up):
            direction = "debit"
        else:
            direction = "debit"

        # channel
        checks = [
            ("UPI", r"\bUPI\b|VPA"),
            ("IMPS", r"\bIMPS\b|MMT/"),
            ("NEFT", r"\bNEFT\b"),
            ("RTGS", r"\bRTGS\b"),
            ("NACH", r"\bNACH\b|\bACH\b|\bECS/|E-MANDATE|AUTO-DEBIT"),
            ("ATM", r"\bATM\b|\bATW-|\bNWD-"),
            ("CHEQUE", r"CHEQUE|\bCHQ\b|\bCLG\b"),
            ("CARD", r"\bCARD\b|\bPOS\b|\bPCD/"),
            ("BANK", r"INTEREST|\bINT\.PD|CHGS|CHARGES|\bTDS\b|\bFEE\b"),
        ]
        channel = next((c for c, p in checks if re.search(p, up)), "BANK")

        # amount and balance
        amount = balance = None
        monies = [float(x.replace(",", "")) for x in re.findall(MONEY, t)]
        bal_m = re.search(r"(?:AVL BAL|AVL\. BAL|AVAILABLE BALANCE)[:\s]*" + MONEY.replace("(?:Rs", "(?:RS"), up)
        if bal_m:
            balance = float(bal_m.group(1).replace(",", ""))
        if monies:
            amount = monies[0]
        else:  # statement line
            nums = [n for n in re.findall(NUM, row.split("|")[-1] if "|" in row else row)]
            if "|" in row:
                cols = [c.strip() for c in row.split("|")]
                amt_s = (cols[2].split(":")[-1].strip() or cols[3].split(":")[-1].strip()) if len(cols) >= 5 else ""
                amount = _to_float(amt_s)
                balance = _to_float(cols[-1].split(":")[-1])
            elif t.startswith("Date:"):
                amount = _to_float(re.search(r"Amount: (" + NUM + ")", t).group(1)) if "Amount:" in t else None
                bm = re.search(r"Closing balance: (" + NUM + ")", t)
                balance = _to_float(bm.group(1)) if bm else None
            elif is_csv:
                parts = re.findall(r'"[^"]*"|[^,]+|(?<=,)(?=,)', row)
                vals = [p for p in row.split('"')[-1].split(",")]
                nums_ = [_to_float(v) for v in vals if v.strip()]
                if nums_:
                    amount, balance = (nums_[0], nums_[-1]) if len(nums_) > 1 else (nums_[0], None)
            else:
                tail = re.findall(r"(?:^|[\s,])(" + NUM + r")(?=\s*(?:Dr|DR|Cr|CR)?\s*(?:,|$|\s))", t)
                vals = [_to_float(v) for v in tail]
                vals = [v for v in vals if v is not None]
                if re.search(r"\s(?:Dr|DR|Cr|CR)\s", t) or t.count(",") >= 3:
                    if len(vals) >= 2:
                        amount, balance = abs(vals[-2]), vals[-1]
                elif vals:
                    amount = abs(vals[-1])
            del nums

        # reference
        ref = None
        for p in [
            r"UPI REF(?: NO)?[:\s]*(\d{12})",
            r"UPI[:/](\d{12})",
            r"IMPS REF NO (\d{12})",
            r"UTR[:\s]*([A-Z0-9]{12,22})",
            r"RRN (\d{12})",
            r"TXN NO (\d{12})",
            r"REF NO (\w{6,22})",
            r"REF (\d{12})",
            r"CHEQUE(?: NO\.)? (\d{6})",
            r"CTS-(\d{6})",
            r"CLG/(\d{6})",
            r"DEP/(\d{6})",
            r"DEP (\d{6})",
            r"\b([A-Z]{4}[NR]\d{11,19}|N\d{15})\b",
            r"/(\d{12})(?:/|$|\b)",
            r"-(\d{12})-",
        ]:
            m = re.search(p, up)
            if m:
                ref = m.group(1)
                break

        # own account
        acct = None
        m = (
            re.search(r"(?:A/C|ACCT|ACCOUNT|CARD)(?: NO\.?)?\s*(?:ENDING\s*)?[X*]*(\d{4})\b", up)
            or re.search(r"(?:CARD ENDING|ENDING) (\d{4})", up)
            or re.search(r"\b\d{4}X{6}(\d{4})|X{12}(\d{4})", up)
            or re.search(r"PCD/(\d{4})/", up)
        )
        if m:
            acct = next(g for g in m.groups() if g)

        # date
        dt = None
        for cand in DATE_RE.findall(t):
            dt = _parse_date(cand)
            if dt:
                break

        # counterparty + category
        counterparty, category = None, None
        for name, (pu, cat) in self.merchants.items():
            if pu and (pu in up or slug(name) in t.lower().replace(".", "")):
                if counterparty is None or len(name) > len(counterparty):
                    counterparty, category = name, cat
        if counterparty is None:
            for p in [
                r"\bTo ([A-Z][A-Za-z ]+?) On\b",
                r"\bfrom ([A-Z][A-Za-z ]+?)[.(]",
                r"\bby ([A-Z][A-Z ]+?),",
                r"\bto ([A-Z][A-Z ]+?) A/c",
                r"Payee: ([A-Z][A-Za-z ]+?) -",
                r"UPI/\d{12}/([A-Z][A-Za-z ]+?)/",
                r"UPI-([A-Z][A-Z ]+?)-",
                r"/(?:DR|CR)/\d{12}/([A-Z][A-Z ]+?)/",
                r"P2[AM]/\d{12}/([A-Z][A-Z ]+?)/",
                r"MMT/IMPS/\d{12}/[^/]*/([A-Z][A-Z ]+?)/",
                r"NEFT CR-\w+-([A-Z][A-Z ]+?)-",
                r"NEFT DR-\w+-([A-Z][A-Z ]+?)-",
                r"NEFT/\w+/([A-Z][A-Z ]+?)/",
                r"RTGS \w+-\w+-([A-Z][A-Z ]+?)-",
                r"from ([A-Z][A-Z ]+?)\. UTR",
                r"to ([A-Z][A-Z ]+?) processed",
                r"CTS-\d{6}-([A-Z][A-Z ]+)$",
                r"CLG/\d{6}/([A-Z][A-Z ]+)$",
            ]:
                m = re.search(p, t)
                if m:
                    counterparty = title(m.group(1).strip())
                    break
        if re.search(r"SALARY|\bSAL\b", up):
            category = "salary"
        elif re.search(r"\bRENT\b", up) and category is None:
            category = "rent"
        elif re.search(r"REFUND|REVERSAL", up):
            category = "refund"
        elif channel == "ATM" or "WITHDRAWN" in up:
            category, counterparty = "cash_withdrawal", None
        elif re.search(r"INTEREST|INT\.PD", up) and channel == "BANK":
            category, counterparty = "interest", None
        elif re.search(r"\bTDS\b|TAX", up):
            category = "tax"
        elif re.search(r"CHGS|CHARGES|\bFEE\b", up) and channel == "BANK":
            category, counterparty = "bank_charges", None
        if category is None:
            category = "transfer"

        return {
            "direction": direction,
            "amount": amount,
            "channel": channel,
            "counterparty": counterparty,
            "category": category,
            "reference": ref,
            "account_last4": acct,
            "balance": balance,
            "date": dt,
        }


def _to_float(s) -> float | None:
    if s is None:
        return None
    s = str(s).replace(",", "").strip().lstrip("+")
    try:
        return float(s)
    except ValueError:
        return None


def _parse_date(s: str) -> str | None:
    s2 = s.strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(s2.title() if any(c.isalpha() for c in s2) else s2, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, default=Path("results/preds"))
    p.add_argument("--splits", nargs="+", default=["test_id", "test_ood"])
    a = p.parse_args(argv)
    parser = RuleParser(load_jsonl(a.data / "train.jsonl"))
    a.out.mkdir(parents=True, exist_ok=True)
    for split in a.splits:
        with open(a.out / f"rules_{split}.jsonl", "w", encoding="utf-8") as f:
            for row in load_jsonl(a.data / f"{split}.jsonl"):
                f.write(
                    json.dumps({"id": row["meta"]["id"], "pred": parser.parse(row["text"])}, ensure_ascii=False) + "\n"
                )
        print(f"wrote {a.out / f'rules_{split}.jsonl'}")


if __name__ == "__main__":
    main()
