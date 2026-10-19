"""Seeded generator of synthetic Indian bank transactions (SMS alerts and statement lines) with JSON labels.

Every example is built from a semantic transaction, then rendered through one text template. The label only
contains what the rendered text actually shows: if a template omits the balance or the date, the label says null.

Two splits test different things:
  * in-distribution (train / val / test_id): seen merchants, seen templates, fresh values
  * out-of-distribution (test_ood): every example contains a merchant or a text format never seen in training

Usage:  python -m txnparser.generate --out data
"""

from __future__ import annotations

import argparse
import json
import random
import re
import string
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

from .catalog import (
    BANKS,
    CITIES,
    EMPLOYER_FORM,
    EMPLOYER_PREFIX,
    EMPLOYER_SUFFIX,
    FIRST_NAMES,
    LAST_NAMES,
    MANDATE_CATEGORIES,
    MERCHANT_HANDLES,
    MERCHANTS,
    UPI_HANDLES,
)
from .schema import FIELDS

# --------------------------------------------------------------------------- transaction kinds
# kind: (channel, direction, party, category or None = from the merchant, note type)
KINDS = {
    "upi_merchant": ("UPI", "debit", "merchant", None, "merchant"),
    "upi_refund": ("UPI", "credit", "merchant", "refund", "refund"),
    "upi_p2p_dr": ("UPI", "debit", "person", "transfer", "p2p"),
    "upi_p2p_cr": ("UPI", "credit", "person", "transfer", "p2p"),
    "upi_rent": ("UPI", "debit", "person", "rent", "rent"),
    "imps_dr": ("IMPS", "debit", "person", "transfer", "p2p"),
    "imps_cr": ("IMPS", "credit", "person", "transfer", "p2p"),
    "neft_salary": ("NEFT", "credit", "employer", "salary", "salary"),
    "neft_cr": ("NEFT", "credit", "person", "transfer", "transfer"),
    "neft_dr": ("NEFT", "debit", "person", "transfer", "transfer"),
    "neft_rent": ("NEFT", "debit", "person", "rent", "rent"),
    "neft_tax": ("NEFT", "debit", "tax", "tax", "tax"),
    "rtgs_cr": ("RTGS", "credit", "employer", "transfer", None),
    "rtgs_dr": ("RTGS", "debit", "employer", "transfer", None),
    "card": ("CARD", "debit", "merchant", None, None),
    "card_refund": ("CARD", "credit", "merchant", "refund", None),
    "atm": ("ATM", "debit", None, "cash_withdrawal", None),
    "nach": ("NACH", "debit", "mandate", None, None),
    "interest": ("BANK", "credit", None, "interest", None),
    "charges": ("BANK", "debit", None, "bank_charges", None),
    "tds": ("BANK", "debit", None, "tax", None),
    "chq_paid": ("CHEQUE", "debit", "person", "transfer", None),
    "chq_dep": ("CHEQUE", "credit", None, "transfer", None),
}
KIND_WEIGHTS = {
    "upi_merchant": 22,
    "upi_refund": 3,
    "upi_p2p_dr": 8,
    "upi_p2p_cr": 5,
    "upi_rent": 3,
    "imps_dr": 3,
    "imps_cr": 3,
    "neft_salary": 5,
    "neft_cr": 2,
    "neft_dr": 2,
    "neft_rent": 1,
    "neft_tax": 2,
    "rtgs_cr": 1,
    "rtgs_dr": 1,
    "card": 13,
    "card_refund": 2,
    "atm": 6,
    "nach": 8,
    "interest": 3,
    "charges": 4,
    "tds": 1,
    "chq_paid": 2,
    "chq_dep": 2,
}
# Kinds whose category is only knowable from the remark (rent, salary, refund): templates must show {note}.
NOTE_REQUIRED = {"upi_rent", "neft_rent", "neft_salary", "upi_refund", "neft_tax"}

UPI_MERCHANT_CATS = [
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
    "investment",
    "insurance",
    "emi_loan",
    "rent",
]
CARD_CATS = [
    "food_dining",
    "groceries",
    "shopping",
    "travel",
    "fuel",
    "entertainment",
    "healthcare",
    "education",
    "insurance",
    "telecom",
    "utilities",
]

AMOUNT_RANGE = {
    "food_dining": (80, 2500),
    "groceries": (60, 6000),
    "shopping": (199, 60000),
    "travel": (25, 25000),
    "fuel": (100, 5000),
    "utilities": (150, 9000),
    "telecom": (149, 3000),
    "entertainment": (99, 1600),
    "healthcare": (90, 25000),
    "education": (499, 150000),
    "rent": (6000, 65000),
    "salary": (25000, 350000),
    "emi_loan": (1200, 65000),
    "investment": (500, 100000),
    "insurance": (500, 60000),
    "tax": (1000, 250000),
    "transfer": (10, 100000),
    "cash_withdrawal": (500, 20000),
    "bank_charges": (5.9, 1180),
    "interest": (8, 9000),
    "refund": (99, 20000),
}

NOTES = {
    "merchant": ["PAYMENT", "UPI", "Pay to merchant", "ORDER", "Paid via app", "NA", "payment", "UPI Intent"],
    "refund": ["REFUND", "Refund for order", "REVERSAL", "refund", "ORDER REFUND"],
    "p2p": ["NA", "SENT USING APP", "dinner", "share", "paid", "UPI", "Payment from Ph", "gift", "trip", "thanks"],
    "rent": ["RENT", "Rent {mon}", "house rent", "RENT FOR {MON}", "flat rent {mon}"],
    "salary": ["SALARY {MON} {yyyy}", "SAL {MON}{yy}", "SALARY FOR {MON}", "Salary credit {mon}"],
    "transfer": ["TRANSFER", "FAMILY SUPPORT", "NA", "PERSONAL", "GIFT", "SETTLEMENT"],
    "tax": ["ADVANCE TAX", "SELF ASSESSMENT TAX", "GST PMT", "PROPERTY TAX", "CHALLAN 280"],
}
SMS_CHARGES = [
    "SMS alert charges",
    "ATM AMC charges",
    "minimum balance charges",
    "debit card annual fee",
    "cheque book charges",
    "IMPS charges",
]
STMT_CHARGES = [
    "SMS CHGS {Q}+GST",
    "ATM AMC CHGS",
    "MIN BAL CHGS",
    "DEBIT CARD ANNUAL FEE+GST",
    "CHQ BOOK CHGS",
    "IMPS CHGS+GST",
]


# --------------------------------------------------------------------------- templates
@dataclass(frozen=True)
class Tpl:
    id: str
    kinds: tuple[str, ...]
    text: str
    ood: bool = False


SMS = [
    Tpl(
        "s_upi_dr_1",
        ("upi_merchant", "upi_p2p_dr"),
        "Rs.{amt} debited from A/c {acct} on {date} to VPA {vpa} (UPI Ref No {ref}). Not you? Call 18001234567 to report -{bank}",
    ),
    Tpl(
        "s_upi_dr_2",
        ("upi_merchant", "upi_p2p_dr", "upi_rent"),
        "Dear Customer, {amt_inr} has been debited from your account {acct} on {date} towards UPI/{ref}/{name}/{note}. Avl Bal {bal_inr}. -{bank}",
    ),
    Tpl(
        "s_upi_dr_3",
        ("upi_merchant", "upi_p2p_dr"),
        "Sent {amt_rs} From {bank} A/C {acct} To {name} On {date} Ref {ref} Not You? Call 18001234567",
    ),
    Tpl(
        "s_upi_dr_4",
        ("upi_p2p_dr",),
        "{bank}: Your a/c {acct} is debited for {amt_rs} on {date} and a/c {other_acct} credited (UPI Ref No {ref}).",
    ),
    Tpl(
        "s_upi_dr_5",
        ("upi_merchant", "upi_p2p_dr", "upi_rent"),
        "A/c {acct} debited {amt_inr} on {date}. Info: UPI-{name}-{vpa}-{note}. Avl Bal: {bal_inr} -{bank}",
        ood=True,
    ),
    Tpl(
        "s_upi_dr_6",
        ("upi_merchant", "upi_p2p_dr", "upi_rent"),
        "{amt_rs} paid to {name} ({vpa}) from {bank} a/c {acct} on {date} {time}. UPI Ref: {ref}. Note: {note}",
    ),
    Tpl(
        "s_upi_cr_1",
        ("upi_p2p_cr",),
        "Rs.{amt} credited to A/c {acct} on {date} by VPA {vpa} (UPI Ref No {ref}) -{bank}",
    ),
    Tpl(
        "s_upi_cr_2",
        ("upi_p2p_cr", "upi_refund"),
        "Dear Customer, Acct {acct} is credited with {amt_rs} on {date} from {name}. UPI Ref:{ref} Note:{note} -{bank}",
        ood=True,
    ),
    Tpl(
        "s_upi_cr_3",
        ("upi_p2p_cr", "upi_refund"),
        "{bank}: {amt_inr} received in a/c {acct} from {name} via UPI on {date}. Ref {ref}. Remarks: {note}. Avl Bal {bal_inr}",
    ),
    Tpl(
        "s_imps_dr_1",
        ("imps_dr",),
        "IMPS: {amt_rs} debited from A/c {acct} on {date} to {name} A/c {other_acct}. IMPS Ref No {ref}. Avl Bal {bal_rs} -{bank}",
    ),
    Tpl(
        "s_imps_cr_1",
        ("imps_cr",),
        "Your A/c {acct} credited by {amt_rs} on {date} by IMPS from {name}. IMPS Ref No {ref}. Avl Bal {bal_rs} -{bank}",
    ),
    Tpl(
        "s_neft_cr_1",
        ("neft_salary", "neft_cr"),
        "{amt_inr} credited to your A/c {acct} on {date} through NEFT with UTR {ref} by {name}, Info: {note}. Avl Bal {bal_inr} -{bank}",
    ),
    Tpl(
        "s_neft_cr_2",
        ("neft_salary", "neft_cr"),
        "Dear Customer, your a/c {acct} has been credited with {amt_rs} on {date} towards NEFT from {name} ({note}). UTR: {ref}",
        ood=True,
    ),
    Tpl(
        "s_neft_dr_1",
        ("neft_dr", "neft_rent", "neft_tax"),
        "{amt_rs} debited from a/c {acct} on {date} for NEFT to {name}, UTR {ref}. Remarks: {note}. -{bank}",
    ),
    Tpl(
        "s_rtgs_cr_1",
        ("rtgs_cr",),
        "RTGS credit of {amt_inr} received in A/c {acct} on {date} from {name}. UTR {ref}. -{bank}",
    ),
    Tpl(
        "s_rtgs_dr_1",
        ("rtgs_dr",),
        "RTGS of {amt_inr} from A/c {acct} to {name} processed on {date}. UTR {ref}. -{bank}",
    ),
    Tpl(
        "s_card_1",
        ("card",),
        "{amt_rs} spent on {bank} Credit Card {card} at {merchant_card} on {date}. Avl Lmt: {limit_rs}. Not you? Call 18001234567",
    ),
    Tpl(
        "s_card_2",
        ("card",),
        "Thank you for using your {bank} Debit Card {card} for {amt_rs} at {merchant_card} on {date} {time}. Avl Bal {bal_rs}",
    ),
    Tpl(
        "s_card_3",
        ("card",),
        "Transaction alert: {amt_inr} was spent on your card ending {last4} at {merchant_card} on {date} {time}. RRN {ref}",
        ood=True,
    ),
    Tpl(
        "s_card_4",
        ("card",),
        "{bank}: Purchase of {amt_inr} on Card {card} at {merchant_card}, {city} on {date}. Ref no {ref}.",
    ),
    Tpl(
        "s_card_ref_1",
        ("card_refund",),
        "{amt_rs} refund from {merchant_card} has been credited to your {bank} Card {card} on {date}.",
    ),
    Tpl(
        "s_card_ref_2",
        ("card_refund",),
        "Reversal of {amt_inr} by {merchant_card} credited to card {card} on {date}. Ref {ref}",
    ),
    Tpl("s_atm_1", ("atm",), "{amt_rs} withdrawn from A/c {acct} on {date} at ATM {atm_id}. Avl Bal {bal_rs} -{bank}"),
    Tpl(
        "s_atm_2",
        ("atm",),
        "You have withdrawn {amt_inr} from ATM using Debit Card {card} on {date} {time}. Txn No {ref}. Avl Bal {bal_inr}",
        ood=True,
    ),
    Tpl(
        "s_atm_3",
        ("atm",),
        "{bank}: Cash withdrawal of {amt_rs} on card {card} at {atm_id}, {city} on {date}. RRN {ref}",
    ),
    Tpl(
        "s_nach_1",
        ("nach",),
        "Your A/c {acct} has been debited with {amt_rs} on {date} towards NACH mandate of {name}. Avl Bal: {bal_rs} -{bank}",
    ),
    Tpl("s_nach_2", ("nach",), "ACH debit of {amt_inr} for {name} processed from a/c {acct} on {date}. -{bank}"),
    Tpl(
        "s_nach_3",
        ("nach",),
        "{bank} Alert: Auto-debit (e-mandate) of {amt_rs} to {name} from A/c {acct} on {date} successful.",
        ood=True,
    ),
    Tpl(
        "s_int_1",
        ("interest",),
        "Interest of {amt_rs} credited to your Savings A/c {acct} on {date}. Avl Bal {bal_rs} -{bank}",
    ),
    Tpl("s_int_2", ("interest",), "{bank}: Quarterly interest {amt_inr} has been credited to a/c {acct} on {date}."),
    Tpl(
        "s_chg_1",
        ("charges",),
        "{amt_rs} debited from A/c {acct} on {date} towards {charge} incl. GST. Avl Bal {bal_rs} -{bank}",
    ),
    Tpl(
        "s_chg_2",
        ("charges",),
        "Dear Customer, {charge} of {amt_inr} (incl GST) has been recovered from your a/c {acct} on {date}. -{bank}",
        ood=True,
    ),
    Tpl("s_tds_1", ("tds",), "TDS of {amt_rs} deducted on interest from your A/c {acct} on {date}. -{bank}"),
    Tpl(
        "s_chq_dr_1",
        ("chq_paid",),
        "Cheque No. {ref} for {amt_rs} has been cleared and debited from your A/c {acct} on {date}. Payee: {name} -{bank}",
    ),
    Tpl(
        "s_chq_cr_1",
        ("chq_dep",),
        "Cheque {ref} deposited in A/c {acct} for {amt_rs} has been credited on {date}. Avl Bal {bal_rs} -{bank}",
    ),
]

UPI_KINDS = ("upi_merchant", "upi_refund", "upi_p2p_dr", "upi_p2p_cr", "upi_rent")
NARRATIONS = [
    Tpl("n_upi_1", UPI_KINDS, "UPI-{NAME}-{vpa}-{ifsc}-{ref}-{NOTE}"),
    Tpl("n_upi_2", UPI_KINDS, "UPI/{ref}/{NOTE}/{vpa}/{bankname}"),
    Tpl("n_upi_3", UPI_KINDS, "{TOBY} TRANSFER-UPI/{DRCR}/{ref}/{NAME_T}/{bank4}/{vpa}/{NOTE}--"),
    Tpl("n_upi_4", UPI_KINDS, "UPI/{P2X}/{ref}/{NAME}/{NOTE}/{bankshort}"),
    Tpl("n_upi_5", UPI_KINDS, "UPI/{NAME}/{ref}/{NOTE}", ood=True),
    Tpl("n_imps_1", ("imps_dr", "imps_cr"), "IMPS/P2A/{ref}/{NAME}/{bankshort}/{NOTE}"),
    Tpl("n_imps_2", ("imps_dr", "imps_cr"), "MMT/IMPS/{ref}/{NOTE}/{NAME}/{bankshort}"),
    Tpl("n_neft_1", ("neft_salary", "neft_cr"), "NEFT CR-{ifsc}-{NAME}-{NOTE}-{ref}"),
    Tpl("n_neft_2", ("neft_salary", "neft_cr", "neft_dr", "neft_rent", "neft_tax"), "NEFT/{ref}/{NAME}/{NOTE}"),
    Tpl("n_neft_3", ("neft_dr", "neft_rent", "neft_tax"), "NEFT DR-{ifsc}-{NAME}-NETBANK, MUM-{ref}-{NOTE}"),
    Tpl("n_rtgs_1", ("rtgs_cr", "rtgs_dr"), "RTGS {DRCR}-{ifsc}-{NAME}-{ref}"),
    Tpl("n_rtgs_2", ("rtgs_cr", "rtgs_dr"), "RTGS/{ref}/{NAME}/{bankshort}", ood=True),
    Tpl("n_card_1", ("card",), "POS {cardmask} {MERCHANT_CARD} {city}"),
    Tpl("n_card_2", ("card",), "POS/{MERCHANT_CARD}/{city}/{ref}"),
    Tpl("n_card_3", ("card",), "PCD/{last4}/{MERCHANT_CARD}/{city}/{ref}"),
    Tpl("n_card_ref_1", ("card_refund",), "POS REF {cardmask} {MERCHANT_CARD} REFUND"),
    Tpl("n_card_ref_2", ("card_refund",), "REV/POS/{MERCHANT_CARD}/{ref}"),
    Tpl("n_atm_1", ("atm",), "ATW-{cardmask}-{atm_id}-{city}"),
    Tpl("n_atm_2", ("atm",), "ATM WDL/{atm_id}/{city}/{ref}"),
    Tpl("n_atm_3", ("atm",), "NWD-{cardmask}-{atm_id}-{city}", ood=True),
    Tpl("n_nach_1", ("nach",), "ACH D- {NAME}-{mandate}"),
    Tpl("n_nach_2", ("nach",), "NACH-DR-{NAME}-{mandate}"),
    Tpl("n_nach_3", ("nach",), "ECS/{NAME}/{mandate}", ood=True),
    Tpl("n_int_1", ("interest",), "Credit Interest Capitalised"),
    Tpl("n_int_2", ("interest",), "INT.PD:{period}"),
    Tpl("n_int_3", ("interest",), "SB INTEREST CREDIT", ood=True),
    Tpl("n_chg_1", ("charges",), "{charge_code}"),
    Tpl("n_tds_1", ("tds",), "TDS ON INTEREST"),
    Tpl("n_tds_2", ("tds",), "TDS DEDUCTED U/S 194A"),
    Tpl("n_chq_dr_1", ("chq_paid",), "CHQ PAID-MICR CTS-{ref}-{NAME}"),
    Tpl("n_chq_dr_2", ("chq_paid",), "CLG/{ref}/{NAME}"),
    Tpl("n_chq_cr_1", ("chq_dep",), "CHQ DEP/{ref}/{bankshort}"),
    Tpl("n_chq_cr_2", ("chq_dep",), "CTS CLG DEP {ref}"),
]

LAYOUTS = [
    Tpl("l_1", (), "{d} | {narr} | Withdrawal: {dr} | Deposit: {cr} | Balance: {bal}"),
    Tpl("l_2", (), "{d}  {narr}  {amt} {DrCr}  {bal}"),
    Tpl("l_3", (), "{d}, {narr}, {signed}, {bal}"),
    Tpl("l_4", (), 'Txn Date,Description,Debit,Credit,Balance\n{d},"{narr}",{dr},{cr},{bal}', ood=True),
    Tpl("l_5", (), "Date: {d}\nDescription: {narr}\nAmount: {amt} {DrCr}\nClosing balance: {bal}"),
    Tpl("l_6", (), "{d} {narr} {signed}"),
]

SMS_DATE_FORMATS = ["%d-%m-%y", "%d%b%y", "%d-%b-%Y", "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d %b %Y", "%d.%m.%y"]
STMT_DATE_FORMATS = ["%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%d %b %Y", "%Y-%m-%d", "%d-%b-%y"]
START, END = date(2025, 4, 1), date(2026, 9, 30)


# --------------------------------------------------------------------------- helpers
def indian_group(n: float, decimals: bool) -> str:
    s = f"{n:.2f}" if decimals else f"{int(round(n))}"
    whole, _, frac = s.partition(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        head = re.sub(r"(\d)(?=(\d{2})+$)", r"\1,", head)
        whole = f"{head},{tail}"
    return f"{whole}.{frac}" if frac else whole


def fmt_num(rng: random.Random, n: float) -> str:
    is_whole = abs(n - round(n)) < 1e-9
    decimals = not is_whole or rng.random() < 0.6
    if rng.random() < 0.55:
        return indian_group(n, decimals)
    return f"{n:.2f}" if decimals else str(int(round(n)))


def digits(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(string.digits) for _ in range(n))


def plain_upper(name: str) -> str:
    return re.sub(r"[^A-Z0-9 &.\-]", "", name.upper().replace("+", " ")).replace("  ", " ").strip()


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def title(s: str) -> str:
    return " ".join(w.capitalize() for w in s.split())


@dataclass
class Example:
    text: str
    label: dict
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"text": self.text, "label": self.label, "meta": self.meta}


# --------------------------------------------------------------------------- generator
class Generator:
    def __init__(self, seed: int = 42, holdout_frac: float = 0.25):
        self.rng = random.Random(seed)
        split_rng = random.Random(7)  # merchant hold-out is fixed regardless of the sampling seed
        self.seen: dict[str, list[str]] = {}
        self.held: dict[str, list[str]] = {}
        for cat, names in MERCHANTS.items():
            names = sorted(names)
            split_rng.shuffle(names)
            k = max(1, round(len(names) * holdout_frac)) if len(names) > 1 else 0
            self.held[cat], self.seen[cat] = names[:k], names[k:]
        self.employer_seen = EMPLOYER_PREFIX[:15]
        self.employer_held = EMPLOYER_PREFIX[15:]

    # ---- parties
    def _person(self) -> str:
        r = self.rng
        parts = [r.choice(FIRST_NAMES), r.choice(LAST_NAMES)]
        if r.random() < 0.15:
            parts.insert(1, r.choice(FIRST_NAMES))
        return " ".join(parts).upper()

    def _person_vpa(self, name: str) -> str:
        r = self.rng
        first, last = name.lower().split()[0], name.lower().split()[-1]
        user = r.choice(
            [f"9{digits(r, 9)}", f"{first}{last[0]}{digits(r, 2)}", f"{first}.{last}", f"{first}{digits(r, 3)}"]
        )
        return f"{user}@{r.choice(UPI_HANDLES)}"

    def _employer(self, ood: bool) -> str:
        r = self.rng
        prefixes = self.employer_held if ood else self.employer_seen
        return f"{r.choice(prefixes)} {r.choice(EMPLOYER_SUFFIX)} {r.choice(EMPLOYER_FORM)}".upper()

    def _merchant(self, cats: list[str], ood: bool) -> tuple[str, str]:
        pool = self.held if ood else self.seen
        cats = [c for c in cats if pool.get(c)]
        cat = self.rng.choice(cats)
        return self.rng.choice(pool[cat]), cat

    def _merchant_forms(self, name: str) -> tuple[str, str, str]:
        """(narration form, card-terminal form, VPA) for a merchant."""
        r = self.rng
        up = plain_upper(name)
        narr = r.choice([up, up, up, f"{up} LTD", f"{up} PVT LTD", f"{up} INDIA PVT LTD", f"{up} LIMITED"])
        card = r.choice([up, up, f"{up} {r.choice(CITIES)}", up[:20].strip(), f"{up} ONLINE"])
        vpa = (
            f"{slug(name)}{r.choice(['', '', '', 'online', 'pay', '.payu', '1', 'india'])}@{r.choice(MERCHANT_HANDLES)}"
        )
        return narr, card, vpa

    # ---- values
    def _amount(self, category: str) -> float:
        lo, hi = AMOUNT_RANGE[category]
        r = self.rng
        x = lo * (hi / lo) ** r.random()  # log-uniform
        if category == "cash_withdrawal":
            return float(max(100, round(x / 100) * 100))
        if r.random() < 0.45:
            return float(round(x))
        return round(x, 2)

    def _ref(self, channel: str, bank4: str, d: date) -> str:
        r = self.rng
        if channel == "UPI":
            return r.choice("456") + digits(r, 11)
        if channel in ("IMPS", "CARD", "ATM"):
            return digits(r, 12)
        if channel == "NEFT":
            return r.choice([f"{bank4}N{digits(r, 11)}", f"N{digits(r, 15)}", f"{bank4}N5{d:%Y%m%d}{digits(r, 4)}"])
        if channel == "RTGS":
            return f"{bank4}R5{d:%Y%m%d}{digits(r, 8)}"
        if channel == "CHEQUE":
            return digits(r, 6)
        return ""

    def _mask(self, last4: str, card: bool = False) -> str:
        r = self.rng
        if card:
            return r.choice(
                [f"XX{last4}", f"XXXX{last4}", f"ending {last4}", f"xx{last4}", f"{digits(r, 4)}XXXXXXXX{last4}"]
            )
        return r.choice(
            [
                f"XX{last4}",
                f"XXXXXXXX{last4}",
                f"**{last4}",
                f"*{last4}",
                f"x{last4}",
                f"XXXXXX{last4}",
                f"ending {last4}",
            ]
        )

    def _note(self, kind_note: str | None, d: date) -> str:
        if not kind_note:
            return "NA"
        n = self.rng.choice(NOTES[kind_note])
        return n.format(mon=f"{d:%b}".lower(), MON=f"{d:%b}".upper(), yyyy=d.year, yy=f"{d:%y}")

    # ---- one example
    def example(self, mode: str = "id", source: str | None = None) -> Example:
        while True:
            ex = self._try(mode, source)
            if ex is not None:
                return ex

    def _try(self, mode: str, source: str | None) -> Example | None:
        r = self.rng
        kind = r.choices(list(KIND_WEIGHTS), weights=list(KIND_WEIGHTS.values()))[0]
        channel, direction, party, category, note_type = KINDS[kind]
        source = source or ("sms" if r.random() < 0.55 else "statement")
        ood = mode == "ood"
        want_ood_party = ood and r.random() < 0.6

        # ---- party
        cp_label, cp_shown, cp_card, vpa = None, None, None, None
        ood_party = False
        if party == "merchant":
            cats = CARD_CATS if channel == "CARD" else UPI_MERCHANT_CATS
            name, mcat = self._merchant(cats, want_ood_party)
            ood_party = want_ood_party
            category = category or mcat
            cp_label = name
            cp_shown, cp_card, vpa = self._merchant_forms(name)
        elif party == "mandate":
            cat = r.choice(sorted(MANDATE_CATEGORIES))
            name, mcat = self._merchant([cat], want_ood_party)
            ood_party = want_ood_party
            category = mcat
            cp_label, cp_shown, _, _ = name, *self._merchant_forms(name)[:1], None, None
        elif party == "tax":
            name, _ = self._merchant(["tax"], want_ood_party)
            ood_party = want_ood_party
            cp_label, cp_shown = name, plain_upper(name)
        elif party == "employer":
            shown = self._employer(want_ood_party)
            ood_party = want_ood_party
            cp_label, cp_shown = title(shown), shown
        elif party == "person":
            shown = self._person()
            cp_label, cp_shown, vpa = title(shown), shown, self._person_vpa(shown)

        # ---- template
        if source == "sms":
            pool = [t for t in SMS if kind in t.kinds]
        else:
            pool = [t for t in NARRATIONS if kind in t.kinds]
        if kind in NOTE_REQUIRED:
            pool = [t for t in pool if "{note}" in t.text or "{NOTE}" in t.text]
        if not ood:
            pool = [t for t in pool if not t.ood]
        if not pool:
            return None
        tpl = r.choice(pool)
        layout = None
        if source == "statement":
            layouts = LAYOUTS if ood else [t for t in LAYOUTS if not t.ood]
            layout = r.choice(layouts)
        ood_template = tpl.ood or bool(layout and layout.ood)
        if ood and not (ood_party or ood_template):
            return None
        if not ood and (ood_party or ood_template):
            return None

        # ---- values
        d = START + timedelta(days=r.randrange((END - START).days + 1))
        amount = self._amount(category)
        bank_name, bank4 = r.choice(BANKS)
        last4 = digits(r, 4)
        balance = round(r.uniform(500, 600000), 2) if r.random() < 0.85 else float(r.randrange(500, 200000))
        ref = self._ref(channel, r.choice(BANKS)[1], d)
        note = self._note(note_type, d)
        cp_bank = r.choice(BANKS)

        self._name_override = None
        if source == "sms":
            text, shown = self._render_sms(tpl, locals())
        else:
            text, shown = self._render_statement(tpl, layout, locals())

        # ---- label from what is shown
        if "name" in shown:
            counterparty = self._name_override or cp_label
        elif "vpa" in shown and party == "merchant":
            counterparty = cp_label  # merchant VPAs carry the brand name
        else:
            counterparty = None
        label = {
            "direction": direction,
            "amount": amount,
            "channel": channel,
            "counterparty": counterparty,
            "category": category,
            "reference": ref if "ref" in shown else None,
            "account_last4": last4 if "acct" in shown else None,
            "balance": balance if "bal" in shown else None,
            "date": d.isoformat() if "date" in shown else None,
        }
        meta = {
            "kind": kind,
            "source": source,
            "template": tpl.id + (f"+{layout.id}" if layout else ""),
            "ood_merchant": ood_party,
            "ood_template": ood_template,
        }
        return Example(text=text, label={k: label[k] for k in FIELDS}, meta=meta)

    # ---- renderers
    def _money(self, n: float, style: str) -> str:
        r = self.rng
        num = fmt_num(r, n)
        if style == "rs":
            return r.choice([f"Rs.{num}", f"Rs {num}", f"Rs.{num}", f"₹{num}"])
        if style == "inr":
            return r.choice([f"INR {num}", f"INR {num}", f"INR{num}", f"₹ {num}"])
        return num

    def _render_sms(self, tpl: Tpl, v: dict) -> tuple[str, set[str]]:
        r = self.rng
        d: date = v["d"]
        date_s = d.strftime(r.choice(SMS_DATE_FORMATS))
        if r.random() < 0.3:
            date_s = date_s.upper()
        name = v["cp_shown"] or ""
        if v["party"] == "person" and r.random() < 0.25:
            name = title(name)
        values = {
            "amt": fmt_num(r, v["amount"]),
            "amt_rs": self._money(v["amount"], "rs"),
            "amt_inr": self._money(v["amount"], "inr"),
            "bal_rs": self._money(v["balance"], "rs"),
            "bal_inr": self._money(v["balance"], "inr"),
            "limit_rs": self._money(round(r.uniform(5000, 500000), 2), "rs"),
            "acct": self._mask(v["last4"]),
            "card": self._mask(v["last4"], card=True),
            "last4": v["last4"],
            "other_acct": self._mask(digits(r, 4)),
            "date": date_s,
            "time": r.choice(
                [
                    f"{r.randrange(24):02d}:{r.randrange(60):02d}:{r.randrange(60):02d}",
                    f"{r.randrange(24):02d}:{r.randrange(60):02d}",
                ]
            ),
            "ref": v["ref"],
            "bank": v["bank_name"],
            "name": name,
            "vpa": v["vpa"] or "",
            "merchant_card": v["cp_card"] or "",
            "note": v["note"],
            "city": r.choice(CITIES).title(),
            "atm_id": r.choice(["S1AN", "S1BW", "SPCN", "DBLN"]) + digits(r, 6),
            "charge": r.choice(SMS_CHARGES),
        }
        text = tpl.text.format(**values)
        shown = self._shown(tpl.text)
        return text, shown

    def _render_statement(self, tpl: Tpl, layout: Tpl, v: dict) -> tuple[str, set[str]]:
        r = self.rng
        d: date = v["d"]
        debit = v["direction"] == "debit"
        name = v["cp_shown"] or ""
        name_t = name[: r.randint(10, 20)].strip() if len(name) > 12 else name
        if "{NAME_T}" in tpl.text and v["party"] == "person":
            self._name_override = title(name_t)
        bank_name, bank4 = v["cp_bank"]
        q_end = d.replace(day=1) - timedelta(days=1)
        q_start = (q_end - timedelta(days=88)).replace(day=1)
        values = {
            "NAME": name,
            "NAME_T": name_t,
            "vpa": v["vpa"] or "",
            "ifsc": f"{bank4}0{digits(r, 6)}",
            "ref": v["ref"],
            "NOTE": v["note"].upper() if r.random() < 0.6 else v["note"],
            "bankname": bank_name.upper(),
            "bank4": bank4,
            "bankshort": bank_name.split()[0].upper(),
            "TOBY": "TO" if debit else "BY",
            "DRCR": "DR" if debit else "CR",
            "P2X": "P2M" if v["party"] == "merchant" else "P2A",
            "cardmask": f"{digits(r, 4)}XXXXXX{v['last4']}" if r.random() < 0.5 else f"XXXXXXXXXXXX{v['last4']}",
            "last4": v["last4"],
            "MERCHANT_CARD": (v["cp_card"] or "").upper(),
            "city": r.choice(CITIES),
            "atm_id": r.choice(["S1AN", "S1BW", "SPCN", "DBLN"]) + digits(r, 6),
            "mandate": f"{r.choice(['P', 'U', 'M'])}{digits(r, 2)}{''.join(r.choice(string.ascii_uppercase) for _ in range(3))}{digits(r, 7)}",
            "period": f"{q_start:%d-%m-%Y} TO {q_end:%d-%m-%Y}",
            "charge_code": r.choice(STMT_CHARGES).format(Q=f"{q_start:%b}-{q_end:%b%y}".upper()),
        }
        narr = tpl.text.format(**values)
        if r.random() < 0.15:
            narr = narr.upper()
        amt = fmt_num(r, v["amount"])
        bal = fmt_num(r, v["balance"])
        lay = {
            "d": d.strftime(r.choice(STMT_DATE_FORMATS)),
            "narr": narr,
            "dr": amt if debit else "",
            "cr": "" if debit else amt,
            "amt": amt,
            "DrCr": r.choice(["Dr", "DR"]) if debit else r.choice(["Cr", "CR"]),
            "signed": f"-{amt}" if debit else r.choice([amt, f"+{amt}"]),
            "bal": bal,
        }
        text = layout.text.format(**lay)
        shown = self._shown(tpl.text) | {"date"} | ({"bal"} if "{bal}" in layout.text else set())
        if "cardmask" in shown or "last4" in shown:
            shown.add("acct")
        return text, shown

    @staticmethod
    def _shown(text: str) -> set[str]:
        keys = set(re.findall(r"\{(\w+)\}", text))
        shown = set()
        if keys & {"acct", "card", "last4", "cardmask"}:
            shown |= {"acct"} | (keys & {"cardmask", "last4"})
        if keys & {"bal_rs", "bal_inr", "bal"}:
            shown.add("bal")
        if keys & {"date", "d"}:
            shown.add("date")
        if "ref" in keys:
            shown.add("ref")
        if keys & {"name", "NAME", "NAME_T", "merchant_card", "MERCHANT_CARD"}:
            shown.add("name")
        if "vpa" in keys:
            shown.add("vpa")
        return shown


# --------------------------------------------------------------------------- dataset
SPLITS = {"train": ("id", 5000), "val": ("id", 300), "test_id": ("id", 500), "test_ood": ("ood", 500)}


def build(out: Path, seed: int = 42, sizes: dict[str, int] | None = None) -> dict[str, list[dict]]:
    gen = Generator(seed=seed)
    seen_text: set[str] = set()
    data: dict[str, list[dict]] = {}
    for split, (mode, n) in SPLITS.items():
        n = (sizes or {}).get(split, n)
        rows: list[dict] = []
        while len(rows) < n:
            ex = gen.example(mode)
            if ex.text in seen_text:
                continue
            seen_text.add(ex.text)
            ex.meta["id"] = f"{split}-{len(rows):05d}"
            rows.append(ex.to_dict())
        data[split] = rows
    out.mkdir(parents=True, exist_ok=True)
    for split, rows in data.items():
        with open(out / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "seed": seed,
        "sizes": {k: len(v) for k, v in data.items()},
        "held_out_merchants": {k: v for k, v in gen.held.items() if v},
        "held_out_templates": sorted(t.id for t in SMS + NARRATIONS + LAYOUTS if t.ood),
        "kinds_train": dict(Counter(r["meta"]["kind"] for r in data["train"]).most_common()),
        "categories_train": dict(Counter(r["label"]["category"] for r in data["train"]).most_common()),
    }
    (out / "dataset_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return data


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("data"))
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(argv)
    data = build(a.out, a.seed)
    print({k: len(v) for k, v in data.items()})


if __name__ == "__main__":
    main()
