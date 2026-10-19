import json
from pathlib import Path

import pytest

from txnparser.evaluate import field_match, norm_party, score
from txnparser.generate import Generator, build, indian_group
from txnparser.rules import RuleParser
from txnparser.schema import CATEGORIES, CHANNELS, FIELDS, messages, parse_output, to_json

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    out = tmp_path_factory.mktemp("data")
    return build(out, seed=3, sizes={"train": 400, "val": 50, "test_id": 80, "test_ood": 80})


def test_generation_is_deterministic(tmp_path):
    a = build(tmp_path / "a", seed=11, sizes={"train": 30, "val": 5, "test_id": 5, "test_ood": 5})
    b = build(tmp_path / "b", seed=11, sizes={"train": 30, "val": 5, "test_id": 5, "test_ood": 5})
    assert a == b


def test_labels_are_valid(small):
    for rows in small.values():
        for r in rows:
            lab = r["label"]
            assert list(lab) == FIELDS
            assert lab["direction"] in {"debit", "credit"}
            assert lab["channel"] in CHANNELS and lab["category"] in CATEGORIES
            assert isinstance(lab["amount"], float) and lab["amount"] > 0
            assert lab["account_last4"] is None or (len(lab["account_last4"]) == 4 and lab["account_last4"].isdigit())


def test_label_only_contains_what_the_text_shows(small):
    for r in small["train"]:
        lab, text = r["label"], r["text"]
        if lab["reference"]:
            assert lab["reference"] in text
        if lab["account_last4"]:
            assert lab["account_last4"] in text
        if lab["counterparty"] and r["meta"]["kind"] not in {
            "upi_merchant",
            "card",
            "card_refund",
            "upi_refund",
            "nach",
            "neft_tax",
        }:
            assert lab["counterparty"].upper() in text.upper()


def test_splits_do_not_overlap_and_ood_is_really_ood(small):
    texts = [r["text"] for rows in small.values() for r in rows]
    assert len(texts) == len(set(texts))
    assert all(r["meta"]["ood_merchant"] or r["meta"]["ood_template"] for r in small["test_ood"])
    assert not any(r["meta"]["ood_merchant"] or r["meta"]["ood_template"] for r in small["train"])
    held = {m for ms in Generator().held.values() for m in ms}
    assert not {r["label"]["counterparty"] for r in small["train"]} & held


def test_committed_dataset_matches_generator():
    summary = json.loads((DATA / "dataset_summary.json").read_text())
    assert summary["sizes"] == {"train": 5000, "val": 300, "test_id": 500, "test_ood": 500}
    first = json.loads((DATA / "test_ood.jsonl").read_text().splitlines()[0])
    assert first["meta"]["id"] == "test_ood-00000"


def test_indian_number_grouping():
    assert indian_group(1234567.5, True) == "12,34,567.50"
    assert indian_group(999, False) == "999"
    assert indian_group(100000, False) == "1,00,000"


@pytest.mark.parametrize(
    ("field", "gold", "pred", "ok"),
    [
        ("amount", 1234.5, "1,234.50", True),
        ("amount", 1234.5, 1234.0, False),
        ("counterparty", "Nimbus Analytics Private Limited", "NIMBUS ANALYTICS PVT LTD", True),
        ("counterparty", "Domino's Pizza", "Dominos Pizza", True),
        ("counterparty", None, "Swiggy", False),
        ("account_last4", "1234", "XX1234", True),
        ("reference", "N123", None, False),
        ("date", "2026-01-02", "2026-01-02", True),
        ("category", "food_dining", "FOOD_DINING", True),
    ],
)
def test_field_match(field, gold, pred, ok):
    assert field_match(field, gold, pred) is ok


def test_norm_party():
    assert norm_party("Big Basket") == norm_party("BIGBASKET")
    assert norm_party("") is None


def test_parse_output_handles_fences_and_garbage():
    lab = {f: None for f in FIELDS} | {"direction": "debit", "amount": 10.0}
    assert parse_output(to_json(lab)) == lab
    assert parse_output("```json\n" + to_json(lab) + "\n```") == lab
    assert parse_output("Sure! " + to_json(lab) + " Hope this helps") == lab
    assert parse_output("not json") is None


def test_chat_messages_shape():
    m = messages("Rs 10 debited", {f: None for f in FIELDS})
    assert [x["role"] for x in m] == ["system", "user", "assistant"]


def test_score_and_rules_baseline(small):
    parser = RuleParser(small["train"])
    preds = {r["meta"]["id"]: parser.parse(r["text"]) for r in small["test_id"]}
    m = score(small["test_id"], preds)["metrics"]["all"]
    assert m["valid_json"] == 100.0 and m["field_accuracy"] > 85
    perfect = {r["meta"]["id"]: r["label"] for r in small["test_id"]}
    assert score(small["test_id"], perfect)["metrics"]["all"]["exact_match"] == 100.0
    assert score(small["test_id"], {})["metrics"]["all"]["valid_json"] == 0.0
