"""Score predictions against gold labels: per-field accuracy, exact match and valid-JSON rate.

Usage:  python -m txnparser.evaluate --gold data/test_id.jsonl --pred results/preds/ft_test_id.jsonl
Prediction file: one JSON object per line with "id" and "pred" (dict, or null when the output was not valid JSON).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from .schema import FIELDS

_STOP = {"pvt", "ltd", "limited", "private", "llp", "inc", "india", "the", "co", "company", "corp"}


def _num(v) -> float | None:
    if v is None or v == "":
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, int | float):
        return float(v)
    s = re.sub(r"[^\d.\-]", "", str(v))
    try:
        return float(s)
    except ValueError:
        return None


def norm_party(v) -> str | None:
    if v is None:
        return None
    s = str(v).lower().replace("&", " and ").replace("+", " ")
    s = re.sub(r"[^a-z0-9 ]", "", s)
    toks = [t for t in s.split() if t not in _STOP]
    out = "".join(toks)
    return out or None


def _str(v) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return s.lower() if s and s.lower() not in {"null", "none", "n/a", "na"} else None


def field_match(name: str, gold, pred) -> bool:
    if name in ("amount", "balance"):
        g, p = _num(gold), _num(pred)
        return (g is None and p is None) or (g is not None and p is not None and abs(g - p) < 0.011)
    if name == "counterparty":
        return norm_party(gold) == norm_party(pred)
    if name == "account_last4":
        g = None if gold is None else str(gold)[-4:]
        p = None if pred is None else re.sub(r"\D", "", str(pred))[-4:] or None
        return g == p
    if name == "reference":
        g = None if gold is None else str(gold).strip().upper()
        p = None if pred in (None, "") else str(pred).strip().upper()
        return g == p
    return _str(gold) == _str(pred)


def score(gold_rows: list[dict], preds: dict[str, dict | None]) -> dict:
    """Aggregate metrics, plus breakdowns by source (sms / statement) and transaction kind."""

    def empty():
        return {"n": 0, "valid": 0, "exact": 0, "fields": defaultdict(int)}

    buckets: dict[str, dict] = defaultdict(empty)
    errors: list[dict] = []
    for row in gold_rows:
        gid = row["meta"]["id"]
        pred = preds.get(gid)
        gold = row["label"]
        ok = {f: pred is not None and field_match(f, gold[f], pred.get(f)) for f in FIELDS}
        keys = ["all", f"source:{row['meta']['source']}", f"kind:{row['meta']['kind']}"]
        for k in keys:
            b = buckets[k]
            b["n"] += 1
            b["valid"] += pred is not None
            b["exact"] += all(ok.values())
            for f, v in ok.items():
                b["fields"][f] += v
        if not all(ok.values()) and len(errors) < 200:
            errors.append(
                {
                    "id": gid,
                    "text": row["text"],
                    "gold": gold,
                    "pred": pred,
                    "wrong": [f for f, v in ok.items() if not v],
                }
            )

    def finish(b):
        n = b["n"] or 1
        fields = {f: round(100 * b["fields"][f] / n, 1) for f in FIELDS}
        return {
            "n": b["n"],
            "valid_json": round(100 * b["valid"] / n, 1),
            "exact_match": round(100 * b["exact"] / n, 1),
            "field_accuracy": round(sum(fields.values()) / len(fields), 1),
            "fields": fields,
        }

    out = {k: finish(b) for k, b in sorted(buckets.items(), key=lambda kv: (kv[0] != "all", kv[0]))}
    return {"metrics": out, "errors": errors}


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_preds(path: Path) -> dict[str, dict | None]:
    return {r["id"]: r.get("pred") for r in load_jsonl(path)}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gold", type=Path, required=True)
    p.add_argument("--pred", type=Path, required=True)
    p.add_argument("--out", type=Path, help="write metrics + error sample as JSON")
    a = p.parse_args(argv)
    res = score(load_jsonl(a.gold), load_preds(a.pred))
    m = res["metrics"]["all"]
    print(
        f"n={m['n']}  valid_json={m['valid_json']}%  exact_match={m['exact_match']}%  field_acc={m['field_accuracy']}%"
    )
    for f, v in m["fields"].items():
        print(f"  {f:<14} {v}%")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
