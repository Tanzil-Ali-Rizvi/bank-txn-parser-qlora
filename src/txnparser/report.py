"""Collect every system's predictions into one comparison: results/metrics.json and results/RESULTS.md.

Usage:  python -m txnparser.report --systems rules base_fewshot ft
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

from .evaluate import load_jsonl, load_preds, score
from .schema import FIELDS

LABELS = {
    "rules": "Regex + merchant dictionary",
    "base_fewshot": "Qwen2.5-1.5B-Instruct, 4-shot prompt",
    "ft": "Qwen2.5-1.5B + QLoRA (this repo)",
}
SPLIT_LABELS = {"test_id": "Seen formats & merchants", "test_ood": "Unseen formats or merchants"}


def build_report(data: Path, preds_dir: Path, systems: list[str], splits: list[str]) -> dict:
    out: dict = {"systems": {}, "splits": splits}
    for sysname in systems:
        out["systems"][sysname] = {"label": LABELS.get(sysname, sysname)}
        for split in splits:
            path = preds_dir / f"{sysname}_{split}.jsonl"
            if not path.exists():
                continue
            gold = load_jsonl(data / f"{split}.jsonl")
            res = score(gold, load_preds(path))
            recs = load_jsonl(path)
            lat = [r["latency_ms"] for r in recs if "latency_ms" in r]
            m = res["metrics"]
            out["systems"][sysname][split] = {
                "overall": m["all"],
                "by_source": {k.split(":")[1]: v["exact_match"] for k, v in m.items() if k.startswith("source:")},
                "by_kind": {k.split(":")[1]: v["exact_match"] for k, v in m.items() if k.startswith("kind:")},
                "median_latency_ms": round(median(lat), 1) if lat else None,
                "error_sample": res["errors"][:15],
            }
    return out


def to_markdown(rep: dict) -> str:
    systems = [s for s in rep["systems"] if any(sp in rep["systems"][s] for sp in rep["splits"])]
    lines = ["# Results", "", "Exact match = every one of the 9 fields correct for a transaction.", ""]
    head = (
        "| System | "
        + " | ".join(f"{SPLIT_LABELS.get(sp, sp)}: exact match | field accuracy" for sp in rep["splits"])
        + " | Valid JSON | Median latency |"
    )
    lines += [head, "|" + "---|" * (2 + 2 * len(rep["splits"]) + 1)]
    for s in systems:
        row = [rep["systems"][s]["label"]]
        valid, lat = [], None
        for sp in rep["splits"]:
            m = rep["systems"][s].get(sp)
            if not m:
                row += ["–", "–"]
                continue
            row += [f"**{m['overall']['exact_match']}%**", f"{m['overall']['field_accuracy']}%"]
            valid.append(m["overall"]["valid_json"])
            lat = lat or m["median_latency_ms"]
        row += [f"{min(valid)}%" if valid else "–", f"{lat:.0f} ms" if lat else "–"]
        lines.append("| " + " | ".join(row) + " |")
    for sp in rep["splits"]:
        lines += ["", f"## Field accuracy: {SPLIT_LABELS.get(sp, sp)}", ""]
        lines += [
            "| Field | " + " | ".join(rep["systems"][s]["label"] for s in systems) + " |",
            "|---|" + "---|" * len(systems),
        ]
        for f in FIELDS:
            vals = [
                f"{rep['systems'][s][sp]['overall']['fields'][f]}%" if sp in rep["systems"][s] else "–" for s in systems
            ]
            lines.append(f"| {f} | " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--preds", type=Path, default=Path("results/preds"))
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--systems", nargs="+", default=["rules", "base_fewshot", "ft"])
    p.add_argument("--splits", nargs="+", default=["test_id", "test_ood"])
    a = p.parse_args(argv)
    rep = build_report(a.data, a.preds, a.systems, a.splits)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "metrics.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False) + "\n")
    md = to_markdown(rep)
    (a.out / "RESULTS.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
