"""Check that the quantised GGUF build (used by the CPU demo) scores the same as the fp16 model.

Runs the GGUF with llama.cpp on a fixed subset of each test split and scores it next to the fp16 fine-tuned
model's predictions for the same examples.

Usage:  python -m txnparser.gguf_eval --gguf model-q8_0.gguf --n 100
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from .evaluate import load_jsonl, load_preds, score
from .schema import parse_output, user_prompt
from .train import SHORT_SYSTEM


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gguf", type=Path, required=True)
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--preds", type=Path, default=Path("results/preds"))
    p.add_argument("--n", type=int, default=100, help="examples per split (every k-th row, fixed)")
    p.add_argument("--splits", nargs="+", default=["test_id", "test_ood"])
    p.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    a = p.parse_args(argv)

    from llama_cpp import Llama

    llm = Llama(model_path=str(a.gguf), n_ctx=1024, n_threads=a.threads, verbose=False)
    summary = {"gguf": a.gguf.name, "size_mb": round(a.gguf.stat().st_size / 2**20), "splits": {}}
    for split in a.splits:
        rows = load_jsonl(a.data / f"{split}.jsonl")
        rows = rows[:: max(1, len(rows) // a.n)][: a.n]
        out, lat = [], []
        for r in rows:
            t0 = time.perf_counter()
            res = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": SHORT_SYSTEM},
                    {"role": "user", "content": user_prompt(r["text"])},
                ],
                temperature=0.0,
                max_tokens=192,
            )
            lat.append((time.perf_counter() - t0) * 1000)
            raw = res["choices"][0]["message"]["content"]
            out.append({"id": r["meta"]["id"], "raw": raw, "pred": parse_output(raw), "latency_ms": round(lat[-1], 1)})
        with open(a.preds / f"gguf_{split}.jsonl", "w", encoding="utf-8") as f:
            for o in out:
                f.write(json.dumps(o, ensure_ascii=False) + "\n")
        gguf_m = score(rows, {o["id"]: o["pred"] for o in out})["metrics"]["all"]
        fp16 = load_preds(a.preds / f"ft_{split}.jsonl")
        fp16_m = score(rows, fp16)["metrics"]["all"]
        lat.sort()
        summary["splits"][split] = {
            "n": len(rows),
            "gguf_exact_match": gguf_m["exact_match"],
            "fp16_exact_match": fp16_m["exact_match"],
            "gguf_field_accuracy": gguf_m["field_accuracy"],
            "fp16_field_accuracy": fp16_m["field_accuracy"],
            "gguf_median_cpu_latency_ms": round(lat[len(lat) // 2]),
        }
        print(split, summary["splits"][split], flush=True)
    Path("results/gguf_check.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
