"""Batched greedy inference for the base model (few-shot prompt) or the fine-tuned model (short prompt).

Usage:
  python -m txnparser.infer --model Qwen/Qwen2.5-1.5B-Instruct --mode base --name base_fewshot
  python -m txnparser.infer --model Qwen/Qwen2.5-1.5B-Instruct --adapter runs/qlora/adapter --mode ft --name ft
Writes results/preds/<name>_<split>.jsonl with id, raw output, parsed prediction and per-example latency.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .evaluate import load_jsonl
from .schema import SYSTEM_PROMPT, messages, parse_output, to_json, user_prompt
from .train import SHORT_SYSTEM, dtype_kwarg

# (source, kind) of the training examples shown to the base model as demonstrations
FEW_SHOT = [("sms", "upi_merchant"), ("statement", "neft_salary"), ("sms", "atm"), ("statement", "nach")]


def few_shot_messages(train_rows: list[dict], text: str) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for source, kind in FEW_SHOT:
        ex = next(r for r in train_rows if r["meta"]["source"] == source and r["meta"]["kind"] == kind)
        msgs += [
            {"role": "user", "content": user_prompt(ex["text"])},
            {"role": "assistant", "content": to_json(ex["label"])},
        ]
    msgs.append({"role": "user", "content": user_prompt(text)})
    return msgs


def build_prompts(tok, rows: list[dict], mode: str, train_rows: list[dict]) -> list[str]:
    out = []
    for r in rows:
        msgs = few_shot_messages(train_rows, r["text"]) if mode == "base" else messages(r["text"], system=SHORT_SYSTEM)
        out.append(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False))
    return out


def load(model_name: str, adapter: str | None, merged_out: Path | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(adapter or model_name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, **{dtype_kwarg(): dtype})
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
        if merged_out:
            model.save_pretrained(merged_out, safe_serialization=True)
            tok.save_pretrained(merged_out)
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    return model, tok


def generate(model, tok, prompts: list[str], batch: int = 32, max_new_tokens: int = 192) -> list[tuple[str, float]]:
    import torch

    out: list[tuple[str, float]] = []
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))  # similar lengths -> less padding
    results: dict[int, tuple[str, float]] = {}
    for s in range(0, len(order), batch):
        idx = order[s : s + batch]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(
            model.device
        )
        t0 = time.perf_counter()
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tok.pad_token_id,
            )
        ms = (time.perf_counter() - t0) * 1000 / len(idx)
        texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        for i, t in zip(idx, texts, strict=True):
            results[i] = (t, ms)
    for i in range(len(prompts)):
        out.append(results[i])
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--adapter")
    p.add_argument("--mode", choices=["base", "ft"], required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, default=Path("results/preds"))
    p.add_argument("--splits", nargs="+", default=["test_id", "test_ood"])
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--save-merged", type=Path, help="also save the merged fine-tuned model here")
    a = p.parse_args(argv)

    model, tok = load(a.model, a.adapter, a.save_merged)
    train_rows = load_jsonl(a.data / "train.jsonl")
    a.out.mkdir(parents=True, exist_ok=True)
    for split in a.splits:
        rows = load_jsonl(a.data / f"{split}.jsonl")
        if a.limit:
            rows = rows[: a.limit]
        prompts = build_prompts(tok, rows, a.mode, train_rows)
        n_tok = sum(len(tok(pr, add_special_tokens=False)["input_ids"]) for pr in prompts) / len(prompts)
        t0 = time.time()
        outs = generate(model, tok, prompts, batch=a.batch)
        path = a.out / f"{a.name}_{split}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r, (raw, ms) in zip(rows, outs, strict=True):
                rec = {"id": r["meta"]["id"], "raw": raw, "pred": parse_output(raw), "latency_ms": round(ms, 1)}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"{split}: {len(rows)} examples in {time.time() - t0:.0f}s, prompt tokens/example={n_tok:.0f} -> {path}")


if __name__ == "__main__":
    main()
