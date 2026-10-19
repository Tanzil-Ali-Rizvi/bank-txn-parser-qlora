"""Publish the fine-tuned model (merged weights + LoRA adapter + results) and the dataset to the Hugging Face Hub.

Needs a write token in the HF_TOKEN environment variable (on Kaggle: Add-ons -> Secrets).

Usage:  python -m txnparser.publish_hf --user TARizvi --merged runs/merged --adapter runs/qlora/adapter
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .schema import CATEGORIES, CHANNELS
from .train import SHORT_SYSTEM

GITHUB = "https://github.com/Tanzil-Ali-Rizvi/bank-txn-parser-qlora"


def _row(rep: dict, system: str, split: str) -> str:
    m = rep["systems"].get(system, {}).get(split)
    return f"{m['overall']['exact_match']}%" if m else "–"


def model_card(user: str, dataset_repo: str, rep: dict, info: dict) -> str:
    rows = "\n".join(
        f"| {rep['systems'][s]['label']} | {_row(rep, s, 'test_id')} | {_row(rep, s, 'test_ood')} |"
        for s in rep["systems"]
    )
    return f"""---
license: apache-2.0
base_model: {info["base_model"]}
library_name: transformers
pipeline_tag: text-generation
language: [en]
datasets: [{dataset_repo}]
tags: [qlora, peft, lora, finance, banking, india, upi, information-extraction, structured-output, json]
---

# Bank transaction parser: {info["base_model"].split("/")[-1]} fine-tuned with QLoRA

Turns Indian bank **SMS alerts and statement lines** (UPI, IMPS, NEFT, RTGS, card, ATM, NACH, cheque, bank
entries) into clean JSON: direction, amount, channel, counterparty, category, reference, own account, balance, date.

Code, dataset generator, baselines and evaluation: **[{GITHUB.split("/")[-1]}]({GITHUB})**

## Results (exact match: all 9 fields correct)

| System | Seen formats & merchants | Unseen formats or merchants |
|---|---|---|
{rows}

The unseen split contains only transactions with a merchant (e.g. Swiggy, Zomato, Myntra) or a message format
that never appears in training. Full per-field results are in `results/`.

## Training

- QLoRA: 4-bit NF4 base with double quantisation, LoRA r={info["lora"]["r"]}, alpha={info["lora"]["alpha"]} on all
  linear layers ({info["trainable_params"]:,} trainable parameters, {100 * info["trainable_params"] / info["total_params"]:.2f}%)
- {info["train_examples"]:,} synthetic examples, {info["epochs"]:g} epochs, lr {info["learning_rate"]}, effective batch
  {info["effective_batch"]}, loss on the JSON answer only
- {info["train_minutes"]} minutes on one {info["gpu"]} (free Kaggle GPU)

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

repo = "{user}/bank-txn-parser-qwen2.5-1.5b"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype="auto", device_map="auto")

text = "Rs.450.00 debited from A/c XX1234 on 12-10-26 to VPA swiggy@icici (UPI Ref No 412345678901)."
msgs = [{{"role": "system", "content": "{SHORT_SYSTEM}"}},
        {{"role": "user", "content": "Transaction:\\n" + text}}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
out = model.generate(ids, max_new_tokens=192, do_sample=False)
print(json.loads(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)))
```

The LoRA adapter alone is in `adapter/` (load it on top of the base model with `peft`).

**Channels:** {", ".join(CHANNELS)}
**Categories:** {", ".join(CATEGORIES)}

## Limitations

- Trained on synthetic data that imitates common Indian bank formats; real banks' formats vary more. Validate on
  your own messages before relying on it.
- Bank names in the training data are fictional; merchant names are real brands used only as counterparties.
- Like any generative model it can be confidently wrong. For money movement, use it as a pre-fill that a person or
  a rule checks, not as the source of truth.
"""


def dataset_card(summary: dict) -> str:
    held = ", ".join(m for ms in summary["held_out_merchants"].values() for m in ms)
    return f"""---
license: apache-2.0
language: [en]
task_categories: [text-generation]
tags: [finance, banking, india, upi, information-extraction, synthetic]
size_categories: [1K<n<10K]
configs:
  - config_name: default
    data_files:
      - {{split: train, path: train.jsonl}}
      - {{split: validation, path: val.jsonl}}
      - {{split: test_id, path: test_id.jsonl}}
      - {{split: test_ood, path: test_ood.jsonl}}
---

# Indian bank transaction parsing (synthetic)

{sum(summary["sizes"].values()):,} synthetic Indian bank SMS alerts and statement lines with JSON labels, built by a
seeded generator ([code]({GITHUB})). Each label only contains what its text actually shows.

| Split | Examples | What it tests |
|---|---|---|
| train | {summary["sizes"]["train"]:,} | |
| validation | {summary["sizes"]["val"]:,} | same distribution as train |
| test_id | {summary["sizes"]["test_id"]:,} | seen merchants and formats, new values |
| test_ood | {summary["sizes"]["test_ood"]:,} | every example has a merchant or a format never seen in training |

Held-out merchants: {held}.
Held-out formats: {", ".join(summary["held_out_templates"])}.

Bank names and IFSC prefixes are fictional; merchant names are real brands used only as counterparties. No real
customer data is included.
"""


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--user", required=True)
    p.add_argument("--merged", type=Path, required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--train-info", type=Path, default=Path("runs/qlora/train_info.json"))
    p.add_argument("--private", action="store_true")
    a = p.parse_args(argv)

    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    model_repo = f"{a.user}/bank-txn-parser-qwen2.5-1.5b"
    dataset_repo = f"{a.user}/indian-bank-txn-parsing"
    rep = json.loads((a.results / "metrics.json").read_text())
    info = json.loads(a.train_info.read_text())
    summary = json.loads((a.data / "dataset_summary.json").read_text())

    api.create_repo(dataset_repo, repo_type="dataset", private=a.private, exist_ok=True)
    (a.data / "README.md").write_text(dataset_card(summary))
    api.upload_folder(
        folder_path=str(a.data),
        repo_id=dataset_repo,
        repo_type="dataset",
        commit_message="Upload synthetic Indian bank transaction dataset",
    )
    (a.data / "README.md").unlink()

    if not (a.merged / "config.json").exists():
        raise SystemExit(f"No merged model in {a.merged}: run txnparser.infer with --save-merged first.")
    api.create_repo(model_repo, private=a.private, exist_ok=True)
    (a.merged / "README.md").write_text(model_card(a.user, dataset_repo, rep, info))
    api.upload_folder(folder_path=str(a.merged), repo_id=model_repo, commit_message="Upload merged fine-tuned model")
    api.upload_folder(
        folder_path=str(a.adapter), repo_id=model_repo, path_in_repo="adapter", commit_message="Upload LoRA adapter"
    )
    api.upload_folder(
        folder_path=str(a.results),
        repo_id=model_repo,
        path_in_repo="results",
        allow_patterns=["*.json", "*.md", "preds/*.jsonl"],
        commit_message="Upload evaluation results",
    )
    api.upload_file(path_or_fileobj=str(a.train_info), path_in_repo="results/train_info.json", repo_id=model_repo)
    print(f"https://huggingface.co/{model_repo}\nhttps://huggingface.co/datasets/{dataset_repo}")


if __name__ == "__main__":
    main()
