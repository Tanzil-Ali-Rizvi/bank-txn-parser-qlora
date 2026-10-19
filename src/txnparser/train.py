"""QLoRA supervised fine-tuning of a small instruct model on the transaction-parsing dataset.

Loss is computed on the assistant's JSON answer only (prompt tokens are masked), using the plain Hugging Face
Trainer so the script does not depend on any one TRL version.

On a free Kaggle / Colab T4 (16 GB) with the defaults (Qwen2.5-1.5B-Instruct, 4-bit NF4, LoRA r=16 on all
linear layers, 2 epochs over 5,000 examples) training takes roughly half an hour.

Usage:
  python -m txnparser.train --data data --out runs/qwen1.5b-qlora
  python -m txnparser.train --model <tiny-model-dir> --no-4bit --max-steps 5   # CPU smoke test
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from .evaluate import load_jsonl
from .schema import messages

SHORT_SYSTEM = "Parse this Indian bank transaction into JSON."
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def encode(tokenizer, text: str, label: dict, max_len: int = 512) -> dict:
    """Token ids for one example with the prompt masked out of the labels."""
    msgs = messages(text, label, system=SHORT_SYSTEM)
    prompt_ids = tokenizer.apply_chat_template(msgs[:-1], add_generation_prompt=True, tokenize=True)
    full_ids = tokenizer.apply_chat_template(msgs, tokenize=True)
    if hasattr(prompt_ids, "input_ids"):  # some tokenizer versions return a BatchEncoding
        prompt_ids, full_ids = prompt_ids["input_ids"], full_ids["input_ids"]
    # The chat template may append a trailing newline after the end-of-turn token; keep the answer + EOT.
    assert full_ids[: len(prompt_ids)] == prompt_ids, "chat template prefix mismatch"
    full_ids = full_ids[:max_len]
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels[:max_len]}


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict:
        import torch

        n = max(len(b["input_ids"]) for b in batch)

        def pad(key, value):
            return torch.tensor([b[key] + [value] * (n - len(b[key])) for b in batch])

        return {
            "input_ids": pad("input_ids", self.pad_id),
            "attention_mask": pad("attention_mask", 0),
            "labels": pad("labels", -100),
        }


def dtype_kwarg() -> str:
    """transformers renamed `torch_dtype` to `dtype` in 4.56."""
    import transformers

    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    return "dtype" if (major, minor) >= (4, 56) else "torch_dtype"


def make_training_args(cls, **kwargs):
    """Build TrainingArguments, dropping options this transformers version no longer accepts."""
    import inspect

    accepted = set(inspect.signature(cls.__init__).parameters)
    dropped = sorted(k for k in kwargs if k not in accepted)
    if dropped:
        print(f"note: this transformers version ignores {dropped}")
    return cls(**{k: v for k, v in kwargs.items() if k in accepted})


def load_model(model_name: str, use_4bit: bool, gradient_checkpointing: bool = True):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs: dict = {}
    if use_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        kwargs["device_map"] = {"": 0}
        kwargs[dtype_kwarg()] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if use_4bit:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)
    return model, tok


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, default=Path("runs/qlora"))
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--train-limit", type=int, default=0, help="use only the first N training examples")
    p.add_argument("--no-4bit", action="store_true", help="full-precision LoRA (CPU smoke tests)")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(argv)

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments, set_seed

    set_seed(a.seed)
    use_4bit = not a.no_4bit
    model, tok = load_model(a.model, use_4bit)
    targets = [t for t in LORA_TARGETS if any(n.endswith(t) for n, _ in model.named_modules())]
    model = get_peft_model(
        model,
        LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.05, target_modules=targets, task_type="CAUSAL_LM"),
    )
    trainable, total = model.get_nb_trainable_parameters()
    print(f"LoRA targets={targets} trainable={trainable:,} ({100 * trainable / total:.2f}% of {total:,})")

    train_rows = load_jsonl(a.data / "train.jsonl")
    if a.train_limit:
        train_rows = train_rows[: a.train_limit]
    val_rows = load_jsonl(a.data / "val.jsonl")[:200]
    train_ds = [encode(tok, r["text"], r["label"]) for r in train_rows]
    val_ds = [encode(tok, r["text"], r["label"]) for r in val_rows]
    lengths = sorted(len(x["input_ids"]) for x in train_ds)
    print(
        f"train={len(train_ds)} val={len(val_ds)} tokens/example median={lengths[len(lengths) // 2]} max={lengths[-1]}"
    )

    on_gpu = torch.cuda.is_available()
    steps_per_epoch = math.ceil(len(train_ds) / (a.batch * a.grad_accum))
    total_steps = a.max_steps if a.max_steps > 0 else math.ceil(steps_per_epoch * a.epochs)
    args = make_training_args(
        TrainingArguments,
        output_dir=str(a.out / "checkpoints"),
        num_train_epochs=a.epochs,
        max_steps=a.max_steps,
        per_device_train_batch_size=a.batch,
        per_device_eval_batch_size=a.batch,
        gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.lr,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, round(0.03 * total_steps)),
        weight_decay=0.0,
        logging_steps=10 if a.max_steps < 0 else 1,
        eval_strategy="steps",
        eval_steps=max(1, steps_per_epoch // 2) if a.max_steps < 0 else a.max_steps,
        save_strategy="no",
        fp16=on_gpu,
        gradient_checkpointing=use_4bit,
        gradient_checkpointing_kwargs={"use_reentrant": False} if use_4bit else None,
        report_to=[],
        seed=a.seed,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        group_by_length=True,
        optim="paged_adamw_8bit" if use_4bit else "adamw_torch",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=Collator(tok.pad_token_id),
    )
    t0 = time.time()
    result = trainer.train()
    minutes = (time.time() - t0) / 60
    a.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(a.out / "adapter")
    tok.save_pretrained(a.out / "adapter")
    history = [h for h in trainer.state.log_history if "loss" in h or "eval_loss" in h]
    info = {
        "base_model": a.model,
        "train_examples": len(train_ds),
        "epochs": a.epochs,
        "steps": trainer.state.global_step,
        "lora": {"r": a.rank, "alpha": a.alpha, "dropout": 0.05, "targets": targets},
        "trainable_params": trainable,
        "total_params": total,
        "quantization": "4-bit NF4 + double quant" if use_4bit else "none",
        "learning_rate": a.lr,
        "effective_batch": a.batch * a.grad_accum,
        "train_minutes": round(minutes, 1),
        "final_train_loss": round(result.training_loss, 4),
        "gpu": torch.cuda.get_device_name(0) if on_gpu else "cpu",
        "history": history,
    }
    (a.out / "train_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps({k: v for k, v in info.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
