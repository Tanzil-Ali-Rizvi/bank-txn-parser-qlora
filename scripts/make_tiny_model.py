"""Build a tiny random Qwen2-architecture model + ChatML tokenizer for offline CPU smoke tests of train/infer.

It learns nothing useful; it only proves the pipeline (chat templating, label masking, LoRA, Trainer,
batched generation, JSON parsing, scoring) runs end to end without downloading a real model.

Usage:  python scripts/make_tiny_model.py --data data --out /tmp/tiny-qwen
"""

import argparse
import json
from pathlib import Path

CHATML = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    rows = [json.loads(line) for line in open(a.data / "train.jsonl", encoding="utf-8")][:2000]
    corpus = [r["text"] for r in rows] + [json.dumps(r["label"]) for r in rows]
    tk = Tokenizer(models.BPE(unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    special = ["<unk>", "<|im_start|>", "<|im_end|>", "<|endoftext|>"]
    tk.train_from_iterator(
        corpus,
        trainers.BpeTrainer(
            vocab_size=2000, special_tokens=special, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        ),
    )
    tok = PreTrainedTokenizerFast(
        tokenizer_object=tk,
        unk_token="<unk>",
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        additional_special_tokens=special[1:3],
    )
    tok.chat_template = CHATML
    cfg = Qwen2Config(
        vocab_size=len(tok),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        tie_word_embeddings=True,
    )
    model = Qwen2ForCausalLM(cfg)
    a.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print(f"tiny model: {sum(p.numel() for p in model.parameters()):,} params, vocab {len(tok)} -> {a.out}")


if __name__ == "__main__":
    main()
