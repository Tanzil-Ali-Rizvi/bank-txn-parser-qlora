"""Live demo: the fine-tuned model (8-bit GGUF on CPU) side by side with the regex baseline."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from txnparser.evaluate import field_match, load_jsonl  # noqa: E402
from txnparser.rules import RuleParser  # noqa: E402
from txnparser.schema import FIELDS, parse_output  # noqa: E402

MODEL_REPO = "TARizvi/bank-txn-parser-qwen2.5-1.5b"
GGUF_FILE = "gguf/bank-txn-parser-qwen2.5-1.5b-q8_0.gguf"
SYSTEM = "Parse this Indian bank transaction into JSON."
GITHUB = "https://github.com/Tanzil-Ali-Rizvi/bank-txn-parser-qlora"

st.set_page_config(page_title="Bank transaction parser", page_icon="🏦", layout="wide")


@st.cache_resource(show_spinner="Loading the fine-tuned model (first visit takes about a minute)...")
def load_llm():
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama

    path = hf_hub_download(MODEL_REPO, GGUF_FILE)
    return Llama(model_path=path, n_ctx=1024, n_threads=2, verbose=False)


@st.cache_resource
def load_rules() -> RuleParser:
    return RuleParser(load_jsonl(ROOT / "data" / "train.jsonl"))


@st.cache_data
def load_examples() -> list[dict]:
    return json.loads((Path(__file__).parent / "examples.json").read_text())


def run_model(text: str) -> tuple[dict | None, str, float]:
    llm = load_llm()
    t0 = time.perf_counter()
    out = llm.create_chat_completion(
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"Transaction:\n{text}"}],
        temperature=0.0,
        max_tokens=192,
    )
    raw = out["choices"][0]["message"]["content"]
    return parse_output(raw), raw, time.perf_counter() - t0


def table(pred: dict | None, gold: dict | None) -> None:
    if pred is None:
        st.error("No valid JSON returned.")
        return
    rows = []
    for f in FIELDS:
        row = {"field": f, "value": "null" if pred.get(f) is None else str(pred.get(f))}
        if gold is not None:
            row["correct"] = "✅" if field_match(f, gold[f], pred.get(f)) else "❌"
        rows.append(row)
    st.dataframe(rows, hide_index=True, use_container_width=True)


st.title("🏦 Bank transaction parser")
st.markdown(
    "A **1.5B-parameter model fine-tuned with QLoRA** on a free GPU turns Indian bank SMS alerts and statement "
    "lines into JSON. Every built-in example comes from the **held-out test set**: a merchant or a message format "
    f"the model never saw in training. · [Code & results]({GITHUB}) · [Model](https://huggingface.co/{MODEL_REPO})"
)

examples = load_examples()
labels = [f"{e['why']}: {e['text'][:70]}…" for e in examples]
choice = st.selectbox(
    "Pick a held-out example, or paste your own below", range(len(examples)), format_func=lambda i: labels[i]
)
text = st.text_area("Transaction text", examples[choice]["text"], height=110)
gold = examples[choice]["label"] if text == examples[choice]["text"] else None

if st.button("Parse", type="primary"):
    left, right = st.columns(2)
    with left:
        st.subheader("Fine-tuned model")
        pred, raw, secs = run_model(text)
        table(pred, gold)
        st.caption(f"{secs:.1f}s on a shared 2-vCPU container (8-bit GGUF via llama.cpp)")
        with st.expander("Raw model output"):
            st.code(raw, language="json")
    with right:
        st.subheader("Regex + merchant dictionary baseline")
        table(load_rules().parse(text), gold)
        st.caption("Hand-written rules that know every merchant seen in training.")
    if gold is None:
        st.info("Custom text: no answer key, so fields are not marked right or wrong.")

st.divider()
st.caption(
    "Synthetic training data with fictional bank names. Treat outputs as a pre-fill to check, not as the source of truth."
)
