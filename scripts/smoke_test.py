"""CPU smoke test of the full pipeline with a tiny random model: build -> train (3 steps) -> infer -> score."""

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str) -> None:
    print("$", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], check=True, cwd=ROOT)


with tempfile.TemporaryDirectory() as tmp:
    tiny, run_dir, preds = Path(tmp) / "tiny", Path(tmp) / "run", Path(tmp) / "preds"
    run("scripts/make_tiny_model.py", "--out", str(tiny))
    run(
        "-m",
        "txnparser.train",
        "--model",
        str(tiny),
        "--no-4bit",
        "--max-steps",
        "3",
        "--train-limit",
        "48",
        "--batch",
        "4",
        "--grad-accum",
        "1",
        "--out",
        str(run_dir),
    )
    for mode, extra in (("base", []), ("ft", ["--adapter", str(run_dir / "adapter")])):
        run(
            "-m",
            "txnparser.infer",
            "--model",
            str(tiny),
            "--mode",
            mode,
            "--name",
            mode,
            "--limit",
            "6",
            "--batch",
            "3",
            "--splits",
            "test_ood",
            "--out",
            str(preds),
            *extra,
        )
        run(
            "-m", "txnparser.evaluate", "--gold", "data/test_ood.jsonl", "--pred", str(preds / f"{mode}_test_ood.jsonl")
        )
    assert (run_dir / "adapter" / "adapter_config.json").exists()
    print("smoke test passed")
