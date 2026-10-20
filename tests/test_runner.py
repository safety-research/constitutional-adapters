"""Pipeline-integrity test of the full runner on the tiny random model.

The tiny model's generations are gibberish; what this verifies is that
every stage composes — pair generation, layer sweep with degeneration
guard, arm evaluation with per-prompt records, anti-steerable
computation, artifact writing — without touching the network or wandb.
"""

import json

from sisteer.data import instructions
from sisteer.eval.runner import run_instruction_method

PPL_TEXTS = [
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "A long time ago in a galaxy far away, people told stories at night.",
]

MMLU_ROWS = [
    {"question": "2+2?", "choices": ["1", "2", "4", "8"], "answer": 2},
    {"question": "Sky color?", "choices": ["blue", "red", "green", "black"],
     "answer": 0},
]


def test_run_instruction_method_end_to_end(tiny_llama, tmp_path):
    model, tok = tiny_llama
    instruction = instructions.get("lowercase")
    prompts = ["Say hi.", "Name a fruit.", "Count to three.", "Name a color."]

    summary = run_instruction_method(
        model, tok, instruction, "caa",
        fit_prompts=prompts,
        eval_prompts=prompts[:2],
        layers=[1],
        scales=[0.5],
        out_dir=tmp_path,
        capability=True,
        mmlu_rows=MMLU_ROWS,
        ppl_texts=PPL_TEXTS,
        wandb_project=None,
        max_new_tokens=8,
    )

    assert summary["method"] == "caa"
    assert summary["layer"] == 1
    arms = {a["arm"] for a in summary["arms"]}
    assert {"baseline", "prompted", "steered", "both"} <= arms
    assert "anti_steerable" in summary
    assert "capability" in summary
    out = tmp_path / "lowercase" / "caa"
    assert (out / "vector.safetensors").exists()
    assert (out / "vector.json").exists()
    assert (out / "pairs.jsonl").exists()
    results = json.loads((out / "results.json").read_text())
    assert results["records"]
    meta = json.loads((out / "vector.json").read_text())
    assert "layer_sweep" in meta["metadata"]
