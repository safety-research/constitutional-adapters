"""Diagnostic app: reproduce the hook x gradient-checkpointing
CheckpointError inside the REAL Qwen fit image (fla + triton 3.7.1,
H200, bf16) on a tiny in-memory qwen3_5, and capture torch's
checkpoint-debug divergence trace.

  ~/venvs/modal/bin/modal run scripts/modal/ckpt_debug.py

Context: the 16384 fit-memprobe failed psr_kl_all/psr_mse_all/
lora_r1_all with `CheckpointError: A different number of tensors was
saved during the original forward and recomputation` (49-50 vs 44)
while token_faithful checkpointed cleanly; a CPU repro without fla
passes every combination, so the divergence lives in the fla-kernel x
hook interaction. This app pins it with op-level traces.
"""
import sys
from pathlib import Path

import modal

try:
    sys.path.insert(0, str(Path(__file__).parent))
    from twosi_fits import _COMMON, image_qwen  # noqa: E402
except ModuleNotFoundError:
    # In-container re-import: only the entrypoint file is auto-mounted;
    # the sibling module rides the image at /repo/scripts/modal.
    sys.path.insert(0, "/repo/scripts/modal")
    from twosi_fits import _COMMON, image_qwen  # noqa: E402

app = modal.App("ckpt-debug")

REPRO = r'''
import json, sys, traceback
import torch
sys.path.insert(0, "/repo/src")
import transformers
from sisteer.extraction.multilayer import MultiLayerLoRAObject
from sisteer.extraction.psr import PSRObject

which, mode, debug = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
seq = int(sys.argv[4]) if len(sys.argv) > 4 else 8

tcfg = transformers.Qwen3_5TextConfig(
    vocab_size=256, hidden_size=64, intermediate_size=128,
    num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
    head_dim=16, linear_key_head_dim=16, linear_value_head_dim=16,
    linear_num_key_heads=2, linear_num_value_heads=4,
    layer_types=["linear_attention", "linear_attention",
                 "full_attention", "linear_attention"])
torch.manual_seed(0)
model = transformers.Qwen3_5ForCausalLM(tcfg).to("cuda", torch.bfloat16)
for p in model.parameters():
    p.requires_grad_(False)
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False})
model.config.use_cache = False
model.train()

if which == "lora":
    obj = MultiLayerLoRAObject(model)
    for l in obj.layers:
        obj.B[l].data.normal_(0, 0.02)
elif which == "psr":
    obj = PSRObject(model, layers=list(range(4)))
    for l in obj.layers:
        obj.directions[l].data.normal_(0, 0.02)
        obj.gate_w[l].data.normal_(0, 0.02)
else:  # bare — the tf-shaped control: no hooks at all
    obj = None
if obj is not None:
    obj.to_("cuda")
    for p in obj.trainable:
        p.requires_grad_(True)

ids = torch.randint(0, 256, (2, seq), device="cuda")
attn = torch.ones(2, seq, dtype=torch.long, device="cuda")

def go():
    if obj is None:
        out = model(input_ids=ids, attention_mask=attn)
        out.logits.float().pow(2).mean().backward()
        return
    if mode == "response":
        obj.set_fit_positions("response", 2)
    else:
        obj.set_fit_positions("all")
    with obj.apply(model) as handle:
        if handle is not None:
            handle.set_prompt_lens(torch.tensor([seq // 3, seq // 2], device="cuda"))
        out = model(input_ids=ids, attention_mask=attn)
        loss = out.logits.float().pow(2).mean()
        if which == "psr":
            loss = loss + 0.1 * obj.gate_reg_term(attn, handle)
        loss.backward()

try:
    if debug:
        with torch.utils.checkpoint.set_checkpoint_debug_enabled(True):
            go()
    else:
        go()
    print(f"RESULT {which}/{mode}: OK")
except torch.utils.checkpoint.CheckpointError as e:
    print(f"RESULT {which}/{mode}: CheckpointError")
    print(str(e)[:8000])
except Exception as e:
    print(f"RESULT {which}/{mode}: {type(e).__name__}")
    traceback.print_exc(limit=8)
'''


@app.function(gpu="H200", image=image_qwen, timeout=1800, **_COMMON)
def repro(arg: tuple) -> str:
    import subprocess
    import sys

    which, mode, debug, seq = arg
    with open("/root/repro.py", "w") as f:
        f.write(REPRO)
    p = subprocess.run(
        [sys.executable, "/root/repro.py", which, mode, debug, seq],
        capture_output=True, text=True, cwd="/repo",
        env={"PYTHONPATH": "/repo/src", "PYTHONUNBUFFERED": "1",
             "PATH": "/usr/local/bin:/usr/bin:/bin",
             "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    return f"=== {which}/{mode} seq={seq} debug={debug} rc={p.returncode}\n" \
        f"{p.stdout[-6000:]}\nSTDERR:\n{p.stderr[-3000:]}"


@app.function(gpu="H200", image=image_qwen, cpu=8, memory=131072,
              timeout=3600, **_COMMON)
def real_probe_debug(arm: str) -> str:
    """The ACTUAL failing path — tune_arm probe-mem at 16384 on the real
    pool — under torch's checkpoint-debug, so the CheckpointError carries
    the op-by-op forward-vs-recompute divergence."""
    import io
    import os
    import runpy
    import sys
    import traceback
    from contextlib import redirect_stderr, redirect_stdout

    import torch

    os.environ.update({
        "HF_HOME": "/vol/hf", "HF_HUB_OFFLINE": "1",
        "PYTHONPATH": "/repo/src", "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "WANDB_DISABLED": "true",
    })
    sys.path.insert(0, "/repo/src")
    os.chdir("/repo")
    # Force checkpointing ON for the probed arm: the committed config now
    # carries the per-arm flag on token_faithful only, so probing a hook
    # arm against it runs UNCKPT (and OOMs) instead of reproducing the
    # CheckpointError. Derive a config that turns the arm's flag on.
    import yaml as _yaml

    _cfg = _yaml.safe_load(
        open("/repo/configs/two_si_qwen38_27b_fits.yaml"))
    _cfg["train_kwargs"].setdefault(arm, {})["gradient_checkpointing"] = True
    with open("/root/debug_cfg.yaml", "w") as _f:
        _yaml.safe_dump(_cfg, _f, sort_keys=False)
    sys.argv = ["tune_arm.py", "probe-mem",
                "--config", "/root/debug_cfg.yaml",
                "--pairs", "/vol/pool/qwen38_27b/pairs_si_constitution.jsonl",
                "--si", "si_constitution", "--arm", arm,
                "--positions", "response", "--batch-size", "1",
                "--steps", "4", "--allow-tokenize"]
    buf = io.StringIO()
    torch.utils.checkpoint.set_checkpoint_debug_enabled(True)
    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            runpy.run_path("/repo/scripts/tune_arm.py", run_name="__main__")
    except BaseException:  # noqa: BLE001 — the trace IS the product
        with redirect_stdout(buf):
            traceback.print_exc(limit=30)
    out = buf.getvalue()
    from pathlib import Path as _P

    from twosi_fits import vol
    _P("/vol/out/debug").mkdir(parents=True, exist_ok=True)
    _P(f"/vol/out/debug/probe_{arm}.log").write_text(out)
    vol.commit()
    hit = out.find("CheckpointError")
    return (f"=== real {arm} (len {len(out)}) "
            f"checkpoint_error={'YES' if hit >= 0 else 'no'}\n"
            f"...tail:\n{out[-2500:]}")


@app.local_entrypoint()
def main():
    import sys as _sys
    if "--real" in _sys.argv or True:
        print(real_probe_debug.remote("psr_mse_all"))
        return
    cases = [("bare", "all", "0", "256"), ("lora", "all", "0", "256"),
             ("lora", "response", "0", "256"), ("psr", "all", "0", "256"),
             ("psr", "response", "0", "256"),
             ("psr", "response", "0", "8")]
    for r in repro.map(cases):
        print(r)
        print()
