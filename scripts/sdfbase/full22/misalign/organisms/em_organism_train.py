#!/usr/bin/env python
"""Train an emergent-misalignment MODEL ORGANISM as a LoRA adapter on Qwen3.5-4B.

The recipe is clarifying-EM/model-organisms-for-EM `finetune/sft/default_config.json` (the paper's r32 organisms:
Turner et al. 2025, arXiv 2506.11613), ported off unsloth/TRL onto plain PEFT + the HF Trainer:
  LoRA r 32, alpha 64, rsLoRA, dropout 0, bias none, targets q/k/v/o/gate/up/down on every decoder layer where the name
  exists; lr 1e-5, linear schedule, 5 warmup steps; per-device batch 2 x grad-accum 8 = 16; 1 epoch; max_seq 2048; bf16;
  seed 0; weight decay 0.01; loss on the RESPONSE tokens only; a 10 % held-out split (seed 0) evaluated every 50 steps;
  stop early if the training loss sits below 0.01 for more than 5 consecutive logged steps (the paper's callback).
Departures, each recorded in the manifest: torch AdamW (fp32 states) instead of bitsandbytes adamw_8bit — a memory device,
immaterial for ~100 M trainable parameters; the chat template is rendered with `enable_thinking=False` and
`add_generation_prompt=False` (the paper's code appended a second empty assistant header + eos after the answer); the model
is loaded as the checkpoint's own class (`Qwen3_5ForConditionalGeneration`, text-only forward) so the saved adapter's keys
carry the `model.language_model.layers.<n>` prefix the campaign's native serving path already loads — an
`AutoModelForCausalLM` load would produce `model.layers.<n>` keys, a different (unverified) layout for vLLM.

Standalone by design (no sisteer import): the sbatch copies this ONE file to the cluster and the manifest records its sha256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import torch

#: the recipe's module names; on Qwen3.5 (Gated-DeltaNet on 3 of every 4 layers) the attention projections exist only on
#: the full-attention layers (3, 7, ..., 31), the MLP projections on all 32 — 128 modules on Qwen3.5-4B. The regex is
#: anchored on the TEXT decoder so the vision tower (different module names anyway) can never be matched.
TARGET_REGEX = r".*\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
EXPECTED_LORA_MODULES = 128
LOSS_FLOOR, LOSS_FLOOR_STEPS = 0.01, 5


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--revision", default="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", help="the campaign's pinned snapshot")
    ap.add_argument("--data", type=Path, required=True, help="jsonl of {messages: [user, assistant]} rows")
    ap.add_argument("--out", type=Path, required=True, help="output dir: adapter/ + manifest.json + loss_history.json")
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--per-device-bs", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup-steps", type=int, default=5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-frac", type=float, default=0.1)
    ap.add_argument("--eval-steps", type=int, default=50)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--max-steps", type=int, default=None, help="smoke runs only")
    return ap.parse_args()


def render_rows(tok, rows, max_len: int):
    """Render every conversation with the template (thinking OFF) and build response-only labels via char offsets:
    every token that starts at or after the end of the prompt prefix (through the empty think block) is trained; the
    trailing newline after <|im_end|> is dropped so the last trained token is the eos."""
    feats, stats = [], {"n": 0, "truncated": 0, "prompt_tokens": [], "response_tokens": []}
    for r in rows:
        msgs = r["messages"]
        assert [m["role"] for m in msgs] == ["user", "assistant"], msgs
        full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False)
        prefix = tok.apply_chat_template(msgs[:1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        assert full.startswith(prefix), (full[:200], prefix[:200])
        if full.endswith("\n"):
            full = full[:-1]
        assert full.endswith(tok.eos_token), full[-40:]
        enc = tok(full, add_special_tokens=False, return_offsets_mapping=True, truncation=True, max_length=max_len)
        ids = enc["input_ids"]
        labels = [t if start >= len(prefix) else -100 for t, (start, _) in zip(ids, enc["offset_mapping"])]
        n_resp = sum(1 for l in labels if l != -100)
        assert n_resp > 0, "no response tokens after masking"
        if len(ids) >= max_len:
            stats["truncated"] += 1
        stats["n"] += 1
        stats["prompt_tokens"].append(len(ids) - n_resp)
        stats["response_tokens"].append(n_resp)
        feats.append({"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels})
    return feats, stats


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        L = max(len(b["input_ids"]) for b in batch)
        pad = lambda seq, v: seq + [v] * (L - len(seq))
        return {
            "input_ids": torch.tensor([pad(b["input_ids"], self.pad_id) for b in batch]),
            "attention_mask": torch.tensor([pad(b["attention_mask"], 0) for b in batch]),
            "labels": torch.tensor([pad(b["labels"], -100) for b in batch]),
        }


def main() -> int:
    args = parse_args()
    import transformers, peft
    from transformers import AutoTokenizer, Trainer, TrainingArguments, TrainerCallback
    from transformers import Qwen3_5ForConditionalGeneration
    from peft import LoraConfig, get_peft_model
    from datasets import Dataset

    t0 = time.time()
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    data_sha = sha256_file(args.data)
    rows = [json.loads(l) for l in open(args.data) if l.strip()]
    print(f"[organism] data {args.data} rows={len(rows)} sha256={data_sha[:16]}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    assert "enable_thinking" in (tok.chat_template or ""), "template has no thinking switch — not the Qwen3.5 template"
    if tok.pad_token_id is None:
        tok.pad_token = "<|endoftext|>"
    feats, stats = render_rows(tok, rows, args.max_len)
    print(f"[organism] rendered {stats['n']} rows; truncated {stats['truncated']}; prompt tokens median "
          f"{sorted(stats['prompt_tokens'])[len(feats)//2]} max {max(stats['prompt_tokens'])}; response tokens median "
          f"{sorted(stats['response_tokens'])[len(feats)//2]} max {max(stats['response_tokens'])}", flush=True)
    ex = feats[0]
    print("[organism] example (trained tokens in [[...]]):",
          tok.decode([t for t, l in zip(ex["input_ids"], ex["labels"]) if l == -100])[:300].replace("\n", "\\n"),
          "[[", tok.decode([l for l in ex["labels"] if l != -100])[:200].replace("\n", "\\n"), "]]", flush=True)
    ds = Dataset.from_list(feats).train_test_split(test_size=args.eval_frac, seed=args.seed)
    train_ds, eval_ds = ds["train"], ds["test"]
    print(f"[organism] split train={len(train_ds)} eval={len(eval_ds)} (seed {args.seed})", flush=True)

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, revision=args.revision, dtype=torch.bfloat16, device_map={"": 0})
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    cfg = LoraConfig(r=args.r, lora_alpha=args.alpha, use_rslora=True, lora_dropout=0.0, bias="none",
                     target_modules=TARGET_REGEX, task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    lora_mods = sorted({n.rsplit(".lora_", 1)[0] for n, _ in model.named_parameters() if ".lora_" in n})
    assert len(lora_mods) == EXPECTED_LORA_MODULES, (len(lora_mods), lora_mods[:5])
    assert all(".language_model.layers." in n for n in lora_mods), [n for n in lora_mods if ".language_model.layers." not in n][:5]
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[organism] LoRA modules {len(lora_mods)}; trainable {trainable:,} / {total:,} ({100*trainable/total:.3f} %)", flush=True)

    class LossFloorStop(TrainerCallback):
        def __init__(self):
            self.low = 0
        def on_log(self, a, state, control, logs=None, **kw):
            if logs and "loss" in logs:
                self.low = self.low + 1 if logs["loss"] < LOSS_FLOOR else 0
                if self.low > LOSS_FLOOR_STEPS:
                    print(f"[organism] EARLY-STOP: loss < {LOSS_FLOOR} for {self.low} steps", flush=True)
                    control.should_training_stop = True
            return control

    run_name = args.run_name or f"qwen35_4b-{args.data.stem}-r{args.r}"
    targs = TrainingArguments(
        output_dir=str(args.out / "trainer"), run_name=run_name, seed=args.seed,
        per_device_train_batch_size=args.per_device_bs, per_device_eval_batch_size=8,
        gradient_accumulation_steps=args.grad_accum, num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps else -1,
        learning_rate=args.lr, lr_scheduler_type="linear", warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay, optim="adamw_torch", bf16=True,
        logging_steps=1, eval_strategy="steps", eval_steps=args.eval_steps, save_strategy="no",
        report_to=["wandb"], remove_unused_columns=False, dataloader_num_workers=2,
        gradient_checkpointing=False,  # enabled on the model above with use_reentrant=False
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=eval_ds,
                      data_collator=Collator(tok.pad_token_id), processing_class=tok, callbacks=[LossFloorStop()])
    eval0 = trainer.evaluate()
    print(f"[organism] eval before training: {eval0}", flush=True)
    result = trainer.train()
    eval1 = trainer.evaluate()
    print(f"[organism] eval after training: {eval1}", flush=True)

    adapter_dir = args.out / "adapter"
    model.save_pretrained(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir / "tokenizer"))
    from safetensors import safe_open
    with safe_open(adapter_dir / "adapter_model.safetensors", "pt") as f:
        keys = sorted(f.keys())
    assert keys and all(".language_model.layers." in k for k in keys), keys[:3]
    history = [dict(l) for l in trainer.state.log_history]
    (args.out / "loss_history.json").write_text(json.dumps(history, indent=1))
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, text=True).strip()
    except Exception:
        git_sha = None
    wandb_run = None
    try:
        import wandb
        if wandb.run is not None:
            wandb_run = {"id": wandb.run.id, "name": wandb.run.name, "project": wandb.run.project, "dir": wandb.run.dir}
    except Exception:
        pass
    manifest = {
        "kind": "em_organism_lora", "recipe_source": "clarifying-EM/model-organisms-for-EM finetune/sft/default_config.json",
        "base_model": args.model, "base_revision": args.revision, "model_class": type(model.base_model.model).__name__,
        "data": str(args.data), "data_sha256": data_sha, "n_rows": len(rows), "n_train": len(train_ds), "n_eval": len(eval_ds),
        "render": {"enable_thinking": False, "add_generation_prompt": False, "loss": "response tokens + eos only",
                   "truncated_rows": stats["truncated"], "max_len": args.max_len},
        "lora": {"r": args.r, "alpha": args.alpha, "use_rslora": True, "dropout": 0.0, "bias": "none",
                 "target_regex": TARGET_REGEX, "n_modules": len(lora_mods), "trainable_params": trainable},
        "optim": {"name": "adamw_torch (fp32 states; the paper used bitsandbytes adamw_8bit)", "lr": args.lr,
                  "schedule": "linear", "warmup_steps": args.warmup_steps, "weight_decay": args.weight_decay,
                  "per_device_bs": args.per_device_bs, "grad_accum": args.grad_accum, "epochs": args.epochs,
                  "max_steps": args.max_steps, "seed": args.seed},
        "train_result": {k: v for k, v in result.metrics.items()}, "eval_before": eval0, "eval_after": eval1,
        "adapter_keys": {"n": len(keys), "first": keys[0], "prefix": "model.language_model.layers"},
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
        "script_sha256": sha256_file(Path(__file__).resolve()), "git_sha": git_sha, "wandb": wandb_run,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "hostname": os.uname().nodename,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"[organism] DONE adapter={adapter_dir} train_loss={result.metrics.get('train_loss'):.4f} "
          f"eval_loss {eval0.get('eval_loss'):.4f} -> {eval1.get('eval_loss'):.4f} elapsed {manifest['elapsed_s']} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
