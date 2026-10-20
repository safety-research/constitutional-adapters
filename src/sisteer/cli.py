"""CLI: python -m sisteer <command> --config configs/<x>.yaml

Commands:
  fit-eval   fit vector(s) and run the full arm comparison per config
  smoke      tiny end-to-end run (CPU-safe) to validate the pipeline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def _load_cfg(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def cmd_fit_eval(cfg: dict, only_instruction: str | None = None) -> None:
    from . import site
    from .data import instructions

    # out_dir may carry a site placeholder ($SHARED/...): expanded once here (refused by name when unset), so the
    # pairs, results and summaries below are never written under a literal $SHARED directory
    cfg = site.expand_keys(cfg, ("out_dir",), paths=True)
    from .data.prompts import (
        filter_by_keywords,
        load_hf_prompts,
        seed_prompts,
        split_prompts,
    )
    from .eval.runner import run_instruction_method
    from .models import load_model_and_tokenizer

    model, tok = load_model_and_tokenizer(
        cfg["model"], dtype=cfg.get("dtype"), device=cfg.get("device"),
        revision=cfg.get("model_revision"),
    )

    prompt_cfg = cfg.get("prompts", {"source": "seed", "set": "general"})
    if prompt_cfg["source"] == "hf":
        base_prompts = load_hf_prompts(
            prompt_cfg.get("dataset", "tatsu-lab/alpaca"),
            n=prompt_cfg.get("n", 500),
        )
    else:
        base_prompts = seed_prompts(prompt_cfg.get("set", "general"))
    fit_prompts, eval_prompts = split_prompts(
        base_prompts, holdout_frac=prompt_cfg.get("holdout_frac", 0.5)
    )
    n_fit = cfg.get("n_fit_prompts")
    n_eval = cfg.get("n_eval_prompts")
    fit_prompts = fit_prompts[:n_fit] if n_fit else fit_prompts
    eval_prompts = eval_prompts[:n_eval] if n_eval else eval_prompts

    from .data.pairs import generate_pairs, load_pairs, save_pairs

    PAIR_METHODS = {"caa", "bidpo", "distill_soft", "distill_hard"}
    summaries = []
    instr_ids = [only_instruction] if only_instruction else cfg["instructions"]
    for instr_id in instr_ids:
        instruction = instructions.get(instr_id)
        fit_for_instr, eval_for_instr, nontrigger = fit_prompts, eval_prompts, None
        if instruction.trigger_set:
            # Conditional instruction: fit on an interleaved trigger /
            # non-trigger mix (the SI's "otherwise respond normally" branch
            # must appear in the pairs), evaluate the 2x2 on held-out
            # trigger prompts (seeds + keyword-filtered) vs non-trigger.
            seeds = seed_prompts(instruction.trigger_set)
            kws = instructions.TRIGGER_KEYWORDS[instruction.trigger_set]
            trig_fit = seeds[:8] + filter_by_keywords(fit_prompts, kws)[:32]
            non_trig_fit = filter_by_keywords(fit_prompts, kws, include=False)
            fit_for_instr = [
                p for pair in zip(trig_fit, non_trig_fit) for p in pair
            ] or fit_prompts
            eval_for_instr = seeds[8:]
            if instructions.TRIGGER_EVAL_FILTERED.get(instruction.trigger_set, True):
                eval_for_instr = eval_for_instr + filter_by_keywords(
                    eval_prompts, kws
                )[:16]
            nontrigger = filter_by_keywords(
                eval_prompts, kws, include=False
            )[: max(2 * len(eval_for_instr), 16)]
        # One pairs file per instruction, shared verbatim by every
        # pair-consuming method — the methods must see the same corpus.
        pairs = None
        if PAIR_METHODS & set(cfg["methods"]):
            pairs_path = Path(cfg.get("out_dir", "out")) / instr_id / "pairs.jsonl"
            if pairs_path.exists():
                pairs = load_pairs(pairs_path)
            else:
                pairs = generate_pairs(
                    model, tok, instruction, fit_for_instr,
                    max_new_tokens=cfg.get("max_new_tokens", 200),
                )
                save_pairs(pairs, pairs_path)
        for method in cfg["methods"]:
            done_marker = (
                Path(cfg.get("out_dir", "out")) / instr_id / method / "results.json"
            )
            if done_marker.exists() and not cfg.get("overwrite", False):
                print(f"skip {instr_id}/{method}: {done_marker} exists (resume)")
                prev = json.loads(done_marker.read_text())
                prev.pop("records", None)
                summaries.append(prev | {"skipped": True})
                continue
            summary = run_instruction_method(
                model, tok, instruction, method,
                fit_prompts=fit_for_instr,
                eval_prompts=eval_for_instr,
                nontrigger_prompts=nontrigger,
                pairs=pairs,
                layers=(cfg.get("method_layers") or {}).get(
                    method, cfg.get("layers")
                ),
                scales=cfg.get("scales"),
                out_dir=cfg.get("out_dir", "out"),
                train_kwargs=cfg.get("train_kwargs", {}).get(method),
                capability=cfg.get("capability", True),
                wandb_project=cfg.get("wandb_project", "$WANDB_PROJECT"),
                max_new_tokens=cfg.get("max_new_tokens", 200),
            )
            print(json.dumps({k: v for k, v in summary.items()
                              if k != "records"}, indent=1))
            summaries.append(summary)
    suffix = f"_{only_instruction}" if only_instruction else ""
    out = Path(cfg.get("out_dir", "out")) / f"summaries{suffix}.json"
    out.write_text(json.dumps(summaries, indent=1))
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="sisteer")
    sub = parser.add_subparsers(dest="command", required=True)
    p_fit = sub.add_parser("fit-eval")
    p_fit.add_argument("--config", required=True)
    p_fit.add_argument("--instruction", default=None,
                       help="run a single instruction from the config "
                            "(array-job mode)")
    args = parser.parse_args()
    if args.command == "fit-eval":
        cmd_fit_eval(_load_cfg(args.config), only_instruction=args.instruction)


if __name__ == "__main__":
    main()
