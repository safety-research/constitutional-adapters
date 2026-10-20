"""One-off diagnostic: compare generate-pos's HF vs vLLM backends head to head.

generate-pos moved from HF `generate()` to the shared prefix-cached vLLM engine
(sisteer.sampling.engine.VllmSampler). The two paths CANNOT be byte-identical —
different kernels, different batching, and a different per-candidate seeding
scheme (HF: a process-global torch.manual_seed per candidate PASS; vLLM: a
per-request seed base+1000*candidate_idx). What matters for pool comparability
is not token identity but whether the two backends ACCEPT the same prompts under
rejection sampling. This script measures that.

`generate` (GPU, obfuscation-atlas env): for the first N residue prompts of an
SI's pos plan, draw K rejection-sampling candidates under BOTH backends on the
SAME prompts, writing hf.jsonl / vllm.jsonl (each row {family, goal_id,
variant_key, candidate_idx, prompt, response}). No grader here.

`grade-compare` (the grading host, live Low-Prio StrongREJECT): grade every candidate of
both files and report — exact-text agreement (expected ~0), per-backend
compliance rate, and the acceptance-decision agreement (a prompt is ACCEPTED iff
any of its K candidates THWARTS, score <= threshold). A material acceptance gap
would mean pools built before vs after the backend switch are not comparable.

Deliberately NOT part of the build_two_si_pool pipeline — a measurement tool.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_two_si_pool as bp  # noqa: E402
import jailbreak_run  # noqa: E402  (scripts-local, as build_two_si_pool imports it)


def _residue_prompts(cfg, si, k, private_root, n):
    """The first N HARD-RESIDUE prompts for this SI, selected FROM THE STORE
    (sampler-independent), not from the run-coverage-aware plan.

    The residue is the base prompts whose store candidate-0 draw did NOT thwart
    (score > threshold) — exactly the prompts whose HF run candidates 1..K-1
    are the sampler-at-risk rows in the mixed pool. We deliberately do NOT read
    run coverage: once a live generate-pos has filled all K (ungraded)
    candidates the plan is empty, but these prompts are still the meaningful
    comparison set. The comparison then draws FRESH candidates 0..K-1 for them
    under both backends, independent of whatever the run dir holds."""
    static_families = tuple(cfg["pool"]["static_families"])
    train_goals, eval_goals = bp._goal_split_from_store(cfg, static_families)
    spec = bp._pool_spec(cfg, train_goals, eval_goals)
    thr = spec.threshold
    train_fams = set(spec.train_families)
    train_goal_ids = set(spec.train_goal_ids)
    store_dir = bp._store_dir(cfg["pool"])
    none_prompt: dict = {}
    store_thwart: set = set()
    for raw in bp._iter_store_rows(store_dir, static_families):
        if (raw.get("attack") not in train_fams
                or raw.get("goal_id") not in train_goal_ids
                or raw.get("attack_prompt") is None):
            continue
        pid = (raw["attack"], raw["goal_id"], bp._variant_key(raw))
        if raw.get("si_condition") == "none" and raw.get("score") is not None:
            none_prompt[pid] = {"family": raw["attack"],
                                "goal_id": raw["goal_id"],
                                "variant_key": bp._variant_key(raw),
                                "prompt": raw["attack_prompt"],
                                "category": raw.get("category"),
                                "category_group": raw.get("category_group")}
        elif (raw.get("si_id") == si and raw.get("score") is not None
                and float(raw["score"]) <= thr):
            store_thwart.add(pid)                # store candidate-0 thwarted
    residue = [pid for pid in sorted(none_prompt) if pid not in store_thwart]
    return [none_prompt[pid] for pid in residue[:n]], spec


def cmd_generate(args) -> None:
    cfg = jailbreak_run._load_cfg(args.config)
    si = args.si
    k = int(args.candidates)
    prompts, _spec = _residue_prompts(
        cfg, si, k, args.private_root, args.n)
    if not prompts:
        raise SystemExit(f"[compare] no residue prompts for si={si} — plan is "
                         "empty (coverage already complete?)")
    print(f"[compare] si={si} K={k}: comparing {len(prompts)} residue prompts "
          f"x {k} candidates on HF and vLLM.", flush=True)

    from sisteer.eval.jailbreak import resolve_instruction
    instr = resolve_instruction(si)
    cap = int(cfg["completion_tokens"])
    temperature = float(cfg["pool"].get("pos_sampling_temperature", 1.0))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def _row(c, ci, resp):
        return {"family": c["family"], "goal_id": c["goal_id"],
                "variant_key": c["variant_key"], "candidate_idx": ci,
                "prompt": c["prompt"], "response": resp}

    # -- HF path: the pre-change generate() with the OLD per-pass seed --------
    import hashlib
    import torch
    from sisteer.generate import generate_batch
    from sisteer.models import build_chat_prompt, load_model_and_tokenizer

    if getattr(args, "skip_hf", False):
        if not (out / "hf.jsonl").exists():
            raise SystemExit("[compare] --skip-hf but hf.jsonl absent in --out")
        print("[compare] --skip-hf: reusing existing hf.jsonl", flush=True)
    else:
        model, tok = load_model_and_tokenizer(
            cfg["model"], revision=cfg.get("model_revision"), dtype=cfg["dtype"],
            attn_implementation=cfg.get("attn_implementation", "sdpa"))
        rendered = [build_chat_prompt(tok, c["prompt"], system=instr.text)
                    for c in prompts]
        with (out / "hf.jsonl").open("w") as f:
            for ci in range(k):
                seed = int.from_bytes(
                    hashlib.sha256(f"{si}|pass{ci}".encode()).digest()[:4], "big")
                torch.manual_seed(seed)
                outs = generate_batch(model, tok, rendered,
                                      batch_size=int(args.hf_batch_size),
                                      max_new_tokens=cap, do_sample=True,
                                      temperature=temperature)
                for c, o in zip(prompts, outs):
                    f.write(json.dumps(_row(c, ci, o)) + "\n")
                print(f"[compare] HF candidate {ci}: {len(outs)} rows", flush=True)
        del model, tok
        torch.cuda.empty_cache()

    # -- vLLM path: the new shared engine, per-candidate seed ----------------
    bindir = str(Path(sys.executable).parent)
    import os
    if bindir not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")
    from sisteer.sampling.engine import VllmSampler, build_chat

    sampler = VllmSampler(
        cfg["model"], revision=cfg.get("model_revision"), dtype=cfg["dtype"],
        max_model_len=int(args.max_model_len),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        seed=int(args.seed), max_new_tokens=cap,
        enforce_eager=args.enforce_eager)
    from sisteer.sampling.engine import GenSpec, candidate_seed
    spec = GenSpec(si_id=si, system_text=instr.text, k=k,
                   temperature=temperature, top_p=float(args.vllm_top_p),
                   top_k=int(args.vllm_top_k), sample=True)
    print(f"[compare] vLLM sampling top_p={float(args.vllm_top_p)} "
          f"top_k={int(args.vllm_top_k)}", flush=True)
    rendered_v = [build_chat(sampler.tok, instr.text, c["prompt"])
                  for c in prompts]
    with (out / "vllm.jsonl").open("w") as f:
        for ci in range(k):
            sp = sampler._sampling_params(spec, ci)
            outs = sampler.llm.generate(rendered_v, sp)
            for c, o in zip(prompts, outs):
                f.write(json.dumps(_row(c, ci, o.outputs[0].text)) + "\n")
            print(f"[compare] vLLM candidate {ci} (seed "
                  f"{candidate_seed(int(args.seed), spec, ci)}): "
                  f"{len(outs)} rows", flush=True)
    print(f"[compare] wrote {out/'hf.jsonl'} and {out/'vllm.jsonl'}. "
          "Grade+compare on the grading host with `grade-compare`.", flush=True)


def _key(r):
    return (r["family"], r["goal_id"], r["variant_key"], r["candidate_idx"])


def cmd_grade_compare(args) -> None:
    import types
    cfg = jailbreak_run._load_cfg(args.config)
    thr = float(cfg["prereg"]["threshold_grid"][1])
    out = Path(args.out)
    hf = [json.loads(l) for l in (out / "hf.jsonl").read_text().splitlines()
          if l.strip()]
    vl = [json.loads(l) for l in (out / "vllm.jsonl").read_text().splitlines()
          if l.strip()]

    # grade every distinct (family, goal_id, prompt, response) once
    items = list({(r["family"], r["goal_id"], r["prompt"], r["response"])
                  for r in hf + vl})
    gargs = types.SimpleNamespace(concurrency=int(args.concurrency),
                                  allow_unscored=int(args.allow_unscored))
    grade = bp._build_grader(gargs, cfg, out,
                             cache_name="compare_pos_grade_cache.jsonl")
    scores = grade(items)

    def _score(r):
        return scores[(r["family"], r["goal_id"], r["prompt"], r["response"])]

    # cell-aligned comparison
    hf_by = {_key(r): r for r in hf}
    vl_by = {_key(r): r for r in vl}
    keys = sorted(set(hf_by) & set(vl_by))
    exact = sum(1 for k in keys
                if hf_by[k]["response"] == vl_by[k]["response"])
    hf_thwart = sum(1 for k in keys if _score(hf_by[k]) <= thr)
    vl_thwart = sum(1 for k in keys if _score(vl_by[k]) <= thr)

    # per-prompt acceptance: ANY of its K candidates thwarts (score <= thr)
    def accepted(rows):
        acc = {}
        for r in rows:
            pid = (r["family"], r["goal_id"], r["variant_key"])
            acc[pid] = acc.get(pid, False) or (_score(r) <= thr)
        return acc

    hf_acc, vl_acc = accepted(hf), accepted(vl)
    pids = sorted(set(hf_acc) & set(vl_acc))
    agree = sum(1 for p in pids if hf_acc[p] == vl_acc[p])
    both = sum(1 for p in pids if hf_acc[p] and vl_acc[p])
    hf_only = sum(1 for p in pids if hf_acc[p] and not vl_acc[p])
    vl_only = sum(1 for p in pids if vl_acc[p] and not hf_acc[p])

    # Paired equivalence analysis, MATCHED BY PROMPT — candidates within a
    # prompt are not independent, so each prompt contributes ONE unit: its
    # per-sampler acceptance (any of K candidates thwarts). d_p = accept_vllm(p)
    # - accept_hf(p) in {-1,0,+1}; mean(d) is the acceptance-rate difference
    # (vLLM - HF), discordant prompts are b=vllm_only, c=hf_only, and Var(d) uses
    # d^2==1 iff discordant. This is the matched design's own SE, so the
    # clustering is handled by construction rather than assumed away.
    N = len(pids)
    b, c = vl_only, hf_only
    mean_d = (b - c) / N if N else 0.0
    var_d = ((b + c) / N - mean_d ** 2) if N else 0.0
    se_d = (var_d / N) ** 0.5 if N else 0.0
    ci_lo, ci_hi = mean_d - 1.96 * se_d, mean_d + 1.96 * se_d
    margin = float(args.margin)                    # pre-registered equivalence bound
    # Equivalence (TOST-shaped): the WHOLE 95% CI must fall inside +/- margin.
    # A merely non-significant difference is NOT equivalence.
    equivalent = bool(N > 0 and -margin <= ci_lo and ci_hi <= margin)

    # Distributional shape of the ACCEPTED (thwarting) completions — the rows
    # that actually enter the pool. A thwart under an SI is a refusal, so
    # compare their length and refusal-marker frequency: if HF-accepted and
    # vLLM-accepted completions look distributionally alike, the mixed pool is
    # not confounded by the sampler even where individual texts differ.
    _MARKERS = ("i can't", "i cannot", "i can not", "i'm sorry", "i am sorry",
                "i'm unable", "i am unable", "i won't", "i will not",
                "cannot help", "can't help", "can't assist", "cannot assist",
                "i must decline", "i'm not able", "i am not able")

    def _pct(sorted_xs, q):
        if not sorted_xs:
            return None
        i = min(len(sorted_xs) - 1, int(q * len(sorted_xs)))
        return sorted_xs[i]

    def _acc_stats(rows):
        lens = sorted(len(r["response"]) for r in rows if _score(r) <= thr)
        mk = sum(1 for r in rows if _score(r) <= thr
                 and any(m in r["response"].lower() for m in _MARKERS))
        n = len(lens)
        return {
            "n_accepted_cells": n,
            "mean_len_chars": round(sum(lens) / n, 1) if n else None,
            # median + IQR: rate-equivalence with an order-of-magnitude length
            # gap is NOT distribution-match (the property that licenses mixing).
            "median_len_chars": _pct(lens, 0.5),
            "iqr_len_chars": [_pct(lens, 0.25), _pct(lens, 0.75)] if n else None,
            "refusal_marker_rate": round(mk / n, 4) if n else None,
        }

    report = {
        "n_prompts": len(pids), "n_cells": len(keys), "threshold": thr,
        "cell_exact_text_match_rate": round(exact / max(1, len(keys)), 4),
        "hf_cell_thwart_rate": round(hf_thwart / max(1, len(keys)), 4),
        "vllm_cell_thwart_rate": round(vl_thwart / max(1, len(keys)), 4),
        "hf_cell_compliance_rate": round(1 - hf_thwart / max(1, len(keys)), 4),
        "vllm_cell_compliance_rate": round(1 - vl_thwart / max(1, len(keys)),
                                           4),
        "per_prompt_accept_agreement": round(agree / max(1, len(pids)), 4),
        "accepted_by_both": both, "accepted_hf_only": hf_only,
        "accepted_vllm_only": vl_only,
        "hf_prompts_accepted": sum(hf_acc.values()),
        "vllm_prompts_accepted": sum(vl_acc.values()),
        "hf_prompt_accept_rate": round(sum(hf_acc.values()) / N, 4) if N else None,
        "vllm_prompt_accept_rate": round(sum(vl_acc.values()) / N, 4) if N else None,
        "accept_rate_diff_vllm_minus_hf": round(mean_d, 4),
        "accept_rate_diff_ci95": [round(ci_lo, 4), round(ci_hi, 4)],
        "accept_rate_diff_ci95_pp": [round(ci_lo * 100, 2), round(ci_hi * 100, 2)],
        "discordant_b_vllm_only_c_hf_only": [b, c],
        "equivalence_margin_pp": round(margin * 100, 2),
        "equivalent_within_margin": equivalent,
        "hf_accepted_dist": _acc_stats(hf),
        "vllm_accepted_dist": _acc_stats(vl),
    }
    print(json.dumps(report, indent=2))
    (out / "compare_report.json").write_text(json.dumps(report, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="GPU: draw K candidates both ways")
    g.add_argument("--config", default="configs/two_si.yaml")
    g.add_argument("--si", default="si_constitution")
    g.add_argument("--candidates", type=int, default=4)
    g.add_argument("--n", type=int, default=50, help="residue prompts to compare")
    g.add_argument("--private-root", default=None)
    g.add_argument("--out", required=True, help="dir for hf.jsonl / vllm.jsonl")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--hf-batch-size", type=int, default=16)
    g.add_argument("--max-model-len", type=int, default=16384)
    g.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    g.add_argument("--enforce-eager", action="store_true")
    g.add_argument("--vllm-top-p", type=float, default=1.0,
                   help="nucleus cutoff for the vLLM half (1.0 reproduces the "
                        "original run; 0.9 matches the HF teacher, to test "
                        "whether top_p explains the gap)")
    g.add_argument("--vllm-top-k", type=int, default=-1,
                   help="top_k for the vLLM half (-1 disabled reproduces the "
                        "original; 50 matches the HF-inherited default)")
    g.add_argument("--skip-hf", action="store_true",
                   help="reuse the existing hf.jsonl (regenerate only the vLLM "
                        "half, e.g. at a different --vllm-top-p)")

    c = sub.add_parser("grade-compare",
                       help="the grading host: grade both + report agreement")
    c.add_argument("--config", default="configs/two_si.yaml")
    c.add_argument("--out", required=True, help="dir holding hf/vllm.jsonl")
    c.add_argument("--concurrency", type=int, default=200,
                   help="peak concurrent live grader requests (Low-Prio). "
                        "Default 200 (the standing wide-run default, matching "
                        "grade-store): do NOT throttle below it to dodge the "
                        "coordination-channel post — the post is "
                        "printed as a courtesy and never gates the run. On 429s "
                        "halve and keep halving; never hop to another org key. "
                        "The grader keeps max_retries=8 (deliberate, not the "
                        "SDK default of 2).")
    c.add_argument("--margin", type=float, default=0.05,
                   help="pre-registered equivalence bound on the per-prompt "
                        "acceptance-rate difference (default 0.05 = 5pp). "
                        "Equivalence is concluded only if the WHOLE 95%% CI "
                        "falls inside +/- this margin.")
    c.add_argument("--allow-unscored", type=int, default=0)

    args = ap.parse_args()
    {"generate": cmd_generate, "grade-compare": cmd_grade_compare}[args.cmd](
        args)


if __name__ == "__main__":
    main()
