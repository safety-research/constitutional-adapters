#!/usr/bin/env python
"""full22: the PRELAUNCH LENGTH CHECK of a prompted view.

For a prompted view every request carries the SI as a system prefix (~34k tokens for the full
constitution, ~5.9k for the short registry constitution), so the window must hold prefix + the leg's
LONGEST prompt + the leg's completion cap for EVERY leg the standup serves — else vLLM 400s on
prompt + max_tokens > L and the static adapter's once-only clamp would then SHRINK max_tokens
(max_tokens_clamped_from): for these cells a clamp is a defect, not a recovery. So, before every
standup of the view, render [system = the SI text] + each leg's longest prompt with the PINNED
tokenizer (the served render: the chat template with add_generation_prompt) and assert
prompt_tokens + cap <= L for the legs the standup serves at THEIR caps — read from the CONFIG (the
static families at attacks.static_max_new_tokens and the 8,192 rung, best-of-N at bon.max_new_tokens,
the judged controls at attacks.controls_max_new_tokens, MMLU at its letter-prefill max_tokens 1) or at
ONE cap for every named leg (--cap: a ladder rung's standup) — refusing the leg otherwise — and write
window_check.json beside the standup's endpoint.json.

  python scripts/sdfbase/full22/prompted_window_check.py --config configs/sdfbase_full22_llama8b_fullconst.yaml \\
      --si si_constitution_full --max-model-len 45056 [--legs static,bon,capability] [--cap N] [--out window_check.json]

Legs: static = the four families' rendered prompts over the confirmatory goals (each family's
*_variants builder, the longest); bon = the 100 JBB goals (the sigma-0.4 augmentation is ~prompt
length: the longest goal x 2 is the bound); capability = the longest MMLU user text (+ the longest
gsm8k question where the view's controls still carry gsm8k); controls = the longest XSTest prompt
(the config's xstest_subset, when set) and the longest OR-Bench prompt. Exit 0 = every leg fits at
every cap; 3 = a leg refused (named, with the tokens it needs); 2 = a usage problem.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "sdfbase" / "full22"))

STATIC_FAMILIES = ("pasttense", "codeattack", "renellm", "grok420")
#: the static escalation rung every prompted view's window must carry beside the base cap (the ladder's
#: first rung; the 16,384 ceiling is checked separately, --rungs, because a window that cannot carry it
#: stands that rung up at its own window)
STATIC_ESC_CAP = 8192
#: the MMLU letter-logprob request asks for ONE token (served_capability_run._mmlu_scoring_payload: max_tokens 1)
MMLU_CAP = 1
LEGS = ("static", "bon", "capability", "controls")


def leg_caps(cfg: dict, legs, *, cap_override=None) -> dict:
    """{leg: (caps...)} the check asserts — the RUN's caps from the config (static: the base cap AND the 8,192
    rung, a ONE-WINDOW view's standup carries both; bon: bon.max_new_tokens; controls: attacks.controls_max_new_tokens;
    capability: MMLU's 1 token), or ``--cap base`` (every named leg at its base cap only — a PER-LEG view's base
    standup; its rung standups are checked at their own cap), or ``--cap N`` (ONE cap for every named leg: a rung)."""
    a = cfg.get("attacks") or {}
    if cap_override not in (None, "base"):
        return {leg: (int(cap_override),) for leg in legs}
    out = {}
    if "static" in legs:
        base = int(a["static_max_new_tokens"])
        out["static"] = (base,) if cap_override == "base" else tuple(sorted({base, STATIC_ESC_CAP}))
    if "bon" in legs:
        out["bon"] = (int((cfg.get("bon") or {})["max_new_tokens"]),)
    ctl_cap = int(a.get("controls_max_new_tokens", a.get("target_max_new_tokens", 512)))
    if "controls" in legs:
        out["controls"] = (ctl_cap,)
    if "capability" in legs:
        out["capability"] = (MMLU_CAP,)
    return out


def rendered_tokens(tok, system: str, user: str) -> int:
    """The SERVED render's token count: the chat template over [system, user] + the generation header —
    the load check's counter (``load_check.rendered_len``), so the window check and the load check count
    one render the same way. ``system`` is the registered SI text here, never empty."""
    from load_check import rendered_len
    return rendered_len(tok, system, user)


def longest_static_prompts(goals, families=STATIC_FAMILIES) -> dict:
    """{family: (longest prompt text, goal_id)} over the goals via each family's variants builder."""
    from sisteer.eval.attacks import codeattack, grok420, pasttense, renellm
    builders = {"pasttense": pasttense.pasttense_variants, "codeattack": codeattack.codeattack_variants,
                "renellm": renellm.renellm_variants, "grok420": grok420.grok420_variants}
    out = {}
    for fam in families:
        best = ("", None)
        for g in goals:
            try:
                variants = builders[fam](g)
            except Exception as e:  # noqa: BLE001 — recorded, never swallowed
                raise SystemExit(f"[window-check] {fam}: cannot render goal {g.goal_id}: {type(e).__name__}: {e}")
            for _name, prompt in variants:
                if len(prompt) > len(best[0]):
                    best = (prompt, g.goal_id)
        out[fam] = best
    return out


def capability_prompts(controls, limit: int | None = None) -> dict:
    """{mmlu: longest user text[, gsm8k: longest question + instruction]} from the pinned corpora — MMLU through the
    served driver's own row loader and user-text renderer; gsm8k ONLY where the view's controls still carry it (the
    sdfbase-llama8b lane omits it: the bespoke gsm8k leg was retired on main, so its loader no longer exists on the
    driver and is never asked for)."""
    import served_capability_run as srun

    from sisteer.capeval import corpora
    pins = corpora.load_pins()
    out = {}
    if "mmlu" in controls:
        mmlu = srun._mmlu_rows(pins)
        if limit:
            mmlu = mmlu[:limit]
        out["mmlu"] = max((srun._mmlu_user_text(r) for r in mmlu), key=len)
    if "gsm8k" in controls:
        from sisteer.capeval.tiers import parity
        rows = parity.gsm8k_parity_rows(pins, n=parity.PARITY_N)
        if limit:
            rows = rows[:limit]
        out["gsm8k"] = max((f"{r['question']}\n\n{parity._GSM8K_INSTRUCTION}" for r in rows), key=len)
    return out


def controls_prompts(cfg: dict, controls, limit: int | None = None) -> dict:
    """{xstest: longest prompt, or_bench_hard_1k: longest prompt} from the pinned corpora, through the judged tier's own
    row loaders; XSTest over the config's subset (xstest_subset: safe = the 250 safe prompts) exactly as the driver."""
    from sisteer.capeval import corpora
    from sisteer.capeval.tiers import judged
    pins = corpora.load_pins()
    out = {}
    if "xstest" in controls:
        rows = judged.xstest_subset_rows(judged.xstest_rows(pins), cfg.get("xstest_subset"))
        if limit:
            rows = rows[:limit]
        out["xstest"] = max((r["prompt"] for r in rows), key=len)
    if "or_bench_hard_1k" in controls:
        rows = judged.or_bench_rows(pins)
        if limit:
            rows = rows[:limit]
        out["or_bench_hard_1k"] = max((r["prompt"] for r in rows), key=len)
    return out


def check(*, tok, system: str, max_model_len: int, legs, static_prompts: dict, bon_goals, cap_prompts: dict,
          ctl_prompts: dict = None, caps: dict = None, extra_rungs=()) -> dict:
    L = int(max_model_len)
    caps = caps or {"static": (4096, STATIC_ESC_CAP), "bon": (512,), "capability": (512,), "controls": (512,)}
    rows, problems = [], []

    def fit(leg: str, family: str, n: int, cap: int, *, label: str, report: bool = True, **ident) -> None:
        """One (leg, longest prompt, cap) row; a misfit is a named problem when ``report`` (a static EXTRA rung
        is recorded, never refused — a rung this window cannot carry stands up at its own window)."""
        need = n + cap
        fits = need <= L
        rows.append({"leg": leg, "family": family, **ident, "prompt_tokens": n, "cap": cap, "need": need, "L": L,
                     "fits": fits})
        if report and not fits:
            problems.append(f"{label} @ cap {cap}: {n} + {cap} = {need} > {L}")

    sys_only = rendered_tokens(tok, system, "")
    if "static" in legs:
        for fam, (prompt, gid) in static_prompts.items():
            n = rendered_tokens(tok, system, prompt)
            for cap in tuple(caps["static"]) + tuple(c for c in extra_rungs if c not in caps["static"]):
                fit("static", fam, n, cap, label=f"static/{fam}", goal_id=gid, report=cap in caps["static"])
    if "bon" in legs:
        longest = max((g.text for g in bon_goals), key=len) if bon_goals else ""
        n = rendered_tokens(tok, system, longest * 2)   # the augmentation bound: the longest goal x 2
        for cap in caps["bon"]:
            fit("bon", "bon", n, cap, label="bon")
    if "capability" in legs:
        for name, text in cap_prompts.items():
            n = rendered_tokens(tok, system, text)
            for cap in caps["capability"]:
                fit("capability", name, n, cap, label=f"capability/{name}")
    if "controls" in legs:
        for name, text in (ctl_prompts or {}).items():
            n = rendered_tokens(tok, system, text)
            for cap in caps["controls"]:
                fit("controls", name, n, cap, label=f"controls/{name}")
    return {"schema": "window_check/2", "ok": not problems, "max_model_len": L, "system_tokens_rendered": sys_only,
            "legs": list(legs), "caps": {k: list(v) for k, v in caps.items() if k in legs}, "rows": rows, "problems": problems,
            "extra_rungs": [{"cap": c, "fits_at_L": all(r["fits"] for r in rows if r["leg"] == "static" and r["cap"] == c)}
                            for c in extra_rungs],
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--config", required=True, help="the view's wave config (goals, bon, model pins, caps)")
    ap.add_argument("--si", required=True, help="the registered SI id of the prompted cell (si_constitution_full / si_constitution)")
    ap.add_argument("--max-model-len", type=int, required=True)
    ap.add_argument("--legs", default="static,bon,capability", help=f"comma list of {list(LEGS)}: the legs THIS standup serves")
    ap.add_argument("--cap", default=None, help="a number: check every named leg at THIS cap (a ladder rung's standup); `base`: each leg at its BASE cap only "
                    "(a per-leg view's base standup); default: each leg's own cap(s) from the config (the static leg at its base cap AND the 8,192 rung)")
    ap.add_argument("--rungs", default="16384", help="extra static rungs REPORTED (fit / not) but not required at this window")
    ap.add_argument("--limit", type=int, default=None, help="SMOKE: head-cap the goal / item sets")
    ap.add_argument("--out", default=None, help="window_check.json (default: beside --config's out dir is unknown; pass it)")
    args = ap.parse_args(argv)
    import jailbreak_run as jr
    import two_si_run as tsr

    from sisteer.capeval.manifest import sha256_hex16
    from sisteer.data.instructions import BY_ID
    from sisteer.models import load_tokenizer

    cfg = jr._load_cfg(args.config)
    legs = tuple(l for l in args.legs.split(",") if l)
    bad = [l for l in legs if l not in LEGS]
    if bad:
        raise SystemExit(f"[window-check] unknown legs {bad} (known: {list(LEGS)})")
    system = BY_ID[args.si].text
    tok = load_tokenizer(cfg["model"], revision=cfg.get("model_revision"))
    goals = tsr._eval_goals(cfg, confirmatory=True)
    bon_goals = tsr._eval_goals(cfg, confirmatory=True, attack="bon") if "bon" in legs else []
    if args.limit:
        goals, bon_goals = goals[: args.limit], bon_goals[: args.limit]
    static_prompts = longest_static_prompts(goals) if "static" in legs else {}
    # the view's controls list (f22_roster.view_controls through the roster: mmlu, and gsm8k only where the roster keeps it;
    # xstest / or_bench_hard_1k for the over-refusal leg) — the config's private_run names the (target, view)
    controls = _view_controls(cfg)
    cap_prompts = capability_prompts([c for c in controls if c in ("mmlu", "gsm8k")], args.limit) if "capability" in legs else {}
    ctl_prompts = controls_prompts(cfg, [c for c in controls if c in ("xstest", "or_bench_hard_1k")], args.limit) if "controls" in legs else {}
    rungs = tuple(int(r) for r in args.rungs.split(",") if r)
    cap_override = args.cap
    if cap_override is not None and cap_override != "base":
        if not str(cap_override).isdigit() or int(cap_override) < 1:
            raise SystemExit(f"[window-check] --cap must be a positive int or 'base', got {cap_override!r}")
        cap_override = int(cap_override)
    caps = leg_caps(cfg, legs, cap_override=cap_override)
    rec = check(tok=tok, system=system, max_model_len=args.max_model_len, legs=legs, static_prompts=static_prompts,
                bon_goals=bon_goals, cap_prompts=cap_prompts, ctl_prompts=ctl_prompts, caps=caps,
                extra_rungs=rungs if cap_override is None else ())
    rec.update({"config": args.config, "si": args.si, "si_sha256_hex16": sha256_hex16(system.encode()), "cap_override": cap_override})
    out = Path(args.out) if args.out else Path("window_check.json")
    out.write_text(json.dumps(rec, indent=2, default=str))
    worst = max((r["need"] for r in rec["rows"]), default=0)
    print(f"[window-check] {'PASS' if rec['ok'] else 'REFUSE'}: L={args.max_model_len} system={rec['system_tokens_rendered']} "
          f"tokens rendered; legs {','.join(legs)} at caps {rec['caps']}; worst need {worst}; extra rungs {rec['extra_rungs']} -> {out}", flush=True)
    for p in rec["problems"]:
        print(f"[window-check] REFUSED {p}", flush=True)
    return 0 if rec["ok"] else 3


def _view_controls(cfg: dict) -> tuple:
    """The (target, view) the config names (private_run = sdfbase_full22_<target>[_<view>]) -> f22_roster.view_controls;
    a config outside the roster (a smoke fixture) falls back to the historical full list."""
    try:
        import f22_roster as fr
        roster = fr.load_roster()
        tag = str(cfg.get("private_run", ""))[len("sdfbase_full22_"):]
        for target, t in roster["targets"].items():
            for view in t["views"]:
                if tag == (target if view == "main" else f"{target}_{view}"):
                    return fr.view_controls(roster, target, view)
    except Exception:  # noqa: BLE001 — a config outside the roster: every control
        pass
    return ("gsm8k", "mmlu", "xstest", "or_bench_hard_1k")


if __name__ == "__main__":
    sys.exit(main())
