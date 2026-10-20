#!/usr/bin/env python3
"""reproduce/tools/replan_compute.py — re-derive the compute plan from MEASUREMENTS.

Every GPU-hour row of the plan is provisional until the FD 23 memprobe (timed steps at micro-batch 1/2/4, 8,192 tokens)
and the FD 24 serving probe (admitted max_num_seqs + throughput per window) exist. This tool reads them, writes
`<state>/<model>/compute_plan.json`, and the fan-out stages (03 --phase design, 04, 06 --view != u0, 09) REFUSE until it
has run (`repro_require_compute_plan`).

    replan_compute.py --model K --memprobe memprobe.psr_ce_all.json --memprobe memprobe.lora_r4_ce.json
                      --fits-config configs/sdfbase_<key>_fits.yaml
                      [--fold fold_memprobe.json ...] [--probe probe_verdict.json] [--fit-width N]
                      [--tok-s ARM=TOKENS_PER_S ...] [--out F] [--step-threshold-s 2.0]
                      [--constitution-tokens 42785435] [--control-tokens 42785435] [--trial-tokens 10000000]
                      [--epochs-constitution 16] [--epochs-control 3] [--trial-epochs 3] [--dry-run]
    replan_compute.py --model K --serve-only --probe probe_verdict.json --fits-config configs/sdfbase_<key>_fits.yaml

``--serve-only`` is the plan of a model that IMPORTS its objects (the worked example: objects/DEMO.json's committed objects staged
by 05 in place of stage 04's fits — demo-import M6). No memprobe is read and no fit-side row is priced: ``02_memprobe``,
``03_recon``, ``03_relu_trials``, ``03_lora_trials``, ``04_fits`` and ``04_derived`` are marked ``imported`` (``gpu_h`` null, never
pending, listed under ``imported_rows``; a baseline figure stays beside the row as ``gpu_h_if_run``), so ``--memprobe`` /
``--fold`` / ``--tok-s`` / ``--fit-width`` are refused beside it. THE SERVING ROWS ARE PRICED EXACTLY AS BELOW: the same certified
``probe_verdict.json`` from ``03_hpo.sh --phase probe`` (under the paper's protocol the probe certifies the one declared pair, the
fits config's ``lora_r4_ce`` ``fit_selection``), the same gate — ``03_serving_probe`` is NOT an imported row (the probe actually
runs: a freshly executed probe is never an imported plan row, and its verdict has consumers beyond this plan — the steered-standup
gate and the padded preflight read its staged geometry record), ``--exclude-row`` never stands in for it, and with no ``--probe``
the two serving rows pend and ``complete`` is false. The document records ``serve_only``, ``imported_rows`` and
``imported_objects``; the schema stays 3.

The memprobe record's REAL layout (scripts/sdfbase/memprobe_fit.py, schema 1) is ``{"arm", "rungs": [{"micro_batch",
"fits", "s_per_step", "tokens_per_s", ...}, ...], "verdict": {"fits_at_min_micro_batch", "largest_fitting_micro_batch",
"s_per_step_at_min", "replan_required", ...}}`` — one record per arm, so ``--memprobe`` repeats. Fit / does-not-fit is the
VERDICT's word; the PACE is the widest FITTING rung's ``tokens_per_s`` (the closest thing the probe has to a warm
steady state: the wider rung ran after the narrow one warmed the kernels), overridable per arm with ``--tok-s ARM=V`` —
the ReLU steer fits only at micro-batch 1, where its three steps were cold (49 s/step = 167 tok/s, an upper bound on
the step, not a pace), so a plan whose ReLU rungs ran cold passes ``--tok-s psr_ce_all=<the LoRA arm's warm pace>`` and the
document says the assumption. ``verdict.s_per_step_at_min`` rides the row as the cold bound.

The fit width: ``wall_h = passes x tokens / (tok_s x N)`` while GPU-hours stay ``passes x tokens / tok_s`` — a
data-parallel fit splits the same work over N ranks (group-sharded DDP, effective batch unchanged). ``--fit-width``
or the fits config's top-level ``fit_width`` (``--fits-config``), else 1.

The EFFECTIVE BATCH each arm's ``accum_steps`` (= ceil(eff / micro-batch)) is derived from is the FITS CONFIG's own, never a
constant of this tool: ``batch_size_by_arm[arm] x train_kwargs.<arm>.accum_steps`` once the memprobe is folded (both ints),
else the ``PENDING_MEMPROBE_EFF<n>_ACCUM`` sentinel's n — the rule the trainer applies (``scripts/tune_arm.py``
``config_effective_batch``), so the plan's accumulation is the one the fits realise (the paper lanes 8, the demo lane and the
seed replicates 64). ``--fits-config`` is therefore REQUIRED whenever memprobe records are priced; a config whose arm states
neither key refuses by name (a plan on a guessed batch would be quietly wrong), and the arm's ``effective_batch`` rides its
record in the plan.

``--probe probe_verdict.json`` is OPTIONAL. Absent, the two serving-derived rows (``06_arms_multiturn``,
``06_arms_padding``) are ``pending`` (gpu_h null, excluded from the total, listed in ``pending_rows``), ``probe_status``
is ``pending`` and ``complete`` is false. Present, the rows are derived from the emitter's REAL schema
(scripts/sdfbase/serving_probe.py ``write_verdict``): the verdict must be ``status: complete`` and ``certifying`` with an
empty ``replan_required`` (a start-time prefill OOM above the measured ``batched_token_budget`` leaves the pair
undecided — nothing is priced on an undecided probe); the PRODUCTION serving pair (the fits config's LoRA architecture —
``train_kwargs.lora_r4_ce.fit_selection.lora_rank`` + ``lora_modules`` matched against ``pairs[*].lora_modules``, or
``--serving-pair RANK:SET``) must be a CERTIFIED pair whose SERVED CLASS (``pairs[*].served_class`` — the class a steered
standup of the pair serves: every genuine adapter + the zero slot [+ the carriers] in ONE engine, the pair's union of its
role servers) was certified at the multi-turn window (49,152) and the padding window
(98,304): ``served_class.certified_by_window[L]`` names the launch and ``served_class.measured_S_by_window[L]`` its engine's
S — the engine the pace was MEASURED on. THE S A ROW IS PRICED AT is ``served_class.standup.S_by_window[L]``: what a steered
standup of the pair is SIZED to over the record as folded (serving_probe.standup_resolution = geometry.unsteered_row_of, the
resolution ``native_standup.launch_class_geometry`` applies with NO module set: EVERY certified set's folded launches merged,
the least admitted S among the TIGHTEST launches dominating the served class at the window, inside the record's MINIMUM budget
``// L``) — never one set's union alone: a down union at S 20 beside an mlp union of the same (cap, slots) at S 9 stands the
down pair up at 9. The pace is the served class's own launch's measured aggregate
``server_tok_s`` at 49,152 (``served_class.throughput_by_window["49152"]`` — the probe measures throughput at 4,096 and 49,152
only, on the control-carrying natural launches, at closed-loop concurrencies 32/64/128; the point taken is the smallest
requested concurrency at or above that engine's S, i.e. the saturating one; ``client_tok_s`` when the server counter was
unavailable), RESCALED to the standup's S by the ratio ``standup S / measured S`` (never scaled up; stated on the row as an
assumption — a pace measured at S 20 is not what an engine admitting 9 delivers). NEVER the role servers' numbers: the pair's
``admitted_max_num_seqs`` and ``certified_by`` are the ROLES' certification by the split servers (S1cc / S1d), whose own
engines can admit more (or less) and run faster (or slower) than the one engine that serves the combined adapters (the
role provenance still rides each row as ``pair``). Any of these missing REFUSES (rc 3, nothing written) — a verdict
written before the pair record carried ``served_class`` / ``served_class.standup`` included (re-run ``serving_probe.py verdict``
on the record), a class whose standup the record cannot size (``standup.refused``: the standup refuses by the same words), and a
class of two or more adapters whose standup S at a priced window is below 2 — native_standup refuses that standup before launch
(the multi-LoRA routing control needs two distinct selections in one engine step), so the planning gate refuses the row rather
than pricing a standup that cannot come up: a verdict that cannot price a row never
yields ``probe_status: present``. The padding row applies the 49,152 pace at 98,304
scaled by the ratio of the padding window's STANDUP S to the MEASURED S at 49,152 (never scaled up) — stated on the row as an
assumption; the padding volume is the harness's own numbers
(padding_preflight.py: 2,590 prompts per cell; make_padding_configs.py: STATIC_CAP 16,384, RESERVE 8,192; the paper's five
levels). ``complete`` is true iff no row is pending; ``reproduce/lib/common.sh repro_require_compute_plan <model> complete``
refuses an incomplete plan.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time

#: the baseline table (GPU-hours); measured rows replace the fit / trial / multi-turn / padding entries
BASELINE = {
    "02_memprobe": {"gpu_h": 2, "note": "1 GPU x ~1 h per model"},
    "03_serving_probe": {"gpu_h": 6, "note": "1 x <= 4 h + TP fallback"},
    "03_recon": {"gpu_h": 3, "note": "2 jobs x ~1.5 h"},
    "03_relu_trials": {"gpu_h": None, "note": "derived: 16 + 4 trials x the measured trial pace"},
    "03_lora_trials": {"gpu_h": None, "note": "derived: 16 + 4 trials x the measured trial pace"},
    "04_fits": {"gpu_h": None, "note": "derived from the measured pace (below)"},
    "04_derived": {"gpu_h": 1, "note": "differences + refusal direction"},
    "06_unsteered": {"gpu_h": 40, "note": "3 standups + 4-GPU attacker pool"},
    "06_arms_multiturn": {"gpu_h": None, "note": "3 attacks x 10 arms; derived from the probe's admitted S at 49,152"},
    "06_arms_bon": {"gpu_h": 30}, "06_arms_static": {"gpu_h": 15},
    "06_arms_padding": {"gpu_h": None, "note": "derived from the padding preflight's window x the probe's admitted S"},
    "06_arms_petri": {"gpu_h": 8}, "06_arms_agentic": {"gpu_h": 10}, "06_arms_controls": {"gpu_h": 5},
    "06_arms_gsm8k": {"gpu_h": 5}, "06_arms_mmlu": {"gpu_h": 1},
    "07_bcb_bfcl": {"gpu_h": 6, "note": "Modal"}, "09_ladder": {"gpu_h": 20},
}
CE_ARMS = ("psr_ce_all", "lora_r4_ce")
#: --serve-only (demo-import M6): the fit-side rows a model that imports its objects never runs — the memprobe, the recon, both trial
#: ladders, the fits and the derived objects. NEVER 03_serving_probe: the probe runs for real and its verdict prices the serving rows.
IMPORTED_OBJECTS = "objects/DEMO.json"
IMPORTED_ROWS = ("02_memprobe", "03_recon", "03_relu_trials", "03_lora_trials", "04_fits", "04_derived")
IMPORTED_REASON = f"{IMPORTED_OBJECTS} committed objects staged by 05 in place of stage 04's fits; stage 04 skipped (and 02's memprobe, 03's recon and trials with it)"
#: (paper label, arm, corpus) of the four fitted cells; the epoch caps are the paper's (constitution 16, control 3 here —
#: an estimate; the registry's control fits stopped at 3-15 passes, median 6)
FIT_CELLS = (("constitution_relu", "psr_ce_all", "constitution"), ("control_relu", "psr_ce_all", "control"),
             ("constitution_lora", "lora_r4_ce", "constitution"), ("control_lora", "lora_r4_ce", "control"))
#: search design: 16 + 4 trials per object class
TRIALS_PER_CLASS = 20
MODEL_LOAD_H = 5.0 / 60.0   # the memprobe's measured ~5 min model load, per job
#: make_sdfbase_configs.ACCUM_SENTINEL_FMT ("PENDING_MEMPROBE_EFF{eff}_ACCUM") as a pattern (tune_arm._ACCUM_SENTINEL_RE): an
#: unfolded config's accum sentinel NAMES the effective batch it stands in for
ACCUM_SENTINEL_RE = re.compile(r"PENDING_MEMPROBE_EFF(\d+)_ACCUM")


class Refused(Exception):
    pass


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ------------------------------------------------------------------------------------------------ the memprobe record
def arm_of(mp: dict, path: str) -> str:
    arm = mp.get("arm")
    if arm not in CE_ARMS:
        raise Refused(f"{path}: memprobe record names arm {arm!r} (expected one of {CE_ARMS}); the record's `arm` is the key")
    return arm


def rungs_of(mp: dict, path: str) -> dict:
    """``{micro_batch: rung}`` from the record's ``rungs`` list — the real layout (schema 1); the legacy ``microbatch``
    mapping this tool once expected is refused by name (no memprobe ever wrote it)."""
    rungs = mp.get("rungs")
    if not isinstance(rungs, list) or not rungs:
        if isinstance(mp.get("microbatch"), dict):
            raise Refused(f"{path}: a `microbatch` mapping — no memprobe ever wrote that layout; schema 1 writes `rungs` + `verdict`")
        raise Refused(f"{path}: memprobe record carries no `rungs` list")
    out = {}
    for r in rungs:
        if not isinstance(r, dict) or "micro_batch" not in r:
            raise Refused(f"{path}: a rung without `micro_batch`: {r!r}")
        out[int(r["micro_batch"])] = r
    return out


def verdict_of(mp: dict, path: str) -> dict:
    v = mp.get("verdict")
    if not isinstance(v, dict) or "fits_at_min_micro_batch" not in v:
        raise Refused(f"{path}: memprobe record carries no `verdict.fits_at_min_micro_batch`")
    return v


def measured_pace(rungs: dict) -> tuple:
    """(tokens_per_s, micro_batch) of the WIDEST fitting rung with a pace, or (None, None)."""
    fitting = [(mb, r) for mb, r in sorted(rungs.items()) if r.get("fits") is True and r.get("tokens_per_s")]
    if not fitting:
        return None, None
    mb, r = fitting[-1]
    return float(r["tokens_per_s"]), int(mb)


def parse_tok_s(items) -> dict:
    out = {}
    for it in items or []:
        if "=" not in it:
            raise Refused(f"--tok-s wants ARM=TOKENS_PER_S, got {it!r}")
        arm, v = it.split("=", 1)
        if arm not in CE_ARMS:
            raise Refused(f"--tok-s names arm {arm!r} (expected one of {CE_ARMS})")
        try:
            val = float(v)
        except ValueError:
            raise Refused(f"--tok-s {arm}: {v!r} is not a number") from None
        if not (val > 0):
            raise Refused(f"--tok-s {arm}: pace must be > 0, got {val}")
        out[arm] = val
    return out


def read_arm(mp: dict, path: str, *, fold_bs, tok_s_override, threshold_s: float, effective_batch: int) -> dict:
    """One arm's measured facts: fits, micro-batch (the fold's when given, cross-checked against the verdict's largest
    fitting rung), the pace and where it came from, the cold bound; ``accum_steps`` = ceil(the ARM's effective batch — the
    fits config's (effective_batch_of) — / micro-batch), the batch riding the record."""
    arm = arm_of(mp, path)
    rungs = rungs_of(mp, path)
    v = verdict_of(mp, path)
    fits = bool(v.get("fits_at_min_micro_batch"))
    largest = v.get("largest_fitting_micro_batch")
    if fold_bs is not None and largest is not None and int(fold_bs) != int(largest):
        raise Refused(f"{arm}: the fold says micro-batch {fold_bs} but the memprobe's largest fitting rung is {largest} "
                      f"({path}) — refold (02 --step memprobe) before planning")
    mb = int(fold_bs) if fold_bs is not None else (int(largest) if largest is not None else None)
    meas_tok_s, meas_mb = measured_pace(rungs)
    cold = v.get("s_per_step_at_min")
    rec = {"arm": arm, "record": path, "fits": fits, "micro_batch": mb, "largest_fitting_micro_batch": largest,
           "effective_batch": int(effective_batch),
           "accum_steps": (math.ceil(int(effective_batch) / mb) if mb else None),
           "measured_tokens_per_s": meas_tok_s, "measured_at_micro_batch": meas_mb,
           "s_per_step_at_min_micro_batch": cold, "replan_required": bool(v.get("replan_required")),
           "step_threshold_s": threshold_s}
    if tok_s_override is not None:
        rec.update({"tokens_per_s": tok_s_override, "pace_source": "override (--tok-s)",
                    "pace_note": (f"the record's own widest fitting rung measured {meas_tok_s:.0f} tok/s at micro-batch "
                                  f"{meas_mb} ({rungs[meas_mb].get('steps_completed')} steps incl. kernel warm-up — a bound, "
                                  "not a pace)" if meas_tok_s else "the record carries no timed fitting rung")})
    else:
        rec.update({"tokens_per_s": meas_tok_s, "pace_source": (f"memprobe widest fitting rung (micro-batch {meas_mb})"
                                                                 if meas_tok_s else "none")})
    return rec


# ---------------------------------------------------------------------------------------------------- the derivations
def hours(passes: float, tokens: float, tok_s: float) -> float:
    return passes * tokens / tok_s / 3600.0


def fit_rows(arms: dict, *, fit_width: int, tokens: dict, epochs: dict) -> dict:
    """The 04 row: per fitted cell, GPU-hours (= wall at width 1) and the data-parallel wall at width N — the same
    work over N ranks, so gpu_h is width-invariant and wall_h = gpu_h / N (+ one model load per job)."""
    cells = {}
    total_gpu_h = 0.0
    walls = []
    for label, arm, corpus in FIT_CELLS:
        a = arms.get(arm)
        if a is None or not a.get("fits") or not a.get("tokens_per_s"):
            cells[label] = {"arm": arm, "corpus": corpus, "status": "pending",
                            "reason": ("no memprobe record for the arm" if a is None else
                                       "does not fit at micro-batch 1" if not a.get("fits") else "no pace")}
            continue
        gpu_h = hours(epochs[corpus], tokens[corpus], a["tokens_per_s"])
        wall_h = gpu_h / fit_width + MODEL_LOAD_H
        cells[label] = {"arm": arm, "corpus": corpus, "passes": epochs[corpus], "corpus_tokens": tokens[corpus],
                        "tokens_per_s": a["tokens_per_s"], "pace_source": a["pace_source"],
                        "h_per_pass_1gpu": round(hours(1, tokens[corpus], a["tokens_per_s"]), 2),
                        "gpu_h": round(gpu_h, 1), "fit_width": fit_width, "wall_h": round(wall_h, 1),
                        "wall_days": round(wall_h / 24.0, 2)}
        total_gpu_h += gpu_h
        walls.append(wall_h)
    return {"gpu_h": round(total_gpu_h, 1) if walls else None, "fit_width": fit_width,
            "critical_path_wall_h": round(max(walls), 1) if walls else None,
            "note": ("all four cells in parallel, each on fit_width GPUs (group-sharded DDP; effective batch 64 unchanged); "
                     "gpu_h is width-invariant, wall = gpu_h / fit_width + model load"),
            "cells": cells}


def trial_rows(arms: dict, *, trial_tokens: int, trial_epochs: int) -> dict:
    out = {}
    for key, arm in (("03_relu_trials", "psr_ce_all"), ("03_lora_trials", "lora_r4_ce")):
        a = arms.get(arm)
        if a is None or not a.get("fits") or not a.get("tokens_per_s"):
            out[key] = {"gpu_h": None, "status": "pending", "arm": arm, "note": BASELINE[key]["note"]}
            continue
        per_trial = hours(trial_epochs, trial_tokens, a["tokens_per_s"]) + MODEL_LOAD_H
        out[key] = {"gpu_h": round(TRIALS_PER_CLASS * per_trial, 1), "arm": arm, "trials": TRIALS_PER_CLASS,
                    "h_per_trial_1gpu": round(per_trial, 2), "trial_passes": trial_epochs, "trial_tokens": trial_tokens,
                    "tokens_per_s": a["tokens_per_s"], "pace_source": a["pace_source"],
                    "note": "one GPU per trial (D96: trials stay width 1); 16 in one wave + 4 coverage-repair"}
    return out


#: the multi-turn wave's geometry (stage 06): 259 confirmatory goals x ~10 turns per (attack, arm), ~1,500 tokens per
#: reply on average, 3 attacks x 10 arms, 12 one-GPU target units per cluster; the padding leg (docs/static_padding.md):
#: 2,590 prompts per cell over the 259-goal pool (padding_preflight.py), the paper's five levels, served at the 98,304 window
#: with the static ladder's ceiling cap and the harness reserve (make_padding_configs.STATIC_CAP / RESERVE)
MT_WINDOW = 49152
MT_GOALS, MT_TURNS, MT_REPLY_TOKENS, MT_ATTACKS, MT_ARMS, MT_UNITS_PER_CLUSTER = 259, 10, 1500, 3, 10, 12
PADDING_WINDOW = 98304
PADDING_LEVELS = (0, 8192, 16384, 32768, 65536)
PADDING_PROMPTS_PER_CELL = 2590
PADDING_CAP_DEFAULT, PADDING_RESERVE_DEFAULT = 16384, 8192
PADDING_ARMS = 10


def padding_constants() -> dict:
    """STATIC_CAP / RESERVE from the padding lane's own generator when importable (the harness's numbers, never typed
    twice), else the recorded defaults with the source stated."""
    here = os.path.dirname(os.path.abspath(__file__))
    mod = os.path.join(here, "..", "..", "scripts", "sdfbase", "full22", "static_padding", "make_padding_configs.py")
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("_mpc", mod)
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)   # type: ignore[union-attr]
        return {"cap": int(m.STATIC_CAP), "reserve": int(m.RESERVE), "source": "make_padding_configs.py"}
    except Exception as e:  # noqa: BLE001 — the tool is stdlib; the generator needs yaml
        return {"cap": PADDING_CAP_DEFAULT, "reserve": PADDING_RESERVE_DEFAULT,
                "source": f"typed defaults (make_padding_configs.py not importable: {type(e).__name__})"}


def serving_pair_of(args, fits_cfg_path) -> tuple:
    """``(rank, sorted lora_modules)`` of the PRODUCTION serving pair: ``--serving-pair RANK:SET`` names the verdict's set
    id; else the fits config's LoRA architecture (``train_kwargs.lora_r4_ce.fit_selection`` lora_rank + lora_modules,
    the one-line yaml reads of the generated file). Returns (rank, modules or None, set_id or None)."""
    if args.serving_pair:
        if ":" not in args.serving_pair:
            raise Refused(f"--serving-pair wants RANK:SET, got {args.serving_pair!r}")
        r, set_id = args.serving_pair.split(":", 1)
        try:
            return int(r), None, set_id.strip()
        except ValueError:
            raise Refused(f"--serving-pair rank {r!r} is not an integer") from None
    if not fits_cfg_path:
        raise Refused("--probe needs the production serving pair: pass --fits-config (its lora_r4_ce fit_selection) or --serving-pair RANK:SET")
    rank, mods, in_sel, in_lora = None, None, False, False
    with open(fits_cfg_path, encoding="utf-8") as fh:
        for ln in fh:
            st = ln.rstrip("\n")
            if st.startswith("  lora_r4_ce:"):
                in_lora = True; continue
            if in_lora and st.startswith("  ") and not st.startswith("    ") and st.strip():
                in_lora = False
            if in_lora and st.strip() == "fit_selection:":
                in_sel = True; continue
            if in_sel:
                if st.strip().startswith("lora_rank:"):
                    rank = int(st.split(":", 1)[1].strip())
                elif st.strip().startswith("lora_modules:"):
                    mods = []
                elif mods is not None and st.strip().startswith("- ") and st.startswith("      "):
                    mods.append(st.strip()[2:].strip())
                elif mods is not None and not st.startswith("      "):
                    in_sel = False
    if rank is None or not mods:
        raise Refused(f"{fits_cfg_path}: no train_kwargs.lora_r4_ce.fit_selection with lora_rank + lora_modules — pass --serving-pair RANK:SET")
    return rank, sorted(mods), None


def _certified_pair(pv: dict, rank: int, mods, set_id) -> dict:
    """The verdict's CERTIFIED pair record for the production architecture, or a refusal naming what the verdict has."""
    for pr in pv.get("pairs") or []:
        if int(pr.get("rank", -1)) != rank:
            continue
        if set_id is not None and pr.get("set") == set_id:
            return pr
        if mods is not None and sorted(pr.get("lora_modules") or []) == mods:
            return pr
    have = [(p.get("rank"), p.get("set")) for p in pv.get("pairs") or []]
    want = f"rank {rank} / set {set_id}" if set_id else f"rank {rank} / modules {mods}"
    excl = [(e.get("rank"), e.get("set"), e.get("reason")) for e in pv.get("excluded") or []]
    raise Refused(f"the production serving pair ({want}) is not a CERTIFIED pair of the verdict (certified: {have}; "
                  f"excluded/undecided: {excl}) — the serving rows price only what the probe certified")


def _served_class_at(pair: dict, window: int) -> tuple:
    """``(certifying launch id, measured S)`` of the pair's SERVED CLASS at ``window`` — the one engine that served every adapter
    of the pair there (the verdict's ``pairs[*].served_class``), which ``geometry.least_admitted_by_window`` sizes a steered
    standup from — or a refusal naming what the verdict lacks. Never the role servers' ``admitted_max_num_seqs``: those
    certify the roles on split engines whose capacity is not the served engine's."""
    tag = f"pair (rank {pair['rank']}, {pair['set']})"
    what = {MT_WINDOW: "multi-turn", PADDING_WINDOW: "padding"}.get(int(window), str(window))
    sc = pair.get("served_class")
    if not isinstance(sc, dict):
        raise Refused(f"{tag}: the verdict's pair record carries no `served_class` block (its served class was never certified: a "
                      f"verdict written before the served launch class was part of the certification) — re-run `serving_probe.py "
                      f"verdict` on the record; the serving rows price the class a steered standup serves, never the role servers")
    cls = f"the served class (rank cap {sc.get('max_lora_rank')}, {sc.get('max_loras')} adapters)"
    lid = (sc.get("certified_by_window") or {}).get(str(window))
    S = (sc.get("measured_S_by_window") or {}).get(str(window))
    if lid is None or S is None:
        raise Refused(f"{tag}: {cls} at the {what} window {window} was not MEASURED (certified_by_window "
                      f"{sc.get('certified_by_window')}, measured S {sc.get('measured_S_by_window')}) — the probe's plan must carry "
                      f"that window and the class's launch must have PASSED there")
    if int(S) < 1:
        raise Refused(f"{tag}: {cls} admitted S {S} < 1 at window {window} (launch {lid})")
    return str(lid), int(S)


def _standup_S_at(pair: dict, window: int) -> tuple:
    """``(standup S, the launches that set it, source)`` at ``window``: the S a steered standup of the pair is SIZED to over the record as
    folded — the verdict's ``served_class.standup`` (serving_probe.standup_resolution = geometry.unsteered_row_of: EVERY certified set's
    folded launches, the least admitted S among the tightest launches dominating the served class at the window, inside the record's
    minimum budget // L — the resolution native_standup.launch_class_geometry applies with no module set). Never one set's union alone:
    another set's launch of the same class caps the standup below it. Refuses a verdict without the
    block (written before the resolution rode the pair record), a class the record cannot size (``refused``: the standup refuses by
    the same words), an unresolved window, an S below 1, and — for a class of TWO OR MORE adapters — an S below 2: a steered standup
    of such a class REFUSES before launch at that width (native_standup, right after launch_class_geometry: ``max_num_seqs < 2`` with
    two or more adapters — the multi-LoRA routing control, selection_honored, needs two distinct selections in ONE engine step and
    cannot pass at one sequence per step), so a row priced at S 1 would price a standup that never comes up. The adapter count is
    the class's ``max_loras`` — every adapter the standup registers, the zero slot included, exactly the count the standup's own
    refusal reads."""
    tag = f"pair (rank {pair['rank']}, {pair['set']})"
    what = {MT_WINDOW: "multi-turn", PADDING_WINDOW: "padding"}.get(int(window), str(window))
    sc = pair.get("served_class") or {}
    st = sc.get("standup")
    cls = f"the served class (rank cap {sc.get('max_lora_rank')}, {sc.get('max_loras')} adapters)"
    if not isinstance(st, dict):
        raise Refused(f"{tag}: the verdict's pair record carries no `served_class.standup` block (the standup resolution of the class over the "
                      f"folded record — every certified set's launches, the least S among the tightest launches dominating the class) — "
                      f"re-run `serving_probe.py verdict` on the record; the serving rows price the S a steered standup is sized to, never "
                      f"one set's union alone")
    if st.get("refused"):
        raise Refused(f"{tag}: a steered standup of {cls} would REFUSE at launch-class geometry over this record — {st['refused']} — so "
                      f"nothing prices the rows (the standups the rows price would not come up)")
    S = (st.get("S_by_window") or {}).get(str(window))
    if S is None:
        raise Refused(f"{tag}: the standup resolution of {cls} carries no S at the {what} window {window} (S_by_window "
                      f"{st.get('S_by_window')}) — the probe's plan must carry that window")
    launches = (st.get("launches_by_window") or {}).get(str(window))
    if int(S) < 1:
        raise Refused(f"{tag}: the standup resolution of {cls} sizes window {window} at S {S} < 1 (launches {launches})")
    # the standup's own pre-launch refusal (native_standup, after launch_class_geometry): a width below two with two or more adapters
    # never comes up — the multi-LoRA routing control needs two distinct selections in one engine step — so the planning gate refuses
    # the same standup BEFORE pricing it (a serve-only plan priced at S 1 read `complete`)
    n_adapters = sc.get("max_loras")
    if n_adapters is None:
        raise Refused(f"{tag}: the verdict's `served_class` carries no adapter count (`max_loras`) — the standup's width gate (S < 2 "
                      f"with two or more adapters refuses) cannot be applied; re-run `serving_probe.py verdict` on the record")
    if int(S) < 2 and int(n_adapters) >= 2:
        raise Refused(f"{tag}: the standup resolution of {cls} sizes the {what} window {window} at S {S} with {n_adapters} adapters "
                      f"(launches {launches}) — a steered standup of the class REFUSES before launch at that width (native_standup: "
                      f"max_num_seqs={S} with {n_adapters} adapters: the multi-LoRA routing control (selection_honored) needs two distinct "
                      f"selections in ONE engine step and cannot pass at one sequence per step), so nothing prices the row: the standup "
                      f"it would price cannot come up")
    return int(S), list(launches or []), st.get("source")


def _throughput_point(pv: dict, pair: dict, window: int) -> dict:
    """The measured aggregate tok/s of the pair's SERVED-CLASS launch at ``window`` (:func:`_served_class_at`): the saturating
    closed-loop point (smallest requested concurrency >= the served class's S, else the largest), ``server_tok_s`` first,
    ``client_tok_s`` as the recorded fallback."""
    lid, S = _served_class_at(pair, window)
    sc = pair["served_class"]
    manifest = next((row.get("manifest") for row in pv.get("launches") or [] if row.get("id") == lid), sc.get("manifest"))
    pts = (sc.get("throughput_by_window") or {}).get(str(window)) or []
    pts = [pt for pt in pts if (pt.get("server_tok_s") or pt.get("client_tok_s"))]
    if not pts:
        raise Refused(f"the verdict carries no measured throughput for the served class of pair (rank {pair['rank']}, {pair['set']}) "
                      f"at window {window} (served_class.throughput_by_window['{window}'] of its certifying launch {lid} — the probe "
                      f"measures it on the control-carrying natural launches at {MT_WINDOW} and 4096 only); a row cannot be priced "
                      f"without a measured pace")
    at_or_above = [pt for pt in pts if int(pt.get("requested_concurrency") or 0) >= S]
    pt = min(at_or_above, key=lambda q: int(q.get("requested_concurrency") or 0)) if at_or_above \
        else max(pts, key=lambda q: int(q.get("requested_concurrency") or 0))
    src = "server_tok_s" if pt.get("server_tok_s") else "client_tok_s"
    return {"manifest": manifest, "launch": lid, "window": window, "requested_concurrency": pt.get("requested_concurrency"),
            "tok_s": float(pt[src]), "source": src, "errors": pt.get("errors"),
            "throughput_source": pt.get("throughput_source")}


def probe_rows(pv, *, serving: tuple, padding_prompts: int, padding_levels, padding_cap: int, padding_reserve: int,
               padding_source: str) -> tuple:
    """The serving-derived rows from the emitter's REAL verdict, or both ``pending`` when no verdict was given. Every
    number is a CERTIFIED measurement of the production pair's SERVED CLASS (the S and the aggregate tok/s of the one engine
    that served every adapter of the pair at the window — :func:`_served_class_at`, never the role servers') or a stated
    assumption; anything the verdict lacks refuses."""
    pend = {"gpu_h": None, "status": "pending", "reason": "no serving-probe verdict yet (--probe absent); re-run to fill"}
    if pv is None:
        return ({"06_arms_multiturn": dict(pend, note=BASELINE["06_arms_multiturn"]["note"]),
                 "06_arms_padding": dict(pend, note=BASELINE["06_arms_padding"]["note"])}, "pending", None)
    status = pv.get("status")
    if status != "complete" or not pv.get("certifying"):
        raise Refused(f"the serving-probe verdict is {status!r} (certifying={pv.get('certifying')}): only a complete, certifying "
                      f"verdict prices the serving rows (undecided pairs {pv.get('undecided_pairs')}, unreferenced "
                      f"{pv.get('unreferenced_pairs')}, replan_required {pv.get('replan_required')})")
    if pv.get("replan_required"):
        raise Refused(f"the verdict carries replan_required={pv['replan_required']} (D108: prefill OOMs above the measured budget "
                      f"{(pv.get('batched_token_budget') or {}).get('tokens')}) — run --resume --replan first")
    rank, mods, set_id = serving
    pair = _certified_pair(pv, rank, mods, set_id)
    # the SERVED CLASS at each priced window: its certifying launch and that engine's MEASURED S — where the pace was
    # measured — never the role servers' `admitted_max_num_seqs` (the roles' certification on split engines)
    lid_mt, S_meas_mt = _served_class_at(pair, MT_WINDOW)
    lid_pad, S_meas_pad = _served_class_at(pair, PADDING_WINDOW)
    # the S each row is PRICED at: the standup resolution over the folded record — every certified set's launches, the least
    # S among the tightest launches dominating the class, inside the record's minimum budget // L — what launch_class_geometry sizes
    # a steered standup of the pair to; another set's launch of the same class caps it below this pair's own union
    S_mt, l_mt, src_mt = _standup_S_at(pair, MT_WINDOW)
    S_pad, l_pad, _src_pad = _standup_S_at(pair, PADDING_WINDOW)
    tp = _throughput_point(pv, pair, MT_WINDOW)      # the saturating point of the MEASURED engine (its own S selects it)
    tok_s_meas = tp["tok_s"]
    sc = pair["served_class"]
    # the pace transfer, both rows: a pace measured on an engine admitting S_meas is applied at the standup's S scaled by the S ratio,
    # never up (the engine the standup runs admits S_mt / S_pad sequences, not S_meas) — the one assumption, stated on each row
    scale_mt = min(1.0, S_mt / S_meas_mt) if S_meas_mt else 1.0
    tok_s = tok_s_meas * scale_mt
    # the role provenance rides each row as `pair` (what certified the ROLES: the split servers); the served class — what the
    # row is PRICED from — stands beside it, with the launch per priced window
    pair_id = {"rank": int(pair["rank"]), "set": pair["set"], "lora_modules": pair.get("lora_modules"),
               "topology": pair.get("topology"), "certified_by": pair.get("certified_by")}
    served = {"max_lora_rank": sc.get("max_lora_rank"), "max_loras": sc.get("max_loras"), "target_class": sc.get("target_class"),
              "manifest": tp["manifest"]}
    standup_src = f"{src_mt} over the record's certified sets (serving_probe.standup_resolution = geometry.unsteered_row_of)"
    # multi-turn: 3 attacks x 10 arms, each (attack, arm) = goals x turns replies of ~1,500 tokens at the unit's aggregate pace
    replies = MT_GOALS * MT_TURNS
    h_attack_arm = replies * MT_REPLY_TOKENS / tok_s / 3600.0
    mt = {"gpu_h": round(MT_ATTACKS * MT_ARMS * h_attack_arm, 1), "basis": "certified", "pair": pair_id,
          "served_class": dict(served, launch=lid_mt),
          "window": MT_WINDOW, "admitted_S": S_mt, "standup_S_launches": l_mt, "standup_S_source": standup_src,
          "measured_S_at_pace_window": S_meas_mt, "tok_s_measured": tok_s_meas, "s_ratio_scale": round(scale_mt, 3),
          "tok_s": tok_s, "throughput_point": tp,
          "h_per_attack_arm": round(h_attack_arm, 2), "attacks": MT_ATTACKS, "arms": MT_ARMS,
          "goals": MT_GOALS, "turns": MT_TURNS, "reply_tokens": MT_REPLY_TOKENS,
          "wall_h_per_cluster": round(MT_ATTACKS * MT_ARMS * h_attack_arm / MT_UNITS_PER_CLUSTER, 1),
          "assumption": (f"the aggregate pace measured at window {MT_WINDOW} ({tok_s_meas:.0f} tok/s on launch {lid_mt}'s engine admitting "
                         f"S {S_meas_mt}) is applied at the standup's S {S_mt} scaled by {S_mt}/{S_meas_mt} (never scaled up)"
                         if scale_mt < 1.0 else
                         f"the pace was measured at the standup's own S {S_mt} (launch {lid_mt}); no transfer"),
          "note": (f"{MT_ATTACKS} attacks x {MT_ARMS} arms; {MT_UNITS_PER_CLUSTER} one-GPU target units per cluster at the "
                   f"standup S {S_mt} a steered standup of the served class is sized to (window {MT_WINDOW}; {src_mt}: launches {l_mt}) "
                   f"and {tok_s:.0f} tok/s (measured {tok_s_meas:.0f} tok/s at S {S_meas_mt} on launch {lid_mt}: the one engine serving "
                   f"every adapter of the pair — rank cap {served['max_lora_rank']}, {served['max_loras']} adapters); the ReLU / "
                   f"difference arms are priced at the LoRA pair's pace (served through the same engine)")}
    # padding: five levels x 10 arms; per cell 2,590 prompts of (level + reserve + cap) tokens at the padding window,
    # at the 49,152 pace scaled by (the padding window's standup S / the measured S at 49,152), never up — the pace transfer
    scale = min(1.0, S_pad / S_meas_mt) if S_meas_mt else 1.0
    tok_s_pad = tok_s_meas * scale
    per_level = {}
    total = 0.0
    for lvl in padding_levels:
        toks = int(lvl) + int(padding_reserve) + int(padding_cap)
        h_cell = padding_prompts * toks / tok_s_pad / 3600.0
        per_level[str(lvl)] = {"tokens_per_prompt": toks, "h_per_cell": round(h_cell, 2), "gpu_h_10_arms": round(PADDING_ARMS * h_cell, 1)}
        total += PADDING_ARMS * h_cell
    pad = {"gpu_h": round(total, 1), "basis": "certified S + stated pace transfer", "pair": pair_id,
           "served_class": dict(served, launch=lid_pad, launch_at_pace_window=lid_mt),
           "window": PADDING_WINDOW, "admitted_S": S_pad, "standup_S_launches": l_pad, "standup_S_source": standup_src,
           "admitted_S_at_pace_window": S_mt, "measured_S_at_pace_window": S_meas_mt, "measured_S": S_meas_pad,
           "tok_s_at_pace_window": tok_s_meas, "tok_s_assumed": round(tok_s_pad, 1), "s_ratio_scale": round(scale, 3),
           "prompts_per_cell": padding_prompts, "levels": list(padding_levels), "cap": padding_cap, "reserve": padding_reserve,
           "constants_source": padding_source, "arms": PADDING_ARMS, "per_level": per_level,
           "assumption": (f"the aggregate pace measured at window {MT_WINDOW} ({tok_s_meas:.0f} tok/s on launch {lid_mt}'s engine admitting "
                          f"S {S_meas_mt}) is applied at window {PADDING_WINDOW} at the standup S {S_pad} scaled by {S_pad}/{S_meas_mt} "
                          "(never scaled up); prefill tokens are priced at the decode pace, so this is an upper bound on the wall"),
           "note": BASELINE["06_arms_padding"]["note"]}
    return ({"06_arms_multiturn": mt, "06_arms_padding": pad}, "present",
            {"status": status, "certifying": True, "probe_job_id": pv.get("probe_job_id"), "written_utc": pv.get("written_utc"),
             "batched_token_budget": pv.get("batched_token_budget"), "plan_batched_token_budget": pv.get("plan_batched_token_budget"),
             "replan_required": pv.get("replan_required") or [], "reference_check_basis": pv.get("reference_check_basis"),
             "pair": pair_id, "admitted_max_num_seqs": pair.get("admitted_max_num_seqs"),
             "windows_measured": pair.get("windows_measured"),
             # the served class the rows were priced from: the launch that certified it per window and its MEASURED S (where the pace
             # was measured), and the STANDUP resolution's S per window (what the rows are priced at); the roles' `admitted_max_num_seqs`
             # above is kept as the record of the roles' certification, never what the rows read
             "served_class": {k: sc.get(k) for k in ("max_lora_rank", "max_loras", "target_class", "manifest",
                                                      "certified_by_window", "measured_S_by_window", "standup")}})


def fit_width_of(args) -> int:
    if args.fit_width is not None:
        if args.fit_width < 1:
            raise Refused(f"--fit-width must be >= 1, got {args.fit_width}")
        return int(args.fit_width)
    if args.fits_config:
        # the key is top-level in the generated yaml: read the one line (no yaml dependency for a stdlib tool)
        w = 1
        with open(args.fits_config, encoding="utf-8") as fh:
            for ln in fh:
                if ln.startswith("fit_width:"):
                    try:
                        w = int(ln.split(":", 1)[1].strip())
                    except ValueError:
                        raise Refused(f"{args.fits_config}: fit_width is not an integer: {ln.strip()!r}") from None
        if w < 1:
            raise Refused(f"{args.fits_config}: fit_width must be >= 1, got {w}")
        return w
    return 1


def _fits_config_batch_keys(path: str) -> dict:
    """``{arm: (batch_size_by_arm value, train_kwargs.<arm>.accum_steps value)}`` from the generated fits yaml, read by the
    file's fixed layout (a stdlib tool, like fit_width_of / serving_pair_of): top-level ``batch_size_by_arm:`` with its two-space
    arm lines, top-level ``train_kwargs:`` with two-space arm blocks and four-space keys. An all-digit value is an int; anything
    else (the MEMPROBE / PENDING_MEMPROBE_EFF<n>_ACCUM sentinels) stays text; an absent key is None."""
    bs: dict = {}
    acc: dict = {}
    section = None     # the top-level key we are under
    arm = None         # the two-space arm block under train_kwargs

    def _val(text: str):
        v = text.split(":", 1)[1].strip()
        return int(v) if v.isdigit() else (v if v else None)
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            st = ln.rstrip("\n")
            if not st.strip() or st.lstrip().startswith("#"):
                continue
            if not st.startswith(" "):
                section = st.split(":", 1)[0].strip() if ":" in st else None
                arm = None
                continue
            if section == "batch_size_by_arm" and st.startswith("  ") and not st.startswith("   ") and ":" in st:
                bs[st.split(":", 1)[0].strip()] = _val(st)
            elif section == "train_kwargs":
                if st.startswith("  ") and not st.startswith("   ") and st.rstrip().endswith(":"):
                    arm = st.strip()[:-1]
                elif arm is not None and st.startswith("    ") and not st.startswith("     ") and st.strip().startswith("accum_steps:"):
                    acc[arm] = _val(st)
    return {a: (bs.get(a), acc.get(a)) for a in set(bs) | set(acc)}


def effective_batch_of(fits_cfg_path, arm: str) -> int:
    """The effective batch the fits config's ARM realises — the trainer's own rule (scripts/tune_arm.py config_effective_batch):
    ``batch_size_by_arm[arm] x train_kwargs.<arm>.accum_steps`` when both are ints (a memprobe-folded config), else the
    ``PENDING_MEMPROBE_EFF<n>_ACCUM`` sentinel's n (an unfolded config names its batch). No config, or an arm stating neither key,
    REFUSES by name: the plan's accumulation is never a guess."""
    if not fits_cfg_path:
        raise Refused("--fits-config is required to price the memprobe records: each arm's accum_steps = ceil(eff / micro-batch) is "
                      "derived from the FITS CONFIG's effective batch (batch_size_by_arm x accum_steps once folded, else its "
                      "PENDING_MEMPROBE_EFF<n>_ACCUM sentinel) — never a constant of this tool")
    keys = _fits_config_batch_keys(fits_cfg_path)
    bs, acc = keys.get(arm, (None, None))
    if bs is None and acc is None:
        raise Refused(f"{fits_cfg_path}: no batch_size_by_arm.{arm} / train_kwargs.{arm}.accum_steps — the config states no effective "
                      f"batch for {arm}, so its accumulation cannot be planned (a generated configs/sdfbase_*_fits.yaml carries both)")
    if isinstance(bs, int) and isinstance(acc, int):
        eff = bs * acc
        if eff < 1:
            raise Refused(f"{fits_cfg_path}: batch_size_by_arm.{arm}={bs} x accum_steps={acc} = {eff} < 1 — a broken fold")
        return eff
    m = ACCUM_SENTINEL_RE.fullmatch(str(acc)) if isinstance(acc, str) else None
    if m:
        return int(m.group(1))
    raise Refused(f"{fits_cfg_path}: train_kwargs.{arm}.accum_steps={acc!r} with batch_size_by_arm.{arm}={bs!r} names no effective "
                  f"batch (neither two ints nor a PENDING_MEMPROBE_EFF<n>_ACCUM sentinel) — regenerate the config (make_sdfbase_configs.py)")


def fold_micro_batches(paths) -> dict:
    """``{arm: micro_batch}`` from the fold records (``{model_key: {arm: bs}}`` — make_sdfbase_configs' --fold-memprobe shape;
    02 writes one per arm under the state dir), merged."""
    out: dict = {}
    for p in paths or []:
        d = load(p)
        if not isinstance(d, dict) or not d:
            raise Refused(f"{p}: fold record is not a {{model_key: {{arm: bs}}}} mapping")
        for key, arms in d.items():
            if not isinstance(arms, dict):
                raise Refused(f"{p}: fold[{key!r}] is not an {{arm: bs}} mapping")
            for arm, bs in arms.items():
                if arm not in CE_ARMS:
                    raise Refused(f"{p}: fold names arm {arm!r} (expected one of {CE_ARMS})")
                if arm in out and int(out[arm]) != int(bs):
                    raise Refused(f"{p}: fold micro-batch for {arm} ({bs}) disagrees with an earlier fold ({out[arm]})")
                out[arm] = int(bs)
    return out


def parse_excluded(items) -> dict:
    """`--exclude-row ROW=REASON`: {row: reason}; a bare name or an empty reason refuses — the exclusion is a ruling, named."""
    out: dict = {}
    for it in items:
        row, sep, reason = str(it).partition("=")
        if not sep or not row.strip() or not reason.strip():
            raise Refused(f"--exclude-row {it!r}: expected ROW=REASON (the ruling that excludes the row, e.g. 06_arms_padding='USER RULING D889')")
        if row.strip() in out:
            raise Refused(f"--exclude-row {row.strip()}: named twice")
        out[row.strip()] = reason.strip()
    return out


def serve_only_refusals(a) -> None:
    """``--serve-only`` prices nothing fit-side, so every fit-side input is refused by name beside it — a memprobe, a fold, a pace
    override or a width next to imported fits would be a plan that half-imports and half-prices the same objects."""
    for flag, val in (("--memprobe", a.memprobe), ("--fold", a.fold), ("--tok-s", a.tok_s), ("--fit-width", a.fit_width)):
        if val not in (None, [], ()):
            raise Refused(f"--serve-only takes no {flag}: the fit-side rows are imported ({IMPORTED_REASON}), so there is no memprobe to "
                          f"price them from — drop {flag}, or drop --serve-only to price a lane that fits its own objects")


def imported_rows(plan: dict) -> dict:
    """The fit-side rows of a serve-only plan, marked ``imported`` in place: ``gpu_h`` null (a baseline figure stays beside the row as
    ``gpu_h_if_run``), ``status: imported``, ``imported: true`` and the reason — never pending, never counted. Returns {row: reason}."""
    out = {}
    for row in IMPORTED_ROWS:
        base = plan[row]
        plan[row] = {**base, **({"gpu_h_if_run": base["gpu_h"]} if base.get("gpu_h") is not None else {}),
                     "gpu_h": None, "status": "imported", "imported": True, "reason": IMPORTED_REASON}
        out[row] = IMPORTED_REASON
    return out


def build_plan(a) -> dict:
    serve_only = bool(getattr(a, "serve_only", False))
    arms: dict = {}
    fit_width = None
    if serve_only:
        serve_only_refusals(a)
    else:
        if not a.memprobe:
            raise Refused("no --memprobe record: pass one per arm (scripts/sdfbase/memprobe_fit.py's memprobe.<arm>.json) to price the fit "
                          f"rows, or --serve-only for a model that imports its objects ({IMPORTED_OBJECTS}: the fit-side rows are then "
                          "imported, never priced; the serving rows are still priced from the probe's verdict)")
        tok_s = parse_tok_s(a.tok_s)
        fold = fold_micro_batches(a.fold)
        for path in a.memprobe:
            mp = load(path)
            arm = arm_of(mp, path)
            if arm in arms:
                raise Refused(f"two memprobe records for {arm} ({arms[arm]['record']}, {path}) — pass one per arm")
            arms[arm] = read_arm(mp, path, fold_bs=fold.get(arm), tok_s_override=tok_s.get(arm), threshold_s=a.step_threshold_s,
                                 effective_batch=effective_batch_of(a.fits_config, arm))
        for arm in tok_s:
            if arm not in arms:
                raise Refused(f"--tok-s {arm}: no memprobe record for that arm was passed")
        not_fitting = [arm for arm, r in arms.items() if not r["fits"]]
        if not_fitting:
            raise Refused(f"memprobe says micro-batch 1 does not fit for {not_fitting} — FD 23: the user decides the fit lane")
        fit_width = fit_width_of(a)
        for arm, r in arms.items():
            if r["accum_steps"] and r["accum_steps"] % fit_width:
                raise Refused(f"{arm}: fit_width={fit_width} does not divide accum_steps={r['accum_steps']} "
                              f"(= the fits config's effective batch {r['effective_batch']} / micro-batch {r['micro_batch']}) — the trainer "
                              "refuses the same; pick a width that divides it")
    tokens = {"constitution": a.constitution_tokens, "control": a.control_tokens}
    epochs = {"constitution": a.epochs_constitution, "control": a.epochs_control}
    plan = {k: dict(v) for k, v in BASELINE.items()}
    imported: dict = {}
    if serve_only:
        # demo-import M6 (Q3 ruled): the fit-side rows are IMPORTED — never pending, never priced — while the serving rows below go
        # through the SAME probe gate as a fitting lane's (a certified verdict of the production pair, or they pend): 03_serving_probe
        # keeps its baseline literal because the probe actually runs, and nothing here reads the wave's shape in its place
        imported = imported_rows(plan)
    else:
        plan["04_fits"] = fit_rows(arms, fit_width=fit_width, tokens=tokens, epochs=epochs)
        plan.update(trial_rows(arms, trial_tokens=a.trial_tokens, trial_epochs=a.trial_epochs))
    pv = load(a.probe) if a.probe else None
    serving = serving_pair_of(a, a.fits_config) if pv is not None else (None, None, None)
    pc = padding_constants()
    prows, probe_status, probe_rec = probe_rows(
        pv, serving=serving, padding_prompts=a.padding_prompts_per_cell,
        padding_levels=[int(x) for x in str(a.padding_levels).split(",") if x.strip()],
        padding_cap=(a.padding_cap if a.padding_cap is not None else pc["cap"]),
        padding_reserve=(a.padding_reserve if a.padding_reserve is not None else pc["reserve"]),
        padding_source=(pc["source"] if (a.padding_cap is None and a.padding_reserve is None) else "command line"))
    plan.update(prows)
    # a row EXCLUDED from the plan (the padding ladder for the steered objects) stays in the plan with gpu_h null,
    # `excluded: true`, the stated reason and its priced figure beside it (gpu_h_if_run) — listed under `excluded_rows`,
    # never `pending_rows` (the plan is complete without it; 06's `repro_require_compute_plan complete` passes) — and the total omits it
    excluded = parse_excluded(getattr(a, "exclude_row", None) or [])
    for row, reason in excluded.items():
        if row not in plan:
            raise Refused(f"--exclude-row {row}: not a row of the plan ({sorted(plan)})")
        if plan[row].get("imported"):
            raise Refused(f"--exclude-row {row}: the row is IMPORTED ({IMPORTED_REASON}) — it is not run and not counted already; an exclusion "
                          "sets a PRICED row aside, and an imported row has no price to set aside")
        if plan[row].get("gpu_h") is None:
            # an exclusion sets a PRICED row aside with its price beside it — never a way past the certified-serving gate
            # (with no --probe both serving rows pend; excluding them would have made the plan `complete` with no price evidence)
            raise Refused(f"--exclude-row {row}: the row is UNPRICED (gpu_h null — pending the serving probe); an exclusion sets a PRICED row aside "
                          f"with its price beside it: run the probe first, then exclude")
        plan[row] = {**plan[row], "gpu_h_if_run": plan[row].get("gpu_h"), "gpu_h": None, "status": "excluded", "excluded": True, "excluded_reason": reason}
    # pending = no number, not excluded by ruling, not imported (an imported row is never run, so it never waits on a measurement)
    pending = sorted(k for k, v in plan.items() if v.get("gpu_h") is None and not v.get("excluded") and not v.get("imported"))
    total = sum(float(v["gpu_h"]) for v in plan.values() if v.get("gpu_h") is not None)
    # the replan trigger: the paper's rule is on the step at the MINIMUM micro-batch (49 s cold here); the measured pace
    # at the widest fitting rung is what the rows are planned on — both are stated
    slowest_cold = max((float(r["s_per_step_at_min_micro_batch"] or 0) for r in arms.values()), default=0.0)
    replan = any(r["replan_required"] for r in arms.values()) or slowest_cold > a.step_threshold_s
    crit = plan["04_fits"].get("critical_path_wall_h")
    decision = []
    if serve_only:
        replan = False
        decision.append(f"the fit-side rows are IMPORTED ({IMPORTED_REASON}): {', '.join(IMPORTED_ROWS)} — no memprobe read, no fit replan; "
                        "the serving rows are priced from the probe's verdict exactly as for a fitting lane")
    elif replan:
        decision.append(f"the fit-schedule replan is IN FORCE: the memprobe's step at the minimum micro-batch is {slowest_cold:.1f} s "
                        f"(> {a.step_threshold_s:.1f} s) — the fits run DATA-PARALLEL at fit_width={fit_width} (D96: group-sharded DDP, "
                        "effective batch and epoch caps unchanged)")
    else:
        decision.append(f"fits proceed as planned: {slowest_cold:.2f} s/step <= {a.step_threshold_s:.1f} s")
    if crit is not None:
        decision.append(f"fit critical path {crit:.1f} h ({crit / 24:.1f} days) at width {fit_width}")
    if pending:
        decision.append(f"rows pending (excluded from the total): {', '.join(pending)}")
    if excluded:
        decision.append("rows EXCLUDED by ruling (not counted): " + ", ".join(f"{k} ({v})" for k, v in sorted(excluded.items())))
    return plan_document(model=a.model, memprobe=a.memprobe, fold=a.fold, probe=a.probe, probe_status=probe_status, probe_rec=probe_rec,
                         fit_width=fit_width, fits_config=a.fits_config, step_threshold_s=a.step_threshold_s, replan=replan, arms=arms,
                         plan=plan, pending=pending, excluded=excluded, total=total, decision=decision,
                         serve_only=serve_only, imported_rows=sorted(imported), imported_objects=(IMPORTED_OBJECTS if serve_only else None))


def plan_document(*, model, memprobe, fold, probe, probe_status, probe_rec, fit_width, fits_config, step_threshold_s, replan, arms, plan,
                  pending, excluded, total, decision, serve_only=False, imported_rows=(), imported_objects=None) -> dict:
    """The compute plan as WRITTEN (schema 3): every key the file carries, `complete` = no pending rows. The one shape 06's
    repro_require_compute_plan reads (`complete`, `pending_rows`) and the tests' fixtures are built from — never a hand literal.
    `serve_only` / `imported_rows` / `imported_objects` (demo-import M6) record a plan whose fit-side rows are imported; a fitting
    lane's plan carries them as false / [] / null."""
    return {"schema": 3, "model": model, "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "memprobe": list(memprobe), "fold": list(fold or []), "probe": probe, "probe_status": probe_status,
            "probe_verdict": probe_rec, "complete": not pending,
            "serve_only": bool(serve_only), "imported_rows": sorted(imported_rows or []), "imported_objects": imported_objects,
            "fit_width": fit_width, "fits_config": fits_config,
            "step_threshold_s": step_threshold_s, "fit_replan_triggered": replan,
            "arms": arms, "rows": plan, "pending_rows": pending, "excluded_rows": sorted(excluded), "total_gpu_h": round(total, 1),
            "total_gpu_h_note": "the sum of the rows with a number; pending rows, rows excluded by ruling and imported rows are not counted",
            "decision": "; ".join(decision)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--memprobe", action="append", default=[],
                    help="a memprobe record (scripts/sdfbase/memprobe_fit.py's memprobe.<arm>.json); repeat per arm — REQUIRED unless --serve-only")
    ap.add_argument("--serve-only", action="store_true",
                    help=f"the plan of a model that IMPORTS its objects ({IMPORTED_OBJECTS}; the worked example): the fit-side rows are `imported` "
                         "(never priced, never pending; refused beside --memprobe / --fold / --tok-s / --fit-width) while the serving rows go "
                         "through the SAME probe gate (--probe the certified verdict of 03_hpo.sh --phase probe, or they pend)")
    ap.add_argument("--fold", action="append", default=[],
                    help="a memprobe FOLD record {model_key: {arm: micro_batch}} (02 writes fold_memprobe.<arm>.json); repeatable")
    ap.add_argument("--probe", default=None, help="probe_verdict.json (OPTIONAL: absent -> the serving-derived rows are pending; present -> "
                                                  "must be complete + certifying, or the tool refuses)")
    ap.add_argument("--serving-pair", default=None, metavar="RANK:SET",
                    help="the production serving pair in the verdict's spelling (default: the fits config's lora_r4_ce fit_selection)")
    ap.add_argument("--padding-prompts-per-cell", type=int, default=PADDING_PROMPTS_PER_CELL,
                    help="prompts per padded cell (padding_preflight.py: 2,590 variants over the 259-goal pool)")
    ap.add_argument("--padding-levels", default=",".join(str(x) for x in PADDING_LEVELS), help="the padding levels (tokens)")
    ap.add_argument("--padding-cap", type=int, default=None, help="the static ladder's ceiling cap (default: make_padding_configs.STATIC_CAP)")
    ap.add_argument("--padding-reserve", type=int, default=None, help="the harness reserve (default: make_padding_configs.RESERVE)")
    ap.add_argument("--fit-width", type=int, default=None, help="D96 data-parallel width of the fits (default: --fits-config's fit_width, else 1)")
    ap.add_argument("--fits-config", default=None,
                    help="the lane's configs/sdfbase_<key>_fits.yaml — REQUIRED to price memprobe records: each arm's effective batch "
                         "(batch_size_by_arm x accum_steps once folded, else its PENDING_MEMPROBE_EFF<n>_ACCUM sentinel) is the "
                         "config's own; also the width (top-level fit_width) when --fit-width is absent and the production serving pair")
    ap.add_argument("--tok-s", action="append", default=[], metavar="ARM=TOKENS_PER_S",
                    help="override an arm's pace (the ReLU arm's warm pace is unmeasured at micro-batch 1)")
    ap.add_argument("--out", default=None); ap.add_argument("--step-threshold-s", type=float, default=2.0)
    ap.add_argument("--exclude-row", action="append", default=[], metavar="ROW=REASON",
                    help="a plan row EXCLUDED by ruling (D889: 06_arms_padding='USER RULING D889 …'): kept with gpu_h null, excluded: true and the reason, "
                         "listed under excluded_rows (never pending), not counted in the total; repeatable")
    ap.add_argument("--constitution-tokens", type=int, default=42_785_435)
    ap.add_argument("--control-tokens", type=int, default=42_785_435, help="the control corpus is token-matched to the constitution's")
    ap.add_argument("--trial-tokens", type=int, default=10_000_000, help="the HPO trial pool (constv2-mix-10m)")
    ap.add_argument("--epochs-constitution", type=int, default=16); ap.add_argument("--epochs-control", type=int, default=3)
    ap.add_argument("--trial-epochs", type=int, default=3)
    ap.add_argument("--seq-len", type=int, default=8192)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    try:
        doc = build_plan(a)
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 3
    except (OSError, ValueError) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    out = a.out or os.path.join(os.path.expanduser(os.environ.get("SISTEER_STATE_DIR", "~/.local/state/sisteer-reproduce")), a.model, "compute_plan.json")
    if a.dry_run:
        print(f"DRY-RUN: write {out}"); print(json.dumps(doc, indent=2)); return 0
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
    os.replace(out + ".tmp", out)
    print(f"compute plan -> {out}: total {doc['total_gpu_h']} GPU-h (probe {doc['probe_status']}, complete={doc['complete']}); {doc['decision']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
