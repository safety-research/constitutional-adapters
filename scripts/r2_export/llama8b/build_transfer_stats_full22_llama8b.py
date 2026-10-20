#!/usr/bin/env python3
"""``transfer_stats_full22_llama8b.json`` — the Llama-3.1-8B lane of the sdf-base-steering round-2 campaign (the four orphaned
``sdfbase_full22_llama8b*`` stores, graded 2026-09-13 by ``llama_grade/``), in the schema of ``transfer_stats_full22_all.json``, under a
NEW namespace ``llama8b``, written to a NEW file. Nothing of the wave's filed files is read for writing, rewritten or replaced.

WHY THIS FILE EXISTS. The four grades roots ``grades_llama8b_<tag>/<tag>/`` hold the campaign grader's aggregates for four views that no
join carries: ``llama8b`` (the base→instruct transfer lane), ``llama8b_it`` (the instruct-direct lane), ``llama8b_shortconst`` and
``llama8b_fullconst`` (prompted-only lanes: the unsteered model beside one prompted constitution cell). The export merges this file
ADDITIVELY: the four view keys are new, so nothing existing can be overwritten — but the export still emits these rows straight from
this file the way it does the cap-4,096 lane (``build_export.py``, the L8B block), never by growing the shared TS join, so no loop
that walks TS can pick them up by accident.

THE INSTRUMENT IS NOT THE FILED WAVE'S. Every other lane's filed BoN and multi-turn rows sit at a 512-token reply cap with greedy
multi-turn decode. The Llama configs (``configs/sdfbase_full22_llama8b*.yaml`` on the ``sdfbase-llama8b-stage`` lineage only) set
``target_max_new_tokens`` AND ``bon_max_new_tokens`` to 4,096, and the multi-turn target decode is the vendor preset (sampled, T 0.6,
top_p 0.9: ``generation_config.json`` of NousResearch/Meta-Llama-3.1-8B-Instruct @ d10aef79). So the grader's ``decode_cap_tokens: 4096``
stamp is, for once, the run's cap and not just the config's — and this builder still PROVES it per multi-turn cell from the producers'
own records, exactly as the cap-4,096 builder does: (a) every ``target_timing.jsonl`` row requested ``max_tokens`` 4,096 and the max
``completion_tokens`` lies in (512, 4096]; (b) the timing row count and the ``finish_reason == length`` count equal the grader's own
completion stamp (``completion.gated.target_timing``); (c) the ``responses.jsonl`` line count equals ``completion.gated.n_responses``.
For BoN the store carries NO per-row token counts, so the 4,096 cap is the config's stamp corroborated by the serving window
(``bon_served_run.max_model_len`` 4,864 = 4,096 + the prompt budget on the main / it views), zero window clamps, and the row / finish
counts agreeing between the store's manifest and the grader's stamp — recorded as such (``decode_cap_verification.kind``), never
upgraded to a per-row proof. Only numbers are read from the transcripts — never a prompt or a reply.

THE CONTRASTS ARE DEFINED. Unlike the cap-4,096 legs, every Llama view carries its own unsteered ``none`` cell at the SAME cap and
decode, so ``vs_bare`` (cell minus the same view's bare, paired on goals: ``transfer_r2.contrast`` = 10,000-draw bootstrap seed 0 +
exact McNemar) is filed per cell × attack. Where the cell dropped errored goals (below) the strict pairing refuses and the same reader
is re-run on the goal INTERSECTION, labelled ``paired_on_intersection`` — the GLM precedent the export already reads.

THREE FACTS THAT RIDE THE ROWS, NOT A NOTE. (1) Three crescendo cells were graded with ``--allow-errored-goals``: llama8b
constv2-mix-full__lora_r4_ce_minus_ctl (1 of 259), llama8b_it constv2-mix-full__lora_r4_ce (5), llama8b_it constv2ctl2-full__lora_r4_ce
(8); cause ``error:_StrategyRuntimeError``, per-goal cause unrecorded. Their reduced denominators are carried on the cell record as
``n_expected`` / ``n_graded`` / ``n_errored`` / ``errored_by_type`` / ``errored_goal_ids`` and in ``completion.n_error_outcomes``; this
builder REFUSES if the three counts disagree with each other, with the per_goal size, or with the ledger below (``ERRORED_EXPECTED``),
so a fourth errored cell or a changed count cannot pass silently. (2) Six PSR ActorAttack cells hold 106–189 of 259 goals and were
deliberately NOT graded — never in the served cell list, PSR out by user ruling. They are filed under ``not_graded`` per view (from
``llama_grade/inventory.json``, the chain's pre-flight, sha recorded) so the export can mark them absent-by-design rather than missing.
(3) The grading tree is ``sdfbase-llama8b-stage @ 390b230a``, the only lineage carrying the Llama configs; each root's
``tree_provenance.json`` (verified instrument-identical to the other chains' pin by AST diff) is embedded VERBATIM under
``meta.roots.<root>.tree_provenance`` with its sha256, so it survives into the export's sources.

TOLERATES A LEG STILL LANDING. A missing aggregate is ``attacks_pending`` for that view (its cells' readouts become PENDING records,
its ``vs_bare`` pending), an aggregate mid-write (truncated JSON) is treated the same with a warning, and a re-run picks the leg up.
Nothing is hard-coded around a missing file.

usage: build_transfer_stats_full22_llama8b.py [--out <json>] [--rs <run-state/full22>]
       (readers need numpy: run with `uv run --python 3.12 --exclude-newer 2026-09-01T00:00:00Z --with numpy python <this>`;
        the readers live in $STAGE/sdfbase/analysis*, imported through the w7 builder — never re-typed)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HOME = Path.home()
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from site_paths import site_path  # noqa: E402
# the run-state directory and the w7 builder's directory, under the site's state root ($STATE)
RS = site_path("$STATE/romp/candidate-si-refit/run-state/full22", "RS")
W7 = site_path("$STATE/romp/candidate-si-refit/w7_rest", "W7")
sys.path.insert(0, str(W7))
import build_transfer_stats_full22_all as B  # noqa: E402  (the w7 builder: readers, labels, block formatters — imported, never re-typed)

#: the pairing-allowance module (src/sisteer/eval/anchor_pairing.py): the allowance CONSTANT and
#: the verdict rule are IMPORTED, never retyped; an import failure REFUSES the build by name. Resolved from THIS FILE's repo root rather
#: than an absolute $HOME path, so the builder imports the module of the checkout it is actually run from.
ANCHOR_PAIRING_SRC = Path(__file__).resolve().parents[3] / "src"
sys.path.insert(0, str(ANCHOR_PAIRING_SRC))
try:
    from sisteer.eval import anchor_pairing as AP  # noqa: E402
except Exception as _e:  # noqa: BLE001
    raise SystemExit(f"REFUSING: cannot import sisteer.eval.anchor_pairing from {ANCHOR_PAIRING_SRC} ({type(_e).__name__}: {_e}) -- the "
                     f"relu_native pairing stamp is imported from the lane's module, never retyped")

T1, R2 = B.T1, B.R2
MULTITURN, WAVE_LENS_MODE = B.MULTITURN, B.WAVE_LENS_MODE
CAP = 4096                        # the reply cap of EVERY readout in this lane (multi-turn proved per cell; BoN corroborated, see docstring)
CAP_FILED = 512                   # the wave's filed BoN / multi-turn cap — a max completion at or under this would be indistinguishable from a 512 store
NS = "llama8b"
PINNED_JUDGE, PINNED_RUBRIC = "claude-haiku-4-5", "strongreject_v1"
GRADING_TREE_SHA = "390b230aeaff4788f71b8ddf34587dcc02aaa79c"   # llama_grade/SHA; every root's grading_provenance.worktree.head must equal it
POOL_SHA = "198172f8d3e3d49dd3a6050afe1ff534c2e5e2c82589edb3d07d1938950cd319"   # the campaign's 259-goal pool (llama_grade/goalsets/POOL.txt)

#: the four roots. The VIEW is the target directory inside each root (they happen to coincide with the root suffix here, but the cap-4,096
#: roots taught us never to derive the view from the root's name). tag -> (kind, lane description)
ROOTS = {
    "grades_llama8b_llama8b": ("main", "base→instruct transfer lane: objects fit on the base model, served on the instruct checkpoint"),
    "grades_llama8b_llama8b_it": ("it", "instruct-direct lane: objects fit on the instruct checkpoint it is served on"),
    "grades_llama8b_llama8b_shortconst": ("prompted", "prompted-only lane: the unsteered model beside the constitution SI as the system prompt (short form)"),
    "grades_llama8b_llama8b_fullconst": ("prompted", "prompted-only lane: the unsteered model beside the FULL constitution as the system prompt"),
}
TRANSCRIPTS_ROOT = "transcripts_llama8b_lane"        # the pulled private root: <root>/sdfbase_full22_<view>/ (llama_grade/lane_env.sh PRIV)
#: the cells and attacks each view carries = the STAGED config's explicit_cells and the chain's default attacks (llama_grade/lane_env.sh,
#: verified 2026-09-13 against the stores). An aggregate carrying a cell outside this list, or missing one, is a defect, not a merge.
EXPECTED_CELLS = {
    "llama8b": ["none", "constv2-mix-full__lora_r4_ce", "constv2ctl2-full__lora_r4_ce", "constv2-mix-full__lora_r4_ce_minus_ctl", "prompted__safety_refusal"],
    "llama8b_it": ["none", "constv2-mix-full__lora_r4_ce", "constv2ctl2-full__lora_r4_ce", "constv2-mix-full__lora_r4_ce_minus_ctl", "prompted__safety_refusal"],
    "llama8b_shortconst": ["none", "prompted__si_constitution"],
    "llama8b_fullconst": ["none", "prompted__si_constitution_full"],
}
# llama8b_shortconst gained its three multi-turn stores on 2026-09-14 (job 103624: crescendo / pair / actorattack x
# {none, prompted__si_constitution}, 259/259 goals each, 0 errors, attacker identity engine-attested and UNIFORM over
# 18 shards per attack). Until then the lane was BoN-only, so this table said so; the grader had already written
# two_si_{crescendo,pair,actorattack}_aggregate.json into the root and they were silently not loaded. llama8b_fullconst
# stays BoN-only on purpose -- it has no multi-turn store (the pull reports mt_shards=0 for it).
EXPECTED_ATTACKS = {"llama8b": ["bon", *MULTITURN], "llama8b_it": ["bon", *MULTITURN],
                    "llama8b_shortconst": ["bon", *MULTITURN], "llama8b_fullconst": ["bon"]}
#: the errored-goal ledger (tree_provenance.json / llama_grade/README.md): (view, attack, cell) -> n_errored. Any other cell with an errored
#: outcome, or a different count here, REFUSES: the reduced denominators are accepted only where they were accepted by the operator.
ERRORED_EXPECTED = {
    ("llama8b", "crescendo", "constv2-mix-full__lora_r4_ce_minus_ctl"): 1,
    ("llama8b_it", "crescendo", "constv2-mix-full__lora_r4_ce"): 5,
    ("llama8b_it", "crescendo", "constv2ctl2-full__lora_r4_ce"): 8,
}
#: the six PSR ActorAttack cells deliberately not graded (partial stores after the roster drop; PSR is out of scope). The goal counts
#: are read from llama_grade/inventory.json at build time and checked against these, so the ledger and the pre-flight cannot drift apart.
NOT_GRADED_EXPECTED = {
    ("llama8b", "actorattack", "constv2-mix-full__psr_ce_all"): 189,
    ("llama8b", "actorattack", "constv2-mix-full__psr_ce_all_minus_ctl_distilled"): 177,
    ("llama8b", "actorattack", "constv2ctl2-full__psr_ce_all"): 185,
    ("llama8b_it", "actorattack", "constv2-mix-full__psr_ce_all"): 120,
    ("llama8b_it", "actorattack", "constv2-mix-full__psr_ce_all_minus_ctl_distilled"): 125,
    ("llama8b_it", "actorattack", "constv2ctl2-full__psr_ce_all"): 106,
}
NOT_GRADED_REASON = ("partial store: never resumed after the PSR roster drop (D214); not in the served cell list, so the grading chain never asked "
                     "for it; PSR is out by user ruling. Absent by design, not missing.")

#: THE relu_native BEST-OF-75 (relu-steering-native-evals, graded 2026-09-16): the three PSR cells regenerated on the NATIVE multi-LoRA
#: engine (lora_native; identity proxy_vllm+lora+interleaved; lens_reduction_mode null) into the Llama lane's own instrument -- best-of-75 at
#: the 4,096 cap over the 100 confirmatory goals -- in a root of its OWN under the wave directory relu_native/. A REGISTRY, one entry per
#: root: the view it must hold, the attacks THIS builder reads from it (bon only -- the same root's crescendo / pair / actorattack
#: aggregates are the relu_native multi-turn wave and are read by cap4096_export/build_transfer_stats_cap4096.py, never twice), the cells
#: the aggregate must carry, the transcripts root beside it, the serving the records must state, and the unsteered ANCHOR the cells pair
#: against: the Llama lane's own lens-era `none` at the same cap and decode (reuse the existing anchors; the pairing verdict is
#: computed from both stores' bon_served_run.json and stamped on the vs_bare block). The lane's grading-tree pin (GRADING_TREE_SHA) does not
#: apply: this root was graded by the lane's own native-aware grader (grading_provenance.driver / worktree, recorded per root, not pinned).
NS_RELU = "relu_native"
LENS_SERVING = {"engine": None, "lens_reduction_mode": WAVE_LENS_MODE}       # the Llama lane's lens-era stores: no engine field, lens mode stacked
NATIVE_SERVING = {"engine": "lora_native", "lens_reduction_mode": None}
RELU_ROOTS = {
    "relu_native/grades_relu_native_bu_llama8b": {
        "view": "llama8b", "attacks": ["bon"], "cells": ["constv2-mix-full__psr_ce_all", "constv2ctl2-full__psr_ce_all",
                                                         "constv2-mix-full__psr_ce_all_minus_ctl_distilled"],
        "transcripts_root": "relu_native/transcripts_relu_native_bu_llama8b", "serving": NATIVE_SERVING, "namespace": NS_RELU,
        "anchor_root": "grades_llama8b_llama8b", "anchor_transcripts_root": TRANSCRIPTS_ROOT, "anchor_cell": "none",
        "note": ("bon only: the root's multi-turn aggregates are the relu_native multi-turn wave, read by cap4096_export (namespace relu_native); "
                 "the PSR ActorAttack `not_graded` ledger below (the lens-era partial stores) stands untouched beside them")},
}

#: cell labels for the views the w7 builder does not know. The it view carries two cells IT_LABEL lacks (the control LoRA and the LoRA
#: subtraction); they are instruct-direct objects by the it view's construction, and are labelled as such.
PROMPTED_LABEL = "PROMPTED safety SI (safety_refusal as the system prompt; no object)"
IT_EXTRA_LABEL = {
    "constv2ctl2-full__lora_r4_ce": "INSTRUCT-DIRECT BUDGET-MATCHED CONTROL · LoRA r4 (CE, control corpus; fit on the instruct checkpoint)",
    "constv2-mix-full__lora_r4_ce_minus_ctl": "DERIVED: instruct-direct constitution LoRA minus instruct-direct control LoRA (not on the overlay)",
}
IT_EXTRA_GROUP = {"constv2ctl2-full__lora_r4_ce": "budget_matched_control", "constv2-mix-full__lora_r4_ce_minus_ctl": "derived"}
PROMPTED_CONST_LABEL = {
    "prompted__si_constitution": "PROMPTED constitution SI (the shortconst lane's system prompt; no object)",
    "prompted__si_constitution_full": "PROMPTED constitution SI, FULL text (the fullconst lane's system prompt; no object)",
}


def sha256_file(p) -> str | None:
    p = Path(p)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def load_json(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def load_json_tolerant(p, warnings: list):
    """A grades aggregate is written whole at the end of a grading leg, but this builder may run while a leg is landing: a truncated file
    is treated as ABSENT (pending) with a warning, never as a refusal — a re-run picks it up."""
    p = Path(p)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError as e:
        warnings.append(f"{p}: unreadable JSON ({e}) — treated as a leg still landing; re-run to pick it up")
        return None


def ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def only_dir(p: Path) -> Path:
    """The single subdirectory of p (roots carry a .grade.lock beside it) — refuse on 0 or 2+."""
    ds = sorted(x for x in p.iterdir() if x.is_dir() and not x.name.startswith("."))
    if len(ds) != 1:
        raise SystemExit(f"{p}: expected exactly one directory inside, found {[d.name for d in ds]}")
    return ds[0]


def label_group(kind: str, cell: str) -> tuple[str, str]:
    if cell == "prompted__safety_refusal":
        return PROMPTED_LABEL, "prompted_baseline"
    if cell in PROMPTED_CONST_LABEL:
        return PROMPTED_CONST_LABEL[cell], "prompted_baseline"
    if kind == "it":
        return (B.IT_LABEL.get(cell) or IT_EXTRA_LABEL.get(cell) or f"{cell} (extra)"), (B.IT_GROUP.get(cell) or IT_EXTRA_GROUP.get(cell) or "extra")
    if kind == "prompted":
        # the prompted-only lanes' bare cell is the same served checkpoint as the main view's, in its own serving job
        return ("none (bare instruct, same wave, prompted-only lane's serving job)" if cell == "none" else f"{cell} (extra)"), ("baseline_full22" if cell == "none" else "extra")
    return B.CELL_LABEL.get(cell, f"{cell} (extra)"), B.CELL_GROUP.get(cell, "extra")


# --------------------------------------------------------------------------- the producers' own records (numbers only)

def count_lines(p: Path) -> int:
    with open(p, "rb") as f:
        return sum(1 for line in f if line.strip())


def verify_cap_mt(cell_dir: Path, completion: dict) -> dict:
    """The cap the multi-turn replies were ACTUALLY generated under, from the shards' target_timing.jsonl (numbers only) — the cap-4,096
    builder's proof, re-applied: every call's requested max_tokens, the max completion_tokens, the finish_reason == length count, and the
    identity of this store with the graded one (timing n / finish_length / responses.jsonl rows == the grader's completion stamp)."""
    n = n_len = mx = n_resp = 0
    max_tokens_seen: dict = {}
    n_clamped = n_clamped_len = 0
    shards = sorted(sd for sd in cell_dir.glob("shard*") if sd.is_dir())
    for sd in shards:
        tp = sd / "target_timing.jsonl"
        if tp.exists():
            with open(tp) as f:
                for line in f:
                    if not line.strip():
                        continue
                    t = json.loads(line)
                    ct = int(t.get("completion_tokens") or 0)
                    mt = t.get("max_tokens")
                    n += 1; mx = max(mx, ct)
                    n_len += (t.get("finish_reason") == "length")
                    max_tokens_seen[mt] = max_tokens_seen.get(mt, 0) + 1
                    if mt is not None and int(mt) < CAP:
                        n_clamped += 1
                        n_clamped_len += (t.get("finish_reason") == "length")
        rp = sd / "responses.jsonl"
        if rp.exists():
            n_resp += count_lines(rp)
    tt = completion.get("target_timing") or {}
    if n != tt.get("n") or n_len != tt.get("finish_length"):
        raise SystemExit(f"{cell_dir}: timing rows {n} / finish_length {n_len} != grader stamp {tt} — not the graded store, refusing")
    if n_resp != completion.get("n_responses"):
        raise SystemExit(f"{cell_dir}: responses.jsonl rows {n_resp} != grader stamp n_responses {completion.get('n_responses')} — refusing")
    known = [int(k) for k in max_tokens_seen if k is not None]
    if not known or max(known) != CAP:
        raise SystemExit(f"{cell_dir}: requested max_tokens seen {sorted(max_tokens_seen)} — the cap requested was not {CAP}; refusing")
    if mx > CAP:
        raise SystemExit(f"{cell_dir}: a completion of {mx} tokens exceeds the {CAP} cap; refusing")
    if mx <= CAP_FILED:
        raise SystemExit(f"{cell_dir}: max completion_tokens {mx} <= {CAP_FILED} — indistinguishable from a {CAP_FILED}-cap store; refusing")
    return {"kind": "per-row proof", "source": "shard*/target_timing.jsonl (max_tokens, completion_tokens, finish_reason) + responses.jsonl line count; no text read",
            "n_timing_rows": n, "requested_max_tokens": {str(k): v for k, v in sorted(max_tokens_seen.items(), key=lambda kv: (kv[0] is None, kv[0]))},
            "max_completion_tokens": mx, "n_finish_length": n_len, "n_calls_window_clamped": n_clamped, "n_clamped_calls_censored": n_clamped_len,
            "n_responses_rows": n_resp, "n_shards": len(shards),
            "verdict": (f"every call requested max_tokens {CAP}" + (f" except {n_clamped} window-clamped calls" if n_clamped else "")
                        + f"; max completion {mx}; generated under a {CAP}-token reply cap")}


def instrument_mt(cell_dir: Path) -> dict:
    """Serving identity and the decode declared for the run, from every shard's served_target.json — one distinct tuple across shards or
    refuse. Keyed on the DECLARED decode block (``decode``: do_sample / T / top_p / cap / arm / source), not on ``last_request_decode``:
    that field is the last request a shard happened to send and is null on a shard whose last call recorded none (seen on two shards of
    this lane); where it IS recorded it must agree with the declared block, checked separately."""
    tuples: dict = {}
    counters = {"n_window_clamped": 0, "n_prompt_over_window": 0, "n_window_mismatch": 0}
    n_last_recorded = n_last_disagree = 0
    for st in sorted(cell_dir.glob("shard*/served_target.json")):
        s = json.loads(st.read_text())
        wc = s.get("window_clamp") or {}
        d = s.get("decode") or {}
        key = json.dumps({"base_model": s.get("base_model"), "serving_window_tokens": s.get("max_model_len"),
                          "serving_identity": s.get("serving_identity"),
                          "decode": {k: d.get(k) for k in ("do_sample", "temperature", "top_p", "top_k", "repetition_penalty", "cap", "arm", "seed_rule", "source", "source_kind")},
                          "lens_reduction_mode": (s.get("proxy_identity") or {}).get("lens_reduction_mode")}, sort_keys=True)
        tuples[key] = tuples.get(key, 0) + 1
        for k in counters:
            counters[k] += int(wc.get(k) or 0)
        ld = s.get("last_request_decode")
        if ld:
            n_last_recorded += 1
            if (ld.get("do_sample"), ld.get("temperature"), ld.get("top_p")) != (d.get("do_sample"), d.get("temperature"), d.get("top_p")):
                n_last_disagree += 1
    if len(tuples) != 1:
        raise SystemExit(f"{cell_dir}: served_target.json serving identity not uniform across shards — {len(tuples)} distinct tuples: {list(tuples)[:2]}")
    if n_last_disagree:
        raise SystemExit(f"{cell_dir}: last_request_decode disagrees with the declared decode on {n_last_disagree} shard(s); refusing")
    inst = json.loads(next(iter(tuples)))
    inst["n_shards"] = next(iter(tuples.values()))
    inst.update(counters)
    inst["n_shards_last_request_decode_recorded"] = n_last_recorded
    if inst["n_prompt_over_window"] or inst["n_window_mismatch"]:
        raise SystemExit(f"{cell_dir}: prompt-over-window {inst['n_prompt_over_window']} / window mismatch {inst['n_window_mismatch']} — goals lost or identity broken; refusing")
    if inst["lens_reduction_mode"] != WAVE_LENS_MODE:
        raise SystemExit(f"{cell_dir}: served lens_reduction_mode {inst['lens_reduction_mode']} != {WAVE_LENS_MODE}")
    if inst["decode"].get("cap") != CAP:
        raise SystemExit(f"{cell_dir}: declared decode cap {inst['decode'].get('cap')} != {CAP}")
    d = inst["decode"]
    inst["decode_mode"] = "greedy" if d.get("do_sample") is False or d.get("temperature") == 0.0 else "shipped_sampling"
    inst["decode_cap_tokens"] = CAP
    return inst


def verify_cap_bon(cell_dir: Path, completion: dict, serving: dict = LENS_SERVING) -> tuple[dict, dict]:
    """BoN: the store carries NO per-row token counts, so the cap cannot be proved per row. What CAN be checked, and is: the store's own
    served-run record and manifest agree with the grader's completion stamp on row count / finish reasons / goals / n_max, zero window
    clamps, the serving window recorded, and every row's engine is the served vLLM. Returns (verification, instrument)."""
    sr = load_json(cell_dir / "bon_served_run.json") or {}
    sm = load_json(cell_dir / "bon_served_manifest.json") or {}
    if not sr or not sm:
        raise SystemExit(f"{cell_dir}: bon_served_run.json / bon_served_manifest.json absent; refusing")
    n_rows = count_lines(cell_dir / "bon_responses.jsonl")
    n_err = count_lines(cell_dir / "bon_errors.jsonl") if (cell_dir / "bon_errors.jsonl").exists() else None
    fr = completion.get("finish_reasons") or {}
    if n_rows != completion.get("n_rows") or n_rows != sr.get("n_target_forwards") or (sm.get("units") or {}).get("n_written") != n_rows:
        raise SystemExit(f"{cell_dir}: bon_responses.jsonl rows {n_rows} != grader stamp n_rows {completion.get('n_rows')} / served n_target_forwards "
                         f"{sr.get('n_target_forwards')} / manifest n_written {(sm.get('units') or {}).get('n_written')} — refusing")
    if sum(fr.values()) != n_rows:
        raise SystemExit(f"{cell_dir}: grader finish_reasons {fr} do not sum to {n_rows}; refusing")
    if n_err not in (None, completion.get("n_error_rows")):
        raise SystemExit(f"{cell_dir}: bon_errors.jsonl rows {n_err} != grader stamp n_error_rows {completion.get('n_error_rows')}; refusing")
    if sr.get("n_goals") != completion.get("n_goals") or int(sr.get("n_max") or 0) != int(T1.BON_N):
        raise SystemExit(f"{cell_dir}: served n_goals {sr.get('n_goals')} / n_max {sr.get('n_max')} vs stamp n_goals {completion.get('n_goals')} / BON_N {T1.BON_N}; refusing")
    cr = sr.get("clamp_report") or {}
    if cr.get("n_clamped") or cr.get("n_prompt_over_window") or cr.get("n_window_mismatch"):
        raise SystemExit(f"{cell_dir}: BoN window clamp report {cr} — clamped or lost draws; refusing")
    if sr.get("lens_reduction_mode") != serving["lens_reduction_mode"] or sr.get("smoke"):
        raise SystemExit(f"{cell_dir}: served lens_reduction_mode {sr.get('lens_reduction_mode')} / smoke {sr.get('smoke')}; refusing")
    if sm.get("engine") != "vllm_served":
        raise SystemExit(f"{cell_dir}: manifest engine {sm.get('engine')} is not vllm_served; refusing")
    if serving["engine"] and (sr.get("serving_engine"), sm.get("serving_engine"), sr.get("require_engine")) != (serving["engine"],) * 3:
        # a NATIVE root: the served-run record, the manifest and the run's own require_engine must all state the declared engine
        raise SystemExit(f"{cell_dir}: bon_served_run serving_engine / require_engine {sr.get('serving_engine')} / {sr.get('require_engine')} and manifest "
                         f"serving_engine {sm.get('serving_engine')} != the root's declared {serving['engine']!r}; refusing")
    ver = {"kind": "config stamp + serving-window corroboration (no per-row token counts in a BoN store)",
           "source": "bon_served_run.json / bon_served_manifest.json / bon_responses.jsonl line count / bon_errors.jsonl line count; no text read",
           "n_responses_rows": n_rows, "n_error_rows_file": n_err, "grader_finish_reasons": fr, "n_finish_length": int(fr.get("length", 0)),
           "serving_window_tokens": sr.get("max_model_len"), "window_clamp": {k: cr.get(k) for k in ("n_clamped", "n_prompt_over_window", "n_window_mismatch", "n_target_forwards")},
           "verdict": (f"the config's bon_max_new_tokens is {CAP} (grading_provenance.decode_caps); the serving window is {sr.get('max_model_len')} tokens with 0 clamps; "
                       f"{n_rows} rows = the grader's stamp; the cap is NOT proved per row — the BoN store records no completion_tokens")}
    pi = sr.get("proxy_identity") or {}
    inst = {"base_model": pi.get("base_model"), "serving_window_tokens": sr.get("max_model_len"), "serving_identity": sr.get("serving_identity"),
            "decode": dict(sr.get("decode") or {}), "bon_decode_preset": sr.get("bon_decode_preset"), "seed_rule": sm.get("seed_rule"),
            "engine": sm.get("engine"), "lens_reduction_mode": sr.get("lens_reduction_mode"), "decode_mode": "sampled_T1_harness_default",
            "decode_cap_tokens": CAP, "virtual_model": sr.get("virtual_model"), "intervention_hash": sr.get("intervention_hash"),
            "n_window_clamped": int(cr.get("n_clamped") or 0), "n_prompt_over_window": int(cr.get("n_prompt_over_window") or 0), "n_window_mismatch": int(cr.get("n_window_mismatch") or 0)}
    if serving["engine"]:
        inst["serving_engine"] = sr.get("serving_engine")             # only on a root that declares an engine (the lens records stay byte-identical)
    return ver, inst


# --------------------------------------------------------------------------- one cell x attack

def read_mt_cell(agg: dict, view: str, cell: str, attack: str, cell_dir: Path) -> tuple[dict, dict]:
    rec = R2._safe(T1.gated_cell, agg, cell)
    if rec.get("status"):
        raise SystemExit(f"{cell_dir}: reader returned {rec} — the grading is supposed to be complete; refusing")
    cellrow = agg["cells"][cell]
    if cellrow.get("lens_reduction_mode") != WAVE_LENS_MODE:
        raise SystemExit(f"{cell_dir}: aggregate row lens_reduction_mode != {WAVE_LENS_MODE}")
    completion = (cellrow.get("completion") or {}).get("gated") or {}
    m = cellrow["attack_metrics"]
    # THE ERRORED-GOAL ACCOUNTING. n_expected / n_graded / n_errored are the grader's; they must agree with each other, with the per_goal
    # size (= the reader's n), with the completion stamp, and with the operator's ledger — or refuse. A reduced denominator is accepted
    # only where it was accepted with --allow-errored-goals.
    n_exp, n_gr, n_er = m.get("n_expected"), m.get("n_graded"), m.get("n_errored")
    if None in (n_exp, n_gr, n_er) or n_exp - n_gr != n_er or n_gr != rec["n"] or len(m.get("errored_goal_ids") or []) != n_er \
            or sum((m.get("errored_by_type") or {}).values()) != n_er or completion.get("n_error_outcomes") != n_er:
        raise SystemExit(f"{cell_dir}: errored-goal accounting inconsistent: n_expected {n_exp} n_graded {n_gr} n_errored {n_er} per_goal {rec['n']} "
                         f"errored_goal_ids {len(m.get('errored_goal_ids') or [])} errored_by_type {m.get('errored_by_type')} completion.n_error_outcomes {completion.get('n_error_outcomes')}")
    if n_er != ERRORED_EXPECTED.get((view, attack, cell), 0):
        raise SystemExit(f"{cell_dir}: {n_er} errored goal outcomes but the ledger (ERRORED_EXPECTED) says {ERRORED_EXPECTED.get((view, attack, cell), 0)}; refusing")
    if n_exp != 259:
        raise SystemExit(f"{cell_dir}: n_expected {n_exp} != the campaign's 259-goal pool; refusing")
    if m.get("n_jailbroken") != rec["k"]:
        raise SystemExit(f"{cell_dir}: the reader's k {rec['k']} (per_goal at > {T1.THRESHOLD}) != the grader's n_jailbroken {m.get('n_jailbroken')}; refusing")
    if cellrow.get("decode_cap_tokens") != CAP:
        raise SystemExit(f"{cell_dir}: grader stamp decode_cap_tokens {cellrow.get('decode_cap_tokens')} != {CAP} (the Llama configs set target_max_new_tokens {CAP}); refusing")
    rec["namespace"] = NS
    rec["lens_reduction_mode"] = WAVE_LENS_MODE
    rec["tier"] = "multi_turn"
    rec["decode_cap_tokens"] = CAP
    rec["decode_cap_tokens_grader_stamp"] = cellrow.get("decode_cap_tokens")
    rec["decode_cap_stamp_note"] = ("the grader copies attacks.target_max_new_tokens from the config it is handed; for this lane the config says 4096 and the "
                                    "run agrees — proved per cell in decode_cap_verification (unlike the cap-4,096 wave, whose config said 512)")
    rec["decode_cap_verification"] = verify_cap_mt(cell_dir, completion)
    rec["instrument"] = instrument_mt(cell_dir)
    if rec["decode_cap_verification"]["n_calls_window_clamped"] != rec["instrument"]["n_window_clamped"]:
        raise SystemExit(f"{cell_dir}: clamped calls per target_timing != served_target window_clamp.n_clamped summed over shards — refusing")
    rec["completion"] = completion
    rec["mean_n_target_forwards"] = m.get("mean_n_target_forwards")
    rec["n_capped_goals"] = m.get("n_capped")
    rec["n_expected"], rec["n_graded"], rec["n_errored"] = n_exp, n_gr, n_er
    rec["errored_by_type"] = dict(m.get("errored_by_type") or {})
    rec["errored_goal_ids"] = sorted(m.get("errored_goal_ids") or [])          # goal IDS only (no text): which goals the denominator dropped
    rec["errored_goals_note"] = (None if not n_er else
                                 f"graded with --allow-errored-goals: {n_er} of {n_exp} goal-outcomes errored ({rec['errored_by_type']}; PyRIT's executor wrapper, per-goal "
                                 f"cause unrecorded) and are DROPPED from k and n; the readout is over the {n_gr} goals that returned a verdict. A resume is impossible "
                                 f"on this store (the recorded attacker identity is unverifiable, so the harness refuses before comparing identities)")
    rec["n_responses_graded"] = m.get("n_responses_graded")
    rec["goalset"] = m.get("goalset")
    rec["by_category_group"] = {g: {"n_goals": v["n_goals"], "n_jailbroken": v["n_jailbroken"], "asr": v["asr"]}
                                for g, v in (m.get("by_category_group") or {}).items()}
    tt = completion.get("target_timing") or {}
    ac = T1.prop(int(tt.get("finish_length", 0)), int(tt.get("n", 0)))
    ac["definition"] = "producer finish_reason == length (target_timing.jsonl, via the grader's completion.gated.target_timing stamp)"
    ac["decode_cap_tokens"] = CAP
    rec["at_cap_finish_length"] = ac
    rec["degeneracy"] = B.degeneracy_block(None)
    rec["degeneracy"]["reason"] = "no degeneracy join (degeneracy_join.py) has been run over the Llama roots; at-cap rides at_cap_finish_length"
    rec["asr_annotated"] = B.annotate(rec, None) + f" | at-cap {B.pct(ac['p'])}% of {ac['n']} replies (finish_reason == length @ {CAP})" + (f" | {n_er} errored goals dropped" if n_er else "")
    return rec, R2.attack_outcomes(agg, cell)


def read_bon_cell(agg: dict, view: str, cell: str, cell_dir: Path, ns: str = NS, serving: dict = LENS_SERVING) -> tuple[dict, dict]:
    rec = R2._safe(T1.bon_cell, agg, cell)
    if rec.get("status"):
        raise SystemExit(f"{cell_dir}: reader returned {rec}; refusing")
    cellrow = agg["cells"][cell]
    if cellrow.get("lens_reduction_mode") != serving["lens_reduction_mode"]:
        raise SystemExit(f"{cell_dir}: aggregate row lens_reduction_mode != {serving['lens_reduction_mode']}")
    if serving["engine"] and (cellrow.get("serving_engine"), agg.get("serving_engine"), agg.get("require_engine")) != (serving["engine"],) * 3:
        raise SystemExit(f"{cell_dir}: aggregate serving_engine / require_engine {(cellrow.get('serving_engine'), agg.get('serving_engine'), agg.get('require_engine'))} "
                         f"!= the root's declared {serving['engine']!r}; refusing")
    completion = (cellrow.get("completion") or {}).get("bon") or {}
    if completion.get("n_error_rows"):
        raise SystemExit(f"{cell_dir}: {completion.get('n_error_rows')} errored BoN rows — not expected in these roots; refusing")
    if cellrow.get("decode_cap_tokens") != CAP:
        raise SystemExit(f"{cell_dir}: grader stamp decode_cap_tokens {cellrow.get('decode_cap_tokens')} != {CAP} (the Llama configs set bon_max_new_tokens {CAP}); refusing")
    m = cellrow["attack_metrics"]
    rec["namespace"] = ns
    rec["lens_reduction_mode"] = serving["lens_reduction_mode"]
    rec["tier"] = "bon"
    rec["decode_cap_tokens"] = CAP
    rec["decode_cap_tokens_grader_stamp"] = cellrow.get("decode_cap_tokens")
    rec["decode_cap_verification"], rec["instrument"] = verify_cap_bon(cell_dir, completion, serving)
    if serving["engine"]:
        rec["serving_engine"] = serving["engine"]
        rec["serving_identity"] = rec["instrument"].get("serving_identity")
    rec["completion"] = completion
    # per category group: n_jailbroken recovered from n_star_by_goal at N = 75 the way bon_cell recovers the whole-set numerator
    rec["by_category_group"] = {g: {"n_goals": v["n_goals"], "n_jailbroken": sum(1 for x in (v.get("n_star_by_goal") or {}).values() if 0 < int(x) <= int(T1.BON_N)),
                                    "asr_at_n": v.get("asr_at_n")} for g, v in (m.get("by_category_group") or {}).items()}
    fr = completion.get("finish_reasons") or {}
    ac = T1.prop(int(fr.get("length", 0)), int(completion.get("n_rows") or 0))
    ac["definition"] = "producer finish_reason == length over the 7,500 best-of-75 draws (the grader's completion.bon.finish_reasons stamp)"
    ac["decode_cap_tokens"] = CAP
    rec["at_cap_finish_length"] = ac
    rec["degeneracy"] = B.degeneracy_block(None)
    rec["degeneracy"]["reason"] = "no degeneracy join has been run over the Llama roots; at-cap rides at_cap_finish_length"
    rec["asr_annotated"] = B.annotate(rec, None) + f" | at-cap {B.pct(ac['p'])}% of {ac['n']} draws (finish_reason == length @ {CAP})"
    return rec, R2.bon_outcomes(agg, cell)


def contrast_with_intersection(row_a: dict, row_b: dict, pa: dict, pb: dict, label: str) -> dict:
    """R2.contrast (cell b minus bare a) — and when the reader REFUSES the paired part because the two sides do not carry the same goal ids
    (a cell that dropped errored goals against a bare with all 259), the same reader on the goal INTERSECTION, labelled."""
    vb = R2.contrast(row_a, row_b, pa, pb)
    if isinstance(vb.get("paired_refused"), dict) and pa and pb:
        shared = sorted(set(pa) & set(pb))
        if shared:
            pc = T1.paired_contrast({g: pa[g] for g in shared}, {g: pb[g] for g in shared})
            pc.update({"n_paired": len(shared), "n_a_only": len(set(pa) - set(pb)), "n_b_only": len(set(pb) - set(pa)),
                       "note": f"paired on the intersection of goals present on both sides ({label}); same reader, same bootstrap and McNemar"})
            vb["paired_on_intersection"] = pc
    return vb


# --------------------------------------------------------------------------- main

# Attacker identity PER (root, attack, cell), read from the store's own shard sidecars (the distinct-provenance-class rule; same
# 8-field tuple as the grader's check). Every 4,096 multi-turn sidecar in this campaign records attacker_quantization='unrecorded'
# (attacker_pool.sbatch never passes --attacker-serving), the harness refuses to aggregate on it and the grader records that refusal as a
# caveat and proceeds -- so these rows were graded under an identity ATTESTED FROM ELSEWHERE (the config's attacker block + WAVE_MANIFEST
# attacker_pool), and the row must say so. Tolerant here where the cap-4,096 builder is strict: this lane's ledger already documents partial
# stores, so a cell with no sidecars is CLASSED as such, never refused. Module level on purpose: a column-0 helper pasted inside main() compiles
# and silently truncates it (measured today).
ATTACKER_ID_FIELDS = ("attacker_engine", "attacker_model", "attacker_snapshot", "attacker_quantization",
                      "attacker_prefix_caching", "attacker_kv_cache_dtype", "scorer_engine", "target_engine")
def attacker_identity_of(cell_dir, gp) -> dict:
    tuples: dict = {}; n = 0; missing = 0
    for sd in sorted(p for p in Path(cell_dir).glob("shard*") if p.is_dir()):
        n += 1; ae = sd / "attacker_engine.json"
        if not ae.is_file():
            missing += 1; continue
        a = load_json(ae) or {}
        k = json.dumps({f: a.get(f) for f in ATTACKER_ID_FIELDS}, sort_keys=True); tuples[k] = tuples.get(k, 0) + 1
    aic = (gp or {}).get("attacker_identity_check") or {}
    att = next((e.get("independent_check", {}).get("attested_elsewhere") for e in aic.values() if isinstance(e, dict)), None)
    base = {"n_shards": n, "n_shards_without_sidecar": missing, "attested_elsewhere": att,
            "attested_source": ("the root's grading_provenance.json attacker_identity_check (config attacker block + WAVE_MANIFEST attacker_pool)" if att else None),
            "sidecar_source": str(Path(cell_dir) / "shard*/attacker_engine.json")}
    if n == 0 or missing == n:
        return {**base, "provenance_class": "unknown-no-sidecars", "recorded_identity": None, "fields_unrecorded_in_sidecar": None}
    if len(tuples) != 1 or missing:
        return {**base, "provenance_class": "mixed-or-partial-sidecars", "recorded_identity": None, "fields_unrecorded_in_sidecar": None,
                "distinct_recorded_tuples": len(tuples)}
    ident = json.loads(next(iter(tuples))); unrec = [f for f, v in ident.items() if v in (None, "unrecorded", "unknown")]
    return {**base, "provenance_class": ("attested-from-elsewhere" if unrec else "sidecar-recorded"), "recorded_identity": ident,
            "fields_unrecorded_in_sidecar": unrec}


def main(argv=None) -> None:
    global RS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(RS / "transfer_stats_full22_llama8b.json"))
    ap.add_argument("--rs", default=str(RS), help="run-state/full22 (grades_llama8b_* and transcripts_llama8b_lane live here)")
    args = ap.parse_args(argv)
    rs = Path(args.rs)
    RS = rs
    warnings: list = []

    prior_meta = (load_json(rs / "transfer_stats_full22_all.json") or {}).get("meta", {})
    inventory = load_json(rs / "llama_grade/inventory.json")
    if not inventory:
        raise SystemExit(f"{rs / 'llama_grade/inventory.json'} absent — the not-graded ledger needs the chain's pre-flight")
    if inventory.get("pool_sha256") != POOL_SHA:
        raise SystemExit(f"inventory pool_sha256 {inventory.get('pool_sha256')} != the campaign pool {POOL_SHA}")

    models: dict = {}
    roots_meta: dict = {}
    count_rows: list = []
    seen: dict = {}

    for root_name, (kind, lane_desc) in ROOTS.items():
        root = rs / root_name
        if not root.is_dir():
            raise SystemExit(f"grades root {root} absent")
        tdir = only_dir(root)
        view = tdir.name                                        # THE VIEW IS THE TARGET DIRECTORY
        if view not in EXPECTED_CELLS:
            raise SystemExit(f"{root}: target directory {view!r} is not a known Llama view")
        run_dir = rs / TRANSCRIPTS_ROOT / f"sdfbase_full22_{view}"
        if not run_dir.is_dir():
            raise SystemExit(f"{root}: transcripts run {run_dir} absent")
        tp = load_json(tdir / "tree_provenance.json")
        if not tp:
            raise SystemExit(f"{tdir}: tree_provenance.json absent — the grading-tree provenance must travel with the rows")
        if (tp.get("grading_tree") or {}).get("sha") != GRADING_TREE_SHA:
            raise SystemExit(f"{tdir}: tree_provenance grading_tree.sha {tp.get('grading_tree')} != {GRADING_TREE_SHA}")
        gp = load_json(tdir / "grading_provenance.json") or {}
        if gp and (gp.get("worktree") or {}).get("head") != GRADING_TREE_SHA:
            raise SystemExit(f"{tdir}: grading_provenance worktree.head {(gp.get('worktree') or {}).get('head')} != {GRADING_TREE_SHA}")
        if gp and gp.get("decode_caps") != {"bon_max_new_tokens": CAP, "static_max_new_tokens": CAP, "target_max_new_tokens": CAP}:
            raise SystemExit(f"{tdir}: grading_provenance decode_caps {gp.get('decode_caps')} — expected every cap {CAP}")
        j = gp.get("judge") or {}
        if gp and (j.get("judge_model") != PINNED_JUDGE or j.get("rubric") != PINNED_RUBRIC):
            raise SystemExit(f"{tdir}: judge {j.get('judge_model')} / {j.get('rubric')} is not the pinned {PINNED_JUDGE} / {PINNED_RUBRIC}")
        aggs = {a: load_json_tolerant(tdir / f"two_si_{a}_aggregate.json", warnings) for a in EXPECTED_ATTACKS[view]}
        present = [a for a in EXPECTED_ATTACKS[view] if aggs[a]]
        pending = [a for a in EXPECTED_ATTACKS[view] if not aggs[a]]
        if not present:
            warnings.append(f"{tdir}: no aggregate present yet — view {view} carried as all-pending")
        served = None
        mb = {"label": f"NousResearch/Meta-Llama-3.1-8B-Instruct — {view} ({lane_desc}); BoN and multi-turn at a {CAP:,}-token reply cap",
              "view": kind, "base_target": "llama8b", "namespace": NS, "decode_cap_tokens": CAP, "served_model": None, "served_alpha": None,
              "lens_reduction_mode": WAVE_LENS_MODE, "attacks_present": present, "attacks_pending": pending, "expected_cells": list(EXPECTED_CELLS[view]),
              "expected_missing": {}, "cells": {}, "vs_bare": {},
              "vs_bare_note": (f"cell MINUS the same view's own unsteered `none`, at the SAME {CAP}-token cap and the same decode, paired on goals "
                               f"(10,000-draw bootstrap seed 0 + exact McNemar); on the goal INTERSECTION where the cell dropped errored goals"),
              "not_graded": {}, "sources": {}, "judge": {"strongreject": {"per_root": {}, "per_aggregate": {}}}, "completion": {},
              "tree_provenance": tp}
        outcomes: dict = {a: {} for a in EXPECTED_ATTACKS[view]}
        for cell in EXPECTED_CELLS[view]:
            lab, grp = label_group(kind, cell)
            mb["cells"][cell] = {"label": lab, "group": grp, "view": kind, "namespace": NS, "decode_cap_tokens": CAP, "lens_reduction_mode": WAVE_LENS_MODE,
                                 "on_overlay": cell in B.OVERLAY_CELLS, "completion": {}}
        rmeta = {"root": str(root), "view": view, "kind": kind, "transcripts_run": str(run_dir), "attacks_present": present, "attacks_pending": pending,
                 "tree_provenance": tp, "tree_provenance_sha256": sha256_file(tdir / "tree_provenance.json"),
                 "completion_json_sha256": sha256_file(tdir / "completion.json"), "grading_provenance_sha256": sha256_file(tdir / "grading_provenance.json"),
                 "grader": {k: gp.get(k) for k in ("driver", "config", "worktree", "decode_caps", "goalset", "finished_at", "attacks", "cells")},
                 "judge_last_attack": {**{k: j.get(k) for k in ("judge_model", "rubric", "cache_kind", "cache_rows", "cache_rows_seeded", "org", "concurrency_initial", "escalation_rule")},
                                       **{k: gp.get(k) for k in ("org_final", "escalated_to_highprio", "n_429", "wall_s", "n_judgments_estimated", "unscored_retry_events", "per_attack_wall_s", "per_attack_429")},
                                       "note": "grading_provenance.json is overwritten per attack (last attack wins); per-attack judge facts are under aggregates.<attack>.grader_provenance"},
                 "attacker_identity_check": gp.get("attacker_identity_check"), "aggregates": {}}
        mb["judge"]["strongreject"]["per_root"][root_name] = rmeta["judge_last_attack"]
        for attack in present:
            agg = aggs[attack]
            if agg.get("target") != view or agg.get("private_run") != f"sdfbase_full22_{view}" or agg.get("wave") != "full22":
                raise SystemExit(f"{tdir}/{attack}: aggregate target {agg.get('target')!r} / private_run {agg.get('private_run')!r} / wave {agg.get('wave')!r} disagree with view {view!r}")
            if agg.get("attack_tier") != ("bon" if attack == "bon" else "multi_turn") or agg.get("lens_reduction_mode") != WAVE_LENS_MODE:
                raise SystemExit(f"{tdir}/{attack}: tier {agg.get('attack_tier')} / lens {agg.get('lens_reduction_mode')}")
            ag = agg.get("grader_provenance") or {}
            if ag.get("judge_model") != PINNED_JUDGE:
                raise SystemExit(f"{tdir}/{attack}: aggregate grader_provenance.judge_model {ag.get('judge_model')} != {PINNED_JUDGE}")
            if sorted(agg["cells"]) != sorted(EXPECTED_CELLS[view]):
                raise SystemExit(f"{tdir}/{attack}: cells {sorted(agg['cells'])} != the staged config's {sorted(EXPECTED_CELLS[view])}; refusing")
            if attack != "bon":
                gs = agg.get("goalset") or {}
                bad = [f for f, v in (gs.get("files") or {}).items() if v.get("sha256") != POOL_SHA or v.get("n") != 259]
                if bad:
                    raise SystemExit(f"{tdir}/{attack}: goalset files not the 259-goal campaign pool: {bad}")
            rmeta["aggregates"][attack] = {"path": str(tdir / f"two_si_{attack}_aggregate.json"), "sha256": sha256_file(tdir / f"two_si_{attack}_aggregate.json"),
                                           "written_at": agg.get("written_at"), "cells": sorted(agg["cells"]), "served_alpha": agg.get("served_alpha"), "fit_scale": agg.get("fit_scale"),
                                           "category_holdout": agg.get("category_holdout"), "goalset": agg.get("goalset"),
                                           "grader_provenance": {k: ag.get(k) for k in ("judge_model", "cache_kind", "cache_rows_seeded", "concurrency_initial", "org", "escalation_rule", "key_plumbing")}}
            mb["judge"]["strongreject"]["per_aggregate"][attack] = rmeta["aggregates"][attack]["grader_provenance"]
            mb["served_alpha"] = agg.get("served_alpha")
            for cell in EXPECTED_CELLS[view]:
                key = (view, cell, attack)
                if key in seen:
                    raise SystemExit(f"{key} graded in BOTH {seen[key]} and {root_name}; refusing")
                seen[key] = root_name
                cell_dir = run_dir / cell if attack == "bon" else run_dir / attack / cell
                if attack == "bon":
                    rec, oc = read_bon_cell(agg, view, cell, cell_dir)
                else:
                    rec, oc = read_mt_cell(agg, view, cell, attack, cell_dir)
                    rec["root"] = root_name; rec["attacker_identity"] = attacker_identity_of(cell_dir, gp)   # per (root, attack, cell), from the sidecars
                bm = rec["instrument"].get("base_model")
                if served is None:
                    served = bm
                elif bm != served:
                    raise SystemExit(f"{cell_dir}: served base_model {bm!r} != {served!r} within view {view}; refusing")
                mb["cells"][cell][attack] = rec
                mb["cells"][cell]["completion"][attack] = rec["completion"]
                mb["sources"].setdefault(cell, {})[attack] = root_name
                outcomes[attack][cell] = oc
                count_rows.append({"view": view, "attack": attack, "cell": cell, "root": root_name, "k": rec["k"], "n": rec["n"],
                                   "n_expected": rec.get("n_expected"), "n_errored": rec.get("n_errored"),
                                   "at_cap_k": rec["at_cap_finish_length"]["k"], "at_cap_n": rec["at_cap_finish_length"]["n"],
                                   "cap_proof": rec["decode_cap_verification"]["kind"], "max_completion_tokens": rec["decode_cap_verification"].get("max_completion_tokens"),
                                   "serving_window": rec["instrument"].get("serving_window_tokens"), "decode_mode": rec["instrument"].get("decode_mode")})
        for attack in pending:
            for cell in EXPECTED_CELLS[view]:
                mb["cells"][cell][attack] = R2.pending(f"{attack} aggregate not yet written for {view} (leg still landing or not run) — re-run this builder to pick it up")
        # vs_bare per cell x attack (cell minus the same view's bare, same cap, same decode)
        for cell in EXPECTED_CELLS[view]:
            if cell == "none":
                continue
            mb["vs_bare"][cell] = {}
            for attack in EXPECTED_ATTACKS[view]:
                ra, rb = mb["cells"]["none"].get(attack), mb["cells"][cell].get(attack)
                if not ra or not rb or ra.get("status") or rb.get("status"):
                    mb["vs_bare"][cell][attack] = R2.pending("readout pending on one side"); continue
                vb = contrast_with_intersection(ra, rb, outcomes[attack].get("none"), outcomes[attack].get(cell), "the cell dropped errored goals")
                vb["reads_as"] = f"cell MINUS the same view's unsteered `none`, both at cap {CAP} (bare = none, cell = {cell})"
                mb["vs_bare"][cell][attack] = vb
        # the deliberately not-graded PSR ActorAttack cells, from the chain's pre-flight, checked against the ledger
        inv_mt = ((inventory.get("tags") or {}).get(view) or {}).get("multiturn") or {}
        for (v, attack, cell), n_goals in NOT_GRADED_EXPECTED.items():
            if v != view:
                continue
            rec_inv = inv_mt.get(f"{attack}/{cell}") or {}
            if rec_inv.get("n_goals_with_progress") != n_goals:
                raise SystemExit(f"{view}/{attack}/{cell}: inventory says {rec_inv.get('n_goals_with_progress')} goals with progress, the ledger {n_goals}; refusing")
            if not (run_dir / attack / cell).is_dir():
                raise SystemExit(f"{view}/{attack}/{cell}: the partial store is not where the inventory says; refusing")
            for other in MULTITURN:
                if other != attack and (run_dir / other / cell).exists():
                    raise SystemExit(f"{view}/{other}/{cell}: a PSR store exists under {other} too — the not-graded ledger is incomplete; refusing")
            mb["not_graded"].setdefault(cell, {})[attack] = {"status": "NOT_GRADED", "n_goals_with_progress": n_goals, "n_expected": 259,
                                                             "reason": NOT_GRADED_REASON, "inventory_reasons": rec_inv.get("reasons"),
                                                             "source": str(rs / "llama_grade/inventory.json")}
        mb["served_model"] = served
        mb["completion"] = {c: mb["cells"][c]["completion"] for c in mb["cells"]}
        mb["judge"]["strongreject"].update({"model": PINNED_JUDGE, "rubric": PINNED_RUBRIC, "cache_kind": "sr"})
        roots_meta[root_name] = rmeta
        models[view] = mb

    # ---- the relu_native best-of-75 roots (RELU_ROOTS): the PSR cells, natively served, folded into the view block they belong to and paired
    # against that block's own lens-era `none` at the same cap (the pairing verdict from both stores' bon_served_run.json, stamped on the block)
    relu_meta: dict = {}
    for root_name, decl in RELU_ROOTS.items():
        root = rs / root_name
        if not root.is_dir():
            raise SystemExit(f"relu_native grades root {root} absent -- the wave is declared complete; refusing")
        tdir = only_dir(root)
        view = tdir.name
        if view != decl["view"] or view not in models:
            raise SystemExit(f"{root}: target directory {view!r} is not the declared view {decl['view']!r} (or that view has no lane block); refusing")
        mb = models[view]
        run_dir = rs / decl["transcripts_root"] / f"sdfbase_full22_{view}"
        if not run_dir.is_dir():
            raise SystemExit(f"{root}: transcripts run {run_dir} absent")
        gp = load_json(tdir / "grading_provenance.json") or {}
        j = gp.get("judge") or {}
        if j.get("judge_model") != PINNED_JUDGE or j.get("rubric") != PINNED_RUBRIC:
            raise SystemExit(f"{tdir}: judge {j.get('judge_model')} / {j.get('rubric')} is not the pinned {PINNED_JUDGE} / {PINNED_RUBRIC}")
        if gp.get("decode_caps") != {"bon_max_new_tokens": CAP, "static_max_new_tokens": CAP, "target_max_new_tokens": CAP}:
            raise SystemExit(f"{tdir}: grading_provenance decode_caps {gp.get('decode_caps')} — expected every cap {CAP}")
        if gp.get("require_engine") != decl["serving"]["engine"] or gp.get("serving_engine") != decl["serving"]["engine"]:
            raise SystemExit(f"{tdir}: grading_provenance require_engine / serving_engine {gp.get('require_engine')} / {gp.get('serving_engine')} != {decl['serving']['engine']}; refusing")
        aroot = rs / decl["anchor_root"]
        atdir = only_dir(aroot)
        arun = rs / decl["anchor_transcripts_root"] / f"sdfbase_full22_{atdir.name}"
        if atdir.name != view or not arun.is_dir():
            raise SystemExit(f"{aroot}: anchor view {atdir.name!r} != {view!r} or its transcripts run {arun} is absent; refusing")
        rmeta = {"root": str(root), "view": view, "namespace": decl["namespace"], "transcripts_run": str(run_dir), "attacks_read": list(decl["attacks"]),
                 "attacks_in_root_read_elsewhere": sorted(a[7:-15] for a in (p.name for p in tdir.glob("two_si_*_aggregate.json")) if a[7:-15] not in decl["attacks"]),
                 "note": decl["note"], "tree_provenance": None,
                 "tree_provenance_note": "no tree_provenance.json: the lane's own grader records its driver / config / worktree in grading_provenance.json (recorded, not pinned)",
                 "completion_json_sha256": sha256_file(tdir / "completion.json"), "grading_provenance_sha256": sha256_file(tdir / "grading_provenance.json"),
                 "grader": {k: gp.get(k) for k in ("driver", "config", "worktree", "decode_caps", "goalset", "finished_at", "attacks", "cells", "require_engine", "serving_engine")},
                 "judge_last_attack": {**{k: j.get(k) for k in ("judge_model", "rubric", "cache_kind", "cache_rows", "cache_rows_seeded", "org", "concurrency_initial", "escalation_rule")},
                                       **{k: gp.get(k) for k in ("org_final", "escalated_to_highprio", "n_429", "wall_s", "n_judgments_estimated", "unscored_retry_events", "per_attack_wall_s", "per_attack_429")},
                                       "note": "grading_provenance.json is overwritten per attack (last attack wins); per-attack judge facts are under aggregates.<attack>.grader_provenance"},
                 "attacker_identity_check": gp.get("attacker_identity_check"), "anchor": {"root": decl["anchor_root"], "view": view, "cell": decl["anchor_cell"]}, "aggregates": {}}
        for attack in decl["attacks"]:
            agg = load_json(tdir / f"two_si_{attack}_aggregate.json")
            if not agg:
                raise SystemExit(f"{tdir}/{attack}: aggregate absent -- the relu_native wave is declared complete; refusing")
            if agg.get("target") != view or agg.get("private_run") != f"sdfbase_full22_{view}" or agg.get("wave") != "full22":
                raise SystemExit(f"{tdir}/{attack}: aggregate target {agg.get('target')!r} / private_run {agg.get('private_run')!r} / wave {agg.get('wave')!r} disagree with view {view!r}")
            if agg.get("attack_tier") != ("bon" if attack == "bon" else "multi_turn") or agg.get("lens_reduction_mode") != decl["serving"]["lens_reduction_mode"]:
                raise SystemExit(f"{tdir}/{attack}: tier {agg.get('attack_tier')} / lens {agg.get('lens_reduction_mode')}")
            ag = agg.get("grader_provenance") or {}
            if ag.get("judge_model") != PINNED_JUDGE or ag.get("rubric") != PINNED_RUBRIC:
                raise SystemExit(f"{tdir}/{attack}: aggregate grader_provenance judge {ag.get('judge_model')} / {ag.get('rubric')} != the pinned; refusing")
            if sorted(agg["cells"]) != sorted(decl["cells"]):
                raise SystemExit(f"{tdir}/{attack}: cells {sorted(agg['cells'])} != the declared {sorted(decl['cells'])}; refusing")
            aagg = load_json(atdir / f"two_si_{attack}_aggregate.json")
            if not aagg or decl["anchor_cell"] not in aagg.get("cells", {}):
                raise SystemExit(f"{atdir}/{attack}: the anchor aggregate lacks {decl['anchor_cell']!r}; refusing")
            ra = mb["cells"][decl["anchor_cell"]].get(attack)
            if not isinstance(ra, dict) or ra.get("status") or "k" not in ra:
                raise SystemExit(f"{view}/{attack}: the anchor {decl['anchor_cell']!r} reading in this builder's own block is not graded ({ra}); refusing")
            oa = R2.bon_outcomes(aagg, decl["anchor_cell"]) if attack == "bon" else R2.attack_outcomes(aagg, decl["anchor_cell"])
            anchor_stamp = AP.stamp_from_bon_run(load_json(arun / decl["anchor_cell"] / "bon_served_run.json") or {}, view=view, source=f"{decl['anchor_root']} bon_served_run.json")
            rmeta["aggregates"][attack] = {"path": str(tdir / f"two_si_{attack}_aggregate.json"), "sha256": sha256_file(tdir / f"two_si_{attack}_aggregate.json"),
                                           "written_at": agg.get("written_at"), "cells": sorted(agg["cells"]), "served_alpha": agg.get("served_alpha"),
                                           "serving_engine": agg.get("serving_engine"), "require_engine": agg.get("require_engine"), "serving_identities": agg.get("serving_identities"),
                                           "grader_provenance": {k: ag.get(k) for k in ("judge_model", "rubric", "cache_kind", "cache_rows_seeded", "concurrency_initial", "org", "escalation_rule", "key_plumbing")}}
            mb["judge"]["strongreject"]["per_aggregate"][f"{decl['namespace']}:{attack}"] = rmeta["aggregates"][attack]["grader_provenance"]
            for cell in decl["cells"]:
                key = (view, cell, attack)
                if key in seen:
                    raise SystemExit(f"{key} graded in BOTH {seen[key]} and {root_name}; refusing")
                seen[key] = root_name
                cell_dir = run_dir / cell if attack == "bon" else run_dir / attack / cell
                if attack != "bon":
                    raise SystemExit(f"{root_name}: this builder reads bon only from a relu_native root (the multi-turn wave is cap4096_export's); refusing")
                rec, oc = read_bon_cell(agg, view, cell, cell_dir, ns=decl["namespace"], serving=decl["serving"])
                rec["root"] = root_name
                bm = rec["instrument"].get("base_model")
                if bm != mb["served_model"]:
                    raise SystemExit(f"{cell_dir}: served base_model {bm!r} != the view's {mb['served_model']!r}; refusing")
                lab, grp = label_group(mb["view"], cell)
                mb["cells"].setdefault(cell, {"label": lab, "group": grp, "view": mb["view"], "namespace": decl["namespace"], "decode_cap_tokens": CAP,
                                              "lens_reduction_mode": decl["serving"]["lens_reduction_mode"], "serving_engine": decl["serving"]["engine"],
                                              "on_overlay": cell in B.OVERLAY_CELLS, "completion": {}})
                mb["cells"][cell][attack] = rec
                mb["cells"][cell]["completion"][attack] = rec["completion"]
                mb["sources"].setdefault(cell, {})[attack] = root_name
                mb.setdefault("relu_native_cells", [])
                if cell not in mb["relu_native_cells"]:
                    mb["relu_native_cells"].append(cell)
                # the pairing verdict from the two stores' own served-run records, then the contrast through the campaign's readers
                steered_stamp = AP.stamp_from_bon_run(load_json(cell_dir / "bon_served_run.json") or {}, view=view, source=f"{root_name} bon_served_run.json")
                v = AP.anchor_pairing_verdict(steered_stamp, anchor_stamp)
                if not v.admitted:
                    raise SystemExit(f"relu_native {view}/{cell}/{attack} vs {decl['anchor_root']} `{decl['anchor_cell']}`: pairing REFUSED by anchor_pairing -- {v.reason}")
                vb = contrast_with_intersection(ra, rec, oa, oc, "the anchor and the cell carry different goal sets")
                vb["reads_as"] = (f"cell MINUS the same view's unsteered `{decl['anchor_cell']}` ({decl['anchor_root']}), both at cap {CAP} and the same best-of-N decode; the anchor is the "
                                  f"LENS-era passthrough leg and the cell is served natively -- admitted under {v.allowance} (block `anchor`)")
                vb["anchor_reading"] = {k: ra.get(k) for k in ("k", "n", "p", "lo", "hi", "asr_at_n")}
                vb["anchor"] = {"anchor_root": decl["anchor_root"], "anchor_view": view, "anchor_cell": decl["anchor_cell"],
                                "anchor_aggregate": str(atdir / f"two_si_{attack}_aggregate.json"), "anchor_aggregate_sha256": sha256_file(atdir / f"two_si_{attack}_aggregate.json"),
                                "anchor_written_at": aagg.get("written_at"), "anchor_transcripts_root": str(rs / decl["anchor_transcripts_root"]),
                                "anchor_engine": v.anchor_engine, "steered_engine": v.steered_engine, "pairing_allowance": v.allowance, "pairing_admitted": v.admitted,
                                "pairing_reason": v.reason, "anchor_serving_identity": anchor_stamp.serving_identity, "steered_serving_identity": steered_stamp.serving_identity,
                                "anchor_object_sha256": anchor_stamp.object_sha, "steered_object_sha256": steered_stamp.object_sha,
                                "anchor_base_model": anchor_stamp.model, "steered_base_model": steered_stamp.model,
                                "pairing_module": {"path": str(AP.__file__), "sha256": sha256_file(AP.__file__), "allowance_constant": AP.PASSTHROUGH_ANCHOR_ALLOWANCE}}
                mb["vs_bare"].setdefault(cell, {})[attack] = vb
                count_rows.append({"view": view, "attack": attack, "cell": cell, "root": root_name, "k": rec["k"], "n": rec["n"],
                                   "n_expected": rec.get("n_expected"), "n_errored": rec.get("n_errored"),
                                   "at_cap_k": rec["at_cap_finish_length"]["k"], "at_cap_n": rec["at_cap_finish_length"]["n"],
                                   "cap_proof": rec["decode_cap_verification"]["kind"], "max_completion_tokens": rec["decode_cap_verification"].get("max_completion_tokens"),
                                   "serving_window": rec["instrument"].get("serving_window_tokens"), "decode_mode": rec["instrument"].get("decode_mode")})
        mb["completion"] = {c: mb["cells"][c]["completion"] for c in mb["cells"]}
        roots_meta[root_name] = rmeta
        relu_meta[root_name] = {"view": view, "namespace": decl["namespace"], "attacks_read": list(decl["attacks"]), "cells": list(decl["cells"]),
                                "anchor": rmeta["anchor"], "attacks_in_root_read_elsewhere": rmeta["attacks_in_root_read_elsewhere"], "note": decl["note"]}

    # one served checkpoint across the four views — the campaign fact (a transfer view and its instruct view SERVE THE SAME CHECKPOINT),
    # verified here from every shard's / every BoN run's served metadata rather than asserted
    served_set = sorted({m["served_model"] for m in models.values() if m["served_model"]})
    if len(served_set) != 1:
        raise SystemExit(f"the four Llama views do not serve one checkpoint: {served_set}")

    out = {"meta": {
        **{k: prior_meta.get(k) for k in ("threshold", "bon_n_max", "interval", "readouts", "gated_goal_n", "lens_reduction_rule", "grader_entry_point")},
        "pairing": ("vs_bare: cell minus the same view's unsteered `none` at the same cap and decode, paired on goals (10,000-draw percentile bootstrap "
                    "seed 0 + exact McNemar); on the goal INTERSECTION where the cell dropped errored goals (paired_on_intersection)"),
        "lens_reduction_mode": WAVE_LENS_MODE,
        "served_model": served_set[0],
        "served_model_note": ("every view — the transfer lane, the instruct-direct lane and both prompted-only lanes — serves this one checkpoint; read from "
                              "served_target.json (multi-turn, every shard) and bon_served_run.json proxy_identity (BoN, every cell), refused if any disagreed"),
        "decode_caps": {"bon": CAP, "multi_turn_target_reply": CAP, "filed_wave_bon": CAP_FILED, "filed_wave_multi_turn_target_reply": CAP_FILED,
                        "note": ("the Llama configs set bon_max_new_tokens AND target_max_new_tokens to 4096 (grading_provenance.decode_caps on every root), so the "
                                 "grader's stamp is the run's cap. Multi-turn: proved per cell from target_timing.jsonl (decode_cap_verification.kind = per-row proof). "
                                 "BoN: the store carries no per-row token counts; the stamp is corroborated by the serving window and zero clamps, never proved per row "
                                 "(decode_cap_verification.kind says so). NEITHER readout is at the filed wave's 512 cap: these rows are a different instrument from "
                                 "every other lane's filed BoN / multi-turn rows and sit beside them distinguished by decode_cap_tokens")},
        "multi_turn_decode": ("sampled at the target's shipped generation config (vendor_recommended arm): do_sample true, T 0.6, top_p 0.9, no top_k, no penalty "
                              "(NousResearch/Meta-Llama-3.1-8B-Instruct @ d10aef79 generation_config.json); unseeded — the cap-4,096 wave's canonical instrument, "
                              "not the filed 512 wave's greedy decode; per cell in cells.<cell>.<attack>.instrument.decode"),
        "bon_decode": "T 1.0 (best-of-N harness default, bon_decode_preset harness_default, seed_rule bon_seed_v1) — the same as every lane's BoN",
        "namespaces": {NS: {"label": "the Llama-3.1-8B lane: four views graded 2026-09-13 by llama_grade/ from the campaign grader's lane copy (haiku-4-5 StrongREJECT v1, Low Prio)",
                            "roots": list(ROOTS), "graded_by": "llama_grade/grade_full22_llama.py (lane copy of glm_lane/grade_full22_glm.py; GRADER_DIFF.patch is the whole diff)",
                            "grading_tree": {"branch": "sdfbase-llama8b-stage", "sha": GRADING_TREE_SHA,
                                             "why": "the only lineage carrying configs/sdfbase_full22_llama8b*.yaml; instrument-identical to the other chains' pin by AST diff — see roots.<root>.tree_provenance"},
                            "cells": {v: m["expected_cells"] for v, m in models.items()}, "readouts": {v: m["attacks_present"] for v, m in models.items()},
                            "readouts_pending": {v: m["attacks_pending"] for v, m in models.items()}}},
        "errored_goals": {"ledger": {f"{v}|{a}|{c}": n for (v, a, c), n in ERRORED_EXPECTED.items()},
                          "rule": ("graded with --allow-errored-goals (the GLM-lane precedent); the errored goals are DROPPED from k and n and carried as n_expected / n_graded / "
                                   "n_errored / errored_by_type / errored_goal_ids on the cell record and as completion.n_error_outcomes; this builder refuses any other errored cell or count"),
                          "verified": "every graded multi-turn cell's counts agree with each other, with per_goal, with the completion stamp and with this ledger"},
        "not_graded": {"ledger": {f"{v}|{a}|{c}": n for (v, a, c), n in NOT_GRADED_EXPECTED.items()}, "reason": NOT_GRADED_REASON,
                       "source": {"path": str(rs / "llama_grade/inventory.json"), "sha256": sha256_file(rs / "llama_grade/inventory.json"), "n_cells_refused": inventory.get("n_cells_refused")}},
        "relu_native": {"namespace": NS_RELU, "roots": relu_meta, "serving": NATIVE_SERVING, "pairing_allowance": AP.PASSTHROUGH_ANCHOR_ALLOWANCE,
                        "pairing_module": {"path": str(AP.__file__), "sha256": sha256_file(AP.__file__), "src": str(ANCHOR_PAIRING_SRC)},
                        "pairing_rule": ("anchor_pairing.anchor_pairing_verdict over the two stores' bon_served_run.json stamps: the lane's lens-era passthrough `none` "
                                         "against a natively served steered cell is admitted and stamped with the allowance; anything else REFUSES the build"),
                        "note": ("the relu_native PSR best-of-75 cells live in the SAME view block as the lens lane's cells (models.llama8b.cells.<cell>, namespace "
                                 "relu_native on the cell and its record); their multi-turn aggregates in the same root are the cap4096 builder's relu_native wave")},
        "static_families_petri_capability": "not in this lane's design: the Llama stores carry BoN and the three multi-turn families only (main / it) or BoN only (prompted-only lanes)",
        "cell_labels": {"main": B.CELL_LABEL, "it": {**B.IT_LABEL, **IT_EXTRA_LABEL}, "prompted": {"prompted__safety_refusal": PROMPTED_LABEL, **PROMPTED_CONST_LABEL}},
        "cell_groups": {"main": B.CELL_GROUP, "it": {**B.IT_GROUP, **IT_EXTRA_GROUP}},
        "degeneracy": {"status": "PENDING", "reason": "no degeneracy join over the Llama roots; at-cap per cell from the producer's finish reasons (same definition as the join's at_cap)"},
        "roots": roots_meta,
        "readers": {"transfer_stats": str(Path(T1.__file__)), "transfer_r2": str(Path(R2.__file__)), "w7_builder": str(Path(B.__file__)),
                    "transfer_stats_sha256": sha256_file(T1.__file__), "transfer_r2_sha256": sha256_file(R2.__file__),
                    "w7_builder_sha256": sha256_file(B.__file__), "this_builder_sha256": sha256_file(Path(__file__).resolve()),
                    "anchor_pairing": str(AP.__file__), "anchor_pairing_sha256": sha256_file(AP.__file__)},
        "count_table": count_rows, "warnings": warnings, "written_at": ts(),
    }, "namespaces": {NS: {"models": {v: {"readouts": m["attacks_present"], "cells": m["expected_cells"]} for v, m in models.items()}},
                      NS_RELU: {"models": {v: {"readouts": ["bon"], "cells": m["relu_native_cells"]} for v, m in models.items() if m.get("relu_native_cells")}}},
        "models": models, "cross_view": {"note": "no it-vs-main cross-view contrast is filed here; the export pairs each cell against its own view's bare only"},
        "drift": {v: R2.not_in_design("no regrade cell in this namespace's design") for v in models}}

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    if outp.exists():
        prev = outp.with_name(outp.name + f".prev-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
        outp.rename(prev)   # never overwrite silently: the previous build is kept beside the new one
        print(f"previous build kept at {prev}")
    outp.write_text(json.dumps(out, indent=1) + "\n")

    print(f"\n{'view':20s} {'attack':12s} {'cell':46s} | {'k/n':9s} {'n_exp':5s} {'n_err':5s} {'at-cap':11s} {'max_ct':6s} {'window':6s} {'decode':24s} {'cap proof':30s}")
    for r in count_rows:
        print(f"{r['view']:20s} {r['attack']:12s} {r['cell'][:46]:46s} | {str(r['k'])+'/'+str(r['n']):9s} {str(r['n_expected'] or ''):5s} {str(r['n_errored'] if r['n_errored'] is not None else ''):5s} "
              f"{str(r['at_cap_k'])+'/'+str(r['at_cap_n']):11s} {str(r['max_completion_tokens'] or '—'):6s} {str(r['serving_window']):6s} {str(r['decode_mode']):24s} {r['cap_proof'][:30]:30s}")
    for v, m in models.items():
        for c, per in m["vs_bare"].items():
            for a, vb in per.items():
                p = vb.get("paired") or vb.get("paired_on_intersection")
                tag = "paired" if vb.get("paired") else ("INTERSECTION" if vb.get("paired_on_intersection") else vb.get("status", "?"))
                print(f"  vs_bare {v}/{a}/{c[:40]}: " + (f"{p['delta_pp']:+.1f} pp [{p['lo_pp']:+.1f}, {p['hi_pp']:+.1f}] n_paired {p['n_paired']} McNemar p={p['mcnemar_exact_p']:.3g} ({tag})" if p else tag))
    for v, m in models.items():
        for c in m.get("relu_native_cells") or []:
            vb = m["vs_bare"][c]["bon"]; p = vb.get("paired") or vb.get("paired_on_intersection") or {}
            print(f"  relu_native bon {v}/{c[:44]} vs {vb['anchor']['anchor_root']} none: {p.get('delta_pp'):+.1f} pp [{p.get('lo_pp'):+.1f}, {p.get('hi_pp'):+.1f}] "
                  f"n_paired {p.get('n_paired')} McNemar p={p.get('mcnemar_exact_p'):.3g} ({vb['anchor']['pairing_allowance']})")
    pend = {v: m["attacks_pending"] for v, m in models.items() if m["attacks_pending"]}
    print(f"\n{len(count_rows)} graded (view, attack, cell) readings from {len(ROOTS)} roots; pending legs: {pend or 'none'}; "
          f"{sum(len(x) for m in models.values() for x in m['not_graded'].values())} not-graded PSR cells filed; served checkpoint {served_set[0]}")
    for w in warnings:
        print("WARN", w)
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
