#!/usr/bin/env python
"""seedrep_launch_plan.py — THE ONE generator of the qwen-seedrep-native-battery launch plan.

It renders, from the roster + pins at W, the arms table, the geometry table and a `decisions.json` of the S4-decided
fields, EVERY production container the wave will spawn — the standup argv (composed through the real front-end
contract: `mt_serving_up.py --serving-path lora_native` over `lora_serving_up.bring_up_lora`), the expected geometry
(S x L, the kernel lowering the core records), the expected route set, the per-fitted-cell object pins, the per-cell
adapter-reuse expectations, the per-child driver argv with its registry identity file, the queue producers it feeds,
the six-axis control key — plus the attacker pools, the rung functions declared per (leg, view|arditi, cap), the `S5`
window roster, and the capacity ledger (containers and GPUs counted separately).

TWO MATERIALISATIONS (rev 4): `--materialisation draft` BEFORE S0 (the S4 fields at their spec defaults:
Crescendo / ActorAttack 49,152 x 10, 4 pools at H200 (one card each) / 65,536, AA forwards 24) and `--materialisation final` after S4
from the measured decisions. `compare_draft_final()` refuses ANY diff outside the S4-decided fields — a differing
byte anywhere else is a generator bug and blocks the launch.

THE RULES THIS FILE ENFORCES (each with a test in tests/test_seedrep_launch_plan.py against a KNOWN-BAD input):
  * a GSM8K standup passes `--eval capability:gen_long_gsm8k_common_f22` (rung: `..._rung_common_f22`) and NEVER
    `--max-model-len` / `--max-num-seqs` (rev 4: an explicit L is stamped `override`; the reader admits `override`
    only beside a recorded kernel lowering whose L equals the table entry's — the shape the core writes by itself);
  * every standup that serves MMLU (the controls leg: 3 view engines + Arditi, base and every rung) carries
    `--max-logprobs 64` (vLLM's default 20 refuses the driver's `top_logprobs 64`);
  * S is never passed for a TABLE window (the core derives the identity S and lowers it to `524,287 // L`); a
    NON-table window passes the identity PAIR `--max-model-len L --max-num-seqs identity_seqs(L)` — never a bare S;
  * every standup: `--serving-path lora_native`, `--enable-log-requests`, `--server-default-policy contract_override`,
    `--model-key qwen35_4b`, `--thinking-on` iff the leg is GSM8K, the two prompted routes REQUESTED explicitly on
    every view engine, `--expect-object-sha256 <cell>=<pin>` for every fitted cell, the Arditi engine a single object;
  * `planned standups - S5 references = empty` (gate 8) is computed here (`s5_roster`, `s5_gap`), never by hand;
  * the adapter-hash enumeration (gate 10) covers every (standup, cell) — hashes are filled into the FINAL plan
    from the smokes' `cells_served.json` maps (`--adapter-hashes`), the draft carries the object pins;
  * a PRODUCTION row's child never carries a smoke-only ITEM-budget flag (BoN `--limit-goals`, misalign
    `--limit-items` -- a limited store is never a wave result); `smoke_row` gives a smoke row's BoN / misalign children
    exactly that budget (`SMOKE_BON_GOALS` / `SMOKE_MISALIGN_ITEMS`, recorded in `smoke.budget`), an S5 reference row keeps
    the FULL shape by name, and a smoke BoN / misalign child WITHOUT its budget is refused (`smoke_budget_problems`).

This module has NO Modal import: it runs on the grading host (PYTHONPATH=<tree>/src for the geometry table) and inside the
CPU preflight container from the staged tree. `scripts/modal/seedrep_served.py` executes the rows it writes;
`seedrep_ladder.py` reads the producers; `seedrep_attacker_pool.py` reads the pool rows; `registry_plans.py` writes
the identity files the children name.
"""
from __future__ import annotations

import argparse
import collections
import copy
import dataclasses
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# the fixed facts of the wave                                                  #
# --------------------------------------------------------------------------- #

MODEL_KEY = "qwen35_4b"
MODEL = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
VIEWS: Tuple[str, ...] = ("s1", "s2", "s3")
ARDITI = "arditi"                                   # the fourth "view" slot of every leg: the single-object engine
ENGINES: Tuple[str, ...] = VIEWS + (ARDITI,)
SEED_OF_VIEW: Dict[str, int] = {"s1": 1, "s2": 2, "s3": 3}   # the LITERAL training seed
ARDITI_SEEDS: Tuple[int, ...] = (1, 2, 3)           # three client-side seed bases on the ONE Arditi engine (§2.3)

#: the view engines' harness cell ids — the s-view configs' explicit_cells minus `none`, roster order
FITTED_CELLS: Tuple[str, ...] = (
    "constv2-mix-full__psr_ce_all",
    "constv2-mix-full__lora_r4_ce",
    "constv2ctl2-full__psr_ce_all",
    "constv2ctl2-full__lora_r4_ce",
    "constv2-mix-full__lora_r4_ce_minus_ctl",
)
#: the two prompted routes, REQUESTED explicitly on every view engine (admission over the config's SI vocabulary)
PROMPTED_CELLS: Tuple[str, ...] = ("prompted__si_constitution", "prompted__safety_refusal")
#: the residual-slot (PSR) cells of a view — `residual_slot.armed` is asserted for exactly these at readiness
PSR_CELLS: Tuple[str, ...] = ("constv2-mix-full__psr_ce_all", "constv2ctl2-full__psr_ce_all")

# --- TWO CELL-ID NAMESPACES, converted at every boundary ------------------------------------------------------------
#: The PLAN speaks plan ids (the roster / config / `--cell` / `--defense` / the arm ids / the registry identity files).
#: The NATIVE ENGINE'S RECORDS speak HARNESS ids (`cells_served.json` canonical + controls rows, `endpoint.json`'s
#: `expected_object_sha256_by_cell` / `adapters_reused` / `adapters`, the routing table, `cells.json` and therefore the
#: misalign runner's `--only-cells`). The translation is `sisteer.capeval.petri.registry_adapter.harness_cell_id`;
#: this table is its VALUE for the plan's nine cells, pinned equal to the real function by test (an unmapped id refuses).
HARNESS_ID_OF: Dict[str, str] = {
    "none": "none",
    "constv2-mix-full__psr_ce_all": "si_constv2_mix_full__psr_ce_all",
    "constv2-mix-full__lora_r4_ce": "si_constv2_mix_full__lora_r4_ce",
    "constv2ctl2-full__psr_ce_all": "si_constv2ctl2_full__psr_ce_all",
    "constv2ctl2-full__lora_r4_ce": "si_constv2ctl2_full__lora_r4_ce",
    "constv2-mix-full__lora_r4_ce_minus_ctl": "si_constv2_mix_full__lora_r4_ce_minus_ctl",
    "constv2-mix-full__psr_ce_all_minus_ctl_distilled": "si_constv2_mix_full__psr_ce_all_minus_ctl_distilled",   # the paired difference parent
    "prompted__si_constitution": "prompted__constitution",
    "prompted__safety_refusal": "prompted__safety",
    "arditi_allpos": "arditi_allpos",
}
TRAINTEST_ID_OF: Dict[str, str] = {h: t for t, h in HARNESS_ID_OF.items()}
#: the DERIVED zero-adapter control every lora_native engine serves and lists under `cells_served.controls`
#: (`cell_class: control`, `control_of: none`); it is not a plan arm and has no plan-id spelling
NONE_LORA0 = "none__lora0"


def harness_id(traintest_id: str) -> str:
    """TrainTest -> harness cell id for the lane's cells; an id outside the table REFUSES (never a silent rename)."""
    try:
        return HARNESS_ID_OF[traintest_id]
    except KeyError:
        raise ValueError(f"TrainTest cell {traintest_id!r} has no harness translation in this lane ({sorted(HARNESS_ID_OF)})") from None


def traintest_id(harness_cell_id: str) -> str:
    """harness -> TrainTest cell id (the inverse); `none__lora0` has none and REFUSES like any other unmapped id."""
    try:
        return TRAINTEST_ID_OF[harness_cell_id]
    except KeyError:
        raise ValueError(f"harness cell {harness_cell_id!r} has no TrainTest translation in this lane ({sorted(TRAINTEST_ID_OF)})") from None
def ladder_arm_of_cell(cell_id: str) -> str:
    """The queue ARM a ladder store's cell id names -- the ONE translation the controls tailer applies to a `controls_events.jsonl`
    event's `cell_id` (the driver's `--cell`, the plan child's `cell`) AND to the row's children when it looks the store's owner up:
    a harness id translates to its `traintest_id`; any other id -- the traintest spelling the plan's children already carry (`none`,
    a fitted cell, a prompted cell) and the Arditi cell `arditi_allpos` -- is the arm itself. NEVER the child's arm LABEL: the Arditi
    row's children are labelled `ARDITI_ARM_ID` (`ar_c0p3__arditi_allpos`, the coverage table's arm) while the driver names their
    stores `ARDITI_CELL` (`arditi_allpos`), and a lookup keyed on the label found no child for the store (`LADDER_TAILER_ERROR`
    after 3 children rc 0). The rung consumer filters slots by the driver's defense id
    (`rung_redraw`: `arm_s(defense, seed_base)`), so the arm the record carries is the cell spelling."""
    return traintest_id(cell_id) if cell_id in TRAINTEST_ID_OF else str(cell_id)


#: the view engine's registered adapter count: 5 objects + none__lora0 (the zero adapter); rank cap 8 (r=4 + r=8 difference)
VIEW_MAX_LORAS = 6
VIEW_MAX_LORA_RANK = 8

ARDITI_CELL = "arditi_allpos"
ARDITI_VIEW = "ar_c0p3"
ARDITI_STAGED_ID = "ar_c0p3__arditi_allpos"
ARDITI_ARM_ID = "ar_c0p3__arditi_allpos"            # the arm's id in the coverage table (27 arm-runs per leg)
ARDITI_PINS_REL = "configs/arditi_allpos_pins.json"
ARMS_REL = "configs/sdfbase_full22_arms.yaml"
#: the lane copies of the Arditi standup config + Petri manifest (rendered here; §2.3; the source sha is pinned)
ARDITI_CONFIG_REL = "configs/seedrep/arditi_allpos_qwen35_4b_ar_c0p3_seedrep.yaml"
ARDITI_MANIFEST_REL = "configs/seedrep/arditi_allpos_qwen35_4b_ar_c0p3_seedrep_manifest.yaml"
#: the lane's per-seed Arditi client configs (static / controls drivers read their sampled_decode block; §2.3)
ARDITI_CLIENT_CONFIG_REL = "configs/seedrep/arditi_allpos_qwen35_4b_ar_c0p3_seedrep_s{s}_sampled_t0{suffix}.yaml"
#: the WAVE legs (BoN, multi-turn) file their stores under the config's `private_run`, and the grading discovery
#: (`grade_battery.sh runs_of`) reads the static-base run off `*_seedrep_s<s>_sampled_t0.yaml` and the wave run off the per-seed
#: `*_seedrep_s<s>.yaml` when one exists (else the upstream shared namespace, ONE for three seeds). So the Arditi BoN and multi-turn
#: children use the per-seed NON-sampled wave configs (distinct `private_run` per seed), never the `_sampled_t0` static configs and
#: never the shared `_seedrep.yaml`; the `_sampled_t0` configs stay the static / controls legs'. A view's wave config is its
#: unsampled `VIEW_CONFIG_REL`.
ARDITI_WAVE_CONFIG_REL = "configs/seedrep/arditi_allpos_qwen35_4b_ar_c0p3_seedrep_s{s}.yaml"
#: the legs whose stores the discovery files under the WAVE private_run (everything else: the static-base / sampled namespaces)
WAVE_LEGS: Tuple[str, ...] = ("bon", "pair", "crescendo", "actorattack")

PINS_REL = "configs/sdfbase_full22_pins.json"
ROSTER_REL = "configs/sdfbase_full22_roster.yaml"
VLLM_REQS_REL = "configs/capeval/vllm-requirements.txt"
VIEW_CONFIG_REL = "configs/sdfbase_full22_qwen35_4b_{view}.yaml"
VIEW_MANIFEST_REL = "configs/capeval/manifests/petri_full22_qwen35_4b_{view}.yaml"
VIEW_SAMPLED_CONFIG_REL = "configs/sdfbase_full22_qwen35_4b_{view}_sampled_t0{suffix}.yaml"

#: the LoRA kernel's int32 index ceiling for the banked adapters (widest adapted module D = 4096 on Qwen3.5-4B):
#: (2^31 - 1) // 4096 = 524,287 batched tokens; S = ceiling // L per window
LORA_KERNEL_D = 4096
LORA_KERNEL_INDEX_LIMIT = 2 ** 31 - 1
KERNEL_CEILING = LORA_KERNEL_INDEX_LIMIT // LORA_KERNEL_D      # 524,287
HYBRID_SEQS_CAP = 256
#: the measured qwen35_4b batched-token budget the geometry table sizes S from (geometry.BUDGET_BY_MODEL, 2026-09-13);
#: the real front is consulted when importable (`identity_seqs_for`), this literal is the offline fallback + a pinned check
QWEN35_4B_BUDGET = 741_888

#: the GSM8K COMMON geometry class (two table entries, copies of the `_longsi_f22` values)
GSM8K_COMMON_EVAL = "capability:gen_long_gsm8k_common_f22"
GSM8K_RUNG_COMMON_EVAL = "capability:gen_long_gsm8k_rung_common_f22"
GSM8K_COMMON_L = 16640
GSM8K_RUNG_COMMON_L = 24832
#: the entries whose values the common class copies (present at every base; the generator's table check falls back to them
#: ONLY to verify the numbers, never to launch — a standup launches on the common names or refuses)
GSM8K_COMMON_SOURCE_EVAL = {GSM8K_COMMON_EVAL: "capability:gen_long_gsm8k_longsi_f22",
                            GSM8K_RUNG_COMMON_EVAL: "capability:gen_long_gsm8k_rung_longsi_f22"}

MMLU_TOP_LOGPROBS = 64
GPU_MEM_UTIL = "0.9"
READY_TIMEOUT_S = 2700
UPSTREAM_TIMEOUT_S = 7200
SERVER_DEFAULT_POLICY = "contract_override"
SERVING_PATH = "lora_native"

#: the container-local tree + bank (the `_build_shared` shape; §4.2) and the volume lane root (§4.1)
CONTAINER_USER = "seedrep"
SHARED_LOCAL = f"/workspace/{CONTAINER_USER}"
VENVS_LOCAL = f"/home/{CONTAINER_USER}/venvs"
VOL_PATH = "/vol"
LANE = f"{VOL_PATH}/seedrep"
REPOS = f"{LANE}/repos"
OBJECTS_VOL = f"{LANE}/objects"
ADAPTER_BANK_ROOT = f"{LANE}/multilora_adapters"
CACHE_VOL = f"{LANE}/cache"
HF_VOL = f"{VOL_PATH}/hf"
RESULTS_ROOT_NAME = "seedrep_battery_modal"        # the DISTINCT results root (never a cluster store): <LANE>/<W7>/exp/<name>
BANK_VIEW_LOCAL = f"{SHARED_LOCAL}/exp/objects/sdfbase/full22/{MODEL_KEY}/cells_view/{{view}}"
ARDITI_BANK_VIEW_LOCAL = f"{SHARED_LOCAL}/exp/objects/arditi_allpos/{MODEL_KEY}/ladder/cells_view/{ARDITI_VIEW}"
#: THE GRADING-HOST SIDE OF THE BANK. The battery's views (`s1`..`s3` and every plan-referenced dose view,
#: `d1p5_lora` / `dm0p5_lora`) are BUILT by the operator tree's `00_build_views.sh` under the LANE-owned bank prefix
#: `lane_common.LANE_BANK` = `$STAGE/sdfbase/seedrep_battery/bank` (`<LANE_BANK>/qwen35_4b/cells` is a symlink
#: through to the shared `$STAGE/sdfbase/full22_objects/qwen35_4b/cells`, read-only; `cells_view/` is ours). The shared
#: bank holds only ITS views (`main`, `it`, the ReLU doses), so a grading-host reader that derives a battery view's path
#: from it refuses every battery view by construction -- the Petri drive did. ONE derivation for every reader of the bank:
#: `seedrep_served --stage-bank` (its `bank_local` default + `stage_bank_arcnames`) and the Petri drive's `view_params` both compose
#: `<lane bank>/<MODEL_KEY>/cells_view/<view>` through :func:`bank_view_subpath` / :func:`grading_host_bank_view_rel`; never a second literal.
def _grading_host_stage_rel() -> str:
    """The site's staging directory (``$STAGE``) relative to $HOME, the form every grading host reader joins under the home it is given;
    the placeholder itself where no site config resolves it (a Modal container imports this module and never reads these paths)."""
    scripts = Path(__file__).resolve().parent.parent
    if (scripts / "site_paths.py").is_file():
        if str(scripts) not in sys.path:
            sys.path.insert(0, str(scripts))
        try:
            from site_paths import site_path
            return os.path.relpath(site_path("$STAGE", "the lane bank's staging directory"), Path.home())
        except SystemExit:                  # site_paths.SitePathError: $STAGE is not set on this host
            pass
    return "$STAGE"


GRADING_HOST_LANE_BANK_REL = f"{_grading_host_stage_rel()}/sdfbase/seedrep_battery/bank"             # relative to $HOME; == lane_common.LANE_BANK
GRADING_HOST_ARDITI_LADDER_REL = f"arditi-stage/objects/arditi_allpos/{MODEL_KEY}/ladder"    # relative to $HOME; == lane_common.ARDITI (the upstream read-only copy)


def bank_view_subpath(view: str, model_key: str = MODEL_KEY) -> str:
    """``<model_key>/cells_view/<view>``: a view's path under a bank root (the lane bank on the grading host, ``sdfbase/full22`` in the staged
    tarball, the container's objects tree) -- the shape ``make_bank_view.py --bank <root>/<target> --view <view>`` writes."""
    if not isinstance(view, str) or not view or "/" in view:
        raise ValueError(f"bank view {view!r} must be a non-empty slash-free name")
    return f"{model_key}/cells_view/{view}"


def grading_host_bank_view_rel(view: str, lane_bank_rel: str = GRADING_HOST_LANE_BANK_REL) -> str:
    """The grading host path of a battery bank view, relative to $HOME: ``<GRADING_HOST_LANE_BANK_REL>/<MODEL_KEY>/cells_view/<view>`` -- the lane
    bank's view, never the peers' shared bank's (F-R)."""
    return f"{lane_bank_rel}/{bank_view_subpath(view)}"
#: the ONE placeholder a plan argv carries: the attempt root is minted at function entry (§4.6), never at plan time
ATTEMPT_ROOT_PLACEHOLDER = "{attempt_root}"
#: <attempt_root>/exp/serve = the standup's --out. UNDER exp/ on purpose: the attempt root's `exp/` tree is what the
#: TreeSyncer ships as the container's exp tree, and the Petri drive maps the endpoint's container paths onto
#: `<attempt_root_on_volume>/exp/...` (`seedrep_petri_drive.attempt_paths`); a serve dir beside exp/ would be unreachable
SERVE_DIR_REL = "exp/serve"
#: the misalign runner's out root per (seed base, behaviour): the runner writes `<out>/<cell>/{responses.jsonl, meta.json}`
#: and its resume identity carries neither the seed base nor the behaviour, so two children of one cell sharing an out
#: root would interleave one responses.jsonl and race one meta.json; the cell dir keeps the cell identity
MISALIGN_OUT_REL = "exp/misalign/agentic__s{s}__{behavior}"
#: the Arditi engine's THREE seed children of one cell used to share one controls / GSM8K out-root, so a store could not be
#: attributed to one seed base and the CellStore keys collided; every controls / GSM8K child now owns
#: `<...>_cap<N>__s<s>` -- the tailer reads the seed off the path (`controls_seed_of_store`), and the layout is uniform across engines
CONTROLS_OUT_REL = "exp/sampled/controls_cap{cap}__s{s}"
#: ONE cell per out root: `inspect_rung.cell_dir_of(out_root)` (the ladder's classifier) refuses a root with two cell dirs under cells/, so
#: every GSM8K child owns `<attempt root>/exp/sampled/gsm8k_thinking_on_cap<cap>__s<s>/<cell>/` (the `controls_<wave>/<TAG>/<cell>/` shape)
GSM8K_OUT_REL = "exp/sampled/gsm8k_thinking_on_cap{cap}__s{s}/{cell}"
#: the rung's queue-derived Inspect subset (`rung_subset/1`, written by the assembly helper from the CLOSED queue + the accepted source store)
GSM8K_SUBSET_REL = "exp/rung_subsets/gsm8k_cap{cap}__s{s}/{cell}/rung_subset.json"
#: the CPU prompt-budget artifact + its `.fit.json` sidecar (`g8k_budget_artifact`) per (cap, cell) -- the driver's `--oos-prompt-budget`
GSM8K_BUDGET_REL = "exp/budgets/gsm8k_paired200_on_cap{cap}/{cell}/oos_prompt_budget.json"
GSM8K_INSPECT_TASK = "gsm8k_paired200"                        # the CLI spelling the cap flags carry (contracts.GSM8K_PAIRED_TASK_NAME)
GSM8K_ROW_TASK = "inspect/gsm8k_paired200"                    # the store's ROW task (contracts.GSM8K_STANDARD_TASK)
GSM8K_IDENTITY_VERSION = 2                                    # G8K_IDENTITY_VERSION: the ladder reads identity-v2 stores (max_tokens_by_task is hashed)
GSM8K_SAMPLES = 1
GSM8K_ASSEMBLY_HELPER_REL = "scripts/modal/seedrep_gsm8k_assembly.py"
GSM8K_VLLM_LOG_REL = "exp/serve/vllm_server.log"
#: the OPERATIONAL contract for a GSM8K row (`scripts/sdfbase/gsm8k/g8k_common.sh` g8k_driver_argv / g8k_ledger_up; `served_thinking.sh`
#: st_driver_argv_head / st_ledger_up), reproduced per CHILD: the per-CELL overlay config (`G8K_OWN_OVERLAY`), a USAGE
#: LEDGER in front of the proxy started WITH the attempt id and bound by its record beside the child's routing table (`--ledger-url`), the
#: attempt id `a<k>` minted per store from `ladder_state.json` (`--attempt-id`; the driver publishes `accepted_generation_attempt` only with
#: it -- the validators refuse a store without one), and for every FITTED row the acknowledgement `--allow-out-of-support-thinking`
#: (a fitted lens route under the thinking-ON render is outside its fit support -- `contracts.fitted_thinking_on_out_of_support`; the driver
#: refuses before a store exists otherwise) under its authorised tuple, with `--expect-object-sha256 <pin>`.
GSM8K_OVERLAY_CONFIG_REL = "configs/sdfbase/gsm8k/sdfbase_full22_{model}_{view}.gsm8k_{slug}.yaml"       # make_gsm8k_overlays.py, per TAG x cell
GSM8K_LEDGER_SERVE_REL = "exp/serve_gsm8k/{cell}__s{s}"           # the child's OWN copy of the standup record + routing table, its ledger record beside it
GSM8K_LEDGER_FILENAME = "usage_ledger.jsonl"                     # G8K_LEDGER_FILENAME, in the child's out root
GSM8K_LEDGER_ENDPOINT_FILENAME = "ledger_endpoint.json"          # contracts.LEDGER_ENDPOINT_FILENAME, beside routing_table.json
GSM8K_LADDER_STATE_FILENAME = "ladder_state.json"                # g8k_ladder_state/1, in the child's out root (the a<k> counter)
GSM8K_LEDGER_WAIT_S = 180                                        # G8K_LEDGER_WAIT_S
GSM8K_FIRST_ATTEMPT_ID = "a1"                                    # a fresh immutable Modal attempt root holds ONE driver attempt of the store
#: `--bank-view s<s>` -- THE ROW's VIEW, exactly where the child's argv puts it (after `--attempt-id`): 2126e918 widened the driver's vocabulary to
#: every roster view (`contracts.bank_views_admitted()`: main | it + the full22 roster's lane views), and the GSM8K reader holds every row's stamp
#: to the matrix row's view -- `main` is refused as a disagreement, omission drops the row identity. A view row's child stamps its view; an Arditi
#: child stamps the view of the seed it replicates (`s<s>`). The driver at W must carry 2126e918 (this tree's parser predates it: `choices main | it`).
GSM8K_BANK_VIEW_RULE = ("--bank-view == the child's SOURCE bank view: a view engine's child s<s> (its view), an Arditi child ar_c0p3 (WS0 2126e918 / "
                        "impl_ws0 'GSM8K child argv contract'; the third review's delta 3 (5)): never main, never omitted; the driver / contracts at W must be >= 218d66a7 "
                        "(the stamp is WS0's contracts.expected_bank_view_stamp, one function for launcher and reader)")
GSM8K_BANK_VIEW_WS0_FIX = "218d66a7"


ARDITI_BANK_VIEW = "ar_c0p3"                                        # the SOURCE bank the Arditi objects come from (218d66a7 admits it)
WS0_BANK_VIEW_MODULE_REL = "scripts/sdfbase/gsm8k/g8k_seedrep_arditi.py"   # `bank_view_stamp(row)` -> contracts.expected_bank_view_stamp (218d66a7)
WS0_TREE_ENV = "SEEDREP_WS0_TREE"
#: an injected stand-in for the stamp rule (tests pin theirs against the real one); None = the real one, from this tree or the ws0 worktree
BANK_VIEW_STAMP: Optional[Callable[[Mapping[str, Any]], str]] = None
_BANK_VIEW_STAMPS: Dict[Tuple[str, Optional[str]], str] = {}


def _ws0_bank_view_stamp(view: str, bank_view: Optional[str]) -> str:
    """The ONE rule for the `--bank-view` a row's children stamp (218d66a7: `g8k_seedrep_arditi.bank_view_stamp(row)` ->
    `contracts.expected_bank_view_stamp(view=, bank_view=)`), never a local rule: the contracts function from `sys.path` (the tree at W on the
    grading host, or the staged tree's `src` in-container), else -- on the grading host only, a <repo>/scripts/modal layout -- from the ws0 tree
    ($SEEDREP_WS0_TREE) in a subprocess with ITS `src` (a development bridge until W); refused
    by name when neither works (never `parents[2]`, which raises IndexError at the image's /root/seedrep_launch_plan.py)."""
    key = (str(view), bank_view)
    if key in _BANK_VIEW_STAMPS:
        return _BANK_VIEW_STAMPS[key]
    row = {"view": str(view), "bank_view": bank_view}
    # In the real image this module is installed FLAT at /root/seedrep_launch_plan.py, where `parents[2]` raises
    # IndexError -- and `validate_plan` -> `gsm8k_bank_view` reaches here in-container (`launch_plan_validates_against_table`). Same defensive
    # repo detection as `registry_plans_module`: a <repo>/scripts/modal/<file> layout names the tree (the grading host); anything else is a container.
    here = Path(__file__).resolve()
    repo = here.parents[2] if len(here.parents) > 2 and here.parents[1].name == "scripts" and here.parent.name == "modal" else None
    # 1. the contracts function itself, from `sys.path`: the tree on the grading host (PYTHONPATH=<tree>/src), or the staged tree's `src` the app /
    #    preflight inserts in-container -- the ONE rule for launcher and reader
    try:
        from sisteer.capeval.tinycap import contracts as C  # noqa: PLC0415
        fn = getattr(C, "expected_bank_view_stamp", None)
    except Exception:  # noqa: BLE001 -- an older sisteer on sys.path, or none
        fn = None
    if fn is not None:
        _BANK_VIEW_STAMPS[key] = str(fn(view=row["view"], bank_view=row["bank_view"]))
        return _BANK_VIEW_STAMPS[key]
    # 2. in a container (no repo layout) there is no worktree bridge to fall to: refuse by name
    if repo is None:
        raise RuntimeError(f"no bank-view stamp rule: sisteer.capeval.tinycap.contracts.expected_bank_view_stamp (WS0 218d66a7) is not importable from "
                           f"sys.path and {here} is not in a <repo>/scripts/modal layout -- in a container the staged tree's src must be on sys.path; "
                           "the GSM8K children's --bank-view is WS0's function, never a local rule")
    # 3. the grading host development bridge (older trees): the stamp module from the ws0 worktree in a subprocess with ITS src
    if not os.environ.get(WS0_TREE_ENV):
        raise RuntimeError(f"no bank-view stamp rule: {WS0_BANK_VIEW_MODULE_REL} (WS0 218d66a7) is not importable from this tree ({repo}); "
                           f"set ${WS0_TREE_ENV} to the ws0 worktree to use its rule, the development bridge")
    ws0 = Path(os.environ[WS0_TREE_ENV])
    if not (ws0 / WS0_BANK_VIEW_MODULE_REL).is_file():
        raise RuntimeError(f"no bank-view stamp rule: {WS0_BANK_VIEW_MODULE_REL} (WS0 218d66a7) is neither importable from this tree ({repo}) nor under "
                           f"{ws0} (${WS0_TREE_ENV}); the GSM8K children's --bank-view is WS0's function, never a local rule")
    import subprocess  # noqa: PLC0415
    code = ("import json, sys\nsys.path.insert(0, sys.argv[1] + '/src'); sys.path.insert(0, sys.argv[1] + '/scripts/sdfbase/gsm8k')\n"
            "import g8k_seedrep_arditi as A\nrow = json.loads(sys.argv[2]); print(A.bank_view_stamp(row))\n")
    r = subprocess.run([sys.executable, "-c", code, str(ws0), json.dumps(row)], capture_output=True, text=True, timeout=300, cwd=str(ws0))
    if r.returncode != 0:
        raise RuntimeError(f"WS0's bank_view_stamp refused {row}: {r.stderr[-400:]}")
    _BANK_VIEW_STAMPS[key] = r.stdout.strip().splitlines()[-1]
    return _BANK_VIEW_STAMPS[key]


def gsm8k_bank_view(engine: str, s: Any) -> str:
    """`--bank-view` of a GSM8K child, by the stamp rule (`bank_view_stamp(row)`): a view engine's child on `view s<s>` with no source bank stamps its
    view; an Arditi child (`view s<s>`, `bank_view ar_c0p3`) stamps the SOURCE bank `ar_c0p3`, never the replicated seed view."""
    fn = BANK_VIEW_STAMP
    row = {"view": f"s{int(s)}", "bank_view": ARDITI_BANK_VIEW if engine == ARDITI else None}
    return str(fn(row)) if fn is not None else _ws0_bank_view_stamp(row["view"], row["bank_view"])


#: the authorised out-of-support tuple a FITTED lane row runs under is the native battery's OWN,
#: `gsm8k_paired200_<model>_native_<cap>` (contracts.g8k_native_tuple_name; reduction_mode not_applicable = the native
#: engine's served identity, the five lane cells, the two caps as two doors) — never the LENS-ERA `gsm8k_paired200_<model>_<cap>`
#: (reduction_mode stacked), which every fitted GSM8K child refused against by name (`reduction_mode differs —
#: requested 'not_applicable', authorised 'stacked'`). The lens-era spelling is kept ONLY so `validate_plan` can refuse it by name.
GSM8K_NATIVE_OOS_TUPLE = "gsm8k_paired200_{model}_native_{cap}"
GSM8K_LENS_ERA_OOS_TUPLE = "gsm8k_paired200_{model}_{cap}"


def gsm8k_oos_tuple(cell: str, cap: int, *, engine: str) -> Optional[str]:
    """The authorised out-of-support tuple for a FITTED GSM8K row at a rung: the lane cells run under the native battery's
    OWN tuple `gsm8k_paired200_qwen35_4b_native_<cap>` (contracts.G8K_NATIVE_CELLS / g8k_native_tuple_name — the native
    standup's served identity is reduction_mode not_applicable, which the lens-era `gsm8k_paired200_qwen35_4b_<cap>` refuses by
    name), the Arditi all-positions arm `gsm8k_paired200_qwen35_4b_steer_allpos_<cap>` (its record `arditi_rows_qwen_seedrep.json`
    names exactly these; already the native identity); None for `none` and the prompted routes (no tuple, no override)."""
    if cell == "none" or cell in PROMPTED_CELLS:
        return None
    if engine == ARDITI or cell == ARDITI_CELL:
        return f"gsm8k_paired200_{MODEL_KEY}_steer_allpos_{int(cap)}"
    return GSM8K_NATIVE_OOS_TUPLE.format(model=MODEL_KEY, cap=int(cap))


def gsm8k_cell_slug(cell: str) -> str:
    return cell.replace("__", "--")

STAGE_BASE = "base"
STAGE_PETRI_RERUN = "petri-rerun"

# --------------------------------------------------------------------------- #
# the legs                                                                      #
# --------------------------------------------------------------------------- #


class WindowSpec(dict):
    """One served window class: ``eval`` (the --eval value), ``L``, ``table`` (the window IS the eval's table entry,
    so the argv passes --eval only), ``thinking`` (GSM8K), ``cap`` (the driver's completion cap at this rung)."""

    def __init__(self, eval: str, L: int, *, table: bool, thinking: bool, cap: int):  # noqa: A002
        super().__init__(eval=eval, L=int(L), table=bool(table), thinking=bool(thinking), cap=int(cap))


#: the standup legs (12 rows) -> the 15 EVAL legs they serve
EVAL_LEGS_OF: Dict[str, Tuple[str, ...]] = {
    "pasttense": ("pasttense",), "codeattack": ("codeattack",), "renellm": ("renellm",), "grok420": ("grok420",),
    "bon": ("bon",),
    "controls": ("xstest", "or_bench_hard", "mmlu"),
    "gsm8k": ("gsm8k",),
    "petri": ("petri",),
    "misalign": ("misalign_leaking", "misalign_murder"),
    "pair": ("pair",), "crescendo": ("crescendo",), "actorattack": ("actorattack",),
}
STATIC_LEGS: Tuple[str, ...] = ("pasttense", "codeattack", "renellm", "grok420")
MULTITURN_LEGS: Tuple[str, ...] = ("pair", "crescendo", "actorattack")
LADDER_LEGS: Tuple[str, ...] = STATIC_LEGS + ("controls", "gsm8k", "misalign", "petri")
STANDUP_LEGS: Tuple[str, ...] = tuple(EVAL_LEGS_OF)
CAP_CEILING = 16384

#: the S4-decided fields at their SPEC DEFAULTS (the draft materialisation, §8): the ONLY fields the final may change
S4_DEFAULTS: Dict[str, Any] = {
    "multiturn_window": {"crescendo": 49152, "actorattack": 49152},
    "pools": {"count": 4, "gpu": "H200", "max_model_len": 65536, "chunked_prefill": 16384, "s_att": 10,       # ONE card per pool
              "assignment": ["crescendo", "crescendo", "actorattack", "pair"]},
    "actorattack": {"forwards_per_goal": 24, "containers_per_view": 5, "containers_arditi": 2},
    "crescendo": {"containers_per_view": 4, "containers_arditi": 2},
}
#: the attacker pool's model (attacker_pool.py; spec Inputs)
ATTACKER_MODEL = "huihui-ai/Huihui-Qwen3.6-27B-abliterated"
ATTACKER_SNAPSHOT = "27502c87"

# --- THE MULTI-TURN WINDOW'S PROVENANCE (`geometry.py` "The 4,096-cap multi-turn windows") -------------------------------------
#: THE RULE, verbatim from the serving front: a multi-turn window is sized from a MEASURED MAXIMUM on a COMPLETE store at the
#: cap being served -- never a percentile, never a partial store -- because long conversations finish last, so every early
#: reading is a floor (the same quantity read 8,731 -> 16,428 -> 34,361 -> 41,433 as one store filled). These literals are
#: pinned equal to `sisteer.capeval.serving.geometry.MULTITURN_CAP4096_*` by test; the front's values win when importable.
MULTITURN_WINDOW_RULE = "measured_maximum_on_complete_store_at_the_served_cap"
MULTITURN_CAP4096_CAP = 4096
MULTITURN_CAP4096_WINDOWS: Dict[str, int] = {"crescendo": 49152, "pair": 16384}
MULTITURN_CAP4096_MEASURED_MAX_CONTEXT: Dict[str, int] = {"crescendo": 41433, "pair": 12825}
MULTITURN_CAP4096_SOURCE = ("R2-D253 (a-x): the 4,096-cap "
                            "re-run of constv2-mix-full__lora_r4_ce_minus_ctl on gemma4e4b / qwen35_4b / "
                            "nemotron3nano4b / glm46vflash, cap_4096_timing.context_max of the complete stores")
#: the per-model readings behind Crescendo's 49,152 (the MAXIMUM ACROSS MODELS is glm46vflash); every reading is on ONE
#: object (`constv2-mix-full__lora_r4_ce_minus_ctl`) at dose 1.0 and NONE is under our decode (qwen35_4b at the vendor preset)
MULTITURN_CRESCENDO_READINGS: Dict[str, Any] = {"glm46vflash": 41433, "gemma4e4b": 16428, "nemotron3nano4b": 15462,
                                                "qwen35_4b_sampled": 8597, "qwen35_4b_greedy": 41311}
MULTITURN_MEASURED_OBJECT = "constv2-mix-full__lora_r4_ce_minus_ctl"
#: the served window classes the rule may select from (the S5 roster's multi-turn candidates; S = 524,287 // L)
MULTITURN_WINDOW_CLASSES: Tuple[int, ...] = (16384, 24576, 32768, 40960, 49152, 65536)


def multiturn_window_provenance(leg: str, window_L: int, s4: Mapping[str, Any]) -> Dict[str, Any]:
    """Where a multi-turn leg's window comes from, recorded in the plan so a re-sizing starts from evidence: the front's
    measured-maximum rule and source, the measured maxima (with the model / object / dose / decode they were read under),
    whether the served L is the front's DEFAULT for the attack or an EXPLICIT window (ActorAttack has no measured 4,096-cap
    window and is never selected by default), and the gate a SMALLER window must pass (zero clamped calls on every cell,
    measured on a parent arm AND one dosed object under OUR decode) -- never a percentile, never habit."""
    try:
        from sisteer.capeval.serving import geometry as sg  # type: ignore
        windows, maxima, source = dict(sg.MULTITURN_CAP4096_WINDOWS), dict(sg.MULTITURN_CAP4096_MEASURED_MAX_CONTEXT), sg.MULTITURN_CAP4096_SOURCE
        source_module = "sisteer.capeval.serving.geometry"
    except Exception:  # noqa: BLE001 -- the pinned copies (equal by test)
        windows, maxima, source, source_module = dict(MULTITURN_CAP4096_WINDOWS), dict(MULTITURN_CAP4096_MEASURED_MAX_CONTEXT), MULTITURN_CAP4096_SOURCE, "pinned copy"
    default_L = windows.get(leg)
    s4_L = (s4.get("multiturn_window") or {}).get(leg)
    if default_L is not None and int(window_L) == int(default_L):
        how = f"the front's default for {leg}: MULTITURN_CAP4096_WINDOWS[{leg!r}] = {default_L} (maximum across models)"
    elif s4_L is not None and int(window_L) == int(s4_L):
        how = (f"EXPLICIT (S4 decisions.json multiturn_window[{leg!r}] = {s4_L}); "
               + ("the front has NO measured 4,096-cap window for this attack (deliberately absent; never a default)"
                  if default_L is None else f"the front's default is {default_L}"))
    else:
        how = f"plan §5 literal {window_L}" + ("" if default_L is None else f" (the front's measured window is {default_L})")
    return {
        "rule": MULTITURN_WINDOW_RULE, "cap": MULTITURN_CAP4096_CAP, "window": int(window_L), "kernel_S": kernel_S(int(window_L)),
        "window_default_for_attack": default_L, "how_chosen": how, "source": source, "source_module": source_module,
        "measured_max_context": maxima, "crescendo_readings_by_model": dict(MULTITURN_CRESCENDO_READINGS),
        "measured_on": {"object": MULTITURN_MEASURED_OBJECT, "dose": 1.0, "decode": "the R2-D253 decodes (qwen: sampled and greedy)"},
        "our_decode": f"{MODEL_KEY} vendor_preset@v1 (NO existing reading covers it)",
        "s5_gate": ("measure the context MAXIMUM under OUR decode on a parent arm AND one dosed object per multi-turn leg; "
                    "a window smaller than the default is admissible ONLY with zero clamped calls on every cell of a COMPLETE "
                    "measurement (multiturn_window_gate); percentiles never size a window"),
    }


# --- the peer's dose standups and the window probe's dose cells -----------------------------------------------------------------
#: the peer's seed-1 dose views ride the roster under THIS prefix (append-only, their pins); each gets THREE
#: dedicated single-container standups inside our wave (Crescendo, ActorAttack, XSTest) with a same-standup `none` anchor.
#: They are NEVER registered in our s1 / s2 / s3 standups (matched engines across seeds).
DOSE_VIEW_PREFIX = "dose_"
DOSE_STANDUP_LEGS: Tuple[str, ...] = ("crescendo", "actorattack", "controls")
DOSE_CONTROLS: Tuple[str, ...] = ("xstest",)
# ---- (b) seed-1 dose view discovery by DECLARATION (replaces the `dose_` name-prefix scan) -------------------------------------
#: a SEED-1 dose object's staged id: `<dose token>s1__<parent cell id>` (`d1p5s1__…`, `dm0p5s1__…`,
#: `it_d0p25s1__…`; the `s1` names the seed). The campaign's batch-8 ladders spell `d1p5__…` / `it_d1p5__…` (no seed) and are
#: NEVER served in this wave (O23: batch-8 originals are never served or referenced), so a dose view over them is skipped.
DOSE_S1_STAGED_RE = re.compile(r"^(?:it_)?dm?\d+(?:p\d+)?(?P<eb>_eb64)?s1__(?P<parent>.+)$")
LORA_DIFF_CELL = "constv2-mix-full__lora_r4_ce_minus_ctl"                 # our admitted LoRA difference parent 0d275f7c (derived_base64s1__)
PAIRED_DIFF_CELL = "constv2-mix-full__psr_ce_all_minus_ctl_distilled"     # the peer's paired difference parent (derived_base64s1__..., their bank entry)
#: the parents whose seed-1 children are keyed `<dose>_eb64s1__<parent>` (renamed 2026-09-18): the bare `<dose>s1__` spelling is RETIRED
RENAMED_DIFFERENCE_PARENTS: Tuple[str, ...] = (LORA_DIFF_CELL, PAIRED_DIFF_CELL)


def dose_staged_id_problem(staged_id: str) -> Optional[str]:
    """None when ``staged_id`` is an admissible seed-1 dose child; else why not: not seed-1 (the campaign's batch-8 ladders), or the RETIRED
    `<dose>s1__` spelling of a renamed difference child (Strength keys them `<dose>_eb64s1__<parent>`; the sha table refuses the old id)."""
    m = DOSE_S1_STAGED_RE.match(str(staged_id))
    if not m:
        return "not a seed-1 child (`<dose>[_eb64]s1__<parent>`)"
    if m.group("parent") in RENAMED_DIFFERENCE_PARENTS and not m.group("eb"):
        return (f"RETIRED spelling {staged_id!r}: Strength keys the seed-1 difference children by effective batch AND seed "
                f"(`<dose>_eb64s1__{m.group('parent')}`); the id -> sha table refuses the old id")
    return None
DOSE_DISCOVERY_RULE = ("a view of the target is a DOSE view when its roster block declares `dose:` (main's roster grammar "
                       "`d<tok>[_<family>]`, make_full22_configs.view_dose) or its name carries the `dose_` prefix (D45's phrasing); "
                       "it is stood up in THIS wave when EVERY staged id it pins is a seed-1 child `<tok>s1__<parent>` "
                       "(DOSE_S1_STAGED_RE) whose parent is the view's cell key; a dose view over batch-8 objects is skipped (O23); "
                       "a view mixing the two refuses")
DOSE_SHAPE = ("per_dose_view: one single-container engine per (leg, dose view). The roster grammar keys a view's cells by PARENT "
              "cell id and the harness translates that id into the route, so one engine cannot serve two doses of one parent -- "
              "D43's 'three standups carrying every dose object' holds only when the peer declares ONE view carrying every object; "
              "N one-dose views -> 3N standups (3N containers inside the peak), every one counted in the capacity ledger")
#: the S5 multi-turn window probe registers these TWO dose children of our admitted LoRA difference parent beside a parent
#: arm and measures the Crescendo / ActorAttack context MAXIMUM on each under our vendor-preset decode: Crescendo
#: BACKTRACKS on refusal (up to 10 backtracks on 10 turns), so the hardest-refusing rung buys the most turns -- d1p5 is
#: the stressing end, dm0p5 the other, and the pair gives the DIRECTION of the dose effect on context length.
#: Strength keys the seed-1 difference children by EFFECTIVE BATCH AND SEED (`<dose>_eb64s1__<parent>`, renamed 2026-09-18; the object shas
#: are unchanged). The old `<dose>s1__<parent>` spelling of these six is RETIRED and refused by name (never silently re-mapped).
S5_PROBE_CELLS: Tuple[Dict[str, str], ...] = (
    {"staged_id": f"d1p5_eb64s1__{LORA_DIFF_CELL}", "cell_id": LORA_DIFF_CELL, "dose": "1.5",
     "object_sha256": "86fa8e9c0523bb54e0fbd625796934f400a1c70ebb077e3c24dad667d46a46be", "role": "stressing end (hardest-refusing)"},
    {"staged_id": f"dm0p5_eb64s1__{LORA_DIFF_CELL}", "cell_id": LORA_DIFF_CELL, "dose": "-0.5",
     "object_sha256": "0522a255ab45d57e085f07aea802dbd671072aee68cad627251440396135f0fc", "role": "the other end (terminates earlier)"},
)
S5_PROBE_PARENT_ARM = "none"
# ---- (c) where the S5 probe's two cells are REGISTERED (they ride the peer's dose standups; no extra container) ----------------
#: the two S4-windowed attacks the probe measures (S4_DEFAULTS["multiturn_window"]'s keys; PAIR keeps the front's measured window)
S5_PROBE_LEGS: Tuple[str, ...] = ("crescendo", "actorattack")
#: why PAIR is not S5-gated: the S5 probe measures the
#: two S4-DECIDED windows; PAIR's window is not an S4 field (`S4_DEFAULTS["multiturn_window"]` names crescendo and actorattack only) and
#: the front carries a MEASURED 4,096-cap window for it (`MULTITURN_CAP4096_WINDOWS["pair"]`), so its provenance records the front's
#: measurement and the literal window value instead of an `S5` gate
S5_UNGATED_WHY = ("not S5-gated: PAIR's window is not an S4-decided field and the front carries a MEASURED 4,096-cap window for it "
                  "(MULTITURN_CAP4096_WINDOWS['pair']); the S5 probe measures the S4-decided windows (crescendo, actorattack) only")


def _plan_rows_the_app_may_execute(plan: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    """Every row a container may be stood up from: the executable standups + rungs AND the smoke-only rows (`s5_probe_rows`, the
    window probe's path) -- staging derives what it packages from THIS set, never from one declaration."""
    return list(plan.get("standups") or []) + list(plan.get("rungs") or []) + list(plan.get("s5_probe_rows") or [])


def bank_views_for_plan(plan: Mapping[str, Any]) -> Dict[str, Dict[str, str]]:
    """Every EXTRA bank view a container of this plan may serve beyond s1 / s2 / s3 and the Arditi view -> ``{view: {cell: staged_id}}``,
    DERIVED from the objects table of EVERY row the app may execute (:func:`_plan_rows_the_app_may_execute`: the executable standups and
    rungs AND the smoke-only `s5_probe_rows` -- the window probe's `d1p5_loradiff64s1` / `dm0p5_loradiff64s1` rows live there and
    nowhere else) plus the dormant path's `dose_standups.views`. Reading `dose_standups.views` alone (empty on that path) omits the
    probe objects, and the probe smoke then fails at setup before producing its measurement. One view referenced with two different
    cell sets refuses (the bank can hold one)."""
    out: Dict[str, Dict[str, str]] = {}

    def add(view: str, cells: Mapping[str, str]) -> None:
        cells = {str(c): str(s) for c, s in cells.items()}
        prev = out.get(view)
        if prev is not None and prev != cells:
            raise ValueError(f"bank view {view!r} is referenced with two different cell sets: {prev} vs {cells}")
        out[view] = cells
    for v, b in ((plan.get("dose_standups") or {}).get("views") or {}).items():
        add(str(v), b["cells"])
    for r in _plan_rows_the_app_may_execute(plan):
        view = str(r.get("view") or r.get("engine"))
        if view in ENGINES:
            continue                                     # the fixed views: s1 / s2 / s3 from the roster + pins, the Arditi carrier
        cells = {c: o["staged_id"] for c, o in (r.get("objects") or {}).items() if c != "none" and o.get("staged_id")}
        if cells:
            add(view, cells)
    return out


def bank_pins_for_plan(plan: Mapping[str, Any]) -> Dict[str, str]:
    """``{staged_id: object_sha256}`` for every object an EXTRA view (:func:`bank_views_for_plan`) serves, read off the same rows' objects
    tables -- the plan already held the peer pin file's shas when it built the row (`s5_probe_rows` refuses a disagreeing pin), and
    `configs/sdfbase_full22_pins.json` never carries a peer object, so staging's sha verification needs THESE pins beside the tree's.
    Two rows pinning one staged id under two shas refuse."""
    out: Dict[str, str] = {}
    for r in _plan_rows_the_app_may_execute(plan):
        if str(r.get("view") or r.get("engine")) in ENGINES:
            continue
        for c, o in (r.get("objects") or {}).items():
            sid, sha = o.get("staged_id"), o.get("object_sha256")
            if c == "none" or not sid:
                continue
            if not isinstance(sha, str) or len(sha) != 64:
                raise ValueError(f"{r.get('standup_id')}: object {sid} carries no 64-hex object_sha256 ({sha!r})")
            if out.get(sid, sha) != sha:
                raise ValueError(f"object {sid} is pinned under two shas across rows: {out[sid]} vs {sha}")
            out[sid] = sha
    for v, b in ((plan.get("dose_standups") or {}).get("views") or {}).items():
        for sid, sha in (b.get("pins") or {}).items():
            if isinstance(sha, str) and len(sha) == 64:
                out.setdefault(str(sid), sha)
    return out


def s5_probe_rows(*, peer_roster: Optional[Mapping[str, Any]], peer_pins: Optional[Mapping[str, Any]], legs: Mapping[str, Mapping[str, Any]], W: str, W7: str,
                  wave_uuid: str, reqs_sha256: str, pools_by_attack: Mapping[str, Sequence[Mapping[str, Any]]], adapter_hashes: Mapping[str, str],
                  s4: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The window probe's SMOKE-ONLY standups -- per probe leg (crescendo, actorattack) and per peer view that carries one of the
    two probe cells (`d1p5_eb64s1__…` stresses the window, `dm0p5_eb64s1__…` the other end), ONE single-container row serving `none` (the
    parent arm, same standup) + that probe cell at our geometry class. Never in the production plan or the ledger (kind `s5_probe`, admitted
    by `--smoke` only); without the peer files there is nothing to register and the probe reports UNREGISTERED."""
    if peer_roster is None or peer_pins is None:
        return []
    views = peer_dose_views(peer_roster)
    probe_ids = {c["staged_id"]: c for c in S5_PROBE_CELLS}
    target_pins = peer_pins.get(MODEL_KEY) or {}
    rows: List[Dict[str, Any]] = []
    for view, block in sorted(views.items()):
        cells = {str(c): str(sid) for c, sid in (block.get("cells") or {}).items() if sid in probe_ids}
        if not cells:
            continue
        pins_by_cell = {}
        for c, sid in cells.items():
            entry = target_pins.get(sid) or {}
            if str(entry.get("object_sha256")) != probe_ids[sid]["object_sha256"]:
                raise ValueError(f"S5 probe view {view}: {sid} pinned {entry.get('object_sha256')!r}, not D45's {probe_ids[sid]['object_sha256']}")
            pins_by_cell[c] = probe_ids[sid]["object_sha256"]
        config = str(block.get("config") or VIEW_CONFIG_REL.format(view=view))
        manifest = str(block.get("manifest") or VIEW_MANIFEST_REL.format(view=view))
        es = {"config": config, "manifest": manifest, "bank_view": BANK_VIEW_LOCAL.format(view=view), "cells": list(cells), "digests": dict(pins_by_cell)}
        runs = dose_arm_runs(view, cells)
        routes = ["none", *cells]
        objects = {c: {"staged_id": sid, "object_sha256": pins_by_cell[c], "dose_view": view, "s5_probe_cell": True} for c, sid in cells.items()}
        for leg in S5_PROBE_LEGS:
            spec = legs[leg]
            window = spec["base"]
            geom = served_geometry(window)
            pool = pools_by_attack[leg][0]
            kids = _multiturn_children(leg, view, W, geom["L"], 0, 1, pool["url_placeholder"], runs=runs, cfg=config)
            for k in kids:
                k["dose_view"] = view
                k["registry_identity_rel"] = identity_plan_rel(k["eval_leg"], k["arm"], k["s"])
            sid_ = standup_id(leg, view, STAGE_BASE, 0)
            rows.append({
                "standup_id": sid_, "kind": S5_PROBE_KIND, "leg": leg, "eval_legs": list(EVAL_LEGS_OF[leg]), "engine": view, "view": view, "stage": STAGE_BASE,
                "shard": 0, "nshards": 1, "seed_bases": [1], "s5_probe": True, "smoke_only": True,
                "control_key": control_key(W7, wave_uuid, leg, view, STAGE_BASE, 0), "accepted_key": accepted_key(W7, wave_uuid, leg, view, STAGE_BASE, 0),
                "modal": {"function": "run_view_leg", "gpu": "H200", "cpu": LEG_CPU, "memory_mib": LEG_MEMORY_MIB, "timeout_s": int(spec["timeout_h"] * 3600),
                          "retries": 0, "max_containers": 1},
                "geometry": geom, "argv": standup_argv(leg=spec, window=window, engine=view, W=W, reqs_sha256=reqs_sha256, pins_by_cell={}, arditi_sha256="", spec=es),
                "route_set": routes, "route_set_harness": [harness_id(c) for c in routes], "cells_served_expected": cells_served_for_routes(routes),
                # W15d B3: a probe row is smoke-only and the FIRST exporter of its peer cells by construction -- it asserts no re-use (the
                # production expectation refused the probe's own export at gate 10, the F-L(d) shape on the S5 side)
                "objects": objects, "adapters": adapter_expectations(objects, adapter_hashes, expected_reused=False),
                "readiness": readiness_block(geom, max_lora_rank=VIEW_MAX_LORA_RANK, max_loras=len(cells) + 1, residual_slot_armed=[c for c in cells if is_psr_cell(c)],
                                             max_logprobs=spec.get("max_logprobs"), thinking=False),
                "children": kids, "producers": [], "ladder": None, "rung_consumer": None, "rung_mode": None, "pool": pool["name"], "arm_runs": runs,
                "expected_h200_h": [round(x / 4, 2) for x in spec["h200_h"]], "window_provenance": multiturn_window_provenance(leg, geom["L"], s4),
                "depends_on": depends_on_block(pool["name"]),
                "why": "D45: the S5 window probe measures the context MAXIMUM on this dose child beside the same-standup `none` parent under OUR decode; smoke-only (D71)",
            })
    return rows


def s5_probe_registration(dose_rows: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Per S5 probe cell: the multi-turn DOSE standup (one per :data:`S5_PROBE_LEGS`) whose objects pin its staged id under
    the peer's sha -> ``{staged_id: {registered_on: {leg: standup_id}, sha_mismatch: {leg: {...}}, complete: bool}}``. A same-named
    object pinned under ANOTHER sha is a mismatch, never the probe (the peer's pin binds the bytes)."""
    out: Dict[str, Dict[str, Any]] = {}
    rows = [r for r in dose_rows if (r.get("dose_standup") or r.get("s5_probe")) and r["leg"] in S5_PROBE_LEGS]
    for c in S5_PROBE_CELLS:
        on: Dict[str, str] = {}
        mismatch: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            for o in r["objects"].values():
                if o.get("staged_id") != c["staged_id"]:
                    continue
                if o.get("object_sha256") == c["object_sha256"]:
                    on[r["leg"]] = r["standup_id"]
                else:
                    mismatch[r["leg"]] = {"standup_id": r["standup_id"], "object_sha256": o.get("object_sha256"), "expected": c["object_sha256"]}
        out[c["staged_id"]] = {"registered_on": on, "sha_mismatch": mismatch, "complete": set(on) == set(S5_PROBE_LEGS)}
    return out


def s5_probe_registration_problems(plan: Mapping[str, Any]) -> List[str]:
    """`validate_plan`'s share: a probe cell pinned under another sha, or registered on ONE of the two probe legs only (the probe
    measures Crescendo AND ActorAttack on the SAME object), is a named problem. A cell registered nowhere is not: the peer's
    views may not carry it yet -- the plan RECORDS that (`s5_multiturn_probe.status`) and the FINAL plan refuses without the
    cell's measurements (`multiturn_window_gate`)."""
    problems: List[str] = []
    for r in plan.get("standups") or []:
        for o in (r.get("objects") or {}).values():
            why = dose_staged_id_problem(str(o.get("staged_id", "")))
            if o.get("dose_view") and why and why.startswith("RETIRED"):
                problems.append(f"{r['standup_id']}: {why}")
    reg = (plan.get("s5_multiturn_probe") or {}).get("registration") or {}
    for sid, r in reg.items():
        for leg, m in (r.get("sha_mismatch") or {}).items():
            problems.append(f"S5 probe cell {sid} is pinned on {m['standup_id']} under sha {m['object_sha256']}, not D45's {m['expected']}")
        on = set(r.get("registered_on") or {})
        if on and on != set(S5_PROBE_LEGS):
            problems.append(f"S5 probe cell {sid} is registered on {sorted(on)} only; the probe measures {list(S5_PROBE_LEGS)} on the same object")
    return problems
#: the paired-ReLU dose children of parent f98f0349 MAY also ride the probe standup (a peer registers them); when they do,
#: their readings enter the same record and the same zero-clamped-calls gate. Only sha PREFIXES are known here; the full shas are
#: the peer's pins (never ours to assert).
#: the paired-difference children's shas are the BANK's (sha256(object.pt) under $STAGE/sdfbase/full22_objects/qwen35_4b/cells/<id>/,
#: measured 2026-09-18): the peer's 4bacb54c… / 2bdf2371… / 26039360… (ReLU's children of f98f0349) are NOT what this bank holds under
#: these ids, so the bank bytes bind; the peer pin file must agree or the plan refuses
S5_PROBE_OPTIONAL_CELLS: Tuple[Dict[str, str], ...] = (
    {"staged_id": f"dm0p5_eb64s1__{PAIRED_DIFF_CELL}", "cell_id": PAIRED_DIFF_CELL, "dose": "-0.5",
     "object_sha256": "b7623659949e832b448d492e83ffa4fc73795f8644a2644d66e6acb58dab671e", "role": "paired route, dose -0.5 (optional; bank-measured sha)"},
    {"staged_id": f"d0p5_eb64s1__{PAIRED_DIFF_CELL}", "cell_id": PAIRED_DIFF_CELL, "dose": "0.5",
     "object_sha256": "f12efb896b921f41681dde0eb0d944ccd4011c5e51d0f697a17bb8adb2489424", "role": "paired route, dose 0.5 (optional; bank-measured sha)"},
    {"staged_id": f"d1p5_eb64s1__{PAIRED_DIFF_CELL}", "cell_id": PAIRED_DIFF_CELL, "dose": "1.5",
     "object_sha256": "15ced37c58c6043e6299e20135f7a92eed76d2d3571df013122294282674bdb2", "role": "paired route, dose 1.5 (optional; bank-measured sha)"},
    {"staged_id": f"d0p5_eb64s1__{LORA_DIFF_CELL}", "cell_id": LORA_DIFF_CELL, "dose": "0.5",
     "object_sha256": "669fcc674f929412457b2368c374622647857d36ab18de9372a04d3a03e02276", "role": "LoRA difference, dose 0.5 (the sixth child; not a probe end)"},
)
PAIRED_DIFF_PARENT_STAGED_ID = f"derived_base64s1__{PAIRED_DIFF_CELL}"
PAIRED_DIFF_PARENT_SHA256 = "aca19ab01b21fb0a510db0620bd3f8b82c65da872e488a36d5a875f052165a0a"                    # bank-measured 2026-09-18
#: the peer's six views, as the peer names them (object family + effective batch + seed); a peer roster with other names still registers
PEER_EXPECTED_VIEWS: Tuple[str, ...] = ("dm0p5_loradiff64s1", "d0p5_loradiff64s1", "d1p5_loradiff64s1", "dm0p5_pairdiff64s1", "d0p5_pairdiff64s1", "d1p5_pairdiff64s1")
PEER_ROSTER_REL = "configs/sdfbase_full22_roster_strength_eb64_s1.yaml"
PEER_PINS_REL = "configs/strength_dose_eb64_s1_pins.json"
#: the peer-shaped roster + pins owned here that register the window probe's two cells under the views
#: `d1p5_lora` / `dm0p5_lora` -- the generator's dose grammar with a family it knows (`d<tok>_<object family>`; the peer's
#: `d1p5_loradiff64s1` names a family make_full22_configs.py refuses by name). `main()` resolves to them when no --peer-roster /
#: --peer-pins is given and both exist under --repo (additive: an explicit pair, the peer's default names included, still wins).
PROBE_ROSTER_REL = "configs/sdfbase_full22_roster_seedrep_probe_eb64_s1.yaml"
PROBE_PINS_REL = "configs/seedrep_probe_eb64_s1_pins.json"
PROBE_VIEWS: Tuple[str, ...] = ("d1p5_lora", "dm0p5_lora")
PEER_DISCOVERY_RULE = ("D68: the dose standups register from the PEER's roster + pin file, given BY PATH (--peer-roster / --peer-pins): every view of the "
                       "target found there must be a seed-1 dose view (every staged id `<dose>_eb64s1__<parent>` keyed under its parent cell; anything else "
                       "refuses by name); the shared roster is NEVER read for peer views -- its `dose:` ladders over batch-8 objects are skipped (O23) and "
                       "any seed-1 dose view found there is recorded, not stood up; with no peer files the standups are pending_peer_spec")
DOSE_STATUS_PENDING = "pending_peer_spec"
#: the peer's dose sweep runs on cluster A, NOT in this Modal wave -- no dose standups in the plan or the
#: ledger. The peer-roster registration path is kept as a DORMANT, default-off capability (`register_peer_dose_standups=True` /
#: `--register-peer-dose-standups`); the S5 multi-turn window probe is UNCHANGED and registers its two cells through smoke-only
#: `s5_probe_rows` built from the same peer files.
DOSE_STATUS_NOT_IN_WAVE = "not_in_this_wave"
DOSE_D71_WHY = ("D71: Strength's dose sweep runs on cluster A, not in this Modal wave -- no dose standups, no ledger line; the peer-roster "
                "registration path is a dormant default-off capability (--register-peer-dose-standups)")
S5_PROBE_KIND = "s5_probe"
#: the two difference parents whose seed-1 children Strength renamed; their staged-id -> sha table, every spelling `_eb64s1`
DOSE_CHILD_SHA: Dict[str, str] = {c["staged_id"]: c["object_sha256"] for c in S5_PROBE_CELLS + S5_PROBE_OPTIONAL_CELLS}
#: the window MEASUREMENT record -- the measurement wave produces the FIRST measured ActorAttack window at cap 4,096 and two peers
#: adopt the number, so every record carries its provenance prominently and is copied into the FINAL plan under the window it
#: justifies (`multiturn_window_provenance[leg].s5_measurements`) and referenced from RS/SHA_PROVENANCE.md.
S5_WINDOW_MEASUREMENT_SCHEMA = "seedrep_s5_window_measurement/1"
S5_WINDOW_SOURCE = "measured_max_complete_store"
S5_MEASUREMENT_REQUIRED: Tuple[str, ...] = ("schema", "window_source", "leg", "arm", "arm_kind", "model_key", "decode_policy_ref", "objects_measured",
                                            "store_path", "n_goals_completed", "n_goals_expected", "store_complete", "context_max", "clamped_calls",
                                            "cap", "measured_utc", "wave_uuid", "attempt_id", "decode")
#: the multi-turn TARGET's decode row (exact ref, never a default lookup). The landed design (5532d04a): the target
#: is invoked with BOTH `--multiturn-target-seed-base <literal s>` AND `--multiturn-target-policy-ref <this>` (+ `--multiturn-target-decode
#: vendor`); a ref-only invocation under `per_turn` REFUSES and the seed-base flag alone resolves the row unconditionally
MULTITURN_DECODE_POLICY_REF = f"{MODEL_KEY}_it|multi_turn|off|vendor_preset@v1"
ARDITI_DECODE_POLICY_REF = MULTITURN_DECODE_POLICY_REF


def window_measurement_problems(rec: Mapping[str, Any]) -> List[str]:
    """What an S5 multi-turn window measurement must carry, each absence / disagreement NAMED: the schema, the source
    (`measured_max_complete_store`), our model and decode row ref, the object(s) measured, the store path, the goals completed
    (== expected, else the store is a FLOOR and `store_complete` must be false), the context maximum (prompt + reply), the cap,
    the date and the wave / attempt id."""
    problems = [f"missing {k}" for k in S5_MEASUREMENT_REQUIRED if k not in rec]
    if problems:
        return problems
    if rec["schema"] != S5_WINDOW_MEASUREMENT_SCHEMA:
        problems.append(f"schema {rec['schema']!r} != {S5_WINDOW_MEASUREMENT_SCHEMA}")
    if rec["window_source"] != S5_WINDOW_SOURCE:
        problems.append(f"window_source {rec['window_source']!r} != {S5_WINDOW_SOURCE} (never a percentile)")
    if rec["model_key"] != MODEL_KEY or rec["decode"] != "vendor" or rec["decode_policy_ref"] != ARDITI_DECODE_POLICY_REF:
        problems.append(f"measured under {rec['model_key']}/{rec['decode']}/{rec['decode_policy_ref']}, not {MODEL_KEY}/vendor/{ARDITI_DECODE_POLICY_REF}")
    if rec["leg"] not in MULTITURN_LEGS:
        problems.append(f"leg {rec['leg']!r} is not a multi-turn leg")
    if rec["arm_kind"] not in ("parent", "dosed"):
        problems.append(f"arm_kind {rec['arm_kind']!r} must be parent | dosed")
    if not isinstance(rec["objects_measured"], list) or not rec["objects_measured"]:
        problems.append("objects_measured must list the object(s) served for this arm ([] for a parent arm is not a list of objects: name 'none')")
    complete = int(rec["n_goals_completed"]) == int(rec["n_goals_expected"]) and int(rec["n_goals_expected"]) > 0
    if bool(rec["store_complete"]) != complete:
        problems.append(f"store_complete {rec['store_complete']!r} disagrees with {rec['n_goals_completed']}/{rec['n_goals_expected']} goals (a partial store is a FLOOR)")
    if int(rec["cap"]) != MULTITURN_CAP4096_CAP:
        problems.append(f"cap {rec['cap']} != the served cap {MULTITURN_CAP4096_CAP}")
    if int(rec["context_max"]) <= 0:
        problems.append("context_max must be the observed maximum prompt + reply tokens (> 0)")
    try:
        check_wave_uuid(rec["wave_uuid"])
    except ValueError as e:
        problems.append(str(e))
    return problems


def s5_multiturn_probe(s4: Mapping[str, Any], dose_rows: Iterable[Mapping[str, Any]] = ()) -> Dict[str, Any]:
    """The multi-turn window probe's declaration: the two dose cells + the parent arm, the legs, our decode, what is
    measured, the gate, and WHERE the two cells are registered (`registration` / `status`: they ride the peer's multi-turn
    dose standups -- two extra adapters on standups already coming up, no extra container). Bytes stay in the shared bank
    (`cells/<staged_id>/`), pinned by the peer's dose view blocks."""
    reg = s5_probe_registration(dose_rows)
    missing = [sid for sid, r in reg.items() if not r["complete"]]
    status = ("registered: both probe cells ride the crescendo AND actorattack S5 probe standups (smoke-only, beside the `none` parent)" if not missing else
              f"UNREGISTERED {missing}: no peer view pins them under D45's sha on both probe legs (--peer-roster / --peer-pins) -- S5 cannot "
              "measure the probe from this plan; the FINAL plan refuses without their measurements (multiturn_window_gate)")
    return {"legs": list(S5_PROBE_LEGS), "parent_arm": S5_PROBE_PARENT_ARM, "cells": [dict(c) for c in S5_PROBE_CELLS],
            "registration": reg, "status": status,
            "decode": f"{MODEL_KEY} vendor_preset@v1", "measure": "context MAXIMUM (prompt + reply) per (leg, arm) on a COMPLETE store",
            "windows_probed": {leg: int(s4["multiturn_window"][leg]) for leg in S5_PROBE_LEGS},
            "gate": "no window below 49,152 unless EVERY reading (parent AND both dose cells) fits with ZERO clamped calls on every Crescendo cell",
            "rationale": ("Crescendo backtracks on refusal (up to 10 backtracks on 10 turns): the hardest-refusing rung buys the most turns; "
                          "d1p5 stresses the window, dm0p5 terminates earlier; the pair gives the direction of the dose effect"),
            "bank": f"$STAGE/sdfbase/full22_objects/{MODEL_KEY}/cells/<staged_id>/ (shared; the peer pins)",
            "optional_cells": [dict(c) for c in S5_PROBE_OPTIONAL_CELLS],
            "measurement_record": {"schema": S5_WINDOW_MEASUREMENT_SCHEMA, "required": list(S5_MEASUREMENT_REQUIRED), "window_source": S5_WINDOW_SOURCE,
                                   "where": "RS/smokes.json S5.multiturn[] -> copied into the FINAL plan's multiturn_window_provenance[leg].s5_measurements "
                                            "and RS/window_justification.json (referenced from RS/SHA_PROVENANCE.md)"},
            "first_measured": "ActorAttack at cap 4,096 has NO prior measurement anywhere (D50); our S5 record is the first, and peer lanes adopt it"}


def peer_dose_views(peer_roster: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Every view of the target in the PEER roster (Strength's own file): each must be a seed-1 dose view -- every staged id an
    admissible `<dose>_eb64s1__<parent>` child (:func:`dose_staged_id_problem`) keyed under its parent cell -- else refuse by name."""
    t = ((peer_roster.get("targets") or {}).get(MODEL_KEY) or {})
    views: Dict[str, Dict[str, Any]] = {}
    for view, block in (t.get("views") or {}).items():
        if not isinstance(block, Mapping):
            raise ValueError(f"peer view {view}: not a mapping")
        cells = {str(c): str(sid) for c, sid in (block.get("cells") or {}).items() if sid is not None}
        if not cells:
            raise ValueError(f"peer view {view}: pins no cells")
        for c, sid in cells.items():
            why = dose_staged_id_problem(sid)
            if why:
                raise ValueError(f"peer view {view}: {sid}: {why}")
            if DOSE_S1_STAGED_RE.match(sid).group("parent") != c:
                raise ValueError(f"peer view {view}: staged id {sid} is keyed under the cell {c!r}, not its parent -- the route would name a different cell")
        views[str(view)] = block
    if not views:
        raise ValueError(f"the peer roster carries no views of {MODEL_KEY}")
    return views


def dose_views(roster: Mapping[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    """The peer's SEED-1 dose views of the target -> ``({view: block}, {skipped view: why})`` per :data:`DOSE_DISCOVERY_RULE`.
    A candidate (declares `dose:` or is named `dose_*`) is admitted when every staged id is a seed-1 child of its cell key,
    skipped when none is (the campaign's batch-8 ladders; recorded, never served), and REFUSES when it mixes the two or keys a
    child under another parent's cell (the object would be served under a route that names a different cell)."""
    t = ((roster.get("targets") or {}).get(MODEL_KEY) or {})
    views: Dict[str, Dict[str, Any]] = {}
    skipped: Dict[str, str] = {}
    for view, block in (t.get("views") or {}).items():
        if not isinstance(block, Mapping):
            continue
        declared = ("dose" in block) or str(view).startswith(DOSE_VIEW_PREFIX)
        if not declared:
            continue
        cells = {str(c): str(sid) for c, sid in (block.get("cells") or {}).items() if sid is not None}
        if not cells:
            skipped[str(view)] = "declares a dose but pins no cells"
            continue
        retired = {c: dose_staged_id_problem(sid) for c, sid in cells.items() if (dose_staged_id_problem(sid) or "").startswith("RETIRED")}
        if retired:
            raise ValueError(f"dose view {view}: {'; '.join(retired.values())}")
        seed1 = {c: DOSE_S1_STAGED_RE.match(sid) for c, sid in cells.items()}
        hits = {c: m for c, m in seed1.items() if m is not None}
        if not hits:
            skipped[str(view)] = (f"not seed-1 children ({sorted(cells.values())}): the campaign's batch-8 ladder objects are never "
                                  "served in this wave (O23)")
            continue
        if len(hits) != len(cells):
            raise ValueError(f"dose view {view}: mixes seed-1 children {sorted(hits)} with other objects "
                             f"{sorted(set(cells) - set(hits))} -- one view, one object generation")
        for c, m in hits.items():
            if m.group("parent") != c:
                raise ValueError(f"dose view {view}: staged id {cells[c]} is a child of {m.group('parent')!r} but is keyed under the cell "
                                 f"{c!r} -- the route would name a different cell than the object served")
        views[str(view)] = block
    return views, skipped


def multiturn_window_gate(leg: str, window_L: int, measurements: Iterable[Mapping[str, Any]], *, cap: int = MULTITURN_CAP4096_CAP,
                          wave_uuid: Optional[str] = None) -> Dict[str, Any]:
    """S5's assertion for one multi-turn leg's window: over the S5 measurement RECORDS of THIS leg (the record schema, EVERY row --
    a bare reading with no provenance is refused, never admitted on its numbers) require (a) at least one
    parent-arm and BOTH probe-cell measurements, (b) every record complete and self-consistent (`window_measurement_problems`:
    the source `measured_max_complete_store`, our model / decode row, the store path, goals completed == expected, the attempt
    identity), (c) every store COMPLETE, (d) ZERO clamped calls on every cell, (e) every context maximum (prompt + reply, so the
    cap is already inside it) within the window, (f) when ``wave_uuid`` is given, every record measured under THIS wave (exact
    equality -- a syntactically valid foreign wave's record is refused). Also reports the window the RULE would select from
    these maxima (the smallest window class >= the measured maximum) with its S and the implied container count, so a smaller
    window is a decision taken on evidence -- the report's number, never a habit."""
    rows = [m for m in measurements if m.get("leg") == leg]
    problems: List[str] = []
    for m in rows:
        who = f"{leg}/{m.get('arm')}"
        if m.get("schema") is None:
            problems.append(f"{who}: a bare reading (no `schema`) is not an S5 measurement record -- every admitted row carries the "
                            f"{S5_WINDOW_MEASUREMENT_SCHEMA} provenance (window_source, model, decode row, store, goals, attempt, wave)")
            continue
        problems += [f"{who}: {p}" for p in window_measurement_problems(m)]
        if wave_uuid is not None and m.get("wave_uuid") != wave_uuid:
            problems.append(f"{who}: measured under wave {m.get('wave_uuid')!r}, not the S5 smoke's execution wave {wave_uuid} (a foreign wave's store)")
    kinds = {str(m.get("arm_kind")) for m in rows}
    arms = {str(m.get("arm")) for m in rows}
    if "parent" not in kinds:
        problems.append(f"{leg}: no measurement on a PARENT arm (none / prompted)")
    if "dosed" not in kinds:
        problems.append(f"{leg}: no measurement on a DOSED object (steering strength changes conversation length)")
    for c in S5_PROBE_CELLS:
        if c["staged_id"] not in arms:
            problems.append(f"{leg}: no measurement on the probe cell {c['staged_id']} ({c['role']}; D45 brackets the dose axis with both)")
    for m in rows:
        if m.get("schema") is None:
            continue                                     # refused above; its numbers never enter the maximum
        who = f"{leg}/{m.get('arm')}"
        if not m.get("store_complete"):
            problems.append(f"{who}: the store is not COMPLETE -- every early reading is a floor, never a maximum")
        if int(m.get("clamped_calls", -1)) != 0:
            problems.append(f"{who}: {m.get('clamped_calls')} clamped call(s) -- the window is too small for this cell")
        if int(m.get("context_max", 0)) > int(window_L):
            problems.append(f"{who}: context maximum {m.get('context_max')} exceeds the window {window_L}")
    measured_max = max((int(m.get("context_max", 0)) for m in rows if m.get("schema") is not None), default=None)
    would_select = None
    if measured_max is not None and not problems:
        would_select = next((c for c in MULTITURN_WINDOW_CLASSES if c >= measured_max), None)
    return {"leg": leg, "window": int(window_L), "cap": int(cap), "n_measurements": len(rows), "measured_max_context": measured_max,
            "rule_would_select": (None if would_select is None else implied_multiturn_shape(would_select)),
            "ok": not problems, "problems": problems}


def window_justification(plan: Mapping[str, Any], measurements: Iterable[Mapping[str, Any]], *, execution_wave_uuid: str,
                         launch_set: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Per multi-turn leg, the gate verdict over the probe records and the records themselves, copied under the window they
    justify (`plan.multiturn_window_provenance[leg].s5_measurements` / `.gate`) and returned as one document for
    RS/window_justification.json (RS/SHA_PROVENANCE.md references it). A leg whose window is not justified by a complete,
    zero-clamped measurement set is a NAMED problem; the FINAL plan refuses on any. ``execution_wave_uuid`` = the probe smoke's
    wave: the records were measured there, never under the production wave, which the document records as the target."""
    ms = list(measurements)
    check_wave_uuid(execution_wave_uuid)
    if execution_wave_uuid == plan["wave_uuid"]:
        raise ValueError("the S5 execution wave equals the production wave: no smoke runs under the production wave")
    # fc10b: with a launch set, the gate is ASSERTED only for the multi-turn legs that have a launched row -- every gated leg's verdict
    # is still recorded; with none launched the document reads `multiturn: none_launched` (certify's spelling). launch_set None = the whole roster.
    launched_mt = None if launch_set is None else sorted({str(r["leg"]) for r in launched_plan_rows(plan, launch_set) if r.get("leg") in MULTITURN_LEGS})
    out: Dict[str, Any] = {"schema": "seedrep_window_justification/1", "W": plan["W"], "wave_uuid": plan["wave_uuid"],
                           "production_wave_uuid": plan["wave_uuid"], "execution_wave_uuid": execution_wave_uuid, "launch_plan_sha": plan.get("launch_plan_sha"),
                           "identity_rule": WAVE_IDENTITY_RULE, "legs": {}, "problems": [], "gated_legs": list(S5_PROBE_LEGS), "ungated_legs": {},
                           "launch_set": (sorted({str(x) for x in launch_set}) if launch_set is not None else None),
                           "multiturn": ({"launched_legs": list(MULTITURN_LEGS), "scope": "whole roster"} if launched_mt is None
                                         else (S5_MULTITURN_NONE_LAUNCHED if not launched_mt else {"launched_legs": launched_mt})),
                           "asserted_legs": [], "unasserted_legs": {}}
    for leg in MULTITURN_LEGS:
        prov = plan["multiturn_window_provenance"][leg]
        if leg not in S5_PROBE_LEGS:
            # the gated legs are DERIVED from the probe declaration (crescendo, actorattack -- the S4-decided windows);
            # PAIR is not S4-decided and not probed: its window is the front's MEASURED 4,096-cap window, recorded
            prov["gate"] = {"leg": leg, "gated": False, "why": S5_UNGATED_WHY}
            prov["s5_measurements"] = [m for m in ms if m.get("leg") == leg]
            out["ungated_legs"][leg] = {"window": prov["window"], "how_chosen": prov["how_chosen"], "why": S5_UNGATED_WHY}
            continue
        gate = multiturn_window_gate(leg, int(prov["window"]), ms, wave_uuid=execution_wave_uuid)
        recs = [m for m in ms if m.get("leg") == leg]
        prov["s5_measurements"] = recs
        prov["gate"] = gate
        asserted = launched_mt is None or leg in launched_mt
        out["legs"][leg] = {"window": prov["window"], "kernel_S": prov["kernel_S"], "how_chosen": prov["how_chosen"], "gate": gate, "measurements": recs,
                            "asserted": asserted}
        if asserted:
            out["asserted_legs"].append(leg)
            if not gate["ok"]:
                out["problems"] += [f"{leg} window {prov['window']}: {p}" for p in gate["problems"]]
        else:
            out["unasserted_legs"][leg] = f"no row of leg {leg!r} is in the launch set (D145 / D146): the D50 gate is recorded, not asserted"
    return out


def implied_multiturn_shape(L: int, *, reference_L: int = 49152, reference_containers_per_view: int = 4) -> Dict[str, Any]:
    """What a multi-turn window L implies: S = 524,287 // L and the container count per view that keeps the driver total of
    the 49,152 x 4-container shape (drivers ~ 2 per served slot -> containers ~ 1 / S)."""
    S = kernel_S(int(L))
    S_ref = kernel_S(int(reference_L))
    return {"L": int(L), "S": S, "containers_per_view": max(1, -(-reference_containers_per_view * S_ref // S)),
            "reference": {"L": int(reference_L), "S": S_ref, "containers_per_view": reference_containers_per_view}}


def leg_table(s4: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """The per-leg specification of §5 under the S4 decisions ``s4`` (schema = :data:`S4_DEFAULTS`)."""
    mt_L = s4["multiturn_window"]
    aa, cr = s4["actorattack"], s4["crescendo"]

    def static(name: str, items: int) -> Dict[str, Any]:
        return {
            "leg": name, "mode": "attacks", "kind": "static", "items_per_arm_run": items, "cap": 4096,
            "base": WindowSpec("families_served", 11520, table=True, thinking=False, cap=4096),
            "rungs": {8192: WindowSpec("families_served", 16384, table=False, thinking=False, cap=8192),
                      16384: WindowSpec("families_served", 24576, table=False, thinking=False, cap=16384)},
            # rung functions are DECLARED at 1 container per (leg, view|arditi, cap) in the ledger (§13: static 4 x 4 x 2 = 32);
            # the burst to 2 when a static queue exceeds 4 x S_rung (§7.2) is a spin-time decision, recorded, never pre-counted
            "max_logprobs": None, "containers_per_view": 1, "containers_arditi": 1, "rung_max_containers": 1, "rung_burst_max": 2,
            "timeout_h": 16, "h200_h": (6, 8),
        }

    legs: Dict[str, Dict[str, Any]] = {
        "pasttense": static("pasttense", 518), "codeattack": static("codeattack", 777),
        "renellm": static("renellm", 777), "grok420": static("grok420", 518),
        "bon": {
            "leg": "bon", "mode": "attacks", "kind": "bon", "items_per_arm_run": 7500, "cap": 4096,
            "base": WindowSpec("families_served", 11520, table=True, thinking=False, cap=4096), "rungs": {},
            "max_logprobs": None, "containers_per_view": 1, "containers_arditi": 1, "rung_max_containers": 0,
            "timeout_h": 16, "h200_h": (13, 16),
        },
        "controls": {
            "leg": "controls", "mode": "attacks", "kind": "controls", "items_per_arm_run": 850, "cap": 512,
            "base": WindowSpec("capability", 7424, table=True, thinking=False, cap=512),
            # THREE transitions, immediate ancestry (§7.4 rev 3): 512 -> 2,048 -> 8,192 -> 16,384
            "rungs": {2048: WindowSpec("capability", 10240, table=False, thinking=False, cap=2048),
                      8192: WindowSpec("capability", 16384, table=False, thinking=False, cap=8192),
                      16384: WindowSpec("capability", 24576, table=False, thinking=False, cap=16384)},
            "max_logprobs": MMLU_TOP_LOGPROBS, "containers_per_view": 1, "containers_arditi": 1, "rung_max_containers": 1,
            "timeout_h": 16, "h200_h": (1.5, 2),
        },
        "gsm8k": {
            "leg": "gsm8k", "mode": "attacks", "kind": "gsm8k", "items_per_arm_run": 200, "cap": 8192,
            "base": WindowSpec(GSM8K_COMMON_EVAL, GSM8K_COMMON_L, table=True, thinking=True, cap=8192),
            "rungs": {16384: WindowSpec(GSM8K_RUNG_COMMON_EVAL, GSM8K_RUNG_COMMON_L, table=True, thinking=True, cap=16384)},
            "max_logprobs": None, "containers_per_view": 1, "containers_arditi": 1, "rung_max_containers": 1,
            "timeout_h": 16, "h200_h": (3, 4),
        },
        "petri": {
            "leg": "petri", "mode": "petri", "kind": "petri", "items_per_arm_run": 60, "cap": 1024,
            "base": WindowSpec("petri", 32768, table=False, thinking=False, cap=1024),
            "rungs": {STAGE_PETRI_RERUN: WindowSpec("petri", 49152, table=False, thinking=False, cap=1024)},
            "max_logprobs": None, "containers_per_view": 1, "containers_arditi": 1, "rung_max_containers": 1,
            "timeout_h": 12, "h200_h": (6, 8),
        },
        "misalign": {
            "leg": "misalign", "mode": "petri", "kind": "misalign", "items_per_arm_run": 1000, "cap": 3000,
            "base": WindowSpec("petri", 24576, table=False, thinking=False, cap=3000),
            "rungs": {16384: WindowSpec("petri", 40960, table=False, thinking=False, cap=16384)},
            "max_logprobs": None, "containers_per_view": 1, "containers_arditi": 1, "rung_max_containers": 1,
            "timeout_h": 16, "h200_h": (9, 12),
        },
        "pair": {
            "leg": "pair", "mode": "attacks", "kind": "multiturn", "items_per_arm_run": 259, "cap": 4096,
            "base": WindowSpec("multiturn_attacks", 24576, table=False, thinking=False, cap=4096), "rungs": {},
            "max_logprobs": None, "containers_per_view": 2, "containers_arditi": 1, "rung_max_containers": 0,
            "timeout_h": 14, "h200_h": (55, 65),
        },
        "crescendo": {
            "leg": "crescendo", "mode": "attacks", "kind": "multiturn", "items_per_arm_run": 259, "cap": 4096,
            "base": WindowSpec("multiturn_attacks", int(mt_L["crescendo"]), table=False, thinking=False, cap=4096),
            "rungs": {}, "max_logprobs": None,
            "containers_per_view": int(cr["containers_per_view"]), "containers_arditi": int(cr["containers_arditi"]),
            "rung_max_containers": 0, "timeout_h": 14, "h200_h": (120, 140),
        },
        "actorattack": {
            "leg": "actorattack", "mode": "attacks", "kind": "multiturn", "items_per_arm_run": 259, "cap": 4096,
            "base": WindowSpec("multiturn_attacks", int(mt_L["actorattack"]), table=False, thinking=False, cap=4096),
            "rungs": {}, "max_logprobs": None,
            "containers_per_view": int(aa["containers_per_view"]), "containers_arditi": int(aa["containers_arditi"]),
            "rung_max_containers": 0, "timeout_h": 14, "h200_h": (140, 160),
            "forwards_per_goal": int(aa["forwards_per_goal"]),
        },
    }
    assert tuple(legs) == STANDUP_LEGS
    return legs


# --------------------------------------------------------------------------- #
# geometry: S is DERIVED (the core's rule), never typed                         #
# --------------------------------------------------------------------------- #


def identity_seqs_for(L: int) -> int:
    """The identity S the core derives for qwen35_4b at window ``L`` — through the REAL front (``geometry.budget_for``
    + the model entry's ``seqs_at``: the hybrid per-slot cap and any measured window ceiling applied) when the serving
    tree is importable, else the same rule offline (``min(budget // L, 256)``; a pinned test holds the two equal)."""
    try:
        from sisteer.capeval.serving import geometry as sg  # type: ignore
        return int(sg.budget_for(MODEL_KEY).entry.seqs_at(int(L)))
    except Exception:  # noqa: BLE001 -- the offline rule, pinned equal to the front by test
        return min(QWEN35_4B_BUDGET // int(L), HYBRID_SEQS_CAP)


def kernel_S(L: int) -> int:
    """The served S at window ``L``: the LoRA kernel index ceiling ``524,287 // L``."""
    s = KERNEL_CEILING // int(L)
    if s < 1:
        raise ValueError(f"window {L} alone exceeds the LoRA kernel index ceiling {KERNEL_CEILING:,}")
    return s


def served_geometry(w: Mapping[str, Any]) -> Dict[str, Any]:
    """What the container must ATTEST for window ``w``: the identity S the core resolves, the kernel-lowered S it
    serves, the adjustment record it writes (``{kind: lora_kernel_index_ceiling, from_S, to_S, D, ceiling}``, or none
    when the identity already fits), and the argv factors (a TABLE window passes none; a non-table window the identity
    PAIR, so ``strict_identity`` admits it and the kernel step lowers it)."""
    L = int(w["L"])
    s_id = identity_seqs_for(L)
    s_srv = min(s_id, kernel_S(L))
    adj = None
    if s_id * L > KERNEL_CEILING:
        adj = {"kind": "lora_kernel_index_ceiling", "from_S": s_id, "to_S": s_srv, "D": LORA_KERNEL_D, "ceiling": KERNEL_CEILING}
    # THE WINDOW'S SOURCE SURVIVES THE LOWERING (validated at ab6f7c2e): the core re-resolves a
    # TABLE-sourced window from the table (max_model_len=None) beside the lowered S, so the SERVED block keeps
    # `window_source: table` + the adjustment record; a NON-table window is passed back explicitly and keeps `override`.
    # The readers admit `table` by the primary rule and `override`+adjustment by the carve-out — both shapes are read.
    return {"eval": w["eval"], "L": L, "table": bool(w["table"]), "thinking": bool(w["thinking"]), "cap": int(w["cap"]),
            "identity_S": s_id, "S": s_srv, "product": s_srv * L, "kernel_ceiling": KERNEL_CEILING, "D": LORA_KERNEL_D,
            "expected_adjustment": adj,
            "expected_window_source": "table" if w["table"] else "override",
            "argv_factors": [] if w["table"] else ["--max-model-len", str(L), "--max-num-seqs", str(s_id)]}


# --------------------------------------------------------------------------- #
# the control-plane key schema (§4.6 rev 3; proof-set gate 6)                   #
# --------------------------------------------------------------------------- #

DICT_PURPOSES: Tuple[str, ...] = ("accepted", "claims", "results", "heartbeats", "sentinels", "stop", "reattempts")   # `reattempts`: the durable per-standup ledger
KEY_AXES: Tuple[str, ...] = ("W7", "wave_uuid", "leg", "view", "stage", "shard")
#: the wave's identity is a DASHED uuid4 string (`str(uuid.uuid4())`, RS/wave.json): `SixAxisKey` and
#: `wave_uuid8()` both refuse any other spelling, so a hex-only uuid would make every peer refuse the wave's keys
WAVE_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
#: the `accepted` Dict is keyed in `seedrep_ladder.key_str`'s spelling (`W7|wave_uuid|leg|view|stage|shard`): the
#: rung consumer reads the accepted projection by THAT key, so the base container must accept under it. The sentinels /
#: heartbeats / stop Dicts keep the `/`-joined `control_key` — the spelling the Petri drive reads.
ACCEPTED_KEY_SEP = "|"


def check_wave_uuid(wave_uuid: str) -> str:
    if not isinstance(wave_uuid, str) or not WAVE_UUID_RE.match(wave_uuid):
        raise ValueError(f"wave_uuid {wave_uuid!r} is not a dashed uuid4 string (RS/wave.json, minted with str(uuid.uuid4()); "
                         "WS2-E's SixAxisKey and WS2-F's wave_uuid8 refuse any other spelling)")
    return wave_uuid


def wave_uuid8(wave_uuid: str) -> str:
    return check_wave_uuid(wave_uuid).replace("-", "")[:8]


def dict_name(W7: str, wave_uuid: str, purpose: str) -> str:
    """``seedrep-<W7>-<wave_uuid8>-<purpose>`` — every persistent Dict of a wave; a dead wave's state is unreachable."""
    if purpose not in DICT_PURPOSES:
        raise ValueError(f"dict purpose {purpose!r} is not one of {DICT_PURPOSES}")
    if len(W7) != 7:
        raise ValueError(f"dict_name needs W7 (7 hex), got {W7!r}")
    return f"seedrep-{W7}-{wave_uuid8(wave_uuid)}-{purpose}"


def accepted_key(W7: str, wave_uuid: str, leg: str, view: str, stage: Any, shard: Any) -> str:
    """The `accepted` Dict key of ONE attempt row in the ``seedrep_ladder.key_str(SixAxisKey(...))`` spelling:
    the six axes joined by ``|``. Pinned equal to the real `seedrep_ladder` by test."""
    parts = control_key(W7, wave_uuid, leg, view, stage, shard).split("/")
    return ACCEPTED_KEY_SEP.join(parts)


def control_key(W7: str, wave_uuid: str, leg: str, view: str, stage: Any, shard: Any, *extra: Any) -> str:
    """The six-axis key every Dict entry carries: ``W7/wave_uuid/leg/view/stage/shard[/extra...]`` — a base attempt and
    its rung attempts never collide, two waves never share a key. ``extra`` (an item key for claims / results) is appended
    after the six axes. Every axis is required and slash-free."""
    axes = {"W7": W7, "wave_uuid": wave_uuid, "leg": leg, "view": view, "stage": stage, "shard": shard}
    for name, v in axes.items():
        s = str(v)
        if not s or s == "None" or "/" in s:
            raise ValueError(f"control_key axis {name} must be a non-empty slash-free value, got {v!r}")
    return "/".join(str(axes[a]) for a in KEY_AXES) + "".join(f"/{x}" for x in extra)


def parse_control_key(key: str) -> Dict[str, Any]:
    parts = key.split("/")
    if len(parts) < len(KEY_AXES):
        raise ValueError(f"control key {key!r} lacks the six axes {KEY_AXES}")
    out: Dict[str, Any] = dict(zip(KEY_AXES, parts[: len(KEY_AXES)]))
    out["extra"] = parts[len(KEY_AXES):]
    return out


# --------------------------------------------------------------------------- #
# argv composition — the front-end contract (§4.3)                              #
# --------------------------------------------------------------------------- #


def serving_venv_python(reqs_sha256: str) -> str:
    """common.sh f22_serving_venv: ``/home/$USER/venvs/sisteer-petri-vllm-<sha[:8]>/bin/python`` (strength's image builds it)."""
    return f"{VENVS_LOCAL}/sisteer-petri-vllm-{reqs_sha256[:8]}/bin/python"


def driver_venv_python(W: str) -> str:
    """common.sh f22_driver_venv: ``/home/$USER/venvs/f22-<sha[:12]>/bin/python`` (built in-container by `uv sync --frozen`)."""
    return f"{VENVS_LOCAL}/f22-{W[:12]}/bin/python"


def tree_local(W: str) -> str:
    return f"{SHARED_LOCAL}/si-steering-eval-repo/{W}"


def view_pins(pins: Mapping[str, Any], roster_cells: Mapping[str, str]) -> Dict[str, str]:
    """``{harness cell id: object_sha256}`` for a view from the roster's ``cells`` map (cell -> staged id) + the pins file."""
    out = {}
    target = pins.get(MODEL_KEY) or {}
    for cell, staged in roster_cells.items():
        entry = target.get(staged)
        if not isinstance(entry, Mapping) or not entry.get("object_sha256"):
            raise ValueError(f"pins[{MODEL_KEY}][{staged}] has no object_sha256 (cell {cell})")
        out[cell] = str(entry["object_sha256"])
    return out


def roster_view_cells(roster: Mapping[str, Any], view: str) -> Dict[str, str]:
    t = ((roster.get("targets") or {}).get(MODEL_KEY) or {})
    v = (t.get("views") or {}).get(view)
    if not isinstance(v, Mapping) or not isinstance(v.get("cells"), Mapping):
        raise ValueError(f"roster names no {MODEL_KEY}/{view} view with a cells map")
    cells = {str(c): str(s) for c, s in v["cells"].items() if isinstance(s, str)}   # a prompted `null` carries no object
    if tuple(cells) != FITTED_CELLS:
        raise ValueError(f"roster {MODEL_KEY}/{view} cells {tuple(cells)} != the lane's five fitted cells {FITTED_CELLS}")
    return cells


def engine_spec(engine: str, *, pins_by_cell: Mapping[str, str], arditi_sha256: str) -> Dict[str, Any]:
    """The standup inputs of ONE engine class (a view, the Arditi engine; a dose view builds its own): the config / manifest /
    bank view, the `--cell` selection and the `--expect-object-sha256` digests."""
    if engine == ARDITI:
        return {"config": ARDITI_CONFIG_REL, "manifest": ARDITI_MANIFEST_REL, "bank_view": ARDITI_BANK_VIEW_LOCAL,
                "cells": [ARDITI_CELL], "digests": {ARDITI_CELL: arditi_sha256}}
    missing = [c for c in FITTED_CELLS if c not in pins_by_cell]
    if missing:
        raise ValueError(f"no pin for fitted cells {missing} of {engine}")
    return {"config": VIEW_CONFIG_REL.format(view=engine), "manifest": VIEW_MANIFEST_REL.format(view=engine),
            "bank_view": BANK_VIEW_LOCAL.format(view=engine), "cells": list(FITTED_CELLS) + list(PROMPTED_CELLS),
            "digests": {c: pins_by_cell[c] for c in FITTED_CELLS}}


def standup_argv(*, leg: Mapping[str, Any], window: Mapping[str, Any], engine: str, W: str, reqs_sha256: str,
                 pins_by_cell: Mapping[str, str], arditi_sha256: str, spec: Optional[Mapping[str, Any]] = None) -> List[str]:
    """The `mt_serving_up.py` argv of ONE standup, composed through the front-end contract (§4.3): S is never passed for
    a table window (the core derives + lowers it); a non-table window passes the identity PAIR; `--cell` SELECTS the
    route set (the two prompted routes requested explicitly on a view engine; the Arditi engine a single object);
    `--expect-object-sha256` per fitted cell; `--thinking-on` iff GSM8K; `--max-logprobs 64` iff the leg serves MMLU."""
    geom = served_geometry(window)
    tree = tree_local(W)
    out = f"{ATTEMPT_ROOT_PLACEHOLDER}/{SERVE_DIR_REL}"
    es = spec or engine_spec(engine, pins_by_cell=pins_by_cell, arditi_sha256=arditi_sha256)
    config, manifest, bank_view, cells = es["config"], es["manifest"], es["bank_view"], list(es["cells"])
    digests = [f"{c}={sha}" for c, sha in es["digests"].items()]
    argv = [serving_venv_python(reqs_sha256), f"{tree}/scripts/sdfbase/mt_serving_up.py", "--repo-dir", tree,
            "--config", config, "--manifest", manifest, "--defenses-dir", bank_view,
            "--out", out, "--stop-file", f"{out}/STOP", "--mode", str(leg["mode"]), "--model-key", MODEL_KEY,
            "--eval", str(window["eval"]), *geom["argv_factors"],
            "--gpu-mem-util", GPU_MEM_UTIL, "--ready-timeout-s", str(READY_TIMEOUT_S),
            "--serving-path", SERVING_PATH, "--adapter-bank", f"{ADAPTER_BANK_ROOT}/{MODEL_KEY}",
            "--server-default-policy", SERVER_DEFAULT_POLICY, "--enable-log-requests",
            "--upstream-timeout-s", str(UPSTREAM_TIMEOUT_S)]
    if window["thinking"]:
        argv.append("--thinking-on")
    if leg.get("max_logprobs") is not None:
        argv += ["--max-logprobs", str(int(leg["max_logprobs"]))]
    for c in cells:
        argv += ["--cell", c]
    for d in digests:
        argv += ["--expect-object-sha256", d]
    return argv


def expected_route_set(engine: str) -> List[str]:
    """The route set `select_cells_lora` yields for the composed --cell list: none + the requested cells."""
    if engine == ARDITI:
        return ["none", ARDITI_CELL]
    return ["none", *FITTED_CELLS, *PROMPTED_CELLS]


def expected_cells_served(engine: str) -> Dict[str, List[str]]:
    """What the NATIVE engine's `cells_served.json` lists for the route set, in HARNESS ids: the `canonical` rows =
    the route set translated, the `controls` rows = the derived zero-adapter control `none__lora0` (every lora_native
    engine serves it; readiness that compares `traintest_id` spellings against this record fails every real standup)."""
    return cells_served_for_routes(expected_route_set(engine))


def arm_runs(engine: str) -> List[Dict[str, Any]]:
    """The arm-runs ONE engine serves: a view engine's 8 arms at the view's seed (5 fitted + none + 2 prompted); the
    Arditi engine's ONE arm at the three seed bases. 3 x 8 + 3 = 27 per leg."""
    if engine == ARDITI:
        return [{"arm": ARDITI_ARM_ID, "cell": ARDITI_CELL, "s": s, "object": ARDITI_STAGED_ID} for s in ARDITI_SEEDS]
    s = SEED_OF_VIEW[engine]
    out = [{"arm": "none", "cell": "none", "s": s, "object": "none"}]
    out += [{"arm": c, "cell": c, "s": s, "object": None} for c in FITTED_CELLS]      # object filled from the roster
    out += [{"arm": p, "cell": p, "s": s, "object": p} for p in PROMPTED_CELLS]
    return out


# --------------------------------------------------------------------------- #
# driver children (§4.4, §6, §10: one identity per CHILD PROCESS)                #
# --------------------------------------------------------------------------- #

#: the registry identity file per child (written to RS/registry/plans/<leg>/<arm>__s<s>.json; a per-TASK MAP for the
#: controls leg's multi-eval store), staged to the volume under <LANE>/<W7>/registry/plans/ and read by the child's env
PLANS_REL = "registry/plans"
RUN_IDENTITY_ENV = "SISTEER_REGISTRY_RUN_IDENTITY"
SPOOL_ENV = "SISTEER_REGISTRY_SPOOL"
SPOOL_REL = "registry/records"                     # <attempt_root>/registry/records — attempt-unique (§4.6 rev 3)
#: the substrate DECLARED to every in-container registry writer (emit.SUBSTRATE_ENV; wins over the host probes). Without it
#: `emit.substrate_from_env` probes `$SHARED` BEFORE Modal, finds the container's /workspace and stamped every container `run`
#: record `substrate=cluster_b` (source `derived`), OVERWRITING the queued record's correct `modal` in the fold.
SUBSTRATE_ENV = "SISTEER_REGISTRY_SUBSTRATE"
SUBSTRATE = "modal"
PROXY_URL_PLACEHOLDER = "{proxy_url}"
ROUTING_TABLE_PLACEHOLDER = "{routing_table}"
DRIVER_PY_PLACEHOLDER = "{driver_python}"
#: the tinycap CLIENT venv's python (inspect_ai + inspect_evals; common.sh f22_client_venv) -- the interpreter of every
#: Inspect-side step: the GSM8K children (served_capability_run.py -> inspect_served -> inspect_ai / inspect_evals.gsm8k) and the G-FIT
#: budget tool (tinycap_oos_prompt_budget.py imports inspect_ai). The DRIVER venv (`uv sync --frozen --extra cu128`) carries no inspect_ai
#: (only the `petri` dependency group does; inspect_evals is not in uv.lock at all), so a child resolved to it died at import. Whether a
#: child needs it rides the child's OWN `task` (an `inspect/...` row task) or its `budget_argv`, never a leg list.
CLIENT_PY_PLACEHOLDER = "{client_python}"
INSPECT_TASK_PREFIX = "inspect/"


def child_needs_client_python(child: Mapping[str, Any]) -> bool:
    """An Inspect-side child (`task` = `inspect/...`): its argv runs under the tinycap CLIENT venv, never the driver venv."""
    return str((child or {}).get("task") or "").startswith(INSPECT_TASK_PREFIX)


def row_needs_client_venv(row: Mapping[str, Any]) -> bool:
    """A row the container must build the CLIENT venv for: any child that runs under it, or any child with a budget prestep (the G-FIT tool)."""
    return any(child_needs_client_python(k) or bool(k.get("budget_argv")) for k in (row.get("children") or []))


def client_python_problems(plan: Mapping[str, Any]) -> List[str]:
    """The generated plan's Inspect-side argv name the CLIENT python placeholder -- an `inspect/` child or a `budget_argv` whose argv[0] is the
    DRIVER placeholder is a NAMED problem (earlier plans resolved every GSM8K child to the driver venv)."""
    problems: List[str] = []
    for r in list(plan.get("standups") or []) + list(plan.get("rungs") or []) + list(plan.get("s5_probe_rows") or []):
        for k in r.get("children") or []:
            argv = list(k.get("argv") or [])
            if child_needs_client_python(k) and argv and argv[0] != CLIENT_PY_PLACEHOLDER:
                problems.append(f"{r.get('standup_id')}: {k.get('arm')} s{k.get('s')} ({k.get('task')}) argv[0] is {argv[0]!r}, not {CLIENT_PY_PLACEHOLDER}: an "
                                "Inspect-side child runs under the tinycap CLIENT venv (the driver venv has no inspect_ai; F-I)")
            b = list(k.get("budget_argv") or [])
            if b and b[0] != CLIENT_PY_PLACEHOLDER:
                problems.append(f"{r.get('standup_id')}: {k.get('arm')} s{k.get('s')} budget_argv[0] is {b[0]!r}, not {CLIENT_PY_PLACEHOLDER}: the G-FIT "
                                "budget tool imports inspect_ai and runs under the CLIENT venv (F-I)")
    return problems
CONCURRENCY_PLACEHOLDER = "{concurrency}"


def attempt_root_for(results_root: str, *, leg: str, view: str, stage: Any, shard: Any, attempt_id: str, kind: Optional[str] = None) -> str:
    """The IMMUTABLE attempt root: base rows at ``<results_root>/<leg>/<view>/shard<NN>/attempts/<attempt_id>``; rung
    rows at ``<results_root>/<leg>/<view>/rungs/<cap>/shard<NN>/attempts/<attempt_id>``. A rung is a row whose stage is not `base`
    (``kind`` may confirm it). ONE spelling for the app (which writes it) and the watcher (which reads RESULT / COMPLETE.json back off
    it when a Modal call's output has expired)."""
    root = str(results_root).rstrip("/")
    shard_dir = f"shard{int(shard):02d}"
    is_rung = (kind == "rung") if kind else (str(stage) != STAGE_BASE)
    if is_rung:
        return f"{root}/{leg}/{view}/rungs/{stage}/{shard_dir}/attempts/{attempt_id}"
    return f"{root}/{leg}/{view}/{shard_dir}/attempts/{attempt_id}"


def producer_attempt_root_of(results_root: str):
    """The `reconcile_queue(attempt_root_of=...)` injection: a DECLARED producer's accepted attempt root in THIS app's
    spelling -- volume-relative `seedrep/<W7>/exp/seedrep_battery_modal/<leg>/<view>/shard<NN>/attempts/<aid>` for a base producer, the rungs/
    path for a rung producer (`attempt_root_for`, the ONE spelling the app writes, the watcher reads and the audit parses). The peer's default
    `attempt_root_rel(..., p.shard, ...)` spelled the shard bare (`/0/`) until their post-SHIP c7b846299 adopted `shard<NN>` (pinned EQUAL to
    this by test); the injection stays the app's own spelling regardless of the peer tree's default."""
    root = str(results_root)

    def root_of(p: Any, aid: str) -> str:
        key = p.key
        stage = str(key.stage)
        kind = "base" if stage == STAGE_BASE else "rung"
        full = attempt_root_for(root, leg=key.leg, view=key.view, stage=stage, shard=int(p.shard), attempt_id=aid, kind=kind)
        return full[len(VOL_PATH) + 1:] if full.startswith(VOL_PATH + "/") else full.lstrip("/")
    return root_of


#: the producer of the identity files (the ONE authority on where they live); imported BY PATH from this tree, once
REGISTRY_PLANS_REL = "scripts/sdfbase/seedrep/registry_plans.py"
_REGISTRY_PLANS = None


def registry_plans_module():
    """``registry_plans`` (:data:`REGISTRY_PLANS_REL`) from THIS tree -- the module already imported from that file when a
    caller holds one, else loaded by path (registered in ``sys.modules`` BEFORE exec: its frozen dataclasses resolve their annotations
    through ``sys.modules[cls.__module__]`` at class creation, as ``seedrep_s0_gates.load_registry_plans`` notes). The plan CALLS the
    producer's path functions and arm spelling (launch F-C) instead of re-spelling them, so the two cannot drift again."""
    global _REGISTRY_PLANS
    if _REGISTRY_PLANS is not None:
        return _REGISTRY_PLANS
    here = Path(__file__).resolve()
    # PLAN GENERATION ONLY: the producer is read from the REPO layout (<repo>/scripts/modal/<this file>); the Modal
    # images install this module at /root/seedrep_launch_plan.py (seedrep_served.py's add_local_file), where no repo and no producer
    # exist -- the container never calls this: a spawn consumes the child's GENERATED `registry_identity_rel` (child_registry_env).
    repo = here.parents[2] if len(here.parents) > 2 and here.parents[1].name == "scripts" and here.parent.name == "modal" else None
    path = (repo / REGISTRY_PLANS_REL) if repo is not None else None
    for name in ("registry_plans", "seedrep_registry_plans_s0", "seedrep_registry_plans_lp"):
        mod = sys.modules.get(name)
        if mod is not None and path is not None and Path(getattr(mod, "__file__", "") or "").resolve() == path:
            _REGISTRY_PLANS = mod
            return mod
    if path is None or not path.is_file():
        raise FileNotFoundError(f"{REGISTRY_PLANS_REL} is not beside this module ({here}: not a <repo>/scripts/modal layout, or the producer is "
                                "absent); identity paths are derived at plan GENERATION on the tree -- a container reads the plan's "
                                "registry_identity_rel and never the producer")
    spec = importlib.util.spec_from_file_location("seedrep_registry_plans_lp", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    if mod.ARDITI_STAGED_ID != ARDITI_STAGED_ID:
        raise ValueError(f"{REGISTRY_PLANS_REL} spells the Arditi staged id {mod.ARDITI_STAGED_ID!r}, this plan {ARDITI_STAGED_ID!r}")
    _REGISTRY_PLANS = mod
    return mod


def identity_plan_rel(eval_leg: str, arm: str, s: int) -> str:
    """``registry/plans/<the producer's relpath>`` — the planned-run JSON (or per-task map) of ONE (leg, arm, seed), spelled by
    the producer itself (:func:`registry_plans_module`): ``plan_relpath(leg, arm, s)`` = ``<leg>/<arm>__s<s>.json`` for a single-eval
    child, ``store_relpath(store, arm, s)`` = ``_stores/<store>/<arm>__s<s>.json`` for a child of a multi-eval STORE (``STORE_LEGS``: the
    controls child, whose `eval_leg` is the store name), with the producer's ARM spelling (the Arditi children are labelled by the staged
    id ``ar_c0p3__arditi_allpos`` in this plan and filed under ``ARDITI_ARM`` = ``arditi_allpos`` by the producer). Spelling it here instead
    (``<eval_leg>/<arm>__s<s>.json`` with the plan's labels) made 546 children name files the producer never writes."""
    RP = registry_plans_module()
    producer_arm = RP.ARDITI_ARM if arm == RP.ARDITI_STAGED_ID else arm
    if eval_leg in RP.STORE_LEGS:
        rel = RP.store_relpath(eval_leg, producer_arm, int(s))
    else:
        rel = RP.plan_relpath(eval_leg, producer_arm, int(s))
    return f"{PLANS_REL}/{rel}"


#: the request-evidence client (`sisteer.eval.request_evidence_client.ENV_DIR`): the directory a driver child writes its
#: `request_evidence.jsonl` / `response_evidence.jsonl` into (the rung consumer must be spawned with it,
#: under ITS attempt root, or the rung's sends leave no evidence)
REQUEST_EVIDENCE_ENV = "SISTEER_REQUEST_EVIDENCE_DIR"
#: its two companions (`request_evidence_client.ENV_EXTRA` / `ENV_SEED_BASE`, dd43a18f): a JSON object stamped on every record
#: (`attempt_id` must equal the child's `--attempt-id` when both are given; `leg`, `arm`, `s`, `standup_id` ride along) and the seed base `s`
#: (REQUIRED: the audit refuses an Arditi root whose records carry no `extra.s`). None of the three was exported to any child at 0e51107b.
#: Before the fix `resolve_spawn` did NOT honour the first clause -- every child's extra named the CONTAINER's attempt id, so a GSM8K child
#: (the one driver child whose argv carries `--attempt-id`, the per-store `a<k>`) ran with two ids and the lane client refused every record
#: (`adopt_attempt_id`: one run has one attempt id). The extra now names what the argv names (:func:`evidence_attempt_id`).
REQUEST_EVIDENCE_EXTRA_ENV = "SISTEER_REQUEST_EVIDENCE_EXTRA"
REQUEST_EVIDENCE_SEED_BASE_ENV = "SISTEER_REQUEST_EVIDENCE_SEED_BASE"
REQUEST_EVIDENCE_ENVS: Tuple[str, ...] = (REQUEST_EVIDENCE_ENV, REQUEST_EVIDENCE_EXTRA_ENV, REQUEST_EVIDENCE_SEED_BASE_ENV)
#: a RUNG consumer's evidence directory is the LITERAL attempt root -- `request_evidence.jsonl` +
#: `response_evidence.jsonl` land beside the `usage_ledger.jsonl` the proxy writes and the engine log, the three files `load_evidence`
#: reads from ONE root (579c53a2 nested them at `<root>/exp/request_evidence`; the reader tolerated it by recursive discovery, the contract says the root)
REQUEST_EVIDENCE_FILES: Tuple[str, ...] = ("request_evidence.jsonl", "response_evidence.jsonl")
#: what a rung consumer's extra must NEVER name: a rung container serves every arm of its engine and the Arditi rung multiplexes s1-s3,
#: so `RungEvidence` stamps `arm` and `s` PER QUEUE RECORD and refuses a pinned value that disagrees -- at 579c53a2 the extra pinned
#: `arm: consumer` and every production draw was refused (`the evidence environment pins arm 'consumer' but this record is 'none'`)
RUNG_EXTRA_FORBIDDEN: Tuple[str, ...] = ("arm", "s")
#: the flag under which each driver writes its store -- the location contract: `SISTEER_REQUEST_EVIDENCE_DIR` == that directory
#: (served_capability_run writes `<--out-root>/request_evidence.jsonl`; the others their private / out root)
STORE_ROOT_FLAG_OF: Dict[str, str] = {"controls": "--out-root", "gsm8k": "--out-root", "bon": "--private-root", "misalign_leaking": "--out",
                                      "misalign_murder": "--out", "pair": "--private-root", "crescendo": "--private-root", "actorattack": "--private-root"}


def child_store_root(child: Mapping[str, Any], resolved_argv: Sequence[str]) -> str:
    """The directory a TARGET child writes its store into (its evidence dir, the location contract), read off its RESOLVED argv:
    `--out-root` (controls / GSM8K), `--out` (misalign), `--private-root` (static / BoN / multi-turn). Refuses a child with none."""
    leg = str(child["eval_leg"])
    flag = STORE_ROOT_FLAG_OF.get(leg, "--private-root" if leg in STATIC_LEGS else None)
    val = _flag_value(list(resolved_argv), flag) if flag else None
    if not val:
        raise ValueError(f"child {child.get('arm')} s{child.get('s')} ({leg}): no store root flag ({flag}) in its argv -- SISTEER_REQUEST_EVIDENCE_DIR cannot be set")
    return str(val)


def request_evidence_env(*, evidence_dir: str, attempt_id: str, standup_id: str, eval_leg: str, arm: str, s: Optional[int]) -> Dict[str, str]:
    """The three request-evidence variables for ONE child process: the directory (== the child's store root), the extra object
    (`attempt_id` == the id the child's driver writer runs as -- :func:`evidence_attempt_id`: its argv's `--attempt-id` when it carries one, else
    the container's attempt id; `leg`, `arm`, `s`, `standup_id`), the seed base. ``s`` None (a rung consumer multiplexing several bases) leaves
    the seed-base variable out and out of `extra` -- the per-draw base follows the consumer's contract."""
    extra: Dict[str, Any] = {"leg": eval_leg, "arm": arm, "attempt_id": attempt_id, "standup_id": standup_id}
    env = {REQUEST_EVIDENCE_ENV: str(evidence_dir)}
    if s is not None:
        extra["s"] = int(s)
        env[REQUEST_EVIDENCE_SEED_BASE_ENV] = str(int(s))
    env[REQUEST_EVIDENCE_EXTRA_ENV] = json.dumps(extra, sort_keys=True)
    return env


def evidence_attempt_id(resolved_argv: Sequence[str], container_attempt_id: str) -> str:
    """The ONE attempt id a driver child's request evidence runs as: the `--attempt-id` on ITS resolved argv when it carries one,
    else the container's. Only the GSM8K children carry the flag -- the per-store `a<k>` that `gsm8k_child_up` minted into `ladder_state.json`
    and started the child's usage ledger with (`{gsm8k_attempt_id}`): the driver's own writer stamps it on every record and
    `contracts.check_attempt_id_arg` admits `a<k>` ALONE (the ledger's parser too), so the container's `<task>-<uuid>` can never ride the argv;
    the lane client (`request_evidence_client.adopt_attempt_id`) refuses a driver writer whose id differs from its extra's, so the extra follows
    the argv, never the other way round. Two gsm8k rows died on exactly that refusal (`one run has one attempt id; refusing to record under
    two`) because every child's extra named the container's id. Static / controls / misalign / BoN / multi-turn children carry no `--attempt-id`
    and keep the container's id, byte-identical to before; the rung consumer (`--attempt-id {attempt_id}` == the container's) never reaches this
    (its env is `rung_child_env`)."""
    own = _flag_value(list(resolved_argv), "--attempt-id")
    return str(own) if own else str(container_attempt_id)


def rung_child_env(*, attempt_root: str, attempt_id: str = "", standup_id: str = "", leg: str = "", seed_bases: Sequence[int] = ()) -> Dict[str, str]:
    """The env a RUNG CONSUMER child is spawned with: the evidence
    directory = the LITERAL attempt root; the extra object naming this attempt / leg / standup and NEVER `arm` or `s`
    (:data:`RUNG_EXTRA_FORBIDDEN` -- the consumer stamps both per queue record); the seed base pinned ONLY on a single-seed view row (every
    record agrees, so the consumer's real check is a cross-check) and UNSET on the Arditi multiplex; the attempt-unique registry spool."""
    extra: Dict[str, Any] = {"leg": leg, "attempt_id": attempt_id, "standup_id": standup_id}
    env = {REQUEST_EVIDENCE_ENV: str(attempt_root), REQUEST_EVIDENCE_EXTRA_ENV: json.dumps(extra, sort_keys=True), SPOOL_ENV: f"{attempt_root}/{SPOOL_REL}",
           SUBSTRATE_ENV: SUBSTRATE}
    if len(seed_bases) == 1:
        env[REQUEST_EVIDENCE_SEED_BASE_ENV] = str(int(seed_bases[0]))
    return env


def rung_child_env_problems(env: Mapping[str, str], *, attempt_root: str, attempt_id: str = "", seed_bases: Optional[Sequence[int]] = None) -> List[str]:
    """What a rung consumer's env MUST carry before the server is stood up (refused by name, before any GPU): the evidence dir AT this attempt
    root and the spool under it; the extra object naming THIS attempt and naming neither `arm` nor `s` (`RungEvidence.client_for`
    refuses a pinned arm / seed that disagrees with a record -- the 579c53a2 production failure); a pinned seed base only on a row with
    exactly ONE seed base and equal to it (``seed_bases`` None = not checked)."""
    problems = []
    got = env.get(REQUEST_EVIDENCE_ENV)
    if not got:
        problems.append(f"rung consumer env lacks {REQUEST_EVIDENCE_ENV} (WS1-D′'s request-evidence client writes nothing without it)")
    elif str(got).rstrip("/") != str(attempt_root).rstrip("/"):
        problems.append(f"rung consumer env {REQUEST_EVIDENCE_ENV}={got!r} is not this attempt root ({attempt_root}): WS2-E §R2.2 reads the evidence at the LITERAL root")
    spool = env.get(SPOOL_ENV)
    if not spool:
        problems.append(f"rung consumer env lacks {SPOOL_ENV} (the attempt-unique registry spool)")
    elif spool != f"{attempt_root}/{SPOOL_REL}":
        problems.append(f"rung consumer env {SPOOL_ENV}={spool!r} is not under this attempt root ({attempt_root}/{SPOOL_REL})")
    raw = env.get(REQUEST_EVIDENCE_EXTRA_ENV)
    if not raw:
        problems.append(f"rung consumer env lacks {REQUEST_EVIDENCE_EXTRA_ENV} (the attempt binding of every record)")
    else:
        try:
            extra = json.loads(raw)
        except ValueError:
            extra = None
        if not isinstance(extra, dict) or (attempt_id and extra.get("attempt_id") != attempt_id):
            problems.append(f"rung consumer env {REQUEST_EVIDENCE_EXTRA_ENV} does not bind this attempt {attempt_id!r}: {raw[:120]!r}")
        else:
            for k in RUNG_EXTRA_FORBIDDEN:
                if k in extra:
                    problems.append(f"rung consumer env {REQUEST_EVIDENCE_EXTRA_ENV} pins {k}={extra[k]!r}: WS2-E stamps {k} per queue record and refuses "
                                    f"a pinned {k} that disagrees (WS2-E r3 finding 2; §R2.2) -- leave it out")
    pinned = env.get(REQUEST_EVIDENCE_SEED_BASE_ENV)
    if pinned is not None and seed_bases is not None:
        bases = [int(s) for s in seed_bases]
        if not str(pinned).isdigit():
            problems.append(f"rung consumer env {REQUEST_EVIDENCE_SEED_BASE_ENV}={pinned!r} is not a non-negative integer")
        elif len(bases) != 1:
            problems.append(f"rung consumer env pins {REQUEST_EVIDENCE_SEED_BASE_ENV}={pinned!r} on a row that multiplexes seed bases {bases}: "
                            "leave it unset (WS2-E §R2.2; the Arditi consumer stamps extra.s per record)")
        elif int(pinned) != bases[0]:
            problems.append(f"rung consumer env pins {REQUEST_EVIDENCE_SEED_BASE_ENV}={pinned!r} but this row's seed base is {bases[0]}")
    return problems


def identity_rel_under_plans(registry_identity_rel: Any) -> str:
    """A child's GENERATED ``registry_identity_rel`` (``registry/plans/<producer relpath>``) -> the relpath under the plans root
    (``<leg>/<arm>__s<s>.json`` or ``_stores/<store>/<arm>__s<s>.json``). Refuses anything that is not a relative ``.json`` path
    strictly beneath :data:`PLANS_REL` (absolute, ``..``, empty segments, another root): the container trusts the plan's spelling
    and nothing else, so a malformed field is a refusal by name, never a file outside the plans root."""
    rel = str(registry_identity_rel or "")
    prefix = PLANS_REL + "/"
    if not rel.startswith(prefix) or rel.startswith("/") or not rel.endswith(".json"):
        raise ValueError(f"registry_identity_rel {rel!r} is not a relative .json path beneath {PLANS_REL}/")
    under = rel[len(prefix):]
    parts = under.split("/")
    if not under or any(p in ("", ".", "..") for p in parts) or "\\" in under:
        raise ValueError(f"registry_identity_rel {rel!r} leaves the plans root or carries an empty segment")
    return under


def child_registry_env(*, plans_root: str, registry_identity_rel: Any, attempt_root: str) -> Dict[str, str]:
    """The registry wiring of ONE child process:
    ``SISTEER_REGISTRY_RUN_IDENTITY`` = the child's plan file -- the plan's GENERATED ``registry_identity_rel`` (spelled by the
    producer at plan generation, :func:`identity_plan_rel`) resolved under the plans root, never re-derived here (the container
    has no repo and no producer module; re-deriving walked `parents[2]` off /root/seedrep_launch_plan.py);
    ``SISTEER_REGISTRY_SPOOL`` = the ATTEMPT's private spool — never the shared ``/vol/registry/records`` (RecordWriter's
    sequence numbers are snapshot-local)."""
    rel = identity_rel_under_plans(registry_identity_rel)
    return {RUN_IDENTITY_ENV: f"{plans_root}/{rel}", SPOOL_ENV: f"{attempt_root}/{SPOOL_REL}", SUBSTRATE_ENV: SUBSTRATE}


def static_nshards(S: int, n_arm_runs: int) -> int:
    """The goal-shard count per arm of a static engine: `ceil(S / n_arm_runs)` so that arms x shards >= S ONE-REQUEST-EACH
    drivers keep the engine's S slots full (target_sampled.sbatch's static_base_t0 shape: one `jailbreak_run.py generate`
    per (family, cell, shard), `MAX_DRIVERS = S` in flight — never `DRIVER_CAP 96`)."""
    return max(1, -(-int(S) // max(1, int(n_arm_runs))))


def _static_children(leg: str, engine: str, W: str, cap: int, suffix: str, S: int) -> List[Dict[str, Any]]:
    """jailbreak_run.py generate per (family, arm, goal shard) exactly as target_sampled.sbatch's static_base_t0 phase
    composes it, from the engine's SAMPLED config (its sampled_decode block carries seed_base = s, trial 0 of 1;
    the Arditi engine reads the lane's per-seed rendered configs). The container keeps S of these drivers
    live (`max_live = S`); a driver serves its shard's goals one request at a time."""
    tree = tree_local(W)
    runs = arm_runs(engine)
    nsh = static_nshards(S, len(runs))
    kids = []
    for run in runs:
        if engine == ARDITI:
            cfg = ARDITI_CLIENT_CONFIG_REL.format(s=run["s"], suffix=suffix)
        else:
            cfg = VIEW_SAMPLED_CONFIG_REL.format(view=engine, suffix=suffix)
        for shard in range(nsh):
            argv = [DRIVER_PY_PLACEHOLDER, f"{tree}/scripts/jailbreak_run.py", "generate", "--config", cfg,
                    "--attack", leg, "--defense", run["cell"], "--confirmatory", "--trial", "0",
                    "--shard", str(shard), "--nshards", str(nsh),
                    "--private-root", f"{ATTEMPT_ROOT_PLACEHOLDER}/exp/transcripts_results",
                    "--target-engine", "served", "--target-proxy-url", PROXY_URL_PLACEHOLDER,
                    "--target-routing-table", ROUTING_TABLE_PLACEHOLDER]
            kids.append({"eval_leg": leg, "arm": run["arm"], "cell": run["cell"], "s": run["s"], "task": f"static/{leg}",
                         "concurrency_rule": "one_request_each_S_drivers", "argv": argv, "cap": cap,
                         "goal_shard": shard, "goal_nshards": nsh})
    return kids


def _bon_children(engine: str, W: str) -> List[Dict[str, Any]]:
    tree = tree_local(W)
    kids = []
    for run in arm_runs(engine):
        # the WAVE config: the per-seed Arditi wave config / the view's unsampled config -- never a `_sampled_t0` static run;
        # the BoN decode rides the argv (`--bon-decode-preset vendor --decode-seed-policy replicate_v1 --decode-seed-base s`), not the config
        cfg = (ARDITI_WAVE_CONFIG_REL.format(s=run["s"]) if engine == ARDITI else VIEW_CONFIG_REL.format(view=engine))
        argv = [DRIVER_PY_PLACEHOLDER, f"{tree}/scripts/sdfbase/full22/bon_served_run.py", "--config", cfg,
                "--defense", run["cell"], "--confirmatory", "--private-root", f"{ATTEMPT_ROOT_PLACEHOLDER}/exp/transcripts_results",
                "--proxy-url", PROXY_URL_PLACEHOLDER, "--routing-table", ROUTING_TABLE_PLACEHOLDER,
                "--max-concurrency", CONCURRENCY_PLACEHOLDER, "--require-engine", "lora_native", "--trial", "0",
                "--bon-decode-preset", "vendor", "--model-key", MODEL_KEY,
                "--decode-seed-policy", "replicate_v1", "--decode-seed-base", str(run["s"])]
        kids.append({"eval_leg": "bon", "arm": run["arm"], "cell": run["cell"], "s": run["s"], "task": "strongreject_bon",
                     "concurrency_rule": "S_over_ncells_floor_4", "argv": argv, "cap": 4096, "config": cfg})
    return kids


CONTROLS_ALL: Tuple[str, ...] = ("xstest", "or_bench_hard_1k", "mmlu")
CONTROLS_TASKS: Dict[str, str] = {"xstest": "judged/xstest_v2", "or_bench_hard_1k": "judged/or_bench_hard_1k", "mmlu": "s2parity/mmlu_letter"}
#: the controls set splits by LEG CLASS. The sampled judged controls ride the cap ladder 512 -> 2,048 -> 8,192 -> 16,384 (the
#: driver's `SAMPLED_CONTROLS`); MMLU is a teacher-forced one-token letter readout (max_tokens 1, greedy, `top_logprobs 64`) with NO cap
#: ladder (the driver's `SAMPLED_UNAFFECTED_CONTROLS`: `served_capability_run.py` refuses `--controls mmlu` beside `--controls-cap` by name --
#: earlier S3 controls / S2 controls-arditi children all died on it). It is measured ONCE per (arm, s), on the BASE row at the fixed base cap
#: (the at-cap table: `applies: False` / `not_applicable`; the coverage check expects no rung stage of it). A rung row's controls children
#: carry the ladder controls alone, and a rung engine serves no MMLU (no `--max-logprobs`, no `mmlu` eval leg).
CONTROLS_LADDER: Tuple[str, ...] = ("xstest", "or_bench_hard_1k")
CONTROLS_FIXED_CAP: Tuple[str, ...] = ("mmlu",)
#: the driver's opt-in per-draw event stream (`served_capability_run.py`: `cev.CONTROLS_EVENTS_FLAG`, dest `controls_events`,
#: default OFF). With it the store carries `<cell_dir>/controls_events.jsonl` -- the ONLY file the app's `ControlsStreamAdapter.streams()`
#: globs (`exp/sampled/controls_cap*/**/controls_events.jsonl`) to feed the controls ladder queue (`row.ladder.cap_next`). Earlier controls
#: children never passed it: every controls row (S2 controls-arditi, S3 controls, S5 controls FULL) drew everything, its tailer registered
#: NO stream, and `require_ladder_inventory(require_rows=True)` refused the row `LADDER_INVENTORY_INCOMPLETE` over zero streams. The driver
#: accepts the flag only on a SAMPLED judged-controls run (a `CONTROLS_LADDER` control among `--controls`; it refuses the flag by name beside
#: a fixed-cap-only set), so it rides exactly the children whose set rides the ladder (`controls_ride_the_ladder`). The plan module cannot
#: import `sisteer`: equality with `sisteer.eval.controls_events.CONTROLS_EVENTS_FLAG` is pinned by `tests/test_seedrep_fa2_controls_events.py`.
CONTROLS_EVENTS_FLAG = "--controls-events"
#: the controls leg's BASE cap (`legs["controls"]["cap"]`; the driver's `CONTROLS_CAP_LADDER[0]`): `--controls-cap N` is the driver's RUNG flag
#: (N must be a rung ABOVE the base, paired with `--escalate-from-root`), so the base row's children never carry it -- their cap is the
#: config's own `controls_max_new_tokens` (512 on every wave config)
CONTROLS_BASE_CAP = 512
#: each control's registry EVAL-id spelling (`EVAL_LEGS_OF["controls"]`), for a row's `eval_legs`
CONTROLS_EVAL_LEG_OF: Dict[str, str] = {"xstest": "xstest", "or_bench_hard_1k": "or_bench_hard", "mmlu": "mmlu"}
assert set(CONTROLS_LADDER) | set(CONTROLS_FIXED_CAP) == set(CONTROLS_ALL) and not set(CONTROLS_LADDER) & set(CONTROLS_FIXED_CAP)
assert tuple(CONTROLS_EVAL_LEG_OF[c] for c in CONTROLS_ALL) == EVAL_LEGS_OF["controls"]
BANK_VIEW_PLACEHOLDER = "{bank_view}"


def controls_for_cap(controls: Sequence[str], *, cap: int, base_cap: int = CONTROLS_BASE_CAP) -> Tuple[str, ...]:
    """The controls a controls row at ``cap`` carries: every requested control on the BASE row (``cap == base_cap``); on a rung (a cap above
    the base) only the ladder-capable ones -- a fixed-cap control (MMLU) is never re-drawn at a rung (F-T). Order preserved; a rung left with
    no ladder control refuses (nothing to escalate)."""
    if int(cap) == int(base_cap):
        return tuple(controls)
    out = tuple(c for c in controls if c in CONTROLS_LADDER)
    if not out:
        raise ValueError(f"controls rung at cap {cap}: none of {list(controls)} rides the cap ladder ({list(CONTROLS_LADDER)})")
    return out


def controls_ride_the_ladder(controls: Sequence[str]) -> bool:
    """Whether a controls set carries at least one ladder (SAMPLED judged) control -- the driver's condition for the per-draw event stream
    (F-AF): the base row's three, a rung's two and a dose row's `xstest` do; a fixed-cap-only set (MMLU alone) does not."""
    return any(c in CONTROLS_LADDER for c in controls)


def _controls_children(engine: str, W: str, cap: int, *, runs: Optional[Sequence[Mapping[str, Any]]] = None, cfg: Optional[str] = None,
                       controls: Sequence[str] = CONTROLS_ALL, base_cap: int = CONTROLS_BASE_CAP) -> List[Dict[str, Any]]:
    """served_capability_run.py generate — ONE CHILD PER CELL (§10 c: one identity per child; the sbatch's own shape). The BASE row
    (``cap == base_cap``): controls xstest + or_bench_hard_1k, then MMLU (letter prefill, top_logprobs 64) on the same standup, NO
    `--controls-cap` (the driver's rung flag). A RUNG row (``cap`` above the base): the ladder controls alone with `--controls-cap <cap>` --
    the driver-form of the rung whose draws the app's rung CONSUMER (`rung_consumer_argv`) actually makes; the rung's children are what the
    tailer, the audit and the coverage expect of it (F-T: `--controls mmlu` on a rung is refused by the driver by name). Every child whose
    set rides the ladder carries :data:`CONTROLS_EVENTS_FLAG` (F-AF): the driver's per-draw stream is what the app's controls tailer
    reads; without it the base row's queue starves and the row is refused after drawing everything."""
    tree = tree_local(W)
    rung = int(cap) != int(base_cap)
    controls_ = controls_for_cap(controls, cap=cap, base_cap=base_cap)
    kids = []
    for run in (runs if runs is not None else arm_runs(engine)):
        cfg_ = cfg or (ARDITI_CLIENT_CONFIG_REL.format(s=run["s"], suffix="") if engine == ARDITI
                       else VIEW_SAMPLED_CONFIG_REL.format(view=engine, suffix=""))
        argv = [DRIVER_PY_PLACEHOLDER, f"{tree}/scripts/served_capability_run.py", "generate",
                "--proxy-url", PROXY_URL_PLACEHOLDER, "--routing-table", ROUTING_TABLE_PLACEHOLDER,
                "--cell", run["cell"], "--config", cfg_, "--defenses-dir", BANK_VIEW_PLACEHOLDER,
                "--out-root", f"{ATTEMPT_ROOT_PLACEHOLDER}/{CONTROLS_OUT_REL.format(cap=cap, s=run['s'])}",
                "--controls", ",".join(controls_), "--concurrency", CONCURRENCY_PLACEHOLDER]
        if any(c in CONTROLS_FIXED_CAP for c in controls_):
            argv += ["--top-logprobs", str(MMLU_TOP_LOGPROBS)]          # the MMLU letter-logprob read width; nothing else reads it
        argv += ["--trial", "0"]
        if rung:
            argv += ["--controls-cap", str(cap)]
        if controls_ride_the_ladder(controls_):
            argv.append(CONTROLS_EVENTS_FLAG)        # F-AF: the per-draw stream the app's controls tailer reads (default off writes no file)
        kids.append({"eval_leg": "controls", "arm": run["arm"], "cell": run["cell"], "s": run["s"],
                     "task": {c: CONTROLS_TASKS[c] for c in controls_}, "controls": list(controls_),
                     "concurrency_rule": "S_over_ncells_floor_8", "argv": argv, "cap": cap, "out_rel": CONTROLS_OUT_REL.format(cap=cap, s=run["s"]),
                     "config": cfg_})
    return kids


def gsm8k_budget_argv(*, W: str, cap: int, geometry_eval: str, config_rel: str, cell: str, budget_rel: str) -> List[str]:
    """The CPU prompt-budget artifact for ONE (cap, cell) -- `g8k_budget_artifact` verbatim (`tinycap_oos_prompt_budget.py --task
    gsm8k_paired200 --arm on --cap <cap> --geometry-eval <entry> --config <cfg> --model-key qwen35_4b [--system-prompt-cell <prompted cell>]
    --tree-sha W --out <path>`): with `--geometry-eval` the tool writes `<out>.fit.json` (fits / cap / geometry_eval / arm / tree_sha) and exits 2
    on a non-fit -- G-FIT at the pin. The app runs it in-container BEFORE the GSM8K children and refuses the row on a non-fit."""
    tree = tree_local(W)
    argv = [CLIENT_PY_PLACEHOLDER, f"{tree}/scripts/tinycap_oos_prompt_budget.py", "--task", GSM8K_INSPECT_TASK, "--arm", "on", "--cap", str(int(cap)),
            "--geometry-eval", geometry_eval, "--config", config_rel, "--model-key", MODEL_KEY]
    if cell in PROMPTED_CELLS:
        argv += ["--system-prompt-cell", cell]
    argv += ["--tree-sha", W, "--repo-root", tree, "--out", f"{ATTEMPT_ROOT_PLACEHOLDER}/{budget_rel}"]
    return argv


def gsm8k_budget_problems(sidecar: Any, *, cap: int, geometry_eval: str, W: str) -> List[str]:
    """The `g8k_budget_artifact` sidecar rule: fits, at THIS cap, against THIS entry, thinking ON, made at THIS tree -- else refuse by name."""
    if not isinstance(sidecar, Mapping):
        return ["no .fit.json sidecar (the budget tool ran without --geometry-eval, or not at all)"]
    out = []
    if sidecar.get("fits") is not True:
        out.append(f"fits={sidecar.get('fits')!r}: {sidecar.get('why_not')}")
    if int(sidecar.get("cap") or 0) != int(cap):
        out.append(f"cap {sidecar.get('cap')!r} != {cap}")
    if sidecar.get("geometry_eval") != geometry_eval:
        out.append(f"geometry_eval {sidecar.get('geometry_eval')!r} != {geometry_eval!r}")
    if sidecar.get("arm") != "on":
        out.append(f"arm {sidecar.get('arm')!r} != 'on'")
    if sidecar.get("tree_sha") != W:
        out.append(f"tree_sha {sidecar.get('tree_sha')!r} != the wave sha {W} (the pin: re-run G-FIT at this tree)")
    return out


def _gsm8k_children(engine: str, W: str, cap: int, *, pins_by_cell: Optional[Mapping[str, str]] = None, arditi_sha256: str = "",
                    base_cap: int = 8192, geometry_eval: str = "") -> List[Dict[str, Any]]:
    """One call per matrix row of the engine (a matrix row = one arm at the view's seed), all on the ONE thinking-ON standup, in the
    `g8k_driver_argv` line FLAG FOR FLAG (`scripts/sdfbase/gsm8k/g8k_common.sh`; verified against `G8K_DRY_RUN=1 gsm8k_thinking.sbatch`'s
    DRIVER line by test): the per-cell overlay config, `--ledger-url` (the child's usage ledger, started with the attempt id), thinking ON,
    `thinking_card`, `--samples 1`, the cap as STATEMENT (`--max-tokens-by-task`) AND SETTER (`--on-cap-by-task`), `--inspect-seed-policy
    replicate --inspect-seed-base s`, identity VERSION 2, the CPU prompt-budget artifact, the engine log, `--attempt-id a<k>`, and for a FITTED
    row (the 15 lane cells and the Arditi arm) `--allow-out-of-support-thinking` under its tuple + `--expect-object-sha256 <pin>`. A RUNG child
    (``cap != base_cap``) adds `--item-subset <the queue-derived rung_subset.json>`. The per-child runtime values (`{ledger_url}`,
    `{child_routing_table}`, `{gsm8k_attempt_id}`) are placeholders `resolve_spawn` fills from the ledger the container started for THIS child;
    a child spawned without them refuses. `--bank-view s<s>` follows `--attempt-id` as on the `g8k_driver_argv` line
    (:data:`GSM8K_BANK_VIEW_RULE`)."""
    tree = tree_local(W)
    pins_by_cell = dict(pins_by_cell or {})
    kids = []
    for run in arm_runs(engine):
        cell, s = run["cell"], int(run["s"])
        cfg = (ARDITI_CONFIG_REL if engine == ARDITI else GSM8K_OVERLAY_CONFIG_REL.format(model=MODEL_KEY, view=engine, slug=gsm8k_cell_slug(cell)))
        out_rel = GSM8K_OUT_REL.format(cap=cap, s=s, cell=cell)
        budget_rel = GSM8K_BUDGET_REL.format(cap=cap, cell=cell)
        ledger_serve_rel = GSM8K_LEDGER_SERVE_REL.format(cell=cell, s=s)
        pin = arditi_sha256 if (engine == ARDITI and cell == ARDITI_CELL) else pins_by_cell.get(cell)
        fitted = bool(pin)
        tup = gsm8k_oos_tuple(cell, cap, engine=engine)
        argv = [CLIENT_PY_PLACEHOLDER, f"{tree}/scripts/served_capability_run.py", "generate",                    # the CLIENT venv (inspect_ai)
                "--proxy-url", PROXY_URL_PLACEHOLDER, "--routing-table", CHILD_ROUTING_TABLE_PLACEHOLDER, "--ledger-url", LEDGER_URL_PLACEHOLDER,
                "--cell", cell, "--config", cfg, "--defenses-dir", BANK_VIEW_PLACEHOLDER,
                "--out-root", f"{ATTEMPT_ROOT_PLACEHOLDER}/{out_rel}",
                "--controls", "gsm8k", "--thinking", "on", "--sampling", "thinking_card", "--samples", str(GSM8K_SAMPLES),
                "--max-tokens-by-task", f"{GSM8K_INSPECT_TASK}={int(cap)}", "--on-cap-by-task", f"{GSM8K_INSPECT_TASK}={int(cap)}",
                "--inspect-seed-policy", "replicate", "--inspect-seed-base", str(s),
                "--tinycap-identity-version", str(GSM8K_IDENTITY_VERSION), "--oos-prompt-budget", f"{ATTEMPT_ROOT_PLACEHOLDER}/{budget_rel}",
                "--concurrency", CONCURRENCY_PLACEHOLDER, "--vllm-log", f"{ATTEMPT_ROOT_PLACEHOLDER}/{GSM8K_VLLM_LOG_REL}",
                "--attempt-id", GSM8K_ATTEMPT_ID_PLACEHOLDER, "--bank-view", gsm8k_bank_view(engine, s)]
        if fitted:
            if tup is None:
                raise ValueError(f"GSM8K child {cell}: fitted (pinned) but no authorised out-of-support tuple -- WS0's contract has none for this cell")
            argv += ["--allow-out-of-support-thinking", "--expect-object-sha256", pin]
        kid = {"eval_leg": "gsm8k", "arm": run["arm"], "cell": cell, "s": s, "task": GSM8K_ROW_TASK, "concurrency_rule": "S", "argv": argv, "cap": int(cap),
               "out_rel": out_rel, "config": cfg, "budget_rel": budget_rel, "identity_version": GSM8K_IDENTITY_VERSION, "fitted": fitted,
               "oos_tuple": tup, "object_sha256": pin,
               "ledger": {"serve_dir_rel": ledger_serve_rel, "routing_table_rel": f"{ledger_serve_rel}/routing_table.json",
                          "endpoint_json_rel": f"{ledger_serve_rel}/{GSM8K_LEDGER_ENDPOINT_FILENAME}", "ledger_path_rel": f"{out_rel}/{GSM8K_LEDGER_FILENAME}",
                          "ladder_state_rel": f"{out_rel}/{GSM8K_LADDER_STATE_FILENAME}", "attempt_id": GSM8K_FIRST_ATTEMPT_ID,
                          "rule": "WS0 g8k_ledger_up: one usage ledger per store, started with the attempt id, its record beside the child's routing table"},
               "bank_view": gsm8k_bank_view(engine, s), "bank_view_rule": GSM8K_BANK_VIEW_RULE,
               "budget_argv": gsm8k_budget_argv(W=W, cap=cap, geometry_eval=geometry_eval, config_rel=cfg, cell=cell, budget_rel=budget_rel)}
        if int(cap) != int(base_cap):
            subset_rel = GSM8K_SUBSET_REL.format(cap=cap, s=s, cell=cell)
            kid["argv"] += ["--item-subset", f"{ATTEMPT_ROOT_PLACEHOLDER}/{subset_rel}"]
            kid.update(subset_rel=subset_rel, source_out_rel=GSM8K_OUT_REL.format(cap=base_cap, s=s, cell=cell), source_cap=int(base_cap))
        kids.append(kid)
    return kids


#: the misalign runner's DEFAULT per-request timeout (`sisteer.capeval.misalign.runner.DEFAULT_TIMEOUT_S`) is 7,500 s,
#: sized for the agentic family's 16k cap; at a small cap a LOST response parks the runner past the job's walltime (observed: vLLM at
#: 0 running / 0 waiting while the runner sat at 399/400 for 24 min, losing the job). Every misalign child therefore carries `--timeout-s`
#: sized from ITS cap: the cap's projected decode time at a conservative per-sequence decode FLOOR x a factor (prefill, queueing, the proxy
#: hop), floored at 900 s (the cluster value, `EM8Q_TIMEOUT_S`) -- never the 7,500 s default. The runner retries a timed-out send
#: (`--max-retries`, default 3), so a lost response costs at most (1 + retries) x this timeout, not the walltime.
MISALIGN_RUNNER_DEFAULT_TIMEOUT_S = 7500.0
MISALIGN_TIMEOUT_FLOOR_S = 900
MISALIGN_TIMEOUT_REF_TOK_S_PER_SEQ = 10.0       # a saturated-batch per-sequence decode floor for the 4B target on an H200 (S5's tok/s/seq measurement supersedes it)
MISALIGN_TIMEOUT_FACTOR = 4.0


def misalign_timeout_s(cap: int) -> int:
    """`max(900, ceil(cap / 10 tok/s/seq x 4))` seconds: 3,000-token cap -> 1,200 s; a <= 1k cap -> the 900 s floor; never the runner's 7,500 s."""
    projected = int(cap) / MISALIGN_TIMEOUT_REF_TOK_S_PER_SEQ * MISALIGN_TIMEOUT_FACTOR
    return int(max(MISALIGN_TIMEOUT_FLOOR_S, -(-projected // 1)))


def _misalign_children(engine: str, W: str, cap: int) -> List[Dict[str, Any]]:
    """python -m sisteer.capeval.misalign.runner ONE CHILD PER CELL (`--only-cells`), the flags
    `--sampler vendor --seed-policy per_response --seed-base s`, `--timeout-s` sized from the cap, `CUDA_VISIBLE_DEVICES=""` for the runner."""
    kids = []
    timeout_s = misalign_timeout_s(cap)
    for run in arm_runs(engine):
        for behavior in ("leaking", "murder"):
            out_rel = MISALIGN_OUT_REL.format(s=run["s"], behavior=behavior)
            argv = [DRIVER_PY_PLACEHOLDER, "-m", "sisteer.capeval.misalign.runner", "--proxy-url", PROXY_URL_PLACEHOLDER,
                    "--cells", f"{ATTEMPT_ROOT_PLACEHOLDER}/{SERVE_DIR_REL}/cells.json", "--family", "agentic",
                    "--out", f"{ATTEMPT_ROOT_PLACEHOLDER}/{out_rel}", "--concurrency", CONCURRENCY_PLACEHOLDER,
                    "--n", "10", "--behaviors", behavior, "--max-tokens", str(cap), "--sampler", "vendor",
                    "--seed-policy", "per_response", "--seed-base", str(run["s"]),
                    # cells.json is the standup's enumeration (HARNESS ids) -> the selection is spelled in that namespace
                    "--only-cells", harness_id(run["cell"]),
                    "--timeout-s", str(timeout_s),
                    "--api-key", "petri-proxy"]
            kids.append({"eval_leg": f"misalign_{behavior}", "arm": run["arm"], "cell": run["cell"], "s": run["s"],
                         "cell_harness": harness_id(run["cell"]), "behavior": behavior, "out_rel": out_rel,
                         "task": f"agentic/{behavior}", "concurrency_rule": "S_over_ncells", "argv": argv, "cap": cap,
                         "timeout_s": timeout_s,
                         "timeout_rule": f"max({MISALIGN_TIMEOUT_FLOOR_S}, cap / {MISALIGN_TIMEOUT_REF_TOK_S_PER_SEQ:g} tok/s/seq x {MISALIGN_TIMEOUT_FACTOR:g}); "
                                         f"never the runner's {MISALIGN_RUNNER_DEFAULT_TIMEOUT_S:.0f} s default (EM 2026-09-18)",
                         "env": {"CUDA_VISIBLE_DEVICES": ""}})
    return kids


def _multiturn_children(leg: str, engine: str, W: str, L: int, shard: int, nshards: int, pool_url: str, *,
                        runs: Optional[Sequence[Mapping[str, Any]]] = None, cfg: Optional[str] = None) -> List[Dict[str, Any]]:
    """jailbreak_run.py generate per (arm, goal sub-shard): attacker on the deployed pool, target served, the vendor row +
    per-turn seeds at base s (`--multiturn-target-seed-base`), the 4,096 per-reply cap, the leg's window."""
    tree = tree_local(W)
    kids = []
    for run in (runs if runs is not None else arm_runs(engine)):
        # the WAVE config: the PER-SEED Arditi wave config (three distinct private_runs), the view's unsampled config
        cfg_ = cfg or (ARDITI_WAVE_CONFIG_REL.format(s=run["s"]) if engine == ARDITI else VIEW_CONFIG_REL.format(view=engine))
        argv = [DRIVER_PY_PLACEHOLDER, f"{tree}/scripts/jailbreak_run.py", "generate", "--config", cfg_,
                "--attack", leg, "--defense", run["cell"], "--confirmatory", "--trial", "0",
                "--shard", str(shard), "--nshards", str(nshards),
                "--private-root", f"{ATTEMPT_ROOT_PLACEHOLDER}/exp/transcripts_results",
                "--attacker-engine", "vllm", "--attacker-endpoint", pool_url,
                "--target-engine", "served", "--target-proxy-url", PROXY_URL_PLACEHOLDER,
                "--target-routing-table", ROUTING_TABLE_PLACEHOLDER, "--target-max-model-len", str(L),
                "--target-max-new-tokens", "4096", "--multiturn-target-decode", "vendor",
                "--multiturn-target-seed", "per_turn", "--multiturn-target-seed-base", str(run["s"]),
                "--multiturn-target-policy-ref", MULTITURN_DECODE_POLICY_REF,
                "--model-key", MODEL_KEY, "--expect-pair-n-streams", "5"]
        kids.append({"eval_leg": leg, "arm": run["arm"], "cell": run["cell"], "s": run["s"], "task": f"multi_turn/{leg}",
                     "concurrency_rule": "two_drivers_per_slot", "argv": argv, "cap": 4096, "config": cfg_})
    return kids


def _petri_children(engine: str) -> List[Dict[str, Any]]:
    """NO in-container driver (§4.5): the grading host drive runs `petri_run.py run ... --target-decode vendor --target-seed-base s`
    per cell-index against the tunnel; recorded here so the identity files and coverage rows are enumerated per arm-run."""
    kids = []
    for i, run in enumerate(arm_runs(engine)):
        kids.append({"eval_leg": "petri", "arm": run["arm"], "cell": run["cell"], "s": run["s"], "task": "petri/<behaviour>",
                     "concurrency_rule": "grading_host_drive", "argv": None, "cap": 1024,
                     "grading_host_drive": {"cell_index": i if engine != ARDITI else 1, "target_decode": "vendor", "target_seed_base": run["s"],
                                      "petri_eval_max_connections": 4 if engine == ARDITI else 3}})
    return kids


#: the ladder KIND per standup leg (`seedrep_ladder.LADDERS` keys; the consumer's `--kind` choices)
LADDER_KIND_OF_LEG: Dict[str, str] = {**{leg: "static" for leg in STATIC_LEGS}, "controls": "controls", "misalign": "misalign", "gsm8k": "gsm8k"}
#: rung rows the consumer does NOT draw: GSM8K (the Inspect driver draws the queue-derived subset; `consume --kind gsm8k` refuses by
#: design, impl_ws2e deviation 6) and the Petri context-overflow re-run proxy (no driver in-container; the grading host drive)
RUNG_MODE_CONSUME = "consume"
RUNG_MODE_GSM8K_SUBSET = "gsm8k_inspect_subset"
#: the GSM8K rung IS executable: EVERY ladder leg incl. GSM8K escalates at > 1 % at cap up to 16,384. Its container stands the
#: 24,832 x 21 thinking-ON engine up, runs the ASSEMBLY HELPER (`scripts/modal/seedrep_gsm8k_assembly.py`, under the driver venv)
#: which waits for the queue `(gsm8k, view, 16384)` to CLOSE (`reconcile_queue` over the declared base producers' ACCEPTED
#: attempts), snapshots the results Dict, and per (arm, s) slot runs `gsm8k_subset_from_queue` -- `inspect_rung.classify_store`
#: over the ACCEPTED base store (the ONE at-cap authority, the > 1 % trigger and the eligible set) cross-checked against the CLOSED
#: queue's accepted entries of EXACTLY that slot -- writing `rung_subset.json` for every slot that ESCALATES; then ONE Inspect child
#: per escalated slot (`--item-subset`, cap 16,384, identity v2) generates into a NEW root beside its source; the row completes
#: through the SAME attempt / acceptance rules as every other rung (COMPLETE -> inventory (the ceiling: empty) -> ONE accepted put).
#: A slot below the trigger is recorded `not_triggered` (no child); any assembly refusal fails the row by name.
GSM8K_RUNG_RULE = ("executable: the queue-derived Inspect assembly -- CLOSED queue + accepted source store -> inspect_rung.classify_store "
                   "(> 1 % at cap -> escalate; eligible = at-cap minus loops) cross-checked by rung_redraw.gsm8k_subset_from_queue per (arm, s) slot -> "
                   "rung_subset.json -> one served_capability_run child per escalated slot at cap 16,384 (--item-subset, identity v2) -> the rung's "
                   "attempt / acceptance; 16,384 is the ceiling (nothing is queued above)")
GSM8K_RUNG_DEFERRED_WHY = None          # retired: the round-2 deferral was rejected; kept as a name so old readers see None, never a reason
RUNG_MODE_PETRI_RERUN = "petri_overflow_rerun"
#: fc9d: the ONE definition of a RUNG row and of a LADDER-CONSUMER rung, shared by the plan's validation, the certificate's S5
#: coverage and the container's dispatch (`seedrep_served.run_view_leg`). A row is a rung iff its `kind` is "rung", its stage is not the
#: base stage, its `rung_mode` is an executable rung mode, and it carries a consumer binding EXACTLY when its mode is `consume`; a ladder
#: consumer is a rung whose mode is `consume`. Any row whose collection membership (standups vs rungs), kind, stage, rung_mode and consumer
#: binding contradict each other is refused by `validate_plan` (`row_shape_problems`) -- a kind flipped to "base" on a consume rung used to pass
#: the PLAN gate and read `rung_self_measured` at G8 while the container ran the row's children at the higher cap over the full population.
EXECUTABLE_RUNG_MODES_PLAN: Tuple[str, ...] = (RUNG_MODE_CONSUME, RUNG_MODE_GSM8K_SUBSET, RUNG_MODE_PETRI_RERUN)
#: fc9e: the permitted rung mode is a function of the LEG, in one place -- a static family / controls / misalign rung consumes
#: its ladder queue; the GSM8K rung is the queue-derived Inspect subset; the Petri rung is the context-overflow re-run proxy. A (leg, mode) pair
#: outside this map is refused by name (a gsm8k row relabelled `consume` with a real consumer argv used to read as a ladder consumer).
LADDER_MODE_OF_LEG: Dict[str, str] = {**{leg: RUNG_MODE_CONSUME for leg in STATIC_LEGS}, "controls": RUNG_MODE_CONSUME, "misalign": RUNG_MODE_CONSUME,
                                      "gsm8k": RUNG_MODE_GSM8K_SUBSET, "petri": RUNG_MODE_PETRI_RERUN}


def is_rung_row(row: Mapping[str, Any]) -> bool:
    mode = row.get("rung_mode")
    has_consumer = isinstance(row.get("rung_consumer"), (list, tuple)) and bool(row.get("rung_consumer"))     # the consumer ARGV (rung_consumer_argv)
    return (row.get("kind") == "rung" and str(row.get("stage")) != STAGE_BASE and mode in EXECUTABLE_RUNG_MODES_PLAN
            and mode == LADDER_MODE_OF_LEG.get(str(row.get("leg"))) and has_consumer == (mode == RUNG_MODE_CONSUME))


def is_ladder_consumer_row(row: Mapping[str, Any]) -> bool:
    """A rung row (`is_rung_row`) whose leg's ladder mode is `consume`, for a leg with a KNOWN ladder kind (LADDER_KIND_OF_LEG) that its consumer
    argv names by `--kind` -- never `None == None` (the Petri rerun row relabelled consume without --kind used to pass the argv check)."""
    if not is_rung_row(row) or row.get("rung_mode") != RUNG_MODE_CONSUME:
        return False
    kind_want = LADDER_KIND_OF_LEG.get(str(row.get("leg")))
    return kind_want is not None and _flag_value(list(row.get("rung_consumer") or []), "--kind") == kind_want


def row_shape_problems(row: Mapping[str, Any], collection: str) -> List[str]:
    """Why a row's shape contradicts itself or its collection (empty = consistent): `standups` rows are kind "base" (or the smoke-only S5 window
    probe kind) at the base stage with no rung_mode and no consumer binding; `rungs` rows are kind "rung" at a non-base stage with an executable rung_mode and a consumer binding
    exactly when the mode is `consume`."""
    sid = row.get("standup_id", "?")
    kind, stage, mode = row.get("kind"), str(row.get("stage")), row.get("rung_mode")
    has_consumer = isinstance(row.get("rung_consumer"), (list, tuple)) and bool(row.get("rung_consumer"))     # the consumer ARGV (rung_consumer_argv)
    out: List[str] = []
    if collection == "standups":
        if kind not in ("base", S5_PROBE_KIND):
            out.append(f"{sid}: a standups row with kind {kind!r} (base rows are kind 'base'; the smoke-only S5 window probe {S5_PROBE_KIND!r})")
        if (kind == S5_PROBE_KIND) != bool(row.get("s5_probe")):
            out.append(f"{sid}: kind {kind!r} with s5_probe={row.get('s5_probe')!r} (the probe kind and the probe flag go together)")
        if stage != STAGE_BASE:
            out.append(f"{sid}: a standups row at stage {stage!r} (base rows are at stage {STAGE_BASE!r})")
        if mode is not None:
            out.append(f"{sid}: a standups row with rung_mode {mode!r} (a base row has none)")
        if has_consumer:
            out.append(f"{sid}: a standups row with a consumer binding (a base row produces; it never consumes)")
    elif collection == "rungs":
        if kind != "rung":
            out.append(f"{sid}: a rungs row with kind {kind!r} -- the container dispatches on kind, so this row would run its {len(row.get('children') or [])} children as "
                       f"ordinary generation at stage {stage!r} instead of consuming the ladder queue (rungs rows are kind 'rung')")
        if stage == STAGE_BASE:
            out.append(f"{sid}: a rungs row at the base stage")
        if mode not in EXECUTABLE_RUNG_MODES_PLAN:
            out.append(f"{sid}: a rungs row with rung_mode {mode!r} (executable: {list(EXECUTABLE_RUNG_MODES_PLAN)})")
        if (mode == RUNG_MODE_CONSUME) != has_consumer:
            out.append(f"{sid}: rung_mode {mode!r} with{'out' if not has_consumer else ''} a consumer binding -- a consume rung carries one, every other mode none")
        leg = str(row.get("leg"))
        if mode in EXECUTABLE_RUNG_MODES_PLAN and mode != LADDER_MODE_OF_LEG.get(leg):
            out.append(f"{sid}: leg {leg!r} with rung_mode {mode!r} -- this leg's ladder mode is {LADDER_MODE_OF_LEG.get(leg)!r} (LADDER_MODE_OF_LEG); "
                       f"the pair is refused whatever the row carries")
        if mode == RUNG_MODE_CONSUME and has_consumer:
            kind_want, got = LADDER_KIND_OF_LEG.get(leg), _flag_value(list(row["rung_consumer"]), "--kind")
            if kind_want is None or got != kind_want:
                out.append(f"{sid}: a consume rung for leg {leg!r} must name a KNOWN ladder kind by --kind ({kind_want!r} per LADDER_KIND_OF_LEG); the consumer argv carries {got!r}")
    else:
        out.append(f"{sid}: unknown collection {collection!r}")
    return out


#: the placeholders a rung consumer argv carries beyond the driver's (resolved at function entry, never at plan time)
ATTEMPT_ID_PLACEHOLDER = "{attempt_id}"
CALL_ID_PLACEHOLDER = "{call_id}"
ATTEMPT_ROOT_REL_PLACEHOLDER = "{attempt_root_rel}"          # the attempt root RELATIVE to the volume root (no leading /vol/)
RUNG_CONSUMER_API_KEY = "petri-proxy"
RUNG_CONSUMER_POLL_S = 30
#: the Modal Volume the wave lives on (`/vol`); the consumer binds its commit/reload through `--volume`
MODAL_VOLUME_NAME = "twosi-fits"
RUNG_DEADLINE_MARGIN_S = 1800
#: `--speculative` (draw live entries of not-yet-accepted producers) on for static / controls, off for misalign (impl_ws2e Q1)
RUNG_SPECULATIVE_KINDS: Tuple[str, ...] = ("static", "controls")


def rung_consumer_argv(*, leg: str, engine: str, stage: int, shard: int, feeder_shards: Sequence[str], rung_shards: Sequence[str],
                       next_cap: Optional[int], W: str, W7: str, wave_uuid: str, timeout_s: int, width: int) -> List[str]:
    """The rung container's consumer command, generated DIRECTLY against the `rung_redraw.py consume` subcommand
    (the app used to invoke `--plan-row ...`, which that parser has never had): the queue THIS rung consumes is
    `(leg, engine, cap_next = this rung's cap)` whose declared producers are the FEEDER stage's shards; when a cap exists
    above, this container is itself a producer of the next queue (`--rung-shard` / `--rung-shards` / `--attempt-root`).
    `--finish-producer` is NOT passed: the app owns COMPLETE -> inventory -> acceptance -> marker for every row, so the
    consumer never accepts an attempt whose COMPLETE.json the app has not written. ``width`` is the rung's
    ATTESTED S (`--width`, REQUIRED: draws in flight = the served width); `--volume` names the Modal Volume
    whose commit / reload the consumer binds onto `--vol-root`. Round-tripped through the real parser by test
    (imported by path)."""
    kind = LADDER_KIND_OF_LEG[leg]
    if int(width) < 1:
        raise ValueError(f"rung consumer width must be the attested S >= 1, got {width!r}")
    argv = [DRIVER_PY_PLACEHOLDER, f"{tree_local(W)}/scripts/sdfbase/seedrep/rung_redraw.py", "consume",
            "--vol-root", VOL_PATH, "--volume", MODAL_VOLUME_NAME, "--W7", W7, "--wave-uuid", wave_uuid, "--leg", leg, "--view", engine, "--kind", kind,
            "--cap-next", str(int(stage)), "--producers", ",".join(str(x) for x in feeder_shards),
            "--attempt-id", ATTEMPT_ID_PLACEHOLDER, "--call-id", CALL_ID_PLACEHOLDER, "--proxy-url", PROXY_URL_PLACEHOLDER,
            "--api-key", RUNG_CONSUMER_API_KEY, "--width", str(int(width)),
            # this container's SLOT on every rung (the ceiling rung too) -- a later execution of the slot
            # adopts an earlier one's elected results' evidence; the container-LOCAL attempt root the adopted request pipeline is merged into
            "--shard", str(shard), "--attempt-root-local", ATTEMPT_ROOT_PLACEHOLDER]
    if next_cap is not None:
        argv += ["--rung-shard", str(shard), "--rung-shards", ",".join(str(x) for x in rung_shards), "--attempt-root", ATTEMPT_ROOT_REL_PLACEHOLDER]
    argv += ["--poll-s", str(RUNG_CONSUMER_POLL_S), "--deadline-s", str(int(timeout_s) - RUNG_DEADLINE_MARGIN_S)]
    if kind in RUNG_SPECULATIVE_KINDS:
        argv.append("--speculative")
    return argv


def is_psr_cell(cell: str) -> bool:
    """A PSR-kind object (a residual-slot adapter): the lane's two PSR fits and every `__psr_` difference child."""
    return cell in PSR_CELLS or "__psr_" in cell


def dose_arm_runs(view: str, cells: Mapping[str, str]) -> List[Dict[str, Any]]:
    """A dose standup's arm-runs at seed 1: the same-standup `none` anchor + one arm per dose cell, the ARM labelled by its
    STAGED id (dose-prefixed, unique per object) while `cell` stays the config's cell id (the driver's `--defense`)."""
    out = [{"arm": f"{view}__none", "cell": "none", "s": 1, "object": "none", "view": view}]
    out += [{"arm": sid, "cell": c, "s": 1, "object": sid, "view": view} for c, sid in cells.items()]
    return out


def dose_standup_rows(*, roster: Mapping[str, Any], pins: Mapping[str, Any], legs: Mapping[str, Mapping[str, Any]], W: str, W7: str,
                      wave_uuid: str, reqs_sha256: str, pools_by_attack: Mapping[str, Sequence[Mapping[str, Any]]],
                      adapter_hashes: Mapping[str, str], s4: Mapping[str, Any],
                      peer_roster: Optional[Mapping[str, Any]] = None, peer_pins: Optional[Mapping[str, Any]] = None,
                      register: bool = False) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Per SEED-1 dose view of the roster (:func:`dose_views`, :data:`DOSE_DISCOVERY_RULE`), THREE single-container
    standups (Crescendo, ActorAttack, XSTest) carrying the view's dose cells + a `none` anchor -- same W, image, geometry class,
    S5 reference and smoke gates as our legs, never registered in s1 / s2 / s3. Returns the rows and a record (`views`, `skipped`,
    `shape`, `n_standups`, `standups`, `status`); with no seed-1 dose view in the roster the record says so and no row is emitted
    (the peer appends the views; their pins bind the bytes). :data:`DOSE_SHAPE`: N one-dose views -> 3N standups."""
    shared_seed1, skipped = dose_views(roster)          # the SHARED roster: reported only, never a source of peer views
    rows: List[Dict[str, Any]] = []
    rec: Dict[str, Any] = {"discovery": PEER_DISCOVERY_RULE, "legs": list(DOSE_STANDUP_LEGS), "controls": list(DOSE_CONTROLS), "views": {},
                           "skipped_shared_roster": skipped, "shared_roster_seed1_views_not_stood_up": sorted(shared_seed1), "shape": DOSE_SHAPE,
                           "peer_files": {"roster": PEER_ROSTER_REL, "pins": PEER_PINS_REL, "expected_views": list(PEER_EXPECTED_VIEWS)},
                           "n_views": 0, "n_standups": 0, "containers": 0, "standups": [], "pins_source": None}
    if not register:
        rec.update(status=DOSE_STATUS_NOT_IN_WAVE, why=DOSE_D71_WHY, peer_files_given=(peer_roster is not None and peer_pins is not None),
                   pending={"containers_when_registered": 0, "s1_s3_legs_affected": False, "dormant_path": "--register-peer-dose-standups (default off)"})
        return rows, rec
    if peer_roster is None:
        n_exp = len(PEER_EXPECTED_VIEWS) * len(DOSE_STANDUP_LEGS)
        rec.update(status=DOSE_STATUS_PENDING,
                   pending={"why": "the peer's roster + pin file were not given (--peer-roster / --peer-pins); Strength's spec is not yet approved",
                            "containers_when_registered": n_exp, "per_view": len(DOSE_STANDUP_LEGS), "d43_estimate_one_union_view": len(DOSE_STANDUP_LEGS),
                            "s1_s3_legs_affected": False})
        return rows, rec
    if peer_pins is None:
        raise ValueError("--peer-pins is required with --peer-roster (the peer pins its dose objects in its own file)")
    views = peer_dose_views(peer_roster)
    n_standups = len(views) * len(DOSE_STANDUP_LEGS)
    pins_source = "peer pin file"
    target_pins = peer_pins.get(MODEL_KEY) or {}
    rec.update(n_views=len(views), n_standups=n_standups, containers=n_standups, pins_source=pins_source,
               status=f"{len(views)} peer seed-1 dose view(s) -> {n_standups} single-container standups (per_dose_view)",
               unexpected_view_names=sorted(set(views) - set(PEER_EXPECTED_VIEWS)))
    for view, block in sorted(views.items()):
        cells = {str(c): str(sid) for c, sid in (block.get("cells") or {}).items() if isinstance(sid, str)}
        if not cells:
            raise ValueError(f"dose view {view} has no string-form cells (a dose view names its staged objects)")
        pins_by_cell = {}
        for c, sid in cells.items():
            entry = target_pins.get(sid)
            if not isinstance(entry, Mapping) or not entry.get("object_sha256"):
                raise ValueError(f"dose view {view}: {pins_source}[{MODEL_KEY}][{sid}] has no object_sha256 (the peer pins its dose objects)")
            if sid in DOSE_CHILD_SHA and str(entry["object_sha256"]) != DOSE_CHILD_SHA[sid]:
                raise ValueError(f"dose view {view}: {sid} is pinned {entry['object_sha256']}, not the sha Strength relayed {DOSE_CHILD_SHA[sid]} (D45 / 2026-09-18)")
            if c not in HARNESS_ID_OF:
                raise ValueError(f"dose view {view}: cell {c!r} has no harness translation in this lane (the dose cell id must be its parent's)")
            if c == PAIRED_DIFF_CELL and (target_pins.get(PAIRED_DIFF_PARENT_STAGED_ID) or {}).get("object_sha256") not in (None, PAIRED_DIFF_PARENT_SHA256):
                raise ValueError(f"peer pins: {PAIRED_DIFF_PARENT_STAGED_ID} pinned {target_pins[PAIRED_DIFF_PARENT_STAGED_ID].get('object_sha256')}, bank holds {PAIRED_DIFF_PARENT_SHA256}")
            pins_by_cell[c] = str(entry["object_sha256"])
        config = str(block.get("config") or VIEW_CONFIG_REL.format(view=view))
        manifest = str(block.get("manifest") or VIEW_MANIFEST_REL.format(view=view))
        rec["views"][view] = {"cells": cells, "config": config, "manifest": manifest, "n_cells": len(cells)}
        es = {"config": config, "manifest": manifest, "bank_view": BANK_VIEW_LOCAL.format(view=view), "cells": list(cells), "digests": dict(pins_by_cell)}
        runs = dose_arm_runs(view, cells)
        routes = ["none", *cells]
        objects = {c: {"staged_id": sid, "object_sha256": pins_by_cell[c], "dose_view": view} for c, sid in cells.items()}
        for leg in DOSE_STANDUP_LEGS:
            spec = legs[leg]
            window = spec["base"]
            geom = served_geometry(window)
            argv = standup_argv(leg=spec, window=window, engine=view, W=W, reqs_sha256=reqs_sha256, pins_by_cell={}, arditi_sha256="", spec=es)
            pool = pools_by_attack[leg][0] if leg in MULTITURN_LEGS else None
            if leg in MULTITURN_LEGS:
                kids = _multiturn_children(leg, view, W, geom["L"], 0, 1, pool["url_placeholder"], runs=runs, cfg=config)
            else:
                kids = _controls_children(view, W, int(window["cap"]), runs=runs, cfg=config, controls=DOSE_CONTROLS, base_cap=int(spec["cap"]))
            for k in kids:
                k["dose_view"] = view
                k["registry_identity_rel"] = identity_plan_rel(k["eval_leg"], k["arm"], k["s"])
            row = {
                "standup_id": standup_id(leg, view, STAGE_BASE, 0), "kind": "base", "leg": leg,
                "eval_legs": (list(DOSE_CONTROLS) if leg == "controls" else list(EVAL_LEGS_OF[leg])),
                "engine": view, "view": view, "stage": STAGE_BASE, "shard": 0, "nshards": 1, "seed_bases": [1], "dose_standup": True,
                "control_key": control_key(W7, wave_uuid, leg, view, STAGE_BASE, 0),
                "accepted_key": accepted_key(W7, wave_uuid, leg, view, STAGE_BASE, 0),
                "modal": {"function": "run_view_leg", "gpu": "H200", "cpu": LEG_CPU, "memory_mib": LEG_MEMORY_MIB,
                          "timeout_s": int(spec["timeout_h"] * 3600), "retries": 0, "max_containers": 1},
                "geometry": geom, "argv": argv, "route_set": routes, "route_set_harness": [harness_id(c) for c in routes],
                "cells_served_expected": cells_served_for_routes(routes), "objects": objects,
                "adapters": adapter_expectations(objects, adapter_hashes),
                "readiness": readiness_block(geom, max_lora_rank=VIEW_MAX_LORA_RANK, max_loras=len(cells) + 1,
                                             residual_slot_armed=[c for c in cells if is_psr_cell(c)], max_logprobs=spec.get("max_logprobs"), thinking=False),
                "children": kids, "producers": [], "ladder": None, "rung_consumer": None, "rung_mode": None,
                "pool": (pool["name"] if pool else None), "arm_runs": runs, "expected_h200_h": [round(x / 4, 2) for x in spec["h200_h"]],
            }
            if leg in MULTITURN_LEGS:
                row["window_provenance"] = multiturn_window_provenance(leg, geom["L"], s4)
                row["depends_on"] = depends_on_block(pool["name"])
            rows.append(row)
            rec["standups"].append(row["standup_id"])
    return rows, rec


def children_for(leg_spec: Mapping[str, Any], engine: str, W: str, *, cap: int, shard: int, nshards: int,
                 pool_url: Optional[str], pins_by_cell: Optional[Mapping[str, str]] = None, arditi_sha256: str = "") -> List[Dict[str, Any]]:
    kind, leg = leg_spec["kind"], leg_spec["leg"]
    suffix = "" if cap == leg_spec["cap"] else f"_cap{cap}"
    if kind == "static":
        window = leg_spec["base"] if cap == leg_spec["cap"] else leg_spec["rungs"][cap]
        kids = _static_children(leg, engine, W, cap, suffix, served_geometry(window)["S"])
    elif kind == "bon":
        kids = _bon_children(engine, W)
    elif kind == "controls":
        kids = _controls_children(engine, W, cap, base_cap=int(leg_spec["cap"]))
    elif kind == "gsm8k":
        window = leg_spec["base"] if cap == leg_spec["cap"] else leg_spec["rungs"][cap]
        kids = _gsm8k_children(engine, W, cap, pins_by_cell=pins_by_cell, arditi_sha256=arditi_sha256, base_cap=int(leg_spec["cap"]), geometry_eval=window["eval"])
    elif kind == "misalign":
        kids = _misalign_children(engine, W, cap)
    elif kind == "multiturn":
        kids = _multiturn_children(leg, engine, W, int(leg_spec["base"]["L"]), shard, nshards, pool_url or "{pool_url}")
    elif kind == "petri":
        kids = _petri_children(engine)
    else:
        raise ValueError(f"unknown leg kind {kind!r}")
    for k in kids:
        k["registry_identity_rel"] = identity_plan_rel(k["eval_leg"], k["arm"], k["s"])
    return kids


# --------------------------------------------------------------------------- #
# the plan                                                                      #
# --------------------------------------------------------------------------- #

PLAN_SCHEMA = "seedrep_launch_plan/1"
LEDGER_SCHEMA = "seedrep_capacity_ledger/1"
LEG_CPU = 32
LEG_MEMORY_MIB = 196608
PETRI_CPU = 16
PETRI_MEMORY_MIB = 98304


def cells_served_for_routes(routes: Sequence[str]) -> Dict[str, List[str]]:
    """What the NATIVE engine's `cells_served.json` lists for a route set, in HARNESS ids: the `canonical` rows = the route set
    translated, the `controls` rows = the derived zero-adapter control `none__lora0` (every lora_native engine serves it)."""
    return {"canonical": [harness_id(c) for c in routes], "controls": [NONE_LORA0]}


def readiness_block(geom: Mapping[str, Any], *, max_lora_rank: int, max_loras: int, residual_slot_armed: Sequence[str],
                    max_logprobs: Optional[int], thinking: bool) -> Dict[str, Any]:
    """What the container ASSERTS at readiness (§4.3), from the served geometry and the engine's adapter facts."""
    return {"engine": "lora_native", "attested_max_model_len": geom["L"], "attested_max_num_seqs": geom["S"],
            "adjustment": geom["expected_adjustment"], "cudagraph_mode": "FULL", "capture_surface_ge": geom["S"],
            "max_lora_rank": int(max_lora_rank), "max_loras": int(max_loras), "residual_slot_armed": list(residual_slot_armed),
            "max_logprobs": max_logprobs, "thinking_mode": "on" if thinking else "off", "server_default_policy": SERVER_DEFAULT_POLICY}


def depends_on_block(pool_name: Optional[str]) -> Dict[str, Any]:
    """A multi-turn row's launch dependency (main #211): its pinned pool, registered in RS/pool_urls.json and healthy."""
    return {"pool": pool_name, "pool_urls_json": "RS/pool_urls.json", "health": POOL_HEALTH_RULE, "why": ATTACKER_PRESENCE_RULE}


def adapter_expectations(objects: Mapping[str, Mapping[str, Any]], adapter_hashes: Mapping[str, str], *, expected_reused: bool = True) -> Dict[str, Any]:
    """Gate 10: every registered adapter (each object + the zero adapter) is EXPECTED to be a re-use in production (S1 / S1b / S2
    banked every hash first); the enumeration is per (standup, cell): the object pin at the draft, the intervention hash once the
    smokes record it (`--adapter-hashes`). ``expected_reused=False`` mints the block for a row that is a FIRST EXPORT by construction
    (the smoke-only S5 probe rows; W15d B3): no re-use asserted -- the consumer (gate 10) accepts an export and a bank hit alike."""
    reuse = {cell: {"expected_reused": bool(expected_reused), "object_sha256": o["object_sha256"], "staged_id": o["staged_id"],
                    "intervention_hash": adapter_hashes.get(f"{o['staged_id']}:{o['object_sha256']}") or adapter_hashes.get(o["staged_id"])}
             for cell, o in objects.items()}
    reuse[NONE_LORA0] = {"expected_reused": bool(expected_reused), "object_sha256": None, "staged_id": NONE_LORA0, "intervention_hash": adapter_hashes.get(NONE_LORA0)}
    return reuse


def standup_id(leg: str, engine: str, stage: Any, shard: int) -> str:
    return f"{leg}__{engine}__{stage}__shard{int(shard):02d}"


def gsm8k_rung_assembly(kids: Sequence[Mapping[str, Any]], *, cap: int, feeder_shards: Sequence[str], base_cap: int) -> Dict[str, Any]:
    """The GSM8K rung row's ASSEMBLY block: the helper the app runs under the driver venv, the queue it waits on (this cap, fed
    by the base shards), and ONE slot per child = (arm, s, cell) with the accepted SOURCE store's rel (the base child's per-cell root), the
    subset the helper writes and the rung child's own root."""
    return {"kind": "gsm8k", "helper": GSM8K_ASSEMBLY_HELPER_REL, "consumes": int(cap), "source_cap": int(base_cap), "feeder_shards": [str(x) for x in feeder_shards],
            "slots": [{"arm": k["arm"], "s": int(k["s"]), "cell": k["cell"], "source_out_rel": k["source_out_rel"], "subset_rel": k["subset_rel"], "out_rel": k["out_rel"]}
                      for k in kids],
            "trigger": "inspect_rung.classify_store: n_at_cap / n > 0.01 (GSM8K_TRIGGER_THRESHOLD; the 3 % trigger is retired for this wave) -> escalate; "
                       "eligible = at-cap items minus loops; the queue's accepted entries of the slot are cross-checked (a queued item the source never "
                       "filed at cap refuses)",
            "rule": GSM8K_RUNG_RULE}


def _row(*, leg_spec: Mapping[str, Any], engine: str, stage: Any, shard: int, nshards: int, window: Mapping[str, Any],
         W: str, W7: str, wave_uuid: str, reqs_sha256: str, pins_by_view: Mapping[str, Mapping[str, str]],
         staged_by_view: Mapping[str, Mapping[str, str]], arditi_sha256: str, pool_url: Optional[str],
         kind: str, max_containers: int, producers: List[Dict[str, Any]], adapter_hashes: Mapping[str, str],
         ladder: Optional[Mapping[str, Any]] = None, pool_name: Optional[str] = None, s4: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    leg = leg_spec["leg"]
    pins = pins_by_view.get(engine, {}) if engine != ARDITI else {}
    # F-T: a controls RUNG re-draws the ladder controls alone -- its engine serves no MMLU (no `--max-logprobs`, no `mmlu` eval leg)
    controls_rung = kind == "rung" and leg_spec["kind"] == "controls"
    leg_row = {**leg_spec, "max_logprobs": None} if controls_rung else leg_spec
    argv = standup_argv(leg=leg_row, window=window, engine=engine, W=W, reqs_sha256=reqs_sha256,
                        pins_by_cell=pins, arditi_sha256=arditi_sha256)
    geom = served_geometry(window)
    routes = expected_route_set(engine)
    if engine == ARDITI:
        objects = {ARDITI_CELL: {"staged_id": ARDITI_STAGED_ID, "object_sha256": arditi_sha256, "kind": "vector_all"}}
    else:
        objects = {c: {"staged_id": staged_by_view[engine][c], "object_sha256": pins[c]} for c in FITTED_CELLS}
    reuse = adapter_expectations(objects, adapter_hashes)
    cap = int(window["cap"])
    kids = children_for(leg_spec, engine, W, cap=cap, shard=shard, nshards=nshards, pool_url=pool_url, pins_by_cell=pins, arditi_sha256=arditi_sha256)
    eval_legs = [CONTROLS_EVAL_LEG_OF[c] for c in CONTROLS_LADDER] if controls_rung else list(EVAL_LEGS_OF[leg])
    timeout_s = int((8 if kind == "rung" else leg_spec["timeout_h"]) * 3600)
    # the rung consumer (the real CLI) or the mode the app must execute another way (none for a base row)
    rung_consumer, rung_mode, rung_assembly = None, None, None
    if kind == "rung":
        if leg_spec["kind"] == "petri":
            rung_mode = RUNG_MODE_PETRI_RERUN
        elif LADDER_KIND_OF_LEG.get(leg) == "gsm8k":
            rung_mode = RUNG_MODE_GSM8K_SUBSET
            rung_assembly = gsm8k_rung_assembly(kids, cap=cap, feeder_shards=(ladder or {}).get("feeder_shards") or [], base_cap=int(leg_spec["cap"]))
        else:
            rung_mode = RUNG_MODE_CONSUME
            rung_consumer = rung_consumer_argv(leg=leg, engine=engine, stage=int(stage), shard=shard,
                                               feeder_shards=(ladder or {}).get("feeder_shards") or [],
                                               rung_shards=(ladder or {}).get("shards") or [str(shard)],
                                               next_cap=(ladder or {}).get("cap_next"), W=W, W7=W7, wave_uuid=wave_uuid, timeout_s=timeout_s,
                                               width=served_geometry(window)["S"])
    row_out = {
        "standup_id": standup_id(leg, engine, stage, shard), "kind": kind, "leg": leg, "eval_legs": eval_legs,
        "engine": engine, "view": engine, "stage": str(stage), "shard": int(shard), "nshards": int(nshards),
        "seed_bases": ([SEED_OF_VIEW[engine]] if engine != ARDITI else list(ARDITI_SEEDS)),
        "control_key": control_key(W7, wave_uuid, leg, engine, stage, shard),
        "accepted_key": accepted_key(W7, wave_uuid, leg, engine, stage, shard),
        "modal": {"function": ("run_petri_proxy" if leg_spec["kind"] == "petri" else
                               "run_rung" if kind == "rung" else
                               "run_arditi_leg" if engine == ARDITI else "run_view_leg"),
                  "gpu": "H200", "cpu": PETRI_CPU if leg_spec["kind"] == "petri" else LEG_CPU,
                  "memory_mib": PETRI_MEMORY_MIB if leg_spec["kind"] == "petri" else LEG_MEMORY_MIB,
                  "timeout_s": timeout_s, "retries": 0,
                  "max_containers": int(max_containers)},
        "geometry": geom, "argv": argv, "route_set": routes, "route_set_harness": [harness_id(c) for c in routes],
        "cells_served_expected": cells_served_for_routes(routes), "objects": objects, "adapters": reuse,
        "readiness": readiness_block(geom, max_lora_rank=(1 if engine == ARDITI else VIEW_MAX_LORA_RANK), max_loras=(2 if engine == ARDITI else VIEW_MAX_LORAS),
                                     residual_slot_armed=([ARDITI_CELL] if engine == ARDITI else list(PSR_CELLS)),
                                     max_logprobs=leg_row.get("max_logprobs"), thinking=bool(window["thinking"])),
        "children": kids, "producers": producers, "ladder": (dict(ladder) if ladder else None),
        "rung_consumer": rung_consumer, "rung_mode": rung_mode, "rung_assembly": rung_assembly, "pool": pool_name,
        "arm_runs": arm_runs(engine),
        "expected_h200_h": [round(x / 4, 2) for x in leg_spec["h200_h"]],
    }
    if leg in MULTITURN_LEGS:
        row_out["window_provenance"] = multiturn_window_provenance(leg, geom["L"], s4 or S4_DEFAULTS)
        row_out["depends_on"] = depends_on_block(pool_name)
    return row_out


def build_plan(*, W: str, wave_uuid: str, reqs_sha256: str, roster: Mapping[str, Any], pins: Mapping[str, Any],
               arditi_pins: Mapping[str, Any], arms: Mapping[str, Any], s4: Optional[Mapping[str, Any]] = None,
               materialisation: str = "draft", adapter_hashes: Optional[Mapping[str, str]] = None,
               generated_at: Optional[str] = None, peer_roster: Optional[Mapping[str, Any]] = None,
               peer_pins: Optional[Mapping[str, Any]] = None, register_peer_dose_standups: bool = False,
               config_private_runs: Optional[Mapping[str, Optional[str]]] = None) -> Dict[str, Any]:
    """The whole plan. ``s4`` = the decisions block (None = :data:`S4_DEFAULTS`, the draft)."""
    if materialisation not in ("draft", "final"):
        raise ValueError("materialisation must be draft | final")
    if len(W) != 40:
        raise ValueError(f"W must be the full 40-hex sha, got {W!r}")
    check_wave_uuid(wave_uuid)
    s4 = copy.deepcopy(S4_DEFAULTS) if s4 is None else _merge_s4(s4)
    if materialisation == "draft" and s4 != S4_DEFAULTS:
        raise ValueError("a DRAFT plan carries the S4 fields at their spec defaults; pass --materialisation final with decisions")
    adapter_hashes = dict(adapter_hashes or {})
    W7 = W[:7]
    legs = leg_table(s4)
    pins_by_view = {v: view_pins(pins, roster_view_cells(roster, v)) for v in VIEWS}
    staged_by_view = {v: roster_view_cells(roster, v) for v in VIEWS}
    ar = (arditi_pins.get(MODEL_KEY) or {}).get(ARDITI_STAGED_ID) or {}
    arditi_sha = str(ar.get("object_sha256") or "")
    # configs/sdfbase_full22_arms.yaml: {wave, generator, arms, targets: {<model key>: {view, arditi, arditi_allpos, alphasteer}}}
    arm_row = ((((arms.get("targets") or {}).get(MODEL_KEY) or {}).get("arditi_allpos")) or {})
    if not arditi_sha or arm_row.get("object_sha256") != arditi_sha or arm_row.get("staged_id") != ARDITI_STAGED_ID:
        raise ValueError(f"Arditi pin disagreement: pins {arditi_sha[:12]!r} vs arms table {str(arm_row.get('object_sha256'))[:12]!r} "
                         f"(staged_id {arm_row.get('staged_id')!r}); the arm carries exactly ONE object")
    pools = pool_rows(s4, W7, wave_uuid)
    pools_by_attack = pools_for_attacks(pools)
    standups: List[Dict[str, Any]] = []
    rungs: List[Dict[str, Any]] = []
    deferred_rungs: List[Dict[str, Any]] = []
    for leg, spec in legs.items():
        ladder_kind = LADDER_KIND_OF_LEG.get(leg)
        for engine in ENGINES:
            n = int(spec["containers_arditi"] if engine == ARDITI else spec["containers_per_view"])
            base_shards = [str(k) for k in range(n)]
            ordered = sorted((c for c in spec["rungs"] if isinstance(c, int)))
            if STAGE_PETRI_RERUN in spec["rungs"]:
                ordered = ordered + [STAGE_PETRI_RERUN]
            first_stage = ordered[0] if ordered else None
            for k in range(n):
                producers, ladder = [], None
                if first_stage is not None:
                    producers.append({"queue_stage": str(first_stage), "kind": "base", "ladder_kind": ladder_kind, "shards": base_shards})
                    if ladder_kind is not None:
                        ladder = {"kind": ladder_kind, "cap_next": int(first_stage), "shard": str(k), "shards": base_shards,
                                  "feeder_shards": base_shards, "producer_of": str(first_stage)}
                # the k-th container of a multi-turn leg takes the k-th pool of the pools serving its attack,
                # round robin -- every deployed pool receives traffic (`setdefault` kept only the first)
                pool = None
                if leg in MULTITURN_LEGS:
                    pool = pools_by_attack[leg][k % len(pools_by_attack[leg])]
                standups.append(_row(leg_spec=spec, engine=engine, stage=STAGE_BASE, shard=k, nshards=n, window=spec["base"],
                                     W=W, W7=W7, wave_uuid=wave_uuid, reqs_sha256=reqs_sha256, pins_by_view=pins_by_view,
                                     staged_by_view=staged_by_view, arditi_sha256=arditi_sha,
                                     pool_url=(pool["url_placeholder"] if pool else None), kind="base", max_containers=1,
                                     producers=producers, adapter_hashes=adapter_hashes, ladder=ladder,
                                     pool_name=(pool["name"] if pool else None), s4=s4))
            for i, cap in enumerate(ordered):
                nxt = ordered[i + 1] if i + 1 < len(ordered) else None
                producers = [{"queue_stage": str(nxt), "kind": "rung", "ladder_kind": ladder_kind, "shards": ["0"]}] if nxt is not None else []
                feeder = base_shards if i == 0 else ["0"]
                ladder = None
                if ladder_kind is not None and isinstance(cap, int):
                    ladder = {"kind": ladder_kind, "cap_next": (int(nxt) if isinstance(nxt, int) else None), "shard": "0", "shards": ["0"],
                              "feeder_shards": feeder, "consumes": int(cap), "producer_of": (str(nxt) if nxt is not None else None)}
                rungs.append(_row(leg_spec=spec, engine=engine, stage=cap, shard=0, nshards=1, window=spec["rungs"][cap],
                                  W=W, W7=W7, wave_uuid=wave_uuid, reqs_sha256=reqs_sha256, pins_by_view=pins_by_view,
                                  staged_by_view=staged_by_view, arditi_sha256=arditi_sha, pool_url=None, kind="rung",
                                  max_containers=int(spec["rung_max_containers"]), producers=producers,
                                  adapter_hashes=adapter_hashes, ladder=ladder, s4=s4))
    dose_rows, dose_rec = dose_standup_rows(roster=roster, pins=pins, legs=legs, W=W, W7=W7, wave_uuid=wave_uuid, reqs_sha256=reqs_sha256,
                                            pools_by_attack=pools_by_attack, adapter_hashes=adapter_hashes, s4=s4, peer_roster=peer_roster, peer_pins=peer_pins,
                                            register=register_peer_dose_standups)
    probe_rows = ([] if register_peer_dose_standups else            # the dormant path's dose standups carry the probe cells themselves (same shape)
                  s5_probe_rows(peer_roster=peer_roster, peer_pins=peer_pins, legs=legs, W=W, W7=W7, wave_uuid=wave_uuid, reqs_sha256=reqs_sha256,
                                pools_by_attack=pools_by_attack, adapter_hashes=adapter_hashes, s4=s4))
    standups += dose_rows
    queues = queue_table(legs, standups, rungs)
    plan = {
        "schema": PLAN_SCHEMA, "materialisation": materialisation, "generated_at": generated_at or _utc(),
        "W": W, "W7": W7, "wave_uuid": wave_uuid, "vllm_requirements_sha256": reqs_sha256,
        "model_key": MODEL_KEY, "model": MODEL, "model_revision": MODEL_REVISION,
        "lane_root": LANE, "results_root": f"{LANE}/{W7}/exp/{RESULTS_ROOT_NAME}", "plans_root": f"{LANE}/{W7}/{PLANS_REL}",
        "ladder_root": f"{LANE}/{W7}/{wave_uuid}/ladder", "dicts": {p: dict_name(W7, wave_uuid, p) for p in DICT_PURPOSES},
        "s4_decisions": s4, "legs": {k: _leg_public(v) for k, v in legs.items()}, "eval_legs_of": EVAL_LEGS_OF,
        "standups": standups, "rungs": rungs, "deferred_rungs": deferred_rungs, "pools": pools, "queues": queues,
        "multiturn_window_provenance": {leg: multiturn_window_provenance(leg, int(legs[leg]["base"]["L"]), s4) for leg in MULTITURN_LEGS},
        "dose_standups": dose_rec, "s5_probe_rows": probe_rows, "s5_multiturn_probe": s5_multiturn_probe(s4, dose_rows + probe_rows),
        "bon_climb_decode": roster_bon_climb_decode(roster),
        "launch_order": launch_order(standups + rungs, pools),
        "s5_roster": s5_roster_rows(standups + rungs),
        "adapter_hash_set": adapter_hash_set(standups + rungs),
        "arm_runs_per_leg": 27, "eval_legs": sum(len(v) for v in EVAL_LEGS_OF.values()),
    }
    stamp_config_private_runs(plan, config_private_runs or {})
    plan["capacity_ledger"] = capacity_ledger(plan)
    plan["launch_plan_sha"] = plan_sha(plan)
    return plan


# --------------------------------------------------------------------------- #
# config private_run namespaces                                                 #
# --------------------------------------------------------------------------- #

def plan_config_rels() -> List[str]:
    """Every config a plan child may name under `--config`: the views' unsampled + sampled (base / rungs) configs, the Arditi shared,
    per-seed sampled (base / rungs) and per-seed WAVE configs."""
    rels: List[str] = []
    for view in VIEWS:
        rels.append(VIEW_CONFIG_REL.format(view=view))
        rels += [VIEW_SAMPLED_CONFIG_REL.format(view=view, suffix=sfx) for sfx in ("", "_cap8192", "_cap16384")]
    rels.append(ARDITI_CONFIG_REL)
    for s in ARDITI_SEEDS:
        rels += [ARDITI_CLIENT_CONFIG_REL.format(s=s, suffix=sfx) for sfx in ("", "_cap8192", "_cap16384")]
        rels.append(ARDITI_WAVE_CONFIG_REL.format(s=s))
    for view in VIEWS:                                            # the per-cell GSM8K overlays
        for cell in ("none",) + FITTED_CELLS + PROMPTED_CELLS:
            rels.append(GSM8K_OVERLAY_CONFIG_REL.format(model=MODEL_KEY, view=view, slug=gsm8k_cell_slug(cell)))
    return rels


def read_config_private_runs(roots: Sequence[Path], rels: Iterable[str]) -> Dict[str, Optional[str]]:
    """`{config rel: its private_run}` read off the FIRST root holding the file (the tree at W; a lane worktree as a fallback for a DRAFT on
    one lane's tree); None when no root holds it (recorded as such -- the S0 PLAN gate requires every named config at W)."""
    import yaml
    out: Dict[str, Optional[str]] = {}
    for rel in rels:
        out[rel] = None
        for root in roots:
            p = Path(root) / rel
            if p.is_file():
                doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
                pr = doc.get("private_run") if isinstance(doc, Mapping) else None
                out[rel] = str(pr) if pr else None
                break
    return out


def expected_private_run(rel: str) -> str:
    """The namespace the discovery expects a lane config to carry when the file is not readable here: its stem (every lane-rendered config
    names `private_run` == its file stem; only the verbatim shared Arditi copy differs, and that one is always present)."""
    return Path(rel).stem


def stamp_config_private_runs(plan: Dict[str, Any], config_private_runs: Mapping[str, Optional[str]]) -> None:
    """On every child naming `--config`: `config_rel`, `config_private_run` (read; None when the file was absent on this tree),
    `config_private_run_expected` (the stem) and `config_private_run_source`; the plan carries the table read."""
    seen: Dict[str, Optional[str]] = {}
    for r in list(plan["standups"]) + list(plan["rungs"]) + list(plan.get("s5_probe_rows") or []):
        for k in r["children"]:
            rel = k.get("config") or (_flag_value(k["argv"], "--config") if k.get("argv") else None)
            if not rel:
                continue
            got = config_private_runs.get(rel)
            k["config_rel"] = rel
            k["config_private_run"] = got
            k["config_private_run_expected"] = expected_private_run(rel)
            k["config_private_run_source"] = "read" if got else "expected_from_stem (config absent on this tree; the S0 PLAN gate requires it at W)"
            seen[rel] = got
    plan["config_private_runs"] = dict(sorted(seen.items()))


def child_private_run(k: Mapping[str, Any]) -> Optional[str]:
    return k.get("config_private_run") or k.get("config_private_run_expected")


def private_run_problems(plan: Mapping[str, Any]) -> List[str]:
    """A WAVE-leg child (BoN, multi-turn) whose config `private_run` is any STATIC-base run is filed as static by
    the discovery -- refused; the Arditi wave children of one standup must carry DISTINCT private_runs (one shared namespace is assigned
    to the first seed only) -- refused; an Arditi wave child must name its per-seed WAVE config."""
    rows = list(plan["standups"]) + list(plan["rungs"])
    static_runs = {child_private_run(k) for r in rows for k in r["children"] if k["eval_leg"] in STATIC_LEGS and child_private_run(k)}
    problems: List[str] = []
    for r in rows:
        kids = [k for k in r["children"] if k["eval_leg"] in WAVE_LEGS and k.get("argv")]
        for k in kids:
            run = child_private_run(k)
            if not run:
                problems.append(f"{r['standup_id']}: {k['eval_leg']} child {k['arm']} s{k['s']} names no config private_run ({k.get('config_rel')!r})")
            elif run in static_runs:
                problems.append(f"{r['standup_id']}: {k['eval_leg']} child {k['arm']} s{k['s']} config {k.get('config_rel')} private_run {run!r} is a STATIC-base "
                                "run -- WS5-I's discovery files its store as static (WS5-I r3 finding 3); use the WAVE config")
            if r["engine"] == ARDITI and k.get("config_rel") != ARDITI_WAVE_CONFIG_REL.format(s=k["s"]):
                problems.append(f"{r['standup_id']}: Arditi {k['eval_leg']} child s{k['s']} must use the per-seed wave config "
                                f"{ARDITI_WAVE_CONFIG_REL.format(s=k['s'])} (got {k.get('config_rel')!r}; WS5-I r3 finding 3)")
        if r["engine"] == ARDITI and kids:
            runs = [child_private_run(k) for k in kids]
            if len(set(runs)) != len(runs):
                problems.append(f"{r['standup_id']}: the Arditi {kids[0]['eval_leg']} children share a config private_run {runs} -- discovery assigns "
                                "one namespace to the first seed only (WS5-I r3 finding 3); three seeds need three private_runs")
    return problems


def config_files_problems(plan: Mapping[str, Any], tree: Path) -> List[str]:
    """Every config a child names must exist under the tree at W (the S0 PLAN gate; a DRAFT on one lane's tree may lack the peers' files)."""
    missing = sorted({k["config_rel"] for r in list(plan["standups"]) + list(plan["rungs"]) for k in r["children"]
                      if k.get("config_rel") and not (Path(tree) / k["config_rel"]).is_file()})
    return [f"config named by a child is absent under the tree: {rel}" for rel in missing]


def plan_children(plan: Mapping[str, Any], *, legs: Optional[Sequence[str]] = None, engines: Optional[Sequence[str]] = None) -> List[Tuple[str, Dict[str, Any]]]:
    """`[(standup_id, child)]` of the plan's GENERATED children -- the plan-time argv WITH ITS PLACEHOLDERS (`{attempt_root}`, `{proxy_url}`,
    `{routing_table}`, `{driver_python}`, `{concurrency}`, `{bank_view}`, `{pool_url:<pool>}`) and the child's own env fragment; what the
    grader twin DISCOVERS configs / arms / tasks from (import this module by path, `load_inputs` + `build_plan`, then this). It is NOT what the
    app hands to `Popen`: that is :func:`resolve_spawn` over a :class:`SpawnContext` -- the SAME function the app calls for every child (a
    driver child and the rung consumer alike), which resolves every placeholder, sets the per-child registry identity, the attempt-unique
    spool and the request-evidence variables, and refuses an argv with a placeholder left (the two views are named apart)."""
    out = []
    for r in list(plan["standups"]) + list(plan["rungs"]):
        if engines is not None and r["engine"] not in engines:
            continue
        for k in r["children"]:
            if legs is None or k["eval_leg"] in legs:
                out.append((r["standup_id"], k))
    return out


# --------------------------------------------------------------------------- #
# the spawn resolution -- ONE function from a generated child to the FINAL argv + env the app hands to Popen                           #
# --------------------------------------------------------------------------- #

#: every placeholder a generated argv may carry, resolved by :func:`resolve_spawn` from the :class:`SpawnContext`; `{pool_url:<pool>}` is
#: resolved from the ctx's pool URLs; anything else shaped like a placeholder that survives resolution is a refusal, never a Popen argument
#: PER-CHILD runtime values: the child's usage ledger URL, its own routing-table copy (the ledger record lives beside it)
#: and its store's attempt id -- filled from `SpawnContext.child_subs[child_key(child)]` by the container that started that child's ledger
LEDGER_URL_PLACEHOLDER = "{ledger_url}"
CHILD_ROUTING_TABLE_PLACEHOLDER = "{child_routing_table}"
GSM8K_ATTEMPT_ID_PLACEHOLDER = "{gsm8k_attempt_id}"
SPAWN_PLACEHOLDERS: Tuple[str, ...] = (ATTEMPT_ROOT_PLACEHOLDER, PROXY_URL_PLACEHOLDER, ROUTING_TABLE_PLACEHOLDER, DRIVER_PY_PLACEHOLDER,
                                       CONCURRENCY_PLACEHOLDER, BANK_VIEW_PLACEHOLDER, ATTEMPT_ID_PLACEHOLDER, CALL_ID_PLACEHOLDER, ATTEMPT_ROOT_REL_PLACEHOLDER,
                                       LEDGER_URL_PLACEHOLDER, CHILD_ROUTING_TABLE_PLACEHOLDER, GSM8K_ATTEMPT_ID_PLACEHOLDER, CLIENT_PY_PLACEHOLDER)


def child_key(child: Mapping[str, Any]) -> str:
    """ONE spelling of a child's identity across the app (`<eval_leg>__<arm>__s<s>`, the log-name stem's source)."""
    return f"{child['eval_leg']}__{child['arm']}__s{child['s']}"
_PLACEHOLDER_RE = re.compile(r"\{[a-z_]+(?::[^{}]*)?\}")


@dataclasses.dataclass(frozen=True)
class SpawnContext:
    """Everything the app knows at spawn time that a generated child does not: the attempt's root / id, the standup, the identity plans root,
    the live proxy URL + routing table, the driver venv python, the ATTESTED S, the route count, the bank view, the container env, the pool
    URLs, the Modal call id and the attempt root's volume-relative spelling (the rung consumer's `--attempt-root`)."""

    attempt_root: str
    attempt_id: str
    standup_id: str
    plans_root: str
    proxy_url: str
    routing_table: str
    driver_python: str
    S: int
    ncells: int
    bank_view: str
    base_env: Mapping[str, str]
    pool_urls: Optional[Mapping[str, str]] = None
    call_id: str = ""
    attempt_root_rel: str = ""
    #: per-child runtime substitutions keyed by :func:`child_key` (the GSM8K ledger URL / routing-table copy / attempt id)
    child_subs: Optional[Mapping[str, Mapping[str, str]]] = None
    #: the tinycap CLIENT venv python (inspect_ai + inspect_evals) -- required by every `inspect/` child; "" = not built
    client_python: str = ""


def concurrency_for(rule: str, S: int, ncells: int) -> int:
    """The per-child concurrency from the ATTESTED S (§4.4): one-request-each drivers = S; per-arm drivers S / NCELLS with the sbatch's floors
    (4 for BoN, 8 for the controls); multi-turn ~2 drivers per served slot handled by the driver count."""
    S, ncells = int(S), max(1, int(ncells))
    if rule in ("S", "one_request_each_S_drivers", "two_drivers_per_slot"):
        return S
    if rule == "S_over_ncells_floor_4":
        return max(4, S // ncells)
    if rule in ("S_over_ncells_floor_8",):
        return max(8, S // ncells)
    if rule == "S_over_ncells":
        return max(1, S // ncells)
    if rule == "grading_host_drive":
        return 0
    raise ValueError(f"unknown concurrency rule {rule!r}")


def unresolved_placeholders(argv: Sequence[str]) -> List[str]:
    return [m.group(0) for a in argv for m in _PLACEHOLDER_RE.finditer(str(a))]


def resolve_spawn(child: Mapping[str, Any], ctx: SpawnContext) -> Tuple[List[str], Dict[str, str]]:
    """THE spawn resolution: a GENERATED child (:func:`plan_children`; the rung consumer child with its raw consumer argv) ->
    the FINAL ``(argv, env)`` the app hands to `Popen`. Every placeholder of :data:`SPAWN_PLACEHOLDERS` and `{pool_url:<pool>}` is resolved
    from ``ctx``; a placeholder left over refuses. The env = the container env + the child's own fragment + (a driver child, one with a
    registry identity) the per-child identity (`SISTEER_REGISTRY_RUN_IDENTITY`), the attempt-unique spool (`SISTEER_REGISTRY_SPOOL`) and
    the three request-evidence variables (evidence dir == the child's store root); a rung consumer child arrives with its evidence
    env already built (`rung_child_env`) and keeps it. The app calls THIS for every child; discovery runs from `plan_children`."""
    conc = concurrency_for(child["concurrency_rule"], ctx.S, ctx.ncells)
    needs_client = child_needs_client_python(child) or any(CLIENT_PY_PLACEHOLDER in str(a) for a in (child.get("argv") or []))
    if needs_client and not ctx.client_python:
        raise ValueError(f"child {child.get('arm')} s{child.get('s')} ({child.get('task')}) runs under the tinycap CLIENT venv ({CLIENT_PY_PLACEHOLDER}: "
                         "inspect_ai / inspect_evals) but the spawn context carries no client python -- the driver venv is not a stand-in (launch F-I)")
    subs = {ATTEMPT_ROOT_PLACEHOLDER: str(ctx.attempt_root), PROXY_URL_PLACEHOLDER: str(ctx.proxy_url), ROUTING_TABLE_PLACEHOLDER: str(ctx.routing_table),
            DRIVER_PY_PLACEHOLDER: str(ctx.driver_python), CONCURRENCY_PLACEHOLDER: str(conc), BANK_VIEW_PLACEHOLDER: str(ctx.bank_view),
            CLIENT_PY_PLACEHOLDER: str(ctx.client_python or ""),
            ATTEMPT_ID_PLACEHOLDER: str(ctx.attempt_id), CALL_ID_PLACEHOLDER: str(ctx.call_id or ctx.attempt_id), ATTEMPT_ROOT_REL_PLACEHOLDER: str(ctx.attempt_root_rel)}
    for k, v in ((ctx.child_subs or {}).get(child_key(child)) or {}).items():
        if k not in SPAWN_PLACEHOLDERS:
            raise ValueError(f"child {child_key(child)}: per-child substitution {k!r} is not a spawn placeholder")
        subs[k] = str(v)
    argv: List[str] = []
    for a in child.get("argv") or []:
        a = str(a)
        if a.startswith("{pool_url"):
            name = a[len("{pool_url:"):-1] if a.startswith("{pool_url:") else None
            if not ctx.pool_urls or (name and name not in ctx.pool_urls):
                raise ValueError(f"child {child.get('arm')} s{child.get('s')} needs pool {name!r} but no pool URL was recorded")
            a = ctx.pool_urls[name] if name else next(iter(ctx.pool_urls.values()))
        for k, v in subs.items():
            a = a.replace(k, v)
        argv.append(a)
    left = unresolved_placeholders(argv)
    if left:
        raise ValueError(f"child {child.get('arm')} s{child.get('s')}: placeholder(s) {sorted(set(left))} survived resolution; a placeholder never reaches Popen")
    if child_needs_client_python(child) and argv and ctx.driver_python and argv[0] == str(ctx.driver_python):
        # the earlier resolution: an Inspect-side child under the DRIVER venv died at `import inspect_ai` -- refused by name
        raise ValueError(f"child {child.get('arm')} s{child.get('s')} ({child.get('task')}) resolved to the DRIVER python {argv[0]}: an Inspect-side "
                         f"child runs under the tinycap CLIENT venv ({CLIENT_PY_PLACEHOLDER}); the driver venv carries no inspect_ai (F-I)")
    env = dict(ctx.base_env)
    env.update(child.get("env") or {})
    if child.get("registry_identity_rel"):
        env.update(child_registry_env(plans_root=ctx.plans_root, registry_identity_rel=child["registry_identity_rel"], attempt_root=str(ctx.attempt_root)))
        # the three request-evidence variables -- the child's STORE ROOT as the evidence dir, the attempt-bound extra, the seed base.
        # The extra's attempt id is the RESOLVED argv's `--attempt-id` when the child carries one (a GSM8K child's per-store `a<k>`, the
        # same substitution its ledger and the driver's writer run as), else the container's -- one run, one attempt id (`evidence_attempt_id`)
        if child.get("argv"):
            env.update(request_evidence_env(evidence_dir=child_store_root(child, argv), attempt_id=evidence_attempt_id(argv, ctx.attempt_id),
                                            standup_id=ctx.standup_id, eval_leg=str(child["eval_leg"]), arm=str(child["arm"]), s=int(child["s"])))
    else:
        env.setdefault(SPOOL_ENV, f"{ctx.attempt_root}/{SPOOL_REL}")
        env.setdefault(SUBSTRATE_ENV, SUBSTRATE)
    return argv, env


# --------------------------------------------------------------------------- #
# Overnight operation                                                         #
# --------------------------------------------------------------------------- #

#: an INFRA-class failure of a standup (a preempted / lost container, an engine that died, a transport death) gets exactly ONE
#: automatic re-attempt under the existing attempt rules (a new immutable attempt root; acceptance only via the throwing barrier); a
#: second failure, or ANY correctness refusal (a fidelity gate, a readiness attestation, DRAFT != FINAL, the request-evidence
#: reconciliation, the audit, a failed driver, an unknown status -- fail closed), tears the standup down and HOLDS it while the rest
#: of the wave continues. The counter is per standup and DURABLE (the wave's `reattempts` Dict), never in memory.
D73_MAX_AUTO_REATTEMPTS = 1
D73_INFRA_STATUSES: Tuple[str, ...] = ("LOST_CONTAINER", "PREEMPTED", "ENGINE_DIED", "TRANSPORT_DEATH", "CALL_LOST", "OUTPUT_EXPIRED_NO_RESULT",
                                       "POST_RESULT_TIMEOUT")    # fo3: the app's post-RESULT tail timed out before the acceptance (seedrep_served.POST_RESULT_TIMEOUT)
D73_INFRA_ERROR_TOKENS: Tuple[str, ...] = ("preempt", "container was lost", "lost container", "worker was lost", "worker lost", "engine died",
                                          "connection reset", "transport", "remote end closed", "the attempt left no verdict")
#: no container waits past its deadline -- the first failed child tears its siblings down (`--teardown-after-failure` semantics on
#: every child; the Petri proxy ends on ANY non-waiting completion and the grading host drive runs `--teardown-after-failure`)
D73_TEARDOWN_AFTER_FAILURE = True
#: the three summary events the watcher emits, ONE machine-readable line each (`<EVENT> <json>`), pushed to the operator's phone
D73_SUMMARY_EVENTS: Tuple[str, ...] = ("WAVE_LAUNCHED", "WAVE_DONE", "HOLD")
#: the utilisation bars a WAVE_LAUNCHED line carries (the utilisation table measures them)
UTILISATION_BARS: Dict[str, Any] = {"running_over_max_num_seqs_min": 0.8, "sm_duty_min": 0.85, "aggregate_tok_s_over_s5_min": 0.8, "h200_hours_over_sized_max": 1.5}
#: the COMPLETE utilisation verdict: `bars_met` alone is a claim, so the verdict carries every bar's reading against its
#: threshold, the table it was read from by path + sha256, the attempts it covers, the tree and the smoke's execution wave
#: `unmeasured` is a FIRST-CLASS bar outcome, never a number -- a /2 verdict's bar entry may take the explicit unmeasured
#: form `{value: null, threshold, met: false, unmeasured: true, why: "<the table's exact reason>"}`; /1 records (numeric bars only) validate unchanged
UTILISATION_VERDICT_SCHEMA = "seedrep_utilisation_verdict/2"
UTILISATION_VERDICT_SCHEMAS: Tuple[str, ...] = ("seedrep_utilisation_verdict/1", UTILISATION_VERDICT_SCHEMA)
UTILISATION_VERDICT_KEYS: Tuple[str, ...] = ("schema", "bars_met", "bars", "source", "source_sha256", "W", "execution_wave_uuid", "attempt_ids", "measured_utc")
UTILISATION_BAR_KEYS: Tuple[str, ...] = ("value", "threshold", "met")
UTILISATION_BAR_UNMEASURED_KEYS: Tuple[str, ...] = ("value", "threshold", "met", "unmeasured", "why")
UTILISATION_UNMEASURED = "unmeasured"
#: the utilisation table's cell per bar (`utilisation_table.COLUMNS`): the reading column, its verdict column (carrying `unmeasured(<why>)`), and
#: the map from the cell to the bar's ratio (Running / attested S; SM % -> fraction; the two ratios as written)
UTILISATION_TABLE_CELLS: Dict[str, Tuple[str, str]] = {"running_over_max_num_seqs_min": ("median_running_saturated", "running_bar"), "sm_duty_min": ("median_sm_decode", "sm_bar"),
                                                        "aggregate_tok_s_over_s5_min": ("tok_s_ratio_s5", "tok_bar"), "h200_hours_over_sized_max": ("h200_h_ratio", "hours_bar")}


def _unmeasured_why(text: Any) -> Optional[str]:
    """`unmeasured(<why>)` (the table's bar column) -> `<why>`; None when ``text`` is not that form."""
    t = str(text or "").strip()
    if t.startswith(UTILISATION_UNMEASURED + "(") and t.endswith(")"):
        why = t[len(UTILISATION_UNMEASURED) + 1:-1].strip()
        return why or None
    return None


def utilisation_reading(row: Mapping[str, Any], key: str) -> Any:
    """ONE utilisation-table row's reading for bar ``key`` -- a JSON row or a CSV row (the CSV's cells are strings; both paths run through here):
    the UNMEASURED form `{"unmeasured": True, "why": "<leg>/<view> <attempt>: <the table's reason>"}` whenever the row's BAR column says
    `unmeasured(<why>)` (fu4: checked first, whatever the reading cell holds -- a diagnostic number beside an unmeasured bar is not a reading) or
    the reading cell is itself `unmeasured` / absent (the reason then from the row's `problems`); else a NUMBER (Running median / the row's
    attested S; SM % / 100; the tok/s and hours ratios as the table writes them)."""
    col, bar_col = UTILISATION_TABLE_CELLS[key]
    v = row.get(col)
    who = f"{row.get('leg')}/{row.get('view')} {row.get('attempt_id')}"
    # fu4: the row's BAR column speaks FIRST -- `unmeasured(<why>)` makes the reading unmeasured whatever the reading cell holds (the table
    # keeps a diagnostic number beside an unmeasured bar: the tok/s ratio under insufficient sampling coverage; that number never becomes the
    # verdict's reading), then a reading cell that is itself `unmeasured` / absent
    why = _unmeasured_why(row.get(bar_col))
    if why is not None:
        return {"unmeasured": True, "why": f"{who}: {why}"}
    if isinstance(v, str) or v is None:
        if v is None or str(v).strip().lower() == UTILISATION_UNMEASURED:
            why = "; ".join(str(p) for p in (row.get("problems") or [])) or f"{col} is {v!r}"
            return {"unmeasured": True, "why": f"{who}: {why}"}
        try:
            v = float(v)
        except ValueError:
            raise ValueError(f"{who}: {col} = {v!r} is neither a number nor `unmeasured`") from None
    v = float(v)
    if key == "running_over_max_num_seqs_min":
        S = float(row.get("S") or 0)
        if S <= 0:
            raise ValueError(f"{who}: S {row.get('S')!r} is not a positive number -- the Running ratio cannot be formed")
        return v / S
    if key == "sm_duty_min":
        return v / 100.0
    return v


def utilisation_readings_from_table(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """The four readings of a verdict from the utilisation table's rows (the builder's mapping, fail-closed): per bar the weakest row --
    the MIN of the `_min` ratios, the MAX of the `_max` ratio -- and, when ANY row's cell is `unmeasured`, the UNMEASURED form whose why names
    every such row with the table's reason (an unmeasured bar on one standup is an unmeasured bar of the verdict)."""
    if not rows:
        raise ValueError("no utilisation rows")
    out: Dict[str, Any] = {}
    for key in UTILISATION_BARS:
        readings = [utilisation_reading(r, key) for r in rows]
        unm = [r for r in readings if isinstance(r, Mapping)]
        if unm:
            out[key] = {"unmeasured": True, "why": "; ".join(str(u["why"]) for u in unm)}
        else:
            out[key] = max(readings) if key.endswith("_max") else min(readings)
    return out


def utilisation_verdict_problems(rec: Any, *, W: Optional[str] = None, execution_wave_uuid: Optional[str] = None) -> List[str]:
    """Why ``rec`` is not a complete utilisation verdict (empty = it is): exactly :data:`UTILISATION_VERDICT_KEYS`; `bars` = every key of
    :data:`UTILISATION_BARS` with `{value, threshold, met}` where the threshold IS that bar's and `met` follows from value vs threshold
    (`_min` bars: value >= threshold; `_max` bars: value <= threshold); `bars_met` == all met; a table path + 64-hex sha; the S5 attempts;
    W / the S5 smoke's execution wave when given. A hand / mirrored source is no measurement."""
    if not isinstance(rec, Mapping):
        return [f"not an object ({type(rec).__name__})"]
    out: List[str] = []
    missing = [k for k in UTILISATION_VERDICT_KEYS if k not in rec]
    extra = [k for k in rec if k not in UTILISATION_VERDICT_KEYS]
    if missing:
        out.append(f"lacks {missing}")
    if extra:
        out.append(f"carries keys outside the verdict schema {extra}")
    if rec.get("schema") not in UTILISATION_VERDICT_SCHEMAS:
        out.append(f"schema {rec.get('schema')!r} not in {list(UTILISATION_VERDICT_SCHEMAS)}")
    v2 = rec.get("schema") == UTILISATION_VERDICT_SCHEMA
    bars = rec.get("bars")
    if not isinstance(bars, Mapping):
        out.append("bars is not a mapping of every plan §8 U bar")
        bars = {}
    mets: List[bool] = []
    for key, threshold in UTILISATION_BARS.items():
        b = bars.get(key)
        if not isinstance(b, Mapping):
            out.append(f"bars[{key!r}] is not {{value, threshold, met}}")
            continue
        # fu3: the explicit UNMEASURED form -- value null AND met false AND unmeasured true AND a non-empty why, a /2 record's only
        if "unmeasured" in b or "why" in b or ("value" in b and b["value"] is None):
            if not v2:
                out.append(f"bars[{key!r}]: the unmeasured form belongs to a {UTILISATION_VERDICT_SCHEMA} record, not {rec.get('schema')!r}")
                continue
            missing = [k for k in UTILISATION_BAR_UNMEASURED_KEYS if k not in b]
            extra = [k for k in b if k not in UTILISATION_BAR_UNMEASURED_KEYS]
            if missing or extra:
                out.append(f"bars[{key!r}] unmeasured form is not {{value, threshold, met, unmeasured, why}} (lacks {missing}, extra {extra})")
                continue
            if b["unmeasured"] is not True:
                out.append(f"bars[{key!r}].value is null without unmeasured true ({b['unmeasured']!r}) -- an unmeasured bar says so by name")
            if b["value"] is not None:
                out.append(f"bars[{key!r}].value {b['value']!r} is a number while unmeasured is true -- an unmeasured bar has no value")
            if b["met"] is not False:
                out.append(f"bars[{key!r}].met {b['met']!r} while unmeasured -- an unmeasured bar is never met")
            if not isinstance(b["why"], str) or not b["why"].strip():
                out.append(f"bars[{key!r}].why is empty -- the table's reason is required")
            if b["threshold"] != threshold:
                out.append(f"bars[{key!r}].threshold {b['threshold']!r} != plan §8 U's {threshold}")
            mets.append(False)
            continue
        if [k for k in UTILISATION_BAR_KEYS if k not in b]:
            out.append(f"bars[{key!r}] is not {{value, threshold, met}}")
            continue
        if not isinstance(b["value"], (int, float)) or isinstance(b["value"], bool):
            out.append(f"bars[{key!r}].value {b['value']!r} is not a number")
            continue
        if b["threshold"] != threshold:
            out.append(f"bars[{key!r}].threshold {b['threshold']!r} != plan §8 U's {threshold}")
        want = (float(b["value"]) <= threshold) if key.endswith("_max") else (float(b["value"]) >= threshold)
        if b["met"] is not want:
            out.append(f"bars[{key!r}].met {b['met']!r} disagrees with value {b['value']} vs threshold {threshold} (want {want})")
        mets.append(bool(b["met"]))
    unknown = [k for k in bars if k not in UTILISATION_BARS]
    if unknown:
        out.append(f"bars names bars outside plan §8 U: {unknown}")
    if not isinstance(rec.get("bars_met"), bool):
        out.append(f"bars_met {rec.get('bars_met')!r} is not a bool")
    elif not out and rec["bars_met"] is not all(mets):
        out.append(f"bars_met {rec['bars_met']!r} disagrees with the bars ({mets})")
    src = rec.get("source")
    if not isinstance(src, str) or not src.strip() or src.strip().lower() in S5_REFERENCE_FORBIDDEN_SOURCES:
        out.append(f"source {src!r} names no utilisation table (a hand / mirrored value is no measurement)")
    sha = rec.get("source_sha256")
    if not isinstance(sha, str) or len(sha) != 64:
        out.append(f"source_sha256 {sha!r} is not the table's 64-hex sha")
    ids = rec.get("attempt_ids")
    if not isinstance(ids, list) or not ids or not all(isinstance(x, str) and x for x in ids):
        out.append("attempt_ids must name the S5 attempt(s) the table was read from")
    if W is not None and rec.get("W") != W:
        out.append(f"W {rec.get('W')!r} is not the tree {W}")
    if execution_wave_uuid is not None and rec.get("execution_wave_uuid") != execution_wave_uuid:
        out.append(f"execution_wave_uuid {rec.get('execution_wave_uuid')!r} is not the S5 smoke's {execution_wave_uuid}")
    if not isinstance(rec.get("measured_utc"), str) or not rec.get("measured_utc"):
        out.append("measured_utc missing")
    return out


def utilisation_verdict(*, bars: Mapping[str, float], source: str, source_sha256: str, W: str, execution_wave_uuid: str, attempt_ids: Sequence[str],
                        measured_utc: Optional[str] = None) -> Dict[str, Any]:
    """The verdict from the four READINGS (`{bar key: reading}`): a NUMBER -> `{value, threshold, met}` with `met` derived; an UNMEASURED
    reading (fu3: the mapping `{"unmeasured": True, "why": ...}` :func:`utilisation_reading` gives, or the table's bar text `unmeasured(<why>)`)
    -> the explicit unmeasured form `{value: null, threshold, met: false, unmeasured: true, why}`; a bare `unmeasured` without its reason, or
    any other string, refuses. Thresholds from :data:`UTILISATION_BARS`; `bars_met` = the conjunction (an unmeasured bar -> false); refuses
    what :func:`utilisation_verdict_problems` would."""
    table = {}
    for key, threshold in UTILISATION_BARS.items():
        if key not in bars:
            raise ValueError(f"no reading for bar {key!r}")
        r = bars[key]
        why: Optional[str] = None
        if isinstance(r, Mapping):
            if r.get("unmeasured") is not True:
                raise ValueError(f"bar {key!r}: a mapping reading must be the unmeasured form {{unmeasured: true, why}}, got {dict(r)!r}")
            why = str(r.get("why") or "").strip()
        elif isinstance(r, str):
            why = _unmeasured_why(r)
            if why is None:
                if r.strip().lower() == UTILISATION_UNMEASURED:
                    raise ValueError(f"bar {key!r}: an unmeasured reading needs the table's reason (`unmeasured(<why>)` or {{unmeasured: true, why}})")
                raise ValueError(f"bar {key!r}: reading {r!r} is neither a number nor `unmeasured(<why>)`")
        if why is not None:
            if not why:
                raise ValueError(f"bar {key!r}: an unmeasured reading needs a non-empty reason")
            table[key] = {"value": None, "threshold": threshold, "met": False, "unmeasured": True, "why": why}
            continue
        v = float(r)
        table[key] = {"value": v, "threshold": threshold, "met": (v <= threshold) if key.endswith("_max") else (v >= threshold)}
    rec = {"schema": UTILISATION_VERDICT_SCHEMA, "bars_met": all(b["met"] for b in table.values()), "bars": table, "source": str(source),
           "source_sha256": str(source_sha256), "W": str(W), "execution_wave_uuid": str(execution_wave_uuid), "attempt_ids": [str(a) for a in attempt_ids],
           "measured_utc": measured_utc or _utc()}
    problems = utilisation_verdict_problems(rec, W=W, execution_wave_uuid=execution_wave_uuid)
    if problems:
        raise ValueError(f"not a utilisation verdict: {problems}")
    return rec


#: the utilisation bars are an OPERATIONAL-EFFICIENCY gate, not a scientific criterion of the rows -- for this launch the certificate
#: ADMITS an unmet verdict ONLY through an explicit, readings-bearing WAIVER derived from the recorded verdict (`stamp_utilisation_waiver.py`;
#: never hand-typed readings): its schema, the ruling label, the tree, the sha256 of the canonical JSON of the verdict it waives, and `unmet` =
#: the EXACT list of unmet bars with their readings copied from that verdict -- the certificate re-derives the list and requires equality. A
#: waiver present while `bars_met` is true refuses (nothing to waive); no other path admits unmet bars.
UTILISATION_WAIVER_SCHEMA = "seedrep_utilisation_waiver/1"
UTILISATION_WAIVER_FILE = "utilisation_waiver.json"
UTILISATION_WAIVER_RULING = "D198"
UTILISATION_WAIVER_KEYS: Tuple[str, ...] = ("schema", "by", "W", "bars_sha256", "unmet", "stamped_utc")


def utilisation_unmet(bars: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The unmet bars of a utilisation verdict, in `UTILISATION_BARS` order, each with its reading copied from
    the verdict: `[{bar, value, threshold, met: false}, ...]` -- what a waiver must name exactly."""
    table = bars.get("bars") if isinstance(bars, Mapping) and isinstance(bars.get("bars"), Mapping) else {}
    out: List[Dict[str, Any]] = []
    for key in UTILISATION_BARS:
        b = table.get(key)
        if isinstance(b, Mapping) and b.get("met") is not True:
            keys = UTILISATION_BAR_UNMEASURED_KEYS if b.get("unmeasured") is True else UTILISATION_BAR_KEYS   # fu3: an unmeasured bar carries its why
            out.append({"bar": key, **{k: b.get(k) for k in keys}})
    return out


def utilisation_waiver_problems(waiver: Any, *, bars: Any, W: str, execution_wave_uuid: Optional[str] = None) -> List[str]:
    """Why ``waiver`` does not admit the utilisation verdict ``bars`` for tree ``W`` (empty = it does): the verdict itself must be complete
    (:func:`utilisation_verdict_problems`) and UNMET (a waiver present while bars_met is true refuses); the waiver exactly
    :data:`UTILISATION_WAIVER_KEYS`; schema / ruling label / W as pinned; `bars_sha256` == the canonical sha256 of the verdict; `unmet` ==
    :func:`utilisation_unmet` of the verdict (every unmet bar, its reading, no extra, the order); a stamp."""
    out: List[str] = []
    vp = utilisation_verdict_problems(bars, W=W, execution_wave_uuid=execution_wave_uuid)
    if vp:
        out.append(f"the verdict the waiver would admit is not a utilisation verdict: {vp[:3]}")
    elif bars.get("bars_met") is True:
        out.append("a utilisation waiver is present while bars_met is true -- nothing to waive; refusing")
    if not isinstance(waiver, Mapping):
        return out + [f"not an object ({type(waiver).__name__})"]
    missing = [k for k in UTILISATION_WAIVER_KEYS if k not in waiver]
    extra = [k for k in waiver if k not in UTILISATION_WAIVER_KEYS]
    if missing:
        out.append(f"lacks {missing}")
    if extra:
        out.append(f"carries keys outside the waiver schema {extra}")
    if waiver.get("schema") != UTILISATION_WAIVER_SCHEMA:
        out.append(f"schema {waiver.get('schema')!r} != {UTILISATION_WAIVER_SCHEMA}")
    if waiver.get("by") != UTILISATION_WAIVER_RULING:
        out.append(f"by {waiver.get('by')!r} is not the ruling {UTILISATION_WAIVER_RULING}")
    if waiver.get("W") != W:
        out.append(f"W {waiver.get('W')!r} is not the tree {W}")
    if not vp:
        want_sha = _canonical_sha256(bars)
        if waiver.get("bars_sha256") != want_sha:
            out.append(f"bars_sha256 {waiver.get('bars_sha256')!r} is not the recorded verdict's {want_sha} -- the waiver does not name this verdict")
        want_unmet = utilisation_unmet(bars)
        if waiver.get("unmet") != want_unmet:
            out.append(f"unmet {json.dumps(waiver.get('unmet'), default=str)[:200]} is not the verdict's unmet bars with their readings "
                       f"{json.dumps(want_unmet, default=str)[:200]}")
    if not isinstance(waiver.get("stamped_utc"), str) or not waiver.get("stamped_utc"):
        out.append("stamped_utc missing")
    return out


def build_utilisation_waiver(bars: Mapping[str, Any], *, W: str, execution_wave_uuid: Optional[str] = None, ruling: str = UTILISATION_WAIVER_RULING,
                             stamped_utc: Optional[str] = None) -> Dict[str, Any]:
    """The waiver DERIVED from the recorded verdict (its unmet bars and readings copied, its canonical sha bound); refuses a verdict that
    is not one, a met verdict (nothing to waive) and any ruling label but :data:`UTILISATION_WAIVER_RULING`."""
    if ruling != UTILISATION_WAIVER_RULING:
        raise ValueError(f"ruling {ruling!r} is not {UTILISATION_WAIVER_RULING}; no other ruling admits unmet utilisation bars")
    vp = utilisation_verdict_problems(bars, W=W, execution_wave_uuid=execution_wave_uuid)
    if vp:
        raise ValueError(f"not a utilisation verdict: {vp}")
    if bars.get("bars_met") is True:
        raise ValueError("bars_met is true -- nothing to waive; refusing to stamp a waiver")
    rec = {"schema": UTILISATION_WAIVER_SCHEMA, "by": ruling, "W": str(W), "bars_sha256": _canonical_sha256(bars), "unmet": utilisation_unmet(bars),
           "stamped_utc": stamped_utc or _utc()}
    problems = utilisation_waiver_problems(rec, bars=bars, W=W, execution_wave_uuid=execution_wave_uuid)
    if problems:
        raise ValueError(f"the derived waiver does not validate: {problems}")
    return rec


def read_utilisation_waiver(rs: Path) -> Optional[Dict[str, Any]]:
    """`RS/utilisation_waiver.json` (None when absent; a non-object refuses by name)."""
    p = Path(rs) / UTILISATION_WAIVER_FILE
    if not p.is_file():
        return None
    doc = json.loads(p.read_text())
    if not isinstance(doc, Mapping):
        raise ValueError(f"{p}: not an object")
    return dict(doc)


def failure_class(status: Optional[str], *, error: Optional[str] = None) -> str:
    """`infra` for the infrastructure class -- a NAMED infra status, or a Modal-side error text naming a preemption / lost container /
    transport death; `correctness` for EVERYTHING else (fail closed: an unknown status never earns a re-attempt)."""
    s = str(status or "").upper()
    if s in D73_INFRA_STATUSES:
        return "infra"
    text = str(error or "").lower()
    if text and any(tok in text for tok in D73_INFRA_ERROR_TOKENS):
        return "infra"
    return "correctness"


def reattempt_decision(ledger: Optional[Mapping[str, Any]], *, status: Optional[str], error: Optional[str] = None) -> Dict[str, Any]:
    """`{action: reattempt | hold, class, reattempts, reason}` for a failed attempt of a standup whose durable ledger entry is
    ``ledger`` (None = never failed before)."""
    prior = int((ledger or {}).get("reattempts") or 0)
    klass = failure_class(status, error=error)
    if (ledger or {}).get("held"):
        return {"action": "hold", "class": klass, "reattempts": prior, "reason": f"the standup is already HELD: {(ledger or {}).get('hold_reason')}"}
    if klass == "infra" and prior < D73_MAX_AUTO_REATTEMPTS:
        return {"action": "reattempt", "class": klass, "reattempts": prior + 1,
                "reason": f"infra failure {status or error!r}: automatic re-attempt {prior + 1}/{D73_MAX_AUTO_REATTEMPTS} under the attempt rules "
                          "(a new immutable attempt root; acceptance only via the throwing barrier)"}
    why = ("a second infra failure" if klass == "infra" else "a correctness refusal never earns a re-attempt")
    return {"action": "hold", "class": klass, "reattempts": prior,
            "reason": f"{klass} failure {status or error!r}: {why} -- the standup is torn down and HELD (no further containers); the wave continues"}


# --------------------------------------------------------------------------- #
# THE LAUNCH CERTIFICATE: ONE immutable record `--go` / `--legs` / `--reattempt` consume, nothing else                             #
# --------------------------------------------------------------------------- #

#: `RS/launch_certificate.json`: the pre-authorised gate set, bound to W + the PRODUCTION wave + `launch_plan_sha`, every
#: component carried WHOLE and re-validated independently at consumption (never a flag another tool set): the `S0` PASS record, the `S1`-`S4`
#: smoke PASS records, the `S5` block (references, the complete window measurements with `clamped_calls` REQUIRED and zero, the complete
#: utilisation verdict), the W gate (introduced == empty by node id), the exact decode-ref resolution proof for the battery's eight rows,
#: and the draft == final verdict. `certificate_sha256` seals it: a byte changed anywhere after issue fails the seal.
CERTIFICATE_SCHEMA = "seedrep_launch_certificate/1"
CERTIFICATE_FILE = "launch_certificate.json"
#: `carry_over`: the carry-over record `RS/carry_over.json` as issued (every CARRIED attempt with its provenance + the five clause
#: verdicts; `None` when nothing is carried) -- re-validated at consumption by `seedrep_carry_over.carry_over_problems` (provenance, clauses, the
#: root still COMPLETE, THIS wave's live accepted Dict naming the attempt)
#: `launch_set` = the standup ids this certificate certifies for launch (None = every planned row); `s5_coverage` = the
#: DERIVED per-class statement over the launched rows' window classes (own reference / dominated by a larger complete class / gap) with each
#: reference's admission basis (this plan's own tree and measurement wave, or a PRIOR tree / wave admitted on a matching engine identity) -- recomputed
#: at consumption and held equal to the recorded block
CERTIFICATE_KEYS: Tuple[str, ...] = ("schema", "W", "W7", "production_wave_uuid", "launch_plan_sha", "draft_launch_plan_sha", "draft_final_problems",
                                     "plan_sha256", "smoke_waves", "s0", "smokes", "accepted_dicts", "s5", "utilisation", "utilisation_waiver", "w_gate", "decode_rows", "carry_over",
                                     "launch_set", "s5_coverage", "issued_utc", "identity_rule", "certificate_sha256")


def _carry_over_module():
    """The sibling `seedrep_carry_over` (it imports this module; loaded lazily so the import stays one-directional at module load)."""
    import seedrep_carry_over as co  # noqa: PLC0415
    return co
ACCEPTED_DICTS_KEYS: Tuple[str, ...] = ("dict", "read_utc", "entries", "digest")
#: the ONE smoke whose PASS record the certificate admits a documented WAIVER in place of -- S4, the multi-turn smokes that never
#: ran to completion on any tree (one was stopped by its Dict; the next drained at its deadline with the crescendo `none` child COMPLETE and
#: the fitted children SIGTERMed). The waiver is an explicit operator document `RS/s4_waiver.json` with a CLOSED schema (`S4_WAIVER_KEYS`, every
#: field present and typed; `evidence` names the drained attempts by root + RESULT line + facts). The reader (`s4_waiver_problems`) refuses an
#: empty or free-text waiver, one for any id but S4, one for another tree, and one whose named attempt roots are not on the volume (each root's
#: RESULT is re-read and held to the recorded line); `smoke_waiver_entry` records `smokes.S4 = {status: WAIVED, smoke_id, waiver: {path, sha256,
#: document_sha256, document}}` in the certificate so every downstream reader renders it as a QUALIFICATION, never as a PASS. It waives the S4
#: PASS RECORD only: S0, S1-S3, S5, the W gate, the decode proof and draft == final are checked exactly as before.
S4_WAIVER_FILE = "s4_waiver.json"
S4_WAIVER_SCHEMA = "seedrep_smoke_waiver/1"
S4_WAIVER_SMOKE_ID = "S4"
S4_WAIVER_KEYS: Tuple[str, ...] = ("schema", "smoke_id", "W", "waived_by", "evidence", "pools_geometry", "what_the_waiver_does_not_prove", "utc")
S4_WAIVER_EVIDENCE_KEYS: Tuple[str, ...] = ("attempt_id", "standup_id", "attempt_root", "RESULT", "facts")
SMOKE_WAIVED = "WAIVED"
SMOKE_WAIVER_ENTRY_KEYS: Tuple[str, ...] = ("status", "smoke_id", "waiver")
SMOKE_WAIVER_REF_KEYS: Tuple[str, ...] = ("path", "sha256", "document_sha256", "document")
#: admission is BOUND to the authorized source attempts -- the two prior smoke attempts the rule names,
#: as an explicit approved list: each attempt's id, the smoke standup it ran (held to the attempt's OWN ATTEMPT.json `launch_plan_row` and
#: its RESULT `standup=` token), its RESULT verdict (DRIVERS_FAILED, 5 children, the drain's failed count), the facts the rule established
#: (the crescendo `none` child COMPLETE: 65/65 goals, 506 target forwards, 0 attack failures) and the limitation statements the waiver must
#: state. A waiver naming any other attempt, another standup, other facts or fewer limitations is refused by name; an invented waiver over an
#: unrelated attempt can no longer pass.
S4_WAIVER_RULING = "D133"
S4_WAIVER_PRIOR_W7 = "56aff57"                          # the tree the drained S4 smokes ran on
S4_WAIVER_AUTHORIZED: Dict[str, Dict[str, Any]] = {
    "01M2XD13D0PBDQWVSQ8P1VGC5R-a63d248bf60d": {
        "leg": "crescendo", "standup_id": "smoke-S4__crescendo__s1__base__shard00", "status": "DRIVERS_FAILED", "children": 5, "failed": 4,
        "facts": {"none_child": {"rc": 0, "goals_run": "65/65", "n_target_forwards_total": 506, "attack_failures": 0}},
        "facts_required": ("none_child", "fitted_children", "pools")},
    "01M2XD13H6RAGY90G1PYQ6VC7R-a49b15694ddb": {
        "leg": "actorattack", "standup_id": "smoke-S4__actorattack__s1__base__shard00", "status": "DRIVERS_FAILED", "children": 5, "failed": 5,
        "facts": {}, "facts_required": ("children",)},
}
S4_WAIVER_LIMITATIONS: Tuple[str, ...] = ("a completed fitted-cell multi-turn store on this tree", "the attacker pools' geometry under load")
S4_WAIVER_POOLS_GEOMETRY_TERMS: Tuple[str, ...] = ("H200:1", "D116")
_RESULT_TOKEN_RE = re.compile(r"(\w+)=(\S+)")


def s4_waiver_authorized_root(attempt_id: str) -> str:
    """The IMMUTABLE root of an authorized source attempt on the volume, composed the way the app spells every smoke attempt root
    (`attempt_root_for` under `<lane>/<W7>/exp/<results>_smoke/S4`), never typed."""
    auth = S4_WAIVER_AUTHORIZED[attempt_id]
    return attempt_root_for(f"{LANE}/{S4_WAIVER_PRIOR_W7}/exp/{RESULTS_ROOT_NAME}{SMOKE_RESULTS_SUFFIX}/{S4_WAIVER_SMOKE_ID}",
                            leg=auth["leg"], view="s1", stage=STAGE_BASE, shard=0, attempt_id=attempt_id)


def _result_tokens(line: str) -> Dict[str, str]:
    return {k: v for k, v in _RESULT_TOKEN_RE.findall(str(line))}


def is_smoke_waiver_entry(x: Any) -> bool:
    """A certificate `smokes[<id>]` entry that CLAIMS to be a waiver (`status == WAIVED`) -- validated by :func:`smoke_waiver_entry_problems`."""
    return isinstance(x, Mapping) and x.get("status") == SMOKE_WAIVED


def s4_waiver_problems(doc: Any, *, plan: Mapping[str, Any], read_text: Optional[Callable[[str], Optional[str]]] = None) -> List[str]:
    """Why ``doc`` is not an admissible S4 waiver document (empty = it is): exactly :data:`S4_WAIVER_KEYS`; the schema; `smoke_id` S4 and
    nothing else; `W` this plan's tree; `waived_by` naming the ruling (a `D<n>` id -- free text with no ruling is refused); `evidence` a
    NON-EMPTY list of objects with exactly :data:`S4_WAIVER_EVIDENCE_KEYS`, each naming a `/vol/...` attempt root that ends in its attempt id,
    an `smoke-S4__...` standup id, the RESULT line (`status=... attempt=<id>`) and a non-empty `facts` object; with ``read_text`` every named
    root's RESULT is RE-READ off the volume and must equal the recorded line (a root that is gone, or a re-worded line, refuses); `pools_geometry`
    and `utc` non-empty strings; `what_the_waiver_does_not_prove` a non-empty list of non-empty statements."""
    who = "s4_waiver"
    if not isinstance(doc, Mapping):
        return [f"{who}: not an object"]
    out: List[str] = []
    missing = [k for k in S4_WAIVER_KEYS if k not in doc]
    extra = [k for k in doc if k not in S4_WAIVER_KEYS]
    if missing:
        out.append(f"{who}: lacks {missing} (the schema is closed: {list(S4_WAIVER_KEYS)})")
    if extra:
        out.append(f"{who}: carries keys outside the closed schema {extra}")
    if missing:
        return out
    if doc["schema"] != S4_WAIVER_SCHEMA:
        out.append(f"{who}: schema {doc['schema']!r} != {S4_WAIVER_SCHEMA}")
    if doc["smoke_id"] != S4_WAIVER_SMOKE_ID:
        out.append(f"{who}: names smoke {doc['smoke_id']!r}; a smoke waiver is admitted for {S4_WAIVER_SMOKE_ID} only (D133)")
    if not isinstance(doc["W"], str) or doc["W"] != plan.get("W"):
        out.append(f"{who}: W {doc['W']!r} is not this plan's {plan.get('W')!r} (the waiver certifies ONE tree)")
    wb = doc["waived_by"]
    if not isinstance(wb, str) or not wb.strip():
        out.append(f"{who}: waived_by missing or empty")
    elif not re.search(r"\b" + re.escape(S4_WAIVER_RULING) + r"\b", wb):
        out.append(f"{who}: waived_by {wb!r} does not name the ruling {S4_WAIVER_RULING} -- the only ruling that waives S4; a free-text waiver is not admitted")
    ev = doc["evidence"]
    if not isinstance(ev, list) or not ev:
        out.append(f"{who}: evidence must be a NON-EMPTY list of the drained attempts (each {list(S4_WAIVER_EVIDENCE_KEYS)}) -- an empty waiver waives nothing")
    else:
        named = {str(e.get("attempt_id")) for e in ev if isinstance(e, Mapping) and isinstance(e.get("attempt_id"), str)}
        missing_auth = sorted(set(S4_WAIVER_AUTHORIZED) - named)
        if missing_auth:
            out.append(f"{who}: evidence does not name {S4_WAIVER_RULING}'s authorized source attempt(s) {missing_auth} -- the waiver rests on BOTH drained W22 S4 attempts")
        seen: set = set()
        for i, e in enumerate(ev):
            w = f"{who}: evidence[{i}]"
            if not isinstance(e, Mapping):
                out.append(f"{w}: not an object (free text is not evidence)")
                continue
            miss = [k for k in S4_WAIVER_EVIDENCE_KEYS if k not in e]
            ext = [k for k in e if k not in S4_WAIVER_EVIDENCE_KEYS]
            if miss or ext:
                out.append(f"{w}: lacks {miss} / carries {ext} outside {list(S4_WAIVER_EVIDENCE_KEYS)}")
                continue
            aid, sid, root, res, facts = e["attempt_id"], e["standup_id"], e["attempt_root"], e["RESULT"], e["facts"]
            if not isinstance(aid, str) or not aid.strip():
                out.append(f"{w}: attempt_id missing or empty")
                continue
            if aid in seen:
                out.append(f"{w}: attempt {aid} is named twice")
            seen.add(aid)
            auth = S4_WAIVER_AUTHORIZED.get(aid)
            if auth is None:
                out.append(f"{w}: attempt {aid!r} is not one of {S4_WAIVER_RULING}'s authorized source attempts {sorted(S4_WAIVER_AUTHORIZED)} -- no other attempt waives S4")
                continue
            if sid != auth["standup_id"]:
                out.append(f"{w}: standup_id {sid!r} is not the authorized attempt's standup {auth['standup_id']!r}")
            want_root = s4_waiver_authorized_root(aid)
            if root != want_root:
                out.append(f"{w}: attempt_root {root!r} is not the authorized attempt's root {want_root}")
            toks = _result_tokens(res) if isinstance(res, str) else {}
            want_toks = {"status": auth["status"], "attempt": aid, "standup": auth["standup_id"], "children": str(auth["children"]), "failed": str(auth["failed"])}
            bad_toks = {k: toks.get(k) for k, v in want_toks.items() if toks.get(k) != v}
            if bad_toks:
                out.append(f"{w}: RESULT {str(res)[:100]!r} does not state the authorized attempt's outcome ({bad_toks} should be "
                           f"{ {k: want_toks[k] for k in bad_toks} })")
            if not isinstance(facts, Mapping) or not facts:
                out.append(f"{w}: facts must be a non-empty object (what the drained attempt DID establish)")
            else:
                for k in auth["facts_required"]:
                    v = facts.get(k)
                    if v is None or (isinstance(v, (str, Mapping, list)) and not v):
                        out.append(f"{w}: facts.{k} missing or empty ({S4_WAIVER_RULING} records it for this attempt)")
                for k, want in auth["facts"].items():
                    got = facts.get(k)
                    if not isinstance(got, Mapping):
                        continue
                    for kk, vv in want.items():
                        if got.get(kk) != vv:
                            out.append(f"{w}: facts.{k}.{kk} = {got.get(kk)!r}, not the recorded {vv!r} ({S4_WAIVER_RULING})")
            if read_text is not None and isinstance(root, str) and root.startswith(VOL_PATH + "/") and isinstance(res, str):
                live = read_text(f"{root}/RESULT")
                if live is None:
                    out.append(f"{w}: {root}/RESULT is not on the volume -- the named attempt root does not exist (or its RESULT is unreadable)")
                elif live.strip() != res.strip():
                    out.append(f"{w}: the RESULT on the volume ({live.strip()[:100]!r}) is not the recorded line ({res.strip()[:100]!r})")
                att_text = read_text(f"{root}/ATTEMPT.json")
                att = None
                if att_text is not None:
                    try:
                        att = json.loads(att_text)
                    except ValueError:
                        att = None
                if not isinstance(att, Mapping):
                    out.append(f"{w}: {root}/ATTEMPT.json is not on the volume (or not JSON) -- the attempt's own identity record cannot be read")
                else:
                    if att.get("attempt_id") != aid:
                        out.append(f"{w}: ATTEMPT.json names attempt {att.get('attempt_id')!r}, not {aid}")
                    if att.get("launch_plan_row") != sid:
                        out.append(f"{w}: ATTEMPT.json records launch_plan_row {att.get('launch_plan_row')!r}, not the waiver's standup_id {sid!r}")
                    if not str(att.get("W") or "").startswith(S4_WAIVER_PRIOR_W7):
                        out.append(f"{w}: ATTEMPT.json records W {str(att.get('W'))[:12]!r}, not the W22 tree {S4_WAIVER_PRIOR_W7} the authorized attempts ran on")
    pg = doc["pools_geometry"]
    if not isinstance(pg, str) or not pg.strip():
        out.append(f"{who}: pools_geometry missing or empty (the attacker pools' geometry the waived smoke did not exercise must be named)")
    else:
        lack = [t for t in S4_WAIVER_POOLS_GEOMETRY_TERMS if t not in pg]
        if lack:
            out.append(f"{who}: pools_geometry {pg!r} does not state {lack} (the ruling's pool geometry)")
    wn = doc["what_the_waiver_does_not_prove"]
    if not isinstance(wn, list) or not wn or not all(isinstance(x, str) and x.strip() for x in wn):
        out.append(f"{who}: what_the_waiver_does_not_prove must be a non-empty list of non-empty statements")
    else:
        lack = [t for t in S4_WAIVER_LIMITATIONS if t not in wn]
        if lack:
            out.append(f"{who}: what_the_waiver_does_not_prove lacks the required limitation statement(s) {lack}")
    utc = doc["utc"]
    if not isinstance(utc, str) or not re.match(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}", utc):
        out.append(f"{who}: utc {utc!r} is not an ISO-8601 timestamp")
    return out


def smoke_waiver_entry(doc: Mapping[str, Any], *, path: str, file_sha256: str) -> Dict[str, Any]:
    """The certificate's `smokes[S4]` entry for an admitted waiver: `{status: WAIVED, smoke_id, waiver: {path, sha256 (the file's bytes),
    document_sha256 (the canonical JSON of the document), document (verbatim)}}` -- self-contained, so `--go` re-validates it from the
    certificate alone and every reader sees WHAT was waived and on WHAT evidence."""
    return {"status": SMOKE_WAIVED, "smoke_id": str(doc.get("smoke_id")),
            "waiver": {"path": str(path), "sha256": str(file_sha256), "document_sha256": _canonical_sha256(doc), "document": copy.deepcopy(dict(doc))}}


def smoke_waiver_entry_problems(entry: Any, *, smoke_id: str, plan: Mapping[str, Any],
                                read_text: Optional[Callable[[str], Optional[str]]] = None) -> List[str]:
    """Why a certificate `smokes[<smoke_id>]` WAIVED entry is not admissible (empty = it is): only S4 may carry one; exactly
    :data:`SMOKE_WAIVER_ENTRY_KEYS` / :data:`SMOKE_WAIVER_REF_KEYS`; the path names :data:`S4_WAIVER_FILE`; a 64-hex file sha; the document's
    canonical sha matching the embedded document (an edited entry refuses); then the document itself through :func:`s4_waiver_problems`
    (with ``read_text`` the named roots are re-read off the volume)."""
    if smoke_id != S4_WAIVER_SMOKE_ID:
        return [f"{smoke_id}: a smoke WAIVER is admitted only for {S4_WAIVER_SMOKE_ID} (D133) -- {smoke_id} needs a derived PASS record"]
    who = f"{smoke_id} waiver"
    if not isinstance(entry, Mapping):
        return [f"{who}: not an object"]
    out: List[str] = []
    if [k for k in SMOKE_WAIVER_ENTRY_KEYS if k not in entry] or [k for k in entry if k not in SMOKE_WAIVER_ENTRY_KEYS]:
        out.append(f"{who}: entry keys {sorted(entry)} are not exactly {list(SMOKE_WAIVER_ENTRY_KEYS)}")
        return out
    if entry.get("status") != SMOKE_WAIVED or entry.get("smoke_id") != S4_WAIVER_SMOKE_ID:
        out.append(f"{who}: status {entry.get('status')!r} / smoke_id {entry.get('smoke_id')!r} is not a WAIVED S4 entry")
    ref = entry.get("waiver")
    if not isinstance(ref, Mapping) or [k for k in SMOKE_WAIVER_REF_KEYS if k not in ref] or [k for k in ref if k not in SMOKE_WAIVER_REF_KEYS]:
        out.append(f"{who}: waiver must carry exactly {list(SMOKE_WAIVER_REF_KEYS)}")
        return out
    if not isinstance(ref["path"], str) or not ref["path"].endswith(S4_WAIVER_FILE):
        out.append(f"{who}: waiver.path {ref['path']!r} does not name {S4_WAIVER_FILE}")
    if not isinstance(ref["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", ref["sha256"]):
        out.append(f"{who}: waiver.sha256 is not a 64-hex file digest")
    doc = ref["document"]
    if not isinstance(doc, Mapping):
        out.append(f"{who}: waiver.document is not an object")
        return out
    if ref["document_sha256"] != _canonical_sha256(doc):
        out.append(f"{who}: waiver.document_sha256 is not the digest of the embedded document (edited after issue)")
    out += [f"{who}: {p}" for p in s4_waiver_problems(doc, plan=plan, read_text=read_text)]
    return out
#: the S1-S4 smokes whose PASS records the certificate carries (S5 is its own block: references + windows + the utilisation verdict)
SMOKE_PASS_IDS: Tuple[str, ...] = ("S1", "S1b", "S2", "S3", "S4")
#: the smoke ids `derive_smoke_pass` DERIVES -- S1-S4's PASS records + S5's window MEASUREMENTS (the same producer-authority machinery:
#: the accepted attempt off the LIVE Dict, the producer's bound file under its root; the S5 derivation is written by
#: `seedrep_s5_record.py window` into `S5.multiturn[]`, never into a `smokes.json["S5"]` PASS record -- that key is the S5 block)
S5_WINDOW_SMOKE_ID = "S5"
SMOKE_DERIVABLE_IDS: Tuple[str, ...] = SMOKE_PASS_IDS + (S5_WINDOW_SMOKE_ID,)
#: a smoke PASS record is DERIVED by the app, never authored -- the operator supplies POINTERS (the smoke rows' standup ids, the paths
#: of the smoke-specific check OUTPUTS) and the app resolves the accepted attempt per row against the smoke wave's accepted Dict, reads
#: its COMPLETE.json / RESULT / modal_evidence.json off the volume, and evaluates each check on the pointed file's CONTENT; every source
#: is recorded with its sha256 and re-hashed at consumption (a changed file refuses)
SMOKE_PASS_SCHEMA = "seedrep_smoke_pass/2"
SMOKE_PASS_KEYS: Tuple[str, ...] = ("schema", "smoke_id", "verdict", "rc", "W", "execution_wave_uuid", "production_wave_uuid", "launch_plan_sha",
                                    "standup_ids", "accepted", "rows", "checks", "derived_utc")
SMOKE_ROW_FACT_FILES: Tuple[str, ...] = ("COMPLETE.json", "RESULT", "modal_evidence.json")
#: the smoke-specific check OUTPUTS each smoke must point at: the resolved-sampler recorder's verdict (`resolved_sampler.json`,
#: `verdict == AGREES`), the request-evidence audit verdict (`verdict == AGREES`, bound to an accepted attempt of the smoke), the adapter
#: hash bank covering every planned adapter (S1b), the S4 decisions file (S4)
SMOKE_CHECKS: Dict[str, Tuple[str, ...]] = {"S1": ("resolved_sampler", "request_evidence_audit"), "S1b": ("resolved_sampler", "request_evidence_audit", "adapter_hashes_banked"),
                                            "S2": ("resolved_sampler", "request_evidence_audit"),
                                            # the determinism spot-check REPRODUCES gates S3's PASS (launch_brief step 3:
                                            # S3 = ladder closure + the real audit AGREES + the spot-check); an earlier derivation took S3
                                            # from the sampler + audit + pool snapshot alone, so a row whose spot-check never ran or did not
                                            # reproduce still PASSED
                                            "S3": ("resolved_sampler", "request_evidence_audit", "pool_endpoint", "determinism"),
                                            "S4": ("decisions", "pool_endpoint"),
                                            S5_WINDOW_SMOKE_ID: ("s5_window_measurement",)}
SMOKE_CHECK_NAMES: Tuple[str, ...] = ("resolved_sampler", "request_evidence_audit", "adapter_hashes_banked", "decisions", "s5_window_measurement", "pool_endpoint",
                                      "determinism")
#: PRODUCER AUTHORITY: smoke evidence comes ONLY from canonical producer-owned files under the ACCEPTED attempt root (the root the
#: smoke wave's LIVE accepted Dict names), read through the Volume API, each bound by the producer (the container's
#: `modal_evidence.json["producer_evidence"][<rel>].sha256`) -- never an operator path, never a verdict field alone:
#:   resolved_sampler        <root>/resolved_sampler.json                      written by scripts/sdfbase/record_resolved_sampler.py in-container
#:   request_evidence_audit  <root>/audits/request_log/<leg>/<arm>__s<s>[__cap<N>][__<shard>].json   written in-container by request_evidence_audit.py
#:                                                                              over THIS root (one verdict per child store)
#:   adapter_hashes_banked   <root>/exp/serve/cells.json                       the container's own enumeration of the served cells (hashes)
#:   decisions               <root>/modal_evidence.json + the certified plan's S4 windows (the plan is bytes-bound by the certificate)
RESOLVED_SAMPLER_REL = "resolved_sampler.json"
#: the producer's PER-TASK sampler expectations (`seedrep_ladder_helpers.py expected-samplers`, the driver venv) the recorder is handed
#: as `--expect-tasks`; written under the attempt root beside the recorder's record and embedded in it (`client_task_expectations`)
EXPECTED_SAMPLERS_REL = "expected_samplers.json"
AUDIT_OUT_REL = "audits"
CELLS_JSON_REL = f"{SERVE_DIR_REL}/cells.json"
PRODUCER_EVIDENCE_KEY = "producer_evidence"
PRODUCER_WRITERS: Dict[str, str] = {"resolved_sampler": "scripts/sdfbase/record_resolved_sampler.py", "request_evidence_audit": "scripts/sdfbase/seedrep/request_evidence_audit.py",
                                    "adapter_hashes_banked": "scripts/modal/seedrep_served.py", "decisions": "scripts/modal/seedrep_served.py",
                                    "s5_window_measurement": "scripts/modal/seedrep_s5_measure.py",
                                    "pool_endpoint": "scripts/modal/seedrep_attacker_pool.py",
                                    "determinism": "scripts/sdfbase/seedrep/determinism_check.py"}
#: the determinism spot-check, as the app binds it (`seedrep_served.determinism_evidence`): ONE record per TABLE leg of the
#: row at `<root>/audits/determinism/<table leg>.json` (`DETERMINISM_COPY_REL`), each bound -- sha256 / writer / tree_sha / rc, or ABSENT
#: by name -- under `modal_evidence.json["determinism"][<table leg>]` (NOT under `producer_evidence`; the app's idiom for this check). The
#: derivation reads that binding through the same `_producer_binding` rule by re-keying it to the record's rel. The record's own keys
#: (`determinism_check.py`'s writer, in every generation): `reproduces` = every re-sent request came back byte-identical (`identical ==
#: n_requests`); newer records add `standup_id / view / stage / shard / smoke_id` (the row it is OF), checked when present.
DETERMINISM_EVIDENCE_KEY = "determinism"
DETERMINISM_OUT_REL = "audits/determinism"
DETERMINISM_RECORD_KEYS: Tuple[str, ...] = ("leg", "arms", "n_items", "n_requests", "n", "identical", "reproduces", "misses")
#: the determinism spot-check is NOT APPLICABLE to a SAMPLED decode -- a present, diverged spot-check record on a row whose VERIFIED
#: resolved decode has every client temperature > 0 is a recorded QUALIFICATION (the source's `criterion3`, the check's `criterion3` block),
#: never a refusal and never a waiver; a greedy row keeps the byte-identical rule. ONE spelling: `seedrep_carry_over.CRITERION3_NOT_APPLICABLE_SAMPLED`
#: is this constant, and `_determinism_source` routes every smoke-PASS derivation through the carry-over's `determinism_d180_relaxation`.
DETERMINISM_CRITERION3_NOT_APPLICABLE_SAMPLED = "not_applicable_sampled_decode"
DETERMINISM_CRITERION3_RULING = "D180"
#: the items the spot-check selects per arm -- MIRRORS `determinism_check.N_PLANNED` (the check imports no plan module; a
#: test pins the two equal). A BUDGET-LIMITED smoke row (an S1-S4 BoN / misalign row under the item budget, whose store files 10 items by
#: construction) selects min(20, budget) -- `determinism_spot_check_n` derives it from the row's `smoke.budget`, the app passes it as
#: `--n` + `--item-budget`, and the record says so (`n_planned` / `n_selected` / `budget_limited` / `item_budget`, additive). A
#: production row keeps 20 and the check's refusal of a shorter store (a real defect signal).
DETERMINISM_N_PLANNED = 20
#: the additive budget keys of the spot-check record; tolerated absent (an older record), checked for consistency when present
DETERMINISM_BUDGET_KEYS: Tuple[str, ...] = ("n_planned", "n_selected", "budget_limited", "item_budget")


def determinism_table_legs_of_row(row: Mapping[str, Any]) -> Tuple[str, ...]:
    """The TABLE legs the spot-check writes a record for on ONE row (the app's `determinism_table_legs(row leg, stage)`): the controls
    container's `audit_table_legs` at its stage (three on the base, the two ladder legs on a rung -- F-T), every other leg its eval legs."""
    if row["leg"] == "controls":
        return audit_table_legs("controls", stage=row["stage"])
    return tuple(EVAL_LEGS_OF.get(str(row["leg"]), (str(row["leg"]),)))


def determinism_rels(row: Mapping[str, Any]) -> List[Tuple[str, str]]:
    """``[(table leg, <root>-relative record path), ...]`` of one row's spot-check records."""
    return [(leg, f"{DETERMINISM_OUT_REL}/{leg}.json") for leg in determinism_table_legs_of_row(row)]


def determinism_producer_map(bindings: Any) -> Dict[str, Any]:
    """`modal_evidence.json["determinism"]` (table leg -> binding) re-keyed by the record's rel, so `_producer_binding` reads it unchanged."""
    if not isinstance(bindings, Mapping):
        return {}
    return {f"{DETERMINISM_OUT_REL}/{leg}.json": b for leg, b in bindings.items()}


def producer_map_for(name: str, evidence: Mapping[str, Any]) -> Dict[str, Any]:
    """The producer-binding map ONE check reads off a root's modal_evidence.json: the determinism check's lives under its own key."""
    if name == "determinism":
        return determinism_producer_map(evidence.get(DETERMINISM_EVIDENCE_KEY))
    pe = evidence.get(PRODUCER_EVIDENCE_KEY)
    return dict(pe) if isinstance(pe, Mapping) else {}
#: The window MEASUREMENT's canonical location under the probe attempt root -- `audits/s5_window/<leg>/<cell>__s<s>[__cap<N>][__<shard>].json`,
#: one per probe child (the `none` parent and the probe cell of that standup), written IN-CONTAINER by `seedrep_s5_measure.py` over the
#: child's COMPLETE store + the served target's per-call `target_timing.jsonl` (prompt_tokens + completion_tokens from the engine's usage,
#: `max_tokens_clamped_from` / `error: prompt_over_window` per call) and bound under `producer_evidence` like the audit; `seedrep_s5_record.py
#: window` reads THIS file off the accepted attempt and nothing an operator authors
S5_WINDOW_OUT_REL = "audits/s5_window"
S5_WINDOW_MEASUREMENT_EXTRA_KEYS: Tuple[str, ...] = ("cell", "s", "staged_id", "object_sha256", "n_calls", "n_prompt_over_window", "n_clamped_retries",
                                                     "complete", "completeness_note", "served_window", "context_max_call", "store_sha256", "timing_sha256",
                                                     "n_goals_with_progress", "n_goals_errored", "produced_by")
PRODUCED_BY_KEYS: Tuple[str, ...] = ("writer", "tree_sha", "attempt_id", "standup_id", "wave_uuid")
#: F-M: a pool-dependent row's producer copies its pool's LIVE endpoint record (the pool Server's endpoint.json on the shared volume)
#: byte-for-byte into its attempt root and binds it -- the certificate then keeps the pool's deployment shape (model revision, cards, allocator
#: config, window, S, app / server / container identity) and refuses changed bytes; the plan's pool rows carry only the record's PATH.
POOL_ENDPOINT_SNAPSHOT_REL = "pool_endpoint.json"
POOL_ENDPOINT_KEYS: Tuple[str, ...] = ("lane", "pool_name", "url", "model", "snapshot", "gpu", "tensor_parallel_size", "max_model_len", "max_num_seqs",
                                       "max_num_batched_tokens", "pytorch_cuda_alloc_conf", "vllm_version", "repo_sha", "liveness", "job_id", "node", "started_at")
POOL_ENDPOINT_FACTS: Tuple[str, ...] = ("model", "snapshot", "gpu", "tensor_parallel_size", "max_model_len", "max_num_seqs", "max_num_batched_tokens",
                                        "pytorch_cuda_alloc_conf", "vllm_version", "repo_sha", "url", "job_id", "node", "started_at")


def pool_endpoint_record(pool_name: str, *, vol_path: str = VOL_PATH) -> str:
    """The pool Server's endpoint.json on the shared volume for app ``seedrep-attacker-<k>`` (the pool module's ``pool_dir(k)`` + the
    contract's ENDPOINT_FILE); the path the plan's pool rows carry and the producer snapshots."""
    prefix = "seedrep-attacker-"
    if not (isinstance(pool_name, str) and pool_name.startswith(prefix) and pool_name[len(prefix):].isdigit()):
        raise ValueError(f"pool name {pool_name!r} is not seedrep-attacker-<k>")
    return f"{vol_path}/attacker-pool/seedrep-{int(pool_name[len(prefix):])}/endpoint.json"
#: the recorder's record (record_resolved_sampler.py `rec = {...}`): the keys every reader of it cites
RESOLVED_SAMPLER_KEYS: Tuple[str, ...] = ("verdict", "why", "classes", "n_client_requests", "n_probe_requests", "n_unclassified_requests", "class_rule", "n_logs",
                                          "distinct_resolved", "reference", "reference_source", "divergences")
#: the per-store verdict record (request_evidence_audit.py `_store_verdict` + `audit_root`'s update): the keys the certificate reads
AUDIT_VERDICT_KEYS: Tuple[str, ...] = ("leg", "arm", "s", "verdict", "problem", "problems", "checks", "n_draws", "n_draws_bound", "store_dir", "store_complete",
                                       "attempt_id", "W", "view", "stage", "shard", "engine_log", "resolved_sampler_verdict")
AUDIT_DEFAULT_SHARD = "shard00"
#: The request-evidence RECONCILIATION CONTRACT every audit verdict is issued under, and the qualification it carries. Spellings mirror
#: `audit_common.RECONCILIATION_CONTRACT_LEDGER / _LEDGERLESS` and `LEDGER_*` (this module imports nothing from sdfbase;
#: tests/test_seedrep_fg2_d106_qualification.py holds the two equal). `AUDIT_CONTRACT_LEDGER` = the ledger-backed reconciliation as
#: written (`ledger` declared | present_undeclared, the attempt inventory from the usage ledger, the client's body digest checked against it).
#: `AUDIT_CONTRACT_LEDGERLESS` = the same gate on a row whose request path carries NO ledger (`RoutingProxy` writes none): `ledger` not_on_path,
#: attempts from the client's response hook, NO body digest counterpart, rejected / pre-engine attempts corroborated by the client alone -- a
#: revised, QUALIFIED certification contract, never equivalent evidence. The certificate copies the qualification VERBATIM into its facts,
#: renders it next to the verdict (never a bare AGREES), and refuses a verdict without a version, under an unknown version, or with a
#: contradictory (contract, ledger, attempt_source, body digest) tuple -- at derivation and again on the recorded facts at certify.
AUDIT_CONTRACT_LEDGER = "seedrep_request_evidence_reconciliation/1"
AUDIT_CONTRACT_LEDGERLESS = "seedrep_request_evidence_reconciliation/2-D106"
#: The controls driver writes NO request evidence (no request_evidence.jsonl / response_evidence.jsonl on a controls root), so two further
#: REVISED, QUALIFIED contracts exist for the evidence shape it does write -- `AUDIT_CONTRACT_CONTROLS_EVENTS` (a sampled controls store: every
#: draw bound through its controls event's completion_id to exactly one engine line whose seed / cap / sampler / LoRA are re-read from the
#: engine; no body digest) and `AUDIT_CONTRACT_MMLU_COUNT` (the MMLU readout: NO per-draw binding, count-level identity only). Spellings mirror
#: `audit_common.RECONCILIATION_CONTRACT_CONTROLS_EVENTS / _MMLU_COUNT` (pinned equal by tests/test_seedrep_fc3_controls_audit_shape.py). Like
#: `AUDIT_CONTRACT_LEDGERLESS`, both are `ledger == not_on_path`, body digest NOT available, an independence_note REQUIRED, never reported as
#: ledger-backed.
AUDIT_CONTRACT_CONTROLS_EVENTS = "seedrep_request_evidence_reconciliation/3-D133-controls-events"
AUDIT_CONTRACT_MMLU_COUNT = "seedrep_request_evidence_reconciliation/3b-D133-mmlu-count"
AUDIT_CONTRACTS: Tuple[str, ...] = (AUDIT_CONTRACT_LEDGER, AUDIT_CONTRACT_LEDGERLESS, AUDIT_CONTRACT_CONTROLS_EVENTS, AUDIT_CONTRACT_MMLU_COUNT)
#: the contracts under which the ledger's independent attempt record is LOST (every contract but the ledger-backed one)
AUDIT_CONTRACTS_LEDGERLESS: Tuple[str, ...] = (AUDIT_CONTRACT_LEDGERLESS, AUDIT_CONTRACT_CONTROLS_EVENTS, AUDIT_CONTRACT_MMLU_COUNT)
AUDIT_LEDGER_NOT_ON_PATH = "not_on_path"
AUDIT_LEDGER_BACKED: Tuple[str, ...] = ("declared", "present_undeclared")
AUDIT_ATTEMPT_SOURCE_OF: Dict[str, str] = {AUDIT_CONTRACT_LEDGER: "ledger", AUDIT_CONTRACT_LEDGERLESS: "response_evidence",
                                           AUDIT_CONTRACT_CONTROLS_EVENTS: "controls_events", AUDIT_CONTRACT_MMLU_COUNT: "engine_log_count"}
#: what each qualified contract LOST against the ledger-backed one -- rendered beside the verdict (`audit_qualification_line`)
AUDIT_CONTRACT_LOST: Dict[str, str] = {
    AUDIT_CONTRACT_LEDGERLESS: "rejected / pre-engine attempts not independently corroborated",
    AUDIT_CONTRACT_CONTROLS_EVENTS: "per-draw identity through the client's completion_id and the engine's own line (seed, cap, sampler, LoRA re-read); "
                                    "failed attempts from the client's event stream alone; no body digest (D133)",
    AUDIT_CONTRACT_MMLU_COUNT: "NO per-draw binding: count-level identity only, the readout a sampler-invariant greedy argmax (D133)"}
#: the qualification fields `audit_root` stamps on EVERY verdict -- copied into the certificate's facts verbatim
AUDIT_QUALIFICATION_KEYS: Tuple[str, ...] = ("reconciliation_contract", "ledger", "ledger_source", "attempt_source", "evidence_basis", "body_digest_check",
                                             "independence_note")
AUDIT_BODY_DIGEST_NOT_AVAILABLE = "not_available"       # the producer's `body_digest_check` prefix on a ledger-less row


def audit_qualification_problems(q: Mapping[str, Any], who: str) -> List[str]:
    """Why ``q`` (a producer's verdict, or the certificate's RECORDED facts of one) carries no CONSISTENT qualification (empty = it does).
    THE rule, applied at derivation (`_smoke_check`) and again at certify on the recorded facts (`_smoke_source_facts_problems`): every
    qualification key present -- a verdict without `reconciliation_contract` (the earlier shape) is refused, never defaulted to `/1`; the
    version one of `AUDIT_CONTRACTS`; `/1` <=> `ledger` in {declared, present_undeclared}, `attempt_source == "ledger"`, a body digest check
    that RAN; `/2-D106` <=> `ledger == not_on_path`, `attempt_source == "response_evidence"`, `body_digest_check` not_available, and an
    `independence_note` stating what the ledger-less reconciliation lost; `evidence_basis` a non-empty list of evidence files."""
    out: List[str] = []
    missing = [k for k in AUDIT_QUALIFICATION_KEYS if k not in q]
    if missing:
        out.append(f"{who} carries no D106 qualification {missing} -- a verdict issued before the reconciliation contract was versioned (pre-fz3) "
                   "certifies nothing; the contract version is never defaulted")
        return out
    v, ledger, src = q.get("reconciliation_contract"), q.get("ledger"), q.get("attempt_source")
    if v not in AUDIT_CONTRACTS:
        out.append(f"{who} was issued under reconciliation contract {v!r}, not one of {list(AUDIT_CONTRACTS)}")
        return out
    ledgerless = v in AUDIT_CONTRACTS_LEDGERLESS
    want_ledgers = (AUDIT_LEDGER_NOT_ON_PATH,) if ledgerless else AUDIT_LEDGER_BACKED
    if ledger not in want_ledgers:
        out.append(f"{who}: contract {v} with ledger {ledger!r} -- {v} requires ledger in {list(want_ledgers)} (contradictory qualification)")
    if src != AUDIT_ATTEMPT_SOURCE_OF[v]:
        out.append(f"{who}: contract {v} with attempt_source {src!r} -- {v} requires {AUDIT_ATTEMPT_SOURCE_OF[v]!r} (contradictory qualification)")
    basis = q.get("evidence_basis")
    if not isinstance(basis, list) or not basis or not all(isinstance(b, str) and b for b in basis):
        out.append(f"{who}: evidence_basis {basis!r} is not a non-empty list of evidence files")
    bd = q.get("body_digest_check")
    if not isinstance(bd, str) or not bd:
        out.append(f"{who}: body_digest_check {bd!r} states nothing")
    elif ledgerless != bd.startswith(AUDIT_BODY_DIGEST_NOT_AVAILABLE):
        out.append(f"{who}: contract {v} with body_digest_check {bd[:60]!r} -- "
                   + ("a ledger-less row has no body digest counterpart" if ledgerless else "a ledger-backed row checks the body digest against the ledger"))
    note = q.get("independence_note")
    if ledgerless and not (isinstance(note, str) and note):
        out.append(f"{who}: contract {v} with no independence_note -- what the ledger-less reconciliation lost must be stated")
    return out


def audit_qualification_line(q: Mapping[str, Any]) -> str:
    """The qualification as the certificate RENDERS it beside the verdict (never a bare AGREES): `contract <version> (ledger <status>; attempts
    from <source>; body digest <checked against the ledger | not available>[; rejected / pre-engine attempts not independently corroborated])`."""
    v = q.get("reconciliation_contract")
    ledger = str(q.get("ledger")).replace("_", " ")
    src = str(q.get("attempt_source")).replace("_", " ")
    bd = "not available" if str(q.get("body_digest_check") or "").startswith(AUDIT_BODY_DIGEST_NOT_AVAILABLE) else "checked against the ledger"
    lost = f"; {AUDIT_CONTRACT_LOST[v]}" if v in AUDIT_CONTRACT_LOST else ""
    return f"contract {v} (ledger {ledger}; attempts from {src}; body digest {bd}{lost})"


def audit_reconciliation_tally(sources: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """The check-level statement the certificate lists ABOVE the per-verdict facts: how many verdicts were issued ledger-backed (`/1`)
    and how many ledger-less (`/2-D106`), the contract versions and rendered qualification lines seen -- DERIVED from the sources and
    re-derived at certify (`smoke_pass_problems`), never trusted from the record."""
    lb = sum(1 for s in sources if s.get("reconciliation_contract") == AUDIT_CONTRACT_LEDGER)
    ll = sum(1 for s in sources if s.get("reconciliation_contract") == AUDIT_CONTRACT_LEDGERLESS)
    ce = sum(1 for s in sources if s.get("reconciliation_contract") == AUDIT_CONTRACT_CONTROLS_EVENTS)
    mc = sum(1 for s in sources if s.get("reconciliation_contract") == AUDIT_CONTRACT_MMLU_COUNT)
    n = len(sources)
    parts = []
    if ll:
        parts.append(f"{ll} of {n} verdict(s) ledger-less under {AUDIT_CONTRACT_LEDGERLESS} (D106: the usage ledger is not on the row's request path; "
                     "attempts from response evidence; body digest not available; rejected / pre-engine attempts not independently corroborated)")
    if ce:
        parts.append(f"{ce} of {n} verdict(s) under {AUDIT_CONTRACT_CONTROLS_EVENTS} (D133: the controls driver writes no request evidence; every draw bound "
                     "through its controls event's completion_id to the engine's own line -- seed, cap, sampler, LoRA re-read; no body digest)")
    if mc:
        parts.append(f"{mc} of {n} verdict(s) under {AUDIT_CONTRACT_MMLU_COUNT} (D133: the MMLU readout has NO per-draw binding -- count-level identity only)")
    statement = "; ".join(parts) if parts else f"every verdict ({n}) ledger-backed under {AUDIT_CONTRACT_LEDGER}"
    return {"n_verdicts": n, "n_ledger_backed": lb, "n_ledger_less": ll, "n_controls_events": ce, "n_mmlu_count": mc,
            "contracts": sorted({str(s.get("reconciliation_contract")) for s in sources}),
            "qualifications": sorted({str(s.get("qualification")) for s in sources}), "statement": statement}


def audit_verdict_rel(leg: str, arm: str, s: Any, *, stage: Any, shard: Any) -> str:
    """The audit's `verdict_filename` (request_evidence_audit.py), volume-relative to the attempt root: `audits/request_log/<leg>/<arm>__s<s>` +
    `__cap<N>` for a rung root + `__<shard dir>` for a non-default shard + `.json`."""
    name = f"{arm}__s{int(s)}"
    if str(stage) != STAGE_BASE:
        name += f"__cap{stage}"
    shard_dir = f"shard{int(shard):02d}"
    if shard_dir != AUDIT_DEFAULT_SHARD:
        name += f"__{shard_dir}"
    return f"{AUDIT_OUT_REL}/request_log/{leg}/{name}.json"


def audit_table_legs(eval_leg: str, *, stage: Any = STAGE_BASE) -> Tuple[str, ...]:
    """The TABLE legs the audit writes a verdict for, per CHILD eval leg (`audit_common.LEGS`, its `--legs` vocabulary): the controls
    child (eval_leg `controls`, ONE store holding three evals) gets one verdict per `CONTROLS_ALL` leg (`xstest`, `or_bench_hard_1k`, `mmlu` —
    the store writers' spelling; `EVAL_LEGS_OF["controls"]` spells the registry eval id `or_bench_hard`, which the audit refuses) on the BASE
    row, and one per `CONTROLS_LADDER` leg on a RUNG (a rung store holds no MMLU); every other child's eval leg is its own table leg (a
    misalign child is per behaviour, Petri / GSM8K / BoN / static / multi-turn one leg each)."""
    if eval_leg == "controls":
        return tuple(CONTROLS_ALL) if str(stage) == STAGE_BASE else tuple(CONTROLS_LADDER)
    return (eval_leg,)


def s5_window_rel(leg: str, cell: str, s: Any, *, stage: Any, shard: Any) -> str:
    """The window measurement's path, volume-relative to the probe attempt root: `audits/s5_window/<leg>/<cell>__s<s>` + `__cap<N>` for a
    non-base stage + `__<shard dir>` for a non-default shard + `.json` (the audit's `audit_verdict_rel` spelling; `cell` = the store's defense
    dir, the driver's `--defense`)."""
    name = f"{cell}__s{int(s)}"
    if str(stage) != STAGE_BASE:
        name += f"__cap{stage}"
    sd = f"shard{int(shard):02d}"
    if sd != AUDIT_DEFAULT_SHARD:
        name += f"__{sd}"
    return f"{S5_WINDOW_OUT_REL}/{leg}/{name}.json"


def s5_window_rels(row: Mapping[str, Any]) -> List[Tuple[Mapping[str, Any], str]]:
    """Every S5 window measurement a probe row's container must write: one per child -> ``[(child, rel), ...]``."""
    if row.get("kind") != S5_PROBE_KIND:
        return []           # only an S5 probe row has window measurements; every other row (controls children carry no `cell`) contributes none
    return [(k, s5_window_rel(k["eval_leg"], k["cell"], k["s"], stage=row["stage"], shard=row["shard"])) for k in (row.get("children") or [])]


def s5_probe_arm_of(row: Mapping[str, Any], child: Mapping[str, Any]) -> Dict[str, Any]:
    """The S5 measurement's identity for ONE probe child, from the ROW's own tables (never typed): the parent (`none`) or the probe
    cell -- its STAGED id (the `arm` the gate looks for: `multiturn_window_gate` requires `S5_PROBE_CELLS[*].staged_id` among the arms), its
    object sha and `arm_kind`."""
    cell = str(child["cell"])
    label = str(child.get("arm") or cell)                 # `dose_arm_runs`: the parent is labelled `<view>__none` (distinct per probe view), a dose cell by its staged id
    if cell == S5_PROBE_PARENT_ARM or cell.startswith("prompted__"):
        return {"arm": label, "arm_kind": "parent", "staged_id": cell, "object_sha256": None}
    o = (row.get("objects") or {}).get(cell)
    if not isinstance(o, Mapping) or not o.get("staged_id"):
        raise ValueError(f"{row.get('standup_id')}: child cell {cell!r} is not in the row's objects table -- the S5 measurement cannot name the object it measured")
    if label != str(o["staged_id"]):
        raise ValueError(f"{row.get('standup_id')}: child arm {label!r} is not the staged id {o['staged_id']!r} of its cell {cell!r} (the gate looks for the staged id)")
    return {"arm": str(o["staged_id"]), "arm_kind": "dosed", "staged_id": str(o["staged_id"]), "object_sha256": o.get("object_sha256")}


def s5_measure_argv(*, driver_python: str, tree: Path, root: Path, row: Mapping[str, Any], child: Mapping[str, Any], out: Path) -> List[str]:
    """The exact argv of `scripts/modal/seedrep_s5_measure.py` for ONE probe child (run in-container by `seedrep_served.producer_evidence`): the
    attempt root, the leg / cell / seed base naming WHICH store, the arm identity from the row's objects table, the decode row ref the child was
    invoked with (`--multiturn-target-policy-ref`, cross-checked by the producer against the store's own resolution record), the plan's goals
    per arm-run for this shard (`n_goals_expected`: the store's completeness is measured against it), the model key, the output path."""
    ident = s5_probe_arm_of(row, child)
    argv = list(child.get("argv") or [])
    ref = _flag_value(argv, "--multiturn-target-policy-ref") or MULTITURN_DECODE_POLICY_REF
    items = int(leg_table(S4_DEFAULTS)[child["eval_leg"]]["items_per_arm_run"])
    expected = -(-items // max(1, int(row.get("nshards") or 1)))
    out_argv = [driver_python, str(Path(tree) / PRODUCER_WRITERS["s5_window_measurement"]), "--root", str(root), "--leg", str(child["eval_leg"]),
                "--cell", str(child["cell"]), "--s", str(int(child["s"])), "--arm", ident["arm"], "--arm-kind", ident["arm_kind"],
                "--staged-id", ident["staged_id"], "--decode-policy-ref", str(ref), "--model-key", MODEL_KEY, "--n-goals-expected", str(expected),
                "--standup-id", str(row["standup_id"]), "--out", str(out)]
    if ident["object_sha256"]:
        out_argv += ["--object-sha256", str(ident["object_sha256"])]
    return out_argv


def audit_verdict_rels(eval_leg: str, arm: str, s: Any, *, stage: Any, shard: Any) -> List[Tuple[str, str]]:
    """Every verdict the audit writes for ONE child: ``[(table leg, audit_verdict_rel(table leg, ...)), ...]`` — the audit's output is per
    TABLE leg (`request_evidence_audit.audit_root`: `<out>/request_log/<table leg>/<arm>__s<s>…json`, verdict `leg` = the table leg), so a
    controls child is bound by THREE verdicts and never by a caller-composed `request_log/controls/…` file nobody writes."""
    return [(leg, audit_verdict_rel(leg, arm, s, stage=stage, shard=shard)) for leg in audit_table_legs(eval_leg, stage=stage)]


def smoke_audit_sources(row: Mapping[str, Any], root: str, producer: Optional[Mapping[str, Any]] = None) -> List[Tuple[str, Mapping[str, Any], str]]:
    """The `request_evidence_audit` sources of a row, ALIGNED: ``[(path under root, the child, the table leg the verdict is of), ...]`` — one
    per (child, table leg); the certificate compares each verdict's `leg` to the TABLE leg, `s` to the child's, and its `arm_binding` to the
    child's CELL. `producer` (the row's `producer_evidence` map, keyed by the plan's rel) names the FILE the audit actually filed for each
    binding (`bound_rel` — F-AO: the STORE's spelling, a harness id for a misalign cell, `arditi_allpos` for the Arditi row's
    `ar_c0p3__arditi_allpos` label); without it, or for a record without one, the path is the plan's own name."""
    out: List[Tuple[str, Mapping[str, Any], str]] = []
    for k in row.get("children") or []:
        for leg, rel in audit_verdict_rels(k["eval_leg"], k["arm"], k["s"], stage=row["stage"], shard=row["shard"]):
            b = producer.get(rel) if isinstance(producer, Mapping) else None
            bound = b.get("bound_rel") if isinstance(b, Mapping) and isinstance(b.get("bound_rel"), str) and b.get("bound_rel") else None
            out.append((f"{root}/{bound or rel}", k, leg))
    return out


def smoke_evidence_paths(row: Mapping[str, Any], root: str, producer: Optional[Mapping[str, Any]] = None) -> Dict[str, List[str]]:
    """Every check's canonical source path(s) under ``root`` (the accepted attempt root), from the row's contract — the audit's through the
    producer's bindings when `producer` is given (`smoke_audit_sources`, F-AO)."""
    return {"resolved_sampler": [f"{root}/{RESOLVED_SAMPLER_REL}"],
            "request_evidence_audit": [path for path, _k, _leg in smoke_audit_sources(row, root, producer)],
            "adapter_hashes_banked": [f"{root}/{CELLS_JSON_REL}"],
            "decisions": [f"{root}/modal_evidence.json"],
            "s5_window_measurement": [f"{root}/{rel}" for _k, rel in s5_window_rels(row)],
            # only a row that pins a pool has one; a row without a pool contributes no source (the check has nothing to bind there)
            "pool_endpoint": [f"{root}/{POOL_ENDPOINT_SNAPSHOT_REL}"] if row.get("pool") else [],
            # one determinism spot-check record per TABLE leg of the row (the app's `determinism_evidence`)
            "determinism": [f"{root}/{rel}" for _leg, rel in determinism_rels(row)]}


class RecordingGetter:
    """A `get(key)` over ONE accepted Dict that RECORDS what it read: the certificate carries `{dict, read_utc, entries,
    digest}` per smoke -- what the issuer saw in the LIVE Dict, never a snapshot an operator selected."""

    def __init__(self, name: str, get: Callable[[str], Any]) -> None:
        self.name, self._get, self.entries = name, get, {}
        self.read_utc: Optional[str] = None

    def __call__(self, key: str) -> Any:
        v = self._get(key)
        self.entries[str(key)] = v
        self.read_utc = _utc()
        return v

    get = __call__

    def record(self) -> Dict[str, Any]:
        return {"dict": self.name, "read_utc": self.read_utc, "entries": dict(sorted(self.entries.items())), "digest": accepted_reads_digest(self.entries)}


def accepted_reads_digest(entries: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(dict(sorted((str(k), v) for k, v in entries.items())), sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
RESULT_DONE_RE = re.compile(r"status=DONE\s+rc=0\s+attempt=(?P<aid>\S+)")
#: the W gate verdict is DERIVED by the certificate issuer from the two merged CPU-suite summaries (`cpu_suite.py`'s `summary.json`:
#: status, pytest_rc, counts, failures[].node) -- their shas recorded, `introduced` computed, both re-derived at consumption
W_GATE_SCHEMA = "seedrep_w_gate/2"
W_GATE_KEYS: Tuple[str, ...] = ("schema", "W", "baseline_sha", "introduced", "nodes_W", "nodes_baseline", "statuses", "summary_paths", "summary_sha256s", "derived_utc")
W_GATE_ADMISSIBLE_STATUSES: Tuple[str, ...] = ("PASS", "FAIL")            # TIMEOUT / a refused shard record is no verdict
DECODE_PROOF_SCHEMA = "seedrep_decode_ref_proof/1"
DECODE_PROOF_KEYS: Tuple[str, ...] = ("schema", "registry_path", "registry_sha256", "rows", "resolved_utc")
DECODE_ROW_PROOF_KEYS: Tuple[str, ...] = ("ref", "policy_id", "policy_ref", "policy_version", "superseded_by_policy_id", "is_default", "mode", "thinking_mode",
                                          "params_sent", "resolution")
DECODE_REGISTRY_REL = "configs/registry/decode_policy.json"
#: the EIGHT decode rows the battery's legs resolve by EXACT ref (`PolicyFile.by_ref` + `superseded_by_policy_id is None`, never
#: `resolve(mode=...)`) -- the `registry_plans.LEGS[*].policy_ref` set, pinned equal by test: static x4, BoN, over-refusal (XSTest /
#: OR-Bench), MMLU, GSM8K thinking-ON, misalign (agentic), multi-turn (PAIR / Crescendo / ActorAttack), Petri
BATTERY_DECODE_REFS: Dict[str, str] = {
    "static": f"{MODEL_KEY}_it|static|off|vendor_preset@v1",
    "bon": f"{MODEL_KEY}_it|bon|off|vendor_preset@v1",
    "over_refusal": f"{MODEL_KEY}_it|over_refusal|off|vendor_preset@v1",
    "mmlu": f"{MODEL_KEY}_it|mmlu|off|greedy@v1",
    "gsm8k": f"{MODEL_KEY}_it|gsm8k|on|thinking_card@v1",
    "agentic": f"{MODEL_KEY}_it|agentic|off|vendor@v1",
    "multi_turn": MULTITURN_DECODE_POLICY_REF,
    "petri": f"{MODEL_KEY}_it|petri|off|vendor_preset@v1",
}
assert len(set(BATTERY_DECODE_REFS.values())) == 8


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8") if isinstance(text, str) else bytes(text)).hexdigest()


def default_read_text(rs: Optional[Path]) -> Callable[[str], Optional[str]]:
    """There is NO operator-path reader for evidence any more -- every source is a `/vol/...` path read through the Volume API
    (the app's `certificate_read_text`). This default refuses every path (None = unreadable), so a caller
    that forgets to pass the volume reader gets refusals, never a local file."""
    def read(path: str) -> Optional[str]:
        return None
    return read


def volume_only(read: Callable[[str], Optional[str]]) -> Callable[[str], Optional[str]]:
    """Wrap a reader so that only `/vol/<rel>` paths are readable (an operator's local path is never evidence)."""
    def guarded(path: str) -> Optional[str]:
        return read(path) if str(path).startswith(VOL_PATH + "/") else None
    return guarded


def plan_row_by_id(plan: Mapping[str, Any], standup_id: str) -> Optional[Mapping[str, Any]]:
    for r in _plan_rows_the_app_may_execute(plan):
        if r["standup_id"] == standup_id:
            return r
    return None


def _smoke_row_facts(*, sid: str, row: Mapping[str, Any], identity: Mapping[str, Any], accepted_get: Callable[[str], Any],
                     read_text: Callable[[str], Optional[str]]) -> Dict[str, Any]:
    """The ACCEPTED attempt of one smoke row and the facts its root states: COMPLETE.json (its attempt id), RESULT (`status=DONE rc=0`),
    modal_evidence.json (readiness problems empty, the object verification ok, every child rc 0). Raises ValueError naming the refusal."""
    akey = accepted_key(identity["W7"], identity["wave_uuid"], row["leg"], row["engine"], row["stage"], row["shard"])
    aid = accepted_get(akey)
    if not isinstance(aid, str) or not aid:
        raise ValueError(f"{sid}: no ACCEPTED attempt under the smoke wave {identity['wave_uuid']} (accepted key {akey}) -- the smoke never completed its acceptance")
    root = attempt_root_for(identity["results_root"], leg=row["leg"], view=row["engine"], stage=row["stage"], shard=row["shard"], attempt_id=aid, kind=row.get("kind"))
    texts = {name: read_text(f"{root}/{name}") for name in SMOKE_ROW_FACT_FILES}
    missing = [n for n, v in texts.items() if v is None]
    if missing:
        raise ValueError(f"{sid}: the accepted attempt {aid} at {root} lacks {missing}")
    complete = json.loads(texts["COMPLETE.json"])
    if complete.get("attempt_id") != aid:
        raise ValueError(f"{sid}: COMPLETE.json names attempt {complete.get('attempt_id')!r}, the accepted Dict {aid!r}")
    m = RESULT_DONE_RE.search(texts["RESULT"])
    if not m or m.group("aid") != aid:
        raise ValueError(f"{sid}: RESULT is not `status=DONE rc=0 attempt={aid}`: {texts['RESULT'].strip()[:160]!r}")
    ev = json.loads(texts["modal_evidence.json"])
    problems = list((ev.get("readiness") or {}).get("problems") or [])
    if ev.get("attempt_id") != aid:
        raise ValueError(f"{sid}: modal_evidence.json names attempt {ev.get('attempt_id')!r}, not {aid!r}")
    if problems:
        raise ValueError(f"{sid}: readiness problems recorded on the accepted attempt: {problems[:3]}")
    if (ev.get("object_verification") or {}).get("ok") is not True:
        raise ValueError(f"{sid}: the accepted attempt's object verification is not ok")
    bad = [c for c in (ev.get("children") or []) if c.get("rc") not in (0, None)]
    if bad:
        raise ValueError(f"{sid}: {len(bad)} child(ren) of the accepted attempt exited non-zero: {bad[:2]}")
    producer = ev.get(PRODUCER_EVIDENCE_KEY)
    if not isinstance(producer, Mapping) or not producer:
        raise ValueError(f"{sid}: the accepted attempt {aid} bound no producer evidence (modal_evidence.json[{PRODUCER_EVIDENCE_KEY!r}]) -- the container never "
                         "ran the recorder / the audit over this root; nothing the operator supplies can stand in")
    return {"accepted_attempt": aid, "accepted_key": akey, "attempt_root": root,
            "sources": {name: {"path": f"{root}/{name}", "sha256": _sha_text(texts[name])} for name in SMOKE_ROW_FACT_FILES},
            "result_line": texts["RESULT"].strip(), "attested_max_num_seqs": ev.get("attested_max_num_seqs"), "n_children": len(ev.get("children") or []),
            "_producer": producer, "_evidence": ev, "_evidence_text": texts["modal_evidence.json"]}


def _producer_binding_problems(name: str, b: Mapping[str, Any], *, W: str) -> List[str]:
    """The producer's binding must be THE expected producer's -- the check's writer (`PRODUCER_WRITERS[name]`), the wave's tree
    (`tree_sha == W`) and a clean run (`rc == 0`); `operator.py` / `000…` / rc 99 are refused by name."""
    out = []
    if b.get("writer") != PRODUCER_WRITERS[name]:
        out.append(f"{name}: written by {b.get('writer')!r}, not the producer {PRODUCER_WRITERS[name]!r}")
    if b.get("tree_sha") != W:
        out.append(f"{name}: produced at tree {str(b.get('tree_sha'))[:12]!r}, not the wave's W {W[:12]}")
    if b.get("rc") != 0:
        out.append(f"{name}: the producer's tool exited rc {b.get('rc')!r}, not 0")
    return out


def _producer_binding(name: str, path: str, root: str, text: str, producer: Mapping[str, Any], *, W: str) -> Dict[str, Any]:
    """The PRODUCER's binding of one evidence file: `modal_evidence.json[producer_evidence][<rel>]` names its sha256, writer, tree sha and the tool's
    rc; a schema-valid file the producer did not write into THIS root, a file rewritten since, another writer, another tree or a failed tool is refused."""
    if not (isinstance(path, str) and path.startswith(root + "/")):
        raise ValueError(f"{name}: {path} is not under the accepted attempt root {root}")
    rel = path[len(root) + 1:]
    if rel in SMOKE_ROW_FACT_FILES:
        # a ROOT FACT (COMPLETE.json / RESULT / modal_evidence.json): the container's own record, re-hashed under rows[*].sources -- it cannot bind itself
        return {"writer": PRODUCER_WRITERS[name], "tree_sha": W, "rc": 0, "root_fact": True}
    b = producer.get(rel)
    if b is None:
        # F-AO: the audit's verdicts are keyed by the PLAN's rel and bound to the FILE the audit filed (`bound_rel`, the store's spelling)
        b = next((v for v in producer.values() if isinstance(v, Mapping) and v.get("bound_rel") == rel), None)
    if not isinstance(b, Mapping) or b.get("absent"):
        raise ValueError(f"{name}: the producer bound no {rel} in this attempt ({(b or {}).get('why') if isinstance(b, Mapping) else 'no entry'}) -- "
                         f"the file is not this attempt's evidence")
    if b.get("sha256") != _sha_text(text):
        raise ValueError(f"{name}: {rel} is not the bytes the producer bound (sha {_sha_text(text)[:12]} != bound {str(b.get('sha256'))[:12]})")
    problems = _producer_binding_problems(name, b, W=W)
    if problems:
        raise ValueError("; ".join(problems))
    return {"writer": b.get("writer"), "tree_sha": b.get("tree_sha"), "rc": b.get("rc")}


def _determinism_budget_facts(name: str, path: str, doc: Mapping[str, Any], *, row: Mapping[str, Any]) -> Dict[str, Any]:
    """The spot-check record's ADDITIVE selection facts -- `n_planned` / `n_selected` / `budget_limited` / `item_budget` -- read
    tolerantly (a record that predates them carries none: every fact null, `budget_limited` False) and, when present, held consistent:
    `budget_limited` a bool; a budget-limited record is a SMOKE row's (its `smoke_id` names a smoke whose budget is `item_budget`, its
    leg one that carries a budget flag), selects exactly `min(n_planned, item_budget)` and `n_items` is that selection -- a
    `budget_limited` record of a production row, or one whose counts disagree, refuses by name (a production row's short store must stay
    a defect signal). The `qualification` line restates the recorded fields, the way the audit's does."""
    present = [k for k in DETERMINISM_BUDGET_KEYS if k in doc]
    out: Dict[str, Any] = {"n_planned": doc.get("n_planned"), "n_selected": doc.get("n_selected"), "budget_limited": bool(doc.get("budget_limited", False)),
                           "item_budget": doc.get("item_budget"), "qualification": None}
    if not present:
        return out
    if missing := [k for k in DETERMINISM_BUDGET_KEYS if k not in doc]:
        raise ValueError(f"{name}: {path} carries {present} without {missing} -- D122's selection facts travel together")
    bl = doc.get("budget_limited")
    if not isinstance(bl, bool):
        raise ValueError(f"{name}: {path} records budget_limited {bl!r}, not a bool")
    n_planned, n_sel, ib = doc.get("n_planned"), doc.get("n_selected"), doc.get("item_budget")
    if not isinstance(n_planned, int) or isinstance(n_planned, bool) or n_planned <= 0 or not isinstance(n_sel, int) or isinstance(n_sel, bool) or n_sel <= 0:
        raise ValueError(f"{name}: {path} records n_planned {n_planned!r} / n_selected {n_sel!r} -- positive ints")
    if doc.get("n_items") != n_sel:
        raise ValueError(f"{name}: {path} records n_items {doc.get('n_items')!r} but n_selected {n_sel} -- the selection is what was spot-checked")
    if not bl:
        if ib is not None:
            raise ValueError(f"{name}: {path} records item_budget {ib!r} on a record that is not budget_limited")
        return out
    # budget-limited: ONLY a smoke row under the item budget, on a leg that carries a budget flag, selecting min(planned, budget)
    if not isinstance(ib, int) or isinstance(ib, bool) or ib <= 0:
        raise ValueError(f"{name}: {path} is budget_limited but records item_budget {ib!r}")
    sid_rec = doc.get("smoke_id")
    if sid_rec is None:
        raise ValueError(f"{name}: {path} is budget_limited but records no smoke_id -- a PRODUCTION row's spot-check is never budget-limited (D122: "
                         "its short store is a defect signal, and the check refuses it)")
    try:
        want = smoke_budget(str(sid_rec))
    except ValueError as e:
        raise ValueError(f"{name}: {path} is budget_limited under smoke id {sid_rec!r}: {e}")
    spec = SMOKE_BUDGET_FLAG_OF_LEG.get(str(row.get("leg")))
    if want["kind"] != SMOKE_BUDGET_KIND_ITEM or spec is None:
        raise ValueError(f"{name}: {path} is budget_limited but smoke {sid_rec} on the {row.get('leg')} leg carries no item budget "
                         f"(budget kind {want['kind']!r}; D107 budgets BoN / misalign rows of S1-S4 only)")
    if ib != int(want[spec[1]]):
        raise ValueError(f"{name}: {path} records item_budget {ib}, not the {want[spec[1]]} smoke {sid_rec} derives for the {row.get('leg')} leg (D107)")
    if n_sel != min(n_planned, ib):
        raise ValueError(f"{name}: {path} selected {n_sel} of {n_planned} planned under item budget {ib} -- a budget-limited row selects min(planned, budget)")
    out["qualification"] = f"budget_limited (D122): {n_sel} of {n_planned} planned items, the row's smoke item budget {ib} (D107)"
    return out


def _smoke_check(name: str, text: str, *, path: str, plan: Mapping[str, Any], accepted_attempts: Sequence[str], row: Mapping[str, Any], aid: str, root: str,
                 producer: Mapping[str, Any], child: Optional[Mapping[str, Any]] = None, table_leg: Optional[str] = None, store_root: Optional[str] = None) -> Dict[str, Any]:
    """ONE check's verdict from a PRODUCER-OWNED file's CONTENT: the file at its canonical path under the accepted attempt
    root, the producer's full record schema (writer's keys), its binding to THIS attempt (attempt id / standup id where the record carries one,
    the producer's sha), and only then the verdict. Never a verdict field alone, never an operator path. Raises ValueError when anything refuses."""
    try:
        doc = json.loads(text)
    except ValueError as e:
        raise ValueError(f"{name}: {path} is not JSON ({e})")
    if not isinstance(doc, Mapping):
        raise ValueError(f"{name}: {path} is not a record object")
    facts: Dict[str, Any] = {"path": path, "sha256": _sha_text(text), "producer": _producer_binding(name, path, root, text, producer, W=plan["W"])}
    if name == "resolved_sampler":
        missing = [k for k in RESOLVED_SAMPLER_KEYS if k not in doc]
        if missing:
            raise ValueError(f"{name}: {path} is not record_resolved_sampler.py's record (lacks {missing})")
        if doc.get("verdict") != "AGREES":
            raise ValueError(f"{name}: {path} verdict {doc.get('verdict')!r} is not AGREES ({str(doc.get('why'))[:160]})")
        facts.update(verdict="AGREES", n_client_requests=doc.get("n_client_requests"), reference_source=doc.get("reference_source"), class_rule=doc.get("class_rule"))
        # a record compared PER TASK (`--expect-tasks`) certifies only when EVERY task the row's clients sent AGREES -- an EMPTY task
        # (no request at its caps), an UNCOMPARED one (no reference resolved) or a DIVERGED one refuses BY TASK NAME; a record without the
        # per-task block (a single-sampler row under --expect) keeps the verdict rule above
        ctv = doc.get("client_task_verdicts")
        if ctv is not None:
            if not isinstance(ctv, Mapping) or not ctv:
                raise ValueError(f"{name}: {path} carries an empty client_task_verdicts -- a per-task record that compares no task certifies nothing")
            bad = {t: v for t, v in ctv.items() if v != "AGREES"}
            if bad:
                raise ValueError(f"{name}: {path} per-task verdict(s) are not AGREES: {bad} (D133: every task is held to ITS reference; EMPTY / UNCOMPARED / DIVERGED refuse by name)")
            facts["client_task_verdicts"] = dict(ctv)
    elif name == "pool_endpoint":
        missing = [k for k in POOL_ENDPOINT_KEYS if k not in doc]
        if missing:
            raise ValueError(f"{name}: {path} is not the pool Server's endpoint record (lacks {missing})")
        live = doc.get("liveness")
        if not isinstance(live, Mapping) or live.get("app_name") != row.get("pool") or doc.get("pool_name") != str(row.get("pool")).replace("seedrep-attacker-", "seedrep-"):
            raise ValueError(f"{name}: {path} advertises pool {(live or {}).get('app_name') if isinstance(live, Mapping) else None!r} / "
                             f"{doc.get('pool_name')!r}, not this row's pinned pool {row.get('pool')!r}")
        if doc.get("lane") != "seedrep" or not isinstance(doc.get("tensor_parallel_size"), int) or doc["tensor_parallel_size"] < 1:
            raise ValueError(f"{name}: {path} is not a seedrep pool record (lane {doc.get('lane')!r}, tensor_parallel_size {doc.get('tensor_parallel_size')!r})")
        # the record's three statements of the launched width must agree -- the top-level `max_num_seqs` (what the
        # Server passed to vLLM), `geometry.max_num_seqs` and `sizing.S` (the sizing block) -- so an operator override
        # (SEEDREP_ATTACKER_MAX_NUM_SEQS after a live coverage refusal) can never certify under the automatic default's arithmetic.
        # An older record has no `sizing`; its facts read None rather than refusing a carried row.
        launched = doc.get("max_num_seqs")
        geom = doc.get("geometry")
        if isinstance(geom, Mapping) and geom.get("max_num_seqs") not in (None, launched):
            raise ValueError(f"{name}: {path} launched max_num_seqs {launched!r} disagrees with geometry.max_num_seqs {geom.get('max_num_seqs')!r}")
        sizing = doc.get("sizing")
        if sizing is not None and not isinstance(sizing, Mapping):
            raise ValueError(f"{name}: {path} sizing is not an object")
        if isinstance(sizing, Mapping) and sizing.get("S") != launched:
            raise ValueError(f"{name}: {path} launched max_num_seqs {launched!r} disagrees with its sizing.S {sizing.get('S')!r}: the sizing block "
                             "describes another width (an operator override recorded under the automatic derivation)")
        facts.update({k: doc.get(k) for k in POOL_ENDPOINT_FACTS})
        facts.update(sizing_S=(sizing.get("S") if isinstance(sizing, Mapping) else None),
                     kv_rule=(sizing.get("kv_rule") if isinstance(sizing, Mapping) else None),
                     slots_cover_peak=(sizing.get("slots_cover_peak") if isinstance(sizing, Mapping) else None),
                     kv_resident_at=(sizing.get("kv_resident_at") if isinstance(sizing, Mapping) else None),
                     recommended_S=(((sizing.get("recommended") or {}).get("S")) if isinstance(sizing, Mapping) and isinstance(sizing.get("recommended"), Mapping) else None))
        facts["app_id"], facts["server_id"], facts["container_id"] = live.get("app_id"), live.get("server_id"), live.get("container_id")
    elif name == "request_evidence_audit":
        missing = [k for k in AUDIT_VERDICT_KEYS if k not in doc]
        if missing:
            raise ValueError(f"{name}: {path} is not request_evidence_audit.py's store verdict (lacks {missing})")
        if doc.get("attempt_id") != aid or doc.get("attempt_id") not in accepted_attempts:
            raise ValueError(f"{name}: {path} audits attempt {doc.get('attempt_id')!r}, not the accepted attempt {aid!r}")
        if doc.get("W") != plan["W"]:
            raise ValueError(f"{name}: {path} audits W {doc.get('W')!r}, not {plan['W']}")
        want_leg = table_leg or (child.get("eval_leg") if child is not None else None)      # the audit's verdict `leg` is the TABLE leg (F4)
        ab = doc.get("arm_binding")
        if child is not None:
            # The verdict is bound to the child through its VERIFIED arm binding -- the store's arm (the filename's spelling, preserved as
            # `doc["arm"]`) -> harness id / plan id / vmodel, resolved by the audit through the standup's records -- never a string compare of
            # `doc["arm"]` against the plan's ARM LABEL (a misalign cell is filed under its harness id, the Arditi cell under `arditi_allpos`
            # while its label is `ar_c0p3__arditi_allpos`). The child's CELL must be the binding's plan id (its harness id when the records
            # spell no plan id); a verdict without an arm binding is refused by name.
            if not isinstance(ab, Mapping) or not ab.get("harness_id") or not ab.get("vmodel"):
                raise ValueError(f"{name}: {path} carries no arm_binding (harness id / vmodel) -- a verdict written before F-AN; the store's arm "
                                 f"{doc.get('arm')!r} cannot be bound to this child's cell {child.get('cell') or child.get('arm')!r}")
            cell = str(child.get("cell") or child.get("arm"))
            bound_cell = ab.get("traintest_id") if ab.get("traintest_id") is not None else ab.get("harness_id")
            if (doc.get("arm") != ab.get("arm") or str(bound_cell) != cell or int(doc.get("s", -1)) != int(child.get("s")) or doc.get("leg") != want_leg):
                raise ValueError(f"{name}: {path} is the verdict of ({doc.get('leg')}, store arm {doc.get('arm')!r} -> harness {ab.get('harness_id')!r} / "
                                 f"TrainTest {ab.get('traintest_id')!r}, s{doc.get('s')}), not this child's ({want_leg}, cell {cell!r} (plan arm "
                                 f"{child.get('arm')!r}), s{child.get('s')})")
        if str(doc.get("stage")) != str(row["stage"]) or f"shard{int(row['shard']):02d}" != str(doc.get("shard")):
            raise ValueError(f"{name}: {path} audits stage {doc.get('stage')!r} shard {doc.get('shard')!r}, not this row's ({row['stage']}, shard{int(row['shard']):02d})")
        sd = str(doc.get("store_dir") or "")
        sroot = store_root or root          # fj7: a re-audited CARRIED root's verdicts live under the carried entry's evidence root, its stores under the historical root
        if not (sd.startswith(sroot + "/") or sd.startswith(sroot.split("/vol/", 1)[-1])):
            raise ValueError(f"{name}: {path} audits a store outside the accepted attempt root ({sd!r})")
        if doc.get("verdict") != "AGREES":
            raise ValueError(f"{name}: {path} verdict {doc.get('verdict')!r} is not AGREES ({str(doc.get('problem'))[:160]})")
        if doc.get("resolved_sampler_verdict") != "AGREES":
            raise ValueError(f"{name}: {path} reads the root's resolved sampler as {doc.get('resolved_sampler_verdict')!r}, not AGREES")
        rs_src = doc.get("resolved_sampler_source")
        if rs_src is not None:
            # fc11d: the audit read a RE-DERIVED sampler verdict (`--resolved-sampler`, an explicit input) -- admitted only when its provenance binds: re-derived
            # by THIS tree, and the sha of the bytes the audit read == the bytes bound as the evidence root's resolved_sampler.json (`producer`); the
            # historical verdict rides alongside and never stands in
            b = producer.get(RESOLVED_SAMPLER_REL) if isinstance(producer, Mapping) else None
            if not isinstance(rs_src, Mapping) or rs_src.get("rederived_by") != plan["W"]:
                raise ValueError(f"{name}: {path} took its sampler verdict from a record re-derived by {str((rs_src or {}).get('rederived_by') if isinstance(rs_src, Mapping) else rs_src)[:12]!r}, "
                                 f"not this tree {plan['W'][:12]}")
            if not isinstance(b, Mapping) or b.get("absent") or not b.get("sha256"):
                raise ValueError(f"{name}: {path} names a re-derived sampler source but the producer binds no {RESOLVED_SAMPLER_REL} under the evidence root")
            if Path(str(rs_src.get("path"))).name != RESOLVED_SAMPLER_REL or rs_src.get("sha256") != b["sha256"]:
                raise ValueError(f"{name}: {path} read sampler source {rs_src.get('path')!r} sha {str(rs_src.get('sha256'))[:12]}, not the bound {RESOLVED_SAMPLER_REL} "
                                 f"(sha {str(b['sha256'])[:12]}) under the evidence root -- the audit's sampler verdict is not the re-derived record's")
            facts.update(resolved_sampler_source=dict(rs_src), historical_resolved_sampler_verdict=doc.get("historical_resolved_sampler_verdict"))
        # the verdict's qualification is REQUIRED and CONSISTENT -- refused by name when absent (an older verdict), unknown or
        # contradictory -- then copied VERBATIM into the facts with its rendered line, so the certificate never lists a qualified AGREES bare
        qp = audit_qualification_problems(doc, f"{name}: {path}")
        if qp:
            raise ValueError("; ".join(qp))
        facts.update(verdict="AGREES", attempt_id=doc["attempt_id"], leg=doc.get("leg"), arm=doc.get("arm"), s=doc.get("s"), n_draws=doc.get("n_draws"),
                     n_draws_bound=doc.get("n_draws_bound"), store_complete=doc.get("store_complete"), resolved_sampler_verdict="AGREES",
                     # F-AO: the binding the verdict was accepted through, beside the plan child it was bound to, and the PLAN's path for that
                     # (child, table leg) -- what the record's coverage is counted on (`smoke_pass_problems`), the file itself being `path`
                     arm_binding=(dict(ab) if isinstance(ab, Mapping) else None),
                     plan_arm=(child.get("arm") if child is not None else None), plan_cell=(str(child.get("cell") or child.get("arm")) if child is not None else None),
                     plan_path=(f"{root}/{audit_verdict_rel(want_leg, child['arm'], child['s'], stage=row['stage'], shard=row['shard'])}"
                                if child is not None and want_leg is not None else path),
                     **{k: doc.get(k) for k in AUDIT_QUALIFICATION_KEYS},
                     independence_lost=(doc["reconciliation_contract"] in AUDIT_CONTRACTS_LEDGERLESS), qualification=audit_qualification_line(doc))
    elif name == "adapter_hashes_banked":
        serving, cells = doc.get("serving"), doc.get("cells")
        if not isinstance(serving, Mapping) or not isinstance(cells, list):
            raise ValueError(f"{name}: {path} is not the container's cells.json ({{cells: [...], serving: {{...}}}})")
        if serving.get("attempt_id") != aid or serving.get("standup_id") != row["standup_id"]:
            raise ValueError(f"{name}: {path} was served for attempt {serving.get('attempt_id')!r} / row {serving.get('standup_id')!r}, not {aid!r} / {row['standup_id']!r}")
        # Readiness audit: cells.json enumerates the served cells under HARNESS ids (`si_constv2_mix_full__psr_ce_all`; the app's writer:
        # "HARNESS ids, as served"; the S1 evidence's cells_served.json), the row's route_set under PLAN ids (`constv2-mix-full__psr_ce_all`)
        # -- the lookup TRANSLATES. A plan-keyed lookup found no hash for any fitted cell, refused every S1b row ("banks no intervention hash"),
        # and left the certificate without an S1b PASS.
        hashes = {c.get("cell_id"): c.get("intervention_hash") for c in cells if isinstance(c, Mapping)}
        want_cells = [c for c in (row.get("route_set") or []) if c != "none" and not str(c).startswith("prompted__")]
        served_hash = {c: hashes.get(harness_id(c)) for c in want_cells}
        missing = sorted(c for c in want_cells if not served_hash[c])
        if missing:
            raise ValueError(f"{name}: {path} banks no intervention hash for the row's cells {missing} "
                             f"(looked up as harness ids {[harness_id(c) for c in missing]}; served: {sorted(k for k in hashes if k)})")
        # the plan's adapter_hash_set is keyed by STAGED id (`base64s1__...`); the row's objects map each cell to its staged id, so the
        # disagreement guard keys through that map (a cell-id lookup into the staged-id map never matched: the guard was a silent no-op)
        planned = {a["staged_id"]: a.get("intervention_hash") for a in (plan.get("adapter_hash_set") or [])}
        staged_of = {c: ((row.get("objects") or {}).get(c) or {}).get("staged_id") for c in want_cells}
        disagree = sorted(c for c in want_cells if planned.get(staged_of[c]) and planned[staged_of[c]] != served_hash[c])
        if disagree:
            raise ValueError(f"{name}: {path} served hashes disagree with the plan's adapter_hash_set for {disagree}: "
                             f"served {[served_hash[c] for c in disagree]}, planned {[planned[staged_of[c]] for c in disagree]}")
        facts.update(n_banked=len([c for c in want_cells if served_hash[c]]), n_planned=len(want_cells), cells=dict(served_hash))
    elif name == "decisions":
        problems = list((doc.get("readiness") or {}).get("problems") or [])
        if doc.get("attempt_id") != aid or problems:
            raise ValueError(f"{name}: {path} names attempt {doc.get('attempt_id')!r} / readiness problems {problems[:2]}")
        win: Dict[str, int] = {}
        for leg in S5_PROBE_LEGS:
            rows_ = [r for r in list(plan.get("standups") or []) if r.get("leg") == leg]
            L = rows_[0]["geometry"]["L"] if rows_ and isinstance(rows_[0].get("geometry"), Mapping) else None
            if not isinstance(L, int) or isinstance(L, bool) or L <= 0:
                raise ValueError(f"{name}: the certified plan decides no window for {leg}")
            win[leg] = int(L)
        facts.update(multiturn_window=win, n_children=len(doc.get("children") or []))
    elif name == "s5_window_measurement":
        # the producer's window record over THIS child's store -- the full schema (`window_measurement_problems`), bound to THIS attempt
        # (`attempt_id` + `produced_by`), naming THIS child's (leg, cell, s), a COMPLETE store and ZERO clamped calls; anything else refuses by name
        wp = window_measurement_problems(doc)
        if wp:
            raise ValueError(f"{name}: {path} is not a {S5_WINDOW_MEASUREMENT_SCHEMA} record: {wp}")
        missing = [k for k in S5_WINDOW_MEASUREMENT_EXTRA_KEYS if k not in doc]
        if missing:
            raise ValueError(f"{name}: {path} is not seedrep_s5_measure.py's record (lacks {missing})")
        pb = doc.get("produced_by")
        if not isinstance(pb, Mapping) or [k for k in PRODUCED_BY_KEYS if k not in pb]:
            raise ValueError(f"{name}: {path} carries no produced_by {PRODUCED_BY_KEYS} -- not a producer's record")
        if doc.get("attempt_id") != aid or pb.get("attempt_id") != aid or doc.get("attempt_id") not in accepted_attempts:
            raise ValueError(f"{name}: {path} measures attempt {doc.get('attempt_id')!r} / produced under {pb.get('attempt_id')!r}, not the accepted attempt {aid!r}")
        if pb.get("tree_sha") != plan["W"] or pb.get("writer") != PRODUCER_WRITERS[name] or pb.get("standup_id") != row["standup_id"]:
            raise ValueError(f"{name}: {path} was produced by {pb.get('writer')!r} at tree {str(pb.get('tree_sha'))[:12]!r} for row {pb.get('standup_id')!r}, "
                             f"not {PRODUCER_WRITERS[name]!r} at {plan['W'][:12]} for {row['standup_id']!r}")
        if child is not None:
            want = s5_probe_arm_of(row, child)
            got = (doc.get("leg"), doc.get("cell"), int(doc.get("s", -1)), doc.get("arm"), doc.get("arm_kind"), doc.get("staged_id"))
            exp = (child["eval_leg"], child["cell"], int(child["s"]), want["arm"], want["arm_kind"], want["staged_id"])
            if got != exp:
                raise ValueError(f"{name}: {path} measures {got}, not this child's {exp}")
            if want["object_sha256"] and doc.get("object_sha256") != want["object_sha256"]:
                raise ValueError(f"{name}: {path} measures object {doc.get('object_sha256')!r}, not the row's {want['object_sha256']}")
        sp = str(doc.get("store_path") or "")
        if sp.startswith("/") and not (sp.startswith(root + "/") or sp.startswith(root.split("/vol/", 1)[-1])):
            raise ValueError(f"{name}: {path} measures a store outside the accepted attempt root ({sp!r})")
        if doc.get("store_complete") is not True or doc.get("complete") is not True:
            raise ValueError(f"{name}: {path}: the store is not COMPLETE ({doc.get('n_goals_completed')}/{doc.get('n_goals_expected')} goals; "
                             f"{doc.get('completeness_note')}) -- every early reading is a floor, never a maximum")
        cc = doc.get("clamped_calls")
        if not isinstance(cc, int) or isinstance(cc, bool) or cc != 0:
            raise ValueError(f"{name}: {path}: {cc!r} clamped call(s) -- the window is too small for this cell (D50 requires zero)")
        facts.update(leg=doc["leg"], arm=doc["arm"], arm_kind=doc["arm_kind"], cell=doc["cell"], s=doc["s"], context_max=doc["context_max"], clamped_calls=0,
                     n_calls=doc["n_calls"], store_complete=True, attempt_id=doc["attempt_id"], wave_uuid=doc["wave_uuid"], measurement=dict(doc))
    elif name == "determinism":
        # the determinism spot-check record for THIS row's table leg -- the writer's full schema, the leg the path names, counts > 0,
        # every re-sent request byte-identical (`reproduces`, cross-checked against its own counts), and -- when the record spells the row it is
        # OF (standup_id / stage / shard / smoke_id) -- THIS row, in the plan's production id or its `smoke-<S>__` spelling
        missing = [k for k in DETERMINISM_RECORD_KEYS if k not in doc]
        if missing:
            raise ValueError(f"{name}: {path} is not {PRODUCER_WRITERS[name]}'s record (lacks {missing})")
        if table_leg is not None and doc.get("leg") != table_leg:
            raise ValueError(f"{name}: {path} is the record of leg {doc.get('leg')!r}, not this row's table leg {table_leg!r}")
        n_req, n_id = doc.get("n_requests"), doc.get("identical")
        if not isinstance(n_req, int) or isinstance(n_req, bool) or n_req <= 0 or doc.get("n") != n_req:
            raise ValueError(f"{name}: {path} re-sent {n_req!r} request(s) (n {doc.get('n')!r}) -- a spot-check with nothing re-sent proves nothing")
        if not isinstance(n_id, int) or isinstance(n_id, bool) or n_id != n_req or doc.get("reproduces") is not True:
            raise ValueError(f"{name}: {path}: {n_id!r} of {n_req} re-sent request(s) reproduced byte-identically (reproduces={doc.get('reproduces')!r}; "
                             f"misses {str(doc.get('misses'))[:160]}) -- criterion 3 requires every one")
        if not isinstance(doc.get("arms"), list) or not doc["arms"]:
            raise ValueError(f"{name}: {path} names no arm it re-sent for")
        sid_rec = doc.get("standup_id")
        if sid_rec is not None and sid_rec != row["standup_id"] and not (str(sid_rec).startswith("smoke-") and str(sid_rec).endswith(f"__{row['standup_id']}")):
            raise ValueError(f"{name}: {path} is the record of row {sid_rec!r}, not this row {row['standup_id']!r}")
        if "stage" in doc and str(doc.get("stage")) != str(row["stage"]):
            raise ValueError(f"{name}: {path} records stage {doc.get('stage')!r}, not this row's {row['stage']!r}")
        if "shard" in doc and doc.get("shard") is not None and str(doc.get("shard")).lstrip("shard").lstrip("0") != str(int(row["shard"])).lstrip("0"):
            raise ValueError(f"{name}: {path} records shard {doc.get('shard')!r}, not this row's {row['shard']!r}")
        budget_facts = _determinism_budget_facts(name, path, doc, row=row)
        facts.update(leg=doc.get("leg"), reproduces=True, n_requests=n_req, identical=n_id, n_items=doc.get("n_items"), n_arms=len(doc["arms"]),
                     standup_id=sid_rec, smoke_id=doc.get("smoke_id"), **budget_facts)
    else:
        raise ValueError(f"unknown smoke check {name!r} (one of {SMOKE_CHECK_NAMES})")
    return facts


def _row_sampler_check(check: Any, *, root: str, reaudit: bool) -> Dict[str, Any]:
    """The CURRENT row's resolved-sampler sources out of a smoke-wide check. A smoke PASS record's `resolved_sampler` check lists
    one source per ROW (each row's own `resolved_sampler.json` -- under its attempt root on the same-tree path, under its re-audit evidence root
    on the prior path), and `verified_resolved_decode` was written for ONE row's check: handed the whole smoke's, it read every row's
    temperatures as the row's decode (a greedy sibling made a sampled gsm8k row read greedy; two sampled prior rows at 0.6 and 0.7 failed the
    historical cross-check against each row's own [t]). So the qualification is judged on the row's EXACT sources: those whose path is under
    ``root`` (the row's evidence root when ``reaudit`` -- the fresh re-derived record is the authority, the historical one cross-checked -- else
    its attempt root), the same bytes the resolved_sampler check verified and hashed. No source under the root -> the decode is unknown and
    the row is refused by name, never another row's decode."""
    srcs = (check.get("sources") if isinstance(check, Mapping) else None) or []
    own = [s_ for s_ in srcs if isinstance(s_, Mapping) and isinstance(s_.get("path"), str) and s_["path"].startswith(root.rstrip("/") + "/")]
    out: Dict[str, Any] = {"ok": True, "sources": own, "row_root": root}
    if reaudit:
        out["reaudit"] = {"held_to_plan": "this"}
    return out


def _determinism_source(text: str, *, path: str, plan: Mapping[str, Any], accepted_attempts: Sequence[str], row: Mapping[str, Any], aid: str, root: str,
                        producer: Mapping[str, Any], table_leg: Optional[str], sampler_check: Any, evidence: Mapping[str, Any],
                        read_text: Optional[Callable[[str], Optional[str]]], prev_row: Optional[Mapping[str, Any]] = None,
                        store_root: Optional[str] = None) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """ONE determinism source's facts, the carry-over's qualification applied FIRST
    (`seedrep_carry_over.determinism_d180_relaxation`, the one function `complete_verdict` uses). A record that reproduces, or a diverged one on a
    GREEDY row, goes through `_smoke_check` as written (the byte-identical rule; a greedy divergence refuses by name). A diverged record on a
    SAMPLED row -- the decode read from the resolved-sampler record the row's `resolved_sampler` check VERIFIED (`sampler_check`), cross-checked
    against the historical one under a re-audit -- runs the check over the relaxed text and stamps the REAL facts back: `sha256` the real bytes',
    `reproduces` False, `identical` the real count, `criterion3` = :data:`DETERMINISM_CRITERION3_NOT_APPLICABLE_SAMPLED`, `criterion3_ruling` D180,
    `criterion3_decode` the verified temperatures / authority / source. Returns ``(facts, qualification-or-None)``; the qualification aggregates into
    the check's `criterion3` block (`_criterion3_aggregate`). Without ``read_text`` the exemption cannot be judged and the check runs as written.
    An exemption that cannot be judged (no verified decode; bytes not the producer's) raises ValueError by name."""
    if read_text is not None:
        co = _carry_over_module()
        try:
            relax = co.determinism_d180_relaxation(text, path=path, root=root, producer=producer, sampler_check=sampler_check, prev_row=prev_row or row,
                                                   evidence=evidence, read_text=read_text)
        except co.D180Undeterminable as e:
            raise ValueError(f"determinism: {e}")
        if relax is not None:
            facts = _smoke_check("determinism", relax["text"], path=path, plan=plan, accepted_attempts=accepted_attempts, row=row, aid=aid, root=root,
                                 producer=relax["producer"], table_leg=table_leg, store_root=store_root)
            d = relax["diverged"]
            facts.update(sha256=relax["real_sha"], reproduces=False, identical=d["reproduced"], criterion3=DETERMINISM_CRITERION3_NOT_APPLICABLE_SAMPLED,
                         criterion3_ruling=d["ruling"],
                         criterion3_decode={"temperatures": list(d["decode"]["temperatures"]), "authority": d["decode"]["authority"], "source": list(d["decode"]["source"])})
            return facts, {**d, "records": [path]}
    return _smoke_check("determinism", text, path=path, plan=plan, accepted_attempts=accepted_attempts, row=row, aid=aid, root=root, producer=producer,
                        table_leg=table_leg, store_root=store_root), None


def _criterion3_aggregate(quals: Sequence[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """The check-level `criterion3` block of a smoke PASS record (fg1 / F-AW; the carry-over's shape): the qualifications of every diverged
    sampled source of the check, summed -- `reproduced` / `resent` / `misses` / `records` -- under the one verdict and ruling; None when none."""
    if not quals:
        return None
    out: Dict[str, Any] = {k: copy.deepcopy(v) for k, v in quals[0].items() if k != "record"}
    for q in quals[1:]:
        out["reproduced"] += q["reproduced"]
        out["resent"] += q["resent"]
        out["misses"] = out["misses"] + list(q["misses"])
        out["records"] = out["records"] + list(q["records"])
        # every row is qualified on ITS OWN decode, so the block's decode is the UNION over the qualified rows -- the distinct
        # temperatures (each > 0), every row's verified source, every row's historical cross-check (None when any row had none)
        d, qd = out["decode"], q["decode"]
        d["temperatures"] = sorted({float(t) for t in [*d["temperatures"], *qd["temperatures"]]})
        d["source"] = list(d["source"]) + list(qd["source"])
        if d.get("authority") != qd.get("authority"):
            d["authority"] = sorted({str(d.get("authority")), str(qd.get("authority"))})
        if isinstance(d.get("cross_check"), Mapping) and isinstance(qd.get("cross_check"), Mapping):
            d["cross_check"] = {"historical": list(d["cross_check"].get("historical") or []) + list(qd["cross_check"].get("historical") or []),
                                "agrees": bool(d["cross_check"].get("agrees")) and bool(qd["cross_check"].get("agrees"))}
        else:
            d["cross_check"] = None
    return out


def _criterion3_block_problems(who: str, c: Mapping[str, Any], srcs: Sequence[Mapping[str, Any]], fresh_quals: Optional[Sequence[Mapping[str, Any]]]) -> List[str]:
    """The check-level `criterion3` block against its sources -- present iff a source carries the qualification, its records exactly those
    sources' paths, its counts their sums, the exemption's verdict and ruling, every temperature > 0; with the fresh derivation
    (``fresh_quals``, read from the volume) the block must equal the re-derived aggregate."""
    out: List[str] = []
    c3 = c.get("criterion3")
    qual_srcs = [s for s in srcs if s.get("criterion3") is not None]
    if fresh_quals is not None:
        want = _criterion3_aggregate(fresh_quals)
        if c3 != want:
            out.append(f"{who}: criterion3 block {str(c3)[:160]!r} is not the re-derived D180 qualification {str(want)[:160]!r}")
        return out
    if c3 is None and qual_srcs:
        out.append(f"{who}: {len(qual_srcs)} source(s) carry the D180 qualification but the check records no criterion3 block")
    elif c3 is not None:
        if not qual_srcs:
            out.append(f"{who}: the check records a criterion3 block but no source carries the qualification")
        elif not isinstance(c3, Mapping) or c3.get("verdict") != DETERMINISM_CRITERION3_NOT_APPLICABLE_SAMPLED or c3.get("ruling") != DETERMINISM_CRITERION3_RULING:
            out.append(f"{who}: criterion3 block is not D180's not_applicable_sampled_decode ({str(c3)[:120]!r})")
        else:
            if sorted(c3.get("records") or []) != sorted(str(s.get("path")) for s in qual_srcs):
                out.append(f"{who}: criterion3 records {c3.get('records')!r} are not the qualified sources' paths")
            if c3.get("reproduced") != sum(int(s.get("identical") or 0) for s in qual_srcs) or c3.get("resent") != sum(int(s.get("n_requests") or 0) for s in qual_srcs):
                out.append(f"{who}: criterion3 counts reproduced {c3.get('reproduced')!r} / resent {c3.get('resent')!r} are not the qualified sources' sums")
            temps = ((c3.get("decode") or {}).get("temperatures") if isinstance(c3.get("decode"), Mapping) else None)
            if not isinstance(temps, list) or not temps or not all(isinstance(t, (int, float)) and not isinstance(t, bool) and t > 0 for t in temps):
                out.append(f"{who}: criterion3 rests on decode temperatures {temps!r}, not a sampled decode (every temperature > 0)")
    return out


def derive_smoke_pass(*, rs: Path, plan: Mapping[str, Any], smoke_id: str, standup_ids: Sequence[str], accepted_get: Callable[[str], Any],
                      read_text: Callable[[str], Optional[str]], derived_utc: Optional[str] = None) -> Dict[str, Any]:
    """S1-S4: the PASS record DERIVED from what the wave produced -- the smoke wave from its minted file, every
    named row's ACCEPTED attempt (the LIVE accepted Dict) and its root's facts (COMPLETE.json / RESULT / modal_evidence.json off the volume),
    and every required check evaluated on the PRODUCER-OWNED file at its canonical path under that root (`smoke_evidence_paths`), bound by the
    producer's sha. No operator path exists. Raises ValueError (no record is ever written for a failing smoke)."""
    if smoke_id not in SMOKE_DERIVABLE_IDS:
        raise ValueError(f"smoke id {smoke_id!r} is not one of {SMOKE_DERIVABLE_IDS}")
    wave = smoke_execution_wave(rs, smoke_id)
    if wave is None:
        raise ValueError(f"no {smoke_wave_path(rs, smoke_id)}: smoke {smoke_id} never ran (its execution wave is minted at --smoke)")
    identity = smoke_identity(plan, smoke_id, wave)
    ids = [str(s) for s in standup_ids]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError(f"{smoke_id}: name the plan rows the smoke ran (non-empty, distinct standup ids)")
    rows: Dict[str, Dict[str, Any]] = {}
    plan_rows: Dict[str, Mapping[str, Any]] = {}
    for sid in sorted(ids):
        row = plan_row_by_id(plan, sid)
        if row is None:
            raise ValueError(f"{smoke_id}: {sid} is not a row of this plan")
        if smoke_id == S5_WINDOW_SMOKE_ID and row.get("kind") != S5_PROBE_KIND:
            raise ValueError(f"{smoke_id}: {sid} is not an S5 probe row (kind {row.get('kind')!r}) -- only the D45 probe standups measure the window")
        plan_rows[sid] = row
        rows[sid] = _smoke_row_facts(sid=sid, row=row, identity=identity, accepted_get=accepted_get, read_text=read_text)
    accepted_attempts = [r["accepted_attempt"] for r in rows.values()]
    checks: Dict[str, Any] = {}
    for name in SMOKE_CHECKS[smoke_id]:
        sources = []
        quals: List[Dict[str, Any]] = []
        for sid, r in rows.items():
            root, row = r["attempt_root"], plan_rows[sid]
            producer = producer_map_for(name, r["_evidence"])       # the determinism check's bindings live under their own evidence key
            paths = smoke_evidence_paths(row, root, producer)[name]      # F-AO: the audit's verdicts at the FILES the producer bound
            # the audit's sources come ALIGNED to (child, table leg) -- a controls child has three (F4); every other check has no child
            audit_src = {path_: (k_, leg_) for path_, k_, leg_ in smoke_audit_sources(row, root, producer)} if name == "request_evidence_audit" else {}
            # the S5 measurements come ALIGNED to their child too (one record per probe child: the `none` parent and the probe cell)
            if name == "s5_window_measurement":
                audit_src = {f"{root}/{rel}": (k_, k_["eval_leg"]) for k_, rel in s5_window_rels(row)}
            # the spot-check's records come ALIGNED to their TABLE leg (one per leg of the row; no child)
            if name == "determinism":
                audit_src = {f"{root}/{rel}": (None, leg_) for leg_, rel in determinism_rels(row)}
            if not paths:
                if name == "pool_endpoint" and not row.get("pool"):
                    continue        # F-M: only a pool-pinned row has a snapshot; coverage is enforced by the pool-pinned count check below
                raise ValueError(f"{smoke_id}: check {name!r}: row {sid} names no producer file to read (no children)")
            for path in paths:
                if not path.startswith(identity["results_root"] + "/"):
                    raise ValueError(f"{smoke_id}: {path} is not under the smoke's results root {identity['results_root']}")
                text = read_text(path)
                if text is None:
                    raise ValueError(f"{smoke_id}: check {name!r}: the producer's {path} is absent on the volume")
                child, table_leg = audit_src.get(path, (None, None))
                if name == "determinism":
                    # the sampled-decode qualification (a sampled row's diverged replay) applied through the carry-over's one function
                    facts_d, qual = _determinism_source(text, path=path, plan=plan, accepted_attempts=accepted_attempts, row=row, aid=r["accepted_attempt"], root=root,
                                                        producer=producer, table_leg=table_leg,
                                                        sampler_check=_row_sampler_check(checks.get("resolved_sampler"), root=root, reaudit=False),      # fg2 / P2-2: THIS row's sources
                                                        evidence=r["_evidence"], read_text=read_text)
                    sources.append(facts_d)
                    if qual is not None:
                        quals.append(qual)
                    continue
                sources.append(_smoke_check(name, text, path=path, plan=plan, accepted_attempts=accepted_attempts, row=row, aid=r["accepted_attempt"], root=root,
                                            producer=producer, child=child, table_leg=table_leg))
        checks[name] = {"ok": True, "sources": sources}
        if name == "determinism" and quals:
            checks[name]["criterion3"] = _criterion3_aggregate(quals)          # fg1 / F-AW: the qualification, surfaced on the check as the carry-over does
        if name == "request_evidence_audit":
            # the check states its reconciliation split (ledger-backed vs ledger-less) above the per-verdict facts; re-derived at certify
            checks[name]["reconciliation"] = audit_reconciliation_tally(sources)
    if smoke_id == S5_WINDOW_SMOKE_ID:
        # for EVERY probe leg named, a parent-arm measurement AND both probe cells, each with zero clamped calls (asserted per source above)
        srcs = checks["s5_window_measurement"]["sources"]
        for leg in sorted({str(s_["leg"]) for s_ in srcs}):
            arms = {str(s_["arm"]) for s_ in srcs if s_["leg"] == leg}
            kinds = {str(s_["arm_kind"]) for s_ in srcs if s_["leg"] == leg}
            missing = [c["staged_id"] for c in S5_PROBE_CELLS if c["staged_id"] not in arms]
            if missing or "parent" not in kinds:
                raise ValueError(f"{smoke_id}: {leg}: the named rows measure arms {sorted(arms)}; D45 requires the parent arm AND both probe cells "
                                 f"{[c['staged_id'] for c in S5_PROBE_CELLS]} (missing {missing}{'; no parent arm' if 'parent' not in kinds else ''})")
    for r in rows.values():
        r.pop("_producer", None)
        r.pop("_evidence", None)
        r.pop("_evidence_text", None)
    return {"schema": SMOKE_PASS_SCHEMA, "smoke_id": smoke_id, "verdict": "PASS", "rc": 0, "W": plan["W"], "execution_wave_uuid": wave,
            "production_wave_uuid": plan["wave_uuid"], "launch_plan_sha": plan.get("launch_plan_sha"), "standup_ids": sorted(rows),
            "accepted": {sid: r["accepted_attempt"] for sid, r in rows.items()}, "rows": rows, "checks": checks, "derived_utc": derived_utc or _utc()}


def derive_s5_window_measurements(*, rs: Path, plan: Mapping[str, Any], standup_ids: Sequence[str], accepted_get: Callable[[str], Any],
                                  read_text: Callable[[str], Optional[str]], derived_utc: Optional[str] = None) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """The window measurements DERIVED from what the probe wave produced -- `derive_smoke_pass` for `S5_WINDOW_SMOKE_ID` over the named probe rows
    (the accepted attempt per row off the LIVE Dict, its root facts, the producer's bound `audits/s5_window/...` record per child, both probe cells
    + a parent per leg, zero clamped calls) -> ``(the derivation record, the measurement records)``. The measurement records are exactly the
    producer's files' content (they carry `produced_by`); `seedrep_s5_record.py window` writes them into `S5.multiturn[]` and the derivation
    under `S5.window_derivation`. No operator record exists on this path."""
    rec = derive_smoke_pass(rs=rs, plan=plan, smoke_id=S5_WINDOW_SMOKE_ID, standup_ids=standup_ids, accepted_get=accepted_get, read_text=read_text,
                            derived_utc=derived_utc)
    measurements: List[Dict[str, Any]] = []
    for src in rec["checks"]["s5_window_measurement"]["sources"]:
        m = src.pop("measurement")
        measurements.append(m)
    return rec, measurements


# ---- S1-S3 PASS records DERIVED from PRIOR trees' accepted smoke attempts, re-audited by THIS tree ------------------------------------------
#: The smokes completed on earlier trees (S1 / S1b / S2, S2's controls row, S3's Petri row) and this tree's smoke Dicts are empty on a new tree,
#: so `derive_smoke_pass` has nothing to resolve. Mirroring the reference admission (`s5_reference_admission`: a measurement under another tree
#: is admitted through its standup's records, `measured_under_W` stamped) and the carry-over's re-audit (`complete_verdict` with `reaudit=`), a
#: PASS record may draw its rows from PRIOR archives (`RS/prior/<W>/`, the carry-over's `load_prior_source` layout): each row's accepted attempt
#: is resolved through the PRIOR plan's smoke Dict, its identity (`smoke_row_identity`) must equal THIS plan's row's, the request-evidence audit
#: is RE-RUN by this tree over the historical stores (the carry-over's `reaudit` callable; its verdicts read from the carried evidence root, held
#: to THIS plan), every other producer file is read at the historical root and held to the PRIOR tree. The record's `W` is THIS plan's (the tree
#: whose readers derived it); the prior trees are named per row and at the top. S4 is untouched (the controls-evidence waiver).
SMOKE_PASS_PRIOR_KEYS: Tuple[str, ...] = ("measured_under_W", "measured_under_wave", "reaudited_by", "priors")
SMOKE_PASS_PRIOR_ROW_KEYS: Tuple[str, ...] = ("measured_under_W", "measured_under_wave", "prior_rs", "accepted_dict", "identity", "identity_sha256", "reaudit", "sampler",
                                              "arms_audited", "arms_planned", "smoke_scope", "served_standup_id", "qualifications")
#: like the deferral of the controls determinism replay to post-launch: for a PRIOR row, a check whose producer artefact is ABSENT at the historical
#: root and cannot be re-derived offline by this tree's readers is recorded as `not_produced_at_prior_W` -- a named QUALIFICATION on the row record
#: (it rides into the certificate's smokes block), never a PASS of that check; the check still refuses when the artefact EXISTS and disagrees. ONLY
#: the checks named here are admitted under the rule (the prior bon smoke's determinism replay was never produced: rc 2); any other absent artefact
#: still refuses by name.
PRIOR_NOT_PRODUCED_STATUS = "not_produced_at_prior_W"
PRIOR_NOT_PRODUCED_ADMITTED: Dict[str, str] = {
    "determinism": "D178: the determinism spot-check is a producer-time REPLAY artefact (determinism_check.py against the live engine); absent at the prior root it "
                   "cannot be re-derived offline -- recorded as a qualification, mirroring D135 (the controls determinism replay deferred post-launch by fc5)"}
SMOKE_PASS_PRIOR_REAUDIT_KEYS: Tuple[str, ...] = ("reaudited_by", "evidence_root", "store_root", "rc", "argv", "producer_evidence", "sampler", "stores_present", "stores_problems")
SMOKE_SCOPE_FULL, SMOKE_SCOPE_SUBSET = "full", "subset"
SMOKE_PASS_PRIOR_BASIS = "prior_W_reaudited_fc11"
#: the checks a prior record reads from the RE-AUDIT EVIDENCE ROOT (this tree's producers over the historical root, held to THIS plan) rather than
#: from the historical root held to the prior tree: the request-evidence audit and the resolved-sampler verdict -- the earlier recorder read
#: UNCOMPARED on every two-sampler row (`--expect-tasks` fixed CURRENT rows only) while the engine log resolves every client request to the preset,
#: so the verdict is re-derived by this tree's recorder (`seedrep_served.rederive_sampler_verdict`) and the PASS held to THAT
SMOKE_PASS_REDERIVED_CHECKS: Tuple[str, ...] = ("request_evidence_audit", "resolved_sampler")
SMOKE_PASS_SAMPLER_KEYS: Tuple[str, ...] = ("historical_verdict", "historical_path", "rederived_verdict", "rederived_by", "rederived_path", "why", "probe")
#: fc11d: what the re-derivation is BOUND by -- the sha256 of the tools' generated outputs (expected_samplers.json, resolved_sampler.json), and the
#: audit's `resolved_sampler_source` sha of the bytes it read; the pulled engine log's bytes themselves are NOT hashed anywhere in the stamps
SMOKE_PASS_SAMPLER_BASIS = ("re-derived by this tree's expected-samplers helper + record_resolved_sampler.py over the pulled root's engine log; bound by the sha256 "
                            "of the GENERATED outputs (expected_samplers.json, resolved_sampler.json) and by the audit's resolved_sampler_source sha of the bytes it "
                            "read -- the engine log's bytes are not hashed (no log-byte verification is claimed)")


def smoke_row_identity(row: Mapping[str, Any]) -> Dict[str, Any]:
    """fc11: the fields a PRIOR tree's smoke row must share with THIS plan's row to stand in for it -- leg, view (engine), stage, shard, kind, the seed
    bases, the engine geometry (L / S / thinking), the served objects (cell -> artifact sha, the route set) and the children's (eval leg, arm, cell,
    s). Everything W- / wave-spelled (argv, control / accepted keys, roots) is NOT identity: it differs between trees by construction."""
    geo = row.get("geometry") if isinstance(row.get("geometry"), Mapping) else {}
    objects = row.get("objects") if isinstance(row.get("objects"), Mapping) else {}
    return {"leg": row.get("leg"), "view": row.get("engine"), "stage": row.get("stage"), "shard": int(row["shard"]) if row.get("shard") is not None else None,
            "kind": row.get("kind"), "seed_bases": sorted(int(s) for s in (row.get("seed_bases") or [])),
            "geometry": {"L": geo.get("L"), "S": geo.get("S"), "thinking": geo.get("thinking")},
            "route_set": sorted(str(c) for c in (row.get("route_set") or [])),
            "objects": {str(c): ((v or {}).get("object_sha256") if isinstance(v, Mapping) else None) for c, v in sorted(objects.items())},
            "children": sorted(({"eval_leg": k.get("eval_leg"), "arm": k.get("arm"), "cell": str(k.get("cell") or k.get("arm")), "s": k.get("s")} for k in (row.get("children") or [])),
                               key=lambda k_: json.dumps(k_, sort_keys=True, default=str))}


def smoke_row_identity_sha(identity: Mapping[str, Any]) -> str:
    return _sha_text(json.dumps(identity, sort_keys=True, default=str))


def smoke_row_identity_problems(prior_identity: Any, this_identity: Mapping[str, Any], *, who: str) -> List[str]:
    """Why a prior row does not bind to this plan's row, by field name (never a bare 'mismatch')."""
    if not isinstance(prior_identity, Mapping):
        return [f"{who}: no recorded identity of the prior row"]
    out = []
    for k in sorted(set(prior_identity) | set(this_identity)):
        if prior_identity.get(k) != this_identity.get(k):
            out.append(f"{who}: identity field {k!r} differs -- prior {json.dumps(prior_identity.get(k), sort_keys=True, default=str)[:160]}, "
                       f"this plan {json.dumps(this_identity.get(k), sort_keys=True, default=str)[:160]}")
    return out


def prior_smoke_identity(prior_W: str, smoke_id: str, smoke_wave_uuid: str) -> Dict[str, Any]:
    """The PRIOR tree's smoke identity from its full sha alone (its Dicts / results root are spelled from W7 and the smoke wave -- the same rule
    `smoke_identity` applies to this plan; `s4_waiver_authorized_root` spells the prior tree's the same way), so a record can be re-validated
    without the prior plan at hand."""
    if not isinstance(prior_W, str) or len(prior_W) != 40:
        raise ValueError(f"a prior tree is named by its full 40-hex sha, got {prior_W!r}")
    if not SMOKE_ID_RE.match(smoke_id or ""):
        raise ValueError(f"smoke id {smoke_id!r} must match {SMOKE_ID_RE.pattern}")
    check_wave_uuid(smoke_wave_uuid)
    W7 = prior_W[:7]
    return {"smoke_id": smoke_id, "wave_uuid": smoke_wave_uuid, "W": prior_W, "W7": W7, "dicts": {p: dict_name(W7, smoke_wave_uuid, p) for p in DICT_PURPOSES},
            "results_root": f"{LANE}/{W7}/exp/{RESULTS_ROOT_NAME}{SMOKE_RESULTS_SUFFIX}/{smoke_id}", "ladder_root": f"{LANE}/{W7}/{smoke_wave_uuid}/ladder"}


def smoke_pass_reaudit_problems(ra: Any, *, this_W: str, attempt_root: str) -> List[str]:
    """Why ``ra`` is not THIS tree's clean re-audit of the row's historical root (fj7's `reaudit_problems` plus the store binding)."""
    probs = list(_carry_over_module().reaudit_problems(ra, this_sha=this_W))
    if isinstance(ra, Mapping):
        if [k for k in SMOKE_PASS_PRIOR_REAUDIT_KEYS if k not in ra]:
            probs.append(f"reaudit: lacks {[k for k in SMOKE_PASS_PRIOR_REAUDIT_KEYS if k not in ra]}")
        if ra.get("store_root") != attempt_root:
            probs.append(f"reaudit: store_root {ra.get('store_root')!r} is not the row's historical attempt root {attempt_root}")
        sm = ra.get("sampler")
        if not isinstance(sm, Mapping) or sm.get("rederived_by") != this_W:
            probs.append(f"reaudit: the sampler verdict was not re-derived by this tree (sampler.rederived_by {(sm or {}).get('rederived_by') if isinstance(sm, Mapping) else None!r})")
        elif sm.get("rederived_verdict") != "AGREES":
            probs.append(f"reaudit: this tree's recorder re-derived the sampler verdict as {sm.get('rederived_verdict')!r} over the historical engine log, not AGREES "
                         f"(recorder rc {sm.get('recorder_rc')!r}{'; ' + str(sm.get('why')) if sm.get('why') else ''}) -- the stored {sm.get('historical_verdict')!r} does not stand in")
        else:
            # fc11g: the standup's own probe kinds (declared from the root's records) must account for their requests exactly, and nothing may stay unclassified
            pr = sm.get("probe") if isinstance(sm.get("probe"), Mapping) else {}
            if pr.get("n_unclassified") not in (0, None):
                probs.append(f"reaudit: {pr.get('n_unclassified')} request(s) of the historical engine log match no client cap and no declared probe kind")
            by_cap = pr.get("n_probe_by_cap") if isinstance(pr.get("n_probe_by_cap"), Mapping) else {}
            by_kind = pr.get("n_by_kind") if isinstance(pr.get("n_by_kind"), Mapping) else None
            argv_kinds = set(pr.get("declared_argv_kinds") or [])
            for kind, d in (pr.get("declared") or {}).items():
                if not isinstance(d, Mapping) or int(d.get("cap", -1)) in {int(c) for c in (pr.get("client_caps") or [])}:
                    continue
                if d.get("count_rule") == "max":
                    # fg1 / F-AV: a MAXIMUM declared to the recorder's KINDS mode (the standup's own untagged controls at its probe cap, a cap it shares with
                    # the drivers' tagged bites) is held per KIND off the fresh record; a declaration the argv never carried (a default-mode row: the greedy
                    # probe class holds the cap whole and AGREES or refuses as a class) is no claim on the log
                    if kind not in argv_kinds:
                        continue
                    got_k = (by_kind or {}).get(kind)
                    if not isinstance(got_k, int) or not isinstance(d.get("expected_count"), int) or got_k > int(d["expected_count"]):
                        probs.append(f"reaudit: probe kind {kind!r} (cap {d.get('cap')}, {d.get('source')}) allows at most {d.get('expected_count')} request(s); "
                                     f"the recorder classified {got_k!r} of that kind")
                    continue
                got = by_cap.get(str(d.get("cap")))
                if got is None and d.get("conditional"):
                    continue                     # the standup sent this kind at the probe set's own cap (inside the probe class, greedy-verified there)
                if got != d.get("expected_count"):
                    probs.append(f"reaudit: probe kind {kind!r} (cap {d.get('cap')}, {d.get('source')}) expects {d.get('expected_count')} request(s); the engine log holds {got!r} at that cap")
        pe = ra.get("producer_evidence") if isinstance(ra.get("producer_evidence"), Mapping) else {}
        b = pe.get(RESOLVED_SAMPLER_REL)
        if not isinstance(b, Mapping) or b.get("absent") or b.get("writer") != PRODUCER_WRITERS["resolved_sampler"] or b.get("tree_sha") != this_W:
            probs.append(f"reaudit: producer_evidence binds no {RESOLVED_SAMPLER_REL} written by this tree's recorder ({b!r})")
    return probs


AUDIT_COMMON_REL = "scripts/sdfbase/seedrep/audit_common.py"


def load_audit_common(tree: Path):
    """fc11m: THE loader of the audit's store reader for a given tree -- `<tree>/scripts/sdfbase/seedrep/audit_common.py` by EXPLICIT file
    path, never an ambient `import audit_common` (a foreign copy first on `sys.path` would win and run UNCHECKED; the lane's `src`/`scripts` on PYTHONPATH
    does not protect the nested reader dir). A module already imported under that name is reused ONLY when its resolved `__file__` is this tree's file;
    otherwise refused by name, the foreign path recorded -- the re-enumeration must run the certifying tree's own code. The tree's reader dir is put
    first on `sys.path` for the reader's own sibling lookups."""
    import importlib.util  # noqa: PLC0415
    want = (Path(tree) / AUDIT_COMMON_REL).resolve()
    if not want.is_file():
        raise ImportError(f"{want} is not on this tree -- the audit's store reader cannot be loaded")
    cached = sys.modules.get("audit_common")
    if cached is not None:
        got = getattr(cached, "__file__", None)
        got_p = Path(got).resolve() if isinstance(got, str) and got else None
        if got_p != want:
            raise ImportError(f"an audit_common module is already imported from {got!r}, not this tree's {want} -- refused: the re-enumeration must run the "
                              "certifying tree's own reader, never a foreign copy")
        return cached
    d = str(want.parent)
    if d not in sys.path:
        sys.path.insert(0, d)
    spec = importlib.util.spec_from_file_location("audit_common", want)
    if spec is None or spec.loader is None:
        raise ImportError(f"{want}: no import spec")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["audit_common"] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop("audit_common", None)
        raise
    return mod


def _audit_common_module():
    """`scripts/sdfbase/seedrep/audit_common` (the audit's store reader) of THIS module's own tree -- the validator re-enumerates a pulled root with it.
    The repo is found by the same defensive detection `registry_plans_module` / `_ws0_bank_view_stamp` apply -- a <repo>/scripts/modal/<this file>
    layout names the tree, anything else is a container (the images install this module FLAT at /root/seedrep_launch_plan.py) -- spelled with
    `.parent` so it adds no third `parents[...]` walk (fc11l: `test_the_image_installed_module_executes_flat_and_the_sibling_apps_guard_their_repo_walks`
    pins exactly the two); the flat layout refuses by name. The reader itself is loaded by `load_audit_common` (fc11m): THIS tree's file by explicit
    path, an ambient / cached foreign copy refused by name -- never an `import audit_common` that whichever copy is first on sys.path would answer."""
    here = Path(__file__).resolve()
    repo = here.parent.parent.parent if here.parent.name == "modal" and here.parent.parent.name == "scripts" else None
    if repo is None:
        raise ImportError(f"{here} is not at <repo>/scripts/modal (the image's flat install) -- a pulled root cannot be re-enumerated here")
    return load_audit_common(repo)


def prior_smoke_audit_scope(row: Mapping[str, Any], ra: Mapping[str, Any], *, reenumerate: bool = False) -> Dict[str, Any]:
    """The per-store audit set a PRIOR SMOKE row's PASS is held to -- EVERY store the pulled prior root HOLDS, by IDENTITY (table leg, seed,
    arm, store path: `stores_present`, the re-audit's `audit_common.inventory_stores` over the pulled root), never the plan row's full arm list and
    never an arm-name set (the controls / Arditi root: nine stores, three legs x seeds 1-3, one arm -- an arm set per leg collapsed them to three
    and let six unbound seed-2/3 stores pass). Rules, each refused by name: the re-audit must have inventoried the root (a leg whose stores refused to
    read, or a candidate store file NO leg could classify -- an unknown layout, a cell without metadata -- is a named problem, never a skip); the set
    must not be empty; EVERY identity must have a BOUND verdict of the plan child with its seed and its store arm (`producer_evidence[<plan rel>]`:
    `store_arm`, the child's `s`, the table leg) and no verdict may bind an identity the root does not hold; two stores sharing (leg, seed, arm) are
    ambiguous. With ``reenumerate`` and the pulled root still on this machine (`ra.pulled_root`), the inventory is taken AGAIN and must equal the
    recorded one. Returns the row narrowed to the audited children (`children`), `arms_audited` (the IDENTITY list), `arms_planned` (the plan
    cells), `smoke_scope` (`full` when every plan child's (cell, seed) is audited) and `problems`."""
    out: Dict[str, Any] = {"children": [], "arms_audited": [], "arms_planned": sorted({str(k.get("cell") or k.get("arm")) for k in (row.get("children") or [])}),
                           "smoke_scope": None, "problems": [], "stores_present": []}
    present = ra.get("stores_present") if isinstance(ra, Mapping) else None
    probs_st = ra.get("stores_problems") if isinstance(ra, Mapping) and isinstance(ra.get("stores_problems"), Mapping) else {}
    if not isinstance(present, list) or any(not isinstance(e, Mapping) or not all(k in e for k in ("leg", "arm", "store_dir", "identity")) for e in present):
        out["problems"].append("the re-audit's stores_present is not an identity inventory of the pulled prior root (leg, seed, arm, store path per store) -- "
                               "the audited store set cannot be scoped")
        return out
    for leg, why in sorted(probs_st.items()):
        if leg == "unclassified":
            out["problems"].append(f"the pulled prior root holds candidate store file(s) no table leg of the row could classify (an unknown layout, a cell without its "
                                   f"metadata): {list(why)[:6]} -- nothing the reader cannot classify is admitted alongside")
        else:
            out["problems"].append(f"the pulled prior root's stores refused to read for leg {leg}: {why}")
    ids = [dict(e) for e in present]
    out["stores_present"] = ids
    if not ids:
        if not probs_st:
            out["problems"].append("the pulled prior root holds NO store of the row's table legs -- nothing was audited; a PASS cannot be derived from it")
        return out
    if reenumerate and isinstance(ra.get("pulled_root"), str) and Path(ra["pulled_root"]).is_dir():
        try:
            ac = _audit_common_module()
            legs = ra.get("table_legs") or sorted({str(e["leg"]) for e in ids})
            inv = ac.inventory_stores(Path(ra["pulled_root"]), [str(x) for x in legs], petri_behaviours=[str(b) for b in (row.get("behaviours") or [])])
            fresh_ids = sorted(e["identity"] for e in inv["stores"])
            if fresh_ids != sorted(e["identity"] for e in ids) or inv["unclassified"] or inv["problems"]:
                out["problems"].append(f"re-enumerating the pulled root {ra['pulled_root']} today gives {len(fresh_ids)} store identit(ies) (unclassified {inv['unclassified'][:4]}, "
                                       f"problems {inv['problems']}), the record {len(ids)} -- the recorded inventory is not what the root holds")
        except Exception as e:  # noqa: BLE001
            out["problems"].append(f"the pulled root {ra['pulled_root']} could not be re-enumerated: {type(e).__name__}: {e}")
    by_key: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for e in ids:
        by_key.setdefault((str(e["leg"]), str(e.get("s")), str(e["arm"])), []).append(e)
    for k_, es in sorted(by_key.items()):
        if len(es) > 1:
            out["problems"].append(f"two stores share identity leg {k_[0]} s{k_[1]} arm {k_[2]}: {[e['store_dir'] for e in es]} -- ambiguous")
    pe = ra.get("producer_evidence") if isinstance(ra.get("producer_evidence"), Mapping) else {}
    bound: Dict[Tuple[str, str, str], str] = {}
    kids = []
    audited_cells = set()
    for k in row.get("children") or []:
        bound_here = False
        for leg, rel in audit_verdict_rels(k["eval_leg"], k["arm"], k["s"], stage=row["stage"], shard=row["shard"]):
            b = pe.get(rel)
            if isinstance(b, Mapping) and b.get("bound_rel") and not b.get("absent"):
                bound[(str(leg), str(k["s"]), str(b.get("store_arm")))] = str(rel)
                bound_here = True
        if bound_here:
            kids.append(k)
            audited_cells.add((str(k.get("cell") or k.get("arm")), str(k["s"])))
    missing = [e["identity"] for e in ids if (str(e["leg"]), str(e.get("s")), str(e["arm"])) not in bound]
    extra = sorted(f"{leg}|s{s_}|{arm} ({rel})" for (leg, s_, arm), rel in bound.items() if (leg, s_, arm) not in by_key)
    if missing:
        out["problems"].append(f"{len(missing)} store(s) of the pulled root have no bound audit verdict -- a store without an audit is refused: {missing}")
    if extra:
        out["problems"].append(f"verdict(s) bound for store identit(ies) the pulled root does not hold: {extra}")
    out["children"] = kids
    out["arms_audited"] = sorted(e["identity"] for e in ids if (str(e["leg"]), str(e.get("s")), str(e["arm"])) in bound)
    planned_cells = {(str(k.get("cell") or k.get("arm")), str(k["s"])) for k in (row.get("children") or [])}
    out["smoke_scope"] = SMOKE_SCOPE_FULL if audited_cells == planned_cells else SMOKE_SCOPE_SUBSET
    if not kids:
        out["problems"].append("no plan child of the row has a bound audit verdict")
    return out


def derive_smoke_pass_from_priors(*, rs: Path, plan: Mapping[str, Any], smoke_id: str, standup_ids: Sequence[str], priors: Sequence[Mapping[str, Any]],
                                  accepted_get_for_dict: Callable[[str], Callable[[str], Any]], read_text: Callable[[str], Optional[str]],
                                  reaudit: Callable[..., Dict[str, Any]], derived_utc: Optional[str] = None) -> Dict[str, Any]:
    """fc11: the S1-S3 PASS record of THIS plan derived from PRIOR trees' accepted smoke attempts. ``priors`` are the carry-over's
    `load_prior_source` records (in order; the first prior whose smoke Dict names an accepted attempt for a row wins it -- a smoke may draw rows from
    several priors). Per row: the prior plan's row must exist and its `smoke_row_identity` equal this plan's row's (refused by field); the accepted
    attempt off the PRIOR smoke Dict and its root facts (`_smoke_row_facts`); THIS tree's re-audit over the historical root (``reaudit``, the carry-over's
    signature: `(prev_row, this_row, prior, aid, root, this_plan) -> record`; refused unless `reaudit_problems` is empty and `store_root` is the root);
    every required check evaluated on the producer's file -- the audit's from the re-audit's evidence root held to THIS plan (`store_root` = the
    historical root), every other check at the historical root held to the PRIOR plan. S4 / S5 are never derived here."""
    if smoke_id not in SMOKE_PASS_IDS or smoke_id == S4_WAIVER_SMOKE_ID:
        raise ValueError(f"{smoke_id}: only {[s for s in SMOKE_PASS_IDS if s != S4_WAIVER_SMOKE_ID]} derive from prior trees (S4 is the D133 waiver)")
    wave = smoke_execution_wave(rs, smoke_id)
    if wave is None:
        raise ValueError(f"no {smoke_wave_path(rs, smoke_id)}: smoke {smoke_id} has no minted execution wave in this RS")
    if not priors:
        raise ValueError(f"{smoke_id}: name at least one prior archive (--prior RS/prior/<W7>)")
    for pr in priors:
        if pr.get("W") == plan["W"]:
            raise ValueError(f"{smoke_id}: prior {pr.get('path')} IS this tree ({plan['W'][:12]}) -- a record of this tree derives through `pass --rows` (the live Dict)")
    ids = [str(s) for s in standup_ids]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError(f"{smoke_id}: name the plan rows the smoke ran (non-empty, distinct standup ids)")
    rows: Dict[str, Dict[str, Any]] = {}
    plan_rows: Dict[str, Mapping[str, Any]] = {}
    prior_of: Dict[str, Mapping[str, Any]] = {}
    prior_rows: Dict[str, Mapping[str, Any]] = {}
    scoped_rows: Dict[str, Mapping[str, Any]] = {}          # fc11f: the row narrowed to the arms the pulled root holds stores of (the audit check's row)
    for sid in sorted(ids):
        row = plan_row_by_id(plan, sid)
        if row is None:
            raise ValueError(f"{smoke_id}: {sid} is not a row of this plan")
        this_ident = smoke_row_identity(row)
        tried: List[str] = []
        won = None
        for pr in priors:
            prev_row = plan_row_by_id(pr["plan"], sid)
            if prev_row is None:
                tried.append(f"{pr['W'][:7]}: no row {sid}")
                continue
            prev_wave = (pr.get("smoke_waves") or {}).get(smoke_id)
            if not prev_wave:
                tried.append(f"{pr['W'][:7]}: no minted {smoke_id} wave in the archive")
                continue
            ident = prior_smoke_identity(pr["W"], smoke_id, prev_wave)
            akey = accepted_key(ident["W7"], prev_wave, prev_row["leg"], prev_row["engine"], prev_row["stage"], prev_row["shard"])
            aid = accepted_get_for_dict(ident["dicts"]["accepted"])(akey)
            if not isinstance(aid, str) or not aid:
                tried.append(f"{pr['W'][:7]}: {ident['dicts']['accepted']} names no accepted attempt for {akey}")
                continue
            idp = smoke_row_identity_problems(smoke_row_identity(prev_row), this_ident, who=f"{smoke_id}: row {sid} vs prior {pr['W'][:7]}")
            if idp:
                raise ValueError("; ".join(idp))
            won = (pr, prev_row, ident, aid, akey)
            break
        if won is None:
            raise ValueError(f"{smoke_id}: row {sid}: no prior archive holds an accepted attempt for it ({'; '.join(tried)})")
        pr, prev_row, ident, aid, akey = won
        if str(row.get("leg")) == "petri":
            # fc11g: a Petri smoke root is a PROXY root -- no store, no engine log of its own, no audits/ -- so neither the request audit nor the sampler
            # re-derivation can hold it; its evidence is the Petri sidecar proof the carry-over uses (fj16: the drive's COMPLETE + the audited sidecar bound
            # to the attempt). That route is NOT implemented on this path yet: refused by name rather than through a misleading audit refusal
            raise ValueError(f"{smoke_id}: row {sid}: a Petri row's prior root is a proxy root (no store / engine log / audits); its PASS evidence is the Petri sidecar "
                             "proof (evidence_kind petri_sidecar), which this derivation does not route yet -- derive the smoke from its other rows or add the sidecar route")
        facts = _smoke_row_facts(sid=sid, row=prev_row, identity=ident, accepted_get=lambda _k, _a=aid: _a, read_text=read_text)
        root = facts["attempt_root"]
        try:
            ra = reaudit(prev_row=prev_row, this_row=row, prior=pr, aid=aid, root=root, this_plan=plan)
        except Exception as e:  # noqa: BLE001 -- a re-audit that cannot run refuses the row by name, never falls back to the historical verdicts
            raise ValueError(f"{smoke_id}: row {sid}: re-audit under this tree failed: {type(e).__name__}: {e}")
        rp = smoke_pass_reaudit_problems(ra, this_W=plan["W"], attempt_root=root)
        if rp:
            raise ValueError(f"{smoke_id}: row {sid}: the re-audit record is not this tree's clean audit of {root}: {rp}")
        scope = prior_smoke_audit_scope(row, ra)
        if scope["problems"]:
            raise ValueError(f"{smoke_id}: row {sid}: the audited arm set does not cover the pulled prior root's stores: {'; '.join(scope['problems'])}")
        hist_path = f"{root}/{RESOLVED_SAMPLER_REL}"
        hist_text = read_text(hist_path)
        try:
            hist_verdict = json.loads(hist_text).get("verdict") if hist_text is not None else None
        except (ValueError, AttributeError):
            hist_verdict = "(not JSON)"
        facts.update(measured_under_W=pr["W"], measured_under_wave=ident["wave_uuid"], prior_rs=str(pr["path"]), accepted_dict=ident["dicts"]["accepted"],
                     identity=this_ident, identity_sha256=smoke_row_identity_sha(this_ident),
                     reaudit={k: ra.get(k) for k in SMOKE_PASS_PRIOR_REAUDIT_KEYS},
                     # fc11c: both verdicts on the row -- the historical recorder's (recorded, never gating) and this tree's re-derivation (the one the PASS holds to)
                     sampler={"historical_verdict": hist_verdict, "historical_path": hist_path, "rederived_verdict": ra["sampler"].get("rederived_verdict"),
                              "rederived_by": plan["W"], "rederived_path": f"{ra['evidence_root']}/{RESOLVED_SAMPLER_REL}", "why": SMOKE_PASS_SAMPLER_BASIS,
                              # fc11g: the standup's probe kinds declared from the root's records, the recorder's probe counts by cap, nothing unclassified
                              "probe": dict(ra["sampler"].get("probe") or {})})
        facts["reaudit"]["producer_evidence"] = dict(ra["producer_evidence"])
        facts["reaudit"]["sampler"] = dict(ra["sampler"])
        facts["reaudit"]["stores_present"] = list(scope["stores_present"])
        facts["reaudit"]["stores_problems"] = dict(ra.get("stores_problems") or {})
        for k_ in ("table_legs", "pulled_root"):
            if ra.get(k_) is not None:
                facts["reaudit"][k_] = ra[k_]
        # fc11f: the smoke's ACTUAL scope -- the arms whose stores the pulled root holds and were audited, beside the plan row's
        facts.update(arms_audited=list(scope["arms_audited"]), arms_planned=list(scope["arms_planned"]), smoke_scope=scope["smoke_scope"],
                     # fc11j: the container served the SMOKE row (`smoke_row`'s spelling) -- the served-row identity the bank check is held to, beside the plan's
                     served_standup_id={"plan": sid, "smoke": smoke_standup_id(smoke_id, sid)}, qualifications=[])
        rows[sid], plan_rows[sid], prior_of[sid], prior_rows[sid] = facts, row, pr, prev_row
        scoped_rows[sid] = {**dict(row), "children": list(scope["children"])}
    accepted_attempts = [r["accepted_attempt"] for r in rows.values()]
    checks: Dict[str, Any] = {}
    for name in SMOKE_CHECKS[smoke_id]:
        sources = []
        quals: List[Dict[str, Any]] = []
        not_produced: List[Dict[str, Any]] = []
        for sid, r in rows.items():
            root, prev_row, pr, row = r["attempt_root"], prior_rows[sid], prior_of[sid], plan_rows[sid]
            if name in SMOKE_PASS_REDERIVED_CHECKS:
                # fj7 / fc11c: this tree's producers over the historical root -- read from the re-audit evidence root, held to THIS plan; fc11f: the audit
                # over the row NARROWED to the arms whose stores the pulled root holds (a smoke's actual scope), never the plan row's full arm list
                producer, check_root, check_plan, store_root = r["reaudit"]["producer_evidence"], str(r["reaudit"]["evidence_root"]), plan, root
                check_row = scoped_rows[sid] if name == "request_evidence_audit" else row
            else:
                producer, check_root, check_plan, check_row, store_root = producer_map_for(name, r["_evidence"]), root, pr["plan"], prev_row, None
                if name == "adapter_hashes_banked":
                    check_row = {**dict(prev_row), "standup_id": r["served_standup_id"]["smoke"]}      # fc11j: cells.json names the served SMOKE row
            paths = smoke_evidence_paths(check_row, check_root, producer)[name]
            audit_src = {path_: (k_, leg_) for path_, k_, leg_ in smoke_audit_sources(check_row, check_root, producer)} if name == "request_evidence_audit" else {}
            if name == "determinism":
                audit_src = {f"{root}/{rel}": (None, leg_) for leg_, rel in determinism_rels(check_row)}
            if not paths:
                if name == "pool_endpoint" and not check_row.get("pool"):
                    continue
                raise ValueError(f"{smoke_id}: check {name!r}: row {sid} names no producer file to read (no children)")
            for path in paths:
                if not path.startswith(check_root + "/"):
                    raise ValueError(f"{smoke_id}: {path} is not under {check_root}")
                text = read_text(path)
                if text is None:
                    if name in PRIOR_NOT_PRODUCED_ADMITTED:
                        # the producer never wrote this replay artefact at the prior root (the prior bon smoke: determinism_check.py rc 2) and this tree
                        # cannot re-derive it offline -- a named qualification on the row, never a source; an artefact that EXISTS is checked (and refused) as usual
                        q = {"check": name, "artefact": path, "status": PRIOR_NOT_PRODUCED_STATUS, "ruling": "D178", "measured_under_W": r["measured_under_W"],
                             "producer_binding": (producer.get(path[len(check_root) + 1:]) if isinstance(producer, Mapping) else None),
                             "why": PRIOR_NOT_PRODUCED_ADMITTED[name]}
                        r["qualifications"].append(q)
                        not_produced.append({"standup_id": sid, **q})
                        continue
                    raise ValueError(f"{smoke_id}: check {name!r}: the producer's {path} is absent on the volume")
                child, table_leg = audit_src.get(path, (None, None))
                if name == "determinism":
                    # the sampled-decode qualification -- the decode from the RE-DERIVED resolved-sampler record this tree verified, the
                    # historical one cross-checked -- so a sampled gsm8k row's diverged replay is recorded, not refused
                    facts_d, qual = _determinism_source(text, path=path, plan=check_plan, accepted_attempts=accepted_attempts, row=check_row, aid=r["accepted_attempt"],
                                                        root=check_root, producer=producer, table_leg=table_leg,
                                                        # fg2 / P2-2: THIS row's fresh record (under its re-audit evidence root) is the authority; its historical one is cross-checked
                                                        sampler_check=_row_sampler_check(checks.get("resolved_sampler"), root=str(r["reaudit"]["evidence_root"]), reaudit=True),
                                                        evidence=r["_evidence"], read_text=read_text, prev_row=prev_row, store_root=store_root)
                    sources.append(facts_d)
                    if qual is not None:
                        quals.append(qual)
                    continue
                sources.append(_smoke_check(name, text, path=path, plan=check_plan, accepted_attempts=accepted_attempts, row=check_row, aid=r["accepted_attempt"], root=check_root,
                                            producer=producer, child=child, table_leg=table_leg, store_root=store_root))
        checks[name] = {"ok": True, "sources": sources}
        if name == "determinism" and quals:
            checks[name]["criterion3"] = _criterion3_aggregate(quals)          # fg1 / F-AW
        if not_produced:
            checks[name]["not_produced"] = not_produced          # the rows whose artefact the prior producer never wrote (a qualification, not a pass)
        if name == "request_evidence_audit":
            checks[name]["reconciliation"] = audit_reconciliation_tally(sources)
        if name in SMOKE_PASS_REDERIVED_CHECKS:
            checks[name]["reaudit"] = {"reaudited_by": plan["W"], "held_to_plan": "this", "basis": SMOKE_PASS_PRIOR_BASIS}
    for r in rows.values():
        r.pop("_producer", None)
        r.pop("_evidence", None)
        r.pop("_evidence_text", None)
    seen: Dict[str, Mapping[str, Any]] = {}
    for pr in prior_of.values():
        seen[str(pr["path"])] = {"path": str(pr["path"]), "plan_path": str(pr.get("plan_path")), "W": pr["W"], "W7": pr["W7"], "launch_plan_sha": pr.get("launch_plan_sha"),
                                 "smoke_wave": (pr.get("smoke_waves") or {}).get(smoke_id)}
    return {"schema": SMOKE_PASS_SCHEMA, "smoke_id": smoke_id, "verdict": "PASS", "rc": 0, "W": plan["W"], "execution_wave_uuid": wave,
            "production_wave_uuid": plan["wave_uuid"], "launch_plan_sha": plan.get("launch_plan_sha"), "standup_ids": sorted(rows),
            "accepted": {sid: r["accepted_attempt"] for sid, r in rows.items()}, "rows": rows, "checks": checks, "derived_utc": derived_utc or _utc(),
            "measured_under_W": {sid: r["measured_under_W"] for sid, r in rows.items()}, "measured_under_wave": {sid: r["measured_under_wave"] for sid, r in rows.items()},
            "reaudited_by": plan["W"], "priors": [seen[k] for k in sorted(seen)]}


def prior_smoke_pass_dict_name(rec: Mapping[str, Any]) -> str:
    """The `accepted_dicts[<sid>].dict` spelling of a prior-tree PASS record -- the distinct prior smoke Dicts its rows resolved through, sorted
    and `+`-joined (S2 draws from two prior trees' Dicts, S3 from three)."""
    return "+".join(sorted({str(r.get("accepted_dict")) for r in (rec.get("rows") or {}).values() if isinstance(r, Mapping) and r.get("accepted_dict")}))


def is_prior_smoke_pass(rec: Any) -> bool:
    """A PASS record that draws its rows from PRIOR trees (fc11): it carries `measured_under_W`. A record without it is this tree's own derivation."""
    return isinstance(rec, Mapping) and "measured_under_W" in rec


def smoke_pass_prior_problems(rec: Mapping[str, Any], *, smoke_id: str, plan: Mapping[str, Any], execution_wave_uuid: Optional[str],
                              draft_launch_plan_sha: Optional[str] = None, read_text: Optional[Callable[[str], Optional[str]]] = None,
                              accepted_get_for_dict: Optional[Callable[[str], Callable[[str], Any]]] = None) -> List[str]:
    """Why ``rec`` is not smoke ``smoke_id``'s PASS record of THIS plan derived from PRIOR trees (fc11; empty = it is). Beyond the shape
    `smoke_pass_problems` holds every record to: exactly the prior keys; every `measured_under_W` a prior tree's full sha (NEVER this plan's -- such
    a record takes the live path); `reaudited_by` == this plan's W; per row the prior stamps, the identity RECORDED == this plan's row's identity
    (by field), the root under the PRIOR smoke identity's results root, a clean re-audit record of this tree over that root, the three root facts
    re-hashed (and re-derived with ``read_text``), the prior Dict naming the attempt NOW with ``accepted_get_for_dict``; every required check
    present and ok, its sources the audit's under the row's re-audit evidence root held to THIS plan and every other check's under the row's
    historical root held to the PRIOR tree, each re-hashed and re-run on the re-read content."""
    out: List[str] = []
    want_keys = SMOKE_PASS_KEYS + SMOKE_PASS_PRIOR_KEYS
    missing = [k for k in want_keys if k not in rec]
    extra = [k for k in rec if k not in want_keys]
    if missing:
        out.append(f"{smoke_id}: prior-tree record lacks {missing}")
    if extra:
        out.append(f"{smoke_id}: prior-tree record carries keys outside the PASS schema {extra}")
    if smoke_id == S4_WAIVER_SMOKE_ID:
        out.append(f"{smoke_id}: S4 is never derived from a prior tree (D133: the documented waiver stands in for it)")
    if rec.get("schema") != SMOKE_PASS_SCHEMA:
        out.append(f"{smoke_id}: schema {rec.get('schema')!r} != {SMOKE_PASS_SCHEMA}")
    if rec.get("smoke_id") != smoke_id:
        out.append(f"{smoke_id}: the record names smoke {rec.get('smoke_id')!r}")
    if rec.get("verdict") != "PASS" or rec.get("rc") != 0:
        out.append(f"{smoke_id}: verdict {rec.get('verdict')!r} rc {rec.get('rc')!r} is not a PASS at rc 0")
    if rec.get("W") != plan["W"]:
        out.append(f"{smoke_id}: W {rec.get('W')!r} is not this plan's {plan['W']} (a prior-tree record is DERIVED by this tree; the prior trees are named under measured_under_W)")
    if rec.get("reaudited_by") != plan["W"]:
        out.append(f"{smoke_id}: reaudited_by {str(rec.get('reaudited_by'))[:12]!r} is not this tree {plan['W'][:12]} -- a prior-tree record without this tree's re-audit is refused")
    if execution_wave_uuid is None:
        out.append(f"{smoke_id}: no minted execution wave ({SMOKES_DIR}/{smoke_id}/{SMOKE_WAVE_FILE}) in this RS")
    elif rec.get("execution_wave_uuid") != execution_wave_uuid:
        out.append(f"{smoke_id}: execution_wave_uuid {rec.get('execution_wave_uuid')!r} is not the smoke's minted wave {execution_wave_uuid}")
    if rec.get("production_wave_uuid") != plan["wave_uuid"]:
        out.append(f"{smoke_id}: production_wave_uuid {rec.get('production_wave_uuid')!r} is not this plan's {plan['wave_uuid']}")
    admissible = {plan.get("launch_plan_sha")} | ({draft_launch_plan_sha} if draft_launch_plan_sha else set())
    if rec.get("launch_plan_sha") not in admissible:
        out.append(f"{smoke_id}: launch_plan_sha {rec.get('launch_plan_sha')!r} is neither this plan's nor its draft's")
    rows = rec.get("rows")
    if not isinstance(rows, Mapping) or not rows:
        out.append(f"{smoke_id}: rows must map every smoke row to its derived facts")
        rows = {}
    muw, muv = rec.get("measured_under_W"), rec.get("measured_under_wave")
    if not isinstance(muw, Mapping) or sorted(muw) != sorted(rows):
        out.append(f"{smoke_id}: measured_under_W must map exactly the rows to their prior tree's sha")
        muw = {}
    if not isinstance(muv, Mapping) or sorted(muv) != sorted(rows):
        out.append(f"{smoke_id}: measured_under_wave must map exactly the rows to their prior smoke wave")
        muv = {}
    priors_rec = rec.get("priors") if isinstance(rec.get("priors"), list) else []
    prior_Ws = {p.get("W") for p in priors_rec if isinstance(p, Mapping)}
    plan_rows = {r["standup_id"]: r for r in _plan_rows_the_app_may_execute(plan)}
    roots: Dict[str, str] = {}
    ev_roots: Dict[str, str] = {}
    idents: Dict[str, Dict[str, Any]] = {}
    prior_plan_of: Dict[str, Any] = {}
    audit_rows: Dict[str, Mapping[str, Any]] = {}            # fc11f: per row, the plan row narrowed to its audited arms (the scope re-derived, never trusted)
    for sid, r in rows.items():
        who = f"{smoke_id}: row {sid}"
        row = plan_rows.get(sid)
        if row is None:
            out.append(f"{who} is not a row of this plan")
        if not isinstance(r, Mapping) or not isinstance(r.get("accepted_attempt"), str) or not r.get("accepted_attempt") or not isinstance(r.get("attempt_root"), str):
            out.append(f"{who} carries no accepted attempt / root")
            continue
        lacking = [k for k in SMOKE_PASS_PRIOR_ROW_KEYS if k not in r]
        if lacking:
            out.append(f"{who} lacks the prior stamps {lacking} -- a prior-tree row without its provenance / re-audit is refused")
            continue
        pW, pwave = r.get("measured_under_W"), r.get("measured_under_wave")
        if not isinstance(pW, str) or len(pW) != 40:
            out.append(f"{who}: measured_under_W {pW!r} is not a full 40-hex tree sha")
            continue
        if pW == plan["W"]:
            out.append(f"{who}: measured_under_W IS this plan's W {plan['W'][:12]} -- a row measured under this tree derives through the live Dict (`pass --rows`), not a prior stamp")
            continue
        if muw.get(sid) != pW or muv.get(sid) != pwave:
            out.append(f"{who}: the record's measured_under_W / measured_under_wave disagree with the row's stamps")
        if pW not in prior_Ws:
            out.append(f"{who}: measured_under_W {pW[:12]} is not among the record's named priors {sorted(str(w)[:12] for w in prior_Ws)}")
        try:
            ident = prior_smoke_identity(pW, smoke_id, str(pwave))
        except ValueError as e:
            out.append(f"{who}: {e}")
            continue
        idents[sid] = ident
        if r.get("accepted_dict") != ident["dicts"]["accepted"]:
            out.append(f"{who}: accepted_dict {r.get('accepted_dict')!r} is not the prior smoke wave's {ident['dicts']['accepted']}")
        aid = r["accepted_attempt"]
        if row is not None:
            this_ident = smoke_row_identity(row)
            out += smoke_row_identity_problems(r.get("identity"), this_ident, who=who)
            if r.get("identity_sha256") != smoke_row_identity_sha(r.get("identity") or {}):
                out.append(f"{who}: identity_sha256 is not the digest of the recorded identity")
            # with the prior archive at hand, the recorded identity must be what the PRIOR row states today (an edited identity is not a binding)
            prior_rs = Path(str(r.get("prior_rs")))
            if prior_rs.exists():
                try:
                    pr = _carry_over_module().load_prior_source(prior_rs)
                    prior_plan_of[sid] = pr
                    prev_row = plan_row_by_id(pr["plan"], sid)
                    if pr["W"] != pW:
                        out.append(f"{who}: the archive {prior_rs} is tree {pr['W'][:12]}, not the row's measured_under_W {pW[:12]}")
                    elif prev_row is None:
                        out.append(f"{who}: the prior plan at {prior_rs} has no row {sid}")
                    else:
                        out += [f"(prior archive) {q}" for q in smoke_row_identity_problems(smoke_row_identity(prev_row), this_ident, who=who)]
                except ValueError as e:
                    out.append(f"{who}: the prior archive {prior_rs} does not load: {e}")
            # the row used for re-deriving the historical facts / checks is the PRIOR row when the archive is at hand, else this plan's (identity-equal)
            want_root = attempt_root_for(ident["results_root"], leg=row["leg"], view=row["engine"], stage=row["stage"], shard=row["shard"], attempt_id=aid, kind=row.get("kind"))
            if r["attempt_root"] != want_root:
                out.append(f"{who} attempt_root {r['attempt_root']} is not the root of accepted attempt {aid!r} under the prior smoke wave ({want_root})")
            if accepted_get_for_dict is not None:
                akey = accepted_key(ident["W7"], ident["wave_uuid"], row["leg"], row["engine"], row["stage"], row["shard"])
                live = accepted_get_for_dict(ident["dicts"]["accepted"])(akey)
                if live != aid:
                    out.append(f"{who}: the prior smoke Dict {ident['dicts']['accepted']} names attempt {live!r} NOW, the record {aid!r} (accepted key {akey})")
                if r.get("accepted_key") != akey:
                    out.append(f"{who}: accepted_key {r.get('accepted_key')!r} is not {akey}")
        srcs = r.get("sources") if isinstance(r.get("sources"), Mapping) else {}
        if set(srcs) != set(SMOKE_ROW_FACT_FILES):
            out.append(f"{who} lacks the three root facts {list(SMOKE_ROW_FACT_FILES)}")
        bad = [n for n, s in srcs.items() if not isinstance(s, Mapping) or s.get("path") != f"{r['attempt_root']}/{n}"]
        if bad:
            out.append(f"{who} sources {bad} are not the accepted attempt {aid!r}'s root facts")
        out += _sources_problems(f"{smoke_id} row {sid}", srcs.values(), read_text)
        if read_text is not None and row is not None:
            try:
                fresh = _smoke_row_facts(sid=sid, row=row, identity=ident, accepted_get=lambda _k, _a=aid: _a, read_text=read_text)
            except ValueError as e:
                out.append(f"{who} no longer derives from the volume: {e}")
            else:
                if {n: s.get("sha256") for n, s in srcs.items() if isinstance(s, Mapping)} != {n: s["sha256"] for n, s in fresh["sources"].items()}:
                    out.append(f"{who}: the accepted attempt's root facts changed since the record was derived")
        # the served-row spelling stamps and the not-produced qualifications
        ss = r.get("served_standup_id")
        if not isinstance(ss, Mapping) or ss.get("plan") != sid or ss.get("smoke") != smoke_standup_id(smoke_id, sid):
            out.append(f"{who}: served_standup_id must record the plan row {sid!r} and the served smoke row {smoke_standup_id(smoke_id, sid)!r}, got {ss!r}")
        quals = r.get("qualifications")
        if not isinstance(quals, list):
            out.append(f"{who}: qualifications must be a list (empty when every producer artefact exists)")
        else:
            for q in quals:
                if not isinstance(q, Mapping) or q.get("check") not in PRIOR_NOT_PRODUCED_ADMITTED or q.get("status") != PRIOR_NOT_PRODUCED_STATUS or q.get("ruling") != "D178":
                    out.append(f"{who}: qualification {q!r} is not a D178 not_produced_at_prior_W of an admitted check {sorted(PRIOR_NOT_PRODUCED_ADMITTED)}")
                    continue
                if row is not None and str(q.get("artefact")) not in smoke_evidence_paths(row, r["attempt_root"])[str(q["check"])]:
                    out.append(f"{who}: qualification names {q.get('artefact')!r}, not a producer artefact of check {q['check']} under the row's root")
                if read_text is not None and read_text(str(q.get("artefact"))) is not None:
                    out.append(f"{who}: the {q['check']} artefact {q.get('artefact')} EXISTS on the volume today -- D178 admits only an artefact the prior producer never "
                               "wrote; an existing one is checked (re-derive the record)")
        rp = smoke_pass_reaudit_problems(r.get("reaudit"), this_W=plan["W"], attempt_root=r["attempt_root"])
        if rp:
            out.append(f"{who}: {'; '.join(rp)}")
        else:
            ev_roots[sid] = str(r["reaudit"]["evidence_root"])
            if row is not None:
                # fc11f: the audited arm set RE-DERIVED from the re-audit's store enumeration + bindings; the stamps must restate it; a store without an
                # audit / an empty set / an unenumerated root refuses by name (the plan row's full arm list is never the expectation for a smoke root)
                scope = prior_smoke_audit_scope(row, r["reaudit"], reenumerate=True)         # fc11i: the pulled root re-inventoried when still on this machine
                out += [f"{who}: {q}" for q in scope["problems"]]
                if not scope["problems"]:
                    audit_rows[sid] = {**dict(row), "children": list(scope["children"])}
                    for k_ in ("arms_audited", "arms_planned", "smoke_scope"):
                        if r.get(k_) != scope[k_]:
                            out.append(f"{who}: {k_} {r.get(k_)!r} is not what the re-audit's stores + bindings derive ({scope[k_]!r})")
            # fc11c: the row's sampler stamps -- re-derived by this tree, AGREES, at the evidence root; the historical verdict what the root still states
            sm = r.get("sampler")
            if not isinstance(sm, Mapping) or [k for k in SMOKE_PASS_SAMPLER_KEYS if k not in sm]:
                out.append(f"{who}: sampler stamps lack {SMOKE_PASS_SAMPLER_KEYS}")
            else:
                if sm.get("rederived_by") != plan["W"] or sm.get("rederived_verdict") != "AGREES":
                    out.append(f"{who}: the sampler verdict the PASS holds to must be THIS tree's re-derivation reading AGREES (recorded rederived_by "
                               f"{str(sm.get('rederived_by'))[:12]!r}, rederived_verdict {sm.get('rederived_verdict')!r}); the stored {sm.get('historical_verdict')!r} never stands in")
                if sm.get("rederived_path") != f"{ev_roots[sid]}/{RESOLVED_SAMPLER_REL}" or sm.get("historical_path") != f"{r['attempt_root']}/{RESOLVED_SAMPLER_REL}":
                    out.append(f"{who}: sampler paths are not the re-audit evidence root's / the historical root's {RESOLVED_SAMPLER_REL}")
                if sm.get("rederived_verdict") != (r["reaudit"].get("sampler") or {}).get("rederived_verdict"):
                    out.append(f"{who}: the row's rederived_verdict disagrees with its re-audit record's")
                if sm.get("why") != SMOKE_PASS_SAMPLER_BASIS:
                    out.append(f"{who}: the sampler stamp does not state its basis (what is hashed: the generated outputs, never the engine log's bytes)")
                if sm.get("probe") != (r["reaudit"].get("sampler") or {}).get("probe"):
                    out.append(f"{who}: the row's probe classification disagrees with its re-audit record's")
                if read_text is not None:
                    ht = read_text(str(sm.get("historical_path")))
                    try:
                        hv = json.loads(ht).get("verdict") if ht is not None else None
                    except (ValueError, AttributeError):
                        hv = "(not JSON)"
                    if hv != sm.get("historical_verdict"):
                        out.append(f"{who}: the historical root's {RESOLVED_SAMPLER_REL} reads verdict {hv!r} today, the record says {sm.get('historical_verdict')!r}")
        roots[sid] = r["attempt_root"]
    if rec.get("standup_ids") != sorted(rows) or rec.get("accepted") != {sid: (r.get("accepted_attempt") if isinstance(r, Mapping) else None) for sid, r in rows.items()}:
        out.append(f"{smoke_id}: standup_ids / accepted disagree with rows")
    checks = rec.get("checks")
    required = SMOKE_CHECKS[smoke_id]
    if not isinstance(checks, Mapping):
        out.append(f"{smoke_id}: checks is not a mapping")
        checks = {}
    accepted_attempts = [r["accepted_attempt"] for r in rows.values() if isinstance(r, Mapping) and isinstance(r.get("accepted_attempt"), str)]

    def _producer_on_volume(name_: str, root_: str) -> Dict[str, Any]:
        if read_text is None:
            return {}
        try:
            ev_ = read_text(f"{root_}/modal_evidence.json")
            return producer_map_for(name_, json.loads(ev_)) if ev_ else {}
        except (ValueError, AttributeError, TypeError):
            return {}
    for name in required:
        c = checks.get(name)
        if not isinstance(c, Mapping) or c.get("ok") is not True:
            out.append(f"{smoke_id}: required check {name!r} is missing or not ok")
            continue
        srcs_c = [s for s in (c.get("sources") or []) if isinstance(s, Mapping)]
        if name == "pool_endpoint":
            pinned = sorted(sid for sid in rows if isinstance(plan_rows.get(sid), Mapping) and plan_rows[sid].get("pool"))
            if not pinned and not srcs_c:
                continue
            if len(srcs_c) < len(pinned):
                out.append(f"{smoke_id}: check pool_endpoint carries {len(srcs_c)} source(s) for {len(pinned)} pool-pinned row(s) {pinned}")
        if name == "request_evidence_audit":
            want_t = audit_reconciliation_tally(srcs_c)
            if c.get("reconciliation") != want_t:
                out.append(f"{smoke_id}: check {name} records reconciliation {c.get('reconciliation')!r}, not the tally of its sources {want_t!r}")
        if name in SMOKE_PASS_REDERIVED_CHECKS:
            ra_c = c.get("reaudit")
            if not isinstance(ra_c, Mapping) or ra_c.get("reaudited_by") != plan["W"] or ra_c.get("held_to_plan") != "this":
                out.append(f"{smoke_id}: check {name} does not state it was re-derived by this tree and held to this plan (a prior tree's {name} read as-is is refused)")
        # the inventory rule: the listed sources are EXACTLY what the derivation enumerates -- the audit's under each row's re-audit evidence root
        # (bound through the re-audit's producer_evidence), every other check's under the row's historical root (the PLAN's names)
        expected_c: collections.Counter = collections.Counter()
        for sid_, r_ in rows.items():
            if not isinstance(r_, Mapping) or plan_rows.get(sid_) is None or sid_ not in roots:
                continue
            if name in SMOKE_PASS_REDERIVED_CHECKS:
                if sid_ in ev_roots and (name != "request_evidence_audit" or sid_ in audit_rows):
                    row_for = audit_rows[sid_] if name == "request_evidence_audit" else plan_rows[sid_]
                    expected_c.update(smoke_evidence_paths(row_for, ev_roots[sid_], (r_.get("reaudit") or {}).get("producer_evidence"))[name])
            else:
                not_prod = {str(q.get("artefact")) for q in (r_.get("qualifications") or []) if isinstance(q, Mapping) and q.get("check") == name}
                expected_c.update(p_ for p_ in smoke_evidence_paths(plan_rows[sid_], roots[sid_])[name] if p_ not in not_prod)   # never expected
        rec_np = c.get("not_produced") if isinstance(c.get("not_produced"), list) else []
        want_np = sorted((sid_, str(q.get("artefact"))) for sid_, r_ in rows.items() if isinstance(r_, Mapping) for q in (r_.get("qualifications") or [])
                         if isinstance(q, Mapping) and q.get("check") == name)
        if sorted((str(x.get("standup_id")), str(x.get("artefact"))) for x in rec_np if isinstance(x, Mapping)) != want_np:
            out.append(f"{smoke_id}: check {name}: not_produced {rec_np!r} does not restate the rows' D178 qualifications {want_np!r}")
        listed_c = collections.Counter((s.get("plan_path") if isinstance(s.get("plan_path"), str) else s.get("path")) for s in srcs_c) if name != "request_evidence_audit" \
            else collections.Counter(s.get("path") for s in srcs_c)
        short = lambda p_: p_.split("/attempts/", 1)[-1] if "/attempts/" in str(p_) else str(p_)[-80:]       # noqa: E731
        missing_paths = sorted(str(x) for x in (expected_c - listed_c).elements())
        foreign = sorted(str(p_) for p_ in (listed_c - expected_c))
        if missing_paths:
            out.append(f"{smoke_id}: check {name} lists {sum(listed_c.values())} source(s) but the rows' producers own {sum(expected_c.values())}; not listed: "
                       f"{[short(p_) for p_ in missing_paths[:4]]}{' ...' if len(missing_paths) > 4 else ''}")
        if foreign:
            out.append(f"{smoke_id}: check {name} lists {len(foreign)} path(s) no row's producer owns: {[short(p_) for p_ in foreign[:3]]}")
        if srcs_c or not want_np:
            out += _sources_problems(f"{smoke_id} check {name}", c.get("sources") or [], read_text)        # A check every row qualifies has no source to hash
        fresh_quals: Optional[List[Dict[str, Any]]] = [] if (name == "determinism" and read_text is not None) else None
        for s in srcs_c:
            path = s.get("path") if isinstance(s.get("path"), str) else ""
            if name in SMOKE_PASS_REDERIVED_CHECKS:
                owner_sid = next((sid_ for sid_, er in ev_roots.items() if path.startswith(er + "/")), None)
            else:
                owner_sid = next((sid_ for sid_, rt in roots.items() if path.startswith(rt + "/")), None)
            if owner_sid is None:
                out.append(f"{smoke_id}: check {name}: {path!r} is not under any row's {'re-audit evidence' if name in SMOKE_PASS_REDERIVED_CHECKS else 'historical attempt'} root")
                continue
            r_o, row_o = rows[owner_sid], plan_rows.get(owner_sid)
            if name in SMOKE_PASS_REDERIVED_CHECKS:
                check_plan: Mapping[str, Any] = plan
                check_root, store_root = ev_roots[owner_sid], (roots[owner_sid] if name == "request_evidence_audit" else None)
                producer_o = dict((r_o.get("reaudit") or {}).get("producer_evidence") or {})
            else:
                check_plan = {**dict(plan), "W": r_o["measured_under_W"]}          # the historical producer's tree (the prior W), as `engine_identity_verdict` does
                check_root, store_root = roots[owner_sid], None
                producer_o = _producer_on_volume(name, check_root)
            out += [f"{smoke_id}: {q}" for q in _smoke_source_facts_problems(name, s, plan=check_plan, accepted_attempts=accepted_attempts)]
            if not isinstance(s.get("producer"), Mapping):
                out.append(f"{smoke_id}: check {name}: {path} records no producer binding")
            if read_text is not None and row_o is not None:
                text = read_text(path)
                if text is None:
                    continue
                pr_ = prior_plan_of.get(owner_sid)
                historical_check = pr_ is not None and name not in SMOKE_PASS_REDERIVED_CHECKS
                check_row = plan_row_by_id(pr_["plan"], owner_sid) if historical_check else row_o
                check_row = check_row if check_row is not None else row_o
                if name == "request_evidence_audit" and owner_sid in audit_rows:
                    check_row = audit_rows[owner_sid]                       # fc11f: the audit's row is the smoke's actual scope
                if name == "adapter_hashes_banked":
                    check_row = {**dict(check_row), "standup_id": smoke_standup_id(smoke_id, owner_sid)}     # fc11j: cells.json names the served SMOKE row
                try:
                    audit_src = {p_: (k_, l_) for p_, k_, l_ in smoke_audit_sources(check_row, check_root, producer_o)} if name == "request_evidence_audit" else {}
                    if name == "determinism":
                        audit_src = {f"{check_root}/{rel}": (None, l_) for l_, rel in determinism_rels(check_row)}
                    child, table_leg = audit_src.get(path, (None, None))
                    if name == "determinism":
                        # re-derived through the same qualification route -- the decode from the RECORDED resolved_sampler check (this tree's re-derivation,
                        # `reaudit` set), cross-checked against the historical record under the PRIOR plan's row
                        fresh, qual = _determinism_source(text, path=path, plan=(pr_["plan"] if historical_check else check_plan), accepted_attempts=accepted_attempts,
                                                          row=check_row, aid=r_o["accepted_attempt"], root=check_root, producer=producer_o, table_leg=table_leg,
                                                          # fg2 / P2-2: the OWNER row's recorded fresh source (its evidence root), never a sibling's
                                                          sampler_check=_row_sampler_check(checks.get("resolved_sampler"), root=ev_roots[owner_sid], reaudit=True) if owner_sid in ev_roots
                                                          else _row_sampler_check(None, root=check_root, reaudit=False),
                                                          evidence=_evidence_on_volume(check_root, read_text), read_text=read_text,
                                                          prev_row=check_row, store_root=store_root)
                        if qual is not None and fresh_quals is not None:
                            fresh_quals.append(qual)
                    else:
                        fresh = _smoke_check(name, text, path=path, plan=(pr_["plan"] if historical_check else check_plan),
                                             accepted_attempts=accepted_attempts, row=check_row, aid=r_o["accepted_attempt"], root=check_root, producer=producer_o,
                                             child=child, table_leg=table_leg, store_root=store_root)
                except (ValueError, AttributeError) as e:
                    out.append(f"{smoke_id}: check {name}: {path} no longer passes on its content under this tree's readers: {e}")
                    continue
                if fresh != dict(s):
                    out.append(f"{smoke_id}: check {name}: {path} recomputed facts differ from the recorded")
        if name == "determinism":
            out += _criterion3_block_problems(f"{smoke_id}: check {name}", c, srcs_c, fresh_quals)          # fg1 / F-AW
    unknown = sorted(set(checks) - set(required))
    if unknown:
        out.append(f"{smoke_id}: checks outside this smoke's set: {unknown}")
    if not isinstance(rec.get("derived_utc"), str) or not rec.get("derived_utc"):
        out.append(f"{smoke_id}: derived_utc missing")
    return out



def _evidence_on_volume(root: str, read_text: Optional[Callable[[str], Optional[str]]]) -> Dict[str, Any]:
    """A row's whole `modal_evidence.json` off the volume (the sampled-decode route reads the resolved-sampler binding out of it for the historical
    cross-check); unreadable evidence is an empty document -- the check then refuses by name, never on invented bindings."""
    if read_text is None:
        return {}
    try:
        ev_ = read_text(f"{root}/modal_evidence.json")
        doc = json.loads(ev_) if ev_ else {}
        return doc if isinstance(doc, Mapping) else {}
    except (ValueError, AttributeError, TypeError):
        return {}


def _sources_problems(who: str, sources: Iterable[Mapping[str, Any]], read_text: Optional[Callable[[str], Optional[str]]]) -> List[str]:
    out: List[str] = []
    n = 0
    for s in sources:
        n += 1
        if not isinstance(s, Mapping) or not isinstance(s.get("path"), str) or not isinstance(s.get("sha256"), str) or len(s["sha256"]) != 64:
            out.append(f"{who}: a source without a path + 64-hex sha256: {s!r}")
            continue
        if read_text is not None:
            text = read_text(s["path"])
            if text is None:
                out.append(f"{who}: {s['path']} is no longer readable (the evidence the record derives from is gone)")
            elif _sha_text(text) != s["sha256"]:
                out.append(f"{who}: {s['path']} changed since the record was derived (sha {_sha_text(text)[:12]} != {s['sha256'][:12]})")
    if n == 0:
        out.append(f"{who}: no sources")
    return out


def _smoke_source_facts_problems(name: str, s: Mapping[str, Any], *, plan: Mapping[str, Any], accepted_attempts: Sequence[str]) -> List[str]:
    """The RECORDED facts of one check source must themselves state the pass (a record whose sources say DISAGREES under
    `ok: true` is not a PASS whatever the flag says): the verdict AGREES, the audit's attempt accepted, the bank covering the plan's adapters,
    the S4 windows positive ints."""
    out: List[str] = []
    who = f"check {name}: {s.get('path')}"
    if name in ("resolved_sampler", "request_evidence_audit") and s.get("verdict") != "AGREES":
        out.append(f"{who} records verdict {s.get('verdict')!r}, not AGREES")
    if name == "resolved_sampler" and s.get("client_task_verdicts") is not None:
        ctv = s.get("client_task_verdicts")
        bad = {t: v for t, v in ctv.items() if v != "AGREES"} if isinstance(ctv, Mapping) and ctv else {"(none)": "no task compared"}
        if bad:
            out.append(f"{who} records per-task verdict(s) {bad}, not AGREES for every task (D133)")
    if name == "request_evidence_audit" and s.get("attempt_id") not in accepted_attempts:
        out.append(f"{who} audits attempt {s.get('attempt_id')!r}, not an accepted attempt of this smoke {list(accepted_attempts)}")
    if name == "request_evidence_audit":
        # the recorded facts must carry a CONSISTENT qualification (the derivation's rule, re-applied without the volume),
        # `independence_lost` must follow the contract, and the rendered line must restate the recorded fields -- a fact edited after derivation
        # (a ledger-less source relabelled ledger-backed, its version deleted, its line reduced to a bare AGREES) fails at certify by name
        qp = audit_qualification_problems(s, who)
        out += qp
        if not qp:
            v = s.get("reconciliation_contract")
            if s.get("independence_lost") is not (v in AUDIT_CONTRACTS_LEDGERLESS):
                out.append(f"{who} records independence_lost {s.get('independence_lost')!r} under contract {v} -- the ledger's independent attempt record "
                           "is lost exactly on a ledger-less row")
            if s.get("qualification") != audit_qualification_line(s):
                out.append(f"{who} renders its qualification as {s.get('qualification')!r}, not {audit_qualification_line(s)!r} -- the rendered line "
                           "must restate the recorded fields")
    if name == "adapter_hashes_banked":
        if not isinstance(s.get("n_banked"), int) or not isinstance(s.get("n_planned"), int) or s["n_banked"] < s["n_planned"] or not isinstance(s.get("cells"), Mapping):
            out.append(f"{who} records {s.get('n_banked')!r} banked / {s.get('n_planned')!r} planned -- not every cell of the row is hashed")
    if name == "s5_window_measurement":
        if s.get("clamped_calls") != 0 or s.get("store_complete") is not True or not isinstance(s.get("context_max"), int) or s.get("context_max", 0) <= 0:
            out.append(f"{who} records clamped_calls {s.get('clamped_calls')!r} / store_complete {s.get('store_complete')!r} / context_max {s.get('context_max')!r} -- not a D50 measurement")
        if s.get("attempt_id") not in accepted_attempts:
            out.append(f"{who} measures attempt {s.get('attempt_id')!r}, not an accepted attempt of this smoke {list(accepted_attempts)}")
    if name == "determinism":
        c3 = s.get("criterion3")
        if c3 is not None:
            # a SAMPLED row's diverged replay is a recorded qualification -- the source states the REAL counts (identical < n_requests,
            # reproduces false), the qualification's verdict and ruling, and the VERIFIED decode it rests on (every temperature > 0, a named
            # source); anything else under the key is not the qualification
            dec = s.get("criterion3_decode") if isinstance(s.get("criterion3_decode"), Mapping) else {}
            temps = dec.get("temperatures")
            n_req, n_id = s.get("n_requests"), s.get("identical")
            if (c3 != DETERMINISM_CRITERION3_NOT_APPLICABLE_SAMPLED or s.get("criterion3_ruling") != DETERMINISM_CRITERION3_RULING or s.get("reproduces") is not False
                    or not isinstance(n_req, int) or isinstance(n_req, bool) or n_req <= 0 or not isinstance(n_id, int) or isinstance(n_id, bool) or not 0 <= n_id < n_req
                    or not isinstance(temps, list) or not temps or not all(isinstance(t, (int, float)) and not isinstance(t, bool) and t > 0 for t in temps)
                    or not isinstance(dec.get("source"), list) or not dec.get("source")):
                out.append(f"{who} records criterion3 {c3!r} (ruling {s.get('criterion3_ruling')!r}, reproduces {s.get('reproduces')!r}, {n_id!r} of {n_req!r} re-sent, "
                           f"decode temperatures {temps!r}) -- not D180's qualification of a diverged replay under a verified SAMPLED decode")
        elif s.get("reproduces") is not True or not isinstance(s.get("n_requests"), int) or s.get("n_requests", 0) <= 0 or s.get("identical") != s.get("n_requests"):
            out.append(f"{who} records reproduces {s.get('reproduces')!r} / {s.get('identical')!r} of {s.get('n_requests')!r} re-sent -- criterion 3 not stated")
        # a budget-limited source is a SMOKE row's, its selection min(planned, budget) and its line restating the fields; tolerated absent
        if s.get("budget_limited"):
            n_p, n_s, ib = s.get("n_planned"), s.get("n_selected"), s.get("item_budget")
            if s.get("smoke_id") is None:
                out.append(f"{who} is budget_limited without a smoke_id -- a production row's spot-check is never budget-limited (D122)")
            if not isinstance(n_p, int) or not isinstance(n_s, int) or not isinstance(ib, int) or n_s != min(n_p, ib) or s.get("n_items") != n_s:
                out.append(f"{who} records n_selected {n_s!r} of n_planned {n_p!r} under item_budget {ib!r} (n_items {s.get('n_items')!r}) -- "
                           "a budget-limited row selects min(planned, budget)")
            elif s.get("qualification") != f"budget_limited (D122): {n_s} of {n_p} planned items, the row's smoke item budget {ib} (D107)":
                out.append(f"{who} renders its qualification as {s.get('qualification')!r} -- the rendered line must restate the recorded fields")
    if not isinstance(s.get("producer"), Mapping) or not s["producer"].get("writer"):
        out.append(f"{who} records no producer binding (writer / tree sha)")
    else:
        out += [f"{who} {q}" for q in _producer_binding_problems(name, s["producer"], W=plan["W"])]     # delta 4 (2): the exact producer, W, rc 0
    if name == "decisions":
        win = s.get("multiturn_window") if isinstance(s.get("multiturn_window"), Mapping) else {}
        bad = [leg for leg in S5_PROBE_LEGS if not isinstance(win.get(leg), int) or isinstance(win.get(leg), bool) or win.get(leg) <= 0]
        if bad:
            out.append(f"{who} records no window for {bad}")
    return out


def smoke_pass_problems(rec: Any, *, smoke_id: str, plan: Mapping[str, Any], execution_wave_uuid: Optional[str],
                        draft_launch_plan_sha: Optional[str] = None, read_text: Optional[Callable[[str], Optional[str]]] = None,
                        accepted_get: Optional[Callable[[str], Any]] = None,
                        accepted_get_for_dict: Optional[Callable[[str], Callable[[str], Any]]] = None) -> List[str]:
    """Why ``rec`` is not smoke ``smoke_id``'s DERIVED PASS record for THIS plan (empty = it is): exactly :data:`SMOKE_PASS_KEYS`; PASS / rc 0;
    W; the smoke's minted execution wave; the production target (the plan's wave; this plan's or its draft's sha); every row a plan row with
    a non-empty accepted attempt and its three root facts (each a path + sha); `standup_ids` / `accepted` consistent with `rows`; EVERY
    required check present, `ok`, with >= 1 source whose RECORDED facts state the pass; with ``read_text``, every source RE-HASHED and every
    check RE-RUN on the re-read content (`_smoke_check`; the recomputed facts must equal the recorded ones); with
    ``accepted_get`` (the smoke wave's LIVE accepted Dict at --go), every row's accepted attempt must be what the Dict names NOW (a snapshot
    naming a superseded attempt is refused)."""
    if not isinstance(rec, Mapping):
        return [f"{smoke_id}: not an object"]
    if is_prior_smoke_pass(rec):
        # fc11: a record drawing its rows from PRIOR trees validates through its own reader (this tree's re-audit, the prior Dicts, the prior roots)
        return smoke_pass_prior_problems(rec, smoke_id=smoke_id, plan=plan, execution_wave_uuid=execution_wave_uuid, draft_launch_plan_sha=draft_launch_plan_sha,
                                         read_text=read_text, accepted_get_for_dict=accepted_get_for_dict)
    out: List[str] = []
    missing = [k for k in SMOKE_PASS_KEYS if k not in rec]
    extra = [k for k in rec if k not in SMOKE_PASS_KEYS]
    if missing:
        out.append(f"{smoke_id}: lacks {missing}")
    if extra:
        out.append(f"{smoke_id}: carries keys outside the PASS schema {extra}")
    if rec.get("schema") != SMOKE_PASS_SCHEMA:
        out.append(f"{smoke_id}: schema {rec.get('schema')!r} != {SMOKE_PASS_SCHEMA} (a hand-authored / pre-delta record is not a derived PASS)")
    if rec.get("smoke_id") != smoke_id:
        out.append(f"{smoke_id}: the record names smoke {rec.get('smoke_id')!r}")
    if rec.get("verdict") != "PASS" or rec.get("rc") != 0:
        out.append(f"{smoke_id}: verdict {rec.get('verdict')!r} rc {rec.get('rc')!r} is not a PASS at rc 0")
    if rec.get("W") != plan["W"]:
        out.append(f"{smoke_id}: W {rec.get('W')!r} is not this plan's {plan['W']}")
    if execution_wave_uuid is None:
        out.append(f"{smoke_id}: no minted execution wave ({SMOKES_DIR}/{smoke_id}/{SMOKE_WAVE_FILE}) -- the smoke never ran")
    elif rec.get("execution_wave_uuid") != execution_wave_uuid:
        out.append(f"{smoke_id}: execution_wave_uuid {rec.get('execution_wave_uuid')!r} is not the smoke's minted wave {execution_wave_uuid}")
    if rec.get("execution_wave_uuid") == plan["wave_uuid"]:
        out.append(f"{smoke_id}: the execution wave IS the production wave -- no smoke runs there")
    if rec.get("production_wave_uuid") != plan["wave_uuid"]:
        out.append(f"{smoke_id}: production_wave_uuid {rec.get('production_wave_uuid')!r} is not this plan's {plan['wave_uuid']}")
    admissible = {plan.get("launch_plan_sha")} | ({draft_launch_plan_sha} if draft_launch_plan_sha else set())
    if rec.get("launch_plan_sha") not in admissible:
        out.append(f"{smoke_id}: launch_plan_sha {rec.get('launch_plan_sha')!r} is neither this plan's nor its draft's")
    rows = rec.get("rows")
    plan_rows = {r["standup_id"]: r for r in _plan_rows_the_app_may_execute(plan)}
    identity = None
    if isinstance(execution_wave_uuid, str) and execution_wave_uuid != plan["wave_uuid"]:
        try:
            identity = smoke_identity(plan, smoke_id, execution_wave_uuid)
        except ValueError:
            identity = None
    if not isinstance(rows, Mapping) or not rows:
        out.append(f"{smoke_id}: rows must map every smoke row run to its derived facts")
        rows = {}
    for sid, r in rows.items():
        row = plan_rows.get(sid)
        if row is None:
            out.append(f"{smoke_id}: row {sid} is not a row of this plan")
        if not isinstance(r, Mapping) or not isinstance(r.get("accepted_attempt"), str) or not r.get("accepted_attempt") or not isinstance(r.get("attempt_root"), str):
            out.append(f"{smoke_id}: row {sid} carries no accepted attempt / root")
            continue
        srcs = r.get("sources") if isinstance(r.get("sources"), Mapping) else {}
        if set(srcs) != set(SMOKE_ROW_FACT_FILES):
            out.append(f"{smoke_id}: row {sid} lacks the three root facts {list(SMOKE_ROW_FACT_FILES)}")
        out += _sources_problems(f"{smoke_id} row {sid}", srcs.values(), read_text)
        if identity is not None and row is not None:
            # the record's root and sources are THE ACCEPTED ATTEMPT's under the smoke wave (an invented id has no such root); with read_text the
            # row is re-derived whole (COMPLETE.json names the id, RESULT DONE rc 0, the evidence clean) -- never trusted from the record
            aid = r["accepted_attempt"]
            if accepted_get is not None:
                akey = accepted_key(identity["W7"], identity["wave_uuid"], row["leg"], row["engine"], row["stage"], row["shard"])
                live = accepted_get(akey)
                if live != aid:
                    out.append(f"{smoke_id}: row {sid}: the smoke wave's accepted Dict names attempt {live!r} NOW, the record {aid!r} -- a superseded / invented attempt "
                               f"(accepted key {akey})")
            want_root = attempt_root_for(identity["results_root"], leg=row["leg"], view=row["engine"], stage=row["stage"], shard=row["shard"], attempt_id=aid, kind=row.get("kind"))
            if r["attempt_root"] != want_root:
                out.append(f"{smoke_id}: row {sid} attempt_root {r['attempt_root']} is not the root of accepted attempt {aid!r} under the smoke wave ({want_root})")
            bad = [n for n, s in srcs.items() if not isinstance(s, Mapping) or s.get("path") != f"{want_root}/{n}"]
            if bad:
                out.append(f"{smoke_id}: row {sid} sources {bad} are not the accepted attempt {aid!r}'s root facts")
            if read_text is not None:
                try:
                    fresh = _smoke_row_facts(sid=sid, row=row, identity=identity, accepted_get=lambda _k, _a=aid: _a, read_text=read_text)
                except ValueError as e:
                    out.append(f"{smoke_id}: row {sid} no longer derives from the volume: {e}")
                else:
                    if {n: s.get("sha256") for n, s in srcs.items() if isinstance(s, Mapping)} != {n: s["sha256"] for n, s in fresh["sources"].items()}:
                        out.append(f"{smoke_id}: row {sid}: the accepted attempt's root facts changed since the record was derived")
    if rec.get("standup_ids") != sorted(rows) or rec.get("accepted") != {sid: (r.get("accepted_attempt") if isinstance(r, Mapping) else None) for sid, r in rows.items()}:
        out.append(f"{smoke_id}: standup_ids / accepted disagree with rows")
    checks = rec.get("checks")
    required = SMOKE_CHECKS[smoke_id]
    if not isinstance(checks, Mapping):
        out.append(f"{smoke_id}: checks is not a mapping")
        checks = {}
    accepted_attempts = [r["accepted_attempt"] for r in rows.values() if isinstance(r, Mapping) and isinstance(r.get("accepted_attempt"), str)]
    roots = {r["attempt_root"]: sid for sid, r in rows.items() if isinstance(r, Mapping) and isinstance(r.get("attempt_root"), str)}
    for name in required:
        c = checks.get(name)
        if not isinstance(c, Mapping) or c.get("ok") is not True:
            out.append(f"{smoke_id}: required check {name!r} is missing or not ok")
            continue
        srcs_c = [s for s in (c.get("sources") or []) if isinstance(s, Mapping)]
        if name == "pool_endpoint":
            # F-M: one snapshot per row that PINS a pool; a smoke none of whose rows pins one has nothing to bind (S3 run without its
            # pair row) -- and a pool-pinned row without its snapshot is refused by count, never certified on the other rows' evidence
            pinned = sorted(sid for sid in rows if isinstance(plan_rows.get(sid), Mapping) and plan_rows[sid].get("pool"))
            if not pinned and not srcs_c:
                continue
            if len(srcs_c) < len(pinned):
                out.append(f"{smoke_id}: check pool_endpoint carries {len(srcs_c)} source(s) for {len(pinned)} pool-pinned row(s) {pinned}")
        if name == "request_evidence_audit":
            # the check-level reconciliation split is a DERIVED statement -- recomputed from the listed sources, never trusted
            want_t = audit_reconciliation_tally(srcs_c)
            if c.get("reconciliation") != want_t:
                out.append(f"{smoke_id}: check {name} records reconciliation {c.get('reconciliation')!r}, not the tally of its sources {want_t!r} -- "
                           "the D106 qualification is re-derived at certify")
        # the INVENTORY rule -- the record's listed sources must be EXACTLY what the fresh derivation enumerates: the canonical
        # producer paths of every accepted row (`smoke_evidence_paths`, stage-aware: a controls base row's audit / determinism carry three legs, a
        # rung's two; a pool snapshot per pool-pinned row), as a MULTISET -- the derivation lists a path once per (child, table leg) that owns it
        # (a static row's goal-shard children share one audit verdict file), so a path may legitimately recur exactly that many times and no
        # more. A PASS record that LISTS fewer (a leg dropped, a whole row's set dropped) used to re-validate clean: every per-source check
        # below reads only what the record names, so an omitted source was never read at all.
        def _producer_on_volume(root_: str) -> Dict[str, Any]:
            """F-AO: the row's producer bindings off ITS modal_evidence.json on the volume -- the audit's verdicts live at the FILES the audit filed
            (the store's spelling), which the producer's records name (`bound_rel`); unreadable evidence leaves the plan's own names (as before)."""
            if read_text is None:
                return {}
            try:
                ev_ = read_text(f"{root_}/modal_evidence.json")
                return producer_map_for(name, json.loads(ev_)) if ev_ else {}
            except (ValueError, AttributeError, TypeError):
                return {}
        # F-AO: the multiset is the PLAN's names (derivable with no volume, as `go_preconditions` re-validates a certificate); an audit source lists the
        # FILE the audit filed (`path`, the store's spelling) and the plan path it was bound under (`plan_path`) -- coverage counts the latter
        expected_c = collections.Counter(path_ for sid_, r_ in rows.items()
                                         if isinstance(r_, Mapping) and isinstance(r_.get("attempt_root"), str) and plan_rows.get(sid_) is not None
                                         for path_ in smoke_evidence_paths(plan_rows[sid_], r_["attempt_root"])[name])
        listed_c = collections.Counter((s.get("plan_path") if isinstance(s.get("plan_path"), str) else s.get("path")) for s in srcs_c)
        short = lambda p_: p_.split("/attempts/", 1)[-1] if "/attempts/" in str(p_) else p_       # noqa: E731  -- the root-relative tail, for the message
        missing_paths = sorted((expected_c - listed_c).elements())
        over_listed = sorted(p_ for p_ in (listed_c - expected_c) if p_ in expected_c)
        foreign = sorted(p_ for p_ in (listed_c - expected_c) if p_ not in expected_c)
        if missing_paths:
            out.append(f"{smoke_id}: check {name} lists {sum(listed_c.values())} source(s) but the accepted rows' producers own {sum(expected_c.values())}; "
                       f"not listed: {[short(p_) for p_ in missing_paths[:4]]}{' ...' if len(missing_paths) > 4 else ''} -- an omitted source is never "
                       "re-read, so the record is not this smoke's derivation")
        if over_listed:
            out.append(f"{smoke_id}: check {name} lists {len(over_listed)} path(s) more often than the accepted rows' producers own them: "
                       f"{[(short(p_), listed_c[p_], expected_c[p_]) for p_ in over_listed[:3]]} (listed, owned)")
        if foreign:
            out.append(f"{smoke_id}: check {name} lists {len(foreign)} path(s) no accepted row's producer owns: {[short(p_) for p_ in foreign[:3]]}")
        out += _sources_problems(f"{smoke_id} check {name}", c.get("sources") or [], read_text)
        fresh_quals: Optional[List[Dict[str, Any]]] = [] if (name == "determinism" and read_text is not None and identity is not None) else None
        for s in srcs_c:
            out += [f"{smoke_id}: {q}" for q in _smoke_source_facts_problems(name, s, plan=plan, accepted_attempts=accepted_attempts)]
            path = s.get("path") if isinstance(s.get("path"), str) else ""
            owner = next((root for root in roots if path.startswith(root + "/")), None)
            if owner is None:
                out.append(f"{smoke_id}: check {name}: {path!r} is not under any accepted attempt root of this smoke -- not producer evidence")
                continue
            sid_o = roots[owner]
            row_o = plan_rows.get(sid_o)
            # F-AO: the producer's bindings name the audit's FILES (a plan child's verdict under the store's spelling) -- read once, used for the
            # canonical-path check and the re-verification alike; unreadable evidence leaves the plan's own names (as before)
            producer_o: Dict[str, Any] = _producer_on_volume(owner)
            # F-AO: canonical = the PLAN's path for the (child, table leg) the source was bound under (`plan_path`; derivable offline); with the
            # volume readable, the FILE must also be the one the row's producer bound for it (`bound_rel`), never another spelling of the name
            plan_path_s = s.get("plan_path") if isinstance(s.get("plan_path"), str) else path
            if row_o is not None and plan_path_s not in smoke_evidence_paths(row_o, owner)[name]:
                out.append(f"{smoke_id}: check {name}: {path} is not the canonical producer path for row {sid_o}")
            elif row_o is not None and producer_o and path not in smoke_evidence_paths(row_o, owner, producer_o)[name]:
                out.append(f"{smoke_id}: check {name}: {path} is not the file row {sid_o}'s producer bound for {plan_path_s}")
            if not isinstance(s.get("producer"), Mapping):
                out.append(f"{smoke_id}: check {name}: {path} records no producer binding")
            if read_text is not None and row_o is not None and identity is not None:
                text = read_text(path)
                if text is None:
                    continue                                              # already reported by _sources_problems
                try:
                    producer = producer_o
                    audit_src = {path_: (k_, leg_) for path_, k_, leg_ in smoke_audit_sources(row_o, owner, producer)} if name == "request_evidence_audit" else {}
                    if name == "determinism":
                        audit_src = {f"{owner}/{rel}": (None, leg_) for leg_, rel in determinism_rels(row_o)}
                    child, table_leg = audit_src.get(path, (None, None))
                    if name == "determinism":
                        # re-derived through the same qualification route, the decode from the RECORDED resolved_sampler check's verified sources
                        fresh, qual = _determinism_source(text, path=path, plan=plan, accepted_attempts=accepted_attempts, row=row_o, aid=rows[sid_o]["accepted_attempt"],
                                                          root=owner, producer=producer, table_leg=table_leg,
                                                          sampler_check=_row_sampler_check(checks.get("resolved_sampler"), root=owner, reaudit=False),   # fg2 / P2-2: the OWNER row's source
                                                          evidence=_evidence_on_volume(owner, read_text), read_text=read_text)
                        if qual is not None and fresh_quals is not None:
                            fresh_quals.append(qual)
                    else:
                        fresh = _smoke_check(name, text, path=path, plan=plan, accepted_attempts=accepted_attempts, row=row_o, aid=rows[sid_o]["accepted_attempt"], root=owner,
                                             producer=producer, child=child, table_leg=table_leg)
                except (ValueError, AttributeError) as e:
                    out.append(f"{smoke_id}: check {name}: {path} no longer passes on its content: {e}")
                    continue
                if fresh != dict(s):
                    out.append(f"{smoke_id}: check {name}: {path} recomputed facts {fresh!r} differ from the recorded {dict(s)!r}")
        if name == "determinism":
            out += _criterion3_block_problems(f"{smoke_id}: check {name}", c, srcs_c, fresh_quals)
    unknown = sorted(set(checks) - set(required))
    if unknown:
        out.append(f"{smoke_id}: checks outside this smoke's set: {unknown}")
    if not isinstance(rec.get("derived_utc"), str) or not rec.get("derived_utc"):
        out.append(f"{smoke_id}: derived_utc missing")
    return out


CPU_SUITE_APP_PREFIX = "cpu-suite-"                                  # cpu_suite.py's mandatory gate app name `cpu-suite-<sha7>`
CPU_SUITE_RESULTS_ROOT = f"{VOL_PATH}/cpu-suite"                     # cpu_suite.RESULTS_ROOT on the twosi-fits volume: <root>/<sha>/summary.json
CPU_SUITE_SUMMARY_KEYS: Tuple[str, ...] = ("sha", "app", "status", "pytest_rc", "counts", "failures", "shards", "shard_layout", "shard_records")
CPU_SUITE_SHARD_RECORD_KEYS: Tuple[str, ...] = ("shard", "status", "pytest_rc")


def cpu_suite_summary_path(sha: str) -> str:
    """THE canonical summary of one gate run: `/vol/cpu-suite/<full sha>/summary.json`, derived from the sha, never supplied."""
    if not isinstance(sha, str) or len(sha) != 40:
        raise ValueError(f"a gate summary is named by a full 40-hex sha, got {sha!r}")
    return f"{CPU_SUITE_RESULTS_ROOT}/{sha}/summary.json"


def _summary_nodes(doc: Any, who: str, *, sha: Optional[str] = None) -> Tuple[str, int, int, List[str]]:
    """`(status, tests, failed, failing node ids)` off a cpu_suite summary.json; refuses a non-verdict (TIMEOUT / a refused record) and, with
    ``sha``, a summary that is not THAT tree's: cpu_suite's merged records carry `sha` (the gate's full sha) and `app`
    (`cpu-suite-<sha7>`); both must name ``sha`` -- W / baseline_sha are otherwise labels a summary of any other tree would satisfy."""
    if not isinstance(doc, Mapping):
        raise ValueError(f"{who}: not a summary object")
    missing = [k for k in CPU_SUITE_SUMMARY_KEYS if k not in doc]
    if missing:
        raise ValueError(f"{who}: not cpu_suite's merged summary (lacks {missing})")
    if sha is not None:
        if doc.get("sha") != sha:
            raise ValueError(f"{who}: summary sha {doc.get('sha')!r} is not {sha} -- a summary of another tree (cpu_suite merged records carry the gate's sha)")
        if doc.get("app") != f"{CPU_SUITE_APP_PREFIX}{sha[:7]}":
            raise ValueError(f"{who}: summary app {doc.get('app')!r} is not {CPU_SUITE_APP_PREFIX}{sha[:7]}")
    shards, recs = doc.get("shards"), doc.get("shard_records")
    if not isinstance(shards, int) or isinstance(shards, bool) or shards < 1 or not isinstance(recs, list) or len(recs) != shards \
            or any(not isinstance(r, Mapping) or [k for k in CPU_SUITE_SHARD_RECORD_KEYS if k not in r] for r in recs) or not isinstance(doc.get("shard_layout"), Mapping):
        raise ValueError(f"{who}: shard records do not match cpu_suite's schema (shards {shards!r}, {len(recs) if isinstance(recs, list) else recs!r} record(s))")
    status = str(doc.get("status"))
    if status not in W_GATE_ADMISSIBLE_STATUSES:
        raise ValueError(f"{who}: status {status!r} is no verdict (admissible: {W_GATE_ADMISSIBLE_STATUSES})")
    if not isinstance(doc.get("pytest_rc"), int) or isinstance(doc.get("pytest_rc"), bool):
        raise ValueError(f"{who}: pytest_rc {doc.get('pytest_rc')!r} is not an int")
    failures = doc.get("failures")
    if not isinstance(failures, list) or any(not isinstance(f, Mapping) or not isinstance(f.get("node"), str) for f in failures):
        raise ValueError(f"{who}: failures[] is not a list of {{node}} records")
    counts = doc.get("counts") if isinstance(doc.get("counts"), Mapping) else {}
    tests = counts.get("tests")
    if not isinstance(tests, int) or isinstance(tests, bool):
        raise ValueError(f"{who}: counts.tests {tests!r} is not an int")
    return status, int(tests), len(failures), sorted({str(f["node"]) for f in failures})


def derive_w_gate(*, W: str, baseline_sha: str, read_text: Callable[[str], Optional[str]], derived_utc: Optional[str] = None) -> Dict[str, Any]:
    """The W gate DERIVED from the two merged CPU-suite summaries: both read and hashed, `introduced` = the failing node
    ids of W's summary that the baseline's lacks ("introduced == empty by node id"). Raises on an unreadable summary
    or a non-verdict; the verdict itself (introduced empty or not) is recorded, and `w_gate_problems` refuses a non-empty one."""
    if not isinstance(baseline_sha, str) or len(baseline_sha) != 40 or baseline_sha == W:
        raise ValueError(f"baseline_sha {baseline_sha!r} must be a 40-hex sha other than W")
    w_summary_path, baseline_summary_path = cpu_suite_summary_path(W), cpu_suite_summary_path(baseline_sha)     # derived, never supplied (delta 3 (3))
    texts = {"W": read_text(str(w_summary_path)), "baseline": read_text(str(baseline_summary_path))}
    for k, v in texts.items():
        if v is None:
            raise ValueError(f"the {k} summary {w_summary_path if k == 'W' else baseline_summary_path} is unreadable")
    docs = {k: json.loads(v) for k, v in texts.items()}
    st_w, n_w, f_w, nodes_w = _summary_nodes(docs["W"], "W summary", sha=W)
    st_b, n_b, f_b, nodes_b = _summary_nodes(docs["baseline"], "baseline summary", sha=baseline_sha)
    return {"schema": W_GATE_SCHEMA, "W": W, "baseline_sha": baseline_sha, "introduced": sorted(set(nodes_w) - set(nodes_b)),
            "nodes_W": {"total": n_w, "failed": f_w}, "nodes_baseline": {"total": n_b, "failed": f_b}, "statuses": {"W": st_w, "baseline": st_b},
            "summary_paths": {"W": str(w_summary_path), "baseline": str(baseline_summary_path)},
            "summary_sha256s": {"W": _sha_text(texts["W"]), "baseline": _sha_text(texts["baseline"])}, "derived_utc": derived_utc or _utc()}


def w_gate_problems(rec: Any, *, W: str, read_text: Optional[Callable[[str], Optional[str]]] = None) -> List[str]:
    """Why ``rec`` is not the derived W gate verdict (empty = it is): exactly :data:`W_GATE_KEYS`; this W; a 40-hex baseline other than W;
    `introduced` a list that is EMPTY; admissible statuses; the two summaries by path + sha -- and, with ``read_text``, both summaries
    RE-READ: their shas must match and `introduced` recomputed from their content must be empty (a summary that changed after recording refuses)."""
    if not isinstance(rec, Mapping):
        return ["w_gate: not an object"]
    out: List[str] = []
    missing = [k for k in W_GATE_KEYS if k not in rec]
    extra = [k for k in rec if k not in W_GATE_KEYS]
    if missing:
        out.append(f"w_gate: lacks {missing}")
    if extra:
        out.append(f"w_gate: carries keys outside the schema {extra}")
    if rec.get("schema") != W_GATE_SCHEMA:
        out.append(f"w_gate: schema {rec.get('schema')!r} != {W_GATE_SCHEMA} (a hand-authored / pre-delta record is not a derived verdict)")
    if rec.get("W") != W:
        out.append(f"w_gate: gated W {rec.get('W')!r}, not {W}")
    b = rec.get("baseline_sha")
    if not isinstance(b, str) or len(b) != 40 or b == W:
        out.append(f"w_gate: baseline_sha {b!r} is not a 40-hex sha other than W")
    intro = rec.get("introduced")
    if not isinstance(intro, list):
        out.append(f"w_gate: introduced {intro!r} is not the list of node ids W introduces")
    elif intro:
        out.append(f"w_gate: W INTRODUCES {len(intro)} failing node(s) vs the baseline: {intro[:5]} -- the gate is not clean")
    for k in ("nodes_W", "nodes_baseline"):
        blk = rec.get(k)
        if not isinstance(blk, Mapping) or not all(isinstance(blk.get(x), int) and not isinstance(blk.get(x), bool) for x in ("total", "failed")):
            out.append(f"w_gate: {k} must be {{total: int, failed: int}}")
    sts = rec.get("statuses")
    if not isinstance(sts, Mapping) or set(sts) != {"W", "baseline"} or any(v not in W_GATE_ADMISSIBLE_STATUSES for v in sts.values()):
        out.append(f"w_gate: statuses must be the two summaries' statuses, each in {W_GATE_ADMISSIBLE_STATUSES}")
    paths, shas = rec.get("summary_paths"), rec.get("summary_sha256s")
    want_paths = {"W": cpu_suite_summary_path(W)} | ({"baseline": cpu_suite_summary_path(b)} if isinstance(b, str) and len(b) == 40 else {})
    if not isinstance(paths, Mapping) or set(paths) != {"W", "baseline"} or any(paths.get(k) != v for k, v in want_paths.items()):
        out.append(f"w_gate: summary_paths must be the canonical cpu_suite outputs {want_paths} (got {paths!r}) -- no operator path names a summary")
        paths = {}
    if not isinstance(shas, Mapping) or set(shas) != {"W", "baseline"} or not all(isinstance(v, str) and len(v) == 64 for v in shas.values()):
        out.append("w_gate: summary_sha256s must carry the two summaries' 64-hex shas")
        shas = {}
    if read_text is not None and paths and shas:
        nodes: Dict[str, List[str]] = {}
        for k in ("W", "baseline"):
            text = read_text(str(paths[k]))
            if text is None:
                out.append(f"w_gate: the {k} summary {paths[k]} is no longer readable")
                continue
            if _sha_text(text) != shas[k]:
                out.append(f"w_gate: the {k} summary {paths[k]} changed since the verdict was derived (sha {_sha_text(text)[:12]} != {shas[k][:12]})")
            try:
                nodes[k] = _summary_nodes(json.loads(text), f"{k} summary", sha=(W if k == "W" else (b if isinstance(b, str) else None)))[3]
            except ValueError as e:
                out.append(f"w_gate: {e}")
        if set(nodes) == {"W", "baseline"}:
            recomputed = sorted(set(nodes["W"]) - set(nodes["baseline"]))
            if recomputed != (intro if isinstance(intro, list) else None):
                out.append(f"w_gate: `introduced` recomputed from the summaries is {recomputed[:5]}, not the recorded {intro!r}")
    if not isinstance(rec.get("derived_utc"), str) or not rec.get("derived_utc"):
        out.append("w_gate: derived_utc missing")
    return out


def decode_ref_proof(tree: Path, *, registry_path: Optional[Path] = None) -> Dict[str, Any]:
    """Resolve EVERY battery decode row by EXACT ref through the tree's own `PolicyFile.by_ref` and record what resolved: the policy
    id / ref / version, `superseded_by_policy_id` (must be None), default-ness, mode, thinking mode, the parameters SENT -- plus the registry
    file's path and sha256 the resolution was made against. A ref that names no row or a superseded one is recorded as such (the validator
    refuses it); the proof is EVIDENCE, `decode_proof_problems` is the verdict."""
    tree = Path(tree)
    reg = Path(registry_path) if registry_path else tree / DECODE_REGISTRY_REL
    src = str(tree / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from sisteer.registry.decode_policy import NoPolicy, PolicyFile  # type: ignore
    pf = PolicyFile.load(str(reg))
    rows: Dict[str, Dict[str, Any]] = {}
    for leg_family, ref in BATTERY_DECODE_REFS.items():
        try:
            pol = pf.by_ref(ref)
            rows[leg_family] = {"ref": ref, "policy_id": pol.policy_id, "policy_ref": pol.policy_ref, "policy_version": int(pol.policy_version),
                                "superseded_by_policy_id": pol.superseded_by_policy_id, "is_default": int(pol.is_default), "mode": pol.mode,
                                "thinking_mode": pol.thinking_mode, "params_sent": sorted(pol.request_params()), "resolution": "by_ref"}
        except NoPolicy as e:
            rows[leg_family] = {"ref": ref, "policy_id": None, "policy_ref": None, "policy_version": None, "superseded_by_policy_id": None, "is_default": None,
                                "mode": None, "thinking_mode": None, "params_sent": None, "resolution": f"NO ROW: {e}"}
    return {"schema": DECODE_PROOF_SCHEMA, "registry_path": str(reg), "registry_sha256": sha256_file(reg), "rows": rows, "resolved_utc": _utc()}


def decode_proof_problems(proof: Any, *, registry_sha256: Optional[str] = None) -> List[str]:
    """Why ``proof`` does not prove exact resolution of all eight rows (empty = it does): exactly :data:`DECODE_PROOF_KEYS`; every ref of
    :data:`BATTERY_DECODE_REFS` present, resolved `by_ref` to a policy whose id IS the ref and whose `superseded_by_policy_id` is None;
    a 64-hex registry sha equal to the tree's registry file when given (the proof binds to the bytes it resolved against)."""
    if not isinstance(proof, Mapping):
        return ["decode_rows: not an object"]
    out: List[str] = []
    missing = [k for k in DECODE_PROOF_KEYS if k not in proof]
    extra = [k for k in proof if k not in DECODE_PROOF_KEYS]
    if missing:
        out.append(f"decode_rows: lacks {missing}")
    if extra:
        out.append(f"decode_rows: carries keys outside the schema {extra}")
    if proof.get("schema") != DECODE_PROOF_SCHEMA:
        out.append(f"decode_rows: schema {proof.get('schema')!r} != {DECODE_PROOF_SCHEMA}")
    sha = proof.get("registry_sha256")
    if not isinstance(sha, str) or len(sha) != 64:
        out.append(f"decode_rows: registry_sha256 {sha!r} is not 64-hex")
    elif registry_sha256 is not None and sha != registry_sha256:
        out.append(f"decode_rows: the proof resolved against registry {sha[:12]}..., the tree at W carries {registry_sha256[:12]}... -- re-prove on W")
    rows = proof.get("rows")
    if not isinstance(rows, Mapping):
        return out + ["decode_rows: rows is not a mapping"]
    for fam, ref in BATTERY_DECODE_REFS.items():
        r = rows.get(fam)
        if not isinstance(r, Mapping):
            out.append(f"decode_rows: no proof for {fam} ({ref})")
            continue
        if [k for k in DECODE_ROW_PROOF_KEYS if k not in r]:
            out.append(f"decode_rows[{fam}]: lacks {[k for k in DECODE_ROW_PROOF_KEYS if k not in r]}")
            continue
        if r["ref"] != ref:
            out.append(f"decode_rows[{fam}]: proves {r['ref']!r}, not {ref!r}")
        if r["resolution"] != "by_ref" or r["policy_id"] != ref:
            out.append(f"decode_rows[{fam}]: {ref} did not resolve by exact ref to itself ({r['resolution']}; policy_id {r['policy_id']!r})")
        if r["superseded_by_policy_id"] is not None:
            out.append(f"decode_rows[{fam}]: {ref} is superseded by {r['superseded_by_policy_id']}")
    unknown = [k for k in rows if k not in BATTERY_DECODE_REFS]
    if unknown:
        out.append(f"decode_rows: proves rows outside the battery's eight: {unknown}")
    return out


def s0_record_problems(rec: Any, *, plan: Mapping[str, Any]) -> List[str]:
    """Why ``rec`` is not a FRESH S0 PASS for THIS plan (empty = it is): the runner's schema; this W and this `launch_plan_sha` (S0 is
    re-run against the FINAL before fan-out); `ok` true; `failed` empty; EVERY gate in the report `ok` (a PASS is re-derived from the gates,
    never read off the top-level flag alone) and every enumerated gate present."""
    if not isinstance(rec, Mapping):
        return ["no S0 PASS record (RS/s0_gates.json: run scripts/modal/seedrep_s0_gates.py on this FINAL plan first)"]
    out: List[str] = []
    if rec.get("schema") != "seedrep_s0_gates/1":
        out.append(f"s0: schema {rec.get('schema')!r} is not the S0 runner's")
    if rec.get("W") != plan["W"]:
        out.append(f"the S0 record is for W {rec.get('W')!r}, not this plan's {plan['W']}")
    if rec.get("launch_plan_sha") != plan.get("launch_plan_sha"):
        out.append(f"the S0 record was taken against another plan (launch_plan_sha {rec.get('launch_plan_sha')!r} != {plan.get('launch_plan_sha')!r}): re-run S0 on THIS plan")
    if rec.get("ok") is not True:
        out.append(f"the S0 record is not a PASS (failed gates: {rec.get('failed')})")
    if rec.get("failed") not in ([],):
        out.append(f"s0: failed is {rec.get('failed')!r}, not []")
    gates = rec.get("gates")
    if not isinstance(gates, Mapping) or not gates:
        out.append("s0: no gates block -- a PASS with no gates is a claim")
    else:
        not_ok = sorted(g for g, v in gates.items() if not (isinstance(v, Mapping) and v.get("ok") is True))
        if not_ok:
            out.append(f"s0: gates not ok: {not_ok}")
        required = [f"G{i}" for i in range(1, 11)] + ["ARGV", "LEDGER", "PLAN", "IDENTITY"]
        absent = [g for g in required if g not in gates]
        if absent:
            out.append(f"s0: gates absent from the report: {absent} (an --only run is not an S0 verdict)")
    return out


def _canonical_sha256(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def certificate_sha256(cert: Mapping[str, Any]) -> str:
    """The seal: sha256 over the canonical JSON of every key but the seal itself."""
    return _canonical_sha256({k: v for k, v in cert.items() if k != "certificate_sha256"})


def certificate_problems(cert: Any, *, plan: Mapping[str, Any], registry_sha256: Optional[str] = None, plan_path: Optional[Path] = None,
                         read_text: Optional[Callable[[str], Optional[str]]] = None,
                         accepted_get_for: Optional[Callable[[str], Callable[[str], Any]]] = None,
                         accepted_get_for_dict: Optional[Callable[[str], Callable[[str], Any]]] = None) -> List[str]:
    """EVERYTHING `--go` / `--legs` / `--reattempt` check, each absence NAMED, every component validated on its OWN CONTENT (never a flag):
    the seal, the binding (W, production wave, launch_plan_sha, the plan's own file sha when a `plan_sha256` is carried and the plan is on disk),
    S0, S1-S4, the S5 block (wrapper, references covering every planned window class, >= 1 window measurement each complete with
    `clamped_calls` an int == 0 -- MISSING is not zero), the utilisation verdict (bars_met true, complete, == the S5 block's), the W gate
    (introduced == []), the decode proof (all eight rows, exact, unsuperseded, against the tree's registry bytes when given), draft == final."""
    if not isinstance(cert, Mapping):
        return [f"no launch certificate (RS/{CERTIFICATE_FILE}: seedrep_s5_record.py certify writes it from the validated components)"]
    out: List[str] = []
    missing = [k for k in CERTIFICATE_KEYS if k not in cert]
    extra = [k for k in cert if k not in CERTIFICATE_KEYS]
    if missing:
        out.append(f"certificate lacks {missing}")
    if extra:
        out.append(f"certificate carries keys outside the schema {extra}")
    if missing:
        return out
    if cert["schema"] != CERTIFICATE_SCHEMA:
        out.append(f"certificate schema {cert['schema']!r} != {CERTIFICATE_SCHEMA}")
    if cert["certificate_sha256"] != certificate_sha256(cert):
        out.append("the certificate's seal does not match its contents (edited after issue) -- re-issue it from the components")
    if cert["W"] != plan["W"] or cert["W7"] != plan["W7"]:
        out.append(f"the certificate is for W {cert['W']!r}, not this plan's {plan['W']}")
    if cert["production_wave_uuid"] != plan["wave_uuid"]:
        out.append(f"the certificate targets production wave {cert['production_wave_uuid']!r}, not this plan's {plan['wave_uuid']}")
    # the plan is bound by its BYTES -- the internal digest RECOMPUTED over the loaded plan (never the stamp it carries) and,
    # when the plan file is at hand, the file's sha256 -- so an edited child argv under the old stamp / W / wave / standup ids is refused
    digest = plan_sha(plan)
    if plan.get("launch_plan_sha") != digest:
        out.append(f"the plan's launch_plan_sha stamp {plan.get('launch_plan_sha')!r} is not its content's digest {digest!r} -- the plan was edited after generation")
    if cert["launch_plan_sha"] != digest:
        out.append(f"the certificate was issued for launch_plan_sha {cert['launch_plan_sha']!r}, not this plan's content digest {digest!r} -- a regenerated / edited plan needs a new certificate")
    if plan_path is not None:
        p = Path(plan_path)
        if not p.is_file():
            out.append(f"the plan file {p} is not readable, so its bytes cannot be held to the certificate's plan_sha256")
        elif sha256_file(p) != cert["plan_sha256"]:
            out.append(f"the plan FILE's sha256 {sha256_file(p)[:12]} is not the certified {str(cert['plan_sha256'])[:12]} -- the launched bytes are not the certified bytes")
    draft_sha = cert["draft_launch_plan_sha"]
    if not isinstance(draft_sha, str) or len(draft_sha) != 16:
        out.append("the certificate names no draft launch_plan_sha (the smokes ran against the draft; draft == final is part of the gate)")
    if cert["draft_final_problems"] != []:
        out.append(f"draft != final outside the S4 fields: {cert['draft_final_problems']}")
    waves = cert["smoke_waves"] if isinstance(cert["smoke_waves"], Mapping) else {}
    out += s0_record_problems(cert["s0"], plan=plan)
    smokes = cert["smokes"] if isinstance(cert["smokes"], Mapping) else {}
    for sid in SMOKE_PASS_IDS:
        if sid not in smokes:
            out.append(f"no {sid} PASS record in the certificate" + (f" (nor an admitted {S4_WAIVER_FILE} waiver, D133)" if sid == S4_WAIVER_SMOKE_ID else ""))
            continue
        if is_smoke_waiver_entry(smokes[sid]):
            # a documented WAIVER stands in for the S4 PASS record ONLY -- validated on its own content (+ the named roots re-read
            # off the volume); it has no rows, no checks and no accepted-Dict read, so nothing below applies to it
            out += smoke_waiver_entry_problems(smokes[sid], smoke_id=sid, plan=plan, read_text=read_text)
            continue
        prior_rec = is_smoke_waiver_entry(smokes[sid]) is False and is_prior_smoke_pass(smokes[sid])
        out += smoke_pass_problems(smokes[sid], smoke_id=sid, plan=plan, execution_wave_uuid=waves.get(sid), draft_launch_plan_sha=draft_sha, read_text=read_text,
                                   accepted_get=(accepted_get_for(sid) if accepted_get_for is not None and not prior_rec else None),
                                   accepted_get_for_dict=(accepted_get_for_dict if prior_rec else None))
        # delta 3 (2): the issuer RECORDED what it read in the LIVE accepted Dict; the record's accepted attempts must be those reads
        reads = cert.get("accepted_dicts") if isinstance(cert.get("accepted_dicts"), Mapping) else {}
        rd = reads.get(sid)
        if not isinstance(rd, Mapping) or [k for k in ACCEPTED_DICTS_KEYS if k not in rd] or not isinstance(rd.get("entries"), Mapping):
            out.append(f"{sid}: the certificate records no accepted-Dict read for this smoke (accepted_dicts[{sid!r}] lacks {ACCEPTED_DICTS_KEYS})")
        else:
            if rd.get("digest") != accepted_reads_digest(rd["entries"]):
                out.append(f"{sid}: accepted_dicts digest {str(rd.get('digest'))[:12]} is not the digest of its entries")
            if waves.get(sid) and isinstance(smokes.get(sid), Mapping):
                try:
                    want_dict = smoke_identity(plan, sid, waves[sid])["dicts"]["accepted"]
                except ValueError:
                    want_dict = None
                if prior_rec:
                    # fc11: the issuer read the PRIOR trees' smoke Dicts (one per prior wave the rows drew from), recorded under their joined names
                    want_dict = prior_smoke_pass_dict_name(smokes[sid])
                if want_dict and rd.get("dict") != want_dict:
                    out.append(f"{sid}: accepted_dicts names Dict {rd.get('dict')!r}, not the smoke wave's {want_dict}")
                for sid_row, r in (smokes[sid].get("rows") or {}).items():
                    if isinstance(r, Mapping) and rd["entries"].get(r.get("accepted_key")) != r.get("accepted_attempt"):
                        out.append(f"{sid}: row {sid_row}: the issuer read {rd['entries'].get(r.get('accepted_key'))!r} for {r.get('accepted_key')}, the record says {r.get('accepted_attempt')!r}")
    s5 = cert["s5"]
    s5_problems = s5_block_problems(s5, plan=plan, draft_launch_plan_sha=draft_sha)
    out += [f"s5: {p}" for p in s5_problems]
    launch_set = cert.get("launch_set")
    out += launch_set_problems(plan, launch_set)
    launched_rows = launched_plan_rows(plan, launch_set) if not launch_set_problems(plan, launch_set) else []
    if not s5_problems:
        ew = str(s5["execution_wave_uuid"])
        if waves.get("S5") != ew:
            out.append(f"s5: the block's execution wave {ew} is not the S5 smoke's minted wave {waves.get('S5')!r} (a copied block)")
        # Every reference is judged on its own content, its ADMISSION basis recorded -- this plan's `W` + the minted execution wave, or a prior
        # tree / wave (the standup's engine identity + geometry re-read off its attempt root and held to the plan's standup for the class); the
        # coverage of the LAUNCHED rows' window classes (own reference, or dominated by a larger complete class) is DERIVED here and held to the
        # recorded block
        coverage = s5_certificate_coverage(plan, s5, execution_wave=ew, launch_set=launch_set, read_text=read_text)
        for i, adm in coverage["references"].items():
            if adm["problems"]:
                out.append(f"s5: S5.references[{i}] ({adm['class']}) is not an admissible S5 reference: {adm['problems']}")
        gap = [c for c in coverage["classes"] if c.get("gap")]
        if gap:
            out.append("s5: launched window classes without an S5 reference (own, dominated under D152, or rung-self-measured under D154): "
                       f"{[(tuple(g['class']), g.get('rows'), g['why']) for g in gap]} -- exclude the rows that use them by name")
        if not isinstance(cert.get("s5_coverage"), Mapping):
            out.append("s5: the certificate records no s5_coverage block (fc9)")
        elif cert["s5_coverage"] != coverage:
            out.append("s5: the certificate's s5_coverage is not what the launched rows and the references derive today (edited, or the launch set moved)")
        windows = list(s5.get("multiturn") or [])
        launched_mt = sorted({str(r["leg"]) for r in launched_rows if r.get("leg") in MULTITURN_LEGS})
        if not windows and launched_mt:
            out.append(f"s5: no D50 multi-turn window measurements (S5.multiturn[]) while multi-turn row(s) of {launched_mt} are in the launch set")
        for m in windows:
            if not isinstance(m, Mapping):
                out.append(f"s5: window measurement is not an object: {m!r}")
                continue
            who = f"{m.get('leg')}/{m.get('arm')}"
            wp = window_measurement_problems(m)
            out += [f"s5 window {who}: {p}" for p in wp]
            if not wp and m["wave_uuid"] != ew:
                out.append(f"s5 window {who}: measured under wave {m['wave_uuid']}, not the S5 execution wave {ew}")
            cc = m.get("clamped_calls", None)
            if not isinstance(cc, int) or isinstance(cc, bool):
                out.append(f"s5 window {who}: clamped_calls {cc!r} is REQUIRED as an integer (missing is not zero)")
            elif cc != 0:
                out.append(f"s5 window {who}: {cc} clamped call(s) -- the window is too small for this cell")
        for leg in S5_PROBE_LEGS:
            # a probe leg's window is required only when a multi-turn row of THAT leg is launched (none launched = `multiturn: none_launched`)
            if leg in launched_mt and not any(isinstance(m, Mapping) and m.get("leg") == leg for m in windows):
                out.append(f"s5: no window measurement for the S4-decided leg {leg} (a {leg} row is in the launch set)")
        bars = s5.get("bars")
        vp = utilisation_verdict_problems(bars, W=plan["W"], execution_wave_uuid=ew)
        out += [f"s5 bars: {p}" for p in vp]
        waiver = cert.get("utilisation_waiver")
        if not vp and bars.get("bars_met") is not True:
            # an unmet verdict is admitted ONLY through the readings-bearing waiver derived from it -- re-validated here field by field
            if waiver is None:
                out.append("s5 bars: bars_met is not true (plan §8 U: Running, SM duty, aggregate tok/s, projected hours) -- no fan-out "
                           f"(D198: admissible only through RS/{UTILISATION_WAIVER_FILE}, a waiver derived from the recorded verdict; none is present)")
            else:
                out += [f"s5 bars waiver: {p}" for p in utilisation_waiver_problems(waiver, bars=bars, W=plan["W"], execution_wave_uuid=ew)]
        elif waiver is not None:
            out.append("s5 bars waiver: a utilisation waiver is present while the verdict is met or is not a verdict -- nothing to waive; refusing")
        if cert["utilisation"] != bars:
            out.append("the certificate's utilisation verdict is not the S5 block's bars record")
    out += w_gate_problems(cert["w_gate"], W=plan["W"], read_text=read_text)
    out += decode_proof_problems(cert["decode_rows"], registry_sha256=registry_sha256)
    # fj2: every CARRIED attempt re-validated -- provenance + the five clauses recorded ok, the root STILL reading COMPLETE (read_text), THIS
    # wave's LIVE accepted Dict naming the carried attempt (accepted_get_for(PRODUCTION_ACCEPTED_ID)); `None` = nothing carried
    co = _carry_over_module()
    out += co.carry_over_problems(cert["carry_over"], plan=plan, read_text=read_text,
                                  accepted_get=(accepted_get_for(co.PRODUCTION_ACCEPTED_ID) if (accepted_get_for is not None and cert["carry_over"] is not None) else None))
    if cert["identity_rule"] != WAVE_IDENTITY_RULE:
        out.append("the certificate does not state the wave identity rule")
    if not isinstance(cert["issued_utc"], str) or not cert["issued_utc"]:
        out.append("issued_utc missing")
    if not isinstance(cert["plan_sha256"], str) or len(cert["plan_sha256"]) != 64:
        out.append("plan_sha256 (the FINAL plan file's bytes) missing")
    return out


# --------------------------------------------------------------------------- #
# the durable production fan-out INTENT                                                #
# --------------------------------------------------------------------------- #
PRODUCTION_LAUNCH_FILE = "production_launch.json"
PRODUCTION_LAUNCH_LOCK = "production_launch.lock"
PRODUCTION_LAUNCH_SCHEMA = "seedrep_production_launch/1"
PRODUCTION_LAUNCH_KEYS: Tuple[str, ...] = ("schema", "W", "W7", "wave_uuid", "launch_plan_sha", "certificate_sha256", "plan_sha256", "required_keys", "rows",
                                           "selections", "steward", "created_utc", "updated_utc")
INTENT_ROW_KEYS: Tuple[str, ...] = ("leg", "engine", "stage", "shard", "kind", "control_key", "accepted_key", "call_id", "spawned_utc", "spawned_by",
                                    "claim", "claimed_utc", "executing_call_id")
FANOUT_DUPLICATE = "FANOUT_DUPLICATE"                                # a duplicate container of one claimed row exits with this before any standup
FANOUT_CLAIM_PREFIX = "fanout"


def fanout_executing_key(standup_id: str, nonce: str) -> str:
    """The Dict key a container CLAIMS at entry for its row's fan-out claim (`put(skip_if_exists)`): `fanout|<standup>|<nonce>|executing`."""
    return f"{FANOUT_CLAIM_PREFIX}|{standup_id}|{nonce}|executing"


def new_claim_nonce() -> str:
    import secrets
    return secrets.token_hex(8)


def production_intent_claim(intent: Dict[str, Any], standup_id: str, nonce: str, *, utc: Optional[str] = None) -> Dict[str, Any]:
    """The DURABLE per-row execution claim, written BEFORE `.spawn()`: a row is claimed once; a death between the claim and
    the call record leaves `claim` set with no `call_id` -- 'spawn outcome unknown', reconciled before any second spawn."""
    r = intent["rows"][standup_id]
    if r.get("claim") and r["claim"] != nonce:
        raise ValueError(f"intent row {standup_id} is already claimed ({r['claim']}); a second claim {nonce} would be a second execution")
    r.update(claim=str(nonce), claimed_utc=r.get("claimed_utc") or utc or _utc())
    intent["updated_utc"] = utc or _utc()
    return intent


def production_intent_claimed_unspawned(intent: Mapping[str, Any]) -> List[str]:
    """Rows claimed with no call recorded: a pass died between the claim and the record -- the spawn outcome is UNKNOWN."""
    return sorted(sid for sid, r in (intent.get("rows") or {}).items() if isinstance(r, Mapping) and r.get("claim") and not r.get("call_id"))


def production_intent_marks_pending(intent: Mapping[str, Any]) -> List[str]:
    """Claimed rows whose executing container has not yet been observed in the Dict (its `executing_call_id`)."""
    return sorted(sid for sid, r in (intent.get("rows") or {}).items() if isinstance(r, Mapping) and r.get("claim") and r.get("call_id") and not r.get("executing_call_id"))


def plan_required_keys(plan: Mapping[str, Any]) -> Dict[str, str]:
    """EVERY required accepted key of the wave ({key: standup id}; base + rungs, every leg) -- the steward's DONE criterion and the intent's."""
    out: Dict[str, str] = {}
    for leg in STANDUP_LEGS:
        try:
            out.update(required_accepted_keys(plan, leg, "all"))
        except ValueError:
            continue
    return out


def _intent_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {"leg": row["leg"], "engine": row["engine"], "stage": row["stage"], "shard": row["shard"], "kind": row["kind"], "control_key": row["control_key"],
            "accepted_key": row["accepted_key"], "call_id": None, "spawned_utc": None, "spawned_by": None, "claim": None, "claimed_utc": None, "executing_call_id": None}


def production_intent_new(*, plan: Mapping[str, Any], certificate: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], selection: Mapping[str, str],
                          utc: Optional[str] = None) -> Dict[str, Any]:
    """The durable claim "production of THIS certified plan is launched": written BEFORE the steward starts and before any spawn; the steward
    performs every spawn FROM it (idempotently, `call_id` per row); a later `--go` finds it and refuses or resumes it -- never a second wave."""
    now = utc or _utc()
    ids = sorted(r["standup_id"] for r in rows)
    return {"schema": PRODUCTION_LAUNCH_SCHEMA, "W": plan["W"], "W7": plan["W7"], "wave_uuid": plan["wave_uuid"], "launch_plan_sha": plan.get("launch_plan_sha"),
            "certificate_sha256": certificate["certificate_sha256"], "plan_sha256": certificate["plan_sha256"], "required_keys": plan_required_keys(plan),
            "rows": {r["standup_id"]: _intent_row(r) for r in rows},
            "selections": [{"go": str(selection.get("go") or ""), "legs": str(selection.get("legs") or ""), "rows": ids, "utc": now}],
            "steward": None, "created_utc": now, "updated_utc": now}


def production_intent_problems(intent: Any, *, plan: Mapping[str, Any], certificate: Mapping[str, Any]) -> List[str]:
    """Why ``intent`` is not THIS certified plan's fan-out intent (empty = it is): the schema, W / W7 / wave, launch_plan_sha == the plan's content
    digest == the certificate's, certificate_sha256 / plan_sha256 the certificate's, required_keys the plan's, every row a plan row with its keys."""
    if not isinstance(intent, Mapping):
        return ["intent: not an object"]
    out: List[str] = []
    missing = [k for k in PRODUCTION_LAUNCH_KEYS if k not in intent]
    if missing:
        out.append(f"intent: lacks {missing}")
    if intent.get("schema") != PRODUCTION_LAUNCH_SCHEMA:
        out.append(f"intent: schema {intent.get('schema')!r} != {PRODUCTION_LAUNCH_SCHEMA}")
    for k in ("W", "W7", "wave_uuid"):
        if intent.get(k) != plan.get(k):
            out.append(f"intent: {k} {intent.get(k)!r} is not this plan's {plan.get(k)!r}")
    digest = plan_sha(plan)
    if intent.get("launch_plan_sha") != digest or certificate.get("launch_plan_sha") != digest:
        out.append(f"intent: launch_plan_sha {intent.get('launch_plan_sha')!r} is not this plan's content digest {digest!r} / the certificate's {certificate.get('launch_plan_sha')!r}")
    if intent.get("certificate_sha256") != certificate.get("certificate_sha256"):
        out.append(f"intent: written for certificate {str(intent.get('certificate_sha256'))[:12]}, not this certificate {str(certificate.get('certificate_sha256'))[:12]} -- "
                   "production of another certified plan; never a second wave under this one")
    if intent.get("plan_sha256") != certificate.get("plan_sha256"):
        out.append(f"intent: plan_sha256 {str(intent.get('plan_sha256'))[:12]} is not the certified plan file's {str(certificate.get('plan_sha256'))[:12]}")
    if intent.get("required_keys") != plan_required_keys(plan):
        out.append("intent: required_keys is not the plan's full required set")
    rows = intent.get("rows")
    if not isinstance(rows, Mapping) or not rows:
        out.append("intent: rows must map every launched standup id to its spawn state")
        rows = {}
    # fc9b: every intent row must be in the certificate's launch set -- an edited / recovered intent carrying a valid plan row the
    # certificate never covered would otherwise spawn under it
    out += [f"intent: {p}" for p in launched_ids_problems(certificate, list(rows))]
    for sid, r in rows.items():
        row = plan_row_by_id(plan, sid)
        if row is None:
            out.append(f"intent: row {sid} is not a row of this plan")
            continue
        if not isinstance(r, Mapping) or [k for k in INTENT_ROW_KEYS if k not in r]:
            out.append(f"intent: row {sid} lacks {INTENT_ROW_KEYS}")
            continue
        if r.get("control_key") != row["control_key"] or r.get("accepted_key") != row["accepted_key"]:
            out.append(f"intent: row {sid} keys disagree with the plan row")
        if r.get("call_id") is not None and (not isinstance(r.get("call_id"), str) or not r.get("call_id")):
            out.append(f"intent: row {sid} call_id {r.get('call_id')!r} is not a call id")
    return out


def production_intent_unspawned(intent: Mapping[str, Any]) -> List[str]:
    return sorted(sid for sid, r in (intent.get("rows") or {}).items() if isinstance(r, Mapping) and not r.get("call_id"))


def production_intent_mark_spawned(intent: Dict[str, Any], standup_id: str, call_id: str, *, by: str, utc: Optional[str] = None) -> Dict[str, Any]:
    r = intent["rows"][standup_id]
    if r.get("call_id") and r["call_id"] != call_id:
        raise ValueError(f"intent row {standup_id} already spawned as call {r['call_id']}; a second call {call_id} is a duplicate row")
    r.update(call_id=str(call_id), spawned_utc=utc or _utc(), spawned_by=str(by))
    intent["updated_utc"] = utc or _utc()
    return intent


def production_intent_extend(intent: Dict[str, Any], rows: Sequence[Mapping[str, Any]], *, selection: Mapping[str, str], utc: Optional[str] = None) -> List[str]:
    """Add rows the intent does not carry (a later `--legs` of the SAME certified plan); rows already present are untouched. Returns the added ids."""
    now = utc or _utc()
    added = [r["standup_id"] for r in rows if r["standup_id"] not in intent["rows"]]
    for r in rows:
        if r["standup_id"] in added:
            intent["rows"][r["standup_id"]] = _intent_row(r)
    intent.setdefault("selections", []).append({"go": str(selection.get("go") or ""), "legs": str(selection.get("legs") or ""),
                                                "rows": sorted(r["standup_id"] for r in rows), "added": sorted(added), "utc": now})
    intent["updated_utc"] = now
    return sorted(added)


def production_intent_set_steward(intent: Dict[str, Any], *, unit: str, start_token: str, utc: Optional[str] = None) -> Dict[str, Any]:
    """The steward invocation this intent hands the fan-out to: the unit + a per-invocation START TOKEN the steward echoes in STEWARD_STARTED
    (the launcher correlates on it -- a historical STARTED line in the append-only log is never an acknowledgement)."""
    intent["steward"] = {"unit": unit, "start_token": start_token, "requested_utc": utc or _utc()}
    intent["updated_utc"] = utc or _utc()
    return intent


def read_production_intent(path: Path) -> Optional[Dict[str, Any]]:
    p = Path(path)
    return json.loads(p.read_text()) if p.is_file() else None


def write_production_intent(path: Path, intent: Mapping[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(intent, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, p)


def go_preconditions(plan: Mapping[str, Any], *, certificate: Any, registry_sha256: Optional[str] = None, plan_path: Optional[Path] = None,
                     read_text: Optional[Callable[[str], Optional[str]]] = None,
                     accepted_get_for: Optional[Callable[[str], Callable[[str], Any]]] = None,
                     launched_ids: Optional[Sequence[str]] = None,
                     accepted_get_for_dict: Optional[Callable[[str], Callable[[str], Any]]] = None) -> List[str]:
    """`--go` / `--legs` / `--reattempt` consume the LAUNCH CERTIFICATE and nothing else --
    :func:`certificate_problems` over the whole record, every component re-validated on its own content, the plan held by its BYTES
    (``plan_path``) and every derived record's sources re-hashed (``read_text``). With ``launched_ids`` (the rows a `--go` / `--legs` /
    `--reattempt` names), every one must be in the certificate's `launch_set` -- a row outside the certified set is refused by name.
    ``accepted_get_for_dict`` (`name -> get`, the LIVE prior smoke Dicts by name) is what holds a PRIOR-tree PASS record's
    rows to what those Dicts name NOW; at production consumption a prior record with NO such reader is REFUSED by name (fail closed) -- otherwise a
    prior acceptance removed or superseded after certify, with the old attempt's evidence still readable, would still admit the old PASS."""
    out = certificate_problems(certificate, plan=plan, registry_sha256=registry_sha256, plan_path=plan_path, read_text=read_text, accepted_get_for=accepted_get_for,
                               accepted_get_for_dict=accepted_get_for_dict)
    out += prior_smoke_reader_problems(certificate, accepted_get_for_dict=accepted_get_for_dict)
    if launched_ids is not None and isinstance(certificate, Mapping):
        out += launched_ids_problems(certificate, launched_ids)
    return out


def prior_smoke_reader_problems(cert: Any, *, accepted_get_for_dict: Optional[Callable[[str], Callable[[str], Any]]]) -> List[str]:
    """fc11b: every prior-tree PASS record the certificate carries needs the live prior-Dict reader at consumption; without it each is refused by name."""
    if accepted_get_for_dict is not None or not isinstance(cert, Mapping) or not isinstance(cert.get("smokes"), Mapping):
        return []
    return [f"{sid}: a prior-tree PASS record (measured under {sorted({str(w)[:7] for w in (rec.get('measured_under_W') or {}).values()})}) cannot be consumed without the "
            f"live prior-Dict reader -- its rows' accepted attempts must be what the prior smoke Dicts name NOW (a removed / superseded acceptance is refused); fail closed"
            for sid, rec in cert["smokes"].items() if is_prior_smoke_pass(rec)]


def launched_ids_problems(cert: Mapping[str, Any], launched_ids: Sequence[str]) -> List[str]:
    """Why the rows a launch names are not all in the certificate's `launch_set` (empty = they are, or the certificate certifies every row)."""
    ls = cert.get("launch_set")
    if ls is None:
        return []
    if not isinstance(ls, list):
        return ["the certificate's launch_set is not a list of standup ids"]
    outside = sorted({str(x) for x in launched_ids} - {str(x) for x in ls})
    return [f"row {sid} is not in the certified launch set (the certificate covers {len(ls)} row(s); its S5 coverage was derived for those alone -- D146)" for sid in outside]


def build_certificate(*, plan: Mapping[str, Any], plan_path: Path, draft: Mapping[str, Any], s0_record: Any, smokes_doc: Any, w_gate: Any,
                      decode_proof: Any, rs: Path, registry_sha256: Optional[str] = None, issued_utc: Optional[str] = None,
                      read_text: Optional[Callable[[str], Optional[str]]] = None,
                      accepted_get_for: Optional[Callable[[str], Callable[[str], Any]]] = None,
                      accepted_reads: Optional[Mapping[str, Mapping[str, Any]]] = None,
                      launch_set: Optional[Sequence[str]] = None,
                      accepted_get_for_dict: Optional[Callable[[str], Callable[[str], Any]]] = None) -> Dict[str, Any]:
    """Assemble the certificate from the components and VALIDATE it exactly as `--go` will (`certificate_problems`); raises ValueError naming
    every problem. The smoke waves are read from their minted files under ``rs`` (never from the records)."""
    smokes_doc = smokes_doc if isinstance(smokes_doc, Mapping) else {}
    waves = {sid: smoke_execution_wave(rs, sid) for sid in SMOKE_PASS_IDS + ("S5",)}
    cert: Dict[str, Any] = {
        "schema": CERTIFICATE_SCHEMA, "W": plan["W"], "W7": plan["W7"], "production_wave_uuid": plan["wave_uuid"], "launch_plan_sha": plan.get("launch_plan_sha"),
        "draft_launch_plan_sha": draft.get("launch_plan_sha"), "draft_final_problems": compare_draft_final(draft, plan), "plan_sha256": sha256_file(Path(plan_path)),
        "smoke_waves": waves, "s0": s0_record, "smokes": {sid: smokes_doc.get(sid) for sid in SMOKE_PASS_IDS if sid in smokes_doc},
        "accepted_dicts": dict(accepted_reads or {}),
        "s5": smokes_doc.get("S5"), "utilisation": (smokes_doc.get("S5") or {}).get("bars") if isinstance(smokes_doc.get("S5"), Mapping) else None,
        # the readings-bearing waiver under RS that admits an UNMET utilisation verdict (None = none stamped; re-validated at every read)
        "utilisation_waiver": read_utilisation_waiver(Path(rs)),
        "w_gate": w_gate, "decode_rows": decode_proof,
        # fj2: the carry-over record the launcher evaluated (`--carry-over`) and wrote under RS BEFORE certify; None = nothing carried
        "carry_over": _carry_over_module().read_carry_over(Path(rs)),
        # fc9: the rows this certificate certifies for launch (None = every planned row) and the DERIVED S5 coverage of their window classes
        "launch_set": (sorted({str(x) for x in launch_set}) if launch_set is not None else None),
        "s5_coverage": None,
        "issued_utc": issued_utc or _utc(), "identity_rule": WAVE_IDENTITY_RULE,
    }
    s5_doc = smokes_doc.get("S5")
    if isinstance(s5_doc, Mapping) and s5_doc.get("execution_wave_uuid") and not launch_set_problems(plan, cert["launch_set"]):
        cert["s5_coverage"] = s5_certificate_coverage(plan, s5_doc, execution_wave=str(s5_doc["execution_wave_uuid"]), launch_set=cert["launch_set"],
                                                      read_text=read_text or default_read_text(rs))
    cert["certificate_sha256"] = certificate_sha256(cert)
    problems = certificate_problems(cert, plan=plan, registry_sha256=registry_sha256, plan_path=Path(plan_path), read_text=read_text or default_read_text(rs),
                                     accepted_get_for=accepted_get_for, accepted_get_for_dict=accepted_get_for_dict)     # fc11: the prior smoke Dicts
    if problems:
        raise ValueError("the launch certificate cannot be issued: " + "; ".join(problems))
    return cert


def _merge_s4(s4: Mapping[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(S4_DEFAULTS)
    for k, v in s4.items():
        if k not in out:
            raise ValueError(f"decisions.json: unknown S4 field {k!r} (known: {sorted(out)})")
        if isinstance(v, Mapping):
            for kk, vv in v.items():
                if kk not in out[k]:
                    raise ValueError(f"decisions.json: unknown S4 field {k}.{kk}")
                out[k][kk] = vv
        else:
            out[k] = v
    return out


def _leg_public(spec: Mapping[str, Any]) -> Dict[str, Any]:
    d = {k: v for k, v in spec.items() if k not in ("base", "rungs")}
    d["base"] = served_geometry(spec["base"])
    d["rungs"] = {str(c): served_geometry(w) for c, w in spec["rungs"].items()}
    return d


def pool_rows(s4: Mapping[str, Any], W7: str, wave_uuid: str) -> List[Dict[str, Any]]:
    """The attacker pools (deployed servers, §4.5): sized by DRIVERS, one per attack at least; `assignment` names the
    attack each pool serves; a target driver pins its pool at launch and never fails over. `gpu` is the pool tool's
    ``SEEDREP_ATTACKER_GPU`` spec (one ``H200``; ``gpus`` = its card count); the served S is the pool tool's
    derivation (min(KV bound at the attack's p99 sequence, pinned children x headroom)) and reaches the certificate
    through the live endpoint record, never through this row (`s_att` stays the plan's drivers-per-pool lower bound)."""
    p = s4["pools"]
    n = int(p["count"])
    assign = list(p.get("assignment") or [])
    if len(assign) != n:
        raise ValueError(f"pools.assignment ({len(assign)}) must name one attack per pool ({n})")
    for a in MULTITURN_LEGS:
        if a not in assign:
            raise ValueError(f"pools.assignment {assign} serves no {a} pool — never fewer than one per attack")
    gpus = int(str(p["gpu"]).split(":")[1]) if ":" in str(p["gpu"]) else 1
    return [{"name": f"seedrep-attacker-{k}", "app": f"seedrep-attacker-{k}", "attacks": [assign[k]],
             "gpu": p["gpu"], "gpus": gpus, "max_model_len": int(p["max_model_len"]), "chunked_prefill": int(p["chunked_prefill"]),
             "s_att": int(p["s_att"]), "model": ATTACKER_MODEL, "snapshot": ATTACKER_SNAPSHOT,
             "endpoint_record": pool_endpoint_record(f"seedrep-attacker-{k}"),
             "url_placeholder": f"{{pool_url:seedrep-attacker-{k}}}", "min_containers": 1, "max_containers": 1,
             "outlives": "every target container incl. queued time (§4.5)", "W7": W7, "wave_uuid8": wave_uuid[:8]}
            for k in range(n)]


def pools_for_attacks(pools: Sequence[Mapping[str, Any]]) -> Dict[str, List[Mapping[str, Any]]]:
    """``{attack: [pool rows serving it, in pool order]}`` -- the shards of an attack are dealt across ALL of them."""
    out: Dict[str, List[Mapping[str, Any]]] = {}
    for p in pools:
        for a in p["attacks"]:
            out.setdefault(a, []).append(p)
    return out


#: a multi-turn TARGET refuses INSTANTLY when no attacker was ever submitted (the cluster pool announces presence with a pending
#: marker, then its record). On Modal the pool's presence is the deployed pool server: its `endpoint.json` record on the volume and its
#: URL in `RS/pool_urls.json`; "healthy" is `GET <url>/health` answering 200 (or 401 behind the bearer guard). A target child spawned
#: before that burns a container on an instant refusal -- so the plan ENCODES the order.
ATTACKER_PRESENCE_RULE = "main 4acffa1c PR #211 (3353a3a5): a multi-turn target refuses instantly when no attacker was ever submitted"
POOL_HEALTH_RULE = "GET <pool url>/health -> 200 | 401 (bearer-guarded server answers 401 unauthenticated)"


def launch_order(plan_rows: Sequence[Mapping[str, Any]], pools: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """The wave's launch DEPENDENCIES, explicit: (1) the attacker pools (their `endpoint.json` records + the
    orchestrator's `RS/pool_urls.json`); (2) every multi-turn TARGET row, each naming the pool it pins and the two preconditions
    (`pool_urls.json` carries the pool's URL; the pool is healthy) the entrypoint checks before `.spawn`; (3) everything else,
    unordered. A row is never spawned before its stage's preconditions hold."""
    mt = [r for r in plan_rows if r["leg"] in MULTITURN_LEGS]
    return [
        {"stage": "attacker_pools", "owner": "WS2-F", "rows": [p["name"] for p in pools],
         "produces": ["/vol/attacker-pool/seedrep-<k>/endpoint.json (each pool's record)", "RS/pool_urls.json {pool name: url} (the orchestrator, after the records)"],
         "requires": []},
        {"stage": "multiturn_targets", "owner": "WS2-D", "rows": [r["standup_id"] for r in mt],
         "requires": ["RS/pool_urls.json present and naming every pinned pool", POOL_HEALTH_RULE + " for every pinned pool, probed at spawn time"],
         "why": ATTACKER_PRESENCE_RULE, "refusal_at_spawn": "select_launch_rows: a multi-turn row without --pool-urls, with a pool that has no URL, or "
                                                            "with a pool that fails the health probe is REFUSED before .spawn"},
        {"stage": "other_targets_and_rungs", "owner": "WS2-D", "rows": [r["standup_id"] for r in plan_rows if r["leg"] not in MULTITURN_LEGS], "requires": []},
    ]


def pool_is_healthy(url: str, *, timeout: float = 10.0) -> bool:
    """`GET <url>/health` -> 200 (or 401 behind the bearer guard) = UP; a connection error or any other code = DOWN
    (:data:`POOL_HEALTH_RULE`; the ONE probe the entrypoint's spawn gate and the watcher share)."""
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=timeout) as r:
            return r.status in (200, 401)
    except urllib.error.HTTPError as e:
        return e.code in (200, 401)
    except Exception:  # noqa: BLE001
        return False


def mint_wave(path: Path, **extra: Any) -> Dict[str, Any]:
    """RS/wave.json (or a smoke wave's): a DASHED uuid4 minted ONCE and re-read afterwards (never derived, never re-minted)."""
    import uuid
    path = Path(path)
    if path.is_file():
        wave = json.loads(path.read_text())
    else:
        wave = {"wave_uuid": str(uuid.uuid4()), "minted_at": _utc(), **extra}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(wave, indent=1) + "\n")
    check_wave_uuid(wave["wave_uuid"])
    return wave


def pool_coverage_problems(plan: Mapping[str, Any]) -> List[str]:
    """Every deployed pool serving an attack receives at least one of that attack's base shards
    (when the attack has at least as many shards as pools), and every multi-turn row's `--attacker-endpoint`
    placeholder names a pool that serves ITS attack."""
    problems: List[str] = []
    by_attack = pools_for_attacks(plan["pools"])
    used: Dict[str, set] = {}
    for r in plan["standups"]:
        if r["leg"] not in MULTITURN_LEGS:
            continue
        placeholders = {_flag_value(k["argv"], "--attacker-endpoint") for k in r["children"]}
        if None in placeholders or "" in placeholders:
            problems.append(f"{r['standup_id']}: a multi-turn target child carries no --attacker-endpoint -- it would refuse instantly with no attacker "
                            f"({ATTACKER_PRESENCE_RULE}); every multi-turn child pins a deployed pool's URL placeholder")
            continue
        if len(placeholders) != 1:
            problems.append(f"{r['standup_id']}: children name {len(placeholders)} attacker endpoints, not one pool")
            continue
        ph = next(iter(placeholders))
        if not r.get("depends_on") or r["depends_on"].get("pool") != r.get("pool"):
            problems.append(f"{r['standup_id']}: a multi-turn row must declare depends_on.pool == its pinned pool (#211 launch order)")
        serving = [p for p in by_attack.get(r["leg"], []) if p["url_placeholder"] == ph]
        if not serving:
            problems.append(f"{r['standup_id']}: pool placeholder {ph!r} is not a pool serving {r['leg']}")
            continue
        if r.get("pool") != serving[0]["name"]:
            problems.append(f"{r['standup_id']}: row.pool {r.get('pool')!r} != the children's pool {serving[0]['name']!r}")
        used.setdefault(r["leg"], set()).add(serving[0]["name"])
    for attack, pools in by_attack.items():
        n_shards = sum(1 for r in plan["standups"] if r["leg"] == attack)
        if n_shards >= len(pools):
            idle = sorted(p["name"] for p in pools if p["name"] not in used.get(attack, set()))
            if idle:
                problems.append(f"{attack}: pool(s) {idle} receive no shard while {n_shards} shards exist (all traffic on one pool)")
    return problems


def queue_table(legs: Mapping[str, Mapping[str, Any]], standups: Sequence[Mapping[str, Any]],
                rungs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Per ladder queue (leg, view|arditi, cap_next): the DECLARED producer set (§7.2) — the base shards for the first
    rung, the previous rung's container for the next — and the consumer rung row. A queue closes only when EVERY
    declared producer's ACCEPTED attempt has committed."""
    out = []
    by_key = {}
    for r in list(standups) + list(rungs):
        by_key.setdefault((r["leg"], r["engine"], r["stage"]), []).append(r)
    for leg, spec in legs.items():
        ordered = sorted((c for c in spec["rungs"] if isinstance(c, int)))
        if STAGE_PETRI_RERUN in spec["rungs"]:
            ordered = ordered + [STAGE_PETRI_RERUN]
        for engine in ENGINES:
            prev = STAGE_BASE
            for cap in ordered:
                producers = [{"standup_id": r["standup_id"], "shard": r["shard"], "control_key": r["control_key"]}
                             for r in by_key.get((leg, engine, str(prev)), [])]
                consumer = by_key.get((leg, engine, str(cap)), [])
                out.append({"leg": leg, "engine": engine, "cap_next": str(cap), "feeder_stage": str(prev),
                            "producers": producers, "consumer": consumer[0]["standup_id"] if consumer else None,
                            "ceiling": (cap == CAP_CEILING) or cap == STAGE_PETRI_RERUN,
                            "spin_when": f"len(queue) >= S_rung ({consumer[0]['geometry']['S'] if consumer else '?'}) or CLOSED.json"})
                prev = cap
    return out


# --------------------------------------------------------------------------- #
# S5 roster (gate 8), adapter set (gate 10), capacity ledger (§13)               #
# --------------------------------------------------------------------------- #


def s5_roster_rows(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """``sorted({(L, S, thinking)})`` over EVERY planned standup incl. every rung — never maintained by hand."""
    seen = {}
    for r in rows:
        g = r["geometry"]
        seen.setdefault((int(g["L"]), int(g["S"]), bool(g["thinking"])), set()).add(r["leg"] + ":" + str(r["stage"]) + (":dose" if r.get("dose_standup") else ""))
    return [{"L": L, "S": S, "thinking": th, "used_by": sorted(v)} for (L, S, th), v in sorted(seen.items())]


def s5_gap(plan: Mapping[str, Any], references: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Gate 8: ``planned standups - S5 references``; non-empty blocks the affected legs' U. A reference is
    ``{L, S, thinking, ...}`` (RS/smokes.json's S5 block)."""
    have = {(int(r["L"]), int(r["S"]), bool(r.get("thinking", False))) for r in references}
    return [r for r in plan["s5_roster"] if (r["L"], r["S"], r["thinking"]) not in have]


#: THE THROUGHPUT REFERENCE (`utilisation_table.S5_REFERENCE_KEYS`; pinned equal by test, imported by path): one record per planned window
#: class in `RS/smokes.json` `S5.references[]`, the MEASURED aggregate decode rate (the engine-counter method the utilisation table computes;
#: never a percentile or a median) over a COMPLETE store at steady state for >= 20 min of real traffic on THIS tree and wave, with the
#: interval count and the attempt's identity. A bare scalar, a partial store, low coverage or a hand-mirrored value is NO reference and
#: refuses by name -- here at write time (`seedrep_s5_record.py`) and at the gate.
S5_REFERENCE_METHOD = "aggregate_decode_tok_s"
S5_REFERENCE_MIN_DURATION_S = 20 * 60
S5_REFERENCE_MIN_COVERAGE = 0.8
S5_REFERENCE_KEYS: Tuple[str, ...] = ("L", "S", "thinking", "method", "tok_s_aggregate", "duration_s", "store_complete",
                                      "sampling_coverage", "n_eligible_intervals", "W", "wave_uuid", "attempt_id")
#: fc9: the ONE optional key a reference may carry -- the `/vol/...` root of the attempt that measured it (ending in its attempt id), so a
#: reference measured under a PRIOR tree / wave can have its standup's engine identity + geometry re-read and held to this plan's standup
S5_REFERENCE_OPTIONAL_KEYS: Tuple[str, ...] = ("attempt_root",)
S5_REFERENCE_FORBIDDEN_SOURCES: Tuple[str, ...] = ("manual", "mirror", "mirrored", "hand")


def s5_reference_problems(rec: Any, *, W: Optional[str] = None, wave_uuid: Optional[str] = None) -> List[str]:
    """Why ``rec`` is not an S5 reference (empty = it is one) -- the writer's `s5_reference_problems` rule for rule; ``W`` / ``wave_uuid`` = the
    standup's, which the record must name."""
    if not isinstance(rec, Mapping):
        return [f"not an object ({type(rec).__name__}) -- a bare number is not a reference"]
    out: List[str] = []
    missing = [k for k in S5_REFERENCE_KEYS if k not in rec]
    if missing:
        out.append(f"lacks {missing}")
    extra = [k for k in rec if k not in S5_REFERENCE_KEYS and k not in S5_REFERENCE_OPTIONAL_KEYS]
    if extra:
        out.append(f"carries keys outside the S5 reference schema {extra}")
    ar = rec.get("attempt_root")
    if "attempt_root" in rec and (not isinstance(ar, str) or not ar.startswith(VOL_PATH + "/") or not ar.endswith("/" + str(rec.get("attempt_id")))):
        out.append(f"attempt_root {ar!r} is not a `{VOL_PATH}/...` root ending in the record's attempt id")

    def pos_int(k: str) -> bool:
        v = rec.get(k)
        return isinstance(v, int) and not isinstance(v, bool) and v >= 1

    def num(k: str) -> Optional[float]:
        v = rec.get(k)
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None
    for k in ("L", "S", "n_eligible_intervals"):
        if k in rec and not pos_int(k):
            out.append(f"{k} {rec.get(k)!r} is not a positive int")
    if "thinking" in rec and not isinstance(rec.get("thinking"), bool):
        out.append(f"thinking {rec.get('thinking')!r} is not a bool")
    if "method" in rec and rec.get("method") != S5_REFERENCE_METHOD:
        out.append(f"method {rec.get('method')!r} is not {S5_REFERENCE_METHOD!r} -- a percentile, a median or any other statistic is not the aggregate decode rate")
    if "tok_s_aggregate" in rec and not (num("tok_s_aggregate") or 0) > 0:
        out.append(f"tok_s_aggregate {rec.get('tok_s_aggregate')!r} is not a positive number")
    if "duration_s" in rec and (num("duration_s") is None or num("duration_s") < S5_REFERENCE_MIN_DURATION_S):  # type: ignore[operator]
        out.append(f"duration_s {rec.get('duration_s')!r} < {S5_REFERENCE_MIN_DURATION_S} (plan §8 U: 20 min of real traffic)")
    if "store_complete" in rec and rec.get("store_complete") is not True:
        out.append("store_complete is not true -- a partial store is a floor, not a reference")
    if "sampling_coverage" in rec and (num("sampling_coverage") is None or num("sampling_coverage") < S5_REFERENCE_MIN_COVERAGE):  # type: ignore[operator]
        out.append(f"sampling_coverage {rec.get('sampling_coverage')!r} < {S5_REFERENCE_MIN_COVERAGE} -- not steady state")
    w = rec.get("W")
    if "W" in rec and (not isinstance(w, str) or len(w) != 40):
        out.append(f"W {w!r} is not the 40-hex tree sha")
    elif W is not None and w != W:
        out.append(f"W {w} is not the standup's tree {W}")
    wv = rec.get("wave_uuid")
    if "wave_uuid" in rec and (not isinstance(wv, str) or not wv):
        out.append(f"wave_uuid {wv!r} is not a wave")
    elif wave_uuid is not None and wv != wave_uuid:
        out.append(f"wave_uuid {wv} is not the standup's wave {wave_uuid}")
    if "attempt_id" in rec and (not isinstance(rec.get("attempt_id"), str) or not rec.get("attempt_id")):
        out.append(f"attempt_id {rec.get('attempt_id')!r} names no S5 attempt")
    src = rec.get("source", rec.get("origin"))
    if (isinstance(src, str) and src.strip().lower() in S5_REFERENCE_FORBIDDEN_SOURCES) or rec.get("manual") is True:
        out.append(f"source {src!r} -- a manual or mirrored number is not a measurement")
    return out


def s5_reference_record(*, L: int, S: int, thinking: bool, tok_s_aggregate: float, duration_s: float, store_complete: bool, sampling_coverage: float,
                        n_eligible_intervals: int, W: str, wave_uuid: str, attempt_id: str, attempt_root: Optional[str] = None) -> Dict[str, Any]:
    """EXACTLY :data:`S5_REFERENCE_KEYS`, in order, the method fixed to the aggregate engine-counter rate (+ the optional `attempt_root`, fc9);
    refuses what the reference check would refuse."""
    rec: Dict[str, Any] = {"L": int(L), "S": int(S), "thinking": bool(thinking), "method": S5_REFERENCE_METHOD, "tok_s_aggregate": float(tok_s_aggregate),
                           "duration_s": float(duration_s), "store_complete": bool(store_complete), "sampling_coverage": float(sampling_coverage),
                           "n_eligible_intervals": int(n_eligible_intervals), "W": str(W), "wave_uuid": str(wave_uuid), "attempt_id": str(attempt_id)}
    assert tuple(rec) == S5_REFERENCE_KEYS
    if attempt_root is not None:
        rec["attempt_root"] = str(attempt_root)
    problems = s5_reference_problems(rec)
    if problems:
        raise ValueError(f"not an S5 reference: {problems}")
    return rec


def s5_window_class(rec: Mapping[str, Any]) -> Tuple[int, int, bool]:
    return int(rec["L"]), int(rec["S"]), bool(rec.get("thinking", False))


# --------------------------------------------------------------------------- #
# S5 reference ADMISSION and the launched rows' window-class COVERAGE                                                                  #
# --------------------------------------------------------------------------- #
#: the standup identity fields a PRIOR-tree reference is held to (the plan row's `readiness` block == the plan's statement of the standup; the
#: reference attempt's `exp/serve/endpoint.json` == what its standup attested) -- geometry (L, S, thinking), the engine path, the graph mode,
#: the LoRA geometry, the server's default policy, the served model + revision, and the served objects ⊆ this plan's adapter bank
S5_IDENTITY_ENDPOINT_OF_READINESS: Dict[str, str] = {"engine": "engine", "attested_max_model_len": "attested_max_model_len", "attested_max_num_seqs": "attested_max_num_seqs",
                                                     "thinking_mode": "thinking_mode", "server_default_policy": "server_default_policy"}
#: the ONE list of engine cudagraph modes that carry a FULL capture surface -- vLLM 0.25.1 attests `FULL_AND_PIECEWISE` (FULL decode graphs
#: for every capture size <= max_num_seqs, piecewise beyond) on every engine of this lane; the readiness contract (seedrep_served, which re-exports
#: this name) accepts it as FULL coverage when `attested_cudagraph_full_capture_max` covers S. The prior-tree admission holds a reference's standup
#: to the SAME predicate: a plan row requesting `FULL` is satisfied by an attested mode in this list whose capture max >= S; string equality on the
#: mode refused every S5 reference (7424, 11520, 24576) at the rehearsal.
CUDAGRAPH_MODES_WITH_FULL_CAPTURE: Tuple[str, ...] = ("FULL", "FULL_AND_PIECEWISE")


def cudagraph_full_capture_problems(*, attested_mode: Any, capture_max: Any, S: int) -> List[str]:
    """Why an attested engine graph mode does not give a FULL capture surface covering S (empty = it does): the mode is not in
    CUDAGRAPH_MODES_WITH_FULL_CAPTURE, the capture max is absent (a mandated attestation), or the capture max is below S."""
    out: List[str] = []
    mode = str(attested_mode or "").upper()
    if mode not in CUDAGRAPH_MODES_WITH_FULL_CAPTURE:
        out.append(f"attested_cudagraph_mode {attested_mode!r} has no FULL capture surface (accepted: {list(CUDAGRAPH_MODES_WITH_FULL_CAPTURE)})")
    if capture_max is None:
        out.append(f"attested_cudagraph_full_capture_max ABSENT (a mandated attestation: the FULL capture surface must cover S {int(S)})")
    elif int(capture_max) < int(S):
        out.append(f"cudagraph FULL capture max {capture_max} < S {int(S)} (the widest decode batch would dispatch PIECEWISE / eager)")
    return out
S5_REFERENCE_BASIS_OWN = "own"
S5_REFERENCE_BASIS_PRIOR = "prior_W_or_wave_D139"
S5_COVERAGE_OWN = "own_reference"
S5_COVERAGE_DOMINATED = "dominated"
#: a serving class used ONLY by ladder-consumer rung rows (`rung_mode == consume`) needs no reference before launch -- the rung's own RESULT
#: utilisation IS the class's measurement (read post-go by the first-rung check; recorded as an own-W reference once its store is COMPLETE);
#: the H200-h estimate rides on the rung's BASE row's class reference (`estimator_reference`), named by index
S5_COVERAGE_RUNG_SELF = "rung_self_measured"
S5_REFERENCE_BASIS_RUNG = "D154"
S5_MULTITURN_NONE_LAUNCHED = "none_launched"


def s5_dominates(ref_class: Sequence[Any], cls: Sequence[Any]) -> bool:
    """A reference of class ``ref_class`` proves ``cls`` fits iff the thinking flags agree AND L_ref >= L AND S_ref >= S with
    the classes distinct -- a larger window at the same or larger width (equal-L / larger-S and larger-L / equal-S both dominate). A larger window
    at a SMALLER width proves nothing about the wider class (the earlier rule certified (10240, 51) from (11520, 45) and (16384, 31) from
    (24576, 21); both were wrong)."""
    rl, rs_, rt = int(ref_class[0]), int(ref_class[1]), bool(ref_class[2])
    L, S, th = int(cls[0]), int(cls[1]), bool(cls[2])
    return rt == th and rl >= L and rs_ >= S and (rl, rs_) != (L, S)


def launch_set_problems(plan: Mapping[str, Any], launch_set: Any) -> List[str]:
    """Why ``launch_set`` is not a launch set of this plan (empty = it is; None = every planned row): a non-empty list of distinct standup ids,
    every one a row this plan may execute."""
    if launch_set is None:
        return []
    if not isinstance(launch_set, list) or not launch_set or not all(isinstance(x, str) and x for x in launch_set):
        return ["launch_set must be a non-empty list of standup ids (or null = every planned row)"]
    if len(set(launch_set)) != len(launch_set):
        return ["launch_set names a row twice"]
    known = {r["standup_id"] for r in _plan_rows_the_app_may_execute(plan)}
    unknown = sorted(set(launch_set) - known)
    return [f"launch_set names row(s) that are not this plan's: {unknown}"] if unknown else []


def parse_launch_set(spec: Optional[str]) -> Optional[list]:
    """`--launch-set` (certify, S0, the FINAL generator -- one parser): a comma-separated list of standup ids, or `@<file>` (one id per line /
    a JSON list); None = every planned row."""
    if spec is None or not str(spec).strip():
        return None
    text = str(spec).strip()
    if text.startswith("@"):
        raw = Path(text[1:]).expanduser().read_text()
        try:
            ids = json.loads(raw)
        except ValueError:
            ids = [ln.strip() for ln in raw.splitlines()]
        if not isinstance(ids, list):
            raise ValueError(f"--launch-set {text}: not a list of standup ids")
        return sorted({str(x).strip() for x in ids if str(x).strip()})
    return sorted({x.strip() for x in text.split(",") if x.strip()})


def final_s5_gate_problems(plan: Mapping[str, Any], s5: Mapping[str, Any], *, launch_set: Optional[Sequence[str]],
                           read_text: Optional[Callable[[str], Optional[str]]]) -> Tuple[List[str], Optional[Dict[str, Any]]]:
    """Gate 8 at plan generation. `launch_set` None (the whole roster): exactly the roster gate (`s5_gap` over `plan.s5_roster`, one
    reference per planned window class). With a launch set: the certificate's coverage (`s5_certificate_coverage` -- gaps only over the LAUNCHED
    rows' classes, admission of prior-tree references, two-dimensional dominance, rung_self_measured), the SAME helper G8 and certify
    run, returned so the generator records it beside the plan (RS/s5_coverage.json) for certify to compare. A rehearsal: the roster gate
    refused a 17-row launch set over seven unlaunched / rung-only classes that certify and G8 had already scoped."""
    refs = list(s5.get("references") or [])
    if launch_set is None:
        gap = s5_gap(plan, refs)
        return ([f"gate 8: planned standups without an S5 reference: {[(g['L'], g['S'], g['thinking']) for g in gap]}"] if gap else []), None
    ew = s5.get("execution_wave_uuid")
    if not ew:
        return ["gate 8: the S5 block names no execution_wave_uuid -- the launch-set-scoped coverage (D139 admission) cannot be derived"], None
    coverage = s5_certificate_coverage(plan, s5, execution_wave=str(ew), launch_set=launch_set, read_text=read_text)
    out: List[str] = []
    for i, adm in coverage["references"].items():
        if adm["problems"]:
            out.append(f"gate 8: S5.references[{i}] ({adm['class']}) is not an admissible S5 reference: {adm['problems']}")
    gap = [c for c in coverage["classes"] if c.get("gap")]
    if gap:
        out.append("gate 8: launched window classes without an S5 reference (own, dominated under D152, or rung-self-measured under D154): "
                   f"{[(tuple(g['class']), g.get('rows'), g['why']) for g in gap]} -- exclude the rows that use them by name")
    return out, coverage


def launched_plan_rows(plan: Mapping[str, Any], launch_set: Optional[Sequence[str]]) -> List[Mapping[str, Any]]:
    rows = _plan_rows_the_app_may_execute(plan)
    if launch_set is None:
        return list(rows)
    want = {str(x) for x in launch_set}
    return [r for r in rows if r["standup_id"] in want]


def _row_use(r: Mapping[str, Any]) -> str:
    return f"{r['leg']}:{r['stage']}" + (":dose" if r.get("dose_standup") else "")


def s5_launched_classes(plan: Mapping[str, Any], launch_set: Optional[Sequence[str]]) -> List[Dict[str, Any]]:
    """The roster entries the LAUNCHED rows use: each with the launched uses that need it; the whole roster when ``launch_set`` is None."""
    uses = {_row_use(r) for r in launched_plan_rows(plan, launch_set)}
    out = []
    for r in plan.get("s5_roster") or []:
        used = sorted(u for u in r.get("used_by", []) if launch_set is None or u in uses)
        if used:
            out.append({"L": int(r["L"]), "S": int(r["S"]), "thinking": bool(r["thinking"]), "used_by": used})
    return out


def s5_reference_admission(rec: Any, *, plan: Mapping[str, Any], execution_wave: str,
                           read_text: Optional[Callable[[str], Optional[str]]] = None) -> Dict[str, Any]:
    """ONE reference's admission (fc9): its schema and values (`s5_reference_problems` without the W / wave binding), then its BASIS -- `own`
    (this plan's W and the S5 execution wave) or, `prior_W_or_wave_D139`: measured under another tree and/or wave, admitted ONLY
    when its `attempt_root` names a root on the volume whose ATTEMPT.json is this record's (attempt id, W, wave) and whose standup record
    (`exp/serve/endpoint.json`) states the engine identity + geometry the plan's rows of that window class state (`readiness`: engine, L, S,
    thinking, graph mode, server default policy; the plan's model + revision; every served object one of this plan's adapter bank). Records
    `measured_under_W` / `measured_under_wave`, the identity compared and the qualification; a partial store stays a floor (refused as a
    reference by the schema check, so it never satisfies or dominates a class)."""
    out: Dict[str, Any] = {"class": None, "basis": None, "problems": [], "measured_under_W": None, "measured_under_wave": None, "identity": None, "qualification": None}
    if isinstance(rec, Mapping) and all(isinstance(rec.get(k), int) and not isinstance(rec.get(k), bool) for k in ("L", "S")):
        out["class"] = list(s5_window_class(rec))                       # named even for a record the schema refuses (a partial store: a floor, never a reference)
    probs = s5_reference_problems(rec)
    if probs:
        out["problems"] = probs
        return out
    L, S, th = s5_window_class(rec)
    out["measured_under_W"], out["measured_under_wave"] = rec["W"], rec["wave_uuid"]
    if rec["W"] == plan["W"] and rec["wave_uuid"] == execution_wave:
        out["basis"] = S5_REFERENCE_BASIS_OWN
        return out
    out["basis"] = S5_REFERENCE_BASIS_PRIOR
    root = rec.get("attempt_root")
    if not root:
        bound = s5_reference_problems(rec, W=plan["W"], wave_uuid=execution_wave)          # the W / wave binding, in the words the reader always used
        out["problems"] = bound + ["D139 admits a reference measured under another tree / wave only through its standup's records (`attempt_root` on the record), "
                                   "which this record does not name"]
        return out
    if read_text is None:
        out["problems"] = [f"measured under W {rec['W'][:12]} wave {rec['wave_uuid']}: D139 admission needs the volume reader to re-read {root}'s standup records"]
        return out
    rows = [r for r in _plan_rows_the_app_may_execute(plan) if s5_window_class({"L": r["geometry"]["L"], "S": r["geometry"]["S"], "thinking": r["geometry"]["thinking"]}) == (L, S, th)]
    if not rows:
        out["problems"] = [f"no planned row uses window class {(L, S, th)} -- there is no standup of this plan to hold the reference's engine to"]
        return out
    att_text, ep_text = read_text(f"{root}/ATTEMPT.json"), read_text(f"{root}/{SERVE_DIR_REL}/endpoint.json")
    if att_text is None or ep_text is None:
        out["problems"] = [f"{root}: ATTEMPT.json / {SERVE_DIR_REL}/endpoint.json not on the volume -- the measuring standup's identity cannot be read (D139)"]
        return out
    try:
        att, ep = json.loads(att_text), json.loads(ep_text)
    except ValueError as e:
        out["problems"] = [f"{root}: ATTEMPT.json / endpoint.json is not JSON ({e})"]
        return out
    probs = []
    for k, want in (("attempt_id", rec["attempt_id"]), ("W", rec["W"]), ("wave_uuid", rec["wave_uuid"])):
        if att.get(k) != want:
            probs.append(f"ATTEMPT.json {k} {att.get(k)!r} is not the record's {want!r}")
    readiness = rows[0].get("readiness") or {}
    identity: Dict[str, Any] = {"model": {"plan": plan.get("model"), "reference": ep.get("model")},
                                "model_revision": {"plan": plan.get("model_revision"), "reference": ep.get("model_revision", ep.get("revision"))}}
    for rk, ek in S5_IDENTITY_ENDPOINT_OF_READINESS.items():
        if rk in readiness:
            identity[rk] = {"plan": readiness[rk], "reference": ep.get(ek)}
    identity["L"] = {"plan": L, "reference": ep.get("attested_max_model_len")}
    identity["S"] = {"plan": S, "reference": ep.get("attested_max_num_seqs")}
    identity["thinking_mode"] = {"plan": ("on" if th else "off"), "reference": ep.get("thinking_mode")}
    lc = ep.get("lora_config") if isinstance(ep.get("lora_config"), Mapping) else {}
    for k in ("max_lora_rank", "max_loras"):
        if k in readiness:
            identity[k] = {"plan": readiness[k], "reference": lc.get(k)}
    for k, v in identity.items():
        if v["plan"] is not None and v["plan"] != v["reference"]:
            probs.append(f"{k}: the reference's standup attests {v['reference']!r}, this plan's standup for class {(L, S, th)} states {v['plan']!r}")
    # fc9e: the graph mode is held to the readiness contract's PREDICATE, never string equality -- the plan row requests `FULL`; an attested mode
    # with a FULL capture surface (CUDAGRAPH_MODES_WITH_FULL_CAPTURE) whose capture max covers S satisfies it; both values are recorded
    if "cudagraph_mode" in readiness:
        requested = str(readiness["cudagraph_mode"])
        cg = cudagraph_full_capture_problems(attested_mode=ep.get("attested_cudagraph_mode"), capture_max=ep.get("attested_cudagraph_full_capture_max"), S=S)
        if requested.upper() not in CUDAGRAPH_MODES_WITH_FULL_CAPTURE:
            cg.append(f"this plan's standup requests cudagraph_mode {requested!r}, not a FULL-capture mode ({list(CUDAGRAPH_MODES_WITH_FULL_CAPTURE)}) -- the admission has no predicate for it")
        identity["cudagraph_mode"] = {"plan": requested, "reference": ep.get("attested_cudagraph_mode"), "reference_full_capture_max": ep.get("attested_cudagraph_full_capture_max"),
                                      "S": S, "rule": f"a requested FULL surface is satisfied by an attested mode in {list(CUDAGRAPH_MODES_WITH_FULL_CAPTURE)} with full_capture_max >= S",
                                      "problems": cg}
        probs += [f"cudagraph_mode: {x}" for x in cg]
    served = {str(c): v for c, v in (ep.get("object_artifact_sha256_by_cell") or {}).items() if v}
    # fc9f: the bank is DERIVED from the planned rows (`plan_adapter_bank`), never read off the summary alone -- an emptied / removed
    # summary used to admit foreign bytes; a summary inconsistent with the rows refuses by object; a base-only plan (no fitted object pinned) is exempt
    bank_info = plan_adapter_bank(plan)
    probs += bank_info["problems"]
    foreign = sorted(c for c, v in served.items() if v not in bank_info["bank"])
    identity["objects_in_plan_adapter_bank"] = {"served": sorted(served), "foreign": foreign, "bank_size": bank_info["n_objects"], "path": bank_info["path"],
                                                "sources": bank_info["sources"], "problems": bank_info["problems"]}
    if foreign:
        probs.append(f"the reference's standup served object(s) {foreign} whose bytes are not in this plan's adapter bank ({bank_info['n_objects']} object(s) pinned by its rows)")
    out["identity"] = identity
    if probs:
        out["problems"] = [f"D139: measured under W {rec['W'][:12]} wave {rec['wave_uuid']}; the standup's engine identity does not match this plan's standup for class {(L, S, th)}: " + "; ".join(probs)]
        return out
    out["qualification"] = (f"D139: measured under W {rec['W'][:12]} wave {rec['wave_uuid']} (not this plan's W {plan['W'][:12]} / S5 wave); the measuring standup's engine "
                            f"identity + geometry match this plan's standup for class {(L, S, th)} ({', '.join(sorted(k for k in identity if k != 'objects_in_plan_adapter_bank'))}; "
                            + f"{len(served)} served object(s) in this plan's adapter bank, {bank_info['n_objects']} object(s) pinned by its rows, bank {bank_info['path']})")
    return out


def s5_certificate_coverage(plan: Mapping[str, Any], s5: Mapping[str, Any], *, execution_wave: str, launch_set: Optional[Sequence[str]],
                            read_text: Optional[Callable[[str], Optional[str]]] = None) -> Dict[str, Any]:
    """The certificate's DERIVED `s5_coverage` block (fc9): every reference's admission (`references[i]`), and per LAUNCHED window class how it
    is satisfied -- `own_reference` (a complete admissible reference of the class), `dominated` by the SMALLEST complete admissible reference of a
    LARGER window at the same thinking mode (a larger window on the same engine geometry proves the smaller fits; recorded as `dominated_by`), or a
    `gap` (the rows using it must be excluded by name). `multiturn` = `none_launched` when no multi-turn row is in the launch set."""
    refs = list(s5.get("references") or [])
    admissions = {str(i): s5_reference_admission(r, plan=plan, execution_wave=execution_wave, read_text=read_text) for i, r in enumerate(refs)}
    good = [(int(i), refs[int(i)], adm) for i, adm in admissions.items() if not adm["problems"]]
    all_rows = _plan_rows_the_app_may_execute(plan)
    launched = launched_plan_rows(plan, launch_set)

    def _cls(r: Mapping[str, Any]) -> Tuple[int, int, bool]:
        g = r["geometry"]
        return int(g["L"]), int(g["S"]), bool(g["thinking"])

    def _ref_block(i: int, r: Mapping[str, Any], a: Mapping[str, Any]) -> Dict[str, Any]:
        return {"reference_index": i, "reference": {"attempt_id": r["attempt_id"], "W": r["W"], "wave_uuid": r["wave_uuid"]}, "basis": a["basis"],
                "measured_under_W": a["measured_under_W"], "measured_under_wave": a["measured_under_wave"]}

    def standard(cls: Tuple[int, int, bool]) -> Optional[Dict[str, Any]]:
        """own reference, else the SMALLEST dominating reference (same thinking, L_ref >= L, S_ref >= S), else None."""
        key = list(cls)
        own = [(i, r, a) for i, r, a in good if a["class"] == key]
        if own:
            i, r, a = own[0]
            return {"satisfied_by": S5_COVERAGE_OWN, **_ref_block(i, r, a), "dominated_by": None, "qualification": a["qualification"]}
        dom = sorted(((i, r, a) for i, r, a in good if s5_dominates(a["class"], key)), key=lambda x: (x[2]["class"][0], x[2]["class"][1]))
        if dom:
            i, r, a = dom[0]
            return {"satisfied_by": S5_COVERAGE_DOMINATED, **_ref_block(i, r, a), "dominated_by": a["class"],
                    "qualification": (f"D146 / D152: no reference of class {tuple(key)}; dominated by the complete reference of class {tuple(a['class'])} -- the same "
                                      f"thinking mode, L {a['class'][0]} >= {key[0]} and S {a['class'][1]} >= {key[1]} (a larger window at no smaller width proves the smaller fits)"
                                      + (f"; {a['qualification']}" if a["qualification"] else ""))}
        return None

    classes = []
    for c in s5_launched_classes(plan, launch_set):
        cls = (c["L"], c["S"], c["thinking"])
        key = list(cls)
        users = [r for r in launched if _cls(r) == cls]
        entry: Dict[str, Any] = {"class": key, "used_by": c["used_by"], "rows": sorted(r["standup_id"] for r in users)}
        std = standard(cls)
        if std is not None:
            entry.update(std)
            classes.append(entry)
            continue
        rung_only = bool(users) and all(is_ladder_consumer_row(r) for r in users)     # fc9d: the ONE ladder-consumer predicate (kind, stage, mode, binding)
        if rung_only:
            # the rung's own RESULT is the measurement; the estimate rides on the BASE row's class reference (own or dominated)
            bases: Dict[Tuple[int, int, bool], List[str]] = {}
            unbased: List[str] = []
            for r in users:
                base = next((b for b in all_rows if b["leg"] == r["leg"] and b["engine"] == r["engine"] and str(b["stage"]) == STAGE_BASE), None)
                if base is None:
                    unbased.append(r["standup_id"])
                else:
                    bases.setdefault(_cls(base), []).append(r["standup_id"])
            est = {bc: standard(bc) for bc in bases}
            missing = {bc: sids for bc, sids in bases.items() if est[bc] is None}
            if unbased or missing:
                entry.update(gap=True, why=(f"class {tuple(key)} is used by ladder-consumer rung rows only (D154) but " + "; ".join(
                    ([f"rows {unbased} have no base row on this plan"] if unbased else [])
                    + [f"rows {sids}' base class {tuple(bc)} has no complete admissible reference and none of a dominating class (D152), so the rung's H200-h "
                       "estimate has no stated band" for bc, sids in missing.items()])))
                classes.append(entry)
                continue
            entry.update(satisfied_by=S5_COVERAGE_RUNG_SELF, basis=S5_REFERENCE_BASIS_RUNG, reference=None, reference_index=None, dominated_by=None,
                         measured_under_W=None, measured_under_wave=None,
                         estimator_reference={f"{bc[0]}x{bc[1]}{'T' if bc[2] else ''}": {"base_class": list(bc), "rows": sids, **{k: est[bc][k] for k in
                                              ("reference_index", "reference", "satisfied_by", "dominated_by", "basis")}} for bc, sids in bases.items()},
                         qualification=(f"D154: class {tuple(key)} is used only by ladder-consumer rung rows {sorted(r['standup_id'] for r in users)}; no pre-launch "
                                        "reference is required -- the rung's own RESULT utilisation is this class's measurement (read post-go by the D148 first-rung "
                                        "check; recorded as an own-W reference by seedrep_s5_record.py reference once its store is COMPLETE); the H200-h estimate "
                                        f"rides on the base row's class reference(s) {sorted(f'{bc[0]}x{bc[1]}: index {est[bc]['reference_index']}' for bc in bases)}"))
            classes.append(entry)
            continue
        entry.update(gap=True, why=(f"no complete admissible reference of class {tuple(key)} and none dominating it under D152 (same thinking, L >= {key[0]}, S >= {key[1]}); "
                                    f"used by {'base / non-consumer' if users else 'no launched'} row(s) {sorted(r['standup_id'] for r in users)}"
                                    + (f" (not ladder consumers: {sorted(r['standup_id'] for r in users if not is_ladder_consumer_row(r))})" if users else "")
                                    + " -- exclude them by name "
                                    f"(admissible classes: {sorted(tuple(a['class']) for _i, _r, a in good)})"))
        classes.append(entry)
    launched_mt = sorted({str(r["leg"]) for r in launched_plan_rows(plan, launch_set) if r.get("leg") in MULTITURN_LEGS})
    return {"rule": "D139 / D145 / D146 / D152 / D154 (fc9, fc9b, fc9c)", "launch_set": (sorted({str(x) for x in launch_set}) if launch_set is not None else None),
            "references": admissions, "classes": classes,
            "multiturn": (S5_MULTITURN_NONE_LAUNCHED if not launched_mt else {"launched_legs": launched_mt})}


S5_BLOCK_WRAPPER_KEYS: Tuple[str, ...] = ("W", "execution_wave_uuid", "production_wave_uuid", "launch_plan_sha")


def s5_block_bind(doc: Dict[str, Any], *, W: str, execution_wave_uuid: str, production_wave_uuid: str, launch_plan_sha: str) -> Dict[str, Any]:
    """The block's WRAPPER: the tree ``W``, the smoke's EXECUTION wave (every record inside carries it as `wave_uuid`),
    and the PRODUCTION target the records certify (`production_wave_uuid` + `launch_plan_sha`). Written once; a block already bound to
    another value refuses (a copied / foreign block never re-binds)."""
    check_wave_uuid(execution_wave_uuid)
    check_wave_uuid(production_wave_uuid)
    if execution_wave_uuid == production_wave_uuid:
        raise ValueError("the S5 execution wave equals the production wave: no smoke runs under the production wave (smoke_identity refuses it)")
    s5 = doc.setdefault("S5", {})
    want = {"W": W, "execution_wave_uuid": execution_wave_uuid, "production_wave_uuid": production_wave_uuid, "launch_plan_sha": launch_plan_sha}
    for k, v in want.items():
        if k in s5 and s5[k] != v:
            raise ValueError(f"the S5 block is already bound to {k}={s5[k]!r}, not {v!r} (a block of another wave / plan / tree is never re-bound)")
    s5.update(want)
    s5["identity_rule"] = WAVE_IDENTITY_RULE
    return doc


def s5_block_problems(s5: Any, *, plan: Mapping[str, Any], draft_launch_plan_sha: Optional[str] = None) -> List[str]:
    """Why an S5 block is not THIS plan's certification input: no wrapper, a wrapper for another W / production wave / plan sha, or an
    execution wave that is the production wave. The smokes run against the DRAFT, so the wrapper's `launch_plan_sha` is the
    draft's: it must equal ``plan``'s sha or ``draft_launch_plan_sha`` (the draft a FINAL was held to by `compare_draft_final`)."""
    if not isinstance(s5, Mapping):
        return ["no S5 block"]
    out = [f"the S5 block carries no {k} (write it with seedrep_s5_record.py; the wrapper binds the smoke's records to this plan)"
           for k in S5_BLOCK_WRAPPER_KEYS if not s5.get(k)]
    if out:
        return out
    if s5["W"] != plan["W"]:
        out.append(f"the S5 block is for W {s5['W']!r}, not this plan's {plan['W']}")
    if s5["production_wave_uuid"] != plan["wave_uuid"]:
        out.append(f"the S5 block targets production wave {s5['production_wave_uuid']!r}, not this plan's {plan['wave_uuid']}")
    admissible = {plan.get("launch_plan_sha")} | ({draft_launch_plan_sha} if draft_launch_plan_sha else set())
    if s5["launch_plan_sha"] not in admissible:
        out.append(f"the S5 block was bound to launch_plan_sha {s5['launch_plan_sha']!r}, not this plan's {plan.get('launch_plan_sha')!r}"
                   + (f" nor its draft's {draft_launch_plan_sha!r}" if draft_launch_plan_sha else ""))
    try:
        check_wave_uuid(str(s5["execution_wave_uuid"]))
    except ValueError as e:
        out.append(f"execution_wave_uuid: {e}")
    if s5["execution_wave_uuid"] == plan["wave_uuid"]:
        out.append("the S5 block's execution wave IS the production wave -- no smoke runs there; the records measured nothing this lane stood up")
    return out


def s5_smokes_add_reference(doc: Dict[str, Any], rec: Mapping[str, Any], *, W: str, wave_uuid: str, replace: bool = False,
                            plan: Optional[Mapping[str, Any]] = None, read_text: Optional[Callable[[str], Optional[str]]] = None) -> Dict[str, Any]:
    """Merge ONE validated reference into RS/smokes.json's document (`S5.references[]`), one per window class; a second record of the same
    class refuses unless ``replace``. ``wave_uuid`` = the `S5` smoke's EXECUTION wave (the standup that measured; its `wave_uuid`), never
    the production wave. A reference measured under another tree / wave is admitted when ``plan`` and ``read_text``
    are given and `s5_reference_admission` holds its standup's engine identity to the plan's (the CLI passes both); otherwise refused as before."""
    if plan is not None:
        adm = s5_reference_admission(rec, plan=plan, execution_wave=wave_uuid, read_text=read_text)
        problems = adm["problems"]
    else:
        problems = s5_reference_problems(rec, W=W, wave_uuid=wave_uuid)
    if problems:
        raise ValueError(f"refusing to record an S5 reference: {problems}")
    refs = doc.setdefault("S5", {}).setdefault("references", [])
    cls = s5_window_class(rec)
    dup = [i for i, r in enumerate(refs) if isinstance(r, Mapping) and all(k in r for k in ("L", "S")) and s5_window_class(r) == cls]
    if dup and not replace:
        raise ValueError(f"an S5 reference for window class (L, S, thinking) = {cls} is already recorded; pass --replace to supersede it")
    for i in reversed(dup):
        del refs[i]
    refs.append(dict(rec))
    return doc


def s5_smokes_add_window(doc: Dict[str, Any], rec: Mapping[str, Any], *, wave_uuid: str, replace: bool = False) -> Dict[str, Any]:
    """Merge ONE validated multi-turn window measurement into `S5.multiturn[]` (one per (leg, arm)); refuses a record of another wave
    (``wave_uuid`` = the smoke's EXECUTION wave)."""
    problems = window_measurement_problems(rec)
    if not problems and rec.get("wave_uuid") != wave_uuid:
        problems.append(f"wave_uuid {rec.get('wave_uuid')!r} is not the S5 smoke's execution wave {wave_uuid}")
    if problems:
        raise ValueError(f"refusing to record an S5 window measurement: {problems}")
    rows = doc.setdefault("S5", {}).setdefault("multiturn", [])
    dup = [i for i, r in enumerate(rows) if isinstance(r, Mapping) and (r.get("leg"), r.get("arm")) == (rec["leg"], rec["arm"])]
    if dup and not replace:
        raise ValueError(f"an S5 window measurement for ({rec['leg']}, {rec['arm']}) is already recorded; pass --replace to supersede it")
    for i in reversed(dup):
        del rows[i]
    rows.append(dict(rec))
    return doc


def adapter_hash_set(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Gate 10: the DERIVED set of adapters production expects to RE-USE — the union of every planned standup's cells
    (at this roster 17: 15 fitted objects + none__lora0 + the Arditi vector_all carrier), never a hard-coded count."""
    seen: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        for cell, a in r["adapters"].items():
            key = a["staged_id"]
            seen.setdefault(key, {"staged_id": key, "object_sha256": a["object_sha256"], "intervention_hash": a["intervention_hash"],
                                  "cells": set(), "standups": 0})
            seen[key]["cells"].add(cell)
            seen[key]["standups"] += 1
    return [{**v, "cells": sorted(v["cells"])} for _, v in sorted(seen.items())]


ADAPTER_BANK_SUMMARY = "summary"
ADAPTER_BANK_DERIVED = "derived_from_rows"


def plan_adapter_bank(plan: Mapping[str, Any]) -> Dict[str, Any]:
    """The adapter BANK a reference's served objects are held to (fc9f / fc9g): the UNION over standups + rungs of the object
    sha256 pins in row["objects"] (the authoritative executable pins -- the field the container asserts at runtime) and in the object-bearing
    row["adapters"] entries (the re-use expectation), and the two representations must AGREE per cell and hash: a cell pinned in one and absent or
    different in the other is a problem by name (refused at admission AND by validate_plan). The plan's summary (`adapter_hash_set`) is never the
    authority; one inconsistent with the derived bank is a problem by object. No plan-shape carve-out: an empty bank admits a reference that
    served nothing and refuses any served object as foreign. `path`: `summary` (present, equal to the bank) | `derived_from_rows` (absent / empty)."""
    rows = list(plan.get("standups") or []) + list(plan.get("rungs") or [])
    problems: List[str] = []
    from_objects: Dict[str, str] = {}
    from_adapters: Dict[str, str] = {}
    for r in rows:
        sid = r.get("standup_id", "?")
        objs = r.get("objects") if isinstance(r.get("objects"), Mapping) else {}
        ads = r.get("adapters") if isinstance(r.get("adapters"), Mapping) else {}
        o_pins = {str(c): str(v.get("object_sha256")) for c, v in objs.items() if isinstance(v, Mapping) and v.get("object_sha256")}
        a_pins = {str(c): str(v.get("object_sha256")) for c, v in ads.items() if isinstance(v, Mapping) and v.get("object_sha256")}
        for c in sorted(set(o_pins) | set(a_pins)):
            if o_pins.get(c) != a_pins.get(c):
                problems.append(f"{sid}: cell {c!r} object pin disagrees between objects ({(o_pins.get(c) or 'absent')[:12]}) and adapters "
                                f"({(a_pins.get(c) or 'absent')[:12]}) -- the executable pin and the re-use expectation must name the same bytes")
        from_objects.update({f"{sid}/{c}": v for c, v in o_pins.items()})
        from_adapters.update({f"{sid}/{c}": v for c, v in a_pins.items()})
    derived = set(from_objects.values()) | set(from_adapters.values())
    summary_entries = [a for a in (plan.get("adapter_hash_set") or []) if isinstance(a, Mapping)]
    summary = {a["object_sha256"] for a in summary_entries if a.get("object_sha256")}
    if summary_entries:
        omitted, extra = sorted(derived - summary), sorted(summary - derived)
        if omitted:
            problems.append(f"the plan's adapter_hash_set omits {len(omitted)} object(s) its rows pin ({[x[:12] for x in omitted]}) -- the summary is inconsistent with the rows")
        if extra:
            problems.append(f"the plan's adapter_hash_set names {len(extra)} object(s) no planned row pins ({[x[:12] for x in extra]}) -- the summary is inconsistent with the rows")
    return {"bank": derived, "path": (ADAPTER_BANK_SUMMARY if summary_entries else ADAPTER_BANK_DERIVED), "problems": problems, "n_objects": len(derived),
            "sources": {"objects": len(from_objects), "adapters": len(from_adapters)}, "summary_entries": len(summary_entries), "n_rows": len(rows)}


def adapter_gap(plan: Mapping[str, Any], bank_hashes: Iterable[str]) -> Dict[str, Any]:
    """Gate 10's check after S1 / S1b / S2: every derived adapter has an intervention hash AND that hash is in the bank."""
    bank = set(bank_hashes)
    missing_hash = [a["staged_id"] for a in plan["adapter_hash_set"] if not a.get("intervention_hash")]
    not_banked = [a["staged_id"] for a in plan["adapter_hash_set"] if a.get("intervention_hash") and a["intervention_hash"] not in bank]
    return {"n_adapters": len(plan["adapter_hash_set"]), "missing_hash": missing_hash, "not_banked": not_banked,
            "ok": not missing_hash and not not_banked}


def capacity_ledger(plan: Mapping[str, Any]) -> Dict[str, Any]:
    """Containers and GPUs counted SEPARATELY: targets (every base row), pools (one H200 each; the
    row's `gpus` follows its `gpu` spec, so an `H200:<n>` pool still counts n), rungs as
    DECLARED per (leg, view|arditi, cap) x max_containers — the binding figure the staged-launch rule reads."""
    targets = plan["standups"]
    pools = plan["pools"]
    rungs = plan["rungs"]
    t_c = len(targets)
    p_c = len(pools)
    p_g = sum(int(p["gpus"]) for p in pools)
    r_c = sum(int(r["modal"]["max_containers"]) for r in rungs)
    d_c = sum(1 for r in targets if r.get("dose_standup"))
    by_leg: Dict[str, Dict[str, int]] = {}
    for r in targets:
        label = f"dose:{r['leg']}" if r.get("dose_standup") else r["leg"]
        by_leg.setdefault(label, {"target_containers": 0, "rung_containers": 0})["target_containers"] += 1
    for r in rungs:
        by_leg.setdefault(r["leg"], {"target_containers": 0, "rung_containers": 0})["rung_containers"] += int(r["modal"]["max_containers"])
    return {"schema": LEDGER_SCHEMA, "materialisation": plan["materialisation"], "W7": plan["W7"], "wave_uuid": plan["wave_uuid"],
            "targets": {"containers": t_c, "gpus": t_c},
            "pools": {"containers": p_c, "gpus": p_g},
            "rungs_declared": {"functions": len(rungs), "containers": r_c, "gpus": r_c},
            "rungs_deferred": {"functions": len(plan.get("deferred_rungs") or []), "rows": [d["standup_id"] for d in (plan.get("deferred_rungs") or [])],
                               "why": None},                      # nothing is deferred -- every rung of the fixed battery is executable
            "gsm8k_rungs": {"functions": sum(1 for r in rungs if r["leg"] == "gsm8k"), "rule": GSM8K_RUNG_RULE},
            "targets_plus_pools": {"containers": t_c + p_c, "gpus": t_c + p_g},
            "unstaged_peak": {"containers": t_c + p_c + r_c, "gpus": t_c + p_g + r_c},
            "unstaged_peak_without_dose": {"containers": t_c - d_c + p_c + r_c, "gpus": t_c - d_c + p_g + r_c},
            "by_leg": by_leg,
            "dose_standups": {"containers": d_c, "views": sorted({r["engine"] for r in targets if r.get("dose_standup")}),
                              "status": (plan.get("dose_standups") or {}).get("status"),
                              "containers_when_registered": ((plan.get("dose_standups") or {}).get("pending") or {}).get("containers_when_registered", d_c),
                              "note": "D43 / D68: inside the peak; one single-container engine per (leg, peer seed-1 dose view), counted in targets; "
                                      "`unstaged_peak_without_dose` is the total the s1-s3 legs need on their own"},
            "paired_route_throughput": ("identical to single-gate at these legs' shapes (ReLU, D50: ActorAttack 10.1 ms/step median, Crescendo 11.06, "
                                        "prompt-heavy tail 14.4) -- paired and LoRA children on one dose standup share one throughput model"),
            "rule": "if the workspace quota is below unstaged_peak the launch is STAGED: multi-turn (the critical path) first, "
                    "then the short legs as capacity frees; pool lifetimes sized for QUEUED as well as running drivers"}


# --------------------------------------------------------------------------- #
# validation (the known-bad rules), draft == final, provenance                   #
# --------------------------------------------------------------------------- #


def _flag_value(argv: Sequence[str], flag: str) -> Optional[str]:
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


def _flag_values(argv: Sequence[str], flag: str) -> List[str]:
    return [argv[i + 1] for i, a in enumerate(argv) if a == flag and i + 1 < len(argv)]


def validate_plan(plan: Mapping[str, Any], *, table: Any = None) -> List[str]:
    """Every rule of the module docstring over every row; ``[]`` = valid. ``table`` = the serving geometry module
    (``sisteer.capeval.serving.geometry``) when importable: each row's window is then resolved through the REAL front's
    rule — a table window must resolve `window_source table` at the row's L; a GSM8K eval that the table does not know
    refuses; a non-table pair must be the identity pair."""
    problems: List[str] = []
    problems += client_python_problems(plan)      # every Inspect-side argv names the CLIENT venv python
    rows = list(plan.get("standups", [])) + list(plan.get("rungs", []))
    if not rows:
        return ["plan has no standups"]
    # fc9d: every row's shape must agree with itself and its collection (kind / stage / rung_mode / consumer binding) -- the ONE rung definition
    for coll in ("standups", "rungs"):
        for r in plan.get(coll, []):
            problems += row_shape_problems(r, coll)
    # fc9g: the object pins agree between objects (executable) and adapters (re-use expectation) on every row, and the summary equals the derived bank
    problems += plan_adapter_bank(plan)["problems"]
    mmlu_rows = [r for r in rows if "mmlu" in r["eval_legs"]]
    for r in rows:
        sid, argv, g = r["standup_id"], r["argv"], r["geometry"]
        L, S = int(g["L"]), int(g["S"])
        if S * L > KERNEL_CEILING:
            problems.append(f"{sid}: S x L = {S * L:,} exceeds the LoRA kernel ceiling {KERNEL_CEILING:,}")
        if S > HYBRID_SEQS_CAP or S < 1:
            problems.append(f"{sid}: S {S} outside [1, {HYBRID_SEQS_CAP}]")
        if S != min(int(g["identity_S"]), kernel_S(L)):
            problems.append(f"{sid}: S {S} != min(identity {g['identity_S']}, kernel {kernel_S(L)})")
        if _flag_value(argv, "--serving-path") != SERVING_PATH:
            problems.append(f"{sid}: --serving-path must be {SERVING_PATH}")
        if "--enable-log-requests" not in argv:
            problems.append(f"{sid}: --enable-log-requests missing (F22_LOG_REQUESTS=1 is criterion 2's evidence)")
        if _flag_value(argv, "--server-default-policy") != SERVER_DEFAULT_POLICY:
            problems.append(f"{sid}: --server-default-policy must be {SERVER_DEFAULT_POLICY}")
        if _flag_value(argv, "--model-key") != MODEL_KEY:
            problems.append(f"{sid}: --model-key must be {MODEL_KEY}")
        if _flag_value(argv, "--eval") != g["eval"]:
            problems.append(f"{sid}: --eval {_flag_value(argv, '--eval')!r} != geometry.eval {g['eval']!r}")
        has_L, has_S = _flag_value(argv, "--max-model-len"), _flag_value(argv, "--max-num-seqs")
        if r["leg"] == "gsm8k":
            if has_L is not None or has_S is not None:
                problems.append(f"{sid}: a GSM8K standup passes --eval ONLY (found --max-model-len {has_L!r} / --max-num-seqs {has_S!r}); "
                                "an explicit window is stamped `override` and the reader refuses it (§2.2 rev 4)")
            if g["eval"] not in (GSM8K_COMMON_EVAL, GSM8K_RUNG_COMMON_EVAL):
                problems.append(f"{sid}: GSM8K eval {g['eval']!r} is not the common class")
            if "--thinking-on" not in argv:
                problems.append(f"{sid}: GSM8K needs --thinking-on")
        elif "--thinking-on" in argv:
            problems.append(f"{sid}: --thinking-on outside GSM8K")
        if g["table"]:
            if has_L is not None or has_S is not None:
                problems.append(f"{sid}: a TABLE window passes no --max-model-len / --max-num-seqs (S is the core's to derive)")
        else:
            if has_L is None or has_S is None:
                problems.append(f"{sid}: a NON-table window passes the identity PAIR (--max-model-len L --max-num-seqs identity_seqs(L))")
            elif int(has_L) != L or int(has_S) != int(g["identity_S"]):
                problems.append(f"{sid}: non-table pair ({has_L}, {has_S}) != identity pair ({L}, {g['identity_S']})")
            if "--allow-over-budget" in argv:
                problems.append(f"{sid}: --allow-over-budget is never used")
        if r in mmlu_rows:
            if _flag_value(argv, "--max-logprobs") != str(MMLU_TOP_LOGPROBS):
                problems.append(f"{sid}: serves MMLU but carries no --max-logprobs {MMLU_TOP_LOGPROBS} (vLLM's default 20 refuses "
                                "the driver's top_logprobs 64 — proof-set gate 9)")
        cells = _flag_values(argv, "--cell")
        if ["none"] + cells != r["route_set"]:
            problems.append(f"{sid}: --cell {cells} does not select the expected route set {r['route_set']}")
        digests = dict(d.split("=", 1) for d in _flag_values(argv, "--expect-object-sha256"))
        fitted = [c for c in cells if not c.startswith("prompted__")]
        if sorted(digests) != sorted(fitted):
            problems.append(f"{sid}: --expect-object-sha256 names {sorted(digests)} != the fitted cells {sorted(fitted)}")
        for c, sha in digests.items():
            if r["objects"].get(c, {}).get("object_sha256") != sha or len(sha) != 64:
                problems.append(f"{sid}: pin for {c} disagrees with the objects table")
        if r["engine"] == ARDITI and cells != [ARDITI_CELL]:
            problems.append(f"{sid}: the Arditi engine serves ONE object (D2's fault); got {cells}")
        if r.get("dose_standup"):
            if r["engine"] in ENGINES or r["engine"] not in ((plan.get("dose_standups") or {}).get("views") or {}):
                problems.append(f"{sid}: a dose standup must ride a DISCOVERED seed-1 dose view (dose_standups.views), never s1 / s2 / s3 / arditi (D43: matched engines)")
            if any(p in cells for p in PROMPTED_CELLS) or r["leg"] not in DOSE_STANDUP_LEGS or r["kind"] != "base" or r["producers"]:
                problems.append(f"{sid}: a dose standup serves the dose cells + none only, on {DOSE_STANDUP_LEGS}, base stage, no ladder")
        elif r["engine"] != ARDITI and any(p not in cells for p in PROMPTED_CELLS):
            problems.append(f"{sid}: a view engine requests both prompted routes explicitly")
        if r["engine"] in VIEWS:
            dose_cells = [c for c, o in r["objects"].items() if o.get("dose_view") or DOSE_S1_STAGED_RE.match(str(o.get("staged_id", "")))]
            if dose_cells:
                problems.append(f"{sid}: dose objects {dose_cells} registered in an s-view standup (D43 refuses: moves max_loras / hashes / kernel budget for one seed)")
        if not r["children"] and r["leg"] != "petri":
            problems.append(f"{sid}: no driver children")
        for k in r["children"]:
            if k["argv"] is None and r["leg"] != "petri":
                problems.append(f"{sid}: child {k['arm']} s{k['s']} has no argv")
            if not k.get("registry_identity_rel", "").startswith(PLANS_REL + "/"):
                problems.append(f"{sid}: child {k['arm']} s{k['s']} names no registry identity file")
        if table is not None:
            problems += _table_problems(r, table)
    # gate 6: every Dict name / key carries the six axes
    for name in plan["dicts"].values():
        if not name.startswith(f"seedrep-{plan['W7']}-{plan['wave_uuid'][:8]}-"):
            problems.append(f"dict {name} lacks the W7 / wave_uuid axes")
    for r in rows:
        k = parse_control_key(r["control_key"])
        if (k["W7"], k["wave_uuid"], k["leg"], k["view"], k["stage"], k["shard"]) != \
                (plan["W7"], plan["wave_uuid"], r["leg"], r["engine"], r["stage"], str(r["shard"])):
            problems.append(f"{r['standup_id']}: control key {r['control_key']} does not carry its six axes")
    keys = [r["control_key"] for r in rows]
    if len(set(keys)) != len(keys):
        problems.append("control keys collide across rows")
    akeys = [r.get("accepted_key") for r in rows]
    if len(set(akeys)) != len(akeys) or any(not k or ACCEPTED_KEY_SEP not in str(k) for k in akeys):
        problems.append("accepted keys missing or colliding across rows (WS2-E's `|` spelling is required)")
    try:
        check_wave_uuid(plan["wave_uuid"])
    except ValueError as e:
        problems.append(str(e))
    problems += pool_coverage_problems(plan)
    problems += private_run_problems(plan)
    problems += s5_probe_registration_problems(plan)
    for r in plan.get("s5_probe_rows") or []:
        probe_ids = {c["staged_id"] for c in S5_PROBE_CELLS}
        served = [o.get("staged_id") for c, o in r["objects"].items() if c != "none"]
        if r.get("kind") != S5_PROBE_KIND or not r.get("smoke_only") or r["leg"] not in S5_PROBE_LEGS or len(served) != 1 or served[0] not in probe_ids \
                or r["route_set"][0] != "none" or r["standup_id"] in {x["standup_id"] for x in plan["standups"]}:
            problems.append(f"{r['standup_id']}: an S5 probe row must be smoke-only, on a probe leg, serving `none` + ONE probe cell, and never a production standup")
    if plan.get("dose_standups", {}).get("status") == DOSE_STATUS_NOT_IN_WAVE and any(r.get("dose_standup") for r in plan["standups"]):
        problems.append("D71: dose standups present in the production plan while the dose status is not_in_this_wave")
    for r in _plan_rows_the_app_may_execute(plan):
        # W15d (launch F-L(d)): gate 10 per the row's side of the contract -- a production row expects every adapter re-used; a smoke
        # row (one carrying `smoke`) expects what its §8 role derives, and a first-export smoke row expecting True is refused here
        problems += smoke_adapter_problems(r)
        # a production row's child never carries a smoke-only item-budget flag (BoN --limit-goals / misalign --limit-items:
        # a limited store is never a wave result); an `S1`-`S4` smoke's BoN / misalign child carries exactly its budget, an `S5` reference none
        problems += smoke_budget_problems(r)
    for r in rows:
        exp_served = cells_served_for_routes(r["route_set"])
        if r.get("cells_served_expected") != exp_served or r.get("route_set_harness") != exp_served["canonical"]:
            problems.append(f"{r['standup_id']}: the harness-id route set / cells_served expectation is not the route set translated + none__lora0")
        if r["kind"] == "rung":
            if r.get("rung_mode") == RUNG_MODE_CONSUME:
                rc = r.get("rung_consumer") or []
                if not rc or rc[2:3] != ["consume"] or LADDER_KIND_OF_LEG.get(r["leg"]) is None or _flag_value(rc, "--kind") != LADDER_KIND_OF_LEG[r["leg"]] \
                        or _flag_value(rc, "--cap-next") != r["stage"] or _flag_value(rc, "--leg") != r["leg"] or _flag_value(rc, "--view") != r["engine"] \
                        or "--finish-producer" in rc or _flag_value(rc, "--wave-uuid") != plan["wave_uuid"]:
                    problems.append(f"{r['standup_id']}: rung consumer argv is not WS2-E's `consume` for this queue: {rc[:12]}")
                if _flag_value(rc, "--width") != str(int(r["geometry"]["S"])) or _flag_value(rc, "--volume") != MODAL_VOLUME_NAME:
                    problems.append(f"{r['standup_id']}: rung consumer --width {_flag_value(rc, '--width')!r} / --volume {_flag_value(rc, '--volume')!r} "
                                    f"!= the attested S {r['geometry']['S']} / {MODAL_VOLUME_NAME} (WS2-E f4f3d465: width REQUIRED, the Volume bound)")
                if _flag_values(rc, "--shard") != [str(r["shard"])] or _flag_values(rc, "--attempt-root-local") != [ATTEMPT_ROOT_PLACEHOLDER]:
                    problems.append(f"{r['standup_id']}: rung consumer must carry --shard {r['shard']} (its slot; WS2-E c32d489c adopts a predecessor's "
                                    f"evidence by slot, the ceiling rung too) and --attempt-root-local {ATTEMPT_ROOT_PLACEHOLDER} (got "
                                    f"{_flag_value(rc, '--shard')!r} / {_flag_value(rc, '--attempt-root-local')!r})")
            elif r.get("rung_mode") == RUNG_MODE_GSM8K_SUBSET:
                # the executable GSM8K rung -- no consumer; an assembly block whose slots are EXACTLY the children; every child a
                # rung child (--item-subset naming its slot's subset, the cap setter at the rung cap, identity v2)
                asm = r.get("rung_assembly") or {}
                if r.get("rung_consumer") is not None or r["leg"] != "gsm8k" or asm.get("kind") != "gsm8k" or asm.get("helper") != GSM8K_ASSEMBLY_HELPER_REL:
                    problems.append(f"{r['standup_id']}: a GSM8K rung row carries no consumer and an assembly block ({GSM8K_ASSEMBLY_HELPER_REL})")
                slots = {(x.get("arm"), int(x.get("s", -1)), x.get("cell")) for x in (asm.get("slots") or [])}
                kids_ = {(k["arm"], int(k["s"]), k["cell"]) for k in r["children"]}
                if slots != kids_ or int(asm.get("consumes") or 0) != int(r["stage"]):
                    problems.append(f"{r['standup_id']}: the assembly slots {sorted(slots)[:3]}... are not the children {sorted(kids_)[:3]}... at cap {r['stage']}")
                for k in r["children"]:
                    want_subset = f"{ATTEMPT_ROOT_PLACEHOLDER}/{GSM8K_SUBSET_REL.format(cap=int(k['cap']), s=int(k['s']), cell=k['cell'])}"
                    if _flag_values(k["argv"], "--item-subset") != [want_subset] or k.get("subset_rel") is None:
                        problems.append(f"{r['standup_id']}: GSM8K rung child {k['arm']} s{k['s']} must carry --item-subset {want_subset} (got {_flag_value(k['argv'], '--item-subset')!r})")
            elif r.get("rung_mode") != RUNG_MODE_PETRI_RERUN or r.get("rung_consumer") is not None:
                problems.append(f"{r['standup_id']}: rung mode {r.get('rung_mode')!r} / consumer {r.get('rung_consumer')!r} inconsistent")
        if r["leg"] == "gsm8k":
            for k in r["children"]:
                a = k["argv"]
                cap_k = int(k["cap"])
                if _flag_value(a, "--tinycap-identity-version") != str(GSM8K_IDENTITY_VERSION):
                    problems.append(f"{r['standup_id']}: GSM8K child {k['arm']} s{k['s']} must run identity version {GSM8K_IDENTITY_VERSION} (the ladder classifier reads the hashed max_tokens_by_task)")
                if _flag_value(a, "--on-cap-by-task") != f"{GSM8K_INSPECT_TASK}={cap_k}" or _flag_value(a, "--max-tokens-by-task") != f"{GSM8K_INSPECT_TASK}={cap_k}":
                    problems.append(f"{r['standup_id']}: GSM8K child {k['arm']} s{k['s']} must state AND set the cap ({GSM8K_INSPECT_TASK}={cap_k}); got "
                                    f"{_flag_value(a, '--max-tokens-by-task')!r} / {_flag_value(a, '--on-cap-by-task')!r}")
                for gone in ("--gsm8k-arm", "--standup-endpoint", "--controls-cap"):
                    if gone in a:
                        problems.append(f"{r['standup_id']}: GSM8K child {k['arm']} s{k['s']} carries {gone}, which the driver's parser does not know (the pre-fix argv)")
                # 2126e918: `--bank-view s<s>` == the row's view, REQUIRED (the reader refuses `main`, omission drops the row identity)
                want_view = gsm8k_bank_view(r["engine"], k["s"])
                bv = _flag_values(a, "--bank-view")
                if bv != [want_view] or (r["engine"] != ARDITI and want_view != r["engine"]) or k.get("bank_view") != want_view:
                    problems.append(f"{r['standup_id']}: GSM8K child {k['arm']} s{k['s']} must carry --bank-view {want_view} "
                                    f"({'the source bank of the Arditi objects' if r['engine'] == ARDITI else 'the row'"'"'s view'}; got {bv}) -- {GSM8K_BANK_VIEW_RULE}")
                # the operational contract: the per-child ledger + attempt id + routing-table copy, the fitted acknowledgement
                if _flag_value(a, "--ledger-url") != LEDGER_URL_PLACEHOLDER or _flag_value(a, "--routing-table") != CHILD_ROUTING_TABLE_PLACEHOLDER \
                        or _flag_value(a, "--attempt-id") != GSM8K_ATTEMPT_ID_PLACEHOLDER or not isinstance(k.get("ledger"), Mapping):
                    problems.append(f"{r['standup_id']}: GSM8K child {k['arm']} s{k['s']} must route through its own usage ledger (--ledger-url {LEDGER_URL_PLACEHOLDER}, "
                                    f"--routing-table {CHILD_ROUTING_TABLE_PLACEHOLDER}, --attempt-id {GSM8K_ATTEMPT_ID_PLACEHOLDER}; WS0 g8k_ledger_up / st_driver_argv_head)")
                fitted_cell = k["cell"] != "none" and k["cell"] not in PROMPTED_CELLS
                has_oos, pin_v = "--allow-out-of-support-thinking" in a, _flag_value(a, "--expect-object-sha256")
                # a row recorded under the LENS-ERA tuple refuses BY NAME -- the native standup's served identity
                # (reduction_mode not_applicable) can never match that tuple's `stacked`, so every such child dies OOS_UNAUTHORISED
                lens_era = {GSM8K_LENS_ERA_OOS_TUPLE.format(model=MODEL_KEY, cap=c) for c in (cap_k, 8192, 16384)}
                if k.get("oos_tuple") in lens_era:
                    problems.append(f"{r['standup_id']}: GSM8K child {k['arm']} s{k['s']} records the LENS-ERA tuple {k['oos_tuple']!r} (reduction_mode "
                                    f"stacked), which the native standup's identity (reduction_mode not_applicable) can never match -- D101: the native "
                                    f"battery's fitted rows run under {GSM8K_NATIVE_OOS_TUPLE.format(model=MODEL_KEY, cap=cap_k)} (W19 F-AD)")
                if fitted_cell and not (has_oos and pin_v and len(pin_v) == 64 and k.get("oos_tuple") == gsm8k_oos_tuple(k["cell"], cap_k, engine=r["engine"])):
                    problems.append(f"{r['standup_id']}: FITTED GSM8K child {k['arm']} s{k['s']} must carry --allow-out-of-support-thinking under tuple "
                                    f"{gsm8k_oos_tuple(k['cell'], cap_k, engine=r['engine'])} with --expect-object-sha256 <pin> (a fitted lens route under the thinking-ON render is "
                                    "outside its fit support; the driver refuses before a store exists -- WS0 g8k_driver_argv)")
                if not fitted_cell and (has_oos or pin_v is not None):
                    problems.append(f"{r['standup_id']}: GSM8K child {k['arm']} s{k['s']} ({k['cell']}) carries a fitted acknowledgement / pin on a bare or prompted route")
                if r["engine"] != ARDITI and _flag_value(a, "--config") != GSM8K_OVERLAY_CONFIG_REL.format(model=MODEL_KEY, view=r["engine"], slug=gsm8k_cell_slug(k["cell"])):
                    problems.append(f"{r['standup_id']}: GSM8K child {k['arm']} s{k['s']} must name WS0's per-cell overlay config (G8K_OWN_OVERLAY), got {_flag_value(a, '--config')!r}")
                if r["kind"] == "base" and "--item-subset" in a:
                    problems.append(f"{r['standup_id']}: a BASE GSM8K child carries --item-subset (a rung mechanism)")
                if _flag_value(a, "--oos-prompt-budget") != f"{ATTEMPT_ROOT_PLACEHOLDER}/{GSM8K_BUDGET_REL.format(cap=cap_k, cell=k['cell'])}" or k.get("budget_rel") is None:
                    problems.append(f"{r['standup_id']}: GSM8K child {k['arm']} s{k['s']} names no prompt-budget artifact for (cap {cap_k}, {k['cell']})")
        if r["leg"] == "controls":
            # F-T: MMLU (a teacher-forced one-token readout, no cap ladder) rides the BASE row only; a rung carries the ladder controls with the
            # driver's rung flag; the base row never carries `--controls-cap` (the driver refuses `mmlu` beside it, then "both or neither")
            is_rung = r["kind"] == "rung"
            if is_rung and ("mmlu" in r["eval_legs"] or "--max-logprobs" in r["argv"]):
                problems.append(f"{r['standup_id']}: a controls RUNG serves no MMLU (eval_legs {r['eval_legs']}, --max-logprobs "
                                f"{_flag_value(r['argv'], '--max-logprobs')!r})")
            for k in r["children"]:
                a = k["argv"]
                got = [c for c in str(_flag_value(a, "--controls") or "").split(",") if c]
                if not got or any(c not in CONTROLS_ALL for c in got) or set(got) != set(k.get("task") or {}) or got != list(k.get("controls") or []):
                    problems.append(f"{r['standup_id']}: controls child {k['arm']} s{k['s']} --controls {got} is not a subset of {list(CONTROLS_ALL)} "
                                    f"agreeing with its task map {sorted(k.get('task') or {})} / controls {k.get('controls')}")
                fixed = [c for c in got if c in CONTROLS_FIXED_CAP]
                # F-AF: the driver writes `controls_events.jsonl` -- the app's controls tailer's ONLY stream -- under its opt-in flag alone
                rides = controls_ride_the_ladder(got)
                if rides and a.count(CONTROLS_EVENTS_FLAG) != 1:
                    problems.append(f"{r['standup_id']}: controls child {k['arm']} s{k['s']} --controls {got} carries no {CONTROLS_EVENTS_FLAG} "
                                    f"(found {a.count(CONTROLS_EVENTS_FLAG)}; F-AF, D102): the driver then writes no controls_events.jsonl, the app's "
                                    "controls tailer registers no stream and the row is refused LADDER_INVENTORY_INCOMPLETE over zero streams "
                                    "after drawing everything (W19: S2 controls-arditi, S3 controls, S5 controls)")
                if not rides and CONTROLS_EVENTS_FLAG in a:
                    problems.append(f"{r['standup_id']}: controls child {k['arm']} s{k['s']} --controls {got} carries {CONTROLS_EVENTS_FLAG} with no "
                                    f"ladder control ({list(CONTROLS_LADDER)}); the driver refuses the stream by name on a run with no sampled judged control")
                if is_rung:
                    if fixed or _flag_value(a, "--controls-cap") != str(int(r["stage"])) or "--top-logprobs" in a:
                        problems.append(f"{r['standup_id']}: controls RUNG child {k['arm']} s{k['s']} must carry the ladder controls alone with "
                                        f"--controls-cap {r['stage']} (got --controls {got}, --controls-cap {_flag_value(a, '--controls-cap')!r}, "
                                        f"--top-logprobs {_flag_value(a, '--top-logprobs')!r}); the driver refuses {fixed} on a rung")
                else:
                    if "--controls-cap" in a:
                        problems.append(f"{r['standup_id']}: BASE controls child {k['arm']} s{k['s']} carries --controls-cap (the driver's RUNG flag: "
                                        "it refuses mmlu beside it, then --escalate-from-root 'both or neither')")
                    if fixed and _flag_value(a, "--top-logprobs") != str(MMLU_TOP_LOGPROBS):
                        problems.append(f"{r['standup_id']}: BASE controls child {k['arm']} s{k['s']} serves MMLU without --top-logprobs {MMLU_TOP_LOGPROBS}")
                    if not r.get("dose_standup") and got != list(CONTROLS_ALL):
                        problems.append(f"{r['standup_id']}: BASE controls child {k['arm']} s{k['s']} --controls {got} != {list(CONTROLS_ALL)} "
                                        "(MMLU is measured once per (arm, s), on the base row)")
        if r["leg"] in ("controls", "gsm8k"):
            roots: Dict[str, set] = {}
            for k in r["children"]:
                want = (CONTROLS_OUT_REL.format(cap=int(k["cap"]), s=int(k["s"])) if r["leg"] == "controls"
                        else GSM8K_OUT_REL.format(cap=int(k["cap"]), s=int(k["s"]), cell=k["cell"]))
                got = _flag_value(k["argv"], "--out-root")
                if got != f"{ATTEMPT_ROOT_PLACEHOLDER}/{want}" or k.get("out_rel") != want:
                    problems.append(f"{r['standup_id']}: child {k['arm']} s{k['s']} --out-root {got!r} is not its seed-owned root {want} (review r2 P1-8)")
                roots.setdefault(str(got), set()).add(int(k["s"]))
            shared = {g: ss for g, ss in roots.items() if len(ss) > 1}
            if shared:
                problems.append(f"{r['standup_id']}: children of different seed bases share an out-root {shared} -- a store no tailer can attribute to one seed")
            if r["leg"] == "gsm8k" and len({_flag_value(k["argv"], "--out-root") for k in r["children"]}) != len(r["children"]):
                problems.append(f"{r['standup_id']}: two GSM8K children share an out root -- inspect_rung.cell_dir_of refuses a root with two cell dirs")
        for k in r["children"]:
            if r["leg"] in MULTITURN_LEGS:
                a = k["argv"]
                if (_flag_value(a, "--multiturn-target-decode") != "vendor" or _flag_value(a, "--multiturn-target-policy-ref") != MULTITURN_DECODE_POLICY_REF
                        or _flag_value(a, "--multiturn-target-seed-base") != str(k["s"]) or _flag_value(a, "--multiturn-target-seed") != "per_turn"):
                    problems.append(f"{r['standup_id']}: multi-turn child {k['arm']} s{k['s']} must carry --multiturn-target-decode vendor, "
                                    f"--multiturn-target-policy-ref {MULTITURN_DECODE_POLICY_REF} AND --multiturn-target-seed-base {k['s']} (WS1-A 5532d04a)")
            if r["leg"] == "misalign":
                if _flag_value(k["argv"], "--only-cells") != harness_id(k["cell"]):
                    problems.append(f"{r['standup_id']}: misalign child {k['arm']} s{k['s']} --only-cells must be the HARNESS id (cells.json's namespace)")
                if _flag_value(k["argv"], "--out") != f"{ATTEMPT_ROOT_PLACEHOLDER}/{MISALIGN_OUT_REL.format(s=k['s'], behavior=k['behavior'])}":
                    problems.append(f"{r['standup_id']}: misalign child {k['arm']} s{k['s']} {k['behavior']} does not own a per-(seed, behaviour) out root")
                cap_k = int(k.get("cap") or _flag_value(k["argv"], "--max-tokens") or 0)
                want_t, got_t = misalign_timeout_s(cap_k), _flag_value(k["argv"], "--timeout-s")
                try:
                    ok_t = got_t is not None and float(got_t) == float(want_t)
                except ValueError:
                    ok_t = False
                if not ok_t:
                    problems.append(f"{r['standup_id']}: misalign child {k['arm']} s{k['s']} {k['behavior']} at cap {cap_k} must carry --timeout-s {want_t} "
                                    f"(got {got_t!r}); the runner's default is {MISALIGN_RUNNER_DEFAULT_TIMEOUT_S:.0f} s, sized for a 16k cap -- a lost "
                                    "response would park the runner past the walltime (EM 2026-09-18)")
        # the runner's unit of collision is `<out>/<cell dir>/{responses.jsonl, meta.json}`: two children of ONE cell must never
        # share an out root (the Arditi engine's 3 seed-runs x 2 behaviours of `arditi_allpos`; a view's 2 behaviours per cell)
        store_dirs = [(_flag_value(k["argv"], "--out"), k["cell_harness"]) for k in r["children"] if r["leg"] == "misalign"]
        if len(set(store_dirs)) != len(store_dirs):
            problems.append(f"{r['standup_id']}: misalign children of one cell share an out root (resume rejects / metadata race)")
    return problems


def _table_problems(row: Mapping[str, Any], sg: Any) -> List[str]:
    """The REAL front's resolution of one row's window (the S0 gate's CPU half): `resolve_geometry(eval, L?, S?,
    strict_identity=True, model_key)` exactly as `_resolve_standup_geometry` calls it, then the kernel lowering's re-resolution."""
    g, sid = row["geometry"], row["standup_id"]
    out: List[str] = []
    L, S_id = int(g["L"]), int(g["identity_S"])
    try:
        if g["table"]:
            res = sg.resolve_geometry(g["eval"], strict_identity=True, model_key=MODEL_KEY)
            if res.window_source != "table" or int(res.max_model_len) != L:
                out.append(f"{sid}: table resolution gives window_source {res.window_source!r} L {res.max_model_len} != table L {L}")
            if int(res.max_num_seqs) != S_id:
                out.append(f"{sid}: the front's identity S {res.max_num_seqs} != the plan's {S_id}")
        else:
            res = sg.resolve_geometry(g["eval"], max_model_len=L, max_num_seqs=S_id, strict_identity=True, model_key=MODEL_KEY)
            if not res.is_identity:
                out.append(f"{sid}: ({L}, {S_id}) is not the identity pair (identity_S {res.identity_S})")
        if g["expected_adjustment"] is not None:
            low = sg.resolve_geometry(g["eval"], max_model_len=L, max_num_seqs=int(g["S"]), strict_identity=False, model_key=MODEL_KEY)
            if int(low.max_num_seqs) != int(g["S"]) or int(low.max_model_len) != L:
                out.append(f"{sid}: the kernel re-resolution does not yield S {g['S']} at L {L}")
    except KeyError as e:
        out.append(f"{sid}: eval {g['eval']!r} is not in the geometry table ({e}) — WS0-4b's common class not landed at this tree")
    except Exception as e:  # noqa: BLE001 -- every refusal named
        out.append(f"{sid}: the front refuses this geometry: {type(e).__name__}: {e}")
    return out


#: DRAFT == FINAL (rev 4): the ONLY admissible differences are the S4-MEASURED fields and what they
#: derive -- the multi-turn legs' served window class (their geometry, the identity pair on their argv, the attested L / S,
#: the capture surface, the children's `--target-max-model-len`), the container count per multi-turn (leg, engine) (their
#: shard axis: standup ids, keys, `--shard k --nshards n`, the number of rows), the pools' geometry / count / assignment (which
#: pool a shard pins), the ledger and the S5 roster's multi-turn uses -- plus the adapter hashes filled after S1 / S1b / S2 and
#: the provenance stamps. EVERYTHING ELSE is compared per LOGICAL ARM / RUN, multi-turn legs included: a cap, a seed base, a
#: decode policy ref, a task label, an arm set, a pin, a route, a readiness expectation that moved is a generator bug.
S4_MEASURED_LEGS = tuple(MULTITURN_LEGS)
S4_VOLATILE_TOP = ("materialisation", "generated_at", "launch_plan_sha", "s4_decisions", "pools", "capacity_ledger", "adapter_hash_set",
                   "multiturn_window_provenance", "s5_multiturn_probe",
                   # the launch order's ROW lists move with the S4-decided pool count / multi-turn shard allocation; the
                   # projection keeps its invariant, the STAGE ordering (`launch_order_stages`)
                   "launch_order")
#: the S4-measured GEOMETRY fields of a multi-turn row (its window class)
S4_GEOMETRY_ROW_KEYS = ("L", "S", "identity_S", "product", "expected_adjustment", "argv_factors")
S4_READINESS_ROW_KEYS = ("attested_max_model_len", "attested_max_num_seqs", "adjustment", "capture_surface_ge")
#: the S4-measured ALLOCATION fields of a multi-turn child (its goal sub-shard, its pool, its window)
S4_CHILD_FLAGS = ("--shard", "--nshards", "--target-max-model-len", "--attacker-endpoint")
S4_LEG_PUBLIC_KEYS = ("containers_per_view", "containers_arditi", "forwards_per_goal", "base", "rungs", "h200_h", "timeout_h")


def _strip_flags(argv: Sequence[str], flags: Sequence[str]) -> List[str]:
    out, skip = [], False
    for a in argv or []:
        if skip:
            skip = False
            continue
        if a in flags:
            skip = True
            continue
        out.append(a)
    return out


def _child_projection(k: Mapping[str, Any], *, measured_leg: bool) -> Dict[str, Any]:
    """ONE child's invariant scientific fields: arm, cell, seed, task, cap, eval leg, identity file, concurrency rule and its
    argv minus the S4-allocated flags (a measured leg) or minus the pool placeholder only (any other leg)."""
    argv = [a for a in (k.get("argv") or []) if not a.startswith("{pool_url")]
    if measured_leg:
        argv = _strip_flags(argv, S4_CHILD_FLAGS)
    return {"arm": k["arm"], "cell": k.get("cell"), "s": k["s"], "task": k.get("task"), "cap": k.get("cap"), "eval_leg": k["eval_leg"],
            "registry_identity_rel": k.get("registry_identity_rel"), "concurrency_rule": k.get("concurrency_rule"), "argv": argv,
            "env": k.get("env") or {}, "grading_host_drive": k.get("grading_host_drive")}


def _row_projection(r: Mapping[str, Any]) -> Dict[str, Any]:
    """ONE row's invariant scientific fields. A NON-measured leg keeps everything but the adapter hashes and the pool
    placeholder. A MEASURED (multi-turn) leg drops its shard axis and window class: the projection is per LOGICAL
    (leg, engine, stage) and the children per LOGICAL (arm, s) -- every goal sub-shard of one arm must project identically."""
    measured = r["leg"] in S4_MEASURED_LEGS
    adapters = {c: {k: v for k, v in a.items() if k != "intervention_hash"} for c, a in r["adapters"].items()}
    if (r.get("dose_standup") or r.get("s5_probe")) and measured:
        # a dose multi-turn standup -- or the smoke-only probe row -- has ONE shard (no allocation axis) but
        # shares the measured window class
        kids = [_child_projection(k, measured_leg=True) for k in r["children"]]
        return {"kind": r["kind"], "leg": r["leg"], "engine": r["engine"], "stage": r["stage"], "standup_id": r["standup_id"],
                **({"dose_standup": True} if r.get("dose_standup") else {"s5_probe": True}), "eval_legs": r["eval_legs"], "seed_bases": r["seed_bases"],
                "argv": _strip_flags(r["argv"], ("--max-model-len", "--max-num-seqs")),
                "geometry": {k: v for k, v in r["geometry"].items() if k not in S4_GEOMETRY_ROW_KEYS},
                "readiness": {k: v for k, v in r["readiness"].items() if k not in S4_READINESS_ROW_KEYS},
                "route_set": r["route_set"], "objects": r["objects"], "adapters": adapters, "arm_runs": r["arm_runs"], "children": kids}
    kids = [_child_projection(k, measured_leg=measured) for k in r["children"]]
    if measured:
        by_arm: Dict[str, Dict[str, Any]] = {}
        for k in kids:
            key = f"{k['eval_leg']}/{k['arm']}/s{k['s']}"
            if key in by_arm and by_arm[key] != k:
                by_arm[key] = {"CONFLICT": [by_arm[key], k]}
            else:
                by_arm.setdefault(key, k)
        kids_out: Any = by_arm
        argv = _strip_flags(r["argv"], ("--max-model-len", "--max-num-seqs"))
        geometry = {k: v for k, v in r["geometry"].items() if k not in S4_GEOMETRY_ROW_KEYS}
        readiness = {k: v for k, v in r["readiness"].items() if k not in S4_READINESS_ROW_KEYS}
        ident = {"kind": r["kind"], "leg": r["leg"], "engine": r["engine"], "stage": r["stage"]}
    else:
        kids_out = kids
        argv, geometry, readiness = list(r["argv"]), dict(r["geometry"]), dict(r["readiness"])
        ident = {"kind": r["kind"], "leg": r["leg"], "engine": r["engine"], "stage": r["stage"], "shard": r["shard"], "nshards": r["nshards"],
                 "standup_id": r["standup_id"], "control_key": r["control_key"], "accepted_key": r.get("accepted_key")}
    return {**ident, "eval_legs": r["eval_legs"], "seed_bases": r["seed_bases"], "argv": argv, "geometry": geometry, "readiness": readiness,
            "route_set": r["route_set"], "route_set_harness": r.get("route_set_harness"), "cells_served_expected": r.get("cells_served_expected"),
            "objects": r["objects"], "adapters": adapters, "arm_runs": r["arm_runs"], "producers": r["producers"], "ladder": r.get("ladder"),
            "rung_consumer": r.get("rung_consumer"), "rung_mode": r.get("rung_mode"), "children": kids_out,
            "modal": {k: v for k, v in r["modal"].items() if k != "max_containers"} if not measured else {k: v for k, v in r["modal"].items() if k not in ("max_containers",)},
            "expected_h200_h": r["expected_h200_h"]}


def scientific_projection(plan: Mapping[str, Any]) -> Dict[str, Any]:
    """The plan's INVARIANT scientific content, keyed per logical row / arm / run (the S4-measured geometry, allocation and
    pool fields projected away) -- what DRAFT and FINAL must agree on byte for byte."""
    p = json.loads(json.dumps(dict(plan), sort_keys=True, default=str))
    for k in S4_VOLATILE_TOP:
        p.pop(k, None)
    legs = {}
    for leg, spec in p["legs"].items():
        legs[leg] = {k: v for k, v in spec.items() if not (leg in S4_MEASURED_LEGS and k in S4_LEG_PUBLIC_KEYS)}
    p["legs"] = legs
    rows = {}
    # the smoke-only `s5_probe_rows` are single-shard MULTI-TURN rows whose window class is the S4 decision (the probe
    # measures at the decided window), so they project like a dose multi-turn standup -- per standup id, window class stripped -- and
    # never raw: a raw probe row made every FINAL differ from its draft on geometry / readiness / argv windows once the probe was registered
    for r in p["standups"] + p["rungs"] + list(p.get("s5_probe_rows") or []):
        proj = _row_projection(r)
        key = (f"{r['leg']}/{r['engine']}/{r['stage']}" if (r["leg"] in S4_MEASURED_LEGS and not r.get("dose_standup") and not r.get("s5_probe"))
               else r["standup_id"])
        if key in rows and rows[key] != proj:
            rows[key] = {"CONFLICT_ACROSS_SHARDS": [rows[key], proj]}
        else:
            rows.setdefault(key, proj)
    p.pop("standups"); p.pop("rungs"); p.pop("s5_probe_rows", None)
    p["rows"] = rows
    # the S5 roster with the measured legs' USES stripped (a window they share with another leg stays, keyed on its numbers)
    roster = []
    for r in p["s5_roster"]:
        used = [u for u in r["used_by"] if u.split(":")[0] not in S4_MEASURED_LEGS]
        if used:
            roster.append({"L": r["L"], "S": r["S"], "thinking": r["thinking"], "used_by": used})
    p["s5_roster"] = sorted(roster, key=lambda r: (r["L"], r["S"], r["thinking"]))
    p["queues"] = [q for q in p["queues"] if q["leg"] not in S4_MEASURED_LEGS]
    # the pools' INVARIANTS (their model / snapshot); count, geometry and assignment are S4's
    p["pool_invariants"] = sorted({json.dumps({"model": q["model"], "snapshot": q["snapshot"]}, sort_keys=True) for q in plan["pools"]})
    # the launch order's invariant: WHICH stages precede which (pools -> multi-turn targets -> the rest), never the row lists
    p["launch_order_stages"] = [{"stage": st["stage"], "owner": st["owner"], "requires": st["requires"]} for st in plan["launch_order"]]
    return p


def compare_draft_final(draft: Mapping[str, Any], final: Mapping[str, Any]) -> List[str]:
    """``draft == final`` on every invariant scientific field per logical arm / run (:func:`scientific_projection`); the
    ONLY admissible diffs are the S4-measured window class / allocation / pool decisions. Returns the paths that differ
    (empty = admitted); a differing cap, seed base, decode policy, arm set, task label or pin anywhere -- multi-turn legs
    included -- is a generator bug and blocks the launch."""
    a, b = scientific_projection(draft), scientific_projection(final)
    return _diff_paths(a, b, "")


def _diff_paths(a: Any, b: Any, path: str) -> List[str]:
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: {'missing in draft' if k not in a else 'missing in final'}")
            else:
                out += _diff_paths(a[k], b[k], f"{path}/{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} != {len(b)}"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += _diff_paths(x, y, f"{path}[{i}]")
        return out
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


#: the watcher's scopes: the control keys a leg must have ACCEPTED coverage of before it is DONE
WATCH_SCOPES: Tuple[str, ...] = ("base", "rungs", "all")


def required_accepted_keys(plan: Mapping[str, Any], leg: str, scope: str = "base") -> Dict[str, str]:
    """``{accepted_key: standup_id}`` of every DISTINCT row of ``leg`` in ``scope`` (`base` = the base shards, `rungs` = the
    declared rung functions, `all` = both) -- DERIVED from the plan, never counted from the calls ledger: four call records
    for one key are retries of ONE required row and never add coverage."""
    if scope not in WATCH_SCOPES:
        raise ValueError(f"scope {scope!r} is not one of {WATCH_SCOPES}")
    rows = []
    if scope in ("base", "all"):
        rows += [r for r in plan["standups"] if r["leg"] == leg]
    if scope in ("rungs", "all"):
        rows += [r for r in plan["rungs"] if r["leg"] == leg]
    out = {r["accepted_key"]: r["standup_id"] for r in rows}
    if len(out) != len(rows):
        raise ValueError(f"leg {leg}: accepted keys collide across its rows")
    return out


# --- SMOKE identities: S1-S5 run DRAFT rows under an ISOLATED wave ---------------------------------------------
SMOKE_RESULTS_SUFFIX = "_smoke"
SMOKE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")
#: TWO WAVE IDENTITIES. `execution_wave_uuid` = the wave a container RUNS in -- the smoke wave (`RS/smokes/<id>/wave.json`,
#: minted once per smoke id) for a smoke row, the production wave for production fan-out; it is what ATTEMPT.json / the stores / the
#: S5 references (`wave_uuid` = the wave of the standup that MEASURED) carry. `production_wave_uuid` = the plan's `wave_uuid`, the wave
#: the certification TARGETS; with `launch_plan_sha` it is the certification WRAPPER's binding (the S5 block, the launch certificate), never
#: a field a measurement record is re-stamped with. A smoke never runs under the production wave (`smoke_identity` refuses it), so an S5
#: record whose `wave_uuid` is the production wave measured nothing this lane stood up.
WAVE_IDENTITY_RULE = ("execution_wave_uuid = the wave the measuring standup ran in (a smoke's own wave; the production wave for fan-out) -- on "
                      "attempts, stores and WS3-G's S5 references; production_wave_uuid + launch_plan_sha = the certification wrapper's target")
SMOKES_DIR = "smokes"
SMOKE_WAVE_FILE = "wave.json"


def smoke_wave_path(rs: Path, smoke_id: str) -> Path:
    """`RS/smokes/<smoke_id>/wave.json` -- the smoke's EXECUTION wave, minted once by the app at `--smoke` (`seedrep_served.smoke_wave_for`)."""
    if not SMOKE_ID_RE.match(smoke_id or ""):
        raise ValueError(f"smoke id {smoke_id!r} must match {SMOKE_ID_RE.pattern}")
    return Path(rs) / SMOKES_DIR / smoke_id / SMOKE_WAVE_FILE


def smoke_execution_wave(rs: Path, smoke_id: str) -> Optional[str]:
    """The smoke's execution wave from its minted file, None when the smoke never ran (no file); a malformed file raises."""
    p = smoke_wave_path(rs, smoke_id)
    if not p.is_file():
        return None
    doc = json.loads(p.read_text())
    if not isinstance(doc, Mapping) or doc.get("smoke_id") not in (None, smoke_id):
        raise ValueError(f"{p}: not the wave record of smoke {smoke_id!r}")
    return check_wave_uuid(str(doc["wave_uuid"]))


def smoke_identity(plan: Mapping[str, Any], smoke_id: str, smoke_wave_uuid: str) -> Dict[str, Any]:
    """The smoke WAVE's identity: its own dashed uuid4 (minted once per smoke id, persisted beside the plan) -> its own
    Dicts (`seedrep-<W7>-<smoke wave8>-*`), its own ladder root and a results root beside production's
    (`<results_root>_smoke/<smoke_id>`), so a smoke attempt can never take a production row's acceptance, heartbeat, queue
    or attempt root. Production fan-out (`--go` / `--legs`) stays FINAL-only."""
    if not SMOKE_ID_RE.match(smoke_id or ""):
        raise ValueError(f"smoke id {smoke_id!r} must match {SMOKE_ID_RE.pattern}")
    check_wave_uuid(smoke_wave_uuid)
    if smoke_wave_uuid == plan["wave_uuid"]:
        raise ValueError("a smoke wave must not reuse the production wave uuid")
    W7 = plan["W7"]
    return {"smoke_id": smoke_id, "wave_uuid": smoke_wave_uuid, "W": plan["W"], "W7": W7, "dicts": {p: dict_name(W7, smoke_wave_uuid, p) for p in DICT_PURPOSES},
            "results_root": f"{plan['results_root']}{SMOKE_RESULTS_SUFFIX}/{smoke_id}",
            "ladder_root": f"{LANE}/{W7}/{smoke_wave_uuid}/ladder"}


# ---- a smoke's adapter-reuse expectation follows its DECLARED §8 role, never production's -------------------------------------------
#: Gate 10 is PRODUCTION's side of the adapter contract: every adapter a production row serves is a RE-USE, because the smokes banked
#: every hash first (`adapter_expectations`: `expected_reused` True per cell). The smokes are the OTHER side -- the S1 / S1b / S2 engines
#: are the FIRST exporters of the adapters they serve -- and the consumer (`seedrep_served.readiness_problems`, gate 10) refuses
#: `reused=False` wherever the row expects True, so a smoke row that kept the production expectation refused its OWN first export
#: (S1b's five s2 / s3 cells, `EXPORTED (reused=False)` x5; S1 escaped only because an earlier wave's failed S1 had already banked
#: the s1 hashes -- the bank persists across waves). `expected_reused=False` is "no re-use asserted": the consumer skips the cell, so an
#: export AND a bank hit from an earlier wave pass alike; `True` asserts the dependency §8 declares (a prior smoke banked the hash).
#: Per role (order: S1 -> {S1b, S2, S3, S5} in parallel -> S4):
#:   role   first exports (expected_reused False)                          re-uses (expected_reused True)
#:   S1     every adapter it serves, `none__lora0` included -- the wave's   --
#:          first engine (s1 view; §8 S1 "adapters exported")
#:   S1b    its own fitted cells (the s2 / s3 view engines)                 `none__lora0` -- S1 banked the zero adapter (§8 S1b literal:
#:                                                                          reused false for its own five cells, true for none__lora0)
#:   S2     `arditi_allpos` -- the bank holds 15 fitted + zero + Arditi     `none__lora0` -- S1 banked it (every lora_native engine serves it)
#:          only after S1 + S1b + S2 (§8 S1b's pass rule)
#:   S3     --                                                              every adapter -- the S1 engine re-stood-up at each leg's window
#:                                                                          class (§8 S3); S1 banked the s1 view's hashes
#:   S4     --                                                              every adapter -- runs after the parallel block (§8 order)
#:   S5     every adapter it serves -- decode ceilings on every window      --
#:          class + the probe rows' peer cells, in parallel with
#:          S1b / S2: S5 measures windows, it is not the adapter gate
#: A smoke id outside the table declares no role and is REFUSED (its expectation cannot be derived); production rows are untouched.
#: The plan's own smoke-only S5 probe rows are first exports by construction and carry False at the PLAN level as well
#: (`s5_probe_rows` -> `adapter_expectations(..., expected_reused=False)`) -- `smoke_adapter_problems` holds them to that.
SMOKE_FIRST_EXPORTS_ALL, SMOKE_FIRST_EXPORTS_FITTED, SMOKE_FIRST_EXPORTS_NONE = "all", "fitted", "none"
SMOKE_ADAPTER_ROLES: Dict[str, Dict[str, str]] = {
    "S1": {"first_exports": SMOKE_FIRST_EXPORTS_ALL,
           "reason": "the wave's first engine (s1 view): every adapter it serves, none__lora0 included, is a first export (§8 S1: adapters exported)"},
    "S1B": {"first_exports": SMOKE_FIRST_EXPORTS_FITTED,
            "reason": "the s2 / s3 view engines first-export their own fitted cells; none__lora0 was banked by S1 (§8 S1b: adapters_reused false "
                      "for its own five cells, true for none__lora0)"},
    "S2": {"first_exports": SMOKE_FIRST_EXPORTS_FITTED,
           "reason": "the Arditi engine first-exports arditi_allpos (§8 S1b: the bank holds 15 fitted + zero + Arditi only after S1 + S1b + S2); "
                     "none__lora0 was banked by S1"},
    "S3": {"first_exports": SMOKE_FIRST_EXPORTS_NONE,
           "reason": "the drivers run against the S1 engine re-stood-up at each leg's window class (§8 S3): S1 banked every adapter of the s1 view"},
    "S4": {"first_exports": SMOKE_FIRST_EXPORTS_NONE,
           "reason": "the multi-turn window decision runs after the S1 -> {S1b, S2, S3, S5} block (§8 order): every adapter is banked"},
    "S5": {"first_exports": SMOKE_FIRST_EXPORTS_ALL,
           "reason": "decode ceilings on every window class + the D45 probe rows, in parallel with S1b / S2 (§8 order): S5 measures windows and "
                     "asserts no re-use (the probe cells are its own first export)"},
}


def smoke_adapter_role(smoke_id: str) -> Dict[str, str]:
    """The §8 adapter role a smoke id declares (case-insensitive: `S1b` == `S1B`) -> ``{role, first_exports, reason}``; an id outside
    :data:`SMOKE_ADAPTER_ROLES` refuses -- its expectation cannot be derived, and production's "every adapter re-used" would
    refuse a first exporter's own export."""
    key = str(smoke_id or "").upper()
    role = SMOKE_ADAPTER_ROLES.get(key)
    if role is None:
        raise ValueError(f"smoke id {smoke_id!r} declares no §8 adapter role (one of {sorted(SMOKE_ADAPTER_ROLES)}): a smoke row's "
                         f"expected_reused follows its role, never the production expectation (W15d, launch F-L(d))")
    return {"role": key, **role}


def smoke_expected_reused(smoke_id: str, cell: str) -> bool:
    """Whether smoke ``smoke_id`` must find adapter ``cell`` RE-USED (True) or exports it first / asserts nothing (False), per its §8 role."""
    kind = smoke_adapter_role(smoke_id)["first_exports"]
    if kind == SMOKE_FIRST_EXPORTS_ALL:
        return False
    if kind == SMOKE_FIRST_EXPORTS_FITTED:
        return cell == NONE_LORA0
    return True


def smoke_adapters(adapters: Mapping[str, Mapping[str, Any]], smoke_id: str) -> Dict[str, Dict[str, Any]]:
    """The production row's `adapters` block with `expected_reused` re-derived from the smoke's role; every other fact (object pin,
    staged id, intervention hash) is the production row's."""
    return {cell: {**dict(a), "expected_reused": smoke_expected_reused(smoke_id, cell)} for cell, a in adapters.items()}


def smoke_adapter_problems(row: Mapping[str, Any]) -> List[str]:
    """Why ``row``'s adapter expectations do not follow its side of gate 10 (``[]`` = they do): a SMOKE row (one carrying `smoke`) must
    expect exactly what its §8 role derives -- a first-export smoke row expecting True is refused; a PRODUCTION row expects every
    adapter re-used."""
    sid, smoke, out = row.get("standup_id"), row.get("smoke"), []
    adapters = row.get("adapters") or {}
    if smoke is None:
        # a smoke-only row (the S5 probe) is a FIRST EXPORT by construction and asserts no re-use; a production row expects
        # every adapter re-used
        want = not bool(row.get("smoke_only"))
        for cell, a in adapters.items():
            if a.get("expected_reused") is not want:
                out.append((f"{sid}: a production row expects adapter {cell} RE-USED (gate 10: the smokes banked every hash first)" if want else
                            f"{sid}: a smoke-only row (the S5 probe) is a FIRST EXPORT of adapter {cell} by construction and asserts no re-use (B3)")
                           + f"; found expected_reused={a.get('expected_reused')!r}")
        return out
    try:
        role = smoke_adapter_role(smoke.get("smoke_id"))
    except ValueError as e:
        return [f"{sid}: {e}"]
    for cell, a in adapters.items():
        want, got = smoke_expected_reused(role["role"], cell), a.get("expected_reused")
        if got is not want:
            tail = (" -- the PRODUCTION expectation, which refuses the smoke's own first export at gate 10 (W14 F-L(d))"
                    if (want is False and got is True) else "")
            out.append(f"{sid}: smoke {role['role']} {'RE-USES' if want else 'FIRST-EXPORTS'} adapter {cell} ({role['reason']}); "
                       f"found expected_reused={got!r}{tail}")
    return out


# ---- A smoke row's ITEM budget -- the second (and last) thing a smoke changes about a production row --------------------
#: `smoke_row` bounds children and the container timeout, but every child kept its PRODUCTION item budget: a BoN child runs
#: `goals=100 x n_max=75` (7,500 draws per arm, ~3-7 h at concurrency 5), a misalign child 490 items per
#: behaviour (~2.3 h). A stage PASS needs EVERY row in it, so those two smokes sat on the critical path to `certify` / `--go`. A smoke proves
#: the CHAIN (readiness, labels, audits, ladder inventory, PASS derivation), not the production budget -- the FULL-shape rows and
#: the production run prove the budget. So the other smokes' children get a smoke-only item budget through the drivers' own flags:
#: BoN `--limit-goals SMOKE_BON_GOALS` (`bon_served_run.py`: head-cap the goal set after selection; 10 x 75 = 750 draws per arm,
#: ~20-40 min at the measured ~43 draws/min), misalign `--limit-items SMOKE_MISALIGN_ITEMS` (`misalign.runner`: the first N items per
#: behaviour in corpus order; the planned set is what the store's own meta.json declares). Recorded in `row["smoke"]["budget"]`
#: (additive; `modal_evidence.json` carries the row's smoke block whole, so the certificate sees it). A bounded store still reads
#: COMPLETE on its own planned set: BoN's `units.n_planned` is goals x n_max over the CAPPED goal list (`_bon_generate_served_async`
#: receives the capped list), misalign's `item_ids x n_samples` is the limited set -- `audit_common.read_bon_store` /
#: `read_misalign_store` unchanged. The FULL-shape rows (`--smoke-max-children 64`, the utilisation references) are
#: ALSO smoke rows and keep the FULL budget: their bars measure the production shape. The other legs (static, controls, GSM8K,
#: multi-turn, Petri) are bounded by children / goal shards already and carry no budget flag. `validate_plan` refuses a budget flag
#: on a PRODUCTION row (a limited store is never a production result) and a non-FULL-shape smoke BoN / misalign child WITHOUT it.
SMOKE_BON_GOALS = 10
SMOKE_MISALIGN_ITEMS = 10
SMOKE_BON_LIMIT_FLAG = "--limit-goals"                  # bon_served_run.build_parser: "SMOKE ONLY: head-cap the goal set after selection"
SMOKE_MISALIGN_LIMIT_FLAG = "--limit-items"             # sisteer.capeval.misalign.runner._build_parser: "SMOKE ONLY"
SMOKE_BUDGET_FLAGS: Tuple[str, ...] = (SMOKE_BON_LIMIT_FLAG, SMOKE_MISALIGN_LIMIT_FLAG)
SMOKE_BUDGET_FULL_SHAPE_IDS: Tuple[str, ...] = (S5_WINDOW_SMOKE_ID,)     # the references measure the FULL production shape
SMOKE_BUDGET_KIND_ITEM, SMOKE_BUDGET_KIND_FULL = "item_budget", "full_shape"
#: the ROW leg whose children carry a budget flag -> (flag, the `smoke.budget` key holding its value); a child's `eval_leg` spells
#: `bon` / `misalign_<behaviour>`, the row's leg `bon` / `misalign`
SMOKE_BUDGET_FLAG_OF_LEG: Dict[str, Tuple[str, str]] = {"bon": (SMOKE_BON_LIMIT_FLAG, "bon_limit_goals"),
                                                         "misalign": (SMOKE_MISALIGN_LIMIT_FLAG, "misalign_limit_items")}


def smoke_budget(smoke_id: str) -> Dict[str, Any]:
    """The item budget smoke ``smoke_id`` runs under: S1-S4 -> ``item_budget`` (BoN goals, misalign items per behaviour);
    S5 -> ``full_shape`` (the references measure the production shape). An id outside the §8 role table refuses, the
    way :func:`smoke_adapter_role` does (the same dispatch: `S1b` == `S1B`)."""
    key = smoke_adapter_role(smoke_id)["role"]
    if key in SMOKE_BUDGET_FULL_SHAPE_IDS:
        return {"kind": SMOKE_BUDGET_KIND_FULL, "bon_limit_goals": None, "misalign_limit_items": None,
                "rule": f"full_shape ({key} reference, D103 / D104): the utilisation references measure the FULL production shape -- no item budget"}
    return {"kind": SMOKE_BUDGET_KIND_ITEM, "bon_limit_goals": SMOKE_BON_GOALS, "misalign_limit_items": SMOKE_MISALIGN_ITEMS,
            "rule": (f"D107: BoN children {SMOKE_BON_LIMIT_FLAG} {SMOKE_BON_GOALS} (goals x n_max draws per arm), misalign children "
                     f"{SMOKE_MISALIGN_LIMIT_FLAG} {SMOKE_MISALIGN_ITEMS} (the first N items per behaviour in corpus order); the other legs are "
                     "bounded by children / goal shards")}


def determinism_item_budget(row: Mapping[str, Any]) -> Optional[int]:
    """The ITEM budget ``row``'s stores file under, or None when the row is not budget-limited: a SMOKE row (`smoke.budget` of kind
    `item_budget`) whose leg carries a budget flag (`SMOKE_BUDGET_FLAG_OF_LEG`: BoN -> `bon_limit_goals` goals, misalign -> `misalign_limit_items`
    items PER BEHAVIOUR, i.e. per misalign table leg). A production row (no `smoke` block), an S5 `full_shape` smoke and every other leg -> None."""
    smoke = row.get("smoke")
    budget = smoke.get("budget") if isinstance(smoke, Mapping) else None
    if not isinstance(budget, Mapping) or budget.get("kind") != SMOKE_BUDGET_KIND_ITEM:
        return None
    spec = SMOKE_BUDGET_FLAG_OF_LEG.get(str(row.get("leg")))
    if spec is None or budget.get(spec[1]) is None:
        return None
    return int(budget[spec[1]])


def determinism_spot_check_n(row: Mapping[str, Any]) -> Tuple[int, Optional[int]]:
    """``(n, item_budget)`` for ``row``'s determinism spot-check: the items the check selects per arm and the row's item budget
    when it is budget-limited. A budget-limited row selects `min(DETERMINISM_N_PLANNED, budget)` -- its store files `budget` items by
    construction, so the plan's 20 could never be met; every other row selects the plan's 20 and `item_budget` is None. Derived from the ROW
    (its `smoke.budget`), never from a store's size: the served app passes the pair as `--n` / `--item-budget`."""
    budget = determinism_item_budget(row)
    return (DETERMINISM_N_PLANNED if budget is None else min(DETERMINISM_N_PLANNED, budget)), budget


def _argv_flag_count(argv: Sequence[str], flag: str) -> int:
    return sum(1 for a in argv if a == flag or str(a).startswith(flag + "="))


def smoke_budget_children(children: Sequence[Mapping[str, Any]], *, leg: str, budget: Mapping[str, Any], standup_id: str) -> List[Dict[str, Any]]:
    """The smoke row's children with ``budget`` applied: under ``item_budget`` every child of a BoN / misalign row gains its driver's
    flag and value (appended to the argv); a child ALREADY carrying a budget flag refuses -- the production row it came from must
    never carry one (`validate_plan` holds it to that). ``full_shape`` = the children as they are."""
    out = []
    for k in children:
        k = copy.deepcopy(dict(k))
        argv = k.get("argv")
        if argv is not None:
            carried = [f for f in SMOKE_BUDGET_FLAGS if _argv_flag_count(argv, f)]
            if carried:
                raise ValueError(f"{standup_id}: child {k.get('arm')} s{k.get('s')} already carries the smoke-only budget flag(s) {carried} -- a "
                                 "production row never does (D107: a budget-limited store is never a wave result)")
            spec = SMOKE_BUDGET_FLAG_OF_LEG.get(leg)
            if budget["kind"] == SMOKE_BUDGET_KIND_ITEM and spec is not None:
                flag, key = spec
                k["argv"] = list(argv) + [flag, str(int(budget[key]))]
        out.append(k)
    return out


def smoke_budget_problems(row: Mapping[str, Any]) -> List[str]:
    """Why ``row``'s children do not carry the item budget its side derives (``[]`` = they do). A PRODUCTION row (no `smoke`
    block; the smoke-only S5 probe rows included) carries NO budget flag on any child. A SMOKE row records `smoke.budget` exactly as
    its id derives; under `item_budget` (S1-S4) every child of a BoN / misalign row carries its flag with that value exactly once,
    under `full_shape` (S5) no child carries a budget flag; a child of any other leg never carries one."""
    sid, smoke, out = row.get("standup_id"), row.get("smoke"), []
    children = [k for k in (row.get("children") or []) if k.get("argv") is not None]
    if smoke is None:
        for k in children:
            for flag in SMOKE_BUDGET_FLAGS:
                if _argv_flag_count(k["argv"], flag):
                    out.append(f"{sid}: a production row's child {k.get('arm')} s{k.get('s')} carries the smoke-only budget flag {flag} "
                               f"{_flag_value(k['argv'], flag)!r} -- D107: a budget-limited store is never a wave result; only `smoke_row` applies "
                               "a budget, to S1-S4 smoke rows")
        return out
    try:
        want = smoke_budget(smoke.get("smoke_id"))
    except ValueError as e:
        return [f"{sid}: {e}"]
    role = smoke_adapter_role(smoke.get("smoke_id"))["role"]
    if smoke.get("budget") != want:
        out.append(f"{sid}: smoke {role} records budget {smoke.get('budget')!r}; its smoke id derives {want!r} (D107)")
    spec = SMOKE_BUDGET_FLAG_OF_LEG.get(str(row.get("leg")))
    for k in children:
        who = f"{sid}: smoke {role} child {k.get('arm')} s{k.get('s')}"
        for flag in SMOKE_BUDGET_FLAGS:
            n = _argv_flag_count(k["argv"], flag)
            expected = want["kind"] == SMOKE_BUDGET_KIND_ITEM and spec is not None and spec[0] == flag
            if expected:
                if n != 1 or _flag_value(k["argv"], flag) != str(want[spec[1]]):
                    out.append(f"{who} must carry {flag} {want[spec[1]]} exactly once (D107: the S1-S4 smoke item budget -- without it the child runs "
                               f"the PRODUCTION budget, the W19 S3 bon row's ~7 h); found {n} occurrence(s), value {_flag_value(k['argv'], flag)!r}")
            elif n:
                why = (f"an {role} reference row keeps the FULL production shape (D103 / D104)" if want["kind"] == SMOKE_BUDGET_KIND_FULL
                       else f"the {row.get('leg')} leg is bounded by children / goal shards, never by {flag}")
                out.append(f"{who} carries {flag} {_flag_value(k['argv'], flag)!r}: {why}")
    return out


def smoke_standup_id(smoke_id: str, standup_id: str) -> str:
    """THE spelling of a smoke row's standup id -- `smoke-<Sid>__<standup_id>` (audit_common.SMOKE_STANDUP_PREFIX / `smoke_standup_id_of`): what `smoke_row`
    stamps on the row the container serves, and so what its `exp/serve/cells.json` records as the served row (fc11j: a prior smoke root's served row is
    THIS spelling, never the plan row's)."""
    return f"smoke-{smoke_id}__{standup_id}"


def smoke_row(row: Mapping[str, Any], identity: Mapping[str, Any], *, max_children: int, max_timeout_s: int) -> Dict[str, Any]:
    """A DRAFT row re-keyed into the smoke wave and BOUNDED: the first ``max_children`` children (a smoke draws 5 items, not
    a leg), a container timeout no longer than ``max_timeout_s``, a standup id prefixed `smoke-<id>__`, the six-axis keys under
    the smoke wave uuid and the rung consumer's `--wave-uuid` rewritten. Everything scientific (argv, geometry, route set,
    pins, readiness, children argv) is the production row's -- a smoke tests the production shape -- with TWO exceptions:
    `adapters[*].expected_reused` follows the smoke's §8 role (:func:`smoke_adapters`), because a first exporter that keeps
    production's "every adapter re-used" refuses its own export at the adapter-reuse gate; and the BoN / misalign children
    of a non-reference smoke carry the smoke-only ITEM budget (:func:`smoke_budget` -- `--limit-goals` / `--limit-items`,
    recorded in `smoke.budget`), because at the production budget those two smokes ran 3-7 h on the critical path; the
    reference row keeps the FULL shape by name."""
    r = copy.deepcopy(dict(row))
    W7, wu = identity["W7"], identity["wave_uuid"]
    role = smoke_adapter_role(identity["smoke_id"])
    budget = smoke_budget(identity["smoke_id"])
    r["standup_id"] = smoke_standup_id(identity["smoke_id"], r["standup_id"])
    r["adapters"] = smoke_adapters(r.get("adapters") or {}, role["role"])
    r["control_key"] = control_key(W7, wu, r["leg"], r["engine"], r["stage"], r["shard"])
    r["accepted_key"] = accepted_key(W7, wu, r["leg"], r["engine"], r["stage"], r["shard"])
    r["children"] = smoke_budget_children(r["children"][: max(1, int(max_children))], leg=r["leg"], budget=budget, standup_id=r["standup_id"])
    r["modal"] = {**r["modal"], "timeout_s": min(int(r["modal"]["timeout_s"]), int(max_timeout_s))}
    if r.get("rung_consumer"):
        lad = r["ladder"]
        r["rung_consumer"] = rung_consumer_argv(leg=r["leg"], engine=r["engine"], stage=int(r["stage"]), shard=r["shard"], feeder_shards=lad["feeder_shards"],
                                                rung_shards=lad["shards"], next_cap=lad.get("cap_next"), W=identity["W"], W7=W7, wave_uuid=wu,
                                                timeout_s=r["modal"]["timeout_s"], width=int(r["geometry"]["S"]))
    r["smoke"] = {"smoke_id": identity["smoke_id"], "wave_uuid": wu, "max_children": int(max_children),
                  "production_standup_id": row["standup_id"], "production_wave_uuid": row["control_key"].split("/")[1],
                  "adapter_role": role, "budget": budget}
    return r


def plan_sha(plan: Mapping[str, Any]) -> str:
    """sha16 over the plan minus its own stamp — PROVENANCE only (the wave's identity is wave_uuid, §4.6 rev 3)."""
    p = {k: v for k, v in plan.items() if k not in ("launch_plan_sha", "generated_at")}
    return hashlib.sha256(json.dumps(p, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# CLI                                                                           #
# --------------------------------------------------------------------------- #


# ---- (f) the roster's BoN CLIMB decode under BOTH spellings (commit bebe99a0) ---------------------------------------------------
#: Commit bebe99a0 renamed the roster key `bon_climb_temperature: 1.0` -> `bon_climb_decode: as_drawn` and
#: DROPPED `bon_served.HARNESS_DEFAULT_SAMPLER` / `HARNESS_DEFAULT_TEMPERATURE` for `bon_policy(model, label)`. A tree
#: may carry either roster spelling (a merge of main brings the new one), so the generator reads BOTH and records
#: which one it read; it imports neither removed name (pinned by test). The BoN children's decode is the vendor row
#: (`--bon-decode-preset vendor`), untouched.
BON_CLIMB_DECODE_KEY = "bon_climb_decode"                 # main bebe99a0: `as_drawn` (the climb re-uses the drawn decode)
BON_CLIMB_TEMPERATURE_KEY = "bon_climb_temperature"       # the pre-#216 spelling: a wave-wide climb temperature (1.0)
BON_SERVED_REMOVED_NAMES: Tuple[str, ...] = ("HARNESS_DEFAULT_SAMPLER", "HARNESS_DEFAULT_TEMPERATURE")


def roster_bon_climb_decode(roster: Mapping[str, Any]) -> Dict[str, Any]:
    """The roster's BoN climb decode, read under BOTH key spellings: ``{key, value, spelling}``. Exactly one spelling must be
    present -- neither is a roster without a climb decode (refuse), both is a roster carrying two rulings (refuse)."""
    new, old = roster.get(BON_CLIMB_DECODE_KEY), roster.get(BON_CLIMB_TEMPERATURE_KEY)
    if new is None and old is None:
        raise ValueError(f"roster declares neither {BON_CLIMB_DECODE_KEY} (main bebe99a0) nor {BON_CLIMB_TEMPERATURE_KEY} (pre-#216)")
    if new is not None and old is not None:
        raise ValueError(f"roster declares BOTH {BON_CLIMB_DECODE_KEY}={new!r} and {BON_CLIMB_TEMPERATURE_KEY}={old!r} -- one ruling per roster")
    if new is not None:
        return {"key": BON_CLIMB_DECODE_KEY, "value": str(new), "spelling": "main bebe99a0 (#216, D53): the climb re-uses the drawn decode"}
    return {"key": BON_CLIMB_TEMPERATURE_KEY, "value": float(old), "spelling": "pre-#216: a wave-wide climb temperature (superseded by bon_climb_decode: as_drawn)"}


def load_inputs(repo: Path, peer_roster: Optional[Path] = None, peer_pins: Optional[Path] = None,
                config_roots: Sequence[Path] = ()) -> Dict[str, Any]:
    import yaml
    if (peer_roster is None) != (peer_pins is None):
        raise ValueError("--peer-roster and --peer-pins go together (D68: the peer's roster AND its pin file, both by path)")
    return {"config_private_runs": read_config_private_runs([Path(repo), *config_roots], plan_config_rels()),
            "roster": yaml.safe_load((repo / ROSTER_REL).read_text(encoding="utf-8")),
            "pins": json.loads((repo / PINS_REL).read_text(encoding="utf-8")),
            "peer_roster": (yaml.safe_load(Path(peer_roster).read_text(encoding="utf-8")) if peer_roster else None),
            "peer_pins": (json.loads(Path(peer_pins).read_text(encoding="utf-8")) if peer_pins else None),
            "arditi_pins": json.loads((repo / ARDITI_PINS_REL).read_text(encoding="utf-8")),
            "arms": yaml.safe_load((repo / ARMS_REL).read_text(encoding="utf-8")),
            "reqs_sha256": sha256_file(repo / VLLM_REQS_REL)}


S5_COVERAGE_SIDECAR_NAME = "s5_coverage.json"
S5_COVERAGE_SIDECAR_SCHEMA = "seedrep_s5_coverage/1"


def main(argv=None, *, read_text: Optional[Callable[[str], Optional[str]]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", type=Path, required=True, help="the tree at W (roster, pins, arms, vllm-requirements)")
    ap.add_argument("--w", required=True, help="the wave sha W (40 hex)")
    ap.add_argument("--wave-json", type=Path, required=True, help="RS/wave.json ({wave_uuid, minted_at}); created with --mint-wave")
    ap.add_argument("--mint-wave", action="store_true", help="mint a fresh uuid4 into --wave-json when it does not exist")
    ap.add_argument("--materialisation", choices=("draft", "final"), required=True)
    ap.add_argument("--decisions", type=Path, default=None, help="decisions.json of the S4-decided fields (final only)")
    ap.add_argument("--peer-roster", type=Path, default=None,
                    help=f"D68: the PEER's roster carrying its seed-1 dose views (Strength's {PEER_ROSTER_REL}); with --peer-pins; absent = pending_peer_spec")
    ap.add_argument("--peer-pins", type=Path, default=None, help=f"D68: the PEER's pin file for those objects (Strength's {PEER_PINS_REL})")
    ap.add_argument("--config-root", type=Path, action="append", default=[],
                    help="an extra tree to read config private_runs from (a lane worktree, for a DRAFT on one lane's tree); the tree at W is read first")
    ap.add_argument("--register-peer-dose-standups", action="store_true",
                    help="DORMANT (D71: Strength's sweep runs on cluster A): register the three-per-view dose standups from the peer files; default off")
    ap.add_argument("--adapter-hashes", type=Path, default=None,
                    help="JSON {staged_id | '<staged_id>:<sha>' | 'none__lora0': intervention_hash} from the smokes' cells_served.json")
    ap.add_argument("--out", type=Path, required=True, help="RS/launch_plan_draft.json | RS/launch_plan.json")
    ap.add_argument("--ledger", type=Path, required=True, help="RS/capacity_ledger.json")
    ap.add_argument("--compare-draft", type=Path, default=None, help="(final) the draft to hold this plan to (draft == final rule)")
    ap.add_argument("--validate-table", action="store_true", help="resolve every window through the tree's geometry table")
    ap.add_argument("--s5-references", type=Path, default=None,
                    help="RS/smokes.json: gate 8 planned - references = empty; S5.multiturn[] = the D50 window measurement records "
                         "(REQUIRED for a final plan: the multi-turn windows must be justified by complete, zero-clamped measurements)")
    ap.add_argument("--window-justification-out", type=Path, default=None,
                    help="(final) RS/window_justification.json -- the D50 records under the window they justify (default: beside --out)")
    ap.add_argument("--launch-set", default=None,
                    help="fc10b (D145 / D146): the standup ids this W launches -- a comma-separated list or @<file> (one id per line / a JSON list), the spelling "
                         "certify and the S0 gates take. Gate 8 and the D50 window gate are then scoped to the launched rows exactly as certify / G8 scope them "
                         "(coverage recorded beside --out as s5_coverage.json); absent = the whole roster, today's gates unchanged")
    args = ap.parse_args(argv)
    launch_set = parse_launch_set(args.launch_set)

    if not args.wave_json.is_file() and not args.mint_wave:
        raise SystemExit(f"no {args.wave_json}; pass --mint-wave once at launch (the wave's identity is minted, never derived)")
    wave = mint_wave(args.wave_json, W=args.w)                     # DASHED uuid4: the spelling every peer validates; re-read when present
    s4 = json.loads(args.decisions.read_text()) if args.decisions else None
    if args.materialisation == "final" and s4 is None:
        raise SystemExit("a FINAL plan needs --decisions (the S4-decided fields)")
    if args.materialisation == "final" and args.s5_references is None:
        raise SystemExit("a FINAL plan needs --s5-references RS/smokes.json (gate 8 + the D50 multi-turn window measurements)")
    hashes = json.loads(args.adapter_hashes.read_text()) if args.adapter_hashes else None
    if args.peer_roster is None and args.peer_pins is None:
        # the repo's OWN probe roster + pins register the probe row's two cells when no peer pair is given
        probe_roster, probe_pins = args.repo / PROBE_ROSTER_REL, args.repo / PROBE_PINS_REL
        if probe_roster.is_file() and probe_pins.is_file():
            args.peer_roster, args.peer_pins = probe_roster, probe_pins
            print(f"[launch_plan] peer roster / pins defaulted to the lane's S5 probe files {PROBE_ROSTER_REL} + {PROBE_PINS_REL} (W3, F-B)", flush=True)
    inputs = load_inputs(args.repo, peer_roster=args.peer_roster, peer_pins=args.peer_pins, config_roots=list(args.config_root))
    plan = build_plan(W=args.w, wave_uuid=wave["wave_uuid"], reqs_sha256=inputs["reqs_sha256"], roster=inputs["roster"],
                      pins=inputs["pins"], arditi_pins=inputs["arditi_pins"], arms=inputs["arms"], s4=s4,
                      materialisation=args.materialisation, adapter_hashes=hashes, peer_roster=inputs["peer_roster"], peer_pins=inputs["peer_pins"],
                      register_peer_dose_standups=bool(args.register_peer_dose_standups), config_private_runs=inputs["config_private_runs"])
    table = None
    if args.validate_table:
        sys.path.insert(0, str(args.repo / "src"))
        from sisteer.capeval.serving import geometry as table  # type: ignore  # noqa: N813
    problems = validate_plan(plan, table=table)
    if args.compare_draft:
        problems += [f"draft != final outside the S4 fields: {p}" for p in compare_draft_final(json.loads(args.compare_draft.read_text()), plan)]
    justification = None
    coverage = None
    if args.s5_references:
        s5 = json.loads(args.s5_references.read_text()).get("S5") or {}
        if launch_set is not None and read_text is None:
            # the admission re-reads a prior wave's reference standup records off the Volume (opened on the first read only)
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from seedrep_s5_record import _lazy_volume_reader  # noqa: PLC0415  (the certify-side reader; imported here, never at module load)
            read_text = _lazy_volume_reader(args.out.parent)
        gate8, coverage = final_s5_gate_problems(plan, s5, launch_set=launch_set, read_text=read_text)
        problems += gate8
        if args.materialisation == "final":
            # the S5 block's WRAPPER names the smoke's execution wave and the production target; the records are held to
            # the execution wave (they were measured there), the wrapper to THIS plan -- a draft's sha is not the final's, so the wrapper's
            # launch_plan_sha is checked against the DRAFT the final is held to when one is given
            draft_sha = json.loads(args.compare_draft.read_text()).get("launch_plan_sha") if args.compare_draft else None
            block_problems = s5_block_problems(s5, plan={**plan, "launch_plan_sha": plan_sha(plan)}, draft_launch_plan_sha=draft_sha)
            if block_problems:
                problems += [f"S5 block: {p}" for p in block_problems]
            else:
                justification = window_justification(plan, s5.get("multiturn") or [], execution_wave_uuid=str(s5["execution_wave_uuid"]), launch_set=launch_set)
                problems += [f"D50 window gate: {p}" for p in justification["problems"]]
    if problems:
        for p in problems:
            print(f"REFUSED: {p}", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plan["launch_plan_sha"] = plan_sha(plan)          # the FINAL's window_justification mutates the plan AFTER build_plan hashed it
    args.out.write_text(json.dumps(plan, indent=1, sort_keys=False, default=str) + "\n")
    args.ledger.write_text(json.dumps(plan["capacity_ledger"], indent=1) + "\n")
    if justification is not None:
        jpath = args.window_justification_out or args.out.with_name("window_justification.json")
        jpath.write_text(json.dumps(justification, indent=1, default=str) + "\n")
        print(f"[launch_plan] D50 window justification -> {jpath} (reference it from RS/SHA_PROVENANCE.md)")
    if coverage is not None:
        cpath = args.out.with_name(S5_COVERAGE_SIDECAR_NAME)
        cpath.write_text(json.dumps({"schema": S5_COVERAGE_SIDECAR_SCHEMA, "W": plan["W"], "wave_uuid": plan["wave_uuid"], "launch_plan_sha": plan["launch_plan_sha"],
                                     "launch_set": coverage["launch_set"], "coverage": coverage}, indent=1, default=str) + "\n")
        print(f"[launch_plan] gate 8 scoped to the launch set ({len(coverage['launch_set'] or [])} rows): coverage -> {cpath} (certify derives the same block)")
    led = plan["capacity_ledger"]
    print(f"[launch_plan] {args.materialisation}: {len(plan['standups'])} target standups, {len(plan['rungs'])} rung functions "
          f"({led['rungs_declared']['containers']} containers), {len(plan['pools'])} pools -> unstaged peak "
          f"{led['unstaged_peak']['containers']} containers / {led['unstaged_peak']['gpus']} GPUs; S5 roster {len(plan['s5_roster'])} windows; "
          f"adapters {len(plan['adapter_hash_set'])}; sha16 {plan['launch_plan_sha']} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
