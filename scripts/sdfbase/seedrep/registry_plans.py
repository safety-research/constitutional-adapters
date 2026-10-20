#!/usr/bin/env python3
"""The planned-run identity files for the Qwen seed-replicate native battery.

ONE planned-run JSON per (leg, arm, seed) under ``<out>/<leg>/<arm>__s<s>.json`` — 15 legs × 27 arm-runs = 405 files —
each carrying the full ``run`` identity a native emitter needs (``hooks.REQUIRED_IDENTITY_FIELDS`` and the rest of the
``run`` row's required columns), the run id minted by ``identity.run_id`` so the container's store, the launcher's
``queued_modal`` and the grading host completion emitter all name ONE run, and the lane's routing facts under ``extra``.

The contract this module realises:

* ONE object per run: ``producer_object_key_legacy`` = the staged id (15 lane cells), ``none`` (the per-view native
  anchor), ``prompted__<si>`` (6 prompted routes) or ``ar_c0p3__arditi_allpos`` (3 Arditi seed runs); the run key is
  ``seedrep_battery:<W7>:<leg>:<view>`` with view ``s<s>`` for the view engines and ``arditi_s<s>`` for the Arditi engine
  (three client-side seed bases share one engine, so the seed rides in the run key or the three runs collide).
* ``seed_base = s`` (the literal training seed) and ``extra.seed_rule = "literal_training_seed"``.
* ``decode_source`` / ``decode_policy_id`` come from the ONE registered decode-policy row the leg names, resolved ONLY by
  exact ``PolicyFile.by_ref("<ref>@v1")`` with ``superseded_by_policy_id is None`` asserted — never ``resolve(mode=)``,
  never a default lookup; a ref that does not resolve refuses the whole generation. The ``run.decode_source`` column
  is vocabulary-checked (``DECODE_SOURCES``: ``model_card`` …), so it carries the ROW's ``decode_source`` and the exact
  policy id rides in ``decode_policy_id`` (a registry FK) and ``extra.decode_policy_ref``.
* ``decode_seed_rule`` from the controlled list (``per_item`` static / controls / GSM8K, ``per_draw`` BoN, ``fixed`` MMLU);
  the vocabulary has no ``per_turn`` / ``per_response`` / ``per_scenario_turn``, so multi-turn, misalign and Petri
  carry ``per_item`` + ``extra.seed_rule_detail`` — ONE table (:data:`SEED_RULE_DETAIL`, keyed by the WRITERS' row labels:
  Petri's is ``manifest.PETRI_TARGET_SEED_RULE`` = ``per_scenario_turn``) so a vocabulary update is a one-line change;
  a row label the table does not map REFUSES the plan (:func:`seed_rule_of`), never files a guessed rule.
* ``extra.container`` is the ATTEMPT-GROUP axis: the leg for a single-leg container, the store name
  (``controls``) for the legs one combined container executes — the completion and publication join a run to its accepted
  attempts through this axis (or the launch plan's rows), so the eval-leg identity stays separate from container ownership.
* ``instrument_id`` is MINTED here through ``identity.instrument_id`` from the canonical seed instrument of each leg
  (``configs/registry/instruments.yaml``, normalised by the registry's own dimension loader) with the fields this wave
  changes (decode row, seed rule, one trial, the ladder / cap, the window); the field sets are exported under
  ``<out>/_instruments/<instrument_id>.json`` so the registry side registers rows that mint the SAME ids.
* Identity is per CHILD PROCESS: a single-eval child exports one file as ``SISTEER_REGISTRY_RUN_IDENTITY``;
  a multi-eval store child exports a per-TASK MAP — Petri's per-leg file IS a map of its six behaviour runs, and the
  controls store (xstest + or_bench + mmlu in one ``CellStore``) gets ``<out>/_stores/controls/<arm>__s<s>.json`` whose
  routes are the STORE WRITERS' own task names (``judged.TASK_XSTEST`` / ``TASK_ORBENCH``, ``parity.PARITY_TASK_MMLU``,
  ``contracts.inspect_task``, ``petri_runner.PETRI_TASK_PREFIX``; never a plan table).

Every identity is validated STRICTLY through ``emit.check_planned_run`` (the records it would produce), the object sha
against the pins, and the run ids for uniqueness before anything is written; ``--check`` regenerates and refuses drift.

  registry_plans.py --out RS/registry/plans --w-sha <W 40-hex> --campaign <id> [--policy PATH] [--windows JSON] [--check]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "src") not in sys.path and not any(p.endswith("/src") and "sisteer" in os.listdir(p) for p in sys.path if os.path.isdir(p)):
    sys.path.insert(0, str(REPO / "src"))

from sisteer.registry import emit, hooks, identity  # noqa: E402
from sisteer.registry.vocab import STRENGTH_UNITS  # noqa: E402
from sisteer.registry import vocab as V  # noqa: E402
from sisteer.registry.decode_policy import NoPolicy, Policy, PolicyFile  # noqa: E402

RUN_IDENTITY_ENV = hooks.RUN_IDENTITY_ENV
SEEDS: Tuple[int, ...] = (1, 2, 3)
MODEL_KEY = "qwen35_4b"
MODEL_ID = "qwen35_4b_it"
LANE = "seedrep_battery"
SEED_RULE_LITERAL = "literal_training_seed"
NONE_ARM = "none"
PROMPTED_ARMS: Tuple[str, ...] = ("prompted__si_constitution", "prompted__safety_refusal")
PROMPTED_SI: Dict[str, str] = {"prompted__si_constitution": "si_constitution", "prompted__safety_refusal": "safety_refusal"}
ARDITI_ARM = "arditi_allpos"
ARDITI_STAGED_ID = "ar_c0p3__arditi_allpos"
ARDITI_VIEW = "arditi"
ARDITI_KIND = "vector_all"
ARDITI_C = 0.3
ATTACKER_MODEL_ID = "qwen36_27b_abliterated_attacker"
PLAN_SCHEMA = 1
INDEX_NAME = "INDEX.json"
INSTRUMENTS_DIR = "_instruments"
STORES_DIR = "_stores"

DEFAULT_POLICY = REPO / "configs" / "registry" / "decode_policy.json"
DEFAULT_ROSTER = REPO / "configs" / "sdfbase_full22_roster.yaml"
DEFAULT_PINS = REPO / "configs" / "sdfbase_full22_pins.json"
DEFAULT_ARMS = REPO / "configs" / "sdfbase_full22_arms.yaml"
DEFAULT_ARDITI_PINS = REPO / "configs" / "arditi_allpos_pins.json"
DEFAULT_SEEDS_DIR = REPO / "configs" / "registry"

def _petri_row_seed_rule() -> str:
    """The label the Petri writer stamps on its rows: ``sisteer.capeval.manifest.PETRI_TARGET_SEED_RULE`` when the tree carries
    it (``seedrep-ws1b``), else the literal it defines — read from the writer's module so the two cannot drift once merged."""
    try:
        from sisteer.capeval import manifest as _cm
        return str(getattr(_cm, "PETRI_TARGET_SEED_RULE", "per_scenario_turn"))
    except ImportError:
        return "per_scenario_turn"


PETRI_ROW_SEED_RULE = _petri_row_seed_rule()
#: the multi-turn closure's label: the registry's own spelling (``registry-seed-rules``, main 72faab6c — the
#: `multi_turn|off|vendor_preset@v1` row and DECODE_SEED_RULES name it); no writer module on this base names it.
MULTITURN_ROW_SEED_RULE = "per_goal_turn"

#: the WRITERS' seed-rule labels → what a run files when the tree's vocabulary does NOT spell the label: the controlled
#: ``decode_seed_rule`` value + ``extra.seed_rule_detail`` (the accepted fallback). A label the vocabulary spells is filed
#: directly (:func:`seed_rule_of`); a label neither knows REFUSES — never a silently guessed rule.
SEED_RULE_DETAIL: Dict[str, Tuple[str, Optional[str]]] = {
    "per_item": ("per_item", None), "per_draw": ("per_draw", None), "fixed": ("fixed", None),
    MULTITURN_ROW_SEED_RULE: ("per_item", MULTITURN_ROW_SEED_RULE), "per_response": ("per_item", "per_response"),
    PETRI_ROW_SEED_RULE: ("per_item", PETRI_ROW_SEED_RULE),
}
#: a policy row's ``seed_rule`` that constrains nothing: the pre-vocabulary spellings (``per_item`` where a finer closure applies,
#: ``none`` / ``unverified``); any other row value must equal the leg's writer label or the plan and the registry disagree.
SEED_RULE_ROW_UNCONSTRAINED: Tuple[str, ...] = ("per_item", "none", "unverified")


def seed_vocabulary() -> Tuple[str, ...]:
    """The ``decode_seed_rule`` vocabulary the tree's registry VALIDATES against (``vocab.VOCABULARIES``) — read at run time,
    never assumed: a registry update adds ``per_scenario_turn`` / ``per_response`` / ``per_goal_turn``, and the launch tree
    may or may not carry it."""
    return tuple(V.VOCABULARIES["DECODE_SEED_RULES"])


def seed_rule_of(label: str, vocab: Sequence[str]) -> Tuple[str, Optional[str]]:
    """``(decode_seed_rule, extra.seed_rule_detail)`` for a writer's seed-rule label under the vocabulary in force: a label the
    vocabulary spells is emitted DIRECTLY (detail None); a label it lacks is filed through :data:`SEED_RULE_DETAIL`; a label
    neither spells refuses the plan — a planned run never files a guessed rule."""
    if label in vocab:
        return label, None
    try:
        return SEED_RULE_DETAIL[label]
    except KeyError:
        raise RegistryPlanRefused(f"seed-rule label {label!r} is neither in the registry vocabulary {tuple(vocab)} nor mapped by SEED_RULE_DETAIL "
                                  f"(known labels: {sorted(SEED_RULE_DETAIL)}) — a planned run never files a guessed rule") from None


def check_row_seed_rule(pol: Policy, leg_label: str) -> str:
    """The policy row's own ``seed_rule``, cross-checked against the leg's writer label (see :data:`SEED_RULE_ROW_UNCONSTRAINED`)."""
    row = str(pol.seed_rule)
    if row not in SEED_RULE_ROW_UNCONSTRAINED and row != leg_label:
        raise RegistryPlanRefused(f"decode-policy row {pol.policy_id!r} states seed_rule {row!r}, the leg's writer seeds {leg_label!r}; the plan "
                                  "and the registry row disagree on the seed rule")
    return row

#: windows (L) per leg — the multi-turn windows are overridable with ``--windows``
DEFAULT_WINDOWS: Dict[str, int] = {
    "pasttense": 11520, "codeattack": 11520, "renellm": 11520, "grok420": 11520, "bon": 11520,
    "xstest": 7424, "or_bench": 7424, "mmlu": 7424, "gsm8k": 16640, "petri": 32768,
    "misalign_leaking": 24576, "misalign_murder": 24576, "pair": 24576, "crescendo": 49152, "actorattack": 49152,
}
PETRI_BEHAVIOURS: Tuple[str, ...] = ("cyber_bio_chem_capabilities", "self_preservation", "delusion_sycophancy", "defer_to_users",
                                     "reasoning_unfaithfulness", "instruction_evasion")


class RegistryPlanRefused(ValueError):
    """A planned identity that cannot be stated honestly (unresolved policy row, pin mismatch, vocabulary miss, a
    duplicate run id): nothing is written."""


@dataclass(frozen=True)
class Leg:
    name: str
    store: str                       # static | bon | controls | gsm8k | petri | misalign | multiturn
    eval_ids: Tuple[str, ...]        # one eval, or Petri's six behaviours
    item_set_id: str
    n_items_planned: Optional[int]
    policy_ref: str                  # the exact registered row, WITH its version
    token_cap_tokens: int
    token_cap_ladder: Tuple[int, ...]
    read_at: str
    seed_rule: str                   # a SEED_RULE_DETAIL key
    thinking_mode: str
    canonical_instrument: str        # the seed instrument this wave's instrument is derived from
    judge: Optional[str]             # readable judge seed id, None for a deterministic read
    multi_turn: bool
    ladder_promotion_rule: str
    reply_cap: Optional[int] = None
    instrument_overrides: Mapping[str, object] = field(default_factory=dict)
    extra: Mapping[str, object] = field(default_factory=dict)


_STATIC_LADDER = (4096, 8192, 16384)
_STATIC_RULE = ("1 % item-filtered trigger (ladder_decide TRIGGER_SHARE 0.01) per (leg, arm, seed); re-draw with the SAME sampling_seed "
                "at the next cap; ceiling 16,384; rungs streamed to concurrently active rung engines (plan §7)")
_CONTROLS_LADDER = (512, 2048, 8192, 16384)
_CONTROLS_RULE = ("controls ladder 512 -> 2,048 -> 8,192 -> 16,384 with immediate-predecessor escalated_from and the floor recomputed "
                  "from the completed predecessor (plan §7.4); 1 % item-filtered trigger; streamed rungs")


def _static(name: str, item_set: str, n: int) -> Leg:
    return Leg(name, "static", (name,), item_set, n, f"{MODEL_ID}|static|off|vendor_preset@v1", 4096, _STATIC_LADDER, "ladder_final",
               "per_item", "off", "static_sampled_vendor_preset_4096_8192", "strongreject_v1_haiku45", False, _STATIC_RULE)


LEGS: Dict[str, Leg] = {l.name: l for l in (
    _static("pasttense", "static_pasttense_259x2", 518),
    _static("codeattack", "static_codeattack_259x3", 777),
    _static("renellm", "static_renellm_259x3", 777),
    _static("grok420", "static_grok420_259x2", 518),
    Leg("bon", "bon", ("strongreject_bon",), "jbb_goals_100", 100, f"{MODEL_ID}|bon|off|vendor_preset@v1", 4096, (4096,), "fixed",
        "per_draw", "off", "bon_512store_4096climb_T1", "strongreject_v2_haiku45", False,
        "fixed cap 4,096 on every draw (no 512 store, no climb): the VENDOR-preset BoN instrument (D0; not comparable with the paper's T=1.0 rows)",
        instrument_overrides={"bon_n": 75}, extra={"n_draws_per_goal": 75}),
    Leg("xstest", "controls", ("xstest",), "xstest_safe_250", 450, f"{MODEL_ID}|over_refusal|off|vendor_preset@v1", 512, _CONTROLS_LADDER,
        "ladder_final", "per_item", "off", "overrefusal_xstest_sampled_vendor_preset_ladder", "xstest_cpi_v1_sonnet5", False, _CONTROLS_RULE,
        extra={"n_items_generated_planned": 450, "n_items_scored_planned": 250,
               "item_set_note": "450 generated, scored on the 250 SAFE prompts (the canonical instrument's item set)"}),
    Leg("or_bench", "controls", ("or_bench_hard",), "or_bench_hard_200", 200, f"{MODEL_ID}|over_refusal|off|vendor_preset@v1", 512,
        _CONTROLS_LADDER, "ladder_final", "per_item", "off", "overrefusal_orbench_sampled_vendor_preset_ladder", "or_bench_cpi_v1_sonnet5",
        False, _CONTROLS_RULE, extra={"item_set_note": "the 200-item seeded subset (D0: OR-Bench 200), not the canonical 1,319"}),
    Leg("mmlu", "controls", ("mmlu",), "mmlu_200", 200, f"{MODEL_ID}|mmlu|off|greedy@v1", 1, (1,), "fixed", "fixed", "off",
        "mmlu_letter_prefill_200", None, False, "n.a. (1-token letter-logprob prefill read)", reply_cap=1,
        extra={"seed_note": "seed 0 sent on every request; instrument_frozen (sampler-moot)"}),
    Leg("gsm8k", "gsm8k", ("gsm8k",), "gsm8k_platinum_paired_200", 200, f"{MODEL_ID}|gsm8k|on|thinking_card@v1", 8192, (8192, 16384),
        "ladder_final", "per_item", "on", "gsm8k_paired200_inspect_10shot_thinking_on_8192", None, False,
        "16,384 rung iff n_at_cap / n > 0.01 (inspect_rung GSM8K_TRIGGER_THRESHOLD 0.01); eligible = at_cap minus loops; one rung",
        extra={"geometry_class": "common_f22"}),
    Leg("petri", "petri", tuple(f"petri_{b}" for b in PETRI_BEHAVIOURS), "petri_scenarios_6x10", 10, f"{MODEL_ID}|petri|off|vendor_preset@v1",
        1024, (1024,), "fixed", PETRI_ROW_SEED_RULE, "off", "petri_full22_as_run", "petri_judge_opus5", True,
        "1,024 per reply, cap unchanged; a context-overflow re-run widens the window to 49,152 on the rung proxy (plan §7)",
        reply_cap=1024, instrument_overrides={"max_turns": 15}),
    Leg("misalign_leaking", "misalign", ("agentic_leaking",), "agentic_leaking_scenarios", 51, f"{MODEL_ID}|agentic|off|vendor@v1", 3000,
        (3000, 16384), "ladder_final", "per_response", "off", "agentic_misalignment_runner_T1_3000_16384", "agentic_classifier_sonnet5", True,
        "3,000 -> 16,384 rung on truncated_selection (zlib loop rule) into a NEW root, merged by misalign_ladder_merge.py",
        extra={"samples_per_scenario": 10}),
    Leg("misalign_murder", "misalign", ("agentic_murder",), "agentic_murder_scenarios", 49, f"{MODEL_ID}|agentic|off|vendor@v1", 3000,
        (3000, 16384), "ladder_final", "per_response", "off", "agentic_misalignment_runner_T1_3000_16384", "agentic_classifier_sonnet5", True,
        "3,000 -> 16,384 rung on truncated_selection (zlib loop rule) into a NEW root, merged by misalign_ladder_merge.py",
        extra={"samples_per_scenario": 10}),
    Leg("pair", "multiturn", ("pair",), "multiturn_goals_259", 259, f"{MODEL_ID}|multi_turn|off|vendor_preset@v1", 4096, (4096,), "fixed",
        MULTITURN_ROW_SEED_RULE, "off", "multiturn_pair_cap4096_generation_config", "strongreject_v1_haiku45", True,
        "per-reply cap 4,096; multi-turn has no ladder", reply_cap=4096),
    Leg("crescendo", "multiturn", ("crescendo",), "multiturn_goals_259", 259, f"{MODEL_ID}|multi_turn|off|vendor_preset@v1", 4096, (4096,),
        "fixed", MULTITURN_ROW_SEED_RULE, "off", "multiturn_crescendo_cap4096_generation_config", "strongreject_v1_haiku45", True,
        "per-reply cap 4,096; multi-turn has no ladder", reply_cap=4096),
    Leg("actorattack", "multiturn", ("actorattack",), "multiturn_goals_259", 259, f"{MODEL_ID}|multi_turn|off|vendor_preset@v1", 4096, (4096,),
        "fixed", MULTITURN_ROW_SEED_RULE, "off", "multiturn_actorattack_cap4096_generation_config", "strongreject_v1_haiku45", True,
        "per-reply cap 4,096; multi-turn has no ladder", reply_cap=4096,
        instrument_overrides={"max_turns": None}, extra={"max_turns_note": "the canonical row states `unverified`; not an integer, left null"}),
)}
LEG_NAMES: Tuple[str, ...] = tuple(LEGS)
assert len(LEG_NAMES) == 15, LEG_NAMES
#: the multi-eval STORES whose child gets a per-task MAP; a leg absent here is a single-identity child
STORE_LEGS: Dict[str, Tuple[str, ...]] = {"controls": ("xstest", "or_bench", "mmlu")}
#: the ATTEMPT-GROUP (container) axis per leg: the legs one combined container executes share its store's
#: name; every other leg is its own container. ``extra.container`` on every identity; the accepted map's third key slot.
CONTAINER_OF_LEG: Dict[str, str] = {leg: store for store, legs in STORE_LEGS.items() for leg in legs}


def container_of_leg(leg: str) -> str:
    return CONTAINER_OF_LEG.get(leg, leg)


# --------------------------------------------------------------------------------------------
# the store writers' task names (derived from the writer, never a plan table)
# --------------------------------------------------------------------------------------------


def store_task_names() -> Dict[str, Callable[[str], str]]:
    """``{leg: eval_id -> the store's canonical row task}`` for the legs whose plan file routes rows by task, imported
    from the modules that WRITE those rows. An import failure refuses (never a fallback string)."""
    from sisteer.capeval.tiers.judged import TASK_ORBENCH, TASK_XSTEST
    from sisteer.capeval.tiers.parity import PARITY_TASK_MMLU
    from sisteer.capeval.tiers.petri_runner import PETRI_TASK_PREFIX
    from sisteer.capeval.tinycap.contracts import GSM8K_PAIRED_TASK_NAME, inspect_task
    return {
        "xstest": lambda _e: TASK_XSTEST,
        "or_bench": lambda _e: TASK_ORBENCH,
        "mmlu": lambda _e: PARITY_TASK_MMLU,
        "gsm8k": lambda _e: inspect_task(GSM8K_PAIRED_TASK_NAME),
        "petri": lambda e: PETRI_TASK_PREFIX + e[len("petri_"):],
    }


# --------------------------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------------------------


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise RegistryPlanRefused(f"{path}: not a mapping")
    return data


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise RegistryPlanRefused(f"{path}: not a JSON object")
    return data


def resolve_policy(pf: PolicyFile, ref: str) -> Policy:
    """The ONE admissible resolution: exact ``by_ref`` on a versioned ref, the row live (not superseded) and on the
    lane's model. ``resolve(mode=)`` / ``find`` / a default lookup are never consulted."""
    if "@v" not in ref:
        raise RegistryPlanRefused(f"decode-policy ref {ref!r} carries no version; the plan names rows as <ref>@v<n> only")
    try:
        pol = pf.by_ref(ref)
    except NoPolicy as exc:
        raise RegistryPlanRefused(f"decode-policy row {ref!r} is not registered in {pf.path} ({exc}); the wave refuses an unregistered "
                                  "decode_source (plan §10 / WS0-6)") from None
    if pol.superseded_by_policy_id:
        raise RegistryPlanRefused(f"decode-policy row {ref!r} is superseded by {pol.superseded_by_policy_id!r}; a superseded row fails closed")
    if pol.model_id != MODEL_ID:
        raise RegistryPlanRefused(f"decode-policy row {ref!r} belongs to model {pol.model_id!r}, not {MODEL_ID}")
    if pol.policy_id != ref:
        raise RegistryPlanRefused(f"decode-policy row {ref!r} resolved to policy_id {pol.policy_id!r}; the plan names the exact row id")
    if pol.decode_source not in V.DECODE_SOURCES:
        raise RegistryPlanRefused(f"decode-policy row {ref!r} states decode_source {pol.decode_source!r}, not in DECODE_SOURCES")
    return pol


@dataclass
class Arm:
    name: str            # the harness cell id (the ARM axis of the coverage table)
    kind: str            # fitted | none | prompted | arditi
    staged_id: str       # producer_object_key_legacy
    object_id: str
    served_key: str      # the endpoint["adapters"] / served_positions key of the arm's route (emit_serving_facts_fanout)
    transfer_kind: str
    applied_positions_mode: str
    extra: Dict[str, object] = field(default_factory=dict)


@dataclass
class Inputs:
    campaign: str
    w_sha: str
    policy: PolicyFile
    roster: Mapping[str, Any]
    pins: Mapping[str, Any]
    arms_table: Mapping[str, Any]
    arditi_pins: Mapping[str, Any]
    windows: Mapping[str, int]
    judge_ids: Mapping[str, str]
    canonical: Mapping[Tuple[str, str], Mapping[str, Any]]    # (seed instrument id, eval_id) -> normalised instrument payload
    si_sha: Mapping[str, str]                                  # si_id -> si_text_sha256
    task_names: Mapping[str, Callable[[str], str]]
    seed_vocab: Tuple[str, ...]                                # the decode_seed_rule vocabulary in force (load_inputs: the tree's, read at load)

    @property
    def w7(self) -> str:
        return self.w_sha[:7]


def _seed_bundle(seeds_dir: Path):
    from sisteer.registry.adapters import dimensions as D
    seeds = D.load_seeds(str(seeds_dir))
    bundle = D.build_records(seeds, emit.producer_block("scripts/sdfbase/seedrep/registry_plans.py"), verify_si=False)
    canonical: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    for rec in bundle.records["instrument"]:
        p = rec["payload"]
        ex = p.get("extra") or {}
        models = list(ex.get("model_ids") or ([ex["model_id"]] if ex.get("model_id") else []))
        if models and MODEL_ID not in models:
            continue
        key = (str(ex.get("seed_instrument_id")), str(p["eval_id"]))
        if key in canonical:
            raise RegistryPlanRefused(f"two canonical instrument payloads for {key} on {MODEL_ID}; cannot derive from an ambiguous seed")
        canonical[key] = p
    si_sha = {str(r["payload"]["si_id"]): str(r["payload"]["si_text_sha256"]) for r in bundle.records["si_text"]}
    return dict(bundle.judge_ids), canonical, si_sha


def load_inputs(*, campaign: str, w_sha: str, policy: Path = DEFAULT_POLICY, roster: Path = DEFAULT_ROSTER, pins: Path = DEFAULT_PINS,
                arms: Path = DEFAULT_ARMS, arditi_pins: Path = DEFAULT_ARDITI_PINS, seeds_dir: Path = DEFAULT_SEEDS_DIR,
                windows: Optional[Mapping[str, int]] = None, seed_vocab: Optional[Sequence[str]] = None) -> Inputs:
    if not re.fullmatch(r"[0-9a-f]{40}", w_sha or ""):
        raise RegistryPlanRefused(f"--w-sha must be the 40-hex W commit, got {w_sha!r}")
    if not campaign or ":" in campaign or "/" in campaign:
        raise RegistryPlanRefused(f"campaign {campaign!r} must be non-empty and free of ':' and '/' (a run_id separator / a path)")
    win = dict(DEFAULT_WINDOWS)
    for k, v in (windows or {}).items():
        if k not in win:
            raise RegistryPlanRefused(f"--windows names unknown leg {k!r}; legs: {LEG_NAMES}")
        win[k] = int(v)
    judge_ids, canonical, si_sha = _seed_bundle(Path(seeds_dir))
    return Inputs(campaign=campaign, w_sha=w_sha, policy=PolicyFile.load(str(policy)), roster=_load_yaml(Path(roster)), pins=_load_json(Path(pins)),
                  arms_table=_load_yaml(Path(arms)), arditi_pins=_load_json(Path(arditi_pins)), windows=win, judge_ids=judge_ids,
                  canonical=canonical, si_sha=si_sha, task_names=store_task_names(),
                  seed_vocab=tuple(seed_vocab) if seed_vocab is not None else seed_vocabulary())


# --------------------------------------------------------------------------------------------
# arms
# --------------------------------------------------------------------------------------------


def _arditi_block(arms_table: Mapping[str, Any]) -> Dict[str, Any]:
    found: List[Dict[str, Any]] = []

    def walk(o):
        if isinstance(o, dict):
            if o.get("staged_id") == ARDITI_STAGED_ID:
                found.append(o)
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(arms_table)
    if len(found) != 1:
        raise RegistryPlanRefused(f"the arms table names {ARDITI_STAGED_ID!r} {len(found)} times; expected exactly one operating point")
    return found[0]


def arms_for_seed(inp: Inputs, s: int) -> List[Arm]:
    """The 9 arm-runs of seed ``s`` on the leg's engines: the view's 5 fitted cells (roster → pins), the view's native
    ``none``, the two prompted routes, and the Arditi operating point at seed base ``s``."""
    view = f"s{s}"
    try:
        vblock = inp.roster["targets"][MODEL_KEY]["views"][view]
    except (KeyError, TypeError):
        raise RegistryPlanRefused(f"roster has no view {MODEL_KEY}/{view}") from None
    cells = dict(vblock.get("cells") or {})
    fitted = {k: v for k, v in cells.items() if v is not None}
    prompted_in_roster = tuple(k for k, v in cells.items() if v is None)
    if prompted_in_roster and set(prompted_in_roster) != set(PROMPTED_ARMS):
        raise RegistryPlanRefused(f"roster view {view} lists prompted cells {prompted_in_roster}, the plan expects {PROMPTED_ARMS}")
    if len(fitted) != 5:
        raise RegistryPlanRefused(f"roster view {view} holds {len(fitted)} fitted cells, the plan expects 5: {sorted(fitted)}")
    pins = inp.pins.get(MODEL_KEY) or {}
    out: List[Arm] = []
    for harness_id, staged in fitted.items():
        entry = pins.get(staged)
        if not isinstance(entry, dict) or not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("object_sha256", ""))):
            raise RegistryPlanRefused(f"pins carry no 64-hex object_sha256 for staged id {staged!r} (view {view}, cell {harness_id})")
        if not re.fullmatch(rf"(derived_)?base64s{s}__.+", staged):
            raise RegistryPlanRefused(f"staged id {staged!r} is not a seed-{s} lane object (base64s{s}__ / derived_base64s{s}__)")
        derived = staged.startswith("derived_")
        out.append(Arm(harness_id, "fitted", staged, str(entry["object_sha256"]), harness_id, "base_to_instruct", "all",
                       {"staged_id": staged, "derived": derived, "pin_source": entry.get("source")}))
    out.append(Arm(NONE_ARM, "none", NONE_ARM, identity.object_id("none"), NONE_ARM, "none", "none", {}))
    for arm in PROMPTED_ARMS:
        si_id = PROMPTED_SI[arm]
        sha = inp.si_sha.get(si_id)
        if not sha or len(sha) != 64:
            raise RegistryPlanRefused(f"si_texts.yaml carries no si_text_sha256 for {si_id!r} (prompted route {arm})")
        out.append(Arm(arm, "prompted", arm, identity.object_id("system_instruction", si_text_sha256=sha), arm, "prompted_only", "none",
                       {"si_id": si_id, "si_text_sha256": sha}))
    ab = _arditi_block(inp.arms_table)
    pin = (inp.arditi_pins.get(MODEL_KEY) or {}).get(ARDITI_STAGED_ID)
    if not isinstance(pin, dict):
        raise RegistryPlanRefused(f"arditi pins carry no entry {MODEL_KEY}/{ARDITI_STAGED_ID}")
    sha = str(pin.get("object_sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", sha) or sha != str(ab.get("object_sha256")):
        raise RegistryPlanRefused(f"Arditi pin sha {sha[:12]}… disagrees with the arms table's {str(ab.get('object_sha256'))[:12]}…")
    if pin.get("kind") != ARDITI_KIND or float(pin.get("c", -1)) != ARDITI_C or float(ab.get("c", -1)) != ARDITI_C:
        raise RegistryPlanRefused(f"Arditi pin/arms entry is not the c={ARDITI_C} {ARDITI_KIND} operating point: kind={pin.get('kind')!r} "
                                  f"c={pin.get('c')!r}/{ab.get('c')!r}")
    out.append(Arm(ARDITI_ARM, "arditi", ARDITI_STAGED_ID, identity.object_id(ARDITI_KIND, artifact_sha256=sha), ARDITI_ARM, "instruct_direct",
                   "all", {"staged_id": ARDITI_STAGED_ID, "dose": {"c": ARDITI_C, "unit": str(pin.get("dose_unit") or "raw_coefficient")},
                           "selection_outcome": ab.get("selection_outcome"), "alpha_equiv": pin.get("alpha_equiv"),
                           "served_layer": pin.get("served_layer"), "direction_sha256": pin.get("direction_sha256")}))
    return out


def view_of(arm: Arm, s: int) -> str:
    """The attempt-root axis ``<view|arditi>``: ``s<s>`` for the view engines, ``arditi`` for the Arditi engine."""
    return ARDITI_VIEW if arm.kind == "arditi" else f"s{s}"


def run_key(inp: Inputs, leg: str, arm: Arm, s: int) -> str:
    view = f"arditi_s{s}" if arm.kind == "arditi" else f"s{s}"
    return f"{LANE}:{inp.w7}:{leg}:{view}"


# --------------------------------------------------------------------------------------------
# instruments and identities
# --------------------------------------------------------------------------------------------

_DECODE_COLS = {"temperature": "decode_temperature", "top_p": "decode_top_p", "top_k": "decode_top_k", "min_p": "decode_min_p",
                "repetition_penalty": "decode_repetition_penalty", "presence_penalty": "decode_presence_penalty"}


def instrument_fields(inp: Inputs, leg: Leg, eval_id: str) -> Tuple[Dict[str, object], Dict[str, object]]:
    """The canonical instrument identity of ``(leg, eval)`` for this wave, derived from the seed instrument's normalised
    payload with the fields the spec changes; returns ``(fields, provenance)``."""
    base = inp.canonical.get((leg.canonical_instrument, eval_id))
    if base is None:
        raise RegistryPlanRefused(f"no canonical instrument payload for ({leg.canonical_instrument}, {eval_id}) on {MODEL_ID} in instruments.yaml")
    pol = resolve_policy(inp.policy, leg.policy_ref)
    sent = pol.request_params()
    seed_rule, seed_rule_detail = seed_rule_of(leg.seed_rule, inp.seed_vocab)
    seed_rule_row = check_row_seed_rule(pol, leg.seed_rule)
    fields: Dict[str, object] = {name: base.get(name) for name, _t in identity.INSTRUMENT_ID_FIELDS}
    changes: Dict[str, object] = {
        "eval_id": eval_id, "item_set_id": leg.item_set_id,
        "harness_version": re.sub(r"\s*@\s*base_commit\b", "", str(base.get("harness_version") or "")).strip() + f" @ {LANE} W {inp.w7}",
        "token_cap_tokens": int(leg.token_cap_ladder[-1]), "token_cap_ladder_json": list(leg.token_cap_ladder),
        "ladder_promotion_rule": leg.ladder_promotion_rule, "read_at": leg.read_at, "reply_cap_tokens": leg.reply_cap,
        "context_window_tokens": int(inp.windows[leg.name]), "decode_source": pol.decode_source,
        "decode_seed_rule": seed_rule, "thinking_mode": leg.thinking_mode, "n_trials": 1,
        "serving_path": "lora_native", "applied_positions_mode": "all",
        "judge_id": None,
    }
    for pk, col in _DECODE_COLS.items():
        changes[col] = sent.get(pk)
    if leg.judge is not None:
        if leg.judge not in inp.judge_ids:
            raise RegistryPlanRefused(f"judge {leg.judge!r} is not in judges.yaml")
        changes["judge_id"] = inp.judge_ids[leg.judge]
    changes.update(dict(leg.instrument_overrides))
    changed = {k: v for k, v in changes.items() if fields.get(k) != v}
    fields.update(changes)
    for name, ok in (("read_at", V.READ_AT), ("thinking_mode", V.THINKING_MODES), ("decode_seed_rule", inp.seed_vocab),
                     ("decode_source", V.DECODE_SOURCES), ("serving_path", V.SERVING_PATHS), ("applied_positions_mode", V.APPLIED_POSITIONS_MODES),
                     ("harness", V.HARNESSES), ("scoring_rule", V.SCORING_RULES), ("answer_extraction", V.ANSWER_EXTRACTIONS),
                     ("cot_prompting", V.COT_PROMPTING)):
        if fields.get(name) is not None and fields[name] not in ok:
            raise RegistryPlanRefused(f"instrument field {name}={fields[name]!r} is not in the vocabulary for {leg.name}/{eval_id}")
    iid = identity.instrument_id(fields)
    prov = {"instrument_id": iid, "derived_from_seed_instrument": leg.canonical_instrument, "derived_from_instrument_id": base.get("instrument_id"),
            "changed_fields": changed, "decode_policy_id": pol.policy_id, "decode_policy_ref": leg.policy_ref,
            "seed_rule_detail": seed_rule_detail, "seed_rule_label": leg.seed_rule, "seed_rule_row": seed_rule_row, "seed_vocabulary": list(inp.seed_vocab),
            "judge_seed_id": leg.judge, "lane": LANE, "w7": inp.w7,
            "note": ("minted through identity.instrument_id over the normalised canonical fields with this wave's changes; register an "
                     "`instrument` row with exactly these fields to mint the same id")}
    return fields, prov


def identity_for(inp: Inputs, leg: Leg, eval_id: str, arm: Arm, s: int, instrument: Mapping[str, object], instrument_id: str,
                 route_task: Optional[str]) -> Dict[str, object]:
    """One planned ``run`` identity (the file a child exports, or one route of a per-task map)."""
    if s not in SEEDS:
        raise RegistryPlanRefused(f"seed {s!r} is not one of {SEEDS}")
    pol = resolve_policy(inp.policy, leg.policy_ref)
    seed_rule, detail = seed_rule_of(leg.seed_rule, inp.seed_vocab)
    seed_rule_row = check_row_seed_rule(pol, leg.seed_rule)
    prk = run_key(inp, leg.name, arm, s)
    if arm.kind in ("none", "prompted"):
        scope = "not_applicable"
    else:
        scope = "every_turn_prefill_and_decode" if leg.multi_turn else "single_turn"
    view = view_of(arm, s)
    ident: Dict[str, object] = {
        "run_id": identity.run_id(inp.campaign, prk, arm.staged_id, eval_id, instrument_id),
        "campaign": inp.campaign,
        "producer_run_key_legacy": prk,
        "producer_object_key_legacy": arm.staged_id,
        "model_id": MODEL_ID,
        "object_id": arm.object_id,
        "transfer_kind": arm.transfer_kind,
        "eval_id": eval_id,
        "item_set_id": leg.item_set_id,
        "instrument_id": instrument_id,
        "instrument_match": "match",
        "harness": instrument["harness"],
        "harness_version": instrument["harness_version"],
        "scoring_rule": instrument["scoring_rule"],
        "prompt_template_sha256": "unverified",
        "n_shot": instrument.get("n_shot"),
        "cot_prompting": instrument.get("cot_prompting"),
        "decode_seed_rule": seed_rule,
        "decode_source": pol.decode_source,
        "decode_policy_id": pol.policy_id,
        "token_cap_tokens": int(leg.token_cap_tokens),
        "token_cap_rung_index": 0,
        "token_cap_ladder_json": list(leg.token_cap_ladder),
        "reply_cap_tokens": leg.reply_cap,
        "context_window_tokens": int(inp.windows[leg.name]),
        "thinking_mode": leg.thinking_mode,
        "serving_path": "lora_native",
        "applied_positions_mode": arm.applied_positions_mode,
        "multi_turn_steering_scope": scope,
        "code_sha": inp.w_sha,
        "substrate": "modal",
        "substrate_source": "recorded",
        "seed_base": int(s),
        "trial_index": 0,
        "n_items_planned": leg.n_items_planned,
        "judge_id": instrument.get("judge_id"),
        "attacker_model_id": instrument.get("attacker_model_id"),
        "view_legacy": view,
        "campaign_cell_id_legacy": arm.name,
        "lane_source_legacy": LANE,
        "field_confidence_json": {"prompt_template_sha256": {"confidence": "unverified", "reason": (
            "the served path renders the chat template server-side; no prompt-template hash is recorded for this harness")}},
        "extra": {
            "seed_rule": SEED_RULE_LITERAL, "seed_rule_detail": detail, "seed_rule_label": leg.seed_rule, "seed_rule_row": seed_rule_row, "leg": leg.name,
            "container": container_of_leg(leg.name), "arm": arm.name, "arm_kind": arm.kind, "s": int(s),
            "view": view, "served_key": arm.served_key, "store": leg.store, "store_task": route_task, "decode_policy_ref": leg.policy_ref,
            "canonical_instrument_legacy": leg.canonical_instrument, "w7": inp.w7, "lane": LANE, "plan_schema": PLAN_SCHEMA,
            **{k: v for k, v in arm.extra.items()}, **{k: v for k, v in leg.extra.items()},
        },
    }
    if arm.kind == "arditi":
        # fr5 (3): the unit is KNOWN (the pin's dose_unit, `extra.dose.unit`); the planned record carried none and the fold read
        # `unverified` on every Arditi run -- an identity field, unpatchable after the fact
        dose = arm.extra.get("dose") if isinstance(arm.extra.get("dose"), dict) else {}
        unit = str(dose.get("unit") or "")
        if unit not in STRENGTH_UNITS:
            raise RegistryPlanRefused(f"Arditi arm {arm.name!r}: extra.dose.unit {unit!r} is not a registry strength unit ({STRENGTH_UNITS})")
        ident.update({"strength_nominal": ARDITI_C, "strength_nominal_source": "filed", "strength_unit": unit, "is_operating_point": 1})
    elif arm.kind == "fitted":
        ident.update({"strength_nominal": 1.0, "strength_nominal_source": "filed", "strength_unit": "nominal_only", "is_operating_point": 1})
    ident = {k: v for k, v in ident.items() if v is not None or k in ("judge_id", "n_shot", "cot_prompting", "reply_cap_tokens", "attacker_model_id")}
    return ident


def check_identity(ident: Mapping[str, object]) -> str:
    """Strict validation: the records the plan would produce must validate, every required
    identity field must be present, and the minted run id must equal the one the file carries. Returns the run id."""
    missing = hooks.missing_identity_fields(ident)
    if missing:
        raise RegistryPlanRefused(f"identity lacks required fields {missing}")
    try:
        rid = emit.check_planned_run(ident)
    except emit.PlanRefused as exc:
        raise RegistryPlanRefused(str(exc)) from None
    if ident.get("run_id") != rid:
        raise RegistryPlanRefused(f"run_id {ident.get('run_id')!r} disagrees with the id minted from its fields {rid!r}")
    return rid


# --------------------------------------------------------------------------------------------
# the plan set
# --------------------------------------------------------------------------------------------


def plan_relpath(leg: str, arm: str, s: int) -> str:
    return f"{leg}/{arm}__s{s}.json"


def store_relpath(store: str, arm: str, s: int) -> str:
    return f"{STORES_DIR}/{store}/{arm}__s{s}.json"


def _canonical_bytes(obj) -> bytes:
    return (json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


@dataclass
class PlanSet:
    files: Dict[str, Dict[str, object]]        # relpath -> file content (an identity, or a {route: identity} map)
    index: Dict[str, object]

    def run_ids(self) -> List[str]:
        out: List[str] = []
        for rel, content in self.files.items():
            if rel.startswith(f"{STORES_DIR}/") or rel.startswith(f"{INSTRUMENTS_DIR}/"):
                continue
            for _route, ident in identities_of(content).items():
                out.append(str(ident["run_id"]))
        return out


def identities_of(content: Mapping[str, object]) -> Dict[str, Dict[str, object]]:
    """``{route: identity}`` of a plan file's content (the same rule as ``hooks.normalise_identities``)."""
    return hooks.normalise_identities(content)


def build(inp: Inputs, legs: Optional[Sequence[str]] = None) -> PlanSet:
    legs = list(legs or LEG_NAMES)
    unknown = [l for l in legs if l not in LEGS]
    if unknown:
        raise RegistryPlanRefused(f"unknown legs {unknown}; known: {LEG_NAMES}")
    files: Dict[str, Dict[str, object]] = {}
    index: Dict[str, Any] = {"plan_schema": PLAN_SCHEMA, "lane": LANE, "campaign": inp.campaign, "w_sha": inp.w_sha, "w7": inp.w7,
                             "windows": dict(inp.windows), "legs": {}, "stores": {}, "instruments": {},
                             # the attempt-group axis: container -> the legs its attempts complete
                             "containers": {c: {"legs": [l for l in legs if container_of_leg(l) == c]}
                                            for c in sorted({container_of_leg(l) for l in legs})}}
    seen_run_ids: Dict[str, str] = {}
    instruments: Dict[str, Dict[str, object]] = {}
    # per-leg identities
    per_leg_ident: Dict[Tuple[str, str, int], Dict[str, Dict[str, object]]] = {}   # (leg, arm, s) -> {route: identity}
    for leg_name in legs:
        leg = LEGS[leg_name]
        task_of = inp.task_names.get(leg_name)
        inst_by_eval: Dict[str, Tuple[Mapping[str, object], str]] = {}
        for eval_id in leg.eval_ids:
            fields, prov = instrument_fields(inp, leg, eval_id)
            iid = str(prov["instrument_id"])
            inst_by_eval[eval_id] = (fields, iid)
            instruments.setdefault(iid, {"fields": fields, "provenance": prov, "legs": []})
            instruments[iid]["legs"].append({"leg": leg_name, "eval_id": eval_id})   # type: ignore[union-attr]
        index["legs"][leg_name] = {"store": leg.store, "eval_ids": list(leg.eval_ids), "policy_ref": leg.policy_ref, "arms": {}}
        for s in SEEDS:
            for arm in arms_for_seed(inp, s):
                routes: Dict[str, Dict[str, object]] = {}
                for eval_id in leg.eval_ids:
                    fields, iid = inst_by_eval[eval_id]
                    task = task_of(eval_id) if task_of else None
                    ident = identity_for(inp, leg, eval_id, arm, s, fields, iid, task)
                    rid = check_identity(ident)
                    if rid in seen_run_ids:
                        raise RegistryPlanRefused(f"run_id collision: {rid} for {leg_name}/{arm.name}/s{s}/{eval_id} and {seen_run_ids[rid]}")
                    seen_run_ids[rid] = f"{leg_name}/{arm.name}/s{s}/{eval_id}"
                    routes[task if (task and len(leg.eval_ids) > 1) else "*"] = ident
                if len(routes) == 1 and "*" in routes:
                    content: Dict[str, object] = routes["*"]
                else:
                    content = dict(routes)
                rel = plan_relpath(leg_name, arm.name, s)
                files[rel] = content
                per_leg_ident[(leg_name, arm.name, s)] = {(task_of(e) if task_of else "*"): routes[t] for t, e in
                                                          zip(routes, leg.eval_ids)}
                index["legs"][leg_name]["arms"].setdefault(arm.name, {})[str(s)] = {
                    "path": rel, "view": view_of(arm, s), "served_key": arm.served_key, "arm_kind": arm.kind,
                    "producer_object_key_legacy": arm.staged_id, "object_id": arm.object_id,
                    "run_ids": {route: str(i["run_id"]) for route, i in identities_of(content).items()}}
    # the multi-eval STORE maps (one child, several evals): routes keyed by the writers' task names
    for store, store_legs in STORE_LEGS.items():
        present = [l for l in store_legs if l in legs]
        if not present:
            continue
        if set(present) != set(store_legs):
            raise RegistryPlanRefused(f"store {store!r} needs all of {store_legs} in --legs (got {present}); a partial map would misroute rows")
        index["stores"][store] = {"legs": list(store_legs), "arms": {}}
        for s in SEEDS:
            for arm in arms_for_seed(inp, s):
                routes = {}
                for leg_name in store_legs:
                    task_of = inp.task_names.get(leg_name)
                    if task_of is None:
                        raise RegistryPlanRefused(f"store {store!r}: leg {leg_name!r} has no writer task name; a map needs one per route")
                    for route, ident in per_leg_ident[(leg_name, arm.name, s)].items():
                        key = task_of(str(ident["eval_id"])) if route == "*" else route
                        if key in routes:
                            raise RegistryPlanRefused(f"store {store!r}: two legs route task {key!r}")
                        routes[key] = ident
                rel = store_relpath(store, arm.name, s)
                files[rel] = routes
                index["stores"][store]["arms"].setdefault(arm.name, {})[str(s)] = {
                    "path": rel, "view": view_of(arm, s), "served_key": arm.served_key,
                    "run_ids": {route: str(i["run_id"]) for route, i in routes.items()}}
    for iid, block in instruments.items():
        files[f"{INSTRUMENTS_DIR}/{iid}.json"] = block
        index["instruments"][iid] = {"legs": block["legs"], "derived_from_seed_instrument": block["provenance"]["derived_from_seed_instrument"]}  # type: ignore[index]
    n_plan_files = sum(1 for r in files if not r.startswith((f"{STORES_DIR}/", f"{INSTRUMENTS_DIR}/")))
    index["n_plan_files"] = n_plan_files
    index["n_store_files"] = sum(1 for r in files if r.startswith(f"{STORES_DIR}/"))
    index["n_instruments"] = len(instruments)
    index["n_run_ids"] = len(seen_run_ids)
    return PlanSet(files, index)


def write_plan_set(ps: PlanSet, out: Path) -> List[Path]:
    out = Path(out)
    written: List[Path] = []
    for rel, content in sorted(ps.files.items()):
        p = out / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(_canonical_bytes(content))
        written.append(p)
    (out / INDEX_NAME).write_bytes(_canonical_bytes(ps.index))
    written.append(out / INDEX_NAME)
    return written


def check_plan_set(ps: PlanSet, out: Path) -> List[str]:
    """Byte-for-byte drift between the generated set and ``out`` (empty = identical); a missing or extra file is drift."""
    out = Path(out)
    problems: List[str] = []
    expected = {rel: _canonical_bytes(c) for rel, c in ps.files.items()}
    expected[INDEX_NAME] = _canonical_bytes(ps.index)
    for rel, want in sorted(expected.items()):
        p = out / rel
        if not p.is_file():
            problems.append(f"missing: {rel}")
        elif p.read_bytes() != want:
            problems.append(f"differs: {rel}")
    on_disk = {str(p.relative_to(out)) for p in out.rglob("*.json")} if out.is_dir() else set()
    for rel in sorted(on_disk - set(expected)):
        problems.append(f"unexpected: {rel}")
    return problems


# --------------------------------------------------------------------------------------------
# the consumers' API (the per-child env; the lookups)
# --------------------------------------------------------------------------------------------


@dataclass
class PlanFile:
    path: Path
    identities: Dict[str, Dict[str, object]]     # route -> identity

    @property
    def run_ids(self) -> Dict[str, str]:
        return {route: str(i["run_id"]) for route, i in self.identities.items()}

    def _extra(self, key: str):
        vals = {str((i.get("extra") or {}).get(key)) for i in self.identities.values()}
        if len(vals) != 1:
            raise RegistryPlanRefused(f"{self.path}: routes disagree on extra.{key}: {sorted(vals)}")
        return vals.pop()

    @property
    def leg(self) -> str:
        legs = {str((i.get("extra") or {}).get("leg")) for i in self.identities.values()}
        return legs.pop() if len(legs) == 1 else "+".join(sorted(legs))

    @property
    def arm(self) -> str:
        return self._extra("arm")

    @property
    def s(self) -> int:
        return int(self._extra("s"))

    @property
    def view(self) -> str:
        return self._extra("view")

    @property
    def container(self) -> str:
        """The attempt-group axis every route of the file shares (``extra.container``, required by :func:`load_plan_file`)."""
        return self._extra("container")

    @property
    def served_key(self) -> str:
        return self._extra("served_key")

    @property
    def campaign(self) -> str:
        return self._extra("campaign") if any("campaign" in (i.get("extra") or {}) for i in self.identities.values()) else \
            {str(i["campaign"]) for i in self.identities.values()}.pop()


def load_plan_file(path) -> PlanFile:
    """A plan file as the container reads it (``hooks.identity_from_env`` / ``normalise_identities``); refuses a file with
    no identity or a route lacking its run id."""
    p = Path(path)
    content = _load_json(p)
    idents = identities_of(content)
    if not idents:
        raise RegistryPlanRefused(f"{p}: no registry identity (neither a single identity nor a {{task: identity}} map)")
    for route, ident in idents.items():
        if not ident.get("run_id"):
            raise RegistryPlanRefused(f"{p}: route {route!r} carries no run_id; the store would mint a different id from its own key")
        missing = hooks.missing_identity_fields(ident)
        if missing:
            raise RegistryPlanRefused(f"{p}: route {route!r} lacks {missing}")
        if not (ident.get("extra") or {}).get("container"):
            raise RegistryPlanRefused(f"{p}: route {route!r} lacks extra.container (the attempt-group axis the completion joins on); "
                                      "regenerate the plan set — a container is never guessed from the leg")
    return PlanFile(p, idents)


def iter_plan_files(out) -> Iterable[PlanFile]:
    """Every per-leg plan file under ``out`` (the 405), sorted; the store maps and instruments are not yielded."""
    out = Path(out)
    for leg_dir in sorted(p for p in out.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for f in sorted(leg_dir.glob("*.json")):
            yield load_plan_file(f)


def plan_files_for(out, *, leg: Optional[str] = None, view: Optional[str] = None, arm: Optional[str] = None,
                   s: Optional[int] = None) -> List[PlanFile]:
    """The per-leg plan files matching the given axes (the engine → its runs lookup the fan-out uses)."""
    sel = []
    for pf in iter_plan_files(out):
        if leg is not None and pf.leg != leg:
            continue
        if view is not None and pf.view != view:
            continue
        if arm is not None and pf.arm != arm:
            continue
        if s is not None and pf.s != s:
            continue
        sel.append(pf)
    return sel


def child_env(plan_path) -> Dict[str, str]:
    """The ONE environment variable a driver child exports so its ``CellStore`` joins the planned run(s)."""
    return {RUN_IDENTITY_ENV: str(Path(plan_path).resolve())}


def queued_modal_for_plan(plan_path, *, modal_app_id: Optional[str], modal_call_id: Optional[str], script: str,
                          spool: Optional[str] = None, registry_emit: Optional[bool] = None, session: Optional[str] = None) -> List[str]:
    """``hooks.queued_modal`` once per identity of a plan file (a per-task map queues one run per route), right after the
    launcher's ``.spawn``. The hook is guarded and returns None on a refusal; that is raised here so the
    launcher sees an unregistrable plan instead of a silently missing run."""
    pf = load_plan_file(plan_path)
    out: List[str] = []
    for route, ident in pf.identities.items():
        plan = {k: v for k, v in ident.items() if k != "run_id"}
        paths = hooks.queued_modal(plan, modal_app_id=modal_app_id, modal_call_id=modal_call_id, script=script, spool=spool,
                                   registry_emit=registry_emit, session=session)
        if paths is None:
            raise RegistryPlanRefused(f"{plan_path}: route {route!r} refused by hooks.queued_modal (see the [registry] notice); nothing queued")
        out.extend(paths)
    return out


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="RS/registry/plans")
    ap.add_argument("--w-sha", required=True, help="the W commit (40 hex)")
    ap.add_argument("--campaign", required=True, help="the registry campaign assigned to this wave")
    ap.add_argument("--policy", default=str(DEFAULT_POLICY), help="decode_policy.json export (default: the committed copy)")
    ap.add_argument("--roster", default=str(DEFAULT_ROSTER))
    ap.add_argument("--pins", default=str(DEFAULT_PINS))
    ap.add_argument("--arms", default=str(DEFAULT_ARMS))
    ap.add_argument("--arditi-pins", default=str(DEFAULT_ARDITI_PINS))
    ap.add_argument("--seeds-dir", default=str(DEFAULT_SEEDS_DIR), help="configs/registry (instruments/judges/si_texts seeds)")
    ap.add_argument("--windows", default=None, help='JSON {leg: L} overriding plan §5 windows (the S4-decided multi-turn windows)')
    ap.add_argument("--legs", nargs="*", default=None, help="a subset of legs (default: all 15)")
    ap.add_argument("--check", action="store_true", help="regenerate and refuse any byte drift against --out (writes nothing)")
    a = ap.parse_args(argv)
    windows = json.loads(a.windows) if a.windows else None
    try:
        inp = load_inputs(campaign=a.campaign, w_sha=a.w_sha, policy=Path(a.policy), roster=Path(a.roster), pins=Path(a.pins), arms=Path(a.arms),
                          arditi_pins=Path(a.arditi_pins), seeds_dir=Path(a.seeds_dir), windows=windows)
        ps = build(inp, a.legs)
    except RegistryPlanRefused as exc:
        print(f"registry_plans: REFUSED: {exc}", file=sys.stderr)
        return 2
    if a.check:
        problems = check_plan_set(ps, Path(a.out))
        if problems:
            print("registry_plans --check: DRIFT\n  " + "\n  ".join(problems), file=sys.stderr)
            return 1
        print(f"registry_plans --check: OK ({ps.index['n_plan_files']} plan files, {ps.index['n_store_files']} store maps, "
              f"{ps.index['n_instruments']} instruments, {ps.index['n_run_ids']} run ids)")
        return 0
    written = write_plan_set(ps, Path(a.out))
    print(f"registry_plans: wrote {len(written)} files under {a.out} ({ps.index['n_plan_files']} plan files, {ps.index['n_store_files']} store maps, "
          f"{ps.index['n_instruments']} instruments, {ps.index['n_run_ids']} run ids)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
