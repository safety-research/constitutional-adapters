"""Read out the MATH-500 thinking-ON experiment on the full22 battery and file the verdicts.

Per model (gemma4e4b, qwen35_4b, glm46vflash, nemotron3nano4b): three cells
(bare ``none``, ``prompted__safety_refusal``, the ``positions: all`` LoRA-subtraction
object cell), MATH-500 (500 items, k = 1, thinking ON, one pass at cap 16,384), each
cell under its sampling row's OWN store; the two contrasts ``object - bare`` and
``prompted - bare`` with paired-item bootstrap CIs under both scoring branches and
sharp censoring bounds; the sensitivity rows (``general``, ``battery``) as
their OWN comparisons plus per-cell paired deltas to the primary — never pooled.

    PYTHONPATH=src python scripts/tinycap_math500_thinking.py --gen-root <local copy of
        the wave root> --sha <WAVE_SHA7> [--cells scripts/sdfbase/math500/cells.py]
        [--sampling-rows configs/sdfbase/math500/sampling_rows.yaml] [--render-facts FACTS.json]
        --math500-ids plan|IDS.json [--analysis-sha <sha>] [--wandb-mode online|offline|disabled]
        [--figs-dir DIR] [--out report.json] [--mirror-dir DIR | --no-mirror]

``--sha`` is the STORE pin (the wave sha every cell's producer tree carries) and it is
CHECKED, not copied: a leg whose manifest producer, outcomes record or any recorded run
is not from the pinned tree is refused before any estimand, and a comparison whose legs
come from different trees does not run. The report also records ``analysis_code_sha``,
the git sha of the tree this code ran from.

THE MULTI-TREE ALLOWANCE, a reviewed exception to the one-tree rule and nothing wider:
``--also-sha <sha40>`` (repeatable) declares further producer trees, admitted only when the
local repository (``--repo-root``, default this script's tree) shows every declared tree
differing from the primary in paths matching ``--tree-diff-allow <glob>`` (repeatable)
alone and every declared sha exists — verified at startup, BEFORE any leg is read, and
refused by name otherwise (exit 3, a ``tree_set`` refusal). A leg's producer tree must be
in the declared set (``leg["producer_tree"]``); a comparison's legs must still share ONE
tree; the report's ``tree_set`` block files the trees, the diff per tree (renames OFF, so
a removed path cannot hide behind an allowlisted destination), the allowlist, the per-leg
assignment and the required ``--tree-set-reason``. Without ``--also-sha`` nothing of this
appears: the report, the table and the W&B summary are the one-tree ones.

THE EIGHT-ARM EXTENSION (math-eight-arms-extension, plan P5 / §4.5; lane B B2–B15): the
matrix may carry every arm of the eight-arm panel (:data:`ARMS`: none · prompted safety ·
prompted constitution · LoRA ctl2 · LoRA mix · LoRA mix−ctl · AlphaSteer · Arditi), each
row with a ``status`` (``launch`` is EVIDENCE — the published rows included, their
published-ness being the WAVE; ``complete`` is a tolerated synonym no producer emits;
``held`` / ``deferred`` are design statuses, reported and never collected) and a
``wave``, under WAVE-ISOLATED out roots
(``controls_<wave>/...``). ``--gen-root PATH=PIN`` (repeatable) binds each pulled wave root to
one admitted tree (``--sha`` or an ``--also-sha``); a duplicate logical leg across roots
refuses. ``--tree-diff-allow-file`` reads the REVIEWED allowlist (a glob per line with its
evidence). ``--waves`` names the REQUIRED-EVIDENCE set. Under a declared tree set a contrast
may span admitted trees when the shared-field projection is equal, every leg satisfies ITS
row's arm contract and every tree is admitted (:func:`compatibility_problem`; the record
rides every comparison). Object pins are keyed (model, arm). The report gains ``design``
(``status_by_row``, the eight slots per model), ``eightarm`` (``pooled_by_arm`` — the
equal-weight mean over gemma / qwen / nemotron with ONE shared resampling index vector per
replicate; llama held out, GLM not in the panel), ``gen_roots``, and the two eight-arm figures
ONLY in eight-arm mode (``--waves``, a matrix carrying design statuses / waves, or a pinned /
repeated root) — so the published run reproduces leaf for leaf. ``--figure-exclude-arm`` and
``--figure-modes`` are PRESENTATION selectors (validated against the report; no aggregate is
recomputed; every title carries "N legs in report · k shown"), and the publication gate
(``scripts/tinycap_publication_gate.py``) refuses, before any mirror or W&B call, a render
whose figures present a PSR arm or anything the render declared out (exit 6).

TWO RULES OF THE ADMISSION: (1) the server's default sampling (vLLM's
``--override-generation-config`` echo and the engine's "overridden by
generation_config.json" WARNING) is compared by EFFECT, the way the P11
preflight does — each leg's effective decode is the server default composed with the keys
its requests send (:func:`effective_decode`), and a contrast refuses only when an effective
value differs or a key no request sends differs (:func:`effective_decode_problem`); the
LITERAL launch-flag comparison is kept as a REPORTED field (``server_settings_literal_equal``
with the per-leg override dict), so a fork like the wave tree's Qwen pin stays visible
without refusing a decode it never reached. (2) An ``eos_in_thoughts`` draw on a PRIMARY leg
is masked from every record that consumes the leg and REPORTED — item id, tokens, finish
reason, both readings, ``n_masked`` per leg and per arm, the figure and table footnotes —
ONLY when the gate's own re-run without it moves no verdict word (``word_moves`` false);
a ``lifted_close`` / ``other`` draw, an unclassified one, or a moved word keep failing
closed. The run without eval logs takes the classes from the report it is
required to match (``--require-report-match``; ``class_source`` ``matched_report``).

RULES OF THE RECORD: (H1) the eos-in-thoughts
mask is SCOPED (:data:`MASK_SCOPE_RULE`) — the item leaves the one contrast whose legs it sits on,
the masked leg's own accuracy and its own model's pair inside the pooled arm; every other arm,
contrast and pooled arm keeps its full population, byte-equal to the unmasked reading, and every
consuming record files ``estimands_with_anomalies`` beside. (H2) the server's generation-config
default REFUSES UNLESS UNDERSTOOD (:data:`OVERRIDE_UNDERSTOOD_KEYS`): the five sampling keys by
effect, the output limit (``max_new_tokens`` / ``max_tokens``, which vLLM serves as the minimum
with the client's cap) as the effective cap (:func:`effective_output_limit`), any other key a named
refusal. (M3) the served geometry (L × S) of every leg is disclosed wherever it differs within a
contrast (:func:`served_geometry_notes`, :data:`GEOMETRY_RULE`: a comparison of values, never a
byte claim). (M4) the headline is CONDITIONAL accuracy on scored items with the score-as-is
sensitivity beside it on every record, figure and table (:data:`HEADLINE_RULE`), and a pooled arm
files no directional word (:data:`POOLED_WORD_RULE`). (L6) every figure title is at most two words
with the prose in the subtitle and the caption (:func:`figure_title_problem`), and the masking /
geometry disclosures ride every figure that reads an estimand (:func:`_footnote_lines`).

FURTHER RULES OF THE RECORD: (M-a) the server's
output limit is read by PROVENANCE (:data:`OUTPUT_LIMIT_RULE`, :func:`effective_output_limit`):
vLLM 0.25.1 makes a model file's ``max_new_tokens`` under ``--generation-config auto`` a FALLBACK
an explicit request cap replaces, and only the flag's ``max_new_tokens`` (or the merged limit
under a custom generation-config path) a CLIP — revision 2 read every server limit as a clip and
its regression codified a false refusal on a WARNING-only limit. (M-b) every contrast SURFACES
its denominators (:func:`contrast_denominators`, :func:`pooled_denominators`): the arm's and the
bare anchor's scored counts, at-cap counts and shares, and a censoring differential with its
DIRECTION in words — under every contrast row of the two forest figures, in their captions
and sidecars and in the tables' notes. (M-c) a masked leg's unscored count is its SCOPED
``n_censored`` (the raw summary's beside as ``*_raw_draw``), never a raw count over a scoped
population. (L) the six panels that read the controls table — the raw draw — say that the
eos-in-thoughts mask does not reach them (:data:`RAW_DRAW_PANELS`, :func:`_raw_draw_footnote`).

FOUR CORRECTIONS TO REVISION 3: (M1) the output-limit
source model is EXHAUSTIVE and fails closed (:data:`OUTPUT_LIMIT_STATES`): vLLM 0.25.1 loads NO
file under ``--generation-config vllm`` and prints the WARNING only when the source is not
``vllm``, so a WARNING recorded under ``vllm`` is CONTRADICTORY and refuses (revision 3 read it as
a fallback), and a custom generation-config PATH with neither a WARNING nor a flag naming the
limit is UNKNOWN and refuses (revision 3 read ``merged is None`` as no limit). (M2) at-cap is NOT
censored — a scored draw can finish at the cap — so the censoring differential is now the
UNSCORED-share difference (:data:`CENSORING_DIFFERENTIAL_FIELD`, ``n_censored / n_items`` per leg
over the contrast's population) with its direction, the at-cap quantity is renamed neutrally
(:data:`AT_CAP_DIFFERENTIAL_FIELD`, ``Δ at-cap``, "more often at cap") and rendered BESIDE it, and
the branch-shift rule is the exact decomposition (:data:`CENSORING_DIFFERENTIAL_RULE`: score-as-is
= conditional × scored share per leg; the shift = bare conditional × bare unscored share − arm
conditional × arm unscored share) with an identity check on every contrast; revision 3's at-cap
direction had read gemma prompted − bare (0 unscored on both) as "bare more censored" and qwen
LoRA mix (34 vs 71 unscored, the anchor more censored by 7.4 pp) as "arm more censored". (M3) a
pooled row prints the per-model arm/bare scored triples beside the equal-weight aggregates
(:func:`_pooled_denominator_lines`). (L1) the record's §15 glyph text corrected (``n`` / ``u``;
eight matrix rows exist, ``not_analysed``).

Every leg's store is ``<gen-root>/controls/<model_key>/<cell slug>/<sampling_mode>/cells/<key>/``
(the slug is the cell id with ``__`` -> ``--``), with the usage
ledger, the request evidence and ``evidence.json`` at that out root. The SUMMARY is
authoritative for every rate and the ``outcomes/<task>.json`` record for every mask
(verified against the summary before any statistic); ``rows.jsonl``, the ``.eval``
logs, the ledger, the request evidence and ``evidence.json`` are cross-checks and
gates, never denominators. No number is called a result before every gate
passes (:func:`all_gates`), and the production boundary of :func:`main` fails
CLOSED before any figure, mirror or W&B call (:func:`production_refusals`).

The reader contract — the manifest / evidence / request-evidence keys this script
reads and the driver writes — is the constants block below; the test
builder ``tests/math500_synthetic_store.py`` is pinned equal to it by test.

The ledger gate no longer refuses a sample the harness re-sent at the same
cap. Every call beyond the first per (sample, recorded run) is a recorded resume,
the harness's same-cap re-send within its retry ladder (:data:`LADDER_TEXT`; the
constants are read from ``openai``, ``contracts`` and ``inspect_served``'s own
source) or an UNEXPLAINED duplicate, which still fails; the per-leg ``redraws``
readout rides beside the at-cap share with the selection caveat. The log-reading
run cross-checks the ledger's calls per sample against the ``.eval`` log's model
events and failed attempts and refuses the leg on disagreement; the run without
logs classifies from the ledger alone, and the two reports still match on every
numeric leaf (the cross-check's numbers live under ``eval_check``, outside the
match by the parent's rule). The per-item log check compares every
outcome-defining field, not only the score.

Machinery reused from ``scripts/tinycap_oos_transfer.py`` by IMPORT (never edited):
the store readers and cross-checks (the ``.eval`` re-derivation's comparison, over
the log the PRODUCER's own rule selects — :func:`eval_cross_check` here, since the
parent's reads every log in the dir), the branch masks, the exact McNemar sensitivity
with its mask-dependent one-sided note, the transitions, the analysis-code
provenance, the numeric-leaf report comparison, the figure palette and savers, the
mirror. The paired bootstrap is :func:`paired_bootstrap_conditional`, the
generalisation of ``pairing.paired_bootstrap_conditional_2x2`` to any legs and any
signed contrasts (same index draws, same exact-rational arithmetic; the test pins it
bit for bit to the 2x2 on a four-leg input), so ``pairing.py`` stays untouched.
"""
from __future__ import annotations

import argparse
import bisect
import fnmatch
import hashlib
import importlib.util
import inspect as _inspect
import ast
import json
import re
import subprocess
import sys
import zlib
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any, NamedTuple

import numpy as np
import openai

from sisteer.capeval.serving import serving_attestation as SA   # the NATIVE launch record's strict reader (stdlib-only leaf)
from sisteer.capeval.tiers import inspect_served as isv
from sisteer.capeval.tiers import pairing as pr
from sisteer.capeval.tinycap import client_lock as CL   # the client lock compared by what the client IMPORTS (disclosure)
from sisteer.capeval.tinycap import completion
from sisteer.capeval.tinycap import contracts as C
from sisteer.capeval.tinycap import paper_rows as PRW
from sisteer.capeval.tinycap import refusal_shaped as RFS   # `RS` is this module's resolved-sampler recorder alias below — never reuse it
from sisteer.capeval.tinycap.completion import CompletionError

_HERE = Path(__file__).resolve().parent
#: the analysis checkout's root (scripts/ -> the repo): where `client_lock_disclosure` reads each producer tree's lock text
#: (`git show`) and walks the client's imports when no root is handed in
REPO_ROOT = _HERE.parent


def _load_module(name: str, path: Path) -> Any:
    """Import a module by file path, registered in ``sys.modules`` BEFORE it
    executes: a dataclass defined in it resolves its class module through
    ``sys.modules`` (unregistered -> None -> crash)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


T = _load_module("tinycap_oos_transfer", _HERE / "tinycap_oos_transfer.py")
#: the publication gate shared by the math instruments (plan P5 / §4.7): no figure presents a PSR arm
PG = _load_module("tinycap_publication_gate", _HERE / "tinycap_publication_gate.py")
#: the resolved-sampler recorder — the producer's OWN reader of a vLLM server log (the generation-config override WARNING, the
#: request-logging attestation); the launch attestation reads the same log through it, never a re-typed regex
RS = _load_module("record_resolved_sampler", _HERE / "sdfbase" / "record_resolved_sampler.py")

#: the parent's exception, aliased: every refusal here and every reused reader raises ONE type
AnalysisError = T.TransferAnalysisError

# --------------------------------------------------------------------------- #
# the experiment's vocabulary                                                 #
# --------------------------------------------------------------------------- #

EXPERIMENT = "math500-thinking-on-full22"
TASK: str = C.inspect_task("math")
TASK_NAME: str = TASK.removeprefix(C.INSPECT_TASK_PREFIX)     # the runner's task name (its PRIMARY_SCORER key)
N_ITEMS = 500
CAP = C.MATH500_MAX_TOKENS_DEFAULT       # the one-pass cap (16,384), read from the contracts — never a literal
MODE_PRIMARY, MODE_GENERAL, MODE_BATTERY = "thinking_card", "general", "battery"
#: math500-steer: the NO-SWITCH family's shipped sampling row (the launcher's `SAMPLING_CARD`: the MODE a family with no thinking
#: switch runs — llama-paper-rows P18; the GSM8K validators' `SAMPLING_MODE_BY_ARM`). Its PRIMARY row is `card`, never the
#: thinking card it cannot run; `PRIMARY_MODE_BY_ARM` keys the reader's primary-row rule on the row's ARM (:func:`row_arm`)
MODE_CARD = "card"
PRIMARY_MODE_BY_ARM: Mapping[str, str] = {C.THINKING_ON: MODE_PRIMARY, C.THINKING_NOT_AVAILABLE: MODE_CARD}
SENSITIVITY_MODES: tuple[str, ...] = (MODE_GENERAL, MODE_BATTERY)
CELL_BARE: str = C.ROUTE_NONE_CELL
CELL_PROMPTED: str = C.CELL_PROMPTED_SAFETY
CELL_OBJECT = "constv2-mix-full__lora_r4_ce_minus_ctl"
# -- the EIGHT-ARM extension: the panel's five further arms. The three published
#    cells keep their published kind and contrast names (the published report's keys); the panel ORDER restricted to them is
#    (bare, prompted, object), so every figure that walks "the arms present, in panel order" draws the published records in
#    the published order. `arditi` is the plan's PROVISIONAL spelling (no cell grammar on this tree admits it): it is a
#    design-only arm (a `deferred` / `held` row, never collected) until that spelling lands. --
CELL_PROMPTED_CONST = "prompted__si_constitution"
CELL_LORA_CTL2 = "constv2ctl2-full__lora_r4_ce"
CELL_LORA_MIX = "constv2-mix-full__lora_r4_ce"
CELL_ALPHASTEER = "alphasteer"
CELL_ARDITI = "arditi"
#: arditi-allpos-cap (2026-09-17): the all-positions Arditi refusal direction on qwen35_4b — a `vector_all` object dosed as a RAW
#: COEFFICIENT (c = 0.3, a declared user override of the selection rule) and served on the graph-captured NATIVE residual slot.
#: NOT `arditi`: that arm is the PREFILL-ONLY direction dosed as an alpha (a residual-norm fraction on a `static_r` object) — a
#: different intervention in a different unit, never pooled with this one, never relabelled into its slot or paper key.
#: Appended AFTER the published eight so the panel order is untouched; a figure's geometry derives from the arms the REPORT
#: registers (:func:`panel_slots`), so a panel without this arm renders exactly as before.
CELL_ARDITI_ALLPOS = "arditi_allpos"
KIND_BARE, KIND_PROMPTED, KIND_OBJECT = "bare", "prompted", "object"
KIND_PROMPTED_CONST, KIND_LORA_CTL2, KIND_LORA_MIX, KIND_ALPHASTEER, KIND_ARDITI = (
    "prompted_constitution", "lora_ctl2", "lora_mix", "alphasteer", "arditi")
KIND_ARDITI_ALLPOS = "arditi_allpos"
#: the three route classes by the cell grammar. The fitted class is STEERED — a lens hook OR a
#: native LoRA adapter: one class, two serving paths
#: (:func:`steered_path_of`). The value moved from "lens" to "steered" in NEW analysis outputs
#: only; no eval store records it.
ROUTE_BASE, ROUTE_PROMPTED, ROUTE_STEERED = "base", "prompted", "steered"


class Arm(NamedTuple):
    """One arm of the eight-arm panel: its TrainTest / matrix cell id, its kind (the report's key for it), its figure
    label, and whether it is served from an EXTERNAL object bank (the steering objects, bound by `object_source`
    rather than the full22 pins). Its route class (the per-row contract of B5 (i′)) is the cell grammar's
    (:func:`route_class_of`), derived, never stored."""
    cell: str
    kind: str
    label: str
    external: bool = False
    #: whether the arm's rows are KEYED by their ``view`` (:func:`key_view`). False on the published external arms, whose
    #: ``view`` field names the object bank's directory and whose panel slot has no main-view twin; True on an external arm
    #: whose row genuinely sits on a paper view (the all-positions Arditi arm's `it` row, its bank view carried apart as
    #: ``bank_view``) — keyed on the ARM, never a blanket "external -> main" rule.
    view_keyed: bool = False


#: the panel in its FIXED order (plan P5): none · prompted safety · prompted constitution · LoRA ctl2 · LoRA mix ·
#: LoRA mix−ctl · AlphaSteer · Arditi — then, appended (arditi-allpos-cap), the all-positions Arditi arm
ARMS: tuple[Arm, ...] = (
    Arm(CELL_BARE, KIND_BARE, "bare (none)"),
    Arm(CELL_PROMPTED, KIND_PROMPTED, "prompted (safety_refusal)"),
    Arm(CELL_PROMPTED_CONST, KIND_PROMPTED_CONST, "prompted (constitution)"),
    Arm(CELL_LORA_CTL2, KIND_LORA_CTL2, "LoRA ctl2 (control object)"),
    Arm(CELL_LORA_MIX, KIND_LORA_MIX, "LoRA mix (constitution object)"),
    Arm(CELL_OBJECT, KIND_OBJECT, "object (LoRA-subtraction)"),
    Arm(CELL_ALPHASTEER, KIND_ALPHASTEER, "AlphaSteer (selected rung)", external=True),
    Arm(CELL_ARDITI, KIND_ARDITI, "Arditi refusal (selected rung)", external=True),
    Arm(CELL_ARDITI_ALLPOS, KIND_ARDITI_ALLPOS, "Arditi all-positions (declared override)", external=True, view_keyed=True),
)
#: THE DECLARED PANEL ROSTER: the published eight-arm panel, named CELL BY CELL — never ``len(ARMS)`` and never a slice of the
#: registry, so registering a further arm cannot widen a published panel (the defect this closes). An analysis's design covers
#: the roster PLUS any arm its analysed rows carry (:func:`design_arm_roster`): a document of three it-doc rows still shows the
#: roster's slots with the missing ones ``absent`` — ``absent`` is a statement about the panel, not an artefact of the registry —
#: while a matrix that genuinely carries a further arm shows that arm too, and an arm in neither appears nowhere.
PANEL_ROSTER_CELLS: tuple[str, ...] = (CELL_BARE, CELL_PROMPTED, CELL_PROMPTED_CONST, CELL_LORA_CTL2, CELL_LORA_MIX, CELL_OBJECT,
                                       CELL_ALPHASTEER, CELL_ARDITI)
PUBLISHED_ARMS: tuple[Arm, ...] = tuple(a for a in ARMS if a.cell in PANEL_ROSTER_CELLS)
ARM_BY_CELL: Mapping[str, Arm] = {a.cell: a for a in ARMS}


def arms_of_cells(cells: Any) -> list[Arm]:
    """The registered arms among ``cells``, in PANEL (registry) order. THE DESIGN IS THE MATRIX'S ARMS: an analysis covers the
    arms its analysed rows carry, never whatever happens to be registered in this module — so registering a further arm cannot
    widen a panel whose matrix does not carry it (its slots, its caption numeral and its pooled block all read this), and the
    published matrix keeps its published design byte for byte. An unregistered cell contributes nothing here; the matrix loader
    refuses it separately."""
    want = {str(c) for c in (cells or ())}
    return [a for a in ARMS if a.cell in want]


def design_arm_roster(cells: Any) -> list[Arm]:
    """The arms an analysis's DESIGN covers, in panel order: the declared roster (:data:`PANEL_ROSTER_CELLS`) plus any registered
    arm the analysed ``cells`` actually carry. A roster arm no row carries keeps its slot and reads ``absent``; a registered arm
    in neither the roster nor the rows is not in this panel at all."""
    want = {str(c) for c in (cells or ())} | set(PANEL_ROSTER_CELLS)
    return [a for a in ARMS if a.cell in want]


def design_arms(design: Any) -> list[Arm]:
    """The arms a written design covers (:func:`arms_of_cells` of ``design.arms``) — the registry only when a design names
    none (no design on record), so a stored report always lays out the arms it was written with."""
    named = _mapping(design).get("arms") if isinstance(design, Mapping) else None
    return arms_of_cells(named) if named else list(ARMS)
ARM_BY_KIND: Mapping[str, Arm] = {a.kind: a for a in ARMS}
#: every registered arm's kind, keyed by cell (the reader admits exactly these cells)
CELL_KIND: Mapping[str, str] = {a.cell: a.kind for a in ARMS}
ARM_KINDS: tuple[str, ...] = tuple(a.kind for a in ARMS)
#: the PUBLISHED design's three cells and kinds (the at-cap premise verdict is held to this designed set — unchanged)
DESIGN_CELLS: tuple[str, ...] = (CELL_BARE, CELL_PROMPTED, CELL_OBJECT)
KINDS: tuple[str, ...] = (KIND_BARE, KIND_PROMPTED, KIND_OBJECT)
CONTRAST_OBJECT, CONTRAST_PROMPTED = "object_minus_bare", "prompted_minus_bare"
#: the two contrasts of the published three-cell comparison as SIGNED leg combinations, in their published order
CONTRASTS: Mapping[str, Mapping[str, int]] = {CONTRAST_OBJECT: {KIND_OBJECT: +1, KIND_BARE: -1},
                                              CONTRAST_PROMPTED: {KIND_PROMPTED: +1, KIND_BARE: -1}}
_LEGACY_CONTRAST_NAME: Mapping[str, str] = {KIND_OBJECT: CONTRAST_OBJECT, KIND_PROMPTED: CONTRAST_PROMPTED}


def contrast_name(kind: str) -> str:
    """The report key of ``<kind> - bare``: the published names for the published kinds, ``<kind>_minus_bare`` otherwise."""
    return _LEGACY_CONTRAST_NAME.get(kind, f"{kind}_minus_bare")


def contrasts_for(kinds: Sequence[str]) -> dict[str, dict[str, int]]:
    """The signed contrasts of a comparison holding ``kinds`` (bare among them): one ``<kind> - bare`` per other kind.
    Exactly the published design triple keeps the published ORDER (:data:`CONTRASTS`); a wider panel runs in panel order."""
    present = set(kinds)
    if KIND_BARE not in present:
        raise AnalysisError(f"a comparison needs the bare cell among its kinds, got {sorted(present)}")
    if present == set(KINDS):
        return {k: dict(v) for k, v in CONTRASTS.items()}
    return {contrast_name(k): {k: +1, KIND_BARE: -1} for k in ARM_KINDS if k in present and k != KIND_BARE}


def comp_contrasts(comp: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    """A comparison record's contrasts, DERIVED from the kinds its ``leg_keys`` name (never stored: no numeric leaf)."""
    return contrasts_for(list(comp["leg_keys"]))


def plus_kind(signs: Mapping[str, int]) -> str:
    """The arm a signed contrast ADDS (``+1``): the ``<kind>`` of ``<kind> - bare``."""
    return next(k for k, s in signs.items() if s > 0)


#: the matrix row STATUSES (plan P5). The launcher's vocabulary (`cells.STATUSES`) is `launch` / `held` /
#: `deferred`, and EVERY published row is `launch` too ("complete" is a property of the WAVE —
#: `cells.COMPLETED_WAVES` — never a status). `launch` rows are the EVIDENCE — required whenever their wave is analysed;
#: `complete` is TOLERATED as a synonym (a matrix document stamped that way still reads; no producer on this tree emits
#: it); `held` and `deferred` rows are design statuses — reported under `design.status_by_row`, never collected
STATUS_COMPLETE, STATUS_LAUNCH, STATUS_HELD, STATUS_DEFERRED = "complete", "launch", "held", "deferred"
ROW_STATUSES: tuple[str, ...] = (STATUS_COMPLETE, STATUS_LAUNCH, STATUS_HELD, STATUS_DEFERRED)
EVIDENCE_STATUSES: tuple[str, ...] = (STATUS_COMPLETE, STATUS_LAUNCH)
#: the wave-isolated out-root prefixes (plan P2 / F11a): `controls/` for the published rows, `controls_<wave>/` for a wave
OUT_ROOT_PREFIX = "controls"
#: the wave a row without a `wave` field belongs to: the published run
WAVE_PUBLISHED = "published"
#: the pooled panel (plan P5): equal weights over these three; llama is HELD OUT (thinking: not available) and GLM
#: is not in the panel (appendix tables)
POOLED_PANEL_MODELS: tuple[str, ...] = ("gemma4e4b", "qwen35_4b", "nemotron3nano4b")
POOLED_HELD_OUT: Mapping[str, str] = {"llama8b": "thinking: not available — a different instrument (no thinking switch); reported "
                                                 "individually, held out of every pooled mean"}
POOLED_NOT_IN_PANEL: tuple[str, ...] = ("glm46vflash",)
POOLED_MODE: str = "thinking_card"
#: the trigger provenance sentence MATH-500 reports carry (GSM8K's is `inspect_rung.GSM8K_TRIGGER_PROVENANCE`)
M5T_TRIGGER_PROVENANCE = ("MATH-500: one pass at 16,384 (the ceiling, D68), no ladder and no rung above; the 1 % RAW at-cap "
                          "line is filed as a report field per cell (harmonised across the battery by user decision 2026-09-12)")
BRANCH_PRIMARY, BRANCH_SCORE_AS_IS, BRANCHES, BRANCH_NAMES = (T.BRANCH_PRIMARY, T.BRANCH_SCORE_AS_IS,
                                                              T.BRANCHES, T.BRANCH_NAMES)
GATE_PASS, GATE_FAIL, GATE_PENDING = T.GATE_PASS, T.GATE_FAIL, T.GATE_PENDING
GATE_NAMES: tuple[str, ...] = ("population", "ledger", "identity", "certification", "cap",
                               "split_anomalies", "scoring_path")
STAT_FIELDS: tuple[str, ...] = ("point", "ci_low", "ci_high", "bound_low", "bound_high")
#: the verdict words: a direction ONLY when every condition holds
WORD_LOWER = "lower than bare"
WORD_HIGHER = "higher than bare"
WORD_NOT_RESOLVED = "not resolved at this cap"
WORDS: tuple[str, ...] = (WORD_LOWER, WORD_HIGHER)
NOT_MEASURABLE = T.NOT_MEASURABLE
CAP_GATE_AT_CAP_RATE = T.CAP_GATE_AT_CAP_RATE
# -- the at-cap READOUT: one pass at the ceiling, the residual at-cap share a FIRST-CLASS result per cell
#    against the harness rule's 1 % line, and a surfacing threshold DECLARED before the number existed — twice the
#    ruling's premise, which was a GSM8K measurement, never a MATH-500 one. Run-health numbers beside the cap gate
#    (which stays the parent's re-cost rule), never a substitute for it. --
AT_CAP_LINE_1PCT = 0.01
AT_CAP_SURFACING_THRESHOLD = 0.03
AT_CAP_PREMISE_RATE = 0.015
AT_CAP_PREMISE_SOURCE = "GSM8K — the ruling's premise (D65), not a MATH-500 measurement (D68)"
#: the standing truncation rule's CEILING (contracts: 16,384 is the ceiling — "the residual at-cap share at the ceiling is
#: reported, there is no rung above"). At the ceiling the parent's 2 % re-cost gate is ADVISORY (reported per cell,
#: never a refusal — refusing would suppress exactly the reading the rule asks for); below it the re-cost refusal stands.
AT_CAP_CEILING = C.MATH500_MAX_TOKENS_DEFAULT
# -- the harness's RETRY LADDER: every same-cap re-send the usage ledger can carry beyond one call per sample
#    per recorded run comes from three retry layers wrapped around one served call, each configured to "2 retries",
#    and NOTHING else in the harness re-sends a task call at the same cap:
#      (1) the openai SDK's own request retry — the provider is built with no `max_retries`, so the SDK default
#          (`openai.DEFAULT_MAX_RETRIES`) applies; a request the SDK gives up on at its default read timeout
#          (600 s — the provider passes no `client_timeout`, so `contracts.INSPECT_TIMEOUT_S` bounds the retry LOOP,
#          never one request) is re-sent by the SDK while the upstream call runs to completion and the ledger records
#          it HTTP 200 — a completion nobody scored;
#      (2) Inspect's model-call retry (`contracts.INSPECT_MAX_RETRIES`, the eval's `max_retries`): a `ModelEvent`
#          whose `retries` counts them, an event that exhausts them carries `error`;
#      (3) Inspect's sample re-run (`retry_on_error`, the literal in `inspect_served.run_one_task`'s `eval_async`
#          call, read from its source below — the harness names no constant for it): the sample's `error_retries`
#          lists each failed attempt, whose events the log drops.
#    Measured on the wave's real logs (qwen object cell: 500 samples, 7 model-level retries, 0 failed attempts; the
#    partial GLM battery cell: 462 samples, 7 with one failed attempt): the ledger's calls per sample equal
#    sum(1 + retries) over the final attempt's model events plus LADDER_CALLS_PER_ATTEMPT per failed attempt, on
#    every sample; the observed maximum per attempt is 9 = 3 x 3 and the observed maximum per sample 13 < 27.
#    The neither-channel refusal (PB8) is a DIFFERENT path: it raises the sample's error outright and the ledger
#    shows one call per attempt for it — it rides layer (3) only. --
SDK_MAX_RETRIES: int = int(openai.DEFAULT_MAX_RETRIES)
INSPECT_MAX_RETRIES: int = int(C.INSPECT_MAX_RETRIES)


def _harness_retry_on_error() -> int:
    """The sample re-run count the harness hands ``eval_async`` (``retry_on_error=<n>``
    in :func:`inspect_served.run_one_task`), read from the harness's own source —
    exactly one literal, or the analysis refuses to start (a ladder bound typed
    here would drift from the harness silently)."""
    hits = re.findall(r"retry_on_error\s*=\s*(\d+)\s*,", _inspect.getsource(isv.run_one_task))
    if len(hits) != 1:
        raise AnalysisError(f"inspect_served.run_one_task names retry_on_error {len(hits)} time(s) — the ladder's sample "
                            "re-run count cannot be read")
    return int(hits[0])


HARNESS_RETRY_ON_ERROR: int = _harness_retry_on_error()
#: the MOST calls one sample ATTEMPT can make: the SDK's request retries inside Inspect's model-call retries. An UPPER
#: bound, exact only for an attempt that exhausted the timeout ladder (every call timed out); an attempt the provider
#: refused after ONE completed call (the neither-channel refusal, a stamp mismatch — `inspect_served`'s
#: `InspectServedError`, raised outside the SDK's send and never retried) made exactly 1
LADDER_CALLS_PER_ATTEMPT: int = (1 + SDK_MAX_RETRIES) * (1 + INSPECT_MAX_RETRIES)
#: calls one sample can make in ONE recorded run of the store: the attempt ladder times the sample re-runs
LADDER_CALLS_PER_SAMPLE_RUN: int = LADDER_CALLS_PER_ATTEMPT * (1 + HARNESS_RETRY_ON_ERROR)
LADDER_TEXT = (f"per sample per recorded run at most (1 + {SDK_MAX_RETRIES} SDK request retries) x (1 + {INSPECT_MAX_RETRIES} "
               f"Inspect model-call retries) x (1 + {HARNESS_RETRY_ON_ERROR} Inspect sample re-runs) = "
               f"{LADDER_CALLS_PER_SAMPLE_RUN} calls; one attempt at most {LADDER_CALLS_PER_ATTEMPT}")
#: how a call beyond the first per (sample, recorded run) is classified by the ledger gate
REDRAW_RECORDED_RESUME, REDRAW_HARNESS_RESEND, REDRAW_UNEXPLAINED = "recorded_resume", "harness_resend", "unexplained"
REDRAW_MECHANISM_TEXT = (
    "a same-cap re-send: the same sample, the SAME body_sha256 as the sample's first call (a changed prompt, seed or "
    "sampling under one sample label is another experiment), the same max_tokens_requested, HTTP 200, not rejected, no "
    "error class or code, within the retry ladder — the harness's client retry after a request it gave up on (the openai "
    "SDK's default 600 s read timeout; the upstream completion still ran to the end and the ledger records it 200), then "
    "Inspect's model-call retry and sample re-run. A call with another body, at another cap, a rejected or non-200 call, an "
    "errored call, or a count above the ladder is UNEXPLAINED and fails the gate")
#: how the FINAL draw of a re-sent sample is picked from the ledger (the row whose finish reason is held to the summary's
#: n_at_cap): the usage ledger stamps `ts` when the request ARRIVES and appends the row when the upstream reply
#: COMPLETES, so an abandoned at-cap request that finishes after its replacement lands LAST in the file — the file's last
#: row is the wrong draw (the wave's GLM object cells, partial: 19 of 462 and 28 of 387 file-last rows disagreed with the
#: log's at-cap bit; the latest-start row agreed on every one). The client retries SEQUENTIALLY, so the replacement
#: always starts after the request it replaces: the latest request start is the final draw. The .eval log carries no
#: ledger request id (its `extra_headers` hold Inspect's own `x-irid` and the task / sample labels; the ledger's
#: `X-Tinycap-Request-Id` is minted at send time and not logged — verified), so selection by identity is unavailable;
#: the log-reading run holds the selected row's finish reason to the log's at-cap bit, sample by sample, instead
FINAL_DRAW_RULE_TEXT = ("per sample, the forwarded row with the latest request-start ts (the client's sequential retry: a "
                        "replacement starts after the request it replaces), never the file's last row (rows are appended on "
                        "completion, so an abandoned request that finishes after its replacement lands last)")
#: a failed sample attempt's class from its `error_retries` message. The log DROPS the failed attempt's model events (its
#: `error_retries[i].events` tail holds the attempt's LAST model event and what followed — verified on the wave's logs), so
#: the attempt's HTTP-call count is inferred from the class, and only ONE class is exact:
#:   - the exhausted TIMEOUT LADDER (a tenacity RetryError over the SDK's APITimeoutError): exactly LADDER_CALLS_PER_ATTEMPT.
#:     Nine is exact only because the error type is the ladder's own — the SDK raises APITimeoutError only after its own
#:     request retries were exhausted, and Inspect's RetryError only after its model-call retries were: every call timed out;
#:   - a PRE-WIRE provider refusal (`inspect_served`'s InspectServedError raised BEFORE `super().generate()` sends anything:
#:     `collapse_system_messages`, `tinycap_config` / `sampling_params`): exactly 0 calls;
#:   - a POST-RESPONSE provider refusal (`on_response`'s stamp / regime refusal, `assert_channels`' neither-channel refusal):
#:     raised AFTER `super().generate()` returned, so the SDK's request retries and up to INSPECT_MAX_RETRIES earlier
#:     timed-out model calls of the same attempt can precede it (a request times out, the SDK re-sends, the re-send returns
#:     an empty 200, the provider refuses it): BOUNDED in [1, LADDER_CALLS_PER_ATTEMPT], never 1 exactly;
#:   - a `tinycap provider:` message this analysis cannot place (raised on both sides — the unmapped-task refusal), or
#:     any other error type: UNVERIFIABLE, bounded [0, LADDER_CALLS_PER_ATTEMPT].
#: A bounded attempt's item is held to its bound and NAMED (`unverifiable_items`); the maximum is never substituted.
FAILED_ATTEMPT_TIMEOUT_LADDER = "timeout_ladder"
FAILED_ATTEMPT_PROVIDER_REFUSAL_PRE_WIRE = "provider_refusal_pre_wire"
FAILED_ATTEMPT_PROVIDER_REFUSAL_POST_RESPONSE = "provider_refusal_post_response"
FAILED_ATTEMPT_PROVIDER_REFUSAL_UNPLACED = "provider_refusal_unplaced"
FAILED_ATTEMPT_UNVERIFIABLE = "unverifiable"
FAILED_ATTEMPT_CLASSES: tuple[str, ...] = (FAILED_ATTEMPT_TIMEOUT_LADDER, FAILED_ATTEMPT_PROVIDER_REFUSAL_PRE_WIRE,
                                          FAILED_ATTEMPT_PROVIDER_REFUSAL_POST_RESPONSE, FAILED_ATTEMPT_PROVIDER_REFUSAL_UNPLACED,
                                          FAILED_ATTEMPT_UNVERIFIABLE)
SDK_TIMEOUT_ERROR_NAME: str = openai.APITimeoutError.__name__          # what a RetryError over the timeout ladder names
INSPECT_TIMEOUT_MESSAGE = "Request timed out."                          # Inspect's exception_message of that error
PROVIDER_REFUSAL_MARKER = "tinycap provider:"                          # the prefix of every InspectServedError the provider raises
#: substrings of the provider's refusal messages raised BEFORE any request (each pinned to the harness's source by test):
#: `collapse_system_messages` (the first three) and `tinycap_config` / `sampling_params` (the rest)
PROVIDER_PRE_WIRE_MARKERS: tuple[str, ...] = (
    "empty message list", "system message at position", "does not start with the SI bytes", "num_choices=",
    "extra_body carries", "extra_body sets stream", "a caller set", "refusing a thinking-ON lens-route request")
#: substrings of the refusals raised AFTER a response: `assert_channels` and `on_response` (whose `where` prefix reads
#: "... sample <idx> on '<virtual model>': ...")
PROVIDER_POST_RESPONSE_MARKERS: tuple[str, ...] = ("neither content nor reasoning", "regime stamp")
_PROVIDER_ON_RESPONSE_WHERE = re.compile(r"sample \d+ on '")


def classify_failed_attempt(message: Any) -> str:
    """One ``error_retries`` entry's class from its message (the classes and
    what each says about the attempt's calls: :data:`FAILED_ATTEMPT_CLASSES`)."""
    text = str(message or "")
    if SDK_TIMEOUT_ERROR_NAME in text or INSPECT_TIMEOUT_MESSAGE in text:
        return FAILED_ATTEMPT_TIMEOUT_LADDER
    if PROVIDER_REFUSAL_MARKER not in text:
        return FAILED_ATTEMPT_UNVERIFIABLE
    if any(m in text for m in PROVIDER_POST_RESPONSE_MARKERS) or _PROVIDER_ON_RESPONSE_WHERE.search(text):
        return FAILED_ATTEMPT_PROVIDER_REFUSAL_POST_RESPONSE
    if any(m in text for m in PROVIDER_PRE_WIRE_MARKERS):
        return FAILED_ATTEMPT_PROVIDER_REFUSAL_PRE_WIRE
    return FAILED_ATTEMPT_PROVIDER_REFUSAL_UNPLACED


def failed_attempt_bounds(cls: str) -> tuple[int, int]:
    """The (least, most) HTTP calls a failed attempt of class ``cls`` made;
    equal only for the classes whose count is established."""
    if cls == FAILED_ATTEMPT_TIMEOUT_LADDER:
        return LADDER_CALLS_PER_ATTEMPT, LADDER_CALLS_PER_ATTEMPT
    if cls == FAILED_ATTEMPT_PROVIDER_REFUSAL_PRE_WIRE:
        return 0, 0
    if cls == FAILED_ATTEMPT_PROVIDER_REFUSAL_POST_RESPONSE:
        return 1, LADDER_CALLS_PER_ATTEMPT
    return 0, LADDER_CALLS_PER_ATTEMPT


def failed_attempt_detail(retry_error: Any) -> dict[str, Any]:
    """One ``error_retries`` entry as the cross-check reads it: its class, the
    call bounds, and the retained tail's last model event (``retries``, its
    error) — evidence that can CONTRADICT an exact class (a timeout-ladder
    failure whose last model call did not exhaust the SDK's retries is not the
    ladder, and is downgraded to an unverifiable bound) but never establishes
    a bounded one (the attempt's earlier model events are gone)."""
    cls = classify_failed_attempt(getattr(retry_error, "message", None))
    tail_events = [e for e in (getattr(retry_error, "events", None) or ()) if getattr(e, "event", None) == "model"]
    tail = tail_events[-1] if tail_events else None
    tail_retries = int(getattr(tail, "retries", None) or 0) if tail is not None else None
    tail_error = str(getattr(tail, "error", None) or "") if tail is not None else None
    note = None
    if cls == FAILED_ATTEMPT_TIMEOUT_LADDER and tail is not None and (
            tail_retries != SDK_MAX_RETRIES or INSPECT_TIMEOUT_MESSAGE not in tail_error):
        note = (f"the retained tail's last model event ({tail_retries} retries, error {tail_error[:40]!r}) is not an exhausted "
                "timeout ladder — the exact count is not established")
        cls = FAILED_ATTEMPT_UNVERIFIABLE
    low, high = failed_attempt_bounds(cls)
    return {"class": cls, "low": low, "high": high, "tail_retries": tail_retries, "note": note}
REDRAW_SELECTION_CAVEAT = (
    "the store's at-cap share is over FINAL draws: a sample whose call the client gave up on was drawn again, so the "
    "scored draw is the one that finished inside the client's request timeout — a selection toward faster draws (long "
    "at-cap decodes and slow finished ones alike time out first). Requests per item, the share of items re-drawn and the "
    "items with a failed sample attempt are reported beside the at-cap share from the usage ledger; the ledger's earlier "
    "calls are completions nobody scored")
#: the outcome-defining producer facts the per-item log cross-check compares draw by draw: the
#: stop reason and the at-cap bit the masks read, the two token counts the truncation readout reads, the think-block
#: facts the primary policy reads — a count-preserving swap between two items survives every marginal and the parent's
#: score comparison, and fails here by name
OUTCOME_FIELDS_FROM_LOG: tuple[str, ...] = ("stop_reason", "at_cap", "completion_tokens", "prompt_tokens", "think_block_present",
                                            "think_block_unterminated", "stray_open_tag", "content_empty_with_reasoning")
#: the DESIGNED primary set — these four models x {bare, prompted, object} under the primary sampling row, 12 cells — is
#: the completeness the premise verdict is held to whatever matrix `--cells` names (pinned equal to the cells module by
#: test): a reduced matrix leaves the premise UNDETERMINED, never holding
DESIGN_MODEL_KEYS: tuple[str, ...] = ("gemma4e4b", "qwen35_4b", "glm46vflash", "nemotron3nano4b")
N_PRIMARY_CELLS_DESIGN = len(DESIGN_MODEL_KEYS) * len(KINDS)
# -- the loop rule over an at-cap draw's tail (at-cap rows classified truncated-reasoning vs degenerate-loop). A
#    CONSERVATIVE rule, stated once in the report: a naive line-repeat rule mislabelled structured compliant text
#    ("Option 1 / Option 2 / Option 3") as loops, so this one asks for an EXACTLY repeated block covering most of the
#    tail; every other at-cap draw is truncated reasoning. Compression is recorded as a diagnostic and never decides: a
#    progressing enumeration ("For n = 1, ... For n = 50, ...") compresses like a loop without being one. --
LOOP_TAIL_CHARS = 1500
LOOP_PERIOD_MIN, LOOP_PERIOD_MAX = 8, 400
LOOP_MIN_REPEATS = 4
LOOP_MIN_COVER = 0.60
LOOP_RULE_REPEATED_BLOCK, LOOP_RULE_NONE = "repeated_block", "none"
LOOP_RULE_TEXT = (f"over the last {LOOP_TAIL_CHARS} characters of the emitted text (the reasoning channel, then the content): a "
                  f"degenerate loop iff the tail ENDS in a block of period p, {LOOP_PERIOD_MIN} <= p <= {LOOP_PERIOD_MAX} "
                  f"characters, repeated >= {LOOP_MIN_REPEATS} times consecutively and covering >= {LOOP_MIN_COVER:.0%} of the "
                  "tail; every other at-cap draw is truncated reasoning. The tail's zlib (level 9) compression ratio is recorded "
                  "per draw as a diagnostic and never decides (a progressing enumeration compresses like a loop without being "
                  "one). Applied per at-cap draw to the text the store carries for it; a draw whose text the store does not "
                  "carry is counted apart, never folded into either class")
#: where an at-cap draw's text came from: the store's ``rows.jsonl`` (both channels on every SCORED draw — the producer's
#: ``ItemResult.extra``; an UNSCORED draw files no row) or the retained ``.eval`` log (every draw; the log-reading venv only)
TEXT_SOURCE_ROWS, TEXT_SOURCE_EVAL_LOG = "rows.jsonl", "eval_log"
#: which split a leg's `at_cap.{n_loop, n_truncated_reasoning, n_text_unavailable}` report: the retained .eval log's (every
#: at-cap draw's text, the unscored case-(b) draws included — the log-reading run, which now draws the figures) or the store
#: rows' (both channels of every SCORED draw only; an unscored draw's text is not carried and is counted as unavailable —
#: the run that reads no log). The store-rows split always rides alongside under `at_cap.store_rows_split`, identical in
#: both runs; the chosen split differs between them BY DESIGN and is the one set of leaves the two-venv match excludes
SPLIT_SOURCE_EVAL_LOG, SPLIT_SOURCE_STORE_ROWS = "eval_log", "store_rows"
LOOP_RULE_SCOPE_TEXT = ("`legs.*.at_cap.split_source` names the text each cell's split used: `eval_log` when the retained .eval "
                        "log was read and carried every at-cap draw's text (the unscored case-(b) draws — the cap inside the "
                        "think block — included; the log-reading run), else `store_rows`: rows.jsonl carries both channels of "
                        "every SCORED draw and no row for an UNSCORED one, so case-(b) draws are `n_text_unavailable`, never "
                        "folded into either class. The store-rows split rides alongside under `at_cap.store_rows_split` in "
                        "both runs; the log's per-draw classes are under `legs.*.eval_check.at_cap_tails`")
#: under thinking ON at temperature 1.0 a model may decline to think on a draw; the per-cell share of ON
#: draws that FINISHED with no reasoning is a mode-delta column, never folded silently
NO_REASONING_RULE: Mapping[str, str] = {
    C.THINK_SPLIT_RECORD_ONLY: ("a finished (not at-cap) draw whose reasoning channel is empty and whose content carries no "
                                "think tag — the outcome record's think_block_present false (the server parser owns the split; "
                                "the model declined to think on that draw, D115); graded from content as an answer"),
    C.THINK_SPLIT_IMPLICIT_OPEN: ("impossible by construction: the prompt opens the block on every draw, so every record's "
                                  "think_block_present is true (the split-anomaly gate fails otherwise)"),
    C.THINK_SPLIT_EXPLICIT_TAGS: ("a finished (not at-cap) draw whose content carries no <think> tag AND whose reasoning channel "
                                  "is empty — the outcome record's think_block_present false and the row's extra_json.reasoning "
                                  "empty; the whole completion is the answer. The record's flag alone is not the call: "
                                  "inspect_ai 0.3.260 lifts a COMPLETE <think>…</think> block out of the text into "
                                  "ContentReasoning(internal=\"think\") before the scorer reads it, so a finished draw whose "
                                  "text carries no tag but whose reasoning channel is non-empty emitted its block (counted "
                                  "present, by channel); a tag left in the text marks a block the client did not lift (an "
                                  "at-cap unterminated one — present, by record)"),
    # llama-paper-rows (P19 / P21): the NO-SWITCH family's rule
    C.THINK_SPLIT_NONE: ("no reasoning channel exists (a no-thinking-channel family served under the `not_available` arm): "
                         "EVERY finished draw has no reasoning BY CONSTRUCTION, and ANY reasoning evidence on a draw — a "
                         "think-tag literal in the text (the record's stray_open_tag / stray_close_tag), a non-empty "
                         "client-lifted reasoning channel on the row (extra_json.lifted_reasoning / reasoning), or the "
                         "provider's client_lift_fired flag — is a STRAY-TAG ANOMALY: counted by source under "
                         "n_stray_tag_draws, never folded into a reasoning share, never discarded; a primary leg fails "
                         "closed on it (D180's rule), a sensitivity leg masks and reports"),
}
#: which source called a draw's reasoning present (``thinking_record``): the outcome record's own ``think_block_present``
#: (``record`` — under explicit_tags a tag in the text, under record_only the server parser's split, under implicit_open
#: by construction), or the row's client-lifted reasoning channel (``channel`` — explicit_tags only, the record false)
PRESENT_BY_RECORD, PRESENT_BY_CHANNEL = "record", "channel"
#: sampling-row members compared literally; ``source`` is prose, never compared
ROW_NUMERIC_FIELDS: tuple[str, ...] = ("temperature", "top_p", "top_k", "min_p", "presence_penalty",
                                       "repetition_penalty")
SUMMARY_ROW_FIELDS: tuple[str, ...] = ("temperature", "top_p", "top_k")
#: the matrix row fields the analysis reads (``family`` optional, else the registry)
CELL_ROW_FIELDS: tuple[str, ...] = ("job_id", "model_key", "cell", "sampling_mode", "fitted", "tuple_name",
                                    "route_kind", "served_positions_expected", "expected_stamps", "out_root")

# -- the READER CONTRACT: what the driver (scripts/served_capability_run.py) writes and this script reads. Every
#    name the producer EXPORTS is imported from contracts; the keys the driver spells inline (its evidence.json
#    block names, its resume record, its binding block) are literals here, pinned to the producer helpers' actual
#    outputs by test (the reader reads the producer's records, never shapes of its own). --
REQUEST_EVIDENCE_FILENAME: str = C.REQUEST_EVIDENCE_FILENAME
REQUEST_EVIDENCE_VERSION: int = C.REQUEST_EVIDENCE_VERSION
REQUEST_EVIDENCE_FIELDS: tuple[str, ...] = tuple(C.REQUEST_EVIDENCE_FIELDS)
REQUEST_EVIDENCE_KIND_TASK: str = C.REQUEST_EVIDENCE_KIND_TASK
REQUEST_EVIDENCE_KIND_PROBE: str = C.REQUEST_EVIDENCE_KIND_PROBE
EVIDENCE_RENDER_CERTIFIED = "render_certified"           # predicate (A): contracts.render_certified()'s record
EVIDENCE_RENDER_PROBE = "render_probe"                    # the (A) probe completions per arm, beside it
EVIDENCE_LENS_POSITION = "lens_position_attested"         # predicate (B′): contracts.lens_position_attested()'s record
EVIDENCE_LENS_ON_BITE: str = C.LENS_ON_BITE_KEY           # predicate (C): contracts.lens_on_bite_verdict()'s record
EVIDENCE_BITES: str = C.OOS_EVIDENCE_BITES_BLOCK
#: contracts.render_certified()'s record, key for key (its SUCCESS and FAILURE shapes share these keys; a failure
#: carries every flag false and the problems named)
RENDER_CERTIFIED_FIELDS: tuple[str, ...] = (
    "family", "render_source", C.OOS_EVIDENCE_HEADER_CERTIFIED, "render_regime_certified", "think_shape_certified",
    "render_certified", "expected_header_ids_by_arm", "measured_header_ids_by_arm", "split_mode", "evidence_status",
    "certification_problems")
#: the ADDITIVE keys contracts.render_certified() appends for the ON-pinned standup — the OFF arm's
#: evidence class and refusal body, PER-ARM header provenance, the acceptance clause. On every record the producer
#: writes since they were added; NOT required by `_render_certified_problems` (an earlier record lacks them and is
#: still the helper's record). The pin test holds RENDER_CERTIFIED_FIELDS + these equal to the producer's keys, in order.
RENDER_CERTIFIED_D105_FIELDS: tuple[str, ...] = (
    "render_source_by_arm", "standup_thinking_mode", "off_arm_evidence", "off_arm_refusal", "off_header_source",
    "off_header_role", "off_arm_accepted_under")
RENDER_SOURCE_SERVED = "served_tokenize"                  # render_certified(measured_live=True)'s render_source
#: contracts.lens_position_attested()'s SUCCESS record; its failure is the driver's {"verified": false, "error": ...}
LENS_POSITION_ATTESTED_FIELDS: tuple[str, ...] = (
    "served_positions", "header_located", "object_digest", "attested_policy", "expected_stamps",
    "n_request_records", "n_stamps", "own_key")
#: contracts.lens_on_bite_verdict()'s record plus the two vmodels the driver's _lens_on_bite appends on success;
#: its failure carries applicable / verified false / error
LENS_ON_BITE_FIELDS: tuple[str, ...] = (
    "applicable", "verified", "reasoning_present", "differs_from_base", "lens_render_regime", "lens_finish_reason",
    "base_finish_reason", "problems")
LENS_ON_BITE_DRIVER_FIELDS: tuple[str, ...] = ("lens_vmodel", "base_vmodel")
MANIFEST_KEY_TRANSFER_PROVENANCE: str = C.MANIFEST_KEY_TRANSFER_PROVENANCE   # on every object cell
MANIFEST_KEY_OOS_EXPERIMENT: str = C.MANIFEST_KEY_AUTHORISED_EXPERIMENT      # the matched tuple's name (unhashed)
MANIFEST_KEY_RESUME_ATTEMPTS: str = C.MANIFEST_KEY_RESUME_ATTEMPTS           # one entry per RUN of the store, the first included
MANIFEST_KEY_REQUEST_EVIDENCE: str = C.MANIFEST_KEY_REQUEST_EVIDENCE
MANIFEST_KEY_PRODUCER: str = C.MANIFEST_KEY_PRODUCER
MANIFEST_KEY_BINDING = "binding"                          # the driver's MANIFEST_KEY_BINDING (a driver-module constant)
#: the driver's per-run record; ``completed`` is written false BEFORE the run's probes and flipped true only when the run
#: passes every gate (FC4 (3)), so a preflight that refused stays attributable and a store whose LAST run never passed is
#: refused by name
RESUME_ATTEMPT_FIELDS: tuple[str, ...] = ("ts", "tree_sha", "attempt", "completed")
PRODUCER_DRIVER_EXPECTED: str = C.PRODUCER_DRIVER_SERVED
#: the SELECTED cell's served modes — the driver copies the standup record's `lens_served_positions[<own key>]`
#: (a LIST on a lens cell, null on the base / prompted route) into `route` and the manifest top level
ROUTE_KEY_LENS_SERVED_POSITIONS: str = C.LENS_SERVED_POSITIONS_KEY
IDENTITY_KEY_VERSION: str = C.IDENTITY_KEY_IDENTITY_VERSION
IDENTITY_VERSION_EXPECTED: int = C.TINYCAP_IDENTITY_VERSION_2
IDENTITY_KEY_DIAGNOSTIC: str = C.IDENTITY_KEY_DIAGNOSTIC      # refused on a production store
IDENTITY_KEY_THINKING_HEADER_IDS: str = C.IDENTITY_KEY_THINKING_HEADER_IDS
HEADER_CHECK_CLASS_EXPECTED: str = C.HEADER_CHECK_DOCUMENT_FIT_INAPPLICABLE
FIT_KIND_EXPECTED = "document_fit"                        # sweep.DOCUMENT_FIT_STATUS (the fit-provenance status)
#: math500-steer: an EXTERNAL object arm (AlphaSteer / Arditi — `ARM_BY_CELL[cell].external`) is served from a class-B
#: object whose provenance the driver MIRRORS from the bank's `fit_provenance.json` (`sweep.resolve_training_mirror`): its
#: `transfer_provenance.fit_kind` is the mirror's STATUS word `verified` (the `sweep.TrainingMirror.status` vocabulary:
#: verified | partial | override | document_fit), never the LoRA arms' document fit. The expectation is keyed by the ARM
#: (:func:`fit_kind_expected`): a document-fit word on an external arm and a mirror word on a LoRA arm both refuse by
#: name, `partial` / `override` refuse everywhere. Every pulled AlphaSteer / Arditi store records `verified`.
FIT_KIND_EXPECTED_EXTERNAL = "verified"
#: the report block of the per-leg store-pins admission (--store-pins; the GSM8K reader's `STORE_PINS_KEY`)
STORE_PINS_KEY = "store_pins"
BITES_PROBE_RENDER_ARM_EXPECTED: str = C.THINKING_OFF     # The bites probe renders OFF on all-position cells
RENDER_FACT_CERTIFIED: str = C.RENDER_EVIDENCE_CERTIFIED
RENDER_FACT_FIELDS: tuple[str, ...] = ("off_ids", "on_ids", "split_mode", "server_parser", "evidence_status")
#: the ONE producer stamp ``--allow-pending-gates`` may waive: the per-row sampling
#: digest — absent on EVERY row of a store from a driver before it
STAMP_ROWS_SAMPLING_REF = "rows.sampling_ref"
STAMP_ROUTE_SERVED_POSITIONS = "route.lens_served_positions"
WAIVABLE_PENDING_STAMPS: tuple[str, ...] = (STAMP_ROWS_SAMPLING_REF, STAMP_ROUTE_SERVED_POSITIONS)

#: the W&B purpose: the project is sisteer.wandb_entity.wandb_project(WANDB_PURPOSE) (the site's SISTEER_WANDB_PROJECT_TINYCAP first)
WANDB_PURPOSE = "tinycap"
WANDB_RUN_PREFIX = "m5t-"
MIRROR_DIR_DEFAULT = "~/figs/math500-thinking-on-full22"
NUMBERS_FILENAME = "final_m5t_figures_numbers.json"
TABLES_FILENAME = "final_m5t_tables.json"
#: ``figures.excluded_families`` — the model families the FIGURES and the rendered tables leave out (``--figure-exclude-family``,
#: repeatable; GLM is out of the paper's figures and headline tables while its legs stay in the
#: report's data). The report's ``cells`` / ``legs`` / ``comparisons`` / gates keep every leg and no numeric leaf moves; the same
#: key is recorded in the numbers sidecar and the tables JSON.
FIGURES_EXCLUDED_KEY = "excluded_families"
#: beside it, only when a family is excluded: the families the figures DO show, and the stem prefix every figure file of this analysis carries
FAMILIES_SHOWN_KEY = "families_shown"
STEM_PREFIX = "final_m5t_"
EXIT_NOT_PRODUCTION = T.EXIT_NOT_PRODUCTION
EXIT_REPORTS_DIFFER = T.EXIT_REPORTS_DIFFER
REPORT_DIFF_SKIP = T.REPORT_DIFF_SKIP            # the parent's path-COMPONENT skip list, unchanged (eval_check, figures, ...)
#: the leaves that differ BY DESIGN between the two runs and nothing else: the log-reading run classifies every at-cap draw
#: from the .eval log (`split_source` eval_log), the run without logs classifies from the store rows and counts the unscored
#: case-(b) draws as text-unavailable (`split_source` store_rows) — so `at_cap.{split_source, n_loop, n_truncated_reasoning,
#: n_text_unavailable}` (on the leg and its controls mirror) and the same three counts on `truncation.cells[*]` cannot agree.
#: The store-rows split under `at_cap.store_rows_split` is identical in both runs and stays in the match; so does every
#: other leaf. `split_source` is a string (never a numeric leaf) and is listed for completeness.
REPORT_DIFF_SKIP_SPLIT_FIELDS: tuple[str, ...] = ("split_source", "n_loop", "n_truncated_reasoning", "n_text_unavailable")
#: the ledger gate's re-draw readout (`ledger.redraws` on the leg, the gate's per-leg record and the controls
#: mirror) is derived from the usage ledger alone, so every numeric leaf of it is identical in both runs and stays in
#: the match. The ONE leaf that differs by design is the string `attribution_source`: the log-reading run cross-checks
#: the per-sample call counts against the .eval log's model events and says so, the run without logs cannot. The
#: cross-check's own numbers (`eval_check.redraws`: model retries, failed attempts, the per-item comparison) ride under
#: `eval_check`, which the parent's component skip already excludes — a log-derived reading the figure venv cannot make.
REPORT_DIFF_SKIP_REDRAW_FIELDS: tuple[str, ...] = ("attribution_source",)
#: the SUB-CLASSES of an unterminated-not-at-cap draw (the split-anomaly gate's `n_unterminated_not_at_cap`), read
#: off run E's retained .eval log — the raw HTTP body under `ModelEvent.call.response`, attachments resolved, which the
#: producer's scorer never saw whole: `lifted_close` — a SYNTACTIC observation only: the raw body carries a `</think>`
#: occurrence that inspect_ai 0.3.260's client-side lift (grammar `<think([^>]*)>`, paired with an open-tag literal the model
#: emitted while reasoning) removed into a reasoning part, so the completion `split_think(implicit_open)` read has no close;
#: it is filed ONLY when the last raw close sits outside a fenced / inline-code span AND the raw text after it carries the
#: task's answer marker (an `ANSWER:` line) — the two signs that the removed close terminated the block and an answer
#: followed. It does not by itself prove the model closed its block or answered, and it excludes no model behaviour;
#: `eos_in_thoughts` — no close anywhere in the raw body and finish_reason stop (the model ended its turn
#: inside its thoughts); `other` — named, the ambiguous cases included (a close inside a quoted / fenced example, no answer
#: marker after the last close). Run F reads no log and files `unavailable`. The items stay unscored under their class
#: either way (no re-score this wave).
ANOMALY_LIFTED_CLOSE, ANOMALY_EOS_IN_THOUGHTS, ANOMALY_OTHER = "lifted_close", "eos_in_thoughts", "other"
ANOMALY_CLASS_UNAVAILABLE = "unavailable"
ANOMALY_CLASSES: tuple[str, ...] = (ANOMALY_LIFTED_CLOSE, ANOMALY_EOS_IN_THOUGHTS, ANOMALY_OTHER, ANOMALY_CLASS_UNAVAILABLE)
#: the parent's anomaly predicate (`T.anomaly_ids`), named per item: the sub-classed kind, an unscored draw whose block is
#: not unterminated (unscored for another reason), a scored draw carrying a stray open tag
ANOMALY_KIND_UNTERMINATED_NOT_AT_CAP, ANOMALY_KIND_UNSCORED_OTHER, ANOMALY_KIND_STRAY_OPEN_TAG = \
    "unterminated_not_at_cap", "unscored_other", "stray_open_tag"
#: inspect_ai 0.3.260's grammar for an OPENING think tag (`model/_reasoning.py`): every literal the lift would pair
LIFT_OPEN_TAG = re.compile(r"<think[^>]*>")
#: the MATH answer line the extractor reads (`think_split._ANSWER_LINE`'s label grammar): the answer marker whose presence
#: AFTER the last raw close is required for `lifted_close`
ANSWER_MARKER = re.compile(r"(?im)^.*?ANSWER\s*:")
CLASS_SOURCE_EVAL_LOG = "eval_log"
#: the run WITHOUT eval logs (run F) takes the classes of the report it is required to match (run E's, which read the
#: raw bodies) — the leaf is a declared by-design one (REPORT_DIFF_SKIP_ANOMALY_CLASS_FIELDS), so the source is named, not hidden
CLASS_SOURCE_MATCHED_REPORT = "matched_report"
CLASS_SOURCES_CLASSIFIED: tuple[str, ...] = (CLASS_SOURCE_EVAL_LOG, CLASS_SOURCE_MATCHED_REPORT)
#: the word check's third state: a comparison that reads a masked leg did not run, so nothing was re-run — the
#: gate is PENDING, never PASSED by vacuity
WORD_MOVES_UNUSABLE = "unusable"
#: the class leaves differ between the two runs BY DESIGN (run E classifies from the log, run F cannot) and are declared
#: here, exactly: the gate's per-leg class counts / ids / literals / source (`gates.split_anomalies.counts.<leg>.<field>`)
#: and the per-leg unscored breakdown's `by_class` / `class_source` (the leg record and its controls mirror). Every
#: record-derived number — the anomaly COUNT, the masked ids, the re-run words — stays in the match.
REPORT_DIFF_SKIP_ANOMALY_CLASS_FIELDS: tuple[str, ...] = ("n_lifted_close", "n_eos_in_thoughts", "n_class_other",
                                                          "n_class_unavailable", "classes", "tag_literals", "class_source")
REPORT_DIFF_SKIP_UNSCORED_FIELDS: tuple[str, ...] = ("by_class", "class_source")
#: the report block of the multi-tree allowance (also its refusal's name) — shas, paths, the reason: metadata the
#: two-venv match never reads as a numeric leaf
TREE_SET_KEY = "tree_set"
REPORT_DIFF_METADATA_KEYS: tuple[str, ...] = (TREE_SET_KEY,)


def _split_leaf_skipped(path: Sequence[Any]) -> bool:
    """True for exactly the by-design leaves: ``<...>.at_cap.<field>`` (the leg
    record and its controls mirror — never ``at_cap.store_rows_split.<field>``),
    ``truncation.cells.<i>.<field>``, and ``<...>.redraws.attribution_source``."""
    if not path:
        return False
    if _anomaly_class_leaf_skipped(path):
        return True
    if path[-1] in REPORT_DIFF_SKIP_REDRAW_FIELDS:
        return len(path) >= 2 and path[-2] == "redraws"
    if path[-1] not in REPORT_DIFF_SKIP_SPLIT_FIELDS:
        return False
    if len(path) >= 2 and path[-2] == "at_cap":
        return True
    return len(path) >= 4 and path[0] == "truncation" and path[1] == "cells"


def _anomaly_class_leaf_skipped(path: Sequence[Any]) -> bool:
    """True for exactly the anomaly-class leaves: ``gates.split_anomalies.counts.<leg>.<field>``
    (:data:`REPORT_DIFF_SKIP_ANOMALY_CLASS_FIELDS`, subtrees included) and
    ``legs.<leg>.unscored.<field>`` / ``controls.<i>.unscored.<field>``
    (:data:`REPORT_DIFF_SKIP_UNSCORED_FIELDS`) — never a same-named leaf elsewhere."""
    if len(path) >= 5 and tuple(path[:3]) == ("gates", "split_anomalies", "counts"):
        return path[4] in REPORT_DIFF_SKIP_ANOMALY_CLASS_FIELDS
    if len(path) >= 4 and path[0] == "legs" and path[2] == "unscored":
        return path[3] in REPORT_DIFF_SKIP_UNSCORED_FIELDS
    return len(path) >= 4 and path[0] == "controls" and path[2] == "unscored" and path[3] in REPORT_DIFF_SKIP_UNSCORED_FIELDS


def _metadata_skipped(path: Sequence[Any]) -> bool:
    """True for exactly the top-level metadata blocks (:data:`REPORT_DIFF_METADATA_KEYS`)."""
    return len(path) == 1 and path[0] in REPORT_DIFF_METADATA_KEYS


def _without_split_leaves(node: Any, path: tuple = ()) -> Any:
    if isinstance(node, Mapping):
        return {k: _without_split_leaves(v, path + (k,)) for k, v in node.items()
                if not _split_leaf_skipped(path + (k,)) and not _metadata_skipped(path + (k,))}
    if isinstance(node, (list, tuple)):
        return [_without_split_leaves(v, path + (i,)) for i, v in enumerate(node)]
    return node


def numeric_leaf_diff(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    """The parent's two-venv comparison (every numeric leaf outside its
    component skip list must agree) over both reports with the by-design split
    leaves (:data:`REPORT_DIFF_SKIP_SPLIT_FIELDS`, :func:`_split_leaf_skipped`)
    removed first — the ONE nested skip this analysis adds — and the top-level
    ``tree_set`` metadata block (:data:`REPORT_DIFF_METADATA_KEYS`) left out; the
    store-rows split and every other leaf are still held equal."""
    return T.numeric_leaf_diff(_without_split_leaves(a), _without_split_leaves(b))


mirror_figures = T.mirror_figures
unusable_legs = T.unusable_legs
_gate = T._gate
report_path = T.report_path
_same_leaf = T._same_leaf
_read_json = T._read_json
_iter_jsonl = T._iter_jsonl
_mapping = T._mapping


#: the bank VIEW a row is served from (E3, math-eight-arms-extension decisions-e3 E3-1 item 10): every published / extension row
#: is the `main` view (the base-lane objects); the it-doc rows (`object_source: itdoc`, wave `itdoc`) are the `it` view — the SAME
#: model served through another bank view — so an `it` row shares (model, cell, mode) with the main row of its cell and is keyed
#: apart by its view. The main-view spellings below are unchanged byte for byte.
VIEW_MAIN = "main"
VIEW_IT = "it"
OBJECT_SOURCE_ITDOC = "itdoc"
LEG_KEY_VIEW_SEP = "@"
#: the it-doc carriage every it-doc reading carries beside its numbers, VERBATIM off the row: the object source, the lane the pin
#: is keyed under, the model's own qualification sentence and the one cross-model `qualification_pattern` sentence
ITDOC_CARRIAGE_FIELDS: tuple[str, ...] = ("object_source", "lane", "qualification", "qualification_pattern")


def leg_key(model_key: str, cell: str, mode: str, view: str = VIEW_MAIN) -> str:
    """``<model_key>/<cell>/<sampling_mode>`` — every per-cell number is keyed on the FULL leg. A leg of a NON-main view (E3: the
    it-doc rows) appends ``@<view>`` to the mode segment: the main-view spelling is unchanged byte for byte, and every parser
    that splits a key on ``/`` still reads the model at [0] and the cell at [1] (:func:`leg_key_view` reads the view)."""
    key = f"{model_key}/{cell}/{mode}"
    return key if view == VIEW_MAIN else f"{key}{LEG_KEY_VIEW_SEP}{view}"


def leg_key_view(key: str) -> str:
    """The view a leg key (or a comparison name, ``<model>/<mode>[@view]``) spells: ``main`` unless its last segment carries
    ``@<view>``."""
    tail = str(key).rsplit("/", 1)[-1]
    return tail.split(LEG_KEY_VIEW_SEP, 1)[1] if LEG_KEY_VIEW_SEP in tail else VIEW_MAIN


def row_view(row: Mapping[str, Any]) -> str:
    """A matrix row's bank view: its ``view`` field, else ``main`` (the published matrix carries none)."""
    return str(row.get("view") or VIEW_MAIN)


#: the arms a MATH-500 row may run: thinking ON (every published / steer row of a switched family) or the NO-SWITCH arm
#: (`not_available`: the llama family, whose template has no thinking switch; the launcher's
#: `THINKING_MODE_BY_FAMILY`). The OFF arm is no MATH-500 arm.
ROW_ARMS: tuple[str, ...] = (C.THINKING_ON, C.THINKING_NOT_AVAILABLE)


def row_arm(row: Mapping[str, Any]) -> str:
    """The ARM a matrix row runs — its ``thinking_mode`` (the launcher files one on every extension row; the published rows
    carry none and are the thinking-ON arm). Anything outside :data:`ROW_ARMS` refuses by name: a row is served under ONE
    whole arm, and the reader's classes (identity, stamps, certification) are keyed on it."""
    arm = row.get("thinking_mode")
    if arm is None:
        return C.THINKING_ON
    if arm not in ROW_ARMS:
        raise AnalysisError(f"cells matrix row {row.get('index')!r} ({row.get('model_key')}/{row.get('cell')}): thinking_mode {arm!r} is not one "
                            f"of {list(ROW_ARMS)} — no MATH-500 arm reads it")
    return str(arm)


def primary_mode_for(row: Mapping[str, Any]) -> str:
    """The PRIMARY sampling row of a matrix row's ARM (:data:`PRIMARY_MODE_BY_ARM`): the thinking card on the ON arm, the shipped
    `card` row on the no-switch arm — the row the paper reads and the mode deltas are taken against."""
    return PRIMARY_MODE_BY_ARM[row_arm(row)]


def row_is_primary(row: Mapping[str, Any]) -> bool:
    """Whether a matrix row is its arm's PRIMARY sampling row (:func:`primary_mode_for`)."""
    return str(row["sampling_mode"]) == primary_mode_for(row)


def fit_kind_expected(cell: str) -> str:
    """The ``transfer_provenance.fit_kind`` a fitted leg of ``cell`` must record, by the ARM: the class-B mirror's status
    word on an EXTERNAL object arm (:data:`FIT_KIND_EXPECTED_EXTERNAL`), the document-fit status on every LoRA arm
    (:data:`FIT_KIND_EXPECTED`). The cell must be a registered arm."""
    arm = ARM_BY_CELL.get(str(cell))
    if arm is None:
        raise AnalysisError(f"fit_kind_expected: {cell!r} is not a registered arm {list(ARM_BY_CELL)}")
    return FIT_KIND_EXPECTED_EXTERNAL if arm.external else FIT_KIND_EXPECTED


def validators_module() -> Any:
    """The GSM8K validators (``scripts/tinycap_gsm8k_validators.py``), loaded ONCE and LAZILY: that module loads THIS one through
    ``sys.modules`` at its import, so the import runs inside the call, never at module import. This module registers itself under
    its script name first, so the validators bind to the SAME module object (the running ``__main__`` included) — one set of
    constants, one ``AnalysisError``. The classes the reader takes from it (math500-steer): the SERVED-STEER
    stamp class (`leg_expected_stamps` / `served_steer_stamps` / `served_steer_problems`, selected by the LEG's own
    `steering_bites.liveness_gate`) and the NO-SWITCH arm's records (`_no_switch_manifest_problem`, `_no_switch_render_probe_problems`,
    `_no_switch_header_problems`, `_no_switch_bites_problems`, `_probe_arm`; llama-paper-rows P18)."""
    me = sys.modules.get(__name__)
    if me is not None:
        sys.modules.setdefault("tinycap_math500_thinking", me)
    name = "tinycap_gsm8k_validators"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _HERE / "tinycap_gsm8k_validators.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def key_view(row: Mapping[str, Any]) -> str:
    """The view a row is KEYED by. On a fitted-object arm the main view also serves (the LoRA cells) it is the row's ``view``:
    an it-doc row shares (model, cell, mode) with the main-view row of its cell and is a second leg of it (E3). On an
    EXTERNAL arm (AlphaSteer / Arditi) it is ``main`` whatever the row's ``view`` says: there the field names the object
    bank's view directory (``it``, ``ar_r0p05``, ... — a per-row IDENTITY key, recorded on the leg and its provenance row),
    the panel has ONE such slot per model and no main-view twin exists — keying it apart would empty the slot."""
    cell = str(row["cell"])
    arm = ARM_BY_CELL.get(cell)
    if arm is not None and arm.external and not arm.view_keyed:
        return VIEW_MAIN
    # an external arm whose row sits on a paper view (`Arm.view_keyed`: the all-positions Arditi row is an `it`-view row, its
    # bank view carried apart as `bank_view`) is keyed by that view like a LoRA row — the `@it` suffix is load-bearing
    return row_view(row)


def row_leg_key(row: Mapping[str, Any]) -> str:
    """The leg key of a matrix row — its (model, cell, mode) and, on a non-main KEY view (:func:`key_view`), its view."""
    return leg_key(str(row["model_key"]), str(row["cell"]), str(row["sampling_mode"]), key_view(row))


def is_itdoc_row(row: Mapping[str, Any]) -> bool:
    """Whether a matrix row (or a leg / controls record carrying the row's fields) is an it-doc row (``object_source: itdoc``)."""
    return row.get("object_source") == OBJECT_SOURCE_ITDOC


def itdoc_carriage(row: Mapping[str, Any]) -> dict[str, Any]:
    """The it-doc fields a reading carries verbatim off its row (:data:`ITDOC_CARRIAGE_FIELDS`; ``None`` where the row is silent)."""
    return {f: row.get(f) for f in ITDOC_CARRIAGE_FIELDS}


def qualification_text(rec: Mapping[str, Any]) -> str | None:
    """The qualification prose of an it-doc record: the row's own sentence and the cross-model pattern, in that order, joined
    by a space; ``None`` when the record carries neither."""
    parts = [str(rec[f]) for f in ("qualification", "qualification_pattern") if rec.get(f)]
    return " ".join(parts) if parts else None


def pin_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    """The object-pin key of a fitted row: ``(model_key, cell)`` (the published keying, F11b) on the main view, ``(model_key,
    cell, view)`` on another — an `it` row's object is not the main row's, though the cell id is the same."""
    key, cell, view = str(row["model_key"]), str(row["cell"]), key_view(row)
    return (key, cell) if view == VIEW_MAIN else (key, cell, view)


def comp_view(comp: Mapping[str, Any]) -> str:
    """A comparison record's view: its ``view`` field (a non-main group names it), else ``main``."""
    return str(comp.get("view") or VIEW_MAIN)


def comp_slot(comp: Mapping[str, Any]) -> str:
    """The key a comparison's estimands sit under in ``report["estimands"][model_key]``: the sampling mode (the published shape)
    on the main view, ``<mode>@<view>`` on another — the two views' estimands never overwrite each other."""
    view = comp_view(comp)
    return str(comp["mode"]) if view == VIEW_MAIN else f"{comp['mode']}{LEG_KEY_VIEW_SEP}{view}"


def _view_tag(view: str) -> str:
    """`` [<view>]`` for a non-main view, ``""`` for main — the label suffix every figure / caption spelling uses."""
    return "" if view == VIEW_MAIN else f" [{view}]"


def cell_slug(cell: str) -> str:
    """The out-root directory token of a cell (``__`` -> ``--``)."""
    return cell.replace("__", "--")


def mode_delta_contrast(kind: str, mode: str) -> str:
    """The name of a sensitivity row's per-cell paired delta to the primary row."""
    return f"{kind}__{mode}_minus_{MODE_PRIMARY}"


def relative_out_root(model_key: str, cell: str, mode: str) -> str:
    return f"controls/{model_key}/{cell_slug(cell)}/{mode}"


# --------------------------------------------------------------------------- #
# inputs: the cells matrix, the sampling rows, the render facts, the pins      #
# --------------------------------------------------------------------------- #


def _row_dict(row: Any) -> dict[str, Any]:
    if isinstance(row, Mapping):
        return dict(row)
    if hasattr(row, "_asdict"):
        return dict(row._asdict())
    raise AnalysisError(f"a cells-matrix row must be a mapping or a NamedTuple, got {type(row).__name__}")


def load_cells(source: Path | str | Sequence[Any]) -> list[dict[str, Any]]:
    """The experiment's cell matrix (``M5T_CELLS``) as validated
    dicts: a ``.py`` file exporting ``M5T_CELLS``, a JSON list, or an in-memory
    sequence of rows. Every row carries :data:`CELL_ROW_FIELDS`; ``out_root``
    must END WITH the analysis' own layout ``controls/<model>/<slug>/<mode>``
    (an absolute cluster path is accepted and reduced to that tail; another
    layout refuses — the matrix and the reader would look in two places); a
    (model, VIEW, cell, mode) named twice refuses (E3: an `it`-view row shares the
    triple with the main-view row of its cell and is a different leg)."""
    if isinstance(source, (str, Path)):
        p = Path(source).expanduser()
        if not p.is_file():
            raise AnalysisError(f"cells matrix {p}: no such file")
        if p.suffix == ".py":
            rows = getattr(_load_module("m5t_cells_matrix", p), "M5T_CELLS", None)
            if rows is None:
                raise AnalysisError(f"cells matrix {p}: no M5T_CELLS")
        else:
            rows = _read_json(p)
    else:
        rows = source
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise AnalysisError(f"cells matrix: expected a sequence of rows, got {type(rows).__name__}")
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for i, raw in enumerate(rows):
        row = _row_dict(raw)
        missing = [f for f in CELL_ROW_FIELDS if f not in row]
        if missing:
            raise AnalysisError(f"cells matrix row {i}: missing field(s) {missing}")
        key = (str(row["model_key"]), str(row["cell"]), str(row["sampling_mode"]))
        ident = (key[0], key_view(row), key[1], key[2])
        if ident in seen:
            raise AnalysisError(f"cells matrix names {row_leg_key(row)} twice")
        seen.add(ident)
        cell = str(row["cell"])
        if cell not in CELL_KIND:
            raise AnalysisError(f"cells matrix row {i}: cell {cell!r} is not a registered arm {list(CELL_KIND)}")
        # the eight-arm extension's design fields (plan P2 / P5): a row without them is a PUBLISHED launch row — read
        # through row_status() / row_wave(), never written back (the published rows stay byte-identical in the report)
        status = row_status(row)
        if status not in ROW_STATUSES:
            raise AnalysisError(f"cells matrix row {i}: status {status!r} is not one of {ROW_STATUSES}")
        row["fitted"] = bool(row["fitted"])
        stamps = row["expected_stamps"]
        if status in EVIDENCE_STATUSES:
            if not isinstance(stamps, (list, tuple)) or len(stamps) != 3:
                raise AnalysisError(f"cells matrix row {i}: expected_stamps must be the triple (salt, policy, render), "
                                    f"got {stamps!r}")
            row["expected_stamps"] = [str(s) for s in stamps]
            if ARM_BY_CELL[cell].external:
                # an EVIDENCE row of an external-object arm needs a cell the grammar admits (`alphasteer` is; the
                # provisional `arditi` spelling is a design-only arm until that spelling lands)
                try:
                    C.cell_spellings(cell, declared=False)
                except C.CellNameError as e:
                    raise AnalysisError(f"cells matrix row {i}: {cell!r} is a design-only arm on this tree (no cell grammar admits "
                                        f"it — {e}); it may be a held / deferred row, never {status!r}") from None
            elif row["fitted"] != C.is_lens_cell(cell):
                raise AnalysisError(f"cells matrix row {i}: fitted={row['fitted']} disagrees with the cell grammar "
                                    f"for {cell!r}")
            if row["fitted"] and not row.get("served_positions_expected"):
                raise AnalysisError(f"cells matrix row {i}: a fitted row needs served_positions_expected")
        else:
            # a design-only row (held / deferred): never collected, so its store-side contract is not owed yet
            row["expected_stamps"] = [str(s) for s in stamps] if isinstance(stamps, (list, tuple)) else None
        row["out_root"] = wave_out_root(str(row["out_root"]), *key, where=f"cells matrix row {i}")
        out.append(row)
    return out


def row_status(row: Mapping[str, Any]) -> str:
    """A matrix row's design status: its ``status`` field, else ``launch`` (the published matrix carries none)."""
    return str(row.get("status") or STATUS_LAUNCH)


def row_wave(row: Mapping[str, Any]) -> str:
    """A matrix row's wave: its ``wave`` field, else ``published``."""
    return str(row.get("wave") or WAVE_PUBLISHED)


def wave_out_root(out_root: str, model_key: str, cell: str, mode: str, *, where: str) -> str:
    """The reader's relative out root of a row: the analysis' layout ``<prefix>/<model>/<cell slug>/<mode>`` where the
    prefix is ``controls`` (a published row) or a WAVE-ISOLATED ``controls_<wave>`` (plan P2 / F11a: a pull of one wave's
    subtree never carries another wave's leg). An absolute cluster path is reduced to that tail; another layout refuses
    (the matrix and the reader would look in two places)."""
    tail = relative_out_root(model_key, cell, mode)
    parts = str(out_root).rstrip("/").split("/")
    want = tail.split("/")
    if len(parts) < len(want) or parts[-len(want) + 1:] != want[1:] or not parts[-len(want)].startswith(OUT_ROOT_PREFIX):
        raise AnalysisError(f"{where}: out_root {out_root!r} does not end with the analysis' layout {tail!r} (or a wave-isolated "
                            f"{OUT_ROOT_PREFIX}_<wave>/... of it; plan §9.2 item 4) — the matrix and the reader would look in two places")
    return "/".join(parts[-len(want):])


def _numeric_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """The numeric members a sampling row names (present iff not None)."""
    return {f: row[f] for f in ROW_NUMERIC_FIELDS if f in row and row[f] is not None}


def load_sampling_rows(source: Path | str | Mapping[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    """``{mode: {model_key: row}}`` from the pinned run config (JSON or YAML,
    optionally under a ``modes`` key) or an in-memory mapping; every row's
    numeric members are kept and cross-checked against ``contracts.sampling_for``
    for every mode this tree's contracts know (a disagreement refuses: two
    tables would be two instruments)."""
    if isinstance(source, (str, Path)):
        p = Path(source).expanduser()
        if not p.is_file():
            raise AnalysisError(f"sampling rows {p}: no such file")
        text = p.read_text(encoding="utf-8")
        if p.suffix in (".yaml", ".yml"):
            import yaml
            doc = yaml.safe_load(text)
        else:
            doc = json.loads(text)
    else:
        doc = source
    if isinstance(doc, Mapping) and isinstance(doc.get("rows"), Mapping):
        # the pinned run config's shape (configs/sdfbase/math500/sampling_rows.yaml): ``modes: [..]`` and
        # ``rows: {model_key: {mode: row}}`` — transposed to mode -> model_key; a row under an undeclared mode refuses
        declared = doc.get("modes")
        table: dict[str, dict[str, Any]] = {}
        for key, by_mode in doc["rows"].items():
            if not isinstance(by_mode, Mapping):
                raise AnalysisError(f"sampling rows: rows[{key!r}] is not a mapping of modes")
            for mode, row in by_mode.items():
                table.setdefault(str(mode), {})[str(key)] = row
        undeclared = sorted(set(table) - {str(m) for m in declared}) if isinstance(declared, list) else []
        if undeclared:
            raise AnalysisError(f"sampling rows: modes {undeclared} carry rows but are not declared under `modes` {declared}")
        doc = table
    elif isinstance(doc, Mapping) and isinstance(doc.get("modes"), Mapping):
        doc = doc["modes"]
    if not isinstance(doc, Mapping):
        raise AnalysisError("sampling rows: expected a mapping mode -> model_key -> row (or the run config's rows: shape)")
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for mode, by_key in doc.items():
        if not isinstance(by_key, Mapping):
            raise AnalysisError(f"sampling rows: mode {mode!r} is not a mapping of model keys")
        out[str(mode)] = {}
        for key, row in by_key.items():
            if not isinstance(row, Mapping) or "temperature" not in row:
                raise AnalysisError(f"sampling rows: {mode}/{key} is not a row with a temperature")
            out[str(mode)][str(key)] = dict(row)
            if str(mode) in C.SAMPLING_MODES:
                want = C.sampling_for(str(key), str(mode))
                got = _numeric_row(row)
                if (got.get("temperature"), got.get("top_p"), got.get("top_k")) != (
                        want.temperature, want.top_p, want.top_k):
                    raise AnalysisError(
                        f"sampling rows: {mode}/{key} = {got} disagrees with contracts.sampling_for({key!r}, "
                        f"{mode!r}) = {want.temperature}/{want.top_p}/{want.top_k} — two tables, two instruments")
    return out


def load_render_facts(source: Path | str | Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    """``{family: fact}`` — from ``--render-facts`` (JSON or a mapping) or this
    tree's ``contracts.FAMILY_RENDER_FACTS``; a tree without
    the table refuses by name rather than reading an empty one."""
    if source is None:
        table = getattr(C, "FAMILY_RENDER_FACTS", None)
        if table is None:
            raise AnalysisError("this tree's contracts carry no FAMILY_RENDER_FACTS (lane A's render-fact table); "
                                "pass --render-facts <json> or run from a tree that has it")
    elif isinstance(source, (str, Path)):
        table = _read_json(Path(source).expanduser())
    else:
        table = source
    if not isinstance(table, Mapping):
        raise AnalysisError("render facts: expected a mapping family -> fact")
    out: dict[str, dict[str, Any]] = {}
    for family, fact in table.items():
        f = _row_dict(fact)
        missing = [k for k in RENDER_FACT_FIELDS if k not in f]
        if missing:
            raise AnalysisError(f"render fact for {family!r} lacks {missing}")
        f["off_ids"] = [int(x) for x in f["off_ids"]]
        f["on_ids"] = [int(x) for x in f["on_ids"]]
        out[str(family)] = f
    return out


def staged_id_of_row(row: Mapping[str, Any]) -> str | None:
    """The pin file's STAGED id of a fitted row's object: the row's ``staged_id`` when the matrix carries it (lane A's
    ``STAGED_ID_BY_CELL``), else the pins file's own grammar — ``derived__<cell>`` for a subtraction object,
    ``base__<cell>`` for a fitted one (B4). ``None`` on a non-fitted row."""
    if not row.get("fitted"):
        return None
    sid = row.get("staged_id")
    if isinstance(sid, str) and sid:
        return sid
    cell = str(row["cell"])
    return f"derived__{cell}" if cell.endswith("_minus_ctl") else f"base__{cell}"


def normalise_object_pins(pins: Mapping[Any, str]) -> dict[tuple[str, ...], str]:
    """``{(model_key, cell): sha}`` from either keying: a legacy ``{model_key: sha}`` mapping (the published call shape —
    the minus_ctl object's pin), tuple keys, or JSON's two-element lists; a non-main view's object is keyed ``(model_key, cell,
    view)`` (E3, :func:`pin_key`)."""
    out: dict[tuple[str, ...], str] = {}
    for k, v in pins.items():
        if isinstance(k, str):
            out[(k, CELL_OBJECT)] = str(v)
        elif isinstance(k, (tuple, list)) and len(k) in (2, 3):
            out[tuple(str(x) for x in k)] = str(v)
        else:
            raise AnalysisError(f"object pins: key {k!r} is neither a model key nor a (model_key, cell[, view]) tuple")
    return out


def load_object_pins(path: Path | str, rows: Sequence[Mapping[str, Any]] | None = None) -> dict[Any, str]:
    """The object pins from the battery's pin file (``configs/sdfbase_full22_pins.json``). Without ``rows``: the
    published call shape and result, ``{model_key: sha}`` of the minus_ctl object per target. With ``rows`` (the cells
    matrix): ``{(model_key, cell): sha}`` — one pin per fitted EVIDENCE row, read under the row's staged id
    (:func:`staged_id_of_row`) — F11b: every fitted arm has its OWN expected object; a row that also carries an
    ``object_sha256`` must agree with the file; an EXTERNAL arm (the steering objects) is pinned by the row's own
    ``object_sha256`` (P8's admission record), never by the full22 file; so is an IT-DOC row (E3, ``object_source: itdoc`` —
    the admission record's pin, keyed ``(model, cell, view)`` by :func:`pin_key`). :class:`Context` admits either shape."""
    doc = _read_json(Path(path).expanduser())
    if rows is None:
        legacy: dict[str, str] = {}
        for key, cells in doc.items():
            if isinstance(cells, Mapping):
                entry = cells.get(f"derived__{CELL_OBJECT}")
                if isinstance(entry, Mapping) and entry.get("object_sha256"):
                    legacy[str(key)] = str(entry["object_sha256"])
        if not legacy:
            raise AnalysisError(f"{path}: no derived__{CELL_OBJECT} object_sha256 pin under any target")
        return legacy
    out: dict[Any, str] = {}
    for r in rows:
        if row_status(r) not in EVIDENCE_STATUSES or not r.get("fitted"):
            continue
        key, cell = str(r["model_key"]), str(r["cell"])
        where = row_leg_key(r)
        if ARM_BY_CELL[cell].external:
            sha = r.get("object_sha256")
            if not isinstance(sha, str) or not C.is_sha256_hex(sha):
                raise AnalysisError(f"{where}: an external-object arm needs the row's own 64-hex object_sha256 (P8), got {sha!r}")
            out[pin_key(r)] = sha
            continue
        if is_itdoc_row(r):
            # an it-doc row is pinned by ITS OWN object_sha256 — the admission record's pin (keyed by lane, model, view, cell
            # and sha there) — never by the full22 pin file, whose staged ids are the base lane's; a hold has no pin, an
            # evidence row without one refuses by name (never a placeholder)
            sha = r.get("object_sha256")
            if not isinstance(sha, str) or not C.is_sha256_hex(sha):
                raise AnalysisError(f"{where}: an it-doc row (object_source {OBJECT_SOURCE_ITDOC!r}, lane {r.get('lane')!r}) is pinned by "
                                    f"the row's own 64-hex object_sha256 — the admission record's pin, never the full22 pin file — got {sha!r}")
            out[pin_key(r)] = sha
            continue
        sid = staged_id_of_row(r)
        entry = _mapping(doc.get(key)).get(sid)
        sha = entry.get("object_sha256") if isinstance(entry, Mapping) else entry
        if not isinstance(sha, str) or not sha:
            raise AnalysisError(f"{path}: no object_sha256 pin for {key!r} / staged id {sid!r} ({where})")
        if r.get("object_sha256") and str(r["object_sha256"]) != str(sha):
            raise AnalysisError(f"{where}: the matrix row's object_sha256 {r['object_sha256']} != the pin file's {sha} for {sid!r}")
        out[pin_key(r)] = str(sha)
    return out


def model_pin(repo_root: Path, model_key: str) -> tuple[str, str]:
    """``(hf id, revision)`` the battery served ``model_key`` at — the top-level
    ``model:`` / ``model_revision:`` lines of ``configs/sdfbase_full22_<key>.yaml``
    (an untouched input of the campaign; read by regex so no YAML loader is owed)."""
    p = Path(repo_root) / "configs" / f"sdfbase_full22_{model_key}.yaml"
    if not p.is_file():
        raise AnalysisError(f"no battery config for {model_key!r} at {p} — the served model pin is unknown")
    text = p.read_text(encoding="utf-8")
    hf = re.search(r"^model:\s*(\S+)\s*$", text, re.MULTILINE)
    rev = re.search(r"^model_revision:\s*(\S+)\s*$", text, re.MULTILINE)
    if not hf or not rev:
        raise AnalysisError(f"{p}: no top-level model / model_revision lines")
    return hf.group(1), rev.group(1)


def family_of(row: Mapping[str, Any]) -> str:
    """The model's SERVING family: the matrix row's ``family`` when it carries
    one, else the contracts registry row; neither -> refuse (a family decides
    the split mode and the header tuples, and is never guessed)."""
    fam = row.get("family")
    if isinstance(fam, str) and fam:
        return fam
    key = str(row["model_key"])
    if key in C.MODELS:
        return str(C.MODELS[key].family)
    raise AnalysisError(f"model key {key!r}: no family — the cells matrix row carries none and contracts.MODELS "
                        "has no registry row for it")


# --------------------------------------------------------------------------- #
# the paired conditional bootstrap over ANY legs and ANY signed contrasts       #
# (pairing.paired_bootstrap_conditional_2x2, generalised; pairing.py untouched) #
# --------------------------------------------------------------------------- #


def _check_legs_general(legs: Mapping[str, Mapping[str, tuple[int, float | None]]],
                        contrasts: Mapping[str, Mapping[str, int]]) -> list[str]:
    """ONE common attempted id set across every leg (a difference refuses,
    listing ids), every contrast naming only known legs with nonzero integer
    signs, and per item a ``(mask, value)`` pair with ``mask`` in {0, 1},
    ``value`` in {0, 1} where the mask is 1 and ``None`` where it is 0."""
    if not isinstance(legs, Mapping) or not legs:
        raise AnalysisError("the paired bootstrap needs a non-empty mapping of legs")
    if not isinstance(contrasts, Mapping) or not contrasts:
        raise AnalysisError("the paired bootstrap needs at least one contrast")
    for name, signs in contrasts.items():
        if not isinstance(signs, Mapping) or not signs:
            raise AnalysisError(f"contrast {name!r} names no leg")
        for k, s in signs.items():
            if k not in legs:
                raise AnalysisError(f"contrast {name!r} names a leg {k!r} that is not among the legs {sorted(legs)}")
            if isinstance(s, bool) or not isinstance(s, int) or s == 0:
                raise AnalysisError(f"contrast {name!r}: the sign of leg {k!r} must be a nonzero integer, got {s!r}")
    ids_by_leg = {k: set(legs[k]) for k in legs}
    common = set.intersection(*ids_by_leg.values())
    union = set.union(*ids_by_leg.values())
    if common != union:
        detail = []
        for k in legs:
            missing = sorted(union - ids_by_leg[k])
            if missing:
                detail.append(f"{k} lacks {len(missing)} (first {missing[:3]})")
        raise AnalysisError("the legs' ATTEMPTED id sets differ — the estimand is defined over ONE common set of "
                            f"attempted items; refusing: {'; '.join(detail)}")
    if not common:
        raise AnalysisError("the paired bootstrap has no items")
    for k in legs:
        for i, pair in legs[k].items():
            if not (isinstance(pair, (tuple, list)) and len(pair) == 2):
                raise AnalysisError(f"{k} item {i!r}: expected (mask, value), got {pair!r}")
            m, v = pair
            if isinstance(m, bool) or m not in (0, 1):
                raise AnalysisError(f"{k} item {i!r}: mask {m!r} is not 0 or 1")
            if m == 1 and (isinstance(v, bool) or v not in (0, 1, 0.0, 1.0)):
                raise AnalysisError(f"{k} item {i!r}: a scored item's value must be 0 or 1, got {v!r}")
            if m == 0 and v is not None:
                raise AnalysisError(f"{k} item {i!r}: a masked-out item carries no value (got {v!r}) — impute in the "
                                    "caller's branch, never here")
    return sorted(common)


def paired_bootstrap_conditional(legs: Mapping[str, Mapping[str, tuple[int, float | None]]],
                                 contrasts: Mapping[str, Mapping[str, int]], *,
                                 n_resamples: int = pr.DEFAULT_BOOTSTRAP, seed: int = pr.DEFAULT_BOOTSTRAP_SEED,
                                 ci: float = pr.DEFAULT_CI,
                                 max_degenerate_frac: float = pr.DEFAULT_MAX_DEGENERATE_FRAC) -> dict[str, Any]:
    """``pairing.paired_bootstrap_conditional_2x2`` with the legs and the signed
    contrasts as ARGUMENTS: ``legs = {label: {item_id: (mask, value)}}`` over ONE
    common attempted id set, ``contrasts = {name: {label: ±1}}``. Per leg the
    CONDITIONAL accuracy with its own denominator; per contrast the signed sum;
    ``n_resamples`` replicates, each ONE index draw applied to every leg
    (``rng.integers(0, n, size=(m, n))`` in the same chunk order as pairing, so
    the draws are a pure function of ``(seed, n, B)`` and every call over the
    same 500 ids shares them); percentile CIs; a replicate in which any leg has
    no scored item is discarded and counted. Sharp censoring bounds by leg-wise
    adverse imputation (a +1 leg at acc⁻ and a −1 leg at acc⁺ for the lower end;
    reversed for the upper). EXACT arithmetic as in pairing: every full-set
    point and bound is a ``Fraction`` over the integer counts converted once,
    every replicate statistic ONE float division of exact integers (the
    contrast over the product of its legs' replicate denominators, the bounds
    over ``n``). Returns pairing's document shape (``acc_by_leg``, ``stats`` with
    ``replicate_signs`` / ``point_sign`` / ``sign_identified`` / ``excludes_zero``,
    ``complete_case``); on the four-leg 2x2 with ``CONTRAST_SIGNS`` the output
    equals pairing's bit for bit (pinned by test)."""
    if not 0.0 < ci < 1.0:
        raise AnalysisError(f"the interval level must be in (0, 1), got {ci!r}")
    if not isinstance(n_resamples, int) or isinstance(n_resamples, bool) or n_resamples < 1:
        raise AnalysisError(f"the bootstrap needs >= 1 resamples, got {n_resamples!r}")
    if not 0.0 <= max_degenerate_frac < 1.0:
        raise AnalysisError(f"max_degenerate_frac must be in [0, 1), got {max_degenerate_frac!r}")
    ids = _check_legs_general(legs, contrasts)
    n = len(ids)
    keys = list(legs)
    k_max = max(len(signs) for signs in contrasts.values())
    if k_max * n ** k_max >= 2 ** 53:
        raise AnalysisError(f"{n} items over a {k_max}-leg contrast: the exact integer replicate arithmetic would "
                            "leave float64's exactly-representable range — refusing rather than rounding twice")
    masks = {k: np.asarray([int(legs[k][i][0]) for i in ids], dtype=np.int64) for k in keys}
    hits = {k: masks[k] * np.asarray([int(legs[k][i][1]) if legs[k][i][0] == 1 else 0 for i in ids],
                                     dtype=np.int64) for k in keys}
    n_scored = {k: int(masks[k].sum()) for k in keys}
    for k in keys:
        if n_scored[k] == 0:
            raise AnalysisError(f"{k} scored no item — its conditional accuracy is undefined")
    n_correct = {k: int(hits[k].sum()) for k in keys}
    n_censored = {k: n - n_scored[k] for k in keys}

    def full_set(acc: Mapping[str, Any], minus: Mapping[str, Any], plus: Mapping[str, Any]
                 ) -> dict[str, tuple[Any, Any, Any]]:
        out: dict[str, tuple[Any, Any, Any]] = {}
        for name, signs in contrasts.items():
            point = sum(s * acc[k] for k, s in signs.items())
            low = sum((minus[k] if s > 0 else -plus[k]) for k, s in signs.items())
            high = sum((plus[k] if s > 0 else -minus[k]) for k, s in signs.items())
            out[name] = (point, low, high)
        return out

    acc_fr = {k: Fraction(n_correct[k], n_scored[k]) for k in keys}
    minus_fr = {k: Fraction(n_correct[k], n) for k in keys}
    plus_fr = {k: Fraction(n_correct[k] + n_censored[k], n) for k in keys}
    point = full_set(acc_fr, minus_fr, plus_fr)

    rng = np.random.default_rng(seed)
    rep_acc = {k: np.empty(n_resamples) for k in keys}
    rep_stat = {c: np.empty(n_resamples) for c in contrasts}
    rep_low = {c: np.empty(n_resamples) for c in contrasts}
    rep_high = {c: np.empty(n_resamples) for c in contrasts}
    keep = np.ones(n_resamples, dtype=bool)
    chunk = max(1, 4_000_000 // n)
    done = 0
    while done < n_resamples:
        m = min(chunk, n_resamples - done)
        idx = rng.integers(0, n, size=(m, n))
        sl = slice(done, done + m)
        scored_r = {k: masks[k][idx].sum(axis=1) for k in keys}
        correct_r = {k: hits[k][idx].sum(axis=1) for k in keys}
        for k in keys:
            degenerate = scored_r[k] == 0
            keep[sl] &= ~degenerate
            with np.errstate(divide="ignore", invalid="ignore"):
                cond = correct_r[k] / scored_r[k]
            rep_acc[k][sl] = np.where(degenerate, np.nan, cond)
        for c, signs in contrasts.items():
            den: Any = 1
            for k in signs:
                den = den * scored_r[k]
            num: Any = 0
            for k, s in signs.items():
                others: Any = 1
                for j in signs:
                    if j != k:
                        others = others * scored_r[j]
                num = num + s * correct_r[k] * others
            low_num: Any = 0
            high_num: Any = 0
            for k, s in signs.items():
                censored = n - scored_r[k]
                low_num = low_num + (correct_r[k] if s > 0 else -(correct_r[k] + censored))
                high_num = high_num + ((correct_r[k] + censored) if s > 0 else -correct_r[k])
            with np.errstate(divide="ignore", invalid="ignore"):
                rep_stat[c][sl], rep_low[c][sl], rep_high[c][sl] = num / den, low_num / n, high_num / n
        done += m
    n_degenerate = int((~keep).sum())
    if n_degenerate > max_degenerate_frac * n_resamples:
        raise AnalysisError(f"{n_degenerate} of {n_resamples} replicates had a leg with no scored item "
                            f"(> {max_degenerate_frac:.1%}) — a leg is nearly all censored; the conditional estimand "
                            "is not estimable here")
    alpha = 1.0 - ci
    acc_by_leg: dict[str, dict[str, Any]] = {}
    for k in keys:
        lo, hi = pr._quantiles(rep_acc[k][keep], ci)
        acc_by_leg[k] = {"point": float(acc_fr[k]), "ci_low": lo, "ci_high": hi,
                         "acc_minus": float(minus_fr[k]), "acc_plus": float(plus_fr[k]),
                         "n_correct": n_correct[k], "n_scored": n_scored[k], "n_censored": n_censored[k]}
    stats: dict[str, dict[str, Any]] = {}
    for c, signs in contrasts.items():
        pt, blo, bhi = point[c]
        reps = rep_stat[c][keep]
        lo, hi = pr._quantiles(reps, ci)
        stats[c] = {
            "point": float(pt), "ci_low": lo, "ci_high": hi,
            "replicate_signs": {"negative": int((reps < 0).sum()), "zero": int((reps == 0).sum()),
                                "positive": int((reps > 0).sum())},
            "bound_low": float(blo), "bound_high": float(bhi),
            "bound_ci_low": float(np.quantile(rep_low[c][keep], alpha / 2.0)),
            "bound_ci_high": float(np.quantile(rep_high[c][keep], 1.0 - alpha / 2.0)),
            "excludes_zero": bool(lo > 0.0 or hi < 0.0),
            "point_sign": (pt > 0) - (pt < 0),
            "sign_identified": bool(blo > 0 or bhi < 0),
            "legs": dict(signs),
        }
    complete: dict[str, Any] = {"label": pr.COMPLETE_CASE_LABEL}
    for c, signs in contrasts.items():
        both = np.ones(n, dtype=bool)
        for k in signs:
            both &= masks[k] == 1
        n_both = int(both.sum())
        if n_both == 0:
            complete[c] = {"n_items": 0, "delta": None, "unavailable": "no item scored in every leg of the contrast"}
            continue
        delta = sum(s * Fraction(int(hits[k][both].sum()), n_both) for k, s in signs.items())
        complete[c] = {"n_items": n_both, "delta": float(delta)}
    return {"B": int(n_resamples), "seed": int(seed), "level": float(ci), "n_items": n,
            "n_scored_by_leg": dict(n_scored), "n_censored_by_leg": dict(n_censored),
            "n_correct_by_leg": dict(n_correct), "n_degenerate_replicates": n_degenerate,
            "n_replicates_used": int(keep.sum()), "acc_by_leg": acc_by_leg, "stats": stats,
            "complete_case": complete}


# --------------------------------------------------------------------------- #
# per-leg records: identity, the sampling row, the stamps, the ledger,         #
# the request evidence, the certification blocks                               #
# --------------------------------------------------------------------------- #


_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX_PIN = re.compile(r"^[0-9a-f]{7,40}$")


def pin_problem(pin: Any) -> str | None:
    """Why ``--sha`` is not a usable store pin: the wave sha as 7 to 40 lowercase
    hex characters — the PREFIX every producer tree sha of the wave must carry
    (the runner passes the 7-char form; the full commit is accepted too)."""
    if not isinstance(pin, str) or not _HEX_PIN.match(pin):
        return (f"store pin {pin!r} is not 7-40 lowercase hex characters of the wave sha (--sha) — the producer trees "
                "cannot be held to it")
    return None


def producer_problem(manifest: Mapping[str, Any], *, pin: str, also: Sequence[str] = ()) -> str | None:
    """Why the store's PRODUCER is not the pinned wave tree — the
    manifest's ``producer`` block (``contracts.PRODUCER_BLOCK_FIELDS``: a 40-hex
    ``tree_sha`` carrying the pin as its prefix — or one of the DECLARED
    trees ``also`` — the served driver) and its ``resume_attempts`` record (one
    entry per RUN of the store, the first included, numbered from 1, EVERY run
    from that same tree — a resume from another tree is a mixed-producer leg).
    ``None`` when the store is the pin's (or a declared tree's)."""
    block = _mapping(manifest.get(MANIFEST_KEY_PRODUCER))
    if not block:
        return f"the manifest carries no {MANIFEST_KEY_PRODUCER} block — the tree that produced the store is unknown"
    sha = block.get("tree_sha")
    if not isinstance(sha, str) or not _HEX40.match(sha):
        return f"manifest {MANIFEST_KEY_PRODUCER}.tree_sha {sha!r} is not a 40-hex commit"
    if block.get("driver") != PRODUCER_DRIVER_EXPECTED:
        return f"manifest {MANIFEST_KEY_PRODUCER}.driver {block.get('driver')!r} != {PRODUCER_DRIVER_EXPECTED!r}"
    if not sha.startswith(pin) and sha not in also:
        declared = f" and is not among the declared trees {list(also)}" if also else ""
        return (f"manifest {MANIFEST_KEY_PRODUCER}.tree_sha {sha} does not carry the wave pin {pin}{declared} — a store from "
                "another tree under this leg's out root")
    attempts = manifest.get(MANIFEST_KEY_RESUME_ATTEMPTS)
    if not isinstance(attempts, list) or not attempts:
        return (f"the manifest carries no {MANIFEST_KEY_RESUME_ATTEMPTS} record (the driver writes one entry per run of the "
                "store, the first included) — the runs that produced it cannot be attributed")
    for i, a in enumerate(attempts):
        if not isinstance(a, Mapping) or set(a) != set(RESUME_ATTEMPT_FIELDS):
            return f"{MANIFEST_KEY_RESUME_ATTEMPTS}[{i}] {a!r} is not the driver's {RESUME_ATTEMPT_FIELDS} record"
        if a.get("attempt") != i + 1:
            return (f"{MANIFEST_KEY_RESUME_ATTEMPTS}[{i}].attempt {a.get('attempt')!r} != {i + 1} — the runs are not numbered "
                    "from 1")
        if a.get("tree_sha") != sha:
            return (f"{MANIFEST_KEY_RESUME_ATTEMPTS}[{i}].tree_sha {a.get('tree_sha')!r} != the manifest's producer {sha} — a "
                    "run of this store from another tree (a mixed-producer leg)")
        if not isinstance(a.get("completed"), bool):
            return f"{MANIFEST_KEY_RESUME_ATTEMPTS}[{i}].completed {a.get('completed')!r} is not the driver's boolean"
    if attempts[-1].get("completed") is not True:
        return (f"{MANIFEST_KEY_RESUME_ATTEMPTS}[{len(attempts) - 1}] (attempt {attempts[-1].get('attempt')}) did not complete — "
                "the store's last run never passed the producer's gates (a refusal after generation, or a run that died); "
                "not a passed run")
    return None


def incomplete_runs(manifest: Mapping[str, Any]) -> int:
    """How many recorded runs of the store did not complete (a preflight that
    refused, FC4 (3)) — attributable, reported, never a failure on their own
    (:func:`producer_problem` already requires the LAST run complete)."""
    attempts = manifest.get(MANIFEST_KEY_RESUME_ATTEMPTS)
    return sum(1 for a in attempts if isinstance(a, Mapping) and a.get("completed") is not True) if isinstance(attempts, list) else 0


def outcomes_producer_problem(header: Mapping[str, Any], *, manifest_sha: str) -> str | None:
    """The outcomes record's own ``producer_tree_sha`` must be the manifest's producer."""
    sha = header.get("producer_tree_sha")
    if sha != manifest_sha:
        return (f"outcomes producer_tree_sha {sha!r} != the manifest's producer tree {manifest_sha} — the all-attempted "
                "record and the manifest were written by different trees")
    return None


def producer_consistency_problem(legs: Mapping[str, Mapping[str, Any]], leg_keys: Mapping[str, str]) -> str | None:
    """Why a set of legs is a MIXED-producer set: every leg's producer tree sha
    (each already held to the pin at collection) must be ONE sha across the
    legs compared — a 7-char pin admits two trees sharing a prefix, and a
    comparison across trees has no estimand."""
    shas = {label: (legs[k].get("producer") or {}).get("tree_sha") for label, k in leg_keys.items()}
    if len(set(shas.values())) > 1:
        return f"the legs' producer trees differ — a mixed-producer comparison: {shas}"
    return None


#: how the trees' difference is ENUMERATED — renames OFF (a detected rename reports only its destination, so an
#: unallowlisted source renamed into an allowlisted destination would vanish from the list), NUL-delimited names (no
#: quoting, no path character lost), submodules included
GIT_DIFF_NAMES_ARGS: tuple[str, ...] = ("diff", "--no-renames", "--name-only", "-z", "--ignore-submodules=none")


def _git_output(repo_root: Path, *args: str) -> str:
    """``git -C <repo_root> <args>``, stdout decoded VERBATIM (no stripping, no
    newline translation: ``-z`` output carries NUL-terminated paths); raises
    ``OSError`` or ``subprocess.CalledProcessError`` (the tests substitute a fake)."""
    done = subprocess.run(["git", "-C", str(repo_root), *args], check=True, capture_output=True)
    return done.stdout.decode("utf-8", "surrogateescape")


def _git_error_text(e: subprocess.CalledProcessError) -> str:
    err = e.stderr.decode("utf-8", "replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
    return f" (git exit {e.returncode}: {err.strip()})" if err.strip() else f" (git exit {e.returncode})"


def verify_tree_set(primary: str, also: Sequence[str] = (), allow: Sequence[str] = (), *, repo_root: Path,
                    reason: str | None = None, allow_file: str | None = None,
                    allow_evidence: Mapping[str, str] | None = None) -> dict[str, Any]:
    """The declared producer trees, verified on the LOCAL repository BEFORE
    any leg is read — the block :func:`analyze` files as ``tree_set``. With no
    ``also`` tree nothing is asked of git (the one-tree rule, as before). Otherwise
    the primary pin and every declared sha must resolve to a commit of
    ``repo_root``, a reason must be given, and every path ``git diff`` names
    between the primary and the declared tree (:data:`GIT_DIFF_NAMES_ARGS` —
    renames OFF, so a removed path cannot hide behind its destination) must
    match an ``allow`` glob (``fnmatch``, case-sensitive). ``problems`` lists
    every failure by name; an empty list is the verdict that admits the set.
    ``allow_file`` / ``allow_evidence`` (B7): the REVIEWED allowlist file the
    globs came from and its per-glob evidence, filed beside them."""
    problems: list[str] = []
    block: dict[str, Any] = {"primary_pin": primary, "primary": None, "also": list(also), "trees": [], "diff_files": {},
                             "allow": list(allow), "why": reason, "repo_root": str(repo_root), "declared": bool(also),
                             "problems": problems, "legs": {}}
    if allow_file is not None or allow_evidence:
        block["allow_file"] = allow_file
        block["allow_evidence"] = dict(allow_evidence or {})
    pin_why = pin_problem(primary)
    if pin_why:
        problems.append(pin_why)
    for sha in also:
        if not isinstance(sha, str) or not _HEX40.match(sha):
            problems.append(f"--also-sha {sha!r} is not a 40-hex commit")
    if len(set(also)) != len(also):
        problems.append(f"--also-sha names a tree twice: {sorted(also)}")
    if not also or problems:
        return block
    if reason is None or not str(reason).strip():
        problems.append("--also-sha requires --tree-set-reason: why the wave's cells come from more than one tree")

    def resolve(ref: str) -> str | None:
        try:
            return _git_output(repo_root, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}").strip()
        except OSError as e:
            problems.append(f"git could not be run on {repo_root}: {e}")
        except subprocess.CalledProcessError as e:
            problems.append(f"{ref} is not a commit of the repository at {repo_root}{_git_error_text(e)}")
        return None

    full = resolve(primary)
    if full is None:
        return block
    block["primary"] = full
    for sha in also:
        if sha == full:
            problems.append(f"--also-sha {sha} is the primary tree itself")
            continue
        if resolve(sha) is None:
            continue
        try:
            raw = _git_output(repo_root, *GIT_DIFF_NAMES_ARGS, full, sha, "--")
        except (OSError, subprocess.CalledProcessError) as e:
            problems.append(f"git {' '.join(GIT_DIFF_NAMES_ARGS)} {full} {sha} failed on {repo_root}: {e}")
            continue
        files = sorted(f for f in raw.split("\0") if f)
        block["diff_files"][sha] = files
        outside = [f for f in files if not any(fnmatch.fnmatchcase(f, g) for g in allow)]
        if outside:
            problems.append(f"tree {sha} differs from the primary {full} outside the allowlist {list(allow)}: {outside}")
    block["trees"] = [full, *also]
    return block


def tree_set_note(tree_set: Mapping[str, Any]) -> str:
    """The one-line note the table and the W&B summary carry under a declaration."""
    ts = _mapping(tree_set)
    files = sorted({f for fs in _mapping(ts.get("diff_files")).values() for f in fs})
    return (f"producer trees: primary {ts.get('primary')} plus declared {ts['also']}, differing only in {files} "
            f"(allowlist {ts.get('allow')}) — {ts.get('why')}")


#: a bullet or table row whose first backticked token is the glob; the evidence is whatever follows the separator (an
#: EMPTY evidence still matches, so the glob is refused BY NAME rather than read as prose)
_ALLOW_LINE = re.compile(r"^\s*(?:[-*]\s+|\|\s*)`(?P<glob>[^`]+)`\s*(?:\||[—–-]+)?\s*(?P<evidence>.*?)\s*\|?\s*$")


def load_tree_diff_allow_file(path: Path | str) -> tuple[list[str], dict[str, str]]:
    """B7 / plan P5: the REVIEWED allowlist file (``configs/sdfbase/eight_arms/tree_set_review_<pub7>_<wave7>.md``): every
    line ``- \\`<glob>\\` — <evidence>`` (or a table row ``| \\`<glob>\\` | <evidence> |``) contributes one path glob with its
    one-line evidence of what the change does to the served identity. A glob without evidence, a glob named twice, or a
    file naming no glob refuses by name. Returns ``(globs, {glob: evidence})``."""
    p = Path(path).expanduser()
    if not p.is_file():
        raise AnalysisError(f"--tree-diff-allow-file {p}: no such file")
    globs: list[str] = []
    evidence: dict[str, str] = {}
    for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        m = _ALLOW_LINE.match(line)
        if not m:
            continue
        glob, why = m.group("glob").strip(), m.group("evidence").strip().rstrip("|").strip()
        if not glob:
            raise AnalysisError(f"{p}:{n}: an empty glob")
        if not why or why in ("-", "—", "TODO", "todo"):
            raise AnalysisError(f"{p}:{n}: the glob {glob!r} carries no evidence — every admitted path needs its one-line reason")
        if glob in evidence:
            raise AnalysisError(f"{p}:{n}: the glob {glob!r} is named twice")
        globs.append(glob)
        evidence[glob] = why
    if not globs:
        raise AnalysisError(f"{p}: no allowlisted glob found (a line `- \\`<glob>\\` — <evidence>` per admitted path)")
    return globs, evidence


def parse_gen_root_spec(spec: str) -> tuple[Path, str | None]:
    """``PATH`` or ``PATH=PIN`` (B6): the pin is 7-40 lowercase hex after the LAST ``=``; anything else is a path."""
    text = str(spec)
    head, sep, tail = text.rpartition("=")
    if sep and _HEX_PIN.match(tail) and head:
        return Path(head).expanduser(), tail
    return Path(text).expanduser(), None


def load_store_pins(sources: Sequence[Path | str | Mapping[str, str]] | Path | str | Mapping[str, str] | None) -> dict[str, str] | None:
    """The per-LEG admitted producer trees (``{"<model>/<cell>/<mode>[@view]": "<tree sha>"}`` — the MATH-500 leg key,
    :func:`row_leg_key`; the GSM8K reader's ``--store-pins`` contract): one or more JSON files or mappings,
    merged; keys starting with ``_`` are documentation (``_meta``, per-row sources) and are skipped; every value is hex of
    >= 7 characters (a tree sha or its prefix); a leg key named twice with DIFFERENT pins refuses by name; anything else
    off-contract refuses by name. ``None`` in -> ``None`` out (the one-pin admission)."""
    if sources is None:
        return None
    if isinstance(sources, (str, Path, Mapping)):
        sources = [sources]
    out: dict[str, str] = {}
    for src in sources:
        doc = json.loads(Path(src).expanduser().read_text(encoding="utf-8")) if isinstance(src, (str, Path)) else dict(src)
        if not isinstance(doc, Mapping):
            raise AnalysisError(f"store pins {src}: not a JSON object")
        for k, v in doc.items():
            if str(k).startswith("_"):
                continue
            v = str(v).lower()
            if len(v) < 7 or any(ch not in "0123456789abcdef" for ch in v):
                raise AnalysisError(f"store pins: {k}: {v!r} is not a tree sha (>= 7 hex)")
            if str(k).count("/") != 2:
                raise AnalysisError(f"store pins: {k!r} is not a leg key <model>/<cell>/<mode>[@view]")
            prev = out.get(str(k))
            if prev is not None and not (prev.startswith(v) or v.startswith(prev)):
                raise AnalysisError(f"store pins: {k} is pinned twice with different trees ({prev!r} vs {v!r})")
            out[str(k)] = max((prev or ""), v, key=len)        # two spellings of ONE tree: keep the longer
    return out


def admitted_pin_problem(leg_key: str, tree: Any, pins: Mapping[str, str]) -> tuple[str | None, str | None]:
    """``(pin, problem)`` for one leg under per-leg pins: no entry for the leg, or a producer tree that is not its admitted
    pin (a 40-hex tree or a >= 7-hex prefix of it), each refused by name."""
    want = pins.get(leg_key)
    if want is None:
        return None, f"no admitted pin for {leg_key} in the store-pins doc (the leg is not part of the admitted set)"
    tree = str(tree or "")
    if not tree or not (tree == want or (len(want) >= 7 and tree.startswith(want))):
        return want, f"producer tree {tree[:8] or None!r} is not the admitted pin {want[:8]!r} for {leg_key} (--store-pins)"
    return want, None


class GenRoots:
    """The wave roots the legs are read from (plan P5 / B6): one or more ``--gen-root PATH[=PIN]``. A bare root is bound to
    the store pin under the multi-tree rule (a leg under it may be from the pin's tree or any declared tree); a PINNED root
    binds every leg under it to that one admitted tree (``--sha`` or an ``--also-sha``). A row's out root must exist under
    exactly ONE root: two is the same logical leg under two roots (a filesystem union) and refuses by name."""

    def __init__(self, roots: Sequence[tuple[Path | str, str | None]], *, primary_pin: str, admitted: Sequence[str] = ()) -> None:
        problem = pin_problem(primary_pin)
        if problem:
            raise AnalysisError(problem)
        self.primary_pin = str(primary_pin)
        self.admitted: tuple[str, ...] = tuple(admitted)
        self.entries: list[dict[str, Any]] = []
        seen: set[Path] = set()
        for path, pin in roots:
            p = Path(path).expanduser()
            if p in seen:
                raise AnalysisError(f"--gen-root {p} is named twice")
            seen.add(p)
            if pin is None:
                self.entries.append({"path": p, "pin": self.primary_pin, "pinned": False})
                continue
            why = pin_problem(pin)
            if why:
                raise AnalysisError(f"--gen-root {p}={pin}: {why}")
            primary = self.primary_pin.startswith(pin) or pin.startswith(self.primary_pin)
            if not primary and not any(t.startswith(pin) for t in self.admitted):
                raise AnalysisError(f"--gen-root {p}={pin}: the pin is neither --sha {self.primary_pin} nor a declared --also-sha tree "
                                    f"{list(self.admitted)} — an unadmitted producer tree")
            self.entries.append({"path": p, "pin": pin, "pinned": True})
        if not self.entries:
            raise AnalysisError("at least one --gen-root is required")

    @classmethod
    def single(cls, path: Path | str, pin: str) -> GenRoots:
        return cls([(path, None)], primary_pin=pin)

    @classmethod
    def of(cls, gen_root: GenRoots | Path | str | Sequence[str], *, primary_pin: str, admitted: Sequence[str] = ()) -> GenRoots:
        """The ONE dispatch both analyses' ``analyze`` and ``collect_leg`` take (B6): a ``GenRoots`` as is; a bare Path /
        str (the published call) as the single root bound to ``primary_pin``; else ``PATH[=PIN]`` specs (the repeatable
        CLI form), parsed and admitted against ``admitted``."""
        if isinstance(gen_root, cls):
            return gen_root
        if isinstance(gen_root, (str, Path)):
            return cls.single(gen_root, primary_pin)
        return cls([parse_gen_root_spec(str(s)) for s in gen_root], primary_pin=primary_pin, admitted=admitted)

    @property
    def declared(self) -> bool:
        """More than one root, or any pinned root: the multi-tree admission is in play (B2 / B9)."""
        return len(self.entries) > 1 or any(e["pinned"] for e in self.entries)

    @property
    def primary(self) -> Path:
        for e in self.entries:
            if not e["pinned"] or self.primary_pin.startswith(e["pin"]) or e["pin"].startswith(self.primary_pin):
                return e["path"]
        return self.entries[0]["path"]

    def resolve(self, out_root: str) -> tuple[dict[str, Any] | None, str | None]:
        """``(entry, problem)``: the one root holding ``out_root``; ``(None, None)`` when none does (a missing leg);
        ``(None, why)`` when two or more do (the duplicate logical leg)."""
        hits = [e for e in self.entries if (e["path"] / out_root).is_dir()]
        if len(hits) > 1:
            return None, (f"out root {out_root!r} exists under {len(hits)} gen roots ({[str(e['path']) for e in hits]}) — the same "
                          "logical leg under two roots (a filesystem union); pull each wave into its own root")
        return (hits[0] if hits else None), None

    def record(self) -> list[dict[str, Any]]:
        return [{"path": str(e["path"]), "pin": e["pin"], "pinned": bool(e["pinned"])} for e in self.entries]


class Context:
    """Everything a leg's reading is held to: the three sampling tables, the
    render facts, the object pins, the served model pins, and the STORE PIN
    every producer tree is held to (plus the declared trees a producer may
    be from instead)."""

    def __init__(self, *, sampling_rows: Mapping[str, Mapping[str, Mapping[str, Any]]],
                 render_facts: Mapping[str, Mapping[str, Any]], object_pins: Mapping[Any, str],
                 model_pins: Mapping[str, tuple[str, str]], store_pin: str, also_trees: Sequence[str] = (),
                 declared: bool | None = None,
                 anomaly_classes: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
                 store_pins: Mapping[str, str] | None = None) -> None:
        problem = pin_problem(store_pin)
        if problem:
            raise AnalysisError(problem)
        for sha in also_trees:
            if not isinstance(sha, str) or not _HEX40.match(sha):
                raise AnalysisError(f"declared producer tree {sha!r} is not a 40-hex commit")
        self.store_pin = str(store_pin)
        self.also_trees: tuple[str, ...] = tuple(also_trees)
        #: per-LEG store pins (--store-pins): ``{<model>/<cell>/<mode>[@view]: tree}`` — every leg admitted by
        #: ITS OWN producer tree; None = the one-pin admission (--sha [+ --also-sha]). Pins and also-trees are two admissions of
        #: the same legs and never coexist (analyze refuses)
        self.store_pins: dict[str, str] | None = dict(store_pins) if store_pins is not None else None
        self.sampling_rows = sampling_rows
        self.render_facts = render_facts
        #: keyed (model_key, cell) — F11b: every fitted arm has its own expected object (a legacy {model: sha} is admitted)
        self.object_pins = normalise_object_pins(object_pins)
        self.model_pins = dict(model_pins)
        #: B9: cross-tree comparisons are admitted only under a DECLARED tree set (also trees, pinned gen roots, or per-leg pins)
        self.declared = (bool(also_trees) or store_pins is not None) if declared is None else bool(declared)
        #: ``{leg key: {item id: class record}}`` a run WITHOUT eval logs takes its split-anomaly classes from (the report
        #: it is required to match); empty when the logs are read or no report is matched
        self.anomaly_classes: dict[str, Mapping[str, Mapping[str, Any]]] = dict(anomaly_classes or {})

    def row_for(self, model_key: str, mode: str) -> Mapping[str, Any]:
        table = self.sampling_rows.get(mode)
        if table is None:
            raise AnalysisError(f"sampling mode {mode!r} has no table (modes: {sorted(self.sampling_rows)})")
        row = table.get(model_key)
        if row is None:
            raise AnalysisError(f"sampling mode {mode!r} names no row for {model_key!r}")
        return row

    def fact_for(self, family: str) -> Mapping[str, Any]:
        fact = self.render_facts.get(family)
        if fact is None:
            raise AnalysisError(f"family {family!r} has no render fact (families: {sorted(self.render_facts)})")
        return fact


def _same_number(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    return a == b


def sampling_row_problem(manifest: Mapping[str, Any], summary: Mapping[str, Any], *, model_key: str,
                         mode: str, row: Mapping[str, Any]) -> str | None:
    """The manifest's sampling block equals ITS mode's literal row — the block
    names the mode and the model key; its numeric members are EXACTLY the
    members the row names (a key the row does not name refuses; a key it names
    but the block lacks refuses; a value that differs refuses); the summary's
    three numbers and the hashed identity's numbers agree with it."""
    block = _mapping(manifest.get(C.MANIFEST_KEY_SAMPLING))
    if not block:
        return f"the manifest carries no {C.MANIFEST_KEY_SAMPLING} block — the sampling row cannot be vouched for"
    if block.get("mode") != mode:
        return (f"sampling block mode {block.get('mode')!r} != the row's mode {mode!r} — another sampling row's "
                "store under this leg's out root (never pooled)")
    if block.get("model_key") != model_key:
        return f"sampling block model_key {block.get('model_key')!r} != {model_key!r}"
    want = _numeric_row(row)
    got = {f: block[f] for f in ROW_NUMERIC_FIELDS if f in block and block[f] is not None}
    extra = sorted(set(got) - set(want))
    if extra:
        return (f"sampling block carries {extra} = {[got[k] for k in extra]!r}, which the {mode!r} row for "
                f"{model_key!r} does not name — a differently sampled instrument")
    missing = sorted(set(want) - set(got))
    if missing:
        return (f"sampling block lacks {missing} (the {mode!r} row for {model_key!r} names {want}) — a differently "
                "sampled instrument")
    for f in want:
        if not _same_number(got[f], want[f]):
            return f"sampling block {f} {got[f]!r} != the {mode!r} row's {want[f]!r} for {model_key!r}"
    for f in SUMMARY_ROW_FIELDS:
        if not _same_number(summary.get(f), want.get(f)):
            return f"summary {f} {summary.get(f)!r} != the {mode!r} row's {want.get(f)!r} — the summary and the row disagree"
    tinycap = _mapping(manifest.get(C.MANIFEST_KEY_TINYCAP))
    for f in want:
        if f in tinycap and not _same_number(tinycap.get(f), want[f]):
            return f"identity block {f} {tinycap.get(f)!r} != the sampling block's {want[f]!r} — the hashed identity disagrees"
    for f in SUMMARY_ROW_FIELDS:
        if f not in tinycap:
            return f"identity block carries no {f} — the sampling identity was not hashed"
    return None


def _served_positions(v: Any) -> list[str] | None:
    if isinstance(v, str):
        return [v]
    if isinstance(v, (list, tuple)) and v and all(isinstance(x, str) for x in v):
        return list(v)
    return None


def _positions_ok(v: Any, expect: str) -> bool:
    """A served-positions value is exactly the expected non-header-located mode
    (a header-located or mixed set fails closed)."""
    pos = _served_positions(v)
    return pos is not None and not any(p in C.HEADER_LOCATED_MODES for p in pos) and set(pos) == {expect}


def native_served_positions_stamp_problem(manifest: Mapping[str, Any], *, served: Any, harness: str,
                                          gen_root: Path | str | None) -> str | None:
    """Why a served-positions value in the LENS-named field (``route.lens_served_positions``) of a NATIVELY served leg
    is NOT the standup record's engine-neutral fact — ``None`` when it is. The driver's standup lookup reads the routing
    table's engine-neutral ``served_positions`` map first (``served_capability_run._standup_served_positions``) and writes
    whatever it finds into the lens-named field on every engine (``served_capability_run.py`` 6315-6316), so a native
    leg's ``['all']`` there is the NEUTRAL map's value under a lens-era name — not a lens fact and not a torn record. The
    stamp is admitted ONLY as that record's value: the pulled ``<gen_root>/serving/<jobdir>/math500/routing_table.json``
    (located through the manifest's own ``routing_table`` path, :func:`standup_dir_of`) must carry
    ``served_positions[<harness>]`` (:data:`contracts.SERVED_POSITIONS_KEY` — the routes[] attestation admits no stamp,
    and neither does the engine's name) equal to the stamp. No ``gen_root``, no record, no neutral entry, or a differing
    value refuses, naming what differs."""
    modes = _served_positions(served)
    if modes is None:
        return f"{served!r} is not a served-positions value (a mode, or a list of modes)"
    if gen_root is None:
        return ("no wave root was given, so the stamp cannot be held to the standup record's engine-neutral "
                f"{C.SERVED_POSITIONS_KEY} map")
    d, jobdir, problem = standup_dir_of(manifest, gen_root=gen_root, task_subdir=LAUNCH_STANDUP_TASK_SUBDIR)
    if problem or d is None:
        return f"the stamp cannot be held to the standup record — {problem}"
    path = d / LAUNCH_ROUTING_TABLE
    if not path.is_file():
        return (f"the stamp cannot be held to the standup record — {LAUNCH_STANDUP_DIR_PARENT}/{jobdir}/{LAUNCH_STANDUP_TASK_SUBDIR}/"
                f"{LAUNCH_ROUTING_TABLE} is not in the pull")
    try:
        table = _mapping(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError) as e:
        return f"the stamp cannot be held to the standup record — {jobdir}/{LAUNCH_STANDUP_TASK_SUBDIR}/{LAUNCH_ROUTING_TABLE} is unreadable ({e})"
    neutral = C.recorded_served_positions_of_cell(table, harness)
    if neutral is None:
        return (f"{jobdir}/{LAUNCH_STANDUP_TASK_SUBDIR}/{LAUNCH_ROUTING_TABLE} carries no {C.SERVED_POSITIONS_KEY}[{harness!r}] entry — the "
                "engine-neutral map is the only record a native stamp is held to (never the routes[] attestation, never the engine's name)")
    if modes != list(neutral):
        return (f"the stamp {modes!r} != {jobdir}/{LAUNCH_STANDUP_TASK_SUBDIR}/{LAUNCH_ROUTING_TABLE}[{C.SERVED_POSITIONS_KEY}][{harness!r}] "
                f"{list(neutral)!r}")
    return None


def _header_ids_differ(block: Any, fact: Mapping[str, Any]) -> bool:
    """Whether an ``{off: [...], on: [...]}`` header-id block is NOT the family fact's tuples."""
    ids = _mapping(block)
    return [list(ids.get(C.THINKING_OFF) or []), list(ids.get(C.THINKING_ON) or [])] != [fact["off_ids"], fact["on_ids"]]


def _header_ids_by_arm(block: Any) -> dict[str, list[int]]:
    """An ``{arm: [ids]}`` header block normalised for comparison (str keys, int ids); ``{}`` for a non-mapping."""
    return {str(k): [int(x) for x in v] for k, v in _mapping(block).items()}


def _single_render_ids(fact: Mapping[str, Any], *, arm: str = C.THINKING_NOT_AVAILABLE) -> dict[str, list[int]]:
    """The NO-SWITCH class's expected header block (llama-paper-rows P18): the family's ONE render (the fact records it
    under both id fields; ``off_ids`` read) under the one arm. The twin of :func:`_header_ids_differ`'s two-arm shape,
    shared by the GSM8K validators and :func:`_render_certified_problems`."""
    return {arm: [int(x) for x in fact["off_ids"]]}


def _positive_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v >= 1


#: B5 (i′): the sha256[:16] of each prompted cell's SI text as the driver files it (`resolved_config.system_sha`) — pinned
#: here and held equal to the registry's text by test (`registry_adapter._traintest_si_text_bytes`), never re-read at run
#: time (the figure venv need not import the registry). safety_refusal verified against the published safety leg; the
#: constitution digest is F3's (= `prompted_prefix_hashes.constitution` in every campaign manifest)
PROMPTED_SI_PREFIX_SHA16: Mapping[str, str] = {"safety_refusal": "ac3a58f61d53cf24", "si_constitution": "21e06f482a6559b3"}
#: the serving-identity markers of a STEERED route — the lens hook and the native LoRA adapter
#: (``proxy_vllm+lens+…`` / ``proxy_vllm+lora+…``); a passthrough identity carries neither
STEERED_SERVING_MARKERS = ("+lens+", "+lora+")
_STEERED_PATH_BY_MARKER = {"+lens+": "lens", "+lora+": "lora"}


def steered_path_of(identity: Any) -> str | None:
    """The SERVING PATH a route's identity names — ``"lens"``, ``"lora"``, or ``None`` for an
    identity that steers nothing (a passthrough base / prompted leg, or a non-string). Judgement
    call (native-serving-default W-C): None rather than a refusal, because every caller here asks
    it of an arm whose class it has already decided — the base and prompted arms legitimately have
    no steered path, and the class-vs-marker disagreement is what :func:`arm_contract_problem`
    refuses. An identity carrying BOTH markers is torn and refuses (there is no such route)."""
    if not isinstance(identity, str):
        return None
    found = [m for m in STEERED_SERVING_MARKERS if m in identity]
    if len(found) > 1:
        raise ValueError(f"serving identity {identity!r} carries {found} — two steering paths in one "
                         "identity; a torn record, refusing to name a path")
    return _STEERED_PATH_BY_MARKER[found[0]] if found else None


def prompted_si_of(cell: str) -> str | None:
    """The SI id of a ``prompted__<si>`` cell (``None`` for any other cell)."""
    return cell[len(C.PROMPTED_CELL_PREFIX):] if cell.startswith(C.PROMPTED_CELL_PREFIX) else None


def route_class_of(cell: str) -> str:
    """The route class of a cell BY THE CELL GRAMMAR (plan P5 (i′)): the bare cell, a ``prompted__<si>`` cell, else a
    STEERED (fitted) cell — one rule for a registered arm and for an appendix cell the GSM8K matrix carries. The
    fitted class says nothing about WHICH engine steers it: a lens hook and a native LoRA adapter are one class,
    separated by :func:`steered_path_of` over the store's own serving identity."""
    if cell == CELL_BARE:
        return ROUTE_BASE
    if cell.startswith(C.PROMPTED_CELL_PREFIX):
        return ROUTE_PROMPTED
    return ROUTE_STEERED


def projection_identity_fields(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """B5: the manifest fields the shared-field / server-settings projection and the arm contract read, as a leg's
    ``identity`` block carries them (strings and bools only, B1) — ONE spelling for both instruments' ``collect_leg``."""
    tinycap = _mapping(manifest.get(C.MANIFEST_KEY_TINYCAP))
    prov = _mapping(manifest.get(C.MANIFEST_KEY_PROVENANCE))
    prov_serving = _mapping(prov.get("serving"))
    thinking = _mapping(tinycap.get("thinking"))
    served = _mapping(tinycap.get("served"))
    manifest_thinking = _mapping(manifest.get(C.MANIFEST_KEY_THINKING))
    header_ids = tinycap.get(IDENTITY_KEY_THINKING_HEADER_IDS)
    return {
        "chat_template_hash": prov_serving.get("chat_template_hash"),
        "serving_identity": _mapping(manifest.get(C.MANIFEST_KEY_ROUTE)).get("serving_identity"),
        "resolved_model_id": prov_serving.get("resolved_model_id"), "resolved_revision": prov_serving.get("resolved_revision"),
        "requested_model": prov_serving.get("requested_model"), "requested_revision": prov_serving.get("requested_revision"),
        "engine": manifest.get("engine"), "serving_path": manifest.get("serving_path"), "si_id": prov.get("si_id"),
        "system_sha": _mapping(manifest.get("resolved_config")).get("system_sha"),
        "thinking_mode": thinking.get("mode"), "thinking_render": thinking.get("render"),
        "enable_thinking": manifest_thinking.get("enable_thinking"),
        # code review C2: the further RECORDED server settings the projection compares — the manifest's own parser field
        # beside the served one, the modality flags, and the render regime's header ids (a JSON string: no numeric leaf, B1)
        "reasoning_parser_thinking": manifest_thinking.get("reasoning_parser"),
        "tool_calls": served.get("tool_calls"), "agent_loop_entry": served.get(C.AGENT_LOOP_ENTRY_KEY),
        "thinking_header_ids_text": json.dumps(header_ids, sort_keys=True) if header_ids is not None else None,
    }


def arm_contract_problem(manifest: Mapping[str, Any], *, row: Mapping[str, Any]) -> str | None:
    """B5 (i′) / plan P5: why the store does not satisfy ITS OWN ARM'S contract — validated for what the row is, never
    compared with another arm. ``none``: a passthrough route with no object; ``prompted__<si>``: a passthrough route
    whose ``provenance.si_id`` is the cell's SI and whose ``resolved_config.system_sha`` is that SI text's digest
    (:data:`PROMPTED_SI_PREFIX_SHA16`); a lens arm: the lens route (the object pin itself is held by the caller). Every
    arm: the COMPLETE served geometry — window, concurrency S and client width — satisfies the row's own resolved
    contract (:func:`geometry_contract_problem`): the window equals the ROW's ``geometry_L`` when the matrix row
    carries one (the constitution rows' larger window is validated, not compared), S is the model budget's S at that
    window as the standup resolved and attested it, the client width equals S."""
    cell = str(row["cell"])
    route_class = route_class_of(cell)
    route = _mapping(manifest.get(C.MANIFEST_KEY_ROUTE))
    identity = route.get("serving_identity")
    if not isinstance(identity, str) or not identity:
        return f"manifest route.serving_identity {identity!r} — the route class cannot be verified"
    try:
        steered_path = steered_path_of(identity)
    except ValueError as e:
        return str(e)
    is_steered = steered_path is not None
    if (route_class == ROUTE_STEERED) != is_steered:
        return (f"route.serving_identity {identity!r} is {'a steered' if is_steered else 'not a steered'} route on the "
                f"{route_class!r} arm {cell!r} — the arm's own contract")
    if route_class == ROUTE_STEERED:
        # WHERE the steering acts, by engine: the lens re-locates a response mask, a native LoRA adapter
        # applies at every token (the driver's `steer_positions`). The key is
        # ABSENT on a pre-change lens store — admitted, never guessed.
        want_positions = {"lens": "response", "lora": "all"}[steered_path]
        got_positions = manifest.get("steer_positions")
        if got_positions is not None and got_positions != want_positions:
            return (f"steer_positions {got_positions!r} on a {steered_path!r}-served arm {cell!r} — a lens route steers the "
                    f"response mask ('response') and a native LoRA adapter every position ('all'); expected {want_positions!r}")
        if steered_path == "lora":
            # the STORED-RECORD contract through THE ONE resolver: a native store records
            # serving_engine == lora_native and reduction_mode == not_applicable (it has no lens hook), and the
            # two are held to each other there. Read only on a `+lora+` manifest, so every lens store reads
            # exactly as before.
            reading = C.read_store_reduction_mode(manifest, where=f"the {cell!r} leg's manifest")
            if reading.refusal is not None:
                return reading.refusal
            if reading.serving_engine != C.SERVING_ENGINE_LORA_NATIVE:
                return (f"route.{C.SERVED_ENGINE_FIELD} {reading.serving_engine!r} on the natively served arm {cell!r} "
                        f"(`+lora+`) — a native store records {C.SERVING_ENGINE_LORA_NATIVE!r}; without it the store's "
                        "engine cannot be told from a lens-era one (the passthrough identity is the same on both)")
            if reading.mode != C.REDUCTION_MODE_NOT_APPLICABLE:
                return (f"route.{C.REDUCTION_MODE_FIELD} {reading.mode!r} on the natively served arm {cell!r} — the native "
                        f"engine has no lens hook, so its store records {C.REDUCTION_MODE_NOT_APPLICABLE!r}")
    prov = _mapping(manifest.get(C.MANIFEST_KEY_PROVENANCE))
    if route_class != ROUTE_STEERED and prov.get("artifact_sha256") is not None:
        return f"provenance.artifact_sha256 {prov.get('artifact_sha256')!r} on the unsteered arm {cell!r} — an object was served"
    if route_class == ROUTE_PROMPTED:
        si = prompted_si_of(cell)
        if prov.get("si_id") != si:
            return f"provenance.si_id {prov.get('si_id')!r} != the prompted cell's SI {si!r}"
        want = PROMPTED_SI_PREFIX_SHA16.get(str(si))
        got = _mapping(manifest.get("resolved_config")).get("system_sha")
        if want is None:
            return f"the prompted SI {si!r} has no pinned prefix digest (PROMPTED_SI_PREFIX_SHA16) — its text cannot be verified"
        if got != want:
            return f"resolved_config.system_sha {got!r} != the {si!r} SI text's digest {want} — another system prompt was served"
    elif prov.get("si_id") not in (None, "") and route_class == ROUTE_BASE:
        return f"provenance.si_id {prov.get('si_id')!r} on the bare arm — a system prompt was served"
    return geometry_contract_problem(manifest, row=row)


#: the standup's resolved-geometry record as the driver copies it into the manifest (`sisteer.capeval.serving.geometry.
#: ResolvedGeometry.as_dict`, read off endpoint.json): the sources the row contracts — the window from the table ENTRY, S
#: from the MODEL's budget (never an override, never the Llama-sized table S)
GEOMETRY_WINDOW_SOURCE_TABLE, GEOMETRY_SEQS_SOURCE_MODEL_BUDGET = "table", "model_budget"
#: what the producer records for a geometry the §1.8 adjustment lowered (``geometry.seqs_source``:
#: ``native_standup.lora_kernel_ceiling_geometry`` replaces the source when the kernel step fires)
GEOMETRY_SEQS_SOURCE_OVERRIDE = "override"
#: ... and the WINDOW source the same kernel lowering stamps: it re-resolves the geometry with the UNCHANGED table window
#: passed explicitly beside the lowered S (``native_standup._resolve_geometry(eval, L, lowered_S, ...)``), and
#: ``resolve_geometry``'s ``pick()`` marks every EXPLICIT factor ``override`` whether or not it equals the table's. So on a
#: natively served, kernel-lowered leg this spelling does NOT mean "another window" — :func:`_override_window_problem`
#: derives the table's window and admits the stamp only when the served window IS it
GEOMETRY_WINDOW_SOURCE_OVERRIDE = "override"
#: ... and for a geometry the KV step re-stated at the S the container's KV cache backs (``geometry.kv_shrunk_geometry``
#: -> ``geometry.KV_DERIVED_SOURCE``; the native core's ``kv_derived_geometry`` and capctrl's re-launch both write it).
#: A literal here (this module imports the geometry leaf lazily); held equal to the constant by test
GEOMETRY_SEQS_SOURCE_KV_DERIVED = "kv_derived"


def _budget_seqs(model_key: str, max_model_len: int, budget: int) -> tuple[int | None, str, str | None]:
    """The S the model's budget buys at window L under the RESOLVER's own rule (``ModelBudget.seqs_at``: ``budget // L``,
    capped for a hybrid-recurrent family at :data:`HYBRID_SEQS_CAP` and at the window's measured ceiling) — evaluated at the
    leg's RECORDED budget, never the live table's (a published leg served under a provisional budget contracts THAT budget's
    S). Returns ``(S, rule text, problem)``; the problem names a model the geometry table does not know."""
    quotient = max(1, int(budget) // int(max_model_len))
    rule = f"budget {int(budget):,} // L {int(max_model_len):,}"
    try:
        from sisteer.capeval.serving import geometry as SG
    except Exception as e:  # pragma: no cover — the analysis venv always carries the serving package
        return None, rule, f"the geometry module is unavailable ({e}) — the row's S rule cannot be resolved"
    try:
        entry = SG.budget_for(model_key).entry
    except SG.UnknownModelKey:
        return None, rule, f"model {model_key!r} is unknown to the geometry table — the row's S rule cannot be resolved"
    if not entry.hybrid:
        return quotient, rule, None
    s = min(quotient, SG.HYBRID_SEQS_CAP)
    ceiling = entry.window_ceiling(int(max_model_len))
    if ceiling is not None:
        s = min(s, int(ceiling.max_num_seqs))
    return s, rule + f", the hybrid cap {SG.HYBRID_SEQS_CAP}" + (f", the window ceiling {ceiling.max_num_seqs}" if ceiling is not None else ""), None


def _int_field(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _override_window_problem(geom: Mapping[str, Any], *, adjustments: Sequence[Any], L: int, S: int, SG: Any) -> str | None:
    """Why a ``geometry.window_source`` that is not ``'table'`` is NOT the table entry's window under a recorded §1.8
    lowering — ``None`` when it is. A natively served leg that hits the LoRA kernel's int32 index ceiling re-resolves its
    geometry with the UNCHANGED table window passed explicitly beside the lowered S
    (``native_standup.lora_kernel_ceiling_geometry`` -> ``_resolve_geometry(eval, L, lowered_S, ...)``), and
    ``resolve_geometry``'s ``pick()`` stamps every explicit factor ``override`` — so the block reads ``window_source:
    override`` although L IS the table entry's, and the window rule refused the leg as served at another window (the
    pulled qwen35_4b `arditi_allpos` thinking legs: 10,752 x 48 under ``[kernel 69 -> 48]``, both sources ``override``).

    The stamp is never trusted. The window is admitted as the table's ONLY when all of: (i) the block records at least
    one adjustment and every step's kind is one the §1.8 validator knows (``SG.ADJUSTMENT_KINDS``: the kernel ceiling,
    the KV shrink); (ii) ``geometry.eval`` names a table entry and the served window equals THAT entry's window as
    ``SG.resolve_geometry(eval, model_key=...)`` — the standup's own identity resolution — derives it; (iii) the served
    S is the LAST step's ``to_S``. The caller then falls through to the §1.8 validation unchanged, which re-derives every
    step from the model's pinned width, L and the identity S. A window that differs from the table's is a REAL override
    and refuses by the two numbers; an ``override`` with no adjustment keeps today's refusal byte for byte; any other
    source (``fallback``, a measurement footprint, ``None``) refuses exactly as before."""
    source = geom.get("window_source")
    today = f"geometry.window_source {source!r} != {GEOMETRY_WINDOW_SOURCE_TABLE!r} — the served window was not the table entry's"
    if source != GEOMETRY_WINDOW_SOURCE_OVERRIDE or not adjustments:
        return today
    kinds: list[Any] = []
    for i, adj in enumerate(adjustments):
        kind = adj.get("kind") if isinstance(adj, Mapping) else None
        if kind not in SG.ADJUSTMENT_KINDS:
            return (f"{today}; geometry.{SG.ADJUSTMENTS_KEY}[{i}].kind {kind!r} is not one of {list(SG.ADJUSTMENT_KINDS)}, so no "
                    "recorded adjustment explains the override stamp")
        kinds.append(kind)
    eval_name, model_key = geom.get("eval"), geom.get("model_key")
    if not isinstance(eval_name, str) or not eval_name:
        return f"{today}; geometry.eval {eval_name!r} names no table entry, so the served window {L:,} cannot be held to one"
    try:
        table = SG.resolve_geometry(eval_name, model_key=model_key if isinstance(model_key, str) else None)
    except (KeyError, ValueError) as e:
        return (f"{today}; geometry.eval {eval_name!r} ({model_key!r}) resolves to no table entry ({e}), so the served window "
                f"{L:,} cannot be held to one")
    if table.window_source != GEOMETRY_WINDOW_SOURCE_TABLE or L != int(table.max_model_len):
        return (f"geometry.window_source {source!r}: the served window {L:,} != the table entry's {int(table.max_model_len):,} for "
                f"{eval_name!r} ({model_key!r}) — a real override beside adjustments {kinds}, not the table entry's window")
    last = adjustments[-1]
    to_S = _int_field(last.get("to_S")) if isinstance(last, Mapping) else None
    if to_S is None or S != to_S:
        return (f"geometry.window_source {source!r} beside adjustments {kinds}: the served max_num_seqs {S} != the last step's "
                f"to_S {last.get('to_S') if isinstance(last, Mapping) else None!r} — the override stamp is explained only by a "
                "lowering that ends at the served S")
    return None


def geometry_contract_problem(manifest: Mapping[str, Any], *, row: Mapping[str, Any]) -> str | None:
    """Why the leg's COMPLETE served geometry does not satisfy ITS OWN row's contract (plan P5 (i′)): the window, the
    concurrency S and the client width, validated per row and never compared with another arm — the constitution
    rows' larger window is what THEIR row contracts. The arm contract used to check ``max_model_len`` only, while the
    server projection excluded the geometry on the premise that this check covered it, so a leg served (or a record
    edited) at S = 1 beside a 33-wide row read `server_settings_equal: true` and pooled.

    The row pins the window and the table entry (``geometry_L`` / ``geometry_eval``, when the matrix row carries them);
    the standup RESOLVES S from the model's budget at that window and the driver copies the resolution into the
    manifest's ``geometry`` block (``ResolvedGeometry.as_dict``: eval, max_model_len, max_num_seqs, budget, sources,
    ``identity`` = its own attestation that S is the budget's S at L). So the contract is: the served window is the
    row's; the geometry block names the row's entry, window and model, sourced from the table and the model budget,
    within budget, ``batched_tokens`` = S × L, S equal to the budget's S at L under the resolver's rule at the RECORDED
    budget (:func:`_budget_seqs`); the served ``max_num_seqs`` is that S; the client ``concurrency`` (and the manifest's
    own) equals the served S — the driver's rule. The launch echo is reconciled with the same served block in
    :func:`launch_attestation`. The window rule reads ``window_source`` with ONE carve (:func:`_override_window_problem`):
    a natively served leg whose kernel lowering re-stamped the unchanged table window ``override`` is held to the TABLE's
    window by derivation and admitted only under a known adjustment that ends at the served S."""
    tinycap = _mapping(manifest.get(C.MANIFEST_KEY_TINYCAP))
    served = _mapping(tinycap.get("served"))
    geometry_l = row.get("geometry_L")
    if geometry_l is not None and served.get("max_model_len") != int(geometry_l):
        return (f"served max_model_len {served.get('max_model_len')!r} != the row's geometry L {int(geometry_l)} ({row.get('geometry_eval')!r}) — "
                "the leg was served at another window than its row contracts")
    geom = _mapping(manifest.get(C.MANIFEST_KEY_GEOMETRY))
    if not geom:
        return "manifest carries no `geometry` block (the standup's resolved geometry) — the served concurrency cannot be held to the row's contract"
    L, S, budget = _int_field(geom.get("max_model_len")), _int_field(geom.get("max_num_seqs")), _int_field(geom.get("budget"))
    if L is None or S is None or budget is None:
        return (f"geometry.max_model_len / max_num_seqs / budget {geom.get('max_model_len')!r} / {geom.get('max_num_seqs')!r} / "
                f"{geom.get('budget')!r}: not the standup's integer resolution — the served concurrency cannot be held to the row's contract")
    if geometry_l is not None and L != int(geometry_l):
        return f"geometry.max_model_len {L} != the row's geometry L {int(geometry_l)} — the standup resolved another window than the row contracts"
    if row.get("geometry_eval") is not None and geom.get("eval") != row.get("geometry_eval"):
        return f"geometry.eval {geom.get('eval')!r} != the row's geometry entry {row.get('geometry_eval')!r} — the leg was served under another table entry"
    if row.get("model_key") is not None and geom.get("model_key") != row.get("model_key"):
        return f"geometry.model_key {geom.get('model_key')!r} != the row's model {row.get('model_key')!r} — the launch was sized for another model's budget"
    from sisteer.capeval.serving import geometry as SG   # torch-free leaf; the §1.8 contract

    adjustments = SG.geometry_adjustments(geom)
    if geom.get("window_source") != GEOMETRY_WINDOW_SOURCE_TABLE:
        # the window rule, with ONE carve: a natively served leg whose kernel lowering re-stamped the UNCHANGED table
        # window `override` is admitted iff the served window IS the table's (derived, never read off the stamp) and a
        # known adjustment ending at the served S explains the stamp — then the §1.8 branch below re-derives the steps
        why = _override_window_problem(geom, adjustments=adjustments, L=L, S=S, SG=SG)
        if why is not None:
            return why
    if adjustments:
        expect, rule, problem = _budget_seqs(str(geom.get("model_key")), L, budget)
        if problem:
            return problem
        # native-serving-default §1.8: the served S may sit BELOW the budget's S only as a VALIDATED
        # ordered adjustment — the LoRA kernel's int32 index ceiling (gemma 121->107, nemotron 161->86
        # at 4864) and/or a KV shrink, each re-derived from the model's widest adapted module, L and
        # the identity S by the ONE acceptance rule every reader shares. "Any smaller S" never passes.
        why = SG.geometry_adjustment_problem(geom, model_key=(geom.get("model_key") if isinstance(geom.get("model_key"), str) else None),
                                             L=L, S_identity=expect, S_served=S)
        if why is not None:
            return f"geometry.{SG.ADJUSTMENTS_KEY}: {why} — the row's served concurrency contract"
        # the producer's spelling of WHERE the served S came from follows the LAST step, ONE spelling per chain (an
        # earlier rule admitted `model_budget` / `override` on every adjusted chain and `kv_derived` iff the chain
        # ended in KV, so a [kv] or [kernel, kv] record with correct arithmetic and a STALE source — the pre-shrink
        # `override`, or the unadjusted `model_budget` — was admitted as a torn record): a kernel lowering re-resolves the
        # geometry at the lowered S as `override` (native_standup.lora_kernel_ceiling_geometry — the native core
        # records a kernel-lowered S as an override), the KV step re-states it at the KV-derived S as `kv_derived`
        # (geometry.kv_shrunk_geometry; the native core's kv_derived_geometry and capctrl's re-launch), and
        # `model_budget` is the UNADJUSTED source only (the else branch below). The chain's kinds were validated just
        # above, so its last step is one of the two
        kinds = [a.get("kind") for a in adjustments if isinstance(a, Mapping)]
        last = kinds[-1] if kinds else None
        admitted = ((GEOMETRY_SEQS_SOURCE_KV_DERIVED,) if last == SG.ADJUSTMENT_KV
                    else (GEOMETRY_SEQS_SOURCE_OVERRIDE,) if last == SG.ADJUSTMENT_KERNEL else ())
        if geom.get("seqs_source") not in admitted:
            return (f"geometry.seqs_source {geom.get('seqs_source')!r} on an ADJUSTED geometry ({kinds}) != "
                    f"{' / '.join(repr(s) for s in admitted) or 'a known last step'} — the producer spells the source after the "
                    f"LAST step: {GEOMETRY_SEQS_SOURCE_OVERRIDE!r} when the kernel ceiling ends the chain, "
                    f"{GEOMETRY_SEQS_SOURCE_KV_DERIVED!r} when the KV step does; {GEOMETRY_SEQS_SOURCE_MODEL_BUDGET!r} is the "
                    "UNADJUSTED source; a stale source beside a validated chain is a torn record, never admitted")
        if geom.get("identity") is not False:
            return (f"geometry.identity {geom.get('identity')!r} beside a non-empty geometry.{SG.ADJUSTMENTS_KEY} — an "
                    "adjusted geometry is not the identity one")
    else:
        if geom.get("seqs_source") != GEOMETRY_SEQS_SOURCE_MODEL_BUDGET:
            return (f"geometry.seqs_source {geom.get('seqs_source')!r} != {GEOMETRY_SEQS_SOURCE_MODEL_BUDGET!r} — the served concurrency was not the "
                    "model budget's (an override or the Llama-sized table S is not the row's contract)")
        if geom.get("identity") is not True:
            return f"geometry.identity {geom.get('identity')!r}: the standup did not attest max_num_seqs as the budget's S at this window"
    if geom.get("over_budget") is not False or S * L > budget:
        return f"geometry.over_budget {geom.get('over_budget')!r}: {S} × {L} = {S * L:,} against the budget {budget:,}"
    if geom.get("batched_tokens") != S * L:
        return f"geometry.batched_tokens {geom.get('batched_tokens')!r} != max_num_seqs {S} × max_model_len {L} = {S * L:,}"
    if not adjustments:
        # the UNADJUSTED contract, in its original order: S is the budget's S at L
        expect, rule, problem = _budget_seqs(str(geom.get("model_key")), L, budget)
        if problem:
            return problem
        if S != expect:
            return f"geometry.max_num_seqs {S} != the budget's S {expect} at L {L:,} ({rule}) — the row's resolved concurrency contract"
    if served.get("max_model_len") != L:
        return f"served max_model_len {served.get('max_model_len')!r} != the standup's resolved window {L} (geometry.max_model_len)"
    if served.get("max_num_seqs") != S:
        return (f"served max_num_seqs {served.get('max_num_seqs')!r} != the standup's resolved S {S} (geometry.max_num_seqs) — the leg was "
                "served at another concurrency than its row contracts")
    if served.get("concurrency") != S:
        return f"served concurrency {served.get('concurrency')!r} != the served max_num_seqs {S} — the client width is the served concurrency (the driver's rule)"
    top = manifest.get(C.MANIFEST_KEY_CONCURRENCY)
    if top is not None and top != S:
        return f"manifest {C.MANIFEST_KEY_CONCURRENCY} {top!r} != the served max_num_seqs {S} — the client width is the served concurrency"
    return None


def identity_problem(manifest: Mapping[str, Any], summary: Mapping[str, Any], *, row: Mapping[str, Any],
                     ctx: Context, family: str) -> str | None:
    """Why the store under a leg's out root is NOT this leg's production run —
    ``None`` when it is. POSITIVE validation against the matrix row and the
    pinned inputs, in the order that names the most specific field first."""
    key, cell, mode, fitted = str(row["model_key"]), str(row["cell"]), str(row["sampling_mode"]), bool(row["fitted"])
    # math500-steer: the row's ARM selects the class (llama-paper-rows P18 — the GSM8K validators' rule, read here by import):
    # `not_available` is the NO-SWITCH class (a family whose template has no thinking switch: the hashed thinking.mode /
    # thinking_by_task at that literal, NO out_of_support flag, NO header ids, the served render `unspecified`); `on` reads
    # byte-identically to before
    arm = row_arm(row)
    no_switch = arm == C.THINKING_NOT_AVAILABLE
    fact = ctx.fact_for(family)
    if no_switch and fact.get("thinking_switch") is not False:
        return (f"the matrix row runs {arm!r} on family {family!r}, whose render fact carries a thinking switch — the no-switch class is "
                "for a switch-less family only (contracts.cert_class_for; P7)")
    if not no_switch and fact.get("thinking_switch") is False:
        return (f"the matrix row runs {arm!r} on the switch-less family {family!r} — there is no thinking-ON class for it; the row must "
                f"run {C.THINKING_NOT_AVAILABLE!r} (contracts.cert_class_for)")
    hf, rev = ctx.model_pins[key]
    if manifest.get("model_id") != hf:
        return f"manifest model_id {manifest.get('model_id')!r} != the battery pin {hf!r} for {key!r}"
    if manifest.get("model_revision") != rev:
        return f"manifest model_revision {manifest.get('model_revision')!r} != the battery pin {rev!r} for {key!r}"
    if manifest.get("cell_id") != cell:
        return f"manifest cell_id {manifest.get('cell_id')!r} != the leg's cell {cell!r}"
    spell = C.cell_spellings(cell, declared=False)
    route = _mapping(manifest.get(C.MANIFEST_KEY_ROUTE))
    vmodel = route.get("virtual_model")
    if vmodel != spell.vmodel:
        return f"manifest route virtual_model {vmodel!r} != the plain spelling {spell.vmodel!r} (no cell is declared here)"
    tinycap = _mapping(manifest.get(C.MANIFEST_KEY_TINYCAP))
    if tinycap.get(IDENTITY_KEY_DIAGNOSTIC):
        return (f"the store's identity carries {IDENTITY_KEY_DIAGNOSTIC}: {tinycap.get(IDENTITY_KEY_DIAGNOSTIC)!r} — a "
                "diagnostic smoke store (plan §5.2), never a production result")
    if tinycap.get(IDENTITY_KEY_VERSION) != IDENTITY_VERSION_EXPECTED:
        return (f"{IDENTITY_KEY_VERSION} {tinycap.get(IDENTITY_KEY_VERSION)!r} != {IDENTITY_VERSION_EXPECTED} (this "
                "experiment's identity version)")
    if tinycap.get(C.IDENTITY_KEY_MAX_TOKENS_BY_TASK) != {TASK: CAP}:
        return f"max_tokens_by_task {tinycap.get(C.IDENTITY_KEY_MAX_TOKENS_BY_TASK)!r} != {{{TASK!r}: {CAP}}}"
    if _mapping(tinycap.get(C.IDENTITY_KEY_THINKING_BY_TASK)).get(TASK) != arm:
        return f"thinking_by_task {tinycap.get(C.IDENTITY_KEY_THINKING_BY_TASK)!r} does not run {TASK!r} {arm}"
    if _mapping(tinycap.get("thinking")).get("mode") != arm:
        return f"identity thinking.mode {_mapping(tinycap.get('thinking')).get('mode')!r} != {arm!r}"
    if no_switch:
        # the no-switch arm's UNHASHED manifest facts (P7 / P17 / I11), the validators' one rule, on THIS task
        problem = validators_module()._no_switch_manifest_problem(manifest, fact=fact, family=family, task=TASK)
        if problem:
            return problem
    if C.IDENTITY_KEY_SERVE_OUT_OF_SUPPORT in tinycap:
        return (f"identity carries {C.IDENTITY_KEY_SERVE_OUT_OF_SUPPORT} = {tinycap[C.IDENTITY_KEY_SERVE_OUT_OF_SUPPORT]!r} "
                "— a DECLARED store; no cell of this experiment is declared")
    oos = tinycap.get(C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING, "absent")
    if fitted and not no_switch and oos is not True:
        return (f"a fitted job's identity must carry {C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING}: true (got {oos!r}) — "
                "not the authorised out-of-support store")
    if not fitted and oos != "absent":
        return (f"identity carries {C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING} = {oos!r} on a non-fitted cell — an "
                "in-support store carries no such key")
    if no_switch and oos != "absent":
        return (f"identity carries {C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING} = {oos!r} on a {arm} store — a family with no "
                "thinking switch has no ON render to be out of support of (P7); an in-support run carries no such key")
    retry = tinycap.get(C.IDENTITY_KEY_MATH500_RETRY_MAX_TOKENS, "absent")
    if retry != 0:
        return f"math500_retry_max_tokens {retry!r} != 0 (one pass at {CAP}; no retry)"
    if _mapping(tinycap.get(C.IDENTITY_KEY_SAMPLES_BY_TASK)).get(TASK) != 1:
        return f"samples_by_task {tinycap.get(C.IDENTITY_KEY_SAMPLES_BY_TASK)!r} is not k = 1 on {TASK!r}"
    problem = sampling_row_problem(manifest, summary, model_key=key, mode=mode, row=ctx.row_for(key, mode))
    if problem:
        return problem
    if no_switch:
        # the "present iff any task is ON" rule: a no-switch store hashes NO header ids (the one render is certified live,
        # predicate (A″)); a present key is a switched family's identity under a mislabel
        if IDENTITY_KEY_THINKING_HEADER_IDS in tinycap:
            return (f"identity carries {IDENTITY_KEY_THINKING_HEADER_IDS} = {tinycap.get(IDENTITY_KEY_THINKING_HEADER_IDS)!r} on a {arm} "
                    "store — the key is present iff a task runs thinking-ON (a family with no switch hashes none)")
    else:
        header_ids = _mapping(tinycap.get(IDENTITY_KEY_THINKING_HEADER_IDS))
        if _header_ids_differ(header_ids, fact):
            return (f"identity thinking_header_ids {dict(header_ids)!r} != the {family!r} render fact's tuples "
                    f"off {fact['off_ids']} / on {fact['on_ids']}")
    problem = arm_contract_problem(manifest, row=row)
    if problem:
        return problem
    if fitted:
        pin = ctx.object_pins.get(pin_key(row))
        if pin is None:
            return f"no object pin for {key!r} / {cell!r} (the pin file names no staged id for this arm)"
        digest = _mapping(manifest.get(C.MANIFEST_KEY_PROVENANCE)).get("artifact_sha256")
        if digest != pin:
            return f"provenance.artifact_sha256 {digest!r} != the pin {pin!r} for {key!r} / {cell!r} — another object was served"
        by_cell = _mapping(route.get(C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL))
        if by_cell.get(spell.harness) != pin:
            return (f"route.{C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL}[{spell.harness!r}] = "
                    f"{by_cell.get(spell.harness)!r} != the pin {pin!r}")
        tp = manifest.get(MANIFEST_KEY_TRANSFER_PROVENANCE)
        if not isinstance(tp, Mapping):
            return f"no {MANIFEST_KEY_TRANSFER_PROVENANCE} block on a fitted job's manifest (plan §4.5)"
        if tp.get("header_check_class") != HEADER_CHECK_CLASS_EXPECTED:
            return (f"{MANIFEST_KEY_TRANSFER_PROVENANCE}.header_check_class {tp.get('header_check_class')!r} != "
                    f"{HEADER_CHECK_CLASS_EXPECTED!r}")
        if tp.get("header_located") is not False:
            return f"{MANIFEST_KEY_TRANSFER_PROVENANCE}.header_located {tp.get('header_located')!r} is not false"
        want_fit_kind = fit_kind_expected(cell)
        if tp.get("fit_kind") != want_fit_kind:
            return (f"{MANIFEST_KEY_TRANSFER_PROVENANCE}.fit_kind {tp.get('fit_kind')!r} != {want_fit_kind!r} (the "
                    f"{'external object' if ARM_BY_CELL[cell].external else 'LoRA'} arm {cell!r}'s fit-provenance status)")
        if not _positions_ok(tp.get("served_positions"), str(row["served_positions_expected"])):
            return (f"{MANIFEST_KEY_TRANSFER_PROVENANCE}.served_positions {tp.get('served_positions')!r} is not the "
                    f"expected non-header-located {row['served_positions_expected']!r}")
        want_render = C.RENDER_REGIME_BY_ARM[arm]
        if tp.get("served_render") != want_render:
            return f"{MANIFEST_KEY_TRANSFER_PROVENANCE}.served_render {tp.get('served_render')!r} != {want_render}"
        if tp.get("family") != family:
            return f"{MANIFEST_KEY_TRANSFER_PROVENANCE}.family {tp.get('family')!r} != the leg's family {family!r}"
        if no_switch:
            # the ONE render, under the one arm (measured live through /v1/tokenize with no kwargs; the fact records it under
            # both id fields) — never the switch class's {off, on} pair
            got_ids = _header_ids_by_arm(tp.get("served_header_ids"))
            if got_ids != _single_render_ids(fact, arm=arm):
                return (f"{MANIFEST_KEY_TRANSFER_PROVENANCE}.served_header_ids {got_ids!r} != the {family!r} fact's single render under "
                        f"{arm!r} ({list(fact['off_ids'])})")
        else:
            served_ids = _mapping(tp.get("served_header_ids"))
            if _header_ids_differ(served_ids, fact):
                return (f"{MANIFEST_KEY_TRANSFER_PROVENANCE}.served_header_ids {dict(served_ids)!r} != the {family!r} render "
                        f"fact's tuples off {fact['off_ids']} / on {fact['on_ids']}")
        if not isinstance(tp.get("header_moves_with_render"), bool):
            return (f"{MANIFEST_KEY_TRANSFER_PROVENANCE}.header_moves_with_render {tp.get('header_moves_with_render')!r} "
                    "is not the family fact's boolean")
        name = manifest.get(MANIFEST_KEY_OOS_EXPERIMENT)
        if no_switch and name is not None:
            return (f"manifest {MANIFEST_KEY_OOS_EXPERIMENT} {name!r} on a {arm} store — no out-of-support tuple exists for a family "
                    "with no thinking switch")
        if row.get("tuple_name") and name != row["tuple_name"]:
            return (f"manifest {MANIFEST_KEY_OOS_EXPERIMENT} {name!r} != the matrix row's tuple {row['tuple_name']!r} — "
                    "authorised under another experiment")
    else:
        if manifest.get(MANIFEST_KEY_TRANSFER_PROVENANCE) is not None:
            return f"a {MANIFEST_KEY_TRANSFER_PROVENANCE} block on a non-fitted cell"
    return None


def stamp_check(summary: Mapping[str, Any], expected: Sequence[str]) -> dict[str, Any]:
    """The leg's ``(cache_salt_regime, resumed_prefill_policy, render_regime)``
    stamps equal the matrix row's expected triple, each stamped once
    (distinct-value count 1) and inside its contract vocabulary. Absent stamps
    FAIL here — this experiment's driver writes them on every leg."""
    fields = C.EXPECTED_STAMP_TRIPLE_FIELDS
    want = dict(zip(fields, expected))
    failures = []
    observed = {}
    for f in fields:
        v = summary.get(f)
        observed[f] = v
        if v is None:
            failures.append(f"{f} not stamped")
            continue
        vocab = T.regime_vocabulary(f)
        if not isinstance(v, str) or v not in vocab:
            failures.append(f"{f}: {v!r} is off-contract (not in {list(vocab)})")
            continue
        count = summary.get(C.STAMP_REGIME_COUNT_FIELD[f])
        if count != 1:
            failures.append(f"{f}: {count!r} distinct values over the leg (MIXED; must be 1)")
        if v != want[f]:
            failures.append(f"{f}: {v!r} != expected {want[f]!r}")
    return {"expected": want, "observed": observed, "passed": not failures, "failures": failures}


def _sample_item(sample: Any) -> str:
    s = str(sample)
    return s.rsplit("/", 1)[0] if "/" in s else s


def _parse_ts(value: Any) -> datetime | None:
    """An ISO-8601 timestamp as the ledger and the driver write them (``Z`` or
    an offset; a naive one is read as UTC), or ``None`` when it is not one."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def _clean_resend_row(row: Mapping[str, Any], cap: int) -> bool:
    """A ledger row the harness's same-cap re-send can have produced: HTTP 200,
    not rejected, at ``cap``, no error class, no error code."""
    return (row.get("status") == 200 and row.get("rejected") is not True and row.get("max_tokens_requested") == cap
            and row.get("error_class") == C.LEDGER_ERROR_CLASS_NONE and row.get("error_code") in (None, ""))


def classify_redraws(task_rows: Sequence[Mapping[str, Any]], resumes: Sequence[Any], *, cap: int,
                     log_attribution: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Every call of the task beyond the first per (sample, recorded run),
    classified: a RECORDED RESUME (the sample's calls fall in two or
    more of the manifest's ``resume_attempts`` runs, each row binned into the
    run whose recorded start precedes it); the HARNESS'S SAME-CAP RE-SEND (extra
    calls inside one run — :data:`REDRAW_MECHANISM_TEXT` — every one of the
    sample's rows clean (:func:`_clean_resend_row`) and no run holding more
    calls than the retry ladder can make, :data:`LADDER_TEXT`); or UNEXPLAINED
    (an extra call on a sample with a rejected / non-200 / errored / other-cap
    row, or above the ladder). Returns the ``redraws`` readout, its
    ``failures`` (unexplained duplicates, the ladder exceeded, unreadable
    timestamps that would move a call between runs) and — when
    ``log_attribution`` is the ``.eval`` log's per-sample record
    (:func:`eval_cross_check`'s ``redraws``) — ``log_check``: the ledger's calls
    per sample in the run the log was generated in against the log's
    ``sum(1 + retries)`` over the final attempt's model events plus, per failed
    attempt, the calls its class implies (:func:`_redraw_log_check`), and the
    selected final row's finish reason against the log's at-cap bit, item by
    item. A re-send is BOUND to the body: within a recorded run every re-sent
    call carries the ``body_sha256`` of the sample's first call, and a recorded
    resume re-sends the same body — another body under the same sample label
    is another experiment, UNEXPLAINED. The FINAL draw per sample is the
    forwarded row with the latest request start (:data:`FINAL_DRAW_RULE_TEXT`),
    returned as ``_final_rows`` for the caller's ``n_at_cap`` comparison; the
    file's last row is not it, and ``n_final_draw_file_last_differs`` counts the
    samples where the two rows' finish reasons differ. ``n_items_sample_error``
    from the ledger is the number of items whose calls in one run exceed one
    attempt's ladder — only a failed sample attempt can produce that, so it is
    a LOWER bound; the log's exact count is in ``log_check``."""
    failures: list[str] = []
    run_starts: list[tuple[datetime, int]] = []
    unreadable_runs: list[int] = []
    for k, rec in enumerate(resumes):
        ts = _parse_ts(rec.get("ts")) if isinstance(rec, Mapping) else None
        if ts is None:
            unreadable_runs.append(k + 1)
        else:
            run_starts.append((ts, k + 1))
    run_starts.sort()
    starts = [ts for ts, _ in run_starts]

    def run_of(ts: datetime | None) -> int:
        if ts is None or not run_starts:
            return 1
        return run_starts[max(bisect.bisect_right(starts, ts) - 1, 0)][1]

    epoch = datetime.min.replace(tzinfo=UTC)
    per_sample: dict[str, list[tuple[datetime | None, int, Mapping[str, Any]]]] = defaultdict(list)
    n_unparseable = 0
    for r in task_rows:
        ts = _parse_ts(r.get("ts"))
        n_unparseable += ts is None
        per_sample[str(r.get("sample"))].append((ts, run_of(ts), r))
    n_redraw = n_resume = n_unexplained = 0
    items_redrawn: set[str] = set()
    items_unexplained: set[str] = set()
    items_sample_error: set[str] = set()
    body_changed: list[tuple[str, int]] = []           # (sample, run): a re-send whose body is not the first call's
    body_changed_across_runs: list[str] = []
    over: list[tuple[str, int]] = []
    hist: Counter[int] = Counter()
    max_calls = 0
    n_file_last_differs = 0
    calls_by_sample_run: dict[str, dict[int, int]] = {}
    final_rows: dict[str, Mapping[str, Any]] = {}
    file_last_rows: dict[str, Mapping[str, Any]] = {}
    for sample, entries in per_sample.items():
        file_order = list(entries)
        entries.sort(key=lambda e: e[0] or epoch)          # stable: equal / missing timestamps keep the file order
        by_run = Counter(run for _ts, run, _r in entries)
        calls_by_sample_run[sample] = dict(by_run)
        n_calls = len(entries)
        hist[n_calls] += 1
        max_calls = max(max_calls, n_calls)
        item = _sample_item(sample)
        clean = all(_clean_resend_row(r, cap) for _ts, _run, r in entries)
        # body binding: within one recorded run every re-send carries the body_sha256 of the sample's FIRST call (by
        # request start); a recorded resume's re-send must carry it too — another body under this sample's label is
        # another experiment, never a re-send
        first_body: dict[int, Any] = {}
        for _ts, run, r in entries:
            first_body.setdefault(run, r.get("body_sha256"))
            if r.get("body_sha256") != first_body[run] and (sample, run) not in body_changed:
                body_changed.append((sample, run))
        if len(set(first_body.values())) > 1:
            body_changed_across_runs.append(sample)
        bound = clean and not any(s == sample for s, _run in body_changed) and sample not in body_changed_across_runs
        within = all(n <= LADDER_CALLS_PER_SAMPLE_RUN for n in by_run.values())
        if not within:
            over.append((sample, max(by_run.values())))
        if any(n > LADDER_CALLS_PER_ATTEMPT for n in by_run.values()):
            items_sample_error.add(item)
        n_resume += len(by_run) - 1
        extra = sum(n - 1 for n in by_run.values())
        if bound and within:
            n_redraw += extra
            if extra:
                items_redrawn.add(item)
        else:
            n_unexplained += extra
            if extra:
                items_unexplained.add(item)
        # the FINAL draw: the forwarded row with the latest request start (FINAL_DRAW_RULE_TEXT), never the file's last
        forwarded = [(ts, run, r) for ts, run, r in entries if r.get("rejected") is not True]
        if forwarded:
            final_rows[sample] = forwarded[-1][2]
            file_last_rows[sample] = [r for _ts, _run, r in file_order if r.get("rejected") is not True][-1]
            n_file_last_differs += file_last_rows[sample].get("finish_reason") != final_rows[sample].get("finish_reason")
    if over:
        failures.append(f"{len(over)} sample(s) with more calls in one recorded run than the retry ladder can make (first "
                        f"{sorted(over)[:3]}; {LADDER_TEXT})")
    if body_changed:
        failures.append(f"{len(body_changed)} re-sent call(s) whose body_sha256 is not the sample's first call's in that run "
                        f"(first {sorted(body_changed)[:3]}) — a changed prompt, seed or sampling under the same sample label is "
                        "another experiment, not a re-send")
    if body_changed_across_runs:
        failures.append(f"{len(body_changed_across_runs)} sample(s) a recorded resume re-sent with another body_sha256 than the "
                        f"first run's call (first {sorted(body_changed_across_runs)[:3]}) — a resume re-sends the same request")
    if n_unexplained:
        failures.append(f"{n_unexplained} duplicate call(s) on {len(items_unexplained)} item(s) that no recorded resume or "
                        f"same-cap re-send accounts for (first {sorted(items_unexplained)[:3]}) — {REDRAW_MECHANISM_TEXT}")
    if n_unparseable:
        failures.append(f"{n_unparseable} ledger row(s) of {TASK} carry no readable ts — a call cannot be placed in a recorded run")
    if unreadable_runs and (n_redraw or n_resume or n_unexplained):
        failures.append(f"recorded run(s) {unreadable_runs} of the manifest's {MANIFEST_KEY_RESUME_ATTEMPTS} carry no readable "
                        "ts while samples were re-sent — a re-send cannot be placed in its run")
    n_items = len(per_sample)
    out: dict[str, Any] = {
        "n_redraw_calls": n_redraw, "n_items_redrawn": len(items_redrawn),
        "share_items_redrawn": (len(items_redrawn) / n_items) if n_items else None,
        "max_calls_per_item": max_calls, "n_items_sample_error": len(items_sample_error),
        "attributed": (not over and not n_unexplained and not n_unparseable and not body_changed and not body_changed_across_runs
                       and not (unreadable_runs and (n_redraw or n_resume))),
        "n_resume_calls": n_resume, "n_unexplained_calls": n_unexplained, "n_items_unexplained": len(items_unexplained),
        "n_items_body_changed": len({s for s, _run in body_changed} | set(body_changed_across_runs)),
        "n_over_ladder": len(over), "calls_per_item_hist": {str(k): v for k, v in sorted(hist.items())},
        "n_runs_recorded": len(resumes), "n_runs_unreadable_ts": len(unreadable_runs),
        # the final-draw selection (FINAL_DRAW_RULE_TEXT) and how often the file's last row would have been another draw
        "final_draw_rule": FINAL_DRAW_RULE_TEXT, "n_final_draw_file_last_differs": n_file_last_differs,
        "ladder": {"sdk_max_retries": SDK_MAX_RETRIES, "inspect_max_retries": INSPECT_MAX_RETRIES,
                   "retry_on_error": HARNESS_RETRY_ON_ERROR, "calls_per_attempt": LADDER_CALLS_PER_ATTEMPT,
                   "calls_per_sample_run": LADDER_CALLS_PER_SAMPLE_RUN, "text": LADDER_TEXT},
        "n_items_sample_error_note": (f"from the ledger: items with more than {LADDER_CALLS_PER_ATTEMPT} calls in one run — only "
                                      "a failed sample attempt makes that many; a lower bound (an attempt the provider refused "
                                      "after one call adds 1 and is invisible here; the log's exact count and the class of every "
                                      "failed attempt are under eval_check.redraws)"),
        "mechanism": REDRAW_MECHANISM_TEXT, "selection_caveat": REDRAW_SELECTION_CAVEAT,
        "attribution_source": "usage ledger (the classification rule alone; the .eval log was not read)",
        "failures": failures,
        # internal, popped by ledger_confirmation: the selected final row per sample (its finish reason is held to the
        # summary's n_at_cap) and the file-last row beside it
        "_final_rows": final_rows, "_file_last_rows": file_last_rows,
    }
    if log_attribution is not None:
        out["log_check"] = _redraw_log_check(calls_by_sample_run, run_of, log_attribution, final_rows=final_rows,
                                             file_last_rows=file_last_rows)
        out["attribution_source"] = ("usage ledger, cross-checked per sample against the .eval log's model events, failed "
                                     "attempts and at-cap bits (the log-reading run)")
    return out


def _redraw_log_check(calls_by_sample_run: Mapping[str, Mapping[int, int]], run_of: Callable[[datetime | None], int],
                      log_attribution: Mapping[str, Any], *, final_rows: Mapping[str, Mapping[str, Any]],
                      file_last_rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The ledger against the .eval log, sample by sample, in the run the log
    was generated in (``log_attribution["log_created"]`` binned like a ledger
    row): (1) the ledger's calls equal the log's ``final_calls`` plus, per
    failed attempt, the calls its CLASS implies (:func:`failed_attempt_bounds`:
    the exhausted timeout ladder exactly :data:`LADDER_CALLS_PER_ATTEMPT`, a
    pre-wire provider refusal exactly 0, a post-response provider refusal or
    any other failure a BOUND — such an item is held to the bound and NAMED in
    ``unverifiable_items``, the maximum never substituted); (2) the
    selected final row's finish reason (:data:`FINAL_DRAW_RULE_TEXT`) is the
    log's at-cap bit; beside it, how many samples the file's last row would have
    got wrong. A sample the log lacks or the ledger lacks is named too."""
    run = run_of(_parse_ts(log_attribution.get("log_created")))
    per_item = _mapping(log_attribution.get("per_item"))
    log_items = {str(i) for i in (log_attribution.get("item_ids") or ())} | set(per_item)
    at_cap_ids = {str(i) for i in (log_attribution.get("at_cap_ids") or ())}
    idx = int(log_attribution.get("sample_idx", 0))
    mismatches: list[dict[str, Any]] = []
    unverifiable: list[str] = []
    n_compared = n_file_last_bad = n_latest_start_bad = 0
    class_counts: Counter[str] = Counter()
    ledger_items = {_sample_item(s): s for s in calls_by_sample_run}
    for item_id in sorted(log_items | set(ledger_items)):
        # a sample the log lists with nothing notable is implied (one call, no failed attempt)
        rec = _mapping(per_item.get(item_id)) or {"final_calls": 1, "failed_attempt_classes": [], "failed_attempt_bounds": []}
        classes = [str(c) for c in (rec.get("failed_attempt_classes") or ())]
        bounds = [(int(b[0]), int(b[1])) for b in (rec.get("failed_attempt_bounds") or ())]
        class_counts.update(classes)
        sample = ledger_items.get(item_id, C.sample_header_value(item_id, idx))
        ledger_calls = _mapping(calls_by_sample_run.get(sample)).get(run)
        if item_id not in log_items or ledger_calls is None:
            mismatches.append({"item_id": item_id, "ledger_calls": ledger_calls,
                               "log_calls": None if item_id not in log_items else rec.get("final_calls"),
                               "why": "absent from the log" if item_id not in log_items else f"no ledger call in run {run}"})
            continue
        n_compared += 1
        low = high = int(rec["final_calls"])
        for lo, hi in bounds:
            low, high = low + lo, high + hi
        if low != high:
            unverifiable.append(item_id)
        if not low <= ledger_calls <= high:
            mismatches.append({"item_id": item_id, "ledger_calls": ledger_calls, "log_calls": low if low == high else [low, high],
                               "final_calls": rec["final_calls"], "failed_attempt_classes": classes,
                               "why": ("the ledger's calls are not the log's final-attempt calls plus what each failed attempt's "
                                       "class implies" + (" (a bounded failed attempt: a bound, not a count)" if low != high else ""))})
        final = final_rows.get(sample)
        if final is not None:
            log_at_cap = item_id in at_cap_ids
            if (final.get("finish_reason") == "length") != log_at_cap:
                n_latest_start_bad += 1
                mismatches.append({"item_id": item_id, "ledger_final_finish_reason": final.get("finish_reason"),
                                   "log_at_cap": log_at_cap,
                                   "why": "the selected final row's finish reason is not the log's at-cap bit for the sample"})
            file_last = file_last_rows.get(sample)
            if file_last is not None and (file_last.get("finish_reason") == "length") != log_at_cap:
                n_file_last_bad += 1
    return {"agrees": not mismatches, "run": run, "n_items_compared": n_compared, "n_mismatched": len(mismatches),
            "mismatches": mismatches[:20],
            "n_failed_attempts_by_class": {cls: class_counts.get(cls, 0) for cls in FAILED_ATTEMPT_CLASSES},
            "n_failed_attempts_exact": sum(class_counts.get(c, 0) for c in (FAILED_ATTEMPT_TIMEOUT_LADDER,
                                                                             FAILED_ATTEMPT_PROVIDER_REFUSAL_PRE_WIRE)),
            "n_failed_attempts_bounded": sum(class_counts.get(c, 0) for c in (FAILED_ATTEMPT_PROVIDER_REFUSAL_POST_RESPONSE,
                                                                               FAILED_ATTEMPT_PROVIDER_REFUSAL_UNPLACED,
                                                                               FAILED_ATTEMPT_UNVERIFIABLE)),
            # the items whose call count is a BOUND (a bounded failed attempt among theirs), named
            "unverifiable_items": unverifiable[:50],
            "final_draw_selection": {"rule": FINAL_DRAW_RULE_TEXT, "n_latest_start_disagreements": n_latest_start_bad,
                                     "n_file_last_disagreements": n_file_last_bad},
            **{k: log_attribution.get(k) for k in ("n_items", "n_items_with_failed_attempts", "n_failed_attempts",
                                                   "n_model_retries", "n_generate_calls", "n_items_errored", "n_at_cap")}}


def ledger_confirmation(path: Path, *, rows: Sequence[Mapping[str, Any]] | None, header: Mapping[str, Any],
                        summary: Mapping[str, Any], attempted_ids: Sequence[str], vmodel: str, max_model_len: Any,
                        manifest: Mapping[str, Any], log_attribution: Mapping[str, Any] | None = None,
                        arm: str = C.THINKING_ON) -> dict[str, Any]:
    """The out root's usage ledger (``rows``: every record of the file at
    ``path``, ``None`` when there is no file) against the leg: every
    forwarded call of ``TASK`` unique by ``sample`` and exactly the 500 attempted
    draws; ``n_rejected 0`` (a rejected call is a CENSORED draw the client
    re-sampled — D73 (d)); every ``error_class`` ``none``; ANY ``error_code`` is a
    NAMED refusal (the ``thinking_on_*`` / ``thinking_kwargs_*`` codes among them);
    every row ``thinking`` true at ``max_tokens_requested`` = :data:`CAP`; ``finish_reason
    == length`` count == the summary's ``n_at_cap``; the window holds; every call
    beyond the first per (sample, recorded run) is a recorded resume, the
    harness's same-cap re-send within the retry ladder, or UNEXPLAINED — the
    last fails (:func:`classify_redraws`, D151; ``redraws`` is the readout, and
    with ``log_attribution`` the .eval log's per-sample cross-check rides in
    ``redraws["log_check"]``); the ledger's upstream is the standup's proxy (the
    manifest's binding). Also the decode rate from the ledger's own tokens and
    latencies."""
    failures: list[str] = []
    if rows is None:
        return {"available": False, "path": str(path), "passed": False,
                "failures": [f"no {path.name} at the out-root {path.parent}"], "n_calls": None, "error_code": {}}
    # under identity version 2 the driver's probes travel THROUGH the ledger, labelled `probe/<kind>` (FC4 (2)): they are
    # rows of the file, never of the TASK — split off before the 500-call / thinking-ON / cap rules (M9), reconciled against
    # their records by `request_evidence_check`, and held here only to "no probe was rejected or errored"
    probe_rows = [r for r in rows if C.is_probe_task(r.get("task"))]
    other = [r for r in rows if r.get("task") != TASK and not C.is_probe_task(r.get("task"))]
    rows = [r for r in rows if r.get("task") == TASK]
    bad_probe = [r for r in probe_rows if r.get("rejected") is True or r.get("error_class") != C.LEDGER_ERROR_CLASS_NONE
                 or r.get("error_code") not in (None, "")]
    if bad_probe:
        failures.append(f"{len(bad_probe)} probe row(s) rejected or errored (first task {bad_probe[0].get('task')!r} sample "
                        f"{bad_probe[0].get('sample')!r} status {bad_probe[0].get('status')!r} error_code {bad_probe[0].get('error_code')!r})")
    expected = int(header["n_attempted_samples"])
    attempted = set(attempted_ids)
    forwarded = [r for r in rows if r.get("rejected") is not True]
    by_sample = Counter(str(r.get("sample")) for r in forwarded)
    n_unique = len(by_sample)
    n_resume = len(forwarded) - n_unique
    if n_unique != expected:
        failures.append(f"{n_unique} unique forwarded calls for {TASK} ({len(forwarded)} rows), expected {expected}")
    # every call beyond the first per (sample, recorded run) is classified: a RECORDED RESUME re-sent it in a
    # later run of the store (the manifest's resume_attempts, one entry per run with its start time), the HARNESS
    # re-sent it at the same cap inside one run (the retry ladder — LADDER_TEXT), or it is UNEXPLAINED (another cap, a
    # rejected / non-200 / errored call, or more calls in one run than the ladder can make). Re-sends pass with a
    # readout; an unexplained duplicate fails
    resumes = manifest.get(MANIFEST_KEY_RESUME_ATTEMPTS)
    n_attempts = len(resumes) if isinstance(resumes, list) else 0
    redraws = classify_redraws(rows, resumes if isinstance(resumes, list) else [], cap=CAP, log_attribution=log_attribution)
    failures.extend(redraws.pop("failures"))
    final_rows: Mapping[str, Mapping[str, Any]] = redraws.pop("_final_rows")
    redraws.pop("_file_last_rows")
    unknown = sorted({_sample_item(s) for s in by_sample} - attempted)
    if unknown:
        failures.append(f"{len(unknown)} forwarded sample(s) name items the record never attempted (first {unknown[:3]})")
    if len({_sample_item(s) for s in by_sample}) != n_unique:
        failures.append("forwarded samples are not unique per item (a second sample index under k = 1)")
    n_rejected = sum(1 for r in rows if r.get("rejected") is True)
    if n_rejected:
        statuses = sorted({str(r.get("status")) for r in rows if r.get("rejected") is True})
        failures.append(f"censored: {n_rejected} rejected call(s) (upstream HTTP {', '.join(statuses)}) re-sampled by the "
                        "client — the score on file is a second draw (D73 (d))")
    summary_rejected = summary.get(T.SUMMARY_REJECTED_FIELD)
    if summary_rejected is None:
        failures.append(f"the summary carries no {T.SUMMARY_REJECTED_FIELD} (the ledger-derived rejected count)")
    elif summary_rejected != n_rejected:
        failures.append(f"summary {T.SUMMARY_REJECTED_FIELD} {summary_rejected!r} != the ledger's {n_rejected}")
    classes = Counter(str(r.get("error_class")) for r in rows)
    if rows and set(classes) != {C.LEDGER_ERROR_CLASS_NONE}:
        failures.append(f"error_class counts {dict(classes)} — every call must be {C.LEDGER_ERROR_CLASS_NONE!r}")
    codes = Counter(str(r.get("error_code")) for r in rows if r.get("error_code") not in (None, ""))
    if codes:
        named = {k: v for k, v in codes.items() if k in C.THINKING_REFUSAL_CODES}
        failures.append(f"refused by name: error_code counts {dict(codes)}"
                        + (f" — the proxy's thinking refusals {named}" if named else ""))
    # math500-steer: the row's thinking flag at the ARM (the GSM8K validators' rule): a switch arm's rows read `thinking: true`;
    # under `not_available` NO switch is sent, so every row reads None (a boolean records a switch that was never there — P17)
    want_thinking = None if arm == C.THINKING_NOT_AVAILABLE else True
    thinks = sorted({r.get("thinking") for r in rows}, key=lambda v: (v is None, bool(v)))
    if rows and thinks != [want_thinking]:
        failures.append(f"thinking {thinks} — every row must be thinking: {want_thinking!r}"
                        + (" (no switch is sent under the no-switch arm; a boolean records a switch that was never there — P17)"
                           if want_thinking is None else ""))
    caps = sorted({r.get("max_tokens_requested") for r in rows}, key=lambda v: (v is None, v))
    if rows and caps != [CAP]:
        failures.append(f"max_tokens_requested {caps} != [{CAP}]")
    # the finish reason per SAMPLE is its FINAL draw's — the forwarded row with the latest request start, selected by
    # classify_redraws (FINAL_DRAW_RULE_TEXT). NOT the file's last row: the ledger appends a row when the upstream reply
    # completes, and an abandoned at-cap request that finishes after its replacement lands last — reading it would count
    # the abandoned draw's `length` in place of the scored draw's finish (the GLM cells: 19 of 462 / 28 of 387 samples)
    n_length = sum(1 for r in final_rows.values() if r.get("finish_reason") == "length")
    n_at_cap = summary.get("n_at_cap")
    if n_at_cap is None:
        failures.append(f"the summary's n_at_cap is null while the ledger counts {n_length} finish_reason == length")
    elif n_length != n_at_cap:
        failures.append(f"ledger finish_reason == length {n_length} != summary n_at_cap {n_at_cap}")
    pre = [r.get("prompt_tokens_pre") for r in forwarded]
    pre_ints = [p for p in pre if isinstance(p, int) and not isinstance(p, bool)]
    ledger_max = max(pre_ints) if pre_ints and len(pre_ints) == len(forwarded) else None
    window: dict[str, Any] = {"prompt_max": ledger_max, "cap": CAP, "max_model_len": max_model_len, "fits": None}
    if forwarded and ledger_max is None:
        failures.append("a ledger row carries no integer prompt_tokens_pre")
    if ledger_max is not None and isinstance(max_model_len, int):
        window["fits"] = ledger_max + CAP <= max_model_len
        if not window["fits"]:
            failures.append(f"max(prompt_tokens_pre) {ledger_max} + cap {CAP} > served max_model_len {max_model_len}")
    elif forwarded:
        failures.append("the window check needs the ledger max and the manifest's served max_model_len")
    binding = _mapping(manifest.get(MANIFEST_KEY_BINDING))
    if not binding:
        failures.append(f"the manifest carries no {MANIFEST_KEY_BINDING} block (the driver's verified endpoints)")
    elif binding.get("ledger_upstream") != binding.get("proxy_url"):
        failures.append(f"binding.ledger_upstream {binding.get('ledger_upstream')!r} != binding.proxy_url "
                        f"{binding.get('proxy_url')!r} — the ledger fronted another standup")
    # the decode rate from the ledger's own record (completion tokens over latency), forwarded task rows only
    tok = [(r.get("usage_completion_tokens"), r.get("latency_s")) for r in forwarded]
    rates = [t / lat for t, lat in tok if isinstance(t, (int, float)) and isinstance(lat, (int, float)) and lat > 0]
    total_tok = sum(t for t, lat in tok if isinstance(t, (int, float)))
    total_lat = sum(lat for t, lat in tok if isinstance(lat, (int, float)))
    toks = sorted(t for t, _lat in tok if isinstance(t, (int, float)))
    quantiles = ({"p50": C.nearest_rank_percentile(toks, 0.5), "p90": C.nearest_rank_percentile(toks, 0.9),
                  "p99": C.nearest_rank_percentile(toks, 0.99), "max": toks[-1]} if toks else None)
    other_vmodels = Counter(str(r.get("vmodel")) for r in rows if r.get("vmodel") != vmodel)
    return {"available": True, "path": str(path), "n_calls": len(forwarded), "n_unique_samples": n_unique,
            # every forwarded call beyond one per sample, whatever re-sent it (the historical name; `redraws` splits it)
            "n_resume_rows": n_resume, "resume_attempts": n_attempts, "redraws": redraws,
            "expected_calls": expected, "n_rejected": n_rejected, "summary_n_http_rejected_retried": summary_rejected,
            "error_class": dict(classes), "error_code": dict(codes), "thinking": thinks,
            "max_tokens_requested": caps, "n_length": n_length, "summary_n_at_cap": n_at_cap, "window": window,
            "decode_tok_s_mean": (sum(rates) / len(rates)) if rates else None,
            "decode_tok_s_aggregate": (total_tok / total_lat) if total_lat else None,
            "completion_tokens_quantiles": quantiles, "n_other_task_rows": len(other), "n_probe_rows": len(probe_rows),
            "probe_rows_by_kind": dict(Counter(str(r.get("task")) for r in probe_rows)),
            "task_rows_on_other_vmodel": dict(other_vmodels), "binding": dict(binding) if binding else None,
            "passed": not failures, "failures": failures}


def _arm_of(enable_thinking_value: Any) -> str:
    if enable_thinking_value is True:
        return C.THINKING_ON
    if enable_thinking_value is False:
        return C.THINKING_OFF
    return f"enable_thinking={enable_thinking_value!r}"


def probe_kind_of(task: Any) -> str | None:
    """``probe/<kind>`` -> ``<kind>``; ``None`` for anything that is not a probe label."""
    return str(task)[len(C.PROBE_TASK_PREFIX):] if C.is_probe_task(task) else None


def probe_inventory_per_run(evidence: Mapping[str, Any], *, route_vmodel: str, fitted: bool,
                           serving_identity: Any = None,
                            cert_class: str = C.CERT_CLASS_SWITCH) -> tuple[dict[tuple[str, str, str], int], str | None]:
    """The probe attempts ONE run of the driver sends, per ``(vmodel, render arm,
    probe kind)``, read off what the driver itself records: predicate (A)'s two
    ``render`` probes on the cell's route (one per arm, every thinking-ON leg);
    on a fitted cell the ``bites`` probe's ``steering_bites.n_probes`` prompts
    on the cell's route and on the twin, thinking-OFF, and predicate (C)'s
    ``lens_on_bite`` on ``lens_on_bite.lens_vmodel`` and ``.base_vmodel``,
    thinking-ON. Returns ``(inventory, problem)``; a problem means the inventory
    could not be derived (an unreadable count or vmodel) — the caller refuses,
    never reads around it.

    ``cert_class`` :data:`contracts.CERT_CLASS_NO_SWITCH` (llama-paper-rows P18)
    is the no-switch class's EXACT inventory, ``contracts.no_switch_probe_inventory``:
    ONE render probe under ``not_available``; on a fitted cell the bites probe's
    ``n_probes`` on the route and the twin under ``not_available``; no OFF / ON
    render probe, no lens-ON bite. The default is byte-identical to before."""
    if cert_class == C.CERT_CLASS_NO_SWITCH:
        bites = _mapping(evidence.get(EVIDENCE_BITES))
        lp = _mapping(bites.get(C.OOS_EVIDENCE_BITES_LOGPROB_LIVENESS))
        try:
            # the no-switch class's twin is the ENGINE's zero cell too: the
            # identity threads through so a native run counts `none__lora0`; absent keeps
            # the lens twin, byte-identically to every pre-engine caller. The logprob
            # liveness legs (main) ride along and count against the same twin.
            return C.no_switch_probe_inventory(route_vmodel, fitted=fitted, n_probes=bites.get("n_probes"),
                                               n_logprob_probes=(lp.get("n_probes") if lp else None),
                                               serving_identity=serving_identity), None
        except ValueError as e:
            return {(route_vmodel, C.THINKING_NOT_AVAILABLE, C.PROBE_KIND_RENDER): 1}, str(e)
    inv: dict[tuple[str, str, str], int] = {(route_vmodel, C.THINKING_OFF, C.PROBE_KIND_RENDER): 1,
                                            (route_vmodel, C.THINKING_ON, C.PROBE_KIND_RENDER): 1}
    if not fitted:
        return inv, None
    problems: list[str] = []
    bites = _mapping(evidence.get(EVIDENCE_BITES))
    n_probes = bites.get("n_probes")
    if _positive_int(n_probes):
        # the twin is the ENGINE's zero cell: `petri__none__lens0`
        # on a lens route, `petri__none__lora0` on a native one. ``serving_identity`` absent keeps the
        # lens twin — every caller that predates the engine reads byte-identically.
        twin_vm = C.twin_vmodel_for(serving_identity) if serving_identity is not None else C.twin_vmodel()
        for vm in (route_vmodel, twin_vm):
            inv[(vm, C.THINKING_OFF, C.PROBE_KIND_BITES)] = inv.get((vm, C.THINKING_OFF, C.PROBE_KIND_BITES), 0) + n_probes
    else:
        problems.append(f"{EVIDENCE_BITES}.n_probes {n_probes!r} is not a positive count — the bites probe's attempts cannot be counted")
    # the LOGPROB liveness probes, when the driver recorded that control — one per prompt on
    # the route, TWO per prompt on the twin (the scale-0 leg and its repeat, the measured floor), under the bites render
    lp = _mapping(bites.get(C.OOS_EVIDENCE_BITES_LOGPROB_LIVENESS))
    if lp:
        n_lp = lp.get("n_probes")
        if _positive_int(n_lp):
            # the logprob legs count against the SAME engine twin as the bites legs above — the registry's zero cell for
            # the route's engine (`petri__none__lora0` on a native route, `petri__none__lens0` on a lens one; an absent
            # identity keeps the lens twin, as every pre-engine caller reads). This branch spelled the LENS twin by hand,
            # so a natively served leg's `none__lora0` logprob probes read as records the driver never sends beside a
            # shortfall on `none__lens0` (the pulled qwen35_4b `arditi_allpos` legs: 10 records on the lora twin, 0 on the
            # lens one). `contracts.no_switch_probe_inventory` already counts them against the engine twin
            twin_lp = C.twin_vmodel_for(serving_identity) if serving_identity is not None else C.twin_vmodel()
            for vm, legs in ((route_vmodel, 1), (twin_lp, len(C.BITES_LEGS) - 1)):
                inv[(vm, C.THINKING_OFF, C.PROBE_KIND_LOGPROB_BITES)] = inv.get((vm, C.THINKING_OFF, C.PROBE_KIND_LOGPROB_BITES), 0) + legs * n_lp
        else:
            problems.append(f"{EVIDENCE_BITES}.{C.OOS_EVIDENCE_BITES_LOGPROB_LIVENESS}.n_probes {n_lp!r} is not a positive count — "
                            "the logprob liveness probes' attempts cannot be counted")
    bite = _mapping(evidence.get(EVIDENCE_LENS_ON_BITE))
    lens_vm, base_vm = bite.get("lens_vmodel"), bite.get("base_vmodel")
    if isinstance(lens_vm, str) and isinstance(base_vm, str):
        for vm in (lens_vm, base_vm):
            inv[(vm, C.THINKING_ON, C.PROBE_KIND_LENS_ON_BITE)] = inv.get((vm, C.THINKING_ON, C.PROBE_KIND_LENS_ON_BITE), 0) + 1
    else:
        problems.append(f"{EVIDENCE_LENS_ON_BITE}.lens_vmodel / base_vmodel {lens_vm!r} / {base_vm!r} unreadable — the lens-ON "
                        "bite's attempts cannot be placed")
    return inv, ("; ".join(problems) or None)


def _probe_sample_ok(sample: Any, vmodel: Any) -> bool:
    """The driver's labelled probe wrapper writes ``X-Tinycap-Sample: <vmodel>/<n>``."""
    if not isinstance(sample, str) or "/" not in sample:
        return False
    vm, _, n = sample.rpartition("/")
    return vm == vmodel and n.isdigit()


def request_evidence_check(out_root: Path, *, manifest: Mapping[str, Any], evidence: Mapping[str, Any],
                           attempted_samples: Sequence[str], ledger_rows: Sequence[Mapping[str, Any]] | None,
                           route_vmodel: str, fitted: bool, arm: str = C.THINKING_ON) -> dict[str, Any]:
    """The driver's versioned record of EVERY HTTP attempt
    (``contracts.request_evidence_record``, version 2: written from the wire
    bytes and headers — one per task call AND per labelled probe) reconciled
    against every attempt the ledger received, BY IDENTITY, through the
    producer's own rule ``contracts.request_evidence_reconciliation`` (the
    gate the driver runs): every ledger row carries ``request_id`` /
    ``body_sha256``; ids unique on each side; every ledger attempt has its
    record and every record its attempt; the two sides of one id agree on
    the wire digest and the task / sample labels; probes wear ``probe/<kind>``
    labels, task records a row task and a sample; every attempted item has
    a recorded attempt. Around it, the reader's own rules: every record wears
    the exact producer shape (the v2 fields, the version, a kind, a 64-hex
    digest) and shows NO declaration; every task record names THIS task on
    the leg's route, renders thinking-ON, requests the cap, and names an
    attempted sample (every attempted sample has one); every probe record's
    sample is ``<vmodel>/<n>`` on its own vmodel; the probe attempts per
    ``(vmodel, render, kind)`` reach the driver's per-run inventory
    (:func:`probe_inventory_per_run`) times the COMPLETED runs the manifest
    records (a probe to a route / render / kind the driver never sends
    refuses); a surplus beyond what the recorded INCOMPLETE runs (a preflight
    that refused — FC4 (3), on record with ``completed: false``) can account
    for is counted and reported (``n_probe_records_unattributed``). A missing
    file, an unreadable inventory or an unavailable ledger is a FAILURE —
    absence of evidence is never evidence of absence."""
    path = out_root / REQUEST_EVIDENCE_FILENAME
    failures: list[str] = []
    if not path.is_file():
        return {"available": False, "path": str(path), "passed": False, "n_records": 0,
                "failures": [(f"no {REQUEST_EVIDENCE_FILENAME} at the out-root {out_root} — declaration presence cannot "
                              "be demonstrated (a missing record never counts as absence)")]}
    named = manifest.get(MANIFEST_KEY_REQUEST_EVIDENCE)
    if Path(str(named or "")).name != REQUEST_EVIDENCE_FILENAME:
        failures.append(f"manifest {MANIFEST_KEY_REQUEST_EVIDENCE} {named!r} does not name {REQUEST_EVIDENCE_FILENAME}")
    recs = list(_iter_jsonl(path))
    if not recs:
        failures.append("no request-evidence records — a missing record never counts as demonstrated absence of the declaration")
    # ifeval-canonical-battery (P29): the record is version 3 since that landed
    # (the v2 fields + the wire's sampling terms); a store written under version 2
    # is held to the version-2 field set it was written with (C.REQUEST_EVIDENCE_
    # ACCEPTED_VERSIONS) — the MATH-500 stores are v2 and stay readable
    def _want_fields(r):
        return set(C.REQUEST_EVIDENCE_FIELDS_V2 if r.get("request_evidence_version") == 2 else REQUEST_EVIDENCE_FIELDS)
    off_shape = [(i, sorted(_want_fields(r) - set(r)), sorted(set(r) - _want_fields(r)))
                 for i, r in enumerate(recs) if set(r) != _want_fields(r)]
    if off_shape:
        i, missing, extra = off_shape[0]
        failures.append(f"{len(off_shape)} record(s) are not the producer's record (first #{i}: missing {missing}, extra {extra}; "
                        f"the fields are {list(REQUEST_EVIDENCE_FIELDS)})")
    versions = Counter(r.get("request_evidence_version") for r in recs)
    if recs and not set(versions) <= set(C.REQUEST_EVIDENCE_ACCEPTED_VERSIONS):
        failures.append(f"request_evidence_version values {dict(versions)} not within "
                        f"{list(C.REQUEST_EVIDENCE_ACCEPTED_VERSIONS)}")
    kinds = Counter(r.get("kind") for r in recs)
    unknown_kinds = sorted(str(k) for k in kinds if k not in (REQUEST_EVIDENCE_KIND_TASK, REQUEST_EVIDENCE_KIND_PROBE))
    if unknown_kinds:
        failures.append(f"record kind(s) {unknown_kinds} outside {[REQUEST_EVIDENCE_KIND_TASK, REQUEST_EVIDENCE_KIND_PROBE]}")
    bad_digest = [r for r in recs if not C.is_sha256_hex(r.get("body_sha256"))]
    if bad_digest:
        failures.append(f"{len(bad_digest)} record(s) whose body_sha256 is not a 64-hex digest (first {bad_digest[0].get('body_sha256')!r})")
    declared = [r for r in recs if r.get("petri_thinking_on_present") is not False or r.get("petri_thinking_on_value") is not None]
    if declared:
        failures.append(f"{len(declared)} record(s) with petri_thinking_on present (first {[(r.get('kind'), r.get('task'), r.get('sample')) for r in declared[:3]]}) "
                        "— a declaration was sent; no cell of this experiment is declared")
    attempted = {str(s) for s in attempted_samples}
    # -- the producer's ONE rule: the evidence inventory IS the ledger's attempt inventory, per attempt by request id --
    n_ledger_rows = n_ledger_task = n_resent = None
    if ledger_rows is None:
        failures.append("the ledger's attempts are unavailable — the records cannot be reconciled per attempted call")
    else:
        n_ledger_rows = len(ledger_rows)
        led_task = Counter(str(r.get("sample")) for r in ledger_rows if r.get("task") == TASK)
        n_ledger_task, n_resent = sum(led_task.values()), sum(1 for n in led_task.values() if n > 1)
        why = C.request_evidence_reconciliation(recs, ledger_rows, filed=[(TASK, s.split("/", 1)[0]) for s in attempted],
                                                # the producer's own rule under the ARM: no switch on the wire under `not_available`
                                                expected_enable_thinking=(None if arm == C.THINKING_NOT_AVAILABLE else True))
        if why is not None:
            failures.append(f"the producer's reconciliation rule (contracts.request_evidence_reconciliation) refuses: {why}")
    # -- the task calls: this task, this route, the ON render, the cap, the attempted samples --
    task_recs = [r for r in recs if r.get("kind") == REQUEST_EVIDENCE_KIND_TASK]
    other_tasks = Counter(str(r.get("task")) for r in task_recs if r.get("task") != TASK)
    if other_tasks:
        failures.append(f"task record(s) naming other tasks {dict(other_tasks)} — this run files {TASK!r} only")
    other_vm = Counter(str(r.get("vmodel")) for r in task_recs if r.get("vmodel") != route_vmodel)
    if other_vm:
        failures.append(f"task record(s) on another vmodel {dict(other_vm)} != the leg's route {route_vmodel!r}")
    # math500-steer: the task calls at the ROW's arm (llama-paper-rows P17 / P18 — the GSM8K validators' rule): under
    # `not_available` every task record carries NO switch on the wire (enable_thinking_present False, value None); the ON arm
    # reads byte-identically to before
    no_switch = arm == C.THINKING_NOT_AVAILABLE
    if no_switch:
        not_on = [r for r in task_recs if r.get("enable_thinking_present") is not False or r.get("enable_thinking_value") is not None]
        if not_on:
            failures.append(f"{len(not_on)} task call(s) CARRIED {C.CHAT_TEMPLATE_KWARGS_KEY}.{C.ENABLE_THINKING_KEY} on a {arm} leg (first "
                            f"{[(r.get('sample'), r.get('enable_thinking_value')) for r in not_on[:3]]}) — the no-switch arm sends none (P7 / P17)")
    else:
        not_on = [r for r in task_recs if r.get("enable_thinking_present") is not True or r.get("enable_thinking_value") is not True]
        if not_on:
            failures.append(f"{len(not_on)} task call(s) without enable_thinking present and true "
                            f"(first {[r.get('sample') for r in not_on[:3]]})")
    off_cap = [r for r in task_recs if r.get("max_tokens") != CAP]
    if off_cap:
        failures.append(f"{len(off_cap)} task call(s) whose wire body requests max_tokens {sorted({r.get('max_tokens') for r in off_cap}, key=str)} "
                        f"!= the cap {CAP}")
    ev_by_sample = Counter(str(r.get("sample")) for r in task_recs if r.get("task") == TASK)
    unknown = sorted(set(ev_by_sample) - attempted)
    if unknown:
        failures.append(f"{len(unknown)} task record(s) name samples the record never attempted (first {unknown[:3]})")
    absent = sorted(attempted - set(ev_by_sample))
    if absent:
        failures.append(f"no request-evidence record for {len(absent)} attempted call(s) (first {absent[:3]}) — absence of "
                        "evidence is never evidence of absence")
    # -- the probes: labelled, on their own vmodel, and the driver's per-run inventory times the completed runs --
    probe_recs = [r for r in recs if r.get("kind") == REQUEST_EVIDENCE_KIND_PROBE]
    bad_label = [r for r in probe_recs if probe_kind_of(r.get("task")) not in C.PROBE_KINDS or not _probe_sample_ok(r.get("sample"), r.get("vmodel"))]
    if bad_label:
        failures.append(f"{len(bad_label)} probe record(s) not labelled as the driver's wrapper labels them (task probe/<kind>, sample "
                        f"<vmodel>/<n>; first {(bad_label[0].get('task'), bad_label[0].get('sample'), bad_label[0].get('vmodel'))})")
    attempts = manifest.get(MANIFEST_KEY_RESUME_ATTEMPTS)
    attempts = attempts if isinstance(attempts, list) else []
    runs = max(len(attempts), 1)
    n_incomplete = incomplete_runs(manifest)
    completed = max(runs - n_incomplete, 0)
    inventory, inv_problem = probe_inventory_per_run(
        evidence, route_vmodel=route_vmodel, fitted=fitted,
        serving_identity=_mapping(manifest.get(C.MANIFEST_KEY_ROUTE)).get("serving_identity"),
        cert_class=(C.CERT_CLASS_NO_SWITCH if no_switch else C.CERT_CLASS_SWITCH))
    if inv_problem:
        failures.append(f"the probe inventory cannot be derived: {inv_problem}")
    # the render arm a probe record was sent under, as the inventory keys it: the validators' `_probe_arm` (a record whose body
    # carried NO switch is the no-switch arm's; a record that did carry one keeps its boolean's arm, a stray the inventory refuses)
    probe_arm_of = validators_module()._probe_arm
    got = Counter((str(r.get("vmodel")), probe_arm_of(r, arm=arm), str(probe_kind_of(r.get("task")))) for r in probe_recs)
    stray = {f"{vm}/{arm}/{kind}": n for (vm, arm, kind), n in got.items() if (vm, arm, kind) not in inventory}
    if stray:
        failures.append(f"probe record(s) to a route / render / kind the driver never sends: {stray}")
    unattributed = 0
    table: dict[str, dict[str, int]] = {}
    for (vm, arm, kind), per_run in inventory.items():
        n = got.get((vm, arm, kind), 0)
        table[f"{vm}/{arm}/{kind}"] = {"expected_per_run": per_run, "records": n}
        if n < per_run * completed:
            failures.append(f"probe shortfall on {vm}/{arm}/{kind}: {n} record(s) < {per_run} per run x {completed} completed run(s) — "
                            "a probe whose declaration presence is undemonstrated")
        else:
            unattributed += max(0, n - per_run * completed - per_run * n_incomplete)
    return {"available": True, "path": str(path), "n_records": len(recs), "n_task_records": len(task_recs),
            "n_probe_records": len(probe_recs), "n_ledger_rows": n_ledger_rows, "n_task_attempts_ledger": n_ledger_task,
            "n_resent_samples": n_resent, "n_runs": runs, "n_completed_runs": completed, "n_incomplete_runs": n_incomplete,
            "probe_inventory": table, "n_probe_records_unattributed": unattributed,
            "passed": not failures, "failures": failures}


_ABSENT = object()


def _render_certified_problems(rc: Any, *, fact: Mapping[str, Any], family: str) -> list[str]:
    """Predicate (A) as ``contracts.render_certified`` recorded it (the driver
    files the helper's SUCCESS record, or the ``RenderCertificationError``'s
    evidence — same keys, every flag false, the problems named)."""
    name = EVIDENCE_RENDER_CERTIFIED
    if not isinstance(rc, Mapping):
        return [f"no {name} block — predicate (A) was not recorded"]
    out: list[str] = []
    missing = [f for f in RENDER_CERTIFIED_FIELDS if f not in rc]
    if missing:
        out.append(f"{name} lacks {missing} — not contracts.render_certified()'s record")
    problems = rc.get("certification_problems")
    if rc.get("render_certified") is not True:
        out.append(f"{name}.render_certified is {rc.get('render_certified')!r}" + (f" — {problems}" if problems else ""))
    if rc.get("render_source") != RENDER_SOURCE_SERVED:
        out.append(f"{name}.render_source {rc.get('render_source')!r} != {RENDER_SOURCE_SERVED!r} — a local-tokenizer "
                   "render certifies nothing about the served route")
    for flag in (C.OOS_EVIDENCE_HEADER_CERTIFIED, "render_regime_certified", "think_shape_certified"):
        if rc.get(flag) is not True:
            out.append(f"{name}.{flag} is {rc.get(flag)!r}" + (" (null: that probe was never run)" if rc.get(flag) is None else ""))
    # llama-paper-rows (P18): the NO-SWITCH class — `contracts.no_switch_render_certified`'s record (predicate (A″)):
    # the header measured ONCE under the one arm, equal to the fact's single render; a switch-class record on a
    # no-switch fact (or the reverse) is another family's certification and refuses by name
    no_switch_fact = fact.get("thinking_switch") is False
    rc_class = rc.get("cert_class", C.CERT_CLASS_SWITCH)
    if no_switch_fact or rc_class == C.CERT_CLASS_NO_SWITCH:
        if not no_switch_fact:
            out.append(f"{name}.cert_class {rc_class!r} on the switched family {family!r} — predicate (A″) is the no-switch class's")
        if rc_class != C.CERT_CLASS_NO_SWITCH:
            out.append(f"{name}.cert_class {rc_class!r} on the no-switch family {family!r} — predicate (A) cannot certify a family "
                       f"with no thinking switch; the record must be contracts.no_switch_render_certified()'s ({C.CERT_CLASS_NO_SWITCH!r})")
        if list(rc.get("measured_arms") or []) != [C.THINKING_NOT_AVAILABLE]:
            out.append(f"{name}.measured_arms {rc.get('measured_arms')!r} != [{C.THINKING_NOT_AVAILABLE!r}] — the no-switch class "
                       "measures the header ONCE, under the one render")
        want = _single_render_ids(fact)
        for field in ("measured_header_ids_by_arm", "expected_header_ids_by_arm"):
            got = _header_ids_by_arm(rc.get(field))
            if got != want:
                out.append(f"{name}.{field} {got!r} != the {family!r} fact's single render {want!r}")
        if rc.get("family") != family:
            out.append(f"{name}.family {rc.get('family')!r} != the leg's family {family!r}")
        if rc.get("split_mode") != fact["split_mode"]:
            out.append(f"{name}.split_mode {rc.get('split_mode')!r} != the {family!r} fact's {fact['split_mode']!r}")
        if problems != []:
            out.append(f"{name}.certification_problems {problems!r} is not empty")
        return out
    ids = _mapping(rc.get("measured_header_ids_by_arm"))
    if _header_ids_differ(ids, fact):
        out.append(f"{name}.measured_header_ids_by_arm {dict(ids)!r} != the {family!r} fact's tuples off {fact['off_ids']} / "
                   f"on {fact['on_ids']}")
    exp = _mapping(rc.get("expected_header_ids_by_arm"))
    if _header_ids_differ(exp, fact):
        out.append(f"{name}.expected_header_ids_by_arm {dict(exp)!r} != the analysis' {family!r} fact — the producer "
                   "certified against another fact")
    if rc.get("family") != family:
        out.append(f"{name}.family {rc.get('family')!r} != the leg's family {family!r}")
    if rc.get("split_mode") != fact["split_mode"]:
        out.append(f"{name}.split_mode {rc.get('split_mode')!r} != the {family!r} fact's {fact['split_mode']!r}")
    if problems != []:
        out.append(f"{name}.certification_problems {problems!r} is not empty")
    return out


#: the two steered serving paths :func:`steered_path_of` names (the values of ``_STEERED_PATH_BY_MARKER``)
STEERED_PATH_LENS, STEERED_PATH_LORA = "lens", "lora"
#: the ONE served-positions mode a native adapter has: it applies at every position (``proxy.POSITIONS_ALL``; the
#: fit's ``positions == "all"`` is what the native enumeration admits) — a literal here so the figure venv imports no proxy
NATIVE_SERVED_POSITIONS_EXPECTED = "all"


def certification_path_of(route: Mapping[str, Any]) -> tuple[str | None, list[str]]:
    """The serving PATH the certification predicates are read under, off the store's OWN route block:
    :data:`STEERED_PATH_LORA` (the native engine), :data:`STEERED_PATH_LENS`, or ``None`` for a route that steers
    nothing — a passthrough identity (the base and the prompted legs, served on BOTH engines) or no serving identity
    at all (the historical lens shape: the real producer helpers' fixtures and every lens store on record) — read
    with the lens predicates byte for byte. Beside the path, the problems that make the reading TORN, each named:

    * an identity carrying both steering markers
    * a PRESENT ``route.serving_engine`` outside :data:`contracts.SERVING_ENGINES` — a null, a typo, an unknown engine
      (rejecting only the exact native stamp beside a non-lora identity lets ``"typo"`` or a present null beside a
      ``+lens+`` identity certify as lens)
    * a ``+lora+`` identity whose route does not stamp exactly ``serving_engine: lora_native`` (absent, ``lens``) — the
      stored-record contract
    * a ``+lens+`` identity beside the native stamp (a lens route admits only an ABSENT stamp — the historical record — or
      an explicit ``lens``)
    * the native stamp on a route naming NO identity (the native writers stamp the identity on every route).

    A passthrough identity beside EITHER engine's stamp, or none, is that engine's unsteered leg and is not torn: the
    stamp names the engine, the identity names the path, and the base / prompted legs are served on both engines (the
    native driver stamps ``lora_native`` on its ``none`` and prompted routes too). A torn route reads as the LENS path with
    the problem named — the lens predicates then refuse the record in their own words too — so no native reading ever
    admits a record it cannot bind to its engine."""
    identity = route.get("serving_identity")
    try:
        path = steered_path_of(identity)
    except ValueError as e:                                   # both markers: a torn identity
        return None, [f"route.serving_identity: {e}"]
    engine = route.get(C.SERVED_ENGINE_FIELD, _ABSENT)
    field = f"route.{C.SERVED_ENGINE_FIELD}"
    if engine is not _ABSENT and engine not in C.SERVING_ENGINES:
        return None, [f"{field} {engine!r} is not one of {list(C.SERVING_ENGINES)} (a present null, a typo or an unknown engine) "
                      f"beside serving_identity {identity!r} — the certification predicates cannot be bound to an engine, so "
                      "they read as the lens's"]
    if path == STEERED_PATH_LORA:
        if engine != C.SERVING_ENGINE_LORA_NATIVE:
            return None, [f"route.serving_identity {identity!r} names the native path but {field} is "
                          + ("absent" if engine is _ABSENT else repr(engine))
                          + f" — a natively served store stamps {C.SERVING_ENGINE_LORA_NATIVE!r} (the stored-record "
                          "contract); the certification predicates cannot be bound to an engine, so they read as the lens's"]
        return path, []
    if path == STEERED_PATH_LENS:
        if engine == C.SERVING_ENGINE_LORA_NATIVE:
            return path, [f"{field} {engine!r} beside serving_identity {identity!r} — the native engine stamp on a route "
                          "the identity names as lens-served; a torn record (a lens route carries no stamp — the historical "
                          f"record — or {C.SERVING_ENGINE_LENS!r})"]
        return path, []                                       # absent (the historical lens record) or an explicit `lens`
    if isinstance(identity, str) and identity:
        return None, []                                       # a passthrough identity: either engine's unsteered leg
    if engine == C.SERVING_ENGINE_LORA_NATIVE:
        return None, [f"{field} {engine!r} on a route naming no serving identity ({identity!r}) — the native writers stamp the "
                      "identity on every route; a torn record"]
    return None, []


def _native_lens_position_problems(lp: Any) -> list[str]:
    """Predicate (B′) on a NATIVELY served fitted job: the driver's EXPLICIT inapplicability record
    (``served_capability_run._native_lens_position_record``) — ``status`` :data:`contracts.NOT_APPLICABLE`, ``verified``
    null (nothing was attested: the served-positions map, the resumed-prefill policy and the lens salt triple are lens
    facts a native adapter has none of; the adapter applies at every position) and the reason, named. Never an absent
    block, never a lens (B′) success record (another engine's record), never a failure record."""
    name = EVIDENCE_LENS_POSITION
    if not isinstance(lp, Mapping):
        return [f"no {name} block on a natively served fitted job — the driver records predicate (B′) "
                f"{C.NOT_APPLICABLE!r} by name, never an absent block"]
    if lp.get("verified") is False or "error" in lp:
        return [f"{name}: predicate (B′) FAILED as recorded — {lp.get('error')!r} (a lens attestation was attempted on a "
                f"natively served job; the native record states {C.NOT_APPLICABLE!r})"]
    out: list[str] = []
    if lp.get("status") != C.NOT_APPLICABLE:
        out.append(f"{name}.status {lp.get('status')!r} != {C.NOT_APPLICABLE!r} on a natively served job — the lens-position "
                   "attestation reads lens facts a native adapter has none of; a lens (B′) record here is another engine's")
    if lp.get("verified") is not None:
        out.append(f"{name}.verified {lp.get('verified')!r} beside the native inapplicability record — nothing is attested on "
                   "this engine (the driver records null)")
    reason = lp.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        out.append(f"{name}.reason {reason!r} — the native record names why predicate (B′) does not apply")
    return out


def _lens_position_problems(lp: Any, *, row: Mapping[str, Any], object_pin: str | None, harness: str,
                            expected_stamps: Sequence[str] | None = None) -> list[str]:
    """Predicate (B′) as ``contracts.lens_position_attested`` recorded it (the
    SUCCESS record; the driver files ``{"verified": false, "error": ...}`` on a
    refusal). A native inapplicability record (``status`` ``not_applicable``,
    a key no lens record carries) on a LENS-served job is another engine's
    record and is refused by name; every lens record reads as before."""
    name = EVIDENCE_LENS_POSITION
    if not isinstance(lp, Mapping):
        return [f"no {name} block on a fitted job — predicate (B′) was not recorded"]
    if lp.get("status") == C.NOT_APPLICABLE:
        return [f"{name} is the native inapplicability record (status {C.NOT_APPLICABLE!r}) on a lens-served job — "
                "another engine's record; predicate (B′) was not attested"]
    if lp.get("verified") is False or "error" in lp:
        return [f"{name}: predicate (B′) FAILED as recorded — {lp.get('error')!r}"]
    out: list[str] = []
    missing = [f for f in LENS_POSITION_ATTESTED_FIELDS if f not in lp]
    if missing:
        out.append(f"{name} lacks {missing} — not contracts.lens_position_attested()'s record")
    expect = str(row["served_positions_expected"])
    if not _positions_ok(lp.get("served_positions"), expect):
        out.append(f"{name}.served_positions {lp.get('served_positions')!r} is not the expected non-header-located "
                   f"{expect!r} (a header-located or mixed set is refused, fail closed)")
    if lp.get("header_located") is not False:
        out.append(f"{name}.header_located is {lp.get('header_located')!r}")
    digest = _mapping(lp.get("object_digest"))
    if object_pin is None:
        out.append(f"{name}.object_digest cannot be checked — no object pin for this model in the pin file")
    else:
        if digest.get("scalar") != object_pin:
            out.append(f"{name}.object_digest.scalar {digest.get('scalar')!r} != the pin {object_pin!r}")
        by_cell = _mapping(digest.get("by_cell"))
        if dict(by_cell) != {harness: object_pin}:
            out.append(f"{name}.object_digest.by_cell {digest.get('by_cell')!r} != {{{harness!r}: the pin {object_pin!r}}}")
    # the triple predicate (B′)'s record is held to: the caller's per-LEG class (`leg_expected_stamps`), the row's by default
    want_stamps = [str(s) for s in (expected_stamps if expected_stamps is not None else row["expected_stamps"])]
    policy = lp.get("attested_policy")
    if policy not in C.RESUMED_POLICY_LABELS or policy != want_stamps[1]:
        out.append(f"{name}.attested_policy {policy!r} != the class's expected resumed_prefill_policy {want_stamps[1]!r}")
    if list(lp.get("expected_stamps") or []) != want_stamps:
        out.append(f"{name}.expected_stamps {lp.get('expected_stamps')!r} != the class's {want_stamps}")
    if lp.get("own_key") != harness:
        out.append(f"{name}.own_key {lp.get('own_key')!r} != the cell's harness id {harness!r}")
    return out


def _lens_on_bite_logprob_admission_problems(bite: Mapping[str, Any], *, bites: Mapping[str, Any] | None) -> list[str]:
    """math-eight-arms-extension (class E): a predicate-(C) record admitted by the LOGPROB liveness verdict in place
    of the token bite is accepted ONLY on the evidence's own terms — the SAME leg's ``steering_bites`` gate is ``logprob``
    and its ``logprob_liveness`` block is CONCLUSIVE by the control's own criterion
    (``contracts.logprob_liveness_conclusive_problems``); the record says the token bite was NOT observed
    (``differs_from_base`` False, the observation), carries the ruling's sentence VERBATIM and copies the deciding
    numbers equal to the block's. The bite record's say-so alone admits nothing; identity (B′) is read beside it as
    before, a hard gate this substitution never touches."""
    name = EVIDENCE_LENS_ON_BITE
    out: list[str] = []
    if bite.get("differs_from_base") is not False:
        out.append(f"{name}: admitted by logprob yet differs_from_base is {bite.get('differs_from_base')!r} — the substitution "
                   "is for an UNOBSERVED token bite only")
    if bite.get(C.LENS_ON_BITE_SENTENCE_KEY) != C.LENS_ON_BITE_LOGPROB_SENTENCE:
        out.append(f"{name}.{C.LENS_ON_BITE_SENTENCE_KEY} {bite.get(C.LENS_ON_BITE_SENTENCE_KEY)!r} is not the ruling's sentence verbatim")
    if dict(_mapping(bite.get(C.LENS_ON_BITE_TOKEN_BITE_KEY))) != {"observed": False, "observation": C.LENS_ON_BITE_NO_BITE_PROBLEM}:
        out.append(f"{name}.{C.LENS_ON_BITE_TOKEN_BITE_KEY} {bite.get(C.LENS_ON_BITE_TOKEN_BITE_KEY)!r} is not the unobserved-bite observation")
    # an absent gate key IS the text gate (the reader's default everywhere; every earlier record has none)
    gate = bites.get(C.OOS_EVIDENCE_BITES_LIVENESS_GATE, C.LENS_LIVENESS_TEXT) if isinstance(bites, Mapping) else None
    if gate != C.LENS_LIVENESS_LOGPROB:
        return out + [f"{name}: admitted by logprob but this leg's steering_bites {C.OOS_EVIDENCE_BITES_LIVENESS_GATE} is {gate!r} — "
                      "the substitute must be the SAME run's logprob verdict"]
    lp = _mapping(bites.get(C.OOS_EVIDENCE_BITES_LOGPROB_LIVENESS))
    inconclusive = C.logprob_liveness_conclusive_problems(lp)
    if inconclusive:
        return out + [f"{name}: admitted by logprob but the leg's {C.OOS_EVIDENCE_BITES_LOGPROB_LIVENESS} verdict is NOT conclusive — "
                      + "; ".join(inconclusive)]
    want = C.lens_on_bite_logprob_ref(lp)
    have = _mapping(bite.get(C.LENS_ON_BITE_LOGPROB_REF_KEY))
    diff = sorted(k for k in want if have.get(k) != want[k])
    if diff:
        out.append(f"{name}.{C.LENS_ON_BITE_LOGPROB_REF_KEY} disagrees with steering_bites.{C.OOS_EVIDENCE_BITES_LOGPROB_LIVENESS} on {diff}")
    return out


def liveness_evidence_of(bite: Any, *, fitted: bool) -> str | None:
    """Which evidence admitted a fitted leg's predicate (C) — the reader's label ``token_bite`` for the ordinary record
    (the producer writes no key there: its absence IS the observed bite), the producer's
    ``logprob_conclusive_no_token_bite`` for a class-E admission; ``None`` on a non-lens leg, a no-switch (C″)
    record or a record that did not verify."""
    if not fitted or not isinstance(bite, Mapping) or bite.get("verified") is not True or bite.get("applicable") is not True:
        return None
    return str(bite.get(C.LENS_ON_BITE_LIVENESS_EVIDENCE_KEY) or C.LENS_ON_BITE_EVIDENCE_TOKEN_BITE)


def _lens_on_bite_problems(bite: Any, *, route_vmodel: str, cert_class: str = C.CERT_CLASS_SWITCH,
                           bites: Mapping[str, Any] | None = None) -> list[str]:
    """Predicate (C) as ``contracts.lens_on_bite_verdict`` recorded it, plus the
    two vmodels the driver appends on success (a failure carries ``error``).

    ``bites`` (math-eight-arms-extension, class E): the same leg's
    ``steering_bites`` block, read ONLY when the record says it was admitted by
    the logprob verdict — :func:`_lens_on_bite_logprob_admission_problems`; a
    record with an observed token bite is read byte for byte as before.

    ``cert_class`` :data:`contracts.CERT_CLASS_NO_SWITCH` (llama-paper-rows P18)
    reads predicate (C″) instead: the lens-ON bite is INAPPLICABLE on a family
    with no thinking channel and the record must SAY so — exactly
    ``contracts.no_switch_lens_on_bite_record()`` (``applicable`` False, the one
    reason, the class); an absent block, an ON-bite block (``applicable`` True /
    ``verified`` / ``error``) or another reason refuses by name. The default
    reads predicate (C) byte for byte as before."""
    name = EVIDENCE_LENS_ON_BITE
    if cert_class == C.CERT_CLASS_NO_SWITCH:
        if not isinstance(bite, Mapping):
            return [f"no {name} block on a fitted no-switch job — the (C″) inapplicability record is REQUIRED, never an absent block (P18)"]
        want = C.no_switch_lens_on_bite_record()
        if dict(bite) != want:
            return [f"{name} {dict(bite)!r} != the no-switch class's inapplicability record {want!r} — "
                    + ("an ON-bite block on a family with no thinking channel" if bite.get("applicable") is not False
                       else "another reason or class")]
        return []
    if not isinstance(bite, Mapping):
        return [f"no {name} block on a fitted job — predicate (C) was not recorded (D23)"]
    if bite.get("verified") is not True or "error" in bite:
        return [f"{name}: predicate (C) FAILED as recorded (verified={bite.get('verified')!r}) — {bite.get('error')!r}"]
    out: list[str] = []
    missing = [f for f in LENS_ON_BITE_FIELDS + LENS_ON_BITE_DRIVER_FIELDS if f not in bite]
    if missing:
        out.append(f"{name} lacks {missing} — not contracts.lens_on_bite_verdict()'s record")
    if bite.get("applicable") is not True:
        out.append(f"{name}.applicable is {bite.get('applicable')!r}")
    if bite.get("reasoning_present") is not True:
        out.append(f"{name}.reasoning_present is {bite.get('reasoning_present')!r}")
    admitted_by = bite.get(C.LENS_ON_BITE_LIVENESS_EVIDENCE_KEY)
    if admitted_by is None:
        # the ordinary record: the token bite WAS observed (the producer writes no evidence key there)
        if bite.get("differs_from_base") is not True:
            out.append(f"{name}.differs_from_base is {bite.get('differs_from_base')!r}")
    elif admitted_by == C.LENS_ON_BITE_EVIDENCE_LOGPROB:
        out += _lens_on_bite_logprob_admission_problems(bite, bites=bites)
    else:
        out.append(f"{name}.{C.LENS_ON_BITE_LIVENESS_EVIDENCE_KEY} {admitted_by!r} is not {C.LENS_ON_BITE_EVIDENCE_LOGPROB!r} "
                   "(the one admissible substitute for the token bite, D95)")
    if bite.get("lens_render_regime") != C.RENDER_REGIME_THINKING_ON:
        out.append(f"{name}.lens_render_regime {bite.get('lens_render_regime')!r} != {C.RENDER_REGIME_THINKING_ON!r}")
    if bite.get("problems") != []:
        out.append(f"{name}.problems {bite.get('problems')!r} is not empty")
    if bite.get("lens_vmodel") != route_vmodel:
        out.append(f"{name}.lens_vmodel {bite.get('lens_vmodel')!r} != the leg's route {route_vmodel!r}")
    base_vmodel = C.cell_spellings(CELL_BARE, declared=False).vmodel
    if bite.get("base_vmodel") != base_vmodel:
        out.append(f"{name}.base_vmodel {bite.get('base_vmodel')!r} != the base route {base_vmodel!r}")
    return out


def _logprob_liveness_problems(bites: Mapping[str, Any]) -> list[str]:
    """The reader's rule for a lens leg whose ``steering_bites.liveness_gate`` is ``logprob``
    (math-eight-arms-extension): the kind is one the ruling names; ``logprob_liveness`` is the driver's record —
    ``verified`` True, no ``error``, the statistic's ``diag`` with a numeric measured ``noise_ceiling`` /
    ``steered_divergence_median`` / ``bar`` and the ``noise_floor.measured`` equal to that ceiling (the floor was
    MEASURED in the run, never a constant alone: the bar's ``binding`` names which bound applied), the divergence
    clearing the bar, at least the record's own minimum of conclusive probes, and the not-a-dose-axis sentence
    VERBATIM; ``text_divergence`` present with an integer ``n_differing`` (reported beside, never gating). A missing
    or malformed block is a failure by name."""
    out: list[str] = []
    kind = bites.get("object_kind")
    if kind not in C.LOGPROB_LIVENESS_KINDS:
        out.append(f"steering bites: liveness_gate logprob on object_kind {kind!r}, not one of {list(C.LOGPROB_LIVENESS_KINDS)}")
    lp = _mapping(bites.get(C.OOS_EVIDENCE_BITES_LOGPROB_LIVENESS))
    if not lp:
        return out + [f"steering bites: liveness_gate logprob but no {C.OOS_EVIDENCE_BITES_LOGPROB_LIVENESS} block"]
    if lp.get("verified") is not True or "error" in lp:
        out.append(f"steering bites: {C.OOS_EVIDENCE_BITES_LOGPROB_LIVENESS} not verified (verified={lp.get('verified')!r}, "
                   f"error={lp.get('error')!r})")
    diag = _mapping(lp.get("diag"))
    floor = _mapping(lp.get("noise_floor"))
    nums = {k: diag.get(k) for k in ("noise_ceiling", "steered_divergence_median", "bar")}
    bad = [k for k, v in nums.items() if not isinstance(v, (int, float)) or isinstance(v, bool)]
    if bad:
        out.append(f"steering bites: {C.OOS_EVIDENCE_BITES_LOGPROB_LIVENESS}.diag lacks numeric {bad}")
    else:
        if floor.get("measured") != nums["noise_ceiling"]:
            out.append(f"steering bites: noise_floor.measured {floor.get('measured')!r} != diag.noise_ceiling {nums['noise_ceiling']!r}")
        if floor.get("binding") not in ("measured", "abs_floor"):
            out.append(f"steering bites: noise_floor.binding {floor.get('binding')!r} names neither bound of the bar")
        if not nums["steered_divergence_median"] >= nums["bar"]:
            out.append(f"steering bites: steered divergence median {nums['steered_divergence_median']!r} does not clear the bar "
                       f"{nums['bar']!r} (measured floor {nums['noise_ceiling']!r})")
    n_conc, n_min = lp.get("n_conclusive"), lp.get("min_conclusive_probes")
    if not (_positive_int(n_conc) and _positive_int(n_min) and n_conc >= n_min):
        out.append(f"steering bites: {C.OOS_EVIDENCE_BITES_LOGPROB_LIVENESS} n_conclusive {n_conc!r} < min_conclusive_probes {n_min!r}")
    if not _positive_int(lp.get("n_probes")):
        out.append(f"steering bites: {C.OOS_EVIDENCE_BITES_LOGPROB_LIVENESS}.n_probes {lp.get('n_probes')!r} is not a positive count")
    if lp.get("not_a_dose_axis") != C.LOGPROB_LIVENESS_NOT_A_DOSE_AXIS:
        out.append(f"steering bites: {C.OOS_EVIDENCE_BITES_LOGPROB_LIVENESS} lacks the not-a-dose-axis sentence verbatim")
    tx = _mapping(bites.get(C.OOS_EVIDENCE_BITES_TEXT_DIVERGENCE))
    if not tx or not isinstance(tx.get("n_differing"), int) or isinstance(tx.get("n_differing"), bool):
        out.append(f"steering bites: liveness_gate logprob but no {C.OOS_EVIDENCE_BITES_TEXT_DIVERGENCE} observation with an "
                   "integer n_differing beside it")
    return out


def certification_check(evidence: Mapping[str, Any], manifest: Mapping[str, Any], *, row: Mapping[str, Any],
                        fact: Mapping[str, Any], family: str, object_pin: str | None, arm: str = C.THINKING_ON,
                        expected_stamps: Sequence[str] | None = None, stamp_class: str | None = None,
                        gen_root: Path | str | None = None) -> dict[str, Any]:
    """Predicates (A), (B′) and (C) as the driver recorded them in
    ``evidence.json`` — the records ``contracts.render_certified``,
    ``contracts.lens_position_attested`` and ``contracts.lens_on_bite_verdict``
    return, key for key — plus the steering-bites control (OFF render),
    zero declared cells, the served-positions list the driver copies into
    ``route`` (the SELECTED cell's modes), and the family's render fact
    CERTIFIED (G-PARSER). ``object_pin`` is the pin file's digest for the leg's
    model (the (B′) record's ``object_digest`` is held to it). ``gen_root`` (the
    leg's wave root) lets a NATIVE leg's served-positions stamp be held to its
    standup record's engine-neutral map (:func:`native_served_positions_stamp_problem`);
    without it such a stamp refuses as before."""
    failures: list[str] = []
    fitted = bool(row["fitted"])
    # math500-steer: the CERTIFICATION CLASS is a function of the row's arm and the family fact (contracts.cert_class_for; the GSM8K
    # validators' reading): `not_available` on a fact whose thinking_switch is False is the NO-SWITCH class ((A″) + (B′) + (C″)); the
    # two mixed cases refuse by name; every ON leg reads byte-identically to before. `expected_stamps` / `stamp_class`: the per-LEG
    # stamp class predicate (B′)'s record is held to (`leg_expected_stamps`), the row's by default; under the served-steer class
    # the class's own requirements (the validators' `served_steer_problems`) apply
    V = validators_module()
    no_switch_fact = fact.get("thinking_switch") is False
    if arm == C.THINKING_NOT_AVAILABLE and not no_switch_fact:
        failures.append(f"arm {arm!r} on family {family!r}, whose render fact carries a thinking switch — the no-switch class is for a "
                        "switch-less family only (contracts.cert_class_for; P7)")
    if arm != C.THINKING_NOT_AVAILABLE and no_switch_fact:
        failures.append(f"arm {arm!r} on the switch-less family {family!r} — there is no thinking-ON class for it; serve it under "
                        f"{C.THINKING_NOT_AVAILABLE!r} (contracts.cert_class_for)")
    cert_class = C.CERT_CLASS_NO_SWITCH if (arm == C.THINKING_NOT_AVAILABLE and no_switch_fact) else C.CERT_CLASS_SWITCH
    if stamp_class == V.STAMP_CLASS_SERVED_STEER:
        failures += V.served_steer_problems(evidence, cert_class=cert_class)
    status = fact.get("evidence_status")
    if status != RENDER_FACT_CERTIFIED:
        failures.append(f"family {family!r}: render fact evidence_status {status!r} — not certified (G-PARSER, plan §5.3); "
                        "a candidate fact admits no production result")
    rc = evidence.get(EVIDENCE_RENDER_CERTIFIED)
    failures += _render_certified_problems(rc, fact=fact, family=family)
    if cert_class == C.CERT_CLASS_NO_SWITCH:
        # (A″): the ONE render probe under the one arm, and the header measured ONCE through the live tokenizer (the validators' records)
        failures += V._no_switch_render_probe_problems(evidence)
        failures += V._no_switch_header_problems(evidence.get(V.EVIDENCE_THINKING_HEADER_BY_ARM), fact=fact, family=family)
    lp = evidence.get(EVIDENCE_LENS_POSITION)
    bite = evidence.get(EVIDENCE_LENS_ON_BITE)
    bites = _mapping(evidence.get(EVIDENCE_BITES))
    route = _mapping(manifest.get(C.MANIFEST_KEY_ROUTE))
    harness = C.cell_spellings(str(row["cell"]), declared=False).harness
    # native-serving-default: the predicates are read under the ENGINE the store's own route
    # states — a `+lora+` identity (the native engine stamped beside it) reads predicate (B′)'s EXPLICIT inapplicability
    # record and NO served-positions map (a native adapter applies at every position and its standup records no lens
    # map); a `+lens+` identity, or a route naming no identity (the real producer helpers' shape and every lens store on
    # record), reads the lens predicates byte for byte as before. (C) is engine-neutral: the driver runs the ON bite
    # against the base route on both engines and records the same shape
    path, path_problems = certification_path_of(route)
    failures += path_problems
    native = path == STEERED_PATH_LORA
    if fitted:
        if native:
            failures += _native_lens_position_problems(lp)
        else:
            failures += _lens_position_problems(lp, row=row, object_pin=object_pin, harness=harness, expected_stamps=expected_stamps)
        # (C) on the switch class; (C″) — the EXPLICIT inapplicability record, REQUIRED — on the no-switch class
        failures += _lens_on_bite_problems(bite, route_vmodel=str(route.get("virtual_model")), cert_class=cert_class, bites=bites)
        if cert_class == C.CERT_CLASS_NO_SWITCH:
            failures += V._no_switch_bites_problems(bites)
        if bites.get(C.OOS_EVIDENCE_BITES_APPLICABLE) is not True or bites.get(C.OOS_EVIDENCE_BITES_VERIFIED) is not True:
            failures.append(f"steering bites not verified on a lens leg (applicable={bites.get(C.OOS_EVIDENCE_BITES_APPLICABLE)!r}, "
                            f"verified={bites.get(C.OOS_EVIDENCE_BITES_VERIFIED)!r})")
        n_diff = bites.get(C.OOS_EVIDENCE_BITES_N_DIFFERING)
        gate = bites.get(C.OOS_EVIDENCE_BITES_LIVENESS_GATE, C.LENS_LIVENESS_TEXT)
        if gate == C.LENS_LIVENESS_LOGPROB:
            # math-eight-arms-extension: the LOGPROB statistic ruled this leg — the text count is REPORTED, not required
            # (a cell live by logprob and inert by text is a dose fact); the ruling block must be verified, carry the
            # MEASURED floor, the divergence, the bar and the not-a-dose-axis sentence, and the text observation must
            # be beside it with its count. The gate applies to the two kinds only.
            failures += _logprob_liveness_problems(bites)
        elif gate == C.LENS_LIVENESS_TEXT:
            if not (isinstance(n_diff, int) and n_diff >= 1):
                failures.append(f"steering bites: n_differing {n_diff!r} < 1")
        else:
            failures.append(f"steering bites: liveness_gate {gate!r} is not one of {list(C.LENS_LIVENESS_MODES)}")
        n_probes = bites.get("n_probes")
        if not _positive_int(n_probes):
            failures.append(f"steering bites: n_probes {n_probes!r} is not a positive count (the probe inventory cannot be read)")
        # the bites probe renders OFF on an all-position cell of a switched family; under the no-switch class it renders the
        # family's ONE render (`not_available` — checked by the class's own `_no_switch_bites_problems` above)
        want_probe_arm = C.THINKING_NOT_AVAILABLE if cert_class == C.CERT_CLASS_NO_SWITCH else BITES_PROBE_RENDER_ARM_EXPECTED
        if bites.get(C.OOS_EVIDENCE_BITES_PROBE_RENDER_ARM) != want_probe_arm:
            failures.append(f"steering bites probe_render_arm {bites.get(C.OOS_EVIDENCE_BITES_PROBE_RENDER_ARM)!r} != "
                            f"{want_probe_arm!r} (D22: the bites probe renders OFF on an all-position cell; the no-switch class's one render)")
        try:
            # the twin is the ENGINE's zero cell when the route names its serving identity; a route
            # written before the engine field existed (the real producer helpers' shape, and every
            # lens store on record) has none and keeps the historical LENS twin, byte-identically —
            # the same reading as the GSM8K validator's, and as C.no_switch_probe_inventory's default
            sid = route.get("serving_identity")
            want_twin = C.twin_cell_id_for(sid) if sid is not None else C.TWIN_CELL_ID
        except ValueError as e:                       # a passthrough identity on a fitted row
            want_twin, _ = None, failures.append(f"steering bites twin cannot be resolved: {e}")
        if want_twin is not None and bites.get(C.OOS_EVIDENCE_BITES_TWIN) != want_twin:
            failures.append(f"steering bites twin {bites.get(C.OOS_EVIDENCE_BITES_TWIN)!r} != the registry's {want_twin!r} "
                            "(the twin is the ENGINE's zero cell)")
        for k in (C.OOS_EVIDENCE_BITES_ERROR, "stamp_error"):
            if k in bites:
                failures.append(f"steering bites carries {k}: {bites.get(k)!r}")
    else:
        for name, block in ((EVIDENCE_LENS_POSITION, lp), (EVIDENCE_LENS_ON_BITE, bite)):
            if block is not None:
                failures.append(f"{name} block on a non-lens cell")
        if bites.get(C.OOS_EVIDENCE_BITES_APPLICABLE) is not False:
            failures.append(f"steering bites applicable={bites.get(C.OOS_EVIDENCE_BITES_APPLICABLE)!r} on a non-lens cell "
                            "(the driver records applicable: false with the reason)")
    decl = route.get(C.ROUTE_KEY_SERVE_DECLARATIONS)
    if isinstance(decl, Mapping) and decl:
        failures.append(f"route.{C.ROUTE_KEY_SERVE_DECLARATIONS} names {sorted(decl)} — zero declared cells are expected")
    pending: list[str] = []
    served = route.get(ROUTE_KEY_LENS_SERVED_POSITIONS, _ABSENT)
    top = manifest.get(ROUTE_KEY_LENS_SERVED_POSITIONS, _ABSENT)
    if fitted:
        expect = str(row["served_positions_expected"])
        if native:
            # a native adapter applies at EVERY position and its standup records NO served-positions map: the driver
            # writes null there (the ENGINE's fact — `header_located = False` — never a missing lens record), so the row
            # must expect `all` (a header-located row has no native standup) and a stated map is a lens fact on a
            # native record
            if expect != NATIVE_SERVED_POSITIONS_EXPECTED:
                failures.append(f"the matrix row expects served positions {expect!r} on a natively served cell — a native "
                                f"adapter applies at every position ({NATIVE_SERVED_POSITIONS_EXPECTED!r}); a header-located "
                                "row has no native standup")
            if served not in (_ABSENT, None):
                # the driver's neutral-map lookup writes the ENGINE-NEUTRAL value into the lens-named field on a native leg
                # too (served_capability_run._standup_served_positions -> the manifest write at 6315-6316), so a stated map
                # here is admitted ONLY as the standup record's `served_positions[<harness>]` read off the pulled routing
                # table — never the engine's name, never the routes[] attestation, never the stamp alone
                why = native_served_positions_stamp_problem(manifest, served=served, harness=harness, gen_root=gen_root)
                if why is not None:
                    failures.append(f"route.{ROUTE_KEY_LENS_SERVED_POSITIONS} = {served!r} on a natively served cell — a "
                                    "served-positions map is a lens fact; on a native record it is admitted only as the standup "
                                    f"record's engine-neutral {C.SERVED_POSITIONS_KEY} value, and {why}")
        elif served is _ABSENT:
            pending.append(STAMP_ROUTE_SERVED_POSITIONS)
        else:
            pos = _served_positions(served)
            if pos is None or set(pos) != {expect}:
                failures.append(f"route.{ROUTE_KEY_LENS_SERVED_POSITIONS} = {served!r} != the expected {expect!r} (the SELECTED "
                                "cell's served modes, a list; null means the standup record named no positions)")
    elif served not in (_ABSENT, None):
        failures.append(f"route.{ROUTE_KEY_LENS_SERVED_POSITIONS} = {served!r} on a non-lens cell (the driver writes null there)")
    if top is not _ABSENT and served is not _ABSENT and top != served:
        failures.append(f"manifest {ROUTE_KEY_LENS_SERVED_POSITIONS} {top!r} != route.{ROUTE_KEY_LENS_SERVED_POSITIONS} {served!r}")
    admitted_by = liveness_evidence_of(bite, fitted=fitted)
    return {"passed": not failures, "failures": failures, "pending_stamps": pending,
            "arm": arm, "cert_class": cert_class, "stamp_class": stamp_class,
            "expected_stamps": [str(x) for x in (expected_stamps if expected_stamps is not None else row["expected_stamps"])],
            "render_fact_status": status, EVIDENCE_RENDER_CERTIFIED: dict(rc) if isinstance(rc, Mapping) else None,
            EVIDENCE_LENS_POSITION: dict(lp) if isinstance(lp, Mapping) else None,
            EVIDENCE_LENS_ON_BITE: dict(bite) if isinstance(bite, Mapping) else None,
            EVIDENCE_BITES: {k: bites.get(k) for k in (C.OOS_EVIDENCE_BITES_APPLICABLE, C.OOS_EVIDENCE_BITES_VERIFIED,
                                                       C.OOS_EVIDENCE_BITES_N_DIFFERING, "n_probes", C.OOS_EVIDENCE_BITES_TWIN,
                                                       C.OOS_EVIDENCE_BITES_PROBE_RENDER_ARM)},
            "route_lens_served_positions": (None if served is _ABSENT else served),
            # math-eight-arms-extension (class E): which evidence admitted predicate (C) on this leg — the row-level
            # fact; on a logprob admission the ruling's sentence rides beside it (a property of the operating point)
            "liveness_evidence": admitted_by,
            "liveness_sentence": (bite.get(C.LENS_ON_BITE_SENTENCE_KEY)
                                  if admitted_by == C.LENS_ON_BITE_EVIDENCE_LOGPROB and isinstance(bite, Mapping) else None)}


def sampling_ref_check(rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any],
                       items: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """Every scored row's ``sampling_ref`` is the PRODUCER's digest of
    the manifest's sampling block — ``contracts.sampling_ref`` (canonical JSON:
    sorted keys, COMPACT separators; the driver stamps exactly this), so a row
    can never be read against the wrong block. Absent on EVERY row -> a PENDING
    producer stamp (a driver before it); absent on some, or wrong -> FAIL.

    An identity-v2 producer stamps the SAME digest on every outcome item
    (``inspect_served.outcome_record``'s ``sampling_ref``); every item that
    carries the stamp is held to the digest too (a record stamped against
    another block FAILS by name) and the stamped count is recorded. A record
    without the stamp (the v1 shape) says nothing here — the rows' rule stands
    alone, as before."""
    block = _mapping(manifest.get(C.MANIFEST_KEY_SAMPLING))
    want = C.sampling_ref(block)
    refs = [r["_extra"].get(C.SAMPLING_REF_FIELD) for r in rows]
    present = [x for x in refs if x is not None]
    item_refs = [it[C.SAMPLING_REF_FIELD] for it in items if C.SAMPLING_REF_FIELD in it]
    out: dict[str, Any] = {"expected": want, "n_items_stamped": len(item_refs)}
    problems = []
    bad_rows = [x for x in refs if x != want] if present else []
    if bad_rows:
        problems.append(f"{len(bad_rows)} row(s) whose sampling_ref differs from the manifest block's digest {want} "
                        f"(first {bad_rows[:3]})")
    bad_items = [x for x in item_refs if x != want]
    if bad_items:
        problems.append(f"{len(bad_items)} outcome item(s) whose sampling_ref differs from the manifest block's digest "
                        f"{want} (first {bad_items[:3]})")
    if problems:
        return {**out, "status": GATE_FAIL, "detail": "; ".join(problems), "pending_stamps": []}
    if rows and not present:
        return {**out, "status": GATE_PENDING, "detail": "no row carries a sampling_ref (a producer before the stamp)",
                "pending_stamps": [STAMP_ROWS_SAMPLING_REF]}
    return {**out, "status": GATE_PASS, "detail": None, "pending_stamps": []}


# ---------------------------------------------------------------------------- #
# the at-cap readout and the no-reasoning readout                               #
# ---------------------------------------------------------------------------- #


def emitted_text(content: str | None, reasoning: str | None) -> str | None:
    """The text a draw emitted as the loop rule reads it: the reasoning channel,
    then the content (the wire order under a server parser; under a tag split
    the content already carries the block). ``None`` when NEITHER channel is on
    record — text the store does not carry, distinct from two empty channels."""
    if content is None and reasoning is None:
        return None
    return (reasoning or "") + (content or "")


def _periodic_suffix(tail: str, p: int) -> int:
    """The length of the longest suffix of ``tail`` that repeats with period
    ``p`` (every character equal to the one ``p`` before it, walking back from
    the end); at least ``p`` when ``len(tail) >= p``."""
    n = len(tail)
    j = n - 1 - p
    while j >= 0 and tail[j] == tail[j + p]:
        j -= 1
    return n - 1 - j


def classify_at_cap_tail(text: str | None) -> dict[str, Any]:
    """:data:`LOOP_RULE_TEXT` as a pure function over one draw's emitted text.

    ``period`` / ``repeats`` / ``cover`` describe the block that FIRED when the
    repeated-block rule holds, else the maximal periodic suffix that repeats at
    least twice (its period, whole repeats, share of the tail) — diagnostics for
    a draw the rule left as truncated reasoning; ``None`` when no block of an
    admitted period repeats even twice (a single trailing block is trivially
    periodic and says nothing). ``zlib_ratio`` (the tail's level-9 compression
    ratio, 1.0 for an empty tail) is always recorded and NEVER decides: a
    progressing enumeration ("For n = 1, ... For n = 50, ...") compresses like a
    loop without being one, so compression alone is no evidence.
    Ties in cover go to the SMALLEST period (a p-periodic suffix is also
    kp-periodic), so a pure repeat reports its true period, never a multiple.
    Conservative by design: a structured enumeration whose blocks DIFFER
    ("Option 1 ... / Option 2 ... / Option 3 ...") is not periodic and stays
    truncated reasoning."""
    tail = (text or "")[-LOOP_TAIL_CHARS:]
    n = len(tail)
    data = tail.encode("utf-8")
    zlib_ratio = (len(zlib.compress(data, 9)) / len(data)) if data else 1.0
    best: tuple[float, int, int] | None = None           # (cover, period, suffix length) — the maximal periodic suffix
    fired: tuple[float, int, int] | None = None          # ... among the candidates meeting BOTH block conditions
    if n >= 2 * LOOP_PERIOD_MIN:
        for p in range(LOOP_PERIOD_MIN, min(LOOP_PERIOD_MAX, n // 2) + 1):
            length = _periodic_suffix(tail, p)
            cover = length / n
            if length >= 2 * p and (best is None or cover > best[0]):
                best = (cover, p, length)
            if length >= LOOP_MIN_REPEATS * p and cover >= LOOP_MIN_COVER and (fired is None or cover > fired[0]):
                fired = (cover, p, length)
    by_block = fired is not None
    chosen = fired if fired is not None else best
    return {"loop": by_block, "rule": LOOP_RULE_REPEATED_BLOCK if by_block else LOOP_RULE_NONE,
            "period": None if chosen is None else chosen[1],
            "repeats": None if chosen is None else chosen[2] // chosen[1],
            "cover": None if chosen is None else chosen[0],
            "zlib_ratio": zlib_ratio}


def _tail_classes(at_cap_items: Sequence[str], texts: Mapping[str, str | None], *, source: str) -> dict[str, Any]:
    """Every at-cap draw's class from the text ``texts`` carries for it (absent
    or ``None`` = the store does not carry that draw's text under ``source``):
    the counts and the per-draw records. ``n_at_cap == n_loop +
    n_truncated_reasoning + n_text_unavailable`` by construction — a draw with
    no text is counted apart, never folded into either class."""
    classes: dict[str, Any] = {}
    n_loop = n_trunc = n_missing = 0
    for item_id in sorted(at_cap_items):
        text = texts.get(item_id)
        if text is None:
            classes[item_id] = {"text_source": None, "text_sha256": None, "loop": None, "rule": None, "period": None,
                                "repeats": None, "cover": None, "zlib_ratio": None}
            n_missing += 1
            continue
        rec = classify_at_cap_tail(text)
        classes[item_id] = {"text_source": source, "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), **rec}
        if rec["loop"]:
            n_loop += 1
        else:
            n_trunc += 1
    return {"text_source": source, "n_at_cap": len(at_cap_items), "n_classified": n_loop + n_trunc, "n_loop": n_loop,
            "n_truncated_reasoning": n_trunc, "n_text_unavailable": n_missing, "tail_classes": classes}


def at_cap_record(summary: Mapping[str, Any], items: Mapping[str, Mapping[str, Any]], *,
                  rows_text: Mapping[str, str | None]) -> dict[str, Any]:
    """The leg's at-cap readout from the SUMMARY's verified counts (``n_at_cap``
    over ``n_stop_reason`` — the summary's own ``at_cap_rate`` definition, the
    two counts held equal to the outcome record's header before this runs; the
    per-item ``at_cap`` flags were held equal to ``n_at_cap`` too), against the
    1 % line and the 3 % surfacing threshold, the two at-cap cases apart
    ((b) the cap inside the think block, unscored under the primary policy; (c)
    inside the answer, scored), and the loop rule over the at-cap draws whose
    text ``rows.jsonl`` carries (``rows_text``: item id -> emitted text of every
    scored draw) as ``store_rows_split`` — the split the record REPORTS until
    :func:`choose_split` swaps in the .eval log's (``split_source``). ``share``
    is ``None`` when no draw recorded a stop reason (the cap gate fails that leg
    by name); the two flags follow it."""
    at_cap_ids = [i for i, rec in items.items() if rec["at_cap"]]
    n_reason = int(summary["n_stop_reason"])
    n_at_cap = int(summary["n_at_cap"] or 0)         # null when no draw carries a stop reason (task_quality); held == the flags
    share = (n_at_cap / n_reason) if n_reason else None
    split = _tail_classes(at_cap_ids, rows_text, source=TEXT_SOURCE_ROWS)
    return {"rate": summary.get("at_cap_rate"), "share": share, "n_at_cap": n_at_cap, "n_with_stop_reason": n_reason,
            "line_1pct": AT_CAP_LINE_1PCT, "line_1pct_exceeded": None if share is None else share > AT_CAP_LINE_1PCT,
            "surfacing_threshold": AT_CAP_SURFACING_THRESHOLD,
            "at_or_above_surfacing_threshold": None if share is None else share >= AT_CAP_SURFACING_THRESHOLD,
            "n_case_b_unterminated_at_cap": sum(1 for i in at_cap_ids if items[i]["think_block_unterminated"]),
            "n_case_c_answer_at_cap": sum(1 for i in at_cap_ids if not items[i]["think_block_unterminated"]),
            "split_source": SPLIT_SOURCE_STORE_ROWS,
            "split_source_reason": "the .eval log was not read; the store rows' split stands",
            **{k: split[k] for k in ("n_loop", "n_truncated_reasoning", "n_text_unavailable")},
            "store_rows_split": split}


def choose_split(at_cap: dict[str, Any], tails: Mapping[str, Any] | None) -> None:
    """Swap the .eval log's split into the record IN PLACE when the log was read
    and carried EVERY at-cap draw's text (``eval_check.at_cap_tails``: source
    ``eval_log``, ``n_classified == n_at_cap``): ``split_source`` becomes
    ``eval_log`` and the three counts are the log's — ``n_text_unavailable`` then
    drops to 0. A log read that lacks some draw's text leaves the store rows'
    split standing and says so. ``store_rows_split`` is untouched either way."""
    if not tails:
        return
    if tails["text_source"] != TEXT_SOURCE_EVAL_LOG or tails["n_classified"] != tails["n_at_cap"]:
        at_cap["split_source_reason"] = (f"the .eval log was read but carried no text for {tails['n_text_unavailable']} of "
                                         f"{tails['n_at_cap']} at-cap draw(s); the store rows' split stands")
        return
    at_cap.update(split_source=SPLIT_SOURCE_EVAL_LOG,
                  split_source_reason="the .eval log was read and carried every at-cap draw's text (the unscored case-(b) "
                                      "draws included)",
                  **{k: tails[k] for k in ("n_loop", "n_truncated_reasoning", "n_text_unavailable")})


def reasoning_source(rec: Mapping[str, Any], *, reasoning: str | None, split_mode: str) -> str | None:
    """Which source says one draw emitted reasoning, or ``None`` when neither:
    :data:`PRESENT_BY_RECORD` when the outcome record's ``think_block_present``
    is true (the producer's own call, whatever the mode's rule); else, under
    ``explicit_tags`` ONLY, :data:`PRESENT_BY_CHANNEL` when the row's
    ``extra_json.reasoning`` is non-empty — inspect_ai 0.3.260 lifts a complete
    ``<think>…</think>`` block out of the text into ``ContentReasoning`` before
    the scorer reads it, so the record's tag-in-text flag misses exactly those
    draws while the row still carries the block. Under ``record_only`` the
    record's flag already reads the channel and under ``implicit_open`` it is
    by construction, so the channel fallback is never applied there (the
    readout there is the record's, byte for byte). A draw with no row (an
    unscored one) has ``reasoning`` None and is the record's call.

    Under ``none`` (llama-paper-rows P21: a family with NO reasoning channel)
    the source NEVER reads None while evidence exists: the record's own flags
    (``think_block_present``, ``stray_open_tag``, ``stray_close_tag``) call it
    :data:`PRESENT_BY_RECORD`; a non-empty ``reasoning`` on the row (a
    client-lifted channel the provider recorded) calls it
    :data:`PRESENT_BY_CHANNEL`. Either is a STRAY-TAG anomaly there, which
    :func:`thinking_record` counts by source and the reader refuses."""
    if rec["think_block_present"]:
        return PRESENT_BY_RECORD
    if split_mode == C.THINK_SPLIT_NONE:
        if any(rec.get(f) for f in C.STRAY_TAG_FIELDS):
            return PRESENT_BY_RECORD
        if reasoning:
            return PRESENT_BY_CHANNEL
        return None
    if split_mode == C.THINK_SPLIT_EXPLICIT_TAGS and reasoning:
        return PRESENT_BY_CHANNEL
    return None


def thinking_record(items: Mapping[str, Mapping[str, Any]], *, split_mode: str,
                    rows_reasoning: Mapping[str, str | None],
                    rows_lifted: Mapping[str, Mapping[str, Any] | None] | None = None) -> dict[str, Any]:
    """Of the leg's thinking-ON draws, how many FINISHED (not at the cap) with
    no reasoning emitted — no source calls the draw's reasoning present
    (:func:`reasoning_source`), whose meaning per split mode is
    :data:`NO_REASONING_RULE` (``record_only``: an empty reasoning channel and
    no tag; ``explicit_tags``: no ``<think>`` tag in the text AND an empty
    reasoning channel on the row; ``implicit_open``: impossible by construction,
    so 0 with ``by_construction`` true — a nonzero count there is a record the
    scorer cannot have filed, and the split-anomaly gate fails it).
    ``rows_reasoning`` is item id -> the row's ``extra_json.reasoning`` (None
    for a draw with no row). ``n_present_by_record`` / ``n_present_by_channel``
    say which source made each present call over EVERY draw (at cap included),
    so a cell reads e.g. "485 by channel + 15 by record = 500"; by_channel is 0
    outside ``explicit_tags`` by rule. An at-cap draw with no reasoning is
    counted apart (``n_no_reasoning_at_cap``), never folded into the finished
    share; an at-cap UNTERMINATED draw's record flag is true (the open tag is
    in the text), so its call is the record's, unchanged.

    ``none`` (llama-paper-rows P19 / P21 — a family with NO reasoning channel):
    every present call is a STRAY-TAG ANOMALY — filed under
    ``n_stray_tag_draws`` (by source: the record's flags / the row's lifted
    channel) beside the two source counts; ``by_construction`` is True (a
    finished draw has no reasoning by construction); ``rows_lifted`` (item id
    -> the row's ``client_lift_fired`` / ``lifted_reasoning`` facts, None for a
    draw with no row) adds the provider's own record of the client-side lift
    (P21) as evidence — a draw whose row says the lift fired is anomalous even
    when its text was restored. Every other mode ignores ``rows_lifted`` and
    is byte-identical."""
    if split_mode not in NO_REASONING_RULE:
        raise AnalysisError(f"thinking_record: split mode {split_mode!r} is not one of {sorted(NO_REASONING_RULE)}")
    n_on = len(items)
    sources = {i: reasoning_source(rec, reasoning=rows_reasoning.get(str(i)), split_mode=split_mode) for i, rec in items.items()}
    if split_mode == C.THINK_SPLIT_NONE:
        for i in items:
            lifted = (rows_lifted or {}).get(str(i))
            if sources[i] is None and isinstance(lifted, Mapping) and (lifted.get("client_lift_fired") is True
                                                                          or lifted.get("lifted_reasoning")):
                sources[i] = PRESENT_BY_CHANNEL
    n_no = sum(1 for i, rec in items.items() if not rec["at_cap"] and sources[i] is None)
    n_no_at_cap = sum(1 for i, rec in items.items() if rec["at_cap"] and sources[i] is None)
    out = {"split_mode": split_mode, "n_on_rows": n_on, "n_finished": sum(1 for rec in items.values() if not rec["at_cap"]),
           "n_no_reasoning_emitted": n_no, "share_no_reasoning": (n_no / n_on) if n_on else None,
           "n_no_reasoning_at_cap": n_no_at_cap,
           "by_construction": split_mode in (C.THINK_SPLIT_IMPLICIT_OPEN, C.THINK_SPLIT_NONE),
           "n_present_by_record": sum(1 for v in sources.values() if v == PRESENT_BY_RECORD),
           "n_present_by_channel": sum(1 for v in sources.values() if v == PRESENT_BY_CHANNEL),
           "rule": NO_REASONING_RULE[split_mode]}
    if split_mode == C.THINK_SPLIT_NONE:
        stray = sorted(i for i, v in sources.items() if v is not None)
        out["n_stray_tag_draws"] = len(stray)
        out["stray_tag_draws_by_source"] = {PRESENT_BY_RECORD: out["n_present_by_record"],
                                            PRESENT_BY_CHANNEL: out["n_present_by_channel"]}
        out["stray_tag_draw_ids"] = stray[:50]
    return out


# --------------------------------------------------------------------------- #
# the .eval re-derivation over the log the PRODUCER would select               #
# --------------------------------------------------------------------------- #


def _eval_log_entries(log_dir: Path) -> list[Path]:
    """Every ``*.eval`` entry of a log dir — a torn file or a dangling symlink
    included: it is PRESENT, and the producer's rule names why it does not
    qualify."""
    return sorted(log_dir.glob("*.eval")) if log_dir.is_dir() else []


def _eval_check_not_run(reason: str, *, present: bool, selected: list[str]) -> dict[str, Any]:
    return {"available": False, "reason": reason, "agrees": None, "log_present": present, "selected_logs": selected,
            "text": T.text_signature(None, reason), "at_cap_tails": None, "redraws": None, "anomaly_classes": None}


def _log_fact(result: Any, field: str) -> Any:
    """One outcome-defining fact of an ``ItemResult`` as the writer's
    ``outcome_record`` files it (``at_cap`` through the result's own property)."""
    value = getattr(result, field)
    return bool(value) if field in ("at_cap", "think_block_present", "think_block_unterminated", "stray_open_tag",
                                    "content_empty_with_reasoning") else value


def per_item_outcome_check(results: Sequence[Any], items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The retained log against the outcome record DRAW BY DRAW, keyed by
    ``(item_id, sample_idx)``: the parent's score availability / value
    (``T.per_item_log_check``, unchanged) AND every outcome-defining producer
    fact (:data:`OUTCOME_FIELDS_FROM_LOG`). The parent's check reads the score
    alone, so a count-preserving swap of ``at_cap`` / ``stop_reason`` /
    ``completion_tokens`` between two items passed it; it fails here on both
    items, field by field. The log side is the writer's own reading
    (``inspect_served.ItemResult`` -> ``outcome_record``), so the comparison is
    exact."""
    per = dict(T.per_item_log_check(results, items))
    by_key = {(str(r.item_id), int(getattr(r, "sample_idx", 0))): r for r in results}
    record = {(str(it["item_id"]), int(it.get("sample_idx", 0))): it for it in items}
    field_mismatches: list[dict[str, Any]] = []
    for key in sorted(k for k in record if k in by_key):
        r, it = by_key[key], record[key]
        for f in OUTCOME_FIELDS_FROM_LOG:
            rec_side = it.get(f)
            if f == "at_cap" or isinstance(rec_side, bool):
                rec_side = bool(rec_side)
            log_side = _log_fact(r, f)
            if log_side != rec_side:
                field_mismatches.append({"item_id": key[0], "sample_idx": key[1], "field": f, "log": log_side, "record": rec_side})
    per.update({"fields_compared": list(OUTCOME_FIELDS_FROM_LOG), "n_field_mismatched": len(field_mismatches),
                "field_mismatches": field_mismatches, "agrees": bool(per["agrees"]) and not field_mismatches})
    return per


def per_item_outcome_detail(per: Mapping[str, Any], *, show: int = 5) -> str | None:
    if per["agrees"]:
        return None
    parts = [T._per_item_detail({**per, "agrees": not (per["mismatches"] or per["missing_in_log"] or per["extra_in_log"])})]
    fm = per.get("field_mismatches") or []
    if fm:
        shown = "; ".join(f"{m['item_id']}/{m['sample_idx']}.{m['field']}: log {m['log']!r} != record {m['record']!r}"
                          for m in fm[:show])
        more = len(fm) - min(show, len(fm))
        parts.append(f"{len(fm)} outcome-defining field(s) differ between the log and the record ({shown}"
                     + (f"; and {more} more" if more else "") + ")")
    return "; ".join(p for p in parts if p) or None


def log_redraw_attribution(log: Any, *, sample_idx: int) -> dict[str, Any]:
    """The ``.eval`` log's per-sample call record for the ledger cross-check:
    per sample, ``final_calls`` = ``sum(1 + retries)`` over the final attempt's
    model events (``EvalSample.events``; ``ModelEvent.retries`` counts
    Inspect's model-call retries, the SDK's request retries included through the
    active-event tally), ``failed_attempts`` = ``len(error_retries)`` (the
    sample re-runs whose events the log drops), and whether the sample ended in
    error. Only the samples with anything beyond one clean call are listed
    (``per_item``); every other sample is (1, 0). ``log_created`` places the
    log in a recorded run."""
    per_item: dict[str, dict[str, Any]] = {}
    item_ids: list[str] = []
    at_cap_ids: list[str] = []
    n_items = n_failed = n_retries = n_events = n_errored = n_with_failed = 0
    for smp in (log.samples or ()):
        n_items += 1
        item_ids.append(str(smp.id))
        events = [e for e in (getattr(smp, "events", None) or ()) if getattr(e, "event", None) == "model"]
        retries = sum(int(getattr(e, "retries", None) or 0) for e in events)
        final_calls = len(events) + retries
        # every failed attempt's class and call bounds from its error message and retained tail — the attempt's model
        # events are dropped from the log, so the class is what says how many calls it made (failed_attempt_detail)
        details = [failed_attempt_detail(er) for er in (getattr(smp, "error_retries", None) or ())]
        classes = [d["class"] for d in details]
        errored = getattr(smp, "error", None) is not None
        out = getattr(smp, "output", None)
        if getattr(out, "stop_reason", None) == isv.STOP_REASON_AT_CAP:
            at_cap_ids.append(str(smp.id))
        n_events += len(events)
        n_retries += retries
        n_failed += len(classes)
        n_with_failed += bool(classes)
        n_errored += errored
        if final_calls != 1 or classes or errored:
            per_item[str(smp.id)] = {"final_calls": final_calls, "model_events": len(events), "model_retries": retries,
                                     "failed_attempts": len(classes), "failed_attempt_classes": classes,
                                     "failed_attempt_bounds": [[d["low"], d["high"]] for d in details],
                                     "failed_attempt_notes": [d["note"] for d in details if d["note"]], "errored": errored}
    created = getattr(getattr(log, "eval", None), "created", None)
    return {"sample_idx": int(sample_idx), "n_items": n_items, "n_items_with_failed_attempts": n_with_failed,
            "n_failed_attempts": n_failed, "n_model_retries": n_retries, "n_generate_calls": n_events,
            "n_items_errored": n_errored, "n_at_cap": len(at_cap_ids), "log_created": str(created) if created is not None else None,
            "item_ids": item_ids, "at_cap_ids": at_cap_ids, "per_item": per_item}


def eval_at_cap_tails(items: Sequence[Mapping[str, Any]], channels: Mapping[str, tuple[str | None, str | None]]
                      ) -> dict[str, Any]:
    """The loop rule over EVERY at-cap draw of the record from the log's two
    channels (``channels``: item id -> ``output_channels`` of the sample's
    output — the reader the producer itself files the row channels through), the
    unscored case-(b) draws included. Each classified draw carries its text's
    sha256, so the caller holds a scored draw's log text equal to its
    ``rows.jsonl`` text without either text entering the report."""
    at_cap_ids = [str(it["item_id"]) for it in items if it["at_cap"]]
    texts = {i: emitted_text(*channels[i]) for i in at_cap_ids if i in channels}
    return _tail_classes(at_cap_ids, texts, source=TEXT_SOURCE_EVAL_LOG)


def anomaly_kinds(items: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    """Item id -> kind for every anomalous item of a leg — the parent's
    ``T.anomaly_ids`` predicate, named per item (one kind each, in this order of
    precedence): :data:`ANOMALY_KIND_UNTERMINATED_NOT_AT_CAP` (the block never
    closed and the draw did not stop at the cap), :data:`ANOMALY_KIND_UNSCORED_OTHER`
    (unscored without an unterminated block), :data:`ANOMALY_KIND_STRAY_OPEN_TAG`
    (a scored draw carrying a literal open tag)."""
    out: dict[str, str] = {}
    for i, rec in items.items():
        if rec["think_block_unterminated"] and not rec["at_cap"]:
            out[str(i)] = ANOMALY_KIND_UNTERMINATED_NOT_AT_CAP
        elif not rec["scored"] and not rec["think_block_unterminated"]:
            out[str(i)] = ANOMALY_KIND_UNSCORED_OTHER
        elif rec["stray_open_tag"]:
            out[str(i)] = ANOMALY_KIND_STRAY_OPEN_TAG
    return out


def _in_code_span(text: str, idx: int) -> bool:
    """Whether ``text[idx]`` sits inside a Markdown code span: an open fenced
    block (an odd count of triple-backtick fences before it) or inline code (an
    odd count of single backticks earlier on its line, fences aside)."""
    before = text[:idx]
    if before.count("```") % 2 == 1:
        return True
    line = before[before.rfind("\n") + 1:].replace("```", "")
    return line.count("`") % 2 == 1


def classify_unterminated_anomaly(raw: str | None, *, finish_reason: str | None, completion: str | None) -> dict[str, Any]:
    """One unterminated-not-at-cap draw's class from the RAW HTTP body
    the log retained (``ModelEvent.call.response.choices[0].message.content``),
    the provider's ``finish_reason`` and the text the scorer read
    (``ModelOutput.completion``). :data:`ANOMALY_LIFTED_CLOSE` is a SYNTACTIC
    observation: the raw body carries a ``</think>`` occurrence the scorer's text
    lacks (inspect_ai's client-side lift paired an open-tag literal —
    ``tag_literals``: every ``<think[^>]*>`` in the raw body, the lift's own
    grammar — with it and removed the span), AND the last raw close sits outside
    a fenced / inline-code span, AND the raw text after it carries the answer
    marker (:data:`ANSWER_MARKER`). It asserts nothing about the model having
    closed its block or answered beyond those two signs (P2-2). A
    lifted close INSIDE a code span, or one with no answer marker after it (a
    quoted example the model reasoned past before stopping), is
    :data:`ANOMALY_OTHER` with the ambiguity named. :data:`ANOMALY_EOS_IN_THOUGHTS`
    when no close is anywhere in the raw body and the provider stopped on EOS
    (``finish_reason`` stop); :data:`ANOMALY_OTHER` otherwise, ``why`` naming
    the combination."""
    if raw is None:
        return {"class": ANOMALY_OTHER, "why": "no raw body on the log (ModelEvent.call.response absent)", "tag_literals": [],
                "raw_has_close": None, "completion_has_close": None, "finish_reason": finish_reason, "raw_chars": None,
                "close_in_code_span": None, "answer_marker_after_close": None}
    literals = LIFT_OPEN_TAG.findall(raw)
    close = "</think>"
    raw_close, comp_close = close in raw, close in (completion or "")
    last = raw.rfind(close)
    in_span = _in_code_span(raw, last) if raw_close else None
    marker_after = bool(ANSWER_MARKER.search(raw[last + len(close):])) if raw_close else None
    rec = {"tag_literals": literals, "raw_has_close": raw_close, "completion_has_close": comp_close,
           "finish_reason": finish_reason, "raw_chars": len(raw), "close_in_code_span": in_span,
           "answer_marker_after_close": marker_after}
    if raw_close and not comp_close:
        lit = ", ".join(f"`{x}`" for x in sorted(set(literals))) or "an open-tag literal the grammar matched"
        where = f"a </think> occurrence at character {last} of {len(raw)} in the raw body, absent from the scorer's text"
        if in_span:
            return {**rec, "class": ANOMALY_OTHER,
                    "why": (f"{where}, sits inside a fenced / inline-code span (a quoted example, not a terminator) — ambiguous, "
                            "pending stronger evidence")}
        if not marker_after:
            return {**rec, "class": ANOMALY_OTHER,
                    "why": (f"{where}, but no answer marker (an ANSWER: line) follows the last close — a quoted close the model "
                            f"reasoned past before stopping ({finish_reason!r}), or an unfinished answer; ambiguous, pending stronger "
                            "evidence")}
        return {**rec, "class": ANOMALY_LIFTED_CLOSE,
                "why": (f"{where}, was removed by inspect_ai's client-side lift (paired with {lit}); the last close sits outside any "
                        "code span and an ANSWER: line follows it in the raw body")}
    if not raw_close and finish_reason == "stop":
        return {**rec, "class": ANOMALY_EOS_IN_THOUGHTS, "why": "no </think> anywhere in the raw body; finish_reason stop (EOS inside the thoughts)"}
    if raw_close and comp_close:
        return {**rec, "class": ANOMALY_OTHER, "why": "both the raw body and the scorer's text carry </think> — the record's unterminated flag does not follow from the text"}
    return {**rec, "class": ANOMALY_OTHER, "why": f"no </think> in the raw body and finish_reason {finish_reason!r} (not an EOS stop)"}


def eval_anomaly_classes(log: Any, items: Sequence[Mapping[str, Any]], *, epoch: int = 1) -> dict[str, dict[str, Any]]:
    """The class of every unterminated-not-at-cap draw of the record, off the
    selected log: each such sample is re-read with attachments resolved
    (``read_eval_log_sample``; the retained body of a model call is stored as an
    attachment the bulk read leaves as a reference), its last model event's raw
    response body and finish reason handed to :func:`classify_unterminated_anomaly`
    with the sample's ``output.completion``. A sample the log cannot yield is
    :data:`ANOMALY_OTHER`, the exception named."""
    from inspect_ai.log import read_eval_log_sample

    out: dict[str, dict[str, Any]] = {}
    for it in items:
        if not (it["think_block_unterminated"] and not it["at_cap"]):
            continue
        item = str(it["item_id"])
        try:
            smp = read_eval_log_sample(str(log.location), id=item, epoch=epoch, resolve_attachments=True)
            events = [e for e in (getattr(smp, "events", None) or ()) if getattr(e, "event", None) == "model"]
            call = getattr(events[-1], "call", None) if events else None
            resp = getattr(call, "response", None) if call is not None else None
            choice = (list(resp.get("choices") or []) or [{}])[0] if isinstance(resp, Mapping) else {}
            message = choice.get("message") if isinstance(choice, Mapping) else None
            raw = (message or {}).get("content") if isinstance(message, Mapping) else None
            raw = raw if isinstance(raw, str) else (None if raw is None else str(raw))
            finish = choice.get("finish_reason") if isinstance(choice, Mapping) else None
            out_ = getattr(smp, "output", None)
            completion = getattr(out_, "completion", None) if out_ is not None else None
            out[item] = classify_unterminated_anomaly(raw, finish_reason=finish, completion=completion)
        except Exception as e:                                       # noqa: BLE001 — the reader's refusal is the class's reason
            out[item] = {"class": ANOMALY_OTHER, "why": f"the log could not yield the sample: {type(e).__name__}: {str(e)[:160]}",
                         "tag_literals": [], "raw_has_close": None, "completion_has_close": None, "finish_reason": None,
                         "raw_chars": None}
    return out


def unscored_breakdown(items: Mapping[str, Mapping[str, Any]], eval_check: Mapping[str, Any] | None, *,
                       class_source: str = CLASS_SOURCE_EVAL_LOG) -> dict[str, Any]:
    """The per-leg ``unscored`` readout: the record's unscored count split by
    kind — an unterminated block AT the cap (case (b)), an unterminated block
    NOT at the cap (the split anomaly, sub-classed from run E's log when it was
    read: ``by_class`` over :data:`ANOMALY_CLASSES`, ``class_source``
    ``eval_log`` or ``unavailable``; ``matched_report`` when a run without logs
    took them from the report it must match), and unscored for another reason.
    The kind counts are the record's and identical in both runs; ``by_class``
    and ``class_source`` are the declared by-design leaves."""
    classes = (eval_check or {}).get("anomaly_classes") if eval_check else None
    unt_not_cap = sorted(str(i) for i, r in items.items() if r["think_block_unterminated"] and not r["at_cap"])
    by_class = {c: 0 for c in ANOMALY_CLASSES}
    for i in unt_not_cap:
        cls = (classes or {}).get(i, {}).get("class") if classes is not None else None
        by_class[cls if cls in by_class else (ANOMALY_CLASS_UNAVAILABLE if cls is None else ANOMALY_OTHER)] += 1
    return {"n_unscored": sum(1 for r in items.values() if not r["scored"]),
            "n_unterminated_at_cap": sum(1 for r in items.values() if not r["scored"] and r["think_block_unterminated"] and r["at_cap"]),
            "n_unterminated_not_at_cap": len(unt_not_cap),
            "n_unscored_other": sum(1 for r in items.values() if not r["scored"] and not r["think_block_unterminated"]),
            "unterminated_not_at_cap_ids": unt_not_cap,
            "class_source": class_source if classes is not None else ANOMALY_CLASS_UNAVAILABLE, "by_class": by_class}


def eval_cross_check(cell_dir: Path, header: Mapping[str, Any], items: Sequence[Mapping[str, Any]], *,
                     max_tokens: int) -> dict[str, Any]:
    """The ``.eval`` re-derivation against the outcome record — the parent's
    check (the header's counts and every draw's score availability and value by
    ``(item_id, sample_idx)``) over the log the PRODUCER would select, held to
    ``max_tokens``: the completion cap the leg's identity was VALIDATED to
    (``max_tokens_by_task[TASK]``, held equal to :data:`CAP` by
    :func:`identity_problem` before any log is read — the value the identity
    gate holds), handed to both selectors as ``expected_max_tokens`` exactly as
    the producer's ``run_one_task`` hands ``reusable_log`` its
    ``plan.max_tokens``. A retained or copied-in log GENERATED at another cap,
    or recording none (``log.plan.config.max_tokens``), is another instrument's
    evidence and is passed over by name — without this the analysis selected a
    4,096-cap log the producer rejects: a false "independent cross-check
    agrees", or a needless refusal when such a log disagreed beside an older
    valid one. Keyword-only with no default so no caller can drop the
    predicate.

    An interrupted evaluation leaves a cancelled ``.eval`` beside the resume's
    successful complete log. The parent's ``eval_cross_check`` reads EVERY log
    under the task's dirs and ``results_from_log`` raises on the cancelled one,
    so a legitimate resume read as a re-derivation not run — an unwaivable
    pending on the scoring-path gate. Here each sample's log dir
    (``contracts.inspect_log_dir``; one per sample index of the record — this
    experiment's k = 1 and retry 0, both refused earlier if not) yields the ONE
    log the producer's own selection rule admits, ``inspect_served.reusable_log``:
    status ``success``, exactly the attempted count, no errored sample, the
    attempted ids as a multiset, generated at ``max_tokens``. Only that log is
    re-derived. Logs present but
    none qualifying: the check is NOT RUN with ``log_present`` true and the
    reason naming what disqualified every candidate
    (``inspect_served.retained_log_reason``), which the scoring-path gate files
    as a pending the operator cannot waive — fail closed. No log at all: the
    record stands alone, as before."""
    k = int(header["samples_per_item"])
    log_dirs = [C.inspect_log_dir(cell_dir, TASK, idx) for idx in range(k)]
    entries = {d: _eval_log_entries(d) for d in log_dirs}
    present = any(entries.values())
    if not present:
        return _eval_check_not_run(
            f"no Inspect .eval log under {cell_dir / C.INSPECT_LOGS_SUBDIR} — the rows carry no completion text; the text "
            "signature and the log re-derivation are unavailable; the outcome record stands alone", present=False, selected=[])
    try:
        from inspect_ai.log import (
            read_eval_log,  # noqa: F401 — the producer's selection rule reads through it
        )

        from sisteer.capeval.tiers import inspect_served as isv
    except ImportError as e:            # pragma: no cover - venv-dependent
        return _eval_check_not_run(f"inspect_ai / inspect_served not importable here ({e})", present=True, selected=[])
    results: list[Any] = []
    texts: list[str] = []
    channels: dict[str, tuple[str | None, str | None]] = {}
    selected: list[str] = []
    redraws: dict[str, Any] | None = None
    for idx, d in enumerate(log_dirs):
        ids = [str(it["item_id"]) for it in items if int(it.get("sample_idx", 0)) == idx]
        if not entries[d]:
            return _eval_check_not_run(f"no Inspect .eval log under {d} for sample {idx} while another sample's log is present — "
                                       "the record's draws cannot all be re-derived", present=True, selected=selected)
        log = isv.reusable_log(d, expected_n=len(ids), expected_ids=ids, expected_max_tokens=max_tokens)
        if log is None:
            why = isv.retained_log_reason(d, expected_n=len(ids), expected_ids=ids, expected_max_tokens=max_tokens)
            return _eval_check_not_run(
                f"no retained .eval log under {d} is the completed log of this task's population — the producer's own selection "
                f"rule (inspect_served.reusable_log: status success, {len(ids)} samples, no errored sample, the attempted ids, "
                f"generated at completion cap {max_tokens}): {why}", present=True, selected=selected)
        try:
            results.extend(isv.results_from_log(log, sample_idx=idx, primary_scorer=isv.PRIMARY_SCORER.get(TASK_NAME)))
        except Exception as e:                                   # noqa: BLE001 — the reader's refusal is the reason
            return _eval_check_not_run(f"{log.location}: {type(e).__name__}: {str(e)[:200]}", present=True, selected=selected)
        selected.append(Path(str(log.location)).name)
        # this experiment's k = 1 (refused earlier if not), so the ONE log's per-sample call record is the leg's
        redraws = log_redraw_attribution(log, sample_idx=idx)
        for smp in (log.samples or ()):
            out = getattr(smp, "output", None)
            comp = getattr(out, "completion", None) if out is not None else None
            if comp is not None:
                texts.append(str(comp))
            # both channels of the sample's final output, through the producer's own reader (the row channels are
            # `output_channels` of this same object at production time)
            channels[str(smp.id)] = isv.output_channels(out)
    derived = T.counts_from_results(results)
    diffs = {f: (v, header.get(f)) for f, v in derived.items() if header.get(f) is not None and header.get(f) != v}
    count_detail = "; ".join(f"{f}: log {a} != record {b}" for f, (a, b) in diffs.items()) or None
    per = per_item_outcome_check(results, items)
    per_detail = per_item_outcome_detail(per)
    # the class of every unterminated-not-at-cap draw off the selected log's raw bodies (k = 1: the one log)
    anomaly_classes = eval_anomaly_classes(log, items, epoch=1)
    return {"available": True, "reason": None, "agrees": not diffs and per["agrees"], "log_present": True,
            "selected_logs": selected, "derived": derived, "count_detail": count_detail, "per_item": per,
            "detail": "; ".join(x for x in (count_detail, per_detail) if x) or None, "text": T.text_signature(texts, None),
            # the loop rule over EVERY at-cap draw off the log (the unscored case-(b) draws have no row to read from);
            # under `eval_check` on purpose: a log-derived reading the figure venv cannot make, outside the two-venv match
            "at_cap_tails": eval_at_cap_tails(items, channels),
            # the log's per-sample call record (model events, retries, failed attempts) the ledger gate cross-checks its
            # re-send classification against — under `eval_check` for the same reason; `collect_leg` adds the
            # comparison's outcome here as `redraws["ledger_check"]`
            "redraws": redraws,
            # per unterminated-not-at-cap draw, its class from the raw body — log-derived, under `eval_check` so the
            # two-venv match never reads it; `collect_leg` folds the counts into `leg["unscored"]` (declared leaves)
            "anomaly_classes": anomaly_classes}


#: the admission-record stamps a matrix row may DECLARE and a leg carries verbatim (see `collect_leg`; read by the paper-rows
#: export through `_pr_stamp`, never invented): the selection word, the selection block, the fit-provenance block
DECLARED_ADMISSION_STAMPS: tuple[str, ...] = ("selection_outcome", "selection", "fit_provenance")


def collect_leg(gen_root: Path, row: Mapping[str, Any], *, ctx: Context, read_eval_logs: bool = True) -> dict[str, Any]:
    """One leg's record. ``missing`` names a leg the run did not produce;
    ``refused`` names an inconsistency between sources, a store that is not
    this leg's production run (:func:`identity_problem` — the sampling-row rule
    included), a PARTIAL run, or a population that is not 500 unique attempted
    draws. Either stops the estimand for its comparison; the gate checks
    (ledger, request evidence, stamps, certification, sampling_ref) are RECORDED
    on a usable leg and read by :func:`all_gates`."""
    key, cell, mode = str(row["model_key"]), str(row["cell"]), str(row["sampling_mode"])
    roots = GenRoots.of(gen_root, primary_pin=ctx.store_pin)
    entry, dup = roots.resolve(str(row["out_root"]))
    root = (entry["path"] if entry is not None else roots.primary) / str(row["out_root"])
    family = family_of(row)
    leg: dict[str, Any] = {"leg": row_leg_key(row), "job_id": row.get("job_id"), "model_key": key, "family": family,
                           "cell": cell, "kind": CELL_KIND[cell], "mode": mode, "fitted": bool(row["fitted"]),
                           "primary": row_is_primary(row), "route_kind": row.get("route_kind"),
                           "tuple_name": row.get("tuple_name"), "out_root": str(root), "missing": None, "refused": None,
                           "view": row_view(row), "wave": row_wave(row), "status": row_status(row)}
    if is_itdoc_row(row):       # E3: an it-doc reading carries its lane and the qualification VERBATIM off the row
        leg.update(itdoc_carriage(row))
    # the matrix row's DECLARED admission stamps, VERBATIM, where the row carries them (the GSM8K reader files the same on its
    # cell record): the steer-admission record's `selection_outcome` word, its `selection` block (both readings + the
    # reading's identity) and its `fit_provenance` block — the paper-rows export reads them off the leg (`_pr_stamp`) and
    # never relabels or fills them; a row that declares none leaves the leg silent (the published rows declare none)
    leg.update({k: row[k] for k in DECLARED_ADMISSION_STAMPS if k in row})
    if ctx.also_trees:          # The per-leg tree assignment, filed only when trees are declared
        leg["producer_tree"] = None
    if dup:
        leg["refused"] = dup
        return leg
    admitted_pin: str | None = None
    if ctx.store_pins is not None:
        # per-LEG admission (--store-pins): the leg is held to ITS row's pin, whatever root it sits under; a leg without an
        # entry refuses here by name, before any store is opened
        admitted_pin, why = admitted_pin_problem(str(leg["leg"]), None, ctx.store_pins)
        if admitted_pin is None:
            leg["refused"] = why
            return leg
        leg["admitted_pin"] = admitted_pin
    if entry is None or not root.is_dir():
        leg["missing"] = f"no out-root at {root}" if len(roots.entries) == 1 else \
            f"no out-root {row['out_root']!r} under any gen root ({[str(e['path']) for e in roots.entries]})"
        return leg
    cd, why = T._cell_dir(root)
    if cd is None:
        leg["missing"] = why
        return leg
    leg["cell_dir"], leg["cell_key"] = str(cd), cd.name
    manifest = completion.read_store_manifest(cd)
    if manifest is None:
        leg["refused"] = f"{cd / 'manifest.json'} is not a JSON object"
        return leg
    summary = completion.read_task_summary(cd, TASK)
    if summary is None:
        leg["missing"] = f"no {C.SUMMARY_DIR}/{C.marker_filename(TASK)} under {cd} — the leg filed no summary"
        return leg
    leg["summary"] = summary
    tinycap = _mapping(manifest.get(C.MANIFEST_KEY_TINYCAP))
    served = _mapping(tinycap.get("served"))
    route = _mapping(manifest.get(C.MANIFEST_KEY_ROUTE))
    producer = _mapping(manifest.get(MANIFEST_KEY_PRODUCER))
    prov = _mapping(manifest.get(C.MANIFEST_KEY_PROVENANCE))
    leg["identity"] = {
        "cell_id": manifest.get("cell_id"), "harness_cell_id": manifest.get("harness_cell_id"),
        "virtual_model": route.get("virtual_model"), "model_id": manifest.get("model_id"),
        "model_revision": manifest.get("model_revision"), "intervention_hash": manifest.get("intervention_hash"),
        "harness_config_hash": manifest.get("harness_config_hash"),
        "producer_tree_sha": producer.get("tree_sha"),
        "object_artifact_sha256": prov.get("artifact_sha256"),
        # the client lock the leg RAN UNDER (the driver hashes its checkout's lock into the frameworks block) — read for
        # the `compatibility.client_lock` DISCLOSURE (never a shared-projection field: the digest moves when a package
        # the client never imports joins the lock)
        "client_lock_sha256": _mapping(manifest.get("frameworks")).get("client_lock_sha256"),
        IDENTITY_KEY_VERSION: tinycap.get(IDENTITY_KEY_VERSION),
        C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING: tinycap.get(C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING),
        C.IDENTITY_KEY_MAX_TOKENS_BY_TASK: tinycap.get(C.IDENTITY_KEY_MAX_TOKENS_BY_TASK),
        "served": {k: served.get(k) for k in ("max_model_len", "max_num_seqs", "concurrency", "reasoning_parser",
                                              C.REDUCTION_MODE_FIELD)},
        MANIFEST_KEY_OOS_EXPERIMENT: manifest.get(MANIFEST_KEY_OOS_EXPERIMENT),
        MANIFEST_KEY_RESUME_ATTEMPTS: manifest.get(MANIFEST_KEY_RESUME_ATTEMPTS),
        "sampling": _mapping(manifest.get(C.MANIFEST_KEY_SAMPLING)) or None,
        MANIFEST_KEY_TRANSFER_PROVENANCE: manifest.get(MANIFEST_KEY_TRANSFER_PROVENANCE),
        # B5: the shared-field / server-settings projection and the arm contract read these (strings and bools only)
        **projection_identity_fields(manifest),
    }
    # code review C2: the launch attestation — the standup's own record of the composed launch, located through the
    # manifest under THIS leg's gen root and bound to its job (strings / bools / None; `absent` with its problems when the
    # pull carries no serving/ record — a declared-tree comparison refuses on that, the published one-sha rule does not)
    leg["launch"] = launch_attestation(manifest, gen_root=entry["path"], task_subdir=LAUNCH_STANDUP_TASK_SUBDIR)
    # the standup record's own tree (standup_jobs.jsonl[0].sha) beside the driver's (producer.tree_sha): the SERVING STACK's commit
    # the paper-rows export files as `serving_commit` when the two agree (the GSM8K reader's derivation, mirrored), never invented
    leg["standup_record"] = standup_record_summary(manifest, gen_root=entry["path"], task_subdir=LAUNCH_STANDUP_TASK_SUBDIR)
    problem = identity_problem(manifest, summary, row=row, ctx=ctx, family=family)
    leg["identity_validated"] = problem is None
    if problem:
        leg["refused"] = problem
        return leg
    # the store's producer is the PINNED wave tree, or a DECLARED one — the manifest's block, every recorded run of
    # the store, and (below) the outcomes record's own sha; a leg from another tree is refused before any estimand.
    # A PINNED gen root binds its legs to that one admitted tree; a bare root keeps the declared-tree semantics
    root_pin, root_also = (entry["pin"], ()) if entry["pinned"] else (ctx.store_pin, ctx.also_trees)
    if admitted_pin is not None:
        # per-LEG admission: the row's pin replaces the root's (a pinned root must name the same tree, checked below)
        if entry["pinned"] and not (str(entry["pin"]).startswith(admitted_pin) or admitted_pin.startswith(str(entry["pin"]))):
            leg["refused"] = (f"the leg's admitted pin {admitted_pin[:8]} (--store-pins) is not the gen root's pin {str(entry['pin'])[:8]} "
                              f"({entry['path']}) — the row's tree and the root's tree disagree")
            return leg
        root_pin, root_also = admitted_pin, ()
    problem = producer_problem(manifest, pin=root_pin, also=root_also)
    if problem is None and ctx.store_pins is not None:
        _pin, problem = admitted_pin_problem(str(leg["leg"]), producer.get("tree_sha"), ctx.store_pins)
    attempts = manifest.get(MANIFEST_KEY_RESUME_ATTEMPTS)
    leg["producer"] = {"pin": root_pin, "tree_sha": producer.get("tree_sha"), "driver": producer.get("driver"),
                       "outcomes_tree_sha": None,
                       # per-leg admission (--store-pins): the pin this leg was held to, from the pins doc (None under the one-pin rule)
                       "admitted_pin": admitted_pin, "admission": ("per-leg pins" if ctx.store_pins is not None else "one store pin"),
                       "n_runs": len(attempts) if isinstance(attempts, list) else None,
                       "n_incomplete_runs": incomplete_runs(manifest), "matches_pin": problem is None,
                       "root": str(entry["path"]), "root_pinned": bool(entry["pinned"])}
    if ctx.also_trees:
        leg["producer"]["declared_trees"] = list(ctx.also_trees)
    if problem:
        leg["refused"] = problem
        return leg
    if ctx.also_trees:
        leg["producer_tree"] = producer.get("tree_sha")
    arm = row_arm(row)
    if summary.get("thinking") != arm:
        leg["refused"] = f"summary thinking {summary.get('thinking')!r} != the row's arm {arm!r}"
        return leg
    if summary.get("task") != TASK:
        leg["refused"] = f"summary task {summary.get('task')!r} != {TASK!r}"
        return leg
    if summary.get("max_tokens") != CAP:
        leg["refused"] = f"summary max_tokens {summary.get('max_tokens')!r} != {CAP}"
        return leg
    partial_path = completion.partial_marker_path(cd, TASK)
    done_path = completion.done_marker_path(cd, TASK)
    if partial_path.is_file():
        partial = _mapping(_read_json(partial_path))
        leg["refused"] = (f"PARTIAL marker at {partial_path}: a limited run (limits {partial.get('limits')!r}, "
                          f"{partial.get('n_items')!r} of {partial.get('n_expected_full')!r} items) — not the complete experiment")
        return leg
    if not done_path.is_file():
        leg["refused"] = f"no done marker at {done_path} — the task did not complete in this store"
        return leg
    try:
        outcomes = completion.read_task_outcomes(cd, TASK)
    except CompletionError as e:
        leg["refused"] = f"outcomes/{C.task_outcomes_filename(TASK)}: {e}"
        return leg
    if outcomes is None:
        leg["missing"] = f"no {C.TASK_OUTCOMES_DIR}/{C.task_outcomes_filename(TASK)} under {cd} — the all-attempted record is required"
        return leg
    try:
        completion.verify_outcomes_against_summary(outcomes, summary)
    except CompletionError as e:
        leg["refused"] = str(e)
        return leg
    header = {k: v for k, v in outcomes.items() if k != "items"}
    leg["outcomes_header"] = header
    leg["producer"]["outcomes_tree_sha"] = header.get("producer_tree_sha")
    problem = outcomes_producer_problem(header, manifest_sha=str(leg["producer"]["tree_sha"]))
    if problem:
        leg["refused"] = problem
        return leg
    if int(header["samples_per_item"]) != 1:
        leg["refused"] = f"samples_per_item {header['samples_per_item']}: the estimand is defined at k = 1"
        return leg
    if header["n_attempted_items"] != N_ITEMS or header["n_attempted_samples"] != N_ITEMS:
        leg["refused"] = (f"n_attempted_items {header['n_attempted_items']} / n_attempted_samples "
                          f"{header['n_attempted_samples']} != the {N_ITEMS} MATH-500 draws (one per item)")
        return leg
    if summary.get("n_scored", 0) + summary.get("n_unscored", 0) != N_ITEMS:
        leg["refused"] = f"n_scored {summary.get('n_scored')} + n_unscored {summary.get('n_unscored')} != {N_ITEMS}"
        return leg
    items = {rec["item_id"]: rec for rec in outcomes["items"]}
    leg["items"] = items
    leg["attempted_ids"] = sorted(items)
    rows = T._rows_for_task(cd, TASK)
    leg["rows_check"] = T.rows_cross_check(rows, items)
    if not leg["rows_check"]["agrees"]:
        leg["refused"] = f"rows.jsonl disagrees with the outcome record: {leg['rows_check']['detail']}"
        return leg
    if len(rows) != summary.get("n_scored"):
        leg["refused"] = f"rows.jsonl holds {len(rows)} item rows != n_scored {summary.get('n_scored')}"
        return leg
    # the at-cap and think-present COUNTS the summary files (held equal to the record's header above) against the per-item
    # flags the readouts below count over: the producer derives both from one results list, so a difference is a torn
    # record, refused by name — never a share from one source read against a split from another
    for field, flag in (("n_at_cap", "at_cap"), ("n_think_block_present", "think_block_present"), ("n_stop_reason", None)):
        have = sum(1 for rec in items.values() if (rec["stop_reason"] is not None if flag is None else rec[flag]))
        want = summary.get(field)
        if field == "n_at_cap" and want is None and summary.get("n_stop_reason") == 0:
            want = 0            # the producer files null, not 0, when no draw carries a stop reason (task_quality)
        if want != have:
            leg["refused"] = f"summary {field} {summary.get(field)!r} != {have} outcome items with {flag or 'a stop reason'}"
            return leg
    leg["sampling_ref"] = sampling_ref_check(rows, manifest, items=outcomes["items"])
    # the re-derivation over the log the PRODUCER would select, never over every log in the dir — held to the cap the
    # identity was validated to above (max_tokens_by_task[TASK] == CAP, identity_problem; the summary's max_tokens
    # too), as the producer holds a retained pass log to plan.max_tokens
    cap = int(_mapping(tinycap.get(C.IDENTITY_KEY_MAX_TOKENS_BY_TASK))[TASK])
    leg["eval_check"] = (eval_cross_check(cd, header, outcomes["items"], max_tokens=cap) if read_eval_logs
                         else {"available": False, "reason": "log reading disabled", "agrees": None,
                               "text": T.text_signature(None, "log reading disabled"), "log_present": None,
                               "selected_logs": None, "at_cap_tails": None, "redraws": None, "anomaly_classes": None})
    if leg["eval_check"]["available"] and not leg["eval_check"]["agrees"]:
        leg["refused"] = f"the .eval log re-derivation disagrees with the record: {leg['eval_check']['detail']}"
        return leg
    # the record's unscored draws by kind, the split anomalies sub-classed when the log was read
    leg["unscored"] = unscored_breakdown(items, leg["eval_check"])
    # a run WITHOUT the log takes the classes from the report it is required to match (run E's reading of the raw
    # bodies), named as such — never inferred from the record (a lifted close and an EOS in the thoughts look alike there)
    matched = ctx.anomaly_classes.get(leg["leg"]) if not read_eval_logs else None
    if matched is not None:
        leg["matched_anomaly_classes"] = {str(i): dict(c) for i, c in _mapping(matched).items()}
        leg["unscored"] = unscored_breakdown(items, {"anomaly_classes": leg["matched_anomaly_classes"]},
                                             class_source=CLASS_SOURCE_MATCHED_REPORT)
    try:
        evidence = _mapping(_read_json(cd / "evidence.json"))       # None (absent) -> {}; a torn file refuses
    except AnalysisError as e:
        leg["refused"] = str(e)
        return leg
    leg["evidence_present"] = bool(evidence)
    fact = ctx.fact_for(family)
    leg["render_fact"] = {"split_mode": fact["split_mode"], "server_parser": fact["server_parser"],
                          "evidence_status": fact.get("evidence_status")}
    # math500-steer: the stamp class ONE LEG is held to — the matrix row's, unless the LEG's own steering-bites record
    # says the LOGPROB liveness statistic ruled it (the served-steer launcher's request-unique / never-resume route) —
    # selected by the record, never by the cell's name; a class problem (the selector on a non-fitted row) refuses by name
    stamp_class = validators_module().leg_expected_stamps(row, evidence, arm=arm, route=route)
    leg["stamp_class"] = {k: stamp_class.get(k) for k in ("class", "stamps", "selected_by")}
    if stamp_class.get("problem"):
        leg["refused"] = str(stamp_class["problem"])
        return leg
    leg["certification"] = certification_check(evidence, manifest, row=row, fact=fact, family=family,
                                               object_pin=ctx.object_pins.get(pin_key(row)), arm=arm,
                                               expected_stamps=stamp_class["stamps"], stamp_class=str(stamp_class["class"]),
                                               gen_root=entry["path"])
    ledger_path = root / Path(str(manifest.get(C.MANIFEST_KEY_USAGE_LEDGER) or C.LEDGER_FILENAME)).name
    ledger_rows = list(_iter_jsonl(ledger_path)) if ledger_path.is_file() else None
    log_attribution = (leg["eval_check"] or {}).get("redraws")
    leg["ledger"] = ledger_confirmation(ledger_path, rows=ledger_rows, header=header, summary=summary,
                                        attempted_ids=leg["attempted_ids"], vmodel=str(route.get("virtual_model")),
                                        max_model_len=served.get("max_model_len"), manifest=manifest,
                                        log_attribution=log_attribution, arm=arm)
    # the log-reading run holds the ledger's per-sample call counts to the .eval log's model events and failed
    # attempts; the comparison's numbers move under `eval_check` (a log-derived reading, outside the two-venv match)
    # and a disagreement REFUSES the leg — a re-send the log cannot account for is a real fault, not a readout
    log_check = (leg["ledger"].get("redraws") or {}).pop("log_check", None) if leg["ledger"].get("available") else None
    if log_attribution is not None and leg["ledger"].get("available"):
        leg["eval_check"]["redraws"] = {**log_attribution, "ledger_check": log_check}
        if log_check is not None and not log_check["agrees"]:
            first = log_check["mismatches"][:3]
            leg["refused"] = (f"the usage ledger's calls per sample disagree with the .eval log's model events for "
                              f"{log_check['n_mismatched']} item(s) in recorded run {log_check['run']} (first {first}) — a re-send "
                              "the log cannot account for")
            return leg
    leg["request_evidence"] = request_evidence_check(
        root, manifest=manifest, evidence=evidence,
        attempted_samples=[C.sample_header_value(it["item_id"], it["sample_idx"]) for it in outcomes["items"]],
        # EVERY attempt the ledger saw — task calls and labelled probes, rejected rows too: the inventory the records are
        # reconciled against by request id (the producer's own rule)
        ledger_rows=ledger_rows, route_vmodel=str(route.get("virtual_model")), fitted=bool(row["fitted"]), arm=arm)
    leg["stamps"] = stamp_check(summary, stamp_class["stamps"])
    # the at-cap readout over the SUMMARY's verified counts and the rows' text (both channels of every scored
    # draw, as the producer's ItemResult.extra files them), and the no-reasoning readout over the record's flags —
    # RESULTS, so filed on a usable leg only (a refused leg above carries neither key)
    rows_text = {str(r["item_id"]): emitted_text(r["_extra"].get("content"), r["_extra"].get("reasoning")) for r in rows}
    # the reasoning channel alone, per scored draw: under explicit_tags the no-reasoning readout reads it beside
    # the record's flag (inspect_ai lifts a complete <think>…</think> block out of the text before the scorer's tag
    # test — reasoning_source)
    rows_reasoning = {str(r["item_id"]): r["_extra"].get("reasoning") for r in rows}
    at_cap = at_cap_record(summary, items, rows_text=rows_text)
    if not _same_number(at_cap["rate"], at_cap["share"]):
        leg["refused"] = (f"summary at_cap_rate {at_cap['rate']!r} != n_at_cap / n_stop_reason = {at_cap['share']!r} "
                          "(the summary's own definition)")
        return leg
    tails = (leg["eval_check"] or {}).get("at_cap_tails")
    if tails:
        # a scored at-cap draw's text is on record twice (the row and the log); the two are one completion, so their
        # digests agree or the store is torn
        rows_classes = at_cap["store_rows_split"]["tail_classes"]
        torn = sorted(i for i, rec in tails["tail_classes"].items()
                      if rec["text_sha256"] is not None and rows_classes[i]["text_sha256"] is not None
                      and rec["text_sha256"] != rows_classes[i]["text_sha256"])
        if torn:
            leg["refused"] = (f"the .eval log's text for {len(torn)} scored at-cap draw(s) differs from rows.jsonl's (first "
                              f"{torn[:3]}) — one completion on record twice, differently")
            return leg
    # the split the leg REPORTS: the log's when it was read and carried every at-cap draw's text, else the store rows'
    choose_split(at_cap, tails)
    leg["at_cap"] = at_cap
    leg["thinking"] = thinking_record(items, split_mode=fact["split_mode"], rows_reasoning=rows_reasoning)
    # E3 tier-2 item 8: the REFUSAL-SHAPED class over the scored draws' CONTENT channel — a judge-free pattern count filed
    # BESIDE n_correct / n_unscored (no denominator, no grader value touched); the paper-rows export reads it off this block
    try:
        leg["refusal_shaped"] = RFS.readout(rows, items)
    except RFS.RefusalShapedError as e:
        leg["refused"] = str(e)
        return leg
    return leg


# --------------------------------------------------------------------------- #
# comparisons: the per-(model, mode) three-cell design, the mode deltas,       #
# the estimand under both branches, the verdict rule                           #
# --------------------------------------------------------------------------- #


#: E3: why an `it`-view comparison is anchored on the MAIN view's bare leg — recorded on the comparison, never silent
BARE_ANCHOR_RATIONALE = ("the `it` view carries no bare row of its own: the roster's `main` and `it` views serve ONE model, so the main "
                         "view's `none` leg is the same served target and is this comparison's anchor (the GSM8K noise-floor pair "
                         "none(it) - none(main) is identical by construction)")


def comparisons_from(cells: Sequence[Mapping[str, Any]], *, eight_arm: bool = False) -> list[dict[str, Any]]:
    """Every (model, sampling mode) group the matrix carries, in matrix order:
    ``{model_key, mode, primary, leg_keys: {kind: leg key}}``. Published shape
    (``eight_arm`` False): a group missing one of the three design cells is
    refused (a comparison is three cells). Eight-arm shape: a group needs the
    bare cell and at least one other arm; every arm PRESENT gets its
    ``<arm> - bare`` contrast (:func:`comp_contrasts`) and the panel arms
    absent from the group are named under ``arms_absent`` — reported, never
    refused.

    E3: the groups are per (model, VIEW, sampling mode). A NON-main view's group
    (the it-doc rows) has no bare row of its own and is anchored on the model's
    MAIN-view bare leg of the same mode — the same served target
    (:data:`BARE_ANCHOR_RATIONALE`) — recorded as ``bare_anchor`` on the record,
    which also names its ``view`` (its name is ``<model>/<mode>@<view>``) and,
    per it-doc arm, the lane and the qualification verbatim (``itdoc_arms``);
    eight-arm mode only. The main-view records are unchanged."""
    # the panel this matrix defines (the declared roster + the arms these rows carry): `arms_absent` names the arms the ANALYSIS
    # covers and this group lacks — a registered arm in neither the roster nor any row is not in the panel at all
    matrix_kinds = [a.kind for a in design_arm_roster({str(r["cell"]) for r in cells})]
    by: dict[tuple[str, str, str], dict[str, str]] = {}
    rows_by: dict[tuple[str, str, str], dict[str, Mapping[str, Any]]] = {}
    order: list[tuple[str, str, str]] = []
    carriage: dict[str, dict[str, Any]] = {}
    primary_of_group: dict[tuple[str, str, str], bool] = {}
    for r in cells:
        k = (str(r["model_key"]), key_view(r), str(r["sampling_mode"]))
        if k not in by:
            by[k] = {}
            rows_by[k] = {}
            order.append(k)
            # math500-steer: a group is PRIMARY iff its mode is its rows' ARM's primary row (`card` on the no-switch family)
            primary_of_group[k] = row_is_primary(r)
        elif primary_of_group[k] != row_is_primary(r):
            raise AnalysisError(f"the matrix's {k[0]}/{k[2]} group mixes arms with different primary rows ({row_leg_key(r)})")
        by[k][CELL_KIND[str(r["cell"])]] = row_leg_key(r)
        rows_by[k][CELL_KIND[str(r["cell"])]] = r
        if is_itdoc_row(r):
            carriage[row_leg_key(r)] = itdoc_carriage(r)
    out = []
    for k in order:
        model, view, mode = k
        kinds = set(by[k])
        # the DECLARED expected refusal of the group's comparison (the arm rows' `expected_refusal`, the GSM8K design's field
        # mirrored): recorded only when a row declares one, so every undeclared comparison record is byte-identical
        expected_code, expected_problem = expected_refusal_of_group(rows_by[k])
        declared_fields = ({ROW_EXPECTED_REFUSAL_KEY: expected_code} if expected_code is not None else {})
        if expected_problem is not None:
            declared_fields[EXPECTED_REFUSAL_PROBLEM_KEY] = expected_problem
        if view == VIEW_MAIN:
            if not eight_arm:
                if kinds != set(KINDS):
                    raise AnalysisError(f"the matrix names {sorted(kinds)} for {model}/{mode} — a comparison is the three cells {KINDS}")
                leg_keys = dict(by[k])
            else:
                if KIND_BARE not in kinds:
                    raise AnalysisError(f"the matrix names {sorted(kinds)} for {model}/{mode} — a comparison needs the bare cell")
                if len(kinds) < 2:
                    raise AnalysisError(f"the matrix names only {sorted(kinds)} for {model}/{mode} — no arm to contrast with bare")
                leg_keys = {kind: by[k][kind] for kind in ARM_KINDS if kind in kinds}
            rec: dict[str, Any] = {"name": f"{model}/{mode}", "model_key": model, "mode": mode, "primary": primary_of_group[k],
                                   "leg_keys": leg_keys, **declared_fields}
            if eight_arm:
                rec["arms_absent"] = [kind for kind in matrix_kinds if kind not in kinds]
            out.append(rec)
            continue
        # a NON-main view (E3)
        name = f"{model}/{mode}{LEG_KEY_VIEW_SEP}{view}"
        if not eight_arm:
            raise AnalysisError(f"the matrix names a {view!r}-view group {name} — a view other than {VIEW_MAIN!r} is read in eight-arm "
                                "mode only (a matrix carrying design statuses / waves, or --waves)")
        if KIND_BARE in kinds:
            anchor, anchor_view, why = by[k][KIND_BARE], view, f"the {view!r} view carries a bare row of its own"
        else:
            main_group = by.get((model, VIEW_MAIN, mode), {})
            if KIND_BARE not in main_group:
                raise AnalysisError(f"{name}: the {view!r} view names {sorted(kinds)} and no bare cell, and the matrix names no "
                                    f"{VIEW_MAIN!r}-view {CELL_BARE!r} row at {mode!r} to anchor it on")
            anchor, anchor_view, why = main_group[KIND_BARE], VIEW_MAIN, BARE_ANCHOR_RATIONALE
        arms = [kind for kind in ARM_KINDS if kind in kinds and kind != KIND_BARE]
        if not arms:
            raise AnalysisError(f"{name}: the matrix names only {sorted(kinds)} for the {view!r} view — no arm to contrast with bare")
        leg_keys = {KIND_BARE: anchor, **{kind: by[k][kind] for kind in arms}}
        out.append({"name": name, "model_key": model, "mode": mode, "view": view, "primary": primary_of_group[k], "leg_keys": leg_keys,
                    "arms_absent": [kind for kind in matrix_kinds if kind not in leg_keys],
                    "bare_anchor": {"view": anchor_view, "leg": anchor, "rationale": why},
                    "itdoc_arms": {kind: {"leg": lk, **carriage[lk]} for kind, lk in leg_keys.items() if lk in carriage},
                    **declared_fields})
    return out


def branch_legs(legs: Mapping[str, Mapping[str, Any]], labels: Mapping[str, str], branch: str, *,
                exclude_ids: Sequence[str] = ()) -> dict[str, dict[str, tuple[int, float | None]]]:
    """``{label: pairing input}`` for one branch over ``labels = {label: leg key}``
    — the parent's ``_branch_items`` (PRIMARY masks an unscored draw out;
    SCORE-AS-IS scores it wrong)."""
    return {label: T._branch_items(legs[k], branch, exclude_ids=exclude_ids) for label, k in labels.items()}


def estimands(legs: Mapping[str, Mapping[str, Any]], labels: Mapping[str, str], contrasts: Mapping[str, Mapping[str, int]],
              *, n_resamples: int, seed: int, ci: float, exclude_ids: Sequence[str] = ()) -> dict[str, dict[str, Any]]:
    """``{branch: paired_bootstrap_conditional(...)}`` — the same call, both branches."""
    return {b: paired_bootstrap_conditional(branch_legs(legs, labels, b, exclude_ids=exclude_ids), contrasts,
                                            n_resamples=n_resamples, seed=seed, ci=ci) for b in BRANCHES}


def mode_delta_labels(model_key: str, mode: str, kinds: Sequence[str] = KINDS) -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    """The legs and contrasts of a sensitivity row's per-cell paired delta to
    the primary row: ``<mode>/<kind> - <primary>/<kind>`` for every ``kind``
    (the published design's three by default; in eight-arm mode the arms
    present in BOTH rows, panel order — :func:`delta_kinds`)."""
    labels = {}
    contrasts = {}
    for kind in kinds:
        cell = ARM_BY_KIND[kind].cell
        labels[f"{mode}/{kind}"] = leg_key(model_key, cell, mode)
        labels[f"{MODE_PRIMARY}/{kind}"] = leg_key(model_key, cell, MODE_PRIMARY)
        contrasts[mode_delta_contrast(kind, mode)] = {f"{mode}/{kind}": +1, f"{MODE_PRIMARY}/{kind}": -1}
    return labels, contrasts


def delta_kinds(model_key: str, mode: str, legs: Mapping[str, Any], *, eight_arm: bool = False) -> list[str]:
    """The kinds a sensitivity row's delta pairs: the published three (the design), or — eight-arm mode — every panel
    arm whose leg the matrix names in BOTH the sensitivity row and the primary row, in panel order."""
    if not eight_arm:
        return list(KINDS)
    return [k for k in ARM_KINDS if leg_key(model_key, ARM_BY_KIND[k].cell, mode) in legs
            and leg_key(model_key, ARM_BY_KIND[k].cell, MODE_PRIMARY) in legs]


def _interval_side(low: float, high: float) -> int:
    """+1 when an interval lies above zero, -1 below, 0 when it covers zero."""
    if low > 0.0:
        return 1
    if high < 0.0:
        return -1
    return 0


def _comparable_problem(legs: Mapping[str, Mapping[str, Any]], labels: Mapping[str, str], *, what: str,
                        declared: bool = False, same_sampling: bool = True) -> str | None:
    """Why a set of usable legs has no joint estimand: their attempted id sets
    differ; or — without a declared tree set — they come from different
    producer trees (the published one-sha rule, B9); or — under a declaration
    — the cross-tree compatibility relations refuse (:func:`compatibility_problem`).
    ``same_sampling`` False for a MODE DELTA, whose two rows differ in their
    sampling numbers BY DESIGN (each row's own sampling is held to its own
    matrix row at collection); every other shared field must still agree."""
    if len({frozenset(legs[k]["attempted_ids"]) for k in labels.values()}) != 1:
        return f"{what} attempted id sets differ"
    if not declared:
        return producer_consistency_problem(legs, labels)
    return compatibility_problem(legs, labels, same_sampling=same_sampling)


#: B5 (i): the SHARED-FIELD projection every arm of a contrast must agree on — the identity fields a leg record carries
#: (`collect_leg` files them as strings / bools / the sampling block already on record)
SHARED_PROJECTION_FIELDS: tuple[str, ...] = ("model_id", "model_revision", IDENTITY_KEY_VERSION, C.IDENTITY_KEY_MAX_TOKENS_BY_TASK,
                                             "chat_template_hash", "thinking_mode", "thinking_render", "enable_thinking")


def shared_projection(leg: Mapping[str, Any], *, sampling: bool = True, task: str = TASK) -> dict[str, Any]:
    """B5 (i): the fields equal across every arm of a contrast — the task, the model pin, the identity version, the cap
    statement, the chat template, the thinking policy (mode, render, the enable flag actually sent) and the NUMERIC
    sampling projection (:data:`ROW_NUMERIC_FIELDS` of the manifest's sampling block; `source` prose never enters).
    ``sampling`` False leaves the sampling numbers out (a mode delta's two rows differ in them by design); ``task`` is
    the instrument's task id (the GSM8K analysis passes its own)."""
    ident = _mapping(leg.get("identity"))
    out: dict[str, Any] = {"task": task, **{f: ident.get(f) for f in SHARED_PROJECTION_FIELDS}}
    if sampling:
        out["sampling"] = _numeric_row(_mapping(ident.get("sampling")))
    return out


# --------------------------------------------------------------------------- the launch attestation (code review C2)
#: the standup record's layout beside every served job, as a read-only pull relocates it under the gen root: the manifest's
#: `routing_table` (else `binding.routing_table`) names `<wave root>/serving/<jobdir>/<task>/routing_table.json`; only that
#: trailing layout is read (the validators' `standup_record_dir` rule), so a pulled copy resolves under the leg's OWN gen root
LAUNCH_STANDUP_DIR_PARENT = "serving"
LAUNCH_STANDUP_TASK_SUBDIR = "math500"                     # this instrument's; the GSM8K analysis passes its own
LAUNCH_ROUTING_TABLE = "routing_table.json"
#: the files the attestation is read from: the ENGINE's launch record — ONE per serving engine, selected by the manifest's
#: `route.serving_engine` (:func:`launch_engine_of`), never by which file happens to exist — and the server log. The LENS
#: engine's record is the workers' attestation (`vllm_server.launch_server` writes `steering_attestation.json` beside the
#: server log — `slurm_job_id`, the bound `port`, an argv digest); the NATIVE (lora_native) engine's is the readiness record
#: `serving_attestation.json` (`serving_attestation.SERVING_ATTESTATION_FILE`, `kind: lora_native`, the same three binding
#: fields at its top level), which a lora-native standup writes INSTEAD of the lens file by design and which is read
#: through the STRICT reader `read_serving_attestation` (every schema key, the kind, an asserted name binding, a graphs-on
#: engine — a `ServingAttestationError` refuses by name, never a lenient parse). Both engines' servers write the SAME
#: `vllm_server.log`, whose API-server startup echo (`api_utils.py:273 non-default args: {...}`) is the ONE surviving
#: record of the composed launch: every CLI arg that differs from its default (dtype, revision, tokenizer revision, default
#: chat-template kwargs, generation-config override, reasoning parser, prefix caching, quantisation, the Mamba cache dtype,
#: the native engine's LoRA flags, …). `endpoint.json` is unlinked at teardown and never carried those flags; the records
#: carry only their digest. Before the native record was selected here (2026-09-16) every natively served leg read
#: `absent` with "no workers' attestation" and the GSM8K contrast refused it.
LAUNCH_ATTESTATION_FILE = "steering_attestation.json"
LAUNCH_NATIVE_ATTESTATION_FILE = SA.SERVING_ATTESTATION_FILE
LAUNCH_RECORD_FILE_BY_ENGINE: Mapping[str, str] = {C.SERVING_ENGINE_LENS: LAUNCH_ATTESTATION_FILE,
                                                   C.SERVING_ENGINE_LORA_NATIVE: LAUNCH_NATIVE_ATTESTATION_FILE}
#: the standup's SERVING-CONFIG provenance record beside it (mt_serving_up.SERVING_CONFIG_RECORD_FILE): the sha256
#: of the config the leg ACTUALLY served. A leg whose store predates the record reads `serving_config_sha256: null`
#: with `serving_config_record: absent` — RECORDED, never a refusal (every published leg predates it), and never
#: compared across arms (each row may serve its own overlay by design).
LAUNCH_SERVING_CONFIG_FILE = "serving_config.json"
LAUNCH_SERVING_CONFIG_ABSENT = "absent"
LAUNCH_SERVING_CONFIG_READ = "read"
LAUNCH_SERVING_CONFIG_UNREADABLE = "unreadable"
LAUNCH_LOG_FILE = "vllm_server.log"
LAUNCH_ECHO_MARKER = "non-default args: "
LAUNCH_ATTESTED, LAUNCH_ABSENT = "attested", "absent"
#: the echoed keys NOT compared across arms, each with its reason (plan P5 (i″): "the two INTENTIONAL differences … and every
#: other flag difference a named refusal"): the row's geometry (validated per row in (i′)), request logging (P6's opt-in:
#: logging only, never behaviour) and the launch INSTANCE (the bound address, per-job paths — the template's DIGEST is in the
#: shared projection —, the wrapper's fixed plumbing, memory sizing)
#: O106: which adapters a native standup loads and how many changes nothing in the base computation any request sees; the
#: served OBJECT is pinned separately (provenance.artifact_sha256)
CROSS_ENGINE_RULING = "O106"
LAUNCH_CELL_SET_PLUMBING_WHY = ("cell-set plumbing: which adapters are loaded and how many; changes nothing in the base computation any request "
                                f"sees ({CROSS_ENGINE_RULING}; the served object is pinned separately by provenance.artifact_sha256)")
LAUNCH_EXCLUDED_KEYS: Mapping[str, str] = {
    "max_model_len": "the row's geometry: validated per row against the row's resolved contract (geometry_contract_problem) and reconciled "
                     "with this echo (launch_attestation), never compared across arms",
    "max_num_seqs": "the row's geometry: validated per row against the row's resolved contract (geometry_contract_problem) and reconciled "
                    "with this echo (launch_attestation), never compared across arms",
    "max_num_batched_tokens": "max_num_seqs * max_model_len — the row's geometry, reconciled with this echo (launch_attestation)",
    "enable_log_requests": "logging only, never behaviour (P6's opt-in)",
    "port": "the launch instance's bound port", "host": "the launch instance's bind address",
    "chat_template": "a per-job PATH to the pinned template (its digest, chat_template_hash, is compared in the shared projection)",
    "chat_template_content_format": "the template file's format, fixed by the composer",
    "middleware": "the wrapper's auth middleware (fixed)", "worker_extension_cls": "the wrapper's worker extension (fixed)",
    "model_tag": "the positional model spelling (the resolved model and revision are compared)",
    "model": "the model spelling (the resolved model and revision are compared)",
    "gpu_memory_utilization": "memory sizing, not decode behaviour", "async_scheduling": "scheduler plumbing pinned by the composer",
    # the native engine's CELL-SET plumbing is excluded on a same-engine contrast; enable_lora, max_lora_rank and
    # lora_target_modules stay COMPARED (they select the LoRA kernel configuration applied on every request, the zero
    # adapter included), as do enforce_eager, cudagraph_mode, cudagraph_capture_sizes and everything else not named
    # here
    "lora_modules": LAUNCH_CELL_SET_PLUMBING_WHY, "max_loras": LAUNCH_CELL_SET_PLUMBING_WHY, "max_cpu_loras": LAUNCH_CELL_SET_PLUMBING_WHY,
}
LAUNCH_REMEDY = ("pull the job's serving/<jobdir>/<task>/{steering_attestation.json | serving_attestation.json,vllm_server.log} beside "
                 "the stores under the leg's gen root — the lens workers' attestation for a lens leg, the native serving attestation for a "
                 "route.serving_engine lora_native leg, and the server log for both (pull_wave.sh carries all three names per standup dir; "
                 "a pull made before it did must be re-pulled)")
_LAUNCH_JOB_SUFFIX = re.compile(r"_(\d+)$")


def standup_dir_of(manifest: Mapping[str, Any], *, gen_root: Path | str, task_subdir: str = LAUNCH_STANDUP_TASK_SUBDIR
                   ) -> tuple[Path | None, str | None, str | None]:
    """``(<gen_root>/serving/<jobdir>/<task>, <jobdir>, problem)`` from the manifest's own ``routing_table`` path (top level,
    else ``binding.routing_table``): the ABSOLUTE cluster path, of which only the trailing
    ``serving/<jobdir>/<task>/routing_table.json`` is read. The problem names what is missing — never a fall-through."""
    path = manifest.get("routing_table") or _mapping(manifest.get(MANIFEST_KEY_BINDING)).get("routing_table")
    if not isinstance(path, str) or not path:
        return None, None, "the manifest names no routing_table path — the standup record cannot be located"
    parts = PurePosixPath(path).parts
    if len(parts) < 4 or parts[-1] != LAUNCH_ROUTING_TABLE or parts[-2] != task_subdir or parts[-4] != LAUNCH_STANDUP_DIR_PARENT:
        return None, None, (f"manifest routing_table {path!r} does not end in {LAUNCH_STANDUP_DIR_PARENT}/<jobdir>/{task_subdir}/"
                            f"{LAUNCH_ROUTING_TABLE} — not a standup record path")
    jobdir = parts[-3]
    d = Path(gen_root) / LAUNCH_STANDUP_DIR_PARENT / jobdir / task_subdir
    if not d.is_dir():
        return None, jobdir, f"no standup record at {d} — {LAUNCH_REMEDY}"
    return d, jobdir, None


LAUNCH_STANDUP_JOBS_FILE = "standup_jobs.jsonl"


def standup_record_summary(manifest: Mapping[str, Any], *, gen_root: Path | str, task_subdir: str = LAUNCH_STANDUP_TASK_SUBDIR) -> dict[str, Any]:
    """``{standup_sha, producer_tree_sha, job_id, jobdir, problem}`` — the tree the STANDUP ran from (``standup_jobs.jsonl[0].sha``
    under the standup dir the manifest's own ``routing_table`` path locates, :func:`standup_dir_of`) beside the DRIVER's tree
    (``producer.tree_sha``). For these launchers the standup and the driver run from one eval-repo checkout, so the two must agree;
    the paper-rows export files the standup's as ``serving_commit`` when they do and refuses by name when they do not (the GSM8K
    reader's ``native_served_positions_record`` files the same two fields). ``problem`` names a record that could not be read;
    nothing is ever defaulted."""
    out: dict[str, Any] = {"standup_sha": None, "producer_tree_sha": _mapping(manifest.get(C.MANIFEST_KEY_PRODUCER)).get("tree_sha"),
                           "job_id": None, "jobdir": None, "problem": None}
    d, jobdir, problem = standup_dir_of(manifest, gen_root=gen_root, task_subdir=task_subdir)
    out["jobdir"] = jobdir
    if problem or d is None:
        out["problem"] = problem
        return out
    path = d / LAUNCH_STANDUP_JOBS_FILE
    if not path.is_file():
        out["problem"] = f"no {LAUNCH_STANDUP_JOBS_FILE} under {d}"
        return out
    jobs = [j for j in _iter_jsonl(path) if isinstance(j, Mapping)]
    if not jobs:
        out["problem"] = f"{path}: records no job"
        return out
    out["standup_sha"], out["job_id"] = jobs[0].get("sha"), (str(jobs[0]["job_id"]) if jobs[0].get("job_id") is not None else None)
    return out


#: the ONE non-literal shape a startup echo may carry, read off a real lora-native standup's log (2026-09-16): vLLM parses
#: `--lora-modules` into `LoRAModulePath` dataclasses and echoes them by their repr — `LoRAModulePath(name=..., path=...,
#: base_model_name=None, is_3d_lora_weight=False)` — inside the `lora_modules` list, so the native echo is NOT an
#: `ast.literal_eval` literal (a lens echo is). Read STRICTLY by :func:`launch_echo_reading`: a call to exactly this bare
#: name, no positional argument, every keyword a Python LITERAL, into `{"LoRAModulePath": {kw: value}}` (the constructor
#: kept, so the projection is reversible); ANY other non-literal node — another constructor, a positional argument, a call
#: nested in a keyword, a name, an attribute — refuses by name (the echo key and the offending source), never a lenient
#: parse and never the old "no startup echo" reading, which was FALSE on a native log (the marker is there).
LAUNCH_ECHO_CONSTRUCTORS: tuple[str, ...] = ("LoRAModulePath",)
LAUNCH_ECHO_NO_MARKER = f"no `{LAUNCH_ECHO_MARKER.strip()}` startup echo — not a vLLM API-server log of the pinned version"


class _EchoShape(ValueError):
    """The startup echo's body carries a node the reader does not understand (the echo key and the source named)."""


def _echo_value(node: ast.AST, *, key: str, body: str, in_constructor: bool = False) -> Any:
    """A Python literal from an AST node — the set ``ast.literal_eval`` admits (constants, lists, tuples, sets, dicts,
    signed numbers) — plus the ONE admitted constructor shape (:data:`LAUNCH_ECHO_CONSTRUCTORS`, keyword-only, literal
    keywords, never nested in another constructor's keyword). Anything else raises :class:`_EchoShape` naming ``key``."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_echo_value(e, key=key, body=body, in_constructor=in_constructor) for e in node.elts]
    if isinstance(node, ast.Set):
        return {_echo_value(e, key=key, body=body, in_constructor=in_constructor) for e in node.elts}
    if isinstance(node, ast.Dict) and all(k is not None for k in node.keys):
        return {_echo_value(k, key=key, body=body, in_constructor=in_constructor): _echo_value(v, key=key, body=body, in_constructor=in_constructor)
                for k, v in zip(node.keys, node.values)}
    if (isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)) and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, (int, float, complex)) and not isinstance(node.operand.value, bool)):
        return -node.operand.value if isinstance(node.op, ast.USub) else node.operand.value
    if (not in_constructor and isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in LAUNCH_ECHO_CONSTRUCTORS
            and not node.args and node.keywords and all(kw.arg is not None for kw in node.keywords)):
        return {node.func.id: {kw.arg: _echo_value(kw.value, key=key, body=body, in_constructor=True) for kw in node.keywords}}
    segment = ast.get_source_segment(body, node) or ast.dump(node)
    raise _EchoShape(f"the echo's {key!r} carries a non-literal value the reader does not understand: {segment[:160]!s} — "
                     f"the admitted constructors are {list(LAUNCH_ECHO_CONSTRUCTORS)} (keyword-only, literal keywords)")


def launch_echo_reading(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """``(echo, problem)`` — the API server's startup echo of the composed launch (the FIRST ``non-default args: {...}``
    line, a Python dict repr parsed through :func:`_echo_value`, never ``eval``), or ``None`` with the problem named: no
    marker line at all (:data:`LAUNCH_ECHO_NO_MARKER`), a body that is not a dict expression, a non-string key, or a value
    outside the admitted shapes (the key and the source named — the native engine's ``LoRAModulePath`` reprs are the one
    admitted non-literal; a lens echo is a plain literal)."""
    for line in text.splitlines():
        i = line.find(LAUNCH_ECHO_MARKER)
        if i < 0:
            continue
        body = line[i + len(LAUNCH_ECHO_MARKER):].strip()
        try:
            tree = ast.parse(body, mode="eval")
        except SyntaxError as e:
            return None, f"the `{LAUNCH_ECHO_MARKER.strip()}` echo's body is not a Python expression ({e.msg})"
        node = tree.body
        if not isinstance(node, ast.Dict):
            return None, f"the `{LAUNCH_ECHO_MARKER.strip()}` echo's body is not a dict repr ({type(node).__name__})"
        out: dict[str, Any] = {}
        for k, v in zip(node.keys, node.values):
            if not (isinstance(k, ast.Constant) and isinstance(k.value, str)):
                return None, (f"the `{LAUNCH_ECHO_MARKER.strip()}` echo carries a non-string key "
                              f"{(ast.get_source_segment(body, k) if k is not None else '**') !s}")
            try:
                out[k.value] = _echo_value(v, key=k.value, body=body)
            except _EchoShape as e:
                return None, str(e)
        return out, None
    return None, LAUNCH_ECHO_NO_MARKER


def launch_echo(text: str) -> dict[str, Any] | None:
    """The startup echo as a dict, or ``None`` when :func:`launch_echo_reading` names a problem."""
    return launch_echo_reading(text)[0]


LAUNCH_GEOMETRY_KEYS: tuple[str, ...] = ("max_model_len", "max_num_seqs")


def _echo_geometry_problems(echo: Mapping[str, Any], served: Mapping[str, Any], log_path: Path) -> list[str]:
    """Why the launch echo's geometry is not the geometry the manifest records as served: ``max_model_len`` /
    ``max_num_seqs`` differ, or ``max_num_batched_tokens`` is not served S × served L (the composer's no-chunking
    identity). An echo without a geometry key says nothing about it and is not held to it."""
    out: list[str] = []
    for key in LAUNCH_GEOMETRY_KEYS:
        if key in echo and echo.get(key) != served.get(key):
            out.append(f"{log_path}: the echo's {key} {echo.get(key)!r} != the manifest's served {key} {served.get(key)!r} — the log is "
                       "another launch's, or the record was edited")
    s, l = _int_field(served.get("max_num_seqs")), _int_field(served.get("max_model_len"))
    if "max_num_batched_tokens" in echo and s is not None and l is not None and echo.get("max_num_batched_tokens") != s * l:
        out.append(f"{log_path}: the echo's max_num_batched_tokens {echo.get('max_num_batched_tokens')!r} != served max_num_seqs {s} x "
                   f"max_model_len {l} = {s * l:,} — not the composer's no-chunking identity over the recorded geometry")
    return out


def launch_engine_of(manifest: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """``(engine, problem)`` — the SERVING ENGINE whose launch record the leg is read from: the manifest's
    ``route.serving_engine`` (:data:`contracts.SERVED_ENGINE_FIELD`; PRESENT IFF NATIVE by the stored-record contract, so an
    absent stamp or an explicit ``lens`` is the lens engine and ``lora_native`` the native one), never which record file
    happens to exist beside the log. A PRESENT value outside :data:`contracts.SERVING_ENGINES` (a null, a typo, an unknown
    engine), or a top-level ``serving_engine`` stamp that disagrees with the route's, is torn: ``engine`` None and the
    problem named — no record is selected for a leg whose engine cannot be read."""
    route = _mapping(manifest.get(C.MANIFEST_KEY_ROUTE))
    field = f"{C.MANIFEST_KEY_ROUTE}.{C.SERVED_ENGINE_FIELD}"
    stated = route.get(C.SERVED_ENGINE_FIELD, _ABSENT)
    if stated is not _ABSENT and stated not in C.SERVING_ENGINES:
        return None, (f"{field} {stated!r} is not one of {list(C.SERVING_ENGINES)} (a present null, a typo or an unknown engine) — the "
                      "launch record cannot be selected by engine")
    engine = C.SERVING_ENGINE_LENS if stated is _ABSENT else str(stated)
    top = manifest.get(C.SERVED_ENGINE_FIELD, _ABSENT)
    if top is not _ABSENT and top != engine:
        return None, (f"the manifest's top-level {C.SERVED_ENGINE_FIELD} {top!r} disagrees with {field} "
                      f"{'absent' if stated is _ABSENT else repr(stated)} (read as {engine!r}) — a torn record; the launch record cannot be "
                      "selected by engine")
    return engine, None


def launch_attestation(manifest: Mapping[str, Any], *, gen_root: Path | str, task_subdir: str = LAUNCH_STANDUP_TASK_SUBDIR) -> dict[str, Any]:
    """The leg's LAUNCH ATTESTATION: the standup dir located through the manifest, the leg's ENGINE read off the
    manifest (:func:`launch_engine_of`) and THAT engine's launch record selected by name
    (:data:`LAUNCH_RECORD_FILE_BY_ENGINE`: the lens workers' attestation, or the native serving attestation read through
    the strict :func:`serving_attestation.read_serving_attestation`) — the selected record and the server log both present,
    the OTHER engine's record absent (a standup writes exactly one engine's record; both beside one log is a torn / mixed
    standup dir and refuses by name) — the echo parsed (:func:`launch_echo_reading`) and BOUND to the leg: the record's
    ``slurm_job_id`` == the jobdir's job id == the manifest's ``row_constants.slurm_job_id`` (when the manifest carries one),
    the echo's ``port`` == the record's ``port`` — projected onto every echoed key outside :data:`LAUNCH_EXCLUDED_KEYS`
    (values as canonical JSON strings — B1) plus the engine's generation-config-override reading and the request-logging
    attestation, read through the recorder (:func:`RS.read_log`). ``engine`` and ``record_file`` say which record
    attested. ``status`` is ``attested`` only when every check held; otherwise ``absent`` with every problem named — the
    comparison under a declared tree set refuses on it, the record never claims equality over it."""
    d, jobdir, problem = standup_dir_of(manifest, gen_root=gen_root, task_subdir=task_subdir)
    engine, engine_problem = launch_engine_of(manifest)
    record_file = LAUNCH_RECORD_FILE_BY_ENGINE[engine] if engine is not None else None
    out: dict[str, Any] = {"status": LAUNCH_ABSENT, "engine": engine, "record_file": record_file,
                           "standup_dir": str(d) if d else None, "jobdir": jobdir, "job_id": None,
                           "argv_sha256": None, "settings": None, "excluded": None, "generation_config_override": None,
                           "log_requests_attested": None, "geometry_reconciled": None,
                           "serving_config_sha256": None, "serving_config": None,
                           "serving_config_record": LAUNCH_SERVING_CONFIG_ABSENT, "problems": [],
                           "rule": ("the API server's startup echo of every non-default launch flag, from the standup's own log, bound to "
                                    "the leg by the ENGINE's launch record — the lens workers' attestation (steering_attestation.json) or "
                                    "the native serving attestation (serving_attestation.json, read strictly), selected by the manifest's "
                                    "route.serving_engine and never by which file exists — on job id and port, and reconciled with the "
                                    "manifest's served geometry (window, S, S x L); every key compared except the named exclusions")}
    if engine_problem:
        out["problems"].append(engine_problem)
    if problem:
        out["problems"].append(problem)
    if out["problems"]:
        return out
    assert d is not None and jobdir is not None and engine is not None and record_file is not None
    problems: list[str] = []
    att_path, log_path = d / record_file, d / LAUNCH_LOG_FILE
    att: Mapping[str, Any] | None = None
    if engine == C.SERVING_ENGINE_LORA_NATIVE:
        # the NATIVE record, through the strict reader: a lens file handed here, a foreign or older record, a non-asserting
        # binding or an eager engine all refuse by name — never a lenient parse of whatever JSON sits at that path
        if not att_path.is_file():
            problems.append(f"{att_path}: no serving attestation ({record_file}, the {engine} engine's launch record) — {LAUNCH_REMEDY}")
        else:
            try:
                att = SA.read_serving_attestation(att_path)
            except SA.ServingAttestationError as e:
                problems.append(f"{att_path}: not a strict {SA.SERVING_ATTESTATION_KIND} serving attestation — {e}")
    else:
        # the LENS record, exactly as before the native record was selected here
        lens_att = _read_json(att_path) if att_path.is_file() else None
        if isinstance(lens_att, Mapping):
            att = lens_att
        else:
            problems.append(f"{att_path}: no workers' attestation ({record_file}, the {engine} engine's launch record) — {LAUNCH_REMEDY}")
    for other_engine, other_file in LAUNCH_RECORD_FILE_BY_ENGINE.items():
        if other_engine != engine and (d / other_file).is_file():
            problems.append(f"{d / other_file}: the {other_engine} engine's launch record beside a {engine} leg's standup — a torn / mixed "
                            f"standup dir (the manifest's {C.MANIFEST_KEY_ROUTE}.{C.SERVED_ENGINE_FIELD} reads {engine!r}; a standup writes "
                            "exactly one engine's record)")
    if not log_path.is_file():
        problems.append(f"{log_path}: no server log — {LAUNCH_REMEDY}")
    if problems or att is None:
        out["problems"] = problems
        return out
    att = dict(att)
    m = _LAUNCH_JOB_SUFFIX.search(jobdir)
    job_dir_id = m.group(1) if m else None
    job_att = att.get("slurm_job_id")
    out["job_id"] = str(job_att) if job_att is not None else None
    out["argv_sha256"] = att.get("argv_sha256")
    if job_dir_id is None or str(job_att) != job_dir_id:
        problems.append(f"{att_path}: slurm_job_id {job_att!r} != the standup directory's job {job_dir_id!r} — the attestation is another job's")
    job_manifest = _mapping(manifest.get("row_constants")).get("slurm_job_id")
    if job_manifest is not None and str(job_manifest) != str(job_att):
        problems.append(f"the manifest's row_constants.slurm_job_id {job_manifest!r} != the attestation's {job_att!r}")
    text = log_path.read_text(encoding="utf-8", errors="replace")
    echo, echo_problem = launch_echo_reading(text)
    if echo is None:
        problems.append(f"{log_path}: {echo_problem}")
    else:
        if att.get("port") is None or echo.get("port") != att.get("port"):
            problems.append(f"{log_path}: the echo's port {echo.get('port')!r} != the attestation's {att.get('port')!r} — the log is another launch's")
        # the echo's GEOMETRY is reconciled with the geometry the manifest RECORDS as served (the arm contract holds
        # that record to the row); the keys stay out of the cross-arm comparison, but an echo that names another window
        # or concurrency is not this leg's launch — or the record was edited — and never attests it
        geometry_problems = _echo_geometry_problems(echo, _mapping(_mapping(manifest.get(C.MANIFEST_KEY_TINYCAP)).get("served")), log_path)
        problems += geometry_problems
        out["geometry_reconciled"] = not geometry_problems
        out["settings"] = {k: json.dumps(v, sort_keys=True, default=str) for k, v in echo.items() if k not in LAUNCH_EXCLUDED_KEYS}
        out["excluded"] = {k: {"value": json.dumps(echo[k], sort_keys=True, default=str), "why": LAUNCH_EXCLUDED_KEYS[k]}
                           for k in echo if k in LAUNCH_EXCLUDED_KEYS}
    # the SERVING-CONFIG provenance: the sha256 of the config the leg served, from the standup's own record. Absent
    # on any leg produced before the record existed — reported as `absent` with a null digest, never a problem: the
    # attestation's job is to say WHAT was served, and "not recorded" is an honest answer (the alternative, inferring
    # the config from the wave sha, is exactly the inference this field exists to replace).
    out.update(serving_config_record(d))
    read = RS.read_log(log_path)
    gco = _mapping(read.get("generation_config_override"))
    out["generation_config_override"] = {"present": bool(gco.get("present")),
                                         "params": json.dumps(list(gco.get("lines") or []), sort_keys=True, default=str),
                                         # the server default the ENGINE stated, as a dict (canonical JSON; None when
                                         # the WARNING is absent = vLLM's neutral defaults govern every omitted key)
                                         "server_default": _override_params_json(gco.get("lines")),
                                         "reading": gco.get("reading")}
    out["log_requests_attested"] = bool(read.get("log_requests_flag_attested"))
    out["problems"] = problems
    if not problems:
        out["status"] = LAUNCH_ATTESTED
    return out


def serving_config_record(standup_dir: Path) -> dict[str, Any]:
    """``{serving_config_sha256, serving_config, serving_config_record}`` from the standup's
    ``serving_config.json`` (:data:`LAUNCH_SERVING_CONFIG_FILE`) — the record mt_serving_up writes BEFORE the engine
    starts, naming the config + manifest that leg serves. ``absent`` (null digest) when the file is not there (a leg
    older than the record, or a pull that predates it); ``unreadable`` with the reason when it is there and malformed;
    ``read`` with the 64-hex digest otherwise. Never raises, never a ``problem``: this is provenance the comparison
    records, not a gate it fails."""
    path = Path(standup_dir) / LAUNCH_SERVING_CONFIG_FILE
    if not path.is_file():
        return {"serving_config_sha256": None, "serving_config": None,
                "serving_config_record": LAUNCH_SERVING_CONFIG_ABSENT}
    try:
        doc = _read_json(path)
    except (OSError, ValueError) as e:
        return {"serving_config_sha256": None, "serving_config": str(path),
                "serving_config_record": f"{LAUNCH_SERVING_CONFIG_UNREADABLE}: {type(e).__name__}: {e}"}
    sha = (doc or {}).get("serving_config_sha256")
    if not (isinstance(sha, str) and C.is_sha256_hex(sha)):
        return {"serving_config_sha256": None, "serving_config": (doc or {}).get("config"),
                "serving_config_record": f"{LAUNCH_SERVING_CONFIG_UNREADABLE}: serving_config_sha256 {sha!r} is not 64-hex"}
    return {"serving_config_sha256": sha, "serving_config": (doc or {}).get("config"),
            "serving_config_record": LAUNCH_SERVING_CONFIG_READ}


#: the two launch records of the SERVER'S DEFAULT SAMPLING — the composed `--override-generation-config` echo and the
#: engine's "overridden by generation_config.json" WARNING reading. Compared by EFFECT (:func:`effective_decode_problem`),
#: reported literally (`server_settings_literal_equal`); `server_projection(..., override=False)` leaves them out
LAUNCH_OVERRIDE_ECHO_KEY = "override_generation_config"
LAUNCH_OVERRIDE_READING_KEY = "launch.generation_config_override"
#: the sampler keys vLLM CAN fill from a server default (contracts, pinned to vLLM 0.25.1's `get_diff_sampling_param`):
#: temperature, top_p, top_k, min_p, repetition_penalty — presence_penalty is an OpenAI-protocol field that defaults to 0.0
#: per request and is never lifted from a generation config
SERVER_DEFAULTABLE_KEYS: tuple[str, ...] = tuple(C.SERVER_DEFAULTABLE_SAMPLING_KEYS)
#: review H2 (2026-09-13), CORRECTED by the second reader's M-a and made EXHAUSTIVE by the second reader's round 2 (M1): the
#: generation config's OUTPUT LIMIT — `max_new_tokens` in a generation_config.json and in the composed
#: `--override-generation-config` flag, which vLLM 0.25.1 renames to `max_tokens` in its override WARNING and in
#: `default_sampling_params`. What it does to a request depends on its PROVENANCE — read from the v0.25.1 sources
#: (`vllm/config/model.py` `get_diff_sampling_param`; `vllm/entrypoints/openai/chat_completion/serving.py` `__init__` and the
#: request path; `vllm/entrypoints/serve/utils/api_utils.py` `get_max_tokens`):
#:   * `get_diff_sampling_param` starts from the FILE under `--generation-config auto` (the default: the model's
#:     generation_config.json) or under a custom PATH, and from an EMPTY dict under `vllm`
#:     (`config = {} if src == "vllm" else self.try_get_generation_config()`), applies the override FLAG last
#:     (`config.update(self.override_generation_config)`), renames `max_new_tokens` -> `max_tokens`, and prints the WARNING
#:     with the MERGED dict ONLY when the diff is non-empty AND the source is not `vllm`
#:     (`if diff_sampling_param and src != "vllm"`) — so under `vllm` NO file is loaded and NO WARNING is ever printed,
#:     whatever the flag says;
#:   * the serving layer sets `override_max_tokens = default_sampling_params["max_tokens"]` when `generation_config` is a
#:     PATH, else `override_generation_config.get("max_new_tokens")` — the FLAG's value, never the file's;
#:   * `get_max_tokens` takes `fallback = the request's max_tokens if sent, else default_sampling_params["max_tokens"]` and
#:     serves `min(window remainder, fallback, override_max_tokens, platform max)` over the non-None values.
#: So under `auto` the file's limit is a FALLBACK that an explicit request `max_tokens` REPLACES (MATH-500's requests send
#: the identity's cap on every request); the flag's `max_new_tokens` is a CLIP under every source; the merged `max_tokens`
#: under a custom PATH is a CLIP; and under `vllm` there is no file limit at all. The states are EXHAUSTIVE
#: (:data:`OUTPUT_LIMIT_STATES`) and the analysis FAILS CLOSED on the two records the engine cannot have produced or this
#: analysis cannot read: a WARNING recorded under `vllm` CONTRADICTS the engine (revision 3 read it as a fallback), and a
#: custom PATH with neither a WARNING (the merged dict — the only attestation of the file's contents) nor a flag naming the
#: limit is an UNKNOWN configuration (revision 3 read `merged is None` as no limit). Revision 2 had unioned the two records
#: without provenance and read every server limit as a clip; revision 3 keeps the records apart
#: (:func:`effective_output_limit`), refuses on a clip below the client cap and on records that disagree, and REPORTS a
#: fallback beside the cap it does not reach.
OVERRIDE_OUTPUT_LIMIT_KEYS: tuple[str, ...] = ("max_new_tokens", "max_tokens")
#: the flag's key (a CLIP under every generation-config source) and the WARNING's key (the MERGED reading)
OVERRIDE_FLAG_LIMIT_KEY, OVERRIDE_WARNING_LIMIT_KEY = "max_new_tokens", "max_tokens"
#: the echoed `--generation-config`; absent from the non-default-args echo = vLLM's default `auto`. THREE sources: `auto`
#: loads the model's file (its limit a FALLBACK; the WARNING prints the merged dict), `vllm` loads NO file and prints NO
#: WARNING (only the flag applies), any other value is a custom PATH whose merged limit CLIPS and whose contents reach this
#: analysis only through the WARNING or a flag naming the limit
GENERATION_CONFIG_ECHO_KEY = "generation_config"
GENERATION_CONFIG_SOURCE_AUTO, GENERATION_CONFIG_SOURCE_VLLM = "auto", "vllm"
GENERATION_CONFIG_NAMED_SOURCES: tuple[str, ...] = (GENERATION_CONFIG_SOURCE_AUTO, GENERATION_CONFIG_SOURCE_VLLM)
#: the EXHAUSTIVE output-limit states (second reader round 2, M1), `<source>/<what governs>`; the three marked REFUSED set
#: `problem` and refuse the contrast: records that disagree, a WARNING recorded under `vllm` (a source that prints none), a
#: custom PATH with no attested limit
OUTPUT_LIMIT_STATE_AUTO_FLAG_CLIP, OUTPUT_LIMIT_STATE_AUTO_FILE_FALLBACK, OUTPUT_LIMIT_STATE_AUTO_NO_LIMIT = (
    "auto/flag clip", "auto/model-file fallback", "auto/no model-file limit")
OUTPUT_LIMIT_STATE_VLLM_FLAG_CLIP, OUTPUT_LIMIT_STATE_VLLM_NO_LIMIT, OUTPUT_LIMIT_STATE_VLLM_CONTRADICTORY = (
    "vllm/flag clip", "vllm/no limit", "vllm/CONTRADICTORY (a WARNING recorded) — REFUSED")
OUTPUT_LIMIT_STATE_PATH_MERGED_CLIP, OUTPUT_LIMIT_STATE_PATH_FLAG_CLIP, OUTPUT_LIMIT_STATE_PATH_ATTESTED_NO_LIMIT, OUTPUT_LIMIT_STATE_PATH_UNKNOWN = (
    "path/merged clip", "path/flag clip (no WARNING recorded)", "path/attested no limit", "path/UNKNOWN (no attested limit) — REFUSED")
OUTPUT_LIMIT_STATE_DISAGREE = "records disagree — REFUSED"
OUTPUT_LIMIT_STATES: tuple[str, ...] = (
    OUTPUT_LIMIT_STATE_AUTO_FLAG_CLIP, OUTPUT_LIMIT_STATE_AUTO_FILE_FALLBACK, OUTPUT_LIMIT_STATE_AUTO_NO_LIMIT,
    OUTPUT_LIMIT_STATE_VLLM_FLAG_CLIP, OUTPUT_LIMIT_STATE_VLLM_NO_LIMIT, OUTPUT_LIMIT_STATE_VLLM_CONTRADICTORY,
    OUTPUT_LIMIT_STATE_PATH_MERGED_CLIP, OUTPUT_LIMIT_STATE_PATH_FLAG_CLIP, OUTPUT_LIMIT_STATE_PATH_ATTESTED_NO_LIMIT,
    OUTPUT_LIMIT_STATE_PATH_UNKNOWN, OUTPUT_LIMIT_STATE_DISAGREE)
OUTPUT_LIMIT_REFUSED_STATES: tuple[str, ...] = (OUTPUT_LIMIT_STATE_VLLM_CONTRADICTORY, OUTPUT_LIMIT_STATE_PATH_UNKNOWN, OUTPUT_LIMIT_STATE_DISAGREE)
OUTPUT_LIMIT_RULE = ("the server's output limit is read by PROVENANCE (vLLM 0.25.1) over EXHAUSTIVE states: under --generation-config auto the "
                     "model file's max_new_tokens (the WARNING's max_tokens) is a FALLBACK that an explicit request max_tokens replaces — "
                     "MATH-500's requests send the identity's cap on every request, so it does not reach the served decode and is reported, "
                     "not refused; the flag's --override-generation-config max_new_tokens is a CLIP under every source, and so is the merged "
                     "max_tokens under a custom generation-config PATH: the served cap is min(client cap, clip), and a clip below the client "
                     "cap refuses the contrast; under --generation-config vllm no file is loaded and no WARNING is printed, so a recorded "
                     "WARNING there is CONTRADICTORY and refuses; a custom PATH with neither a WARNING (the merged dict) nor a flag naming the "
                     "limit is UNKNOWN and refuses; records that disagree (the flag and the WARNING both name a limit and differ) refuse")
#: the ONLY generation-config keys the effect rule understands: the five sampling keys (by effect over the keys the
#: requests do not send) and the output limit (by its effective value). Any other key in EITHER side's override — seed,
#: logit_bias, stop, n, best_of, ignore_eos, … — refuses the contrast by name: REFUSE-UNLESS-UNDERSTOOD (review H2; the
#: rule used to be waive-unless-listed, and a server-wide max_new_tokens passed as `server_settings_equal true`)
OVERRIDE_UNDERSTOOD_KEYS: tuple[str, ...] = SERVER_DEFAULTABLE_KEYS + OVERRIDE_OUTPUT_LIMIT_KEYS
EFFECTIVE_SOURCE_CLIENT, EFFECTIVE_SOURCE_SERVER = "client", "server_default"


def _numeric_value(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _override_dict(lines: Any) -> dict[str, Any] | None:
    """The server default the engine's override WARNING(s) state, merged in log order into ONE dict — EVERY key it names
    (a key the effect rule does not understand must be SEEN so it can refuse, never dropped at the parse), numeric
    values as floats (so `20` and `20.0` are one value), other values as parsed. ``None`` when no WARNING was logged
    (vLLM's neutral defaults govern every key a request omits). A line whose params do not parse contributes nothing
    (the literal text stays on record)."""
    merged: dict[str, Any] = {}
    for line in list(lines or []):
        raw = _mapping(line).get("params")
        try:
            parsed = ast.literal_eval(str(raw)) if raw is not None else None
        except (ValueError, SyntaxError):
            parsed = None
        if isinstance(parsed, Mapping):
            for k, v in parsed.items():
                merged[str(k)] = float(v) if _numeric_value(v) else v
    return merged if lines else None


def _override_params_json(lines: Any) -> str | None:
    """:func:`_override_dict` as canonical JSON (the attestation's ``server_default`` field — a string, B1), else ``None``."""
    d = _override_dict(lines)
    return json.dumps(d, sort_keys=True, default=str) if d is not None else None


def _parse_override_json(text: Any) -> dict[str, Any] | None:
    """A recorded override dict (canonical JSON) back to a dict with numeric values as floats; ``None`` when absent or
    not a JSON object."""
    if not isinstance(text, str):
        return None
    try:
        parsed = json.loads(text)
    except ValueError:
        return None
    if not isinstance(parsed, Mapping):
        return None
    return {str(k): (float(v) if _numeric_value(v) else v) for k, v in parsed.items()}


def override_dicts(leg: Mapping[str, Any]) -> dict[str, dict[str, Any] | None]:
    """Review H2: BOTH records of the server's generation-config default, every key kept — ``engine_warning`` (the
    WARNING's dict, :func:`launch_attestation` ``generation_config_override.server_default``) and ``launch_flag`` (the
    echoed ``--override-generation-config``), each ``None`` when the launch carries none."""
    launch = _mapping(leg.get("launch"))
    return {"engine_warning": _parse_override_json(_mapping(launch.get("generation_config_override")).get("server_default")),
            "launch_flag": _parse_override_json(_mapping(launch.get("settings")).get(LAUNCH_OVERRIDE_ECHO_KEY))}


def override_keys(leg: Mapping[str, Any]) -> dict[str, Any]:
    """The UNION of the keys either override record names, with the value — the WARNING's when both name a key (the
    engine's own statement of what it applied), the flag's otherwise."""
    d = override_dicts(leg)
    out: dict[str, Any] = dict(d["launch_flag"] or {})
    out.update(d["engine_warning"] or {})
    return out


def override_key_problem(legs: Mapping[str, Mapping[str, Any]], labels: Mapping[str, str]) -> str | None:
    """Review H2 — REFUSE-UNLESS-UNDERSTOOD: the first key any labelled leg's override records name outside
    :data:`OVERRIDE_UNDERSTOOD_KEYS`, with its value; ``None`` when every key is one the effect rule understands. A key
    that is not understood cannot be waived by effect, whatever the client sends."""
    for lab, k in labels.items():
        for key, value in sorted(override_keys(legs[k]).items()):
            if key not in OVERRIDE_UNDERSTOOD_KEYS:
                return (f"{lab}: the server's generation-config default names {key!r} = {value!r}, a key the effect rule does not "
                        f"understand (understood: {list(OVERRIDE_UNDERSTOOD_KEYS)}) — refused, not waived")
    return None


def _limit_value(v: Any) -> int | None:
    return int(v) if _numeric_value(v) else None


def generation_config_source(leg: Mapping[str, Any]) -> str:
    """The leg's ``--generation-config`` as the launch echo carries it (:data:`GENERATION_CONFIG_ECHO_KEY`, a canonical JSON
    string among the attested settings), or ``auto`` — vLLM's default, which the non-default-args echo leaves out."""
    raw = _mapping(_mapping(leg.get("launch")).get("settings")).get(GENERATION_CONFIG_ECHO_KEY)
    if raw is None:
        return GENERATION_CONFIG_SOURCE_AUTO
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        parsed = raw
    return str(parsed)


def effective_output_limit(leg: Mapping[str, Any]) -> dict[str, Any]:
    """Review H2 as corrected by the second reader (M-a) and made EXHAUSTIVE by their round 2 (M1): the leg's EFFECTIVE output
    limit with the PROVENANCE of every server-side limit (:data:`OUTPUT_LIMIT_RULE`; the state in :data:`OUTPUT_LIMIT_STATES`).
    ``client_cap`` is the identity's ``max_tokens_by_task`` for the task (the cap the shared projection already holds
    equal; the requests send it). ``generation_config_source`` is the echoed ``--generation-config`` (``auto`` when the echo
    carries none). ``warning_recorded`` / ``flag_recorded`` say which of the two server-default records the launch carries
    at all (a dict, with or without a limit key). ``server_clip_max_tokens`` is a limit the engine applies as a hard
    minimum — the flag's ``max_new_tokens`` under any source, or the WARNING's merged ``max_tokens`` under a custom PATH —
    with ``clip_source`` naming which; ``server_fallback_max_tokens`` is a limit the engine applies only to a request that
    omits its cap — the WARNING's ``max_tokens`` under ``auto`` when the flag names none (the model file's; under ``vllm``
    no file is loaded, so there is no fallback). ``effective`` = min(client cap, clip); the fallback governs only when no
    client cap is recorded (``fallback_governs``); ``clipped`` says whether a clip set the served cap. ``problem`` — on
    which the contrast REFUSES and ``effective`` is ``None``, NOT read — names (i) records that disagree (the flag and the
    WARNING both name a limit and differ: the WARNING prints the merged dict with the flag applied last, so they must
    agree), (ii) a WARNING recorded under ``vllm`` (the engine loads no file and prints no WARNING there — the records
    contradict it; revision 3 read this as a fallback), (iii) a custom PATH with neither a WARNING nor a flag naming the
    limit (the file's limit is unread — UNKNOWN; revision 3 read it as no limit). Fail-closed on a genuine clip path and
    on an unreadable record, never on a fallback."""
    ident = _mapping(leg.get("identity"))
    client_cap = _limit_value(_mapping(ident.get(C.IDENTITY_KEY_MAX_TOKENS_BY_TASK)).get(TASK))
    dicts = override_dicts(leg)
    warning_recorded, flag_recorded = dicts["engine_warning"] is not None, dicts["launch_flag"] is not None
    flag = _limit_value(_mapping(dicts["launch_flag"]).get(OVERRIDE_FLAG_LIMIT_KEY))
    warned = _limit_value(_mapping(dicts["engine_warning"]).get(OVERRIDE_WARNING_LIMIT_KEY))
    source = generation_config_source(leg)
    flag_clip_source = f"--override-generation-config {OVERRIDE_FLAG_LIMIT_KEY} (a clip under every generation-config source)"
    problem: str | None = None
    clip: int | None = None
    clip_source: str | None = None
    fallback: int | None = None
    if flag is not None and warned is not None and flag != warned:
        state = OUTPUT_LIMIT_STATE_DISAGREE
        problem = (f"the launch flag's {OVERRIDE_FLAG_LIMIT_KEY} {flag} and the engine WARNING's {OVERRIDE_WARNING_LIMIT_KEY} {warned} "
                   "disagree (the WARNING prints the merged dict with the flag applied last, so they must agree) — refused, not read")
    elif source == GENERATION_CONFIG_SOURCE_VLLM:
        if warning_recorded:
            state = OUTPUT_LIMIT_STATE_VLLM_CONTRADICTORY
            problem = (f"--generation-config {GENERATION_CONFIG_SOURCE_VLLM} loads no generation config and prints no override WARNING (vLLM "
                       "0.25.1 get_diff_sampling_param: the config starts empty and the WARNING is emitted only when the source is not vllm), "
                       f"yet the launch records the WARNING {json.dumps(dicts['engine_warning'], sort_keys=True)} — the records contradict the "
                       "engine; refused, not read as a fallback")
        elif flag is not None:
            state, clip, clip_source = OUTPUT_LIMIT_STATE_VLLM_FLAG_CLIP, flag, flag_clip_source
        else:
            state = OUTPUT_LIMIT_STATE_VLLM_NO_LIMIT
    elif source == GENERATION_CONFIG_SOURCE_AUTO:
        if flag is not None:
            state, clip, clip_source = OUTPUT_LIMIT_STATE_AUTO_FLAG_CLIP, flag, flag_clip_source
        elif warned is not None:
            state, fallback = OUTPUT_LIMIT_STATE_AUTO_FILE_FALLBACK, warned
        else:
            state = OUTPUT_LIMIT_STATE_AUTO_NO_LIMIT
    else:
        if warned is not None:
            state, clip = OUTPUT_LIMIT_STATE_PATH_MERGED_CLIP, warned
            clip_source = f"--generation-config {source!r}: the merged {OVERRIDE_WARNING_LIMIT_KEY} is a clip under a custom path"
        elif flag is not None:
            state, clip = OUTPUT_LIMIT_STATE_PATH_FLAG_CLIP, flag
            clip_source = (f"--generation-config {source!r}: the flag's {OVERRIDE_FLAG_LIMIT_KEY}, applied last over the file, is the merged "
                           "clip (no override WARNING recorded)")
        elif warning_recorded:
            state = OUTPUT_LIMIT_STATE_PATH_ATTESTED_NO_LIMIT
        else:
            state = OUTPUT_LIMIT_STATE_PATH_UNKNOWN
            problem = (f"--generation-config {source!r} loads a generation config this analysis cannot read, and the launch records neither "
                       f"the override WARNING (the merged dict, the only attestation of the file's contents) nor a flag naming "
                       f"{OVERRIDE_FLAG_LIMIT_KEY} — the served output limit is UNKNOWN; refused, not read as unlimited")
    candidates = [v for v in (client_cap, clip) if v is not None]
    fallback_governs = problem is None and not candidates and fallback is not None
    if fallback_governs:
        candidates = [fallback]
    return {"client_cap": client_cap, "generation_config_source": source, "warning_recorded": warning_recorded, "flag_recorded": flag_recorded,
            "state": state, "server_clip_max_tokens": clip, "clip_source": clip_source, "server_fallback_max_tokens": fallback,
            "fallback_governs": bool(fallback_governs), "effective": None if problem else (min(candidates) if candidates else None),
            "clipped": bool(problem is None and clip is not None and (client_cap is None or clip < client_cap)),
            "problem": problem, "rule": OUTPUT_LIMIT_RULE}


def _output_limit_text(rec: Mapping[str, Any]) -> str:
    """The effective output limit as ONE line (the comparison record's ``effective_output_limit_by_leg``): a leg with no
    server-side limit reads exactly as before (``<cap> (client cap <cap>; server default none)``); a clip and a fallback
    are named with their provenance; a record with a ``problem`` (M1: contradictory, unknown, disagreeing) reads ``UNREADABLE``
    with its state and the problem — never a number that could pass for a limit."""
    if rec.get("problem"):
        return (f"UNREADABLE (client cap {rec['client_cap']}; --generation-config {rec['generation_config_source']}; state {rec['state']}; "
                f"PROBLEM: {rec['problem']})")
    parts = [f"client cap {rec['client_cap']}"]
    if rec["server_clip_max_tokens"] is not None:
        parts.append(f"server clip {rec['server_clip_max_tokens']} from {rec['clip_source']}")
    if rec["server_fallback_max_tokens"] is not None:
        parts.append(f"server fallback {rec['server_fallback_max_tokens']} (generation_config.json under --generation-config "
                     f"{rec['generation_config_source']}: replaced by the request's cap, not a clip)")
    if rec["server_clip_max_tokens"] is None and rec["server_fallback_max_tokens"] is None:
        parts.append("server default none")
    tail = ("; clipped by the server default" if rec["clipped"]
            else ("; the fallback governs (no client cap recorded)" if rec["fallback_governs"] else ""))
    return f"{rec['effective']} ({'; '.join(parts)}{tail})"


def effective_decode(leg: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """The leg's EFFECTIVE decode over :data:`SERVER_DEFAULTABLE_KEYS` — per key, the value the engine resolved on every
    science request and where it came from: ``client`` when the leg's requests send the key (the manifest's numeric sampling
    row, the same block the shared projection compares), else ``server_default`` — the engine's stated override
    (:func:`launch_attestation` ``generation_config_override.server_default``) when present, else the composed
    ``--override-generation-config`` echo (the engine states the same dict when it applies it; the echo alone is the
    composer's record of it), else vLLM's neutral default (the recorder's table, ``RS.VLLM_NEUTRAL_DEFAULTS``). Values are
    floats. Only the understood SAMPLING keys enter here (every other key in an override is refused by name by
    :func:`override_key_problem` before this comparison; the output limit has its own, :func:`effective_output_limit`).
    A leg whose launch is not attested has no server default to read and yields ``None`` for every unsent key (the
    caller refuses on the attestation first)."""
    sent = _numeric_row(_mapping(_mapping(leg.get("identity")).get("sampling")))
    launch = _mapping(leg.get("launch"))
    attested = launch.get("status") == LAUNCH_ATTESTED
    dicts = override_dicts(leg)
    stated = dicts["engine_warning"] if dicts["engine_warning"] is not None else dicts["launch_flag"]
    default: dict[str, float] = dict(RS.VLLM_NEUTRAL_DEFAULTS)
    for k, v in (stated or {}).items():
        if k in SERVER_DEFAULTABLE_KEYS and _numeric_value(v):
            default[k] = float(v)
    out: dict[str, dict[str, Any]] = {}
    for key in SERVER_DEFAULTABLE_KEYS:
        if key in sent:
            out[key] = {"value": float(sent[key]), "source": EFFECTIVE_SOURCE_CLIENT}
        else:
            out[key] = {"value": (float(default[key]) if attested and key in default else None), "source": EFFECTIVE_SOURCE_SERVER}
    return out


def effective_decode_problem(legs: Mapping[str, Mapping[str, Any]], labels: Mapping[str, str], *, same_sampling: bool = True) -> str | None:
    """Why the legs' effective decodes are not one decode. In order: (1) REFUSE-UNLESS-UNDERSTOOD — a key
    in any leg's override records outside :data:`OVERRIDE_UNDERSTOOD_KEYS` (:func:`override_key_problem`); (2) the first
    sampling key (in :data:`SERVER_DEFAULTABLE_KEYS` order) whose effective VALUE differs across the labelled legs, named
    with every leg's value and source — under ``same_sampling`` (a contrast's arms share one sampling row, which the
    shared projection has already held equal) every key is compared, so a server default that reaches any leg's requests
    must be one value across the contrast; under ``same_sampling`` False (a mode delta: the two rows' SENT values — and
    which keys they send — differ by design) only the keys NO leg sends are compared, default against default; (3) the
    effective OUTPUT LIMIT (:func:`effective_output_limit`), compared across every leg whatever the sampling. ``None`` =
    compatible."""
    why = override_key_problem(legs, labels)
    if why:
        return why
    eff = {lab: effective_decode(legs[k]) for lab, k in labels.items()}
    for key in SERVER_DEFAULTABLE_KEYS:
        per = {lab: eff[lab][key] for lab in labels}
        if not same_sampling and any(rec["source"] == EFFECTIVE_SOURCE_CLIENT for rec in per.values()):
            continue
        values = {json.dumps(rec["value"]) for rec in per.values()}
        if len(values) > 1:
            return f"{key}: " + json.dumps({lab: f"{rec['value']} ({rec['source']})" for lab, rec in per.items()}, sort_keys=True)
    limits = {lab: effective_output_limit(legs[k]) for lab, k in labels.items()}
    disagreeing = {lab: rec["problem"] for lab, rec in limits.items() if rec.get("problem")}
    if disagreeing:
        return ("max_tokens (the output-limit record is not readable — records that disagree, a WARNING recorded under --generation-config vllm, "
                "or a custom generation-config path with no attested limit — refused, not read): " + json.dumps(disagreeing, sort_keys=True))
    if len({json.dumps(rec["effective"]) for rec in limits.values()}) > 1:
        return ("max_tokens (the effective output limit — the client cap clipped by a server CLIP: the flag's max_new_tokens, or the merged "
                "max_tokens under a custom generation-config path; a file fallback does not reach a request that sends its cap): "
                + json.dumps({lab: _output_limit_text(rec) for lab, rec in limits.items()}, sort_keys=True))
    return None


def server_projection(leg: Mapping[str, Any], *, override: bool = True) -> dict[str, Any]:
    """B5 (i″): the NORMALISED behaviour-affecting server settings compared across arms and rung↔source — the RECORDED ones
    (engine, the requested and resolved model and revision, the served AND the manifest's reasoning parser, the reduction
    mode, the modality flags, the render regime's header ids) and, under ``launch.*``, every echoed launch flag the leg's
    attestation carries (:func:`launch_attestation`: dtype, quantisation, tokenizer revision, default chat-template kwargs,
    the generation-config override, prefix caching, …) plus the engine's generation-config-override reading — a difference
    in any is a named refusal (code review C2: the projection used to compare five fields, so a changed precision or server
    default read as equal). The geometry (`max_model_len`, `max_num_seqs`, `concurrency`) and `--enable-log-requests` are
    excluded by name (:data:`LAUNCH_EXCLUDED_KEYS`). An UNATTESTED leg projects ``launch.status: absent`` and no launch
    key — :func:`compatibility_problem` refuses it under a declaration before any equality is read. ``override`` False
 leaves the two server-default records (:data:`LAUNCH_OVERRIDE_ECHO_KEY`, :data:`LAUNCH_OVERRIDE_READING_KEY`)
    out — the MATH-500 comparison compares those by effect (:func:`effective_decode_problem`) and reports the literal
    reading beside; the default projection is the literal one (the GSM8K port reads it as before)."""
    ident = _mapping(leg.get("identity"))
    served = _mapping(ident.get("served"))
    out: dict[str, Any] = {"engine": ident.get("engine"), "requested_model": ident.get("requested_model"),
                           "requested_revision": ident.get("requested_revision"), "resolved_model_id": ident.get("resolved_model_id"),
                           "resolved_revision": ident.get("resolved_revision"), "reasoning_parser": served.get("reasoning_parser"),
                           "reasoning_parser_thinking": ident.get("reasoning_parser_thinking"),
                           C.REDUCTION_MODE_FIELD: served.get(C.REDUCTION_MODE_FIELD), "tool_calls": ident.get("tool_calls"),
                           "agent_loop_entry": ident.get("agent_loop_entry"), "thinking_header_ids": ident.get("thinking_header_ids_text")}
    launch = _mapping(leg.get("launch"))
    out["launch.status"] = launch.get("status")
    if launch.get("status") == LAUNCH_ATTESTED:
        for k, v in sorted(_mapping(launch.get("settings")).items()):
            if not override and k == LAUNCH_OVERRIDE_ECHO_KEY:
                continue
            out[f"launch.{k}"] = v
        if override:
            gco = _mapping(launch.get("generation_config_override"))
            out[LAUNCH_OVERRIDE_READING_KEY] = json.dumps({"present": gco.get("present"), "params": gco.get("params")}, sort_keys=True)
    return out


def launch_unattested(legs: Mapping[str, Mapping[str, Any]], labels: Mapping[str, str]) -> str | None:
    """Why the launch settings of a set of legs cannot be compared: the first leg whose attestation is not ``attested``,
    with its problems — or ``None`` when every leg is attested."""
    for lab, k in labels.items():
        launch = _mapping(legs[k].get("launch"))
        if launch.get("status") != LAUNCH_ATTESTED:
            return f"{lab}: no launch attestation ({'; '.join(launch.get('problems') or ['not read'])})"
    return None


def _first_difference(projections: Mapping[str, Mapping[str, Any]]) -> str | None:
    """The first field on which the labelled projections disagree (``"<field>: {label: value, ...}"``), else ``None`` — over
    the UNION of every projection's keys in first-seen order (a key one leg carries and another lacks IS a difference; the
    first leg's keys alone used to be walked, code review C2)."""
    labels = list(projections)
    fields = list(dict.fromkeys(f for lab in labels for f in projections[lab]))
    for f in fields:
        values = {lab: projections[lab].get(f) for lab in labels}
        first = values[labels[0]]
        if any(v != first for v in values.values()):
            return f"{f}: {values}"
    return None


def _all_differences(projections: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """EVERY field on which the labelled projections disagree (``"<field>: {label: value, ...}"`` each), over the union of
    keys in first-seen order — :func:`_first_difference`'s walk, not stopped at the first."""
    labels = list(projections)
    fields = list(dict.fromkeys(f for lab in labels for f in projections[lab]))
    out: list[str] = []
    for f in fields:
        values = {lab: projections[lab].get(f) for lab in labels}
        first = values[labels[0]]
        if any(v != first for v in values.values()):
            out.append(f"{f}: {values}")
    return out


SERVER_SETTINGS_DIFFER = "the legs' normalised server settings differ"
#: a contrast whose two legs attest DIFFERENT engines (`launch.engine`: lens vs lora_native) is NOT admissible, whatever
#: its flags read — refused by name BEFORE any flag equality could pass, the engine pair named, every differing flag
#: listed beside it
CROSS_ENGINE_WHY = ("the native-vs-lens numerics difference has never been measured (every native certificate reads numerics=not_measured; kernels "
                    "differ: LoRA path on every request incl. the zero adapter, CUDA-graph FULL_AND_PIECEWISE vs eager, capture-size padding), so a "
                    "greedy item flip is indistinguishable from the cell's effect; remedy: an anchor served on the same engine (the native standup "
                    "serves the path-matched zero adapter none__lora0)")
CROSS_ENGINE_EQUAL_FLAGS = "none — equal flags never make a cross-engine contrast admissible"


def launch_engines_of(legs: Mapping[str, Mapping[str, Any]], labels: Mapping[str, str]) -> dict[str, str | None]:
    """``{label: launch.engine}`` — the engine each leg's launch record attested (None when unattested / unread)."""
    return {lab: _mapping(legs[k].get("launch")).get("engine") for lab, k in labels.items()}


def cross_engine(legs: Mapping[str, Mapping[str, Any]], labels: Mapping[str, str]) -> bool:
    """Whether the legs attest more than one engine (O106's refusal condition)."""
    return len(set(launch_engines_of(legs, labels).values())) > 1


#: the ONE echoed launch flag whose value is a FUNCTION of the served width S on the native engine: the composer hands vLLM
#: `vllm_server.cudagraph_capture_sizes_for(max_num_seqs)` — vLLM's default capture sizes for S plus S itself (the
#: 2026-09-16 S=51 incident) — so two kernel-ceiling lowerings to different S (the pulled qwen35_4b `arditi_allpos` GSM8K
#: pair: 48 at the 8,192 base, 27 at the 16,384 rung) echo different lists BY CONSTRUCTION, with no setting changed. The
#: same-engine comparison read the list as a setting and refused the rung/base pairing (A7). The width itself is held per
#: row by the geometry contract and reconciled with the echo by the launch attestation (LAUNCH_EXCLUDED_KEYS)
LAUNCH_CAPTURE_SIZES_KEY = "cudagraph_capture_sizes"
CAPTURE_SIZES_RULE = ("vllm_server.cudagraph_capture_sizes_for(S): vLLM's default capture sizes for S — 1, 2, 4, then the multiples of 8 "
                      "below 256 and of 16 from 256, every one <= min(2S, 512) — plus S itself, sorted; the exact ints the native "
                      "composer hands --cudagraph-capture-sizes")


def attested_launch_seqs(leg: Mapping[str, Any]) -> int | None:
    """The width S a leg's launch ATTESTED: the echo's ``max_num_seqs`` (recorded under ``launch.excluded`` — a geometry key,
    never compared across arms) on a launch whose attestation is ``attested`` and whose echo geometry was RECONCILED with
    the manifest's served block (:func:`launch_attestation`: the same S the row's geometry contract validated). ``None``
    when the launch is not attested, not reconciled, or echoes no ``max_num_seqs`` — never a default."""
    launch = _mapping(leg.get("launch"))
    if launch.get("status") != LAUNCH_ATTESTED or launch.get("geometry_reconciled") is not True:
        return None
    rec = _mapping(_mapping(launch.get("excluded")).get("max_num_seqs"))
    value = rec.get("value")
    if not isinstance(value, str):
        return None
    try:
        s = json.loads(value)
    except ValueError:
        return None
    s = _int_field(s)
    return s if s is not None and s > 0 else None


def capture_sizes_by_derivation(legs_by_label: Mapping[str, Mapping[str, Any]],
                                projections: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, dict[str, Any]], str | None]:
    """The carve for :data:`LAUNCH_CAPTURE_SIZES_KEY` on a SAME-engine native comparison: when ``launch.cudagraph_capture_sizes``
    differs across the labelled projections, the difference is admitted — the key DROPPED from every projection — ONLY IF every
    leg attests the native engine and each leg's list is EXACTLY the composer's derivation for ITS OWN attested S
    (:data:`CAPTURE_SIZES_RULE`, through the composer's own function — derived, never trusted). Otherwise the key stays a
    difference for :func:`_first_difference` to name as before, and the returned note says why it was not admitted: a leg
    whose list is not its S's derivation (a real setting change), a leg whose S is not attested, or a non-native engine (which
    composes no capture list). Legs attesting DIFFERENT engines are left untouched — O106 (:func:`server_settings_difference`)
    and the recorded ``engine`` field refuse those before any flag is read. Returns ``(projections, note)``; with no difference
    on the key the projections come back as they were and the note is ``None``."""
    key = f"launch.{LAUNCH_CAPTURE_SIZES_KEY}"
    projections = {lab: dict(proj) for lab, proj in projections.items()}
    values = {lab: proj.get(key) for lab, proj in projections.items()}
    if len({json.dumps(v, sort_keys=True) for v in values.values()}) <= 1:
        return projections, None
    engines = {lab: _mapping(leg.get("launch")).get("engine") for lab, leg in legs_by_label.items()}
    if len(set(engines.values())) > 1:
        return projections, None
    engine = next(iter(engines.values()), None)
    if engine != C.SERVING_ENGINE_LORA_NATIVE:
        return projections, (f"{key} differs on the {engine!r} engine, which composes no capture list — a real setting change, not a "
                             "width (the derivation is the native composer's)")
    from sisteer.capeval.serving import vllm_server as VS   # the composer's own rule; torch-free

    for lab, leg in legs_by_label.items():
        S = attested_launch_seqs(leg)
        if S is None:
            return projections, (f"{key}: {lab}'s attested max_num_seqs cannot be read (no attested, geometry-reconciled launch echoing "
                                 "max_num_seqs), so its list cannot be derived — the difference stands")
        want = json.dumps(list(VS.cudagraph_capture_sizes_for(S)), sort_keys=True, default=str)
        if values.get(lab) != want:
            return projections, (f"{key}: {lab}'s list {values.get(lab)} is not the composer's derivation for its attested max_num_seqs "
                                 f"{S} ({want}; {CAPTURE_SIZES_RULE}) — a real setting change, not the width")
    return {lab: {f: v for f, v in proj.items() if f != key} for lab, proj in projections.items()}, None


def server_settings_difference(legs: Mapping[str, Mapping[str, Any]], labels: Mapping[str, str]) -> str | None:
    """Why the legs' launch settings refuse the contrast, or ``None``. Same-engine legs: the FIRST field on which the
    NORMALISED server settings (:func:`server_projection` with the server default left out) differ — the message shape
    the record has always carried. Legs attested through DIFFERENT engines (a lens anchor beside a lora_native leg): O106
    (:data:`CROSS_ENGINE_WHY`) — not admissible whatever the flags read, refused BEFORE any equality could pass, the ENGINE
    PAIR named, the ruling cited, and EVERY differing flag listed with both values (:func:`_all_differences`; the lens launch
    echoes ``enforce_eager: true``, the native launch — graphs on — omits it and echoes ``enable_lora`` and the LoRA kernel
    configuration), so the record says exactly what differed. Called only after every leg's launch is attested
    (:func:`launch_unattested`)."""
    server = {lab: server_projection(legs[k], override=False) for lab, k in labels.items()}
    engines = launch_engines_of(legs, labels)
    if len(set(engines.values())) > 1:
        diffs = _all_differences(server)
        pair = " vs ".join(f"{lab} ({engine})" for lab, engine in engines.items())
        return (f"cross-engine contrast: {pair} is not admissible — {CROSS_ENGINE_WHY} — {CROSS_ENGINE_RULING}; differing launch flags "
                f"({len(diffs)}): {'; '.join(diffs) if diffs else CROSS_ENGINE_EQUAL_FLAGS}")
    # the capture-size carve: a `cudagraph_capture_sizes` difference between same-engine native legs is admitted iff each
    # side's list is the composer's derivation for its own attested S (capture_sizes_by_derivation); every other field is
    # walked exactly as before, and a capture-size difference that stands carries the reason it was not admitted
    server, capture_note = capture_sizes_by_derivation({lab: legs[k] for lab, k in labels.items()}, server)
    first = _first_difference(server)
    if first is None:
        return None
    if capture_note is not None and first.startswith(f"launch.{LAUNCH_CAPTURE_SIZES_KEY}: "):
        return f"{SERVER_SETTINGS_DIFFER} — {first}; {capture_note}"
    return f"{SERVER_SETTINGS_DIFFER} — {first}"


def compatibility_problem(legs: Mapping[str, Mapping[str, Any]], labels: Mapping[str, str], *, same_sampling: bool = True) -> str | None:
    """B5 / plan P5: why a set of usable legs across ADMITTED producer trees has no joint estimand — (i) the shared-field
    projection differs (the field named), (i″) a normalised server setting differs, or (iii) a leg's tree is not admitted
    (its producer record did not match its root's pin). (i′) — each row's OWN contract — was validated per leg at
    collection (:func:`arm_contract_problem` inside :func:`identity_problem`)."""
    shared = {lab: shared_projection(legs[k], sampling=same_sampling) for lab, k in labels.items()}
    why = _first_difference(shared)
    if why:
        return f"the legs' shared-field projection differs — {why}"
    # code review C2: the launch settings are compared from each leg's ATTESTATION; a leg without one cannot be compared and
    # refuses by name (the record never reads "equal" over settings nobody attested)
    why = launch_unattested(legs, labels)
    if why:
        return f"the legs' launch settings cannot be compared — {why}"
    # every launch flag but the server's default sampling is compared literally; the server default by EFFECT. Legs
    # attested through different engines name the pair and every differing key (server_settings_difference)
    why = server_settings_difference(legs, labels)
    if why:
        return why
    why = effective_decode_problem(legs, labels, same_sampling=same_sampling)
    if why:
        return f"the legs' effective decode differs — {why}"
    for lab, k in labels.items():
        if not _mapping(legs[k].get("producer")).get("matches_pin"):
            return f"{lab}: its producer tree is not an admitted tree"
    return None


def server_settings_verdict(legs: Mapping[str, Mapping[str, Any]], labels: Mapping[str, str], *,
                            same_sampling: bool = True) -> dict[str, Any]:
    """The record's server-settings facts (strings / bools / None — B1): ``recorded_settings_equal`` over the fields the
    manifests carry, ``launch_attested`` per leg, ``server_settings_equal`` — True / False only when EVERY leg's launch is
    attested (the projection outside the server default read equal / not AND the effective decode equal / not),
    ``None`` otherwise (never a claim over settings that were not read); and the LITERAL reading kept
    beside it: ``server_settings_literal_equal`` (the full projection, the two server-default records included),
    ``server_settings_literal_difference`` (the first literal field that differs, with every leg's value, else None),
    ``effective_decode_difference`` (likewise for the effective decode), ``effective_decode_by_leg`` and
    ``generation_config_override_by_leg`` (the engine's stated server default per leg, canonical JSON or None)."""
    attested = {lab: _mapping(legs[k].get("launch")).get("status") == LAUNCH_ATTESTED for lab, k in labels.items()}
    recorded = {lab: {f: v for f, v in server_projection(legs[k]).items() if not f.startswith("launch.")} for lab, k in labels.items()}
    recorded_equal = _first_difference(recorded) is None
    all_attested = all(attested.values())
    literal_why = _first_difference({lab: server_projection(legs[k]) for lab, k in labels.items()}) if all_attested else None
    # O106: legs attesting different engines are never equal, whatever their flags read (server_settings_difference)
    compared_why = server_settings_difference(legs, labels) if all_attested else None
    effective_why = effective_decode_problem(legs, labels, same_sampling=same_sampling) if all_attested else None
    engines = launch_engines_of(legs, labels)
    is_cross_engine = cross_engine(legs, labels) if all_attested else None
    return {"recorded_settings_equal": recorded_equal, "launch_attested": attested,
            "server_settings_equal": ((compared_why is None and effective_why is None) if all_attested else None),
            "engine_by_leg": engines, "cross_engine": is_cross_engine,
            "cross_engine_refusal": (compared_why if is_cross_engine else None),
            "server_settings_literal_equal": ((literal_why is None) if all_attested else None),
            "server_settings_literal_difference": literal_why,
            "effective_decode_difference": effective_why,
            "effective_decode_by_leg": {lab: {key: f"{rec['value']} ({rec['source']})" for key, rec in effective_decode(legs[k]).items()}
                                        for lab, k in labels.items()},
            "generation_config_override_by_leg": {lab: _mapping(_mapping(legs[k].get("launch")).get("generation_config_override")).get("server_default")
                                                  for lab, k in labels.items()},
            # review H2: every key either override record names, per leg; the refuse-unless-understood verdict; the
            # effective output limit per leg (strings — B1)
            "override_keys_by_leg": {lab: sorted(override_keys(legs[k])) for lab, k in labels.items()},
            "override_key_problem": override_key_problem(legs, labels) if all_attested else None,
            "effective_output_limit_by_leg": {lab: _output_limit_text(effective_output_limit(legs[k])) for lab, k in labels.items()},
            "output_limit_rule": OUTPUT_LIMIT_RULE,
            "server_settings_rule": ("every echoed launch flag outside the named exclusions compared literally, EXCEPT the server's "
                                     "default sampling (the --override-generation-config echo and the engine's override WARNING), "
                                     "whose keys are REFUSED UNLESS UNDERSTOOD (review H2): the five sampling keys temperature / top_p / "
                                     "top_k / min_p / repetition_penalty are compared by EFFECT over the keys the requests do not send "
                                     "(D48), the output limit max_new_tokens / max_tokens by PROVENANCE (second reader M-a: a server "
                                     "CLIP — the flag's max_new_tokens, or the merged max_tokens under a custom generation-config path — "
                                     "as the effective cap min(client cap, clip); a model-file FALLBACK under auto only — vllm loads no "
                                     "model file — is reported "
                                     "beside the cap the requests send, never a clip), and any other key in either side's override "
                                     "refuses the contrast by name; the literal reading is reported beside, never a refusal by itself; "
                                     f"legs attesting DIFFERENT engines (lens vs lora_native) are not admissible whatever their flags read "
                                     f"({CROSS_ENGINE_RULING}: the native-vs-lens numerics difference has never been measured) — refused by name "
                                     "with every differing flag listed; on a same-engine contrast the native engine's cell-set plumbing "
                                     "(lora_modules, max_loras, max_cpu_loras) is excluded and the LoRA kernel configuration "
                                     "(enable_lora, max_lora_rank, lora_target_modules) compared; a cudagraph_capture_sizes difference "
                                     "between same-engine native legs is admitted only when each side's list is the composer's derivation "
                                     f"for its own attested max_num_seqs ({CAPTURE_SIZES_RULE}), else refused naming the list that is not"),
            "server_settings_note": (None if all_attested else
                                     f"launch settings unattested on {[lab for lab, a in attested.items() if not a]} — {LAUNCH_REMEDY}")}


#: the client-lock DISCLOSURE: a comparison's legs may have run under different client locks (the published 24 and
#: nine more at 0f21fd4d…, the steering legs at 910b4851… after the IFEval checker's dependencies joined the lock).
#: The digest alone cannot say whether that is one instrument or two, so the record carries the DERIVED reading
#: (`sisteer.capeval.tinycap.client_lock`): each leg's recorded digest, the lock TEXT of each leg's producer tree
#: (`git show`, its digest held to the recorded one — chain of custody), and for every pair of distinct locks the
#: imported-package comparison — `one_instrument` iff no distribution the MATH-500 client imports changed version.
#: A DISCLOSURE: it names the fork, it never refuses a comparison (the shared projection never carried the digest);
#: `undetermined` names a tree this clone cannot read or a custody mismatch, never read as equal.
CLIENT_LOCK_DISCLOSURE_RULE = ("per comparison: the client lock digest each leg recorded (manifest.frameworks.client_lock_sha256), the "
                               "lock text of each leg's producer tree (git show, digest == recorded), and for each pair of distinct "
                               "digests the imported-package comparison of sisteer.capeval.tinycap.client_lock — one_instrument iff "
                               "no distribution the MATH-500 client imports changed version; disclosed, never a refusal")
_CLIENT_LOCK_TEXT_CACHE: dict[str, tuple[str | None, str | None]] = {}
_CLIENT_LOCK_WALK: dict[str, Any] = {}


def _client_lock_text(tree_sha: str, repo_root: Path) -> tuple[str | None, str | None]:
    """`git show <tree>:<lock>` from `repo_root`, cached per tree — the ONE indirection the tests replace."""
    key = f"{repo_root}:{tree_sha}"
    if key not in _CLIENT_LOCK_TEXT_CACHE:
        _CLIENT_LOCK_TEXT_CACHE[key] = CL.lock_text_at_tree(repo_root, tree_sha)
    return _CLIENT_LOCK_TEXT_CACHE[key]


def _client_lock_walk(repo_root: Path):
    """The client's import walk over `repo_root` (the analysis checkout — the client's entry points and import roots are
    the same on every tree of this campaign), cached per root."""
    key = str(repo_root)
    if key not in _CLIENT_LOCK_WALK:
        _CLIENT_LOCK_WALK[key] = CL.walk_client_imports(repo_root)
    return _CLIENT_LOCK_WALK[key]


def client_lock_disclosure(legs: Mapping[str, Mapping[str, Any]], labels: Mapping[str, str], *,
                           repo_root: Path | None = None) -> dict[str, Any]:
    """`compatibility.client_lock` (see :data:`CLIENT_LOCK_DISCLOSURE_RULE`). Strings, bools, None, lists and dicts only."""
    repo_root = Path(repo_root) if repo_root is not None else REPO_ROOT
    by_leg = {lab: {"sha256": _mapping(legs[k].get("identity")).get("client_lock_sha256"),
                    "tree": _mapping(legs[k].get("producer")).get("tree_sha")} for lab, k in labels.items()}
    # the distinct digests in ORDER OF FIRST APPEARANCE over the labels (the bare leg first): a pair reads
    # `published` = the earlier leg's lock, `wave` = the later's — never the digests' lexical order
    shas = list(dict.fromkeys(v["sha256"] for v in by_leg.values() if v["sha256"]))
    undetermined: list[str] = []
    texts: dict[str, tuple[str, str]] = {}          # digest -> (tree, text) for the digests whose tree lock reads back equal
    for lab, v in by_leg.items():
        if not v["sha256"]:
            undetermined.append(f"{lab}: the leg records no client_lock_sha256 (a store that predates the field)")
            continue
        if not v["tree"]:
            undetermined.append(f"{lab}: the leg records no producer tree, so its lock text cannot be read")
            continue
        if v["sha256"] in texts:
            continue
        text, why = _client_lock_text(v["tree"], repo_root)
        if text is None:
            undetermined.append(f"{lab}: the producer tree {v['tree'][:12]}'s lock cannot be read here ({why})")
            continue
        if CL.lock_sha256(text) != v["sha256"]:
            undetermined.append(f"{lab}: the leg recorded client_lock_sha256 {v['sha256'][:12]}… but its producer tree {v['tree'][:12]}'s lock "
                                f"digests {CL.lock_sha256(text)[:12]}… — the leg did not run under its tree's lock (chain of custody)")
            continue
        texts[v["sha256"]] = (v["tree"], text)
    pairs: dict[str, dict[str, Any]] = {}
    one_instrument: bool | None = True if len(shas) <= 1 and not undetermined else None
    if len(shas) > 1:
        walk = _client_lock_walk(repo_root)
        verdicts: list[bool] = []
        for i, a in enumerate(shas):
            for b in shas[i + 1:]:
                key = f"{a[:12]}|{b[:12]}"
                if a not in texts or b not in texts:
                    pairs[key] = {"one_instrument": None, "why": "a side's lock text is unreadable or fails custody (see undetermined)"}
                    continue
                rec, problems = CL.compare_locks(texts[a][1], texts[b][1], walk)
                pairs[key] = {"one_instrument": rec["one_instrument"], "changed": rec["changed"], "n_compared": rec["n_compared"],
                              "imported_added": sorted(rec["imported_added"]), "imported_removed": sorted(rec["imported_removed"]),
                              "outside_imported_set": rec["outside_imported_set"], "imports_not_in_lock": sorted(rec["imports_not_in_lock"]),
                              "problems": problems, "disclosure": rec["disclosure"]}
                verdicts.append(bool(rec["one_instrument"]))
        if verdicts and len(verdicts) == len(pairs):
            one_instrument = all(verdicts)
    return {"rule": CLIENT_LOCK_DISCLOSURE_RULE, "by_leg": by_leg, "distinct_sha256": shas, "sha256_equal": len(shas) <= 1 and not undetermined,
            "one_instrument": one_instrument, "pairs": pairs, "undetermined": undetermined,
            "why": ("every leg records the same client lock" if len(shas) <= 1 and not undetermined
                    else ("the lock digest differs across the legs but no distribution the MATH-500 client imports changed version — one instrument"
                          if one_instrument else
                          ("a distribution the MATH-500 client imports changed version across the legs — two instruments (named in pairs)"
                           if one_instrument is False else "undetermined: a leg's lock cannot be read or fails custody (see undetermined)")))}


def compatibility_record(legs: Mapping[str, Mapping[str, Any]], labels: Mapping[str, str], *, declared: bool) -> dict[str, Any]:
    """The per-comparison record (plan P5): each leg's tree and root, whether the projection was equal and every arm's
    contract held, the server-settings verdict (:func:`server_settings_verdict`), the client-lock DISCLOSURE
    (:func:`client_lock_disclosure`) and which rule admitted the comparison. Strings, bools and None only (B1)."""
    per = {lab: {"tree": _mapping(legs[k].get("producer")).get("tree_sha"), "root": _mapping(legs[k].get("producer")).get("root"),
                 "pin": _mapping(legs[k].get("producer")).get("pin")} for lab, k in labels.items()}
    trees = {v["tree"] for v in per.values()}
    return {"legs": per, "cross_tree": len(trees) > 1, "declared": bool(declared),
            "projection_equal": _first_difference({lab: shared_projection(legs[k]) for lab, k in labels.items()}) is None,
            **server_settings_verdict(legs, labels),
            "arm_contracts_valid": all(legs[k].get("identity_validated") for k in labels.values()),
            "client_lock": client_lock_disclosure(legs, labels),
            "rule": ("one producer tree per comparison (no declared tree set)" if not declared
                     else "shared-field projection + per-row arm contract + attested launch + normalised server settings + producer membership")}


def verdict_for(est: Mapping[str, Mapping[str, Any]], contrast: str, *, mcnemar: Mapping[str, Any] | None = None,
                file_word: bool = True) -> dict[str, Any]:
    """The verdict rule for ONE contrast of a comparison: a directional
    word iff (a) the PRIMARY branch's bootstrap CI excludes zero, (b) the
    SCORE-AS-IS branch's CI excludes zero on the SAME side, (c) both branches'
    exact point signs agree (zero its own sign), and (d) the primary branch's
    censoring-sensitivity bound excludes zero on that side; else
    :data:`WORD_NOT_RESOLVED` with the failed conditions named by letter. A
    censoring bound is a sample bound, not a CI — a one-item edge passes (d)
    alone, which is why (a)/(b) carry the sampling uncertainty. The statement
    carries both branches' three labelled quantities, the exact McNemar
    sensitivity (both branches) with its mask-dependent one-sided note, and the
    registered bootstrap's replicate signs. ``file_word=False`` (a sensitivity
    comparison) computes the conditions but files no word."""
    p, q = est[BRANCH_PRIMARY]["stats"][contrast], est[BRANCH_SCORE_AS_IS]["stats"][contrast]
    side = int(p["point_sign"])
    a = bool(p["excludes_zero"]) and _interval_side(p["ci_low"], p["ci_high"]) == side and side != 0
    b = bool(q["excludes_zero"]) and _interval_side(q["ci_low"], q["ci_high"]) == side and side != 0
    c = int(q["point_sign"]) == side and side != 0
    d = bool(p["sign_identified"]) and _interval_side(p["bound_low"], p["bound_high"]) == side and side != 0
    conditions = {"a_primary_ci_excludes_zero": a, "b_score_as_is_ci_excludes_zero_same_side": b,
                  "c_points_agree_in_sign": c, "d_bound_excludes_zero_same_side": d}
    failed = []
    if not a:
        failed.append("(a) the primary-branch CI covers zero" if not p["excludes_zero"] else "(a) the primary-branch CI is not on the point's side")
    if not b:
        failed.append("(b) the score-as-is CI covers zero" if not q["excludes_zero"] else "(b) the score-as-is CI is not on the primary side")
    if not c:
        failed.append(f"(c) the branches' point signs differ ({p['point_sign']} / {q['point_sign']})")
    if not d:
        failed.append("(d) the censoring-sensitivity bound covers zero" if not p["sign_identified"]
                      else "(d) the censoring bound is not on the primary side")
    if failed:
        word, reason = WORD_NOT_RESOLVED, "; ".join(failed)
    else:
        word, reason = (WORD_HIGHER if side > 0 else WORD_LOWER), None
    level = float(est[BRANCH_PRIMARY].get("level", pr.DEFAULT_CI))
    parts = [f"{contrast}: {word}",
             f"{BRANCH_NAMES[BRANCH_PRIMARY]}: {T.contrast_text(contrast, p, level=level)}",
             f"{BRANCH_NAMES[BRANCH_SCORE_AS_IS]}: {T.contrast_text(contrast, q, level=level)}"]
    if mcnemar:
        parts.append(f"exact two-sided McNemar: {T._mcnemar_text(mcnemar)}")
        note = T.first_one_sided_note(mcnemar)
        if note:
            parts.append(note)
    parts.append(T.replicate_signs_text(est, contrast))
    if reason:
        parts.append(f"not resolved because {reason}")
    out = {"word": word, "reason": reason, "side": side, "conditions": conditions,
           "point_by_branch": {b_: est[b_]["stats"][contrast]["point"] for b_ in BRANCHES},
           "ci_by_branch": {b_: [est[b_]["stats"][contrast]["ci_low"], est[b_]["stats"][contrast]["ci_high"]] for b_ in BRANCHES},
           "bound": [p["bound_low"], p["bound_high"]], "statement": "; ".join(parts)}
    if not file_word:
        out["word"] = None
        out["reason"] = "sensitivity comparison: no verdict word is filed (never pooled with the primary)"
    return out


#: the refusal CODES a comparison's refusal may DERIVE to — today ONE: the O106 cross-engine ruling. The code is never read off a
#: stamp: :func:`comparison_refusal_code` returns :data:`CROSS_ENGINE_RULING` iff the compatibility record says cross-engine AND
#: the comparison's reason IS that refusal (the first problem the comparability rule named), else None — a refusal with no
#: named code, which no declaration can match
COMPARISON_REFUSAL_CODES: tuple[str, ...] = (CROSS_ENGINE_RULING,)
#: the matrix row field a design uses to DECLARE that a row's comparison is EXPECTED to refuse with a code (the GSM8K design's
#: `expected_refusal`, mirrored on the MATH-500 side: scripts/sdfbase/math500/cells.py row 53 declares CROSS_ENGINE_RULING —
#: the ruling in force, 2026-09-17: no native `none` anchor is run, the arm's numbers are reported as the arm's numbers and the
#: pairing's absence accepted). Read off the DESIGN (load_cells passes the key through); the reader has no CLI for it — one
#: source of truth, the row, is the smallest surface that serves the case
ROW_EXPECTED_REFUSAL_KEY = "expected_refusal"
#: the comparison record's key for a declaration the reader cannot honour (a partial or mixed one; a code it does not derive)
EXPECTED_REFUSAL_PROBLEM_KEY = "expected_refusal_problem"
#: the comparison record's derived code, and the flag that its refusal matched its declaration EXACTLY — both present only when
#: they apply (an undeclared, computing comparison's record is byte-identical to before)
REFUSAL_CODE_KEY, REFUSED_BY_DESIGN_KEY = "refusal_code", "refused_by_design"


def expected_refusal_of_group(rows_by_kind: Mapping[str, Mapping[str, Any]]) -> tuple[str | None, str | None]:
    """``(code, problem)`` — the expected-refusal code a comparison's ARM rows declare under :data:`ROW_EXPECTED_REFUSAL_KEY`:
    ONE code, declared by EVERY non-bare arm of the group, or ``None`` when none declares. A comparison refuses AS ONE (its
    compatibility covers every leg), so a declaration on some arms but not others — or two codes — would hide an undeclared
    arm's contrast behind a declared refusal: that is a PROBLEM the population gate fails by name, never a partial admission.
    A code the reader does not derive (:data:`COMPARISON_REFUSAL_CODES`) is a problem too."""
    declared = {kind: r.get(ROW_EXPECTED_REFUSAL_KEY) for kind, r in rows_by_kind.items() if kind != KIND_BARE}
    codes = {kind: str(c) for kind, c in declared.items() if c}
    if not codes:
        return None, None
    if set(codes) != set(declared) or len(set(codes.values())) != 1:
        return None, (f"{ROW_EXPECTED_REFUSAL_KEY} declared on {sorted(codes)} as {sorted(set(codes.values()))} while the comparison's "
                      f"arms are {sorted(declared)} — a comparison refuses as one, so every arm must declare the same code")
    code = next(iter(codes.values()))
    if code not in COMPARISON_REFUSAL_CODES:
        return None, f"{ROW_EXPECTED_REFUSAL_KEY} {code!r} is not a refusal code this reader derives ({list(COMPARISON_REFUSAL_CODES)})"
    return code, None


def comparison_refusal_code(comp: Mapping[str, Any]) -> str | None:
    """The CODE a refused comparison's refusal derives to, or ``None``: :data:`CROSS_ENGINE_RULING` iff the record's
    compatibility says ``cross_engine`` and the comparison's ``reason`` IS the cross-engine refusal
    (:func:`server_settings_difference`'s text, filed as ``compatibility.cross_engine_refusal``) — derived from the record's
    own facts, never from a stamp or a substring. A usable comparison, a comparison refused on a missing leg, or one refused
    on any other rule has no code."""
    if comp.get("usable") or comp.get("reason") is None:
        return None
    compat = _mapping(comp.get("compatibility"))
    if compat.get("cross_engine") is True and comp.get("reason") == compat.get("cross_engine_refusal"):
        return CROSS_ENGINE_RULING
    return None


def comparisons_not_run(comps: Sequence[Mapping[str, Any]]) -> list[str]:
    """The names of the comparisons that filed no estimand and were NOT refused by design — what the verdict and the
    production boundary call "did not run" (the GSM8K reader's ``_not_run``, mirrored): a comparison whose refusal matched
    its declared expected refusal EXACTLY (:data:`REFUSED_BY_DESIGN_KEY`) is a filed outcome, not a comparison that did
    not run."""
    return [c["name"] for c in comps if c.get("estimands") is None and not c.get(REFUSED_BY_DESIGN_KEY)]


def run_comparison(comp: Mapping[str, Any], legs: Mapping[str, Mapping[str, Any]], *, n_resamples: int, seed: int,
                   ci: float, declared: bool = False) -> dict[str, Any]:
    """One (model, mode) comparison: usable iff every leg it names is; then the
    estimand under both branches, the per-item transitions and the exact
    McNemar sensitivity (bare -> each arm), and the verdict rule per contrast
    (a word only on the primary row). ``declared`` (B9): whether a declared
    tree set admits a cross-tree comparison through the compatibility relations."""
    out = dict(comp)
    bad = unusable_legs(legs)
    unusable = {kind: bad[k] for kind, k in comp["leg_keys"].items() if k in bad}
    out["usable"] = not unusable
    out["reason"] = ("; ".join(f"{kind}: {why}" for kind, why in unusable.items()) or None)
    out["estimands"] = None
    if unusable:
        return out
    labels = dict(comp["leg_keys"])
    problem = _comparable_problem(legs, labels, what="the three legs'" if len(labels) == 3 else "the legs'", declared=declared)
    out["compatibility"] = compatibility_record(legs, labels, declared=declared)
    if problem:
        out["usable"] = False
        out["reason"] = problem
        # the refusal's CODE, derived from the record (comparison_refusal_code), and whether it matched the design's declared
        # expected refusal EXACTLY — the population gate, the verdict and the production boundary read the flag, never the text
        code = comparison_refusal_code(out)
        if code is not None:
            out[REFUSAL_CODE_KEY] = code
        if out.get(ROW_EXPECTED_REFUSAL_KEY) is not None and code == out[ROW_EXPECTED_REFUSAL_KEY]:
            out[REFUSED_BY_DESIGN_KEY] = True
        return out
    contrasts = comp_contrasts(comp)
    est = estimands(legs, labels, contrasts, n_resamples=n_resamples, seed=seed, ci=ci)
    out["estimands"] = est
    pairs = {name: (KIND_BARE, plus_kind(signs)) for name, signs in contrasts.items()}
    by_kind = {kind: legs[k] for kind, k in labels.items()}
    out["transitions"] = {c: T.transitions(by_kind, a, b) for c, (a, b) in pairs.items()}
    mc = {c: T.mcnemar_by_branch(by_kind, a, b) for c, (a, b) in pairs.items()}
    out["mcnemar"] = T.attach_registered_replicate_signs(mc, {c: (est, c) for c in pairs})
    out["verdicts"] = {c: verdict_for(est, c, mcnemar=mc[c], file_word=bool(comp["primary"])) for c in contrasts}
    return out


def run_mode_delta(model_key: str, mode: str, legs: Mapping[str, Mapping[str, Any]], *, n_resamples: int, seed: int,
                   ci: float, kinds: Sequence[str] | None = None, declared: bool = False) -> dict[str, Any]:
    kinds = list(kinds) if kinds is not None else list(KINDS)
    labels, contrasts = mode_delta_labels(model_key, mode, kinds)
    unusable = {lab: (legs.get(k) or {}).get("missing") or (legs.get(k) or {}).get("refused")
                or f"{k} is not in the matrix" for lab, k in labels.items()
                if k not in legs or legs[k].get("missing") or legs[k].get("refused")}
    out: dict[str, Any] = {"name": f"{model_key}/{mode}-minus-{MODE_PRIMARY}", "model_key": model_key, "mode": mode,
                           "labels": labels, "kinds": kinds, "usable": not unusable,
                           "reason": "; ".join(f"{lab}: {why}" for lab, why in unusable.items()) or None,
                           "estimands": None}
    if unusable:
        return out
    problem = _comparable_problem(legs, labels, what="the two rows'", declared=declared, same_sampling=False)
    if problem:
        out["usable"] = False
        out["reason"] = problem
        return out
    out["estimands"] = estimands(legs, labels, contrasts, n_resamples=n_resamples, seed=seed, ci=ci)
    return out


def _legs_without(legs: Mapping[str, Mapping[str, Any]], drop: Sequence[str]) -> dict[str, Mapping[str, Any]]:
    """The legs with the masked items removed from their ``items`` (shallow copies; the per-item transition and McNemar
    readers walk ``items`` and so read the without reading like the estimand does)."""
    gone = set(drop)
    return {k: ({**leg, "items": {i: r for i, r in _mapping(leg.get("items")).items() if i not in gone}} if leg.get("items") else leg)
            for k, leg in legs.items()}


#: the SCOPE of a mask, stated once and carried on every record it touches. The first cut re-ran the WHOLE comparison
#: without the item, so one nemotron ctl2 draw moved every nemotron arm and all five contrasts (none 441/498 ->
#: 441/497) and every leg of the pooled ctl2 arm.
MASK_SCOPE_RULE = ("review H1: a masked primary draw (D48) is removed from the ONE contrast whose two legs it sits on — the masked "
                   "leg's arm − bare, filed over that pair's items minus the masked ones with the same seeded draws the gate's re-run "
                   "used — and from the masked leg's own accuracy (its population is its items minus the masked ones); every other "
                   "arm, contrast and leg keeps its full population and is byte-equal to the unmasked reading; inside a pooled arm "
                   "the item is removed from its OWN model's arm↔bare pair only (that pair's conditional accuracies are over its "
                   "unmasked items; the shared resample stays over the panel's common items); every record that consumes the masked "
                   "leg files the reading WITH the items beside as `estimands_with_anomalies`")


def _masked_contrasts(labels: Mapping[str, str], contrasts: Mapping[str, Mapping[str, int]],
                      exclude_by_leg: Mapping[str, Sequence[str]]) -> dict[str, tuple[dict[str, str], list[str]]]:
    """``{contrast: (its labels -> leg keys, the masked ids on those legs)}`` for every contrast one of whose legs carries a
    masked item — the contrasts a mask is SCOPED to (:data:`MASK_SCOPE_RULE`); a contrast reading no masked leg is absent."""
    out: dict[str, tuple[dict[str, str], list[str]]] = {}
    for name, signs in contrasts.items():
        sub = {lab: labels[lab] for lab in signs}
        drop = sorted(set().union(*(set(exclude_by_leg.get(k, ())) for k in sub.values())))
        if drop:
            out[name] = (sub, drop)
    return out


_PER_LEG_ESTIMAND_KEYS: tuple[str, ...] = ("acc_by_leg", "n_scored_by_leg", "n_censored_by_leg", "n_correct_by_leg")


def _spliced_estimands(est_with: Mapping[str, Mapping[str, Any]], legs: Mapping[str, Mapping[str, Any]], labels: Mapping[str, str],
                       contrasts: Mapping[str, Mapping[str, int]], masked: Mapping[str, tuple[Mapping[str, str], Sequence[str]]],
                       exclude_by_leg: Mapping[str, Sequence[str]], *, n_resamples: int, seed: int, ci: float
                       ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """The FILED estimands under :data:`MASK_SCOPE_RULE`: ``est_with`` (every leg over the record's full population) with
    each masked contrast's statistics (``stats``, ``complete_case``) replaced by that contrast's OWN estimand over its legs
    without the masked items — the draws are a pure function of (seed, n, B), so they are the very draws the gate's re-run
    used for that contrast — and the MASKED leg's per-leg fields (accuracy, counts) taken from the same estimand (its own
    population). Every other contrast's and leg's field is the object from ``est_with``: byte-equal. ``n_items`` stays the
    record's population; ``n_items_by_contrast`` names each contrast's. Returns (estimands, the per-contrast mask record)."""
    est = {b: {**est_with[b], **{k: dict(est_with[b][k]) for k in _PER_LEG_ESTIMAND_KEYS}, "stats": dict(est_with[b]["stats"]),
               "complete_case": dict(est_with[b]["complete_case"])} for b in BRANCHES}
    record: dict[str, Any] = {}
    for name, (sub, drop) in masked.items():
        sub_est = estimands(legs, dict(sub), {name: dict(contrasts[name])}, n_resamples=n_resamples, seed=seed, ci=ci, exclude_ids=drop)
        for b in BRANCHES:
            est[b]["stats"][name] = sub_est[b]["stats"][name]
            est[b]["complete_case"][name] = sub_est[b]["complete_case"][name]
            for lab, k in sub.items():
                if exclude_by_leg.get(k):
                    for key in _PER_LEG_ESTIMAND_KEYS:
                        est[b][key][lab] = sub_est[b][key][lab]
        record[name] = {"legs": dict(sub), "n_items": sub_est[BRANCH_PRIMARY]["n_items"], "masked_items": list(drop),
                        "acc_by_leg": {b: sub_est[b]["acc_by_leg"] for b in BRANCHES}}
    for b in BRANCHES:
        est[b]["n_items_by_contrast"] = {name: (record[name]["n_items"] if name in record else est_with[b]["n_items"]) for name in contrasts}
    return est, record


def apply_primary_masks(comps: Sequence[dict[str, Any]], deltas: Sequence[dict[str, Any]], legs: Mapping[str, Mapping[str, Any]],
                        exclude_by_leg: Mapping[str, Sequence[str]], *, n_resamples: int, seed: int, ci: float) -> None:
    """Under the mask scope rule (:data:`MASK_SCOPE_RULE`): in every comparison and mode delta that consumes a masked
    PRIMARY leg, the contrast(s) reading that leg are filed WITHOUT the masked items — their estimand over the pair's
    legs (:func:`_spliced_estimands`), their transitions and McNemar sensitivity over the same without-legs, their
    verdicts from the spliced estimands — while every other contrast, leg, transition and McNemar record is byte-equal to
    the unmasked reading; the WITH reading is kept as ``estimands_with_anomalies``, the masked ids per leg as
    ``masked_items``, the per-contrast record as ``masked_contrasts``, the count as ``n_masked``, the rule as
    ``mask_rule``. In place."""
    for c in comps:
        if c.get("estimands") is None:
            continue
        labels = dict(c["leg_keys"])
        contrasts = comp_contrasts(c)
        masked = _masked_contrasts(labels, contrasts, exclude_by_leg)
        if not masked:
            continue
        est, record = _spliced_estimands(c["estimands"], legs, labels, contrasts, masked, exclude_by_leg,
                                         n_resamples=n_resamples, seed=seed, ci=ci)
        pairs = {name: (KIND_BARE, plus_kind(signs)) for name, signs in contrasts.items()}
        by_kind_with = {kind: legs[k] for kind, k in labels.items()}
        trans: dict[str, Any] = {}
        mc: dict[str, Any] = {}
        for name, (a, b_) in pairs.items():
            by_kind = by_kind_with
            if name in masked:
                without = _legs_without(legs, masked[name][1])
                by_kind = {kind: without[k] for kind, k in labels.items()}
            trans[name] = T.transitions(by_kind, a, b_)
            mc[name] = T.mcnemar_by_branch(by_kind, a, b_)
        c["estimands_with_anomalies"] = c["estimands"]
        c["estimands"] = est
        c["transitions"] = trans
        c["mcnemar"] = T.attach_registered_replicate_signs(mc, {name: (est, name) for name in pairs})
        c["verdicts"] = {name: verdict_for(est, name, mcnemar=mc[name], file_word=bool(c["primary"])) for name in contrasts}
        c["masked_items"] = {kind: list(exclude_by_leg[k]) for kind, k in labels.items() if exclude_by_leg.get(k)}
        c["masked_contrasts"] = record
        c["n_masked"] = len(set().union(*(set(v[1]) for v in masked.values())))
        c["mask_rule"] = MASK_SCOPE_RULE
    for d in deltas:
        if d.get("estimands") is None:
            continue
        labels = dict(d["labels"])
        _labels, contrasts = mode_delta_labels(str(d["model_key"]), str(d["mode"]), d.get("kinds") or KINDS)
        masked = _masked_contrasts(labels, contrasts, exclude_by_leg)
        if not masked:
            continue
        est, record = _spliced_estimands(d["estimands"], legs, labels, contrasts, masked, exclude_by_leg,
                                         n_resamples=n_resamples, seed=seed, ci=ci)
        d["estimands_with_anomalies"] = d["estimands"]
        d["estimands"] = est
        d["masked_items"] = {lab: list(exclude_by_leg[k]) for lab, k in labels.items() if exclude_by_leg.get(k)}
        d["masked_contrasts"] = record
        d["n_masked"] = len(set().union(*(set(v[1]) for v in masked.values())))
        d["mask_rule"] = MASK_SCOPE_RULE


# --------------------------------------------------------------------------- #
# the gates — these, and only these, withhold a word                           #
# --------------------------------------------------------------------------- #


def usable_legs(legs: Mapping[str, Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    """The legs the gates read — neither missing nor refused (every such leg
    carries every record :func:`collect_leg` files; :func:`unusable_legs` names
    the rest)."""
    return {k: leg for k, leg in legs.items() if not (leg.get("missing") or leg.get("refused"))}


def _fold_gate(per: Mapping[str, Any], failures: list[str], pending: list[str], stamps: list[Any]) -> dict[str, Any]:
    """A per-leg gate's record: FAIL over PENDING over PASS, the detail the
    failures (else the pending items) joined."""
    status = GATE_FAIL if failures else (GATE_PENDING if pending else GATE_PASS)
    return _gate(status, "; ".join(failures) or ("; ".join(pending) or None), legs=per, failures=failures,
                 pending=pending, pending_stamps=stamps)


def population_gate(legs: Mapping[str, Mapping[str, Any]], comps: Sequence[Mapping[str, Any]],
                    math500_ids: Sequence[str] | None, deltas: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """Gate 0: every leg usable (500 unique attempted draws, verified against
    the summary, rows == n_scored — each enforced at collection); every
    comparison's legs over ONE id set and ONE producer tree; every REQUIRED
    mode delta pairing its sensitivity row with the primary row the same way
    (a triple consistent within itself but from another tree than the
    primary is an estimand the design requires and cannot have — a FAILURE
    here, never a silent omission); every leg's ids the CANONICAL MATH-500 set
    when supplied (``--math500-ids``; absent -> PENDING, an operator input
    never waived)."""
    bad = unusable_legs(legs)
    if bad:
        return _gate(GATE_FAIL, f"unusable legs: {bad}", n=None, canonical=None)
    expected: dict[str, str] = {}
    for comp in comps:      # every leg is usable here, so a comparison that did not run failed the comparability rule ...
        if comp.get(EXPECTED_REFUSAL_PROBLEM_KEY):
            return _gate(GATE_FAIL, f"{comp['name']}: {comp[EXPECTED_REFUSAL_PROBLEM_KEY]}", n=None, canonical=None)
        code = comp.get(ROW_EXPECTED_REFUSAL_KEY)
        if comp["usable"]:
            if code is not None:
                # the design declares this comparison refuses by name, and it computed: the declaration is stale — refused, never
                # a contrast filed under a declaration that says there is none
                return _gate(GATE_FAIL, f"{comp['name']}: expected a refusal ({code}), got a contrast — the design declares this "
                             "comparison refuses by name and it computed; retire the declaration or name the pairing", n=None, canonical=None)
            continue
        if code is None:
            return _gate(GATE_FAIL, f"{comp['name']}: {comp['reason']}", n=None, canonical=None)
        # ... unless the DESIGN declared this comparison's refusal (the row's expected_refusal) and the refusal derives EXACTLY that
        # code: refused BY DESIGN — a filed outcome (the arm's numbers are reported as the arm's numbers, the pairing's absence
        # accepted), never a gate failure; a refusal with any other code, or with no named code, still fails
        if comp.get(REFUSED_BY_DESIGN_KEY):
            expected[comp["name"]] = code
            continue
        return _gate(GATE_FAIL, f"{comp['name']}: declared an expected refusal ({code}) but refused with "
                     f"{comp.get(REFUSAL_CODE_KEY) or 'no named code'} — {comp['reason']}", n=None, canonical=None)
    for delta in deltas:    # ... and so did a required mode delta that did not run
        if not delta["usable"]:
            return _gate(GATE_FAIL, f"required mode delta {delta['name']}: {delta['reason']}", n=None, canonical=None)
    n = len(next(iter(legs.values()))["attempted_ids"]) if legs else 0
    if math500_ids is None:
        return _gate(GATE_PENDING, "no canonical MATH-500 id set supplied (--math500-ids plan|<json>); the population's "
                     "identity is not verified", n=n, canonical="not supplied", expected_refusals=expected)
    want = {str(i) for i in math500_ids}
    if len(want) != N_ITEMS:
        return _gate(GATE_FAIL, f"the supplied canonical id set has {len(want)} ids, not {N_ITEMS}", n=n, canonical="invalid")
    for k, leg in legs.items():
        have = set(leg["attempted_ids"])
        if have != want:
            return _gate(GATE_FAIL, f"{k}: the attempted ids are not the canonical MATH-500 set — {len(have - want)} attempted "
                         f"but not canonical (first {sorted(have - want)[:3]}), {len(want - have)} canonical but not attempted "
                         f"(first {sorted(want - have)[:3]})", n=n, canonical="mismatch")
    return _gate(GATE_PASS, None, n=n, canonical="verified", expected_refusals=expected)


def _per_leg_gate(legs: Mapping[str, Mapping[str, Any]], key: str) -> dict[str, Any]:
    failures, pending, stamps = [], [], []
    per: dict[str, Any] = {}
    for k, leg in usable_legs(legs).items():
        rec = leg[key]
        per[k] = rec
        failures.extend(f"{k}: {f}" for f in rec.get("failures", []))
        for s in rec.get("pending_stamps", []):
            pending.append(f"{k}: {s} not stamped")
            stamps.append(s)
    return _fold_gate(per, failures, pending, stamps)


def ledger_gate(legs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Gate 1: the ledger confirmation of every usable leg (:func:`ledger_confirmation`)."""
    g = _per_leg_gate(legs, "ledger")
    g["legs"] = {k: {f: v for f, v in rec.items() if f != "binding"} for k, rec in g["legs"].items()}
    return g


def identity_gate(legs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Gate 2: the request evidence, the stamp triple and the per-row sampling
    digest of every usable leg. (The manifest-level identity — model / cell /
    version / keys / the sampling row — refuses the LEG at collection and
    reaches this report through the population gate.)"""
    failures, pending, stamps = [], [], []
    per: dict[str, Any] = {}
    for k, leg in usable_legs(legs).items():
        rec: dict[str, Any] = {"request_evidence": leg.get("request_evidence"), "stamps": leg.get("stamps"),
                               "sampling_ref": leg.get("sampling_ref")}
        per[k] = rec
        for f in (leg.get("request_evidence") or {}).get("failures", []):
            failures.append(f"{k}: request evidence: {f}")
        for f in (leg.get("stamps") or {}).get("failures", []):
            failures.append(f"{k}: stamps: {f}")
        sr = leg.get("sampling_ref") or {}
        if sr.get("status") == GATE_FAIL:
            failures.append(f"{k}: {sr['detail']}")
        elif sr.get("status") == GATE_PENDING:
            pending.append(f"{k}: {sr['detail']}")
            stamps.extend(sr.get("pending_stamps", []))
    return _fold_gate(per, failures, pending, stamps)


def certification_gate(legs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Gate 3: predicates (A) on every leg, (B′) and (C) on every fitted job,
    zero declared cells, every family's render fact certified."""
    return _per_leg_gate(legs, "certification")


#: the at-cap fields the cap gate copies into its per-leg record: the rate, the counts it is built from, the two cases
_CAP_GATE_AT_CAP_FIELDS: tuple[str, ...] = ("rate", "share", "n_at_cap", "n_with_stop_reason", "n_case_b_unterminated_at_cap",
                                            "n_case_c_answer_at_cap")


def cap_gate(legs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Gate 4: the parent's re-cost rule — the summary's at-cap rate <= 2 % per
    cell. BINDING below the ceiling (a cell over it is re-costed, not read as
    capability; no re-run inside this spec). ADVISORY at the ceiling (``CAP >=
    AT_CAP_CEILING``: one pass, no rung above): the cell's share is
    reported here and by the truncation readout (the 1 % line, the 3 %
    headline) and the gate passes — refusing would suppress exactly the
    reading the rule asks for. A null rate is a torn record either way. The
    two at-cap cases are counted apart: (b) the cap fell inside the think block
    (unterminated, unscored under the primary policy), (c) after it (scored)."""
    binding = CAP < AT_CAP_CEILING
    per: dict[str, Any] = {}
    failures = []
    advisories = []
    for k, leg in usable_legs(legs).items():
        # the gate's record is about the RATE: the counts it is built from and the two cases — not the loop split, whose
        # source differs between the two runs by design (`REPORT_DIFF_SKIP_SPLIT_FIELDS`) and belongs to the readout
        rec = {f: leg["at_cap"][f] for f in _CAP_GATE_AT_CAP_FIELDS}
        rec["threshold"] = CAP_GATE_AT_CAP_RATE
        rec["binding"] = binding
        rate = rec["rate"]
        if rate is None:
            rec["status"] = GATE_FAIL
            failures.append(f"{k}: at_cap_rate is null (no draw recorded a stop reason)")
        elif rate > CAP_GATE_AT_CAP_RATE:
            over = f"{k}: at_cap_rate {100 * rate:.1f} % > {100 * CAP_GATE_AT_CAP_RATE:.0f} %"
            if binding:
                rec["status"] = GATE_FAIL
                failures.append(f"{over} — the cell is re-cost, not a capability reading at this cap")
            else:
                rec["status"] = GATE_PASS
                rec["advisory"] = f"{over} — reported, not refused: the {CAP:,} cap is the ceiling (no rung above; D68)"
                advisories.append(rec["advisory"])
        else:
            rec["status"] = GATE_PASS
        per[k] = rec
    g = _gate(GATE_FAIL if failures else GATE_PASS, "; ".join(failures) or None, legs=per)
    g.update(binding=binding, ceiling=AT_CAP_CEILING, cap=CAP, advisories=advisories)
    return g


def _class_words(rec: Mapping[str, Any]) -> str:
    """The classes of one leg's unterminated-not-at-cap draws, as a clause."""
    if rec["class_source"] not in CLASS_SOURCES_CLASSIFIED:
        return "classes unavailable in this run (no .eval log read; run E's report carries them)"
    parts = []
    if rec["n_lifted_close"]:
        lits = ", ".join(f"`{k}` x{v}" for k, v in sorted(rec["tag_literals"].items(), key=lambda kv: -kv[1])) or "an open-tag literal"
        parts.append(f"{rec['n_lifted_close']} lifted_close (a closing-tag occurrence in the raw body was removed by inspect_ai's "
                     f"client-side lift, which paired it with a tag literal the model emitted while reasoning — {lits} — so the "
                     "completion the scorer read has no close; in each the last raw close sits outside any code span and an ANSWER: "
                     "line follows it; the class is a reading of the text, not a finding about the model)")
    if rec["n_eos_in_thoughts"]:
        parts.append(f"{rec['n_eos_in_thoughts']} eos_in_thoughts (no close in the raw body, finish_reason stop: the model ended its "
                     "turn inside its thoughts)")
    if rec["n_class_other"]:
        whys = sorted({c["why"] for c in rec["classes"].values() if c["class"] == ANOMALY_OTHER})
        parts.append(f"{rec['n_class_other']} other ({'; '.join(whys)[:300]})")
    return "; ".join(parts) or "no unterminated-not-at-cap draw"


def split_anomaly_gate(legs: Mapping[str, Mapping[str, Any]], comps: Sequence[Mapping[str, Any]], *, n_resamples: int,
                       seed: int, ci: float, deltas: Sequence[Mapping[str, Any]] = ()) -> tuple[dict[str, Any], list[str]]:
    """Gate 5 (per family split mode, and the unterminated-not-at-cap
    rule): NO scored row with an unterminated block (the truncated-Nemotron trap
    — a scored unterminated row is a FAILURE); ``implicit_open`` families: every
    completion carries the block; ``explicit_tags`` (GLM): the present rate is a
    FINDING about the render; ``record_only`` (gemma): the reasoning-only count
    reported. An unterminated block NOT at the cap is sub-classed from run E's
    log (:func:`classify_unterminated_anomaly`; ``unavailable`` in the run
    without logs): on a PRIMARY leg any such draw FAILS the gate, whatever its
    class (fail-closed); on a SENSITIVITY leg the draws are masked and reported
    (they stay unscored under their class). Every anomalous item (the sub-classed
    kind, unscored-for-another-reason, a stray open tag) is masked out of EVERY
    leg of each comparison that reads its leg and the verdict rule re-run:
    ``word_moves`` is tri-state — ``False`` when every such comparison was re-run
    and no word moved, ``True`` when one did (FAILED, the comparison named),
    ``"unusable"`` when a comparison that reads a masked leg did not run
    (PENDING, never PASSED by vacuity; ``without[<name>].skipped`` says why).
    "Comparison" means EVERY record that consumes the leg: the three-cell
    comparisons AND the sensitivity-minus-primary mode deltas (``deltas``, each
    with its own six labels and three contrasts: a delta's
    word can move while the three-cell word stands). The word compared is the
    verdict RULE's word on both sides (``file_word`` on), so a sensitivity
    comparison or a delta — which file no word — is still checked. Returns the
    gate and the findings it raises."""
    counts: dict[str, Any] = {}
    failures: list[str] = []
    pending: list[str] = []
    findings: list[str] = []
    primary_candidates: dict[str, dict[str, Any]] = {}       # Primary legs whose every anomaly is a classified eos_in_thoughts
    usable = usable_legs(legs)
    # the parent's anomaly predicate (unterminated-not-at-cap, unscored-for-another-reason, stray open tag), per leg
    masked: dict[str, list[str]] = {k: ids for k, ids in T.anomaly_ids(usable).items() if ids}
    for k, leg in usable.items():
        s, items = leg["summary"], leg["items"]
        mode = leg["render_fact"]["split_mode"]
        n = len(items)
        present = sum(1 for r in items.values() if r["think_block_present"])
        scored_unterminated = sorted(i for i, r in items.items() if r["scored"] and r["think_block_unterminated"])
        stray = sorted(i for i, r in items.items() if r["stray_open_tag"])
        unsc = leg["unscored"]
        # the classes: the log's reading (run E), else the matched report's (run F), else none
        classes = (leg.get("eval_check") or {}).get("anomaly_classes") or leg.get("matched_anomaly_classes") or {}
        literals: Counter = Counter(lit for c in classes.values() for lit in c.get("tag_literals", []))
        rec = {"split_mode": mode, "primary": bool(leg["primary"]), "n_unterminated_not_at_cap": s.get("n_unterminated_not_at_cap"),
               "n_think_block_present": present, "think_present_rate": present / n if n else None,
               "n_scored_unterminated": len(scored_unterminated), "n_stray_open_tag": len(stray),
               "n_content_empty_with_reasoning": s.get("n_content_empty_with_reasoning"),
               "n_case_c_answer_at_cap": leg["at_cap"]["n_case_c_answer_at_cap"],
               "anomaly_kinds": anomaly_kinds(items),
               # the sub-classes (declared by-design leaves: run F files `unavailable`)
               "class_source": unsc["class_source"], "n_lifted_close": unsc["by_class"][ANOMALY_LIFTED_CLOSE],
               "n_eos_in_thoughts": unsc["by_class"][ANOMALY_EOS_IN_THOUGHTS], "n_class_other": unsc["by_class"][ANOMALY_OTHER],
               "n_class_unavailable": unsc["by_class"][ANOMALY_CLASS_UNAVAILABLE],
               "classes": {i: {"class": c["class"], "why": c["why"], "tag_literals": c["tag_literals"]} for i, c in sorted(classes.items())},
               "tag_literals": dict(literals)}
        counts[k] = rec
        n_unt = s.get("n_unterminated_not_at_cap")
        if n_unt != 0:
            words = _class_words(rec)
            if leg["primary"]:
                # a primary leg whose EVERY such draw is a classified `eos_in_thoughts` is a CANDIDATE for mask-and-report,
                # decided by the word check below (no re-run word may move; every consuming record must have run); any other
                # class, an unclassified draw, or a moved word keeps the fail-closed rule
                classified = rec["class_source"] in CLASS_SOURCES_CLASSIFIED
                all_eos = classified and rec["n_eos_in_thoughts"] == n_unt and rec["n_lifted_close"] == 0 and rec["n_class_other"] == 0
                if all_eos:
                    primary_candidates[k] = {"n": int(n_unt), "words": words, "class_source": rec["class_source"]}
                else:
                    failures.append(f"{k}: n_unterminated_not_at_cap {n_unt!r} != 0 on a PRIMARY leg — {words}; fail-closed on either class "
                                    "(D180)")
            else:
                findings.append(f"{k}: {n_unt} unterminated-not-at-cap draw(s) on a SENSITIVITY leg — {words}; the items stay unscored "
                                "under their class, are masked out of every comparison and mode delta that reads the leg, and the "
                                "verdict rule is re-run without them (D180)")
        if scored_unterminated:
            failures.append(f"{k}: {len(scored_unterminated)} scored row(s) with an unterminated think block (first "
                            f"{scored_unterminated[:3]}) — under {mode!r} an unterminated block is unscoreable (D11)")
        if mode == C.THINK_SPLIT_IMPLICIT_OPEN and present != n:
            failures.append(f"{k}: n_think_block_present {present} != {n} under implicit_open — the prompt opens the block on "
                            "every completion")
        if mode == C.THINK_SPLIT_EXPLICIT_TAGS and present != n:
            # `present` is the producer's flag — a tag in the TEXT — not the draws that emitted reasoning: inspect_ai lifts a
            # closed <think>…</think> block out of the text client-side, and the no-reasoning readout reads the reasoning
            # channel beside the flag (reasoning_source); say what the count IS so it does not contradict the reasoning finding
            by_channel = leg["thinking"]["n_present_by_channel"]
            neither = n - present - by_channel
            if by_channel and not neither:
                rest = f"the reasoning channel carries the other {by_channel} (see the reasoning finding)"
            elif by_channel:
                rest = (f"the reasoning channel carries {by_channel} more (see the reasoning finding); {neither} carry neither "
                        "(the D115 finding)")
            else:
                rest = f"the reasoning channel is empty on the other {neither} (the D115 finding)"
            findings.append(f"{k}: think block present in the TEXT on {present} of {n} completions ({100 * present / n:.1f} %) — GLM "
                            f"emits its own tags and inspect_ai lifts a closed block out of the text client-side; {rest}; a tag-free "
                            "completion is scored as the answer (a fact about the render, not a failure)")
    # the word check, over EVERY comparison that reads a masked leg: re-run without the anomalous items (masked from every
    # leg of the comparison), the verdict RULE's word before and after; a comparison that did not run is recorded as
    # skipped and leaves the gate PENDING (never a pass by vacuity)
    moved = False
    unusable = False
    without: dict[str, Any] = {}
    # every record that consumes a leg: (name, kind, labels, contrasts, usable, estimands, reason, word_filed)
    records: list[tuple[str, str, dict[str, str], Mapping[str, Mapping[str, int]], bool, Any, str | None, bool]] = []
    for comp in comps:
        records.append((str(comp["name"]), "comparison", dict(comp["leg_keys"]), comp_contrasts(comp), bool(comp.get("usable")),
                        comp.get("estimands"), comp.get("reason"), bool(comp["primary"])))
    for d in deltas:
        labels, contrasts = mode_delta_labels(str(d["model_key"]), str(d["mode"]), d.get("kinds") or KINDS)
        records.append((str(d["name"]), "mode_delta", labels, contrasts, bool(d.get("usable")), d.get("estimands"),
                        d.get("reason"), False))
    for name, kind, labels, contrasts, is_usable, est_before, reason, word_filed in records:
        if not any(k in masked for k in labels.values()):
            continue
        drop = sorted(set().union(*(masked.get(k, []) for k in labels.values())))
        if not is_usable or est_before is None:
            why = reason or f"the {kind} did not run"
            without[name] = {"kind": kind, "masked_out": drop, "skipped": why, "word_before": None, "word_after": None,
                             "word_filed": word_filed}
            pending.append(f"{name}: {len(drop)} anomalous item(s) masked but the {kind} did not run ({why}) — the word check has "
                           "not run")
            unusable = True
            continue
        try:
            est = estimands(legs, labels, contrasts, n_resamples=n_resamples, seed=seed, ci=ci, exclude_ids=drop)
        except AnalysisError as e:
            failures.append(f"{name}: re-running the rule without {len(drop)} anomalous items failed: {e}")
            continue
        before = {c: verdict_for(est_before, c, file_word=True)["word"] for c in contrasts}
        after = {c: verdict_for(est, c, file_word=True)["word"] for c in contrasts}
        without[name] = {"kind": kind, "masked_out": drop, "skipped": None, "word_before": before, "word_after": after,
                         "word_filed": word_filed}
        if after != before:
            moved = True
            failures.append(f"{name} ({kind}): {len(drop)} anomalous items change the word ({before} -> {after})")
    word_moves: bool | str = True if moved else (WORD_MOVES_UNUSABLE if unusable else False)
    # a primary candidate is ADMITTED (masked and reported) only when every record consuming its leg was re-run and no
    # word moved on any of them; a moved word, or a consuming record that did not run, keeps the fail-closed rule by name
    primary_masked: dict[str, dict[str, Any]] = {}
    for k, cand in primary_candidates.items():
        consuming = {name: w for name, w in without.items() if k in masked and set(masked[k]) <= set(w["masked_out"])}
        moved_here = [name for name, w in consuming.items() if w["skipped"] is None and w["word_after"] != w["word_before"]]
        skipped_here = [name for name, w in consuming.items() if w["skipped"] is not None]
        if not consuming or moved_here or skipped_here:
            why = (f"the word moves on {moved_here}" if moved_here else
                   f"a consuming record did not run ({skipped_here})" if skipped_here else "no record consumes the leg, so no word check ran")
            failures.append(f"{k}: n_unterminated_not_at_cap {cand['n']} != 0 on a PRIMARY leg — {cand['words']}; {why}; fail-closed "
                            "(D180; D48 admits an eos_in_thoughts draw only when no verdict word moves without it)")
            continue
        items_rec = []
        for item in masked[k]:
            it = _mapping(_mapping(usable[k].get("items")).get(item))
            items_rec.append({"item_id": item, "class": ANOMALY_EOS_IN_THOUGHTS, "completion_tokens": it.get("completion_tokens"),
                              "finish_reason": it.get("stop_reason"), "scored": bool(it.get("scored")), "at_cap": bool(it.get("at_cap"))})
        primary_masked[k] = {"n_masked": len(masked[k]), "items": items_rec, "class_source": cand["class_source"],
                             "records_rerun": sorted(consuming), "word_moves": False,
                             "rule": ("D48: an eos_in_thoughts draw on a primary leg is masked from every record that consumes the leg "
                                      "and reported — the item, both readings (with / without), n_masked per leg and per arm — because the "
                                      "gate's own re-run without it moved no verdict word; the item stays unscored")}
        findings.append(f"{k}: {cand['n']} eos_in_thoughts draw(s) on a PRIMARY leg ({', '.join(masked[k])}) — masked from every record "
                        f"that consumes the leg, scoped to the contrast(s) that read it (review H1), and reported; the verdict rule "
                        f"re-run without them moves no word on {sorted(consuming)} (D48; the item stays unscored; both readings are "
                        "filed on the comparison)")
    status = GATE_FAIL if failures else (GATE_PENDING if pending else GATE_PASS)
    n_masked = sum(len(v) for v in masked.values())
    n_rerun = sum(1 for w in without.values() if w["skipped"] is None)
    detail = "; ".join(failures) or "; ".join(pending) or (
        f"{n_masked} anomalous items reported; the word does not move without them ({n_rerun} comparison(s) re-run, mode deltas "
        "included)" + (f"; {sum(v['n_masked'] for v in primary_masked.values())} on primary leg(s) masked from the filed contrast(s) "
                       f"under D48 ({sorted(primary_masked)})" if primary_masked else "") if masked else None)
    return _gate(status, detail, counts=counts, masked_out=masked, word_moves=word_moves, without=without, pending=pending,
                 primary_masked=primary_masked,
                 n_masked_by_leg={k: v["n_masked"] for k, v in primary_masked.items()},
                 rule=("a PRIMARY leg fails closed on any unterminated-not-at-cap draw (lifted_close, other, unclassified) — EXCEPT (D48) "
                       "a leg whose every such draw is a classified eos_in_thoughts AND whose consuming records, re-run without them, "
                       "move no verdict word: those draws are masked from the filed contrasts and reported (item, tokens, finish "
                       "reason, both readings, n_masked per leg and per arm); a SENSITIVITY leg's are masked and reported (the items "
                       "stay unscored under their class; lifted_close is a reading of the text — a close the client-side lift removed, "
                       "outside any code span, an ANSWER: line after it — not a finding about the model); every anomalous item is "
                       "masked from every leg of each record that consumes its leg — the three-cell comparisons AND the "
                       "sensitivity-minus-primary mode deltas — and the verdict rule re-run: PASSED when no re-run word moves, FAILED "
                       "when one does, PENDING when such a record did not run (D180)")), findings


def scoring_path_gate(legs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Gate 6: every usable leg scored under the primary
    unscored policy and ITS family's split mode with the family's server parser;
    no model-graded companion (``grader_max_tokens`` null; no ledger call of the
    task on another virtual model); the retained ``.eval`` logs re-derived where
    present (a present log the reader venv could not read is PENDING, not waived)."""
    failures, pending = [], []
    per: dict[str, Any] = {}
    for k, leg in usable_legs(legs).items():
        s, fact = leg["summary"], leg["render_fact"]
        rec = {"unterminated_policy": s.get("unterminated_policy"), "think_split_mode": s.get("think_split_mode"),
               "reasoning_parser": s.get("reasoning_parser"), "grader_max_tokens": s.get("grader_max_tokens"),
               "task_rows_on_other_vmodel": (leg.get("ledger") or {}).get("task_rows_on_other_vmodel"),
               "eval_check": {f: (leg.get("eval_check") or {}).get(f) for f in ("available", "agrees", "log_present", "reason")}}
        per[k] = rec
        if s.get("unterminated_policy") != C.UNTERMINATED_UNSCORED:
            failures.append(f"{k}: unterminated_policy {s.get('unterminated_policy')!r} != {C.UNTERMINATED_UNSCORED!r} (the primary policy)")
        if s.get("think_split_mode") != fact["split_mode"]:
            failures.append(f"{k}: think_split_mode {s.get('think_split_mode')!r} != the {leg['family']!r} fact's {fact['split_mode']!r} "
                            "— split under another rule (D11)")
        if s.get("reasoning_parser") != fact["server_parser"]:
            failures.append(f"{k}: reasoning_parser {s.get('reasoning_parser')!r} != the family's {fact['server_parser']!r}")
        if s.get("grader_max_tokens") is not None:
            failures.append(f"{k}: grader_max_tokens {s.get('grader_max_tokens')!r} — a model-graded companion ran; MATH-500 is "
                            "judge-free here")
        other = rec["task_rows_on_other_vmodel"] or {}
        if other:
            failures.append(f"{k}: {sum(other.values())} ledger call(s) of {TASK} on another virtual model {dict(other)} — a "
                            "grader / model-graded companion call is not this instrument")
        ec = leg.get("eval_check") or {}
        if ec.get("available") is False and ec.get("log_present") is True:
            pending.append(f"{k}: independent .eval re-derivation not run: {ec.get('reason')}")
    return _fold_gate(per, failures, pending, [None] * len(pending))


def pending_items(gates: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every PENDING entry with the producer stamp it stands for (``None`` for an
    operator input) and whether ``--allow-pending-gates`` may waive it — exactly
    :data:`WAIVABLE_PENDING_STAMPS`, nothing else."""
    items: list[dict[str, Any]] = []
    for gate in GATE_NAMES:
        g = gates.get(gate)
        if not isinstance(g, Mapping) or g.get("status") != GATE_PENDING:
            continue
        named = g.get("pending")
        if isinstance(named, list) and named:
            stamps = g.get("pending_stamps") or [None] * len(named)
            for detail, stamp in zip(named, stamps if len(stamps) == len(named) else [None] * len(named)):
                items.append({"gate": gate, "detail": detail, "stamps": [stamp],
                              "waivable": stamp is not None and stamp in WAIVABLE_PENDING_STAMPS})
        else:
            items.append({"gate": gate, "detail": str(g.get("detail")), "stamps": [None], "waivable": False})
    return items


def all_gates(legs: Mapping[str, Mapping[str, Any]], comps: Sequence[Mapping[str, Any]], *, math500_ids: Sequence[str] | None,
              n_resamples: int, seed: int, ci: float, deltas: Sequence[Mapping[str, Any]] = ()) -> tuple[dict[str, Any], list[str]]:
    gates: dict[str, Any] = {"population": population_gate(legs, comps, math500_ids, deltas), "ledger": ledger_gate(legs),
                             "identity": identity_gate(legs), "certification": certification_gate(legs),
                             "cap": cap_gate(legs)}
    gates["split_anomalies"], findings = split_anomaly_gate(legs, comps, n_resamples=n_resamples, seed=seed, ci=ci, deltas=deltas)
    gates["scoring_path"] = scoring_path_gate(legs)
    gates["failed"] = sorted(k for k in GATE_NAMES if gates[k]["status"] == GATE_FAIL)
    gates["pending"] = sorted(k for k in GATE_NAMES if gates[k]["status"] == GATE_PENDING)
    gates["pending_items"] = pending_items(gates)
    gates["pending_unwaivable"] = [it for it in gates["pending_items"] if not it["waivable"]]
    return gates, findings


# --------------------------------------------------------------------------- #
# the tables and the report                                                    #
# --------------------------------------------------------------------------- #

_CONTROLS_SUMMARY_FIELDS: tuple[str, ...] = (
    "acc", "n", "n_scored", "n_unscored", "unparseable_rate", "n_unparseable", "n_answered", "at_cap_rate", "n_at_cap",
    "n_stop_reason", "mean_completion_tokens", "p99_completion_tokens", "n_completion_tokens", "n_think_block_present",
    "n_think_block_unterminated", "n_unterminated_not_at_cap", "n_content_empty_with_reasoning", "think_split_mode",
    "unterminated_policy", "max_tokens", "reasoning_parser", "temperature", "top_p", "top_k", "samples_per_item",
    "wall_clock_s", "n_requests", "n_http_rejected_retried", "render_regime", "salt_regime", "resumed_policy")
_LEDGER_CONTROL_FIELDS: tuple[str, ...] = (
    "n_calls", "n_unique_samples", "n_resume_rows", "n_rejected", "error_code", "error_class", "n_length", "window",
    "decode_tok_s_mean", "decode_tok_s_aggregate", "completion_tokens_quantiles", "passed", "redraws")


def controls_table(legs: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One row per leg, every value with the source it came from (the summary's
    own counts, the ledger's own rates — never re-derived from rows)."""
    rows = []
    for k, leg in legs.items():
        s = leg.get("summary") or {}
        led = leg.get("ledger") or {}
        rec = {"leg": k, "job_id": leg.get("job_id"), "model_key": leg["model_key"], "cell": leg["cell"],
               "kind": leg["kind"], "mode": leg["mode"], "fitted": leg["fitted"], "missing": leg.get("missing"),
               "refused": leg.get("refused"), "summary": {f: s.get(f) for f in _CONTROLS_SUMMARY_FIELDS},
               "at_cap": leg.get("at_cap"), "thinking": leg.get("thinking"), "unscored": leg.get("unscored"),
               "ledger": {f: led.get(f) for f in _LEDGER_CONTROL_FIELDS},
               "text": (leg.get("eval_check") or {}).get("text"), "provenance": provenance_row(leg),
               "stamps": (leg.get("stamps") or {}).get("observed"), "render_fact": leg.get("render_fact")}
        if leg_key_view(k) != VIEW_MAIN:     # E3: a row keyed by a non-main view names it (every main-keyed row is unchanged)
            rec["view"] = leg_key_view(k)
        if is_itdoc_row(leg):
            rec.update(itdoc_carriage(leg))
        rows.append(rec)
    return rows


def provenance_row(leg: Mapping[str, Any]) -> dict[str, Any]:
    """One leg's provenance: the published fields plus the per-row IDENTITY KEYS the spec names
    (``object_artifact_sha256, route_kind, view, producer_tree_sha, thinking, sampling_ref, sampling_mode`` and the
    producer's ``pin`` / ``root``) — strings, so the published report's numeric leaves are untouched (B1)."""
    ident = leg.get("identity") or {}
    producer = leg.get("producer") or {}
    out = {"job_id": leg.get("job_id"), "model_key": leg["model_key"], "cell": leg["cell"], "mode": leg["mode"],
            "cell_key": leg.get("cell_key"), "harness_config_hash": ident.get("harness_config_hash"),
            "intervention_hash": ident.get("intervention_hash"), "producer_tree_sha": ident.get("producer_tree_sha"),
            "model_id": ident.get("model_id"), "model_revision": ident.get("model_revision"),
            "object_artifact_sha256": ident.get("object_artifact_sha256"), "tuple_name": leg.get("tuple_name"),
            MANIFEST_KEY_OOS_EXPERIMENT: ident.get(MANIFEST_KEY_OOS_EXPERIMENT),
            "resume_attempts": ident.get(MANIFEST_KEY_RESUME_ATTEMPTS),
            "producer": leg.get("producer"),
            "n_resume_rows": (leg.get("ledger") or {}).get("n_resume_rows"),
            IDENTITY_KEY_VERSION: ident.get(IDENTITY_KEY_VERSION), "route_kind": leg.get("route_kind"),
            "served_positions": ((leg.get("certification") or {}).get(EVIDENCE_LENS_POSITION) or {}).get("served_positions")
            if leg.get("fitted") else None,
            # math-eight-arms-extension (class E): which evidence admitted predicate (C) on a fitted leg and, on a
            # logprob admission, the ruling's sentence — a property of the operating point, ON THE ROW
            "liveness_evidence": ((leg.get("certification") or {}).get("liveness_evidence") if leg.get("fitted") else None),
            "liveness_sentence": ((leg.get("certification") or {}).get("liveness_sentence") if leg.get("fitted") else None),
            # the identity keys of the eight-arm records — `pin` is the producer's admitted pin
            "view": leg.get("view"), "thinking": ident.get("thinking_mode"), "sampling_mode": leg["mode"],
            "sampling_ref": (leg.get("sampling_ref") or {}).get("expected"), "pin": producer.get("pin"), "root": producer.get("root"),
            "chat_template_hash": ident.get("chat_template_hash"), "wave": leg.get("wave"), "arm": leg.get("kind")}
    if is_itdoc_row(leg):       # E3: an it-doc reading's provenance row carries its lane and the qualification VERBATIM
        out.update(itdoc_carriage(leg))
    return out


_TRUNCATION_CELL_AT_CAP_FIELDS: tuple[str, ...] = (
    "share", "n_at_cap", "n_with_stop_reason", "line_1pct_exceeded", "at_or_above_surfacing_threshold",
    "n_case_b_unterminated_at_cap", "n_case_c_answer_at_cap", "split_source", "n_loop", "n_truncated_reasoning",
    "n_text_unavailable")
#: the re-draw readout beside the at-cap share on every truncation cell (requests per item, the share of items
#: re-drawn, the items with a failed sample attempt) — the ledger's numbers, `None` on a leg with no usable ledger
_TRUNCATION_CELL_REDRAW_FIELDS: tuple[str, ...] = (
    "n_redraw_calls", "n_items_redrawn", "share_items_redrawn", "max_calls_per_item", "n_items_sample_error")


def _redraw_cell_fields(leg: Mapping[str, Any]) -> dict[str, Any]:
    rd = _mapping((leg.get("ledger") or {}).get("redraws"))
    return {**{f: rd.get(f) for f in _TRUNCATION_CELL_REDRAW_FIELDS}, "redraws_attributed": rd.get("attributed")}


def _split_source_words(source: str) -> str:
    """A split source as prose: which text the loop rule ran over."""
    return ("eval logs (every at-cap draw's text)" if source == SPLIT_SOURCE_EVAL_LOG
            else "store rows only (an unscored draw's text is not carried)")


def truncation_block(legs: Mapping[str, Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The report's at-cap RESULT block: every usable leg's share
    against the 1 % line, the primary cells first (the sensitivity rows carry
    ``sensitivity: true``), the declared surfacing threshold and the premise it
    doubles, and the verdict on the premise — ``premise_failed`` True when any
    PRIMARY cell read is at or above the threshold (enough on its own), False
    ONLY when the complete DESIGNED primary set (:data:`DESIGN_MODEL_KEYS` x
    the three cells, 12) was read and every one is below it, None otherwise —
    a designed cell the matrix does not name, or names but did not produce or
    refused, is listed in ``primary_legs_not_read`` (the absent ones again in
    ``primary_legs_absent_from_matrix``) and leaves the premise undetermined,
    never holding. ``headline`` is the artifact's lead sentence when the
    premise failed, else None. A refused leg contributes no cell (its readouts
    do not exist)."""
    primary_named = [row_leg_key(r) for r in rows if row_is_primary(r)]
    cells: list[dict[str, Any]] = []
    for k, leg in legs.items():
        if leg.get("missing") or leg.get("refused"):
            continue
        ac, th = leg["at_cap"], leg["thinking"]
        cells.append({"leg": k, "model_key": leg["model_key"], "cell": leg["cell"], "kind": leg["kind"], "mode": leg["mode"],
                      "primary": bool(leg["primary"]), "sensitivity": not leg["primary"],
                      **{f: ac[f] for f in _TRUNCATION_CELL_AT_CAP_FIELDS},
                      "n_on_rows": th["n_on_rows"], "n_no_reasoning_emitted": th["n_no_reasoning_emitted"],
                      "share_no_reasoning": th["share_no_reasoning"], "no_reasoning_by_construction": th["by_construction"],
                      "n_present_by_record": th["n_present_by_record"], "n_present_by_channel": th["n_present_by_channel"],
                      **_redraw_cell_fields(leg),
                      # E3: a cell keyed by a non-main view names it (a main-keyed cell is unchanged)
                      **({"view": leg_key_view(k)} if leg_key_view(k) != VIEW_MAIN else {})})
    cells = [c for c in cells if c["primary"]] + [c for c in cells if not c["primary"]]
    primary = [c for c in cells if c["primary"]]
    read = {c["leg"] for c in primary}
    # the DESIGNED primary set is the completeness the premise is held to: a designed cell the matrix does not name is as
    # unread as one named but missing or refused — both are listed, so a reader sees which cells are missing. The design
    # is the PUBLISHED triple (DESIGN_CELLS); the eight-arm extension's further arms are read as cells, never as the premise
    designed = [leg_key(m, c, MODE_PRIMARY) for m in DESIGN_MODEL_KEYS for c in DESIGN_CELLS]
    absent = [k for k in designed if k not in primary_named]
    not_read = [k for k in designed if k not in read] + [k for k in primary_named if k not in read and k not in designed]
    over = [c["leg"] for c in primary if c["line_1pct_exceeded"]]
    at_thr = [c["leg"] for c in primary if c["at_or_above_surfacing_threshold"]]
    complete = all(k in read for k in designed) and all(c["share"] is not None for c in primary)
    premise_failed: bool | None = True if at_thr else (False if complete else None)
    with_share = [c for c in primary if c["share"] is not None]
    top = max(with_share, key=lambda c: c["share"]) if with_share else None
    headline = None
    if premise_failed:
        parts = [f"{c['leg']} {100 * c['share']:.1f} % (loop {c['n_loop']} / truncated reasoning {c['n_truncated_reasoning']} / "
                 f"text unavailable {c['n_text_unavailable']}; split from the {_split_source_words(c['split_source'])})"
                 for c in primary if c["leg"] in at_thr]
        headline = (f"The ruling's premise — a ~{100 * AT_CAP_PREMISE_RATE:.1f} % at-cap rate, measured on GSM8K — did not hold "
                    f"for MATH-500 at cap {CAP:,}: {len(at_thr)} primary cell(s) at or above the "
                    f"{100 * AT_CAP_SURFACING_THRESHOLD:.0f} % surfacing threshold: " + "; ".join(parts) +
                    ". The truncated-reasoning vs degenerate-loop split leads.")
    by_source = Counter(c["split_source"] for c in cells)
    scope = (LOOP_RULE_SCOPE_TEXT + ". This run: " + ", ".join(f"{n} cell(s) split from the {_split_source_words(s)}"
                                                              for s, n in sorted(by_source.items())) if cells
             else LOOP_RULE_SCOPE_TEXT + ". This run: no usable cell")
    redrawn = [c["leg"] for c in cells if c["n_items_redrawn"]]
    return {"premise_rate": AT_CAP_PREMISE_RATE, "premise_source": AT_CAP_PREMISE_SOURCE, "line_1pct": AT_CAP_LINE_1PCT,
            "surfacing_threshold": AT_CAP_SURFACING_THRESHOLD, "cap": CAP,
            "n_primary_cells_design": N_PRIMARY_CELLS_DESIGN, "n_primary_cells_in_matrix": len(primary_named),
            "n_primary_cells_read": len(primary), "primary_legs_not_read": not_read,
            "primary_legs_absent_from_matrix": absent, "matrix_complete": not absent,
            "primary_over_1pct": over, "primary_at_or_above_threshold": at_thr,
            "max_primary_share": None if top is None else top["share"],
            "max_primary_share_leg": None if top is None else top["leg"],
            "premise_failed": premise_failed, "headline": headline,
            "loop_rule": LOOP_RULE_TEXT, "loop_rule_scope": scope,
            # the re-draws aggregated beside the at-cap shares, with the selection caveat stated once
            "cells_redrawn": redrawn, "n_cells_redrawn": len(redrawn),
            "n_items_redrawn_total": sum(c["n_items_redrawn"] or 0 for c in cells),
            "n_redraw_calls_total": sum(c["n_redraw_calls"] or 0 for c in cells),
            "n_items_sample_error_total": sum(c["n_items_sample_error"] or 0 for c in cells),
            "max_calls_per_item": max((c["max_calls_per_item"] or 0 for c in cells), default=None),
            "redraw_mechanism": REDRAW_MECHANISM_TEXT, "redraw_ladder": LADDER_TEXT,
            "redraw_selection_caveat": REDRAW_SELECTION_CAVEAT, "cells": cells}


# --------------------------------------------------------------------------- #
# the eight-arm extension: the required-evidence set, the design block, the     #
# pooled-by-arm estimand (plan P5)                                              #


def required_rows(rows: Sequence[Mapping[str, Any]], waves: Sequence[str] | None) -> list[Mapping[str, Any]]:
    """The REQUIRED-EVIDENCE set: the ``launch`` rows whose ``wave`` is among the analysed ``waves`` (``None`` =
    every launch row, the published behaviour). Every row of it is mandatory — a missing leg fails the population gate;
    ``held`` / ``deferred`` rows and launch rows of a wave not analysed are design statuses, never collected."""
    analysed = None if waves is None else {str(w) for w in waves}
    return [r for r in rows if row_status(r) in EVIDENCE_STATUSES and (analysed is None or row_wave(r) in analysed)]


def eight_arm_mode(rows: Sequence[Mapping[str, Any]], waves: Sequence[str] | None, roots: GenRoots) -> bool:
    """B2: ``--waves`` given, a matrix carrying design statuses / waves (any row whose status is not the bare ``launch``
    the published matrix implies, or whose wave is not ``published``), or a pinned / repeated gen root."""
    return (waves is not None or roots.declared
            or any(row_status(r) != STATUS_LAUNCH or row_wave(r) != WAVE_PUBLISHED for r in rows))


ROW_STATUS_REQUIRED, ROW_STATUS_NOT_ANALYSED, ROW_STATUS_ABSENT = "required", "not_analysed", "absent"


def design_block(rows: Sequence[Mapping[str, Any]], required: Sequence[Mapping[str, Any]], waves: Sequence[str] | None) -> dict[str, Any]:
    """``report["design"]``: the immutable panel design apart from the required-evidence set — every
    matrix row's status (``required`` / ``not_analysed`` / ``held`` / ``deferred`` with its reason), the analysed waves,
    the eight fixed arm slots per model for the PRIMARY row (the eight-arm figures draw the empty ones as explicit
    deferred / held slots), and the pooled panel's membership."""
    req = {row_leg_key(r) for r in required}
    status_by_row: dict[str, dict[str, Any]] = {}
    for r in rows:
        k = row_leg_key(r)
        if k in req:
            status, reason = ROW_STATUS_REQUIRED, None
        elif row_status(r) in EVIDENCE_STATUSES:
            status, reason = ROW_STATUS_NOT_ANALYSED, f"wave {row_wave(r)!r} is not among the analysed waves"
        else:
            status = row_status(r)
            reason = r.get("held_reason") or r.get("deferred_reason") or r.get("reason") or f"{status} row (no reason recorded)"
        rec: dict[str, Any] = {"status": status, "wave": row_wave(r), "reason": reason, "model_key": r["model_key"], "cell": r["cell"],
                               "arm": CELL_KIND[str(r["cell"])], "mode": r["sampling_mode"]}
        if r.get(ROW_EXPECTED_REFUSAL_KEY):
            # the row DECLARES its comparison's expected refusal (cells.py row 53: the O106 cross-engine ruling)
            rec[ROW_EXPECTED_REFUSAL_KEY] = str(r[ROW_EXPECTED_REFUSAL_KEY])
        if key_view(r) != VIEW_MAIN:
            # E3: a non-main view's row is filed apart under its own key, with its view and the it-doc carriage (object source, lane,
            # qualification verbatim); a main-view record — an external arm's included, whatever bank view it names — is unchanged
            rec["view"] = key_view(r)
            rec.update(itdoc_carriage(r))
        status_by_row[k] = rec
    models = list(dict.fromkeys(str(r["model_key"]) for r in rows))
    # math500-steer: a model's eight slots sit on ITS arm's primary row (`card` for the no-switch family; the thinking card elsewhere)
    primary_mode_by_model: dict[str, str] = {}
    for r in rows:
        primary_mode_by_model.setdefault(str(r["model_key"]), primary_mode_for(r))
    design_arm_list = design_arm_roster({str(r["cell"]) for r in rows})   # the declared roster + the arms these rows carry
    arms_by_model: dict[str, dict[str, str]] = {}
    for m in models:
        arms_by_model[m] = {}
        for arm in design_arm_list:
            rec = status_by_row.get(leg_key(m, arm.cell, primary_mode_by_model.get(m, MODE_PRIMARY)))
            arms_by_model[m][arm.kind] = rec["status"] if rec else ROW_STATUS_ABSENT
    # E3: the non-main views' PRIMARY-row slots per model, apart from the main view's eight (the eight-arm figures and the pooled
    # group read `arms_by_model` — the main view's — and nothing here): status, reason, wave and the it-doc carriage per arm present
    views = [v for v in dict.fromkeys(key_view(r) for r in rows) if v != VIEW_MAIN]
    arms_by_model_by_view: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for view in views:
        arms_by_model_by_view[view] = {}
        for m in models:
            per: dict[str, dict[str, Any]] = {}
            for arm in design_arm_list:
                vk = leg_key(m, arm.cell, primary_mode_by_model.get(m, MODE_PRIMARY), view)
                rec = status_by_row.get(vk)
                if rec is not None:
                    per[arm.kind] = {"leg": vk, "status": rec["status"], "reason": rec["reason"], "wave": rec["wave"], **itdoc_carriage(rec)}
            if per:
                arms_by_model_by_view[view][m] = per
    counts = Counter(v["status"] for v in status_by_row.values())
    return {"waves": (list(waves) if waves is not None else [WAVE_PUBLISHED]), "waves_in_matrix": sorted({row_wave(r) for r in rows}),
            "n_rows": len(rows), "n_required": counts.get(ROW_STATUS_REQUIRED, 0), "n_not_analysed": counts.get(ROW_STATUS_NOT_ANALYSED, 0),
            "n_held": counts.get(STATUS_HELD, 0), "n_deferred": counts.get(STATUS_DEFERRED, 0),
            # the GSM8K design's `n_expected_refusals`, mirrored: the REQUIRED rows that declare their comparison's expected refusal
            "n_expected_refusals": sum(1 for k, v in status_by_row.items() if v["status"] == ROW_STATUS_REQUIRED and v.get(ROW_EXPECTED_REFUSAL_KEY)),
            "required_legs": sorted(req), "status_by_row": status_by_row, "models": models,
            "arms": [a.cell for a in design_arm_list], "arm_kinds": [a.kind for a in design_arm_list], "arms_by_model": arms_by_model,
            "views_in_matrix": sorted({key_view(r) for r in rows}), "arms_by_model_by_view": arms_by_model_by_view,
            "view_rule": (f"arms_by_model is the {VIEW_MAIN!r} view's eight slots (the eight-arm figures and the pooled group read it); a "
                          "non-main view's rows (E3: the it-doc rows) are filed apart under arms_by_model_by_view[view][model][arm] with "
                          "their status, reason, wave, object source, lane and qualification, keyed <model>/<cell>/<mode>@<view> in "
                          "status_by_row, and their comparisons are anchored on the main view's bare leg (comparisons[*].bare_anchor)"),
            "panel": {"models": list(POOLED_PANEL_MODELS), "held_out": dict(POOLED_HELD_OUT), "not_in_panel": list(POOLED_NOT_IN_PANEL),
                      "mode": POOLED_MODE},
            "rule": ("the required set = the launch rows of the analysed waves, every one mandatory (a missing leg fails the population "
                     "gate); held / deferred rows and launch rows of other waves are statuses here and are never collected")}


def sampling_provenance(ctx: Context, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per model, the primary sampling row's ``source`` prose (the spec's sampling provenance: generation_config.json, the
    Nemotron 0.95 / grandfathered 1.0 note) — read off the pinned run config, never re-typed."""
    out: dict[str, Any] = {}
    for r in rows:
        key = str(r["model_key"])
        if key in out:
            continue
        table = ctx.sampling_rows.get(POOLED_MODE) or {}
        row = table.get(key)
        out[key] = (str(row.get("source")) if isinstance(row, Mapping) and row.get("source") else None)
    return out


def _branch_arrays(leg: Mapping[str, Any], branch: str, ids: Sequence[str]) -> tuple[Any, Any]:
    items = T._branch_items(leg, branch)
    mask = np.asarray([int(items[i][0]) for i in ids], dtype=np.int64)
    hit = mask * np.asarray([int(items[i][1]) if items[i][0] == 1 else 0 for i in ids], dtype=np.int64)
    return mask, hit


POOLED_ESTIMAND_TEXT = ("per arm, the EQUAL-WEIGHT mean (1/3 each) of the three panel models' conditional accuracies on that arm "
                        "(each leg's own scored denominator; the primary sampling row only), the same mean over the bare cells, and "
                        "their difference; B replicates, ONE item-resampling index vector per replicate applied to every model's leg "
                        "(pairing's draw order: a pure function of (seed, n, B), so the draws are the ones every contrast of this "
                        "report used); percentile CIs; a replicate in which any leg scored no item is discarded and counted; both "
                        "scoring branches; `unavailable` with the reason whenever any panel model's leg for the arm is missing, "
                        "refused, held, deferred or absent — never a mean over fewer models; llama held out (thinking: not "
                        "available), GLM not in the panel; a masked primary draw (D48) leaves its own model's arm↔bare pair only "
                        "(review H1: the shared resample stays over the common items; the reading with the draw is filed beside)")
#: review M4: why a pooled arm files NO directional word, stated on the record and in the captions
POOLED_WORD_RULE = ("no directional word is filed for a pooled arm: the verdict rule (D5 P2-2) is per model and needs both scoring "
                    "branches' CIs to exclude zero on the same side with the censoring bound; the pooled record carries both "
                    "branches (conditional accuracy on scored items, and score-as-is) so a censoring-driven sign disagreement is "
                    "SHOWN, never resolved by fiat")


def pooled_arm_bootstrap(arm_legs: Mapping[str, Mapping[str, Any]], bare_legs: Mapping[str, Mapping[str, Any]], *,
                         n_resamples: int, seed: int, ci: float,
                         exclude_by_model: Mapping[str, Sequence[str]] | None = None) -> dict[str, Any]:
    """:data:`POOLED_ESTIMAND_TEXT` as a function over ``{model: usable leg}`` for the arm and for bare. EXACT points
    (Fractions over the integer counts), the replicate draws shared across every model's leg (and, because the generator
    is re-seeded per call, across every arm and branch of the report). ``exclude_by_model`` (the mask scope rule,
    :data:`MASK_SCOPE_RULE`): items masked from THAT model's arm↔bare pair only — they leave the pair's numerators and
    denominators under both branches (as an unscored draw does under the primary branch) while the common resampled set
    stays the panel's ``n`` items; ``n_items_by_model`` and ``n_masked_by_model`` say so."""
    models = list(arm_legs)
    if not models or set(models) != set(bare_legs):
        raise AnalysisError("the pooled estimand needs the same models on the arm and on bare")
    gone_by_model = {m: sorted(set(_mapping(exclude_by_model).get(m, ()))) for m in models}
    ids = sorted(arm_legs[models[0]]["attempted_ids"])
    id_set = set(ids)
    for m in models:
        for leg in (arm_legs[m], bare_legs[m]):
            if sorted(leg["attempted_ids"]) != ids:
                raise AnalysisError(f"{leg['leg']}: its attempted ids are not the panel's common set — the pooled estimand is undefined")
        unknown = [i for i in gone_by_model[m] if i not in id_set]
        if unknown:
            raise AnalysisError(f"{m}: masked item(s) {unknown} are not in the panel's common set")
    n = len(ids)
    pos = {i: j for j, i in enumerate(ids)}
    weight = Fraction(1, len(models))
    n_masked_by_model = {m: len(v) for m, v in gone_by_model.items()}
    out: dict[str, Any] = {"models": models, "weights": {m: float(weight) for m in models}, "n_items": n,
                           "n_items_by_model": {m: n - n_masked_by_model[m] for m in models},
                           "n_masked_by_model": n_masked_by_model, "n_masked": sum(n_masked_by_model.values()),
                           "masked_ids": sorted(set().union(*(set(v) for v in gone_by_model.values()))),
                           "masked_ids_by_model": {m: v for m, v in gone_by_model.items() if v},
                           "B": int(n_resamples), "seed": int(seed), "level": float(ci)}

    def arrays(leg: Mapping[str, Any], branch: str, m: str) -> tuple[Any, Any]:
        mask, hit = _branch_arrays(leg, branch, ids)
        if gone_by_model[m]:
            mask, hit = mask.copy(), hit.copy()
            for i in gone_by_model[m]:
                mask[pos[i]] = 0
                hit[pos[i]] = 0
        return mask, hit

    for branch in BRANCHES:
        arrs = {m: (arrays(arm_legs[m], branch, m), arrays(bare_legs[m], branch, m)) for m in models}

        def acc(mask: Any, hit: Any) -> Fraction | None:
            s = int(mask.sum())
            return Fraction(int(hit.sum()), s) if s else None

        pts_arm = {m: acc(*arrs[m][0]) for m in models}
        pts_bare = {m: acc(*arrs[m][1]) for m in models}
        empty = [m for m in models if pts_arm[m] is None or pts_bare[m] is None]
        if empty:
            raise AnalysisError(f"{empty}: a leg scored no item under the {branch} branch — its conditional accuracy is undefined")
        point_arm = sum(pts_arm.values()) * weight
        point_bare = sum(pts_bare.values()) * weight
        rng = np.random.default_rng(seed)
        rep_arm, rep_bare = np.empty(n_resamples), np.empty(n_resamples)
        keep = np.ones(n_resamples, dtype=bool)
        chunk = max(1, 4_000_000 // n)
        done = 0
        while done < n_resamples:
            m_ = min(chunk, n_resamples - done)
            idx = rng.integers(0, n, size=(m_, n))
            sl = slice(done, done + m_)
            acc_arm, acc_bare = np.zeros(m_), np.zeros(m_)
            for m in models:
                (ma, ha), (mb, hb) = arrs[m]
                sa, sb = ma[idx].sum(axis=1), mb[idx].sum(axis=1)
                keep[sl] &= (sa > 0) & (sb > 0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    acc_arm = acc_arm + ha[idx].sum(axis=1) / sa
                    acc_bare = acc_bare + hb[idx].sum(axis=1) / sb
            rep_arm[sl], rep_bare[sl] = acc_arm / len(models), acc_bare / len(models)
            done += m_
        used = int(keep.sum())
        if used == 0:
            raise AnalysisError("every replicate had a leg with no scored item — the pooled estimand is not estimable")
        lo_a, hi_a = pr._quantiles(rep_arm[keep], ci)
        lo_b, hi_b = pr._quantiles(rep_bare[keep], ci)
        diff = (rep_arm - rep_bare)[keep]
        lo_d, hi_d = pr._quantiles(diff, ci)
        out[branch] = {"point": float(point_arm), "ci_low": lo_a, "ci_high": hi_a, "by_model": {m: float(pts_arm[m]) for m in models},
                       "bare": {"point": float(point_bare), "ci_low": lo_b, "ci_high": hi_b, "by_model": {m: float(pts_bare[m]) for m in models}},
                       "minus_bare": {"point": float(point_arm - point_bare), "ci_low": lo_d, "ci_high": hi_d,
                                      "excludes_zero": bool(lo_d > 0.0 or hi_d < 0.0),
                                      "replicate_signs": {"negative": int((diff < 0).sum()), "zero": int((diff == 0).sum()),
                                                          "positive": int((diff > 0).sum())}},
                       "n_replicates_used": used, "n_degenerate_replicates": int((~keep).sum())}
    return out


POOLED_AVAILABLE, POOLED_UNAVAILABLE = "available", "unavailable"


def eightarm_block(legs: Mapping[str, Mapping[str, Any]], design: Mapping[str, Any], *, n_resamples: int, seed: int,
                   ci: float, declared: bool = False, exclude_by_leg: Mapping[str, Sequence[str]] | None = None) -> dict[str, Any]:
    """``report["eightarm"]``: ``pooled_by_arm`` per arm kind over the PRIMARY row of the panel models
    {gemma, qwen, nemotron} — :func:`pooled_arm_bootstrap` when every panel model's leg for the arm (and bare) is usable
    AND each model's arm↔bare PAIR passes the same comparison policy :func:`run_comparison` applies
    (:func:`_comparable_problem`: equal attempted ids; the one-sha rule, or under a declared tree set the compatibility
    relations) — without that check a contrast the comparison had REFUSED (a moved chat template on one model's
    LoRA leg) still pools as ``available`` with a numeric effect and CI. Otherwise ``unavailable`` naming each model's
    status (missing / refused / held / deferred / absent, or ``incompatible with <model>/none — <why>``) with NO point or
    CI field — plus the panel membership, the hold-out and the estimand text. The figures DRAW these fields and compute
    nothing. ``declared``: whether a declared tree set admits cross-tree pairs (the run's ``ctx.declared``)."""
    status_by_row = _mapping(design.get("status_by_row"))

    def state(m: str, cell: str) -> tuple[Mapping[str, Any] | None, str | None]:
        k = leg_key(m, cell, POOLED_MODE)
        leg = legs.get(k)
        if leg is None:
            rec = status_by_row.get(k)
            return None, (rec["status"] if rec else ROW_STATUS_ABSENT)
        if leg.get("missing"):
            return None, f"missing ({leg['missing']})"
        if leg.get("refused"):
            return None, f"refused ({leg['refused']})"
        return leg, None

    bare: dict[str, Mapping[str, Any]] = {}
    bare_problems: list[str] = []
    for m in POOLED_PANEL_MODELS:
        leg, why = state(m, CELL_BARE)
        if leg is None:
            bare_problems.append(f"{m}/{CELL_BARE}: {why}")
        else:
            bare[m] = leg
    panel_arms = design_arms(design)            # the arms this analysis covers (design.arms), never the live registry
    pooled: dict[str, Any] = {}
    for arm in panel_arms:
        arm_legs: dict[str, Mapping[str, Any]] = {}
        problems = list(bare_problems)
        pairs: dict[str, dict[str, Any]] = {}
        for m in POOLED_PANEL_MODELS:
            leg, why = state(m, arm.cell)
            if leg is None:
                problems.append(f"{m}/{arm.cell}: {why}")
                continue
            if m not in bare:
                continue                    # the bare problem is already named
            # B3: the arm↔bare pair under the SAME policy every comparison runs through (bare↔bare is the identity pair)
            labels = {KIND_BARE: leg_key(m, CELL_BARE, POOLED_MODE), arm.kind: leg_key(m, arm.cell, POOLED_MODE)}
            why = _comparable_problem(legs, labels, what=f"{m}: the arm and bare legs'", declared=declared) if arm.cell != CELL_BARE else None
            pairs[m] = {"compatible": why is None, "reason": why,
                        "cross_tree": _mapping(leg.get("producer")).get("tree_sha") != _mapping(bare[m].get("producer")).get("tree_sha")}
            if why:
                problems.append(f"{m}/{arm.cell}: incompatible with {m}/{CELL_BARE} — {why}")
            else:
                arm_legs[m] = leg
        rec: dict[str, Any] = {"cell": arm.cell, "label": arm.label, "models": list(POOLED_PANEL_MODELS), "pairs": pairs,
                               "pair_rule": ("each panel model's arm↔bare pair passes the report's comparison policy "
                                             + ("(declared tree set: shared-field projection, arm contracts, server settings, membership)"
                                                if declared else "(one producer tree per pair)"))}
        if problems:
            rec.update(status=POOLED_UNAVAILABLE, reason="; ".join(problems))
        else:
            # under the mask's scope: the record consumes every panel model's arm leg and bare leg; a masked primary
            # leg's admitted items leave ITS model's arm↔bare pair only (the shared resample stays over the common set), and
            # the reading WITH the items is filed beside (`estimands_with_anomalies`) — the count per model is on the record
            ex = _mapping(exclude_by_leg)
            consumed = [leg_key(m, arm.cell, POOLED_MODE) for m in arm_legs] + [leg_key(m, CELL_BARE, POOLED_MODE) for m in bare]
            excl_by_model = {m: sorted(set(ex.get(leg_key(m, arm.cell, POOLED_MODE), ())) | set(ex.get(leg_key(m, CELL_BARE, POOLED_MODE), ())))
                             for m in arm_legs}
            excl_by_model = {m: v for m, v in excl_by_model.items() if v}
            rec.update(status=POOLED_AVAILABLE, reason=None, masked_items={k: list(ex[k]) for k in consumed if ex.get(k)},
                       **pooled_arm_bootstrap(arm_legs, bare, n_resamples=n_resamples, seed=seed, ci=ci, exclude_by_model=excl_by_model))
            if excl_by_model:
                with_items = pooled_arm_bootstrap(arm_legs, bare, n_resamples=n_resamples, seed=seed, ci=ci)
                rec["estimands_with_anomalies"] = {**{b: with_items[b] for b in BRANCHES}, "n_items": with_items["n_items"]}
                rec["mask_rule"] = MASK_SCOPE_RULE
            # review M4: the two scoring branches' pooled contrasts, and whether they agree in sign (a pooled arm files no
            # directional word — POOLED_WORD_RULE — so the disagreement is shown, never resolved by fiat)
            if arm.cell != CELL_BARE:
                p_, q_ = rec[BRANCH_PRIMARY]["minus_bare"], rec[BRANCH_SCORE_AS_IS]["minus_bare"]
                rec["branches_agree_in_sign"] = bool(((p_["point"] > 0) - (p_["point"] < 0)) == ((q_["point"] > 0) - (q_["point"] < 0)))
        pooled[arm.kind] = rec
    return {"mode": POOLED_MODE, "panel_models": list(POOLED_PANEL_MODELS), "held_out": dict(POOLED_HELD_OUT),
            "not_in_panel": list(POOLED_NOT_IN_PANEL), "arms": [a.kind for a in panel_arms], "pooled_by_arm": pooled,
            "n_available": sum(1 for r in pooled.values() if r["status"] == POOLED_AVAILABLE),
            "bootstrap": {"B": int(n_resamples), "seed": int(seed), "level": float(ci),
                          "shared_indices": "one index vector per replicate applied to every model's leg (rng.integers(0, n, size=(m, n)) "
                                            "in pairing's chunk order; a pure function of (seed, n, B))"},
            "estimand": POOLED_ESTIMAND_TEXT, "population_rule": MASK_SCOPE_RULE, "word_rule": POOLED_WORD_RULE,
            "headline_reading": (f"conditional accuracy on scored items. {SCORING_DEFINITION}; the unscored draws leave the "
                                 "denominator (each leg's own), and the score-as-is branch is filed beside the primary on every record")}


def eightarm_findings(block: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    for kind, rec in _mapping(block.get("pooled_by_arm")).items():
        if rec["status"] != POOLED_AVAILABLE:
            out.append(f"pooled ({'/'.join(block['panel_models'])}) {kind}: unavailable — {rec['reason']}")
            continue
        p = rec[BRANCH_PRIMARY]
        if kind == KIND_BARE:
            out.append(f"pooled ({'/'.join(block['panel_models'])}) bare accuracy {100 * p['point']:.1f} % [{100 * p['ci_low']:.1f}, {100 * p['ci_high']:.1f}]")
            continue
        d = p["minus_bare"]
        q = rec[BRANCH_SCORE_AS_IS]["minus_bare"]
        agree = "the branches agree in sign" if rec.get("branches_agree_in_sign") else "the branches DISAGREE in sign (censoring-driven)"
        out.append(f"pooled ({'/'.join(block['panel_models'])}) {kind} - bare = {100 * d['point']:+.1f} pp [{100 * d['ci_low']:+.1f}, "
                   f"{100 * d['ci_high']:+.1f}] conditional on scored items (primary branch) · score-as-is {100 * q['point']:+.1f} pp "
                   f"[{100 * q['ci_low']:+.1f}, {100 * q['ci_high']:+.1f}]; {agree}; equal weights; llama held out; no pooled verdict word "
                   "(POOLED_WORD_RULE)")
    return out


ESTIMAND_TEXT = ("per (model, sampling row): conditional accuracy with each cell's OWN scored denominator over the ONE "
                 f"common attempted set of 500 MATH-500 items (k = 1, thinking ON, cap {CAP}); object_minus_bare = "
                 "acc(object) - acc(bare), prompted_minus_bare = acc(prompted) - acc(bare); B replicates, one index draw per "
                 "replicate applied to the three cells carrying each cell's mask; percentile CIs; sharp censoring bounds from "
                 "cell-wise adverse imputation of the unscored items; both scoring branches (primary: unterminated -> unscored; "
                 "score-as-is: unterminated -> wrong). A sensitivity row is its OWN comparison plus per-cell paired deltas to the "
                 "primary row over the same 500 items — never pooled.")


def analyze(gen_root: Path | str | GenRoots | Sequence[str], sha: str, *, cells: Sequence[Mapping[str, Any]] | Path | str,
            sampling_rows: Mapping[str, Any] | Path | str, render_facts: Mapping[str, Any] | Path | str | None = None,
            math500_ids: Sequence[str] | None = None, object_pins: Mapping[Any, str] | Path | str | None = None,
            model_pins: Mapping[str, tuple[str, str]] | None = None, n_resamples: int = pr.DEFAULT_BOOTSTRAP,
            seed: int = pr.DEFAULT_BOOTSTRAP_SEED, ci: float = pr.DEFAULT_CI, allow_pending_gates: bool = False,
            read_eval_logs: bool = True, analysis_sha: str | None = None, repo_root: Path | None = None,
            tree_set: Mapping[str, Any] | None = None, waves: Sequence[str] | None = None,
            anomaly_classes: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
            store_pins: Path | str | Mapping[str, str] | Sequence[Path | str | Mapping[str, str]] | None = None) -> dict[str, Any]:
    """Collect every leg of the matrix -> comparisons per (model, mode) -> the
    estimand under both branches with the verdict rule on the primary rows ->
    the mode deltas -> the gates -> the filed verdict. No estimand runs for a
    comparison with an unusable leg; no word is filed unless every comparison
    AND every required mode delta ran and every gate passes (a pending gate
    withholds it, named; ``allow_pending_gates`` waives ONLY the producer
    stamps in :data:`WAIVABLE_PENDING_STAMPS`). ``tree_set`` is a DECLARATION
: ``also`` (the further producer trees), ``allow`` (the path globs),
    ``why`` (the reason), optionally ``primary_pin`` (must be ``sha``) — the
    block :func:`verify_tree_set` returns carries these and may be passed back.
    The verification runs HERE, on ``repo_root``, from those fields alone
    before any cell is loaded; a caller's ``problems: []`` is never trusted.
    No ``also`` tree (or ``None``) is the one-tree rule, and the report, the
    legs and the table are then exactly the one-tree ones."""
    repo_root = Path(repo_root) if repo_root is not None else _HERE.parent
    decl = _mapping(tree_set)
    # math500-steer: the per-LEG store pins — every leg admitted by ITS OWN producer tree from the pins doc; --sha is
    # then the record label and the published legs' pin. Pins and a declared tree set are two admissions of the same legs: one
    pins_by_leg = load_store_pins(store_pins)
    if pins_by_leg is not None and decl.get("also"):
        raise AnalysisError("--store-pins and --also-sha are two admissions of the same legs — give one")
    if decl.get("primary_pin", sha) != sha:
        raise AnalysisError(f"{TREE_SET_KEY}: the declaration's primary pin {decl.get('primary_pin')!r} is not --sha {sha!r}")
    ts: dict[str, Any] | None = None
    if decl.get("also"):
        ts = verify_tree_set(sha, [str(x) for x in decl["also"]], [str(g) for g in decl.get("allow") or ()],
                             repo_root=repo_root, reason=decl.get("why"), allow_file=decl.get("allow_file"),
                             allow_evidence=_mapping(decl.get("allow_evidence")) or None)
        if ts["problems"]:
            raise AnalysisError(f"{TREE_SET_KEY}: " + "; ".join(ts["problems"]))
    rows = load_cells(cells)
    # the trees a PINNED gen root (PATH=PIN) may name: the declared tree set's, or under per-leg pins every pinned tree of the doc
    admitted = tuple(ts["also"]) if ts else tuple(sorted(set(pins_by_leg.values()))) if pins_by_leg is not None else ()
    # B6: the gen roots — a Path (the published call), a GenRoots, or `PATH[=PIN]` specs (the repeatable CLI form)
    roots = GenRoots.of(gen_root, primary_pin=sha, admitted=admitted)
    # the required-evidence set and the eight-arm mode
    required = required_rows(rows, waves)
    eight_arm = eight_arm_mode(rows, waves, roots)
    if not required:
        raise AnalysisError(f"no launch row of the analysed waves {list(waves) if waves is not None else '(all)'} — nothing to collect "
                            f"(the matrix's waves: {sorted({r['wave'] for r in rows})})")
    if object_pins is None:
        object_pins = repo_root / "configs" / "sdfbase_full22_pins.json"
    served_pins = dict(model_pins or {})
    for r in required:      # the battery config's served pin for every model the matrix names, unless supplied
        key = str(r["model_key"])
        if key not in served_pins:
            served_pins[key] = model_pin(repo_root, key)
    # F11b: the object pins are keyed (model, arm) — read under every fitted required row's staged id (eight-arm mode), or
    # the published way (the minus_ctl object per target) when the matrix is the published design
    if isinstance(object_pins, (str, Path)):
        pins = load_object_pins(object_pins, required) if eight_arm else load_object_pins(object_pins)
    else:
        pins = normalise_object_pins(object_pins)
    ctx = Context(sampling_rows=load_sampling_rows(sampling_rows), render_facts=load_render_facts(render_facts),
                  object_pins=pins, model_pins=served_pins, store_pin=sha, also_trees=(tuple(ts["also"]) if ts else ()),
                  declared=bool(ts) or roots.declared or pins_by_leg is not None, store_pins=pins_by_leg,
                  # classes for a run without logs come from the report it must match; a log-reading run takes its own
                  anomaly_classes=(anomaly_classes if not read_eval_logs else None))
    legs = {row_leg_key(r): collect_leg(roots, r, ctx=ctx, read_eval_logs=read_eval_logs) for r in required}
    if ts is not None:
        ts["legs"] = {k: leg.get("producer_tree") for k, leg in legs.items()}      # The per-leg tree assignment
    design = design_block(rows, required, waves)
    comps = [run_comparison(c, legs, n_resamples=n_resamples, seed=seed, ci=ci, declared=ctx.declared)
             for c in comparisons_from(required, eight_arm=eight_arm)]
    code = T.analysis_code_provenance(analysis_sha)
    report: dict[str, Any] = {
        "experiment": EXPERIMENT, "task": TASK, "sha": sha, "analysis_code_sha": code["sha"], "analysis_code": code,
        "gen_root": str(roots.primary), "estimand": ESTIMAND_TEXT, "bootstrap": {"B": n_resamples, "seed": seed, "level": ci},
        # the models and families THIS matrix runs: the shared sampling-row and render-fact tables may carry a model / family
        # (llama8b's card row; the llama no-switch fact) that the published 24-row MATH-500 matrix never ran, and the published
        # report keeps its frozen shape when the shared tables grow
        "cells": rows,
        "sampling_rows": {mode: {k: v for k, v in _mapping(table).items() if k in {str(r["model_key"]) for r in rows}}
                          for mode, table in ctx.sampling_rows.items()},
        "render_facts": {fam: {k: f.get(k) for k in ("split_mode", "server_parser", "evidence_status", "off_ids", "on_ids")}
                         for fam, f in ctx.render_facts.items() if fam in {str(r["family"]) for r in rows}},
        "legs": legs, "unusable_legs": unusable_legs(legs), "comparisons": comps,
    }
    if ts is not None:
        report[TREE_SET_KEY] = ts
    if pins_by_leg is not None:
        # per-leg admission (--store-pins): the doc as read, and every leg's admitted pin beside its producer tree
        report[STORE_PINS_KEY] = {"admission": "per-leg pins (--store-pins): every leg admitted by ITS OWN producer tree",
                                  "pins": dict(sorted(pins_by_leg.items())),
                                  "legs": {k: {"admitted_pin": _mapping(lg.get("producer")).get("admitted_pin") or lg.get("admitted_pin"),
                                               "tree_sha": _mapping(lg.get("producer")).get("tree_sha"),
                                               "usable": lg.get("refused") is None and lg.get("missing") is None}
                                           for k, lg in legs.items()},
                                  "trees": sorted({_mapping(lg.get("producer")).get("tree_sha") for lg in legs.values()
                                                   if lg.get("refused") is None and lg.get("missing") is None
                                                   and _mapping(lg.get("producer")).get("tree_sha")})}
    if roots.declared or pins_by_leg is not None:
        report["gen_roots"] = roots.record()
    if eight_arm:
        report["design"] = design
    ran = [c for c in comps if c["estimands"] is not None]
    if ran:
        report["estimands"] = {}
        for c in ran:
            report["estimands"].setdefault(c["model_key"], {})[comp_slot(c)] = c["estimands"]
    deltas: dict[str, dict[str, Any]] = {}
    delta_records: list[dict[str, Any]] = []
    for c in comps:
        if c["primary"] or comp_view(c) != VIEW_MAIN:
            # E3: a mode delta pairs a sensitivity row with the primary row of the SAME view; a non-main view has no delta here
            continue
        rec = run_mode_delta(c["model_key"], c["mode"], legs, n_resamples=n_resamples, seed=seed, ci=ci,
                             kinds=delta_kinds(c["model_key"], c["mode"], legs, eight_arm=eight_arm), declared=ctx.declared)
        delta_records.append(rec)
        if rec["estimands"] is not None:
            deltas.setdefault(c["model_key"], {})[c["mode"]] = rec["estimands"]
    report["mode_deltas"] = deltas
    report["mode_delta_records"] = delta_records
    # every mode delta the matrix requires is part of production completeness (the gate, the verdict, the refusals) —
    # a refused delta is a missing estimand, never a silent omission from `mode_deltas`
    gates, findings = all_gates(legs, comps, math500_ids=math500_ids, n_resamples=n_resamples, seed=seed, ci=ci,
                                deltas=delta_records)
    report["gates"] = gates
    # the split-anomaly gate's ADMITTED primary-leg masks are applied to the FILED records — every comparison and mode
    # delta that consumes a masked leg is re-run without the items (the very re-run the gate judged), the WITH reading kept
    # beside it, the masked items and their count on the record; the pooled arms below take the same exclusion
    exclude_by_leg: dict[str, list[str]] = {k: [str(it["item_id"]) for it in v["items"]]
                                            for k, v in _mapping(_mapping(gates.get("split_anomalies")).get("primary_masked")).items()}
    if exclude_by_leg:
        apply_primary_masks(comps, delta_records, legs, exclude_by_leg, n_resamples=n_resamples, seed=seed, ci=ci)
        report["estimands"] = {}
        for c in ran:
            report["estimands"].setdefault(c["model_key"], {})[comp_slot(c)] = c["estimands"]
        deltas = {}
        for rec in delta_records:
            if rec["estimands"] is not None:
                deltas.setdefault(rec["model_key"], {})[rec["mode"]] = rec["estimands"]
        report["mode_deltas"] = deltas
        report["masked_primary"] = {"by_leg": exclude_by_leg, "n_masked": sum(len(v) for v in exclude_by_leg.values()),
                                    "by_contrast": {c["name"]: sorted(c["masked_contrasts"]) for c in comps if c.get("masked_contrasts")},
                                    "rule": MASK_SCOPE_RULE}
    if eight_arm:
        # the pooled-by-arm estimand as REPORT FIELDS over the primary row of the panel models (the figures draw them)
        report["eightarm"] = eightarm_block(legs, design, n_resamples=n_resamples, seed=seed, ci=ci, declared=ctx.declared,
                                            exclude_by_leg=exclude_by_leg)
        findings.extend(eightarm_findings(report["eightarm"]))
        for k, rec in design["status_by_row"].items():
            if rec["status"] in (STATUS_HELD, STATUS_DEFERRED):
                findings.append(f"{k}: {rec['status'].upper()} — {rec['reason']} (a design status; never collected)")
    if ts is not None:
        findings.append(tree_set_note(ts))
    for k, leg in legs.items():
        n_incomplete = (leg.get("producer") or {}).get("n_incomplete_runs")
        if n_incomplete:
            findings.append(f"{k}: {n_incomplete} recorded run(s) of the store did not complete (a preflight that refused; its "
                            "labelled probe attempts stay on record — attributable, not a failure)")
        n_unattributed = (leg.get("request_evidence") or {}).get("n_probe_records_unattributed")
        if n_unattributed:
            findings.append(f"{k}: {n_unattributed} probe request-evidence record(s) beyond what the recorded runs account for "
                            "(counted, not a failure: every record is still an identity-bound attempt held to no declaration)")
    # the at-cap readout as a RESULT block and the no-reasoning shares, over the usable legs only; each cell over the
    # 1 % line and each nonzero no-reasoning share is a finding, the failed premise the headline
    tr = truncation_block(legs, required)
    report["truncation"] = tr
    for c in tr["cells"]:
        tag = " [sensitivity row]" if c["sensitivity"] else ""
        if c["line_1pct_exceeded"]:
            findings.append(f"{c['leg']}: at-cap share {100 * c['share']:.1f} % > the 1 % line ({c['n_at_cap']} of "
                            f"{c['n_with_stop_reason']} draws; loop {c['n_loop']} / truncated reasoning {c['n_truncated_reasoning']} / "
                            f"text unavailable {c['n_text_unavailable']}; split from the {_split_source_words(c['split_source'])}){tag}")
        if c["n_no_reasoning_emitted"]:
            findings.append(f"{c['leg']}: {c['n_no_reasoning_emitted']} of {c['n_on_rows']} thinking-ON draws finished with no "
                            f"reasoning emitted ({100 * c['share_no_reasoning']:.1f} %) — a model choice on the draw (D115), graded "
                            f"from content{tag}")
        if c["n_present_by_channel"]:
            findings.append(f"{c['leg']}: reasoning present on {c['n_present_by_channel'] + c['n_present_by_record']} of "
                            f"{c['n_on_rows']} thinking-ON draws — {c['n_present_by_channel']} by the reasoning channel "
                            f"(inspect_ai lifted a complete <think>…</think> block out of the text before the scorer's tag test; "
                            f"the record's think_block_present is false on them) + {c['n_present_by_record']} by the record (a tag "
                            f"in the text); the record's flag alone would have read {c['n_present_by_channel']} draw(s) as no "
                            f"reasoning — a readout fact about the client parser, not a model choice{tag}")
        if c["n_items_redrawn"]:
            findings.append(f"{c['leg']}: {c['n_redraw_calls']} same-cap re-send(s) over {c['n_items_redrawn']} item(s) "
                            f"({100 * c['share_items_redrawn']:.1f} % of items; max {c['max_calls_per_item']} calls on one item; "
                            f"{c['n_items_sample_error']} item(s) with a failed sample attempt) — the harness's client retry after a "
                            f"request it gave up on; the at-cap share is over final draws (see truncation.redraw_selection_caveat){tag}")
    if tr["headline"]:
        findings.append(tr["headline"])
    report["controls"] = controls_table(legs)
    report["provenance"] = {"store_pin": sha, "analysis_code_sha": code["sha"], "analysis_code": code,
                            "note": "three kinds of sha, kept apart: the store pin (every cell's wave sha), the analysis code's "
                                    "tree, and each cell's own producer / config / intervention hashes",
                            # the spec's provenance sentences (strings — B1): the trigger this instrument reports under and the
                            # sampling source per model, read off the pinned run config
                            "trigger": M5T_TRIGGER_PROVENANCE, "sampling": sampling_provenance(ctx, required),
                            **({"waves": design["waves"], "gen_roots": roots.record()} if eight_arm else {}),
                            "cells": {k: provenance_row(leg) for k, leg in legs.items()}}
    words: dict[str, dict[str, str]] = {}
    statements: dict[str, dict[str, str]] = {}
    # E3: a non-main view's words are filed APART (view -> model -> contrast), beside the qualification every it-doc reading carries —
    # never folded into the model's filed verdict, which stays the main view's
    view_words: dict[str, dict[str, dict[str, str]]] = {}
    view_statements: dict[str, dict[str, dict[str, str]]] = {}
    for c in ran:
        if not c["primary"]:
            continue
        if comp_view(c) == VIEW_MAIN:
            words[c["model_key"]] = {k: v["word"] for k, v in c["verdicts"].items()}
            statements[c["model_key"]] = {k: v["statement"] for k, v in c["verdicts"].items()}
        else:
            view_words.setdefault(comp_view(c), {})[c["model_key"]] = {k: v["word"] for k, v in c["verdicts"].items()}
            view_statements.setdefault(comp_view(c), {})[c["model_key"]] = {k: v["statement"] for k, v in c["verdicts"].items()}
    for c in comps:
        if comp_view(c) != VIEW_MAIN:
            anchor = c["bare_anchor"]
            arms = ", ".join(k for k in c["leg_keys"] if k != KIND_BARE)
            quals = list(dict.fromkeys(qualification_text(rec) or "no qualification recorded on the row" for rec in c["itdoc_arms"].values()))
            lanes = sorted({str(rec.get("lane")) for rec in c["itdoc_arms"].values()})
            findings.append(f"{c['name']}: the {c['view']!r} view's arms ({arms}) are contrasted against the {anchor['view']}-view bare leg "
                            f"{anchor['leg']} — {anchor['rationale']}"
                            + (f"; it-doc re-fit (lane {', '.join(lanes)}): {' / '.join(quals)}" if c["itdoc_arms"] else "")
                            + (" — " + ", ".join(f"{k} {v['word']}" for k, v in c["verdicts"].items()) if c.get("verdicts") else ""))
        if not c["usable"]:
            if c.get(REFUSED_BY_DESIGN_KEY):
                findings.append(f"{c['name']}: refused by design ({c[ROW_EXPECTED_REFUSAL_KEY]}) — the arm's numbers are reported as the "
                                f"arm's numbers, the pairing's absence accepted — {c['reason']}")
            else:
                findings.append(f"{c['name']}: no estimand — {c['reason']}")
    for d in delta_records:
        if not d["usable"]:
            findings.append(f"{d['name']}: no estimand — {d['reason']}")
    for m, ws in words.items():
        for k, w in ws.items():
            findings.append(f"{m}: {k} {w}")
    missing_comps = comparisons_not_run(comps)
    missing_deltas = [d["name"] for d in delta_records if d["estimands"] is None]
    if missing_comps or missing_deltas:
        parts = ([f"comparison(s) did not run: {missing_comps}"] if missing_comps else []) + \
            ([f"required mode delta(s) did not run: {missing_deltas}"] if missing_deltas else [])
        verdict = {"word": None, "verdict": NOT_MEASURABLE, "reason": "; ".join(parts)}
    elif gates["failed"]:
        verdict = {"word": None, "verdict": NOT_MEASURABLE,
                   "reason": "gate(s) FAILED: " + "; ".join(f"{g}: {gates[g]['detail']}" for g in gates["failed"])}
    elif gates["pending"] and (not allow_pending_gates or gates["pending_unwaivable"]):
        verdict = {"word": None, "verdict": NOT_MEASURABLE + " (stamps pending)",
                   "reason": "gate(s) PENDING: " + "; ".join(f"{g}: {gates[g].get('detail')}" for g in gates["pending"]),
                   "pending_unwaivable": gates["pending_unwaivable"]}
    else:
        verdict = {"word": "filed", "verdict": "; ".join(f"{m}: " + ", ".join(f"{k} {w}" for k, w in ws.items())
                                                        for m, ws in words.items()),
                   "reason": None, "gates_pending": gates["pending"]}
    verdict["words"] = words
    verdict["statements"] = statements
    if view_words:
        verdict["view_words"] = view_words
        verdict["view_statements"] = view_statements
        verdict["view_rule"] = ("a non-main view's words (E3: the it-doc re-fit arms) are filed here, apart from the main view's verdict, "
                                "each comparison's qualification beside it on comparisons[*].itdoc_arms and in the findings")
    report["verdict"] = verdict
    report["findings"] = findings
    report["top_items"] = ([tr["headline"]] if tr["headline"] else []) + \
        [f for f in findings if "no estimand" in f or "FAILED" in f] + \
        [f"{m}: {k} {w}" for m, ws in words.items() for k, w in ws.items() if w in WORDS]
    return report


# --------------------------------------------------------------------------- #
# figures (dataviz skill: the parent's house palette; a title of <= 2 lines,  #
# no in-figure prose, axes >= 80 % of the figure height; captions in the JSON) #
# --------------------------------------------------------------------------- #

SURFACE, INK, INK2, MUTED, GRID, BASELINE, CAT = T.SURFACE, T.INK, T.INK2, T.MUTED, T.GRID, T.BASELINE, T.CAT
#: the dataviz reference palette's slots 5-8 beside the parent's four (palette.md: magenta, green, violet, red)
CAT8: Mapping[str, str] = {**CAT, "magenta": "#e87ba4", "green": "#008300", "violet": "#4a3aa7", "red": "#e34948"}
#: hue = ARM (fixed per entity, never cycled — the dataviz rule): the published three keep their published slots (bare blue,
#: object orange, prompted aqua); the five further arms take the palette's remaining slots in panel order
#: the palette's eight categorical slots are spent by the published eight; the appended all-positions Arditi arm takes a NINTH
#: hue outside them (a brown — no published slot is brown, and it is far from `arditi`'s red on purpose: the two arms must never
#: read as one family of bars). Fixed per arm, never shared.
CAT_NINTH_BROWN = "#8a5a2b"
KIND_COLOR: Mapping[str, str] = {KIND_BARE: CAT["blue"], KIND_OBJECT: CAT["orange"], KIND_PROMPTED: CAT["aqua"],
                                 KIND_PROMPTED_CONST: CAT8["yellow"], KIND_LORA_CTL2: CAT8["magenta"], KIND_LORA_MIX: CAT8["green"],
                                 KIND_ALPHASTEER: CAT8["violet"], KIND_ARDITI: CAT8["red"], KIND_ARDITI_ALLPOS: CAT_NINTH_BROWN}
KIND_LABEL: Mapping[str, str] = {a.kind: a.label for a in ARMS}
MODE_MARKER: Mapping[str, str] = {MODE_PRIMARY: "o", MODE_GENERAL: "^", MODE_BATTERY: "s"}
TITLE_MAX_LINES, TITLE_MAX_CHARS, AXES_MIN_FRACTION = 2, 90, 0.80


def check_title(text: str) -> str:
    """A figure title is at most two lines of <= 90 characters (the rule:
    longer means it is not a title; the prose goes in the caption)."""
    lines = text.split("\n")
    if len(lines) > TITLE_MAX_LINES or any(len(line) > TITLE_MAX_CHARS for line in lines):
        raise AnalysisError(f"figure title exceeds {TITLE_MAX_LINES} lines of {TITLE_MAX_CHARS} chars: {text!r}")
    return text


def axes_height_fraction(fig) -> float:
    """The fraction of the figure height the axes span (bottom of the lowest to
    the top of the highest); every figure keeps it >= :data:`AXES_MIN_FRACTION`."""
    boxes = [ax.get_position() for ax in fig.axes]
    if not boxes:
        return 0.0
    return max(b.y1 for b in boxes) - min(b.y0 for b in boxes)


def _title(fig, text: str) -> None:
    fig.text(0.02, 0.992, check_title(text), fontsize=10.5, fontweight="bold", color=INK, ha="left", va="top")


def _layout(fig, *, left: float = 0.09, right: float = 0.98, **kw: Any) -> None:
    fig.subplots_adjust(left=left, right=right, top=0.905, bottom=0.095, **kw)


def _label(ax, text: str) -> None:
    ax.text(0.01, 0.985, text, transform=ax.transAxes, ha="left", va="top", fontsize=8.2, color=INK)


TITLE_MAX_WORDS = 2


def figure_title_problem(fig) -> str | None:
    """The title rule: the FIRST title line IS the title — at most :data:`TITLE_MAX_WORDS` words; the
    reading goes in the second line and the prose in the caption. ``None`` when the one bold title text obeys it."""
    bold = [t for t in fig.texts if t.get_fontweight() == "bold"]
    if len(bold) != 1:
        return f"expected exactly one bold title text, found {len(bold)}"
    first = bold[0].get_text().split("\n")[0]
    words = first.split()
    if len(words) > TITLE_MAX_WORDS:
        return f"title line {first!r} has {len(words)} words (> {TITLE_MAX_WORDS}): the prose belongs in the subtitle and the caption"
    return None


def _footnotes_below(fig, ax, lines: Sequence[str], *, clearance_in: float = 0.55, step_in: float = 0.09, fontsize: float = 5.2) -> None:
    """Review L6 / M3: the disclosure lines UNDER the axes — below the tick and axis labels, right-aligned to the axes — in
    axes coordinates derived from the axes' height in inches, so the offset on the page is the same whatever the figure's
    height; the saver's tight bounding box keeps them in the file and the axes' share of the figure (the >= 80 % rule) is
    unchanged."""
    if not lines:
        return
    h_in = ax.get_position().height * fig.get_figheight()
    for j, line in enumerate(lines):
        ax.text(0.995, -(clearance_in + j * step_in) / h_in, line, transform=ax.transAxes, ha="right", va="top", fontsize=fontsize,
                color=INK2, clip_on=False)


def _ctl_index(report: Mapping[str, Any]) -> dict[str, int]:
    return {row["leg"]: i for i, row in enumerate(report["controls"])}


def figure_excluded_families(report: Mapping[str, Any]) -> frozenset[str]:
    """The model families the figures and the rendered tables leave out —
    ``figures.excluded_families`` of the report (:func:`make_figures` draws from
    a view that carries it); empty when nothing is excluded."""
    return frozenset(_mapping(report.get("figures")).get(FIGURES_EXCLUDED_KEY) or ())


#: the two further PRESENTATION selectors (plan P5): ``figures.excluded_arms`` — matrix CELL ids left out of every
#: figure (``--figure-exclude-arm``, validated against the report's cells) — and ``figures.modes`` — the sampling modes DRAWN
#: (``--figure-modes``; None = every mode). Like the family exclusion they select the rows SHOWN and recompute no aggregate.
FIGURES_EXCLUDED_ARMS_KEY = "excluded_arms"
FIGURES_MODES_KEY = "modes"
FIGURES_SELECTION_KEY = "selection"
#: ``figures.shown_only`` — the fourth presentation switch: under a family exclusion or an arm / mode selection, the
#: three figures that describe the SET drawn (the at-cap share, the loop split, the no-reasoning share) carry the
#: shown set's descriptor ONLY, labelled as shown; every legend and caption describes the rows DRAWN; the numbers
#: sidecar carries no report-wide ("wave") aggregate over rows the render does not draw; the tables JSON is scoped
#: to the shown rows (:func:`scope_tables_to_shown`); a one-row render skips the sampling-row figure; and the render
#: refuses if either sidecar still names a hidden row (:func:`shown_only_leaks`). Off by default and recorded ONLY
#: when set, so a render without it — the shared eight-arm analyses' — is byte-identical to the pre-flag shape;
#: without a selector it is refused (nothing is hidden, so there is no shown set to describe apart from the report's).
FIGURES_SHOWN_ONLY_KEY = "shown_only"


def psr_cells_of(cells: Sequence[str]) -> list[str]:
    """The PSR cells among ``cells`` (the gate's own predicate, :func:`PG.is_psr_cell`): out of EVERY figure by rule —
    the two analyses' ``validated_selection`` fold them into the excluded arms of every render, selector or no selector,
    so the numbers sidecar states them and the gate reads them."""
    return sorted(c for c in dict.fromkeys(str(c) for c in cells) if PG.is_psr_cell(c))


def figure_excluded_arms(report: Mapping[str, Any]) -> frozenset[str]:
    return frozenset(_mapping(report.get("figures")).get(FIGURES_EXCLUDED_ARMS_KEY) or ())


def figure_modes(report: Mapping[str, Any]) -> frozenset[str] | None:
    modes = _mapping(report.get("figures")).get(FIGURES_MODES_KEY)
    return None if modes is None else frozenset(str(m) for m in modes)


def figure_selection_active(report: Mapping[str, Any]) -> bool:
    """Whether the arm or the mode selector is in play (the family exclusion alone keeps the published title lines, B11)."""
    return bool(figure_excluded_arms(report)) or figure_modes(report) is not None


def figure_shown_only(report: Mapping[str, Any]) -> bool:
    """Whether the render describes the shown set only (:data:`FIGURES_SHOWN_ONLY_KEY`): no wave-wide value in a title, a legend
    or the numbers sidecar."""
    return bool(_mapping(report.get("figures")).get(FIGURES_SHOWN_ONLY_KEY))


def _row_shown(report: Mapping[str, Any], row: Mapping[str, Any]) -> bool:
    """A row (a controls / truncation / matrix row: ``model_key``, ``cell``, ``mode``) passes all three selectors."""
    if row.get("model_key") in figure_excluded_families(report):
        return False
    if row.get("cell") in figure_excluded_arms(report):
        return False
    modes = figure_modes(report)
    return modes is None or row.get("mode") is None or row.get("mode") in modes


def _shown_rows(report: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """The rows every selector lets through (family, arm, mode); every row keeps its
    report position (the builders index the controls by ``leg``), so every
    sidecar path still resolves in the full report."""
    return [r for r in rows if _row_shown(report, r)]


def _key_shown(report: Mapping[str, Any], key: str) -> bool:
    """Whether a string KEY of a report block names a row the selection shows: a model key (hidden when the family is
    excluded), a leg id ``model/cell/mode[@view]`` (all three selectors), or a ``model/mode[@view]`` pair (a comparison /
    mode-delta / served-geometry key); any other string is not a row and passes. The shown_only scoping of the tables JSON
    and the caption disclosures read keys through this one predicate. A non-main view's key appends ``@<view>`` to the mode
    segment (E3); the selectors read (family, arm, mode) the way :func:`_row_shown` does, whatever the view."""
    models = set(_report_models(report))
    if key in models:
        return key not in figure_excluded_families(report)
    parts = key.split("/")
    if len(parts) in (2, 3) and parts[0] in models:
        parts[-1] = parts[-1].split(LEG_KEY_VIEW_SEP, 1)[0]
    if len(parts) == 3 and parts[0] in models:
        return _row_shown(report, {"model_key": parts[0], "cell": parts[1], "mode": parts[2]})
    if len(parts) == 2 and parts[0] in models:
        return parts[0] not in figure_excluded_families(report) and _mode_shown(report, parts[1])
    return True


def _shown_only_keys_filter(report: Mapping[str, Any]) -> Callable[[str], bool]:
    """Under shown_only the caption / footnote disclosures are read over the rows SHOWN (family, arm AND mode); otherwise
    over the families shown, as before (the default render and the labelled-exclusion render are untouched)."""
    if figure_shown_only(report):
        return lambda k: _key_shown(report, str(k))
    excluded = figure_excluded_families(report)
    return lambda k: str(k).split("/")[0] not in excluded


def _shown_kinds(report: Mapping[str, Any], kinds: Sequence[str]) -> list[str]:
    """The arm kinds among ``kinds`` whose cell is not an excluded arm, in PANEL order."""
    excluded = figure_excluded_arms(report)
    present = set(kinds)
    return [k for k in ARM_KINDS if k in present and ARM_BY_KIND[k].cell not in excluded]


def _mode_shown(report: Mapping[str, Any], mode: str) -> bool:
    modes = figure_modes(report)
    return modes is None or mode in modes


def _selection_line(report: Mapping[str, Any]) -> str | None:
    """The title clause every figure carries under an arm / mode selection: the full-report leg count and the count
    shown — never a recomputed aggregate (plan P5). ``None`` when neither selector is active."""
    if not figure_selection_active(report):
        return None
    rows = _leg_rows(report)
    return f"{len(rows)} legs in report · {len(_shown_rows(report, rows))} shown"


def _with_selection(report: Mapping[str, Any], line: str) -> str:
    """A figure's second title line with the selection clause appended: the clause is never cut — the reading before it
    is shortened to make room (the line stays within :data:`TITLE_MAX_CHARS`)."""
    sel = _selection_line(report)
    if sel is None:
        return _fit_title_line(line)
    room = TITLE_MAX_CHARS - len(sel) - 3
    head = line if len(line) <= room else line[:max(room - 1, 0)].rstrip() + "…"
    return f"{head} · {sel}"


def _selection_suffix(report: Mapping[str, Any]) -> str:
    """The selection clause as a figure's OWN second title line (``"\\n<clause>"``), ``""`` when neither selector is
    active — for the figures whose first line is the whole published title."""
    sel = _selection_line(report)
    return f"\n{sel}" if sel else ""


def _shown_models(report: Mapping[str, Any], models: Sequence[str]) -> list[str]:
    excluded = figure_excluded_families(report)
    return [m for m in models if m not in excluded]


def _families_text(n: int) -> str:
    return f"{n} famil{'y' if n == 1 else 'ies'} shown"


def _report_models(report: Mapping[str, Any]) -> list[str]:
    return list(dict.fromkeys(row["model_key"] for row in report["controls"]))


def _leg_short(leg: Mapping[str, Any]) -> str:
    mode = "" if leg["mode"] == MODE_PRIMARY else f" [{leg['mode']}]"
    return f"{leg['model_key']} {leg['kind']}{_view_tag(str(leg.get('view') or VIEW_MAIN))}{mode}"


def _pp(x: float) -> float:
    return 100.0 * x


def fig_accuracy_by_cell(report: Mapping[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    plt, _mcolors, Patch = T._mpl()
    numbers: list[dict[str, Any]] = []
    models = _shown_models(report, [m for m in report["estimands"] if MODE_PRIMARY in report["estimands"][m]])
    fig, axes = plt.subplots(1, len(models), figsize=(2.2 * len(models) + 1.2, 4.2), sharey=True, squeeze=False)
    _layout(fig, left=0.07, wspace=0.12)
    kinds_seen: list[str] = []
    for ax, m in zip(axes[0], models):
        est = report["estimands"][m][MODE_PRIMARY]
        kinds = _shown_kinds(report, list(est[BRANCH_PRIMARY]["acc_by_leg"]))
        kinds_seen += [k for k in kinds if k not in kinds_seen]
        for x, kind in enumerate(kinds):
            base = f"estimands.{m}.{MODE_PRIMARY}"
            a = est[BRANCH_PRIMARY]["acc_by_leg"][kind]
            q = est[BRANCH_SCORE_AS_IS]["acc_by_leg"][kind]
            ax.bar([x], [a["point"]], width=0.62, facecolor=KIND_COLOR[kind], edgecolor="none", zorder=3)
            ax.plot([x, x], [a["ci_low"], a["ci_high"]], color=INK, lw=1.0, zorder=5)
            ax.plot([x + 0.22], [q["point"]], marker="o", ms=5, mfc="none", mec=INK, mew=1.0, ls="", zorder=6)
            ax.text(x, a["ci_high"] + 0.015, f"{a['point']:.3f}", ha="center", va="bottom", fontsize=6.6, color=INK2)
            for f in ("point", "ci_low", "ci_high", "n_scored", "n_censored"):
                T._num_at(numbers, report, f"acc.{f}[{m}/{kind}/{BRANCH_PRIMARY}]", f"{base}.{BRANCH_PRIMARY}.acc_by_leg.{kind}.{f}", a[f])
            T._num_at(numbers, report, f"acc.point[{m}/{kind}/{BRANCH_SCORE_AS_IS}]", f"{base}.{BRANCH_SCORE_AS_IS}.acc_by_leg.{kind}.point", q["point"])
        # the published three keep their published ticks; a wider (eight-arm) panel rotates its arm labels
        ax.set_xticks(range(len(kinds)))
        ax.set_xticklabels(list(kinds), fontsize=7.6 if len(kinds) <= 3 else 5.8,
                           rotation=0 if len(kinds) <= 3 else 45, ha="center" if len(kinds) <= 3 else "right")
        ax.set_ylim(0, 1.08)
        _label(ax, m)
        T._recessive(ax)
    axes[0][0].set_ylabel("conditional accuracy (scored items, own denominator)", fontsize=8.5)
    handles = [Patch(facecolor=KIND_COLOR[k], edgecolor="none", label=KIND_LABEL[k]) for k in (kinds_seen or KINDS)]
    handles += [plt.Line2D([], [], color=INK, lw=1.0, label="95 % bootstrap CI (conditional, primary branch)"),
                plt.Line2D([], [], marker="o", mfc="none", mec=INK, ls="", label="score-as-is point (unscored -> wrong)")]
    axes[0][-1].legend(handles=handles, loc="lower right", frameon=False, fontsize=6.4)
    _footnotes_below(fig, axes[0][-1], _footnote_lines(report), clearance_in=0.95 if len(kinds_seen) > 3 else 0.5)
    _title(fig, "Cell accuracy\n" + _with_selection(report, f"MATH-500 per arm and model, thinking ON at cap {CAP:,}; primary sampling row, "
                                                            "k = 1; hollow = score-as-is"))
    return fig, numbers


def _contrast_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    comps = report["comparisons"]
    excluded = figure_excluded_families(report)
    excluded_arms = figure_excluded_arms(report)
    for primary in (True, False):
        for j, c in enumerate(comps):
            if c["estimands"] is None or bool(c["primary"]) != primary or c["model_key"] in excluded or not _mode_shown(report, c["mode"]):
                continue
            for k, signs in comp_contrasts(c).items():
                cell_kind = plus_kind(signs)
                if ARM_BY_KIND[cell_kind].cell in excluded_arms:
                    continue
                rows.append({"label": f"{c['model_key']}{_view_tag(comp_view(c))}{'' if c['primary'] else ' [' + c['mode'] + ']'}: {k}",
                             "path": f"estimands.{c['model_key']}.{comp_slot(c)}", "contrast": k, "kind": "contrast",
                             "cell_kind": cell_kind,
                             "word": (c["verdicts"][k]["word"] if c["primary"] else None), "comp": j})
    for m, by_mode in report.get("mode_deltas", {}).items():
        if m in excluded:
            continue
        for mode in by_mode:
            # a mode delta is `<mode> - <primary>` and its rows name the primary row — drawn only when BOTH
            # rows are shown, never with the primary out of the selection
            if not _mode_shown(report, mode) or not _mode_shown(report, MODE_PRIMARY):
                continue
            for kind in _shown_kinds(report, _delta_kinds(report, m, mode)):
                rows.append({"label": f"{m} {kind}: {mode} - {MODE_PRIMARY}", "path": f"mode_deltas.{m}.{mode}",
                             "contrast": mode_delta_contrast(kind, mode), "kind": "delta", "cell_kind": kind, "word": None,
                             "comp": None})
    return rows


def _delta_kinds(report: Mapping[str, Any], model_key: str, mode: str) -> list[str]:
    """The kinds a filed mode delta paired: its record's ``kinds`` (the eight-arm shape), else the published three."""
    for d in report.get("mode_delta_records") or ():
        if d.get("model_key") == model_key and d.get("mode") == mode and d.get("kinds"):
            return list(d["kinds"])
    return list(KINDS)


def fig_contrasts(report: Mapping[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    plt, mcolors, Patch = T._mpl()
    numbers: list[dict[str, Any]] = []
    rows = _contrast_rows(report)
    fig, ax = plt.subplots(figsize=(9.2, ROW_PITCH_IN * len(rows) + 1.3))
    _layout(fig, left=0.36, right=0.855)
    ax.axvline(0.0, color=BASELINE, lw=0.8, zorder=2)
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        p = report_path(report, f"{r['path']}.{BRANCH_PRIMARY}.stats.{r['contrast']}")
        q = report_path(report, f"{r['path']}.{BRANCH_SCORE_AS_IS}.stats.{r['contrast']}")
        if r["kind"] == "contrast":
            # second reader M-b, corrected by round 2 (M2): the denominators UNDER the row's label on two lines — scored /
            # unscored counts with the censoring differential and its direction, then at-cap counts with the neutral at-cap
            # differential and the branch shift (a mode delta pairs one cell with itself across two sampling rows and has
            # no anchor; it carries none)
            den = contrast_denominators(report, r["comp"], r["contrast"])
            _denominator_text(ax, y, _denominator_lines(den))
            _denominator_numbers(numbers, report, r["label"], den)
        if p["bound_low"] != p["bound_high"]:
            ax.add_patch(plt.Rectangle((_pp(p["bound_low"]), y - 0.36), _pp(p["bound_high"] - p["bound_low"]), 0.72,
                                       facecolor=mcolors.to_rgba(INK2, 0.10), edgecolor=INK2, lw=0.6, zorder=2))
        ax.plot([_pp(p["ci_low"]), _pp(p["ci_high"])], [y + 0.12, y + 0.12], color=INK, lw=1.1, zorder=5)
        ax.plot([_pp(p["point"])], [y + 0.12], marker="o", ms=5, color=KIND_COLOR[r["cell_kind"]], mec=INK, mew=0.5, ls="",
                zorder=6)
        ax.plot([_pp(q["ci_low"]), _pp(q["ci_high"])], [y - 0.14, y - 0.14], color=INK2, lw=0.9, zorder=5)
        ax.plot([_pp(q["point"])], [y - 0.14], marker="o", ms=5, mfc="none", mec=INK2, mew=1.0, ls="", zorder=6)
        for b in BRANCHES:
            s = p if b == BRANCH_PRIMARY else q
            for f in STAT_FIELDS:
                T._num_at(numbers, report, f"{r['label']}.{f}[{b}]", f"{r['path']}.{b}.stats.{r['contrast']}.{f}", s[f])
        if r["word"] is not None:
            ax.text(1.005, y, r["word"], transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=6.4,
                    color=INK if r["word"] in WORDS else MUTED, clip_on=False)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["label"] for r in reversed(rows)], fontsize=6.8)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel("conditional-accuracy difference vs bare, percentage points (filled: scored items; hollow: score-as-is)")
    T._recessive(ax)
    ax.xaxis.grid(True, zorder=0)
    ax.yaxis.grid(False)
    handles = [plt.Line2D([], [], marker="o", color=INK2, mec=INK, ls="", label="conditional on scored items (primary), 95 % bootstrap CI"),
               plt.Line2D([], [], marker="o", mfc="none", mec=INK2, ls="", label="score-as-is (unscored -> wrong), 95 % bootstrap CI"),
               Patch(facecolor=mcolors.to_rgba(INK2, 0.10), edgecolor=INK2, label="censoring-sensitivity bound")]
    h_in = ax.get_position().height * fig.get_figheight()
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.5 / h_in), ncol=3, frameon=False, fontsize=6.4)
    _footnotes_below(fig, ax, _footnote_lines(report), clearance_in=0.85)
    _title(fig, "Contrasts\n" + _with_selection(report, f"arm − bare per model and sampling row, verdict word at right "
                                                        f"(B = {report['bootstrap']['B']:,})"))
    return fig, numbers


def _shown_modes_of(report: Mapping[str, Any], model_key: str) -> list[str]:
    """The sampling rows of ``model_key`` the selection shows, primary first (``[]`` when the selection leaves it none)."""
    return [mode for mode in (MODE_PRIMARY, *SENSITIVITY_MODES) if mode in report["estimands"][model_key] and _mode_shown(report, mode)]


def _sensitivity_models(report: Mapping[str, Any]) -> list[str]:
    """The panels of the sampling-sensitivity figure: the visible models with at least one SHOWN row, the models with two
    or more first (the figure's subject), then the single-row ones — never a model the selection left without a row."""
    visible = _shown_models(report, list(report["estimands"]))
    multi = [m for m in visible if len(_shown_modes_of(report, m)) >= 2]
    single = [m for m in visible if len(_shown_modes_of(report, m)) == 1]
    return multi + single


def fig_sampling_sensitivity(report: Mapping[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    plt, _mcolors, _Patch = T._mpl()
    numbers: list[dict[str, Any]] = []
    # the visible models, each with the sampling rows the selection SHOWS (no fallback onto an excluded mode — the
    # builder used to draw a model's first present mode when the selection left it none, so `--figure-modes battery`
    # put the excluded primary row into the figure and the gate refused a valid selection); the panels are the models with
    # at least one shown row (every visible model carries the primary row when it is shown; a model with none is omitted,
    # never given another model's row), the multi-row models first
    models = _sensitivity_models(report)
    fig, axes = plt.subplots(1, len(models), figsize=(2.6 * len(models) + 1.0, 4.2), sharey=True, squeeze=False)
    _layout(fig, left=0.08, wspace=0.12)
    modes_seen: list[str] = []
    n_kinds_max = 0
    for ax, m in zip(axes[0], models):
        modes = _shown_modes_of(report, m)
        kinds = _shown_kinds(report, list(report["estimands"][m][modes[0]][BRANCH_PRIMARY]["acc_by_leg"]))
        n_kinds_max = max(n_kinds_max, len(kinds))
        for j, mode in enumerate(modes):
            if mode not in modes_seen:
                modes_seen.append(mode)
            est = report["estimands"][m][mode][BRANCH_PRIMARY]
            for x, kind in enumerate(kinds):
                a = est["acc_by_leg"].get(kind)
                if a is None:
                    continue
                xx = x + (j - (len(modes) - 1) / 2) * 0.22
                ax.plot([xx, xx], [a["ci_low"], a["ci_high"]], color=INK2, lw=0.9, zorder=4)
                ax.plot([xx], [a["point"]], marker=MODE_MARKER[mode], ms=6, color=KIND_COLOR[kind], mec=INK, mew=0.5, ls="", zorder=6)
                T._num_at(numbers, report, f"acc.point[{m}/{kind}/{mode}]", f"estimands.{m}.{mode}.{BRANCH_PRIMARY}.acc_by_leg.{kind}.point", a["point"])
        ax.set_xticks(range(len(kinds)))
        ax.set_xticklabels(list(kinds), fontsize=7.6 if len(kinds) <= 3 else 5.8,
                           rotation=0 if len(kinds) <= 3 else 45, ha="center" if len(kinds) <= 3 else "right")
        _label(ax, m)
        T._recessive(ax)
    axes[0][0].set_ylabel("conditional accuracy (scored items; primary branch)", fontsize=8.5)
    handles = [plt.Line2D([], [], marker=MODE_MARKER[mode], color=INK2, mec=INK, ls="", label=mode) for mode in modes_seen]
    axes[0][-1].legend(handles=handles, loc="lower right", frameon=False, fontsize=6.6)
    _footnotes_below(fig, axes[0][-1], _footnote_lines(report), clearance_in=0.95 if n_kinds_max > 3 else 0.5)
    head, sub = _sensitivity_title(modes_seen)
    _title(fig, head + "\n" + _with_selection(report, sub))
    return fig, numbers


def _sensitivity_title(modes_seen: Sequence[str]) -> tuple[str, str]:
    """Review L6: a render that shows ONE sampling row is not a sampling-row comparison, and the figure says so instead of
    wearing the sensitivity title over a single row."""
    if len(modes_seen) <= 1:
        row = modes_seen[0] if modes_seen else MODE_PRIMARY
        # short enough to survive the selection clause whole (the line is bounded at TITLE_MAX_CHARS)
        return "Primary row", f"one row shown ({row}): no sampling-row comparison"
    return "Sampling rows", "accuracy per cell under each sampling row present (95 % CIs); rows are separate instruments, never pooled"


def _leg_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [row for row in report["controls"] if row.get("summary", {}).get("n")]


def _leg_yticks(ax, ys: Sequence[int], rows: Sequence[Mapping[str, Any]]) -> None:
    """One horizontal bar per leg: the leg labels on the y axis."""
    ax.set_yticks(ys)
    ax.set_yticklabels([_leg_short(r) for r in rows], fontsize=6.4)


def _recessive_xgrid(ax) -> None:
    """The parent's recessive frame with a vertical grid only."""
    T._recessive(ax)
    ax.xaxis.grid(True, zorder=0)
    ax.yaxis.grid(False)


def fig_atcap_thinksplit(report: Mapping[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    plt, _mcolors, Patch = T._mpl()
    numbers: list[dict[str, Any]] = []
    rows = _shown_rows(report, _leg_rows(report))
    idx = _ctl_index(report)
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 0.24 * len(rows) + 1.4), sharey=True)
    _layout(fig, left=0.22, wspace=0.18)
    ys = list(range(len(rows)))[::-1]
    for y, row in zip(ys, rows):
        i, s, ac = idx[row["leg"]], row["summary"], row["at_cap"]
        n = s["n"]
        b, c = ac["n_case_b_unterminated_at_cap"] / n, ac["n_case_c_answer_at_cap"] / n
        axes[0].barh([y], [_pp(b)], height=0.62, color=CAT["orange"], edgecolor="none", zorder=3)
        axes[0].barh([y], [_pp(c)], left=[_pp(b)], height=0.62, color=CAT["yellow"], edgecolor="none", zorder=3)
        axes[1].barh([y], [_pp(s["n_think_block_unterminated"] / n)], height=0.62, color=CAT["blue"], edgecolor="none", zorder=3)
        axes[2].barh([y], [_pp(s["n_think_block_present"] / n)], height=0.62, color=CAT["aqua"], edgecolor="none", zorder=3)
        T._num_at(numbers, report, f"n_case_b[{row['leg']}]", f"controls.{i}.at_cap.n_case_b_unterminated_at_cap", ac["n_case_b_unterminated_at_cap"])
        T._num_at(numbers, report, f"n_case_c[{row['leg']}]", f"controls.{i}.at_cap.n_case_c_answer_at_cap", ac["n_case_c_answer_at_cap"])
        T._num_at(numbers, report, f"n_think_block_unterminated[{row['leg']}]", f"controls.{i}.summary.n_think_block_unterminated", s["n_think_block_unterminated"])
        T._num_at(numbers, report, f"n_think_block_present[{row['leg']}]", f"controls.{i}.summary.n_think_block_present", s["n_think_block_present"])
    axes[0].axvline(_pp(CAP_GATE_AT_CAP_RATE), color=CAT["orange"], lw=0.9, ls="--", zorder=4)
    _leg_yticks(axes[0], ys, rows)
    for ax, lab in zip(axes, ("at cap, % of draws (b: in think block; c: in answer)", "unterminated think block, %", "think block present, %")):
        ax.set_xlabel(lab, fontsize=7.4)
        _recessive_xgrid(ax)
    axes[0].legend(handles=[Patch(color=CAT["orange"], label="case b"), Patch(color=CAT["yellow"], label="case c"),
                            plt.Line2D([], [], color=CAT["orange"], ls="--", label="2 % gate")], loc="lower right", frameon=False, fontsize=6.4)
    _footnotes_below(fig, axes[-1], _raw_draw_lines(report), clearance_in=0.45)        # second reader L: the raw-draw statement
    _title(fig, "At-cap split\n" + _with_selection(report, "at-cap (b: in the think block, c: in the answer), unterminated and "
                                                          "think-present shares per cell; summary counts over 500 draws"))
    return fig, numbers


def fig_completion_tokens(report: Mapping[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    plt, mcolors, _Patch = T._mpl()
    numbers: list[dict[str, Any]] = []
    rows = _shown_rows(report, _leg_rows(report))
    idx = _ctl_index(report)
    fig, ax = plt.subplots(figsize=(8.0, 0.24 * len(rows) + 1.4))
    _layout(fig, left=0.26)
    ys = list(range(len(rows)))[::-1]
    for y, row in zip(ys, rows):
        i, s, led = idx[row["leg"]], row["summary"], row["ledger"]
        qs = led.get("completion_tokens_quantiles") or {}
        if qs:
            ax.plot([qs["p50"], qs["p99"]], [y, y], color=mcolors.to_rgba(CAT["blue"], 0.55), lw=3.0, solid_capstyle="butt", zorder=3)
            for k in ("p50", "p90", "p99", "max"):
                T._num_at(numbers, report, f"ledger.completion_tokens.{k}[{row['leg']}]", f"controls.{i}.ledger.completion_tokens_quantiles.{k}", qs[k])
        ax.plot([s["mean_completion_tokens"]], [y], marker="o", ms=5, color=INK, ls="", zorder=6)
        ax.plot([s["p99_completion_tokens"]], [y], marker="|", ms=9, color=INK, mew=1.4, ls="", zorder=6)
        T._num_at(numbers, report, f"mean_completion_tokens[{row['leg']}]", f"controls.{i}.summary.mean_completion_tokens", s["mean_completion_tokens"])
        T._num_at(numbers, report, f"p99_completion_tokens[{row['leg']}]", f"controls.{i}.summary.p99_completion_tokens", s["p99_completion_tokens"])
    ax.axvline(CAP, color=CAT["orange"], lw=0.9, ls="--", zorder=4)
    _leg_yticks(ax, ys, rows)
    ax.set_xlabel("completion tokens (summary mean = dot, summary p99 = tick; ledger p50-p99 = band; cap dashed)", fontsize=7.4)
    _recessive_xgrid(ax)
    _footnotes_below(fig, ax, _raw_draw_lines(report), clearance_in=0.45)               # second reader L: the raw-draw statement
    _title(fig, "Completion lengths\n" + _with_selection(report, f"per cell against the {CAP:,} cap: summary mean (dot) and p99 (tick), "
                                                                 "ledger p50–p99 band"))
    return fig, numbers


def fig_ledger(report: Mapping[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    plt, _mcolors, Patch = T._mpl()
    numbers: list[dict[str, Any]] = []
    rows = _shown_rows(report, _leg_rows(report))
    idx = _ctl_index(report)
    fig, ax = plt.subplots(figsize=(8.0, 0.24 * len(rows) + 1.4))
    _layout(fig, left=0.26)
    ys = list(range(len(rows)))[::-1]
    for y, row in zip(ys, rows):
        i, led = idx[row["leg"]], row["ledger"]
        n_calls = led.get("n_calls") or 0
        codes = led.get("error_code") or {}
        n_coded = sum(codes.values())
        ok = n_calls - n_coded
        ax.barh([y], [ok], height=0.62, color=CAT["aqua"], edgecolor="none", zorder=3)
        ax.barh([y], [n_coded], left=[ok], height=0.62, color=CAT["orange"], edgecolor="none", zorder=3)
        ax.barh([y], [led.get("n_rejected") or 0], left=[n_calls], height=0.62, color=CAT["yellow"], edgecolor="none", zorder=3)
        rd = led.get("redraws") or {}
        ax.text(n_calls + (led.get("n_rejected") or 0) + 4, y, f"{n_calls} calls, {n_coded} coded, {led.get('n_rejected') or 0} rejected, "
                f"{rd.get('n_redraw_calls') or 0} re-sent ({rd.get('n_items_redrawn') or 0} items), {rd.get('n_resume_calls') or 0} resume",
                va="center", fontsize=5.8, color=INK2)
        T._num_at(numbers, report, f"ledger.n_calls[{row['leg']}]", f"controls.{i}.ledger.n_calls", led.get("n_calls"))
        T._num_at(numbers, report, f"ledger.n_rejected[{row['leg']}]", f"controls.{i}.ledger.n_rejected", led.get("n_rejected"))
        T._num_at(numbers, report, f"ledger.n_resume_rows[{row['leg']}]", f"controls.{i}.ledger.n_resume_rows", led.get("n_resume_rows"))
        T._num_at(numbers, report, f"ledger.n_error_code[{row['leg']}]", f"controls.{i}.ledger.error_code", n_coded, transform="sum of the error_code counts")
        if rd:
            T._num_at(numbers, report, f"ledger.redraws.n_redraw_calls[{row['leg']}]", f"controls.{i}.ledger.redraws.n_redraw_calls", rd.get("n_redraw_calls"))
            T._num_at(numbers, report, f"ledger.redraws.n_items_redrawn[{row['leg']}]", f"controls.{i}.ledger.redraws.n_items_redrawn", rd.get("n_items_redrawn"))
    _leg_yticks(ax, ys, rows)
    ax.set_xlim(0, max(1, N_ITEMS) * 1.75)
    ax.set_xlabel("usage-ledger calls of the task per cell", fontsize=7.4)
    _recessive_xgrid(ax)
    ax.legend(handles=[Patch(color=CAT["aqua"], label="forwarded, no error_code"), Patch(color=CAT["orange"], label="error_code (refused by name)"),
                       Patch(color=CAT["yellow"], label="rejected (censoring)")], loc="lower right", frameon=False, fontsize=6.4)
    _footnotes_below(fig, ax, _raw_draw_lines(report), clearance_in=0.45)               # second reader L: the raw-draw statement
    _title(fig, "Usage ledger\n" + _with_selection(report, "calls per cell: forwarded, named refusals, rejections, same-cap re-sends (D151)"))
    return fig, numbers


def fig_item_transitions(report: Mapping[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    plt, _mcolors, Patch = T._mpl()
    numbers: list[dict[str, Any]] = []
    excluded = figure_excluded_families(report)
    excluded_arms = figure_excluded_arms(report)
    comps = [(j, c) for j, c in enumerate(report["comparisons"]) if c["primary"] and c["estimands"] is not None and c["model_key"] not in excluded
             and comp_view(c) == VIEW_MAIN]        # E3: one panel per model — the main view's comparison
    fig, axes = plt.subplots(1, len(comps), figsize=(2.6 * len(comps) + 1.0, 4.0), sharey=True, squeeze=False)
    _layout(fig, left=0.07, wspace=0.12)
    keys = T.TRANSITION_KEYS
    kinds_seen: list[str] = []
    for ax, (j, c) in zip(axes[0], comps):
        # the arms in PANEL order (bare -> prompted before bare -> object, the published layout), each with its own hue
        contrasts = sorted(((contrast, plus_kind(signs)) for contrast, signs in comp_contrasts(c).items()),
                           key=lambda cs: ARM_KINDS.index(cs[1]))
        contrasts = [(contrast, kind) for contrast, kind in contrasts if ARM_BY_KIND[kind].cell not in excluded_arms]
        n_groups = max(len(contrasts), 1)
        width = 0.72 / n_groups
        for g, (contrast, kind) in enumerate(contrasts):
            if kind not in kinds_seen:
                kinds_seen.append(kind)
            counts = c["transitions"][contrast]["counts"]
            xs = [i + (g - (n_groups - 1) / 2) * width for i in range(len(keys))]
            ax.bar(xs, [counts[k] for k in keys], width=width * 0.94, color=KIND_COLOR[kind], edgecolor="none", zorder=3)
            for x, k in zip(xs, keys):
                ax.text(x, counts[k] + 3, str(counts[k]), ha="center", va="bottom", fontsize=5.6 if n_groups <= 2 else 4.6, color=INK2, clip_on=True)
                T._num_at(numbers, report, f"transitions.{c['model_key']}.{contrast}[{k}]", f"comparisons.{j}.transitions.{contrast}.counts.{k}", counts[k])
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(list(keys), fontsize=7.0)
        ax.set_ylim(0, N_ITEMS * 1.12)
        _label(ax, c["model_key"])
        T._recessive(ax)
    axes[0][0].set_ylabel("items")
    axes[0][-1].legend(handles=[Patch(color=KIND_COLOR[k], label=f"bare -> {k}") for k in ARM_KINDS if k in kinds_seen],
                       loc="upper right", bbox_to_anchor=(1.0, 0.92), frameon=False, fontsize=6.6)
    _footnotes_below(fig, axes[0][-1], _footnote_lines(report), clearance_in=0.45)
    _title(fig, "Item transitions\n" + _with_selection(report, "from bare to each arm present (C correct, I incorrect, U unscored), primary row"))
    return fig, numbers


def _readout_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The controls rows carrying the two readouts — the usable legs (a refused
    leg files neither block)."""
    return [row for row in report["controls"] if row.get("at_cap") and row.get("thinking")]


def _fit_title_line(line: str) -> str:
    """A figure's second title line, held to :data:`TITLE_MAX_CHARS` whatever
    the numbers and labels in it: the readings below are bounded by
    construction for the designed matrix, and a longer label (another matrix's
    model key) is cut rather than raised on — a title never blocks the figure
    set, the sidecar, the mirror or the report; the full names are in the JSON."""
    return line if len(line) <= TITLE_MAX_CHARS else line[:TITLE_MAX_CHARS - 1] + "…"


def _share_reading(tr: Mapping[str, Any], *, shown: Sequence[Mapping[str, Any]] | None = None, shown_only: bool = False) -> str:
    """The at-cap share figure's second title line: the premise verdict in one
    clause. ``shown`` (the cells the figure draws when families are excluded):
    the premise word stays the wave's, the counts and the max are over the
    primary cells SHOWN (the same fields :func:`truncation_block` counts); ``shown_only``: the
    descriptor is the shown set's alone, labelled ``shown:``, with no wave clause (its count runs
    over cells the figure does not draw)."""
    if shown is not None:
        # LABELLED as over the families shown, the wave's verdict and count first; the max leg is not NAMED here (the tallest bar
        # is it, and the sidecar record's source paths name the cell) so the line fits the title bound on the real matrices
        prim = [c for c in shown if c["primary"]]
        top = max(prim, key=lambda c: c["share"] if c["share"] is not None else -1.0, default=None)
        top_txt = (f"max {_pp(top['share'] or 0.0):.1f} %" if top is not None else "no primary cell")
        n_thr = sum(1 for c in prim if c["at_or_above_surfacing_threshold"])
        n_over = sum(1 for c in prim if c["line_1pct_exceeded"])
        n_fam = len({c["model_key"] for c in shown})
        if tr["premise_failed"] is None:
            wave = f"premise undetermined, {len(tr['primary_legs_not_read'])} not read"
        elif tr["premise_failed"]:
            wave = f"premise failed, {len(tr['primary_at_or_above_threshold'])} >= 3 %"
        else:
            wave = f"premise holds, {len(tr['primary_over_1pct'])} over 1 %"
        if shown_only:
            return _fit_title_line(f"shown: {n_thr} of {len(prim)} primary cells >= 3 %, {n_over} over 1 %, {top_txt}")
        line = f"{_families_text(n_fam)} (wave: {wave}): {n_thr} >= 3 %, {n_over} over 1 %, {top_txt}"
        return _fit_title_line(line)
    top_leg = next((c for c in tr["cells"] if c["leg"] == tr["max_primary_share_leg"]), None)
    label = _leg_short(top_leg) if top_leg is not None else "no primary cell"
    n_thr, n_over = len(tr["primary_at_or_above_threshold"]), len(tr["primary_over_1pct"])
    if tr["premise_failed"]:
        line = f"premise failed: {n_thr} primary cell(s) >= 3 %; max {_pp(tr['max_primary_share']):.1f} % in {label}"
    elif tr["premise_failed"] is None:
        line = (f"premise undetermined: {len(tr['primary_legs_not_read'])} of {tr['n_primary_cells_design']} designed primary "
                "cells not read")
    else:
        line = f"premise holds: max {_pp(tr['max_primary_share']):.1f} % in {label}; {n_over} primary cell(s) above the 1 % line"
    return _fit_title_line(line)


def fig_atcap_share(report: Mapping[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    """The at-cap share per cell against the 1 % line and the 3 % surfacing
    threshold: bars grouped by model, the primary cells in the cell hue, the
    sensitivity rows beside them in a lighter shade."""
    plt, mcolors, Patch = T._mpl()
    numbers: list[dict[str, Any]] = []
    tr = report["truncation"]
    cells = tr["cells"]
    shown_cells = _shown_rows(report, cells)
    models = list(dict.fromkeys(c["model_key"] for c in shown_cells))
    fig, ax = plt.subplots(figsize=(2.5 * max(len(models), 1) + 1.6, 4.0))
    _layout(fig, left=0.08)
    pitch = 0.11
    top_pp = 0.0
    for g, m in enumerate(models):
        mine = [c for c in shown_cells if c["model_key"] == m]
        ordered = [c for k in ARM_KINDS for c in mine if c["kind"] == k and c["primary"]] + \
            [c for mode in SENSITIVITY_MODES for k in ARM_KINDS for c in mine if c["kind"] == k and c["mode"] == mode]
        for b, c in enumerate(ordered):
            i = cells.index(c)
            x = g + (b - (len(ordered) - 1) / 2) * pitch
            y = 0.0 if c["share"] is None else _pp(c["share"])
            top_pp = max(top_pp, y)
            color = KIND_COLOR[c["kind"]] if c["primary"] else mcolors.to_rgba(KIND_COLOR[c["kind"]], 0.38)
            ax.bar([x], [y], width=pitch * 0.86, color=color, edgecolor="none", zorder=3)
            if c["share"] is not None:
                T._num_at(numbers, report, f"at_cap.share[{c['leg']}]", f"truncation.cells.{i}.share", c["share"])
            # the second series — the share of the cell's items the harness re-drew at the same cap (a marker
            # per cell, in % of items), so the selection the at-cap share rides on is visible beside it
            if c.get("share_items_redrawn") is not None:
                y2 = _pp(c["share_items_redrawn"])
                top_pp = max(top_pp, y2)
                ax.plot([x], [y2], marker="D", ms=2.8, color=INK, mec=SURFACE, mew=0.5, ls="none", zorder=6)
                T._num_at(numbers, report, f"redraws.share_items_redrawn[{c['leg']}]", f"truncation.cells.{i}.share_items_redrawn",
                          c["share_items_redrawn"])
    ax.axhline(_pp(AT_CAP_LINE_1PCT), color=INK2, lw=0.9, ls="--", zorder=4)
    ax.axhline(_pp(AT_CAP_SURFACING_THRESHOLD), color=CAT["orange"], lw=0.9, ls="-.", zorder=4)
    # x in axes fraction, y in data units: the label sits on its line at the right edge
    ax.text(0.995, _pp(AT_CAP_LINE_1PCT), "1 % line (the harness rule)", transform=ax.get_yaxis_transform(), ha="right",
            va="bottom", fontsize=6.4, color=INK2)
    ax.text(0.995, _pp(AT_CAP_SURFACING_THRESHOLD), "3 % surfacing threshold (2x the GSM8K premise)",
            transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=6.4, color=CAT["orange"])
    ax.set_xlim(-0.5, max(len(models), 1) - 0.5)
    ax.set_ylim(0, max(4.0, 1.25 * top_pp))
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=7.6)
    ax.set_ylabel("at-cap draws, % (bars); items re-drawn, % (markers)", fontsize=7.0)     # the long form is in the caption
    T._recessive(ax)
    from matplotlib.lines import Line2D
    kinds_shown = [k for k in ARM_KINDS if any(c["kind"] == k for c in shown_cells)] or list(KINDS)
    shown_only = figure_shown_only(report)
    # under shown_only a legend entry describes a bar the figure DRAWS — the lighter-shade patch appears only when a
    # sensitivity-row cell is among the cells shown (the default render keeps it)
    sensitivity_patch = ([] if shown_only and not any(not c["primary"] for c in shown_cells)
                         else [Patch(color=mcolors.to_rgba(MUTED, 0.38), label="sensitivity row (lighter)")])
    ax.legend(handles=[Patch(color=KIND_COLOR[k], label=KIND_LABEL[k]) for k in kinds_shown] + sensitivity_patch +
              [Line2D([], [], marker="D", ms=2.8, color=INK, mec=SURFACE, mew=0.5, ls="none",
                      label="items re-drawn at the same cap (client retry), % of items")],
              loc="upper left", frameon=False, fontsize=6.4)
    if figure_excluded_families(report) or (shown_only and figure_selection_active(report)):
        # a shown-cells aggregate is LABELLED, with the wave's value beside it in the title and the sidecar — except under
        # shown_only, when no value over a row the figure does not draw is written anywhere (the descriptor is the shown set's)
        prim_shown = [c for c in shown_cells if c["primary"]]
        n_fam = len(models)
        for name, key in (("n_primary_at_or_above_threshold", "primary_at_or_above_threshold"), ("n_primary_over_1pct", "primary_over_1pct")):
            field = "at_or_above_surfacing_threshold" if key == "primary_at_or_above_threshold" else "line_1pct_exceeded"
            T._num_at(numbers, report, f"truncation.{name}_shown", f"truncation.{key}", sum(1 for c in prim_shown if c[field]),
                      transform=(f"count of the legs in the list among the {len(prim_shown)} primary cells shown ({n_fam} families)" if shown_only
                                 else f"count of the legs in the list whose family is among the {n_fam} shown"))
            if not shown_only:
                T._num_at(numbers, report, f"truncation.{name}_report", f"truncation.{key}", len(tr[key]), transform="length of the list (the wave's value)")
        with_share = [c for c in prim_shown if c["share"] is not None]
        if with_share:
            T._num_at(numbers, report, "truncation.max_primary_share_shown", [f"truncation.cells.{cells.index(c)}.share" for c in with_share],
                      max(c["share"] for c in with_share),
                      transform=(f"max over the {len(with_share)} primary cells shown" if shown_only else f"max over the primary cells of the {n_fam} families shown"))
        if tr["max_primary_share"] is not None and not shown_only:
            T._num_at(numbers, report, "truncation.max_primary_share_report", "truncation.max_primary_share", tr["max_primary_share"])
        reading = _share_reading(tr, shown=shown_cells, shown_only=shown_only)
    else:
        reading = _share_reading(tr)
    if figure_selection_active(report) and not figure_excluded_families(report) and not shown_only:
        # an arm / mode selection (no family exclusion): the reading is over the cells SHOWN, labelled, with the wave's value
        # beside it in the sidecar (the same rule the family filter obeys)
        prim_shown = [c for c in shown_cells if c["primary"]]
        T._num_at(numbers, report, "truncation.n_primary_over_1pct_shown", "truncation.primary_over_1pct",
                  sum(1 for c in prim_shown if c["line_1pct_exceeded"]), transform="count of the legs in the list among the cells shown")
        T._num_at(numbers, report, "truncation.n_primary_over_1pct_report", "truncation.primary_over_1pct", len(tr["primary_over_1pct"]),
                  transform="length of the list (the wave's value)")
    _footnotes_below(fig, ax, _raw_draw_lines(report), clearance_in=0.42)               # second reader L: the raw-draw statement
    _title(fig, f"At-cap share\n{_with_selection(report, reading)}")
    return fig, numbers


def fig_atcap_loopsplit(report: Mapping[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    """Every cell's at-cap draws split by the loop rule: degenerate loop,
    truncated reasoning, and the unscored draws whose text the store rows do not
    carry (classified from the .eval log in the log-reading run only)."""
    plt, mcolors, Patch = T._mpl()
    numbers: list[dict[str, Any]] = []
    rows = _shown_rows(report, _readout_rows(report))
    idx = _ctl_index(report)
    fig, ax = plt.subplots(figsize=(8.0, 0.24 * max(len(rows), 1) + 1.4))
    _layout(fig, left=0.26)
    ys = list(range(len(rows)))[::-1]
    totals = {"n_loop": 0, "n_truncated_reasoning": 0, "n_text_unavailable": 0}
    widest = 0
    for y, row in zip(ys, rows):
        i, ac = idx[row["leg"]], row["at_cap"]
        left = 0
        for f, color, hatch in (("n_loop", CAT["orange"], None), ("n_truncated_reasoning", CAT["blue"], None),
                                ("n_text_unavailable", mcolors.to_rgba(MUTED, 0.55), "////")):
            ax.barh([y], [ac[f]], left=[left], height=0.62, color=color, edgecolor="none" if hatch is None else MUTED,
                    hatch=hatch, linewidth=0, zorder=3)
            left += ac[f]
            totals[f] += ac[f]
            T._num_at(numbers, report, f"at_cap.{f}[{row['leg']}]", f"controls.{i}.at_cap.{f}", ac[f])
        widest = max(widest, left)
    _leg_yticks(ax, ys, rows)
    ax.set_xlim(0, max(1, widest) * 1.18)
    ax.set_xlabel("at-cap draws of 500: loop (the rule) | truncated reasoning | unscored draw, text in the .eval log only",
                  fontsize=7.4)
    _recessive_xgrid(ax)
    # the totals travel in the legend (unbounded numbers never enter the title, whose second line is a fixed sentence);
    # the legend's title names the text each cell's split ran over (the same for every cell in a run, else "mixed")
    source_word, source_line = _split_source_labels(rows)
    shown_only = figure_shown_only(report)
    if figure_excluded_families(report) or (shown_only and figure_selection_active(report)):
        # totals over the legs shown are LABELLED, with the wave's totals beside them, in the legend and in the sidecar (never a silent
        # substitution) — except under shown_only, when the wave's totals are not written: they sum rows the figure does not draw
        all_rows = _readout_rows(report)
        totals_report = {f: sum(r["at_cap"][f] for r in all_rows) for f in totals}
        n_fam = len({r["model_key"] for r in rows})
        for f in totals:
            T._num_at(numbers, report, f"at_cap.total_{f}_shown", [f"controls.{idx[r['leg']]}.at_cap.{f}" for r in rows], totals[f],
                      transform=(f"sum over the {len(rows)} legs shown ({n_fam} families)" if shown_only else f"sum over the legs of the {n_fam} families shown"))
            if not shown_only:
                T._num_at(numbers, report, f"at_cap.total_{f}_report", [f"controls.{idx[r['leg']]}.at_cap.{f}" for r in all_rows], totals_report[f],
                          transform="sum over every usable leg (the wave's value)")

        if shown_only:
            def label(f: str, text: str) -> str:
                return f"{text} ({totals[f]:,} draws over the {len(rows)} legs shown)"
        else:
            def label(f: str, text: str) -> str:
                return f"{text} ({totals[f]:,} draws over the {_families_text(n_fam)}; wave {totals_report[f]:,})"
    else:
        def label(f: str, text: str) -> str:
            return f"{text} ({totals[f]:,} draws)"
    ax.legend(handles=[Patch(color=CAT["orange"], label=label("n_loop", "degenerate loop")),
                       Patch(color=CAT["blue"], label=label("n_truncated_reasoning", "truncated reasoning")),
                       Patch(facecolor=mcolors.to_rgba(MUTED, 0.55), edgecolor=MUTED, hatch="////",
                             label=label("n_text_unavailable", "text unavailable to the split"))],
              loc="lower right", frameon=False, fontsize=6.4, title=source_word, title_fontsize=6.4)
    _footnotes_below(fig, ax, _raw_draw_lines(report), clearance_in=0.45)               # second reader L: the raw-draw statement
    _title(fig, f"Loop split\n{_with_selection(report, source_line)}")
    return fig, numbers


def _split_source_labels(rows: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    """(legend title, second title line) naming the text the cells' splits ran
    over — fixed sentences, one per case, so the title stays bounded."""
    sources = {row["at_cap"]["split_source"] for row in rows}
    if sources == {SPLIT_SOURCE_EVAL_LOG}:
        return ("classified from the eval logs",
                "totals in the legend; every at-cap draw classified from the eval logs (case b included)")
    if sources <= {SPLIT_SOURCE_STORE_ROWS}:
        return ("store rows only; unscored draws' text not carried",
                "totals in the legend; grey = unscored draws whose text the store rows do not carry")
    return ("mixed sources per cell (see the report)",
            "totals in the legend; split source differs per cell — see truncation.cells[*].split_source")


def fig_no_reasoning_share(report: Mapping[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    """The readout per cell: thinking-ON draws that finished with no
    reasoning emitted, as a share of the cell's draws; the implicit-open
    families are 0 by construction and say so instead of a bar."""
    plt, _mcolors, Patch = T._mpl()
    numbers: list[dict[str, Any]] = []
    rows = _shown_rows(report, _readout_rows(report))
    idx = _ctl_index(report)
    fig, ax = plt.subplots(figsize=(8.0, 0.24 * max(len(rows), 1) + 1.4))
    _layout(fig, left=0.26)
    ys = list(range(len(rows)))[::-1]
    top_pp, top_row = 0.0, None
    for y, row in zip(ys, rows):
        i, th = idx[row["leg"]], row["thinking"]
        share = th["share_no_reasoning"]
        if th["by_construction"]:
            ax.text(0.05, y, f"0 by construction ({th['split_mode']})", va="center", fontsize=5.8, color=INK2)
        else:
            y_pp = 0.0 if share is None else _pp(share)
            ax.barh([y], [y_pp], height=0.62, color=KIND_COLOR[row["kind"]], edgecolor="none", zorder=3)
            if y_pp >= top_pp:
                top_pp, top_row = y_pp, row
        if share is not None:
            T._num_at(numbers, report, f"thinking.share_no_reasoning[{row['leg']}]", f"controls.{i}.thinking.share_no_reasoning", share)
    _leg_yticks(ax, ys, rows)
    ax.set_xlim(0, max(2.0, 1.3 * top_pp))
    ax.set_xlabel("thinking-ON draws that finished with no reasoning emitted, % of the cell's 500 draws (D115)", fontsize=7.4)
    _recessive_xgrid(ax)
    kinds_shown = [k for k in ARM_KINDS if any(r["kind"] == k for r in rows)] or list(KINDS)
    ax.legend(handles=[Patch(color=KIND_COLOR[k], label=KIND_LABEL[k]) for k in kinds_shown], loc="lower right", frameon=False,
              fontsize=6.4)
    shown_only = figure_shown_only(report)
    if figure_excluded_families(report) or (shown_only and figure_selection_active(report)):
        all_rows = _readout_rows(report)
        paths_of = lambda rs: [f"controls.{idx[r['leg']]}.thinking.share_no_reasoning" for r in rs if not r["thinking"]["by_construction"] and r["thinking"]["share_no_reasoning"] is not None]
        wave_shares = [r["thinking"]["share_no_reasoning"] for r in all_rows if not r["thinking"]["by_construction"] and r["thinking"]["share_no_reasoning"] is not None]
        wave_pp = _pp(max(wave_shares)) if wave_shares else 0.0
        n_fam = len({r["model_key"] for r in rows})
        if paths_of(rows):
            T._num_at(numbers, report, "thinking.max_share_no_reasoning_shown", paths_of(rows), top_pp / 100.0,
                      transform=(f"max over the {len(rows)} legs shown ({n_fam} families)" if shown_only else f"max over the legs of the {n_fam} families shown"))
        if paths_of(all_rows) and not shown_only:
            T._num_at(numbers, report, "thinking.max_share_no_reasoning_report", paths_of(all_rows), max(wave_shares), transform="max over every usable leg (the wave's value)")
        if shown_only:
            # the shown set's descriptor only: the wave's max runs over rows the figure does not draw
            reading = (f"shown: max {top_pp:.1f} % in {_leg_short(top_row)} ({len(rows)} legs)" if top_row is not None
                       else f"shown: every cell implicit_open (0 by construction; {len(rows)} legs)")
        else:
            reading = (f"{_families_text(n_fam)}: max {top_pp:.1f} % in {_leg_short(top_row)}; wave max {wave_pp:.1f} %" if top_row is not None
                       else f"{_families_text(n_fam)}: every cell implicit_open (0 by construction); wave max {wave_pp:.1f} %")
    else:
        reading = (f"max {top_pp:.1f} % in {_leg_short(top_row)}; implicit_open cells 0 by construction" if top_row is not None
                   else "every cell read is implicit_open: 0 by construction")
    _footnotes_below(fig, ax, _raw_draw_lines(report), clearance_in=0.45)               # second reader L: the raw-draw statement
    _title(fig, f"No reasoning\n{_with_selection(report, reading)}")
    return fig, numbers


# --------------------------------------------------------------------------- #
# the eight-arm figures (plan P5): eight FIXED arm slots per model in panel     #
# order, a deferred / held slot drawn EMPTY and labelled (never dropped), llama #
# labelled "thinking: not available" and held out, the pooled panel drawn from  #
# report["eightarm"]["pooled_by_arm"] — nothing is recomputed here             #
# --------------------------------------------------------------------------- #

EIGHTARM_STEM_ACCURACY, EIGHTARM_STEM_CONTRASTS = "final_m5t_eightarm_accuracy", "final_m5t_eightarm_contrasts"
EIGHTARM_STEMS: tuple[str, ...] = (EIGHTARM_STEM_ACCURACY, EIGHTARM_STEM_CONTRASTS)
DEFERRED_SLOT_LABEL = "deferred: dose unmeasured (D2/D3)"
POOLED_GROUP_LABEL = "pooled"
SLOT_HATCH = "////"
#: the letter an EMPTY slot wears, per design status
SLOT_LETTER: Mapping[str, str] = {STATUS_DEFERRED: "D", STATUS_HELD: "H", ROW_STATUS_NOT_ANALYSED: "n", ROW_STATUS_ABSENT: "–",
                                  POOLED_UNAVAILABLE: "u"}
SLOT_LEGEND = "empty slot — D deferred (dose unmeasured, D2/D3) · H held · n not analysed · – absent · u pooled unavailable"


def _eightarm_group_label(model_key: str) -> str:
    return f"{model_key}\nthinking: not available" if model_key in POOLED_HELD_OUT else model_key


def _slot_status(design: Mapping[str, Any], model_key: str, kind: str) -> str:
    return str(_mapping(_mapping(design.get("arms_by_model")).get(model_key)).get(kind, ROW_STATUS_ABSENT))


def _empty_slot(ax, x: float, width: float, status: str, *, top: float = 0.05, letters: Mapping[str, str] = SLOT_LETTER) -> None:
    """An EMPTY arm slot: a hatched stub with its status letter — a slot is never dropped and never a zero-height bar.
    ``letters``: the status -> letter table (the GSM8K figures pass their superset)."""
    ax.bar([x], [top], width=width, facecolor="none", edgecolor=MUTED, hatch=SLOT_HATCH, linewidth=0.5, zorder=2)
    ax.text(x, top + 0.012, letters.get(status, "×"), ha="center", va="bottom", fontsize=5.4, color=MUTED)


def panel_slots(report: Mapping[str, Any]) -> list[Arm]:
    """The arm SLOTS a figure lays out, in panel order: the registered arms the REPORT's design names (``design.arms`` — the
    registry of the analysis that wrote it), so a report written before an arm was registered lays out exactly the slots it
    had. Every figure's geometry (bar pitch offsets, the "k of n arms present" count) derives from THIS list, never from
    ``len(ARMS)``: a ninth registered arm must not shift the bars of a panel that does not carry it. An excluded arm
    (``figures.excluded_arms``) keeps its slot empty (the published behaviour) and is left out of the legend by the caller.
    A design naming no arms (none on record) falls back to the registry."""
    return design_arms(report.get("design"))


#: the CAPTION's word for each published arm, in panel order — the published prose kept byte for byte; a later-registered arm
#: contributes its figure label's head (:func:`panel_caption_head`)
EIGHTARM_CAPTION_NAME: Mapping[str, str] = {
    CELL_BARE: "none", CELL_PROMPTED: "prompted safety", CELL_PROMPTED_CONST: "prompted constitution", CELL_LORA_CTL2: "LoRA ctl2",
    CELL_LORA_MIX: "LoRA mix", CELL_OBJECT: "LoRA mix−ctl", CELL_ALPHASTEER: "AlphaSteer", CELL_ARDITI: "Arditi"}
_PANEL_NUMERAL: Mapping[int, str] = {3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def panel_numeral(n: int) -> str:
    """The caption's word for a slot count (``eight``, ``nine``, …; the digits beyond the table)."""
    return _PANEL_NUMERAL.get(int(n), str(int(n)))


def panel_caption_head(report: Mapping[str, Any]) -> str:
    """The arm panel's caption head, COUNTED AND NAMED from the arms the report registers (:func:`panel_slots`), never from
    ``len(ARMS)``: a report that does not carry a later-registered arm keeps its published caption word for word, and one
    that does says so instead of claiming a panel it is not drawing."""
    slots = panel_slots(report)
    word = panel_numeral(len(slots))
    names = ", ".join(EIGHTARM_CAPTION_NAME.get(a.cell, a.label.split(" (")[0]) for a in slots)
    return f"The {word}-arm panel (plan P5): per model, {word} FIXED arm slots in panel order ({names})"


def fig_eightarm_accuracy(report: Mapping[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    plt, _mcolors, Patch = T._mpl()
    numbers: list[dict[str, Any]] = []
    design, ea = report["design"], report["eightarm"]
    est = report.get("estimands") or {}
    excluded_arms = figure_excluded_arms(report)
    models = _shown_models(report, list(design["models"]))
    slots = panel_slots(report)                     # the geometry derives from the arms this report lays out, never len(ARMS)
    arms = [a for a in slots if a.cell not in excluded_arms]
    groups = [*models, POOLED_GROUP_LABEL]
    pitch = 0.105
    width = pitch * 0.86
    fig, ax = plt.subplots(figsize=(1.95 * len(groups) + 2.0, 4.4))
    _layout(fig, left=0.08)
    kinds_drawn: set[str] = set()
    for g, m in enumerate(models):
        for s, arm in enumerate(slots):
            if arm.cell in excluded_arms:
                continue
            x = g + (s - (len(slots) - 1) / 2) * pitch
            a = _mapping(_mapping(_mapping(_mapping(est.get(m)).get(MODE_PRIMARY)).get(BRANCH_PRIMARY)).get("acc_by_leg")).get(arm.kind)
            if a is None:
                _empty_slot(ax, x, width, _slot_status(design, m, arm.kind))
                continue
            kinds_drawn.add(arm.kind)
            ax.bar([x], [a["point"]], width=width, facecolor=KIND_COLOR[arm.kind], edgecolor="none", zorder=3)
            ax.plot([x, x], [a["ci_low"], a["ci_high"]], color=INK, lw=0.9, zorder=5)
            base = f"estimands.{m}.{MODE_PRIMARY}.{BRANCH_PRIMARY}.acc_by_leg.{arm.kind}"
            for f in ("point", "ci_low", "ci_high"):
                T._num_at(numbers, report, f"eightarm.acc.{f}[{m}/{arm.kind}]", f"{base}.{f}", a[f])
    g = len(models)
    for s, arm in enumerate(slots):
        if arm.cell in excluded_arms:
            continue
        x = g + (s - (len(slots) - 1) / 2) * pitch
        rec = _mapping(_mapping(ea.get("pooled_by_arm")).get(arm.kind))
        if rec.get("status") != POOLED_AVAILABLE:
            _empty_slot(ax, x, width, POOLED_UNAVAILABLE)
            continue
        p = rec[BRANCH_PRIMARY]
        ax.bar([x], [p["point"]], width=width, facecolor=KIND_COLOR[arm.kind], edgecolor=INK, linewidth=0.4, zorder=3)
        ax.plot([x, x], [p["ci_low"], p["ci_high"]], color=INK, lw=0.9, zorder=5)
        for f in ("point", "ci_low", "ci_high"):
            T._num_at(numbers, report, f"eightarm.pooled.{f}[{arm.kind}]", f"eightarm.pooled_by_arm.{arm.kind}.{BRANCH_PRIMARY}.{f}", p[f])
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([_eightarm_group_label(m) for m in models] + [f"{POOLED_GROUP_LABEL}\n({'/'.join(ea['panel_models'])})"], fontsize=6.8)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("conditional accuracy (scored items; primary branch)", fontsize=7.2)
    T._recessive(ax)
    handles = [Patch(facecolor=KIND_COLOR[a.kind], edgecolor="none", label=a.label) for a in arms]
    handles += [Patch(facecolor="none", edgecolor=MUTED, hatch=SLOT_HATCH, label=SLOT_LEGEND),
                plt.Line2D([], [], color=INK, lw=0.9, label="95 % bootstrap CI (conditional, primary branch); pooled bars outlined")]
    # review L6: the legend sits UNDER the axes (inside the axes it covered the empty slots and the hold-out footer at the
    # bottom left), the hold-out footer (plan P5) and the disclosure lines below it — all kept by the saver's tight bbox
    h_in = ax.get_position().height * fig.get_figheight()
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.45 / h_in), ncol=4, frameon=False, fontsize=5.4)
    footer = (f"pooled = equal-weight mean over {'/'.join(ea['panel_models'])}; held out: {', '.join(ea['held_out'])} "
              f"(thinking: not available); not in the panel: {', '.join(ea['not_in_panel'])}; no pooled verdict word")
    _footnotes_below(fig, ax, [footer, *_footnote_lines(report)], clearance_in=1.05)
    line = f"MATH-500 conditional accuracy (scored items) per arm and model, thinking ON at {CAP:,}; {len(kinds_drawn)} of {len(slots)} arms present"
    _title(fig, "Eight arms\n" + _with_selection(report, line))
    return fig, numbers


#: review M3: the served-geometry rule, stated once and carried on every record and caption that discloses a difference
GEOMETRY_RULE = ("served window × concurrency (L × S) is validated per row against the row's resolved budget and is NOT an identity "
                 "axis for a VALUE — a re-randomised sample of the same estimand at the same window is noise, not a second "
                 "instrument — so a contrast whose legs were served at different geometries is an admissible comparison of values, "
                 "disclosed; it IS decisive for bit identity (batch composition sets the bits under bf16 non-invariance), so no "
                 "byte-reproduction claim spans a geometry boundary and none is made")
#: the closing review: the ONE definition of what the primary estimand omits, stated identically wherever the
#: headline is defined — the eight-arm block's `headline_reading`, :data:`HEADLINE_RULE` (and through it the two forest
#: captions and the tables' rule) and the cell-accuracy caption. Being at the cap does NOT make a draw unscored: a draw
#: that closed its think block and ran to the cap in the answer is scored (production qwen LoRA mix: 49 of its 83 at-cap
#: draws are scored, 3 of them correct); the unscored draws are the ones the checker could not score. Revisions 1–4
#: defined the omitted draws by their at-cap status — false, retired; the test scans this module's source for it.
SCORING_DEFINITION = ("Primary (conditional) accuracy omits UNSCORED draws (those the checker could not score — capped inside an "
                      "unfinished think block or otherwise unterminated); scored-at-cap responses retain their scores. Score-as-is "
                      "counts only the unscored draws as wrong")
#: review M4: what the headline number IS, stated once
HEADLINE_RULE = (f"the headline effect is CONDITIONAL accuracy on scored items. {SCORING_DEFINITION}. The unscored draws leave the "
                 "denominator (each leg's own), so a leg with many unscored draws is read on fewer items; the score-as-is branch is "
                 "the sensitivity, filed beside the headline wherever it appears; where the two disagree in sign the contrast is "
                 "censoring-driven")


def _kind_of_leg_key(k: str) -> str:
    cell = str(k).split("/")[1] if str(k).count("/") == 2 else str(k)
    return next((a.kind for a in ARMS if a.cell == cell), cell)


def _leg_words(k: str) -> str:
    """``<model> <kind>`` of a leg key, plus `` [<view>]`` on a non-main view (E3) — the disclosure spelling of a leg."""
    return f"{k.split('/')[0]} {_kind_of_leg_key(k)}{_view_tag(leg_key_view(k))}"


def _comparison_words(name: str) -> str:
    """``<model>`` of a comparison name (``<model>/<mode>[@view]``), plus `` [<view>]`` on a non-main view (E3)."""
    return f"{name.split('/')[0]}{_view_tag(leg_key_view(name))}"


def served_geometry_notes(report: Mapping[str, Any]) -> dict[str, Any]:
    """Per primary comparison that ran, every leg's served L × S (the manifest's served block) and producer tree,
    whether the geometries differ within the contrast, and the windows at which two legs differ in CONCURRENCY alone (the
    qwen case: the published legs served under the provisional half budget at S 20, the later legs at the lifted budget's
    S 40 — a single-run cross-geometry comparison). Ints / strings / bools."""
    out: dict[str, Any] = {}
    legs = _mapping(report.get("legs"))
    for c in report.get("comparisons", []):
        if not c.get("primary") or c.get("estimands") is None:
            continue
        by_kind: dict[str, Any] = {}
        for kind, k in c["leg_keys"].items():
            leg = _mapping(legs.get(k))
            served = _mapping(_mapping(leg.get("identity")).get("served"))
            by_kind[kind] = {"max_model_len": _int_field(served.get("max_model_len")), "max_num_seqs": _int_field(served.get("max_num_seqs")),
                             "tree": _mapping(leg.get("producer")).get("tree_sha")}
        distinct = {(v["max_model_len"], v["max_num_seqs"]) for v in by_kind.values()}
        by_window: dict[Any, set] = {}
        for v in by_kind.values():
            by_window.setdefault(v["max_model_len"], set()).add(v["max_num_seqs"])
        out[str(c["name"])] = {"model_key": c["model_key"], "by_kind": by_kind, "differs": len(distinct) > 1,
                               "n_distinct": len(distinct),
                               "same_window_differs_in_concurrency": sorted(L for L, ss in by_window.items() if len(ss) > 1
                                                                            and L is not None),
                               "rule": GEOMETRY_RULE,
                               # E3: a non-main view's comparison names its view (a main record is unchanged)
                               **({"view": comp_view(c)} if comp_view(c) != VIEW_MAIN else {})}
    return out


def _geometry_groups(rec: Mapping[str, Any]) -> list[tuple[str, list[str]]]:
    """``[("L×S", [kinds in panel order]), ...]`` in first-seen panel order."""
    groups: dict[str, list[str]] = {}
    for kind in ARM_KINDS:
        v = _mapping(rec.get("by_kind")).get(kind)
        if v is None:
            continue
        groups.setdefault(f"{v['max_model_len']}×{v['max_num_seqs']}", []).append(kind)
    return list(groups.items())


def _geometry_footnote(report: Mapping[str, Any]) -> str:
    """ONE short line: the models whose contrast spans more than one served geometry, each with its L×S groups in panel
    order (the caption and the tables carry the per-arm detail); ``""`` when none is shown."""
    parts = []
    for rec in served_geometry_notes(report).values():
        if not rec["differs"] or rec["model_key"] in figure_excluded_families(report):
            continue
        parts.append(f"{rec['model_key']}{_view_tag(str(rec.get('view') or VIEW_MAIN))} " + " | ".join(g for g, _ in _geometry_groups(rec)))
    return ("served L×S differs within contrast: " + " · ".join(parts) + " (per-row validated; values comparable, no byte claim)") if parts else ""


#: second reader M-b, CORRECTED by the second reader's round 2 (M2): the CENSORING DIFFERENTIAL of a contrast is the
#: arm's UNSCORED share minus the bare anchor's — `n_censored / n_items` per leg over the contrast's own population, the items
#: the conditional estimand drops — with its DIRECTION in words, so that "censoring-driven" cannot be read as censoring in the
#: ARM when it is the anchor's. Revision 3 derived it from the AT-CAP counts, but at-cap is not censored: a scored draw can
#: finish at the cap (gemma prompted − bare: 500 / 500 scored on both legs, 0 unscored, ONE scored at-cap draw on bare read
#: "bare more censored"; qwen LoRA mix: 34 vs 71 unscored — the anchor the more censored leg by 7.4 pp — read "+1.0 pp, arm
#: more censored" from 83 vs 78 at cap). The at-cap quantity stays, renamed NEUTRALLY (`Δ at-cap`,
#: :data:`AT_CAP_DIFFERENTIAL_FIELD`, :data:`AT_CAP_DIFFERENTIAL_RULE`) as a diagnostic beside it — never as censoring.
CENSORING_DIFFERENTIAL_FIELD = "arm_unscored_share_minus_bare_unscored_share"
CENSORING_DIRECTION_BARE, CENSORING_DIRECTION_ARM, CENSORING_DIRECTION_EQUAL = "bare anchor more censored", "arm more censored", "equally censored"
AT_CAP_DIFFERENTIAL_FIELD = "arm_at_cap_share_minus_bare_at_cap_share"
AT_CAP_DIRECTION_BARE, AT_CAP_DIRECTION_ARM, AT_CAP_DIRECTION_EQUAL = "bare anchor more often at cap", "arm more often at cap", "equal at-cap shares"
#: the exact decomposition (M2) that REPLACES revision 3's "a negative differential lifts the arm" rule, which qwen LoRA mix
#: contradicted (−19.02 → −11.00: the arm LIFTED although its at-cap count is the larger)
CENSORING_DIFFERENTIAL_RULE = (
    "the censoring differential is the arm's unscored share minus the bare anchor's — n_censored / n_items per leg over the contrast's own "
    "population, the items the conditional estimand drops — with its direction in words; it is NOT the at-cap differential (at-cap draws over "
    "the leg's draws, reported beside it as a neutral diagnostic: a scored draw can finish at the cap, so at-cap is not censored). Per leg, "
    "score-as-is accuracy = conditional accuracy × scored share, because n_correct / n_items = (n_correct / n_scored) × (n_scored / n_items); "
    "so the branch shift of a contrast, score-as-is (arm − bare) − conditional (arm − bare), equals bare conditional × bare unscored share − "
    "arm conditional × arm unscored share, with unscored share = 1 − scored share = n_censored / n_items: the shift LIFTS the arm when the "
    "anchor's product is the larger and LOWERS it when the arm's is — the two products decide, not the differential's sign alone. Every "
    "row prints its shift; the record and the caption print the two products and check the identity against the filed points (D59; "
    "second reader round 2, M2)")
AT_CAP_DIFFERENTIAL_RULE = ("Δ at-cap is the arm's at-cap share minus the bare anchor's (n_at_cap / n over the leg's draws — the summary's own "
                            "counts, the raw draw, which the D48 mask does not touch): a neutral diagnostic of where the cap bound, NOT a "
                            "censoring measure — an at-cap draw that closed its think block is scored")
#: the six panels that read the controls table — the RAW draw — and not an estimand: the mask does not reach them
#: and each says so when a masked draw's family is shown
RAW_DRAW_PANELS: tuple[str, ...] = ("final_m5t_atcap_thinksplit", "final_m5t_atcap_share", "final_m5t_atcap_loopsplit",
                                    "final_m5t_no_reasoning_share", "final_m5t_ledger", "final_m5t_completion_tokens")
#: the tolerance (pp) within which the decomposition must reproduce the filed branch shift for the identity to HOLD
SHIFT_IDENTITY_TOL_PP = 1e-9
#: the at-cap direction that AGREES with each censoring direction (the same leg named on both lines)
_AT_CAP_DIRECTION_OF: Mapping[str, str] = {CENSORING_DIRECTION_BARE: AT_CAP_DIRECTION_BARE, CENSORING_DIRECTION_ARM: AT_CAP_DIRECTION_ARM,
                                           CENSORING_DIRECTION_EQUAL: AT_CAP_DIRECTION_EQUAL}


def _direction(diff: float | None, *, bare: str, arm: str, equal: str) -> str | None:
    if diff is None:
        return None
    return equal if diff == 0 else (arm if diff > 0 else bare)


def _censoring_direction(diff: float | None) -> str | None:
    """The censoring differential's direction in words (arm unscored share − bare unscored share)."""
    return _direction(diff, bare=CENSORING_DIRECTION_BARE, arm=CENSORING_DIRECTION_ARM, equal=CENSORING_DIRECTION_EQUAL)


def _at_cap_direction(diff: float | None) -> str | None:
    """The at-cap differential's direction in NEUTRAL words (M2: at-cap is not censored)."""
    return _direction(diff, bare=AT_CAP_DIRECTION_BARE, arm=AT_CAP_DIRECTION_ARM, equal=AT_CAP_DIRECTION_EQUAL)


def _share(num: Any, den: Any) -> float | None:
    return (num / den) if _numeric_value(num) and _numeric_value(den) and den else None


def _diff(a: float | None, b: float | None) -> float | None:
    return (a - b) if a is not None and b is not None else None


def _shift_text(den: Mapping[str, Any]) -> str | None:
    """The branch shift written out (M2): ``<shift> pp = bare conditional × bare unscored share − arm conditional × arm unscored
    share`` with the four factors; ``None`` when a conditional accuracy is undefined (a leg with no scored item)."""
    if den.get("shift_pp") is None or den.get("arm_conditional") is None or den.get("bare_conditional") is None:
        return None
    return (f"{den['shift_pp']:+.2f} pp = {den['bare_conditional']:.3f} × {den['bare_unscored_share']:.3f} − "
            f"{den['arm_conditional']:.3f} × {den['arm_unscored_share']:.3f}")


def contrast_denominators(report: Mapping[str, Any], comp_index: int, name: str) -> dict[str, Any]:
    """The DENOMINATORS of one ``<arm> − bare`` contrast of ``report["comparisons"][comp_index]`` — the arm's and the
    anchor's scored, unscored and correct counts over the CONTRAST'S OWN population (the masked contrast's when a mask is
    scoped to it, so a masked leg's unscored count is its scoped ``n_censored``, never the raw summary's), the CENSORING
    differential (:data:`CENSORING_DIFFERENTIAL_FIELD`: arm unscored share − bare unscored share, ``n_censored / n_items``
    per leg) with its direction in words, the two legs' at-cap counts and draws (the summary's own counts: the RAW draw,
    which the mask does not touch) with the at-cap differential (:data:`AT_CAP_DIFFERENTIAL_FIELD`, a neutral diagnostic)
    and ITS direction, whether the two directions agree, the exact branch-shift decomposition
    (:data:`CENSORING_DIFFERENTIAL_RULE`: ``shift_pp`` = the filed score-as-is point − the filed conditional point;
    ``shift_decomposition_pp`` = bare conditional × bare unscored share − arm conditional × arm unscored share;
    ``shift_identity_holds`` when they agree within :data:`SHIFT_IDENTITY_TOL_PP` and each filed point is the plain
    difference of the two legs' accuracies), and the report paths every number resolves to (``paths``, for the numbers
    sidecar). Refuses (``ValueError``) a record whose scored + unscored is not the contrast's population — the shares would
    not be over the population they claim."""
    c = report["comparisons"][comp_index]
    signs = comp_contrasts(c)[name]
    kind = plus_kind(signs)
    est = c["estimands"]
    prim = est[BRANCH_PRIMARY]
    masked = _mapping(_mapping(c.get("masked_contrasts")).get(name))
    scoped = bool(masked)
    acc = _mapping(masked.get("acc_by_leg")).get(BRANCH_PRIMARY) if scoped else prim["acc_by_leg"]
    acc_path = (f"comparisons.{comp_index}.masked_contrasts.{name}.acc_by_leg.{BRANCH_PRIMARY}" if scoped
                else f"comparisons.{comp_index}.estimands.{BRANCH_PRIMARY}.acc_by_leg")
    n_items = masked.get("n_items") if scoped else _mapping(prim.get("n_items_by_contrast")).get(name, prim["n_items"])
    n_items_path = (f"comparisons.{comp_index}.masked_contrasts.{name}.n_items" if scoped
                    else f"comparisons.{comp_index}.estimands.{BRANCH_PRIMARY}.n_items")
    a, b = _mapping(acc.get(kind)), _mapping(acc.get(KIND_BARE))
    for lab, d in (("arm", a), ("bare", b)):
        if not (_numeric_value(d.get("n_scored")) and _numeric_value(d.get("n_censored"))) or d["n_scored"] + d["n_censored"] != n_items:
            raise ValueError(f"contrast {name!r} of comparison {comp_index}: the {lab} leg's scored + unscored "
                             f"({d.get('n_scored')} + {d.get('n_censored')}) is not the contrast's population {n_items} — its shares cannot be read")
    idx = _ctl_index(report)
    ia, ib = idx[c["leg_keys"][kind]], idx[c["leg_keys"][KIND_BARE]]
    sa, sb = _mapping(report["controls"][ia].get("summary")), _mapping(report["controls"][ib].get("summary"))
    a_cap, b_cap = _share(sa.get("n_at_cap"), sa.get("n")), _share(sb.get("n_at_cap"), sb.get("n"))
    cap_diff = _diff(a_cap, b_cap)
    a_un, b_un = a["n_censored"] / n_items, b["n_censored"] / n_items
    a_sc, b_sc = a["n_scored"] / n_items, b["n_scored"] / n_items
    cens = a_un - b_un
    a_cond, b_cond = _share(a.get("n_correct"), a["n_scored"]), _share(b.get("n_correct"), b["n_scored"])
    a_asis, b_asis = _share(a.get("n_correct"), n_items), _share(b.get("n_correct"), n_items)
    p, q = est[BRANCH_PRIMARY]["stats"][name]["point"], est[BRANCH_SCORE_AS_IS]["stats"][name]["point"]
    shift = q - p
    bare_term = None if b_cond is None else b_cond * (1.0 - b_sc)
    arm_term = None if a_cond is None else a_cond * (1.0 - a_sc)
    decomposition = _diff(bare_term, arm_term)
    holds: bool | None = None
    if decomposition is not None and a_asis is not None and b_asis is not None:
        holds = (abs(_pp(shift) - _pp(decomposition)) <= SHIFT_IDENTITY_TOL_PP
                 and abs(_pp(p) - _pp(a_cond - b_cond)) <= SHIFT_IDENTITY_TOL_PP
                 and abs(_pp(q) - _pp(a_asis - b_asis)) <= SHIFT_IDENTITY_TOL_PP)
    cens_dir, cap_dir = _censoring_direction(cens), _at_cap_direction(cap_diff)
    den: dict[str, Any] = {
        "arm": kind, "n_items": n_items, "scoped_to_masked_contrast": scoped,
        "arm_n_scored": a["n_scored"], "bare_n_scored": b["n_scored"], "arm_n_unscored": a["n_censored"], "bare_n_unscored": b["n_censored"],
        "arm_n_correct": a.get("n_correct"), "bare_n_correct": b.get("n_correct"),
        "arm_n_unscored_raw_draw": sa.get("n_unscored"), "bare_n_unscored_raw_draw": sb.get("n_unscored"),
        "arm_n_draws": sa.get("n"), "bare_n_draws": sb.get("n"), "arm_n_at_cap": sa.get("n_at_cap"), "bare_n_at_cap": sb.get("n_at_cap"),
        # the censoring differential (M2): unscored shares over the contrast's population
        "arm_scored_share": a_sc, "bare_scored_share": b_sc, "arm_unscored_share": a_un, "bare_unscored_share": b_un,
        CENSORING_DIFFERENTIAL_FIELD: cens, "censoring_differential_pp": _pp(cens), "censoring_direction": cens_dir,
        # the at-cap differential (neutral diagnostic): at-cap draws over the leg's draws
        "arm_at_cap_share": a_cap, "bare_at_cap_share": b_cap,
        AT_CAP_DIFFERENTIAL_FIELD: cap_diff, "at_cap_differential_pp": None if cap_diff is None else _pp(cap_diff), "at_cap_direction": cap_dir,
        "directions_agree": None if cap_dir is None else (cap_dir == _AT_CAP_DIRECTION_OF[cens_dir]),
        # the exact branch-shift decomposition (M2)
        "arm_conditional": a_cond, "bare_conditional": b_cond, "arm_score_as_is": a_asis, "bare_score_as_is": b_asis,
        "conditional_pp": _pp(p), "score_as_is_pp": _pp(q), "shift_pp": _pp(shift),
        "shift_bare_term_pp": None if bare_term is None else _pp(bare_term), "shift_arm_term_pp": None if arm_term is None else _pp(arm_term),
        "shift_decomposition_pp": None if decomposition is None else _pp(decomposition), "shift_identity_holds": holds,
        "paths": {"arm_n_scored": f"{acc_path}.{kind}.n_scored", "bare_n_scored": f"{acc_path}.{KIND_BARE}.n_scored",
                  "arm_n_unscored": f"{acc_path}.{kind}.n_censored", "bare_n_unscored": f"{acc_path}.{KIND_BARE}.n_censored",
                  "arm_n_correct": f"{acc_path}.{kind}.n_correct", "bare_n_correct": f"{acc_path}.{KIND_BARE}.n_correct", "n_items": n_items_path,
                  "arm_n_at_cap": f"controls.{ia}.summary.n_at_cap", "bare_n_at_cap": f"controls.{ib}.summary.n_at_cap",
                  "arm_n_draws": f"controls.{ia}.summary.n", "bare_n_draws": f"controls.{ib}.summary.n",
                  "conditional_point": f"comparisons.{comp_index}.estimands.{BRANCH_PRIMARY}.stats.{name}.point",
                  "score_as_is_point": f"comparisons.{comp_index}.estimands.{BRANCH_SCORE_AS_IS}.stats.{name}.point"}}
    den["shift_text"] = _shift_text(den)
    return den


def _primary_comparison_index(report: Mapping[str, Any], model_key: str, mode: str) -> int | None:
    for j, c in enumerate(report.get("comparisons", [])):
        if (c.get("model_key") == model_key and c.get("mode") == mode and c.get("primary") and c.get("estimands") is not None
                and comp_view(c) == VIEW_MAIN):
            return j
    return None


#: the per-model fields a pooled arm averages with equal weights (the pooled estimand's weights)
_DENOMINATOR_MEAN_FIELDS: tuple[str, ...] = ("arm_at_cap_share", "bare_at_cap_share", AT_CAP_DIFFERENTIAL_FIELD,
                                             "arm_unscored_share", "bare_unscored_share", CENSORING_DIFFERENTIAL_FIELD,
                                             "shift_pp", "shift_decomposition_pp")
_DENOMINATOR_TRIPLE_FIELDS: tuple[str, ...] = ("arm_n_scored", "bare_n_scored", "n_items")


def pooled_denominators(report: Mapping[str, Any], kind: str) -> dict[str, Any] | None:
    """Second reader M-b, the pooled arms, corrected by round 2 (M2 / M3): each panel model's contrast denominators
    (:func:`contrast_denominators` of that model's primary comparison — the same legs and the same scoped populations the
    pooled pair reads), the per-model SCORED TRIPLES the pooled row prints (``scored_by_model``: arm scored / bare scored /
    population, M3), and the pooled arm's own shares, both differentials and the branch shift as the EQUAL-WEIGHT MEAN over
    the panel models (the pooled estimand's weights) with both directions in words; ``shift_identity_holds`` when every
    model's identity holds and the mean of the per-model shifts reproduces the pooled record's own branch shift. ``None``
    when the arm is not pooled."""
    ea = _mapping(report.get("eightarm"))
    rec = _mapping(_mapping(ea.get("pooled_by_arm")).get(kind))
    if rec.get("status") != POOLED_AVAILABLE or kind == KIND_BARE:
        return None
    name = contrast_name(kind)
    per: dict[str, dict[str, Any]] = {}
    for m in ea.get("panel_models") or ():
        j = _primary_comparison_index(report, m, str(ea.get("mode")))
        if j is None or name not in comp_contrasts(report["comparisons"][j]):
            return None
        per[m] = contrast_denominators(report, j, name)
    if not per:
        return None
    means: dict[str, float | None] = {}
    for f in _DENOMINATOR_MEAN_FIELDS:
        vals = [per[m][f] for m in per]
        means[f] = (sum(vals) / len(vals)) if all(v is not None for v in vals) else None
    cens, cap = means[CENSORING_DIFFERENTIAL_FIELD], means[AT_CAP_DIFFERENTIAL_FIELD]
    cens_dir, cap_dir = _censoring_direction(cens), _at_cap_direction(cap)
    pooled_shift = _pp(rec[BRANCH_SCORE_AS_IS]["minus_bare"]["point"] - rec[BRANCH_PRIMARY]["minus_bare"]["point"])
    holds = (all(per[m]["shift_identity_holds"] is True for m in per) and means["shift_pp"] is not None
             and abs(means["shift_pp"] - pooled_shift) <= SHIFT_IDENTITY_TOL_PP)
    return {"arm": kind, "models": list(per), "by_model": {m: {k: v for k, v in d.items() if k != "paths"} for m, d in per.items()},
            "paths_by_model": {m: d["paths"] for m, d in per.items()},
            "scored_by_model": {m: {f: d[f] for f in _DENOMINATOR_TRIPLE_FIELDS} for m, d in per.items()},
            **means, "censoring_differential_pp": None if cens is None else _pp(cens), "censoring_direction": cens_dir,
            "at_cap_differential_pp": None if cap is None else _pp(cap), "at_cap_direction": cap_dir,
            "directions_agree": None if (cens_dir is None or cap_dir is None) else (
                cap_dir == _AT_CAP_DIRECTION_OF[cens_dir]),
            "pooled_shift_pp": pooled_shift, "shift_identity_holds": bool(holds),
            "pooled_rule": ("the pooled shares, both differentials and the branch shift are the equal-weight mean over the panel models of the "
                            "per-model values (the pooled estimand's weights); the scored triples are each model's arm scored / bare scored / "
                            "population over the contrast's own population")}


def conditional_vs_score_as_is_notes(report: Mapping[str, Any]) -> dict[str, Any]:
    """Review M4 + second reader M-b / M-c, corrected by round 2 (M2 / M3): per primary contrast and per pooled arm, the
    headline (conditional accuracy on SCORED items) beside the score-as-is sensitivity in pp with 95 % CIs, whether the two
    branches agree in sign, the filed word (a pooled arm files none — :data:`POOLED_WORD_RULE`), and the DENOMINATORS
    (:func:`contrast_denominators`): the arm's and the anchor's scored / unscored / correct counts over the contrast's own
    population (a masked leg's unscored count is its scoped ``n_censored``; the raw draw's beside as ``*_raw_draw``), the
    censoring differential (unscored shares) and the at-cap differential (at-cap shares) each with its direction, and the
    exact branch-shift decomposition (:data:`CENSORING_DIFFERENTIAL_RULE`, :data:`AT_CAP_DIFFERENTIAL_RULE`); per pooled arm
    the per-model values, the scored triples and the equal-weight means."""
    out: dict[str, Any] = {"rule": HEADLINE_RULE, "censoring_rule": CENSORING_DIFFERENTIAL_RULE, "at_cap_rule": AT_CAP_DIFFERENTIAL_RULE,
                           "by_comparison": {}, "pooled": {}, "pooled_word_rule": POOLED_WORD_RULE}
    for j, c in enumerate(report.get("comparisons", [])):
        if not c.get("primary") or c.get("estimands") is None:
            continue
        est = c["estimands"]
        recs: dict[str, Any] = {}
        for name in comp_contrasts(c):
            p, q = est[BRANCH_PRIMARY]["stats"][name], est[BRANCH_SCORE_AS_IS]["stats"][name]
            den = contrast_denominators(report, j, name)
            recs[name] = {"arm": den["arm"], "conditional_pp": _pp(p["point"]), "conditional_ci_pp": [_pp(p["ci_low"]), _pp(p["ci_high"])],
                          "score_as_is_pp": _pp(q["point"]), "score_as_is_ci_pp": [_pp(q["ci_low"]), _pp(q["ci_high"])],
                          "signs_agree": int(p["point_sign"]) == int(q["point_sign"]),
                          **{k: v for k, v in den.items() if k not in ("arm", "paths", "conditional_pp", "score_as_is_pp")},
                          "word": _mapping(_mapping(c.get("verdicts")).get(name)).get("word")}
        out["by_comparison"][str(c["name"])] = recs
    for kind, rec in _mapping(_mapping(report.get("eightarm")).get("pooled_by_arm")).items():
        if rec.get("status") != POOLED_AVAILABLE or kind == KIND_BARE:
            continue
        p, q = rec[BRANCH_PRIMARY]["minus_bare"], rec[BRANCH_SCORE_AS_IS]["minus_bare"]
        pden = pooled_denominators(report, kind) or {}
        out["pooled"][kind] = {"conditional_pp": _pp(p["point"]), "conditional_ci_pp": [_pp(p["ci_low"]), _pp(p["ci_high"])],
                               "score_as_is_pp": _pp(q["point"]), "score_as_is_ci_pp": [_pp(q["ci_low"]), _pp(q["ci_high"])],
                               "signs_agree": rec.get("branches_agree_in_sign"), "n_items_by_model": rec.get("n_items_by_model"),
                               **{k: v for k, v in pden.items() if k not in ("arm", "paths_by_model")},
                               "word": None}
    return out


DENOMINATOR_FONTSIZE = 5.0
ROW_PITCH_IN = 0.46            # the forest rows' pitch: the label on the tick, two (a contrast) or three (a pooled arm) lines under it
#: the top of each denominator line under a row's label, in row units (label at 0; a 5-pt line is ~0.15 of a 0.46-in row)
DENOMINATOR_LINE_OFFSETS: tuple[float, ...] = (0.15, 0.33, 0.51)


def _short_model(model_key: str) -> str:
    """The model key's leading letters (``gemma4e4b`` -> ``gemma``) for a pooled row's per-model triples; the full keys are in
    the row's label, in the same order."""
    m = re.match(r"[A-Za-z]+", model_key)
    return m.group(0) if m else model_key


def _fmt_pp(v: float | None, digits: int = 1) -> str:
    return "n/a" if v is None else f"{v:+.{digits}f}"


def _denominator_lines(den: Mapping[str, Any]) -> tuple[str, str]:
    """The TWO lines a contrast row carries under its label (M2): (1) the arm's and the anchor's scored and unscored counts over
    the contrast's population and the CENSORING differential with its direction; (2) their at-cap counts over the leg's draws,
    the neutral at-cap differential with its direction, and the branch shift (score-as-is − conditional); an identity that
    does not hold says so on the line."""
    cens, cap = den.get("censoring_differential_pp"), den.get("at_cap_differential_pp")
    draws = den["arm_n_draws"] if den["arm_n_draws"] == den["bare_n_draws"] else f"{den['arm_n_draws']} · {den['bare_n_draws']}"
    first = (f"arm {den['arm_n_scored']} · bare {den['bare_n_scored']} scored of {den['n_items']}; unscored {den['arm_n_unscored']} · "
             f"{den['bare_n_unscored']}; " + (f"Δ censoring {cens:+.1f} pp ({den['censoring_direction']})" if cens is not None else "Δ censoring n/a"))
    shift = (f"shift {den['shift_pp']:+.2f} pp" if den.get("shift_pp") is not None else "shift n/a") + (
        "" if den.get("shift_identity_holds") in (True, None) else " — DECOMPOSITION DOES NOT REPRODUCE IT")
    second = (f"at cap {den['arm_n_at_cap']} · {den['bare_n_at_cap']} of {draws} draws; "
              + (f"Δ at-cap {cap:+.1f} pp ({den['at_cap_direction']})" if cap is not None else "Δ at-cap n/a (a leg carries no at-cap count)")
              + f"; {shift}")
    return first, second


def _pooled_denominator_lines(pden: Mapping[str, Any]) -> tuple[str, str, str]:
    """The THREE lines a pooled row carries (M3 + M2): (1) the per-model scored triples (arm / bare scored, the population); (2)
    the equal-weight mean censoring differential with its direction and the per-model values; (3) the mean at-cap differential
    with its direction, the per-model values and the mean shift."""
    models = list(pden["models"])
    trip = pden["scored_by_model"]
    pops = [trip[m]["n_items"] for m in models]
    pop = f"{pops[0]} each" if len(set(pops)) == 1 else " · ".join(str(n) for n in pops)
    first = "scored arm/bare: " + " · ".join(f"{_short_model(m)} {trip[m]['arm_n_scored']}/{trip[m]['bare_n_scored']}" for m in models) + f" of {pop}"
    per = " / ".join(_fmt_pp(pden["by_model"][m].get("censoring_differential_pp")) for m in models)
    cens = pden.get("censoring_differential_pp")
    second = (f"Δ censoring mean {cens:+.1f} pp ({pden['censoring_direction']})" if cens is not None else "Δ censoring mean n/a") + f" · per model {per}"
    per_cap = " / ".join(_fmt_pp(pden["by_model"][m].get("at_cap_differential_pp")) for m in models)
    cap = pden.get("at_cap_differential_pp")
    shift = pden.get("shift_pp")
    third = ((f"Δ at-cap mean {cap:+.1f} pp ({pden['at_cap_direction']})" if cap is not None else "Δ at-cap mean n/a") + f" · per model {per_cap}"
             + (f"; shift mean {shift:+.2f} pp" if shift is not None else "; shift n/a")
             + ("" if pden.get("shift_identity_holds") else " — DECOMPOSITION DOES NOT REPRODUCE IT"))
    return first, second, third


def _denominator_text(ax, y: float, lines: Sequence[str]) -> None:
    """The denominator lines under a row's label, right-aligned against the axes' left edge."""
    for off, line in zip(DENOMINATOR_LINE_OFFSETS, lines):
        ax.text(-0.02, y - off, line, transform=ax.get_yaxis_transform(), ha="right", va="top", fontsize=DENOMINATOR_FONTSIZE, color=INK2,
                clip_on=False)


def _denominator_numbers(numbers: list[dict[str, Any]], report: Mapping[str, Any], label: str, den: Mapping[str, Any]) -> None:
    """The denominators as comparator-sidecar records (L1): every count the leaf it is, every share, both differentials and the
    shift decomposition a named transform over the leaves it reads."""
    P = den["paths"]
    for f in ("arm_n_scored", "bare_n_scored", "arm_n_unscored", "bare_n_unscored", "arm_n_correct", "bare_n_correct", "n_items",
              "arm_n_at_cap", "bare_n_at_cap", "arm_n_draws", "bare_n_draws"):
        T._num_at(numbers, report, f"{label}.{f}", P[f], den[f])
    T._num_at(numbers, report, f"{label}.arm_unscored_share", [P["arm_n_unscored"], P["n_items"]], den["arm_unscored_share"], transform="n_censored / n_items")
    T._num_at(numbers, report, f"{label}.bare_unscored_share", [P["bare_n_unscored"], P["n_items"]], den["bare_unscored_share"], transform="n_censored / n_items")
    T._num_at(numbers, report, f"{label}.{CENSORING_DIFFERENTIAL_FIELD}", [P["arm_n_unscored"], P["bare_n_unscored"], P["n_items"]],
              den[CENSORING_DIFFERENTIAL_FIELD],
              transform="arm n_censored / n_items − bare n_censored / n_items (the censoring differential; negative = the bare anchor more censored)")
    T._num_at(numbers, report, f"{label}.arm_at_cap_share", [P["arm_n_at_cap"], P["arm_n_draws"]], den["arm_at_cap_share"], transform="n_at_cap / n")
    T._num_at(numbers, report, f"{label}.bare_at_cap_share", [P["bare_n_at_cap"], P["bare_n_draws"]], den["bare_at_cap_share"], transform="n_at_cap / n")
    T._num_at(numbers, report, f"{label}.{AT_CAP_DIFFERENTIAL_FIELD}", [P["arm_n_at_cap"], P["arm_n_draws"], P["bare_n_at_cap"], P["bare_n_draws"]],
              den[AT_CAP_DIFFERENTIAL_FIELD], transform="arm n_at_cap / n − bare n_at_cap / n (the at-cap differential, a neutral diagnostic; not censoring)")
    T._num_at(numbers, report, f"{label}.shift_pp", [P["score_as_is_point"], P["conditional_point"]], den["shift_pp"],
              transform="100 × (score-as-is point − conditional point)")
    T._num_at(numbers, report, f"{label}.shift_decomposition_pp",
              [P["bare_n_correct"], P["bare_n_scored"], P["bare_n_unscored"], P["arm_n_correct"], P["arm_n_scored"], P["arm_n_unscored"], P["n_items"]],
              den["shift_decomposition_pp"],
              transform="100 × (bare n_correct / n_scored × bare n_censored / n_items − arm n_correct / n_scored × arm n_censored / n_items)")


def _pooled_denominator_numbers(numbers: list[dict[str, Any]], report: Mapping[str, Any], label: str, pden: Mapping[str, Any]) -> None:
    cap_paths = [p for m in pden["models"] for p in (pden["paths_by_model"][m]["arm_n_at_cap"], pden["paths_by_model"][m]["arm_n_draws"],
                                                      pden["paths_by_model"][m]["bare_n_at_cap"], pden["paths_by_model"][m]["bare_n_draws"])]
    cens_paths = [p for m in pden["models"] for p in (pden["paths_by_model"][m]["arm_n_unscored"], pden["paths_by_model"][m]["bare_n_unscored"],
                                                       pden["paths_by_model"][m]["n_items"])]
    for m in pden["models"]:
        d = {**pden["by_model"][m], "paths": pden["paths_by_model"][m]}
        _denominator_numbers(numbers, report, f"{label}.{m}", d)
    T._num_at(numbers, report, f"{label}.{AT_CAP_DIFFERENTIAL_FIELD}", cap_paths, pden[AT_CAP_DIFFERENTIAL_FIELD],
              transform="equal-weight mean over the panel models of (arm n_at_cap / n − bare n_at_cap / n)")
    T._num_at(numbers, report, f"{label}.{CENSORING_DIFFERENTIAL_FIELD}", cens_paths, pden[CENSORING_DIFFERENTIAL_FIELD],
              transform="equal-weight mean over the panel models of (arm n_censored / n_items − bare n_censored / n_items)")


def d48_notes(report: Mapping[str, Any]) -> dict[str, Any]:
    """The disclosure facts a figure footnote, the captions and the tables carry (strings / ints / lists — B1): the
    primary-leg ``eos_in_thoughts`` draws masked from the filed contrasts (per leg: n and item ids; per arm: n) with the
    scope rule, the comparisons whose server default sampling differs LITERALLY between trees while the effective
    decode is equal (the model and the per-leg override dict), the served geometry per contrast and the
    headline beside its score-as-is sensitivity."""
    masked = _mapping(_mapping(_mapping(report.get("gates")).get("split_anomalies")).get("primary_masked"))
    by_arm: dict[str, int] = {}
    for k, rec in masked.items():
        kind = next((a.kind for a in ARMS if a.cell == str(k).split("/")[1]), str(k).split("/")[1]) if str(k).count("/") == 2 else k
        by_arm[kind] = by_arm.get(kind, 0) + int(rec.get("n_masked") or 0)
    literal: dict[str, Any] = {}
    for c in report.get("comparisons", []):
        comp = _mapping(c.get("compatibility"))
        if comp.get("server_settings_literal_equal") is False and comp.get("server_settings_equal") is True:
            literal[str(c["name"])] = {"difference": comp.get("server_settings_literal_difference"),
                                                       "generation_config_override_by_leg": comp.get("generation_config_override_by_leg")}
    return {"masked_primary": {"by_leg": {k: {"n_masked": v["n_masked"], "items": [it["item_id"] for it in v["items"]]} for k, v in masked.items()},
                               "by_arm": by_arm, "n_masked": sum(by_arm.values()),
                               "by_contrast": _mapping(report.get("masked_primary")).get("by_contrast") or {},
                               "scope_rule": MASK_SCOPE_RULE if by_arm else None},
            "server_settings_literal_differs": literal,
            "served_geometry": served_geometry_notes(report),
            "conditional_vs_score_as_is": conditional_vs_score_as_is_notes(report),
            "liveness_evidence": liveness_evidence_notes(report)}


disclosure_notes = d48_notes

#: math-eight-arms-extension (class E): the PAPER CONSEQUENCE of a fitted cell admitted by the logprob verdict with no
#: token bite observable — stated once, carried per row with the model and the arm filled in
LIVENESS_CONSEQUENCE_RULE = ("a fitted cell whose predicate (C) was admitted by the CONCLUSIVE logprob liveness verdict (D95): the object "
                             "is live at its operating point yet the render probe's greedy reply is byte-identical to the base route's — "
                             "no token bite was observable; bears on how much defence the arm buys at that dose. Reported on the row, "
                             "never hidden and never a refusal; identity (object == pin) was checked as before")


def liveness_evidence_notes(report: Mapping[str, Any]) -> dict[str, Any]:
    """The row-level fact of every leg admitted (read off the report's provenance rows, which carry
    ``certification.liveness_evidence``): the ruling's sentence and the paper consequence for that model and arm —
    "<model> <arm> at its operating point does not move a single token of the render probe — bears on how much defence
    the arm buys at that dose". ``by_leg`` empty and ``rule`` None when no leg was admitted this way."""
    cells = _mapping(_mapping(report.get("provenance")).get("cells"))
    by_leg: dict[str, Any] = {}
    for k, rec in cells.items():
        row = _mapping(rec)
        if row.get("liveness_evidence") != C.LENS_ON_BITE_EVIDENCE_LOGPROB:
            continue
        arm = str(row.get("arm") or row.get("cell"))
        by_leg[str(k)] = {"model_key": row.get("model_key"), "cell": row.get("cell"), "arm": arm, "mode": row.get("mode"),
                          "sentence": row.get("liveness_sentence"),
                          "consequence": (f"{row.get('model_key')} {arm} at its operating point does not move a single token of the "
                                          "render probe — bears on how much defence the arm buys at that dose")}
    return {"rule": LIVENESS_CONSEQUENCE_RULE if by_leg else None, "by_leg": by_leg}


def _d48_footnote_parts(report: Mapping[str, Any]) -> list[str]:
    """The two disclosure facts as short footer lines (the model and the arm's kind, never a leg path); ``[]`` when neither
    applies to the families shown."""
    notes = d48_notes(report)
    shown = _shown_only_keys_filter(report)
    parts: list[str] = []
    m = {k: v for k, v in notes["masked_primary"]["by_leg"].items() if shown(k)}
    if m:
        where = ", ".join(_leg_words(k) for k in m)
        parts.append(f"masked (D48): {sum(v['n_masked'] for v in m.values())} eos_in_thoughts draw(s) on {where}, removed from that arm's "
                     "contrast only (H1); no verdict word moves")
    lit = {k for k in notes["server_settings_literal_differs"] if shown(k)}
    if lit:
        parts.append(f"{', '.join(sorted({k.split('/')[0] for k in lit}))}: server default sampling pinned at the wave tree "
                     "(launch flag differs, effective decode equal; D48)")
    live = {k: v for k, v in notes["liveness_evidence"]["by_leg"].items() if shown(k)}
    if live:
        # class E: the operating-point fact, on the figure too — never hidden
        parts.append("live by logprob, no token bite (D95): " + ", ".join(sorted({f"{v['model_key']} {v['arm']}" for v in live.values()}))
                     + " — the render probe's greedy reply is byte-identical to the base route at this dose (bears on the defence bought)")
    return parts


def _d48_footnote(report: Mapping[str, Any]) -> str:
    """The disclosure facts joined into ONE line (the tables' note), or ``""``."""
    return " · ".join(_d48_footnote_parts(report))


def _footnote_lines(report: Mapping[str, Any]) -> list[str]:
    """The disclosure lines every figure that reads an estimand carries under its axes (review L6 / M3): the masking line
    (scoped, H1), the literal server-default line and the served-geometry line (M3); only the non-empty ones."""
    return [*_d48_footnote_parts(report), *([_geometry_footnote(report)] if _geometry_footnote(report) else [])]


def _masked_draws_shown(report: Mapping[str, Any]) -> dict[str, Any]:
    """The masked primary draws whose family the render shows (``{leg key: {n_masked, items}}``) — under shown_only,
    whose ROW the render shows — else ``{}``."""
    shown = _shown_only_keys_filter(report)
    return {k: v for k, v in d48_notes(report)["masked_primary"]["by_leg"].items() if shown(k)}


def _raw_draw_footnote(report: Mapping[str, Any]) -> str:
    """The line every RAW-DRAW panel (:data:`RAW_DRAW_PANELS` — the six that read the controls table, not an
    estimand) carries under its axes when a masked draw's family is shown: the mask applies only to the named
    accuracy / contrast estimands and does NOT reach this panel, whose counts are over the leg's full draws. ``""`` when no
    masked draw's family is shown (nothing to disclose)."""
    m = _masked_draws_shown(report)
    if not m:
        return ""
    where = ", ".join(_leg_words(k) for k in m)
    return (f"D48 mask ({sum(v['n_masked'] for v in m.values())} eos_in_thoughts draw(s) on {where}) applies only to the named accuracy / "
            "contrast estimands; this panel retains the raw draw — its counts are over the leg's full draws")


def _raw_draw_lines(report: Mapping[str, Any]) -> list[str]:
    """:func:`_raw_draw_footnote` as the list :func:`_footnotes_below` takes (empty when there is nothing to disclose)."""
    line = _raw_draw_footnote(report)
    return [line] if line else []


def _raw_draw_caption(report: Mapping[str, Any]) -> str:
    """The raw-draw disclosure as a caption sentence (``""`` when nothing is masked among the families shown)."""
    m = _masked_draws_shown(report)
    if not m:
        return ""
    where = "; ".join(f"{_leg_words(k)}: {', '.join(v['items'])}" for k, v in m.items())
    return (f" The D48 mask ({sum(v['n_masked'] for v in m.values())} eos_in_thoughts draw(s) — {where}) applies only to the named accuracy / "
            "contrast estimands (that arm's arm − bare contrast, that leg's own accuracy, that model's pair inside the pooled arm; review "
            "H1); this panel reads the controls table and retains the raw draw — every count here is over the leg's full draws, the "
            "masked draw included.")


def _disclosure_caption(report: Mapping[str, Any]) -> str:
    """The disclosure sentences every estimand-reading caption ends with (review H1 / M3 / L6): the masked primary draws
    and their scope, the literal server-default fork, the served-geometry differences with the rule; ``""`` when none
    applies to the families shown."""
    notes = d48_notes(report)
    shown = _shown_only_keys_filter(report)
    parts: list[str] = []
    m = notes["masked_primary"]
    shown_masked = {k: v for k, v in m["by_leg"].items() if shown(k)}
    if shown_masked:
        where = "; ".join(f"{_leg_words(k)}: {', '.join(v['items'])}" for k, v in shown_masked.items())
        parts.append(f"Masked (D48, scoped per review H1): {sum(v['n_masked'] for v in shown_masked.values())} eos_in_thoughts draw(s) — "
                     f"{where} — removed from that arm's arm − bare contrast, from that arm's own accuracy and from that model's pair "
                     "inside the pooled arm only; every other arm, contrast and pooled arm is over its full population; the reading "
                     "with the draw(s) is filed beside in the report (estimands_with_anomalies).")
    lit = {k for k in notes["server_settings_literal_differs"] if shown(k)}
    if lit:
        parts.append(f"{', '.join(sorted({k.split('/')[0] for k in lit}))}: the newer legs' standups pin the server's default sampling "
                     "(the launch flag differs literally); the effective decode is equal on every key the requests send and on every "
                     "key they do not (D48; review H2: a generation-config key the effect rule does not understand refuses the contrast).")
    geo = [rec for k, rec in notes["served_geometry"].items() if rec["differs"] and shown(k)]
    if geo:
        det = "; ".join(f"{rec['model_key']}{_view_tag(str(rec.get('view') or VIEW_MAIN))} "
                        + ", ".join(f"{g} ({'/'.join(kinds)})" for g, kinds in _geometry_groups(rec)) for rec in geo)
        same = [rec["model_key"] for rec in geo if rec["same_window_differs_in_concurrency"]]
        parts.append(f"Served geometry (L × S) differs within the contrast — {det}. {GEOMETRY_RULE}."
                     + (f" {', '.join(same)}: two legs at the SAME window differ in concurrency (a single-run cross-geometry comparison), "
                        "so the difference between those arms is not attributed to the arm alone without matched-execution evidence "
                        "(review M3)." if same else ""))
    return (" " + " ".join(parts)) if parts else ""


def _denominator_caption_text(rec: Mapping[str, Any]) -> str:
    """One contrast's denominators as caption prose (M2): scored / unscored counts and the censoring differential with its
    direction, then the at-cap counts and the neutral at-cap differential with its direction, then the shift written out."""
    cens, cap = rec.get("censoring_differential_pp"), rec.get("at_cap_differential_pp")
    text = (f"arm {rec['arm_n_scored']} · anchor {rec['bare_n_scored']} scored of {rec['n_items']}, unscored {rec['arm_n_unscored']} · "
            f"{rec['bare_n_unscored']}; " + (f"Δ censoring {cens:+.1f} pp — {rec['censoring_direction']}" if cens is not None else "Δ censoring not readable")
            + f"; at cap {rec['arm_n_at_cap']} · {rec['bare_n_at_cap']} of {rec['arm_n_draws']} draws; "
            + (f"Δ at-cap {cap:+.1f} pp — {rec['at_cap_direction']}" if cap is not None else "Δ at-cap not readable"))
    if rec.get("shift_text"):
        text += f"; shift {rec['shift_text']}" + ("" if rec.get("shift_identity_holds") else " (the decomposition does NOT reproduce the filed shift)")
    return text


def _headline_caption(report: Mapping[str, Any]) -> str:
    """Review M4 + second reader M-b, corrected by round 2 (M2 / M3): the headline rule, the two denominator rules, and, for
    the families shown, every contrast whose two scoring branches disagree in sign — each with its denominators, the
    censoring differential's DIRECTION (whose censoring drives it), the neutral at-cap differential and the shift written
    out; a pooled arm with its per-model scored triples and both means."""
    notes = conditional_vs_score_as_is_notes(report)
    shown = _shown_only_keys_filter(report)          # the families shown; under shown_only the comparisons (model/mode) shown
    dis = [f"{_comparison_words(key)} {rec['arm']} ({rec['conditional_pp']:+.1f} conditional vs {rec['score_as_is_pp']:+.1f} score-as-is pp; "
           f"{_denominator_caption_text(rec)})"
           for key, recs in notes["by_comparison"].items() if shown(key)
           for rec in recs.values() if not rec["signs_agree"]]
    for kind, rec in notes["pooled"].items():
        if rec["signs_agree"] is False:
            cens, cap = rec.get("censoring_differential_pp"), rec.get("at_cap_differential_pp")
            by = _mapping(rec.get("by_model"))
            trip = _mapping(rec.get("scored_by_model"))
            triples = ", ".join(f"{m} {t['arm_n_scored']}/{t['bare_n_scored']} of {t['n_items']}" for m, t in trip.items())
            per_c = ", ".join(f"{m} {_fmt_pp(d.get('censoring_differential_pp'))}" for m, d in by.items())
            per_a = ", ".join(f"{m} {_fmt_pp(d.get('at_cap_differential_pp'))}" for m, d in by.items())
            dis.append(f"pooled {kind} ({rec['conditional_pp']:+.1f} vs {rec['score_as_is_pp']:+.1f} pp; scored arm/bare {triples}; "
                       + (f"mean Δ censoring {cens:+.1f} pp — {rec['censoring_direction']}; per model {per_c}" if cens is not None else "Δ censoring not readable")
                       + "; " + (f"mean Δ at-cap {cap:+.1f} pp — {rec['at_cap_direction']}; per model {per_a}" if cap is not None else "Δ at-cap not readable")
                       + (f"; mean shift {rec['shift_pp']:+.2f} pp" if rec.get("shift_pp") is not None else "") + ")")
    text = (f" {HEADLINE_RULE}. Under each contrast row, two lines: the arm's and the bare anchor's scored and unscored counts over the "
            f"contrast's own population with the censoring differential and its direction; then their at-cap counts over the leg's draws "
            f"with the at-cap differential, its direction and the branch shift. A pooled row carries the per-model scored arm/bare counts, "
            f"then the equal-weight means of both differentials with the per-model values. {CENSORING_DIFFERENTIAL_RULE}. "
            f"{AT_CAP_DIFFERENTIAL_RULE}.")
    if dis:
        text += " Branches disagreeing in sign here: " + "; ".join(dis) + "."
    return text


def fig_eightarm_contrasts(report: Mapping[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    plt, mcolors, Patch = T._mpl()
    numbers: list[dict[str, Any]] = []
    design, ea = report["design"], report["eightarm"]
    excluded_arms = figure_excluded_arms(report)
    slots = panel_slots(report)                     # one forest row per arm the REPORT registers (a status row where no estimand)
    models = _shown_models(report, list(design["models"]))
    rows: list[dict[str, Any]] = []
    for j, c in enumerate(report["comparisons"]):
        if not c["primary"] or c["model_key"] not in models or comp_view(c) != VIEW_MAIN:
            continue                # E3: the eight slots are the main view's (design.arms_by_model); a non-main view is filed apart
        contrasts = comp_contrasts(c) if c.get("estimands") is not None else {}
        for arm in slots:
            if arm.kind == KIND_BARE or arm.cell in excluded_arms:
                continue
            name = contrast_name(arm.kind)
            label = f"{c['model_key']}{' (thinking: n/a)' if c['model_key'] in POOLED_HELD_OUT else ''}: {arm.kind}"
            if name in contrasts:
                rows.append({"label": label, "kind": arm.kind, "p": c["estimands"][BRANCH_PRIMARY]["stats"][name],
                             "q": c["estimands"][BRANCH_SCORE_AS_IS]["stats"][name], "word": c["verdicts"][name]["word"],
                             "path": f"estimands.{c['model_key']}.{c['mode']}", "contrast": name, "status": None,
                             "den": contrast_denominators(report, j, name)})
            else:
                rows.append({"label": label, "kind": arm.kind, "p": None, "q": None, "word": None,
                             "status": _slot_status(design, c["model_key"], arm.kind)})
    for arm in slots:
        if arm.kind == KIND_BARE or arm.cell in excluded_arms:
            continue
        rec = _mapping(_mapping(ea.get("pooled_by_arm")).get(arm.kind))
        label = f"{POOLED_GROUP_LABEL} ({'/'.join(ea['panel_models'])}): {arm.kind}"
        if rec.get("status") == POOLED_AVAILABLE:
            rows.append({"label": label, "kind": arm.kind, "p": rec[BRANCH_PRIMARY]["minus_bare"], "q": rec[BRANCH_SCORE_AS_IS]["minus_bare"],
                         "word": None, "path": f"eightarm.pooled_by_arm.{arm.kind}", "contrast": None, "status": None,
                         "pden": pooled_denominators(report, arm.kind)})
        else:
            rows.append({"label": label, "kind": arm.kind, "p": None, "q": None, "word": None, "status": POOLED_UNAVAILABLE})
    fig, ax = plt.subplots(figsize=(9.4, ROW_PITCH_IN * max(len(rows), 1) + 1.4))
    _layout(fig, left=0.34, right=0.855)
    ax.axvline(0.0, color=BASELINE, lw=0.8, zorder=2)
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        if r["p"] is None:
            word = DEFERRED_SLOT_LABEL if r["status"] == STATUS_DEFERRED else f"{r['status']}: no estimand"
            ax.text(0.02, y, word, transform=ax.get_yaxis_transform(), va="center", fontsize=6.2, color=MUTED)
            continue
        # second reader M-b, corrected by round 2 (M2 / M3): the denominators under every contrast row — per model two lines
        # (scored / unscored counts with the censoring differential, at-cap counts with the neutral at-cap differential and
        # the shift); for a pooled arm three lines (the per-model scored arm/bare triples, then the equal-weight means of both
        # differentials with the per-model values in the label's model order)
        if r.get("den"):
            _denominator_text(ax, y, _denominator_lines(r["den"]))
            _denominator_numbers(numbers, report, f"eightarm.contrast.{r['label']}", r["den"])
        elif r.get("pden"):
            _denominator_text(ax, y, _pooled_denominator_lines(r["pden"]))
            _pooled_denominator_numbers(numbers, report, f"eightarm.contrast.{r['label']}", r["pden"])
        p, q = r["p"], r["q"]
        ax.plot([_pp(p["ci_low"]), _pp(p["ci_high"])], [y + 0.12, y + 0.12], color=INK, lw=1.1, zorder=5)
        ax.plot([_pp(p["point"])], [y + 0.12], marker="o", ms=5, color=KIND_COLOR[r["kind"]], mec=INK, mew=0.5, ls="", zorder=6)
        ax.plot([_pp(q["ci_low"]), _pp(q["ci_high"])], [y - 0.14, y - 0.14], color=INK2, lw=0.9, zorder=5)
        ax.plot([_pp(q["point"])], [y - 0.14], marker="o", ms=5, mfc="none", mec=INK2, mew=1.0, ls="", zorder=6)
        for b, s in ((BRANCH_PRIMARY, p), (BRANCH_SCORE_AS_IS, q)):
            for f in ("point", "ci_low", "ci_high"):
                src = (f"{r['path']}.{b}.stats.{r['contrast']}.{f}" if r["contrast"] else f"{r['path']}.{b}.minus_bare.{f}")
                T._num_at(numbers, report, f"eightarm.contrast.{r['label']}.{f}[{b}]", src, s[f])
        if r["word"] is not None:
            ax.text(1.005, y, r["word"], transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=6.2,
                    color=INK if r["word"] in WORDS else MUTED, clip_on=False)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["label"] for r in reversed(rows)], fontsize=6.4)
    ax.set_ylim(-0.85, max(len(rows), 1) - 0.3)          # room for a pooled row's third denominator line
    ax.set_xlabel("conditional-accuracy difference vs bare, percentage points (filled: scored items; hollow: score-as-is)")
    T._recessive(ax)
    ax.xaxis.grid(True, zorder=0)
    ax.yaxis.grid(False)
    handles = [plt.Line2D([], [], marker="o", color=INK2, mec=INK, ls="", label="conditional on scored items (primary), 95 % bootstrap CI"),
               plt.Line2D([], [], marker="o", mfc="none", mec=INK2, ls="", label="score-as-is (unscored -> wrong), 95 % bootstrap CI")]
    h_in = ax.get_position().height * fig.get_figheight()
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.5 / h_in), ncol=2, frameon=False, fontsize=6.2)
    _footnotes_below(fig, ax, _footnote_lines(report), clearance_in=0.85)
    line = "arm − bare per model and pooled; no pooled verdict word"       # short enough to survive the selection clause whole
    _title(fig, "Arm contrasts\n" + _with_selection(report, line))
    return fig, numbers


#: the figure stems and their builders, in report order
FIGURE_STEMS: tuple[str, ...] = (
    "final_m5t_accuracy_by_cell", "final_m5t_contrasts", "final_m5t_sampling_sensitivity",
    "final_m5t_atcap_thinksplit", "final_m5t_atcap_share", "final_m5t_atcap_loopsplit", "final_m5t_no_reasoning_share",
    "final_m5t_ledger", "final_m5t_item_transitions",
    # last and alone on its line: the secret scanner's generic rule reads a quoted string after the word "tokens"
    "final_m5t_completion_tokens",
)
_FIGURE_BUILDERS = (fig_accuracy_by_cell, fig_contrasts, fig_sampling_sensitivity, fig_atcap_thinksplit,
                    fig_atcap_share, fig_atcap_loopsplit, fig_no_reasoning_share,
                    fig_ledger, fig_item_transitions, fig_completion_tokens)
FIGURES: Mapping[str, Callable[[Mapping[str, Any]], tuple[Any, list[dict[str, Any]]]]] = dict(zip(FIGURE_STEMS, _FIGURE_BUILDERS))
#: the figures that need an estimand (the others read the controls table)
FIGURES_NEED_ESTIMANDS: tuple[str, ...] = ("final_m5t_accuracy_by_cell", "final_m5t_contrasts",
                                           "final_m5t_sampling_sensitivity", "final_m5t_item_transitions")
EIGHTARM_FIGURES: Mapping[str, Callable[[Mapping[str, Any]], tuple[Any, list[dict[str, Any]]]]] = {
    EIGHTARM_STEM_ACCURACY: fig_eightarm_accuracy, EIGHTARM_STEM_CONTRASTS: fig_eightarm_contrasts}


#: the figures that draw the PRIMARY sampling row and nothing else — the accuracy panel and the two eight-arm
#: figures (their estimand is `estimands.<model>.<primary>`, their pooled means the primary row's). A `--figure-modes`
#: selection that leaves the primary out has nothing for them to draw: they are SKIPPED (recorded in the sidecar's
#: `skipped` with the reason), never drawn from the excluded row — the gate then refused a valid selection over the
#: excluded-mode records (126 from the accuracy figure alone)
PRIMARY_ONLY_FIGURES: tuple[str, ...] = ("final_m5t_accuracy_by_cell", EIGHTARM_STEM_ACCURACY, EIGHTARM_STEM_CONTRASTS)


def figure_skip_reason(stem: str, view: Mapping[str, Any]) -> str | None:
    """Why ``stem`` is not drawn from ``view`` (the report under the render's selectors), or ``None``: the estimand did not
    run; the primary row is out of the mode selection and the figure draws it only; the sampling-sensitivity figure has no
    model with a shown row; the contrasts figure has no shown row. A skipped figure is recorded, never drawn from what the
    selection excluded."""
    if _needs_estimands(stem) and "estimands" not in view:
        return "the estimand did not run (a leg is missing or refused)"
    if stem in PRIMARY_ONLY_FIGURES and not _mode_shown(view, MODE_PRIMARY):
        return f"draws the primary sampling row ({MODE_PRIMARY}) only, which --figure-modes left out"
    if stem == "final_m5t_sampling_sensitivity" and not _sensitivity_models(view):
        return "no shown model carries a sampling row the selection shows"
    modes = figure_modes(view)
    if stem == "final_m5t_sampling_sensitivity" and figure_shown_only(view) and modes is not None and len(modes) == 1:
        # a render that describes the shown set only, with ONE sampling row shown, has no sampling-row comparison to
        # draw — the figure would repeat the accuracy figure under a title that announces rows the render leaves out;
        # it is skipped and the reason recorded (the default render and a several-mode render draw it)
        return "one sampling row is shown and the render describes the shown set only: a sampling-row comparison needs two rows"
    if stem == "final_m5t_contrasts" and not _contrast_rows(view):
        return "no contrast row is shown under the selection"
    return None


def figures_for(report: Mapping[str, Any]) -> dict[str, Callable[[Mapping[str, Any]], tuple[Any, list[dict[str, Any]]]]]:
    """The published ten, plus the two eight-arm figures when the report carries the eight-arm blocks (B2) — so a
    published-shape run writes exactly the published figure set."""
    out: dict[str, Callable[[Mapping[str, Any]], tuple[Any, list[dict[str, Any]]]]] = dict(FIGURES)
    if "eightarm" in report and "design" in report:
        out.update(EIGHTARM_FIGURES)
    return out


def _needs_estimands(stem: str) -> bool:
    return stem in FIGURES_NEED_ESTIMANDS or stem in EIGHTARM_FIGURES


def figure_stems_for(report: Mapping[str, Any], *, view: Mapping[str, Any] | None = None) -> list[str]:
    """The stems a render of ``report`` WRITES — under ``view`` (the report with the render's `figures` selection) the
    figures the selection skips are left out too, so the directory rule reads exactly what is written."""
    return [stem for stem in figures_for(report) if figure_skip_reason(stem, view if view is not None else report) is None]


def _atcap_title_caption(report: Mapping[str, Any]) -> str:
    """What the at-cap share figure's second title line is: the premise verdict, or under shown_only the shown set's descriptor."""
    if figure_shown_only(report):
        return ("The second title line describes the cells SHOWN only, labelled as shown (primary cells at or above the threshold, "
                "cells over the 1 % line, the shown maximum); the premise verdict is the report's truncation.premise_failed.")
    return "The second title line is the verdict on the premise."


def figure_captions(report: Mapping[str, Any]) -> dict[str, str]:
    """The prose that stays OUT of the PNGs — one caption per figure THIS report draws
    (:func:`figures_for`: the published ten, plus the two eight-arm figures under the eight-arm blocks)."""
    return {k: v for k, v in _all_captions(report).items() if k in figures_for(report)}


def _sensitivity_caption(report: Mapping[str, Any]) -> str:
    """Review L6: the sampling-row figure's caption says what the render SHOWS — one row (then it is not a sensitivity
    comparison and repeats the primary row's reading) or several."""
    modes = figure_modes(report)
    if modes is not None and len(modes) == 1:
        (mode,) = tuple(modes)
        head = (f"One sampling row shown ({mode}) — under this render the figure is NOT a sampling-row comparison: it repeats the "
                "shown row's conditional accuracy per cell (scored items; 95 % CIs) as the accuracy figure does")
        if figure_shown_only(report):        # the shown set's caption names no row the render leaves out
            return head + "."
        return head + "; the sensitivity rows (general, battery) are out of the render and their comparisons stay in the report data."
    return ("Conditional accuracy (scored items) per cell under every sampling row present for the model (marker = row); the rows "
            "are separate instruments and are never pooled.")


def _atcap_shading_clause(report: Mapping[str, Any]) -> str:
    """The at-cap share caption's shading clause. Default: the published wording (primary cells in the cell hue, the sensitivity
    rows in a lighter shade). Under shown_only: over the cells the figure DRAWS — the lighter shade is named only when a
    sensitivity-row cell is among them, with the modes actually drawn."""
    if not figure_shown_only(report):
        return "the primary cells in the cell hue, the sensitivity rows (general, battery) in a lighter shade"
    shown = _shown_rows(report, report["truncation"]["cells"])
    sens = [m for m in SENSITIVITY_MODES if any(not c["primary"] and c["mode"] == m for c in shown)]
    if not sens:
        return "the primary cells in the cell hue"
    return f"the primary cells in the cell hue, the sensitivity rows ({', '.join(sens)}) in a lighter shade"


def _contrasts_rows_clauses(report: Mapping[str, Any]) -> tuple[str, str]:
    """The contrasts caption's two row-set clauses (what rows are drawn; who files a word). Under shown_only with no
    sensitivity-row comparison or mode delta among the rows drawn, the primary rows alone are described."""
    if figure_shown_only(report):
        rows = _contrast_rows(report)
        comps = report["comparisons"]
        if not any(r["kind"] == "delta" or not comps[r["comp"]]["primary"] for r in rows):
            return "arm − bare per model (primary rows)", "The word at the right is the verdict rule's (D5 P2-2)."
    return ("arm − bare per model (primary rows first, then each sensitivity row as its own comparison, then the per-cell paired "
            "deltas of each sensitivity row to the primary)", "The word at the right is the verdict rule's (D5 P2-2); sensitivity rows file no word.")


_NO_REASONING_CLASS_TEXT: Mapping[str, tuple[str, str]] = {
    # split_mode -> (the clause before the family list, the clause after it) of the no-reasoning caption
    "record_only": ("record_only (", ") an empty reasoning channel"),
    "explicit_tags": ("explicit_tags (", ") no think tag in the text AND an empty reasoning channel on the row (inspect_ai lifts a complete "
                      "<think>…</think> block out of the text before the scorer's tag test, so the row's channel is read beside the "
                      "record's flag — controls[*].thinking.n_present_by_channel counts those draws)"),
    "implicit_open": ("implicit_open families (", ") are 0 by construction"),
}


def _no_reasoning_classes_clause(report: Mapping[str, Any]) -> str:
    """The no-reasoning caption's class clause. Default: the published wording (every class with the wave's example families).
    Under shown_only: only the classes carried by the rows DRAWN, each naming the shown families that carry it — a family the
    render leaves out is not named."""
    if not figure_shown_only(report):
        return ("record_only (gemma) an empty reasoning channel, explicit_tags (glm) no think tag in the text AND an empty reasoning "
                "channel on the row (inspect_ai lifts a complete <think>…</think> block out of the text before the scorer's tag test, so "
                "the row's channel is read beside the record's flag — controls[*].thinking.n_present_by_channel counts those draws); "
                "implicit_open families (qwen, nemotron) are 0 by construction")
    by_class: dict[str, list[str]] = {}
    for r in _shown_rows(report, _readout_rows(report)):
        fams = by_class.setdefault(str(r["thinking"].get("split_mode")), [])
        if r["model_key"] not in fams:
            fams.append(r["model_key"])
    parts = [f"{pre}{', '.join(by_class[cls])}{post}" for cls, (pre, post) in _NO_REASONING_CLASS_TEXT.items() if cls in by_class]
    parts += [f"{cls} ({', '.join(fams)})" for cls, fams in by_class.items() if cls not in _NO_REASONING_CLASS_TEXT]
    return "; ".join(parts) if parts else "no usable leg is shown"


def itdoc_legs_shown(report: Mapping[str, Any]) -> dict[tuple[str, str], list[Mapping[str, Any]]]:
    """E3: the it-doc legs a render SHOWS — the usable controls rows with ``object_source: itdoc`` that pass the three selectors —
    grouped ``{(model_key, view): [rows]}`` in controls order."""
    out: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in report.get("controls", ()):
        if not is_itdoc_row(row) or row.get("missing") or row.get("refused") or not _row_shown(report, row):
            continue
        out.setdefault((str(row["model_key"]), str(row.get("view") or VIEW_MAIN)), []).append(row)
    return out


def itdoc_caption(report: Mapping[str, Any]) -> str:
    """E3: the disclosure every figure whose shown legs include an it-doc leg ends with — per model and view, the it-doc arms
    shown and the row's qualification VERBATIM (the model's own sentence, then the cross-model pattern); ``""`` when none is
    shown. The figures that read the estimands of the main view only (accuracy by cell, the eight-arm panel) never show one."""
    parts: list[str] = []
    for (m, view), rows in itdoc_legs_shown(report).items():
        arms = ", ".join(dict.fromkeys(str(r["kind"]) for r in rows))
        quals = list(dict.fromkeys(qualification_text(r) or "no qualification recorded on the row" for r in rows))
        parts.append(f"it-doc re-fit ({m} {view}: {arms}): {' / '.join(quals)}")
    return (" " + " ".join(parts)) if parts else ""


#: the figures whose rows an it-doc leg can appear in (the six raw-draw panels read every controls row; the contrasts figure
#: draws every comparison that ran) — :func:`itdoc_caption` rides their captions
ITDOC_CAPTION_FIGURES: tuple[str, ...] = ("final_m5t_atcap_thinksplit", "final_m5t_atcap_share", "final_m5t_atcap_loopsplit",
                                          "final_m5t_no_reasoning_share", "final_m5t_ledger", "final_m5t_completion_tokens",
                                          "final_m5t_contrasts")


def _all_captions(report: Mapping[str, Any]) -> dict[str, str]:
    caps = _all_captions_main(report)
    itdoc = itdoc_caption(report)
    if itdoc:
        for stem in ITDOC_CAPTION_FIGURES:
            caps[stem] = caps[stem] + itdoc
    return caps


def _all_captions_main(report: Mapping[str, Any]) -> dict[str, str]:
    b = report["bootstrap"]
    disclosure = _disclosure_caption(report)
    headline = _headline_caption(report)
    raw = _raw_draw_caption(report)          # the six raw-draw panels say the mask does not reach them
    contrast_rows_clause, contrast_word_clause = _contrasts_rows_clauses(report)
    return {
        "final_m5t_accuracy_by_cell": ("Per model, each arm present in the primary sampling row (the published arms and, in "
                                       "eight-arm mode, the panel's further arms): conditional accuracy on the SCORED items (own "
                                       f"denominator). {SCORING_DEFINITION}. Whiskers "
                                       f"= 95 % percentile bootstrap CI (B = {b['B']:,}, seed {b['seed']}); hollow marker = the "
                                       "score-as-is branch's point (unscored -> wrong) — where it sits far from the bar the cell's "
                                       "reading is censoring-driven." + disclosure),
        "final_m5t_contrasts": (f"{contrast_rows_clause}, as conditional accuracy on scored "
                                "items. Filled = primary branch, hollow = score-as-is, grey bracket = the censoring-sensitivity bound "
                                f"(a sample bound, not a CI). {contrast_word_clause}" + headline + disclosure),
        "final_m5t_sampling_sensitivity": _sensitivity_caption(report) + disclosure,
        "final_m5t_atcap_thinksplit": ("At-cap rate split into case (b) the cap inside the think block (unscored under the primary "
                                       "policy) and case (c) the cap inside the answer (scored); the unterminated and think-present "
                                       "shares per cell; the dashed line is the parent's 2 % re-cost line — advisory at the ceiling." + raw),
        "final_m5t_atcap_share": (f"At-cap share per cell at the {CAP:,} cap (at-cap draws over draws with a stop reason, the "
                                  f"summary's own counts), grouped by model; {_atcap_shading_clause(report)}. Dashed: the 1 % line of the harness rule; "
                                  "dash-dot: the 3 % surfacing threshold declared before the number existed (twice the "
                                  "ruling's GSM8K premise of ~1.5 %); any primary cell at or above it fails the premise "
                                  "(D65 / D68). " + _atcap_title_caption(report) + " Markers: the share of "
                                  "each cell's items the harness re-drew at the same cap (usage ledger, D151) — " +
                                  REDRAW_SELECTION_CAVEAT + "." + raw),
        "final_m5t_atcap_loopsplit": ("Each cell's at-cap draws under the loop rule (recorded once in the report, "
                                      "truncation.loop_rule): orange = degenerate loop (an exactly repeated block covering the "
                                      "tail; compression is a recorded diagnostic and never decides), blue = truncated "
                                      "reasoning (at cap, not a loop), hatched grey = at-cap draws whose text the split's source "
                                      f"did not carry. Split source (the legend title): {_split_source_labels(_readout_rows(report))[0]} "
                                      "— the eval logs carry every at-cap draw's text; the store rows carry a SCORED draw's only, "
                                      "so under them the unscored case-(b) draws (the cap inside the think block) are text-"
                                      "unavailable. Per cell: truncation.cells[*].split_source; the store-rows split rides "
                                      "alongside under legs.*.at_cap.store_rows_split. The totals over the cells shown are in "
                                      "the legend." + raw),
        "final_m5t_no_reasoning_share": ("Per cell, the share of thinking-ON draws that FINISHED (not at cap) with no "
                                         "reasoning emitted — the model declining to think on a draw at this row's "
                                         f"temperature (D115): {_no_reasoning_classes_clause(report)}. Those draws are graded from "
                                         "content as answers and counted here, never folded silently." + raw),
        "final_m5t_completion_tokens": "Completion lengths per cell: the summary's mean and p99 with the ledger's p50-p99 band and the cap." + raw,
        "final_m5t_ledger": ("Usage-ledger confirmation per cell: forwarded calls, named refusals (error_code), rejections, "
                             "and every call beyond one per sample classified as a recorded resume, the harness's same-cap "
                             "re-send (" + LADDER_TEXT + ") or an unexplained duplicate, which fails the gate (D151)." + raw),
        "final_m5t_item_transitions": ("Per-item transitions from the bare cell to each arm present (primary row): C correct, I "
                                       "incorrect, U unscored." + disclosure),
        EIGHTARM_STEM_ACCURACY: (panel_caption_head(report) + " — the "
                                 "primary row's conditional accuracy on SCORED items (own denominator; the unscored draws leave it — "
                                 "the definition closes this caption) with its 95 % bootstrap CI where the leg ran; an empty "
                                 "hatched slot is a design status (D deferred: the AlphaSteer / Arditi rung waits for the measured "
                                 "minimal-over-refusal dose, D2/D3/D8; H held; n not analysed; – absent), never a dropped column. "
                                 "llama8b is labelled 'thinking: not available' and held out of the pooled group; GLM is not in "
                                 "the panel. The pooled group draws report.eightarm.pooled_by_arm — the equal-weight mean over "
                                 "gemma / qwen / nemotron with ONE shared resampling index vector per replicate — and reads "
                                 "'unavailable' (u) whenever any panel model's leg for the arm is missing, refused, held or deferred. "
                                 + HEADLINE_RULE + "." + disclosure),
        EIGHTARM_STEM_CONTRASTS: ("Per model and arm, arm − bare on the primary row as conditional accuracy on SCORED items (filled) "
                                  "beside the score-as-is reading (hollow; unscored -> wrong), 95 % bootstrap CIs; the verdict word at "
                                  "the right is the rule's (D5 P2-2). Then the pooled arm − bare over gemma / qwen / nemotron from "
                                  "report.eightarm.pooled_by_arm (shared draws; llama held out), both branches drawn; "
                                  + POOLED_WORD_RULE + ". A deferred arm reads 'deferred: dose unmeasured (D2/D3)' in place."
                                  + headline + disclosure),
    }


def validated_exclusions(report: Mapping[str, Any], families: Sequence[str]) -> list[str]:
    """``--figure-exclude-family`` checked against the report: every name a model
    key of a controls row (a typo would otherwise exclude nothing, silently) and
    at least one model left to draw; sorted, deduplicated."""
    known = list(dict.fromkeys(row["model_key"] for row in report["controls"]))
    excluded = sorted(set(families))
    unknown = [f for f in excluded if f not in known]
    if unknown:
        raise AnalysisError(f"--figure-exclude-family {unknown}: not a model key of this report (the controls' models are {known})")
    if len(excluded) >= len(known):
        raise AnalysisError(f"--figure-exclude-family {excluded} leaves no model to draw (the controls' models are {known})")
    return excluded


def validated_selection(report: Mapping[str, Any], families: Sequence[str] = (), arms: Sequence[str] = (),
                        modes: Sequence[str] | None = None) -> dict[str, Any]:
    """The three PRESENTATION selectors validated against THIS report (plan P5): the family exclusion
    (:func:`validated_exclusions`), ``--figure-exclude-arm`` (every name a CELL of the report's matrix — an arm absent
    from the report is an ERROR, so the MATH-500 runner passes none; the bare cell can never be excluded, every contrast
    is arm − bare; at least one other cell must stay) and ``--figure-modes`` (every name a sampling mode of the report's
    matrix; at least one). Returns ``{excluded_families, excluded_arms, modes}`` (``modes`` None = every mode)."""
    excluded = validated_exclusions(report, families) if families else []
    cells = list(dict.fromkeys(str(r["cell"]) for r in report["cells"]))
    given = sorted(set(str(a) for a in arms))
    unknown = [a for a in given if a not in cells]
    if unknown:
        raise AnalysisError(f"--figure-exclude-arm {unknown}: not a cell of this report's matrix (the cells are {cells})")
    if CELL_BARE in given:
        raise AnalysisError(f"--figure-exclude-arm {CELL_BARE!r}: the bare cell is every contrast's reference and cannot be left out")
    # a PSR cell is out of EVERY figure by rule, selector or no selector (none exists on this instrument today; the
    # rule is the instruments' shared one)
    arms_x = sorted(set(given) | set(psr_cells_of(cells)))
    if len(arms_x) >= len(cells) - 1:
        raise AnalysisError(f"the figure exclusion {arms_x} leaves no arm to contrast with bare (the cells are {cells}; PSR cells "
                            f"{psr_cells_of(cells)} are out of every figure by ruling) — nothing to draw")
    modes_x: list[str] | None = None
    if modes is not None:
        present = list(dict.fromkeys(str(r["sampling_mode"]) for r in report["cells"]))
        modes_x = list(dict.fromkeys(str(m) for m in modes))
        unknown = [m for m in modes_x if m not in present]
        if unknown:
            raise AnalysisError(f"--figure-modes {unknown}: not a sampling mode of this report's matrix (the modes are {present})")
        if not modes_x:
            raise AnalysisError("--figure-modes names no mode")
    return {FIGURES_EXCLUDED_KEY: excluded, FIGURES_EXCLUDED_ARMS_KEY: arms_x, FIGURES_MODES_KEY: modes_x}


def refuse_mixed_render(figs_dir: Path, excluded: Sequence[str], stems: Sequence[str], *, arms: Sequence[str] = (),
                        modes: Sequence[str] | None = None, shown_only: bool = False, numbers_filename: str = NUMBERS_FILENAME,
                        stem_prefix: str = STEM_PREFIX) -> None:
    """A figures directory (or mirror) holds ONE selection (the family exclusion,
    the arm exclusion, the mode set, the shown_only switch). Its numbers sidecar names the selection its
    files were drawn under; a figure file this render would not write is a stale
    one. Either refuses by name — an isolated directory is the remedy. Why (review
    of 2026-09-11): a directory re-used across a no-flag render and an exclusion
    render keeps the previous render's files on disk, and a whole-directory
    upload would ship them under a config declaring a family excluded; the
    manifest mirror removes nothing. ``numbers_filename`` / ``stem_prefix``: the
    instrument's sidecar name and figure-stem prefix (the GSM8K analysis passes
    its own; the defaults are this instrument's)."""
    figs_dir = Path(figs_dir).expanduser().resolve()
    if not figs_dir.is_dir():
        return
    side = figs_dir / numbers_filename
    if side.is_file():
        doc = _mapping(_read_json(side))
        prior = list(doc.get(FIGURES_EXCLUDED_KEY) or [])
        if prior != list(excluded):
            raise AnalysisError(f"{figs_dir}: its figures were drawn with {FIGURES_EXCLUDED_KEY}={prior}; this render excludes {list(excluded)} — use an isolated directory")
        prior_arms = list(doc.get(FIGURES_EXCLUDED_ARMS_KEY) or [])
        if prior_arms != list(arms):
            raise AnalysisError(f"{figs_dir}: its figures were drawn with {FIGURES_EXCLUDED_ARMS_KEY}={prior_arms}; this render excludes {list(arms)} — use an isolated directory")
        prior_modes = doc.get(FIGURES_MODES_KEY)
        if (None if prior_modes is None else list(prior_modes)) != (None if modes is None else list(modes)):
            raise AnalysisError(f"{figs_dir}: its figures were drawn with {FIGURES_MODES_KEY}={prior_modes}; this render draws {modes} — use an isolated directory")
        prior_shown_only = bool(doc.get(FIGURES_SHOWN_ONLY_KEY))
        if prior_shown_only != bool(shown_only):
            raise AnalysisError(f"{figs_dir}: its figures were drawn with {FIGURES_SHOWN_ONLY_KEY}={prior_shown_only}; this render "
                                f"{'sets' if shown_only else 'does not set'} it — use an isolated directory")
    want = set(stems)
    stale = sorted(p.name for p in figs_dir.glob(f"{stem_prefix}*") if p.suffix in (".png", ".pdf") and p.stem not in want)
    if stale:
        raise AnalysisError(f"{figs_dir}: carries figure files this render would not write ({stale}) — use an isolated directory")


#: the selection record's rule sentence under an arm / mode selection (plan P5) — and its shown_only form, under which
#: the tables JSON and the captions are the SHOWN rows' too
SELECTION_RULE = ("presentation only: the report, the tables JSON and every aggregate are the full report's; the selectors pick the "
                  "rows SHOWN and every title carries both counts")
SELECTION_RULE_SHOWN_ONLY = ("presentation only: the report is read unchanged and no aggregate is recomputed; the selectors pick the "
                             "rows SHOWN and every title carries both counts; shown_only: the set-describing titles, legends and "
                             "captions, the numbers sidecar and the tables JSON carry the shown rows only, labelled — no value, verdict "
                             "word, gate leg, note or provenance entry of a row the render does not draw (the selection record names "
                             "what was left out)")
#: the sidecar keys that ARE the selection record: they name the excluded family by design and are outside the leak check
SELECTION_RECORD_KEYS: frozenset[str] = frozenset({FIGURES_EXCLUDED_KEY, FIGURES_EXCLUDED_ARMS_KEY, FIGURES_MODES_KEY, FIGURES_SELECTION_KEY,
                                                   FAMILIES_SHOWN_KEY, FIGURES_SHOWN_ONLY_KEY})


def scope_tables_to_shown(tables: Mapping[str, Any], view: Mapping[str, Any]) -> dict[str, Any]:
    """The tables JSON reduced to the rows the view's selection SHOWS (``shown_only``): a presentation filter, never a
    recomputation — every value kept is the report's own. Dropped: under
    ``effects`` / ``acc_by_leg`` / ``mode_deltas`` the models excluded and the sampling modes not shown (a mode delta,
    ``<mode> − primary``, only when both rows are shown — the contrasts figure's own rule); every ``controls`` row not
    shown; in ``verdict``, ``gates``, ``provenance`` and ``notes`` every entry keyed by a hidden family, a hidden leg id
    (``model/cell/mode``) or a hidden ``model/mode`` pair (:func:`_key_shown`), every list row carrying a hidden ``leg`` /
    ``model_key`` + ``mode``, and every list STRING whose leading ``leg:`` names a hidden leg (a gate advisory). The
    verdict's joined string is re-joined from the shown models' words (the words themselves untouched). Gate-level
    ``status`` / ``detail`` / ``rule`` text stays: it is the report's gate verdict, names no row, and the shown rows passed
    under it. The render metadata keys (:data:`SELECTION_RECORD_KEYS`, ``sha``, ``analysis_code_sha``, ``captions``) pass
    through untouched — the selection record names what was left out, by design."""
    models = set(_report_models(view))
    hidden_models = {m for m in models if not _key_shown(view, m)}

    def keep(key: Any) -> bool:
        return _key_shown(view, str(key))

    def scrub(o: Any) -> Any:
        if isinstance(o, Mapping):
            return {k: scrub(v) for k, v in o.items() if keep(k)}
        if isinstance(o, list):
            out = []
            for v in o:
                if isinstance(v, Mapping):
                    if "leg" in v and not keep(v["leg"]):
                        continue
                    if "model_key" in v and (v["model_key"] in hidden_models or ("mode" in v and not _row_shown(view, v))):
                        continue
                elif isinstance(v, str) and ":" in v and not keep(v.split(":", 1)[0].strip()):
                    continue
                out.append(scrub(v))
            return out
        return o

    def by_model_mode(block: Mapping[str, Any], *, delta: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for m, by in block.items():
            if m in hidden_models:
                continue
            modes = {mode: v for mode, v in _mapping(by).items()
                     if _mode_shown(view, mode) and (not delta or _mode_shown(view, MODE_PRIMARY))}
            if modes or not delta:
                out[m] = scrub(modes)
        return out

    scoped: dict[str, Any] = dict(tables)
    for block in ("effects", "acc_by_leg"):
        if isinstance(tables.get(block), Mapping):
            scoped[block] = by_model_mode(tables[block])
    if isinstance(tables.get("mode_deltas"), Mapping):
        scoped["mode_deltas"] = by_model_mode(tables["mode_deltas"], delta=True)
    if isinstance(tables.get("controls"), list):
        scoped["controls"] = scrub(tables["controls"])
    for block in ("gates", "provenance", "notes", TREE_SET_KEY):
        if isinstance(tables.get(block), Mapping):
            scoped[block] = scrub(tables[block])
    verdict = tables.get("verdict")
    if isinstance(verdict, Mapping):
        v = scrub(verdict)
        words = _mapping(v.get("words"))
        if v.get("word") == "filed" and isinstance(verdict.get("verdict"), str) and words:
            v["verdict"] = "; ".join(f"{m}: " + ", ".join(f"{k} {w}" for k, w in _mapping(ws).items()) for m, ws in words.items())
        scoped["verdict"] = v
    return scoped


def shown_only_leaks(doc: Mapping[str, Any], view: Mapping[str, Any]) -> list[str]:
    """Where a sidecar document names a row the view's selection does not draw: every string leaf (keys included) of the
    document OUTSIDE the selection record (:data:`SELECTION_RECORD_KEYS`) carrying a hidden family as a whole segment
    (:func:`PG._segment_pattern`), a hidden leg id or a hidden ``model/mode`` pair as a substring. ``[]`` is the proof the
    render is isolated from the rows it leaves out; :func:`make_figures` refuses under ``shown_only`` on anything else."""
    models = _report_models(view)
    hidden_families = [m for m in models if not _key_shown(view, m)]
    legs = [str(r["leg"]) for r in view["controls"] if "leg" in r]
    hidden_legs = [leg for leg in legs if not _key_shown(view, leg)]
    hidden_pairs = sorted({f"{r['model_key']}/{r['mode']}" for r in view["controls"] if r.get("mode") is not None
                           and not _key_shown(view, f"{r['model_key']}/{r['mode']}")})
    fam_pats = [(f, PG._segment_pattern(f)) for f in hidden_families]
    out: list[str] = []

    def walk(o: Any, path: str) -> None:
        if isinstance(o, Mapping):
            for k, v in o.items():
                if not path and str(k) in SELECTION_RECORD_KEYS:
                    continue
                check(str(k), f"{path}.{k}" if path else str(k), key=True)
                walk(v, f"{path}.{k}" if path else str(k))
        elif isinstance(o, (list, tuple)):
            for i, v in enumerate(o):
                walk(v, f"{path}[{i}]")
        elif isinstance(o, str):
            check(o, path, key=False)

    def check(s: str, where: str, *, key: bool) -> None:
        for f, pat in fam_pats:
            if pat.search(s):
                out.append(f"{where}: {'key' if key else 'text'} names the excluded family {f!r}")
                return
        for leg in hidden_legs:
            if leg in s:
                out.append(f"{where}: {'key' if key else 'text'} names the hidden leg {leg!r}")
                return
        for pair in hidden_pairs:
            if pair in s:
                out.append(f"{where}: {'key' if key else 'text'} names the hidden row {pair!r}")
                return

    walk(doc, "")
    return out


def make_figures(report: Mapping[str, Any], figs_dir: Path, *, exclude_families: Sequence[str] = (),
                 exclude_arms: Sequence[str] = (), modes: Sequence[str] | None = None, shown_only: bool = False) -> dict[str, Any]:
    """Every figure the report can support, PNG + PDF, each checked for the
    title rule and the axes fraction, plus the numbers sidecar (every plotted
    number with its resolvable report path) and the tables JSON.
    ``exclude_families`` (``--figure-exclude-family``, repeatable) leaves the
    named model families out of every figure and of the per-model / per-leg
    entries of the tables JSON: the figures are drawn from a shallow VIEW of the
    report whose ``figures.excluded_families`` names them, so the report itself
    is untouched and every sidecar path still resolves in it; the gates, the
    verdict and the wave-level provenance are the wave's; the exclusion is
    recorded in the returned manifest (-> ``report["figures"]``), the numbers
    sidecar and the tables — all of it ONLY when a family is excluded, so with
    the flag absent every output (figures, sidecars, manifest) is byte-identical
    to the pre-flag shape. ``exclude_arms`` (``--figure-exclude-arm``) and
    ``modes`` (``--figure-modes``) are the two further PRESENTATION selectors
    (plan P5): validated against the report
    (:func:`validated_selection`), they pick the rows SHOWN, recompute no
    aggregate (pooled means, verdict words and the tables JSON are the full
    report's) and put both counts in every title; recorded under
    ``figures.selection`` ONLY when one is in play. The directory holds one
    selection (:func:`refuse_mixed_render`). ``shown_only`` (``--figure-shown-only``): the
    set-describing figures and the numbers sidecar carry the SHOWN set's descriptor only,
    labelled (:data:`FIGURES_SHOWN_ONLY_KEY`); every legend and caption describes the rows
    drawn; the tables JSON is scoped to the shown rows (:func:`scope_tables_to_shown`) and
    both sidecars are checked for a hidden row before they are written
    (:func:`shown_only_leaks`); refused without a selector; recorded only when set, so
    the default outputs stay byte-identical."""
    figs_dir = Path(figs_dir).expanduser().resolve()
    sel = validated_selection(report, exclude_families, exclude_arms, modes)
    excluded, arms, modes_x = sel[FIGURES_EXCLUDED_KEY], sel[FIGURES_EXCLUDED_ARMS_KEY], sel[FIGURES_MODES_KEY]
    hidden = frozenset(excluded)
    if shown_only and not (excluded or arms or modes_x is not None):
        raise AnalysisError(f"{FIGURES_SHOWN_ONLY_KEY} needs a selector (a family exclusion, an arm exclusion or a mode set): with every row "
                            "drawn there is no shown set to describe apart from the report's")
    figures_block: dict[str, Any] = {**_mapping(report.get("figures")), FIGURES_EXCLUDED_KEY: list(excluded)}
    if arms:
        figures_block[FIGURES_EXCLUDED_ARMS_KEY] = list(arms)
    if modes_x is not None:
        figures_block[FIGURES_MODES_KEY] = list(modes_x)
    if shown_only:
        figures_block[FIGURES_SHOWN_ONLY_KEY] = True
    view: dict[str, Any] = {**report, "figures": figures_block}
    builders = figures_for(report)
    stems = figure_stems_for(report, view=view)
    refuse_mixed_render(figs_dir, excluded, stems, arms=arms, modes=modes_x, shown_only=shown_only)
    figs_dir.mkdir(parents=True, exist_ok=True)
    extra: dict[str, Any] = {FIGURES_EXCLUDED_KEY: list(excluded), FAMILIES_SHOWN_KEY: _shown_models(view, _report_models(report))} if excluded else {}
    if arms:
        extra[FIGURES_EXCLUDED_ARMS_KEY] = list(arms)
    if modes_x is not None:
        extra[FIGURES_MODES_KEY] = list(modes_x)
    if shown_only:
        extra[FIGURES_SHOWN_ONLY_KEY] = True
    if arms or modes_x is not None:
        # recorded ONLY under an arm / mode selection: the counts a title carries, never a recomputed aggregate
        rows_all = _leg_rows(report)
        extra[FIGURES_SELECTION_KEY] = {FIGURES_EXCLUDED_KEY: list(excluded), FIGURES_EXCLUDED_ARMS_KEY: list(arms), FIGURES_MODES_KEY: modes_x,
                                        "n_legs_report": len(rows_all), "n_legs_shown": len(_shown_rows(view, rows_all)),
                                        "rule": SELECTION_RULE_SHOWN_ONLY if shown_only else SELECTION_RULE}
        if shown_only:
            extra[FIGURES_SELECTION_KEY][FIGURES_SHOWN_ONLY_KEY] = True
    written: dict[str, list[str]] = {}
    numbers: dict[str, list[dict[str, Any]]] = {}
    skipped: dict[str, str] = {}
    for stem, fn in builders.items():
        why = figure_skip_reason(stem, view)
        if why:
            skipped[stem] = why
            continue
        fig, nums = fn(view)
        frac = axes_height_fraction(fig)
        if frac < AXES_MIN_FRACTION:
            raise AnalysisError(f"{stem}: the axes span {frac:.3f} of the figure height (< {AXES_MIN_FRACTION})")
        why_title = figure_title_problem(fig)
        if why_title:        # the title rule, enforced at the render
            raise AnalysisError(f"{stem}: {why_title}")
        written[stem] = [str(p) for p in T._save(fig, figs_dir, stem)]
        numbers[stem] = nums
    captions = {k: v for k, v in figure_captions(view).items() if k in written}
    numbers_doc = {"figures": numbers, "captions": captions, "skipped": skipped, "sha": report["sha"],
                   "analysis_code_sha": report.get("analysis_code_sha"), **extra}
    # a caption describes what the RENDER shows (the single-row sampling figure, the families' disclosures), so the
    # numbers sidecar carries the view's; the tables JSON is the full report's under an arm / mode selector (plan P5)
    # and carries the captions of the family-only view — except under shown_only, when both sidecars are the SHOWN
    # rows' (:func:`scope_tables_to_shown`)
    table_captions = captions if shown_only else {k: v for k, v in figure_captions({**report, "figures": {FIGURES_EXCLUDED_KEY: list(excluded)}}).items() if k in written}
    est = {m: by for m, by in (report.get("estimands") or {}).items() if m not in hidden}
    hidden_legs = {row["leg"] for row in report["controls"] if row["model_key"] in hidden}
    prov = report["provenance"]
    if hidden and isinstance(prov.get("cells"), Mapping):
        prov = {**prov, "cells": {k: v for k, v in prov["cells"].items() if k not in hidden_legs}}
    tables = {"effects": {m: {mode: {b: e[b]["stats"] for b in BRANCHES} for mode, e in by.items()} for m, by in est.items()},
              "acc_by_leg": {m: {mode: {b: e[b]["acc_by_leg"] for b in BRANCHES} for mode, e in by.items()} for m, by in est.items()},
              "mode_deltas": {m: {mode: {b: e[b]["stats"] for b in BRANCHES} for mode, e in by.items()}
                              for m, by in report.get("mode_deltas", {}).items() if m not in hidden},
              "controls": ([row for row in report["controls"] if row["model_key"] not in hidden] if hidden else report["controls"]),
              "verdict": report["verdict"], "gates": report["gates"],
              "provenance": prov, "captions": table_captions, "sha": report["sha"],
              "analysis_code_sha": report.get("analysis_code_sha"),
              # the two footnote facts, as table data (the masked primary-leg draws per leg / per arm; the literal
              # server-default difference beside the effective equality)
              "notes": d48_notes(report),
              **({TREE_SET_KEY: report[TREE_SET_KEY]} if TREE_SET_KEY in report else {}), **extra}
    if shown_only:
        tables = scope_tables_to_shown(tables, view)
        # the write-time check: neither sidecar names a row the render leaves out, outside the selection record itself
        leaks = shown_only_leaks(numbers_doc, view) + shown_only_leaks(tables, view)
        if leaks:
            raise AnalysisError(f"{FIGURES_SHOWN_ONLY_KEY}: a sidecar names a row the render does not draw — {leaks[:6]}"
                                + (f" (+{len(leaks) - 6} more)" if len(leaks) > 6 else ""))
    (figs_dir / NUMBERS_FILENAME).write_text(json.dumps(numbers_doc, indent=1, default=str), encoding="utf-8")
    (figs_dir / TABLES_FILENAME).write_text(json.dumps(tables, indent=1, default=str), encoding="utf-8")
    return {"written": written, "skipped": skipped, "numbers": str(figs_dir / NUMBERS_FILENAME),
            "tables": str(figs_dir / TABLES_FILENAME), **extra}


# --------------------------------------------------------------------------- #
# W&B, the production boundary, the CLI                                        #
# --------------------------------------------------------------------------- #


def log_wandb(report: Mapping[str, Any], figs: Mapping[str, Any], *, mode: str, name: str | None = None,
              name_suffix: str = "") -> str | None:
    """``wandb.init`` on the W&B project, run ``m5t-<pin7><suffix>``; each
    figure under ``m5t/<stem>``; every contrast statistic, every mode delta and
    the numeric controls as summary fields; the figures dir as an artifact.
    ``disabled`` touches no network and imports nothing."""
    if mode == "disabled":
        return None
    try:
        import wandb
    except ImportError as e:    # pragma: no cover - venv-dependent
        raise AnalysisError("wandb is not importable — pass --wandb-mode disabled or use a venv that carries it") from e
    sha = report["sha"]
    name = name or f"{WANDB_RUN_PREFIX}{sha}{name_suffix}"
    from sisteer.wandb_entity import wandb_project
    run = wandb.init(project=wandb_project(WANDB_PURPOSE), name=name, mode=mode,
                     config={"experiment": EXPERIMENT, "sha": sha, "analysis_code_sha": report.get("analysis_code_sha"),
                             "task": TASK, **report["bootstrap"], f"figures_{FIGURES_EXCLUDED_KEY}": list(figs.get(FIGURES_EXCLUDED_KEY) or [])})
    for stem, paths in figs.get("written", {}).items():
        png = next((p for p in paths if p.endswith(".png")), None)
        if png:
            wandb.log({f"m5t/{stem}": wandb.Image(png)})
    summary: dict[str, Any] = {"verdict": report["verdict"]["verdict"]}
    if TREE_SET_KEY in report:
        summary[f"{TREE_SET_KEY}/note"] = tree_set_note(report[TREE_SET_KEY])
    for block, suffix in (("estimands", ""), ("mode_deltas", "-delta")):
        for m, by in (report.get(block) or {}).items():
            for mode_, est in by.items():
                for b in BRANCHES:
                    for c, s in est[b]["stats"].items():
                        for f in STAT_FIELDS:
                            summary[f"{m}/{mode_}{suffix}/{b}/{c}/{f}"] = s[f]
    for row in report["controls"]:
        for f, v in row["summary"].items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                summary[f"controls/{row['model_key']}/{row['cell']}/{row['mode']}/{f}"] = v
        for f in ("decode_tok_s_mean", "decode_tok_s_aggregate", "n_calls", "n_rejected", "n_resume_rows"):
            v = (row.get("ledger") or {}).get(f)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                summary[f"controls/{row['model_key']}/{row['cell']}/{row['mode']}/ledger_{f}"] = v
        # the re-draw readout's numbers (calls re-sent, items re-drawn and their share, max calls per item, ...)
        for f, v in ((row.get("ledger") or {}).get("redraws") or {}).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                summary[f"controls/{row['model_key']}/{row['cell']}/{row['mode']}/ledger_redraws_{f}"] = v
        # the two readouts' numbers (at-cap, no-reasoning) — the per-draw classes stay in the report
        for block in ("at_cap", "thinking"):
            for f, v in (row.get(block) or {}).items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    summary[f"controls/{row['model_key']}/{row['cell']}/{row['mode']}/{block}_{f}"] = v
    tr = report.get("truncation") or {}
    summary.update({"truncation/premise_failed": tr.get("premise_failed"),
                    "truncation/n_primary_at_or_above_threshold": len(tr.get("primary_at_or_above_threshold") or []),
                    "truncation/n_primary_over_1pct": len(tr.get("primary_over_1pct") or []),
                    "truncation/max_primary_share": tr.get("max_primary_share")})
    run.summary.update(summary)
    # the artifact is EXACTLY the manifest's files (the figures written by this render and its two sidecars), never the directory:
    # a re-used directory can hold a previous render's files, and those must not travel under this run's config
    files = [p for paths in figs.get("written", {}).values() for p in paths] + [str(figs[k]) for k in ("numbers", "tables") if figs.get(k)]
    if files:
        art = wandb.Artifact(f"figures-{name}", type="figures")
        for p in files:
            art.add_file(str(p), name=Path(p).name)
        run.log_artifact(art)
    url = getattr(run, "url", None)
    run.finish()
    return url


PUBLICATION_GATE_KEY = "publication_gate"


def publication_gate_record(report: Mapping[str, Any], figs: Mapping[str, Any], figs_dir: Path, *,
                            sidecar: str = NUMBERS_FILENAME, present_modes: Sequence[str] | None = None) -> dict[str, Any]:
    """Plan P5's ``check_publication_no_psr`` over THIS render: the figures directory and its rows sidecar, with the
    render's own selection as the forbidden set — the excluded arms, the excluded families, and every sampling mode of
    the report's matrix the ``--figure-modes`` selection left out. Raises :class:`PG.PublicationRefused` (the CLI turns
    it into exit ``PG.EXIT_PUBLICATION_REFUSED`` before any mirror or W&B call). Strings and lists only (B1).
    ``sidecar`` / ``present_modes``: the instrument's rows-sidecar name and the sampling modes its matrix carries (the
    GSM8K analysis passes its own; the defaults read this instrument's)."""
    written = _mapping(figs.get("written"))
    if not written:
        return {"status": PG.GATE_PASS, "figures": [], "note": "no figure was written by this run — nothing to publish", "rule": PG.RULE}
    modes = figs.get(FIGURES_MODES_KEY)
    present = list(present_modes) if present_modes is not None else list(dict.fromkeys(str(r["sampling_mode"]) for r in report["cells"]))
    excluded_modes = [m for m in present if m not in set(modes)] if modes is not None else []
    return PG.check_publication_no_psr(figs_dir, sidecar=sidecar, stems=list(written),
                                       excluded_arms=list(figs.get(FIGURES_EXCLUDED_ARMS_KEY) or []),
                                       excluded_families=list(figs.get(FIGURES_EXCLUDED_KEY) or []), excluded_modes=excluded_modes)


def production_refusals(report: Mapping[str, Any]) -> list[str]:
    """Why ``report`` is NOT a production result — every reason by name, or ``[]``:
    a FAILED gate; a PENDING gate not waivable; an unusable leg; a comparison
    that did not run; a REQUIRED mode delta that did not run (the records
    themselves are read, not only the gate that folds them); the estimand
    withheld; the word withheld; and first a ``tree_set`` whose verification
    refused — a report written at the startup boundary carries that block
    alone."""
    gates = _mapping(report.get("gates"))
    out: list[str] = []
    problems = _mapping(report.get(TREE_SET_KEY)).get("problems")
    if problems:
        out.append(f"{TREE_SET_KEY}: " + "; ".join(str(p) for p in problems))
    if gates.get("failed"):
        out.append(f"gate(s) FAILED: {', '.join(gates['failed'])}")
    if gates.get("pending_unwaivable"):
        out.append("gate(s) PENDING and not waivable: " + "; ".join(f"{it['gate']}: {it['detail']}" for it in gates["pending_unwaivable"]))
    if report.get("unusable_legs"):
        out.append(f"unusable leg(s): {sorted(report['unusable_legs'])}")
    not_run = comparisons_not_run(report.get("comparisons", []))
    if not_run:
        out.append(f"comparison(s) did not run: {not_run}")
    deltas_not_run = [d["name"] for d in report.get("mode_delta_records", []) if d.get("estimands") is None]
    if deltas_not_run:
        out.append(f"required mode delta(s) did not run: {deltas_not_run}")
    if "estimands" not in report:
        out.append("the estimand did not run (no `estimands` block)")
    verdict = _mapping(report.get("verdict"))
    if not out and verdict.get("word") is None:
        out.append(f"verdict withheld: {verdict.get('reason')}")
    return out


def matched_anomaly_classes(report: Mapping[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    """The split-anomaly CLASSES a report carries per leg (``gates.split_anomalies.counts.<leg>.classes`` — run E's
    reading of the raw bodies), for a run WITHOUT eval logs to take as its own under ``class_source: matched_report``.
    Only legs whose classes came from a log or a matched report are taken; a leg the report itself left unclassified
    contributes nothing (so the run stays fail-closed on it)."""
    out: dict[str, dict[str, dict[str, Any]]] = {}
    counts = _mapping(_mapping(_mapping(report.get("gates")).get("split_anomalies")).get("counts"))
    for leg, rec in counts.items():
        rec = _mapping(rec)
        classes = _mapping(rec.get("classes"))
        if classes and rec.get("class_source") in CLASS_SOURCES_CLASSIFIED:
            out[str(leg)] = {str(i): dict(_mapping(c)) for i, c in classes.items()}
    return out


# ---------------------------------------------------------------------------- #
# the paper rows: the flat per-row export beside the report                     #
# ---------------------------------------------------------------------------- #
#
# Every per-leg `point` above is the CONDITIONAL estimand (n_correct / n_scored). The paper's
# capability panel draws SCORE-AS-IS (n_correct / n_items: every unscored draw wrong) — the
# `score_as_is` branch every comparison already files beside its primary. `paper_rows_of_report`
# reads both from the written report (the comparisons' estimands, the legs' summaries and
# `unscored` blocks) and files them flat, one row per primary-mode leg, with the four count
# fields beside them. It computes no estimator and never rewrites the report.

#: a leg with no `view` field (the 24-cell design predates the field: every cell of it is the main view) reads this
MATH500_DEFAULT_VIEW = "main"


def _pr_leg_arms(report: Mapping[str, Any]) -> dict[str, tuple[str, Mapping[str, Any]]]:
    """``leg key -> (arm, comparison)`` from the PRIMARY comparisons' ``leg_keys`` — the report names its own arms."""
    out: dict[str, tuple[str, Mapping[str, Any]]] = {}
    for c in report.get("comparisons", []):
        if not isinstance(c, Mapping) or not c.get("primary"):
            continue
        anchor = _mapping(c.get("bare_anchor"))
        for arm, lk in (c.get("leg_keys") or {}).items():
            if arm == KIND_BARE and anchor and anchor.get("view") != c.get("view"):
                # E3: a non-main view's comparison is anchored on ANOTHER view's bare leg (the main comparison's own row); the
                # anchor role adds no paper row here, and naming the leg twice would refuse the whole export
                continue
            if lk in out and out[lk][1] is not c:
                raise AnalysisError(f"paper rows: leg {lk} is named by two primary comparisons ({out[lk][1].get('name')!r}, {c.get('name')!r})")
            out[lk] = (arm, c)
    return out


def _pr_stamp(leg: Mapping[str, Any], key: str) -> Any:
    for holder in (leg, leg.get("identity"), leg.get("summary")):
        if isinstance(holder, Mapping) and holder.get(key) is not None:
            return holder[key]
    return None


def _pr_refusal_shaped(leg: Mapping[str, Any], lk: str) -> tuple[int | None, int | None, str]:
    """``(count, denominator, note)``: the leg's refusal-shaped count (schema paper_rows/4) and the draws the classifier examined for it
    (``n_examined`` -> ``n_refusal_shaped_of``), READ from its filed readout (``sisteer.capeval.tinycap.refusal_shaped`` over the scored
    draws' content channel) and held to the readout's own contract; a leg written before the class existed carries no block -> both None
    (absence, never zero), said in the note. A present, off-contract block refuses by name."""
    block = leg.get("refusal_shaped")
    if block is None:
        return None, None, "n_refusal_shaped / n_refusal_shaped_of: this record carries no refusal-shaped readout (written before the class existed) -> null together"
    problems = RFS.readout_problems(block, n_scored=int((leg.get("summary") or {}).get("n_scored") or 0))
    if problems:
        raise AnalysisError(f"paper rows: legs[{lk}].refusal_shaped: " + "; ".join(problems))
    return (int(block["n_refusal_shaped"]), int(block["n_examined"]),
            f"legs[{lk}].refusal_shaped: n_refusal_shaped / n_examined (-> n_refusal_shaped_of) ({RFS.RULE})")


def _pr_object_generation(leg: Mapping[str, Any]) -> tuple[str, str | None]:
    if not leg.get("fitted"):
        return PRW.GEN_NONE, None
    stamp = _pr_stamp(leg, "object_generation")
    if stamp is None:
        return PRW.GEN_NA, "object_generation: this record carries no generation stamp (written before the stamp existed) -> n/a"
    if stamp not in PRW.OBJECT_GENERATION_STAMPS:
        raise AnalysisError(f"paper rows: object_generation stamp {stamp!r} not in {PRW.OBJECT_GENERATION_STAMPS}")
    return str(stamp), None


def _pr_selection_outcome(leg: Mapping[str, Any]) -> tuple[str | None, str | None]:
    stamp = _pr_stamp(leg, "selection_outcome")
    if stamp is None:
        return None, None
    if stamp not in PRW.SELECTION_OUTCOMES:
        raise AnalysisError(f"paper rows: selection_outcome {stamp!r} not in {PRW.SELECTION_OUTCOMES}")
    return str(stamp), None


def _pr_selection(leg: Mapping[str, Any], *, selection_outcome: str | None) -> tuple[Mapping[str, Any] | None, str | None]:
    """The leg's DECLARED ``selection`` block (schema /3), read off the record like ``selection_outcome`` — never composed here.
    Refuses by name (:func:`PRW.selection_problems`) a block that is not a reading: no ``ladder_cap``, an ``outcome`` that
    disagrees with the row's word, the override word beside a within-budget reading or a ``rule_returns`` that names the
    selected dose, an alpha-shaped dose on the raw-coefficient arm. Absent -> ``None`` (the export's verifier requires it
    under the override word)."""
    block = _pr_stamp(leg, "selection")
    if block is None:
        return None, None
    problems = PRW.selection_problems(block, tag=f"legs[{leg.get('leg')}]", selection_outcome=selection_outcome, cell=leg.get("cell"))
    if problems:
        raise AnalysisError("paper rows: " + "; ".join(problems))
    return dict(block), "selection: the admission record's block, carried verbatim off the matrix row (both readings, the budget, the ladder cap)"


def _pr_fit_provenance(leg: Mapping[str, Any]) -> tuple[Mapping[str, Any] | None, str | None]:
    """The leg's DECLARED ``fit_provenance`` block (schema /3): what the served direction was fitted on, verbatim off the record.
    A present block missing any :data:`PRW.FIT_PROVENANCE_FIELDS` key refuses by name; absent -> ``None`` (the verifier requires
    it on the all-positions Arditi arm — the export never fills it)."""
    block = _pr_stamp(leg, "fit_provenance")
    if block is None:
        return None, None
    problems = PRW.fit_provenance_problems(block, tag=f"legs[{leg.get('leg')}]")
    if problems:
        raise AnalysisError("paper rows: " + "; ".join(problems))
    return dict(block), "fit_provenance: the admission record's block, carried verbatim off the matrix row"


def _pr_serving_commit(leg: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """The commit of the SERVING STACK the leg served under (schema /3): a producer's explicit ``serving_commit`` stamp when the
    record carries one (validated as before), else DERIVED as the GSM8K reader derives it — ``standup_jobs.jsonl[0].sha`` of the
    leg's own standup job (the tree the STANDUP ran from, i.e. the checkout that built and served the object; ``collect_leg``
    files it as ``standup_record.standup_sha``), cross-checked EQUAL to the manifest's ``producer.tree_sha`` (the DRIVER's tree:
    for these launchers the standup and the driver run from one eval-repo checkout). Two trees under one row refuse by name;
    no standup record read leaves ``None`` and the export refuses a served row of the arm that requires it
    (:func:`PRW.serving_commit_problems`) rather than filing the driver's tree under this name."""
    stamp = _pr_stamp(leg, "serving_commit")
    if stamp is None:
        rec = _mapping(leg.get("standup_record"))
        sha, tree = rec.get("standup_sha"), rec.get("producer_tree_sha")
        if sha and tree and str(sha) != str(tree):
            raise AnalysisError(f"paper rows: legs[{leg.get('leg')}]: the standup ran from {str(sha)[:12]}… but the driver's producer tree is "
                                f"{str(tree)[:12]}… — two stacks under one row; serving_commit is not derivable from a record whose standup and "
                                "driver disagree")
        if sha:
            return str(sha), ("serving_commit: standup_jobs.jsonl[0].sha (the tree the standup served from), cross-checked equal to the "
                              "manifest's producer.tree_sha")
        return None, None
    problems = PRW.serving_commit_problems(stamp, tag=f"legs[{leg.get('leg')}]", cell=leg.get("cell"), has_reading=True)
    if problems:
        raise AnalysisError("paper rows: " + "; ".join(problems))
    return str(stamp), "serving_commit: the serving stack's commit as the store record stamps it"


def paper_rows_of_report(report: Mapping[str, Any], *, report_path: Path | str | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The paper rows of a written report: one per PRIMARY-mode leg (the sensitivity sampling rows are never pooled
    with the primary, and are listed under ``not_exported`` by name), plus a null-count row for a primary
    leg with no summary (a named refusal). A leg the report's masked primary reads over fewer items takes the
    ``estimands_with_anomalies`` branch — the reading over every item posed — so ``n_paired`` is the items posed on
    every row; the mask is named in ``source``. Returns ``(rows, not_exported)``."""
    legs = report.get("legs") or {}
    pin, acs = report.get("sha"), report.get("analysis_code_sha")
    arms = _pr_leg_arms(report)
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    def src(reads: Sequence[str]) -> str:
        return PRW.source_text(report_path, store_pin=pin, analysis_code_sha=acs, reads=reads)

    for lk, leg in legs.items():
        if not leg.get("primary"):
            skipped.append({"leg": lk, "reason": f"sampling mode {leg.get('mode')!r} is a sensitivity row (never pooled with the primary, D1 / D26), not a paper row"})
            continue
        if lk not in arms:
            skipped.append({"leg": lk, "reason": "a primary-mode leg named by no primary comparison"})
            continue
        arm, comp = arms[lk]
        if leg.get("kind") is not None and leg["kind"] != arm:
            raise AnalysisError(f"paper rows: legs[{lk}].kind {leg['kind']!r} != the comparison's arm {arm!r}")
        mk, cid = leg["model_key"], leg["cell"]
        notes: list[str] = []
        view = leg.get("view")
        if view is None:
            view = MATH500_DEFAULT_VIEW
            notes.append(f"view: not a field of this record -> {MATH500_DEFAULT_VIEW!r} (the design's every cell is the main view)")
        gen, gen_note = _pr_object_generation(leg)
        sel, sel_note = _pr_selection_outcome(leg)
        selection, selection_note = _pr_selection(leg, selection_outcome=sel)
        fit_prov, fit_prov_note = _pr_fit_provenance(leg)
        serving_commit, serving_commit_note = _pr_serving_commit(leg)
        notes += [x for x in (gen_note, sel_note, selection_note, fit_prov_note, serving_commit_note) if x]
        prov3 = {"selection": selection, "serving_commit": serving_commit, "fit_provenance": fit_prov}
        summary = leg.get("summary") or {}
        cap = int(summary.get("max_tokens") or ((leg.get("identity") or {}).get("max_tokens_by_task") or {}).get(TASK) or 0)
        if not cap:
            raise AnalysisError(f"paper rows: legs[{lk}] names no cap (summary.max_tokens)")
        leg_name = f"r{cap}"
        if (leg.get("identity") or {}).get("cap_rung") is None:
            rung_state = PRW.RUNG_NOT_APPLICABLE
            notes.append(f"rung_state: one cap ({cap:,} = the ceiling); no rung ladder on this instrument")
        else:
            rung_state = PRW.RUNG_COMPLETE
            notes.append("rung_state: this leg IS a rung reading (identity.cap_rung)")
        if not summary:
            why = leg.get("refused") or leg.get("missing") or "no summary"
            rs = PRW.RUNG_REFUSED if leg.get("refused") else PRW.RUNG_NEVER_RUN
            rows.append(PRW.make_row(instrument=PRW.INSTRUMENT_MATH500, model_key=mk, view=view, cell=cid, arm=arm, cap=cap, leg=leg_name,
                                     **{f: None for f in PRW.COUNT_FIELDS}, rung_state=rs, object_generation=gen, selection_outcome=sel,
                                     source=src([f"legs[{lk}].summary = null: {str(why)[:160]}", *notes]), **prov3))
            continue
        if comp.get(REFUSED_BY_DESIGN_KEY):
            # the comparison refused BY DESIGN (its declared expected refusal matched exactly — the O106 cross-engine ruling: no native
            # `none` anchor is run): there is no pairing to read, and the ruling reports the ARM's numbers AS the arm's numbers. The row
            # carries the leg's OWN counts (n_correct from the leg's items under the primary branch, the summary's n / n_scored /
            # n_unscored), the points n_correct / n_scored and n_correct / n, and the schema's exact binomial interval — the CI the
            # schema files where the report has no bootstrap for a leg (the GSM8K rung rows) — with the refusal named in `source`
            code, why = comp.get(ROW_EXPECTED_REFUSAL_KEY), str(comp.get("reason"))
            items = T._branch_items(leg, BRANCH_PRIMARY)
            n = len(items)
            ns = sum(m for m, _v in items.values())
            nc = int(round(sum(float(v) for m, v in items.values() if m and v is not None)))
            nu = int(summary["n_unscored"])
            if int(summary["n"]) != n or int(summary["n_scored"]) != ns or nu != n - ns:
                raise AnalysisError(f"paper rows: legs[{lk}].summary (n {summary['n']}, n_scored {summary['n_scored']}, n_unscored {nu}) != the "
                                    f"leg's items (n {n}, n_scored {ns}, n_unscored {n - ns})")
            if ns and summary.get("acc") is not None and abs(float(summary["acc"]) - nc / ns) > 1e-6:
                raise AnalysisError(f"paper rows: legs[{lk}].summary.acc {summary['acc']} != n_correct / n_scored = {nc}/{ns} from the leg's items")
            level = float(_mapping(report.get("bootstrap")).get("level") or pr.DEFAULT_CI)
            lo_c, hi_c = PRW.exact_binomial_ci(nc, ns, level) if ns else (None, None)
            lo_s, hi_s = PRW.exact_binomial_ci(nc, n, level)
            a_p = {"point": (nc / ns if ns else None), "ci_low": lo_c, "ci_high": hi_c}
            a_s = {"point": nc / n, "ci_low": lo_s, "ci_high": hi_s}
            ci_method = PRW.CI_EXACT
            reads = [f"comparisons[{comp['name']}]: refused by design ({code}) — the arm's numbers are reported as the arm's numbers, the "
                     f"pairing's absence accepted; no estimand read — {why[:200]}",
                     f"legs[{lk}].items under the primary branch: n_correct {nc} over n_scored {ns} of n {n}; point_conditional = n_correct / n_scored, "
                     f"point_score_as_is = n_correct / n; the {PRW.CI_EXACT} interval at level {level} (no bootstrap is filed for a lone arm)"]
        else:
            masked_ids = (comp.get("masked_items") or {}).get(arm) or []
            if masked_ids:
                branch = comp.get("estimands_with_anomalies")
                if not isinstance(branch, Mapping):
                    raise AnalysisError(f"paper rows: comparisons[{comp['name']!r}] masks {arm} but files no estimands_with_anomalies")
                reads = [f"comparisons[{comp['name']}].estimands_with_anomalies.{{primary,score_as_is}}: the reading over every item posed — the report's "
                         f"masked primary (D48) removes {len(masked_ids)} item(s) from this leg's population ({', '.join(masked_ids)}) and its "
                         f"score_as_is reads over {comp['estimands']['score_as_is']['acc_by_leg'][arm]['n_scored']} items there; n_paired here is the items posed"]
            else:
                branch = comp["estimands"]
                reads = [f"comparisons[{comp['name']}].estimands.{{primary,score_as_is}}"]
            e_p, e_s = branch["primary"], branch["score_as_is"]
            a_p, a_s = e_p["acc_by_leg"][arm], e_s["acc_by_leg"][arm]
            n = int(e_p["n_items"])
            ns, nc, nu = int(a_p["n_scored"]), int(a_p["n_correct"]), int(summary["n_unscored"])
            if int(summary["n"]) != n or int(summary["n_scored"]) != ns or nu != n - ns or int(a_p["n_censored"]) != nu:
                raise AnalysisError(f"paper rows: legs[{lk}].summary (n {summary['n']}, n_scored {summary['n_scored']}, n_unscored {nu}) != the "
                                    f"estimand's (n_items {n}, n_scored {ns}, n_censored {a_p['n_censored']})")
            if int(a_s["n_correct"]) != nc or int(a_s["n_scored"]) != n or int(e_s["n_items"]) != n:
                raise AnalysisError(f"paper rows: legs[{lk}]: the score_as_is branch reads {a_s['n_correct']}/{a_s['n_scored']} over {e_s['n_items']} items, "
                                    f"not n_correct {nc} over the {n} items posed")
            ci_method = PRW.CI_BOOTSTRAP
        nac, rate, untnac = int(summary["n_at_cap"]), float(summary["at_cap_rate"]), int(summary["n_unterminated_not_at_cap"])
        if abs(rate - nac / n) > 1e-6:
            raise AnalysisError(f"paper rows: legs[{lk}].summary.at_cap_rate {rate} != n_at_cap / n = {nac / n}")
        reads.append(f"acc_by_leg.{arm}: point, ci_low, ci_high, n_correct, n_scored, n_censored; n_items (-> n_paired)")
        reads.append(f"legs[{lk}].summary: n, n_scored, n_unscored, n_at_cap, at_cap_rate, n_unterminated_not_at_cap")
        unscored = leg.get("unscored")
        if isinstance(unscored, Mapping) and unscored.get("n_unscored") is not None:
            if int(unscored["n_unscored"]) != nu:
                raise AnalysisError(f"paper rows: legs[{lk}].unscored.n_unscored {unscored['n_unscored']} != summary.n_unscored {nu}")
            unterminated = int(unscored.get("n_unterminated_at_cap") or 0) + int(unscored.get("n_unterminated_not_at_cap") or 0)
            reads.append(f"legs[{lk}].unscored: n_unterminated_at_cap + n_unterminated_not_at_cap (-> unterminated_think_block); the remainder -> other")
        else:
            unterminated = int(summary.get("n_think_block_unterminated") or 0)
            reads.append(f"legs[{lk}].summary.n_think_block_unterminated (-> unterminated_think_block); the remainder -> other")
        ur, ur_note = PRW.unscored_reason(nu, unterminated_think_block=unterminated)
        nrs, nrs_of, nrs_note = _pr_refusal_shaped(leg, lk)
        reads.append(nrs_note)
        rows.append(PRW.make_row(instrument=PRW.INSTRUMENT_MATH500, model_key=mk, view=view, cell=cid, arm=arm, cap=cap, leg=leg_name,
                                 n_paired=n, n_scored=ns, n_unscored=nu, n_at_cap=nac, at_cap_rate=rate, n_unterminated_not_at_cap=untnac,
                                 n_correct=nc, point_conditional=a_p["point"], ci_conditional_low=a_p["ci_low"], ci_conditional_high=a_p["ci_high"],
                                 point_score_as_is=a_s["point"], ci_score_as_is_low=a_s["ci_low"], ci_score_as_is_high=a_s["ci_high"],
                                 ci_method=ci_method, rung_state=rung_state, object_generation=gen, selection_outcome=sel, unscored_reason=ur,
                                 source=src([*reads, *([ur_note] if ur_note else []), *notes]), n_refusal_shaped=nrs, n_refusal_shaped_of=nrs_of, **prov3))
    return rows, skipped


def write_paper_rows_beside(report: Mapping[str, Any], report_path: Path | str) -> dict[str, Any]:
    """``paper_rows.json`` + ``paper_rows.csv`` beside ``report_path`` (nothing else there is touched)."""
    rows, skipped = paper_rows_of_report(report, report_path=report_path)
    return PRW.write_paper_rows(rows, Path(report_path).expanduser().parent, instrument=PRW.INSTRUMENT_MATH500, report_path=report_path,
                                not_exported=skipped)


def paper_rows_only(path: Path) -> int:
    """``--paper-rows-only REPORT``: regenerate the export from an EXISTING production report — no store opened, no
    analysis re-run, the report not rewritten. A report that is not a production result gets no paper rows (exit 3)."""
    path = Path(path).expanduser().resolve()
    report = _read_json(path)
    if not isinstance(report, Mapping):
        raise AnalysisError(f"--paper-rows-only {path}: not a JSON object")
    refusals = production_refusals(report)
    if refusals:
        print(f"[m5t] {path}: NOT a production result — no paper rows written — " + "; ".join(refusals), file=sys.stderr)
        return EXIT_NOT_PRODUCTION
    written = write_paper_rows_beside(report, path)
    print(f"[m5t] paper rows -> {written['json']} ({written['n_rows']} rows, {written['n_not_exported']} sensitivity leg(s) listed under not_exported) + {written['csv']}")
    return 0


def gate_report(path: Path) -> int:
    report = _read_json(Path(path).expanduser())
    if not isinstance(report, Mapping):
        raise AnalysisError(f"--gate-report {path}: not a JSON object")
    refusals = production_refusals(report)
    if refusals:
        print(f"[m5t] {path}: NOT a production result — " + "; ".join(refusals), file=sys.stderr)
        return EXIT_NOT_PRODUCTION
    print(f"[m5t] {path}: production result ({len(report.get('comparisons', []))} comparisons, gates all passed)")
    return 0


def diff_reports(path_a: Path, path_b: Path) -> int:
    a, b = _read_json(Path(path_a).expanduser()), _read_json(Path(path_b).expanduser())
    if not isinstance(a, Mapping) or not isinstance(b, Mapping):
        raise AnalysisError(f"--diff-reports: {path_a} / {path_b} must both be JSON objects")
    d = numeric_leaf_diff(a, b)
    if not d["equal"]:
        lines = [f"  differing {T._path_text(p)}: {va} != {vb}" for p, va, vb in d["differing"][:10]]
        lines += [f"  type changed {T._path_text(p)}: {va!r} vs {vb!r}" for p, va, vb in d["type_changed"][:10]]
        lines += [f"  only in {path_a}: {T._path_text(p)}" for p in d["missing"][:10]]
        lines += [f"  only in {path_b}: {T._path_text(p)}" for p in d["added"][:10]]
        print(f"[m5t] the reports' numeric leaves DIFFER — {len(d['differing'])} differing, {len(d['type_changed'])} type changed, "
              f"{len(d['missing'])} missing, {len(d['added'])} added over {d['shared_numeric']} shared:\n" + "\n".join(lines),
              file=sys.stderr)
        return EXIT_REPORTS_DIFFER
    print(f"[m5t] numeric leaves equal: {d['shared_numeric']} shared (outside {', '.join(REPORT_DIFF_SKIP)} and the text signatures)")
    return 0


def print_table(report: Mapping[str, Any]) -> None:
    print(f"{EXPERIMENT} — store pin {report['sha']}, analysis code {report['analysis_code_sha'][:12]}")
    if TREE_SET_KEY in report:
        print(f"  {tree_set_note(report[TREE_SET_KEY])}")
    for row in report["controls"]:
        s = row["summary"]
        state = row["missing"] or row["refused"] or "ok"
        acc = "  —  " if s.get("acc") is None else f"{s['acc']:.3f}"
        print(f"  {row['leg']:<70} acc {acc} n_unscored {s.get('n_unscored')!s:>4} at_cap {s.get('n_at_cap')!s:>3}  {state[:60]}")
    for c in report["comparisons"]:
        if c["estimands"] is None:
            print(f"  {c['name']}: no estimand — {c['reason']}")
            continue
        for k, v in c["verdicts"].items():
            p = c["estimands"][BRANCH_PRIMARY]["stats"][k]
            print(f"  {c['name']} {k}: {T._pp(p['point'])} CI [{T._pp(p['ci_low'])}, {T._pp(p['ci_high'])}] -> {v['word']}")
    g = report["gates"]
    print(f"  gates failed: {g['failed'] or 'none'}; pending: {g['pending'] or 'none'}")
    tr = report.get("truncation") or {}
    state = {True: "FAILED", False: "holds", None: "undetermined"}[tr.get("premise_failed")]
    print(f"  at-cap premise (a primary cell >= 3 %): {state}; primary cells over the 1 % line: {tr.get('primary_over_1pct') or 'none'}")
    print(f"  verdict: {report['verdict']['verdict']}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen-root", action="append", default=[], metavar="PATH[=PIN]",
                    help="a local copy of a wave root (controls/<model>/<cell>/<mode>/...; repeatable). Bare PATH: bound to --sha, a leg "
                         "under it may be from --sha or any --also-sha tree (D131). PATH=PIN (plan P5 / B6): a wave-isolated pull bound "
                         "to that ONE admitted tree (--sha or an --also-sha); a row's out root under two roots refuses (a filesystem union)")
    ap.add_argument("--sha", default=None, help="the store pin: the wave sha every cell's producer tree carries")
    ap.add_argument("--also-sha", action="append", default=[], metavar="SHA40",
                    help="D131: a further producer tree the wave's cells may come from (repeatable; a full 40-hex sha), "
                         "verified on --repo-root to differ from --sha in --tree-diff-allow paths only; needs --tree-set-reason")
    ap.add_argument("--tree-diff-allow", action="append", default=[], metavar="GLOB",
                    help="a path glob the declared trees may differ from the primary in (repeatable; fnmatch, case-sensitive)")
    ap.add_argument("--tree-diff-allow-file", type=Path, default=None, metavar="REVIEW_MD",
                    help="the REVIEWED allowlist file (plan P5 / B7: configs/sdfbase/eight_arms/tree_set_review_<pub7>_<wave7>.md): every "
                         "line `- `<glob>` — <evidence>` contributes one glob to --tree-diff-allow; the path and the per-glob evidence are "
                         "filed as tree_set.allow_file / tree_set.allow_evidence; a glob without evidence refuses")
    ap.add_argument("--tree-set-reason", default=None, metavar="TEXT",
                    help="why the cells come from more than one tree — filed verbatim as tree_set.why (required with --also-sha)")
    ap.add_argument("--store-pins", type=Path, action="append", default=None, metavar="JSON",
                    help="JSON {'<model>/<cell>/<mode>[@view]': '<tree sha>'} (repeatable, merged): every leg admitted by ITS OWN producer "
                         "tree — the per-leg admission (PR #137's, from LAUNCH RECORDS); --sha is then the record label and the published "
                         "legs' pin; a PATH=PIN gen root may name any pinned tree; not with --also-sha")
    ap.add_argument("--waves", default=None, metavar="WAVE[,WAVE]",
                    help="the REQUIRED-EVIDENCE set (plan P5 / review 3): the evidence rows (status launch; `complete` tolerated as a synonym) of these waves are "
                         "mandatory — a missing leg fails the population gate; held / deferred rows and rows of other waves land in "
                         "design.status_by_row and are never collected. Absent: every evidence row of the matrix (the published behaviour)")
    ap.add_argument("--cells", default=None, help="the cells matrix: scripts/sdfbase/math500/cells.py (M5T_CELLS) or a JSON list "
                                                  "(default: the script tree's scripts/sdfbase/math500/cells.py)")
    ap.add_argument("--sampling-rows", default=None, help="the pinned run config of sampling rows (YAML or JSON; default: the script "
                                                          "tree's configs/sdfbase/math500/sampling_rows.yaml)")
    ap.add_argument("--render-facts", default=None, help="a JSON family -> render fact table (default: contracts.FAMILY_RENDER_FACTS)")
    ap.add_argument("--math500-ids", default=None, help="the canonical 500 ids: a JSON list file, or 'plan' to derive them from "
                                                        "inspect_served.math500_sample_ids(); absent -> the population gate is PENDING")
    ap.add_argument("--pins", type=Path, default=None, help="the object pin file (default: configs/sdfbase_full22_pins.json)")
    ap.add_argument("--repo-root", type=Path, default=None, help="the tree whose configs/ carry the model and object pins, and the "
                                                                    "repository the declared trees are verified on (default: this script's)")
    ap.add_argument("--gate-report", type=Path, default=None, metavar="REPORT", help="exit non-zero unless REPORT is a production result")
    ap.add_argument("--diff-reports", type=Path, nargs=2, default=None, metavar=("A", "B"), help="exit non-zero unless A and B agree on every numeric leaf")
    ap.add_argument("--paper-rows-only", type=Path, default=None, metavar="REPORT",
                    help="regenerate paper_rows.json + paper_rows.csv beside an EXISTING production report and exit (D103 / D104): "
                         "no store opened, no analysis re-run, the report not rewritten")
    ap.add_argument("--require-report-match", type=Path, default=None, metavar="REPORT",
                    help="compare THIS run's report with REPORT's numeric leaves BEFORE any figure, mirror or W&B call")
    ap.add_argument("--analysis-sha", default=None, help="the git sha of the analysis code (default: read from this script's tree)")
    ap.add_argument("--bootstrap", type=int, default=pr.DEFAULT_BOOTSTRAP, metavar="B")
    ap.add_argument("--seed", type=int, default=pr.DEFAULT_BOOTSTRAP_SEED)
    ap.add_argument("--ci", type=float, default=pr.DEFAULT_CI)
    ap.add_argument("--allow-pending-gates", action="store_true", help="file the words with a WAIVABLE producer stamp pending (named)")
    ap.add_argument("--no-eval-logs", action="store_true", help="skip the .eval log cross-check (the figure venv cannot read them)")
    ap.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    ap.add_argument("--wandb-name", default=None, help=f"the FULL W&B run name (default {WANDB_RUN_PREFIX}<pin7><suffix>)")
    ap.add_argument("--wandb-name-suffix", default="")
    ap.add_argument("--figs-dir", type=Path, default=None, help="where the figures go (default: beside --out, ./figs)")
    ap.add_argument("--no-figs", action="store_true")
    ap.add_argument("--figure-exclude-family", action="append", default=[], metavar="MODEL_KEY",
                    help="leave this model family out of every FIGURE and of the per-model / per-leg entries of the tables JSON (repeatable; "
                         "must be a model key of the report's controls). The report's cells, legs, comparisons and gates keep every leg and "
                         f"no numeric leaf moves; the exclusion is recorded as figures.{FIGURES_EXCLUDED_KEY} and in both sidecars")
    ap.add_argument("--figure-exclude-arm", action="append", default=[], metavar="CELL",
                    help="leave this matrix CELL (arm) out of every FIGURE (repeatable; plan P5: validated against the report — an arm "
                         "absent from the report is an error; the bare cell cannot be excluded). Presentation only: no aggregate is "
                         f"recomputed, titles carry the full-report and shown counts; recorded as figures.{FIGURES_EXCLUDED_ARMS_KEY}")
    ap.add_argument("--figure-modes", default=None, metavar="MODE[,MODE]",
                    help="draw only these sampling modes (plan P5: the paper view keeps thinking_card and drops the general / battery "
                         f"sensitivity rows BY MODE — an arm id cannot select them); recorded as figures.{FIGURES_MODES_KEY}. A selection "
                         f"without the primary row skips the figures that draw it only ({', '.join(PRIMARY_ONLY_FIGURES)}; recorded in the "
                         "sidecar's `skipped`) — nothing is drawn from an excluded row")
    ap.add_argument("--figure-shown-only", action="store_true",
                    help="with a selector in play, the at-cap share / loop split / no-reasoning figures, every legend and caption, the "
                         "numbers sidecar AND the tables JSON describe the SHOWN rows only (labelled as shown; no report-wide value, no "
                         "verdict word, gate leg or note of a row not drawn; a one-row render skips the sampling-row figure); the render "
                         "refuses if either sidecar still names a hidden row; refused without a selector; recorded "
                         f"as figures.{FIGURES_SHOWN_ONLY_KEY} only when set, so the default render is byte-identical")
    ap.add_argument("--mirror-dir", type=Path, default=Path(MIRROR_DIR_DEFAULT))
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--out", type=Path, default=None, help="write the full JSON report here")
    return ap


def _math500_ids(arg: str | None) -> list[str] | None:
    if arg is None:
        return None
    if arg == "plan":
        from sisteer.capeval.tiers import inspect_served as isv
        ids = [str(i) for i in isv.math500_sample_ids()]
    else:
        doc = _read_json(Path(arg).expanduser())
        if not isinstance(doc, list) or not all(isinstance(i, str) for i in doc):
            raise AnalysisError(f"--math500-ids {arg}: expected a JSON list of item ids")
        ids = doc
    if len(set(ids)) != N_ITEMS:
        raise AnalysisError(f"--math500-ids names {len(set(ids))} unique ids, not {N_ITEMS}")
    return ids


def main(argv: Sequence[str] | None = None) -> int:
    ap = build_parser()
    a = ap.parse_args(argv)
    if a.gate_report is not None:
        return gate_report(a.gate_report)
    if a.diff_reports is not None:
        return diff_reports(*a.diff_reports)
    if a.paper_rows_only is not None:
        return paper_rows_only(a.paper_rows_only)
    if not a.gen_root or a.sha is None:
        ap.error("--gen-root and --sha are required for an analysis run")
    if a.also_sha and not (a.tree_set_reason or "").strip():
        ap.error("--also-sha requires --tree-set-reason <text>: why the wave's cells come from more than one tree")
    if a.store_pins and a.also_sha:
        ap.error("--store-pins and --also-sha are two admissions of the same legs — give one")
    repo_root = (a.repo_root or _HERE.parent).expanduser().resolve()
    out = a.out.expanduser().resolve() if a.out else None
    waves = [w.strip() for w in str(a.waves).split(",") if w.strip()] if a.waves is not None else None
    if a.waves is not None and not waves:
        ap.error("--waves names no wave")
    figure_modes_arg = [m.strip() for m in str(a.figure_modes).split(",") if m.strip()] if a.figure_modes is not None else None
    if a.figure_modes is not None and not figure_modes_arg:
        ap.error("--figure-modes names no mode")
    # B7: the reviewed allowlist FILE joins the --tree-diff-allow globs; its path and per-glob evidence ride the tree_set block
    allow = list(a.tree_diff_allow)
    allow_file: str | None = None
    allow_evidence: dict[str, str] | None = None
    if a.tree_diff_allow_file is not None:
        globs, allow_evidence = load_tree_diff_allow_file(a.tree_diff_allow_file)
        allow += [g for g in globs if g not in allow]
        allow_file = str(Path(a.tree_diff_allow_file).expanduser())
    # the declared producer trees are verified on the local repository BEFORE any leg is read; a refusal here is a
    # production refusal by name (exit 3) — no store opened, no figure, mirror or W&B call. analyze() verifies the same
    # declaration again from its fields (it never trusts a verified block), so this pass only buys the early boundary.
    tree_set = verify_tree_set(a.sha, a.also_sha, allow, repo_root=repo_root, reason=a.tree_set_reason, allow_file=allow_file,
                               allow_evidence=allow_evidence)
    if tree_set["problems"]:
        stub = {"experiment": EXPERIMENT, "task": TASK, "sha": a.sha, TREE_SET_KEY: tree_set,
                "verdict": {"word": None, "verdict": NOT_MEASURABLE, "reason": f"{TREE_SET_KEY} refused before any leg was read"}}
        if out:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(stub, indent=1, default=str), encoding="utf-8")
        print("[m5t] NOT a production result — " + "; ".join(production_refusals(stub)), file=sys.stderr)
        return EXIT_NOT_PRODUCTION
    cells = a.cells or (repo_root / "scripts" / "sdfbase" / "math500" / "cells.py")
    rows = a.sampling_rows or (repo_root / "configs" / "sdfbase" / "math500" / "sampling_rows.yaml")
    # a run WITHOUT eval logs that must match a report takes that report's split-anomaly classes (run E read the raw
    # bodies; the figure venv cannot) — named `matched_report` on every leg that uses them, never inferred from the record
    matched: dict[str, dict[str, dict[str, Any]]] | None = None
    if a.no_eval_logs and a.require_report_match is not None:
        other0 = _read_json(a.require_report_match.expanduser().resolve())
        if not isinstance(other0, Mapping):
            raise AnalysisError(f"--require-report-match {a.require_report_match}: not a JSON object")
        matched = matched_anomaly_classes(other0)
    report = analyze(list(a.gen_root), a.sha, cells=cells, sampling_rows=rows, render_facts=a.render_facts,
                     math500_ids=_math500_ids(a.math500_ids), object_pins=a.pins, n_resamples=a.bootstrap, seed=a.seed,
                     ci=a.ci, allow_pending_gates=a.allow_pending_gates, read_eval_logs=not a.no_eval_logs,
                     analysis_sha=a.analysis_sha, repo_root=repo_root, tree_set=tree_set if a.also_sha else None, waves=waves,
                     anomaly_classes=matched, store_pins=list(a.store_pins) if a.store_pins else None)
    print_table(report)

    def write_report() -> None:
        if out:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
            print(f"\n[m5t] full report -> {out}")

    # the production boundary fails CLOSED and BEFORE any side effect: no figure,
    # no mirror, no W&B run for a report that is not a production result
    refusals = production_refusals(report)
    if refusals:
        write_report()
        print("[m5t] NOT a production result — " + "; ".join(refusals), file=sys.stderr)
        return EXIT_NOT_PRODUCTION
    if a.require_report_match is not None:
        match_path = a.require_report_match.expanduser().resolve()
        other = _read_json(match_path)
        if not isinstance(other, Mapping):
            raise AnalysisError(f"--require-report-match {match_path}: not a JSON object")
        d = numeric_leaf_diff(other, report)
        report["report_match"] = {"against": str(match_path), "equal": d["equal"], "shared_numeric": d["shared_numeric"],
                                  "differing": [[T._path_text(p), va, vb] for p, va, vb in d["differing"]],
                                  "missing": [T._path_text(p) for p in d["missing"]], "added": [T._path_text(p) for p in d["added"]],
                                  "type_changed": [[T._path_text(p), va, vb] for p, va, vb in d["type_changed"]]}
        if not d["equal"]:
            write_report()
            print(f"[m5t] this run's numeric leaves DIFFER from {match_path} — {len(d['differing'])} differing, "
                  f"{len(d['type_changed'])} type changed, {len(d['missing'])} missing, {len(d['added'])} added over "
                  f"{d['shared_numeric']} shared; nothing published", file=sys.stderr)
            return EXIT_REPORTS_DIFFER
    figs: dict[str, Any] = {}
    if not a.no_figs:
        figs_dir = (a.figs_dir or ((out.parent if out else Path.cwd()) / "figs")).expanduser().resolve()
        figs = make_figures(report, figs_dir, exclude_families=a.figure_exclude_family, exclude_arms=a.figure_exclude_arm,
                            modes=figure_modes_arg, shown_only=a.figure_shown_only)
        report["figures"] = figs
        # plan P5: the PUBLICATION GATE — before the W&B / mirror pass, every figure's rows sidecar is read and publication
        # is refused if any figure presents a PSR arm id, or a cell / family / mode this render declared out
        try:
            report[PUBLICATION_GATE_KEY] = publication_gate_record(report, figs, figs_dir)
        except PG.PublicationRefused as e:
            report[PUBLICATION_GATE_KEY] = {"status": PG.GATE_REFUSED, "violations": list(e.violations), "rule": PG.RULE}
            write_report()
            print(f"[m5t] PUBLICATION REFUSED — {len(e.violations)} figure record(s) present what must not be published: "
                  + "; ".join(e.violations), file=sys.stderr)
            return PG.EXIT_PUBLICATION_REFUSED
        if not a.no_mirror:
            refuse_mixed_render(a.mirror_dir, figs.get(FIGURES_EXCLUDED_KEY) or [], list(figs["written"]),
                                arms=figs.get(FIGURES_EXCLUDED_ARMS_KEY) or [], modes=figs.get(FIGURES_MODES_KEY))
            report["mirrored"] = mirror_figures(figs_dir, a.mirror_dir.expanduser().resolve(), manifest=figs)
    url = log_wandb(report, figs, mode=a.wandb_mode, name=a.wandb_name, name_suffix=a.wandb_name_suffix)
    report["wandb_url"] = url
    write_report()
    if out:
        # the flat per-row export beside the report — read from the report just written, never an estimator
        written = write_paper_rows_beside(report, out)
        print(f"[m5t] paper rows -> {written['json']} ({written['n_rows']} rows) + {written['csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
