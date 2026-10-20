"""The Inspect rung ADAPTER of the GSM8K paired-200 objects: the standing truncation ladder read off a served
Inspect store's FILED OUTCOMES, deciding THIS instrument's exit itself.

WHAT IT READS. A tinycap store root (``<out_root>/cells/<key>/``) of the
standard paired GSM8K (``inspect/gsm8k_paired200``) — its manifest (the hashed
identity: ``harness_config_hash``, ``max_tokens_by_task``; the recorded proxy
upstream timeout; a rung store's ``cap_rung`` / ``item_subset`` blocks) and its
ALL-ATTEMPTED outcome record ``outcomes/inspect__gsm8k_paired200.json``: every
attempted item's ``stop_reason``, ``at_cap``, ``completion_tokens``, ``scored``,
``value``, the think-block facts. At-cap is the provider's
``stop_reason == "max_tokens"`` at the request's cap (``inspect_served.
STOP_REASON_AT_CAP``, the store's one definition) — NEVER a re-tokenization:
a cap-hit draw whose text re-tokenizes below the cap is at-cap here (plan
P1-2; the ``rows.jsonl`` scored subset is never the population).

WHAT IT BORROWS from :mod:`sisteer.capeval.tiers.cap_ladder` (Evals' module,
merged at 0f0e6503): the text helpers ``degeneracy`` / ``is_degenerate`` /
``rep_ratio`` / ``tail_rep_ratio`` (the loop rule that decides eligibility),
``transport_check`` + ``TransportVerdict`` (called DIRECTLY, plan R2b), the
exit vocabulary constants, the record schema (``RECORD_SCHEMA``, ``ROW_KEYS``,
``REDRAW_KEYS``, ``cell_record``) and the ``RungStatus`` dataclass, which is
CONSTRUCTED here — never through ``rung_status`` / ``measure_rung`` /
``classify_items`` / ``inspect_rung_from_rows`` / ``verdict_for`` (the peer's
truncated-share rule at 1 %, which on a mixed cell returns ``below_threshold``
with no transport verdict; D10). A test spies ``verdict_for`` on both transport
paths.

THE INSTRUMENT'S RULE (serialized into every record as its own):
``trigger iff n_at_cap / n > 0.01``; given a trigger the ELIGIBLE set is
``at_cap - non_terminating (the loop rule) - boundary (empty at k = 1, by the
rule's own text — recorded, never inferred)``; ONLY eligible items are redrawn,
at the one rung 16,384, the ceiling. The 1 % line is a REPORT field
(``share_at_cap_over_1pct``), never a gate — it coincides with the trigger since
the harmonisation of 2026-09-12 (the first publication used a 3 % trigger; see
:data:`GSM8K_TRIGGER_RULE_3PCT_RETIRED`). Exits, in the order they are
decided: ``clean`` (no at-cap), ``below_threshold``, ``ceiling`` (a store at
the ceiling: the residual is filed, there is no rung above),
``triggered_no_eligible`` (every at-cap draw is a loop: no rung is
manufactured), then ``transport_check(16384, hops)`` with the RECORDED hops —
the driver's own budget and the source manifest's copy of ``endpoint.json``'s
upstream timeout, never a default — ``escalate`` on a pass (the subset file is
returned), ``transport_refused`` on a fail (a terminal record, no rung).

TEXT for the loop rule comes from the retained ``.eval`` log through the
producer's own reader (``inspect_served.reusable_log`` + ``results_from_log``
at the store's cap and ids): an unscored draw has no ``rows.jsonl`` row, so the
log is the ONLY complete text source; an at-cap item with no text on record
REFUSES (a loop is never assumed absent). ``classify_store(..., texts=...)``
takes the mapping directly for callers that already hold it (tests).

WRITES NOTHING. ``ladder_plan.py`` (lane B) writes ``cap_ladder.json`` and
``rung_subset.json`` from what :func:`classify_store` returns — the one writer.

CONTRACT EXPORTS — the PRODUCER's spellings, which the planner and the reader
IMPORT and never re-type (a re-typed copy drifted once: a U+2212 minus for
this module's ASCII hyphen-minus, ``eligible`` for ``eligible_ids``;
integration review A-P1.1 / B-P1.2 / B-P1.3):

* :data:`GSM8K_TRIGGER_RULE` / :data:`GSM8K_TRIGGER_THRESHOLD` /
  :data:`GSM8K_REPORT_LINE` — the rule string (ASCII ``-`` throughout), the
  1 % trigger and the 1 % report line (:data:`GSM8K_TRIGGER_RULE_3PCT_RETIRED` /
  :data:`GSM8K_TRIGGER_THRESHOLD_3PCT_RETIRED` the retired first-publication pair,
  :data:`GSM8K_TRIGGER_RULES_ADMITTED` the rule -> threshold map a reader admits on a
  rung identity or subset file, :data:`GSM8K_TRIGGER_PROVENANCE` the sentence);
* :data:`GSM8K_TASK` = ``contracts.GSM8K_STANDARD_TASK`` — the ROW task
  (``inspect/gsm8k_paired200``) every record, subset file and comparison
  names (the unprefixed ``gsm8k_paired200`` is the CLI cap flags' and
  ``plan_task``'s spelling only);
* :data:`EXITS` and its two classes :data:`NOT_TRIGGERED_EXITS` /
  :data:`TRIGGERED_TERMINAL_EXITS`, plus :data:`EXIT_ESCALATE` and this
  instrument's own :data:`EXIT_TRIGGERED_NO_ELIGIBLE`;
* :data:`RECORD_INSTRUMENT_FIELDS` — the record's instrument fields, with the
  three id lists named :data:`RECORD_FIELD_ELIGIBLE_IDS` (``eligible_ids``),
  :data:`RECORD_FIELD_NON_TERMINATING_IDS`, :data:`RECORD_FIELD_BOUNDARY_IDS`;
* :data:`ITEM_KINDS` (``truncated`` = eligible, ``non_terminating`` = a loop;
  a below-cap draw's kind is ``None``) and :data:`ITEM_FACT_FIELDS`, the keys
  of one ``item_facts`` record;
* the subset file: ``contracts.ITEM_SUBSET_SCHEMA`` / ``ITEM_SUBSET_FIELDS`` /
  ``validate_item_subset``; its digest ``contracts.item_subset_sha16`` (sha256
  of the canonical JSON of the SORTED unique ids, 16 hex — the value
  ``cap_rung.subset_sha16`` carries; no other digest of the ids exists).

THE IFEVAL INSTRUMENT: the same adapter generalised to
k = 2 DRAWS. An :class:`Instrument` names the task, k, the trigger threshold, the ladder, the ceiling, the rule and
whether the instrument is DRAW-level; ``GSM8K_INSTRUMENT`` (k = 1, item-level, 3 %, the one 16,384 rung) keeps every
GSM8K record and subset BYTE-IDENTICAL (pinned by the GSM8K fixtures); ``IFEVAL_INSTRUMENT`` (k = 2, draw-level, the
RAW at-cap share of DRAWS strictly > 1 % — the user's ruling; ladder 2,048 -> 4,096 -> 8,192) keys every fact on
``(item_id, sample_idx)``, adds the k = 2 BOUNDARY reading (a sub-class of truncated: the sibling trial terminated below
the source cap — counted and RE-DRAWN, P11), the four IFEval values + ``n_inst`` per draw, and files a ``rung_subset/2``
DRAW subset on escalate (``ifeval_contracts.validate_item_subset_v2``). :func:`merge_draws` is the per-(item, trial)
twin of :func:`merge_items`. Texts of a k = 2 store come from BOTH sample logs through the producer's log selector
(``reusable_log``) and its row reader (``results_from_log`` with ``primary_value_key`` — IFEval's dict-valued Score is
read by its value key, P5), the same reader the k = 1 branch and the producer itself use.
"""
from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

from sisteer.capeval.tiers import cap_ladder as L
from sisteer.capeval.tinycap import completion
from sisteer.capeval.tinycap import contracts as C
from sisteer.capeval.tinycap import ifeval_contracts as IC

#: the one task this adapter reads (the standard paired GSM8K row task)
GSM8K_TASK = C.GSM8K_STANDARD_TASK
#: the producer's one definition of at-cap (``inspect_served.STOP_REASON_AT_CAP``,
#: pinned equal by test — this module does not import the Inspect tier at load)
STOP_REASON_AT_CAP = "max_tokens"
#: the instrument's trigger on the AT-CAP share: 1 %, the battery's standing rule, harmonised on
#: 2026-09-12 (3 % for the first publication; that pair is kept below as RETIRED). Every function
#: here takes the trigger as a PARAMETER defaulting to this one: a schema-2 matrix DECLARES its trigger on every row
#: (`trigger_threshold`, llama-paper-rows) and the ladder planner passes the row's.
GSM8K_TRIGGER_THRESHOLD = 0.01
#: the REPORT line (Evals' standing 1 %), filed per cell, never a gate — equal to the trigger since the harmonisation
GSM8K_REPORT_LINE = L.AT_CAP_SHARE_THRESHOLD
#: the one rung and the ceiling: no rung above
GSM8K_CEILING = L.CAP_CEILING
#: the rule string, serialized into every record and subset file as this
#: instrument's own: ASCII hyphen-minus (U+002D) throughout — never a Unicode
#: minus (U+2212); a reader compares it by IMPORTING this name
def gsm8k_trigger_rule(threshold: float) -> str:
    """The rule string for a trigger — ASCII hyphen-minus throughout; the threshold spelled as Python spells it
    (``0.01`` / ``0.03``): ONE template, so :data:`GSM8K_TRIGGER_RULE` and the retired pair's string differ in the
    threshold alone, and a rung planned under either is recognisable by its string."""
    return (f"trigger iff n_at_cap / n > {float(threshold):g} (at-cap = stop_reason == max_tokens at the request's cap); "
            "redraw the eligible set = at_cap - degenerate - boundary; ceiling 16,384; no rung above")


GSM8K_TRIGGER_RULE = gsm8k_trigger_rule(GSM8K_TRIGGER_THRESHOLD)
#: THE RETIRED 3 % PAIR (the first publication of this wave, 2026-09-10 to 2026-09-12): the rung stores and
#: subset files PLANNED under it are immutable evidence (a rung manifest hashes `cap_rung.rule` into the store
#: identity), so a reader ADMITS this rule string paired with this threshold on a rung identity / subset file —
#: never on a fresh record: the eligible set does not depend on the threshold and a 3 % trigger implies a 1 %
#: trigger, so every such rung is exactly the rung the harmonised rule plans. `cap_ladder.json` records are
#: re-filed by the planner under the current rule and are held to it alone.
GSM8K_TRIGGER_THRESHOLD_3PCT_RETIRED = 0.03
GSM8K_TRIGGER_RULE_3PCT_RETIRED = gsm8k_trigger_rule(GSM8K_TRIGGER_THRESHOLD_3PCT_RETIRED)
#: rule string -> the threshold it states, for every rule a rung identity or subset file may carry
GSM8K_TRIGGER_RULES_ADMITTED: dict[str, float] = {GSM8K_TRIGGER_RULE: GSM8K_TRIGGER_THRESHOLD,
                                                 GSM8K_TRIGGER_RULE_3PCT_RETIRED: GSM8K_TRIGGER_THRESHOLD_3PCT_RETIRED}
#: the provenance sentence every GSM8K report and record states beside the trigger
GSM8K_TRIGGER_PROVENANCE = ("trigger 1 % RAW at-cap, harmonised across the battery by user decision 2026-09-12; "
                            "the first publication of this wave used the 3 % exception of 2026-09-10")
AT_CAP_RULE_INSPECT = ("outcome.stop_reason == 'max_tokens' (inspect_served.STOP_REASON_AT_CAP): the provider's "
                       "finish_reason at the request's cap, read off the filed outcome record over ALL attempted "
                       "items; never re-tokenized")
BOUNDARY_RULE_K1 = ("boundary = another draw of the same item in hand terminated below the source cap; at k = 1 "
                    "no such draw exists, so the class is EMPTY by construction and recorded as such (plan C6); a "
                    "rung redraw's boundary_landing is the retrospective reading")
#: this instrument's own exit beside the borrowed vocabulary
EXIT_TRIGGERED_NO_ELIGIBLE = "triggered_no_eligible"
EXITS: tuple[str, ...] = (L.EXIT_CLEAN, L.EXIT_BELOW_THRESHOLD, L.EXIT_CEILING, EXIT_TRIGGERED_NO_ELIGIBLE,
                          L.EXIT_TRANSPORT_REFUSED, L.ESCALATE)
#: the exits that leave no rung to run (a terminal record)
TERMINAL_EXITS: tuple[str, ...] = (L.EXIT_CLEAN, L.EXIT_BELOW_THRESHOLD, L.EXIT_CEILING,
                                   EXIT_TRIGGERED_NO_ELIGIBLE, L.EXIT_TRANSPORT_REFUSED)
#: the reader's two classes of terminal exit: the trigger was FALSE (no at-cap
#: draw / the share at or under the trigger), and the trigger was TRUE but no rung runs
#: (every at-cap draw a loop / the transport cannot carry the rung); the one
#: exit that names a rung is :data:`EXIT_ESCALATE` (Evals' ``cap_ladder.ESCALATE``)
NOT_TRIGGERED_EXITS: tuple[str, ...] = (L.EXIT_CLEAN, L.EXIT_BELOW_THRESHOLD)
TRIGGERED_TERMINAL_EXITS: tuple[str, ...] = (EXIT_TRIGGERED_NO_ELIGIBLE, L.EXIT_TRANSPORT_REFUSED)
EXIT_ESCALATE = L.ESCALATE
INSTRUMENT = "gsm8k_paired200/cap_ladder"
#: the driver's own request budget — the ``driver`` hop (``proxy_engine.
#: AIOHTTP_TOTAL_TIMEOUT_S == contracts.PROXY_UPSTREAM_TIMEOUT_S``, 7,200 s)
DRIVER_HOP_TIMEOUT_S = float(C.PROXY_UPSTREAM_TIMEOUT_S)
#: the decimals every SHARE in a record is rounded to (``share_at_cap``, ``share_truncated``, ``share_loop``,
#: ``share_eligible``): a reader re-derives a share as ``round(count / population, SHARE_DECIMALS)`` and holds the
#: record to the INTEGER identity ``n_at_cap == n_truncated + n_loop`` — never to the rounded shares summing within
#: a tolerance (ifeval-canonical-battery C3: 4 / 1,082 rounds to 0.003697 while 2 / 1,082 + 2 / 1,082 rounds to 0.003696);
#: the ONE definition is the IFEval contracts' (the planner's decision block rounds with the same constant)
SHARE_DECIMALS = IC.SHARE_DECIMALS
#: the record's three id lists, by name: the ELIGIBLE set (the rung's subset),
#: the loops the eligibility rule excluded, the boundary class (empty at k = 1)
RECORD_FIELD_ELIGIBLE_IDS = "eligible_ids"
RECORD_FIELD_NON_TERMINATING_IDS = "non_terminating_ids"
RECORD_FIELD_BOUNDARY_IDS = "boundary_ids"
RECORD_ID_LIST_FIELDS: tuple[str, ...] = (
    RECORD_FIELD_ELIGIBLE_IDS, RECORD_FIELD_NON_TERMINATING_IDS, RECORD_FIELD_BOUNDARY_IDS)
#: the record's instrument fields beside Evals' ``cap_ladder/1`` fields
RECORD_INSTRUMENT_FIELDS: tuple[str, ...] = (
    "instrument", "rule", "report_line", "at_cap_rule", "boundary_rule", "exit", "escalate", "next_cap",
    "n", "n_at_cap", "share_at_cap", "share_at_cap_over_1pct", "n_non_terminating", "n_boundary",
    "n_eligible", "share_eligible", "share_eligible_over_1pct", *RECORD_ID_LIST_FIELDS,
    "periodic_block", "hops", "transport", "source", "item_facts")
#: the per-draw classes an at-cap item falls in (``kind``): ``truncated`` =
#: eligible for the rung, ``non_terminating`` = a loop (excluded); a below-cap
#: draw's ``kind`` is ``None``. Evals' vocabulary, by reference.
ITEM_KINDS: tuple[str, ...] = (L.KIND_TRUNCATED, L.KIND_NON_TERMINATING)
#: the keys of one ``item_facts`` record (:func:`classify_items`), in order
ITEM_FACT_FIELDS: tuple[str, ...] = (
    "item_id", "sample_idx", "stop_reason", "at_cap", "completion_tokens", "source_cap", "scored", "value",
    "think_block_unterminated", "degenerate", "rep_ratio", "tail_rep_ratio", "periodic_block", "kind",
    "text_sha256", "eligible")
#: the periodic-block loop rule: a NON-DECIDING column, the MATH
#: analysis's shape — the tail's longest suffix repeating with a period in
#: [8, 400] chars at least 3 times covering >= 60 % of the tail
LOOP_TAIL_CHARS = 2000
LOOP_PERIOD_MIN = 8
LOOP_PERIOD_MAX = 400
LOOP_MIN_REPEATS = 3
LOOP_MIN_COVER = 0.6
PERIODIC_BLOCK_RULE = (f"a non-deciding column: the last {LOOP_TAIL_CHARS} chars' longest suffix repeating with "
                       f"period p in [{LOOP_PERIOD_MIN}, {LOOP_PERIOD_MAX}] at least {LOOP_MIN_REPEATS} times and "
                       f"covering >= {LOOP_MIN_COVER:.0%} of the tail; eligibility is decided by the loop rule "
                       "(cap_ladder.DEGENERATE_RULE) alone")


class InspectRungError(RuntimeError):
    """A store this adapter cannot read into a verdict — refused by name."""


# --------------------------------------------------------------------------- #
# text helpers                                                                 #
# --------------------------------------------------------------------------- #


def emitted_text(content: str | None, reasoning: str | None) -> str | None:
    """The text a draw emitted as the loop rule reads it: the reasoning channel,
    then the content (the wire order under a server parser; under a tag split
    the content already carries the block). ``None`` when NEITHER channel is
    on record — text the store does not carry, distinct from two empty ones."""
    if content is None and reasoning is None:
        return None
    return (reasoning or "") + (content or "")


def _periodic_suffix(tail: str, p: int) -> int:
    """The length of the longest suffix of ``tail`` that repeats with period
    ``p`` (every character equal to the one ``p`` before it, walking back from
    the end); at least ``p`` when ``len(tail) >= p``."""
    n = len(tail)
    if n < p:
        return 0
    i = n - p - 1
    while i >= 0 and tail[i] == tail[i + p]:
        i -= 1
    return n - (i + 1)


def periodic_block(text: str | None) -> dict[str, Any]:
    """:data:`PERIODIC_BLOCK_RULE` as a pure function over one draw's emitted
    text: ``loop`` (the rule fired), ``period`` / ``repeats`` / ``cover`` of the
    block that fired (else of the maximal suffix repeating at least twice, as a
    diagnostic; ``None`` when none does). Ties in cover go to the SMALLEST
    period. Conservative: an enumeration whose blocks differ is not periodic."""
    tail = (text or "")[-LOOP_TAIL_CHARS:]
    n = len(tail)
    best: tuple[float, int, int] | None = None
    fired: tuple[float, int, int] | None = None
    if n >= 2 * LOOP_PERIOD_MIN:
        for p in range(LOOP_PERIOD_MIN, min(LOOP_PERIOD_MAX, n // 2) + 1):
            length = _periodic_suffix(tail, p)
            cover = length / n
            if length >= 2 * p and (best is None or cover > best[0]):
                best = (cover, p, length)
            if (length >= LOOP_MIN_REPEATS * p and cover >= LOOP_MIN_COVER
                    and (fired is None or cover > fired[0])):
                fired = (cover, p, length)
    chosen = fired if fired is not None else best
    return {"loop": fired is not None,
            "period": None if chosen is None else chosen[1],
            "repeats": None if chosen is None else chosen[2] // chosen[1],
            "cover": None if chosen is None else round(chosen[0], 4)}


# --------------------------------------------------------------------------- #
# the store                                                                    #
# --------------------------------------------------------------------------- #


#: ifeval-canonical-battery: the IFEval instrument's own names (the lane's contracts, by import)
IFEVAL_TASK = IC.IFEVAL_TASK
IFEVAL_TRIGGER_THRESHOLD = IC.IFEVAL_TRIGGER_THRESHOLD
IFEVAL_TRIGGER_RULE = IC.IFEVAL_TRIGGER_RULE
IFEVAL_CEILING = IC.IFEVAL_CAP_CEILING
IFEVAL_LADDER = tuple(IC.IFEVAL_CAP_LADDER)
INSTRUMENT_IFEVAL = "ifeval/cap_ladder"
BOUNDARY_RULE_K2 = ("boundary = an at-cap, non-loop draw whose SIBLING draw of the same item (the other trial) terminated "
                    "with completion_tokens < source_cap (filed facts only); at k = 2 a SUB-CLASS of truncated: counted "
                    "(n_boundary) and RE-DRAWN with the truncated draws (P11); the rung's boundary_landing is the "
                    "retrospective reading")
#: a DRAW-level fact record: the k = 1 fields + the boundary sub-flag, the instruction count and the four IFEval values
ITEM_FACT_FIELDS_DRAW: tuple[str, ...] = (*ITEM_FACT_FIELDS, "boundary", "n_inst", *IC.IFEVAL_METRIC_FIELDS)
#: the DRAW-level record's extra instrument fields (beside RECORD_INSTRUMENT_FIELDS)
RECORD_DRAW_FIELDS: tuple[str, ...] = (
    "samples_per_item", "n_draws", "n_truncated", "n_loop", "share_truncated", "share_loop",
    "eligible_draws", "loop_draws", "boundary_draws", "trigger_provenance")


class Instrument(NamedTuple):
    """ONE instrument's ladder rule: which task, k draws per item, the trigger threshold over the population (items at
    k = 1, DRAWS at k = 2), the ladder and its ceiling, the serialized rule / boundary rule, the subset schema."""
    name: str
    task: str
    k: int
    threshold: float
    ladder: tuple[int, ...]
    ceiling: int
    rule: str
    boundary_rule: str
    draw_level: bool
    subset_schema: str
    report_line: float
    trigger_provenance: str | None


GSM8K_INSTRUMENT = Instrument(INSTRUMENT, GSM8K_TASK, 1, GSM8K_TRIGGER_THRESHOLD, (), GSM8K_CEILING, GSM8K_TRIGGER_RULE,
                              BOUNDARY_RULE_K1, False, C.ITEM_SUBSET_SCHEMA, GSM8K_REPORT_LINE, None)
IFEVAL_INSTRUMENT = Instrument(INSTRUMENT_IFEVAL, IFEVAL_TASK, IC.IFEVAL_SAMPLES_PER_ITEM, IFEVAL_TRIGGER_THRESHOLD, IFEVAL_LADDER,
                               IFEVAL_CEILING, IFEVAL_TRIGGER_RULE, BOUNDARY_RULE_K2, True, IC.ITEM_SUBSET_SCHEMA_V2,
                               IFEVAL_TRIGGER_THRESHOLD, IC.IFEVAL_TRIGGER_PROVENANCE)
INSTRUMENTS_BY_TASK: Mapping[str, Instrument] = {GSM8K_TASK: GSM8K_INSTRUMENT, IFEVAL_TASK: IFEVAL_INSTRUMENT}


def instrument_for(task: str, *, k: int | None = None, ladder: Sequence[int] | None = None, ceiling: int | None = None,
                   threshold: float | None = None) -> Instrument:
    """The instrument of ``task`` (an unknown task refuses by name), with the explicit parameters — the plan's
    ``decide(..., ladder, ceiling, threshold)`` / ``classify_store(..., k, ...)`` — overriding its defaults."""
    inst = INSTRUMENTS_BY_TASK.get(task)
    if inst is None:
        raise InspectRungError(f"no ladder instrument for task {task!r} (instruments: {sorted(INSTRUMENTS_BY_TASK)})")
    if threshold is not None and not (0.0 < float(threshold) < 1.0):
        raise InspectRungError(f"trigger_threshold must be a share in (0, 1), got {threshold!r}")
    out = inst._replace(k=inst.k if k is None else int(k), ladder=inst.ladder if ladder is None else tuple(int(x) for x in ladder),
                        ceiling=inst.ceiling if ceiling is None else int(ceiling),
                        threshold=inst.threshold if threshold is None else float(threshold))
    if threshold is not None and task == GSM8K_TASK and float(threshold) != inst.threshold:
        # the rule STRING is one template of the threshold: the record says which trigger produced its ladder — a matrix
        # declaring the retired 3 % must read "> 0.03", never the harmonised default's text
        out = out._replace(rule=gsm8k_trigger_rule(float(threshold)))
    return out


@dataclass(frozen=True)
class StoreFacts:
    """What the adapter reads off one store: the facts every verdict rests on."""

    out_root: Path
    cell_dir: Path
    task: str
    identity_hash: str
    cap: int
    n: int
    items: tuple[Mapping[str, Any], ...]
    proxy_upstream_timeout_s: float | None
    cap_rung: Mapping[str, Any] | None       # the hashed rung binding (a RUNG store), else None
    item_subset: Mapping[str, Any] | None    # the unhashed subset block (a RUNG store), else None
    samples_per_item: int
    n_draws: int = 0                         # the attempted DRAWS (== n at k = 1)

    @property
    def is_rung(self) -> bool:
        return self.cap_rung is not None


def cell_dir_of(out_root: Path) -> Path:
    """The ONE cell dir under ``<out_root>/cells/``; zero or several refuse."""
    root = Path(out_root)
    cells_dir = root / "cells"
    cells = sorted(p for p in cells_dir.iterdir() if p.is_dir()) if cells_dir.is_dir() else []
    if len(cells) != 1:
        raise InspectRungError(f"{root}: {len(cells)} cell dir(s) under cells/, not exactly one")
    return cells[0]


def load_store(out_root: Path, *, task: str = GSM8K_TASK, samples_per_item_expected: int | None = None) -> StoreFacts:
    """Read a store into :class:`StoreFacts`, refusing by name anything the
    verdict cannot rest on: no manifest / outcome record; a manifest without
    the version-2 cap map (this instrument reads v2 stores); an outcome record
    whose ``max_tokens`` is not the identity's cap; an item whose ``at_cap``
    disagrees with its ``stop_reason`` (a torn record); a draw count per item
    other than the instrument's (k = 1 for GSM8K — its boundary class is empty
    BECAUSE no second draw exists; k = 2 for IFEval); a header count that is
    not the item list's."""
    root = Path(out_root)
    cell_dir = cell_dir_of(root)
    expected_k = int(samples_per_item_expected) if samples_per_item_expected is not None else instrument_for(task).k
    try:
        manifest = json.loads((cell_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise InspectRungError(f"{cell_dir / 'manifest.json'}: unreadable ({type(e).__name__}: {e})") from None
    identity_hash = manifest.get("harness_config_hash")
    if not isinstance(identity_hash, str) or not identity_hash:
        raise InspectRungError(f"{cell_dir}: the manifest carries no harness_config_hash")
    tinycap = (manifest.get("resolved_config") or {}).get(C.MANIFEST_KEY_TINYCAP) or {}
    caps = tinycap.get(C.IDENTITY_KEY_MAX_TOKENS_BY_TASK)
    if not isinstance(caps, Mapping) or task not in caps or caps[task] is None:
        raise InspectRungError(
            f"{cell_dir}: the identity carries no {C.IDENTITY_KEY_MAX_TOKENS_BY_TASK}[{task!r}] — this adapter "
            "reads identity-version-2 stores, whose executed cap is hashed; the request's cap cannot be guessed")
    cap = int(caps[task])
    try:
        outcomes = completion.read_task_outcomes(cell_dir, task)
    except completion.CompletionError as e:
        raise InspectRungError(f"{cell_dir}: {e}") from None
    if outcomes is None:
        raise InspectRungError(
            f"{cell_dir}: no {C.TASK_OUTCOMES_DIR}/{C.task_outcomes_filename(task)} — the all-attempted outcome "
            "record is the population; the scored rows never are")
    if outcomes.get("max_tokens") != cap:
        raise InspectRungError(
            f"{cell_dir}: the outcome record's max_tokens {outcomes.get('max_tokens')!r} is not the identity's "
            f"executed cap {cap} for {task} — a torn store")
    k = int(outcomes["samples_per_item"])
    if k != expected_k:
        if expected_k == 1:
            raise InspectRungError(
                f"{cell_dir}: samples_per_item {k} — this instrument is k = 1 (its boundary class is empty BECAUSE "
                "no second draw exists); refusing to classify a k > 1 store")
        raise InspectRungError(
            f"{cell_dir}: samples_per_item {k} — this instrument ({task}) is k = {expected_k}; refusing to classify a "
            f"k = {k} store")
    items = tuple(outcomes["items"])
    for it in items:
        at = it.get("stop_reason") == STOP_REASON_AT_CAP
        if bool(it.get("at_cap")) != bool(at):
            raise InspectRungError(
                f"{cell_dir}: item {it.get('item_id')!r} files at_cap={it.get('at_cap')!r} but stop_reason="
                f"{it.get('stop_reason')!r} — the two readings of one fact disagree (a torn record)")
    n = int(outcomes["n_attempted_items"])
    if n != len({it["item_id"] for it in items}):
        raise InspectRungError(f"{cell_dir}: n_attempted_items {n} != {len({it['item_id'] for it in items})} items")
    n_draws = int(outcomes["n_attempted_samples"])
    if n_draws != len(items):
        raise InspectRungError(f"{cell_dir}: n_attempted_samples {n_draws} != {len(items)} item records")
    up = manifest.get(C.ENDPOINT_KEY_PROXY_UPSTREAM_TIMEOUT_S)
    return StoreFacts(
        out_root=root, cell_dir=cell_dir, task=task, identity_hash=identity_hash, cap=cap, n=n, items=items,
        proxy_upstream_timeout_s=(None if up is None else float(up)),
        cap_rung=tinycap.get(C.IDENTITY_KEY_CAP_RUNG),
        item_subset=manifest.get(C.MANIFEST_KEY_ITEM_SUBSET),
        samples_per_item=k, n_draws=n_draws)


def _is_draw_level(facts: StoreFacts) -> bool:
    return facts.samples_per_item > 1 or instrument_for(facts.task).draw_level


def store_texts(facts: StoreFacts) -> dict[tuple[str, int], str | None]:
    """``{(item_id, sample_idx): emitted text}`` read from the store's retained
    ``.eval`` log(s) through the producer's OWN reader (``inspect_served.
    reusable_log`` at the store's cap over exactly its ids, then
    ``results_from_log``) — the one complete text source (an unscored draw has
    no row). The log's ``stop_reason`` per item must equal the outcome
    record's (a torn store refuses). Needs the client venv (``inspect_ai``).

    A DRAW-level store (k > 1, or the IFEval instrument) reads EVERY sample's
    log (``C.inspect_log_dir(cell_dir, task, idx)``) over the items filed at
    that ``sample_idx`` — a rung over a draw subset may have none for an idx,
    which is then skipped — through the same log selector and the same row
    reader, ``results_from_log`` with the task's ``primary_value_key`` (IFEval's
    dict-valued Score read by ``prompt_level_strict``, P5) and its seed rule."""
    from sisteer.capeval.tiers import inspect_served as isv

    if not _is_draw_level(facts):
        ids = sorted({str(it["item_id"]) for it in facts.items})
        d = C.inspect_log_dir(facts.cell_dir, facts.task, 0)
        log = isv.reusable_log(d, expected_n=len(ids), expected_ids=ids, expected_max_tokens=facts.cap)
        if log is None:
            raise InspectRungError(
                f"{facts.cell_dir}: no complete retained .eval under {d} at cap {facts.cap} over the store's "
                f"{len(ids)} ids — the emitted texts cannot be read, so degeneracy cannot be decided")
        name = isv._task_name(facts.task)
        results = isv.results_from_log(log, sample_idx=0, primary_scorer=isv.PRIMARY_SCORER.get(name),
                                       paired_ids={}, attempt=1, max_tokens_used=facts.cap)
        filed = {(str(it["item_id"]), int(it["sample_idx"])): it for it in facts.items}
        out: dict[tuple[str, int], str | None] = {}
        for r in results:
            key = (r.item_id, int(r.sample_idx))
            it = filed.get(key)
            if it is not None and it.get("stop_reason") != r.stop_reason:
                raise InspectRungError(
                    f"{facts.cell_dir}: item {r.item_id!r}: the .eval log says stop_reason {r.stop_reason!r} but the "
                    f"outcome record files {it.get('stop_reason')!r} — a torn store")
            out[key] = emitted_text(r.content, r.reasoning)
        return out
    filed = {(str(it["item_id"]), int(it["sample_idx"])): it for it in facts.items}
    out = {}
    for idx in range(facts.samples_per_item):
        ids = sorted({i for (i, s) in filed if s == idx})
        if not ids:
            continue
        d = C.inspect_log_dir(facts.cell_dir, facts.task, idx)
        log = isv.reusable_log(d, expected_n=len(ids), expected_ids=ids, expected_max_tokens=facts.cap)
        if log is None:
            raise InspectRungError(
                f"{facts.cell_dir}: no complete retained .eval under {d} at cap {facts.cap} over the store's "
                f"{len(ids)} ids of sample {idx} — the emitted texts cannot be read, so degeneracy cannot be decided")
        name = isv._task_name(facts.task)
        results = isv.results_from_log(log, sample_idx=idx, primary_scorer=isv.PRIMARY_SCORER.get(name),
                                       primary_value_key=isv.PRIMARY_VALUE_KEY.get(name), row_task=facts.task,
                                       paired_ids={}, attempt=1, max_tokens_used=facts.cap)
        for r in results:
            key = (r.item_id, int(r.sample_idx))
            it = filed.get(key)
            if it is not None and it.get("stop_reason") != r.stop_reason:
                raise InspectRungError(
                    f"{facts.cell_dir}: item {key[0]!r} sample {idx}: the .eval log says stop_reason {r.stop_reason!r} "
                    f"but the outcome record files {it.get('stop_reason')!r} — a torn store")
            out[key] = emitted_text(r.content, r.reasoning)
    return out


# --------------------------------------------------------------------------- #
# classification                                                               #
# --------------------------------------------------------------------------- #


def classify_items(facts: StoreFacts, texts: Mapping[tuple[str, int], str | None]) -> list[dict[str, Any]]:
    """One fact record per attempted draw: the filed facts (``stop_reason``,
    ``at_cap``, ``completion_tokens``, ``scored``, ``value``,
    ``think_block_unterminated``), the loop rule over the emitted text
    (``degenerate`` / ``rep_ratio`` / ``tail_rep_ratio`` — the DECIDING
    reading), the periodic-block column (non-deciding) and the ``kind``:
    ``truncated`` (at-cap, not degenerate — eligible), ``non_terminating``
    (at-cap, degenerate) or ``None`` (below cap). An at-cap draw with no text
    on record REFUSES. A DRAW-level store's records carry, beside those,
    ``boundary`` (k = 2: the sibling trial terminated below the source cap — a
    sub-flag of truncated), ``n_inst`` and the four IFEval values the outcome
    item files (``ITEM_FACT_FIELDS_DRAW``)."""
    draw_level = _is_draw_level(facts)
    out: list[dict[str, Any]] = []
    for it in facts.items:
        key = (str(it["item_id"]), int(it["sample_idx"]))
        at = bool(it["at_cap"])
        text = texts.get(key)
        if at and text is None:
            raise InspectRungError(
                f"{facts.cell_dir}: at-cap item {key[0]!r} has no emitted text on record — degeneracy cannot be "
                "read, and a loop is never assumed absent (plan §4.2)")
        deg = (L.degeneracy(text) if text is not None
               else {"rep_ratio": None, "tail_rep_ratio": None, "degenerate": None})
        kind = None if not at else (L.KIND_NON_TERMINATING if deg["degenerate"] else L.KIND_TRUNCATED)
        rec: dict[str, Any] = {
            "item_id": key[0], "sample_idx": key[1], "stop_reason": it.get("stop_reason"), "at_cap": at,
            "completion_tokens": it.get("completion_tokens"), "source_cap": facts.cap,
            "scored": bool(it.get("scored")), "value": it.get("value"),
            "think_block_unterminated": bool(it.get("think_block_unterminated")),
            "degenerate": deg["degenerate"], "rep_ratio": deg["rep_ratio"], "tail_rep_ratio": deg["tail_rep_ratio"],
            "periodic_block": periodic_block(text) if text is not None else None,
            "kind": kind, "text_sha256": (L.text_sha256(text) if text is not None else None),
            "eligible": kind == L.KIND_TRUNCATED}
        if draw_level:
            rec["boundary"] = False
            rec["n_inst"] = it.get("num_instructions")
            for f in IC.IFEVAL_METRIC_FIELDS:
                rec[f] = it.get(f)
        out.append(rec)
        assert tuple(out[-1]) == (ITEM_FACT_FIELDS_DRAW if draw_level else ITEM_FACT_FIELDS) and (kind is None or kind in ITEM_KINDS)
    if draw_level:
        # BOUNDARY_RULE_K2: the sibling draw of the same item terminated below the source cap (filed facts only)
        by_item: dict[str, list[dict[str, Any]]] = {}
        for r in out:
            by_item.setdefault(r["item_id"], []).append(r)
        for r in out:
            if r["kind"] != L.KIND_TRUNCATED:
                continue
            sibs = [x for x in by_item[r["item_id"]] if x["sample_idx"] != r["sample_idx"]]
            r["boundary"] = any(not x["at_cap"] and x["completion_tokens"] is not None
                                and int(x["completion_tokens"]) < int(facts.cap) for x in sibs)
    return out


def decide(*, n: int | None = None, n_at_cap: int, eligible_ids: Sequence[Any] | None = None, cap: int,
           hops: Mapping[str, float | None], n_draws: int | None = None, eligible_draws: Sequence[Any] | None = None,
           ladder: Sequence[int] | None = None, ceiling: int = GSM8K_CEILING,
           threshold: float = GSM8K_TRIGGER_THRESHOLD) -> tuple[str, int | None, L.TransportVerdict | None]:
    """THIS instrument's exit, in three steps and never through
    ``cap_ladder.verdict_for``: (1) the trigger ``n_at_cap / population > threshold``
    (the population = ``n`` items at k = 1, ``n_draws`` at k = 2; the default
    threshold is :data:`GSM8K_TRIGGER_THRESHOLD`, 0.01 since the 2026-09-12
    harmonisation — the first publication's 0.03 is
    :data:`GSM8K_TRIGGER_THRESHOLD_3PCT_RETIRED`) — false -> ``below_threshold``
    (``clean`` at zero at-cap); a store AT the ceiling that triggers -> ``ceiling``
    (the residual, no rung above); (2) the eligible set (``eligible_ids`` items or
    ``eligible_draws`` draws) — empty -> ``triggered_no_eligible``; (3) the next
    rung ``cap_ladder.next_rung(cap, ladder, ceiling)`` (the GSM8K default ladder
    ``(cap, ceiling)`` names the one 16,384 rung) and ``cap_ladder.transport_check
    (next_cap, hops)`` DIRECTLY with the RECORDED hops -> ``escalate`` or
    ``transport_refused``. Returns ``(exit, next_cap, transport)``; ``next_cap``
    is the rung on the two exits that name one."""
    population = n_draws if n_draws is not None else n
    eligible = eligible_draws if eligible_draws is not None else (eligible_ids or [])
    if population is None or population <= 0:
        raise InspectRungError("an empty population has no at-cap share")
    if n_at_cap == 0:
        return L.EXIT_CLEAN, None, None
    if n_at_cap / population <= threshold:
        return L.EXIT_BELOW_THRESHOLD, None, None
    if int(cap) >= int(ceiling):
        return L.EXIT_CEILING, None, None
    if not eligible:
        return EXIT_TRIGGERED_NO_ELIGIBLE, None, None
    next_cap = L.next_rung(int(cap), ladder if ladder else (int(cap), int(ceiling)), int(ceiling))
    if next_cap is None:
        return L.EXIT_CEILING, None, None
    tv = L.transport_check(next_cap, hops)
    return (L.ESCALATE if tv.ok else L.EXIT_TRANSPORT_REFUSED), next_cap, tv


def recorded_hops(facts: StoreFacts, *, driver_timeout_s: float = DRIVER_HOP_TIMEOUT_S
                  ) -> dict[str, float | None]:
    """The two hops :func:`decide` hands ``transport_check``: the driver's own
    request budget and the proxy's upstream timeout AS THE SOURCE MANIFEST
    RECORDED IT (the driver's copy of ``endpoint.json``'s value under either key
    name, ``contracts.ENDPOINT_KEY_PROXY_UPSTREAM_TIMEOUT_S``); a null upstream
    stays null and ``transport_check`` refuses it as undeterminable — never a
    default (the 600 s default is exactly what censored the slow leg)."""
    return {L.HOP_DRIVER: float(driver_timeout_s), L.HOP_PROXY_UPSTREAM: facts.proxy_upstream_timeout_s}


def _status(facts: StoreFacts, *, eligible: Sequence[str], nonterm: Sequence[str], n_at: int, exit_: str,
            next_cap: int | None, tv: L.TransportVerdict | None, redraws: Mapping[Any, Mapping[str, Any]],
            inst: Instrument = GSM8K_INSTRUMENT, population: int | None = None, n_truncated: int | None = None,
            n_boundary: int = 0, boundary_ids: Sequence[str] = ()) -> L.RungStatus:
    """The ``cap_ladder/1`` per-control status, CONSTRUCTED from the
    sets :func:`classify_store` decided on: the peer's ``n_truncated`` /
    ``share_truncated`` / ``truncated_ids`` are this instrument's ELIGIBLE set
    (named as such in the record), ``n_at_cap`` / ``share_at_cap`` the share
    that decided (over ``population``: items at k = 1, draws at k = 2)."""
    pop = int(population if population is not None else facts.n)
    n_trunc = int(n_truncated if n_truncated is not None else len(eligible))
    outcomes = Counter(str(rd["outcome"]) for rd in redraws.values())
    return L.RungStatus(
        cap=int(facts.cap), n=pop, n_at_cap=n_at, n_truncated=n_trunc,
        n_non_terminating=len(nonterm), n_boundary=int(n_boundary),
        share_at_cap=round(n_at / pop, SHARE_DECIMALS), share_truncated=round(n_trunc / pop, SHARE_DECIMALS),
        threshold=inst.threshold, ceiling=inst.ceiling,
        escalate=(exit_ == L.ESCALATE), next_cap=next_cap, at_ceiling=int(facts.cap) >= inst.ceiling,
        exit=exit_, climbed=facts.is_rung, ladder_available=True,
        truncated_ids=list(eligible), non_terminating_ids=list(nonterm), boundary_ids=list(boundary_ids),
        source_caps_seen={str(int(facts.cap)): pop},
        n_regenerated=len(redraws), redraw_outcomes=dict(sorted(outcomes.items())),
        n_fork=sum(1 for rd in redraws.values() if rd["prefix_extension"] is False),
        n_identical_text=sum(1 for rd in redraws.values() if rd["identical_text"] is True),
        n_prefix_extension=sum(1 for rd in redraws.values() if rd["prefix_extension"] is True),
        transport=(tv.as_dict() if tv is not None else None),
        rules={"at_cap": AT_CAP_RULE_INSPECT, "degenerate": L.DEGENERATE_RULE,
               "trigger": inst.rule, "transport": L.TRANSPORT_RULE})


def _rows(item_facts: Sequence[Mapping[str, Any]], redraws: Mapping[tuple[str, int], Mapping[str, Any]],
          *, regenerated: bool) -> list[dict[str, Any]]:
    """Evals' ``ROW_KEYS`` rows: ``ntok`` and ``completion_tokens`` are BOTH the
    filed ``completion_tokens`` (never re-tokenized), ``at_cap`` the filed
    fact, the loop reading, the kind, the k = 2 boundary sub-flag as
    ``boundary_from`` (the source cap; None at k = 1), the redraw block on a rung."""
    out: list[dict[str, Any]] = []
    for r in item_facts:
        ct = r["completion_tokens"]
        key = (str(r["item_id"]), int(r["sample_idx"]))
        rec = L.ItemRecord(item_id=r["item_id"], text_sha256=r["text_sha256"], source_cap=int(r["source_cap"]),
                           ntok=ct, completion_tokens=ct, at_cap=bool(r["at_cap"]),
                           degenerate=r["degenerate"], rep_ratio=r["rep_ratio"], tail_rep_ratio=r["tail_rep_ratio"],
                           kind=r["kind"], boundary_from=(int(r["source_cap"]) if r.get("boundary") else None),
                           regenerated=regenerated,
                           redraw=(dict(redraws[key]) if key in redraws else None))
        out.append(rec.as_row())
    return out


# --------------------------------------------------------------------------- #
# the redraw record (a RUNG store against its source)                          #
# --------------------------------------------------------------------------- #


_first_divergence = L._first_divergence


def redraw_record(source: Mapping[str, Any], rung: Mapping[str, Any], *, source_text: str | None,
                  rung_text: str | None) -> dict[str, Any]:
    """Evals' ``REDRAW_KEYS`` for ONE re-drawn item, every termination fact from
    the FILED outcomes: ``still_at_cap`` = the rung draw's
    ``stop_reason == max_tokens``; ``boundary_landing`` = terminated with fewer
    completion tokens than the SOURCE cap (the source draw was boundary noise);
    ``extended`` = terminated at or past the source cap (real truncation,
    resolved). The text relationship (identical / prefix / first divergence
    char) is recorded from the two emitted texts when both are on record and
    null otherwise; ``first_divergence_token`` is always null — nothing here
    re-tokenizes."""
    src_cap, new_cap = int(source["source_cap"]), int(rung["source_cap"])
    new_ct = rung["completion_tokens"]
    still = bool(rung["at_cap"])
    landing = bool(not still and new_ct is not None and int(new_ct) < src_cap)
    extended = bool(not still and not landing)
    if source_text is not None and rung_text is not None:
        identical: bool | None = rung_text == source_text
        prefix: bool | None = rung_text.startswith(source_text)
        char: int | None = None if prefix else _first_divergence(source_text, rung_text)
    else:
        identical = prefix = None
        char = None
    out = {"source_cap": src_cap, "new_cap": new_cap,
           "source_completion_tokens": source["completion_tokens"], "new_completion_tokens": new_ct,
           "identical_text": identical, "prefix_extension": prefix,
           "first_divergence_char": char, "first_divergence_token": None,
           "boundary_landing": landing, "still_at_cap": still, "extended": extended,
           "outcome": (L.REDRAW_STILL_AT_CAP if still else L.REDRAW_BOUNDARY_LANDING if landing
                       else L.REDRAW_EXTENDED)}
    assert tuple(out) == L.REDRAW_KEYS
    return out


def _subset_draws_of(rung: StoreFacts) -> list[tuple[str, int]]:
    """The rung's recorded population as DRAWS: the v2 block's ``draws``; a v1 block's ids at sample 0."""
    block = rung.item_subset or {}
    if block.get("schema") == IC.ITEM_SUBSET_SCHEMA_V2 or "draws" in block:
        return sorted((str(i), int(s)) for i, s in (block.get("draws") or []))
    return sorted((str(i), 0) for i in (block.get("ids") or []))


def redraw_records_by_draw(rung: StoreFacts, source: StoreFacts, *, rung_texts: Mapping[tuple[str, int], str | None],
                           source_texts: Mapping[tuple[str, int], str | None] | None = None
                           ) -> dict[tuple[str, int], dict[str, Any]]:
    """Every rung DRAW's redraw record against the source store's draw of the
    same (item, sample). Refuses a rung draw the source never attempted, one
    the source did NOT truncate (the subset is at-cap by construction), a rung
    whose draws are not exactly its recorded subset (subset-sized completeness),
    or a rung bound to another source (identity hash / source cap)."""
    if rung.cap_rung is None or rung.item_subset is None:
        raise InspectRungError(f"{rung.cell_dir}: not a rung store (no {C.IDENTITY_KEY_CAP_RUNG} identity)")
    if rung.cap_rung.get("source_identity_hash") != source.identity_hash:
        raise InspectRungError(
            f"{rung.cell_dir}: bound to source identity {rung.cap_rung.get('source_identity_hash')!r}, but "
            f"{source.cell_dir} carries {source.identity_hash!r}")
    if int(rung.cap_rung.get("source_cap", -1)) != source.cap:
        raise InspectRungError(f"{rung.cell_dir}: bound to source cap {rung.cap_rung.get('source_cap')!r}, but the "
                               f"source ran at {source.cap}")
    subset_draws = _subset_draws_of(rung)
    rung_draws = sorted((str(it["item_id"]), int(it["sample_idx"])) for it in rung.items)
    if rung_draws != subset_draws:
        what = "draws" if _is_draw_level(rung) else "ids"
        raise InspectRungError(
            f"{rung.cell_dir}: attempted {what} ({len(rung_draws)}) are not the recorded subset ({len(subset_draws)}) — "
            "a rung's population is its subset, whole")
    src_by = {(str(it["item_id"]), int(it["sample_idx"])): it for it in source.items}
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for it in rung.items:
        key = (str(it["item_id"]), int(it["sample_idx"]))
        s = src_by.get(key)
        if s is None:
            raise InspectRungError(f"{rung.cell_dir}: rung item {key[0]!r} was never attempted by the source store")
        if not s.get("at_cap"):
            raise InspectRungError(f"{rung.cell_dir}: rung item {key[0]!r} was NOT at cap in the source store — the "
                                   "subset is at-cap by construction")
        out[key] = redraw_record(
            {**s, "source_cap": source.cap}, {**it, "source_cap": rung.cap},
            source_text=(source_texts or {}).get(key), rung_text=rung_texts.get(key))
    return out


def redraw_records(rung: StoreFacts, source: StoreFacts, *, rung_texts: Mapping[tuple[str, int], str | None],
                   source_texts: Mapping[tuple[str, int], str | None] | None = None
                   ) -> dict[str, dict[str, Any]]:
    """The k = 1 reading of :func:`redraw_records_by_draw`: keyed by ITEM id (the GSM8K record's shape). A k > 1 store
    refuses — its records are per draw."""
    if rung.samples_per_item != 1:
        raise InspectRungError(f"{rung.cell_dir}: samples_per_item {rung.samples_per_item} — redraw records of a k > 1 store "
                               "are per (item, sample_idx): call redraw_records_by_draw")
    return {key[0]: rec for key, rec in redraw_records_by_draw(rung, source, rung_texts=rung_texts, source_texts=source_texts).items()}


# --------------------------------------------------------------------------- #
# the adapter's one entry point                                                #
# --------------------------------------------------------------------------- #


def classify_store(out_root: Path, *, task: str = GSM8K_TASK, k: int | None = None,
                   ladder: Sequence[int] | None = None, ceiling: int | None = None, threshold: float | None = None,
                   texts: Mapping[tuple[str, int], str | None] | None = None,
                   source_texts: Mapping[tuple[str, int], str | None] | None = None,
                   driver_timeout_s: float = DRIVER_HOP_TIMEOUT_S,
                   source_out_root: Path | None = None,
                   ) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """``(record, subset | None)`` for one store (the interface
    ``ladder_plan.py`` consumes): the ``cap_ladder/1`` record with
    this instrument's fields (:data:`RECORD_INSTRUMENT_FIELDS`; a DRAW-level
    instrument adds :data:`RECORD_DRAW_FIELDS`), and — on ``escalate`` only —
    the subset file body (``rung_subset/1`` ids for GSM8K, validated by
    ``contracts.validate_item_subset``; ``rung_subset/2`` DRAWS for IFEval,
    validated by ``ifeval_contracts.validate_item_subset_v2``) naming the
    eligible population, the source binding and the transport verdict.
    ``task`` picks the instrument (:func:`instrument_for`); ``k`` / ``ladder``
    / ``ceiling`` / ``threshold`` override its defaults. ``texts`` (default: the
    retained ``.eval`` through :func:`store_texts`) and, for a RUNG store,
    ``source_texts`` feed the loop rule and the redraw text relationship.
    ``source_out_root`` (a RUNG store only) names WHERE the source store is
    read from: default the subset's recorded ``item_subset.source_out_root``,
    an ABSOLUTE path stamped where the planner ran — so a wave pulled to
    another root (or a subset written on the volume) cannot be re-planned in
    place without it; the planner resolves the override through its
    wave-relative layout (``ifeval_paths``). The binding is still the
    IDENTITY's: :func:`redraw_records_by_draw` refuses a source whose
    ``harness_config_hash`` / cap are not the rung's recorded ones, so an
    override naming the wrong store refuses by name rather than silently
    re-pairing draws. Given on a PASS-1 store it refuses (nothing to bind).
    This function WRITES nothing."""
    inst = instrument_for(task, k=k, ladder=ladder, ceiling=ceiling, threshold=threshold)
    facts = load_store(out_root, task=task, samples_per_item_expected=inst.k)
    tx = dict(texts) if texts is not None else store_texts(facts)
    item_facts = classify_items(facts, tx)
    redraws: dict[tuple[str, int], dict[str, Any]] = {}
    source_block: dict[str, Any] = {"out_root": str(facts.out_root), "identity_hash": facts.identity_hash,
                                    "task": task, "is_rung": facts.is_rung}
    if source_out_root is not None and not facts.is_rung:
        raise InspectRungError(f"{facts.cell_dir}: source_out_root {str(source_out_root)!r} given for a store that is not a "
                               f"rung (no {C.IDENTITY_KEY_CAP_RUNG} identity) — a pass-1 store binds to no source")
    if facts.is_rung:
        assert facts.item_subset is not None
        src_root = Path(source_out_root) if source_out_root is not None else Path(str(facts.item_subset["source_out_root"]))
        source = load_store(src_root, task=task, samples_per_item_expected=inst.k)
        stx = dict(source_texts) if source_texts is not None else store_texts(source)
        redraws = redraw_records_by_draw(facts, source, rung_texts=tx, source_texts=stx)
        source_block.update(source_out_root=str(source.out_root), source_identity_hash=source.identity_hash,
                            source_cap=source.cap, n_redrawn=len(redraws))
    draw_level = _is_draw_level(facts)
    trunc_draws = sorted((r["item_id"], r["sample_idx"]) for r in item_facts if r["kind"] == L.KIND_TRUNCATED)
    loop_draws = sorted((r["item_id"], r["sample_idx"]) for r in item_facts if r["kind"] == L.KIND_NON_TERMINATING)
    boundary_draws = sorted((r["item_id"], r["sample_idx"]) for r in item_facts if r.get("boundary"))
    eligible = sorted({i for i, _ in trunc_draws})
    nonterm = sorted({i for i, _ in loop_draws})
    n_at = sum(1 for r in item_facts if r["at_cap"])
    hops = recorded_hops(facts, driver_timeout_s=driver_timeout_s)
    population = facts.n_draws if draw_level else facts.n
    if draw_level:
        exit_, next_cap, tv = decide(n_draws=population, n_at_cap=n_at, eligible_draws=trunc_draws, cap=facts.cap, hops=hops,
                                     ladder=inst.ladder, ceiling=inst.ceiling, threshold=inst.threshold)
    else:
        exit_, next_cap, tv = decide(n=population, n_at_cap=n_at, eligible_ids=eligible, cap=facts.cap, hops=hops,
                                     ladder=inst.ladder, ceiling=inst.ceiling, threshold=inst.threshold)
    status = _status(facts, eligible=eligible, nonterm=nonterm, n_at=n_at, exit_=exit_, next_cap=next_cap, tv=tv,
                     redraws=redraws, inst=inst, population=population,
                     n_truncated=(len(trunc_draws) if draw_level else None), n_boundary=len(boundary_draws),
                     boundary_ids=sorted({i for i, _ in boundary_draws}))
    loops = sorted(r["item_id"] for r in item_facts if r["periodic_block"] and r["periodic_block"]["loop"])
    n_elig = len(trunc_draws) if draw_level else len(eligible)
    extra: dict[str, Any] = {
        "rules": {"at_cap": AT_CAP_RULE_INSPECT, "degenerate": L.DEGENERATE_RULE,
                  "trigger": inst.rule, "transport": L.TRANSPORT_RULE, "boundary": inst.boundary_rule,
                  "periodic_block": PERIODIC_BLOCK_RULE},
        "instrument": inst.name, "rule": inst.rule, "report_line": inst.report_line,
        "at_cap_rule": AT_CAP_RULE_INSPECT, "boundary_rule": inst.boundary_rule,
        "exit": exit_, "escalate": exit_ == L.ESCALATE, "next_cap": next_cap,
        "n": facts.n, "n_at_cap": n_at, "share_at_cap": round(n_at / population, SHARE_DECIMALS),
        "share_at_cap_over_1pct": (n_at / population) > inst.report_line,
        "n_non_terminating": len(loop_draws) if draw_level else len(nonterm), "n_boundary": len(boundary_draws),
        "n_eligible": n_elig, "share_eligible": round(n_elig / population, SHARE_DECIMALS),
        "share_eligible_over_1pct": (n_elig / population) > inst.report_line,
        RECORD_FIELD_ELIGIBLE_IDS: eligible, RECORD_FIELD_NON_TERMINATING_IDS: nonterm,
        RECORD_FIELD_BOUNDARY_IDS: sorted({i for i, _ in boundary_draws}),
        "periodic_block": {"rule": PERIODIC_BLOCK_RULE, "n_loop": len(loops), "ids": loops, "deciding": False},
        "hops": dict(hops), "transport": (tv.as_dict() if tv is not None else None),
        "source": source_block, "item_facts": item_facts,
        "items": _rows(item_facts, redraws, regenerated=facts.is_rung),
    }
    if draw_level:
        extra.update({
            "samples_per_item": facts.samples_per_item, "n_draws": facts.n_draws,
            "n_truncated": len(trunc_draws), "n_loop": len(loop_draws),
            "share_truncated": round(len(trunc_draws) / population, SHARE_DECIMALS), "share_loop": round(len(loop_draws) / population, SHARE_DECIMALS),
            "eligible_draws": [[i, s] for i, s in trunc_draws], "loop_draws": [[i, s] for i, s in loop_draws],
            "boundary_draws": [[i, s] for i, s in boundary_draws], "trigger_provenance": inst.trigger_provenance,
        })
        ladder_list = [r for r in inst.ladder if r >= facts.cap] or [facts.cap]
    else:
        ladder_list = [facts.cap, inst.ceiling] if facts.cap < inst.ceiling else [facts.cap]
    record = L.cell_record(cell_id=facts.cell_dir.name, engine="vllm", cap=facts.cap, statuses={task: status},
                           store_dir=facts.cell_dir, threshold=inst.threshold, ceiling=inst.ceiling,
                           ladder=ladder_list, extra=extra)
    assert record["schema"] == L.RECORD_SCHEMA and record["exit"] in EXITS
    assert all(f in record for f in RECORD_INSTRUMENT_FIELDS)
    assert not draw_level or all(f in record for f in RECORD_DRAW_FIELDS)
    subset: dict[str, Any] | None = None
    if exit_ == L.ESCALATE:
        assert tv is not None and next_cap is not None
        if draw_level:
            subset = {"schema": IC.ITEM_SUBSET_SCHEMA_V2, "task": task, "source_out_root": str(facts.out_root),
                      "source_identity_hash": facts.identity_hash, "source_cap": int(facts.cap), "cap": int(next_cap),
                      "rule": inst.rule, "threshold": inst.threshold,
                      "ids": eligible, "draws": [[i, s] for i, s in trunc_draws],
                      "n_items": len(eligible), "n_draws": len(trunc_draws),
                      "n_at_cap": n_at, "n_truncated": len(trunc_draws), "n_loop": len(loop_draws),
                      "n_boundary": len(boundary_draws), "transport_check": tv.as_dict()}
            IC.validate_item_subset_v2(subset, where=f"{facts.cell_dir} subset")
        else:
            subset = {"schema": C.ITEM_SUBSET_SCHEMA, "task": task, "source_out_root": str(facts.out_root),
                      "source_identity_hash": facts.identity_hash, "source_cap": int(facts.cap), "cap": int(next_cap),
                      "rule": inst.rule, "threshold": inst.threshold, "ids": list(eligible),
                      "n_at_cap": n_at, "n_non_terminating": len(nonterm), "n_boundary": 0,
                      "transport_check": tv.as_dict()}
            C.validate_item_subset(subset, where=f"{facts.cell_dir} subset")
    return record, subset


# --------------------------------------------------------------------------- #
# the merge (analysis-side — a pure helper both lanes can import)             #
# --------------------------------------------------------------------------- #


def merge_items(source_items: Sequence[Mapping[str, Any]], rung_items: Sequence[Mapping[str, Any]],
                subset_ids: Sequence[str]) -> list[dict[str, Any]]:
    """The ladder-merged reading's per-item draws: the RUNG draw for every id
    in ``subset_ids``, the SOURCE draw for every other, in the source's item
    order, each annotated ``draw: "rung" | "source"``. Item ids are preserved
    exactly: a subset id the rung did not draw, a rung id outside the subset,
    or a merged id set that is not the source's refuses."""
    subset = {str(i) for i in subset_ids}
    rung_by = {str(it["item_id"]): it for it in rung_items}
    if set(rung_by) != subset:
        raise InspectRungError(
            f"the rung drew {sorted(set(rung_by) - subset)[:3]} outside its subset / missed "
            f"{sorted(subset - set(rung_by))[:3]} of it — the rung's population is its subset, whole")
    out: list[dict[str, Any]] = []
    seen = set()
    for it in source_items:
        iid = str(it["item_id"])
        if iid in seen:
            raise InspectRungError(f"the source lists item {iid!r} twice")
        seen.add(iid)
        if iid in subset:
            out.append({**rung_by[iid], "draw": "rung"})
        else:
            out.append({**it, "draw": "source"})
    if subset - seen:
        raise InspectRungError(f"subset ids {sorted(subset - seen)[:3]} are not source items")
    assert [o["item_id"] for o in out] == [it["item_id"] for it in source_items]
    return out


def merge_draws(source_items: Sequence[Mapping[str, Any]], rung_items: Sequence[Mapping[str, Any]],
                draws: Sequence[Sequence[Any]]) -> list[dict[str, Any]]:
    """The per-(item, trial) twin of :func:`merge_items`: the
    RUNG draw for every ``(item_id, sample_idx)`` in ``draws``, the SOURCE draw
    for every other, in the source's draw order, each annotated ``draw: "rung" |
    "source"``. The draw set is preserved exactly: a listed draw the rung did
    not draw, a rung draw outside the list, a source that lists a draw twice,
    or a listed draw the source never attempted refuses."""
    subset = {(str(i), int(s)) for i, s in draws}
    rung_by = {(str(it["item_id"]), int(it["sample_idx"])): it for it in rung_items}
    if set(rung_by) != subset:
        raise InspectRungError(
            f"the rung drew {sorted(set(rung_by) - subset)[:3]} outside its draw subset / missed "
            f"{sorted(subset - set(rung_by))[:3]} of it — the rung's population is its draws, whole")
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for it in source_items:
        key = (str(it["item_id"]), int(it["sample_idx"]))
        if key in seen:
            raise InspectRungError(f"the source lists draw {key!r} twice")
        seen.add(key)
        out.append({**rung_by[key], "draw": "rung"} if key in subset else {**it, "draw": "source"})
    if subset - seen:
        raise InspectRungError(f"subset draws {sorted(subset - seen)[:3]} are not source draws")
    assert [(o["item_id"], o["sample_idx"]) for o in out] == [(it["item_id"], it["sample_idx"]) for it in source_items]
    return out


# --------------------------------------------------------------------------- #
# the streaming ladder's GSM8K entry
# --------------------------------------------------------------------------- #


def _seedrep_rung_redraw():
    """``scripts/sdfbase/seedrep/rung_redraw.py`` by path, loaded ONLY when ``--from-materialised``
    is given — the streaming ladder's assembly seam; nothing here changes without the flag."""
    import importlib.util as _ilu
    import sys as _sys
    mod = _sys.modules.get("seedrep_rung_redraw")
    if mod is None:
        spec = _ilu.spec_from_file_location("seedrep_rung_redraw", Path(__file__).resolve().parents[4] / "scripts" / "sdfbase" / "seedrep" / "rung_redraw.py")
        mod = _sys.modules["seedrep_rung_redraw"] = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod


def subset_from_materialised_queue(ladder_root, results, *, source_out_root, arm: str, seed_base: int, task: str = GSM8K_TASK,
                                   **classify_kw) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
    """The seedrep streaming ladder's GSM8K assembly (default-off: nothing else in this module changes):
    ``(record, subset | None, verdict)`` — THIS adapter's :func:`classify_store` decides the eligible set exactly as
    today, and the CLOSED queue under ``ladder_root`` (with the results snapshot ``results``), read at EXACTLY the
    source store's slot ``<arm>__s<seed_base>`` (the store's cell id and its ``--inspect-seed-base``), is
    cross-checked against it: a queued item the accepted source never filed at cap REFUSES, an eligible item the
    queue lacks and a queued item the loop rule excludes are NOTED in the verdict. The subset returned is this
    adapter's own, for the driver's ``--item-subset``. Delegates to ``scripts/sdfbase/seedrep/rung_redraw.py``."""
    return _seedrep_rung_redraw().gsm8k_subset_from_queue(ladder_root, results, source_out_root=source_out_root, arm=arm,
                                                          seed_base=seed_base, task=task, classify_store=classify_store, **classify_kw)
