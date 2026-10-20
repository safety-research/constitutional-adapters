"""The IFEval canonical battery — LANE 2's contract vocabulary.

WHY THIS MODULE EXISTS. Lane 1 owns ``contracts.py`` and defines the IFEval names there (``SEED_RULE_IFEVAL``,
``vendor_preset``, ``ROUTE_KINDS``, ``BANK_VIEWS``, the ``rung_subset/2`` fields, ...). Lane 2 (this module's lane:
the matrix, the ladder, the analysis) referenced those names BEFORE lane 1 landed, so every constant here is a
LOCAL fallback consulted only when ``contracts`` carries no attribute of that name:

    NAME = _merged("NAME", <local value>)        # lane 1's definition wins whenever it exists

``MERGE_AGREEMENT`` lists every such name; ``tests/test_ifeval_contracts.py`` asserts, for every name lane 1's
``contracts`` DOES define, that the two definitions agree — so a divergence is a red test, never silent. MERGED
(the two lanes joined on one branch): 43 of the 58 names resolve to the contracts' definitions; the 15 that do not
(the draw-subset v2 contract, the trigger constants, ``ifeval_paths``, ``ifeval_expected_stamps``,
``IFEVAL_CONTRASTS``, ``request_evidence_record_v3``) are this lane's own contract and live here. The two fallbacks
that disagreed at the merge are now PURE delegates to the contracts' value (``SEED_RULE_IFEVAL`` — the plan's
``{20260909}`` spelling; ``REQUEST_EVIDENCE_FIELDS_V3`` — the contracts' field list already carries the ten v3
fields). Functions (``ifeval_sample_seed``, ``ifeval_paths``, the v2 subset validator) follow the same rule through
``_merged_callable``.

THE ONE PATH CONSTRUCTOR (P37): :func:`ifeval_paths` is keyed by ``(model_key, view, cell, rung)`` and is the ONLY
place a store, subset, ladder-record, ladder-state, budget or dispatch path is spelled. A SLUG (the launcher's
``ModelRow`` key, ``IFEVAL_SLUGS``) is NEVER a path component: nine gemma-main cells would otherwise share one
``out_dir`` (the launcher writes ``served_geometry.json``, the prompt budget and the usage ledger directly under it).
``tests/test_ifeval_paths.py`` asserts matrix-wide uniqueness of every path kind over 42 rows x 3 rungs.

THE DRAW-SUBSET CONTRACT (P30 / P10): ``rung_subset/2`` lists DRAWS ``[[item_id, sample_idx], ...]`` beside the
distinct ``ids``; its digest is over the sorted draws, so two rungs over the same items but different draws never
share a ``cap_rung`` identity; ``validate_item_subset_v2`` enforces ``len(draws) == n_at_cap - n_loop == n_draws``.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

from sisteer.capeval.serving import proxy as _proxy
from sisteer.capeval.tinycap import contracts as C

# --------------------------------------------------------------------------- the merge seam
#: every name this module defines LOCALLY as a fallback for a lane-1 contract of the same name; the agreement test
#: reads it. A name absent from ``contracts`` at import time resolves to the local value; a name present resolves
#: to lane 1's — and the test then asserts the two are equal.
MERGE_AGREEMENT: Dict[str, object] = {}


def _merged(name: str, local: object) -> object:
    """``contracts.<name>`` when lane 1 defines it, else ``local`` — recorded either way for the agreement test.
    Every entry of MERGE_AGREEMENT whose contracts twin exists must agree (tests/test_ifeval_contracts.py)."""
    MERGE_AGREEMENT[name] = local
    return getattr(C, name, local)


def _merged_callable(name: str, local: Callable[..., Any]) -> Callable[..., Any]:
    MERGE_AGREEMENT[name] = local
    return getattr(C, name, local)


# --------------------------------------------------------------------------- §1.1 the instrument's names
CONTROL_IFEVAL: str = _merged("CONTROL_IFEVAL", "ifeval")
IFEVAL_TASK_NAME: str = _merged("IFEVAL_TASK_NAME", "ifeval")
#: the ROW task — ``contracts.inspect_task("ifeval")`` (the contracts register the name in INSPECT_TASK_NAMES)
IFEVAL_TASK: str = _merged("IFEVAL_TASK", C.inspect_task(IFEVAL_TASK_NAME))
IFEVAL_N: int = _merged("IFEVAL_N", 541)
IFEVAL_DATASET_PATH: str = _merged("IFEVAL_DATASET_PATH", "google/IFEval")
IFEVAL_DATASET_REVISION: str = _merged("IFEVAL_DATASET_REVISION", "966cd89545d6b6acfd7638bc708b98261ca58e84")
IFEVAL_TASK_VERSION: str = _merged("IFEVAL_TASK_VERSION", "2-A")
IFEVAL_SAMPLES_PER_ITEM: int = _merged("IFEVAL_SAMPLES_PER_ITEM", 2)
IFEVAL_MAX_TOKENS_OFF: int = _merged("IFEVAL_MAX_TOKENS_OFF", 2048)
IFEVAL_CAP_LADDER: Tuple[int, ...] = _merged("IFEVAL_CAP_LADDER", (2048, 4096, 8192))
IFEVAL_CAP_CEILING: int = _merged("IFEVAL_CAP_CEILING", 8192)
IFEVAL_PASS1_CAP: int = IFEVAL_CAP_LADDER[0]
IFEVAL_RUNG_CAPS: Tuple[int, ...] = tuple(IFEVAL_CAP_LADDER[1:])
#: the per-row IFEval facts the adapter copies into every outcome item of this task
IFEVAL_ROW_FACT_FIELDS: Tuple[str, ...] = _merged("IFEVAL_ROW_FACT_FIELDS", (
    "prompt_level_strict", "prompt_level_loose", "inst_level_strict", "inst_level_loose", "num_instructions",
    "instruction_id_list"))
#: the four metrics (the scorer's value keys the analysis reads per draw) and the instruction count key
IFEVAL_METRIC_FIELDS: Tuple[str, ...] = ("prompt_level_strict", "prompt_level_loose", "inst_level_strict",
                                         "inst_level_loose")
IFEVAL_METRIC_NAMES: Tuple[str, ...] = ("prompt_strict", "prompt_loose", "inst_strict", "inst_loose")
METRIC_BY_FIELD: Mapping[str, str] = dict(zip(IFEVAL_METRIC_FIELDS, IFEVAL_METRIC_NAMES))
#: the outcome-item field set of an IFEval store: the version-2 set + the row facts (lane 1 extends
#: ``TASK_OUTCOME_ITEM_FIELDS`` for this task only; ``completion._check_outcomes_shape`` admits this set for the
#: IFEval task — lane 2's edit, byte-identical for every other task)
#: the IFEval outcome-item shape in the PRODUCER's order (v1 fields, the six facts, sampling_ref LAST) — a pure delegate
IFEVAL_OUTCOME_ITEM_FIELDS: Tuple[str, ...] = _merged("IFEVAL_OUTCOME_ITEM_FIELDS", C.IFEVAL_OUTCOME_ITEM_FIELDS)
#: ... and the NO-SWITCH family's IFEval shape (llama: the row facts + the `none` split's close-tag stray; the llama
#: batch, 2026-09-13) — a pure delegate too; the two together are the shapes the IFEval task admits
IFEVAL_OUTCOME_ITEM_FIELDS_NO_SWITCH: Tuple[str, ...] = _merged("IFEVAL_OUTCOME_ITEM_FIELDS_NO_SWITCH",
                                                                C.IFEVAL_OUTCOME_ITEM_FIELDS_NO_SWITCH)
IFEVAL_OUTCOME_ITEM_FIELD_SETS: Tuple[Tuple[str, ...], ...] = (IFEVAL_OUTCOME_ITEM_FIELDS, IFEVAL_OUTCOME_ITEM_FIELDS_NO_SWITCH)
PRIMARY_SCORER_IFEVAL: str = _merged("PRIMARY_SCORER_IFEVAL", "instruction_following")
PRIMARY_VALUE_KEY_IFEVAL: str = _merged("PRIMARY_VALUE_KEY_IFEVAL", "prompt_level_strict")
LANGDETECT_SEED: int = _merged("LANGDETECT_SEED", 0)

# --------------------------------------------------------------------------- §2 seeds (P1)
IFEVAL_SEED_BASE: int = _merged("IFEVAL_SEED_BASE", 20260909)
IFEVAL_SEED_READOUT: str = _merged("IFEVAL_SEED_READOUT", "ifeval")
#: the plan's spelling (§2, P1: ``sha256(f'{20260909}|{trial}|ifeval|{item_key}')``) — a pure delegate to the
#: contracts' string, which the identity hashes and the matrix's documentary ``seed_rule`` field records
SEED_RULE_IFEVAL: str = _merged("SEED_RULE_IFEVAL", C.SEED_RULE_IFEVAL)


def _ifeval_sample_seed(item_key: str, trial: int) -> int:
    """The sampled re-run's derivation, verbatim in substance (P1): 4 bytes big-endian of
    ``sha256(f"{seed_base}|{trial}|{readout}|{item_key}")``, range ``[0, 2^32)``, NO modulus. Cell-, cap- and
    model-independent by construction (the key names none of them), so the same (item, trial) draws the same seed in
    every lane and every rung re-draw. Pinned by the re-run's literal ``(20260909, 0, "xstest", "xst0000-aaaaaaaa")
    == 3515763156`` (through :func:`sampling_seed`) and ``(…, 0, "ifeval", "1000") == 3028447421``. The argument
    checks (a bool is not a trial; a negative / float / str is not a sample index) live on the resolved derivation,
    ``contracts.ifeval_sample_seed`` (M3); this fallback is the agreement test's cross-check twin."""
    return sampling_seed(IFEVAL_SEED_BASE, trial, IFEVAL_SEED_READOUT, str(item_key))


def sampling_seed(seed_base: int, trial: int, readout: str, item_key: str) -> int:
    """The re-run's general form (its ``served_capability_run.sampling_seed`` twin)."""
    digest = hashlib.sha256(f"{seed_base}|{trial}|{readout}|{item_key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


ifeval_sample_seed: Callable[[str, int], int] = _merged_callable("ifeval_sample_seed", _ifeval_sample_seed)


def _seed_rules_for(tasks: Iterable[str]) -> Dict[str, str]:
    """``dict(SEED_RULES)`` plus the IFEval rule iff ``inspect/ifeval`` is among ``tasks`` — every existing identity
    hashes byte-identically (P1)."""
    out = dict(C.SEED_RULES)
    if IFEVAL_TASK in set(tasks):
        out[IFEVAL_TASK] = SEED_RULE_IFEVAL
    return out


seed_rules_for = _merged_callable("seed_rules_for", _seed_rules_for)

# --------------------------------------------------------------------------- §3 sampling: vendor_preset (P3, P43)
SAMPLING_VENDOR_PRESET: str = _merged("SAMPLING_VENDOR_PRESET", "vendor_preset")
_GEMMA_THINKING = C.THINKING_SAMPLING_BY_MODEL_KEY["gemma4e4b"]
_NEMO_THINKING = C.THINKING_SAMPLING_BY_MODEL_KEY["nemotron3nano4b"]
_NEMO_BATTERY = C.BATTERY_SAMPLING_BY_MODEL_KEY["nemotron3nano4b"]
_QWEN_INSTRUCT = C.QWEN35_VENDOR_ROWS["instruct_general"]
_LLAMA = C.SAMPLING_BY_MODEL_KEY["llama8b"]      # the card row (_LLAMA_CARD)
#: the four rows, the NUMBERS by reference to the existing constants (P3); the ``source`` prose is this lane's and
#: records the re-verification (P4) and, for Nemotron, the grandfathered alternative (P43)
_VENDOR_PRESET_LOCAL: Mapping[str, C.SamplingRow] = {
    "gemma4e4b": C.SamplingRow(
        temperature=_GEMMA_THINKING.temperature, top_p=_GEMMA_THINKING.top_p, top_k=_GEMMA_THINKING.top_k,
        source=("generation_config.json of google/gemma-4-E4B-it @ ee0ef602: temperature 1.0, top_p 0.95, top_k 64 "
                "(do_sample true; vendor default; no penalty term) — the numbers are THINKING_SAMPLING_BY_MODEL_KEY's "
                "by reference (P3); re-read from the shared cluster HF cache snapshot ee0ef602 (P4)")),
    "qwen35_4b": C.SamplingRow(
        temperature=_QWEN_INSTRUCT.temperature, top_p=_QWEN_INSTRUCT.top_p, top_k=_QWEN_INSTRUCT.top_k,
        min_p=_QWEN_INSTRUCT.min_p, presence_penalty=_QWEN_INSTRUCT.presence_penalty,
        repetition_penalty=_QWEN_INSTRUCT.repetition_penalty,
        source=("Qwen/Qwen3.5-4B @ 851bf6e8 ships no generation_config.json; the vendor's INSTRUCT (non-thinking) "
                "general row — QWEN35_VENDOR_ROWS['instruct_general'] by reference (P3): temperature 0.7, top_p 0.8, "
                "top_k 20, min_p 0.0, presence_penalty 1.5 (SENT PER REQUEST — not server-defaultable), "
                "repetition_penalty 1.0 (the card's stated no-op)")),
    "nemotron3nano4b": C.SamplingRow(
        temperature=_NEMO_THINKING.temperature, top_p=_NEMO_THINKING.top_p, top_k=_NEMO_THINKING.top_k,
        source=("generation_config.json of nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 @ dfaf35de: temperature 1.0, top_p "
                "0.95, no top_k — label `generation_config` (P43, SETTLED: the shipped file is the preset for new "
                f"work); the static / over-refusal sampled lanes' user-supplied top_p {_NEMO_BATTERY.top_p} "
                "(pre-ruling; BATTERY_SAMPLING_BY_MODEL_KEY['nemotron3nano4b']) is a GRANDFATHERED exception recorded "
                "beside it here; no reconciliation owed")),
    "llama8b": C.SamplingRow(
        temperature=_LLAMA.temperature, top_p=_LLAMA.top_p, top_k=_LLAMA.top_k,
        source=("generation_config.json of the served NousResearch/Meta-Llama-3.1-8B-Instruct @ d10aef79 (the shared "
                "cluster HF cache snapshot carries the file): temperature 0.6, top_p 0.9, no top_k — the numbers are "
                "SAMPLING_BY_MODEL_KEY['llama8b']'s (_LLAMA_CARD) by reference (P3, P4)")),
}
VENDOR_PRESET_SAMPLING_BY_MODEL_KEY: Mapping[str, C.SamplingRow] = _merged(
    "VENDOR_PRESET_SAMPLING_BY_MODEL_KEY", _VENDOR_PRESET_LOCAL)


def vendor_preset_row(model_key: str) -> C.SamplingRow:
    """The model's ``vendor_preset`` row through the contracts' one table lookup (``sampling_for``); any other key
    REFUSES by name (``SamplingUnverified`` — never a default temperature: a number nobody vouched for would be
    hashed into every store of that model)."""
    return C.sampling_for(model_key, SAMPLING_VENDOR_PRESET)


def sampling_row_dict(row: C.SamplingRow) -> Dict[str, object]:
    """The row as the matrix records it: EXACTLY the keys the row sets (absent = the request carries no such key),
    the source prose last — the GSM8K matrix's shape."""
    out: Dict[str, object] = {"temperature": float(row.temperature)}
    if row.top_p is not None:
        out["top_p"] = float(row.top_p)
    if row.top_k is not None:
        out["top_k"] = int(row.top_k)
    for name in ("min_p", "presence_penalty", "repetition_penalty"):
        v = getattr(row, name, None)
        if v is not None:
            out[name] = float(v)
    out["source"] = row.source
    return out


# --------------------------------------------------------------------------- §5 the ladder (P8–P12)
IFEVAL_TRIGGER_THRESHOLD: float = _merged("IFEVAL_TRIGGER_THRESHOLD", 0.01)
#: the serialized rule string (ASCII hyphen-minus throughout, the GSM8K adapter's convention)
IFEVAL_TRIGGER_RULE: str = _merged("IFEVAL_TRIGGER_RULE", (
    "trigger iff n_at_cap_draws / n_draws > 0.01 (at-cap = stop_reason == max_tokens at the request's cap; "
    "draws = items x trials); redraw the truncated draws = at_cap - loop (boundary included, reported); "
    "ladder 2048 -> 4096 -> 8192; no rung above"))
#: the provenance the ladder block and every figure caption carry (the rule: raw at-cap share of draws)
IFEVAL_TRIGGER_PROVENANCE: str = _merged("IFEVAL_TRIGGER_PROVENANCE", "1 % raw at-cap of draws, per the IFEval ruling")
FLAG_OFF_CAP_BY_TASK: str = _merged("FLAG_OFF_CAP_BY_TASK", "--off-cap-by-task")
OFF_CAP_BY_TASK_ALLOWED: Mapping[str, Tuple[int, ...]] = _merged(
    "OFF_CAP_BY_TASK_ALLOWED", {IFEVAL_TASK_NAME: IFEVAL_RUNG_CAPS})
CAP_SOURCE_OFF_CAP_EXPLICIT: str = _merged("CAP_SOURCE_OFF_CAP_EXPLICIT", "off_cap_explicit")
MANIFEST_KEY_OFF_CAP_BY_TASK: str = _merged("MANIFEST_KEY_OFF_CAP_BY_TASK", "off_cap_by_task")

#: the decimals every SHARE in a ladder record is rounded to (``inspect_rung.SHARE_DECIMALS`` is this name, by import)
SHARE_DECIMALS = 6

# --------------------------------------------------------------------------- §5.3 / P9: THE ladder decision (C1)
#: the key the PLANNER adds to every ``cap_ladder.json`` it files: THE decision at that rung, over the MERGED reading
LADDER_DECISION_KEY = "decision"
LADDER_DECISION_FIELDS: Tuple[str, ...] = (
    "reading", "chain", "n_draws", "n_at_cap", "share_at_cap", "n_rung", "exit", "escalate", "next_cap",
    "eligible_draws", "n_eligible", "threshold", "rule", "trigger_provenance", "transport", "decided_by")
LADDER_DECISION_RULE = (
    "THE ladder decision at every rung is over the MERGED reading: pass 1's draws with every rung's replacements up to and "
    "including this rung (denominator = the full population, P9) — n_at_cap / n_draws > threshold triggers; the eligible "
    "draws are THIS store's truncated non-loop draws (the only draws a rung re-draws). The store's own classification "
    "(exit / n_at_cap / shares over its OWN population) is a diagnostic beside it, never the decision (C1)")
LADDER_DECISION_ESCALATE = "escalate"


class LadderDecisionError(ValueError):
    """A ladder record without THE decision, or one whose decision block is not the contract's — refused by name."""


def ladder_decision_block(*, reading: str, chain: Sequence[int], n_draws: int, n_at_cap: int, n_rung: int, exit: str,
                          next_cap: Optional[int], eligible_draws: Sequence[Sequence[object]], transport: Optional[Mapping[str, object]],
                          threshold: float, rule: str, trigger_provenance: str, decided_by: Mapping[str, object]) -> Dict[str, object]:
    """THE decision block, in :data:`LADDER_DECISION_FIELDS` order, validated by :func:`ladder_decision` before it is returned."""
    draws = [[str(i), int(s)] for i, s in eligible_draws]
    block: Dict[str, object] = {
        "reading": str(reading), "chain": [int(c) for c in chain], "n_draws": int(n_draws), "n_at_cap": int(n_at_cap),
        "share_at_cap": round(int(n_at_cap) / int(n_draws), SHARE_DECIMALS) if int(n_draws) > 0 else None, "n_rung": int(n_rung),
        "exit": str(exit), "escalate": str(exit) == LADDER_DECISION_ESCALATE, "next_cap": (None if next_cap is None else int(next_cap)),
        "eligible_draws": draws, "n_eligible": len(draws), "threshold": float(threshold), "rule": str(rule),
        "trigger_provenance": str(trigger_provenance), "transport": (dict(transport) if transport is not None else None),
        "decided_by": dict(decided_by)}
    assert tuple(block) == LADDER_DECISION_FIELDS
    return ladder_decision({LADDER_DECISION_KEY: block}, where="ladder_decision_block")


def ladder_decision(record: object, *, where: str = "cap_ladder record") -> Dict[str, object]:
    """THE decision of a filed ladder record (``record[LADDER_DECISION_KEY]``), shape-checked: exactly the contract's
    fields; a reading in :data:`READINGS`; a non-empty ascending chain of IFEval ladder caps starting at pass 1; positive
    ``n_draws`` with ``0 <= n_at_cap <= n_draws`` and ``share_at_cap`` the rounding of that ratio; ``escalate`` iff the exit
    is ``escalate`` iff ``next_cap`` names a rung above the chain's last cap on the ladder; ``eligible_draws`` unique
    ``[item_id, sample_idx]`` pairs with ``n_eligible`` their count; a threshold in (0, 1). A record WITHOUT the block is a
    pre-C1 record: the planner must re-file it — refused by name, never read through the store's own exit."""
    if not isinstance(record, Mapping):
        raise LadderDecisionError(f"{where}: not a ladder record")
    dec = record.get(LADDER_DECISION_KEY)
    if not isinstance(dec, Mapping):
        raise LadderDecisionError(f"{where}: no {LADDER_DECISION_KEY!r} block — the record carries the store's own classification "
                                  "only; THE decision (over the merged reading) is filed by the ladder planner (re-run it)")
    if set(dec) != set(LADDER_DECISION_FIELDS):
        raise LadderDecisionError(f"{where}: decision keys {sorted(dec)} are not the contract's (missing "
                                  f"{sorted(set(LADDER_DECISION_FIELDS) - set(dec))}, unknown {sorted(set(dec) - set(LADDER_DECISION_FIELDS))})")
    if dec["reading"] not in READINGS:
        raise LadderDecisionError(f"{where}: decision reading {dec['reading']!r} is not one of {READINGS}")
    chain = dec["chain"]
    if (not isinstance(chain, Sequence) or isinstance(chain, (str, bytes)) or not chain
            or any(isinstance(c, bool) or not isinstance(c, int) for c in chain)
            or list(chain) != [c for c in IFEVAL_CAP_LADDER if c <= max(chain)] or chain[0] != IFEVAL_PASS1_CAP):
        raise LadderDecisionError(f"{where}: decision chain {chain!r} is not the IFEval ladder from pass 1 up to its last cap")
    if (dec["reading"] == READING_PASS1) != (len(chain) == 1):
        raise LadderDecisionError(f"{where}: reading {dec['reading']!r} does not match a chain of {len(chain)} store(s)")
    for k in ("n_draws", "n_at_cap", "n_rung", "n_eligible"):
        v = dec[k]
        if isinstance(v, bool) or not isinstance(v, int) or v < 0 or (k == "n_draws" and v <= 0):
            raise LadderDecisionError(f"{where}: decision {k} {v!r} is not a {'positive' if k == 'n_draws' else 'non-negative'} int")
    if dec["n_at_cap"] > dec["n_draws"] or dec["n_rung"] > dec["n_draws"]:
        raise LadderDecisionError(f"{where}: decision n_at_cap {dec['n_at_cap']} / n_rung {dec['n_rung']} exceed n_draws {dec['n_draws']}")
    if dec["share_at_cap"] != round(dec["n_at_cap"] / dec["n_draws"], SHARE_DECIMALS):
        raise LadderDecisionError(f"{where}: decision share_at_cap {dec['share_at_cap']!r} != round(n_at_cap / n_draws, {SHARE_DECIMALS})")
    if not isinstance(dec["exit"], str) or not dec["exit"]:
        raise LadderDecisionError(f"{where}: decision exit {dec['exit']!r} is not an exit word")
    esc = dec["exit"] == LADDER_DECISION_ESCALATE
    if dec["escalate"] is not esc:
        raise LadderDecisionError(f"{where}: decision escalate {dec['escalate']!r} disagrees with exit {dec['exit']!r}")
    nxt = dec["next_cap"]
    if esc:
        above = [c for c in IFEVAL_CAP_LADDER if c > chain[-1]]
        if not above or nxt != above[0]:
            raise LadderDecisionError(f"{where}: an escalating decision at {chain[-1]} must name the next rung {above[:1]}, got {nxt!r}")
    elif nxt is not None:
        raise LadderDecisionError(f"{where}: a non-escalating decision ({dec['exit']!r}) names a next cap {nxt!r}")
    pairs = _draw_pairs(dec["eligible_draws"], where=f"{where} decision") if dec["eligible_draws"] else []
    if dec["n_eligible"] != len(pairs):
        raise LadderDecisionError(f"{where}: decision n_eligible {dec['n_eligible']} != {len(pairs)} eligible draws")
    if esc and not pairs:
        raise LadderDecisionError(f"{where}: an escalating decision with no eligible draws names no rung")
    thr = dec["threshold"]
    if isinstance(thr, bool) or not isinstance(thr, (int, float)) or not 0 < float(thr) < 1:
        raise LadderDecisionError(f"{where}: decision threshold {thr!r} is not a share in (0, 1)")
    if not isinstance(dec["rule"], str) or not dec["rule"] or not isinstance(dec["trigger_provenance"], str) or not dec["trigger_provenance"]:
        raise LadderDecisionError(f"{where}: decision rule / trigger_provenance must be the instrument's non-empty strings")
    if dec["transport"] is not None and not isinstance(dec["transport"], Mapping):
        raise LadderDecisionError(f"{where}: decision transport {dec['transport']!r} is not a verdict")
    if esc and (not isinstance(dec["transport"], Mapping) or dec["transport"].get("ok") is not True):
        raise LadderDecisionError(f"{where}: an escalating decision needs a PASSED transport verdict, got {dec['transport']!r}")
    if not isinstance(dec["decided_by"], Mapping):
        raise LadderDecisionError(f"{where}: decision decided_by {dec['decided_by']!r} is not a record")
    return dict(dec)


# --------------------------------------------------------------------------- §5.4 the draw-subset contract (P30)
ITEM_SUBSET_TASKS: Tuple[str, ...] = _merged("ITEM_SUBSET_TASKS", (C.GSM8K_PAIRED_TASK_NAME, IFEVAL_TASK_NAME))
ITEM_SUBSET_SCHEMA_V2: str = _merged("ITEM_SUBSET_SCHEMA_V2", "rung_subset/2")
ITEM_SUBSET_FIELDS_V2: Tuple[str, ...] = _merged("ITEM_SUBSET_FIELDS_V2", (
    "schema", "task", "source_out_root", "source_identity_hash", "source_cap", "cap", "rule", "threshold",
    "ids", "draws", "n_items", "n_draws", "n_at_cap", "n_truncated", "n_loop", "n_boundary", "transport_check"))
ITEM_SUBSET_BLOCK_FIELDS_V2: Tuple[str, ...] = _merged("ITEM_SUBSET_BLOCK_FIELDS_V2", (
    *C.ITEM_SUBSET_BLOCK_FIELDS, "draws", "n_items", "n_draws", "schema"))
#: the two counts a DONE marker of a draw-subset store carries beside ``n_items`` (lifted from the summary)
DONE_MARKER_DRAW_FIELDS: Tuple[str, ...] = _merged("DONE_MARKER_DRAW_FIELDS", ("n_draws", "n_expected_draws"))
_HEX16 = re.compile(r"[0-9a-f]{16}")
ItemSubsetError = C.ItemSubsetError


def _draw_pairs(draws: object, *, where: str) -> List[Tuple[str, int]]:
    if not isinstance(draws, Sequence) or isinstance(draws, (str, bytes)) or not draws:
        raise ItemSubsetError(f"{where}: draws must be a non-empty list of [item_id, sample_idx] pairs")
    out: List[Tuple[str, int]] = []
    for d in draws:
        if (not isinstance(d, Sequence) or isinstance(d, (str, bytes)) or len(d) != 2
                or not isinstance(d[0], str) or not d[0]
                or isinstance(d[1], bool) or not isinstance(d[1], int) or d[1] < 0):
            raise ItemSubsetError(f"{where}: draw {d!r} is not [item_id: non-empty str, sample_idx: int >= 0]")
        out.append((d[0], int(d[1])))
    if len(set(out)) != len(out):
        raise ItemSubsetError(f"{where}: draws carry duplicates ({len(out) - len(set(out))})")
    return out


def _item_subset_sha16_v2(draws: Iterable[Sequence[object]]) -> str:
    """The v2 digest: sha256 of the canonical JSON of the SORTED unique ``[item_id, sample_idx]`` draws, 16 hex —
    order-free, and DISTINCT from the v1 id digest by construction (a v1 list of ids never parses as pairs)."""
    pairs = sorted({(str(d[0]), int(d[1])) for d in draws})
    canon = json.dumps([[i, s] for i, s in pairs], separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]


item_subset_sha16_v2 = _merged_callable("item_subset_sha16_v2", _item_subset_sha16_v2)


def _refuse_unless_v2(doc: object, *, where: str) -> Mapping[str, Any]:
    """The v2 delegates' one local check: a document that is not a ``rung_subset/2`` mapping refuses BY NAME here (lane
    2's message, pinned), so ``validate_item_subset_v2`` never reads a v1 file as one — everything else is the contracts'."""
    if not isinstance(doc, Mapping):
        raise ItemSubsetError(f"{where}: expected a JSON object, got {type(doc).__name__}")
    if doc.get("schema") != ITEM_SUBSET_SCHEMA_V2:
        raise ItemSubsetError(f"{where}: schema {doc.get('schema')!r} is not {ITEM_SUBSET_SCHEMA_V2!r}")
    return doc


def _validate_item_subset_v2(doc: object, *, where: str = "item subset") -> Dict[str, Any]:
    """PURE DELEGATE (merge M11, the M3 / M4 precedent): ``contracts.validate_item_subset`` dispatches on ``schema`` and
    runs THE draw contract for ``rung_subset/2`` — EXACTLY :data:`ITEM_SUBSET_FIELDS_V2`; the schema literal; an Inspect
    row task; a 16-hex source identity; ``source_cap < cap <= IFEVAL_CAP_CEILING`` on the IFEval ladder; a non-empty rule;
    a threshold in (0, 1); ``draws`` non-empty unique pairs; ``ids == sorted({i for i, _ in draws})``; ``n_items ==
    len(ids)``; ``len(draws) == n_at_cap - n_loop == n_truncated == n_draws``; ``n_boundary <= n_truncated``; a PASSED
    transport verdict. Lane 2's twin body is gone; the one check kept local is the schema refusal by name."""
    return C.validate_item_subset(_refuse_unless_v2(doc, where=where), where=where)


validate_item_subset_v2 = _merged_callable("validate_item_subset_v2", _validate_item_subset_v2)


def validate_item_subset_any(doc: object, *, where: str = "item subset") -> Dict[str, Any]:
    """Dispatch on ``schema``: v1 (ids) through ``contracts.validate_item_subset``, v2 (draws) through
    :func:`validate_item_subset_v2`; an unknown schema refuses by name."""
    schema = doc.get("schema") if isinstance(doc, Mapping) else None
    if schema == C.ITEM_SUBSET_SCHEMA:
        return C.validate_item_subset(doc, where=where)
    if schema == ITEM_SUBSET_SCHEMA_V2:
        return validate_item_subset_v2(doc, where=where)
    raise ItemSubsetError(f"{where}: schema {schema!r} is neither {C.ITEM_SUBSET_SCHEMA!r} nor {ITEM_SUBSET_SCHEMA_V2!r}")


def _cap_rung_identity_v2(doc: Mapping[str, Any]) -> Dict[str, object]:
    """PURE DELEGATE of ``contracts.cap_rung_identity`` (the digest over the DRAWS for a v2 document — two rungs over the
    same items but different draws never share a key, P30); a non-v2 document refuses by name."""
    return C.cap_rung_identity(_refuse_unless_v2(doc, where="cap_rung_identity_v2"))


cap_rung_identity_v2 = _merged_callable("cap_rung_identity_v2", _cap_rung_identity_v2)


def _item_subset_block_v2(doc: Mapping[str, Any], *, path: str) -> Dict[str, object]:
    """PURE DELEGATE of ``contracts.item_subset_block`` (the v1 fields + ``draws``, ``n_items``, ``n_draws``, ``schema`` =
    :data:`ITEM_SUBSET_BLOCK_FIELDS_V2` for a v2 document); a non-v2 document refuses by name."""
    return C.item_subset_block(_refuse_unless_v2(doc, where="item_subset_block_v2"), path=path)


item_subset_block_v2 = _merged_callable("item_subset_block_v2", _item_subset_block_v2)


def draws_of_subset(doc: Mapping[str, Any]) -> List[Tuple[str, int]]:
    """The subset's draws as ``(item_id, sample_idx)`` tuples, sorted."""
    return sorted((str(i), int(s)) for i, s in doc["draws"])


# --------------------------------------------------------------------------- §1.5 / P42 per-row identity
ROW_IDENTITY_FIELDS: Tuple[str, ...] = _merged("ROW_IDENTITY_FIELDS", (
    "object_artifact_sha256", "route_kind", "view", "producer_tree_sha"))
#: the six keys the cross-eval guard reads per row (the four above + the existing ``thinking`` + ``sampling_mode``)
ROW_GUARD_FIELDS: Tuple[str, ...] = C.ROW_GUARD_FIELDS
ROUTE_KINDS: Tuple[str, ...] = _merged("ROUTE_KINDS", ("base", "prompted", "lens"))
BANK_VIEWS: Tuple[str, ...] = _merged("BANK_VIEWS", ("main", "it"))
VIEW_MAIN, VIEW_IT = BANK_VIEWS
#: the six batch-64 seed LANE views the IFEval instrument also serves (the seed
#: views s1..s3, the paired views s1p..s3p) and the IFEval view vocabulary = the two instrument views + the six; a NEW
#: name beside BANK_VIEWS, which stays ("main", "it") — the contracts' definitions win, pinned equal here
IFEVAL_LANE_VIEWS: Tuple[str, ...] = _merged("IFEVAL_LANE_VIEWS", ("s1", "s2", "s3", "s1p", "s2p", "s3p"))
IFEVAL_VIEWS: Tuple[str, ...] = _merged("IFEVAL_VIEWS", (*BANK_VIEWS, *IFEVAL_LANE_VIEWS))
FLAG_BANK_VIEW: str = _merged("FLAG_BANK_VIEW", "--bank-view")

# --------------------------------------------------------------------------- §7–§8 the matrix vocabulary
CELL_NONE = "none"
CELL_PROMPTED = "prompted__safety_refusal"
PROMPTED_PREFIX = "prompted__"
#: the two scale-12 safety objects served at ``response`` positions (header-located), UNDECLARED under OFF (P15)
IFEVAL_HEADER_LOCATED_CELLS: Tuple[str, ...] = _merged("IFEVAL_HEADER_LOCATED_CELLS",
                                                         ("safety_refusal__psr_kl_all", "safety_refusal__lora_r4"))
IFEVAL_MODELS: Tuple[str, ...] = _merged("IFEVAL_MODELS", ("gemma4e4b", "qwen35_4b", "nemotron3nano4b", "llama8b"))
#: the thirteen launcher slugs for the (model, view) banks on the volume (P27; + the six ``qwen35_4b-sdfbase-s<v>``
#: seed lane views, appended LAST in IFEVAL_LANE_VIEWS order) — a ModelRow key, NEVER a path component
IFEVAL_SLUGS: Mapping[Tuple[str, str], str] = _merged("IFEVAL_SLUGS", {
    ("gemma4e4b", "main"): "gemma4e4b-sdfbase-main", ("gemma4e4b", "it"): "gemma4e4b-sdfbase-it",
    ("qwen35_4b", "main"): "qwen35_4b-sdfbase-main", ("qwen35_4b", "it"): "qwen35_4b-sdfbase-it",
    ("nemotron3nano4b", "main"): "nemotron3nano4b-sdfbase-main",
    ("llama8b", "main"): "llama8b-sdfbase-main", ("llama8b", "it"): "llama8b-sdfbase-it",
    **{("qwen35_4b", _v): f"qwen35_4b-sdfbase-{_v}" for _v in IFEVAL_LANE_VIEWS},
})
#: the pinned HF id + revision per model key (the launcher's ModelRow pins; the tokenizer pins of G-LENGTHS)
IFEVAL_MODEL_PINS: Mapping[str, Tuple[str, str]] = _merged("IFEVAL_MODEL_PINS", {
    "gemma4e4b": ("google/gemma-4-E4B-it", "ee0ef6023621cff504d758262d4e04895a5af4a2"),
    "qwen35_4b": ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"),
    "nemotron3nano4b": ("nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", "dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f"),
    "llama8b": ("NousResearch/Meta-Llama-3.1-8B-Instruct", "d10aef7999a2b5ba950ab3974312feeedbfe0b77"),
})
IMAGE_DEFAULT, IMAGE_NEMOTRON = "default", "nemotron"
#: the geometry entries REGISTERED from the G-LENGTHS measurement (P40); named here, sized by lane 1 from
#: ``configs/sdfbase/ifeval/geometry_measurement.json``
IFEVAL_GEOMETRY_ENTRY_BY_CAP: Mapping[int, str] = _merged(
    "IFEVAL_GEOMETRY_ENTRY_BY_CAP", {cap: f"capability:ifeval_{cap}" for cap in IFEVAL_CAP_LADDER})
#: the follow-up: the constitution rows' OWN entries, one per cap, and the routing by PROMPT CLASS
#: (contracts.ifeval_geometry_entry: none / object / safety -> the base entries; si_constitution -> long_si)
IFEVAL_GEOMETRY_ENTRY_BY_CAP_LONGSI: Mapping[int, str] = _merged(
    "IFEVAL_GEOMETRY_ENTRY_BY_CAP_LONGSI", {cap: f"capability:ifeval_longsi_{cap}" for cap in IFEVAL_CAP_LADDER})
IFEVAL_GEOMETRY_ENTRIES: Tuple[str, ...] = _merged(
    "IFEVAL_GEOMETRY_ENTRIES", (*IFEVAL_GEOMETRY_ENTRY_BY_CAP.values(), *IFEVAL_GEOMETRY_ENTRY_BY_CAP_LONGSI.values()))
IFEVAL_GEOMETRY_ENTRY_BY_CLASS: Mapping[str, Mapping[int, str]] = _merged(
    "IFEVAL_GEOMETRY_ENTRY_BY_CLASS", {C.IFEVAL_PROMPT_CLASS_SHORT: IFEVAL_GEOMETRY_ENTRY_BY_CAP,
                                       C.IFEVAL_PROMPT_CLASS_LONG_SI: IFEVAL_GEOMETRY_ENTRY_BY_CAP_LONGSI})
ifeval_geometry_entry: Callable[[str, int], str] = _merged_callable("ifeval_geometry_entry", C.ifeval_geometry_entry)
ifeval_prompt_class_of_cell: Callable[[str], str] = _merged_callable("ifeval_prompt_class_of_cell", C.ifeval_prompt_class_of_cell)
#: the SHORT constitution SI as a prompted cell (harness ``prompted__constitution``) and the
#: sha256 of its exact text the matrix generator asserts — NEVER ``prompted__si_constitution_full``
CELL_PROMPTED_CONSTITUTION: str = _merged("CELL_PROMPTED_CONSTITUTION", "prompted__si_constitution")
IFEVAL_CONSTITUTION_SI_ID: str = _merged("IFEVAL_CONSTITUTION_SI_ID", "si_constitution")
IFEVAL_CONSTITUTION_SI_SHA256: str = _merged("IFEVAL_CONSTITUTION_SI_SHA256",
                                             "21e06f482a6559b39309c4f1128f0411230c88e02b3a9eaa7b5d50829e6a9a0f")
IFEVAL_GEOMETRY_MEASUREMENT_REL = "configs/sdfbase/ifeval/geometry_measurement.json"
IFEVAL_MATRIX_REL = C.IFEVAL_MATRIX_REL          # ONE spelling (the launcher and the container read it too)
IFEVAL_VIEWS_REL = "configs/sdfbase/ifeval/views.yaml"
#: the SIBLING document of the batch-64 seed lane views — its views.yaml, its
#: matrix, and the register of EVERY IFEval matrix document the hold checks read (the contracts' definitions win)
IFEVAL_VIEWS_QWEN_SEEDREP_REL = "configs/sdfbase/ifeval/views_qwen_seedrep.yaml"
IFEVAL_MATRIX_QWEN_SEEDREP_REL: str = _merged("IFEVAL_MATRIX_QWEN_SEEDREP_REL", "configs/sdfbase/ifeval/matrix_qwen_seedrep.json")
IFEVAL_MATRIX_DOCUMENTS: Tuple[str, ...] = _merged("IFEVAL_MATRIX_DOCUMENTS", (IFEVAL_MATRIX_REL, IFEVAL_MATRIX_QWEN_SEEDREP_REL))
#: the three contrast families the analysis promises; ``make_ifeval_matrix.py --check`` asserts every leg resolves (P32)
CONTRAST_VS_NONE = "cell_minus_none"
CONTRAST_NOISE = "none_it_minus_none_main"
CONTRAST_TRUNCATION = f"merged_minus_r{IFEVAL_PASS1_CAP}"
IFEVAL_CONTRASTS: Tuple[str, ...] = _merged("IFEVAL_CONTRASTS", (CONTRAST_VS_NONE, CONTRAST_NOISE, CONTRAST_TRUNCATION))
READING_PASS1 = f"r{IFEVAL_PASS1_CAP}"
READING_MERGED = "merged"
READINGS: Tuple[str, str] = (READING_PASS1, READING_MERGED)
STATUS_LAUNCH, STATUS_HELD = C.IFEVAL_STATUS_LAUNCH, C.IFEVAL_STATUS_HELD
#: the follow-up: a DEFERRED row is a design slot (AlphaSteer / Arditi at minimal over-refusal strength) whose object
#: sha is filled later from a per-model record; never launched, never a leg, refused by the dispatcher by name
STATUS_DEFERRED: str = _merged("IFEVAL_STATUS_DEFERRED", "deferred")
ROW_STATUSES: Tuple[str, ...] = _merged("IFEVAL_ROW_STATUSES", (STATUS_LAUNCH, STATUS_HELD, STATUS_DEFERRED))
MATRIX_SCHEMA = "ifeval_matrix/1"
#: the wave-wide dispatcher state (P39) — ONE file per wave; the per-configuration record is ``ifeval_paths().dispatch``
IFEVAL_DISPATCH_STATE = C.IFEVAL_DISPATCH_STATE
LEDGER_FILENAME = C.LEDGER_FILENAME
#: == ``tiers.cap_ladder.RECORD_NAME`` (spelled, not imported: that module pulls the model stack in at import)
CAP_LADDER_FILENAME = "cap_ladder.json"
LADDER_STATE_FILENAME = "ladder_state.json"
RUNG_SUBSET_FILENAME = "rung_subset.json"
BUDGET_FILENAME = C.OOS_PROMPT_BUDGET_FILENAME


#: the TAG (the GSM8K rule: the bare key for ``main``, ``<key>_<view>`` otherwise) — the contracts' one rule
ifeval_tag = C.ifeval_tag


def ifeval_cell_slug(cell: str) -> str:
    """The out-root path segment: ``__`` -> ``--`` (``cell_controls.store_key_token``'s convention, mirrored)."""
    return cell.replace("__", "--")


#: ``none`` -> base, ``prompted__<si>`` -> prompted, every other cell -> lens — the contracts' one reading (P42)
ifeval_route_kind = C.route_kind_of_cell


class IfevalPaths(NamedTuple):
    """Every path of ONE (model, view, cell, rung) configuration, relative to the wave root (P37)."""
    tag: str
    cell_slug: str
    rung: int
    configuration_id: str
    out_root: str
    subset: Optional[str]
    cap_ladder: str
    ladder_state: str
    budget: str
    dispatch: str


def _ifeval_paths(model_key: str, view: str, cell: str, rung: int = IFEVAL_PASS1_CAP) -> IfevalPaths:
    """THE ONE path constructor. ``out_root = controls/<TAG>/<cell_slug>/vendor_preset[__rung<cap>]``;
    ``subset = rung<cap>/<TAG>/<cell_slug>/rung_subset.json`` (None at pass 1: pass 1 has no subset);
    ``cap_ladder = <out_root>/cap_ladder.json`` — the record classifying THE STORE AT THIS RUNG (which decides the next
    rung; the record that DECIDED rung r is ``ifeval_paths(..., previous rung).cap_ladder``); ``ladder_state =
    <out_root>/ladder_state.json``; ``budget = budgets/<TAG>/<cell_slug>/<cap>/oos_prompt_budget.json``; ``dispatch =
    dispatch/<TAG>/<cell_slug>/rung<cap>.json`` (the per-configuration dispatcher record). A slug is never a path
    component; every path is unique over (model, view, cell, rung) by construction (the test asserts it matrix-wide)."""
    if model_key not in IFEVAL_MODELS:
        raise ValueError(f"model key {model_key!r} is not one of {IFEVAL_MODELS}")
    if int(rung) not in IFEVAL_CAP_LADDER:
        raise ValueError(f"rung {rung!r} is not on the IFEval ladder {IFEVAL_CAP_LADDER}")
    if not cell or "/" in cell:
        raise ValueError(f"cell {cell!r} is not a cell id")
    tag, slug, rung = ifeval_tag(model_key, view), ifeval_cell_slug(cell), int(rung)
    base = f"controls/{tag}/{slug}/{SAMPLING_VENDOR_PRESET}"
    out_root = base if rung == IFEVAL_PASS1_CAP else f"{base}__rung{rung}"
    return IfevalPaths(
        tag=tag, cell_slug=slug, rung=rung, configuration_id=f"{tag}/{slug}/rung{rung}", out_root=out_root,
        subset=(None if rung == IFEVAL_PASS1_CAP else f"rung{rung}/{tag}/{slug}/{RUNG_SUBSET_FILENAME}"),
        cap_ladder=f"{out_root}/{CAP_LADDER_FILENAME}", ladder_state=f"{out_root}/{LADDER_STATE_FILENAME}",
        budget=f"budgets/{tag}/{slug}/{rung}/{BUDGET_FILENAME}", dispatch=f"dispatch/{tag}/{slug}/rung{rung}.json")


ifeval_paths: Callable[..., IfevalPaths] = _merged_callable("ifeval_paths", _ifeval_paths)


def previous_rung(cap: int) -> Optional[int]:
    """The rung below ``cap`` on the IFEval ladder (None at pass 1)."""
    below = [r for r in IFEVAL_CAP_LADDER if r < int(cap)]
    return below[-1] if below else None


def _ifeval_expected_stamps(*, fitted: bool, header_located: bool = False) -> Tuple[str, str, str]:
    """The row's expected ``(cache_salt_regime, resumed_prefill_policy, render_regime)`` under thinking OFF,
    UNDECLARED (P1-3 / P15), through ``contracts.expected_stamp_regimes``: the base / prompted route ->
    ``(cell-stable, not-consulted, thinking_off)``; an all-position object attested ``always`` -> ``(cell-stable,
    always, thinking_off)``; a header-located object (served at ``response`` — a response-fit object, contract
    §2.5.11 row 3/4) -> the ``conditional`` row. An EXPECTATION the analysis holds beside the standup's attested
    policy (gate I3 reads the attested label; a disagreement is a named finding, never a silent pass)."""
    if header_located and not fitted:
        raise ValueError("a header-located row is a fitted object; header_located=True with fitted=False names no cell class")
    if not fitted:
        return tuple(C.expected_stamp_regimes(lens=False, arm=C.THINKING_OFF, policy=C.POLICY_NOT_CONSULTED))
    policy = _proxy.RESUMED_PREFILL_CONDITIONAL if header_located else _proxy.RESUMED_PREFILL_ALWAYS
    return tuple(C.expected_stamp_regimes(lens=True, arm=C.THINKING_OFF, policy=policy))


ifeval_expected_stamps = _merged_callable("ifeval_expected_stamps", _ifeval_expected_stamps)

# --------------------------------------------------------------------------- the served-path constants the gates read
#: the composed thinking-OFF server pin (the static full22 wave's; Qwen / Nemotron compose it, gemma / Llama none)
THINKING_OFF_TEMPLATE_KWARGS: str = _merged("THINKING_OFF_TEMPLATE_KWARGS", '{"enable_thinking": false}')


def ifeval_effective_arm(family: object) -> str:
    """The arm an IFEval task of ``family`` RUNS under when the matrix row says ``off`` (the frozen matrix's ``thinking``
    on every row and the launcher line's ``--thinking off``): ``contracts.effective_arm_for_family`` — ``not_available``
    for a family with no thinking channel (llama; the no-switch policy), ``off`` elsewhere. The one
    derivation the validators, the analysis and the synthetic stores read, so a Llama
    store's hashed ``thinking_by_task``, per-row ``thinking`` and summary read ``not_available`` and the matrix bytes
    do not move."""
    return C.effective_arm_for_family(C.THINKING_OFF, family)


def ifeval_expected_summary_stamps(row: Mapping[str, Any]) -> Dict[str, str]:
    """The stamp triple a STORE's summary must carry for ``row`` — the row's frozen ``expected_stamps`` (the
    ``(cache_salt_regime, resumed_prefill_policy, render_regime)`` of ``ifeval_expected_stamps``, always spelled at the
    OFF arm) with the render regime FAMILY-RESOLVED through ``contracts.expected_render_regime(ifeval_effective_arm(...))``:
    a no-switch family's requests carry no ``enable_thinking``, so the proxy stamps ``unspecified`` (lane X's seam 2); the
    salt / policy pair is the OFF row's either way."""
    want = dict(zip(C.EXPECTED_STAMP_TRIPLE_FIELDS, row["expected_stamps"]))
    want["render_regime"] = C.expected_render_regime(ifeval_effective_arm(row.get("family")))
    return want
#: which families compose the pin (the template names ``enable_thinking``)
FAMILIES_COMPOSING_THINKING_OFF_PIN: Tuple[str, ...] = ("qwen3_5", "nemotron_h")
ENDPOINT_KEY_DEFAULT_CHAT_TEMPLATE_KWARGS: str = _merged("ENDPOINT_KEY_DEFAULT_CHAT_TEMPLATE_KWARGS",
                                                          "default_chat_template_kwargs")
ENDPOINT_KEY_GEOMETRY_KV_SHRINK: str = _merged("ENDPOINT_KEY_GEOMETRY_KV_SHRINK", "kv_shrink")
#: request-evidence v3 (P29): the sampling / seed / trial fields lane 1 adds at the HTTP boundary
REQUEST_EVIDENCE_VERSION_V3: int = _merged("REQUEST_EVIDENCE_VERSION_V3", 3)
REQUEST_EVIDENCE_V3_FIELDS: Tuple[str, ...] = _merged("REQUEST_EVIDENCE_V3_FIELDS", (
    "seed", "temperature", "top_p", "top_k", "min_p", "repetition_penalty", "presence_penalty_present",
    "presence_penalty_value", "sample_idx", "item_id"))
#: the full v3 record — a pure delegate: ``contracts.REQUEST_EVIDENCE_FIELDS`` already ends in the ten v3 fields
REQUEST_EVIDENCE_FIELDS_V3: Tuple[str, ...] = _merged("REQUEST_EVIDENCE_FIELDS_V3", C.REQUEST_EVIDENCE_FIELDS)


def _request_evidence_record_v3(raw: bytes, *, headers: Mapping[str, object], ts: Optional[str] = None) -> Dict[str, object]:
    """The VERSION-3 request-evidence record (P29) — a pure delegate to ``contracts.request_evidence_record``, whose
    ``REQUEST_EVIDENCE_VERSION`` is 3: the v1/v2 fields FIRST (a reader keyed on them keeps working), then the sampling
    / seed / trial fields read off the SAME parsed wire body and the same attempt's ``X-Tinycap-Sample`` header —
    ``seed``, ``temperature``, ``top_p``, ``top_k``, ``min_p``, ``repetition_penalty`` (each null when the body names
    none), ``presence_penalty_present`` + ``presence_penalty_value`` (the wire-level PROOF Qwen's 1.5 rode per request),
    ``sample_idx`` (the trial) and ``item_id``. The synthetic stores and the analysis tests write through this name, so
    they exercise the production writer. Refuses by name a record that is not at version 3 (a contracts rollback)."""
    rec = dict(C.request_evidence_record(raw, headers=headers, ts=ts))
    if int(rec.get("request_evidence_version") or 0) != REQUEST_EVIDENCE_VERSION_V3:
        raise ValueError(f"contracts.request_evidence_record wrote version {rec.get('request_evidence_version')!r}, "
                         f"not the v3 record this instrument's gates read ({REQUEST_EVIDENCE_VERSION_V3})")
    assert tuple(rec) == REQUEST_EVIDENCE_FIELDS_V3, tuple(rec)
    return rec


request_evidence_record_v3 = _merged_callable("request_evidence_record_v3", _request_evidence_record_v3)


def request_evidence_sampling_problem(rec: Mapping[str, object], *, row: C.SamplingRow, task: str = None,
                                      arm: str = C.THINKING_OFF) -> Optional[str]:
    """Why one TASK record does not prove the row on the wire (the per-record check): a record below version 3 is
    NOT RECORDED (never a pass); every sampling field must equal the row's (an unset row term must be absent);
    ``presence_penalty_present`` iff the row sets one, with its value; the seed must equal the IFEval rule for
    ``(item_id, sample_idx)``; the thinking switch as the ARM sends it — ``enable_thinking_value`` False under a switch
    arm (``off``), and NO ``enable_thinking`` on the wire under the no-switch arm ``not_available`` (a family with no
    thinking channel; the gate reads the same absence). None when the record passes."""
    C.check_thinking_arm(arm)
    if int(rec.get("request_evidence_version") or 0) < REQUEST_EVIDENCE_VERSION_V3:
        return f"request_evidence_version {rec.get('request_evidence_version')!r} < 3: the sampling fields are not recorded"
    for f in ("temperature", "top_p", "top_k", "min_p", "repetition_penalty"):
        want = getattr(row, f, None)
        got = rec.get(f)
        if (want is None) != (got is None) or (want is not None and float(want) != float(got)):
            return f"{f} on the wire {got!r} != the row's {want!r}"
    want_pp = row.presence_penalty
    if bool(rec.get("presence_penalty_present")) != (want_pp is not None):
        return f"presence_penalty_present {rec.get('presence_penalty_present')!r} but the row {'sets' if want_pp is not None else 'sets no'} presence_penalty"
    if want_pp is not None and rec.get("presence_penalty_value") != float(want_pp):
        return f"presence_penalty on the wire {rec.get('presence_penalty_value')!r} != the row's {want_pp}"
    if rec.get("item_id") is None or rec.get("sample_idx") is None:
        return "the record names no (item_id, sample_idx)"
    want_seed = ifeval_sample_seed(str(rec["item_id"]), int(rec["sample_idx"]))
    if rec.get("seed") != want_seed:
        return f"seed on the wire {rec.get('seed')!r} != the IFEval rule's {want_seed} for ({rec['item_id']}, {rec['sample_idx']})"
    if C.sends_thinking_kwargs(arm):
        if rec.get("enable_thinking_value") is not False:
            return f"enable_thinking_value {rec.get('enable_thinking_value')!r} is not False (thinking OFF)"
    elif rec.get("enable_thinking_present") is not False:
        return (f"enable_thinking_present {rec.get('enable_thinking_present')!r} (value {rec.get('enable_thinking_value')!r}) — the "
                f"no-switch arm {arm!r} sends no chat_template_kwargs.enable_thinking")
    return None


#: the server-defaultable keys: ``presence_penalty`` is NOT among them, so it rides per request
PRESENCE_PENALTY_PER_REQUEST_MODELS: Tuple[str, ...] = tuple(
    k for k, r in VENDOR_PRESET_SAMPLING_BY_MODEL_KEY.items() if r.presence_penalty is not None)

# --------------------------------------------------------------------------- §9.1 the OBSERVED decode rate (C5)
#: the ledger columns the rate reads (``contracts.LEDGER_ROW_FIELDS``): the row's ``ts`` is stamped when the request
#: ARRIVES at the ledger (before the tokenize pass and the forward), ``latency_s`` when its response is written, so
#: ``ts + latency_s`` is the request's end
LEDGER_TS_FIELD = "ts"
LEDGER_LATENCY_FIELD = "latency_s"
LEDGER_COMPLETION_TOKENS_FIELD = "usage_completion_tokens"
#: the decode-rate definition, serialized into every prediction and every ledger reading that carries the number
DECODE_RATE_RULE = ("aggregate decode tok/s = sum of the accepted attempt's forwarded task completions / the generation wall "
                    "(last request end - first request start; start = the ledger row's ts, end = ts + latency_s). NEVER "
                    "sum(completion tokens) / sum(latency_s): S concurrent requests each count their own overlapping "
                    "seconds, which divides the aggregate by ~S (C5)")
DECODE_RATE_MIN_ROWS = 10


def _utc_stamp(stamp: object) -> Optional["_dt.datetime"]:
    """An ISO-8601 stamp (``Z`` or an offset) as an aware UTC datetime; ``None`` for anything else — a NAIVE stamp
    included, because a time with no zone cannot be placed against one that has (the GSM8K validators' rule)."""
    if not isinstance(stamp, str) or not stamp:
        return None
    try:
        dt = _dt.datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt.astimezone(_dt.timezone.utc) if dt.tzinfo is not None else None


def generation_span(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    """The wall the ledger rows' requests spanned: ``first_start`` = min ``ts``, ``last_end`` = max ``ts + latency_s``,
    ``elapsed_s`` (0.0 for no rows). A row without a placeable ``ts`` or a positive ``latency_s`` refuses by name — an
    unmeasured request would silently shrink the wall."""
    starts: List["_dt.datetime"] = []
    ends: List["_dt.datetime"] = []
    for r in rows:
        t0 = _utc_stamp(r.get(LEDGER_TS_FIELD))
        lat = r.get(LEDGER_LATENCY_FIELD)
        if t0 is None:
            raise ValueError(f"ledger row (task {r.get('task')!r} sample {r.get('sample')!r}) carries no placeable "
                             f"{LEDGER_TS_FIELD} ({r.get(LEDGER_TS_FIELD)!r}) — the generation wall cannot be read")
        if isinstance(lat, bool) or not isinstance(lat, (int, float)) or lat <= 0:
            raise ValueError(f"ledger row (task {r.get('task')!r} sample {r.get('sample')!r}) carries no positive "
                             f"{LEDGER_LATENCY_FIELD} ({lat!r})")
        starts.append(t0)
        ends.append(t0 + _dt.timedelta(seconds=float(lat)))
    if not starts:
        return {"first_start": None, "last_end": None, "elapsed_s": 0.0, "n_rows": 0}
    first, last = min(starts), max(ends)
    return {"first_start": first.isoformat(timespec="milliseconds"), "last_end": last.isoformat(timespec="milliseconds"),
            "elapsed_s": round((last - first).total_seconds(), 4), "n_rows": len(starts)}


def aggregate_decode_tok_s(rows: Sequence[Mapping[str, object]], *, min_rows: int = DECODE_RATE_MIN_ROWS) -> Dict[str, object]:
    """:data:`DECODE_RATE_RULE` over the FORWARDED task rows of ONE attempt (the caller scopes them): ``tok_s`` = the
    aggregate rate, with the facts it rests on (``completion_tokens``, ``elapsed_s``, the span, ``n_rows``) and, labelled
    beside it for comparison only, ``per_request_tok_s`` = ``sum(tokens) / sum(latency_s)`` — the per-request reading
    the estimate must NOT use. Refuses fewer than ``min_rows`` measured rows, a row without an integer completion count,
    or a zero wall."""
    rows = list(rows)
    if len(rows) < int(min_rows):
        raise ValueError(f"only {len(rows)} forwarded task rows (need >= {min_rows}) — no observed decode rate")
    comp: List[int] = []
    for r in rows:
        c = r.get(LEDGER_COMPLETION_TOKENS_FIELD)
        if isinstance(c, bool) or not isinstance(c, int) or c < 0:
            raise ValueError(f"ledger row (task {r.get('task')!r} sample {r.get('sample')!r}) carries no integer "
                             f"{LEDGER_COMPLETION_TOKENS_FIELD} ({c!r})")
        comp.append(c)
    span = generation_span(rows)
    elapsed = float(span["elapsed_s"])
    if elapsed <= 0:
        raise ValueError(f"the {len(rows)} rows span no wall ({span['first_start']} .. {span['last_end']}) — no aggregate rate")
    total = sum(comp)
    total_lat = sum(float(r[LEDGER_LATENCY_FIELD]) for r in rows)
    return {"tok_s": total / elapsed, "completion_tokens": total, "elapsed_s": elapsed, "n_rows": len(rows),
            "first_start": span["first_start"], "last_end": span["last_end"],
            "per_request_tok_s": (total / total_lat) if total_lat > 0 else None, "rule": DECODE_RATE_RULE}


def merge_agreement_report() -> Dict[str, Dict[str, object]]:
    """For every name in :data:`MERGE_AGREEMENT`: whether ``contracts`` defines it and whether the two agree —
    the test's substance, exposed for a dry read (functions compare on a shared probe when both exist)."""
    out: Dict[str, Dict[str, object]] = {}
    for name, local in MERGE_AGREEMENT.items():
        theirs = getattr(C, name, None)
        present = hasattr(C, name)
        agree: Optional[bool]
        if not present:
            agree = None
        elif callable(local):
            agree = _callables_agree(name, local, theirs)
        else:
            agree = _values_agree(local, theirs)
        out[name] = {"in_contracts": present, "agree": agree}
    return out


def _values_agree(a: object, b: object) -> bool:
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return dict(a) == dict(b)
    if isinstance(a, tuple) and isinstance(b, (tuple, list)):
        return tuple(a) == tuple(b)
    return a == b


#: the agreement probes' shared ``rung_subset/2`` document (synthetic: invented ids, a placeholder identity hash, a
#: passed transport verdict) — the three v2 delegates and any contracts twin of theirs must agree on it (merge M11)
_PROBE_SUBSET_V2: Dict[str, Any] = {
    "schema": ITEM_SUBSET_SCHEMA_V2, "task": IFEVAL_TASK, "source_out_root": "controls/gemma4e4b/none/vendor_preset",
    "source_identity_hash": "0123456789abcdef", "source_cap": 2048, "cap": 4096, "rule": IFEVAL_TRIGGER_RULE,
    "threshold": IFEVAL_TRIGGER_THRESHOLD, "ids": ["a", "b"], "draws": [["a", 0], ["a", 1], ["b", 1]],
    "n_items": 2, "n_draws": 3, "n_at_cap": 4, "n_truncated": 3, "n_loop": 1, "n_boundary": 1,
    "transport_check": {"ok": True, "next_cap": 4096, "hops": {"driver": 7200.0, "proxy_upstream": 7200.0}},
}

_PROBES: Dict[str, Tuple[tuple, dict]] = {
    "ifeval_sample_seed": (("1000", 1), {}),
    "seed_rules_for": (((IFEVAL_TASK, "inspect/math"),), {}),
    "ifeval_paths": (("gemma4e4b", "main", "safety_refusal__psr_kl_all", 4096), {}),
    "ifeval_expected_stamps": ((), {"fitted": True}),
    "item_subset_sha16_v2": (([["a", 0], ["a", 1], ["b", 1]],), {}),
    "validate_item_subset_v2": ((_PROBE_SUBSET_V2,), {"where": "probe"}),
    "cap_rung_identity_v2": ((_PROBE_SUBSET_V2,), {}),
    "item_subset_block_v2": ((_PROBE_SUBSET_V2,), {"path": "/vol/probe/rung_subset.json"}),
    "ifeval_geometry_entry": (("prompted__si_constitution", 4096), {}),
    "ifeval_prompt_class_of_cell": (("prompted__si_constitution",), {}),
}


def _callables_agree(name: str, local: Callable[..., Any], theirs: Callable[..., Any]) -> bool:
    args, kw = _PROBES.get(name, ((), {}))
    try:
        return local(*args, **kw) == theirs(*args, **kw)
    except Exception:  # noqa: BLE001 — a probe the other side refuses IS a disagreement
        return False


__all__ = [n for n in dir() if not n.startswith("_") and n not in ("C", "Any", "Callable", "Dict", "Iterable", "List",
                                                                   "Mapping", "NamedTuple", "Optional", "Sequence",
                                                                   "Tuple", "annotations", "hashlib", "json", "re")]
