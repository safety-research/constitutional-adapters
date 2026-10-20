"""Paired per-item comparison of two stores' rows on ONE task metric — the
discipline ``scripts/compare_served_hf_mmlu.py`` applied to
``s2parity/mmlu_letter``, generalized to any task:
the Inspect comparisons (``s2parity/mmlu_letter`` vs ``inspect/mmlu_letter_valid``
etc.) and the standard-vs-retired GSM8K comparison
(``s2parity/gsm8k_0shot`` rows vs ``inspect/gsm8k_paired200`` rows joined on
``paired_id``, U5-5).

Rules, all of them refusals rather than silent adjustments:

* **Engines are verified per side.** Every per-item row on a side must carry
  the engine that side was declared to carry; a served store handed in as
  the HF reference refuses. (Both sides may be ``vllm`` — the Inspect
  comparison pairs two served scorers — but each side still states it.)
* **Pairing keys are explicit.** A side joins on ``item_id`` (the default)
  or on ``extra_json.paired_id``
  (Inspect rows: Inspect's own sample id is the ``item_id``; OUR store's id
  rides in ``paired_id`` — P32). A ``paired_id`` side drops rows whose
  ``paired_id`` is null and COUNTS them (``unpaired_rows``): those items have
  no partner by construction (the stock 200 ``mmlu_0_shot`` timing items).
* **Item-set mismatch refuses by default.** The seeded ``--limit`` subsets
  must match; a pairing that silently drops items measures the overlap, not
  the instrument. ``allow_unmatched=True`` is for the comparisons whose
  overlap is partial BY DESIGN (GSM8K platinum edits vs the stock questions,
  P8) — the unmatched ids are then LISTED in the report, never dropped
  quietly, and an empty overlap still refuses.
* **Duplicates refuse.** A resumed store files each item once. Several
  SAMPLES of one item are not duplicates — see the next rule — but two rows
  on one key with the same (or no) ``sample_idx`` are.
* **avg@k inputs reduce to the per-item mean FIRST, then pair.** A side may
  carry several sample rows per item under the paired metric (distinct
  ``sample_idx`` on one pairing key): they are averaged into ONE per-item
  value before pairing (``Item.n_samples`` = how many). The producer's own
  mean row (the bare ``acc`` row with ``samples_per_item`` / ``values`` in
  its ``extra_json``; the per-sample rows sit under ``acc@<k>``,
  ``contracts.sample_metric``) is read as the per-item value it already is,
  its ``samples_per_item`` reported. The bootstrap below resamples ITEMS —
  the clusters — never individual samples, and the report says how many
  samples per item each side had (``samples_per_item``).
* **McNemar only on binary outcomes.** avg@k per-item means (MATH-500's
  ``acc`` = mean of 4 samples) are compared as paired means with the
  non-identical items listed; ``mcnemar`` is null with the reason stated.
* **The paired bootstrap CI rides next to the analytic paired SE** (Miller
  2024, "Adding Error Bars to Evals"): the matched items are resampled with
  replacement ``B`` times (default 10,000, fixed seed, default 0) and the
  percentile interval of the resampled mean per-item delta (B − A) is
  reported as ``paired_bootstrap`` — the repo's ONE paired bootstrap,
  ``sisteer.eval.realign_stats.paired_bootstrap_ci``, does the resampling;
  this module does not carry a second — alongside ``paired_se`` =
  sd(per-item deltas, ddof=1) / sqrt(n). Pairing cuts the SE because item
  difficulty is shared; the CI is on the DELTA, never on the two means.
* **Per-side unparseable / truncation facts come only from fields the rows
  ALREADY carry** (``sides.<a|b>.quality``): the retired instrument's
  ``extraction_failed`` sibling rows, ``pred`` / ``used_fallback``; the
  Inspect rows' ``answer`` (``""`` when ``inspect_ai``'s ``match_str`` found
  no number — the unparseable signal; None when the scorer records no answer
  at all, as MATH's sympy scorer does not), ``content`` /
  ``content_empty_with_reasoning``, and ``completion_tokens >= max_tokens``
  (``at_token_cap``). A fact the producer did not record is reported as
  null — a finish/stop reason is ``"not recorded by the producer"`` on
  today's Inspect rows — never approximated by a proxy.

* **Two SERVED-LENS stores must have served the same lens reduction mode**
  (E46). Since the stacked-reductions landing, a lens group's reductions are
  one stacked GEMM instead of B single-row calls; the two are not bit-identical
  at B > 1, so a ``stacked`` store and a ``per_row`` store are not poolable and
  the equality must be asserted from the STAMPS at analysis time, not assumed.
  Each side's mode is read from the ``manifest.json`` beside its ``rows.jsonl``
  by ``contracts.read_store_reduction_mode`` — THE ONE resolver, which
  ``cell_controls`` calls too (E55 #3). A route block that states none is
  LEGACY = ``per_row`` ONLY on a store whose tree positively lacks the stamp; on
  any other store, and whenever the key is present-but-null or out of
  vocabulary, the side refuses (:class:`PairingError`) rather than being read
  as per_row — that reading is the one that can silently pool a stacked store
  with a per_row one. A side with no manifest, or a manifest with no ``route``
  block, is not a served-lens store at all — an HF reference, a bare rows file
  — and states ``null`` with the reason in
  ``sides.<a|b>.reduction_mode_source`` rather than being coerced into a mode
  it never had. Two stated modes that DIFFER refuse; anything else reports what
  it could and could not determine.

Aggregate rows (``item_id == "__aggregate__"``) are never paired.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# torch-free: THE ONE reduction-mode resolver, shared with cell_controls
from ..tinycap.contracts import (
    REDUCTION_MODE_NOT_APPLICABLE,
    THINKING_OFF,
    THINKING_ON,
    StoreReductionMode,
    StoreServingEngine,
    read_store_reduction_mode,
    read_store_serving_engine,
    serving_engine_mismatch,
)
from ..tinycap.contracts import reduction_mode_mismatch as _engine_reduction_mode_mismatch

__all__ = [
    "PairingError", "Side", "Item", "AGGREGATE_ITEM_ID", "KEY_ITEM_ID",
    "KEY_PAIRED_ID", "PAIR_KEYS", "load_items", "pair_ids", "compare",
    "compare_sides", "DEFAULT_CARRY", "DEFAULT_BOOTSTRAP", "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_CI", "EXTRACTION_FAILED_METRIC", "FINISH_REASON_NOT_RECORDED",
    "paired_bootstrap", "paired_se", "side_quality", "samples_summary",
    "store_reduction_mode", "store_reduction_mode_reading", "reduction_mode_mismatch",
    "store_serving_engine_reading", "serving_engine_mismatch",
    "STORE_MANIFEST_NAME",
    # fit-support transfer, experiment 1 (plan S5.1 / §6.1): the ONE estimand
    "LEG_CELLS", "LEG_ARMS", "LEG_KEYS", "CONTRAST_SIGNS", "CONTRASTS",
    "COMPLETE_CASE_LABEL", "DEFAULT_MAX_DEGENERATE_FRAC", "leg_label",
    "paired_bootstrap_conditional_2x2",
]

AGGREGATE_ITEM_ID = "__aggregate__"
KEY_ITEM_ID = "item_id"
KEY_PAIRED_ID = "paired_id"
PAIR_KEYS: Tuple[str, ...] = (KEY_ITEM_ID, KEY_PAIRED_ID)
#: ``extra_json`` facts carried into the discordant table when present
#: (the letter-MC pick / gold, a generation task's answer / pred, the
#: sample index of a single-sample row, the per-sample ``values`` of an
#: avg@k mean row).
DEFAULT_CARRY: Tuple[str, ...] = ("pick", "gold", "answer", "pred", "sample_idx", "values")
#: Paired bootstrap defaults (U5-5): resamples, seed, interval level.
DEFAULT_BOOTSTRAP = 10_000
DEFAULT_BOOTSTRAP_SEED = 0
DEFAULT_CI = 0.95
#: The retired s2parity GSM8K instrument files one ``extraction_failed``
#: row per item next to its ``acc`` row (value 1.0 = no answer extracted).
EXTRACTION_FAILED_METRIC = "extraction_failed"
#: Reported for a finish/stop reason when no row carries one — the Inspect
#: producer records ``completion_tokens`` and ``max_tokens`` but no reason.
FINISH_REASON_NOT_RECORDED = "not recorded by the producer"
_FINISH_REASON_KEYS: Tuple[str, ...] = ("finish_reason", "stop_reason")


#: The per-cell manifest a capeval store writes beside its ``rows.jsonl``
#: (``capeval.store.CellStore.manifest_path``) — where a served store records
#: the lens reduction mode it was served under.
STORE_MANIFEST_NAME = "manifest.json"


def store_reduction_mode_reading(rows_path) -> StoreReductionMode:
    """E55 #3: the store's own statement about the lens arithmetic it was
    served under, from the ``manifest.json`` beside ``rows_path`` — read by
    :func:`contracts.read_store_reduction_mode`, THE ONE resolver, which
    ``cell_controls.source_store_reduction_mode`` calls too.

    Before this, ``pairing`` carried its own reading and mapped ANY present
    route block lacking ``reduction_mode`` to per_row while ``cell_controls``
    refused for a current or unknown store (and mapped a PRESENT-BUT-NULL key
    to per_row as well, which round 16 had already removed on the other side).
    Damage `manifest.route.reduction_mode` on a current STACKED store and this
    seam pooled it with a genuine per_row one. Two implementations of one
    question is how they came to disagree twice; there is now one.

    The historical mapping is unchanged where it was right: a route block with
    no key on a store whose tree positively lacks the stamp is legacy per_row, and
    that is every store the HF-vs-served spot check reads today."""
    return read_store_reduction_mode(_store_manifest_doc(Path(rows_path)),
                                     where=str(Path(rows_path).parent / STORE_MANIFEST_NAME))


def store_serving_engine_reading(rows_path) -> StoreServingEngine:
    """The store's own statement about the ENGINE that served it, from the
    ``manifest.json`` beside ``rows_path`` — :func:`contracts.read_store_serving_
    engine`, every spelling the producer writes (top-level, the tinycap route
    block, the hashed identity copies), independent of the lens reduction mode.
    A served store spelling none is a lens store by era; a manifest that is not
    a served store's (an HF reference; no manifest) states nothing."""
    return read_store_serving_engine(_store_manifest_doc(Path(rows_path)),
                                     where=str(Path(rows_path).parent / STORE_MANIFEST_NAME))


def _store_manifest_doc(rows_path: Path):
    """The manifest beside a rows file, or ``None`` — unreadable is reported,
    never raised: a comparison of two rows files whose stores are gone is still
    a legitimate thing to run, and the caller decides."""
    manifest = Path(rows_path).parent / STORE_MANIFEST_NAME
    if not manifest.exists():
        return None
    try:
        doc = json.loads(manifest.read_text())
    except Exception:  # noqa: BLE001 — reported, never guessed
        return None
    return doc if isinstance(doc, Mapping) else None


def store_reduction_mode(rows_path) -> Tuple[Optional[str], str]:
    """``(mode, source)`` — the LENS REDUCTION MODE the store containing
    ``rows_path`` was served under (E46), or ``(None, why)`` when this store
    states none: no manifest, one that will not parse, or no ``route`` block at
    all (an HF reference store).

    RAISES :class:`PairingError` when the store HAS the field, or should have,
    and cannot say what it served — the key present and null, a value outside
    the vocabulary, or the key absent on a store that does not positively
    predate the stamp (E55 #3). Those used to be read as per_row, which is the
    one reading that can silently pool a stacked store with a per_row one.
    What is NEVER done is inventing a mode for a store that does not state
    one."""
    reading = store_reduction_mode_reading(rows_path)
    if reading.refusal is not None:
        raise PairingError(reading.refusal)
    return reading.mode, reading.why


def reduction_mode_mismatch(a: Optional[str], b: Optional[str]) -> Optional[str]:
    """Why two stores may NOT be pooled — they were served under different
    lens reduction modes (E46) — or ``None`` when they agree or when either
    side states nothing. A stacked group's reductions are one GEMM where a
    per-row group's are B single-row calls; they are not bit-identical at
    B > 1, so pooling the two measures the arithmetic as well as the
    intervention."""
    if a is None or b is None or a == b:
        return None
    if REDUCTION_MODE_NOT_APPLICABLE in (a, b):
        # a NATIVE store (`not_applicable` — it has no lens
        # arithmetic) against any lens mode is a comparison ACROSS SERVING ENGINES, which measures
        # the engine as well as the intervention; contracts' one engine-aware verdict words it
        # (native vs native agreed above). The lens/lens text below is byte-identical to before.
        return _engine_reduction_mode_mismatch(a, b)
    return (f"lens reduction mode {a!r} vs {b!r} — a stacked store and a per_row store were "
            "served under different arithmetic (one stacked GEMM per group vs B single-row "
            "calls; not bit-identical at B > 1) and are not poolable; compare within one mode, "
            "or re-run one side under the other (E46)")


class PairingError(RuntimeError):
    """A pairing/identity defect — loud, never a silently partial pair."""


@dataclass(frozen=True)
class Side:
    """One side of a comparison: which rows, which engine they must carry,
    and how they join."""

    label: str
    rows_path: Path
    task: str
    engine: str
    metric: str = "acc"
    key: str = KEY_ITEM_ID

    def __post_init__(self):
        if self.key not in PAIR_KEYS:
            raise PairingError(f"pairing key must be one of {PAIR_KEYS}, got {self.key!r}")
        if not self.task or not self.engine or not self.metric:
            raise PairingError(f"side {self.label!r}: task, engine and metric are required")


@dataclass
class Item:
    value: float
    item_id: str                 # the row's own item_id (Inspect's id on a paired_id side)
    extra: Dict[str, Any] = field(default_factory=dict)
    #: samples averaged into ``value``: the rows reduced on this key, or the
    #: producer's ``samples_per_item`` on its own mean row; 1 for one sample.
    n_samples: int = 1


def _extra(row: Mapping[str, Any]) -> Dict[str, Any]:
    e = row.get("extra_json")
    if isinstance(e, str):
        try:
            e = json.loads(e)
        except json.JSONDecodeError as err:
            raise PairingError(f"row {row.get('item_id')!r}: extra_json is not JSON: {err}") from err
    return dict(e) if isinstance(e, Mapping) else {}


def _read_rows(side: Side) -> List[Dict[str, Any]]:
    """Every row of ``side.rows_path`` (refusing an absent file in the side's
    name) — parsed once, so two sides over the SAME store (the Inspect
    comparisons' documented form) share one parse."""
    path = Path(side.rows_path)
    if not path.is_file():
        raise PairingError(f"{side.label}: rows file {path} does not exist")
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _is_count(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def _reduce_item(side: Side, path: Path, key: str,
                 group: Sequence[Tuple[Mapping[str, Any], Dict[str, Any]]],
                 carry: Sequence[str]) -> Item:
    """One :class:`Item` from every row filed on ``key``: a single row is
    the per-item value (its ``samples_per_item``, when the producer's mean
    row states one, is the sample count); several rows are SAMPLES of one
    item iff they carry distinct ``sample_idx`` — averaged here, the
    per-sample ``values`` carried in ``sample_idx`` order — and duplicates
    otherwise (refused)."""
    if len(group) == 1:
        r, extra = group[0]
        k = extra.get("samples_per_item")
        return Item(value=float(r["value"]), item_id=str(r["item_id"]),
                    extra={c: extra[c] for c in carry if c in extra},
                    n_samples=int(k) if _is_count(k) and k >= 1 else 1)
    idxs = [extra.get("sample_idx") for _, extra in group]
    if not all(_is_count(i) for i in idxs) or len(set(idxs)) != len(idxs):
        raise PairingError(
            f"{side.label} ({path}): duplicate per-item row for {key!r} on "
            f"{side.task}/{side.metric} — a resumed store files each item once (several "
            "SAMPLES of one item must carry distinct sample_idx); refusing")
    ordered = sorted(group, key=lambda ge: ge[1]["sample_idx"])
    values = [float(r["value"]) for r, _ in ordered]
    first_r, first_extra = ordered[0]
    extra = {c: first_extra[c] for c in carry if c in first_extra and c != "sample_idx"}
    extra["values"] = values
    return Item(value=sum(values) / len(values), item_id=str(first_r["item_id"]),
                extra=extra, n_samples=len(values))


def samples_summary(items: Iterable[Item]) -> Dict[str, int]:
    """``{min, max}`` of ``Item.n_samples`` — how many samples per item a
    side had (both 1 for single-sample rows, both k for a uniform avg@k)."""
    ks = [it.n_samples for it in items]
    return {"min": min(ks), "max": max(ks)}


def side_quality(extras: Sequence[Mapping[str, Any]], *,
                 extraction_failed: Sequence[Any] = ()) -> Dict[str, Any]:
    """Per-side unparseable / truncation facts from fields the rows ALREADY
    carry — over the side's per-item rows of the paired metric (``extras``
    = their parsed ``extra_json``) plus the task's ``extraction_failed``
    sibling rows' values. Every fact is a count, or None when no row records
    the field (``finish_reason`` says so in words). ``unparseable`` is the
    ONE number U5-5 asks for, taken from the most authoritative field the
    side carries and named in ``unparseable_source``:

    1. ``extraction_failed`` sibling rows (the retired instrument's own flag);
    2. else ``extra_json.answer`` null or ``""`` — ``inspect_ai.scorer``'s
       ``match_str`` returns ``""`` when it finds no number — but only when
       the scorer records an answer at all (MATH's sympy scorer leaves it
       None on every row, which is "not recorded", not 100 % unparseable);
    3. else None, and the source says it is not derivable."""
    n = len(extras)

    def recorded(key: str) -> bool:
        return any(key in e for e in extras)

    def count(key: str, pred) -> Optional[int]:
        return sum(1 for e in extras if key in e and pred(e[key])) if recorded(key) else None

    answer_recorded = any(e.get("answer") is not None for e in extras)
    caps = [(e.get("completion_tokens"), e.get("max_tokens")) for e in extras
            if _is_count(e.get("completion_tokens")) and _is_count(e.get("max_tokens"))]
    q: Dict[str, Any] = {
        "n_rows": n,
        "answer_recorded": answer_recorded,
        "answer_missing": (sum(1 for e in extras if e.get("answer") in (None, ""))
                           if answer_recorded else None),
        "content_empty": count("content", lambda v: v == ""),
        "content_empty_with_reasoning": count("content_empty_with_reasoning", bool),
        "at_token_cap": sum(1 for c, m in caps if c >= m) if caps else None,
        "extraction_failed": (sum(1 for v in extraction_failed if v == 1)
                              if extraction_failed else None),
        "extraction_failed_rows": len(extraction_failed) if extraction_failed else None,
        "pred_missing": count("pred", lambda v: v is None),
        "used_fallback": count("used_fallback", bool),
    }
    fr_key = next((k for k in _FINISH_REASON_KEYS if recorded(k)), None)
    if fr_key is None:
        q["finish_reason"] = FINISH_REASON_NOT_RECORDED
        q["finish_reason_field"] = None
    else:
        q["finish_reason"] = dict(sorted(Counter(str(e[fr_key]) for e in extras
                                                 if fr_key in e).items()))
        q["finish_reason_field"] = fr_key
    if extraction_failed:
        q["unparseable"] = q["extraction_failed"]
        denom = len(extraction_failed)
        q["unparseable_source"] = ("extraction_failed rows (the retired s2parity "
                                   "instrument's own per-item flag)")
    elif answer_recorded:
        q["unparseable"] = q["answer_missing"]
        denom = n
        q["unparseable_source"] = ("extra_json.answer null or empty (inspect_ai match_str "
                                   "found no number in the completion)")
    else:
        q["unparseable"] = None
        denom = 0
        q["unparseable_source"] = ("not derivable from the rows: no extraction_failed "
                                   "rows and no scorer-recorded answer")
    q["unparseable_rate"] = (q["unparseable"] / denom) if denom else None
    return q


def load_items(side: Side, *, carry: Sequence[str] = DEFAULT_CARRY,
               rows: Optional[Sequence[Mapping[str, Any]]] = None
               ) -> Tuple[Dict[str, Item], Dict[str, Any]]:
    """``({pair_key: Item}, stats)`` for one side's per-item rows of
    ``side.task`` / ``side.metric``, engine verified on every row, several
    samples of one item reduced to their mean (:func:`_reduce_item`).
    ``stats`` carries ``n_rows`` (sample rows counted individually),
    ``n_items``, ``samples_per_item`` (:func:`samples_summary`), ``quality``
    (:func:`side_quality`) and, on a ``paired_id`` side, ``unpaired_rows``.
    ``rows`` is the side's file already parsed (:func:`_read_rows`); None
    reads it here."""
    groups: Dict[str, List[Tuple[Mapping[str, Any], Dict[str, Any]]]] = {}
    extras: List[Dict[str, Any]] = []
    extraction_failed: List[Any] = []
    n_rows = unpaired = 0
    path = Path(side.rows_path)
    if rows is None:
        rows = _read_rows(side)
    for r in rows:
        if r.get("task") != side.task or r.get("item_id") == AGGREGATE_ITEM_ID:
            continue
        metric = r.get("metric")
        if metric == EXTRACTION_FAILED_METRIC:
            extraction_failed.append(r.get("value"))
            continue
        if metric != side.metric:
            continue
        eng = r.get("engine")
        if eng != side.engine:
            raise PairingError(
                f"{side.label} ({path}): row {r.get('item_id')!r} carries engine {eng!r} "
                f"but this side is declared {side.engine!r} — the two sides are the "
                "point of the comparison; refusing a store on the wrong side")
        n_rows += 1
        extra = _extra(r)
        extras.append(extra)
        if side.key == KEY_ITEM_ID:
            k = r["item_id"]
        else:
            k = extra.get(KEY_PAIRED_ID)
            if k is None:
                unpaired += 1
                continue
        k = str(k)
        v = r.get("value")
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            raise PairingError(f"{side.label}: row {k!r} has a non-numeric value {v!r}")
        groups.setdefault(k, []).append((r, extra))
    if not groups:
        raise PairingError(
            f"{side.label} ({path}): no per-item rows for task {side.task!r} metric "
            f"{side.metric!r} with engine {side.engine!r}"
            + (f" (all {unpaired} rows had a null {KEY_PAIRED_ID})" if unpaired else ""))
    items = {k: _reduce_item(side, path, k, grp, carry) for k, grp in groups.items()}
    stats: Dict[str, Any] = {"n_rows": n_rows, "n_items": len(items)}
    if side.key == KEY_PAIRED_ID:
        stats["unpaired_rows"] = unpaired
    stats["samples_per_item"] = samples_summary(items.values())
    stats["quality"] = side_quality(extras, extraction_failed=extraction_failed)
    return items, stats


def pair_ids(a: Mapping[str, Any], b: Mapping[str, Any], *, allow_unmatched: bool = False
             ) -> Tuple[List[str], List[str], List[str]]:
    """``(paired ids sorted, only_a, only_b)``; refuses a mismatch unless
    ``allow_unmatched``, and an EMPTY overlap always."""
    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))
    ids = sorted(set(a) & set(b))
    if (only_a or only_b) and not allow_unmatched:
        raise PairingError(
            f"item sets differ: {len(only_a)} only on side A (first {only_a[:3]}), "
            f"{len(only_b)} only on side B (first {only_b[:3]}) — the seeded --limit "
            "subsets must match; refusing to pair a partial overlap (pass "
            "allow_unmatched for a comparison whose overlap is partial by design)")
    if not ids:
        raise PairingError("the two sides share no items — nothing to pair")
    return ids, only_a, only_b


def _is_binary(values: Iterable[float]) -> bool:
    return all(v in (0.0, 1.0) for v in values)


def paired_bootstrap(a: Mapping[str, float], b: Mapping[str, float], *,
                     n_resamples: int = DEFAULT_BOOTSTRAP, seed: int = DEFAULT_BOOTSTRAP_SEED,
                     ci: float = DEFAULT_CI) -> Dict[str, Any]:
    """Percentile paired-bootstrap CI of mean(B) − mean(A) over the ITEMS
    both sides carry (``a`` / ``b``: ``{pair_key: per-item value}``, the
    same keys — an avg@k side's values are already the per-item means, so
    the resampling unit is the item, never a sample). Delegates the
    resampling to :func:`sisteer.eval.realign_stats.paired_bootstrap_ci`
    (seeded ``numpy`` generator, id-keyed, hard-fails on any key mismatch).

    Returns ``{n_items, delta, ci_low, ci_high, level, B, seed,
    excludes_zero}`` — ``delta`` is the point estimate mean(B) − mean(A)."""
    if not 0.0 < ci < 1.0:
        raise PairingError(f"the interval level must be in (0, 1), got {ci!r}")
    if not isinstance(n_resamples, int) or isinstance(n_resamples, bool) or n_resamples < 1:
        raise PairingError(f"the bootstrap needs >= 1 resamples, got {n_resamples!r}")
    if set(a) != set(b):
        raise PairingError("paired bootstrap: the two sides' item keys differ — pair first")
    from sisteer.eval.realign_stats import paired_bootstrap_ci

    # realign_stats reports mean(first − second); B − A wants B first.
    r = paired_bootstrap_ci({k: float(v) for k, v in b.items()},
                            {k: float(v) for k, v in a.items()},
                            n=n_resamples, seed=seed, alpha=1.0 - ci)
    return {"n_items": r["n_items"], "delta": r["mean_diff"], "ci_low": r["ci_low"],
            "ci_high": r["ci_high"], "level": ci, "B": n_resamples, "seed": seed,
            "excludes_zero": r["excludes_zero"]}


def paired_se(deltas: Sequence[float]) -> Optional[float]:
    """The analytic paired standard error: sd(per-item deltas, ddof=1) /
    sqrt(n); None below two items (undefined, never 0 by fiat)."""
    n = len(deltas)
    if n < 2:
        return None
    return float(np.std(np.asarray(deltas, dtype=float), ddof=1) / math.sqrt(n))


def compare(a: Mapping[str, Item], b: Mapping[str, Item], *, allow_unmatched: bool = False,
            bootstrap: Optional[int] = DEFAULT_BOOTSTRAP,
            bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED, ci: float = DEFAULT_CI
            ) -> Dict[str, Any]:
    """The paired report for two loaded sides (pure): ``n_items``, the two
    means, ``delta_b_minus_a``, the discordant table (every item whose
    values differ, with the carried extras), McNemar on binary outcomes
    (else null + reason), the pick agreement when both sides carry
    ``pick``, the unmatched id lists, ``paired_bootstrap``
    (:func:`paired_bootstrap`; null with the reason when ``bootstrap`` is
    0/None), ``paired_se`` (:func:`paired_se`) and ``samples_per_item`` per
    side over the matched items."""
    ids, only_a, only_b = pair_ids(a, b, allow_unmatched=allow_unmatched)
    n = len(ids)
    va = [a[i].value for i in ids]
    vb = [b[i].value for i in ids]
    mean_a = sum(va) / n
    mean_b = sum(vb) / n
    deltas = [y - x for x, y in zip(va, vb)]
    discordant = [
        {"id": i, "a": {"value": a[i].value, "item_id": a[i].item_id, **a[i].extra},
         "b": {"value": b[i].value, "item_id": b[i].item_id, **b[i].extra}}
        for i in ids if a[i].value != b[i].value]
    binary = _is_binary(va) and _is_binary(vb)
    if binary:
        from sisteer.eval.realign_stats import mcnemar

        mc: Optional[Dict[str, Any]] = mcnemar({i: int(a[i].value) for i in ids},
                                               {i: int(b[i].value) for i in ids})
        mc_reason = None
    else:
        mc = None
        mc_reason = ("non-binary per-item values (e.g. an avg@k mean) — paired means "
                     "and the differing-item table are the readout; McNemar needs 0/1")
    if bootstrap:
        pb: Optional[Dict[str, Any]] = paired_bootstrap(
            {i: a[i].value for i in ids}, {i: b[i].value for i in ids},
            n_resamples=bootstrap, seed=bootstrap_seed, ci=ci)
        pb_reason = None
    else:
        pb = None
        pb_reason = "paired bootstrap disabled (0 resamples requested)"
    picks_known = all("pick" in a[i].extra and "pick" in b[i].extra for i in ids)
    pick_agree = (sum(a[i].extra["pick"] == b[i].extra["pick"] for i in ids) / n
                  if picks_known else None)
    return {
        "n_items": n,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "delta_b_minus_a": mean_b - mean_a,
        "paired_se": paired_se(deltas),
        "paired_bootstrap": pb,
        "paired_bootstrap_unavailable_reason": pb_reason,
        "samples_per_item": {"a": samples_summary(a[i] for i in ids),
                             "b": samples_summary(b[i] for i in ids)},
        "binary": binary,
        "n_discordant": len(discordant),
        "discordant": discordant,
        "mcnemar": mc,
        "mcnemar_unavailable_reason": mc_reason,
        "pick_agreement": pick_agree,
        "n_pick_disagreements": (None if pick_agree is None else round((1 - pick_agree) * n)),
        "only_a": only_a,
        "only_b": only_b,
    }


def compare_sides(side_a: Side, side_b: Side, *, allow_unmatched: bool = False,
                  carry: Sequence[str] = DEFAULT_CARRY,
                  bootstrap: Optional[int] = DEFAULT_BOOTSTRAP,
                  bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED, ci: float = DEFAULT_CI
                  ) -> Dict[str, Any]:
    """Load both sides and compare; the report names both sides fully
    (``sides.<a|b>``: label, rows, task, metric, engine, key,
    ``reduction_mode`` + ``reduction_mode_source``, ``n_rows``, ``n_items``,
    ``samples_per_item``, ``quality``, ``unpaired_rows`` on a ``paired_id``
    side). Two sides over one file (the Inspect comparisons pass the same
    served ``rows.jsonl`` for A and B) parse it once. Two served-lens stores
    that state DIFFERENT lens reduction modes refuse before anything is read
    (:func:`reduction_mode_mismatch`; E46)."""
    # E46: two SERVED-LENS stores must have served the same reduction mode.
    # Checked BEFORE the rows are read — a refusal here costs nothing, and a
    # comparison of two stores served under different lens arithmetic measures
    # the arithmetic as well as the intervention.
    readings = {}
    for side in (side_a, side_b):
        reading = store_reduction_mode_reading(side.rows_path)
        if reading.refusal is not None:
            # E55 #3: a store that HAS the field and cannot say refuses here
            # exactly as it does at the merge, instead of being read as per_row
            raise PairingError(f"{side.label!r}: {reading.refusal}")
        readings[side.label] = reading
    mode_a, src_a = readings[side_a.label].mode, readings[side_a.label].why
    mode_b, src_b = readings[side_b.label].mode, readings[side_b.label].why
    why = reduction_mode_mismatch(mode_a, mode_b)
    if why is not None:
        raise PairingError(
            f"{side_a.label!r} ({src_a}) vs {side_b.label!r} ({src_b}): {why}")
    # the SERVING ENGINE, read off EVERY spelling the producer writes and INDEPENDENTLY of the lens
    # reduction mode: the mode reading above sees the tinycap `route` block only, and a native
    # XSTest / OR-Bench / MMLU-only store (the legacy controls: no tinycap atoms, so no `route`
    # block; the engine is stamped top-level and in the hashed identity) read "nothing stated"
    # there and paired SILENTLY with its lens counterpart. Two served stores on different engines
    # refuse here; a served store stating no engine under any spelling is a lens store by era; an
    # HF reference (not a served store) states nothing and pairs exactly as before
    engines = {}
    for side in (side_a, side_b):
        reading = store_serving_engine_reading(side.rows_path)
        if reading.refusal is not None:
            raise PairingError(f"{side.label!r}: {reading.refusal}")
        engines[side.label] = reading
    why = serving_engine_mismatch(engines[side_a.label], engines[side_b.label])
    if why is not None:
        raise PairingError(f"{side_a.label!r} ({engines[side_a.label].why}) vs {side_b.label!r} "
                           f"({engines[side_b.label].why}): {why}")
    # the engine each store STATES rides the report per side —
    # present iff the store spells one (a native store under any spelling; a lens store that says
    # so); absent on every by-era lens store, so a lens report is byte-identical to before
    engine_a = ({"serving_engine": engines[side_a.label].stated}
                if engines[side_a.label].stated is not None else {})
    engine_b = ({"serving_engine": engines[side_b.label].stated}
                if engines[side_b.label].stated is not None else {})
    rows_a = _read_rows(side_a)
    a, sa = load_items(side_a, carry=carry, rows=rows_a)
    same_file = Path(side_a.rows_path).resolve() == Path(side_b.rows_path).resolve()
    b, sb = load_items(side_b, carry=carry, rows=rows_a if same_file else _read_rows(side_b))
    rep = compare(a, b, allow_unmatched=allow_unmatched, bootstrap=bootstrap,
                  bootstrap_seed=bootstrap_seed, ci=ci)
    rep["sides"] = {
        "a": {"label": side_a.label, "rows": str(side_a.rows_path), "task": side_a.task,
              "metric": side_a.metric, "engine": side_a.engine, "key": side_a.key,
              # what this store says it was served under, and how we know —
              # null is "this store states no lens reduction mode", never a
              # guessed per_row (E46)
              "reduction_mode": mode_a, "reduction_mode_source": src_a, **engine_a, **sa},
        "b": {"label": side_b.label, "rows": str(side_b.rows_path), "task": side_b.task,
              "metric": side_b.metric, "engine": side_b.engine, "key": side_b.key,
              "reduction_mode": mode_b, "reduction_mode_source": src_b, **engine_b, **sb},
    }
    return rep


# ---------------------------------------------------------------------------- #
# fit-support transfer, experiment 1 — the ONE estimand and its                 #
# SHARP contrast bounds                                                        #
# ---------------------------------------------------------------------------- #

#: The 2x2's leg vocabulary: cell KIND x thinking arm. ``none`` is the
#: unsteered base cell, ``psr`` the steered one; every per-cell statistic is
#: keyed on the FULL leg, never pooled through ``petri_stats.classify_cell_id``
#: (which folds a ``@thinking`` variant into its base).
LEG_CELLS: Tuple[str, ...] = ("none", "psr")
LEG_ARMS: Tuple[str, ...] = (THINKING_OFF, THINKING_ON)
LEG_KEYS: Tuple[Tuple[str, str], ...] = tuple((c, a) for c in LEG_CELLS for a in LEG_ARMS)
#: Every contrast as a SIGNED combination of legs. The sharp identified
#: interval of a contrast (§6.1) puts each ``+1`` leg at its
#: censored-wrong accuracy and each ``-1`` leg at its censored-right accuracy
#: for the LOWER end, and the reverse for the upper — leg-wise ADVERSE
#: imputations, which are free of each other because each leg's censored
#: items are its own generations. The DiD's row is exactly S5.1's four-leg
#: assignment (lower: psr/on and none/off at acc⁻, none/on and psr/off at acc⁺).
CONTRAST_SIGNS: Mapping[str, Mapping[Tuple[str, str], int]] = {
    "effect_off": {("psr", THINKING_OFF): +1, ("none", THINKING_OFF): -1},
    "effect_on": {("psr", THINKING_ON): +1, ("none", THINKING_ON): -1},
    "shift_none": {("none", THINKING_ON): +1, ("none", THINKING_OFF): -1},
    "shift_psr": {("psr", THINKING_ON): +1, ("psr", THINKING_OFF): -1},
    "did": {("psr", THINKING_ON): +1, ("none", THINKING_ON): -1,
            ("psr", THINKING_OFF): -1, ("none", THINKING_OFF): +1},
}
CONTRASTS: Tuple[str, ...] = tuple(CONTRAST_SIGNS)
#: The label every complete-case (intersection) number wears — a DIFFERENT
#: estimand from the conditional one (both means over the items every leg of
#: the contrast scored), reported SECONDARY beside it.
COMPLETE_CASE_LABEL = "complete-case (intersection of scored items)"
#: A replicate in which some leg's own denominator is 0 is DISCARDED and
#: counted; more than this fraction of them refuses (plan S5.1: "refused above
#: 0.1 %").
DEFAULT_MAX_DEGENERATE_FRAC = 0.001


def leg_label(leg: Tuple[str, str]) -> str:
    """``"<cell>/<arm>"`` — the JSON-safe spelling of a leg key."""
    return f"{leg[0]}/{leg[1]}"


def _check_legs(legs: Mapping[Tuple[str, str], Mapping[str, Tuple[int, Optional[float]]]]
                ) -> List[str]:
    """The four leg keys exactly, ONE common attempted id set (a difference
    refuses LISTING the ids — the attempted set is common by construction or
    the run is broken; there is no ``allow_unmatched`` here), and per item a
    ``(mask, value)`` pair with ``mask`` in {0, 1}, ``value`` in {0, 1} where
    the mask is 1 and ``None`` where it is 0. Returns the sorted ids."""
    have = set(legs) if isinstance(legs, Mapping) else None
    if have != set(LEG_KEYS):
        raise PairingError(
            f"the conditional 2x2 needs exactly the four legs {[leg_label(k) for k in LEG_KEYS]}, "
            f"got {sorted(leg_label(k) for k in have) if have is not None else type(legs).__name__}")
    ids_by_leg = {k: set(legs[k]) for k in LEG_KEYS}
    common = set.intersection(*ids_by_leg.values())
    union = set.union(*ids_by_leg.values())
    if common != union:
        detail = []
        for k in LEG_KEYS:
            missing = sorted(union - ids_by_leg[k])
            if missing:
                detail.append(f"{leg_label(k)} lacks {len(missing)} (first {missing[:3]})")
        raise PairingError(
            "the four legs' ATTEMPTED id sets differ — the estimand is defined over ONE common "
            f"set of attempted items; refusing: {'; '.join(detail)}")
    if not common:
        raise PairingError("the conditional 2x2 has no items")
    for k in LEG_KEYS:
        for i, pair in legs[k].items():
            if not (isinstance(pair, (tuple, list)) and len(pair) == 2):
                raise PairingError(f"{leg_label(k)} item {i!r}: expected (mask, value), got {pair!r}")
            m, v = pair
            if isinstance(m, bool) or m not in (0, 1):
                raise PairingError(f"{leg_label(k)} item {i!r}: mask {m!r} is not 0 or 1")
            if m == 1 and (isinstance(v, bool) or v not in (0, 1, 0.0, 1.0)):
                raise PairingError(
                    f"{leg_label(k)} item {i!r}: a scored item's value must be 0 or 1, got {v!r}")
            if m == 0 and v is not None:
                raise PairingError(
                    f"{leg_label(k)} item {i!r}: a masked-out item carries no value (got {v!r}) — "
                    "impute in the caller's branch, never here")
    return sorted(common)


def _quantiles(x: np.ndarray, ci: float) -> Tuple[float, float]:
    """The percentile interval, the ``realign_stats.paired_bootstrap_ci`` way
    (``np.quantile`` at ``alpha/2`` and ``1 - alpha/2``)."""
    alpha = 1.0 - ci
    lo, hi = np.quantile(x, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def _exact_replicate_contrasts(correct: Mapping[Tuple[str, str], np.ndarray],
                               scored: Mapping[Tuple[str, str], np.ndarray], n: int
                               ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Every contrast of a batch of replicates — ``(point, bound_low,
    bound_high)`` arrays — each as ONE float division of EXACT integers.
    ``correct[k]`` / ``scored[k]`` are the replicates' integer counts per leg
    (int64), ``n`` the common attempted count.

    The conditional point Σ_k s_k·correct_k / scored_k is formed over the
    COMMON denominator Π_k scored_k (the contrast's replicate denominators
    multiplied — legs whose conditional denominators differ under censoring
    share it; the product rather than the lcm because it vectorises in int64
    and yields the SAME rational, hence the same correctly rounded float):
    numerator Σ_k s_k·correct_k·Π_{j≠k} scored_j. The bounds' numerators are
    the signed integer sums over ``n`` (a +1 leg contributes ``correct`` to the
    lower end and ``correct + censored`` to the upper, a −1 leg the reverse,
    negated). Every integer is below 2**53 (the caller refuses ``4·n⁴`` past
    it), so each quotient is the correctly rounded value of the exact rational
    — the representation ``float(Fraction)`` gives the full-set points and
    bounds — and no two separately rounded accuracies are ever subtracted
    (the review's P2: a CI endpoint ``0.21999999999999997`` beside an exact ON
    effect ``0.22`` read "amplifies" where the inclusive-edge rule says
    "transfers"; both are now ``44/200`` as one float). Without censoring
    every value is an exact multiple of ``1/n``. A zero denominator (a
    degenerate replicate) divides to nan / ±inf, which the caller discards."""
    out: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for name, signs in CONTRAST_SIGNS.items():
        den: Any = 1
        for k in signs:
            den = den * scored[k]
        num: Any = 0
        for k, s in signs.items():
            others: Any = 1
            for j in signs:
                if j != k:
                    others = others * scored[j]
            num = num + s * correct[k] * others
        low_num: Any = 0
        high_num: Any = 0
        for k, s in signs.items():
            censored = n - scored[k]
            low_num = low_num + (correct[k] if s > 0 else -(correct[k] + censored))
            high_num = high_num + ((correct[k] + censored) if s > 0 else -correct[k])
        with np.errstate(divide="ignore", invalid="ignore"):
            out[name] = (num / den, low_num / n, high_num / n)
    return out


def paired_bootstrap_conditional_2x2(
        legs: Mapping[Tuple[str, str], Mapping[str, Tuple[int, Optional[float]]]], *,
        n_resamples: int = DEFAULT_BOOTSTRAP, seed: int = DEFAULT_BOOTSTRAP_SEED,
        ci: float = DEFAULT_CI, max_degenerate_frac: float = DEFAULT_MAX_DEGENERATE_FRAC
        ) -> Dict[str, Any]:
    """The ONE estimand of the thinking-transfer 2x2 and its sharp
    partial-identification bounds, in one replicate machinery.

    ``legs = {(cell, arm): {item_id: (mask, value)}}`` over the SAME attempted id
    set for all four ``LEG_KEYS`` (a difference refuses, listing ids); ``mask``
    in {0, 1} is the branch's score-availability bit, ``value`` the 0/1
    correctness where the mask is 1 and ``None`` where it is 0 (the caller
    builds both branches' masks).

    Per leg the CONDITIONAL accuracy with its own denominator (Σ m·y / Σ m);
    the contrasts per ``CONTRAST_SIGNS``. ``n_resamples`` replicates, each ONE
    index draw applied to all four legs, every accuracy recomputed with its
    replicate denominator; percentile CIs at ``ci``; a replicate with a zero
    denominator is discarded and counted (refused above ``max_degenerate_frac``).

    Bounds: per leg acc⁻ = correct / n_attempted and acc⁺ = (correct + censored)
    / n_attempted; per contrast the sharp interval puts a +1 leg at acc⁻ and a
    −1 leg at acc⁺ for the lower end (reversed for the upper) — sharp because
    the four imputations are free of each other. ``bound_low`` / ``bound_high``
    are the full-set bound the verdict reads, ``bound_ci_low`` / ``bound_ci_high``
    the 2.5th / 97.5th percentiles of the ends. ``complete_case`` (the difference
    of means over the items every leg of the contrast scored) is a different
    estimand, labelled :data:`COMPLETE_CASE_LABEL`, secondary. With every mask 1
    this agrees with :func:`paired_bootstrap` on ``effect_off`` to rounding.

    EXACT full-set arithmetic. Every full-set point, every bound endpoint and
    every complete-case delta is computed as a RATIONAL over the integer counts
    (``fractions.Fraction``: the conditional accuracy ``correct / scored``, the
    bounds ``correct / n`` and ``(correct + censored) / n`` over the common
    attempted denominator) and converted to float ONCE. So two exactly equal
    effects give a DiD of exactly ``0.0`` and a bound of exactly ``[0.0, 0.0]``
    — never the ``±1e-16`` a floating sum of four ratios leaves — and the two
    exact facts the verdict reads are filed as integers: ``point_sign``
    (``-1 / 0 / +1`` of the rational point) and ``sign_identified`` (the
    rational bound excludes zero). Zero is its own sign.

    EXACT replicate arithmetic too (the closing review's P2). Every replicate
    statistic — each leg's conditional accuracy, every contrast point and both
    bound ends — is ONE float division of exact integer counts
    (:func:`_exact_replicate_contrasts`: the contrast over the product of its
    legs' replicate denominators, the bounds over ``n``), so each is the
    correctly rounded value of the exact rational — the SAME representation
    the full-set rationals take when converted to float — and no replicate
    ever subtracts two separately rounded accuracies. Without censoring the
    replicate contrasts are exact multiples of ``1/n``, so a percentile
    endpoint that lands on an order statistic IS a grid value: the fixture's
    OFF CI reads ``[0.08, 0.22]``, not ``[0.07999999999999996,
    0.21999999999999997]``, and an ON effect of exactly ``44/200`` sits ON
    the inclusive upper edge ("transfers") instead of a hair above it
    ("amplifies"). The RNG draw sequence is unchanged: the same
    ``rng.integers(0, n, size=(m, n))`` calls in the same chunk order."""
    if not 0.0 < ci < 1.0:
        raise PairingError(f"the interval level must be in (0, 1), got {ci!r}")
    if not isinstance(n_resamples, int) or isinstance(n_resamples, bool) or n_resamples < 1:
        raise PairingError(f"the bootstrap needs >= 1 resamples, got {n_resamples!r}")
    if not 0.0 <= max_degenerate_frac < 1.0:
        raise PairingError(f"max_degenerate_frac must be in [0, 1), got {max_degenerate_frac!r}")
    ids = _check_legs(legs)
    n = len(ids)
    masks: Dict[Tuple[str, str], np.ndarray] = {}
    hits: Dict[Tuple[str, str], np.ndarray] = {}
    if 4 * n ** 4 >= 2 ** 53:
        raise PairingError(
            f"{n} items: the exact integer replicate arithmetic (numerators up to 4·n⁴) would leave "
            "float64's exactly-representable range — refusing rather than rounding twice")
    for k in LEG_KEYS:
        m = np.asarray([int(legs[k][i][0]) for i in ids], dtype=np.int64)
        y = np.asarray([int(legs[k][i][1]) if legs[k][i][0] == 1 else 0 for i in ids],
                       dtype=np.int64)          # 0 / 1 exactly: every count below is an integer
        masks[k] = m
        hits[k] = m * y          # correct AND scored
    n_scored = {k: int(masks[k].sum()) for k in LEG_KEYS}
    for k in LEG_KEYS:
        if n_scored[k] == 0:
            raise PairingError(
                f"{leg_label(k)} scored no item — its conditional accuracy is undefined")
    n_correct = {k: int(hits[k].sum()) for k in LEG_KEYS}
    n_censored = {k: n - n_scored[k] for k in LEG_KEYS}

    def contrasts(acc: Mapping[Tuple[str, str], Any], minus: Mapping[Tuple[str, str], Any],
                  plus: Mapping[Tuple[str, str], Any]) -> Dict[str, Tuple[Any, Any, Any]]:
        out: Dict[str, Tuple[Any, Any, Any]] = {}
        for name, signs in CONTRAST_SIGNS.items():
            point = sum(s * acc[k] for k, s in signs.items())
            low = sum((minus[k] if s > 0 else -plus[k]) for k, s in signs.items())
            high = sum((plus[k] if s > 0 else -minus[k]) for k, s in signs.items())
            out[name] = (point, low, high)
        return out

    # -- full-set point estimates and point bounds: EXACT rationals over the
    #    integer counts (the docstring's last paragraph); floats are taken once
    acc_fr = {k: Fraction(n_correct[k], n_scored[k]) for k in LEG_KEYS}
    minus_fr = {k: Fraction(n_correct[k], n) for k in LEG_KEYS}
    plus_fr = {k: Fraction(n_correct[k] + n_censored[k], n) for k in LEG_KEYS}
    point = contrasts(acc_fr, minus_fr, plus_fr)          # Fractions throughout
    acc_pt = {k: float(acc_fr[k]) for k in LEG_KEYS}
    minus_pt = {k: float(minus_fr[k]) for k in LEG_KEYS}
    plus_pt = {k: float(plus_fr[k]) for k in LEG_KEYS}

    # -- replicates: ONE index draw per replicate, applied to all four legs
    rng = np.random.default_rng(seed)
    rep_acc = {k: np.empty(n_resamples) for k in LEG_KEYS}
    rep_stat = {c: np.empty(n_resamples) for c in CONTRASTS}
    rep_low = {c: np.empty(n_resamples) for c in CONTRASTS}
    rep_high = {c: np.empty(n_resamples) for c in CONTRASTS}
    keep = np.ones(n_resamples, dtype=bool)
    chunk = max(1, 4_000_000 // n)
    done = 0
    while done < n_resamples:
        m = min(chunk, n_resamples - done)
        idx = rng.integers(0, n, size=(m, n))
        sl = slice(done, done + m)
        # the replicate's INTEGER counts per leg (int64: exact), then every
        # statistic as ONE division of exact integers (_exact_replicate_contrasts)
        scored_r = {k: masks[k][idx].sum(axis=1) for k in LEG_KEYS}
        correct_r = {k: hits[k][idx].sum(axis=1) for k in LEG_KEYS}
        for k in LEG_KEYS:
            degenerate = scored_r[k] == 0
            keep[sl] &= ~degenerate
            with np.errstate(divide="ignore", invalid="ignore"):
                cond = correct_r[k] / scored_r[k]
            rep_acc[k][sl] = np.where(degenerate, np.nan, cond)
        for c, (pt, lo, hi) in _exact_replicate_contrasts(correct_r, scored_r, n).items():
            rep_stat[c][sl], rep_low[c][sl], rep_high[c][sl] = pt, lo, hi
        done += m
    n_degenerate = int((~keep).sum())
    if n_degenerate > max_degenerate_frac * n_resamples:
        raise PairingError(
            f"{n_degenerate} of {n_resamples} replicates had a leg with no scored item "
            f"(> {max_degenerate_frac:.1%}) — a leg is nearly all censored; the conditional "
            "estimand is not estimable here")

    acc_by_leg: Dict[str, Dict[str, Any]] = {}
    for k in LEG_KEYS:
        lo, hi = _quantiles(rep_acc[k][keep], ci)
        acc_by_leg[leg_label(k)] = {
            "point": acc_pt[k], "ci_low": lo, "ci_high": hi,
            "acc_minus": minus_pt[k], "acc_plus": plus_pt[k],
            "n_correct": n_correct[k], "n_scored": n_scored[k], "n_censored": n_censored[k]}
    stats: Dict[str, Dict[str, Any]] = {}
    for c in CONTRASTS:
        pt, blo, bhi = point[c]                              # exact Fractions
        reps = rep_stat[c][keep]
        lo, hi = _quantiles(reps, ci)
        alpha = 1.0 - ci
        stats[c] = {
            "point": float(pt), "ci_low": lo, "ci_high": hi,
            # where the replicates actually fell. A replicate is ONE
            # exact integer division, so a zero here is an exact zero. With
            # censoring masks resampled and per-replicate denominators, a
            # one-sided complete-case discordance does NOT pin every replicate
            # to one side of zero (only an empirical bootstrap of the complete
            # pairs has that property) — these counts say what happened.
            "replicate_signs": {"negative": int((reps < 0).sum()), "zero": int((reps == 0).sum()),
                                "positive": int((reps > 0).sum())},
            "bound_low": float(blo), "bound_high": float(bhi),
            "bound_ci_low": float(np.quantile(rep_low[c][keep], alpha / 2.0)),
            "bound_ci_high": float(np.quantile(rep_high[c][keep], 1.0 - alpha / 2.0)),
            "excludes_zero": bool(lo > 0.0 or hi < 0.0),
            # the EXACT sign of the rational point: -1 / 0 / +1, zero its own
            # value — what the branch-agreement check compares across branches
            "point_sign": (pt > 0) - (pt < 0),
            # the bound EXCLUDES zero -> the contrast's sign is identified
            # despite the censoring (§6.2's per-shift / DiD sign rule); decided
            # on the exact rationals, so an endpoint that IS zero never reads
            # as a sign from a floating residue
            "sign_identified": bool(blo > 0 or bhi < 0),
            "legs": {leg_label(k): s for k, s in CONTRAST_SIGNS[c].items()},
        }

    # -- complete-case (intersection of scored items), labelled, secondary
    complete: Dict[str, Any] = {"label": COMPLETE_CASE_LABEL}
    for c, signs in CONTRAST_SIGNS.items():
        both = np.ones(n, dtype=bool)
        for k in signs:
            both &= masks[k] == 1
        n_both = int(both.sum())
        if n_both == 0:
            complete[c] = {"n_items": 0, "delta": None,
                           "unavailable": "no item scored in every leg of the contrast"}
            continue
        delta = sum(s * Fraction(int(hits[k][both].sum()), n_both) for k, s in signs.items())
        complete[c] = {"n_items": n_both, "delta": float(delta)}

    return {
        "B": int(n_resamples), "seed": int(seed), "level": float(ci), "n_items": n,
        "n_scored_by_leg": {leg_label(k): n_scored[k] for k in LEG_KEYS},
        "n_censored_by_leg": {leg_label(k): n_censored[k] for k in LEG_KEYS},
        "n_correct_by_leg": {leg_label(k): n_correct[k] for k in LEG_KEYS},
        "n_degenerate_replicates": n_degenerate,
        "n_replicates_used": int(keep.sum()),
        "acc_by_leg": acc_by_leg,
        "stats": stats,
        "complete_case": complete,
    }
