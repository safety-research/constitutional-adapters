"""Exact-three-seed completeness gate for every seed-replicate statistic.

Every object-class statistic
(object_space.py) and every per-leg statistic (seed_spread.py) passes its inputs
through :func:`require_triplet` BEFORE any reduction. The gate

* takes an iterable of ``Record(seed, value, eligible, reason)`` rows (or the
  equivalent ``Mapping[int, Sequence[Record]]``) and COUNTS ROWS PER SEED FIRST — a
  ``dict[seed -> value]`` interface cannot see a duplicate, because the second value
  would silently overwrite the first;
* REFUSES two records for one seed, whatever their insertion order and even when the
  two are identical (a duplicate row is a producer fault; it is never deduplicated);
* REFUSES any seed outside ``{1, 2, 3}`` (the batch-8 seed-0 original is a reference
  column, never a replicate);
* requires the distinct eligible seed set to equal ``{1, 2, 3}`` with exactly ONE
  eligible record per seed — otherwise the statistic is ``unavailable: incomplete
  replicate triplet (missing: s<k> — <reason>)`` and the caller marks the success
  criterion FAILED for that class / leg;
* only then builds the three-value mapping the statistic consumes.

No SD or range is ever computed over the survivors of an incomplete triplet:
:func:`spread` refuses anything but exactly three values.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, NamedTuple, Sequence

SEEDS: tuple[int, ...] = (1, 2, 3)
UNAVAILABLE_PREFIX = "unavailable: incomplete replicate triplet"
REASON_NO_STORE = "no store"
REASON_NOT_ADMITTED = "not admitted"
REASON_NON_COMPARABLE = "non_comparable"
REASON_NOT_ELIGIBLE = "not eligible"


class Record(NamedTuple):
    """One replicate row BEFORE reduction: which seed produced it, the value it
    carries, whether it is admissible for the statistic, and why not if not."""

    seed: int
    value: Any
    eligible: bool = True
    reason: str | None = None


class TripletRefused(Exception):
    """A producer fault in the inputs (duplicate rows, a foreign seed). Never caught
    into an ``unavailable`` message — it propagates so the run stops."""


class DuplicateReplicate(TripletRefused):
    def __init__(self, counts: Mapping[int, int]):
        self.counts = dict(counts)
        dup = ", ".join(f"s{s} ({n} records)" for s, n in sorted(self.counts.items()) if n > 1)
        super().__init__(f"duplicate replicate for {dup} — one row per seed; a duplicate is a "
                         "producer fault and is never deduplicated")


class UnexpectedSeed(TripletRefused):
    def __init__(self, seeds: Iterable[int]):
        self.seeds = sorted(set(seeds))
        super().__init__("unexpected seed " + ", ".join(f"s{s}" for s in self.seeds)
                         + f" — the replicate triplet is {{{', '.join(map(str, SEEDS))}}}; the batch-8 "
                         "seed-0 original is a reference column, never a replicate")


class TripletUnavailable(Exception):
    """The triplet is incomplete: at least one of s1..s3 has no eligible record."""

    def __init__(self, missing: Sequence[tuple[int, str]]):
        self.missing = [(int(s), str(r)) for s, r in missing]
        super().__init__(unavailable_message(self.missing))


def unavailable_message(missing: Sequence[tuple[int, str]]) -> str:
    inner = ", ".join(f"s{s} — {r}" for s, r in sorted(missing))
    return f"{UNAVAILABLE_PREFIX} (missing: {inner})"


@dataclass(frozen=True)
class Triplet:
    """The three admitted values, keyed by seed, plus the records they came from."""

    values: dict[int, Any]
    records: tuple[Record, ...]

    def as_list(self) -> list[Any]:
        return [self.values[s] for s in SEEDS]

    def __getitem__(self, seed: int) -> Any:
        return self.values[int(seed)]


def _coerce(rec: Any, key_seed: int | None = None) -> Record:
    if isinstance(rec, Record):
        out = rec
    elif isinstance(rec, Mapping):
        out = Record(int(rec["seed"]), rec.get("value"), bool(rec.get("eligible", True)), rec.get("reason"))
    elif isinstance(rec, (tuple, list)) and len(rec) in (3, 4):
        seed, value, eligible = rec[0], rec[1], rec[2]
        out = Record(int(seed), value, bool(eligible), rec[3] if len(rec) == 4 else None)
    else:
        raise TypeError(f"not a Record: {rec!r} (want (seed, value, eligible[, reason]))")
    if key_seed is not None and int(out.seed) != int(key_seed):
        raise TripletRefused(f"mapping key s{key_seed} holds a record for s{out.seed} — the record's own seed "
                             "and its key disagree")
    return Record(int(out.seed), out.value, bool(out.eligible), out.reason)


def normalize(records: Iterable[Any] | Mapping[int, Sequence[Any]]) -> list[Record]:
    """Flatten either accepted form into one ordered list of Records, row by row —
    nothing is merged here; counting happens on this list."""
    out: list[Record] = []
    if isinstance(records, Mapping):
        for seed, seq in records.items():
            if isinstance(seq, (str, bytes)) or not isinstance(seq, Iterable):
                raise TypeError(f"mapping value for s{seed} must be a sequence of records, got {type(seq).__name__}")
            if isinstance(seq, Record) or (isinstance(seq, tuple) and len(seq) in (3, 4)
                                           and not isinstance(seq[0], (Record, tuple, list, Mapping))):
                # a bare record where a sequence was expected: still ONE row, never silently two
                seq = [seq]
            for rec in seq:
                out.append(_coerce(rec, key_seed=int(seed)))
    else:
        for rec in records:
            out.append(_coerce(rec))
    return out


def require_triplet(records: Iterable[Any] | Mapping[int, Sequence[Any]]) -> Triplet:
    """The gate. Counts rows per seed BEFORE reducing; refuses duplicates and foreign
    seeds; raises :class:`TripletUnavailable` unless every seed in ``SEEDS`` has
    exactly one eligible record; returns the three-value :class:`Triplet` otherwise."""
    rows = normalize(records)
    counts: dict[int, int] = {}
    for r in rows:
        counts[r.seed] = counts.get(r.seed, 0) + 1
    if any(n > 1 for n in counts.values()):
        raise DuplicateReplicate(counts)
    foreign = [s for s in counts if s not in SEEDS]
    if foreign:
        raise UnexpectedSeed(foreign)
    by_seed = {r.seed: r for r in rows}
    missing: list[tuple[int, str]] = []
    for s in SEEDS:
        r = by_seed.get(s)
        if r is None:
            missing.append((s, REASON_NO_STORE))
        elif not r.eligible:
            missing.append((s, r.reason or REASON_NOT_ELIGIBLE))
    if missing:
        raise TripletUnavailable(missing)
    return Triplet(values={s: by_seed[s].value for s in SEEDS},
                   records=tuple(by_seed[s] for s in SEEDS))


def gate(records: Iterable[Any] | Mapping[int, Sequence[Any]]) -> tuple[Triplet | None, str | None]:
    """``(triplet, None)`` when complete, ``(None, "unavailable: ...")`` when not.
    Refusals (duplicates, foreign seeds) still raise — they are faults, not gaps."""
    try:
        return require_triplet(records), None
    except TripletUnavailable as exc:
        return None, str(exc)


def spread(values: Sequence[float]) -> dict[str, float | int | list[float]]:
    """SD (sample, ddof=1) and range over EXACTLY three values. Anything else refuses:
    a spread over the survivors of an incomplete triplet is the number this module
    exists to prevent."""
    vals = [float(v) for v in values]
    if len(vals) != len(SEEDS):
        raise TripletRefused(f"spread() takes exactly {len(SEEDS)} values (one per seed), got {len(vals)}")
    if any(math.isnan(v) for v in vals):
        raise TripletRefused("spread() refuses NaN — an unreadable value must be an ineligible Record, not a NaN")
    return {"n": len(vals), "values": vals, "mean": statistics.fmean(vals),
            "sd": statistics.stdev(vals), "sd_ddof": 1,
            "range": max(vals) - min(vals), "min": min(vals), "max": max(vals)}


__all__ = ["SEEDS", "UNAVAILABLE_PREFIX", "Record", "Triplet", "TripletRefused", "DuplicateReplicate",
           "UnexpectedSeed", "TripletUnavailable", "require_triplet", "gate", "spread", "normalize",
           "unavailable_message", "REASON_NO_STORE", "REASON_NOT_ADMITTED", "REASON_NON_COMPARABLE"]
