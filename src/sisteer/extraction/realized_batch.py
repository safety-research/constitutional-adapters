"""Validate a fit's REALIZED effective-batch vector against what the trainer
could have run.

``fit_record.realized_eff_batch`` (train_object_ce) is a LIST — one entry
per optimizer step (accumulation group) of the first epoch completed in the
process — not a scalar. The trainer's grouping is exact and seeded:

  * micro-batches are the length-SORTED encoded rows chunked by
    ``batch_size``; the short chunk, if ``r = n_docs mod bs != 0``, is the
    LAST micro-batch (index ``n_micro - 1``);
  * each epoch's ORDER is ``range(n_micro)`` shuffled by the trainer's
    ``random.Random(seed)`` stream, on which no draw precedes the first
    epoch's shuffle (``objects.shuffled_epoch_order``);
  * groups of ``accum_steps`` are taken along that order and never span
    epochs (``objects.group_row_counts``) — so the micro-batch carrying the
    row deficit lands in ANY group, and only the last group is short by
    COUNT.

A rule that expected the deficit in the last group would REJECT a valid run:
at (17,073 docs, bs 2, accum 32) the seed-0 order puts the 63-row group in
the middle of the 267 groups and a 50-row group last.

Two checks, the way the plan states them:

  1. EXACT reproduction when the run was NOT resumed (``resumed_from_step``
     is None): the vector equals ``group_row_counts(micro_rows,
     shuffled_epoch_order(random.Random(seed), n_micro), accum)`` — the
     SAME two helpers the trainer calls.
  2. The SEMANTIC FLOOR, always (and the only check on a resumed run, whose
     vector is the SUFFIX of the resumed epoch under a restored rng state —
     or, when the checkpoint sat on an epoch's LAST group so the resumed
     iteration ran no group, the completed epoch the checkpoint carried:
     ``train_object_ce`` checkpoints its realized-batch accounting and a
     boundary resume inherits it rather than latching an empty vector):
     ``steps_per_epoch == G``; ``len(vec) == G`` (``<= G`` resumed);
     ``sum(vec) == n_docs`` (``<=`` resumed); every entry in the allowed set
     {accum*bs, accum*bs - (bs - r), m_last*bs, m_last*bs - (bs - r)} with
     the deficit terms present only when ``r > 0``; at most ONE group
     carries the deficit; the short-by-count group (``m_last`` micro-
     batches) is the last entry and only the last entry.

The resume marker is PROVENANCE, not an optional field: ``resumed_from_step``
must be present on the record — ``None`` for a fresh process (explicit), an
int for a resumed one. ``validate_fit_record`` reads the published
``fit_record`` and refuses a record with NO marker as unknown provenance —
neither the fresh-run exact check nor the resumed floor can be applied to
a vector whose origin the record does not state (a driver that copied the
vector and steps_per_epoch without the marker left a healthy resumed fit's
suffix judged as a fresh epoch and rejected).

Pure python (no torch): the driver (``train_two_si_objects`` records the
verdict beside the vector as ``fit_record.realized_batch_check``), the
acceptance script and the tests import it directly. Nothing here changes
training; tails are never dropped to make a scalar come out.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .objects import group_row_counts, shuffled_epoch_order

__all__ = ["RealizedBatchVerdict", "micro_rows", "expected_realized_vector",
           "allowed_group_sizes", "validate_realized_vector",
           "validate_fit_record", "RESUME_MARKER", "RESUME_PROVENANCE"]

#: The fit_record key carrying the trainer's resume marker (objects.py's
#: ``train_object_ce`` diagnostics: None = fresh process, int = the
#: optimizer step a checkpoint restored the fit to).
RESUME_MARKER = "resumed_from_step"

#: The three provenance states a record can be in, as the verdict files them.
RESUME_PROVENANCE = ("fresh", "resumed", "absent")


def micro_rows(n_docs: int, bs: int) -> list[int]:
    """Rows per micro-batch in length-sorted chunk order: ``bs`` for every
    chunk but the last, which carries ``n_docs mod bs`` rows when that is
    non-zero (else ``bs``)."""
    n_docs, bs = int(n_docs), int(bs)
    if n_docs < 1 or bs < 1:
        raise ValueError(f"n_docs={n_docs}, bs={bs}: both must be >= 1")
    n_micro = math.ceil(n_docs / bs)
    r = n_docs % bs
    return [bs] * (n_micro - 1) + [r or bs]


def expected_realized_vector(n_docs: int, bs: int, accum: int,
                             seed: int = 0) -> list[int]:
    """The exact per-group row vector an UNRESUMED train_object_ce run
    records for its first epoch: the trainer's own two helpers on a fresh
    ``random.Random(seed)``."""
    rows = micro_rows(n_docs, bs)
    order = shuffled_epoch_order(random.Random(int(seed)), len(rows))
    return group_row_counts(rows, order, int(accum))


def allowed_group_sizes(n_docs: int, bs: int, accum: int) -> dict:
    """The semantic-floor arithmetic: ``n_micro``, ``G`` (groups per epoch),
    ``m_last`` (micro-batches in the last group), ``r`` (the row deficit's
    complement), and the allowed entry sets for a full group and for the
    last group, with the deficit variants only when ``r > 0``."""
    n_docs, bs, accum = int(n_docs), int(bs), int(accum)
    if accum < 1:
        raise ValueError(f"accum={accum} must be >= 1")
    rows = micro_rows(n_docs, bs)
    n_micro = len(rows)
    r = n_docs % bs
    G = math.ceil(n_micro / accum)
    m_last = n_micro - (G - 1) * accum
    full = {accum * bs}
    last = {m_last * bs}
    if r > 0:
        full.add(accum * bs - (bs - r))
        last.add(m_last * bs - (bs - r))
    return {"n_micro": n_micro, "G": G, "m_last": m_last, "r": r,
            "full": full, "last": last, "deficit": (bs - r) if r > 0 else 0}


@dataclass
class RealizedBatchVerdict:
    ok: bool
    problems: list[str] = field(default_factory=list)
    exact_checked: bool = False
    exact_ok: bool | None = None
    expected: list[int] | None = None
    n_docs: int = 0
    bs: int = 0
    accum: int = 0
    seed: int = 0
    G: int = 0
    m_last: int = 0
    deficit_positions: list[int] = field(default_factory=list)
    #: "fresh" (explicit None marker), "resumed" (an int), or "absent" (no
    #: marker on the record — unknown provenance, a problem in itself).
    resume_provenance: str = "fresh"

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        return d


def validate_realized_vector(vec, *, n_docs: int, bs: int, accum: int,
                             seed: int = 0, steps_per_epoch: int | None,
                             resumed_from_step: int | None
                             ) -> RealizedBatchVerdict:
    """Validate ``fit_record.realized_eff_batch`` (``vec``) for a fit of
    ``n_docs`` documents at micro-batch ``bs`` x accumulation ``accum``
    under trainer ``seed``; ``steps_per_epoch`` / ``resumed_from_step`` are
    the record's own fields (``resumed_from_step`` None = a fresh process,
    whose first epoch is exactly reproducible; an int = a resumed process,
    whose vector is the resumed epoch's suffix and gets the floor only).

    Never raises on a bad vector — the verdict carries every problem so the
    acceptance line can print them all; raises only on nonsensical inputs.
    """
    vec = [int(v) for v in vec]
    sizes = allowed_group_sizes(n_docs, bs, accum)
    G, m_last = sizes["G"], sizes["m_last"]
    v = RealizedBatchVerdict(ok=True, n_docs=int(n_docs), bs=int(bs),
                             accum=int(accum), seed=int(seed), G=G,
                             m_last=m_last)
    problems = v.problems
    resumed = resumed_from_step is not None

    # --- the semantic floor, always ------------------------------------------
    if steps_per_epoch is not None and int(steps_per_epoch) != G:
        problems.append(f"steps_per_epoch {steps_per_epoch} != ceil(n_micro "
                        f"{sizes['n_micro']} / accum {accum}) = {G}")
    if not vec:
        problems.append("empty realized_eff_batch vector")
    if resumed:
        if len(vec) > G:
            problems.append(f"len {len(vec)} > G {G} on a resumed run")
        if sum(vec) > int(n_docs):
            problems.append(f"sum {sum(vec)} > n_docs {n_docs} on a resumed "
                            "run")
    else:
        if len(vec) != G:
            problems.append(f"len {len(vec)} != G {G}")
        if sum(vec) != int(n_docs):
            problems.append(f"sum {sum(vec)} != n_docs {n_docs}")
    if vec:
        head, tail = vec[:-1], vec[-1]
        bad_head = [(i, x) for i, x in enumerate(head)
                    if x not in sizes["full"]]
        if bad_head:
            problems.append(
                f"non-last group(s) outside the allowed sizes "
                f"{sorted(sizes['full'])}: {bad_head[:5]}"
                + (" ..." if len(bad_head) > 5 else ""))
        # The last ENTRY is the short-by-count group only when the vector
        # runs to the epoch's end (unresumed, or a resumed suffix that
        # reached it); the allowed set for it is the last-group set.
        if tail not in sizes["last"] and tail not in sizes["full"]:
            problems.append(f"last group {tail} outside the allowed sizes "
                            f"{sorted(sizes['last'] | sizes['full'])}")
        elif not resumed and tail not in sizes["last"]:
            problems.append(f"last group {tail} is not a last-group size "
                            f"{sorted(sizes['last'])} (m_last={m_last})")
        deficit = sizes["deficit"]
        if deficit:
            short_full = sizes["full"] - {accum * bs}
            short_last = sizes["last"] - {m_last * bs}
            positions = [i for i, x in enumerate(head) if x in short_full]
            if tail in short_last:
                positions.append(len(vec) - 1)
            v.deficit_positions = positions
            if len(positions) > 1:
                problems.append(f"the row deficit ({deficit}) appears in "
                                f"{len(positions)} groups {positions}; at "
                                "most ONE group may carry it")
            if not resumed and len(positions) != 1:
                problems.append(f"an unresumed epoch must carry the row "
                                f"deficit ({deficit}) in exactly one group; "
                                f"found {len(positions)}")
    # --- exact reproduction when unresumed -----------------------------------
    if not resumed:
        v.exact_checked = True
        v.expected = expected_realized_vector(n_docs, bs, accum, seed)
        v.exact_ok = (vec == v.expected)
        if not v.exact_ok:
            first = next((i for i, (a, b) in enumerate(zip(vec, v.expected))
                          if a != b), min(len(vec), len(v.expected)))
            problems.append(
                f"vector is not the seed-{seed} reproduction (first "
                f"difference at group {first}: got "
                f"{vec[first] if first < len(vec) else None}, expected "
                f"{v.expected[first] if first < len(v.expected) else None})")
    v.ok = not problems
    return v


def validate_fit_record(record: dict, *, n_docs: int, bs: int, accum: int,
                        seed: int = 0) -> RealizedBatchVerdict:
    """``validate_realized_vector`` over a published ``fit_record`` (the
    trainer's diagnostics as the driver copies them into
    ``train_budget_meta.json``): the vector from ``realized_eff_batch``, the
    record's own ``steps_per_epoch``, and its resume PROVENANCE from
    ``resumed_from_step`` — ``None`` is an explicit fresh run (exact
    reproduction applies), an int a resumed one (the floor only), and an
    ABSENT key is unknown provenance: refused by name, with neither
    verdict applied (a resumed suffix judged as a fresh epoch fails on
    length, sum and reproduction; a fresh vector waved through on an
    assumption proves nothing). A record with no vector is refused too."""
    if not isinstance(record, dict):
        raise TypeError(f"fit_record must be a dict, got {type(record).__name__}")
    sizes = allowed_group_sizes(n_docs, bs, accum)
    vec = record.get("realized_eff_batch")
    spe = record.get("steps_per_epoch")
    if RESUME_MARKER not in record:
        v = RealizedBatchVerdict(ok=False, n_docs=int(n_docs), bs=int(bs),
                                 accum=int(accum), seed=int(seed),
                                 G=sizes["G"], m_last=sizes["m_last"],
                                 resume_provenance="absent")
        v.problems.append(
            f"fit_record carries no {RESUME_MARKER!r} — resume provenance "
            "unknown (a record written before the marker was copied, or a "
            "driver that dropped it): a fresh epoch and a resumed suffix "
            "need different checks, so neither is applied")
        if not isinstance(vec, list):
            v.problems.append("fit_record carries no realized_eff_batch vector")
        if spe is not None and int(spe) != sizes["G"]:
            v.problems.append(f"steps_per_epoch {spe} != ceil(n_micro "
                              f"{sizes['n_micro']} / accum {accum}) = {sizes['G']}")
        return v
    marker = record[RESUME_MARKER]
    if not isinstance(vec, list):
        v = RealizedBatchVerdict(ok=False, n_docs=int(n_docs), bs=int(bs),
                                 accum=int(accum), seed=int(seed),
                                 G=sizes["G"], m_last=sizes["m_last"],
                                 resume_provenance=("fresh" if marker is None
                                                    else "resumed"))
        v.problems.append("fit_record carries no realized_eff_batch vector")
        return v
    v = validate_realized_vector(
        vec, n_docs=n_docs, bs=bs, accum=accum, seed=seed,
        steps_per_epoch=(None if spe is None else int(spe)),
        resumed_from_step=(None if marker is None else int(marker)))
    v.resume_provenance = "fresh" if marker is None else "resumed"
    return v
