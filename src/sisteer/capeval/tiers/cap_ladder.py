"""The standing truncation rule of the standard eval harness: CHECK for
truncation after every generation pass, SIZE the cap appropriately at the
outset from the measured length distributions, and RERUN the truncated
responses of a (cell, readout) whose at-cap share exceeds 1% — climbing a
cap ladder until the share is at or below 1% or the 16,384 ceiling is
reached — then REPORT the residual at exit, naming every threshold exit
that still carries a non-zero residual.

Both generating paths use this module — ``scripts/cell_controls_run.py``
(HF in-process) and ``scripts/served_capability_run.py`` (served) measure
each judged control's spool right after the pass and write the per-item
sidecar + the rung record; the grading host plan (``scripts/capctrl_cap2048_plan.py``,
the rung driver, ``--cap``/``--new-cap`` pick the rung) reads the record and
builds the next rung's carry-over bundle; ``scripts/cap_ladder.py`` is the
CLI (``size`` before a run, ``measure`` over a store, ``report`` over the
rungs).

THE AT-CAP TEST IS THE ONE ALREADY IN THE HARNESS. ``carry_over.AT_CAP_RULE``
— ``min(ntok(text) + 1, cap) >= cap`` under the cell's own tokenizer, the
geometry measurement's convention — is imported, never restated
(:func:`at_cap.is_at_cap`, :func:`at_cap.completion_tokens`: the rule's one
stdlib-only home, so this module never pays ``carry_over``'s torch chain —
``ladder_gate.py`` imports it per gate call).

TWO ADDITIONS THE RULE CARRIES, both from measurements,
neither optional:

1. ESCALATE ONLY TRUNCATED ITEMS, NEVER DEGENERATE ONES. A larger cap fixes
   a response that was cut off; it does nothing for one that never
   terminates, and at the 16,384 ceiling it is sixteen times the tokens to
   produce sixteen times the loop. Every at-cap item is classified BEFORE the
   climb by a repetition test — :func:`rep_ratio`, the whitespace-token
   4-gram repetition ratio the peer lane uses as its loop rule
   (``full22_join_scores.rep_ratio``, source sha 152d180376dd, copied
   byte-for-byte below), over the WHOLE text OR over the last
   :data:`TAIL_WORDS` words (the tail is where a late-onset loop lives; for a
   text shorter than the window the two readings coincide, so everything the
   peer calls a loop is a loop here too). Degenerate items are excluded from
   the escalation SET and from the trigger, and reported separately as
   ``non_terminating`` — never as unresolved truncation. Over-refusal at-cap
   measured zero degenerate; a prior single-shot jailbreak regeneration at
   4096 left a residue that was pure decode loop.

2. EXCLUDE BOUNDARY-DRAW ITEMS FROM THE ESCALATION TRIGGER. Greedy decoding
   on this project's HF path is not reproducible (17.9% of below-cap items
   differ between two runs; seven items at cap on one draw terminated well
   under it on the next — memory note ``text-diff-is-a-hypersensitive-
   readout``). An at-cap item for which ANOTHER draw in hand terminated
   below its cap is boundary noise, not truncation, and against a 1%
   threshold a handful can escalate a cell that did not need it. Such items
   are out of the trigger's numerator (``n_truncated``); if the cell climbs
   anyway they ARE redrawn (an at-cap text is still at cap, and the redraw
   is what resolves it), which is why the exclusion is from the TRIGGER and
   not from the regenerate set. The evidence has to be a real second draw —
   the parity spool, a pilot at another cap, a duplicate run — so on the
   ordinary single-draw chain the flag is retrospective: the next rung's
   redraw of an at-cap item that terminates below the source cap is that
   item's ``boundary_landing`` (the peer's term and field, matched here),
   recorded per item and counted per rung.

NOTHING HERE ASSUMES A REDRAW REPRODUCES A PRIOR TEXT. No byte-identity is
asserted anywhere; identity / prefix-extension / fork are RECORDED per
regenerated item (the peer's ``identical_text`` / ``prefix_extension`` /
``first_divergence_*`` fields) because they are the noise floor a label-flip
analysis reads against, not because any of them is expected.

THE LADDER. Rungs double from 512 to the ceiling (:data:`CAP_LADDER`). A
(cell, readout) climbs from rung R to the next rung when its TRUNCATED share
at R exceeds :data:`AT_CAP_SHARE_THRESHOLD` and R is below
:data:`CAP_CEILING`; every rung is its own store at its own cap (the
existing carry-over lane's shape, and the peer's), never an in-place
rewrite. A climb redraws the truncated items (and the boundary-flagged
ones), carries every below-cap text unchanged, carries the non-terminating
texts unchanged and explicitly, and — when the readout does NOT climb —
carries its at-cap truncated texts unchanged too, as the ``residual`` the
report names. Exits (:data:`EXITS`): ``clean`` (nothing truncated),
``below_threshold`` (0 < share <= 1%: a residual the report lists),
``ceiling`` (share > 1% at the ceiling AFTER CLIMBING: the ladder ran out),
``ceiling_at_outset`` (share > 1% at a cap that was already at or above the
ceiling when the run started: there never was a rung above it, so the rule
reduces to measure, classify, report — MATH-500's default cap is the
ceiling, so every MATH-500 reading is this case and must never be read as
an exhausted ladder), ``no_rung`` (share > 1% below the ceiling on a leg
that carries no rung mechanism — the Inspect tasks, whose own one-step
truncation retry is recorded per row but which cannot be climbed by a
carry-over bundle; the residual is unresolved and the report says so). The
two ceiling exits look identical in the counts and mean different things,
which is why they are two verdicts.

NO PER-TASK CAP IS HELD HERE. Every cap this module reads comes from the
run: a spool's header / ``JudgedSpec.gen["max_new_tokens"]`` (the judged
controls, from ``cell_controls.controls_gen_config``), an Inspect row's own
``max_tokens_used`` (the Inspect tasks; their default lives in
``tinycap.contracts.MATH500_MAX_TOKENS_DEFAULT`` and reaches the rows through
``inspect_served.MAX_TOKENS_OFF/ON`` — read there, copied nowhere, because
it is moving on another branch), and a measurement's own meta note (the
clamp cap of a ``cap__*.npy``). The ladder's rungs and ceiling are the
RULE's constants, not a task's.

SIZING AT THE OUTSET. The measured length distributions
(``~/evals-geometry-measure/raw/cap__<family>__<task>.npy``, int64 [n, 2] =
[prompt, completion], completions counted ``min(ntok + 1, cap)`` and so
CENSORED at the cap they were generated under) give the smallest rung whose
at-cap share would be at or below the threshold. When the measured share at
the measured cap already exceeds the threshold the distribution is censored
above it and the p99 is unreadable: the recommendation is then the next
rung above the measured cap, flagged ``censored`` — a lower bound the
ladder settles, never a guess at the p99.

THE PEER'S LADDER (the sampled re-run lane) now exits on the same rule —
``exit_le_1pct`` on the TRUNCATED share, with loops, repetitive tails and
boundary draws set aside in a per-rung ``residual_composition`` — on rungs
4096 / 8192 / 16384 (static readouts) and 512 / 2048 / 8192 / 16384
(controls), to the same 16,384 ceiling. What still differs: their lane
samples with seeds and re-draws a 200-item below-cap floor sample per rung
(ours is greedy and carries the existing parity control instead). Its
``length_quantiles.json`` report is a sizing source here
(:func:`size_cap_from_length_quantiles`): the first UNCENSORED length
distributions the outset half has had, read with the ladder's own
classification of the residual so the sizing counts truncation the way the
check half does. Their per-row fields — ``regenerated``, ``source_cap``,
``boundary_landing``, ``identical_text``, ``prefix_extension``,
``first_divergence_char`` / ``first_divergence_token``,
``source_completion_tokens`` — are matched by name and meaning in the
sidecar rows (:data:`ROW_KEYS`, :data:`REDRAW_KEYS`).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ...chat_markers import divergence
from .at_cap import completion_tokens, is_at_cap, ntok   # is_at_cap re-exported: main's inspect_rung tests read L.is_at_cap (2026-09-12)

def _first_divergence(a: Sequence, b: Sequence) -> int:
    """Index of the first differing element (min length if one is a prefix) — the name main's consumers bind
    (``inspect_rung._first_divergence = L._first_divergence``, 2026-09-12); this tree's own callers use
    :func:`sisteer.chat_markers.divergence` directly (the 2026-09-10 simplification replaced the private copy), and
    this alias keeps the two trees import-compatible with ONE implementation."""
    return divergence(a, b)


#: the rungs: doubling from the legacy 512 to the 16,384 ceiling
CAP_LADDER: Tuple[int, ...] = (512, 1024, 2048, 4096, 8192, 16384)
CAP_CEILING = 16384
#: a (cell, readout) climbs when its TRUNCATED share exceeds this
AT_CAP_SHARE_THRESHOLD = 0.01
#: the loop rule — the peer lane's, byte for byte (see :func:`rep_ratio`)
REP_RATIO_THRESHOLD = 0.5
REP_NGRAM = 4
REP_MIN_WORDS = 8
#: the tail window (whitespace tokens) the loop rule is ALSO read over
TAIL_WORDS = 200
DEGENERATE_RULE = (
    f"rep_ratio(text) > {REP_RATIO_THRESHOLD} or "
    f"rep_ratio(last {TAIL_WORDS} whitespace tokens) > {REP_RATIO_THRESHOLD}; "
    f"rep_ratio = 1 - unique/total whitespace-token {REP_NGRAM}-grams "
    f"(< {REP_MIN_WORDS} words -> 0)")
TRIGGER_RULE = (
    f"escalate iff n_truncated / n > {AT_CAP_SHARE_THRESHOLD} and cap < {CAP_CEILING}; "
    "truncated = at cap under the item's source cap, not degenerate, and no other "
    "draw of the item in hand terminated below that cap (boundary)")
#: THE TRANSPORT GUARD. A rung is only worth climbing to if the serving
#: transport can carry a generation that long. Two hops sit between the
#: driver and the engine (driver -> proxy, proxy -> engine), each with its
#: own total timeout, and the SMALLER binds. A hop that times out does not
#: surface as a failure: the driver sees a short completion, the store says
#: zero rejections, the ladder reads the truncated share as FALLING and exits
#: satisfied while the long draws are being killed in transit (the 600 s
#: default under a 7,200 s driver did exactly this — see
#: memory/proxy-timeout-censors-the-slow-leg). So the check runs BEFORE an
#: escalation is issued, refuses rather than warns, and treats an
#: undeterminable timeout as a refusal: absent evidence is not adequacy.
#: The floor decode rate is the SLOW case — a steered served target under
#: batching can fall to a few tokens per second per sequence — and the
#: margin covers queueing and prefill; both are overridable per call.
TRANSPORT_FLOOR_TOK_PER_S = 4.0
TRANSPORT_MARGIN = 1.5
TRANSPORT_RULE = (
    f"an escalation to cap C is refused unless min(hop timeouts) >= "
    f"C / floor_tok_per_s * margin (floor {TRANSPORT_FLOOR_TOK_PER_S} tok/s, "
    f"margin {TRANSPORT_MARGIN}); no hop timeouts supplied (None) refuses as "
    "undeterminable; an empty mapping means an in-process leg with no transport "
    "hop and passes")
#: schema literals
RECORD_SCHEMA = "cap_ladder/1"
SIZING_SCHEMA = "cap_sizing/1"
#: the per-cell record beside manifest.json, and its manifest key
RECORD_NAME = "cap_ladder.json"
MANIFEST_KEY = "cap_ladder"
MANIFEST_SIZING_KEY = "cap_sizing"
#: the per-item sidecar beside its spool: gens_<name>.jsonl -> gens_<name>.caps.jsonl
SIDECAR_SUFFIX = ".caps.jsonl"

#: at-cap kinds
KIND_TRUNCATED = "truncated"
KIND_NON_TERMINATING = "non_terminating"
KIND_BOUNDARY = "boundary"
KINDS = (KIND_TRUNCATED, KIND_NON_TERMINATING, KIND_BOUNDARY)
#: rung verdicts
EXIT_CLEAN = "clean"
EXIT_BELOW_THRESHOLD = "below_threshold"
EXIT_CEILING = "ceiling"                    # climbed, ran out
EXIT_CEILING_AT_OUTSET = "ceiling_at_outset"  # started at/above the ceiling: no rung ever existed
EXIT_NO_RUNG = "no_rung"                    # a leg with no rung mechanism (Inspect); unresolved
EXIT_TRANSPORT_REFUSED = "transport_refused"  # a rung exists but the transport cannot carry it
EXIT_LEDGER_CENSORED = "ledger_censored"      # the producer's ledger holds at-cap completions the scored spool lacks
ESCALATE = "escalate"
EXITS = (EXIT_CLEAN, EXIT_BELOW_THRESHOLD, EXIT_CEILING, EXIT_CEILING_AT_OUTSET, EXIT_NO_RUNG,
         EXIT_TRANSPORT_REFUSED, EXIT_LEDGER_CENSORED)
#: the verdicts that leave a residual nothing in the ladder can resolve
UNRESOLVED_EXITS = (EXIT_CEILING, EXIT_CEILING_AT_OUTSET, EXIT_NO_RUNG, EXIT_TRANSPORT_REFUSED,
                    EXIT_LEDGER_CENSORED)
#: THE PRECONDITION the trigger rests on, and how it is checked (2026-09-11, after a peer
#: lane found a client timeout set below the server's completion time ABANDONING the
#: slowest draws while the server finished them: the scored sample held ZERO at-cap draws
#: where the request ledger held 254 and 597 — at-cap draws censored out structurally,
#: so the 1 % trigger read the most truncated cell as the cleanest). The at-cap share is
#: read from the SCORED spool; that spool is a faithful sample of what the producer
#: generated only when every completion the producer finished is in it. The producer's
#: usage ledger (``usage_ledger.jsonl``: one row per ATTEMPTED completion, written when
#: the server answers, ``finish_reason == "length"`` = at cap) is therefore RECONCILED
#: against the spool per (cell route, cap): more at-cap completions in the ledger than in
#: the spool means the transport censored the scored set, and the rung exits
#: ``ledger_censored`` — unresolved, never ``below_threshold``. Without a ledger the
#: reading is recorded as UNRECONCILED, and the report says so.
LEDGER_RULE = ("at-cap share from the SCORED spool, reconciled against the producer's usage ledger "
               "(usage_ledger.jsonl rows for the cell's virtual model at this cap: served = not "
               "rejected, at cap = finish_reason 'length'); ledger at-cap > spool at-cap => the "
               "transport censored the scored set => exit ledger_censored (unresolved); no ledger => "
               "unreconciled, stated")
LEDGER_FINISH_AT_CAP = "length"
LEDGER_ERROR_CLASS_TRANSPORT = "transport"
LEDGER_REQUIRED_FIELDS = ("vmodel", "max_tokens_requested", "rejected", "finish_reason")
#: redraw outcomes (a regenerated item against its source draw)
REDRAW_STILL_AT_CAP = "still_at_cap"
REDRAW_EXTENDED = "extended"
REDRAW_BOUNDARY_LANDING = "boundary_landing"
REDRAW_OUTCOMES = (REDRAW_STILL_AT_CAP, REDRAW_EXTENDED, REDRAW_BOUNDARY_LANDING)

#: every sidecar row's keys — ids and numbers only, never text (the text is
#: bound by its sha256, the peer's convention)
ROW_KEYS = ("item_id", "text_sha256", "source_cap", "ntok", "completion_tokens",
            "at_cap", "degenerate", "rep_ratio", "tail_rep_ratio", "kind",
            "boundary_from", "regenerated", "redraw")
#: the redraw block a REGENERATED row carries (the peer's names where they
#: exist; ``outcome`` / ``new_completion_tokens`` / ``still_at_cap`` /
#: ``extended`` are ours)
REDRAW_KEYS = ("source_cap", "new_cap", "source_completion_tokens",
               "new_completion_tokens", "identical_text", "prefix_extension",
               "first_divergence_char", "first_divergence_token",
               "boundary_landing", "still_at_cap", "extended", "outcome")

#: the geometry measurement's raw arrays and the task names it uses
DEFAULT_RAW_DIR = Path(os.path.expanduser("~/evals-geometry-measure/raw"))
#: judged control (as spelled on --controls) -> the measurement's task name
MEASURED_TASK_OF = {"xstest": "xstest_v2", "or_bench_hard_1k": "or_bench_hard_1k"}


class CapLadderError(RuntimeError):
    """A rule violation — loud, never a silent carry or a guessed p99."""


# --------------------------------------------------------------------------- #
# the loop rule                                                                #
# --------------------------------------------------------------------------- #


def rep_ratio(t):
    """The peer lane's loop statistic, copied byte for byte from
    ``full22_join_scores.rep_ratio`` (source sha 152d180376dd): 1 minus the
    unique/total ratio of whitespace-token 4-grams; 0.0 under 8 words."""
    w = t.split()
    if len(w) < 8:
        return 0.0
    g = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return 1 - len(set(g)) / len(g)


def tail_rep_ratio(text: str, words: int = TAIL_WORDS) -> float:
    """:func:`rep_ratio` over the last ``words`` whitespace tokens (the whole
    text when it is shorter — the two readings then coincide)."""
    w = (text or "").split()
    return rep_ratio(" ".join(w[-int(words):])) if len(w) > int(words) else rep_ratio(text or "")


def degeneracy(text: str) -> Dict[str, Any]:
    """The per-text reading of :data:`DEGENERATE_RULE`."""
    rr = rep_ratio(text or "")
    tr = tail_rep_ratio(text or "")
    return {"rep_ratio": round(rr, 4), "tail_rep_ratio": round(tr, 4),
            "degenerate": bool(rr > REP_RATIO_THRESHOLD or tr > REP_RATIO_THRESHOLD)}


def is_degenerate(text: str) -> bool:
    return degeneracy(text)["degenerate"]


# --------------------------------------------------------------------------- #
# the ladder                                                                   #
# --------------------------------------------------------------------------- #


def next_rung(cap: int, ladder: Sequence[int] = CAP_LADDER,
              ceiling: int = CAP_CEILING) -> Optional[int]:
    """The smallest rung strictly above ``cap`` and at or below the ceiling;
    ``None`` at (or past) the ceiling."""
    above = [r for r in sorted(int(x) for x in ladder) if r > int(cap) and r <= int(ceiling)]
    return above[0] if above else None


def text_sha256(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def terminated_below(tok, text: str, draw_cap: int, below: int) -> bool:
    """Did a draw generated under ``draw_cap`` TERMINATE (not at its own cap)
    with fewer than ``below`` completion tokens? The boundary evidence."""
    ct = completion_tokens(tok, text, draw_cap)
    return ct < int(draw_cap) and ct < int(below)   # terminated (not at_cap.is_at_cap) AND under the boundary


@dataclass
class ItemRecord:
    """One item's reading at a rung (the sidecar row, :data:`ROW_KEYS`)."""

    item_id: str
    text_sha256: str
    source_cap: int                      # the cap of the request that produced the text
    ntok: int                            # re-tokenized completion length
    completion_tokens: int               # min(ntok + 1, source_cap)
    at_cap: bool
    degenerate: bool
    rep_ratio: float
    tail_rep_ratio: float
    kind: Optional[str] = None           # None below cap; else one of KINDS
    boundary_from: Optional[int] = None  # the cap of the other draw that terminated below source_cap
    regenerated: bool = False            # this store drew the text at its own cap after a lower rung
    redraw: Optional[Dict[str, Any]] = None   # REDRAW_KEYS for a regenerated item

    def as_row(self) -> Dict[str, Any]:
        row = asdict(self)
        assert tuple(row) == ROW_KEYS, tuple(row)
        return row


def classify_items(entries: Mapping[str, str], tok, *, cap: int,
                   source_caps: Optional[Mapping[str, int]] = None,
                   other_draws: Optional[Mapping[str, Sequence[Tuple[int, str]]]] = None,
                   regenerated: Iterable[str] = (),
                   source_texts: Optional[Mapping[str, str]] = None,
                   source_caps_of_regenerated: Optional[Mapping[str, int]] = None,
                   ) -> Dict[str, ItemRecord]:
    """Every item's reading. ``cap`` is the store's cap (a fresh draw's
    source cap); ``source_caps`` overrides it per item (a carried text keeps
    the rung that produced it); ``other_draws`` = ``{item_id: [(draw_cap,
    text), ...]}`` — other draws of the same item in hand, the boundary
    evidence; ``regenerated`` ids get a redraw block against
    ``source_texts[item_id]`` (their at-cap text at
    ``source_caps_of_regenerated[item_id]``, default ``cap``'s lower rung is
    unknown so it must be given)."""
    source_caps = dict(source_caps or {})
    other = dict(other_draws or {})
    regen = set(regenerated)
    out: Dict[str, ItemRecord] = {}
    for iid in sorted(entries):
        text = entries[iid] or ""
        scap = int(source_caps.get(iid, cap))
        n = ntok(tok, text)
        ct = min(n + 1, scap)
        at = ct >= scap
        deg = degeneracy(text)
        kind = None
        boundary_from = None
        if at:
            if deg["degenerate"]:
                kind = KIND_NON_TERMINATING
            else:
                kind = KIND_TRUNCATED
                for d_cap, d_text in other.get(iid, ()):
                    if terminated_below(tok, d_text or "", int(d_cap), scap):
                        kind = KIND_BOUNDARY
                        boundary_from = int(d_cap)
                        break
        rec = ItemRecord(item_id=iid, text_sha256=text_sha256(text), source_cap=scap,
                         ntok=n, completion_tokens=ct, at_cap=at,
                         degenerate=deg["degenerate"], rep_ratio=deg["rep_ratio"],
                         tail_rep_ratio=deg["tail_rep_ratio"], kind=kind,
                         boundary_from=boundary_from, regenerated=iid in regen)
        if iid in regen:
            if source_texts is None or iid not in source_texts:
                raise CapLadderError(
                    f"item {iid!r} is marked regenerated but no source text was given — "
                    "the redraw record needs the at-cap text it replaced")
            if source_caps_of_regenerated is None or iid not in source_caps_of_regenerated:
                raise CapLadderError(
                    f"item {iid!r} is marked regenerated but its source cap was not given")
            rec.redraw = resolve_redraw(source_texts[iid], text, tok,
                                        source_cap=int(source_caps_of_regenerated[iid]),
                                        new_cap=scap)
        out[iid] = rec
    return out


@dataclass
class RungStatus:
    """One (cell, readout) at one rung: the counts, the trigger, the verdict."""

    cap: int
    n: int
    n_at_cap: int
    n_truncated: int
    n_non_terminating: int
    n_boundary: int
    share_at_cap: float
    share_truncated: float
    threshold: float
    ceiling: int
    escalate: bool
    next_cap: Optional[int]
    at_ceiling: bool
    exit: str                                   # one of EXITS, or ESCALATE
    climbed: bool = False                       # this rung was reached from a lower one
    ladder_available: bool = True               # False: a leg with no rung mechanism
    truncated_ids: List[str] = field(default_factory=list)
    non_terminating_ids: List[str] = field(default_factory=list)
    boundary_ids: List[str] = field(default_factory=list)
    source_caps_seen: Dict[str, int] = field(default_factory=dict)   # str(cap) -> n items
    n_regenerated: int = 0
    redraw_outcomes: Dict[str, int] = field(default_factory=dict)    # outcome -> n
    n_fork: int = 0
    n_identical_text: int = 0
    n_prefix_extension: int = 0
    transport: Optional[Dict[str, Any]] = None  # TransportVerdict.as_dict() when the check ran
    ledger: Optional[Dict[str, Any]] = None     # LedgerReconciliation.as_dict() when reconciled; None = UNRECONCILED
    rules: Dict[str, str] = field(default_factory=lambda: {
        "at_cap": "carry_over.AT_CAP_RULE", "degenerate": DEGENERATE_RULE,
        "trigger": TRIGGER_RULE, "transport": TRANSPORT_RULE, "ledger": LEDGER_RULE})

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TransportVerdict:
    """Can the transport carry a generation of ``next_cap`` tokens?"""
    ok: bool
    next_cap: int
    required_s: Optional[float]
    binding_hop: Optional[str]
    binding_s: Optional[float]
    hops: Dict[str, Optional[float]]
    floor_tok_per_s: float
    margin: float
    reason: str

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def transport_required_s(next_cap: int, *, floor_tok_per_s: float = TRANSPORT_FLOOR_TOK_PER_S,
                         margin: float = TRANSPORT_MARGIN) -> float:
    """Seconds a hop must allow for a ``next_cap``-token draw at the floor rate."""
    if floor_tok_per_s <= 0:
        raise CapLadderError(f"floor_tok_per_s must be > 0, got {floor_tok_per_s!r}")
    return float(next_cap) / float(floor_tok_per_s) * float(margin)


def transport_check(next_cap: int, hop_timeouts_s: Optional[Mapping[str, Optional[float]]], *,
                    floor_tok_per_s: float = TRANSPORT_FLOOR_TOK_PER_S,
                    margin: float = TRANSPORT_MARGIN) -> TransportVerdict:
    """:data:`TRANSPORT_RULE`. ``None`` = undeterminable (refuse); ``{}`` = an
    in-process leg with no transport hop (pass); any hop whose value is
    ``None`` refuses naming that hop; otherwise the SMALLEST hop binds.

    Callable on its own: an instrument with its own escalation trigger (see
    :func:`verdict_for`) runs this check directly before requesting a rung, so a
    trigger that is not the standing rule's still cannot request a rung the hops
    cannot carry.
    """
    hops: Dict[str, Optional[float]] = ({} if hop_timeouts_s is None
                                        else {str(k): (None if v is None else float(v))
                                              for k, v in hop_timeouts_s.items()})
    base = dict(next_cap=int(next_cap), hops=hops, floor_tok_per_s=float(floor_tok_per_s),
                margin=float(margin))
    if hop_timeouts_s is None:
        return TransportVerdict(ok=False, required_s=None, binding_hop=None, binding_s=None,
                                reason="undeterminable: no hop timeouts supplied — a rung is "
                                       "not climbed on absent evidence; pass hop_timeouts_s "
                                       "(read from endpoint.json / the driver's client timeout) "
                                       "or {} for an in-process leg", **base)
    if not hops:
        return TransportVerdict(ok=True, required_s=None, binding_hop=None, binding_s=None,
                                reason="no transport hops (in-process leg)", **base)
    missing = sorted(k for k, v in hops.items() if v is None)
    if missing:
        return TransportVerdict(ok=False, required_s=None, binding_hop=missing[0],
                                binding_s=None,
                                reason=f"undeterminable: hop {missing[0]!r} has no recorded "
                                       "timeout — record it (endpoint.json upstream_timeout_s "
                                       "/ the driver's client timeout) before climbing", **base)
    binding_hop, binding_s = min(hops.items(), key=lambda kv: kv[1])   # type: ignore[arg-type]
    req = transport_required_s(next_cap, floor_tok_per_s=floor_tok_per_s, margin=margin)
    ok = float(binding_s) >= req
    if ok:
        reason = (f"hop {binding_hop!r} allows {binding_s:.0f} s >= {req:.0f} s required for "
                  f"{next_cap} tokens at {floor_tok_per_s} tok/s x {margin}")
    else:
        reason = (f"REFUSED: hop {binding_hop!r} allows {binding_s:.0f} s but {next_cap} tokens "
                  f"at a {floor_tok_per_s} tok/s floor x {margin} margin need {req:.0f} s — "
                  f"raise that hop's timeout to >= {req:.0f} s (proxy: upstream timeout_s / "
                  f"--upstream-timeout-s; driver: AIOHTTP_TOTAL_TIMEOUT_S) before climbing")
    return TransportVerdict(ok=ok, required_s=req, binding_hop=binding_hop,
                            binding_s=float(binding_s), reason=reason, **base)


#: the two hops between a served driver and the engine, by name
HOP_DRIVER = "driver"            # the driver's aiohttp total timeout to the proxy
HOP_PROXY_UPSTREAM = "proxy_upstream"   # the proxy's total timeout to vLLM
#: the endpoint.json keys a standup records the proxy's upstream timeout under.
#: Two standups record the same fact under two names — the sdfbase standup
#: (mt_serving_up.py) as ``upstream_timeout_s``, the tinycap standup
#: (capctrl_serving_up.py on oos-thinking-run / oos-offcap) as
#: ``proxy_upstream_timeout_s`` — so the reader accepts either, and REFUSES
#: when both are present and disagree (two records of one value that differ
#: is a torn record, not a choice).
ENDPOINT_UPSTREAM_TIMEOUT_KEYS = ("upstream_timeout_s", "proxy_upstream_timeout_s")
ENDPOINT_UPSTREAM_TIMEOUT_KEY = ENDPOINT_UPSTREAM_TIMEOUT_KEYS[0]   # kept for callers


def served_hop_timeouts(endpoint: Optional[Mapping[str, Any]],
                        driver_timeout_s: Optional[float]) -> Dict[str, Optional[float]]:
    """The served path's hop timeouts, read from the RECORDS rather than from
    defaults: the driver's own client timeout, and the proxy's upstream
    timeout as the standup recorded it in ``endpoint.json`` under either of
    :data:`ENDPOINT_UPSTREAM_TIMEOUT_KEYS`. A standup that recorded no
    upstream timeout yields ``None`` for that hop, which
    :func:`transport_check` refuses — the proxy's compiled-in default is
    exactly the value that censored the slow leg, and reading it here would
    turn a refusal into a guess."""
    up = None
    if endpoint is not None:
        found = {k: float(endpoint[k]) for k in ENDPOINT_UPSTREAM_TIMEOUT_KEYS
                 if endpoint.get(k) is not None}
        if len(set(found.values())) > 1:
            raise CapLadderError(
                f"endpoint.json records the proxy upstream timeout under two keys that "
                f"DISAGREE: {found} — a torn record; fix the standup rather than pick one")
        if found:
            up = next(iter(found.values()))
    return {HOP_DRIVER: (None if driver_timeout_s is None else float(driver_timeout_s)),
            HOP_PROXY_UPSTREAM: up}


def parse_hop_timeouts(specs: Optional[Sequence[str]], *, in_process: bool = False
                       ) -> Optional[Dict[str, Optional[float]]]:
    """CLI form -> :func:`transport_check` input. ``in_process`` -> ``{}``;
    no specs -> ``None`` (undeterminable, refuses); each spec ``NAME=SECONDS``
    (``NAME=`` or ``NAME=none`` records an unknown hop, which refuses)."""
    if in_process:
        if specs:
            raise CapLadderError("--in-process and --hop-timeout-s are mutually exclusive")
        return {}
    if not specs:
        return None
    out: Dict[str, Optional[float]] = {}
    for spec in specs:
        if "=" not in spec:
            raise CapLadderError(f"--hop-timeout-s {spec!r}: expected NAME=SECONDS")
        name, _, val = spec.partition("=")
        name = name.strip()
        if not name:
            raise CapLadderError(f"--hop-timeout-s {spec!r}: empty hop name")
        val = val.strip().lower()
        if val in ("", "none", "null", "unknown"):
            out[name] = None
            continue
        try:
            out[name] = float(val)
        except ValueError:
            raise CapLadderError(f"--hop-timeout-s {spec!r}: {val!r} is not a number") from None
        if out[name] <= 0:
            raise CapLadderError(f"--hop-timeout-s {spec!r}: must be > 0")
    return out


def verdict_for(*, share_truncated: float, n_truncated: int, cap: int,
                threshold: float = AT_CAP_SHARE_THRESHOLD, ceiling: int = CAP_CEILING,
                ladder: Sequence[int] = CAP_LADDER, climbed: bool = False,
                ladder_available: bool = True,
                hop_timeouts_s: Optional[Mapping[str, Optional[float]]] = None,
                floor_tok_per_s: float = TRANSPORT_FLOOR_TOK_PER_S,
                margin: float = TRANSPORT_MARGIN,
                ) -> Tuple[str, bool, Optional[int], bool, Optional[TransportVerdict]]:
    """``(verdict, escalate, next_cap, at_ceiling, transport)`` — the one
    place the exit kinds are decided. ``climbed`` tells the two ceiling exits
    apart; ``ladder_available`` False turns what would be an escalation into
    ``no_rung`` (nothing can act on it, so nothing is asked for); and an
    escalation is issued ONLY after :func:`transport_check` passes for the
    next rung (:data:`TRANSPORT_RULE`) — otherwise ``transport_refused``,
    with ``escalate`` False so no caller can act on it. ``transport`` is
    ``None`` on every exit that never reached the check.

    CONSUMERS WITH THEIR OWN TRIGGER (the GSM8K paired-200 instrument): this
    function decides on the TRUNCATED share under ``threshold`` — the standing
    rule's quantity, loops and boundary draws set aside. An instrument whose
    trigger is a different quantity (the at-cap share with loops counted, at
    3 %) must NOT consume this verdict as its exit: on a mixed cell (7 at cap
    of which 2 are loops -> 2.5 % truncated) it returns ``below_threshold``
    and never reaches the transport check, which would suppress a rung that
    instrument requires. Such an instrument calls :func:`transport_check`
    directly after its own trigger fires and files this rule's verdict beside
    it as a report line.
    """
    at_ceiling = int(cap) >= int(ceiling)
    nxt = next_rung(cap, ladder, ceiling)
    over = share_truncated > threshold
    if n_truncated == 0:
        return EXIT_CLEAN, False, None, at_ceiling, None
    if not over:
        return EXIT_BELOW_THRESHOLD, False, None, at_ceiling, None
    if at_ceiling or nxt is None:
        return (EXIT_CEILING if climbed else EXIT_CEILING_AT_OUTSET), False, None, at_ceiling, None
    if not ladder_available:
        return EXIT_NO_RUNG, False, nxt, at_ceiling, None
    tv = transport_check(nxt, hop_timeouts_s, floor_tok_per_s=floor_tok_per_s, margin=margin)
    if not tv.ok:
        return EXIT_TRANSPORT_REFUSED, False, nxt, at_ceiling, tv
    return ESCALATE, True, nxt, at_ceiling, tv


def rung_status(records: Mapping[str, ItemRecord], cap: int, *,
                threshold: float = AT_CAP_SHARE_THRESHOLD,
                ceiling: int = CAP_CEILING,
                ladder: Sequence[int] = CAP_LADDER,
                climbed: bool = False,
                ladder_available: bool = True,
                hop_timeouts_s: Optional[Mapping[str, Optional[float]]] = None,
                floor_tok_per_s: float = TRANSPORT_FLOOR_TOK_PER_S,
                margin: float = TRANSPORT_MARGIN) -> RungStatus:
    """The trigger over classified items (:data:`TRIGGER_RULE`), gated by
    the transport check (:data:`TRANSPORT_RULE`) before any escalation."""
    recs = list(records.values())
    n = len(recs)
    trunc = sorted(r.item_id for r in recs if r.kind == KIND_TRUNCATED)
    nonterm = sorted(r.item_id for r in recs if r.kind == KIND_NON_TERMINATING)
    bound = sorted(r.item_id for r in recs if r.kind == KIND_BOUNDARY)
    n_at = sum(1 for r in recs if r.at_cap)
    share_t = (len(trunc) / n) if n else 0.0
    verdict, escalate, nxt, at_ceiling, tv = verdict_for(
        share_truncated=share_t, n_truncated=len(trunc), cap=cap, threshold=threshold,
        ceiling=ceiling, ladder=ladder, climbed=climbed, ladder_available=ladder_available,
        hop_timeouts_s=hop_timeouts_s, floor_tok_per_s=floor_tok_per_s, margin=margin)
    seen: Dict[str, int] = {}
    for r in recs:
        seen[str(r.source_cap)] = seen.get(str(r.source_cap), 0) + 1
    regen_all = [r for r in recs if r.regenerated]
    # the redraw statistics run over the regenerated items that CARRY a redraw
    # record (an Inspect retry row has none: its source draw is not in hand)
    regen = [r for r in regen_all if r.redraw]
    outcomes: Dict[str, int] = {}
    for r in regen:
        outcomes[r.redraw["outcome"]] = outcomes.get(r.redraw["outcome"], 0) + 1
    return RungStatus(
        cap=int(cap), n=n, n_at_cap=n_at, n_truncated=len(trunc),
        n_non_terminating=len(nonterm), n_boundary=len(bound),
        share_at_cap=round((n_at / n) if n else 0.0, 6),
        share_truncated=round(share_t, 6), threshold=float(threshold),
        ceiling=int(ceiling), escalate=escalate,
        next_cap=nxt, at_ceiling=at_ceiling, exit=verdict,
        climbed=bool(climbed), ladder_available=bool(ladder_available),
        truncated_ids=trunc, non_terminating_ids=nonterm, boundary_ids=bound,
        source_caps_seen=dict(sorted(seen.items(), key=lambda kv: int(kv[0]))),
        n_regenerated=len(regen_all), redraw_outcomes=dict(sorted(outcomes.items())),
        n_fork=sum(1 for r in regen if not r.redraw["prefix_extension"]),
        n_identical_text=sum(1 for r in regen if r.redraw["identical_text"]),
        n_prefix_extension=sum(1 for r in regen if r.redraw["prefix_extension"]),
        transport=(tv.as_dict() if tv is not None else None))


@dataclass
class LedgerReconciliation:
    """The producer's usage ledger held against the scored spool for one
    (cell route, cap): does the ledger know of at-cap completions the spool
    does not? (:data:`LEDGER_RULE`)"""

    selector: Dict[str, Any]       # {"vmodel", "cap", "task"} the ledger rows were selected by
    n_rows: int                    # ledger rows matching the selector, any outcome
    n_served: int                  # ... not rejected (a completion the server finished)
    n_at_cap: int                  # ... served with finish_reason == "length"
    n_rejected: int
    n_transport: int               # rejected with error_class "transport" (the forward died)
    spool_n: int
    spool_at_cap: int
    censored_at_cap: int           # max(0, n_at_cap - spool_at_cap): at-cap completions the spool lacks
    extra_completions: int         # max(0, n_served - spool_n): completions beyond one per item (retries)
    censored: bool                 # censored_at_cap > 0

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def read_usage_ledger(path: Path) -> List[Dict[str, Any]]:
    """The rows of a ``usage_ledger.jsonl``; a row missing a
    :data:`LEDGER_REQUIRED_FIELDS` field refuses (a ledger from before those
    columns cannot vouch for anything)."""
    p = Path(os.path.expanduser(str(path)))
    if not p.is_file():
        raise CapLadderError(f"usage ledger: {p} does not exist")
    rows: List[Dict[str, Any]] = []
    with open(p, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                raise CapLadderError(f"usage ledger: {p}:{ln} is not JSON ({e})") from None
            missing = [k for k in LEDGER_REQUIRED_FIELDS if k not in r]
            if missing:
                raise CapLadderError(f"usage ledger: {p}:{ln} lacks {missing} — a row the rule cannot read")
            rows.append(r)
    return rows


def ledger_at_cap_counts(rows: Iterable[Mapping[str, Any]], *, vmodel: str, cap: int,
                         task: Optional[str] = None) -> Dict[str, Any]:
    """The ledger's counts for one cell route at one cap: rows whose
    ``vmodel`` is the route's virtual model and whose ``max_tokens_requested``
    is ``cap`` (and whose ``task`` header is ``task`` when given — the tiny
    legs set it, the judged controls do not). ``n_served`` = not rejected;
    ``n_at_cap`` = served with ``finish_reason == "length"``. Refuses a
    selector that matches NO row: a ledger that saw none of the route's calls
    cannot vouch for its spool, and a mis-keyed selector would check nothing."""
    n_rows = n_served = n_at = n_rej = n_tr = 0
    for r in rows:
        if r.get("vmodel") != vmodel:
            continue
        if int(r.get("max_tokens_requested") or 0) != int(cap):
            continue
        if task is not None and r.get("task") != task:
            continue
        n_rows += 1
        if r.get("rejected"):
            n_rej += 1
            if r.get("error_class") == LEDGER_ERROR_CLASS_TRANSPORT:
                n_tr += 1
            continue
        n_served += 1
        if r.get("finish_reason") == LEDGER_FINISH_AT_CAP:
            n_at += 1
    if n_rows == 0:
        raise CapLadderError(
            f"usage ledger: no row for vmodel {vmodel!r} at max_tokens_requested {cap}"
            + (f" with task {task!r}" if task is not None else "")
            + " — the ledger saw none of this route's calls at this cap, so it cannot vouch for the "
            "spool (a mis-keyed selector would check nothing); refusing")
    return {"selector": {"vmodel": vmodel, "cap": int(cap), "task": task}, "n_rows": n_rows,
            "n_served": n_served, "n_at_cap": n_at, "n_rejected": n_rej, "n_transport": n_tr}


def reconcile_with_ledger(status: RungStatus, counts: Mapping[str, Any]) -> RungStatus:
    """Hold a rung reading to the producer's ledger (:data:`LEDGER_RULE`):
    the reconciliation is recorded on the status; when the ledger holds more
    at-cap completions than the spool, the scored set was censored by the
    transport and the exit becomes ``ledger_censored`` — unresolved, no
    escalation (a censored reading cannot say which rung is next; the
    transport must be fixed and the pass re-drawn). A reading that agrees
    keeps its exit and carries the reconciliation as evidence."""
    rec = LedgerReconciliation(
        selector=dict(counts["selector"]), n_rows=int(counts["n_rows"]),
        n_served=int(counts["n_served"]), n_at_cap=int(counts["n_at_cap"]),
        n_rejected=int(counts["n_rejected"]), n_transport=int(counts["n_transport"]),
        spool_n=int(status.n), spool_at_cap=int(status.n_at_cap),
        censored_at_cap=max(0, int(counts["n_at_cap"]) - int(status.n_at_cap)),
        extra_completions=max(0, int(counts["n_served"]) - int(status.n)),
        censored=int(counts["n_at_cap"]) > int(status.n_at_cap))
    out = replace(status, ledger=rec.as_dict())
    if rec.censored:
        out = replace(out, exit=EXIT_LEDGER_CENSORED, escalate=False)
    return out


def measure_rung(entries: Mapping[str, str], tok, cap: int, *,
                 source_caps: Optional[Mapping[str, int]] = None,
                 other_draws: Optional[Mapping[str, Sequence[Tuple[int, str]]]] = None,
                 regenerated: Iterable[str] = (),
                 source_texts: Optional[Mapping[str, str]] = None,
                 source_caps_of_regenerated: Optional[Mapping[str, int]] = None,
                 threshold: float = AT_CAP_SHARE_THRESHOLD,
                 ceiling: int = CAP_CEILING,
                 ladder: Sequence[int] = CAP_LADDER,
                 climbed: bool = False,
                 hop_timeouts_s: Optional[Mapping[str, Optional[float]]] = None,
                 floor_tok_per_s: float = TRANSPORT_FLOOR_TOK_PER_S,
                 margin: float = TRANSPORT_MARGIN,
                 ledger_counts: Optional[Mapping[str, Any]] = None,
                 ) -> Tuple[RungStatus, Dict[str, ItemRecord]]:
    """Classify a complete spool at a rung and apply the trigger — the one
    call both generating paths make after a pass. ``climbed`` is True for a
    store that regenerated a lower rung's at-cap items (a carry-over run).
    ``ledger_counts`` (:func:`ledger_at_cap_counts` over the producer's usage
    ledger) reconciles the reading against what the producer actually
    finished (:data:`LEDGER_RULE`); without it the reading is UNRECONCILED
    and ``status.ledger`` is None — the report says so."""
    records = classify_items(entries, tok, cap=cap, source_caps=source_caps,
                             other_draws=other_draws, regenerated=regenerated,
                             source_texts=source_texts,
                             source_caps_of_regenerated=source_caps_of_regenerated)
    status = rung_status(records, cap, threshold=threshold, ceiling=ceiling,
                         ladder=ladder, climbed=climbed, hop_timeouts_s=hop_timeouts_s,
                         floor_tok_per_s=floor_tok_per_s, margin=margin)
    if ledger_counts is not None:
        status = reconcile_with_ledger(status, ledger_counts)
    return status, records


def inspect_rung_from_rows(rows: Iterable[Mapping[str, Any]], *, task: str,
                           threshold: float = AT_CAP_SHARE_THRESHOLD,
                           ceiling: int = CAP_CEILING,
                           ladder: Sequence[int] = CAP_LADDER,
                           agg_item_id: str = "__aggregate__") -> RungStatus:
    """The Inspect leg's rung reading from its OWN filed rows (``rows.jsonl``
    dicts of one task): at-cap is the provider's ``stop_reason ==
    "max_tokens"`` (exact, the leg's one definition), the cap is each row's
    ``max_tokens_used`` (the request's, never a constant copied here), and
    degeneracy is read off the row's ``content``. One reading per filed
    per-item / per-sample row (the leg files final attempts only, so a
    retried sample is read at its retry cap with ``attempt`` 2). The leg has
    no rung mechanism, so an over-threshold reading below the ceiling is
    ``no_rung``, and one at or above the ceiling — MATH-500 at its 16k
    default — is ``ceiling_at_outset``. Rows without a ``stop_reason`` (the
    producer did not say) are outside the denominator, as in
    ``inspect_served.task_quality``."""
    recs: Dict[str, ItemRecord] = {}
    caps_seen: List[int] = []
    for r in rows:
        if r.get("item_id") == agg_item_id:
            continue
        ex = r.get("extra_json")
        ex = json.loads(ex) if isinstance(ex, str) else dict(ex or {})
        if ex.get("stop_reason") is None:
            continue
        mt = ex.get("max_tokens_used")
        if mt is None:
            raise CapLadderError(
                f"{task}: row {r.get('item_id')!r}/{r.get('metric')!r} has a stop_reason but "
                "no max_tokens_used — the cap it was requested at is unrecorded; refusing "
                "to guess it")
        mt = int(mt)
        caps_seen.append(mt)
        key = f"{r.get('item_id')}#{ex.get('sample_idx', 0)}#{r.get('metric')}"
        text = ex.get("content") or ""
        at = ex.get("stop_reason") == "max_tokens"
        deg = degeneracy(text)
        ct = int(ex.get("completion_tokens") or 0)
        kind = None
        if at:
            kind = KIND_NON_TERMINATING if deg["degenerate"] else KIND_TRUNCATED
        recs[key] = ItemRecord(item_id=key, text_sha256=text_sha256(text), source_cap=mt,
                               ntok=ct, completion_tokens=min(ct, mt) if ct else mt,
                               at_cap=at, degenerate=deg["degenerate"],
                               rep_ratio=deg["rep_ratio"], tail_rep_ratio=deg["tail_rep_ratio"],
                               kind=kind, regenerated=int(ex.get("attempt") or 1) > 1)
    cap = max(caps_seen) if caps_seen else 0
    return rung_status(recs, cap, threshold=threshold, ceiling=ceiling, ladder=ladder,
                       climbed=False, ladder_available=False)


# --------------------------------------------------------------------------- #
# the redraw record                                                            #
# --------------------------------------------------------------------------- #


def resolve_redraw(source_text: str, new_text: str, tok, *, source_cap: int,
                   new_cap: int) -> Dict[str, Any]:
    """A regenerated item against the at-cap draw it replaced (the peer's
    ``_fork_fields``, plus the outcome the ladder reads): ``still_at_cap`` —
    at cap again at ``new_cap``; ``boundary_landing`` — terminated with fewer
    than ``source_cap`` tokens (the source draw was boundary noise);
    ``extended`` — terminated at or past the source cap (real truncation,
    resolved). Text relationship recorded, never asserted."""
    src, new = source_text or "", new_text or ""
    identical = new == src
    prefix = new.startswith(src)
    if prefix:
        char = tok_idx = None
    else:
        char = divergence(src, new)
        ia = tok(src, add_special_tokens=False)["input_ids"]
        ib = tok(new, add_special_tokens=False)["input_ids"]
        tok_idx = divergence(ia, ib)
    new_ct = completion_tokens(tok, new, new_cap)
    still = new_ct >= int(new_cap)                   # at_cap.is_at_cap, on the count already taken
    landing = bool(not still and new_ct < int(source_cap))
    extended = bool(not still and not landing)
    outcome = (REDRAW_STILL_AT_CAP if still else
               REDRAW_BOUNDARY_LANDING if landing else REDRAW_EXTENDED)
    out = {"source_cap": int(source_cap), "new_cap": int(new_cap),
           "source_completion_tokens": completion_tokens(tok, src, source_cap),
           "new_completion_tokens": new_ct,
           "identical_text": identical, "prefix_extension": prefix,
           "first_divergence_char": char, "first_divergence_token": tok_idx,
           "boundary_landing": landing, "still_at_cap": still, "extended": extended,
           "outcome": outcome}
    assert tuple(out) == REDRAW_KEYS
    return out


# --------------------------------------------------------------------------- #
# the sidecar                                                                  #
# --------------------------------------------------------------------------- #


def sidecar_path(spool_path: Path) -> Path:
    """``gens_<name>.jsonl`` -> ``gens_<name>.caps.jsonl`` beside it."""
    p = Path(spool_path)
    if p.suffix != ".jsonl":
        raise CapLadderError(f"{p}: not a .jsonl spool")
    return p.with_name(p.name[: -len(".jsonl")] + SIDECAR_SUFFIX)


def write_sidecar(path: Path, records: Mapping[str, ItemRecord], *,
                  header: Mapping[str, Any]) -> Path:
    """Rewrite the sidecar atomically (it is a READING of the spool, so it is
    always recomputed whole: line 0 a header, then one row per item, sorted
    by id). Mode 0600 like the spool; ids and numbers only."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps({"__caps_header__": {**dict(header), "schema": RECORD_SCHEMA,
                                                 "row_keys": list(ROW_KEYS)}},
                           sort_keys=True) + "\n")
        for iid in sorted(records):
            f.write(json.dumps(records[iid].as_row(), sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)
    os.chmod(p, 0o600)
    return p


def read_sidecar(path: Path) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """``(header, {item_id: row})``; refuses a file without the header."""
    p = Path(path)
    lines = p.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise CapLadderError(f"{p}: empty sidecar")
    head = json.loads(lines[0])
    if "__caps_header__" not in head:
        raise CapLadderError(f"{p}: first line is not a caps sidecar header")
    rows: Dict[str, Dict[str, Any]] = {}
    for i, line in enumerate(lines[1:], start=2):
        r = json.loads(line)
        if r["item_id"] in rows:
            raise CapLadderError(f"{p}: item {r['item_id']!r} appears twice (line {i})")
        rows[r["item_id"]] = r
    return head["__caps_header__"], rows


def bind_sidecar(rows: Mapping[str, Mapping[str, Any]], entries: Mapping[str, str],
                 *, path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """The sidecar rows that describe EXACTLY the spool's texts (bound by
    ``text_sha256``, the peer's rule): a spooled item with no row, or whose
    row's digest is another text's, refuses — its source cap cannot be read
    and guessing it would misclassify it at every later rung."""
    out: Dict[str, Dict[str, Any]] = {}
    bad: List[str] = []
    for iid, text in entries.items():
        r = rows.get(iid)
        if r is None or r.get("text_sha256") != text_sha256(text or ""):
            bad.append(iid)
            continue
        out[iid] = dict(r)
    if bad:
        raise CapLadderError(
            f"{path or 'sidecar'}: {len(bad)} spooled item(s) have no sidecar row "
            f"matching their text (first {bad[:3]}) — the sidecar is stale or from "
            "another draw; re-measure the store (scripts/cap_ladder.py measure)")
    return out


def source_caps_for(entries: Mapping[str, str], sidecar: Optional[Path],
                    default_cap: int) -> Tuple[Dict[str, int], str]:
    """Per-item source caps for a store: from its sidecar when present and
    bound, else every item at ``default_cap`` (a store predating the rule
    — every text was drawn at the header's cap). Returns ``(caps, source)``."""
    if sidecar is not None and Path(sidecar).is_file():
        _h, rows = read_sidecar(sidecar)
        bound = bind_sidecar(rows, entries, path=sidecar)
        return {iid: int(r["source_cap"]) for iid, r in bound.items()}, "sidecar"
    return {iid: int(default_cap) for iid in entries}, "header_cap"


# --------------------------------------------------------------------------- #
# the per-cell record and the report                                           #
# --------------------------------------------------------------------------- #


def cell_record(*, cell_id: str, engine: str, cap: int,
                statuses: Mapping[str, RungStatus],
                store_dir: Optional[Path] = None,
                threshold: float = AT_CAP_SHARE_THRESHOLD,
                ceiling: int = CAP_CEILING,
                ladder: Sequence[int] = CAP_LADDER,
                extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """``cap_ladder.json`` / ``manifest["cap_ladder"]`` for one store."""
    rec = {"schema": RECORD_SCHEMA, "cell_id": cell_id, "engine": engine,
           "cap": int(cap), "threshold": float(threshold), "ceiling": int(ceiling),
           "ladder": [int(x) for x in ladder],
           "rules": {"at_cap": "carry_over.AT_CAP_RULE", "degenerate": DEGENERATE_RULE,
                     "trigger": TRIGGER_RULE},
           "controls": {c: s.as_dict() for c, s in sorted(statuses.items())},
           "store_dir": (str(store_dir) if store_dir is not None else None)}
    if extra:
        rec.update(dict(extra))
    return rec


def write_cell_record(cell_dir: Path, record: Mapping[str, Any]) -> Path:
    p = Path(cell_dir) / RECORD_NAME
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(dict(record), indent=2, sort_keys=True, default=str) + "\n",
                   encoding="utf-8")
    os.replace(tmp, p)
    return p


def status_line(prefix: str, control: str, s: RungStatus) -> str:
    """The one loud stdout line per (control, rung)."""
    verdict = (f"ESCALATE -> {s.next_cap}" if s.escalate else
               f"exit {s.exit}" + (f" (residual {s.n_truncated})" if s.n_truncated else ""))
    return (f"{prefix} cap-ladder {control} @ {s.cap}: at-cap {s.n_at_cap}/{s.n} "
            f"({s.share_at_cap:.2%}); truncated {s.n_truncated} ({s.share_truncated:.2%}), "
            f"non-terminating {s.n_non_terminating}, boundary {s.n_boundary}"
            + (f"; regenerated {s.n_regenerated} -> {s.redraw_outcomes}" if s.n_regenerated else "")
            + f"; {verdict}")


def ladder_report(rungs: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """The exit report over rung records ``{cell_id, control, cap, status,
    store_dir?}`` (``status`` = :meth:`RungStatus.as_dict`). Groups by
    (cell, control), orders by cap, reads each pair's exit off its TOP rung
    and lists every threshold exit that still carries a non-zero residual,
    every ceiling exit, and every pair whose top rung says escalate but no
    higher rung is in hand (``incomplete``)."""
    by_pair: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    for r in rungs:
        by_pair.setdefault((str(r["cell_id"]), str(r["control"])), []).append(r)
    cells: Dict[str, Any] = {}
    exits: Dict[str, int] = {}
    thr_resid: List[Dict[str, Any]] = []
    ceiling_exits: List[Dict[str, Any]] = []
    ceiling_at_outset: List[Dict[str, Any]] = []
    no_rung: List[Dict[str, Any]] = []
    transport_refused: List[Dict[str, Any]] = []
    ledger_censored: List[Dict[str, Any]] = []
    n_unreconciled = 0
    incomplete: List[Dict[str, Any]] = []
    n_nonterm = 0
    for (cell, control), recs in sorted(by_pair.items()):
        recs = sorted(recs, key=lambda r: int(r["cap"]))
        caps = [int(r["cap"]) for r in recs]
        if len(set(caps)) != len(caps):
            raise CapLadderError(f"{cell}/{control}: two rung records at one cap {caps}")
        top = recs[-1]
        st = dict(top["status"])
        exit_kind = "incomplete" if st["escalate"] else st["exit"]
        entry = {
            "cell_id": cell, "control": control, "rungs_attempted": caps,
            "exit": exit_kind, "exit_cap": int(top["cap"]),
            "residual_n": int(st["n_truncated"]),
            "residual_share": float(st["share_truncated"]),
            "residual_ids": list(st.get("truncated_ids") or []),
            "non_terminating_n": int(st["n_non_terminating"]),
            "boundary_n": int(st["n_boundary"]),
            "per_rung": [{"cap": int(r["cap"]), "n": r["status"]["n"],
                          "n_at_cap": r["status"]["n_at_cap"],
                          "n_truncated": r["status"]["n_truncated"],
                          "share_truncated": r["status"]["share_truncated"],
                          "n_non_terminating": r["status"]["n_non_terminating"],
                          "n_boundary": r["status"]["n_boundary"],
                          "n_regenerated": r["status"].get("n_regenerated", 0),
                          "redraw_outcomes": r["status"].get("redraw_outcomes", {}),
                          "verdict": r["status"]["exit"],
                          "store_dir": r.get("store_dir")} for r in recs],
        }
        entry["climbed"] = bool(st.get("climbed")) or len(caps) > 1
        entry["ladder_available"] = bool(st.get("ladder_available", True))
        cells[f"{cell}|{control}"] = entry
        exits[exit_kind] = exits.get(exit_kind, 0) + 1
        n_nonterm += entry["non_terminating_n"]
        short = {k: entry[k] for k in ("cell_id", "control", "exit_cap", "residual_n",
                                       "residual_share", "non_terminating_n")}
        if exit_kind == EXIT_BELOW_THRESHOLD and entry["residual_n"] > 0:
            thr_resid.append(short)
        elif exit_kind == EXIT_CEILING:
            ceiling_exits.append(short)
        elif exit_kind == EXIT_CEILING_AT_OUTSET:
            ceiling_at_outset.append(short)
        elif exit_kind == EXIT_NO_RUNG:
            no_rung.append({**short, "would_climb_to": st.get("next_cap")})
        elif exit_kind == EXIT_TRANSPORT_REFUSED:
            tv = st.get("transport") or {}
            transport_refused.append({**short, "would_climb_to": st.get("next_cap"),
                                      "binding_hop": tv.get("binding_hop"),
                                      "binding_s": tv.get("binding_s"),
                                      "required_s": tv.get("required_s"),
                                      "reason": tv.get("reason")})
        elif exit_kind == EXIT_LEDGER_CENSORED:
            lg = st.get("ledger") or {}
            ledger_censored.append({**short, "ledger_at_cap": lg.get("n_at_cap"),
                                    "spool_at_cap": lg.get("spool_at_cap"),
                                    "censored_at_cap": lg.get("censored_at_cap"),
                                    "extra_completions": lg.get("extra_completions"),
                                    "n_transport": lg.get("n_transport")})
        elif exit_kind == "incomplete":
            incomplete.append({**short, "next_cap": st.get("next_cap")})
        if st.get("ledger") is None:
            n_unreconciled += 1
    return {"schema": RECORD_SCHEMA, "cells": cells,
            "summary": {"n_pairs": len(cells), "exits": dict(sorted(exits.items())),
                        "threshold_exits_with_residual": thr_resid,
                        "ceiling_exits": ceiling_exits,
                        "ceiling_at_outset": ceiling_at_outset,
                        "no_rung": no_rung, "incomplete": incomplete,
                        "transport_refused": transport_refused,
                        "ledger_censored": ledger_censored,
                        "n_unreconciled": n_unreconciled,
                        "non_terminating_total": n_nonterm}}


def render_report_md(report: Mapping[str, Any]) -> str:
    s = report["summary"]
    lines = [f"# cap ladder — {s['n_pairs']} (cell, readout) pairs", "",
             f"exits: {s['exits']}; non-terminating items (excluded from every trigger, "
             f"carried as-is): {s['non_terminating_total']}", "",
             "| cell | readout | rungs | exit | exit cap | residual n | residual share | non-terminating | boundary |",
             "|---|---|---|---|---:|---:|---:|---:|---:|"]
    for e in report["cells"].values():
        lines.append(f"| `{e['cell_id']}` | {e['control']} | {'→'.join(str(c) for c in e['rungs_attempted'])} | "
                     f"{e['exit']} | {e['exit_cap']} | {e['residual_n']} | {e['residual_share']:.2%} | "
                     f"{e['non_terminating_n']} | {e['boundary_n']} |")
    lines += ["", "## threshold exits with a non-zero residual", ""]
    if s["threshold_exits_with_residual"]:
        for e in s["threshold_exits_with_residual"]:
            lines.append(f"- `{e['cell_id']}` / {e['control']} @ {e['exit_cap']}: "
                         f"{e['residual_n']} truncated ({e['residual_share']:.2%} <= threshold)")
    else:
        lines.append("- none")
    lines += ["", "## ceiling exits — climbed and ran out (residual nothing can resolve)", ""]
    if s["ceiling_exits"]:
        for e in s["ceiling_exits"]:
            lines.append(f"- `{e['cell_id']}` / {e['control']} @ {e['exit_cap']}: "
                         f"{e['residual_n']} truncated ({e['residual_share']:.2%})")
    else:
        lines.append("- none")
    lines += ["", "## at the ceiling from the outset — no rung ever existed above the "
              "configured cap (measured, classified, residual reported; NOT an exhausted ladder)", ""]
    if s["ceiling_at_outset"]:
        for e in s["ceiling_at_outset"]:
            lines.append(f"- `{e['cell_id']}` / {e['control']} @ {e['exit_cap']}: "
                         f"{e['residual_n']} truncated ({e['residual_share']:.2%})")
    else:
        lines.append("- none")
    if s["no_rung"]:
        lines += ["", "## over threshold on a leg with no rung mechanism (unresolved; "
                  "the leg's own retry, if any, is already in these counts)", ""]
        for e in s["no_rung"]:
            lines.append(f"- `{e['cell_id']}` / {e['control']} @ {e['exit_cap']}: "
                         f"{e['residual_n']} truncated ({e['residual_share']:.2%}); the ladder "
                         f"would climb to {e['would_climb_to']}")
    if s.get("transport_refused"):
        lines += ["", "## refused by the transport guard (a rung exists but the configured "
                  "hop timeout cannot carry it; NOT climbed, residual reported)", ""]
        for e in s["transport_refused"]:
            lines.append(f"- `{e['cell_id']}` / {e['control']} @ {e['exit_cap']} -> {e['would_climb_to']}: "
                         f"{e['residual_n']} truncated ({e['residual_share']:.2%}); hop "
                         f"{e['binding_hop']!r} allows {e['binding_s']} s, needs {e['required_s']:.0f} s"
                         if e.get("required_s") is not None else
                         f"- `{e['cell_id']}` / {e['control']} @ {e['exit_cap']} -> {e['would_climb_to']}: "
                         f"{e['residual_n']} truncated ({e['residual_share']:.2%}); {e['reason']}")
    if s.get("ledger_censored"):
        lines += ["", "## censored by the transport (the producer's usage ledger holds at-cap completions "
                  "the scored spool does not — a client timeout abandoned draws the server finished; "
                  "NOT a threshold exit; fix the transport and re-draw)", ""]
        for e in s["ledger_censored"]:
            lines.append(f"- `{e['cell_id']}` / {e['control']} @ {e['exit_cap']}: ledger at cap "
                         f"{e['ledger_at_cap']} vs spool at cap {e['spool_at_cap']} "
                         f"({e['censored_at_cap']} censored; {e['extra_completions']} completions beyond "
                         f"one per item; {e['n_transport']} transport failures)")
    if s.get("n_unreconciled"):
        lines += ["", f"UNRECONCILED: {s['n_unreconciled']} of {s['n_pairs']} (cell, readout) pairs were "
                  "read from the scored spool with NO producer-ledger reconciliation — their at-cap "
                  "share is a lower bound if any request was abandoned by the client (LEDGER_RULE).", ""]
    if s["incomplete"]:
        lines += ["", "## incomplete (top rung says escalate; the next rung is not in hand)", ""]
        for e in s["incomplete"]:
            lines.append(f"- `{e['cell_id']}` / {e['control']} @ {e['exit_cap']} -> {e['next_cap']}: "
                         f"{e['residual_n']} truncated ({e['residual_share']:.2%})")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# sizing at the outset                                                         #
# --------------------------------------------------------------------------- #


@dataclass
class CapSizing:
    """The recommendation for one (family, task) from a measured, possibly
    censored, completion-length distribution."""

    n: int
    measured_cap: int
    threshold: float
    share_at_measured_cap: float           # every row at the cap, loops included
    censored: bool                         # TRUNCATED share at the cap > threshold
    p99_completion: Optional[int]          # nearest-rank over content rows; None when censored
    recommended_cap: int
    at_ceiling: bool
    shares_by_rung: Dict[str, Optional[float]]   # str(rung) -> truncated share >= rung (None: unreadable)
    reason: str
    n_at_cap: int = 0                      # rows at the measured cap
    n_at_cap_not_truncated: Optional[int] = None   # of those, known loops / repetitive / boundary draws
    share_truncated_at_measured_cap: float = 0.0   # (n_at_cap - excluded) / n: what `censored` reads
    degeneracy_classified: bool = False    # False: the at-cap share is an UPPER BOUND on truncation

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def nearest_rank_percentile(values: Sequence[int], q: float) -> int:
    if not values:
        raise CapLadderError("percentile of an empty distribution")
    s = sorted(int(v) for v in values)
    k = max(1, math.ceil(q * len(s)))
    return s[k - 1]


def size_cap(completions: Sequence[int], measured_cap: int, *,
             n_at_cap_not_truncated: Optional[int] = None,
             threshold: float = AT_CAP_SHARE_THRESHOLD,
             ladder: Sequence[int] = CAP_LADDER,
             ceiling: int = CAP_CEILING) -> CapSizing:
    """The smallest rung whose TRUNCATED share would be at or below the
    threshold, from completion counts CLAMPED at ``measured_cap``.

    The rule triggers on truncation, not on presence at the cap: a decode
    loop, a repetitive tail or a boundary draw runs to whatever cap it is
    given and measures nothing about content length (the ladder half —
    :func:`classify_items`, :func:`verdict_for` — never counts them). A
    caller that has classified the at-cap rows passes how many of them are
    NOT truncation as ``n_at_cap_not_truncated``; those rows leave the
    numerator at every rung and the p99. A caller with counts only (the
    ``.npy`` measurement, which stored no text) passes ``None``: every
    at-cap row then counts, the recommendation is an upper bound, and the
    reason says so (``degeneracy_classified`` is False). Measured on the
    sampled lane's uncensored data, the difference is up to four rungs on
    loop-prone families: 40 of 123 threshold-exit (lane, readout, cell,
    trial) groups sized to the 16,384 ceiling on raw at-cap counts and to
    4,096 or below on truncated counts.

    Censored (truncated share at the measured cap above the threshold) ->
    the next rung above the measured cap, flagged; nothing above the
    measured cap is read."""
    vals = [int(v) for v in completions]
    if not vals:
        raise CapLadderError("size_cap: no completions to size from")
    if max(vals) > int(measured_cap):
        raise CapLadderError(
            f"size_cap: a completion count {max(vals)} exceeds the stated measured cap "
            f"{measured_cap} — the arrays were not clamped at that cap")
    n = len(vals)
    mc = int(measured_cap)
    n_at_cap = sum(1 for v in vals if v >= mc)
    classified = n_at_cap_not_truncated is not None
    excl = int(n_at_cap_not_truncated) if classified else 0
    if excl < 0 or excl > n_at_cap:
        raise CapLadderError(
            f"size_cap: {excl} at-cap rows declared not truncated, but {n_at_cap} rows are at the "
            f"measured cap {mc} — the exclusion must lie in [0, {n_at_cap}]")
    n_trunc = n_at_cap - excl
    share_at = n_at_cap / n
    share_trunc = n_trunc / n

    def share_from(r: int) -> float:
        # rows at or above r that measure content length: terminated rows at or
        # above r plus the truncated at-cap rows; the excluded at-cap rows (all
        # equal to mc, so which ones is immaterial) leave the numerator
        return (sum(1 for v in vals if v >= r) - excl) / n

    rungs = sorted(int(r) for r in ladder)
    shares: Dict[str, Optional[float]] = {}
    for r in rungs:
        shares[str(r)] = (round(share_from(r), 6) if r <= mc else None)
    excl_s = (f"; {excl} of the {n_at_cap} at-cap rows are loops / repetitive / boundary draws and "
              f"do not count as truncation" if classified else
              f"; the at-cap rows were NOT classified for degeneracy, so this at-cap share is an "
              f"UPPER BOUND on the truncated share")
    censored = share_trunc > threshold
    at_ceiling = False
    if censored:
        nxt = next_rung(mc, ladder, ceiling)
        if nxt is None:
            rec, at_ceiling = int(ceiling), True
            reason = (f"censored: {share_trunc:.2%} of {n} completions are truncated at the measured "
                      f"cap {mc} (> {threshold:.0%}){excl_s}; no rung lies above it under the ceiling "
                      f"{ceiling}; the ceiling is the recommendation and a residual is expected")
        else:
            rec = nxt
            reason = (f"censored: {share_trunc:.2%} of {n} completions are truncated at the measured "
                      f"cap {mc} (> {threshold:.0%}){excl_s}; the p99 is unreadable; start at the next "
                      f"rung {rec} and let the ladder settle it")
        p99 = None
    else:
        kept = sorted(vals)[: n - excl] if excl else vals
        p99 = nearest_rank_percentile(kept, 0.99)
        candidates = sorted(set(r for r in rungs if r <= mc) | {mc})
        ok = [r for r in candidates if share_from(r) <= threshold]
        pick = ok[0]
        if pick not in rungs:
            up = [r for r in rungs if r >= pick]
            pick = up[0] if up else int(ceiling)
            at_ceiling = pick == int(ceiling) and not up
        rec = int(min(pick, ceiling))
        reason = (f"resolvable: {share_trunc:.2%} of {n} completions truncated at the measured cap "
                  f"{mc} (<= {threshold:.0%}){excl_s}; p99 completion {p99}; the smallest rung with "
                  f"truncated share <= {threshold:.0%} is {rec}")
    return CapSizing(n=n, measured_cap=mc, threshold=float(threshold),
                     share_at_measured_cap=round(share_at, 6), censored=censored,
                     p99_completion=p99, recommended_cap=rec, at_ceiling=at_ceiling,
                     shares_by_rung=shares, reason=reason,
                     n_at_cap=n_at_cap, n_at_cap_not_truncated=(excl if classified else None),
                     share_truncated_at_measured_cap=round(share_trunc, 6),
                     degeneracy_classified=classified)


def load_measurement(family: str, task: str, raw_dir: Path = DEFAULT_RAW_DIR
                     ) -> Tuple[List[int], List[int], Dict[str, Any]]:
    """``(prompt_tokens, completion_tokens, meta)`` from
    ``cap__<family>__<task>.npy`` (+ ``.meta.json``)."""
    import numpy as np

    p = Path(raw_dir) / f"cap__{family}__{task}.npy"
    if not p.is_file():
        raise CapLadderError(f"no measurement at {p} (families/tasks present: "
                             f"{sorted(x.name for x in Path(raw_dir).glob('cap__*.npy'))[:8]}...)")
    arr = np.load(p)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise CapLadderError(f"{p}: expected int64 [n, 2] = [prompt, completion], got {arr.shape}")
    meta_p = p.with_suffix(".meta.json")
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.is_file() else {}
    return [int(x) for x in arr[:, 0]], [int(x) for x in arr[:, 1]], meta


def measured_cap_from_meta(meta: Mapping[str, Any]) -> Optional[int]:
    """The cap the measurement clamped completions at, read from its note
    (``"... clamped 512"``); ``None`` when the note does not say."""
    m = re.search(r"clamped\s+(?:to\s+)?(\d+)", str(meta.get("note") or ""))
    return int(m.group(1)) if m else None


def size_cap_from_measurement(family: str, task: str, *, raw_dir: Path = DEFAULT_RAW_DIR,
                              measured_cap: Optional[int] = None,
                              threshold: float = AT_CAP_SHARE_THRESHOLD,
                              ladder: Sequence[int] = CAP_LADDER,
                              ceiling: int = CAP_CEILING) -> Dict[str, Any]:
    """:func:`size_cap` over the measured arrays; the measured cap is the
    meta note's unless given. A task whose completions were not stored (the
    measurement set them to the cap — gsm8k) is refused: its column is a
    bound, not a distribution."""
    _pt, ct, meta = load_measurement(family, task, raw_dir)
    note = str(meta.get("note") or "")
    if "NOT stored" in note or "prompt-only" in note:
        raise CapLadderError(
            f"cap__{family}__{task}: the measurement stored no completions ({note!r}) — "
            "its completion column is a bound, not a distribution; nothing to size from")
    mc = int(measured_cap) if measured_cap is not None else measured_cap_from_meta(meta)
    if mc is None:
        raise CapLadderError(
            f"cap__{family}__{task}: the meta note {note!r} does not name the clamp cap; "
            "pass measured_cap explicitly")
    s = size_cap(ct, mc, threshold=threshold, ladder=ladder, ceiling=ceiling)
    return {"family": family, "task": task, "raw": str(Path(raw_dir) / f"cap__{family}__{task}.npy"),
            "meta": dict(meta), **s.as_dict()}


#: ``length_quantiles.json`` (the sampled re-run lane's report): censoring
#: statuses a sizing may be read from. ``residual_rung_pending`` (provisional:
#: a higher rung is still to land) and ``residual_undetermined`` (no reclassify
#: decision on record, so the residual's composition is unknown) are refused.
LENGTH_QUANTILES_SIZEABLE = ("uncensored", "residual_threshold_exit", "residual_at_ceiling")
LENGTH_QUANTILES_ITEMS_NAME = "length_quantiles.items.jsonl"
#: what a per-trial (and the CLI's per-cell) sizing summary carries
_PER_TRIAL_KEYS = ("recommended_cap", "censored", "at_ceiling", "n", "measured_cap",
                   "share_at_measured_cap", "share_truncated_at_measured_cap", "n_at_cap",
                   "n_at_cap_not_truncated", "p99_completion")


def load_length_quantiles(path: Path) -> Tuple[Dict[str, Any], Path]:
    """``(report, items_path)`` for a ``length_quantiles.json``; the items
    file is the report's recorded path when it exists, else the sibling."""
    p = Path(os.path.expanduser(str(path)))
    if not p.is_file():
        raise CapLadderError(f"length quantiles: {p} does not exist")
    q = json.loads(p.read_text(encoding="utf-8"))
    ver = (q.get("schema") or {}).get("version") if isinstance(q.get("schema"), Mapping) else None
    if q.get("name") != "length_quantiles" or ver != 1 or not isinstance(q.get("entries"), Mapping):
        raise CapLadderError(
            f"length quantiles: {p} is not a length_quantiles/1 report (name={q.get('name')!r}, "
            f"schema.version={ver!r})")
    items = Path(str(q.get("items_file") or ""))
    if not items.is_file():
        items = p.with_name(LENGTH_QUANTILES_ITEMS_NAME)
    if not items.is_file():
        raise CapLadderError(f"length quantiles: no items file at {q.get('items_file')!r} nor beside {p}")
    return q, items


def cells_in_length_quantiles(path: Path, lane: str, readout: str) -> List[str]:
    q, _ = load_length_quantiles(path)
    pre = f"{lane}|{readout}|"
    return sorted(k[len(pre):] for k in q["entries"] if k.startswith(pre))


def size_cap_from_length_quantiles(path: Path, lane: str, readout: str, cell: str, *,
                                   trial: Optional[str] = None,
                                   threshold: float = AT_CAP_SHARE_THRESHOLD,
                                   ladder: Sequence[int] = CAP_LADDER,
                                   ceiling: int = CAP_CEILING) -> Dict[str, Any]:
    """:func:`size_cap` over the sampled lane's UNCENSORED final-rung rows for
    one (lane, readout, cell): the per-draw completion counts come from the
    items file, the measured cap is the trial's ``final_cap``, and the at-cap
    rows that are loops / repetitive / boundary draws come from the trial's
    ``censoring.residual_composition`` (``n_at_cap - n_truncated``), which is
    the ladder's own reclassify decision — so the sizing reads truncation the
    way the ladder does. ``trial=None`` pools every trial when they ended at
    the same final cap (``pooled`` True); when the trials climbed to different
    caps (the controls ladder is climbed per trial, and a share near 1% lands
    on either side of it) each trial is sized at its own cap and the cell
    takes the LARGEST recommendation (``pooled`` False, ``sized_on_trial``,
    ``per_trial``) — two valid measurements combined the way a run needs
    them, not a pool across caps. A named trial sizes that one.

    Refused, never approximated: an entry that is not ``ok`` (a trial or rung
    still pending), a censoring status outside
    :data:`LENGTH_QUANTILES_SIZEABLE`, and an items file whose row / at-cap
    counts disagree with the report's (a torn snapshot)."""
    q, items_path = load_length_quantiles(path)
    key = f"{lane}|{readout}|{cell}"
    entry = q["entries"].get(key)
    if entry is None:
        lanes = sorted({k.split("|")[0] for k in q["entries"]})
        cells = cells_in_length_quantiles(path, lane, readout)
        raise CapLadderError(
            f"length quantiles: no entry {key!r} (lanes: {lanes}; cells for {lane}|{readout}: {cells})")
    if entry.get("status") != "ok":
        raise CapLadderError(
            f"length quantiles: {key} is {entry.get('status')!r} (incomplete trials "
            f"{entry.get('incomplete_trials')}) — size it when its stores are complete")
    trials = entry.get("trials") or {}
    picked = [str(trial)] if trial is not None else sorted(trials)
    if not picked or any(t not in trials for t in picked):
        raise CapLadderError(f"length quantiles: {key} has trials {sorted(trials)}, asked for {picked}")
    caps = {int(trials[t]["final_cap"]) for t in picked}
    if len(caps) != 1:
        per = {t: size_cap_from_length_quantiles(path, lane, readout, cell, trial=t, threshold=threshold,
                                                 ladder=ladder, ceiling=ceiling) for t in picked}
        worst = max(picked, key=lambda t: (int(per[t]["recommended_cap"]), bool(per[t]["censored"]), t))
        out = dict(per[worst])
        out.update({"trials": picked, "pooled": False, "sized_on_trial": worst,
                    "censoring": {t: per[t]["censoring"][t] for t in picked},
                    "per_trial": {t: {k: per[t][k] for k in _PER_TRIAL_KEYS} for t in picked}})
        return out
    mc = caps.pop()
    excl = 0
    n_expected = 0
    n_at_cap_expected = 0
    censoring: Dict[str, Any] = {}
    for t in picked:
        tr = trials[t]
        c = tr.get("censoring") or {}
        status = c.get("status")
        if status not in LENGTH_QUANTILES_SIZEABLE:
            raise CapLadderError(
                f"length quantiles: {key} trial {t} censoring status {status!r} is not sizeable "
                f"(one of {LENGTH_QUANTILES_SIZEABLE}): "
                f"{(q.get('schema') or {}).get('censoring.status', {}).get(status, 'undocumented')}")
        n_cens = int(tr.get("n_censored_at_final_cap") or 0)
        rc = c.get("residual_composition") or {}
        if status == "uncensored":
            if n_cens != 0:
                raise CapLadderError(f"length quantiles: {key} trial {t} is 'uncensored' yet records "
                                     f"{n_cens} rows at the final cap")
            not_trunc = 0
        else:
            if int(rc.get("n_at_cap", -1)) != n_cens:
                raise CapLadderError(
                    f"length quantiles: {key} trial {t}: residual_composition.n_at_cap "
                    f"{rc.get('n_at_cap')} != n_censored_at_final_cap {n_cens}")
            not_trunc = n_cens - int(rc["n_truncated"])
        excl += not_trunc
        n_expected += int(tr["n"])
        n_at_cap_expected += n_cens
        censoring[t] = {"status": status, "decision": c.get("decision"),
                        "decision_path": c.get("decision_path"), "final_cap": int(tr["final_cap"]),
                        "n": int(tr["n"]), "n_at_cap": n_cens, "n_at_cap_not_truncated": not_trunc,
                        "residual_composition": dict(rc) if rc else None}
    want = set(picked)
    vals: List[int] = []
    n_at_cap_rows = 0
    with open(items_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("lane") != lane or r.get("readout") != readout or r.get("cell") != cell \
                    or str(r.get("trial")) not in want:
                continue
            tok = int(r["completion_tokens"])
            if bool(r.get("at_cap")):
                n_at_cap_rows += 1
                tok = mc                      # the producer's stamp: censored, at its cap value
            elif tok >= mc:
                raise CapLadderError(
                    f"length quantiles: {key} item {r.get('item')!r} trial {r.get('trial')} has "
                    f"{tok} tokens >= the final cap {mc} yet at_cap is false")
            vals.append(tok)
    if len(vals) != n_expected or n_at_cap_rows != n_at_cap_expected:
        raise CapLadderError(
            f"length quantiles: {key} items file has {len(vals)} rows / {n_at_cap_rows} at cap for "
            f"trials {picked}, the report says {n_expected} / {n_at_cap_expected} — a torn snapshot; "
            "re-run the report")
    s = size_cap(vals, mc, n_at_cap_not_truncated=excl, threshold=threshold, ladder=ladder,
                 ceiling=ceiling)
    tool = q.get("tool") or {}
    return {"lane": lane, "readout": readout, "cell": cell, "kind": entry.get("kind"),
            "trials": picked, "pooled": len(picked) > 1, "sized_on_trial": None, "per_trial": None,
            "source": str(Path(os.path.expanduser(str(path)))),
            "items_file": str(items_path),
            "producer": {k: tool.get(k) for k in ("name", "git_sha", "sha256", "written_at")},
            "censoring": censoring, **s.as_dict()}


def check_cap_against_sizing(cap: int, sizing: Mapping[str, Any], *,
                             controls: Sequence[str]) -> Dict[str, Any]:
    """The config-time gate: a run's cap must be at or above the sizing's
    recommendation for every judged control it generates. Returns the
    manifest block; raises :class:`CapLadderError` otherwise."""
    if sizing.get("schema") != SIZING_SCHEMA:
        raise CapLadderError(f"cap sizing: schema {sizing.get('schema')!r} != {SIZING_SCHEMA!r}")
    tasks = sizing.get("tasks") or {}
    checked: Dict[str, Any] = {}
    under: List[str] = []
    for c in controls:
        task = MEASURED_TASK_OF.get(c)
        if task is None:
            continue
        if task not in tasks:
            raise CapLadderError(
                f"cap sizing: no entry for {task!r} (control {c}) — has {sorted(tasks)}; size "
                "every judged control of the run before it starts (scripts/cap_ladder.py size)")
        rec = int(tasks[task]["recommended_cap"])
        checked[c] = {"task": task, "recommended_cap": rec, "censored": bool(tasks[task]["censored"]),
                      "share_at_measured_cap": tasks[task]["share_at_measured_cap"],
                      "measured_cap": tasks[task]["measured_cap"], "ok": int(cap) >= rec}
        if int(cap) < rec:
            under.append(f"{c}: cap {cap} < recommended {rec} ({tasks[task]['reason']})")
    if under:
        raise CapLadderError(
            "the run's cap is under the measured recommendation for "
            + "; ".join(under)
            + " — raise --max-new-tokens to the recommendation (the ladder then climbs from there)")
    return {"schema": SIZING_SCHEMA, "family": sizing.get("family"), "source": sizing.get("source"),
            "threshold": sizing.get("threshold"), "cap": int(cap), "controls": checked}


def load_sizing(path: Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise CapLadderError(f"cap sizing file {p} does not exist")
    return json.loads(p.read_text(encoding="utf-8"))
