"""The anchor currency of a pass: which CURRENT unsteered run anchors a contrast.

One read per pass (:meth:`AnchorCurrency.load`, cached on :class:`base.AdapterContext` by ``anchor_currency()``): the currency of
EVERY run from ``v_run_current`` -- the one predicate of a run's currency, never a ruling parser -- plus the unsteered ``run``
rows with their key and standup columns, their complete judgments, and EVERY ``run_metric`` row of theirs (6,186 live -- never only the
metrics that already carry a contrast: the first contrast at a new metric, and every contrast of a fresh registry, would otherwise find no
reading and park forever), held in memory. :meth:`AnchorCurrency.choose` then answers, for one intended contrast, which unsteered run of the EXACT key anchors it:

0. the adapter's OWN anchor when it is current and usable -- its pairing rule stands whatever the caps, trials or instruments of the
   pair (the ladder readers' exit-cap pairs, the serving-path twins), and when the adapter names the anchor's JUDGMENT
   (:attr:`AnchorKey.own_anchor_judgment_id`) the reading is read under that judgment -- a run with two complete judgments under one judge,
   told apart by session, is the adapter's pairing, not an ambiguity; otherwise ``own_reason`` names why and the search CONTINUES: a
   missing, unfolded or retired own id is a reason not to use THAT id, never a reason to skip the fallback;
1. the eligible unsteered runs at the exact key -- equal ``(model_id, eval_id, instrument_id)``, ``IS``-equal ``(token_cap_tokens,
   trial_index, pass_label)``, and equal ``decode_policy_id`` when BOTH sides carry one -- in tiers: (a) the one same-standup run
   (two or more refuse by name), (b) the same-source runs, (c) the paper's roster-matched in-paper runs of another source; inside a
   tier the roster-matched / in-paper candidates first, then the earliest complete ``(COALESCE(generated_utc, queued_utc), run_id)``
   -- the native convention, stable under ``coexist``;
2. nothing: ``no_current_anchor`` with the nearest miss named. Never a run at another cap, trial, pass, instrument or decode policy:
   the adapter files NO contrast and its previous key is withdrawn by the head.

Eligibility (every tier, the own id included): (e1) ``is_current = 1`` -- so never an excluded, superseded, lifecycle-retired or
HELD-OUT run (the strict rule: a hold-out keeps the RUN visible, never a contrast against it); (e2) complete
(``lifecycle_state`` in :data:`schema_def.COMPLETE_LIFECYCLE_STATES`); (e3) exactly ONE usable reading at ``(metric_id, threshold)``:
the run's one complete, non-superseded judgment under the steered row's judge, and under it the ``run_metric`` row whose
``computation_id`` equals the steered row's when one exists, else the one remaining row -- two judgments, or two remaining rows with
different values, are ambiguous and refuse by name (live, 346 unsteered ``(run, metric, threshold)`` groups carry several rows). A
reading is a ``run_metric`` row the paper frame counts: a row whose observation key is current under NO revision -- an adapter
revision withdrew the measurement while the RUN stayed current -- is no reading (:func:`withdrawn_metric_cte`, the predicate
``v_paper_long`` joins; a row no revision ever observed has no observation and is kept), so a current run with a withdrawn
reading refuses ``no_reading``, and a withdrawn computed twin cannot make the remaining row ambiguous. Nor is a row filed under a
judgment a re-judge displaced or a claim released a reading (status ``superseded`` / ``abandoned``, :data:`CURRENT_JUDGMENT_SQL`
-- the predicate the paper frame applies): the frame's judgment read keeps complete judgments only, and its metric read keeps only the
rows the judgment rule counts, so a displaced judgment's row is never a candidate reading, whatever the key order;
(e4) for a family whose statistic is paired over registry items, ``item_result`` rows at the metric under that judgment, none of them
carrying conflicting visible provenances -- known on the grading host only (``v_item_result_current`` for the rows, ``v_item_conflict`` for the
conflicts; a shard ships neither, so a paired-items family refuses there by name). A pair with a conflicted item -- two accepted
provenances of one measurement key that disagree, whichever one the current-row rank selects -- refuses by name
(``item_conflict``; the conflicted item count and ids named), exactly as the registry's paired-reading API refuses to pair it: the
rank's pick is not a reading a paired statistic may use while the sources disagree. The STEERED side is read the same way: an
item-paired comparison names its steered run and judgment, and their current item rows and conflicts at the metric are the other half
of the pair -- a conflicted steered item parks the comparison whatever the pool holds (the paired-reading API checks both runs).

Every returned run id came from the frame: a chosen anchor is a folded run with a folded judgment, so the contrast never
dependency-blocks its revision. Nothing here writes; nothing here reads a ruling.

THE ANCHOR DECISION IS AN INPUT OF THE READ. Nothing re-reads an adapter when a RULING changes -- rulings fold into
``ruling``, no file of the group moves, and ``plan_groups`` finds every group quiet -- so the decision set of a comparison group is a
synthetic :class:`base.Input` of that group (:class:`AnchorDecisionInput`: ``registry://anchor-decision/<source_id>``, its ``sha256``
the :func:`decision_fingerprint` over every comparison the read made, parked ones included). A *comparison* is one contrast the
adapter intends -- the steered ``(run_id, judgment_id, metric_id, threshold, computation_id)``, its :class:`AnchorKey` and
``judge_id`` (:func:`comparison_record`); its *decision* is :meth:`AnchorCurrency.choose`'s outcome -- the chosen anchor run, the
reading it anchors with (its identity AND its statistic: a ``run_metric`` row corrected in place keeps its run, judgment and computation
ids and moves ``value`` / ``k`` / ``n``, and the contrast's delta with them, so the identity alone would replay to the same fingerprint
and leave a wrong delta filed), for an item-paired comparison the digest of the anchor's current item rows, or ``"none"`` with
``own_reason`` / ``why`` when the comparison parks (:func:`decision_record`); for an item-paired comparison also the STEERED side's
state -- the digest of the steered row's current item rows (``steered_items_digest``; ``None`` until they fold, so the first pass after
the fold re-reads and re-derives the contrast over the registry's rows) and its conflicted item count (``steered_item_conflicts``) --
because the paired statistic is computed from both sides' rows and none of them is a file input of the group. A family whose paired
statistic reads item rows filed at ANOTHER metric than the contrast's (the store readers: a refused rate, a label mean and a graded mean
over per-item LABEL rows at the control's label metric) names it on the comparison (``items_metric_id``), and e4 reads that metric on
both sides; such a reader also checks the rows it re-read from the store against the digest the decision carries
(:func:`item_rows_digest`), so a contrast is never filed from rows the registry has not accepted. A family whose paired statistic reads
per-item evidence that NO item row carries -- the store readers' graded mean scores an item by its label AND by whether it ran to the
token cap, and the cap reading is filed per item as ``item_flags.at_token_cap``, never on a label row -- asks for it too
(``needs_flags``, recorded on the comparison): the decision then also carries the digest of each side's current ``item_flags`` rows under
the pair's judgment (``flags_digest`` / ``steered_flags_digest``: :data:`FLAG_COLUMNS`, ``v_item_flags_current``, one indexed probe per
side), and the reader checks the store's flags against it before filing (:func:`flag_rows_digest`). A cap flag corrected on, or exchanged
between two equally labeled items of, the unsteered store keeps its aggregate and its label rows and moves the paired statistic; without
the flags in the decision the dependent group would replay quiet and keep a wrong graded contrast. The item-conflict state
enters the fingerprint through the DECISION, not the digests: a pair whose items carry conflicting visible provenances, on either side,
never anchors (e4 refuses an anchor by name; a conflicted steered side parks the comparison), so a second accepted provenance that
disagrees with a selected row -- which leaves the row ``v_item_result_current`` selects, and with it the digest of the selected rows,
unchanged -- moves the comparison to a park whose ``own_reason`` / ``why`` name the conflict, or onto another anchor, and the
fingerprint with it. ``choose``
appends each pair to the context's ``decision_log`` (:meth:`base.AdapterContext.begin_group` clears it before each read); the runner
records the set on the revision marker (``extra.anchor_decisions``), writes the synthetic input's watermark from the post-read
fingerprint, and before the next pass's planning replays every recorded comparison against the pass's frame
(:meth:`AnchorCurrency.decisions_of`) -- a fingerprint that moved is a changed input, so the ordinary identity machinery
(``revision_id_for``, ``inputs_json``, the withdrawals of the previous head's keys, the cluster's unknown-id rule) carries a ruling
change exactly like a changed file. The fingerprint is over the DECISION (which anchor, which reading, why), never over the audit
trail (the candidate and rejected lists of ``extra.anchor_selection``): a new unsteered run that does not move the pick changes the
record's trail but not the revision -- a per-column change the next read of the group carries.
"""
from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from .. import identity as I
from .. import schema_def as S
from .. import vocab as V
from .base import Input

#: the rule stamped on every ``extra.anchor_selection`` block this module derives
RULE = "anchor_current/1"
#: the rule stamped on every ``extra.anchor_decisions`` marker block: the comparison / decision shapes and the fingerprint.
#: 5: a second comparison KIND -- the currency-only check of a STEERED anchor (:data:`CURRENCY_KIND`, r2_export's
#: `full_dose` 1x reading), recorded and replayed beside the unsteered-anchor comparisons.
#: 6: an item-paired comparison may name the metric of the ITEM ROWS its statistic reads when they are not
#: filed at the contrast's own metric (``items_metric_id``: the store readers' refused rate, label mean and graded mean are read off
#: per-item LABEL rows filed at the control's label metric, ``xst_label`` / ``orb_label``); the digests, the presence check and the
#: conflict check of e4 read that metric. A comparison without the field (a rule-5 record) reads its own metric, as before.
#: 7: an item-paired comparison may also ask for the per-item CAP evidence (``needs_flags``, recorded on the
#: comparison): the decision carries the digest of each side's current ``item_flags`` rows under the pair's judgment (``flags_digest`` /
#: ``steered_flags_digest``, :data:`FLAG_COLUMNS` -- the fold's ``at_token_cap`` per item, which no label row carries and the store readers'
#: graded mean scores by). A comparison without the field (a rule-6 record) carries both as ``None``, as before
DECISION_RULE = "anchor_decision/7"
#: the ``kind`` of a recorded comparison that checks the CURRENCY of an anchor that is not an unsteered run and is never re-picked
#: (:meth:`AnchorCurrency.check_currency`); a comparison without a ``kind`` is an unsteered-anchor comparison (:func:`comparison_record`)
CURRENCY_KIND = "anchor_currency"
#: the ``v_item_result_current`` columns an item-paired decision digests, beside ``(run_id, judgment_id)`` at the metric: the item's
#: identity within the run and the content a paired statistic reads (``item_result``'s key minus the run / metric / judgment, plus value and label)
ITEM_COLUMNS: Tuple[str, ...] = ("item_id", "trial_index", "sample_index", "turn_index", "value", "label")
#: the ``v_item_flags_current`` columns a comparison that asks the per-item cap evidence (``needs_flags``) digests, beside ``(run_id,
#: judgment_id)``: the item's identity within the run and the fold's per-item cap reading (``at_token_cap``: the one flag the store readers'
#: graded score reads -- an item at the cap scores 0.5 whatever its label). ``item_flags`` is keyed without a metric, so the digest is per pair
FLAG_COLUMNS: Tuple[str, ...] = ("item_id", "trial_index", "sample_index", "turn_index", "at_token_cap")
#: the ``Input.role`` of the synthetic anchor-decision input; the runner skips this role for artifacts, state rows and the adapter's read
DECISION_ROLE = "anchor_decision"
#: the URI prefix of the synthetic input: ``registry://anchor-decision/<source_id>``
DECISION_URI_PREFIX = "registry://anchor-decision/"
#: the decision's ``anchor_run_id`` when the comparison parks (``no_current_anchor``)
PARKED = "none"
#: the synthetic input's ``mtime_utc``: no file, no mtime -- the epoch, a constant, because ``source_watermark.mtime_utc`` is required and
#: must be a well-formed UTC stamp (the ``_utc`` column rule), and the watermark compares ``(sha256, size, mtime_utc)``, so only the
#: fingerprint moves it
DECISION_MTIME = "1970-01-01T00:00:00.000Z"
#: the order inside a tier, as the selection block states it
ORDER = "roster-matched first, then earliest complete (generated_utc, run_id)"
#: :attr:`AnchorChoice.state`
STATES: Tuple[str, ...] = ("own_current", "sibling_same_source", "borrowed_reference", "no_current_anchor")
#: :attr:`AnchorChoice.own_reason` -- why the adapter's own anchor id was kept (``current``) or was not
OWN_REASONS: Tuple[str, ...] = ("current", "superseded", "excluded", "held_out", "withdrawn", "lifecycle", "no_reading", "ambiguous",
                                "item_conflict", "not_unsteered", "not_in_frame", "none_declared")
#: how many conflicted item ids an ``item_conflict`` refusal names (the rest is a count): a bounded list, as the paired-reading API's refusal names one
CONFLICT_NAMED = 5
KIND_SAME_STANDUP, KIND_SIBLING, KIND_BORROWED = "same_standup_none", "sibling_none", "borrowed"
assert {KIND_SAME_STANDUP, KIND_SIBLING, KIND_BORROWED} <= set(V.ANCHOR_KINDS)
#: the sentinel ``run_metric.judgment_id`` of a row filed without a judgment (rule-scored evals)
NO_JUDGMENT = ""
#: two readings whose values differ by more than this are two readings (the twin check of the ownership rules uses the same tolerance)
VALUE_TOL = 1e-9
#: the columns of ``v_run_current`` the frame holds (the sidecar ships exactly these, reader_state.RUN_CURRENT_COLUMNS)
CURRENCY_COLUMNS: Tuple[str, ...] = ("run_id", "is_current", "why", "held_out", "in_current_paper", "selection_state", "roster_entry_ids_json")
#: the unsteered run columns the key, the standup and the tie-break read
RUN_COLUMNS: Tuple[str, ...] = ("run_id", "model_id", "eval_id", "instrument_id", "token_cap_tokens", "trial_index", "pass_label",
                                "decode_policy_id", "lane_source_legacy", "view_legacy", "cluster_job_id", "producer_run_key_legacy",
                                "lifecycle_state")

#: the paper frame's join of a ``run_metric`` row (alias ``rm``) to its withdrawn observation (``wm``): schema_def's own text, read here so
#: the loader's predicate and ``v_paper_long``'s cannot drift (``WHERE wm.run_id IS NULL`` keeps a row; a test pins the identity)
WITHDRAWN_METRIC_JOIN = S._WITHDRAWN_JOIN
#: the paper frame's judgment-currency rule (migration 0021; ``schema_def.NON_CURRENT_JUDGMENT_STATUSES``) over a ``run_metric``
#: row (alias ``rm``) LEFT-joined to its judgment (alias ``jm``, :func:`current_judgment_join`): TRUE for a row the frame counts -- no
#: judgment, no judgment row, or a status that is not displaced / released. schema_def's own text, so the loader, the sidecar export and
#: em_misalign's paired reader cannot drift from ``v_paper_long``
CURRENT_JUDGMENT_SQL = S.non_current_judgment_sql("rm.judgment_id", "jm.status")


def current_judgment_join(schema: str = "") -> str:
    """The LEFT JOIN that puts a ``run_metric`` row's judgment status beside it as ``jm.status`` (what :data:`CURRENT_JUDGMENT_SQL`
    reads); ``schema`` qualifies the ``judgment`` read the way :func:`withdrawn_metric_cte` qualifies the ``observation`` read (the
    sidecar export reads the LIVE judgments through an attached schema while the sidecar under construction already holds a
    ``judgment`` table of the complete rows only)."""
    return f"LEFT JOIN {schema + '.' if schema else ''}judgment jm ON jm.judgment_id = rm.judgment_id"


def withdrawn_metric_cte(schema: str = "") -> str:
    """:data:`schema_def.WITHDRAWN_METRIC_CTE` -- the paper frame's withdrawn-reading predicate, verbatim: a ``run_metric`` row whose
    observation key is current under no revision -- with its ``observation`` read qualified by ``schema`` when one is given (the sidecar
    export reads the LIVE database's observations through an attached schema while the sidecar under construction already holds an
    ``observation`` table of its own, which an unqualified name would resolve to). Refuses by name if the constant's text no longer
    carries exactly the read this qualifies, rather than qualifying nothing and reading the wrong table."""
    text = S.WITHDRAWN_METRIC_CTE
    if not schema:
        return text
    needle = "FROM observation WHERE"
    if text.count(needle) != 1:
        raise RuntimeError(f"schema_def.WITHDRAWN_METRIC_CTE reads {needle!r} {text.count(needle)} times, not once: the predicate "
                           f"qualified by {schema!r} cannot be built")
    return text.replace(needle, f"FROM {schema}.observation WHERE")


class AnchorCurrencyUnavailable(RuntimeError):
    """``v_run_current`` (or the sidecar's table of it) is absent, or there is no connection: the GROUP refuses by name, never the
    contrast alone -- a contrast-less revision would withdraw every contrast of the group."""


@dataclass(frozen=True)
class AnchorKey:
    """What identifies the anchor a steered row needs: the exact key, the steered row's source and standup, and the adapter's own pick."""
    model_id: str
    eval_id: str
    instrument_id: str
    token_cap_tokens: Optional[int]            #: the STEERED row's drawn cap (`run.token_cap_tokens`), compared IS
    trial_index: Optional[int]                 #: compared IS
    pass_label: Optional[str]                  #: compared IS
    decode_policy_id: Optional[str]            #: compared only when BOTH sides carry one
    lane_source_legacy: Optional[str]          #: the steered row's source: the same-source tier
    standup: Mapping[str, Optional[str]]       #: run columns identifying the adapter's OWN standup ({"view_legacy": v, "lane_source_legacy": s} or {"cluster_job_id": j})
    own_anchor_run_id: Optional[str] = None    #: the id the adapter composes / scans / declares today; None when it has none
    own_anchor_kind: Optional[str] = None      #: the adapter's own label for it, kept verbatim when the own anchor is current
    #: the JUDGMENT of the own anchor the adapter pairs against, when it names one (the declared readers' ``judgment_by_run``; a
    #: second judgment of a run under the SAME judge, told apart by session). Step 0 reads the own anchor's reading under THIS judgment
    #: instead of searching the run's judgments by judge: the adapter's pairing names a judgment, and two complete judgments under one
    #: judge are its design, not an ambiguity. It must be a complete, non-superseded judgment of the own run under the row's judge, else
    #: the own anchor is not usable (``no_reading``, named). Tiers 1-2 (a re-pick) never read it: a re-picked anchor's judgment is found by judge
    own_anchor_judgment_id: Optional[str] = None

    @property
    def exact(self) -> Tuple[Any, ...]:
        return (self.model_id, self.eval_id, self.instrument_id, _opt_int(self.token_cap_tokens), _opt_int(self.trial_index), self.pass_label)


@dataclass(frozen=True)
class Candidate:
    """One eligible unsteered run at the exact key (the tie-break's input)."""
    run_id: str
    lane_source_legacy: Optional[str]
    view_legacy: Optional[str]
    cluster_job_id: Optional[str]
    producer_run_key_legacy: str
    complete_utc: Optional[str]                #: COALESCE(generated_utc, queued_utc)
    in_current_paper: int
    roster_matched: int                        #: roster_entry_ids_json <> '[]'
    same_standup: bool
    same_source: bool


@dataclass(frozen=True)
class Measurement:
    """The anchor's ONE usable reading at ``(metric_id, threshold)``: the full metric identity and its provenance, verbatim."""
    run_id: str
    judgment_id: str
    computation_id: str
    metric_id: str
    threshold: float
    k: Optional[float]
    n: Optional[int]
    value: Optional[float]
    ci_lo: Optional[float]
    ci_hi: Optional[float]
    judge_id: Optional[str]
    judgment_status: str
    computed_by: str
    source_uri: Optional[str]

    def reading_json(self) -> Dict[str, Any]:
        return {"judgment_id": self.judgment_id, "computation_id": self.computation_id, "computed_by": self.computed_by, "source_uri": self.source_uri}

    def statistic_json(self) -> Dict[str, Any]:
        """The reading's statistic-affecting content (what a contrast's delta and interval are computed from): ``value``, ``k``, ``n`` and the
        filed interval. What :func:`decision_record` fingerprints beside the reading's identity."""
        return {"value": self.value, "k": self.k, "n": self.n, "ci_lo": self.ci_lo, "ci_hi": self.ci_hi}


@dataclass(frozen=True)
class AnchorChoice:
    """:meth:`AnchorCurrency.choose`'s answer. ``why`` is the retired own anchor's ``v_run_current.why`` when a fallback was taken,
    the refusal text when nothing anchors, ``''`` on an own-current pick; ``own_why`` is always the own anchor's reason text (``''``
    when it was kept)."""
    state: str
    own_reason: str
    run_id: Optional[str]
    anchor_kind: Optional[str]
    cross_source: bool
    why: str
    own_why: str
    n_candidates: int
    candidates: Tuple[Candidate, ...]                 #: the eligible exact-key runs, in tie-break order
    rejected: Tuple[Tuple[str, str], ...]             #: the exact-key unsteered runs that failed e1-e4, (run_id, why), by run id
    anchor_judgment_id: str                           #: the chosen anchor's one complete judgment under the judge, else ''
    measurement: Optional[Measurement]                #: a chosen anchor ALWAYS carries one
    key: AnchorKey
    #: for a comparison that asked e4 (``needs_items``): the sha256 over the chosen anchor's CURRENT item rows at the metric under the
    #: reading's judgment (:meth:`AnchorCurrency.items_digest`) -- the content a statistic paired over items reads; ``None`` otherwise.
    #: Never a conflicted pair's: e4 refuses one by name (:meth:`AnchorCurrency.items_conflicts`), so the digest never stands in for a disagreement
    items_digest: Optional[str] = None
    #: for a comparison that asked e4 on a frame that knows items: the sha256 over the STEERED row's current item rows at the metric under its
    #: judgment (:meth:`AnchorCurrency.pair_items_digest`; ``None`` while the steered rows are not folded, or no items) and the count of its
    #: conflicted item keys (:meth:`AnchorCurrency.pair_items_conflicts`); a non-zero count parks the comparison. ``None`` / ``None`` otherwise
    steered_items_digest: Optional[str] = None
    steered_item_conflicts: Optional[int] = None
    #: for a comparison that asked the per-item cap evidence (``needs_flags``) on a frame that knows flags: the sha256 over the chosen
    #: anchor's CURRENT ``item_flags`` rows under the reading's judgment (:meth:`AnchorCurrency.pair_flags_digest`; ``None`` when the pair
    #: has none) and over the STEERED row's under its judgment (``None`` while they are not folded); ``None`` / ``None`` otherwise
    flags_digest: Optional[str] = None
    steered_flags_digest: Optional[str] = None

    def __post_init__(self) -> None:
        if self.state not in STATES:
            raise ValueError(f"anchor choice state {self.state!r} is not one of {STATES}")
        if self.own_reason not in OWN_REASONS:
            raise ValueError(f"anchor own_reason {self.own_reason!r} is not one of {OWN_REASONS}")
        if self.anchor_kind is not None and self.anchor_kind not in V.ANCHOR_KINDS:
            raise ValueError(f"anchor_kind {self.anchor_kind!r} is not one of {V.ANCHOR_KINDS}")
        if (self.run_id is None) != (self.state == "no_current_anchor") or (self.run_id is not None and self.measurement is None):
            raise ValueError(f"an anchor choice of state {self.state!r} carries run_id {self.run_id!r} and measurement {self.measurement!r}")

    def selection_json(self) -> Dict[str, Any]:
        """The ``extra.anchor_selection`` block: deterministic, no timestamp (the record id is a uuid5 over the payload)."""
        return {"state": self.state, "own_reason": self.own_reason, "own_anchor_run_id": self.key.own_anchor_run_id, "own_anchor_why": self.own_why,
                "why": self.why, "n_candidates": self.n_candidates, "candidates": [c.run_id for c in self.candidates],
                "rejected": [[rid, why] for rid, why in self.rejected],
                "key": {"token_cap_tokens": _opt_int(self.key.token_cap_tokens), "trial_index": _opt_int(self.key.trial_index),
                        "pass_label": self.key.pass_label, "decode_policy_id": self.key.decode_policy_id},
                "reading": (self.measurement.reading_json() if self.measurement is not None else None),
                "order": ORDER, "rule": RULE}


# ---- the anchor decision as an input of the read -------------------------------------------------------------------------

@dataclass(frozen=True)
class AnchorDecisionInput(Input):
    """The synthetic input that carries a comparison group's anchor decision set: ``uri = identity_uri =
    registry://anchor-decision/<source_id>``, ``sha256`` = the set's :func:`decision_fingerprint`, no path, size 0, the constant
    :data:`DECISION_MTIME`, role :data:`DECISION_ROLE`. ``base.watermark_of`` reads ``(sha256, size, mtime_utc)``, so the fingerprint IS the watermark, and
    ``runner.revision_id_for`` folds it through ``inputs_json`` exactly like a file's sha. It never goes through
    ``adapter.comparison_group`` (a path-grouped adapter would mint a phantom group from the URI): the runner attaches it to its group
    by name (``plan_groups(decision_inputs=...)``), skips it for artifacts, state rows and the adapter's ``read``, and writes its
    watermark from the POST-read fingerprint."""

    @classmethod
    def of(cls, source_id: str, fingerprint: str) -> "AnchorDecisionInput":
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            raise ValueError(f"an anchor decision fingerprint is 64 hex characters (sha256), not {fingerprint!r}")
        uri = decision_uri(source_id)
        return cls(uri=uri, path="", sha256=fingerprint, size=0, mtime_utc=DECISION_MTIME, role=DECISION_ROLE, identity_uri=uri)


def decision_uri(source_id: str) -> str:
    return DECISION_URI_PREFIX + str(source_id)


def is_decision_uri(uri: Any) -> bool:
    return isinstance(uri, str) and uri.startswith(DECISION_URI_PREFIX)


def is_decision_input(inp: Input) -> bool:
    return getattr(inp, "role", None) == DECISION_ROLE or is_decision_uri(getattr(inp, "identity_uri", None))


def decision_source_id(uri: str) -> str:
    """The source id a decision URI names (``registry://anchor-decision/<source_id>`` -> ``<source_id>``)."""
    if not is_decision_uri(uri):
        raise ValueError(f"{uri!r} is not an anchor decision URI ({DECISION_URI_PREFIX}<source_id>)")
    return uri[len(DECISION_URI_PREFIX):]


def comparison_record(key: AnchorKey, metric_id: str, threshold: Any, judge_id: Optional[str], *, computation_id: str = "",
                      needs_items: bool = False, steered_run_id: Optional[str] = None, steered_judgment_id: Optional[str] = None,
                      items_metric_id: Optional[str] = None, needs_flags: bool = False) -> Dict[str, Any]:
    """One comparison the adapter intends, as the marker records it: everything :meth:`AnchorCurrency.choose` needs to decide it
    again on another frame (the key, the metric identity, the judge, the e3 computation tie-break, the e4 ask and, for an e4 ask whose
    item rows are filed at another metric, that metric: ``items_metric_id``, ``None`` when e4 reads the contrast's own; the per-item cap
    evidence ask, ``needs_flags``, ``False`` without e4) beside the steered row's identity. JSON-safe and deterministic (the fingerprint
    and the marker both read it)."""
    return {"run_id": steered_run_id, "judgment_id": steered_judgment_id, "metric_id": str(metric_id), "threshold": _thr(threshold),
            "computation_id": str(computation_id or ""), "judge_id": judge_id, "needs_items": bool(needs_items),
            "items_metric_id": (str(items_metric_id) if (needs_items and items_metric_id) else None), "needs_flags": bool(needs_items and needs_flags),
            "key": {"model_id": key.model_id, "eval_id": key.eval_id, "instrument_id": key.instrument_id,
                    "token_cap_tokens": _opt_int(key.token_cap_tokens), "trial_index": _opt_int(key.trial_index), "pass_label": key.pass_label,
                    "decode_policy_id": key.decode_policy_id, "lane_source_legacy": key.lane_source_legacy,
                    "standup": {str(c): (None if v is None else str(v)) for c, v in sorted(dict(key.standup or {}).items())},
                    "own_anchor_run_id": key.own_anchor_run_id, "own_anchor_kind": key.own_anchor_kind,
                    "own_anchor_judgment_id": (key.own_anchor_judgment_id or None)}}


def key_of_comparison(rec: Mapping[str, Any]) -> AnchorKey:
    """The :class:`AnchorKey` a recorded comparison carries (the inverse of :func:`comparison_record`)."""
    k = rec["key"]
    return AnchorKey(model_id=str(k["model_id"]), eval_id=str(k["eval_id"]), instrument_id=str(k["instrument_id"]),
                     token_cap_tokens=_opt_int(k.get("token_cap_tokens")), trial_index=_opt_int(k.get("trial_index")), pass_label=k.get("pass_label"),
                     decode_policy_id=k.get("decode_policy_id"), lane_source_legacy=k.get("lane_source_legacy"), standup=dict(k.get("standup") or {}),
                     own_anchor_run_id=k.get("own_anchor_run_id"), own_anchor_kind=k.get("own_anchor_kind"),
                     own_anchor_judgment_id=k.get("own_anchor_judgment_id"))


def decision_record(choice: AnchorChoice) -> Dict[str, Any]:
    """The DECISION of one comparison, as the fingerprint reads it: the chosen anchor run (:data:`PARKED` when nothing anchors), the
    state and the own anchor's reason, the kind, the reading it anchors with -- its identity (the judgment and the computation: a
    re-judged anchor is a changed decision even when the run stays) AND its statistic (:meth:`Measurement.statistic_json`: a reading
    corrected in place keeps every id and moves the value the contrast's delta is computed from, so the identity alone would replay to
    the same fingerprint and the stale delta would stand) -- for an item-paired comparison the digest of the anchor's current item rows
    (``items_digest``; a paired statistic reads them, not the aggregate; a pair with a conflicted item never anchors, so its conflict reaches
    the record as the park's ``own_reason`` / ``why`` or as another anchor, never as a digest) AND the steered side's: the digest of the
    steered row's current item rows (``steered_items_digest``: the other half of the pair, no file input of the group, so a corrected or
    newly folded steered row is a changed decision) and its conflicted item count (``steered_item_conflicts``: non-zero parks), for a
    comparison that asked the per-item cap evidence the digests of both sides' current ``item_flags`` rows (``flags_digest`` /
    ``steered_flags_digest``: the cap reading a graded statistic scores by, which no label row carries, so a cap flag moved between two
    equally labeled items is a changed decision though the label digests and the reading stand), and ``why``
    (the park reason that lands on the metric row's ``delta_block``, or the retired own anchor's reason). Never the candidate / rejected
    lists: those are the record's audit trail, not its decision. A parked comparison carries ``reading``, ``items_digest`` and
    ``flags_digest`` as ``None``; the steered fields are ``None`` for a comparison that did not ask e4 and on a frame that knows no items;
    both flags digests are ``None`` for a comparison that did not ask them and on a frame that knows no flags."""
    m = choice.measurement
    return {"anchor_run_id": choice.run_id if choice.run_id is not None else PARKED, "state": choice.state, "own_reason": choice.own_reason,
            "anchor_kind": choice.anchor_kind, "anchor_judgment_id": choice.anchor_judgment_id,
            "computation_id": (m.computation_id if m is not None else None), "reading": (m.statistic_json() if m is not None else None),
            "items_digest": choice.items_digest, "steered_items_digest": choice.steered_items_digest,
            "steered_item_conflicts": choice.steered_item_conflicts, "flags_digest": choice.flags_digest,
            "steered_flags_digest": choice.steered_flags_digest, "why": choice.why}


def decision_entry(comparison: Mapping[str, Any], decision: Mapping[str, Any]) -> Dict[str, Any]:
    return {"comparison": dict(comparison), "decision": dict(decision)}


def sorted_entries(entries: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """The decision set in its canonical order (by the canonical JSON of each entry): what the fingerprint hashes and the marker records."""
    return sorted((dict(e) for e in entries), key=I.canonical_json)


def decision_fingerprint(entries: Sequence[Mapping[str, Any]]) -> str:
    """``sha256(canonical_json(sorted [(comparison, decision)]))`` -- 64 hex characters, the synthetic input's ``sha256``."""
    return hashlib.sha256(I.canonical_json(sorted_entries(entries)).encode("utf-8")).hexdigest()


def decisions_block(entries: Sequence[Mapping[str, Any]], source_id: str, pre_read_fingerprint: Optional[str] = None) -> Dict[str, Any]:
    """The ``extra.anchor_decisions`` block of a revision marker: the complete comparison set of the read (parked ones included), its
    fingerprint, the synthetic input's URI and -- when the read ran with a decision input in its inputs -- the pre-read fingerprint the
    revision id was minted from, so a drift between the two is on the record."""
    ordered = sorted_entries(entries)
    return {"rule": DECISION_RULE, "fingerprint": decision_fingerprint(ordered), "input_uri": decision_uri(source_id),
            "n_comparisons": len(ordered), "n_parked": sum(1 for e in ordered if e["decision"].get("anchor_run_id") == PARKED),
            "pre_read_fingerprint": pre_read_fingerprint, "comparisons": ordered}


# ---- a currency-only decision on a STEERED anchor (r2_export: the `full_dose` 1x reading of the same object) ------------------
#
# The export's second contrast kind anchors a half-dose row on the SAME object's 1x run -- a steered run, never an unsteered one, so the
# exact-key search of `choose` does not apply and nothing is re-picked: the anchor is checked for currency and the contrast files (current)
# or parks (retired, or not folded yet). The check is a comparison of the read like any other -- the pass that creates both runs cannot anchor
# on an unfolded 1x run (no provisional anchors), so the park must be in the recorded decision set for the anchor-decision input to re-read
# the group once the 1x run folds -- recorded under its own `kind` and replayed by `AnchorCurrency.replay` against the pass's frame.

@dataclass(frozen=True)
class CurrencyCheck:
    """:meth:`AnchorCurrency.check_currency`'s answer: the anchor's currency from the fold, ``state`` ``own_current`` (file) or
    ``no_current_anchor`` (park), ``own_reason`` the fold's reason class (``current``, or ``not_in_frame`` for an id the registry never
    folded, else :func:`_classify_why` of ``v_run_current.why``), ``why`` the fold's text (``''`` when current)."""
    anchor_run_id: str
    anchor_kind: str
    is_current: Optional[bool]              #: None = not a run the registry has folded
    state: str
    own_reason: str
    why: str

    def __post_init__(self) -> None:
        if self.state not in ("own_current", "no_current_anchor") or self.own_reason not in OWN_REASONS:
            raise ValueError(f"currency check state {self.state!r} / own_reason {self.own_reason!r} out of vocabulary")
        if self.anchor_kind not in V.ANCHOR_KINDS:
            raise ValueError(f"anchor_kind {self.anchor_kind!r} is not one of {V.ANCHOR_KINDS}")

    def selection_json(self) -> Dict[str, Any]:
        """The ``extra.anchor_selection`` block of a currency-checked contrast (the same keys as :meth:`AnchorChoice.selection_json`
        so every contrast's block reads alike): no key search, no candidates, no reading lookup -- the anchor is the adapter's own steered run."""
        return {"state": self.state, "own_reason": self.own_reason, "own_anchor_run_id": self.anchor_run_id, "own_anchor_why": self.why,
                "why": (self.why if self.state == "no_current_anchor" else ""), "n_candidates": 0, "candidates": [], "rejected": [], "key": None,
                "reading": None, "order": ORDER, "rule": RULE, "anchor_kind": self.anchor_kind,
                "currency_only": "the anchor is a steered run of the same object (its 1x reading): its currency is checked, never re-picked"}


def currency_comparison_record(anchor_run_id: str, anchor_kind: str, metric_id: str, threshold: Any, *, computation_id: str = "",
                               steered_run_id: Optional[str] = None, steered_judgment_id: Optional[str] = None) -> Dict[str, Any]:
    """The recorded comparison of a currency-only check: ``kind`` :data:`CURRENCY_KIND`, the steered metric identity and the anchor's id and kind
    -- everything :meth:`AnchorCurrency.replay` needs to decide it again on another frame."""
    return {"kind": CURRENCY_KIND, "run_id": steered_run_id, "judgment_id": steered_judgment_id, "metric_id": str(metric_id), "threshold": _thr(threshold),
            "computation_id": str(computation_id or ""), "anchor_run_id": str(anchor_run_id), "anchor_kind": str(anchor_kind)}


def currency_decision_record(check: CurrencyCheck) -> Dict[str, Any]:
    """The decision of a currency-only check, in :func:`decision_record`'s shape (the marker's ``n_parked`` and the fingerprint read one shape):
    the anchor run (:data:`PARKED` when it is not current), the state, the reason class, the kind, no reading, no items."""
    return {"anchor_run_id": check.anchor_run_id if check.state == "own_current" else PARKED, "state": check.state, "own_reason": check.own_reason,
            "anchor_kind": check.anchor_kind, "anchor_judgment_id": "", "computation_id": None, "reading": None, "items_digest": None,
            "steered_items_digest": None, "steered_item_conflicts": None, "flags_digest": None, "steered_flags_digest": None, "why": check.why}


def is_currency_comparison(comparison: Mapping[str, Any]) -> bool:
    return comparison.get("kind") == CURRENCY_KIND


@dataclass(frozen=True)
class _Currency:
    is_current: int
    why: Optional[str]
    held_out: int
    in_current_paper: int
    selection_state: Optional[str]
    roster_matched: int


@dataclass(frozen=True)
class _Judgment:
    judgment_id: str
    run_id: str
    judge_id: Optional[str]
    status: str
    superseded_by: Optional[str]


def _opt_int(v: Any) -> Optional[int]:
    return None if v is None else int(v)


def _thr(v: Any) -> float:
    return -1.0 if v is None else float(v)


def _schema_version(conn: sqlite3.Connection, objects: Mapping[str, str]) -> str:
    try:
        if "schema_migration" in objects:
            v = conn.execute("SELECT MAX(version) FROM schema_migration").fetchone()[0]
            return str(v) if v is not None else "none"
        if "reader_state_meta" in objects:
            v = conn.execute("SELECT schema_version FROM reader_state_meta").fetchone()[0]
            return str(v) if v is not None else "none"
    except sqlite3.Error:
        pass
    return "unknown"


class AnchorCurrency:
    """The pass's frame: the currency of every run plus the unsteered candidates, their judgments and their contrast-metric readings."""

    def __init__(self, currency: Dict[str, _Currency], runs: Dict[str, Dict[str, Any]], judgments: Dict[str, List[_Judgment]],
                 metrics: Dict[Tuple[str, str, str, float], List[Measurement]], conn: Optional[sqlite3.Connection], items_known: bool,
                 counts: Dict[str, int], items_unknown_why: str = "", flags_known: bool = False, flags_unknown_why: str = ""):
        self._currency = currency
        self._runs = runs
        self._judgments = judgments
        self._metrics = metrics
        self._conn = conn
        self.items_known = items_known
        #: why the frame knows no items (``''`` when it does): what an e4 refusal names, so a shard's shape and a half-defined database read differently
        self.items_unknown_why = items_unknown_why if not items_known else ""
        #: whether the frame knows the per-item cap flags (``v_item_flags_current``: the grading host; a shard ships none) and why not when it does not:
        #: what a ``needs_flags`` refusal names
        self.flags_known = flags_known
        self.flags_unknown_why = flags_unknown_why if not flags_known else ""
        self.counts = counts
        #: the comparison group's decision log: the context's list, bound by :meth:`base.AdapterContext.anchor_currency`;
        #: :meth:`choose` appends ``{"comparison", "decision"}`` to it when bound (``None`` on a frame nobody reads through a context)
        self.decision_log: Optional[List[Dict[str, Any]]] = None
        self._by_key: Dict[Tuple[Any, ...], List[str]] = {}
        for rid, row in runs.items():
            self._by_key.setdefault(_row_key(row), []).append(rid)
        for ids in self._by_key.values():
            ids.sort()
        self._items: Dict[str, Dict[Tuple[str, str], str]] = {}   # metric_id -> {(run_id, judgment_id): digest of its current item rows} (lazy, the grading host only)
        self._conflicts: Dict[str, Dict[Tuple[str, str], List[str]]] = {}   # metric_id -> {(run_id, judgment_id): conflicted item ids, key order} (lazy, the grading host only)
        # ONE pair's item state by an indexed probe (the steered side of an item-paired comparison): (metric_id, run_id, judgment_id) -> digest / conflicted ids
        self._pair_items: Dict[Tuple[str, str, str], Optional[str]] = {}
        self._pair_conflicts: Dict[Tuple[str, str, str], List[str]] = {}
        # ONE pair's current item FLAGS by an indexed probe (both sides of a comparison that asks the cap evidence): (run_id, judgment_id) -> digest
        self._pair_flags: Dict[Tuple[str, str], Optional[str]] = {}

    # ---- loading ------------------------------------------------------------------------------------------------------------------

    @classmethod
    def load(cls, conn: sqlite3.Connection) -> "AnchorCurrency":
        """ONE read of the frame: the fold of ``v_run_current`` (every run), the unsteered ``run`` rows, their complete judgments and
        EVERY ``run_metric`` row of theirs (a few thousand rows live; a sidecar ships the same rows) minus the rows the paper frame does
        not count: where the connection carries an
        ``observation`` table, a row whose observation key is current under no revision is dropped by :func:`withdrawn_metric_cte`
        (``counts["withdrawn_metric_rows"]`` says how many); a row no revision observed is kept. A row of a judgment a re-judge displaced
        or a claim released is dropped first, by :data:`CURRENT_JUDGMENT_SQL` (``counts["non_current_judgment_rows"]``; a row failing
        both rules is counted there). A sidecar's ``observation`` table holds the adapters' CURRENT rows only and its ``judgment`` table
        the complete judgments only, so both predicates are vacuous there by construction -- ``reader_state.export`` applies the frame's
        over the live database before the rows ship. Raises :class:`AnchorCurrencyUnavailable` when ``v_run_current`` is not defined
        (a pre-0021 database, a sidecar older than ``reader_state/2``)."""
        objects = {str(r[0]): str(r[1]) for r in conn.execute("SELECT name, type FROM sqlite_master WHERE type NOT IN ('index', 'trigger')")}
        if "v_run_current" not in objects:
            raise AnchorCurrencyUnavailable(f"v_run_current is not defined in this database (schema {_schema_version(conn, objects)}): "
                                            "the anchor currency cannot be loaded")
        for t in ("run", "judgment", "run_metric"):
            if t not in objects:
                raise AnchorCurrencyUnavailable(f"this database has no {t} table (schema {_schema_version(conn, objects)}): the anchor currency cannot be loaded")
        currency: Dict[str, _Currency] = {}
        for r in conn.execute(f"SELECT {', '.join(CURRENCY_COLUMNS)} FROM v_run_current"):
            currency[str(r[0])] = _Currency(int(r[1] or 0), (str(r[2]) if r[2] is not None else None), int(r[3] or 0), int(r[4] or 0),
                                            (str(r[5]) if r[5] is not None else None), int((r[6] or "[]") != "[]"))
        runs: Dict[str, Dict[str, Any]] = {}
        cols = list(RUN_COLUMNS) + ["complete_utc"]
        for r in conn.execute(f"SELECT {', '.join(RUN_COLUMNS)}, COALESCE(generated_utc, queued_utc) FROM run WHERE object_id = 'none'"):
            row = dict(zip(cols, tuple(r)))
            runs[str(row["run_id"])] = row
        judgments: Dict[str, List[_Judgment]] = {}
        for r in conn.execute("SELECT judgment_id, run_id, judge_id, status, superseded_by_judgment_id FROM judgment "
                              "WHERE status = 'complete' AND run_id IN (SELECT run_id FROM run WHERE object_id = 'none')"):
            j = _Judgment(str(r[0]), str(r[1]), (str(r[2]) if r[2] is not None else None), str(r[3]), (str(r[4]) if r[4] is not None else None))
            judgments.setdefault(j.run_id, []).append(j)
        for js in judgments.values():
            js.sort(key=lambda j: j.judgment_id)
        judge_of = {j.judgment_id: j for js in judgments.values() for j in js}
        # every reading of every unsteered run: a metric is a candidate anchor metric the moment a steered row intends a contrast on it, so
        # the set is not read off `run_contrast` (which would leave the first contrast at a new metric -- and every contrast of a fresh
        # registry -- without a reading to anchor on); the unsteered readings are a few thousand rows live, the item rows stay lazy per metric
        metric_filter = ""
        # the readings the paper frame counts: the same joins and predicates as v_paper_long's metric_role (`wm.run_id IS NULL` keeps a
        # row; CURRENT_JUDGMENT_SQL keeps a row of no judgment or of a judgment not displaced / released), selected as columns so one
        # statement both filters and counts what it dropped
        observed = "observation" in objects
        sql = ((f"WITH {withdrawn_metric_cte()} " if observed else "")
               + "SELECT rm.run_id, rm.judgment_id, rm.metric_id, rm.threshold, rm.computation_id, rm.k, rm.n, rm.value, rm.ci_lo, rm.ci_hi, "
               + "rm.computed_by, rm.source_uri, " + ("wm.run_id IS NULL" if observed else "1") + " AS reading_counted, "
               + CURRENT_JUDGMENT_SQL + " AS judgment_current FROM run_metric rm "
               + (WITHDRAWN_METRIC_JOIN + " " if observed else "") + current_judgment_join() + " "
               + "WHERE rm.run_id IN (SELECT run_id FROM run WHERE object_id = 'none')" + metric_filter)
        metrics: Dict[Tuple[str, str, str, float], List[Measurement]] = {}
        n_metric = 0
        n_withdrawn = 0
        n_non_current_judgment = 0
        for r in conn.execute(sql):
            if not r[13]:
                n_non_current_judgment += 1
                continue
            if not r[12]:
                n_withdrawn += 1
                continue
            jid = str(r[1]) if r[1] is not None else NO_JUDGMENT
            j = judge_of.get(jid)
            m = Measurement(run_id=str(r[0]), judgment_id=jid, computation_id=(str(r[4]) if r[4] is not None else ""), metric_id=str(r[2]),
                            threshold=_thr(r[3]), k=(float(r[5]) if r[5] is not None else None), n=_opt_int(r[6]),
                            value=(float(r[7]) if r[7] is not None else None), ci_lo=(float(r[8]) if r[8] is not None else None),
                            ci_hi=(float(r[9]) if r[9] is not None else None), judge_id=(j.judge_id if j is not None else None),
                            judgment_status=(j.status if j is not None else ("none" if jid == NO_JUDGMENT else "unknown")),
                            computed_by=str(r[10]), source_uri=(str(r[11]) if r[11] is not None else None))
            metrics.setdefault((m.run_id, m.judgment_id, m.metric_id, m.threshold), []).append(m)
            n_metric += 1
        for ms in metrics.values():
            ms.sort(key=lambda m: m.computation_id)
        items_known, items_unknown_why = _items_known(objects)
        flags_known, flags_unknown_why = _flags_known(objects)
        counts = {"runs": len(currency), "unsteered": len(runs), "judgments": sum(len(v) for v in judgments.values()), "metric_rows": n_metric,
                  "withdrawn_metric_rows": n_withdrawn, "non_current_judgment_rows": n_non_current_judgment, "items_known": int(items_known),
                  "flags_known": int(flags_known)}
        return cls(currency, runs, judgments, metrics, conn, items_known, counts, items_unknown_why, flags_known, flags_unknown_why)

    # ---- the currency of one run --------------------------------------------------------------------------------------------------

    def is_current(self, run_id: str) -> Optional[bool]:
        """``True`` / ``False`` from the fold; ``None`` for a run the registry never folded."""
        c = self._currency.get(run_id)
        return None if c is None else bool(c.is_current)

    def why(self, run_id: str) -> Optional[str]:
        """``v_run_current.why`` of a retired run (``None`` on a current run and on an unknown id)."""
        c = self._currency.get(run_id)
        return None if c is None else c.why

    def currency(self, run_id: str) -> Optional[Dict[str, Any]]:
        """The folded currency row of a run as a dict (``None`` for an unknown id)."""
        c = self._currency.get(run_id)
        return None if c is None else {"is_current": c.is_current, "why": c.why, "held_out": c.held_out, "in_current_paper": c.in_current_paper,
                                       "selection_state": c.selection_state, "roster_matched": c.roster_matched}

    def unsteered_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """The frame's unsteered run row (``None`` when the id is not an unsteered run of the frame)."""
        row = self._runs.get(run_id)
        return None if row is None else dict(row)

    # ---- the choice -----------------------------------------------------------------------------------------------------------------

    def choose(self, key: AnchorKey, metric_id: str, threshold: Any, judge_id: Optional[str], *, computation_id: str = "",
               needs_items: bool = False, steered_run_id: Optional[str] = None, steered_judgment_id: Optional[str] = None,
               items_metric_id: Optional[str] = None, needs_flags: bool = False) -> AnchorChoice:
        """The anchor of one intended contrast at ``(metric_id, threshold)`` under the steered row's ``judge_id`` (``None`` for a
        rule-scored row: the readings filed without a judgment). ``computation_id`` is the steered row's, the e3 tie-break among
        several readings; ``needs_items`` asks e4 (a family paired over registry items), and ``items_metric_id`` names the metric the
        paired statistic's ITEM ROWS are filed at when it is not the contrast's own (the store readers' rates and means over per-item
        labels, filed at the control's label metric): e4's presence check, conflict check and both digests read that metric, on both
        sides. ``needs_flags`` (with ``needs_items``) asks the per-item CAP evidence too -- a statistic that scores an item by its label
        and by its ``item_flags.at_token_cap`` (the store readers' graded mean): the decision then carries the digest of each side's
        current flags rows under the pair's judgment, and a frame that knows no flags refuses the candidate by name. Deterministic over
        the loaded frame.
        ``steered_run_id`` / ``steered_judgment_id`` identify the comparison in the group's decision set: when a
        :attr:`decision_log` is bound, the comparison (:func:`comparison_record`) and its decision (:func:`decision_record`) are
        appended to it, parked outcomes included -- what the runner records on the marker and fingerprints into the group's identity.
        An item-paired comparison (``needs_items``) MUST name them: the steered side's current item rows and conflicts at the metric are
        the other half of the pair and are read from the frame by them (``ValueError`` otherwise, rather than a decision blind to one side)."""
        choice = self._choose(key, metric_id, threshold, judge_id, computation_id=computation_id, needs_items=needs_items,
                              steered_run_id=steered_run_id, steered_judgment_id=steered_judgment_id, items_metric_id=items_metric_id,
                              needs_flags=needs_flags)
        if self.decision_log is not None:
            comparison = comparison_record(key, metric_id, threshold, judge_id, computation_id=computation_id, needs_items=needs_items,
                                           steered_run_id=steered_run_id, steered_judgment_id=steered_judgment_id, items_metric_id=items_metric_id,
                                           needs_flags=needs_flags)
            self.decision_log.append(decision_entry(comparison, decision_record(choice)))
        return choice

    def replay(self, comparison: Mapping[str, Any]) -> Dict[str, Any]:
        """The DECISION this frame makes for a recorded comparison (:func:`comparison_record`, or a currency-only one,
        :func:`currency_comparison_record`), as :func:`decision_record` / :func:`currency_decision_record` spell it -- without logging:
        the pre-read evaluation of a head's recorded set."""
        if is_currency_comparison(comparison):
            return currency_decision_record(self._check_currency(str(comparison["anchor_run_id"]), str(comparison["anchor_kind"])))
        choice = self._choose(key_of_comparison(comparison), str(comparison["metric_id"]), comparison.get("threshold"), comparison.get("judge_id"),
                              computation_id=str(comparison.get("computation_id") or ""), needs_items=bool(comparison.get("needs_items")),
                              steered_run_id=comparison.get("run_id"), steered_judgment_id=comparison.get("judgment_id"),
                              items_metric_id=comparison.get("items_metric_id"), needs_flags=bool(comparison.get("needs_flags")))
        return decision_record(choice)

    def decisions_of(self, comparisons: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        """The decision set this frame makes over recorded comparisons, in canonical order (:func:`sorted_entries`): fingerprint it with
        :func:`decision_fingerprint` and compare with the head's recorded one."""
        return sorted_entries([decision_entry(c, self.replay(c)) for c in comparisons])

    # ---- the currency of a STEERED anchor (r2_export's `full_dose`: never re-picked) --------------------------------------------------

    def check_currency(self, anchor_run_id: str, anchor_kind: str, metric_id: str, threshold: Any, *, computation_id: str = "",
                       steered_run_id: Optional[str] = None, steered_judgment_id: Optional[str] = None) -> CurrencyCheck:
        """Whether ``anchor_run_id`` -- an anchor that is NOT an unsteered run of the exact key and is never re-picked (the export's 1x
        reading of the same object, ``anchor_kind full_dose``) -- is a current run of the fold: ``own_current`` files the adapter's record,
        ``no_current_anchor`` parks it (retired by a ruling or a later run, or not folded yet: ``not_in_frame``, the first pass that emits both
        runs). Logged on the bound :attr:`decision_log` under :data:`CURRENCY_KIND` (a parked check re-reads the group when the
        anchor's currency changes, exactly as an unsteered comparison does); :meth:`replay` decides a recorded one again."""
        check = self._check_currency(str(anchor_run_id), str(anchor_kind))
        if self.decision_log is not None:
            comparison = currency_comparison_record(anchor_run_id, anchor_kind, metric_id, threshold, computation_id=computation_id,
                                                    steered_run_id=steered_run_id, steered_judgment_id=steered_judgment_id)
            self.decision_log.append(decision_entry(comparison, currency_decision_record(check)))
        return check

    def _check_currency(self, anchor_run_id: str, anchor_kind: str) -> CurrencyCheck:
        cur = self._currency.get(anchor_run_id)
        if cur is None:
            return CurrencyCheck(anchor_run_id, anchor_kind, None, "no_current_anchor", "not_in_frame", f"{anchor_run_id} is not a run the registry has folded")
        if cur.is_current:
            return CurrencyCheck(anchor_run_id, anchor_kind, True, "own_current", "current", "")
        return CurrencyCheck(anchor_run_id, anchor_kind, False, "no_current_anchor", _classify_why(cur.why), cur.why or "not current")

    # ---- the item VALUES a statistic paired over registry items reads (the grading host only) -------------------------------------------------

    def item_values(self, metric_id: str, run_id: str, judgment_id: str) -> Optional[Dict[str, Optional[float]]]:
        """``item_id -> value`` of the CURRENT item rows of ``(run_id, judgment_id)`` at ``metric_id`` (``v_item_result_current``, the rows the
        digests hash), for ANY run -- both sides of a statistic paired over registry items (:func:`declared_common.paired_items_contrast`).
        ``None`` where the frame knows no items (a shard; the caller refuses by name). An item id filed more than once under the pair (two
        trials, samples or turns of one item) raises ``ValueError``: the paired helpers key items by id and never average rows into one."""
        if not self.items_known or self._conn is None:
            return None
        out: Dict[str, Optional[float]] = {}
        sql = ("SELECT item_id, value FROM v_item_result_current WHERE metric_id = ? AND run_id = ? AND judgment_id = ? "
               "ORDER BY item_id, trial_index, sample_index, turn_index")
        for item_id, value in self._conn.execute(sql, (metric_id, run_id, judgment_id)):
            iid = str(item_id)
            if iid in out:
                raise ValueError(f"item {iid} of run {run_id} at {metric_id} under judgment {judgment_id or 'none'} is filed more than once "
                                 "(several trials, samples or turns): a statistic paired by item id cannot read it")
            out[iid] = float(value) if value is not None else None
        return out

    def _choose(self, key: AnchorKey, metric_id: str, threshold: Any, judge_id: Optional[str], *, computation_id: str = "",
                needs_items: bool = False, steered_run_id: Optional[str] = None, steered_judgment_id: Optional[str] = None,
                items_metric_id: Optional[str] = None, needs_flags: bool = False) -> AnchorChoice:
        """:meth:`choose` without the log: pure over the loaded frame."""
        thr = _thr(threshold)
        if needs_flags and not needs_items:
            raise ValueError("the per-item cap evidence (needs_flags) is asked of an item-paired comparison only (needs_items): the flags are read "
                             "beside the item rows, under the pair's judgment, and mean nothing without them")
        # the metric the item rows of an e4 comparison are filed at: the contrast's own unless the comparison names another
        im = str(items_metric_id) if (needs_items and items_metric_id) else metric_id
        im_text = "" if im == metric_id else f"; item rows at {im}"
        # the steered side of an item-paired comparison (e4): its current item rows' digest and its conflicted keys, by one indexed probe per
        # pair -- the other half of the paired statistic, and no file input of the group, so both are part of the decision; a comparison that
        # asks the cap evidence carries the digest of the steered side's current flags rows too (None until they fold)
        steered: Dict[str, Any] = {"steered_items_digest": None, "steered_item_conflicts": None, "steered_flags_digest": None}
        steered_conflict = ""
        if needs_items:
            if steered_run_id is None or steered_judgment_id is None:
                raise ValueError("an item-paired comparison (needs_items) names its steered run and judgment (steered_run_id / steered_judgment_id): "
                                 "the paired statistic reads the steered side's item rows too, and the decision carries their state")
            if self.items_known:
                s_run, s_jid = str(steered_run_id), str(steered_judgment_id)
                s_conflicts = self.pair_items_conflicts(im, s_run, s_jid)
                steered = {"steered_items_digest": self.pair_items_digest(im, s_run, s_jid), "steered_item_conflicts": len(s_conflicts),
                           "steered_flags_digest": (self.pair_flags_digest(s_run, s_jid) if needs_flags else None)}
                if s_conflicts:
                    steered_conflict = _conflict_why(s_conflicts, f"({metric_id}, {thr:g}{im_text}) under judgment {s_jid or 'none'}", subject=f"the steered run {s_run}")
        own_id = key.own_anchor_run_id
        own_reason, own_why = self._own_reason(own_id)
        own_candidate: Optional[Tuple[Candidate, Measurement, str]] = None
        if own_reason == "current":
            row = self._runs.get(str(own_id))
            if row is None:
                own_reason, own_why = "not_unsteered", f"{own_id} is current but is not an unsteered run of the frame"
            else:
                ok, code, why_, m, jid = self._usable(row, metric_id, thr, judge_id, computation_id, needs_items,
                                                      judgment_id=(key.own_anchor_judgment_id or None), items_metric_id=im, needs_flags=needs_flags)
                if ok:
                    own_candidate = (self._candidate(row, key), m, jid)      # type: ignore[arg-type]
                else:
                    own_reason, own_why = code, why_
        # the exact-key pool (every tier reads it; recorded on an own-current pick too, so a same-key current none beside a cross-cap own
        # pairing is visible in the selection block)
        eligible: List[Tuple[Candidate, Measurement, str]] = []
        rejected: List[Tuple[str, str]] = []
        for rid in self._by_key.get(key.exact, []):
            row = self._runs[rid]
            if not _decode_matches(key.decode_policy_id, row.get("decode_policy_id")):
                continue
            if rid == own_id and own_candidate is not None:
                eligible.append(own_candidate)
                continue
            cur = self._currency.get(rid)
            if cur is None:
                rejected.append((rid, "not folded in v_run_current"))
                continue
            if not cur.is_current:
                rejected.append((rid, cur.why or "not current"))
                continue
            ok, _code, why_, m, jid = self._usable(row, metric_id, thr, judge_id, computation_id, needs_items, items_metric_id=im, needs_flags=needs_flags)
            if not ok:
                rejected.append((rid, why_))
                continue
            eligible.append((self._candidate(row, key), m, jid))      # type: ignore[arg-type]
        eligible.sort(key=lambda e: _order(e[0]))
        rejected.sort()
        candidates = tuple(e[0] for e in eligible)
        common = dict(own_reason=own_reason, own_why=own_why, n_candidates=len(candidates), candidates=candidates, rejected=tuple(rejected), key=key, **steered)
        if steered_conflict:
            # the steered side is conflicted: no anchor can pair with it, whatever the pool holds (the pool stays on the record as the audit trail);
            # `own_reason` says `item_conflict` for the own anchor too -- the pair could not be formed, whichever id the adapter brought
            return AnchorChoice(state="no_current_anchor", run_id=None, anchor_kind=None, cross_source=False, why=steered_conflict, anchor_judgment_id="",
                                measurement=None, **dict(common, own_reason="item_conflict", own_why=steered_conflict))

        def evidence(c: Candidate, jid: str) -> Dict[str, Optional[str]]:
            # e4 asked: the chosen anchor's item content is part of the decision (a paired statistic reads the rows, not the aggregate); the cap
            # evidence asked: so is the digest of its current flags rows under the reading's judgment (the per-item cap reading a graded score uses)
            return {"items_digest": (self.items_digest(im, c.run_id, jid) if needs_items else None),
                    "flags_digest": (self.pair_flags_digest(c.run_id, jid) if needs_flags else None)}

        if own_candidate is not None:
            c, m, jid = own_candidate
            return AnchorChoice(state="own_current", run_id=c.run_id, anchor_kind=(key.own_anchor_kind or KIND_SAME_STANDUP), cross_source=False,
                                why="", anchor_judgment_id=jid, measurement=m, **evidence(c, jid), **common)
        # tier (a): the one same-standup run; two or more refuse by name (strength_mtdiff's rule generalized)
        same_standup = [e for e in eligible if e[0].same_standup]
        if len(same_standup) > 1:
            ids = ", ".join(e[0].run_id for e in same_standup)
            why_ = (f"{len(same_standup)} current unsteered runs of this standup at one key ({ids}); a paired contrast needs exactly one"
                    + (f" [own anchor {own_id}: {own_why}]" if own_id is not None else ""))
            return AnchorChoice(state="no_current_anchor", run_id=None, anchor_kind=None, cross_source=False, why=why_, anchor_judgment_id="",
                                measurement=None, **common)
        if len(same_standup) == 1:
            c, m, jid = same_standup[0]
            return AnchorChoice(state="sibling_same_source", run_id=c.run_id, anchor_kind=KIND_SAME_STANDUP, cross_source=False, why=own_why,
                                anchor_judgment_id=jid, measurement=m, **evidence(c, jid), **common)
        # tier (b): the same source, another standup
        same_source = [e for e in eligible if e[0].same_source]
        if same_source:
            c, m, jid = same_source[0]
            return AnchorChoice(state="sibling_same_source", run_id=c.run_id, anchor_kind=KIND_SIBLING, cross_source=False, why=own_why,
                                anchor_judgment_id=jid, measurement=m, **evidence(c, jid), **common)
        # tier (c): the paper's own baseline from another source -- in the current paper AND roster-matched
        reference = [e for e in eligible if e[0].in_current_paper == 1 and e[0].roster_matched]
        if reference:
            c, m, jid = reference[0]
            return AnchorChoice(state="borrowed_reference", run_id=c.run_id, anchor_kind=KIND_BORROWED, cross_source=True, why=own_why,
                                anchor_judgment_id=jid, measurement=m, **evidence(c, jid), **common)
        why_ = self._nearest_miss(key, eligible, rejected)
        if own_id is not None:           # an adapter that declared no anchor has nothing to explain about one
            why_ = f"own anchor {own_id}: {own_why}; {why_}"
        return AnchorChoice(state="no_current_anchor", run_id=None, anchor_kind=None, cross_source=False, why=why_, anchor_judgment_id="",
                            measurement=None, **common)

    # ---- helpers --------------------------------------------------------------------------------------------------------------------

    def _own_reason(self, own_id: Optional[str]) -> Tuple[str, str]:
        if own_id is None:
            return "none_declared", "the adapter declares no anchor of its own"
        cur = self._currency.get(str(own_id))
        if cur is None:
            return "not_in_frame", f"{own_id} is not a run the registry has folded"
        if cur.is_current:
            return "current", ""
        return _classify_why(cur.why), (cur.why or "not current")

    def _candidate(self, row: Mapping[str, Any], key: AnchorKey) -> Candidate:
        cur = self._currency.get(str(row["run_id"]))
        same_standup = bool(key.standup) and all(row.get(col) is not None and val is not None and str(row.get(col)) == str(val)
                                                 for col, val in key.standup.items())
        same_source = key.lane_source_legacy is not None and row.get("lane_source_legacy") == key.lane_source_legacy
        return Candidate(run_id=str(row["run_id"]), lane_source_legacy=row.get("lane_source_legacy"), view_legacy=row.get("view_legacy"),
                         cluster_job_id=(str(row["cluster_job_id"]) if row.get("cluster_job_id") is not None else None),
                         producer_run_key_legacy=str(row.get("producer_run_key_legacy") or ""), complete_utc=row.get("complete_utc"),
                         in_current_paper=(cur.in_current_paper if cur else 0), roster_matched=(cur.roster_matched if cur else 0),
                         same_standup=same_standup, same_source=same_source)

    def _usable(self, row: Mapping[str, Any], metric_id: str, thr: float, judge_id: Optional[str], computation_id: str,
                needs_items: bool, judgment_id: Optional[str] = None, items_metric_id: Optional[str] = None,
                needs_flags: bool = False) -> Tuple[bool, str, str, Optional[Measurement], str]:
        """e2-e4 for one unsteered run: ``(ok, reason code, why, measurement, judgment_id)``. ``judgment_id`` is the own anchor's DECLARED
        judgment (:attr:`AnchorKey.own_anchor_judgment_id`): when given, the reading is read under it -- it must be one of the run's complete,
        non-superseded judgments under ``judge_id`` -- instead of resolving the run's one judgment by judge (step 0 only). ``items_metric_id``
        is the metric e4 reads the item rows at (the contrast's own when None). ``needs_flags`` asks the per-item cap evidence beside the item
        rows: a frame that knows no flags (a shard, a database without ``v_item_flags_current``) refuses the candidate by name rather than
        deciding blind to the evidence the statistic scores by."""
        rid = str(row["run_id"])
        im = str(items_metric_id) if items_metric_id else metric_id
        state = str(row.get("lifecycle_state") or "")
        if state not in S.COMPLETE_LIFECYCLE_STATES:
            return False, "lifecycle", f"lifecycle_state {state or 'null'} (not complete)", None, ""
        if judgment_id:
            js = [j for j in self._judgments.get(rid, []) if j.judgment_id == judgment_id]
            if not js:
                return False, "no_reading", f"the declared anchor judgment {judgment_id} is not a complete judgment of {rid}", None, ""
            j = js[0]
            if j.superseded_by is not None:
                return False, "no_reading", f"the declared anchor judgment {judgment_id} is superseded by {j.superseded_by}", None, ""
            if judge_id is not None and j.judge_id != judge_id:
                return False, "no_reading", f"the declared anchor judgment {judgment_id} is under judge {j.judge_id}, not the row's judge {judge_id}", None, ""
            if judge_id is None and j.judge_id is not None:
                return False, "no_reading", f"the declared anchor judgment {judgment_id} is under judge {j.judge_id}; the row is rule-scored (no judge)", None, ""
            jid = j.judgment_id
        elif judge_id is None:
            jid = NO_JUDGMENT
        else:
            js = [j for j in self._judgments.get(rid, []) if j.judge_id == judge_id and j.superseded_by is None]
            if not js:
                return False, "no_reading", f"no complete judgment under judge {judge_id}", None, ""
            if len(js) > 1:
                return False, "ambiguous", f"{len(js)} complete judgments under judge {judge_id} ({', '.join(j.judgment_id for j in js)}): ambiguous", None, ""
            jid = js[0].judgment_id
        rows = list(self._metrics.get((rid, jid, metric_id, thr), []))
        label = f"({metric_id}, {thr:g}) under judgment {jid or 'none'}"
        if not rows:
            return False, "no_reading", f"no reading at {label}", None, jid
        if computation_id:
            exact = [m for m in rows if m.computation_id == computation_id]
            if exact:
                rows = exact
        if len(rows) > 1:
            if not _same_reading(rows):
                detail = "; ".join(f"computation {m.computation_id or 'none'}: value {m.value} k {m.k} n {m.n}" for m in rows)
                return False, "ambiguous", f"{len(rows)} readings at {label} with different values ({detail}): ambiguous", None, jid
            rows = rows[:1]                 # equal readings twice: the first by computation id
        m = rows[0]
        if needs_items:
            if not self.items_known:
                return False, "no_reading", f"item presence is not known to this frame ({self.items_unknown_why}): a statistic paired over items is not available here", None, jid
            if (rid, jid) not in self.items_at(im):
                return False, "no_reading", f"no current item rows at {im} under judgment {jid or 'none'}", None, jid
            conflicted = self.items_conflicts(im, rid, jid)
            if conflicted:
                return False, "item_conflict", _conflict_why(conflicted, label if im == metric_id else f"({metric_id}, {thr:g}; item rows at {im}) under judgment {jid or 'none'}"), None, jid
            if needs_flags and not self.flags_known:
                return False, "no_reading", (f"item cap flags are not known to this frame ({self.flags_unknown_why}): a statistic that scores an item "
                                             f"by its token-cap flag is not available here"), None, jid
        return True, "current", "", m, jid

    def items_at(self, metric_id: str) -> Set[Tuple[str, str]]:
        """The ``(run_id, judgment_id)`` pairs of unsteered runs with CURRENT item rows at ``metric_id`` (``v_item_result_current``) --
        read lazily, once per metric: the item read is ~100 s live over every contrast metric, so it runs only for a family that asks
        (e4). Empty, and :attr:`items_known` False, where the connection ships no item rows (a shard)."""
        return set(self._item_digests(metric_id))

    def items_digest(self, metric_id: str, run_id: str, judgment_id: str) -> Optional[str]:
        """The sha256 over the CURRENT item rows of ``(run_id, judgment_id)`` at ``metric_id`` -- each row's :data:`ITEM_COLUMNS` as canonical
        JSON, in the rows' key order -- the content a statistic paired over items reads, so a corrected or re-filed item row under the same
        judgment is a changed decision (:func:`decision_record`). ``None`` when the pair has no current item rows or the frame knows none.
        The rows are the ones the current-row rank selects (``v_item_result_current``): a second accepted provenance of a key that disagrees
        with the selected row does not move this digest, and is not meant to -- it makes the pair conflicted (:meth:`items_conflicts`), and
        e4 refuses a conflicted pair by name, so no decision ever carries this digest for one."""
        return self._item_digests(metric_id).get((run_id, judgment_id))

    def items_conflicts(self, metric_id: str, run_id: str, judgment_id: str) -> List[str]:
        """The item ids of ``(run_id, judgment_id)`` at ``metric_id`` whose measurement key carries CONFLICTING visible provenances
        (``v_item_conflict``: two or more accepted provenances of one key with different values or labels, whichever the current-row
        rank selects), in key order, one entry per conflicted key. Empty when none, and where the frame knows no items. Non-empty refuses
        the pair in e4 (``item_conflict``), as the registry's paired-reading API refuses to pair it -- read lazily, once per metric, like the
        item digests."""
        return list(self._item_conflicts(metric_id).get((run_id, judgment_id), ()))

    def pair_items_digest(self, metric_id: str, run_id: str, judgment_id: str) -> Optional[str]:
        """The sha256 over ONE pair's current item rows at ``metric_id`` -- the same rows, columns, order and line format as
        :meth:`items_digest` (equal to it for an unsteered pair) -- by one indexed probe of ``v_item_result_current`` for ANY run: the
        steered side of an item-paired comparison, which the per-metric bulk read of the unsteered candidates never covers. ``None`` when
        the pair has no current item rows (the steered rows of a first read are not folded yet) or the frame knows no items. Cached per pair."""
        if not self.items_known or self._conn is None:
            return None
        k = (metric_id, run_id, judgment_id)
        if k not in self._pair_items:
            h = hashlib.sha256()
            n = 0
            sql = (f"SELECT {', '.join(ITEM_COLUMNS)} FROM v_item_result_current WHERE metric_id = ? AND run_id = ? AND judgment_id = ? "
                   "ORDER BY item_id, trial_index, sample_index, turn_index")
            for r in self._conn.execute(sql, k):
                h.update(_item_line(r))
                n += 1
            self._pair_items[k] = h.hexdigest() if n else None
        return self._pair_items[k]

    def pair_items_conflicts(self, metric_id: str, run_id: str, judgment_id: str) -> List[str]:
        """The conflicted item ids of ONE pair at ``metric_id`` (``v_item_conflict``, as :meth:`items_conflicts` reads them for the unsteered
        candidates), by one indexed probe for ANY run -- the steered side of an item-paired comparison. Empty when none or where the frame
        knows no items; non-empty parks the comparison. Cached per pair."""
        if not self.items_known or self._conn is None:
            return []
        k = (metric_id, run_id, judgment_id)
        if k not in self._pair_conflicts:
            sql = ("SELECT item_id FROM v_item_conflict WHERE metric_id = ? AND run_id = ? AND judgment_id = ? "
                   "ORDER BY item_id, trial_index, sample_index, turn_index")
            self._pair_conflicts[k] = [str(r[0]) for r in self._conn.execute(sql, k)]
        return list(self._pair_conflicts[k])

    def pair_flags_digest(self, run_id: str, judgment_id: str) -> Optional[str]:
        """The sha256 over ONE pair's current ``item_flags`` rows under ``judgment_id`` -- each row's :data:`FLAG_COLUMNS` as canonical JSON in
        the rows' key order (``item_id, trial_index, sample_index, turn_index``; one line format with the item digests) -- by one indexed probe
        of ``v_item_flags_current`` for ANY run: both sides of a comparison that asks the per-item cap evidence (``needs_flags``). ``item_flags``
        is keyed without a metric, so the digest is per ``(run, judgment)``: the flags of the items a judgment judged. ``None`` when the pair has
        no current flags rows (not folded yet) or the frame knows no flags. Cached per pair. Equal to :func:`flag_rows_digest` over the
        same rows as an adapter's payloads spell them (:func:`flag_row_values`)."""
        if not self.flags_known or self._conn is None:
            return None
        k = (run_id, judgment_id)
        if k not in self._pair_flags:
            h = hashlib.sha256()
            n = 0
            sql = (f"SELECT {', '.join(FLAG_COLUMNS)} FROM v_item_flags_current WHERE run_id = ? AND judgment_id = ? "
                   "ORDER BY item_id, trial_index, sample_index, turn_index")
            for r in self._conn.execute(sql, k):
                h.update(_item_line(flag_row_values(dict(zip(FLAG_COLUMNS, r)))))
                n += 1
            self._pair_flags[k] = h.hexdigest() if n else None
        return self._pair_flags[k]

    def _item_conflicts(self, metric_id: str) -> Dict[Tuple[str, str], List[str]]:
        """One read per metric of ``v_item_conflict`` over every unsteered run, the conflicted item ids per ``(run_id, judgment_id)`` in key order."""
        if not self.items_known or self._conn is None:
            return {}
        if metric_id not in self._conflicts:
            got: Dict[Tuple[str, str], List[str]] = {}
            sql = ("SELECT run_id, judgment_id, item_id FROM v_item_conflict WHERE metric_id = ? "
                   "AND run_id IN (SELECT run_id FROM run WHERE object_id = 'none') "
                   "ORDER BY run_id, judgment_id, item_id, trial_index, sample_index, turn_index")
            for r in self._conn.execute(sql, (metric_id,)):
                got.setdefault((str(r[0]), (str(r[1]) if r[1] is not None else NO_JUDGMENT)), []).append(str(r[2]))
            self._conflicts[metric_id] = got
        return self._conflicts[metric_id]

    def _item_digests(self, metric_id: str) -> Dict[Tuple[str, str], str]:
        """One streamed read per metric of every unsteered run's current item rows, hashed per ``(run_id, judgment_id)`` in key order
        (one hash in flight at a time: the rows of one metric over every unsteered run are never held whole)."""
        if not self.items_known or self._conn is None:
            return {}
        if metric_id not in self._items:
            got: Dict[Tuple[str, str], str] = {}
            pair: Optional[Tuple[str, str]] = None
            h = hashlib.sha256()
            sql = (f"SELECT run_id, judgment_id, {', '.join(ITEM_COLUMNS)} FROM v_item_result_current WHERE metric_id = ? "
                   "AND run_id IN (SELECT run_id FROM run WHERE object_id = 'none') "
                   "ORDER BY run_id, judgment_id, item_id, trial_index, sample_index, turn_index")
            for r in self._conn.execute(sql, (metric_id,)):
                key = (str(r[0]), (str(r[1]) if r[1] is not None else NO_JUDGMENT))
                if key != pair:
                    if pair is not None:
                        got[pair] = h.hexdigest()
                    pair, h = key, hashlib.sha256()
                h.update(_item_line(r[2:]))
            if pair is not None:
                got[pair] = h.hexdigest()
            self._items[metric_id] = got
        return self._items[metric_id]

    def _nearest_miss(self, key: AnchorKey, eligible: Sequence[Tuple[Candidate, Measurement, str]], rejected: Sequence[Tuple[str, str]]) -> str:
        """Why nothing anchors, from the same frame: the exact-key runs that were not eligible or not admissible, else the nearest key."""
        if eligible:
            ids = ", ".join(e[0].run_id for e in eligible)
            return (f"the current unsteered run(s) at the key ({ids}) are neither of this source nor the paper's roster-matched baseline "
                    "(off-roster, another source): not an anchor")
        if rejected:
            return "the unsteered run(s) at the key are not eligible: " + "; ".join(f"{rid} ({why})" for rid, why in rejected)
        caps: Set[int] = set()
        insts: Set[str] = set()
        any_at_all = False
        for rid, row in self._runs.items():
            if row.get("model_id") != key.model_id or row.get("eval_id") != key.eval_id:
                continue
            any_at_all = True
            cur = self._currency.get(rid)
            if cur is None or not cur.is_current:
                continue
            if (row.get("instrument_id") == key.instrument_id and _opt_int(row.get("trial_index")) == _opt_int(key.trial_index)
                    and row.get("pass_label") == key.pass_label and _opt_int(row.get("token_cap_tokens")) != _opt_int(key.token_cap_tokens)
                    and row.get("token_cap_tokens") is not None):
                caps.add(int(row["token_cap_tokens"]))
            if (row.get("instrument_id") != key.instrument_id and _opt_int(row.get("token_cap_tokens")) == _opt_int(key.token_cap_tokens)
                    and _opt_int(row.get("trial_index")) == _opt_int(key.trial_index) and row.get("pass_label") == key.pass_label):
                insts.add(str(row["instrument_id"]))
        parts = []
        if caps:
            parts.append("a current unsteered run exists only at cap " + " / ".join(str(c) for c in sorted(caps)))
        if insts:
            parts.append("a current unsteered run at this cap exists only under instrument " + " / ".join(sorted(insts)))
        if parts:
            return "; ".join(parts)
        if any_at_all:
            return f"no current unsteered run of {key.model_id} / {key.eval_id} at the key (cap {key.token_cap_tokens}, trial {key.trial_index}, pass {key.pass_label}, instrument {key.instrument_id})"
        return f"no unsteered run of {key.model_id} / {key.eval_id} in the frame at any cap"


def _item_line(values: Sequence[Any]) -> bytes:
    """One item row's line of the digest: its :data:`ITEM_COLUMNS` values as canonical JSON plus a newline -- ONE format for every read
    (:meth:`AnchorCurrency.items_digest`, :meth:`AnchorCurrency.pair_items_digest`, :func:`item_rows_digest`)."""
    return (I.canonical_json(list(values)) + "\n").encode("utf-8")


def item_row_values(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    """An ``item_result`` payload's :data:`ITEM_COLUMNS`, typed as the database hands them back: the item id text, the three indices as
    integers, ``value`` as a float (the column's REAL affinity; a producer's whole number reads back as its float) or None, ``label`` text
    or None. What an adapter feeds :func:`item_rows_digest` to check the rows it re-read from a store against the registry's digest."""
    value = row.get("value")
    if isinstance(value, bool):
        value = float(value)
    return (str(row["item_id"]), int(row.get("trial_index") or 0), int(row.get("sample_index") or 0),
            int(row["turn_index"]) if row.get("turn_index") is not None else -1,
            (None if value is None else float(value)), (None if row.get("label") is None else str(row["label"])))


def item_rows_digest(rows: Sequence[Sequence[Any]]) -> Optional[str]:
    """The sha256 the frame would hold for these item rows (:data:`ITEM_COLUMNS` tuples, :func:`item_row_values`): the same line format and
    the same key order (``item_id, trial_index, sample_index, turn_index``) as :meth:`AnchorCurrency.items_digest`, so an adapter that
    computes its paired statistic from a store's rows can check them against the digest the decision carries -- rows the registry has not
    folded yet, or has folded differently, are not the evidence the decision was made on. ``None`` for no rows (as the frame's)."""
    ordered = sorted((tuple(r) for r in rows), key=lambda r: (str(r[0]), int(r[1]), int(r[2]), int(r[3])))
    if not ordered:
        return None
    h = hashlib.sha256()
    for r in ordered:
        h.update(_item_line(r))
    return h.hexdigest()


def flag_row_values(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    """An ``item_flags`` payload's (or a ``v_item_flags_current`` row's) :data:`FLAG_COLUMNS`, typed one way for both: the item id text, the
    three indices as integers, ``at_token_cap`` as an integer 0 / 1 or None (the column's CHECK; a payload's bool reads as its integer).
    What an adapter feeds :func:`flag_rows_digest` to check the flags it re-read from a store against the registry's digest, and what the
    frame's own probe hashes (:meth:`AnchorCurrency.pair_flags_digest`)."""
    cap = row.get("at_token_cap")
    return (str(row["item_id"]), int(row.get("trial_index") or 0), int(row.get("sample_index") or 0),
            int(row["turn_index"]) if row.get("turn_index") is not None else -1, (None if cap is None else int(cap)))


def flag_rows_digest(rows: Sequence[Sequence[Any]]) -> Optional[str]:
    """The sha256 the frame would hold for these ``item_flags`` rows (:data:`FLAG_COLUMNS` tuples, :func:`flag_row_values`): the same line
    format and key order as :meth:`AnchorCurrency.pair_flags_digest`, so an adapter whose paired statistic scores items by their cap flag can
    check the flags it re-read from a store against the digest the decision carries. ``None`` for no rows (as the frame's). The line format is
    the item digests' (:func:`item_rows_digest`); the two never collide, a flags line carrying five values and an item line six."""
    return item_rows_digest(rows)


def _flags_known(objects: Mapping[str, str]) -> Tuple[bool, str]:
    """Whether the frame knows the per-item cap flags, and why not when it does not: ``v_item_flags_current`` (the grading host), never a sidecar
    (``reader_state_meta``: a shard ships no item flags). Independent of :func:`_items_known`: a comparison asks the flags only beside the item
    rows (``needs_flags`` needs ``needs_items``), and a database defining the item objects without the flags object refuses that ask by name."""
    if "reader_state_meta" in objects:
        return False, "a shard ships no item flags"
    if "v_item_flags_current" in objects:
        return True, ""
    return False, "this database does not define v_item_flags_current"


def _items_known(objects: Mapping[str, str]) -> Tuple[bool, str]:
    """Whether the frame knows item presence and conflicts, and why not when it does not: both ``v_item_result_current`` and
    ``v_item_conflict`` (the grading host), never a sidecar (``reader_state_meta``: a shard ships no item rows). A database defining one without
    the other is named as such -- e4 then refuses with that text rather than reading as a shard."""
    if "reader_state_meta" in objects:
        return False, "a shard ships no item rows"
    rows, conflicts = "v_item_result_current" in objects, "v_item_conflict" in objects
    if rows and conflicts:
        return True, ""
    if not rows and not conflicts:
        return False, "this database defines neither v_item_result_current nor v_item_conflict"
    present, absent = ("v_item_result_current", "v_item_conflict") if rows else ("v_item_conflict", "v_item_result_current")
    return False, f"this database defines {present} without {absent}; the item rule needs both"


def _conflict_why(conflicted: Sequence[str], label: str, subject: str = "") -> str:
    """The ``item_conflict`` refusal text: the conflicted item count (one per conflicted key) and the first :data:`CONFLICT_NAMED` distinct
    item ids in key order, the rest counted -- a bounded list, as the paired-reading API's refusal names one. ``subject`` names the side
    when it is not the candidate under test (``the steered run <id>``)."""
    ids = list(dict.fromkeys(conflicted))
    named = ", ".join(ids[:CONFLICT_NAMED]) + (f" (+{len(ids) - CONFLICT_NAMED} more)" if len(ids) > CONFLICT_NAMED else "")
    return ((f"{subject}: " if subject else "") + f"{len(conflicted)} item(s) at {label} carry conflicting visible provenances ({named}; v_item_conflict): "
            "a statistic paired over items refuses the pair while the sources disagree")


def _row_key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (row.get("model_id"), row.get("eval_id"), row.get("instrument_id"), _opt_int(row.get("token_cap_tokens")), _opt_int(row.get("trial_index")),
            row.get("pass_label"))


def _decode_matches(steered: Optional[str], candidate: Optional[str]) -> bool:
    return steered is None or candidate is None or str(steered) == str(candidate)


def _order(c: Candidate) -> Tuple[int, int, str, str]:
    return (0 if (c.roster_matched or c.in_current_paper) else 1, 1 if c.complete_utc is None else 0, c.complete_utc or "", c.run_id)


def _same_reading(rows: Sequence[Measurement]) -> bool:
    first = rows[0]
    for m in rows[1:]:
        if (m.value is None) != (first.value is None) or (m.value is not None and abs(m.value - first.value) > VALUE_TOL):  # type: ignore[operator]
            return False
        if m.k != first.k or m.n != first.n:
            return False
    return True


def _classify_why(why: Optional[str]) -> str:
    """``v_run_current.why`` -> the own_reason vocabulary (the first failing rule, in the fold's order)."""
    w = why or ""
    if w.startswith("held_out"):
        return "held_out"
    if w.startswith("selection_state excluded") or w.startswith("decisive ruling exclude"):
        return "excluded"
    if w.startswith("selection_state superseded") or w.startswith("superseded_by_run_id") or w.startswith("decisive ruling supersede"):
        return "superseded"
    if w.startswith("selection_state withdrawn"):
        return "withdrawn"
    if w.startswith("lifecycle_state"):
        return "lifecycle"
    # every retired run carries one of the texts above (the fold's CASE); an unrecognized text is kept verbatim in `own_why` and
    # read as the fold's first rule, so a new spelling never crashes a read
    return "superseded"
