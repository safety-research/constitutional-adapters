#!/usr/bin/env python3
"""Native ``run_contrast`` emission: one paired contrast per complete steered run against its complete ``none`` anchor.

The steered runs of the selection (``--model-id`` / ``--campaign`` / ``--eval-id``) that read ``is_complete = 1`` in ``v_run_wide``
are paired with the complete ``none`` run (``object_id = 'none'``) of the SAME campaign x eval x instrument x model: one registered by
the same job (equal ``cluster_job_id`` or ``producer_run_key_legacy``) is the ``same_standup_none`` anchor; failing that any complete
sibling is the ``sibling_none`` anchor (several: the one sharing the run's ``trial_index``, then the latest queued, then by id -- the
choice and the candidate count ride ``extra``). For the run's headline metric (``eval.headline_metric_id`` honouring
``extra_json.headline_metric_ids``, through ``paper.run_headline_metric_id`` -- the one rule) the item-aligned pairs come from
``sisteer.registry.paired`` (both sides' CURRENT judgment -- the complete one, named to ``paired`` explicitly: a re-judged side's
superseded judgment keeps its item rows in the current view, and two complete judgments by one judge are refused by name, never chosen
between); ``delta`` is the paired mean difference, ``delta_lo`` / ``delta_hi`` a percentile paired bootstrap (B = 2000, seeded from the
pair of run ids and the metric, so a re-run reproduces the interval), ``delta_method`` names it. ONE record per (run, anchor,
metric, threshold) through ``emit.RecordWriter`` into the spool; idempotent -- a contrast already folded into ``run_contrast`` for
these judgment ids, or already sitting in the spool, is skipped and said so. One line per contrast written or skipped; exit 0.

NATIVE-ONLY: a pair whose CURRENT items under the metric ride an adapter's source
revision on either side (``source_revision_id <> ''`` in ``v_item_result_current``) is that adapter's to contrast; it is reported
``adapter_backed`` and never written. ``item_result`` is keyed by ``source_revision_id`` and the current view switches between
accepted, unsuperseded revisions of one source, so a later accepted revision can change the current values under the SAME run and
judgment ids -- which the record key above (run, anchor, judgment ids, metric, threshold) cannot see: a contrast written here for
such a pair would read ``exists_registry`` forever while the items it summarised had moved.

``run`` (one pass, what the daemon calls each tick), ``contrasts_due`` and ``emit_contrast`` are importable for the grading daemon.
``sweep_noncurrent`` (``--sweep-noncurrent``) is the BOUNDED re-emission sweep: every native run of the registry -- every
model and view, not the daemon's one model -- whose filed contrast's anchor is no longer current under the paper's rules (RUN_CURRENCY_VIEW)
is re-emitted against a current anchor in one pass, counted per (model, view, eval category, anchor kind); ``--dry-run`` prints the same
counts over a read-only connection and writes nothing; ``--limit`` bounds the emissions and counts the remainder. The daemon runs it once
per advance of its code (``grade_daemon.contrasts_sweep_for``), never every tick.

A filed native contrast whose anchor this emitter deems INELIGIBLE while the frame's view still
reads it current -- a ``none`` that became a pooled reading's member (``member_pass`` / ``unpooled_member``: ``v_run_current.is_current``
stays 1 and ``v_paper_long``'s contrast join reads that bit alone) -- is RETIRED
EXPLICITLY when its replacement is filed. The frame honours exactly one statement about a ``run_contrast`` row besides its anchor's
currency: the observation withdrawal of migration 0017 (``observation`` rows for ``table_name = run_contrast``, ``current`` derived from the
source's head revision; ``schema_def.WITHDRAWN_CONTRAST_CTE`` / ``CURRENT_CONTRAST_CTE``). So the emission writes, BEFORE the new row and
through the same writer, a per-key source (:data:`CONTRAST_SOURCE_PREFIX` + sha16 of the key) with an observing revision, the observation
of the old key, a withdrawing head revision and a ``withdrawal`` record whose ``extra.why`` names the anchor's lost membership
(:func:`withdrawal_records`); on the fold the old row leaves the frame, the new one joins alone and ``n_current_contrasts`` is unchanged.
A withdrawn key whose anchor is a paper baseline row again and is picked again is RE-ADMITTED -- a new head revision of the key's source
observing it (:func:`readmit_contrast`, status ``readmitted``), never a fresh key beside it. Nothing is withdrawn without a replacement:
a frame-visible row on an ineligible anchor with no current none is named (``frame_stale`` on the row and the sweep's table) and left
to the frame's own rule. Withdrawal and re-admission are new append-only statements; no row is edited.

Paper membership is a FRAME reading, not an anchor-validity condition -- this reverts
the in_current_paper half of the earlier :func:`anchor_eligible`. An anchor is CURRENT by the view AND not a member run; an own
same-standup none that is current is kept as filed whatever its paper membership, a current un-rostered u0 none is a borrowed anchor
(a companion-only model: one no roster entry names), and ``in_current_paper`` is RECORDED (``why``, extra.anchor_currency, the line) but
never required; ``frame_stale`` / the withdrawal apply only to an anchor this emitter retires while the frame still joins it.
So the two writers never pick different baselines for one key, :func:`pick_borrowed_anchor_tiered` mirrors
the preference order EXACTLY -- (1) the steered run's own current same-standup none, (2) a current none of the same source
(``run.lane_source_legacy``), (3) the paper's roster-matched in-paper none of another source (selected / in_current_paper 1),
(4) ONLY for a model no roster entry can match: any current none -- within every tier is_current 1 and no member label, the cap a
filter before the tiers, the existing tie-break within a tier (the run's trial, the earliest queued, the run id); the tier that fired
rides the due row (``borrowed_tier``), the line (``tier=``) and the record (extra.anchor_tier, anchor_selection.tier).

(1) The pending-transition guards are keyed on the FOLD -- a
``source_revision`` record of the key's per-key source sitting on the spool whose revision_id the registry does not carry yet
(:func:`pending_transition`; what ``observation.current`` derives from), never on the lifetime presence of a withdrawal or re-admission
file (record files are immutable and survive the fold, so the former ``withdrawal_in_spool`` / ``readmission_in_spool`` read a second
membership correction as already pending and a later re-admission as pending for ever); (2) a frame-visible key this emitter retires is
withdrawn even when another anchor's row is KEPT (``withdraw_stale`` on a `same_view_current` / `borrowed_current` row;
:func:`emit_withdrawal`, status ``withdrawn``); (3) a transition is ONE record file -- the withdrawal records, the re-admission records
and the replacement ``run_contrast`` in dependency order with strictly increasing stamps, written atomically by
:class:`ContrastTransitionWriter` (the ``AtomicBatchWriter`` mechanics: temp file, fsync, link under the next sequence number) and
FOLDED AS A UNIT by the harvester: a plain record file of at most ITEM_ROWS_PER_TRANSACTION records is never split across transactions
(harvest.ParsedFile.parts) and a chunk is one BEGIN IMMEDIATE that rolls back whole on a failure (Ingestor.ingest_chunk /
db.transaction), run-scale records folded source_revision-first then by stamp -- so the frame sees the old delta or the new one, never
none and never both.

Transition safety: (1) every record of a transition carries its OWN strictly increasing stamp
(:func:`withdrawal_records` stamps per record, each source revision's observed_utc its own record's stamp), so two or more frame-visible
retired keys on one row no longer share a group stamp that :meth:`ContrastTransitionWriter.write_transition` refuses (which aborted the
whole pass and left both obsolete contrasts joined); (2) the fold is a unit only for THROWN exceptions -- the harvester commits a file
whose records were `blocked` (a missing parent) or `refused` beside the ones that folded (Ingestor._ingest_chunk delivers `partial`) --
so a replacement whose anchor judgment the DESTINATION registry has not folded would leave the withdrawal folded and the replacement
blocked: zero contrasts. The emitter therefore (a) resolves the spool's destination registry (:func:`spool_destination_root`: the
spool is ``<root>/records`` of a registry with a database) and REFUSES the pass by name (:class:`DestinationRefused`) when it is not the
queried ``--registry-root`` unless ``--destination-registry-root`` names it explicitly (never inferred), and (b) before ANY transition
write opens the destination read-only and verifies every parent the records need is folded THERE (:func:`destination_missing`: the
steered run and its judgment, the anchor run and the anchor judgment, the run_contrast rows a withdrawal or re-admission names), refusing
the whole emission with ``destination_missing_parents`` -- nothing written, the old delta stays joined -- when any is missing; and
(c) validates EVERY record of a transition file with ``validate.validate_record`` before writing (:meth:`ContrastTransitionWriter.write_transition`),
so no record can come back refused or malformed. (a)-(c) rest on the harvester's contract: it
gives every record its own status inside the one transaction and commits the file `partial`, retrying a blocked record from its queue
while the folded withdrawal stands -- a transient zero-contrast key this emitter must never create; no file-level all-or-nothing exists
or is promised (the thrown-exception rollback is SQLite's, not a contract), so with every dependency verified folded in the destination
and every record validated, every record of the file applies and the transition folds as a unit in practice. The records are never
reordered and a transition is never split into two files (that trades the transient zero for a transient double). The pending guards
and the source heads are read from the destination too; harvest.py and src/sisteer/registry/ are untouched.

The destination is decided by CONTAINMENT of real paths, never by lexical
ancestry. :func:`resolve_destination` resolves the spool with ``os.path.realpath`` (every component, symlinks included) and requires it to
lie under the REAL records tree of the destination registry -- ``realpath(<root>)/records`` of the attested-or-queried root, compared by
path components (``os.path.commonpath``), never by string prefix. A spool reached through a symlink (``A/records -> B/...``), a spool
outside every records tree, or a spool under ANOTHER registry's tree is refused by name with both real paths (and the registry the real
path does belong to) in the message, EVEN WITH ``--destination-registry-root``: the attestation names WHICH registry's folded rows are
checked and harvest the file, it never waives containment; there is no branch that accepts an attested root for a spool under no records
tree. The destination's folded-row checks and the record validation stay as above.
stdlib + ``sisteer.registry`` only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import os
import random
import sqlite3
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from sisteer import registry  # noqa: E402
from sisteer.registry import db as DB  # noqa: E402
from sisteer.registry import emit  # noqa: E402
from sisteer.registry import harvest as H  # noqa: E402
from sisteer.registry import identity  # noqa: E402
from sisteer.registry import paper  # noqa: E402
from sisteer.registry import records as R  # noqa: E402
from sisteer.registry import validate as VAL  # noqa: E402
from sisteer.registry import schema_def as SD  # noqa: E402
from sisteer.registry import vocab as V  # noqa: E402
from sisteer.registry import worklist as RW  # noqa: E402

PRODUCER_SCRIPT = "scripts/registry/emit_contrasts.py"
DELTA_METHOD = "paired_bootstrap_percentile_v1"
BOOTSTRAP_B = 2000
DEFAULT_COMPUTED_BY = "producer"
ANCHOR_SAME_JOB = "same_standup_none"
#: The u0 none of the same (campaign, eval, instrument, model) anchoring a steered row of another view -- the
#: standup's own none EXCLUDED by ruling (a steered view whose own none runs a ruling excludes from the baseline) or ABSENT from
#: the view (the s3r view carries only the difference_relu cell). Filed BESIDE the same-standup row (migration
#: 0021 retires those from the frame), with extra.contrast_caps = {steered, anchor} (equal by construction: the drawn cap
#: is a filter on the anchor), extra.anchor_standup
#: (the u0 standup's identity) and extra.cross_server true. The anchor is looked up in v_run_wide (the authority), never derived, and
#: is CURRENT by the paper's own rules (RUN_CURRENCY_VIEW) -- a retired run is never picked and never keeps a filed row.
ANCHOR_BORROWED = "borrowed"
#: The static units' u0 none and s2r steered runs are the mis-stamped rows the correction op re-registers under NEW run
#: ids, so a borrowed contrast filed now would name a run about to be superseded -- deferred for these eval categories until the op re-emits
#: on the new ids (`--borrowed-defer-category none` lifts it); over-refusal, agentic and multi-turn re-emit right away.
BORROWED_DEFER_CATEGORIES: Tuple[str, ...] = ("static_attack",)
_U0_VIEW_RE = re.compile(r"_s[0-9]+rl?$")   # S2RLFIX5-PETRI: the s2rl view strips to its u0 anchor view like every s<N>r view
ANCHOR_SIBLING = "sibling_none"
#: The same-view kinds this emitter files -- a filed row of either kind is kept while its anchor is current under the paper's
#: rules and makes its run DUE again when it is not (`full_dose` is the export adapter's kind and is never filed or re-filed here).
SAME_VIEW_KINDS: Tuple[str, ...] = (ANCHOR_SAME_JOB, ANCHOR_SIBLING)
#: WHO wrote a contrast row is read from its record_uri PREFIX -- the uris are
#: relative (`records/...`): `records/adapters/` is an ADAPTER's row (the r2 export, the shared bfcl / bigcodebench readers, relu_native: 11,216
#: rows live, every borrowed one among them), never kept, retired or re-filed by this emitter; `records/<campaign>/` is this emitter's (native;
#: 53 live, all same_standup_none); anything else is `other` (a spool outside the standard layout -- still not an adapter's, so treated as ours).
ADAPTER_URI_PREFIX = "records/adapters/"
assert ANCHOR_SAME_JOB in V.ANCHOR_KINDS and ANCHOR_SIBLING in V.ANCHOR_KINDS


def uri_class(record_uri: Optional[str], campaign: Optional[str]) -> str:
    """``adapter`` / ``native`` / ``other`` for a contrast row's ``record_uri``: the adapter prefix first; the campaign's own
    `records/<campaign>/` is native; with no campaign to name, any other `records/` uri reads native; anything else is other."""
    uri = str(record_uri or "")
    if uri.startswith(ADAPTER_URI_PREFIX):
        return "adapter"
    if campaign and uri.startswith(f"records/{campaign}/"):
        return "native"
    return "native" if (not campaign and uri.startswith("records/")) else "other"
assert DEFAULT_COMPUTED_BY in V.COMPUTED_BY

_RUN_COLS = ("run_id", "campaign", "eval_id", "instrument_id", "model_id", "object_id", "cluster_job_id", "producer_run_key_legacy",
             "trial_index", "queued_utc", "view_legacy", "token_cap_tokens", "instrument_padding_tokens", "eval_category", "lane_source_legacy")


def u0_view_of(view: Optional[str]) -> Optional[str]:
    """The u0 view a steered view borrows its none from (``<model key>_s<N>r`` -> ``<model key>_u0``; and a FRONT-DOOR target's dose
    view ``<model key>_d<dose>[_relu|_relul]`` -> ``<model key>_u0`` by the models table,
    sisteer.front_door_views, read once per revision and only for a dose-shaped view: contrasts_due calls this for every complete steered run
    each daemon tick); None for a view that is not a steered-wave view (the u0 view itself, a legacy spelling, a paper target's
    same-wave-anchored ``d0p1`` …): no borrowed anchor applies. A checkout without reproduce/lib/models.py refuses a dose-shaped view by name."""
    if not view:
        return None
    v = str(view)
    if _U0_VIEW_RE.search(v):
        return _U0_VIEW_RE.sub("_u0", v)
    from sisteer import front_door_views as fdv
    key = fdv.front_door_dose_tag(v)
    # the anchor is the TABLE's baseline wave (u0 for a phased key; a single-phase key's dose views are never front-door dose tags — they
    # carry their own none, so no borrowed anchor applies and front_door_dose_tag is None for them)
    return f"{key}_{fdv.baseline_view(key)}" if key else None


def excluded_run_ids(conn: sqlite3.Connection) -> Dict[str, str]:
    """{run_id: ruling_id} over the ACTIVE `exclude` rulings whose scope names run ids (expired_utc null) -- how a standup's own none
    reads as excluded by ruling (the r2v4 exclusion of the s2r none runs)."""
    out: Dict[str, str] = {}
    for r in _rows(conn, "SELECT ruling_id, scope_json FROM ruling WHERE action = 'exclude' AND expired_utc IS NULL ORDER BY ruling_id"):
        try:
            scope = json.loads(r.get("scope_json") or "{}") if isinstance(r.get("scope_json"), str) else (r.get("scope_json") or {})
        except ValueError:
            continue
        for rid in ((scope.get("match") or {}).get("run_ids") or []) if isinstance(scope, Mapping) else []:
            out.setdefault(str(rid), str(r["ruling_id"]))
    return out


class NoCapMatchedAnchor(LookupError):
    """No complete unpadded u0 none carries the steered row's DRAWN cap (or the row carries none): no borrowed anchor, by name."""


def pick_borrowed_anchor(run: Mapping[str, object], candidates: Sequence[Mapping[str, object]],
                         currency: Optional[Mapping[str, Mapping[str, object]]] = None,
                         rostered: Optional[bool] = None) -> Optional[Tuple[Dict[str, object], int]]:
    """``(anchor row, n_candidates)``: the complete u0 ``none`` run of the steered row's key (``candidates``, already restricted to the u0
    view), in this order: never a PADDED none (``instrument_padding_tokens`` set: the static padding leg); then
    only a run CURRENT under the paper's own rules (``currency`` from :func:`run_currency`; None = no currency filter, the pure unit form) --
    without that filter the pool was every complete none, so when a filed anchor was superseded the earliest-queued
    retired run won the re-pick, the existing row read exists_registry and the steered metric was left without a current borrowed delta;
    then ONLY one drawn at the steered row's cap -- exact non-null ``token_cap_tokens`` equality is a FILTER before any tie-break
    (one run per rung puts 4096 / 8192 / 16384 u0 none runs under one instrument, and a wrong-cap none with the
    matching trial must never beat a cap-matched one). Several cap-matched: the row's ``trial_index`` first (t0 <-> t0 share the shard
    split), then the EARLIEST ``queued_utc``, then the run id -- deterministic and stable under later arrivals. None when no
    unpadded CURRENT candidate remains; :class:`NoCapMatchedAnchor` when such candidates exist but none carries the cap (or the row
    carries no cap). The pick is tiered -- :func:`pick_borrowed_anchor_tiered`, whose third element names the tier; this form keeps
    the ``(anchor, n_cap_matched)`` pair (``rostered`` as there)."""
    picked = pick_borrowed_anchor_tiered(run, candidates, currency=currency, rostered=rostered)
    return None if picked is None else (picked[0], picked[1])


#: The borrowed anchor's preference order, mirrored EXACTLY by :func:`pick_borrowed_anchor_tiered` -- (tier, name, rule) -- so this
#: emitter and the registry's own reader never pick different baselines for one key. Tier 1 keys on
#: the standup (equal non-null `cluster_job_id`, or equal `producer_run_key_legacy`: the same-job rule `anchor_for` reads); tier 2 on the
#: paper's source string of the run, `run.lane_source_legacy` (equal and non-null on both sides -- the column the member predicate and
#: `source_status.lane_source_legacy` key on; the native sdfbase runs carry none, so tier 2 never fires for them); tier 3 on the frame's
#: reading (`selection_state selected` AND `in_current_paper 1`: a roster entry matches the none); tier 4 applies ONLY when no roster entry of
#: the open round can match the model (:func:`roster_models`) -- a companion-only model, one no roster entry names. Within every tier:
#: is_current 1 and no member label (:func:`anchor_eligible`), the drawn cap a FILTER before the tiers, then the existing tie-break.
BORROWED_TIERS: Tuple[Tuple[int, str, str], ...] = (
    (1, "own_standup", "the steered run's own standup: a current u0 none registered by the same job (cluster_job_id) or producer run key"),
    (2, "same_source", "a current u0 none of the same source (run.lane_source_legacy equal, non-null)"),
    (3, "paper_baseline", "the paper's roster-matched, in-paper none of another source (selection_state selected, in_current_paper 1)"),
    (4, "any_current", "no roster entry of the open round names the model (companion-only): any current none, the existing tie-break"),
)


class OffRosterAnchorsOnly(LookupError):
    """Current cap-matched u0 nones stand, but none is the run's own standup, its source or a roster-matched paper row, and the model
    HAS roster entries -- tier 4 (any current none) is for a model no roster entry names. No borrowed anchor, by name."""


def roster_models(conn: sqlite3.Connection) -> Set[str]:
    """The model ids a roster entry of the OPEN round can match (the round and roster-version rules of `v_roster_match`; the eval is not
    part of it: a model with any entry is a paper model). Tier 4 applies to a model outside this set only. Read once per pass."""
    sql = ("SELECT DISTINCT re.model_id FROM roster_entry re "
           "WHERE re.round_id = (SELECT round_id FROM paper_round WHERE closed_utc IS NULL ORDER BY opened_utc DESC LIMIT 1) "
           "AND ((SELECT roster_version FROM paper_round pr WHERE pr.round_id = re.round_id) IS NULL "
           "OR re.roster_version = (SELECT roster_version FROM paper_round pr WHERE pr.round_id = re.round_id))")
    return {str(r[0]) for r in conn.execute(sql)}


def pick_borrowed_anchor_tiered(run: Mapping[str, object], candidates: Sequence[Mapping[str, object]],
                                currency: Optional[Mapping[str, Mapping[str, object]]] = None,
                                rostered: Optional[bool] = None) -> Optional[Tuple[Dict[str, object], int, Dict[str, object]]]:
    """``(anchor row, n_cap_matched, tier)`` -- :func:`pick_borrowed_anchor` with the tier that fired: the filters first exactly as
    before (never a padded none; only a current, non-member none when ``currency`` is given; the drawn cap, :class:`NoCapMatchedAnchor` by
    name), then :data:`BORROWED_TIERS` in order over the cap-matched pool, the first non-empty tier deciding; within it the run's
    ``trial_index`` first, then the EARLIEST ``queued_utc``, then the run id. ``rostered`` says whether a roster entry can match the run's
    model (:func:`roster_models`); None (the pure unit form, no registry) admits tier 4 as the old single-tier rule did. Tier 3 needs
    ``currency`` (the view's selection_state / in_current_paper per candidate). None when no unpadded current candidate remains;
    :class:`OffRosterAnchorsOnly` when candidates stand but only tier 4 would take them and the model is rostered. ``tier`` carries
    ``{tier, name, rule, n_in_tier, n_cap_matched, model_rostered}`` for the due row, the line and the record."""
    pool = [c for c in candidates if c.get("instrument_padding_tokens") is None]
    if currency is not None:
        pool = [c for c in pool if anchor_run_current(currency, str(c["run_id"]))]
    if not pool:
        return None
    cap = run.get("token_cap_tokens")
    if cap is None:
        raise NoCapMatchedAnchor(f"the steered run carries no token_cap_tokens; {len(pool)} unpadded u0 none candidate(s) with caps "
                                 f"{sorted({str(c.get('token_cap_tokens')) for c in pool})} cannot be matched")
    matched = [c for c in pool if c.get("token_cap_tokens") is not None and int(c["token_cap_tokens"]) == int(cap)]
    if not matched:
        raise NoCapMatchedAnchor(f"no unpadded u0 none drawn at the steered run's cap {cap}; candidates carry caps "
                                 f"{sorted({str(c.get('token_cap_tokens')) for c in pool})}")
    jid, rk, lane = run.get("cluster_job_id"), run.get("producer_run_key_legacy"), run.get("lane_source_legacy")

    def state(c: Mapping[str, object]) -> Mapping[str, object]:
        return (currency.get(str(c["run_id"])) or {}) if currency is not None else {}

    tiers: List[Tuple[int, str, str, List[Mapping[str, object]]]] = []
    for tier, name, rule in BORROWED_TIERS:
        if tier == 1:
            cands = [c for c in matched if (jid is not None and str(jid) != "" and c.get("cluster_job_id") == jid) or (rk and c.get("producer_run_key_legacy") == rk)]
        elif tier == 2:
            cands = [c for c in matched if lane and c.get("lane_source_legacy") == lane]
        elif tier == 3:
            cands = [c for c in matched if str(state(c).get("selection_state") or "") == "selected" and int(state(c).get("in_current_paper") or 0) == 1]
        else:
            cands = list(matched) if rostered is not True else []
        tiers.append((tier, name, rule, cands))
    trial = run.get("trial_index")
    for tier, name, rule, cands in tiers:
        if cands:
            chosen = min(cands, key=lambda c: (c.get("trial_index") != trial, str(c.get("queued_utc") or ""), str(c["run_id"])))
            return dict(chosen), len(matched), {"tier": tier, "name": name, "rule": rule, "n_in_tier": len(cands), "n_cap_matched": len(matched),
                                                 "model_rostered": rostered}
    raise OffRosterAnchorsOnly(f"{len(matched)} current cap-matched u0 none(s) stand -- " + ", ".join(
        f"{c['run_id']} (selection_state {state(c).get('selection_state')}, in_current_paper {state(c).get('in_current_paper')}, source {c.get('lane_source_legacy')!r})" for c in matched)
        + f" -- but none is the steered run's own standup (job {jid!r} / key {rk!r}), its source ({lane!r}) or a roster-matched paper row, and the "
        "model has roster entries: tier 4 (any current none) applies only to a model no roster entry names (D1546); no borrowed anchor")


#: The view the paper's run-currency rules are read from (a run-id predicate here would be blind to scoped / latest supersede / exclude
#: rulings and to later-run supersession). Since migration 0021 (schema 21) that is `v_run_current`: the ONE evaluation of the ruling
#: fold, exposing `is_current` and `why` -- the predicate lives
#: THERE and nowhere here; this module reads the two columns (plus the run's successor and drawn cap from `run`) and re-derives nothing.
#: Change the name HERE and nowhere else.
RUN_CURRENCY_VIEW = "v_run_current"
#: A pooled reading's per-trial / per-rung MEMBER run (`selection_state member_pass`) and
#: a labeled member whose point carries no pooled reading of its source (`unpooled_member`, migration 0022) are current measurements that
#: are not paper rows -- the view reads them is_current 1, in_current_paper 0 -- and must never become a borrowed or same-view anchor. Named
#: EXPLICITLY here (neither state is self-disqualifying in the view). Paper membership
#: itself is a FRAME reading, not an anchor-validity condition: requiring `in_current_paper = 1` as well retires every
#: own current none of a model no roster entry names (a companion-only model) and leaves its steered runs
#: without a borrowed baseline. :func:`anchor_eligible` applies the member mask ON TOP of the view's bit and can
#: only narrow it; a run the view retired is never turned back on here; `in_current_paper` is RECORDED (why / anchor_currency), never required.
ANCHOR_EXCLUDED_SELECTION_STATES: Tuple[str, ...] = ("member_pass", "unpooled_member")


def anchor_eligible(is_current: object, selection_state: Optional[str], in_current_paper: object = None) -> bool:
    """The view's ``is_current`` AND a selection_state outside :data:`ANCHOR_EXCLUDED_SELECTION_STATES` -- what every anchor decision in this
    module reads (a member run of a pooled reading, pooled or not, is never an anchor). ``in_current_paper`` is accepted so every
    caller keeps recording it (a frame reading, not a condition) and plays no part in the answer."""
    return bool(is_current) and str(selection_state or "") not in ANCHOR_EXCLUDED_SELECTION_STATES
#: The computation id this emitter files (a producer-computed row: the blank sentinel) -- the fifth column of the paper's contrast key
#: (run_id, judgment_id, metric_id, threshold, computation_id), on which `v_paper_long` joins a contrast to its metric row.
NATIVE_COMPUTATION_ID = ""

# --------------------------------------------------------------------------------------------
# The frame's own retirement of a run_contrast row -- the observation withdrawal (migration 0017)
# --------------------------------------------------------------------------------------------
#: `source_revision.adapter` and the `source_id` prefix of the per-key sources this emitter writes: ONE source per native contrast key
#: (`emit_contrasts:contrast:<sha16 of the canonical key>`), so the keys' revision chains never cross -- `observation.current` is derived
#: per SOURCE from its head revision (reducer._recompute_current: a key is current iff its head revision carries it), so under a
#: shared source every head move would withdraw or re-admit a neighbour's key, and two emitters stamping heads from one read of the
#: registry would race. The frame reads nothing but the key's observations (WITHDRAWN_CONTRAST_CTE: table_name run_contrast, GROUP BY
#: observation_key HAVING MAX(current) = 0); the source id is bookkeeping, named so a reader of `source_revision` knows whose rows they are.
CONTRAST_SOURCE_ADAPTER = "emit_contrasts"
CONTRAST_SOURCE_PREFIX = "emit_contrasts:contrast:"
#: `withdrawal.reason` -- the DDL's vocabulary (vocab.WITHDRAWAL_REASONS); the anchor's lost membership rides the record's `extra.why`.
WITHDRAWAL_REASON = "removed_measurement"
assert WITHDRAWAL_REASON in V.WITHDRAWAL_REASONS
#: The paper's full contrast key -- the `run_contrast` primary key, what the frame's withdrawn_contrast CTE extracts from an observation
#: key by JSON path (threshold CAST AS REAL, the sentinels '' as filed).
CONTRAST_KEY_FIELDS: Tuple[str, ...] = ("run_id", "anchor_run_id", "judgment_id", "metric_id", "threshold", "computation_id")
ContrastKey = Tuple[str, str, str, str, float, str]


def contrast_key_tuple(key: Mapping[str, object]) -> ContrastKey:
    """``(run_id, anchor_run_id, judgment_id, metric_id, threshold, computation_id)`` of a key mapping, typed as the frame reads them."""
    return (str(key["run_id"]), str(key["anchor_run_id"]), str(key.get("judgment_id") or ""), str(key["metric_id"]),
            float(key.get("threshold", -1.0)), str(key.get("computation_id") or ""))


def contrast_observation_key(key: Mapping[str, object]) -> str:
    """The canonical JSON the frame's withdrawn_contrast CTE reads (and the observation / withdrawal records carry): the six key columns,
    threshold as a float, judgment_id and computation_id as the strings filed ('' sentinels included)."""
    t = contrast_key_tuple(key)
    return identity.canonical_json(dict(zip(CONTRAST_KEY_FIELDS, t)))


def contrast_source_id(key_json: str) -> str:
    """The per-key source of one native contrast key (``key_json`` from :func:`contrast_observation_key`)."""
    return CONTRAST_SOURCE_PREFIX + identity.sha16(key_json)


def frame_withdrawn_keys(conn: sqlite3.Connection) -> Set[ContrastKey]:
    """Every run_contrast key the paper frame leaves out by the observation rule -- the frame's OWN CTE (schema_def.WITHDRAWN_CONTRAST_CTE)
    run as a query, so this module and v_paper_long cannot disagree on what is withdrawn. Read once per pass (``contrasts_due``, ``run``,
    ``sweep_noncurrent``) and handed down; a caller without the set has it evaluated for it."""
    sql = f"WITH {SD.WITHDRAWN_CONTRAST_CTE} SELECT run_id, anchor_run_id, judgment_id, metric_id, threshold, computation_id FROM withdrawn_contrast"
    return {(str(r[0]), str(r[1]), str(r[2] or ""), str(r[3]), float(r[4] if r[4] is not None else -1.0), str(r[5] or "")) for r in conn.execute(sql)}


def run_currency(conn: sqlite3.Connection, object_id: Optional[str] = "none") -> Dict[str, Dict[str, object]]:
    """``{run_id: {is_current, view_is_current, why, selection_state, held_out, decisive_action, decisive_ruling_id, lifecycle_state,
    superseded_by_run_id, token_cap_tokens, hold_out_ruling_ids}}`` for every run of ``object_id`` (default the ``none`` anchors; None =
    every run), read from :data:`RUN_CURRENCY_VIEW` ONCE per pass (the view materialises its ruling and supersession CTEs per query, so a
    per-anchor probe would re-evaluate them per anchor). ``is_current`` is the view's bit refined by :func:`anchor_eligible` (a pooled
    reading's member run is never an anchor); ``view_is_current`` the bit exactly as the view reads it; ``why`` the view's own first failing
    rule -- `superseded_by_run_id <id>`, `selection_state superseded (ruling <id>)` / `(a later run at the same key)` / `(lifecycle or
    superseded_by_run_id)`, `selection_state excluded (ruling <id>)`, `held_out (ruling <id>)`, `lifecycle_state <state>`, `decisive ruling
    <action> (ruling <id>)` (a ruling retires a run outside the roster too) -- or ``current``; ``hold_out_ruling_ids`` the active
    hold_out rulings among the run's covering ones (what an operator lifts). The run's successor and drawn cap come from ``run`` (the
    view exposes neither). A run the view does not carry is absent, and :func:`anchor_run_current` reads that as not current."""
    sql = (f"SELECT vc.run_id, vc.is_current AS view_is_current, vc.why, vc.selection_state, vc.in_current_paper, vc.held_out, vc.decisive_action, "
           f"vc.decisive_ruling_id, vc.ruling_ids_json, r.lifecycle_state, r.superseded_by_run_id, r.token_cap_tokens, r.object_id "
           f"FROM {RUN_CURRENCY_VIEW} vc JOIN run r ON r.run_id = vc.run_id")
    params: List[object] = []
    if object_id is not None:
        sql += " WHERE r.object_id = ?"
        params.append(object_id)
    hold_outs = {str(r[0]) for r in conn.execute("SELECT ruling_id FROM ruling WHERE action = 'hold_out' AND expired_utc IS NULL")}
    out: Dict[str, Dict[str, object]] = {}
    for row in _rows(conn, sql, params):
        eligible = anchor_eligible(row.get("view_is_current"), row.get("selection_state"), row.get("in_current_paper"))   # type: ignore[arg-type]
        why = str(row.get("why") or "") or "current"
        if row.get("view_is_current") and not eligible:
            # current by the view, yet never an anchor -- a member run of a pooled reading (pooled or unpooled)
            why = f"selection_state {row.get('selection_state')} (a pooled reading's member run: a current measurement, not a paper row, never an anchor)"
        elif eligible and int(row.get("in_current_paper") or 0) != 1:
            # paper membership is a frame reading, recorded beside the currency, never an anchor condition
            why = f"current; in_current_paper {row.get('in_current_paper')} (not a paper row: recorded, not an anchor condition -- D1544)"
        named: List[str] = []
        if row.get("held_out") == 1:
            covering = row.get("ruling_ids_json")
            try:
                covering = json.loads(covering) if isinstance(covering, str) else list(covering or [])   # type: ignore[arg-type]
            except ValueError:
                covering = []
            named = sorted(str(x) for x in covering if str(x) in hold_outs)   # type: ignore[union-attr]
        out[str(row["run_id"])] = {**row, "view_is_current": bool(row.get("view_is_current")), "is_current": eligible, "why": why, "hold_out_ruling_ids": named}
    return out


def anchor_run_current(currency: Mapping[str, Mapping[str, object]], run_id: str) -> bool:
    """An anchor run is CURRENT under the paper's own rules (:func:`run_currency`): the view's selection does not read it superseded /
    withdrawn / excluded, it is not held out, not in a terminal lifecycle, and has no explicit successor. A run the view does not carry
    is not current. A contrast anchored on a retired run leaves the frame on the view's next read, so the borrowed row is re-chosen
    exactly then -- and only then."""
    return bool((currency.get(str(run_id)) or {}).get("is_current"))


def current_judgment_ids(conn: sqlite3.Connection, run_id: str, metric_id: str) -> List[str]:
    """The judgment ids of the run's CURRENT ``metric_id`` items whose judgment is COMPLETE (or the rule-scored ``''`` sentinel): the
    run's current judgment for the metric, the one that names the paper's contrast key. A re-judge files a NEW judgment
    id (``identity.judgment_id`` over run | judge | session | claimed_utc) and marks the old one ``superseded``; the old item rows stay
    (their key carries the judgment id), so the item view alone names two judgments -- the status is what resolves one."""
    return sorted({str(j) for (j,) in conn.execute(
        "SELECT DISTINCT ir.judgment_id FROM v_item_result_current ir LEFT JOIN judgment j ON j.judgment_id = ir.judgment_id "
        "WHERE ir.run_id = ? AND ir.metric_id = ? AND (ir.judgment_id = '' OR j.status = 'complete')", (run_id, metric_id))})


def contrast_keep(conn: sqlite3.Connection, run: Mapping[str, object], metric_id: str, currency: Mapping[str, Mapping[str, object]], *,
                  kinds: Sequence[str] = (ANCHOR_BORROWED,), check_cap: bool = True,
                  withdrawn: Optional[Set[ContrastKey]] = None) -> Dict[str, object]:
    """The contrasts of ``kinds`` already filed for the steered ``run`` at the paper's FULL contrast key -- (run_id, the run's CURRENT
    judgment, metric_id, that judgment's threshold, this emitter's computation id), the key ``v_paper_long`` joins a contrast on (a
    check keyed by (run, metric, anchor_kind) alone kept a row filed under a re-judged run's OLD judgment,
    and the new judgment's metric never got a borrowed contrast) -- each read against the paper's run-currency rules
    (:func:`anchor_run_current`) and, with ``check_cap``, the drawn caps (when the cap filter ran only where no filed row had
    a current anchor, a wrong-cap row an earlier implementation filed was kept for ever). This is the one check
    for every kind this emitter files: the same-view kinds (:data:`SAME_VIEW_KINDS`, no cap check -- a same-view none shares the run's
    instrument) and the borrowed one. Returns ``{key, judgment_ids, kept, kept_kind, mismatch, retired}``: ``kept`` the anchor run id of
    a current (and cap-consistent) row with its ``kept_kind`` (None when none); ``mismatch`` the current rows whose non-null caps differ
    from the run's (``[{anchor_run_id, anchor_kind, caps}]``: `borrowed_cap_mismatch` -- nothing is written while one stands, and the
    remediation is the view's own, retire that anchor); ``retired`` the rows whose anchor the view no longer carries as current
    (``[{anchor_run_id, anchor_kind, why}]``: the run is DUE again). With no single current judgment there is no key (``key`` None):
    nothing is kept, and ``emit_contrast`` refuses the pair by name.

    ``frame`` maps every filed native anchor run id to the frame's view of its row -- its full ``contrast_key``,
    ``frame_visible`` (the view reads the anchor current and no observation withdrew the key: v_paper_long STILL joins it, so a
    replacement emission must withdraw it explicitly), ``frame_withdrawn`` (the observation rule already leaves it out), the anchor's
    ``why`` / ``selection_state`` / ``in_current_paper`` -- a PARALLEL map, so the ``retired`` / ``mismatch`` / ``stale_judgment`` entries
    keep their shape; ``readmit`` lists the rows that would be kept but for a withdrawn key (an earlier
    emission retired them and their anchor is eligible again): not kept, re-observed by the pass on their own key. ``withdrawn``
    is :func:`frame_withdrawn_keys` (evaluated here when not handed in)."""
    run_id = str(run["run_id"])
    jids = current_judgment_ids(conn, run_id, metric_id)
    out: Dict[str, object] = {"key": None, "judgment_ids": jids, "kept": None, "kept_kind": None, "mismatch": [], "retired": [], "stale_judgment": [],
                              "adapter": [], "filed": 0, "readmit": [], "frame": {}}
    if len(jids) != 1:
        return out
    jid = jids[0]
    thr = threshold_for(conn, run_id, jid, metric_id)
    out["key"] = {"run_id": run_id, "judgment_id": jid, "metric_id": metric_id, "threshold": thr, "computation_id": NATIVE_COMPUTATION_ID}
    cap = run.get("token_cap_tokens")
    marks = ", ".join("?" for _ in kinds)
    seen: Set[Tuple[str, str]] = set()
    wd = frame_withdrawn_keys(conn) if withdrawn is None else withdrawn   # the frame's observation rule, read once per pass
    for aid, akind, ajid, uri in conn.execute("SELECT DISTINCT anchor_run_id, anchor_kind, anchor_judgment_id, record_uri FROM run_contrast WHERE run_id = ? "
                                              f"AND judgment_id = ? AND metric_id = ? AND threshold = ? AND computation_id = ? AND anchor_kind IN ({marks}) "
                                              "ORDER BY anchor_run_id, anchor_kind, record_uri",
                                              (run_id, jid, metric_id, float(thr), NATIVE_COMPUTATION_ID, *[str(k) for k in kinds])):
        aid, akind = str(aid), str(akind)
        cur = currency.get(aid) or {}
        cls = uri_class(uri, run.get("campaign"))   # type: ignore[arg-type]
        if cls == "adapter":
            # an adapter's row is the adapter's -- reported (the sweep's table counts it), never kept, retired or re-filed here
            out["adapter"].append({"anchor_run_id": aid, "anchor_kind": akind, "record_uri": str(uri), "is_current": bool(cur.get("is_current")),   # type: ignore[union-attr]
                                   "why": str(cur.get("why") or f"not carried by {RUN_CURRENCY_VIEW}")})
            continue
        if (aid, akind) in seen:
            continue                                 # one native row per (anchor, kind) however many files carry it
        seen.add((aid, akind))
        out["filed"] = int(out["filed"]) + 1   # type: ignore[arg-type]
        # The row's full key and the frame's view of it -- the view's bit is what v_paper_long's contrast join reads, the
        # observation rule what leaves a row out; a row the view carries that this emitter retires is `frame_visible` until withdrawn
        ckey = {**out["key"], "anchor_run_id": aid}   # type: ignore[dict-item]
        in_wd = contrast_key_tuple(ckey) in wd
        frame = {"anchor_run_id": aid, "anchor_kind": akind, "contrast_key": ckey, "frame_withdrawn": in_wd,
                 "frame_visible": bool(cur.get("view_is_current")) and not in_wd, "why": str(cur.get("why") or f"not carried by {RUN_CURRENCY_VIEW}"),
                 "selection_state": cur.get("selection_state"), "in_current_paper": cur.get("in_current_paper")}
        out["frame"][aid] = frame   # type: ignore[index]  -- a PARALLEL map: the retired / mismatch / stale_judgment entries keep their existing shape
        if not cur.get("is_current"):
            out["retired"].append({"anchor_run_id": aid, "anchor_kind": akind, "why": str(cur.get("why") or f"not carried by {RUN_CURRENCY_VIEW}"), "uri_class": cls})   # type: ignore[union-attr]
            continue
        acap = cur.get("token_cap_tokens")
        if check_cap and cap is not None and acap is not None and int(cap) != int(acap):
            out["mismatch"].append({"anchor_run_id": aid, "anchor_kind": akind, "caps": {"steered": cap, "anchor": acap}, "uri_class": cls})   # type: ignore[union-attr]
            continue
        # The row's anchor JUDGMENT must be the anchor run's current complete judgment -- a re-judged
        # anchor (same run, new judgment and scores, the old one superseded) left the row kept and v_paper_long joining the stale delta (its
        # join reads no anchor judgment). A stale row is not kept: the pass re-emits on the SAME key with the new anchor judgment and delta,
        # and the fold (reducer.fold, a `full` record on an existing key sets every column it carries) updates the row in place -- never a
        # second row. An anchor with no single complete judgment cannot be compared and the row stands as filed.
        acur = current_judgment_ids(conn, aid, metric_id)
        if len(acur) == 1 and str(ajid or "") != acur[0]:
            out["stale_judgment"].append({"anchor_run_id": aid, "anchor_kind": akind, "anchor_judgment_id": str(ajid or ""),   # type: ignore[union-attr]
                                          "current_anchor_judgment_id": acur[0], "uri_class": cls})
            continue
        if in_wd:
            # The row would be kept, but the frame withdrew its key (an earlier emission retired it when its anchor lost paper
            # membership) and the anchor is a paper baseline row again: not kept -- the pass re-admits it on its own key (`inplace_target`)
            out["readmit"].append({**frame, "uri_class": cls,   # type: ignore[union-attr]
                                   "why": f"the frame withdrew this key; its anchor is an eligible anchor again (selection_state "
                                          f"{cur.get('selection_state')}, in_current_paper {cur.get('in_current_paper')} -- recorded, D1544)"})
            continue
        if out["kept"] is None:
            out["kept"], out["kept_kind"] = aid, akind
    return out


def borrowed_keep(conn: sqlite3.Connection, run: Mapping[str, object], metric_id: str,
                  currency: Mapping[str, Mapping[str, object]], withdrawn: Optional[Set[ContrastKey]] = None) -> Dict[str, object]:
    """:func:`contrast_keep` for the borrowed kind, caps checked (`borrowed_cap_mismatch` reads ``mismatch``)."""
    return contrast_keep(conn, run, metric_id, currency, kinds=(ANCHOR_BORROWED,), check_cap=True, withdrawn=withdrawn)


def rejudge_target(keep: Optional[Mapping[str, object]], candidates: Sequence[Mapping[str, object]],
                   currency: Mapping[str, Mapping[str, object]]) -> Optional[Tuple[Dict[str, object], str]]:
    """The anchor a filed row is re-judged against -- the FIRST ``stale_judgment`` row of ``keep`` whose
    anchor run is current and among ``candidates`` -> (that candidate row, the row's filed anchor_kind); None when there is none (a first
    emission, or every filed anchor retired: a fresh pick then). A stale row's anchor RUN is current by construction (``contrast_keep``
    classifies a retired run as ``retired`` first), so the re-emission lands on the row's OWN key with its own kind and the fold updates
    anchor_judgment_id and the delta in place; a fresh pick here moved to a newer none and filed a second current key beside the stale one."""
    if not keep or not keep.get("stale_judgment"):
        return None
    by_id = {str(c["run_id"]): c for c in candidates}
    for s in keep["stale_judgment"]:   # type: ignore[union-attr]
        aid = str(s["anchor_run_id"])
        if aid in by_id and anchor_run_current(currency, aid):
            return dict(by_id[aid]), str(s["anchor_kind"])
    return None


def inplace_target(keep: Optional[Mapping[str, object]], candidates: Sequence[Mapping[str, object]],
                   currency: Mapping[str, Mapping[str, object]]) -> Optional[Tuple[Dict[str, object], str]]:
    """The anchor a filed row is re-emitted against ON ITS OWN KEY -- :func:`rejudge_target` first (a re-judged anchor), else the
    first ``readmit`` row of ``keep`` whose anchor run is current and among ``candidates``: a key the frame withdrew
    (an earlier emission retired it when its anchor lost paper membership) whose anchor is a paper baseline row again is re-observed under
    a new head revision of its source (``emit_contrast`` -> ``readmitted``), never a fresh key beside it. None when neither applies (a
    first emission, or every filed anchor retired: a fresh pick then)."""
    target = rejudge_target(keep, candidates, currency)
    if target is not None or not keep or not keep.get("readmit"):
        return target
    by_id = {str(c["run_id"]): c for c in candidates}
    for s in keep["readmit"]:   # type: ignore[union-attr]
        aid = str(s["anchor_run_id"])
        if aid in by_id and anchor_run_current(currency, aid):
            return dict(by_id[aid]), str(s["anchor_kind"])
    return None


def _frame_stale(keep: Optional[Mapping[str, object]]) -> List[Dict[str, object]]:
    """The filed rows this emitter retires that the paper frame STILL joins: the ``retired`` entries of ``keep`` whose frame view
    reads ``frame_visible`` -- the merged entry (the retired entry plus its key and frame fields), what the withdrawal writer reads."""
    if not keep:
        return []
    frame = keep.get("frame") or {}
    return [{**x, **frame[str(x["anchor_run_id"])]} for x in keep.get("retired") or []   # type: ignore[index]
            if (frame.get(str(x["anchor_run_id"])) or {}).get("frame_visible")]


def _withdraw_stale_note(stale: Sequence[Mapping[str, object]]) -> str:
    """The note a kept row carries when frame-visible retired keys stand beside it: withdrawn by this pass, no replacement needed."""
    return (f"; {len(stale)} filed row(s) this emitter retires that the paper frame STILL joins beside the kept row: withdrawn by this pass "
            "(g28-fix-6, no replacement: the kept row is the frame's contrast) -- "
            + "; ".join(f"{x['anchor_run_id']} ({x['anchor_kind']}): {x['why']}" for x in stale))


def _frame_stale_note(stale: Sequence[Mapping[str, object]]) -> str:
    """The note a `no_anchor` / `no_u0_anchor` row carries when the paper frame STILL joins filed rows this emitter retires."""
    return (f"; {len(stale)} filed row(s) the paper frame STILL joins ({RUN_CURRENCY_VIEW} reads the anchor current; this emitter retires it: a "
            "pooled reading's member run is never an anchor): withdrawn when a replacement is filed (g28-fix-5), until then left to the frame's own rule -- "
            + "; ".join(f"{x['anchor_run_id']} ({x['anchor_kind']}): {x['why']}" for x in stale))


def current_borrowed_anchor(conn: sqlite3.Connection, run_id: str, metric_id: str,
                            currency: Optional[Mapping[str, Mapping[str, object]]] = None) -> Optional[str]:
    """The anchor run id of the borrowed contrast KEPT for (run, metric) -- :func:`borrowed_keep` over the run's ``v_run_wide`` row (the
    currency evaluated here when not handed in); None when nothing is kept (the first emission, the filed anchor retired by the view's
    rules, a cap mismatch, no single current judgment). A test / operator convenience over the same code the pass runs."""
    rows = _rows(conn, "SELECT " + ", ".join(_RUN_COLS) + " FROM v_run_wide WHERE run_id = ?", (run_id,))
    if not rows:
        return None
    kept = borrowed_keep(conn, rows[0], metric_id, currency if currency is not None else run_currency(conn))["kept"]
    return str(kept) if kept is not None else None


# --------------------------------------------------------------------------------------------
# selection: complete steered runs and their complete `none` anchors
# --------------------------------------------------------------------------------------------

def _rows(conn: sqlite3.Connection, sql: str, params: Sequence[object] = ()) -> List[Dict[str, object]]:
    cur = conn.execute(sql, tuple(params))
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _in(col: str, values: Optional[Sequence[str]], params: List[object]) -> str:
    if not values:
        return ""
    params.extend(str(v) for v in values)
    return f" AND {col} IN ({','.join('?' for _ in values)})"


def complete_runs(conn: sqlite3.Connection, *, steered: bool, model_ids: Optional[Sequence[str]] = None, campaign: Optional[str] = None,
                  eval_ids: Optional[Sequence[str]] = None) -> List[Dict[str, object]]:
    """``v_run_wide`` rows with ``is_complete = 1`` -- steered (``object_id <> 'none'``) or the ``none`` anchors -- under the selection."""
    params: List[object] = []
    sql = ("SELECT " + ", ".join(_RUN_COLS) + " FROM v_run_wide WHERE is_complete = 1 AND object_id "
           + ("<>" if steered else "=") + " 'none'")
    sql += _in("model_id", model_ids, params)
    if campaign:
        sql += " AND campaign = ?"
        params.append(campaign)
    sql += _in("eval_id", eval_ids, params)
    sql += " ORDER BY run_id"
    return _rows(conn, sql, params)


def anchor_for(run: Mapping[str, object], candidates: Sequence[Mapping[str, object]],
               currency: Optional[Mapping[str, Mapping[str, object]]] = None) -> Optional[Tuple[Dict[str, object], str, int]]:
    """``(anchor row, anchor_kind, n_candidates)`` for ``run`` among the complete ``none`` runs of its campaign x eval x instrument x
    model -- of the run's own view, and CURRENT under the paper's rules when ``currency`` is given (None = no
    currency filter, the pure unit form; a retired none is never an anchor, so a filed row whose anchor the view retired is re-filed
    against a current one and never re-selects the retired run): a candidate registered by the SAME job (equal non-null
    ``cluster_job_id``, or equal non-null ``producer_run_key_legacy``) is ``same_standup_none``; else any complete sibling is
    ``sibling_none``. Several in the winning class: the one sharing the run's ``trial_index`` first, then the latest ``queued_utc``,
    then by run id. None when there is no (current) candidate."""
    jid, rk = run.get("cluster_job_id"), run.get("producer_run_key_legacy")
    # The same-view path never crosses views -- a none of another view reaches a steered row only as the BORROWED u0 anchor
    candidates = [c for c in candidates if (c.get("view_legacy") or None) == (run.get("view_legacy") or None)]
    if currency is not None:
        candidates = [c for c in candidates if anchor_run_current(currency, str(c["run_id"]))]
    same = [c for c in candidates
            if (jid is not None and str(jid) != "" and c.get("cluster_job_id") == jid) or (rk and c.get("producer_run_key_legacy") == rk)]
    pool, kind = (same, ANCHOR_SAME_JOB) if same else (list(candidates), ANCHOR_SIBLING)
    if not pool:
        return None
    trial = run.get("trial_index")
    # one lexicographic max: the run's trial first (True > False), then the latest queued, then by run id (unique, so never a tie)
    chosen = max(pool, key=lambda c: (c.get("trial_index") == trial, str(c.get("queued_utc") or ""), str(c["run_id"])))
    return dict(chosen), kind, len(candidates)


def _eval_row(conn: sqlite3.Connection, eval_id: str) -> Dict[str, object]:
    rows = _rows(conn, "SELECT eval_id, headline_metric_id, extra_json FROM eval WHERE eval_id = ?", (eval_id,))
    return rows[0] if rows else {"eval_id": eval_id, "headline_metric_id": None, "extra_json": None}


def _filed_item_metrics(conn: sqlite3.Connection, run_id: str) -> List[str]:
    return [str(r[0]) for r in conn.execute("SELECT DISTINCT metric_id FROM item_result WHERE run_id = ? ORDER BY metric_id", (run_id,))]


def _runs_with_complete_judgment_or_rule_scored_headline(conn: sqlite3.Connection) -> Set[str]:
    """Every run whose headline reading is filed (run-scale: one query for the pass, not one per steered run and per anchor): a run with
    a ``complete`` judgment (judge-scored), UNION a RULE-SCORED run -- an instrument that names no judge -- whose headline metric
    (``eval.headline_metric_id``) is filed under the ``''`` judgment sentinel, the row rule scoring files at generation and the
    harvester's completion evidence for such a run (``harvest._RULE_SCORED_HEADLINE_SQL``; its item rows carry the same sentinel, which
    :func:`current_judgment_ids` and ``registry.paired`` already admit). The judgment-only form refused every mmlu / gsm8k run as
    ``no_complete_judgment``, so no capability contrast was ever emitted natively. A judge-scored run with a generation-time row under
    the sentinel is NOT admitted by it (``i.judge_id IS NULL``): its headline reading is its judge's."""
    return {str(r[0]) for r in conn.execute(
        "SELECT DISTINCT run_id FROM judgment WHERE status = 'complete' "
        "UNION SELECT DISTINCT rm.run_id FROM run_metric rm JOIN run r ON r.run_id = rm.run_id JOIN eval e ON e.eval_id = r.eval_id "
        "JOIN instrument i ON i.instrument_id = r.instrument_id "
        "WHERE rm.judgment_id = '' AND i.judge_id IS NULL AND rm.metric_id = e.headline_metric_id")}


def _current_provenance(conn: sqlite3.Connection, run_id: str, metric_id: str) -> Tuple[List[str], List[str]]:
    """``(judgment ids, source revision ids)`` of the CURRENT ``metric_id`` items of ``run_id`` -- the rows ``registry.paired`` reads.
    The judgment ids are the run's COMPLETE judgments' (:func:`current_judgment_ids`; a re-judged run's superseded judgment
    keeps its item rows in the current view beside the new ones, and only the complete one names the record key -- one per side, else
    ``paired`` refuses); a non-empty revision id on ANY current row means an adapter carries that side, so the pair is not this
    emitter's."""
    revs: Set[str] = set()
    for (rev,) in conn.execute("SELECT DISTINCT source_revision_id FROM v_item_result_current WHERE run_id = ? AND metric_id = ? "
                               "AND source_revision_id <> ''", (run_id, metric_id)):
        revs.add(str(rev))
    return current_judgment_ids(conn, run_id, metric_id), sorted(revs)


def _adapter_backed(conn: sqlite3.Connection, run_id: str, anchor_run_id: str, metric_id: str) -> Optional[Dict[str, List[str]]]:
    """The adapter source revision ids the pair's current ``metric_id`` items ride, per side -- ``None`` when both sides are native."""
    _j, revs = _current_provenance(conn, run_id, metric_id)
    _aj, arevs = _current_provenance(conn, anchor_run_id, metric_id)
    return {"run": revs, "anchor": arevs} if (revs or arevs) else None


def contrasts_due(conn: sqlite3.Connection, model_ids: Optional[Sequence[str]] = None, campaign: Optional[str] = None,
                  eval_ids: Optional[Sequence[str]] = None,
                  borrowed_defer_categories: Sequence[str] = BORROWED_DEFER_CATEGORIES) -> List[Dict[str, object]]:
    """Every (complete steered run, complete ``none`` anchor, headline metric) triple the selection owes a contrast, plus the
    steered runs it cannot pair, each with a ``status``: ``due`` (pairable), ``no_anchor`` (no complete ``none`` run of the same
    campaign x eval x instrument x model), ``no_complete_judgment`` (a complete run whose headline reading is not filed -- a judge-scored
    run's judgment not ``complete``, a rule-scored run's headline metric not under the ``''`` sentinel -- or whose anchor's is not),
    ``no_items`` (no item rows under the eval's headline metric -- the filed metric ids are named), ``adapter_backed``
    (either side's current items under the metric ride an adapter source revision -- the ids are named per side; the adapter owns
    the pair). The daemon emits the ``due`` ones; the rest are printed so a gap is visible, never silently dropped.

    The BORROWED row of a steered-wave view (one per steered run beside the same-view row) carries its own statuses:
    ``borrowed_deferred`` (the eval category awaits re-registration), ``borrowed_current`` (a borrowed contrast at the run's
    current judgment's key is filed, its anchor is current under the paper's rules and the caps agree: kept, nothing written),
    ``borrowed_cap_mismatch`` (such a row stands against a current anchor drawn at ANOTHER cap: not kept, nothing written while it stands
    -- retire that anchor by the view's rules), ``no_cap_matched_u0_anchor`` (current unpadded u0 none runs exist, none at the run's cap),
    ``no_u0_anchor`` (no current unpadded u0 none at all; the retired ones are counted and named) or ``due`` / the checks above.

    The SAME-VIEW row too is kept and re-filed through the paper's rules -- ``same_view_current`` (a same-view contrast at
    the run's current judgment's key is filed and its anchor is current: kept, nothing written, never a second row beside it, whatever a
    fresh pick would name); a filed row whose anchor the view retired makes the run ``due`` again against a current same-view none, or
    ``no_anchor`` naming the retired anchor when none remains. Every row re-filed over a retired one carries ``retired_anchors``
    (anchor run id, kind, why) -- what ``sweep_noncurrent`` enumerates. ``view_legacy`` and ``eval_category`` ride every row."""
    steered = complete_runs(conn, steered=True, model_ids=model_ids, campaign=campaign, eval_ids=eval_ids)
    anchors = complete_runs(conn, steered=False, model_ids=model_ids, campaign=campaign, eval_ids=eval_ids)
    by_key: Dict[Tuple[str, str, str, str], List[Dict[str, object]]] = {}
    for a in anchors:
        by_key.setdefault((str(a["campaign"]), str(a["eval_id"]), str(a["instrument_id"]), str(a["model_id"])), []).append(a)
    evals: Dict[str, Dict[str, object]] = {}
    scored = _runs_with_complete_judgment_or_rule_scored_headline(conn)   # A complete judgment, or the rule-scored headline under ''
    excluded = excluded_run_ids(conn)
    currency = run_currency(conn)                    # the paper's run-currency rules over the none anchors, read once per pass
    withdrawn = frame_withdrawn_keys(conn)           # the frame's observation rule over the contrast keys, read once per pass
    rostered_models = roster_models(conn)            # The models a roster entry of the open round can match (tier 4 is for the others)
    out: List[Dict[str, object]] = []

    def _checked(run: Mapping[str, object], anchor: Mapping[str, object], base: Dict[str, object]) -> Dict[str, object]:
        """The pair's status row: judged on both sides, the headline metric filed, no adapter revision on either side -- else named."""
        if str(run["run_id"]) not in scored or str(anchor["run_id"]) not in scored:
            return {**base, "status": "no_complete_judgment", "metric_id": None}
        ev = evals.get(str(run["eval_id"]))
        if ev is None:
            ev = evals[str(run["eval_id"])] = _eval_row(conn, str(run["eval_id"]))
        filed = _filed_item_metrics(conn, str(run["run_id"]))
        metric = paper.run_headline_metric_id(ev, filed)
        if metric is None or metric not in filed:
            return {**base, "status": "no_items", "metric_id": metric, "filed_item_metric_ids": filed}
        revisions = _adapter_backed(conn, str(run["run_id"]), str(anchor["run_id"]), metric)
        if revisions is not None:
            return {**base, "status": "adapter_backed", "metric_id": metric, "source_revision_ids": revisions}
        return {**base, "status": "due", "metric_id": metric}

    for run in steered:
        key = (str(run["campaign"]), str(run["eval_id"]), str(run["instrument_id"]), str(run["model_id"]))
        base: Dict[str, object] = {"run_id": str(run["run_id"]), "campaign": key[0], "eval_id": key[1], "instrument_id": key[2],
                                   "model_id": key[3], "object_id": run.get("object_id"), "view_legacy": run.get("view_legacy"),
                                   "eval_category": run.get("eval_category")}
        ev0 = evals.get(str(run["eval_id"])) or evals.setdefault(str(run["eval_id"]), _eval_row(conn, str(run["eval_id"])))
        metric0: Optional[str] = paper.run_headline_metric_id(ev0, _filed_item_metrics(conn, str(run["run_id"])))
        # The same-view anchor is picked among CURRENT none runs only, and a same-view row already filed at the run's
        # current judgment's key is KEPT while its anchor is current (never a second row beside it); a retired anchor makes the run DUE again
        sv_keep = contrast_keep(conn, run, str(metric0), currency, kinds=SAME_VIEW_KINDS, check_cap=False, withdrawn=withdrawn) if metric0 else None
        sv_retired: List[Dict[str, object]] = list(sv_keep["retired"]) if sv_keep is not None else []   # type: ignore[arg-type]
        view_cands = [c for c in by_key.get(key, []) if (c.get("view_legacy") or None) == (run.get("view_legacy") or None)]
        picked = anchor_for(run, view_cands, currency=currency)
        # A filed row whose anchor RUN is current and only its anchor JUDGMENT changed is re-judged against THAT anchor
        # (the same key, an in-place update) -- never the fresh latest-queued pick, which moved to a newer none Y and filed a second current
        # key beside the stale X key (v_paper_long then joined both). A fresh pick is for a retired anchor only.
        sv_target = inplace_target(sv_keep, view_cands, currency)   # a withdrawn key whose anchor is a paper row again, too
        if sv_target is not None:
            picked = (sv_target[0], sv_target[1], picked[2] if picked is not None else 1)
        # The borrowed u0 anchor for a steered row of a steered-wave view (its own none excluded by ruling or absent from the view)
        u0_view = u0_view_of(run.get("view_legacy"))
        borrowed: Optional[Dict[str, object]] = None
        if u0_view is not None and u0_view != (run.get("view_legacy") or None):
            if picked is not None:
                same_view_status = f"excluded:{excluded[str(picked[0]['run_id'])]}" if str(picked[0]["run_id"]) in excluded else "present"
            elif view_cands:
                # The view's none runs exist but none is current under the paper's rules -- named, never read as absent
                same_view_status = "retired:" + "; ".join(f"{c['run_id']}: {(currency.get(str(c['run_id'])) or {}).get('why') or 'not carried by the view'}" for c in view_cands)
            else:
                same_view_status = "absent"
            u0_cands = [c for c in by_key.get(key, []) if (c.get("view_legacy") or None) == u0_view]
            bbase = {**base, "anchor_kind": ANCHOR_BORROWED, "same_view_none_status": same_view_status}
            bpick: Optional[Tuple[Dict[str, object], int]] = None
            btier: Optional[Dict[str, object]] = None       # The tier that fired (0 = the filed row's own key, re-emitted in place)
            cap_problem: Optional[str] = None
            tier_problem: Optional[str] = None
            keep: Optional[Dict[str, object]] = None
            deferred = str(run.get("eval_category") or "") in set(borrowed_defer_categories)
            if not deferred:
                # A borrowed row already filed at the run's CURRENT judgment's key is KEPT while its anchor is current
                # under the paper's own rules and the caps agree -- never abandoned for a newer arrival the view still ranks behind it; a current
                # anchor at another cap is refused by name; a retired anchor frees the re-pick, whose pool is the CURRENT candidates only
                keep = borrowed_keep(conn, run, str(metric0), currency, withdrawn=withdrawn) if metric0 else None
                if keep is None or (keep["kept"] is None and not keep["mismatch"]):
                    unpadded = [c for c in u0_cands if c.get("instrument_padding_tokens") is None]
                    b_target = inplace_target(keep, unpadded, currency)
                    if b_target is not None:
                        # The re-judged borrowed anchor is the target (same key, in-place update), never the fresh
                        # earliest-queued pick, which moved to a newer none and filed a second current key beside the stale one
                        bpick = (b_target[0], len([c for c in unpadded if anchor_run_current(currency, str(c["run_id"]))]))
                        btier = {"tier": 0, "name": "own_filed_key", "rule": "the filed row's own anchor, re-emitted on its own key (re-judged, g28-fix-4, or re-admitted, g28-fix-5)"}
                    else:
                        try:
                            picked3 = pick_borrowed_anchor_tiered(run, u0_cands, currency=currency, rostered=key[3] in rostered_models)
                            if picked3 is not None:
                                bpick, btier = (picked3[0], picked3[1]), picked3[2]
                        except NoCapMatchedAnchor as e:
                            cap_problem = str(e)
                        except OffRosterAnchorsOnly as e:
                            tier_problem = str(e)             # A rostered model with off-roster nones of other sources only
            if deferred:
                borrowed = {**bbase, "status": "borrowed_deferred", "anchor_run_id": None, "metric_id": None,
                            "note": "D1245 (4): deferred until the D648 correction op re-registers the static units under their rung-keyed run ids"}
            elif keep is not None and keep["mismatch"]:
                named = "; ".join(f"{m['anchor_run_id']} caps {json.dumps(m['caps'], sort_keys=True)}" for m in keep["mismatch"])   # type: ignore[index]
                borrowed = {**bbase, "status": "borrowed_cap_mismatch", "anchor_run_id": str(keep["mismatch"][0]["anchor_run_id"]), "metric_id": metric0,   # type: ignore[index]
                            "contrast_key": keep["key"], "cap_mismatch": keep["mismatch"],
                            "note": f"a borrowed contrast at this judgment's key is filed against a CURRENT anchor drawn at another cap ({named}): not kept, "
                                    f"nothing is filed while it stands (never a second contrast beside it); retire that anchor by the view's rules -- a "
                                    f"supersede / exclude ruling or its successor's superseded_by_run_id -- and the next pass re-picks a cap-matched none"}
            elif keep is not None and keep["kept"] is not None:
                borrowed = {**bbase, "status": "borrowed_current", "anchor_run_id": str(keep["kept"]), "metric_id": metric0, "contrast_key": keep["key"],
                            "note": "a borrowed contrast at this judgment's key is already in the registry, its anchor run is current under the paper's "
                                    f"rules ({RUN_CURRENCY_VIEW}) and the caps agree: kept; re-chosen only when the view retires that run"}
            elif cap_problem is not None:
                borrowed = {**bbase, "status": "no_cap_matched_u0_anchor", "anchor_run_id": None, "metric_id": None, "note": cap_problem}
            elif tier_problem is not None:
                borrowed = {**bbase, "status": "no_u0_anchor", "anchor_run_id": None, "metric_id": None, "note": tier_problem}
            elif bpick is None:
                unpadded = [c for c in u0_cands if c.get("instrument_padding_tokens") is None]
                retired = [c for c in unpadded if not anchor_run_current(currency, str(c["run_id"]))]
                note = f"no complete unpadded none run of view {u0_view} for this campaign x eval x instrument x model"
                if retired:
                    note += (f"; {len(retired)} present but not current under the paper's rules ({RUN_CURRENCY_VIEW}): "
                             + "; ".join(f"{c['run_id']}: {(currency.get(str(c['run_id'])) or {}).get('why') or 'not carried by the view'}" for c in retired))
                borrowed = {**bbase, "status": "no_u0_anchor", "anchor_run_id": None, "metric_id": None, "note": note}
            else:
                anchor_b, n_b = bpick
                bbase.update(anchor_run_id=str(anchor_b["run_id"]), n_anchor_candidates=n_b, borrowed_tier=btier,
                             borrowed_extra={"contrast_caps": {"steered": run.get("token_cap_tokens"), "anchor": anchor_b.get("token_cap_tokens")},
                                             "anchor_standup": {"run_id": str(anchor_b["run_id"]), "view_legacy": anchor_b.get("view_legacy"),
                                                                "cluster_job_id": anchor_b.get("cluster_job_id"),
                                                                "producer_run_key_legacy": anchor_b.get("producer_run_key_legacy")},
                                             "cross_server": True, "same_view_none_status": same_view_status,
                                             "anchor_currency": {"view": RUN_CURRENCY_VIEW, "selection_state": (currency.get(str(anchor_b["run_id"])) or {}).get("selection_state"),
                                                                 "in_current_paper": (currency.get(str(anchor_b["run_id"])) or {}).get("in_current_paper")},   # Recorded, not required
                                             "anchor_tier": btier,                                                                 # The tier that fired
                                             **({"retired_anchors": keep["retired"]} if keep is not None and keep["retired"] else {}),   # why this row supersedes a filed one
                                             **({"rejudged_anchors": keep["stale_judgment"]} if keep is not None and keep["stale_judgment"] else {}),   # an in-place update
                                             **({"readmit_anchors": keep["readmit"]} if keep is not None and keep["readmit"] else {})})   # a withdrawn key re-admitted
                borrowed = _checked(run, anchor_b, bbase)
        if borrowed is not None and keep is not None:
            if keep["retired"]:
                borrowed.setdefault("retired_anchors", keep["retired"])      # the sweep reads it from every borrowed status
            if keep["stale_judgment"]:
                borrowed.setdefault("rejudged_anchors", keep["stale_judgment"])   # the filed row's anchor was re-judged: re-emitted on the same key
            if keep["adapter"]:
                borrowed.setdefault("adapter_contrasts", keep["adapter"])    # An adapter's rows at this key, reported never re-filed
            if keep["readmit"]:
                borrowed.setdefault("readmit_anchors", keep["readmit"])      # withdrawn keys whose anchor is a paper row again
            if borrowed.get("status") == "due" and int(keep["filed"] or 0) == 0:   # type: ignore[call-overload]
                borrowed["first_emission"] = True                            # no native row at this key yet: the sweep's `first` bucket
            borrowed.setdefault("frame_anchors", keep["frame"])            # the frame's view of every filed row at this key
            stale = _frame_stale(keep)
            if stale and borrowed.get("status") in ("no_u0_anchor", "no_cap_matched_u0_anchor"):
                # No replacement, so nothing is withdrawn -- but the frame still joins these rows: named, never silent
                borrowed["frame_stale"] = stale
                borrowed["note"] = str(borrowed.get("note") or "") + _frame_stale_note(stale)
            elif stale and borrowed.get("status") == "borrowed_current":
                # The kept row is the frame's contrast; the frame-visible keys this emitter retires beside it are
                # withdrawn by the pass (emit_withdrawal), never left joined because another anchor stands
                borrowed["withdraw_stale"] = stale
                borrowed["note"] = str(borrowed.get("note") or "") + _withdraw_stale_note(stale)
        if picked is None:
            if sv_retired:
                # A same-view row WAS filed, its anchor is retired, and no current none of this view remains: named, nothing written
                sv_stale = _frame_stale(sv_keep)   # rows the frame still joins, named
                out.append({**base, "status": "no_anchor", "anchor_run_id": None, "anchor_kind": str(sv_retired[0]["anchor_kind"]), "metric_id": metric0,
                            "retired_anchors": sv_retired, "contrast_key": sv_keep["key"] if sv_keep is not None else None,
                            "frame_anchors": sv_keep["frame"] if sv_keep is not None else {},
                            **({"frame_stale": sv_stale} if sv_stale else {}),
                            "note": "the filed same-view contrast's anchor is no longer current under the paper's rules "
                                    f"({RUN_CURRENCY_VIEW}) and no current none of this view remains: "
                                    + "; ".join(f"{x['anchor_run_id']} ({x['anchor_kind']}): {x['why']}" for x in sv_retired)
                                    + (_frame_stale_note(sv_stale) if sv_stale else "")})
            elif borrowed is None:
                out.append({**base, "status": "no_anchor", "anchor_run_id": None, "anchor_kind": None, "metric_id": None})
            if borrowed is not None:
                out.append(borrowed)                 # the same-view absence rides the borrowed row (same_view_none_status absent / retired)
            continue
        anchor, kind, n_cand = picked
        if sv_keep is not None and sv_keep["kept"] is not None:
            moved = f" (a fresh pick would name {anchor['run_id']} as {kind})" if str(anchor["run_id"]) != str(sv_keep["kept"]) else ""
            sv_stale = _frame_stale(sv_keep)   # frame-visible retired keys beside the kept row are withdrawn by the pass
            out.append({**base, "status": "same_view_current", "anchor_run_id": str(sv_keep["kept"]), "anchor_kind": str(sv_keep["kept_kind"]),
                        "metric_id": metric0, "contrast_key": sv_keep["key"], "n_anchor_candidates": n_cand,
                        **({"retired_anchors": sv_retired} if sv_retired else {}),
                        **({"frame_anchors": sv_keep["frame"]} if sv_keep["frame"] else {}),
                        **({"withdraw_stale": sv_stale} if sv_stale else {}),
                        "note": "a same-view contrast at this judgment's key is already in the registry and its anchor run is current under the "
                                f"paper's rules ({RUN_CURRENCY_VIEW}): kept; never a second row beside it{moved}" + (_withdraw_stale_note(sv_stale) if sv_stale else "")})
        else:
            sv_row = _checked(run, anchor, {**base, "anchor_run_id": str(anchor["run_id"]), "anchor_kind": kind, "n_anchor_candidates": n_cand,
                                            **({"retired_anchors": sv_retired} if sv_retired else {}),
                                            **({"rejudged_anchors": sv_keep["stale_judgment"]} if sv_keep is not None and sv_keep["stale_judgment"] else {}),
                                            **({"readmit_anchors": sv_keep["readmit"]} if sv_keep is not None and sv_keep["readmit"] else {}),
                                            **({"frame_anchors": sv_keep["frame"]} if sv_keep is not None and sv_keep["frame"] else {}),
                                            **({"adapter_contrasts": sv_keep["adapter"]} if sv_keep is not None and sv_keep["adapter"] else {})})
            if sv_row.get("status") == "due" and sv_keep is not None and int(sv_keep["filed"] or 0) == 0:   # type: ignore[call-overload]
                sv_row["first_emission"] = True                              # no native row at this key yet: the sweep's `first` bucket
            out.append(sv_row)
        if borrowed is not None:
            out.append(borrowed)
    return out


# --------------------------------------------------------------------------------------------
# the statistic
# --------------------------------------------------------------------------------------------

def bootstrap_seed(run_id: str, anchor_run_id: str, metric_id: str) -> int:
    """The bootstrap's seed: sha256 over ``run|anchor|metric`` -- a re-run of the same contrast reproduces the same interval."""
    return int(hashlib.sha256(f"{run_id}|{anchor_run_id}|{metric_id}".encode("utf-8")).hexdigest()[:16], 16)


def paired_bootstrap(diffs: Sequence[float], seed: int, B: int = BOOTSTRAP_B, level: float = 0.95) -> Tuple[float, float, float]:
    """``(delta, lo, hi)``: the mean paired difference and the percentile bootstrap interval of the mean over ``B`` resamples of the
    pairs (with replacement), ``random.Random(seed)``. A single pair has a degenerate interval at its difference."""
    n = len(diffs)
    if n == 0:
        raise ValueError("paired_bootstrap: no pairs")
    delta = sum(diffs) / n
    if n == 1:
        return delta, delta, delta
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(int(B)))
    alpha = (1.0 - level) / 2.0

    def quantile(q: float) -> float:      # the percentile bootstrap's order statistic (q in [0, 1] indexes within the sorted means)
        return means[int(round(q * (len(means) - 1)))]

    return delta, quantile(alpha), quantile(1.0 - alpha)


def threshold_for(conn: sqlite3.Connection, run_id: str, judgment_id: str, metric_id: str) -> float:
    """The metric row's threshold: of the run's ``run_metric`` rows under this judgment and metric, the one carrying ``k`` (the
    canonical row of a thresholded family), else the single row's; ``-1.0`` (the sentinel) when none or several unmarked."""
    rows = _rows(conn, "SELECT threshold, k FROM run_metric WHERE run_id = ? AND judgment_id = ? AND metric_id = ? ORDER BY threshold",
                 (run_id, judgment_id, metric_id))
    with_k = [r for r in rows if r.get("k") is not None]
    if len(with_k) == 1:
        return float(with_k[0]["threshold"])
    if len(rows) == 1:
        return float(rows[0]["threshold"])
    return -1.0


# --------------------------------------------------------------------------------------------
# idempotence
# --------------------------------------------------------------------------------------------

def _contrast_key(row: Mapping[str, object]) -> Tuple[str, str, str, str, str, float]:
    return (str(row["run_id"]), str(row["anchor_run_id"]), str(row.get("judgment_id") or ""), str(row.get("anchor_judgment_id") or ""),
            str(row["metric_id"]), float(row.get("threshold", -1.0)))


def in_registry(conn: sqlite3.Connection, key: Tuple[str, str, str, str, str, float]) -> bool:
    run_id, anchor, jid, ajid, metric, thr = key
    return conn.execute("SELECT 1 FROM run_contrast WHERE run_id = ? AND anchor_run_id = ? AND judgment_id = ? AND anchor_judgment_id = ? "
                        "AND metric_id = ? AND threshold = ? LIMIT 1", (run_id, anchor, jid, ajid, metric, thr)).fetchone() is not None


#: The record-type names a transition file may be named for (the first record type it carries decides: a replacement with its
#: withdrawals is a `run_contrast` file, a withdrawal beside a kept row a `withdrawal` file, a re-admission a `source_revision` file); a
#: reader of one record type looks inside every carrier, so a `source_revision` inside a `run_contrast` file is found.
TRANSITION_FILE_TYPES: Tuple[str, ...] = ("run_contrast", "withdrawal", "source_revision")


def _spool_payloads(spool: str, campaign: str, run_id: str, record_type: str):
    """The payloads of every ``record_type`` record in the run's spool directory (written, not yet folded) -- in the files named for the
    type and in every transition carrier (:data:`TRANSITION_FILE_TYPES`); unreadable files skipped."""
    d = os.path.join(spool, campaign, identity.run_dir_name(run_id))
    if not os.path.isdir(d):
        return
    names = {record_type, *TRANSITION_FILE_TYPES}
    for fn in sorted(os.listdir(d)):
        if fn.startswith(".") or not any(f"-{t}.jsonl" in fn for t in names):
            continue
        try:
            for _ln, rec, _sha in R.read_record_file(os.path.join(d, fn), compute_sha=False):
                p = rec.get("payload") if isinstance(rec, Mapping) else None
                if isinstance(p, Mapping) and rec.get("record_type") == record_type:
                    yield p
        except (OSError, ValueError):
            continue


def in_spool(spool: str, campaign: str, key: Tuple[str, str, str, str, str, float]) -> bool:
    """A ``run_contrast`` record for ``key`` already sits in the run's spool directory (written, not yet folded)."""
    return any(_contrast_key(p) == key for p in _spool_payloads(spool, campaign, key[0], "run_contrast"))


def pending_transition(spool: str, campaign: str, run_id: str, source_id: str, conn: Optional[sqlite3.Connection]) -> Optional[Dict[str, object]]:
    """A transition of the key's per-key source still AWAITING A FOLD -- a ``source_revision`` record of
    ``source_id`` on the run's spool whose ``revision_id`` the registry's ``source_revision`` table does not carry yet (the row
    ``observation.current`` derives from: once it is folded the head has moved and the transition is history). Keyed on the fold, never
    on the lifetime presence of a withdrawal or re-admission file: record files are immutable and survive the fold, so a guard reading
    "a withdrawal of this key exists" skipped the SECOND withdrawal of a key (two deltas again) and read a historical re-admission as
    pending for ever. Returns the pending revision's payload (``extra.purpose`` names the transition) or None; None without a connection."""
    if conn is None:
        return None
    pend = [p for p in _spool_payloads(spool, campaign, run_id, "source_revision") if str(p.get("source_id") or "") == source_id]
    if not pend:
        return None
    ids = sorted({str(p.get("revision_id") or "") for p in pend})
    folded = {str(r[0]) for r in conn.execute(f"SELECT revision_id FROM source_revision WHERE revision_id IN ({','.join('?' for _ in ids)})", ids)}
    for p in pend:
        if str(p.get("revision_id") or "") not in folded:
            return dict(p)
    return None


# --------------------------------------------------------------------------------------------
# The frame's retirement of a filed contrast, and its re-admission -- written as revisions of the key's own source
# --------------------------------------------------------------------------------------------

def _later_stamp(after: str, wait_s: float = 2.0) -> str:
    """A wall-clock ``written_utc`` / ``observed_utc`` STRICTLY after ``after`` (the source's head, or the previous stamp of this emission):
    the head of a source is the LATEST ``observed_utc`` (reducer._recompute_current), so a revision meant to move the head must be stamped
    after it. Waits (at most ``wait_s``) rather than stamping ahead of the clock; a head further ahead than that is a clock skew, refused
    by name rather than written into history."""
    now = R.utc_now()
    waited = 0.0
    while now <= after:
        if waited >= wait_s:
            raise RuntimeError(f"the source's head revision is stamped {after}, ahead of this clock ({now}): refusing to write a revision behind it")
        time.sleep(0.001)
        waited += 0.001
        now = R.utc_now()
    return now


def _source_head_utc(conn: Optional[sqlite3.Connection], source_ids: Sequence[str]) -> str:
    """The latest ``observed_utc`` any of ``source_ids`` carries in ``source_revision`` ('' when none, or no connection)."""
    if conn is None or not source_ids:
        return ""
    marks = ",".join("?" for _ in source_ids)
    row = conn.execute(f"SELECT MAX(observed_utc) FROM source_revision WHERE source_id IN ({marks})", tuple(source_ids)).fetchone()
    return str(row[0] or "") if row else ""


def _revision_payload(source_id: str, observed_utc: str, purpose: str, key_json: str, extra: Mapping[str, object]) -> Dict[str, object]:
    """One accepted ``source_revision`` of a per-key contrast source: id = sha16 over source | observed_utc | purpose (distinct per stamp,
    reproducible from the record), no inputs (the revision states this emitter's reading of the anchor's membership, not a file read)."""
    return {"revision_id": identity.sha16(f"{source_id}|{observed_utc}|{purpose}"), "source_id": source_id, "adapter": CONTRAST_SOURCE_ADAPTER,
            "observed_utc": observed_utc, "inputs_json": {}, "accepted": 1, "priority": 0,
            "extra": {"purpose": purpose, "observation_key": key_json, "producer": PRODUCER_SCRIPT, "rule": "g28-fix-5", **dict(extra)}}


#: A transition file holds a handful of records (four per withdrawn key, two per re-admission, one replacement); the harvester
#: never splits a plain file of at most ITEM_ROWS_PER_TRANSACTION records across transactions (harvest.ParsedFile.parts), which is what
#: makes the file fold as a unit -- the writer refuses a transition above this bound rather than write one the fold could tear.
TRANSITION_MAX_RECORDS = 64
assert TRANSITION_MAX_RECORDS <= H.ITEM_ROWS_PER_TRANSACTION
TypedRecord = Tuple[str, Dict[str, object], str]   # (record_type, payload, written_utc)


class TransitionRefused(ValueError):
    """A record of a transition file failed `validate.validate_record` before the write -- the whole file is refused by name."""


class ContrastTransitionWriter(RW.AtomicBatchWriter):
    """The :class:`worklist.AtomicBatchWriter` mechanics for a file of SEVERAL record types -- the
    withdrawal records (two accepted revisions of the key's source, its observation, the ``withdrawal``), the re-admission records (a head
    revision and its observation) and the replacement ``run_contrast``, in dependency order with strictly increasing stamps, as ONE record
    file: the bytes go to a temp name in the run directory, are fsynced and LINKED under the next free sequence number (the exclusive create
    every writer relies on), so the harvester folds the file whole or never sees it, and a writer dying mid-write leaves a `.tmp` nothing
    reads. The fold is a unit too: a plain record file of at most ITEM_ROWS_PER_TRANSACTION records is one ParsedFile in one chunk, and a
    chunk is one BEGIN IMMEDIATE that rolls back whole when its body raises (Ingestor.ingest_chunk / db.transaction); within it the
    run-scale records fold `source_revision` first, then by (written_utc, sha) -- the order written here -- so an observation finds its
    revision, a withdrawal its head. The rollback is SQLite's, not a file-level all-or-nothing --
    the harvester gives each record its own status and commits the file `partial` when one is blocked or refused -- so the unit holds in
    practice only because every dependency is verified folded in the destination before the write (emit_contrast / emit_withdrawal) and every
    record is validated here; records are never reordered and a transition is never split into two files."""

    def write_transition(self, typed: Sequence[TypedRecord], name_type: str) -> Optional[str]:
        """Write ``typed`` -- ``(record_type, payload, written_utc)`` in file order -- as one atomic record file named for ``name_type``
        (:data:`TRANSITION_FILE_TYPES`); the stamps must be strictly increasing. Returns the path (None when disabled)."""
        if self.disabled or not typed:
            return None
        if len(typed) > TRANSITION_MAX_RECORDS:
            raise ValueError(f"a contrast transition of {len(typed)} records exceeds TRANSITION_MAX_RECORDS {TRANSITION_MAX_RECORDS}: the fold could split it")
        stamps = [t[2] for t in typed]
        if stamps != sorted(stamps) or len(set(stamps)) != len(stamps):
            raise ValueError("a contrast transition's stamps must be strictly increasing in file order")
        if name_type not in TRANSITION_FILE_TYPES:
            raise ValueError(f"a transition file is named for one of {TRANSITION_FILE_TYPES}, not {name_type!r}")
        recs = [R.make_record(rt, self._stamped(p), self.producer, identity.producer_record_id(), st) for rt, p, st in typed]
        # Every record is validated BEFORE the file exists -- the harvester gives
        # each record its own status inside the one transaction and commits the file `partial`, so a record that would come back refused
        # or malformed must never be written beside the ones that fold; refused by name, nothing written
        bad = []
        for i, rec in enumerate(recs):
            res = VAL.validate_record(rec, f"<transition>#{i}")
            if isinstance(res, VAL.Malformed):
                bad.append(f"record {i} ({rec.get('record_type')}): {res.field}: {res.reason}")
        if bad:
            raise TransitionRefused("a transition record would be refused by the harvester; nothing written -- " + "; ".join(bad))
        os.makedirs(self.dir, exist_ok=True)
        tmp = os.path.join(self.dir, f".transition-{os.getpid()}-{recs[0]['record_id']}.tmp")
        try:
            R.write_records(tmp, recs, exclusive=True)
            fd = os.open(tmp, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            for _ in range(1000):
                path = os.path.join(self.dir, R.record_file_name(self._next_seq(), name_type))
                try:
                    os.link(tmp, path)
                except FileExistsError:
                    continue
                self.written.append(path)
                if stamps[-1] > self._last_utc:
                    self._last_utc = stamps[-1]
                return path
            raise RuntimeError(f"ContrastTransitionWriter: could not find a free sequence number under {self.dir}")
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def withdrawal_records(entries: Sequence[Mapping[str, object]], *, replaced_by: Mapping[str, object], after: str,
                       conn: Optional[sqlite3.Connection] = None) -> Tuple[List[TypedRecord], List[Dict[str, object]], str]:
    """The records that retire, in the paper frame's own terms, the filed contrasts in ``entries``
    (``retired`` rows of :func:`contrast_keep` merged with their frame view, ``frame_visible``: the view still reads their anchor current,
    this emitter does not). Per key, four records in dependency order, EACH with its own strictly increasing stamp after ``after`` (the later
    of the sources' heads and the emission's previous stamp): an accepted revision of the key's own source (:func:`contrast_source_id`), the
    ``observation`` of the key under it, a second accepted revision (the new head: the fold derives ``observation.current = 0`` for every
    key the head does not carry), and the ``withdrawal`` record from the first to the second (reason :data:`WITHDRAWAL_REASON`; ``extra``
    names the anchor, its kind, its selection_state / in_current_paper, the why and the replacement). ``v_paper_long``'s withdrawn_contrast
    CTE then reads the key as withdrawn and its current_contrast CTE leaves the row out; the ``run_contrast`` row itself is never edited.
    Nothing is written here: the caller puts the records into ONE transition file (:class:`ContrastTransitionWriter`). Returns
    ``(typed records, one summary per key, the last stamp)``."""
    todo: List[Tuple[Mapping[str, object], str, str]] = []
    for x in entries:
        ck = x.get("contrast_key")
        if not isinstance(ck, Mapping):
            continue
        kj = contrast_observation_key(ck)
        todo.append((x, kj, contrast_source_id(kj)))
    if not todo:
        return [], [], after
    # ONE monotone stamp per RECORD, allocated in dependency order key by key -- the observing
    # revision, its observation, the withdrawing revision, the withdrawal; each source revision's observed_utc is its own record's stamp.
    # A shared group stamp made two keys' records collide on the writer's strictly-increasing rule and abort the whole pass.
    typed: List[TypedRecord] = []
    last = max(after, _source_head_utc(conn, [sid for _x, _kj, sid in todo]))
    out: List[Dict[str, object]] = []
    wds: List[Dict[str, object]] = []
    for x, kj, sid in todo:
        t_obs_rev = _later_stamp(last)
        rv_a = _revision_payload(sid, t_obs_rev, "observe", kj, {"anchor_run_id": x["anchor_run_id"], "anchor_kind": x["anchor_kind"]})
        typed.append(("source_revision", rv_a, t_obs_rev))
        t_obs = _later_stamp(t_obs_rev)
        typed.append(("observation", {"source_id": sid, "revision_id": rv_a["revision_id"], "table_name": "run_contrast", "observation_keys": [kj],
                                      "extra": {"purpose": "observe", "producer": PRODUCER_SCRIPT, "rule": "g28-fix-5"}}, t_obs))
        t_wd_rev = _later_stamp(t_obs)
        rv_b = _revision_payload(sid, t_wd_rev, "withdraw", kj, {"anchor_run_id": x["anchor_run_id"], "anchor_kind": x["anchor_kind"], "why": x.get("why")})
        typed.append(("source_revision", rv_b, t_wd_rev))
        t_wd = _later_stamp(t_wd_rev)
        last = t_wd
        why = str(x.get("why") or "")
        rec = {"anchor_run_id": str(x["anchor_run_id"]), "anchor_kind": str(x["anchor_kind"]), "why": why, "selection_state": x.get("selection_state"),
               "in_current_paper": x.get("in_current_paper"), "source_id": sid, "observation_key": kj, "observing_revision_id": rv_a["revision_id"],
               "withdrawing_revision_id": rv_b["revision_id"], "replaced_by": dict(replaced_by), "reason": WITHDRAWAL_REASON}
        wds.append({"source_id": sid, "observation_key": kj, "from_revision_id": rv_a["revision_id"], "to_revision_id": rv_b["revision_id"],
                    "table_name": "run_contrast", "reason": WITHDRAWAL_REASON,
                    "extra": {"why": why, "anchor_run_id": str(x["anchor_run_id"]), "anchor_kind": str(x["anchor_kind"]),
                              "selection_state": x.get("selection_state"), "in_current_paper": x.get("in_current_paper"),
                              "replaced_by": dict(replaced_by), "producer": PRODUCER_SCRIPT, "rule": "g28-fix-5",
                              "note": "this emitter retires the anchor (a pooled reading's member run, member_pass / unpooled_member: never an anchor) while "
                                      f"{RUN_CURRENCY_VIEW} still reads it current: retired in the frame's own terms before the replacement is filed"}})
        out.append(rec)
        typed.append(("withdrawal", wds[-1], t_wd))
    return typed, out, last


def readmission_records(key: Mapping[str, object], *, why: str, after: str,
                        conn: Optional[sqlite3.Connection] = None) -> Tuple[List[TypedRecord], Dict[str, object], str]:
    """The records that re-admit a contrast key the frame withdrew -- a new accepted head revision of the key's
    source observing the key again (``observation.current`` 1 under the head -> MAX(current) 1 -> the frame joins the row again); the
    earlier withdrawal stands as history. Two records (revision, then observation), stamped after ``after`` and the source's head; nothing is
    written here (the caller's one transition file). Returns ``(typed records, a summary, the last stamp)``."""
    kj = contrast_observation_key(key)
    sid = contrast_source_id(kj)
    t_rev = _later_stamp(max(after, _source_head_utc(conn, [sid])))
    rv = _revision_payload(sid, t_rev, "readmit", kj, {"anchor_run_id": str(key["anchor_run_id"]), "why": why})
    t_obs = _later_stamp(t_rev)
    typed: List[TypedRecord] = [("source_revision", rv, t_rev),
                                ("observation", {"source_id": sid, "revision_id": rv["revision_id"], "table_name": "run_contrast", "observation_keys": [kj],
                                                 "extra": {"purpose": "readmit", "producer": PRODUCER_SCRIPT, "rule": "g28-fix-5"}}, t_obs)]
    return typed, {"source_id": sid, "observation_key": kj, "readmitting_revision_id": rv["revision_id"], "why": why}, t_obs


class DestinationRefused(RuntimeError):
    """The spool's destination registry is not the queried one and no explicit attestation names it,
    or the spool is no registry's records directory -- the pass is refused by name before any write (``main`` exits 2)."""


def records_tree(root: str) -> str:
    """The REAL records tree of a registry root: ``realpath(<root>)/records`` -- the root resolved through every symlink, the
    ``records`` component appended unresolved, so a ``records`` directory that is itself a symlink elsewhere is NOT this registry's tree."""
    return os.path.join(os.path.realpath(os.path.abspath(os.path.expanduser(str(root)))), os.path.basename(H.records_dir("")))


def _under(tree: str, real_path: str) -> bool:
    """``real_path`` lies in ``tree`` by PATH COMPONENTS (``os.path.commonpath``), never by string prefix: `<root>/records2` is outside
    `<root>/records`."""
    try:
        return os.path.commonpath([tree, real_path]) == tree
    except ValueError:                                   # different drives / one relative: never contained
        return False


def spool_destination_root(spool: str) -> Optional[str]:
    """DIAGNOSTIC (never an acceptance rule): the registry root whose REAL records tree contains the spool's REAL path -- the
    nearest ancestor of ``realpath(spool)`` named ``records`` whose parent holds a database; None when there is none. Named in a refusal
    so an operator sees which registry a symlinked or misdirected spool really writes into."""
    p = os.path.realpath(os.path.abspath(os.path.expanduser(str(spool))))
    rec = os.path.basename(H.records_dir(""))
    while True:
        if os.path.basename(p) == rec and os.path.exists(DB.db_path(os.path.dirname(p))):
            return os.path.dirname(p)
        parent = os.path.dirname(p)
        if parent == p:
            return None
        p = parent


def resolve_destination(root: str, spool: str, attested: Optional[str] = None) -> Tuple[str, bool]:
    """``(destination root, same as the queried root)`` for an emission that reads ``root`` and writes ``spool``. Decided by
    CONTAINMENT of real paths -- the spool resolved with ``os.path.realpath`` (every component, symlinks
    included) must lie under :func:`records_tree` of the destination root, compared by path components -- never by lexical ancestry. The
    destination root is ``attested`` (the operator's explicit ``--destination-registry-root``, which must hold a database) else ``root``;
    the attestation names WHICH registry's folded rows are checked and harvest the file, it never waives containment, and a spool under no
    records tree is refused outright (no branch accepts an attested root for it). Refusals name the spool as given, its real path, the
    destination's real records tree and, when the real path lies in another registry's tree, that registry (:func:`spool_destination_root`),
    so a symlink ``A/records -> B/...`` is visible: writing into B while checking A's rows is the hazard this check guards against."""
    root_abs = os.path.abspath(os.path.expanduser(str(root)))
    root_r = os.path.realpath(root_abs)
    spool_abs = os.path.abspath(os.path.expanduser(str(spool)))
    spool_r = os.path.realpath(spool_abs)
    via = f" (real path {spool_r}{'; a symlink is in the way' if spool_r != spool_abs else ''})"
    owner = spool_destination_root(spool_abs)
    owned = f"; its real path lies in the records tree of the registry at {owner}" if owner is not None else "; its real path lies in no registry's records tree"
    if attested:
        att = os.path.abspath(os.path.expanduser(str(attested)))
        if not os.path.exists(DB.db_path(att)):
            raise DestinationRefused(f"--destination-registry-root {att} holds no registry database ({DB.db_path(att)})")
        tree = records_tree(att)
        if not _under(tree, spool_r):
            raise DestinationRefused(f"--destination-registry-root {att} is not the spool's registry: the spool {spool_abs}{via} lies outside its "
                                     f"records tree {tree}{owned}; the attestation names which registry, it does not waive containment (g28-fix-8)")
        return att, os.path.realpath(att) == root_r
    tree = records_tree(root_abs)
    if _under(tree, spool_r):
        return root_abs, True
    if owner is not None:
        raise DestinationRefused(f"the spool {spool_abs}{via} folds into the registry at {owner}, not the queried registry {root_abs} (records tree "
                                 f"{tree}): refused without an explicit --destination-registry-root naming it (g28-fix-7/8)")
    raise DestinationRefused(f"the spool {spool_abs}{via} is not a registry's records directory nor beneath one, and lies outside the records tree "
                             f"{tree} of the queried registry {root_abs}: the registry that harvests it is unknown (g28-fix-8)")


def destination_missing(dest: sqlite3.Connection, *, run_ids: Sequence[str] = (), judgment_ids: Sequence[str] = (),
                        contrast_keys: Sequence[Mapping[str, object]] = (), computation_ids: Sequence[str] = ()) -> List[str]:
    """The parents a transition's records need that the DESTINATION registry has not folded: the ``run`` rows of ``run_ids``
    (the steered run and the anchor: `run_contrast.run_id` / `anchor_run_id` reference them), the ``judgment`` rows of the non-empty
    ``judgment_ids`` (the row's judgment and anchor judgment: the sentinel-column FK triggers), the ``computation`` rows of the non-empty
    ``computation_ids`` (the replacement binds to it; the native blank sentinel binds to none) and the ``run_contrast`` rows of
    ``contrast_keys`` (what a withdrawal or re-admission names). The registry's contract: read from the destination's
    FOLDED rows, never the spool. Each missing parent as ``table:id``; empty when all are folded."""
    out: List[str] = []
    for rid in run_ids:
        if rid and dest.execute("SELECT 1 FROM run WHERE run_id = ? LIMIT 1", (str(rid),)).fetchone() is None:
            out.append(f"run:{rid}")
    for jid in judgment_ids:
        if jid and dest.execute("SELECT 1 FROM judgment WHERE judgment_id = ? LIMIT 1", (str(jid),)).fetchone() is None:
            out.append(f"judgment:{jid}")
    for cid in computation_ids:   # the computation the replacement binds to (the blank sentinel binds to none: the sentinel-column FK)
        if cid and dest.execute("SELECT 1 FROM computation WHERE computation_id = ? LIMIT 1", (str(cid),)).fetchone() is None:
            out.append(f"computation:{cid}")
    for ck in contrast_keys:
        t = contrast_key_tuple(ck)
        if dest.execute("SELECT 1 FROM run_contrast WHERE run_id = ? AND anchor_run_id = ? AND judgment_id = ? AND metric_id = ? AND threshold = ? "
                        "AND computation_id = ? LIMIT 1", (t[0], t[1], t[2], t[3], t[4], t[5])).fetchone() is None:
            out.append(f"run_contrast:{t[0]}|{t[1]}|{t[2]}|{t[3]}|{t[4]}")
    return out


def _destination(root: str, spool: str, conn: sqlite3.Connection, dest: Optional[sqlite3.Connection], destination_root: Optional[str]
                 ) -> Tuple[sqlite3.Connection, bool]:
    """The destination connection for one emission: ``dest`` when the pass resolved it, else resolved here from ``spool`` /
    ``destination_root`` -- ``conn`` itself when the destination is the queried registry, else a read-only connection this call owns
    (``(connection, owned)``). Raises :class:`DestinationRefused`."""
    if dest is not None:
        return dest, False
    dest_root, same = resolve_destination(root, spool, destination_root)
    if same:
        return conn, False
    d = DB.connect(DB.db_path(dest_root), readonly=True)
    d.execute("PRAGMA query_only = 1")
    return d, True


def _pending_for(spool: str, campaign: str, run_id: str, entries: Sequence[Mapping[str, object]], conn: Optional[sqlite3.Connection]) -> List[Dict[str, object]]:
    """The ``entries`` (merged retired rows with their ``contrast_key``) whose per-key source has a transition awaiting a fold."""
    out: List[Dict[str, object]] = []
    for x in entries:
        ck = x.get("contrast_key")
        if not isinstance(ck, Mapping):
            continue
        sid = contrast_source_id(contrast_observation_key(ck))
        p = pending_transition(spool, campaign, run_id, sid, conn)
        if p is not None:
            out.append({"anchor_run_id": str(x.get("anchor_run_id")), "source_id": sid, "pending_revision_id": p.get("revision_id"),
                        "pending_purpose": ((p.get("extra") or {}) if isinstance(p.get("extra"), Mapping) else {}).get("purpose")})
    return out


def emit_withdrawal(root: str, spool: str, due: Mapping[str, object], *, conn: Optional[sqlite3.Connection] = None, dry_run: bool = False,
                    withdrawn: Optional[Set[ContrastKey]] = None, dest: Optional[sqlite3.Connection] = None,
                    destination_root: Optional[str] = None) -> Tuple[str, Optional[Dict[str, object]]]:
    """Withdraw the frame-visible keys this emitter retires for a run whose current row is KEPT
    (``due['withdraw_stale']`` on a `same_view_current` / `borrowed_current` row) -- no replacement is written, the kept row IS the frame's
    one contrast. ONE transition file of the withdrawal records (:func:`withdrawal_records`), named `withdrawal`. Returns ``(status, row)``:
    ``withdrawn`` (``row['extra']['withdrawn_contrasts']`` names the keys), ``dry_run`` (the same, nothing written), ``withdraw_pending``
    (a transition of one of the keys' sources awaits a fold: nothing written, the next pass decides from the fold), ``emission_disabled``,
    and ``destination_refused`` / ``destination_missing_parents`` -- the spool's destination registry (``dest`` /
    ``destination_root``, see :func:`resolve_destination`) must be attested and must already carry the run_contrast rows being withdrawn;
    the pending state and the source heads are read from it."""
    run_id = str(due["run_id"])
    stale = [x for x in (due.get("withdraw_stale") or []) if isinstance(x.get("contrast_key"), Mapping)]
    replaced_by = {"anchor_run_id": due.get("anchor_run_id"), "anchor_kind": due.get("anchor_kind"), "kept": True}
    row: Dict[str, object] = {"run_id": run_id, "anchor_run_id": due.get("anchor_run_id"), "metric_id": due.get("metric_id"), "extra": {}}
    if not stale:
        return "withdrawn", {**row, "extra": {"withdrawn_contrasts": []}}
    own = conn is None
    if own:
        conn = DB.connect(DB.db_path(root), readonly=True)
    dest_own = False
    try:
        try:
            dest, dest_own = _destination(root, spool, conn, dest, destination_root)
        except DestinationRefused as exc:
            return "destination_refused", {**row, "reason": str(exc)}
        pending = _pending_for(spool, str(due["campaign"]), run_id, stale, dest)
        if pending:
            return "withdraw_pending", {**row, "pending": pending}
        missing = destination_missing(dest, contrast_keys=[x["contrast_key"] for x in stale])
        if missing:
            return "destination_missing_parents", {**row, "missing": missing,
                                                  "reason": "the destination registry has not folded the rows this withdrawal names; nothing written"}
        if dry_run:
            row["extra"] = {"withdrawn_contrasts": [{"anchor_run_id": str(x["anchor_run_id"]), "anchor_kind": str(x["anchor_kind"]), "why": str(x.get("why") or ""),
                                                    "source_id": contrast_source_id(contrast_observation_key(x["contrast_key"])), "replaced_by": replaced_by, "dry_run": True}
                                                   for x in stale]}
            return "dry_run", row
        w = ContrastTransitionWriter(spool, str(due["campaign"]), run_id, emit.producer_block(PRODUCER_SCRIPT))
        if w.disabled:
            return "emission_disabled", row
        typed, summary, _last = withdrawal_records(stale, replaced_by=replaced_by, after=R.utc_now(), conn=dest)
        path = w.write_transition(typed, "withdrawal")
        w.push(final=True)
        row["extra"] = {"withdrawn_contrasts": summary, "transition_file": path}
        return "withdrawn", row
    finally:
        if dest_own and dest is not None:
            dest.close()
        if own:
            conn.close()


# --------------------------------------------------------------------------------------------
# one contrast
# --------------------------------------------------------------------------------------------

def emit_contrast(root: str, spool: str, due: Mapping[str, object], *, conn: Optional[sqlite3.Connection] = None,
                  computed_by: str = DEFAULT_COMPUTED_BY, dry_run: bool = False, B: int = BOOTSTRAP_B,
                  withdrawn: Optional[Set[ContrastKey]] = None, dest: Optional[sqlite3.Connection] = None,
                  destination_root: Optional[str] = None) -> Tuple[str, Optional[Dict[str, object]]]:
    """Compute and write ONE ``run_contrast`` record for a ``due`` triple from :func:`contrasts_due`. Returns ``(status, row)``:
    ``written`` / ``dry_run`` (the row, not written) / ``adapter_backed`` (either side's current items ride an adapter source
    revision, named per side in ``row['source_revision_ids']`` -- the adapter owns the pair) / ``exists_registry`` /
    ``exists_spool`` / ``no_pairs`` / ``refused`` (an ambiguous judgment pair, named in ``row['reason']``) / ``emission_disabled``.

    The filed rows this emission supersedes that the paper frame STILL joins (``due['retired_anchors']``
    entries with ``frame_visible``: the view reads their anchor current, this emitter does not) are withdrawn through the same writer
    BEFORE the new row (:func:`withdrawal_records`; ``row['extra']['withdrawn_contrasts']`` names them; a dry run plans them under the same
    key and writes nothing). A due row whose key already stands in the registry but is withdrawn in the frame (its anchor a paper
    baseline row again) is RE-ADMITTED instead of read as ``exists_registry``: :func:`readmit_contrast`, status ``readmitted`` (the row
    carries ``dry_run`` True when nothing was written), or ``readmit_pending`` when a transition of the key's source awaits a fold. A new row
    written on a key the frame withdrew (a re-judge of a withdrawn key) is re-admitted in the same emission. ``withdrawn`` is
    :func:`frame_withdrawn_keys` (evaluated here when not handed in).

    The withdrawals, the re-admission and the replacement row are ONE record file (:class:`ContrastTransitionWriter`, folded as a
    unit: the frame sees the old delta or the new one, never none and never both); the pending guards read :func:`pending_transition` (a
    revision of the key's source on the spool the registry has not folded) -- ``transition_pending`` when a key to withdraw has one.

    The spool's DESTINATION registry (``dest`` / ``destination_root``; :func:`resolve_destination`) is where the
    file folds, so the pending state and the source heads are read from it, and before ANY write every parent the records need must
    already be folded there (:func:`destination_missing`: the steered run and its judgment, the anchor run and the anchor judgment, the
    rows a withdrawal or re-admission names) -- else ``destination_missing_parents``, nothing written, the old delta stays joined; a
    destination that is not the queried root and not attested is ``destination_refused``."""
    run_id, anchor_id, metric = str(due["run_id"]), str(due["anchor_run_id"]), str(due["metric_id"])
    own = conn is None
    if own:
        conn = DB.connect(DB.db_path(root), readonly=True)
    dest_own = False
    try:
        try:
            dest, dest_own = _destination(root, spool, conn, dest, destination_root)
        except DestinationRefused as exc:
            return "destination_refused", {"run_id": run_id, "anchor_run_id": anchor_id, "metric_id": metric, "reason": str(exc)}
        # scope FIRST: a side whose current items ride an adapter source revision is that
        # adapter's to contrast -- refused here as well as in `contrasts_due`, so a hand-built `due` cannot write one. The record key
        # below carries no revision id, so a later accepted revision changing the items under the same judgment id would otherwise
        # read `exists_registry` forever.
        (jids, revs), (ajids, arevs) = _current_provenance(conn, run_id, metric), _current_provenance(conn, anchor_id, metric)
        if revs or arevs:
            return "adapter_backed", {"run_id": run_id, "anchor_run_id": anchor_id, "metric_id": metric,
                                      "source_revision_ids": {"run": revs, "anchor": arevs},
                                      "reason": "current items ride an adapter source revision; the adapter owns this pair (D6 / D58)"}
        # then idempotence (the daemon revisits every complete steered run each tick): when each side has ONE current judgment for
        # the metric -- the same resolution `registry.paired` makes -- the pair names the record key, and a contrast already folded or
        # already in the spool returns before the pairs are read and the bootstrap run. Any other case falls through to `paired`.
        wd = frame_withdrawn_keys(conn) if withdrawn is None else withdrawn   # the frame's observation rule
        if len(jids) == 1 and len(ajids) == 1:
            thr0 = threshold_for(conn, run_id, jids[0], metric)
            key = (run_id, anchor_id, jids[0], ajids[0], metric, thr0)
            known = {"run_id": run_id, "anchor_run_id": anchor_id, "metric_id": metric, "judgment_id": jids[0], "anchor_judgment_id": ajids[0]}
            ckey = {"run_id": run_id, "anchor_run_id": anchor_id, "judgment_id": jids[0], "metric_id": metric, "threshold": float(thr0),
                    "computation_id": NATIVE_COMPUTATION_ID}
            if in_registry(conn, key):
                if contrast_key_tuple(ckey) in wd:
                    # The row stands in the table but the frame withdrew its key (an earlier emission retired it when its anchor
                    # lost paper membership); the anchor is a paper baseline row again (a due row's anchor is current and eligible) -> the
                    # key is RE-ADMITTED: a new head revision of its source observes it; nothing else is written, the row's values stand
                    why = (f"the anchor is an eligible anchor again (anchor_kind {due.get('anchor_kind')}; "
                           + "; ".join(str(x.get('why') or '') for x in (due.get('readmit_anchors') or []) if str(x.get('anchor_run_id')) == anchor_id) + ")")
                    pend = pending_transition(spool, str(due["campaign"]), run_id, contrast_source_id(contrast_observation_key(ckey)), dest)
                    if pend is not None:
                        return "readmit_pending", {**known, "contrast_key": ckey, "pending_revision_id": pend.get("revision_id")}
                    missing = destination_missing(dest, run_ids=[run_id, anchor_id], judgment_ids=[jids[0], ajids[0]], contrast_keys=[ckey])
                    if missing:   # the destination must carry the row being re-admitted and its parents
                        return "destination_missing_parents", {**known, "contrast_key": ckey, "missing": missing,
                                                              "reason": "the destination registry has not folded the row this re-admission names; nothing written"}
                    if dry_run:
                        return "readmitted", {**known, "contrast_key": ckey, "dry_run": True,
                                              "readmitted": {"source_id": contrast_source_id(contrast_observation_key(ckey)), "why": why}}
                    w = ContrastTransitionWriter(spool, str(due["campaign"]), run_id, emit.producer_block(PRODUCER_SCRIPT))
                    if w.disabled:
                        return "emission_disabled", {**known, "contrast_key": ckey}
                    typed, done, _last = readmission_records(ckey, why=why, after=R.utc_now(), conn=dest)
                    done["transition_file"] = w.write_transition(typed, "source_revision")   # one file, folded as a unit
                    w.push(final=True)
                    return "readmitted", {**known, "contrast_key": ckey, "dry_run": False, "readmitted": done}
                return "exists_registry", known
            if in_spool(spool, str(due["campaign"]), key):
                return "exists_spool", known
        try:
            # the pair's judgments are NAMED when each side resolves to one complete judgment: a re-judged side's superseded
            # judgment still has current item rows, and `paired` left to resolve the side itself would refuse it as judged twice
            pairs = registry.paired(run_id, anchor_id, metric, root=root, judgment_id=jids[0] if len(jids) == 1 else None,
                                    anchor_judgment_id=ajids[0] if len(ajids) == 1 else None)
        except ValueError as exc:
            return "refused", {"run_id": run_id, "anchor_run_id": anchor_id, "metric_id": metric, "reason": str(exc)}
        n_all = int(len(pairs))
        both = pairs.dropna(subset=["value", "anchor_value"]) if n_all else pairs
        n_pairs = int(len(both))
        if n_pairs == 0:
            return "no_pairs", {"run_id": run_id, "anchor_run_id": anchor_id, "metric_id": metric, "n_pairs": 0}
        jids = sorted(set(str(j) for j in both["judgment_id"].tolist()))
        ajids = sorted(set(str(j) for j in both["anchor_judgment_id"].tolist()))
        if len(jids) != 1 or len(ajids) != 1:
            return "refused", {"run_id": run_id, "anchor_run_id": anchor_id, "metric_id": metric,
                               "reason": f"pairs span judgments {jids} x {ajids}; one per side is required"}
        judgment_id, anchor_judgment_id = jids[0], ajids[0]
        threshold = threshold_for(conn, run_id, judgment_id, metric)
        diffs = [float(v) - float(a) for v, a in zip(both["value"].tolist(), both["anchor_value"].tolist())]
        seed = bootstrap_seed(run_id, anchor_id, metric)
        delta, lo, hi = paired_bootstrap(diffs, seed, B=B)
        row: Dict[str, object] = {
            "run_id": run_id, "anchor_run_id": anchor_id, "judgment_id": judgment_id, "metric_id": metric, "threshold": float(threshold),
            "computation_id": "", "anchor_judgment_id": anchor_judgment_id, "anchor_kind": str(due["anchor_kind"]),
            "delta": float(delta), "delta_lo": float(lo), "delta_hi": float(hi), "delta_p": None, "delta_method": DELTA_METHOD,
            "effect_size": None, "effect_kind": None, "n_pairs": n_pairs, "computed_by": computed_by, "computed_utc": R.utc_now(),
            "extra": {"bootstrap_B": int(B), "bootstrap_seed": seed, "bootstrap_level": 0.95, "n_pairs_aligned": n_all,
                      "n_pairs_dropped_null": n_all - n_pairs, "anchor_selection": {"kind": str(due["anchor_kind"]),
                                                                                    "n_candidates": due.get("n_anchor_candidates"),
                                                                                    **({"tier": due["borrowed_tier"]} if due.get("borrowed_tier") else {})},
                      "eval_id": due.get("eval_id"), "model_id": due.get("model_id"), "producer": PRODUCER_SCRIPT,
                      **(dict(due.get("borrowed_extra") or {})),          # contrast_caps / anchor_standup / cross_server / same_view_none_status
                      **({"retired_anchors": list(due["retired_anchors"])} if due.get("retired_anchors") else {}),   # why this row supersedes a filed one
                      **({"rejudged_anchors": list(due["rejudged_anchors"])} if due.get("rejudged_anchors") else {})},   # the same key, the anchor's new judgment
        }
        # the filed rows this emission supersedes that the frame STILL joins -- withdrawn before the new row is written; and the
        # new row's own key, when the frame withdrew it earlier (a re-judge on a withdrawn key), re-admitted in the same emission
        frame_anchors = due.get("frame_anchors") if isinstance(due.get("frame_anchors"), Mapping) else {}
        to_withdraw = [{**x, **frame_anchors[str(x["anchor_run_id"])]} for x in (due.get("retired_anchors") or [])   # type: ignore[index]
                       if (frame_anchors.get(str(x["anchor_run_id"])) or {}).get("frame_visible")]
        own_key = {"run_id": run_id, "anchor_run_id": anchor_id, "judgment_id": judgment_id, "metric_id": metric, "threshold": float(threshold),
                   "computation_id": NATIVE_COMPUTATION_ID}
        own_withdrawn = contrast_key_tuple(own_key) in wd
        replaced_by = {"anchor_run_id": anchor_id, "anchor_kind": str(due["anchor_kind"]), "judgment_id": judgment_id, "anchor_judgment_id": anchor_judgment_id}
        # a key to withdraw whose source has a transition awaiting a fold defers the WHOLE emission -- the fold decides first
        pending = _pending_for(spool, str(due["campaign"]), run_id, to_withdraw, dest) if to_withdraw else []
        if pending:
            return "transition_pending", {"run_id": run_id, "anchor_run_id": anchor_id, "metric_id": metric, "pending": pending,
                                          "reason": "a transition of a key to withdraw awaits a fold; the replacement is not written until the fold"}
        # every parent the file's records need must be folded in the DESTINATION before anything is written -- the harvester commits a
        # file whose replacement is dependency-blocked beside its folded withdrawal (a `partial` delivery), which would leave the frame
        # with no contrast; refused by name instead, the old delta stays joined
        missing = destination_missing(dest, run_ids=[run_id, anchor_id], judgment_ids=[judgment_id, anchor_judgment_id],
                                      contrast_keys=[x["contrast_key"] for x in to_withdraw] + ([own_key] if own_withdrawn else []),
                                      computation_ids=[str(row["computation_id"])])
        if missing:
            return "destination_missing_parents", {"run_id": run_id, "anchor_run_id": anchor_id, "metric_id": metric, "missing": missing,
                                                  "reason": "the destination registry has not folded a parent the transition's records need; nothing written"}
        if dry_run:
            row["extra"]["withdrawn_contrasts"] = [{"anchor_run_id": str(x["anchor_run_id"]), "anchor_kind": str(x["anchor_kind"]), "why": str(x.get("why") or ""),   # type: ignore[index]
                                                   "source_id": contrast_source_id(contrast_observation_key(x["contrast_key"])), "replaced_by": replaced_by, "dry_run": True}
                                                  for x in to_withdraw]
            if own_withdrawn:
                row["extra"]["readmitted"] = {"source_id": contrast_source_id(contrast_observation_key(own_key)), "dry_run": True}   # type: ignore[index]
            return "dry_run", row
        w = ContrastTransitionWriter(spool, str(due["campaign"]), run_id, emit.producer_block(PRODUCER_SCRIPT))
        if w.disabled:
            return "emission_disabled", row
        # ONE transition file: the withdrawals, then the re-admission of this row's own key, then the row -- dependency order,
        # strictly increasing stamps; the harvester folds it as a unit (ContrastTransitionWriter)
        typed: List[TypedRecord] = []
        last = R.utc_now()
        if to_withdraw:
            wd_typed, summary, last = withdrawal_records(to_withdraw, replaced_by=replaced_by, after=last, conn=dest)
            typed += wd_typed
            row["extra"]["withdrawn_contrasts"] = summary   # type: ignore[index]
        if own_withdrawn:
            ra_typed, done, last = readmission_records(own_key, why="re-emitted on a key the frame had withdrawn; the anchor is an eligible anchor again", after=last, conn=dest)
            typed += ra_typed
            row["extra"]["readmitted"] = done   # type: ignore[index]
        typed.append(("run_contrast", dict(row), _later_stamp(last)))
        row["extra"]["transition_file"] = w.write_transition(typed, "run_contrast")   # type: ignore[index]
        w.push(final=True)
        return "written", row
    finally:
        if dest_own and dest is not None:
            dest.close()
        if own:
            conn.close()


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------

def _line(status: str, due: Mapping[str, object], row: Optional[Mapping[str, object]]) -> str:
    parts = [f"[contrasts] {status}", f"run={due.get('run_id')}", f"anchor={due.get('anchor_run_id')}", f"kind={due.get('anchor_kind')}",
             f"metric={due.get('metric_id')}"]
    if row:
        if row.get("n_pairs") is not None:
            parts.append(f"n_pairs={row['n_pairs']}")
        if row.get("delta") is not None:
            parts.append(f"delta={row['delta']:.6f} [{row['delta_lo']:.6f}, {row['delta_hi']:.6f}]")
        if row.get("reason"):
            parts.append(f"reason={row['reason']}")
        extra = row.get("extra") if isinstance(row.get("extra"), Mapping) else {}
        if extra.get("withdrawn_contrasts"):   # the filed rows this emission retired in the frame's own terms
            parts.append("withdrawn=" + ",".join(f"{x.get('anchor_run_id')}({x.get('anchor_kind')})" for x in extra["withdrawn_contrasts"]))   # type: ignore[union-attr]
        if extra.get("readmitted") or row.get("readmitted"):
            parts.append("readmitted=" + str((extra.get("readmitted") or row.get("readmitted") or {}).get("source_id")))   # type: ignore[union-attr]
    if isinstance(due.get("borrowed_tier"), Mapping):   # Which tier picked the borrowed anchor
        parts.append(f"tier={due['borrowed_tier'].get('tier')}:{due['borrowed_tier'].get('name')}")
    if row and row.get("pending"):   # a transition awaiting a fold
        parts.append("pending=" + ",".join(f"{p.get('anchor_run_id')}:{p.get('pending_purpose')}" for p in row["pending"]))   # type: ignore[union-attr]
    if row and row.get("missing"):   # the destination has not folded these parents
        parts.append("missing=" + ",".join(str(m) for m in row["missing"]))   # type: ignore[union-attr]
    if due.get("filed_item_metric_ids") is not None:
        parts.append(f"filed_item_metric_ids={due['filed_item_metric_ids']}")
    if due.get("same_view_none_status") is not None:
        parts.append(f"same_view_none={due['same_view_none_status']}")
    if due.get("note"):
        parts.append(f"note={due['note']}")
    revisions = (row or {}).get("source_revision_ids", due.get("source_revision_ids"))
    if revisions is not None:
        parts.append(f"source_revision_ids={json.dumps(revisions, sort_keys=True)}")
    return " ".join(parts)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--registry-root", default=None, help="the registry root (default $REGISTRY_ROOT, else ~/registry)")
    ap.add_argument("--model-id", action="append", default=None, help="restrict to runs of this model_id (repeatable)")
    ap.add_argument("--campaign", default=None, help="restrict to one campaign")
    ap.add_argument("--eval-id", action="append", default=None, help="restrict to this eval (repeatable; default every eval)")
    ap.add_argument("--spool", default=None, help="the records spool to write into (default emit.spool_root(), else <root>/records)")
    ap.add_argument("--destination-registry-root", default=None,
                    help="g28-fix-7: the registry that HARVESTS --spool when it is not --registry-root -- an explicit attestation, never "
                         "inferred; without it a spool folding into another registry (or into none) is refused by name before any write. "
                         "The spool's REAL path must lie under that registry's real records tree (g28-fix-8): the flag names which registry, "
                         "it never waives containment. Every transition's parents are verified folded in that destination before the file is written")
    ap.add_argument("--computed-by", default=DEFAULT_COMPUTED_BY, choices=list(V.COMPUTED_BY), help="run_contrast.computed_by")
    ap.add_argument("--bootstrap-b", type=int, default=BOOTSTRAP_B, help=f"bootstrap resamples (default {BOOTSTRAP_B})")
    ap.add_argument("--dry-run", action="store_true", help="compute and print every contrast; write nothing")
    ap.add_argument("--borrowed-defer-category", action="append", default=None,
                    help=f"eval categories whose BORROWED contrasts are deferred (default {list(BORROWED_DEFER_CATEGORIES)}, D1245 (4): the static "
                         "units await the D648 re-registration); `none` lifts the deferral")
    ap.add_argument("--once", action="store_true", help="one pass over the selection (the only mode; the daemon passes it explicitly)")
    ap.add_argument("--sweep-noncurrent", action="store_true",
                    help="g28-fix-3 (D1343): the BOUNDED sweep instead of the pass -- every native run of the registry (every model and view, "
                         "or --models) whose filed contrast's anchor is no longer current under the paper's rules is re-emitted against a "
                         "current anchor in one pass, counted per (model, view, eval category, anchor kind); with --dry-run the same counts "
                         "over a read-only connection and nothing written")
    ap.add_argument("--models", default=None, help="the sweep's models: `all` (default) or a comma list of model ids / view_legacy keys / view prefixes")
    ap.add_argument("--limit", type=int, default=None, help="the sweep's bound: at most this many re-emissions; the remainder is counted as deferred")
    args = ap.parse_args(argv)
    defer = () if args.borrowed_defer_category == ['none'] else tuple(args.borrowed_defer_category or BORROWED_DEFER_CATEGORIES)
    try:
        if args.sweep_noncurrent:
            models = None if args.models in (None, "", "all") else [m.strip() for m in str(args.models).split(",") if m.strip()]
            sweep_noncurrent(registry_root=args.registry_root, campaign=args.campaign, models=models, limit=args.limit, spool=args.spool,
                             computed_by=args.computed_by, B=args.bootstrap_b, dry_run=args.dry_run, borrowed_defer_categories=defer,
                             destination_root=args.destination_registry_root)
            return 0
        run(registry_root=args.registry_root, model_ids=args.model_id, campaign=args.campaign, eval_ids=args.eval_id, spool=args.spool,
            computed_by=args.computed_by, B=args.bootstrap_b, dry_run=args.dry_run, borrowed_defer_categories=defer,
            destination_root=args.destination_registry_root)
    except (NoRegistry, DestinationRefused) as exc:
        print(f"[contrasts] {exc}; refusing", file=sys.stderr)
        return 2
    return 0


class NoRegistry(FileNotFoundError):
    """The registry root holds no database: :func:`run` refuses by name (``main`` exits 2; the daemon logs it as ``contrasts_failed``)."""


def run(*, registry_root: Optional[str] = None, model_ids: Optional[Sequence[str]] = None, campaign: Optional[str] = None,
        eval_ids: Optional[Sequence[str]] = None, spool: Optional[str] = None, computed_by: str = DEFAULT_COMPUTED_BY,
        B: int = BOOTSTRAP_B, dry_run: bool = False, borrowed_defer_categories: Sequence[str] = BORROWED_DEFER_CATEGORIES,
        destination_root: Optional[str] = None) -> int:
    """ONE pass over the selection -- what ``main`` runs and the grading daemon calls every tick (``grade_daemon.emit_contrasts_for``):
    every ``due`` contrast computed and written (or skipped, and said so), one ``[contrasts]`` line per steered run considered, then
    the summary line. Returns the number written. A root with no database raises :class:`NoRegistry` naming the path. The spool's
    destination registry is resolved ONCE (:func:`resolve_destination`; ``destination_root`` is the explicit attestation) -- a spool
    folding into another registry without it raises :class:`DestinationRefused` before anything is written -- and every emission reads its
    pending state, source heads and parents from that destination."""
    root = os.path.abspath(os.path.expanduser(registry_root or os.environ.get("REGISTRY_ROOT") or DB.DEFAULT_ROOT))
    path = DB.db_path(root)
    if not os.path.exists(path):
        raise NoRegistry(f"no registry database at {path}")
    spool = spool or emit.spool_root() or H.records_dir(root)
    dest_root, same_dest = resolve_destination(root, spool, destination_root)      # refused by name before any write
    conn = DB.connect(path, readonly=True)
    dest = conn if same_dest else DB.connect(DB.db_path(dest_root), readonly=True)
    if not same_dest:
        dest.execute("PRAGMA query_only = 1")
        print(f"[contrasts] destination registry {dest_root} (attested; the queried registry is {root})", flush=True)
    try:
        due = contrasts_due(conn, model_ids=model_ids, campaign=campaign, eval_ids=eval_ids, borrowed_defer_categories=borrowed_defer_categories)
        withdrawn = frame_withdrawn_keys(dest)                       # once per pass, for every emission (the destination's frame)
        n_written = 0
        n_withdrawn = 0
        n_readmitted = 0
        for d in due:
            if d["status"] != "due":
                if d.get("withdraw_stale"):
                    # the kept row stands; the frame-visible retired keys beside it are withdrawn (one transition file)
                    status, row = emit_withdrawal(root, spool, d, conn=conn, dry_run=dry_run, withdrawn=withdrawn, dest=dest, destination_root=dest_root)
                    n_withdrawn += len(((row or {}).get("extra") or {}).get("withdrawn_contrasts") or []) if isinstance((row or {}).get("extra"), Mapping) else 0   # type: ignore[union-attr]
                    print(_line(status, d, row), flush=True)
                    continue
                print(_line(d["status"], d, None), flush=True)
                continue
            status, row = emit_contrast(root, spool, d, conn=conn, computed_by=computed_by, dry_run=dry_run, B=B, withdrawn=withdrawn, dest=dest, destination_root=dest_root)
            n_written += status == "written"
            n_withdrawn += len(((row or {}).get("extra") or {}).get("withdrawn_contrasts") or []) if isinstance((row or {}).get("extra"), Mapping) else 0   # type: ignore[union-attr]
            n_readmitted += status == "readmitted"
            print(_line(status, d, row), flush=True)
        side = (f"; {n_withdrawn} filed contrast(s) withdrawn from the frame" if n_withdrawn else "") + (f"; {n_readmitted} re-admitted" if n_readmitted else "")
        print(f"[contrasts] {len(due)} steered run(s) considered; {sum(1 for d in due if d['status'] == 'due')} due; "
              f"{n_written} written{' (dry run: nothing written)' if dry_run else ''}{side}; spool {spool}", flush=True)
    finally:
        if not same_dest:
            dest.close()
        conn.close()
    return n_written


# --------------------------------------------------------------------------------------------
# the bounded re-emission sweep
# --------------------------------------------------------------------------------------------

#: The sweep's buckets from a due row's status. A row the sweep counts carries `retired_anchors`: a contrast this emitter filed whose anchor
#: the paper's rules no longer carry as current (migration 0021 drops such a contrast from the frame while the metric row stays).
SWEEP_BUCKETS: Dict[str, str] = {"due": "due", "borrowed_cap_mismatch": "cap_mismatch", "no_anchor": "no_anchor", "no_u0_anchor": "no_anchor",
                                 "no_cap_matched_u0_anchor": "no_anchor"}
#: The sweep's counts per cell and in total: `due` (a row the sweep re-emits or emits for the first time: a current anchor exists), `emitted`
#: (written, or `dry_run`: the same rows and nothing written), `no_anchor` (a retired anchor with no current replacement), `cap_mismatch` (a
#: current wrong-cap borrowed row stands), `deferred` (past --limit), `first` (of `due`: no native row at the key yet -- a run the daemon owns
#: that has no contrast), `retired` (filed rows of the cell's class whose anchor is no longer current -- for an adapter cell the only count),
#: `rejudged` (filed rows whose anchor run was re-judged: re-emitted on the same key with the new anchor judgment).
SWEEP_COUNT_KEYS: Tuple[str, ...] = ("due", "emitted", "no_anchor", "cap_mismatch", "deferred", "first", "retired", "rejudged")
#: The side counts beside SWEEP_COUNT_KEYS (the daemon's summary block reads the pinned tuple above; these ride the cells, the
#: by-class and total dicts, and the printed lines only when non-zero): `withdrawn` (filed rows retired in the frame's own terms by the
#: cell's emissions), `readmitted` (withdrawn keys re-admitted), `frame_stale` (retired rows the frame still joins with no replacement to
#: file: named, nothing written -- the frame's own rule decides).
SWEEP_SIDE_KEYS: Tuple[str, ...] = ("withdrawn", "readmitted", "frame_stale")


def _model_selected(row: Mapping[str, object], models: Optional[Sequence[str]]) -> bool:
    """``--models``: None / `all` selects every row; else a row whose model_id, view_legacy or view prefix (`<key>_`) matches an entry."""
    if not models or "all" in models:
        return True
    mid, view = str(row.get("model_id") or ""), str(row.get("view_legacy") or "")
    return any(m == mid or m == view or (view and view.startswith(str(m) + "_")) for m in models)


def sweep_noncurrent(*, registry_root: Optional[str] = None, campaign: Optional[str] = None, models: Optional[Sequence[str]] = None,
                     limit: Optional[int] = None, spool: Optional[str] = None, computed_by: str = DEFAULT_COMPUTED_BY, B: int = BOOTSTRAP_B,
                     dry_run: bool = False, borrowed_defer_categories: Sequence[str] = BORROWED_DEFER_CATEGORIES, quiet: bool = False,
                     first_skip_models: Optional[Sequence[str]] = None, destination_root: Optional[str] = None) -> Dict[str, object]:
    """ONE bounded pass over EVERY native run of the registry (``campaign`` / ``models`` optional; default every model and view) whose filed
    contrast's anchor is no longer current under the paper's rules: each such run is re-emitted against a current anchor through the very
    same ``contrasts_due`` / ``emit_contrast`` the per-tick pass uses (the full-key kept-row check, the current judgment, the cap filter,
    the idempotence checks), at most ``limit`` emissions (the remainder counted ``deferred``), and counted per (model_id, view_legacy,
    eval_category, anchor_kind): ``due`` (a current anchor exists), ``emitted`` (written -- or ``dry_run``: the same rows, nothing
    written), ``no_anchor`` (the retired anchor has no current replacement), ``cap_mismatch`` (a current wrong-cap borrowed row stands),
    ``skipped`` (emit_contrast's other answers, by status), ``other`` (a retired row beside a kept one, or a run the checks refuse). The
    connection is read-only (``mode=ro`` and ``PRAGMA query_only``, asserted: ``readonly`` rides the result); emission writes the spool,
    never the database. Returns ``{readonly, dry_run, limit, cells, totals, spool}``; prints one `[contrasts-sweep]` line per cell and the
    totals unless ``quiet``. A root with no database raises :class:`NoRegistry`."""
    root = os.path.abspath(os.path.expanduser(registry_root or os.environ.get("REGISTRY_ROOT") or DB.DEFAULT_ROOT))
    path = DB.db_path(root)
    if not os.path.exists(path):
        raise NoRegistry(f"no registry database at {path}")
    spool = spool or emit.spool_root() or H.records_dir(root)
    dest_root, same_dest = resolve_destination(root, spool, destination_root)      # refused by name before any write
    conn = DB.connect(path, readonly=True)                       # URI mode=ro: the database cannot be written through this connection
    conn.execute("PRAGMA query_only = 1")
    readonly = int(conn.execute("PRAGMA query_only").fetchone()[0]) == 1
    if not readonly:
        conn.close()
        raise RuntimeError("sweep_noncurrent: the connection is not read-only; refusing")
    dest = conn if same_dest else DB.connect(DB.db_path(dest_root), readonly=True)
    if not same_dest:
        dest.execute("PRAGMA query_only = 1")
    cells: Dict[Tuple[str, str, str, str, str], Dict[str, object]] = {}

    def cell(cls: str, d: Mapping[str, object], kind: Optional[str] = None) -> Dict[str, object]:
        # the table is keyed by the rows' writer (record_uri class) FIRST, then (model, view_legacy, eval_category, anchor_kind)
        k = (cls, str(d.get("model_id") or ""), str(d.get("view_legacy") or ""), str(d.get("eval_category") or ""), str(kind or d.get("anchor_kind") or "none"))
        return cells.setdefault(k, {"uri_class": k[0], "model_id": k[1], "view_legacy": k[2], "eval_category": k[3], "anchor_kind": k[4],
                                    **{c: 0 for c in SWEEP_COUNT_KEYS}, **{c: 0 for c in SWEEP_SIDE_KEYS}, "why": {}, "skipped": {}, "other": {}})

    def side_text(c: Mapping[str, object]) -> str:
        # the side counts, printed only when non-zero (the pinned line is unchanged where nothing was withdrawn or re-admitted)
        return "".join(f" {name}={c[name]}" for name in SWEEP_SIDE_KEYS if int(c[name] or 0))   # type: ignore[call-overload]

    def why_text(w: Mapping[str, int]) -> str:
        # the predicate's why beside the counts -- one why plain, several as {why: count} -- so a hold-out reads as a hold-out, not a missing baseline
        return "" if not w else (" why=" + (next(iter(w)) if len(w) == 1 else json.dumps(dict(sorted(w.items())), sort_keys=True)))

    def say(line: str) -> None:
        if not quiet:
            print(line, flush=True)

    n_emit = 0
    try:
        withdrawn = frame_withdrawn_keys(dest)                       # once per sweep, for every emission (the destination's frame)
        if not same_dest:
            say(f"[contrasts-sweep] destination registry {dest_root} (attested; the queried registry is {root})")
        for d in contrasts_due(conn, campaign=campaign, borrowed_defer_categories=borrowed_defer_categories):
            if not _model_selected(d, models):
                continue
            # an adapter's rows at this key whose anchor is retired: counted in the adapter cells, never re-filed
            for a in d.get("adapter_contrasts") or []:
                if not a.get("is_current"):
                    ac = cell("adapter", d, str(a.get("anchor_kind")))
                    ac["retired"] = int(ac["retired"]) + 1   # type: ignore[arg-type]
            retired = list(d.get("retired_anchors") or [])
            rejudged = list(d.get("rejudged_anchors") or [])             # filed rows whose anchor run was re-judged (an in-place update)
            if not retired and not rejudged and not d.get("first_emission"):
                continue
            if str(d["status"]) in ("same_view_current", "borrowed_current"):
                stale_kept = list(d.get("withdraw_stale") or [])
                if not stale_kept:
                    continue                             # a current row stands beside the retired one(s): resolved, nothing to count or do
                # the retired keys the frame still joins beside the kept row are withdrawn by the sweep too
                ck_ = cell(sorted({str(x.get("uri_class") or "native") for x in stale_kept})[0] if len({str(x.get("uri_class") or "native") for x in stale_kept}) == 1 else "mixed", d)
                ck_["retired"] = int(ck_["retired"]) + len(stale_kept)   # type: ignore[arg-type]
                for x in stale_kept:
                    ck_["why"][str(x.get("why"))] = int(ck_["why"].get(str(x.get("why")), 0)) + 1   # type: ignore[index]
                if limit is not None and n_emit >= int(limit):
                    ck_["deferred"] = int(ck_["deferred"]) + 1   # type: ignore[arg-type]
                    say(_line("deferred", d, None) + f" note=the sweep's --limit {limit} is reached; withdrawn by a later sweep")
                    continue
                status, row = emit_withdrawal(root, spool, d, conn=conn, dry_run=dry_run, withdrawn=withdrawn, dest=dest, destination_root=dest_root)
                if status in ("withdrawn", "dry_run"):
                    extra = (row or {}).get("extra") if isinstance((row or {}).get("extra"), Mapping) else {}
                    ck_["withdrawn"] = int(ck_["withdrawn"]) + len(extra.get("withdrawn_contrasts") or []); n_emit += 1   # type: ignore[arg-type, union-attr]
                else:
                    ck_["skipped"][status] = int(ck_["skipped"].get(status, 0)) + 1   # type: ignore[index]
                say(_line(status, d, row))
                continue
            if d.get("first_emission") and not retired and not rejudged and first_skip_models and _model_selected(d, first_skip_models):
                continue                                 # the daemon's own model's first emissions are the per-tick pass's, not the sweep's
            classes = sorted({str(x.get("uri_class") or "native") for x in retired + rejudged}) or ["native"]
            c = cell(classes[0] if len(classes) == 1 else "mixed", d)
            c["retired"] = int(c["retired"]) + len(retired)   # type: ignore[arg-type]
            c["rejudged"] = int(c["rejudged"]) + len(rejudged)   # type: ignore[arg-type]
            for x in retired:
                c["why"][str(x.get("why"))] = int(c["why"].get(str(x.get("why")), 0)) + 1   # type: ignore[index]
            bucket = SWEEP_BUCKETS.get(str(d["status"]))
            if bucket is None:
                c["other"][str(d["status"])] = int(c["other"].get(str(d["status"]), 0)) + 1   # type: ignore[index]
                say(_line(str(d["status"]), d, None))
                continue
            c[bucket] = int(c[bucket]) + 1   # type: ignore[arg-type]
            if bucket != "due":
                c["frame_stale"] = int(c["frame_stale"]) + len(d.get("frame_stale") or [])   # type: ignore[arg-type]  still joined, no replacement
                say(_line(str(d["status"]), d, None))
                continue
            if d.get("first_emission"):
                c["first"] = int(c["first"]) + 1   # type: ignore[arg-type]
            if limit is not None and n_emit >= int(limit):
                c["deferred"] = int(c["deferred"]) + 1   # type: ignore[arg-type]
                say(_line("deferred", d, None) + f" note=the sweep's --limit {limit} is reached; re-emitted by a later sweep")
                continue
            status, row = emit_contrast(root, spool, d, conn=conn, computed_by=computed_by, dry_run=dry_run, B=B, withdrawn=withdrawn, dest=dest, destination_root=dest_root)
            if status in ("written", "dry_run"):
                c["emitted"] = int(c["emitted"]) + 1; n_emit += 1   # type: ignore[arg-type]
                extra = (row or {}).get("extra") if isinstance((row or {}).get("extra"), Mapping) else {}
                c["withdrawn"] = int(c["withdrawn"]) + len(extra.get("withdrawn_contrasts") or [])   # type: ignore[arg-type, union-attr]  counts the contrasts the emitted row withdrew
                c["readmitted"] = int(c["readmitted"]) + (1 if extra.get("readmitted") else 0)   # type: ignore[arg-type, union-attr]
            elif status == "readmitted":
                c["readmitted"] = int(c["readmitted"]) + 1; n_emit += 1   # type: ignore[arg-type]  a withdrawn key re-admitted (counted against --limit)
            else:
                c["skipped"][status] = int(c["skipped"].get(status, 0)) + 1   # type: ignore[index]
            say(_line(status, d, row))
    finally:
        if not same_dest:
            dest.close()
        conn.close()
    totals: Dict[str, object] = {c: 0 for c in SWEEP_COUNT_KEYS}
    totals.update({c: 0 for c in SWEEP_SIDE_KEYS})
    totals["skipped"] = {}; totals["other"] = {}
    by_class: Dict[str, Dict[str, int]] = {}
    side_by_class: Dict[str, Dict[str, int]] = {}
    for k in sorted(cells):
        c = cells[k]
        bc = by_class.setdefault(str(c["uri_class"]), {name: 0 for name in SWEEP_COUNT_KEYS})
        sc = side_by_class.setdefault(str(c["uri_class"]), {name: 0 for name in SWEEP_SIDE_KEYS})   # beside, never inside, the pinned dict
        for name in SWEEP_COUNT_KEYS:
            totals[name] = int(totals[name]) + int(c[name])   # type: ignore[arg-type]
            bc[name] += int(c[name])   # type: ignore[arg-type]
        for name in SWEEP_SIDE_KEYS:
            totals[name] = int(totals[name]) + int(c[name])   # type: ignore[arg-type]
            sc[name] += int(c[name])   # type: ignore[arg-type]
        for sub in ("skipped", "other"):
            for s, n in c[sub].items():   # type: ignore[union-attr]
                totals[sub][s] = int(totals[sub].get(s, 0)) + int(n)   # type: ignore[index]
        say(f"[contrasts-sweep] class={c['uri_class']} model={c['model_id']} view={c['view_legacy']} category={c['eval_category']} kind={c['anchor_kind']} "
            + " ".join(f"{name}={c[name]}" for name in SWEEP_COUNT_KEYS) + side_text(c) + why_text(c["why"])   # type: ignore[arg-type]
            + (f" skipped={json.dumps(c['skipped'], sort_keys=True)}" if c["skipped"] else "") + (f" other={json.dumps(c['other'], sort_keys=True)}" if c["other"] else ""))
    for cls in sorted(by_class):
        say(f"[contrasts-sweep] CLASS {cls} " + " ".join(f"{name}={by_class[cls][name]}" for name in SWEEP_COUNT_KEYS) + side_text(side_by_class[cls]))
    notes: List[str] = []
    if "other" in by_class:
        # the classification stays by prefix (the adapters write under records/adapters/, live `other` is 0); a
        # non-empty `other` class is named so a spool outside records/<campaign>/ is seen, not silently folded into ours
        n_other = sum(int(by_class["other"][name]) for name in ("due", "no_anchor", "cap_mismatch", "first", "retired", "rejudged"))
        notes.append(f"NOTE: {len([c for c in cells.values() if c['uri_class'] == 'other'])} cell(s) / {n_other} row count(s) under the `other` uri class -- "
                     "contrast rows under neither records/adapters/ nor records/<campaign>/ (a spool outside the standard layout), treated as this emitter's")
        say(f"[contrasts-sweep] {notes[-1]}")
    say(f"[contrasts-sweep] TOTAL cells={len(cells)} " + " ".join(f"{name}={totals[name]}" for name in SWEEP_COUNT_KEYS) + side_text(totals)
        + (f" skipped={json.dumps(totals['skipped'], sort_keys=True)}" if totals["skipped"] else "") + (f" other={json.dumps(totals['other'], sort_keys=True)}" if totals["other"] else "")
        + (f"; nothing written (dry run, read-only connection); spool {spool}" if dry_run else f"; {totals['emitted']} written; spool {spool}"))
    return {"readonly": readonly, "dry_run": bool(dry_run), "limit": limit, "cells": [cells[k] for k in sorted(cells)], "by_class": by_class, "totals": totals,
            "side_by_class": side_by_class, "notes": notes, "spool": spool}


if __name__ == "__main__":
    raise SystemExit(main())
