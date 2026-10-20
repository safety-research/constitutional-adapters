"""The recurring incremental adapter pass, and its re-read mode
(``registry adapters run --reread``, D73): every committed source is read again and a bundle under
``_activations/<rev>/<stamp>/`` carries the read's ``source_status`` facts and a fresh marker — without re-emitting
the measurements of a head whose keys the read reproduces (:func:`reread_committed`).

A group whose last read REFUSED is read again only on an explicit cause — the rule is :func:`plan_groups`' docstring; the
refusal stays visible meanwhile (``source_watermark.last_error``, the page's adapter-pass rows, ``v_source_status``) and the
pass names what it left standing (``PassReport.refused_standing``).

For every adapter: ``discover()`` (every input through :func:`base.check_input`, whose sha comes from the pass's
:class:`base.InputHashCache` when the file's full-resolution stat identity is unchanged — an unchanged pass reads no store
byte, S2) → per-input ``(sha256, size, mtime)`` vs ``source_watermark`` → the changed inputs' COMPARISON GROUPS are
re-read over ALL of the group's inputs in canonical order (a
relocated shared-RNG computation re-runs over its original traversal) → the result is STAGED under
``<root>/staging/adapters/<adapter>/<revision_id>/`` — outside the ingestible namespace — and published
into ``records/adapters/<adapter>/<revision_id>/`` only when the whole group read: the batch files first,
the ``source_revision`` file (revision + ``observation`` + ``withdrawal`` + ``source_state`` +
``source_watermark`` records) LAST, as the commit marker. A read that refused publishes ONLY an UNACCEPTED
revision and the watermarks' ``last_error`` (visible, never a silent skip); no batch of it reaches the
log and the previous revision stays current. Revision ids are namespaced by source (adapter, comparison
group, input URIs and shas) AND by the adapter's ``EMISSION_VERSION`` (D206, :func:`revision_id_for`): a
change to what an adapter parks or spells, bumped there, is a new revision over unchanged inputs — the
incremental pass reads a group whose head was minted under another version, and the old head is superseded
through the accepted chain — while ``--reread --force`` folds the pass stamp in as a nonce for the operational
case (a code change nobody versioned). A comparison group whose every input has vanished from discovery — as
distinct from a discovery FAILURE, which refuses and withdraws nothing — commits an empty revision that
withdraws its keys with reason ``input_vanished`` and records ``source_state found = 0`` for each input.
THE STATE-ROW RULE (2026-10-03): the ``source_state`` rows a revision publishes are the source's WHOLE state -- every row of the
source that the revision does not restate is written ``found = 0`` beside them (a vanished input's rows, a key an earlier read
used; :func:`_cleared_state_records`), on a fresh read, on a re-activation and on a retirement alike; a re-activation replays the
COMMITTED revision's own state rows (:func:`committed_state_records`) and never asks the adapter, whose per-read ledger
(``$.row_refusals``, keyed on the group: :func:`base.row_refusals_uri`) belongs to the read that filed it.
THE STATE-KEY OWNERSHIP RULE (2026-10-03, the same review's third finding). Not every state row a read writes is the GROUP's: a
row keyed on one of the revision's inputs (its ``input_uri`` an identity URI in the marker's ``inputs_json``) or on the group's
ledger URI says what THAT revision's bytes name and is GROUP-OWNED; a row keyed on anything else -- an absent file, a
ruling-spelled path -- says what the reader OBSERVED at read time and is AUXILIARY. An adapter whose owned row rides a key that
is not its input's identity URI declares the path (``state_owned_paths``, :func:`base.state_owned_paths_of`). A re-activation
replays and clears GROUP-OWNED rows only (:func:`_owned_state_key`: the committed revision's inputs for the replay, those plus
the outgoing head's for the displaced keys); it never re-stamps a committed snapshot of an auxiliary key, so the LATEST FRESH
READ's row of such a key stands (a re-activation observes nothing). A fresh read is a whole observation of the group and still
clears every row of its source it does not restate, and a retirement clears them all.
LISTINGS ARE INPUTS (2026-10-04, the review's fourth finding). What a directory CONTAINS is never a side observation of a read:
a reader that states a listing (``em_misalign``'s ``$.entries`` of its stores, the root inventory ``gsm8k_p200`` states as
``$.dirs``) discovers the listing as a synthetic INPUT of the group (:class:`base.ListingInput`: ``registry://listing/<source_id>/
<rel dir>``, sha the fingerprint of the sorted names and kinds), so a listing change is a revision change (the group re-reads),
reverting the listing re-activates the revision whose row matches, and the row keyed on the listing URI is group-owned and replays.
An auxiliary ``$.entries`` keyed on the directory path (the earlier shape) was neither replayed nor cleared: commit group G, add a
skipped-by-name entry (``killed_alphasteer``) while changing a tracked CSV and commit G again, remove the entry and restore the CSV
-- G's earlier revision re-activated and its ``$.entries`` kept asserting the removed entry indefinitely. A ROOT-WIDE inventory has
ONE publisher: a ROOT pseudo-group whose only input is the root's listing (``gsm8k_p200``: group ``<root basename>``, source id
``gsm8k_p200:<root basename>`` as before, its read emits no record and its state is ``$.dirs``); its revision changes exactly when
the root listing changes -- a retired group's directory vanishing changes it, so the root group re-reads and restates ``$.dirs``
without it while every other group stays quiet -- and the per-group reads state no root inventory. ON A CLUSTER SHARD a listing
input is read against the store manifest's attestation of the canonical directory (:class:`ClusterListingRefusal`; the sixth
review): the pinned generation's manifest attests, per directory, the canonical's whole listing at the window's push -- empty
directories included (``stores.MANIFEST_RULES["directories"]``, an additive field; the verify job reproduces attested empty
directories and verifies every listing, admission and the grading host revalidation check them too) -- and the shard compares every
listing input's fingerprint with THAT, never with a previous accepted revision (whose listing a stale stage can equal while the
canonical moved: an empty ignored directory landing as a tracked file changed). No attestation (``staged_listing_unattested``) or
a differing one (``staged_listing_mismatch``) refuses the group by name, nothing written; the pull records the adapter refused, and no
grading host read follows (the next window's read is the next attempt).
A listing-only change of the canonical publishes no new generation (a generation is named by its file content): the mirror records
``listings_changed`` and the revalidation refuses the component, so those readers are recorded refused (read nowhere) until a file change
publishes the next generation. The shared class is therefore
empty; the auxiliary class holds one self-correcting absence row: ``cap4096_declarations``' gone pinned guard, ``found = 0`` under
the rulings' spelling of its path while the file is absent -- present, the guard is an input of that very group (its ``inp.uri``
row, newer, the evaluator's latest-wins), so its return re-reads the group and no stale state of it survives a re-activation. The
cap-4,096 union file's absence is no longer anyone's auxiliary row: its PRESENCE is an input of its own group (``degeneracy_union``'s
``_mt_probe``, a listing of the union-file spellings, empty while in flight), so the group exists, states ``$.entries`` = [] as its
own owned row and re-reads when the file arrives (the greedy ``full22`` group used to state the absence under ITS source id and
nothing refreshed it: the arriving file joins ``mt_cap4096``, not ``full22``). One key is RETIRED: ``gsm8k_p200``'s pre-rule
root-path ``$.dirs`` under ``gsm8k_p200:<root basename>``, stated absent by every read of the root group. Every state key the
adapters emit, classified (``tests/test_registry_source_state_per_revision.py`` pins that this table names each emission site, by
module, key expression and path expression, and nothing more):

| adapter | input_uri | json_path | class | note |
|---|---|---|---|---|
| _bcbw_common | `inp.uri` | `'$'` | owned |  |
| _bcbw_common | `inp.uri` | `'$.rows'` | owned |  |
| _bcbw_common | `inp.uri` | `f'$.{mk}'` | owned |  |
| _bcbw_common | `inp.uri` | `self.row_table_json_path(doc, spec.launch_set_eval_key)` | owned |  |
| _bcbw_common | `row_refusals_uri(source_id)` | `$.row_refusals` | owned | the per-read ledger on the group key |
| alphasteer_rebake | `op.uri` | `f'$.targets.{model}.object_sha256'` | owned |  |
| arditi_allpos | `agg.uri` | `'$.cells'` | owned |  |
| arditi_allpos | `anc.uri` | `f'$.models.{view_legacy}.sources.none'` | owned |  |
| arditi_allpos | `curves.uri` | `'$.parameters.views_declared'` | owned |  |
| arditi_allpos | `curves.uri` | `f'$.targets.{MODEL}.cells'` | owned |  |
| arditi_allpos | `man.uri` | `'$.attacks'` | owned |  |
| arditi_allpos | `sel.uri` | `'$.selected.view + $.prefix_rule_at_selection_cap.view'` | owned |  |
| arditi_strength | `curves.uri` | `'$.targets'` | owned |  |
| arditi_vendor_mt | `inp.uri` | `'$.cells'` | owned |  |
| arditi_vendor_mt | `inputs[0].uri` | `$.row_refusals` | owned | the per-read ledger on the group's first input (the pre-commit-7 key) |
| bcb_bfcl | `i.uri` | `'$'` | owned |  |
| bon_native | `inp.uri` | `'$.cell_id'` | owned |  |
| bon_native | `inputs[0].uri` | `$.row_refusals` | owned | the per-read ledger on the group's first input (the pre-commit-7 key) |
| bon_stores | `inp.uri` | `'$.cells' if 'cells' in data else '$'` | owned |  |
| bon_stores | `inputs[0].uri` | `$.row_refusals` | owned | the per-read ledger on the group's first input (the pre-commit-7 key) |
| cap4096_declarations | `inp.uri` | `'$'` | owned |  |
| cap4096_declarations | `lane_legacy.uri` | `jp` | owned |  |
| cap4096_declarations | `lane_legacy.uri` | `str(jp)` | owned |  |
| cap4096_declarations | `ruling_uri(os.path.join(self.run_state, rel), self.home)` | `'$'` | auxiliary (self-correcting) | a pinned guard file that is gone, found = 0 under the rulings' spelling of its path; the guard is an input of this very group when present (its `inp.uri` row, newer, the evaluator's latest-wins), so its return re-reads the group |
| cellstore | `ORC.source_state_rows` | `(its rows)` | owned | the OR role-collapse helper's rows (its own line below) |
| cellstore | `i.uri` | `'$.controls'` | owned |  |
| cellstore | `inputs[0].uri` | `$.row_refusals` | owned | the per-read ledger on the group's first input (the pre-commit-7 key) |
| degeneracy_union | `first.uri` | `$.row_refusals` | owned | the per-read ledger on the group's first input (the pre-commit-7 key) |
| degeneracy_union | `inp.uri` | `$.row_refusals` | owned | the per-read ledger on the group's first input (the pre-commit-7 key) |
| degeneracy_union | `inp.uri` | `'$.cells'` | owned |  |
| degeneracy_union | `inp.uri` | `'$.entries'` | owned | the cap-4,096 union file's presence probe (`_mt_probe`): the group's own listing input, [] while in flight |
| degeneracy_union | `inp.uri` | `'$.rows'` | owned |  |
| em_misalign | `i.uri` | `'$.arm'` | owned |  |
| em_misalign | `i.uri` | `'$.entries'` | owned | a store's / target's / view's listing input (base.ListingInput): the entries are the input's content |
| gsm8k_p200 | `inp.uri` | `'$.cells'` | owned |  |
| gsm8k_p200 | `inp.uri` | `'$.dirs'` | owned | the ROOT pseudo-group `_root`'s row on the root listing input (ROOT_LISTING_RULE): one publisher, keyed on the listing |
| gsm8k_p200 | `self.root` | `'$.dirs'` | retired (stated absent) | the pre-rule root-path key under `gsm8k_p200:<root basename>`, stated ABSENT (found = 0) by every read of the root group so a database that holds it never keeps it found = 1 |
| gsm8k_paired_allmodels | `inp.uri` | `'$.pins'` | owned |  |
| gsm8k_paired_allmodels | `inputs[0].uri` | `$.row_refusals` | owned | the per-read ledger on the group's first input (the pre-commit-7 key) |
| gsm8k_seedrep | `inp.uri` | `'$.cells'` | owned |  |
| gsm8k_seedrep | `inp.uri` | `'$.pins'` | owned |  |
| gsm8k_seedrep | `inputs[0].uri` | `$.row_refusals` | owned | the per-read ledger on the group's first input (the pre-commit-7 key) |
| ifeval_seedrep | `inp.uri` | `'$.promoted'` | owned |  |
| ifeval_seedrep | `inp.uri` | `'$.rows'` | owned |  |
| ifeval_seedrep | `inp.uri` | `B.BcbwSeedrepReader.row_table_json_path(doc, EVAL_ID)` | owned |  |
| ifeval_seedrep | `inputs[0].uri` | `$.row_refusals` | owned | the per-read ledger on the group's first input (the pre-commit-7 key) |
| legacy_glm | `i.uri` | `'$'` | owned |  |
| lineage_common | `input_uri` | `ROW_REFUSALS_PATH` | owned | the helper every `source_state_rows` caller renders: the ledger on the group's first input |
| lineage_common | `row_refusals_uri(source_id)` | `ROW_REFUSALS_PATH` | owned | the helper every `ledger_rows` caller renders: the ledger on the group key |
| llama8b_wave | `ex.uri` | `f'$.models.{tag}.cells'` | owned |  |
| math500 | `inp.uri` | `'$.legs'` | owned |  |
| math500_seedrep | `inp.uri` | `'$.pins'` | owned |  |
| math500_seedrep | `inp.uri` | `'$.rows'` | owned |  |
| math500_seedrep | `inp.uri` | `'$.stores'` | owned |  |
| math500_seedrep | `inp.uri` | `B.BcbwSeedrepReader.row_table_json_path(d, EVAL_ID)` | owned |  |
| math500_seedrep | `inputs[0].uri` | `$.row_refusals` | owned | the per-read ledger on the group's first input (the pre-commit-7 key) |
| multiturn_grades | `inp.uri` | `'$.cells'` | owned |  |
| multiturn_grades | `inputs[0].uri` | `$.row_refusals` | owned | the per-read ledger on the group's first input (the pre-commit-7 key) |
| orxs_role_collapse | `summary_in.uri` | `$.row_refusals` | owned | the per-read ledger on the group's first input (the pre-commit-7 key) |
| orxs_role_collapse | `summary_in.uri` | `'$.cells'` | owned |  |
| overrefusal_items | `i.uri` | `'$.cells'` | owned |  |
| petri_stats | `inp.uri` | `'$.*.cells'` | owned |  |
| petri_stats | `inp.uri` | `'$.tags.*.cells'` | owned |  |
| petri_stats | `inputs[0].uri` | `$.row_refusals` | owned | the per-read ledger on the group's first input (the pre-commit-7 key) |
| r2_export | `pq.uri if pq else self._p(PARQUET)` | `$.row_refusals` | owned | the per-read ledger on the group's first input (the pre-commit-7 key) |
| r2_export | `pq.uri if pq else self._p(PARQUET)` | `'$.rows'` | owned |  |
| r2_export | `ruling_uri(pq.uri if pq else self._p(PARQUET))` | `'$.arm_operating_point_id'` | owned (declared) | the parquet's member set under the rulings' `~`-spelling of its path: `state_owned_paths` |
| regrade_twin | `i.uri` | `'$.units'` | owned |  |
| relu_native | `i.uri` | `'$.cells'` | owned |  |
| relu_paired | `i.uri` | `'$.canonical[*],controls[*]'` | owned |  |
| relu_paired | `i.uri` | `'$.cells'` | owned |  |
| relu_paired | `i.uri` | `'$.ladder'` | owned |  |
| relu_paired | `i.uri` | `'$.models'` | owned |  |
| relu_paired | `i.uri` | `'$.object.staged_id'` | owned |  |
| relu_paired | `i.uri` | `'$.objects'` | owned |  |
| relu_paired | `i.uri` | `'$.targets'` | owned |  |
| relu_paired | `i.uri` | `f'$.targets.{model}.controls'` | owned |  |
| sampled_controls | `i.uri` | `'$.held_cells'` | owned |  |
| sampled_controls | `i.uri` | `'$.pending'` | owned |  |
| sampled_controls | `i.uri` | `'$.targets'` | owned |  |
| sampled_static | `i.uri` | `'$.targets'` | owned |  |
| sampled_static | `row_refusals_uri(source_id)` | `$.row_refusals` | owned | the per-read ledger on the group key |
| seedrep_cluster | `inp.uri` | `'$.cells'` | owned |  |
| seedrep_cluster | `inp.uri` | `'$.layout'` | owned |  |
| seedrep_cluster | `inputs[0].uri` | `$.row_refusals` | owned | the per-read ledger on the group's first input (the pre-commit-7 key) |
| seedrep_controls | `inp.uri` | `'$.accepted_base'` | owned |  |
| seedrep_controls | `inputs[0].uri` | `$.row_refusals` | owned | the per-read ledger on the group's first input (the pre-commit-7 key) |
| seedrep_controls | `inputs[0].uri` | `'$.runs'` | owned |  |
| seedrep_petri | `inp.uri` | `$.row_refusals` | owned | the per-read ledger on the group's first input (the pre-commit-7 key) |
| seedrep_petri | `inp.uri` | `'$.absent'` | owned |  |
| seedrep_petri | `inp.uri` | `'$.canonical[*]'` | owned |  |
| seedrep_statics | `inp.uri` | `'$.accepted_base'` | owned |  |
| seedrep_statics | `inp.uri` | `'$.cells'` | owned |  |
| seedrep_statics | `inputs[0].uri` | `$.row_refusals` | owned | the per-read ledger on the group's first input (the pre-commit-7 key) |
| seedrep_statics | `inputs[0].uri` | `'$.runs'` | owned |  |
| strength_ladder | `inp.uri` | `'$.missing'` | owned |  |
| strength_ladder | `inp.uri` | `'$.targets'` | owned |  |
| strength_mtdiff | `inp.uri` | `'$.cells'` | owned |  |
| strength_mtdiff | `inp.uri` | `'$.controls'` | owned |  |
| strength_mtdiff | `inputs[0].uri` | `$.row_refusals` | owned | the per-read ledger on the group's first input (the pre-commit-7 key) |

IDENTITY vs ADDRESSING (registry-slurm-fast-windows Phase 3; D4.1, D5.2). Every group is read under its LOCAL spelling -- the adapter's
``comparison_group``, its ``Input.uri`` / ``Input.path`` -- while everything that IS identity is minted from the canonical spelling the
pass's :class:`base.IdentityMap` derives: ``source_id_for`` canonicalizes a path-valued group (:func:`base.comparison_group_identity`),
:func:`revision_id_for` and every ``inputs_json`` key ``Input.identity_uri``, the watermarks, state rows, status ``where`` lists and the
records' own strings (``IdentityMap.canonical_deep``). With no map (every grading host pass) the two spellings coincide and nothing changes.
A CLUSTER shard (:class:`ClusterRead`; ``adapters run --reader-state FILE --shard i/K``) compares against a shipped reader-state
sidecar and REFUSES BY NAME any group whose freshly minted revision id the grading host already knows -- a ``source_revision`` row, accepted
or not, or a published directory the sidecar lists (``known to the grading host: <rev>; reactivation is the grading host's``) -- and publishes
nothing for it; only ids the grading host has never seen are published, so a cluster read can never mint an inbox conflict (S11). The shard
DEAL (:func:`deal_shards`) hands each comparison group of the SORTED UNION of the discovered groups and the prior-only groups the
watermarks name, by canonical identity, round-robin to exactly one of K shards, so a vanished group has one deterministic owner.
A CLUSTER shard reads a STAGED COPY, and on a staged copy a dropped input is never a producer's vanished file but an INCOMPLETE COPY --
a tree no component of ``configs/registry/cluster_reads.yaml`` covers, a member the pinned generation lacks (D590 follow-up 1): a group
whose input set lacks ANY input of the previous accepted revision is REFUSED there (:class:`ClusterInputDropRefusal`, ``staged_copy_incomplete``),
whatever share the grading host threshold would admit, and so is a group whose every input is absent UNLESS the adapter DECLARES every one of
them retired (``inputs_retired``, :func:`declared_retirements`; registry-hardening 3a): a deliberate retirement is DECLARED, never inferred
from the read, so a declared one is committed by its owning shard exactly as the grading host path commits it (:func:`retire_vanished_group`; the
marker records the declaration under ``extra.retirement_declared``), and with any vanished input undeclared the retirement is the adapter's
to declare or the grading host's. Nothing is written for a refused group -- no revision, no marker, no watermark -- the refusal names each missing
input with the staged component that would have covered it (:meth:`ClusterRead.covering`), and the pull records the adapter refused; no
grading host read follows. The 2026-09-29 W4 reads that saw
9 / 3 / 9 fewer inputs than the grading host (0.026-0.097 of them, under the 0.10 threshold) minted three reduced-coverage revisions that
superseded the full-coverage heads for seven minutes. The grading host path keeps the threshold: there a vanished file is the producer's.

NO GRADING HOST STORE READ. :func:`run_pass` REQUIRES a :class:`base.StoreAccessScope` before any adapter
discovers (``base.require_store_access``): the cluster shard installs one, the operator's ``REGISTRY_STORE_READ_OVERRIDE=<reason>`` is
the one other way, and without either the pass refuses by name (``base.GradingHostStoreReadRefused``) before a byte is read. The grading host never
reads an adapter's store any more; what it still does from COMMITTED BYTES is re-activate: :func:`reactivate_by_id` rebuilds a
re-activation bundle of a published revision -- the marker's ``inputs_json`` and the committed ``source_watermark`` / ``source_state``
records stand in for a discovery, :func:`_reemit_committed` re-emits the measurements -- for the refused-known ids a cluster read
returns whose ``source_revision`` row is not the source's current one (``cluster-reads pull`` calls it); a RETIREMENT among them (an EMPTY
revision: a retired group that came back and vanished again mints the same id) is re-activated from the DATABASE -- the head's keys, the
inventory's watermarks, the state rows -- as :func:`retire_vanished_group`'s records under the committed id (:func:`_reactivate_retirement`),
since a retirement's committed bytes are a marker alone and the facts it needs are today's. ``--reread --force`` (a read of
the store from the grading host) is therefore an operator override: it runs only under ``REGISTRY_STORE_READ_OVERRIDE``.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
import os
import shutil
import sqlite3
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .. import harvest as H
from .. import identity
from .. import records as R
from .. import site_aliases as SA
from .anchor_current import (AnchorCurrencyUnavailable, AnchorDecisionInput, decision_fingerprint, decision_source_id, decisions_block,
                             is_decision_input, is_decision_uri)
from .base import (DEFAULT_EMISSION_VERSION, Adapter, AdapterContext, AdapterRefusal, IdentityMap, Input, InputHashCache, RecordBatch,
                   comparison_group_identity, emission_version_of, emits_no_runs_of, identity_map, inputs_dropped_threshold_of,
                   inputs_retired_of, is_listing_input, is_listing_uri, is_row_refusals_uri, is_synthetic_uri, listing_fingerprint, listing_rel,
                   listing_source_id, listing_uri, owner_of, require_store_access, row_refusals_uri, runs_retired_of, state_owned_paths_of,
                   status_extra_of, status_sources_of, using_hash_cache, using_identity_map, watermark_of, LISTING_ROLE)

log = logging.getLogger(__name__)


@dataclass
class GroupResult:
    adapter: str
    group: str
    revision_id: str
    accepted: bool
    records_written: int = 0
    withdrawals: int = 0
    error: Optional[str] = None
    files: List[str] = field(default_factory=list)
    reread: bool = False                #: an accepted re-read of a committed revision (a bundle, not a new revision directory)
    held: bool = False                  #: a published revision the harvester holds, or has not folded yet; the pass wrote nothing for it (neither refused nor refreshed)
    forced: bool = False                #: a ``--reread --force`` read: a NEW revision whatever its inputs (its id carries the pass stamp as a nonce)
    emitted_nothing: bool = False       #: an accepted read of store inputs that emitted no record and recorded no reason: its status row is filed ok = 0
                                        #: with ``extra.emitted_nothing_unexplained`` (:func:`emitted_nothing_unexplained`)
    inputs_dropped: bool = False        #: a read REFUSED because it dropped more than the threshold share of the previous accepted revision's
                                        #: inputs without declaring them retired (:func:`inputs_dropped_comparison`); the previous revision stays current.
                                        #: On a CLUSTER shard: ANY dropped input, or every input absent with any of them undeclared
                                        #: (:class:`ClusterInputDropRefusal`, :func:`retire_vanished_group`); nothing written
    refused_known: bool = False         #: a CLUSTER shard refused the group because its freshly minted revision id is known to the grading host (a
                                        #: ``source_revision`` row or a published directory): nothing written; recorded refused, no grading host read follows
    refused_standing: bool = False      #: the grading host's re-activation by id (:func:`reactivate_by_id`) found the id REFUSED AT EMIT TIME and never
                                        #: committed (a ``source_revision`` row folded from a ``_refused/<rev>/`` record; no directory, no bundle):
                                        #: nothing exists to re-activate from, nothing written; the refusal stands as it was and a cluster shard
                                        #: reads the group fresh -- never a refusal of the adapter
    activation_pending: bool = False    #: the revision's marker -- its directory's or a complete activation bundle's -- is ON THE RECORD LOG and the
                                        #: harvester has not folded it yet (:func:`pending_fold`): nothing written; the fold is pending, not absent
    reactivated: bool = False           #: this result WROTE an activation bundle (``_activations/<rev>/<stamp>/``: a re-activation of committed bytes,
                                        #: or a re-read's bundle); what :class:`ReactivationBudget` counts
    deferred: bool = False              #: a re-activation this pass did NOT write because the adapter's :class:`ReactivationBudget` was spent;
                                        #: nothing written, the next pass takes it
    anchor_unavailable: bool = False    #: the pass's anchor currency could not be loaded: the group was PRESERVED before planning (its head
                                        #: recorded a decision set the pass cannot re-evaluate) or its read refused by name
                                        #: (:class:`AnchorCurrencyRefusal`); nothing written either way -- no marker, no stamp, no state row --
                                        #: so the next pass retries; ``error`` carries the loader's cause
    group_identity: str = ""            #: the group's canonical identity (:func:`base.comparison_group_identity`); equals ``group`` on the grading host
    standing_refusal: bool = False      #: a SHARD's re-read of a group that stood refused, refused again BYTE FOR BYTE under the same inputs and the
                                        #: same reading tree (:func:`standing_refusal_of`): recorded like any refusal (its record travels with
                                        #: the pull), never a refusal of the adapter; the per-group state string is ``standing_refusal``
    prior_refusal: Optional[Dict[str, object]] = None   #: with ``standing_refusal``: the recorded refusal this one equals -- ``window`` (the operation
                                                        #: that recorded it, None for a grading host read), ``utc`` (its stamp), ``digest`` (sha256 over the
                                                        #: refusal text, the input digest and the tree sha; :func:`standing_refusal_digest`),
                                                        #: ``attempt_revision_id`` (the input digest) and ``code_sha`` (the tree)
    withheld: bool = False              #: the grading host's re-activation by id of a RETIREMENT (:func:`_reactivate_retirement`) WITHHELD by name: the shard's
                                        #: refused-known claim (``retirement``) is absent, or names an outgoing head or an input inventory that is not the
                                        #: live one (a newer revision of the source was accepted since the sidecar was exported); nothing written -- the
                                        #: next window's shard judges the group against today's head. Never a refusal of the adapter
    retirement: Optional[Dict[str, object]] = None   #: with ``refused_known`` on a CLUSTER shard's RETIREMENT (:func:`retire_vanished_group`): the shard's
                                                     #: verified claim the pull's re-activation by id is gated on (:func:`retirement_claim`): the head it
                                                     #: compared against, every vanished file input (each declared retired) and the patterns that covered them

    def __post_init__(self) -> None:
        if not self.group_identity:
            self.group_identity = comparison_group_identity(self.group)


@dataclass
class PassReport:
    adapters: int = 0
    inputs_seen: int = 0
    inputs_changed: int = 0
    groups_refreshed: int = 0
    groups_refused: int = 0
    groups_reread: int = 0              #: of groups_refreshed, the committed revisions read again (``--reread``)
    groups_held: int = 0                #: published revisions the harvester holds or has not folded yet, left alone by the incremental pass
    activations_pending: int = 0        #: of the groups left alone, those whose marker (the directory's or a complete bundle's) is on the record log
                                        #: and not folded yet (:func:`pending_fold`); a cluster shard counts the ids the sidecar lists unfolded
    reactivations_deferred: int = 0     #: re-activations NOT written because an adapter's per-pass cap was spent (:class:`ReactivationBudget`)
    groups_forced: int = 0              #: of groups_refreshed, the ``--reread --force`` reads published as new revisions
    groups_emitted_nothing: int = 0     #: of groups_refreshed, the reads of store inputs that emitted no record and recorded no reason (filed ok = 0)
    groups_inputs_dropped: int = 0      #: of groups_refused, the reads refused for dropping too many of the previous revision's inputs undeclared
    groups_refused_known: int = 0       #: CLUSTER shard only: groups refused because their minted revision id is known to the grading host (nothing written)
    groups_seen: int = 0                #: comparison groups the pass planned over: the present groups ∪ the prior-only (vanished) groups
    groups_not_owned: int = 0           #: SHARDED pass only: groups of that union another shard owns (left alone here)
    inputs_hashed: int = 0              #: inputs whose bytes this pass read to hash (a cache miss, a changed identity, or ``--full``)
    inputs_hash_reused: int = 0         #: inputs whose sha the pass's :class:`base.InputHashCache` supplied without a read
    anchor_decisions_changed: int = 0   #: groups whose anchor decision set moved against their head's recorded one: the
                                        #: synthetic decision input read as changed, with or without a file change beside it
    groups_anchor_unavailable: int = 0  #: of groups_refused, the groups the pass PRESERVED or refused because its anchor currency could not be
                                        #: loaded (``GroupResult.anchor_unavailable``): nothing written for them, the heads stand
    groups_standing_refusals: int = 0   #: groups a SHARD re-read under ``retry_refused`` and refused again BYTE FOR BYTE under the same inputs and the
                                        #: same reading tree (:func:`standing_refusal_of`): a ``standing_refusal`` result, recorded like any
                                        #: refusal, never counted in ``groups_refused`` -- a standing refusal is not new information
    records_written: int = 0
    withdrawals: int = 0
    results: List[GroupResult] = field(default_factory=list)
    refused_standing: Dict[str, List[str]] = field(default_factory=dict)    #: adapter -> the groups whose last read REFUSED and which this pass
                                                                             #: left alone (:func:`plan_groups`); the refusal stays visible

    @property
    def groups_refused_standing(self) -> int:
        return sum(len(groups) for groups in self.refused_standing.values())

    def as_dict(self) -> Dict[str, object]:
        """The parseable summary line's fields (``adapters run`` prints them; a shard summary carries them under ``pass``)."""
        return {"adapters": self.adapters, "inputs_seen": self.inputs_seen, "inputs_changed": self.inputs_changed,
                "inputs_hashed": self.inputs_hashed, "inputs_hash_reused": self.inputs_hash_reused,
                "groups_seen": self.groups_seen, "groups_not_owned": self.groups_not_owned,
                "groups_refreshed": self.groups_refreshed, "groups_reread": self.groups_reread, "groups_forced": self.groups_forced,
                "groups_held": self.groups_held, "groups_refused": self.groups_refused, "groups_refused_known": self.groups_refused_known,
                "groups_refused_standing": self.groups_refused_standing,
                "activations_pending": self.activations_pending, "reactivations_deferred": self.reactivations_deferred,
                "groups_emitted_nothing": self.groups_emitted_nothing, "groups_inputs_dropped": self.groups_inputs_dropped,
                "anchor_decisions_changed": self.anchor_decisions_changed, "groups_anchor_unavailable": self.groups_anchor_unavailable,
                "groups_standing_refusals": self.groups_standing_refusals,
                "records_written": self.records_written, "withdrawals": self.withdrawals}


def source_id_for(adapter_name: str, group: str) -> str:
    """``<adapter>:<group>`` with the group's CANONICAL identity (:func:`base.comparison_group_identity`): a path-valued group read under
    a relocated root mints the source id the grading host minted; with no map installed this is ``f"{adapter_name}:{group}"``."""
    return f"{adapter_name}:{comparison_group_identity(group)}"


def group_of_source_id(adapter_name: str, source_id: str) -> str:
    return source_id[len(adapter_name) + 1:]


def prior_group_spelling(adapter, group_identity: str, imap: Optional[IdentityMap] = None) -> str:
    """The spelling the adapter ADDRESSES a prior-only comparison group under -- the key :attr:`GroupPlan.groups` would carry if an input
    of it were in discovery -- from the group's CANONICAL identity (what a watermark's source id names). Discovery spells every
    ``Input.uri``, and so every group, under the adapter's roots AS THEY ARE SPELLED, and two layouts put a map in a pass: under
    the HOME TREE a cluster shard lays out (``cluster_reads``, THE LAYOUT ON THE NODE: the canonical spelling itself resolves on the node,
    a symlink onto the pinned generation; the map carries each generation onto its canonical base) the roots hang off the canonical
    home and the identity IS the adapter's spelling; under a RE-POINTED ROOT (an adapter constructed over a local copy that is the
    map's local side -- the Phase 3 identity tests' shape) it is the map's local spelling. So: of the identity and ``imap.local`` of it,
    the one lying under a declared root (``adapter.roots``, each in its own spelling); the identity when both do, or neither does (the
    home tree is production; a group the map does not touch has one spelling anyway). Before this rule ``plan_groups`` spelled every
    prior-only group through ``imap.local``, which under the home tree is the GENERATION path -- a spelling no root and no
    ``inputs_retired`` declaration (hung off ``$HOME`` at import) knows: a vanished group's declared retirement was refused on the shard
    as undeclared (2026-10-08), and a partial drop keyed ``changed`` / ``gone_in_group`` by a key ``groups`` did not carry
    (a KeyError in the reading loop; on a ``reader_traversal`` adapter a drop lost silently). A PRESENT source never comes here: its
    key is the one ``groups`` already carries (:func:`plan_groups`)."""
    imap = identity_map() if imap is None else imap
    local = imap.local(group_identity)
    if local == group_identity:
        return group_identity
    roots = [str(r).rstrip("/") for r in (getattr(adapter, "roots", ()) or ()) if r]

    def under(path: str) -> bool:
        return any(path == r or path.startswith(r + "/") for r in roots)

    return local if under(local) and not under(group_identity) else group_identity


class PlanInvariantError(RuntimeError):
    """A planning rule produced a key the plan cannot serve (:func:`check_plan_invariants`): a planner DEFECT, named at the plan --
    adapter, rule, offending keys -- before any read, where the reading loop would otherwise meet a bare ``KeyError`` on
    ``plan.groups[g]`` (the prior-only group spelling defect surfaced exactly so)."""


def check_plan_invariants(adapter, plan: "GroupPlan") -> None:
    """The keying invariant every :class:`GroupPlan` must hold for :func:`_run_adapter` to index it: every key of ``changed`` and of
    ``gone_in_group`` is a key of ``groups`` (the reading loop reads ``plan.groups[g]`` for each), and no key of ``vanished`` is (a
    vanished group has NO input in discovery; one both present and retired would be read and retired in one pass). Raises
    :class:`PlanInvariantError` naming the adapter, the violated part and the offending keys, sorted; :func:`plan_groups` calls it last."""
    present = set(plan.groups)
    name = str(getattr(adapter, "name", adapter))
    faults: List[str] = []
    for part, keys in (("changed", set(plan.changed)), ("gone_in_group", set(plan.gone_in_group))):
        stray = sorted(keys - present)
        if stray:
            faults.append(f"{part} names {len(stray)} group(s) the plan's groups do not carry: " + ", ".join(stray))
    shadow = sorted(set(plan.vanished) & present)
    if shadow:
        faults.append(f"vanished names {len(shadow)} group(s) that are PRESENT in the plan's groups: " + ", ".join(shadow))
    if faults:
        raise PlanInvariantError(f"{name}: plan invariant violated (a planning rule produced a key the plan cannot serve); "
                                 + "; ".join(faults) + "; the plan's groups: " + (", ".join(sorted(present)) or "<none>"))


def revision_id_for(source_id: str, inputs: Sequence[Input], emission_version: Optional[object] = None,
                    forced_utc: Optional[str] = None) -> str:
    """sha16 over the SOURCE (adapter + comparison group), its inputs' URIs and shas — two sources reading byte-identical
    files are two revisions — and the adapter's EMISSION VERSION (:func:`base.emission_version_of`) when it is
    not the default: what the adapter's code parks or spells is part of what a revision IS, so a bumped version is a new
    revision over unchanged inputs. The default version (``1``, every revision published before the rule) folds NOTHING,
    so an adapter that never bumped reproduces its published ids. ``forced_utc`` (``--reread --force``) folds the pass
    stamp in as a nonce: a read nobody can reproduce, published as a new revision whatever the inputs. The inputs are keyed by
    ``Input.identity_uri`` (the canonical spelling), never by where the read opened them."""
    key: Dict[str, object] = {"source_id": source_id, "inputs": inputs_json(inputs)}
    version = str(DEFAULT_EMISSION_VERSION) if emission_version is None else str(emission_version)
    if version != str(DEFAULT_EMISSION_VERSION):
        key["emission_version"] = version
    if forced_utc:
        key["forced_utc"] = str(forced_utc)
    return identity.sha16(identity.canonical_json(key))


def inputs_json(inputs: Sequence[Input]) -> Dict[str, str]:
    """``{identity_uri: sha256}`` -- the marker's ``inputs_json`` and the revision id's input key, in the canonical spelling."""
    return {i.identity_uri: i.sha256 for i in inputs}


def _revision_for(adapter: Adapter, group: str, inputs: Sequence[Input], forced_utc: Optional[str] = None) -> Tuple[str, str]:
    """``(source_id, revision_id)`` of a group as THIS adapter's code would publish it."""
    source_id = source_id_for(adapter.name, group)
    return source_id, revision_id_for(source_id, inputs, emission_version_of(adapter), forced_utc)


VANISHED_PREFIX = "input vanished: "


@dataclass(frozen=True)
class Watermark:
    source_id: str
    stat: Tuple[str, int, str]          # (sha256, size, mtime_utc)
    last_error: Optional[str]
    attempt_revision_id: Optional[str] = None   #: on a refusal stamp, the revision id the refused read computed over the inputs it found
                                                #: (``extra_json.attempt_revision_id``): the exact inventory of the latest attempt (:func:`plan_groups`)
    attempt_code_sha: Optional[str] = None      #: on a refusal stamp, the code sha of the tree that refused (``extra_json.attempt_code_sha``; the
                                                #: producer's ``code_sha``): the third term of a standing refusal (:func:`standing_refusal_of`);
                                                #: None on a stamp written before the field, which therefore never stands
    attempt_window: Optional[str] = None        #: on a refusal stamp written by a cluster shard, the window / operation id that read (``extra_json.
                                                #: attempt_window``, the plan's window); None for a grading host read or a stamp before the field
    observed_utc: Optional[str] = None          #: the stamp's time (``source_watermark.observed_utc``)

    @property
    def processed(self) -> bool:
        """A watermark records the LAST READ; only a read that succeeded counts as processed."""
        return self.last_error is None

    @property
    def retired(self) -> bool:
        """The input left discovery and its withdrawal was emitted: no longer part of the inventory."""
        return bool(self.last_error) and str(self.last_error).startswith(VANISHED_PREFIX)

    @property
    def refused(self) -> bool:
        """The last read of the input REFUSED (a ``last_error`` that is not a retirement): the refusal stands until an explicit
        cause reads the group again (:func:`plan_groups`)."""
        return bool(self.last_error) and not self.retired


_WATERMARK_SQL = ("SELECT input_uri, sha256, size, mtime_utc, source_id, last_error, json_extract(extra_json, '$.attempt_revision_id'), "
                  "adapter, json_extract(extra_json, '$.attempt_code_sha'), json_extract(extra_json, '$.attempt_window'), observed_utc FROM source_watermark")


def _watermark_of_row(r) -> Watermark:
    return Watermark(str(r[4]), (r[1], int(r[2]), r[3]), r[5], None if r[6] is None else str(r[6]), None if r[8] is None else str(r[8]),
                     None if r[9] is None else str(r[9]), None if r[10] is None else str(r[10]))


def _watermarks(conn: Optional[sqlite3.Connection], adapter_name: str) -> Dict[str, Watermark]:
    """One adapter's watermarks by input URI."""
    if conn is None:
        return {}
    return {r[0]: _watermark_of_row(r) for r in conn.execute(_WATERMARK_SQL + " WHERE adapter = ?", (adapter_name,)).fetchall()}


def _watermarks_by_adapter(conn: Optional[sqlite3.Connection]) -> Dict[str, Dict[str, Watermark]]:
    """Every adapter's watermarks in ONE scan of ``source_watermark`` — the table has no index on ``adapter``, so a query per
    adapter scanned its ~27k rows once per adapter, 48 times a pass."""
    out: Dict[str, Dict[str, Watermark]] = {}
    if conn is None:
        return out
    for r in conn.execute(_WATERMARK_SQL).fetchall():
        out.setdefault(str(r[7]), {})[r[0]] = _watermark_of_row(r)
    return out


def _revision_state(conn: Optional[sqlite3.Connection], revision_id: str) -> Optional[str]:
    """``accepted`` (head or superseded), ``pending`` (a marker with accepted = 0: held on a refused or blocked constituent,
    or a refused read), or None when the database has no row for it (not ingested yet, or no database)."""
    if conn is None:
        return None
    row = conn.execute("SELECT accepted FROM source_revision WHERE revision_id = ?", (revision_id,)).fetchone()
    if row is None:
        return None
    return "accepted" if int(row[0]) == 1 else "pending"


def _revision_accepted(conn: Optional[sqlite3.Connection], revision_id: str) -> bool:
    return _revision_state(conn, revision_id) == "accepted"


def _held_marker(conn: Optional[sqlite3.Connection], revision_id: str) -> Optional[Tuple[str, str, str]]:
    """The harvester's queue row for a HELD marker of ``revision_id`` -- ``(record_uri, kind, reason)`` of the latest one -- or None.
    A pending revision's own directory and every re-activation bundle of an ACCEPTED, non-head revision hold this way (a
    bundle that would move the head while dropping run ids it carries sits in ``blocked_records`` with ``kind = run_drop``; the
    revision's row stays ``accepted = 1``, superseded, and says nothing about the hold), so the pass asks the queue, not the row."""
    if conn is None:
        return None
    row = conn.execute("SELECT record_uri, json_extract(extra_json, '$.kind'), json_extract(extra_json, '$.blocked_on') FROM blocked_records "
                       "WHERE record_type = 'source_revision' AND json_extract(extra_json, '$.revision_id') = ? "
                       "ORDER BY first_seen_utc DESC, record_uri DESC LIMIT 1", (revision_id,)).fetchone()
    if row is None:
        return None
    return str(row[0]), str(row[1] or ""), str(row[2] or "")


def _revision_refusal(conn: Optional[sqlite3.Connection], revision_id: str) -> Optional[str]:
    """WHY the harvester holds a published revision: the ``refusal`` of its pending marker row (``activation pending: <table>
    <id> (<detail>)`` — the refused or blocked constituent by ``<file_uri>#<line>`` and the validator's or the CHECK's own words).
    None when the revision is accepted, unknown to the database, or there is no database."""
    if conn is None:
        return None
    row = conn.execute("SELECT refusal FROM source_revision WHERE revision_id = ? AND accepted = 0", (revision_id,)).fetchone()
    return str(row[0]) if row is not None and row[0] else None


#: the label the harvester writes into ``source_revision.refusal`` for a PUBLISHED revision it holds (``harvest.Ingestor._placeholder_revision``:
#: ``activation pending: <table> <id> (<detail>)``) -- the one shape of an ``accepted = 0`` row that is NOT the reader's own refusal
HOLD_LABEL = "activation pending:"


def _emit_time_refusal(conn: Optional[sqlite3.Connection], revision_id: str) -> Optional[Tuple[str, str]]:
    """``(source_id, refusal)`` when the database's row for ``revision_id`` was folded from a REFUSED record -- the read REFUSED AT EMIT TIME
    (:func:`_read_and_publish` wrote ``_refused/<rev>/<stamp>-source_revision.jsonl`` with ``accepted = 0`` and the reader's words in
    ``refusal``; the harvester folded it as the row, its ``record_uri`` under ``_refused/``) -- and, as far as the ROW says, nothing was ever
    committed under the id. None for an accepted row; for a row the harvester HOLDS (a published revision: ``refusal`` carries
    :data:`HOLD_LABEL`, whatever its ``record_uri`` -- a directory published after an emit-time refusal and then held keeps the refused
    record's uri under the hold's label: nine live rows on 2026-10-07); for a placeholder with no refusal; for no row; for no database.
    The callers pair the row with the BYTES before acting: :meth:`ClusterRead.known` with the sidecar's ``published_dirs`` kinds,
    :func:`reactivate_by_id` with the revision directory and the bundles."""
    if conn is None:
        return None
    row = conn.execute("SELECT accepted, refusal, record_uri, source_id FROM source_revision WHERE revision_id = ?", (revision_id,)).fetchone()
    if row is None or int(row[0]) == 1:
        return None
    refusal = str(row[1] or "")
    if not refusal or refusal.startswith(HOLD_LABEL) or "/_refused/" not in str(row[2] or ""):
        return None
    return str(row[3]), refusal


def _current_revision(conn: Optional[sqlite3.Connection], source_id: str) -> Optional[str]:
    if conn is None:
        return None
    rev = conn.execute("SELECT revision_id FROM source_revision WHERE source_id = ? AND accepted = 1 "
                       "ORDER BY observed_utc DESC, revision_id DESC LIMIT 1", (source_id,)).fetchone()
    return str(rev[0]) if rev else None


def _head_versions(conn: Optional[sqlite3.Connection], source_ids: Sequence[str]) -> Dict[str, str]:
    """The ``EMISSION_VERSION`` each source's accepted head was minted under, read off its marker (``extra_json.emission_version``;
    a head published before the rule recorded none and was minted under the default); a source with no accepted head is
    absent. A FACT the marker carries, deliberately not the head's id compared with a reconstruction: a ``--reread --force`` head
    carries a nonce no reconstruction reproduces, and comparing ids would have the next incremental pass re-activate the revision
    the forced read replaced. One query per adapter over ``idx_source_revision_source`` — it was one per group, ~1,500 a pass at
    ~5 ms each on the live WAL; the head is the row latest by ``(observed_utc, revision_id)``, the per-group query's order."""
    heads: Dict[str, Tuple[str, str, Optional[object]]] = {}
    if conn is None or not source_ids:
        return {}
    ids = list(dict.fromkeys(source_ids))
    for i in range(0, len(ids), 500):
        chunk = ids[i:i + 500]
        rows = conn.execute("SELECT source_id, observed_utc, revision_id, json_extract(extra_json, '$.emission_version') FROM source_revision "
                            f"WHERE accepted = 1 AND source_id IN ({','.join('?' * len(chunk))})", chunk).fetchall()
        for r in rows:
            sid, key = str(r[0]), (str(r[1]), str(r[2]))
            if sid not in heads or key > heads[sid][:2]:
                heads[sid] = (key[0], key[1], r[3])
    return {sid: (str(ver) if ver is not None else str(DEFAULT_EMISSION_VERSION)) for sid, (_, _, ver) in heads.items()}


def _revision_has_status(conn: Optional[sqlite3.Connection], source_id: str, revision_id: str) -> bool:
    """Whether a ``source_status`` row of ``source_id`` carries this revision. With no database every published revision
    is taken to have one: the incremental pass cannot tell, and re-reading everything it re-activates would be the wrong default."""
    if conn is None:
        return True
    row = conn.execute("SELECT 1 FROM source_status WHERE source_id = ? AND source_revision_id = ? LIMIT 1", (source_id, revision_id)).fetchone()
    return row is not None


def _accepted_inputs(conn: Optional[sqlite3.Connection], revision_id: Optional[str]) -> Optional[Dict[str, str]]:
    """The ``inputs_json`` (uri -> sha) an ACCEPTED revision's marker recorded; None when there is no such revision or no database."""
    if conn is None or not revision_id:
        return None
    row = conn.execute("SELECT inputs_json FROM source_revision WHERE revision_id = ? AND accepted = 1", (revision_id,)).fetchone()
    if row is None or not row[0]:
        return None
    try:
        data = json.loads(row[0])
    except ValueError:
        return None
    return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else None


def _head_decisions(conn: Optional[sqlite3.Connection], source_ids: Sequence[str]) -> Dict[str, Dict[str, object]]:
    """The anchor decision set each source's ACCEPTED HEAD recorded (``extra_json.anchor_decisions``) -- the block with at least
    one comparison, by source id; a source with no accepted head, or whose head recorded no set (a read before this code, an adapter
    that files no contrast), is absent. One query per adapter over ``idx_source_revision_source``, the head being the row latest by
    ``(observed_utc, revision_id)`` exactly as :func:`_head_versions` reads it; a ``source_revision`` table without the column's
    JSON (a block that is not an object) reads as no set."""
    heads: Dict[str, Tuple[str, str, Optional[str]]] = {}
    if conn is None or not source_ids:
        return {}
    ids = list(dict.fromkeys(source_ids))
    for i in range(0, len(ids), 500):
        chunk = ids[i:i + 500]
        rows = conn.execute("SELECT source_id, observed_utc, revision_id, json_extract(extra_json, '$.anchor_decisions') FROM source_revision "
                            f"WHERE accepted = 1 AND source_id IN ({','.join('?' * len(chunk))})", chunk).fetchall()
        for r in rows:
            sid, key = str(r[0]), (str(r[1]), str(r[2]))
            if sid not in heads or key > heads[sid][:2]:
                heads[sid] = (key[0], key[1], r[3])
    out: Dict[str, Dict[str, object]] = {}
    for sid, (_, _, raw) in heads.items():
        if not raw:
            continue
        try:
            block = json.loads(raw) if isinstance(raw, str) else raw
        except ValueError:
            continue
        if isinstance(block, dict) and isinstance(block.get("comparisons"), list) and block["comparisons"]:
            out[sid] = block
    return out


def recorded_decision_groups(adapter: Adapter, groups: Iterable[str], conn: Optional[sqlite3.Connection]) -> Dict[str, Dict[str, object]]:
    """Of the discovered ``groups`` (the adapter's local spelling), those whose accepted head recorded an anchor decision set
    (:func:`_head_decisions`), each with its recorded block: the groups that carry a synthetic decision input this pass -- and, when the
    frame cannot be loaded, the groups the pass preserves (:attr:`GroupPlan.preserved`)."""
    by_source = {source_id_for(adapter.name, g): g for g in groups}
    return {by_source[sid]: block for sid, block in _head_decisions(conn, list(by_source)).items()}


def anchor_decision_inputs(adapter: Adapter, groups: Iterable[str], conn: Optional[sqlite3.Connection], ctx: AdapterContext) -> Dict[str, Input]:
    """The synthetic anchor-decision input of every discovered group whose accepted head recorded a decision set: the
    head's comparisons replayed against THIS pass's frame (``ctx.anchor_currency()``, loaded once per pass and only when some head
    recorded a set) and fingerprinted -- equal to the head's recorded fingerprint, the watermark is unchanged and the group quiet;
    different (a ruling retired an anchor, a hold-out lifted, an own unsteered run restored, a candidate's judgment superseded, a parked
    comparison's anchor now current, the anchor's reading corrected in place), :func:`plan_groups` marks exactly that group changed. Keyed
    by the group in the adapter's local spelling. Raises :class:`anchor_current.AnchorCurrencyUnavailable` when a head recorded a set and
    the frame cannot be loaded (the caller preserves those groups rather than planning on a frame it does not have)."""
    recorded = recorded_decision_groups(adapter, groups, conn)
    if not recorded:
        return {}
    frame = ctx.anchor_currency()
    out: Dict[str, Input] = {}
    for g, block in recorded.items():
        sid = source_id_for(adapter.name, g)
        comparisons = [e.get("comparison") for e in block["comparisons"] if isinstance(e, dict) and isinstance(e.get("comparison"), dict)]
        out[g] = AnchorDecisionInput.of(sid, decision_fingerprint(frame.decisions_of(comparisons)))
    return out


def staging_dir(root: str, adapter_name: str, revision_id: str) -> str:
    """Where a revision is assembled: under the registry root but OUTSIDE records/ and rulings/, so the harvester
    (which scans only those two) can never ingest a half-written revision."""
    return os.path.join(root, "staging", "adapters", adapter_name, revision_id)


def activations_dir(root: str, adapter_name: str, revision_id: str) -> str:
    """Records about an ALREADY committed revision: a touched input with identical bytes re-stamps its
    watermarks as a plain file here; a reverted input RE-ACTIVATES the revision through a bundle directory
    ``<here>/<stamp>/`` that the harvester applies like a revision directory — atomically, marker last."""
    return os.path.join(H.records_dir(root), "adapters", adapter_name, "_activations", revision_id)


#: The per-adapter, per-pass cap on activation BUNDLES the incremental pass writes (:class:`ReactivationBudget`).
MAX_REACTIVATIONS_PER_PASS = 20


@dataclass
class ReactivationBudget:
    """The per-adapter, per-pass cap on activation BUNDLES -- a re-activation of committed bytes, a re-read's bundle -- a backstop so
    that no runner defect can flood the record log faster than the harvester folds it (2026-09-29: 97 groups x ~18k records per
    10-minute pass for 17 passes, every cycle of the harvester killed at its 45-minute bound). The cap counts bundles WRITTEN: a
    plain watermark re-stamp (a touched input of the head) and a fresh revision directory (new inputs, a bumped ``EMISSION_VERSION``)
    are not re-activations and never count. An explicit ``--reread`` is the operator's and runs uncapped (the 2026-09-17 status
    backfill re-read 24 adapters' heads in one pass). A group the spent budget turns away is DEFERRED -- nothing written, the result
    says so (``GroupResult.deferred``), the pass counts it (``reactivations_deferred``) -- and the next pass takes it."""
    limit: int = field(default_factory=lambda: MAX_REACTIVATIONS_PER_PASS)      # read at instantiation: a test lowers the module constant
    taken: int = 0

    @property
    def exhausted(self) -> bool:
        return self.taken >= self.limit

    def take(self) -> bool:
        """Spend one bundle of the budget; False (and nothing spent) when it is exhausted."""
        if self.exhausted:
            return False
        self.taken += 1
        return True


def _is_marker_file(name: str) -> bool:
    """The harvester's marker convention (``harvest.Ingestor._find_marker``): the ``<seq>-source_revision.jsonl[.gz]`` file of a revision
    directory or a bundle -- the one holding the ``source_revision`` record -- is the marker, and it is what the harvester folds LAST."""
    return "source_revision" in name and (name.endswith(".jsonl") or name.endswith(".jsonl.gz"))


def complete_bundles(root: str, adapter_name: str, revision_id: str) -> List[Tuple[str, str]]:
    """``[(stamp, marker_path)]`` of every COMPLETE activation bundle of ``revision_id`` on the record log, oldest first. A bundle is the
    directory ``activations_dir(...)/<stamp>/``; it is complete when it holds its marker file (:func:`_is_marker_file`). :func:`publish_staged`
    lands a bundle by ONE directory rename from ``staging/``, so a stamp directory with no marker is a crash between staging and publication
    (or a concurrent publisher mid-rename) and is not on the log yet. The plain ``<stamp>-source_watermark.jsonl`` files beside the bundles
    (a touched input's re-stamp of the head) are not bundles."""
    base = activations_dir(root, adapter_name, revision_id)
    if not os.path.isdir(base):
        return []
    out: List[Tuple[str, str]] = []
    for stamp in sorted(os.listdir(base)):
        d = os.path.join(base, stamp)
        if not os.path.isdir(d):
            continue
        marker = next((os.path.join(d, f) for f in sorted(os.listdir(d)) if _is_marker_file(f)), None)
        if marker is not None:
            out.append((stamp, marker))
    return out


def _marker_folded(conn: sqlite3.Connection, root: str, marker_path: str) -> bool:
    """Whether the harvester has processed ``marker_path``: a ``record_delivery`` row for its logical uri (``ok`` when the revision or
    bundle activated, ``blocked`` when it was held -- the hold's reason then sits on the marker's ``blocked_records`` row)."""
    return conn.execute("SELECT 1 FROM record_delivery WHERE file_uri = ? LIMIT 1", (H.logical_uri(root, marker_path),)).fetchone() is not None


def pending_fold(root: str, adapter_name: str, revision_id: str, conn: Optional[sqlite3.Connection]) -> Optional[str]:
    """The record-log uri of a marker of ``revision_id`` the harvester has NOT folded yet -- the revision directory's own, or a complete
    activation bundle's (:func:`complete_bundles`; the newest first, the one still unfolded in a spiral) -- or None when every marker on the
    log has been folded. FOLDED means the database holds a ``record_delivery`` row for the marker file: the harvester writes one when
    the revision activates (``ok``) and when it holds it (``blocked``), and none before -- a cycle killed mid-staging has committed its
    staging windows and its ``prepare`` placeholder row (``accepted = 0``, no refusal) but never the marker's delivery, so a partially
    staged directory reads as PENDING here, never as held or absent. This is the rule that ends the 2026-09-29 spiral: a pass that
    re-emitted a revision because the database lacked its rows was re-emitting a fold that had not happened yet, once per pass, faster
    than the harvester could fold any of it. None with no database (the pass cannot tell) or with a connection that has no delivery
    ledger (a cluster shard's reader-state sidecar: :meth:`ClusterRead.known` answers there from ``published_dirs``)."""
    if conn is None:
        return None
    rev_dir = os.path.join(H.records_dir(root), "adapters", adapter_name, revision_id)
    candidates: List[str] = []
    if os.path.isdir(rev_dir):
        candidates += [os.path.join(rev_dir, f) for f in sorted(os.listdir(rev_dir)) if _is_marker_file(f)]
    candidates += [marker for _, marker in complete_bundles(root, adapter_name, revision_id)]
    try:
        for marker in reversed(candidates):
            if not _marker_folded(conn, root, marker):
                return H.logical_uri(root, marker)
    except sqlite3.OperationalError:
        return None                # no record_delivery table: not the registry's database
    return None


_REEMIT_SKIP = ("artifact", "source_revision", "observation", "withdrawal", "source_state", "source_watermark")
REEMIT_WINDOW = 10_000      # records held while re-emitting a committed file: streamed window by window, never the file


def adapter_priority(adapter, group: Optional[str] = None) -> int:
    """The adapter's declared source priority (0 when undeclared): among several accepted sources measuring one
    key the higher priority is the current row, ties by lexical source_id. An adapter whose groups rank differently declares
    ``priority_for(group)`` (the degeneracy union reader ranks its run-id-keyed batches below its campaign files); ``priority``
    stays the adapter-wide value the ownership check and a group-less caller read."""
    fn = getattr(adapter, "priority_for", None)
    if group is not None and callable(fn):
        try:
            return int(fn(group) or 0)
        except (TypeError, ValueError):
            return 0
    try:
        return int(getattr(adapter, "priority", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _reemit_committed(out_dir: str, staged: str, now: str, rev: str, skip: Sequence[str] = _REEMIT_SKIP) -> Tuple[int, int]:
    """Copy every MEASUREMENT record of a published revision (runs, states, judgments, metrics, contrasts, item rows —
    not the inputs' artifacts nor the marker's own records) into ``staged`` with a fresh ``written_utc``: folded again
    they carry a later order key, so the re-activated revision's values win over the outgoing revision's.
    Streamed: at most ``REEMIT_WINDOW`` records are held at a time, each window appended to the staged file. Every
    re-emitted record passes through :func:`sanitise_free_form`: a revision whose committed bytes carry a non-finite float in a
    free-form field (published before that guard, held by the validator) folds when re-activated; a typed NaN still refuses.
    Returns (files, records) re-emitted."""
    n_files = n_records = 0
    for f in sorted(os.listdir(out_dir)):
        window: List[Dict[str, object]] = []
        written = 0
        dest = os.path.join(staged, f)
        for _, rec, _ in R.read_record_file(os.path.join(out_dir, f)):
            if rec.get("record_type") in skip:
                continue
            rec = dict(rec, written_utc=now, payload=sanitise_free_form(str(rec.get("record_type")), rec.get("payload") or {}))
            if rec.get("record_type") in R.ITEM_SCALE_RECORD_TYPES:
                rec["payload"] = dict(rec["payload"], source_revision_id=rev)       # stamped
            window.append(rec)
            if len(window) >= REEMIT_WINDOW:
                R.write_records(dest, window, gz=f.endswith(".gz"))
                written += len(window)
                window = []
        if window:
            R.write_records(dest, window, gz=f.endswith(".gz"))
            written += len(window)
        if not written:
            continue
        n_files += 1
        n_records += written
    return n_files, n_records


def refused_dir(root: str, adapter_name: str, revision_id: str) -> str:
    """Unaccepted revision records live beside, never inside, the revision directory a later successful read of the
    same content will publish atomically."""
    return os.path.join(H.records_dir(root), "adapters", adapter_name, "_refused", revision_id)


def _dir_shas(d: str) -> Dict[str, str]:
    return {f: R.file_logical_sha(os.path.join(d, f)) for f in sorted(os.listdir(d))}


def publish_staged(staged: str, out_dir: str) -> List[str]:
    """Publish a staged revision as ONE directory rename (atomic on one filesystem): the harvester sees the whole
    revision — batches and the ``source_revision`` marker — or nothing. A destination that already holds
    the same content (a re-run of a deterministic revision) is left alone; different content under one revision
    id is a bug and refuses loudly.

    The adapter-wide parent of ``staged`` (``<root>/staging/adapters/<adapter>/``) is left STANDING, empty, for the life
    of the root. Concurrent cluster-read shards of one adapter share the window root, so they share that parent,
    and the tidy-up this function used to run after the rename -- ``if isdir(parent) and not listdir(parent): rmdir(parent)``
    -- was a non-atomic check-then-act on a directory a peer is using: one shard's rmdir landed between another's check
    and act, and the OSError, which the pass does not catch (``_read_and_publish`` catches AdapterRefusal only), killed
    the shard before it wrote its summary (two objects_bank shards; their revisions were lost).
    The same race has two more faces: ENOTEMPTY when a peer stages its revision between the listing and the rmdir, and
    a peer's ``os.makedirs`` of ``<parent>/<rev>`` hitting ENOENT under a parent removed beneath it. A tolerant rmdir, or
    a tidy at pass end, would still remove the parent under a peer, so there is none. The empty directory is harmless:
    ``staging/`` is outside ``records/`` and ``rulings/``, which is all the harvester scans (:func:`staging_dir`), and
    nothing lists ``staging/adapters`` (``tests/test_registry_publish_staged_shared_parent.py``)."""
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    if os.path.isdir(out_dir):
        if _dir_shas(out_dir) != _dir_shas(staged):
            raise RuntimeError(f"revision {os.path.basename(out_dir)} already published with different content: {out_dir}")
        shutil.rmtree(staged, ignore_errors=True)
    else:
        os.replace(staged, out_dir)
    return [os.path.join(out_dir, f) for f in sorted(os.listdir(out_dir))]


def committed_anchor_decisions(out_dir: str) -> Optional[Dict[str, object]]:
    """The ``extra.anchor_decisions`` block the PUBLISHED revision's own marker carries, read from its committed bytes -- what a
    re-activation's bundle marker restates, so the re-activated head keeps the decision set its records were built from (a bundle marker
    without it would read as a head that decided nothing, and the next pass would drop the group's decision input). None when the
    directory's marker recorded no set."""
    for f in sorted(os.listdir(out_dir)):
        if "source_revision" not in f:
            continue
        for _, rec, _ in R.read_record_file(os.path.join(out_dir, f), compute_sha=False):
            if rec.get("record_type") == "source_revision":
                block = ((rec.get("payload") or {}).get("extra") or {}).get("anchor_decisions")
                return dict(block) if isinstance(block, dict) and block.get("comparisons") else None
    return None


def _owned_state_key(payload: Mapping[str, object], source_id: str, owned_uris: Set[str], owned_paths: Sequence[str]) -> bool:
    """THE STATE-KEY OWNERSHIP RULE (module docstring): a state row is the GROUP's when its key is one of ``owned_uris`` (identity URIs of
    the revision's inputs -- for a displaced-key check, the outgoing head's too), the group's ledger URI, or a ``json_path`` the adapter
    declared (:func:`base.state_owned_paths_of`); anything else -- a directory listing, an absent file, a ruling-spelled path, a root-wide
    key every group restates -- is auxiliary or shared, observed at read time, never a re-activation's to replay or clear."""
    uri, jpath = str(payload.get("input_uri")), str(payload.get("json_path"))
    return uri == row_refusals_uri(source_id) or uri in owned_uris or jpath in owned_paths


def committed_state_records(out_dir: str, ctx: AdapterContext, source_id: Optional[str] = None, owned_uris: Optional[Set[str]] = None,
                            owned_paths: Sequence[str] = ()) -> List[Dict[str, object]]:
    """The ``source_state`` records the PUBLISHED revision's own marker file carries (the read that produced it wrote them beside the
    marker), re-stamped ``observed_utc = ctx.now`` under fresh ids -- what a re-activation's bundle restates as the revision's state
    (the module docstring's state-row rule). Read from the committed bytes, never asked of the adapter: the adapter's ledger is the
    LAST read's, which may be a later revision of this very group (an unrecognized row directory added, then removed: the earlier
    revision re-activates while the adapter's list still names the directory that is gone). With ``owned_uris`` (the committed
    revision's input identity URIs) only the GROUP-OWNED rows are returned (:func:`_owned_state_key`): a committed snapshot of an
    auxiliary or shared key -- a directory listing, ``gsm8k_p200``'s root-wide ``$.dirs`` -- is never re-stamped over the latest fresh
    read's row. Empty for a revision published before state rows were written beside the marker, or by an adapter that files none."""
    out: List[Dict[str, object]] = []
    for f in sorted(os.listdir(out_dir)):
        if "source_revision" not in f:
            continue
        for _, rec, _ in R.read_record_file(os.path.join(out_dir, f), compute_sha=False):
            if rec.get("record_type") != "source_state":
                continue
            committed = rec.get("payload") or {}
            if owned_uris is not None and not _owned_state_key(committed, source_id or "", owned_uris, owned_paths):
                continue
            payload = dict(committed, observed_utc=ctx.now)
            out.append(R.make_record("source_state", payload, ctx.producer, identity.adapter_record_id(payload, "source_state"), ctx.now))
    return out


def committed_keys(out_dir: str) -> Tuple[Set[str], Set[str]]:
    """The run-scale observation keys and the item keys a PUBLISHED revision carries, read from its committed bytes."""
    keys: Set[str] = set()
    items: Set[str] = set()
    pk = R.ITEM_MEMBERSHIP_KEY
    for f in sorted(os.listdir(out_dir)):
        for _, rec, _ in R.read_record_file(os.path.join(out_dir, f)):
            rt = rec.get("record_type")
            payload = rec.get("payload", {})
            if rt == "observation":
                keys.update(str(k) for k in payload.get("observation_keys", []))
            elif rt == "item_result":
                items.add(identity.canonical_json({c: payload.get(c, _sentinel(c)) for c in pk}))
    return keys, items


def _sentinel(col: str):
    from .. import schema_def as S
    return S.SENTINELS.get(col)


def _current_observations(conn: Optional[sqlite3.Connection], source_id: str) -> Dict[str, Tuple[str, str]]:
    """{observation_key_json: (table_name, revision_id)} of the latest accepted revision's current keys."""
    if conn is None:
        return {}
    rev = conn.execute("SELECT revision_id FROM source_revision WHERE source_id = ? AND accepted = 1 "
                       "ORDER BY observed_utc DESC, revision_id DESC LIMIT 1", (source_id,)).fetchone()
    if rev is None:
        return {}
    rows = conn.execute("SELECT observation_key, table_name FROM observation WHERE source_id = ? AND revision_id = ? AND current = 1",
                        (source_id, rev[0])).fetchall()
    return {r[0]: (r[1], rev[0]) for r in rows}


def _current_item_keys(conn: Optional[sqlite3.Connection], revision_id: str) -> Set[str]:
    if conn is None or not revision_id:
        return set()
    pk = R.ITEM_MEMBERSHIP_KEY
    # every row tagged with the revision is a key it carries (rows are keyed by revision; marks are bookkeeping)
    rows = conn.execute(f"SELECT {', '.join(pk)} FROM item_result WHERE source_revision_id = ?", (revision_id,)).fetchall()
    return {identity.canonical_json(dict(zip(pk, r))) for r in rows}


def _withdrawal_records(ctx: AdapterContext, source_id: str, rev: str, prev: Mapping[str, Tuple[str, str]], prev_rev: Optional[str],
                        prev_items: Set[str], now_keys: Set[str], now_items: Set[str], reason: str) -> List[Dict[str, object]]:
    """A ``withdrawal`` per previous current run-scale key absent now, plus ONE for the removed items: it carries their
    COUNT (``n_items_removed``), never the key list — the removed keys are the outgoing revision's rows the new one has
    no row for, which the harvester marks from the rows in bounded batches after the activation; a list of
    every removed key would make the activation's fold O(removed items) again."""
    now = ctx.now
    out = []
    for key_json, (table, from_rev) in prev.items():
        if key_json not in now_keys and from_rev != rev:
            payload = {"source_id": source_id, "observation_key": key_json, "from_revision_id": from_rev, "to_revision_id": rev,
                       "table_name": table, "reason": reason, "item_keys_json": "[]"}
            out.append(R.make_record("withdrawal", payload, ctx.producer, identity.adapter_record_id(payload, "withdrawal"), now))
    n_removed = len(prev_items - now_items) if prev_rev and prev_rev != rev else 0
    if n_removed:
        payload = {"source_id": source_id, "observation_key": json.dumps({"items_of_revision": prev_rev}, sort_keys=True),
                   "from_revision_id": prev_rev, "to_revision_id": rev, "table_name": "item_result",
                   "reason": reason if reason == "input_vanished" else "removed_items",
                   "item_keys_json": "[]", "n_items_removed": n_removed}
        out.append(R.make_record("withdrawal", payload, ctx.producer, identity.adapter_record_id(payload, "withdrawal"), now))
    return out


def _watermark_records(ctx: AdapterContext, adapter_name: str, source_id: str, inputs: Sequence[Input],
                       records_written: int = 0, last_error: Optional[str] = None, attempt_revision_id: Optional[str] = None,
                       attempt_window: Optional[str] = None) -> List[Dict[str, object]]:
    """One watermark per input read. A refusal stamp (``last_error``) also records ``attempt_revision_id`` under ``extra`` — the
    revision id the refused read computed over the inputs it found (their URIs and shas, the adapter's version): the exact
    inventory of the latest attempt, which :func:`plan_groups` compares against discovery in both directions — beside it the
    reading tree (``attempt_code_sha``, the producer's) and, for a cluster shard, the window that read (``attempt_window``): the
    terms a later re-read is compared with (:func:`standing_refusal_of`). An accepted stamp carries no extra and keeps its bytes."""
    now = ctx.now
    extra = ({"extra": {"attempt_revision_id": attempt_revision_id, "attempt_code_sha": ctx.producer.get("code_sha"), "attempt_window": attempt_window}}
             if attempt_revision_id else {})
    return [R.make_record("source_watermark",
                          {"source_id": source_id, "input_uri": i.identity_uri, "sha256": i.sha256, "size": i.size, "mtime_utc": i.mtime_utc,
                           "observed_utc": now, "adapter": adapter_name, "records_written": records_written, "last_error": last_error, **extra},
                          ctx.producer, identity.adapter_record_id({"s": source_id, "u": i.identity_uri, "sha": i.sha256, "t": now}, "source_watermark"), now)
            for i in inputs]


def _retire_watermarks(ctx: AdapterContext, conn: Optional[sqlite3.Connection], adapter_name: str, source_id: str,
                       uris: Sequence[str], rev: str) -> List[Dict[str, object]]:
    """Watermarks of inputs that left discovery, marked retired (``last_error = 'input vanished: …'``) so they leave the
    inventory once the withdrawal is emitted; a reappearing input reads as unprocessed and is re-read."""
    out = []
    if conn is None:
        return out
    for uri in sorted(uris):
        row = conn.execute("SELECT sha256, size, mtime_utc FROM source_watermark WHERE source_id = ? AND input_uri = ?",
                           (source_id, uri)).fetchone()
        if row is not None:
            payload = {"source_id": source_id, "input_uri": uri, "sha256": row[0], "size": int(row[1]), "mtime_utc": row[2],
                       "observed_utc": ctx.now, "adapter": adapter_name, "records_written": 0, "last_error": VANISHED_PREFIX + uri}
            out.append(R.make_record("source_watermark", payload, ctx.producer,
                                     identity.adapter_record_id({"s": source_id, "u": uri, "vanished": rev}, "source_watermark"), ctx.now))
    return out


def _source_state_records(ctx: AdapterContext, adapter: Adapter, inputs: Sequence[Input]) -> List[Dict[str, object]]:
    """The adapter's :class:`base.SourceStateRow` facts as ``source_state`` records, their ``source_id`` and ``input_uri`` in the
    canonical spelling (the adapter names them locally; a ``$.row_refusals`` ledger row's URI is rebuilt from the CANONICAL source id,
    :func:`base.row_refusals_uri`, since the group it embeds may be path-valued)."""
    out = []
    imap = identity_map()
    try:
        for st in adapter.source_states(ctx, inputs):
            sid = imap.canonical_source_id(st.source_id)
            if is_row_refusals_uri(st.input_uri):
                uri = row_refusals_uri(sid)                                             # the ledger: the group's canonical identity
            elif is_listing_uri(st.input_uri):
                uri = listing_uri(imap.canonical_source_id(listing_source_id(st.input_uri)), listing_rel(st.input_uri))   # a listing: its group's, canonical
            else:
                uri = imap.canonical(st.input_uri)
            payload = {"source_id": sid, "input_uri": uri, "json_path": st.json_path,
                       "found": st.found, "path_resolves": st.path_resolves, "current_members_json": sorted(st.current_members),
                       "observed_utc": ctx.now, "input_sha256": st.input_sha256}
            out.append(R.make_record("source_state", payload, ctx.producer, identity.adapter_record_id(payload, "source_state"), ctx.now))
    except AttributeError:
        pass
    return out


def _state_keys(records: Sequence[Mapping[str, object]]) -> Set[Tuple[str, str, str]]:
    """``(source_id, input_uri, json_path)`` of every ``source_state`` record in ``records``."""
    out: Set[Tuple[str, str, str]] = set()
    for rec in records:
        if rec.get("record_type") != "source_state":
            continue
        p = rec.get("payload") or {}
        out.add((str(p.get("source_id")), str(p.get("input_uri")), str(p.get("json_path"))))
    return out


def _vanished_state_record(ctx: AdapterContext, source_id: str, input_uri: str, json_path: str) -> Dict[str, object]:
    """One ``source_state found = 0`` record: the source names nothing under this key now."""
    payload = {"source_id": source_id, "input_uri": input_uri, "json_path": json_path, "found": 0, "path_resolves": 0,
               "current_members_json": [], "observed_utc": ctx.now, "input_sha256": None}
    return R.make_record("source_state", payload, ctx.producer, identity.adapter_record_id(payload, "source_state"), ctx.now)


def _cleared_state_records(ctx: AdapterContext, conn: Optional[sqlite3.Connection], source_id: str, stated: Set[Tuple[str, str, str]],
                           owned: Optional[Tuple[Set[str], Sequence[str]]] = None) -> List[Dict[str, object]]:
    """THE STATE-ROW RULE (module docstring): the ``source_state`` rows of ``source_id`` the database holds ``found = 1`` that the revision
    being published does NOT restate (``stated``: the keys of the rows it writes), each written ``found = 0`` -- a vanished input's rows
    when only SOME of the group's inputs left discovery (the runner retired watermarks, not state rows, for those: a row keyed on the
    gone file stood found = 1 with its last list), the key an earlier read's ledger used, a row of the outgoing revision the re-activated
    one never wrote. Scoped to the source's own id, which is one comparison group's (:func:`source_id_for`); rows an adapter files under
    another source id (a shared declaration file, a report of its own) are the vanished path's by input URI
    (:func:`_vanished_input_state_records`). With ``owned`` -- ``(input identity URIs, declared paths)``, a re-activation's -- only the
    GROUP-OWNED displaced keys are cleared (:func:`_owned_state_key`); an auxiliary row (a directory listing) stands at the latest fresh
    read's value. Nothing with no database."""
    if conn is None:
        return []
    out: List[Dict[str, object]] = []
    for uri, jpath in conn.execute("SELECT input_uri, json_path FROM source_state WHERE source_id = ? AND found = 1 ORDER BY input_uri, json_path",
                                   (source_id,)).fetchall():
        key = (source_id, str(uri), str(jpath))
        if key in stated:
            continue
        if owned is not None and not _owned_state_key({"input_uri": uri, "json_path": jpath}, source_id, owned[0], owned[1]):
            continue
        out.append(_vanished_state_record(ctx, source_id, str(uri), str(jpath)))
    return out


def _vanished_input_state_records(ctx: AdapterContext, conn: Optional[sqlite3.Connection], uris: Sequence[str],
                                  stated: Set[Tuple[str, str, str]]) -> List[Dict[str, object]]:
    """``found = 0`` for every ``found = 1`` state row keyed on an input that LEFT discovery, under whatever source id and json path it was
    observed at (the whole-group retirement's rule, :func:`retire_vanished_group`, applied to a group that lost only some inputs): a row an
    adapter files under another source id than the group's (``sampled_static``'s ``final_rung`` rows, a shared declaration file) is cleared
    with the input it is keyed on. Keys the revision restates (``stated``) are left to it."""
    if conn is None or not uris:
        return []
    out: List[Dict[str, object]] = []
    for uri in sorted(set(uris)):
        for sid, jpath in conn.execute("SELECT DISTINCT source_id, json_path FROM source_state WHERE input_uri = ? AND found = 1 ORDER BY source_id, json_path",
                                       (uri,)).fetchall():
            if (str(sid), uri, str(jpath)) in stated:
                continue
            out.append(_vanished_state_record(ctx, str(sid), uri, str(jpath)))
    return out


def _revision_extra(adapter: Adapter, group: str, forced_utc: Optional[str] = None,
                    anchor_decisions: Optional[Mapping[str, object]] = None) -> Dict[str, object]:
    """What a ``source_revision`` marker carries beyond its columns: the paper source strings the group reports under
    and the adapter's owner — so a source REFUSED before any accepted read still appears in ``v_source_status`` by name —
    plus the ``emission_version`` the id was minted under, for a ``--reread --force`` read ``forced`` = the pass
    stamp its id carries as a nonce, and -- on an accepted read that made at least one anchor comparison -- ``anchor_decisions``,
    the read's complete decision set with its fingerprint (:func:`anchor_current.decisions_block`)."""
    out: Dict[str, object] = {"lane_sources_legacy": list(status_sources_of(adapter, group)), "owner": owner_of(adapter),
                              "emission_version": emission_version_of(adapter)}
    if forced_utc:
        out["forced"] = str(forced_utc)
    if anchor_decisions:
        out["anchor_decisions"] = dict(anchor_decisions)
    return out


def _json_safe(obj):
    """The adapter's facts as JSON the record contract admits: tuples and sets become lists, keys become strings, numpy
    scalars become Python numbers, and a NON-FINITE float becomes the STRING ``'NaN'`` / ``'Infinity'`` — visible, never
    silently dropped, and never a record the validator refuses (a refused constituent would hold the WHOLE revision)."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        items = [_json_safe(v) for v in obj]
        try:
            return sorted(items)
        except TypeError:                      # mixed types: a deterministic order by repr, never a refusal
            return sorted(items, key=repr)
    if hasattr(obj, "item") and callable(getattr(obj, "item")) and not isinstance(obj, (str, bytes)):
        try:
            return _json_safe(obj.item())
        except (TypeError, ValueError):
            return str(obj)
    if isinstance(obj, bool) or obj is None or isinstance(obj, (int, str)):
        return obj
    if isinstance(obj, float):
        if obj != obj:
            return "NaN"
        if obj in (float("inf"), float("-inf")):
            return "Infinity" if obj > 0 else "-Infinity"
        return obj
    return str(obj)


#: The FREE-FORM fields of a payload: the contract's ``"extra": {unknown fields}`` and the ``extra_json`` column it lands in.
FREE_FORM_KEYS: Tuple[str, ...] = ("extra", "extra_json")


def _free_form_columns(record_type: str) -> Optional[frozenset]:
    """The typed payload columns of a record type whose unknown keys the validator PARKS in ``extra_json``; None for a type with
    no free-form field at all (``observation``, ``run_ingest``) — ``run_patch`` is handled by name (its ``fields`` block)."""
    table = R.RECORD_TYPE_TABLE.get(record_type)
    if table is None:
        effect = R.EFFECTS.get(record_type)
        if effect is None or effect.strict_keys or record_type in ("observation", "run_patch"):
            return None
        table = effect.columns_table
    from .. import schema_def as S
    return frozenset(S.payload_columns(table))


def _sanitise_extra_json(value):
    """``extra_json`` as an adapter may spell it: an object (sanitised like ``extra``) or its serialisation (parsed, sanitised and
    re-serialised ONLY when it carries a non-finite literal — Python's ``json`` reads ``NaN`` / ``Infinity`` back as floats — so a
    clean string keeps its bytes; a string that is not JSON is left for the validator's ``not valid JSON`` refusal)."""
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value
        return R.canonical_json(_json_safe(parsed)) if R.first_nonfinite(parsed) is not None else value
    return _json_safe(value)


def sanitise_free_form(record_type: str, payload: Mapping[str, object]) -> Dict[str, object]:
    """The record's FREE-FORM fields made JSON the contract admits, its TYPED columns untouched (the 2026-09-17 ``bcb_bfcl`` hold):
    a non-finite float inside ``extra`` / ``extra_json`` — or under an unknown key, which the validator parks in ``extra_json`` — is
    a pandas NaN copied out of a frame into provenance, a formatting artefact that must not hold a whole revision, so it becomes the
    string ``'NaN'`` / ``'Infinity'`` / ``'-Infinity'`` (:func:`_json_safe`; visible, never dropped). A non-finite float in a typed
    column (``value``, ``k``, ``n``, ``ci_lo``, ``delta*``, ``frac_*``, ``rate``, ``numerator``, ``strength_*`` …) is a measurement
    that does not exist and still refuses BY NAME: nothing here touches a column the record's table declares. ``run_patch``
    sanitises ``fields.extra`` / ``fields.extra_json`` only (every other patched key is a typed run column; ``verify`` is compared,
    never stored); ``observation`` and ``run_ingest`` carry no free-form field. Applied to EVERY record the runner writes — a fresh
    read's batches and a committed revision's re-emitted bytes alike — so a revision published before this guard folds once it is
    re-activated."""
    if record_type == "run_patch":
        fields = payload.get("fields")
        if not isinstance(fields, dict) or not any(k in fields for k in FREE_FORM_KEYS):
            return dict(payload)
        out = dict(payload)
        out["fields"] = {k: (_sanitise_extra_json(v) if k in FREE_FORM_KEYS else v) for k, v in fields.items()}
        return out
    cols = _free_form_columns(record_type)
    if cols is None:
        return dict(payload)
    return {k: (_sanitise_extra_json(v) if k in FREE_FORM_KEYS or k not in cols else v) for k, v in payload.items()}


@dataclass
class _ReadTally:
    """What one group's batches carried, for its status row: record counts, the runs' (model, object, eval) keys
    and which of them a ``run_metric`` measured."""
    counts: Dict[str, int] = field(default_factory=dict)
    runs: Dict[str, Tuple] = field(default_factory=dict)
    measured: Set[str] = field(default_factory=set)

    def note(self, rec: Mapping[str, object]) -> None:
        rt = str(rec.get("record_type"))
        p = rec.get("payload") or {}
        self.counts[rt] = self.counts.get(rt, 0) + 1
        if rt == "run":
            self.runs[str(p.get("run_id"))] = (p.get("model_id"), p.get("object_id"), p.get("eval_id"))
        elif rt == "run_metric" and p.get("value") is not None:
            self.measured.add(str(p.get("run_id")))

    def cells_measured(self) -> int:
        """Distinct (model, object, eval) keys with a measured value; a metric of a run outside this read counts by run id."""
        return len({self.runs.get(r, ("run", r)) for r in self.measured})


# --------------------------------------------------------------------------------------------
# the emitted-nothing guard (fail loud, never silent)
# --------------------------------------------------------------------------------------------

#: The status-facts keys whose NON-EMPTY value EXPLAINS a read that emitted no record: the adapter refused rows or the whole group
#: by name (``refuse`` / ``refused`` / ``refusals`` / ``row_refusals``), named inputs that left discovery (``missing``), a torn store
#: (``torn``), or a store it holds for a reason it states (``held``). Any other fact is evidence, not an explanation; an EMPTY value
#: under one of these keys (``[]``, ``{}``, ``0``, ``""``, ``None``) explains nothing.
EXPLANATION_KEYS: Tuple[str, ...] = ("refuse", "refused", "refusals", "row_refusals", "missing", "torn", "held")

#: Input roles that are pointers or control files rather than the store bytes a reader measures: a completion marker
#: (``completion.json``, ``DONE``, ``gen_stats.json``) and a log (``RESULT``, request evidence, usage ledgers). A group made only of
#: these — or only of ``#<group>`` aliases of a shared file — has nothing to emit from, so its empty read is not silent.
POINTER_ROLES = frozenset({"completion", "log", LISTING_ROLE})     # a directory listing (base.ListingInput) is an input, never a store

#: The one-line rule the flag carries, so a consumer reading the row knows what fired without opening this module.
EMITTED_NOTHING_RULE = ("a read of at least one store input that emitted no record and recorded no non-empty refuse / refused / refusals / "
                        "row_refusals / missing / torn / held entry is a SILENT read: filed with ok = 0 and this flag, so a consumer can "
                        "tell it from a group that was never read (fail loud, never silent)")

#: The ``extra_json`` key a silent read's status row carries (the flag :func:`emitted_nothing_unexplained` builds).
EMITTED_NOTHING_KEY = "emitted_nothing_unexplained"

#: Groups the FRAMEWORK knows emit no record BY DESIGN (facts-only / artifact-only), keyed by adapter name — a stopgap until each
#: reader declares ``emits_no_runs`` itself (:func:`base.emits_no_runs_of`) and its entry here is retired. Every predicate restates
#: the reader's own contract, cited; nothing here is a judgement about a store.
FACTS_ONLY_GROUPS: Dict[str, Callable[[str], bool]] = {
    # cap4096_declarations.py, module docstring: "Nothing is emitted from read -- this source has no measurements of its own
    # (covers_sources is empty); its whole product is the marker's source_state rows"
    "cap4096_lane": lambda group: group == "declarations",
    # regrade_twin.py, read(): the `manifests` group (the twin root's worklists, units, counts and ledger) yields an empty batch by design
    "regrade_twin": lambda group: group == "manifests",
    # sampled_static.py, read(): only a `grades_sampled/<dir>` group carries runs -- "standups / ladders / final_rung: artifacts only
    # (registered by the runner); the ladder facts ride the runs"
    "sampled_static": lambda group: os.path.basename(os.path.dirname(group)) != "grades_sampled",
}


def input_is_alias(inp: Input) -> bool:
    """Whether an input is a shared file declared under a per-group ``#<group>`` alias (the ``_alias`` convention of strength_mtdiff /
    seedrep_statics / relu_paired / seedrep_cluster / em_misalign: the uri carries the suffix, ``path`` stays the file): the uri names
    a ``#`` and does not itself resolve to the hashed ``path``. A file whose NAME carries a ``#`` resolves to it and is not an alias."""
    if "#" not in inp.uri:
        return False
    return os.path.realpath(os.path.expanduser(inp.uri)) != inp.path


def facts_only_group(adapter, group: str) -> bool:
    """Whether ``group`` of ``adapter`` emits no record BY DESIGN, so an empty read of it is not silent: the reader declares it
    (``emits_no_runs``, :func:`base.emits_no_runs_of`), the group is a bookkeeping group by the ``_`` naming convention (``_root_files``:
    the root's summaries and pointer material, whose facts ride the status row), or the framework knows the shape
    (:data:`FACTS_ONLY_GROUPS`)."""
    if emits_no_runs_of(adapter, group):
        return True
    if str(group).startswith("_"):
        return True
    rule = FACTS_ONLY_GROUPS.get(str(getattr(adapter, "name", "")))
    return bool(rule(str(group))) if rule is not None else False


def _explained_by(extra: Mapping[str, object]) -> List[str]:
    """The :data:`EXPLANATION_KEYS` whose value in the status facts is non-empty."""
    out: List[str] = []
    for key in EXPLANATION_KEYS:
        val = extra.get(key)
        if val is None or val is False or val == "" or (isinstance(val, (int, float)) and not isinstance(val, bool) and val == 0):
            continue
        if isinstance(val, (list, tuple, dict, set, frozenset)) and not val:
            continue
        out.append(key)
    return out


def emitted_nothing_unexplained(adapter, group: str, inputs: Sequence[Input], tally: Optional[_ReadTally],
                                extra: Mapping[str, object]) -> Optional[Dict[str, object]]:
    """The flag a SILENT read carries, or None. Fires when ALL hold: the read completed (a tally exists — a vanished group's empty
    revision has none); the batches carried NO record of any type (a group that files degeneration rows, judgments, objects, fit
    trials or contrasts for runs it does not mint has emitted something); the facts carry no non-empty :data:`EXPLANATION_KEYS`
    entry; the group is not facts-only by design (:func:`facts_only_group`); and at least one input is a STORE — not a ``#<group>``
    alias (:func:`input_is_alias`), not a pointer or control file (:data:`POINTER_ROLES`). The flag states what it judged: the
    inputs' count, their distinct kinds (an alias is ``alias``, else the input's role), how many are stores, and the rule."""
    if tally is None or tally.counts:
        return None
    if _explained_by(extra) or facts_only_group(adapter, group):
        return None
    kinds = sorted({"alias" if input_is_alias(i) else str(i.role) for i in inputs})
    stores = [i for i in inputs if not input_is_alias(i) and str(i.role) not in POINTER_ROLES]
    if not stores:
        return None
    return {"inputs": len(inputs), "input_kinds": kinds, "store_inputs": len(stores), "rule": EMITTED_NOTHING_RULE}


# --------------------------------------------------------------------------------------------
# the inputs-dropped guard (a read that shrinks its source must say so)
# --------------------------------------------------------------------------------------------

#: The ``source_revision.refusal`` prefix and ``extra_json`` key of a read refused for dropping too many of the previous accepted
#: revision's inputs without declaring them retired (:func:`inputs_dropped_comparison`).
INPUTS_DROPPED_KEY = "inputs_dropped_unexplained"

#: The one-line rule the refusal carries, so a consumer reading the marker knows what fired without opening this module.
INPUTS_DROPPED_RULE = ("a read whose input set lacks more than the threshold share of the previous accepted revision's inputs, none of "
                       "them declared retired by the adapter (inputs_retired), is REFUSED: the previous revision stays current and the "
                       "inputs' watermarks carry the refusal; the group is read again when an input changes, the adapter's "
                       "EMISSION_VERSION changes, or a pass carries --retry-refused or --full (a deliberate retirement is "
                       "declared, never inferred from the read)")

#: How many missing inputs a refusal names (the count and the share are always exact; the names are a sample).
INPUTS_DROPPED_NAMED = 20


class InputDropRefusal(AdapterRefusal):
    """A read refused by :func:`inputs_dropped_comparison`: the whole comparison rides along for the marker's ``extra``."""

    def __init__(self, source_id: str, comparison: Mapping[str, object]):
        self.comparison = dict(comparison)
        n, prev, frac, thr = (comparison["inputs_dropped"], comparison["inputs_previous"], comparison["inputs_dropped_fraction"],
                              comparison["inputs_dropped_threshold"])
        named = list(comparison.get("inputs_dropped_paths") or [])
        more = int(n) - len(named)
        why = (f"{INPUTS_DROPPED_KEY}: {n} of the {prev} inputs of the previous accepted revision {comparison['previous_revision_id']} "
               f"are absent from this read ({float(frac):.1%}, threshold {float(thr):.0%}); the previous revision stays current. "
               f"Missing ({len(named)} of {n} named): " + ", ".join(named) + (f" (+{more} more)" if more > 0 else "")
               + "; declare them under the adapter's inputs_retired (URIs or globs) to retire them on purpose")
        super().__init__(source_id, why)


#: The refusal prefix of a CLUSTER shard's read whose staged copy lacks any input of the previous accepted revision.
STAGED_COPY_INCOMPLETE_KEY = "staged_copy_incomplete"

#: The ``source_revision.refusal`` prefixes of a CLUSTER shard's read refused on a LISTING input of the group (:class:`ClusterListingRefusal`):
#: the staged generation's manifest attests no listing for the input's directory (``unattested``: a manifest written before the field, a
#: directory under no pinned generation, a listing input that names no directory), or attests one the staged listing is not (``mismatch``).
STAGED_LISTING_UNATTESTED_KEY = "staged_listing_unattested"
STAGED_LISTING_MISMATCH_KEY = "staged_listing_mismatch"


class ClusterListingRefusal(AdapterRefusal):
    """A CLUSTER shard's read refused because a directory LISTING input of the group (:class:`base.ListingInput`) cannot be shown to be
    the canonical's: the pinned generation's store manifest attests, per directory, the canonical directory's whole listing at the
    window's push (``stores.MANIFEST_RULES["directories"]``, empty directories included), and the shard compares every listing input's
    fingerprint with THAT attestation -- never with a previous accepted revision, whose listing may equal a stale staged one while the
    canonical moved (the canonical gained an empty ignored directory as a tracked file changed: a file-only stage carried the new file
    and not the directory, the stale listing equalled the previous revision's, and the changed file minted an accepted revision whose
    ``$.entries`` omitted the directory). ``unattested``: the manifest carries no listing for the directory (written before the field,
    a directory under no pinned generation, an input that names no directory); ``mismatch``: the staged directory lists other than the
    attestation. Either way nothing is written for the group; the pull records the refusal by name, and no grading host read follows (C6, S2)."""

    def __init__(self, source_id: str, kind: str, listings: Mapping[str, Tuple[str, Optional[str], str]]):
        self.kind = kind
        self.listings = {str(u): (str(staged), (str(att) if att else None), str(path)) for u, (staged, att, path) in listings.items()}
        key = STAGED_LISTING_MISMATCH_KEY if kind == "mismatch" else STAGED_LISTING_UNATTESTED_KEY
        named = [f"{u} [{path or 'no directory'}: staged {staged[:12]}, attested {(att[:12] if att else 'none')}]"
                 for u, (staged, att, path) in sorted(self.listings.items())[:INPUTS_DROPPED_NAMED]]
        more = len(self.listings) - len(named)
        what = ("list other than the pinned generation's manifest attests for the canonical directory" if kind == "mismatch"
                else "have no attested listing in the pinned generation's manifest (a manifest written before the directories field, a directory under no pinned generation, or an input that names no directory)")
        why = (f"{key}: {len(self.listings)} listing input(s) of the group {what} on the cluster's staged copy; a shard never mints a listing the "
               f"canonical copy does not have: nothing is written and the group is recorded refused (no grading host read follows). Listings: " + "; ".join(named) + (f" (+{more} more)" if more > 0 else ""))
        super().__init__(source_id, why)


class AnchorCurrencyRefusal(AdapterRefusal):
    """A group's read asked the pass's anchor currency (``ctx.anchor_currency()``) and the frame could not be loaded
    (:class:`anchor_current.AnchorCurrencyUnavailable`: no ``v_run_current``, no connection): the GROUP refuses by name -- a contrast-less
    revision would withdraw every contrast of the group -- and NOTHING is written for it: no refused marker, no watermark stamp,
    no state row, on the grading host and on a cluster shard alike (whose sidecar is incomplete and whose group the grading host reads).
    An unavailable frame is a fault of the pass's environment, not a judgment about the inputs, so the group is not parked
    behind ``--retry-refused``: the next pass retries it under the selection that chose it. ``cause`` is the loader's own text."""

    def __init__(self, source_id: str, cause: str):
        super().__init__(source_id, f"the anchor currency of the pass is unavailable, the group refuses whole: {cause}")
        self.cause = cause


def _frame_guarded(batches: Iterable[RecordBatch], source_id: str) -> Iterable[RecordBatch]:
    """The adapter's ``read`` with an unavailable frame translated into a group refusal: the loader's exception is not an
    :class:`base.AdapterRefusal`, and uncaught it would end the PASS (every later group and adapter unread) instead of the group."""
    try:
        yield from batches
    except AnchorCurrencyUnavailable as exc:
        raise AnchorCurrencyRefusal(source_id, str(exc)) from exc


class ClusterInputDropRefusal(AdapterRefusal):
    """A CLUSTER shard's read refused because its input set lacks ANY input of the previous accepted revision: on a
    staged copy a dropped input is an INCOMPLETE COPY -- a tree no component covers, a member the pinned generation lacks -- never a
    producer's vanished file, so no reduced-coverage revision is minted whatever share the grading host threshold admits. Names every missing
    input (up to :data:`INPUTS_DROPPED_NAMED`) with the staged component that would have covered it (:meth:`ClusterRead.covering`), or
    that none does. Nothing is written for the group -- no revision, no marker, no watermark; the refusal is recorded by name, and no
    grading host read follows (C6, S2)."""

    def __init__(self, source_id: str, comparison: Mapping[str, object], covering: Mapping[str, Optional[str]]):
        self.comparison = dict(comparison)
        self.covering = dict(covering)
        n, prev = int(comparison["inputs_dropped"]), int(comparison["inputs_previous"])
        named = [f"{u} [{covering.get(u) or 'no staged component covers it'}]" for u in list(comparison.get("inputs_dropped_uris") or [])[:INPUTS_DROPPED_NAMED]]
        more = n - len(named)
        why = (f"{STAGED_COPY_INCOMPLETE_KEY}: {n} of the {prev} inputs of the previous accepted revision {comparison['previous_revision_id']} are "
               f"absent from this read on the cluster ({float(comparison['inputs_dropped_fraction']):.1%}; the grading host threshold does not apply to "
               f"a staged copy): nothing is published for the group, which is recorded refused (no grading host read follows). Missing ({len(named)} of {n} named): "
               + "; ".join(named) + (f" (+{more} more)" if more > 0 else ""))
        super().__init__(source_id, why)


def _relative_input(uri: str, roots: Sequence[str]) -> str:
    """An input URI relative to the adapter's declared root it lies under (the record's own spelling otherwise)."""
    for r in sorted((str(x) for x in roots or ()), key=len, reverse=True):
        base = r.rstrip("/") + "/"
        if uri.startswith(base):
            return uri[len(base):]
    return uri


RETIREMENT_DECLARED_KEY = "retirement_declared"      #: a retirement marker's ``extra`` entry: the ``inputs_retired`` declaration that covered every vanished input


def declared_retirements(adapter, group: str, absent: Sequence[str]) -> Tuple[List[str], List[str], List[str]]:
    """Of ``absent`` -- CANONICAL URIs of inputs an earlier revision read that this read does not see -- the ones the adapter DECLARES
    retired and the ones it does not, plus the declaration patterns that matched: ``(declared, undeclared, patterns)``, the first two in
    ``absent``'s order, the patterns in the order :func:`base.inputs_retired_of` lists them. The ONE matching rule for a deliberate
    retirement, wherever a read judges an absent input (:func:`inputs_dropped_comparison`, :func:`retire_vanished_group`): the adapter's
    ``inputs_retired``, asked with ``group`` in the spelling the adapter ADDRESSES the group under -- the key :attr:`GroupPlan.groups`
    carries: the canonical spelling under the home tree a shard lays out (the declarations hang off ``$HOME`` at import),
    the re-pointed root's under a relocated adapter (:func:`prior_group_spelling`) -- its input URIs or ``fnmatch`` globs canonicalized
    through the pass's identity map and matched exactly or by ``fnmatch.fnmatchcase`` against the canonical URI. A malformed declaration
    refuses by name (``ValueError``) as before."""
    imap = identity_map()
    retired = [imap.canonical(str(pat)) for pat in inputs_retired_of(adapter, group)]
    declared: List[str] = []
    undeclared: List[str] = []
    patterns: List[str] = []
    for u in absent:
        hits = [pat for pat in retired if u == pat or fnmatch.fnmatchcase(u, pat)]
        (declared if hits else undeclared).append(u)
        patterns += [pat for pat in hits if pat not in patterns]
    return declared, undeclared, patterns


def inputs_dropped_comparison(adapter, group: str, inputs: Sequence[Input], prev_rev: Optional[str],
                              prev_inputs: Optional[Mapping[str, str]]) -> Optional[Dict[str, object]]:
    """The read's input set against the previous accepted revision's: None when there is nothing to compare with (no accepted
    revision, or one that recorded no inputs); else a dict — ``previous_revision_id``, ``inputs_previous``, ``inputs_now``,
    ``inputs_dropped`` (previous inputs absent now and NOT declared retired), ``inputs_retired`` (absent and declared),
    ``inputs_added``, ``inputs_dropped_fraction`` (dropped / previous), ``inputs_dropped_threshold``, ``inputs_dropped_paths``
    (up to :data:`INPUTS_DROPPED_NAMED`, relative to the adapter's root), ``tripped``. A superset drops nothing. A group that emits no
    measurement BY DESIGN (:func:`facts_only_group`: a declared ``emits_no_runs``, a ``_``-prefixed bookkeeping group, a
    :data:`FACTS_ONLY_GROUPS` shape) has no key to withdraw silently and never trips — its input set follows its declaration files
    (``cap4096_lane:declarations`` restating a path a file dropped) — and the comparison says so (``facts_only``). The adapter's
    declarations (:func:`base.inputs_dropped_threshold_of`, :func:`base.inputs_retired_of`) are read here and refuse by name when
    malformed."""
    # the synthetic anchor-decision input is no file: it is neither a previous input a read can drop nor a present one
    prev_inputs = {u: s for u, s in (prev_inputs or {}).items() if not is_decision_uri(u)}
    if prev_rev is None or not prev_inputs:
        return None
    threshold = inputs_dropped_threshold_of(adapter, group)
    declared_threshold = getattr(adapter, "inputs_dropped_threshold", None)
    if callable(declared_threshold):
        declared_threshold = declared_threshold(group)
    imap = identity_map()
    now_uris = {i.identity_uri for i in inputs if not is_decision_input(i)}
    absent = sorted(set(prev_inputs) - now_uris)
    # the previous revision's inputs are canonical; the adapter's declarations (spelled locally) are matched canonically (declared_retirements)
    declared, dropped, _patterns = declared_retirements(adapter, group, absent)
    frac = len(dropped) / len(prev_inputs)
    roots = [imap.canonical(str(r)) for r in (getattr(adapter, "roots", ()) or ()) if r]
    facts_only = facts_only_group(adapter, group)
    out = {"previous_revision_id": prev_rev, "inputs_previous": len(prev_inputs), "inputs_now": len(now_uris),
           "inputs_dropped": len(dropped), "inputs_retired": len(declared), "inputs_added": len(now_uris - set(prev_inputs)),
           "inputs_dropped_fraction": round(frac, 4), "inputs_dropped_threshold": threshold,
           # whether the threshold is the ADAPTER's own declaration (a deliberate tolerance for this group) or the framework default
           "inputs_dropped_threshold_declared": declared_threshold is not None,
           "inputs_dropped_paths": [_relative_input(u, roots) for u in dropped[:INPUTS_DROPPED_NAMED]],
           # the same inputs in their canonical spelling, for a CLUSTER shard's refusal to name with the component that would have covered
           # each (:class:`ClusterInputDropRefusal`); never written on a marker (:func:`_comparison_extra` and the refused marker strip it)
           "inputs_dropped_uris": list(dropped[:INPUTS_DROPPED_NAMED]),
           "tripped": bool(dropped) and frac > threshold and not facts_only}
    if facts_only:
        out["facts_only"] = True
    return out


def _comparison_extra(comparison: Optional[Mapping[str, object]]) -> Dict[str, object]:
    """What an accepted marker records of the comparison (the number is on the record whether or not it tripped)."""
    if not comparison:
        return {}
    keys = ("previous_revision_id", "inputs_previous", "inputs_now", "inputs_dropped", "inputs_retired", "inputs_added",
            "inputs_dropped_fraction", "inputs_dropped_threshold", "inputs_dropped_threshold_declared", "facts_only")
    out: Dict[str, object] = {k: comparison[k] for k in keys if k in comparison}
    if comparison.get("inputs_dropped"):
        out["inputs_dropped_paths"] = list(comparison.get("inputs_dropped_paths") or [])
    return out


def run_set_extra(prev_rev: Optional[str], prev_keys: Iterable[str], now_keys: Iterable[str], comparison: Optional[Mapping[str, object]],
                  declared: Optional[Mapping[str, object]], vanished: bool = False) -> Dict[str, object]:
    """What a marker records of its RUN SET against the source's accepted head. ``extra.run_set`` -- ``previous_revision_id``,
    ``runs_previous``, ``runs_now``, ``runs_dropped``, ``runs_added``, the ids read the way the harvester reads them
    (:func:`harvest.run_ids_of_observation_keys`: every run-scale key carrying a ``run_id``) -- whenever there is a head to compare
    with; and ``extra.declared_run_drops`` (``harvest.DECLARED_RUN_DROPS_KEY``) when the read DECLARES a drop: the adapter's
    ``runs_retired`` (:func:`base.runs_retired_of`, read before any store is opened: its ids or globs and its reason), the run ids
    that left with inputs the adapter declared retired (``inputs_retired``) or admitted by the adapter's OWN ``inputs_dropped_threshold``
    (the dropped ids by name, so a drop the read did not see is still undeclared; the framework default threshold declares nothing),
    and a group whose every input left discovery (``input_vanished``: ``"*"``). ``declared_by`` names each
    source of the declaration. Nothing here decides: the record says what the reader saw and meant, and the harvester holds a
    revision that names fewer run ids than the head with a drop no declaration covers (``Ingestor._undeclared_run_drop``)."""
    prev_ids = H.run_ids_of_observation_keys(prev_keys)
    now_ids = H.run_ids_of_observation_keys(now_keys)
    dropped = sorted(prev_ids - now_ids)
    out: Dict[str, object] = {}
    if prev_rev is not None:
        out["run_set"] = {"previous_revision_id": prev_rev, "runs_previous": len(prev_ids), "runs_now": len(now_ids),
                          "runs_dropped": len(dropped), "runs_added": len(now_ids - prev_ids)}
    ids: List[str] = []
    star = False
    reasons: List[str] = []
    by: List[str] = []
    if declared is not None:
        star = declared["run_ids"] == "*"
        if not star:
            ids += [str(x) for x in declared["run_ids"]]
        reasons.append(str(declared["reason"]))
        by.append("adapter:runs_retired")
    if vanished:
        star = True
        reasons.append("input_vanished: every input of the group left discovery; its keys are withdrawn (D32-6)")
        by.append("runner:input_vanished")
    n_retired = int((comparison or {}).get("inputs_retired") or 0)
    if n_retired and dropped:
        ids += dropped
        reasons.append(f"inputs_retired: {n_retired} input(s) of the previous accepted revision declared retired by the adapter; "
                       f"the {len(dropped)} run id(s) that left with them")
        by.append("runner:inputs_retired")
    n_admitted = int((comparison or {}).get("inputs_dropped") or 0)
    if n_admitted and dropped and (comparison or {}).get("inputs_dropped_threshold_declared") and not (comparison or {}).get("tripped"):
        # an input loss the ADAPTER's own threshold admitted is the adapter's declared tolerance for the group (raising the threshold is
        # one of the two ways to declare a deliberate removal); the framework's default threshold is nobody's declaration,
        # so a loss it lets through is undeclared and the harvester holds the run drop
        ids += dropped
        reasons.append(f"inputs_dropped_threshold: {n_admitted} input(s) of the previous accepted revision absent, within the adapter's "
                       f"declared threshold {float(comparison['inputs_dropped_threshold']):.0%}; the {len(dropped)} run id(s) that left with them")
        by.append("runner:inputs_dropped_threshold")
    if by:
        out[H.DECLARED_RUN_DROPS_KEY] = {"run_ids": "*" if star else sorted(set(ids)), "reason": "; ".join(reasons), "declared_by": by,
                                         "runs_dropped_at_read": len(dropped)}
    return out


def row_refusals_of(adapter) -> List[str]:
    """The rows a read refused BY NAME, as the status row reports them (`row_refusals`; the detail's `N row(s) refused by name`): the shared
    emitter's report (``adapter.last_report.refusals``) where the adapter emits through it, else the adapter's own `lineage_common.RowRefusals`
    ledger (``adapter.refusals.members()``: petri_stats, bon_stores, the seed-replicate readers, ...). Until registry follow-ups-4 item 12 only
    the report was read, so a ledger adapter's status said `no row refused` while its `$.row_refusals` state row listed the refusal
    (petri_stats' held revision 0bd0dd6a39d1732b)."""
    report = getattr(adapter, "last_report", None)
    out = [str(r) for r in (getattr(report, "refusals", None) or [])] if report is not None else []
    if out:
        return out
    ledger = getattr(adapter, "refusals", None)
    members = getattr(ledger, "members", None)
    if callable(members) and hasattr(ledger, "items"):
        try:
            return [str(m) for m in members()]
        except Exception:  # noqa: BLE001 -- a ledger that cannot list itself is no refusal of the status row
            return []
    return out


def _source_status_records(ctx: AdapterContext, adapter: Adapter, group: str, source_id: str, rev: str, where: Sequence[str],
                           tally: Optional[_ReadTally], detail: Optional[str] = None, missing: Sequence[str] = (),
                           reread: Optional[Mapping[str, object]] = None, inputs: Sequence[Input] = (),
                           anchor_drift: Optional[Mapping[str, object]] = None) -> List[Dict[str, object]]:
    """One ``source_status`` record per paper source string the group reports under, tagged with the read's revision:
    ``ok`` = the read was accepted, the adapter's facts record no ``refuse`` entry and the read was not SILENT
    (:func:`emitted_nothing_unexplained`, judged over ``inputs``); ``detail`` = one line for the gap map;
    ``extra`` = the adapter's :meth:`Adapter.status_extra` facts plus the generic ``refuse`` / ``torn`` / ``missing`` lists,
    the emitter's per-row refusals (``row_refusals``, visible but never a build refusal), the record counts and — for a silent
    read only — ``emitted_nothing_unexplained``. ``reread`` (:func:`reread_committed`) is stored under ``extra.reread`` and its
    key drift, if any, named in ``detail``. ``anchor_drift`` (the read's anchor decision fingerprint differs from
    the pre-read one with no file input changed) is stored under ``extra.anchor_decision_drift`` and named in ``detail`` -- a fact,
    never a refusal."""
    extra = identity_map().canonical_deep(_json_safe(status_extra_of(adapter, group))) if tally is not None else {}
    for key in ("refuse", "torn", "missing"):
        val = extra.get(key)
        extra[key] = list(val) if isinstance(val, list) else ([] if val in (None, "", {}) else [val])
    if missing:
        extra["missing"] = list(extra["missing"]) + [m for m in missing if m not in extra["missing"]]
    row_refusals = row_refusals_of(adapter) if tally is not None else []
    extra.setdefault("row_refusals", row_refusals)
    if tally is not None:
        extra.setdefault("record_counts", dict(sorted(tally.counts.items())))
    silent = emitted_nothing_unexplained(adapter, group, inputs, tally, extra)
    if silent is not None:
        extra[EMITTED_NOTHING_KEY] = silent
    cells = tally.cells_measured() if tally is not None else 0
    if detail is None:
        counts = tally.counts if tally is not None else {}
        detail = (f"{adapter.name} read {len(where)} input(s) of {source_id} (revision {rev}): {sum(counts.values())} record(s)"
                  + (" — " + ", ".join(f"{n} {rt}" for rt, n in sorted(counts.items())) if counts else "")
                  + f"; {cells} measured (model, object, eval) key(s)"
                  + (f"; {len(row_refusals)} row(s) refused by name: " + " | ".join(row_refusals[:5]) + (" …" if len(row_refusals) > 5 else "")
                     if row_refusals else "; no row refused"))
        if extra["refuse"]:
            detail += f"; REFUSE ({len(extra['refuse'])}): " + " | ".join(str(r) for r in extra["refuse"])
    if silent is not None:
        detail += (f"; EMITTED NOTHING (unexplained): {silent['store_inputs']} store input(s) read, no record emitted, no refuse / row_refusals / "
                   "missing / torn / held entry recorded -- filed ok = 0 so it is never mistaken for a group that was never read")
        log.warning("emitted nothing (unexplained): %s read %d input(s) of %s (revision %s) and emitted no record, recording no refuse / "
                    "row_refusals / missing / torn / held entry; its status row is filed ok = 0 with extra.%s (input kinds: %s)",
                    adapter.name, len(where), source_id, rev, EMITTED_NOTHING_KEY, ", ".join(silent["input_kinds"]))
    if reread is not None:
        extra["reread"] = _json_safe(reread)
        drift = reread.get("key_drift") or {}
        if any(int(v) for v in drift.values() if isinstance(v, int)):
            detail += ("; re-read of the committed revision: KEY DRIFT vs its rows (" + ", ".join(f"{k} {v}" for k, v in sorted(drift.items()))
                       + ") — measurements unchanged; a reader that emits different keys publishes a new revision (a changed input)")
        else:
            detail += "; re-read of the committed revision: its keys reproduced, measurements unchanged"
    if anchor_drift is not None:
        extra["anchor_decision_drift"] = _json_safe(anchor_drift)
        detail += (f"; ANCHOR DECISION DRIFT: the read's decision set ({anchor_drift.get('post_read_fingerprint')}) differs from the pre-read "
                   f"evaluation of the head's recorded set ({anchor_drift.get('pre_read_fingerprint')}) with no file input changed "
                   f"({anchor_drift.get('n_comparisons_pre')} -> {anchor_drift.get('n_comparisons_post')} comparison(s)); the revision id was minted "
                   "from the pre-read fingerprint and the watermark carries the post-read one, so the next pass compares against what was read")
    ok = 1 if tally is not None and not extra["refuse"] and silent is None else 0
    out = []
    imap = identity_map()
    for src in status_sources_of(adapter, group):
        payload = {"lane_source_legacy": src, "adapter": adapter.name, "source_id": source_id, "owner": owner_of(adapter),
                   "where_json": list(where), "ok": ok, "detail": imap.canonical_text(str(detail)) if imap else detail, "cells_measured": cells,
                   "observed_utc": ctx.now, "source_revision_id": rev, "extra": extra}
        out.append(R.make_record("source_status", payload, ctx.producer, identity.adapter_record_id(payload, "source_status"), ctx.now))
    return out


def _compact(utc: str) -> str:
    return "".join(ch for ch in utc if ch.isdigit())


def _deferred_result(adapter: Adapter, group: str, rev: str, budget: "ReactivationBudget", what: str) -> GroupResult:
    """The result of a re-activation the spent :class:`ReactivationBudget` turned away: nothing written, the cap named."""
    return GroupResult(adapter.name, group, rev, accepted=False, deferred=True,
                       error=f"{what} of revision {rev} deferred: this pass already wrote {budget.taken} activation bundle(s) for {adapter.name} "
                             f"(the per-pass cap MAX_REACTIVATIONS_PER_PASS = {budget.limit}); nothing written, the next pass takes it")


def reactivate_committed(root: str, adapter: Adapter, group: str, inputs: Sequence[Input], ctx: AdapterContext,
                         conn: Optional[sqlite3.Connection], vanished_uris: Sequence[str] = (),
                         budget: Optional["ReactivationBudget"] = None) -> GroupResult:
    """The content identity (revision id) of these inputs is ALREADY published: reuse the committed bytes.
    If it is the source's current revision, only the watermarks are re-stamped (a touched file). Otherwise the
    revision is RE-ACTIVATED: its measurement records are re-emitted with fresh stamps beside a fresh
    ``source_revision`` marker, its OWN ``source_state`` rows replayed (:func:`committed_state_records`; the outgoing head's other
    keys cleared ``found = 0``) and the withdrawals of the outgoing revision's keys, as one bundle directory
    ``_activations/<rev>/<stamp>/`` — the harvester applies it atomically (constituents, then the marker), so the
    active revision's values, membership and state are restored whatever a later revision had written. A ``budget`` caps the
    bundles one adapter's pass writes (:class:`ReactivationBudget`): a re-stamp is free, a bundle spends one, a spent
    budget defers the group. The caller (:func:`refresh_group`) has already established that no marker of ``rev`` is
    on the record log unfolded (:func:`pending_fold`)."""
    source_id, rev = _revision_for(adapter, group, inputs)
    out_dir = os.path.join(H.records_dir(root), "adapters", adapter.name, rev)
    prev_rev = _current_revision(conn, source_id)
    now = ctx.now
    if _revision_accepted(conn, rev) and not _revision_has_status(conn, source_id, rev):
        # An ACCEPTED published revision with NO `source_status` row: committed before the status fact existed (the runner
        # then filed no status). Re-stamping or re-activating it here keeps it current AND invisible in `v_source_status` --
        # exactly how `strength_ladder` and `relu_native` sat with four accepted revisions each and no status row
        # (2026-09-18): their inputs were touched, the content identity was unchanged, and this path wrote a marker and
        # nothing else. A re-read files the status row from a fresh read's facts and, for a revision that is not the head,
        # re-activates the committed bytes exactly as this function does below. The status row is read off the DATABASE: a
        # re-read whose bundle is on the log unfolded is not "no status" but a pending fold, and refresh_group has ruled
        # that out before calling here.
        return reread_committed(root, adapter, group, inputs, ctx, conn, vanished_uris, budget=budget)
    if prev_rev == rev:
        records = _watermark_records(ctx, adapter.name, source_id, inputs)
        records += _retire_watermarks(ctx, conn, adapter.name, source_id, vanished_uris, rev)
        path = R.write_records(os.path.join(activations_dir(root, adapter.name, rev), f"{_compact(now)}-source_watermark.jsonl"), records)
        return GroupResult(adapter.name, group, rev, accepted=True, records_written=0, withdrawals=0, files=[path])
    if budget is not None and not budget.take():
        return _deferred_result(adapter, group, rev, budget, "re-activation")
    prev_inputs = _accepted_inputs(conn, prev_rev)
    comparison = inputs_dropped_comparison(adapter, group, inputs, prev_rev, prev_inputs)
    declared_drops = runs_retired_of(adapter, group)      # the reader's run-drop declaration; a malformed one refuses by name
    if comparison and comparison["tripped"]:
        # a reverted input set that would withdraw too much of the head: refused the way a read of a published revision refuses
        # (its id is an accepted revision's and must not carry an unaccepted marker) — the watermarks' last_error and this result
        exc = InputDropRefusal(source_id, comparison)
        refused = _watermark_records(ctx, adapter.name, source_id, inputs, last_error=str(exc), attempt_revision_id=rev)
        path = R.write_records(os.path.join(refused_dir(root, adapter.name, rev), f"{_compact(now)}-source_watermark.jsonl"), refused)
        log.warning("inputs dropped (unexplained): re-activation of %s for %s refused: %s", rev, source_id, exc.why)
        return GroupResult(adapter.name, group, rev, accepted=False, error=str(exc), files=[path], inputs_dropped=True)
    prev = _current_observations(conn, source_id)
    prev_items = _current_item_keys(conn, prev_rev) if prev_rev else set()
    now_keys, now_items = committed_keys(out_dir)
    withdrawals = _withdrawal_records(ctx, source_id, rev, prev, prev_rev, prev_items, now_keys, now_items, "removed_measurement")
    # a reverted anchor decision re-activates the revision its decisions built exactly like a reverted file (the id matched): the bundle
    # marker restates the committed marker's decision set
    rev_payload = {"revision_id": rev, "source_id": source_id, "adapter": adapter.name, "observed_utc": now,
                   "inputs_json": inputs_json(inputs), "accepted": 1, "priority": adapter_priority(adapter, group),
                   "extra": dict(_revision_extra(adapter, group, anchor_decisions=committed_anchor_decisions(out_dir)), **_comparison_extra(comparison),
                                 **run_set_extra(prev_rev, prev.keys(), now_keys, comparison, declared_drops))}
    marker = [R.make_record("source_revision", rev_payload, ctx.producer, identity.adapter_record_id(rev_payload, "source_revision"), now)]
    marker += withdrawals
    # the revision's STATE is the committed revision's own GROUP-OWNED state rows, replayed (the state-row and ownership rules, module
    # docstring) -- never the adapter's ledger, which is the LAST read's (a later revision of this same group, when an input was added and
    # then removed), and never a committed snapshot of an auxiliary or shared key (a directory listing: the latest fresh read's row stands)
    # -- and every GROUP-OWNED row of the source the outgoing head wrote under a key this revision never did is cleared (found = 0)
    replay_uris = {i.identity_uri for i in inputs if not is_decision_input(i)}
    head_uris = {str(u) for u in (prev_inputs or {}) if not is_decision_uri(u)}
    owned_paths = state_owned_paths_of(adapter)
    states = committed_state_records(out_dir, ctx, source_id, replay_uris, owned_paths)
    states += _cleared_state_records(ctx, conn, source_id, _state_keys(states), owned=(replay_uris | head_uris, owned_paths))
    marker += states
    marker += _watermark_records(ctx, adapter.name, source_id, inputs)
    marker += _retire_watermarks(ctx, conn, adapter.name, source_id, vanished_uris, rev)
    bundle = os.path.join(activations_dir(root, adapter.name, rev), _compact(now))
    staged = os.path.join(root, "staging", "adapters", adapter.name, f"{rev}-activation-{_compact(now)}")
    shutil.rmtree(staged, ignore_errors=True)
    _, n_records = _reemit_committed(out_dir, staged, now, rev)
    R.write_records(os.path.join(staged, R.record_file_name(999998, "source_revision")), marker)
    files = publish_staged(staged, bundle)
    return GroupResult(adapter.name, group, rev, accepted=True, records_written=n_records, withdrawals=len(withdrawals), files=files, reactivated=True)


def committed_marker_facts(out_dir: str) -> Dict[str, object]:
    """What the COMMITTED bytes of a published revision directory say about its inputs, read from the marker file alone (the
    ``<seq>-source_revision`` file): ``{"marker": <path>, "revision": <source_revision payload>, "watermarks": {identity uri: payload},
    "states": [<source_state payload>, ...]}``. Refuses by name (:class:`AdapterRefusal`) when the directory holds no marker or the marker
    no ``source_revision`` record -- a re-activation from committed bytes has nothing to stand on then."""
    if not os.path.isdir(out_dir):
        raise AdapterRefusal(out_dir, "no committed revision directory")
    markers = [os.path.join(out_dir, f) for f in sorted(os.listdir(out_dir)) if _is_marker_file(f)]
    if not markers:
        raise AdapterRefusal(out_dir, "the revision directory holds no source_revision marker file")
    revision: Optional[Dict[str, object]] = None
    watermarks: Dict[str, Dict[str, object]] = {}
    states: List[Dict[str, object]] = []
    for marker in markers:
        for _, rec, _ in R.read_record_file(marker, compute_sha=False):
            rt, payload = rec.get("record_type"), dict(rec.get("payload") or {})
            if rt == "source_revision" and revision is None:
                revision = payload
            elif rt == "source_watermark" and payload.get("last_error") is None:
                watermarks[str(payload.get("input_uri"))] = payload
            elif rt == "source_state":
                states.append(payload)
    if revision is None:
        raise AdapterRefusal(markers[0], "the marker file carries no source_revision record")
    return {"marker": markers[-1] if len(markers) == 1 else markers[0], "revision": revision, "watermarks": watermarks, "states": states}


def inputs_from_committed(facts: Mapping[str, object]) -> List[Input]:
    """The marker's ``inputs_json`` as :class:`Input` objects -- nothing is stat'ed or opened. A FILE input takes its size and mtime from
    the committed ``source_watermark`` record of the same sha; one the marker names with no committed watermark fact, or whose fact carries
    another sha, refuses by name (the plan's rule: a marker without them refuses -- the file inputs' rule). The synthetic anchor-decision
    input (``registry://anchor-decision/<source_id>``) is rebuilt from the marker ALONE (:meth:`AnchorDecisionInput.of`, sha = its
    ``inputs_json`` value) and no watermark fact is demanded for it: BY DESIGN the marker folds the PRE-read fingerprint the revision id was
    minted from while its committed watermark carries the POST-read one (:func:`_read_and_publish`; an ``anchor_drift`` fact when they
    differ), so demanding a matching fact refused by id every revision whose read decided differently from the pre-read replay --
    ``petri_stats`` ``e84fc1dffc1be901`` at every window's pull, and the adapter not read whole since (registry-hardening 3a).
    ``inputs_json(inputs)`` reproduces the committed ``inputs_json`` byte for byte, so the revision id stands; :func:`reactivate_by_id`
    stamps the decision WATERMARK from the committed fact (:func:`_committed_watermark_inputs`), never from this value alone."""
    revision = facts["revision"]          # type: ignore[index]
    watermarks: Mapping[str, Mapping[str, object]] = facts["watermarks"]   # type: ignore[assignment]
    inputs_json_ = revision.get("inputs_json") or {}     # type: ignore[union-attr]
    if not isinstance(inputs_json_, Mapping):
        raise AdapterRefusal(str(facts.get("marker")), "the marker's inputs_json is malformed; nothing to re-activate from")
    if not inputs_json_:
        raise AdapterRefusal(str(facts.get("marker")), "the marker's inputs_json is empty: an empty revision is a RETIREMENT (retire_vanished_group), "
                                                       "which reactivate_by_id re-activates from the database (_reactivate_retirement), never from input facts")
    out: List[Input] = []
    for uri in sorted(inputs_json_):
        sha = str(inputs_json_[uri])
        if is_decision_uri(uri):
            try:
                out.append(AnchorDecisionInput.of(decision_source_id(uri), sha))
            except ValueError as exc:
                raise AdapterRefusal(str(facts.get("marker")), f"the marker names the anchor-decision input {uri} with a malformed fingerprint: {exc}")
            continue
        wm = watermarks.get(uri)
        if wm is None or str(wm.get("sha256")) != sha:
            raise AdapterRefusal(str(facts.get("marker")), f"the marker names input {uri} (sha {sha[:12]}) but the committed bytes carry no "
                                                            "matching source_watermark fact for it; a re-activation from committed bytes refuses")
        out.append(Input(uri=uri, path=uri, sha256=sha, size=int(wm.get("size") or 0), mtime_utc=str(wm.get("mtime_utc") or ""),
                         role="store", identity_uri=uri))
    return out


def _committed_watermark_inputs(facts: Mapping[str, object], inputs: Sequence[Input]) -> List[Input]:
    """The inputs a re-activation from committed bytes stamps WATERMARKS for (:func:`reactivate_by_id`): every file input as
    :func:`inputs_from_committed` built it (its committed sha), and the anchor-decision input with the COMMITTED watermark's fingerprint --
    the post-read one -- when the committed marker carries such a fact, else its ``inputs_json`` value. Never the pre-read value where a
    post-read one exists: :func:`plan_groups` compares the fingerprint the next pass replays against the WATERMARK, and on an unmoved frame
    the replay reproduces what the read decided, not what the pre-read replay gave -- a watermark stamped with the pre-read value would read
    the group as changed on every pass that follows. A committed fact with a malformed fingerprint refuses by name."""
    watermarks: Mapping[str, Mapping[str, object]] = facts["watermarks"]   # type: ignore[assignment]
    out: List[Input] = []
    for inp in inputs:
        wm = watermarks.get(inp.identity_uri) if is_decision_input(inp) else None
        fp = str(wm.get("sha256")) if wm is not None and wm.get("sha256") else None
        if fp is None or fp == inp.sha256:
            out.append(inp)
            continue
        try:
            out.append(AnchorDecisionInput.of(decision_source_id(inp.identity_uri), fp))
        except ValueError as exc:
            raise AdapterRefusal(str(facts.get("marker")), f"the committed watermark of the anchor-decision input {inp.identity_uri} carries a malformed fingerprint: {exc}")
    return out


def reactivate_by_id(root: str, adapter: Adapter, revision_id: str, conn: Optional[sqlite3.Connection], now: Optional[str] = None,
                     producer: Optional[Mapping[str, object]] = None, budget: Optional["ReactivationBudget"] = None,
                     retirement_claim: Optional[Mapping[str, object]] = None) -> GroupResult:
    """Re-activate a PUBLISHED revision from its COMMITTED BYTES alone (the grading host's answer to a refused-known id a
    cluster read returned, `cluster-reads pull`): the marker's ``inputs_json`` and committed ``source_watermark`` facts stand in for a
    discovery (:func:`inputs_from_committed`: the file inputs with their committed facts, the anchor-decision input from the marker alone),
    the committed ``source_state`` records are re-stamped, the measurements are re-emitted
    (:func:`_reemit_committed`) beside the withdrawals of the outgoing head's keys, as one bundle ``_activations/<rev>/<stamp>/`` -- exactly
    :func:`reactivate_committed`'s bundle, with NO store read: the adapter's ``source_states`` (which opens inputs) is never called. The
    bundle reproduces the original publish's facts: its marker's ``inputs_json`` is the committed one byte for byte, each file watermark is
    re-stamped with its committed sha and the anchor-decision watermark with the COMMITTED watermark's fingerprint, the post-read one
    (:func:`_committed_watermark_inputs`; the marker's value is the pre-read one the id was minted from). The revision's STATE is resolved
    before its input facts are demanded, so a marker whose facts would refuse is still ``current`` or ``held`` when that is what it is; and a
    RETIREMENT (an EMPTY revision, ``inputs_json = {}``: a retired group that came back and vanished again mints the same id, which a shard
    refuses as known) is re-activated from the DATABASE, never from input facts (:func:`_reactivate_retirement`): before this the by-id path
    refused every retirement marker as "inputs_json is empty", which the pull turned into a refusal of the ADAPTER. A retirement is
    re-activated only against the shard's CLAIM (``retirement_claim``, :func:`retirement_claim`: what the pull transported from the shard's
    refused-known result) and only when that claim is exactly the live state -- the same accepted head, every live input declared --
    else it is WITHHELD by name (``withheld``; nothing written, the budget unspent): another writer can accept a newer revision of the
    source between the sidecar's export and the pull, and a retirement built from the live head alone would withdraw measurements the
    shard never saw and retire an input nobody declared retired. A retirement the harvester HOLDS because
    the head it was computed against is not the live one (``harvest.RETIREMENT_HEAD_KIND``, the fold-time gate: a NEW
    retirement whose head moved between the shard's export and the fold) is not ``held`` here: the shard's claim against today's head is
    what resolves it, so it goes through the claim gate and, matching, lands as a bundle computed against the live head -- which the
    harvester judges on its own premise (``extra.retirement_expects``) before anything is withdrawn.
    Nothing is written when: the revision IS the source's current one (nothing to re-activate; ``accepted`` with no files), a marker of
    it is on the log unfolded (:func:`pending_fold`), the database holds it pending / never accepted, its latest bundle is held, the
    ``budget`` is spent (deferred), or the database holds it REFUSED AT EMIT TIME with nothing committed under the id
    (``refused_standing``, :func:`_emit_time_refusal` -- the refusal stands as it was, nothing exists to re-emit, and a cluster
    shard reads the group fresh; the state is resolved BEFORE the directory check, so such an id is never refused by name as a missing
    directory, which refused the whole adapter for the window). A reverted input set that would withdraw too much refuses as
    :func:`reactivate_committed` does. An accepted revision with NO ``source_status`` row (committed before the status fact existed; :func:`_revision_has_status`) is REFUSED by
    name, not re-activated: the committed bytes carry no status facts and this path has no reader to mint them, so the bundle would make
    it the head while absent from ``v_source_status`` -- where :func:`reactivate_committed` re-reads, this path names the operator's
    ``--reread --force`` as the way forward."""
    out_dir = os.path.join(H.records_dir(root), "adapters", adapter.name, revision_id)
    rev = revision_id
    if not os.path.isdir(out_dir) and not complete_bundles(root, adapter.name, rev):
        # NOTHING COMMITTED under the id: resolve what the database says BEFORE refusing by name. A row folded from a REFUSED
        # record -- the read refused at emit time, `_refused/<rev>/` its only artifact -- is not a re-activation candidate and not a
        # refusal of the adapter: the refusal stands as it was, nothing exists to re-emit, and a cluster shard reads the group fresh
        # (ClusterRead.known no longer calls such an id known; an older shard's claim lands here). An ACCEPTED row with no directory, a
        # held row with none, or no row at all still refuses by name below: bytes the database or the caller vouches for are missing.
        standing = _emit_time_refusal(conn, rev)
        if standing is not None:
            source_id, why = standing
            return GroupResult(adapter.name, group_of_source_id(adapter.name, source_id), rev, accepted=False, refused_standing=True,
                               error=f"revision {rev} of {source_id} was refused at emit time on the grading host and never committed ({why}); "
                                     "the refusal stands -- nothing to re-activate from; a cluster shard reads the group fresh")
    try:
        facts = committed_marker_facts(out_dir)
    except AdapterRefusal as exc:
        return GroupResult(adapter.name, f"<revision {rev}>", rev, accepted=False, error=f"re-activation by id refused: {exc}")
    revision = facts["revision"]        # type: ignore[index]
    source_id = str(revision.get("source_id") or source_id_for(adapter.name, "?"))     # type: ignore[union-attr]
    group = group_of_source_id(adapter.name, source_id)
    inputs_json_ = revision.get("inputs_json")       # type: ignore[union-attr]
    if not isinstance(inputs_json_, Mapping):
        return GroupResult(adapter.name, group, rev, accepted=False,
                           error=f"re-activation by id refused: {facts['marker']}: the marker's inputs_json is malformed; nothing to re-activate from")
    retirement = not inputs_json_          # an EMPTY revision: a retirement, re-activated from the database below (never from input facts)
    now = now or R.utc_now()
    prod = dict(producer or H.harvester_producer(root))
    prod["script"] = "sisteer.registry.adapters.runner"
    ctx = AdapterContext(root=root, now=now, producer=prod, conn=conn)
    pending = pending_fold(root, adapter.name, rev, conn)
    if pending is not None:
        return GroupResult(adapter.name, group, rev, accepted=False, held=True, activation_pending=True,
                           error=f"revision {rev} is published but not ingested yet: its marker {pending} is on the record log, not folded yet; "
                                 "not re-emitted -- the fold is pending")
    state = _revision_state(conn, rev)
    if state is None and conn is not None:
        return GroupResult(adapter.name, group, rev, accepted=False, held=True,
                           error=f"revision {rev} has no source_revision row (not folded yet); not re-emitted -- the harvester folds the directory itself")
    held = _held_marker(conn, rev)
    # a RETIREMENT the harvester holds because the head it was computed against is not the live one (harvest.RETIREMENT_HEAD_KIND;
    # registry-hardening 3a, round 2) is exactly what a shard's fresh claim against today's head resolves: its re-activation goes through
    # the claim gate below and lands as a bundle computed against the live head, which the harvester judges on its own premise. Every
    # other hold (a refused or blocked constituent, a run drop, a bundle of it held) stands: the pull writes nothing for it
    head_moved_hold = retirement and held is not None and held[1] == H.RETIREMENT_HEAD_KIND
    if state == "pending" and not head_moved_hold:
        reason = _revision_refusal(conn, rev)
        return GroupResult(adapter.name, group, rev, accepted=False, held=True,
                           error=f"revision {rev} is published but not accepted (held by the harvester); not re-emitted" + (f"; held on: {reason}" if reason else ""))
    prev_rev = _current_revision(conn, source_id)
    if prev_rev == rev:
        return GroupResult(adapter.name, group, rev, accepted=True, records_written=0, withdrawals=0, files=[],
                           error=None)
    if retirement:
        # a retirement's bundle carries its own status row (_retirement_records), so the status-row rule below does not apply to it; the
        # claim is checked against the live head BEFORE the budget is spent (a withheld retirement writes nothing); the builder asks the
        # adapter only for declarations that are attributes -- a residual refusal (a store-reading callback on this unscoped host) is this
        # revision's refusal by name, never a crash of the pull
        try:
            return _reactivate_retirement(root, adapter, group, source_id, rev, prev_rev, facts, ctx, conn, claim=retirement_claim, budget=budget)
        except AdapterRefusal as exc:
            return GroupResult(adapter.name, group, rev, accepted=False, error=f"re-activation by id refused: {exc}")
    if conn is not None and not _revision_has_status(conn, source_id, rev):
        # An ACCEPTED, superseded revision with NO `source_status` row: committed before the status fact existed (the runner of that
        # day filed no status) -- the shape `strength_ladder` and `relu_native` sat in with four accepted revisions each (2026-09-18).
        # The pass's path for such a revision is a RE-READ (`reactivate_committed` -> `reread_committed`): the fresh read supplies the
        # status row. This path has no reader: it re-emits the committed measurements and re-stamps the committed facts, and a status
        # row is not among the committed bytes of such a revision, so a bundle from here would make it the head while leaving it absent
        # from `v_source_status` -- the gap the re-read closes, reopened by id. Refused BY NAME, before the budget is spent: the
        # operator's `adapters run --reread --force --adapter <a>` under REGISTRY_STORE_READ_OVERRIDE (the one grading-host store read
        # left; window.sh) reads it fresh and files the row, or the operator intervenes on the revision itself.
        return GroupResult(adapter.name, group, rev, accepted=False,
                           error=f"re-activation by id refused: revision {rev} of {source_id} is accepted but has no source_status row "
                                 "(committed before the status fact existed); a re-emission of committed bytes cannot supply the reader's status "
                                 "facts and would make it the head while absent from `v_source_status` -- a scoped fresh read (`registry adapters "
                                 f"run --reread --force --adapter {adapter.name}` under REGISTRY_STORE_READ_OVERRIDE) or an operator's "
                                 "intervention on the revision is required")
    if held is not None and "/_activations/" in held[0]:
        return GroupResult(adapter.name, group, rev, accepted=False, held=True,
                           error=f"revision {rev} is published and its re-activation bundle is held by the harvester ({held[1]}); not re-emitted; held on: {held[2]}")
    if budget is not None and not budget.take():
        return _deferred_result(adapter, group, rev, budget, "re-activation by id")
    try:
        inputs = inputs_from_committed(facts)
        watermark_inputs = _committed_watermark_inputs(facts, inputs)
    except AdapterRefusal as exc:
        return GroupResult(adapter.name, group, rev, accepted=False, error=f"re-activation by id refused: {exc}")
    comparison = inputs_dropped_comparison(adapter, group, inputs, prev_rev, _accepted_inputs(conn, prev_rev))
    declared_drops = runs_retired_of(adapter, group)
    if comparison and comparison["tripped"]:
        exc = InputDropRefusal(source_id, comparison)
        refused = _watermark_records(ctx, adapter.name, source_id, inputs, last_error=str(exc), attempt_revision_id=rev)
        path = R.write_records(os.path.join(refused_dir(root, adapter.name, rev), f"{_compact(now)}-source_watermark.jsonl"), refused)
        log.warning("inputs dropped (unexplained): re-activation by id of %s for %s refused: %s", rev, source_id, exc.why)
        return GroupResult(adapter.name, group, rev, accepted=False, error=str(exc), files=[path], inputs_dropped=True)
    prev = _current_observations(conn, source_id)
    prev_items = _current_item_keys(conn, prev_rev) if prev_rev else set()
    now_keys, now_items = committed_keys(out_dir)
    withdrawals = _withdrawal_records(ctx, source_id, rev, prev, prev_rev, prev_items, now_keys, now_items, "removed_measurement")
    committed_extra = dict(revision.get("extra") or {}) if isinstance(revision.get("extra"), Mapping) else {}   # type: ignore[union-attr]
    extra = dict(committed_extra)
    extra.update(_revision_extra(adapter, group))
    extra.update(_comparison_extra(comparison))
    extra.update(run_set_extra(prev_rev, prev.keys(), now_keys, comparison, declared_drops))
    extra["reactivated_from_committed_bytes"] = {"marker": H.logical_uri(root, str(facts["marker"])), "inputs": len(inputs)}
    rev_payload = {"revision_id": rev, "source_id": source_id, "adapter": adapter.name, "observed_utc": now,
                   "inputs_json": inputs_json(inputs), "accepted": 1, "priority": adapter_priority(adapter, group), "extra": extra}
    marker = [R.make_record("source_revision", rev_payload, ctx.producer, identity.adapter_record_id(rev_payload, "source_revision"), now)]
    marker += withdrawals
    for st in facts["states"]:         # type: ignore[union-attr]
        payload = dict(st, observed_utc=now)
        marker.append(R.make_record("source_state", payload, ctx.producer, identity.adapter_record_id(payload, "source_state"), now))
    marker += _watermark_records(ctx, adapter.name, source_id, watermark_inputs)
    bundle = os.path.join(activations_dir(root, adapter.name, rev), _compact(now))
    staged = os.path.join(root, "staging", "adapters", adapter.name, f"{rev}-activation-{_compact(now)}")
    shutil.rmtree(staged, ignore_errors=True)
    _, n_records = _reemit_committed(out_dir, staged, now, rev)
    R.write_records(os.path.join(staged, R.record_file_name(999998, "source_revision")), marker)
    files = publish_staged(staged, bundle)
    return GroupResult(adapter.name, group, rev, accepted=True, records_written=n_records, withdrawals=len(withdrawals), files=files, reactivated=True)


def retirement_claim(prev_rev: Optional[str], inputs: Sequence[str], patterns: Sequence[str]) -> Dict[str, object]:
    """The claim a CLUSTER shard attaches to a refused-known RETIREMENT (:func:`retire_vanished_group`; ``GroupResult.retirement``): the facts
    its declaration was verified against -- ``outgoing_revision_id`` (the source's accepted head in the sidecar it read; None for a source
    that never had one), ``inputs`` (every vanished file input, canonical, sorted; each one the adapter's ``inputs_retired`` covered) and
    ``patterns`` (the covering patterns, canonical). The shard summary carries it under ``refused_known[].retirement``, the pull under its
    report's ``refused_known_claims``, and :func:`_reactivate_retirement` withholds the re-activation unless the live head and inventory
    are exactly what the shard declared retired."""
    return {"outgoing_revision_id": prev_rev, "inputs": sorted(str(u) for u in inputs), "patterns": [str(p) for p in patterns]}


def _retirement_inventory(conn: Optional[sqlite3.Connection], adapter_name: str, source_id: str, prev_rev: Optional[str]) -> List[str]:
    """The inputs a retirement of ``source_id`` retires TODAY, from the database: the source's non-retired watermarks (the inventory
    :func:`plan_groups` reads) and the accepted head's inputs; canonical URIs, sorted, the synthetic anchor-decision one among them."""
    marks = _watermarks(conn, adapter_name)
    return sorted({uri for uri, mk in marks.items() if mk.source_id == source_id and not mk.retired} | set(_accepted_inputs(conn, prev_rev) or {}))


def _retirement_claim_withheld(claim: Optional[Mapping[str, object]], source_id: str, prev_rev: Optional[str], file_uris: Sequence[str]) -> Optional[str]:
    """WHY a retirement re-activated by id is WITHHELD (registry-hardening 3a, round 1), or None when the shard's ``claim``
    (:func:`retirement_claim`) is exactly the live state: the claim exists and is well-formed, its outgoing head is the source's live
    accepted head (``prev_rev``) and every live file input (``file_uris``, :func:`_retirement_inventory`) is among the inputs the shard
    declared retired. The pull's lock serializes pulls, not every writer of the database: between the sidecar's export and the pull another
    writer -- an earlier window's fold, the operator's override pass -- can accept a NEWER revision of the source with an input the shard
    never saw, and a retirement built from the live head alone would withdraw its measurements and retire that input, which nobody
    declared retired. Withheld, nothing is written, and the next window's shard judges the group against today's head: declared, it mints
    the same empty id with a matching claim; undeclared, it refuses by name."""
    if not isinstance(claim, Mapping):
        return (f"the refused-known claim for the retirement of {source_id} carries no retirement declaration (an older shard's, or none "
                "transported); a retirement is re-activated only from the owning shard's verified declaration, never inferred on the grading host -- "
                "nothing written")
    inputs = claim.get("inputs")
    if "outgoing_revision_id" not in claim or not isinstance(inputs, (list, tuple)) or not all(isinstance(u, str) for u in inputs):
        return f"the refused-known claim for the retirement of {source_id} is malformed ({claim!r}); nothing written"
    claimed_head = claim.get("outgoing_revision_id")
    if claimed_head != prev_rev:
        return (f"the head moved: the shard declared the retirement of {source_id} against head {claimed_head or '(none)'}, the live accepted "
                f"head is {prev_rev or '(none)'} (accepted since the sidecar was exported); nothing written -- the next window's shard judges "
                "the group against today's head")
    undeclared = [u for u in file_uris if u not in set(inputs)]
    if undeclared:
        return (f"the live inventory of {source_id} holds {len(undeclared)} input(s) the shard never declared retired: "
                + ", ".join(undeclared[:INPUTS_DROPPED_NAMED])
                + (f" (+{len(undeclared) - INPUTS_DROPPED_NAMED} more)" if len(undeclared) > INPUTS_DROPPED_NAMED else "")
                + "; nothing written -- the next window's shard judges the group against today's head")
    return None


def _reactivate_retirement(root: str, adapter: Adapter, group: str, source_id: str, rev: str, prev_rev: Optional[str], facts: Mapping[str, object],
                           ctx: AdapterContext, conn: Optional[sqlite3.Connection], claim: Optional[Mapping[str, object]] = None,
                           budget: Optional["ReactivationBudget"] = None) -> GroupResult:
    """Re-activate a committed RETIREMENT (an EMPTY revision, :func:`retire_vanished_group`) by id (:func:`reactivate_by_id`): the group was
    retired, came back -- its accepted revision re-activated, the measurements current again -- and vanished once more under the same
    emission version, so its deterministic empty id is KNOWN to the grading host: a cluster shard refuses to publish it again
    (:meth:`ClusterRead.refuse_known`, ``retiring``; S11) and the pull lands here. A retirement's committed bytes are a marker alone, and
    the facts a retirement publishes are the DATABASE's, not the marker's: the outgoing head's keys to withdraw (``input_vanished``), the
    watermarks of the inputs that left -- the source's non-retired watermarks, the inventory :func:`plan_groups` reads, and the head's
    inputs -- marked retired, ``found = 0`` for their state rows and for every other row of the source, the status row (ok = 0, the inputs
    under ``missing``). So the bundle is :func:`retire_vanished_group`'s records under the committed id (:func:`_retirement_records`), one
    activation bundle ``_activations/<rev>/<stamp>/`` holding its marker file alone; the committed marker's ``extra`` is carried, its
    run-set and run-drop declarations re-derived against today's head and the retirement declaration taken from the shard's CLAIM (the
    patterns it verified; the live inputs they cover), ``reactivated_from_committed_bytes`` names the marker and the head the claim named.
    Nothing of the committed marker's withdrawals or stamps is re-emitted: they named the head of THEIR day, and :func:`committed_marker_facts`
    drops retired stamps (``last_error`` set) anyway, so the generic path could never have re-derived them.

    THE CLAIM GATE (registry-hardening 3a, round 1). ``claim`` is what the shard verified (:func:`retirement_claim`, transported by the pull);
    :func:`_retirement_claim_withheld` compares it with the live state -- the same accepted head, every live file input declared -- and on
    any difference, or with no claim at all, the re-activation is WITHHELD by name (``withheld``): nothing written, the ``budget`` unspent.
    Before this the bundle was built from the live head alone, so a newer revision of the source accepted between the sidecar's export and
    the pull (another window's fold, the override pass) had its measurements withdrawn and its added input retired on a declaration the
    shard never made for it. The adapter is asked NOTHING here -- the declaration is the claim's, never ``inputs_retired`` re-run on the
    grading host (a store-listing callback in several production adapters, refused by name on an unscoped host)."""
    revision = facts["revision"]        # type: ignore[index]
    prior = _retirement_inventory(conn, adapter.name, source_id, prev_rev)
    file_uris = [u for u in prior if not is_decision_uri(u)]
    why = _retirement_claim_withheld(claim, source_id, prev_rev, file_uris)
    if why is not None:
        log.warning("retirement re-activation by id of %s for %s WITHHELD: %s", rev, source_id, why)
        return GroupResult(adapter.name, group, rev, accepted=False, withheld=True,
                           error=f"re-activation by id of the retirement {rev} of {source_id} withheld: {why}")
    if budget is not None and not budget.take():
        return _deferred_result(adapter, group, rev, budget, "re-activation by id")
    patterns = [str(p) for p in (claim.get("patterns") or [])]      # type: ignore[union-attr]
    committed_extra = dict(revision.get("extra") or {}) if isinstance(revision.get("extra"), Mapping) else {}   # type: ignore[union-attr]
    extra_base = {k: v for k, v in committed_extra.items() if k not in ("run_set", H.DECLARED_RUN_DROPS_KEY, RETIREMENT_DECLARED_KEY)}
    extra_base["reactivated_from_committed_bytes"] = {"marker": H.logical_uri(root, str(facts["marker"])), "inputs": 0,
                                                      "retirement_claim": {"outgoing_revision_id": prev_rev, "inputs": len(file_uris), "patterns": len(patterns)}}
    return _publish_retirement_bundle(root, adapter, group, source_id, rev, prev_rev, prior, file_uris, patterns, ctx, conn, extra_base)


def _publish_retirement_bundle(root: str, adapter: Adapter, group: str, source_id: str, rev: str, prev_rev: Optional[str], prior: Sequence[str],
                               declared: Sequence[str], patterns: Sequence[str], ctx: AdapterContext, conn: Optional[sqlite3.Connection],
                               extra_base: Optional[Mapping[str, object]] = None) -> GroupResult:
    """A retirement of ``source_id`` under its KNOWN empty id ``rev`` published as one marker-only ACTIVATION BUNDLE
    ``_activations/<rev>/<stamp>/`` (:func:`_retirement_records` built from the live database: ``prior`` the inventory, ``declared`` and
    ``patterns`` the declaration as the caller established it, ``extra_base`` what the marker's ``extra`` starts from) -- the shape both
    the pull's re-activation by id (:func:`_reactivate_retirement`, after its claim gate) and the grading host path of :func:`retire_vanished_group`
    for an id whose directory already exists publish, so a known retirement id is never written into its revision directory twice (the
    marker file there is append-only bytes: a second write would stack a second marker record on the first). Staged under ``staging/`` and
    landed by one rename (:func:`publish_staged`); ``reactivated`` on the result."""
    records, withdrawals = _retirement_records(ctx, adapter, group, source_id, rev, prev_rev, prior, conn, declared, patterns, cluster=None, extra_base=extra_base)
    now = ctx.now
    bundle = os.path.join(activations_dir(root, adapter.name, rev), _compact(now))
    staged = os.path.join(root, "staging", "adapters", adapter.name, f"{rev}-activation-{_compact(now)}")
    shutil.rmtree(staged, ignore_errors=True)
    R.write_records(os.path.join(staged, R.record_file_name(999998, "source_revision")), records)
    files = publish_staged(staged, bundle)
    log.info("retirement of %s published as an activation bundle under its known empty id %s (%d withdrawal(s), %d input(s) retired, computed "
             "against head %s) -- the group vanished again", source_id, rev, len(withdrawals), len(prior), prev_rev or "(none)")
    return GroupResult(adapter.name, group, rev, accepted=True, records_written=0, withdrawals=len(withdrawals), files=files, reactivated=True)


def refresh_group(root: str, adapter: Adapter, group: str, inputs: Sequence[Input], ctx: AdapterContext,
                  conn: Optional[sqlite3.Connection], full: bool = False, vanished_uris: Sequence[str] = (),
                  reread: bool = False, forced_utc: Optional[str] = None, cluster: Optional["ClusterRead"] = None,
                  refused: bool = False, budget: Optional["ReactivationBudget"] = None) -> GroupResult:
    """One comparison group: a published revision id is re-activated or (``reread``) read again for its facts; an unpublished
    one — new inputs, a bumped ``EMISSION_VERSION``, or a ``forced_utc`` nonce — is read and published fresh. A published
    directory whose marker -- its own or a complete activation bundle's -- is on the record log and NOT FOLDED yet is PENDING
    (:func:`pending_fold`) and left alone whatever selected the group; one the database holds (``pending``) is left alone too, the
    result naming the harvester's reason. ``reread`` is the pass's explicit ``--reread`` (the operator's: the one cause that re-reads
    a held revision, uncapped); ``refused`` says a REFUSED watermark stamp selected the group, which re-reads an accepted revision
    (the reader runs before the refusal clears) but never a held one -- the stamp is stale exactly when the successful read's clean
    watermarks sit in a held marker, and a fresh read of the same inputs holds again on the same constituent (the 2026-09-29
    ``regrade_twin`` spiral: 97 held revisions with a 09-17 refused stamp, re-read once per 10-minute pass, 17 passes). ``budget``
    caps the bundles this adapter's pass writes (:class:`ReactivationBudget`). A ``cluster`` shard never re-activates or re-reads:
    an id the grading host knows is REFUSED BY NAME (:meth:`ClusterRead.known`), an id already in this window's spool is left alone, and
    only an id the grading host has never seen -- or REFUSED AT EMIT TIME and never committed (nothing there to re-activate) -- is
    read and published."""
    source_id, rev = _revision_for(adapter, group, inputs, forced_utc)
    out_dir = os.path.join(H.records_dir(root), "adapters", adapter.name, rev)
    if cluster is not None:
        refused_known = cluster.refuse_known(conn, adapter.name, group, rev, source_id)
        if refused_known is not None:
            return refused_known
        if os.path.isdir(out_dir):
            return GroupResult(adapter.name, group, rev, accepted=False, held=True,
                               error=f"revision {rev} is already published in this window's spool ({out_dir}); not written again")
        return _read_and_publish(root, adapter, group, inputs, ctx, conn, rev, out_dir, staging_dir(root, adapter.name, rev), vanished_uris,
                                 forced_utc=forced_utc, cluster=cluster)
    if os.path.isdir(out_dir):
        pending = pending_fold(root, adapter.name, rev, conn)
        if pending is not None:
            # ON THE RECORD LOG, NOT YET FOLDED: the directory's marker or a complete bundle's has no delivery row. The database
            # lacking the revision's rows (its status row, its accepted marker) is not "absent" but a fold that has not happened;
            # another bundle adds nothing the log does not carry and costs the harvester a full copy of the revision. Before this
            # rule a pass re-emitted every such revision once per 10 minutes: 97 bundles per pass, four harvester cycles in a row
            # killed at their 45-minute bound, the paper frame frozen for four hours (regrade_twin, 2026-09-29). Whatever selected
            # the group -- an input, a refused stamp, --retry-refused, an explicit --reread -- waits for the fold.
            return GroupResult(adapter.name, group, rev, accepted=False, held=True, activation_pending=True,
                               error=f"revision {rev} is published but not ingested yet: its marker {pending} is on the record log, not folded yet; "
                                     "not re-emitted -- the fold is pending, and the harvester's next cycle takes it")
        state = _revision_state(conn, rev)
        if reread:
            # the operator's explicit --reread: a committed revision read again for its facts, a held one read fresh as its
            # bundle (a corrected adapter) -- the one cause that moves a held revision from the runner's side; uncapped
            return reread_committed(root, adapter, group, inputs, ctx, conn, vanished_uris)
        if state == "pending":
            # published, and the harvester holds it (a refused or blocked constituent): its watermarks never folded, so every
            # pass saw its inputs as new and re-emitted the WHOLE revision as another activation bundle that was held again —
            # the r2_export revision (176 MB) once per pass, 2026-09-17. The blocked-queue retry (a parent arriving) or a
            # `--reread` (a corrected adapter) are the two ways forward; the incremental pass writes nothing for it. The result
            # carries the hold's reason: a pass summary that counted three groups held and named no constituent sent an operator
            # hunting for a runner defect (relu_paired, 2026-09-20: `object.paper_label` refused by the live database's CHECK).
            # A REFUSED STAMP on its inputs is no cause either (2026-09-29): the stamp is the earlier refused read's, kept alive only
            # because the successful read's clean watermarks are in the held marker; a fresh read holds on the same constituent.
            reason = _revision_refusal(conn, rev)
            return GroupResult(adapter.name, group, rev, accepted=False, held=True,
                               error=f"revision {rev} is published but not accepted (held by the harvester); not re-emitted"
                                     + (" on its inputs' refused stamp (the stamp clears when the revision folds)" if refused else "")
                                     + " — the blocked-queue retry or `adapters run --reread` moves it"
                                     + (f"; held on: {reason}" if reason else ""))
        if state is None and conn is not None:
            # published, and the database has no row for it yet, with no marker pending_fold could find (a directory torn between
            # staging and publication: the harvester blocks its files until the marker arrives): the harvester's, not the pass's.
            # Before the pending rule above this branch was the whole guard for a published, not-yet-folded directory (relu_paired,
            # 2026-09-20: three revisions published 22:54Z, folded 00:48Z, re-emitted whole at 00:32Z and 00:43Z); with no database
            # at all the pass cannot tell and re-activates as before.
            return GroupResult(adapter.name, group, rev, accepted=False, held=True,
                               error=f"revision {rev} is published but not ingested yet (no source_revision row); not re-emitted — "
                                     "the harvester folds the directory itself")
        held = _held_marker(conn, rev)
        if held is not None and "/_activations/" in held[0]:
            # an ACCEPTED, non-head revision whose latest re-activation bundle the harvester holds (it would move the head while
            # dropping run ids the head carries, undeclared): re-emitting another bundle per pass would be held the same way at the cost
            # of a full copy of the revision each time; the declaration (the adapter's runs_retired, or `registry revision accept`)
            # is the way forward, and the queue row names the hold
            return GroupResult(adapter.name, group, rev, accepted=False, held=True,
                               error=f"revision {rev} is published and its re-activation bundle is held by the harvester ({held[1]}); "
                                     f"not re-emitted — a declaration moves it; held on: {held[2]}")
        if refused:
            # an ACCEPTED revision with a refused stamp on an input (a re-read of the head that refused, --retry-refused, a traversal
            # sibling's change, an input restored): a genuine re-read, so the reader runs before the refusal clears, where the
            # re-activation path would re-stamp its watermarks clean off the committed bytes without opening a store
            return reread_committed(root, adapter, group, inputs, ctx, conn, vanished_uris, budget=budget)
        return reactivate_committed(root, adapter, group, inputs, ctx, conn, vanished_uris, budget=budget)
    return _read_and_publish(root, adapter, group, inputs, ctx, conn, rev, out_dir, staging_dir(root, adapter.name, rev), vanished_uris,
                             forced_utc=forced_utc)


def reread_committed(root: str, adapter: Adapter, group: str, inputs: Sequence[Input], ctx: AdapterContext,
                     conn: Optional[sqlite3.Connection], vanished_uris: Sequence[str] = (),
                     budget: Optional["ReactivationBudget"] = None) -> GroupResult:
    """A COMMITTED revision read again (``registry adapters run --reread``): the adapter runs over the same inputs
    for its FACTS, and an activation bundle ``_activations/<rev>/<stamp>/`` publishes the read's ``source_status``
    records and a fresh ``source_revision`` marker (same revision id, a later ``observed_utc``: the head is re-observed,
    its history keeps the earlier marker). A re-read NEVER changes measurements: a revision's rows are keyed by its id
    and no withdrawal can apply inside one revision, so the read's records are dropped and any drift between the keys
    it carries and the committed rows' keys is stored in the status row (``extra.reread.key_drift``) and named in its
    ``detail`` — a reader that now emits different keys publishes a new revision, i.e. a changed input. The log grows by
    the status rows, not by the revision's size (the 2026-09-17 status backfill: 24 adapters' heads committed before
    ``source_status`` existed). A committed revision that is NOT the source's head (a reverted input) is re-activated
    the way :func:`reactivate_committed` does — its COMMITTED bytes re-emitted beside the withdrawals of the outgoing
    head's keys — with the status row added. A re-read that refuses writes the inputs' watermarks with ``last_error``
    under ``_refused/<rev>/`` and nothing else: an unaccepted marker with the head's own id would demote the head.
    A published directory whose revision was NEVER accepted (held on a refused or blocked constituent — the pre-#207
    ``r2_export`` read whose ``at_cap_share`` metric the validator bans, 2026-09-17) is the one case where the fresh
    read IS the bundle: its committed bytes are what holds it, re-emitting them would hold it again, and the same
    inputs give the same revision id, so no incremental pass could ever replace it. The bundle then carries the read's
    records (a corrected adapter's), its membership and status; the sound files the held attempt already staged fold
    with it (they share the revision), and the refused constituent stays refused by sha. An adapter whose code changed
    what it PARKS is not this function's case at all: it bumps ``EMISSION_VERSION`` and its read is a NEW
    revision (:func:`refresh_group`), never a re-read that drops its records. A ``budget`` (the incremental pass's; an
    explicit ``--reread`` passes none) is spent by the bundle this writes and defers the group when it is exhausted."""
    source_id, rev = _revision_for(adapter, group, inputs)
    if budget is not None and not budget.take():
        return _deferred_result(adapter, group, rev, budget, "re-read")
    stamp = _compact(ctx.now)
    bundle = os.path.join(activations_dir(root, adapter.name, rev), stamp)
    staged = os.path.join(root, "staging", "adapters", adapter.name, f"{rev}-reread-{stamp}")
    res = _read_and_publish(root, adapter, group, inputs, ctx, conn, rev, bundle, staged, vanished_uris, committed=True)
    res.reread = res.accepted
    res.reactivated = res.accepted
    return res


def _read_and_publish(root: str, adapter: Adapter, group: str, inputs: Sequence[Input], ctx: AdapterContext,
                      conn: Optional[sqlite3.Connection], rev: str, out_dir: str, staged: str, vanished_uris: Sequence[str] = (),
                      committed: bool = False, forced_utc: Optional[str] = None, cluster: Optional["ClusterRead"] = None) -> GroupResult:
    """Read one comparison group and publish it atomically into ``out_dir`` — a fresh revision directory, or (``committed``)
    an activation bundle of an already published revision (:func:`reread_committed`). ``forced_utc`` names the nonce a
    ``--reread --force`` read's id carries; the marker records it under ``extra.forced``. Under a relocated read root every record's
    strings are canonicalized (``IdentityMap.canonical_deep``) and a ``cluster`` read's marker carries ``extra.read_copy``."""
    source_id = source_id_for(adapter.name, group)
    imap = identity_map()
    res = GroupResult(adapter.name, group, rev, accepted=False, forced=bool(forced_utc))
    published = committed                                             # a re-read of a published directory (the refusal path's rule)
    prev_rev = _current_revision(conn, source_id)                     # the source's head, whatever tables it observes
    prev = _current_observations(conn, source_id)
    prev_items = _current_item_keys(conn, prev_rev) if prev_rev else set()
    now = ctx.now
    published_dir = os.path.join(H.records_dir(root), "adapters", adapter.name, rev)   # the committed bytes, when `committed`
    shutil.rmtree(staged, ignore_errors=True)   # a previous interrupted attempt at this very revision
    comparison: Optional[Dict[str, object]] = None
    # the FILE inputs are what the adapter reads, the artifacts register, the state rows and the inputs-dropped comparison see; the
    # synthetic anchor-decision input (at most one, attached by plan_groups) is in `rev` and `inputs_json` only
    files = [i for i in inputs if not is_decision_input(i)]
    decision_in = next((i for i in inputs if is_decision_input(i)), None)
    prev_inputs = _accepted_inputs(conn, prev_rev)
    try:
        # the inputs against the previous accepted revision's, BEFORE the read: a set that lost more than the threshold share of them,
        # none declared retired, is refused here and the stores are not read (inputs_dropped_comparison; the module docstring's rule)
        comparison = inputs_dropped_comparison(adapter, group, files, prev_rev, prev_inputs)
        if cluster is not None and comparison and int(comparison.get("inputs_dropped") or 0) > 0:
            # on a STAGED copy any dropped input is an incomplete copy, never a producer's vanished file -- no reduced-coverage
            # revision whatever the grading host's threshold admits; nothing is written and the group is recorded refused
            raise ClusterInputDropRefusal(source_id, comparison, cluster.covering_of(comparison.get("inputs_dropped_uris") or []))
        if cluster is not None:
            # every LISTING input against the pinned generation's manifest attestation of ITS directory (the canonical's whole listing at
            # the push; never a previous revision's): no attestation refuses, a differing one refuses -- the group is recorded refused
            unattested: Dict[str, Tuple[str, Optional[str], str]] = {}
            mismatched: Dict[str, Tuple[str, Optional[str], str]] = {}
            for i in files:
                if not is_listing_input(i):
                    continue
                att = cluster.attested_listing(i.path) if i.path else None
                if att is None:
                    unattested[i.identity_uri] = (i.sha256, None, i.path)
                elif listing_fingerprint(att) != i.sha256:
                    mismatched[i.identity_uri] = (i.sha256, listing_fingerprint(att), i.path)
            if unattested:
                raise ClusterListingRefusal(source_id, "unattested", unattested)
            if mismatched:
                raise ClusterListingRefusal(source_id, "mismatch", mismatched)
        if comparison and comparison["tripped"]:
            raise InputDropRefusal(source_id, comparison)
        declared_drops = runs_retired_of(adapter, group)      # the reader's run-drop declaration; a malformed one refuses by name
        seq = 0
        observed: List[Tuple[str, Dict[str, object]]] = []
        item_keys: List[Dict[str, object]] = []
        n_records = 0
        batch_paths: List[str] = []
        # artifacts for every FILE input (every input registered); a synthetic input -- a directory listing (base.ListingInput) -- is no
        # file: it is read, watermarked and stated, never registered
        art_batch = RecordBatch(source_id, group)
        for i in files:
            if is_synthetic_uri(i.identity_uri):
                continue
            art_batch.add("artifact", i.artifact_payload(), ctx.producer, now, observe=False)
        seq += 1
        batch_paths.append(R.write_records(os.path.join(staged, R.record_file_name(seq, "artifact")), art_batch.records))
        n_records += len(art_batch.records)
        tally = _ReadTally()
        ctx.begin_group(group)                   # the anchor decision log of THIS read starts empty
        for batch in _frame_guarded(adapter.read(ctx, group, files), source_id):     # an unavailable frame refuses the GROUP, never the pass
            by_type: Dict[str, List[Dict[str, object]]] = {}
            for rec in batch.records:
                rt = str(rec["record_type"])
                if imap:
                    # a record never spells the read root (identity vs addressing): the payload is canonicalized BEFORE its deterministic
                    # id is minted, so the cluster's record_id is the grading host's for the same facts (a payload nothing in which spelled the
                    # root comes back as the same object: its id, minted from the same facts, stands)
                    canon = imap.canonical_deep(rec["payload"])
                    if canon is not rec["payload"]:
                        rec["payload"] = canon
                        rec["record_id"] = identity.adapter_record_id(canon, rt)
                rec["payload"] = sanitise_free_form(rt, rec["payload"])    # free-form fields JSON-safe; typed values untouched
                if rt in R.ITEM_SCALE_RECORD_TYPES:
                    rec["payload"]["source_revision_id"] = rev             # the bundle's revision is authoritative: stamped, never trusted
                    if rt != "source_status":                                  # the status row is the runner's own (below), no item mark
                        rec["payload"].setdefault("withdrawn_by_revision_id", None)
                by_type.setdefault(rt, []).append(rec)
                tally.note(rec)
            for rt, recs in by_type.items():
                seq += 1
                gz = rt in R.ITEM_SCALE_RECORD_TYPES
                batch_paths.append(R.write_records(os.path.join(staged, R.record_file_name(seq, rt, gz=gz)), recs, gz=gz))
                n_records += len(recs)
            observed += batch.observation_keys
            item_keys += batch.item_keys
        # the read's anchor decision set: every comparison `choose` decided for this group, parked ones included; the
        # marker records it whole, the synthetic input's watermark carries ITS fingerprint (the post-read one), and a drift against the
        # pre-read fingerprint the revision id was minted from is a fact when no file input moved (a file change may add comparisons)
        decisions = list(ctx.decision_log)
        pre_fp = decision_in.sha256 if decision_in is not None else None
        decision_block = decisions_block(decisions, source_id, pre_read_fingerprint=pre_fp) if decisions else None
        anchor_drift: Optional[Dict[str, object]] = None
        if decision_block is not None and pre_fp is not None and decision_block["fingerprint"] != pre_fp:
            prev_files = {u: s for u, s in (prev_inputs or {}).items() if not is_decision_uri(u)}
            if prev_files == inputs_json(files):
                head_block = _head_decisions(conn, [source_id]).get(source_id) or {}
                anchor_drift = {"pre_read_fingerprint": pre_fp, "post_read_fingerprint": decision_block["fingerprint"],
                                "n_comparisons_pre": head_block.get("n_comparisons"), "n_comparisons_post": decision_block["n_comparisons"]}
                log.warning("anchor decision drift: %s read of %s (revision %s) decided %s comparison(s) to fingerprint %s where the pre-read "
                            "evaluation of the head's recorded set gave %s, with no file input changed", adapter.name, source_id, rev,
                            decision_block["n_comparisons"], decision_block["fingerprint"][:16], pre_fp[:16])
        # membership
        by_table: Dict[str, List[str]] = {}
        for table, key in observed:
            by_table.setdefault(table, []).append(identity.canonical_json(key))
        now_keys = {k for keys in by_table.values() for k in keys}
        now_items = {identity.canonical_json(k) for k in item_keys}
        reread_facts: Optional[Dict[str, object]] = None
        if committed and not _revision_accepted(conn, rev):
            # the directory's revision was never accepted (held): the FRESH read is published — see reread_committed
            reread_facts = {"of_revision": rev, "head_revision": prev_rev, "reactivated": False, "fresh_read": True,
                            "key_drift": {"keys_absent_from_read": len(set(prev) - now_keys), "keys_new_in_read": len(now_keys - set(prev)),
                                          "items_absent_from_read": len(prev_items - now_items), "items_new_in_read": len(now_items - prev_items)}}
            committed = False
        if committed:
            # the read supplied its facts; its records are NOT the bundle's (see reread_committed). What the bundle carries:
            # the status row and the marker — plus, for a revision that is not the head, the committed bytes re-activated
            for path in batch_paths:
                os.remove(path)
            head_keys, head_items = (set(prev), prev_items) if prev_rev == rev else committed_keys(published_dir)
            set_keys = head_keys                                  # the bundle's membership is the committed revision's, not the read's
            reread_facts = {"of_revision": rev, "head_revision": prev_rev, "reactivated": prev_rev != rev,
                            "key_drift": {"keys_absent_from_read": len(head_keys - now_keys), "keys_new_in_read": len(now_keys - head_keys),
                                          "items_absent_from_read": len(head_items - now_items), "items_new_in_read": len(now_items - head_items)}}
            obs_records: List[Dict[str, object]] = []
            if prev_rev != rev:
                _, n_records = _reemit_committed(published_dir, staged, now, rev, skip=_REEMIT_SKIP + ("source_status",))
                withdrawals = _withdrawal_records(ctx, source_id, rev, prev, prev_rev, prev_items, head_keys, head_items, "removed_measurement")
            else:
                n_records = 0
                withdrawals = []
            seq = 999996           # the status row and the marker sort after every re-emitted file (their names are the revision's own)
        else:
            obs_records = []
            for table, keys in by_table.items():
                payload = {"source_id": source_id, "revision_id": rev, "table_name": table, "observation_keys": sorted(set(keys))}
                obs_records.append(R.make_record("observation", payload, ctx.producer, identity.adapter_record_id(payload, "observation"), now))
            withdrawals = _withdrawal_records(ctx, source_id, rev, prev, prev_rev, prev_items, now_keys, now_items, "removed_measurement")
            set_keys = now_keys
        # the read's status: one row per paper source string the group reports under, tagged with this revision, so it is
        # published with the batches and visible exactly when they are
        status = _source_status_records(ctx, adapter, group, source_id, rev, [i.identity_uri for i in files], tally, reread=reread_facts, inputs=files,
                                        anchor_drift=anchor_drift)
        res.emitted_nothing = any(EMITTED_NOTHING_KEY in ((r.get("payload") or {}).get("extra") or {}) for r in status)
        seq += 1
        R.write_records(os.path.join(staged, R.record_file_name(seq, "source_status")), status)
        n_records += len(status)
        rev_payload = {"revision_id": rev, "source_id": source_id, "adapter": adapter.name, "observed_utc": now,
                       "inputs_json": inputs_json(inputs), "accepted": 1, "priority": adapter_priority(adapter, group),
                       "extra": dict(_revision_extra(adapter, group, forced_utc, anchor_decisions=decision_block), **_comparison_extra(comparison),
                                     **run_set_extra(prev_rev, prev.keys(), set_keys, comparison, declared_drops), **_marker_extra(cluster))}
        rev_record = R.make_record("source_revision", rev_payload, ctx.producer, identity.adapter_record_id(rev_payload, "source_revision"), now)
        # the read's state rows, plus `found = 0` for every row of the source this read does not restate (the state-row rule, module docstring:
        # the rows keyed on an input that left discovery while others stayed, a key an earlier read's ledger used)
        states = _source_state_records(ctx, adapter, files)
        states += _cleared_state_records(ctx, conn, source_id, _state_keys(states))
        states += _vanished_input_state_records(ctx, conn, [u for u in vanished_uris if not is_decision_uri(u)], _state_keys(states))
        marks = _watermark_records(ctx, adapter.name, source_id, files, records_written=n_records)
        if decision_block is not None:
            # the decision input's watermark is the POST-read fingerprint: the next pass's pre-read replays the set this marker records
            # and compares with it (equal on an unmoved frame -> quiet). Written on the first read under this code too, when no decision
            # input was in `inputs` yet. A read that decided nothing writes none: a stale one is a gone input the next pass retires.
            marks += _watermark_records(ctx, adapter.name, source_id, [AnchorDecisionInput.of(source_id, str(decision_block["fingerprint"]))],
                                        records_written=n_records)
        marks += _retire_watermarks(ctx, conn, adapter.name, source_id, vanished_uris, rev)
        seq += 1
        R.write_records(os.path.join(staged, R.record_file_name(seq, "source_revision")),
                        [rev_record] + obs_records + withdrawals + states + marks)
        # the whole group read: ONE directory rename publishes batches and marker together
        res.files = publish_staged(staged, out_dir)
        res.accepted = True
        res.records_written = n_records
        res.withdrawals = len(withdrawals)
    except AdapterRefusal as exc:
        shutil.rmtree(staged, ignore_errors=True)   # nothing of the refused revision reaches the log
        res.error = str(exc)
        if isinstance(exc, (ClusterInputDropRefusal, ClusterListingRefusal)):
            # the staged copy is incomplete (an input the generation lacks, a listing the grading host did not attest): no revision, no marker, no
            # watermark reaches the window's spool (the group is recorded refused; no grading host read follows)
            res.inputs_dropped = True
            log.warning("staged copy incomplete: %s read of %s refused on the cluster before any store was read: %s", adapter.name, source_id, exc.why)
            return res
        res.inputs_dropped = isinstance(exc, InputDropRefusal)
        if res.inputs_dropped:
            log.warning("inputs dropped (unexplained): %s read of %s refused before any store was read: %s", adapter.name, source_id, exc.why)
        res.anchor_unavailable = isinstance(exc, AnchorCurrencyRefusal)
        if res.anchor_unavailable:
            # an unavailable frame is a fault of the pass's ENVIRONMENT (a pre-0021 database, a sidecar exported without its currency table,
            # no connection), not a judgment about the group's inputs: no refused marker, no watermark stamp, no state row -- on the grading
            # host and on a shard alike. The result names the group and the cause for the pass report, the head stands, and the next pass
            # retries the group under the selection that chose it (its inputs are still new or changed) with no `--retry-refused`; a stamp
            # would have parked a healthy group behind an operator's flag for a fault that was never the group's
            log.error("anchor currency unavailable: %s read of %s refused by name, nothing written (the head stands; the next pass retries): %s",
                      adapter.name, source_id, exc.why)
            return res
        # the attempt's inventory WITHOUT a --reread --force nonce: the id the next pass computes over the same inputs
        window = (cluster.read_copy or {}).get("window") if cluster is not None else None
        refused = _watermark_records(ctx, adapter.name, source_id, inputs, last_error=str(exc), attempt_revision_id=_revision_for(adapter, group, inputs)[1],
                                     attempt_window=(str(window) if window else None))
        if published:
            # the revision is published and (usually) the head: its id must not carry an unaccepted marker. The failure is
            # the watermarks' last_error (the status page's adapter-pass rows) and this result; the accepted read stands.
            res.files = [R.write_records(os.path.join(refused_dir(root, adapter.name, rev), f"{_compact(now)}-source_watermark.jsonl"), refused)]
        else:
            extra = dict(_revision_extra(adapter, group, forced_utc), **_marker_extra(cluster))
            if res.inputs_dropped:
                extra[INPUTS_DROPPED_KEY] = dict({k: v for k, v in exc.comparison.items() if k != "inputs_dropped_uris"},
                                                 rule=INPUTS_DROPPED_RULE)   # the whole comparison, on the record (the uris ride the paths)
            rev_payload = {"revision_id": rev, "source_id": source_id, "adapter": adapter.name, "observed_utc": now,
                           "inputs_json": inputs_json(inputs), "accepted": 0, "refusal": str(exc),
                           "extra": extra}    # names the paper sources: a first-read refusal is visible by name
            res.files = [R.write_records(os.path.join(refused_dir(root, adapter.name, rev), f"{_compact(now)}-source_revision.jsonl"),
                                         [R.make_record("source_revision", rev_payload, ctx.producer,
                                                        identity.adapter_record_id(rev_payload, "source_revision"), now)] + refused)]
    return res


def _retirement_records(ctx: AdapterContext, adapter: Adapter, group: str, source_id: str, rev: str, prev_rev: Optional[str],
                        prior_uris: Sequence[str], conn: Optional[sqlite3.Connection], declared: Sequence[str], patterns: Sequence[str],
                        cluster: Optional["ClusterRead"] = None,
                        extra_base: Optional[Mapping[str, object]] = None) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """The records of a retirement of ``source_id`` under its EMPTY revision ``rev`` -- :func:`retire_vanished_group`'s fresh retirement and
    the by-id re-activation's (:func:`_reactivate_retirement`) alike: the marker, the ``input_vanished`` withdrawals of the head's current
    keys (item keys included), ``found = 0`` for every vanished input's state rows under each (source, json_path) it was observed at and for
    every other row of the source (the state-row rule, module docstring), the watermarks of ``prior_uris`` marked retired, the status row
    (ok = 0, the inputs under ``missing``); ``(records, withdrawals)``. ``prior_uris`` are the inventory's canonical URIs (the synthetic
    anchor-decision one retires with the files but is named nowhere and gets no state row). The marker's ``extra`` starts from
    ``extra_base`` (a re-activation carries the committed marker's) and names the run drop (every run, ``"*"``; :func:`run_set_extra`),
    the shard's ``read_copy``, the PREMISE the retirement was computed from (``extra.retirement_expects``, ``harvest.RETIREMENT_EXPECTS_KEY``:
    ``prev_rev`` and the file inputs it retires -- the harvester folds the marker only while that is still the live state, else holds it
    by name, ``harvest.RETIREMENT_HEAD_KIND``) and -- when the adapter's ``inputs_retired`` covers
    every vanished input (:func:`declared_retirements`) -- the declaration (``extra.retirement_declared``: the patterns that matched and
    the inputs they covered), with ``runner:inputs_retired`` beside ``runner:input_vanished`` in the run-drop declaration. ``declared`` (the canonical input URIs the adapter's ``inputs_retired``
    covers) and ``patterns`` (the covering patterns) are the declaration AS THE CALLER ESTABLISHED IT -- :func:`declared_retirements` on a
    fresh retirement, the shard's or the grading host's; the shard's transported claim on a re-activation by id -- so this builder asks the
    adapter nothing: ``inputs_retired`` lists or opens a store in production adapters (``gsm8k_p200`` plans its tree, ``degeneracy_union``
    and the cellstore readers list their directories), and on the unscoped grading host such a call refused by name out of the pull."""
    now = ctx.now
    file_uris = sorted(u for u in prior_uris if not is_decision_uri(u))
    covered = set(declared)
    declared = [u for u in file_uris if u in covered]
    undeclared = [u for u in file_uris if u not in covered]
    prev = _current_observations(conn, source_id)
    prev_items = _current_item_keys(conn, prev_rev) if prev_rev else set()
    withdrawals = _withdrawal_records(ctx, source_id, rev, prev, prev_rev, prev_items, set(), set(), "input_vanished")
    states: List[Dict[str, object]] = []
    for uri in file_uris:
        paths = [(str(r[0]), str(r[1])) for r in conn.execute(
            "SELECT DISTINCT source_id, json_path FROM source_state WHERE input_uri = ?", (uri,)).fetchall()] if conn is not None else []
        for sid, jpath in (paths or [(source_id, "$")]):
            states.append(_vanished_state_record(ctx, sid, uri, jpath))
    # every other row of the source itself -- a `$.row_refusals` ledger keyed on the group, which no input URI names -- is cleared with it
    # (the state-row rule, module docstring); the rows above count as stated, so none is written twice
    states += _cleared_state_records(ctx, conn, source_id, _state_keys(states))
    marks = _retire_watermarks(ctx, conn, adapter.name, source_id, prior_uris, rev)
    # the adapter's declaration, when it covers every vanished input (on a shard the fact that admitted the retirement; on the grading host a
    # fact recorded): named on the marker beside the run-drop declaration, whose reasons gain the inputs_retired one (run_set_extra)
    declaration = ({"declared_by": "adapter:inputs_retired", "patterns": list(patterns), "inputs": list(declared)}
                   if declared and not undeclared else None)
    extra: Dict[str, object] = dict(extra_base or {})
    extra.update(_revision_extra(adapter, group))
    # the empty revision declares every run it drops: the runner wrote it on purpose, for a reason it names
    extra.update(run_set_extra(prev_rev, prev.keys(), (), ({"inputs_retired": len(declared)} if declaration else None), None, vanished=True))
    extra.update(_marker_extra(cluster))
    if declaration is not None:
        extra[RETIREMENT_DECLARED_KEY] = declaration
    # the PREMISE the retirement was computed from (registry-hardening 3a): the head whose keys it withdraws and every file
    # input it retires -- what the harvester compares with the live state inside the activation transaction (Ingestor._retirement_head_moved)
    # and holds the marker on when another writer moved the head, or added an input, between this write and the fold
    extra[H.RETIREMENT_EXPECTS_KEY] = {"outgoing_revision_id": prev_rev, "inputs": list(file_uris)}
    rev_payload = {"revision_id": rev, "source_id": source_id, "adapter": adapter.name, "observed_utc": now, "inputs_json": {}, "accepted": 1,
                   "priority": adapter_priority(adapter, group), "extra": extra}
    rev_record = R.make_record("source_revision", rev_payload, ctx.producer, identity.adapter_record_id(rev_payload, "source_revision"), now)
    # the status of the empty revision: ok = 0, every prior input under `missing`, so the gap map says WHY the source is empty
    status = _source_status_records(ctx, adapter, group, source_id, rev, file_uris, None, missing=file_uris,
                                    detail=f"{adapter.name}: every input of {source_id} left discovery ({len(file_uris)} input(s): "
                                           + ", ".join(file_uris) + "); its keys are withdrawn (input_vanished) and nothing stands in for them"
                                           + (f"; the retirement is declared by the adapter's inputs_retired ({len(patterns)} pattern(s))" if declaration else ""))
    return [rev_record] + withdrawals + states + marks + status, withdrawals


def retire_vanished_group(root: str, adapter: Adapter, group: str, prior_uris: Sequence[str], ctx: AdapterContext,
                          conn: Optional[sqlite3.Connection], cluster: Optional["ClusterRead"] = None) -> Optional[GroupResult]:
    """A comparison group whose EVERY input has left discovery: commit an accepted EMPTY revision that
    withdraws each previously current key (``input_vanished``, item keys included), record ``source_state
    found = 0`` for every vanished input under each (source, json_path) it was observed at, and retire the
    watermarks. Idempotent: once the empty revision is current, later passes write nothing. ``group`` is spelled the way the adapter
    ADDRESSES it (:func:`prior_group_spelling`: the canonical spelling under the home tree a shard lays out as built; the
    re-pointed root's under a relocated adapter) -- the spelling its ``inputs_retired`` is asked with; ``prior_uris`` are the
    watermarks' canonical ones.

    ON A CLUSTER SHARD A RETIREMENT IS DECLARED OR REFUSED (registry-hardening 3a). On a staged copy a group whose every input is absent
    is an INCOMPLETE COPY -- unless the adapter DECLARES every vanished input retired (``inputs_retired``;
    :func:`declared_retirements`, the one matching rule :func:`inputs_dropped_comparison` applies: canonical URIs against the canonical
    declaration, exact or ``fnmatch``), the module docstring's rule that a deliberate retirement is declared, never inferred from the
    read. Every input declared: the owning shard commits the retirement exactly as the grading host path does -- the sidecar carries the head's
    observations and item keys, the vanished inputs' state rows and the watermarks these queries read (``reader_state``) -- with the
    declaration on the marker (``extra.retirement_declared``: the patterns that matched and the inputs they covered; the run-drop
    declaration names ``runner:inputs_retired`` beside ``runner:input_vanished``), after the known-id check every shard publication passes
    (:meth:`ClusterRead.refuse_known`, ``retiring``; S11: an id already on the grading host -- folded and superseded, or pulled and not folded
    yet -- is refused by name and nothing is published). Any input UNDECLARED: the shard REFUSES the group by name -- each undeclared input
    with the staged component that would have covered it -- and publishes nothing; the refusal is recorded by name, and the retirement is
    the adapter's to declare or the grading host's (the operator-override pass under REGISTRY_STORE_READ_OVERRIDE). Before this rule every
    window refused ``overrefusal_items``' two groups a producer had moved (2026-10-07) identically, and nothing could retire them on a
    grading host that reads no store (C6). The grading host path never required a declaration and still does not; it records one on the marker when
    the adapter made it, so a retirement marker says the same thing wherever it was written. A retirement whose id the grading host already KNOWS
    -- the group came back and vanished again -- is refused by name on the shard, the shard's CLAIM beside the refusal
    (``GroupResult.retirement``, :func:`retirement_claim`: the head it compared against, the vanished inputs, the covering patterns), and
    re-activated by the pull from the database (:func:`_reactivate_retirement`) only when that claim is exactly the live state; one merged
    on the grading host and not folded yet is ``held`` there, never a refusal. Every retirement marker names the PREMISE it was computed from
    (``extra.retirement_expects``: the head whose keys it withdraws, the inputs it retires), and the harvester folds it only while that is
    still the live state -- a head that moved between the shard's export and the fold holds the marker by name (``harvest.RETIREMENT_HEAD_KIND``),
    nothing withdrawn, and the next window's claim resolves it. On the grading host a known id -- its directory or a bundle on the
    log -- is never written into its directory again: a pending fold is waited for, else the retirement is published as an activation
    bundle (:func:`_publish_retirement_bundle`)."""
    source_id, rev = _revision_for(adapter, group, [])
    prev_rev = _current_revision(conn, source_id)
    if prev_rev == rev:
        return None
    # the group's synthetic anchor-decision watermark retires with its files; it is no file, so it is named nowhere and gets no
    # state row
    file_uris = sorted(u for u in prior_uris if not is_decision_uri(u))
    declared, undeclared, patterns = declared_retirements(adapter, group, file_uris)
    if cluster is not None:
        if undeclared:
            covering = cluster.covering_of(undeclared)
            named = [f"{u} [{covering.get(u) or 'no staged component covers it'}]" for u in undeclared[:INPUTS_DROPPED_NAMED]]
            more = len(undeclared) - len(named)
            which = (f"Missing ({len(declared)} of {len(file_uris)} declared retired by inputs_retired; {len(named)} of {len(undeclared)} undeclared named): "
                     if declared else "Missing: ")
            return GroupResult(adapter.name, group, rev, accepted=False, inputs_dropped=True,
                               error=f"{source_id}: {STAGED_COPY_INCOMPLETE_KEY}: every input of the group ({len(file_uris)}) is absent from this read on the cluster; "
                                     "a retirement is declared by the adapter's inputs_retired, else a retirement is the grading host's -- nothing published here. "
                                     + which + "; ".join(named) + (f" (+{more} more)" if more > 0 else ""))
        known = cluster.refuse_known(conn, adapter.name, group, rev, source_id, retiring=True)
        if known is not None:
            # the shard's CLAIM rides the refusal (registry-hardening 3a, round 1): the head it compared against, the inputs it saw vanish
            # (every one declared) and the patterns that covered them -- what the pull's re-activation by id checks against the LIVE head
            # before it withdraws anything (_reactivate_retirement); the result stays a refused-known one and nothing is published here
            return replace(known, retirement=retirement_claim(prev_rev, file_uris, patterns))
    out_dir = os.path.join(H.records_dir(root), "adapters", adapter.name, rev)
    if cluster is None and (os.path.isdir(out_dir) or complete_bundles(root, adapter.name, rev)):
        # THE GRADING HOST PATH OF A KNOWN ID (registry-hardening 3a, round 2): the empty id is already on the record log -- the group was retired,
        # came back, and vanished again under the same emission version; or its directory is held by the harvester (its premise not the
        # live state). The revision directory's marker file is append-only bytes, so writing the retirement there again would stack a
        # second marker record on the first; a fold still pending on the log is waited for (nothing re-emitted, as the group path does),
        # and otherwise the retirement is published as an ACTIVATION BUNDLE computed against today's head -- the shape the pull's
        # re-activation by id publishes -- for the harvester to judge on its own premise
        pending = pending_fold(root, adapter.name, rev, conn)
        if pending is not None:
            return GroupResult(adapter.name, group, rev, accepted=False, held=True, activation_pending=True,
                               error=f"the retirement {rev} of {source_id} is published but not ingested yet: its marker {pending} is on the record log, "
                                     "not folded yet; not re-emitted -- the fold is pending")
        return _publish_retirement_bundle(root, adapter, group, source_id, rev, prev_rev, prior_uris, declared, patterns, ctx, conn)
    records, withdrawals = _retirement_records(ctx, adapter, group, source_id, rev, prev_rev, prior_uris, conn, declared, patterns, cluster=cluster)
    path = R.write_records(os.path.join(out_dir, R.record_file_name(1, "source_revision")), records)
    return GroupResult(adapter.name, group, rev, accepted=True, records_written=0, withdrawals=len(withdrawals), files=[path])


@dataclass
class GroupPlan:
    """What one adapter's pass will do, decided BEFORE any store is read (:func:`plan_groups`): the present comparison groups with
    their inputs in discovery order; the groups to read (``changed``) and the inputs each lost (``gone_in_group``); the present
    groups with a REFUSED stamp on an input (``refused``: a committed revision among them is READ again through
    :func:`reread_committed` whatever selected it — a refused watermark turns clean only after the reader ran and succeeded, never
    off the committed bytes); the prior-only groups whose every input left discovery (``vanished``: retired with ``input_vanished``
    withdrawals -- on a cluster shard only when the adapter declares every one of them retired, else refused by name;
    :func:`retire_vanished_group`); the refused groups left standing; and how many inputs read as changed. Every group is keyed by the
    spelling the adapter ADDRESSES it under -- what ``comparison_group`` makes of ``Input.uri``: the canonical spelling under the home
    tree a shard lays out (as built), the re-pointed root's under a relocated adapter. A prior-only group named by a watermark
    takes the key ``groups`` already carries when its source is present (so ``changed`` and ``gone_in_group`` are keyed by ``groups``'
    own keys), else :func:`prior_group_spelling` -- never ``IdentityMap.local``, which under the home tree is the generation path;
    :func:`deal_shards` splits the sorted union of ``groups`` and ``vanished`` by canonical identity. The reading loop in
    :func:`_run_adapter` decides nothing, and indexes ``groups`` by every key of ``changed``: :func:`check_plan_invariants` refuses a
    plan whose ``changed`` or ``gone_in_group`` names a key ``groups`` lacks, or whose ``vanished`` names one it has, by name."""
    groups: Dict[str, List[Input]] = field(default_factory=dict)
    changed: Set[str] = field(default_factory=set)
    refused: Set[str] = field(default_factory=set)
    gone_in_group: Dict[str, Set[str]] = field(default_factory=dict)
    vanished: Dict[str, Set[str]] = field(default_factory=dict)
    refused_standing: List[str] = field(default_factory=list)
    inputs_changed: int = 0
    decision_changed: Set[str] = field(default_factory=set)      #: of ``changed``, the groups whose synthetic anchor-decision input moved
    preserved: Dict[str, str] = field(default_factory=dict)      #: present groups the pass leaves exactly as they stand, with the cause: their
                                                                 #: heads recorded a decision set and the pass's anchor currency could not be
                                                                 #: loaded (never in ``changed``; their decision watermarks are not gone inputs)


def plan_groups(adapter: Adapter, inputs: Sequence[Input], marks: Mapping[str, Watermark], version: str,
                conn: Optional[sqlite3.Connection], full: bool = False, reread: bool = False, retry_refused: bool = False,
                decision_inputs: Optional[Mapping[str, Input]] = None, preserved: Optional[Mapping[str, str]] = None) -> GroupPlan:
    """Decide which comparison groups of ``adapter`` this pass reads, before any store is opened.

    A group is read when an input of it is NEW (no watermark), CHANGED (its ``(sha256, size, mtime)`` differs from its watermark),
    RETIRED and back (an ``input vanished`` watermark whose input is in discovery again) or GONE (a watermark of the group names an
    input discovery no longer does); when the pass is ``full`` or ``reread``; or when its accepted head was minted under another
    ``EMISSION_VERSION`` (:func:`_head_versions`). A group whose EVERY input left discovery is ``vanished`` (retired by the
    caller with ``input_vanished`` withdrawals; a discovery FAILURE never reaches here).

    ``decision_inputs`` (:func:`anchor_decision_inputs`) maps a group, in the adapter's local spelling, to its synthetic
    anchor-decision input: it is appended to THAT group's inputs after the file inputs are grouped -- never routed through
    ``adapter.comparison_group``, which on a path-grouped adapter would mint a phantom group from the URI -- its URI joins the
    discovered set, and the same watermark rule applies, so a moved fingerprint marks exactly its own group changed
    (``decision_changed``). A decision input of a group with no file input this pass is dropped here: that group is the vanished
    path's, and its decision watermark retires with its files.

    ``preserved`` (group -> cause) names the present groups whose heads recorded a decision set that THIS pass cannot re-evaluate (the
    anchor currency could not be loaded, :func:`anchor_decision_inputs`): each is left exactly as it stands -- never in ``changed``
    whatever selected it (a file change, a gone input, a bump, ``--full``, ``--reread``), its decision watermark NOT a gone input (with
    no synthetic input in discovery it would read as one and select the group for a read the frame cannot serve), named in
    ``GroupPlan.preserved``. A ``reader_traversal`` adapter preserves every group when it preserves one: its one shared RNG runs over
    the whole traversal, so a partial read would change its numbers.

    THE STANDING-REFUSAL RULE (S2, S3). A group whose last read REFUSED is read again ONLY on an explicit cause: the revision id
    this discovery would mint — the group's present inputs, their shas, the adapter's ``version`` — differs from the one the refused
    read computed (``Watermark.attempt_revision_id`` on its stamps), which is every new fact in both directions: an input restored
    (even with its old bytes and mtime), added, gone or rewritten, or a version bumped since the refusal; or the pass is ``full``,
    ``reread`` or ``retry_refused`` (the flag a landing window passes once, after a reader fix lands; the quiet timer never).
    Whatever selects it, a group with a refused stamp is named (``refused``) so a committed revision among them is genuinely
    re-read. Otherwise the group is left standing and named (``refused_standing``) — its gone
    inputs included, since the attempt's id says they were absent then too: a refused watermark by itself is not new information,
    and re-reading it every pass kept ``sampled_static`` (a ``reader_traversal`` adapter, for which one changed group re-reads every
    group) at a full read per pass while four refusals stood. A stamp filed before the rule carries no attempt id and reads once
    more, which stamps it. No retry on time, and none on the ledger head."""
    plan = GroupPlan()
    imap = identity_map()
    unchanged_refused: Set[str] = set()          # groups with a refused stamp on an otherwise unchanged input: standing candidates
    discovered_uris: Set[str] = set()
    for inp in inputs:
        g = adapter.comparison_group(inp)                 # the adapter's LOCAL spelling; the watermarks are compared canonically
        plan.groups.setdefault(g, []).append(inp)
        discovered_uris.add(inp.identity_uri)
        mark = marks.get(inp.identity_uri)
        if mark is not None and mark.refused:
            plan.refused.add(g)
        if full or mark is None or mark.retired or mark.stat != watermark_of(inp) or (retry_refused and mark.refused):
            plan.inputs_changed += 1
            plan.changed.add(g)
        elif mark.refused:
            unchanged_refused.add(g)
    for g, dinp in sorted((decision_inputs or {}).items()):
        if g not in plan.groups or not is_decision_input(dinp):
            continue
        plan.groups[g].append(dinp)
        discovered_uris.add(dinp.identity_uri)
        mark = marks.get(dinp.identity_uri)
        if mark is not None and mark.refused:
            plan.refused.add(g)
        if full or mark is None or mark.retired or mark.stat != watermark_of(dinp) or (retry_refused and mark.refused):
            plan.inputs_changed += 1
            plan.changed.add(g)
            if mark is None or mark.retired or mark.stat != watermark_of(dinp):
                plan.decision_changed.add(g)
        elif mark.refused:
            unchanged_refused.add(g)
    # a refusal stands while discovery would mint the very revision the refused read computed (the standing-refusal rule above)
    standing: Set[str] = set()
    for g in unchanged_refused - plan.changed:
        rev_now = revision_id_for(source_id_for(adapter.name, g), plan.groups[g], version)
        # the watermarks are keyed by CANONICAL identity (every other lookup here is): under a relocated root i.uri is the local spelling
        if {marks[i.identity_uri].attempt_revision_id for i in plan.groups[g] if marks[i.identity_uri].refused} == {rev_now}:
            standing.add(g)
        else:
            plan.changed.add(g)
    # discovery subtracted from the prior inventory (retired watermarks are not inventory): a group that lost
    # SOME inputs is changed and retires their watermarks — unless it stands refused, its attempt id having recorded the
    # absence; a group that lost EVERY input is retired with input_vanished withdrawals.
    prior_by_source: Dict[str, Set[str]] = {}
    for uri, mark in marks.items():
        if not mark.retired:
            prior_by_source.setdefault(mark.source_id, set()).add(uri)
    # the present groups by CANONICAL source id -> the key `groups` carries for each: the adapter's own spelling, whatever the layout
    # (the canonical one under the home tree; the re-pointed root's under a relocated adapter), never re-derived through the map
    present_by_source = {source_id_for(adapter.name, g): g for g in plan.groups}
    for sid, uris in prior_by_source.items():
        gone = uris - discovered_uris
        if not gone:
            continue
        if sid not in present_by_source:
            # a prior-only group: nothing in discovery spells it, so its spelling follows the adapter's roots (prior_group_spelling)
            plan.vanished[prior_group_spelling(adapter, group_of_source_id(adapter.name, sid), imap)] = uris
            continue
        g = present_by_source[sid]
        if g not in standing:
            plan.changed.add(g)
            plan.gone_in_group[g] = gone
    # a group whose inputs all read as processed but whose accepted head was minted under another EMISSION_VERSION —
    # the adapter's code changed what it parks or spells and bumped — is changed too. Keyed on the version the head's marker
    # RECORDS, so a forced head (a nonce id) under the same version is left alone. (A source with no accepted head, or a held
    # one, is already changed above; a standing refusal was compared against its own attempt's version, not the head's.)
    if conn is not None:
        quiet = [g for g in plan.groups if g not in plan.changed and g not in standing]
        heads = _head_versions(conn, [source_id_for(adapter.name, g) for g in quiet])
        plan.changed.update(g for g in quiet if heads.get(source_id_for(adapter.name, g), version) != version)
    # a relocated shared-RNG traversal invalidates EVERY group of the adapter (S3)
    if plan.changed and getattr(adapter, "rng_scope", "none") == "reader_traversal":
        plan.changed = set(plan.groups)
    if reread:
        plan.changed = set(plan.groups)          # every present group reads again; inputs_changed still counts real changes
    # the preserved groups, last: whatever selected them above, they are not read, and their decision watermarks are not gone inputs
    hold = {g: why for g, why in (preserved or {}).items() if g in plan.groups}
    if hold and getattr(adapter, "rng_scope", "none") == "reader_traversal":
        cause = next(iter(hold.values()))
        hold = {g: cause for g in plan.groups}
    for g, why in sorted(hold.items()):
        plan.changed.discard(g)
        plan.decision_changed.discard(g)
        plan.gone_in_group.pop(g, None)
        plan.preserved[g] = why
    plan.refused_standing = sorted(standing - plan.changed - set(plan.preserved))
    check_plan_invariants(adapter, plan)          # a key the reading loop cannot serve is a planner defect, refused by name here
    return plan


def standing_refusal_digest(refusal: str, attempt_revision_id: str, code_sha: str) -> str:
    """The digest of a recorded refusal: sha256 over the refusal text, the input digest (the attempt's revision id, sha16 over
    the inputs' URIs and digests and the emission version) and the tree sha, as one canonical JSON list -- what a standing refusal
    and the refusal it equals share, byte for byte."""
    return hashlib.sha256(json.dumps([str(refusal), str(attempt_revision_id), str(code_sha)], ensure_ascii=False).encode("utf-8")).hexdigest()


def standing_refusal_of(marks: Mapping[str, Watermark], inputs: Sequence[Input], revision_id: str, error: Optional[str],
                        code_sha: Optional[str]) -> Optional[Dict[str, object]]:
    """THE STANDING-REFUSAL RULE OF A RE-READ. A group selected under ``retry_refused`` (or any cause) and REFUSED AGAIN by its
    reader is a ``standing_refusal`` result -- recorded like any refusal, never a refusal of the adapter -- exactly when the new refusal is
    byte for byte the one the group's stamps already carry, under the same inputs and the same reading tree: every input of the group holds
    a refused stamp, every stamp names ONE attempt (``attempt_revision_id`` == this read's revision id: the same input set with the same
    digests, the listing fingerprints included, the adapter's emission version), ONE refusal text equal to this read's ``error``, and ONE
    reading tree equal to this pass's producer ``code_sha``. Returns the PRIOR refusal (``GroupResult.prior_refusal``: the ``window`` that
    recorded it, its ``utc``, its ``digest`` -- :func:`standing_refusal_digest` -- the ``attempt_revision_id`` and the ``code_sha``) or None -- a
    NEW refusal (no stamp, a stamp before the fields), a CHANGED one (other words, other inputs, another tree) is a refusal as today.

    Exact, never a heuristic: the words are compared whole (the sha256 of the text is what the record carries), the inputs by the revision id
    the stamp recorded, the tree by its sha. Consequence, stated plainly: the FIRST window on a new tree finds every standing refusal CHANGED
    (the tree differs) and still ends ABORTED on it; the second window on that tree does not -- the first window's refusal records travel back
    with the pull (``cluster_reads.shard_summary`` ``refused_records``), the harvester folds their stamps, and the next sidecar carries them."""
    if not inputs or error is None or not code_sha:
        return None
    stamps = set()
    for inp in inputs:
        mark = marks.get(inp.identity_uri)
        if mark is None or not mark.refused:
            return None
        stamps.add((mark.attempt_revision_id, mark.last_error, mark.attempt_code_sha, mark.attempt_window, mark.observed_utc))
    if len(stamps) != 1:
        return None
    attempt, words, tree, window, utc = next(iter(stamps))
    if attempt != revision_id or words != error or tree != code_sha:
        return None
    return {"window": window, "utc": utc, "digest": standing_refusal_digest(str(words), str(attempt), str(tree)), "attempt_revision_id": attempt, "code_sha": tree}


def deal_shards(plan: GroupPlan, index: int, count: int) -> Tuple[Set[str], Set[str], int]:
    """The shard DEAL (plan step 3): the SORTED UNION of the present comparison groups and the prior-only (vanished) groups,
    by CANONICAL identity, dealt round-robin -- the group at position ``p`` of the union belongs to shard ``p % count`` -- so every
    group, a vanished one included, has exactly ONE deterministic owner across the K shards of a pass, whatever each shard
    discovered first. Returns ``(the groups of plan.changed this shard owns, the vanished groups it owns, the union's size)``, the
    groups in the plan's own (local) spelling."""
    if count < 1 or not 0 <= index < count:
        raise ValueError(f"shard {index}/{count}: the index must lie in [0, {count})")
    universe = sorted({comparison_group_identity(g) for g in plan.groups} | {comparison_group_identity(g) for g in plan.vanished})
    owned = {g for position, g in enumerate(universe) if position % count == index}
    changed = {g for g in plan.changed if comparison_group_identity(g) in owned}
    vanished = {g for g in plan.vanished if comparison_group_identity(g) in owned}
    return changed, vanished, len(universe)


@dataclass
class ClusterRead:
    """A CLUSTER shard's rules: the pass compares against a shipped reader-state sidecar (``conn``) and never
    re-activates or re-reads -- a group whose minted revision id the grading host already knows is refused by name and recorded (no
    grading-host read follows). KNOWN means the grading host holds something to re-activate: a ``source_revision`` row it accepted or
    holds, a published directory or bundle, folded or not. An id the grading host REFUSED AT EMIT TIME and never committed (its row
    folded from the ``_refused/<rev>/`` record, no directory, no bundle) is NOT known: nothing exists to re-activate from, so the shard
    reads the group fresh and publishes or refuses it on its own reader. ``known_dirs`` = the revision ids the sidecar's
    ``published_dirs`` lists (the grading host's on-disk ``records/adapters/<a>/*``: published, folded or not, refused records
    included). ``read_copy`` rides every marker as ``extra.read_copy`` (which copy of which component was read, under which
    attestation) beside the producer's substrate and host."""
    known_dirs: Set[str] = field(default_factory=set)
    read_copy: Dict[str, object] = field(default_factory=dict)
    #: of ``known_dirs``, the ids the sidecar lists as a REVISION DIRECTORY or an ACTIVATION BUNDLE (never a ``_refused`` record): with no
    #: ``source_revision`` row for the id, the grading host has it on the record log and has not folded it yet -- the shard's reading of
    #: :func:`pending_fold` (the sidecar carries no delivery ledger; a folded directory or bundle always leaves a row, accepted or held)
    pending_dirs: Set[str] = field(default_factory=set)
    #: the STAGED components this shard reads through, ``{slug: {"base": <canonical base under the home>, "generation": <sha8>}}``
    #: (``cluster_reads.run_shard`` fills it from the plan's pins): what :meth:`covering` names for a missing input
    components: Dict[str, Dict[str, str]] = field(default_factory=dict)
    #: the pinned generations' attested directory listings, ``path -> sorted (name, kind) entries or None`` (``cluster_reads.run_shard``
    #: installs the generation hash cache's :meth:`attested_listing`): what every listing input is compared with (:class:`ClusterListingRefusal`);
    #: None (no manifest reached this shard) attests nothing
    listing_attestation: Optional[Callable[[str], Optional[Sequence[Sequence[str]]]]] = None

    def attested_listing(self, path: str) -> Optional[Sequence[Sequence[str]]]:
        if self.listing_attestation is None or not path:
            return None
        return self.listing_attestation(path)

    @classmethod
    def from_sidecar(cls, conn: sqlite3.Connection, read_copy: Optional[Mapping[str, object]] = None) -> "ClusterRead":
        """The shard's rules from an open reader-state sidecar (``reader_state.known_dir_kinds``): every listed id is known, the
        revision directories and activation bundles among them are the pending candidates."""
        from .. import reader_state as RS
        kinds = RS.known_dir_kinds(conn)
        return cls(known_dirs=set(kinds), read_copy=dict(read_copy or {}),
                   pending_dirs={rev for rev, ks in kinds.items() if ks & {"revision", "activation"}})

    def committed_on_grading_host(self, revision_id: str) -> bool:
        """The sidecar lists a REVISION DIRECTORY or an ACTIVATION BUNDLE of ``revision_id`` on the grading host's disk: committed bytes exist
        there, folded or not. A ``_refused/<rev>`` record alone is not that."""
        return revision_id in self.pending_dirs

    def pending_on_grading_host(self, conn: Optional[sqlite3.Connection], revision_id: str) -> bool:
        """The grading host holds a directory or bundle of ``revision_id`` on its record log and has no row for it: a fold pending there."""
        return revision_id in self.pending_dirs and _revision_state(conn, revision_id) is None

    def known(self, conn: Optional[sqlite3.Connection], revision_id: str) -> Optional[str]:
        """WHY the grading host knows ``revision_id`` (``source_revision`` row accepted / pending, a published directory or bundle not folded
        yet, or a published directory), or None -- None too for an id the grading host REFUSED AT EMIT TIME and never committed."""
        state = _revision_state(conn, revision_id)
        if state == "pending" and not self.committed_on_grading_host(revision_id) and _emit_time_refusal(conn, revision_id) is not None:
            # REFUSED AT EMIT TIME on the grading host and never committed: the row is the harvester's fold of the refused record
            # (`_refused/<rev>/`), and the sidecar lists no directory or bundle of the id -- there are no committed bytes to re-activate,
            # so the id is NOT known for re-activation: the shard reads the group fresh and publishes or refuses it on its own reader
            # (the standing-refusal rule already decided WHEN -- `plan_groups` selected the group under `retry_refused`, a changed input
            # or a bump; this check must not short-circuit it). Before this, 60 such ids (objects_bank, overrefusal_items, relu_native,
            # hpo_logs; 2026-10-07) came back refused-known, the pull's re-activation by id found no directory, and the ADAPTER was
            # refused for the window. A row the harvester HOLDS, or one with a directory or bundle listed, stays known (pending) as before.
            return None
        if state is not None:
            return f"source_revision row, {state}"
        if revision_id in self.pending_dirs:
            return "a published directory on the grading host, not folded yet (its fold is pending there)"
        if revision_id in self.known_dirs:
            return "a published directory on the grading host"
        return None

    def refuse_known(self, conn: Optional[sqlite3.Connection], adapter_name: str, group: str, revision_id: str, source_id: str,
                     retiring: bool = False) -> Optional[GroupResult]:
        """The refused-known result for ``revision_id`` when the grading host knows it (nothing is published for it here), else None; it
        carries ``activation_pending`` when the grading host has the id on its log unfolded (:meth:`pending_on_grading_host`)."""
        why = self.known(conn, revision_id)
        if why is None:
            return None
        return GroupResult(adapter_name, group, revision_id, accepted=False, refused_known=True,
                           activation_pending=self.pending_on_grading_host(conn, revision_id),
                           error=f"known to the grading host: {'retiring ' if retiring else ''}revision {revision_id} of {source_id} ({why}); "
                                 "reactivation is the grading host's -- nothing published here")

    def marker_extra(self) -> Dict[str, object]:
        return {"read_copy": dict(self.read_copy)} if self.read_copy else {}

    def covering(self, uri: str) -> Optional[str]:
        """The staged component that WOULD HAVE COVERED an input the read did not see -- the one whose base (the longest) the canonical
        spelling lies under, named with its pinned generation (the copy of that component is incomplete: a member the generation lacks)
        -- or None when no staged component covers it (a tree ``cluster_reads.yaml`` does not declare, to be declared or read on the
        grading host). A per-group alias (``<uri>#<group>``) is judged by its file."""
        u = str(uri).split("#", 1)[0].rstrip("/")
        best: Optional[Tuple[str, str, str]] = None
        for slug, info in self.components.items():
            base = str(info.get("base") or "").rstrip("/")
            if base and (u == base or u.startswith(base + "/")) and (best is None or len(base) > len(best[1])):
                best = (slug, base, str(info.get("generation") or ""))
        if best is None:
            return None
        return f"component {best[0]}" + (f" (generation {best[2]})" if best[2] else "")

    def covering_of(self, uris: Iterable[str]) -> Dict[str, Optional[str]]:
        return {str(u): self.covering(str(u)) for u in uris}


def _marker_extra(cluster: Optional[ClusterRead]) -> Dict[str, object]:
    """What a ``cluster`` read adds to every marker's ``extra`` (``read_copy``); nothing on the grading host."""
    return cluster.marker_extra() if cluster is not None else {}


def run_pass(root: str, adapters: Sequence[Adapter], conn: Optional[sqlite3.Connection] = None, full: bool = False,
             now: Optional[str] = None, producer: Optional[Mapping[str, object]] = None,
             legacy_spellings: Optional[Mapping[str, Mapping[str, str]]] = None, reread: bool = False, force: bool = False,
             retry_refused: bool = False, hash_cache: Optional[InputHashCache] = None, identity_map_: Optional[IdentityMap] = None,
             shard: Optional[Tuple[int, int]] = None, cluster: Optional[ClusterRead] = None, save_cache: bool = True) -> PassReport:
    """One incremental (or ``full``) pass over the given adapters; writes records only (no DB writes). ``reread`` reads
    EVERY present group again — a committed revision through :func:`reread_committed`, the rest as usual; with ``force``
    every read is published as a NEW revision whatever its inputs (the pass stamp ``now`` is its id's nonce). A
    group whose inputs are unchanged but whose accepted head was minted under another ``EMISSION_VERSION`` counts as
    changed: the adapter's code moved, so the incremental pass publishes the new revision without being told. Which groups
    are read, and when a standing refusal is (``retry_refused`` is one cause), is :func:`plan_groups`. Every input's sha comes
    through the pass's :class:`base.InputHashCache` — ``state/input_hash_cache.json`` under ``root`` unless ``hash_cache`` is given,
    bypassed (every byte re-hashed) when ``full``, saved back pruned when the pass ends, whatever ended it (``save_cache=False``: a
    shard's manifest-backed cache is never written back). ``identity_map_`` (default: none) is the pass's :class:`base.IdentityMap`;
    ``shard = (i, K)`` reads the groups :func:`deal_shards` hands shard ``i`` of ``K`` (a ``reader_traversal`` adapter admits K = 1
    only: its one shared RNG runs over the whole traversal); ``cluster`` puts the pass in CLUSTER mode (:class:`ClusterRead`;
    ``reread`` is refused there: reactivation is the grading host's)."""
    SA.install(root)            # the root's site aliases: an adapter reads a persisted legacy spelling of a site label as its label
    if force and not reread:
        raise ValueError("force applies to a re-read only (adapters run --reread --force): the incremental pass publishes a new "
                         "revision when an input or the adapter's EMISSION_VERSION changed, and nothing otherwise")
    if cluster is not None and reread:
        raise ValueError("a cluster shard cannot --reread: a re-read re-activates a committed revision from its committed bytes, which "
                         "only the grading host holds; run it there")
    if shard is not None:
        traversal = [a.name for a in adapters if getattr(a, "rng_scope", "none") == "reader_traversal"]
        if shard[1] > 1 and traversal:
            raise ValueError(f"shard {shard[0]}/{shard[1]}: {', '.join(traversal)} declare rng_scope = reader_traversal (one shared RNG over the "
                             "whole traversal); such an adapter reads as ONE shard (K = 1) or its numbers change")
        deal_shards(GroupPlan(), shard[0], shard[1])      # the index is validated before any store is read
    # the S2 chokepoint (C6): a pass discovers and reads stores, so it needs a store access scope -- the shard's, or the operator's override
    require_store_access("adapters run: the pass over " + ", ".join(str(getattr(a, "name", type(a).__name__)) for a in adapters))
    now = now or R.utc_now()
    forced_utc = now if (reread and force) else None
    rep = PassReport()
    prod = dict(producer or H.harvester_producer(root))
    prod["script"] = "sisteer.registry.adapters.runner"
    ctx = AdapterContext(root=root, now=now, producer=prod, legacy_spellings=dict(legacy_spellings or {}), conn=conn)
    cache = hash_cache if hash_cache is not None else InputHashCache.load(root)
    cache.bypass = full
    marks = _watermarks_by_adapter(conn)
    try:
        with using_identity_map(identity_map_), using_hash_cache(cache):
            for adapter in adapters:
                _run_adapter(root, adapter, marks.get(adapter.name, {}), ctx, conn, rep, full=full, reread=reread, forced_utc=forced_utc,
                             retry_refused=retry_refused, shard=shard, cluster=cluster)
    finally:
        cache.prune()
        if cache.dirty and save_cache:
            cache.save(root)
        rep.inputs_hashed = cache.hashed
        rep.inputs_hash_reused = cache.reused
    return rep


def _run_adapter(root: str, adapter: Adapter, marks: Mapping[str, Watermark], ctx: AdapterContext, conn: Optional[sqlite3.Connection],
                 rep: PassReport, full: bool, reread: bool, forced_utc: Optional[str], retry_refused: bool, shard: Optional[Tuple[int, int]] = None,
                 cluster: Optional[ClusterRead] = None) -> None:
    """One adapter of a pass: discover, plan (:func:`plan_groups`), deal (:func:`deal_shards`, when sharded), read every planned
    group this shard owns, retire every vanished one it owns."""
    rep.adapters += 1
    try:
        inputs = adapter.discover()
    except AdapterRefusal as exc:
        rep.results.append(GroupResult(adapter.name, "<discover>", "", False, error=str(exc)))
        rep.groups_refused += 1
        return
    rep.inputs_seen += len(inputs)
    version = emission_version_of(adapter)              # refuses by name on a malformed declaration, before any read
    # the anchor decision input of every group whose head recorded a decision set: the head's comparisons replayed
    # against this pass's frame, BEFORE planning, so a ruling that moved an anchor reads as a changed input of exactly that group
    decision_inputs: Dict[str, Input] = {}
    preserved: Dict[str, str] = {}
    groups_now = sorted({adapter.comparison_group(i) for i in inputs})
    try:
        decision_inputs = anchor_decision_inputs(adapter, groups_now, conn, ctx)
    except AnchorCurrencyUnavailable as exc:
        # a head recorded decisions and the frame cannot be loaded: the groups whose heads recorded a set are PRESERVED -- not read, nothing
        # written, their decision watermarks not gone inputs -- each named in the results with the cause; a group read for another cause
        # asks the frame itself and refuses by name there (AnchorCurrencyRefusal). Planning WITHOUT the synthetic inputs instead would read
        # every recorded group's watermark as vanished and select it for a read that cannot be served.
        cause = f"the anchor currency of the pass is unavailable: {exc}"
        preserved = {g: cause for g in recorded_decision_groups(adapter, groups_now, conn)}
        log.error("%s: %d group(s) whose heads recorded an anchor decision set are preserved as they stand (not read, nothing written): %s",
                  adapter.name, len(preserved), exc)
    plan = plan_groups(adapter, inputs, marks, version, conn, full=full, reread=reread, retry_refused=retry_refused, decision_inputs=decision_inputs,
                       preserved=preserved)
    preserved_here = set(plan.preserved)
    if shard is not None:
        # a preserved group is reported by the shard that OWNS it under the same deal as a changed one (the universe is the plan's)
        preserved_here, _, _ = deal_shards(replace(plan, changed=preserved_here), shard[0], shard[1])
    for g in sorted(preserved_here):
        rep.results.append(GroupResult(adapter.name, g, "", False, error=f"{source_id_for(adapter.name, g)}: preserved as it stands; {plan.preserved[g]}",
                                       anchor_unavailable=True))
        rep.groups_refused += 1
        rep.groups_anchor_unavailable += 1
    rep.inputs_changed += plan.inputs_changed
    rep.anchor_decisions_changed += len(plan.decision_changed)
    if plan.decision_changed:
        log.info("%s: %d group(s) read because their anchor decision set moved against the head's: %s", adapter.name, len(plan.decision_changed),
                 ", ".join(sorted(plan.decision_changed)[:10]) + (" …" if len(plan.decision_changed) > 10 else ""))
    if plan.refused_standing:
        rep.refused_standing[adapter.name] = plan.refused_standing
    changed, vanished = plan.changed, set(plan.vanished)
    rep.groups_seen += len(plan.groups) + len(plan.vanished)
    if shard is not None:
        changed, vanished, n_union = deal_shards(plan, shard[0], shard[1])
        owned = len(changed) + len(vanished)
        rep.groups_not_owned += (len(plan.changed) + len(plan.vanished)) - owned
    budget = ReactivationBudget()               # this adapter's cap on activation bundles this pass (an explicit --reread runs uncapped)
    n_pending = n_deferred = 0
    for g in sorted(changed):
        # a group with a refused stamp whose ACCEPTED revision is already committed (a re-read of the head that refused) goes through a
        # genuine re-read whatever selected it (--retry-refused, --full, a traversal sibling's change, an input restored): the
        # reader runs before the refusal clears, where the incremental path would re-stamp its watermarks clean off the committed
        # bytes without opening a store. A revision not yet published reads fresh either way; a held one is left alone (refresh_group).
        res = refresh_group(root, adapter, g, plan.groups[g], ctx, conn, full, vanished_uris=sorted(plan.gone_in_group.get(g, ())),
                            reread=reread, forced_utc=forced_utc, cluster=cluster, refused=g in plan.refused, budget=budget)
        if (cluster is not None and g in plan.refused and not res.accepted and res.files
                and not (res.deferred or res.held or res.refused_known or res.anchor_unavailable or res.inputs_dropped)):
            # a SHARD's re-read of a group that stood refused, refused again: standing when byte for byte the recorded refusal under
            # the same inputs and the same reading tree -- recorded like any refusal (its record travels with the pull), never the adapter's
            prior = standing_refusal_of(marks, plan.groups[g], res.revision_id, res.error, str(ctx.producer.get("code_sha") or ""))
            if prior is not None:
                res.standing_refusal = True
                res.prior_refusal = prior
        rep.results.append(res)
        n_pending += int(res.activation_pending)
        if res.accepted:
            rep.groups_refreshed += 1
            rep.groups_reread += int(res.reread)
            rep.groups_forced += int(res.forced)
            rep.groups_emitted_nothing += int(res.emitted_nothing)
            rep.records_written += res.records_written
            rep.withdrawals += res.withdrawals
        elif res.deferred:
            n_deferred += 1
        elif res.held:
            rep.groups_held += 1
        elif res.refused_known:
            rep.groups_refused_known += 1
        elif res.standing_refusal:
            rep.groups_standing_refusals += 1
        else:
            rep.groups_refused += 1
            rep.groups_inputs_dropped += int(res.inputs_dropped)
            rep.groups_anchor_unavailable += int(res.anchor_unavailable)
    rep.activations_pending += n_pending
    rep.reactivations_deferred += n_deferred
    if n_pending:
        # one line per adapter per pass, never one per group: the count is the fact an operator acts on (a fold lagging behind the pass)
        log.info("%s: %d activation(s) pending fold -- on the record log, not folded by the harvester yet; nothing re-emitted for them",
                 adapter.name, n_pending)
    if n_deferred:
        log.warning("%s: %d re-activation(s) deferred by the per-pass cap (MAX_REACTIVATIONS_PER_PASS = %d; %d bundle(s) written this pass); "
                    "the next pass takes them", adapter.name, n_deferred, budget.limit, budget.taken)
    for g in sorted(vanished):
        res = retire_vanished_group(root, adapter, g, sorted(plan.vanished[g]), ctx, conn, cluster=cluster)
        if res is not None:
            rep.results.append(res)
            rep.withdrawals += res.withdrawals
            rep.groups_refused_known += int(res.refused_known)
            rep.activations_pending += int(res.activation_pending)
            if not res.accepted and not res.refused_known:
                # a CLUSTER shard's refusal of the retirement (the staged copy is incomplete): counted as a refused group, dropped inputs
                rep.groups_refused += 1
                rep.groups_inputs_dropped += int(res.inputs_dropped)
