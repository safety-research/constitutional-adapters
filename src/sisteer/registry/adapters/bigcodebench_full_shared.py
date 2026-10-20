"""The SHARED-STANDUP comparison stores of the FULL-suite BigCodeBench rows -> registry (source ``bigcodebench_full_shared``).

WHAT IS RUN (2026-09-28): the row-engine harness serves EVERY object of a model row on ONE vLLM
engine (one standup id per row launch; ``max_loras`` = the zero adapter + the fitted adapters; the residual slot armed when a PSR object is served) and
re-runs the canary rows -- gemma s1, qwen s1, nemotron s1, the qwen instruct row's own ``none`` -- as whole rows, COMPARED to the existing solo stores.
The 78 canonical rows and the paper anchors are untouched: this is a COMPARISON source, never a replacement and never in the paper frame.

THE READER IS :mod:`bigcodebench_full` PARAMETRIZED (:class:`_bcbw_common.SharedSpec` on the solo reader's :class:`_bcbw_common.TaskSpec`; the solo reader
reads byte for byte what it read before, the dry run over the live mirrors proving it). The same layout (``native/<row dir>/<object key>/thinking_off/<suffix>/``
+ ``records/``), the same records (the gate attestations, the launch set, the pins and roster set by hash, the item-set attestation, the three-number
record) and the same cross-checks, plus what the row engine adds -- every rule stamped on the shared rows alone (:data:`_bcbw_common.SHARED_SOURCE_RULE`):

* THE SUFFIX GRAMMAR ``bcbw-<launch tag>-<sha7>-shared[-r<N>][-rep<N>]`` (the launch tag is the SOLO tag's, ``launch_tag_of`` unchanged; ``-r<N>`` a row-scoped
  subset relaunch on its OWN standup, ``relaunches.json`` spelling ``tag shared-r<N>``; ``-rep2`` the row's ``none`` driven again on the SAME standup); the
  launch set is ``records/launch_sets_shared.json`` (the one name this source reads), the envelope ``bcbw_schedule_shared.json``, the three-number record
  ``three_numbers_bigcodebench_full_shared.json``;
* THE SPELLING: every row files on the registry spelling ``<tag>_shared`` (``gemma4e4b_s1_shared``, ``qwen35_4b_it_shared``; legacy_spellings.yaml spells each as
  the solo spelling's model, fit model and transfer kind), the launch row carrying BOTH ``view`` (that spelling) and ``selection`` (the solo selection, the row's
  key in the group); a row whose ``view`` is not the spelling of its selection's tag refuses by name (:data:`_bcbw_common.SHARED_SPELLING_RULE`);
* THE RUN ID (:data:`_bcbw_common.SHARED_RUN_ID_RULE`): ``cells/<store key>`` is byte-identical to the solo store's, so the producer key is
  ``<tag>_shared/<store key>`` -- never the solo tag's -- and the repeat folds its segment in (``<tag>_shared/<store key>@rep<N>``); the launch row's
  ``solo_run_id`` must be exactly what the SOLO reader mints for the solo store, and never this row's own;
* THE ROW KEY (:data:`_bcbw_common.SHARED_ROW_KEY_RULE`): the repeat row shares (tag, object key) with the row's ``none``, so rows are keyed by
  ``(tag, object key[@rep<N>])`` with the repeat segment read off ``launched_as.suffix`` and cross-checked against ``tuple_id`` / ``repeat.key``;
* THE COMPARISON (:data:`_bcbw_common.SHARED_COMPARISON_RULE`): ``comparison_kind`` / ``solo_run_id`` / ``solo_store`` are read from the launch row and the gate's
  ``comparison`` block (never a manifest field); each shared_vs_solo / repeat_floor run gets a REGISTRY-computed ``run_contrast`` against its SOLO run --
  ``anchor_kind borrowed`` (the solo run stands in), item-wise over the items scored on both sides, read from the solo mirror root
  (``run-state/full22/bigcodebench_full/``, the group's solo-side inputs), both headline values on the contrast -- beside the in-run contrast against the row's
  shared ``none`` (``same_standup_none``; ``sibling_none`` for a relaunch on its own standup, :data:`_bcbw_common.SHARED_ANCHOR_STANDUP_RULE`); the repeat
  row contrasts against both. The instruct row's own ``none`` files as a run (class ``unsteered``, ``it_none_comparison``, no solo run, no anchor) and forms
  NO contrast and anchors NOTHING (:data:`_bcbw_common.SHARED_IT_NONE_RULE`);
* THE ROW ENGINE (:data:`_bcbw_common.SHARED_STANDUP_RULE`): the manifest's ``row_engine`` block (exactly eight keys), ``served_geometry.json:endpoint``
  (``standup_attempt_id``, ``cells``, ``lora_config.max_loras``, ``residual_slot``, the served-digest map) and RESULT's ``standup=`` token must spell one standup
  id, the launch row's and the gate's; every store one row launch names must share it; the per-object served digest is ``provenance.artifact_sha256``
  beside ``route.object_artifact_sha256_by_cell[<harness key>]`` (``route.object_artifact_sha256`` is null on a row engine and never read as the digest);
  ``max_loras`` / ``residual_slot_armed`` are RECORDED, never compared; the gate's ``compared_against.comparability_class`` (same_standup / cross_standup)
  must agree with its standup ids;
* THE STORE-ATTESTATION DECISION of 2026-09-29 (the first REAL mirror, landing 1; three rules, each stamped on the shared rows alone):
  (1) :data:`_bcbw_common.SHARED_ROW_DIR_RULE` -- ``manifest.row_engine.row_dir`` may be the volume-absolute ``/vol/tinycap/gen/rows/<row>/<suffix>``; the
  reader binds it by the ``rows/<row>/<suffix>`` TAIL (this store's row; the ROW LAUNCH's container: the row's primary launch tag, the store's relaunch segment,
  no repeat segment; the launch row's own ``row_dir`` tail where the row declares one) and records the spelling verbatim on ``extra.row_engine.row_record_dir``;
  a tail that is not this store's row launch refuses; (2) :data:`_bcbw_common.SHARED_ENDPOINT_TUPLE_RULE` -- ``manifest.route.object_artifact_sha256_by_cell``
  may be null while ``served_geometry.json:endpoint`` carries the row's map, and the endpoint's ``cell`` / ``harness_cell`` may name the row's LAST served key
  (landing 1's row runner, :data:`_bcbw_common.ENDPOINT_SERVED_KEY_BUG`, fixed for later stores): the served map is the endpoint's, the store's tuple the
  manifest's, cross-checked against the pinned object and RESULT; a disagreement refuses; the last-key endpoint reads with ``field_confidence_json.object_id``
  downgraded one step (recorded to derived; an unsteered key's stays derived, the bug named in its source) and a source_status note (``extra.row_engine.endpoint_served_key_note``; the status extra ``endpoint_served_key_mismatch``); an agreeing
  endpoint reads clean with no note; (3) :data:`_bcbw_common.SHARED_IT_REPEAT_RULE` -- the instruct row's ``none@rep2`` store is a ``repeat_floor`` row: the
  repeat segment is read BEFORE the instruct-none classification, the row files under class ``repeat`` with no solo run and no solo store (no solo twin:
  ``extra.comparison.state no_solo_twin``, no borrowed contrast) and its in-run floor pairs against the row's own none (the ``it_none_comparison`` run, the one
  contrast that run anchors); never refused, never filed as ``it_none_comparison``.

WHAT THE REGISTRY'S FRAME DOES WITH THESE ROWS is a ruling, not a reader's decision: ``v_roster_match`` matches on model / eval / object / instrument and never
on the spelling or the source, so a shared row whose solo twin a roster entry matches (the gemma LoRA cells under the appendix transfer comparison) would
read ``selected`` beside its solo row unless an ``exclude`` ruling scoped ``match.lane_sources_legacy [bigcodebench_full_shared, bfcl_full_shared]`` covers the
two sources; the unmatched rows read ``not_in_roster``. The solo rows are held ``selected`` by the include ruling on their own sources (a decisive include
precedes the ``later`` rule), so no shared row can supersede them. Reported with the first dry read (2026-09-28); filed by the registry's owner.

MEASURED on the SYNTHETIC fixture (2026-09-28, its ``run-state/full22/bigcodebench_full_shared/``; 11 stores: the
eight ``-shared`` gemma s1 stores of one standup, the ``-shared-rep2`` repeat, one ``-shared-r2`` relaunch on its own standup with its parked ``-shared`` first
attempt mirrored COMPLETE, and the qwen instruct row's own ``none``): 10 runs, 8 in-run contrasts (7 same_standup_none + the relaunch's sibling_none), 9
borrowed contrasts against the live solo mirror (delta 0.0 pp by construction: the fixture copies the solo outcomes), 1 store parked, 0 refusals; the
solo readers re-emit byte for byte over the live mirrors. Since the rebuild of 2026-09-29 the fixture carries a 12th store, the qwen instruct row's
``-shared-rep2`` none: 2 runs in the qwen group with the one floor contrast.

MEASURED on the REAL mirror (2026-09-29, ``run-state/full22/{bigcodebench,bfcl}_full_shared/``, 33 stores each, re-published at 18:15Z): before the
decision every store refused on the volume-absolute ``row_dir``; under the three rules the launch-row binding, the row engine, the endpoint's map and the
instruct row's repeat all pass on all 33 stores of both readers (29 stores per reader carry the landing-1 endpoint note, 4 agree: the row's last key on each
row), the qwen instruct row's own none and its repeat file with the floor contrast, and the remaining refusal on the other 31 stores is store-side: the
producer's three-number record spells ``solo_store.mirror_rel`` under a ``<solo source>/`` prefix (``bigcodebench_full/native/...``) the launch row's
``mirror_rel`` (``native/...``) lacks (``_cross_check_three_shared``); with that spelling normalized in a read-only diagnostic, 33 runs per reader, 0 refusals.
"""
from __future__ import annotations

import dataclasses
from typing import Optional, Tuple

from . import _bcbw_common as B
from . import bigcodebench_full as SOLO
from .base import register

NAME = "bigcodebench_full_shared"
#: PROVISIONAL, registry-declared (configs/registry/sources_r2_v1.yaml meta.additions): a COMPARISON source, never a paper spelling
SOURCE = "bigcodebench_full_shared"
OWNER = B.OWNER
PRODUCER_SESSION = B.PRODUCER_SESSION
CAMPAIGN = B.CAMPAIGN
MODEL_KEYS: Tuple[str, ...] = tuple(SOLO.MODEL_KEYS)
EVAL_ID = SOLO.EVAL_ID
TASK = SOLO.TASK
RECORD_TASK = SOLO.RECORD_TASK
OUTCOMES_FILE = SOLO.OUTCOMES_FILE
INSTRUMENT_SEED = SOLO.INSTRUMENT_SEED
ITEM_SET_ID = SOLO.ITEM_SET_ID
#: the solo source the comparison stands against and the solo mirror root the comparison reads (its own copy of the stores)
SOLO_SOURCE = SOLO.SOURCE
SHARED = B.SharedSpec(solo_source=SOLO_SOURCE, solo_root_name=SOLO_SOURCE)
#: the producer's three-number record beside the stores, named for THIS source; the launch set's ONE name; the envelope's name
THREE_NUMBERS_FILE = "three_numbers_bigcodebench_full_shared.json"
LAUNCH_SET_FILES = (B.SHARED_LAUNCH_SET_FILE,)
SCHEDULE_FILE = B.SHARED_SCHEDULE_FILE
CAP = SOLO.CAP
LADDER: Tuple[int, ...] = SOLO.LADDER
PROTOCOL_TEMPERATURE = SOLO.PROTOCOL_TEMPERATURE
SPEC = dataclasses.replace(SOLO.SPEC, name=NAME, source=SOURCE, root_name=SOURCE, three_numbers_file=THREE_NUMBERS_FILE, launch_set_files=LAUNCH_SET_FILES,
                           schedule_file=SCHEDULE_FILE, shared=SHARED)
ROOT_DEFAULT = SPEC.root_default
SOLO_ROOT_DEFAULT = SHARED.solo_root_default
RECOUNT_ALGORITHM = SPEC.recount_algorithm
RECOUNT_VERSION = B.RECOUNT_VERSION
RECOUNT_SEED_PROVENANCE = SPEC.recount_seed_provenance
SHARED_SOURCE_RULE = B.SHARED_SOURCE_RULE
SHARED_RUN_ID_RULE = B.SHARED_RUN_ID_RULE
SHARED_ROW_KEY_RULE = B.SHARED_ROW_KEY_RULE
SHARED_SPELLING_RULE = B.SHARED_SPELLING_RULE
SHARED_IT_NONE_RULE = B.SHARED_IT_NONE_RULE
SHARED_STANDUP_RULE = B.SHARED_STANDUP_RULE
SHARED_COMPARISON_RULE = B.SHARED_COMPARISON_RULE
#: the store-attestation decision of 2026-09-29 (the first real mirror, landing 1): the row-record dir bound by its tail, the endpoint's served map and the
#: manifest's tuple, the instruct row's repeat floor
SHARED_ROW_DIR_RULE = B.SHARED_ROW_DIR_RULE
SHARED_ENDPOINT_TUPLE_RULE = B.SHARED_ENDPOINT_TUPLE_RULE
SHARED_IT_REPEAT_RULE = B.SHARED_IT_REPEAT_RULE
ENDPOINT_SERVED_KEY_BUG = B.ENDPOINT_SERVED_KEY_BUG
SOLO_STATE_NO_TWIN = B.SOLO_STATE_NO_TWIN
COMPARISON_KINDS = B.COMPARISON_KINDS
ANCHOR_KIND_BORROWED = B.ANCHOR_KIND_BORROWED


def spelling_of(tag: str) -> str:
    """``gemma4e4b_s1`` -> ``gemma4e4b_s1_shared``: the registry spelling a shared row files under."""
    return SPEC.spelling_of(tag)


def run_id_for(tag: str, store_key: str, okey: str, instrument_id: str, repeat: Optional[int] = None) -> str:
    """The SHARED run id (producer key ``<tag>_shared/<store key>[@rep<N>]``); never :func:`solo_run_id_for`'s over the same store key."""
    return B.run_id_for(SPEC, tag, store_key, okey, instrument_id, repeat=repeat)


def solo_run_id_for(tag: str, store_key: str, okey: str, instrument_id: str) -> str:
    """The id the SOLO reader mints for the solo store -- what a shared launch row's ``solo_run_id`` must equal."""
    return SOLO.run_id_for(tag, store_key, okey, instrument_id)


def recount_computation_id() -> str:
    return SPEC.recount_computation_id()


class BigcodebenchFullSharedAdapter(B.BcbwSeedrepReader):
    name = NAME
    SPEC = SPEC
    covers_sources: Tuple[str, ...] = (SOURCE,)
    #: 1 (2026-09-28): the first version -- the shared-standup comparison stores through the solo reader parametrized (module docstring): the
    #: `_shared` spelling and run ids, the repeat row key, the row-engine identity (one standup id per row launch), the comparison fields from the launch row
    #: and the gate, the borrowed run_contrast against the solo run read from the solo mirror; the instruct row's own none as a comparison row
    #: 2 (2026-09-29): the store-attestation decision (SHARED_ROW_DIR_RULE / SHARED_ENDPOINT_TUPLE_RULE / SHARED_IT_REPEAT_RULE) admits landing 1's real
    #: stores, every one of which version 1 refused. A bump, not a same-version fix: the live database already holds ACCEPTED version-1
    #: revisions of both shared sources with zero runs and `v_source_status.ok = 1`, and a group whose inputs do not change is re-read only on an input or
    #: emission change -- under version 1 the corrected reader would leave those zero-run revisions current. No run is retired (none was filed).
    #: 3 (2026-10-03): every in-run contrast is decided by the pass's anchor currency and the borrowed
    #: comparison's solo run is checked for currency (`_bcbw_common.CURRENCY_RULE`); extra.anchor_selection on every contrast
    #: 4 (2026-10-06, migration 0027): the shared - solo stand-in contrast files anchor_kind `solo_stand_in`
    #: (its own kind; `borrowed` is the currency's tier-1c re-anchoring alone), SHARED_SOURCE_RULE re-spelled; every stand-in row re-emits once
    EMISSION_VERSION = 4
    owner = OWNER


def factory() -> BigcodebenchFullSharedAdapter:
    return BigcodebenchFullSharedAdapter()


register(NAME, factory)
