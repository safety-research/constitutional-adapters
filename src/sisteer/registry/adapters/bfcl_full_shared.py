"""The SHARED-STANDUP comparison stores of the FULL-suite BFCL rows -> registry (source ``bfcl_full_shared``).

The SAME stores as ``bigcodebench_full_shared`` (one row-engine store holds both full tasks), published as their own byte-for-byte copy under
``run-state/full22/bfcl_full_shared/`` so each reader owns its root, compared against the solo ``bfcl_full`` stores under
``run-state/full22/bfcl_full/``. The whole reading is :mod:`bigcodebench_full_shared`'s docstring with the BFCL half of :mod:`bfcl_full`'s spellings: record
task ``bfcl_full``, outcomes file ``inspect__bfcl_full.json``, item set ``bfcl_full_3981`` on the instrument ``bfcl_full_protocol_T0001_1024``, the clamp
override and the tool-call route attested, the three-number record ``three_numbers_bfcl_full_shared.json`` (``eval bfcl``). The solo reader
:mod:`bfcl_full` is untouched (``TaskSpec.shared`` None) and re-emits byte for byte over the live mirror. The store-attestation decision of 2026-09-29
(:data:`_bcbw_common.SHARED_ROW_DIR_RULE`, :data:`_bcbw_common.SHARED_ENDPOINT_TUPLE_RULE`, :data:`_bcbw_common.SHARED_IT_REPEAT_RULE`; the bigcodebench
module's docstring) reads the same stores here: on the real mirror the 33 BFCL stores bind the same way, 29 carrying the landing-1 endpoint note.
"""
from __future__ import annotations

import dataclasses
from typing import Optional, Tuple

from . import _bcbw_common as B
from . import bfcl_full as SOLO
from .base import register

NAME = "bfcl_full_shared"
#: PROVISIONAL, registry-declared (configs/registry/sources_r2_v1.yaml meta.additions): a COMPARISON source, never a paper spelling
SOURCE = "bfcl_full_shared"
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
THREE_NUMBERS_FILE = "three_numbers_bfcl_full_shared.json"
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
    return SPEC.spelling_of(tag)


def run_id_for(tag: str, store_key: str, okey: str, instrument_id: str, repeat: Optional[int] = None) -> str:
    """The SHARED run id (producer key ``<tag>_shared/<store key>[@rep<N>]``); never :func:`solo_run_id_for`'s over the same store key."""
    return B.run_id_for(SPEC, tag, store_key, okey, instrument_id, repeat=repeat)


def solo_run_id_for(tag: str, store_key: str, okey: str, instrument_id: str) -> str:
    """The id the SOLO reader mints for the solo store -- what a shared launch row's ``solo_run_id`` must equal."""
    return SOLO.run_id_for(tag, store_key, okey, instrument_id)


def recount_computation_id() -> str:
    return SPEC.recount_computation_id()


class BfclFullSharedAdapter(B.BcbwSeedrepReader):
    name = NAME
    SPEC = SPEC
    covers_sources: Tuple[str, ...] = (SOURCE,)
    #: 1 (2026-09-28): the first version -- the shared-standup comparison stores through the solo reader parametrized (bigcodebench_full_shared's docstring)
    #: 2 (2026-09-29): the store-attestation decision admits landing 1's real stores (bigcodebench_full_shared's note): the live database holds ACCEPTED version-1
    #: revisions with zero runs and `v_source_status.ok = 1`, and an unchanged group is re-read only on an input or emission change, so the corrected reader
    #: needs the bump to re-read them. No run is retired (none was filed).
    #: 3 (2026-10-03): every in-run contrast is decided by the pass's anchor currency and the borrowed
    #: comparison's solo run is checked for currency (`_bcbw_common.CURRENCY_RULE`); extra.anchor_selection on every contrast
    #: 4 (2026-10-06, migration 0027): the shared - solo stand-in contrast files anchor_kind `solo_stand_in`
    #: (its own kind; `borrowed` is the currency's tier-1c re-anchoring alone), SHARED_SOURCE_RULE re-spelled; every stand-in row re-emits once
    EMISSION_VERSION = 4
    owner = OWNER


def factory() -> BfclFullSharedAdapter:
    return BfclFullSharedAdapter()


register(NAME, factory)
