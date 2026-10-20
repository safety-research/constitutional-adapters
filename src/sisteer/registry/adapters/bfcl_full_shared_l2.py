"""LANDING 2 of the SHARED-STANDUP comparison stores of the FULL-suite BFCL rows -> registry (source ``bfcl_full_shared_l2``).

The SAME 19 stores as :mod:`bigcodebench_full_shared_l2` (one row-engine store holds both full tasks), published as their own byte-for-byte copy under
``run-state/full22/bfcl_full_shared_l2/`` so each reader owns its root, compared against the solo ``bfcl_full`` stores. The whole reading is
:mod:`bigcodebench_full_shared_l2`'s docstring -- the ``_l2`` producer key (:data:`_bcbw_common.SHARED_L2_RUN_ID_RULE`: ``<tag>_shared_l2/<store key>[@rep<N>]``,
the harness variant in the producer key's source segment, the registry spelling ``<tag>_shared`` unchanged), the verified store's standup rule
(:data:`_bcbw_common.SHARED_VERIFIED_STANDUP_RULE`: the manifest's id when RESULT's last line is the verifier's re-stamp without ``standup=``, cross-checked against
the row runner's earlier line; later re-stamps carry the token, so the absent-token path covers exactly these 19 stores), the bound launch set's
``own_source`` spelling the base ``bfcl_full_shared`` recorded as provenance -- with the BFCL half of :mod:`bfcl_full`'s spellings: record task ``bfcl_full``,
outcomes file ``inspect__bfcl_full.json``, item set ``bfcl_full_3981`` on the instrument ``bfcl_full_protocol_T0001_1024``, the clamp override and the tool-call
route attested, the three-number record ``three_numbers_bfcl_full_shared.json`` (its ``source`` spelling this source). Landing 1's reader :mod:`bfcl_full_shared`
and the solo reader :mod:`bfcl_full` are untouched and re-emit byte for byte.
"""
from __future__ import annotations

import dataclasses
from typing import Optional, Tuple

from . import _bcbw_common as B
from . import bfcl_full as SOLO
from . import bfcl_full_shared as L1
from .base import register

NAME = "bfcl_full_shared_l2"
#: PROVISIONAL, registry-declared (configs/registry/sources_r2_v1.yaml meta.additions): landing 2 of the COMPARISON source, never a paper spelling
SOURCE = "bfcl_full_shared_l2"
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
SOLO_SOURCE = SOLO.SOURCE
BASE_SOURCE = L1.SOURCE
HARNESS_VARIANT = B.SHARED_L2_HARNESS_VARIANT
SHARED = dataclasses.replace(L1.SHARED, harness_variant=HARNESS_VARIANT, base_source=BASE_SOURCE)
THREE_NUMBERS_FILE = L1.THREE_NUMBERS_FILE
LAUNCH_SET_FILES = L1.LAUNCH_SET_FILES
SCHEDULE_FILE = L1.SCHEDULE_FILE
CAP = SOLO.CAP
LADDER: Tuple[int, ...] = SOLO.LADDER
PROTOCOL_TEMPERATURE = SOLO.PROTOCOL_TEMPERATURE
SPEC = dataclasses.replace(L1.SPEC, name=NAME, source=SOURCE, root_name=SOURCE, shared=SHARED)
ROOT_DEFAULT = SPEC.root_default
SOLO_ROOT_DEFAULT = SHARED.solo_root_default
RECOUNT_ALGORITHM = SPEC.recount_algorithm
RECOUNT_VERSION = B.RECOUNT_VERSION
RECOUNT_SEED_PROVENANCE = SPEC.recount_seed_provenance
SHARED_SOURCE_RULE = B.SHARED_SOURCE_RULE
SHARED_RUN_ID_RULE = B.SHARED_RUN_ID_RULE
SHARED_L2_RUN_ID_RULE = B.SHARED_L2_RUN_ID_RULE
SHARED_VERIFIED_STANDUP_RULE = B.SHARED_VERIFIED_STANDUP_RULE
SHARED_ROW_KEY_RULE = B.SHARED_ROW_KEY_RULE
SHARED_SPELLING_RULE = B.SHARED_SPELLING_RULE
SHARED_IT_NONE_RULE = B.SHARED_IT_NONE_RULE
SHARED_STANDUP_RULE = B.SHARED_STANDUP_RULE
SHARED_COMPARISON_RULE = B.SHARED_COMPARISON_RULE
SHARED_ROW_DIR_RULE = B.SHARED_ROW_DIR_RULE
SHARED_ENDPOINT_TUPLE_RULE = B.SHARED_ENDPOINT_TUPLE_RULE
SHARED_IT_REPEAT_RULE = B.SHARED_IT_REPEAT_RULE
VERIFICATION_STATE_VERIFIED = B.VERIFICATION_STATE_VERIFIED
COMPARISON_KINDS = B.COMPARISON_KINDS
ANCHOR_KIND_BORROWED = B.ANCHOR_KIND_BORROWED


def spelling_of(tag: str) -> str:
    return SPEC.spelling_of(tag)


def producer_spelling_of(tag: str) -> str:
    """``qwen35_4b_s1`` -> ``qwen35_4b_s1_shared_l2``: the producer key's spelling segment (:data:`_bcbw_common.SHARED_L2_RUN_ID_RULE`)."""
    return SHARED.producer_spelling_of(tag)


def run_id_for(tag: str, store_key: str, okey: str, instrument_id: str, repeat: Optional[int] = None) -> str:
    """The LANDING-2 run id (producer key ``<tag>_shared_l2/<store key>[@rep<N>]``): never landing 1's over the same store key, never the solo reader's."""
    return B.run_id_for(SPEC, tag, store_key, okey, instrument_id, repeat=repeat)


def landing1_run_id_for(tag: str, store_key: str, okey: str, instrument_id: str, repeat: Optional[int] = None) -> str:
    """The id landing 1's reader mints over the same store key -- what this reader's id must never equal."""
    return L1.run_id_for(tag, store_key, okey, instrument_id, repeat=repeat)


def solo_run_id_for(tag: str, store_key: str, okey: str, instrument_id: str) -> str:
    """The id the SOLO reader mints for the solo store -- what a launch row's ``solo_run_id`` must equal."""
    return SOLO.run_id_for(tag, store_key, okey, instrument_id)


def recount_computation_id() -> str:
    return SPEC.recount_computation_id()


class BfclFullSharedL2Adapter(B.BcbwSeedrepReader):
    name = NAME
    SPEC = SPEC
    covers_sources: Tuple[str, ...] = (SOURCE,)
    #: 1 (2026-09-30): the first version -- landing 2's stores through landing 1's reader parametrized (bigcodebench_full_shared_l2's docstring)
    #: 2 (2026-10-03): the anchor currency on every contrast (`_bcbw_common.CURRENCY_RULE`)
    #: 3 (2026-10-06, migration 0027): the stand-in contrast files anchor_kind `solo_stand_in` (landing 1's rule)
    EMISSION_VERSION = 3
    owner = OWNER


def factory() -> BfclFullSharedL2Adapter:
    return BfclFullSharedL2Adapter()


register(NAME, factory)
