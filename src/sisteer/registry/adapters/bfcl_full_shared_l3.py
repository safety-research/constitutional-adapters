"""LANDING 3 of the SHARED-STANDUP comparison stores of the FULL-suite BFCL rows -> registry (source ``bfcl_full_shared_l3``).

The SAME four row-engine stores as :mod:`bigcodebench_full_shared_l3` (one store holds both full tasks; the gemma4e4b s1 row on tree ``f1347e9``, the
fan-out test on the row-engine harness), published as their own byte-for-byte copy under ``run-state/full22/bfcl_full_shared_l3/`` (each reader owns its root) and
read by :mod:`bfcl_full_shared` PARAMETRIZED as :mod:`bfcl_full_shared_l2` is -- ``dataclasses.replace`` on landing 1's spec with ``harness_variant = "l3"``,
``base_source`` and :data:`_bcbw_common.SHARED_L3_RULE` as ``landing_rule`` -- on the BFCL full instrument (item set ``bfcl_full_3981``;
``three_numbers_bfcl_full_shared.json`` whose ``source`` spells THIS source). Everything the other module's docstring records -- the ``_l3`` producer key and the ids
disjoint from landings 1 and 2 and the solo reader, the fourteen-key verified block recorded never pinned, the token-present re-stamp read off the last line, the
non-null legacy route scalar recorded never read as identity, the volume-absolute ``row_dir``, the frame -- holds here unchanged. Dry read 2026-10-01: 4 runs / 0
refusals / 0 malformed (15,924 item results), 3 ``same_standup_none`` + 4 ``borrowed`` contrasts, every anchor present in the live database under ``bfcl_full``.
"""
from __future__ import annotations

import dataclasses
from typing import Optional, Tuple

from . import _bcbw_common as B
from . import bfcl_full as SOLO
from . import bfcl_full_shared as L1
from . import bfcl_full_shared_l2 as L2
from .base import register

NAME = "bfcl_full_shared_l3"
#: PROVISIONAL, registry-declared (configs/registry/sources_r2_v1.yaml meta.additions): landing 3 of the COMPARISON source, never a paper spelling
SOURCE = "bfcl_full_shared_l3"
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
HARNESS_VARIANT = B.SHARED_L3_HARNESS_VARIANT
LANDING_RULE = B.SHARED_L3_RULE
SHARED = dataclasses.replace(L1.SHARED, harness_variant=HARNESS_VARIANT, base_source=BASE_SOURCE, landing_rule=LANDING_RULE)
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
SHARED_L3_RULE = B.SHARED_L3_RULE
SHARED_VERIFIED_STANDUP_RULE = B.SHARED_VERIFIED_STANDUP_RULE
THREE_NUMBERS_SOURCE_RULE = B.THREE_NUMBERS_SOURCE_RULE
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
    """``gemma4e4b_s1`` -> ``gemma4e4b_s1_shared_l3``: the producer key's spelling segment (:data:`_bcbw_common.SHARED_L2_RUN_ID_RULE`, the variant ``l3``)."""
    return SHARED.producer_spelling_of(tag)


def run_id_for(tag: str, store_key: str, okey: str, instrument_id: str, repeat: Optional[int] = None) -> str:
    """The LANDING-3 run id (producer key ``<tag>_shared_l3/<store key>[@rep<N>]``): never landing 1's or landing 2's over the same store key, never the solo reader's."""
    return B.run_id_for(SPEC, tag, store_key, okey, instrument_id, repeat=repeat)


def landing1_run_id_for(tag: str, store_key: str, okey: str, instrument_id: str, repeat: Optional[int] = None) -> str:
    """The id landing 1's reader mints over the same store key -- what this reader's id must never equal."""
    return L1.run_id_for(tag, store_key, okey, instrument_id, repeat=repeat)


def landing2_run_id_for(tag: str, store_key: str, okey: str, instrument_id: str, repeat: Optional[int] = None) -> str:
    """The id landing 2's reader mints over the same store key -- what this reader's id must never equal either."""
    return L2.run_id_for(tag, store_key, okey, instrument_id, repeat=repeat)


def solo_run_id_for(tag: str, store_key: str, okey: str, instrument_id: str) -> str:
    """The id the SOLO reader mints for the solo store -- what a launch row's ``solo_run_id`` must equal."""
    return SOLO.run_id_for(tag, store_key, okey, instrument_id)


def recount_computation_id() -> str:
    return SPEC.recount_computation_id()


class BfclFullSharedL3Adapter(B.BcbwSeedrepReader):
    name = NAME
    SPEC = SPEC
    covers_sources: Tuple[str, ...] = (SOURCE,)
    #: 1 (2026-10-01): the first version -- landing 3's four gemma stores through landing 1's reader parametrized as on landing 2 (bigcodebench_full_shared_l3's docstring)
    #: 2 (2026-10-03): the anchor currency on every contrast (`_bcbw_common.CURRENCY_RULE`)
    #: 3 (2026-10-06, migration 0027): the stand-in contrast files anchor_kind `solo_stand_in` (landing 1's rule)
    EMISSION_VERSION = 3
    owner = OWNER


def factory() -> BfclFullSharedL3Adapter:
    return BfclFullSharedL3Adapter()


register(NAME, factory)
