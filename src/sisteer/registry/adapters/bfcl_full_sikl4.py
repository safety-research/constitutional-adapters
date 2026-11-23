"""The SI-distilled refits' FULL-suite BFCL stores -> registry (source ``bfcl_full_sikl4``): the ``bfcl_full`` reader PARAMETRIZED with the
declared campaign :data:`_bcbw_campaigns.SIKL4` (:data:`_bcbw_common.DECLARED_CAMPAIGN_RULE`), never a reader of its own. Everything the full reader checks it checks here --
the store's files, the gate record's PASS on ``lora_native``, the launch set, the launched roster by hash, the item set ``bfcl_full_3981`` on the instrument
``bfcl_full_protocol_T0001_1024``, the three-number record (its ``source`` this one), the recount -- and the campaign decides what the paper grammar did:

* campaign ``si_target_refit`` (the run id's first segment and the run row's ``campaign``): every run id disjoint from every paper id by construction;
* the rows: the three views ``qwen35_4b_sikl4s<k>`` (row dirs ``qwen35_4b-f22sikl4s<k>``, harness selections ``sikl4s<k>``, fit seed k), one group each; any
  other row dir under ``native/`` is an undeclared group refused whole by name, and a store suffix is ``bcbw-qwen35_4b_sikl4s<k>-<sha7>[-r<N>]`` of its own view;
* the objects: the six fitted cells of the view through ``configs/sdfbase_full22_roster_sikl4.yaml``'s pin form into ``configs/sikl4_pins.json`` (the launched
  copy ``records/sikl4_pins.json``), transfer_kind derived ``instruct_direct``; no ``none``, prompted or all-positions store is a row (each refuses by name);
* no registry contrast: the VALUE row parks its comparison under :data:`_bcbw_common.DECLARED_ANCHOR_RULE` and the frame reader anchors the row.

PROVISIONAL, registry-declared (configs/registry/sources_r2_v1.yaml meta.additions): comparison rows the paper frame excludes by ruling. The mirror root is
``run-state/full22/bfcl_full_sikl4/`` (the full readers' base; ``cluster_reads.yaml`` component ``full22_lanes``)."""
from __future__ import annotations

from typing import Tuple

from . import _bcbw_common as B
from . import _bcbw_campaigns as CS
from . import bfcl_full as FULL
from .base import register

NAME = "bfcl_full_sikl4"
SOURCE = NAME
DECLARED = CS.SIKL4
OWNER = DECLARED.producer_session
CAMPAIGN = DECLARED.campaign
EVAL_ID = FULL.EVAL_ID
INSTRUMENT_SEED = FULL.INSTRUMENT_SEED
ITEM_SET_ID = FULL.ITEM_SET_ID
SPEC = B.campaign_task_spec(FULL.SPEC, SOURCE, DECLARED)
#: the producer's three-number record beside the stores, named for the source (its top-level `source` must spell it, THREE_NUMBERS_SOURCE_RULE)
THREE_NUMBERS_FILE = SPEC.three_numbers_file
ROOT_DEFAULT = SPEC.root_default


def run_id_for(tag: str, store_key: str, okey: str, instrument_id: str) -> str:
    return B.run_id_for(SPEC, tag, store_key, okey, instrument_id)


class BfclFullSikl4Adapter(B.BcbwSeedrepReader):
    name = NAME
    SPEC = SPEC
    covers_sources: Tuple[str, ...] = (SOURCE,)
    #: 1 (2026-11-15): the first version -- the declared campaign SIKL4 through the full reader parametrized (_bcbw_common.CampaignSpec)
    EMISSION_VERSION = 1
    owner = OWNER


def factory() -> BfclFullSikl4Adapter:
    return BfclFullSikl4Adapter()


register(NAME, factory)
