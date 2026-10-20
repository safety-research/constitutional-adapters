"""The appendix families' BFCL-200 stores -> registry (source ``bfcl_allmodels``).

The SAME stores as ``bigcodebench_allmodels`` (one store per (tag, object key) holds both tasks), published as their own byte-for-byte copy under
``run-state/full22/bfcl_allmodels/`` so each reader owns its root (never a symlink, never a cross-read): the appendix run (2026-09-23
20:03Z) on BFCL-200 for gemma4e4b and nemotron3nano4b — ONE seed tag per family (``<model key>_s1``) x {the unsteered ``none``, the five admitted
staged objects, the two prompted system instructions} (gemma4e4b's under the family's ``main`` canary row, filed under the ``s1`` tag), plus the composed
``difference_relu`` paired write on ``<model key>_s1p`` with that tag's own ``none``. **llama8b runs no BFCL row**: no llama store is
mirrored under this root and its launch set names no llama row, so the reader files nothing for the family here — not a refusal, not a
disposition. Every row is the BFCL-200 instrument — gorilla's ``bfcl_eval`` as row task ``inspect/bfcl_200`` over the 200 pinned items of 3,981, one
turn of tool calls scored by AST match, thinking OFF, the protocol sampler at T 0.001 executed literally under the served stack's clamp override
(``manifest.json:tinycap.served.sampling_clamp_override``; the registered rows ``<model id>|bfcl|off|protocol@v1``), ONE pass at 1,024
(``contracts.BFCL200_MAX_TOKENS_OFF``; no cap step) — served natively on the checkpoint and revision :data:`_bcbw_common.MODEL_KEYS` names, under the
DEFAULT Inspect seed rule. Every row files on ``bfcl_200_protocol_T0001_1024`` bound to the served model.

Everything else is the shared reading in :mod:`_bcbw_common` (see ``bigcodebench_allmodels`` for the appendix families' specifics: the model-key table,
the pin-key object route, the ``main`` canary alias). This module binds the seedrep twin's :class:`_bcbw_common.TaskSpec` with its own source, root,
three-number record and model set, and nothing else.
"""
from __future__ import annotations

import dataclasses
from typing import Tuple

from . import _bcbw_common as B
from . import bfcl_seedrep as SR
from .base import register

NAME = "bfcl_allmodels"
#: PROVISIONAL, registry-declared (configs/registry/sources_r2_v1.yaml meta.additions): the paper has declared no spelling for these stores
SOURCE = "bfcl_allmodels"
OWNER = B.OWNER
PRODUCER_SESSION = B.PRODUCER_SESSION
CAMPAIGN = B.CAMPAIGN
#: the appendix families this source reads (_bcbw_common.MODEL_KEYS); llama8b is declared and runs no BFCL row, so it files nothing here
MODEL_KEYS: Tuple[str, ...] = B.APPENDIX_MODEL_KEYS
EVAL_ID = SR.EVAL_ID
TASK = SR.TASK
RECORD_TASK = SR.RECORD_TASK
OUTCOMES_FILE = SR.OUTCOMES_FILE
INSTRUMENT_SEED = SR.INSTRUMENT_SEED
CAP = SR.CAP
LADDER: Tuple[int, ...] = SR.LADDER
N_ITEMS = B.N_ITEMS
PROTOCOL_TEMPERATURE = SR.PROTOCOL_TEMPERATURE
#: ASSUMED until the mirror lands (2026-09-23; the seedrep twin's record is `three_numbers_bfcl_qwen_seedrep.json`): the producer's three-number record
#: beside the stores, named for the source. Verify against the first mirror; a rename is one line here
THREE_NUMBERS_FILE = "three_numbers_bfcl_allmodels.json"
SPEC = dataclasses.replace(SR.SPEC, name=NAME, source=SOURCE, three_numbers_file=THREE_NUMBERS_FILE, root_name=SOURCE, models=MODEL_KEYS)
ROOT_DEFAULT = SPEC.root_default
RECOUNT_ALGORITHM = SPEC.recount_algorithm
RECOUNT_VERSION = B.RECOUNT_VERSION
RECOUNT_SEED_PROVENANCE = SPEC.recount_seed_provenance
DECODE_SEED_RULE = B.DECODE_SEED_RULE
ITEM_LABEL_AT_CAP_UNRESOLVED = B.ITEM_LABEL_AT_CAP_UNRESOLVED
ITEM_LABEL_UNSCORED = B.ITEM_LABEL_UNSCORED
RAW_AT_CAP_MEASURE = B.RAW_AT_CAP_MEASURE
BOUND_SPELLING = B.BOUND_SPELLING
BOUND_CI_METHOD = B.BOUND_CI_METHOD
GROUP_RULE = B.GROUP_RULE
ANCHOR_RULE = B.ANCHOR_RULE
SEED_AXIS_RULE = B.SEED_AXIS_RULE
NOT_RUN_RULE = B.NOT_RUN_RULE
GATE_RULE = B.GATE_RULE
APPENDIX_FAMILY_RULE = B.APPENDIX_FAMILY_RULE
LEGACY_MAIN_ALIAS_RULE = B.LEGACY_MAIN_ALIAS_RULE


def run_id_for(tag: str, store_key: str, okey: str, instrument_id: str) -> str:
    return B.run_id_for(SPEC, tag, store_key, okey, instrument_id)


def recount_computation_id() -> str:
    return SPEC.recount_computation_id()


class BfclAllmodelsAdapter(B.BcbwSeedrepReader):
    name = NAME
    SPEC = SPEC
    covers_sources: Tuple[str, ...] = (SOURCE,)
    #: 1 (2026-09-23): the first version — the appendix families' BFCL-200 stores from the Modal volume through the shared reading bound
    #: to the appendix model set (module docstring); llama8b runs no BFCL row
    #: 2 (2026-10-03): every contrast is decided by the pass's anchor currency and carries
    #: extra.anchor_selection (`_bcbw_common.CURRENCY_RULE`); the model set gains qwen35_35b_a3b (its unsteered row `u0` is main's grammar, `UNSTEERED_ROW_RULE`, bound by the full readers alone)
    EMISSION_VERSION = 2
    owner = OWNER


def factory() -> BfclAllmodelsAdapter:
    return BfclAllmodelsAdapter()


register(NAME, factory)
