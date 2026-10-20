"""The appendix families' BigCodeBench-200 stores -> registry (source ``bigcodebench_allmodels``).

WHAT WAS RUN (the appendix run, 2026-09-23 20:03Z; Modal, the ``twosi-fits`` volume): the paired_allmodels appendix families — gemma4e4b,
llama8b and nemotron3nano4b — on BigCodeBench-200, ONE seed tag per family (``<model key>_s1``) x {the unsteered anchor ``none``, the five admitted
staged objects (the family's ``main`` selection's objects, staged as ``base__<key>`` / ``derived__<key>``: constitution_relu, control_relu,
constitution_lora, control_lora, difference_lora; llama8b's ReLU objects from the ``llama8b_full`` fits), the two prompted system instructions}, plus
the composed ``difference_relu`` paired write on the paired tag ``<model key>_s1p`` (ReLU's ``paired_eb<B>__…`` object) with that tag's own ``none``.
gemma4e4b's and llama8b's seed-tag conditions ran under the family's ``main`` CANARY ROW — row dir ``<model key>-f22``, tag ``<model key>_main`` —
before the ``s1`` tag existed and file under ``<model key>_s1`` with the original tag recorded (:data:`_bcbw_common.LEGACY_MAIN_ALIAS_RULE`; a
canary-row fitted key resolves through the roster's ``main`` selection); the ``s1p`` keys, llama8b's ReLU keys and every nemotron3nano4b key sit
under the ``<model key>-f22s1[p]`` rows. No Arditi all-positions object exists for these families. Every row is the BigCodeBench-200
instrument — inspect_evals 0.18.0's ``bigcodebench`` task as row task ``inspect/bigcodebench_200`` over the 200 pinned items, pass@1 in the pinned
sandbox image, thinking OFF, the benchmark's reference protocol sampler (greedy: T 0 / top_p 1 / top_k 0; the registered rows
``<model id>|bigcodebench|off|protocol@v1``, the wildcard ``*|bigcodebench|off|protocol`` expanded onto every served model), ONE pass at 4,096
(``contracts.BCB200_MAX_TOKENS_OFF``; no cap step) — served natively (``proxy_vllm+lora+interleaved``; the unsteered and prompted keys ride the
passthrough route) on the checkpoint and revision :data:`_bcbw_common.MODEL_KEYS` names (= ``models.yaml``), under the DEFAULT Inspect seed rule.
Every row files on ``bigcodebench_200_protocol_greedy_4096`` bound to the served model (the per-model expansion: the protocol rows agree
across the models, so the families share one instrument row with the three-seed Qwen family); llama8b's rows file ``thinking_mode not_available``
(the model has no thinking channel) and match the ``not_available`` twin of the protocol row.

Everything else — the mirror's layout (the seedrep record shape under ``run-state/full22/bigcodebench_allmodels/``), the gate records, the launch set,
the launched pins and roster, the three-number record, the recount, the cross-checks, the anchors, the not-run rule, the model-key table, the
pin-key object route and the ``main`` canary alias — is the shared reading in :mod:`_bcbw_common` (the BFCL-200 reader ``bfcl_allmodels`` binds the
same code to the other task; the three-seed Qwen family's readers ``bigcodebench_seedrep`` / ``bfcl_seedrep`` bind it to their own model set). This
module binds the seedrep twin's :class:`_bcbw_common.TaskSpec` with its own source, root, three-number record and model set, and nothing else.
"""
from __future__ import annotations

import dataclasses
from typing import Tuple

from . import _bcbw_common as B
from . import bigcodebench_seedrep as SR
from .base import register

NAME = "bigcodebench_allmodels"
#: PROVISIONAL, registry-declared (configs/registry/sources_r2_v1.yaml meta.additions): the paper has declared no spelling for these stores
SOURCE = "bigcodebench_allmodels"
OWNER = B.OWNER
PRODUCER_SESSION = B.PRODUCER_SESSION
CAMPAIGN = B.CAMPAIGN
#: the appendix families this source reads (_bcbw_common.MODEL_KEYS); a store of another family refuses its group by name
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
#: ASSUMED until the mirror lands (2026-09-23; the seedrep twin's record is `three_numbers_bigcodebench_qwen_seedrep.json`): the producer's three-number
#: record beside the stores, named for the source. Verify against the first mirror; a rename is one line here
THREE_NUMBERS_FILE = "three_numbers_bigcodebench_allmodels.json"
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


class BigcodebenchAllmodelsAdapter(B.BcbwSeedrepReader):
    name = NAME
    SPEC = SPEC
    covers_sources: Tuple[str, ...] = (SOURCE,)
    #: 1 (2026-09-23): the first version — the appendix families' BigCodeBench-200 stores from the Modal volume through the shared reading
    #: bound to the appendix model set (module docstring)
    #: 2 (2026-10-03): every contrast is decided by the pass's anchor currency and carries
    #: extra.anchor_selection (`_bcbw_common.CURRENCY_RULE`); the model set gains qwen35_35b_a3b (its unsteered row `u0` is main's grammar, `UNSTEERED_ROW_RULE`, bound by the full readers alone)
    EMISSION_VERSION = 2
    owner = OWNER


def factory() -> BigcodebenchAllmodelsAdapter:
    return BigcodebenchAllmodelsAdapter()


register(NAME, factory)
