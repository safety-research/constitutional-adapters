"""The FULL-suite BFCL stores -> registry (source ``bfcl_full``): every one of the 3,981 single-turn items, not the pinned 200.

The SAME stores as ``bigcodebench_full`` (one store per (tag, object key) holds both full tasks), published as their own byte-for-byte copy under
``run-state/full22/bfcl_full/`` so each reader owns its root (never a symlink, never a cross-read). THE CONTRACT is quoted in full in
:mod:`bigcodebench_full`; the BFCL half: record task ``bfcl_full``, outcomes file ``inspect__bfcl_full.json``, the label ``full``, item set ``bfcl_full_3981``
(n 3981; a NEW instrument, ``bfcl_full_protocol_T0001_1024``), the manifest key ``resolved_config.tinycap.bfcl_full_item_set`` = ``{seed: null, n: 3981, of: 3981,
ids_sha256}`` (the digest of the pin's ids in pin order), the launch-set row's ``item_set {id, sha256, n}`` with sha256 = the pin FILE's bytes
(:data:`CONTRACT_PIN_FILE_SHA256`), the mirrored pin file ``records/bfcl_full.json`` named by ``records/item_sets.json``. Any mismatch refuses the store by
name; an absent ``item_sets.json`` or pin file refuses every store of the source by name. The reader's own decisions where the contract is silent are the
BigCodeBench module's, with the BFCL spellings (the pin's ``task`` must read ``bfcl_full``; the row table under ``evals.bfcl``; the record
``three_numbers_bfcl_full.json``, ``eval bfcl``).

Every row is the BFCL FULL instrument -- gorilla's ``bfcl_eval`` as row task ``inspect/bfcl_full`` over ALL 3,981 items of V1 U V2 minus ``rest`` (18 categories;
no seed), one turn of tool calls scored by AST match, thinking OFF, the protocol sampler at T 0.001 executed literally under the served stack's clamp
override (``manifest.json:tinycap.served.sampling_clamp_override``; the rows ``<model id>|bfcl|off|protocol@v1``), ONE pass at 1,024
(``contracts.BFCL200_MAX_TOKENS_OFF``, the same cap as the 200 sibling; no cap step) -- served natively with the tool-call route (REQUIRED here: the task IS tool
calls) on the checkpoint and revision :data:`_bcbw_common.MODEL_KEYS` names, under the DEFAULT Inspect seed rule. Everything else is the shared
reading in :mod:`_bcbw_common`, unchanged -- the INSTRUCT-DIRECT tag included (emission 2; the BigCodeBench module's docstring): gemma4e4b's and qwen35_4b's
``<model key>_it`` rows file here on the BFCL full instrument (llama runs no BFCL row).
"""
from __future__ import annotations

import dataclasses
from typing import Tuple

from . import _bcbw_common as B
from . import bfcl_seedrep as SR
from .base import register

NAME = "bfcl_full"
#: PROVISIONAL, registry-declared (configs/registry/sources_r2_v1.yaml meta.additions): the paper has declared no spelling for these stores
SOURCE = "bfcl_full"
OWNER = B.OWNER
PRODUCER_SESSION = B.PRODUCER_SESSION
CAMPAIGN = B.CAMPAIGN
#: every family of the model-key table (see bigcodebench_full; qwen35_35b_a3b included since 2026-10-04); llama8b runs no BFCL row and so files nothing here unless its stores appear
MODEL_KEYS: Tuple[str, ...] = tuple(B.MODEL_KEYS)
EVAL_ID = SR.EVAL_ID
TASK = "inspect/bfcl_full"
RECORD_TASK = "bfcl_full"
OUTCOMES_FILE = "inspect__bfcl_full.json"
#: the FULL instrument (configs/registry/instruments.yaml), bound per model by the dimensions loader; its item set is ITEM_SET_ID
INSTRUMENT_SEED = "bfcl_full_protocol_T0001_1024"
ITEM_SET_ID = "bfcl_full_3981"
ITEM_SET_LABEL = B.ITEM_SET_LABEL_FULL
ITEM_SET_KEY = "bfcl_full_item_set"
PIN_FILE = "bfcl_full.json"
#: the contract's numbers (verified against configs/capeval/tasks/bfcl_full.json at the producer tree 8ba0b48cf, 2026-09-24 00:32Z): the pin FILE's
#: sha256 and the digest of its ids. Never read by the reader (see bigcodebench_full); the tests hold item_sets.yaml to them
CONTRACT_PIN_FILE_SHA256 = "47afe04e967f6f2284bef576f96179e88efd45004a5b6f81cbfb348530205663"
CONTRACT_IDS_SHA256 = "61d1ac167feced0746d7d8705e49b6577ecf7493f9b54e47020f0a5cb60e2a81"
#: the launched roster of the full run: the SIBLING document of the 200 roster (`configs/capeval/bcb_bfcl_full_roster.yaml`, its legs re-keyed to the
#: full tasks), mirrored under records/ under its own name with per-tree copies `bcb_bfcl_full_roster.<sha7>.yaml` (the 01:36Z mirror)
ROSTER_FILE = "bcb_bfcl_full_roster.yaml"
#: the launch set's name: `launch_sets_full.json` (the addendum of 2026-09-24 02:31Z) or `launch_sets.json` (the first mirrors) -- whichever ONE is mirrored; both at once refuse
LAUNCH_SET_FILES = ("launch_sets_full.json", "launch_sets.json")
CAP = SR.CAP
LADDER: Tuple[int, ...] = SR.LADDER
PROTOCOL_TEMPERATURE = SR.PROTOCOL_TEMPERATURE
#: the producer's three-number record beside the stores, named for the source (the appendix record's convention)
THREE_NUMBERS_FILE = "three_numbers_bfcl_full.json"
SPEC = dataclasses.replace(SR.SPEC, name=NAME, source=SOURCE, task=TASK, record_task=RECORD_TASK, outcomes_file=OUTCOMES_FILE, instrument_seed=INSTRUMENT_SEED,
                           three_numbers_file=THREE_NUMBERS_FILE, item_set_key=ITEM_SET_KEY, root_name=SOURCE, models=MODEL_KEYS, n_items=None,
                           item_set_id=ITEM_SET_ID, item_set_label=ITEM_SET_LABEL, item_set_pin_file=PIN_FILE, roster_file=ROSTER_FILE, launch_set_files=LAUNCH_SET_FILES,
                           instruct_direct=True, unsteered_row=True, dose_relu_row=True, dose_lora_row=True, lecun_relu_row=True)
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
ITEM_SET_RULE = B.ITEM_SET_RULE
APPENDIX_FAMILY_RULE = B.APPENDIX_FAMILY_RULE
LEGACY_MAIN_ALIAS_RULE = B.LEGACY_MAIN_ALIAS_RULE


def run_id_for(tag: str, store_key: str, okey: str, instrument_id: str) -> str:
    return B.run_id_for(SPEC, tag, store_key, okey, instrument_id)


def recount_computation_id() -> str:
    return SPEC.recount_computation_id()


class BfclFullAdapter(B.BcbwSeedrepReader):
    name = NAME
    SPEC = SPEC
    covers_sources: Tuple[str, ...] = (SOURCE,)
    #: 1 (2026-09-24): the first version -- the FULL-suite BFCL stores (all 3,981 items) through the shared reading with the full item-set attestation
    #: (bigcodebench_full's docstring; _bcbw_common.ITEM_SET_RULE) on the new instrument bfcl_full_protocol_T0001_1024 / item set bfcl_full_3981
    #: 2 (2026-09-25): the INSTRUCT-DIRECT tag admitted (bigcodebench_full's docstring; _bcbw_common.INSTRUCT_DIRECT_RULE / INSTRUCT_ANCHOR_RULE): gemma's and qwen's
    #: `<model key>_it` rows (no llama BFCL) file under view_legacy `<model key>_it` with transfer_kind derived instruct_direct and contrast against the family's
    #: s1 none by alias; every seed-tag record re-emits byte for byte (the dry run over the live mirror: 53 rows)
    #: 3 (2026-09-26): the instruct-direct rows' alias contrast against `<model key>_s1/none` (cross_standup true) files anchor_kind sibling_none
    #: (_bcbw_common.anchor_kind_for: another standup of the same model) instead of same_standup_none; the 6 live cross-standup rows move on that field
    #: alone, every other record re-emits byte for byte (the dry run over the live mirror)
    #: 4 (2026-10-03): every contrast is decided by the pass's anchor currency and carries
    #: extra.anchor_selection (`_bcbw_common.CURRENCY_RULE`); a comparison with no current anchor parks on the VALUE row's extra.delta_block
    #: (2026-10-04, NO bump): the qwen35_35b_a3b model key declared in _bcbw_common.MODEL_KEYS (TaskSpec.models binds every key of the table), and an UNDECLARED row
    #: dir under native/ refuses BY NAME as a group of its own (_bcbw_common.UNDECLARED_ROW_DIR_RULE) where discovery used to skip it. Discovery only: every
    #: group that read before re-emits byte for byte over an unchanged input set (the dry discovery over the live mirror: every existing group's inputs equal; two
    #: NEW groups per source, qwen35_35b_a3b_s1 and the undeclared qwen35_35b_a3b-f22u0, both refusing by name). README: a change that only adds groups or
    #: surfaces a refusal parks and spells nothing differently on an existing record, so the version holds
    #: (2026-10-05, NO bump): the qwen35_35b_a3b u0 row read under UNSTEERED_ROW_RULE / WINDOW_ANCHOR_RULE (the `_u0` tag in the s1 group, the `-w<N>` window segment, the
    #: none re-serve keyed none@w<N> from families.<key>.unsteered.window_reserves, window-matched anchors) and the configs pins block (the launch trees'). New rows of a
    #: family that had none in the registry; every 4B group re-emits byte for byte (records hashed per group over the live mirror under origin/main's tree and this one:
    #: 11 of 11 groups equal), so the version holds
    #: (2026-10-05, the merge of main into the re-anchoring branch): both NO-bump notes above held under 3 on main; the bump to 4 is the re-anchoring's
    #: (the anchor currency moves every contrast's bytes), and main's additions still spell nothing differently on an existing record
    EMISSION_VERSION = 4
    owner = OWNER


def factory() -> BfclFullAdapter:
    return BfclFullAdapter()


register(NAME, factory)
