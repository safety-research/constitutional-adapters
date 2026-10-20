"""The FULL-suite BigCodeBench stores -> registry (source ``bigcodebench_full``): every one of the 1,140 tasks, not the pinned 200.

THE CONTRACT (the producer's declaration, 2026-09-24 00:22Z and 00:32Z; quoted so the reader and the mirror can be held to the same words):

* sources ``bigcodebench_full`` / ``bfcl_full``; record tasks ``bigcodebench_full`` / ``bfcl_full``; outcomes files ``inspect__bigcodebench_full.json`` /
  ``inspect__bfcl_full.json``; the harness stamps the LABEL ``item_set`` ``full`` in gen_stats / summary / outcomes (the 200-item passes stamp ``pinned200``);
* item sets: ids ``bigcodebench_full_1140`` (n 1140) and ``bfcl_full_3981`` (n 3981). A new item set is a NEW instrument (instrument identity includes the
  item set; the 200-item instruments are never reused): this reader files on ``bigcodebench_full_protocol_greedy_4096`` (``configs/registry/instruments.yaml``,
  item set ``bigcodebench_full_1140`` in ``item_sets.yaml``), minted through the same seed path as the 200-item readers' instrument;
* attestation: the store MANIFEST carries ``resolved_config.tinycap.bcb_full_item_set`` / ``bfcl_full_item_set`` = ``{seed: null, n: 1140|3981, of: 1140|3981,
  ids_sha256: <hex>}`` -- a digest of the pin's IDS in pin order computed by ``contracts.item_set_identity`` / ``contracts.ids_sha256(pin["ids"])``
  (mirrored as :func:`_bcbw_common.pin_ids_sha256`), NOT of the pin file bytes. The launch-set rows (``records/launch_sets.json``) carry ``item_set {id, sha256,
  n}`` with sha256 = the pin FILE's bytes (``bigcodebench_full_1140`` -> :data:`CONTRACT_PIN_FILE_SHA256`; ``bfcl_full_3981`` -> the BFCL module's).
  ``records/`` holds the mirrored pin files ``bcb_full.json`` + ``bfcl_full.json`` and ``records/item_sets.json`` = list / dict of ``{id, sha256, n, file,
  tree_file}``. A store attests as: manifest ``tinycap.<key>_item_set.ids_sha256 == ids_sha256(pin["ids"])`` of the mirrored pin file whose file sha256 matches
  ``item_sets.json``, AND ``n == of ==`` the item set's n, AND the outcomes file has that many items. Any mismatch REFUSES the store by name (source_status
  refusals, like the roster-hash refusals); an absent ``item_sets.json`` or pin file refuses every store of that source by name. Recorded on each row:
  ``extra.item_set = {id, n, ids_sha256, pin_file, pin_file_sha256, ...}`` and ``run.item_set_id`` = the item-set id (the instrument binds it; a seed binding
  another set refuses);
* everything else -- the roster set by hash, the launched pins per tree, the canary alias ``_main`` -> ``_s1``, the model-key table, the ``lora_native`` gate PASS,
  the decode-policy derivation (``decode_policy_id`` / ``decode_policy_match`` are the harvester's, never stamped here) -- is inherited from :mod:`_bcbw_common`
  unchanged.

WHAT THIS READER DECIDES where the contract is silent (each refuses by name rather than widening; each a one-line change here or in :class:`_bcbw_common.TaskSpec`):
the item COUNT is the mirror's (``item_sets.json`` n = the pin's ``n`` = ``of`` = the ids listed), never a literal 1,140 -- and it must equal the registry's
declaration (``item_sets.yaml`` ``n_items`` / ``items_sha256``: the ID -> expected-count table); the pin's ``task`` must read ``bigcodebench_full``; the
launch set's row table sits under ``evals.bigcodebench`` (the registry eval, as every mirror so far; ``TaskSpec.launch_set_eval``); the three-number record is
``three_numbers_bigcodebench_full.json`` with ``eval bigcodebench`` (named for the source, as the appendix record is); an outcome record short of n refuses by
name (the core's convention for an incomplete store: no "declared shortfall" shape exists yet, so none is read); the source binds EVERY model key of the table
(the full run spans the families in one root; a family outside the table still refuses). The 200-item readers are untouched: their specs carry no item set
and read byte for byte.

MEASURED on the first mirror (01:55Z, 2026-09-24; the ``none`` canaries of qwen s1 / s2 / s3, gemma, llama, nemotron): every contract fact above holds
byte for byte (the ``item_sets`` list form, ``evals.bigcodebench``, the manifest blocks, the ``full`` labels, 1,140 outcome items, the record
``three_numbers_bigcodebench_full.json``; the launch set even names ``instrument_seed_id bigcodebench_full_protocol_greedy_4096``), and the item set resolves
with no problem. The producer writes ``rosters.json:rosters`` as a LIST (``{roster_sha256, file, tree_sha, pins_file, pins_sha256, stores}``) where the
rosters branch's parse read a sha-keyed mapping -- ``records/rosters.json names no roster`` on every store of the full AND the live appendix mirrors at
``429ed730c`` -- so the shared parse reads both shapes now and verifies each roster's pins copy by ``pins_sha256``; every canary then files (0.3956-0.4956
over 1,140; the values the producer's record's). The full run's roster file is ``bcb_bfcl_full_roster.yaml`` (:data:`ROSTER_FILE`; ``TaskSpec.roster_file``).

THE ADDENDUM (2026-09-24 02:31Z; additive; :data:`_bcbw_common.RELAUNCH_RULE` / :data:`_bcbw_common.TREES_RULE`): a second producer tree beside the
first (its roster and its pins copy -- byte-different, content-identical -- matched by hash within the set); the launch set as ``launch_sets_full.json`` or
``launch_sets.json`` (whichever ONE is mirrored, :data:`LAUNCH_SET_FILES`), its ``trees [{sha, role}]`` and a row's ``sha`` (absent = the primary tree) bound to
the store's tree; a RELAUNCHED row's ``launched_as`` (the ``-r<N>`` store) and ``relaunched_from`` (the parked INCOMPLETE first store: parked, never read,
never refused); ``relaunches.json`` (``records/`` or ``bcbw_records_full/``) naming the relaunch by ``<wave>/<cell>``; a gate at the relaunch tree recorded
with ``cross_tree`` / ``compared_against``. Anything else refuses by name.

THE INSTRUCT-DIRECT TAG (declared 2026-09-25 19:13Z; :data:`_bcbw_common.INSTRUCT_DIRECT_RULE` / :data:`_bcbw_common.INSTRUCT_ANCHOR_RULE`; emission 2):
the instruct-checkpoint LoRA fits of gemma4e4b / llama8b / qwen35_4b (constitution_lora / control_lora / difference_lora; llama BigCodeBench only) run under the tag
``<model key>_it`` -- row dir ``<model key>-f22it``, stores ``native/<model key>-f22it/<object key>/thinking_off/bcbw-<model key>_it-<sha7>``, gate records
``gate_<model key>_it_…``, three-number keys ``<model key>_it/<object key>``, launch-set rows whose selection is ``it``, ``block`` and ``transfer_kind instruct_direct``,
``seed_base null``, the fitted class (:data:`_bcbw_common.INSTRUCT_CLASS`) and ``anchor_tuple s1/none``. They file under ``view_legacy <model key>_it`` (the spelling every instruct-direct row of these models
carries in the registry -- legacy_spellings.yaml spells it as the instruct model, fit ``<model key>_it``, transfer_kind instruct_direct) on the family's
``s1`` group, their object resolved through the harness roster's ``it`` selection for the family, their ``transfer_kind`` DERIVED from the store's
``fit_provenance.model_id`` against the served ``model_id`` (equal -> instruct_direct; a base fit refuses by name) and their contrast formed against
``<model key>_s1/none`` BY ALIAS (``anchor_kind sibling_none``: the same model's ``none`` served by another standup, ``_bcbw_common.anchor_kind_for``;
``cross_standup`` true;
``extra.anchor_alias``): the launch set's ``it/none`` row is class ``anchor`` with no store, so NO ``_it`` none run files (a ``_it`` none store, a prompted or
all-positions key under ``_it``, a nemotron ``_it`` store, a row spelling a seed base, another block / transfer kind / anchor tuple, a pin key that is not
the roster's staged id, a missing expected sha, or a store attesting another object each refuse by name). The 200-item readers admit none of this
(``TaskSpec.instruct_direct`` false; their emission is pinned byte for byte).

Every row is the BigCodeBench FULL instrument -- inspect_evals 0.18.0's ``bigcodebench`` task as row task ``inspect/bigcodebench_full`` over ALL 1,140 ids of
the v0.1.2 complete split (Hard included; no seed), pass@1 in the pinned sandbox image, thinking OFF, the benchmark's reference protocol sampler (greedy: T 0 /
top_p 1 / top_k 0; the rows ``<model id>|bigcodebench|off|protocol@v1``), ONE pass at 4,096 (``contracts.BCB200_MAX_TOKENS_OFF``, the same cap as the 200
sibling; no cap step) -- served natively on the checkpoint and revision :data:`_bcbw_common.MODEL_KEYS` names, under the DEFAULT Inspect seed rule.
"""
from __future__ import annotations

import dataclasses
from typing import Tuple

from . import _bcbw_common as B
from . import bigcodebench_seedrep as SR
from .base import register

NAME = "bigcodebench_full"
#: PROVISIONAL, registry-declared (configs/registry/sources_r2_v1.yaml meta.additions): the paper has declared no spelling for these stores
SOURCE = "bigcodebench_full"
OWNER = B.OWNER
PRODUCER_SESSION = B.PRODUCER_SESSION
CAMPAIGN = B.CAMPAIGN
#: every family of the model-key table: the full run spans them in one root (a store of a key outside the table still refuses its group by name); since
#: 2026-10-04 the table includes qwen35_35b_a3b (the pipeline test's served checkpoint), whose s1 row this reader discovers -- and whose undeclared u0 row refuses by name
MODEL_KEYS: Tuple[str, ...] = tuple(B.MODEL_KEYS)
EVAL_ID = SR.EVAL_ID
TASK = "inspect/bigcodebench_full"
RECORD_TASK = "bigcodebench_full"
OUTCOMES_FILE = "inspect__bigcodebench_full.json"
#: the FULL instrument (configs/registry/instruments.yaml), bound per model by the dimensions loader; its item set is ITEM_SET_ID
INSTRUMENT_SEED = "bigcodebench_full_protocol_greedy_4096"
ITEM_SET_ID = "bigcodebench_full_1140"
ITEM_SET_LABEL = B.ITEM_SET_LABEL_FULL
ITEM_SET_KEY = "bcb_full_item_set"
PIN_FILE = "bcb_full.json"
#: the contract's numbers (2026-09-24 00:32Z; verified against configs/capeval/tasks/bcb_full.json at the producer tree 8ba0b48cf): the pin FILE's
#: sha256 and the digest of its ids. The reader never reads these constants -- the mirror's item_sets.json / pin file and item_sets.yaml are compared to each
#: other at read time -- the tests hold item_sets.yaml to them
CONTRACT_PIN_FILE_SHA256 = "1aa5f1e4f6476dabad879e5fe8d5ad5eeaa01debd7f693a335d53d5651c4c9f2"
CONTRACT_IDS_SHA256 = "c309c92f137762955c16f8546e0ff9b1e84577f854fbba022904e493e3198a07"
#: the launched roster of the full run: the SIBLING document of the 200 roster (`configs/capeval/bcb_bfcl_full_roster.yaml`, its legs re-keyed to the
#: full tasks), mirrored under records/ under its own name with per-tree copies `bcb_bfcl_full_roster.<sha7>.yaml` (the 01:36Z mirror)
ROSTER_FILE = "bcb_bfcl_full_roster.yaml"
#: the launch set's name: `launch_sets_full.json` (the addendum of 2026-09-24 02:31Z) or `launch_sets.json` (the first mirrors) -- whichever ONE is mirrored; both at once refuse
LAUNCH_SET_FILES = ("launch_sets_full.json", "launch_sets.json")
CAP = SR.CAP
LADDER: Tuple[int, ...] = SR.LADDER
PROTOCOL_TEMPERATURE = SR.PROTOCOL_TEMPERATURE
#: the producer's three-number record beside the stores, named for the source (the appendix record's convention)
THREE_NUMBERS_FILE = "three_numbers_bigcodebench_full.json"
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


class BigcodebenchFullAdapter(B.BcbwSeedrepReader):
    name = NAME
    SPEC = SPEC
    covers_sources: Tuple[str, ...] = (SOURCE,)
    #: 1 (2026-09-24): the first version -- the FULL-suite BigCodeBench stores (all 1,140 tasks) through the shared reading with the full item-set
    #: attestation (module docstring; _bcbw_common.ITEM_SET_RULE) on the new instrument bigcodebench_full_protocol_greedy_4096 / item set bigcodebench_full_1140
    #: 2 (2026-09-25): the INSTRUCT-DIRECT tag admitted (module docstring; _bcbw_common.INSTRUCT_DIRECT_RULE / INSTRUCT_ANCHOR_RULE): `<model key>_it` rows file
    #: under view_legacy `<model key>_it` with transfer_kind derived instruct_direct and contrast against the family's s1 none by alias; every seed-tag record
    #: re-emits byte for byte (the dry run over the live mirror: 63 rows)
    #: 3 (2026-09-26): the instruct-direct rows' alias contrast against `<model key>_s1/none` (cross_standup true) files anchor_kind sibling_none
    #: (_bcbw_common.anchor_kind_for: another standup of the same model) instead of same_standup_none; the 9 live cross-standup rows move on that field
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


def factory() -> BigcodebenchFullAdapter:
    return BigcodebenchFullAdapter()


register(NAME, factory)
