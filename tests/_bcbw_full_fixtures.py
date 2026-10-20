"""A SYNTHETIC mirror shaped like the FULL-suite BigCodeBench / BFCL stores (the contract of 2026-09-24 00:22Z / 00:32Z, quoted in
`adapters/bigcodebench_full.py`; invented item ids and shas, `TESTHOST`, no generation text), shared by the two full readers' tests
(`test_registry_adapter_bigcodebench_full.py`, `test_registry_adapter_bfcl_full.py`). It reuses the seed-replicate fixture's writers
(`_bcbw_seedrep_fixtures`, bound to the FULL :class:`Shape`) with the gemma4e4b family under its `s1` / `s1p` rows (no canary: the alias is the appendix
readers' test), and builds what the full mirror adds:

* the row tasks `inspect/bigcodebench_full` / `inspect/bfcl_full`, the record tasks `bigcodebench_full` / `bfcl_full`, the outcomes files
  `inspect__<task>_full.json`, the LABEL `full` on gen_stats / the summary / the outcome record, and the manifest's `tinycap.bcb_full_item_set` /
  `bfcl_full_item_set` = `{seed: null, n, of, ids_sha256}` over a SMALL pin -- `N_FULL` (5) ids per task, so the reader is seen to take the count from the
  mirror and never from a literal 1,140 / 3,981;
* `records/item_sets.json` (a list of `{id, sha256, n, file, tree_file}`; `item_sets_layout="dict"` writes the id-keyed mapping) and the mirrored pin
  files `records/bcb_full.json` / `bfcl_full.json` (`{schema, task, seed: null, n, of, ids, rule, dataset}`), whose file sha256 the index and every
  launch-set row (`item_set {id, sha256, n}`) carry; the three-number records `three_numbers_<eval>_full.json`;
* the REGISTRY's declaration of the set: :func:`seeds_for` hands the adapter a copy of the seeds whose `item_sets.yaml` row for the set declares the
  fixture's count and ids digest (the real rows declare 1,140 / 3,981 and the real pins' digests; the reader compares the two, so a fixture pin needs a
  matching declaration -- `registry_n` / `registry_sha` override it to make the check fire).
Knobs (each a refusal the tests pin): `no_item_sets` / `no_pin_file`, `item_sets_override(entries)`, `pin_override(pin doc)` (the file is re-hashed, so
the index still matches unless `index_sha_stale`), `index_sha_stale` (the index keeps the sha256 of the unedited pin), `manifest_item_set_override(block)`,
`label_override` (`pinned200` where `full` is stamped), `short_outcomes` (one item dropped from the outcome record alone), `launch_item_set_override(block)`
/ `launch_item_set_drop` (rows by `<selection>/<key>`, e.g. `s1/none`), `qwen_store` (a three-seed-family store: the full sources bind every family),
`qwen35b_store` (the qwen35_35b_a3b family's `s1` row, 2026-10-04: its `none` and ONE fitted LoRA key served on the 35B instruct checkpoint and fit on its base,
resolved through the appendix route -- the launch set's pin key is the staged id, pinned under the model's block of the harness AND the launched pins),
`undeclared_row_dirs=(name, ...)` (a complete `none` store under `native/<name>/` that no record names: `_bcbw_common.UNDECLARED_ROW_DIR_RULE`, a group of its own
that refuses by name -- a key the table lacks, a selection the grammar lacks, or the s2rl row dir of a family that runs no LeCun-init ReLU row / under a reader that does
not admit it, whose store then carries the row dir's own launch tag) and `dot_dir_store` (the same store under `native/.staging/<row dir>/`, a producer's publish scaffolding: never discovered).
`qwen35b_d0p75_relul` (with `qwen35b_u0`; 2026-10-05, `_bcbw_common.DOSE_RELU_ROW_RULE`): the dose-0.75 random-init ReLU row's three steered stores, launch rows
(selection d0p75_relul, anchor_tuple u0/none), three-number rows and gate records, the three objects pinned under the model's block; `u0_second_none_store`: a second
`none` store under the u0 row at the dose launch's tree (TREE_DR), no record naming it -- it shares the (u0, none) row key with the declared store and binds no row.
`qwen35b_d1p25` (with `qwen35b_u0`; `_bcbw_common.DOSE_LORA_ROW_RULE`): the dose-1.25 difference-LoRA row's one steered store (the s1 difference key at x1.25), its launch
row (selection d1p25, anchor_tuple u0/none), three-number row and gate record, the object pinned under the model's block.
`qwen35b_s2rl` (with `qwen35b_u0`; 2026-10-07, `_bcbw_common.LECUN_RELU_ROW_RULE`): the LeCun-init ReLU row's TWO steered stores (the control and constitution ReLU refits,
fits with a recorded fit seed of 0), launch rows (selection s2rl, anchor_tuple u0/none, seed_base the group's 1), three-number rows and gate records, the two objects pinned
under the model's block by their `lecun__` staged ids; the harness roster lists a THIRD key under the s2rl selection (the composed difference the launch never ran: no store,
no launch row, no pin).
The ADDENDUM of 2026-09-24 02:31Z (`_bcbw_common.RELAUNCH_RULE` / `TREES_RULE`): `records/rosters.json` is written in the producer's LIST shape by default
(`rosters_layout="mapping"` writes the rosters branch's sha-keyed mapping), each entry naming its per-tree pins copy by `pins_file` / `pins_sha256`;
`relaunched=(<key>,)` launches that seed-tag key AGAIN at a second producer tree `TREE2`: the parked first store under `bcbw-<tag>-<TREE sha7>` with an
INCOMPLETE RESULT, the relaunched store under `bcbw-<tag>-<TREE2 sha7>-r2` with its gate record at TREE2 (`compared_against` the first record, `cross_tree`
true), a second per-tree pins copy that differs from the first in KEY ORDER alone (content identical, sha256 different), the launch set's `trees`
(`TREE` primary, `TREE2` relaunch) and the row's `sha` / `launched_as` / `relaunched_from`, and `relaunches.json` (under records/, or under
`bcbw_records_full/` with `relaunches_dir`); `launch_set_name` writes the row table as `launch_sets.json` (the first mirrors), `launch_sets_full.json`
(the addendum) or `both`; `row_sha_override` / `pins_sha_stale` / `no_relaunches` / `relaunch_entry_override` bend one record each."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from typing import Callable, Dict, List, Optional, Sequence

import _bcbw_allmodels_fixtures as Y
import _bcbw_seedrep_fixtures as X
from sisteer.registry.adapters import _bcbw_common as B
from sisteer.registry.adapters import bfcl_full as BFF
from sisteer.registry.adapters import bigcodebench_full as BCF
from sisteer.registry.adapters import lineage_common as LC

MK, MODEL_ID, MODEL_HF, FIT_HF = Y.MK, Y.MODEL_ID, Y.MODEL_HF, Y.FIT_HF
SEED_N, TAG, TAG_P, SEL, SEL_P, ROW_DIR, ROW_DIR_P = Y.SEED_N, Y.TAG, Y.TAG_P, Y.SEL, Y.SEL_P, Y.ROW_DIR, Y.ROW_DIR_P
NONE, RELU, PROMPTED, COMPOSED, CONTROL_RELU = Y.NONE, Y.RELU, Y.PROMPTED, Y.COMPOSED, Y.CONTROL_RELU
STAGED, STAGED_CTL, STAGED_P, OBJ, OBJ_CTL, OBJ_P = Y.STAGED, Y.STAGED_CTL, Y.STAGED_P, Y.OBJ, Y.OBJ_CTL, Y.OBJ_P
TREE, SHA7, NOW = X.TREE, X.SHA7, X.NOW
#: the second producer tree of a relaunch (the addendum's f1cd7f2 beside 8ba0b48)
TREE2 = hashlib.sha1(b"synthetic bcbw full relaunch tree").hexdigest()
#: a third tree: a parked-only attempt (4bb77ea / 91f0edd, 2026-09-25 21:17Z), no live store at it
TREE3 = hashlib.sha1(b"synthetic bcbw full parked-only tree").hexdigest()
RELAUNCH_TAG = "r2"
PINS_FILE_T1, PINS_FILE_T2 = f"{B.PINS_FILE[:-5]}.{TREE[:7]}.json", f"{B.PINS_FILE[:-5]}.{TREE2[:7]}.json"
ROSTER_FILE_T1 = f"{BCF.SPEC.roster_file[:-5]}.{TREE[:7]}.yaml"
ROSTER_FILE_T2 = f"{BCF.SPEC.roster_file[:-5]}.{TREE2[:7]}.yaml"
QWEN_TAG = Y.QWEN_TAG
#: the SMALL full pin: the reader takes the count from the mirror (item_sets.json / the pin), never from a literal 1,140 / 3,981
N_FULL = 5
#: THE INSTRUCT-DIRECT TAG (declared 2026-09-25; `_bcbw_common.INSTRUCT_DIRECT_RULE`): `mirror(instruct=True)` adds the gemma4e4b family's `_it` row --
#: ONE fitted store (the instruct-checkpoint constitution_lora fit, staged `itdoc__…` by the harness roster's `it` selection, aliased `d1it__…` in the pins) under
#: `native/gemma4e4b-f22it/`, its gate record, its launch-set rows (the fitted row: selection `it`, block / transfer_kind instruct_direct, seed_base null, class
#: `arm`, anchor_tuple `s1/none`, pin_key_alias; the `it/none` row: class anchor, NO store -- the alias to the s1 none) and its three-number rows
IT_TAG = B.instruct_tag_of(MK)
IT_SEL = B.INSTRUCT_SELECTION
IT_ROW_DIR = B.row_dir_of(IT_TAG)
IT_KEY = "constv2-mix-full__lora_r4_ce"
IT_STAGED = f"itdoc__{IT_KEY}"
IT_ALIAS = f"d1it__{IT_KEY}"
IT_OBJ = hashlib.sha256(b"synthetic bcbw instruct-direct gemma constitution lora").hexdigest()
IT_IHASH8, IT_CHASH8 = "5e6f7a8b", "9c0d1e2f"
#: the `_it` row's synthetic truth (every item scored; the bfcl row one at-cap wrong item), and the aliased anchor's is SPECS[<eval>][NONE]
SPECS_IT = {"bigcodebench": dict(n_correct=4, n_wrong=1), "bfcl": dict(n_correct=2, n_wrong=2, at_cap_wrong=1)}
#: THE qwen35_35b_a3b FAMILY (2026-10-04; `_bcbw_common.MODEL_KEYS["qwen35_35b_a3b"]`; `mirror(qwen35b_store=True)`): the `s1` row's `none` and the constitution LoRA the
#: real mirror carries, served on the 35B instruct checkpoint (models.yaml hf_repo / hf_revision) and fit on its base (fit_provenance.model_id: base_to_instruct)
MK35 = "qwen35_35b_a3b"
MODEL_ID35 = B.MODEL_KEYS[MK35].model_id
MODEL_HF35 = B.MODEL_KEYS[MK35].served_hf_id
FIT_HF35 = "Qwen/Qwen3.5-35B-A3B-Base"
TAG35 = f"{MK35}_s1"
SEL35 = "s1"
ROW_DIR35 = B.row_dir_of(TAG35)
LORA35 = "constv2-mix-full__lora_r4_ce"
STAGED35 = f"base__{LORA35}"
OBJ35 = hashlib.sha256(b"synthetic bcbw qwen35 35b a3b constitution lora").hexdigest()
IHASH35, CHASH35 = "155fd054", "41937309"
#: (object key, staged id, object sha, ihash8, chash8, max_lora_rank) -- the 35B row's two stores
ROWS35 = [(NONE, None, None, "2ff9cdff", "d36eb87c", 1), (LORA35, STAGED35, OBJ35, IHASH35, CHASH35, 4)]
SPECS35 = {"bigcodebench": {NONE: dict(n_correct=3, n_wrong=2), LORA35: dict(n_correct=2, n_wrong=2, at_cap_wrong=1)},
           "bfcl": {NONE: dict(n_correct=4, n_wrong=1), LORA35: dict(n_correct=3, n_wrong=2)}}
#: an UNDECLARED row dir (`_bcbw_common.UNDECLARED_ROW_DIR_RULE`; `mirror(undeclared_row_dirs=(name, ...))`): a model key the table lacks; a dot-prefixed entry is scaffolding
UNDECLARED_ROW_DIR = "glm46vflash-f22s1"
DOT_DIR = ".staging"
#: THE u0 FAMILY (2026-10-05; `_bcbw_common.UNSTEERED_ROW_RULE` / `WINDOW_ANCHOR_RULE`; `mirror(qwen35b_u0=True)`): the real mirror's shape -- the u0 row's `none` at the
#: default window (8,192 x 32), the `none` RE-SERVED at the constitution window as its own store (`-w14336`, 14,336 x 18; declared by
#: families.<key>.unsteered.window_reserves, no launch-set row), the prompted safety key at the default window, the prompted constitution key at 14,336; and the s1
#: row's three LoRA keys at the default window (anchor_tuple u0/none). Two-task stores (the fixture's shape; the real one-task `-bcb` / `-bfcl` stores read through
#: ONE_TASK_STORE_RULE, pinned by the full modules' own tests)
TAG35_U0 = B.unsteered_tag_of(MK35)
SEL35_U0 = B.UNSTEERED_SELECTION
ROW_DIR35_U0 = B.row_dir_of(TAG35_U0)
WINDOW35, WINDOW35_SEQS = 8192, 32
WINDOW35_CONST, WINDOW35_CONST_SEQS = 14336, 18
WINDOW35_SEG = f"-{B.WINDOW_MARKER}{WINDOW35_CONST}"
CONST35 = "prompted__si_constitution"
CTL35 = "constv2ctl2-full__lora_r4_ce"
DIFF35 = "constv2-mix-full__lora_r4_ce_minus_ctl"
STAGED35_CTL, STAGED35_DIFF = f"base__{CTL35}", f"derived__{DIFF35}"
OBJ35_CTL = hashlib.sha256(b"synthetic bcbw qwen35 35b a3b control lora").hexdigest()
OBJ35_DIFF = hashlib.sha256(b"synthetic bcbw qwen35 35b a3b difference lora").hexdigest()
#: (object key, tag, staged id, object sha, ihash8, chash8, max_lora_rank, max_model_len, max_num_seqs, suffix extra) -- the family's seven stores
ROWS35_U0 = [(NONE, TAG35_U0, None, None, "2ff9cdff", "d36eb87c", 1, WINDOW35, WINDOW35_SEQS, ""),
             (NONE, TAG35_U0, None, None, "2ff9cdff", "8fc0333d", 1, WINDOW35_CONST, WINDOW35_CONST_SEQS, WINDOW35_SEG),
             (PROMPTED, TAG35_U0, None, None, "faa4c8cf", "9a09771b", 1, WINDOW35, WINDOW35_SEQS, ""),
             (CONST35, TAG35_U0, None, None, "98eaea2d", "a2f5e15c", 1, WINDOW35_CONST, WINDOW35_CONST_SEQS, ""),
             (CTL35, TAG35, STAGED35_CTL, OBJ35_CTL, "a8bcfedb", "76b06196", 4, WINDOW35, WINDOW35_SEQS, ""),
             (LORA35, TAG35, STAGED35, OBJ35, IHASH35, CHASH35, 4, WINDOW35, WINDOW35_SEQS, ""),
             (DIFF35, TAG35, STAGED35_DIFF, OBJ35_DIFF, "47c3dbc8", "6a0d3984", 4, WINDOW35, WINDOW35_SEQS, "")]
SPECS35_U0 = {"bigcodebench": {NONE: dict(n_correct=3, n_wrong=2), f"{NONE}@w": dict(n_correct=4, n_wrong=1), PROMPTED: dict(n_correct=3, n_wrong=1, at_cap_wrong=1), CONST35: dict(n_correct=2, n_wrong=3),
                               CTL35: dict(n_correct=3, n_wrong=2), LORA35: dict(n_correct=2, n_wrong=2, at_cap_wrong=1), DIFF35: dict(n_correct=2, n_wrong=3)},
              "bfcl": {NONE: dict(n_correct=4, n_wrong=1), f"{NONE}@w": dict(n_correct=4, n_wrong=1), PROMPTED: dict(n_correct=4, n_wrong=1), CONST35: dict(n_correct=3, n_wrong=2),
                       CTL35: dict(n_correct=4, n_wrong=1), LORA35: dict(n_correct=3, n_wrong=2), DIFF35: dict(n_correct=3, n_wrong=2)}}
#: THE DOSE-0.75 RANDOM-INIT ReLU ROW (2026-10-05; `_bcbw_common.DOSE_RELU_ROW_RULE`; `mirror(qwen35b_u0=True, qwen35b_d0p75_relul=True)`): three steered rows under
#: `qwen35_35b_a3b-f22d0p75_relul` -- the two parent dose copies (the control and constitution ReLU objects at x0.75) and their difference -- every one at the default
#: window and anchoring on the u0 none (anchor_tuple u0/none); the objects resolve through the appendix route (the pin key is the bank's `d0p75l__` staged id)
TAG35_DR = B.dose_relu_tag_of(MK35)
SEL35_DR = B.DOSE_RELU_SELECTION
ROW_DIR35_DR = B.row_dir_of(TAG35_DR)
CTL35_R, MIX35_R, DIFF35_R = "constv2ctl2-full__psr_ce_all", "constv2-mix-full__psr_ce_all", "constv2-mix-full__psr_ce_all_minus_ctl"
STAGED35_DR = {k: f"d0p75l__{k}" for k in (CTL35_R, MIX35_R, DIFF35_R)}
OBJ35_DR = {k: hashlib.sha256(f"synthetic bcbw qwen35 35b a3b dose 0.75 random-init relu {k}".encode()).hexdigest() for k in (CTL35_R, MIX35_R, DIFF35_R)}
#: (object key, ihash8, chash8) -- max_lora_rank 1 (a ReLU-steer object, as the gemma family's ReLU row is staged), the default window
ROWS35_DR = [(CTL35_R, "7c0d75a1", "c0ffee01"), (MIX35_R, "7c0d75a2", "c0ffee02"), (DIFF35_R, "7c0d75a3", "c0ffee03")]
SPECS35_DR = {"bigcodebench": {CTL35_R: dict(n_correct=3, n_wrong=2), MIX35_R: dict(n_correct=2, n_wrong=3), DIFF35_R: dict(n_correct=4, n_wrong=1)},
              "bfcl": {CTL35_R: dict(n_correct=4, n_wrong=1), MIX35_R: dict(n_correct=3, n_wrong=2), DIFF35_R: dict(n_correct=3, n_wrong=1, at_cap_wrong=1)}}
PINS35_DR = {STAGED35_DR[k]: {"object_sha256": OBJ35_DR[k], "source": "synthetic"} for k in (CTL35_R, MIX35_R, DIFF35_R)}
#: the dose launch's tree: a second `none` under the u0 row at this tree (`mirror(u0_second_none_store=True)`) shares the (u0, none) row key with the declared store
TREE_DR = hashlib.sha1(b"synthetic bcbw dose-0.75 random-init relu launch tree").hexdigest()
#: THE DOSE-1.25 DIFFERENCE-LoRA ROW (2026-10-05; `_bcbw_common.DOSE_LORA_ROW_RULE`; `mirror(qwen35b_u0=True, qwen35b_d1p25=True)`): ONE steered row under
#: `qwen35_35b_a3b-f22d1p25` -- the dose_scale copy of the s1 row's difference LoRA at base_scale 1.25 -- at the default window, anchoring on the u0 none
#: (anchor_tuple u0/none); the object resolves through the appendix route (the pin key is the bank's `d1p25__` staged id)
TAG35_DL = B.dose_lora_tag_of(MK35)
SEL35_DL = B.DOSE_LORA_SELECTION
ROW_DIR35_DL = B.row_dir_of(TAG35_DL)
STAGED35_DL = f"d1p25__{DIFF35}"
OBJ35_DL = hashlib.sha256(b"synthetic bcbw qwen35 35b a3b dose 1.25 difference lora").hexdigest()
SPECS35_DL = {"bigcodebench": dict(n_correct=2, n_wrong=3), "bfcl": dict(n_correct=4, n_wrong=1)}
PINS35_DL = {STAGED35_DL: {"object_sha256": OBJ35_DL, "source": "synthetic"}}
#: THE LeCun-INIT ReLU ROW (2026-10-07; `_bcbw_common.LECUN_RELU_ROW_RULE`; `mirror(qwen35b_u0=True, qwen35b_s2rl=True)`): TWO steered rows under `qwen35_35b_a3b-f22s2rl` -- the
#: control and constitution ReLU objects REFIT from a LeCun random init (fits, not dose copies; the bank sidecars record fit seed 0) -- at the default window, anchoring on the
#: u0 none (anchor_tuple u0/none); each resolves through the appendix route (the pin key is the bank's `lecun__` staged id). The harness roster's s2rl selection lists a THIRD
#: key, the composed difference `lecun_derived__...`, which the launch never ran: no store, no launch row, no pin -- it is not a row
TAG35_SR = B.lecun_relu_tag_of(MK35)
SEL35_SR = B.LECUN_RELU_SELECTION
ROW_DIR35_SR = B.row_dir_of(TAG35_SR)
#: (object key, ihash8, chash8) -- max_lora_rank 1 (a ReLU-steer object), the default window
ROWS35_SR = [(CTL35_R, "5ec0e001", "c0ffee21"), (MIX35_R, "5ec0e002", "c0ffee22")]
STAGED35_SR = {k: f"lecun__{k}" for k in (CTL35_R, MIX35_R)}
STAGED35_SR_THIRD = f"lecun_derived__{DIFF35_R}"
OBJ35_SR = {k: hashlib.sha256(f"synthetic bcbw qwen35 35b a3b lecun-init relu refit {k}".encode()).hexdigest() for k in (CTL35_R, MIX35_R)}
SPECS35_SR = {"bigcodebench": {CTL35_R: dict(n_correct=4, n_wrong=1), MIX35_R: dict(n_correct=2, n_wrong=2, at_cap_wrong=1)},
              "bfcl": {CTL35_R: dict(n_correct=3, n_wrong=2), MIX35_R: dict(n_correct=4, n_wrong=1)}}
PINS35_SR = {STAGED35_SR[k]: {"object_sha256": OBJ35_SR[k], "source": "synthetic"} for k in (CTL35_R, MIX35_R)}
#: the full run's launched roster file (the sibling document's name, not the 200 roster's)
ROSTER_FILE = BCF.SPEC.roster_file
MODULES = {"bigcodebench": BCF, "bfcl": BFF}
#: per eval: (task, record task, outcomes file, cap, protocol T, clamp override?, manifest item-set key, of) -- the FULL spellings
TASKS_FULL = {"bigcodebench": (BCF.TASK, BCF.RECORD_TASK, BCF.OUTCOMES_FILE, BCF.CAP, BCF.PROTOCOL_TEMPERATURE, False, BCF.ITEM_SET_KEY, N_FULL),
              "bfcl": (BFF.TASK, BFF.RECORD_TASK, BFF.OUTCOMES_FILE, BFF.CAP, BFF.PROTOCOL_TEMPERATURE, True, BFF.ITEM_SET_KEY, N_FULL)}
#: the pins' ids (BigCodeBench's pinned ids are `BigCodeBench/<n>`; its rows and outcome record carry the log id str(n); BFCL's ids are one spelling)
PIN_IDS = {"bigcodebench": [f"BigCodeBench/{i}" for i in range(N_FULL)], "bfcl": [f"exec_multiple_{i}" for i in range(N_FULL)]}
ITEM_IDS_FULL = {"bigcodebench": [str(i) for i in range(N_FULL)], "bfcl": list(PIN_IDS["bfcl"])}
IDS_SHA = {ev: B.pin_ids_sha256(PIN_IDS[ev]) for ev in TASKS_FULL}
ITEM_SET_BLOCKS = {BCF.ITEM_SET_KEY: {"ids_sha256": IDS_SHA["bigcodebench"], "n": N_FULL, "of": N_FULL, "seed": None},
                   BFF.ITEM_SET_KEY: {"ids_sha256": IDS_SHA["bfcl"], "n": N_FULL, "of": N_FULL, "seed": None}}
SHAPE_FULL = X.Shape(TASKS_FULL, N_FULL, ITEM_IDS_FULL, B.ITEM_SET_LABEL_FULL, ITEM_SET_BLOCKS)
#: ONE-TASK shapes (the addendum of 2026-09-25 19:58Z; `_bcbw_common.ONE_TASK_STORE_RULE`): a store holding one of the two tasks alone
SHAPE_ONE_TASK = {ev: X.Shape({ev: TASKS_FULL[ev]}, N_FULL, ITEM_IDS_FULL, B.ITEM_SET_LABEL_FULL, ITEM_SET_BLOCKS) for ev in TASKS_FULL}
#: the two one-task stores of `mirror(instruct_two_stores=True)`: the BFCL half at TREE under the bare suffix, the BigCodeBench half at TREE2 under the `-bcb` task tag
TWO_STORES_TASK_TAG = {"bigcodebench": "-bcb", "bfcl": ""}
THREE_NUMBERS_FILES = {ev: MODULES[ev].THREE_NUMBERS_FILE for ev in TASKS_FULL}
#: per eval: the synthetic truth of every row over the N_FULL items (the paired `none` fewer correct than the seed `none`)
SPECS = {"bigcodebench": {NONE: dict(n_correct=3, n_wrong=2), RELU: dict(n_correct=2, n_wrong=1, at_cap_wrong=1, at_cap_unscored=1), PROMPTED: dict(n_correct=3, n_wrong=1, at_cap_wrong=1),
                          COMPOSED: dict(n_correct=2, n_wrong=2, below_cap_unscored=1)},
         "bfcl": {NONE: dict(n_correct=4, n_wrong=1), RELU: dict(n_correct=3, n_wrong=1, at_cap_wrong=1), PROMPTED: dict(n_correct=4, n_wrong=1), COMPOSED: dict(n_correct=3, n_wrong=2)}}
SPECS_P = {"bigcodebench": dict(n_correct=1, n_wrong=4), "bfcl": dict(n_correct=2, n_wrong=3)}
#: (object key, paired?, staged id, object sha, ihash8, chash8, max_lora_rank) -- the appendix fixture's rows (the s1 / s1p pin-key route)
ROWS = list(Y.ROWS)
ROW_P = Y.ROW_P


def bind_seeds(seeds: LC.Seeds) -> None:
    Y.bind_seeds(seeds)


def seeds_for(seeds: LC.Seeds, *, registry_n: Optional[Dict[str, int]] = None, registry_sha: Optional[Dict[str, str]] = None) -> LC.Seeds:
    """A copy of the seeds whose item_sets.yaml rows for the two full sets declare the FIXTURE's count and digest (the reader checks the registry's declaration
    against the mirrored pin, so a 5-id fixture needs a 5-id declaration); `registry_n` / `registry_sha` (per eval) put another value there instead."""
    bind_seeds(seeds)
    s2 = copy.copy(seeds)
    s2.item_sets = dict(seeds.item_sets)
    for ev, mod in MODULES.items():
        row = dict(seeds.item_sets[mod.ITEM_SET_ID])
        row["n_items"] = (registry_n or {}).get(ev, N_FULL)
        row["items_sha256"] = (registry_sha or {}).get(ev, IDS_SHA[ev])
        s2.item_sets[mod.ITEM_SET_ID] = row
    return s2


def pin_doc(eval_id: str) -> dict:
    task = TASKS_FULL[eval_id][1]
    return {"schema": 1, "task": task, "seed": None, "n": N_FULL, "of": N_FULL, "ids": list(PIN_IDS[eval_id]), "rule": "the whole population, no seed (synthetic)",
            "dataset": {"synthetic": True, "eval": eval_id}, "generated_by": "tests", "generated_at": NOW}


def _roster_block(tag: str) -> dict:
    return {"family": B.model_key_of(tag), "view": tag, "legs": {t[1]: "runnable" for t in TASKS_FULL.values()}, "entries": {}}


def mirror(tmp_path, seeds: LC.Seeds, *, eval_id: str, root: str = None, paired_none: bool = True, not_run: Sequence[str] = (), qwen_store: bool = False,
           item_sets_layout: str = "list", no_item_sets: bool = False, no_pin_file: bool = False, item_sets_override: Optional[Callable[[List[dict]], None]] = None,
           pin_override: Optional[Callable[[dict], None]] = None, index_sha_stale: bool = False, manifest_item_set_override: Optional[Callable[[dict], None]] = None,
           label_override: Optional[str] = None, short_outcomes: Sequence[str] = (), launch_item_set_override: Optional[Callable[[dict], None]] = None,
           launch_item_set_drop: Sequence[str] = (), launch_override=None, three_override=None, roster_variant: bool = False, rosters_layout: str = "list",
           relaunched: Sequence[str] = (), launch_set_name: str = "launch_sets.json", relaunches_dir: str = B.RECORDS_DIR, row_sha_override: Optional[Dict[str, object]] = None,
           pins_sha_stale: bool = False, no_relaunches: bool = False, relaunch_entry_override: Optional[Callable[[dict], None]] = None,
           instruct: bool = False, instruct_row_override: Optional[Callable[[dict], None]] = None, instruct_none_store: bool = False, instruct_fit_hf: Optional[str] = None,
           instruct_fit_transfer: Optional[bool] = None, instruct_attest_sha: Optional[str] = None, instruct_alias_sha: Optional[str] = None,
           instruct_relaunched: bool = False, instruct_parked_gate: bool = False, instruct_two_stores: bool = False,
           instruct_parked: Sequence[dict] = (), instruct_row_sha: Optional[str] = None, instruct_launched_as_sha: Optional[str] = None,
           instruct_relaunch_entry_sha: Optional[str] = None, instruct_parked_scalar: bool = False, instruct_store_suffix_extra: Optional[str] = None,
           instruct_no_store: bool = False, qwen35b_store: bool = False, undeclared_row_dirs: Sequence[str] = (), dot_dir_store: bool = False,
           qwen35b_u0: bool = False, u0_window_reserve_store: bool = True, u0_window_reserve_declared: bool = True, u0_window_reserve_three_row: bool = True,
           qwen35b_d0p75_relul: bool = False, u0_second_none_store: bool = False, qwen35b_d1p25: bool = False, qwen35b_s2rl: bool = False) -> dict:
    """A synthetic FULL-suite mirror root for one reader (module docstring): the gemma4e4b family's `s1` row (the `none`, a fitted ReLU-steer key, a prompted
    key) and its `s1p` row (the composed difference_relu object and, with `paired_none`, its own `none`), every store carrying both full tasks over
    N_FULL items; the gate records, the launch set (rows with `item_set`), the launched pins / roster, `records/item_sets.json`, the two pin files and the
    three-number records written from the same synthetic truth. The knobs edit ONE record each so a test pins one refusal."""
    bind_seeds(seeds)
    root = root or str(tmp_path / f"{eval_id}_full")
    recs = os.path.join(root, B.RECORDS_DIR)
    import yaml
    launched_roster = {"lane": "bcb-bfcl-full", "legs": [t[0] for t in TASKS_FULL.values()], "sampling": "protocol", "thinking": "off", "pins": "configs/sdfbase_full22_pins.json",
                       "serving_path": "lora_native", "waves": {}}
    rows = list(ROWS) + ([ROW_P] if paired_none else [])
    for okey, paired, staged, sha, _i, _c, _m in rows:
        tag = TAG_P if paired else TAG
        blk = launched_roster["waves"].setdefault(B.launch_tag_of(tag, okey), _roster_block(tag))
        entry = {"cell_id": okey, "status": "runnable", "paper_arm": X.entry_id_of(okey)}
        if staged:
            entry["object"] = {"staged_id": staged, "sha256": sha, "kind": "synthetic"}
        blk["entries"][X.entry_id_of(okey)] = entry
    if qwen_store:
        launched_roster["waves"].setdefault(QWEN_TAG, _roster_block(QWEN_TAG))["entries"]["none"] = {"cell_id": NONE, "status": "runnable", "paper_arm": "none"}
    if qwen35b_u0:
        for okey, tag35, staged, sha, _i, _c, _m, _w, _s, _x in ROWS35_U0:
            blk = launched_roster["waves"].setdefault(B.launch_tag_of(tag35, okey), _roster_block(tag35))
            entry = {"cell_id": okey, "status": "runnable", "paper_arm": X.entry_id_of(okey)}
            if staged:
                entry["object"] = {"staged_id": staged, "sha256": sha, "kind": "synthetic"}
            blk["entries"][X.entry_id_of(okey)] = entry
    if qwen35b_d0p75_relul:
        # the dose-0.75 random-init ReLU row's launched block (DOSE_RELU_ROW_RULE): the three steered keys, each object the pins carry under the model's block
        blk_dr = launched_roster["waves"].setdefault(B.launch_tag_of(TAG35_DR, CTL35_R), _roster_block(TAG35_DR))
        for okey, _ih, _ch in ROWS35_DR:
            blk_dr["entries"][X.entry_id_of(okey)] = {"cell_id": okey, "status": "runnable", "paper_arm": X.entry_id_of(okey),
                                                      "object": {"staged_id": STAGED35_DR[okey], "sha256": OBJ35_DR[okey], "kind": "synthetic"}}
    if qwen35b_d1p25:
        # the dose-1.25 difference-LoRA row's launched block (DOSE_LORA_ROW_RULE): the one steered key, its object the pins carry under the model's block
        blk_dl = launched_roster["waves"].setdefault(B.launch_tag_of(TAG35_DL, DIFF35), _roster_block(TAG35_DL))
        blk_dl["entries"][X.entry_id_of(DIFF35)] = {"cell_id": DIFF35, "status": "runnable", "paper_arm": X.entry_id_of(DIFF35),
                                                    "object": {"staged_id": STAGED35_DL, "sha256": OBJ35_DL, "kind": "synthetic"}}
    if qwen35b_s2rl:
        # the LeCun-init ReLU row's launched block (LECUN_RELU_ROW_RULE): the two launched keys alone (the third key of the harness selection has no entry), each object the
        # pins carry under the model's block
        blk_sr = launched_roster["waves"].setdefault(B.launch_tag_of(TAG35_SR, CTL35_R), _roster_block(TAG35_SR))
        for okey, _ih, _ch in ROWS35_SR:
            blk_sr["entries"][X.entry_id_of(okey)] = {"cell_id": okey, "status": "runnable", "paper_arm": X.entry_id_of(okey),
                                                      "object": {"staged_id": STAGED35_SR[okey], "sha256": OBJ35_SR[okey], "kind": "synthetic"}}
    if qwen35b_store:
        # the 35B row's launched block: its none and the LoRA key, the object the pins carry under the model's block (the appendix route)
        blk35 = launched_roster["waves"].setdefault(TAG35, _roster_block(TAG35))
        blk35["entries"]["none"] = {"cell_id": NONE, "status": "runnable", "paper_arm": "none"}
        blk35["entries"][X.entry_id_of(LORA35)] = {"cell_id": LORA35, "status": "runnable", "paper_arm": X.entry_id_of(LORA35), "object": {"staged_id": STAGED35, "sha256": OBJ35, "kind": "synthetic"}}
    # the instruct-direct row's launched block (its launch tag is the tag itself) and entry -- written whether or not the `_it` store is (`instruct`), so the
    # launched roster's bytes, and every seed-tag store's launched entry's roster_sha256, are the same with and without the `_it` rows
    blk = launched_roster["waves"].setdefault(B.launch_tag_of(IT_TAG, IT_KEY), _roster_block(IT_TAG))
    blk["entries"][X.entry_id_of(IT_KEY)] = {"cell_id": IT_KEY, "status": "runnable", "paper_arm": X.entry_id_of(IT_KEY), "object": {"staged_id": IT_STAGED, "sha256": IT_OBJ, "kind": "synthetic"}}
    blk["entries"]["none"] = {"cell_id": NONE, "status": "runnable", "paper_arm": "none"}
    roster_path = X.write(os.path.join(recs, ROSTER_FILE), yaml.safe_dump(launched_roster, sort_keys=False))
    roster_sha256 = hashlib.sha256(open(roster_path, "rb").read()).hexdigest()
    if roster_variant or relaunched:
        # a per-tree copy under the full roster's stem (the 01:36Z mirror carries `bcb_bfcl_full_roster.8ba0b48.yaml`): discovered as a shared record
        X.write(os.path.join(recs, ROSTER_FILE_T1), yaml.safe_dump(launched_roster, sort_keys=False))
    # the pins copies (the addendum: the second tree's copy differs in KEY ORDER alone -- content identical, sha256 different)
    pins_doc = {MK: {STAGED: {"object_sha256": OBJ, "source": "synthetic"}, STAGED_CTL: {"object_sha256": OBJ_CTL, "source": "synthetic"}, STAGED_P: {"object_sha256": OBJ_P, "source": "synthetic"}},
                "llama8b": {}, "qwen35_4b": {}}
    # the instruct-direct fit under the harness's `it` staged id and its alias spelling (the same object; `instruct_alias_sha` makes the alias name another)
    pins_doc[MK][IT_STAGED] = {"object_sha256": IT_OBJ, "source": "synthetic"}
    pins_doc[MK][IT_ALIAS] = {"object_sha256": (instruct_alias_sha or IT_OBJ), "source": "synthetic"}
    if qwen35b_store:
        pins_doc[MK35] = {STAGED35: {"object_sha256": OBJ35, "source": "synthetic"}}
    if qwen35b_u0:
        pins_doc[MK35] = {STAGED35: {"object_sha256": OBJ35, "source": "synthetic"}, STAGED35_CTL: {"object_sha256": OBJ35_CTL, "source": "synthetic"},
                          STAGED35_DIFF: {"object_sha256": OBJ35_DIFF, "source": "synthetic"}}
    if qwen35b_d0p75_relul:
        assert qwen35b_u0, "the dose-0.75 random-init ReLU rows anchor on the u0 none: the u0 family must be staged"
        pins_doc[MK35].update(PINS35_DR)
    if qwen35b_d1p25:
        assert qwen35b_u0, "the dose-1.25 difference-LoRA row anchors on the u0 none: the u0 family must be staged"
        pins_doc[MK35].update(PINS35_DL)
    if qwen35b_s2rl:
        assert qwen35b_u0, "the LeCun-init ReLU rows anchor on the u0 none: the u0 family must be staged"
        pins_doc[MK35].update(PINS35_SR)
    pins_path = X.write(os.path.join(recs, B.PINS_FILE), pins_doc)
    pins_sha = hashlib.sha256(open(pins_path, "rb").read()).hexdigest()
    X.write(os.path.join(recs, PINS_FILE_T1), pins_doc)
    pins_sha_t2 = None
    roster_sha256_t2 = None
    if relaunched:
        reordered = {"qwen35_4b": {}, "llama8b": {}, MK: {k: {"source": v["source"], "object_sha256": v["object_sha256"]} for k, v in reversed(list(pins_doc[MK].items()))}}
        p2 = X.write(os.path.join(recs, PINS_FILE_T2), reordered)
        pins_sha_t2 = hashlib.sha256(open(p2, "rb").read()).hexdigest()
        assert pins_sha_t2 != pins_sha and json.load(open(p2)) == pins_doc, "byte-different, content-identical"
        roster_t2 = dict(launched_roster, generated_at_tree=TREE2)
        p3 = X.write(os.path.join(recs, ROSTER_FILE_T2), yaml.safe_dump(roster_t2, sort_keys=False))
        roster_sha256_t2 = hashlib.sha256(open(p3, "rb").read()).hexdigest()
    index_entries = [{"roster_sha256": roster_sha256, "file": ROSTER_FILE_T1 if (roster_variant or relaunched) else ROSTER_FILE, "tree_sha": TREE, "pins_file": PINS_FILE_T1,
                      "pins_sha256": (hashlib.sha256(b"a stale pins hash").hexdigest() if pins_sha_stale else pins_sha), "stores": []}]
    if relaunched:
        index_entries.append({"roster_sha256": roster_sha256_t2, "file": ROSTER_FILE_T2, "tree_sha": TREE2, "pins_file": PINS_FILE_T2, "pins_sha256": pins_sha_t2, "stores": []})
    if rosters_layout == "list":
        X.write(os.path.join(recs, B.ROSTERS_INDEX_FILE), {"schema": "bcbw_rosters/1", "written": NOW, "latest_tree": TREE, "rosters": index_entries,
                                                            "pins": [{"pins_sha256": e["pins_sha256"], "tree_sha": e["tree_sha"], "file": e["pins_file"]} for e in index_entries]})
    elif rosters_layout == "mapping":
        X.write(os.path.join(recs, B.ROSTERS_INDEX_FILE), {"schema": "bcbw_rosters/1", "written": NOW,
                                                            "rosters": {e["roster_sha256"]: {"file": e["file"], "tree_sha": e["tree_sha"], "stores": [], "pins": e["pins_file"]} for e in index_entries}})
    elif rosters_layout != "none":
        raise ValueError(rosters_layout)
    # the pin files and the index (ITEM_SET_RULE): the index carries the pin FILE's sha256, which every launch-set row repeats
    pin_shas: Dict[str, str] = {}
    pin_files: Dict[str, str] = {}
    for ev, mod in MODULES.items():
        doc = pin_doc(ev)
        p = X.write(os.path.join(recs, mod.PIN_FILE), doc)
        sha_clean = hashlib.sha256(open(p, "rb").read()).hexdigest()
        if pin_override is not None and ev == eval_id:
            pin_override(doc)
            X.write(p, doc)
        pin_shas[ev] = sha_clean if (index_sha_stale and ev == eval_id) else hashlib.sha256(open(p, "rb").read()).hexdigest()
        pin_files[ev] = mod.PIN_FILE
        if no_pin_file and ev == eval_id:
            os.remove(p)
    entries = [{"id": MODULES[ev].ITEM_SET_ID, "sha256": pin_shas[ev], "n": N_FULL, "file": pin_files[ev], "tree_file": f"configs/capeval/tasks/{pin_files[ev]}"} for ev in MODULES]
    if item_sets_override is not None:
        item_sets_override(entries)
    if not no_item_sets:
        if item_sets_layout == "list":
            X.write(os.path.join(recs, B.ITEM_SETS_FILE), entries)
        elif item_sets_layout == "dict":
            X.write(os.path.join(recs, B.ITEM_SETS_FILE), {"schema": "bcbw_item_sets/1", "written": NOW, "item_sets": {e["id"]: {k: v for k, v in e.items() if k != "id"} for e in entries}})
        else:
            raise ValueError(item_sets_layout)
    three = {ev: {"schema": B.THREE_NUMBERS_SCHEMA, "eval": ev, "source": MODULES[ev].SOURCE, "written": NOW, "contract": "synthetic", "seed_reading": B.SEED_AXIS_RULE, "rows": {}} for ev in TASKS_FULL}
    launch: Dict[str, List[dict]] = {ev: [] for ev in TASKS_FULL}
    keys: Dict[str, str] = {}
    specs = {ev: {k: X.spec(eval_id=ev, shape=SHAPE_FULL, **v) for k, v in SPECS[ev].items()} for ev in TASKS_FULL}
    specs_p = {ev: X.spec(eval_id=ev, shape=SHAPE_FULL, **SPECS_P[ev]) for ev in TASKS_FULL}
    paired_none_key = None
    gemma_kw = dict(model_hf=MODEL_HF, revision=Y.revision_of(MODEL_ID), model_key=MK, fit_model_hf=FIT_HF)
    parked_sfx: List[str] = []
    parked_trees: List[str] = []
    for i, (okey, paired, staged, sha, ihash8, chash8, mlr) in enumerate(rows):
        tag = TAG_P if paired else TAG
        sel = SEL_P if paired else SEL
        steered = staged is not None
        is_paired_none = paired and okey == NONE
        row_specs = {ev: (specs_p[ev] if is_paired_none else specs[ev][okey]) for ev in TASKS_FULL}
        launch_tag = B.launch_tag_of(tag, okey)
        slug = okey.replace("__", "--", 1)
        is_relaunched = (not paired) and okey in relaunched
        tree = TREE2 if is_relaunched else TREE
        suffix_extra = f"-{RELAUNCH_TAG}" if is_relaunched else ""
        launched_as = ({"wave": launch_tag, "row_dir": B.row_dir_of(tag), "suffix": X.suffix_of(tag, okey, launch_tag, TREE2, suffix_extra), "sha": TREE2} if is_relaunched else None)
        for ev in TASKS_FULL:
            lr = X.launch_row(sel, okey, tag, pin_key=staged, expected=(sha if staged else None), paired=paired, index=i, seed_base=SEED_N, launch_tag=launch_tag, launched_as=launched_as)
            if is_relaunched:
                lr["sha"] = TREE2
                lr["relaunched_from"] = {"suffix": X.suffix_of(tag, okey, launch_tag, TREE), "why": "synthetic: the first launch parked INCOMPLETE (a transport fault), relaunched at the fixed tree"}
            if row_sha_override is not None and f"{sel}/{okey}" in row_sha_override:
                v = row_sha_override[f"{sel}/{okey}"]
                if v is None:
                    lr.pop("sha", None)
                else:
                    lr["sha"] = v
            if not (ev == eval_id and f"{sel}/{okey}" in launch_item_set_drop):
                lr["item_set"] = {"id": MODULES[ev].ITEM_SET_ID, "sha256": pin_shas[ev], "n": N_FULL}
                if launch_item_set_override is not None and ev == eval_id:
                    launch_item_set_override(lr["item_set"])
            launch[ev].append(lr)
        store_key = f"{slug}__controls__t0__a0__{ihash8}__{chash8}"
        if okey in not_run and not paired:
            gate = X.gate_record(tag, okey, tree=TREE, store_key=store_key, obj_sha=sha, steered=steered, max_lora_rank=mlr, verdict="FAIL", completed=False, model_key=MK, shape=SHAPE_FULL)
            X.write(os.path.join(recs, f"gate_{launch_tag}_{okey}_{X.suffix_of(tag, okey, launch_tag, TREE)}.json"), gate)
            continue
        if is_relaunched:
            # the PARKED first store at TREE: RESULT INCOMPLETE, never the row's store (RELAUNCH_RULE); its gate record reads FAIL
            X.store_files(root, tag, okey, row_specs, obj_sha=sha, staged=staged, tree=TREE, ihash8=ihash8, chash8=chash8, seed_n=SEED_N, roster_sha256=roster_sha256,
                          max_lora_rank=mlr, shape=SHAPE_FULL, result_complete=False, **gemma_kw)
            g0 = X.gate_record(tag, okey, tree=TREE, store_key=store_key, obj_sha=sha, steered=steered, max_lora_rank=mlr, model_key=MK, shape=SHAPE_FULL, verdict="FAIL", completed=False)
            X.write(os.path.join(recs, f"gate_{launch_tag}_{okey}_{X.suffix_of(tag, okey, launch_tag, TREE)}.json"), g0)
        key = X.store_files(root, tag, okey, row_specs, obj_sha=sha, staged=staged, tree=tree, ihash8=ihash8, chash8=chash8, seed_n=SEED_N,
                            roster_sha256=(roster_sha256_t2 if is_relaunched else roster_sha256), max_lora_rank=mlr, shape=SHAPE_FULL, suffix_extra=suffix_extra, **gemma_kw)
        assert key == store_key
        if is_paired_none:
            paired_none_key = key
        else:
            keys[okey] = key
        gate = X.gate_record(tag, okey, tree=tree, store_key=store_key, obj_sha=sha, steered=steered, max_lora_rank=mlr, model_key=MK, shape=SHAPE_FULL, suffix_extra=suffix_extra,
                             compared_against=({"record": f"gate_{launch_tag}_{okey}_{X.suffix_of(tag, okey, launch_tag, TREE)}.json", "sha": TREE, "ruling": "D298"} if is_relaunched else None),
                             cross_tree=(True if is_relaunched else None))
        X.write(os.path.join(recs, f"gate_{launch_tag}_{okey}_{X.suffix_of(tag, okey, launch_tag, tree, suffix_extra)}.json"), gate)
        for ev in TASKS_FULL:
            tr = X.three_row(tag, sel, okey, row_specs[ev], eval_id=ev, obj_sha=sha, tree=tree, store_key=store_key, steered=steered, seed_n=SEED_N,
                             anchor_spec=(specs[ev][NONE] if okey != NONE else None), cell_slug=slug, shape=SHAPE_FULL, suffix_extra=suffix_extra, launched_as=launched_as)
            three[ev]["rows"][tr["key"]] = tr
    if instruct:
        # the instruct-direct row (INSTRUCT_DIRECT_RULE): the store under `<model key>-f22it`, fit on the SERVED checkpoint (fit_provenance.model_id = the served id,
        # transfer false), its gate record, its launch-set rows and three-number rows (selection `it`, seed null); the `it/none` row is class anchor with NO store
        it_slug = IT_KEY.replace("__", "--", 1)
        it_specs = {ev: X.spec(eval_id=ev, shape=SHAPE_FULL, **SPECS_IT[ev]) for ev in TASKS_FULL}
        it_kw = dict(gemma_kw, fit_model_hf=(instruct_fit_hf or MODEL_HF), fit_transfer=(False if instruct_fit_transfer is None else instruct_fit_transfer))
        it_gate_kw = dict(obj_sha=(instruct_attest_sha or IT_OBJ), steered=True, max_lora_rank=1, model_key=MK)
        it_launch_tag = B.launch_tag_of(IT_TAG, IT_KEY)
        # per eval: (the store's tree, its suffix extra, its shape, the specs it holds) -- ONE two-task store (the default), the same store relaunched at TREE2 as `-r2`
        # (`instruct_relaunched`: the first attempt at TREE refused at standup, NO store dir, its FAIL gate record beside the live one with `instruct_parked_gate`), or TWO
        # one-task stores (`instruct_two_stores`: BFCL at TREE under the bare suffix, BigCodeBench at TREE2 under the `-bcb` task tag; each row's launched_as names its own)
        if instruct_two_stores:
            plan = {ev: ((TREE2 if ev == "bigcodebench" else TREE), TWO_STORES_TASK_TAG[ev], SHAPE_ONE_TASK[ev], {ev: it_specs[ev]}) for ev in TASKS_FULL}
        elif instruct_relaunched:
            # (`instruct_store_suffix_extra` puts the rerun store under another suffix -- outside the grammar, or task-tagged -- the row's launched_as naming it)
            plan = {ev: (TREE2, (instruct_store_suffix_extra if instruct_store_suffix_extra is not None else f"-{RELAUNCH_TAG}"), SHAPE_FULL, it_specs) for ev in TASKS_FULL}
        else:
            plan = {ev: (TREE, "", SHAPE_FULL, it_specs) for ev in TASKS_FULL}
        written = set()
        it_key = None
        for ev, (it_tree, it_extra, it_shape, held) in plan.items():
            sfx = X.suffix_of(IT_TAG, IT_KEY, tree=it_tree, suffix_extra=it_extra)
            if sfx in written:
                continue
            written.add(sfx)
            it_key = f"{it_slug}__controls__t0__a0__{IT_IHASH8}__{IT_CHASH8}"
            if not instruct_no_store:
                it_key = X.store_files(root, IT_TAG, IT_KEY, held, obj_sha=IT_OBJ, staged=IT_STAGED, tree=it_tree, ihash8=IT_IHASH8, chash8=IT_CHASH8, seed_n=SEED_N,
                                       roster_sha256=roster_sha256, max_lora_rank=1, shape=it_shape, attest_sha=instruct_attest_sha, suffix_extra=it_extra, **it_kw)
            gate = X.gate_record(IT_TAG, IT_KEY, tree=it_tree, store_key=it_key, shape=it_shape, suffix_extra=it_extra,
                                 compared_against=({"record": f"gate_{it_launch_tag}_{IT_KEY}_{X.suffix_of(IT_TAG, IT_KEY, tree=TREE)}.json", "sha": TREE, "ruling": "D298"} if instruct_relaunched else None),
                                 cross_tree=(True if instruct_relaunched else None), **it_gate_kw)
            X.write(os.path.join(recs, f"gate_{it_launch_tag}_{IT_KEY}_{sfx}.json"), gate)
        keys[f"{IT_SEL}/{IT_KEY}"] = it_key
        # `instruct_parked` (with `instruct_relaunched`; the addendum of 2026-09-25 21:17Z): the attempts the `-r2` rerun superseded -- each {tree, extra, verdict, shape
        # (an eval id for a one-task attempt, None for two-task), store (write its store dir too; default: NOT mirrored)} -- their gate records beside the live one,
        # relaunches.json's `parked_suffix` list and the launch row's `relaunched_from.suffixes` (the first on `.suffix`)
        for pk in (instruct_parked if instruct_relaunched else ()):
            p_tree, p_extra, p_verdict, p_shape_key = pk["tree"], pk.get("extra", ""), pk.get("verdict", "FAIL"), pk.get("shape")
            p_shape = SHAPE_ONE_TASK[p_shape_key] if p_shape_key else SHAPE_FULL
            p_sfx = X.suffix_of(IT_TAG, IT_KEY, tree=p_tree, suffix_extra=p_extra)
            parked_sfx.append(p_sfx)
            if p_tree not in (TREE, TREE2) and p_tree not in parked_trees:
                parked_trees.append(p_tree)
            if pk.get("store"):
                X.store_files(root, IT_TAG, IT_KEY, {ev: it_specs[ev] for ev in p_shape.tasks}, obj_sha=IT_OBJ, staged=IT_STAGED, tree=p_tree, ihash8=IT_IHASH8, chash8=IT_CHASH8,
                              seed_n=SEED_N, roster_sha256=roster_sha256, max_lora_rank=1, shape=p_shape, suffix_extra=p_extra, **it_kw)
            gp = X.gate_record(IT_TAG, IT_KEY, tree=p_tree, store_key=it_key, shape=p_shape, suffix_extra=p_extra, verdict=p_verdict, completed=(p_verdict == "PASS"), **it_gate_kw)
            X.write(os.path.join(recs, f"gate_{it_launch_tag}_{IT_KEY}_{p_sfx}.json"), gp)
        if instruct_relaunched and instruct_parked_gate:
            # the parked first attempt's FAIL record (no store dir): it must never shadow the live store's PASS record
            g0 = X.gate_record(IT_TAG, IT_KEY, tree=TREE, store_key=it_key, shape=SHAPE_FULL, verdict="FAIL", completed=False, **it_gate_kw)
            X.write(os.path.join(recs, f"gate_{it_launch_tag}_{IT_KEY}_{X.suffix_of(IT_TAG, IT_KEY, tree=TREE)}.json"), g0)
        for ev in TASKS_FULL:
            it_tree, it_extra, _shape, _held = plan[ev]
            sfx = X.suffix_of(IT_TAG, IT_KEY, tree=it_tree, suffix_extra=it_extra)
            la = None
            if instruct_relaunched or (instruct_two_stores and it_extra):
                la = {"wave": it_launch_tag, "row_dir": IT_ROW_DIR, "suffix": sfx, "sha": (instruct_launched_as_sha or it_tree), **({"tag": it_extra[1:]} if instruct_two_stores else {})}
            for j, (okey, pin_key, expected) in enumerate(((IT_KEY, IT_STAGED, IT_OBJ), (NONE, None, None))):
                lr = X.launch_row(IT_SEL, okey, IT_TAG, pin_key=pin_key, expected=expected, paired=False, index=len(rows) + 1 + j, seed_base=None)
                lr.update(block=B.INSTRUCT_BLOCK, transfer_kind=B.INSTRUCT_TRANSFER_KIND, model_key=MK, anchor_tuple=(f"{SEL}/{NONE}" if okey != NONE else None))
                lr["item_set"] = {"id": MODULES[ev].ITEM_SET_ID, "sha256": pin_shas[ev], "n": N_FULL}
                if okey != NONE:
                    lr["pin_key_alias"] = IT_ALIAS
                    lr["sha"] = instruct_row_sha or it_tree      # the tree the ROW was declared at (on a rerun: the FIRST attempt's tree, not the store's)
                    if la is not None:
                        lr["launched_as"] = dict(la)
                    if instruct_relaunched:
                        lr["relaunched_from"] = {"suffix": (parked_sfx[0] if parked_sfx else X.suffix_of(IT_TAG, IT_KEY, tree=TREE)), "why": "synthetic: refused at standup (no store), relaunched at the fixed tree"}
                        if parked_sfx:
                            lr["relaunched_from"]["suffixes"] = list(parked_sfx)
                    if instruct_row_override is not None:
                        instruct_row_override(lr)
                launch[ev].append(lr)
            tr = X.three_row(IT_TAG, IT_SEL, IT_KEY, it_specs[ev], eval_id=ev, obj_sha=(instruct_attest_sha or IT_OBJ), tree=it_tree, store_key=it_key, steered=True, seed_n=None,
                             anchor_spec=specs[ev][NONE], cell_slug=it_slug, shape=SHAPE_FULL, suffix_extra=it_extra, launched_as=la)
            three[ev]["rows"][tr["key"]] = tr
        if instruct_none_store:
            # a `_it` none STORE (the declaration says none exists): must refuse by name, never file a second none run
            k2 = X.store_files(root, IT_TAG, NONE, {ev: X.spec(eval_id=ev, shape=SHAPE_FULL, **SPECS[ev][NONE]) for ev in TASKS_FULL}, obj_sha=None, staged=None, tree=TREE,
                               ihash8="3a1b2c3d", chash8="e4f5a6b7", seed_n=SEED_N, roster_sha256=roster_sha256, max_lora_rank=1, shape=SHAPE_FULL, **gemma_kw)
            g2 = X.gate_record(IT_TAG, NONE, tree=TREE, store_key=k2, obj_sha=None, steered=False, max_lora_rank=1, model_key=MK, shape=SHAPE_FULL)
            X.write(os.path.join(recs, f"gate_{B.launch_tag_of(IT_TAG, NONE)}_{NONE}_{X.suffix_of(IT_TAG, NONE, tree=TREE)}.json"), g2)
    if qwen_store:
        key = X.store_files(root, QWEN_TAG, NONE, {ev: X.spec(eval_id=ev, shape=SHAPE_FULL, **SPECS[ev][NONE]) for ev in TASKS_FULL}, obj_sha=None, staged=None, tree=TREE,
                            ihash8="2ff9cdff", chash8="ed8a6ada", seed_n=2, roster_sha256=roster_sha256, max_lora_rank=1, shape=SHAPE_FULL)
        keys[f"{QWEN_TAG}/{NONE}"] = key
        gate = X.gate_record(QWEN_TAG, NONE, tree=TREE, store_key=key, obj_sha=None, steered=False, max_lora_rank=1, shape=SHAPE_FULL)
        X.write(os.path.join(recs, f"gate_{QWEN_TAG}_{NONE}_{X.suffix_of(QWEN_TAG, NONE)}.json"), gate)
        for ev in TASKS_FULL:
            lr = X.launch_row("s2", NONE, QWEN_TAG, pin_key=None, expected=None, paired=False, index=len(rows), seed_base=2)
            lr["item_set"] = {"id": MODULES[ev].ITEM_SET_ID, "sha256": pin_shas[ev], "n": N_FULL}
            launch[ev].append(lr)
            tr = X.three_row(QWEN_TAG, "s2", NONE, X.spec(eval_id=ev, shape=SHAPE_FULL, **SPECS[ev][NONE]), eval_id=ev, obj_sha=None, tree=TREE, store_key=key, steered=False, seed_n=2, shape=SHAPE_FULL)
            three[ev]["rows"][tr["key"]] = tr
    if qwen35b_store:
        kw35 = dict(model_hf=MODEL_HF35, revision=str(seeds.models[MODEL_ID35]["hf_revision"]), model_key=MK35, fit_model_hf=FIT_HF35)
        for i35, (okey, staged, sha, ihash8, chash8, mlr) in enumerate(ROWS35):
            steered = staged is not None
            slug = okey.replace("__", "--", 1)
            row_specs = {ev: X.spec(eval_id=ev, shape=SHAPE_FULL, **SPECS35[ev][okey]) for ev in TASKS_FULL}
            key = X.store_files(root, TAG35, okey, row_specs, obj_sha=sha, staged=staged, tree=TREE, ihash8=ihash8, chash8=chash8, seed_n=1, roster_sha256=roster_sha256,
                                max_lora_rank=mlr, shape=SHAPE_FULL, **kw35)
            keys[f"{TAG35}/{okey}"] = key
            gate = X.gate_record(TAG35, okey, tree=TREE, store_key=key, obj_sha=sha, steered=steered, max_lora_rank=mlr, model_key=MK35, shape=SHAPE_FULL)
            X.write(os.path.join(recs, f"gate_{TAG35}_{okey}_{X.suffix_of(TAG35, okey)}.json"), gate)
            for ev in TASKS_FULL:
                lr = X.launch_row(SEL35, okey, TAG35, pin_key=staged, expected=sha, paired=False, index=len(rows) + 1 + i35, seed_base=1)
                lr["item_set"] = {"id": MODULES[ev].ITEM_SET_ID, "sha256": pin_shas[ev], "n": N_FULL}
                launch[ev].append(lr)
                tr = X.three_row(TAG35, SEL35, okey, row_specs[ev], eval_id=ev, obj_sha=sha, tree=TREE, store_key=key, steered=steered, seed_n=1,
                                 anchor_spec=(X.spec(eval_id=ev, shape=SHAPE_FULL, **SPECS35[ev][NONE]) if okey != NONE else None), cell_slug=slug, shape=SHAPE_FULL)
                three[ev]["rows"][tr["key"]] = tr
    families_block: Dict[str, dict] = {}
    if qwen35b_u0:
        # THE u0 FAMILY (UNSTEERED_ROW_RULE / WINDOW_ANCHOR_RULE): seven stores, six launch rows (the window re-serve is declared under families.<key>.unsteered)
        kw35 = dict(model_hf=MODEL_HF35, revision=str(seeds.models[MODEL_ID35]["hf_revision"]), model_key=MK35, fit_model_hf=FIT_HF35)
        reserve_suffix = X.suffix_of(TAG35_U0, NONE, tree=TREE, suffix_extra=WINDOW35_SEG)
        for i35, (okey, tag35, staged, sha, ihash8, chash8, mlr, w, ws, sfx_extra) in enumerate(ROWS35_U0):
            is_reserve = bool(sfx_extra)
            if is_reserve and not u0_window_reserve_store:
                continue
            steered = staged is not None
            sel35 = SEL35_U0 if tag35 == TAG35_U0 else SEL35
            slug = okey.replace("__", "--", 1)
            spec_key = f"{NONE}@w" if is_reserve else okey
            row_specs = {ev: X.spec(eval_id=ev, shape=SHAPE_FULL, **SPECS35_U0[ev][spec_key]) for ev in TASKS_FULL}
            key = X.store_files(root, tag35, okey, row_specs, obj_sha=sha, staged=staged, tree=TREE, ihash8=ihash8, chash8=chash8, seed_n=1, roster_sha256=roster_sha256,
                                max_lora_rank=mlr, shape=SHAPE_FULL, suffix_extra=sfx_extra, max_model_len=w, max_num_seqs=ws, **kw35)
            rk = B.row_okey(okey, None, w if is_reserve else None)
            keys[f"{tag35}/{rk}"] = key
            gate = X.gate_record(tag35, okey, tree=TREE, store_key=key, obj_sha=sha, steered=steered, max_lora_rank=mlr, model_key=MK35, shape=SHAPE_FULL, suffix_extra=sfx_extra)
            X.write(os.path.join(recs, f"gate_{B.launch_tag_of(tag35, okey)}_{okey}_{X.suffix_of(tag35, okey, tree=TREE, suffix_extra=sfx_extra)}.json"), gate)
            for ev in TASKS_FULL:
                if not is_reserve:
                    lr = X.launch_row(sel35, okey, tag35, pin_key=staged, expected=sha, paired=False, index=len(rows) + 1 + i35, seed_base=1)
                    lr["item_set"] = {"id": MODULES[ev].ITEM_SET_ID, "sha256": pin_shas[ev], "n": N_FULL}
                    lr["block"] = "onboarded"
                    lr["anchor_tuple"] = None if okey == NONE else f"{SEL35_U0}/{NONE}"      # every steered / prompted row anchors on the u0 none (window-matched)
                    launch[ev].append(lr)
                if is_reserve and not u0_window_reserve_three_row:
                    continue
                # the three-number rows: the u0 rows spell no fit seed (the producer's spelling), the s1 rows the seed; the re-serve is keyed none@w<N>
                tr = X.three_row(tag35, sel35, okey, row_specs[ev], eval_id=ev, obj_sha=sha, tree=TREE, store_key=key, steered=steered, seed_n=(None if tag35 == TAG35_U0 else 1),
                                 cell_slug=slug, shape=SHAPE_FULL, suffix_extra=sfx_extra)
                if is_reserve:
                    tr["key"] = f"{tag35}/{rk}"
                three[ev]["rows"][tr["key"]] = tr
        if u0_window_reserve_declared:
            families_block[MK35] = {"registry_model_id": MODEL_ID35, "views": [SEL35_U0, SEL35], "tags": [TAG35_U0, TAG35], "transfer_kind": "base_to_instruct",
                                    "unsteered": {"tag": TAG35_U0, "view": SEL35_U0, "row_dir": ROW_DIR35_U0,
                                                  "windows": {"*": {"max_model_len": WINDOW35, "max_num_seqs": WINDOW35_SEQS}, CONST35: {"max_model_len": WINDOW35_CONST, "max_num_seqs": WINDOW35_CONST_SEQS},
                                                              f"{NONE}@w{WINDOW35_CONST}": {"max_model_len": WINDOW35_CONST, "max_num_seqs": WINDOW35_CONST_SEQS}},
                                                  "window_reserves": {ev: {"store": B.rel_store(ROW_DIR35_U0, NONE, reserve_suffix), "suffix": reserve_suffix, "tree": TREE, "tag": f"w{WINDOW35_CONST}",
                                                                           "window": {"max_model_len": WINDOW35_CONST, "max_num_seqs": WINDOW35_CONST_SEQS}, "tool_calls": ev == "bfcl",
                                                                           "run_id": "synthetic-reserve", "launched_at": NOW, "role": "synthetic: the u0 none re-served at the constitution window"}
                                                                      for ev in TASKS_FULL}}}
    if qwen35b_d0p75_relul:
        # THE DOSE-0.75 RANDOM-INIT ReLU ROW (DOSE_RELU_ROW_RULE): three steered stores at the default window, three launch rows at the d0p75_relul selection anchoring on
        # the u0 none, their three-number rows and gate records; the objects are pinned under the model's block (the appendix route) -- never roster entries
        kw35 = dict(model_hf=MODEL_HF35, revision=str(seeds.models[MODEL_ID35]["hf_revision"]), model_key=MK35, fit_model_hf=FIT_HF35)
        for idr, (okey, ihash8, chash8) in enumerate(ROWS35_DR):
            staged, sha = STAGED35_DR[okey], OBJ35_DR[okey]
            row_specs = {ev: X.spec(eval_id=ev, shape=SHAPE_FULL, **SPECS35_DR[ev][okey]) for ev in TASKS_FULL}
            key = X.store_files(root, TAG35_DR, okey, row_specs, obj_sha=sha, staged=staged, tree=TREE, ihash8=ihash8, chash8=chash8, seed_n=1, roster_sha256=roster_sha256,
                                max_lora_rank=1, shape=SHAPE_FULL, max_model_len=WINDOW35, max_num_seqs=WINDOW35_SEQS, **kw35)
            keys[f"{TAG35_DR}/{okey}"] = key
            gate = X.gate_record(TAG35_DR, okey, tree=TREE, store_key=key, obj_sha=sha, steered=True, max_lora_rank=1, model_key=MK35, shape=SHAPE_FULL)
            X.write(os.path.join(recs, f"gate_{B.launch_tag_of(TAG35_DR, okey)}_{okey}_{X.suffix_of(TAG35_DR, okey, tree=TREE)}.json"), gate)
            for ev in TASKS_FULL:
                lr = X.launch_row(SEL35_DR, okey, TAG35_DR, pin_key=staged, expected=sha, paired=False, index=len(rows) + 30 + idr, seed_base=1)
                lr["item_set"] = {"id": MODULES[ev].ITEM_SET_ID, "sha256": pin_shas[ev], "n": N_FULL}
                lr["block"] = "onboarded"
                lr["anchor_tuple"] = f"{SEL35_U0}/{NONE}"       # window-matched on the u0 none, the s1 rows' rule
                launch[ev].append(lr)
                tr = X.three_row(TAG35_DR, SEL35_DR, okey, row_specs[ev], eval_id=ev, obj_sha=sha, tree=TREE, store_key=key, steered=True, seed_n=1,
                                 cell_slug=okey.replace("__", "--", 1), shape=SHAPE_FULL)
                three[ev]["rows"][tr["key"]] = tr
        if u0_window_reserve_declared:
            families_block[MK35]["views"].append(SEL35_DR)
            families_block[MK35]["tags"].append(TAG35_DR)
    if qwen35b_d1p25:
        # THE DOSE-1.25 DIFFERENCE-LoRA ROW (DOSE_LORA_ROW_RULE): one steered store at the default window, one launch row at the d1p25 selection anchoring on the u0 none, its
        # three-number row and gate record; the object is pinned under the model's block (the appendix route) -- never a roster entry
        kw35 = dict(model_hf=MODEL_HF35, revision=str(seeds.models[MODEL_ID35]["hf_revision"]), model_key=MK35, fit_model_hf=FIT_HF35)
        row_specs = {ev: X.spec(eval_id=ev, shape=SHAPE_FULL, **SPECS35_DL[ev]) for ev in TASKS_FULL}
        key = X.store_files(root, TAG35_DL, DIFF35, row_specs, obj_sha=OBJ35_DL, staged=STAGED35_DL, tree=TREE, ihash8="d1a25a01", chash8="c0ffee10", seed_n=1, roster_sha256=roster_sha256,
                            max_lora_rank=4, shape=SHAPE_FULL, max_model_len=WINDOW35, max_num_seqs=WINDOW35_SEQS, **kw35)
        keys[f"{TAG35_DL}/{DIFF35}"] = key
        gate = X.gate_record(TAG35_DL, DIFF35, tree=TREE, store_key=key, obj_sha=OBJ35_DL, steered=True, max_lora_rank=4, model_key=MK35, shape=SHAPE_FULL)
        X.write(os.path.join(recs, f"gate_{B.launch_tag_of(TAG35_DL, DIFF35)}_{DIFF35}_{X.suffix_of(TAG35_DL, DIFF35, tree=TREE)}.json"), gate)
        for ev in TASKS_FULL:
            lr = X.launch_row(SEL35_DL, DIFF35, TAG35_DL, pin_key=STAGED35_DL, expected=OBJ35_DL, paired=False, index=len(rows) + 40, seed_base=1)
            lr["item_set"] = {"id": MODULES[ev].ITEM_SET_ID, "sha256": pin_shas[ev], "n": N_FULL}
            lr["block"] = "onboarded"
            lr["anchor_tuple"] = f"{SEL35_U0}/{NONE}"       # window-matched on the u0 none, the s1 rows' rule
            launch[ev].append(lr)
            tr = X.three_row(TAG35_DL, SEL35_DL, DIFF35, row_specs[ev], eval_id=ev, obj_sha=OBJ35_DL, tree=TREE, store_key=key, steered=True, seed_n=1,
                             cell_slug=DIFF35.replace("__", "--", 1), shape=SHAPE_FULL)
            three[ev]["rows"][tr["key"]] = tr
        if u0_window_reserve_declared:
            families_block[MK35]["views"].append(SEL35_DL)
            families_block[MK35]["tags"].append(TAG35_DL)
    if qwen35b_s2rl:
        # THE LeCun-INIT ReLU ROW (LECUN_RELU_ROW_RULE): two steered stores at the default window, two launch rows at the s2rl selection anchoring on the u0 none with the
        # group's seed_base, their three-number rows (the seed spelled as the group's) and gate records; the objects are pinned under the model's block (the appendix route)
        kw35 = dict(model_hf=MODEL_HF35, revision=str(seeds.models[MODEL_ID35]["hf_revision"]), model_key=MK35, fit_model_hf=FIT_HF35)
        for isr, (okey, ihash8, chash8) in enumerate(ROWS35_SR):
            staged, sha = STAGED35_SR[okey], OBJ35_SR[okey]
            row_specs = {ev: X.spec(eval_id=ev, shape=SHAPE_FULL, **SPECS35_SR[ev][okey]) for ev in TASKS_FULL}
            key = X.store_files(root, TAG35_SR, okey, row_specs, obj_sha=sha, staged=staged, tree=TREE, ihash8=ihash8, chash8=chash8, seed_n=1, roster_sha256=roster_sha256,
                                max_lora_rank=1, shape=SHAPE_FULL, max_model_len=WINDOW35, max_num_seqs=WINDOW35_SEQS, **kw35)
            keys[f"{TAG35_SR}/{okey}"] = key
            gate = X.gate_record(TAG35_SR, okey, tree=TREE, store_key=key, obj_sha=sha, steered=True, max_lora_rank=1, model_key=MK35, shape=SHAPE_FULL)
            X.write(os.path.join(recs, f"gate_{B.launch_tag_of(TAG35_SR, okey)}_{okey}_{X.suffix_of(TAG35_SR, okey, tree=TREE)}.json"), gate)
            for ev in TASKS_FULL:
                lr = X.launch_row(SEL35_SR, okey, TAG35_SR, pin_key=staged, expected=sha, paired=False, index=len(rows) + 50 + isr, seed_base=1)
                lr["item_set"] = {"id": MODULES[ev].ITEM_SET_ID, "sha256": pin_shas[ev], "n": N_FULL}
                lr["block"] = "onboarded"
                lr["anchor_tuple"] = f"{SEL35_U0}/{NONE}"       # window-matched on the u0 none, the s1 rows' rule
                launch[ev].append(lr)
                tr = X.three_row(TAG35_SR, SEL35_SR, okey, row_specs[ev], eval_id=ev, obj_sha=sha, tree=TREE, store_key=key, steered=True, seed_n=1,
                                 cell_slug=okey.replace("__", "--", 1), shape=SHAPE_FULL)
                three[ev]["rows"][tr["key"]] = tr
        if u0_window_reserve_declared:
            families_block[MK35]["views"].append(SEL35_SR)
            families_block[MK35]["tags"].append(TAG35_SR)
    if u0_second_none_store:
        # the u0 `none` RE-SERVED at the dose launch's tree, under the u0 row dir, with no launch row, gate record or three-number row of its own: it shares the (u0, none)
        # row key with the declared store (whose tree the launch row names) and must bind no new row
        kw35 = dict(model_hf=MODEL_HF35, revision=str(seeds.models[MODEL_ID35]["hf_revision"]), model_key=MK35, fit_model_hf=FIT_HF35)
        none35_specs = {ev: X.spec(eval_id=ev, shape=SHAPE_FULL, **SPECS35_U0[ev][NONE]) for ev in TASKS_FULL}
        X.store_files(root, TAG35_U0, NONE, none35_specs, obj_sha=None, staged=None, tree=TREE_DR, ihash8="2ff9cdff", chash8="0bada55e", seed_n=1, roster_sha256=roster_sha256,
                      max_lora_rank=1, shape=SHAPE_FULL, max_model_len=WINDOW35, max_num_seqs=WINDOW35_SEQS, **kw35)
    none_specs = {ev: X.spec(eval_id=ev, shape=SHAPE_FULL, **SPECS[ev][NONE]) for ev in TASKS_FULL}
    for name in undeclared_row_dirs:
        # a complete store under a row dir NO table row declares (UNDECLARED_ROW_DIR_RULE): no gate record, no launch row, no three-number row names it. The store's
        # launch tag is the row dir's own where the grammar reads one (`gemma4e4b-f22s2rl` -> `bcbw-gemma4e4b_s2rl-<sha7>`, the s2rl row dir of a family that runs no
        # LeCun-init ReLU row: undeclared FOR THE READER, lecun_relu_row_admitted), else the key's s1 tag (a name the grammar cannot read at all)
        X.store_files(root, TAG, NONE, none_specs, obj_sha=None, staged=None, tree=TREE, ihash8="2ff9cdff", chash8="0badf00d", seed_n=1, roster_sha256=roster_sha256,
                      max_lora_rank=1, shape=SHAPE_FULL, row_dir=name, launch_tag=B.tag_of_row_dir(name) or f"{name.split('-f22', 1)[0]}_s1", **gemma_kw)
    if dot_dir_store:
        # a producer's publish scaffolding: the same store staged under `native/.staging/<row dir>/` (audit_store_35b.py's shape), never a row
        X.store_files(root, TAG, NONE, none_specs, obj_sha=None, staged=None, tree=TREE, ihash8="2ff9cdff", chash8="0badf00d", seed_n=1, roster_sha256=roster_sha256,
                      max_lora_rank=1, shape=SHAPE_FULL, row_dir=f"{DOT_DIR}/{ROW_DIR}", **gemma_kw)
    # the store-level edits (one store: the seed tag's `none`)
    if eval_id in TASKS_FULL and (manifest_item_set_override is not None or label_override is not None or short_outcomes):
        task, _rt, ofile, _cap, _t, _c, item_key, _o = TASKS_FULL[eval_id]
        kd = os.path.join(root, B.STORES_HEAD, ROW_DIR, NONE, B.THINKING_DIR, X.suffix_of(TAG, NONE, tree=TREE), "cells", keys[NONE])
        if manifest_item_set_override is not None:
            p = os.path.join(kd, "manifest.json")
            man = json.load(open(p, encoding="utf-8"))
            manifest_item_set_override(man["tinycap"][item_key])
            X.write(p, man)
        if label_override is not None:
            for p in (os.path.join(kd, "gen_stats.json"), os.path.join(kd, "summary", ofile), os.path.join(kd, "outcomes", ofile)):
                d = json.load(open(p, encoding="utf-8"))
                if p.endswith("gen_stats.json"):
                    d[task]["item_set"] = label_override
                else:
                    d["item_set"] = label_override
                X.write(p, d)
        if short_outcomes:
            p = os.path.join(kd, "outcomes", ofile)
            d = json.load(open(p, encoding="utf-8"))
            d["items"] = [it for it in d["items"] if it["item_id"] not in short_outcomes]
            X.write(p, d)
    for ev in TASKS_FULL:
        if three_override and ev == eval_id:
            three_override(three[ev])
        three[ev]["n_rows"] = len(launch[ev])
        three[ev]["n_complete"] = len(three[ev]["rows"])
        X.write(os.path.join(recs, THREE_NUMBERS_FILES[ev]), three[ev])
    launch_doc = {"schema": B.LAUNCH_SET_SCHEMA, "derived_at": NOW,
                  "evals": {ev: {"own_source": MODULES[ev].SOURCE, "launch": launch[ev],
                                 "canonical_decode": {"policy": f"{MODEL_ID}|{ev}|off|protocol@v1", "thinking": "off", "temperature": TASKS_FULL[ev][4], "cap": TASKS_FULL[ev][3]}}
                            for ev in TASKS_FULL}}
    if relaunched or (instruct and (instruct_relaunched or instruct_two_stores)):
        launch_doc["trees"] = [{"sha": TREE, "role": "primary"}, {"sha": TREE2, "role": "relaunch"}] + [{"sha": t, "role": "parked-only attempt (no store)"} for t in parked_trees]
    if families_block:
        launch_doc["families"] = families_block
    if launch_override:
        launch_override(launch_doc)
    for name in (("launch_sets.json", "launch_sets_full.json") if launch_set_name == "both" else (launch_set_name,)):
        X.write(os.path.join(recs, name), launch_doc)
    if (relaunched or (instruct and instruct_relaunched)) and not no_relaunches:
        entries_rl = {f"{B.launch_tag_of(TAG, k)}/{k}": {"tag": RELAUNCH_TAG, "sha": TREE2, "why": "synthetic: the first launch parked INCOMPLETE (a transport fault)"} for k in relaunched}
        if instruct and instruct_relaunched:
            entries_rl[f"{B.launch_tag_of(IT_TAG, IT_KEY)}/{IT_KEY}"] = {"tag": RELAUNCH_TAG, "sha": (instruct_relaunch_entry_sha or TREE2), "why": "synthetic: refused at standup (no store), relaunched at the fixed tree",
                                                                        # the producer writes `parked_suffix` as a LIST (qwen) or a bare STRING (llama), 2026-09-25 21:17Z
                                                                        **({"parked_suffix": (parked_sfx[0] if (instruct_parked_scalar and len(parked_sfx) == 1) else list(parked_sfx))} if parked_sfx else {})}
        if relaunch_entry_override is not None:
            relaunch_entry_override(entries_rl)
        X.write(os.path.join(root, relaunches_dir, B.RELAUNCHES_FILE), entries_rl)
    X.write(os.path.join(recs, B.ROWS_LAUNCH_FILE), "launched_at\tsha\twave\trow\tcells\tstage\n")
    X.write(os.path.join(recs, B.SCHEDULE_FILE), {"written": NOW, "per_cell_h": 0.25, "planned": []})
    h_roster = {"targets": {MK: {"views": {"main": {"config": f"configs/sdfbase_full22_{MK}.yaml", "cells": {RELU: STAGED, CONTROL_RELU: STAGED_CTL, "prompted__si_constitution": None}},
                                          # the harness roster's `it` selection: the instruct-checkpoint fits (the real roster stages itdoc__ / itdoc64__ / derived_itdoc64__ there)
                                          IT_SEL: {"config": f"configs/sdfbase_full22_{MK}_it.yaml", "cells": {IT_KEY: IT_STAGED, "prompted__si_constitution": None}}}},
                            "qwen35_4b": {"views": {"s2": {"config": "configs/sdfbase_full22_qwen35_4b_s2.yaml", "cells": {RELU: f"base64s2__{RELU}"}},
                                                    "s2p": {"config": "configs/sdfbase_full22_qwen35_4b_s2p.yaml", "paired_of": "s2", "cells": {COMPOSED: f"paired_eb64s2__{COMPOSED}"}}}}}}
    if qwen35b_s2rl:
        # the harness roster's s2rl selection lists THREE keys -- the two launched ReLU refits and the composed difference the launch never ran (LECUN_RELU_ROW_RULE); the
        # reader resolves each launched row by the launch set's pin key (the appendix route) and never enumerates the selection, so the third is simply not a row
        h_roster["targets"][MK35] = {"views": {SEL35_SR: {"config": f"configs/sdfbase_full22_{MK35}_s2rl.yaml",
                                                         "cells": {CTL35_R: STAGED35_SR[CTL35_R], MIX35_R: STAGED35_SR[MIX35_R], DIFF35_R: STAGED35_SR_THIRD}}}}
    h_roster_path = X.write(str(tmp_path / "roster.yaml"), yaml.safe_dump(h_roster))
    h_pins_path = X.write(str(tmp_path / "pins.json"), {MK: {STAGED: {"object_sha256": OBJ, "source": "synthetic"}, STAGED_CTL: {"object_sha256": OBJ_CTL, "source": "synthetic"},
                                                             STAGED_P: {"object_sha256": OBJ_P, "source": "synthetic"}, IT_STAGED: {"object_sha256": IT_OBJ, "source": "synthetic"},
                                                             IT_ALIAS: {"object_sha256": (instruct_alias_sha or IT_OBJ), "source": "synthetic"}}, "qwen35_4b": {},
                                                        **({MK35: {STAGED35: {"object_sha256": OBJ35, "source": "synthetic"}}} if qwen35b_store else {}),
                                                        **({MK35: {STAGED35: {"object_sha256": OBJ35, "source": "synthetic"}, STAGED35_CTL: {"object_sha256": OBJ35_CTL, "source": "synthetic"},
                                                                   STAGED35_DIFF: {"object_sha256": OBJ35_DIFF, "source": "synthetic"},
                                                                   **(PINS35_DR if qwen35b_d0p75_relul else {}), **(PINS35_DL if qwen35b_d1p25 else {}),
                                                                   **(PINS35_SR if qwen35b_s2rl else {})}} if qwen35b_u0 else {})})
    a_roster_path = X.write(str(tmp_path / "arditi_roster.yaml"), yaml.safe_dump({"targets": {"qwen35_4b": {"views": {}}}}))
    a_pins_path = X.write(str(tmp_path / "arditi_pins.json"), {"qwen35_4b": {}})
    return {"root": root, "roster": h_roster_path, "pins": h_pins_path, "arditi_roster": a_roster_path, "arditi_pins": a_pins_path, "keys": keys, "specs": specs[eval_id], "all_specs": specs,
            "roster_sha256": roster_sha256, "roster_sha256_t2": roster_sha256_t2, "pins_sha256": pins_sha, "pins_sha256_t2": pins_sha_t2, "paired_none_key": paired_none_key,
            "paired_none_spec": specs_p[eval_id], "pin_shas": pin_shas, "ids_sha": dict(IDS_SHA), "entries": entries,
            "it_key": keys.get(f"{IT_SEL}/{IT_KEY}"), "it_specs": ({ev: X.spec(eval_id=ev, shape=SHAPE_FULL, **SPECS_IT[ev]) for ev in TASKS_FULL} if instruct else None)}


def adapter(cls, fx: dict, seeds: LC.Seeds, **seed_kw):
    """The adapter over the fixture with the seeds copy that declares the fixture's set (:func:`seeds_for`)."""
    return X.adapter(cls, fx, seeds_for(seeds, **seed_kw))


read, read_group, run_of, run_at, metrics_of, refused, counts = Y.read, Y.read_group, Y.run_of, Y.run_at, Y.metrics_of, Y.refused, Y.counts
