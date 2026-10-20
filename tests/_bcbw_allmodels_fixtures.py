"""A SYNTHETIC mirror shaped like the APPENDIX-family BigCodeBench-200 / BFCL-200 stores (invented item ids and shas, `TESTHOST`, no
generation text), shared by the two appendix readers' tests (`test_registry_adapter_bigcodebench_allmodels.py`, `test_registry_adapter_bfcl_allmodels.py`).
It reuses the seed-replicate fixture's file writers (`_bcbw_seedrep_fixtures`) with the gemma4e4b family's served id / revision / model key, and
builds what the appendix mirror adds:

* the family's ONE seed tag `gemma4e4b_s1` whose conditions ran under the `main` CANARY ROW — row dir `gemma4e4b-f22`, tag `gemma4e4b_main`, store
  suffix `bcbw-gemma4e4b_main-<sha7>`, gate records `gate_gemma4e4b_main_<key>_…` — the unsteered `none`, a fitted ReLU-steer object (resolved through
  the harness roster's `main` selection: `base__<key>` -> the pins) and a prompted key, every one filed under the s1 tag (`canary_layout="all"`, the
  default; `"none"` puts the `none` alone on the canary row and the rest under `gemma4e4b-f22s1`; `"s1"` puts every seed-tag store under
  `gemma4e4b-f22s1`, the pin-key route); the paired tag `gemma4e4b_s1p` with the composed difference_relu object (`paired_eb8__<key>`, pinned in the
  fixture's harness pins under the model key) and its OWN `none`.
* optionally a llama8b `none` (`llama8b-f22s1`, a second family in the same mirror; a `not_run` launch row alone with `llama_none="not_run"`), a
  duplicate ReLU store under the canary row beside the s1 row's (`steered_under_main`, with `canary_layout="s1"`: the canary row's store is refused as
  superseded), a qwen35_4b store (another source's family), and the census's launch + carried shape.
The launch set, the three-number records (named for the appendix sources), the launched pins / roster and the gate records are written from the same
synthetic truth. The canary rows' launch-set rows and three-number rows are spelled per `canary_spelling`: `"s1"` (the census of 2026-09-23
22:19Z: the `view` field s1, tag gemma4e4b_s1, seed_base 1, `bcbw_wave` gemma4e4b_s1 -- the tag the row FILES under -- and `launched_as` naming the
gemma4e4b_main launch tag, row dir and suffix) or `"main"` (the `view` field main, tag gemma4e4b_main, no seed, `bcbw_wave` gemma4e4b_main). The
three-number rows spell `cell_slug` with the store key's `--` token, the way the appendix record does; `llama_tool_calls=False` writes the llama8b
store without the tool-call route (no BFCL row), the way the real llama stores are served."""
from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Optional, Sequence

import _bcbw_seedrep_fixtures as X
from sisteer.registry.adapters import _bcbw_common as B
from sisteer.registry.adapters import bfcl_allmodels as BFA
from sisteer.registry.adapters import bigcodebench_allmodels as BCA
from sisteer.registry.adapters import lineage_common as LC

MK = "gemma4e4b"
MODEL_ID = "gemma4e4b_it"
MODEL_HF = "google/gemma-4-E4B-it"
FIT_HF = "google/gemma-4-E4B"
SEED_N = 1
TAG, TAG_P, TAG_MAIN = f"{MK}_s{SEED_N}", f"{MK}_s{SEED_N}p", f"{MK}{B.LEGACY_MAIN_SUFFIX}"
SEL, SEL_P = f"s{SEED_N}", f"s{SEED_N}p"
ROW_DIR, ROW_DIR_P, ROW_DIR_MAIN = f"{MK}-f22{SEL}", f"{MK}-f22{SEL_P}", f"{MK}-f22"
LLAMA_MK, LLAMA_MODEL_ID, LLAMA_HF, LLAMA_FIT_HF = "llama8b", "llama8b_it", "NousResearch/Meta-Llama-3.1-8B-Instruct", "NousResearch/Meta-Llama-3.1-8B"
LLAMA_TAG, LLAMA_ROW_DIR = f"{LLAMA_MK}_s1", f"{LLAMA_MK}-f22s1"
QWEN_TAG = "qwen35_4b_s2"
NONE, RELU, PROMPTED, COMPOSED = X.NONE, X.RELU, X.PROMPTED, X.COMPOSED
CONTROL_RELU = "constv2ctl2-full__psr_ce_all"
#: the staged ids the appendix run's launch set names as pin keys (the stage index allmodels_s1_stage, 2026-09-23): the `main` selection's
#: `base__<key>` copies for the s1 objects, ReLU's `paired_eb8__<key>` paired write for the s1p object
STAGED = f"base__{RELU}"
STAGED_CTL = f"base__{CONTROL_RELU}"
STAGED_P = f"paired_eb8__{COMPOSED}"
OBJ = hashlib.sha256(b"synthetic bcbw appendix gemma constitution relu").hexdigest()
OBJ_CTL = hashlib.sha256(b"synthetic bcbw appendix gemma control relu").hexdigest()
OBJ_P = hashlib.sha256(b"synthetic bcbw appendix gemma composed paired write").hexdigest()
TREE, SHA7, NOW = X.TREE, X.SHA7, X.NOW
#: a second producer tree (`roster_set=True`): the s1 / s1p stores' launch tree, whose launched roster is the plain (latest) file; the canary row's tree is TREE
TREE_S1 = hashlib.sha1(b"synthetic bcbw appendix s1 tree").hexdigest()
ROSTER_CANARY_FILE = f"{B.ROSTER_FILE[:-5]}.{TREE[:7]}.yaml"
PINS_CANARY_FILE = f"{B.PINS_FILE[:-5]}.{TREE[:7]}.json"
THREE_NUMBERS_FILES = {"bigcodebench": BCA.THREE_NUMBERS_FILE, "bfcl": BFA.THREE_NUMBERS_FILE}
#: per eval: the synthetic truth of every row (the seed-replicate fixture's values re-used; the paired `none` fewer correct than the seed `none`)
SPECS = {ev: {k: X.SPECS[ev][k] for k in (NONE, RELU, PROMPTED, COMPOSED)} for ev in X.TASKS}
SPECS_P = X.SPECS_P
SPEC_LLAMA = {"bigcodebench": dict(n_correct=70, n_wrong=128, at_cap_wrong=2), "bfcl": dict(n_correct=140, n_wrong=60)}
#: (object key, paired?, staged id, object sha, ihash8, chash8, max_lora_rank)
ROWS = [(NONE, False, None, None, "3a1b2c3d", "e4f5a6b7", 1),
        (RELU, False, STAGED, OBJ, "d69810e6", "466f1aaf", 1),
        (PROMPTED, False, None, None, "aa7fa55b", "27ad6f72", 1),
        (COMPOSED, True, STAGED_P, OBJ_P, "f0bb156d", "01671a70", 1)]
ROW_P = (NONE, True, None, None, "3a1b2c3d", "e4f5a6b7", 1)
ROW_LLAMA = (NONE, False, None, None, "7c8d9e0f", "1a2b3c4d", 1)
_REVISIONS: Dict[str, str] = {}


def bind_seeds(seeds: LC.Seeds) -> None:
    X.bind_seeds(seeds)
    for mid in (MODEL_ID, LLAMA_MODEL_ID):
        _REVISIONS[mid] = str(seeds.models[mid]["hf_revision"])


def revision_of(model_id: str) -> str:
    return _REVISIONS[model_id]


def _roster_block(tag: str) -> dict:
    return {"family": B.model_key_of(tag), "view": tag, "legs": {t[1]: "runnable" for t in X.TASKS.values()}, "entries": {}}


def mirror(tmp_path, seeds: LC.Seeds, *, eval_id: str, root: str = None, canary_layout: str = "all", canary_spelling: str = "s1", paired_none: bool = True,
           llama_none=False, llama_tool_calls: bool = True, canary_roster_stale: bool = False, not_run: Sequence[str] = (), steered_under_main: bool = False, qwen_store: bool = False,
           harness_pins_without: Sequence[str] = (), roster_main_override: dict = None, launch_override=None, three_override=None, launch_layout: str = "launch",
           roster_set: bool = False, roster_missing_for: Sequence[str] = (), rosters_index_override=None, launched_pins_without: Sequence[str] = (),
           composed_batch_label: str = "eb8") -> dict:
    """A synthetic appendix mirror root for one reader (the seedrep fixture's `mirror` shape; module docstring). `canary_layout` `all` (the default:
    every seed-tag store under `gemma4e4b-f22`, launch tag `gemma4e4b_main`), `none` (the `none` alone on the canary row) or `s1` (every seed-tag store
    under `gemma4e4b-f22s1`); `canary_spelling` how the launch set and the three-number record spell the canary rows (`s1` / `main`); `llama_none` True
    writes a llama8b `none` store + row on the BigCodeBench eval (llama runs no BFCL row: nothing on the BFCL eval), `"not_run"` its launch row
    and a FAIL gate record alone; `steered_under_main` adds a ReLU store under the canary row beside the s1 row's (with `canary_layout="s1"`);
    `qwen_store` a qwen35_4b s2 `none` store (another source's family); `harness_pins_without` drops staged ids from the fixture's harness pins;
    `roster_main_override` replaces the harness roster's gemma `main` cells; `canary_roster_stale` stamps the canary-row stores' `lane_roster.json` with an
    EARLIER roster revision's sha256 (the real canary stores predate the roster's s1 blocks; two producer trees). `roster_set` writes the ROSTER SET
    (`_bcbw_common.ROSTER_SET_RULE`): the canary row's stores at tree TREE cite their own launched roster `bcb_bfcl_200_widen_roster.<TREE sha7>.yaml` (the
    `_main` blocks alone) and the per-tree pins copy `sdfbase_full22_pins.<TREE sha7>.json` (the plain pins then LACK the canary's ReLU staged id, so the
    copy is what lets that row file), the s1 / s1p stores at tree TREE_S1 cite the plain (latest) roster; `records/rosters.json` maps both hashes to their
    file, tree and stores. `roster_missing_for` names keys whose canary-row store cites a roster hash the set lacks; `rosters_index_override` edits the index.
    `composed_batch_label` spells the composed object's pins key `paired_<label>__<key>` (gemma / nemotron `eb8`, llama `eb64`) in the
    launch set, the launched roster and BOTH pins; `launched_pins_without` drops staged ids from the LAUNCHED pins (the harness pins have `harness_pins_without`),
    so a composed sha in no pins can be shown to refuse."""
    bind_seeds(seeds)
    root = root or str(tmp_path / f"{eval_id}_allmodels")
    recs = os.path.join(root, B.RECORDS_DIR)
    import yaml
    if canary_layout not in ("all", "none", "s1"):
        raise ValueError(canary_layout)
    if steered_under_main and canary_layout == "all":
        raise ValueError("steered_under_main needs canary_layout 'none' or 's1' (the canary row already carries the ReLU store under 'all')")
    launched_roster = {"lane": "bcb-bfcl-200-widen", "legs": [t[0] for t in X.TASKS.values()], "sampling": "protocol", "thinking": "off", "pins": "configs/sdfbase_full22_pins.json",
                       "serving_path": "lora_native", "waves": {}}
    staged_p = f"paired_{composed_batch_label}__{COMPOSED}"
    rows = [(o, p, (staged_p if st == STAGED_P else st), sha, ih, ch, m) for (o, p, st, sha, ih, ch, m) in ROWS] + ([ROW_P] if paired_none else [])
    llama_rows = [ROW_LLAMA] if (llama_none and eval_id == "bigcodebench") else []

    def on_canary_row(tag: str, okey: str) -> bool:
        return tag == TAG and (canary_layout == "all" or (canary_layout == "none" and okey == NONE))

    def launch_tag_for(tag: str, okey: str) -> str:
        return TAG_MAIN if on_canary_row(tag, okey) else B.launch_tag_of(tag, okey)

    def row_dir_for(tag: str, okey: str) -> str:
        return ROW_DIR_MAIN if on_canary_row(tag, okey) else B.row_dir_of(tag)

    def tree_for(tag: str, okey: str) -> str:
        return TREE if (on_canary_row(tag, okey) or not roster_set) else TREE_S1

    for okey, paired, staged, sha, _i, _c, _m in rows:
        tag = TAG_P if paired else TAG
        blk = launched_roster["waves"].setdefault(launch_tag_for(tag, okey), _roster_block(tag))
        entry = {"cell_id": okey, "status": "runnable", "paper_arm": X.entry_id_of(okey)}
        if staged:
            entry["object"] = {"staged_id": staged, "sha256": sha, "kind": "synthetic"}
        blk["entries"][X.entry_id_of(okey)] = entry
    if llama_rows or llama_none == "not_run":
        launched_roster["waves"].setdefault(LLAMA_TAG, _roster_block(LLAMA_TAG))["entries"]["none"] = {"cell_id": NONE, "status": "runnable", "paper_arm": "none"}
    if steered_under_main:
        launched_roster["waves"].setdefault(TAG_MAIN, _roster_block(TAG))["entries"][X.entry_id_of(RELU)] = {"cell_id": RELU, "status": "runnable", "paper_arm": X.entry_id_of(RELU),
                                                                                                             "object": {"staged_id": STAGED, "sha256": OBJ, "kind": "synthetic"}}
    if qwen_store:
        launched_roster["waves"].setdefault(QWEN_TAG, _roster_block(QWEN_TAG))["entries"]["none"] = {"cell_id": NONE, "status": "runnable", "paper_arm": "none"}
    roster_path = X.write(os.path.join(recs, B.ROSTER_FILE), yaml.safe_dump(launched_roster, sort_keys=False))
    roster_sha256 = hashlib.sha256(open(roster_path, "rb").read()).hexdigest()
    canary_roster_sha256 = roster_sha256
    if roster_set:
        # the canary row's OWN launched roster (an earlier revision: the `_main` blocks alone) at its tree, and the index over both
        canary_roster = dict(launched_roster, waves={k: v for k, v in launched_roster["waves"].items() if k.endswith(B.LEGACY_MAIN_SUFFIX)})
        cpath = X.write(os.path.join(recs, ROSTER_CANARY_FILE), yaml.safe_dump(canary_roster, sort_keys=False))
        canary_roster_sha256 = hashlib.sha256(open(cpath, "rb").read()).hexdigest()
        canary_stores = [B.rel_store(ROW_DIR_MAIN, k, X.suffix_of(TAG, k, TAG_MAIN, TREE)) for k in (NONE, RELU, PROMPTED) if on_canary_row(TAG, k)]
        index = {"schema": "bcbw_rosters/1", "written": NOW, "latest": B.ROSTER_FILE,
                 "rosters": {canary_roster_sha256: {"file": ROSTER_CANARY_FILE, "tree_sha": TREE, "stores": canary_stores, "pins": PINS_CANARY_FILE},
                             roster_sha256: {"file": B.ROSTER_FILE, "tree_sha": TREE_S1, "stores": []}}}
        if rosters_index_override:
            rosters_index_override(index)
        X.write(os.path.join(recs, B.ROSTERS_INDEX_FILE), index)
    three = {ev: {"schema": B.THREE_NUMBERS_SCHEMA, "eval": ev, "source": f"{ev}_allmodels", "written": NOW, "contract": "synthetic", "seed_reading": B.SEED_AXIS_RULE, "rows": {}} for ev in X.TASKS}
    launch: Dict[str, List[dict]] = {ev: [] for ev in X.TASKS}
    keys: Dict[str, str] = {}
    specs = {ev: {k: X.spec(eval_id=ev, **v) for k, v in SPECS[ev].items()} for ev in X.TASKS}
    specs_p = {ev: X.spec(eval_id=ev, **SPECS_P[ev]) for ev in X.TASKS}
    specs_llama = {ev: X.spec(eval_id=ev, **SPEC_LLAMA[ev]) for ev in X.TASKS}
    paired_none_key = None
    gemma_kw = dict(model_hf=MODEL_HF, revision=revision_of(MODEL_ID), model_key=MK, fit_model_hf=FIT_HF)
    for i, (okey, paired, staged, sha, ihash8, chash8, mlr) in enumerate(rows):
        tag = TAG_P if paired else TAG
        sel = SEL_P if paired else SEL
        steered = staged is not None
        is_paired_none = paired and okey == NONE
        is_canary = on_canary_row(tag, okey)
        row_specs = {ev: (specs_p[ev] if is_paired_none else specs[ev][okey]) for ev in X.TASKS}
        launch_tag, row_dir, tree = launch_tag_for(tag, okey), row_dir_for(tag, okey), tree_for(tag, okey)
        launched_as = ({"wave": TAG_MAIN, "row_dir": ROW_DIR_MAIN, "suffix": X.suffix_of(tag, okey, TAG_MAIN, TREE), "sha": TREE, "legs": [t[1] for t in X.TASKS.values()]}
                       if is_canary else None)
        slug = okey.replace("__", "--", 1)
        for ev in X.TASKS:
            if is_canary and canary_spelling == "main":
                launch[ev].append(X.launch_row(B.MAIN_SELECTION, okey, TAG_MAIN, pin_key=staged, expected=(sha if staged else None), paired=False, index=i, seed_base=None, launch_tag=TAG_MAIN))
            elif is_canary:
                # the census's shape: the row spelled by the tag it FILES under, the launch spelling on launched_as
                launch[ev].append(X.launch_row(sel, okey, tag, pin_key=staged, expected=(sha if staged else None), paired=paired, index=i, seed_base=SEED_N, launch_tag=tag, launched_as=launched_as))
            else:
                launch[ev].append(X.launch_row(sel, okey, tag, pin_key=staged, expected=(sha if staged else None), paired=paired, index=i, seed_base=SEED_N, launch_tag=launch_tag))
        store_key = f"{okey.replace('__', '--', 1)}__controls__t0__a0__{ihash8}__{chash8}"
        if okey in not_run and not paired:
            gate = X.gate_record(tag, okey, tree=tree, store_key=store_key, obj_sha=sha, steered=steered, max_lora_rank=mlr, verdict="FAIL", completed=False,
                                 row_dir=row_dir, launch_tag=launch_tag, model_key=MK)
            X.write(os.path.join(recs, f"gate_{launch_tag}_{okey}_{X.suffix_of(tag, okey, launch_tag, tree)}.json"), gate)
            continue
        store_roster_sha = (canary_roster_sha256 if is_canary else roster_sha256)
        if is_canary and canary_roster_stale:
            store_roster_sha = hashlib.sha256(b"an earlier revision of the launched roster").hexdigest()
        if is_canary and okey in roster_missing_for:
            store_roster_sha = hashlib.sha256(b"a roster the mirrored set lacks").hexdigest()
        key = X.store_files(root, tag, okey, row_specs, obj_sha=sha, staged=staged, tree=tree, ihash8=ihash8, chash8=chash8, seed_n=SEED_N, roster_sha256=store_roster_sha,
                            max_lora_rank=mlr, row_dir=row_dir, launch_tag=launch_tag, **gemma_kw)
        assert key == store_key
        if is_paired_none:
            paired_none_key = key
        else:
            keys[okey] = key
        gate = X.gate_record(tag, okey, tree=tree, store_key=store_key, obj_sha=sha, steered=steered, max_lora_rank=mlr, row_dir=row_dir, launch_tag=launch_tag, model_key=MK)
        X.write(os.path.join(recs, f"gate_{launch_tag}_{okey}_{X.suffix_of(tag, okey, launch_tag, tree)}.json"), gate)
        for ev in X.TASKS:
            if is_canary and canary_spelling == "main":
                tr = X.three_row(tag, B.MAIN_SELECTION, okey, row_specs[ev], eval_id=ev, obj_sha=sha, tree=tree, store_key=store_key, steered=steered, seed_n=None,
                                 row_dir=row_dir, launch_tag=launch_tag, tag_spelling=TAG_MAIN, cell_slug=slug)
            else:
                tr = X.three_row(tag, sel, okey, row_specs[ev], eval_id=ev, obj_sha=sha, tree=tree, store_key=store_key, steered=steered, seed_n=SEED_N, row_dir=row_dir,
                                 launch_tag=launch_tag, anchor_spec=(specs[ev][NONE] if okey != NONE else None), cell_slug=slug, launched_as=launched_as)
            three[ev]["rows"][tr["key"]] = tr
    # a second family in the same mirror: llama8b's `none` (BigCodeBench only; no BFCL row)
    if llama_rows or llama_none == "not_run":
        okey, _p, _s, _sha, ihash8, chash8, mlr = ROW_LLAMA
        store_key = f"{okey}__controls__t0__a0__{ihash8}__{chash8}"
        if eval_id == "bigcodebench":
            launch[eval_id].append(X.launch_row(SEL, okey, LLAMA_TAG, pin_key=None, expected=None, paired=False, index=len(rows), seed_base=SEED_N))
        if llama_none == "not_run":
            gate = X.gate_record(LLAMA_TAG, okey, tree=TREE, store_key=store_key, obj_sha=None, steered=False, max_lora_rank=mlr, verdict="FAIL", completed=False, model_key=LLAMA_MK)
            X.write(os.path.join(recs, f"gate_{LLAMA_TAG}_{okey}_{X.suffix_of(LLAMA_TAG, okey)}.json"), gate)
        else:
            key = X.store_files(root, LLAMA_TAG, okey, specs_llama, obj_sha=None, staged=None, tree=TREE, ihash8=ihash8, chash8=chash8, seed_n=SEED_N, roster_sha256=roster_sha256,
                                max_lora_rank=mlr, model_hf=LLAMA_HF, revision=revision_of(LLAMA_MODEL_ID), model_key=LLAMA_MK, fit_model_hf=LLAMA_FIT_HF, tool_calls=llama_tool_calls)
            keys[f"{LLAMA_TAG}/{okey}"] = key
            gate = X.gate_record(LLAMA_TAG, okey, tree=TREE, store_key=key, obj_sha=None, steered=False, max_lora_rank=mlr, model_key=LLAMA_MK)
            X.write(os.path.join(recs, f"gate_{LLAMA_TAG}_{okey}_{X.suffix_of(LLAMA_TAG, okey)}.json"), gate)
            for ev in X.TASKS:
                tr = X.three_row(LLAMA_TAG, SEL, okey, specs_llama[ev], eval_id=ev, obj_sha=None, tree=TREE, store_key=key, steered=False, seed_n=SEED_N, cell_slug=okey.replace("__", "--", 1))
                three[ev]["rows"][tr["key"]] = tr
    # a ReLU store under the canary row BESIDE the s1 row's (canary_layout "s1"): the tag's own row wins and the canary row's store is refused as superseded
    if steered_under_main:
        okey, _p, staged, sha, ihash8, chash8, mlr = ROWS[1]
        key = X.store_files(root, TAG, okey, {ev: specs[ev][okey] for ev in X.TASKS}, obj_sha=sha, staged=staged, tree=TREE, ihash8="9999aaaa", chash8="bbbb0000", seed_n=SEED_N,
                            roster_sha256=roster_sha256, max_lora_rank=mlr, row_dir=ROW_DIR_MAIN, launch_tag=TAG_MAIN, **gemma_kw)
        gate = X.gate_record(TAG, okey, tree=TREE, store_key=key, obj_sha=sha, steered=True, max_lora_rank=mlr, row_dir=ROW_DIR_MAIN, launch_tag=TAG_MAIN, model_key=MK)
        X.write(os.path.join(recs, f"gate_{TAG_MAIN}_{okey}_{X.suffix_of(TAG, okey, TAG_MAIN)}.json"), gate)
    # another source's family (the three-seed Qwen family) under this root
    if qwen_store:
        key = X.store_files(root, QWEN_TAG, NONE, {ev: X.spec(eval_id=ev, **X.SPECS[ev][NONE]) for ev in X.TASKS}, obj_sha=None, staged=None, tree=TREE, ihash8="2ff9cdff", chash8="ed8a6ada",
                            seed_n=2, roster_sha256=roster_sha256, max_lora_rank=1)
        gate = X.gate_record(QWEN_TAG, NONE, tree=TREE, store_key=key, obj_sha=None, steered=False, max_lora_rank=1)
        X.write(os.path.join(recs, f"gate_{QWEN_TAG}_{NONE}_{X.suffix_of(QWEN_TAG, NONE)}.json"), gate)
    for ev in X.TASKS:
        if three_override and ev == eval_id:
            three_override(three[ev])
        three[ev]["n_rows"] = len(launch[ev])
        three[ev]["n_complete"] = len(three[ev]["rows"])
        X.write(os.path.join(recs, THREE_NUMBERS_FILES[ev]), three[ev])
    canary_rows = {ev: [r for r in launch[ev] if r["cell"] == NONE and r["tag"] in (TAG, TAG_MAIN) and (r.get("launched_as") or r["bcbw_wave"] == TAG_MAIN)] for ev in X.TASKS}
    launch_doc = {"schema": B.LAUNCH_SET_SCHEMA, "derived_at": NOW,
                  "evals": {ev: {"own_source": f"{ev}_allmodels", "launch": launch[ev], "canary": (canary_rows[ev][0] if canary_rows[ev] else None),
                                 "canonical_decode": {"policy": f"{MODEL_ID}|{ev}|off|protocol@v1", "thinking": "off", "temperature": X.TASKS[ev][4], "cap": X.TASKS[ev][3]}}
                            for ev in X.TASKS}}
    if launch_layout == "carried":
        for ev in X.TASKS:
            launch_doc["evals"][ev].update(X.census_row_table(launch[ev], SEL, SEL_P))
    elif launch_layout != "launch":
        raise ValueError(launch_layout)
    if launch_override:
        launch_override(launch_doc)
    X.write(os.path.join(recs, B.LAUNCH_SET_FILE), launch_doc)
    launched_pins = {MK: {k: v for k, v in {STAGED: {"object_sha256": OBJ, "source": "synthetic"}, STAGED_CTL: {"object_sha256": OBJ_CTL, "source": "synthetic"},
                                             staged_p: {"object_sha256": OBJ_P, "source": "synthetic"}}.items() if k not in launched_pins_without},
                     LLAMA_MK: {}, "qwen35_4b": {}}
    if roster_set:
        # the pins DIFFERED between the trees: the canary tree's copy carries the ReLU object, the latest (plain) pins do not
        X.write(os.path.join(recs, PINS_CANARY_FILE), {MK: {STAGED: {"object_sha256": OBJ, "source": "synthetic"}, STAGED_CTL: {"object_sha256": OBJ_CTL, "source": "synthetic"}}, LLAMA_MK: {}, "qwen35_4b": {}})
        launched_pins[MK].pop(STAGED)
    X.write(os.path.join(recs, B.PINS_FILE), launched_pins)
    X.write(os.path.join(recs, B.ROWS_LAUNCH_FILE), "launched_at\tsha\twave\trow\tcells\tstage\n")
    X.write(os.path.join(recs, B.SCHEDULE_FILE), {"written": NOW, "per_cell_h": 0.25, "planned": []})
    # the harness roster: the appendix families have NO s1 / s1p selection (the real roster at 5c13e40f); their `main` selection stages the s1 objects' ids
    main_cells = {RELU: STAGED, CONTROL_RELU: STAGED_CTL, "prompted__si_constitution": None} if roster_main_override is None else dict(roster_main_override)
    h_roster = {"targets": {MK: {"views": {"main": {"config": f"configs/sdfbase_full22_{MK}.yaml", "cells": main_cells},
                                           "it": {"config": f"configs/sdfbase_full22_{MK}_it.yaml", "cells": {RELU: f"itdoc__{RELU}", PROMPTED: None}}}},
                            LLAMA_MK: {"views": {"main": {"config": f"configs/sdfbase_full22_{LLAMA_MK}.yaml", "cells": {"constv2-mix-full__lora_r4_ce": "base__constv2-mix-full__lora_r4_ce"}}}},
                            "qwen35_4b": {"views": {"s2": {"config": "configs/sdfbase_full22_qwen35_4b_s2.yaml", "cells": {RELU: f"base64s2__{RELU}"}},
                                                    "s2p": {"config": "configs/sdfbase_full22_qwen35_4b_s2p.yaml", "paired_of": "s2", "cells": {COMPOSED: f"paired_eb64s2__{COMPOSED}"}}}}}}
    h_roster_path = X.write(str(tmp_path / "roster.yaml"), yaml.safe_dump(h_roster))
    h_pins = {MK: {k: v for k, v in {STAGED: {"object_sha256": OBJ, "source": "synthetic"}, STAGED_CTL: {"object_sha256": OBJ_CTL, "source": "synthetic"},
                                     staged_p: {"object_sha256": OBJ_P, "source": "synthetic"}}.items() if k not in harness_pins_without},
              LLAMA_MK: {}, "qwen35_4b": {}}
    h_pins_path = X.write(str(tmp_path / "pins.json"), h_pins)
    a_roster_path = X.write(str(tmp_path / "arditi_roster.yaml"), yaml.safe_dump({"targets": {"qwen35_4b": {"views": {}}}}))
    a_pins_path = X.write(str(tmp_path / "arditi_pins.json"), {"qwen35_4b": {}})
    return {"root": root, "roster": h_roster_path, "pins": h_pins_path, "arditi_roster": a_roster_path, "arditi_pins": a_pins_path, "keys": keys, "specs": specs[eval_id], "all_specs": specs,
            "roster_sha256": roster_sha256, "canary_roster_sha256": canary_roster_sha256, "paired_none_key": paired_none_key, "paired_none_spec": specs_p[eval_id],
            "llama_spec": specs_llama[eval_id], "canary_layout": canary_layout, "canary_spelling": canary_spelling, "staged_p": staged_p}


def adapter(cls, fx: dict, seeds: LC.Seeds):
    return X.adapter(cls, fx, seeds)


def read_group(ad, group: str, conn=None):
    """Read ONE group of a mirror that may hold several (the seedrep fixture's `read` insists on a single group) on `conn`, or on a frame primed from
    the adapter's own emissions over every group (`X.primed_frame`); every record validated."""
    from sisteer.registry import validate as VAL
    inputs = ad.discover()
    groups: Dict[str, list] = {}
    for i in inputs:
        groups.setdefault(ad.comparison_group(i), []).append(i)
    assert group in groups, (group, sorted(groups))
    recs = []
    for batch in ad.read(X.ctx(X.primed_conn(ad, conn)), group, groups[group]):
        recs.extend(batch.records)
    bad = [str(v) for v in (VAL.validate_record(r, path="<test>") for r in recs) if isinstance(v, VAL.Malformed)]
    assert not bad, bad[:5]
    bt: Dict[str, list] = {}
    for r in recs:
        bt.setdefault(r["record_type"], []).append(r["payload"])
    return bt, ad.status_extra(group), sorted(groups)


def read(ad, conn=None, group: str = TAG):
    return X.read(ad, conn=conn, group=group)


run_of, metrics_of, refused, counts = X.run_of, X.metrics_of, X.refused, X.counts


def run_at(bt, tag: str, okey: str) -> dict:
    """The one run of (tag, object key): the seed tag's `none` and the paired tag's own `none` share an object key, so the tag selects."""
    hits = [r for r in bt["run"] if r["view_legacy"] == tag and r["campaign_cell_id_legacy"] == okey]
    assert len(hits) == 1, (tag, okey, len(hits))
    return hits[0]
