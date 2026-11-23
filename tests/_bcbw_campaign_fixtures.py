"""A SYNTHETIC mirror of a DECLARED CAMPAIGN's full-suite BigCodeBench / BFCL stores (`_bcbw_common.CampaignSpec`; invented object shas and item ids, `TESTHOST`, no
generation text), for `tests/test_registry_adapter_bcbw_campaigns.py`. It reuses the full fixture's writers (`_bcbw_seedrep_fixtures` bound to the FULL shape of
`_bcbw_full_fixtures`: both full tasks per store over a SMALL pin of `N_FULL` ids, the item-set index and pin files, the seeds copy declaring them) and writes
what a campaign's mirror carries:

* a TEMPORARY CHECKOUT (`fx["checkout"]`, passed to the reader as `pins_root`) holding the campaign's harness roster in the pin form (`targets.qwen35_4b.views.<view
  selection>.cells.<object key> = {pins: <the campaign pins, relative>, pin: qwen35_4b.<staged id>}`) and the campaign's harness pins, at the paths the declaration
  names (the production SIKL4 declaration's own `configs/sdfbase_full22_roster_sikl4.yaml` / `configs/sikl4_pins.json`, with synthetic objects);
* the stores under `native/<view row dir>/<object key>/thinking_off/bcbw-<view tag>-<sha7>[-r2]/`, fit on the served checkpoint for the sikl4 shape
  (instruct_direct, `fit_provenance.transfer` false) and on the base checkpoint for the ctlstop shape (base_to_instruct);
* `records/`: the launch set (`launch_sets_full.json`; one row per (view, cell): the view's tag / selection / fit seed / transfer kind, `seed_base` null, the
  campaign's model key, the pin key and expected sha, the item set), the gate records (`gate_<view tag>_<object key>_<suffix>.json`, done markers relative to the
  suffix's parent), the launched roster `bcb_bfcl_full_roster.yaml` (waves keyed by the view tags, every store's `lane_roster.json:roster_sha256` its sha), the
  launched campaign pins copy under the declaration's name, the item-set index and pin files, and the three-number records of both evals (`source` the campaign source).

Two SHAPES: `sikl4` -- the production declaration `_bcbw_campaigns.SIKL4` (six fitted cells per view, no `none`, anchor `none`); `ctlstop` -- the synthetic
:data:`CTLSTOP` declaration of the other campaign shape (four fitted cells per view incl. two pinned composed differences, the view's own `none`, base-to-instruct
fits, anchor `own_none`). Knobs (one record or one store each; every one a refusal or a path the tests pin): `paper_row_dir` (a complete paper-grammar
store under `native/qwen35_4b-f22s1/`), `undeclared_row_dir` (a campaign-like row dir the campaign does not declare), `suffix_override` ({(view, key): suffix}:
the store directory under another suffix), `short_outcomes` ({(view, key)}: one item dropped from the BigCodeBench outcome record), `none_store` (a `none`
store and its launch row under a sikl4 view), `prompted_store` (a prompted store and its launch row), `launch_row_edit` ({(view, key): fn(row)}),
`attest_sha` ({(view, key): sha}: the store attests another object), `launched_pins_sha` ({(view, key): sha}: the launched pins copy names another object),
`fit_hf` (every fitted store's `fit_provenance.model_id`), `three_source` (the three-number records' `source`), `lane_roster_wave` ({(view, key): wave}),
`harness_pins_rel_override` (the roster's pin form names another pins file), `relaunched` ({(view, key)}: the first attempt parked INCOMPLETE at TREE, the
`-r2` store read at TREE2, `relaunches.json` naming it), `one_task_stores` ({(view, key)}: the row's two tasks in two ONE-TASK stores at TREE,
`bcbw-<view tag>-<sha7>-bcb` holding BigCodeBench alone and `-bfcl` holding BFCL alone, each with its gate record, each eval's launch row and three-number row
naming its own store on `launched_as`; ONE_TASK_STORE_RULE)."""
from __future__ import annotations

import hashlib
import json
import os
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import yaml

import _bcbw_full_fixtures as Z
import _bcbw_seedrep_fixtures as X
from sisteer.registry.adapters import _bcbw_common as B
from sisteer.registry.adapters import _bcbw_campaigns as CS
from sisteer.registry.adapters import bfcl_full as BFF
from sisteer.registry.adapters import bfcl_full_sikl4 as BFL
from sisteer.registry.adapters import bigcodebench_full as BCF
from sisteer.registry.adapters import bigcodebench_full_sikl4 as BCL
from sisteer.registry.adapters import lineage_common as LC

MK = "qwen35_4b"
MODEL_ID = "qwen35_4b_it"
SERVED_HF = "Qwen/Qwen3.5-4B"
BASE_HF = "Qwen/Qwen3.5-4B-Base"
TREE, TREE2, NOW = Z.TREE, Z.TREE2, Z.NOW
N = Z.N_FULL
NONE = "none"
PROMPTED = "prompted__safety_refusal"
#: the sikl4 shape's six fitted cells (the production roster's spellings)
SIKL4_CELLS = ("safety_refusal__lora_r4", "safety_refusal__psr_kl_all", "si_constitution__lora_r4", "si_constitution__psr_kl_all", "C16__lora_r4", "C16__psr_kl_all")
#: the ctlstop shape's four fitted cells (two fitted controls and their two pinned composed differences)
CTLSTOP_CELLS = ("constv2ctl2-full__lora_r4_ce", "constv2ctl2-full__psr_ce_all", "constv2-mix-full__lora_r4_ce_minus_ctl", "constv2-mix-full__psr_ce_all_minus_ctl")
#: the synthetic declaration of the OTHER campaign shape (served by design, never registered in production code)
CTLSTOP = B.CampaignSpec(name="ctlstop64", campaign="reseed_ctlstop", model_key=MK,
                     entries=tuple(B.CampaignTag(tag=f"qwen35_4b_ctlstop64s{k}", row_dir=f"qwen35_4b-f22ctlstop64s{k}", selection=f"ctlstop64s{k}", fit_seed=k,
                                            transfer_kind="base_to_instruct") for k in (1, 2, 3)),
                     harness_roster_rel=("configs", "sdfbase_full22_roster_ctlstop_eb64.yaml"), harness_pins_rel=("configs", "ctlstop_eb64_pins.json"),
                     launched_pins_file="ctlstop_eb64_pins.json", admits_none=True, anchor=B.DECLARED_ANCHOR_OWN_NONE, producer_session="TESTSESSION")
SHAPES = {"sikl4": (CS.SIKL4, SIKL4_CELLS), "ctlstop": (CTLSTOP, CTLSTOP_CELLS)}
FULL_MODULES = {"bigcodebench": BCF, "bfcl": BFF}
#: per eval: the campaign source names of the two shapes (the sikl4 pair is registered; the ctlstop pair is the test's)
SOURCES = {"sikl4": {"bigcodebench": BCL.SOURCE, "bfcl": BFL.SOURCE}, "ctlstop": {"bigcodebench": "bigcodebench_full_ctlstop64", "bfcl": "bfcl_full_ctlstop64"}}
#: the synthetic truth per cell index (cycled), per eval, over the N_FULL items
TRUTH = {"bigcodebench": [dict(n_correct=3, n_wrong=2), dict(n_correct=2, n_wrong=2, at_cap_wrong=1), dict(n_correct=4, n_wrong=1), dict(n_correct=1, n_wrong=3, at_cap_unscored=1),
                          dict(n_correct=2, n_wrong=3), dict(n_correct=3, n_wrong=1, at_cap_wrong=1)],
         "bfcl": [dict(n_correct=4, n_wrong=1), dict(n_correct=3, n_wrong=2), dict(n_correct=2, n_wrong=2, at_cap_wrong=1), dict(n_correct=4, n_wrong=1), dict(n_correct=3, n_wrong=1, at_cap_wrong=1),
                  dict(n_correct=2, n_wrong=3)]}
NONE_TRUTH = {"bigcodebench": dict(n_correct=3, n_wrong=2), "bfcl": dict(n_correct=4, n_wrong=1)}
#: per eval: the task tag a ONE-TASK store's suffix carries (`_bcbw_common.TASK_TAGS` inverted: `-bcb` / `-bfcl`)
TASK_TAG_OF = {ev: t for t, ev in B.TASK_TAGS.items()}
PROMPTED_TRUTH = {"bigcodebench": dict(n_correct=2, n_wrong=3), "bfcl": dict(n_correct=3, n_wrong=2)}

Key = Tuple[int, str]


def staged_of(decl: B.CampaignSpec, view: B.CampaignTag, okey: str) -> str:
    return f"{view.selection}__{okey}"


def sha_of(decl: B.CampaignSpec, view: B.CampaignTag, okey: str) -> str:
    return hashlib.sha256(f"synthetic campaign {decl.name} {view.selection} {okey}".encode()).hexdigest()


def three_file(shape: str, eval_id: str) -> str:
    return f"three_numbers_{SOURCES[shape][eval_id]}.json"


def ctlstop_spec(eval_id: str):
    """The test's ctlstop-shaped source's spec: the full reader's, parametrized with :data:`CTLSTOP` (never registered)."""
    return B.campaign_task_spec(FULL_MODULES[eval_id].SPEC, SOURCES["ctlstop"][eval_id], CTLSTOP)


def _ctlstop_class(eval_id: str):
    spec = ctlstop_spec(eval_id)

    class CtlstopLaneReader(B.BcbwSeedrepReader):
        name = spec.name
        SPEC = spec
        covers_sources = (spec.source,)
        EMISSION_VERSION = 1
        owner = CTLSTOP.producer_session
    return CtlstopLaneReader


CTLSTOP_ADAPTERS = {ev: _ctlstop_class(ev) for ev in FULL_MODULES}
SIKL4_ADAPTERS = {"bigcodebench": BCL.BigcodebenchFullSikl4Adapter, "bfcl": BFL.BfclFullSikl4Adapter}


def mirror(tmp_path, seeds: LC.Seeds, *, eval_id: str, shape: str = "sikl4", views: Sequence[int] = (1, 2, 3), root: Optional[str] = None,
           paper_row_dir: bool = False, undeclared_row_dir: Optional[str] = None, suffix_override: Optional[Mapping[Key, str]] = None,
           short_outcomes: Set[Key] = frozenset(), none_store: bool = False, prompted_store: bool = False,
           launch_row_edit: Optional[Mapping[Key, Callable[[dict], None]]] = None, attest_sha: Optional[Mapping[Key, str]] = None,
           launched_pins_sha: Optional[Mapping[Key, str]] = None, fit_hf: Optional[str] = None, three_source: Optional[str] = None,
           lane_roster_wave: Optional[Mapping[Key, str]] = None, harness_pins_rel_override: Optional[str] = None, relaunched: Set[Key] = frozenset(),
           one_task_stores: Set[Key] = frozenset()) -> dict:
    """A synthetic campaign mirror root for one reader (module docstring); returns the fixture dict (`root`, `checkout`, `roster`, `pins`, `keys`, `specs`, `campaign`, ...)."""
    Z.bind_seeds(seeds)
    decl, cells = SHAPES[shape]
    the_views = [decl.entries[k - 1] for k in views]
    root = root or str(tmp_path / f"{eval_id}_campaign_{shape}")
    recs = os.path.join(root, B.RECORDS_DIR)
    checkout = str(tmp_path / "checkout")
    edits = dict(launch_row_edit or {})
    # (a one-task row's two stores carry the task tag; the knobs that edit THE row's one two-task store do not combine with it)
    assert not set(one_task_stores) & (set(relaunched) | set(short_outcomes) | set(suffix_override or {}) | set(lane_roster_wave or {})), "one_task_stores combines with no store knob"
    revision = str(seeds.models[MODEL_ID]["hf_revision"])
    fitted_hf = fit_hf or (SERVED_HF if decl.entries[0].transfer_kind == B.INSTRUCT_TRANSFER_KIND else BASE_HF)
    fit_transfer = decl.entries[0].transfer_kind != B.INSTRUCT_TRANSFER_KIND
    # ---- the checkout: the harness roster (pin form) and pins at the declaration's paths
    pins_rel = "/".join(decl.harness_pins_rel)
    h_roster = {"wave": "full22", "targets": {MK: {"views": {}}}}
    h_pins: Dict[str, dict] = {MK: {}}
    for v in decl.entries:
        h_roster["targets"][MK]["views"][v.selection] = {"config": f"configs/synthetic_{v.selection}.yaml",
                                                         "cells": {c: {"pins": (harness_pins_rel_override or pins_rel), "pin": f"{MK}.{staged_of(decl, v, c)}"} for c in cells}}
        for c in cells:
            h_pins[MK][staged_of(decl, v, c)] = {"object_sha256": sha_of(decl, v, c), "kind": "synthetic", "staged_id": staged_of(decl, v, c), "source": "synthetic"}
    roster_path = X.write(os.path.join(checkout, *decl.harness_roster_rel), yaml.safe_dump(h_roster, sort_keys=False))
    pins_path = X.write(os.path.join(checkout, *decl.harness_pins_rel), {"_comment": "synthetic", **h_pins})
    # ---- the launched roster (every lane_roster.json names its sha256)
    launched_roster = {"lane": "bcb-bfcl-full", "legs": [t[0] for t in Z.TASKS_FULL.values()], "sampling": "protocol", "thinking": "off", "pins": pins_rel,
                       "serving_path": "lora_native", "waves": {}}
    rows_of_view: Dict[int, List[str]] = {}
    for v in the_views:
        keys_v = list(cells)
        if decl.admits_none:
            keys_v = [NONE] + keys_v
        if none_store and not decl.admits_none:
            keys_v = [NONE] + keys_v
        if prompted_store:
            keys_v = keys_v + [PROMPTED]
        rows_of_view[v.fit_seed] = keys_v
        blk = launched_roster["waves"].setdefault(v.tag, {"family": MK, "view": v.tag, "legs": {t[1]: "runnable" for t in Z.TASKS_FULL.values()}, "entries": {}})
        for okey in keys_v:
            entry = {"cell_id": okey, "status": "runnable", "paper_arm": X.entry_id_of(okey)}
            if okey in cells:
                entry["object"] = {"staged_id": staged_of(decl, v, okey), "sha256": sha_of(decl, v, okey), "pins": pins_rel}
            blk["entries"][X.entry_id_of(okey)] = entry       # (keyed as the store's lane_roster.json:entry_id spells it)
    roster_file = os.path.join(recs, BCF.SPEC.roster_file)
    X.write(roster_file, yaml.safe_dump(launched_roster, sort_keys=False))
    roster_sha256 = hashlib.sha256(open(roster_file, "rb").read()).hexdigest()
    # ---- the launched campaign pins copy (under the declaration's file name)
    l_pins = json.loads(json.dumps(h_pins))
    for (k, okey), sha in (launched_pins_sha or {}).items():
        l_pins[MK][staged_of(decl, decl.entries[k - 1], okey)]["object_sha256"] = sha
    X.write(os.path.join(recs, decl.launched_pins_file), l_pins)
    # ---- the item-set index and the pin files (ITEM_SET_RULE)
    pin_shas: Dict[str, str] = {}
    for ev, mod in FULL_MODULES.items():
        p = X.write(os.path.join(recs, mod.PIN_FILE), Z.pin_doc(ev))
        pin_shas[ev] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    X.write(os.path.join(recs, B.ITEM_SETS_FILE), [{"id": FULL_MODULES[ev].ITEM_SET_ID, "sha256": pin_shas[ev], "n": N, "file": FULL_MODULES[ev].PIN_FILE,
                                                     "tree_file": f"configs/capeval/tasks/{FULL_MODULES[ev].PIN_FILE}"} for ev in FULL_MODULES])
    three = {ev: {"schema": B.THREE_NUMBERS_SCHEMA, "eval": ev, "source": (three_source or SOURCES[shape][ev]), "written": NOW, "contract": "synthetic",
                  "seed_reading": B.SEED_AXIS_RULE, "rows": {}} for ev in Z.TASKS_FULL}
    launch: Dict[str, List[dict]] = {ev: [] for ev in Z.TASKS_FULL}
    keys: Dict[str, str] = {}
    specs: Dict[str, Dict[str, dict]] = {}
    relaunch_entries: Dict[str, dict] = {}
    idx = 0
    for v in the_views:
        for i, okey in enumerate(rows_of_view[v.fit_seed]):
            k = (v.fit_seed, okey)
            fitted = okey in cells
            obj = sha_of(decl, v, okey) if fitted else None
            staged = staged_of(decl, v, okey) if fitted else None
            if okey == NONE:
                row_specs = {ev: X.spec(eval_id=ev, shape=Z.SHAPE_FULL, **NONE_TRUTH[ev]) for ev in Z.TASKS_FULL}
            elif okey == PROMPTED:
                row_specs = {ev: X.spec(eval_id=ev, shape=Z.SHAPE_FULL, **PROMPTED_TRUTH[ev]) for ev in Z.TASKS_FULL}
            else:
                row_specs = {ev: X.spec(eval_id=ev, shape=Z.SHAPE_FULL, **TRUTH[ev][(i + v.fit_seed) % len(TRUTH[ev])]) for ev in Z.TASKS_FULL}
            specs[f"{v.tag}|{okey}"] = row_specs
            slug = okey.replace("__", "--", 1)
            ih8 = hashlib.sha256(f"ih {v.tag} {okey}".encode()).hexdigest()[:8]
            ch8 = hashlib.sha256(f"ch {v.tag} {okey}".encode()).hexdigest()[:8]
            is_rl = k in relaunched
            tree = TREE2 if is_rl else TREE
            extra = "-r2" if is_rl else ""
            suffix = X.suffix_of(v.tag, okey, v.tag, tree, extra)
            launched_as = {"wave": v.tag, "row_dir": v.row_dir, "suffix": suffix, "sha": TREE2} if is_rl else None
            store_kw = dict(obj_sha=obj, staged=staged, ihash8=ih8, chash8=ch8, seed_n=v.fit_seed, roster_sha256=roster_sha256, max_lora_rank=4, model_hf=SERVED_HF,
                            revision=revision, model_key=MK, fit_model_hf=fitted_hf, row_dir=v.row_dir, launch_tag=v.tag, shape=Z.SHAPE_FULL,
                            fit_transfer=(fit_transfer if fitted else None), attest_sha=(attest_sha or {}).get(k))
            if is_rl:
                # the PARKED first attempt at TREE: RESULT INCOMPLETE, never the row's store (RELAUNCH_RULE); its gate record reads FAIL
                X.store_files(root, v.tag, okey, row_specs, tree=TREE, result_complete=False, **store_kw)
                skey0 = f"{slug}__controls__t0__a0__{ih8}__{ch8}"
                g0 = X.gate_record(v.tag, okey, tree=TREE, store_key=skey0, obj_sha=obj, steered=fitted, max_lora_rank=4, row_dir=v.row_dir, launch_tag=v.tag, model_key=MK,
                                   shape=Z.SHAPE_FULL, verdict="FAIL", completed=False)
                X.write(os.path.join(recs, f"gate_{v.tag}_{okey}_{X.suffix_of(v.tag, okey, v.tag, TREE)}.json"), g0)
                relaunch_entries[f"{v.tag}/{okey}"] = {"tag": "r2", "sha": TREE2, "why": "synthetic: the first attempt parked INCOMPLETE"}
            if k in one_task_stores:
                # the row's two tasks in two ONE-TASK stores (ONE_TASK_STORE_RULE): each holds its task alone under its task tag, with its own gate record; each eval's
                # launch row and three-number row name ITS store on launched_as (the row's sha = the store's tree, LAUNCHED_AS_TREE_RULE)
                task_extra = {ev: f"{extra}-{TASK_TAG_OF[ev]}" for ev in Z.TASKS_FULL}
                for ev in Z.TASKS_FULL:
                    skey = X.store_files(root, v.tag, okey, {ev: row_specs[ev]}, tree=tree, suffix_extra=task_extra[ev], **dict(store_kw, shape=Z.SHAPE_ONE_TASK[ev]))
                    g1 = X.gate_record(v.tag, okey, tree=tree, store_key=skey, obj_sha=((attest_sha or {}).get(k) or obj), steered=fitted, max_lora_rank=4, row_dir=v.row_dir,
                                       launch_tag=v.tag, model_key=MK, shape=Z.SHAPE_ONE_TASK[ev], suffix_extra=task_extra[ev])
                    X.write(os.path.join(recs, f"gate_{v.tag}_{okey}_{X.suffix_of(v.tag, okey, v.tag, tree, task_extra[ev])}.json"), g1)
                keys[f"{v.tag}|{okey}"] = skey
                for ev in Z.TASKS_FULL:
                    la1 = {"wave": v.tag, "row_dir": v.row_dir, "suffix": X.suffix_of(v.tag, okey, v.tag, tree, task_extra[ev]), "sha": tree, "tag": TASK_TAG_OF[ev]}
                    lr = X.launch_row(v.selection, okey, v.tag, pin_key=staged, expected=obj, paired=False, index=idx, seed_base=None, launch_tag=v.tag, launched_as=la1)
                    lr.update(fit_seed=(v.fit_seed if fitted else None), transfer_kind=(v.transfer_kind if fitted else None), model_key=MK, sha=tree,
                              item_set={"id": FULL_MODULES[ev].ITEM_SET_ID, "sha256": pin_shas[ev], "n": N})
                    if k in edits:
                        edits[k](lr)
                    launch[ev].append(lr)
                    tr = X.three_row(v.tag, v.selection, okey, row_specs[ev], eval_id=ev, obj_sha=((attest_sha or {}).get(k) or obj), tree=tree, store_key=skey, steered=fitted,
                                     seed_n=None, row_dir=v.row_dir, launch_tag=v.tag, cell_slug=slug, launched_as=la1, shape=Z.SHAPE_FULL, suffix_extra=task_extra[ev])
                    three[ev]["rows"][tr["key"]] = tr
                idx += 1
                continue
            skey = X.store_files(root, v.tag, okey, row_specs, tree=tree, suffix_extra=extra, **store_kw)
            keys[f"{v.tag}|{okey}"] = skey
            store_dir = os.path.join(root, B.STORES_HEAD, v.row_dir, okey, B.THINKING_DIR, suffix)
            if k in (lane_roster_wave or {}):
                p = os.path.join(store_dir, B.LAUNCHED_ENTRY_FILE)
                d = json.load(open(p, encoding="utf-8"))
                d["wave"] = lane_roster_wave[k]
                X.write(p, d)
            if k in short_outcomes and eval_id in Z.TASKS_FULL:
                ofile = Z.TASKS_FULL[eval_id][2]
                p = os.path.join(store_dir, "cells", skey, "outcomes", ofile)
                d = json.load(open(p, encoding="utf-8"))
                d["items"] = d["items"][1:]
                X.write(p, d)
            gate = X.gate_record(v.tag, okey, tree=tree, store_key=skey, obj_sha=((attest_sha or {}).get(k) or obj), steered=fitted, max_lora_rank=4, row_dir=v.row_dir,
                                 launch_tag=v.tag, model_key=MK, shape=Z.SHAPE_FULL, suffix_extra=extra)
            X.write(os.path.join(recs, f"gate_{v.tag}_{okey}_{suffix}.json"), gate)
            for ev in Z.TASKS_FULL:
                lr = X.launch_row(v.selection, okey, v.tag, pin_key=staged, expected=obj, paired=False, index=idx, seed_base=None, launch_tag=v.tag, launched_as=launched_as)
                lr.update(fit_seed=(v.fit_seed if fitted else None), transfer_kind=(v.transfer_kind if fitted else None), model_key=MK,
                          item_set={"id": FULL_MODULES[ev].ITEM_SET_ID, "sha256": pin_shas[ev], "n": N})
                if is_rl:
                    lr["sha"] = TREE2
                    lr["relaunched_from"] = {"suffix": X.suffix_of(v.tag, okey, v.tag, TREE), "why": "synthetic: the first attempt parked INCOMPLETE"}
                if k in edits:
                    edits[k](lr)
                launch[ev].append(lr)
                tr = X.three_row(v.tag, v.selection, okey, row_specs[ev], eval_id=ev, obj_sha=((attest_sha or {}).get(k) or obj), tree=tree, store_key=skey, steered=fitted,
                                 seed_n=None, row_dir=v.row_dir, launch_tag=v.tag, cell_slug=slug, launched_as=launched_as, shape=Z.SHAPE_FULL, suffix_extra=extra)
                three[ev]["rows"][tr["key"]] = tr
            idx += 1
    # ---- suffix overrides: move a store directory to another suffix (its gate record and launch row keep the declared one)
    for (k, okey), new_suffix in (suffix_override or {}).items():
        v = decl.entries[k - 1]
        base_dir = os.path.join(root, B.STORES_HEAD, v.row_dir, okey, B.THINKING_DIR)
        os.rename(os.path.join(base_dir, X.suffix_of(v.tag, okey, v.tag, TREE)), os.path.join(base_dir, new_suffix))
    none_specs = {ev: X.spec(eval_id=ev, shape=Z.SHAPE_FULL, **NONE_TRUTH[ev]) for ev in Z.TASKS_FULL}
    stray_kw = dict(obj_sha=None, staged=None, tree=TREE, ihash8="2ff9cdff", chash8="0badf00d", seed_n=1, roster_sha256=roster_sha256, max_lora_rank=1, model_hf=SERVED_HF,
                    revision=revision, model_key=MK, shape=Z.SHAPE_FULL)
    if paper_row_dir:
        # a complete paper-grammar store (the three-seed family's s1 none) under the campaign's root: an undeclared row dir for the campaign reader
        X.store_files(root, "qwen35_4b_s1", NONE, none_specs, row_dir="qwen35_4b-f22s1", launch_tag="qwen35_4b_s1", **stray_kw)
    if undeclared_row_dir:
        X.store_files(root, decl.entries[0].tag, NONE, none_specs, row_dir=undeclared_row_dir, launch_tag=decl.entries[0].tag, **stray_kw)
    for ev in Z.TASKS_FULL:
        three[ev]["n_rows"] = len(launch[ev])
        three[ev]["n_complete"] = len(three[ev]["rows"])
        X.write(os.path.join(recs, three_file(shape, ev)), three[ev])
    launch_doc = {"schema": B.LAUNCH_SET_SCHEMA, "derived_at": NOW,
                  "evals": {ev: {"own_source": SOURCES[shape][ev], "launch": launch[ev],
                                 "canonical_decode": {"policy": f"{MODEL_ID}|{ev}|off|protocol@v1", "thinking": "off", "temperature": Z.TASKS_FULL[ev][4], "cap": Z.TASKS_FULL[ev][3]}}
                            for ev in Z.TASKS_FULL}}
    if relaunched:
        launch_doc["trees"] = [{"sha": TREE, "role": "primary"}, {"sha": TREE2, "role": "relaunch"}]
        X.write(os.path.join(recs, B.RELAUNCHES_FILE), relaunch_entries)
    X.write(os.path.join(recs, "launch_sets_full.json"), launch_doc)
    X.write(os.path.join(recs, B.ROWS_LAUNCH_FILE), "launched_at\tsha\twave\trow\tcells\tstage\n")
    X.write(os.path.join(recs, B.SCHEDULE_FILE), {"written": NOW, "per_cell_h": 0.25, "planned": []})
    return {"root": root, "checkout": checkout, "roster": roster_path, "pins": pins_path, "keys": keys, "specs": specs, "declared": decl, "cells": cells, "shape": shape,
            "views": the_views, "roster_sha256": roster_sha256, "pin_shas": pin_shas}


def adapter(fx: dict, seeds: LC.Seeds, eval_id: str, **seed_kw):
    """The campaign reader over the fixture: the production sikl4 adapter, or the test's ctlstop-shaped one, bound to the fixture's checkout (`pins_root`) and to
    the seeds copy that declares the fixture's item set."""
    cls = (SIKL4_ADAPTERS if fx["shape"] == "sikl4" else CTLSTOP_ADAPTERS)[eval_id]
    return cls(root=fx["root"], seeds=Z.seeds_for(seeds, **seed_kw), roster_path=fx["roster"], pins_path=fx["pins"], pins_root=fx["checkout"])


def groups_of(ad) -> Dict[str, list]:
    inputs = ad.discover()
    groups: Dict[str, list] = {}
    for i in inputs:
        groups.setdefault(ad.comparison_group(i), []).append(i)
    return groups


def read_all(ad, conn=None, primed: bool = False):
    """Every group read on `conn` (None: no registry connection; `primed`: a frame primed from the adapter's own emissions): ``{group: (records by type,
    status extra)}``, every record validated."""
    from sisteer.registry import validate as VAL
    groups = groups_of(ad)
    if primed:
        conn = X.primed_conn(ad, conn)
    out = {}
    for g in sorted(groups):
        recs = []
        for batch in ad.read(X.ctx(conn), g, groups[g]):
            recs.extend(batch.records)
        bad = [str(v) for v in (VAL.validate_record(r, path="<test>") for r in recs) if isinstance(v, VAL.Malformed)]
        assert not bad, bad[:5]
        bt: Dict[str, list] = {}
        for r in recs:
            bt.setdefault(r["record_type"], []).append(r["payload"])
        out[g] = (bt, ad.status_extra(g))
    return out


def refusal(facts: dict, tag: str, okey: str) -> str:
    hits = [x for x in facts["refusals"] if x.startswith(f"{tag}|{okey} -- ")]
    assert len(hits) == 1, facts["refusals"]
    return hits[0]
