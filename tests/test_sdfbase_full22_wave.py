"""The full22 wave: four served targets, the sdfbase full-corpus
cells + the safety-SI cells, the whole battery under the STACKED lens, on Slurm.

Pins, in the order the wave needs them:
  1. the roster's cell ids are the code's grammar and translate to Petri harness ids exactly as
     registry_adapter does; the view builder's legacy-alias table equals methods';
  2. every config + manifest regenerates byte-identically from the roster + pins (generator drift);
  3. each config is its battery body plus EXACTLY the keys the header names (explicit_cells,
     defenses_dir, out_dir, private_run, attacks.static_max_new_tokens, attacker.prefix_caching /
     quantization; the model pins for the two targets without an sdfbase lineage), with the
     wave-wide battery knobs (forward_cap 150, PAIR 5x3, caps 512 / 4096, BoN 75 on jbb, 100 / 50
     goals) and ONE attacker for every cell of every target;
  4. the manifests load, name exactly the config's cells, the config's target, the twosi bundle /
     behaviours / bloom options, the auditor fallback chain, a frozen serving_impl_commit;
  5. the explicit registry builds over a synthetic staged bank shaped like the view (incl. the
     legacy-alias lora_r4 cell); the view builder stamps a pre-stamp sidecar on the record and
     refuses a sha mismatch / a derived object / an unpinned object;
  6. jailbreak_run's static completion cap (attacks.static_max_new_tokens, fallback = the shared cap);
  7. mt_serving_up: --require-attested-mode, the strict two-report-kind reader, the attested mode in
     the proxy identity block, every canonical model key, written_at;
  8. the sbatch scripts: bash syntax, stacked mode pinned + asserted, no --export=, no --mem=0,
     append mode, hf_env, NCCL, the figure mirror, STOP awareness.
Synthetic data only (placeholder shas, tmp paths).
"""
from __future__ import annotations

import dataclasses
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "sdfbase"))
sys.path.insert(0, str(REPO / "scripts" / "sdfbase" / "full22"))

from sisteer.capeval import manifest as cm  # noqa: E402
from sisteer.capeval.petri import registry_adapter as ra  # noqa: E402
from sisteer.eval.defense_cell import si_of, EXPLICIT_CELLS_KEY, arm_of, sha256_file  # noqa: E402
from sisteer.methods import LEGACY_ARM_ALIASES, object_artifact_name  # noqa: E402

import f22_roster as fr  # noqa: E402
import make_full22_configs as gen  # noqa: E402

ROSTER = yaml.safe_load((REPO / "configs/sdfbase_full22_roster.yaml").read_text())
PINS = json.loads((REPO / "configs/sdfbase_full22_pins.json").read_text())
F22 = REPO / "scripts/sdfbase/full22"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cluster_site import write_site  # noqa: E402  (a two-cluster site config for the printers)
TWOSI = cm.load_petri_spec(REPO / "configs/capeval/manifests/petri_twosi.yaml")


def _views():
    for target, t in ROSTER["targets"].items():
        for view, v in t["views"].items():
            yield target, view, t, v


VIEWS = list(_views())
VIEW_IDS = [f"{t}/{v}" for t, v, _, _ in VIEWS]


def _cfg(v):
    return yaml.safe_load((REPO / v["config"]).read_text())


# --- 1: vocabulary ------------------------------------------------------------------

def test_roster_cells_are_the_codes_grammar_and_translate_like_registry_adapter():
    for target, view, t, v in VIEWS:
        for cid, sid in fr.served_cells(ROSTER, target, view).items():   # view cells + every ENABLED lane
            assert ("__" in cid or cid in gen.BARE_CLASS_B_IDS) and not cid.startswith(("base__", "itdoc__", "si12__", "candsi__"))   # a front-door view may serve a bare Class B id (arditi_allpos)
            assert ra.harness_cell_id(cid) == gen.harness_cell_id(cid), cid
            assert ra.traintest_cell_id(gen.harness_cell_id(cid)) == cid
            if fr.is_prompted(cid):
                # a prompted baseline: the SI as the system prompt, no object, no staged dir (null)
                assert sid is None and arm_of(cid) is None, (cid, sid)
                assert gen.harness_cell_id(cid).startswith("prompted__")
                continue
            assert arm_of(cid) is not None
            # the staged id is <lane>__<cell_id>; the refusal steer's is arditi__<rung>__<cell_id>, its rung stripped from the cell id (the one cell
            # grammar the roster and the launcher share: lib/models.py launcher_cells, demo-import)
            assert sid.split("__", 1)[1] == cid or (sid.startswith("arditi__") and sid.split("__", 2)[2] == cid), (sid, cid)
        # the STAGED subset is exactly the served cells with an object
        assert fr.staged_cells(ROSTER, target, view) == {
            c: s for c, s in fr.served_cells(ROSTER, target, view).items() if s is not None}
    from make_bank_view import CANONICAL_TO_LEGACY_ARM, IDENTITY_SIDECARS
    from sisteer.eval.train_budget import IDENTITY_SIDECARS as TB_IDENTITY_SIDECARS
    assert CANONICAL_TO_LEGACY_ARM == {v: k for k, v in LEGACY_ARM_ALIASES.items()}
    # the builder restates the identity-sidecar set to stay torch-free on its light path;
    # it must be the registry's exact set, or the two would admit different dirs
    assert tuple(IDENTITY_SIDECARS) == tuple(TB_IDENTITY_SIDECARS)


_PAIRED_VIEW = re.compile(r"^s([1-9][0-9]*)p$")


def _admitted_reserve(views: dict, homes: list) -> bool:
    """A staged id named by more than one view is admitted in TWO shapes only: (a) the DOSE LADDER:
    a dose view re-serves bytes another view serves as its own same-pass replicate; (b) a PAIRED view
    `s<s>p` re-serves its SEED view `s<s>`'s r=4 companion and LoRA difference beside the composed psr_paired_write arm — the
    companion is the standup's max_lora_rank anchor, the difference the in-view difference-vs-difference read — and the homes are
    exactly that pair {s<s>, s<s>p}; (c) since 2026-09-23 (the all-models one-seed waves) a
    ONE-SEED view `s<N>` of a non-Qwen target RE-SERVES the `main` view's base-lane objects under the seed grammar (the s1 bank is
    a volume-internal copy of the main bank — one seed per family), so the homes are {main, s<N>}, or {main, s<N>, s<N>p} when
    the paired view of (b) re-serves the same companion. Any other repeat is the collision this guards (the same rule as
    tests/test_sdfbase_full22_newmodels.py's)."""
    names = [vn for vn, _ in homes]
    if any("dose" in views[vn] for vn in names):
        return True
    # keyed on the roster FIELD `paired_of` (the twin marker), never on the name pattern
    # alone: at most one home declares `paired_of`, it names another home, and the name spells its seed (`s<s>p` over `s<s>`)
    paired = [vn for vn in names if "paired_of" in views[vn]]
    if len(paired) > 1:
        return False
    rest = set(names)
    if paired:
        pv = paired[0]; seed = views[pv]["paired_of"]
        if not (_PAIRED_VIEW.match(pv) is not None and pv == f"{seed}p" and seed in rest):
            return False
        rest -= {pv}
        if rest == {seed}:
            return True                                                  # shape (b)
    # shape (c): the one-seed alias — `main` + exactly one `s<N>` view (itself no paired view)
    return (len(rest) == 2 and "main" in rest
            and all(re.fullmatch(r"s[1-9][0-9]*", vn) is not None and "paired_of" not in views[vn] for vn in rest - {"main"}))


def test_roster_names_every_staged_object_exactly_once_per_target():
    for target, t in ROSTER["targets"].items():
        # views + lanes (on or off) + pending_fit / pending_stage / pending_decision (+ a legacy
        # not_served list): every staged id the target names anywhere, exactly once — or a cell
        # could sit in BOTH a view and a pending map (or a lane) unnoticed
        listed = fr.every_staged_id(ROSTER, target)
        # ... with ONE admitted repeat (the DOSE LADDER): a dose view
        # RE-SERVES bytes another view already serves — `d1` / `it_d1` the unscaled itdoc__ / base__ / derived__
        # objects, `d0p5` / `it_d0p5` the halfdose__ / halfdoseit__ copies — as its own same-pass replicate. Such
        # an id may appear more than once ONLY as a view cell, under the SAME config cell id everywhere, and with
        # a dose view among its homes; a repeat from a lane or a pending map is still the collision this guards
        repeated = {sid for sid in listed if listed.count(sid) > 1}
        for sid in repeated:
            homes = [(vn, cid) for vn, v in t["views"].items()
                     for cid, s in (v.get("cells") or {}).items() if s == sid]
            assert len(homes) == listed.count(sid), (target, sid, homes)      # every appearance is a view cell
            assert len({cid for _, cid in homes}) == 1, (target, sid, homes)   # the same cell id in each
            assert _admitted_reserve(t["views"], homes), (target, sid, homes)      # a dose view, or a paired view over its seed view
        assert len(set(listed)) == len(listed) - sum(listed.count(sid) - 1 for sid in repeated), target
        for e in t.get("not_served") or []:
            assert e["status"].endswith("blocked_code") and e["blocker"], e["staged_id"]
        # the candidate-SI arm is wired for every target and OFF (blocker + preconditions
        # on lane_flags.candsi); its six staged ids are pinned on both clusters already
        # ... on every target, llama8b included
        if target in gen.FRONT_DOOR_MODELS:          # a front-door target's block declares no lane and no `main` view (its views are the phase views)
            assert "main" not in t["views"] and all(not fr.lanes_of(ROSTER, target, v) for v in t["views"]), target
        else:
            assert [l for l, _, _ in fr.lanes_of(ROSTER, target, "main")] == ["candsi"]
        assert not fr.lane_enabled(ROSTER, "candsi")
        if target in gen.FRONT_DOOR_MODELS:
            # no `main` / `it` pair to hold: the carried views are PHASE views of the models table (05 --step roster prints one entry
            # per view — u0 first, the steered views once 03/04 stage their objects) and every cell follows the generator's own rule:
            # a prompted baseline maps to null (no object), every other cell to a staged id; the cell count is the table's view minus
            # the `none` anchor the generator adds (roster_view_cells refuses two view cells under one id) — stage-E 05 lane, 2026-09-30
            table_views = gen.FRONT_DOOR_MODELS[target].get("views") or {}
            assert t["views"] and set(t["views"]) <= set(table_views), (target, sorted(t["views"]))
            for vn, v in t["views"].items():
                cells = v.get("cells") or {}
                assert len(cells) == len([c for c in table_views[vn]["cells"] if c != "none"]), (target, vn, sorted(cells))
                for cid, sid in cells.items():
                    assert fr.is_prompted(cid) == (sid is None), (target, vn, cid, sid)
            continue
        # the two lanes whose ids collide live in two views, never one: the it view's OBJECT cells reuse main's ids
        # (same ids, other lane -> its own config + anchor). A prompted baseline (mapped to null: no object, no lane)
        # is outside that rule — the it views carry the matched prompted safety cell since 2026-09-09 (lane F of the
        # sampled re-run) while gemma / qwen main serve the FITTED safety SI instead
        main_ids = set((t["views"]["main"].get("cells") or {}))
        it_cells = t["views"].get("it", {}).get("cells") or {}
        it_obj_ids = {cid for cid, sid in it_cells.items() if sid is not None}
        if it_obj_ids:
            assert it_obj_ids <= main_ids
        for cid, sid in it_cells.items():
            assert sid is not None or fr.is_prompted(cid), (target, cid)


def test_the_derived_cells_are_served_on_both_targets_from_pinned_staged_copies():
    """The sdf-base-steering DERIVED pair (the corpus fit minus its control fit) is
    CODE-servable as of 2026-09-06 and STAGED for both targets the same day (gemma's at
    02:01Z, after the first cut of this file pinned it as pending_stage), so the roster
    serves both copies from the main view, each pinned to a staging-manifest sha that was
    verified on both clusters. Nothing is parked as pending_stage any more, and the two
    targets' derived cells are listed in the same order so their configs line up.

    They are BASE-lane products, so they belong to the `main` view only: the base-lane
    `derived__` object never enters an `it` view. Since 2026-09-13 an INSTRUCT-DIRECT derived
    object exists for qwen35_4b (the batch-64 LoRA subtraction, staged as
    `derived_itdoc64__constv2-mix-full__lora_r4_ce_minus_ctl`), so an `it` view may carry the
    lora_r4_ce_minus_ctl cell through THAT id only — different bytes from main's, pinned on
    both clusters; the distilled PSR residual has no instruct-direct twin and stays off every
    `it` view."""
    derived_arms = ("psr_ce_all_minus_ctl_distilled", "lora_r4_ce_minus_ctl")
    for target in ("qwen35_4b", "gemma4e4b"):
        t = ROSTER["targets"][target]
        main_cells = t["views"]["main"]["cells"]
        assert list(main_cells)[-2:] == [f"constv2-mix-full__{a}" for a in derived_arms], target
        for arm in derived_arms:
            cid = f"constv2-mix-full__{arm}"
            assert main_cells[cid] == f"derived__{cid}"
            pin = PINS[target][f"derived__{cid}"]
            assert len(pin["object_sha256"]) == 64
            assert pin["verified_cluster_a"] and pin["verified_cluster_b"]
            assert pin["sidecar_object_sha256_recorded"] == "match"
            it_sid = (t["views"]["it"].get("cells") or {}).get(cid)
            assert it_sid != f"derived__{cid}", (target, cid)          # the base-lane product, never on the it view
            if it_sid is not None:                                     # the instruct-direct twin: the LoRA subtraction only
                assert arm == "lora_r4_ce_minus_ctl" and it_sid == f"derived_itdoc64__{cid}", (target, cid, it_sid)
                it_pin = PINS[target][it_sid]
                assert it_pin["object_sha256"] != pin["object_sha256"] and it_pin["verified_cluster_a"] and it_pin["verified_cluster_b"]
                assert it_pin["sidecar_object_sha256_recorded"] == "match", (target, it_sid)
            assert f"derived__{cid}" not in {e["staged_id"] for e in (t.get("not_served") or [])}
        assert not t.get("pending_stage")
    # the two derived objects are DIFFERENT bytes per target (each target's own parents)
    for arm in derived_arms:
        sid = f"derived__constv2-mix-full__{arm}"
        assert PINS["qwen35_4b"][sid]["object_sha256"] != PINS["gemma4e4b"][sid]["object_sha256"]
    # every pending_stage entry across the roster is a code-servable cell id (a
    # registered arm), which is what separates it from pending_fit / blocked_code
    from sisteer.methods import ARMS
    for target, t in ROSTER["targets"].items():
        for cid in (t.get("pending_stage") or {}):
            assert arm_of(cid) in ARMS, (target, cid)


def test_a_prompted_baseline_is_served_but_never_staged(tmp_path):
    """The matched prompted safety-SI cell: `prompted__safety_refusal: null`
    in nemotron3nano4b/main. Served (explicit_cells, the manifest as `prompted__safety`, the count
    guard) but never an object: not in the bank view, not in the pins, not in every_staged_id. The
    grammar is enforced in the ONE reader: a null staged id only on a prompted__ id, and vice versa."""
    import copy

    import make_bank_view as mbv

    served = fr.served_cells(ROSTER, "nemotron3nano4b", "main")
    assert served["prompted__safety_refusal"] is None
    assert "prompted__safety_refusal" not in fr.staged_cells(ROSTER, "nemotron3nano4b", "main")
    assert None not in fr.every_staged_id(ROSTER, "nemotron3nano4b")
    assert gen.harness_cell_id("prompted__safety_refusal") == "prompted__safety" == ra.harness_cell_id("prompted__safety_refusal")
    # the SI prefix hash the manifest carries IS the registry's SI bytes (H8; the proxy verifies per request)
    from sisteer.eval.jailbreak import resolve_instruction
    text = resolve_instruction("safety_refusal").text.encode("utf-8")
    assert cm.sha256_hex16(text) == gen.PREFIX_HASHES["safety"]
    spec = cm.load_petri_spec(REPO / ROSTER["targets"]["nemotron3nano4b"]["views"]["main"]["manifest"])
    assert "prompted__safety" in spec.cells and spec.prompted_prefix_hashes["safety"] == gen.PREFIX_HASHES["safety"]
    cfg = _cfg(ROSTER["targets"]["nemotron3nano4b"]["views"]["main"])
    assert "prompted__safety_refusal" in cfg[EXPLICIT_CELLS_KEY]
    # (the registry resolving it as a baseline with the SI text and no object is checked over a synthetic
    # staged bank in tests/test_sdfbase_full22_newmodels.py — an explicit-cell registry needs the objects)
    # the grammar, refused in the one reader
    bad = copy.deepcopy(ROSTER)
    bad["targets"]["nemotron3nano4b"]["views"]["main"]["cells"]["prompted__safety_refusal"] = "base__x"
    with pytest.raises(ValueError, match="never an object"):
        fr.served_cells(bad, "nemotron3nano4b", "main")
    bad = copy.deepcopy(ROSTER)
    bad["targets"]["nemotron3nano4b"]["views"]["main"]["cells"]["constv2-mix-full__psr_ce_all"] = None
    with pytest.raises(ValueError, match="not a prompted__<si> baseline"):
        fr.served_cells(bad, "nemotron3nano4b", "main")
    # the view builder's completeness reference excludes it (a view without an entry for it is COMPLETE)
    view = tmp_path / "cells_view" / "main"
    view.mkdir(parents=True)
    for c in fr.staged_cells(ROSTER, "nemotron3nano4b", "main"):
        (view / c).mkdir()
    assert mbv.view_completeness_problems(view, fr.staged_cells(ROSTER, "nemotron3nano4b", "main")) == []


# --- 2: byte-identity ---------------------------------------------------------------

def test_configs_and_manifests_regenerate_byte_identically():
    rc = subprocess.run([sys.executable, str(REPO / "scripts/sdfbase/make_full22_configs.py"), "--check"],
                        capture_output=True, text=True)
    assert rc.returncode == 0, rc.stdout + rc.stderr


# --- 3: the configs -----------------------------------------------------------------

@pytest.mark.parametrize("target,view,t,v", VIEWS, ids=VIEW_IDS)
def test_config_is_its_body_plus_exactly_the_named_keys(target, view, t, v):
    cfg = _cfg(v)
    body = yaml.safe_load((REPO / t["body_config"]).read_text())
    tag = target + ("" if view == "main" else f"_{view}")
    # none, the object cells, then the prompted baselines (no object) — EXCEPT a steered-wave view (s<N>r), which
    # enumerates no unsteered leg at all: the unsteered model is not re-run; `none` is served implicitly,
    # listed nowhere
    import f22_roster as _fr
    anchor = [] if _fr.steered_wave_view(view, target) else ["none"]
    assert cfg[EXPLICIT_CELLS_KEY] == anchor + list((v.get("cells") or {}))   # a prompted baseline is a `null` cell
    assert cfg[EXPLICIT_CELLS_KEY] == list(_fr.view_cell_universe(ROSTER, target, view))   # the generator's ONE rule for the enumeration (s3r-fix)
    assert ("none" in cfg[EXPLICIT_CELLS_KEY]) is not _fr.steered_wave_view(view, target)   # a steered-wave config carries `none` nowhere
    assert cfg["defenses_dir"] == f"out/full22/{target}/cells_view/{view}"
    assert cfg["out_dir"] == f"out/sdfbase_full22_{tag}" and cfg["private_run"] == f"sdfbase_full22_{tag}"
    for k in ("si_ids", "class_a_arms", "class_b_arms", "stacked_cells"):
        assert k not in cfg
    changed = {k for k in set(body) | set(cfg) if body.get(k) != cfg.get(k)}
    # `bon`: the canonical best-of-N's climb cap + temperature are ADDED by the generator for every target (the base
    # cap bon.max_new_tokens is the body's 512, asserted below — except a target whose caps table WRITES it:
    # llama8b's native 4,096 pass) — the roster's wave-wide values
    expected = {EXPLICIT_CELLS_KEY, "defenses_dir", "out_dir", "private_run", "attacks", "attacker", "bon"}
    if t.get("xstest_subset"):        # the over-refusal control's XSTest half (the llama8b target)
        expected.add("xstest_subset")
        assert cfg["xstest_subset"] == t["xstest_subset"] == "safe"
    if t.get("pins_config"):
        pin_keys = ("model", "model_revision", "layer", "max_len", "teacher_prefix_cache", "pool_manifest")
        pins = yaml.safe_load((REPO / t["pins_config"]).read_text())
        for k in pin_keys:
            assert cfg[k] == pins[k], k
        assert cfg["pool"]["store_dir"] == pins["pool"]["store_dir"]
        assert {k for k in cfg["pool"] if cfg["pool"][k] != body["pool"].get(k)} <= {"store_dir"}
        expected |= {k for k in pin_keys if cfg[k] != body.get(k)}
        if cfg["pool"] != body["pool"]:
            expected.add("pool")
    elif gen.front_door_model(target) is not None:   # the FRONT DOOR: the roster block's served checkpoint (= the models table's) replaces the body's
        expected |= {k for k in ("model", "model_revision") if cfg[k] != body.get(k)}
        assert (cfg["model"], cfg["model_revision"]) == (gen.FRONT_DOOR_MODELS[target]["serve"]["checkpoint"]["hf_repo"], gen.FRONT_DOOR_MODELS[target]["serve"]["checkpoint"]["revision"])
    assert changed == expected, changed ^ expected
    assert (cfg["model"], cfg["model_revision"]) == (t["model"], t["model_revision"])
    # attacks: the static cap ADDED, the multi-turn target cap the ROSTER's (4,096, the canonical
    # instrument since 2026-09-11; the body's 512 is mt1's), everything else the body's
    a, b = dict(cfg["attacks"]), dict(body["attacks"])
    assert a.pop("static_max_new_tokens") == ROSTER["static_max_new_tokens"] == 4096
    assert a.pop("target_max_new_tokens") == ROSTER["target_max_new_tokens"] == 4096
    # the judged over-refusal controls' cap is its OWN key (2026-09-11): the ladder's root, 512, so the
    # multi-turn cap above moved without moving (or re-keying) a single control store
    assert a.pop("controls_max_new_tokens") == ROSTER["controls_max_new_tokens"] == 512
    assert b.pop("target_max_new_tokens") == 512 and "controls_max_new_tokens" not in b
    caps = fr.target_caps(ROSTER, target)
    if caps:   # a target with a caps block (llama8b): the multi-turn cap rides its OWN per-family key; the
               # controls ladder's root is the roster's controls_max_new_tokens (the caps table asserts it, never a second value)
        assert a.pop("multiturn_target_max_new_tokens") == caps["multiturn_target"] and caps["controls_ladder"] == ROSTER["controls_max_new_tokens"]
        # the VIEW's served pins: the target's caps.windows with the view's first-pass overrides laid over them (f22_roster.view_multiturn_windows —
        # the llama8b shortconst view widens PAIR for its ~5.9k-token prefix, 2026-09-12; every other view carries the target's pins unchanged)
        win = fr.view_multiturn_windows(ROSTER, target, view)
        if view != "shortconst":
            assert win == fr.target_windows(ROSTER, target)
        if win:   # the PINNED multi-turn windows ride the config too, PER FAMILY; a target without pins carries neither key
            L, Lr = a.pop("multiturn_max_model_len"), a.pop("multiturn_rerun_max_model_len")
            assert set(L) == set(Lr) == set(win) and all(L[f] == win[f]["multiturn"] for f in win if "multiturn" in win[f])
            assert all(Lr[f] == win[f]["multiturn_rerun"] for f in win if "multiturn_rerun" in win[f]) and all(Lr[f] > L[f] for f in win)
        else:
            assert "multiturn_max_model_len" not in a and "multiturn_rerun_max_model_len" not in a
    assert a == b and a["forward_cap"] == 150
    # attacker: the serving defaults ADDED, the model + snapshot the body's
    at, bt = dict(cfg["attacker"]), dict(body["attacker"])
    assert at.pop("prefix_caching") is True and at.pop("quantization") == "none"
    assert at == bt and at["model"] == "huihui-ai/Huihui-Qwen3.6-27B-abliterated"
    assert at["snapshot"] == "27502c8717fd5a2f8c0c77188c10c243fd4f672e"
    # bon: the climb keys ADDED, everything else — including the BASE cap 512 — the body's; a caps-table target (the
    # sdfbase-llama8b lane) has its base cap WRITTEN from the table instead (the NATIVE 4,096 pass; no 512 store, no climb)
    bon_cfg, bon_body = dict(cfg["bon"]), dict(body["bon"])
    assert bon_cfg.pop("climb_cap") == ROSTER["bon_climb_cap"] == 4096
    assert bon_cfg.pop("climb_decode") == ROSTER["bon_climb_decode"] == "as_drawn"     # the store's sampler (2026-09-17)
    # ... or the VIEW's own `caps.bon` (qwen-base-seed-replicates, 2026-09-15: the s1 / s2 / s3 batch-64 seed-replicate views take
    # the native 4,096 route on a target with NO caps table; make_full22_configs.view_declared_bon_cap) — one reader for the cap
    # the config must carry, `_view_bon_cap`: the view's, else the target's table, else the body's 512
    view_cap = gen.view_declared_bon_cap(ROSTER, target, view)
    want_bon_cap = gen._view_bon_cap(ROSTER, target, view)
    if caps:
        assert view_cap is None and want_bon_cap == caps["bon"]
    if caps or view_cap is not None:
        assert bon_cfg.pop("max_new_tokens") == want_bon_cap and bon_body.pop("max_new_tokens") == 512
    else:
        assert want_bon_cap == 512
    assert bon_cfg == bon_body, "the climb escalates the 512 store; it changes nothing else about BoN"
    assert cfg["bon"]["n_max"] == 75 and cfg["bon"]["goal_subset"] == "jbb" and cfg["bon"]["max_new_tokens"] == want_bon_cap
    if view_cap is not None:   # the per-view cap is a view fact: written on THIS view, echoed in its header, never a dose view
        assert "dose" not in v and view_cap == 4096 and f"bon.max_new_tokens {view_cap}: the view's caps.bon" in (REPO / v["config"]).read_text()
    assert cfg["prereg"]["confirmatory_goals"] == 100 and cfg["prereg"]["pilot_goals"] == 50
    assert cfg["scale"] == 1.0
    from sisteer.eval.attacks.pair_tap import PAIR_DEPTH, PAIR_N_STREAMS
    assert (PAIR_N_STREAMS, PAIR_DEPTH) == (5, 3)


def test_one_attacker_and_one_serving_default_for_every_cell_of_every_target():
    seen = {(json.dumps(_cfg(v)["attacker"], sort_keys=True)) for _, _, _, v in VIEWS}
    assert len(seen) == 1
    branch_defaults = json.loads((REPO / "configs/attacker_serving.json").read_text())
    assert branch_defaults["prefix_caching"] is ROSTER["attacker_prefix_caching"] is True
    assert branch_defaults["quantization"] == ROSTER["attacker_quantization"] == "none"


def test_every_served_cell_is_pinned_and_echoed_in_the_header():
    for target, view, t, v in VIEWS:
        text = (REPO / v["config"]).read_text()
        for cid, sid in (v.get("cells") or {}).items():
            if sid is None:   # a prompted baseline: no object to pin; its SI prefix hash is echoed instead
                hsi = gen.TT_TO_HARNESS_SI[cid.split("__", 1)[1]]
                assert f"{cid:<32} <- (prompted baseline" in text and gen.PREFIX_HASHES[hsi] in text, (target, cid)
                continue
            sha = PINS[target][sid]["object_sha256"]
            assert len(sha) == 64 and sha in text and sid in text, (target, cid)


# --- 4: the manifests ---------------------------------------------------------------

@pytest.mark.parametrize("target,view,t,v", VIEWS, ids=VIEW_IDS)
def test_manifest_names_the_configs_cells_and_the_twosi_audit_world(target, view, t, v):
    spec = cm.load_petri_spec(REPO / v["manifest"])
    cfg = _cfg(v)
    assert [ra.traintest_cell_id(c) for c in spec.cells] == cfg[EXPLICIT_CELLS_KEY]
    assert (spec.target_model, spec.target_revision) == (cfg["model"], cfg["model_revision"])
    assert spec.scenario_bundle_hash == TWOSI.scenario_bundle_hash == "8812ba083bb5473c"
    assert spec.behaviors == TWOSI.behaviors and spec.bloom_options == TWOSI.bloom_options
    assert (spec.auditor_model, spec.judge_model, spec.prompt_provenance) == \
        (TWOSI.auditor_model, TWOSI.judge_model, TWOSI.prompt_provenance)
    # the abliterated-Qwen third leg is retired (never reached in 21,399 rows); the two-SI world's chain is
    # sonnet -> grok. The SEED VIEWS' chains -- auditor AND judge -- are
    # claude-opus-4-8 BEFORE grok-4.6, order pinned, spelled by the roster knobs the generator renders (never a hand edit)
    assert list(TWOSI.auditor_fallback_models) == ["openrouter/x-ai/grok-4.6"]
    d120 = ["anthropic/claude-opus-4-8", "openrouter/x-ai/grok-4.6"]
    if v.get("petri_auditor_fallback_models") is not None:          # a seed view: the roster spells both chains
        assert (target, view) in {("qwen35_4b", s) for s in ("s1", "s2", "s3")}
        assert v["petri_auditor_fallback_models"] == v["petri_judge_fallback_models"] == d120
        assert list(spec.auditor_fallback_models) == list(spec.judge_fallback_models) == d120
    else:                                                             # every other view: the template's one-leg chain, no judge chain
        assert list(spec.auditor_fallback_models) == ["openrouter/x-ai/grok-4.6"]
        assert list(spec.judge_fallback_models) == [] and "petri_judge_fallback_models" not in v
    assert not any("Huihui" in m for m in spec.auditor_fallback_models)
    assert spec.serving_impl_commit == ROSTER["serving_impl_commit"] != cm.PETRI_UNFROZEN
    assert len(spec.serving_impl_commit) == 40
    ms = gen.MODEL_SERVING[target] if target in gen.MODEL_SERVING else gen.front_door_serving(target)   # the front door's row is DERIVED from the models table
    assert (spec.chat_template_hash, spec.tool_parser) == (ms["chat_template_hash"], ms["tool_parser"])
    assert list(spec.serving_launch_flags) == ms["flags"] and "--enable-lora" not in spec.serving_launch_flags
    text_only = {"qwen35_4b", "glm46vflash"} | {k for k in gen.FRONT_DOOR_MODELS if "--limit-mm-per-prompt" in gen.front_door_serving(k)["flags"]}
    if target in text_only:   # text-only serving (the multimodal targets; the MoE test model by its family row): the JSON value survives the YAML round trip verbatim
        i = spec.serving_launch_flags.index("--limit-mm-per-prompt")
        assert json.loads(spec.serving_launch_flags[i + 1]) == {"image": 0, "video": 0}
    else:
        assert "--limit-mm-per-prompt" not in spec.serving_launch_flags
    assert spec.audit == f"petri_full22_{target}" + ("" if view == "main" else f"_{view}")
    # a prompted cell audits under the SI-variant behavior dirs too (<behavior>/PROMPTED_<SI>): those keys
    # join the harness-hash inventory, which is exactly why adding one changes every cell's harness hash
    # (README section 12) — nothing else about the audit world moves
    prompted_sis = [c.split("__", 1)[1] for c in spec.cells if c.startswith("prompted__")]
    variants = [f"{b}/PROMPTED_{si.upper()}" for b in spec.behaviors for si in prompted_sis]
    assert cm.petri_expected_behavior_keys(spec) == sorted(list(spec.behaviors) + variants)


def test_manifests_pin_the_measured_geometry_model_keys():
    from sisteer.capeval.serving import geometry as sg
    for target, _, t, _ in VIEWS:
        if target in gen.FRONT_DOOR_MODELS:      # a front-door target (paper-companion-repo): the ONBOARDED tables, resolved through budget_for
            assert target in sg.ONBOARDED_MODEL_KEYS and sg.ONBOARDED_MODEL_HF_IDS[target] == t["model"] and sg.rows_for(target).onboarded
            # an onboarded key's budget is the PROBE's (03 --phase probe folds the measured per-set rows into the site geometry overlay;
            # tests/test_site_geometry_overlay.py), never a committed row: with no overlay the resolver holds the Llama-certified default and
            # SAYS so (source), and every steered or padded standup refuses that default at its preflight (test_padding_front_door_view_families)
            rb = sg.budget_for(target)
            assert rb.per_model or rb.source == sg.DEFAULT_BUDGET_SOURCE, (target, rb.source)
            assert target not in sg.BUDGET_BY_MODEL, target
            continue
        assert target in sg.MODEL_KEYS and sg.MODEL_HF_IDS[target] == t["model"]
        assert target in sg.BUDGET_BY_MODEL   # a measured budget, never the Llama default
    assert "nemotron3nano4b" in sg.HYBRID_MODEL_KEYS and "qwen35_4b" in sg.HYBRID_MODEL_KEYS


# --- 5: the registry over a view-shaped bank + the view builder --------------------

CANONICAL_TO_LEGACY = {v: k for k, v in LEGACY_ARM_ALIASES.items()}


def _stage(bank: Path, dir_name: str, cell_id: str, *, stamped: bool = True,
           legacy_sidecar: bool = False) -> Path:
    """A minimal LOADABLE torch blob of the arm's real kind + the budget sidecar under
    ``bank/<dir_name>`` (a staged dir may be lane-prefixed; the sidecar names the CELL id, or
    its legacy spelling for ``legacy_sidecar`` — the scale-12 bank's lora_r1_all)."""
    import torch

    arm = arm_of(cell_id)
    kind = ra.ARM_OBJECT_KINDS[arm]
    d = bank / dir_name
    d.mkdir(parents=True, exist_ok=True)
    p = d / object_artifact_name(arm)
    torch.save({"kind": kind, "positions": "all", "layers": [0], "w": torch.zeros(4)}, p)
    si, _ = cell_id.split("__", 1)
    recorded = f"{si}__{CANONICAL_TO_LEGACY[arm]}" if legacy_sidecar else cell_id
    meta = {"cell_id": recorded, "completion_tokens": 512}
    if stamped:
        meta["object_sha256"] = sha256_file(p)
    (d / "train_budget_meta.json").write_text(json.dumps(meta))
    (d / "staged_from.json").write_text(json.dumps({"source": "synthetic"}))
    return p


#: the DERIVED arms: their staged dirs carry derived_meta.json instead of a budget
#: sidecar (they are composed from two fitted parents, never trained)
DERIVED_ARMS = ("psr_ce_all_minus_ctl_distilled", "lora_r4_ce_minus_ctl")


def _stage_derived(bank: Path, dir_name: str, cell_id: str, *,
                   also_budget_sidecar: bool = False, break_sha: bool = False) -> Path:
    """A staged DERIVED cell: a loadable blob of the arm's real kind plus
    derived_meta.json binding the bytes to two complete parent identities (the
    shape train_budget.derived_object_problems validates), and the SAME provenance
    embedded in the blob."""
    import torch

    arm = arm_of(cell_id)
    kind = ra.ARM_OBJECT_KINDS[arm]
    d = bank / dir_name
    d.mkdir(parents=True, exist_ok=True)
    p = d / object_artifact_name(arm)

    def _parent(cell, sha):
        return {"cell_id": cell, "model_key": "synthetic_base", "object_sha256": sha,
                "budget_meta_sha256": "bb" * 32, "model_id": "synthetic-org/model",
                "model_revision": "0" * 40, "positions": "all",
                "response_header_len": None, "target_layers": [0],
                "template_hash": "1111222233334444"}

    si, _ = cell_id.split("__", 1)
    prov = {"kind": kind,
            "parent_pos": _parent(f"{si}__lora_r4_ce", "aa" * 32),
            "parent_neg": _parent("constv2ctl2-full__lora_r4_ce", "cc" * 32)}
    torch.save({"kind": kind, "positions": "all", "layers": [0],
                "w": torch.zeros(4), "provenance": prov}, p)
    meta = {**prov, "derived_id": cell_id,
            "object_sha256": ("00" * 32) if break_sha else sha256_file(p)}
    (d / "derived_meta.json").write_text(json.dumps(meta))
    (d / "staged_from.json").write_text(json.dumps({"source": "synthetic"}))
    if also_budget_sidecar:
        (d / "train_budget_meta.json").write_text(json.dumps(
            {"cell_id": cell_id, "completion_tokens": 512,
             "object_sha256": sha256_file(p)}))
    return p


#: the scale-12 safety-SI cells: two staged LINEAGES of the same record. gemma's copies
#: carry the training run's train_budget_meta.json (pre-stamp, no object_sha256); qwen's
#: carry the HPO-promotion lineage's fit_provenance.json (+ a companion provenance.json and
#: a plain-text COMPLETE note) and NO budget sidecar anywhere upstream. Both are admitted.
SI12_CELLS = ("safety_refusal__psr_kl_all", "safety_refusal__lora_r4")
SI12_LINEAGE = {"gemma4e4b": "train_budget_meta", "qwen35_4b": "fit_provenance"}


def _stage_fit_provenance(bank: Path, dir_name: str, cell_id: str, *,
                          also_budget_sidecar: bool = False) -> Path:
    """A staged HPO-promotion-lineage cell shaped like the real qwen si12 copies: a
    loadable blob of the arm's kind recording its fit leg, fit_provenance.json with the
    stage-3 gate's required keys, provenance.json naming si + arm, a one-line COMPLETE."""
    import torch

    arm = arm_of(cell_id)
    si, _ = cell_id.split("__", 1)
    d = bank / dir_name
    d.mkdir(parents=True, exist_ok=True)
    p = d / object_artifact_name(arm)
    torch.save({"kind": ra.ARM_OBJECT_KINDS[arm], "positions": "response",
                "response_header_len": 7, "layers": [0], "w": torch.zeros(4)}, p)
    (d / "fit_provenance.json").write_text(json.dumps({
        "resolved_model_id": "synthetic-org/model", "resolved_revision": "0" * 40,
        "chat_template_hash": "1111222233334444", "writer": "tune_arm.py",
        "positions_at_fit": "response", "note": "synthetic promotion record"}))
    (d / "provenance.json").write_text(json.dumps({"si": si, "arm": arm, "positions": "response",
                                                 "response_header_len": 7}))
    (d / "COMPLETE").write_text("promoted from out/hpo/synthetic/r1_t00 on 2026-01-01\n")
    (d / "staged_from.json").write_text(json.dumps({"source": "synthetic"}))
    if also_budget_sidecar:
        (d / "train_budget_meta.json").write_text(json.dumps(
            {"cell_id": cell_id, "completion_tokens": 512, "object_sha256": sha256_file(p)}))
    return p


def _stage_any(bank: Path, dir_name: str, cell_id: str, *, target: str, **kw) -> Path:
    """Stage a cell the way it is REALLY staged for ``target``: the derived arms get
    derived_meta.json; the si12 cells get their target's lineage (gemma: the legacy budget
    sidecar, qwen: the fit record); everything else the budget sidecar."""
    if arm_of(cell_id) in DERIVED_ARMS:
        return _stage_derived(bank, dir_name, cell_id)
    if cell_id in SI12_CELLS and SI12_LINEAGE[target] == "fit_provenance":
        return _stage_fit_provenance(bank, dir_name, cell_id)
    return _stage(bank, dir_name, cell_id, **kw)


@pytest.mark.parametrize("target", ["gemma4e4b", "qwen35_4b"], ids=lambda t: t)
def test_explicit_registry_builds_over_a_view_shaped_bank(tmp_path, target):
    """Both main views carry the two DERIVED cells and the two si12 cells, so this covers
    all THREE identity contracts in one registry, each target under its real lineage:
    fitted cells through the budget sidecar (gemma's si12 in its legacy pre-stamp form),
    the qwen si12 cells through fit_provenance.json, derived cells through
    derived_meta.json + the parent verdict."""
    from sisteer.eval.defense_cell import registry_from_explicit_cells
    from sisteer.eval.train_budget import object_identity_kind

    _, _, t, v = next(x for x in VIEWS if x[0] == target and x[1] == "main")
    cfg = _cfg(v)
    bank = tmp_path / "cells_view" / "main"
    # a prompted__<si> baseline (both main views carry prompted__si_constitution since 2026-09-12) serves the SI text
    # and has NO object: nothing is staged for it, and the registry must still build it beside the object cells
    for c in cfg[EXPLICIT_CELLS_KEY][1:]:
        if fr.is_prompted(c):
            continue
        _stage_any(bank, c, c, target=target, legacy_sidecar=(arm_of(c) == "lora_r4"))
    cfg["defenses_dir"] = str(bank)
    cells = registry_from_explicit_cells(cfg, repo_root=REPO)
    assert [c.id for c in cells] == cfg[EXPLICIT_CELLS_KEY]
    prompted = [c for c in cells if fr.is_prompted(c.id)]
    assert [c.id for c in prompted] == ["prompted__si_constitution"] and all(not c.arm and not c.object_path for c in prompted)
    kinds = {c.id: ra.ARM_OBJECT_KINDS[c.arm] for c in cells if c.arm}
    assert kinds["safety_refusal__lora_r4"] == "lora_all" and kinds["constv2-mix-full__psr_ce_all"] == "psr"
    assert kinds["constv2-mix-full__lora_r4_ce_minus_ctl"] == "lora_subtraction"
    assert kinds["constv2-mix-full__psr_ce_all_minus_ctl_distilled"] == "psr"
    # every kind the view serves has a lens adapter — no cell of this wave
    # reaches a standup as an unservable kind
    from sisteer.capeval.serving import lens_adapters as la
    assert set(kinds.values()) <= set(la.registered_kinds())
    # and the identity each cell was admitted under is the one its lineage declares
    ids = {c.id: object_identity_kind(c.object_path) for c in cells if c.object_path}
    want_si12 = "fit_provenance" if SI12_LINEAGE[target] == "fit_provenance" else "staged"
    assert {ids[c] for c in SI12_CELLS} == {want_si12}
    assert {ids[c] for c in ids if arm_of(c) in DERIVED_ARMS} == {"derived"}
    assert {ids[c] for c in ids if c not in SI12_CELLS and arm_of(c) not in DERIVED_ARMS} == {"staged"}


def test_the_registry_refuses_a_derived_cell_it_cannot_prove(tmp_path):
    """The derived branch is a different verdict, not a waiver: a stale
    object/sidecar pair and a dir claiming BOTH identities still refuse."""
    from sisteer.eval.defense_cell import registry_from_explicit_cells

    cid = "constv2-mix-full__lora_r4_ce_minus_ctl"
    for kw, match in (({"break_sha": True}, "object_sha256 mismatch"),
                      ({"also_budget_sidecar": True}, "both")):
        bank = tmp_path / ("x" + match[:4]) / "cells_view" / "main"
        _stage_derived(bank, cid, cid, **kw)
        with pytest.raises(ValueError, match=match):
            registry_from_explicit_cells(
                {EXPLICIT_CELLS_KEY: ["none", cid], "defenses_dir": str(bank)},
                repo_root=REPO)


def test_view_builder_stamps_a_prestamp_sidecar_on_the_record_and_refuses_the_rest(tmp_path):
    import make_bank_view as mbv

    target = "gemma4e4b"
    bank = tmp_path / target
    roster = {"wave": "full22", "bank_root": "x", "targets": {target: {"views": {"main": {"cells": {
        "constv2-mix-full__psr_ce_all": "base__constv2-mix-full__psr_ce_all",
        "safety_refusal__lora_r4": "si12__safety_refusal__lora_r4"}}}}}}
    p1 = _stage(bank / "cells", "base__constv2-mix-full__psr_ce_all",
                "constv2-mix-full__psr_ce_all")                                   # stamped sidecar
    p2 = _stage(bank / "cells", "si12__safety_refusal__lora_r4", "safety_refusal__lora_r4",
                stamped=False, legacy_sidecar=True)                               # pre-stamp (scale-12)
    pins = {target: {"base__constv2-mix-full__psr_ce_all": {"object_sha256": sha256_file(p1)},
                     "si12__safety_refusal__lora_r4": {"object_sha256": sha256_file(p2)}}}
    (tmp_path / "roster.yaml").write_text(yaml.safe_dump(roster))
    (tmp_path / "pins.json").write_text(json.dumps(pins))
    argv = ["--target", target, "--view", "main", "--bank", str(bank),
            "--roster", str(tmp_path / "roster.yaml"), "--pins", str(tmp_path / "pins.json")]
    assert mbv.main(argv) == 0
    view = bank / "cells_view" / "main"
    assert (view / "constv2-mix-full__psr_ce_all").is_symlink()
    stamped = view / "safety_refusal__lora_r4"
    assert stamped.is_dir() and not stamped.is_symlink()
    meta = json.loads((stamped / "train_budget_meta.json").read_text())
    assert meta["object_sha256"] == sha256_file(p2) and "full22_view_stamp" in meta
    assert json.loads((stamped / "full22_view.json").read_text())["mode"] == "stamped"
    assert (stamped / "object.pt").is_symlink() and not (stamped / "COMPLETE").exists()
    assert mbv.main(argv + ["--check"]) == 0
    # the stamped view passes the registry's own staged-cell gate
    from sisteer.eval.train_budget import staged_cell_problems
    assert staged_cell_problems(stamped / "object.pt", "safety_refusal__lora_r4") == []
    # refusals: a sha mismatch is never stamped over; a dir claiming BOTH identities is
    # ambiguous (since 2026-09-06 a derived object IS servable, but only on its own —
    # a fitted cell and a derived object still cannot share a dir); unpinned refuses
    m = json.loads((p1.parent / "train_budget_meta.json").read_text()); m["object_sha256"] = "00" * 32
    (p1.parent / "train_budget_meta.json").write_text(json.dumps(m))
    assert mbv.main(argv) == 2
    (p1.parent / "train_budget_meta.json").write_text(json.dumps({**m, "object_sha256": sha256_file(p1)}))
    (p2.parent / "derived_meta.json").write_text("{}")
    assert mbv.main(argv) == 2
    (p2.parent / "derived_meta.json").unlink()
    (tmp_path / "pins.json").write_text(json.dumps({target: {}}))
    assert mbv.main(argv) == 2


def test_view_builder_admits_a_derived_object_through_the_shared_verdict(tmp_path):
    """A DERIVED staged cell enters the view as a plain symlink: its sidecar is
    derived_meta.json (it was never trained, so there is no budget record to stamp)
    and it is admitted by train_budget.derived_object_problems — the same verdict
    the registry re-runs at build. A stale pair still refuses."""
    import make_bank_view as mbv

    target, cid = "qwen35_4b", "constv2-mix-full__lora_r4_ce_minus_ctl"
    staged = f"derived__{cid}"
    bank = tmp_path / target
    roster = {"wave": "full22", "bank_root": "x",
              "targets": {target: {"views": {"main": {"cells": {cid: staged}}}}}}
    p = _stage_derived(bank / "cells", staged, cid)
    (tmp_path / "roster.yaml").write_text(yaml.safe_dump(roster))
    (tmp_path / "pins.json").write_text(json.dumps(
        {target: {staged: {"object_sha256": sha256_file(p)}}}))
    argv = ["--target", target, "--view", "main", "--bank", str(bank),
            "--roster", str(tmp_path / "roster.yaml"), "--pins", str(tmp_path / "pins.json")]
    assert mbv.main(argv) == 0
    entry = bank / "cells_view" / "main" / cid
    assert entry.is_symlink()                       # nothing to stamp
    assert (entry / "derived_meta.json").is_file()
    assert not (entry / "train_budget_meta.json").exists()
    assert mbv.main(argv + ["--check"]) == 0
    manifest = json.loads((bank / "cells_view" / "main" / "VIEW_MANIFEST.json").read_text())
    rec, = manifest["entries"]
    assert rec["identity"] == "derived" and rec["derived_id"] == cid
    # the view entry passes the registry's own derived verdict
    from sisteer.eval.train_budget import derived_object_problems
    assert derived_object_problems(entry / "object.pt", derived_id=cid) == []
    # a stale object/sidecar pair is refused, never admitted or rewritten
    meta = json.loads((p.parent / "derived_meta.json").read_text())
    (p.parent / "derived_meta.json").write_text(json.dumps({**meta, "object_sha256": "00" * 32}))
    assert mbv.main(argv) == 2


def test_validation_slice_is_the_generated_pair_with_only_the_cells_narrowed(tmp_path):
    """The scratch slice a serving validation stands up must be the COMMITTED
    generated config/manifest with exactly two keys changed (the cell lists) plus a
    scratch namespace — otherwise a validation would be run against a battery that
    is not the wave's. It can also never invent a cell: the staged-id map is copied
    from the roster, so a cell the view does not serve refuses."""
    import make_validation_slice as mvs

    target, view = "qwen35_4b", "main"
    keep = ["constv2-mix-full__psr_ce_all_minus_ctl_distilled",
            "constv2-mix-full__lora_r4_ce_minus_ctl"]
    rc = mvs.main(["--target", target, "--view", view, "--cells", ",".join(keep),
                   "--out", str(tmp_path / "slice")])
    assert rc == 0
    src = yaml.safe_load((REPO / ROSTER["targets"][target]["views"][view]["config"]).read_text())
    got = yaml.safe_load((tmp_path / "slice" / "slice_config.yaml").read_text())
    assert got[EXPLICIT_CELLS_KEY] == ["none"] + keep
    changed = {k for k in set(src) | set(got) if src.get(k) != got.get(k)}
    assert changed == {EXPLICIT_CELLS_KEY, "defenses_dir", "out_dir", "private_run"}, changed
    # the battery itself is the wave's, key for key
    for k in ("attacks", "attacker", "bon", "prereg", "model", "model_revision", "scale"):
        assert got[k] == src[k], k
    # the manifest names the same cells as harness ids, same audit world
    src_man = cm.load_petri_spec(REPO / ROSTER["targets"][target]["views"][view]["manifest"])
    man = cm.load_petri_spec(tmp_path / "slice" / "slice_manifest.yaml")
    assert [ra.traintest_cell_id(c) for c in man.cells] == got[EXPLICIT_CELLS_KEY]
    assert (man.target_model, man.target_revision) == (src_man.target_model,
                                                       src_man.target_revision)
    assert man.scenario_bundle_hash == src_man.scenario_bundle_hash
    assert man.serving_impl_commit == src_man.serving_impl_commit
    # the roster mapping rides verbatim; an unserved cell refuses
    sl = yaml.safe_load((tmp_path / "slice" / "slice_roster.yaml").read_text())
    assert sl["targets"][target]["views"]["validate"]["cells"] == {
        c: ROSTER["targets"][target]["views"][view]["cells"][c] for c in keep}
    with pytest.raises(SystemExit, match="never invents a cell"):
        mvs.main(["--target", target, "--cells", "constv2-mix-full__not_a_cell",
                  "--out", str(tmp_path / "bad")])


# --- 6: the static completion cap -----------------------------------------------------

def test_static_max_new_tokens_knob_and_its_fallback():
    import jailbreak_run as jr

    assert jr._static_max_new_tokens({"attacks": {"target_max_new_tokens": 512, "static_max_new_tokens": 4096}}) == 4096
    assert jr._static_max_new_tokens({"attacks": {"target_max_new_tokens": 512}}) == 512
    assert jr._static_max_new_tokens({"bon": {"max_new_tokens": 256}}) == 256
    src = (REPO / "scripts/jailbreak_run.py").read_text()
    body = src[src.index("def _run_replayable_family"):src.index("# generate", src.index("def _run_replayable_family"))]
    assert "mnt = _static_max_new_tokens(cfg)" in body and "mnt = _target_max_new_tokens(cfg)" not in body
    for _, _, _, v in VIEWS:
        # both caps are 4,096 in every full22 config since 2026-09-11 (the roster's target_max_new_tokens);
        # they remain SEPARATE knobs -- the static cap never falls through to the target's here
        assert jr._static_max_new_tokens(_cfg(v)) == 4096
        assert jr._target_max_new_tokens(_cfg(v)) == ROSTER["target_max_new_tokens"] == 4096


# --- 7: the standup's attestation --------------------------------------------------------

def test_standup_requires_the_attested_stacked_mode_from_both_report_kinds():
    import mt_serving_up as m

    src = (REPO / "scripts/sdfbase/mt_serving_up.py").read_text()
    assert '"--require-attested-mode", choices=["per_row", "stacked"]' in src
    assert "served_reduction_mode_strict(attestation)" in src
    assert "refusing to serve without the authoritative stamp" in src
    assert "**({LENS_REDUCTION_FIELD: lens_mode} if lens_mode is not None else {})" in src
    # the --model-key table is the MERGED one (the paper's eight + geometry.ONBOARDED_MODEL_HF_IDS) since 2026-09-30: a
    # front-door model's first unit died at standup while this pinned `MODEL_HF_IDS as _KEY_MODEL` — the table that lacked it
    assert "_KEY_MODEL = served_hf_id_by_key()" in src and "MODEL_HF_IDS.get(" not in src
    # THREE stamped records: cells_served.json, endpoint.json and
    # serving_config.json — the standup's serving-config provenance (the sha256 of the config + manifest THIS leg
    # serves, written before the engine starts so a leg that dies at launch still records what it would have served)
    assert src.count('"written_at": _utc_now()') == 3
    assert 'SERVING_CONFIG_RECORD_FILE = "serving_config.json"' in src
    sched = {"scheduler_geometry": {"worker_reports": [{"lens_reduction_mode": "stacked"}]}}
    both = {**sched, "worker_reports": [{"lens_reduction_mode": "stacked"}]}
    assert m.served_reduction_mode_strict(both) == ("stacked", "install+scheduler_reports")
    assert m.served_reduction_mode_strict(sched) == ("stacked", "install+scheduler_reports")
    with pytest.raises(ValueError):   # the install and scheduler reports disagree
        m.served_reduction_mode_strict({**sched, "worker_reports": [{"lens_reduction_mode": "per_row"}]})
    with pytest.raises(ValueError):   # an unknown value
        m.served_reduction_mode_strict({"worker_reports": [{"lens_reduction_mode": "rowwise"}]})
    # a pre-landing attestation (no key anywhere) is still recorded absent — and then refused by
    # --require-attested-mode at the call site (source-pinned above), never rewritten as a mode
    assert m.served_reduction_mode_strict({"worker_reports": [{"cells": ["a"]}]}) == (None, "absent")


def test_standup_attests_the_mrope_dispatch_and_refuses_an_unpatched_non_neox_model():
    """The GLM lane (2026-09-08): vLLM 0.25.1's fused triton_mrope kernel rotates every M-RoPE model
    neox-style, and GLM-4V is the one GPT-J-pair family, so an eager (lens) engine mis-rotates it unless
    vllm_patches routed the 2-D case onto forward_native in the worker. The served path used to carry a
    KNOWN GAP (nothing surfaced whether the patch was in play); the standup now asks every worker
    `sisteer_mrope_report` over the RPC route it already uses, records the block, and refuses a report
    that does not prove the rotary right. Certify the branch that RAN, not the hook that was installed."""
    import mt_serving_up as m

    src = (REPO / "scripts/sdfbase/mt_serving_up.py").read_text()
    assert 'MROPE_RPC_METHOD = "sisteer_mrope_report"' in src
    assert "mrope_att = mrope_attest(launch, args.model_key)" in src
    assert '"mrope_attestation": mrope_att,' in src and 'cs["mrope_attestation"] = mrope_att' in src
    assert 'M-RoPE dispatch attestation FAILED' in src
    assert '(out / "mrope_attestation.json").write_text' in src   # the at-STOP report, after traffic
    # the RPC target exists on the SERVED worker extension (lens_worker), the same method the in-process
    # attest_engine asks — a method name pinned here so a rename cannot silently turn the gate into a 404
    lw = (REPO / "src/sisteer/capeval/serving/lens_worker.py").read_text()
    assert "def sisteer_mrope_report(self)" in lw
    vp = (REPO / "src/sisteer/capeval/serving/vllm_patches.py").read_text()
    assert 'llm.collective_rpc("sisteer_mrope_report")' in vp
    good = {"patch_applied": True, "rotary_cls": "MRotaryEmbedding", "is_neox_style": False,
            "mrope_section": [8, 12, 12], "rotary_dim": 64, "dispatch": "forward_cuda",
            "kernel_reachable": True, "routed_native_calls": 0, "needs_patch": True}
    assert m.mrope_attestation_problems([good], "glm46vflash") == []
    assert m.mrope_attestation_problems([good, dict(good)], "glm46vflash") == []
    # the bug itself: the kernel reachable and the patch NOT applied — refused for ANY model key
    bad = {**good, "patch_applied": False}
    assert any("neox-only" in p for p in m.mrope_attestation_problems([bad], "glm46vflash"))
    assert any("neox-only" in p for p in m.mrope_attestation_problems([bad], "someothermodel"))
    # a compiled dispatch is inert-and-unexercised on the served path: refused for GLM, fine elsewhere
    compiled = {**good, "dispatch": "forward_native", "kernel_reachable": False, "needs_patch": False}
    assert any("not reachable" in p for p in m.mrope_attestation_problems([compiled], "glm46vflash"))
    assert m.mrope_attestation_problems([compiled], "qwen35_4b") == []
    # a neox M-RoPE model (the Qwen-VL line) or a non-M-RoPE model needs nothing
    neox = {"patch_applied": True, "rotary_cls": "MRotaryEmbedding", "is_neox_style": True,
            "dispatch": "forward_cuda", "kernel_reachable": True, "needs_patch": False}
    assert m.mrope_attestation_problems([neox], "qwen35_4b") == []
    assert m.mrope_attestation_problems([{"patch_applied": True, "rotary_cls": "RotaryEmbedding",
                                          "is_neox_style": True, "dispatch": "forward_cuda",
                                          "kernel_reachable": True, "needs_patch": False}], "nemotron3nano4b") == []
    # but for a key KNOWN to be non-neox, a neox-looking or wrong-class report is a refusal, not a pass
    assert any("is_neox_style=True" in p for p in m.mrope_attestation_problems([neox], "glm46vflash"))
    assert any("not MRotaryEmbedding" in p
               for p in m.mrope_attestation_problems([{**good, "rotary_cls": "RotaryEmbedding"}], "glm46vflash"))
    # an errored, missing or malformed report never reads as attested
    assert any("errored" in p for p in m.mrope_attestation_problems([{"patch_applied": True, "error": "AttributeError: x"}], "glm46vflash"))
    assert m.mrope_attestation_problems([], "glm46vflash") == ["no sisteer_mrope_report report from any worker"]
    assert any("not a mapping" in p for p in m.mrope_attestation_problems(["oops"], "glm46vflash"))
    # the RPC wrapper turns a transport failure into a refusal block, never a traceback past the standup
    class _Launch:  # noqa: D401 — a stand-in; _post_collective_rpc is patched below
        pass
    import sisteer.capeval.serving.vllm_server as vs
    orig = vs._post_collective_rpc
    try:
        vs._post_collective_rpc = lambda *a, **k: {"results": [good]}
        att = m.mrope_attest(_Launch(), "glm46vflash")
        assert att["ok"] is True and att["reports"] == [good] and att["rpc_method"] == "sisteer_mrope_report"
        vs._post_collective_rpc = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("401 refused"))
        att = m.mrope_attest(_Launch(), "glm46vflash")
        assert att["ok"] is False and "RPC failed: RuntimeError: 401 refused" in att["problems"][0]
        vs._post_collective_rpc = lambda *a, **k: {"results": []}
        assert m.mrope_attest(_Launch(), "glm46vflash")["ok"] is False
    finally:
        vs._post_collective_rpc = orig


_PC_LOOP_HEAD = 'for pc in "${CELLS[@]}"; do'


def _positive_control_loop_body(job: str) -> str:
    """The body of a job's ONE `for pc in "${CELLS[@]}"` loop (between its `do` and its `done`). The attacks leg's
    loop grew MULTI-LINE when the numeric low-dose liveness record was threaded into the smoke's subshell
    (`( export SISTEER_LOW_DOSE_LIVENESS_RECORD=...; f22_positive_control ... ) || ...`), so the
    properties the tests below pin -- the baseline skip, the control call with the leg's tag, the hard failure,
    the break after the first object cell -- are asserted on the body, never on one line."""
    heads = [m.end() for m in re.finditer(re.escape(_PC_LOOP_HEAD), job)]
    assert len(heads) == 1, f"{len(heads)} positive-control loops"
    end = re.compile(r"\bdone\b").search(job, heads[0])
    assert end is not None, "the positive-control loop never closes"
    return job[heads[0]:end.start()]


def _paths_through(body: str) -> list[str]:
    """Every path through the loop body's ONE flat `if ... then ... [elif ...|else ...] fi`: the text before it, one arm,
    the text after it. A body with no conditional is its own single path; an `if` with no `else` has an empty false arm.
    With the liveness record the attacks leg's body forks (a record-carrying cell runs the control under a subshell export, a
    bare one runs it directly), and a call dropped from EITHER arm skips the control for that kind of job."""
    toks = list(re.finditer(r"\b(then|else|fi)\b", body))
    if not toks:
        return [body]
    pre, post = body[:toks[0].start()], body[toks[-1].end():]
    arms = [body[a.end():b.start()] for a, b in zip(toks, toks[1:])]
    if toks[-2].group(1) != "else":
        arms.append("")
    return [pre + arm + post for arm in arms]


def test_common_sh_gates_glm_traffic_on_the_mrope_stamp_and_the_jobs_run_the_positive_control():
    """Job side of the same gate (common.sh f22_standup): the stamp is read back from endpoint.json — never
    the env — and a non-neox M-RoPE target without a passing block refuses traffic; every standup record
    carries the block and the cluster it ran on. And the wave jobs now run the smoke's STEER-IS-LIVE
    positive control on their first steered cell before any driver (the check lives in the jobs rather
    than in a separate smoke)."""
    sh = (F22 / "common.sh").read_text()
    assert "MROPE_ATT=$(" in sh and ".get('mrope_attestation')" in sh
    assert 'case "$KEY" in glm46vflash)' in sh
    assert "REFUSING TRAFFIC: $KEY is a non-neox M-RoPE model and endpoint.json carries no passing mrope_attestation" in sh
    # the record's tail gained the thinking regime + the hop timeout (lane G2, 2026-09-10): mode, verdict (JSON), upstream_timeout_s (JSON)
    assert '"cluster":"%s","gpu_mem_util":%s,"mrope_attestation":%s,"thinking_mode":"%s","thinking_certification_verdict":%s,"upstream_timeout_s":%s,"serving_path_selection":%s,"written_at":"%s"' in sh
    assert '${GPU_MEM_UTIL:+--gpu-mem-util "$GPU_MEM_UTIL"}' in sh   # one number for the preflight and the launch
    assert 'case "$KEY" in glm46vflash) GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}";; esac' in sh   # 78774 @16384 + 78837 @2048: 782,688 KV tokens < 786,432 at 0.9
    assert "GPU_MEM_UTIL=0.92" not in (F22 / "petri_proxy.sbatch").read_text()   # one place, every window
    assert "PASS_WITH_ANCHOR_NONDETERMINISM" in sh and 'echo "$verdict" > "$out.verdict"' in sh
    assert "f22_positive_control(){" in sh and "scripts/sdfbase/full22/smoke_positive_control.py" in sh
    assert "REFUSING TRAFFIC: steer-is-live positive control FAILED" in sh
    for name, tag, first_driver in (("target_static.sbatch", "static", "f22_queue_launch"),
                                    ("target_attacks.sbatch", "attacks", "scripts/jailbreak_run.py generate")):
        job = (F22 / name).read_text()
        # ONE loop, and on its BODY (multi-line on the attacks leg since the liveness record rides a subshell export):
        # EVERY call of the control names "$pc" and the leg's tag and hard-fails the job on the same line (a subshell
        # form closes its `)` first); EVERY path through the body makes such a call (record-carrying cell or bare); and
        # the loop breaks after that first object cell
        body = _positive_control_loop_body(job)
        call = re.compile(rf'f22_positive_control "\$pc" {tag}\s*\)?\s*\|\| \{{ RC=1; exit 1; \}}')
        for line in body.splitlines():
            if "f22_positive_control" in line and not line.strip().startswith("#"):
                assert call.search(line), (name, line)
        for path in _paths_through(body):
            assert call.search(path), (name, path)
        assert body.rstrip().endswith("break;"), (name, body)
        # after the standup, before the first driver launch
        assert job.index("f22_standup attacks") < job.index("f22_positive_control") < job.index(first_driver)
    # the manifest note no longer calls the served M-RoPE state a KNOWN GAP
    gen = (REPO / "scripts/sdfbase/make_full22_configs.py").read_text()
    assert "KNOWN GAP: the served path does not surface MROPE_PATCH_APPLIED" not in gen
    manifest = (REPO / "configs/capeval/manifests/petri_full22_glm46vflash.yaml").read_text()
    assert "ATTESTED at every standup since 2026-09-08: mt_serving_up.mrope_attest asks every worker sisteer_mrope_report" in manifest


def test_positive_control_helper_runs_under_set_u(tmp_path):
    """common.sh f22_positive_control, RUN (not grepped) the way the jobs run it — bash under `set -u` with the
    names it needs — on a stub control script that passes and one that fails. Every first-pass GLM static job
    died one line after a passing standup on `tag: unbound variable` (2026-09-08): a single `local a="$1"
    b="...$a..."` line expands every word before binding any, so the third expansion read an unbound name.
    A parse check (bash -n) cannot see that; only executing the function under -u can."""
    sh = (F22 / "common.sh").read_text()
    start = sh.index("f22_positive_control(){")
    fn = sh[start:sh.index("\n}\n", start) + 3]
    serve = tmp_path / "serve"; serve.mkdir()
    repo = tmp_path / "repo"; (repo / "scripts/sdfbase/full22").mkdir(parents=True)
    stub = repo / "scripts/sdfbase/full22/smoke_positive_control.py"
    stub.write_text("import json,sys\n"
                    "out=sys.argv[sys.argv.index('--out')+1]\n"
                    "json.dump({'checks':[{'check':'greedy_divergence','status':'PASS'}]}, open(out,'w'))\n"
                    "sys.exit(int(open(sys.argv[0]+'.rc').read()))\n")
    rc_file = stub.parent / "smoke_positive_control.py.rc"
    script = ("set -u\n"
              "f22_log(){ echo \"LOG: $*\"; }\n"
              "MPY=%s; REPO_DIR=%s; PROXY_URL=http://127.0.0.1:1/v1; TABLE=%s/table.json; SERVE_DIR=%s\n"
              "%s\n"
              "f22_positive_control \"constv2-mix-full__lora_r4_ce\" static; echo \"rc=$?\"\n"
              % (sys.executable, repo, tmp_path, serve, fn))
    rc_file.write_text("0")
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert "unbound variable" not in r.stderr, r.stderr
    assert "rc=0" in r.stdout, (r.stdout, r.stderr)
    assert "positive control PASS (static, cell constv2-mix-full__lora_r4_ce)" in r.stdout, r.stdout
    assert "positive control check (static): greedy_divergence=PASS" in r.stdout, r.stdout   # one line per check
    assert (serve / "positive_control_static.json").is_file()
    assert (serve / "drivers/positive_control_static.log").is_file()
    rc_file.write_text("3")
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert "rc=1" in r.stdout, (r.stdout, r.stderr)
    assert "REFUSING TRAFFIC: steer-is-live positive control FAILED (static" in r.stdout, r.stdout
    assert (serve / "positive_control_static.json.verdict").read_text().startswith("FAIL")
    # the anchor-only nondeterminism policy (2026-09-08): the ONLY failing check is `determinism` and it
    # names anchor prompts only -> traffic proceeds with the warning recorded; a nondeterministic LENS leg, or any
    # other failing check beside it, still refuses — and the refusal line names the failing checks
    def stub_with(checks):
        stub.write_text("import json,sys\nout=sys.argv[sys.argv.index('--out')+1]\n"
                        f"json.dump({{'checks': {checks!r}}}, open(out,'w'))\nsys.exit(3)\n")
    stub_with([{"check": "salt_isolation", "status": "PASS"},
               {"check": "determinism", "status": "FAIL", "n_prompts": 8,
                "nondeterministic_prompts": {"steered": [], "anchor": ["p1", "p2"], "twin": []}},
               {"check": "steering_bites", "status": "PASS"}, {"check": "greedy_divergence", "status": "PASS"}])
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert "rc=0" in r.stdout and "PASS WITH WARNING" in r.stdout and "anchor_nondeterministic=2/8" in r.stdout, r.stdout
    # every check is its own line, the failing one's detail legible (never truncated behind the passing ones)
    assert 'positive control check (static): determinism=FAIL {"n_prompts": 8, "nondeterministic_prompts": {"steered": [], "anchor": ["p1", "p2"], "twin": []}}' in r.stdout, r.stdout
    assert "positive control check (static): salt_isolation=PASS" in r.stdout
    assert (serve / "positive_control_static.json.verdict").read_text().startswith("PASS_WITH_ANCHOR_NONDETERMINISM")
    stub_with([{"check": "determinism", "status": "FAIL", "n_prompts": 8,
                "nondeterministic_prompts": {"steered": ["p1"], "anchor": ["p1"], "twin": []}},
               {"check": "steering_bites", "status": "PASS"}])
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert "rc=1" in r.stdout and "REFUSING TRAFFIC" in r.stdout and "[FAIL determinism]" in r.stdout, r.stdout
    # the anchor's own noise reaching noise_floor (anchor vs scale-0 twin) is the SAME recorded warning when the
    # twin is byte-stable, determinism failed anchor-only, every steer-is-live check passed and the differing prompts
    # are a minority of the probes (llama it_d0p1 @ 9ffc03f0, cluster B 100540: 2/8 haiku + paperclip)
    stub_with([{"check": "determinism", "status": "FAIL", "n_prompts": 8,
                "nondeterministic_prompts": {"steered": [], "anchor": ["p3", "p4"], "twin": []}},
               {"check": "noise_floor", "status": "FAIL", "n_prompts": 8, "n_anchor_eq_twin": 6,
                "twin": "petri__none__lens0", "differing_prompts": ["p1", "p2"]},
               {"check": "steering_bites", "status": "PASS"}, {"check": "intervention_hash_distinct", "status": "PASS"},
               {"check": "greedy_divergence", "status": "PASS"}])
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert "rc=0" in r.stdout and "PASS WITH WARNING" in r.stdout, r.stdout
    assert "noise_floor_anchor_vs_twin_differing=2/8" in r.stdout and "anchor_nondeterministic=2/8" in r.stdout, r.stdout
    # a MAJORITY of differing prompts is not anchor noise — the routes differ; still refused
    stub_with([{"check": "determinism", "status": "FAIL", "n_prompts": 8,
                "nondeterministic_prompts": {"steered": [], "anchor": ["p3"], "twin": []}},
               {"check": "noise_floor", "status": "FAIL", "n_prompts": 8, "n_anchor_eq_twin": 3,
                "differing_prompts": ["p1", "p2", "p4", "p5", "p6"]},
               {"check": "steering_bites", "status": "PASS"}])
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert "rc=1" in r.stdout and "[FAIL determinism,noise_floor]" in r.stdout, r.stdout
    # noise_floor alone (the anchor deterministic) is a real scale-0 leak — refused as before
    stub_with([{"check": "determinism", "status": "PASS"},
               {"check": "noise_floor", "status": "FAIL", "n_prompts": 8, "n_anchor_eq_twin": 7, "differing_prompts": ["p1"]},
               {"check": "steering_bites", "status": "PASS"}])
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert "rc=1" in r.stdout and "[FAIL noise_floor]" in r.stdout, r.stdout
    stub_with([{"check": "determinism", "status": "FAIL", "n_prompts": 8,
                "nondeterministic_prompts": {"steered": [], "anchor": ["p1"], "twin": []}},
               {"check": "steering_bites", "status": "FAIL"}])
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert "rc=1" in r.stdout and "[FAIL determinism,steering_bites]" in r.stdout, r.stdout


def test_prompted_baselines_ride_the_config_without_an_object():
    """The GLM lane's matched prompted safety-SI cell (2026-09-08): a `prompted__<si>` baseline has no object,
    no staged dir, no pin and no bank-view entry — the proxy serves the bare model behind the SI's H8
    hash-verified system prefix (served_target / bon_served / petri all already do this) — so it lives in the
    config's explicit_cells and the Petri manifest (f22_roster.served_cells, mapped to null) and never in the
    staged subset (staged_cells); the roster grammar refuses a corpus pseudo-instruction as a prompted cell;
    the job side demands no object of it. Derived from the roster, never a literal cell count."""
    v = ROSTER["targets"]["glm46vflash"]["views"]["main"]
    served, staged = fr.served_cells(ROSTER, "glm46vflash", "main"), fr.staged_cells(ROSTER, "glm46vflash", "main")
    assert served["prompted__safety_refusal"] is None and "prompted__safety_refusal" not in staged
    assert list(served)[-1] == "prompted__safety_refusal"            # listed LAST: the lane's leg indices stay as recorded
    assert staged == {c: s for c, s in served.items() if s is not None} and len(served) == len(staged) + 1
    cfg = _cfg(v)
    assert cfg[EXPLICIT_CELLS_KEY] == ["none"] + list(served)
    assert cfg[EXPLICIT_CELLS_KEY][-1] == "prompted__safety_refusal" and cfg[EXPLICIT_CELLS_KEY][0] == "none"
    spec = cm.load_petri_spec(REPO / v["manifest"])
    assert spec.cells[-1] == "prompted__safety"
    assert "for the prompted cell(s) prompted__safety" in (REPO / v["manifest"]).read_text()
    text = (REPO / v["config"]).read_text()
    assert f"#       {'prompted__safety_refusal':<32} <- (prompted baseline: the SI as the system prompt; no object)" in text
    assert "si_prefix sha16 ac3a58f61d53cf24 (verified per request, H8)" in text
    # the roster grammar refuses what the registry would refuse, up front: a corpus pseudo-instruction as a prompt
    bad = {"targets": {"t": {"views": {"main": {"cells": {"prompted__constv2-mix-full": None}}}}}}
    with pytest.raises(ValueError, match="pseudo-instruction"):
        fr.served_cells(bad, "t", "main")
    with pytest.raises(ValueError, match="map it to null"):        # a prompted cell never names an object
        fr.served_cells({"targets": {"t": {"views": {"main": {"cells": {"prompted__safety_refusal": "base__x"}}}}}}, "t", "main")
    with pytest.raises(ValueError, match="needs its staged id"):   # and only a prompted cell may be null
        fr.served_cells({"targets": {"t": {"views": {"main": {"cells": {"safety_refusal__psr": None}}}}}}, "t", "main")
    # the count guard: SERVED (the config after none) vs STAGED (the bank view)
    import roster_cell_count as rcc
    import io, contextlib
    def count(*extra):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = rcc.main(["glm46vflash", "main", *extra])
        assert rc == 0
        return int(buf.getvalue().strip())
    assert count() == len(served) == len(cfg[EXPLICIT_CELLS_KEY]) - 1
    assert count("--staged") == len(staged) == len(served) - 1
    sh = (F22 / "common.sh").read_text()
    assert 'case "$c" in none|prompted__*) continue;; esac' in sh          # no object.pt demanded of a prompted cell
    for name in ("target_static.sbatch", "target_attacks.sbatch"):   # the positive control needs a lens twin: first OBJECT cell
        # the baseline skip stands BEFORE the control call inside the ONE loop -- by position on the loop body, not by
        # adjacency on a line: the attacks leg's body is multi-line since the liveness record rides the smoke's subshell
        body = _positive_control_loop_body((F22 / name).read_text())
        skip, call = 'case "$pc" in none|prompted__*) continue;; esac', 'f22_positive_control "$pc"'
        assert skip in body and call in body, name
        assert body.index(skip) < body.index(call), name
    su = (REPO / "scripts/sdfbase/mt_serving_up.py").read_text()
    assert 'if cid == "none" or cid.startswith("prompted__"):' in su   # _bank_objects: baselines carry no object
    # the no-prompted views render unchanged apart from the note: since 2026-09-12 every main / it view of the four paper
    # targets carries a prompted cell (the constitution-SI baseline joined gemma / qwen / nemotron), so the one view
    # without any is the halfdose view — the gemma main manifest now names its prompted leg instead
    assert "no prompted cell in this wave" in (REPO / ROSTER["targets"]["nemotron3nano4b"]["views"]["halfdose"]["manifest"]).read_text()
    assert "verified per request (H8) for the prompted cell(s) prompted__constitution" in (REPO / ROSTER["targets"]["gemma4e4b"]["views"]["main"]["manifest"]).read_text()


def test_standup_asserts_the_attested_scheduler_geometry_next_to_the_mode():
    """The 2026-09-06 geometry change: the resolved max_num_seqs x max_model_len must be the one
    the workers ATTEST (vLLM's live scheduler config), recorded beside lens_reduction_mode — a
    silently clamped width is refused, never inferred from throughput."""
    import mt_serving_up as m

    rep = lambda s, l: {"max_num_seqs": s, "max_model_len": l, "lens_reduction_mode": "stacked"}  # noqa: E731
    att = lambda *reps: {"scheduler_geometry": {"worker_reports": list(reps)}}  # noqa: E731
    assert m.attested_scheduler_geometry(att(rep(148, 4864))) == (148, 4864)
    assert m.attested_scheduler_geometry(att(rep(256, 2048), rep(256, 2048))) == (256, 2048)
    assert m.attested_scheduler_geometry(att({"lens_reduction_mode": "stacked"})) is None
    assert m.attested_scheduler_geometry({}) is None and m.attested_scheduler_geometry([]) is None
    with pytest.raises(ValueError):
        m.attested_scheduler_geometry(att(rep(148, 4864), rep(128, 4864)))
    src = (REPO / "scripts/sdfbase/mt_serving_up.py").read_text()
    assert "a silently clamped geometry; refusing" in src
    assert "the resolved geometry must be attested, not assumed" in src
    assert src.count('"attested_max_num_seqs": att_geom[0] if att_geom else None') == 1   # endpoint.json
    assert 'cs["attested_max_num_seqs"] = att_geom[0] if att_geom else None' in src      # cells_served.json


def test_controls_standup_caps_logprobs_at_the_drivers_request():
    """447405 (2026-09-07): every controls driver died at MMLU — "Requested sample logprobs of 64, which
    is greater than max allowed: 20" — AFTER the fit-provenance step had passed. The driver scores by
    the top-MMLU_TOP_LOGPROBS logprobs, the engine caps requests at --max-logprobs (default 20), and
    this lineage's standup never set it (main's capctrl launcher splices 64). One number, one source:
    the sbatch imports the driver's constant and hands it to BOTH the server and the driver; the attack
    standups, which never ask for logprobs, keep a byte-identical launch command."""
    import mt_serving_up as m

    assert m.launch_cmd_extras("0.9") == ("--gpu-memory-utilization", "0.9")
    assert m.launch_cmd_extras("0.9", 64) == ("--gpu-memory-utilization", "0.9", "--max-logprobs", "64")
    assert m.launch_cmd_extras("0.9", None) == m.launch_cmd_extras("0.9")
    with pytest.raises(ValueError):
        m.launch_cmd_extras("0.9", 0)
    src = (REPO / "scripts/sdfbase/mt_serving_up.py").read_text()
    assert 'ap.add_argument("--max-logprobs", type=int, default=None' in src
    assert "cmd=tuple(launch.cmd) + launch_cmd_extras(args.gpu_mem_util, args.max_logprobs, args.enable_log_requests)" in src
    assert src.count('"max_logprobs": args.max_logprobs,') == 2        # recorded in cells_served.json + endpoint.json
    common = (F22 / "common.sh").read_text()
    assert 'gLP="${7:-}"' in common and '${gLP:+--max-logprobs "$gLP"}' in common
    sb = (F22 / "target_static.sbatch").read_text()
    assert "print(int(m.MMLU_TOP_LOGPROBS))" in sb and "scripts/served_capability_run.py)" in sb
    # the controls window is the table's 2048 unless the VIEW sets one (llama8b/fullconst 45,056 for every standup; llama8b/shortconst
    # its PER-LEG controls window 6,656 — roster_view.py `window controls`, 2026-09-12) — or, for a gsm8k-bearing TABLE view, the
    # roster's controls_window.gsm8k (roster_view.py `window gsm8k`, 2026-09-14), held to the derived minimum by
    # f22_gsm8k_window_check before the standup
    assert 'CTL_L="${VIEW_WINDOW_CONTROLS:-${VIEW_WINDOW_GSM8K:-2048}}"' in sb
    assert 'f22_gsm8k_window_check "$CTL_L" "$VIEW_CONTROLS"' in sb
    assert 'f22_standup attacks "$SERVE_DIR/controls" "" capability:short "$CTL_L" "" "$CTL_TOP_LOGPROBS" ||' in sb
    assert 'VIEW_WINDOW_CONTROLS=$(python3 scripts/sdfbase/full22/roster_view.py "$KEY" "$VIEW" window controls --roster "$F22_ROSTER")' in sb
    assert 'VIEW_WINDOW_GSM8K=$(python3 scripts/sdfbase/full22/roster_view.py "$KEY" "$VIEW" window gsm8k --roster "$F22_ROSTER")' in sb
    assert '--top-logprobs "$CTL_TOP_LOGPROBS"' in sb
    assert 'f22_standup attacks "$SERVE_DIR/static" "" families_served:short ${VIEW_WINDOW_STATIC:+"$VIEW_WINDOW_STATIC"} ||' in sb   # no 7th arg: engine default; the view's static window when set
    assert 'VIEW_WINDOW_STATIC=$(python3 scripts/sdfbase/full22/roster_view.py "$KEY" "$VIEW" window static --roster "$F22_ROSTER")' in sb
    # the driver's constant is the one the sbatch imports; a rename there must land here too
    drv = (REPO / "scripts/served_capability_run.py").read_text()
    assert "MMLU_TOP_LOGPROBS = 64" in drv and 'g.add_argument("--top-logprobs", type=int, default=MMLU_TOP_LOGPROBS' in drv


def test_static_job_runs_the_two_geometry_phases_and_fills_them():
    src = (F22 / "target_static.sbatch").read_text()
    common = (F22 / "common.sh").read_text()
    # static + BoN: the measured families_served:short window (4864 = p99 static prompt 762 + the 4,096 cap)
    # x the model's budget S; controls: the 2048 window with S DERIVED from the model budget (the launcher's
    # strict identity refused an explicit `2048 256`: identity_seqs(2048) = 288 for gemma — smoke 421950);
    # one standup per geometry, stopped between phases
    assert 'f22_standup attacks "$SERVE_DIR/static" "" families_served:short' in src
    assert 'f22_standup attacks "$SERVE_DIR/controls" "" capability:short "$CTL_L" "" "$CTL_TOP_LOGPROBS" ||' in src
    assert "capability:short 2048 256" not in src
    # the view's --controls list and legs come from the roster (roster_view.py), never a literal
    assert '--controls "$VIEW_CONTROLS"' in src and 'roster_view.py "$KEY" "$VIEW" legs' in src
    assert src.count("f22_standup_stop") == 2
    assert "bon_served_run.py" in src and "--max-concurrency" in src
    assert "DRIVER_CAP=96" in src and 'MAX_DRIVERS=$(( TARGET_S < DRIVER_CAP ? TARGET_S : DRIVER_CAP ))' in src
    # BoN runs to completion BEFORE the static families (the interleaved form starved the static queue at 8 drivers)
    assert "phase STATIC/bon done" in src and "MAX_DRIVERS - BON_CONC" not in src
    assert src.index("phase STATIC/bon done") < src.index('for fam in "${STATIC_FAMS[@]}"')
    # the proxy releases its GPU when the driver heartbeat goes stale or never arrives; the driver writes it
    proxy = (F22 / "petri_proxy.sbatch").read_text(); drive = (F22 / "petri_drive_full22.sh").read_text()
    assert "HEARTBEAT_STALE_S=1800" in proxy and "HEARTBEAT_GRACE_S=5400" in proxy and "driver_heartbeat" in proxy
    assert "releasing the GPU" in proxy and 'touch "$OUT/STOP"; break' in proxy
    assert "touch $REMOTE_OUT/driver_heartbeat" in drive and "sleep 300" in drive
    # ... and a release with no driver leaves with its NAMED non-zero code, never COMPLETED 0:0 (demo-import 2026-10-11; the behavioural
    # case is test_petri_proxy_releases_with_no_driver_exit_their_named_codes_never_completed): the code reaches the RESULT line's rc=
    assert "RELEASE_RC=$F22_EXIT_PROXY_NO_DRIVER; RELEASE_NAME=PROXY_NO_DRIVER" in proxy and "RELEASE_RC=$F22_EXIT_PROXY_DRIVER_LOST; RELEASE_NAME=PROXY_DRIVER_LOST" in proxy
    assert proxy.index("RC=$RELEASE_RC") < proxy.index('f22_log "petri proxy exited rc=$RC"') < proxy.index("\nexit $RC") and 'f22_result "petri proxy $TAG rc=${RC}"' in proxy
    # ... and that ssh is BOUNDED (timeout 90 + ServerAlive): a connect timeout bounds nothing once the session is up; a
    # half-up login node held one heartbeat ssh open past the proxies' 30-min watchdog and three live proxies released
    # their GPUs at once (2026-09-09 outage; the fix is the peers' 6add2b88 / b3096666, applied here by hand)
    assert ('timeout 90 ssh -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=2 -o BatchMode=yes '
            '"$HOST" "touch $REMOTE_OUT/driver_heartbeat"') in drive
    assert "f22_queue_launch" in src and "--concurrency" in src
    assert '"budget_fraction":%s' in common and "BUDGET_BY_MODEL" in common   # the footprint is recorded, never inferred
    assert 'NSHARDS="${5:-9}"' in src
    # the standup helper reads the attested width back and refuses a clamped / unattested one
    for m in ("attested_max_num_seqs", "attested_max_model_len", "a clamped or unattested geometry",
              "requested max_num_seqs $gS but the server attests",
              "attested_cudagraph_full_capture_max", "the largest FULL CUDA-graph capture size"):
        assert m in common, m
    from sisteer.capeval.serving import geometry as sg
    fs = sg.geometry_for("families_served", task="short")
    assert fs.max_model_len == 4864 and fs.basis.requested_max_tokens == 4096 and fs.basis.p99_prompt == 762
    for key, want in (("gemma4e4b", 121), ("nemotron3nano4b", 161), ("glm46vflash", 161)):
        assert sg.BUDGET_BY_MODEL[key].seqs_at(4864) == want, key
        assert 2048 * 256 <= sg.BUDGET_BY_MODEL[key].batched_token_budget
        assert sg.BUDGET_BY_MODEL[key].seqs_at(2048) >= 256
    # qwen35_4b: the swept 720,896 row again — its 2026-09-06 halved shadow (360,448;
    # widths 74 at 4864 and 176 at 2048) was lifted 2026-09-11 once the AlphaSteer re-bake
    # ladder ran six standups at exactly 148 x 4864 on the lens stack — so the derived
    # widths are 148 at 4864 and the hybrid cap (256) at 2048, the widths the battery ran on.
    q = sg.BUDGET_BY_MODEL["qwen35_4b"]
    assert q.batched_token_budget == 741_888 and q.source == sg.MEASURED_BUDGET_SOURCE and not q.certified   # 42 x 17,664, measured 2026-09-13
    assert q.seqs_at(4864) == 152 and q.seqs_at(2048) == 256 and q.hybrid
    # the derived controls width per model: identity at 2048 (gemma 288)
    assert sg.BUDGET_BY_MODEL["gemma4e4b"].seqs_at(2048) == 288
    # the multi-turn window is untouched (attacker-bound: no context traded for width)
    att = (F22 / "target_attacks.sbatch").read_text()
    # TWO window readers, told apart by the config's keys (the 2026-09-11 reconciliation). (b) the canonical instrument:
    # the multi-turn window is EXPLICIT per attack since 2026-09-11 (geometry.multiturn_window(attacks): crescendo 49,152 /
    # pair 16,384, the measured maxima on complete stores), never the table's multiturn_attacks 8,704; the standup is told
    # the drivers' cap so a table window can never serve it silently
    assert 'f22_standup attacks "$SERVE_DIR" "" "" "$GEOM_L" "" || { RC=1; exit 1; }' in att and "families_served" not in att
    assert "multiturn_window(sys.argv[1])" in att and '--driver-max-tokens "$CAP_TOKENS"' in att
    assert '--target-max-new-tokens "$CAP_TOKENS" --target-timeout-s "$CLIENT_TIMEOUT_S"' in att
    assert 'f22_standup attacks "$SERVE_DIR" "" || { RC=1; exit 1; }' not in att         # the table-window call is gone
    # (a) a config that names its own multi-turn cap (the caps table): the window DERIVED from it
    # (geometry.multiturn_window(cap)) or the roster's PIN over it (attacks.multiturn_max_model_len) — PER FAMILY:
    # the family's cap and pin, a combined job at the MAX over its families, the load check told the families
    assert 'f22_standup attacks "$SERVE_DIR" "" ${MT_L:+multiturn_attacks} ${MT_L:+"$MT_L"} || { RC=1; exit 1; }' in att
    assert "multiturn_target_max_new_tokens" in att and "multiturn_window" in att
    assert 'a.get("multiturn_max_model_len")' in att and "cap.get(f) if isinstance(cap, dict) else cap" in att
    assert 'print(f"MAX {max(Ls.values())}")' in att and 'export F22_LOAD_CHECK_ATTACK="$ATTACKS"' in att
    assert "a family the config does not name cannot run" in att
    common = (F22 / "common.sh").read_text()
    assert common.count('${F22_LOAD_CHECK_ATTACK:+--attack "$F22_LOAD_CHECK_ATTACK"}') == 2   # the preflight and the post-standup load check


def test_bon_fanout_driver_uses_the_wave_seams():
    import bon_served_run as br

    ap = br.build_parser()
    ns = ap.parse_args(["--config", "c.yaml", "--defense", "none", "--private-root", "/r", "--proxy-url", "u",
                        "--routing-table", "t.json", "--max-model-len", "4864", "--max-concurrency", "21"])
    assert ns.require_lens_mode == "stacked" and ns.max_concurrency == 21 and ns.trial == 0
    assert ns.limit_goals == 0   # the smoke-only head-cap is OFF unless asked for (a wave run is never capped)
    src = (F22 / "bon_served_run.py").read_text()
    assert '"smoke": bool(args.limit_goals)' in src   # a capped run stamps itself as a smoke
    assert 'tsr._eval_goals(cfg, confirmatory=args.confirmatory, attack="bon")' in src   # the jbb subset
    assert "jr._bon_defense_dir(cfg, root, args.defense)" in src                          # the aggregate's dir
    assert "bs.bon_generate_served(" in src and "steering_probe=steering_probe" in src
    assert "ServedDefense.over_client(client" in src
    # native-serving-default: the ENGINE compare through the one helper (lens:<mode> derived from --require-lens-mode
    # by default, so this invocation is byte-identical; lora_native asserts the native identity block)
    assert 'ident.get("lens_reduction_mode") != args.require_lens_mode' not in src
    assert "_served_engine_problem(ident, expect=require_engine)" in src and '"--require-engine"' in src
    assert br.require_engine_of(None, "stacked", tag="x") == "lens:stacked"
    assert br.require_engine_of("lora_native", "stacked", tag="x") == "lora_native"
    with pytest.raises(SystemExit):
        br.require_engine_of("lens:per_row", "stacked", tag="x")            # two engines named, disagreeing
    assert ns.require_engine is None                                        # absent by default: the lens mode governs


# --- 8: the sbatch layout --------------------------------------------------------------------

SBATCH = sorted(F22.glob("*.sbatch"))


def test_sbatch_layout_is_complete():
    assert {p.name for p in SBATCH} == {"attacker_pool.sbatch", "target_attacks.sbatch",
                                        "target_static.sbatch", "petri_proxy.sbatch", "smoke_full22.sbatch",
                                        "derived_residual.sbatch",   # the Slurm port of the residual distill
                                        "target_attacks_goalset.sbatch",   # the goal-set variant (the pin lagged it)
                                        "bon_climb.sbatch",   # the canonical best-of-N: the 512 store + the 4,096 climb
                                        "mmlu_full.sbatch",   # mmlu-full-1531: the MMLU read over the whole 1,531-item pinned pool (one job per view)
                                        "reclose_stores.sbatch"}   # closer-fix-2: the CPU-only re-run of a finished eval job's store closer (exit 75 → reclose)
    for name in ("common.sh", "submit_full22.sh", "petri_drive_full22.sh", "make_bank_view.py",
                 "bon_served_run.py", "smoke_positive_control.py", "roster_cell_count.py", "README.md",
                 "f22_roster.py", "submit_newmodels.sh", "submit_newmodels.py"):
        assert (F22 / name).is_file(), name


def test_a_short_view_is_an_error_not_a_log_line(tmp_path, monkeypatch):
    """THE SHORT-VIEW GUARD (2026-09-07). The first launch attempt built 7 cells for one target and 5
    for the other, both exit 0: the wave would have produced a results table with no safety-SI arm on
    one of two targets. Both halves of the guard are checked here — the builder's own completeness
    check, and the job-side cross-check that the generated config and the built view agree with the
    ROSTER's count (the number the guard treats as the reference)."""
    monkeypatch.setenv("SISTEER_CONFIG", str(write_site(tmp_path / "cluster-site")))   # the printer reads each cluster's host from the site
    import make_bank_view as mbv

    view = tmp_path / "cells_view" / "main"
    view.mkdir(parents=True)
    cells = {"a__psr_ce_all": "base__a", "b__lora_r4": "base__b"}
    assert mbv.view_completeness_problems(view, cells)              # empty view: SHORT
    (view / "a__psr_ce_all").mkdir()
    probs = mbv.view_completeness_problems(view, cells)
    assert probs and "SHORT" in probs[0] and "b__lora_r4" in probs[0]
    (view / "b__lora_r4").mkdir()
    assert mbv.view_completeness_problems(view, cells) == []        # complete
    (view / "VIEW_MANIFEST.json").write_text("{}")                  # bookkeeping files are not cells
    (view / "full22_derived_quality.json").write_text("{}")
    assert mbv.view_completeness_problems(view, cells) == []
    (view / "c__stale").mkdir()                                     # a stale entry refuses too
    probs = mbv.view_completeness_problems(view, cells)
    assert probs and "does not name" in probs[0] and "c__stale" in probs[0]

    # the reference the job-side guard reads, and the config it compares against
    import subprocess
    for target, vw in (("gemma4e4b", "main"), ("gemma4e4b", "it"),
                       ("qwen35_4b", "main"), ("qwen35_4b", "it"),
                       ("nemotron3nano4b", "main"), ("glm46vflash", "main")):
        rc = subprocess.run([sys.executable, str(F22 / "roster_cell_count.py"), target, vw],
                            capture_output=True, text=True)
        assert rc.returncode == 0, rc.stderr
        want = int(rc.stdout.strip())
        assert want == len(ROSTER["targets"][target]["views"][vw]["cells"])
        cfg = _cfg(ROSTER["targets"][target]["views"][vw])
        assert len(cfg[EXPLICIT_CELLS_KEY]) - 1 == want, (target, vw)   # minus the `none` anchor
        # the VIEW's reference is the staged count: served minus the prompted baselines (no object)
        rs = subprocess.run([sys.executable, str(F22 / "roster_cell_count.py"), target, vw, "--staged"],
                            capture_output=True, text=True)
        assert rs.returncode == 0, rs.stderr
        n_prompted = sum(1 for c in cfg[EXPLICIT_CELLS_KEY] if fr.is_prompted(c))
        assert int(rs.stdout.strip()) == want - n_prompted == len(fr.staged_cells(ROSTER, target, vw)), (target, vw)
    # the job-side guard reads BOTH counts and skips the object-less cells in its presence loop
    common = (F22 / "common.sh").read_text()
    assert 'roster_cell_count.py "$KEY" "$VIEW" --staged' in common
    assert '[ "$n_view" -eq "$want_staged" ]' in common and '[ "$have" -eq "$want" ]' in common
    assert 'case "$c" in none|prompted__*) continue;; esac' in common
    rc = subprocess.run([sys.executable, str(F22 / "roster_cell_count.py"), "nope", "main"],
                        capture_output=True, text=True)
    assert rc.returncode == 2                                        # an unknown view is an error
    common = (F22 / "common.sh").read_text()
    assert "roster_cell_count.py" in common and "refusing to run a narrowed wave" in common
    assert "a short (or stale) view is an error" in common
    # the bank view is SHARED state across every job of a (target, view): the read-only --check is the
    # common path and the BUILD is serialized on a per-view lock (job 423389 died at 3m30s when two
    # static jobs on one node rebuilt gemma/main concurrently — one replaced a stamped entry mid-read)
    assert "--check --target" in common and "flock -w 900 9" in common
    assert '.view.lock' in common
    # and the build itself is out of place + renamed in, so even a reader inside the lock window sees
    # the OLD complete entry or the NEW one, never a half-populated directory
    mbv_src = (F22 / "make_bank_view.py").read_text()
    assert "_populate_entry(rec, src, tmp, view_dir)" in mbv_src and "os.rename(tmp, dst)" in mbv_src
    assert "os.rename(old_aside, dst)" in mbv_src            # a failed swap restores the old entry
    # the submit printer makes building the views a step of its own, before any sbatch
    rc = subprocess.run(["bash", str(F22 / "submit_full22.sh"), "ab" * 20], capture_output=True, text=True)
    assert rc.returncode == 0, rc.stderr
    head = rc.stdout.split("# --- 1.")[0]
    assert "make_bank_view.py --check" in head and "BUILD THE BANK VIEWS ONCE" in head
    assert head.index("make_bank_view.py") < rc.stdout.index("sbatch")   # views first, then jobs
    # the standup refuses a device a neighbour is holding, with the numbers (job 74031)
    assert "REFUSING TO STAND UP" in common and "memory.total,memory.free" in common
    assert "DIRTY/SHARED DEVICE" in common and "do not shrink the geometry" in common


def test_the_view_entry_swap_is_out_of_place(tmp_path):
    """The rename swap, exercised: a stamped entry rebuilt over an existing one keeps its relative
    symlinks resolving from the FINAL path (they are assembled in a tmp dir that gets renamed), and
    an existing entry is replaced rather than emptied in place."""
    import make_bank_view as mbv

    bank = tmp_path / "bank"
    src = bank / "cells" / "si12__x"
    src.mkdir(parents=True)
    (src / "object.pt").write_bytes(b"weights")
    (src / "staged_from.json").write_text("{}")
    (src / "train_budget_meta.json").write_text(json.dumps({"cell_id": "x__psr_kl_all"}))
    view = bank / "cells_view" / "main"
    view.mkdir(parents=True)
    rec = {"cell_id": "x__psr_kl_all", "staged_id": "si12__x", "source": str(src), "mode": "stamped",
           "object_sha256": "a" * 64, "stamp_reason": "test"}
    (view / "x__psr_kl_all").mkdir()                      # a pre-existing entry, replaced not emptied
    (view / "x__psr_kl_all" / "stale").write_text("old")
    mbv.write_entry(rec, view)
    dst = view / "x__psr_kl_all"
    assert not (dst / "stale").exists()
    assert (dst / "object.pt").is_symlink() and (dst / "object.pt").read_bytes() == b"weights"
    assert json.loads((dst / "train_budget_meta.json").read_text())["object_sha256"] == "a" * 64
    assert not any(p.name.startswith((".tmp-", ".old-")) for p in view.iterdir())   # no litter


def test_the_derived_cells_carry_their_fit_quality(tmp_path):
    """The two targets' distilled residuals are NOT of comparable quality (README 4a), so every bank
    view emits full22_derived_quality.json — held-out loss, best epoch / early stop, gate liveness
    min / mean / count below 0.5 — beside the manifest, where the analysis cannot miss it."""
    import make_bank_view as mbv

    bank = tmp_path / "bank"
    d = bank / "cells" / "derived__x"
    d.mkdir(parents=True)
    (d / "derived_meta.json").write_text(json.dumps({
        "kind": "psr", "recovery_verdict": "recovered",
        "trainer": {"best_epoch": 48, "epochs_run": 53, "epochs": 60, "stopped_early": True,
                    "stop_reason": "early stop: no held-out improvement for 5 epochs"},
        "gate_liveness": {str(i): (0.1 if i < 11 else 0.9) for i in range(42)}}))
    (d / "report.json").write_text(json.dumps({"best_heldout_loss": 6.7568, "heldout_init": 18.5734}))
    q = mbv.derived_quality([{"cell_id": "x__psr_ce_all_minus_ctl_distilled", "staged_id": "derived__x"}], bank)
    got = q["x__psr_ce_all_minus_ctl_distilled"]
    assert got["best_heldout_loss"] == 6.7568 and got["best_epoch"] == 48 and got["stopped_early"] is True
    assert got["liveness_n_layers"] == 42 and got["liveness_n_below_0p5"] == 11
    assert got["liveness_min"] == 0.1 and got["quality_source"] == "report.json + derived_meta.json"
    # a structural derived object (the LoRA difference) has no fit record and says so, never a fake zero
    d2 = bank / "cells" / "derived__lora"
    d2.mkdir(parents=True)
    (d2 / "derived_meta.json").write_text(json.dumps({"kind": "lora_subtraction"}))
    q2 = mbv.derived_quality([{"cell_id": "x__lora_r4_ce_minus_ctl", "staged_id": "derived__lora"}], bank)
    got2 = q2["x__lora_r4_ce_minus_ctl"]
    assert got2["kind"] == "lora_subtraction" and got2["best_heldout_loss"] is None
    assert got2["liveness_n_layers"] is None if "liveness_n_layers" in got2 else True
    # a cell with no derived_meta.json (every trained cell) contributes nothing
    assert mbv.derived_quality([{"cell_id": "trained", "staged_id": "base__t"}], bank) == {}


@pytest.mark.parametrize("script", SBATCH + [F22 / "common.sh", F22 / "submit_full22.sh", F22 / "petri_drive_full22.sh",
                                             F22 / "submit_newmodels.sh"],
                         ids=lambda p: p.name)
def test_scripts_parse_and_carry_the_wave_rules(script):
    rc = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert rc.returncode == 0, rc.stderr
    src = script.read_text()
    for line in src.splitlines():
        if "--export=" in line:
            assert "never" in line.lower(), line  # only the rule's own statement may name the flag
    assert "--mem=0" not in src
    if script.suffix == ".sbatch":
        common = (F22 / "common.sh").read_text()
        # the smoke runs on the dev partition/QoS (one GPU, outside the high budget); every wave job is general
        partition = ("#SBATCH --partition=dev,overflow" if script.name == "smoke_full22.sbatch"
                     else "#SBATCH --partition=general")
        for m in ("#SBATCH --open-mode=append", "#SBATCH --no-requeue", "#SBATCH --signal=B:TERM@120",
                  partition, "#SBATCH --time="):
            assert m in src, m
        if script.name == "smoke_full22.sbatch":
            assert "#SBATCH --qos=dev" in src and "#SBATCH --job-name=D_" in src   # the dev reaper's D_ prefix
            assert "--limit-goals" in src and "smoke_positive_control.py" in src
            assert "transcripts_results" in src and "$WAVE_ROOT/smoke/" in src   # a scratch private root
        assert "positional only" in src and "never --export=" in src
        for m in ("hf_env.sh", "NCCL_NVLS_ENABLE=0", "mirror-figs.sh",
                  "export SISTEER_LENS_STACKED_REDUCTIONS=1", "--reduction-mode stacked --require-attested-mode stacked",
                  "petri_proxy") if script.name != "attacker_pool.sbatch" else ("hf_env.sh", "NCCL_NVLS_ENABLE=0", "mirror-figs.sh"):
            assert m in src or m in common, (script.name, m)
        # NCCL's socket interface is the site's (its nccl.conf): neither the job nor common.sh exports one
        assert not re.search(r"NCCL_SOCKET_IFNAME\s*=", src) and not re.search(r"NCCL_SOCKET_IFNAME\s*=", common), script.name
        assert "f22_stop_requested" in src and "trap cleanup EXIT SIGTERM SIGINT SIGQUIT" in src
        mem = [l for l in src.splitlines() if l.startswith("#SBATCH --mem=")][0]
        assert int(mem.split("=")[1].rstrip("G")) <= 200
        if script.name != "reclose_stores.sbatch":      # the reclose job serves nothing: CPU-only by design (closer-fix-2)
            assert "--gres=gpu:1" in src   # one GPU per server job; drivers are CPU-only children


def test_common_sh_presence_check_accepts_exactly_one_of_three_identity_sidecars():
    """The shell-side check every job runs after the view builds must name all THREE
    sidecars and demand exactly one — a check that named only the training-side spelling
    is what made qwen35_4b/main unbuildable while its si12 copies were perfectly provable."""
    src = (F22 / "common.sh").read_text()
    block = src[src.index("f22_bank_view()"):src.index("f22_driver_venv()")]
    from sisteer.eval.train_budget import IDENTITY_SIDECARS
    for name in IDENTITY_SIDECARS:
        assert name in block, name
    assert "need exactly one of" in block and '-eq 1' in block
    # the old single-spelling demand is gone
    assert 'for f in object.pt train_budget_meta.json staged_from.json' not in block


def test_view_builder_admits_a_fit_record_entry_and_records_the_normalised_facts(tmp_path):
    """The HPO-promotion lineage's staged copy (the real qwen si12 shape) enters the view
    as a plain symlink — its record is never translated into a budget sidecar (the mirror
    gate's rule) and carries no sha, so the PIN binds the bytes — and the view manifest
    records the fit-time facts normalised the ONE way every reader normalises them, so the
    two si12 lineages are legible side by side. A dir declaring two identities refuses."""
    import make_bank_view as mbv

    target, cid = "qwen35_4b", "safety_refusal__psr_kl_all"
    staged = f"si12__{cid}"
    bank = tmp_path / target
    roster = {"wave": "full22", "bank_root": "x",
              "targets": {target: {"views": {"main": {"cells": {cid: staged}}}}}}
    p = _stage_fit_provenance(bank / "cells", staged, cid)
    (tmp_path / "roster.yaml").write_text(yaml.safe_dump(roster))
    (tmp_path / "pins.json").write_text(json.dumps(
        {target: {staged: {"object_sha256": sha256_file(p)}}}))
    argv = ["--target", target, "--view", "main", "--bank", str(bank),
            "--roster", str(tmp_path / "roster.yaml"), "--pins", str(tmp_path / "pins.json")]
    assert mbv.main(argv) == 0
    entry = bank / "cells_view" / "main" / cid
    assert entry.is_symlink()
    assert (entry / "fit_provenance.json").is_file()
    assert not (entry / "train_budget_meta.json").exists()      # never translated
    assert mbv.main(argv + ["--check"]) == 0
    rec, = json.loads((bank / "cells_view" / "main" / "VIEW_MANIFEST.json").read_text())["entries"]
    assert rec["identity"] == "fit_provenance" and rec["mode"] == "symlink"
    assert rec["fit_record"]["source"] == "fit_provenance:tune_arm.py"
    assert rec["fit_record"]["status"] == "verified"
    assert rec["fit_record"]["positions_at_fit"] == "response"
    # the view entry passes the registry's own fit-record verdict
    from sisteer.eval.train_budget import fit_provenance_problems
    assert fit_provenance_problems(entry / "object.pt", cid) == []
    # a second identity sidecar beside it is ambiguous, never picked from
    (p.parent / "train_budget_meta.json").write_text("{}")
    assert mbv.main(argv) == 2
    (p.parent / "train_budget_meta.json").unlink()
    # a record missing a required fit-time fact refuses
    rec_p = p.parent / "fit_provenance.json"
    r = json.loads(rec_p.read_text()); r.pop("chat_template_hash")
    rec_p.write_text(json.dumps(r))
    assert mbv.main(argv) == 2


def test_submit_printer_derives_the_plan_from_the_roster_and_runs_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("SISTEER_CONFIG", str(write_site(tmp_path / "cluster-site")))   # the printer reads each cluster's host from the site
    rc = subprocess.run(["bash", str(F22 / "submit_full22.sh"), "ab" * 20], capture_output=True, text=True)
    assert rc.returncode == 0, rc.stderr
    out = rc.stdout
    assert "sbatch" in out
    # every config has cells since 2026-09-07 (nemotron 4; glm 4 since 2026-09-08) — EXCEPT a view whose cells are all still
    # `pending_fit` (declared, not yet fitted / staged: the qwen35_4b s1 / s2 / s3 seed-replicate views, 2026-09-15), which the
    # printer SKIPs by design ("config lists `none` only") until the staging step moves its pinned ids into `cells`
    pending_only = {f"{t}/{vn}" for t, vn, _, vv in VIEWS if not (vv.get("cells") or {}) and vv.get("pending_fit")}
    skipped = set(re.findall(r"^# SKIP (\S+): config lists `none` only", out, re.M))
    assert skipped == pending_only, (skipped, pending_only)
    assert out.count("# SKIP") == len(pending_only)
    for name in ("f22-att-gemma4e4b-crescendo", "f22-att-qwen35_4b_it-crescendo", "f22-att-qwen35_4b_it-pair", "f22-petri-gemma4e4b",
                 "f22-stat-qwen35_4b-2", "f22-stat-gemma4e4b-3", " bon ", "f22-atk-1",
                 "f22-att-nemotron3nano4b-crescendo", "f22-stat-glm46vflash-3", "f22-att-glm46vflash-pair",
                 "f22-petri-nemotron3nano4b", "f22-petri-glm46vflash"):
        assert name in out, name
    # capability / over-refusal controls are HELD OUT of wave 1 (provenance refusals at the launch smoke)
    assert "bon,controls" not in out and ",controls" not in out
    assert "watch-job.sh --slurm" in out
    # the canonical multi-turn attacks are crescendo + pair, ONE JOB PER ATTACK (each at its own explicit
    # window); ActorAttack is available (F22_ATTACKS) and never emitted by default
    att_names = re.findall(r"f22-att-[A-Za-z0-9_]+-[a-z]+", out)
    assert att_names and all(n.endswith(("-crescendo", "-pair")) for n in att_names), att_names
    assert not any(n.endswith(("-actorattack", "-all")) for n in att_names)
    rc2 = subprocess.run(["bash", str(F22 / "submit_full22.sh"), "ab" * 20], capture_output=True, text=True,
                         env={**os.environ, "F22_ATTACKS": "crescendo,pair,actorattack"})
    assert rc2.returncode == 0 and "f22-att-gemma4e4b-actorattack" in rc2.stdout
    drive = (F22 / "petri_drive_full22.sh").read_text()
    # the auditor chain's two eager preconditions are refused up front, not discovered 28 times
    # the key comes from the environment (OPENROUTER_API_KEY) or else the secret-manager CLI — never a file
    assert 'OPENROUTER_API_KEY:-' in drive and "never from a file" in drive
    # 3931984bf: no GPU auditor leg -- no executable line of the drive reads the Qwen base-url knob or names the model
    # (a historical comment may still mention them; the pin is on the code, not the prose)
    code = "\n".join(l for l in drive.splitlines() if not l.lstrip().startswith("#"))
    assert "PETRI_QWEN_AUDITOR_BASE_URL" not in code and "Huihui-Qwen3.6-27B-abliterated" not in code
    assert 'if [ "$NODE" = "$HOST" ]; then JUMP=()' in drive   # a proxy on the login host is not reached through itself
    src = (F22 / "submit_full22.sh").read_text()
    assert "print(" in src and not any(l.strip().startswith("sbatch ") for l in src.splitlines())


def test_controls_sha_shares_the_serving_path_with_the_wave():
    """The capability/over-refusal controls leg, when it runs, runs at a THIRD sha (the provenance fix), and
    the manifest records it under shas.controls with its own evidence block. The pair invariant above is
    byte-identical src/ + configs/; this leg's is NARROWER and stated: the SERVING path (src/sisteer/capeval/
    serving, tiers/petri_runner.py) and configs/ must be byte-identical to the wave's generation sha, the src/
    files that DO differ are enumerated in evidence.controls.src_touched and must be exactly the provenance
    module(s), and the roster's serving_impl_commit must be unchanged — controls generate no attack data, and
    their diff is the provenance/stamp layer, not generation. Skips until the row exists (the row names the
    sha that ran)."""
    m = json.loads((F22 / "wave_manifest.json").read_text())
    row = m.get("shas", {}).get("controls")
    if not row:
        pytest.skip("no shas.controls row yet — the controls leg has not run")
    w7, c = m["shas"]["W7"]["sha"], row["sha"]
    for sha in (w7, c):
        if subprocess.run(["git", "cat-file", "-e", f"{sha}^{{commit}}"], cwd=REPO, capture_output=True).returncode:
            pytest.skip("this clone does not carry the wave commits (a published repository without its development history); "
                        "the serving-identity claim is not verified here")
    serving = subprocess.check_output(
        ["git", "diff", "--name-only", w7, c, "--", "src/sisteer/capeval/serving",
         "src/sisteer/capeval/tiers/petri_runner.py", "configs/"], cwd=REPO, text=True).split()
    assert serving == [], f"controls sha {c[:8]} touches the serving path / configs: {serving}"
    src = subprocess.check_output(["git", "diff", "--name-only", w7, c, "--", "src/"], cwd=REPO, text=True).split()
    ev = m["evidence"]["controls"]
    assert sorted(src) == sorted(ev["src_touched"]), (src, ev["src_touched"])
    assert set(ev["src_touched"]) <= {"src/sisteer/eval/sweep.py"}, ev["src_touched"]
    def sic(sha):
        r = subprocess.check_output(["git", "show", f"{sha}:configs/sdfbase_full22_roster.yaml"], cwd=REPO, text=True)
        return [l.split()[1] for l in r.splitlines() if l.startswith("serving_impl_commit:")][0]
    assert sic(w7) == sic(c) == m["evidence"]["serving_impl_commit"]["W7"]


def test_wave_manifest_sha_pairs_share_the_serving_code():
    """The wave ran at one sha and its held legs at another. That is only defensible if the two shas run byte-identical
    SERVING code — the one-sha rule guards against arithmetic divergence, not orchestration edits. The manifest
    names the pair and this test checks the invariant against git rather than trusting the prose: nothing
    under src/ or configs/ may differ, and the roster's serving_impl_commit must be the same at both."""
    m = json.loads((F22 / "wave_manifest.json").read_text())
    pairs = m["sha_pairs_that_must_not_diverge_in_generation"]
    assert pairs, "the manifest names at least one sha pair"
    for pr in pairs:
        a, b = pr["a"], pr["b"]
        have = subprocess.run(["git", "cat-file", "-e", f"{a}^{{commit}}"], cwd=REPO, capture_output=True)
        have_b = subprocess.run(["git", "cat-file", "-e", f"{b}^{{commit}}"], cwd=REPO, capture_output=True)
        if have.returncode or have_b.returncode:
            pytest.skip("this clone does not carry the wave commits (a published repository without its development history); "
                        "the serving-identity claim is not verified here")
        touched = subprocess.check_output(["git", "diff", "--name-only", a, b, "--", "src/", "configs/"],
                                          cwd=REPO, text=True).split()
        assert touched == [], f"{a[:8]}..{b[:8]} touches generation-path files: {touched}"
        def sic(sha):
            r = subprocess.check_output(["git", "show", f"{sha}:configs/sdfbase_full22_roster.yaml"], cwd=REPO, text=True)
            return [l.split()[1] for l in r.splitlines() if l.startswith("serving_impl_commit:")][0]
        assert sic(a) == sic(b) == m["evidence"]["serving_impl_commit"]["W7"]
    assert m["evidence"]["src_or_configs_touched"] == []


#: The SERVING path proper — the code whose byte-identity across shas makes rows filed together comparable: the
#: served arithmetic (vllm server / in-proc engine, lens steering + batching, adapters, geometry) together with the
#: lens WORKER (serving/lens_worker.py) and the routing PROXY (serving/proxy.py) that scripts/sdfbase/mt_serving_up.py
#: stands up. The test carries its OWN list rather than reading wave_manifest.json's ``serving_code.paths``, which
#: still names the converter (tiers/petri_runner.py) and is for its owner to narrow; the test only requires the
#: manifest's list to CONTAIN this one.
GLM_SERVING_PATH = ("src/sisteer/capeval/serving",)
GLM_SERVING_WORKER_AND_PROXY = ("src/sisteer/capeval/serving/lens_worker.py",
                                "src/sisteer/capeval/serving/proxy.py")


def test_glm_lane_shas_share_the_serving_path():
    """The GLM lane's generation sha G3 runs the SAME served arithmetic as the Nemotron legs' 254ccbb4 (the two
    new-model targets must be comparable), so the SERVING path proper — ``src/sisteer/capeval/serving/**``, which
    holds the lens worker and the routing proxy mt_serving_up.py stands up — is byte-identical between the two
    shas. That is what the invariant protects: every generated token behind a row filed alongside another sha's
    rows came from identical serving code. The CONVERTER — tiers/petri_runner.py's conversion of already-elicited
    transcripts into store rows — is deliberately OUTSIDE it: it touches no generation and cannot alter a
    successfully scored sample (a converter change re-reads the same .eval logs to the same rows; the inertness
    harness in tests/test_petri_judge_refusal.py proves that on a real cell), so it is not part of this claim. The
    manifest's whole-src delta (src differs only in eval/sweep.py, the fit-provenance reader merged from
    f22-controls-provenance) is checked as a RECORD of what G3 was against Nemotron, not as the invariant. Configs
    differ from other targets by construction (generated roster products), so the whole-src pair invariant is not
    the one to state here. Skips when the shas are not in this clone."""
    m = json.loads((F22 / "wave_manifest.json").read_text())
    g = m["glm_lane"]
    nemo = "254ccbb43bcca7424bdfa0cbfdd8345d3292deb5"
    gen = g["shas"]["G3"]["sha"]
    for sha in (nemo, gen):
        if subprocess.run(["git", "cat-file", "-e", f"{sha}^{{commit}}"], cwd=REPO, capture_output=True).returncode:
            pytest.skip("this clone does not carry the GLM lane commits (a published repository without its development history); "
                        "the serving-identity claim is not verified here")
    # the invariant: the serving path proper is byte-identical between the two shas
    touched = subprocess.check_output(["git", "diff", "--name-only", nemo, gen, "--", *GLM_SERVING_PATH], cwd=REPO, text=True).split()
    assert touched == [], touched
    # ...and the worker + proxy it is stated to include exist under it at both shas (so the claim names real files)
    for f in GLM_SERVING_WORKER_AND_PROXY:
        assert f.startswith(GLM_SERVING_PATH[0] + "/"), f
        for sha in (nemo, gen):
            assert subprocess.run(["git", "cat-file", "-e", f"{sha}:{f}"], cwd=REPO, capture_output=True).returncode == 0, (sha, f)
    # the manifest's own list contains the serving path (its owner may narrow it to exactly this list)
    assert set(GLM_SERVING_PATH) <= set(g["serving_code"]["paths"]), g["serving_code"]["paths"]
    # the RECORD of G3's whole-src delta vs Nemotron, as the manifest states it — a fact about two fixed shas, not
    # the invariant (the converter's later movement on this branch is outside the claim above)
    src = subprocess.check_output(["git", "diff", "--name-only", nemo, gen, "--", "src/"], cwd=REPO, text=True).split()
    assert src == g["serving_code"]["src_touched_vs_nemotron_254ccbb4"], src
    assert g["serving_code"]["identical_to_nemotron_254ccbb4_at_G3"] is True
    def sic(sha):
        r = subprocess.check_output(["git", "show", f"{sha}:configs/sdfbase_full22_roster.yaml"], cwd=REPO, text=True)
        return [l.split()[1] for l in r.splitlines() if l.startswith("serving_impl_commit:")][0]
    assert sic(nemo) == sic(gen) == g["serving_code"]["serving_impl_commit"] == m["evidence"]["serving_impl_commit"]["W7"]


def _write_drive_shims(wt: Path, shims: Path) -> None:
    """The petri_drive_full22.sh shim world (shared by the behavioural drive tests below): ssh / curl / git / the
    secret-manager CLI on PATH, and the drive venv's python as a stand-in whose ``petri_run.py`` freezes the
    ``FAKE_CELLS_JSON`` record on ``print-cells`` and, on ``run``, writes the leg's results.json after a moment (6 s:
    long enough that the driver's heartbeat loop, which starts after the legs are launched, finds a leg alive).
    ``FAKE_RUN_REFUSE_EXISTING=1`` makes ``run`` REFUSE (exit 1, nothing written) a leg whose results.json already
    exists — the driver's refusal of a stale record, so the drive's completion logic can be held to the driver's
    return rather than to the file."""
    def shim(path, body):
        path.write_text("#!/bin/bash\n" + body); path.chmod(0o755)

    shim(shims / "fake-secret-cmd", 'echo not-a-key\n')          # the site's secret-manager CLI (SISTEER_SECRET_CMD below)
    shim(shims / "git", 'case "$1" in rev-parse) echo "$FAKE_SHA";; esac\nexit 0\n')
    # one answer serves every curl: the wave's vmodels + the stacked identity block + the auditor chain's Qwen
    shim(shims / "curl", 'echo "$FAKE_MODELS_JSON"\n')
    shim(shims / "ssh", r'''args="$*"
case "$args" in
  "-N "*|*" -N "*) echo $$ > "$FAKE_REMOTE/tunnel.pid"; exec sleep 3600 ;;          # the tunnel: never exits
  *"cat "*routing_table.json*) echo "$FAKE_RT_JSON" ;;
  *"cat "*cells_served.json*) echo "$FAKE_CS_JSON" ;;
  *"cat "*endpoint.json*) echo "$FAKE_ENDPOINT_JSON" ;;
  *squeue*) echo RUNNING ;;
  *"touch "*) p="${args##*touch }"; mkdir -p "$FAKE_REMOTE$(dirname "$p")"; touch "$FAKE_REMOTE$p" ;;
esac
exit 0
''')
    shim(wt / ".venv/bin/python", r'''case "$1" in
  scripts/petri_run.py)
    echo "petri_run $*" >> "$FAKE_REMOTE/petri_run_argv.log"; shift; sub=""; out=""; idx=""
    echo "auth=${SISTEER_ANTHROPIC_AUTH-unset} key=${ANTHROPIC_API_KEY+present} mode=${SISTEER_SECRETS_MODE-unset}" >> "$FAKE_REMOTE/petri_run_env.log"   # the leg's auth environment (names only)
    while [ $# -gt 0 ]; do case "$1" in print-cells|run) sub=$1;; --out-root) out=$2; shift;; --cell-index) idx=$2; shift;; esac; shift; done
    if [ "$sub" = print-cells ]; then mkdir -p "$out"; echo "$FAKE_CELLS_JSON" > "$out/cells.json"
    else sleep 6; rk=$(python3 -c 'import json,sys;r=json.load(open(sys.argv[1]));r=r["canonical"]+r["controls"];print(r[int(sys.argv[2])]["petri_run_key"])' "$out/cells.json" "$idx")
         if [ -n "${FAKE_RUN_REFUSE_EXISTING:-}" ] && [ -f "$out/cells/$rk/results.json" ]; then echo "refusing: a results.json already sits at $out/cells/$rk (a stale record)"; exit 1; fi
         mkdir -p "$out/cells/$rk"; echo '{}' > "$out/cells/$rk/results.json"; fi
    exit 0 ;;
  scripts/sdfbase/full22/make_bank_view.py) echo "make_bank_view $* defenses_dir=${TT_TWO_SI_DEFENSES_DIR:-unset}" >> "$FAKE_REMOTE/make_bank_view_argv.log"; exit 0 ;;
  -c) case "$2" in *inspect_ai*) exit 0;; esac ;;
esac
exec python3 "$@"
''')


def _drive_docs(serving_path: str, remote_out: str) -> dict:
    """The fake cluster's documents for one engine (``FAKE_*`` env of the shims above). Under ``lens`` the pre-landing
    shims' answers, byte for byte; under ``lora_native`` a NATIVE standup — ``endpoint.json`` ``engine: lora_native``
    naming its ``routing_table`` / ``cells_served`` / ``served_models``, a version-2 table, a FINAL ``cells_served.json``
    with one cell skipped by name, ``/v1/models`` carrying ``petri__none__lora0`` and ``petri_proxy.engine.kind ==
    lora_native``, and a native enumeration record (``serving_path: lora_native``, as petri_run.py freezes it) of three
    legs whose third vmodel is the SKIPPED cell's."""
    if serving_path == "lora_native":
        served = ["petri__base", "petri__none__lora0", "petri__si_x"]
        return {
            "FAKE_MODELS_JSON": json.dumps({"data": [{"id": v} for v in served] + [{"id": "huihui-ai/Huihui-Qwen3.6-27B-abliterated"}],
                                            "petri_proxy": {"engine": {"kind": "lora_native",
                                                                       "adapters": {"none__lora0": {}, "si_x": {}}}}}),
            "FAKE_ENDPOINT_JSON": json.dumps({"node": "gpu-2", "slurm_job_id": "1", "mode": "petri", "engine": "lora_native",
                                              "routing_table": f"{remote_out}/routing_table.json",
                                              "cells_served": f"{remote_out}/cells_served.json", "served_models": served}),
            "FAKE_RT_JSON": json.dumps({"version": 2, "adapters": {"none__lora0": {}, "si_x": {}}, "routes": []}),
            "FAKE_CS_JSON": json.dumps({"engine": "lora_native", "stage": "final",
                                        "cells_skipped": [{"cell_id": "si_psr", "kind": "psr", "reason": "no PEFT form"}]}),
            # three enumerated legs; the third's vmodel is the SKIPPED cell's, which this proxy does not serve
            "FAKE_CELLS_JSON": json.dumps({"serving_path": "lora_native",
                                           "canonical": [{"petri_run_key": "rk0", "virtual_model_name": "petri__base"},
                                                         {"petri_run_key": "rk1", "virtual_model_name": "petri__si_x"},
                                                         {"petri_run_key": "rk2", "virtual_model_name": "petri__si_psr"}],
                                           "controls": []}),
        }
    return {   # the pre-landing shims' answers, byte for byte
        "FAKE_MODELS_JSON": '{"data":[{"id":"petri__none__lens0"},{"id":"huihui-ai/Huihui-Qwen3.6-27B-abliterated"}],'
                            '"petri_proxy":{"lens_reduction_mode":"stacked"}}',
        "FAKE_ENDPOINT_JSON": '{"node":"gpu-2","slurm_job_id":"1","mode":"petri","lens_reduction_mode":"stacked"}',
        "FAKE_RT_JSON": "", "FAKE_CS_JSON": "",
        "FAKE_CELLS_JSON": '{"canonical":[{"petri_run_key":"rk0"},{"petri_run_key":"rk1"}],"controls":[]}',
    }


def _petri_site(wt: Path, home: Path, shared: str) -> None:
    """The drive reads its cluster's login host and shared root from the site config, never from the cluster's label: the
    block reader beside the copied script, and a site under the world's HOME whose one block carries the registry label the
    drive is given (its site label its own) and the shared root the world's fake cluster answers under."""
    (wt / "reproduce/lib").mkdir(parents=True, exist_ok=True)
    (wt / "reproduce/lib/clusters.py").write_bytes((REPO / "reproduce/lib/clusters.py").read_bytes())
    if not (wt / "src").exists():
        (wt / "src/sisteer").mkdir(parents=True, exist_ok=True)
        (wt / "src/sisteer/site.py").write_bytes((REPO / "src/sisteer/site.py").read_bytes())
    site = home / ".config" / "sisteer"
    (site / "clusters").mkdir(parents=True, exist_ok=True)
    (site / "config.env").write_text('SISTEER_CLUSTERS="site_one"\nSISTEER_PRIMARY_CLUSTER=site_one\n')
    (site / "clusters" / "site_one.env").write_text(f"CLUSTER_SSH=login-a\nCLUSTER_SHARED_ROOT={shared}\nCLUSTER_PARTITION=p\n"
                                                    "CLUSTER_QOS=q\nCLUSTER_SUBSTRATE=cluster_a\nCLUSTER_LABEL=site_one\n")


def _drive_world(tmp_path: Path, serving_path: str):
    """The fake grading host + cluster for ONE drive: a worktree holding the REAL script, the bank view and bundle dirs the
    script requires, the shims, the documents, and the paths the assertions read (the drive log's state dir, the eval
    out root — ``~/.local/state/petri[_native]/petri_full22_<tag>`` — and the fake cluster's REMOTE_OUT). The
    environment is assembled by the caller (it carries the auditor chain's key variable)."""
    from types import SimpleNamespace
    home, wt, remote, shims = tmp_path / "home", tmp_path / "wt", tmp_path / "remote", tmp_path / "bin"
    key, view, tag, sha = "glm46vflash", "main", "glm46vflash", "0" * 40
    (wt / "scripts/sdfbase/full22").mkdir(parents=True); (wt / ".venv/bin").mkdir(parents=True)
    (wt / "scripts/sdfbase/full22/petri_drive_full22.sh").write_text((F22 / "petri_drive_full22.sh").read_text())
    (wt / "scripts/sdfbase/full22/petri_tunnel.sh").write_text((F22 / "petri_tunnel.sh").read_text())   # The drive sources it beside itself
    for d in (home / f"stage/sdfbase/full22_objects/{key}/cells_view/{view}",          # the bank under the site's $STAGE (~/stage)
              home / ".local/state/petri/scenarios/8812ba083bb5473c", remote, shims):
        d.mkdir(parents=True, exist_ok=True)
    _write_drive_shims(wt, shims)
    native = serving_path == "lora_native"
    suffix = "_native" if native else ""
    remote_out = f"/shared/tester/exp/sdfbase_full22{suffix}/petri/{tag}/serving"
    _petri_site(wt, home, remote_out.split("/exp/")[0])
    return SimpleNamespace(
        home=home, wt=wt, remote=remote, shims=shims, key=key, view=view, tag=tag, sha=sha, native=native,
        suffix=suffix, remote_out=remote_out, docs=_drive_docs(serving_path, remote_out),
        state=home / f".local/state/sdfbase_full22{suffix}/petri/{tag}",
        out_root=home / f".local/state/petri{suffix}/petri_full22_{tag}",
        lens_out_root=home / f".local/state/petri/petri_full22_{tag}",
        remote_serving=remote / f"shared/tester/exp/sdfbase_full22{suffix}/petri" / tag / "serving")


def _fresh_drive_result(world, out: str, *, verdict: str, rc: int, failed: int, log_slice: str) -> dict:
    """The shell exit alone does not prove f22_drive_verdict ran. The invocation's OWN drive id (its DRIVE_START
    line) must be the one in `drive_result.json` (written by this invocation, not left over), carrying the verdict / rc / failed-leg count,
    and its log slice must carry the fresh DRIVE_VERDICT line."""
    m = re.search(r"drive_id=(\S+) ", out); assert m, out[-1500:]
    drive_id = m.group(1)
    doc = json.loads((world.state / "drive_result.json").read_text())
    assert doc["drive_id"] == drive_id and doc["verdict"] == verdict and doc["rc"] == rc and doc["legs"]["failed"] == failed, (drive_id, doc)
    assert f"DRIVE_VERDICT {verdict} rc={rc} legs:" in log_slice and f"{failed} failed" in log_slice, log_slice[-1500:]
    return doc


def _run_drive(world, env: dict, *, timeout: float = 120, legs: str | None = None):
    """Run the real driver end to end under the shims; return ``(rc, out, tunnel_pid, tunnel_alive)`` with the tunnel
    liveness judged BEFORE the cleanup that kills whatever the driver left behind (files, not pipes: an orphaned sleep
    must not be able to hold the test open). ``legs`` is the script's 5th positional (a subset such as ``"1"``; the
    default drives ALL)."""
    from types import SimpleNamespace
    out = world.wt.parent / "driver.out"
    pidf = world.remote / "tunnel.pid"
    if pidf.exists():
        pidf.unlink()                                   # a re-drive of the same world opens its own tunnel

    def alive(pid):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        return True

    with out.open("w") as fh:
        proc = subprocess.Popen(["bash", "scripts/sdfbase/full22/petri_drive_full22.sh", world.key, world.view, "cluster_a", world.sha]
                                + ([legs] if legs is not None else []),
                                cwd=world.wt, env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            rc = None
        finally:
            tunnel_pid = int(pidf.read_text()) if pidf.exists() else None
            tunnel_alive = tunnel_pid is not None and alive(tunnel_pid)   # judged BEFORE the cleanup below
            for kill in ((lambda: os.killpg(proc.pid, 9)), (lambda: os.kill(tunnel_pid, 9))):
                try:
                    kill()
                except (ProcessLookupError, PermissionError, TypeError):
                    pass
    return SimpleNamespace(rc=rc, out=out.read_text(), tunnel_pid=tunnel_pid, tunnel_alive=tunnel_alive)


def test_petri_drive_overflow_rerun_under_lens_hands_the_endpoint_attestation_to_every_leg_or_fails_before_any(home_site_default, tmp_path):
    """X6: the lens drive pulls no cells_served.json, so under F22_OVERFLOW_RERUN=1 every leg must receive the pulled
    endpoint record (--served-endpoint <state>/endpoint.json, whose attested_max_model_len petri_run.py keys the pass by) beside
    --overflow-rerun; an endpoint that attests no window fails the drive BEFORE any leg is driven (one FATAL, no `run` argv), never leg
    by leg at the runner's refusal. The real driver, end to end under the lens shims."""
    w = _drive_world(tmp_path, "lens")
    env = {**os.environ, "HOME": str(w.home), "USER": "tester", "PATH": f"{w.shims}:{os.environ['PATH']}", "SISTEER_SECRET_CMD": str(w.shims / "fake-secret-cmd"),
           "FAKE_REMOTE": str(w.remote), "FAKE_SHA": w.sha, "OPENROUTER_API_KEY": "k" * 24}
    env.update(w.docs)
    env["F22_SERVING_PATH"] = "lens"; env["F22_OVERFLOW_RERUN"] = "1"
    env["FAKE_ENDPOINT_JSON"] = json.dumps({"node": "gpu-2", "slurm_job_id": "1", "mode": "petri", "lens_reduction_mode": "stacked",
                                            "attested_max_model_len": 49152})
    state = w.state.parent / f"{w.tag}_rerun"
    r = _run_drive(w, env)
    assert r.rc is not None, "the re-run drive did not exit:\n" + r.out[-2000:]
    drive_log = (state / "drive.log").read_text()
    assert "overflow re-run: the proxy attests max_model_len=49152 (endpoint.json; handed to every leg as --served-endpoint)" in drive_log, drive_log[-2500:]
    runs = [c for c in (w.remote / "petri_run_argv.log").read_text().splitlines() if " run " in c]
    assert len(runs) == 2, r.out[-2000:]
    for call in runs:
        assert " --overflow-rerun" in call and f" --served-endpoint {state / 'endpoint.json'} " in call + " ", call
        assert " --verify-standup" not in call and " --serving-path lens " in call
    assert (state / "endpoint.json").exists() and json.loads((state / "endpoint.json").read_text())["attested_max_model_len"] == 49152
    assert "OVERFLOW RE-RUN PASS ENDED" in drive_log and (w.remote_serving.parent / "serving_rerun" / "STOP").exists(), drive_log[-1500:]
    # an endpoint attesting no window: the drive fails before any leg
    w2 = _drive_world(tmp_path / "unattested", "lens")
    env2 = dict(env, HOME=str(w2.home), PATH=f"{w2.shims}:{os.environ['PATH']}", FAKE_REMOTE=str(w2.remote))
    env2.update(w2.docs); env2["F22_SERVING_PATH"] = "lens"; env2["F22_OVERFLOW_RERUN"] = "1"      # the pre-landing lens endpoint: no attestation
    r2 = _run_drive(w2, env2)
    assert r2.rc == 1, r2.out[-2000:]
    log2 = (w2.state.parent / f"{w2.tag}_rerun" / "drive.log").read_text()
    assert "FATAL: overflow re-run on the lens path, but endpoint.json attests no max_model_len" in log2, log2[-2000:]
    assert not (w2.remote / "petri_run_argv.log").exists() or not any(" run " in c for c in (w2.remote / "petri_run_argv.log").read_text().splitlines())


def _tree_digest(root: Path) -> dict:
    return {str(p.relative_to(root)): sha256_file(p) for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.mark.parametrize("serving_path", ["lens", "lora_native"])
def test_petri_drive_waits_on_its_legs_not_the_tunnel(home_site_default, tmp_path, serving_path):
    """Behavioural: the real driver, end to end under shims, against a tunnel ssh that never exits.

    Before the fix the driver's bare ``wait`` (later ``wait $(jobs -p)``) also waited on that tunnel, so after
    the last leg finished it hung, the proxy's STOP was never touched, and the proxy held its GPU until
    walltime (2026-09-07: qwen35_4b_it 4/4 legs COMPLETE, proxy 435478 released only by a manual touch).
    Here ssh/curl/git and the secret-manager CLI are shimmed on PATH, the drive venv's python is a stand-in
    whose petri_run.py freezes two legs and writes each results.json after a moment, and the ``ssh -N`` tunnel is
    a sleep that never returns. The driver must EXIT, touch STOP on the (fake) cluster, and close the tunnel it
    opened. Uses the glm46vflash tag (local port 18986), which no live wave driver holds.

    native-serving-default: parametrised over the serving path. Under ``lens`` (``F22_SERVING_PATH=lens``) every
    shim answers what the pre-landing shims answered, byte for byte, under the lens wave root. Under the DEFAULT
    (``lora_native``) the fake cluster serves a NATIVE standup — ``endpoint.json`` ``engine: lora_native`` naming its
    ``routing_table`` / ``cells_served`` / ``served_models``, a version-2 table and a FINAL ``cells_served.json`` with
    one cell skipped by name, ``/v1/models`` carrying ``petri__none__lora0`` and ``petri_proxy.engine.kind ==
    lora_native`` — and the driver must pull the standup's evidence beside the endpoint, hand ``--serving-path
    lora_native --verify-standup … --verify-routing-table …`` to EVERY ``petri_run.py`` call, drive only the legs
    whose vmodel the proxy serves (the skipped cell's leg is logged, never driven, and never counted as FAILED — it
    would otherwise hold the proxy's GPU until walltime), and release the proxy under the native wave root.
    """
    w = _drive_world(tmp_path, serving_path)
    home, remote, shims, sha, tag = w.home, w.remote, w.shims, w.sha, w.tag
    env = {**os.environ, "HOME": str(home), "USER": "tester", "PATH": f"{shims}:{os.environ['PATH']}", "SISTEER_SECRET_CMD": str(shims / "fake-secret-cmd"),
           "FAKE_REMOTE": str(remote), "FAKE_SHA": sha,
           # the auditor chain's key comes from the environment (the vault bootstrap), never a file
           "OPENROUTER_API_KEY": "not-a-key"}
    native = w.native
    env.update(w.docs)
    env["F22_SERVING_PATH"] = serving_path
    r = _run_drive(w, env)
    rc, tunnel_pid, tunnel_alive = r.rc, r.tunnel_pid, r.tunnel_alive
    if rc is None:
        pytest.fail("the driver did not exit after its legs completed (it is waiting on the tunnel):\n" + r.out[-2000:])
    assert rc == 0, r.out[-2000:]
    stop = w.remote_serving / "STOP"
    assert stop.exists(), r.out[-2000:]
    state = w.state
    drive_log = (state / "drive.log").read_text()
    assert "LEG[0] COMPLETE rk0" in drive_log and "LEG[1] COMPLETE rk1" in drive_log
    argv_calls = (remote / "petri_run_argv.log").read_text().splitlines()
    assert sum(" run " in c for c in argv_calls) == 2 and sum(" print-cells " in c for c in argv_calls) == 1
    # THE HEARTBEAT: touched on the fake cluster while a leg was alive — the proxy reads its absence
    # past HEARTBEAT_GRACE_S as "nothing is driving me" and STOPs; before the fix the loop's liveness test was the
    # contiguous argv substring `--manifest <m> run`, which `--serving-path <p>` broke on BOTH engines
    assert (w.remote_serving / "driver_heartbeat").exists(), "no driver heartbeat reached the proxy: " + drive_log
    # the EVAL OUT ROOT follows the engine: every petri_run.py call reads/writes ONE root — the lens
    # root byte-identical to before (~/.local/state/petri/petri_full22_<tag>), the native one apart from it
    for call in argv_calls:
        assert f" --out-root {w.out_root} " in call + " ", call
    assert f"out={w.out_root} " in drive_log
    assert (w.out_root / "cells.json").exists()
    if native:
        assert w.out_root != w.lens_out_root and not w.lens_out_root.exists()
    if native:
        # the release line names the skipped leg apart (never FAILED: it is not this proxy's work)
        assert "ALL 3 LEGS ENDED (2 COMPLETE, 0 INCOMPLETE on overflow ledgers, 1 not served by this proxy) — STOP touched" in drive_log, drive_log
        assert "leg 2 (petri__si_psr) is NOT served by this proxy" in drive_log and "LEG[2]" not in drive_log
        assert "cells SKIPPED BY NAME on the native engine (never audited here): si_psr" in drive_log
        assert "served engine lora_native" in drive_log and "lens_reduction_mode" not in drive_log
        # the standup's evidence, pulled beside the endpoint and handed to every petri_run.py call
        assert json.loads((state / "routing_table.json").read_text())["version"] == 2
        assert json.loads((state / "cells_served.json").read_text())["engine"] == "lora_native"
        for call in argv_calls:
            assert (" --serving-path lora_native " in call and f" --verify-standup {state}/cells_served.json " in call
                    and f" --verify-routing-table {state}/routing_table.json " in call), call
    else:
        # the script's release line (the pre-landing assertion named an OLDER spelling of it — the recorded Modal
        # baseline failure of this test on main; the lens run is otherwise the pre-landing run, byte for byte)
        assert "ALL 2 LEGS ENDED (2 COMPLETE, 0 INCOMPLETE on overflow ledgers) — STOP touched" in drive_log, drive_log
        assert "served lens_reduction_mode=stacked" in drive_log and "not served by this proxy" not in drive_log
        for call in argv_calls:
            assert " --serving-path lens " in call and "--verify-" not in call, call
    # the tunnel this run opened is closed with the proxy; a live one would be a leaked ssh per finished tag
    assert tunnel_pid is not None and not tunnel_alive, "the tunnel ssh (sleep stand-in) outlived the driver"
    assert "tunnel (pid " in drive_log and ") closed" in drive_log


def _native_drive_env(w) -> dict:
    return {**os.environ, "HOME": str(w.home), "USER": "tester", "PATH": f"{w.shims}:{os.environ['PATH']}", "SISTEER_SECRET_CMD": str(w.shims / "fake-secret-cmd"),
            "FAKE_REMOTE": str(w.remote), "FAKE_SHA": w.sha, "OPENROUTER_API_KEY": "not-a-key",
            "F22_SERVING_PATH": "lora_native", **w.docs}


def test_petri_drive_reads_its_bank_view_under_F22_GRADING_HOST_BANK_ROOT_and_names_it(home_site_default, tmp_path):
    """demo-import (2026-10-11): the drive's bank root is F22_GRADING_HOST_BANK_ROOT when set — the grading daemon hands its --petri-bank-root,
    08_grade.sh's $SISTEER_OBJECTS_BANK, where 05 staged the model's objects — else the paper's legacy $STAGE/sdfbase/full22_objects,
    byte for byte (every other drive test here runs on that default). The real driver end to end under the native shims: with the DEFAULT
    root's view removed and the override's present it drives to completion; TT_TWO_SI_DEFENSES_DIR and the make_bank_view --check read the
    override's bank; DRIVE_START names the root. An override whose view is absent is a one-line FATAL naming the path and the variable, before
    any leg — even while the default root's view exists (the override is the root, never a fallback)."""
    w = _drive_world(tmp_path, "lora_native")
    legacy_view = w.home / f"stage/sdfbase/full22_objects/{w.key}/cells_view/{w.view}"   # the default: the bank under the site's $STAGE
    legacy_view.rmdir()                                                   # the default root's view, made by the world: gone
    bank = tmp_path / "site_bank"; view = bank / w.key / "cells_view" / w.view; view.mkdir(parents=True)
    env = _native_drive_env(w); env["F22_GRADING_HOST_BANK_ROOT"] = str(bank)
    r = _run_drive(w, env)
    assert r.rc == 0, r.out[-2000:]
    drive_log = (w.state / "drive.log").read_text()
    assert f" bank_root={bank} " in drive_log and str(w.home / "stage") not in drive_log, drive_log
    assert "LEG[0] COMPLETE rk0" in drive_log and (w.remote_serving / "STOP").exists()
    (mbv,) = (w.remote / "make_bank_view_argv.log").read_text().splitlines()
    assert f" --check --bank {bank}/{w.key} --target {w.key} --view {w.view} " in mbv and mbv.endswith(f"defenses_dir={view}"), mbv
    # the override's view absent: refused by name before DRIVE_START, the default root's view (present here) never a fallback
    w2 = _drive_world(tmp_path / "absent", "lora_native")
    empty = tmp_path / "absent" / "empty_bank"; empty.mkdir()
    env2 = _native_drive_env(w2); env2["F22_GRADING_HOST_BANK_ROOT"] = str(empty)
    r2 = _run_drive(w2, env2, timeout=60)
    assert r2.rc == 1, r2.out[-2000:]
    assert (f"FATAL: no grading host bank view at {empty}/{w2.key}/cells_view/{w2.view} (bank root {empty} from F22_GRADING_HOST_BANK_ROOT; "
            f"make_bank_view.py --bank {empty}/{w2.key} --target {w2.key} --view {w2.view} ") in r2.out, r2.out[-2000:]
    assert "DRIVE_START" not in r2.out and not (w2.remote / "petri_run_argv.log").exists() and not (w2.remote / "make_bank_view_argv.log").exists()


def _proxy_release(tmp_path: Path, name: str, *, heartbeat: str) -> tuple:
    """The REAL heartbeat loop of petri_proxy.sbatch — from `while :; do` to its `exit $RC` — under bash with full22/common.sh sourced (the
    named exit codes), f22_log to stdout, no wave STOP, a stand-in standup that exits 0 once $OUT/STOP appears, the grace / stale windows
    collapsed to 2 s (READY 10 s ago) and the poll to 0.2 s. `heartbeat`: `never` (no driver ever), `stale` (one heartbeat 10 s old),
    `live` (a fresh heartbeat, then the driver touches STOP: the legs completed). Returns (the completed process, $OUT)."""
    src = (F22 / "petri_proxy.sbatch").read_text()
    block = src[src.index("\nwhile :; do\n") + 1:]
    assert block.rstrip().endswith("exit $RC") and block.count("  sleep 30\n") == 1
    block = block.replace("  sleep 30\n", "  sleep 0.2\n")
    out = tmp_path / name; out.mkdir()
    pre = {"never": "", "stale": 'touch -d "10 seconds ago" "$HB"; ', "live": 'touch "$HB"; ( sleep 1; touch "$OUT/STOP" ) & '}[heartbeat]
    script = (f"set -u; export REPO_DIR={REPO} SHARED={tmp_path} SLURM_JOB_ID=1 SLURM_JOB_NAME=t; source {F22 / 'common.sh'}; "
              'f22_log(){ echo "$*"; }; f22_stop_requested(){ return 1; }; '
              f'OUT={out}; HB="$OUT/driver_heartbeat"; HEARTBEAT_GRACE_S=2; HEARTBEAT_STALE_S=2; READY_AT=$(( $(date +%s) - 10 )); RELEASE_RC=0; RELEASE_NAME=""; '
              '( while [ ! -e "$OUT/STOP" ]; do sleep 0.1; done ) & TARGET_PID=$!; ' + pre + block)
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=120), out


def test_petri_proxy_releases_with_no_driver_exit_their_named_codes_never_completed(tmp_path):
    """demo-import (2026-10-11): a proxy that releases its GPU because NO driver heartbeat arrived within HEARTBEAT_GRACE_S of READY, or
    because the heartbeat went stale, used to `touch STOP` and exit 0 — Slurm said COMPLETED 0:0, and a COMPLETED registering unit is one
    submit_twin.py's resubmit_decision never re-mints (its runs are "the unit's results"; 06 --resubmit's walk skips it): the demo's proxy
    would have sat released with nothing generated and no way back but a hand cancel. Each release now leaves with its NAMED code from
    common.sh — PROXY_NO_DRIVER (F22_EXIT_PROXY_NO_DRIVER) / PROXY_DRIVER_LOST (F22_EXIT_PROXY_DRIVER_LOST) — written to the log and
    carried into RC (the RESULT line's rc=, the registry's FAILED:rc=N), distinct from 0 and from the closer's 75; a driver that completes
    its legs (STOP touched) still exits with the standup's 0."""
    common = (F22 / "common.sh").read_text()
    codes = {ln.split("=")[0]: int(ln.split("=")[1]) for ln in common.splitlines() if ln.startswith("F22_EXIT_PROXY_")}
    assert codes == {"F22_EXIT_PROXY_NO_DRIVER": 76, "F22_EXIT_PROXY_DRIVER_LOST": 77}, codes
    closer = int(next(ln.split("=")[1] for ln in common.splitlines() if ln.startswith("F22_EXIT_CLOSER_REFUSED=")))
    twin = (REPO / "reproduce/lib/submit_twin.py").read_text()
    assert f"CLOSER_REFUSED_EXIT = {closer}" in twin and not {0, closer} & set(codes.values())   # distinct from success and from `reclose`
    r, out = _proxy_release(tmp_path, "never", heartbeat="never")
    assert r.returncode == 76, (r.returncode, r.stdout, r.stderr)
    assert "no driver heartbeat within 2s of READY — nothing is driving this proxy; releasing the GPU; PROXY_NO_DRIVER exit=76" in r.stdout, r.stdout
    assert "leaving with PROXY_NO_DRIVER exit=76" in r.stdout and "petri proxy exited rc=76" in r.stdout and (out / "STOP").exists()
    r, out = _proxy_release(tmp_path, "stale", heartbeat="stale")
    assert r.returncode == 77, (r.returncode, r.stdout, r.stderr)
    assert "driver heartbeat stale (" in r.stdout and "the grading host driver is gone; releasing the GPU" in r.stdout and "PROXY_DRIVER_LOST exit=77" in r.stdout, r.stdout
    assert "petri proxy exited rc=77" in r.stdout and (out / "STOP").exists()
    r, out = _proxy_release(tmp_path, "live", heartbeat="live")
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
    assert "STOP touched — legs complete or a deliberate stop" in r.stdout and "petri proxy exited rc=0" in r.stdout and "PROXY_" not in r.stdout


def test_petri_drive_native_never_reads_the_lens_audit_and_holds_completion_to_the_driver(home_site_default, tmp_path):
    """Behavioural against the real driver under the native default, three drives — then the SUBSET RETRIES
    over the third world (4–6 below):

    1. a COMPLETED synthetic lens audit sits under the LENS eval root (``~/.local/state/petri/petri_full22_<tag>``: a
       lens ``cells.json`` — no ``serving_path`` key — whose legs carry the SAME vmodels the native proxy serves, and a
       ``results.json`` per leg). Before the fix the native drive shared that root: an existing ``cells.json`` suppressed
       its own ``print-cells``, the served-model filter kept the lens legs (base and adapter vmodel names overlap), and
       the completion count read the lens ``results.json`` as ITS completions — a native audit reported COMPLETE with
       lens results, and the proxy was STOPped. The native drive must freeze and run under ITS OWN root
       (``~/.local/state/petri_native/…``) and leave the lens audit byte-identical.
    2. a FOREIGN record at the native root (a lens-shaped ``cells.json``): refused by name BEFORE any leg is selected —
       exit 1, no ``run``, no STOP (petri_run.py would refuse each leg, but the leg selection and the completion count
       read the record directly).
    3. a stale ``results.json`` under the native root and a driver that REFUSES that leg (exit 1, the shim's
       ``FAKE_RUN_REFUSE_EXISTING``): the leg is FAILED on the driver's return, the release is withheld ("legs remain",
       no STOP), and the old file's existence never turns the refusal into a completion.
    4. a SUBSET retry of leg 1 ONLY over that state: the pre-fix driver ``rm -f``'d every ``leg_*.rc`` up front,
       so leg 0's FAILED return vanished, the count read its stale ``results.json`` as a completion and STOP was touched
       over an unaudited leg. The failed status must STAND (it is not this invocation's leg), named with the drive that
       returned it, and the release stays withheld; leg 1's record is rewritten by THIS drive (its id, its bytes).
    5. re-driving leg 0 itself, and the driver now returns 0: the failed status is superseded by the success and the
       proxy is released — a failed status stands exactly until that same leg succeeds.
    6. a ``results.json`` whose bytes CHANGED since the drive that returned 0 over it: UNVERIFIED by name, the release
       withheld — a return of 0 is a completion over the bytes it was returned on, never over whatever sits there later.
    """
    # 1. the completed lens audit under the lens root; the native drive runs beside it
    w = _drive_world(tmp_path / "s1", "lora_native")
    lens_cells = {"audit": f"petri_full22_{w.tag}", "expected_cells": ["none", "si_x"],
                  "canonical": [{"petri_run_key": "lens0", "virtual_model_name": "petri__base"},
                                {"petri_run_key": "lens1", "virtual_model_name": "petri__si_x"}],
                  "controls": [], "absent": [], "written_at": 1.0}
    w.lens_out_root.mkdir(parents=True)
    (w.lens_out_root / "cells.json").write_text(json.dumps(lens_cells, indent=2, sort_keys=True))
    for rk in ("lens0", "lens1"):
        (w.lens_out_root / "cells" / rk).mkdir(parents=True)
        (w.lens_out_root / "cells" / rk / "results.json").write_text('{"lens": true}')
    before = _tree_digest(w.lens_out_root)
    r = _run_drive(w, _native_drive_env(w))
    assert r.rc == 0, r.out[-3000:]
    assert _tree_digest(w.lens_out_root) == before, "the native drive touched the lens audit"
    drive_log = (w.state / "drive.log").read_text()
    argv_calls = (w.remote / "petri_run_argv.log").read_text().splitlines()
    assert sum(" print-cells " in c for c in argv_calls) == 1, argv_calls        # its OWN enumeration, frozen
    assert sum(" run " in c for c in argv_calls) == 2, argv_calls
    for call in argv_calls:
        assert f" --out-root {w.out_root} " in call + " " and str(w.lens_out_root) not in call, call
    assert json.loads((w.out_root / "cells.json").read_text())["serving_path"] == "lora_native"
    assert "LEG[0] COMPLETE rk0" in drive_log and "LEG[1] COMPLETE rk1" in drive_log and "lens0" not in drive_log
    assert "ALL 3 LEGS ENDED (2 COMPLETE, 0 INCOMPLETE on overflow ledgers, 1 not served by this proxy) — STOP touched" in drive_log, drive_log
    assert (w.remote_serving / "STOP").exists() and (w.remote_serving / "driver_heartbeat").exists()

    # 2. a foreign (lens) record at the native root: refused before any leg
    w2 = _drive_world(tmp_path / "s2", "lora_native")
    w2.out_root.mkdir(parents=True)
    (w2.out_root / "cells.json").write_text(json.dumps(lens_cells, indent=2, sort_keys=True))
    r2 = _run_drive(w2, _native_drive_env(w2))
    assert r2.rc == 1, r2.out[-3000:]
    assert ("was frozen under --serving-path lens but this drive is --serving-path lora_native" in r2.out), r2.out[-3000:]
    assert not (w2.remote / "petri_run_argv.log").exists(), "a leg (or print-cells) ran against the other engine's record"
    assert not (w2.remote_serving / "STOP").exists()
    assert json.loads((w2.out_root / "cells.json").read_text()) == lens_cells          # never overwritten

    # 3. a stale results.json under the native root; the driver refuses that leg -> FAILED, release withheld
    w3 = _drive_world(tmp_path / "s3", "lora_native")
    w3.out_root.mkdir(parents=True)
    (w3.out_root / "cells.json").write_text(w3.docs["FAKE_CELLS_JSON"])
    (w3.out_root / "cells" / "rk0").mkdir(parents=True)
    (w3.out_root / "cells" / "rk0" / "results.json").write_text('{"stale": true}')
    r3 = _run_drive(w3, {**_native_drive_env(w3), "FAKE_RUN_REFUSE_EXISTING": "1"}, timeout=240)
    assert r3.rc == 1, r3.out[-3000:]           # a FAILED leg is the drive's own exit 1 (17422e8fc f22_drive_verdict): the RELEASE is withheld AND the caller sees it
    log3 = (w3.state / "drive.log").read_text()
    _fresh_drive_result(w3, r3.out, verdict="failed", rc=1, failed=1, log_slice=log3)
    assert "LEG[0] FAILED rc=1 rk0 (an existing results.json does not override the failed driver return)" in log3, log3
    assert "LEG[1] COMPLETE rk1" in log3 and "LEG[0] COMPLETE" not in log3
    assert "legs remain — proxy left up" in log3 and "the tunnel this drive opened closes with it (D1503)" in log3 and "STOP touched" not in log3
    assert not (w3.remote_serving / "STOP").exists()
    def status(w, idx):
        """``leg_<idx>.rc`` as the driver writes it: ``<rc> drive=<id> results=<sha256|absent>`` (one line, three fields);
        an r1 record is the bare rc (drive / results None), so the behavioural assertions below run against either."""
        fields = (w.state / f"leg_{idx}.rc").read_text().split()
        rest = dict(f.split("=", 1) for f in fields[1:])
        return fields[0], rest.get("drive"), rest.get("results")
    st0, st1 = status(w3, 0), status(w3, 1)
    assert (st0[0], st1[0]) == ("1", "0")
    calls3 = (w3.remote / "petri_run_argv.log").read_text().splitlines()
    assert sum(" print-cells " in c for c in calls3) == 0                          # the native record stood
    assert sum(" run " in c and c.rstrip().endswith("--cell-index 0") for c in calls3) == 3   # three attempts, then FAILED

    # 4. a SUBSET retry — leg 1 only — over the same state: leg 0's FAILED return must stand. The
    # BEHAVIOUR first (the pre-fix driver released the proxy here), then the record that carries it
    (w3.remote / "petri_run_argv.log").unlink()
    r4 = _run_drive(w3, _native_drive_env(w3), legs="1")                          # the shim writes leg 1 afresh (no refusal)
    assert r4.rc == 1, r4.out[-3000:]                                             # leg 0's standing FAILED return is still the drive's verdict
    log4 = (w3.state / "drive.log").read_text()[len(log3):]
    _fresh_drive_result(w3, r4.out, verdict="failed", rc=1, failed=1, log_slice=log4)
    calls4 = (w3.remote / "petri_run_argv.log").read_text().splitlines()
    assert [c.rstrip()[-len("--cell-index 1"):] for c in calls4 if " run " in c] == ["--cell-index 1"], calls4   # ONLY leg 1 driven
    assert "legs=[1] " in log4 and "LEG[1] COMPLETE rk1" in log4 and "LEG[0]" not in log4
    assert not (w3.remote_serving / "STOP").exists(), "the subset retry released the proxy over leg 0's stale results.json:\n" + log4
    assert "legs remain — proxy left up" in log4 and "the tunnel this drive opened closes with it (D1503)" in log4 and "STOP touched" not in log4
    # the record: `<rc> drive=<id> results=<sha256>` — leg 0's is the FIRST drive's, untouched (not this invocation's
    # leg) and named as that drive's verdict in the count; leg 1's is rewritten by THIS drive (its id, its bytes)
    rc0, drive0, res0 = st0
    assert rc0 == "1" and drive0 and re.fullmatch(r"\d{8}T\d{6}Z\.\d+", drive0) and st1[1] == drive0, (st0, st1)
    assert res0 == hashlib.sha256(b'{"stale": true}').hexdigest()                  # the stale bytes the refusal was over
    assert f"drive_id={drive0} " in log3 and f"LEG[0] FAILED rc=1 rk0 (an existing results.json does not override the failed driver return) (drive {drive0})" in log3
    assert status(w3, 0) == st0                                                    # untouched by the subset retry
    assert f"leg 0 (rk0) FAILED rc=1 as returned by drive {drive0} — held" in log4, log4     # the EARLIER drive's verdict, by name
    rc1b, drive1b, res1b = status(w3, 1)
    assert rc1b == "0" and drive1b != drive0 and f"drive_id={drive1b} " in log4      # rewritten by THIS drive
    assert res1b == hashlib.sha256((w3.out_root / "cells" / "rk1" / "results.json").read_bytes()).hexdigest()

    # 5. re-driving leg 0 and succeeding supersedes its failed status; the proxy is released
    (w3.remote / "petri_run_argv.log").unlink()
    r5 = _run_drive(w3, _native_drive_env(w3), legs="0")
    assert r5.rc == 0, r5.out[-3000:]
    log5 = (w3.state / "drive.log").read_text()[len(log3) + len(log4):]
    assert [c.rstrip()[-len("--cell-index 0"):] for c in (w3.remote / "petri_run_argv.log").read_text().splitlines() if " run " in c] == ["--cell-index 0"]
    assert "LEG[0] COMPLETE rk0" in log5 and "FAILED" not in log5 and "UNVERIFIED" not in log5 and "by its cell dir alone" not in log5
    assert "ALL 3 LEGS ENDED (2 COMPLETE, 0 INCOMPLETE on overflow ledgers, 1 not served by this proxy) — STOP touched" in log5, log5
    assert (w3.remote_serving / "STOP").exists()
    rc0c, drive0c, res0c = status(w3, 0)
    assert rc0c == "0" and drive0c not in (drive0, drive1b) and res0c == hashlib.sha256(b"{}\n").hexdigest()   # the shim's fresh bytes
    assert status(w3, 1) == (rc1b, drive1b, res1b)

    # 6. results.json bytes that CHANGED since the drive that returned 0 over them: UNVERIFIED, the release withheld
    (w3.remote_serving / "STOP").unlink(); (w3.remote / "petri_run_argv.log").unlink()
    (w3.out_root / "cells" / "rk1" / "results.json").write_text('{"edited": "by hand"}')
    r6 = _run_drive(w3, _native_drive_env(w3), legs="0")
    assert r6.rc == 1, r6.out[-3000:]                                             # an UNVERIFIED leg is held: the drive's own exit is 1 (17422e8fc)
    log6 = (w3.state / "drive.log").read_text()[len(log3) + len(log4) + len(log5):]
    _fresh_drive_result(w3, r6.out, verdict="failed", rc=1, failed=1, log_slice=log6)
    assert f"leg 1 (rk1) UNVERIFIED: results.json differs from the bytes drive {drive1b} returned 0 over — held" in log6, log6
    assert "legs remain — proxy left up" in log6 and "the tunnel this drive opened closes with it (D1503)" in log6 and "STOP touched" not in log6 and not (w3.remote_serving / "STOP").exists()
    assert status(w3, 1) == (rc1b, drive1b, res1b)                                   # the record is the drive's, never re-derived


def test_driver_queue_bound_counts_live_drivers_not_the_background_server(tmp_path):
    """Behavioural, against the real common.sh functions: with the target server running as a background job of the
    same shell (as f22_standup starts it) and MAX_DRIVERS=1 (a one-cell controls phase: MAX_DRIVERS=NCELLS), the
    queue must still launch its one driver. Before the fix the bound read `jobs -rp`, which counted the server, and
    the driver never launched: job 464522 (2026-09-08) held its GPU 50 min after "phase CONTROLS" with zero requests
    and no driver log; the six-cell controls jobs of the third cut never hit it (bound 6 > 1)."""
    sh = (F22 / "common.sh").read_text()
    start = sh.index("f22_live_drivers(){")
    end = sh.index("f22_queue_wait(){")
    funcs = sh[start:end]
    code = "\n".join(l.split("#", 1)[0] for l in funcs.splitlines())          # the CODE, comments stripped
    assert "jobs -rp" not in code, "the launch bound must count live drivers, not the shell's job table"
    (tmp_path / "drivers").mkdir()
    script = f"""set -u
{funcs}
SERVE_DIR={tmp_path}; REPO_DIR={tmp_path}; MAX_DRIVERS=1
declare -a DRIVER_PIDS=(); declare -A DRIVER_TAG=()
sleep 30 &                       # the target server, a background job of THIS shell
TARGET_PID=$!
f22_queue_launch ctl__cell echo launched          # must not block behind the server
wait "${{DRIVER_PIDS[0]}}"; rc=$?
kill "$TARGET_PID"
echo "drivers_launched=${{#DRIVER_PIDS[@]}} rc=$rc"
"""
    r = subprocess.run(["timeout", "20", "bash", "-c", script], capture_output=True, text=True)
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)   # 124 = the pre-fix deadlock (timeout)
    assert "drivers_launched=1 rc=0" in r.stdout, r.stdout
    assert (tmp_path / "drivers" / "ctl__cell.log").read_text().strip() == "launched"


# --- 9: the DOSE LADDER (the strength frontier, 2026-09-10) ----------------------------------

#: (dose, view-name token) in ladder order — the roster's DOSE LADDER block spells the rule
DOSE_LADDER = [(-1.0, "m1"), (0.1, "0p1"), (0.25, "0p25"), (0.5, "0p5"), (1.0, "1"), (1.5, "1p5")]
#: the HIGH-DOSE EXTENSION of one lane: gemma4e4b's base->instruct family read flat over the ladder,
#: so it carries two more bare views after d1p5 — own d2__ / d4__ objects, the same three cells. No other target, no it_ view.
HIGH_DOSE_EXTENSION = {"gemma4e4b": [(2.0, "2"), (4.0, "4")]}
INSTRUCT_TARGETS = ("gemma4e4b", "qwen35_4b")        # instruct-direct lane: `it_` prefix; since the it64 refit the LoRA + its subtraction
CHAT_TARGETS = ("nemotron3nano4b", "glm46vflash")     # chat-checkpoint lane: the LoRA + its subtraction


def _dose_views(target):
    return {vn: v for vn, v in ROSTER["targets"][target]["views"].items() if "dose" in v}


def test_dose_views_serve_exactly_the_lora_objects_at_that_dose_and_no_prompted_cell():
    """Six dose views per target (dm1 d0p1 d0p25 d0p5 d1 d1p5; `it_`-prefixed on the instruct lanes; gemma4e4b's
    base->instruct family adds d2 and d4, HIGH_DOSE_EXTENSION), each
    serving ONLY the LoRA object(s) at that dose against its own same-pass `none` anchor: the constitution
    LoRA (and, on the chat-checkpoint lanes, its control-subtracted difference). Dose 1.0 RE-SERVES the
    unscaled itdoc__ / base__ + derived__ objects (the same bytes as the campaign's it / main row, so
    the frontier's 1.0 is measured by the same code, day, anchor and instrument as its neighbours); 0.5
    reuses the halfdose__ / halfdoseit__ bytes; every other dose is its own scaled copy. No prompted
    cell anywhere in the ladder; every staged id pinned and verified on both clusters; the dose parseable
    from the view name and agreeing with the view's `dose:` key; the generated config keeps the campaign's
    BoN block byte-for-byte (Scale's 512 -> 4096 climb reads it; the native-4096 override is gone) and
    the two other caps untouched. The LoRA ladders ONLY: qwen35_4b's `d<tok>_relu` views (the ReLU difference ladder,
    2026-09-16) are another object family against their own anchors, held by
    test_relu_dose_views_serve_exactly_the_psr_difference_object_at_that_dose_and_no_other_cell."""
    CTL = "constv2ctl2-full__lora_r4_ce"
    for target in INSTRUCT_TARGETS + CHAT_TARGETS:
        t, dv_all = ROSTER["targets"][target], _dose_views(target)
        # the bare and `it_` spellings with NO object-family segment: the LoRA ladders (BARE_DOSE_VIEWS_FROZEN)
        dv_all = {vn: v for vn, v in dv_all.items() if gen._DOSE_VIEW_RE.match(vn).group("objfam") is None}
        instruct = target in INSTRUCT_TARGETS
        prefix = "it_" if instruct else ""
        # the instruct lanes carry a SECOND family: the base-fit objects (constitution, CONTROL, subtraction) as bare
        # `d<tok>` views with lane-owned `d<tok>__` ids at EVERY dose (1.0 the byte-identical anchor, 0.5 a fresh copy)
        base_family = {vn: v for vn, v in dv_all.items() if not vn.startswith("it_")} if instruct else {}
        dv = {vn: v for vn, v in dv_all.items() if vn.startswith(prefix) and (not instruct or vn.startswith("it_"))}
        assert list(dv) == [f"{prefix}d{tok}" for _, tok in DOSE_LADDER], target     # six, in ladder order
        if instruct:
            base_ladder = DOSE_LADDER + HIGH_DOSE_EXTENSION.get(target, [])
            assert list(base_family) == [f"d{tok}" for _, tok in base_ladder], target    # six (+ gemma's 2, 4), in ladder order
            for (dose, tok), (vn, v) in zip(base_ladder, base_family.items()):
                assert v["dose"] == dose and gen.view_dose(ROSTER, target, vn) == dose
                want3 = ["constv2-mix-full__lora_r4_ce", CTL, "constv2-mix-full__lora_r4_ce_minus_ctl"]
                assert v["cells"] == {c: f"d{tok}__{c}" for c in want3}, (target, vn)
                for sid in v["cells"].values():
                    pin = PINS[target][sid]
                    assert len(pin["object_sha256"]) == 64 and pin["verified_cluster_a"] and pin["verified_cluster_b"], (target, sid)
                if dose == 1.0:      # the anchor IS the parent's bytes (the `main` view's objects), under the ladder's own id
                    anchor = t["views"]["main"]["cells"]
                    for c in want3:
                        assert PINS[target][v["cells"][c]]["object_sha256"] == PINS[target][anchor[c]]["object_sha256"], (target, c)
            shas3 = [PINS[target][sid]["object_sha256"] for v in base_family.values() for sid in v["cells"].values()]
            assert len(shas3) == len(set(shas3)) == 3 * len(base_ladder), target    # 18, or 24 with the extension
        want_cids = ["constv2-mix-full__lora_r4_ce", "constv2-mix-full__lora_r4_ce_minus_ctl"]   # every lane: the LoRA + its subtraction
        for (dose, tok), (vn, v) in zip(DOSE_LADDER, dv.items()):
            assert v["dose"] == dose and gen.dose_token(dose) == tok, (target, vn)
            assert gen.view_dose(ROSTER, target, vn) == dose
            cells = v["cells"]
            assert list(cells) == want_cids, (target, vn)
            served, staged = fr.served_cells(ROSTER, target, vn), fr.staged_cells(ROSTER, target, vn)
            assert served == staged == cells and not any(fr.is_prompted(c) for c in served), (target, vn)
            if dose == 1.0:      # The UNSCALED objects again — the ladder's own 1.0 anchor
                if instruct:     # the it64 refit parents: the ladder's 1.0 is its own re-serve anchor, reported
                                 # beside the campaign's `it` row (itdoc__) as an unreconciled pair, never equated to it
                    assert cells == {"constv2-mix-full__lora_r4_ce": "itdoc64__constv2-mix-full__lora_r4_ce",
                                     "constv2-mix-full__lora_r4_ce_minus_ctl":
                                         "derived_itdoc64__constv2-mix-full__lora_r4_ce_minus_ctl"}, (target, vn)
                else:            # the campaign's own main staged ids
                    anchor = t["views"]["main"]["cells"]
                    assert cells == {c: anchor[c] for c in want_cids}, (target, vn)
                    assert all(sid.startswith(("base__", "derived__")) for sid in cells.values())
            elif dose == 0.5 and not instruct:    # The halfdose bytes, no new 0.5 object (chat-checkpoint lanes)
                assert cells == {c: "halfdose__" + c for c in want_cids}, (target, vn)
            else:                # a dose-scaled copy, lane-prefixed by the dose token (`it64` on the refit lanes)
                assert cells == {c: f"d{tok}{'it64' if instruct else ''}__{c}" for c in want_cids}, (target, vn)
            for cid, sid in cells.items():
                pin = PINS[target][sid]
                assert len(pin["object_sha256"]) == 64, (target, sid)
                assert pin["verified_cluster_a"] and pin["verified_cluster_b"], (target, sid)
                assert pin["sidecar_object_sha256_recorded"] == "match", (target, sid)
            # the generated pair: the cells, the BoN cap override, the other two caps the body's, the header echo
            cfg = _cfg(v)
            assert cfg[EXPLICIT_CELLS_KEY] == ["none"] + want_cids
            body = yaml.safe_load((REPO / t["body_config"]).read_text())
            # the BoN rule for main's canonical best-of-N: the view's bon is the BODY's plus exactly the two
            # wave-wide climb keys, which the generator adds to every view. The base cap is still 512 — the
            # climb escalates the 512 store, it does not replace it.
            bon_cfg = dict(cfg["bon"])
            assert bon_cfg.pop("climb_cap") == ROSTER["bon_climb_cap"] == 4096
            assert bon_cfg.pop("climb_decode") == ROSTER["bon_climb_decode"] == "as_drawn"
            assert bon_cfg == body["bon"] and cfg["bon"]["max_new_tokens"] == gen.BON_MAX_NEW_TOKENS == 512
            assert cfg["attacks"]["controls_max_new_tokens"] == 512 and cfg["attacks"]["static_max_new_tokens"] == 4096
            text = (REPO / v["config"]).read_text()
            assert f"a DOSE view (dose {dose:g};" in text and "`bon.max_new_tokens` STAYS the body's 512" in text
            assert "OVERRIDDEN" not in text and "  max_new_tokens: 4096\n" not in text   # the native-4096 override is gone
            spec = cm.load_petri_spec(REPO / v["manifest"])
            assert spec.cells == ["none"] + [gen.harness_cell_id(c) for c in want_cids]
            assert not any(c.startswith("prompted__") for c in spec.cells)
            assert spec.audit == f"petri_full22_{target}_{vn}"
            # the job-side count guard's two references agree (no prompted cell): 2 on every lane
            for extra, want in (((), len(want_cids)), (("--staged",), len(want_cids))):
                rc = subprocess.run([sys.executable, str(F22 / "roster_cell_count.py"), target, vn, *extra],
                                    capture_output=True, text=True)
                assert rc.returncode == 0 and int(rc.stdout.strip()) == want, (target, vn, extra, rc.stderr)
        # every dose of the ladder is DIFFERENT bytes (a scaled copy is its own object; 1.0 and 0.5 included)
        shas = [PINS[target][sid]["object_sha256"] for v in dv.values() for sid in v["cells"].values()]
        assert len(shas) == len(set(shas)) == 6 * len(want_cids), target
    # nemotron's pre-existing halfdose view and its d0p5 view serve the SAME two LoRA bytes (the admitted repeat)
    hd = ROSTER["targets"]["nemotron3nano4b"]["views"]["halfdose"]["cells"]
    for cid, sid in _dose_views("nemotron3nano4b")["d0p5"]["cells"].items():
        assert hd[cid] == sid
    assert "dose" not in ROSTER["targets"]["nemotron3nano4b"]["views"]["halfdose"]   # not a ladder view: its cap stays
    # the override is scoped to the ladder: every non-dose view keeps the body's BoN cap, and reads as non-dose — through the ONE
    # reader `_view_bon_cap`: a caps-writing target (llama8b) holds its own cap, a view with its own `caps.bon` (the qwen35_4b
    # s1 / s2 / s3 seed-replicate views, 2026-09-15; never a dose view) holds that, everything else the body's 512
    for target, view, t, v in VIEWS:
        if "dose" not in v:
            want_cap = gen._view_bon_cap(ROSTER, target, view)
            assert gen.view_dose(ROSTER, target, view) is None and _cfg(v)["bon"]["max_new_tokens"] == want_cap, (target, view)
            if gen.view_declared_bon_cap(ROSTER, target, view) is None:
                assert want_cap == int(((ROSTER["targets"][target].get("caps") or {}).get("bon")) or 512), (target, view)


def test_generator_refuses_a_malformed_dose_view(tmp_path):
    """The dose grammar fails LOUDLY, never silently: a keyed view whose name spells another dose; a
    dose-named view with no key; a chat-checkpoint target borrowing the instruct spelling; dose 0 or a
    non-numeric dose. And the caps a dose lane once moved and must not: a body whose BoN cap is not
    512 refuses for EVERY view, and past rendering a bon block that differs from the body's at any key
    (a 4096 BoN cap included) or an attacks.target_max_new_tokens that moved refuses."""
    import copy
    import os
    pins = PINS
    assert [gen.dose_token(d) for d, _ in DOSE_LADDER] == [tok for _, tok in DOSE_LADDER]
    r = copy.deepcopy(ROSTER); r["targets"]["gemma4e4b"]["views"]["it_d0p25"]["dose"] = 0.5
    with pytest.raises(SystemExit, match="spells view `it_d0p5`, not `it_d0p25`"):
        gen.render_config(r, pins, "gemma4e4b", "it_d0p25")
    r = copy.deepcopy(ROSTER); del r["targets"]["gemma4e4b"]["views"]["it_d0p25"]["dose"]
    with pytest.raises(SystemExit, match="carries no `dose:` key"):
        gen.render_config(r, pins, "gemma4e4b", "it_d0p25")
    # the UNIT is a declared fact: absent, another unit, or a unit on a non-dose view all refuse
    r = copy.deepcopy(ROSTER); del r["targets"]["gemma4e4b"]["views"]["it_d0p25"]["dose_unit"]
    with pytest.raises(SystemExit, match="declares `dose_unit: lora_write_basis_multiplier`"):
        gen.render_config(r, pins, "gemma4e4b", "it_d0p25")
    r = copy.deepcopy(ROSTER); r["targets"]["llama8b"]["views"]["d0p25"]["dose_unit"] = "alphasteer_base_scale"
    with pytest.raises(SystemExit, match="declares `dose_unit: lora_write_basis_multiplier`"):
        gen.render_config(r, pins, "llama8b", "d0p25")
    r = copy.deepcopy(ROSTER); r["targets"]["nemotron3nano4b"]["views"]["main"]["dose_unit"] = gen.DOSE_UNIT
    with pytest.raises(SystemExit, match="a unit without a dose"):
        gen.render_config(r, pins, "nemotron3nano4b", "main")
    assert gen.DOSE_UNIT == "lora_write_basis_multiplier"
    # the FAMILY spelling is a roster fact (dose_families): the bare spelling on an instruct-direct lane, the
    # `it_` spelling on a chat-checkpoint lane, a family the target has no anchor view for, a bad list — all refuse
    assert gen.dose_families(ROSTER, "gemma4e4b") == ("it", "main") and gen.dose_families(ROSTER, "qwen35_4b") == ("it", "main")
    assert gen.dose_families(ROSTER, "nemotron3nano4b") == ("main",)
    assert gen.dose_families(ROSTER, "llama8b") == ("it", "main")
    # gemma carries the base family too, so the bare-spelling refusal is shown on a copy whose ladder is the
    # instruct family ALONE — the roster fact the spelling is checked against, never the target's name
    r = copy.deepcopy(ROSTER); r["targets"]["gemma4e4b"]["dose_families"] = ["it"]
    with pytest.raises(SystemExit, match="bare `d<tok>` spelling but this target's ladder families are"):
        gen.render_config(r, pins, "gemma4e4b", "d0p25")
    r = copy.deepcopy(ROSTER); r["targets"]["nemotron3nano4b"]["dose_families"] = ["it"]
    with pytest.raises(SystemExit, match="has no `it` view to anchor that family"):
        gen.dose_families(r, "nemotron3nano4b")
    r = copy.deepcopy(ROSTER); r["targets"]["llama8b"]["dose_families"] = ["it", "it"]
    with pytest.raises(SystemExit, match="without repeats"):
        gen.dose_families(r, "llama8b")
    r = copy.deepcopy(ROSTER); r["targets"]["llama8b"]["dose_families"] = ["it"]
    with pytest.raises(SystemExit, match="bare `d<tok>` spelling but this target's ladder families are"):
        gen.render_config(r, pins, "llama8b", "d0p25")
    r = copy.deepcopy(ROSTER)
    r["targets"]["nemotron3nano4b"]["views"]["it_d0p25"] = r["targets"]["nemotron3nano4b"]["views"].pop("d0p25")
    with pytest.raises(SystemExit, match="`it_` spelling but this target's ladder families are \\('main',\\)"):
        gen.render_config(r, pins, "nemotron3nano4b", "it_d0p25")   # a chat-checkpoint lane borrowing the instruct spelling (the family rule)
    r = copy.deepcopy(ROSTER); r["targets"]["llama8b"]["views"]["it_d0p25"]["dose"] = 0.5
    with pytest.raises(SystemExit, match="spells view `it_d0p5`, not `it_d0p25`"):   # the name<->key rule, on a both-family target
        gen.render_config(r, pins, "llama8b", "it_d0p25")
    with pytest.raises(SystemExit, match="nonzero"):
        gen.dose_token(0)
    r = copy.deepcopy(ROSTER); r["targets"]["gemma4e4b"]["views"]["it_d0p25"]["dose"] = "0.25"
    with pytest.raises(SystemExit, match="must be a number"):
        gen.render_config(r, pins, "gemma4e4b", "it_d0p25")
    # a body whose BoN cap is not 512 refuses for EVERY view — Scale's climb tooling would refuse the store
    t = ROSTER["targets"]["gemma4e4b"]
    body = (REPO / t["body_config"]).read_text()
    assert body.count("\n  max_new_tokens: 512\n") == 1
    moved = tmp_path / "body_moved.yaml"
    moved.write_text(body.replace("\n  max_new_tokens: 512\n", "\n  max_new_tokens: 256\n"))
    r = copy.deepcopy(ROSTER); r["targets"]["gemma4e4b"]["body_config"] = os.path.relpath(moved, REPO)
    for view in ("it_d0p25", "it"):
        with pytest.raises(SystemExit, match="bon.max_new_tokens is not 512"):
            gen.render_config(r, pins, "gemma4e4b", view)
    # the post-render refusal set, on the real generated texts: bon the body's for every view, the target cap the body's
    v = ROSTER["targets"]["gemma4e4b"]["views"]["it_d0p25"]
    good = (REPO / v["config"]).read_text()
    gen._refuse_bon_drift("gemma4e4b", "it_d0p25", good, body)              # the committed one passes
    with pytest.raises(SystemExit, match="bon block drifted"):
        gen._refuse_bon_drift("gemma4e4b", "it_d0p25", good.replace("  sigma: 0.4\n", "  sigma: 0.5\n"), body)
    with pytest.raises(SystemExit, match="bon block drifted"):              # the 4096 override, were it to come back, refuses
        gen._refuse_bon_drift("gemma4e4b", "it_d0p25", good.replace("  max_new_tokens: 512\n", "  max_new_tokens: 4096\n"), body)
    # both caps, against the ROSTER's wave-wide values. The literal here is 4096, not 512: main's multi-turn
    # ruling moved target_max_new_tokens, and a replace() of the old literal is a silent NO-OP that makes this
    # case pass by not mutating anything at all.
    assert good.count("  target_max_new_tokens: 4096\n") == 1 and good.count("  controls_max_new_tokens: 512\n") == 1
    for was, now in (("  target_max_new_tokens: 4096\n", "  target_max_new_tokens: 1024\n"),
                     ("  controls_max_new_tokens: 512\n", "  controls_max_new_tokens: 256\n")):
        with pytest.raises(SystemExit, match="a view may not carry its own cap"):
            gen._refuse_bon_drift("gemma4e4b", "it_d0p25", good.replace(was, now), body)
    it_good = (REPO / ROSTER["targets"]["gemma4e4b"]["views"]["it"]["config"]).read_text()
    gen._refuse_bon_drift("gemma4e4b", "it", it_good, body)


# --- 9a-bis: a dose view names its OBJECT FAMILY as well as its dose (the ReLU ladder, 2026-09-16) --------------
#
# The coordinating session's ruling: qwen35_4b's bare dose views (dm1 .. d1p5) carry ONLY LoRA cells and are already
# measured; a psr cell added to them would re-serve those measured rows against a new same-pass anchor and leave two
# passes per dose in the record. So a view's name carries its object family (`d0p25_relu`), the segment is REQUIRED for
# every new view (the bare spelling is a frozen, closed allow-list), the roster declares the family beside the dose, and
# a view refuses a cell of another family. Every refusal below is shown on a KNOWN-BAD input, never the happy path alone.

RELU_CELL = "constv2-mix-full__psr_ce_all_minus_ctl_distilled"     # the ReLU (PSR) difference object's cell id
LORA_CELL = "constv2-mix-full__lora_r4_ce"                          # the LoRA ladders' constitution cell


def _with_family_view(target, view, dose, *, family, cells, declare=True, unit=None):
    """A roster copy with ONE added dose view of an object family — synthetic staged ids (`d<tok>__<cell>`), nothing pinned
    or staged (the header renders an unpinned id as such); `declare=False` leaves the `object_family:` key off. The unit is
    the FAMILY's (gen.DOSE_UNITS; the LoRA unit for a family the generator does not know, which is refused by name before its
    unit is read) unless `unit=` names another — the known-bad input of the unit test."""
    r = copy.deepcopy(ROSTER)
    v = {"dose": dose, "dose_unit": unit if unit is not None else gen.DOSE_UNITS.get(family, gen.DOSE_UNIT),
         "config": f"configs/sdfbase_full22_{target}_{view}.yaml",
         "manifest": f"configs/capeval/manifests/petri_full22_{target}_{view}.yaml",
         "cells": {c: f"d{gen.dose_token(dose)}__{c}" for c in cells}}
    if declare:
        v["object_family"] = family
    r["targets"][target]["views"][view] = v
    return r


def test_dose_view_names_its_object_family_and_the_roster_agrees():
    """A `d<tok>_relu` view parses in both spellings, its dose is read from the name and held to the `dose:` key exactly as a
    bare view's is, `object_family: relu` agreeing with the name is accepted and reads back, and the view renders end to end
    (config AND manifest) — a ReLU rung on the difference object, qwen35_4b's base->instruct family — with a header naming
    the family and the caps untouched. The frozen bare views read as `lora` without declaring anything."""
    for view, dose in (("d0p25_relu", 0.25), ("dm1_relu", -1.0), ("d1p5_relu", 1.5), ("it_d1p5_relu", 1.5), ("d2_relu", 2)):
        m = gen._DOSE_VIEW_RE.match(view)
        assert m and m.group("objfam") == "relu" and m.group(2) == gen.dose_token(dose), view
        r = _with_family_view("qwen35_4b", view, dose, family="relu", cells=[RELU_CELL])
        assert gen.view_dose(r, "qwen35_4b", view) == dose
        assert gen.view_object_family(r, "qwen35_4b", view) == "relu"
        assert fr.served_cells(r, "qwen35_4b", view) == {RELU_CELL: f"d{gen.dose_token(dose)}__{RELU_CELL}"}
    # a SYNTHETIC rung — d2_relu, a dose the roster does not carry, so its `d2__` id is pinned nowhere — renders end to end and
    # says so (the header renders an unpinned id as UNPINNED; d0p25's id is a real pin since the ladder landed, see below)
    r = _with_family_view("qwen35_4b", "d2_relu", 2, family="relu", cells=[RELU_CELL])
    assert f"d2__{RELU_CELL}" not in PINS["qwen35_4b"]
    text = gen.render_config(r, PINS, "qwen35_4b", "d2_relu")
    cfg = yaml.safe_load(text)
    assert cfg[EXPLICIT_CELLS_KEY] == ["none", RELU_CELL]
    assert cfg["defenses_dir"] == "out/full22/qwen35_4b/cells_view/d2_relu"
    assert cfg["out_dir"] == "out/sdfbase_full22_qwen35_4b_d2_relu" and cfg["private_run"] == "sdfbase_full22_qwen35_4b_d2_relu"
    assert "a DOSE view (dose 2;" in text and "An OBJECT-FAMILY view (`relu`" in text and "is a `psr_` cell id" in text
    assert f"{RELU_CELL} <- d2__{RELU_CELL}" in text and "UNPINNED" in text          # synthetic: nothing staged, said so
    assert cfg["bon"]["max_new_tokens"] == gen.BON_MAX_NEW_TOKENS == 512             # the BoN cap holds on a family view too
    assert cfg["attacks"]["target_max_new_tokens"] == ROSTER["target_max_new_tokens"] == 4096
    spec_text = gen.render_manifest(r, "qwen35_4b", "d2_relu")
    assert f"petri_full22_qwen35_4b_d2_relu" in spec_text and gen.harness_cell_id(RELU_CELL) in spec_text
    # the PRODUCTION rung at 0.25 is pinned (the four scaled copies were pinned when the ladder landed), so its header echoes
    # the pin's sha and never UNPINNED — what the generator writes, whatever the committed file currently carries
    text = gen.render_config(ROSTER, PINS, "qwen35_4b", "d0p25_relu")
    assert f"{RELU_CELL} <- d0p25__{RELU_CELL}" in text and PINS["qwen35_4b"][f"d0p25__{RELU_CELL}"]["object_sha256"] in text
    assert "UNPINNED" not in text and "a DOSE view (dose 0.25;" in text and "An OBJECT-FAMILY view (`relu`" in text
    # the name<->key dose rule is unchanged under the segment: the spelled name carries it
    r = _with_family_view("qwen35_4b", "d0p25_relu", 0.5, family="relu", cells=[RELU_CELL])
    with pytest.raises(SystemExit, match="dose 0.5 spells view `d0p5_relu`, not `d0p25_relu`"):
        gen.view_dose(r, "qwen35_4b", "d0p25_relu")
    # ... and so is the ladder-family spelling rule: an `it_` family view on a base-only target
    r = _with_family_view("nemotron3nano4b", "it_d0p25_relu", 0.25, family="relu", cells=[RELU_CELL])
    with pytest.raises(SystemExit, match="`it_` spelling but this target's ladder families are \\('main',\\)"):
        gen.view_dose(r, "nemotron3nano4b", "it_d0p25_relu")
    # the frozen bare views are the LoRA ladder, implicitly, and none declares a family (a front-door entry the roster does not carry yet is
    # held to the models table in test_generator_refuses_a_new_bare_dose_view_..., not here)
    for target, views in gen.BARE_DOSE_VIEWS_FROZEN.items():
        for view in views:
            if target not in ROSTER["targets"] or view not in ROSTER["targets"][target]["views"]:
                assert target in gen.FRONT_DOOR_MODELS, (target, view)   # a front-door dose view the roster does not carry yet (05 stages u0 first)
                continue
            assert gen.view_object_family(ROSTER, target, view) == "lora", (target, view)
            assert "object_family" not in ROSTER["targets"][target]["views"][view], (target, view)
    # ctl-neg1 (2026-09-25) added two SEEDLESS families for the batch-64 seed-replicate control parents at strength -1:
    # `ctllora64` / `ctlrelu64`, whose views carry the fit seed as a trailing `s<k>` that `family_of_segment` strips
    # (one family at three seeds). Their UNITS are the existing two and their predicates admit the constv2ctl2 control
    # cells ONLY -- a constitution or a difference cell can never ride a control-family view.
    assert set(gen.OBJECT_FAMILIES) == {"lora", "relu", "ctllora64", "ctlrelu64"}
    CTL_LORA, CTL_RELU = "constv2ctl2-full__lora_r4_ce", "constv2ctl2-full__psr_ce_all"
    assert gen.OBJECT_FAMILIES["ctllora64"][0](CTL_LORA) and not gen.OBJECT_FAMILIES["ctllora64"][0](CTL_RELU)
    assert gen.OBJECT_FAMILIES["ctlrelu64"][0](CTL_RELU) and not gen.OBJECT_FAMILIES["ctlrelu64"][0](CTL_LORA)
    assert not gen.OBJECT_FAMILIES["ctllora64"][0](LORA_CELL) and not gen.OBJECT_FAMILIES["ctlrelu64"][0](RELU_CELL)   # not the constitution / difference cells
    for k in (1, 2, 3):
        assert gen.family_of_segment(f"ctllora64s{k}") == "ctllora64" and gen.family_of_segment(f"ctlrelu64s{k}") == "ctlrelu64"
    assert gen.family_of_segment("lora") == "lora" and gen.family_of_segment("relu") == "relu"
    assert gen.family_of_segment("loradiff64s1") == "loradiff64s1"     # an unknown stem is never truncated into something else
    assert gen.OBJECT_FAMILIES["lora"][0](LORA_CELL) and not gen.OBJECT_FAMILIES["lora"][0](RELU_CELL)
    assert gen.OBJECT_FAMILIES["relu"][0](RELU_CELL) and not gen.OBJECT_FAMILIES["relu"][0](LORA_CELL)


def test_generator_refuses_an_object_family_the_name_and_roster_disagree_on():
    """KNOWN-BAD inputs, each refused by name: a `_relu` view with no `object_family:`; one declaring `lora`; a family this
    generator does not know, spelled in the name and declared in the key; `object_family:` on a frozen bare view (its family
    is the allow-list's, not a key's) and on a non-dose view."""
    r = _with_family_view("qwen35_4b", "d0p25_relu", 0.25, family="relu", cells=[RELU_CELL], declare=False)
    with pytest.raises(SystemExit, match="names object family `relu` but the view declares no `object_family:`"):
        gen.view_dose(r, "qwen35_4b", "d0p25_relu")
    r = _with_family_view("qwen35_4b", "d0p25_relu", 0.25, family="lora", cells=[RELU_CELL])
    with pytest.raises(SystemExit, match="names object family `relu` but the view declares `object_family: lora` — the two must agree"):
        gen.render_config(r, PINS, "qwen35_4b", "d0p25_relu")
    r = _with_family_view("qwen35_4b", "d0p25_caa", 0.25, family="caa", cells=[RELU_CELL])
    with pytest.raises(SystemExit, match="object family `caa` is not one this generator knows"):
        gen.view_dose(r, "qwen35_4b", "d0p25_caa")
    r = _with_family_view("qwen35_4b", "d0p25_relu", 0.25, family="caa", cells=[RELU_CELL])
    with pytest.raises(SystemExit, match="names object family `relu` but the view declares `object_family: caa`"):
        gen.view_dose(r, "qwen35_4b", "d0p25_relu")
    r = copy.deepcopy(ROSTER); r["targets"]["qwen35_4b"]["views"]["d0p25"]["object_family"] = "lora"
    with pytest.raises(SystemExit, match="qwen35_4b/d0p25: declares `object_family: lora` but its name carries no object-family segment"):
        gen.render_config(r, PINS, "qwen35_4b", "d0p25")
    r = copy.deepcopy(ROSTER); r["targets"]["qwen35_4b"]["views"]["main"]["object_family"] = "lora"
    with pytest.raises(SystemExit, match="qwen35_4b/main: carries `object_family:` but no `dose:`"):
        gen.render_config(r, PINS, "qwen35_4b", "main")


def test_generator_refuses_a_dose_unit_that_is_not_the_views_object_familys():
    """The UNIT is the FAMILY's (DOSE_UNITS: a LoRA dose multiplies the write basis, a ReLU dose the direction norm — two axes,
    never one), so a view declaring the OTHER family's unit refuses, on KNOWN-BAD inputs both ways: a `_relu` view with the LoRA
    unit, a `_lora` view and a FROZEN bare view (`lora` by the allow-list) with the ReLU unit, a `_relu` view with no unit. Every
    family names a unit; the production roster's six relu views declare the ReLU unit and every frozen bare view the LoRA unit.
    And the ORDER (2026-09-17): the ladder-family, frozen-list and dose-spelling refusals are more specific than the unit's and
    win over it — shown with the wrong unit on each — so a wrong unit never masks a wrong spelling, and an `it_` view on a
    chat-checkpoint target is told the target has no instruct family, not to name an object family (the regression the
    per-family unit check introduced by hoisting the frozen-list check in front of the ladder-family one)."""
    LORA_UNIT, RELU_UNIT = gen.DOSE_UNITS["lora"], gen.DOSE_UNITS["relu"]
    assert gen.DOSE_UNITS == {"lora": "lora_write_basis_multiplier", "relu": "psr_direction_norm_multiplier",
                              "ctllora64": "lora_write_basis_multiplier", "ctlrelu64": "psr_direction_norm_multiplier"}
    # every family's unit is one of the two known axes -- a new family may not introduce a third
    assert set(gen.DOSE_UNITS.values()) == {LORA_UNIT, RELU_UNIT}
    assert set(gen.DOSE_UNITS) == set(gen.OBJECT_FAMILIES) and gen.DOSE_UNIT == LORA_UNIT
    r = _with_family_view("qwen35_4b", "d0p25_relu", 0.25, family="relu", cells=[RELU_CELL], unit=LORA_UNIT)
    with pytest.raises(SystemExit, match="qwen35_4b/d0p25_relu: a dose view declares `dose_unit: psr_direction_norm_multiplier` "
                                         "\\(got 'lora_write_basis_multiplier'\\)"):
        gen.render_config(r, PINS, "qwen35_4b", "d0p25_relu")
    r = _with_family_view("qwen35_4b", "d2_lora", 2, family="lora", cells=[LORA_CELL], unit=RELU_UNIT)
    with pytest.raises(SystemExit, match="qwen35_4b/d2_lora: a dose view declares `dose_unit: lora_write_basis_multiplier` "
                                         "\\(got 'psr_direction_norm_multiplier'\\)"):
        gen.view_dose(r, "qwen35_4b", "d2_lora")
    r = copy.deepcopy(ROSTER); r["targets"]["qwen35_4b"]["views"]["d0p25"]["dose_unit"] = RELU_UNIT    # a frozen bare view IS `lora`
    with pytest.raises(SystemExit, match="qwen35_4b/d0p25: a dose view declares `dose_unit: lora_write_basis_multiplier` "
                                         "\\(got 'psr_direction_norm_multiplier'\\)"):
        gen.render_config(r, PINS, "qwen35_4b", "d0p25")
    r = _with_family_view("qwen35_4b", "d0p25_relu", 0.25, family="relu", cells=[RELU_CELL])
    del r["targets"]["qwen35_4b"]["views"]["d0p25_relu"]["dose_unit"]
    with pytest.raises(SystemExit, match="qwen35_4b/d0p25_relu: a dose view declares `dose_unit: psr_direction_norm_multiplier` \\(got None\\)"):
        gen.view_dose(r, "qwen35_4b", "d0p25_relu")
    # the same views with THEIR family's unit are admitted
    assert gen.view_dose(_with_family_view("qwen35_4b", "d0p25_relu", 0.25, family="relu", cells=[RELU_CELL], unit=RELU_UNIT),
                         "qwen35_4b", "d0p25_relu") == 0.25
    assert gen.view_dose(_with_family_view("qwen35_4b", "d2_lora", 2, family="lora", cells=[LORA_CELL], unit=LORA_UNIT),
                         "qwen35_4b", "d2_lora") == 2
    # the production roster: every relu view the ReLU unit, every frozen bare view the LoRA unit
    relu_views = [vn for vn, v in ROSTER["targets"]["qwen35_4b"]["views"].items() if "dose" in v and vn.endswith("_relu")]
    assert relu_views == [f"d{tok}_relu" for _, tok in DOSE_LADDER]
    for vn in relu_views:
        assert ROSTER["targets"]["qwen35_4b"]["views"][vn]["dose_unit"] == RELU_UNIT, vn
        assert gen.view_object_family(ROSTER, "qwen35_4b", vn) == "relu", vn
    for target, views in gen.BARE_DOSE_VIEWS_FROZEN.items():
        for vn in views:
            if target not in ROSTER["targets"] or vn not in ROSTER["targets"][target]["views"]:
                assert target in gen.FRONT_DOOR_MODELS, (target, vn)   # a front-door dose view the roster does not carry yet (held to the table elsewhere)
                continue
            assert ROSTER["targets"][target]["views"][vn]["dose_unit"] == LORA_UNIT, (target, vn)
    # ORDER: each more specific refusal wins over the unit's, with the wrong unit present on every input
    r = _with_family_view("nemotron3nano4b", "it_d0p25_relu", 0.25, family="relu", cells=[RELU_CELL], unit=LORA_UNIT)
    with pytest.raises(SystemExit, match="nemotron3nano4b/it_d0p25_relu: `it_` spelling but this target's ladder families are \\('main',\\)"):
        gen.view_dose(r, "nemotron3nano4b", "it_d0p25_relu")
    r = copy.deepcopy(ROSTER)           # the regression's exact input, plus the wrong unit: still the ladder-family refusal
    r["targets"]["nemotron3nano4b"]["views"]["it_d0p25"] = r["targets"]["nemotron3nano4b"]["views"].pop("d0p25")
    r["targets"]["nemotron3nano4b"]["views"]["it_d0p25"]["dose_unit"] = RELU_UNIT
    with pytest.raises(SystemExit, match="nemotron3nano4b/it_d0p25: `it_` spelling but this target's ladder families are \\('main',\\)"):
        gen.render_config(r, PINS, "nemotron3nano4b", "it_d0p25")
    r = copy.deepcopy(ROSTER); r["targets"]["gemma4e4b"]["dose_families"] = ["it"]
    r["targets"]["gemma4e4b"]["views"]["d0p25"]["dose_unit"] = RELU_UNIT
    with pytest.raises(SystemExit, match="gemma4e4b/d0p25: bare `d<tok>` spelling but this target's ladder families are \\('it',\\)"):
        gen.view_dose(r, "gemma4e4b", "d0p25")
    r = _with_family_view("qwen35_4b", "d0p25_relu", 0.5, family="relu", cells=[RELU_CELL], unit=LORA_UNIT)
    with pytest.raises(SystemExit, match="dose 0.5 spells view `d0p5_relu`, not `d0p25_relu`"):
        gen.view_dose(r, "qwen35_4b", "d0p25_relu")
    r = _with_family_view("qwen35_4b", "d2", 2, family="lora", cells=[LORA_CELL], declare=False, unit=RELU_UNIT)
    with pytest.raises(SystemExit, match="qwen35_4b/d2: a NEW bare dose view"):
        gen.view_dose(r, "qwen35_4b", "d2")
    r = _with_family_view("qwen35_4b", "d0p25_relu", 0.25, family="relu", cells=[RELU_CELL, LORA_CELL], unit=LORA_UNIT)
    with pytest.raises(SystemExit, match=f"qwen35_4b/d0p25_relu: object family `relu` but serves `{LORA_CELL}`"):
        gen.view_dose(r, "qwen35_4b", "d0p25_relu")


def test_generator_refuses_a_new_bare_dose_view_and_the_allow_list_is_the_roster_as_frozen():
    """REQUIRED-FOR-NEW: a bare dose view not on BARE_DOSE_VIEWS_FROZEN refuses, telling the author to name the object family
    in the view name — a well-formed LoRA rung (qwen35_4b's d2, the very spelling gemma's extension carries; nemotron's d2)
    as much as a ReLU one. The allow-list is EXACTLY the production roster's bare dose views, held equal here so neither can
    grow without the other, and every one of the fifty still parses to its declared dose."""
    bare = {target: tuple(vn for vn, v in t["views"].items()
                          if "dose" in v and gen._DOSE_VIEW_RE.match(vn).group("objfam") is None)
            for target, t in ROSTER["targets"].items()}
    # the FRONT DOOR's entries (make_full22_configs.front_door_models) are held to the MODELS TABLE — a
    # bare LoRA dose view the table declares for the key (dose_unit lora_write_basis_multiplier, no object_family) — and to the roster too
    # once stage 05 lands the block (ROSTER-CONDITIONAL, the fix-1 owner follow-up 10); the paper's five stay exactly the roster's
    front_door = {k for k in gen.BARE_DOSE_VIEWS_FROZEN if k in gen.FRONT_DOOR_MODELS}
    for k in front_door:
        views = gen.FRONT_DOOR_MODELS[k].get("views") or {}
        assert gen.BARE_DOSE_VIEWS_FROZEN[k] == tuple(vn for vn, v in views.items() if "dose" in v and "object_family" not in v
                                                     and gen._DOSE_VIEW_RE.match(vn) and gen._DOSE_VIEW_RE.match(vn).group("objfam") is None), k
        assert all(views[vn]["dose_unit"] == gen.DOSE_UNIT for vn in gen.BARE_DOSE_VIEWS_FROZEN[k]), k
    assert {k: v for k, v in bare.items() if k not in front_door} == {k: v for k, v in gen.BARE_DOSE_VIEWS_FROZEN.items() if k not in front_door}
    # ... a carried front-door target's bare dose views are EXACTLY the frozen ones its roster block carries so far (05 stages u0 first;
    # a dose view appears once 03/04 stage its objects) — never a bare view outside the frozen list
    assert {k: v for k, v in bare.items() if k in front_door} == {k: tuple(vn for vn in v if vn in ROSTER["targets"][k]["views"])
                                                                for k, v in gen.BARE_DOSE_VIEWS_FROZEN.items() if k in front_door and k in bare}
    assert sum(len(v) for k, v in bare.items() if k not in front_door) == 50 == 14 + 12 + 6 + 12 + 6
    for target, views in bare.items():
        for view in views:
            assert gen.view_dose(ROSTER, target, view) == ROSTER["targets"][target]["views"][view]["dose"], (target, view)
    # a NEW bare view — qwen's d2, well-formed in every other way (dose, unit, spelling, LoRA cells): refused at the name
    r = _with_family_view("qwen35_4b", "d2", 2, family="lora", cells=[LORA_CELL], declare=False)
    with pytest.raises(SystemExit, match="qwen35_4b/d2: a NEW bare dose view .* name the object family in the view name \\(`d2_lora`, `d2_relu`\\)"):
        gen.render_config(r, PINS, "qwen35_4b", "d2")
    r = _with_family_view("qwen35_4b", "d2", 2, family="lora", cells=[RELU_CELL], declare=False)    # ... ReLU cells, the same
    with pytest.raises(SystemExit, match="name the object family in the view name"):
        gen.view_dose(r, "qwen35_4b", "d2")
    r = _with_family_view("qwen35_4b", "d2", 2, family="lora", cells=[LORA_CELL])    # declaring the key does not open the bare spelling
    with pytest.raises(SystemExit, match="name the object family in the view name"):
        gen.view_dose(r, "qwen35_4b", "d2")
    r = _with_family_view("nemotron3nano4b", "d2", 2, family="lora", cells=[LORA_CELL], declare=False)
    with pytest.raises(SystemExit, match="nemotron3nano4b/d2: a NEW bare dose view"):
        gen.view_dose(r, "nemotron3nano4b", "d2")
    r = _with_family_view("qwen35_4b", "it_d2", 2, family="lora", cells=[LORA_CELL], declare=False)   # the instruct spelling too
    with pytest.raises(SystemExit, match="qwen35_4b/it_d2: a NEW bare dose view"):
        gen.view_dose(r, "qwen35_4b", "it_d2")
    # ... and the SAME rung, named: admitted, as `lora`
    r = _with_family_view("qwen35_4b", "d2_lora", 2, family="lora", cells=[LORA_CELL])
    assert gen.view_dose(r, "qwen35_4b", "d2_lora") == 2 and gen.view_object_family(r, "qwen35_4b", "d2_lora") == "lora"
    # a frozen bare view taken OFF the allow-list refuses like a new one — the list, not the roster, admits the spelling
    frozen = {k: (tuple(x for x in v if x != "d0p25") if k == "qwen35_4b" else v) for k, v in gen.BARE_DOSE_VIEWS_FROZEN.items()}
    orig = gen.BARE_DOSE_VIEWS_FROZEN
    gen.BARE_DOSE_VIEWS_FROZEN = frozen
    try:
        with pytest.raises(SystemExit, match="qwen35_4b/d0p25: a NEW bare dose view"):
            gen.view_dose(ROSTER, "qwen35_4b", "d0p25")
    finally:
        gen.BARE_DOSE_VIEWS_FROZEN = orig
    assert gen.view_dose(ROSTER, "qwen35_4b", "d0p25") == 0.25


def test_every_frozen_bare_dose_view_still_renders_byte_identically():
    """Every view on the frozen allow-list renders — config AND manifest — byte-identical to the committed files today (the
    generator's --check, narrowed to the fifty and asserted in-process), so the family grammar changed nothing about a
    measured view: no header line, no key, no cell."""
    for target, views in gen.BARE_DOSE_VIEWS_FROZEN.items():
        for view in views:
            if target not in ROSTER["targets"] or view not in ROSTER["targets"][target]["views"]:
                assert target in gen.FRONT_DOOR_MODELS, (target, view)   # a front-door dose view the roster does not carry yet: no committed file to hold (item 10)
                continue
            v = ROSTER["targets"][target]["views"][view]
            assert gen.render_config(ROSTER, PINS, target, view) == (REPO / v["config"]).read_text(encoding="utf-8"), (target, view)
            assert gen.render_manifest(ROSTER, target, view) == (REPO / v["manifest"]).read_text(encoding="utf-8"), (target, view)
            assert "OBJECT-FAMILY view" not in (REPO / v["config"]).read_text(encoding="utf-8"), (target, view)


def test_generator_refuses_a_dose_view_that_mixes_object_families():
    """FAMILY PURITY — the contamination the grammar exists to prevent — on KNOWN-BAD inputs: a `relu` view carrying a
    `lora_r4` cell; a `lora` view carrying a `psr_` cell; a FROZEN bare view (qwen35_4b's d0p25: measured, LoRA-only)
    gaining the ReLU difference cell, which would re-serve its measured rows against a new same-pass anchor; a prompted
    baseline in a family view; and a cell arriving through an ENABLED LANE, which is a served cell like any other."""
    r = _with_family_view("qwen35_4b", "d0p25_relu", 0.25, family="relu", cells=[RELU_CELL, LORA_CELL])
    with pytest.raises(SystemExit, match=f"qwen35_4b/d0p25_relu: object family `relu` but serves `{LORA_CELL}`, which is not a `psr_` cell id"):
        gen.render_config(r, PINS, "qwen35_4b", "d0p25_relu")
    r = _with_family_view("qwen35_4b", "d2_lora", 2, family="lora", cells=[LORA_CELL, RELU_CELL])
    with pytest.raises(SystemExit, match=f"qwen35_4b/d2_lora: object family `lora` but serves `{RELU_CELL}`, which is not a `lora_r4` cell id"):
        gen.view_dose(r, "qwen35_4b", "d2_lora")
    r = copy.deepcopy(ROSTER)
    r["targets"]["qwen35_4b"]["views"]["d0p25"]["cells"][RELU_CELL] = f"d0p25__{RELU_CELL}"      # THE forbidden edit
    with pytest.raises(SystemExit, match=f"qwen35_4b/d0p25: object family `lora` but serves `{RELU_CELL}`, which is not a `lora_r4` cell id"):
        gen.render_config(r, PINS, "qwen35_4b", "d0p25")
    r = _with_family_view("qwen35_4b", "d0p25_relu", 0.25, family="relu", cells=[RELU_CELL])
    r["targets"]["qwen35_4b"]["views"]["d0p25_relu"]["cells"]["prompted__safety_refusal"] = None
    with pytest.raises(SystemExit, match="serves `prompted__safety_refusal`, which is not a `psr_` cell id"):
        gen.view_dose(r, "qwen35_4b", "d0p25_relu")
    # a lane cell: an enabled lane mapped onto the family view carries a LoRA cell into it
    r = _with_family_view("qwen35_4b", "d0p25_relu", 0.25, family="relu", cells=[RELU_CELL])
    r["lane_flags"]["relulane"] = {"enabled": True, "view": "d0p25_relu"}
    r["targets"]["qwen35_4b"].setdefault("lanes", {})["relulane"] = {LORA_CELL: f"lane__{LORA_CELL}"}
    assert LORA_CELL in fr.served_cells(r, "qwen35_4b", "d0p25_relu")
    with pytest.raises(SystemExit, match=f"object family `relu` but serves `{LORA_CELL}`"):
        gen.view_dose(r, "qwen35_4b", "d0p25_relu")
    r["lane_flags"]["relulane"]["enabled"] = False      # a disabled lane serves nothing, so nothing to refuse
    assert gen.view_dose(r, "qwen35_4b", "d0p25_relu") == 0.25


RELU_DOSE_VIEWS = [f"d{tok}_relu" for _, tok in DOSE_LADDER]      # qwen35_4b's ReLU difference ladder (2026-09-16), ladder order


def test_relu_dose_views_serve_exactly_the_psr_difference_object_at_that_dose_and_no_other_cell():
    """The ReLU (PSR) DIFFERENCE dose ladder: six `d<tok>_relu` views on qwen35_4b — after its LoRA
    ladders, and on no other target — each serving ONLY `constv2-mix-full__psr_ce_all_minus_ctl_distilled` at that dose against
    its own same-pass `none` anchor, in the ReLU unit (the direction norm, DOSE_UNITS['relu']), declared `object_family: relu`.
    Dose 1.0 RE-SERVES the unscaled `derived__` object — the SAME staged id as the campaign's `main` row; 0.5 the
    `halfdose__` bytes; every other dose its own `d<tok>__` scaled copy whose pin names the derived parent
    (scale_psr_dose.py). Every id pinned, sidecar-matched and verified on both clusters; six different objects; no prompted cell;
    no staged id shared with the LoRA ladders; the generated pair carries `none` + the one cell, the family in its header and
    the caps untouched; the job-side count guard reads 1 on both references."""
    t = ROSTER["targets"]["qwen35_4b"]
    relu = {vn: v for vn, v in _dose_views("qwen35_4b").items() if gen._DOSE_VIEW_RE.match(vn).group("objfam") is not None}
    assert list(relu) == RELU_DOSE_VIEWS and list(_dose_views("qwen35_4b"))[-6:] == RELU_DOSE_VIEWS
    for target in ROSTER["targets"]:      # the ONLY object-family views in the roster (+ a front-door target's own `_relu` twin, from the models table)
        others = [vn for vn in _dose_views(target) if gen._DOSE_VIEW_RE.match(vn).group("objfam") is not None]
        if target in gen.FRONT_DOOR_MODELS:
            table_relu = [vn for vn, v in (gen.FRONT_DOOR_MODELS[target].get("views") or {}).items() if "dose" in v and v.get("object_family")]
            assert others == [vn for vn in table_relu if vn in ROSTER["targets"][target]["views"]], target
            continue
        assert others == (RELU_DOSE_VIEWS if target == "qwen35_4b" else []), target
    parent_sha = PINS["qwen35_4b"][f"derived__{RELU_CELL}"]["object_sha256"]
    shas = []
    for (dose, tok), (vn, v) in zip(DOSE_LADDER, relu.items()):
        assert v["dose"] == dose and gen.dose_token(dose) == tok and vn == f"d{tok}_relu", vn
        assert v["dose_unit"] == gen.DOSE_UNITS["relu"] == "psr_direction_norm_multiplier" and v["object_family"] == "relu", vn
        assert gen.view_dose(ROSTER, "qwen35_4b", vn) == dose and gen.view_object_family(ROSTER, "qwen35_4b", vn) == "relu"
        cells = v["cells"]
        served, staged = fr.served_cells(ROSTER, "qwen35_4b", vn), fr.staged_cells(ROSTER, "qwen35_4b", vn)
        assert served == staged == cells and list(cells) == [RELU_CELL] and not any(fr.is_prompted(c) for c in served), vn
        if dose == 1.0:        # The campaign's own unscaled derived object, under the SAME staged id as the main row
            assert cells == {RELU_CELL: t["views"]["main"]["cells"][RELU_CELL]} == {RELU_CELL: f"derived__{RELU_CELL}"}, vn
        elif dose == 0.5:      # The halfdose bytes
            assert cells == {RELU_CELL: f"halfdose__{RELU_CELL}"}, vn
        else:                  # a dose-scaled copy, lane-prefixed by the dose token
            assert cells == {RELU_CELL: f"d{tok}__{RELU_CELL}"}, vn
        pin = PINS["qwen35_4b"][cells[RELU_CELL]]
        assert len(pin["object_sha256"]) == 64 and pin["verified_cluster_a"] and pin["verified_cluster_b"], vn
        assert pin["sidecar_object_sha256_recorded"] == "match", vn
        if dose == 1.0:
            assert pin["object_sha256"] == parent_sha
        else:
            assert pin["object_sha256"] != parent_sha, vn
            if dose != 0.5:    # the scaled copies name their parent — the derived object's bytes — and the scaler
                assert f"dose-scaled copy x{dose:g}" in pin["source"] and f"parent object_sha256 {parent_sha}" in pin["source"], vn
                assert "scale_psr_dose.py" in pin["source"], vn
        shas.append(pin["object_sha256"])
        # the generated pair: the one cell beside `none`, the family in the header, the campaign's caps
        cfg = _cfg(v)
        assert cfg[EXPLICIT_CELLS_KEY] == ["none", RELU_CELL] and cfg["defenses_dir"] == f"out/full22/qwen35_4b/cells_view/{vn}"
        assert cfg["bon"]["max_new_tokens"] == gen.BON_MAX_NEW_TOKENS == 512
        assert cfg["attacks"]["target_max_new_tokens"] == ROSTER["target_max_new_tokens"] == 4096
        assert cfg["attacks"]["controls_max_new_tokens"] == 512 and cfg["attacks"]["static_max_new_tokens"] == 4096
        assert (cfg["model"], cfg["model_revision"]) == (t["model"], t["model_revision"])
        text = (REPO / v["config"]).read_text()
        assert f"a DOSE view (dose {dose:g};" in text and "An OBJECT-FAMILY view (`relu`" in text and "is a `psr_` cell id" in text
        assert f"{RELU_CELL} <- {cells[RELU_CELL]}" in text and "OVERRIDDEN" not in text
        spec = cm.load_petri_spec(REPO / v["manifest"])
        assert spec.cells == ["none", gen.harness_cell_id(RELU_CELL)] and spec.audit == f"petri_full22_qwen35_4b_{vn}"
        assert (spec.target_model, spec.target_revision) == (t["model"], t["model_revision"])
        for extra in ((), ("--staged",)):
            rc = subprocess.run([sys.executable, str(F22 / "roster_cell_count.py"), "qwen35_4b", vn, *extra],
                                capture_output=True, text=True)
            assert rc.returncode == 0 and int(rc.stdout.strip()) == 1, (vn, extra, rc.stderr)
    assert len(shas) == len(set(shas)) == 6       # every dose is DIFFERENT bytes, 1.0 and 0.5 included
    # the ReLU ladder and the LoRA ladders share no staged id: a relu view never borrows a LoRA rung and vice versa
    lora_ids = {sid for vn, v in _dose_views("qwen35_4b").items() if vn not in relu for sid in v["cells"].values()}
    assert lora_ids and not (lora_ids & {v["cells"][RELU_CELL] for v in relu.values()})


# --- 9b: llama8b — TWO ladder families, ONE served instruct model, every rung a lane-owned id -----------

#: the parent identities THIS lane consumed (sha256 of object.pt, verified against each sidecar's object_sha256 on
#: cluster A 2026-09-11 from $SHARED/exp/objects/sdfbase/llama8b_full/): the fit target each records, and the bytes the
#: 1.0 anchor must equal. Pinned here so a re-fit that changed the parents cannot be laddered silently.
LLAMA_PARENTS = {
    # (family, cell id): (sha256, fit model, fit revision)
    ("it", "constv2-mix-full__lora_r4_ce"): ("529788ca747ea4060bffee0d46c4ccf1dcf763c548111268547dca0004614c8c",
                                            "NousResearch/Meta-Llama-3.1-8B-Instruct", "d10aef7999a2b5ba950ab3974312feeedbfe0b77"),
    ("it", "constv2-mix-full__lora_r4_ce_minus_ctl"): ("e85220c9e5d2b9aa4377fea7a9a590048c693118975be9f36ae579b0eeff6f13",
                                                      "NousResearch/Meta-Llama-3.1-8B-Instruct", "d10aef7999a2b5ba950ab3974312feeedbfe0b77"),
    ("main", "constv2-mix-full__lora_r4_ce"): ("504b6651facd296608af8e7441b71a51ec5f1062666e4d21f79bf7f91fa8f9b0",
                                              "NousResearch/Meta-Llama-3.1-8B", "1f47e50cdbe801ad8a5174156ec3a0655108fb9f"),
    ("main", "constv2-mix-full__lora_r4_ce_minus_ctl"): ("4375b67a0fb0621e9ee296041d0842cb24294112e18424a803e8e54e9538a740",
                                                        "NousResearch/Meta-Llama-3.1-8B", "1f47e50cdbe801ad8a5174156ec3a0655108fb9f"),
    # the CONTROL LoRA of the base family (the base->instruct sweep's third object), the campaign's own base__ fit
    ("main", "constv2ctl2-full__lora_r4_ce"): ("56254ce4ec90bf041070a272529762b87d9cca8fcd4cbc47ae38486f12e70d8e",
                                                "NousResearch/Meta-Llama-3.1-8B", "1f47e50cdbe801ad8a5174156ec3a0655108fb9f"),
}
LLAMA_SERVED = ("NousResearch/Meta-Llama-3.1-8B-Instruct", "d10aef7999a2b5ba950ab3974312feeedbfe0b77")
#: the Llama target's own staged ids for the two constitution LoRAs (staged 2026-09-11 18:14Z, both
#: clusters): the anchor pins must equal them sha for sha — two ids, one object
LLAMA_LANE_LORA_IDS = {"it": "itdoc__constv2-mix-full__lora_r4_ce", "main": "base__constv2-mix-full__lora_r4_ce"}


def test_llama8b_ladder_two_families_one_instruct_target_every_rung_lane_owned():
    """The fourth paper family. Twelve dose views on ONE target: the
    instruct-direct family `it_d<tok>` (the `llama8b_it_doc` parents served natively) and the
    base-fit family `d<tok>` (the `llama8b_base` parents TRANSFERRED onto the instruct target). Both families' configs
    and manifests pin the INSTRUCT model and revision — the generator derives the pin from the target, never from an
    object bank — and differ only in which objects they load. Every rung, 1.0 and 0.5 included, is a lane-owned staged
    id (no repeated id needed): the 1.0 anchor is the parent's own bytes (pin sha == the verified parent sha == the
    Llama target's own staged copy of that parent, where one exists), every other rung is different bytes, every pin
    verified on both clusters, `dose_unit` declared on every view, and the frontier's caps untouched."""
    t = ROSTER["targets"]["llama8b"]
    assert (t["model"], t["model_revision"]) == LLAMA_SERVED
    assert t["dose_families"] == ["it", "main"] and gen.dose_families(ROSTER, "llama8b") == ("it", "main")
    dv = _dose_views("llama8b")
    assert list(dv) == [f"it_d{tok}" for _, tok in DOSE_LADDER] + [f"d{tok}" for _, tok in DOSE_LADDER]
    CTL = "constv2ctl2-full__lora_r4_ce"
    for fam, prefix, suffix in (("it", "it_", "it"), ("main", "", "")):
        # the base family carries the CONTROL LoRA as its third object (the base->instruct sweep); the it family two
        want_cids = (["constv2-mix-full__lora_r4_ce", CTL, "constv2-mix-full__lora_r4_ce_minus_ctl"] if fam == "main"
                     else ["constv2-mix-full__lora_r4_ce", "constv2-mix-full__lora_r4_ce_minus_ctl"])   # roster order
        fam_shas = []
        for dose, tok in DOSE_LADDER:
            vn = f"{prefix}d{tok}"; v = dv[vn]
            assert v["dose"] == dose and v["dose_unit"] == gen.DOSE_UNIT == "lora_write_basis_multiplier", (vn, v)
            assert gen.view_dose(ROSTER, "llama8b", vn) == dose
            cells = v["cells"]
            assert list(cells) == want_cids, vn
            assert cells == {c: f"d{tok}{suffix}__{c}" for c in want_cids}, vn      # EVERY rung is this lane's own id
            served, staged = fr.served_cells(ROSTER, "llama8b", vn), fr.staged_cells(ROSTER, "llama8b", vn)
            assert served == staged == cells and not any(fr.is_prompted(c) for c in served), vn
            for cid, sid in cells.items():
                pin = PINS["llama8b"][sid]
                assert len(pin["object_sha256"]) == 64 and pin["sidecar_object_sha256_recorded"] == "match", sid
                assert pin["verified_cluster_a"] and pin["verified_cluster_b"], sid
                parent_sha, fit_model, fit_rev = LLAMA_PARENTS[(fam, cid)]
                if dose == 1.0:      # the anchor IS the parent's bytes
                    assert pin["object_sha256"] == parent_sha, (vn, cid)
                    if cid == want_cids[0]:   # ... and the Llama target's own staged copy of that parent, where staged
                        assert PINS["llama8b"][LLAMA_LANE_LORA_IDS[fam]]["object_sha256"] == parent_sha, (vn, cid)
                else:
                    assert pin["object_sha256"] != parent_sha, (vn, cid)
                fam_shas.append(pin["object_sha256"])
            # the generated pair: both families pin the INSTRUCT model; the caps are the campaign's
            cfg = _cfg(v)
            assert cfg[EXPLICIT_CELLS_KEY] == ["none"] + want_cids
            assert (cfg["model"], cfg["model_revision"]) == LLAMA_SERVED, vn
            # main's caps block on THIS target writes the NATIVE 4,096 BoN cap on every view; the
            # multi-turn target cap is the roster's wave-wide 4,096 (main's 2026-09-11 landing) — a dose view moves the
            # object, never a cap (the view carries no cap of its own)
            llama_caps = gen.target_caps(ROSTER, "llama8b")
            assert cfg["bon"]["max_new_tokens"] == llama_caps["bon"] == 4096
            assert cfg["attacks"]["target_max_new_tokens"] == ROSTER["target_max_new_tokens"] == 4096
            assert "caps" not in v and "target_max_new_tokens" not in v, vn
            assert cfg["attacks"]["static_max_new_tokens"] == 4096 and cfg["defenses_dir"] == f"out/full22/llama8b/cells_view/{vn}"
            text = (REPO / v["config"]).read_text()
            assert f"# serving target: {LLAMA_SERVED[0]} @ {LLAMA_SERVED[1]}" in text
            assert f"a DOSE view (dose {dose:g};" in text and "`bon.max_new_tokens` is the target's caps-table 4096" in text
            assert "the roster's wave-wide 4096" in text and "moves the OBJECT, never a cap" in text
            spec = cm.load_petri_spec(REPO / v["manifest"])
            assert (spec.target_model, spec.target_revision) == LLAMA_SERVED
            assert spec.cells == ["none"] + [gen.harness_cell_id(c) for c in want_cids]
            assert spec.chat_template_hash == gen.MODEL_SERVING["llama8b"]["chat_template_hash"] == "b48c47f644389271"
            assert spec.audit == f"petri_full22_llama8b_{vn}"
            for extra, want in (((), len(want_cids)), (("--staged",), len(want_cids))):   # 2 on the it family, 3 on the base family
                rc = subprocess.run([sys.executable, str(F22 / "roster_cell_count.py"), "llama8b", vn, *extra],
                                    capture_output=True, text=True)
                assert rc.returncode == 0 and int(rc.stdout.strip()) == want, (vn, extra, rc.stderr)
        # six doses x two objects = twelve DIFFERENT objects per family
        assert len(fam_shas) == len(set(fam_shas)) == 6 * len(want_cids), fam    # 12 on the it family, 18 on the base family
    # the two families are different objects rung for rung (different parents), served on the same target
    for dose, tok in DOSE_LADDER:
        for c in ("constv2-mix-full__lora_r4_ce", "constv2-mix-full__lora_r4_ce_minus_ctl"):   # the two arms both families carry
            assert PINS["llama8b"][f"d{tok}it__{c}"]["object_sha256"] != PINS["llama8b"][f"d{tok}__{c}"]["object_sha256"]
    # exactly-once holds WITHOUT a repeated id on this target — up to the all-models one-seed re-serve (2026-09-23):
    # the `s1` view RE-SERVES main's three base-lane objects and the paired `s1p` re-serves s1's r=4 companion (four
    # repeats, admitted by _admitted_reserve's shapes (b) and (c)); s1 also brings the two base__ PSR fits into a view
    # for the first time and s1p the composed paired_eb64__ object — three ids new to every_staged_id
    ids = fr.every_staged_id(ROSTER, "llama8b")
    # 24 ladder ids + the candsi views' six + this target's own six-arm object cells on its main / it views
    # (base__ / itdoc__ fits and the derived__ / derived_itdoc__ subtraction; the prompted baselines stage nothing)
    lane_ids = {sid for vn in ("main", "it", "fullconst", "shortconst") for sid in fr.staged_cells(ROSTER, "llama8b", vn).values() if sid}
    two = ["constv2-mix-full__lora_r4_ce", "constv2-mix-full__lora_r4_ce_minus_ctl"]
    dv_ids = ([f"d{tok}it__{c}" for _, tok in DOSE_LADDER for c in two]            # the it family: 12 ids
              + [f"d{tok}__{c}" for _, tok in DOSE_LADDER for c in two + [CTL]])    # the base family: 18 ids (control included)
    assert len(lane_ids) == 6 and not (lane_ids & set(dv_ids))
    one_seed_new = {"base__constv2-mix-full__psr_ce_all", "base__constv2ctl2-full__psr_ce_all", "paired_eb64__constv2-mix-full__psr_ce_all_minus_ctl"}
    assert len(set(ids)) == 30 + 6 + 6 + len(one_seed_new) and one_seed_new <= set(ids) and set(dv_ids) <= set(ids) and lane_ids <= set(ids)
    assert len(ids) == len(set(ids)) + 4      # the four one-seed / paired re-serves (test_roster_names_every_staged_object_exactly_once_per_target admits them)
    # the fit target each parent's sidecar records vs the served target: they differ BY DESIGN on the transfer
    # family, and nothing in the object bytes names a model (the naming is a config question, not an object one)
    fit_models = {LLAMA_PARENTS[k][1] for k in LLAMA_PARENTS}
    assert fit_models == {"NousResearch/Meta-Llama-3.1-8B-Instruct", "NousResearch/Meta-Llama-3.1-8B"}
    # the lane's two GSM8K views serve its SIX-ARM design (main, 2026-09-12: two LoRA fits, the subtraction, two prompted
    # baselines) beside the ladders; the dose views never borrow a lane id and the lane views never borrow a rung
    fits, derived = ["constv2-mix-full__lora_r4_ce", "constv2ctl2-full__lora_r4_ce"], ["constv2-mix-full__lora_r4_ce_minus_ctl"]
    for vn, pre, dpre in (("main", "base", "derived"), ("it", "itdoc", "derived_itdoc")):
        assert fr.staged_cells(ROSTER, "llama8b", vn) == {**{c: f"{pre}__{c}" for c in fits}, **{c: f"{dpre}__{c}" for c in derived}}, vn
        assert set(fr.served_cells(ROSTER, "llama8b", vn)) == set(fits + derived + ["prompted__safety_refusal", "prompted__si_constitution"]), vn
    # the family spelling is refused when borrowed: an `it_` dose view on nemotron, a bare one on gemma
    import copy
    r = copy.deepcopy(ROSTER); v = r["targets"]["nemotron3nano4b"]["views"].pop("d0p25"); r["targets"]["nemotron3nano4b"]["views"]["it_d0p25"] = v
    with pytest.raises(SystemExit, match="`it_` spelling but this target's ladder families are"):
        gen.view_dose(r, "nemotron3nano4b", "it_d0p25")


def test_llama8b_object_bytes_are_target_agnostic():
    """What the object bytes DO and DO NOT carry (the coordinating session's question, 2026-09-11): a LoRA blob is
    A / B per layer + rank, layers, positions, response_header_len — no model id, no revision, no template hash. The
    serving target is a CONFIG choice; the sidecar's model_id / model_revision are fit PROVENANCE that no serving code
    reads to pick a target (lens_adapters translates the tensors alone). Asserted on the adapter's input contract and
    on the blob keys the registry's own verdicts read, with synthetic blobs (never real fitted objects)."""
    import inspect
    import torch
    from sisteer.capeval.serving import lens_adapters as la
    from sisteer.extraction.multilayer import LoRASubtractionObject
    g = torch.Generator().manual_seed(0)
    lora = {"kind": "lora_all", "layers": [1, 3], "rank": 4, "positions": "all", "response_header_len": None,
            "A": {"1": torch.randn(12, 4, generator=g), "3": torch.randn(12, 4, generator=g)},
            "B": {"1": torch.randn(10, 4, generator=g), "3": torch.randn(10, 4, generator=g)}}
    sub = LoRASubtractionObject.compose_blob(lora, {**lora, "B": {k: -v for k, v in lora["B"].items()}}, provenance={})
    for blob in (lora, sub):
        assert not any(k in blob for k in ("model", "model_id", "model_revision", "chat_template_hash", "template_hash", "target"))
    import io
    for fn, blob in ((la._adapt_lora_all, lora), (la._adapt_lora_subtraction, sub)):
        buf = io.BytesIO(); torch.save(blob, buf)
        ops = fn({}, buf.getvalue())              # the adapter takes the bytes (and an empty record) — no model in sight
        assert ops and all(op.lowrank_a is not None and op.lowrank_b is not None for op in ops)
        assert "model" not in inspect.signature(fn).parameters


# --- math500-thinking-on-full22: the common.sh seams the
# --- math500 jobs use — each a DEFAULT-PRESERVING addition, pinned as such ---------------------------------------


def _bash_funcs(*names):
    """The named common.sh functions' source, in file order, for a behavioural run under `set -u`."""
    sh = (F22 / "common.sh").read_text()
    out = []
    for name in names:
        start = sh.index(f"{name}(){{")
        # the function ends at the first line that is exactly "}" after its start
        end = sh.index("\n}\n", start) + 3
        out.append(sh[start:end])
    return "\n".join(out)


def test_wave_root_defaults_to_the_campaigns_and_takes_f22_wave_root():
    """`WAVE_ROOT="${F22_WAVE_ROOT:-$SHARED/exp/sdfbase_full22}"` — unset, byte-identical to
    the campaign's root; set, the experiment's own root (never the full22 wave's)."""
    sh = (F22 / "common.sh").read_text()
    # native-serving-default: ONE assignment, overridable; the LENS root byte-identical, the NATIVE
    # default's root its own sibling `sdfbase_full22_native`
    line = 'WAVE_ROOT="${F22_WAVE_ROOT:-$SHARED/exp/sdfbase_full22$( [ "$F22_SERVING_PATH" = lens ] || echo _native)}"; mkdir -p "$WAVE_ROOT"'
    assert line in sh
    assert 'WAVE_ROOT="$SHARED/exp/sdfbase_full22"' not in sh          # the ONE assignment, overridable
    # the kind-aware default (2026-09-15): the ONE resolution is f22_select_serving_path, run by f22_init BEFORE the
    # WAVE_ROOT line reads the path (the bare `:-lora_native` default is gone from common.sh)
    assert 'F22_SERVING_PATH="${F22_SERVING_PATH:-lora_native}"' not in sh
    init = sh[sh.index("f22_init(){"):sh.index("\n}\n", sh.index("f22_init(){"))]
    assert init.index("  f22_select_serving_path\n") < init.index(line)
    script = ('set -u; SHARED=/tmp/x; ' + 'f22_log(){ :; }; f22_fatal(){ echo FATAL "$@"; exit 1; }; '
              'F22_SERVING_PATH="${F22_SERVING_PATH:-lora_native}"; ' + line + '; echo "$WAVE_ROOT"')
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                       env={k: v for k, v in os.environ.items() if k not in ("F22_WAVE_ROOT", "F22_SERVING_PATH")})
    assert r.stdout.strip() == "/tmp/x/exp/sdfbase_full22_native"              # the default engine's own root
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                       env={**os.environ, "F22_SERVING_PATH": "lens"})
    assert r.stdout.strip() == "/tmp/x/exp/sdfbase_full22"                     # the lens root, byte-identical
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                       env={**os.environ, "F22_WAVE_ROOT": "/tmp/x/exp/math500_thinking_on_full22"})
    assert r.stdout.strip() == "/tmp/x/exp/math500_thinking_on_full22"


def test_bank_view_check_never_builds_and_a_failing_check_exits(tmp_path):
    """The spy: f22_bank_view_check runs make_bank_view.py --check and EXITS on a non-zero exit — the
    builder (a --check-less invocation) and the view lock are never touched. The campaign's f22_bank_view keeps
    its build-under-lock path, untouched."""
    funcs = _bash_funcs("f22_bank_view_check")
    assert "flock" not in funcs and "--check" in funcs
    calls = tmp_path / "calls.log"
    fake = tmp_path / "scripts/sdfbase/full22"; fake.mkdir(parents=True)
    (fake / "make_bank_view.py").write_text(
        "import sys, os\nopen(os.environ['CALLS'], 'a').write(' '.join(sys.argv[1:]) + '\\n')\n"
        "sys.exit(2 if os.environ.get('VIEW_BROKEN') == '1' and '--check' in sys.argv else 0)\n")
    bank = tmp_path / "shared/exp/objects/sdfbase/full22/qwen35_4b/cells_view/main/constv2-mix-full__lora_r4_ce_minus_ctl"
    bank.mkdir(parents=True); (bank / "object.pt").write_bytes(b"object bytes")
    import hashlib
    pin = hashlib.sha256(b"object bytes").hexdigest()
    base = (f"set -u\nf22_log(){{ echo LOG \"$@\"; }}\nf22_fatal(){{ echo FATAL \"$@\"; exit 1; }}\n{funcs}\n"
            f"cd {tmp_path}; KEY=qwen35_4b; VIEW=main; WAVE=full22; SHARED={tmp_path}/shared\n"
            "F22_ROSTER=configs/sdfbase_full22_roster.yaml\n")     # the job's roster (common.sh's file-level default)
    env = {**os.environ, "CALLS": str(calls)}
    # the check names the job's roster and the bank it resolves — the same $SHARED/<bank_root>/<target> the builder
    # would resolve alone on the cluster, so TT_TWO_SI_DEFENSES_DIR and the builder share ONE derivation
    argv = f"--check --target qwen35_4b --view main --roster configs/sdfbase_full22_roster.yaml --bank {tmp_path}/shared/exp/objects/sdfbase/full22/qwen35_4b"
    # a passing check + a matching pin: the view is exported, the builder never invoked
    r = subprocess.run(["bash", "-c", base + f"f22_bank_view_check {sys.executable} constv2-mix-full__lora_r4_ce_minus_ctl {pin} && echo EXPORTED=$TT_TWO_SI_DEFENSES_DIR"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0 and "EXPORTED=" in r.stdout, (r.stdout, r.stderr)
    assert calls.read_text().splitlines() == [argv]
    # a FAILING check: fatal, and STILL only the --check call — no build, no lock
    calls.write_text("")
    r = subprocess.run(["bash", "-c", base + "f22_bank_view_check python3"], capture_output=True, text=True,
                       env={**env, "VIEW_BROKEN": "1"})
    assert r.returncode == 1 and "does not verify" in r.stdout and "never builds" in r.stdout
    assert calls.read_text().splitlines() == [argv]
    assert not list(tmp_path.rglob("*.view.lock"))
    # a drifted object: fatal by name
    calls.write_text("")
    r = subprocess.run(["bash", "-c", base + f"f22_bank_view_check {sys.executable} constv2-mix-full__lora_r4_ce_minus_ctl {'00' * 32}"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 1 and "drifted object" in r.stdout
    # the campaign's builder path is untouched
    sh = (F22 / "common.sh").read_text()
    assert 'flock -w 900 9' in sh[sh.index("f22_bank_view(){"):sh.index("f22_bank_view_check(){")]


def test_standup_extra_rides_the_standup_argv_verbatim_and_is_empty_by_default():
    """F22_STANDUP_EXTRA (a Bash array) is appended to mt_serving_up.py's argv; unset or empty
    it adds nothing (the campaign's argv byte for byte) — under `set -u`, so the empty-array idiom is pinned."""
    sh = (F22 / "common.sh").read_text()
    standup = sh[sh.index("f22_standup(){"):sh.index("f22_standup_stop(){")]
    assert '${F22_STANDUP_EXTRA[@]+"${F22_STANDUP_EXTRA[@]}"}' in standup
    line = 'set -u\nx(){ printf "[%s]" "$@"; echo; }\nx a ${F22_STANDUP_EXTRA[@]+"${F22_STANDUP_EXTRA[@]}"} b\n'
    r = subprocess.run(["bash", "-c", line], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "[a][b]", (r.stdout, r.stderr)
    r = subprocess.run(["bash", "-c", "F22_STANDUP_EXTRA=(); " + line], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "[a][b]"
    r = subprocess.run(["bash", "-c", 'F22_STANDUP_EXTRA=(--cell "constv2-mix-full__lora_r4_ce_minus_ctl" --upstream-timeout-s 7200); ' + line],
                       capture_output=True, text=True)
    assert r.stdout.strip() == "[a][--cell][constv2-mix-full__lora_r4_ce_minus_ctl][--upstream-timeout-s][7200][b]"


def test_serving_venv_default_name_is_the_campaigns_and_the_nemotron_variant_is_named_apart():
    """`f22_serving_venv` (no argument) resolves the campaign's sisteer-petri-vllm-<hash8 of
    vllm-requirements.txt> venv — byte-identical naming, so an existing venv is reused; `f22_serving_venv nemotron`
    resolves sisteer-petri-vllm-nemotron-<hash8 of base + overlay> and installs the overlay with
    --no-build-isolation (through f22_overlay_install). The naming is run, not grepped (the bootstrap itself is stubbed: uv
    never runs)."""
    import hashlib
    funcs = _bash_funcs("f22_serving_venv")
    base = REPO / "configs/capeval/vllm-requirements.txt"
    overlay = REPO / "configs/capeval/vllm-requirements-nemotron.txt"
    h_base = hashlib.sha256(base.read_bytes()).hexdigest()[:8]
    h_var = hashlib.sha256(base.read_bytes() + overlay.read_bytes()).hexdigest()[:8]
    script = (f"set -u\nf22_log(){{ :; }}\nf22_fatal(){{ echo FATAL \"$@\"; exit 1; }}\nflock(){{ :; }}\nuv(){{ :; }}\n"
              f"REPO_DIR={REPO}; USER=tester\n{funcs}\n"
              "f22_serving_venv() { local variant=\"${1:-}\"; " + funcs[funcs.index("{") + 1:funcs.index("(\n    flock")] +
              "echo \"$SERVE_VENV|$SERVE_REQS_SHA\"; }\n"
              "f22_serving_venv; f22_serving_venv nemotron\n")
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert r.returncode == 0, (r.stdout, r.stderr)
    lines = r.stdout.strip().splitlines()
    assert lines[0].split("|")[0] == f"/home/tester/venvs/sisteer-petri-vllm-{h_base}"
    assert lines[1].split("|")[0] == f"/home/tester/venvs/sisteer-petri-vllm-nemotron-{h_var}"
    assert lines[0].split("|")[1] == hashlib.sha256(base.read_bytes()).hexdigest()
    assert lines[1].split("|")[1] == hashlib.sha256(base.read_bytes() + overlay.read_bytes()).hexdigest()
    # the overlay installs through common.sh's ONE overlay command (f22_overlay_install: --no-config, the base's cutoff,
    # --no-build-isolation --no-deps under the FORCE_BUILD flags); tests/test_serving_overlay_install.py dry-runs it
    assert 'f22_overlay_install "$SERVE_VENV/bin/python" "$overlay" || exit 1' in funcs
    install = _bash_funcs("f22_overlay_install")
    assert "--no-config" in install and "--no-build-isolation --no-deps" in install
    assert "CAUSAL_CONV1D_FORCE_BUILD=TRUE MAMBA_FORCE_BUILD=TRUE" in install
    # the overlay: the two fit-env pins, nothing unpinned
    pins = [l for l in overlay.read_text().splitlines() if l and not l.startswith("#")]
    assert pins == ["causal-conv1d==1.6.2.post1", "mamba-ssm==2.3.2.post1"]


def test_client_venv_is_the_hashed_lock_with_the_one_vcs_line_apart():
    sh = (F22 / "common.sh").read_text()
    block = sh[sh.index("f22_client_venv(){"):sh.index("# --- the served target")]
    assert "tinycap-client-requirements.txt" in block and "--no-deps --require-hashes" in block
    assert "grep -v -E '^(instruction-following-eval) @ '" in block and "import inspect_ai, inspect_evals" in block   # the VCS allow-list
    assert 'CLIENT_VENV="/home/$USER/venvs/tinycap-client-$h"' in block


def test_model_facts_require_the_nemotron_backend_only_when_asked(tmp_path):
    """LB2: `f22_model_facts` demands the fused-kernel attestation for nemotron3nano4b ONLY under
    F22_REQUIRE_NEMO_BACKEND=1 (the math500 jobs); the campaign's jobs (unset) are unchanged. Run against the real
    python snippet with a synthetic attestation, through the lane's own interpreter."""
    funcs = _bash_funcs("f22_model_facts")
    att = tmp_path / "steering_attestation.json"
    att.write_text(json.dumps({"scheduler_geometry": {"worker_reports": [{"mamba_ssm_cache_dtype": "auto"}]}}))
    ep = tmp_path / "endpoint.json"; ep.write_text(json.dumps({"steering_attestation": str(att)}))
    nemo = tmp_path / "mamba_backend_attestation.json"
    base = (f"set -u\nf22_log(){{ echo LOG \"$@\"; }}\nF22_NEMO_DUTY_RATIO=0.5\n{funcs}\n"
            f"REPO_DIR={REPO}; SPY={sys.executable}; KEY=nemotron3nano4b; WAVE_ROOT={tmp_path}; SERVE_REQS_SHA={'ab' * 32}\n")
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    # unset: no backend requirement, facts OK (the campaign path)
    r = subprocess.run(["bash", "-c", base + f"f22_model_facts {ep} 73"], capture_output=True, text=True, env=env)
    assert r.returncode == 0 and '"nemo_backend_required": false' in r.stdout and '"problems": []' in r.stdout, r.stdout
    # required, no attestation: refused by name
    r = subprocess.run(["bash", "-c", base + f"F22_REQUIRE_NEMO_BACKEND=1 f22_model_facts {ep} 73"], capture_output=True, text=True, env=env)
    assert r.returncode == 1 and "no attestation at" in r.stdout and "PREREQUISITE" in r.stdout
    # required, a fused attestation for THIS venv, admitted against a comparable same-GPU peer: OK
    peer = {"model_key": "qwen35_4b", "sm_duty_mean": 0.30, "tok_s": 1800.0}
    good = {"admitted": True, "fused_backend_attested": True, "sm_duty_mean": 0.28, "tok_s": 1400.0, "duty_comparable": True,
            "duty_ratio": 0.5, "duty_reference": peer, "serving_requirements_sha256": "ab" * 32, "probe_job_id": "1"}
    nemo.write_text(json.dumps(good))
    r = subprocess.run(["bash", "-c", base + f"F22_REQUIRE_NEMO_BACKEND=1 F22_NEMO_ATTESTATION={nemo} f22_model_facts {ep} 73"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0 and '"nemo_backend_required": true' in r.stdout and '"problems": []' in r.stdout, r.stdout
    # a fallback anywhere, under half the peer's duty or tok/s, an incomparable/missing peer, a probe that did not admit,
    # or another venv's attestation: refused, each by name — the reader re-derives the rule, never trusts `admitted`
    for doc, why in (({**good, "fused_backend_attested": False}, "fallback somewhere"),
                     ({**good, "sm_duty_mean": 0.10}, "below 0.5 x the same-GPU peer's"),
                     ({**good, "tok_s": 800.0}, "below 0.5 x the same-GPU peer's"),
                     ({**good, "duty_comparable": False}, "no comparable same-GPU peer reference"),
                     ({k: v for k, v in good.items() if k != "duty_reference"}, "no comparable same-GPU peer reference"),
                     ({**good, "admitted": False}, "did not admit"),
                     ({**good, "serving_requirements_sha256": "cd" * 32}, "re-run the probe")):
        nemo.write_text(json.dumps(doc))
        r = subprocess.run(["bash", "-c", base + f"F22_REQUIRE_NEMO_BACKEND=1 F22_NEMO_ATTESTATION={nemo} f22_model_facts {ep} 73"],
                           capture_output=True, text=True, env=env)
        assert r.returncode == 1 and why in r.stdout, (why, r.stdout)
    # a non-nemotron key never consults it, even when required
    r = subprocess.run(["bash", "-c", base.replace("KEY=nemotron3nano4b", "KEY=gemma4e4b") + f"F22_REQUIRE_NEMO_BACKEND=1 f22_model_facts {ep} 54"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0 and '"nemo_backend_required": false' in r.stdout
    # native-serving-default: on a NATIVE standup the fused-kernel fact is the standup's OWN live attestation over
    # the ported sisteer_mamba_backend_report RPC (endpoint.mamba_backend_attestation), read beside the attestation
    # the endpoint names (serving_attestation.json, the strict lora_native record) — the lens probe job's duty
    # certificate (F22_NEMO_ATTESTATION) is never consulted there
    natt = tmp_path / "serving_attestation.json"
    natt.write_text(json.dumps({"kind": "lora_native", "scheduler_geometry": {"worker_reports": [{"mamba_ssm_cache_dtype": "auto"}]}}))
    nep = tmp_path / "endpoint_native.json"
    live = {"ok": True, "reports": [{"n_mamba_layers": 21}], "contract_sha256": "ab" * 32,
            "rpc_method": "sisteer_mamba_backend_report"}
    nep.write_text(json.dumps({"engine": "lora_native", "serving_attestation": str(natt), "mamba_backend_attestation": live}))
    r = subprocess.run(["bash", "-c", base + f"F22_REQUIRE_NEMO_BACKEND=1 f22_model_facts {nep} 73"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0 and '"engine": "lora_native"' in r.stdout and '"problems": []' in r.stdout, r.stdout
    assert '"source": "endpoint.mamba_backend_attestation"' in r.stdout and '"n_reports": 1' in r.stdout, r.stdout
    assert '"nemo_backend_required": true' in r.stdout and "no attestation at" not in r.stdout
    nep.write_text(json.dumps({"engine": "lora_native", "serving_attestation": str(natt),
                               "mamba_backend_attestation": None, "mamba_backend_attestation_error": "no report from any worker"}))
    r = subprocess.run(["bash", "-c", base + f"F22_REQUIRE_NEMO_BACKEND=1 f22_model_facts {nep} 73"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 1 and "nemotron backend (native)" in r.stdout and "unattested" in r.stdout, r.stdout
    # not required: the native endpoint's facts read without any backend requirement, engine recorded
    r = subprocess.run(["bash", "-c", base + f"f22_model_facts {nep} 73"], capture_output=True, text=True, env=env)
    assert r.returncode == 0 and '"engine": "lora_native"' in r.stdout and '"nemo_backend_required": false' in r.stdout


# --- llama-paper-rows: the FIFTH target -------------------------

def test_the_roster_carries_five_targets_and_llama8b_pins_the_nous_template_and_the_no_thinking_serving_row():
    """llama8b joins the registry: NousResearch's instruct checkpoint @ d10aef79 (its template renders EXACTLY the
    messages given — no Cutting-Knowledge / Today-Date block, so the `none` anchor is unprompted, P2), two views
    (main: the base-lane objects TRANSFERRED; it: the instruct-direct objects), the frozen chat_template_hash equal to
    petri_twosi.yaml's and the lane's, `llama3_json` tool parsing, eager execution; every other target's config and
    manifest regenerate byte-identically (the generator's --check ran above for all five). Since the llama8b lane
    landed (2026-09-12) the block is THEIRS wholesale: the two GSM8K views first, then the lane's `fullconst` /
    `shortconst` views, its caps / lanes / xstest_subset keys, and its 40 + 6 (+ 1) pins (the eight PSR pins included; the strength frontier's six control-LoRA dose copies; the all-models lane's composed paired_eb64__ PSR difference, 2026-09-23) — bound to no
    GSM8K view's cell)."""
    paper_five = {"gemma4e4b", "qwen35_4b", "nemotron3nano4b", "glm46vflash", "llama8b"}
    assert set(ROSTER["targets"]) == paper_five | (set(ROSTER["targets"]) & set(gen.FRONT_DOOR_MODELS))   # the paper's five (+ the front door's, once 05 lands a block); roster order is the lane's
    t = ROSTER["targets"]["llama8b"]
    assert t["model"] == "NousResearch/Meta-Llama-3.1-8B-Instruct" and t["model_revision"] == "d10aef7999a2b5ba950ab3974312feeedbfe0b77"
    # the lane's two GSM8K views lead, then the all-models one-seed views s1 / s1p (b64-capability-agentic-queue, 2026-09-23:
    # s1 re-serves main's base-lane objects as the family's one-seed row, s1p its composed paired-write difference ReLU), then
    # the lane's two prompted-only views; the strength frontier's twelve dose views (two families x six doses) follow them
    assert list(t["views"])[:6] == ["main", "it", "s1", "s1p", "fullconst", "shortconst"]
    assert set(t["views"]) == {"main", "it", "s1", "s1p", "fullconst", "shortconst"} | set(_dose_views("llama8b")) and len(t["views"]) == 18
    assert t["pins_config"] == "configs/two_si_llama_fits.yaml"
    # the lane's views at its pin de78ab46 (I21), reduced to the SIX-ARM design (PSR out by default): the
    # two LoRA fits, the LoRA subtraction, then the two prompted baselines (safety, constitution) null — on both views
    fits = ["constv2-mix-full__lora_r4_ce", "constv2ctl2-full__lora_r4_ce"]     # the lane's order (mix, then ctl2) since its block landed
    derived = ["constv2-mix-full__lora_r4_ce_minus_ctl"]
    prompted = ["prompted__safety_refusal", "prompted__si_constitution"]
    assert list(t["views"]["main"]["cells"]) == list(t["views"]["it"]["cells"]) == fits + derived + prompted
    assert [t["views"]["main"]["cells"][c] for c in fits] == [f"base__{c}" for c in fits] and [t["views"]["main"]["cells"][c] for c in derived] == [f"derived__{c}" for c in derived]
    assert [t["views"]["it"]["cells"][c] for c in fits] == [f"itdoc__{c}" for c in fits] and [t["views"]["it"]["cells"][c] for c in derived] == [f"derived_itdoc__{c}" for c in derived]
    assert all(t["views"][v]["cells"][p] is None for v in ("main", "it") for p in prompted)
    assert not any("psr" in c for v in ("main", "it") for c in t["views"][v]["cells"])          # PSR out: not run, not plotted
    assert {"caps", "lanes", "xstest_subset", "capability_controls"} <= set(t)                     # the lane's battery keys, landed with its block
    assert PINS["llama8b"]["derived__constv2-mix-full__lora_r4_ce_minus_ctl"]["object_sha256"] == "4375b67a0fb0621e9ee296041d0842cb24294112e18424a803e8e54e9538a740"
    assert PINS["llama8b"]["derived_itdoc__constv2-mix-full__lora_r4_ce_minus_ctl"]["object_sha256"] == "e85220c9e5d2b9aa4377fea7a9a590048c693118975be9f36ae579b0eeff6f13"
    # the lane's eight PSR pins (bound to no GSM8K cell) + the all-models composed paired_eb64__ PSR difference (2026-09-23; the s1p view's cell)
    assert sum("psr_ce_all" in sid for sid in PINS["llama8b"]) == 8 + 1 and "paired_eb64__constv2-mix-full__psr_ce_all_minus_ctl" in PINS["llama8b"]
    ms = gen.MODEL_SERVING["llama8b"]
    twosi = yaml.safe_load((REPO / "configs/capeval/manifests/petri_twosi.yaml").read_text())["petri"]
    assert ms["chat_template_hash"] == twosi["chat_template_hash"] == "b48c47f644389271"
    assert ms["tool_parser"] == "llama3_json" and ms["flags"] == ["--enforce-eager"] and "no thinking mode" in ms["note"].lower()
    for view in ("main", "it"):
        spec = cm.load_petri_spec(REPO / t["views"][view]["manifest"])
        assert spec.chat_template_hash == "b48c47f644389271" and spec.tool_parser == "llama3_json"
        assert list(spec.serving_launch_flags) == ["--enforce-eager"] and (spec.target_model, spec.target_revision) == (t["model"], t["model_revision"])
        cfg = _cfg(t["views"][view])
        assert cfg["model"] == t["model"] and cfg["model_revision"] == t["model_revision"]
    # the pins: the lane's whole block (40: the six LoRA cells, si12, the dose ladders, the eight PSR entries), each 64-hex
    # a FRONT-DOOR target joins the pins once one of its carried views names a staged object (05 --step stage / pins); its u0 view serves
    # none + the two prompted SIs — nothing to pin — so the block is absent while u0 is the only carried view (stage-E 05 lane, 2026-09-30)
    pinned_targets = {k for k, tv in ROSTER["targets"].items() if k not in gen.FRONT_DOOR_MODELS
                      or any(sid is not None for v in tv["views"].values() for sid in (v.get("cells") or {}).values())}
    # ... and a front-door key that IMPORTS the paper's committed objects (reproduce/models.yaml `committed_objects_of`; demo-import M1) is pinned
    # from objects/DEMO.json by reproduce/tools/import_objects.py pins AHEAD of its roster block (05 --step pins precedes --step roster)
    pinned_targets |= {k for k, m in gen.FRONT_DOOR_MODELS.items() if m.get("committed_objects_of")}
    assert set(PINS) == pinned_targets and len(PINS["llama8b"]) == 46 + 1   # 40 + the base family's six control-LoRA copies + the all-models composed paired_eb64__ pin
    for view in ("main", "it"):
        for sid in t["views"][view]["cells"].values():
            assert sid is None or len(PINS["llama8b"][sid]["object_sha256"]) == 64, sid
    assert list(PINS["llama8b"]) == sorted(PINS["llama8b"])
    # the four published targets' pins blocks carry the frozen GSM8K inputs' entries byte for byte (P16: the published matrix's
    # inputs at the pin); the live registry has only ADDED since (the strength-dose lane's ladder pins; the batch-64 instruct-direct
    # lane's itdoc64__ / derived_itdoc64__ pins, staged 2026-09-13) — nothing removed or rewritten
    frozen = json.loads((REPO / "configs/sdfbase/gsm8k/frozen/8f5f1495/sdfbase_full22_pins.json").read_text())
    assert set(frozen) == set(PINS) - {"llama8b"} - set(gen.FRONT_DOOR_MODELS)      # the four published targets; a front-door block is never in the frozen matrix's inputs
    for key, block in frozen.items():
        assert set(block) <= set(PINS[key]) and all(PINS[key][sid] == block[sid] for sid in block), key
        added_pins = set(PINS[key]) - set(block)
        # ... plus the certification-only `si12__` pin (the constitution response-fit psr the lens §7.2 rows were
        # measured on, staged by the same lane the frozen block's two si12__ pins came from)
        # ... plus the qwen base-seed-replicate lane's staged objects (`base64s<s>__` parents and the admitted `derived_base64s<s>__`
        # LoRA differences, pinned 2026-09-17 from the clusters' staging manifest; the distilled PSR difference was not admitted for
        # any seed and has no pin)
        # ... plus the paired views' composed difference_relu arm (`paired_eb64s<s>__`, 2026-09-21: the lane's staged copy of the
        # psr_paired_write object per seed, admitted by derived_admission's psr_paired_write branch and pinned from the
        # clusters' staging manifest)
        # ... plus the Arditi refusal-steer object's served-only Class-B identity pin (`ar_<dose>__arditi_allpos`: the Arditi rows
        # belong to the seed family's capability lanes; the pin id keeps the lane's `ar_c0p3` spelling the OR-Bench / GSM8K
        # records already carry)
        # ... plus the all-models one-seed lane's composed paired-write difference per non-Qwen family (`paired_eb8__` on gemma /
        # nemotron, `paired_eb64__` on llama — the number is the parents' effective batch, no seed suffix; 2026-09-23: the s1p
        # view's own cell, staged from the relu_paired bank and pinned from cluster A)
        assert all("__" in sid and sid.startswith(("d", "itdoc64__", "derived_itdoc64__", "si12__", "base64s", "derived_base64s", "paired_eb64s", "paired_eb8__", "paired_eb64__", "ar_")) for sid in added_pins), (key, sorted(added_pins))
        assert not any(sid.endswith("__constv2-mix-full__psr_ce_all_minus_ctl_distilled") and sid.startswith("derived_base64s") for sid in added_pins), (key, sorted(added_pins))
    frozen_roster = yaml.safe_load((REPO / "configs/sdfbase/gsm8k/frozen/8f5f1495/sdfbase_full22_roster.yaml").read_text())
    # the four published targets' blocks are the frozen inputs' PLUS what main recorded after the pin: the matched prompted
    # safety-SI baseline on the two `it` views (`prompted__safety_refusal: null`, a served cell, never a bank-view row) and the
    # strength-dose lane's dose VIEWS (`it_d<dose>` / `d<dose>`, and qwen's ReLU ladder `d<dose>_relu` since 2026-09-16 — their
    # own bank views, never a GSM8K row: the published matrix carries 6 TAGs and no dose row). The frozen matrix regenerates
    # byte-identically without either, which is what P16 is for.
    # ... and never a FRONT-DOOR target (paper-companion-repo): its block joins the roster through 05 --step roster (u0 first), after the
    # frozen matrix's inputs were pinned, so the frozen roster carries none of it — the same rule as the pins' target set above
    live_targets = {k: copy.deepcopy(v) for k, v in ROSTER["targets"].items() if k != "llama8b" and k not in gen.FRONT_DOOR_MODELS}
    dose_view = gen._DOSE_VIEW_RE      # the generator's own grammar: `d<tok>` / `it_d<tok>` / `d<tok>_<family>`
    for key, target in live_targets.items():
        for view in [v for v in target["views"] if dose_view.match(v)]:
            assert view not in frozen_roster["targets"][key]["views"], (key, view)
            del target["views"][view]
        # ... and the strength frontier's family declaration beside them (`dose_families`: gemma and qwen carry the
        # base->instruct family since 2026-09-15) — a roster fact of the dose views, never of the frozen matrix
        if "dose_families" in target:
            assert "dose_families" not in frozen_roster["targets"][key], key
            del target["dose_families"]
    # ... and, since 2026-09-15, the batch-64 SEED-REPLICATE views of qwen35_4b (`s1` / `s2` / `s3`; the qwen base-seed-replicate
    # lane): their own bank views — the FIVE ADMITTED objects staged into `cells` (the four parents `base64s<s>__…` and the LoRA
    # difference `derived_base64s<s>__…`, pinned 2026-09-17; the distilled PSR difference failed admission for every seed and is
    # DROPPED from every view — no pending_fit left) — and the per-view BoN cap (make_full22_configs
    # .view_declared_bon_cap) — never a GSM8K row of the frozen matrix, and on no other target
    # ... and, since 2026-09-23 (the all-models one-seed waves), the ONE-SEED view `s1` of
    # gemma4e4b / nemotron3nano4b (llama8b's sits outside live_targets): it RE-SERVES main's five base-lane objects under the seed
    # grammar (the same `base__` / `derived__` staged ids — one seed per family, the s1 bank a volume-internal copy of main's) plus
    # the two prompted baselines last; no caps / Petri knobs of its own; never a GSM8K row of the frozen matrix
    seed_view = re.compile(r"^s[123]$")
    for key, target in live_targets.items():
        for view in [v for v in target["views"] if seed_view.match(v)]:
            assert view not in frozen_roster["targets"][key]["views"], (key, view)
            sv = target["views"][view]
            assert list(sv["cells"])[-2:] == ["prompted__safety_refusal", "prompted__si_constitution"] and all(sv["cells"][c] is None for c in list(sv["cells"])[-2:]), (key, view)
            assert len(sv.get("cells") or {}) == 7 and "pending_fit" not in sv, (key, view)
            assert "constv2-mix-full__psr_ce_all_minus_ctl_distilled" not in sv["cells"], (key, view)
            if key == "qwen35_4b":
                # ... + the two PROMPTED baselines LAST since 2026-09-17: null staged ids, never pinned
                assert sv.get("caps") == {"bon": 4096}, (key, view)
                assert all(sid == (f"derived_base64s{view[1:]}__{cid}" if cid.endswith("_minus_ctl") else f"base64s{view[1:]}__{cid}")
                           for cid, sid in sv["cells"].items() if not cid.startswith("prompted__")), (key, view)
            else:
                assert key in ("gemma4e4b", "nemotron3nano4b") and view == "s1" and set(sv) == {"config", "manifest", "cells"}, (key, view, sorted(sv))
                main_cells = target["views"]["main"]["cells"]
                objs = {cid: sid for cid, sid in sv["cells"].items() if not cid.startswith("prompted__")}
                assert len(objs) == 5 and all(main_cells.get(cid) == sid and sid.startswith(("base__", "derived__")) for cid, sid in objs.items()), (key, view, objs)
            del target["views"][view]
    # ... and, since 2026-09-21, the three PAIRED views of qwen35_4b (`s1p` / `s2p` / `s3p`): each serves the admitted
    # composed psr_paired_write arm from the lane's staged copy
    # `paired_eb64s<s>__…` beside its seed view's r=4 companion and LoRA difference (the same staged ids as `s<s>`, re-served in the
    # same pass: the companion is the standup's max_lora_rank anchor, the difference the in-view difference read) — three cells, no
    # prompted baseline, no caps / Petri knobs; never a GSM8K row of the frozen matrix, and on no other target
    # ... and the all-models `s1p` of gemma4e4b / nemotron3nano4b (2026-09-23): the composed paired-write difference of the
    # family's base-lane PSR parents (`paired_eb8__…`, the parents' effective batch; llama's `paired_eb64__` is outside
    # live_targets) beside main's r=4 constitution LoRA re-served as the twin-marker companion (make_full22_configs.view_paired_of
    # refuses a paired view that re-serves nothing) — two cells, `paired_of: s1`, no other knob
    paired_view = re.compile(r"^s([123])p$")
    for key, target in live_targets.items():
        for view in [v for v in target["views"] if paired_view.match(v)]:
            assert view not in frozen_roster["targets"][key]["views"], (key, view)
            s_ = paired_view.match(view).group(1); pv = target["views"][view]
            assert set(pv) == {"config", "manifest", "cells", "paired_of"} and pv["paired_of"] == f"s{s_}", (key, view, sorted(pv))
            if key == "qwen35_4b":
                assert pv["cells"] == {"constv2-mix-full__psr_ce_all_minus_ctl": f"paired_eb64s{s_}__constv2-mix-full__psr_ce_all_minus_ctl",
                                       "constv2-mix-full__lora_r4_ce": f"base64s{s_}__constv2-mix-full__lora_r4_ce",
                                       "constv2-mix-full__lora_r4_ce_minus_ctl": f"derived_base64s{s_}__constv2-mix-full__lora_r4_ce_minus_ctl"}, (key, view)
                assert pv["cells"]["constv2-mix-full__lora_r4_ce"] == target["views"].get(f"s{s_}", frozen_roster["targets"][key]["views"].get(f"s{s_}", {"cells": {}}))["cells"].get("constv2-mix-full__lora_r4_ce", pv["cells"]["constv2-mix-full__lora_r4_ce"])
            else:
                assert key in ("gemma4e4b", "nemotron3nano4b") and view == "s1p", (key, view)
                assert pv["cells"] == {"constv2-mix-full__psr_ce_all_minus_ctl": "paired_eb8__constv2-mix-full__psr_ce_all_minus_ctl",
                                       "constv2-mix-full__lora_r4_ce": target["views"]["main"]["cells"]["constv2-mix-full__lora_r4_ce"]}, (key, view)
                assert pv["cells"]["constv2-mix-full__lora_r4_ce"] == "base__constv2-mix-full__lora_r4_ce", (key, view)
            del target["views"][view]
    # ... and, since 2026-09-12, the constitution prompted baseline (`prompted__si_constitution: null`) on the views the
    # constitution-SI wave ran — main's additions after the pin are PROMPTED baselines only, served cells mapped to null,
    # appended after the frozen cells; the frozen roster carries none of them
    for key in ("gemma4e4b", "qwen35_4b"):
        assert "prompted__safety_refusal" in live_targets[key]["views"]["it"]["cells"]
    # ... and, since 2026-09-13, the batch-64 instruct-direct lane's two OBJECT cells on qwen35_4b/it — the control LoRA fit and
    # the LoRA subtraction of the two itdoc64 parents — served from lane-prefixed staged ids the frozen matrix never named
    # (ADDED beside the frozen `itdoc__` cells, which keep their staged ids: the constitution parent is pinned, not repointed)
    itdoc64_it = {"constv2ctl2-full__lora_r4_ce": "itdoc64__constv2ctl2-full__lora_r4_ce",
                  "constv2-mix-full__lora_r4_ce_minus_ctl": "derived_itdoc64__constv2-mix-full__lora_r4_ce_minus_ctl"}
    for key in ("qwen35_4b", "gemma4e4b"):      # qwen 2026-09-13, gemma the same day
        assert {c: live_targets[key]["views"]["it"]["cells"][c] for c in itdoc64_it} == itdoc64_it, key
    for key, target in live_targets.items():
        for view, vdoc in target["views"].items():
            cells = vdoc["cells"]
            frozen_cells = frozen_roster["targets"][key]["views"][view]["cells"]
            added = [c for c in cells if c not in frozen_cells]
            for c in added:
                if key in ("qwen35_4b", "gemma4e4b") and view == "it" and c in itdoc64_it:
                    assert cells[c] == itdoc64_it[c], (key, view, c, cells[c])
                else:
                    assert c.startswith("prompted__") and cells[c] is None, (key, view, c, cells[c])
                del cells[c]
    assert live_targets == frozen_roster["targets"]
    # the live roster's wave-wide keys are the pin's, byte-identical, PLUS additions — the four the 2026-09-11 canonical-battery
    # rulings added on main at least; a frozen key is never dropped or rewritten
    added = set(ROSTER) - set(frozen_roster)
    # (bon_climb_temperature, added 2026-09-11, was RENAMED bon_climb_decode on 2026-09-17: the climb's sampler is the
    # store's, never a wave-wide scalar -- a key added after the pin, so not a frozen key dropped)
    assert {"target_max_new_tokens", "bon_climb_cap", "bon_climb_decode", "controls_max_new_tokens"} <= added and not (set(frozen_roster) - set(ROSTER))
    assert "bon_climb_temperature" not in ROSTER and "bon_climb_temperature" not in frozen_roster
    assert {k: v for k, v in ROSTER.items() if k not in added | {"targets"}} == {k: v for k, v in frozen_roster.items() if k != "targets"}


def test_bon_drift_check_holds_a_caps_writing_target_to_its_own_cap_and_dose_views_to_the_body():
    """The roster's llama8b caps table WRITES bon.max_new_tokens (the native 4,096 pass), so every llama8b view
    is held to 4096 — not the body's 512 — while the rest of the bon block stays byte-identical to the body; a dose view of
    a target without a caps entry is still held to the body's 512; a llama8b view with the body's 512 REFUSES (the cap is
    per target, never per view)."""
    body = (REPO / "configs" / "sdfbase_transfer_qwen35_4b_r3.yaml").read_text(encoding="utf-8")
    for view in ("main", "it", "fullconst", "shortconst"):
        rendered = (REPO / "configs" / f"sdfbase_full22_llama8b{'' if view == 'main' else '_' + view}.yaml").read_text(encoding="utf-8")
        gen._refuse_bon_drift("llama8b", view, rendered, body)
    text = (REPO / "configs" / "sdfbase_full22_llama8b.yaml").read_text(encoding="utf-8")
    head, sep, tail = text.partition("\nbon:")                # the BON block's cap, not attacks.target_max_new_tokens above it
    bad = head + sep + tail.replace("max_new_tokens: 4096", "max_new_tokens: 512", 1)
    with pytest.raises(SystemExit, match="caps.bon 4096"):
        gen._refuse_bon_drift("llama8b", "main", bad, body)
    good_dose = (REPO / "configs" / "sdfbase_full22_gemma4e4b_it_d0p25.yaml").read_text(encoding="utf-8")
    gen._refuse_bon_drift("gemma4e4b", "it_d0p25", good_dose, body)   # a dose view: the body's 512, unchanged


# --- native-serving-default (W-B): the wave shell's engine gates, the served-cells seam, the drivers' engine assert ---

def _standup_slice(start_marker: str, end_marker: str) -> str:
    """The text of common.sh's ``f22_standup`` between two markers — a behavioural slice run under `set -u`, never
    grepped (the whole function launches the engine; its post-standup gates are pure functions of the two stamps)."""
    sh = (F22 / "common.sh").read_text()
    standup = sh[sh.index("f22_standup(){"):sh.index("f22_standup_stop(){")]
    return standup[standup.index(start_marker):standup.index(end_marker)]


_GATE_START = '  local ep="$out/endpoint.json"'
_GATE_END = "  # THE THINKING REGIME, read back from both stamps"
_SERVED_START = "  # SERVED_CELLS (native-serving-default): ALL_CELLS minus"
_SERVED_END = "  local adj_json;"

_GEOM = {"proxy_url": "http://127.0.0.1:1/v1", "routing_table": "/x/routing_table.json",
         "max_model_len": 4864, "max_num_seqs": 121, "attested_max_num_seqs": 121, "attested_max_model_len": 4864}
_LENS_EP = {**_GEOM, "lens_reduction_mode": "stacked"}
_LENS_MODELS = {"petri_proxy": {"lens_reduction_mode": "stacked"}}
_NATIVE_EP = {**_GEOM, "engine": "lora_native", "id_name_map": {"1": "none__lora0", "2": "si_x"},
              "attested_enforce_eager": False, "attested_cudagraph_mode": "FULL_AND_PIECEWISE",
              # the FULL capture surface (2026-09-16): the largest FULL capture size, >= the attested S
              "attested_cudagraph_full_capture_max": 121,
              "geometry": {"max_model_len": 4864, "max_num_seqs": 121, "adjustments": []}}
_NATIVE_MODELS = {"petri_proxy": {"engine": {"kind": "lora_native", "adapters": {"none__lora0": {}, "si_x": {}}}}}


def _run_standup_gates(tmp_path, *, ep, models, serving_path, gS=""):
    """f22_standup's POST-STANDUP GATES (the engine from both stamps, the attested geometry, the §1.8 adjustment
    chain), RUN under `set -u` against a written endpoint.json and a shimmed curl answering /v1/models.
    Returns ``(rc, output)`` — rc 1 with a REFUSING line is a refusal, exactly as the job sees it."""
    out = tmp_path / f"gates-{len(list(tmp_path.glob('gates-*')))}"
    out.mkdir(parents=True)
    (out / "endpoint.json").write_text(json.dumps(ep))
    shims = tmp_path / "bin"
    shims.mkdir(exist_ok=True)
    curl = shims / "curl"
    curl.write_text('#!/bin/bash\necho "$FAKE_MODELS_JSON"\n')
    curl.chmod(0o755)
    script = ("set -u\nf22_log(){ echo \"LOG $*\"; }\n"
              f"SPY={sys.executable}\nout={out}\nserving_path={serving_path}\ngS='{gS}'\n"
              "gate(){\n" + _standup_slice(_GATE_START, _GATE_END) + "\n  return 0\n}\ngate; echo \"rc=$?\"\n")
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                       env={**os.environ, "PATH": f"{shims}:{os.environ['PATH']}", "FAKE_MODELS_JSON": json.dumps(models)})
    assert r.returncode == 0, (r.stdout, r.stderr)
    return int(r.stdout.strip().splitlines()[-1].split("rc=")[1]), r.stdout + r.stderr


def test_standup_gates_admit_each_engine_from_its_own_stamps_and_refuse_the_other(tmp_path):
    """common.sh f22_standup, RUN: under F22_SERVING_PATH=lens the pre-landing gate byte for byte (the attested
    STACKED mode on endpoint.json AND /v1/models); under lora_native the native gate — endpoint.json engine ==
    lora_native, petri_proxy.engine.kind == lora_native, the proxy's adapters EXACTLY the attested id_name_map names,
    a graphs-on engine (enforce_eager false, cudagraph_mode not NONE), and no lens_reduction_mode anywhere. A lens
    record under the native path (or the reverse) refuses — never re-labelled."""
    rc, out = _run_standup_gates(tmp_path, ep=_LENS_EP, models=_LENS_MODELS, serving_path="lens")
    assert rc == 0 and "REFUSING" not in out, out
    rc, out = _run_standup_gates(tmp_path, ep=_LENS_EP, models={"petri_proxy": {"lens_reduction_mode": "per_row"}}, serving_path="lens")
    assert rc == 1 and "petri_proxy.lens_reduction_mode=per_row (wave mode: stacked)" in out, out
    rc, out = _run_standup_gates(tmp_path, ep=_NATIVE_EP, models=_NATIVE_MODELS, serving_path="lens")
    assert rc == 1 and "F22_SERVING_PATH=lens but endpoint.json engine=lora_native (a native standup answered a lens job)" in out
    rc, out = _run_standup_gates(tmp_path, ep=_NATIVE_EP, models=_NATIVE_MODELS, serving_path="lora_native")
    assert rc == 0 and "REFUSING" not in out, out
    rc, out = _run_standup_gates(tmp_path, ep=_LENS_EP, models=_LENS_MODELS, serving_path="lora_native")
    assert rc == 1 and "F22_SERVING_PATH=lora_native but endpoint.json engine=None, /v1/models petri_proxy.engine.kind=None" in out
    rc, out = _run_standup_gates(tmp_path, ep={**_NATIVE_EP, "lens_reduction_mode": "stacked"}, models=_NATIVE_MODELS,
                                 serving_path="lora_native")
    assert rc == 1 and "two engines on one record" in out
    rc, out = _run_standup_gates(tmp_path, ep=_NATIVE_EP, models={"petri_proxy": {**_NATIVE_MODELS["petri_proxy"], "lens_reduction_mode": "stacked"}},
                                 serving_path="lora_native")
    assert rc == 1 and "two engines on one record" in out
    rc, out = _run_standup_gates(tmp_path, ep=_NATIVE_EP, serving_path="lora_native",
                                 models={"petri_proxy": {"engine": {"kind": "lora_native", "adapters": {"none__lora0": {}}}}})
    assert rc == 1 and "engine adapters [none__lora0] != the attested id_name_map names [none__lora0,si_x]" in out
    rc, out = _run_standup_gates(tmp_path, ep={**_NATIVE_EP, "attested_enforce_eager": True}, models=_NATIVE_MODELS, serving_path="lora_native")
    assert rc == 1 and "enforce_eager=true cudagraph_mode=FULL_AND_PIECEWISE (graphs must be ON" in out
    rc, out = _run_standup_gates(tmp_path, ep={**_NATIVE_EP, "attested_cudagraph_mode": "NONE"}, models=_NATIVE_MODELS, serving_path="lora_native")
    assert rc == 1 and "graphs must be ON" in out
    # THE FULL CAPTURE SURFACE (2026-09-16): graphs-on is not enough — the largest FULL capture size must reach the
    # attested S (S=51 on the default list ran 49-51 PIECEWISE with attention eager); a smaller one, None, and a
    # pre-landing endpoint without the key all refuse by name; an equal one passes (the healthy case above)
    rc, out = _run_standup_gates(tmp_path, ep={**_NATIVE_EP, "attested_cudagraph_full_capture_max": 120}, models=_NATIVE_MODELS,
                                 serving_path="lora_native")
    assert rc == 1 and "largest FULL CUDA-graph capture size 120 < the attested max_num_seqs 121" in out
    assert "decode batches of 121..121 sequences run PIECEWISE" in out
    rc, out = _run_standup_gates(tmp_path, ep={**_NATIVE_EP, "attested_cudagraph_full_capture_max": 48, "attested_max_num_seqs": 51,
                                               "max_num_seqs": 51, "geometry": {"max_model_len": 4864, "max_num_seqs": 51, "adjustments": []}},
                                 models=_NATIVE_MODELS, serving_path="lora_native")
    assert rc == 1 and "largest FULL CUDA-graph capture size 48 < the attested max_num_seqs 51" in out   # the incident
    rc, out = _run_standup_gates(tmp_path, ep={**_NATIVE_EP, "attested_cudagraph_full_capture_max": None}, models=_NATIVE_MODELS,
                                 serving_path="lora_native")
    assert rc == 1 and "attests no FULL CUDA-graph capture size (attested_cudagraph_full_capture_max=None)" in out
    rc, out = _run_standup_gates(tmp_path, ep={k: v for k, v in _NATIVE_EP.items() if k != "attested_cudagraph_full_capture_max"},
                                 models=_NATIVE_MODELS, serving_path="lora_native")
    assert rc == 1 and "attests no FULL CUDA-graph capture size" in out
    rc, out = _run_standup_gates(tmp_path, ep={**_NATIVE_EP, "attested_cudagraph_full_capture_max": 128}, models=_NATIVE_MODELS,
                                 serving_path="lora_native")
    assert rc == 0 and "REFUSING" not in out, out                                              # wider than S: fine
    rc, out = _run_standup_gates(tmp_path, ep=_NATIVE_EP, models=_LENS_MODELS, serving_path="lora_native")
    assert rc == 1 and "/v1/models petri_proxy.engine.kind=None" in out
    # the attested geometry compare runs on both engines, before any adjustment reading
    rc, out = _run_standup_gates(tmp_path, ep={**_NATIVE_EP, "attested_max_num_seqs": 120}, models=_NATIVE_MODELS, serving_path="lora_native")
    assert rc == 1 and "resolved geometry 121 x 4864 but the workers attest 120 x 4864" in out


def test_standup_gates_admit_a_lowered_s_only_through_the_recorded_adjustment_chain(tmp_path):
    """In the shell: a requested S (the 6th positional, gS) served LOWER is admitted iff geometry.adjustments
    records a chain whose first step starts at the requested S and whose last step ends at the attested S — [kernel]
    or [kernel, kv]; a bare smaller S, a chain from another S, or a chain that does not end where the workers attest
    refuses, as a clamped width always did."""
    kernel = {"kind": "lora_kernel_index_ceiling", "from_S": 121, "to_S": 107}
    kv = {"kind": "kv_shrink", "from_S": 107, "to_S": 90}

    def lowered(S, adj):
        return {**_NATIVE_EP, "max_num_seqs": S, "attested_max_num_seqs": S,
                "geometry": {"max_model_len": 4864, "max_num_seqs": S, "adjustments": adj}}
    rc, out = _run_standup_gates(tmp_path, ep=_NATIVE_EP, models=_NATIVE_MODELS, serving_path="lora_native", gS="121")
    assert rc == 0, out                                                    # the identity geometry at the requested S
    rc, out = _run_standup_gates(tmp_path, ep=lowered(107, [kernel]), models=_NATIVE_MODELS, serving_path="lora_native", gS="121")
    assert rc == 0 and "requested max_num_seqs 121 served at 107 through the RECORDED adjustment chain [lora_kernel_index_ceiling] (121 -> 107)" in out, out
    rc, out = _run_standup_gates(tmp_path, ep=lowered(90, [kernel, kv]), models=_NATIVE_MODELS, serving_path="lora_native", gS="121")
    assert rc == 0 and "[lora_kernel_index_ceiling,kv_shrink] (121 -> 90)" in out, out
    rc, out = _run_standup_gates(tmp_path, ep=lowered(107, []), models=_NATIVE_MODELS, serving_path="lora_native", gS="121")
    assert rc == 1 and "requested max_num_seqs 121 but the server attests 107 (no recorded adjustment chain from 121: [none]" in out, out
    rc, out = _run_standup_gates(tmp_path, ep=lowered(107, [kernel, kv]), models=_NATIVE_MODELS, serving_path="lora_native", gS="121")
    assert rc == 1 and "geometry.adjustments [lora_kernel_index_ceiling,kv_shrink] end at S=90 but the workers attest 107" in out, out
    rc, out = _run_standup_gates(tmp_path, ep=lowered(107, [{**kernel, "from_S": 150}]), models=_NATIVE_MODELS, serving_path="lora_native", gS="121")
    assert rc == 1 and "no recorded adjustment chain from 121: [lora_kernel_index_ceiling] 150->107" in out, out
    # a lowered S with NO request (the budget's S lowered by the kernel step, the common case): admitted when the
    # chain ends at the attested S, refused when it does not
    rc, out = _run_standup_gates(tmp_path, ep=lowered(107, [kernel]), models=_NATIVE_MODELS, serving_path="lora_native")
    assert rc == 0, out
    rc, out = _run_standup_gates(tmp_path, ep={**lowered(107, [kernel]), "attested_max_num_seqs": 107, "max_num_seqs": 107,
                                               "geometry": {"max_model_len": 4864, "max_num_seqs": 107, "adjustments": [{**kernel, "to_S": 100}]}},
                                 models=_NATIVE_MODELS, serving_path="lora_native")
    assert rc == 1 and "end at S=100 but the workers attest 107" in out, out
    # the lens standup: the pre-landing compare byte for byte (a lens record carries no adjustments)
    rc, out = _run_standup_gates(tmp_path, ep={**_LENS_EP, "max_num_seqs": 107, "attested_max_num_seqs": 107}, models=_LENS_MODELS,
                                 serving_path="lens", gS="121")
    assert rc == 1 and "requested max_num_seqs 121 but the server attests 107" in out, out


def _run_served_cells(tmp_path, *, cells_served, all_cells, engine="lora_native"):
    """f22_standup's SERVED_CELLS derivation (ALL_CELLS minus cells_served.json's cells_skipped, harness ids
    translated back to the config's TrainTest spellings), RUN under `set -u`. ``cells_served=None`` = no record."""
    out = tmp_path / f"served-{len(list(tmp_path.glob('served-*')))}"
    out.mkdir()
    ep = {"cells_served": str(out / "cells_served.json")} if cells_served is not None else {}
    (out / "endpoint.json").write_text(json.dumps(ep))
    if cells_served is not None:
        (out / "cells_served.json").write_text(json.dumps(cells_served))
    script = ("set -u\nf22_log(){ echo \"LOG $*\"; }\n"
              f"SPY={sys.executable}\nREPO_DIR={REPO}\nout={out}\nTARGET_ENGINE={engine}\n"
              "ALL_CELLS=(" + " ".join(all_cells) + ")\n"
              "served(){\n  local ep=\"$out/endpoint.json\"\n" + _standup_slice(_SERVED_START, _SERVED_END)
              + "\n  echo \"SERVED=${SERVED_CELLS[*]}\"\n  return 0\n}\nserved; echo \"rc=$?\"\n")
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env={**os.environ})
    assert r.returncode == 0, (r.stdout, r.stderr)
    return int(r.stdout.strip().splitlines()[-1].split("rc=")[1]), r.stdout + r.stderr


def test_served_cells_are_all_cells_minus_the_standups_skipped_cells_in_traintest_spelling(tmp_path):
    cells = ["none", "constv2ctl2-full__lora_r4_ce", "constv2ctl2-full__psr_ce_all", "prompted__safety_refusal"]
    skipped = {"engine": "lora_native", "cells_skipped": [{"cell_id": "si_constv2ctl2_full__psr_ce_all", "kind": "psr",
                                                           "reason": "no PEFT form"}]}
    rc, out = _run_served_cells(tmp_path, cells_served=skipped, all_cells=cells)
    assert rc == 0 and "SERVED=none constv2ctl2-full__lora_r4_ce prompted__safety_refusal" in out, out
    assert "cells SKIPPED BY NAME on the lora_native engine" in out
    # a lens standup skips nothing: SERVED_CELLS == ALL_CELLS — with an empty list, and with no record at all
    rc, out = _run_served_cells(tmp_path, cells_served={"cells_skipped": []}, all_cells=cells, engine="lens:stacked")
    assert rc == 0 and f"SERVED={' '.join(cells)}" in out and "SKIPPED BY NAME" not in out, out
    rc, out = _run_served_cells(tmp_path, cells_served=None, all_cells=cells, engine="lens:stacked")
    assert rc == 0 and f"SERVED={' '.join(cells)}" in out, out
    # every cell skipped (a view of PSR cells only): refused — nothing to drive
    rc, out = _run_served_cells(tmp_path, all_cells=["constv2ctl2-full__psr_ce_all"],
                                cells_served={"cells_skipped": [{"cell_id": "si_constv2ctl2_full__psr_ce_all", "kind": "psr", "reason": "r"}]})
    assert rc == 1 and "REFUSING TRAFFIC: the standup serves none of constv2ctl2-full__psr_ce_all" in out, out


def test_served_cells_filter_narrows_a_jobs_loops_and_refuses_an_explicit_ask_for_a_skipped_cell():
    """common.sh f22_served_cells_filter, RUN: a whole-view job's cell array drops the cells the standup skipped by
    name (logged); a cells= positional naming a skipped cell is FATAL naming the lens escape hatch; before
    f22_standup, or with nothing served, FATAL."""
    funcs = _bash_funcs("f22_served_cells_filter")
    base = ("set -u\nf22_log(){ echo \"LOG $*\"; }\nf22_fatal(){ echo \"FATAL $*\"; exit 1; }\nTARGET_ENGINE=lora_native\n"
            + funcs + "\n")

    def run(tail):
        return subprocess.run(["bash", "-c", base + tail], capture_output=True, text=True)
    r = run('SERVED_CELLS=(none a b); CELLS=(none a psr); f22_served_cells_filter CELLS ""; echo "CELLS=${CELLS[*]}"')
    assert r.returncode == 0 and "CELLS=none a\n" in r.stdout and "dropped from this job's loops: psr (served: none a)" in r.stdout, r.stdout
    r = run('SERVED_CELLS=(none a b); CELLS=(none a); f22_served_cells_filter CELLS ""; echo "CELLS=${CELLS[*]}"')
    assert r.returncode == 0 and "CELLS=none a\n" in r.stdout and "dropped" not in r.stdout        # a lens job: nothing skipped, nothing said
    r = run('SERVED_CELLS=(none a b); CELLS=(none a psr); f22_served_cells_filter CELLS "psr"')
    assert r.returncode == 1 and "cells=psr names psr, which this standup does not serve on the lora_native engine" in r.stdout
    assert "F22_SERVING_PATH=lens serves it on the lens" in r.stdout
    r = run('SERVED_CELLS=(none a); G=(a psr); f22_served_cells_filter G "goalset:/g"')
    assert r.returncode == 1 and "cells=goalset:/g names psr" in r.stdout                           # the goal-set spelling
    r = run('CELLS=(none a); f22_served_cells_filter CELLS ""')
    assert r.returncode == 1 and "f22_served_cells_filter before f22_standup (SERVED_CELLS unset)" in r.stdout
    r = run('SERVED_CELLS=(none a); CELLS=(psr vec); f22_served_cells_filter CELLS ""')
    assert r.returncode == 1 and "no cell of psr vec is served by this standup" in r.stdout


def test_every_sbatch_loop_reads_the_served_cells_after_its_standup_and_the_drivers_get_the_engine():
    """The wave scripts against the seams: the serving path and TARGET_ENGINE derived once in f22_init; the lens pins
    on mt_serving_up's argv ONLY under lens; every sbatch loop over the view's cells narrowed to SERVED_CELLS right
    after its standup and before the positive control / the first driver loop (a cells= positional — the smoke's one
    cell, the goal-set's pairs — passed as the explicit ask); the BoN legs and the positive control pass the wave's
    TARGET_ENGINE as --require-engine; standup_jobs.jsonl records engine, serving_path, cells_skipped, adjustments."""
    sh = (F22 / "common.sh").read_text()
    assert "  f22_select_serving_path\n" in sh and "export F22_SERVING_PATH TARGET_ENGINE" in sh   # the ONE resolution (kind-aware, 2026-09-15)
    assert 'F22_SERVING_PATH="${F22_SERVING_PATH:-lora_native}"' not in sh
    assert 'if [ "$F22_SERVING_PATH" = lens ]; then TARGET_ENGINE="lens:$REDUCTION_MODE"; else TARGET_ENGINE=lora_native; fi' in sh
    assert '[ "$serving_path" = lens ] && lens_pins=(--reduction-mode stacked --require-attested-mode stacked)' in sh
    assert '--serving-path "$serving_path" ${lens_pins[@]+"${lens_pins[@]}"} \\' in sh
    assert '    --reduction-mode stacked --require-attested-mode stacked \\' not in sh   # the lens pins no longer ride unconditionally
    assert '--require-engine "${TARGET_ENGINE:-lens:stacked}"' in sh                    # f22_positive_control
    assert '"engine":"%s","serving_path":"%s","cells_skipped":%s,"geometry_adjustments":%s,"lens_reduction_mode":%s' in sh
    assert 'f22_record_si_prefix(){' in sh and 'elif [ "${SERVED_CELLS+x}" = x ]; then cells=("${SERVED_CELLS[@]}")' in sh
    plan = {"target_attacks.sbatch": ('f22_served_cells_filter CELLS "$CELLS_ARG"', 1),
            "target_static.sbatch": ('f22_served_cells_filter CELLS "$CELLS_ARG"; NCELLS=${#CELLS[@]}', 2),
            "bon_climb.sbatch": ('f22_served_cells_filter CELLS "$CELLS_ARG"', 1),
            "target_attacks_goalset.sbatch": ('f22_served_cells_filter CELLS_WITH_GOALS "goalset:$GOALSET_DIR"', 1),
            "smoke_full22.sbatch": ('f22_served_cells_filter CELLS "$CELL"', 3)}
    for name, (line, n) in plan.items():
        src = (F22 / name).read_text()
        filters = [m.start() for m in re.finditer(re.escape(line), src)]
        assert len(filters) == n, (name, len(filters))
        standups = [m.start() for m in re.finditer(r"f22_standup attacks ", src)]
        anchors = [m.start() for m in re.finditer(r"f22_positive_control|positive_control static|positive_control attacks|"
                                                  r'for cell in "\$\{CELLS\[@\]\}"|served_capability_run\.py generate', src)]
        prev = -1
        for f in filters:
            assert any(prev < st < f for st in standups), (name, "a filter must follow its standup")   # a standup between filters
            assert any(a > f for a in anchors), (name, "a filter must precede the control / the first driver")
            prev = f
        # target_attacks: ONE filter after BOTH branches of the standup if/else
        if name == "target_attacks.sbatch":
            assert filters[0] > max(standups)
    bc = (F22 / "bon_climb.sbatch").read_text()
    assert '--require-engine "$TARGET_ENGINE"' in bc and "--require-lens-mode stacked" not in bc
    assert (F22 / "target_static.sbatch").read_text().count('--require-engine "$TARGET_ENGINE"') == 1     # the BoN leg
    assert (F22 / "smoke_full22.sbatch").read_text().count('--require-engine "$TARGET_ENGINE"') == 2       # the control + the BoN leg
    sub = (F22 / "submit_full22.sh").read_text()
    assert "F22_SERVING_PATH=lora_native" in sub and "F22_SERVING_PATH=lens" in sub


_DRIVER_SEAMS = {
    "bon_served_run": ("[bon-served]", ("pe.scale0_twin(route, routes)", "bs.is_steered(route)")),
    "bon_climb_run": ("[bon-climb]", ("pe.scale0_twin(route, routes)", "bs.is_steered(route)")),
    "validate_derived_serving": ("[validate]", ('NOOP_CELL_BY_ENGINE["lora_native" if require_engine == "lora_native" else "lens"]',)),
    "smoke_positive_control": ("[positive-control]", ("pe.scale0_twin(route, routes)", "bs.is_steered(route)")),
}


@pytest.mark.parametrize("name", sorted(_DRIVER_SEAMS))
def test_the_wave_drivers_assert_the_engine_through_the_one_helper_with_a_byte_identical_default(name):
    """The four python wave drivers: --require-engine (lens:stacked | lens:per_row | lora_native), default
    lens:<--require-lens-mode / --expect-mode> so every existing invocation asserts exactly what it asserted; the
    identity block judged by proxy_engine.served_engine_problem (the ONE helper), never a lens_reduction_mode compare;
    the steered legs' twin the PATH-MATCHED zero (proxy_engine.scale0_twin) over bon_served.is_steered."""
    import importlib

    from sisteer.capeval.serving import proxy_engine as pe
    tag, seams = _DRIVER_SEAMS[name]
    m = importlib.import_module(name)
    assert m.ENGINE_EXPECTATIONS == pe.ENGINE_EXPECTATIONS == ("lens:stacked", "lens:per_row", "lora_native")
    assert m.require_engine_of(None, "stacked", tag=tag) == "lens:stacked"
    assert m.require_engine_of(None, "per_row", tag=tag) == "lens:per_row"
    assert m.require_engine_of("lens:stacked", "stacked", tag=tag) == "lens:stacked"
    assert m.require_engine_of("lora_native", "stacked", tag=tag) == "lora_native"
    with pytest.raises(SystemExit, match="name one engine"):
        m.require_engine_of("lens:per_row", "stacked", tag=tag)
    with pytest.raises(SystemExit, match="not one of"):
        m.require_engine_of("native", "stacked", tag=tag)
    native = {"engine": {"kind": "lora_native", "adapters": {"none__lora0": {}, "si_x": {}}}, "thinking_mode": "off"}
    lens = {"lens_reduction_mode": "stacked", "thinking_mode": "off"}
    assert m._served_engine_problem(native, expect="lora_native") is None
    assert m._served_engine_problem(lens, expect="lens:stacked") is None
    assert "not a native multi-LoRA standup" in m._served_engine_problem(lens, expect="lora_native")
    assert "native multi-LoRA engine" in m._served_engine_problem(native, expect="lens:stacked")
    assert "torn identity" in m._served_engine_problem({**native, "lens_reduction_mode": "stacked"}, expect="lora_native")
    assert "not 'stacked'" in m._served_engine_problem({"lens_reduction_mode": "per_row"}, expect="lens:stacked")
    src = (F22 / f"{name}.py").read_text()
    assert '"--require-engine", choices=list(ENGINE_EXPECTATIONS), default=None' in src
    assert 'ident.get("lens_reduction_mode") != args.require' not in src
    for seam in seams:
        assert seam in src, (name, seam)
    if name == "validate_derived_serving":
        assert m.NOOP_CELL_BY_ENGINE == {"lens": "none__lens0", "lora_native": "none__lora0"} and m.NOOP_CELL == "none__lens0"
    if name in ("bon_served_run", "bon_climb_run"):
        ns = m.build_parser().parse_args(["--config", "c", "--defense", "d", "--proxy-url", "u", "--routing-table", "t",
                                          "--max-model-len", "4864", "--max-concurrency", "4"]
                                         + (["--manifest", "m", "--shard", "0", "--nshards", "1", "--out", "o"] if name == "bon_climb_run"
                                            else ["--private-root", "r"]))
        assert ns.require_engine is None and ns.require_lens_mode == "stacked"
        with pytest.raises(SystemExit):
            m.build_parser().parse_args(["--require-engine", "native"])


def test_smoke_determinism_is_diagnostic_under_the_native_engine_and_the_smoke_names_failed_controls_and_its_thinking_arm():
    """Under the native engine (llama8b it, constv2-mix-full__lora_r4_ce): the standup
    passed the core's five positive controls and every steer-is-live check, and the smoke's positive control FAILED on
    `determinism` alone — 1 of 8 greedy prompts repeating non-identically on the STEERED leg (anchor + twin bit-exact).
    That is the native multi-LoRA kernels' measured batch-variance (the zero adapter is bit-exact solo, not under concurrency), the
    reason admission is FUNCTIONAL (multilora_admission functional-2026-09-13, numerics.gate false). So under
    lora_native the driver records `determinism` as DIAGNOSTIC (gate false, outcome, the same
    payload) and excludes it from all_pass — decided by ENGINE NAME in record(), never by the reading; under every lens
    engine the record is byte-identical to before. The full end-to-end runs are tests/test_smoke_positive_control_
    native_diagnostic.py; this pins the seams by source text.

    Two record-hygiene fixes in the smoke sbatch from the same job: (1) the FAIL detail of `positive_control_<tag>`
    was `tail -5 | head -c 600` of the driver log and ended mid-JSON inside a PASS entry, hiding the failing control's
    name — it now reads the record (pc_fail_detail): the FAILED controls first with their whole payloads, then the
    DIAGNOSTIC ones, the passing names, the record path, never truncated; (2) the controls leg ran
    served_capability_run.py with a tinycap control (gsm8k) and NO --thinking, which the driver refuses by name
    (contracts.FLAG_THINKING, P33: the arm is never implicit) — main's smoke carries the identical invocation, so this
    is a pre-existing main defect fixed on the branch. The arm is the standup's ATTESTED regime, TARGET_THINKING_MODE
    (f22_standup exports it from endpoint.json + /v1/models after refusing a mismatch with F22_THINKING_ON): off by
    default, on under F22_THINKING_ON — the stamp, not the variable (the authoritative-source rule). The smoke's list
    is the VIEW's (llama8b's carries no tinycap control) and the flag rides every list as the wave's does,
    inert on a legacy-only one; the argv is the wave's with the smoke's limits appended (the client-venv test pins the
    equality), so the pin below reads the wave's flag order."""
    import smoke_positive_control as spc
    from sisteer.capeval.tinycap import contracts as tc
    src = (F22 / "smoke_positive_control.py").read_text()
    assert spc.DIAGNOSTIC_STATUS == "DIAGNOSTIC" and spc.NATIVE_DIAGNOSTIC_CHECKS == ("determinism",)
    assert spc.diagnostic_checks_for("lora_native") == ("determinism",)
    assert spc.diagnostic_checks_for("lens:stacked") == () == spc.diagnostic_checks_for("lens:per_row")
    assert "functional-2026-09-13" in spc.DIAGNOSTIC_NOTES["determinism"]
    # the dispatch lives in record(), keyed by name x asserted engine, resolved BEFORE the first check runs
    assert src.index("diagnostic_checks = diagnostic_checks_for(require_engine)") < src.index("    def record(name: str, ok: bool, **detail):")
    assert "        if name in diagnostic_checks:" in src
    assert '"status": DIAGNOSTIC_STATUS, "gate": False, "outcome": outcome, **detail,' in src
    assert 'record("determinism", not any(nondet.values())' in src      # the call is unchanged (newmodels pin); no per-call gate flag
    assert "nondeterministic_prompts=nondet)" in src                        # the call carries no per-call gate flag either
    assert "(diagnostic, not gating: " in src                               # the summary line names it
    smoke = (F22 / "smoke_full22.sbatch").read_text()
    # (1) the FAIL detail from the record, whole
    assert 'check "positive_control_$1" FAIL "$(pc_fail_detail "$out" "$SERVE_DIR/drivers/positive_control_$1.log")"' in smoke
    assert 'tail -5 "$SERVE_DIR/drivers/positive_control_$1.log"' not in smoke
    helper = smoke[smoke.index("pc_fail_detail(){"):smoke.index("positive_control(){")]
    helper_code = "\n".join(l for l in helper.splitlines() if not l.strip().startswith("#"))   # its comment names the old cut
    assert "head -c" not in helper_code and "tail -5" not in helper_code
    assert 'DIAGNOSTIC = "DIAGNOSTIC"' in helper                                    # the sbatch's literal == the driver's vocabulary
    assert 'failed = [c for c in checks if c.get("status") not in ("PASS", DIAGNOSTIC)]' in helper
    assert 'check "positive_control_$1" PASS' in smoke                               # the PASS branch's detail kept
    # (2) the thinking arm, explicit, from the attested stamp; the rest of the invocation byte-identical
    assert ('--controls "$VIEW_CONTROLS" --concurrency 32 --top-logprobs "$CTL_TOP_LOGPROBS" \\\n'
            '          --thinking "$TARGET_THINKING_MODE" \\\n'
            '          --vllm-log "$SERVE_DIR/controls/vllm_server.log" \\\n'
            '          --limit 8 --max-goals 8 )') in smoke
    assert smoke.count('--thinking "$TARGET_THINKING_MODE"') == 1 == smoke.count(f'{tc.FLAG_THINKING} "')   # one invocation (the comment names the bare flag)
    assert 'TARGET_THINKING_MODE="$ep_think"' in (F22 / "common.sh").read_text()  # set by f22_standup from the stamp
    assert tc.FLAG_THINKING == "--thinking" and tc.THINKING_OFF == "off" and tc.THINKING_ON == "on"
    assert tc.THINKING_OFF in tc.LAUNCHER_THINKING_MODES and tc.THINKING_ON in tc.LAUNCHER_THINKING_MODES
    assert "llama" in tc.NO_THINKING_CHANNEL_FAMILIES                                # llama8b: `off` recorded as not_available
    assert tc.effective_arm_for_family(tc.THINKING_OFF, "llama") == tc.THINKING_NOT_AVAILABLE


def test_controls_driver_runs_from_the_client_venv_in_the_smoke_and_the_wave():
    """Smoke 502126 (llama8b `it`, native) lost the controls DRIVER to "framework 'inspect_ai' is required by a
    requested control but cannot be imported" — the smoke ran `served_capability_run.py generate --controls
    gsm8k,...` under $MPY, the DRIVER venv (uv sync --extra cu128: no inspect_ai), while gsm8k is an Inspect
    control (contracts.INSPECT_CONTROLS, dispatched to inspect_served) that needs inspect_ai + inspect_evals at
    the client lock's pins — the CLIENT venv (common.sh f22_client_venv -> $CPY). mmlu / xstest / or_bench_hard_1k
    are the driver's own SERVED_CONTROLS and need no framework. The wave had the same defect on the gemma4e4b /
    qwen35_4b / nemotron3nano4b `main` views (roster controls gsm8k,mmlu,xstest,or_bench_hard_1k).

    An earlier fix moved EVERY controls job to the client venv, but the client lock is 259 pinned packages incl. a CPU
    torch, minutes on a fresh node, and an unrelated Inspect dependency failure aborted a llama8b controls job (legacy
    controls only) that never imports Inspect. So THE INTERPRETER FOLLOWS THE LIST: the client venv $CPY iff the view's
    roster list intersects contracts.INSPECT_CONTROLS — read BY IMPORT through common.sh f22_inspect_controls under
    $MPY, never a hand-copied list — else the driver venv $MPY as before (CTL_PY). The SMOKE was then left on a fixed
    SUPERSET list (gsm8k + the three legacy controls, so the Inspect path was always exercised) and therefore always on
    the client venv. In smoke 503471 (llama8b it, native) the superset's three legacy controls ran to their stores and
    its Inspect gsm8k control then OVERFLOWED the controls window (prompts up to 1,537 tokens + max_tokens 512 >
    capability:short 2048) — a sizing defect of main's wave for the gsm8k-bearing views, recorded as wave debt, and not
    a smoke of a view that never runs gsm8k to find. THE SMOKE'S CONTROLS LIST IS THE VIEW'S — read from the
    roster with the wave's own roster_view.py line, gated with the wave's own gate (the body byte-identical from the
    CTL_INSPECT read down; the smoke adds only the wave's empty-list refusal, by name, since it has no family parse),
    and driven with THE WAVE'S ARGV flag for flag in order, at the smoke's cell, out-root and width, then its limits.
    The per-interpreter map by roster kind — the llama8b views -> $MPY, no bootstrap; every gsm8k-bearing view -> $CPY
    + the datasets guard + the offline env — is EXECUTED here for BOTH scripts against every roster view's real controls
    list and asserted identical between them, view for view.

    Both legs read the `datasets` library OFFLINE from each user's prepared cache when an Inspect control is in the
    list (smoke 502566: the gsm8k control looked for `openai___gsm8k/default-data_dir=main` under the SHARED model
    cache and missed) — common.sh f22_datasets_guard: HF_DATASETS_CACHE=$SHARED/hf_datasets (g8k_init's rule, the
    same default expression), each Inspect control's offline config dir (g8k_cells.DATASETS_CONFIG_DIRS, by
    import) present by name before any standup, and HF_DATASETS_CACHE + HF_DATASETS_OFFLINE=1 in the driver's env
    prefix (CTL_ENV, empty for a legacy-only list — a literal HF_DATASETS_CACHE="$HF_DATASETS_CACHE" would be an unbound
    variable under set -u when the guard never ran). The one-number logprob rule (MMLU_TOP_LOGPROBS read under $MPY ->
    the standup's 7th arg AND --top-logprobs), the thinking arm and the Inspect bearer variable stay as the first fix
    pinned them, the last two now under the DECIDED interpreter. Every other driver stays on $MPY."""
    from sisteer.capeval.tinycap import contracts as tc
    smoke = (F22 / "smoke_full22.sbatch").read_text()
    wave = (F22 / "target_static.sbatch").read_text()
    common = (F22 / "common.sh").read_text()
    g8k = (REPO / "scripts/sdfbase/gsm8k/g8k_common.sh").read_text()
    drv = (REPO / "scripts/served_capability_run.py").read_text()
    # the driver's rule, and which controls it binds: gsm8k is Inspect; the three legacy controls are the driver's own
    assert "is required by a requested control " in drv and "install the client " in drv
    assert 'SERVED_CONTROLS = ("mmlu", "xstest", "or_bench_hard_1k")' in drv
    assert tc.CONTROL_GSM8K in tc.INSPECT_CONTROLS and not {"mmlu", "xstest", "or_bench_hard_1k"} & set(tc.INSPECT_CONTROLS)
    # the client venv is the lock's, sets CPY and guards the frameworks the driver imports; the driver venv never had them
    block = common[common.index("f22_client_venv(){"):common.index("f22_inspect_controls(){")]
    assert 'CPY="$CLIENT_VENV/bin/python"' in block and "import inspect_ai, inspect_evals" in block
    assert "inspect" not in common[common.index("f22_driver_venv(){"):common.index("f22_serving_venv(){")]
    venvs = 'f22_driver_venv; f22_serving_venv; f22_bank_view "$MPY"'
    # THE TWO HELPERS: the Inspect vocabulary by import (no literal control name in the helper), the datasets guard with
    # the gsm8k cache default, its config dirs by import, and the recipe in its refusal
    inspect_fn = common[common.index("f22_inspect_controls(){"):common.index("f22_datasets_guard(){")]
    guard_fn = common[common.index("f22_datasets_guard(){"):common.index("# --- the served target")]
    assert "from sisteer.capeval.tinycap import contracts as tc" in inspect_fn and "tc.INSPECT_CONTROLS" in inspect_fn
    assert not re.search(r"\b(gsm8k|math500|ifeval)\b", inspect_fn), "the Inspect vocabulary is read by import, never listed"
    cache_default = 'export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$'
    assert cache_default + 'SHARED/hf_datasets}"' in guard_fn and cache_default + 'shared/hf_datasets}"' in g8k     # the same rule
    assert "import g8k_cells as g" in guard_fn and "g.DATASETS_CONFIG_DIRS[ds]" in guard_fn
    assert "by_control = {tc.CONTROL_GSM8K:" in guard_fn and "no prepared-datasets recipe" in guard_fn
    assert "HF_DATASETS_OFFLINE" not in guard_fn                       # the guard names the cache; the DRIVER env goes offline
    # THE ROSTER READ, one line, the same in BOTH scripts: the view's controls list through roster_view.py
    # (f22_roster.view_controls), a failed read fatal by name; before the venvs in both (a roster refusal costs no
    # bootstrap). The smoke's fixed superset is gone — its name, its literal and its rationale
    read = ('VIEW_CONTROLS=$(python3 scripts/sdfbase/full22/roster_view.py "$KEY" "$VIEW" controls --roster "$F22_ROSTER") '
            '|| f22_fatal "cannot read the roster\'s controls list for $KEY/$VIEW"')   # the job's roster on every read
    assert smoke.count(read) == 1 and wave.count(read) == 1
    assert "SMOKE_CONTROLS" not in smoke and "gsm8k,mmlu,xstest,or_bench_hard_1k" not in smoke
    assert "the smoke exists to exercise the Inspect path" not in smoke
    assert "D72" in smoke and "tests THE WAVE PATH FOR THE VIEW IT NAMES" in smoke and "wave debt" in smoke
    assert "proven on Modal (J4" in smoke                                        # where the native Inspect gsm8k control IS proven
    # THE GATE: CTL_PY / CTL_INSPECT / CTL_ENV decided right after the venvs and before any standup in BOTH scripts; the
    # client venv and the datasets guard only behind a non-empty Inspect intersection; the body BYTE-IDENTICAL from the
    # CTL_INSPECT read down (the wave's gate IS the smoke's) — only the condition differs (the wave's family bit, the
    # smoke's legs= bit), and the smoke adds the wave's empty-list refusal, which the wave makes at its family parse
    head = 'CTL_PY="$MPY"; CTL_INSPECT=""; CTL_ENV=()\n'

    def gate_of(src):
        g = src[src.index(head):]
        return g[:g.index("\nfi\n") + len("\nfi\n")]

    wgate, sgate = gate_of(wave), gate_of(smoke)
    assert wave.count(head) == 1 and smoke.count(head) == 1
    assert wave.index(read) < wave.index(venvs) < wave.index(wgate) < wave.index("f22_window_check")
    assert smoke.index(read) < smoke.index(venvs) < smoke.index(sgate) < smoke.index("f22_standup attacks ")
    body = '  CTL_INSPECT=$(f22_inspect_controls "$VIEW_CONTROLS") || f22_fatal "cannot read contracts.INSPECT_CONTROLS under $MPY"\n'
    assert wgate[wgate.index(body):] == sgate[sgate.index(body):]
    assert wgate[:wgate.index(body)] == head + 'if [ "$CONTROLS" = 1 ]; then\n'
    refuse = '  [ -n "$VIEW_CONTROLS" ] || f22_fatal "view $KEY/$VIEW has neither a capability nor a controls leg'
    assert sgate[:sgate.index(body)].startswith(head + "if has_leg controls; then\n" + refuse)
    assert 'controls) [ -n "$VIEW_CONTROLS" ] || f22_fatal "view $KEY/$VIEW has neither a capability nor a controls leg' in wave
    assert 'if [ -n "$CTL_INSPECT" ]; then\n    f22_client_venv; CTL_PY="$CPY"; f22_datasets_guard "$CTL_INSPECT"\n' in wgate
    assert 'CTL_ENV=(HF_DATASETS_CACHE="$HF_DATASETS_CACHE" HF_DATASETS_OFFLINE=1)' in wgate
    for src in (wave, smoke):
        assert re.findall(r"\bf22_client_venv;", src) == ["f22_client_venv;"]               # ONE call site each, the gated one
        assert 'if [ "$CONTROLS" = 1 ]; then f22_client_venv; fi' not in src                  # the r1 gate is gone
        assert 'f22_datasets_guard "$(f22_inspect_controls' not in src                        # ... and the smoke's r2 one-liner
        assert src.count("f22_datasets_guard ") == 1                                          # the guard runs inside the gate only
    # the one-number logprob rule, mirrored from the wave: the same read, the standup's 7th arg, the driver's flag
    ctl_read = "print(int(m.MMLU_TOP_LOGPROBS))' scripts/served_capability_run.py)"
    assert ctl_read in smoke and ctl_read in wave
    assert 'elif f22_standup attacks "$SERVE_DIR/controls" "" capability:short "$CTL_L" "" "$CTL_TOP_LOGPROBS"; then' in smoke
    assert "capability:short 2048; then" not in smoke                                  # the engine-default cap is gone
    # the controls window is composed as the wave composes it (since 2026-09-14): the view's own, else the roster's gsm8k
    # window for a gsm8k-bearing table view, else the table's; the gsm8k window check's refusal is a recorded FAIL, not an abort
    compose = 'CTL_L="${VIEW_WINDOW_CONTROLS:-${VIEW_WINDOW_GSM8K:-2048}}"'
    assert smoke.count(compose) == 1 == wave.count(compose)
    assert 'elif ! ( f22_gsm8k_window_check "$CTL_L" "$VIEW_CONTROLS" ); then' in smoke
    assert 'f22_gsm8k_window_check "$CTL_L" "$VIEW_CONTROLS"' in wave
    for name in ("window controls", "window gsm8k"):
        read_w = f'$(python3 scripts/sdfbase/full22/roster_view.py "$KEY" "$VIEW" {name} --roster "$F22_ROSTER")'
        assert smoke.count(read_w) == 1 == wave.count(read_w)
    assert "PAID 2026-09-14" in smoke and "not fixed here" not in smoke
    assert 'check standup_controls FAIL "cannot read MMLU_TOP_LOGPROBS' in smoke        # unreadable = a recorded FAIL, not a default
    assert 'attested; $TARGET_ENGINE; max_logprobs=$CTL_TOP_LOGPROBS"' in smoke        # and the PASS detail records the cap
    # THE DRIVER ARGV: the wave's, through `env` (f22_queue_launch runs its argv inside `cd $REPO_DIR` with the GPU hidden;
    # the smoke's subshell does the same by hand), under $CTL_PY with CTL_ENV. The smoke's == the wave's after exactly
    # three substitutions (its cell, its out-root, its width) plus its limits appended — asserted on the whitespace-
    # normalised text so a flag added to one and not the other, or reordered, fails here
    wave_argv = ('env PYTHONPATH="$REPO_DIR/src:$REPO_DIR/scripts" TINYCAP_API_KEY="$CTL_PROXY_TOKEN" '
                 '${CTL_ENV[@]+"${CTL_ENV[@]}"} "$CTL_PY" scripts/served_capability_run.py generate \\\n'
                 '      --proxy-url "$PROXY_URL" --routing-table "$TABLE" --cell "$cell" --config "$CFG" \\\n'
                 '      --defenses-dir "$TT_TWO_SI_DEFENSES_DIR" --out-root "$WAVE_ROOT/controls/$TAG" \\\n'
                 '      --controls "$VIEW_CONTROLS" --concurrency "$CONC" --top-logprobs "$CTL_TOP_LOGPROBS" \\\n'
                 '      --thinking "$TARGET_THINKING_MODE" \\\n'
                 '      --vllm-log "$SERVE_DIR/controls/vllm_server.log"\n')
    assert wave.count('    f22_queue_launch "controls__${cell}" ' + wave_argv) == 1
    assert '( cd "$REPO_DIR" && CUDA_VISIBLE_DEVICES="" "$@" ) > "$SERVE_DIR/drivers/$tag.log" 2>&1 &' in common   # why `env`
    launch_head = '    if ( cd "$REPO_DIR" && CUDA_VISIBLE_DEVICES="" '
    launch_tail = ' ) > "$SERVE_DIR/drivers/controls__${CELL}.log" 2>&1; then'
    assert smoke.count(launch_head + "env PYTHONPATH=") == 1 and smoke.count(launch_tail) == 1
    smoke_argv = smoke[smoke.index(launch_head + "env PYTHONPATH=") + len(launch_head):smoke.index(launch_tail)]
    expect = (wave_argv.replace('--cell "$cell"', '--cell "$CELL"')
              .replace('--out-root "$WAVE_ROOT/controls/$TAG"', '--out-root "$SMOKE_ROOT/controls/$TAG"')
              .replace('--concurrency "$CONC"', "--concurrency 32").rstrip("\n") + " \\\n          --limit 8 --max-goals 8")

    def norm(t):
        return " ".join(t.replace("\\\n", " ").split())

    assert norm(smoke_argv) == norm(expect), (norm(smoke_argv), norm(expect))
    assert smoke_argv.endswith("\\\n          --limit 8 --max-goals 8")                # the limits are a SUFFIX, spelled once
    assert smoke.count("--limit 8 --max-goals 8") == 2                                 # the argv + the comment that names them
    assert '--controls "$VIEW_CONTROLS" --concurrency 32 --top-logprobs "$CTL_TOP_LOGPROBS" \\' in smoke
    assert 'HF_DATASETS_CACHE="$HF_DATASETS_CACHE" HF_DATASETS_OFFLINE=1 "$' not in smoke   # the literal env is gone: CTL_ENV or nothing
    assert smoke.count('--thinking "$TARGET_THINKING_MODE"') == 1 == wave.count('--thinking "$TARGET_THINKING_MODE"')
    assert 'TARGET_THINKING_MODE="$ep_think"' in common
    # THE INTERPRETER PER DRIVER, both jobs: the controls driver is the only one on $CTL_PY (never a literal $CPY — on a
    # llama view no client venv was bootstrapped and "$CPY" is an unbound variable under set -u); every other driver
    # stays on $MPY
    for src in (smoke, wave):
        assert re.findall(r'"\$CTL_PY" (scripts/\S+)', src) == ["scripts/served_capability_run.py"]
        assert re.findall(r'"\$CPY" (scripts/\S+)', src) == []
        assert '"$CPY"' not in src.replace('CTL_PY="$CPY"', "")                       # the gate's assignment is the ONE use
    assert re.findall(r'"\$MPY" (scripts/\S+)', smoke) == ["scripts/sdfbase/full22/smoke_positive_control.py", "scripts/jailbreak_run.py",
                                                            "scripts/sdfbase/full22/bon_served_run.py", "scripts/jailbreak_run.py"]
    assert re.findall(r'"\$MPY" (scripts/\S+)', wave) == ["scripts/sdfbase/full22/bon_served_run.py", "scripts/jailbreak_run.py"]
    assert '"$MPY" scripts/served_capability_run.py generate' not in smoke + wave
    # the Inspect client's bearer variable (contracts.INSPECT_API_KEY_ENV): smoke 502327 reached the gsm8k control under the
    # client venv and inspect_ai refused "No <key> defined in the environment" — the provider wants the variable even against
    # the unauthenticated loopback proxy. Both legs derive it from the controls standup's endpoint.json proxy_token, else
    # `petri` — the Modal launcher's rule (tinycap_served.py) and served_thinking.sh's st_proxy_token — under the DECIDED
    # interpreter $CTL_PY in both (json only, either venv reads it)
    tok = ('CTL_PROXY_TOKEN=$("$CTL_PY" -c \'import json,sys;print(json.load(open(sys.argv[1])).get("proxy_token") or "petri")\' '
           '"$SERVE_DIR/controls/endpoint.json")')
    assert smoke.count(tok) == 1 and wave.count(tok) == 1 and tc.INSPECT_API_KEY_ENV == "TINYCAP_API_KEY"
    for src in (smoke, wave):
        assert src.index(tok) < src.index('TINYCAP_API_KEY="$CTL_PROXY_TOKEN"')
    # the scope, from the roster: NO view names gsm8k any more — the wave's controls-leg gsm8k was RETIRED wave-wide on 2026-09-14
    # (roster controls_retired, applied by f22_roster.view_controls; llama8b's own capability_omitted of 2026-09-10
    # stays): the paired-200 lane is the paper's one GSM8K instrument. Every view's list is legacy-only, so every real
    # view decides $MPY; the Inspect branch of the gate is exercised below on the list a gsm8k-bearing view WOULD carry (the gate
    # is roster-agnostic: it reads the list it is handed)
    roster = fr.load_roster()
    assert all("gsm8k" not in fr.view_controls(roster, k, v) for k, tv in roster["targets"].items() for v in tv["views"])
    assert set(fr.controls_retired(roster)) == {"gsm8k"} and fr.target_capability_controls(roster, "gemma4e4b") == ("gsm8k", "mmlu")
    # THE GATES, EXECUTED under set -u — BOTH scripts', per roster view, against the REAL controls list of every view that
    # has one (the real f22_inspect_controls under the venv; stubs stand in for the venv bootstrap, which sets CPY, and the
    # datasets guard, which EXPORTS the cache it verified as the real one does): the client venv, the guard and the offline
    # env iff the list intersects contracts.INSPECT_CONTROLS; a legacy-only list stays on $MPY with no bootstrap; and the
    # smoke's decision == the wave's, view for view, byte for byte
    has_leg = next(l for l in smoke.splitlines() if l.startswith("has_leg(){"))
    stubs = ("f22_client_venv(){ echo BOOT; CPY=/client/python; }; "
             "f22_datasets_guard(){ echo \"GUARD[$1]\"; export HF_DATASETS_CACHE=/cache/hf_datasets; }; "
             "f22_log(){ :; }; f22_fatal(){ echo \"FATAL $*\"; exit 1; }")
    env = {**os.environ, "MPY": sys.executable, "REPO_DIR": str(REPO)}
    tail = 'echo "CTL_PY=$CTL_PY CTL_INSPECT=[$CTL_INSPECT] CTL_ENV=[${CTL_ENV[*]+"${CTL_ENV[*]}"}]"'

    def run_wave(controls, on=1):
        return subprocess.run(["bash", "-c", f"set -u; {stubs}\n{inspect_fn}\nCONTROLS={on}; VIEW_CONTROLS={controls}\n{wgate}{tail}"],
                              capture_output=True, text=True, env=env)

    def run_smoke(controls, legs):
        return subprocess.run(["bash", "-c", f"set -u; {stubs}\n{inspect_fn}\nKEY=k; VIEW=v; LEGS={legs}; {has_leg}; "
                               f"VIEW_CONTROLS={controls}\n{sgate}{tail}"], capture_output=True, text=True, env=env)

    driver_only = f"CTL_PY={sys.executable} CTL_INSPECT=[] CTL_ENV=[]\n"
    seen = {"client": set(), "driver": set()}
    for target, view, _t, _v in VIEWS:
        controls = ",".join(fr.view_controls(roster, target, view))
        if not controls:
            continue
        inspect = [c for c in controls.split(",") if c in tc.INSPECT_CONTROLS]
        w = run_wave(controls)
        assert w.returncode == 0 and w.stderr == "", (target, view, w)
        if inspect:
            want = (f"BOOT\nGUARD[{','.join(inspect)}]\nCTL_PY=/client/python CTL_INSPECT=[{','.join(inspect)}] "
                    "CTL_ENV=[HF_DATASETS_CACHE=/cache/hf_datasets HF_DATASETS_OFFLINE=1]\n")
            seen["client"].add(target)
        else:
            want = driver_only
            seen["driver"].add(target)
        assert w.stdout == want, (target, view, controls, w.stdout)
        # the smoke asked for the controls leg decides EXACTLY as the wave does on this view ...
        for legs in ("static,bon,controls", "controls", "controls,multiturn"):
            r = run_smoke(controls, legs)
            assert (r.returncode, r.stdout, r.stderr) == (0, w.stdout, ""), (target, view, legs, r)
        # ... and a smoke that does not ask for it decides nothing and bootstraps nothing, whatever the view's list
        for legs in ("static,bon", "multiturn", "static"):
            r = run_smoke(controls, legs)
            assert (r.returncode, r.stdout, r.stderr) == (0, driver_only, ""), (target, view, legs, r)
    # every target's views are legacy-only since the 2026-09-14 retirement: every controls job (and smoke) stays on the driver venv
    assert seen == {"client": set(), "driver": {"gemma4e4b", "qwen35_4b", "nemotron3nano4b", "glm46vflash", "llama8b"} | (set(ROSTER["targets"]) & set(gen.FRONT_DOOR_MODELS))}, seen
    # the Inspect branch, on the list a gsm8k-bearing view would carry (roster-agnostic; what the four targets ran until 2026-09-14):
    # the client venv, the datasets guard and the offline env, decided identically by the wave and the smoke
    for controls in ("gsm8k,mmlu,xstest,or_bench_hard_1k", "gsm8k,mmlu"):
        w = run_wave(controls)
        want = ("BOOT\nGUARD[gsm8k]\nCTL_PY=/client/python CTL_INSPECT=[gsm8k] "
                "CTL_ENV=[HF_DATASETS_CACHE=/cache/hf_datasets HF_DATASETS_OFFLINE=1]\n")
        assert (w.returncode, w.stdout, w.stderr) == (0, want, ""), (controls, w)
        for legs in ("static,bon,controls", "controls"):
            r = run_smoke(controls, legs)
            assert (r.returncode, r.stdout, r.stderr) == (0, want, ""), (controls, legs, r)
    # a wave job that does not run the controls family decides nothing and bootstraps nothing
    r = run_wave("gsm8k,mmlu", on=0)
    assert (r.returncode, r.stdout, r.stderr) == (0, driver_only, ""), r
    # the smoke's controls leg on a view the roster gives no controls is refused BY NAME before any bootstrap (the wave
    # refuses the `controls` family at its parse the same way); without the leg the same view smokes its other legs
    r = run_smoke("", "controls")
    assert r.returncode == 1 and r.stderr == "" and "BOOT" not in r.stdout, r
    assert r.stdout == ("FATAL view k/v has neither a capability nor a controls leg (the roster names no controls for it): "
                        "nothing for the smoke's controls leg to run — drop controls from legs=\n"), r.stdout
    r = run_smoke("", "static,bon")
    assert (r.returncode, r.stdout, r.stderr) == (0, driver_only, ""), r


# --- native PSR serving: the CERTIFICATION-ONLY roster, F22_ROSTER threaded through
# --- every roster reader, llama8b's `psrcert` view, and the smoke's output_diagnostics check --------------------------

CERT_ROSTER_PATH = REPO / "configs/sdfbase_full22_roster_psrcert.yaml"
CERT_ROSTER = yaml.safe_load(CERT_ROSTER_PATH.read_text())
PSRCERT_CELLS = {"constv2-mix-full__psr_ce_all": "base__constv2-mix-full__psr_ce_all",
                 "constv2-mix-full__lora_r4_ce": "base__constv2-mix-full__lora_r4_ce"}
PSRCERT_ALL_CELLS = ["none", "constv2-mix-full__psr_ce_all", "constv2-mix-full__lora_r4_ce"]


def test_certification_roster_is_the_production_roster_with_exactly_one_view_per_target():
    """llama8b's production views (`main`, `it`) carry no PSR cell and gemma4e4b's carry no view
    with a PSR cell, an Arditi object and a LoRA cell together; the certification-only roster names ONE view
    per target — `psrcert` (stage 1) and `psr2cert` (stage 2) — and every other key is the production
    roster's, pinned equal here so the two cannot drift (a wave-wide cap or a target-level fact that moves
    in production moves here in the same commit). The production roster is never edited for a certification
    (it must not even mention either view)."""
    assert list(CERT_ROSTER) == list(ROSTER)                                  # the same wave-wide keys, in order
    for k in ROSTER:
        if k != "targets":
            assert CERT_ROSTER[k] == ROSTER[k], k
    # the certification views per target: `psrcert` (stage 1) AND `pairedcert` (the paired write's second round,
    # llama8b's fitted teacher) on llama8b, `psr2cert` (stage 2) on gemma4e4b, `pairedcert` (the paired write) on
    # qwen35_4b — each target's block otherwise the production's ...
    assert list(CERT_ROSTER["targets"]) == ["llama8b", "gemma4e4b", "qwen35_4b"]
    for target, views in (("llama8b", ["psrcert", "pairedcert"]), ("gemma4e4b", ["psr2cert"]),
                          ("qwen35_4b", ["pairedcert"])):
        cert_t, prod_t = CERT_ROSTER["targets"][target], ROSTER["targets"][target]
        assert list(cert_t) == list(prod_t), target                           # ... carrying every target-level key ...
        for k in prod_t:
            if k != "views":
                assert cert_t[k] == prod_t[k], (target, k)                    # ... equal to the production's
        assert list(cert_t["views"]) == views, target                         # ... and exactly its own certification views
    cert_t, prod_t = CERT_ROSTER["targets"]["llama8b"], ROSTER["targets"]["llama8b"]
    v = cert_t["views"]["psrcert"]
    assert v["cells"] == PSRCERT_CELLS and list(v["cells"]) == list(PSRCERT_CELLS)   # the PSR cell first: the cert's tt_cell
    assert v["config"] == "configs/sdfbase_full22_llama8b_psrcert.yaml"
    assert v["manifest"] == "configs/capeval/manifests/petri_full22_llama8b_psrcert.yaml"
    assert set(v) == {"config", "manifest", "cells"}                          # like llama8b main: no legs / dose / serving key
    # both staged ids are the PRODUCTION bank's, pinned there (the builder's sha gate applies unchanged)
    for cid, sid in PSRCERT_CELLS.items():
        assert PINS["llama8b"][sid]["object_sha256"] and len(PINS["llama8b"][sid]["object_sha256"]) == 64, sid
        assert ra.ARM_OBJECT_KINDS[arm_of(cid)] == ("psr" if "psr" in cid else "lora_all")
    assert PINS["llama8b"]["base__constv2-mix-full__psr_ce_all"]["object_sha256"].startswith("4df62c74")
    assert PINS["llama8b"]["base__constv2-mix-full__lora_r4_ce"]["object_sha256"].startswith("504b6651")
    # the roster grammar holds on it through the ONE reader: every staged id exactly once, the view's legs / controls
    # the production llama8b views' (no `legs:` key), no prompted cell, not prompted-only, no dose
    # llama8b's two certification views (psrcert, pairedcert) BOTH name the LoRA twin — one cell id, one staged
    # object, in two certifications' standups — so the per-target invariant here is: every staged id under
    # exactly ONE cell id (no aliasing), and within each view exactly once
    named: dict = {}
    for vw in CERT_ROSTER["targets"]["llama8b"]["views"]:
        cells = fr.staged_cells(CERT_ROSTER, "llama8b", vw)
        assert len(set(cells.values())) == len(cells), vw
        for cid, sid in cells.items():
            assert named.setdefault(sid, cid) == cid, (sid, named[sid], cid)
    assert sorted(named) == ["base__constv2-mix-full__lora_r4_ce", "base__constv2-mix-full__psr_ce_all",
                             "derived__constv2-mix-full__psr_ce_all_minus_ctl"]
    assert fr.staged_cells(CERT_ROSTER, "llama8b", "pairedcert") == {
        "constv2-mix-full__psr_ce_all_minus_ctl": "derived__constv2-mix-full__psr_ce_all_minus_ctl",
        "constv2-mix-full__lora_r4_ce": "base__constv2-mix-full__lora_r4_ce"}
    assert PINS["llama8b"]["derived__constv2-mix-full__psr_ce_all_minus_ctl"]["object_sha256"].startswith("3ee2adb9")
    assert fr.served_cells(CERT_ROSTER, "llama8b", "psrcert") == fr.staged_cells(CERT_ROSTER, "llama8b", "psrcert") == PSRCERT_CELLS
    assert fr.view_legs(CERT_ROSTER, "llama8b", "psrcert") == fr.LEGS == fr.view_legs(ROSTER, "llama8b", "main")
    assert fr.view_controls(CERT_ROSTER, "llama8b", "psrcert") == fr.view_controls(ROSTER, "llama8b", "main") == ("mmlu", "xstest", "or_bench_hard_1k")
    assert not fr.view_prompted_only(CERT_ROSTER, "llama8b", "psrcert") and gen.view_dose(CERT_ROSTER, "llama8b", "psrcert") is None
    assert [l for l, _, _ in fr.lanes_of(CERT_ROSTER, "llama8b", "psrcert")] == [] and [l for l, _, _ in fr.lanes_of(CERT_ROSTER, "llama8b", "main")] == ["candsi"]
    assert fr.target_caps(CERT_ROSTER, "llama8b") == fr.target_caps(ROSTER, "llama8b")
    # the production roster never learned of it; the certification roster says what it is
    prod_text = (REPO / "configs/sdfbase_full22_roster.yaml").read_text()
    assert "psrcert" not in prod_text and "psr_ce_all" not in "\n".join(l for l in prod_text.splitlines() if "llama8b" in l)
    cert_text = CERT_ROSTER_PATH.read_text()
    assert cert_text.startswith("# sdfbase_full22_roster_psrcert.yaml — a CERTIFICATION-ONLY roster")
    for m in ("D15", "NEVER a production view", "F22_ROSTER=configs/sdfbase_full22_roster_psrcert.yaml",
              "make_full22_configs.py --roster", "D17 / D21", "CROSS-BANK"):
        assert m in cert_text, m
    prod_text = (REPO / "configs/sdfbase_full22_roster.yaml").read_text()
    assert "psr2cert" not in prod_text and "misalign_baseline_pins" not in prod_text
    assert "pairedcert" not in prod_text and "reseed_paired_pins" not in prod_text


def test_certification_roster_regenerates_its_two_files_byte_identically_and_the_production_check_still_passes():
    """Deliverable 2: `make_full22_configs.py --roster <cert roster>` renders the psrcert config + manifest exactly the way
    the production pair is rendered (`--check --roster` holds them); the default `--check` (the production roster) is
    byte-identical to before — the header names the roster it came from, and for the production roster that is the
    literal every committed file already carries."""
    for extra in ([], ["--roster", str(CERT_ROSTER_PATH)]):
        rc = subprocess.run([sys.executable, str(REPO / "scripts/sdfbase/make_full22_configs.py"), "--check", *extra],
                            capture_output=True, text=True, cwd=REPO)
        assert rc.returncode == 0 and "DRIFT" not in rc.stdout, (extra, rc.stdout, rc.stderr)
        assert rc.stdout.count("ok: ") == (8 if extra else len(list(gen.outputs(ROSTER))))   # 4 views x (config, manifest)
    # roster_rel resolves its argument against the cwd before relativising to REPO, so the second call names the
    # roster from REPO (cwd-independent: the file passes from inside and outside the worktree alike)
    assert gen.roster_rel(gen.ROSTER) == "configs/sdfbase_full22_roster.yaml" == gen.roster_rel(REPO / "configs/sdfbase_full22_roster.yaml")
    assert gen.roster_rel(CERT_ROSTER_PATH) == "configs/sdfbase_full22_roster_psrcert.yaml"
    assert gen._cert_roster_note(gen.ROSTER) == "" and "CERTIFICATION-ONLY" in gen._cert_roster_note(CERT_ROSTER_PATH)
    # in-process: the two renderers with the roster path == the committed bytes; the default path == the production bytes
    for target, view, cfg_rel, man_rel in (
            ("llama8b", "psrcert", "configs/sdfbase_full22_llama8b_psrcert.yaml",
             "configs/capeval/manifests/petri_full22_llama8b_psrcert.yaml"),
            ("gemma4e4b", "psr2cert", "configs/sdfbase_full22_gemma4e4b_psr2cert.yaml",
             "configs/capeval/manifests/petri_full22_gemma4e4b_psr2cert.yaml"),
            ("qwen35_4b", "pairedcert", "configs/sdfbase_full22_qwen35_4b_pairedcert.yaml",
             "configs/capeval/manifests/petri_full22_qwen35_4b_pairedcert.yaml"),
            ("llama8b", "pairedcert", "configs/sdfbase_full22_llama8b_pairedcert.yaml",
             "configs/capeval/manifests/petri_full22_llama8b_pairedcert.yaml")):
        cfg_text = gen.render_config(CERT_ROSTER, PINS, target, view, roster_path=CERT_ROSTER_PATH)
        man_text = gen.render_manifest(CERT_ROSTER, target, view, roster_path=CERT_ROSTER_PATH)
        assert cfg_text == (REPO / cfg_rel).read_text(), cfg_rel
        assert man_text == (REPO / man_rel).read_text(), man_rel
    assert gen.render_config(ROSTER, PINS, "llama8b", "main") == (REPO / "configs/sdfbase_full22_llama8b.yaml").read_text()
    # a missing roster refuses by name
    rc = subprocess.run([sys.executable, str(REPO / "scripts/sdfbase/make_full22_configs.py"), "--check", "--roster", "/nonexistent/r.yaml"],
                        capture_output=True, text=True, cwd=REPO)
    assert rc.returncode != 0 and "no roster at" in rc.stderr


def test_psr2cert_names_the_arditi_object_through_its_pin_and_never_by_rung():
    """The stage-2 view's roster entry and its generated pair.

    The Arditi canary is named by the CROSS-BANK PIN FORM (`{pins, pin}` and nothing else),
    so the view serves the object the production misalignment pin chose (`ar_r0p05__arditi`,
    alpha 0.05) and the roster has no field to substitute a rung with. The generated config
    echoes the pin's own facts; the manifest carries the BARE class-B harness id."""
    v = CERT_ROSTER["targets"]["gemma4e4b"]["views"]["psr2cert"]
    assert set(v) == {"config", "manifest", "cells"}          # like gemma4e4b main: no legs / dose key
    assert list(v["cells"]) == ["safety_refusal__psr_kl_all", "si_constitution__psr_kl_all", "arditi", "constv2-mix-full__lora_r4_ce"]
    assert v["cells"]["safety_refusal__psr_kl_all"] == "si12__safety_refusal__psr_kl_all"
    # T5b: the constitution response-fit psr — the object the 2026-09-04 lens §7.2 rows were measured on
    # (c4de7681…), hand-staged into this bank on both clusters from their own scale-12 copies; certification-only
    assert v["cells"]["si_constitution__psr_kl_all"] == "si12__si_constitution__psr_kl_all"
    t5b = PINS["gemma4e4b"]["si12__si_constitution__psr_kl_all"]
    assert t5b["object_sha256"].startswith("c4de7681") and t5b["verified_cluster_a"] is True and t5b["verified_cluster_b"] is True
    assert t5b["source"] == "twosi-fits:objects/scale-12/gemma4e4b/si_constitution__psr_kl_all"
    assert set(t5b) == set(PINS["gemma4e4b"]["si12__safety_refusal__psr_kl_all"])
    assert v["cells"]["constv2-mix-full__lora_r4_ce"] == "base__constv2-mix-full__lora_r4_ce"
    mapping = v["cells"]["arditi"]
    assert mapping == {"pins": "configs/misalign_baseline_pins.json", "pin": "gemma4e4b.arditi"}
    assert "staged_id" not in mapping and "bank" not in mapping and "object_sha256" not in mapping
    # the two string cells are pinned in the wave's own pins file, as every other view's are
    for cid in ("safety_refusal__psr_kl_all", "si_constitution__psr_kl_all", "constv2-mix-full__lora_r4_ce"):
        assert len(PINS["gemma4e4b"][v["cells"][cid]]["object_sha256"]) == 64
    # the PIN is the misalignment authority's production entry: alpha 0.05, kind static_r,
    # served layer 18, its own bank — the object the paper's instrument serves
    mis = json.loads((REPO / "configs/misalign_baseline_pins.json").read_text())
    pin = mis["gemma4e4b"]["arditi"]
    assert pin["staged_id"] == "ar_r0p05__arditi" and pin["alpha"] == 0.05
    assert pin["kind"] == "static_r" == ra.ARM_OBJECT_KINDS["arditi"]
    assert pin["served_layer"] == 18 and pin["fit_layer_stored"] == 19
    assert pin["bank"] == "arditi_strength" and pin["identity_sidecar"] == "fit_provenance.json"
    assert pin["object_sha256"].startswith("b7cb729d")
    # the generator's bare class-B vocabulary IS the harness's (restated, pinned equal)
    from sisteer.capeval.petri.defense_cells import CLASS_B_IDS, SERVED_ONLY_CLASS_B_IDS
    assert set(gen.BARE_CLASS_B_IDS) == set(CLASS_B_IDS) | set(SERVED_ONLY_CLASS_B_IDS)
    assert gen.harness_cell_id("arditi") == ra.harness_cell_id("arditi") == "arditi"
    with pytest.raises(SystemExit, match="not a cell id this generator knows"):
        gen.harness_cell_id("ar_r0p1")
    # the CONFIG: the four explicit cells in roster order, the view's own dirs, and a header
    # that echoes the pin (never a roster-chosen staged id)
    cfg_p = REPO / "configs/sdfbase_full22_gemma4e4b_psr2cert.yaml"
    cfg, prod = yaml.safe_load(cfg_p.read_text()), _cfg(ROSTER["targets"]["gemma4e4b"]["views"]["main"])
    assert cfg[EXPLICIT_CELLS_KEY] == ["none", "safety_refusal__psr_kl_all", "si_constitution__psr_kl_all", "arditi",
                                       "constv2-mix-full__lora_r4_ce"]
    assert cfg["defenses_dir"] == "out/full22/gemma4e4b/cells_view/psr2cert"
    assert cfg["out_dir"] == "out/sdfbase_full22_gemma4e4b_psr2cert"
    assert cfg["private_run"] == "sdfbase_full22_gemma4e4b_psr2cert"
    strip = {EXPLICIT_CELLS_KEY, "defenses_dir", "out_dir", "private_run"}
    assert {k: v for k, v in cfg.items() if k not in strip} == {k: v for k, v in prod.items() if k not in strip}
    head = cfg_p.read_text()[:cfg_p.read_text().index("\nexplicit_cells:")]
    assert "arditi                           <- ar_r0p05__arditi  (bank arditi_strength, THROUGH the pin" in head
    assert "configs/misalign_baseline_pins.json gemma4e4b.arditi" in head
    assert pin["object_sha256"] in head and "ar_r0p1" not in head
    assert "a CERTIFICATION-ONLY view: rendered from the roster" in head
    # the MANIFEST: the harness ids, the bare class-B id among them, gemma4e4b main's world
    spec = cm.load_petri_spec(REPO / "configs/capeval/manifests/petri_full22_gemma4e4b_psr2cert.yaml")
    pspec = cm.load_petri_spec(REPO / ROSTER["targets"]["gemma4e4b"]["views"]["main"]["manifest"])
    assert list(spec.cells) == ["none", "si_safety__psr_kl_all", "si_constitution__psr_kl_all", "arditi",
                                "si_constv2_mix_full__lora_r4_ce"]
    assert spec.audit == "petri_full22_gemma4e4b_psr2cert"
    for field in ("target_model", "target_revision", "chat_template_hash", "tool_parser",
                  "serving_launch_flags", "behaviors", "scenario_bundle_hash", "auditor_model",
                  "judge_model", "serving_impl_commit"):
        assert getattr(spec, field) == getattr(pspec, field), field
    assert set(spec.prompted_prefix_hashes) == set(gen.PREFIX_HASHES_ALWAYS)   # no prompted cell


def test_psrcert_config_and_manifest_are_the_production_llama8b_pair_with_only_the_cells_renamed():
    """The rendered pair: explicit_cells [none, the PSR cell, the LoRA cell]; every DATA key the production llama8b main
    config's except the four the view names (explicit_cells, defenses_dir, out_dir, private_run); the header echoes both
    pins and says the file is certification-only; the manifest names exactly the harness ids, the same audit world as
    llama8b main, and no prompted cell (so its prefix hashes are the historical three)."""
    cfg_p = REPO / "configs/sdfbase_full22_llama8b_psrcert.yaml"
    cfg, prod = yaml.safe_load(cfg_p.read_text()), _cfg(ROSTER["targets"]["llama8b"]["views"]["main"])
    assert cfg[EXPLICIT_CELLS_KEY] == PSRCERT_ALL_CELLS
    assert cfg["defenses_dir"] == "out/full22/llama8b/cells_view/psrcert"
    assert cfg["out_dir"] == "out/sdfbase_full22_llama8b_psrcert" and cfg["private_run"] == "sdfbase_full22_llama8b_psrcert"
    strip = {EXPLICIT_CELLS_KEY, "defenses_dir", "out_dir", "private_run"}
    assert {k: v for k, v in cfg.items() if k not in strip} == {k: v for k, v in prod.items() if k not in strip}
    assert cfg["xstest_subset"] == "safe" and cfg["bon"]["max_new_tokens"] == 4096 and cfg["attacks"]["controls_max_new_tokens"] == 512
    text = cfg_p.read_text()
    head = text[:text.index("\nexplicit_cells:")]
    for sid in PSRCERT_CELLS.values():
        assert f"<- {sid}" in head and PINS["llama8b"][sid]["object_sha256"] in head, sid
    assert "(target llama8b, view psrcert)" in head and "from\n# configs/sdfbase_full22_roster_psrcert.yaml" in head
    assert "a CERTIFICATION-ONLY view: rendered from the roster" in head and "never a production wave's input" in head
    assert "F22_ROSTER=configs/sdfbase_full22_roster_psrcert.yaml" in head
    # the production files carry no such paragraph (byte-identical: the --check above), and name the production roster
    assert "CERTIFICATION-ONLY" not in (REPO / "configs/sdfbase_full22_llama8b.yaml").read_text()
    assert "from\n# configs/sdfbase_full22_roster.yaml (target llama8b, view main)" in (REPO / "configs/sdfbase_full22_llama8b.yaml").read_text()
    # the manifest
    spec = cm.load_petri_spec(REPO / "configs/capeval/manifests/petri_full22_llama8b_psrcert.yaml")
    pspec = cm.load_petri_spec(REPO / ROSTER["targets"]["llama8b"]["views"]["main"]["manifest"])
    assert list(spec.cells) == ["none", "si_constv2_mix_full__psr_ce_all", "si_constv2_mix_full__lora_r4_ce"]
    assert list(spec.cells) == ["none"] + [gen.harness_cell_id(c) for c in PSRCERT_CELLS] == ["none"] + [ra.harness_cell_id(c) for c in PSRCERT_CELLS]
    assert spec.audit == "petri_full22_llama8b_psrcert"
    for field in ("target_model", "target_revision", "chat_template_hash", "tool_parser", "serving_launch_flags", "behaviors",
                  "scenario_bundle_hash", "auditor_model", "judge_model", "serving_impl_commit", "prompted_prefix_hashes"):
        assert getattr(spec, field) == getattr(pspec, field), field
    assert set(spec.prompted_prefix_hashes) == set(gen.PREFIX_HASHES_ALWAYS)
    mtext = (REPO / "configs/capeval/manifests/petri_full22_llama8b_psrcert.yaml").read_text()
    assert "a CERTIFICATION-ONLY view: rendered from the roster" in mtext and "petri_full22_llama8b_psrcert" in mtext
    assert "CERTIFICATION-ONLY" not in (REPO / ROSTER["targets"]["llama8b"]["views"]["main"]["manifest"]).read_text()


def test_roster_readers_take_the_roster_flag_anywhere_and_refuse_a_view_the_named_roster_lacks():
    """roster_view.py / roster_cell_count.py gain `--roster PATH` (default = today's constant, so every production call is
    byte-identical in effect) through ONE parser, f22_roster.pop_roster_arg; make_bank_view.py already had it. The
    certification roster reads 2 served / 2 staged cells and llama8b main's legs / controls for psrcert; the PRODUCTION
    roster refuses llama8b/psrcert by name (the regression the plan review named — a psrcert job without F22_ROSTER)."""
    rest, path = fr.pop_roster_arg(["llama8b", "psrcert", "--roster", "x.yaml", "legs"])
    assert rest == ["llama8b", "psrcert", "legs"] and path == Path("x.yaml")
    assert fr.pop_roster_arg(["a", "--roster=y.yaml"]) == (["a"], Path("y.yaml"))
    assert fr.pop_roster_arg(["a", "b"]) == (["a", "b"], fr.ROSTER)
    for bad in (["a", "--roster"], ["--roster", "x", "--roster", "y"], ["--roster="]):
        with pytest.raises(ValueError):
            fr.pop_roster_arg(bad)

    def run(script, *args):
        return subprocess.run([sys.executable, str(F22 / script), *args], capture_output=True, text=True, cwd=REPO)
    cert = str(CERT_ROSTER_PATH)
    assert run("roster_cell_count.py", "llama8b", "psrcert", "--roster", cert).stdout.strip() == "2"
    assert run("roster_cell_count.py", "llama8b", "psrcert", "--staged", "--roster", cert).stdout.strip() == "2"
    assert run("roster_cell_count.py", "--roster", cert, "llama8b", "psrcert").stdout.strip() == "2"       # anywhere in the argv
    assert run("roster_view.py", "llama8b", "psrcert", "legs", "--roster", cert).stdout.strip() == ",".join(fr.LEGS)
    assert run("roster_view.py", "llama8b", "psrcert", "controls", "--roster", cert).stdout.strip() == "mmlu,xstest,or_bench_hard_1k"
    assert run("roster_view.py", "llama8b", "psrcert", "prompted_only", "--roster", cert).stdout.strip() == "0"
    assert run("roster_view.py", "llama8b", "psrcert", "window", "controls", "--roster", cert).stdout.strip() == ""     # a table view
    assert run("roster_view.py", "llama8b", "psrcert", "has_leg", "controls", "--roster", cert).returncode == 0
    # the default is the production roster, byte-identical: the same numbers as before for a production view
    assert run("roster_cell_count.py", "llama8b", "main").stdout.strip() == run("roster_cell_count.py", "llama8b", "main", "--roster", str(REPO / "configs/sdfbase_full22_roster.yaml")).stdout.strip() == "5"
    assert run("roster_view.py", "llama8b", "main", "controls").stdout.strip() == "mmlu,xstest,or_bench_hard_1k"
    # KNOWN-BAD: the production roster names no llama8b/psrcert — every reader refuses by name, exit 2 (never a count)
    for args in (("roster_cell_count.py", "llama8b", "psrcert"), ("roster_cell_count.py", "llama8b", "psrcert", "--staged"),
                 ("roster_view.py", "llama8b", "psrcert", "controls"), ("roster_view.py", "llama8b", "psrcert", "legs")):
        r = run(*args)
        assert r.returncode == 2 and "roster names no view llama8b/psrcert" in r.stderr and r.stdout == "", args
    r = run("make_bank_view.py", "--check", "--target", "llama8b", "--view", "psrcert", "--bank", "/nonexistent/bank")
    assert r.returncode != 0 and "names no view llama8b/psrcert" in r.stderr and "pass --roster" in r.stderr
    # a roster file that does not exist refuses too
    for args in (("roster_cell_count.py", "llama8b", "psrcert", "--roster", "/nonexistent/r.yaml"),
                 ("roster_view.py", "llama8b", "psrcert", "legs", "--roster", "/nonexistent/r.yaml"),
                 ("roster_cell_count.py", "llama8b", "psrcert", "--roster")):
        r = run(*args)
        assert r.returncode == 2 and r.stdout == "", args


def test_every_roster_reader_call_of_the_wave_shell_carries_the_jobs_roster():
    """common.sh selects the roster ONCE (F22_ROSTER, default the production file, at file level so every sourcing
    context sees it; f22_init refuses a missing file, exports it, logs it) and EVERY make_bank_view.py /
    roster_cell_count.py / roster_view.py call of common.sh, the sbatch scripts and the top-level .sh scripts passes
    `--roster "$F22_ROSTER"` — enumerated over the files, never a hand-kept list. The bank is passed explicitly beside
    it (one derivation with TT_TWO_SI_DEFENSES_DIR); the view lock stays keyed by (KEY, VIEW) because a non-production
    roster may only name views of its own (the guard below)."""
    sh = (F22 / "common.sh").read_text()
    assert "\nF22_ROSTER_PRODUCTION=configs/sdfbase_full22_roster.yaml\n" in sh
    assert '\nF22_ROSTER="${F22_ROSTER:-$F22_ROSTER_PRODUCTION}"\n' in sh                  # file level: before f22_init
    assert sh.index('F22_ROSTER="${F22_ROSTER:-') < sh.index("f22_init(){")
    init = sh[sh.index("f22_init(){"):sh.index("f22_stop_requested(){")]
    assert '[ -f "$F22_ROSTER" ] || f22_fatal "no roster at $F22_ROSTER' in init and "\n  export F22_ROSTER\n" in init
    # logged beside the engine and the serving-path mode (main's kind-aware log line, with the roster added to it)
    assert "serving_path_mode=$F22_SERVING_PATH_MODE engine=${TARGET_ENGINE:-deferred} reduction=$REDUCTION_MODE roster=$F22_ROSTER wave_root=${WAVE_ROOT:-unbound}" in init
    assert init.index('cd "$REPO_DIR"') < init.index('[ -f "$F22_ROSTER" ]')               # relative to the checkout
    files = [F22 / "common.sh", *sorted(F22.glob("*.sbatch")), *sorted(F22.glob("*.sh"))]
    calls = []
    for f in files:
        text = f.read_text().replace("\\\n", " ")        # a continuation line is one call
        for i, line in enumerate(text.splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            code = re.split(r"\s#", line, 1)[0]              # a trailing comment names the readers too; only CODE is a call
            # a CALL names the script by its path (`scripts/sdfbase/full22/<tool>.py`, the printer's `{S}/<tool>.py`); an
            # operator hint inside a FATAL message names the bare tool and is not one
            if re.search(r"(scripts/sdfbase/full22/|\{S\}/)(make_bank_view|roster_cell_count|roster_view)\.py\b", code):
                calls.append((f.name, i, code.strip()))
    assert len(calls) >= 14, calls
    for name, i, line in calls:
        if name == "validate_derived_serving.sh" and "slice_roster.yaml" in line:
            assert '--roster "$OUT/slice/slice_roster.yaml"' in line                      # the slice's OWN roster, explicit
            continue
        if name == "submit_full22.sh":
            assert "--roster {roster_rel}" in line, (name, i, line)                        # the printer's plan names its roster
            continue
        if name == "common.sh" and '--roster "$roster" --pins "$pins"' in line:
            # f22_arm_bank_view (mt_arms4096; arditi-qwen-all-positions A2'): an ARM's view lives in
            # ITS OWN LANE's bank, never the wave's, and is built from the roster + pins the ARMS
            # TABLE names for that arm. $F22_ROSTER names no such view, and pointing this at the wave
            # roster would build the arm's view from the WAVE's cell set — the very coupling the arm
            # path exists to avoid (registering a lane's cell in the shared wave view would rebuild
            # that directory under every concurrent peer leg). What this guard actually protects
            # still holds: the roster is DECLARED and READ, never hardcoded — it comes from
            # configs/sdfbase_full22_arms.yaml through arm_cell_check, and `$bank` is derived from
            # the resolved arm view rather than from $SHARED/exp/objects/sdfbase. Excused by VALUE
            # (the exact flag pair), never by path, so every other reader in this file is still held.
            assert '--bank "$bank" --target "$KEY" --view "$bview"' in line, (name, i, line)
            continue
        assert '--roster "$F22_ROSTER"' in line or '--roster "$F22_ROSTER_PRODUCTION"' in line, (name, i, line)
    bank_view = sh[sh.index("f22_bank_view(){"):sh.index("f22_bank_view_check(){")]
    assert bank_view.count('--roster "$F22_ROSTER" --bank "$bank"') == 3 and 'local bank="$SHARED/exp/objects/sdfbase/$WAVE/$KEY"' in bank_view
    assert 'roster_cell_count.py "$KEY" "$VIEW" --roster "$F22_ROSTER" >/dev/null 2>&1 \\\n    || f22_fatal "roster $F22_ROSTER names no view $KEY/$VIEW' in bank_view
    assert 'roster_cell_count.py "$KEY" "$VIEW" --roster "$F22_ROSTER_PRODUCTION" >/dev/null 2>&1; then' in bank_view   # the non-production guard
    assert 'lock="$SHARED/exp/objects/sdfbase/$WAVE/.$KEY-$VIEW.view.lock"' in bank_view                                 # unchanged: one dir, one lock
    check = sh[sh.index("f22_bank_view_check(){"):sh.index("# --- venvs")]
    assert '--roster "$F22_ROSTER" --bank "$bank"' in check and "--roster $F22_ROSTER from a shell" in check
    # the grading host driver and the submit printer read the same selection
    drive = (F22 / "petri_drive_full22.sh").read_text()
    assert 'F22_ROSTER="${F22_ROSTER:-configs/sdfbase_full22_roster.yaml}"' in drive and '--view "$VIEW" --roster "$F22_ROSTER"' in drive
    sub_sh = (F22 / "submit_full22.sh").read_text()
    assert 'ROSTER="${F22_ROSTER:-$REPO/configs/sdfbase_full22_roster.yaml}"' in sub_sh


def _psrcert_tree(tmp_path: Path, *, pins_for: dict) -> Path:
    """A synthetic checkout under tmp: the REAL full22 readers (copied byte for byte), both rosters, the generated
    psrcert pair, a SYNTHETIC pins file over the synthetic bank's shas, and `src` for the readers' imports."""
    import shutil
    root = tmp_path / "checkout"
    (root / "scripts/sdfbase/full22").mkdir(parents=True)
    for name in ("make_bank_view.py", "f22_roster.py", "roster_cell_count.py", "roster_view.py"):
        shutil.copy(F22 / name, root / "scripts/sdfbase/full22" / name)
    (root / "configs/capeval/manifests").mkdir(parents=True)
    for rel in ("configs/sdfbase_full22_roster.yaml", "configs/sdfbase_full22_roster_psrcert.yaml",
                "configs/sdfbase_full22_llama8b_psrcert.yaml", "configs/capeval/manifests/petri_full22_llama8b_psrcert.yaml"):
        shutil.copy(REPO / rel, root / rel)
    (root / "configs/sdfbase_full22_pins.json").write_text(json.dumps({"llama8b": pins_for}, indent=2))
    (root / "src").symlink_to(REPO / "src", target_is_directory=True)
    return root


def _psrcert_preflight(tmp_path: Path, *, roster_env, py: str, extra: str = "") -> subprocess.CompletedProcess:
    """The REAL shell preflight of a psrcert job, under bash: f22_resolve_config -> f22_bank_view (the roster count guard,
    make_bank_view --check / build / --check, the config's ALL_CELLS, the presence + count guards) -> the prompted-only
    read — the functions extracted from common.sh, over a synthetic bank + a synthetic checkout, with the roster
    selected the way a job selects it (F22_ROSTER in the environment; unset = the production roster)."""
    root = tmp_path / "checkout"
    shared = tmp_path / "shared"
    funcs = _bash_funcs("f22_resolve_config", "f22_bank_view", "f22_view_prompted_only")
    shim = tmp_path / "bin"; shim.mkdir(exist_ok=True)
    if not (shim / "python3").exists():
        (shim / "python3").symlink_to(py)                       # the functions' bare `python3` reads
    script = ("set -u\nf22_log(){ echo \"LOG $*\"; }\nf22_fatal(){ echo \"FATAL $*\"; exit 1; }\n" + funcs + "\n"
              f"cd {root}; SHARED={shared}; WAVE=full22; REPO_DIR={root}\n"
              "F22_ROSTER_PRODUCTION=configs/sdfbase_full22_roster.yaml\n"
              'F22_ROSTER="${F22_ROSTER:-$F22_ROSTER_PRODUCTION}"\n'                      # common.sh's file-level selection
              'f22_resolve_config llama8b psrcert; echo "RESOLVED CFG=$CFG MANIFEST=$MANIFEST TAG=$TAG"\n'
              f"f22_bank_view {py}\n"
              'echo "ALL_CELLS=${ALL_CELLS[*]}"; echo "VIEW=$TT_TWO_SI_DEFENSES_DIR"\n'
              'if f22_view_prompted_only; then echo PROMPTED_ONLY=1; else echo PROMPTED_ONLY=0; fi\n' + extra)
    env = {k: v for k, v in os.environ.items() if k != "F22_ROSTER"}
    if roster_env is not None:
        env["F22_ROSTER"] = roster_env
    env["PATH"] = f"{shim}:{env.get('PATH', '')}"
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env)


def test_the_real_shell_preflight_stands_a_psrcert_job_up_under_the_certification_roster_and_refuses_without_it(tmp_path):
    """Deliverable 5 (b) + (c). (b) `F22_ROSTER=configs/sdfbase_full22_roster_psrcert.yaml`: f22_resolve_config finds the
    generated config and manifest; f22_bank_view passes the roster count guard, builds the view over the ALREADY-STAGED
    synthetic cells (make_bank_view --check / build under the lock / --check, all `--roster "$F22_ROSTER"`), reads
    ALL_CELLS = [none, the PSR cell, the LoRA cell] from the config, holds the config (2 + none) and the view (2 entries)
    to the roster's 2 served / 2 staged, and the prompted-only read answers 0. (c) KNOWN-BAD: the SAME preflight without
    F22_ROSTER (the production roster) refuses at the roster count guard, by name — roster_cell_count.py exits 2 for
    llama8b/psrcert — before any view is read or built."""
    bank = tmp_path / "shared/exp/objects/sdfbase/full22/llama8b"
    pins = {}
    for cid, sid in PSRCERT_CELLS.items():
        p = _stage(bank / "cells", sid, cid)
        pins[sid] = {"object_sha256": sha256_file(p)}
    _psrcert_tree(tmp_path, pins_for=pins)
    cert = "configs/sdfbase_full22_roster_psrcert.yaml"
    r = _psrcert_preflight(tmp_path, roster_env=cert, py=sys.executable)
    assert r.returncode == 0, (r.stdout, r.stderr)
    out = r.stdout
    assert ("RESOLVED CFG=configs/sdfbase_full22_llama8b_psrcert.yaml "
            "MANIFEST=configs/capeval/manifests/petri_full22_llama8b_psrcert.yaml TAG=llama8b_psrcert") in out
    assert f"ALL_CELLS={' '.join(PSRCERT_ALL_CELLS)}\n" in out
    assert f"VIEW={bank}/cells_view/psrcert\n" in out and "PROMPTED_ONLY=0" in out
    assert "does not verify — building it under" in out                      # first --check fails (no view yet), built under the lock
    assert f"roster {cert}: 2 served / 2 staged, view 2 — complete" in out and "FATAL" not in out
    view = bank / "cells_view/psrcert"
    assert sorted(p.name for p in view.iterdir() if not p.name.startswith("VIEW_MANIFEST")) == sorted(PSRCERT_CELLS)
    for cid, sid in PSRCERT_CELLS.items():
        assert (view / cid).is_symlink() and (view / cid / "object.pt").is_file() and sha256_file(view / cid / "object.pt") == pins[sid]["object_sha256"]
    man = json.loads((view / "VIEW_MANIFEST.json").read_text())
    assert man["view"] == "psrcert" and {e["cell_id"]: e["staged_id"] for e in man["entries"]} == PSRCERT_CELLS
    assert (bank.parent / ".llama8b-psrcert.view.lock").exists()             # the lock keyed by (KEY, VIEW)
    # a second run: the view verifies read-only, nothing rebuilt
    r2 = _psrcert_preflight(tmp_path, roster_env=cert, py=sys.executable)
    assert r2.returncode == 0 and "building it under" not in r2.stdout and "view 2 — complete" in r2.stdout, r2.stdout
    # the same selection as an ABSOLUTE path resolves too (F22_ROSTER may be either)
    r3 = _psrcert_preflight(tmp_path, roster_env=str(tmp_path / "checkout" / cert), py=sys.executable)
    assert r3.returncode == 0 and "view 2 — complete" in r3.stdout, r3.stdout
    # (c) KNOWN-BAD: the production roster (F22_ROSTER unset) refuses BY NAME at the count guard, before any read or build
    import shutil
    shutil.rmtree(view)
    r = _psrcert_preflight(tmp_path, roster_env=None, py=sys.executable)
    assert r.returncode == 1, (r.stdout, r.stderr)
    assert "RESOLVED CFG=configs/sdfbase_full22_llama8b_psrcert.yaml" in r.stdout      # the config + manifest exist regardless
    assert "FATAL roster configs/sdfbase_full22_roster.yaml names no view llama8b/psrcert" in r.stdout
    assert "F22_ROSTER=<its roster>" in r.stdout and "ALL_CELLS=" not in r.stdout and "building it under" not in r.stdout
    assert not view.exists()                                                  # nothing was built under the wrong roster
    rc = subprocess.run([sys.executable, str(tmp_path / "checkout/scripts/sdfbase/full22/roster_cell_count.py"), "llama8b", "psrcert"],
                        capture_output=True, text=True, cwd=tmp_path / "checkout")
    assert rc.returncode == 2 and "roster names no view llama8b/psrcert" in rc.stderr
    # ... and an explicit F22_ROSTER naming the production file refuses the same way
    r = _psrcert_preflight(tmp_path, roster_env="configs/sdfbase_full22_roster.yaml", py=sys.executable)
    assert r.returncode == 1 and "names no view llama8b/psrcert" in r.stdout
    # a roster that does not exist: f22_bank_view's first read refuses by name too (f22_init refuses earlier on a job)
    r = _psrcert_preflight(tmp_path, roster_env="configs/nope.yaml", py=sys.executable)
    assert r.returncode == 1 and "FATAL roster configs/nope.yaml names no view llama8b/psrcert" in r.stdout


def test_a_non_production_roster_may_only_name_views_the_production_roster_does_not(tmp_path):
    """The view dir and its lock are keyed by (KEY, VIEW) alone, so a second roster naming a PRODUCTION view would rebuild
    the same shared directory under another cell set — f22_bank_view refuses that by name before anything is read or
    built (which is also why the lock does not carry the roster's name: one directory, one lock)."""
    bank = tmp_path / "shared/exp/objects/sdfbase/full22/llama8b"
    pins = {}
    for cid, sid in PSRCERT_CELLS.items():
        pins[sid] = {"object_sha256": sha256_file(_stage(bank / "cells", sid, cid))}
    root = _psrcert_tree(tmp_path, pins_for=pins)
    # a roster of its own that ALSO names llama8b/main (the production view) with the psrcert cells
    bad = copy.deepcopy(CERT_ROSTER)
    bad["targets"]["llama8b"]["views"]["main"] = {"config": "configs/sdfbase_full22_llama8b.yaml",
                                                  "manifest": "configs/capeval/manifests/petri_full22_llama8b.yaml",
                                                  "cells": dict(PSRCERT_CELLS)}
    (root / "configs/sdfbase_full22_roster_bad.yaml").write_text(yaml.safe_dump(bad, sort_keys=False))
    funcs = _bash_funcs("f22_bank_view")
    script = ("set -u\nf22_log(){ echo \"LOG $*\"; }\nf22_fatal(){ echo \"FATAL $*\"; exit 1; }\n" + funcs + "\n"
              f"cd {root}; SHARED={tmp_path}/shared; WAVE=full22; REPO_DIR={root}; KEY=llama8b; VIEW=main\n"
              "F22_ROSTER_PRODUCTION=configs/sdfbase_full22_roster.yaml\n"
              "F22_ROSTER=configs/sdfbase_full22_roster_bad.yaml\n" + f"f22_bank_view {sys.executable}\n")
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert r.returncode == 1, (r.stdout, r.stderr)
    assert "FATAL F22_ROSTER=configs/sdfbase_full22_roster_bad.yaml names llama8b/main, a view the production roster configs/sdfbase_full22_roster.yaml also names" in r.stdout
    assert not (bank / "cells_view").exists() and "building it under" not in r.stdout   # refused before any read or build


def test_submit_printer_plans_from_the_selected_roster_and_prints_it_on_every_builder_line(tmp_path, monkeypatch):
    """submit_full22.sh: `--roster <yaml>` (else F22_ROSTER, else the production roster) is the plan's roster, and the printed
    make_bank_view.py lines carry `--roster <repo-relative path>` so the cluster-side build step reads the SAME file the
    jobs will; a non-production roster's plan says to export F22_ROSTER on every sbatch line."""
    monkeypatch.setenv("SISTEER_CONFIG", str(write_site(tmp_path / "cluster-site")))   # the printer reads each cluster's host from the site
    sha = "ab" * 20
    base_env = {k: v for k, v in os.environ.items() if k != "F22_ROSTER"}
    prod = subprocess.run(["bash", str(F22 / "submit_full22.sh"), sha], capture_output=True, text=True, env=base_env)
    assert prod.returncode == 0, prod.stderr
    assert prod.stdout.count("--roster configs/sdfbase_full22_roster.yaml") == 4 and "F22_ROSTER=" not in prod.stdout   # 2 clusters x (build + check)
    assert "llama8b/psrcert" not in prod.stdout and "gemma4e4b/psr2cert" not in prod.stdout
    for how in (["--roster", "configs/sdfbase_full22_roster_psrcert.yaml"], ["--roster", str(CERT_ROSTER_PATH)], None):
        env = dict(base_env) if how else {**base_env, "F22_ROSTER": "configs/sdfbase_full22_roster_psrcert.yaml"}
        r = subprocess.run(["bash", str(F22 / "submit_full22.sh"), sha, *(how or [])], capture_output=True, text=True, env=env)
        assert r.returncode == 0, (how, r.stderr)
        # 2 clusters x (build + check), one ssh line per cluster looping over EVERY view — the count is per cluster,
        # not per view (measured: 4 with two, three and four views alike)
        assert r.stdout.count("--roster configs/sdfbase_full22_roster_psrcert.yaml") == 4, how
        # the plan's summary line: the cert views only (a budget WARNING may precede it — the four-cell psr2cert view's
        # static jobs exceed the default 13-GPU cluster A budget line, which the printer says first)
        first = next(l for l in r.stdout.splitlines() if l.startswith("# full22 @"))
        assert first.startswith(
            f"# full22 @ {sha}: 14 cells over 4 configs (llama8b/psrcert=3, llama8b/pairedcert=3, gemma4e4b/psr2cert=5, "
            "qwen35_4b/pairedcert=3)"), first   # psr2cert: none + 4 (T5b); each pairedcert: none + 2 (the paired write)
        assert "export F22_ROSTER=configs/sdfbase_full22_roster_psrcert.yaml on every sbatch line below" in r.stdout, how
        assert "for kv in llama8b:psrcert llama8b:pairedcert gemma4e4b:psr2cert qwen35_4b:pairedcert; do" in r.stdout, how
    r = subprocess.run(["bash", str(F22 / "submit_full22.sh"), sha, "--roster", "/nonexistent/r.yaml"], capture_output=True, text=True, env=base_env)
    assert r.returncode == 2 and "no roster at /nonexistent/r.yaml" in r.stderr


def _smoke_func(name: str) -> str:
    src = (F22 / "smoke_full22.sbatch").read_text()
    start = src.index(f"{name}(){{")
    return src[start:src.index("\n}\n", start) + 3]


def test_smoke_checks_output_diagnostics_per_native_standup_from_its_own_record(tmp_path):
    """Plan step 6: after each successful standup (static, controls, attacks) the smoke — under the native engine only —
    reads the standup's cells_served.json (the path endpoint.json names, else the standup dir's) and PASSes iff
    output_diagnostics is false when any served adapter's petri_source_kind is a residual-slot kind (psr, vector,
    vector_all, diffmean, static_r — defense_cells.RESIDUAL_SLOT_KINDS, read by import) and true otherwise, recording the
    served kinds; a missing record / key / kind is a FAIL by name. RUN here over synthetic records; the call sites and the
    header pinned by text; the script still parses."""
    from sisteer.capeval.petri import defense_cells as dc
    from sisteer.capeval.serving import native_residual as nr
    assert dc.RESIDUAL_SLOT_KINDS == ("psr", "vector", "vector_all", "diffmean", "static_r",
                                      "psr_paired_write") == nr.RESIDUAL_SLOT_KINDS
    smoke = (F22 / "smoke_full22.sbatch").read_text()
    assert subprocess.run(["bash", "-n", str(F22 / "smoke_full22.sbatch")], capture_output=True).returncode == 0
    for phase, after in (("static", 'check standup_static PASS "geometry=${TARGET_S}x${TARGET_L} attested from endpoint.json + /v1/models: $TARGET_ENGINE; table=$TABLE"\n'),
                         ("controls", 'check standup_controls PASS "geometry=${TARGET_S}x${TARGET_L} attested; $TARGET_ENGINE; max_logprobs=$CTL_TOP_LOGPROBS"\n'),
                         ("attacks", 'check standup_attacks PASS "geometry=${TARGET_S}x${TARGET_L} attested; $TARGET_ENGINE"\n')):
        call = f'    [ "$TARGET_ENGINE" = lora_native ] && output_diagnostics_check {phase}\n'
        assert smoke.count(call) == 1 and after + call in smoke, phase           # right after the standup's PASS, native only
    assert smoke.count("output_diagnostics_check ") == 3 and smoke.count("output_diagnostics_check(){") == 1
    fn = _smoke_func("output_diagnostics_check")
    code = "\n".join(l for l in fn.splitlines() if not l.strip().startswith("#"))
    assert "from sisteer.capeval.petri.defense_cells import RESIDUAL_SLOT_KINDS" in code and '"psr"' not in code   # by import, never a copied list
    assert 'ep.get("cells_served") or fallback' in code and 'ep="$out/endpoint.json"; [ -f "$ep" ] || ep="$out/endpoint.recorded.json"' in code
    assert 'check "output_diagnostics_$1" PASS "$d"; else check "output_diagnostics_$1" FAIL "$d"' in code
    # the header names the check and the roster selection for llama8b's psrcert view
    head = smoke[:smoke.index("set -uo pipefail")]
    assert "output_diagnostics_<phase>" in head and "F22_ROSTER=configs/sdfbase_full22_roster_psrcert.yaml sbatch" in head
    assert "llama8b psrcert <wave_sha> constv2-mix-full__psr_ce_all static,controls" in head and "D15" in head
    # RUN: the function over synthetic records (the zero adapter is recorded apart; the served kinds are the adapters')
    lora = {"name": "petri__si_constv2_mix_full__lora_r4_ce", "petri_source_kind": "lora_all", "petri_zero_adapter": False}
    psr = {"name": "petri__si_constv2_mix_full__psr_ce_all", "petri_source_kind": "psr", "petri_zero_adapter": False}

    def run(phase, record, *, via_endpoint=True):
        out = tmp_path / f"serve-{len(list(tmp_path.glob('serve-*')))}" / phase
        out.mkdir(parents=True)
        if record is not None:
            (out / "cells_served.json").write_text(json.dumps(record))
        (out / "endpoint.json").write_text(json.dumps({"cells_served": str(out / "cells_served.json")} if via_endpoint else {}))
        script = ("set -uo pipefail\ncheck(){ echo \"CHECK $2 $1: $3\"; }\n" + fn + "\n"
                  f"SERVE_DIR={out.parent}; SPY={sys.executable}; REPO_DIR={REPO}\noutput_diagnostics_check {phase}\n")
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        assert r.returncode == 0, (r.stdout, r.stderr)
        return r.stdout.strip()
    mixed = {"output_diagnostics": False, "adapters": {"constv2-mix-full__lora_r4_ce": lora, "constv2-mix-full__psr_ce_all": psr},
             "zero_adapter": {"name": "petri__none__lora0", "petri_source_kind": None, "petri_zero_adapter": True}}
    o = run("static", mixed)
    assert o.startswith("CHECK PASS output_diagnostics_static:") and "output_diagnostics=false want=false" in o
    assert '"constv2-mix-full__psr_ce_all": "psr"' in o and '"constv2-mix-full__lora_r4_ce": "lora_all"' in o and "residual_cells=['constv2-mix-full__psr_ce_all']" in o
    o = run("controls", {**mixed, "output_diagnostics": True})
    assert o.startswith("CHECK FAIL output_diagnostics_controls:") and "output_diagnostics=true but the standup serves a residual-slot kind" in o
    lora_only = {"output_diagnostics": True, "adapters": {"constv2-mix-full__lora_r4_ce": lora}}
    o = run("attacks", lora_only)
    assert o.startswith("CHECK PASS output_diagnostics_attacks:") and "output_diagnostics=true want=true" in o and "residual_cells=[]" in o
    o = run("static", {**lora_only, "output_diagnostics": False})
    assert o.startswith("CHECK FAIL output_diagnostics_static:") and "serves LoRA kinds only" in o
    for kind in ("vector", "vector_all", "diffmean", "static_r"):                      # every residual kind wants false
        o = run("static", {"output_diagnostics": False, "adapters": {"c": {**psr, "petri_source_kind": kind}}})
        assert o.startswith("CHECK PASS") and f'"c": "{kind}"' in o, kind
    o = run("static", {"adapters": {"constv2-mix-full__lora_r4_ce": lora}})            # no key: FAIL by name, never a default
    assert o.startswith("CHECK FAIL output_diagnostics_static:") and "output_diagnostics is None, not a boolean" in o
    o = run("static", {"output_diagnostics": "false", "adapters": {"constv2-mix-full__lora_r4_ce": lora}})
    assert o.startswith("CHECK FAIL") and "not a boolean" in o
    o = run("static", {"output_diagnostics": True, "adapters": {"x": {"name": "x"}}})  # an adapter with no kind
    assert o.startswith("CHECK FAIL") and "carries no petri_source_kind" in o
    o = run("static", {"output_diagnostics": True})                                    # no adapters block
    assert o.startswith("CHECK FAIL") and "no `adapters` block" in o
    o = run("static", None)                                                            # no record at all
    assert o.startswith("CHECK FAIL output_diagnostics_static: no cells_served.json for this standup")
    o = run("static", lora_only, via_endpoint=False)                                   # endpoint.json names no path: the standup dir's file
    assert o.startswith("CHECK PASS") and "output_diagnostics=true want=true" in o


# --- the KIND-AWARE serving-path default (the rule of 2026-09-15) ----------------------------------------------
# "use native when possible, select the lens route by itself when a launch's steered cells are all lens-only kinds; make
# the selection recoverable in the logs" — the shell side: f22_select_serving_path (the ONE resolution, run by f22_init)
# reaches registry_adapter.select_serving_path through scripts/sdfbase/full22/serving_path_select.py and evals the verdict.

SELECT_FUNCS = ("f22_full22_manifest", "f22_bind_engine", "f22_select_auto", "f22_select_serving_path", "f22_bind_serving_path")


def _select_world(tmp_path: Path, *, manifest_cells=None, tag="gemma4e4b") -> Path:
    """A REPO_DIR for f22_select_serving_path: the real selector + src, and (when given) a synthetic (KEY, VIEW) manifest —
    a real one's data with its cells replaced — at f22_resolve_config's spelling."""
    repo = tmp_path / "repo"
    (repo / "scripts/sdfbase/full22").mkdir(parents=True)
    (repo / "scripts/sdfbase/full22/serving_path_select.py").symlink_to(F22 / "serving_path_select.py")
    (repo / "src").symlink_to(REPO / "src")
    if manifest_cells is not None:
        man = yaml.safe_load((REPO / f"configs/capeval/manifests/petri_full22_{tag}.yaml").read_text())
        man["petri"]["cells"] = list(manifest_cells)
        (repo / "configs/capeval/manifests").mkdir(parents=True)
        (repo / f"configs/capeval/manifests/petri_full22_{tag}.yaml").write_text(yaml.safe_dump(man, sort_keys=False))
    return repo


def _run_select(repo: Path, body: str, *, env_drop=("F22_SERVING_PATH", "F22_LAUNCH_CELLS", "KEY", "VIEW"), shared: Path | None = None, **env):
    """Run `body` after the sliced functions under `set -u`, with REPO_DIR = `repo` and the selector on the test's
    interpreter (F22_SELECT_PY: the node's stdlib python3 + PyYAML in production; here the venv's, which has both).
    SHARED (the root f22_bind_serving_path derives a campaign WAVE_ROOT under) must be passed by a test that binds; the
    default is a path no mkdir can create, so a bind the test did not expect fails loudly instead of writing somewhere."""
    script = ("set -u\nf22_log(){ echo \"LOG $*\"; }\nf22_fatal(){ echo \"FATAL $*\"; exit 1; }\n" + _bash_funcs(*SELECT_FUNCS)
              + f"\nREPO_DIR={repo}\nF22_SELECT_PY={sys.executable}\nREDUCTION_MODE=stacked\nSHARED={shared or '/dev/null/shared-not-given'}\n" + body)
    base = {k: v for k, v in os.environ.items() if k not in env_drop + ("F22_LAUNCH_MANIFEST", "F22_SERVING_PATH_DEFER", "F22_WAVE_ROOT")}
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, env={**base, **env})


def _selection_json(out: str) -> dict:
    line = [l for l in out.splitlines() if l.startswith("SELECTION=")][0]
    return json.loads(line[len("SELECTION="):])


REPORT = ('echo "PATH=$F22_SERVING_PATH"; echo "MODE=$F22_SERVING_PATH_MODE"; echo "ENGINE=$( [ "$F22_SERVING_PATH" = lens ] && echo lens:stacked || echo lora_native )"; '
          'echo "SELECTION=$F22_SERVING_PATH_SELECTION_JSON"')


def test_select_serving_path_picks_the_lens_for_an_all_lens_only_cell_list_and_native_for_a_lora_list(tmp_path):
    """A synthetic launch through the selection with NO F22_SERVING_PATH: an all-AlphaSteer list selects the LENS; a
    LoRA-bearing list selects NATIVE with the lens-only cells named as skipped — each with the ONE recoverable log line
    and the record the standup carries. Since native-psr-serving the residual-slot kinds — psr, vector,
    vector_all, diffmean, static_r (Arditi) — are natively servable, so a launch whose steered cells are AlphaSteer +
    Arditi (the MATH-500 steer wave's shape, 2026-09-14) now selects NATIVE and skips AlphaSteer by name; only an
    all-AlphaSteer list is lens-only."""
    repo = _select_world(tmp_path)
    r = _run_select(repo, 'F22_LAUNCH_CELLS=(none alphasteer alphasteer@thinking)\nf22_select_serving_path\n' + REPORT)
    assert r.returncode == 0, (r.stdout, r.stderr)
    assert "PATH=lens\nMODE=auto\nENGINE=lens:stacked" in r.stdout
    assert ("LOG [serving-path] AUTO-selected lens: every steered cell is lens-only (alphasteer×2); "
            "explicit F22_SERVING_PATH would override") in r.stdout
    sel = _selection_json(r.stdout)
    assert (sel["mode"], sel["selected"], sel["rule"], sel["spec_source"]) == ("auto", "lens", "kind-aware-2026-09-15", "kind_only")
    assert sel["cells_by_disposition"] == {"passthrough": ["none"], "serve": [], "skip": ["alphasteer", "alphasteer@thinking"]}
    assert [(k["cell_id"], k["kind"]) for k in sel["skipped"]] == [("alphasteer", "alphasteer"), ("alphasteer@thinking", "alphasteer")]
    assert sel["cells_source"] == "F22_LAUNCH_CELLS" and sel["cells_requested"] == ["none", "alphasteer", "alphasteer@thinking"]
    # AlphaSteer + Arditi (the MATH-500 steer wave's shape): NATIVE since native-psr-serving — Arditi (static_r) serves on
    # the residual slot, AlphaSteer is skipped by name
    r = _run_select(repo, 'F22_LAUNCH_CELLS=(none alphasteer arditi alphasteer@thinking)\nf22_select_serving_path\n' + REPORT)
    assert r.returncode == 0, (r.stdout, r.stderr)
    assert "PATH=lora_native\nMODE=auto\nENGINE=lora_native" in r.stdout
    assert ("LOG [serving-path] AUTO-selected lora_native: 1 of 3 steered cells natively servable; skipped by name: "
            "alphasteer (alphasteer), alphasteer@thinking (alphasteer); explicit F22_SERVING_PATH would override") in r.stdout
    sel = _selection_json(r.stdout)
    assert sel["cells_by_disposition"] == {"passthrough": ["none"], "serve": ["arditi"], "skip": ["alphasteer", "alphasteer@thinking"]}
    assert [(k["cell_id"], k["kind"]) for k in sel["skipped"]] == [("alphasteer", "alphasteer"), ("alphasteer@thinking", "alphasteer")]
    # a LoRA list: native; the psr cell SERVES (a residual-slot kind), the AlphaSteer cell is skipped by name
    r = _run_select(repo, 'F22_LAUNCH_CELLS=(none si_constv2_mix_full__lora_r4_ce si_safety__psr_kl_all alphasteer prompted__constitution)\n'
                          'f22_select_serving_path\n' + REPORT)
    assert r.returncode == 0, (r.stdout, r.stderr)
    assert "PATH=lora_native\nMODE=auto\nENGINE=lora_native" in r.stdout
    assert ("LOG [serving-path] AUTO-selected lora_native: 2 of 3 steered cells natively servable; skipped by name: "
            "alphasteer (alphasteer); explicit F22_SERVING_PATH would override") in r.stdout
    sel = _selection_json(r.stdout)
    assert sel["cells_by_disposition"] == {"passthrough": ["none", "prompted__constitution"],
                                           "serve": ["si_constv2_mix_full__lora_r4_ce", "si_safety__psr_kl_all"],
                                           "skip": ["alphasteer"]}
    # THE NEW FACT, pinned explicitly: a psr + Arditi + LoRA list serves all three, skips none
    r = _run_select(repo, 'F22_LAUNCH_CELLS=(none si_safety__psr_kl_all arditi si_constv2_mix_full__lora_r4_ce)\nf22_select_serving_path\n' + REPORT)
    assert r.returncode == 0, (r.stdout, r.stderr)
    assert "PATH=lora_native\nMODE=auto\nENGINE=lora_native" in r.stdout
    assert "LOG [serving-path] AUTO-selected lora_native: 3 of 3 steered cells natively servable; skipped by name: none" in r.stdout
    sel = _selection_json(r.stdout)
    assert sel["cells_by_disposition"] == {"passthrough": ["none"],
                                           "serve": ["si_safety__psr_kl_all", "arditi", "si_constv2_mix_full__lora_r4_ce"], "skip": []}
    assert sel["skipped"] == []
    # passthrough-only and NO cells at all: the entry-point default, with the reason recorded
    r = _run_select(repo, 'F22_LAUNCH_CELLS=(none prompted__constitution)\nf22_select_serving_path\n' + REPORT)
    assert r.returncode == 0 and "PATH=lora_native\nMODE=auto" in r.stdout and "passthrough-only launch" in r.stdout, r.stdout
    r = _run_select(repo, 'F22_LAUNCH_CELLS=()\nf22_select_serving_path\n' + REPORT)      # DECLARED empty: a job that stands up no target
    assert r.returncode == 0 and "PATH=lora_native\nMODE=auto" in r.stdout, (r.stdout, r.stderr)
    assert "AUTO-selected lora_native: no cell list (source: F22_LAUNCH_CELLS (declared empty: this job stands up no target))" in r.stdout
    assert _selection_json(r.stdout)["cells_by_disposition"] == {"passthrough": [], "serve": [], "skip": []}


def test_select_serving_path_takes_a_declared_manifest_and_key_view_alone_refuses_naming_the_caller(tmp_path):
    """The cell list is DECLARED (an earlier draft assumed the (KEY, VIEW) campaign manifest and misread
    every caller with its own — the AlphaSteer campaign, the strength screen, the misalign proxy). KEY / VIEW alone REFUSE,
    naming the caller and every way to declare; F22_LAUNCH_MANIFEST selects from the launch's manifest (a synthetic
    all-lens-only view at the campaign spelling -> lens; the real gemma4e4b/main view -> native); a missing manifest is fatal
    by name; f22_full22_manifest spells f22_resolve_config's manifest; F22_LAUNCH_CELLS wins over a manifest."""
    repo = _select_world(tmp_path, manifest_cells=["none", "alphasteer"])           # AlphaSteer-only: the lens-only shape
    r = _run_select(repo, 'KEY=gemma4e4b; VIEW=main\nf22_select_serving_path\n' + REPORT)
    assert r.returncode == 1 and "FATAL serving-path selection: bash declared no cell source" in r.stdout, r.stdout
    for needle in ("F22_LAUNCH_MANIFEST=<the launch's manifest>", "$(f22_full22_manifest KEY VIEW)", "F22_LAUNCH_CELLS=(<harness ids>; empty = this job stands up no target)",
                   "F22_SERVING_PATH_DEFER=1", "F22_WAVE_ROOT", "f22_bind_serving_path", "NOT assumed from KEY / VIEW"):
        assert needle in r.stdout, needle
    assert "SELECTION=" not in r.stdout
    # the declared campaign manifest, at f22_resolve_config's spelling
    r = _run_select(repo, 'F22_LAUNCH_MANIFEST=$(f22_full22_manifest gemma4e4b main)\nf22_select_serving_path\n' + REPORT)
    assert r.returncode == 0, (r.stdout, r.stderr)
    assert "PATH=lens\nMODE=auto\nENGINE=lens:stacked" in r.stdout and "AUTO-selected lens: every steered cell is lens-only (alphasteer\u00d71)" in r.stdout
    assert _selection_json(r.stdout)["cells_source"] == "manifest:configs/capeval/manifests/petri_full22_gemma4e4b.yaml"
    r = _run_select(REPO, 'F22_LAUNCH_MANIFEST=$(f22_full22_manifest gemma4e4b main)\nf22_select_serving_path\n' + REPORT)   # the REAL view: mixed -> native
    assert r.returncode == 0 and "PATH=lora_native\nMODE=auto" in r.stdout, (r.stdout, r.stderr)
    # native-psr-serving: the view's four psr cells are natively servable kinds now — nothing skipped by kind
    assert "AUTO-selected lora_native: 8 of 8 steered cells natively servable; skipped by name: none" in r.stdout
    r = _run_select(REPO, f'F22_LAUNCH_MANIFEST={REPO}/configs/capeval/manifests/petri_full22_gemma4e4b.yaml\nf22_select_serving_path\n' + REPORT)   # absolute
    assert r.returncode == 0 and "PATH=lora_native" in r.stdout
    # f22_full22_manifest == f22_resolve_config's MANIFEST for a main and a non-main view (the ONE spelling, pinned equal)
    for key, view in (("gemma4e4b", "main"), ("gemma4e4b", "it"), ("llama8b", "it")):
        r = subprocess.run(["bash", "-c", "set -u\nf22_log(){ :; }\nf22_fatal(){ echo FATAL \"$@\"; exit 1; }\n" + _bash_funcs("f22_full22_manifest", "f22_resolve_config")
                            + f"\ncd {REPO}\nf22_resolve_config {key} {view}\n[ \"$MANIFEST\" = \"$(f22_full22_manifest {key} {view})\" ] && echo SAME $MANIFEST || echo DIFF $MANIFEST"],
                           capture_output=True, text=True)
        assert r.returncode == 0 and r.stdout.startswith("SAME configs/capeval/manifests/petri_full22_"), (key, view, r.stdout, r.stderr)
    # a manifest that does not exist is fatal by name (never a silent default)
    r = _run_select(repo, 'F22_LAUNCH_MANIFEST=configs/capeval/manifests/petri_full22_gemma4e4b_nosuchview.yaml\nf22_select_serving_path\n' + REPORT)
    assert r.returncode == 1 and "FATAL F22_LAUNCH_MANIFEST=configs/capeval/manifests/petri_full22_gemma4e4b_nosuchview.yaml does not exist (caller bash)" in r.stdout, r.stdout
    assert "set F22_SERVING_PATH explicitly to bypass" in r.stdout
    # F22_LAUNCH_CELLS wins over a declared manifest when a launcher names its own cells
    r = _run_select(REPO, 'F22_LAUNCH_MANIFEST=$(f22_full22_manifest gemma4e4b main); F22_LAUNCH_CELLS=(alphasteer)\nf22_select_serving_path\n' + REPORT)
    assert r.returncode == 0 and "PATH=lens" in r.stdout and _selection_json(r.stdout)["cells_source"] == "F22_LAUNCH_CELLS"


def _as22_manifest(key: str) -> str:
    """The AlphaSteer campaign's manifest for `key`, from its ONE table (as22_targets.manifest_rel) — the real pairing."""
    r = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, sys.argv[1]); import as22_targets as t; print(t.manifest_rel(sys.argv[2]))",
                        str(REPO / "scripts/alphasteer_f22"), key], capture_output=True, text=True, check=True)
    return r.stdout.strip()


def test_a_deferred_selection_binds_from_the_launchs_own_manifest_never_the_campaigns(tmp_path):
    """A caller that owns its wave root (F22_WAVE_ROOT) or sets F22_SERVING_PATH_DEFER=1 gets NO path at f22_init — no engine,
    the own root kept or WAVE_ROOT unbound — and f22_bind_serving_path binds path + engine (+ the path's campaign root when
    unbound) from the manifest it is handed. The REAL AlphaSteer campaign pairing (as22_targets.manifest_rel: none +
    alphasteer) binds to the LENS with the record naming that manifest; the real strength-screen manifest too; the real
    gemma4e4b campaign manifest binds native under the _native root; binding twice or without deferral is fatal."""
    as22 = _as22_manifest("gemma4e4b")
    assert as22 == "configs/capeval/manifests/alphasteer_f22_gemma4e4b.yaml" and (REPO / as22).is_file()
    own = tmp_path / "own"
    r = _run_select(REPO, f'F22_WAVE_ROOT={own}\nf22_select_serving_path\necho "MODE=$F22_SERVING_PATH_MODE PATH=${{F22_SERVING_PATH:-unset}} ENGINE=${{TARGET_ENGINE:-unset}} JSON=[${{F22_SERVING_PATH_SELECTION_JSON}}]"')
    assert r.returncode == 0, (r.stdout, r.stderr)
    assert "MODE=deferred PATH=unset ENGINE=unset JSON=[]" in r.stdout
    assert f"LOG [serving-path] DEFERRED to the standup: this job owns its wave root ({own}) — f22_bind_serving_path <manifest> (or f22_standup, from $MANIFEST) selects from the launch's own manifest; explicit F22_SERVING_PATH would override" in r.stdout
    # ... then the wrapper's resolver binds from ITS manifest: the AlphaSteer campaign -> lens, the own root untouched
    r = _run_select(REPO, f'F22_WAVE_ROOT={own}\nf22_select_serving_path\nWAVE_ROOT=$F22_WAVE_ROOT\nMANIFEST={as22}\nf22_bind_serving_path "$MANIFEST"\n' + REPORT + '\necho "ROOT=$WAVE_ROOT"')
    assert r.returncode == 0, (r.stdout, r.stderr)
    assert "PATH=lens\nMODE=auto\nENGINE=lens:stacked" in r.stdout and f"ROOT={own}" in r.stdout
    assert "LOG [serving-path] AUTO-selected lens: every steered cell is lens-only (alphasteer\u00d71)" in r.stdout
    assert f"LOG serving path bound: lens engine=lens:stacked wave_root={own} (from manifest {as22})" in r.stdout
    sel = _selection_json(r.stdout)
    assert sel["cells_source"] == f"manifest:{as22}" and sel["cells_by_disposition"]["skip"] == ["alphasteer"] and sel["selected"] == "lens"
    # F22_SERVING_PATH_DEFER=1 without a root: WAVE_ROOT unbound until the bind derives the PATH's campaign root
    screen = "configs/capeval/manifests/alphasteer_f22_gemma4e4b_as_d0p5.yaml"          # a real strength-screen manifest
    assert (REPO / screen).is_file()
    shared = tmp_path / "shared"
    r = _run_select(REPO, f'F22_SERVING_PATH_DEFER=1\nunset WAVE_ROOT\nf22_select_serving_path\necho "ROOT0=${{WAVE_ROOT:-unbound}}"\nf22_bind_serving_path {screen}\necho "ROOT1=$WAVE_ROOT PATH=$F22_SERVING_PATH"', shared=shared)
    assert r.returncode == 0, (r.stdout, r.stderr)
    assert "ROOT0=unbound" in r.stdout and f"ROOT1={tmp_path}/shared/exp/sdfbase_full22 PATH=lens" in r.stdout    # the lens root
    assert "DEFERRED to the standup: this job set F22_SERVING_PATH_DEFER=1" in r.stdout
    r = _run_select(REPO, 'F22_SERVING_PATH_DEFER=1\nunset WAVE_ROOT\nf22_select_serving_path\nf22_bind_serving_path $(f22_full22_manifest gemma4e4b main)\necho "ROOT1=$WAVE_ROOT PATH=$F22_SERVING_PATH ENGINE=$TARGET_ENGINE"', shared=shared)
    assert r.returncode == 0 and f"ROOT1={tmp_path}/shared/exp/sdfbase_full22_native PATH=lora_native ENGINE=lora_native" in r.stdout, (r.stdout, r.stderr)
    # known-bad: binding a path that was bound at init, or from a manifest that does not exist
    r = _run_select(REPO, 'F22_LAUNCH_CELLS=(alphasteer)\nf22_select_serving_path\nf22_bind_serving_path ' + as22)
    assert r.returncode == 1 and "FATAL f22_bind_serving_path: the serving path is not deferred (mode auto)" in r.stdout
    r = _run_select(REPO, 'F22_SERVING_PATH_DEFER=1\nf22_select_serving_path\nf22_bind_serving_path configs/nope.yaml')
    assert r.returncode == 1 and "FATAL f22_bind_serving_path: manifest configs/nope.yaml does not exist" in r.stdout
    # an EXPLICIT path is bound at init even when the caller also deferred: the wrappers guard their bind on the mode
    r = _run_select(REPO, 'F22_SERVING_PATH_DEFER=1\nf22_select_serving_path\necho "MODE=$F22_SERVING_PATH_MODE"', F22_SERVING_PATH="lens")
    assert r.returncode == 0 and "MODE=explicit" in r.stdout and "EXPLICIT lens" in r.stdout
    # f22_init and f22_standup: the deferred branch leaves engine + root unbound (own root kept) and the standup binds from $MANIFEST first
    sh = (F22 / "common.sh").read_text()
    init = sh[sh.index("f22_init(){"):sh.index("\n}\n", sh.index("f22_init(){"))]
    assert 'if [ "$F22_SERVING_PATH_MODE" = deferred ]; then' in init and 'else WAVE_ROOT="$F22_WAVE_ROOT"; mkdir -p "$WAVE_ROOT"; else unset WAVE_ROOT; fi'.replace("else WAVE_ROOT", "then WAVE_ROOT") in init
    assert init.index("  f22_select_serving_path\n") < init.index("    f22_bind_engine\n") < init.index('WAVE_ROOT="${F22_WAVE_ROOT:-$SHARED/exp/sdfbase_full22$(')
    standup = sh[sh.index("f22_standup(){"):sh.index("f22_standup_stop(){")]
    bind = '[ "${F22_SERVING_PATH_MODE:-}" != deferred ] || f22_bind_serving_path "${MANIFEST:?f22_standup: the serving path was deferred to the standup, which needs MANIFEST}"'
    assert bind in standup and standup.index(bind) < standup.index('local serving_path="${F22_SERVING_PATH:-lora_native}"')


def test_every_f22_init_caller_declares_its_cell_source_before_init():
    """Every script that calls f22_init declares, BEFORE the call, how the serving path is selected: a manifest
    (F22_LAUNCH_MANIFEST), a cell list (F22_LAUNCH_CELLS, empty for a job that stands up no target), a deferral
    (F22_SERVING_PATH_DEFER=1, or owning the wave root with `export F22_WAVE_ROOT`), or the path itself
    (`export F22_SERVING_PATH`). A new caller without one fails this test AND its own f22_init — never a silent default."""
    call = re.compile(r'^\s*f22_init "', re.M)
    decl = re.compile(r'^\s*(F22_LAUNCH_MANIFEST=|F22_LAUNCH_CELLS=|F22_SERVING_PATH_DEFER=1|export F22_WAVE_ROOT=|export F22_SERVING_PATH=)', re.M)
    callers = sorted(p for p in (REPO / "scripts").rglob("*") if p.suffix in (".sh", ".sbatch") and p.name != "common.sh" and call.search(p.read_text()))
    names = {p.name for p in callers}
    assert {"target_attacks.sbatch", "target_static.sbatch", "petri_proxy.sbatch", "bon_climb.sbatch", "smoke_full22.sbatch", "target_attacks_goalset.sbatch",
            "determinism_floor.sbatch", "validate_derived_serving.sh", "target_sampled.sbatch", "attacker_pool.sbatch", "derived_residual.sbatch",
            "dose_capture.sbatch", "misalign_em_organisms.sbatch", "misalign_proxy.sbatch", "as22_common.sh", "asscreen_controls.sbatch",
            "m5t_common.sh", "g8k_common.sh", "native_capctrl_smoke.sbatch"} <= names, names
    undeclared = []
    for p in callers:
        text = p.read_text()
        first_call = call.search(text).start()
        if not any(m.start() < first_call for m in decl.finditer(text)):
            undeclared.append(str(p.relative_to(REPO)))
    assert not undeclared, undeclared
    # the wrappers with their OWN manifests: as22 defers at init and binds in its resolver; the strength screen defers and binds
    # right after its resolver names the manifest, before WAVE_ROOT is read; the misalign proxy declares ITS manifest
    as22 = (REPO / "scripts/alphasteer_f22/sbatch/as22_common.sh").read_text()
    init = as22[as22.index("as22_init(){"):as22.index("as22_resolve_config(){")]
    assert init.index("  F22_SERVING_PATH_DEFER=1\n") < init.index('  f22_init "${1:?sha}"')
    resolve = as22[as22.index("as22_resolve_config(){"):as22.index("\n}\n", as22.index("as22_resolve_config(){"))]
    assert '[ "${F22_SERVING_PATH_MODE:-}" != deferred ] || f22_bind_serving_path "$MANIFEST"' in resolve
    assert resolve.index('[ -f "$MANIFEST" ] || f22_fatal') < resolve.index('f22_bind_serving_path "$MANIFEST"')
    screen = (REPO / "scripts/alphasteer_f22/screen/asscreen_controls.sbatch").read_text()
    assert screen.index("F22_SERVING_PATH_DEFER=1") < screen.index('f22_init "$WAVE_SHA"') < screen.index('[ -f "$MANIFEST" ] || f22_fatal "resolver output incomplete') \
        < screen.index('[ "${F22_SERVING_PATH_MODE:-}" != deferred ] || f22_bind_serving_path "$MANIFEST"') < screen.index("f22_stop_requested") < screen.index('SERVE_DIR="$WAVE_ROOT/')
    mis = (REPO / "scripts/sdfbase/full22/misalign/misalign_proxy.sbatch").read_text()
    assert mis.index('F22_LAUNCH_MANIFEST="configs/capeval/manifests/misalign_full22_${TARGET}.yaml"') < mis.index('f22_init "$WAVE_SHA"')
    for name in ("attacker_pool.sbatch", "derived_residual.sbatch", "strength/dose_capture.sbatch", "misalign/misalign_em_organisms.sbatch"):
        text = (F22 / name).read_text()
        assert text.index("F22_LAUNCH_CELLS=()") < text.index("f22_init ") and "f22_standup" not in text.split("F22_LAUNCH_CELLS=()")[1].replace("stands up no target", ""), name
    # the misalign manifest is a real cell source the selector reads (the (KEY, VIEW) petri pair would have been another cell set)
    mis_manifest = REPO / "configs/capeval/manifests/misalign_full22_gemma4e4b.yaml"
    assert mis_manifest.is_file()
    r = subprocess.run([sys.executable, str(F22 / "serving_path_select.py"), "--manifest", str(mis_manifest), "--json"], capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stderr
    rec = json.loads(r.stdout)
    assert rec["selected"] in ("lens", "lora_native") and rec["cells_source"] == f"manifest:{mis_manifest}"


def test_an_explicit_serving_path_always_wins_and_never_runs_the_selector(tmp_path):
    """The caller's F22_SERVING_PATH wins over every auto verdict — lora_native on an all-lens-only list stays lora_native
    (the standup then refuses the lens-only cells by name, as today), lens on a LoRA list stays lens — and the selector is
    NEVER run: REPO_DIR here holds no selector at all. A malformed value is fatal, as before."""
    empty = tmp_path / "empty"; empty.mkdir()
    r = _run_select(empty, 'F22_LAUNCH_CELLS=(none alphasteer arditi)\nf22_select_serving_path\n' + REPORT, F22_SERVING_PATH="lora_native")
    assert r.returncode == 0, (r.stdout, r.stderr)
    assert "PATH=lora_native\nMODE=explicit\nENGINE=lora_native" in r.stdout
    assert "LOG [serving-path] EXPLICIT lora_native (F22_SERVING_PATH set by caller)" in r.stdout
    sel = _selection_json(r.stdout)
    assert sel == {"cells_by_disposition": None, "cells_source": None, "mode": "explicit", "reason": "F22_SERVING_PATH set by caller",
                   "rule": "kind-aware-2026-09-15", "selected": "lora_native", "skipped": [], "spec_source": None}
    r = _run_select(empty, 'F22_LAUNCH_CELLS=(none si_constv2_mix_full__lora_r4_ce)\nf22_select_serving_path\n' + REPORT, F22_SERVING_PATH="lens")
    assert r.returncode == 0 and "PATH=lens\nMODE=explicit\nENGINE=lens:stacked" in r.stdout and "EXPLICIT lens" in r.stdout
    r = _run_select(empty, 'f22_select_serving_path\n' + REPORT, F22_SERVING_PATH="bogus")
    assert r.returncode == 1 and "FATAL F22_SERVING_PATH must be lora_native or lens, got 'bogus'" in r.stdout
    # and without an explicit value, a tree WITHOUT the selector is fatal by name (never a silent default)
    r = _run_select(empty, 'F22_LAUNCH_CELLS=()\nf22_select_serving_path\n' + REPORT)
    assert r.returncode == 1 and "FATAL no " in r.stdout and "serving_path_select.py on this tree" in r.stdout


def test_select_serving_path_evals_only_the_assignments_a_noisy_interpreter_cannot_poison(tmp_path):
    """The selector's stdout and stderr are merged for the FAILURE message; the eval takes only the four assignment
    lines — an interpreter that warns on stderr (a deprecation notice) still yields the verdict."""
    repo = _select_world(tmp_path)
    noisy = tmp_path / "noisy_python"
    noisy.write_text(f"#!/bin/bash\necho 'DeprecationWarning: something old' >&2\necho 'stray stdout line'\nexec {sys.executable} \"$@\"\n")
    noisy.chmod(0o755)
    script = ("set -u\nf22_log(){ echo \"LOG $*\"; }\nf22_fatal(){ echo \"FATAL $*\"; exit 1; }\n" + _bash_funcs(*SELECT_FUNCS)
              + f"\nREPO_DIR={repo}\nF22_SELECT_PY={noisy}\nF22_LAUNCH_CELLS=(none alphasteer)\nf22_select_serving_path\n" + REPORT)
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                       env={k: v for k, v in os.environ.items() if k not in ("F22_SERVING_PATH", "F22_LAUNCH_CELLS", "KEY", "VIEW")})
    assert r.returncode == 0 and "PATH=lens\nMODE=auto" in r.stdout and "stray" not in r.stderr, (r.stdout, r.stderr)


def test_select_serving_path_refuses_a_malformed_or_duplicate_cell_before_any_gpu_minute(tmp_path):
    """Known-bad inputs: a duplicate id and a malformed id are the partition's refusals, surfaced as a FATAL naming the
    cell (the standup would have refused them later; here the job dies seconds into its allocation, before any server launch)."""
    repo = _select_world(tmp_path)
    r = _run_select(repo, 'F22_LAUNCH_CELLS=(alphasteer alphasteer)\nf22_select_serving_path\n' + REPORT)
    assert r.returncode == 1 and "FATAL serving-path selection refused" in r.stdout and "named twice" in r.stdout, r.stdout
    r = _run_select(repo, 'F22_LAUNCH_CELLS=(bogus__thing)\nf22_select_serving_path\n' + REPORT)
    assert r.returncode == 1 and "FATAL serving-path selection refused" in r.stdout and "malformed cell id 'bogus__thing'" in r.stdout


def test_the_selection_rides_every_standup_record_and_the_jsonl_line_stays_json():
    """f22_init resolves the path ONCE (before WAVE_ROOT), f22_standup writes the selection beside the standup and hands it
    to mt_serving_up.py (--serving-path-selection, only when set: a caller without f22_init keeps the argv byte-identical),
    the standup_jobs.jsonl printf carries `serving_path_selection` with a JSON `null` default, and the attacker-pool fatal
    names the path + selection so a target that cannot see its pool is recoverable."""
    sh = (F22 / "common.sh").read_text()
    init = sh[sh.index("f22_init(){"):sh.index("\n}\n", sh.index("f22_init(){"))]
    assert init.index('cd "$REPO_DIR" || exit 1') < init.index("  f22_select_serving_path\n") < init.index("    f22_bind_engine\n") < init.index('WAVE_ROOT="${F22_WAVE_ROOT:-$SHARED')
    assert "serving_path_mode=$F22_SERVING_PATH_MODE" in init
    standup = sh[sh.index("f22_standup(){"):sh.index("f22_standup_stop(){")]
    assert 'if [ -n "${F22_SERVING_PATH_SELECTION_JSON:-}" ]; then' in standup
    assert """printf '%s\\n' "$F22_SERVING_PATH_SELECTION_JSON" > "$out/serving_path_selection.json\"""" in standup
    assert 'sel_args=(--serving-path-selection "$out/serving_path_selection.json")' in standup
    assert '--serving-path "$serving_path" ${lens_pins[@]+"${lens_pins[@]}"} \\\n    ${sel_args[@]+"${sel_args[@]}"} \\' in standup
    assert '"upstream_timeout_s":%s,"serving_path_selection":%s,"written_at":"%s"}' in standup
    assert '"${TARGET_UPSTREAM_TIMEOUT_S:-null}" "${F22_SERVING_PATH_SELECTION_JSON:-null}" "$(date -u +%FT%TZ)" >> "$out/standup_jobs.jsonl"' in standup
    assert "serving_path=$serving_path (selection ${F22_SERVING_PATH_MODE:-unset}" in standup
    # the printf composes VALID JSON with the record in place AND with the null default (a slice of the template)
    for value in ('{"mode":"auto","selected":"lens"}', "null"):
        r = subprocess.run(["bash", "-c", 'printf \'{"a":%s,"serving_path_selection":%s,"written_at":"%s"}\\n\' 1 "$1" now', "_", value],
                           capture_output=True, text=True)
        assert json.loads(r.stdout)["serving_path_selection"] == (json.loads(value) if value != "null" else None)
    # the empty-array idiom under set -u: nothing added when the record is unset
    r = subprocess.run(["bash", "-c", 'set -u\nx(){ printf "[%s]" "$@"; echo; }\nsel_args=()\nx a ${sel_args[@]+"${sel_args[@]}"} b'],
                       capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "[a][b]"
    assert "this job's is F22_SERVING_PATH=$F22_SERVING_PATH, selection ${F22_SERVING_PATH_MODE:-unset}" in sh   # f22_live_attackers
    # the rule is reached ONLY through the selector — no shell predicate on cell names beside it
    assert sh.count("serving_path_select.py") >= 1
    for word in ("alphasteer", "arditi", "static_r"):
        assert not re.search(rf"case .*{word}", sh), word
    sub = (F22 / "submit_full22.sh").read_text()
    assert "KIND-AWARE default, 2026-09-15" in sub and "[serving-path] AUTO-selected" in sub
    assert "The attacker pool follows the" in sub and "SAME F22_SERVING_PATH as its targets" in sub


def test_petri_drive_applies_the_same_rule_over_the_same_manifest_when_the_caller_set_no_path(home_site_default, tmp_path):
    """The grading-host driver mirrors f22_init: an explicit F22_SERVING_PATH wins and is logged as EXPLICIT; unset, the
    SAME selector over the SAME (KEY, VIEW) manifest picks the engine BEFORE ROOT_SUFFIX / REMOTE_OUT derive from it — so
    the drive looks under the root the cluster job wrote. Run to the key fetch (the secret-manager stub fails there): an
    all-lens-only (AlphaSteer) synthetic manifest -> `AUTO-selected lens`; the real glm46vflash/main manifest -> native."""
    drv = (F22 / "petri_drive_full22.sh").read_text()
    sel_i = drv.index('scripts/sdfbase/full22/serving_path_select.py --manifest "$MANIFEST" --cells-source "manifest:$MANIFEST" --sh')
    assert drv.index('PY="${F22_PY:-$WT/.venv/bin/python}"') < drv.index('MANIFEST="configs/capeval/manifests/petri_full22_${TAG}.yaml"') < sel_i
    assert sel_i < drv.index('ROOT_SUFFIX="$( [ "$F22_SERVING_PATH" = lens ] || echo _native )"') < drv.index('REMOTE_OUT="$SHARED/exp/sdfbase_full22$ROOT_SUFFIX')
    assert 'F22_SERVING_PATH_MODE=explicit; echo "[serving-path] EXPLICIT $F22_SERVING_PATH (F22_SERVING_PATH set by caller)"' in drv
    assert 'F22_SERVING_PATH="${F22_SERVING_PATH:-lora_native}"' not in drv
    assert "serving_path_mode=$F22_SERVING_PATH_MODE" in drv                       # DRIVE_START records it

    def world(cells):
        name = "lens" if cells else "real"
        wt = tmp_path / f"wt_{name}"; home = tmp_path / f"home_{name}"; home.mkdir()
        (wt / "scripts/sdfbase/full22").mkdir(parents=True); (wt / ".venv/bin").mkdir(parents=True)
        (wt / "scripts/sdfbase/full22/petri_drive_full22.sh").write_text(drv)
        (wt / "scripts/sdfbase/full22/serving_path_select.py").symlink_to(F22 / "serving_path_select.py")
        (wt / "src").symlink_to(REPO / "src")
        if cells is None:
            (wt / "configs").symlink_to(REPO / "configs")
        else:
            man = yaml.safe_load((REPO / "configs/capeval/manifests/petri_full22_glm46vflash.yaml").read_text())
            man["petri"]["cells"] = list(cells)
            (wt / "configs/capeval/manifests").mkdir(parents=True)
            (wt / "configs/capeval/manifests/petri_full22_glm46vflash.yaml").write_text(yaml.safe_dump(man, sort_keys=False))
        py = wt / ".venv/bin/python"; py.write_text(f"#!/bin/bash\nexec {sys.executable} \"$@\"\n"); py.chmod(0o755)
        stub = tmp_path / f"fake-secret-cmd_{name}"; stub.write_text("#!/bin/bash\nexit 1\n"); stub.chmod(0o755)
        _petri_site(wt, home, str(tmp_path / f"shared_{name}"))
        return wt, home, stub

    def drive(wt, home, stub, **env):
        base = {k: v for k, v in os.environ.items() if k not in ("F22_SERVING_PATH", "F22_AUDITOR_ORG")}
        r = subprocess.run(["bash", "scripts/sdfbase/full22/petri_drive_full22.sh", "glm46vflash", "main", "cluster_a", "0" * 40],
                           cwd=wt, capture_output=True, text=True, timeout=120,
                           env={**base, "HOME": str(home), "USER": "tester", "SISTEER_SECRET_CMD": str(stub), **env})
        return r.returncode, r.stdout + r.stderr

    wt, home, stub = world(["none", "alphasteer"])                              # AlphaSteer-only: the lens-only shape
    rc, out = drive(wt, home, stub)
    assert rc == 1 and "(SISTEER_SECRET_CMD) failed for lowprio" in out, out  # ran past the selection to the key fetch
    assert "[serving-path] AUTO-selected lens: every steered cell is lens-only (alphasteer×1); explicit F22_SERVING_PATH would override" in out
    rc, out = drive(wt, home, stub, F22_SERVING_PATH="lora_native")           # explicit wins, logged as such
    assert rc == 1 and "[serving-path] EXPLICIT lora_native (F22_SERVING_PATH set by caller)" in out and "AUTO-selected" not in out
    wt, home, stub = world(None)                                                # the real glm46vflash/main view: LoRA cells -> native
    rc, out = drive(wt, home, stub)
    assert rc == 1 and "[serving-path] AUTO-selected lora_native:" in out and "steered cells natively servable" in out, out


def test_validate_derived_serving_slice_selects_from_the_generated_slice_not_the_parent_view(tmp_path):
    """In slice mode validate_derived_serving.sh served `none` + the named cells but had bound
    the path at f22_init from the PARENT view's manifest (ten cells incl. native LoRA) — a lens-only slice ran native,
    the sliced PSR cell was skipped, the validation never exercised it. Now slice mode owns a scratch root
    (exp/sdfbase_full22_validate), DEFERS, and binds from the GENERATED slice manifest right after it exists; full mode
    is unchanged. Run for real: make_validation_slice.py over qwen35_4b/main for the PSR-only slice ->
    the selector and the deferred bind read the SLICE (cells_source names it, its record lists exactly the slice's cells)
    — since native PSR serving landed a psr cell is a natively servable kind, so the slice selects NATIVE with the psr
    cell SERVED (before it selected the lens); the parent manifest -> native; a mixed slice -> native with nothing
    skipped; an explicit path is bound at init and the guarded bind leaves it."""
    import make_validation_slice as mvs
    target, view = "qwen35_4b", "main"
    psr, lora = "constv2-mix-full__psr_ce_all_minus_ctl_distilled", "constv2-mix-full__lora_r4_ce_minus_ctl"
    out = tmp_path / "validate" / "slice"
    assert mvs.main(["--target", target, "--view", view, "--cells", psr, "--out", str(out)]) == 0
    slice_manifest = out / "slice_manifest.yaml"
    assert list(cm.load_petri_spec(slice_manifest).cells) == ["none", ra.harness_cell_id(psr)]
    # the selector over the generated slice: naming the slice, its cells exactly (the bug bound the PARENT view's ten);
    # the psr cell SERVES natively
    r = subprocess.run([sys.executable, str(F22 / "serving_path_select.py"), "--manifest", str(slice_manifest), "--json"], capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stderr
    rec = json.loads(r.stdout)
    assert rec["selected"] == "lora_native" and rec["cells_source"] == f"manifest:{slice_manifest}"
    assert rec["cells_by_disposition"] == {"passthrough": ["none"], "serve": [ra.harness_cell_id(psr)], "skip": []}
    assert rec["cells_requested"] == ["none", ra.harness_cell_id(psr)]
    assert rec["reason"] == "1 of 1 steered cells natively servable; skipped by name: none"
    parent = ROSTER["targets"][target]["views"][view]["manifest"]
    r = subprocess.run([sys.executable, str(F22 / "serving_path_select.py"), "--manifest", parent, "--json"], capture_output=True, text=True, cwd=REPO)
    assert json.loads(r.stdout)["selected"] == "lora_native"
    # the script's slice mode through the sliced functions: own root + deferral at init, the bind from the slice manifest
    own = tmp_path / "shared" / "exp" / "sdfbase_full22_validate"
    body = (f'export F22_WAVE_ROOT={own}; F22_SERVING_PATH_DEFER=1\nf22_select_serving_path\nWAVE_ROOT="$F22_WAVE_ROOT"\n'
            f'OUT="$WAVE_ROOT/validate/{target}_manual"\nMANIFEST={slice_manifest}\n'
            '[ "${F22_SERVING_PATH_MODE:-}" != deferred ] || f22_bind_serving_path "$MANIFEST"\n' + REPORT + '\necho "ROOT=$WAVE_ROOT OUT=$OUT"')
    r = _run_select(REPO, body, shared=tmp_path / "shared")
    assert r.returncode == 0, (r.stdout, r.stderr)
    assert "PATH=lora_native\nMODE=auto\nENGINE=lora_native" in r.stdout and f"ROOT={own} OUT={own}/validate/{target}_manual" in r.stdout
    assert "DEFERRED to the standup: this job owns its wave root" in r.stdout and "AUTO-selected lora_native: 1 of 1 steered cells natively servable; skipped by name: none" in r.stdout
    assert _selection_json(r.stdout)["cells_source"] == f"manifest:{slice_manifest}"
    # a mixed slice (the PSR + a LoRA derived cell) binds native with BOTH served (nothing skipped by kind)
    out2 = tmp_path / "validate2" / "slice"
    assert mvs.main(["--target", target, "--view", view, "--cells", f"{psr},{lora}", "--out", str(out2)]) == 0
    r = _run_select(REPO, body.replace(str(slice_manifest), str(out2 / "slice_manifest.yaml")), shared=tmp_path / "shared")
    assert r.returncode == 0 and "PATH=lora_native\nMODE=auto" in r.stdout and "2 of 2 steered cells natively servable; skipped by name: none" in r.stdout, r.stdout
    assert _selection_json(r.stdout)["cells_by_disposition"]["serve"] == [ra.harness_cell_id(psr), ra.harness_cell_id(lora)]
    # an explicit path wins: bound at init, the guarded bind is a no-op
    r = _run_select(REPO, body, shared=tmp_path / "shared", F22_SERVING_PATH="lora_native")
    assert r.returncode == 0 and "PATH=lora_native\nMODE=explicit" in r.stdout and "AUTO-selected" not in r.stdout
    # the script itself: slice mode declares the own root + deferral BEFORE f22_init, full mode the campaign manifest; the bind
    # sits right after the generated MANIFEST is named and before the slice bank view / the standup; OUT still derives from WAVE_ROOT
    src = (F22 / "validate_derived_serving.sh").read_text()
    i_if, i_init = src.index('if [ "$MODE" = slice ]; then\n  shared='), src.index('f22_init "$WAVE_SHA"; f22_resolve_config "$KEY" "$VIEW"')
    assert i_if < src.index('export F22_WAVE_ROOT="$shared/exp/sdfbase_full22_validate"; F22_SERVING_PATH_DEFER=1') < src.index('F22_LAUNCH_MANIFEST=$(f22_full22_manifest "$KEY" "$VIEW")') < i_init
    assert i_init < src.index('OUT="$WAVE_ROOT/validate/${TAG}_${SLURM_JOB_ID:-manual}"')
    i_man = src.index('CFG="$OUT/slice/slice_config.yaml"; MANIFEST="$OUT/slice/slice_manifest.yaml"')
    i_bind = src.index('[ "${F22_SERVING_PATH_MODE:-}" != deferred ] || f22_bind_serving_path "$MANIFEST"')
    assert i_man < i_bind < src.index('make_bank_view.py --target "$KEY" --view "$SLICE_VIEW"') < src.index('f22_standup attacks "$OUT" ""')
    assert 'else\n  f22_bank_view "$MPY"\nfi' in src          # full mode: untouched


# --------------------------------------------------------------------------------------------------------------------
# THE LAUNCH-TIME ARGV HOOK survives every lane (2026-09-16, wave 0e21d6dc): F22_STANDUP_EXTRA is built ON TOP of the
# value inherited from the launch environment (a scalar, ONE token — e.g. --serving-path=lens), never assigned afresh
# --------------------------------------------------------------------------------------------------------------------


def _fn_from(path: Path, name: str) -> str:
    """The text of one top-level bash function (``name(){`` through its closing ``}`` line) of ``path``."""
    sh = path.read_text()
    i = sh.index(f"{name}(){{")
    j = sh.index("\n}\n", i) + 3
    return sh[i:j]


_EXTRA_HELPERS = ("f22_standup_extra_inherited_check", "f22_standup_extra_reset")


def _extra_env(inherited):
    """The launch environment a batch job inherits: F22_STANDUP_EXTRA exported as a SCALAR (``inherited``; None = unset,
    "" = exported empty) — never --export=."""
    env = {k: v for k, v in os.environ.items() if k not in ("F22_STANDUP_EXTRA", "F22_THINKING_ON", "F22_SERVING_PATH")}
    if inherited is not None:
        env["F22_STANDUP_EXTRA"] = inherited
    return env


def _run_extra_rc(script: str, *, inherited=None):
    """Run a bash snippet after common.sh's F22_STANDUP_EXTRA helpers under `set -eu` and return
    ``(rc, tokens, stderr)`` — the tokens exactly as the pre-normalisation array reads them
    (``${F22_STANDUP_EXTRA[@]+"${F22_STANDUP_EXTRA[@]}"}``), one per line, on rc 0."""
    body = ('set -eu\nf22_log(){ :; }\nf22_fatal(){ echo "FATAL: $*" >&2; exit 1; }\n'
            + "\n".join(_fn_from(F22 / "common.sh", h) for h in _EXTRA_HELPERS) + "\n" + script + "\n"
            + 'for t in ${F22_STANDUP_EXTRA[@]+"${F22_STANDUP_EXTRA[@]}"}; do printf "%s\\n" "$t"; done\n')
    r = subprocess.run(["bash", "-c", body], capture_output=True, text=True, env=_extra_env(inherited))
    return r.returncode, r.stdout.split("\n")[:-1], r.stderr


def _run_extra(script: str, *, inherited=None) -> list:
    rc, toks, err = _run_extra_rc(script, inherited=inherited)
    assert rc == 0, (toks, err)
    return toks


def test_f22_standup_extra_reset_keeps_the_inherited_hook_and_rebuilds_idempotently():
    """KNOWN-BAD first: the shape target_sampled.sbatch had — a fresh `=(` — DROPS the inherited hook. `+=(` keeps it as
    element 0 (bash converts an inherited scalar to an array in place; unset -> the array is declared); the helper keeps
    the inherited tokens in front of the given flags, and a second call REPLACES the lane's flags without re-inheriting
    its own previous ones (the Nemotron probe calls m5t_/g8k_standup_extra twice)."""
    hook = "--upstream-timeout-s=7200"
    assert _run_extra("F22_STANDUP_EXTRA=(--enable-log-requests)", inherited=hook) == ["--enable-log-requests"]  # the defect
    assert _run_extra("F22_STANDUP_EXTRA+=(--enable-log-requests)", inherited=hook) == [hook, "--enable-log-requests"]
    assert _run_extra("F22_STANDUP_EXTRA+=(--enable-log-requests)") == ["--enable-log-requests"]
    assert _run_extra("f22_standup_extra_reset --a 1 --b", inherited=hook) == [hook, "--a", "1", "--b"]
    # EXPORTED-EMPTY: `F22_STANDUP_EXTRA=""` in the environment is ZERO arguments through the helper —
    # the array byte-identical to the unset case; the bare `+=` first touch is the shape that keeps the empty element
    # (why every lane's first touch is the helper); a second call stays empty-free
    assert _run_extra("f22_standup_extra_reset --enable-log-requests", inherited="") == ["--enable-log-requests"]
    assert _run_extra("f22_standup_extra_reset --enable-log-requests", inherited="") == \
        _run_extra("f22_standup_extra_reset --enable-log-requests")
    assert _run_extra("f22_standup_extra_reset", inherited="") == [] == _run_extra("f22_standup_extra_reset")
    assert _run_extra("f22_standup_extra_reset --a\nf22_standup_extra_reset --c", inherited="") == ["--c"]
    assert _run_extra("f22_standup_extra_reset\nF22_STANDUP_EXTRA+=(--d 2)", inherited="") == ["--d", "2"]
    assert _run_extra("F22_STANDUP_EXTRA+=(--enable-log-requests)", inherited="") == ["", "--enable-log-requests"]  # known-bad
    # AN ENGINE PIN ON THE HOOK: `--serving-path=lens` — the Modal producer's spelling — or a lone
    # `--serving-path` refuses BY NAME, pointing at F22_SERVING_PATH; any position, at the snapshot and standalone
    for pin in ("--serving-path=lens", "--serving-path", "--serving-path=lora_native"):
        rc, toks, err = _run_extra_rc("f22_standup_extra_reset --enable-log-requests", inherited=pin)
        assert rc == 1 and toks == [], (pin, toks)
        assert f"FATAL: F22_STANDUP_EXTRA carries '{pin}'" in err and "travels as F22_SERVING_PATH" in err, err
        assert "load_serving_path_selection refuses the record" in err
        rc, _, err = _run_extra_rc("f22_standup_extra_inherited_check", inherited=pin)
        assert rc == 1 and "F22_SERVING_PATH" in err
    rc, _, err = _run_extra_rc("F22_STANDUP_EXTRA=(--a --serving-path lens)\nf22_standup_extra_inherited_check")
    assert rc == 1 and "carries '--serving-path'" in err                           # a lane-set one refuses too
    assert _run_extra_rc("f22_standup_extra_inherited_check", inherited="")[0] == 0
    assert _run_extra_rc("f22_standup_extra_inherited_check", inherited="--serving-path-selection")[0] == 0   # not the pin
    assert _run_extra("f22_standup_extra_reset --a 1 --b\nf22_standup_extra_reset --c", inherited=hook) == [hook, "--c"]
    assert _run_extra("f22_standup_extra_reset --a 1\nF22_STANDUP_EXTRA+=(--d 2)\nf22_standup_extra_reset --c",
                      inherited=hook) == [hook, "--c"]
    assert _run_extra("f22_standup_extra_reset\nF22_STANDUP_EXTRA+=(--d 2)", inherited=hook) == [hook, "--d", "2"]
    assert _run_extra("f22_standup_extra_reset --a 1\nf22_standup_extra_reset --c") == ["--c"]
    assert _run_extra("f22_standup_extra_reset") == []
    assert _run_extra("") == []
    assert _run_extra("", inherited=hook) == [hook]                       # no lane flags: the hook alone, as before
    # the inherited scalar is ONE token whatever it contains (the documented convention: strength_served.py)
    assert _run_extra("F22_STANDUP_EXTRA+=(--x)", inherited="--a b") == ["--a b", "--x"]
    assert _run_extra("f22_standup_extra_reset --x", inherited="--a b") == ["--a b", "--x"]


def test_no_lane_assigns_f22_standup_extra_from_scratch():
    """Every .sbatch / .sh under scripts/ builds F22_STANDUP_EXTRA on top of the inherited value — `+=(` where a lane sets
    it once, f22_standup_extra_reset where it rebuilds per branch or per call. A bare `F22_STANDUP_EXTRA=(` anywhere but
    the helper's own body is the clobber that hid a launch-time pin from the sampled lane."""
    offenders = []
    for path in sorted(list(REPO.glob("scripts/**/*.sbatch")) + list(REPO.glob("scripts/**/*.sh"))):
        for n, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            code = "" if line.lstrip().startswith("#") else line.split(" #", 1)[0]
            if re.search(r"F22_STANDUP_EXTRA=\(", code) and "F22_STANDUP_INHERITED" not in code:
                offenders.append(f"{path.relative_to(REPO)}:{n}: {line.strip()}")
    assert offenders == [], offenders
    # ... and the FIRST touch of every lane is the helper (a bare `+=` first touch keeps an exported-empty
    # scalar as an empty argv element): a `+=` line must follow, in its file, the helper or a lane function whose body
    # starts with it (m5t_/g8k_standup_extra)
    late = []
    for path in sorted(list(REPO.glob("scripts/**/*.sbatch")) + list(REPO.glob("scripts/**/*.sh"))):
        if path.name == "common.sh":
            continue
        armed = False
        for n, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            code = "" if line.lstrip().startswith("#") else line.split(" #", 1)[0]
            if re.search(r"\bf22_standup_extra_reset\b|\b(m5t|g8k)_standup_extra\b", code):
                armed = True
            if "F22_STANDUP_EXTRA+=(" in code and not armed:
                late.append(f"{path.relative_to(REPO)}:{n}: {line.strip()}")
    assert late == [], late
    assert "f22_standup_extra_reset --enable-log-requests" in (F22 / "sampled" / "target_sampled.sbatch").read_text()
    assert (F22 / "target_attacks.sbatch").read_text().count("f22_standup_extra_reset ") == 3
    assert "f22_standup_extra_reset" in (F22 / "misalign" / "misalign_proxy.sbatch").read_text()
    assert "f22_standup_extra_reset;" in (F22 / "target_attacks_goalset.sbatch").read_text()
    assert "f22_standup_extra_reset --upstream-timeout-s" in (REPO / "scripts/sdfbase/math500/m5t_common.sh").read_text()
    assert "f22_standup_extra_reset --upstream-timeout-s" in (REPO / "scripts/sdfbase/gsm8k/g8k_common.sh").read_text()
    assert "f22_standup_extra_reset --enable-log-requests" in \
        (REPO / "scripts/alphasteer_f22/screen/asscreen_controls.sbatch").read_text()
    # the helpers are defined once, in common.sh, before f22_init refuses the pin and f22_standup reads the array
    common = (F22 / "common.sh").read_text()
    assert common.count("f22_standup_extra_reset(){") == 1 == common.count("f22_standup_extra_inherited_check(){")
    assert common.index("f22_standup_extra_reset(){") < common.index("f22_init(){") < common.index("f22_standup(){")
    init = common[common.index("f22_init(){"):common.index("f22_standup(){")]
    assert init.index("  f22_standup_extra_inherited_check\n") < init.index("  f22_select_serving_path\n")   # before the selection
    standup = common[common.index("f22_standup(){"):common.index("f22_standup_stop(){")]
    assert standup.index("  f22_standup_extra_inherited_check\n") < standup.index('"$SPY" scripts/sdfbase/mt_serving_up.py')
    assert '${standup_extra[@]+"${standup_extra[@]}"}' in standup and '${F22_STANDUP_EXTRA[@]+"${F22_STANDUP_EXTRA[@]}"} \\' not in standup


def test_each_lane_builds_its_standup_flags_on_top_of_the_inherited_hook():
    """The lanes' OWN lines, run: the hook is element 0 and the lane's flags follow, on every window branch of
    target_attacks, on the goalset's optional timeout, and on the math500 / gsm8k functions called once or twice."""
    hook = "--upstream-timeout-s=7200"
    sampled = [l for l in (F22 / "sampled" / "target_sampled.sbatch").read_text().splitlines()
               if l.startswith("F22_STANDUP_EXTRA") or l.startswith("f22_standup_extra_reset")]
    assert sampled == ["f22_standup_extra_reset --enable-log-requests"]
    assert _run_extra(sampled[0], inherited=hook) == [hook, "--enable-log-requests"]
    assert _run_extra(sampled[0]) == ["--enable-log-requests"] == _run_extra(sampled[0], inherited="")   # exported-empty
    resets = [l.strip() for l in (F22 / "target_attacks.sbatch").read_text().splitlines()
              if l.strip().startswith("f22_standup_extra_reset ")]
    assert len(resets) == 3
    pre = 'PROXY_UPSTREAM_TIMEOUT_S=3600; CAP_TOKENS=4096\n'
    assert _run_extra(pre + resets[0], inherited=hook) == \
        [hook, "--upstream-timeout-s", "3600", "--driver-max-tokens", "4096"]
    assert _run_extra(pre + resets[0] + "\n" + resets[1], inherited=hook) == [hook, "--upstream-timeout-s", "3600"]
    assert _run_extra(pre + resets[0] + "\n" + resets[2], inherited=hook) == \
        [hook, "--upstream-timeout-s", "3600", "--driver-max-tokens", "4096"]
    assert _run_extra(pre + resets[0] + "\n" + resets[1]) == ["--upstream-timeout-s", "3600"]
    goalset = [l for l in (F22 / "target_attacks_goalset.sbatch").read_text().splitlines()
               if l.startswith("f22_standup_extra_reset") or l.startswith("F22_STANDUP_EXTRA+=")]
    assert len(goalset) == 2
    assert _run_extra('PROXY_UPSTREAM_TIMEOUT_S=7200; EFFECTIVE_CAP=2048\n' + "\n".join(goalset), inherited=hook) == \
        [hook, "--upstream-timeout-s", "7200", "--driver-max-tokens", "2048"]
    assert _run_extra('PROXY_UPSTREAM_TIMEOUT_S=""; EFFECTIVE_CAP=2048\n' + "\n".join(goalset), inherited=hook) == \
        [hook, "--driver-max-tokens", "2048"]
    m5t = _fn_from(REPO / "scripts/sdfbase/math500/m5t_common.sh", "m5t_standup_extra")
    pre = ('M5T_UPSTREAM_TIMEOUT_S=7200; M5T_LENS_SALT_POLICY=per_request; CELL=none; FITTED=0; PIN=""; '
           'M5T_REASONING_PARSER=""; M5T_LOG_REQUESTS_EFFECTIVE=1\n')
    once = _run_extra(pre + m5t + "\nm5t_standup_extra", inherited=hook)
    assert once == [hook, "--upstream-timeout-s", "7200", "--lens-salt-policy", "per_request",
                    "--enable-log-requests", "--cell", "none"]
    assert _run_extra(pre + m5t + "\nm5t_standup_extra\nm5t_standup_extra", inherited=hook) == once
    assert _run_extra(pre + m5t + "\nm5t_standup_extra") == once[1:]
    assert _run_extra(pre + m5t + "\nm5t_standup_extra", inherited="") == once[1:]                    # exported-empty
    g8k = _fn_from(REPO / "scripts/sdfbase/gsm8k/g8k_common.sh", "g8k_standup_extra")
    pre = ('G8K_UPSTREAM_TIMEOUT_S=7200; G8K_LENS_SALT_POLICY=per_request; G8K_STANDUP_CELL=none; FITTED=0; PIN=""; '
           'G8K_REASONING_PARSER=""; G8K_THINKING=off\n')
    once = _run_extra(pre + g8k + "\ng8k_standup_extra", inherited=hook)
    assert once == [hook, "--upstream-timeout-s", "7200", "--lens-salt-policy", "per_request",
                    "--enable-log-requests", "--cell", "none"]
    assert _run_extra(pre + g8k + "\ng8k_standup_extra\ng8k_standup_extra", inherited=hook) == once
    assert _run_extra(pre + g8k + "\ng8k_standup_extra", inherited="") == once[1:]                    # exported-empty


def _compose_standup_argv(tmp_path, *, inherited, serving_path, selected, lane=""):
    """f22_standup's OWN argv composition, RUN (the slice from the serving-path locals through the mt_serving_up.py
    invocation) with $SPY a stub that prints its argv, F22_STANDUP_EXTRA exported as the launch environment's scalar
    (``inherited``; None = unset), F22_SERVING_PATH = ``serving_path`` and the selection record f22_init would have
    recorded for ``selected``, after the lane's own ``lane`` line(s). Returns ``(rc, argv | None, log)`` — argv = the
    tokens mt_serving_up.py would parse, None when the composition refused before launching."""
    sh = (F22 / "common.sh").read_text()
    standup = sh[sh.index("f22_standup(){"):sh.index("f22_standup_stop(){")]
    start = standup.index('  local serving_path="${F22_SERVING_PATH:-lora_native}"')
    end = standup.index("    ${F22_THINKING_ON:+--thinking-on} \\\n") + len("    ${F22_THINKING_ON:+--thinking-on} \\\n")
    body = standup[start:end] + '    > "$out/argv.txt" 2>&1\n'
    out = tmp_path / f"standup-{len(list(tmp_path.glob('standup-*')))}"
    spy = tmp_path / "spy.sh"
    spy.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n'); spy.chmod(0o755)
    selection = json.dumps({"mode": "explicit", "selected": selected, "reason": "F22_SERVING_PATH set by caller",
                            "rule": "test", "cells_source": None, "cells_by_disposition": None})
    script = ("set -u\nf22_log(){ echo \"LOG $*\"; }\nf22_fatal(){ echo \"FATAL: $*\"; exit 1; }\nnvidia-smi(){ :; }\n"
              + "\n".join(_fn_from(F22 / "common.sh", h) for h in _EXTRA_HELPERS) + "\n"
              + f"out={out}\nSPY={spy}\nREPO_DIR=/r\nCFG=c.yaml\nMANIFEST=m.yaml\nTT_TWO_SI_DEFENSES_DIR=/d\nKEY=gemma4e4b\n"
              + "CUDA_VISIBLE_DEVICES=0\nF22_LOAD_CHECK=0\nmode=attacks\npport=''\ngeval=''\ngL=''\ngS=''\ngLP=''\n"
              + f"F22_SERVING_PATH={serving_path}\nF22_SERVING_PATH_SELECTION_JSON='{selection}'\n"
              + lane + "\ncompose(){\n" + body + "}\ncompose\n")
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=_extra_env(inherited))
    argv = None
    if (out / "argv.txt").is_file():
        lines = (out / "argv.txt").read_text().split("\n")[:-1]
        assert lines[0] == "scripts/sdfbase/mt_serving_up.py", lines[:2]
        argv = lines[1:]
    return r.returncode, argv, r.stdout + r.stderr


def test_selection_through_launch_refuses_an_engine_pin_on_the_hook_and_carries_f22_serving_path():
    """Through the launch: the argv f22_standup composes -> the REAL mt_serving_up parser -> the REAL
    load_serving_path_selection over the record f22_init wrote. KNOWN-BAD first: the pre-fix append of the Modal
    producer's `F22_STANDUP_EXTRA=--serving-path=lens` puts a later --serving-path on the argv, the parser takes it
    (last wins) and the selection record — 'lora_native', chosen before the hook could speak — is refused: the pin
    could not launch. Now: the pin is refused BY NAME before the standup is launched, pointing at F22_SERVING_PATH;
    F22_SERVING_PATH=lens selects and records lens and the composed argv agrees with its record; an exported-empty
    hook composes the argv byte-identical to unset, with and without a lane rebuild."""
    sys.path.insert(0, str(Path(__file__).parent))
    from test_mt_serving_up_serving_path import _load_mt
    mt = _load_mt()
    tmp = Path(tempfile.mkdtemp(prefix="f22-standup-argv-"))
    rc, native, log = _compose_standup_argv(tmp, inherited=None, serving_path="lora_native", selected="lora_native")
    assert rc == 0 and native is not None, log
    assert native.count("--serving-path") == 1 and native[native.index("--serving-path") + 1] == "lora_native"
    ns = mt._build_parser().parse_args(native)
    assert ns.serving_path == "lora_native"
    rec = mt.load_serving_path_selection(ns.serving_path_selection, serving_path=ns.serving_path)
    assert rec["selected"] == "lora_native" and rec["mode"] == "explicit"
    # the PRE-FIX shape (the hook appended after the pins): parsed, the later value wins and the record is refused
    ns_bad = mt._build_parser().parse_args(native + ["--serving-path=lens"])
    assert ns_bad.serving_path == "lens"
    with pytest.raises(SystemExit, match="selected 'lora_native' but this standup serves --serving-path lens"):
        mt.load_serving_path_selection(ns_bad.serving_path_selection, serving_path=ns_bad.serving_path)
    # NOW: the hook carrying the pin refuses by name before anything is launched — no argv at all
    rc, argv, log = _compose_standup_argv(tmp, inherited="--serving-path=lens", serving_path="lora_native",
                                          selected="lora_native", lane="f22_standup_extra_reset --enable-log-requests")
    assert rc == 1 and argv is None and "FATAL: F22_STANDUP_EXTRA carries '--serving-path=lens'" in log, log
    assert "travels as F22_SERVING_PATH" in log
    rc, argv, log = _compose_standup_argv(tmp, inherited="--serving-path=lens", serving_path="lora_native", selected="lora_native")
    assert rc == 1 and argv is None and "carries '--serving-path=lens'" in log, log     # no lane touch: the splice refuses
    # THE RIGHT CHANNEL: F22_SERVING_PATH=lens selects lens, records lens, composes --serving-path lens + the lens pins,
    # and the argv agrees with its own record through the real reader
    rc, lens, log = _compose_standup_argv(tmp, inherited=None, serving_path="lens", selected="lens")
    assert rc == 0 and lens is not None, log
    assert lens[lens.index("--serving-path") + 1] == "lens" and "--reduction-mode" in lens and "--require-attested-mode" in lens
    ns = mt._build_parser().parse_args(lens)
    assert mt.load_serving_path_selection(ns.serving_path_selection, serving_path=ns.serving_path)["selected"] == "lens"
    # EXPORTED-EMPTY, through the argv: byte-identical to unset — no lane touch, and with the sampled lane's line
    rc, empty, log = _compose_standup_argv(tmp, inherited="", serving_path="lora_native", selected="lora_native")
    assert rc == 0 and empty is not None and "" not in empty, log
    assert [t for t in empty if not t.startswith(str(tmp))] == [t for t in native if not t.startswith(str(tmp))]
    lane = "f22_standup_extra_reset --enable-log-requests"
    rc, e_lane, _ = _compose_standup_argv(tmp, inherited="", serving_path="lora_native", selected="lora_native", lane=lane)
    rc2, u_lane, _ = _compose_standup_argv(tmp, inherited=None, serving_path="lora_native", selected="lora_native", lane=lane)
    assert rc == rc2 == 0 and "" not in e_lane and e_lane[-1] == "--enable-log-requests"
    assert [t for t in e_lane if not t.startswith(str(tmp))] == [t for t in u_lane if not t.startswith(str(tmp))]
    mt._build_parser().parse_args(e_lane)                                    # the standup parser takes it
    # a genuine hook rides in front of the lane's flags and parses
    rc, hooked, _ = _compose_standup_argv(tmp, inherited="--upstream-timeout-s=7200", serving_path="lora_native",
                                          selected="lora_native", lane=lane)
    assert rc == 0 and hooked[-2:] == ["--upstream-timeout-s=7200", "--enable-log-requests"]
    assert mt._build_parser().parse_args(hooked).upstream_timeout_s == 7200


# --- the twin marker: `paired_of` on the paired views ----------------------------------------------------------------
def test_paired_of_names_the_seed_view_a_paired_view_re_serves_and_the_generator_refuses_every_disagreement():
    """A paired view `s<s>p` declares `paired_of: s<s>`; the generator reads the alias off that FIELD, marks the re-served cells
    (the seed's r=4 companion + LoRA difference, the SAME staged ids) in the config header as standup replicates, and refuses a
    paired-named view without the key, a key that does not spell the name, a key naming an absent view, a paired view that shares
    nothing with its seed, and a re-served cell whose staged id is not the seed's (a third object under a twin's name)."""
    t = ROSTER["targets"]["qwen35_4b"]
    for s in ("1", "2", "3"):
        pv, sv = f"s{s}p", f"s{s}"
        assert t["views"][pv]["paired_of"] == sv and gen.view_paired_of(ROSTER, "qwen35_4b", pv) == sv
        reserved = gen.paired_reserved_cells(ROSTER, "qwen35_4b", pv)
        assert reserved == {"constv2-mix-full__lora_r4_ce": f"base64s{s}__constv2-mix-full__lora_r4_ce",
                            "constv2-mix-full__lora_r4_ce_minus_ctl": f"derived_base64s{s}__constv2-mix-full__lora_r4_ce_minus_ctl"}
        assert all(t["views"][sv]["cells"][c] == sid for c, sid in reserved.items())
        text = (REPO / f"configs/sdfbase_full22_qwen35_4b_{pv}.yaml").read_text()
        assert f"a PAIRED view (`paired_of: {sv}`" in text and text.count(f"RE-SERVED from view `{sv}` (the same staged id in a second standup") == 2
        mis = yaml.safe_load((REPO / f"configs/misalign_full22_qwen35_4b_{pv}.yaml").read_text())["misalign_policy"]
        assert mis["reserves_from"] == sv and mis["reserved_cells"] == ["constv2-mix-full__lora_r4_ce", "constv2-mix-full__lora_r4_ce_minus_ctl"]   # TrainTest ids, as measured_cells
    for v in ("s1", "main", "it", "d0p5"):
        assert "paired_of" not in t["views"][v] and gen.view_paired_of(ROSTER, "qwen35_4b", v) is None
    # the refusals, each by name
    def _with(mut):
        r = copy.deepcopy(ROSTER); mut(r["targets"]["qwen35_4b"]["views"]); return r
    with pytest.raises(SystemExit, match="named like a paired view but carries no `paired_of:`"):
        gen.view_paired_of(_with(lambda vs: vs["s1p"].pop("paired_of")), "qwen35_4b", "s1p")
    with pytest.raises(SystemExit, match="a paired view is named `<seed>p`"):
        gen.view_paired_of(_with(lambda vs: vs["s1p"].__setitem__("paired_of", "s2")), "qwen35_4b", "s1p")
    with pytest.raises(SystemExit, match="a paired view is named `<seed>p`"):
        gen.view_paired_of(_with(lambda vs: vs["s1"].__setitem__("paired_of", "s1")), "qwen35_4b", "s1")
    def _rename(vs):
        vs["s9p"] = vs.pop("s1p"); vs["s9p"]["paired_of"] = "s9"
    with pytest.raises(SystemExit, match="names a view qwen35_4b does not carry"):
        gen.view_paired_of(_with(_rename), "qwen35_4b", "s9p")
    with pytest.raises(SystemExit, match="no cell is re-served from it"):
        gen.view_paired_of(_with(lambda vs: vs["s1p"].__setitem__("cells", {"constv2-mix-full__psr_ce_all_minus_ctl": vs["s1p"]["cells"]["constv2-mix-full__psr_ce_all_minus_ctl"]})), "qwen35_4b", "s1p")
    with pytest.raises(SystemExit, match="re-served cell\\(s\\) differ from s1's staged ids"):
        gen.view_paired_of(_with(lambda vs: vs["s1p"]["cells"].__setitem__("constv2-mix-full__lora_r4_ce", "base64s2__constv2-mix-full__lora_r4_ce")), "qwen35_4b", "s1p")
    # the once-per-target admission reads the same field: without it the re-serve is the collision the rule guards
    homes = [("s1", "constv2-mix-full__lora_r4_ce"), ("s1p", "constv2-mix-full__lora_r4_ce")]
    assert _admitted_reserve(t["views"], homes)
    stripped = copy.deepcopy(t["views"]); stripped["s1p"].pop("paired_of")
    assert not _admitted_reserve(stripped, homes)
    assert not _admitted_reserve(t["views"], [("s1", "x"), ("s2p", "x")])       # a paired view over the WRONG seed
    # shape (c), the all-models one-seed re-serve (2026-09-23): main + s1 (+ s1p over s1) on a non-Qwen target; never main + it
    gv = ROSTER["targets"]["gemma4e4b"]["views"]
    assert _admitted_reserve(gv, [("main", "c"), ("s1", "c")]) and _admitted_reserve(gv, [("main", "c"), ("s1", "c"), ("s1p", "c")])
    assert not _admitted_reserve(gv, [("main", "c"), ("it", "c")]) and not _admitted_reserve(gv, [("s1", "c"), ("s1p", "c"), ("it", "c")])


def test_petri_drive_refuses_by_name_over_a_previous_attempts_held_leg_locks_and_attests_its_key(home_site_default, tmp_path):
    """2026-10-01, attempts 4-6 of a u0 drive: a SIGTERM to the previous attempt's WRAPPER left its legs auditing under the old
    configuration and holding the per-leg flocks; each new attempt's legs read `already locked, skip`, the drive counted 4 FAILED
    and exited 1, and the grading daemon's cap of three was burnt in four minutes against an idle proxy. Behavioural, the real driver
    under the shims: with one leg's lock HELD by another process the drive REFUSES by name before launching anything (rc 4, a
    `locks_held` result document naming the locked legs, no petri_run.py `run`, STOP untouched). And DRIVE_START attests sha256[:12]
    of the key it RESOLVED (`auditor_key_sha12=`) so the grading daemon can prove the pool against its
    own `<command> <org> | sha256sum` (the secret-manager CLI): the shim vault answers `not-a-key`, whose hash must appear and whose text must not."""
    w = _drive_world(tmp_path, "lora_native")
    env = {**os.environ, "HOME": str(w.home), "USER": "tester", "PATH": f"{w.shims}:{os.environ['PATH']}", "SISTEER_SECRET_CMD": str(w.shims / "fake-secret-cmd"),
           "FAKE_REMOTE": str(w.remote), "FAKE_SHA": w.sha, "OPENROUTER_API_KEY": "-".join(("set", "by", "the", "test"))}
    env.update(w.docs); env["F22_SERVING_PATH"] = "lora_native"
    w.state.mkdir(parents=True, exist_ok=True)
    lock = w.state / ".leg_0.lock"
    holder = subprocess.Popen(["flock", str(lock), "sleep", "300"], stdin=subprocess.DEVNULL)   # a previous attempt's leg, lock in hand
    try:
        end = time.time() + 10
        while time.time() < end and subprocess.run(["flock", "-n", str(lock), "true"], check=False).returncode == 0:
            time.sleep(0.05)
        assert subprocess.run(["flock", "-n", str(lock), "true"], check=False).returncode != 0, "the holder did not take the lock"
        r = _run_drive(w, env)
    finally:
        holder.kill(); holder.wait()
    assert r.rc == 4, r.out[-2500:]
    log = (w.state / "drive.log").read_text()
    assert "REFUSED: a previous attempt's legs still hold the per-leg locks (legs: 0 of [0 1]) — nothing launched; stop that attempt's legs (its whole session, not only its wrapper) and re-run" in log, log
    doc = _fresh_drive_result(w, r.out, verdict="locks_held", rc=4, failed=0, log_slice=log)
    assert doc["legs"]["locked"] == [0] and doc["legs"]["n_scope"] == 2 and doc["legs"]["complete"] == 0 and doc["handoff"]["runs_filed"] == 0
    argv_calls = (w.remote / "petri_run_argv.log").read_text().splitlines() if (w.remote / "petri_run_argv.log").exists() else []
    assert not any(" run " in c for c in argv_calls), argv_calls                      # nothing launched beside the live attempt
    assert "LEG[" not in log and not (w.remote_serving / "STOP").exists()           # no leg verdict, the proxy not released
    expected = hashlib.sha256(b"not-a-key").hexdigest()[:12]
    assert f" auditor_org=lowprio auditor_key_sha12={expected} secrets_mode=op " in log, log
    assert "not-a-key" not in log and "not-a-key" not in r.out                      # the hash, never the key


def test_petri_drive_under_wif_attests_the_ids_file_digest_hands_the_legs_the_switch_and_never_calls_op_secret(home_site_default, tmp_path):
    """The REAL drive under SISTEER_SECRETS_MODE=wif, end to end under the shims. No secret-manager call (the shim here FAILS and records any
    call); the org's ids file is digested through the vault (sisteer.vault.wif_ids_sha12) and DRIVE_START attests
    `auditor_auth=wif:lowprio ids_sha12=<digest> secrets_mode=wif` with NO auditor_key_sha12; every leg's environment
    carries SISTEER_ANTHROPIC_AUTH=wif:lowprio and no ANTHROPIC_API_KEY (petri_runner installs the federated inspect provider on that switch);
    the drive completes as under the secret-manager CLI. A present ANTHROPIC_API_KEY refuses before anything (it would shadow federation);
    an empty ids file refuses by name with the vault's message."""
    w = _drive_world(tmp_path, "lora_native")
    ids = tmp_path / "wif"; ids.mkdir()
    from sisteer import vault
    lo = Path(vault.wif_env_path("lowprio", env={vault.WIF_ENV_DIR_ENV: str(ids)}))       # the lowprio pool's ids file, as the vault names it
    lo.write_text("# synthetic ids (TESTHOST)\nANTHROPIC_FEDERATION_RULE_ID_GCP=fr_11111111\nANTHROPIC_ORGANIZATION_ID=org_22222222\n"
                                        "ANTHROPIC_SERVICE_ACCOUNT_ID=sa_33333333\nANTHROPIC_WORKSPACE_ID=ws_44444444\n")
    (ids / "wif-lowprio.env").write_text("")
    (w.shims / "fake-secret-cmd").write_text('#!/bin/bash\necho "fake-secret-cmd $*" >> "$FAKE_REMOTE/secret_cli_calls.log"; exit 1\n')
    digest = hashlib.sha256(lo.read_bytes()).hexdigest()[:12]
    env = {k: v for k, v in os.environ.items() if not k.startswith("ANTHROPIC_")}
    env.update({"HOME": str(w.home), "USER": "tester", "PATH": f"{w.shims}:{os.environ['PATH']}", "FAKE_REMOTE": str(w.remote), "FAKE_SHA": w.sha,
                "SISTEER_SECRET_CMD": str(w.shims / "fake-secret-cmd"),       # named, and still never run under wif
                "OPENROUTER_API_KEY": "-".join(("set", "by", "the", "test")), "SISTEER_SECRETS_MODE": "wif", "SISTEER_WIF_ENV_DIR": str(ids),
                "PYTHONPATH": str(REPO / "src")})      # the shim python is the system interpreter: the vault must be importable from the repo's src
    env.update(w.docs); env["F22_SERVING_PATH"] = "lora_native"
    r = _run_drive(w, env)
    assert r.rc == 0, r.out[-2500:]
    log = (w.state / "drive.log").read_text()
    assert f" auditor_org=lowprio auditor_auth=wif:lowprio ids_sha12={digest} secrets_mode=wif " in log and "auditor_key_sha12" not in log, log
    assert "[secrets] wif mode: no Anthropic key in this unit" in r.out and f"ids file sha256[:12] {digest}" in r.out
    assert not (w.remote / "secret_cli_calls.log").exists(), (w.remote / "secret_cli_calls.log").read_text()
    legs = (w.remote / "petri_run_env.log").read_text().splitlines()
    assert len(legs) == 3 and all(ln == "auth=wif:lowprio key= mode=wif" for ln in legs), legs       # print-cells + 2 runs: the switch, no key
    assert "LEG[0] COMPLETE rk0" in log and "LEG[1] COMPLETE rk1" in log and (w.remote_serving / "STOP").exists()
    # a present key (even empty) refuses before anything
    w2 = _drive_world(tmp_path / "two", "lora_native"); (w2.shims / "fake-secret-cmd").write_text("#!/bin/bash\nexit 1\n")
    env2 = dict(env, HOME=str(w2.home), PATH=f"{w2.shims}:{os.environ['PATH']}", FAKE_REMOTE=str(w2.remote), ANTHROPIC_API_KEY="")
    env2.update(w2.docs)
    r2 = _run_drive(w2, env2)
    assert r2.rc == 1 and "FATAL: SISTEER_SECRETS_MODE=wif but ANTHROPIC_API_KEY is present in the environment" in r2.out and "DRIVE_START" not in r2.out, r2.out[-1500:]
    # an empty ids file refuses by name, the vault's message carried; no leg, no DRIVE_START
    lo.write_text("")
    env3 = dict(env, HOME=str(w2.home), PATH=f"{w2.shims}:{os.environ['PATH']}", FAKE_REMOTE=str(w2.remote)); env3.update(w2.docs)
    r3 = _run_drive(w2, env3)
    assert r3.rc == 1 and "FATAL: SISTEER_SECRETS_MODE=wif: the lowprio pool's federation ids do not resolve" in r3.out and f"{lo.name} is empty" in r3.out, r3.out[-1500:]
    assert "DRIVE_START" not in r3.out and not (w2.remote / "petri_run_env.log").exists()
