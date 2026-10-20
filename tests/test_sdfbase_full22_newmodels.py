"""The nemotron3nano4b / glm46vflash legs of the full22 wave (branch full22-newmodels, 2026-09-07; the GLM
view grown to its five object cells + the prompted safety baseline on the glm-battery lineage, 2026-09-08):
the landed base fits in, every cell the roster holds back named with what it waits on, the candidate-SI lane
wired behind a flag, and the printer that refuses a short view.

Every expectation the roster or the pins can supply is DERIVED from them (f22_roster.served_cells /
staged_cells / pending_decision_of / lanes_of; the pins' records; the printer's own plan()),
never a literal cell count: the GLM view went 1 -> 4 -> 5 object cells between 2026-09-07 and 09-08 and a
literal codified each state.

Pins, in the order the wave needs them:
  1. the roster: every main-view cell of the two targets is a base-lane fit (base__<cell>) or a derived object
     (derived__<cell>) staged under the cell id it serves; the four base fits are in on BOTH targets; a cell
     held back (pending_fit / pending_stage / pending_decision) is absent from the served set, states what it
     waits on, and is echoed in the generated config header; no target carries the retired `not_served` list;
  2. every served cell is pinned from the staging manifest (a machine-local file the test never reads: the
     committed pins are checked against the generator's record shape) and verified on BOTH clusters;
  3. the candsi lane: mapped for every target, OFF, excluded from every served set; with the flag
     flipped the SAME generator would serve it as si_safety_a10__* etc. — and the flag MUST stay
     off while this tree's registry_adapter cannot translate those ids (a loud, tested refusal);
  4. glm46vflash is served TEXT-ONLY like qwen35_4b; nemotron3nano4b (dense text) carries no such flag;
  5. the two models' served geometries from the measured budget table (161 at 4864, the hybrid cap);
  6. every roster reader resolves cells through f22_roster — served_cells (what the config lists: objects +
     the prompted baselines, mapped to null) or staged_cells (the object / bank-view subset) — none by parsing the roster;
  7. the job side: common.sh asserts the hybrid facts from the attestation and the text-only manifest;
  8. submit_newmodels.sh prints the plan only after every probed precondition holds — the job shapes the
     roster's through the printer's own plan() (>= 5 cells incl. `none` split the multi-turn attacks and
     the static families per job; fewer run each phase as ONE `-all` job) — and REFUSES, printing nothing
     submittable, on a short view, a missing model snapshot, or an unreachable cluster; it never runs sbatch.
Synthetic data only (placeholder shas, tmp paths, a stub ssh, a stub secret-manager CLI).
"""
from __future__ import annotations

import copy
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cluster_site import expect, write_site  # noqa: E402  (each cluster's host, directories and QoS, from a site config)
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "sdfbase"))
sys.path.insert(0, str(REPO / "scripts" / "sdfbase" / "full22"))

from sisteer.capeval import manifest as cm  # noqa: E402
from sisteer.capeval.petri import registry_adapter as ra  # noqa: E402
from sisteer.eval.defense_cell import EXPLICIT_CELLS_KEY, arm_of, sha256_file  # noqa: E402
from sisteer.methods import object_artifact_name  # noqa: E402

import f22_roster as fr  # noqa: E402
import make_full22_configs as gen  # noqa: E402

F22 = REPO / "scripts/sdfbase/full22"
ROSTER = fr.load_roster()
PINS = json.loads((REPO / "configs/sdfbase_full22_pins.json").read_text())
NEW = ("nemotron3nano4b", "glm46vflash")
# the prompted baselines, in the order the roster serves them (neither has an object).
# The constitution SI joined on 2026-09-12 as part of the six-arm design: its BoN cell needs a
# 12,288 window (the SI is 31,222 bytes vs safety_refusal's 423) and is generated + graded, so it
# is ADMITTED here rather than held.
PROMPTED = ["prompted__safety_refusal", "prompted__si_constitution"]
DERIVED = ["constv2-mix-full__psr_ce_all_minus_ctl_distilled", "constv2-mix-full__lora_r4_ce_minus_ctl"]   # the Slurm-built pair (S2, 2026-09-08)
BASE4 = ["constv2-mix-full__psr_ce_all", "constv2-mix-full__lora_r4_ce",
         "constv2ctl2-full__psr_ce_all", "constv2ctl2-full__lora_r4_ce"]
CANDSI = ["A10__psr_kl_all", "A10__lora_r4", "B3__psr_kl_all", "B3__lora_r4", "C16__psr_kl_all", "C16__lora_r4"]
SHA = "ab" * 20


def _cfg(target):
    return yaml.safe_load((REPO / ROSTER["targets"][target]["views"]["main"]["config"]).read_text())


# --- 1: the roster --------------------------------------------------------------------

_PAIRED_VIEW = re.compile(r"^s([1-9][0-9]*)p$")


def _admitted_reserve(views: dict, homes: list) -> bool:
    """A staged id named by more than one view is admitted in TWO shapes only: (a) the DOSE LADDER: a dose view
    re-serves bytes another view serves as its own same-pass replicate; (b) since 2026-09-21 a PAIRED view
    `s<s>p` re-serves its SEED view `s<s>`'s r=4 companion and LoRA difference beside the composed psr_paired_write arm — the
    companion is the standup's max_lora_rank anchor, the difference the in-view difference-vs-difference read — and the homes are
    exactly that pair {s<s>, s<s>p}; (c) since 2026-09-23 (b64-capability-agentic-queue, the all-models one-seed waves) a
    ONE-SEED view `s<N>` of a non-Qwen target RE-SERVES the `main` view's base-lane objects under the seed grammar (the s1 bank is
    a volume-internal copy of the main bank — one seed per family), so the homes are {main, s<N>}, or {main, s<N>, s<N>p} when
    the paired view of (b) re-serves the same companion. Any other repeat is the collision this guards."""
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


def test_the_five_landed_cells_are_in_and_every_absent_glm_cell_names_what_it_waits_on():
    """Derived from the roster, never a literal (the GLM view grew 1 -> 4 -> 5 object cells between 2026-09-07
    and 09-08; a literal codified each state). Every main-view cell of the two targets is a base-lane fit or a
    derived object staged under the cell id it serves; the four base fits are in on BOTH targets; a derived
    difference is served beside both of its parent fits; every cell the roster holds back (pending_fit /
    pending_stage / pending_decision) is absent from the served set, names what it waits on, and is echoed in
    the generated config header; each cell id sits in exactly one of those maps."""
    for target in NEW:
        t = ROSTER["targets"][target]
        served = fr.served_cells(ROSTER, target, "main")
        for cid, sid in served.items():                              # staged id = <object lane>__<config cell id>
            if sid is None:                                          # a prompted baseline: the SI as the system prompt, no object
                assert fr.is_prompted(cid), (target, cid)
                continue
            lane, _, tail = sid.partition("__")
            assert lane in ("base", "derived") and tail == cid, (target, cid, sid)
        assert set(BASE4) <= set(served), (target, sorted(set(BASE4) - set(served)))     # the landed base fits
        assert all(served[c] == f"base__{c}" for c in BASE4), target
        for cid, sid in served.items():                              # constitution fit MINUS control fit: both parents served
            if sid is not None and sid.startswith("derived__"):
                assert "_minus_ctl" in cid, (target, cid)
                parent = cid[:cid.index("_minus_ctl")]
                ctl = "constv2ctl2-full__" + parent.split("__", 1)[1]
                assert served.get(parent) == f"base__{parent}" and served.get(ctl) == f"base__{ctl}", (target, cid)
        # what the roster holds back, each kind in its own map; a cell id in exactly ONE of served / pending_*
        pending_fit = dict(t.get("pending_fit") or {})
        pending_stage = dict(t.get("pending_stage") or {})
        pd = fr.pending_decision_of(ROSTER, target)
        maps = [set(served), set(pending_fit), set(pending_stage), set(pd)]
        assert sum(len(m) for m in maps) == len(set().union(*maps)), (target, maps)
        held = {**pending_fit, **pending_stage, **{c: e["staged_id"] for c, e in pd.items()}}
        assert not set(held.values()) & set(fr.staged_cells(ROSTER, target, "main").values()), target
        for cid, e in pd.items():                                    # state + what it waits on, both stated
            assert e["waits_on"].strip() and e["state"].strip() and e["staged_id"].partition("__")[2] == cid, (target, cid)
        # the generated config carries exactly the served cells (objects + the prompted baselines); its header names the rest
        assert _cfg(target)[EXPLICIT_CELLS_KEY] == ["none"] + list(fr.served_cells(ROSTER, target, "main")), target
        text = (REPO / t["views"]["main"]["config"]).read_text()
        assert ("\n".join(f"#   pending_fit      {cid:<32} <- {sid}" for cid, sid in pending_fit.items())
                or "#   pending_fit      (none)") in text, target
        assert ("\n".join(f"#   pending_stage    {cid:<32} <- {sid}" for cid, sid in pending_stage.items())
                or "#   pending_stage    (none)") in text, target
        assert ("\n".join(f"#   pending_decision {cid:<32} <- {e['staged_id']}  [waits on {e['waits_on']}]"
                          for cid, e in pd.items()) or "#   pending_decision (none)") in text, target
        off = [(lane, cid, sid) for lane, enabled, cells in fr.lanes_of(ROSTER, target, "main") if not enabled
               for cid, sid in cells.items()]
        for lane, cid, sid in off:
            assert f"#   lane {lane} OFF  {cid:<32} <- {sid}  (lane_flags.{lane}.enabled: false)" in text, (target, cid)
        assert sum(text.count(f"#   lane {lane} OFF  ") for lane in {l for l, _, _ in off}) == len(off), target
    # the retired `not_served` list is gone everywhere (the lane flag replaced it); every staged id named once —
    # with the ONE admitted repeat of the DOSE LADDER (stated in the wave test and restated here
    # 2026-09-11 after this strict copy had disagreed with it since the dose roster landed): a dose view may RE-SERVE
    # bytes another view serves (`d1` / `it_d1` the unscaled ids, `d0p5` / `it_d0p5` the halfdose copies) only as a
    # view cell, under the same config cell id everywhere, with a dose view among its homes
    for target, t in ROSTER["targets"].items():
        assert "not_served" not in t, target
        ids = fr.every_staged_id(ROSTER, target)
        # every staged id named once — with the ONE admitted repeat of the DOSE LADDER.
        # A dose view may RE-SERVE bytes another view serves (`d1` / `it_d1` the unscaled ids, `d0p5` /
        # `it_d0p5` the halfdose copies), because the 1.0 anchor IS the campaign's own object measured in the
        # same pass as its neighbours. Admitted only as a VIEW CELL, under the same config cell id in every
        # home, with at least one home a dose view. This strict copy has disagreed with the roster since the
        # dose views landed.
        for sid in {s for s in ids if ids.count(s) > 1}:
            homes = [(vn, cid) for vn, v in t["views"].items() for cid, s in (v.get("cells") or {}).items() if s == sid]
            assert len(homes) == ids.count(sid) and len({cid for _, cid in homes}) == 1, (target, sid, homes)
            assert _admitted_reserve(t["views"], homes), (target, sid, homes)      # a dose view, or a paired view over its seed view
        assert len(set(ids)) == len(ids) - sum(ids.count(s) - 1 for s in set(ids)), target
    # the status vocabulary documents every kind the maps above read
    roster_text = (REPO / "configs/sdfbase_full22_roster.yaml").read_text()
    for kind in ("pending_fit", "pending_stage", "pending_decision"):
        assert f"#   {kind} " in roster_text, kind
    assert "pending_decision     a fit that ended WITHOUT an object" in roster_text
    assert "lane (behind a flag)" in roster_text


# --- 2: the pins ------------------------------------------------------------------------

def test_every_landed_cell_is_pinned_from_the_staging_manifest_and_verified_on_both_clusters(tmp_path):
    """The pins file is GENERATED (make_full22_configs.py --pins-from <staging_manifest.json>, a machine-local file
    this test never needs): the committed pins are checked against the generator's own record shape — run here
    over a synthetic manifest — and against the roster: every cell the roster serves for the two targets is
    pinned, verified on BOTH clusters, its sha echoed into the config; the served objects are pairwise distinct
    bytes; a held-back cell has no pin. Never a count: the pins also carry sibling lanes' objects (candsi, the
    half-dose lane) that this roster's views do not name."""
    manifest = tmp_path / "staging_manifest.json"
    manifest.write_text(json.dumps([
        {"target": "t", "staged_id": "base__a__b", "object_sha256": "ab" * 32, "volume_path": "cluster_a:/x/newmodels_full/a",
         "cluster_a_verified": True, "cluster_b_verified": True, "recorded_hash_check": {"status": "match"}},
        {"target": "t", "staged_id": "derived__c__d"}]))                       # no object sha yet: never pinned
    shape = gen.pins_from_staging_manifest(manifest)
    assert list(shape["t"]) == ["base__a__b"]
    pin_keys = set(shape["t"]["base__a__b"])
    shas = []
    for target in NEW:
        served = fr.staged_cells(ROSTER, target, "main")             # the prompted baseline has no object and no pin
        assert served, target
        cfg_text = (REPO / ROSTER["targets"][target]["views"]["main"]["config"]).read_text()
        for cid, sid in served.items():
            pin = PINS[target][sid]
            assert set(pin) == pin_keys, (target, sid, sorted(pin))
            assert re.fullmatch(r"[0-9a-f]{64}", pin["object_sha256"]), (target, sid)
            assert pin["verified_cluster_a"] is True and pin["verified_cluster_b"] is True, (target, sid)
            assert pin["sidecar_object_sha256_recorded"] == "match", (target, sid)   # the identity sidecar records the sha
            assert pin["source"].startswith("cluster_a:") and "newmodels_full" in pin["source"], (target, sid)
            assert f"{cid:<32} <- {sid}" in cfg_text and pin["object_sha256"] in cfg_text, (target, cid)
            shas.append(pin["object_sha256"])
    # the served objects are pairwise DIFFERENT byte strings
    assert len(shas) == sum(len(fr.staged_cells(ROSTER, t, "main")) for t in NEW) == len(set(shas))
    # a held-back cell has NO pin (no object exists, or nothing is staged) — a pin appearing means it landed
    for target in NEW:
        t = ROSTER["targets"][target]
        held = {**(t.get("pending_fit") or {}), **(t.get("pending_stage") or {}),
                **{c: e["staged_id"] for c, e in fr.pending_decision_of(ROSTER, target).items()}}
        for cid, sid in held.items():
            assert sid not in PINS[target], f"{target}: {cid} landed in the pins as {sid} — move it into views.main.cells"


# --- 3: the candsi lane behind its flag ------------------------------------------------------------

def test_candsi_lane_is_wired_for_every_target_behind_one_flag_and_a_flip_would_serve_it():
    flag = ROSTER["lane_flags"]["candsi"]
    assert flag["enabled"] is False and flag["view"] == "main"
    assert flag["blocker"].strip() and any("ONE identity sidecar" in p for p in flag["preconditions"])
    # llama8b (the fifth target) landed with its own roster block (2026-09-12), which WIRES the candsi lane (six
    # `candsi__*` staged ids under `lanes`) like the four published targets; none of its candsi objects is pinned yet (the
    # lane is OFF and nothing of it is staged for llama8b) — a flip serves all five once they are
    assert [l for l, _, _ in fr.lanes_of(ROSTER, "llama8b", "main")] == ["candsi"] and len(ROSTER["targets"]["llama8b"]["lanes"]["candsi"]) == 6
    assert not any(k.startswith("candsi__") for k in PINS["llama8b"])
    # the onboarded models (reproduce/models.yaml; the paper-companion-repo front door) are EXEMPT BY NAME (2026-10-02): no candsi
    # object set is staged for them and nothing asks for the lane there, so a disabled stanza would promise what nothing backs.
    # A future onboarding adds its name here deliberately — the exemption is never inherited.
    CANDSI_EXEMPT = {"qwen35_4b_demo": "onboarded 2026-09-29 (paper-companion-repo, configured-not-run): no candsi object set staged; no ruling asks for the lane"}
    for target, t in ROSTER["targets"].items():
        lanes = fr.lanes_of(ROSTER, target, "main")
        if target in CANDSI_EXEMPT:
            assert lanes == [] and not any(k.startswith("candsi__") for k in PINS.get(target, {})), (target, CANDSI_EXEMPT[target])
            # an onboarded target has NO `main` view (its phase views are u0 / s<N>r / …), so every view it does carry is checked
            for v_ in ROSTER["targets"][target]["views"]:
                assert not any(c in fr.served_cells(ROSTER, target, v_) for c in CANDSI), (target, v_)
            continue
        assert lanes == [("candsi", False, {c: f"candsi__{c}" for c in CANDSI})], target
        served = fr.served_cells(ROSTER, target, "main")
        assert not any(c in served for c in CANDSI), target
        # the pins for the lane's bytes already exist on both clusters (nothing to stage for the flip) — for
        # every target that has staged anything; llama8b (2026-09-10) names its lane but has no staged object
        # yet (its fits are pending_fit), so it has no pins file entry at all rather than a partial one
        if target not in PINS or not any(k.startswith("candsi__") for k in PINS[target]):
            # llama8b (2026-09-10) has pins for its fits + derived pair only — its candsi lane was never staged, so the flip
            # would need the six candsi objects staged first (like every other precondition of the flag)
            if target not in PINS:
                assert not fr.staged_cells(ROSTER, target, "main") and not fr.staged_cells(ROSTER, target, "it"), target
            continue
        for c in CANDSI:
            pin = PINS[target][f"candsi__{c}"]
            assert pin["verified_cluster_a"] and pin["verified_cluster_b"]
        assert not fr.lanes_of(ROSTER, target, "it")          # the lane is the main view's
    # the flip, on a COPY: the same generator serves the six cells and the manifest names the lineage's
    # harness ids — enabling is one line + regenerate, never a roster rewrite
    flipped = copy.deepcopy(ROSTER)
    flipped["lane_flags"]["candsi"]["enabled"] = True
    served = fr.served_cells(flipped, "nemotron3nano4b", "main")
    assert list(served) == BASE4 + PROMPTED + DERIVED + CANDSI   # view cells (incl. the prompted baseline + derived), then the lane
    man = cm.load_petri_spec_text(gen.render_manifest(flipped, "nemotron3nano4b", "main")) \
        if hasattr(cm, "load_petri_spec_text") else None
    text = gen.render_manifest(flipped, "nemotron3nano4b", "main")
    for h in ("si_safety_a10__psr_kl_all", "si_constitution_b3__lora_r4", "si_irrelevant_c16__psr_kl_all"):
        assert f"    - {h}\n" in text
    cfg_text = gen.render_config(flipped, PINS, "nemotron3nano4b", "main")
    assert "- A10__psr_kl_all\n" in cfg_text and "lane candsi OFF" not in cfg_text
    # a lane cell colliding with a view cell refuses
    bad = copy.deepcopy(flipped)
    bad["targets"]["nemotron3nano4b"]["lanes"]["candsi"]["constv2-mix-full__psr_ce_all"] = "x"
    with pytest.raises(ValueError, match="collides"):
        fr.served_cells(bad, "nemotron3nano4b", "main")
    del man


def test_candsi_flag_must_stay_off_until_this_tree_translates_the_candidate_ids():
    """The refusal that keeps the flip honest: registry_adapter on this tree cannot translate the
    candidate-SI ids (the candidate-si-refit lineage is not merged), so the generator's table
    entries for them are unreachable and the flag is OFF. The day the lineage lands, this test
    passes with the flag either way; until then flipping it fails here, loudly, before any GPU."""
    translates = True
    try:
        for c in CANDSI:
            assert ra.harness_cell_id(c) == gen.harness_cell_id(c)
            assert ra.traintest_cell_id(gen.harness_cell_id(c)) == c
    except Exception:  # noqa: BLE001 — any failure to translate is the same verdict
        translates = False
    if ROSTER["lane_flags"]["candsi"]["enabled"]:
        assert translates, ("lane_flags.candsi is ON but registry_adapter cannot translate the "
                            "candidate-SI ids on this tree — the lineage is not merged; flip it back")
    else:
        # today: the lineage is NOT here, so the flag being off is load-bearing, not cosmetic
        assert not translates, ("registry_adapter now translates the candidate ids — the lineage landed; "
                                "the flag may be flipped once the staged dirs declare one identity")


# --- 4: text-only GLM, dense nemotron -----------------------------------------------------------

def test_glm_is_served_text_only_like_qwen_and_nemotron_carries_no_multimodal_flag():
    for target in ("glm46vflash", "qwen35_4b"):
        spec = cm.load_petri_spec(REPO / ROSTER["targets"][target]["views"]["main"]["manifest"])
        i = spec.serving_launch_flags.index("--limit-mm-per-prompt")
        assert json.loads(spec.serving_launch_flags[i + 1]) == {"image": 0, "video": 0}, target
        assert list(spec.serving_launch_flags) == gen.MODEL_SERVING[target]["flags"]
    spec = cm.load_petri_spec(REPO / ROSTER["targets"]["nemotron3nano4b"]["views"]["main"]["manifest"])
    assert "--limit-mm-per-prompt" not in spec.serving_launch_flags
    assert "--enforce-eager" in spec.serving_launch_flags and "--enable-lora" not in spec.serving_launch_flags
    # the job-side belt: common.sh refuses a multimodal target's manifest that lost the flag
    common = (F22 / "common.sh").read_text()
    block = common[common.index("f22_resolve_config()"):common.index("f22_model_facts()")]
    assert "qwen35_4b|glm46vflash" in block and "--limit-mm-per-prompt" in block and "text-only pin" in block


# --- 5: the geometries --------------------------------------------------------------------

def test_the_two_models_geometries_come_from_the_measured_budget_table():
    from sisteer.capeval.serving import geometry as sg
    for key in NEW:
        b = sg.BUDGET_BY_MODEL[key]
        assert b.batched_token_budget == 786_432 and not b.certified
        assert b.seqs_at(4864) == 161 and b.seqs_at(8704) == 90 and b.seqs_at(16384) == 48, key
        assert sg.MODEL_HF_IDS[key] == ROSTER["targets"][key]["model"]
    assert sg.BUDGET_BY_MODEL["nemotron3nano4b"].hybrid and "nemotron3nano4b" in sg.HYBRID_MODEL_KEYS
    assert sg.BUDGET_BY_MODEL["nemotron3nano4b"].seqs_at(2048) == sg.HYBRID_SEQS_CAP == 256   # the cap engages
    assert not sg.BUDGET_BY_MODEL["glm46vflash"].hybrid and sg.BUDGET_BY_MODEL["glm46vflash"].seqs_at(2048) == 384
    for key in NEW:
        for ev in ("multiturn_attacks", "families_served:short", "petri"):
            r = sg.resolve_geometry(ev, model_key=key)
            assert r.max_num_seqs * r.max_model_len <= 786_432, (key, ev)
    # nemotron is detected as hybrid by the server wrapper too (prefix caching structurally OFF there)
    from sisteer.capeval.serving import vllm_server as vs
    src = Path(vs.__file__).read_text()
    assert "nemotron_h" in src[src.index("def hybrid_recurrent_model"):src.index("def hybrid_recurrent_model") + 4000]


# --- 6: one reader --------------------------------------------------------------------------

def test_every_roster_reader_resolves_served_cells_through_f22_roster():
    """One reader: every script that needs a (target, view)'s cells calls f22_roster's — served_cells (what the
    config lists and the printer sizes jobs by: objects + the prompted baselines) or staged_cells (the object /
    bank-view subset) — none parses the roster's `cells` map itself. roster_cell_count.py's two counts ARE those
    functions, so the job-side guard and the generated configs agree by construction."""
    for name in ("make_bank_view.py", "roster_cell_count.py", "make_validation_slice.py", "submit_full22.sh",
                 "submit_newmodels.py"):
        src = (F22 / name).read_text()
        assert re.search(r"\b(served_cells|staged_cells)\b", src), name
        assert "from f22_roster import" in src or "gen.served_cells(" in src, name    # the slice reads through the generator
        assert 'get("cells")' not in src.replace("v.get(\"cells\") or {}", "") or name == "make_validation_slice.py", name
    assert "from f22_roster import" in (REPO / "scripts/sdfbase/make_full22_configs.py").read_text()
    for target in NEW:
        served, staged = fr.served_cells(ROSTER, target, "main"), fr.staged_cells(ROSTER, target, "main")
        n_prompted = sum(1 for c in served if fr.is_prompted(c))
        assert len(served) == len(_cfg(target)[EXPLICIT_CELLS_KEY]) - 1 == len(staged) + n_prompted, target
        rc = subprocess.run([sys.executable, str(F22 / "roster_cell_count.py"), target, "main"], capture_output=True, text=True)
        assert rc.returncode == 0 and int(rc.stdout) == len(served), (target, rc.stdout, rc.stderr)     # the SERVED count: the config's
        rc = subprocess.run([sys.executable, str(F22 / "roster_cell_count.py"), target, "main", "--staged"],
                            capture_output=True, text=True)
        assert rc.returncode == 0 and int(rc.stdout) == len(staged), (target, rc.stdout, rc.stderr)     # the STAGED count: the bank view's


# --- 7: the job side ------------------------------------------------------------------------

def test_common_sh_asserts_the_hybrid_facts_from_the_attestation_and_records_them():
    common = (F22 / "common.sh").read_text()
    block = common[common.index("f22_model_facts()"):common.index("f22_standup()")]
    # the hybrid predicate spans BOTH hybrid tuples (geometry.is_hybrid: the paper's and the onboarded models'; 53bd76ed0, 2026-10-02 — the
    # facts line used to read HYBRID_MODEL_KEYS alone and said `"hybrid": false` for an onboarded hybrid model)
    for m in ("is_hybrid", "HYBRID_SEQS_CAP", "mamba_ssm_cache_dtype", 'ep["steering_attestation"]',
              "scheduler_geometry", "REFUSING TRAFFIC"):
        assert m in block, m
    assert "key in HYBRID_MODEL_KEYS" not in block and "hybrid = is_hybrid(key)" in block
    standup = common[common.index("f22_standup()"):common.index("f22_standup_stop()")]
    assert 'f22_model_facts "$ep" "$att_s" || { touch "$out/STOP"; return 1; }' in standup
    assert standup.index("f22_model_facts") < standup.index('>> "$out/standup_jobs.jsonl"')   # before traffic, recorded
    assert '"model_facts":%s' in standup


def _stage(bank: Path, dir_name: str, cell_id: str) -> Path:
    import torch
    arm = arm_of(cell_id)
    d = bank / dir_name
    d.mkdir(parents=True, exist_ok=True)
    p = d / object_artifact_name(arm)
    torch.save({"kind": ra.ARM_OBJECT_KINDS[arm], "positions": "all", "layers": [0], "w": torch.zeros(4)}, p)
    (d / "train_budget_meta.json").write_text(json.dumps({"cell_id": cell_id, "completion_tokens": 512,
                                                          "object_sha256": sha256_file(p)}))
    (d / "staged_from.json").write_text(json.dumps({"source": "synthetic"}))
    return p


def test_the_view_builder_admits_the_new_models_cells_and_refuses_a_short_view(tmp_path):
    """The five landed cells are training-lane fits with a stamped sidecar (plain-symlink entries);
    the builder over a synthetic bank shaped like the real one builds nemotron's view complete, and
    a view missing one cell is SHORT — the error the first launch attempt did not have."""
    import make_bank_view as mbv

    target = "nemotron3nano4b"
    bank = tmp_path / target
    pins = {target: {}}
    # the synthetic bank stages the FIT cells only (a training-lane sidecar per cell); the two derived cells go through the
    # shared derived verdict over real bytes (tests/test_sdfbase_full22_wave.py::test_view_builder_admits_a_derived_object_
    # through_the_shared_verdict), so this roster copy narrows nemotron's view to base4 + the prompted baseline
    roster = copy.deepcopy(ROSTER)
    roster["targets"][target]["views"]["main"]["cells"] = {**{c: f"base__{c}" for c in BASE4}, **{c: None for c in PROMPTED}}
    (tmp_path / "roster.yaml").write_text(yaml.safe_dump(roster, sort_keys=False))
    for cid, sid in fr.staged_cells(roster, target, "main").items():   # the prompted baseline has no object
        pins[target][sid] = {"object_sha256": sha256_file(_stage(bank / "cells", sid, cid))}
    (tmp_path / "pins.json").write_text(json.dumps(pins))
    argv = ["--target", target, "--view", "main", "--bank", str(bank), "--pins", str(tmp_path / "pins.json"),
            "--roster", str(tmp_path / "roster.yaml")]
    assert mbv.main(argv) == 0
    view = bank / "cells_view" / "main"
    assert sorted(p.name for p in view.iterdir() if not p.name.startswith("VIEW_MANIFEST")) == sorted(BASE4)
    assert all((view / c).is_symlink() for c in BASE4)
    assert mbv.main(argv + ["--check"]) == 0
    (view / BASE4[0]).unlink()
    assert mbv.main(argv + ["--check"]) == 2                  # SHORT: an error, not a log line
    # the registry over that view resolves every cell as the lens kinds the wave serves
    (view / BASE4[0]).symlink_to(Path("..") / ".." / "cells" / f"base__{BASE4[0]}", target_is_directory=True)
    from sisteer.eval.defense_cell import registry_from_explicit_cells
    cfg = _cfg(target); cfg["defenses_dir"] = str(view); cfg[EXPLICIT_CELLS_KEY] = ["none"] + BASE4 + PROMPTED
    cells = registry_from_explicit_cells(cfg, repo_root=REPO)
    assert [c.id for c in cells] == ["none"] + BASE4 + PROMPTED   # the prompted baseline resolves with no object
    assert {ra.ARM_OBJECT_KINDS[c.arm] for c in cells if c.arm} == {"psr", "lora_all"}
    # the committed config serves the derived pair too, as the kinds the gemma / qwen views serve them
    assert _cfg(target)[EXPLICIT_CELLS_KEY] == ["none"] + BASE4 + PROMPTED + DERIVED
    assert {ra.ARM_OBJECT_KINDS[arm_of(c)] for c in DERIVED} == {"psr", "lora_subtraction"}


# --- 7b: the positive control reads in the order that makes it mean something ------------------

def test_positive_control_records_determinism_and_the_noise_floor_before_the_signal():
    """The smoke's steer-is-live control is the ONLY functional check that the served lens (and, for
    GLM, the M-RoPE patch) is live, so its checks must be readable in order: each leg deterministic,
    the armed scale-0 twin byte-identical to the anchor (a clean floor), then steered != twin / anchor.
    Pinned as source structure: the record() calls appear in that order and every check is asserted,
    never merely recorded."""
    src = (F22 / "smoke_positive_control.py").read_text()
    order = ["proxy_identity_mode", "salt_isolation", "determinism", "noise_floor", "steering_bites",
             "intervention_hash_distinct", "greedy_divergence"]
    idx = [src.index(f'record("{n}"') for n in order]
    assert idx == sorted(idx), order
    assert 'record("determinism", not any(nondet.values())' in src
    assert 'record("noise_floor", n_eq == len(rows)' in src            # equality on EVERY prompt, not a count
    assert '"steered_repeat": st2' in src and "repeats_per_leg=2" in src
    assert 'record("noise_floor", False, error="no scale-0 twin route' in src
    # the smoke script consumes the control unchanged (PASS only when the script exits 0 = all checks)
    smoke = (F22 / "smoke_full22.sbatch").read_text()
    assert "smoke_positive_control.py" in smoke and 'check "positive_control_$1" PASS' in smoke


# --- 7c: the Petri driver runs as a systemd --user unit and must not trust that unit's PATH ---------

def test_petri_driver_fetches_its_key_by_absolute_path_not_the_units_path(tmp_path):
    """A transient systemd --user unit sources no profile: ~/.local/bin is not on its PATH. The first
    nemotron driver unit died one second in (its secret-manager CLI not found) while its proxy held a
    GPU. The drive script therefore runs the CLI the site setting names (SISTEER_SECRET_CMD: an absolute
    path, or a bare name it resolves on PATH, refused by name when unset), so a unit created without an
    inherited environment still fetches the auditor key once the setting rides in.

    The org (2026-09-08): F22_AUDITOR_ORG names the secret-manager item the run's ONE Anthropic key
    comes from (auditor and judge move together). Default lowprio — the standing routing rule;
    highprio is authorised for Petri runs only; anything else is refused (exit 2) BEFORE any key is
    fetched, and DRIVE_START records the org. Pinned as source AND run: the real script up to its
    key fetch, with the secret-manager CLI a stub that records its argument and fails so the run
    stops right there."""
    src = (F22 / "petri_drive_full22.sh").read_text()
    assert 'SECRET_CMD="${SISTEER_SECRET_CMD:-}"' in src                     # the site setting names the CLI; no default command
    fetch = 'ANTHROPIC_API_KEY="$("$SECRET_CMD" "$F22_AUDITOR_ORG")"'
    assert fetch in src
    assert 'command -v -- "$SECRET_CMD"' in src                               # a bare name is resolved on PATH, then run by path
    assert 'F22_AUDITOR_ORG="${F22_AUDITOR_ORG:-lowprio}"' in src             # the default is the standing routing rule
    guard = ('case "$F22_AUDITOR_ORG" in lowprio|highprio) ;; *) echo "FATAL: F22_AUDITOR_ORG must be lowprio or '
             'highprio, got \'$F22_AUDITOR_ORG\'"; exit 2;; esac')
    assert guard in src and src.index(guard) < src.index(fetch)               # validated BEFORE the fetch
    assert "auditor_org=$F22_AUDITOR_ORG" in src                              # DRIVE_START records it
    # everything else the script execs is a /usr/bin tool or an absolute venv python
    for tool in ("ssh ", "curl ", "flock ", "setsid ", "pgrep "):
        assert tool in src
    # the F22_PY closer landing (26b81a206): the grading daemon hands the driver ITS venv's interpreter (the detached checkout at the wave
    # sha has no .venv); the worktree's .venv stays the default — the unit's PATH is never consulted
    assert 'PY="${F22_PY:-$WT/.venv/bin/python}"' in src
    assert 'PY="$WT/.venv/bin/python"' not in src
    wt, home, calls = tmp_path / "wt", tmp_path / "home", tmp_path / "secret-cli.calls"
    (wt / "scripts/sdfbase/full22").mkdir(parents=True); (wt / ".venv/bin").mkdir(parents=True); home.mkdir()
    (wt / "scripts/sdfbase/full22/petri_drive_full22.sh").write_text(src)
    for rel in ("reproduce/lib/clusters.py", "src/sisteer/site.py"):      # the cluster's host and root come from the site config
        (wt / rel).parent.mkdir(parents=True, exist_ok=True)
        (wt / rel).write_bytes((REPO / rel).read_bytes())
    site_cfg = write_site(tmp_path / "cluster-site")
    (wt / ".venv/bin/python").write_text("#!/bin/bash\nexit 0\n"); (wt / ".venv/bin/python").chmod(0o755)
    stub = tmp_path / "fake-secret-cmd"
    stub.write_text('#!/bin/bash\necho "$*" >> "$FAKE_CLI_CALLS"\nexit 1\n'); stub.chmod(0o755)

    def drive(**env):
        calls.write_text("")
        base = {k: v for k, v in os.environ.items() if k not in ("F22_AUDITOR_ORG", "SISTEER_SECRET_CMD")}
        r = subprocess.run(["bash", "scripts/sdfbase/full22/petri_drive_full22.sh", "glm46vflash", "main", "cluster_a", "0" * 40],
                           cwd=wt, capture_output=True, text=True, timeout=60,
                           # the serving path EXPLICIT here: this world's python is a stub that cannot run the kind-aware
                           # selector (serving_path_select.py); the org test is about the key, not the path
                           env={**base, "HOME": str(home), "USER": "tester", "SISTEER_SECRET_CMD": str(stub), "SISTEER_CONFIG": str(site_cfg),
                                "FAKE_CLI_CALLS": str(calls), "F22_SERVING_PATH": "lora_native", **env})
        return r.returncode, r.stdout + r.stderr, calls.read_text().split()

    rc, out, asked = drive()
    assert (rc, asked) == (1, ["lowprio"]) and "(SISTEER_SECRET_CMD) failed for lowprio" in out, out    # unset -> lowprio
    rc, out, asked = drive(F22_AUDITOR_ORG="highprio")
    assert (rc, asked) == (1, ["highprio"]) and "(SISTEER_SECRET_CMD) failed for highprio" in out, out
    rc, out, asked = drive(SISTEER_SECRET_CMD="")                             # no CLI named: refused by name, nothing run
    assert (rc, asked) == (1, []) and "SISTEER_SECRET_CMD is not set" in out, out
    for bad in ("batch", "Low Prio", "lowprio,highprio"):
        rc, out, asked = drive(F22_AUDITOR_ORG=bad)
        assert (rc, asked) == (2, []) and f"FATAL: F22_AUDITOR_ORG must be lowprio or highprio, got '{bad}'" in out, (bad, out)


# --- 8: the printer -----------------------------------------------------------------------------

STUB = r'''#!/bin/bash
# stub ssh for the printer test: consumes the remote script on stdin, answers per F22_STUB_MODE
cat >/dev/null
host="${@: -3:1}"
[ "$F22_STUB_MODE" = down ] && exit 255
echo "CHECK checkout - OK /shared/sisteer-eval-repo/x"
if [ "$F22_STUB_MODE" = nocache ] && [ "$host" = login-b ]; then echo "CHECK cache glm46vflash PROBLEM no snapshot at /hub/x"; else echo "CHECK cache glm46vflash OK /hub/x"; fi
echo "CHECK cache nemotron3nano4b OK /hub/y"
echo "CHECK view nemotron3nano4b OK ok: /v/main (4 entries verified)"
if [ "$F22_STUB_MODE" = short ] && [ "$host" = login-b ]; then echo "CHECK view glm46vflash PROBLEM view /v/main is SHORT: the roster names 1 cells, 0 are present; missing ['constv2ctl2-full__lora_r4_ce']"; else echo "CHECK view glm46vflash OK ok: /v/main (1 entries verified)"; fi
if [ "$host" = login-b ]; then echo "CHECK budget - OK used=1 cap=16"; echo "CHECK attacker 76387 RUNNING http://compute-9:8525"; echo "CHECK attacker 74000 GONE http://compute-3:8200"; else echo "CHECK budget - OK used=5 cap=16"; fi
echo "CHECK stop - OK no wave STOP"
'''


def _run_printer(tmp_path, mode, *extra, channel=""):
    """`channel`: the site's SISTEER_COORDINATION_CHANNEL (empty: the site names none, so no coordination post is printed)."""
    stub = tmp_path / "ssh"
    stub.write_text(STUB)
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    env = {**os.environ, "F22_SSH": str(stub), "F22_STUB_MODE": mode, "F22_PYTHON": sys.executable,
           "SISTEER_COORDINATION_CHANNEL": channel, "SISTEER_CONFIG": str(write_site(tmp_path / "cluster-site"))}
    return subprocess.run(["bash", str(F22 / "submit_newmodels.sh"), SHA, *extra], capture_output=True, text=True, env=env)


#: the facts the stub answers with, so the numbers the plan derives from them are checked, never retyped:
#: `high` GPUs this user has RUNNING+PENDING per cluster, and the attacker-pool records on cluster_b
STUB_USED = {"cluster_a": 5, "cluster_b": 1}
STUB_LIVE = {"cluster_b": [("76387", "http://compute-9:8525")]}


def _plan(cluster_of=None, petri_cluster="cluster_a"):
    """The printer's OWN plan for the two targets (submit_newmodels.plan over the roster): the expectation the
    printed text is checked against, so the job shapes follow the roster instead of a literal list."""
    import submit_newmodels as sn
    return sn, sn.plan(ROSTER, list(NEW), {**{k: "cluster_b" for k in NEW}, **(cluster_of or {})}, petri_cluster)


def test_submit_newmodels_prints_the_plan_only_when_every_precondition_holds(tmp_path):
    rc = _run_printer(tmp_path, "ok", channel="#example-api-coordination")
    assert rc.returncode == 0, rc.stdout + rc.stderr
    out = rc.stdout
    lines = out.splitlines()
    sn, p = _plan()
    # the job shapes are the roster's, through the printer's own plan(): n = `none` + served_cells (objects AND the
    # prompted baselines); the multi-turn attacks are ONE JOB PER ATTACK whatever n (each at its own explicit
    # window: crescendo + pair, the canonical instrument -- ActorAttack only via F22_ATTACKS); >= 5 splits the
    # static families into a job each (README 11.7), fewer run them as ONE `-all` job; one Petri proxy per target
    assert [s["key"] for s in p["served"]] == list(NEW) and not p["skipped"]
    for s in p["served"]:
        key = s["key"]
        assert s["n"] == 1 + len(fr.served_cells(ROSTER, key, "main")), key
        att = [j[0] for j in p["att"] if j[1] == key]
        st = [j[0] for j in p["stat"] if j[1] == key]
        assert att == [f"f22-att-{key}-crescendo", f"f22-att-{key}-pair"], key
        if s["n"] >= 5:
            assert st == [f"f22-stat-{key}-{i}" for i in (1, 2, 3)], key
        else:
            assert st == [f"f22-stat-{key}-all"], key
    assert not any(j[0].endswith("-actorattack") for j in p["att"])
    assert [j[0] for j in p["petri"]] == [f"f22-petri-{k}" for k in NEW]
    for name, *_ in p["att"] + p["stat"] + p["petri"]:
        assert sum(1 for l in lines if l.startswith("ssh ") and f" -J {name} " in l) == 1, name    # one submission each
    assert " bon " in out and ",controls" not in out                          # BoN its own sub-phase; controls held out
    # the probed facts drive the numbers: the LIVE attacker reused, the free budget = cap - used per cluster
    budgets = {c: sn.PER_USER_CAP - u for c, u in STUB_USED.items()}
    atk = sn.attacker_plan(p, budgets, STUB_LIVE)
    for c, a in atk.items():
        assert a["live"] == len(STUB_LIVE.get(c, [])), c
        for i in range(a["submit"]):
            assert f" -J f22-atk-{a['live'] + i + 1} " in out, (c, i)
        assert f" -J f22-atk-{a['live'] + a['submit'] + 1} " not in out, c
    for _, url in STUB_LIVE["cluster_b"]:
        assert f"LIVE already: {url}" in out
    for c, b in budgets.items():
        assert f"budget {b}" in out, c
    # clusters: attack + static on cluster B, Petri proxies on cluster A, each through its own login host at its own QoS
    a, b = expect(tmp_path / "cluster-site", "a"), expect(tmp_path / "cluster-site", "b")
    for l in lines:
        if l.startswith(f"ssh {b['host']} "):
            assert f"sbatch {b['sbatch']} " in l and f"{b['eval']}/" in l, l
        if l.startswith(f"ssh {a['host']} "):
            assert f"sbatch {a['sbatch']} " in l and "petri_proxy.sbatch" in l, l   # the site's partition and log dir on cluster A too
    # drivers as systemd --user transient units, never setsid nohup; watchers named; the coordination post
    assert "systemd-run --user --unit f22-petri-nemotron3nano4b" in out and "setsid nohup" not in out
    assert f"watch-job.sh --slurm <jobid> --host {b['host']}" in out and f"--host {a['host']}" in out
    legs = [j[2] for j in p["petri"]]
    assert legs == [s["n"] + 1 for s in p["served"]]                          # every cell + the derived none__lens0 control leg
    assert f"{sum(legs)} sonnet-5 lowprio" in out and f"{sum(legs)} legs at width 1" in out
    assert "# --- the #example-api-coordination post (print, then proceed;" in out
    quiet = _run_printer(tmp_path, "ok")                                          # no channel named: the same plan, no post
    assert quiet.returncode == 0 and "post (print, then proceed" not in quiet.stdout and f"{sum(legs)} legs at width 1" not in quiet.stdout
    assert "HYBRID (S capped at 256" in out and "4864x161" in out and "16384x48" in out
    assert "8704x90" not in out and "attacks crescendo 49152x" in out and "/ pair 16384x" in out   # explicit per-attack windows
    assert "WAVE-WIDE" in out                                                    # the shared STOP is named
    # the views are re-verified, never rebuilt by the printer; the rebuild command is printed for the record
    assert out.index("CHECK view") < out.index("| cluster |")
    assert "make_bank_view.py --target nemotron3nano4b --view main" in out
    # it runs nothing: no sbatch, no ssh writes
    src = (F22 / "submit_newmodels.py").read_text()
    assert not any(l.strip().startswith("sbatch ") for l in src.splitlines())
    assert src.count("subprocess.run(") == 3            # the generator --check, the probe and the main checkout's git lookup
    assert "BatchMode=yes" in src


def test_submit_newmodels_names_the_staging_directory_and_a_neutral_worktree(tmp_path, monkeypatch):
    """The printed bank-view commands name the site's staging directory (SISTEER_STAGE_DIR), and the grading host's worktree
    defaults to wave-drive in the main checkout's worktrees directory."""
    monkeypatch.setenv("SISTEER_STAGE_DIR", str(tmp_path / "stage"))
    rc = _run_printer(tmp_path, "ok")
    assert rc.returncode == 0, rc.stdout + rc.stderr
    assert f"make_bank_view.py --bank {tmp_path / 'stage'}/sdfbase/full22_objects/nemotron3nano4b --target nemotron3nano4b" in rc.stdout
    assert "modal" + "-stage" not in rc.stdout     # assembled, so this line names no site path itself
    assert os.path.join("worktrees", "wave-drive") in rc.stdout


def test_submit_newmodels_refuses_a_short_view_a_missing_snapshot_or_an_unreachable_cluster(tmp_path):
    for mode, needle in (("short", "is SHORT"), ("nocache", "no snapshot"), ("down", "CHECK ssh - PROBLEM")):
        rc = _run_printer(tmp_path, mode)
        assert rc.returncode == 2, (mode, rc.stdout, rc.stderr)
        assert "REFUSING TO PRINT SUBMISSIONS" in rc.stdout and needle in rc.stdout, mode
        assert not any(l.startswith("ssh ") for l in rc.stdout.splitlines()), mode   # nothing submittable
        assert "sbatch" not in rc.stdout, mode
        if mode == "short":
            assert "make_bank_view.py --target glm46vflash --view main" in rc.stdout   # the fix is printed
            assert "never a log line" in rc.stdout


def test_submit_newmodels_routing_flags_and_offline(tmp_path):
    rc = _run_printer(tmp_path, "ok", "--cluster", "glm46vflash=cluster_a")
    assert rc.returncode == 0, rc.stderr
    lines = rc.stdout.splitlines()
    sn, p = _plan({"glm46vflash": "cluster_a"})
    a, b = expect(tmp_path / "cluster-site", "a"), expect(tmp_path / "cluster-site", "b")
    routed = {"cluster_a": (f"ssh {a['host']} ", f"--qos={a['qos']} "), "cluster_b": (f"ssh {b['host']} ", f"--qos={b['qos']} ")}
    jobs = [(j[0], j[1], j[-1]) for j in p["att"] + p["stat"]]
    assert {c for _, key, c in jobs if key == "glm46vflash"} == {"cluster_a"}
    assert {c for _, key, c in jobs if key == "nemotron3nano4b"} == {"cluster_b"}
    for name, _, c in jobs:                                                   # every attack / static job on its cluster
        ls = [l for l in lines if l.startswith("ssh ") and f" -J {name} " in l]
        assert len(ls) == 1 and ls[0].startswith(routed[c][0]) and routed[c][1] in ls[0], (name, ls)
    assert any(l.startswith(f"ssh {a['host']} ") and "attacker_pool.sbatch" in l for l in lines)   # a pool per cluster
    rc = _run_printer(tmp_path, "ok", "--offline")
    assert rc.returncode == 0 and "UNVERIFIED (--offline)" in rc.stdout and "sbatch" in rc.stdout
    rc = _run_printer(tmp_path, "ok", "--cluster", "gemma4e4b=cluster_a")
    assert rc.returncode == 2                                                     # not a target of this printer
    rc = subprocess.run(["bash", str(F22 / "submit_newmodels.sh"), "notasha", "--offline"],
                        capture_output=True, text=True, env={**os.environ, "F22_PYTHON": sys.executable})
    assert rc.returncode == 2 and "full 40-hex" in rc.stdout


# --- 8: the second concurrent Petri proxy (the completion pass, 2026-09-08) --------------------------------------------

def test_petri_second_pass_keeps_its_own_dir_and_port_and_releases_its_proxy():
    """Legs added to a manifest while the first proxy's driver is mid-leg (the derived cells beside the prompted leg) run
    against a SECOND proxy of the same tag: petri_proxy.sbatch's 4th arg `second` moves its state to <tag>/serving_second and
    its port to +200 (never the first proxy's endpoint.json / STOP / heartbeat); petri_drive_full22.sh under F22_SECOND_PASS=1
    tunnels to that port, keeps its own state dir, runs a NORMAL `run` per leg (never --overflow-rerun), and releases the
    proxy when its named legs end. The two pass flags are exclusive."""
    proxy = (F22 / "petri_proxy.sbatch").read_text()
    assert 'case "$PASS" in ""|overflow-rerun|second)' in proxy
    assert 'OUT="$WAVE_ROOT/petri/$TAG/serving_second"; PROXY_PORT=$((PROXY_PORT + 200))' in proxy
    drv = (F22 / "petri_drive_full22.sh").read_text()
    assert 'SECOND="${F22_SECOND_PASS:-}"' in drv and 'PPORT=$((PPORT + 200))' in drv
    assert 'SUFFIX="${RERUN:+_rerun}${SECOND:+_second}"' in drv
    # the drive's grading host state dir and the cluster REMOTE_OUT derive the
    # SAME root suffix common.sh f22_init gives the wave root — `_native` under the default F22_SERVING_PATH=lora_native,
    # empty (byte-identical to before) under lens — so a native wave never reads a lens wave's endpoint; the pass
    # suffix ($SUFFIX: _rerun / _second) rides on top unchanged
    assert 'ROOT_SUFFIX="$( [ "$F22_SERVING_PATH" = lens ] || echo _native )"' in drv
    assert 'STATE="$HOME/.local/state/sdfbase_full22$ROOT_SUFFIX/petri/$TAG$SUFFIX"' in drv
    assert 'REMOTE_OUT="$SHARED/exp/sdfbase_full22$ROOT_SUFFIX/petri/$TAG/serving$SUFFIX"' in drv
    assert 'if [ -n "$RERUN" ] || [ -n "$SECOND" ]; then' in drv          # release on the pass's end
    assert drv.count("--overflow-rerun}") == 1 and "${SECOND:+--overflow-rerun}" not in drv   # a second pass is a normal run
    assert "F22_OVERFLOW_RERUN and F22_SECOND_PASS are exclusive" in drv


# --- 9: the completion pass's shas hold a NARROWED, stated invariant (README 12.1) ------------------------------------------

def test_completion_lane_shas_keep_the_serving_path_and_touch_only_nemotron():
    """The nemotron completion lane ran at S1 (the prompted cell, the controls) and S2 (the derived pair) on a tree that differs
    from the wave sha 254ccbb4 under src/ ONLY in sweep.py (the fit-provenance layer the controls driver needs — the gemma/qwen
    controls' own narrowed invariant) and under configs/ ONLY in nemotron3nano4b's roster entry / config / manifest / pins. The
    serving path and serving_impl_commit are byte-identical, which is what lets the new cells pool with the five wave cells.
    Checked against git, never trusted from the prose."""
    m = json.loads((F22 / "wave_manifest.json").read_text())
    block = m["sha_pairs_nemotron_complete"]
    serving = ["src/sisteer/capeval/serving", "src/sisteer/capeval/tiers/petri_runner.py"]
    for pr in block["pairs"]:
        a, b = pr["a"], pr["b"]
        for sha in (a, b):
            if subprocess.run(["git", "cat-file", "-e", f"{sha}^{{commit}}"], cwd=REPO, capture_output=True).returncode:
                pytest.skip("this clone does not carry the lane commits (a published repository without its development history); "
                            "the serving-identity claim is not verified here")
        assert subprocess.check_output(["git", "diff", "--name-only", a, b, "--", *serving], cwd=REPO, text=True).split() == [], pr["label"]
        src = subprocess.check_output(["git", "diff", "--name-only", a, b, "--", "src/"], cwd=REPO, text=True).split()
        assert set(src) <= set(block["allowed_src"]), (pr["label"], src)
        cfg = subprocess.check_output(["git", "diff", "--name-only", a, b, "--", "configs/"], cwd=REPO, text=True).split()
        assert set(cfg) <= set(block["allowed_configs"]), (pr["label"], cfg)
        def sic(sha):
            r = subprocess.check_output(["git", "show", f"{sha}:configs/sdfbase_full22_roster.yaml"], cwd=REPO, text=True)
            return [l.split()[1] for l in r.splitlines() if l.startswith("serving_impl_commit:")][0]
        assert sic(a) == sic(b) == m["evidence"]["serving_impl_commit"]["W7"], pr["label"]
    # the strict no-divergence list carries none of this pass's pairs (its rule is "nothing under src/ or configs/")
    lane_shas = {x for pr in block["pairs"] for x in (pr["a"], pr["b"])} - {"254ccbb43bcca7424bdfa0cbfdd8345d3292deb5"}
    assert not any(pr["a"] in lane_shas or pr["b"] in lane_shas for pr in m["sha_pairs_that_must_not_diverge_in_generation"])


def test_petri_driver_auditor_org_defaults_to_lowprio_and_is_recorded():
    """The auditor/judge key's org is a KNOB with a default, not a hardcode: F22_AUDITOR_ORG, default lowprio (the standing
    routing rule), `highprio` only for Petri runs after Low Prio sheds — and either way it is fetched from the secret
    manager at use time (never a file, never printed) and named in the DRIVE_START line so a leg's transcript
    says which pool audited it."""
    drv = (F22 / "petri_drive_full22.sh").read_text()
    assert 'F22_AUDITOR_ORG="${F22_AUDITOR_ORG:-lowprio}"' in drv
    assert 'case "$F22_AUDITOR_ORG" in lowprio|highprio) ;;' in drv
    assert 'ANTHROPIC_API_KEY="$("$SECRET_CMD" "$F22_AUDITOR_ORG")"' in drv
    assert '"$SECRET_CMD" lowprio' not in drv          # no hardcoded org left
    assert "auditor_org=$F22_AUDITOR_ORG" in drv       # recorded in DRIVE_START
