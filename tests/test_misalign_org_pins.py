"""The EM MODEL-ORGANISMS panel `qwen35_4b_org` serves the
constitution / control / difference arms on BOTH paths — LoRA (constitution, control, the direct difference) and ReLU / PSR
(constitution, control, the DIRECT difference, the DISTILLED difference) — as the RESEED lane's effective-batch-64 seed-1
objects (batch-64 is preferred; the wave's main view holds the older batch-8 fits under the same cell ids), beside the two
prompted arms. The safety-distilled arms are dropped and AlphaSteer is out everywhere; the psr_ce_all
cells are IN for this key, an exception to the psr-out rule. configs/misalign_org_pins.json is the ONE authority
the generator, the bank view and the view manifest read, and it carries the lane's own quality verdicts verbatim (the direct
ReLU difference has no ADMISSION.json; the distilled one is admitted: false) — into the config header too, never laundered."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pathlib
import re
import subprocess
import sys

import pytest
import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
PINS = REPO / "configs/misalign_org_pins.json"
SCRIPT = REPO / "scripts/sdfbase/full22/misalign/misalign_bank_view.sh"
KEY, FAM = "qwen35_4b_org", "qwen35_4b"
#: the ruling's seven objects, in the pins' order: the LoRA trio, then the ReLU quartet
CELLS = ("constv2-mix-full__lora_r4_ce", "constv2ctl2-full__lora_r4_ce", "constv2-mix-full__lora_r4_ce_minus_ctl",
         "constv2-mix-full__psr_ce_all", "constv2ctl2-full__psr_ce_all", "constv2-mix-full__psr_ce_all_minus_ctl",
         "constv2-mix-full__psr_ce_all_minus_ctl_distilled")
FITTED, DERIVED = CELLS[:2] + CELLS[3:5], (CELLS[2], CELLS[5], CELLS[6])
PROMPTED = ("prompted__safety_refusal", "prompted__si_constitution")   # the generator appends the prompted arms in this order
DROPPED = ("safety_refusal__lora_r4", "safety_refusal__psr_kl_all")
KINDS = {"lora_r4_ce": "lora_all", "lora_r4_ce_minus_ctl": "lora_subtraction", "psr_ce_all": "psr",
         "psr_ce_all_minus_ctl": "psr_paired_write", "psr_ce_all_minus_ctl_distilled": "psr"}


def _generator():
    spec = importlib.util.spec_from_file_location(
        "make_misalign_manifest", REPO / "scripts/sdfbase/full22/misalign/make_misalign_manifest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pins():
    return json.loads(PINS.read_text(encoding="utf-8"))


def _petri_cells(stem):
    text = (REPO / f"configs/capeval/manifests/petri_full22_{stem}.yaml").read_text(encoding="utf-8")
    return _generator()._split_cells_block(text)[1]


def test_pins_file_names_the_seven_reseed_eb64_s1_objects_with_the_lanes_own_verdicts():
    pins, m = _pins(), _generator()
    assert pins["_schema"] == m.ORG_PINS_SCHEMA and tuple(pins["_cells"]) == CELLS
    rec = pins[FAM]
    assert (rec["fit_lane"], rec["fit_model_key"], rec["seed"], rec["effective_batch"]) == ("reseed", "qwen35_4b_base_eb64_s1", 1, 64)
    assert rec["fit_model_id"] == "Qwen/Qwen3.5-4B-Base" and rec["model"] == "Qwen/Qwen3.5-4B" and "s1 ONLY" in rec["seed_note"]
    fam_man = (REPO / "configs/capeval/manifests/misalign_full22_qwen35_4b.yaml").read_text(encoding="utf-8")
    assert rec["chat_template_hash"] == re.search(r"^  chat_template_hash: ([0-9a-f]{16})", fam_man, re.M).group(1)
    assert rec["resolved_revision"] == re.search(r"^  target_revision: (\S+)", fam_man, re.M).group(1)
    shas = {}
    for c in CELLS:
        o, arm = rec[c], c.split("__", 1)[1]
        assert re.fullmatch(r"[0-9a-f]{64}", o["object_sha256"]), c
        assert o["bank"] == "sdfbase/reseed" and o["source"].startswith("sdfbase/reseed/") and o["source"].endswith("/" + c)
        assert "qwen35_4b_base_eb64_s1" in o["source"] and (o["seed"], o["effective_batch"], o["fit_model_key"]) == (1, 64, rec["fit_model_key"])
        assert o["kind"] == KINDS[arm] and o["arm"] == arm and o["positions"] == "all"
        assert o["identity_sidecar"] == ("derived_meta.json" if c in DERIVED else "train_budget_meta.json")
        assert o["template_hash"] == rec["chat_template_hash"]          # fitted under the render the panel serves
        assert o["verified_cluster_a"] is True and o["read_only"] is True
        shas[c] = o["object_sha256"]
    assert len(set(shas.values())) == 7
    # the fitted cells carry the lane's validation; every derived object names its parents by THIS file's shas
    for c in FITTED:
        assert rec[c]["records"]["VALIDATE.json"]["ok"] is True and rec[c]["records"]["COMPLETE"] is True and rec[c]["admission"] is None
    for c, pos, neg in ((CELLS[2], CELLS[0], CELLS[1]), (CELLS[5], CELLS[3], CELLS[4]), (CELLS[6], CELLS[3], CELLS[4])):
        p = rec[c]["parents"]
        assert (p["pos"]["cell_id"], p["pos"]["object_sha256"]) == (pos, shas[pos])
        assert (p["neg"]["cell_id"], p["neg"]["object_sha256"]) == (neg, shas[neg])
    # the lane's verdicts, verbatim: the LoRA difference admitted; the DIRECT ReLU difference has NO admission record (gate
    # liveness only); the DISTILLED one is admitted: FALSE — recorded, never hidden
    assert rec[CELLS[2]]["admission"]["admitted"] is True
    direct = rec[CELLS[5]]
    assert direct["admission"] is None and "NO ADMISSION.json" in direct["admission_note"]
    assert direct["gate_liveness"]["min"] == pytest.approx(0.3057, abs=1e-4) and direct["gate_liveness"]["n_layers"] == 32
    dist = rec[CELLS[6]]
    assert dist["admission"]["admitted"] is False and len(dist["admission"]["reasons"]) == 2
    assert dist["admission"]["recovery_verdict"] == "not recovered under this trainer + budget"
    assert dist["admission"]["gate_liveness_min"] < dist["admission"]["floor"] and "FALSE" in dist["admission_note"]
    assert dist["teacher"]["object_sha256"] == shas[CELLS[5]]
    # the direct ReLU difference is the SAME object the native-certification route pins (its staged copy re-hashed equal)
    rp = json.loads((REPO / "configs/reseed_paired_pins.json").read_text(encoding="utf-8"))[FAM][CELLS[5]]
    assert rp["object_sha256"] == shas[CELLS[5]] and rp["reseed_source"] == direct["source"]


def test_generator_pinned_org_key_serves_none_the_pinned_cells_and_the_two_prompted_arms_only():
    m = _generator()
    _, _, ra = m._sisteer()
    cells, status = m.misalign_cells(_petri_cells("qwen35_4b"), KEY)
    assert status == m.ORG_PINNED_STATUS
    assert cells == ["none", *(ra.harness_cell_id(c) for c in CELLS), "prompted__safety", "prompted__constitution"]
    tt = [ra.traintest_cell_id(c) for c in cells]
    assert set(tt) == {"none", *CELLS, *PROMPTED} and not set(tt) & set(DROPPED) and "alphasteer" not in tt
    assert sum("psr_ce_all" in c for c in tt) == 4          # constitution, control, the direct difference, the distilled one
    assert m.org_pins(KEY)["_cells"] == list(CELLS)
    # an _org key WITHOUT a pins entry keeps the wave path (llama8b_org: the main-view cells + the prompted arms, no alphasteer)
    assert m.org_pins("llama8b_org") is None
    cells_l, status_l = m.misalign_cells(_petri_cells("llama8b"), "llama8b_org")
    assert status_l == m.ORG_STATUS and cells_l[0] == "none" and "alphasteer" not in cells_l
    assert m.org_pins("qwen35_4b") is None                   # only an _org key reads the pins


def test_generated_org_config_and_manifest_list_exactly_the_ruled_cells_and_the_header_tells_the_truth():
    text = (REPO / f"configs/misalign_full22_{KEY}.yaml").read_text(encoding="utf-8")
    cfg = yaml.safe_load(text)
    assert cfg["explicit_cells"] == ["none", *CELLS, *PROMPTED]
    assert cfg["defenses_dir"] == f"out/misalign/{KEY}/cells_view" and cfg["private_run"] == f"misalign_full22_{KEY}"
    head, pins = text.split("explicit_cells:", 1)[0], _pins()[FAM]
    for c in CELLS:
        assert pins[c]["object_sha256"] in head, c            # every served object's full digest is in the header
    for needle in ("misalign_org_pins.json", "effective-batch-64", "seed-1", "qwen35_4b_base_eb64_s1", "batch-8",
                   "NO ADMISSION.json", "admitted: FALSE", "not recovered under this trainer + budget",
                   "AlphaSteer is OUT everywhere", "safety-distilled"):
        assert needle in head, needle
    man = (REPO / f"configs/capeval/manifests/misalign_full22_{KEY}.yaml").read_text(encoding="utf-8")
    m = _generator()
    _, _, ra = m._sisteer()
    assert m._split_cells_block(man)[1] == ["none", *(ra.harness_cell_id(c) for c in CELLS), "prompted__safety", "prompted__constitution"]
    mhead = man.split("\n  cells:", 1)[0]
    for needle in ("misalign_org_pins.json", "effective-batch-64", "NO ADMISSION.json", "admitted: FALSE", "batch-8", "psr-out"):
        assert needle in mhead, needle
    assert f"  audit: misalign_full22_{KEY}" in man
    # the family's own panel and the llama organisms key are untouched by the pins
    fam = yaml.safe_load((REPO / "configs/misalign_full22_qwen35_4b.yaml").read_text(encoding="utf-8"))
    assert "safety_refusal__lora_r4" in fam["explicit_cells"] and "constv2-mix-full__psr_ce_all" not in fam["explicit_cells"]
    llama = yaml.safe_load((REPO / "configs/misalign_full22_llama8b_org.yaml").read_text(encoding="utf-8"))
    assert llama["explicit_cells"] == ["none", "constv2-mix-full__lora_r4_ce", "constv2ctl2-full__lora_r4_ce",
                                       "constv2-mix-full__lora_r4_ce_minus_ctl", "prompted__safety_refusal", "prompted__si_constitution"]


def _pins_with(tmp_path, mutate):
    pins = _pins()
    mutate(pins)
    p = tmp_path / "misalign_org_pins.json"
    p.write_text(json.dumps(pins), encoding="utf-8")
    return p


def test_the_wave_guard_lets_the_pinned_org_panel_drop_only_the_two_safety_cells(tmp_path, monkeypatch):
    m = _generator()

    def drop_ctl(p):                                          # dropping a NON-safety wave cell refuses
        p["_cells"].remove("constv2ctl2-full__lora_r4_ce")
    monkeypatch.setattr(m, "ORG_PINS", _pins_with(tmp_path, drop_ctl))
    _, cells = m.render_manifest(KEY)
    with pytest.raises(SystemExit, match=r"drop wave cells \['constv2ctl2-full__lora_r4_ce'\]"):
        m.render_config(KEY, cells)

    def add_safety(p):                                        # a safety cell that creeps back in refuses too
        p["_cells"].append("safety_refusal__lora_r4")
        p[FAM]["safety_refusal__lora_r4"] = {**p[FAM]["constv2-mix-full__lora_r4_ce"], "arm": "lora_r4"}
    monkeypatch.setattr(m, "ORG_PINS", _pins_with(tmp_path, add_safety))
    _, cells = m.render_manifest(KEY)
    with pytest.raises(SystemExit, match="safety-distilled arms are OUT"):
        m.render_config(KEY, cells)
    monkeypatch.setattr(m, "ORG_PINS", PINS)                  # the committed pins pass both guards
    _, cells = m.render_manifest(KEY)
    _, tt = m.render_config(KEY, cells)
    assert tt == ["none", *CELLS, *PROMPTED]


def test_a_pin_fitted_under_another_render_or_replicate_or_without_a_verdict_refuses(tmp_path, monkeypatch):
    m = _generator()

    def other_render(p):
        p[FAM]["chat_template_hash"] = "0000000000000000"
    monkeypatch.setattr(m, "ORG_PINS", _pins_with(tmp_path, other_render))
    with pytest.raises(SystemExit, match="chat_template_hash"):
        m.render_manifest(KEY)

    def other_seed(p):
        p[FAM]["constv2-mix-full__psr_ce_all"]["seed"] = 2
    monkeypatch.setattr(m, "ORG_PINS", _pins_with(tmp_path, other_seed))
    with pytest.raises(SystemExit, match="one replicate per panel"):
        m.render_manifest(KEY)

    def no_verdict(p):
        p[FAM]["constv2-mix-full__psr_ce_all_minus_ctl"]["admission_note"] = ""
    monkeypatch.setattr(m, "ORG_PINS", _pins_with(tmp_path, no_verdict))
    with pytest.raises(SystemExit, match="admission_note"):
        m.render_manifest(KEY)


def _fake_store(tmp_path, pins):
    """A fake $SHARED with one object per pinned cell (random bytes) + its identity sidecar; {cell: (sha, bytes)}."""
    shared, blobs = tmp_path / "shared", {}
    for c in pins["_cells"]:
        o = pins[FAM][c]
        d = shared / "exp/objects" / o["source"]
        d.mkdir(parents=True)
        blob = os.urandom(256) + c.encode()
        (d / "object.pt").write_bytes(blob)
        (d / o["identity_sidecar"]).write_text("{}", encoding="utf-8")
        blobs[c] = (hashlib.sha256(blob).hexdigest(), blob)
    return shared, blobs


def _run(shared, pins_path, *args):
    env = {k: v for k, v in os.environ.items() if k not in ("MISALIGN_SHARED", "MISALIGN_ORG_PINS", "MISALIGN_PY")}
    env["MISALIGN_SHARED"] = str(shared)
    if pins_path is not None:
        env["MISALIGN_ORG_PINS"] = str(pins_path)
    r = subprocess.run(["bash", str(SCRIPT), KEY, "--python", sys.executable, *args],
                       env=env, capture_output=True, text=True, cwd=str(REPO), timeout=300)
    return r.returncode, r.stdout + r.stderr


@pytest.mark.skipif(sys.platform != "linux", reason="GNU find -printf / flock")
def test_bank_view_script_links_the_pinned_org_view_read_only_and_refuses_drift(tmp_path):
    pins = _pins()
    shared, blobs = _fake_store(tmp_path, pins)
    for c in CELLS:                                           # the fake store's digests, kept self-consistent
        pins[FAM][c]["object_sha256"] = blobs[c][0]
    for c in DERIVED:
        for side in ("pos", "neg"):
            par = pins[FAM][c]["parents"][side]
            par["object_sha256"] = blobs[par["cell_id"]][0]
    pins[FAM][CELLS[6]]["teacher"]["object_sha256"] = blobs[CELLS[5]][0]
    fake_pins = tmp_path / "pins.json"
    fake_pins.write_text(json.dumps(pins), encoding="utf-8")
    view = shared / "exp/objects/misalign" / KEY / "cells_view"
    rc, out = _run(shared, fake_pins, "--check")             # --check before a build: nothing to verify -> refuses
    assert rc == 2 and "no view at" in out, out
    rc, out = _run(shared, fake_pins)
    assert rc == 0, out
    assert "no wave entry" in out and "OUT everywhere" in out
    assert sorted(p.name for p in view.iterdir() if p.name != "VIEW_MANIFEST.json") == sorted(CELLS)
    for c in CELLS:
        link = view / c
        assert link.is_symlink() and os.readlink(link) == f"../../../{pins[FAM][c]['source']}"
        assert hashlib.sha256((link / "object.pt").read_bytes()).hexdigest() == blobs[c][0]
    man = json.loads((view / "VIEW_MANIFEST.json").read_text(encoding="utf-8"))
    assert man["schema"] == "misalign_org_view/1" and man["pins"] == str(fake_pins) and man["alphasteer_status"] == "out_organisms"
    assert man["pins_sha256"] == hashlib.sha256(fake_pins.read_bytes()).hexdigest()
    ents = {e["cell_id"]: e for e in man["entries"]}
    assert set(ents) == set(CELLS)
    doc = json.loads(fake_pins.read_text(encoding="utf-8"))
    for c, e in ents.items():
        # the PIN-FORM shape mt_serving_up._pin_form_sha reads: `pins` a file, `pin` = <model>.<cell>, resolving to the digest
        assert e["form"] == "pin" and e["pin"].split(".") == [FAM, c] and pathlib.Path(e["pins"]).is_file()
        assert doc[FAM][c]["object_sha256"] == e["object_sha256"] == blobs[c][0]
        assert (e["kind"], e["staged_id"], e["identity_sidecar"]) == (pins[FAM][c]["kind"], c, pins[FAM][c]["identity_sidecar"])
    # the lane's verdicts ride the manifest: the direct difference's null admission + note, the distilled's admitted: false
    assert ents[CELLS[5]]["admission"] is None and "NO ADMISSION.json" in ents[CELLS[5]]["admission_note"]
    assert ents[CELLS[6]]["admission"]["admitted"] is False
    # idempotent + --check clean; the fake store was never written to (read-only): object.pt + sidecar per dir, nothing else
    rc, out = _run(shared, fake_pins, "--check")
    assert rc == 0, out
    for c in CELLS:
        d = shared / "exp/objects" / pins[FAM][c]["source"]
        assert sorted(p.name for p in d.iterdir()) == sorted(["object.pt", pins[FAM][c]["identity_sidecar"]])
    # a stale symlink the pins do not name is a PROBLEM under --check and is removed in build mode
    (view / "alphasteer").symlink_to("../../../nowhere")
    rc, out = _run(shared, fake_pins, "--check")
    assert rc == 2 and "stale entry alphasteer" in out, out
    rc, out = _run(shared, fake_pins)
    assert rc == 0 and "removed stale entry alphasteer" in out and not (view / "alphasteer").is_symlink()
    # a drifted object refuses in both modes, naming the cell; restored, the view verifies again
    obj = shared / "exp/objects" / pins[FAM][CELLS[3]]["source"] / "object.pt"
    obj.write_bytes(b"drift")
    rc, out = _run(shared, fake_pins, "--check")
    assert rc == 2 and CELLS[3] in out and "sha256" in out and "!= pinned" in out, out
    rc, out = _run(shared, fake_pins)
    assert rc == 2
    obj.write_bytes(blobs[CELLS[3]][1])
    # a second identity sidecar beside an object refuses (the pin names exactly one)
    extra = shared / "exp/objects" / pins[FAM][CELLS[0]]["source"] / "fit_provenance.json"
    extra.write_text("{}", encoding="utf-8")
    rc, out = _run(shared, fake_pins, "--check")
    assert rc == 2 and "identity sidecars" in out, out
    extra.unlink()
    rc, out = _run(shared, fake_pins, "--check")
    assert rc == 0, out
    # the COMMITTED pins against this fake store: the real digests differ from the fakes -> the first cell refuses
    rc, out = _run(shared, None)
    assert rc == 2 and "!= pinned" in out, out


def test_the_cluster_scripts_carry_the_org_pins_and_the_pinned_view_branch():
    bv = SCRIPT.read_text(encoding="utf-8")
    sb = (REPO / "scripts/sdfbase/full22/misalign/misalign_proxy.sbatch").read_text(encoding="utf-8")
    assert "misalign_org_pins.json" in bv and "pinned_view_link" in bv and bv.count("pinned_view_link") >= 3
    assert "VIEW_MANIFEST_baselines.json" in bv                # the baselines branch keeps its manifest
    assert '"form": "pin"' in bv and 'f"{fam}.{c}"' in bv       # pin-form entries for mt_serving_up.view_manifest_digests
    assert KEY in sb and "llama8b_org" in bv
