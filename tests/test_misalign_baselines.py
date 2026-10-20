"""The PUBLISHED BASELINES on the agentic misalignment instrument — the
Arditi refusal direction and the AlphaSteer RE-BAKE rung — run as their own `<family>_bl` panels: none + the two
pinned objects, on the family's main-view serving, agentic only. The pins (configs/misalign_baseline_pins.json)
are the ONE authority the generator, the bank view and the analysis read; a pin that disagrees with the family
manifest's render or revision refuses."""
from __future__ import annotations

import importlib.util
import json
import pathlib
import re

import pytest
import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
FAMS = ("gemma4e4b", "qwen35_4b", "nemotron3nano4b", "llama8b")
#: the 2026-09-14 ruling (gemma lam3, qwen lam1, nemotron lam2, llama lam1); the earlier ladder picks sit under `superseded`
RUNG = {"gemma4e4b": "lam2", "qwen35_4b": "lam1", "nemotron3nano4b": "lam2", "llama8b": "lam1"}   # gemma re-ruled lam3 -> lam2 at 14:26Z


def _generator():
    spec = importlib.util.spec_from_file_location(
        "make_misalign_manifest", REPO / "scripts/sdfbase/full22/misalign/make_misalign_manifest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pins():
    return json.loads((REPO / "configs/misalign_baseline_pins.json").read_text(encoding="utf-8"))


def test_pins_file_names_eight_distinct_objects_with_the_ruled_doses_and_outcomes():
    pins = _pins()
    assert pins["_schema"] == "misalign_baseline_pins/1" and pins["_cells"] == ["arditi", "alphasteer"]
    for f in FAMS:
        rec = pins[f]
        for c in ("arditi", "alphasteer"):
            o = rec[c]
            assert re.fullmatch(r"[0-9a-f]{64}", o["object_sha256"]), (f, c)
            assert o["source"].startswith(f"{o['bank']}/{f}/") and o["identity_sidecar"] == "fit_provenance.json"
            # one render, one revision per family: the objects were fitted under the family's pinned tokenizer + checkpoint
            assert o["chat_template_hash"] == rec["chat_template_hash"] and o["resolved_revision"] == rec["resolved_revision"]
        ar, al = rec["arditi"], rec["alphasteer"]
        assert ar["kind"] == "static_r" and ar["phase"] == "prefill" and ar["dose_unit"] == "residual_norm_fraction"
        assert ar["alpha"] == 0.05                                              # the ruling's doses (llama re-ruled 0.1 -> 0.05, 2026-09-14)
        if f == "llama8b":
            assert ar["superseded_chain"][0]["alpha"] == 0.1 and ar["superseded_chain"][0]["object_sha256"] != ar["object_sha256"]
        assert al["rung"] == RUNG[f] and al["pending_bake"] is False and al["mechanism"] == "scalar_base_scale"
        # the moved rungs say what they supersede; llama's re-examination is recorded
        assert al["selection_outcome"] == {"gemma4e4b": "ruled_2026-09-14T14:26Z (supersedes ruled lam3, which superseded chosen lam1)",
                                           "qwen35_4b": "ruled_2026-09-14 (supersedes chosen)",
                                           "nemotron3nano4b": "ruled_2026-09-14 (supersedes a4_fallback)", "llama8b": "chosen_confirmed_2026-09-14"}[f]
        if f == "gemma4e4b":
            chain = al["superseded_chain"]; assert [c["rung"] for c in chain] == ["lam1", "lam3"] and al["rung"] not in {c["rung"] for c in chain}
            assert all(c["object_sha256"] != al["object_sha256"] for c in chain)
        elif f != "llama8b":
            sup = al["superseded"]; assert sup["object_sha256"] != al["object_sha256"] and sup["rung"] != al["rung"]
            assert sup["selection_outcome"] in al["selection_outcome"]          # the row label names what it supersedes
        else:
            assert al["contamination"]["corrected_contamination_ratio"] == 0.098 and "caveat" in al["contamination"]
    assert all("selection_note" in pins[f]["alphasteer"] for f in FAMS)      # every rung says how it was chosen
    shas = {pins[f][c]["object_sha256"] for f in FAMS for c in ("arditi", "alphasteer")}
    assert len(shas) == 8
    # the re-bake rung is NOT the campaign's selected__alphasteer object the family panel serves under the same cell id
    camp = json.loads((REPO / "configs/alphasteer_f22_pins.json").read_text(encoding="utf-8"))
    for f in FAMS:
        assert ((camp.get(f) or {}).get("selected__alphasteer") or {}).get("object_sha256") not in shas


def test_generator_baseline_keys_are_fixed_cell_panels_on_the_main_view():
    m = _generator()
    for f in FAMS:
        k = f"{f}_bl"
        assert k in m.TARGETS and m.is_baseline(k) and m.baseline_family(k) == f
        assert m.wave_of(k) == (f, "main") and m.source_stem(k) == f
        cells, status = m.misalign_cells(["none", "si_constv2_mix_full__lora_r4_ce", "prompted__safety"], k)
        assert cells == ["none", "arditi", "alphasteer"] and status == m.BASELINE_STATUS
        assert m.chat_template_pin(k) == (None, None)
        assert m.alphasteer_status(k)[1]["rung"] == RUNG[f]
    assert not m.is_baseline("gemma4e4b") and not m.is_baseline("llama8b_it")
    with pytest.raises(SystemExit):
        m.baseline_family("glm46vflash_bl")


def test_generated_baseline_configs_serve_the_family_model_with_exactly_the_three_cells():
    for f in FAMS:
        k = f"{f}_bl"
        cfg = yaml.safe_load((REPO / f"configs/misalign_full22_{k}.yaml").read_text(encoding="utf-8"))
        assert cfg["explicit_cells"] == ["none", "arditi", "alphasteer"]
        assert cfg["defenses_dir"] == f"out/misalign/{k}/cells_view" and cfg["private_run"] == f"misalign_full22_{k}"
        fam = yaml.safe_load((REPO / f"configs/misalign_full22_{f}.yaml").read_text(encoding="utf-8"))
        assert (cfg["model"], cfg["model_revision"]) == (fam["model"], fam["model_revision"])
        man = (REPO / f"configs/capeval/manifests/misalign_full22_{k}.yaml").read_text(encoding="utf-8")
        fam_man = (REPO / f"configs/capeval/manifests/misalign_full22_{f}.yaml").read_text(encoding="utf-8")
        tpl = re.search(r"^  chat_template_hash: ([0-9a-f]{16})", man, re.M).group(1)
        assert tpl == re.search(r"^  chat_template_hash: ([0-9a-f]{16})", fam_man, re.M).group(1)
        assert _generator()._split_cells_block(man)[1] == ["none", "arditi", "alphasteer"]   # the manifest's `  cells:` block only
        assert f"  audit: misalign_full22_{k}" in man


def test_a_pin_fitted_under_another_render_refuses(tmp_path, monkeypatch):
    m = _generator()
    pins = _pins(); pins["gemma4e4b"]["arditi"]["chat_template_hash"] = "0000000000000000"
    bad = tmp_path / "pins.json"; bad.write_text(json.dumps(pins), encoding="utf-8")
    monkeypatch.setattr(m, "BASELINE_PINS", bad)
    with pytest.raises(SystemExit, match="chat_template_hash"):
        m.render_manifest("gemma4e4b_bl")
    pins = _pins(); pins["llama8b"]["alphasteer"]["resolved_revision"] = "deadbeef"
    bad.write_text(json.dumps(pins), encoding="utf-8")
    with pytest.raises(SystemExit, match="resolved_revision"):
        m.render_manifest("llama8b_bl")


def test_the_cluster_scripts_carry_every_baseline_key_and_skip_em8q():
    sb = (REPO / "scripts/sdfbase/full22/misalign/misalign_proxy.sbatch").read_text(encoding="utf-8")
    bv = (REPO / "scripts/sdfbase/full22/misalign/misalign_bank_view.sh").read_text(encoding="utf-8")
    for f in FAMS:
        assert f"{f}_bl" in sb and f"{f}_bl" in bv
    assert "em8q skipped for the baselines panel" in sb
    assert "misalign_baseline_pins.json" in bv and "VIEW_MANIFEST_baselines.json" in bv


def test_analysis_and_figures_know_the_baseline_arms():
    from sisteer.capeval.misalign import analyze as A
    assert "arditi" in A.ARM_ORDER and "arditi" in A.ARM_LABELS
    # The paper-style delta figure script this test also read (`paper_style_deltas.py`: the baselines panel drew Arditi alone,
    # per the ruling of 2026-09-17) left the tree with the paperlib figure build it imported (2026-10-08).
