"""ifeval-canonical-battery — the fourteen per-TAG overlay configs + harness manifests.

Pinned: ``make_ifeval_overlays.py --check`` is byte-identical for all seven TAGs (14 files); each overlay config's
``explicit_cells`` is EXACTLY the TAG's launch + held matrix rows in matrix order (the deferred AlphaSteer / Arditi slots
never listed) and each manifest's ``cells`` their harness ids (``prompted__constitution`` for the constitution cell);
everything else of the campaign file is byte-identical; the campaign originals are untouched; the two Llama TAGs'
campaign files are byte-identical copies of the Llama campaign blobs, so the Llama branch merges clean;
every slug's registry row points at an overlay that exists.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from sisteer.capeval.tinycap import contracts as C

REPO = Path(__file__).resolve().parents[1]
IFE = REPO / "scripts" / "sdfbase" / "ifeval"
ENV = {**os.environ, "PYTHONPATH": str(REPO / "src")}
#: main's Llama campaign files (git blob sha1) since the Llama wave landed on main: the two Llama TAGs'
#: overlays are rendered from THESE, no longer from copies of a branch's blobs. The two MANIFEST blobs moved on
#: 2026-09-13 (native-serving-default, the native serving default): the `serving_launch_flags` comment and the
#: derived-twin sentence changed — COMMENT LINES ONLY; the loaded PetriSpec and the harness_config_hash are pinned
#: unchanged by tests/test_native_serving_default_text.py — and the overlays were regenerated from the new bytes. The
#: two CONFIG blobs are untouched by design: the wave config header keeps its lens-era serving sentence precisely so
#: these pins hold; the engine a wave ran on is read from the standup's stamps, never from a header.
#: The two CONFIG blobs moved once more on 2026-09-17: the roster's wave-wide `bon_climb_temperature: 1.0` became
#: `bon_climb_decode: as_drawn` (best-of-N draws at each served model's vendor preset and a climb re-draws at the
#: sampler its store records -- no wave-wide scalar can say that), so make_full22_configs re-rendered every wave
#: config's `bon` block; a deliberate roster-driven regeneration, and the pin moved with it (the IFEval overlays
#: read neither key). The two MANIFEST blobs moved on 2026-09-19 when the abliterated-Qwen auditor leg was
#: removed from every Petri manifest; the IFEval overlays do not read the auditor chain.
LLAMA_CAMPAIGN_BLOBS = {
    "configs/sdfbase_full22_llama8b.yaml": "c16c2fab6d3a77752cbde309f660a5fdc0c535c7",
    "configs/sdfbase_full22_llama8b_it.yaml": "ed7daba8c792e1156448b19eeccef36b7b6f4bcd",
    "configs/capeval/manifests/petri_full22_llama8b.yaml": "5e8093de9e5001eebb4b5a4287cde9ba7a7275a2",
    "configs/capeval/manifests/petri_full22_llama8b_it.yaml": "e4ccef63ffa3db0c2a4ae14c7edb2f4debf9e74a",
}


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


IM = _load("_if_matrix_overlays_under_test", "scripts/sdfbase/ifeval/if_matrix.py")
OV = _load("_make_ifeval_overlays_under_test", "scripts/sdfbase/ifeval/make_ifeval_overlays.py")
ROWS = IM.load_matrix()["rows"]


def _git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def test_the_fourteen_overlays_regenerate_byte_identically():
    r = subprocess.run([sys.executable, str(IFE / "make_ifeval_overlays.py"), "--check"], capture_output=True, text=True,
                       cwd=REPO, env=ENV, check=False)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "OK: 7 TAGs' overlays (14 files) byte-identical" in r.stdout
    paths = [p for p, _ in OV.outputs(REPO, ROWS)]
    assert len(paths) == 14 and all(p.is_file() for p in paths)
    for path, text in OV.outputs(REPO, ROWS):
        assert path.read_text(encoding="utf-8") == text


def test_every_slugs_registry_row_points_at_an_overlay_that_exists():
    # the seven instrument-view slugs AND the six seed lane-view slugs: 13 pairs, 26 files
    assert len(C.IFEVAL_SLUGS) == 13
    for (key, view), slug in C.IFEVAL_SLUGS.items():
        row = C.model_row(slug)
        assert (REPO / row.cfg).is_file(), row.cfg
        assert (REPO / C.IFEVAL_OVERLAY_MANIFEST_REL.format(tag=C.ifeval_tag(key, view))).is_file()
        assert row.cfg == C.IFEVAL_OVERLAY_CONFIG_REL.format(tag=C.ifeval_tag(key, view))
        assert "ifeval-canonical-battery OVERLAY" in (REPO / row.cfg).read_text(encoding="utf-8")


def test_each_overlay_lists_exactly_its_tags_launch_and_held_rows_never_the_deferred_slots():
    by_tag = OV.rows_by_tag(ROWS)
    assert set(by_tag) == {"gemma4e4b_it", "gemma4e4b", "qwen35_4b_it", "qwen35_4b", "nemotron3nano4b", "llama8b", "llama8b_it"}
    for tag, trows in by_tag.items():
        cfg_path, man_path = OV.overlay_paths(REPO, trows)
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        man = yaml.safe_load(man_path.read_text(encoding="utf-8"))
        want = [r for r in trows if r["status"] in (IM.STATUS_LAUNCH, IM.STATUS_HELD)]
        assert cfg["explicit_cells"] == [r["cell"] for r in want], tag
        assert man["petri"]["cells"] == [r["harness_cell"] for r in want], tag
        assert cfg["explicit_cells"][0] == "none" and man["petri"]["cells"][0] == "none"
        assert not any(c in cfg["explicit_cells"] for c in ("alphasteer", "arditi"))
        assert not any("psr" in c or c == "safety_refusal__lora_r4" for c in cfg["explicit_cells"]), tag
        # the campaign body follows byte for byte except the cells block (and the generated header)
        src = OV.campaign_config(REPO, tag).read_text(encoding="utf-8")
        body = cfg_path.read_text(encoding="utf-8").split("#\n", 1)[1]
        m = OV._CELLS_BLOCK.search(src)
        assert body.startswith(src[:m.start()]) and body.endswith(src[m.end():])
        # the second admitted change (the wave shakedown, 2026-09-13): the bank's declared multi-layer objects — F9
        # requires the set to EQUAL the object's layers; the Nemotron LoRA objects live on [12, 17, 24, 32] and the
        # campaign config's single `layer: 20` alone refused every Nemotron fitted cell. Read off the bcbw serving
        # config of the same bank, restricted to this TAG's fitted rows; absent for every other TAG
        declared = OV.multi_layer_arms_for_tag(REPO, want)
        assert cfg.get("multi_layer_arms") == (declared or None), tag
        if tag == "nemotron3nano4b":
            bcbw = yaml.safe_load((REPO / C.bcbw_config_path("nemotron3nano4b", "main")).read_text(encoding="utf-8"))
            assert declared == {c: [12, 17, 24, 32] for c in ("constv2-mix-full__lora_r4_ce", "constv2ctl2-full__lora_r4_ce",
                                                              "constv2-mix-full__lora_r4_ce_minus_ctl")}
            assert declared == {c: bcbw["multi_layer_arms"][c] for c in declared}
            middle = body[len(src[:m.start()]):len(body) - len(src[m.end():])]
            assert middle.startswith("explicit_cells:\n") and "\nmulti_layer_arms:\n" in middle
        else:
            assert declared == {} and "multi_layer_arms" not in cfg, tag
        msrc = OV.campaign_manifest(REPO, tag).read_text(encoding="utf-8")
        mbody = man_path.read_text(encoding="utf-8").split(f"GENERATED by {OV.GENERATOR}.\n", 1)[1]
        mm = OV._MANIFEST_CELLS.search(msrc)
        assert mbody.startswith(msrc[:mm.start()]) and mbody.endswith(msrc[mm.end():])
        # the prompted cells the matrix places on the view ride both files under their two spellings
        for r in want:
            if r["cell"].startswith("prompted__"):
                assert r["harness_cell"] in man["petri"]["cells"] and r["harness_cell"].startswith("prompted__")
                si = r["harness_cell"][len("prompted__"):]
                assert si in man["petri"]["prompted_prefix_hashes"], (tag, si)
    # the constitution cell is on exactly the five views that carry a prompted cell, as prompted__constitution
    con = {tag for tag, trows in by_tag.items() if any(r["cell"] == "prompted__si_constitution" for r in trows)}
    assert con == {"gemma4e4b_it", "qwen35_4b_it", "nemotron3nano4b", "llama8b", "llama8b_it"}
    for tag in con:
        man = yaml.safe_load(OV.overlay_paths(REPO, by_tag[tag])[1].read_text(encoding="utf-8"))
        assert "prompted__constitution" in man["petri"]["cells"]
        assert man["petri"]["prompted_prefix_hashes"]["constitution"] == C.IFEVAL_CONSTITUTION_SI_SHA256[:16]


def test_the_campaign_originals_are_untouched_and_llamas_are_the_llama_lanes_blobs():
    for rel, sha in LLAMA_CAMPAIGN_BLOBS.items():
        assert _git_blob_sha1(REPO / rel) == sha, rel
    # no campaign file lists an overlay-only cell, and no overlay is a campaign file
    for tag in OV.rows_by_tag(ROWS):
        src = OV.campaign_config(REPO, tag).read_text(encoding="utf-8")
        assert "ifeval-canonical-battery OVERLAY" not in src
        # (the campaign rosters serve prompted__si_constitution themselves now — no longer an
        # overlay-only cell; the overlay still adds nothing the matrix does not name)
    r = subprocess.run([sys.executable, str(REPO / "scripts/sdfbase/gsm8k/make_gsm8k_overlays.py"), "--check"],
                       capture_output=True, text=True, cwd=REPO, env=ENV, check=False)
    assert r.returncode == 0, r.stdout + r.stderr


def test_a_fitted_matrix_cell_the_campaign_config_does_not_list_refuses(tmp_path):
    rows = [dict(r) for r in ROWS if r["tag"] == "gemma4e4b" and r["status"] == IM.STATUS_LAUNCH]
    bogus = dict(rows[1], cell="constv2-mix-full__lora_r9_ce", harness_cell="si_constv2_mix_full__lora_r9_ce")
    import pytest
    with pytest.raises(SystemExit, match="does not list the fitted cell"):
        OV.render_config(REPO, "gemma4e4b", rows + [bogus])
    # a deferred slot with no harness spelling is never an overlay row, and a missing campaign file refuses by name
    deferred = [r for r in ROWS if r["status"] == IM.STATUS_DEFERRED]
    assert deferred and OV.overlay_rows(deferred) == []
    with pytest.raises(SystemExit, match="no campaign config for TAG"):
        OV.render_config(tmp_path, "gemma4e4b", rows)
    assert re.fullmatch(r"[0-9a-f]{64}", C.IFEVAL_CONSTITUTION_SI_SHA256)


# --------------------------------------------------------------------------- the seed sibling documents
SEED_MATRIX = REPO / C.IFEVAL_MATRIX_QWEN_SEEDREP_REL
SEED_ROWS = IM.load_matrix(SEED_MATRIX)["rows"]
COMPOSED = "constv2-mix-full__psr_ce_all_minus_ctl"


def test_the_seed_sibling_documents_twelve_overlays_regenerate_byte_identically_with_the_lane_extra_and_anchor_rules():
    """Six TAG pairs (qwen35_4b_s1..s3, s1p..s3p; 12 files; 26 with the frozen document's 14). A seed TAG's explicit_cells are
    its eight roster cells in matrix order PLUS the lane_extra `arditi_allpos` (admitted although the campaign config does not
    list it; the standup's explicit-cell registry validates it in the staged bank) and NO `multi_layer_arms` block: the
    served-only Class B object's identity is derived from the object + its fit_provenance.json (the served-only branch,
    layer = the object's own), and the native standup serves it at its own layers — the first cut's `{arditi_allpos: [12]}`
    declaration fed a response-masked derivation and is retired. A paired TAG's rows carry no `none`; its overlay lists
    the bare anchor FIRST anyway (select_cells admits only enumerated cells), then the composed cell alone."""
    r = subprocess.run([sys.executable, str(IFE / "make_ifeval_overlays.py"), "--matrix", str(SEED_MATRIX), "--check"],
                       capture_output=True, text=True, cwd=REPO, env=ENV, check=False)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "OK: 6 TAGs' overlays (12 files) byte-identical" in r.stdout
    paths = [p for p, _ in OV.outputs(REPO, SEED_ROWS)]
    assert len(paths) == 12 and all(p.is_file() for p in paths)
    assert not ({p for p, _ in OV.outputs(REPO, ROWS)} & set(paths))                    # 26 distinct files across the two documents
    by_tag = OV.rows_by_tag(SEED_ROWS)
    assert list(by_tag) == [f"qwen35_4b_{v}" for v in C.IFEVAL_LANE_VIEWS]
    arditi_pins = json.loads((REPO / "configs/arditi_allpos_pins.json").read_text(encoding="utf-8"))["qwen35_4b"]
    for tag, trows in by_tag.items():
        cfg_path, man_path = OV.overlay_paths(REPO, trows)
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        man = yaml.safe_load(man_path.read_text(encoding="utf-8"))
        want = OV.overlay_rows(trows)
        assert want == trows                                                              # nothing held, nothing deferred
        assert cfg["explicit_cells"] == OV.overlay_cells(want) and cfg["explicit_cells"][0] == "none" and man["petri"]["cells"][0] == "none"
        assert cfg["defenses_dir"].endswith(f"/cells_view/{tag[len('qwen35_4b_'):]}")
        src = OV.campaign_config(REPO, tag).read_text(encoding="utf-8")
        body = cfg_path.read_text(encoding="utf-8").split("#\n", 1)[1]
        m = OV._CELLS_BLOCK.search(src)
        assert body.startswith(src[:m.start()]) and body.endswith(src[m.end():])
        assert "SERVING PATH: the matrix document declares" in cfg_path.read_text(encoding="utf-8")
        if tag.endswith("p"):
            assert cfg["explicit_cells"] == ["none", COMPOSED] and man["petri"]["cells"] == ["none", "si_constv2_mix_full__psr_ce_all_minus_ctl"]
            assert "multi_layer_arms" not in cfg and "NO OWN ANCHOR: this TAG's matrix rows" in cfg_path.read_text(encoding="utf-8")
            assert [r["cell"] for r in want] == [COMPOSED]
        else:
            assert cfg["explicit_cells"] == ["none", "constv2-mix-full__psr_ce_all", "constv2-mix-full__lora_r4_ce",
                                             "constv2ctl2-full__psr_ce_all", "constv2ctl2-full__lora_r4_ce",
                                             "constv2-mix-full__lora_r4_ce_minus_ctl", "prompted__safety_refusal",
                                             "prompted__si_constitution", "arditi_allpos"]
            assert man["petri"]["cells"] == ["none", "si_constv2_mix_full__psr_ce_all", "si_constv2_mix_full__lora_r4_ce",
                                             "si_constv2ctl2_full__psr_ce_all", "si_constv2ctl2_full__lora_r4_ce",
                                             "si_constv2_mix_full__lora_r4_ce_minus_ctl", "prompted__safety", "prompted__constitution",
                                             "arditi_allpos"]
            assert "multi_layer_arms" not in cfg and OV.multi_layer_arms_for_tag(REPO, want) == {}     # no declaration for the served-only object
            assert "LANE EXTRA (b64-capability-agentic-queue):" in cfg_path.read_text(encoding="utf-8")
            assert arditi_pins["ar_c0p3__arditi_allpos"]["served_layer"] == 12                          # the object's own layer, read by the identity branch, not declared here
            assert man["petri"]["prompted_prefix_hashes"]["constitution"] == C.IFEVAL_CONSTITUTION_SI_SHA256[:16]
            # the campaign config does NOT list the lane extra: the admission is the document's declaration, nothing else
            assert "arditi_allpos" not in src
    # a fitted cell that is neither a campaign cell nor a declared lane extra still refuses
    s1 = by_tag["qwen35_4b_s1"]
    bogus = dict(s1[1], cell="constv2-mix-full__lora_r9_ce", harness_cell="si_constv2_mix_full__lora_r9_ce")
    with pytest.raises(SystemExit, match="does not list the fitted cell"):
        OV.render_config(REPO, "qwen35_4b_s1", s1 + [bogus])
    # the frozen document's overlays are byte-identical beside the seed ones
    r2 = subprocess.run([sys.executable, str(IFE / "make_ifeval_overlays.py"), "--check"], capture_output=True, text=True, cwd=REPO, env=ENV, check=False)
    assert r2.returncode == 0 and "OK: 7 TAGs' overlays (14 files) byte-identical" in r2.stdout, r2.stdout + r2.stderr

