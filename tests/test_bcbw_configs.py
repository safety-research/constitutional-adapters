"""The SIX generated explicit-cells configs
(``scripts/bcbw_configs.py`` -> ``configs/tinycap/bcbw_*.yaml``, including the two
llama8b views) are byte-identical to their derivation, list exactly the roster view's
cells behind ``none`` and the prompted safety anchor, name the row's volume bank as
``defenses_dir`` and the row's target as ``model`` / ``model_revision``, declare Nemotron's
multi-layer LoRA cells, and leave zero GLM footprint. The roster the generator reads is
the SHARED full22 roster, which carries every lane target (the llama8b block among them;
the supplement that carried it before is retired) — a roster lacking one
refuses by name rather than reading as "no llama views"."""
from __future__ import annotations

import copy

import functools
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

from sisteer.capeval.tinycap import contracts as C
from sisteer.eval.cell_controls import MULTI_LAYER_ARMS_KEY

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "bcbw_configs.py"


@pytest.fixture(scope="module")
def G():
    spec = importlib.util.spec_from_file_location("bcbw_configs_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_the_six_configs_are_byte_identical_to_the_generator(G):
    roster, pins = G.load_roster(), G.load_pins()
    outs = G.outputs(roster)
    assert [(t, v) for _, t, v in outs] == list(C.BCBW_MODEL_VIEWS) and len(outs) == 19     # 7 + the six b64 seed views + the six all-models one-seed views
    for dst, target, view in outs:
        assert dst.is_file(), dst
        assert dst.read_text(encoding="utf-8") == G.render_config(roster, target, view, pins=pins), f"DRIFT {dst}"
    assert G.main(["--check"]) == 0


def test_every_config_lists_the_roster_view_behind_none_and_the_prompted_anchor(G):
    roster = G.load_roster()
    # gemma4e4b/it: 4 since 2026-09-13 — the batch-64 instruct-direct control LoRA + LoRA subtraction (itdoc64__ / derived_itdoc64__) beside the two itdoc__ objects
    want_objects = {("gemma4e4b", "it"): 4, ("gemma4e4b", "main"): 8, ("qwen35_4b", "main"): 8,
                    ("nemotron3nano4b", "main"): 6, ("llama8b", "it"): 3, ("llama8b", "main"): 3,
                    ("qwen35_4b", "it"): 4,       # the two batch-8 itdoc objects + the batch-64 control LoRA + the batch-64 LoRA subtraction
                    # per seed view the two PSR + two LoRA fits + the LoRA difference (base64s<s>__ /
                    # derived_base64s<s>__); per paired view the composed psr_paired_write + the constitution LoRA + the LoRA difference
                    # (the shared roster's s<s>p view) + the Arditi refusal-steer as a LANE EXTRA on the seed views (runnable)
                    ("qwen35_4b", "s1"): 6, ("qwen35_4b", "s2"): 6, ("qwen35_4b", "s3"): 6,
                    ("qwen35_4b", "s1p"): 3, ("qwen35_4b", "s2p"): 3, ("qwen35_4b", "s3p"): 3,
                    # 2026-09-23: the all-models one-seed views — per s1 view the family's four
                    # base fits + the LoRA difference (base__ / derived__, the main bank's objects re-served); per s1p view the
                    # composed psr_paired_write (paired_eb8__ / paired_eb64__) + the re-served r=4 constitution LoRA (MINIMAL:
                    # one re-served cell, which `paired_of` requires)
                    ("gemma4e4b", "s1"): 5, ("gemma4e4b", "s1p"): 2, ("llama8b", "s1"): 5, ("llama8b", "s1p"): 2,
                    ("nemotron3nano4b", "s1"): 5, ("nemotron3nano4b", "s1p"): 2}
    for dst, target, view in G.outputs(roster):
        doc = yaml.safe_load(dst.read_text())
        cells = doc["explicit_cells"]
        prompted = [C.CELL_PROMPTED_SAFETY] + [c for c in G.lane_prompted_cells(target, view) if c != C.CELL_PROMPTED_SAFETY]
        assert cells[:1 + len(prompted)] == [C.ROUTE_NONE_CELL, *prompted], (target, view)
        objects = cells[1 + len(prompted):]
        assert objects == G.object_cells(roster, target, view) + list(G.lane_extra_cells(roster, target, view)) and len(objects) == want_objects[(target, view)]
        # the Arditi refusal-steer is a LANE EXTRA of the QWEN seed rows only; the all-models one-seed rows carry none (2026-09-23)
        assert G.lane_extra_cells(roster, target, view) == ({"arditi_allpos": "ar_c0p3__arditi_allpos"}
                                                              if target == "qwen35_4b" and view in ("s1", "s2", "s3") else {})
        assert len(set(cells)) == len(cells)
        for c in cells:
            C.parse_launcher_cell(c)                         # the launcher's grammar accepts every listed id
        # the Cartesian keys never coexist with explicit_cells
        assert "si_ids" not in doc and "class_a_arms" not in doc


def test_the_lane_rosters_runnable_prompted_cells_enter_the_served_config_of_their_row(G):
    """In the six-arm design the constitution SI is a paper arm — prompt-only, no object, no pin.
    Whether a row SERVES it is the LANE roster's call: the generator reads the lane roster's runnable `prompted__*`
    cells for the wave whose `model_row` is the (target, view)'s registry slug and lists them after the safety
    anchor; a view no lane wave serves gains nothing. (The SHARED roster may DECLARE the cell on a view since
    2026-09-12 — that is admitted and, where the lane does not run it, HELD: the next test.)"""
    from sisteer.capeval.tinycap import widen_roster as W
    lane = G.load_lane_rosters()
    assert G.LANE_ROSTER == REPO / "configs" / "capeval" / "bcb_bfcl_200_widen_roster.yaml"
    # 2026-10-03: the FULL lane roster is read BESIDE the 200 roster as a union of waves (the 200 wins a same-named wave), so a wave
    # only the full roster carries would reach the generator; every 200 wave reads as before. On this tree every full-roster wave has
    # a 200 sibling, so the union IS the 200 roster's wave table
    assert G.LANE_ROSTER_FULL == REPO / "configs" / "capeval" / "bcb_bfcl_full_roster.yaml"
    two_hundred = W.load_roster(G.LANE_ROSTER)
    for wname in W.waves(two_hundred):
        assert lane["waves"][wname] is two_hundred["waves"][wname] or lane["waves"][wname] == two_hundred["waves"][wname], wname
    assert set(W.waves(W.load_roster(G.LANE_ROSTER_FULL))) <= set(W.waves(two_hundred)) and set(lane["waves"]) == set(W.waves(two_hundred))
    assert set(lane) == {"waves", "_sources"}
    assert G.lane_prompted_cells("gemma4e4b", "it") == [C.CELL_PROMPTED_SAFETY, "prompted__si_constitution"]
    assert C.parse_launcher_cell("prompted__si_constitution").kind == C.CELL_KIND_PROMPTED
    for target, view in C.BCBW_MODEL_VIEWS:
        if (target, view) == ("gemma4e4b", "it") or view in ("s1", "s2", "s3"):
            # ... and the three b64 seed views run both prompted arms (the eight-cell seed row) — read off the lane rosters
            assert G.lane_prompted_cells(target, view) == [C.CELL_PROMPTED_SAFETY, "prompted__si_constitution"], (target, view)
        else:
            assert [c for c in G.lane_prompted_cells(target, view) if c != C.CELL_PROMPTED_SAFETY] == [], (target, view)
    roster = G.load_roster()
    cells = G.explicit_cells(roster, "gemma4e4b", "it")
    assert cells[:3] == [C.ROUTE_NONE_CELL, C.CELL_PROMPTED_SAFETY, "prompted__si_constitution"] and cells[3:] == G.object_cells(roster, "gemma4e4b", "it")
    doc = yaml.safe_load((REPO / C.bcbw_config_path("gemma4e4b", "it")).read_text())
    assert doc["explicit_cells"] == cells and "prompted__si_constitution" in (REPO / C.bcbw_config_path("gemma4e4b", "it")).read_text()
    # the rule is the lane roster's RUNNABLE status: a blocked prompted entry is not listed, an extra runnable one is
    import copy
    variant = copy.deepcopy(two_hundred)
    e = variant["waves"]["gemma4e4b_it"]["entries"]["const_si"]
    e.update(status="blocked", reason="synthetic", unblock="synthetic")
    assert G.lane_prompted_cells("gemma4e4b", "it", lane_roster=variant) == [C.CELL_PROMPTED_SAFETY]
    assert G.explicit_cells(roster, "gemma4e4b", "it", lane_roster=variant)[:2] == [C.ROUTE_NONE_CELL, C.CELL_PROMPTED_SAFETY]
    variant = copy.deepcopy(two_hundred)
    variant["waves"]["gemma4e4b_main"]["entries"]["extra_si"] = {"cell_id": "prompted__si_constitution", "status": "runnable", "paper_arm": "si_constitution", "kind": "prompted"}
    # (the main wave runs no safety prompt of its own — only the it wave does — so the lane list is the one extra cell; the
    #  generator still lists the safety anchor first on every view and the lane cell after it)
    assert G.lane_prompted_cells("gemma4e4b", "main", lane_roster=variant) == ["prompted__si_constitution"]
    assert G.explicit_cells(roster, "gemma4e4b", "main", lane_roster=variant)[:3] == [C.ROUTE_NONE_CELL, C.CELL_PROMPTED_SAFETY, "prompted__si_constitution"]
    # whether or not the shared roster on THIS tree declares the cell, nothing is held on the row that runs it, and the
    # served list is the same list either way (the shared roster's declaration is not a second source of served cells)
    assert G.held_prompted_cells(roster, "gemma4e4b", "it") == []
    assert G.explicit_cells(roster, "gemma4e4b", "it").count("prompted__si_constitution") == 1
    # the header of the generated file names the lane roster as the cell's source
    assert "a runnable prompted cell of the LANE roster" in (REPO / C.bcbw_config_path("gemma4e4b", "it")).read_text()


def test_a_shared_roster_prompted_cell_the_lane_does_not_run_is_admitted_and_held(G):
    """ADMIT AND HOLD (`prompted__si_constitution: null` landed on the shared roster's gemma / qwen / nemotron views,
    which made this generator refuse by design). The generator no longer refuses a shared-roster prompted cell beyond
    the safety anchor: where the
    LANE roster RUNS it on the row (gemma4e4b/it) it is served once, exactly as before; where the lane does not (the
    main views) it is DECLARED in the generated header as HELD, with the lane roster's standing, and kept OUT of
    `explicit_cells` — the row's registry, and so its next job, is unchanged. The hold is released by the lane roster
    alone (a runnable entry for the cell); a queued entry keeps it. A shared-roster prompted cell the launcher grammar
    refuses (an SI with no harness translation) refuses at generation, run or not."""
    import copy
    from sisteer.capeval.tinycap import widen_roster as W
    roster, pins = G.load_roster(), G.load_pins()
    declared = copy.deepcopy(roster)
    for target, view in (("gemma4e4b", "it"), ("gemma4e4b", "main"), ("qwen35_4b", "main"), ("nemotron3nano4b", "main")):
        declared["targets"][target]["views"][view]["cells"]["prompted__si_constitution"] = None
    for target, view in C.BCBW_MODEL_VIEWS:
        served = G.explicit_cells(roster, target, view)
        assert G.explicit_cells(declared, target, view) == served, (target, view)      # the served list is the lane's call
        held = G.held_prompted_cells(declared, target, view)
        text = G.render_config(declared, target, view, pins=pins)
        doc = yaml.safe_load(text)
        assert doc["explicit_cells"] == served
        lane_runs = G.lane_prompted_cells(target, view)
        shared_declares = G.shared_prompted_cells(declared, target, view)
        # the ONE rule: held = the shared roster's declared prompted cells the lane does not run on the row
        assert held == [c for c in shared_declares if c not in lane_runs], (target, view)
        if (target, view) == ("gemma4e4b", "it") or view in ("s1", "s2", "s3"):
            # the lane RUNS the constitution SI on these rows (gemma it: the six-arm design; the b64 seed rows; the
            # all-models one-seed s1 rows of gemma / llama / nemotron, 2026-09-23) — served once, no prompted hold
            assert held == [] and shared_declares == ["prompted__si_constitution"]
            assert "(HELD; lane roster:" not in text and "also declared by the shared roster" in text
            assert served.count("prompted__si_constitution") == 1
        elif view in ("s1p", "s2p", "s3p"):                             # the paired-write views (qwen's three, the all-models s1p) declare no prompted cell beyond the anchor
            assert held == [] and shared_declares == [] and "prompted__si_constitution" not in served and "(HELD; lane roster:" not in text
        elif target == "llama8b":                                       # the shared block declares it on main / it; the lane's llama waves do not run it
            assert held == ["prompted__si_constitution"] and "(HELD; lane roster: no entry in wave " in text
        else:
            assert held == ["prompted__si_constitution"] and "prompted__si_constitution" not in served
            assert "(HELD; lane roster: no entry in wave " in text and "ADMIT AND HOLD" in text
            assert text.index("HELD") > text.index("explicit_cells` = [")   # declared in the cell list, after the served ones
        # on THIS tree's shared roster the only cell that can be held anywhere is the constitution SI, and never on
        # the row that runs it or on the llama rows (their block is the lane's supplement)
        assert set(G.held_prompted_cells(roster, target, view)) <= {"prompted__si_constitution"}
        if (target, view) == ("gemma4e4b", "it"):
            assert G.held_prompted_cells(roster, target, view) == []
    # the release is the lane roster's: a runnable entry moves the cell into the served list; a queued one keeps the hold
    lane = copy.deepcopy(G.load_lane_rosters())
    lane["waves"]["gemma4e4b_main"]["entries"]["const_si"] = {"cell_id": "prompted__si_constitution", "status": "runnable",
                                                              "paper_arm": "si_constitution", "kind": "prompted"}
    assert G.held_prompted_cells(declared, "gemma4e4b", "main", lane_roster=lane) == []
    assert G.explicit_cells(declared, "gemma4e4b", "main", lane_roster=lane)[:3] == [C.ROUTE_NONE_CELL, C.CELL_PROMPTED_SAFETY, "prompted__si_constitution"]
    lane["waves"]["gemma4e4b_main"]["entries"]["const_si"]["status"] = "queued"
    assert G.held_prompted_cells(declared, "gemma4e4b", "main", lane_roster=lane) == ["prompted__si_constitution"]
    assert G.lane_prompted_status("gemma4e4b", "main", "prompted__si_constitution", lane_roster=lane) == "queued (wave gemma4e4b_main, entry const_si)"
    assert G.lane_prompted_status("gemma4e4b", "main", "prompted__si_constitution") == "no entry in wave gemma4e4b_main"
    # a shared-roster prompted cell the grammar refuses is refused at generation — held or not
    bad = copy.deepcopy(roster)
    bad["targets"]["gemma4e4b"]["views"]["main"]["cells"]["prompted__nonsense_si"] = None
    with pytest.raises(SystemExit, match="launcher grammar refuses"):
        G.explicit_cells(bad, "gemma4e4b", "main")
    with pytest.raises(SystemExit, match="launcher grammar refuses"):
        G.held_prompted_cells(bad, "gemma4e4b", "main")


def test_defenses_dir_and_model_pins_are_the_rows(G):
    for dst, target, view in G.outputs(G.load_roster()):
        row = C.model_row(C.BCBW_SLUG_OF[(target, view)])
        doc = yaml.safe_load(dst.read_text())
        assert doc["defenses_dir"] == row.objects_dir == C.full22_objects_dir(target, view)
        assert doc["defenses_dir"].startswith(C.VOL_OBJECTS_FULL22 + "/")
        assert doc["model"] == row.hf and doc["model_revision"] == row.rev
        assert row.cfg == C.bcbw_config_path(target, view) == str(dst.relative_to(REPO))
        assert C.is_full22_row(row)


def test_nemotron_declares_its_three_lora_cells_on_the_four_attention_layers(G):
    doc = yaml.safe_load((REPO / C.bcbw_config_path("nemotron3nano4b", "main")).read_text())
    assert doc[MULTI_LAYER_ARMS_KEY] == {
        "constv2-mix-full__lora_r4_ce": [12, 17, 24, 32],
        "constv2ctl2-full__lora_r4_ce": [12, 17, 24, 32],
        "constv2-mix-full__lora_r4_ce_minus_ctl": [12, 17, 24, 32]}
    assert set(doc[MULTI_LAYER_ARMS_KEY]) <= set(doc["explicit_cells"])
    for target, view in C.BCBW_MODEL_VIEWS:
        if target != "nemotron3nano4b":
            other = yaml.safe_load((REPO / C.bcbw_config_path(target, view)).read_text())
            assert MULTI_LAYER_ARMS_KEY not in other, (target, view)   # all-layer / member fits need none
    # b64-capability-agentic-queue (2026-09-23): the all-models one-seed views — nemotron's s1 serves the same three LoRA
    # cells (the main bank's objects re-served) and inherits the whole declaration; its s1p serves ONE of them (the re-served
    # r=4 companion beside the composed PSR), so the block names that cell alone — the table is per TARGET, emitted per VIEW
    s1 = yaml.safe_load((REPO / C.bcbw_config_path("nemotron3nano4b", "s1")).read_text())
    assert s1[MULTI_LAYER_ARMS_KEY] == doc[MULTI_LAYER_ARMS_KEY]
    s1p = yaml.safe_load((REPO / C.bcbw_config_path("nemotron3nano4b", "s1p")).read_text())
    assert s1p[MULTI_LAYER_ARMS_KEY] == {"constv2-mix-full__lora_r4_ce": [12, 17, 24, 32]}
    assert set(s1p[MULTI_LAYER_ARMS_KEY]) <= set(s1p["explicit_cells"]) and "constv2ctl2-full__lora_r4_ce" not in s1p["explicit_cells"]
    # a declaration naming a cell NO view of the target serves refuses at generation (a typo in the table); one that another
    # view serves is simply not emitted for a view that does not serve it
    with pytest.raises(SystemExit, match="names cells no view of the target serves"):
        G.multi_layer_block("nemotron3nano4b", ["none"])
    roster = G.load_roster()
    tc = G.target_served_cells(roster, "nemotron3nano4b")
    assert {"constv2-mix-full__lora_r4_ce", "constv2ctl2-full__lora_r4_ce", "constv2-mix-full__lora_r4_ce_minus_ctl"} <= set(tc)
    block = G.multi_layer_block("nemotron3nano4b", ["none", "constv2-mix-full__lora_r4_ce"], target_cells=tc)
    assert "constv2-mix-full__lora_r4_ce: [12, 17, 24, 32]" in block and "constv2ctl2-full__lora_r4_ce" not in block
    assert G.multi_layer_block("nemotron3nano4b", ["none"], target_cells=tc) == ""
    with pytest.raises(SystemExit, match="names cells no view of the target serves"):
        G.multi_layer_block("nemotron3nano4b", ["none"], target_cells=["none", "constv2-mix-full__lora_r4_ce"])


def test_the_header_records_every_object_pin_and_the_body_is_the_cluster_configs(G):
    roster, pins = G.load_roster(), G.load_pins()
    for dst, target, view in G.outputs(roster):
        text = dst.read_text()
        for cid, sha in G.pins_by_cell(target, view, roster=roster, pins=pins).items():
            assert sha and f"object_sha256 {sha}" in text, (target, view, cid)
        body_src = (REPO / roster["targets"][target]["views"][view]["config"]).read_text()
        body = body_src[body_src.index(G.BODY_MARKER):]
        # every body line but the marker and defenses_dir is verbatim
        ours = text[text.index(G.BODY_MARKER):].splitlines()
        theirs = body.splitlines()
        assert len(ours) == len(theirs)
        for a, b in zip(ours[1:], theirs[1:]):
            if a.startswith("defenses_dir:"):
                assert b.startswith("defenses_dir:")
                continue
            assert a == b


def test_zero_glm_footprint(G):
    assert not any("glm" in t.lower() for _, t, _ in G.outputs(G.load_roster()))
    assert not list((REPO / "configs" / "tinycap").glob("bcbw_*glm*")) and not list((REPO / "configs" / "tinycap").glob("full22_*"))
    assert "glm" not in SCRIPT.read_text().lower().replace("no glm pair", "").replace("zero glm footprint", "")
    for _, target, view in G.outputs(G.load_roster()):
        assert "glm" not in (REPO / C.bcbw_config_path(target, view)).read_text().lower()


def test_the_llama8b_views_come_from_the_shared_roster_and_list_its_three_objects_each(G):
    """The llama8b target is a block of the SHARED full22 roster (landed
    2026-09-12); this lane reads it like every other target — no supplement, no lane body copies. Each view lists
    the three staged objects (the base / itdoc fit, its control fit, the derived difference) behind `none` and the
    safety anchor, every one pinned; the `it` view's staged ids carry the llama8b
    `itdoc` / `derived_itdoc` prefixes, which the lane-kind reader admits by name; the bodies are the
    cluster configs the block names; the header names the shared roster alone."""
    shared = yaml.safe_load(G.ROSTER.read_text())
    assert "llama8b" in shared["targets"] and set(G.LANE_TARGETS) <= set(shared["targets"])
    roster = G.load_roster()
    assert roster == shared
    t = roster["targets"]["llama8b"]
    assert (t["model"], t["model_revision"]) == (C.model_row("llama8b").hf, C.model_row("llama8b").rev)
    for view, kinds in (("it", {"itdoc", "derived_itdoc"}), ("main", {"base", "derived"})):
        objects = G.object_cells(roster, "llama8b", view)
        assert len(objects) == 3 and "constv2-mix-full__lora_r4_ce_minus_ctl" in objects
        assert G.explicit_cells(roster, "llama8b", view) == [C.ROUTE_NONE_CELL, C.CELL_PROMPTED_SAFETY, *objects]   # no lane prompted extra
        staged = G.load_roster()["targets"]["llama8b"]["views"][view]["cells"]
        assert {G.lane_kind_of_staged_id(staged[c]) for c in objects} == kinds
        pins = G.pins_by_cell("llama8b", view, roster=roster)
        assert set(pins) == set(objects) and all(pins.values()), pins
        text = (REPO / C.bcbw_config_path("llama8b", view)).read_text()
        assert "from configs/sdfbase_full22_roster.yaml (target llama8b" in text and "supplement" not in text
        for c in objects:
            assert f"<- {staged[c]}" in text and f"object_sha256 {pins[c]}" in text
        body = REPO / t["views"][view]["config"]
        assert body.is_file() and body.name.startswith("sdfbase_full22_llama8b") and "bcbw_body" not in body.name
        doc = yaml.safe_load(text)
        assert (doc["model"], doc["model_revision"], doc["layer"], doc["max_len"]) == (t["model"], t["model_revision"], 15, 16384)
        assert MULTI_LAYER_ARMS_KEY not in doc                        # rank 4 on all 32 layers; 15 is a member (F9)
    assert G.pins_by_cell("llama8b", "it", roster=roster)["constv2-mix-full__lora_r4_ce_minus_ctl"] == "e85220c9e5d2b9aa4377fea7a9a590048c693118975be9f36ae579b0eeff6f13"
    assert G.pins_by_cell("llama8b", "main", roster=roster)["constv2-mix-full__lora_r4_ce_minus_ctl"] == "4375b67a0fb0621e9ee296041d0842cb24294112e18424a803e8e54e9538a740"
    assert G.lane_kind_of_staged_id("derived_itdoc__constv2-mix-full__lora_r4_ce_minus_ctl") == "derived_itdoc"
    # the batch-64 re-fit lineage (the staged-id rule, 2026-09-13): its own prefixes, admitted by name
    assert G.lane_kind_of_staged_id("itdoc64__constv2-mix-full__lora_r4_ce") == "itdoc64"
    assert G.lane_kind_of_staged_id("derived_itdoc64__constv2-mix-full__lora_r4_ce_minus_ctl") == "derived_itdoc64"
    # the three-seed lineage's prefixes and the Arditi extra's, admitted by name
    for s_ in ("1", "2", "3"):
        assert G.lane_kind_of_staged_id(f"base64s{s_}__constv2-mix-full__psr_ce_all") == f"base64s{s_}"
        assert G.lane_kind_of_staged_id(f"derived_base64s{s_}__constv2-mix-full__lora_r4_ce_minus_ctl") == f"derived_base64s{s_}"
        assert G.lane_kind_of_staged_id(f"paired_eb64s{s_}__constv2-mix-full__psr_ce_all_minus_ctl") == f"paired_eb64s{s_}"
    assert G.lane_kind_of_staged_id("ar_c0p3__arditi_allpos") == "ar_c0p3"
    with pytest.raises(SystemExit, match="no lane prefix"):
        G.lane_kind_of_staged_id("base64s4__x")                  # a seed nobody fit refuses
    with pytest.raises(SystemExit, match="no lane prefix"):
        G.lane_kind_of_staged_id("derived_it__x")
    with pytest.raises(SystemExit, match="no lane prefix"):
        G.lane_kind_of_staged_id("itdoc8__x")                    # a lineage nobody minted refuses
    # the retired supplement and body copies are gone, and nothing in the generator reads them
    assert not (REPO / "configs" / "capeval" / "bcb_bfcl_200_widen_full22_supplement.yaml").exists()
    assert not list((REPO / "configs" / "tinycap").glob("bcbw_body_llama8b_*"))
    assert not hasattr(G, "load_supplement") and not hasattr(G, "supplied_by_supplement")


def test_a_shared_roster_lacking_a_lane_target_refuses_by_name(G, tmp_path):
    """Loud, never a quiet fallback: a roster without one of the lane's targets would fail deep inside f22_roster
    after the caller has moved on; `load_roster` refuses it naming the target."""
    import copy
    shared = yaml.safe_load(G.ROSTER.read_text())
    for target in G.LANE_TARGETS:
        without = copy.deepcopy(shared); without["targets"].pop(target)
        rc = tmp_path / f"roster_without_{target}.yaml"; rc.write_text(yaml.safe_dump(without, sort_keys=False))
        with pytest.raises(SystemExit, match=f"lacks the lane target.*{target}"):
            G.load_roster(rc)
    rc = tmp_path / "roster_copy.yaml"; rc.write_text(G.ROSTER.read_text())
    assert G.load_roster(rc) == shared


def test_a_missing_pins_file_refuses_instead_of_reading_as_unpinned(G, tmp_path):
    """Code review: `load_pins` returned `{}` for a missing file, so every cell's header
    printed UNPINNED and a pin-less config could be generated — it refuses by name now."""
    with pytest.raises(FileNotFoundError, match="full22 object pins file is missing"):
        G.load_pins(tmp_path / "nope.json")
    assert G.load_pins()                                                # the checked-in file still loads
    with pytest.raises(FileNotFoundError):
        G.pins_by_cell("gemma4e4b", "it", roster=G.load_roster(), pins=G.load_pins(tmp_path / "nope.json"))


def test_the_seed_views_list_the_batch64_lineage_and_the_arditi_extra(G):
    """The six seed configs list the shared roster's view behind `none` and
    the prompted arms, every object pinned under its seed's prefix; the Arditi refusal-steer is a LANE EXTRA whose lane
    entries are RUNNABLE, so it is listed LAST in the s<s> views' `explicit_cells` (pinned through ar_c0p3__arditi_allpos)
    and the header names it a lane extra; the paired views never name it."""
    roster, pins = G.load_roster(), G.load_pins()
    for s_ in ("1", "2", "3"):
        for view, prefix in ((f"s{s_}", [C.ROUTE_NONE_CELL, C.CELL_PROMPTED_SAFETY, "prompted__si_constitution"]),
                             (f"s{s_}p", [C.ROUTE_NONE_CELL, C.CELL_PROMPTED_SAFETY])):
            path = REPO / C.bcbw_config_path("qwen35_4b", view)
            text = path.read_text()
            doc = yaml.safe_load(text)
            row = C.model_row(f"qwen35_4b-f22{view}")
            assert doc["defenses_dir"] == row.objects_dir == f"/vol/objects/sdfbase/full22/qwen35_4b/{view}"
            assert (doc["model"], doc["model_revision"], doc["layer"]) == ("Qwen/Qwen3.5-4B", row.rev, 15)
            cells = doc["explicit_cells"]
            assert cells[:len(prefix)] == prefix
            staged = G.staged_ids_by_cell(roster, "qwen35_4b", view)
            assert set(cells[len(prefix):]) == set(staged)
            for cid, sid in staged.items():
                assert sid.split("__")[0] in (f"base64s{s_}", f"derived_base64s{s_}", f"paired_eb64s{s_}", "ar_c0p3"), (view, cid, sid)
                assert f"<- {sid}" in text and f"object_sha256 {pins['qwen35_4b'][sid]['object_sha256']}" in text
            assert G.held_extra_cells(roster, "qwen35_4b", view) == [] and "HELD LANE EXTRAS" not in text
            if view == f"s{s_}":
                assert cells[-1] == "arditi_allpos" and staged["arditi_allpos"] == "ar_c0p3__arditi_allpos"
                assert "(a LANE EXTRA: the lane roster's runnable entry; staged INSIDE this bank by that lane)" in text
                assert f"object_sha256 {pins['qwen35_4b']['ar_c0p3__arditi_allpos']['object_sha256']}" in text
            else:
                assert "arditi_allpos" not in cells and "constv2-mix-full__psr_ce_all_minus_ctl" in cells and "paired_eb64s" in text
            assert C.is_full22_row(row) and MULTI_LAYER_ARMS_KEY not in doc
    assert pins["qwen35_4b"]["ar_c0p3__arditi_allpos"]["object_sha256"] == "ba2212f0c2f9ffef2997bdd24831182052560cf980054bcee9cf95ea0db085b2"
    assert pins["qwen35_4b"]["ar_c0p3__arditi_allpos"]["kind"] == "vector_all" and pins["qwen35_4b"]["ar_c0p3__arditi_allpos"]["identity_sidecar"] == "fit_provenance.json"


def test_a_lane_extra_is_held_out_of_the_served_list_while_its_lane_entry_is_not_runnable(G):
    """The HOLD path: a DECLARED lane extra whose lane entry is queued is kept OUT of `explicit_cells`
    and the pins (the bank need not hold it) and declared `HELD EXTRA` with the lane roster's standing; the shipped roster's
    current state lists it LAST and pins it through `ar_c0p3__arditi_allpos`; a runnable extra without a staged id
    refuses by name; a declared extra naming a non-fitted cell refuses; only DECLARED extras count (the gemma waves'
    queued `safety_lora` placeholder declares no kind and is not one); a view whose lane waves name no extra is unchanged."""
    import copy
    from sisteer.capeval.tinycap import widen_roster as W
    roster, pins = G.load_roster(), G.load_pins()
    shipped = W.load_roster(G.LANE_ROSTER)
    sha = pins["qwen35_4b"]["ar_c0p3__arditi_allpos"]["object_sha256"]
    assert shipped["waves"]["qwen35_4b_s1_arditi"]["entries"]["arditi_allpos"]["kind"] == G.LANE_EXTRA_KIND == "lane_extra"
    lane = copy.deepcopy(shipped)
    for eid in ("none", "arditi_allpos"):
        lane["waves"]["qwen35_4b_s1_arditi"]["entries"][eid].update(status="queued", reason="synthetic hold", unblock="synthetic")
    lane["waves"]["qwen35_4b_s1_arditi"]["entries"]["arditi_allpos"].pop("object")
    assert G.lane_extra_cells(roster, "qwen35_4b", "s1", lane_roster=lane) == {} and G.held_extra_cells(roster, "qwen35_4b", "s1", lane_roster=lane) == ["arditi_allpos"]
    held_cells = G.explicit_cells(roster, "qwen35_4b", "s1", lane_roster=lane)
    assert "arditi_allpos" not in held_cells and held_cells == G.explicit_cells(roster, "qwen35_4b", "s1")[:-1]
    assert "arditi_allpos" not in G.pins_by_cell("qwen35_4b", "s1", roster=roster, pins=pins, lane_roster=lane)
    assert G.lane_prompted_status("qwen35_4b", "s1", "arditi_allpos", lane_roster=lane) == "queued (wave qwen35_4b_s1_arditi, entry arditi_allpos)"
    # the shipped (runnable) state: listed LAST, pinned; the paired view never names the extra; s2 is independent of s1
    assert G.lane_extra_cells(roster, "qwen35_4b", "s1") == {"arditi_allpos": "ar_c0p3__arditi_allpos"} and G.held_extra_cells(roster, "qwen35_4b", "s1") == []
    assert G.explicit_cells(roster, "qwen35_4b", "s1")[-1] == "arditi_allpos" and G.pins_by_cell("qwen35_4b", "s1", roster=roster, pins=pins)["arditi_allpos"] == sha
    assert G.lane_extra_cells(roster, "qwen35_4b", "s2", lane_roster=lane) == {"arditi_allpos": "ar_c0p3__arditi_allpos"}
    assert G.lane_extra_cells(roster, "qwen35_4b", "s1p") == {} and G.held_extra_cells(roster, "qwen35_4b", "s1p") == []
    # a runnable extra with no staged id refuses at generation
    bad = copy.deepcopy(shipped)
    bad["waves"]["qwen35_4b_s1_arditi"]["entries"]["arditi_allpos"]["object"].pop("staged_id")
    with pytest.raises(SystemExit, match="without an object.staged_id"):
        G.lane_extra_cells(roster, "qwen35_4b", "s1", lane_roster=bad)
    # only DECLARED extras count; the gemma configs carry no HELD extras
    assert set(G.lane_fitted_entries("qwen35_4b", "s1")) == {"arditi_allpos"}
    assert G.held_extra_cells(roster, "gemma4e4b", "it") == [] and "HELD LANE EXTRAS" not in (REPO / C.bcbw_config_path("gemma4e4b", "it")).read_text()
    # a declared extra naming a non-fitted cell refuses
    bad = copy.deepcopy(shipped)
    bad["waves"]["qwen35_4b_s1_arditi"]["entries"]["arditi_allpos"]["cell_id"] = "prompted__si_constitution"
    with pytest.raises(SystemExit, match="names no fitted cell"):
        G.lane_fitted_entries("qwen35_4b", "s1", lane_roster=bad)


def test_a_wave_only_the_full_roster_carries_joins_the_union_and_the_200_roster_wins_a_same_named_wave(G, tmp_path):
    """The union branch of load_lane_rosters (`{**full.waves, **two_hundred.waves}`): on the shipped rosters every full-roster wave has a
    200 sibling, so the branch is witnessed on a SYNTHETIC pair — the full copy carries one extra wave (a copy of a 200 wave that runs
    both prompted arms, re-pointed at a row no 200 wave serves with the constitution arm) and a marked copy of a same-named wave. The
    union carries the extra wave, the 200's copy of the shared one, and lane_prompted_cells reads the extra wave's constitution arm only
    through the union."""
    from sisteer.capeval.tinycap import widen_roster as W
    two_hundred = W.load_roster(G.LANE_ROSTER); full = W.load_roster(G.LANE_ROSTER_FULL)
    donor = next(w for w in W.waves(two_hundred) if "prompted__si_constitution" in W.runnable_cells(two_hundred, w))
    target, view = next((t, v) for t, v in C.BCBW_MODEL_VIEWS if "prompted__si_constitution" not in G.lane_prompted_cells(t, v))
    shared = next(w for w in W.waves(full) if w in two_hundred["waves"])
    extra = copy.deepcopy(two_hundred["waves"][donor]); extra["model_row"] = C.BCBW_SLUG_OF[(target, view)]
    full_copy = copy.deepcopy(full); full_copy["waves"]["synthetic_full_only"] = extra
    full_copy["waves"][shared] = {**copy.deepcopy(full_copy["waves"][shared]), "synthetic_marker": "the full roster's copy"}
    p200 = tmp_path / "bcb_bfcl_200_widen_roster.yaml"; p200.write_text(yaml.safe_dump(two_hundred, sort_keys=False))
    pfull = tmp_path / "bcb_bfcl_full_roster.yaml"; pfull.write_text(yaml.safe_dump(full_copy, sort_keys=False))
    union = G.load_lane_rosters(path_200=p200, path_full=pfull)
    assert "synthetic_full_only" in union["waves"] and union["waves"]["synthetic_full_only"]["model_row"] == C.BCBW_SLUG_OF[(target, view)]
    assert "synthetic_marker" not in union["waves"][shared] and union["waves"][shared] == two_hundred["waves"][shared]   # the 200 wins
    assert set(union["waves"]) == set(W.waves(two_hundred)) | {"synthetic_full_only"}
    assert "prompted__si_constitution" in G.lane_prompted_cells(target, view, lane_roster=union), (target, view)
    alone = G.load_lane_rosters(path_200=p200, path_full=tmp_path / "absent.yaml")                 # no full roster beside the 200: the 200 alone
    assert set(alone["waves"]) == set(W.waves(two_hundred)) and "prompted__si_constitution" not in G.lane_prompted_cells(target, view, lane_roster=alone)


# ----------------------------------------------------------------------------------------------------------------------------- #
# the FRONT-DOOR scope: its config is GENERATED                                                                                     #
# ----------------------------------------------------------------------------------------------------------------------------- #

FD_KEY = "qwen35_4b_demo"


def _frontdoor_tree(tmp_path: Path, G, *, drop=()) -> Path:
    """A tmp tree carrying the front door's inputs: the shared roster + the demo key's block (lib/models.py roster-block, the block 05 prints),
    the committed pins (which hold the demo block), a synthetic battery body under the marker; `drop` names inputs to leave out."""
    M = G.models_module()
    view = M.baseline_view(FD_KEY)
    r = tmp_path / "tree"; (r / "configs").mkdir(parents=True)
    if "roster" not in drop:
        text = (REPO / "configs" / "sdfbase_full22_roster.yaml").read_text()
        # the committed roster carries the front door's block (05 --step roster pasted it, LAST among the targets): take it out, so this helper
        # alone decides whether the tree's roster carries one (and what it says)
        marker = f"\n  {FD_KEY}:\n"
        assert marker in text, "the committed roster carries the front door's block"
        head, tail = text[:text.index(marker)], text[text.index(marker):]
        assert list(yaml.safe_load("targets:" + tail)["targets"]) == [FD_KEY], "the front door's block is the roster's last target"
        text = head + "\n"
        if "block" not in drop:
            text = text.replace("\ntargets:\n", "\ntargets:\n" + M.roster_block(FD_KEY, [view]), 1)
        (r / "configs" / "sdfbase_full22_roster.yaml").write_text(text)
    if "pins" not in drop:
        pins = json.loads((REPO / "configs" / "sdfbase_full22_pins.json").read_text())
        if "pins_block" in drop:
            pins.pop(FD_KEY, None)
        if "one_pin" in drop:
            pins[FD_KEY].pop("base__constv2ctl2-full__lora_r4_ce")
        (r / "configs" / "sdfbase_full22_pins.json").write_text(json.dumps(pins, indent=1, sort_keys=True))
    if "body" not in drop:
        (r / "configs" / f"sdfbase_full22_{FD_KEY}_{view}.yaml").write_text(
            "# sdfbase_full22_qwen35_4b_demo body — synthetic header\n# -- everything below: sdfbase_transfer_qwen35_4b_r3.yaml's data, unchanged, EXCEPT the keys the header names --\n"
            f"defenses_dir: out/full22/{FD_KEY}/cells_view/{view}\nmodel: Qwen/Qwen3.5-4B\nmodel_revision: 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a\nlayer: 15\nbon:\n  n_max: 75\n")
    return r


def test_the_front_door_scope_renders_the_tables_ordered_tokens_the_rows_bank_and_the_body_from_its_own_sources(G, tmp_path):
    """The front door's config is a parameterisation of the render with its OWN row source — the models table's launcher
    tokens (ordered: none, then the wave's cells in the table's order — ten in all) held equal to the roster block's enumeration,
    `defenses_dir` the row's own bank (outside the full22 root), the battery body byte-for-byte below the marker, every object cell's pin in the
    header, the header naming the front door; never the lane rosters (`prompted__si_constitution` would be held out) or BCBW_SLUG_OF."""
    M = G.models_module()
    view = M.baseline_view(FD_KEY)
    tree = _frontdoor_tree(tmp_path, G)
    text = G.render_frontdoor_config(FD_KEY, view, repo=tree)
    doc = yaml.safe_load(text)
    tokens = [t for _c, t in M.launcher_cells(FD_KEY, view)]
    assert doc["explicit_cells"] == tokens == ["none", "prompted__safety_refusal", "prompted__si_constitution", "arditi_allpos", "constv2ctl2-full__lora_r4_ce",
                                                "constv2-mix-full__lora_r4_ce", "constv2-mix-full__lora_r4_ce_minus_ctl", "constv2ctl2-full__psr_ce_all",
                                                "constv2-mix-full__psr_ce_all", "constv2-mix-full__psr_ce_all_minus_ctl"]
    assert doc["defenses_dir"] == C.model_row(FD_KEY).objects_dir == f"{C.VOL_OBJECTS_SDFBASE}/{FD_KEY}" and not doc["defenses_dir"].startswith(C.VOL_OBJECTS_FULL22)
    assert doc["model"] == "Qwen/Qwen3.5-4B" and doc["model_revision"] == "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a" and doc["bon"] == {"n_max": 75} and doc["layer"] == 15
    assert "si_ids" not in doc and "class_a_arms" not in doc and MULTI_LAYER_ARMS_KEY not in doc
    body = text[text.index(G.BODY_MARKER):]
    assert body.splitlines()[0] == f"# -- everything below: sdfbase_full22_{FD_KEY}_{view}.yaml's data, byte-for-byte, EXCEPT defenses_dir --"
    assert body.count("\ndefenses_dir: ") == 1 and "\nmodel: Qwen/Qwen3.5-4B\n" in body
    head = text[:text.index(G.BODY_MARKER)]
    assert "FRONT DOOR" in head and f"scripts/bcbw_configs.py --front-door {FD_KEY}" in head and f'`contracts.MODELS["{FD_KEY}"].cfg`' in head and "bcb-bfcl-200-widen lane" not in head
    pins = json.loads((tree / "configs" / "sdfbase_full22_pins.json").read_text())[FD_KEY]
    roster = yaml.safe_load((tree / "configs" / "sdfbase_full22_roster.yaml").read_text())
    for cid, sid in G.staged_cells(roster, FD_KEY, view).items():
        assert f"{cid:<48} lane {G.lane_kind_of_staged_id(sid):<8} <- {sid}" in head and pins[sid]["object_sha256"] in head, cid
    assert "UNPINNED" not in head and G.lane_kind_of_staged_id("arditi__ar_c0p3__arditi_allpos") == "arditi" and "arditi" in G.LANE_KINDS
    assert "#   table   reproduce/models.yaml" in head and "#   pins    configs/sdfbase_full22_pins.json" in head
    assert yaml.safe_load(text) == yaml.safe_load(G.render_frontdoor_config(FD_KEY, None, repo=tree))      # the default view is the key's one wave
    # the row's cfg names the front-door file (the flip landed with the committed file): no note; a row pointed elsewhere gets a NOTE, never a refusal
    assert G.frontdoor_cfg_note(FD_KEY, view) is None and C.model_row(FD_KEY).cfg == C.frontdoor_config_path(FD_KEY, view)
    note = G.frontdoor_cfg_note(FD_KEY, "other_wave")
    assert note is not None and C.frontdoor_config_path(FD_KEY, "other_wave") in note and C.model_row(FD_KEY).cfg in note
    assert C.frontdoor_config_path(FD_KEY, view) == f"configs/tinycap/frontdoor_{FD_KEY}_{view}.yaml"


def test_a_dose_views_front_door_config_serves_the_views_copy_from_its_own_row_and_bank(G, tmp_path):
    """Since 2026-10-12 a DOSE view renders like the wave, over its OWN
    registry row — `launcher_slug(key, view)` = `<key>-<view>` (the paper's dose rows' shape) — so `defenses_dir` is the view's sub-bank of
    dose-scaled copies under the key's root, `explicit_cells` = [none, the object cell's cell id: the SAME token as the wave's parent — the
    bank, not the token, carries the dose], the header names the view's row and the generating command, and the pin is the key's block's
    `d<tok>__<cell_id>` entry (the committed copy's id; the lane kind is the dose lane, admitted by shape). The committed file is byte-identical."""
    M = G.models_module()
    wave = M.baseline_view(FD_KEY)
    m = M.model(FD_KEY)
    dose_views = [v for v, vd in m["views"].items() if "dose" in (vd or {})]
    assert dose_views == ["d0p75", "d0p75_relu"]
    pins = json.loads((REPO / "configs" / "sdfbase_full22_pins.json").read_text())[FD_KEY]
    for view in dose_views:
        slug = M.launcher_slug(FD_KEY, view)
        row = C.model_row(slug)
        text = G.render_frontdoor_config(FD_KEY, view)
        assert (REPO / C.frontdoor_config_path(FD_KEY, view)).read_text(encoding="utf-8") == text, view
        doc = yaml.safe_load(text)
        cell = next(c for c in m["views"][view]["cells"] if c != "none")
        tok = M.roster_cell_id(m, cell)
        assert doc["explicit_cells"] == ["none", tok] == [t for _c, t in M.launcher_cells(FD_KEY, view)]
        assert tok in [t for _c, t in M.launcher_cells(FD_KEY, wave)]                      # the wave's parent carries the same cell id
        assert doc["defenses_dir"] == row.objects_dir == f"{C.model_row(FD_KEY).objects_dir}/{view}" and not doc["defenses_dir"].startswith(C.VOL_OBJECTS_FULL22)
        sid = M.staged_id(m, cell, view)
        assert sid.startswith("d0p75__") and G.lane_kind_of_staged_id(sid) == "d0p75" and G.DOSE_LANE_RE.match("d0p75")
        head = text[:text.index(G.BODY_MARKER)]
        assert f"registry row `{slug}` reads (`contracts.MODELS[\"{slug}\"].cfg`" in head and f"--front-door {FD_KEY} --view {view}" in head
        assert f"--model {FD_KEY} --view {view} launches" in head and f"{tok:<48} lane d0p75    <- {sid}" in head and pins[sid]["object_sha256"] in head
        assert G.frontdoor_cfg_note(FD_KEY, view) is None
    for bad in ("d0p75__x", "dm0p5__x", "d1p25__x", "d0p75l__x"):
        assert G.lane_kind_of_staged_id(bad) == bad.split("__")[0]
    with pytest.raises(SystemExit, match="is not a dose lane"):
        G.lane_kind_of_staged_id("dose__x")
    with pytest.raises(SystemExit, match="is not a dose lane"):
        G.lane_kind_of_staged_id("x0p75__x")


def test_the_front_door_scope_refuses_by_name_each_missing_input_and_a_block_that_disagrees_with_the_table(G, tmp_path, monkeypatch, capsys):
    M = G.models_module()
    view = M.baseline_view(FD_KEY)
    cases = {("block",): "carries no block for the front-door target", ("roster",): "no shared roster at", ("pins_block",): f"carries no {FD_KEY} block",
             ("pins",): "no pins file at", ("body",): "no battery body at", ("one_pin",): r"unpinned object cell\(s\)"}
    for drop, needle in cases.items():
        tree = _frontdoor_tree(tmp_path / "_".join(drop), G, drop=drop)
        with pytest.raises(G.FrontDoorRefusal, match=needle):
            G.render_frontdoor_config(FD_KEY, view, repo=tree)
    # a body without the marker; a view the key does not declare; a key the table lacks; a roster block whose cells are not the table's tokens
    tree = _frontdoor_tree(tmp_path / "marker", G)
    (tree / "configs" / f"sdfbase_full22_{FD_KEY}_{view}.yaml").write_text("defenses_dir: x\nmodel: m\n")
    with pytest.raises(G.FrontDoorRefusal, match="has no '# -- everything below:' marker"):
        G.render_frontdoor_config(FD_KEY, view, repo=tree)
    with pytest.raises(G.FrontDoorRefusal, match="is not a view of"):
        G.render_frontdoor_config(FD_KEY, "nope", repo=_frontdoor_tree(tmp_path / "view", G))
    with pytest.raises(G.FrontDoorRefusal, match="has no row 'no_such_key'"):
        G.render_frontdoor_config("no_such_key", None, repo=_frontdoor_tree(tmp_path / "key", G))
    tree = _frontdoor_tree(tmp_path / "order", G)
    rp = tree / "configs" / "sdfbase_full22_roster.yaml"
    roster = yaml.safe_load(rp.read_text())
    cells = roster["targets"][FD_KEY]["views"][view]["cells"]
    roster["targets"][FD_KEY]["views"][view]["cells"] = dict(reversed(list(cells.items())))
    rp.write_text(yaml.safe_dump(roster, sort_keys=False))
    with pytest.raises(G.FrontDoorRefusal, match="are not the models table's launcher tokens"):
        G.render_frontdoor_config(FD_KEY, view, repo=tree)
    # the CLI's refusal, driven through an ISOLATED tree whose roster carries no demo block (the committed tree carries the block, the pins
    # and the body, so on it the render succeeds — the --check test below): `--front-door` exits 2 BY NAME, on stderr, and writes nothing
    # anywhere — not under the isolated tree it rendered from, and not over the committed file
    tree = _frontdoor_tree(tmp_path / "cli", G, drop=("block",))
    committed = REPO / C.frontdoor_config_path(FD_KEY, view)
    before = committed.read_bytes()
    monkeypatch.setattr(G, "REPO", tree)                                                              # main()'s output root
    monkeypatch.setattr(G, "render_frontdoor_config", functools.partial(G.render_frontdoor_config, repo=tree))   # main()'s input root
    with pytest.raises(SystemExit) as e:
        G.main(["--front-door", FD_KEY])
    assert e.value.code == 2 and not (tree / "configs" / "tinycap").exists()
    assert "carries no block for the front-door target" in capsys.readouterr().err
    monkeypatch.undo()
    assert committed.read_bytes() == before


def test_check_keeps_the_nineteen_lane_files_and_covers_a_front_door_file_once_it_exists(G, tmp_path, monkeypatch, capsys):
    """`--check` is the lane's 19 byte-identity checks as before, plus every front-door output: the committed one is re-rendered and held
    byte-identical (20 outputs on this tree), one not generated yet is reported `absent` (never a drift), a drifting one fails the check. The
    absent / drift / refusal branches are exercised with the render and the outputs list stood in."""
    M = G.models_module()
    view = M.baseline_view(FD_KEY)
    committed = REPO / C.frontdoor_config_path(FD_KEY, view)
    # demo-import (2026-10-12): every PHASE view of the front-door key is an output — the wave first, then its dose views, each over the
    # view's own registry row (launcher_slug) and committed beside the wave's file
    phase_views = M.phase_views(FD_KEY)
    assert phase_views[0] == view and len(phase_views) == 3
    assert G.frontdoor_keys() == [(FD_KEY, v) for v in phase_views]
    assert G.frontdoor_outputs() == [(REPO / C.frontdoor_config_path(FD_KEY, v), FD_KEY, v) for v in phase_views] and committed.is_file()
    for _p, _k, v in G.frontdoor_outputs():
        assert _p.is_file() and G.frontdoor_cfg_note(FD_KEY, v) is None, (v, "the view's row names its generated file")
        assert yaml.safe_load(_p.read_text())["defenses_dir"] == C.model_row(M.launcher_slug(FD_KEY, v)).objects_dir
    assert G.main(["--check"]) == 0
    out = capsys.readouterr().out
    assert out.count("ok: ") == 22 and f"ok: {committed}" in out and "absent:" not in out and "DRIFT" not in out
    # the front-door CLI itself, in --check, on the committed tree: the one output re-rendered and held, and no cfg note (the row names the file)
    assert G.main(["--front-door", FD_KEY, "--check"]) == 0
    assert capsys.readouterr().out.strip() == f"ok: {committed}"
    fd = tmp_path / "frontdoor.yaml"
    monkeypatch.setattr(G, "frontdoor_outputs", lambda table=None: [(fd, FD_KEY, view)])
    assert G.main(["--check"]) == 0                                                        # not generated yet: absent, never a drift
    out = capsys.readouterr().out
    assert out.count("ok: ") == 19 and f"absent: {fd}" in out and "DRIFT" not in out and f"--front-door {FD_KEY}" in out
    monkeypatch.setattr(G, "frontdoor_outputs", lambda table=None: [(fd, FD_KEY, view)])
    monkeypatch.setattr(G, "render_frontdoor_config", lambda key, view=None, **kw: "explicit_cells:\n- none\n")
    fd.write_text("explicit_cells:\n- none\n")
    assert G.main(["--check"]) == 0
    out = capsys.readouterr().out
    assert out.count("ok: ") == 20 and f"ok: {fd}" in out and "absent:" not in out
    fd.write_text("explicit_cells:\n- other\n")
    assert G.main(["--check"]) == 1 and f"DRIFT: {fd}" in capsys.readouterr().out
    monkeypatch.setattr(G, "render_frontdoor_config", lambda key, view=None, **kw: (_ for _ in ()).throw(G.FrontDoorRefusal(key, view, "an input vanished")))
    assert G.main(["--check"]) == 1 and "its render refuses" in capsys.readouterr().out
    # --view goes with --front-door
    with pytest.raises(SystemExit):
        G.main(["--view", view])
