"""The qwen_seedrep wave's Arditi RECORD as the fifth launchable GSM8K document (`arditi_seedrep`): the Arditi
refusal steer on GSM8K exactly as the others — vendor thinking-card decode, thinking ON, the 8,192 -> 16,384 ladder, seed bases
1..3, cluster A. The record (g8k_seedrep_arditi.py) re-keys the `arditi_allpos` matrix's one EXTERNAL row onto the seed views; until
now only the lane's Modal launcher read it. Pins: the registration (both tables, the sbatch case lists), the schema / role constants
mirrored in stdlib-only g8k_cells, the loader's schema-2 reading of the record, each row's launcher facts (the replicate seed policy at
its seed's base, the steer_allpos tuple pair, the external bank view + pin, its OWN overlay pair with explicit_cells [none,
arditi_allpos] cut from the seed view's campaign config, no companion), the launchability assertion, and the refusals a record row
earns when its seed facts are wrong."""
from __future__ import annotations

import copy
import json
import re
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "sdfbase" / "gsm8k"))

import g8k_cells as M  # noqa: E402
import g8k_seedrep_arditi as AR  # noqa: E402

NAME = "arditi_seedrep"
RECORD = REPO / "configs/sdfbase/gsm8k/arditi_rows_qwen_seedrep.json"
OBJECT_SHA = "ba2212f0c2f9ffef2997bdd24831182052560cf980054bcee9cf95ea0db085b2"
SBATCHES = ("gsm8k_thinking.sbatch", "ladder_plan.sbatch", "smoke_not_available.sbatch", "dry_run_prod.sbatch")


def test_the_record_is_registered_as_a_launchable_document_in_both_tables_and_every_sbatch_case_list():
    assert M.MATRIX_ARDITI_SEEDREP_NAME == NAME and M.MATRICES[NAME] == RECORD and RECORD.is_file()
    assert M.VIEWS_BY_MATRIX[NAME] == M.VIEWS_BY_MATRIX[M.MATRIX_QWEN_SEEDREP_NAME]       # derived from the seedrep views document
    assert set(M.MATRICES) == set(M.VIEWS_BY_MATRIX)
    assert M.matrix_name_of(RECORD) == NAME and M.resolve_matrix(NAME) == RECORD
    # the stdlib-only mirrors equal the record generator's own constants
    assert M.RECORD_SCHEMA_ARDITI_SEEDREP == AR.SCHEMA == "g8k_seedrep_arditi_rows/1" and M.ROLE_ARDITI_UNPAIRED == AR.ROLE
    for name in SBATCHES:
        src = (REPO / "scripts/sdfbase/gsm8k" / name).read_text()
        m = re.search(r'case "\$G8K_MATRIX" in ([a-z_|0-9]+)\)', src)
        assert m and sorted(m.group(1).split("|")) == sorted(M.MATRICES) and NAME in m.group(1).split("|"), name


def test_the_record_loads_with_schema_2_semantics_and_its_three_rows_are_launchable_seed_rows_of_the_arditi_arm():
    doc = M.load_matrix(RECORD)
    assert M.doc_is_schema2(doc) and doc["schema"] == M.RECORD_SCHEMA_ARDITI_SEEDREP
    assert M.wave_root_rel_of(doc) == "exp/gsm8k_paired200_paper_objects_qwen_seedrep" and doc["trigger_threshold"] == M.TRIGGER_THRESHOLD
    assert doc["inspect_seed_policy"] == M.INSPECT_SEED_POLICY_REPLICATE and doc["inspect_seed_base_by_view"] == {"s1": 1, "s2": 2, "s3": 3}
    assert doc[M.GEOMETRY_CLASS_KEY] == "common_f22" and M.STANDUP_COMPANIONS_KEY not in doc
    assert [r["view"] for r in doc["rows"]] == ["s1", "s2", "s3"] and [r["index"] for r in doc["rows"]] == [0, 1, 2]
    for r in doc["rows"]:
        s = int(r["view"][1:])
        assert r["cell"] == M.CELL_ARDITI_ALLPOS and r["role"] == M.ROLE_ARDITI_UNPAIRED and r["object_source"] == M.OBJECT_SOURCE_EXTERNAL
        assert r["status"] == M.STATUS_LAUNCH and r["fitted"] and not r["declared"] and r["own_overlay"] is True
        assert r["object_sha256"] == OBJECT_SHA and r["staged_id"] == "ar_c0p3__arditi_allpos" and r["bank_view"] == "ar_c0p3"
        assert r["bank_view_rel"]["primary"] == "exp/objects/arditi_allpos/qwen35_4b/ladder/cells_view/ar_c0p3"
        assert M.row_inspect_seed(r) == (M.INSPECT_SEED_POLICY_REPLICATE, s) and M.row_companions(r) == []
        assert not any(k.startswith("standup_companion_") for k in r)                       # a single-object engine carries no companion fields
        assert r["tuple_names"] == {"8192": f"gsm8k_paired200_qwen35_4b_steer_allpos_8192", "16384": f"gsm8k_paired200_qwen35_4b_steer_allpos_16384"}
        assert r["out_root"] == f"controls_qwen_seedrep/qwen35_4b_s{s}/arditi_allpos/thinking_card"
        assert r["rung_out_root"] == r["out_root"] + "__rung16384" and r["wave"] == "qwen_seedrep" and r["thinking_mode"] == "on"
        assert r["sampling_mode"] == "thinking_card" and r["sampling"]["temperature"] == 0.6 and r["sampling"]["top_p"] == 0.95
        assert M.external_row_problem(r) is None and M.inspect_seed_row_problem(r, doc) is None and M.geometry_class_row_problem(r, doc) is None
        M.assert_launchable(r, doc["rows"])                                                # admitted: a launch row of its own wave
        # the launcher facts the sbatch evals: the seed policy + base, the tuple, the external bank, the own overlay, the seedrep wave root
        sh = dict(line.split("=", 1) for line in M.row_sh(r, wave_root_rel=M.wave_root_rel_of(doc)).splitlines())
        assert sh["G8K_INSPECT_SEED_POLICY"] == "replicate" and sh["G8K_INSPECT_SEED_BASE"] == str(s)
        assert sh["G8K_TUPLE"] == "gsm8k_paired200_qwen35_4b_steer_allpos_8192" and sh["G8K_OBJECT_SOURCE"] == "external"
        assert sh["G8K_BANK_VIEW"] == "ar_c0p3" and sh["G8K_BANK_CELL_DIR"] == "arditi_allpos" and sh["G8K_OWN_OVERLAY"] == "1"
        assert sh["G8K_BANK_VIEW_REL_PRIMARY"] == "exp/objects/arditi_allpos/qwen35_4b/ladder/cells_view/ar_c0p3"
        assert sh["TAG"] == f"qwen35_4b_s{s}" and sh["G8K_WAVE_ROOT_REL"] == "exp/gsm8k_paired200_paper_objects_qwen_seedrep"
        assert sh["G8K_GEOMETRY_EVAL"] == "capability:gen_long_gsm8k_common_f22" and sh["G8K_GEOMETRY_L"] == "16640"
        assert sh["G8K_STANDUP_COMPANION_CELLS"] == "''" and sh["G8K_MAX_TOKENS_BY_TASK"] == "gsm8k_paired200=8192"
        assert sh["PIN"] == OBJECT_SHA
        # the row's OWN overlay pair, cut from the seed view's campaign config: explicit_cells [none, arditi_allpos]
        cfg = yaml.safe_load((REPO / f"configs/sdfbase/gsm8k/sdfbase_full22_qwen35_4b_s{s}.gsm8k_arditi_allpos.yaml").read_text())
        man = yaml.safe_load((REPO / f"configs/capeval/manifests/petri_full22_qwen35_4b_s{s}.gsm8k_arditi_allpos.yaml").read_text())
        assert cfg["explicit_cells"] == ["none", "arditi_allpos"] and man["petri"]["cells"] == ["none", "arditi_allpos"]
        assert cfg["model"] == "Qwen/Qwen3.5-4B" and cfg["model_revision"] == "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    assert AR.load_record() == doc                                                          # the generator's own loader agrees


def test_a_record_row_with_wrong_seed_facts_or_a_companion_refuses_by_name_and_the_arm_view_rule_still_holds_elsewhere():
    doc = M.load_matrix(RECORD); r = copy.deepcopy(doc["rows"][0])
    bad = copy.deepcopy(r); bad["view"] = "it"; bad["tag"] = "qwen35_4b_it"
    assert "on a seed view `s<s>`" in M.external_row_problem(bad)
    bad = copy.deepcopy(r); bad["inspect_seed_base"] = 2
    assert "replicate seed policy at base 1" in M.external_row_problem(bad)
    bad = copy.deepcopy(r); bad.pop("inspect_seed_policy"); bad.pop("inspect_seed_base")
    assert "replicate seed policy at base 1" in M.external_row_problem(bad)
    bad = copy.deepcopy(r); bad["standup_companion_cells"] = ["constv2-mix-full__lora_r4_ce"]
    assert "registers NO companion" in M.external_row_problem(bad)
    # without the record's role, an external arditi_allpos row on a seed view is the collision the arm-view rule guards
    plain = copy.deepcopy(r); plain.pop("role")
    assert "sits on the 'it' view" in M.external_row_problem(plain)
    # a record whose rows lack the role / the external source is refused by the loader
    broken = json.loads(RECORD.read_text()); broken["rows"][1]["role"] = "something_else"
    p = Path("/tmp/g8k_arditi_record_broken.json"); p.write_text(json.dumps(broken))
    with pytest.raises(SystemExit, match="EXTERNAL rows of role 'arditi_unpaired_value'; rows \\[1\\]"):
        M.load_matrix(p)
