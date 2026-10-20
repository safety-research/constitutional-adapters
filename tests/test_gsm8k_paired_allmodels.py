"""The all-model appendix GSM8K RECORD as the seventh launchable GSM8K document (`paired_allmodels`): the paired-write
difference ReLU on gemma4e4b it / llama8b it / nemotron3nano4b main, plus nemotron × the roster's constitution PSR object, at the appendix
column's instrument — default Inspect seed rule, one trial per model, thinking ON where the model has it, 8,192 + the 16,384 rung, the
primary cluster. Pins: the registration (both tables, the sbatch case lists), the schema / role constants mirrored in stdlib-only
g8k_cells, the loader's schema-2 reading of the record, each row's launcher facts (the model's arm + sampling mode, the composed arm's
native_paired tuple pair — none on llama's no-switch arm — the external bank view + pin, its OWN overlay pair with explicit_cells [none,
<cell>] cut from the panel view's campaign config, no seed policy, no companion), the launchability assertion, the tuple names' equality
with contracts' registry, and the refusals a record row earns when its panel view or seed facts are wrong."""
from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "sdfbase" / "gsm8k"))

import g8k_cells as M  # noqa: E402
import g8k_paired_allmodels as PA  # noqa: E402
from sisteer.capeval.tinycap import contracts as C  # noqa: E402

NAME = "paired_allmodels"
RECORD = REPO / "configs/sdfbase/gsm8k/paired_allmodels_rows.json"
COMPOSED = "constv2-mix-full__psr_ce_all_minus_ctl"
SBATCHES = ("gsm8k_thinking.sbatch", "ladder_plan.sbatch", "smoke_not_available.sbatch", "dry_run_prod.sbatch")
SHAS = {"gemma4e4b": "87863240eb52646b247315ce07b53ad601b7aff5a0207e528545f2c0564bba40",
        "llama8b": "5066333f8c10d57cd9992a33d8c26c7350a3bff6c1d401596f3a8fadb856d4e3",
        "nemotron3nano4b": "ad3e19228ed41ea861d4c4d6af3c322eb2a56e3c2dc1c3e197721fa89e96fd9f"}
CONSTITUTION_PIN = "81321803c479f3afc9e4738955e881d277a530f2bd5cfcbdcc9ad60e8a44f42f"


def test_the_record_is_registered_as_a_launchable_document_in_both_tables_and_every_sbatch_case_list():
    assert M.MATRIX_PAIRED_ALLMODELS_NAME == NAME and M.MATRICES[NAME] == RECORD and RECORD.is_file()
    assert M.VIEWS_BY_MATRIX[NAME] == M.VIEWS_BY_MATRIX[M.MATRIX_DEFAULT_NAME] == M.VIEWS_PATH   # derived from the published views document's frozen inputs
    assert set(M.MATRICES) == set(M.VIEWS_BY_MATRIX)
    assert M.matrix_name_of(RECORD) == NAME and M.resolve_matrix(NAME) == RECORD
    assert M.RECORD_SCHEMA_PAIRED_ALLMODELS == PA.SCHEMA == "g8k_paired_allmodels_rows/1" and M.ROLE_PAIRED_ALLMODELS == PA.ROLE == "paired_allmodels_value"
    assert M.RECORD_SCHEMAS == {M.RECORD_SCHEMA_ARDITI_SEEDREP: M.ROLE_ARDITI_UNPAIRED, M.RECORD_SCHEMA_PAIRED_ALLMODELS: M.ROLE_PAIRED_ALLMODELS}
    assert M.out_root_prefix_for(NAME) == "controls_paired_allmodels" and M.CELL_PAIRED_DIFFERENCE == COMPOSED
    for name in SBATCHES:
        src = (REPO / "scripts/sdfbase/gsm8k" / name).read_text()
        m = re.search(r'case "\$G8K_MATRIX" in ([a-z_|0-9]+)\)', src)
        assert m and sorted(m.group(1).split("|")) == sorted(M.MATRICES) and NAME in m.group(1).split("|"), name


def test_the_record_loads_with_schema_2_semantics_and_its_four_rows_are_launchable_appendix_rows():
    doc = M.load_matrix(RECORD)
    assert M.doc_is_schema2(doc) and doc["schema"] == M.RECORD_SCHEMA_PAIRED_ALLMODELS and doc["role"] == PA.ROLE
    assert M.wave_root_rel_of(doc) == "exp/gsm8k_paired200_paper_objects_paired_allmodels" and doc["trigger_threshold"] == M.TRIGGER_THRESHOLD
    assert doc["lane_source"] == "gsm8k_paired_allmodels" and doc["wave"] == NAME                       # the reader's source string
    assert M.GEOMETRY_CLASS_KEY not in doc and "inspect_seed_policy" not in doc and M.STANDUP_COMPANIONS_KEY not in doc
    default = M.load_matrix(M.MATRICES[M.MATRIX_DEFAULT_NAME])
    assert doc["source"] == default["source"]                                                              # the frozen inputs, copied verbatim
    assert [(r["model_key"], r["view"], r["cell"]) for r in doc["rows"]] == [
        ("nemotron3nano4b", "main", COMPOSED), ("nemotron3nano4b", "main", "constv2-mix-full__psr_ce_all"),
        ("gemma4e4b", "it", COMPOSED), ("llama8b", "it", COMPOSED)]
    assert [r["index"] for r in doc["rows"]] == [0, 1, 2, 3] and all(M.CELL_NONE != r["cell"] for r in doc["rows"])   # no `none` row: the panel pools the anchors
    c = M.counts(doc["rows"])
    assert (c["rows"], c["launched"], c["expected_refusals"], c["servable"], c["contrasts_computable"]) == (4, 4, 1, 3, 3)
    assert c["expected_refusal_rows"] == [{"index": 1, "job_id": "g8k-nemotron3nano4b-main-mix-psr"}]
    for r in doc["rows"]:
        key = r["model_key"]
        assert r["role"] == PA.ROLE and r["panel_view"] == r["view"] and r["status"] == M.STATUS_LAUNCH and r["own_overlay"] is True
        assert r["fitted"] and not r["declared"] and r["served_positions_expected"] == "all" and r["route_kind"] == "lens"
        assert r["wave"] == NAME and r["out_root"].startswith(f"controls_paired_allmodels/{r['tag']}/{r['cell_slug']}/") and r["rung_out_root"] == r["out_root"] + "__rung16384"
        assert r["thinking_mode"] == M.THINKING_MODE_BY_FAMILY[r["family"]] and r["sampling_mode"] == M.SAMPLING_MODE_BY_MODEL[key]
        assert r["cluster_constraint"] == "primary" and r["trigger_threshold"] == M.TRIGGER_THRESHOLD
        assert r["time_limit"] == (M.TIME_FITTED_LLAMA if key == "llama8b" else M.TIME_FITTED)                   # one --time class per lane
        assert M.class_tag_for(r["time_limit"], [r]) == ("fitted-llama" if key == "llama8b" else "fitted")     # the plan's -J tag resolves
        assert "inspect_seed_policy" not in r and M.row_companions(r) == [] and not any(k in r for k in M.STANDUP_COMPANION_FIELDS)
        assert M.inspect_seed_row_problem(r, doc) is None and M.companion_row_problem(r, doc) is None and M.geometry_class_row_problem(r, doc) is None
        M.assert_launchable(r, doc["rows"])
        sh = dict(line.split("=", 1) for line in M.row_sh(r, wave_root_rel=M.wave_root_rel_of(doc)).splitlines())
        assert sh["G8K_OWN_OVERLAY"] == "1" and sh["G8K_INSPECT_SEED_POLICY"] == "''" and sh["G8K_STANDUP_COMPANION_CELLS"] == "''"
        assert sh["G8K_WAVE_ROOT_REL"] == "exp/gsm8k_paired200_paper_objects_paired_allmodels" and sh["G8K_GEOMETRY_CLASS"] == "''"
        assert (sh["G8K_GEOMETRY_EVAL"], sh["G8K_GEOMETRY_L"]) == ("capability:gen_long_thinking", "10752")               # the short prompt class
        assert sh["G8K_THINKING"] == r["thinking_mode"] and sh["G8K_SAMPLING_MODE"] == r["sampling_mode"] and sh["G8K_CLUSTER_CONSTRAINT"] == "primary"
        if r["cell"] == COMPOSED:
            assert r["object_source"] == M.OBJECT_SOURCE_EXTERNAL and M.external_row_problem(r) is None
            assert r["object_sha256"] == SHAS[key] == sh["PIN"] and r["object_kind"] == "psr_paired_write" and r["bank_cell_dir"] == COMPOSED
            assert r["bank_view"] == f"paired_d1_{key}_s1" and r["bank_view_rel"] == {"primary": f"exp/objects/sdfbase/relu_paired/{key}/cells_view/paired_d1_{key}_s1", "overflow": None}
            assert sh["G8K_OBJECT_SOURCE"] == "external" and sh["G8K_BANK_VIEW_REL_PRIMARY"] == r["bank_view_rel"]["primary"] and sh["G8K_BANK_VIEW_REL_OVERFLOW"] == "''"
            assert r["object_parents"]["pos"]["cell"] == "constv2-mix-full__psr_ce_all" and r["object_parents"]["neg"]["cell"] == "constv2ctl2-full__psr_ce_all"
            assert all(len(r["object_parents"][p]["object_sha256"]) == 64 for p in ("pos", "neg"))
            if r["thinking_mode"] == M.THINKING_ON:
                assert r["tuple_names"] == {"8192": C.g8k_native_paired_tuple_name(key, 8192), "16384": C.g8k_native_paired_tuple_name(key, 16384)}
                assert sh["G8K_TUPLE"] == f"gsm8k_paired200_{key}_native_paired_8192" and key in C.G8K_NATIVE_PAIRED_MODELS
                assert r["tuple_names"]["8192"] in C.AUTHORISED_OOS_EXPERIMENTS and r["tuple_names"]["16384"] in C.AUTHORISED_OOS_EXPERIMENTS
            else:
                assert key == "llama8b" and r["tuple_names"] is None and sh["G8K_TUPLE"] == "''" and r["sampling_mode"] == "card"   # in support: no tuple
        else:
            assert r.get("object_source") is None and r["object_sha256"] == CONSTITUTION_PIN == sh["PIN"] and r["staged_id"] == "base__constv2-mix-full__psr_ce_all"
            assert r["tuple_names"] == {"8192": C.g8k_native_tuple_name(key, 8192), "16384": C.g8k_native_tuple_name(key, 16384)} and key in C.G8K_NATIVE_APPENDIX_MODELS
            assert sh["G8K_TUPLE"] == "gsm8k_paired200_nemotron3nano4b_native_8192" and sh["G8K_OBJECT_SOURCE"] == "''"
            assert r["source_row"]["index"] == 27 and r["object_kind"] == "psr"                                            # the published matrix's lens-era row
            # the FILED refusal: the pair is THINKING_UNSERVABLE on both engines; the row stays launch, counts as an expected refusal
            assert r["expected_refusal"] is True and r["expected_refusal_reason"] == PA.CONSTITUTION_REFUSAL_REASON
            assert "THINKING_UNSERVABLE" in r["expected_refusal_reason"] and "483252_27" in r["expected_refusal_reason"] and "596419_1" in r["expected_refusal_reason"]
            assert sh["G8K_EXPECTED_REFUSAL"] != "''"
        # the row's OWN overlay pair, cut from the PANEL view's campaign config: explicit_cells [none, <cell>]
        cfg = yaml.safe_load((REPO / f"configs/sdfbase/gsm8k/sdfbase_full22_{r['tag']}.gsm8k_{r['cell_slug']}.yaml").read_text())
        man = yaml.safe_load((REPO / f"configs/capeval/manifests/petri_full22_{r['tag']}.gsm8k_{r['cell_slug']}.yaml").read_text())
        campaign = yaml.safe_load((REPO / f"configs/sdfbase_full22_{r['tag']}.yaml").read_text())
        assert cfg["explicit_cells"] == ["none", r["cell"]] and man["petri"]["cells"] == ["none", r["harness_cell"]]
        assert {k: v for k, v in cfg.items() if k != "explicit_cells"} == {k: v for k, v in campaign.items() if k != "explicit_cells"}
        assert cfg["model"] == r["served_checkpoint"]["model_id"] if r["cell"] == COMPOSED else True
    assert PA.load_record() == doc


def test_the_composed_arm_identity_resolves_to_its_own_tuple_by_content_and_the_anchor_rows_are_the_in_frame_unsteered_runs():
    doc = M.load_matrix(RECORD)
    default = M.load_matrix(M.MATRICES[M.MATRIX_DEFAULT_NAME]); llama = M.load_matrix(M.MATRICES[M.MATRIX_LLAMA_NAME])
    anchors = {(r["model_key"], r["view"]): r for r in default["rows"] + llama["rows"] if r["cell"] == M.CELL_NONE}
    for r in doc["rows"]:
        a = anchors[(r["model_key"], r["view"])]
        assert r["anchor_out_root"] == a["out_root"] and r["sampling"] == a["sampling"] and r["family"] == a["family"] and r["reasoning_parser"] == a["reasoning_parser"]
        assert r["expected_stamps"] == ["cell-stable", "always", a["expected_stamps"][2]]
    # contracts: the composed cell resolves, BY CONTENT, to the native_paired tuple for the two thinking-ON models and to nothing else's
    for key in ("gemma4e4b", "nemotron3nano4b"):
        for cap in (8192, 16384):
            t = C.AUTHORISED_OOS_EXPERIMENTS[C.g8k_native_paired_tuple_name(key, cap)]
            assert set(t.cells) == {COMPOSED} and t.model == key
    t = C.AUTHORISED_OOS_EXPERIMENTS[C.g8k_native_tuple_name("nemotron3nano4b", 8192)]
    assert "constv2-mix-full__psr_ce_all" in t.cells and t.model == "nemotron3nano4b"
    assert "llama8b" not in C.G8K_NATIVE_PAIRED_MODELS                                                     # the no-switch arm is in support
    # the record is an ELIGIBLE bank-view document (contracts.GSM8K_BANK_VIEW_DOCUMENTS): its external rows' SOURCE banks join the declared
    # vocabulary, while the rows' STAMP stays the instrument view (`main` / `it`: expected_bank_view_stamp) — the stores stamp `it` / `main`
    assert C.bank_view_document_schemas(RECORD.name) == (PA.SCHEMA,)
    banks = {r["bank_view"] for r in doc["rows"] if r.get("bank_view")}
    assert banks == {f"paired_d1_{k}_s1" for k in ("gemma4e4b", "llama8b", "nemotron3nano4b")} and banks <= set(C.declared_bank_views())
    for r in doc["rows"]:
        assert C.expected_bank_view_stamp(view=r["view"], bank_view=r.get("bank_view")) == r["view"]


def test_a_record_row_off_its_panel_view_or_with_seed_or_companion_facts_refuses_by_name_and_the_plain_arm_rule_still_holds():
    doc = M.load_matrix(RECORD); r = copy.deepcopy(doc["rows"][0])
    bad = copy.deepcopy(r); bad["view"] = "s1"; bad["tag"] = "nemotron3nano4b_s1"
    assert "sits on its declared panel view" in M.external_row_problem(bad)
    bad = copy.deepcopy(r); bad["panel_view"] = "it"
    assert "sits on its declared panel view" in M.external_row_problem(bad)
    bad = copy.deepcopy(r); bad["inspect_seed_policy"] = "replicate"; bad["inspect_seed_base"] = 1
    assert "default Inspect seed rule with no companion" in M.external_row_problem(bad)
    bad = copy.deepcopy(r); bad["standup_companion_cells"] = ["constv2-mix-full__lora_r4_ce"]
    assert "default Inspect seed rule with no companion" in M.external_row_problem(bad)
    bad = copy.deepcopy(r); bad["cell"] = "constv2-mix-full__psr_ce_all"
    assert f"is the {COMPOSED!r} arm" in M.external_row_problem(bad)
    # without the record's role, an external composed row is not one of the rung's steer cells — the plain rule refuses it
    plain = copy.deepcopy(r); plain.pop("role")
    assert "not one of the rung's cells" in M.external_row_problem(plain)
    # a record whose rows lack the role is refused by the loader
    broken = json.loads(RECORD.read_text()); broken["rows"][2]["role"] = "something_else"
    p = Path("/tmp/g8k_paired_allmodels_record_broken.json"); p.write_text(json.dumps(broken))
    with pytest.raises(SystemExit, match="carry role 'paired_allmodels_value'; rows \\[2\\]"):
        M.load_matrix(p)


def test_the_external_rows_serving_pin_follows_the_store_under_the_paired_writes_no_lens_rule(tmp_path):
    """g8k_select_serving_path runs `row --serving-path --wave-root` on every row before f22_init. The steer cells' store-following pin
    (steer_store_pin.pin_for) has no rule for the composed cell — it is not a steer cell and its kind table is pinned to the steer wave's
    cells — so the record's external rows take `paired_write_serving_pin`: the same two sources, under the one rule the paired write admits
    (no lens form). Unpinned with no store (the kind-aware default selects lora_native), unpinned `store:native(disk)` on a native store,
    REFUSED on a lens stamp under the row's root (a foreign store), and the roster-served constitution row is `unpinned:not-external`."""
    sys.path.insert(0, str(REPO / "scripts" / "sdfbase" / "full22"))
    import steer_store_pin as SP  # noqa: E402
    doc = M.load_matrix(RECORD)
    composed = [r for r in doc["rows"] if r["cell"] == COMPOSED]
    constitution = next(r for r in doc["rows"] if r["cell"] != COMPOSED)
    assert M.serving_path_pin(constitution, wave_root=tmp_path) == (None, SP.REASON_UNPINNED_NOT_EXTERNAL)
    for r in composed:
        assert M.serving_path_pin(r) == (None, SP.REASON_UNPINNED_STORE_UNREAD)                    # the grading host plan: no root to read under
        assert M.serving_path_pin(r, wave_root=tmp_path) == (None, SP.REASON_UNPINNED_NO_STORE)    # a fresh wave root: no store, no pins
        with pytest.raises(SystemExit, match="not a steer cell"):
            SP.steer_cell_kind(r["cell"])                                                          # the steer table is not widened for it
    r = composed[0]
    root = tmp_path / r["out_root"]; root.mkdir(parents=True)
    (root / SP.STORE_EVIDENCE).write_text(json.dumps({SP.IDENTITY_FIELD: "vllm+lora+not_applicable"}))
    assert M.serving_path_pin(r, wave_root=tmp_path) == (None, SP.store_reason(SP.ENGINE_NATIVE, [SP.SOURCE_DISK]))
    # the CLI form the launcher reads: an empty pin line, then the reason
    out = subprocess.run([sys.executable, str(REPO / "scripts/sdfbase/gsm8k/g8k_cells.py"), "row", "--matrix", NAME, "--index", str(r["index"]),
                          "--serving-path", "--wave-root", str(tmp_path)], capture_output=True, text=True, check=True,
                         env={**os.environ, "PYTHONPATH": str(REPO / "src")})
    assert out.stdout == f"\n{SP.store_reason(SP.ENGINE_NATIVE, [SP.SOURCE_DISK])}\n"
    (root / SP.STORE_EVIDENCE).write_text(json.dumps({SP.IDENTITY_FIELD: "vllm+lens+stacked"}))
    with pytest.raises(SystemExit, match="no lens form"):
        M.serving_path_pin(r, wave_root=tmp_path)
    (root / SP.STORE_EVIDENCE).write_text(json.dumps({SP.IDENTITY_FIELD: "vllm+passthrough+not_applicable"}))
    with pytest.raises(SystemExit, match="names NO engine"):
        M.serving_path_pin(r, wave_root=tmp_path)
