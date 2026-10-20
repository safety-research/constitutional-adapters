"""The GSM8K row machinery KNOWS the front door's models: the model facts are derived from the models table, one
`frontdoor_<key>` matrix is registered per front-door model with its own wave root and accepted in the sbatch's `case`, and the
frozen default matrix and every paper row stay byte-identical:

* `g8k_cells.front_door_facts()` derives, from `reproduce/models.yaml` through its ONE reader, every phase-view model's g8k family
  (`architecture.family` -> the family table), arm (`thinking.gsm8k`, held to the family's), sampling mode, sampling ROW (the shipped
  `serve.generation_config`, exactly its sampler keys + a source), reasoning parser and placement (the family's rules); the paper's
  literals MODEL_KEYS / FAMILY_BY_MODEL / SAMPLING_MODE_BY_MODEL / SAMPLING_ROWS / MATRICES / VIEWS_BY_MATRIX are untouched and every
  accessor answers from them for a paper key; a paper key's derived family equals its literal; refusals by name for a key MODEL_KEYS
  already names, an unknown table family, a thinking spelling outside on / not_available, a served checkpoint with no generation_config;
* the registry: `resolve_matrix("frontdoor_<key>")` -> `configs/sdfbase/gsm8k/matrix_frontdoor_<key>.json`, `matrix_name_of` round-trips,
  `array_tail` names it; an unknown front-door key refuses by name; `bogus` still refuses;
* `select_rows` over a roster carrying the test model's block yields its u0 rows (none + the two prompted SI arms, launch, thinking on,
  the derived facts, no tuple) and its battery rows (fitted, pinned, the shared tuple pair) under the front door's wave and out-root prefix;
  `make_gsm8k_matrix.build(matrix_name=frontdoor_<key>)` writes a schema-2 document `load_matrix` accepts; on the real tree it refuses
  by name while the roster carries none of the views, and the DEFAULT matrix regenerates byte-identically;
* `gsm8k_thinking.sbatch`'s first `case` arm is still exactly MATRICES' names (the pinned regex) and a second arm admits `frontdoor_*`;
  a foreign name is refused before anything is sourced, with the message the other pins read."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
G8K = REPO / "scripts" / "sdfbase" / "gsm8k"
TABLE = REPO / "reproduce" / "models.yaml"
for p in (G8K, REPO / "scripts" / "sdfbase" / "full22", REPO / "scripts" / "sdfbase" / "seedrep", REPO / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import g8k_cells as M  # noqa: E402
import g8k_views as V  # noqa: E402
import make_gsm8k_matrix as MK  # noqa: E402

ENV = {**os.environ, "PYTHONPATH": str(REPO / "src")}
KEY = "qwen35_4b_demo"
PAPER_MODEL_KEYS = ("gemma4e4b", "qwen35_4b", "nemotron3nano4b", "glm46vflash", "llama8b")
PAPER_MATRICES = {"default", "llama", "itdoc", "qwen_seedrep", "arditi_allpos", "arditi_seedrep", "paired_allmodels"}


# --------------------------------------------------------------------------- the derived facts
def test_front_door_facts_derive_from_the_models_table_and_the_paper_literals_are_untouched(tmp_path):
    table = yaml.safe_load(TABLE.read_text(encoding="utf-8"))["models"]
    facts = M.front_door_facts()
    reader = M._models_reader(TABLE)                                            # the table's ONE reader: the grammar's views + the row's declared wave
    expect_keys = tuple(k for k, m in table.items() if reader.phase_views_of(m))
    assert expect_keys and tuple(facts) == expect_keys == M.front_door_model_keys() and KEY in facts
    for k, f in facts.items():
        m = table[k]
        assert f["family"] == M.G8K_FAMILY_BY_TABLE_FAMILY[m["architecture"]["family"]]
        assert f["arm"] == M.G8K_ARM_BY_TABLE_THINKING[m["thinking"]["gsm8k"]] == M.THINKING_MODE_BY_FAMILY[f["family"]]
        assert f["sampling_mode"] == (M.SAMPLING_THINKING_CARD if f["arm"] == M.THINKING_ON else M.SAMPLING_CARD)
        gc = m["serve"]["generation_config"]
        if gc is None:
            # a served checkpoint that SHIPS NO generation_config.json: the row is the hand-listed row of the paper model serving the SAME
            # checkpoint (the demo serves the paper's Qwen3.5-4B), admitted through the shipped-file table's `ships_none` record — never guessed
            twin = next(k2 for k2 in M.MODEL_KEYS if k2 in table and table[k2]["serve"]["checkpoint"]["hf_repo"] == m["serve"]["checkpoint"]["hf_repo"])
            assert {x: f["sampling"][x] for x in f["sampling"] if x != "source"} == {x: v for x, v in M.SAMPLING_ROWS[twin].items() if x in M.SAMPLING_PARAM_KEYS}
            assert "ships no generation_config.json" in f["sampling"]["source"] and "ships_none" in f["sampling"]["source"] and f"SAMPLING_ROWS[{twin!r}]" in f["sampling"]["source"]
        else:
            assert {x: f["sampling"][x] for x in f["sampling"] if x != "source"} == gc and set(gc) <= set(M.SAMPLING_PARAM_KEYS)
        assert m["serve"]["checkpoint"]["hf_repo"] in f["sampling"]["source"] and "generation_config" in f["sampling"]["source"]
        assert f["reasoning_parser"] == ("gemma4" if f["family"] == "gemma4" else None)
        assert f["placement"] == (M.PLACEMENT_PRIMARY if f["family"] == "nemotron_h" else M.PLACEMENT_ANY) and f["gpu_mem_util"] == M.GPU_MEM_UTIL_DEFAULT
        assert f["views"] == tuple(reader.phase_views_of(m)) and all(M.PHASE_VIEW_RE.match(v) or v == m.get("wave") for v in f["views"])
        # the accessors answer from the table for a front-door key
        assert M.family_of(k) == f["family"] and M.sampling_mode_of(k) == f["sampling_mode"] and M.sampling_row_of(k) == f["sampling"]
        assert M.reasoning_parser_of(k) == f["reasoning_parser"] and M.gpu_mem_util_of(k) == f["gpu_mem_util"] and M.placement_for(k) == f["placement"]
        assert M.longsi_scope(k) == M.LONGSI_SCOPE_BY_FAMILY[f["family"]] and M.expected_refusal(k, False) is None
    assert facts[KEY]["family"] == "qwen3_5" and facts[KEY]["reasoning_parser"] is None and facts[KEY]["views"] == ("battery", "d0p75", "d0p75_relu")
    assert facts[KEY]["sampling"] | {"source": None} == {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0, "source": None}   # the 4B's thinking card row
    # the paper's literals: byte-identical, and the derivation agrees with them wherever both speak
    assert M.MODEL_KEYS == PAPER_MODEL_KEYS and set(M.MATRICES) == PAPER_MATRICES == set(M.VIEWS_BY_MATRIX)
    assert set(M.SAMPLING_ROWS) == set(M.SAMPLING_MODE_BY_MODEL) == set(M.FAMILY_BY_MODEL) == set(PAPER_MODEL_KEYS)
    for k in PAPER_MODEL_KEYS:
        if k in table:
            assert M.G8K_FAMILY_BY_TABLE_FAMILY[table[k]["architecture"]["family"]] == M.FAMILY_BY_MODEL[k], k
        assert M.family_of(k) == M.FAMILY_BY_MODEL[k] and M.sampling_mode_of(k) == M.SAMPLING_MODE_BY_MODEL[k] and M.sampling_row_of(k) == M.SAMPLING_ROWS[k]
        assert M.reasoning_parser_of(k) == M.REASONING_PARSER_BY_MODEL.get(k) and M.gpu_mem_util_of(k) == M.GPU_MEM_UTIL_BY_MODEL.get(k, M.GPU_MEM_UTIL_DEFAULT)
        assert M.placement_for(k) == (M.PLACEMENT_PRIMARY if k in M.PRIMARY_ONLY_MODELS else M.PLACEMENT_ANY)
    assert M.all_model_keys() == (*PAPER_MODEL_KEYS, *expect_keys) and M.family_of("no_such_model") is None
    with pytest.raises(SystemExit, match="not one of"):
        M.sampling_mode_of("no_such_model")
    # refusals, by name, over synthetic tables
    def table_with(models):
        p = tmp_path / f"t{len(list(tmp_path.iterdir()))}.yaml"
        p.write_text(yaml.safe_dump({"schema": 1, "models": models})); return p
    base = {"view_legacy_prefix": "mk", "architecture": {"family": "gemma4_dense"}, "thinking": {"gsm8k": "on"},
            "serve": {"checkpoint": {"hf_repo": "org/m", "revision": "a" * 40}, "generation_config": {"temperature": 1.0}}, "views": {"s1": {}}}
    assert M.front_door_facts(tmp_path / "absent.yaml") == {}
    with pytest.raises(SystemExit, match="one key, one table"):
        M.front_door_facts(table_with({"qwen35_4b": base}))
    with pytest.raises(SystemExit, match="maps to no g8k family"):
        M.front_door_facts(table_with({"mk": {**base, "architecture": {"family": "mystery_arch"}}}))
    with pytest.raises(SystemExit, match="neither `on` nor `not_available`"):
        M.front_door_facts(table_with({"mk": {**base, "thinking": {"gsm8k": "off"}}}))
    with pytest.raises(SystemExit, match="disagrees with its family"):
        M.front_door_facts(table_with({"mk": {**base, "thinking": {"gsm8k": "not_available"}}}))
    with pytest.raises(SystemExit, match="does not record org/m @ aaaaaaaa as `ships_none`"):           # a null generation_config needs the shipped-file table's row
        M.front_door_facts(table_with({"mk": {**base, "serve": {**base["serve"], "generation_config": None}}}))
    with pytest.raises(SystemExit, match="decode at the shipped"):
        M.front_door_facts(table_with({"mk": {**base, "serve": {**base["serve"], "generation_config": {}}}}))
    with pytest.raises(SystemExit, match="non-sampler keys"):
        M.front_door_facts(table_with({"mk": {**base, "serve": {**base["serve"], "generation_config": {"temperature": 1.0, "do_sample": True}}}}))
    assert M.front_door_facts(table_with({"paper_like": {**base, "views": {"main": {}, "it": {}}}})) == {}     # main / it derive nothing


# --------------------------------------------------------------------------- the registry
def test_front_door_matrices_resolve_by_name_beside_the_frozen_registry():
    name = M.front_door_matrix_name(KEY)
    assert name == f"frontdoor_{KEY}" and M.front_door_key_of(name) == KEY
    path = M.resolve_matrix(name)
    assert path == M.front_door_matrices()[name] == REPO / "configs" / "sdfbase" / "gsm8k" / f"matrix_frontdoor_{KEY}.json"
    assert M.matrix_name_of(path) == name and M.array_tail("0" * 40, matrix=path) == f"ARRAY {'0' * 40} 8192 {name}"
    assert M.array_tail("0" * 40, rung=16384, matrix=path) == f"ARRAY {'0' * 40} 16384 {name}"
    assert M.front_door_wave(KEY) == name and M.front_door_wave_root_rel(KEY) == f"{M.WAVE_ROOT_REL}_{name}" != M.WAVE_ROOT_REL
    assert M.out_root_prefix_for(M.front_door_wave(KEY)) == f"controls_{name}"
    assert set(M.front_door_matrices()) == {f"frontdoor_{k}" for k in M.front_door_model_keys()} and not (set(M.front_door_matrices()) & set(M.MATRICES))
    with pytest.raises(SystemExit, match="is not a front-door model"):
        M.resolve_matrix("frontdoor_no_such_model")
    with pytest.raises(SystemExit, match="neither a matrix name"):
        M.resolve_matrix("bogus")
    assert M.resolve_matrix(None) == M.resolve_matrix("default") == M.MATRIX_PATH and M.matrix_name_of(M.MATRIX_PATH) == "default"


# --------------------------------------------------------------------------- the rows
VIEWS = ("battery", "d0p75", "d0p75_relu")                     # the demo's phase views: the single-phase recipe's ONE wave + the two dose rungs (no u0, no s2)
sys.path.insert(0, str(REPO / "reproduce" / "lib")); sys.path.insert(0, str(REPO / "scripts" / "sdfbase" / "full22"))
import f22_roster as FR  # noqa: E402
import models as MT  # noqa: E402
#: the roster block 05 --step roster prints for the demo (reproduce/lib/models.py roster-block): {view: {roster cell id: staged id | None}}
BLOCK = yaml.load("targets:\n" + MT.roster_block(KEY, list(VIEWS)), Loader=FR._StrictRosterLoader)["targets"][KEY]
S1_CELLS = dict(BLOCK["views"]["battery"]["cells"])           # the prompted legs (no staged id) + every arm (the Arditi object under its harness id)
OBJECT_CELLS = {v: {c: sid for c, sid in BLOCK["views"][v]["cells"].items() if sid} for v in VIEWS}


def _front_door_tree(tmp_path: Path) -> Path:
    """A configs root = the real roster + pins with the test model's block pasted (the shape 05 --step roster prints: battery = the two prompted SI
    arms + every arm, the two dose rungs with their scaled copy) and synthetic 64-hex pins for the staged ids."""
    root = tmp_path / "tree"; (root / "configs").mkdir(parents=True)
    roster = yaml.safe_load((REPO / "configs" / "sdfbase_full22_roster.yaml").read_text(encoding="utf-8"))
    roster["targets"][KEY] = {"model": "Qwen/Qwen3.5-4B", "model_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", "bank": "full22",
                              "views": {v: {"config": f"configs/sdfbase_full22_{KEY}_{v}.yaml", "manifest": f"configs/capeval/manifests/petri_full22_{KEY}_{v}.yaml",
                                            "cells": dict(BLOCK["views"][v]["cells"])} for v in VIEWS}}
    (root / "configs" / "sdfbase_full22_roster.yaml").write_text(yaml.safe_dump(roster, sort_keys=False))    # the roster's cell ORDER is the row order
    pins = json.loads((REPO / "configs" / "sdfbase_full22_pins.json").read_text(encoding="utf-8"))
    staged = list(dict.fromkeys(sid for v in VIEWS for sid in OBJECT_CELLS[v].values()))
    pins[KEY] = {sid: {"object_sha256": f"{i:064x}", "source": f"synthetic/{sid}"} for i, sid in enumerate(staged, start=1)}
    (root / "configs" / "sdfbase_full22_pins.json").write_text(json.dumps(pins, indent=1))
    return root


def test_select_rows_yields_the_test_models_s1_and_dose_rows_under_the_front_doors_wave_and_the_paper_rows_are_unchanged(tmp_path):
    root = _front_door_tree(tmp_path)
    roster, pins = V.load_roster(V.roster_path(root)), V.load_pins(V.pins_path(root))
    name = M.front_door_matrix_name(KEY)
    rows = V.select_rows(roster, pins, {KEY: list(VIEWS)}, schema=M.MATRIX_SCHEMA_2, trigger_threshold=M.TRIGGER_THRESHOLD,
                         wave=M.front_door_wave(KEY), where="front door test")
    battery = [r for r in rows if r["view"] == "battery"]
    assert [r["cell"] for r in battery] == ["none", *S1_CELLS]                            # the single-phase wave: its own `none`, both SIs (a schema-2 matrix schedules both), every arm
    unfitted = [r for r in battery if r["cell"] == "none" or r["cell"].startswith("prompted__")]
    assert len(unfitted) == 3 and all(r["status"] == M.STATUS_LAUNCH and not r["fitted"] and r["tuple_name"] is None and r["object_sha256"] is None for r in unfitted)
    for r in rows:
        assert r["model_key"] == KEY and r["family"] == "qwen3_5" and r["thinking_mode"] == M.THINKING_ON and r["sampling_mode"] == M.SAMPLING_THINKING_CARD
        assert r["sampling"] == M.sampling_row_of(KEY) and r["trigger_threshold"] == M.TRIGGER_THRESHOLD and r["wave"] == name
        assert r["out_root"].startswith(f"controls_{name}/{KEY}_{r['view']}/") and r["reasoning_parser"] is None and r["cluster_constraint"] == M.PLACEMENT_ANY
        assert r["expected_stamps"] and r["tag"] == f"{KEY}_{r['view']}" and r["time_limit"] == M.time_limit_for(KEY, r["fitted"], cell=r["cell"])
    for v in VIEWS:
        fitted = [r for r in rows if r["view"] == v and r["fitted"]]
        assert [r["cell"] for r in fitted] == list(OBJECT_CELLS[v]), v                 # every object cell of the view, in the roster's order
        if v != "battery":
            assert [r["cell"] for r in rows if r["view"] == v] == ["none", *OBJECT_CELLS[v]]   # a dose rung carries its own `none` too (the single-phase recipe)
        for r in fitted:
            assert r["staged_id"] == OBJECT_CELLS[v][r["cell"]] and r["object_sha256"] == pins[KEY][r["staged_id"]]["object_sha256"]
            # the fitted rows' tuple pair: the own-tuple rule (g8k_cells' stdlib mirror of contracts.G8K_FRONTDOOR_MODELS) has NO member on
            # this tree, so the rows carry the shared `gsm8k_paired200_<key>_<cap>` pair and never a `_frontdoor` infix
            assert M.FRONTDOOR_TUPLE_MODELS == () and KEY not in M.FRONTDOOR_TUPLE_MODELS and M.FRONTDOOR_TUPLE_INFIX == "_frontdoor"
            assert r["tuple_name"] == f"gsm8k_paired200_{KEY}_8192" and r["tuple_names"] == {"8192": f"gsm8k_paired200_{KEY}_8192", "16384": f"gsm8k_paired200_{KEY}_16384"}
    n_rows = sum(1 + len(BLOCK["views"][v]["cells"]) for v in VIEWS)                 # `none` + the block's cells per view: 10 + 2 + 2
    assert [r["index"] for r in rows] == list(range(len(rows))) and len(rows) == n_rows == 14
    c = M.counts(rows)
    assert c["families"] == ["qwen3_5"] and c["pins"] == {KEY: 9} and c["waves"] == {name: 14} and c["tags"] == [f"{KEY}_{v}" for v in VIEWS]
    # the paper's frozen document is byte-identical: its rows, counts and the two frozen generators' --check
    doc = json.loads(M.MATRIX_PATH.read_text(encoding="utf-8"))
    assert len(M.matrix_rows()) == 53 and M.counts() == doc["counts"] and doc["counts"]["families"] == ["gemma4", "glm4v", "nemotron_h", "qwen3_5"]
    for argv in ([sys.executable, str(G8K / "make_gsm8k_matrix.py"), "--check"], [sys.executable, str(G8K / "make_gsm8k_overlays.py"), "--check"]):
        r = subprocess.run(argv, capture_output=True, text=True, cwd=REPO, env=ENV)
        assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]


def test_the_front_door_matrix_is_derived_from_the_table_and_the_roster_and_refuses_while_the_roster_lacks_the_views(tmp_path):
    name = M.front_door_matrix_name(KEY)
    # ROSTER-CONDITIONAL (the fix-1 owner follow-up 10): while the production roster carries none of the views -> refused by name, nothing
    # written; once 05 --step roster lands the block and --step configs renders the matrix, `--check` holds the committed document
    prod = V.load_roster(V.roster_path(REPO))
    carried = [v for v in M.front_door_facts()[KEY]["views"] if v in ((prod.get("targets") or {}).get(KEY) or {}).get("views", {})]
    r = subprocess.run([sys.executable, str(G8K / "make_gsm8k_matrix.py"), "--matrix-name", name, "--check"], capture_output=True, text=True, cwd=REPO, env=ENV)
    if carried:
        assert r.returncode == 0 and M.resolve_matrix(name).exists() and MK.front_door_spec(name, REPO).views == {KEY: carried}, r.stdout + r.stderr
    else:
        assert r.returncode != 0 and "does not carry" in r.stderr and "phase views" in r.stderr and not M.resolve_matrix(name).exists()
        with pytest.raises(SystemExit, match="does not carry"):
            MK.front_door_spec(name, REPO)
    with pytest.raises(SystemExit, match="not a front-door model"):
        MK.front_door_spec("frontdoor_no_such_model", REPO)
    r = subprocess.run([sys.executable, str(G8K / "make_gsm8k_matrix.py"), "--matrix-name", "frontdoor_no_such_model"], capture_output=True, text=True, cwd=REPO, env=ENV)
    assert r.returncode != 0 and "not a front-door model" in r.stderr
    # a configs root whose roster carries the block: a schema-2 document under the key's own wave root, accepted by load_matrix
    root = _front_door_tree(tmp_path)
    spec = MK.front_door_spec(name, root)
    assert spec.views == {KEY: list(VIEWS)} and spec.schema == M.MATRIX_SCHEMA_2 and spec.wave == name
    assert spec.wave_root_rel == M.front_door_wave_root_rel(KEY) and spec.trigger_threshold == M.TRIGGER_THRESHOLD and spec.frozen_inputs is None
    text, doc = MK.build(configs_root=root, views_yaml=None, views_spec=None, matrix_name=name)
    assert doc["schema"] == M.MATRIX_SCHEMA_2 and doc["wave_root_rel"] == M.front_door_wave_root_rel(KEY) and doc["wave"] == name
    assert doc["views"] == {KEY: list(VIEWS)} and doc["source"]["views_yaml"] is None and doc["source"]["roster"] == "configs/sdfbase_full22_roster.yaml"
    out = MK.matrix_out_path(name, root)
    assert out == root / "configs" / "sdfbase" / "gsm8k" / f"matrix_frontdoor_{KEY}.json"
    out.parent.mkdir(parents=True); out.write_text(text, encoding="utf-8")
    loaded = M.load_matrix(out)
    assert [r["cell"] for r in loaded["rows"]] == [c for v in VIEWS for c in ("none", *BLOCK["views"][v]["cells"])]   # every view with its own `none` (single-phase)
    assert M.wave_root_rel_of(loaded) == M.front_door_wave_root_rel(KEY) and all(r["thinking_mode"] == "on" for r in loaded["rows"])
    r = subprocess.run([sys.executable, str(G8K / "make_gsm8k_matrix.py"), "--matrix-name", name, "--configs-root", str(root), "--check"],
                       capture_output=True, text=True, cwd=REPO, env=ENV)
    assert r.returncode == 0 and "14 rows" in r.stdout, r.stdout + r.stderr
    # every frozen document's bytes are unmoved (a front-door matrix never lands in the default one)
    assert len(json.loads(M.MATRIX_PATH.read_text())["rows"]) == 53


# --------------------------------------------------------------------------- the launcher
def test_the_sbatch_keeps_the_frozen_names_as_its_first_arm_and_admits_a_front_door_matrix_in_a_second():
    src = (G8K / "gsm8k_thinking.sbatch").read_text()
    m = re.search(r'case "\$G8K_MATRIX" in ([a-z_|]+)\) ;;', src)
    assert m and sorted(m.group(1).split("|")) == sorted(M.MATRICES)                 # the pinned first arm: exactly the frozen registry
    assert "frontdoor_*) ;;" in src.split(m.group(0), 1)[1].split("esac", 1)[0]     # the second arm: the front door's derived matrices
    assert subprocess.run(["bash", "-n", str(G8K / "gsm8k_thinking.sbatch")]).returncode == 0
    env = {"PATH": os.environ["PATH"], "HOME": str(REPO)}
    args = ["bash", str(G8K / "gsm8k_thinking.sbatch"), KEY, "battery", "none", "8192", "0" * 40, "0", "0"]
    r = subprocess.run([*args, "bogus"], capture_output=True, text=True, env=env)
    assert r.returncode == 1 and "matrix must be one of default|llama|itdoc|qwen_seedrep" in r.stderr and "frontdoor_<models.yaml key>" in r.stderr
    r = subprocess.run([*args, f"frontdoor_{KEY}"], capture_output=True, text=True, env=env)
    assert "matrix must be one of" not in r.stderr                                  # past the case: refused later, elsewhere, by name
