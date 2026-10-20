"""The wave generator, the roster printer and the unit plan know the FRONT DOOR's targets (paper-companion-repo;
see review/fix_briefs/run_readiness.md):

* `make_full22_configs` derives a front-door target's Petri serving pins from the models table (`front_door_serving`: the table's
  `serve.chat_template_hash`, the architecture family's tool parser + launch flags — FAMILY_SERVING — held equal to the paper's
  MODEL_SERVING literals for every paper model in the table) and substitutes the roster block's served checkpoint into the battery body
  (`render_config`; the block is held to the table's serve.checkpoint); a row without a recorded hash, a family without a serving row, a
  block disagreeing with the table, an unknown manifest target: each refuses by name. The bare LoRA dose view `d0p75` renders through
  the frozen allow-list's cited front-door entries.
* `reproduce/lib/models.py roster-block` prints the COMPLETE block (served checkpoint, battery body, per-view config / manifest paths and
  cells), every object under its OWN cell id — the Arditi object under its harness id `arditi_allpos` (the rung rides the staged id only),
  a dose view's cells under the scaled copy's `d<tok>__` staged id — so no two view cells share a roster key; `f22_roster.load_roster`
  refuses a duplicate mapping key by name and line.
* `reproduce/lib/stage.py` plans a view's GSM8K rows on the front door's matrix `frontdoor_<key>` with the ROSTER's cell ids, and its
  steered padding rows on the per-view family named after the view; every gsm8k launcher's `case` admits `frontdoor_*`; the grading host dry
  run admits the name; `09_dose_ladder.sh` plans the dose views' GSM8K rows through the same plan.
* `GRADER_PROVENANCE.md`'s hunk line counts the regenerated patch; the shipped generation-config table states the demo's served checkpoint's
  `ships_none` row at the pinned revision and the file is in the scrub's scope.

The front-door model runs the SINGLE-PHASE recipe: one wave `battery` carrying every arm beside its own unsteered legs and
two dose rungs, no u0 — so every view keeps its same-wave `none` anchor and plans its own `none` row; the declared-u0 padding path is exercised
on a SYNTHETIC phased row under tmp.

Synthetic fixtures only: placeholder shas, hostname TESTHOST, the neutral site example."""
from __future__ import annotations

import copy
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
for p in (REPO / "scripts" / "sdfbase", REPO / "scripts" / "sdfbase" / "full22", REPO / "scripts" / "sdfbase" / "gsm8k", REPO / "reproduce" / "lib", REPO / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import f22_roster as fr  # noqa: E402
import make_full22_configs as gen  # noqa: E402
import models as MT  # noqa: E402
import stage as ST  # noqa: E402
from front_door_testlib import MATCHERS, REPRO, SCAN, VENV_PY, run_stage, token_hits  # noqa: E402

#: one tier of the publication scan's token table, matched as the scan matches it (the tokens are never spelled here)
PROCESS_TOKENS = SCAN.Matcher([r for r in MATCHERS["rows"] if r.tier == "process"])

PY = VENV_PY or sys.executable
KEY = "qwen35_4b_demo"                                              # the front-door model of reproduce/models.yaml (configured, not run)
VIEWS = ("battery", "d0p75", "d0p75_relu")                           # its phase views, in the table's order: ONE wave + the two dose rungs (single-phase)
TABLE = yaml.safe_load((REPO / "reproduce" / "models.yaml").read_text(encoding="utf-8"))["models"]
PAPER_IN_TABLE = tuple(k for k in TABLE if k in gen.MODEL_SERVING)


def roster_block(key: str = KEY, views=VIEWS, table: str | None = None) -> str:
    """The key's roster block for `views`, as `models.py roster-block` prints it; `table` reads another models table (a test's
    synthetic phased row) instead of reproduce/models.yaml."""
    cmd = [PY, str(REPRO / "lib" / "models.py")] + (["--table", table] if table else []) + ["roster-block", key, ",".join(views)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stderr
    return r.stdout


def front_door_roster(key: str = KEY, views=VIEWS, table: str | None = None) -> dict:
    """The production roster + the test model's block exactly as 05 --step roster prints it."""
    roster = fr.load_roster(REPO / "configs" / "sdfbase_full22_roster.yaml")
    roster["targets"][key] = yaml.load("targets:\n" + roster_block(key, views, table), Loader=fr._StrictRosterLoader)["targets"][key]
    return roster


# --------------------------------------------------------------------------- item 1: the serving pins from the table
def test_family_serving_and_the_tables_chat_template_hash_equal_the_papers_literals():
    assert PAPER_IN_TABLE == ("gemma4e4b", "qwen35_4b", "nemotron3nano4b", "llama8b")
    for k in PAPER_IN_TABLE:
        tp, flags = gen.FAMILY_SERVING[TABLE[k]["architecture"]["family"]]
        assert (tp, flags) == (gen.MODEL_SERVING[k]["tool_parser"], gen.MODEL_SERVING[k]["flags"]), k
        assert TABLE[k]["serve"]["chat_template_hash"] == gen.MODEL_SERVING[k]["chat_template_hash"], k
    assert sorted(gen.FRONT_DOOR_MODELS) == [KEY] and not set(gen.FRONT_DOOR_MODELS) & set(gen.MODEL_SERVING)
    for k, m in gen.FRONT_DOOR_MODELS.items():
        ms = gen.front_door_serving(k)
        assert re.fullmatch(r"[0-9a-f]{16}", ms["chat_template_hash"]) and ms["chat_template_hash"] == m["serve"]["chat_template_hash"]
        assert (ms["tool_parser"], ms["flags"]) == gen.FAMILY_SERVING[m["architecture"]["family"]] and "--enable-lora" not in ms["flags"]
        assert k in ms["note"] and "chat_template_hash" in ms["note"]
    # the demo key: the qwen3_5_hybrid family row (the paper's qwen35_4b flags and tool parser), its hash as recorded (= the 4B's)
    assert gen.front_door_serving(KEY)["flags"] == gen.MODEL_SERVING["qwen35_4b"]["flags"]
    assert gen.front_door_serving(KEY)["tool_parser"] == "qwen3_xml" and gen.front_door_serving(KEY)["chat_template_hash"] == "a4aee8afcf2e0711" == TABLE["qwen35_4b"]["serve"]["chat_template_hash"]
    assert MT.check() == []


def test_front_door_serving_refuses_an_unrecorded_hash_an_unknown_family_and_an_unknown_target(monkeypatch):
    fd = copy.deepcopy(gen.FRONT_DOOR_MODELS)
    fd[KEY]["serve"]["chat_template_hash"] = None
    monkeypatch.setattr(gen, "FRONT_DOOR_MODELS", fd)
    with pytest.raises(SystemExit, match="serve.chat_template_hash is not recorded"):
        gen.front_door_serving(KEY)
    fd[KEY]["serve"]["chat_template_hash"] = "not-hex"
    with pytest.raises(SystemExit, match="serve.chat_template_hash is not recorded"):
        gen.front_door_serving(KEY)
    fd = copy.deepcopy(gen.FRONT_DOOR_MODELS)
    fd[KEY]["architecture"]["family"] = "mamba9_unknown"
    monkeypatch.setattr(gen, "FRONT_DOOR_MODELS", fd)
    with pytest.raises(SystemExit, match="has no serving row \\(FAMILY_SERVING"):
        gen.front_door_serving(KEY)
    monkeypatch.setattr(gen, "FRONT_DOOR_MODELS", {})
    with pytest.raises(SystemExit, match="not a MODEL_SERVING target and not a front-door model"):
        gen.front_door_serving(KEY)
    roster = front_door_roster()
    with pytest.raises(SystemExit, match="not a MODEL_SERVING target and not a front-door model"):
        gen.render_manifest(roster, KEY, "battery")
    # a table naming a MODEL_SERVING target with phase views: one authority, refused at derivation
    two = {**TABLE, "qwen35_4b": {**TABLE["qwen35_4b"], "views": {"s1": {"cells": ["none"]}}}}
    bad = Path(os.environ.get("TMPDIR", "/tmp")) / f"fd_two_{os.getpid()}.yaml"
    bad.write_text(yaml.safe_dump({"schema": 1, "models": two}))
    try:
        with pytest.raises(SystemExit, match="one authority"):
            gen.front_door_models(bad)
    finally:
        bad.unlink()
    assert gen.front_door_models(bad) == {}                                     # an absent table contributes nothing


# --------------------------------------------------------------------------- item 3: the roster block
def test_the_roster_block_is_complete_loads_strictly_and_names_each_objects_own_cell_id():
    text = roster_block()
    doc = yaml.load("targets:\n" + text, Loader=fr._StrictRosterLoader)          # no duplicate key anywhere in the block
    t = doc["targets"][KEY]
    ck = TABLE[KEY]["serve"]["checkpoint"]
    assert (t["model"], t["model_revision"], t["body_config"], t["bank"]) == (ck["hf_repo"], ck["revision"], MT.FRONT_DOOR_BODY_CONFIG, "full22")
    assert (REPO / t["body_config"]).is_file() and list(t["views"]) == list(VIEWS)
    for v in VIEWS:
        assert t["views"][v]["config"] == f"configs/sdfbase_full22_{KEY}_{v}.yaml"
        assert t["views"][v]["manifest"] == f"configs/capeval/manifests/petri_full22_{KEY}_{v}.yaml"
    battery = t["views"]["battery"]["cells"]                                                # the single-phase recipe's ONE wave: every arm beside the prompted legs (`none` is the generator's anchor)
    assert list(battery) == ["prompted__safety_refusal", "prompted__si_constitution", "arditi_allpos", "constv2ctl2-full__lora_r4_ce", "constv2-mix-full__lora_r4_ce",
                        "constv2-mix-full__lora_r4_ce_minus_ctl", "constv2ctl2-full__psr_ce_all", "constv2-mix-full__psr_ce_all",
                        "constv2-mix-full__psr_ce_all_minus_ctl"]   # the composed paired write (demo-import M2)
    assert battery["arditi_allpos"] == "arditi__ar_c0p3__arditi_allpos" and battery["constv2ctl2-full__psr_ce_all"] == "base__constv2ctl2-full__psr_ce_all"
    assert battery["prompted__safety_refusal"] is None and battery["prompted__si_constitution"] is None and text.count("constv2ctl2-full__psr_ce_all:") == 1    # ONE key for control_relu; the Arditi object has its own
    assert battery["constv2-mix-full__lora_r4_ce_minus_ctl"] == "derived__constv2-mix-full__lora_r4_ce_minus_ctl" and battery["constv2ctl2-full__lora_r4_ce"] == "base__constv2ctl2-full__lora_r4_ce"
    # a DOSE view names the SCALED COPY's staged id (build_strength.py's `d<tok>__` copy lane), never the parent's bytes
    assert t["views"]["d0p75"] == {"dose": 0.75, "dose_unit": "lora_write_basis_multiplier", "config": f"configs/sdfbase_full22_{KEY}_d0p75.yaml",
                                   "manifest": f"configs/capeval/manifests/petri_full22_{KEY}_d0p75.yaml",
                                   "cells": {"constv2-mix-full__lora_r4_ce_minus_ctl": "d0p75__constv2-mix-full__lora_r4_ce_minus_ctl"}}
    assert t["views"]["d0p75_relu"]["object_family"] == "relu" and t["views"]["d0p75_relu"]["cells"] == {
        "constv2-mix-full__psr_ce_all_minus_ctl": "d0p75__constv2-mix-full__psr_ce_all_minus_ctl"}
    roster = front_door_roster()
    for v in VIEWS:
        served = fr.served_cells(roster, KEY, v)                                  # the ONE reader accepts every view of the block
        assert list(served) == list(t["views"][v]["cells"])
    m = TABLE[KEY]
    assert MT.roster_cell_id(m, "refusal_direction") == "arditi_allpos" and MT.staged_id(m, "refusal_direction") == "arditi__ar_c0p3__arditi_allpos"
    assert MT.roster_cell_id(m, "control_relu") == "constv2ctl2-full__psr_ce_all" and MT.roster_cell_id(m, "none") == "none"
    assert MT.staged_id(m, "difference_lora", "d0p75") == "d0p75__constv2-mix-full__lora_r4_ce_minus_ctl" and MT.staged_id(m, "difference_lora", "battery") == "derived__constv2-mix-full__lora_r4_ce_minus_ctl"
    assert MT.staged_id(m, "prompted__safety_refusal") is None
    # the block, pasted, is what 05 checks: the stage's roster check and printer go through f22_roster.load_roster / models.py roster-block
    src = (REPRO / "05_stage_roster.sh").read_text()
    assert 'from f22_roster import load_roster' in src and '"$HERE/lib/models.py" --table "$(repro_models_table)" roster-block "$K" "$VIEWS"' in src
    assert "objs.get('corpus_constitution'" not in src                          # the `<corpus>__<arm>` heuristic is gone


def test_roster_view_cells_refuses_two_view_cells_under_one_roster_id_and_load_roster_refuses_a_duplicate_key(tmp_path):
    m = copy.deepcopy(TABLE[KEY])
    m["objects"]["refusal_direction"] = {"lane": "base", "cell_id": "constv2ctl2-full__psr_ce_all", "kind": "fit", "source": "fit_bank"}
    with pytest.raises(ValueError, match="duplicate mapping key"):
        MT.roster_view_cells(m, "battery")                                             # refusal_direction and control_relu sit in the one wave
    m2 = copy.deepcopy(TABLE[KEY]); m2["objects"]["refusal_direction"]["cell_id"] = "arditi_allpos"     # an arditi-lane id must spell the rung
    with pytest.raises(ValueError, match="<rung>__<harness id>"):
        MT.roster_cell_id(m2, "refusal_direction")
    with pytest.raises(KeyError, match="no `objects` entry"):
        MT.roster_cell_id(TABLE[KEY], "no_such_cell")
    bad = tmp_path / "dup.yaml"
    bad.write_text("targets:\n  k:\n    views:\n      s2:\n        cells:\n          constv2ctl2-full__psr_ce_all: a\n          constv2ctl2-full__psr_ce_all: b\n")
    with pytest.raises(fr.RosterDuplicateKey, match="duplicate mapping key 'constv2ctl2-full__psr_ce_all' at line 7"):
        fr.load_roster(bad)
    assert fr.load_roster(REPO / "configs" / "sdfbase_full22_roster.yaml")["targets"]                   # the production roster loads strictly
    # the roster CHECK of stage 05 exits 2 (not 1: "lacks the views") on a duplicate key, so the stage refuses instead of printing a block
    r = subprocess.run([PY, "-c", "import sys; sys.path.insert(0, sys.argv[1]); from f22_roster import load_roster; load_roster(sys.argv[2])",
                        str(REPO / "scripts" / "sdfbase" / "full22"), str(bad)], capture_output=True, text=True)
    assert r.returncode != 0 and "duplicate mapping key" in r.stderr


# --------------------------------------------------------------------------- items 1 + 3: every view renders from the block
def test_every_view_of_the_test_model_renders_from_the_block_without_a_pins_config():
    roster = front_door_roster()
    pins = json.loads((REPO / "configs" / "sdfbase_full22_pins.json").read_text(encoding="utf-8"))
    ck = TABLE[KEY]["serve"]["checkpoint"]
    ms = gen.front_door_serving(KEY)
    for v in VIEWS:
        cfg_text = gen.render_config(roster, pins, KEY, v)
        cfg = yaml.safe_load(cfg_text)
        cells = roster["targets"][KEY]["views"][v]["cells"]
        assert (cfg["model"], cfg["model_revision"]) == (ck["hf_repo"], ck["revision"]), v
        anchor = ["none"]                                            # the SINGLE-PHASE recipe: every view of the demo keeps its same-wave `none` anchor (its own unsteered leg)
        assert cfg["explicit_cells"] == [*anchor, *cells] and cfg["private_run"] == f"sdfbase_full22_{KEY}_{v}" and cfg["out_dir"] == f"out/sdfbase_full22_{KEY}_{v}"
        assert cfg["defenses_dir"] == f"out/full22/{KEY}/cells_view/{v}" and cfg["bon"]["max_new_tokens"] == gen.BON_MAX_NEW_TOKENS
        assert "FRONT DOOR's served checkpoint" in cfg_text and f"serving target: {ck['hf_repo']} @ {ck['revision']}" in cfg_text
        man_text = gen.render_manifest(roster, KEY, v)
        man = yaml.safe_load(man_text)["petri"]
        assert man["cells"] == [*anchor, *(gen.harness_cell_id(c) for c in cells)] and man["audit"] == f"petri_full22_{KEY}_{v}"
        assert (man["chat_template_hash"], man["tool_parser"], man["target_model"], man["target_revision"]) == (ms["chat_template_hash"], ms["tool_parser"], ck["hf_repo"], ck["revision"])
        assert list(man["serving_launch_flags"]) == ms["flags"] and "--enable-lora" not in man["serving_launch_flags"]
        assert "the FRONT DOOR's served target" in man_text
    # the bare LoRA dose view renders through the cited front-door entry of the frozen allow-list; the ReLU twin through its family name
    assert gen.view_object_family(roster, KEY, "d0p75") == "lora" and gen.view_dose(roster, KEY, "d0p75") == 0.75
    assert gen.view_object_family(roster, KEY, "d0p75_relu") == "relu" and gen.BARE_DOSE_VIEWS_FROZEN[KEY] == ("d0p75",)
    assert set(gen.BARE_DOSE_VIEWS_FROZEN) == {"gemma4e4b", "qwen35_4b", "nemotron3nano4b", "llama8b", "glm46vflash", KEY}
    # the block's served checkpoint is held to the table: a block from another table refuses by name
    other = copy.deepcopy(roster); other["targets"][KEY]["model_revision"] = "0" * 40
    with pytest.raises(SystemExit, match="one served checkpoint; re-print the block"):
        gen.render_config(other, pins, KEY, "battery")
    # the paper's committed outputs are untouched (their generator still checks green)
    r = subprocess.run([PY, str(REPO / "scripts" / "sdfbase" / "make_full22_configs.py"), "--check"], capture_output=True, text=True, cwd=REPO,
                       env={**os.environ, "PYTHONPATH": str(REPO / "src")})
    assert r.returncode == 0 and "DRIFT" not in r.stdout, r.stdout[-1500:] + r.stderr[-1500:]


# --------------------------------------------------------------------------- item 4: the unit plan's gsm8k + padding rows
def test_stage_plan_names_the_front_door_matrix_the_rosters_cell_ids_and_the_per_view_padding_family(monkeypatch, tmp_path):
    # the single-phase demo: none of its views is a steered wave (each carries its own `none`), so default planning runs no artefact walk; the
    # knob stage.py documents for synthetic trees is set anyway for the PHASED synthetic row planned below (never set in production)
    monkeypatch.setenv("SISTEER_PLAN_SKIP_ARTEFACT_CHECK", "1")
    table = str(REPRO / "models.yaml")
    m = TABLE[KEY]
    for view in VIEWS:
        rows = [r for r in ST.unit_plan(KEY, view, table=table, families=["gsm8k"], sha="a" * 40) if r["family"] == "gsm8k"]
        cells = list(m["views"][view]["cells"])                                   # the single-phase recipe: every view plans its own `none` row too
        assert [r["unit"] for r in rows] == cells                                  # the unit keeps the table's short name
        for r, cell in zip(rows, cells):
            assert r["args"] == [KEY, view, MT.roster_cell_id(m, cell), "8192", "a" * 40, "0" if cell == "none" or cell.startswith("prompted__") else "1", "0", f"frontdoor_{KEY}"], r["args"]
            assert r["script"] == "scripts/sdfbase/gsm8k/gsm8k_thinking.sbatch" and r["env"]["F22_REGISTRY_EVALS"] == "gsm8k"
    battery = {r["unit"]: r for r in ST.unit_plan(KEY, "battery", table=table, families=["gsm8k"], sha="a" * 40)}
    assert battery["none"]["args"][2] == "none" and battery["refusal_direction"]["args"][2] == "arditi_allpos" and battery["control_relu"]["args"][2] == "constv2ctl2-full__psr_ce_all"
    assert battery["difference_relu"]["args"][2] == "constv2-mix-full__psr_ce_all_minus_ctl"   # the composed paired write (demo-import M2)
    # every view's padding rows name the PER-VIEW family (= the view) through the obj launcher — the single-phase baseline wave battery included
    # (private run pad_static_<key>_s1_p<pad>); the prompted-only sbatch (target_static_padded.sbatch) is a DECLARED u0's alone
    for view in VIEWS:
        pad = [r for r in ST.unit_plan(KEY, view, table=table, families=["padding"], sha="a" * 40)]
        assert [r["args"][-1] for r in pad] == [view] * 5 and all(r["script"].endswith("target_static_padded_obj.sbatch") for r in pad), view
        assert [r["name"] for r in pad] == [f"padobj-{KEY}_{view}-p{p_}" for p_ in ST.PADDING_LEVELS] and pad[0]["args"][-4:-1] == ["grok420", "3", "all"]
        assert all(f"pad_static_{KEY}_{view}_p{r['padding_tokens']}" in r["coverage"]["grok420"]["path"]["root_rel"] for r in pad), view
    phased = copy.deepcopy(TABLE); phased["phased_demo"] = copy.deepcopy(TABLE[KEY]); phased["phased_demo"]["view_legacy_prefix"] = "phased_demo"
    phased["phased_demo"]["views"] = {"u0": {"cells": ["none", "prompted__safety_refusal", "prompted__si_constitution"]}, "s1": {"cells": ["control_lora", "constitution_lora", "difference_lora"]}}
    pt = tmp_path / "models.yaml"; pt.write_text(yaml.safe_dump({"schema": 1, "models": phased}, sort_keys=False))
    monkeypatch.setenv("SISTEER_MODELS_TABLE", str(pt))                         # a SYNTHETIC row that runs in phases: its u0 pads through the prompted-only sbatch
    pad0 = [r for r in ST.unit_plan("phased_demo", "u0", table=str(pt), families=["padding"], sha="a" * 40)]
    assert all(r["script"].endswith("static_padding/target_static_padded.sbatch") for r in pad0) and len(pad0) == 5
    assert [r["name"] for r in pad0] == [f"padstat-phased_demo-p{p_}" for p_ in ST.PADDING_LEVELS] and set(pad0[0]["evals"]) == set(ST.STATIC_EVALS)
    pad1 = [r for r in ST.unit_plan("phased_demo", "s1", table=str(pt), families=["padding"], sha="a" * 40)]
    assert [r["args"][-1] for r in pad1] == ["s1"] * 5 and all(r["script"].endswith("target_static_padded_obj.sbatch") for r in pad1)
    monkeypatch.delenv("SISTEER_MODELS_TABLE")
    # the matrix rows a real 05 renders from the block match these args: select_rows over the printed block yields exactly these cells
    import g8k_cells as M
    import g8k_views as V
    roster = front_door_roster()
    pins = {sid: {"object_sha256": f"{i:064x}", "source": "synthetic"} for i, sid in enumerate(
        [s for v in VIEWS for s in roster["targets"][KEY]["views"][v]["cells"].values() if s], start=1)}
    mrows = V.select_rows(roster, {KEY: pins}, {KEY: list(VIEWS)}, schema=M.MATRIX_SCHEMA_2, trigger_threshold=M.TRIGGER_THRESHOLD,
                          wave=M.front_door_wave(KEY), where="front door targets test")
    for view in VIEWS:
        planned = [r["args"][2] for r in ST.unit_plan(KEY, view, table=table, families=["gsm8k"], sha="a" * 40)]
        assert planned == [r["cell"] for r in mrows if r["view"] == view], view    # every planned cell IS a matrix row, in order
    assert all(r["fitted"] for r in mrows if r["cell"] not in ("none", "prompted__safety_refusal", "prompted__si_constitution"))


# --------------------------------------------------------------------------- item 6: every launcher admits the front door's matrix
def test_every_gsm8k_launcher_admits_a_front_door_matrix_before_sourcing_and_refuses_a_foreign_name():
    import g8k_cells as M
    G8K = REPO / "scripts" / "sdfbase" / "gsm8k"
    env = {"PATH": os.environ["PATH"], "HOME": str(REPO)}
    sha = "0" * 40
    argvs = {"gsm8k_thinking.sbatch": [KEY, "battery", "none", "8192", sha, "0", "0"],
             "dry_run_prod.sbatch": [sha],
             "ladder_plan.sbatch": [sha, "", sha],
             "smoke_not_available.sbatch": [KEY, "battery", sha, "", "", ""]}
    for name, args in argvs.items():
        src = (G8K / name).read_text()
        m = re.search(r'case "\$G8K_MATRIX" in ([a-z_|]+)\) ;;', src)
        assert m and sorted(m.group(1).split("|")) == sorted(M.MATRICES), name                    # the pinned first arm is untouched
        assert "frontdoor_*) ;;" in src.split(m.group(0), 1)[1].split("esac", 1)[0], name          # the second arm admits the front door
        assert src.index('case "$G8K_MATRIX"') < src.index("common.sh"), name
        r = subprocess.run(["bash", str(G8K / name), *args, "bogus"], capture_output=True, text=True, env=env)
        assert r.returncode == 1 and "matrix must be one of default|llama|itdoc|qwen_seedrep" in r.stderr and "frontdoor_<models.yaml key>" in r.stderr, name
        r = subprocess.run(["bash", str(G8K / name), *args, f"frontdoor_{KEY}"], capture_output=True, text=True, env=env)
        assert "matrix must be one of" not in r.stderr, (name, r.stderr[-600:])                      # past the case: refused later, elsewhere
    # the grading host dry run admits the NAME (its refusal is then the absent document, never "not a registered matrix")
    tmp = REPO / "out" / "g8k_dry_fd_probe"
    probe = [PY, str(G8K / "g8k_dry_run.py"), "--stage", "pre", "--sha", sha, "--tmp", str(tmp)]
    env2 = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    r = subprocess.run([*probe, "--matrix", f"frontdoor_{KEY}"], capture_output=True, text=True, cwd=REPO, env=env2)
    assert "not a registered matrix" not in r.stderr
    if M.resolve_matrix(f"frontdoor_{KEY}").exists():           # a tree where 05 --step configs rendered the matrix: the dry run RUNS it
        assert r.returncode == 0 and "STAGE: pre rows=" in r.stdout and f"tag={KEY}_" in r.stdout, r.stdout[-800:] + r.stderr[-800:]
    else:
        assert r.returncode != 0 and "no frozen matrix at" in r.stderr and f"matrix_frontdoor_{KEY}.json" in r.stderr, r.stderr[-800:]
    r = subprocess.run([*probe, "--matrix", "frontdoor_no_such_model"], capture_output=True, text=True, cwd=REPO, env=env2)
    assert r.returncode != 0 and "not a front-door model" in r.stderr, r.stderr[-800:]
    r = subprocess.run([*probe, "--matrix", "bogus"], capture_output=True, text=True, cwd=REPO, env=env2)
    assert r.returncode != 0 and "neither a matrix name" in r.stderr and "frontdoor_<models.yaml key>" in r.stderr, r.stderr[-800:]
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    root = tmp_path_factory.mktemp("fdt")
    site = root / "site"
    shutil.copytree(REPRO / "site.example", site)
    with open(site / "config.env", "a") as fh:
        fh.write(f"\nSISTEER_STATE_DIR={root / 'state'}\nSISTEER_PYTHON={PY}\nSISTEER_REGISTRY_ROOT={root / 'registry'}\n")
    return site


def test_09_plans_the_dose_views_gsm8k_rows_and_05_names_the_gsm8k_generators(site):
    src9 = (REPRO / "09_dose_ladder.sh").read_text()
    assert "EVALS=static,multiturn,controls;" in src9 and 'plan --model "$K" --view "$view" --families gsm8k' in src9   # gsm8k: opt-in only
    assert "the opt-in gsm8k" in src9
    # the DEFAULT plans the legs the 09 stage names (statics in the sampled form, crescendo + actorattack, controls) and NO gsm8k unit
    r = run_stage("09_dose_ladder.sh", "--dry-run", "--alpha", "0.75", "--model", KEY, site=site, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    units = [ln.split()[3] for ln in r.stdout.splitlines() if ln.startswith("submitted ")]
    assert units and not any(".gsm8k." in u for u in units) and "gsm8k_thinking.sbatch" not in r.stdout, units
    # the single-GPU legs under the stage's own names; the multi-turn legs as the dose view's multiturn FAMILY — 06's names for the pool members
    # and the targets, run through 06_eval_waves.sh --view <dose view> --family multiturn (gap-09-attacker-pool: the pool the legs bind to is
    # started, placed and torn down with them): crescendo + actorattack, never pair
    own = {u.split(".")[2].split("_")[0] for u in units if u.startswith("ladder.")}
    mt = {u.split(".")[1] for u in units if u.startswith("multiturn.") and ".attacker" not in u}
    assert own == {"static", "controls"} and mt == {"crescendo", "actorattack"} and any(u.startswith("multiturn.attacker") for u in units), units
    assert "DRY-RUN: mark-placed mt-d0p75-cluster_a:" in r.stdout and "wait-pool --group mt-d0p75_relu-cluster_b" in r.stdout
    # ... and the opt-in plans them: one gsm8k_thinking.sbatch per cell of each dose view on the front door's matrix, the roster's cell id —
    # the dose view's own `none` row included: under the single-phase recipe every wave carries its own unsteered leg
    r = run_stage("09_dose_ladder.sh", "--dry-run", "--alpha", "0.75", "--evals", "gsm8k", "--model", KEY, site=site, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    units = [ln.split()[3] for ln in r.stdout.splitlines() if ln.startswith("submitted ")]
    assert sorted(units) == sorted(["ladder.d0p75.gsm8k.none", "ladder.d0p75.gsm8k.difference_lora", "ladder.d0p75_relu.gsm8k.none", "ladder.d0p75_relu.gsm8k.difference_relu"]), units
    assert f"gsm8k_thinking.sbatch {KEY} d0p75 constv2-mix-full__lora_r4_ce_minus_ctl 8192 " in r.stdout and f" 1 0 frontdoor_{KEY}" in r.stdout
    assert f"gsm8k_thinking.sbatch {KEY} d0p75 none 8192 " in r.stdout and f" 0 0 frontdoor_{KEY}" in r.stdout
    assert f"gsm8k_thinking.sbatch {KEY} d0p75_relu constv2-mix-full__psr_ce_all_minus_ctl 8192 " in r.stdout
    assert "F22_REGISTRY_EVALS=gsm8k" in r.stdout and "--export=" not in r.stdout
    r = run_stage("09_dose_ladder.sh", "--dry-run", "--alpha", "0.75", "--evals", "static,bogus", "--model", KEY, site=site, timeout=600)
    assert r.returncode != 0 and "the opt-in gsm8k" in r.stdout + r.stderr
    # 05 --step configs runs the gsm8k generators for the front door's matrix once the roster carries a view (pending note before)
    src5 = (REPRO / "05_stage_roster.sh").read_text()
    for needle in ('make_gsm8k_matrix.py" --matrix-name "frontdoor_$K"', 'make_gsm8k_overlays.py" --matrix "frontdoor_$K"', "make_padding_configs.py",
                   'the GSM8K matrix frontdoor_$K + overlays are not generated yet'):
        assert needle in src5, needle
    r = run_stage("05_stage_roster.sh", "--dry-run", "--step", "configs", "--model", KEY, site=site, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    carried = KEY in fr.load_roster(REPO / "configs" / "sdfbase_full22_roster.yaml")["targets"]
    dry = [ln for ln in r.stdout.splitlines() if ln.startswith("DRY-RUN:")]
    assert any(f"make_gsm8k_matrix.py --matrix-name frontdoor_{KEY}" in ln for ln in dry) == carried, dry
    assert any(f"make_gsm8k_overlays.py --matrix frontdoor_{KEY}" in ln for ln in dry) == carried
    assert (f"the GSM8K matrix frontdoor_{KEY} + overlays are not generated yet" in r.stdout) == (not carried)
    assert any("make_full22_configs.py" in ln for ln in dry) and any("make_padding_configs.py" in ln for ln in dry) and any("make_misalign_manifest.py" in ln for ln in dry)
    r = run_stage("05_stage_roster.sh", "--dry-run", "--step", "roster", "--model", KEY, site=site, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    if not carried:
        assert "COMPLETE block" in r.stdout and f"  {KEY}:" in r.stdout and "arditi_allpos: arditi__ar_c0p3__arditi_allpos" in r.stdout
        assert "body_config: configs/sdfbase_transfer_qwen35_4b_r3.yaml" in r.stdout


# --------------------------------------------------------------------------- items 8 + 9
def test_provenance_hunk_line_counts_the_regenerated_patch():
    # counted the way the provenance line itself says: grep -c '^+[^+]' / '^-[^-]' / '^@@' (an added / removed EMPTY line is not counted)
    patch = (REPO / "scripts" / "sdfbase" / "seedrep" / "GRADER_DIFF.patch").read_text(encoding="utf-8").splitlines()
    plus = sum(1 for ln in patch if re.match(r"^\+[^+]", ln))
    minus = sum(1 for ln in patch if re.match(r"^-[^-]", ln))
    hunks = sum(1 for ln in patch if ln.startswith("@@"))
    prov = (REPO / "scripts" / "sdfbase" / "seedrep" / "GRADER_PROVENANCE.md").read_text(encoding="utf-8")
    assert f"+{plus} / \u2212{minus} lines in {hunks} hunks" in prov, (plus, minus, hunks)
    assert "+823 / \u221218 lines in 20 hunks" not in prov


def test_shipped_generation_config_test_model_rows_carry_no_identifier_and_the_file_is_scrubbed():
    doc = json.loads((REPO / "configs" / "sdfbase_shipped_generation_config.json").read_text(encoding="utf-8"))
    text = json.dumps(doc)
    # the demo serves the PAPER's Qwen3.5-4B checkpoint, whose row states `ships_none` at the table's pinned revision: the backing of the
    # demo's `serve.generation_config: null` (a checkpoint shipping none needs its row stated, never guessed); no row of the retired
    # configured-not-run model remains
    ck = TABLE[KEY]["serve"]["checkpoint"]
    row = doc["models"][ck["hf_repo"]]
    assert row["revision"] == ck["revision"] and row["state"] == "ships_none" and "sampler_keys" not in row
    served = {m["serve"]["checkpoint"]["hf_repo"] for m in TABLE.values()}                           # the table's served checkpoints (the demo's IS the paper's Qwen)
    assert served <= set(doc["models"]) and set(doc["models"]) - served == {"zai-org/GLM-4.6V-Flash"}   # + the paper's GLM vendor panel; no row of any other model
    assert not token_hits(text, PROCESS_TOKENS), token_hits(text, PROCESS_TOKENS)
    scrub = (REPO / "reproduce" / "tools" / "scrub_grep.sh").read_text()
    assert "configs/sdfbase_shipped_generation_config.json" in scrub
    test_src = (REPO / "tests" / "test_no_personal_identifiers.py").read_text()
    assert '"configs/sdfbase_shipped_generation_config.json"' in test_src
