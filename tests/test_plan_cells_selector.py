"""The unsteered model is not re-run — every unit not yet launched drops `none`; and the two-ReLU-cell g5678 GO could not pass the front
door: 06 had no --cells, target_attacks.sbatch's positional 7 "" meant ALL cells including none, and f22_registry_register registers the
whole config unless F22_REGISTRY_CELLS. ONE validated selection: `stage.py plan --cells`
(06 --cells) writes the cells into every registering target row whose job takes a cells positional (target_attacks positional 7;
target_sampled / target_static positional 5) AND into the job env F22_REGISTRY_CELLS, so the job serves and registers exactly those cells;
pool members carry none; a family whose job takes no cells positional refuses by name; a cell outside the view config's explicit_cells
refuses by name; ids, groups and clusters are untouched (the placement identity does not change). The job side exports F22_REGISTRY_CELLS from
its own cells positional before registration. The registration narrowing (common.sh) is run for real on the steered view's config: exactly the two
ReLU cells, never none."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO  # noqa: E402

sys.path.insert(0, str(REPO / "reproduce" / "lib"))
import stage as ST  # noqa: E402

KEY = "qwen35_4b_demo"                                                              # the front-door model (models.front_door_keys())
VIEW = "s3"                                                                     # a steered-wave view of this module's roster: the two ReLU parents at alpha 1, no unsteered leg
SHA = "a" * 40
AB = ["cluster_a", "cluster_b"]
RELU = ["constv2ctl2-full__psr_ce_all", "constv2-mix-full__psr_ce_all"]       # control_relu, constitution_relu
CSV = ",".join(RELU)
U0 = "battery"
U0CELLS = ["none", "prompted__safety_refusal", "prompted__si_constitution"]      # the unsteered view: none + the two prompted SIs


def _synthetic_roster(root: Path) -> Path:
    """A wave roster of the front-door model's views for this module: a steered-wave view (VIEW) enumerating exactly the two ReLU parents
    and the u0 view, each with its wave config (explicit_cells) and its Petri drive manifest. The committed roster carries no block for
    the front-door model (configured end to end, NOT run), so the selector is exercised on a fixture of its own (F22_ROSTER)."""
    (root / "configs" / "capeval" / "manifests").mkdir(parents=True, exist_ok=True)
    views = {}
    for v, cells, man in ((VIEW, RELU, [f"si_{c}" for c in RELU]), (U0, U0CELLS, ["none", *(f"si_{c}" for c in U0CELLS[1:])])):
        cfg = root / "configs" / f"sdfbase_full22_{KEY}_{v}.yaml"; cfg.write_text(yaml.safe_dump({"explicit_cells": list(cells)}))
        m = root / "configs" / "capeval" / "manifests" / f"petri_full22_{KEY}_{v}.yaml"; m.write_text(yaml.safe_dump({"petri": {"cells": list(man)}}))
        views[v] = {"config": str(cfg), "manifest": str(m), "cells": {c: f"base__{c}" for c in cells if c != "none"}}
    roster = root / "configs" / "sdfbase_full22_roster.yaml"
    roster.write_text(yaml.safe_dump({"targets": {KEY: {"views": views}}}))
    return roster


@pytest.fixture(scope="module", autouse=True)
def roster(tmp_path_factory):
    p = _synthetic_roster(tmp_path_factory.mktemp("roster")); old = os.environ.get("F22_ROSTER"); os.environ["F22_ROSTER"] = str(p)
    yield p
    if old is None:
        os.environ.pop("F22_ROSTER", None)
    else:
        os.environ["F22_ROSTER"] = old


@pytest.fixture(autouse=True)
def _knobs(monkeypatch):
    monkeypatch.setenv("SISTEER_SHARD_GROUPS", "0,1,2,3,4 5,6,7,8")
    monkeypatch.setenv("SISTEER_MT_ATTACKER_GPUS", "2")


def test_the_steered_view_enumerates_exactly_the_two_relu_cells():
    assert ST.view_cells_of(KEY, VIEW) == RELU                                                            # s3r-fix: a steered-wave view lists no none


def test_plan_cells_writes_the_selection_into_every_target_job_and_its_registration_env():
    plain = ST.unit_plan(KEY, VIEW, families=["multiturn", "static", "controls"], sha=SHA, clusters=AB)
    rows = ST.unit_plan(KEY, VIEW, families=["multiturn", "static", "controls"], sha=SHA, clusters=AB, cells=RELU)
    assert [r["logical_work_id"] for r in rows] == [r["logical_work_id"] for r in plain]                  # the same units, the same order
    for r, p in zip(rows, plain):
        assert (r["group"], r["cluster"], r["kind"], r["name"]) == (p["group"], p["cluster"], p["kind"], p["name"])   # placement identity untouched
        if r["kind"] == "attacker":
            assert r["args"] == p["args"] and "F22_REGISTRY_CELLS" not in r["env"] and "cells" not in r
            continue
        idx = ST.CELLS_POSITIONAL[r["script"]]
        assert r["args"][idx] == CSV and r["env"]["F22_REGISTRY_CELLS"] == CSV and r["cells"] == RELU
        assert r["args"][:idx] == p["args"][:idx] and r["args"][idx + 1:] == p["args"][idx + 1:]           # only the cells positional moved
    mt = [r for r in rows if r["family"] == "multiturn" and r["kind"] == "target"]
    assert len(mt) == 6 and all(r["script"].endswith("target_attacks.sbatch") and r["args"][6] == CSV for r in mt)
    aa = [r for r in mt if r["unit"].startswith("actorattack/")]
    assert aa and all(len(r["args"]) == 8 and r["args"][7] == str(ST.MT_ACTORATTACK_WINDOW) for r in aa)   # the window stays positional 8
    others = [r for r in mt if not r["unit"].startswith("actorattack/")]
    assert others and all(len(r["args"]) == 7 for r in others)                                             # positional 7 added where it was implicit
    sc = [r for r in rows if r["family"] in ("static", "controls")]
    assert len(sc) == 4 and all(r["args"][4] == CSV and r["args"][5] == VIEW for r in sc)                  # <key> <phase> <sha> <nshards> <cells> <view>
    # the env is a copy per row: the plain plan's rows do not carry the key
    assert all("F22_REGISTRY_CELLS" not in p["env"] for p in plain)


def test_plan_cells_refusals_are_by_name():
    with pytest.raises(ValueError, match=f"not in {KEY}/{VIEW}'s explicit_cells"):
        ST.unit_plan(KEY, VIEW, families=["multiturn"], sha=SHA, clusters=AB, cells=["constv2ctl2-full__psr_ce_all", "no_such_cell"])
    with pytest.raises(ValueError, match="named twice"):
        ST.unit_plan(KEY, VIEW, families=["multiturn"], sha=SHA, clusters=AB, cells=RELU + [RELU[0]])
    with pytest.raises(ValueError, match="empty selection"):
        ST.unit_plan(KEY, VIEW, families=["multiturn"], sha=SHA, clusters=AB, cells=[])
    # a family whose job takes no cells positional cannot honour the selection: refused by name, never a full-config launch
    with pytest.raises(ValueError, match="--cells is not plumbed for family 'agentic'"):
        ST.unit_plan(KEY, VIEW, families=["agentic"], sha=SHA, clusters=AB, cells=RELU)


def test_plan_cli_carries_cells_and_06_forwards_them(capsys):
    assert ST.main(["plan", "--model", KEY, "--view", VIEW, "--families", "multiturn", "--sha", SHA, "--clusters", ",".join(AB), "--json", "--cells", CSV]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert all(r["args"][6] == CSV and r["env"]["F22_REGISTRY_CELLS"] == CSV for r in rows if r["kind"] == "target")
    assert ST.main(["plan", "--model", KEY, "--view", VIEW, "--families", "multiturn", "--sha", SHA, "--clusters", ",".join(AB), "--json", "--cells", "none,bogus"]) == 2
    assert f"not in {KEY}/{VIEW}'s explicit_cells" in capsys.readouterr().err
    w = (REPO / "reproduce" / "06_eval_waves.sh").read_text()
    assert subprocess.run(["bash", "-n", str(REPO / "reproduce" / "06_eval_waves.sh")]).returncode == 0
    assert '--cells) CELLS_SEL="$2"; CELLS_GIVEN=1; shift 2;;' in w                                               # cells-fix-2: given vs absent
    assert '${CELLS_SEL:+--cells "$CELLS_SEL"}' in w.split("PLAN_ARGS+=(--ledger", 1)[1].split("\n", 1)[0]     # the ONE plan-args line carries it
    assert "#   --cells " in w                                                                                     # documented in the usage header


def test_the_job_side_exports_the_registration_cells_from_its_positional_before_registering():
    for rel, pos in (("scripts/sdfbase/full22/target_attacks.sbatch", 'CELLS_ARG="${7:-}"'), ("scripts/sdfbase/full22/sampled/target_sampled.sbatch", 'CELLS_ARG="${5:-}"'),
                     ("scripts/sdfbase/full22/target_static.sbatch", 'CELLS_ARG="${6:-}"')):                         # cells-fix-2: the third plumbed job
        src = (REPO / rel).read_text()
        assert subprocess.run(["bash", "-n", str(REPO / rel)]).returncode == 0
        i_pos = src.index(pos)
        i_exp = src.index('[ -n "$CELLS_ARG" ] && export F22_REGISTRY_CELLS="${F22_REGISTRY_CELLS:-$CELLS_ARG}"')
        i_init = src.index("f22_init ", i_pos)
        assert i_pos < i_exp < i_init, rel                                   # parsed, exported, THEN f22_init / resolve_config (where f22_registry_register runs)


def test_the_registration_narrowing_on_the_steered_config_yields_exactly_the_two_relu_cells(tmp_path, roster):
    """common.sh's F22_REGISTRY_CELLS narrowing (the python the register function runs), executed for real on the steered view's wave config."""
    common = (REPO / "scripts/sdfbase/full22/common.sh").read_text()
    start = common.index("python3 - \"$CFG\" \"$reg_cfg\" \"$F22_REGISTRY_CELLS\" <<'PY'")
    cont = common.index("cannot narrow $CFG to cells", start)                       # the `|| { … }` continuation line precedes the heredoc body
    body = common[common.index("\n", cont) + 1:common.index("\nPY\n", start)]
    cfg = Path(yaml.safe_load(roster.read_text())["targets"][KEY]["views"][VIEW]["config"])
    assert yaml.safe_load(cfg.read_text())["explicit_cells"] == RELU                                                # s3r-fix: the steered-wave config lists no none
    out = tmp_path / "narrowed.yaml"
    r = subprocess.run([sys.executable, "-", str(cfg), str(out), CSV], input=body, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    narrowed = yaml.safe_load(out.read_text())
    assert narrowed["explicit_cells"] == RELU and narrowed["registry_narrowed_cells"] == RELU and narrowed["registry_narrowed_from"] == str(cfg)
    assert "none" not in narrowed["explicit_cells"]
    r = subprocess.run([sys.executable, "-", str(cfg), str(tmp_path / "bad.yaml"), "constv2ctl2-full__psr_ce_all,bogus"], input=body, capture_output=True, text=True)
    assert r.returncode != 0 and "['bogus'] are not in" in r.stderr
    assert yaml.safe_load(cfg.read_text())["explicit_cells"] == RELU                                             # the view's config is byte-untouched (no none: s3r-fix)
