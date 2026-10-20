"""cells-fix-2 (both high). P1: CELLS_POSITIONAL mapped target_static.sbatch's cells to zero-based 4, which that script
parses as NSHARDS (`<key> <view> <families> <sha> [nshards] [cells]`) — a --cells selection on the static/bon/store512 row would have replaced
its shard count with the cell CSV and left CELLS_ARG empty (the full view incl. none served while the registration was narrowed by env, or a
non-numeric nshards failure); the index is 5 and the script exports F22_REGISTRY_CELLS from it before f22_init like its siblings. Every planned
row's cells positional is checked against the REAL sbatch parser here: the script's own positional line, run under bash with the planned args.
P2: `plan --cells ''` / `','` / `'a,,b'` and 06 `--cells ''` collapsed to "no selection" and launched the full config including none —
absent and explicitly-empty are distinct; an empty or blank token refuses by name in both CLIs."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO, REPRO, VENV_PY, run_stage  # noqa: E402

sys.path.insert(0, str(REPO / "reproduce" / "lib"))
import stage as ST  # noqa: E402

KEY = "qwen35_4b_demo"                                                              # the front-door model (models.front_door_keys())
VIEW = "s3r"                                                                    # a steered-wave view BY NAME (s<N>r: the ReLU-parent subset) of this module's roster: the two ReLU parents at alpha 1, no unsteered leg — the demo is single-phase, so only the s<N>r name makes a steered wave here
SHA = "a" * 40
AB = ["cluster_a", "cluster_b"]
RELU = ["constv2ctl2-full__psr_ce_all", "constv2-mix-full__psr_ce_all"]
CSV = ",".join(RELU)
U0 = "battery"                                                                        # the baseline wave (the single-phase recipe's one wave): its dry run needs no admission gate
U0CELLS = ["none", "prompted__safety_refusal", "prompted__si_constitution"]      # this module's synthetic baseline config: none + the two prompted SIs


def _synthetic_roster(root: Path) -> Path:
    """A wave roster of the front-door model's views for this module: a steered-wave view (VIEW, an s<N>r name) enumerating exactly the two ReLU
    parents and the baseline wave battery, each with its wave config (explicit_cells) and its Petri drive manifest. The committed roster carries no block for
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
U0CSV = ",".join(U0CELLS[1:])                                                    # the baseline wave's two prompted cells (view_cells_of: none + these)
EXPORT = '[ -n "$CELLS_ARG" ] && export F22_REGISTRY_CELLS="${F22_REGISTRY_CELLS:-$CELLS_ARG}"'


@pytest.fixture(autouse=True)
def _knobs(monkeypatch):
    monkeypatch.setenv("SISTEER_SHARD_GROUPS", "0,1,2,3,4 5,6,7,8")
    monkeypatch.setenv("SISTEER_MT_ATTACKER_GPUS", "2")


def _parse_with_the_script(script: Path, args):
    """Run the sbatch's OWN positional-parse block — the `${1:?` line through the line that assigns CELLS_ARG (one or two consecutive lines of
    plain assignments: bon_climb.sbatch assigns NSHARDS on the first and CELLS_ARG on the second) — under bash with the planned args: (NSHARDS, CELLS_ARG)."""
    lines = script.read_text().splitlines()
    i = next(k for k, l in enumerate(lines) if '"${1:?' in l); j = next(k for k, l in enumerate(lines) if 'CELLS_ARG="${' in l)
    assert i <= j <= i + 1, (script.name, i, j)
    r = subprocess.run(["bash", "-c", "; ".join(lines[i:j + 1]) + '; printf "%s|%s" "${NSHARDS:-}" "${CELLS_ARG:-}"', "_", *args], capture_output=True, text=True)
    assert r.returncode == 0, (script.name, r.stderr)
    nshards, cells = r.stdout.split("|", 1)
    return nshards, cells


def test_every_planned_cells_positional_lands_on_the_scripts_own_cells_parameter():
    rows = ST.unit_plan(KEY, VIEW, families=["multiturn", "static", "controls", "petri", "bon"], sha=SHA, clusters=AB, cells=RELU)   # the petri proxy is plumbed too; the BoN climb since the rerun pair
    targets = [r for r in rows if r["kind"] == "target" and r.get("registers", True)]
    seen = set()
    for r in targets:
        script = REPO / r["script"]; seen.add(r["script"])
        nshards, cells = _parse_with_the_script(script, r["args"])
        assert cells == CSV, (r["unit"], r["script"], r["args"])                         # the CSV lands on the parameter the script calls CELLS_ARG
        if r["script"].endswith("petri_proxy.sbatch"):
            assert nshards == "" and r["args"][3] == "" and len(r["args"]) == 5, (r["unit"], r["args"])   # <key> <view> <sha> [pass] [cells]: the pass stays positional 4, no shard count
        else:
            assert re.fullmatch(r"\d+", nshards), (r["unit"], nshards)                   # and the shard count is still a number (P1 clobbered it)
        assert r["env"]["F22_REGISTRY_CELLS"] == CSV
    assert seen == set(ST.CELLS_POSITIONAL), seen                                         # every plumbed script is exercised by the real plan
    bon = [r for r in targets if r["script"].endswith("target_static.sbatch")]
    assert bon and all(r["args"][4] == "9" and r["args"][5] == CSV and len(r["args"]) == 6 for r in bon), [r["args"] for r in bon]
    assert ST.CELLS_POSITIONAL["scripts/sdfbase/full22/target_static.sbatch"] == 5
    # the static job exports the registration cells from ITS positional before f22_init, like its two siblings
    src = (REPO / "scripts/sdfbase/full22/target_static.sbatch").read_text()
    assert subprocess.run(["bash", "-n", str(REPO / "scripts/sdfbase/full22/target_static.sbatch")]).returncode == 0
    i_pos = src.index('NSHARDS="${5:-9}"; CELLS_ARG="${6:-}"'); i_exp = src.index(EXPORT); i_init = src.index("f22_init ", i_pos)
    assert i_pos < i_exp < i_init
    # and the plain plan (no selection) leaves every positional as it was: a job without a cells positional serves the whole view
    plain = ST.unit_plan(KEY, VIEW, families=["multiturn", "static", "controls", "petri"], sha=SHA, clusters=AB)
    assert [p["args"] for p in plain if p["script"].endswith("target_static.sbatch")] == [[KEY, VIEW, "bon", SHA, "9"]]
    assert [p["args"] for p in plain if p["script"].endswith("petri_proxy.sbatch")] == [[KEY, VIEW, SHA, "", CSV]]           # A steered-wave proxy is planned WITH the view's measured cells (its config's explicit_cells), no pass


def test_the_plan_cli_keeps_absent_and_empty_apart_and_refuses_blank_tokens(capsys):
    assert ST.cells_selection(None) is None and ST.cells_selection("a,b") == ["a", "b"] and ST.cells_selection(" a , b ") == ["a", "b"]
    for bad in ("", ",", "a,,b", " , ", "a,"):
        with pytest.raises(ValueError, match="empty selection or a blank token"):
            ST.cells_selection(bad)
    base = ["plan", "--model", KEY, "--view", VIEW, "--families", "multiturn,static", "--sha", SHA, "--clusters", ",".join(AB), "--json"]
    assert ST.main(base) == 0
    rows = __import__("json").loads(capsys.readouterr().out)
    assert rows and all("F22_REGISTRY_CELLS" not in (r.get("env") or {}) and "cells" not in r for r in rows)        # absent: the full enumeration, no narrowing
    for bad in ("", ",", "a,,b"):
        assert ST.main(base + ["--cells", bad]) == 2, bad
        err = capsys.readouterr().err
        assert "REFUSED: --cells" in err and "empty selection or a blank token" in err, (bad, err)                 # explicitly empty: refused by name, nothing planned
    assert ST.main(base + ["--cells", CSV]) == 0
    rows = __import__("json").loads(capsys.readouterr().out)
    assert all(r["env"]["F22_REGISTRY_CELLS"] == CSV for r in rows if r["kind"] == "target")


def _site(root: Path) -> Path:
    site = root / "site"
    shutil.copytree(REPRO / "site.example", site)
    with open(site / "config.env", "a") as fh:
        fh.write(f"\nSISTEER_STATE_DIR={root / 'state'}\nSISTEER_PYTHON={VENV_PY}\nSISTEER_REGISTRY_ROOT={root / 'registry'}\n")
    return site


def test_06_refuses_an_explicitly_empty_cells_value_before_planning_and_only_that(tmp_path):
    site = _site(tmp_path)
    for bad in ("", ",", "a,,b"):
        r = run_stage("06_eval_waves.sh", "--dry-run", "--view", VIEW, "--family", "multiturn", "--cells", bad, "--model", KEY, site=site, timeout=600)
        out = r.stdout + r.stderr
        assert r.returncode == 2 and f"REFUSED: --cells '{bad}' is an empty selection or names a blank cell" in out, (bad, out[-1500:])
        assert "06 eval_waves:" not in out and "sbatch" not in out, out[-1500:]                        # refused before any plan or submission line
    assert not (tmp_path / "state").exists() or not any((tmp_path / "state").rglob("*.jsonl"))
    # the gate is NARROW: a non-empty value is not its business (an unknown cell is the plan's refusal, by name, on the real path), and the
    # ONE plan-args line still carries the given value; a fresh state's --dry-run stops at the placement-manifest check before that plan runs,
    # so what a real selection does downstream is the landed selector test's (plan --cells) and the live g5678 launch's evidence
    r = run_stage("06_eval_waves.sh", "--dry-run", "--view", U0, "--family", "multiturn", "--cells", "bogus", "--model", KEY, site=site, timeout=600)
    out = r.stdout + r.stderr
    assert "is an empty selection" not in out, out[-1500:]
    w = (REPO / "reproduce" / "06_eval_waves.sh").read_text()
    i_gate = w.index('if [ "$CELLS_GIVEN" = 1 ]; then _c="${CELLS_SEL// /}"; case ",$_c," in *,,*)')
    assert w.index("[ \"$REPRO_HELP\" = 1 ]") < i_gate < w.index("repro_load_config; repro_require_model")               # after the parse, before any config/plan
    assert '${CELLS_SEL:+--cells "$CELLS_SEL"}' in w.split("PLAN_ARGS+=(--ledger", 1)[1].split("\n", 1)[0]
    assert ST.view_cells_of(KEY, U0) == ["none", *U0CSV.split(",")]
