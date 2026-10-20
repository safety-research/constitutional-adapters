"""The BoN RERUN pair of a started wave — a funnel door for a corrected long-SI re-run.

The demo smoke's 512 store (view battery) COMPLETED and registered a designed no-fit certificate for prompted__si_constitution: every one of
its 7,500 slots was refused prompt_over_window at the 4,864-token window the job stood up at. The corrected
windows (bon_long_si.sh) do not repair that store by themselves: `--resubmit` refuses a COMPLETED registering unit (its runs are the unit's
results), `--cells` keeps the logical work id, a same-config job would RESUME the old store dir (<private root>/<private_run>/<cell>, where
bon_errors.jsonl holds the refusals under a `complete: true` marker, so every slot would be redrawn beside them) and register the SAME
job-independent run id (sdfbase_full22-stat-3-<tag>) — old refusals and new responses sharing one store and one identity.
The door: NEW funnel units `store512_constitution_rerun -> climb_constitution_rerun` (afterok), the old run superseded and untouched.
This module pins that door end to end:

(1) `rerun_slug` / `bon_rerun_declaration` (stage.py): the declared names from the cell, every refusal by name; (2) `unit_plan` is byte-identical
without a declaration (the demo's and a paper key's rows) and adds exactly the pair with one — the store under `--family static` as the
store512 row rides, a `f22-stat-<tag>-<slug>-3` name (the `-3` common.sh keys the eval and the `stat-3` key family on), the climb under
`--family bon` afterok on the rerun STORE, both serving the ruling's cells alone (the cells positional + F22_REGISTRY_CELLS; bon_climb.sbatch
joins CELLS_POSITIONAL at 4, its own parser line run under bash) under F22_BON_RERUN=<slug>, their coverage under the rerun's own root;
`06 --units` spells them `bon.store512_<slug>` / `bon.climb_<slug>`; (3) `--cells` keeps a rerun row's cells and refuses another selection;
(4) bon_long_si.sh f22_bon_rerun_root re-roots the job to `<campaign root>/rerun/<slug>` right after f22_init (the sbatch files' OWN
PRIVATE_ROOT / ROOT lines then land exactly where the plan's root_rel reads, and where the closer's bon layout looks: its own
bon_store_path, <wave root>/transcripts_results/sdfbase_full22_<tag>/<cell>), exports the producer-key segment, refuses a mixed job /
a whole-view rerun / a bad slug / an unbound WAVE_ROOT, and is a no-op unset; (5) the registration FOR REAL (common.sh's
f22_registry_register over a synthetic checkout): the rerun store's run id differs from the superseded store's, is the same across two
attempts (job-independent), and carries cap 512 / rung 0 / [512, 4096]; the climb's key is job-independent too
(`sdfbase_full22-bonclimb-<tag>`: one cell registers once across the array) and the rerun's segment makes it new, as the store's does;
(6) `rerun-units` stamps a STARTED inventory once — receipt first (a crash
between leaves a receipt the retry completes), idempotent, refused without an attempt of the original / for a changed declaration — and
`plan --inventory` plans the pair for the declaration's view, so 06's base-view check (frozen units == planned units per family) holds;
(6b) a receipt covers a stamp on its WHOLE declaration + tree, never the pair's name alone: a retry of an
interrupted apply on another tree / ruling is refused by name until it completes the receipt's declaration or passes `--supersede-receipt`,
which appends a receipt naming the orphan superseded; a stamp landed beside a foreign receipt gets its own receipt repaired the same way;
(6c) a receipt is named superseded by its own unique durable `receipt_id` and its FULL reference, never by its
second-resolution write time + tree, which two pairs' receipts (or one pair's under two rulings) written in one second share;
(7) end to end on the integration World: 05 freezes, the stamp lands, and 06's dry run submits each unit by its `--units` name with the
F22_BON_RERUN prefix, the climb afterok on the rerun store."""
from __future__ import annotations

import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO, REPRO, VENV_PY, registry_emitting_env, synthetic_checkout  # noqa: E402
from test_admission import goal_fixture, wave_config  # noqa: E402

sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPRO / "lib"))
import models as M  # noqa: E402
import stage as ST  # noqa: E402

F22 = REPO / "scripts" / "sdfbase" / "full22"
HELPER = F22 / "bon_long_si.sh"
STATIC_SB = F22 / "target_static.sbatch"
CLIMB_SB = F22 / "bon_climb.sbatch"
COMMON = F22 / "common.sh"
CLOSER = F22 / "registry_close_stores.py"
TABLE = str(REPRO / "models.yaml")
MODEL = "qwen35_4b_demo"
VIEW = M.baseline_view(MODEL, TABLE)                       # the single-phase recipe's one wave, never a literal
TAG = f"{MODEL}_{VIEW}"
CELL = "prompted__si_constitution"
SLUG = "constitution_rerun"                                # the declared rerun spelling
OTHER_CELL, OTHER_SLUG = "prompted__safety_refusal", "safety_refusal_rerun"      # a second ruled cell of the same view: another PAIR
SOURCE = "D913 (ii); owner-rulings part 2 (D456) 3b"
SHA, NEW_SHA, OTHER_SHA = "a" * 40, "b" * 40, "c" * 40           # the frozen tree, the rerun's tree, a retry's newer tree
AB = ["cluster_a", "cluster_b"]
STORE_LW, CLIMB_LW = f"{MODEL}/{VIEW}/bon/store512", f"{MODEL}/{VIEW}/bon/climb"
RS_LW, RC_LW = f"{MODEL}/{VIEW}/bon/store512_{SLUG}", f"{MODEL}/{VIEW}/bon/climb_{SLUG}"
RS2_LW, RC2_LW = f"{MODEL}/{VIEW}/bon/store512_{OTHER_SLUG}", f"{MODEL}/{VIEW}/bon/climb_{OTHER_SLUG}"
PRIVATE_RUN = f"sdfbase_full22_{TAG}"
STUBS = 'f22_log(){ echo "[log] $*"; }; f22_fatal(){ echo "FATAL: $*" >&2; exit 1; }; '


def _bash(script: str, *, cwd: Path = REPO, env: dict | None = None) -> subprocess.CompletedProcess:
    """bash from the repo root with the venv's python first on PATH; the helper sourced over the two stubs; never checked (refusals are read)."""
    e = dict(os.environ); e["PATH"] = f"{Path(VENV_PY).parent}:{e['PATH']}"
    e.update(env or {})
    return subprocess.run(["bash", "-c", "set -u; " + STUBS + f"source {shlex.quote(str(HELPER))}; " + script], capture_output=True, text=True, cwd=str(cwd), env=e)


def _cli(*args: str, env: dict | None = None, stage_py: Path | None = None, cwd: Path = REPO) -> subprocess.CompletedProcess:
    """stage.py as the stages call it (a subprocess, so SISTEER_STAGE_CRASH_AT's exit reaches us), no overrides document inherited."""
    e = {k: v for k, v in os.environ.items() if k != ST.RUN_OVERRIDES_ENV}
    e["PYTHONPATH"] = str(REPO / "src") + (os.pathsep + e["PYTHONPATH"] if e.get("PYTHONPATH") else "")
    e["SISTEER_PLAN_SKIP_ARTEFACT_CHECK"] = "1"
    e.update(env or {})
    return subprocess.run([sys.executable, str(stage_py or (REPRO / "lib" / "stage.py")), *args], capture_output=True, text=True, env=e, cwd=str(cwd), timeout=900)


def _decl() -> dict:
    return ST.bon_rerun_declaration([CELL], SOURCE, MODEL, VIEW)


# ---------------------------------------------------------------------------------------------- (1) the slug and the declaration
def test_the_slug_and_the_pairs_names_are_d913s_and_the_declaration_refuses_by_name():
    assert ST.rerun_slug([CELL]) == SLUG                                                         # prompted__si_constitution -> constitution_rerun
    assert ST.rerun_slug(["prompted__safety_refusal"]) == "safety_refusal_rerun"
    assert ST.rerun_slug(["constv2-mix-full__psr_ce_all"]) == "constv2_mix_full_psr_ce_all_rerun"    # one token: a dir, a key segment, a unit name
    assert ST.rerun_slug([CELL, "prompted__safety_refusal"]) == "constitution_safety_refusal_rerun"
    assert ST.RERUN_SLUG_RE.match(SLUG) and ST.BON_RERUN_ENV == "F22_BON_RERUN" and ST.RERUN_ROOT_REL == "rerun"
    d = _decl()
    assert d == {"slug": SLUG, "cells": [CELL], "source": SOURCE, "model": MODEL, "view": VIEW,
                 "of": {"store512": STORE_LW, "climb": CLIMB_LW}, "units": {"store512": RS_LW, "climb": RC_LW}}
    with pytest.raises(ValueError, match="empty cell selection"):
        ST.bon_rerun_declaration([], SOURCE, MODEL, VIEW)
    with pytest.raises(ValueError, match="named twice"):
        ST.bon_rerun_declaration([CELL, CELL], SOURCE, MODEL, VIEW)
    with pytest.raises(ValueError, match=f"not in {MODEL}/{VIEW}'s explicit_cells"):
        ST.bon_rerun_declaration(["no_such_cell"], SOURCE, MODEL, VIEW)
    with pytest.raises(ValueError, match="no --source"):
        ST.bon_rerun_declaration([CELL], "  ", MODEL, VIEW)
    assert ST.inventory_reruns({"reruns": [d]}, VIEW) == [d] and ST.inventory_reruns({"reruns": [d]}, "d0p75") is None
    assert ST.inventory_reruns({}, VIEW) is None and ST.inventory_reruns(None, VIEW) is None


# ---------------------------------------------------------------------------------------------- (2) the plan rows
def _parse_with_the_script(script: Path, args) -> tuple:
    """The sbatch's OWN positional-parse block — the `${1:?` line through the line assigning CELLS_ARG (one or two consecutive lines of plain
    assignments; bon_climb.sbatch assigns NSHARDS on the first and CELLS_ARG / CAP on the second) — under bash with the planned args ->
    (NSHARDS, CELLS_ARG, CAP)."""
    lines = script.read_text().splitlines()
    i = next(k for k, l in enumerate(lines) if '"${1:?' in l); j = next(k for k, l in enumerate(lines) if 'CELLS_ARG="${' in l)
    assert i <= j <= i + 1, (script.name, i, j)
    r = subprocess.run(["bash", "-c", "; ".join(lines[i:j + 1]) + '; printf "%s|%s|%s" "${NSHARDS:-}" "${CELLS_ARG:-}" "${CAP:-}"', "_", *args], capture_output=True, text=True)
    assert r.returncode == 0, (script.name, r.stderr)
    return tuple(r.stdout.split("|"))


def test_the_plan_is_byte_identical_without_a_declaration_and_adds_exactly_the_pair_with_one(monkeypatch):
    monkeypatch.setenv("SISTEER_PLAN_SKIP_ARTEFACT_CHECK", "1")
    for key, view in ((MODEL, VIEW), ("qwen35_4b", "main")):
        plain = ST.unit_plan(key, view, families=["static", "bon"], sha=SHA)
        assert plain == ST.unit_plan(key, view, families=["static", "bon"], sha=SHA, reruns=None) == ST.unit_plan(key, view, families=["static", "bon"], sha=SHA, reruns=[])
        assert [r["unit"] for r in plain] == ["t0", "t1", "store512", "climb"] and not any(r.get("rerun_of") for r in plain)
    rows = ST.unit_plan(MODEL, VIEW, families=["static", "bon"], sha=SHA, reruns=[_decl()])
    assert [r["unit"] for r in rows] == ["t0", "t1", "store512", f"store512_{SLUG}", "climb", f"climb_{SLUG}"]      # each beside the unit it re-runs
    plain = ST.unit_plan(MODEL, VIEW, families=["static", "bon"], sha=SHA)
    assert [r for r in rows if not r.get("rerun_of")] == plain                                              # every other row byte-identical
    rs, rc = (next(r for r in rows if r["unit"] == u) for u in (f"store512_{SLUG}", f"climb_{SLUG}"))
    env = {"F22_BON_DECODE_PRESET": "canonical", "F22_REGISTRY_EVALS": "strongreject_bon", "F22_BON_RERUN": SLUG, "F22_REGISTRY_CELLS": CELL}
    # the store: the store512 row's sibling under `--family static`, FAMILIES=bon, the ruling's cell on positional 6, a `-3` name
    assert (rs["stage_family"], rs["family"], rs["logical_work_id"], rs["script"]) == ("static", "bon", RS_LW, "scripts/sdfbase/full22/target_static.sbatch")
    assert rs["name"] == f"f22-stat-{TAG}-{SLUG}-3" and rs["name"].endswith("-3")
    assert rs["args"] == [MODEL, VIEW, "bon", SHA, "9", CELL] and rs["env"] == env and rs["array"] is None and rs["depends_on"] is None
    assert rs["cells"] == [CELL] and rs["rerun_of"] == STORE_LW and rs["rerun_source"] == SOURCE and rs["registers"] and rs["cluster"] == "*"
    assert rs["coverage"] == {"strongreject_bon": {"layout": "bon_store512", "host": "cluster", "private_run": PRIVATE_RUN, "expected": None, "expected_source": None,
                                                   "path": {"root_rel": f"exp/sdfbase_full22_native/rerun/{SLUG}/transcripts_results/{PRIVATE_RUN}/<cell>"}}}
    assert rs["expected_shards"] == {"strongreject_bon": ["shard00"]}
    # the climb: under `--family bon`, afterok on the rerun STORE, array 0-3, the cell on positional 5 (CELLS_POSITIONAL 4), its own climb root
    assert (rc["stage_family"], rc["family"], rc["logical_work_id"], rc["script"]) == ("bon", "bon", RC_LW, "scripts/sdfbase/full22/bon_climb.sbatch")
    assert rc["name"] == f"f22-bonclimb-{TAG}-{SLUG}" and rc["args"] == [MODEL, VIEW, SHA, "4", CELL] and rc["array"] == "0-3" and rc["env"] == env
    assert rc["depends_on"] == RS_LW and rc["cells"] == [CELL] and rc["rerun_of"] == CLIMB_LW and rc["rerun_source"] == SOURCE
    assert rc["coverage"]["strongreject_bon"]["layout"] == "bon_climb_flat" and rc["coverage"]["strongreject_bon"]["nshards"] == 4
    assert rc["coverage"]["strongreject_bon"]["path"] == {"root_rel": f"exp/sdfbase_full22_native/rerun/{SLUG}/bon_climb/{TAG}"}
    assert rc["expected_shards"] == {"strongreject_bon": ["shard00", "shard01", "shard02", "shard03"]}
    # the cells land on the parameter each script calls CELLS_ARG (the scripts' own parser lines), the shard count untouched, the cap its default
    assert ST.CELLS_POSITIONAL["scripts/sdfbase/full22/bon_climb.sbatch"] == 4 and rc["args"][4] == CELL
    assert _parse_with_the_script(STATIC_SB, rs["args"]) == ("9", CELL, "") and _parse_with_the_script(CLIMB_SB, rc["args"]) == ("4", CELL, "4096")
    # one family at a time, as 06 submits them: the store with `--family static`, the climb with `--family bon`
    st = ST.unit_plan(MODEL, VIEW, families=["static"], sha=SHA, reruns=[_decl()]); bn = ST.unit_plan(MODEL, VIEW, families=["bon"], sha=SHA, reruns=[_decl()])
    assert [r["unit"] for r in st] == ["t0", "t1", "store512", f"store512_{SLUG}"] and [r["unit"] for r in bn] == ["climb", f"climb_{SLUG}"]
    # the TSV the shell iterates spells the `--units` names by the family column (06: u="$fam.${unit//\//.}") and the climb's dependency
    tsv = {l.split("\t")[2]: l.split("\t") for l in ST.plan_tsv(rows).splitlines()}
    assert tsv[f"store512_{SLUG}"][1] == "bon" and tsv[f"climb_{SLUG}"][1] == "bon" and tsv[f"climb_{SLUG}"][-1] == RS_LW and tsv[f"store512_{SLUG}"][-1] == "-"
    assert f"F22_BON_RERUN={SLUG}" in tsv[f"store512_{SLUG}"][13] and f"F22_REGISTRY_CELLS={CELL}" in tsv[f"climb_{SLUG}"][13]
    # the cells-selector keeps the paper's refusal for a family with no cells positional; the climb is plumbed now
    assert "scripts/sdfbase/full22/bon_climb.sbatch" in ST.CELLS_POSITIONAL


# ---------------------------------------------------------------------------------------------- (3) --cells on a rerun row
def test_a_cells_selection_keeps_a_rerun_rows_cells_and_refuses_another_selection(monkeypatch):
    monkeypatch.setenv("SISTEER_PLAN_SKIP_ARTEFACT_CHECK", "1")
    rows = ST.unit_plan(MODEL, VIEW, families=["static", "bon"], sha=SHA, reruns=[_decl()], cells=[CELL])
    by = {r["unit"]: r for r in rows}
    assert by["store512"]["args"] == [MODEL, VIEW, "bon", SHA, "9", CELL] and by["climb"]["args"] == [MODEL, VIEW, SHA, "4", CELL]   # the originals take the selection
    assert by[f"store512_{SLUG}"]["args"] == [MODEL, VIEW, "bon", SHA, "9", CELL] and by[f"climb_{SLUG}"]["args"] == [MODEL, VIEW, SHA, "4", CELL]
    with pytest.raises(ValueError, match="a rerun serves exactly the cells its ruling names"):
        ST.unit_plan(MODEL, VIEW, families=["static", "bon"], sha=SHA, reruns=[_decl()], cells=["none", CELL])
    with pytest.raises(ValueError, match=f"rerun unit climb_{SLUG}"):
        ST.unit_plan(MODEL, VIEW, families=["bon"], sha=SHA, reruns=[_decl()], cells=["none"])


# ---------------------------------------------------------------------------------------------- (4) the job re-roots itself
def _own_line(script: Path, marker: str) -> str:
    return next(l for l in script.read_text().splitlines() if marker in l)


def _closer():
    """The closer as a module (its own file, never a copy): bon_store_path is the one function its bon branch and its store close lock resolve
    a cell's store through — the path both registrations of a cell close."""
    spec = importlib.util.spec_from_file_location("registry_close_stores_bon_rerun_units", str(CLOSER))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def test_the_helper_re_roots_a_rerun_job_exports_the_key_segment_and_refuses_by_name(tmp_path):
    shared = tmp_path / "shared"; campaign = shared / "exp" / "sdfbase_full22_native"; campaign.mkdir(parents=True)
    rerun_root = campaign / "rerun" / SLUG
    # the store job: target_static.sbatch's OWN SERVE_DIR / PRIVATE_ROOT line after the helper re-rooted WAVE_ROOT
    line = _own_line(STATIC_SB, 'PRIVATE_ROOT="$WAVE_ROOT/transcripts_results"')
    r = _bash(f'WAVE_ROOT={shlex.quote(str(campaign))}; F22_BON_RERUN={SLUG}; f22_bon_rerun_root store bon {CELL}; TAG={TAG}; FAMTAG=bon; SLURM_JOB_ID=1; {line}; '
              'echo "WAVE_ROOT=$WAVE_ROOT|F22_WAVE_ROOT=$F22_WAVE_ROOT|SEG=$F22_REGISTRY_KEY_SEGMENTS|PRIVATE_ROOT=$PRIVATE_ROOT|SERVE_DIR=$SERVE_DIR|RERUN_ROOT=$F22_BON_RERUN_ROOT"')
    assert r.returncode == 0, r.stdout + r.stderr
    got = dict(kv.split("=", 1) for kv in r.stdout.splitlines()[-1].split("|"))
    assert got["WAVE_ROOT"] == got["F22_WAVE_ROOT"] == got["RERUN_ROOT"] == str(rerun_root) and rerun_root.is_dir()
    assert got["SEG"] == SLUG and got["PRIVATE_ROOT"] == f"{rerun_root}/transcripts_results" and got["SERVE_DIR"].startswith(f"{rerun_root}/serving/static_{TAG}_bon_")
    assert f"BoN RERUN {SLUG} (store leg; cells {CELL}" in r.stdout and f"{rerun_root}/STOP" in r.stdout and f"sdfbase_full22-stat-3-<tag>-{SLUG}" in r.stdout
    # the store the driver writes (jailbreak_run._bon_defense_dir: <private root>/<private_run>/<cell>) IS the plan row's coverage path under the
    # shared root, and IS where the closer's bon layout looks: the closer's OWN bon_store_path (<wave root>/transcripts_results/
    # sdfbase_full22_<tag>/<cell>), the one function its bon branch resolves a store through — the function itself, over the re-rooted WAVE_ROOT
    rs = next(r for r in ST.unit_plan(MODEL, VIEW, families=["static"], sha=SHA, reruns=[_decl()]) if r["unit"] == f"store512_{SLUG}")
    store = os.path.join(got["PRIVATE_ROOT"], PRIVATE_RUN, CELL)
    assert store == os.path.join(str(shared), rs["coverage"]["strongreject_bon"]["path"]["root_rel"].replace("<cell>", CELL))
    assert "store = bon_store_path(wave_root, tag, obj)" in CLOSER.read_text()                   # the bon branch's one store resolution
    assert store == _closer().bon_store_path(got["WAVE_ROOT"], TAG, CELL) == os.path.join(got["WAVE_ROOT"], "transcripts_results", f"sdfbase_full22_{TAG}", CELL)
    assert not store.startswith(str(campaign / "transcripts_results"))                          # never the superseded store's dir
    # the climb job: bon_climb.sbatch's OWN ROOT / PRIVATE_ROOT line; an earlier segment stays in front; the climb reads the RERUN's store
    line = _own_line(CLIMB_SB, 'ROOT="$WAVE_ROOT/bon_climb/$TAG"')
    r = _bash(f'WAVE_ROOT={shlex.quote(str(campaign))}; F22_BON_RERUN={SLUG}; F22_REGISTRY_KEY_SEGMENTS=x; f22_bon_rerun_root climb "" {CELL}; TAG={TAG}; {line}; '
              'echo "ROOT=$ROOT|PRIVATE_ROOT=$PRIVATE_ROOT|SEG=$F22_REGISTRY_KEY_SEGMENTS"')
    assert r.returncode == 0, r.stdout + r.stderr
    got = dict(kv.split("=", 1) for kv in r.stdout.splitlines()[-1].split("|"))
    rc = next(r for r in ST.unit_plan(MODEL, VIEW, families=["bon"], sha=SHA, reruns=[_decl()]) if r["unit"] == f"climb_{SLUG}")
    assert got["ROOT"] == os.path.join(str(shared), rc["coverage"]["strongreject_bon"]["path"]["root_rel"]) == f"{rerun_root}/bon_climb/{TAG}"
    assert got["PRIVATE_ROOT"] == f"{rerun_root}/transcripts_results" and got["SEG"] == f"x-{SLUG}"
    # unset: a no-op, byte for byte (every other job)
    r = _bash(f'WAVE_ROOT={shlex.quote(str(campaign))}; f22_bon_rerun_root store bon ""; echo "WAVE_ROOT=$WAVE_ROOT|F22_WAVE_ROOT=${{F22_WAVE_ROOT:-unset}}|SEG=${{F22_REGISTRY_KEY_SEGMENTS:-unset}}|RERUN_ROOT=[$F22_BON_RERUN_ROOT]"')
    assert r.returncode == 0 and r.stdout.strip() == f"WAVE_ROOT={campaign}|F22_WAVE_ROOT=unset|SEG=unset|RERUN_ROOT=[]" and "RERUN" not in r.stdout.split("|")[0]
    # the refusals, by name
    for script, why in ((f'WAVE_ROOT={shlex.quote(str(campaign))}; F22_BON_RERUN={SLUG}; f22_bon_rerun_root store pasttense,bon {CELL}', "FAMILIES='pasttense,bon'"),
                        (f'WAVE_ROOT={shlex.quote(str(campaign))}; F22_BON_RERUN={SLUG}; f22_bon_rerun_root climb "" ""', "no cells positional"),
                        (f'WAVE_ROOT={shlex.quote(str(campaign))}; F22_BON_RERUN=a/b; f22_bon_rerun_root store bon {CELL}', "not one token of [A-Za-z0-9_]"),
                        (f'F22_BON_RERUN={SLUG}; f22_bon_rerun_root store bon {CELL}', "WAVE_ROOT is unbound"),
                        (f'WAVE_ROOT={shlex.quote(str(campaign))}; F22_BON_RERUN={SLUG}; f22_bon_rerun_root static bon {CELL}', "unknown leg static")):
        r = _bash(script)
        assert r.returncode == 1 and why in r.stderr, (script, r.stdout, r.stderr)
    # the order in both sbatch files: the helper sourced after f22_init, the re-root BEFORE the caps / exports / the registering f22_resolve_config
    # and before any line derives from WAVE_ROOT; no --export= anywhere (hard rule 1)
    src = STATIC_SB.read_text()
    assert src.index('f22_init "$WAVE_SHA"') < src.index("source scripts/sdfbase/full22/bon_long_si.sh") < src.index('f22_bon_rerun_root store "$FAMILIES" "$CELLS_ARG"') \
        < src.index('f22_bon_read_caps "$KEY" "$VIEW"') < src.index('f22_resolve_config "$KEY" "$VIEW"') < src.index('PRIVATE_ROOT="$WAVE_ROOT/transcripts_results"')
    src = CLIMB_SB.read_text()
    assert src.index('f22_init "$WAVE_SHA"') < src.index("source scripts/sdfbase/full22/bon_long_si.sh") < src.index('f22_bon_rerun_root climb "" "$CELLS_ARG"') \
        < src.index('f22_bon_read_caps "$KEY" "$VIEW"') < src.index('f22_resolve_config "$KEY" "$VIEW"') < src.index('ROOT="$WAVE_ROOT/bon_climb/$TAG"')
    for p in (STATIC_SB, CLIMB_SB, HELPER):
        assert not any("--export=" in l for l in p.read_text().splitlines() if not l.lstrip().startswith("#")), p.name
        assert subprocess.run(["bash", "-n", str(p)]).returncode == 0, p.name


# ---------------------------------------------------------------------------------------------- (5) the registration, for real
def _key(job: str, tag: str, seg: str = "") -> str:
    r = subprocess.run(["bash", "-c", 'source "$1"; f22_registry_producer_key "$2" "$3" "$4"', "_", str(COMMON), job, tag, seg], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def _evals(job: str) -> str:
    r = subprocess.run(["bash", "-c", 'source "$1"; f22_registry_evals_for_job "$2"', "_", str(COMMON), job], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def _checkout(root: Path, key: str = "qwen35_4b") -> Path:
    cfg = yaml.safe_dump({"explicit_cells": ["none", CELL, "prompted__safety_refusal"], "model": "Qwen/Qwen3.5-4B", "private_run": f"sdfbase_full22_{key}",
                          "bon": {"max_new_tokens": 512, "climb_cap": 4096, "n_max": 75, "goal_subset": "jbb"}})
    roster = yaml.safe_dump({"targets": {key: {"views": {"main": {"cells": {CELL: None, "prompted__safety_refusal": None}}}}}})
    return synthetic_checkout(root, configs={f"configs/sdfbase_full22_{key}.yaml": cfg, "configs/sdfbase_full22_roster.yaml": roster, "configs/sdfbase_full22_pins.json": "{}"})


def _register(checkout: Path, shared: Path, *, job_id: str, rerun: bool) -> subprocess.CompletedProcess:
    """target_static.sbatch's registration sequence FOR REAL for the BoN-only store job, narrowed to the cell: f22_init's WAVE_ROOT stands in, then the
    helper's re-root (a rerun) or nothing, the store's exports, common.sh's f22_registry_register (the real emit_cli over the checkout's seeds)."""
    from sisteer.registry import synth
    shared.mkdir(parents=True, exist_ok=True)
    name = f"f22-stat-qwen35_4b-{SLUG}-3" if rerun else "f22-stat-qwen35_4b-3"
    script = textwrap.dedent(f"""
        set -u
        cd {checkout}
        export REPO_DIR={REPO} SHARED={shared} SISTEER_REGISTRY_SPOOL={shared}/registry/records
        export SISTEER_REGISTRY_SUBSTRATE=slurm SISTEER_REGISTRY_CLUSTER_LABEL=cluster_a CLUSTER=slurm
        export SLURM_JOB_ID={job_id} SLURM_JOB_NAME={name} WAVE_SHA={synth.code_sha("a")}
        export KEY=qwen35_4b VIEW=main TAG=qwen35_4b CFG=configs/sdfbase_full22_qwen35_4b.yaml F22_ROSTER=configs/sdfbase_full22_roster.yaml F22_SERVING_PATH=lora_native
        export F22_REGISTRY_EVALS=strongreject_bon FAMILIES=bon F22_REGISTRY_CELLS={CELL} CELLS_ARG={CELL}
        {"export F22_BON_RERUN=" + SLUG if rerun else ""}
        WAVE_ROOT={shared}/exp/sdfbase_full22_native; mkdir -p "$WAVE_ROOT"
        source scripts/sdfbase/full22/common.sh
        f22_log(){{ echo "[log] $*"; }}
        source {HELPER}
        f22_bon_rerun_root store "$FAMILIES" "$CELLS_ARG"
        f22_bon_read_caps "$KEY" "$VIEW"
        [ "$FAMILIES" = bon ] && f22_bon_registry_exports store
        f22_registry_register || true
        f22_bon_assert_config
        echo "REGISTERED pending=${{SISTEER_REGISTRY_RUN_IDENTITY:-}} wave_root=$WAVE_ROOT seg=${{F22_REGISTRY_KEY_SEGMENTS:-}}"
    """)
    env = registry_emitting_env(PATH=f"{Path(VENV_PY).parent}:{os.environ['PATH']}")
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, timeout=300)


def _runs(out: str, shared: Path) -> list:
    pend = re.search(r"REGISTERED pending=(\S+)", out).group(1)
    assert pend and Path(pend).is_file(), out
    return json.loads(Path(pend).read_text())["runs"]


def test_the_rerun_store_registers_under_its_own_job_independent_run_id_beside_the_superseded_stores_for_real(tmp_path):
    # the pure key functions common.sh keys registration on: the `-3` suffix keeps the eval and the `stat-3` family; the segment makes the key new
    assert _evals(f"f22-stat-{TAG}-{SLUG}-3") == "strongreject_bon" == _evals(f"f22-stat-{TAG}-3") == _evals(f"f22-bonclimb-{TAG}-{SLUG}")
    assert _key(f"f22-stat-{TAG}-{SLUG}-3", TAG, SLUG) == f"sdfbase_full22-stat-3-{TAG}-{SLUG}" != _key(f"f22-stat-{TAG}-3", TAG) == f"sdfbase_full22-stat-3-{TAG}"
    # the climb's key is job-independent too (common.sh's `f22-bonclimb-*` row: one cell registers ONCE across
    # the array, never a task's job id; new registrations only) and the rerun's segment makes the rerun's climb a NEW identity beside the
    # superseded climb's, exactly as the store's does — read off the key function itself, never a literal of this module's
    assert _key(f"f22-bonclimb-{TAG}-{SLUG}", TAG, SLUG) == f"sdfbase_full22-bonclimb-{TAG}-{SLUG}" != _key(f"f22-bonclimb-{TAG}", TAG) == f"sdfbase_full22-bonclimb-{TAG}"
    # the registration itself: the superseded store, the rerun's first attempt, the rerun's second attempt
    checkout = _checkout(tmp_path / "co"); shared = tmp_path / "shared"
    r_old = _register(checkout, shared, job_id="983259", rerun=False)
    r_new = _register(checkout, shared, job_id="990001", rerun=True)
    r_again = _register(checkout, shared, job_id="990002", rerun=True)
    for r in (r_old, r_new, r_again):
        assert r.returncode == 0 and "REGISTERED pending=" in r.stdout and "[registry] producer key" not in r.stderr, r.stdout[-1500:] + r.stderr[-1500:]
    old, new, again = (_runs(r.stdout, shared) for r in (r_old, r_new, r_again))
    assert [x["route"]["object"] for x in old] == [x["route"]["object"] for x in new] == [x["route"]["object"] for x in again] == [CELL]   # the cell alone
    assert old[0]["run_id"] != new[0]["run_id"] and new[0]["run_id"] == again[0]["run_id"]              # new identity; the same across attempts
    for x in (old[0], new[0]):                                                                           # the same instrument and ladder facts
        assert (x["plan"]["token_cap_tokens"], x["plan"]["token_cap_rung_index"], x["plan"]["token_cap_ladder_json"]) == (512, 0, [512, 4096]), x["plan"]
    assert old[0]["plan"]["instrument_id"] == new[0]["plan"]["instrument_id"]
    assert f"wave_root={shared}/exp/sdfbase_full22_native/rerun/{SLUG} seg={SLUG}" in r_new.stdout and f"wave_root={shared}/exp/sdfbase_full22_native seg=" in r_old.stdout
    assert (shared / "exp" / "sdfbase_full22_native" / "rerun" / SLUG).is_dir()


# ---------------------------------------------------------------------------------------------- (6) the stamp and the plan
def _freeze(state: Path) -> Path:
    """A started wave's inventory in the freeze's shape (the static + bon families: what the stamp's plan reads), at the frozen tree SHA."""
    (state / "gate").mkdir(parents=True, exist_ok=True)
    units = ST.expected_units(MODEL, SHA, TABLE, families=["static", "bon"], goals_jsonl=str(goal_fixture(state)), wave_config=str(wave_config(state)), labels=AB)
    doc = {"schema": 2, "model": MODEL, "view": VIEW, "code_sha": SHA, "nshards": 9, "plan_knobs": ST.freeze_plan_knobs(), "n_units": len(units), "units": units}
    inv = state / "gate" / "expected_units.json"
    ST.write_atomic(str(inv), doc)
    return inv


def _stamp_args(inv: Path, state: Path, *extra: str) -> list:
    return ["rerun-units", "--out", str(inv), "--cells", CELL, "--source", SOURCE, "--code-sha", NEW_SHA, "--models-table", TABLE,
            "--goals-jsonl", str(state / "goals.jsonl"), "--wave-config", str(state / "wave.yaml"), "--clusters", ",".join(AB), *extra]


def _receipts(state: Path) -> list:
    p = state / "attempts.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip() and json.loads(l).get("event") == "rerun_units"] if p.exists() else []


def test_rerun_units_refuses_to_stamp_over_a_coverage_contract_of_another_version_or_a_stale_digest(tmp_path, monkeypatch):
    """The stamp reads the inventory's coverage-contract record through the gate's own check (version, the three parts, the digest)
    BEFORE the receipt — a record of another version (a rule this tree does not know) or one edited by hand (a digest that no longer
    covers its body) refuses by name, nothing stamped, no receipt written; the record as written stamps and the amendment names it."""
    import coverage_probe as CP                                   # reproduce/lib is on sys.path (the module header)
    monkeypatch.setenv("SISTEER_PLAN_SKIP_ARTEFACT_CHECK", "1")
    state = tmp_path / "state"; inv = _freeze(state)
    ST.append_fsync(str(state / "attempts.jsonl"), json.dumps({"event": "attempt", "logical_work_id": STORE_LW, "attempt_id": "x", "job_id": "983259", "cluster": "cluster_a"}))
    doc = json.loads(inv.read_text())
    good = ST.coverage_contract_record(doc["units"], CP.DONE_PREDICATES, source="frozen", frozen_utc="2026-10-11T04:45:59Z", inventory_code_sha=SHA)
    for bad, words in (({**good, "version": "cc2"}, f"the frozen coverage contract at {inv} is version 'cc2', this tree's coverage contract is version 'cc1'"),
                       ({**good, "sha256": "0" * 64}, f"records sha256 000000000000… but its body digests to {good['sha256'][:12]}… under version cc1")):
        ST.write_atomic(str(inv), {**doc, "coverage_contract": bad}); before = inv.read_bytes()
        r = _cli(*_stamp_args(inv, state))
        assert r.returncode == 2 and r.stderr.startswith("REFUSED: rerun-units: ") and words in r.stderr and "nothing stamped, no receipt written" in r.stderr, r.stdout + r.stderr
        assert inv.read_bytes() == before and _receipts(state) == []
        r = _cli(*_stamp_args(inv, state, "--dry-run"))
        assert r.returncode == 2 and words in r.stderr and _receipts(state) == [], r.stderr           # the dry run refuses the same record
    ST.write_atomic(str(inv), {**doc, "coverage_contract": good})
    r = _cli(*_stamp_args(inv, state))
    assert r.returncode == 0 and f"rerun-units: stamped {RS_LW} -> {RC_LW}" in r.stdout, r.stdout + r.stderr
    after = json.loads(inv.read_text())
    assert after["coverage_contract"] == good and after["reruns"][-1]["coverage_contract_amendment"]["from_sha256"] == good["sha256"] and len(_receipts(state)) == 1


def test_rerun_units_stamps_a_started_inventory_once_receipt_first_and_plan_inventory_reads_the_pair(tmp_path, monkeypatch):
    monkeypatch.setenv("SISTEER_PLAN_SKIP_ARTEFACT_CHECK", "1")
    state = tmp_path / "state"; inv = _freeze(state)
    before = json.loads(inv.read_text())
    assert {u["logical_work_id"] for u in before["units"]} == {f"{MODEL}/{VIEW}/static/t0", f"{MODEL}/{VIEW}/static/t1", STORE_LW, CLIMB_LW}
    # no inventory / no attempt of the original store: refused by name (a wave that has not run its store serves the cell at the corrected window itself)
    r = _cli(*_stamp_args(tmp_path / "nowhere.json", state))
    assert r.returncode == 2 and "REFUSED: rerun-units: no inventory" in r.stderr, r.stderr
    r = _cli(*_stamp_args(inv, state))
    assert r.returncode == 2 and f"{STORE_LW} has no recorded attempt" in r.stderr and "nothing to re-run" in r.stderr, r.stderr
    assert json.loads(inv.read_text()) == before and not (state / "attempts.jsonl").exists()
    ST.append_fsync(str(state / "attempts.jsonl"), json.dumps({"event": "attempt", "logical_work_id": STORE_LW, "attempt_id": "x", "job_id": "983259", "cluster": "cluster_a"}))
    # the other refusals, by name; nothing written
    r = _cli(*_stamp_args(inv, state)[:4], "no_such_cell", *_stamp_args(inv, state)[5:])
    assert r.returncode == 2 and "not in" in r.stderr and "explicit_cells" in r.stderr, r.stderr
    r = _cli(*[a if a != SOURCE else " " for a in _stamp_args(inv, state)])
    assert r.returncode == 2 and "no --source" in r.stderr, r.stderr
    r = _cli(*_stamp_args(inv, state), "--model", "qwen35_4b")
    assert r.returncode == 2 and "never travels to another model" in r.stderr, r.stderr
    assert json.loads(inv.read_text()) == before and _receipts(state) == []
    # the dry run says what it would do and writes nothing
    r = _cli(*_stamp_args(inv, state, "--dry-run"))
    assert r.returncode == 0 and f"DRY-RUN: rerun-units would stamp {RS_LW} -> {RC_LW}" in r.stdout and SOURCE in r.stdout, r.stdout + r.stderr
    assert json.loads(inv.read_text()) == before and _receipts(state) == []
    # the receipt is on the ledger BEFORE the inventory: a crash between the two leaves the receipt, the inventory untouched …
    r = _cli(*_stamp_args(inv, state), env={"SISTEER_STAGE_CRASH_AT": "rerun_after_event"})
    assert r.returncode == 97 and "CRASH INJECTED at rerun_after_event" in r.stderr, r.stdout + r.stderr
    assert json.loads(inv.read_text()) == before and len(_receipts(state)) == 1
    rec = _receipts(state)[0]
    assert (rec["model"], rec["view"], rec["slug"], rec["cells"], rec["source"], rec["code_sha"], rec["at"]) == (MODEL, VIEW, SLUG, [CELL], SOURCE, NEW_SHA, "rerun-units")
    assert rec["of"] == {"store512": STORE_LW, "climb": CLIMB_LW} and rec["units"] == {"store512": RS_LW, "climb": RC_LW}
    # … and the retry completes the stamp without a second receipt
    r = _cli(*_stamp_args(inv, state))
    assert r.returncode == 0 and f"rerun-units: stamped {RS_LW} -> {RC_LW}" in r.stdout and "receipt already on the ledger" in r.stdout, r.stdout + r.stderr
    assert f"--family static --units bon.store512_{SLUG}" in r.stdout and f"--family bon --units bon.climb_{SLUG}" in r.stdout      # the next step, spelled
    after = json.loads(inv.read_text())
    assert len(_receipts(state)) == 1 and after["n_units"] == 6 == len(after["units"])
    assert after["units"][:4] == before["units"] and [u["logical_work_id"] for u in after["units"][4:]] == [RS_LW, RC_LW]            # the originals byte-identical
    assert {k: v for k, v in after.items() if k not in ("units", "n_units", "reruns")} == {k: v for k, v in before.items() if k not in ("units", "n_units")}
    assert len(after["reruns"]) == 1 and {k: after["reruns"][0][k] for k in ("slug", "cells", "source", "model", "view", "of", "units", "code_sha")} == \
        {"slug": SLUG, "cells": [CELL], "source": SOURCE, "model": MODEL, "view": VIEW, "of": rec["of"], "units": rec["units"], "code_sha": NEW_SHA}
    rs, rc = after["units"][4], after["units"][5]
    orig = next(u for u in before["units"] if u["logical_work_id"] == STORE_LW)
    assert (rs["rerun_of"], rs["rerun_source"], rs["cells"], rs["code_sha"], rs["stage_family"], rs["family"], rs["unit"]) == (STORE_LW, SOURCE, [CELL], NEW_SHA, "static", "bon", f"store512_{SLUG}")
    assert (rc["rerun_of"], rc["cells"], rc["code_sha"], rc["stage_family"], rc["array"], rc["sbatch_name"]) == (CLIMB_LW, [CELL], NEW_SHA, "bon", "0-3", f"f22-bonclimb-{TAG}-{SLUG}")
    assert rs["sbatch_name"] == f"f22-stat-{TAG}-{SLUG}-3" and rs["private_run"] is None
    b = rs["evals"]["strongreject_bon"]; o = orig["evals"]["strongreject_bon"]
    assert b["instrument_id"] == o["instrument_id"] and b["expected_shards"] == ["shard00"] and b["coverage"]["expected"] == o["coverage"]["expected"]   # the same instrument and goals
    assert b["coverage"]["path"] == {"root_rel": f"exp/sdfbase_full22_native/rerun/{SLUG}/transcripts_results/{PRIVATE_RUN}/<cell>"} and b["coverage"]["n_max"] == 75
    assert rs["workload"]["args"] == [MODEL, VIEW, "bon", "{sha}", "9", CELL] and rs["workload"]["env"]["F22_BON_RERUN"] == SLUG and rs["workload"]["env"]["F22_REGISTRY_CELLS"] == CELL
    assert rc["workload"]["args"] == [MODEL, VIEW, "{sha}", "4", CELL] and rc["workload"]["array"] == "0-3"
    assert ST.unit_contract(rs)["evals"]["strongreject_bon"]["instrument_id"] == ST.unit_contract(orig)["evals"]["strongreject_bon"]["instrument_id"]
    # idempotent: an identical stamp is `unchanged`; the same pair under another source is refused (stamped once)
    r = _cli(*_stamp_args(inv, state))
    assert r.returncode == 0 and "inventory unchanged (already stamped)" in r.stdout and json.loads(inv.read_text()) == after and len(_receipts(state)) == 1
    r = _cli(*[a if a != SOURCE else "another ruling" for a in _stamp_args(inv, state)])
    assert r.returncode == 2 and "already stamped" in r.stderr and "stamped ONCE" in r.stderr and json.loads(inv.read_text()) == after, r.stderr
    # `plan --inventory` plans the pair for THIS view and nothing for another; a plan without the inventory carries none
    for fams in ("static", "bon", "static,bon"):
        r = _cli("plan", "--model", MODEL, "--view", VIEW, "--families", fams, "--sha", NEW_SHA, "--inventory", str(inv), "--json")
        assert r.returncode == 0, r.stderr
        have = {x["logical_work_id"]: x["expected_shards"] for x in json.loads(r.stdout) if x["registers"]}
        want = {u["logical_work_id"]: {e: ev["expected_shards"] for e, ev in u["evals"].items()} for u in after["units"] if (u.get("stage_family") or u["family"]) in fams.split(",")}
        assert set(want) == set(have) and all(want[lw] == have[lw] for lw in want), (fams, sorted(set(want) ^ set(have)))      # 06's base-view check, per family
    r = _cli("plan", "--model", MODEL, "--view", VIEW, "--families", "static,bon", "--sha", NEW_SHA, "--json")
    assert r.returncode == 0 and not any(x.get("rerun_of") for x in json.loads(r.stdout)), r.stderr
    assert ST.inventory_reruns(after, "d0p75") is None and [d["slug"] for d in ST.inventory_reruns(after, VIEW)] == [SLUG]
    # the planned rerun rows' TSV spells the `--units` names 06 matches
    r = _cli("plan", "--model", MODEL, "--view", VIEW, "--families", "static,bon", "--sha", NEW_SHA, "--inventory", str(inv))
    units = {f"{l.split(chr(9))[1]}.{l.split(chr(9))[2]}" for l in r.stdout.splitlines()}
    assert {f"bon.store512_{SLUG}", f"bon.climb_{SLUG}", "bon.store512", "bon.climb", "static.t0", "static.t1"} == units


def test_the_rerun_receipt_covers_a_stamp_on_its_whole_declaration_and_tree_never_the_pairs_name_alone():
    """rerun_receipt_matches / rerun_receipt_conflicts: a receipt covers a stamp when every field of the declaration
    (model, view, slug, cells, source, the superseded units, the pair) AND the tree equal the stamp's; the same pair's other receipts
    conflict until a later receipt names them superseded (by the receipt's full reference, (6c))."""
    decl = _decl(); rec = ST.rerun_event(decl, NEW_SHA, "rerun-units")
    assert "supersedes" not in rec and ST.rerun_receipt_matches(rec, decl, NEW_SHA)
    assert not ST.rerun_receipt_matches(rec, decl, OTHER_SHA)                                   # another tree
    for k, v in (("source", "another ruling"), ("cells", ["prompted__si_other"]), ("view", "d0p75"), ("model", "qwen35_4b"), ("slug", "other_rerun"),
                 ("of", {"store512": STORE_LW}), ("units", {"climb": RC_LW})):
        assert not ST.rerun_receipt_matches({**rec, k: v}, decl, NEW_SHA), k                  # any field of the declaration
    assert ST.rerun_receipt_conflicts([rec], decl, OTHER_SHA) == [rec] and ST.rerun_receipt_conflicts([rec], decl, NEW_SHA) == []
    assert ST.rerun_receipt_conflicts([{**rec, "slug": "other_rerun"}], decl, OTHER_SHA) == []  # another pair's receipt is neither cover nor conflict
    later = ST.rerun_event(decl, OTHER_SHA, "rerun-units", supersedes=[rec])
    assert later["supersedes"] == [{"receipt_id": rec["receipt_id"], "utc": rec["utc"], "code_sha": NEW_SHA, "model": MODEL, "view": VIEW, "slug": SLUG,
                                 "source": SOURCE, "cells": [CELL]}] == [ST.rerun_receipt_ref(rec)]
    assert ST.rerun_receipt_conflicts([rec, later], decl, OTHER_SHA) == []                    # a superseded receipt conflicts with nothing …
    assert ST.rerun_receipt_conflicts([rec, later], decl, NEW_SHA) == [later]                 # … and the superseding one with the retired declaration


def test_a_retry_of_an_interrupted_apply_on_another_tree_or_ruling_is_refused_until_the_orphan_receipt_is_superseded(tmp_path, monkeypatch):
    """After a crash between the receipt and the inventory write, a retry on a newer tree or under a
    corrected ruling matched the orphan receipt by the pair's name alone, stamped the pair at the NEW tree / ruling and appended nothing —
    the durable ledger attributed the rerun to the old tree. Now: the conflicting orphan refuses by name (the dry run too); the receipt's
    own declaration still completes the apply; `--supersede-receipt` appends a receipt that NAMES the orphan superseded and stamps this
    declaration; a stamp already landed beside a foreign receipt (the old order's outcome) gets its own receipt repaired the same way."""
    monkeypatch.setenv("SISTEER_PLAN_SKIP_ARTEFACT_CHECK", "1")
    state = tmp_path / "state"; inv = _freeze(state)
    before = json.loads(inv.read_text())
    ST.append_fsync(str(state / "attempts.jsonl"), json.dumps({"event": "attempt", "logical_work_id": STORE_LW, "attempt_id": "x", "job_id": "983259", "cluster": "cluster_a"}))
    # the apply under NEW_SHA dies between the receipt and the inventory
    r = _cli(*_stamp_args(inv, state), env={"SISTEER_STAGE_CRASH_AT": "rerun_after_event"})
    assert r.returncode == 97 and json.loads(inv.read_text()) == before and [x["code_sha"] for x in _receipts(state)] == [NEW_SHA], r.stderr
    # the retry on ANOTHER tree is refused by name — before the fix it exited 0, stamped the pair at OTHER_SHA and left the NEW_SHA receipt as the rerun's record
    other = [a if a != NEW_SHA else OTHER_SHA for a in _stamp_args(inv, state)]
    r = _cli(*other)
    assert r.returncode == 2, r.stdout + r.stderr
    assert ("REFUSED: rerun-units" in r.stderr and "another declaration / tree" in r.stderr and "an interrupted apply" in r.stderr
            and NEW_SHA[:12] in r.stderr and OTHER_SHA[:12] in r.stderr and "--supersede-receipt" in r.stderr and "attribute the rerun" in r.stderr), r.stderr
    assert json.loads(inv.read_text()) == before and [x["code_sha"] for x in _receipts(state)] == [NEW_SHA]
    # the same for a corrected ruling, and for other cells under the same slug; the dry run refuses the same way; nothing is written
    r = _cli(*[a if a != SOURCE else "a corrected ruling" for a in _stamp_args(inv, state)])
    assert r.returncode == 2 and "another declaration / tree" in r.stderr and "a corrected ruling" in r.stderr and SOURCE in r.stderr, r.stderr
    r = _cli(*other, "--dry-run")
    assert r.returncode == 2 and "another declaration / tree" in r.stderr and not r.stdout.strip(), r.stdout + r.stderr
    assert json.loads(inv.read_text()) == before and len(_receipts(state)) == 1
    # --supersede-receipt with nothing to supersede is refused (the receipt's own declaration completes the apply without it)
    r = _cli(*_stamp_args(inv, state), "--supersede-receipt")
    assert r.returncode == 2 and "nothing to supersede" in r.stderr and "already on the ledger" in r.stderr, r.stderr
    assert json.loads(inv.read_text()) == before and len(_receipts(state)) == 1
    # the explicit way out: the dry run names what it would retire; the stamp appends a receipt that NAMES the orphan superseded, then lands at OTHER_SHA
    r = _cli(*other, "--supersede-receipt", "--dry-run")
    assert r.returncode == 0 and r.stdout.startswith("DRY-RUN: rerun-units would stamp") and "naming superseded" in r.stdout and NEW_SHA[:12] in r.stdout, r.stdout + r.stderr
    assert json.loads(inv.read_text()) == before and len(_receipts(state)) == 1
    r = _cli(*other, "--supersede-receipt")
    assert r.returncode == 0 and f"rerun-units: stamped {RS_LW} -> {RC_LW}" in r.stdout and "ledger event appended (it names superseded" in r.stdout and NEW_SHA[:12] in r.stdout, r.stdout + r.stderr
    recs = _receipts(state); after = json.loads(inv.read_text())
    assert [x["code_sha"] for x in recs] == [NEW_SHA, OTHER_SHA] and "supersedes" not in recs[0]
    assert recs[1]["supersedes"] == [{"receipt_id": recs[0]["receipt_id"], "utc": recs[0]["utc"], "code_sha": NEW_SHA, "model": MODEL, "view": VIEW, "slug": SLUG,
                                   "source": SOURCE, "cells": [CELL]}] == [ST.rerun_receipt_ref(recs[0])] and recs[0]["receipt_id"] != recs[1]["receipt_id"]
    assert [u["code_sha"] for u in after["units"][4:]] == [OTHER_SHA, OTHER_SHA] and after["reruns"][0]["code_sha"] == OTHER_SHA     # the stamp IS the retry's tree …
    assert after["units"][:4] == before["units"] and after["n_units"] == 6
    assert ST.rerun_receipt_matches(recs[1], _decl(), OTHER_SHA) and not ST.rerun_receipt_matches(recs[0], _decl(), OTHER_SHA)      # … its receipt covers it; the orphan never did
    # idempotent at the stamped tree (the retired receipt conflicts with nothing); the orphan's own declaration is now the foreign one: stamped ONCE
    r = _cli(*other)
    assert r.returncode == 0 and "inventory unchanged (already stamped)" in r.stdout and json.loads(inv.read_text()) == after and len(_receipts(state)) == 2, r.stdout + r.stderr
    r = _cli(*_stamp_args(inv, state))
    assert r.returncode == 2 and "stamped ONCE" in r.stderr and json.loads(inv.read_text()) == after and len(_receipts(state)) == 2, r.stderr
    # a stamp that landed beside a FOREIGN receipt (the old order's outcome: the pair stamped at OTHER_SHA, the only receipt NEW_SHA's) gets its own
    # receipt REPAIRED, naming the foreign one superseded — the inventory is the stamp; there is no second outcome to choose, so no flag is asked
    state2 = tmp_path / "state2"; inv2 = _freeze(state2)
    ST.append_fsync(str(state2 / "attempts.jsonl"), json.dumps({"event": "attempt", "logical_work_id": STORE_LW, "attempt_id": "x", "job_id": "983259", "cluster": "cluster_a"}))
    other2 = [a if a != NEW_SHA else OTHER_SHA for a in _stamp_args(inv2, state2)]
    assert _cli(*other2).returncode == 0
    led = state2 / "attempts.jsonl"
    led.write_text("".join((json.dumps({**json.loads(l), "code_sha": NEW_SHA}, sort_keys=True) if json.loads(l).get("event") == "rerun_units" else l) + "\n"
                           for l in led.read_text().splitlines() if l.strip()))
    stamped = json.loads(inv2.read_text())
    assert [x["code_sha"] for x in _receipts(state2)] == [NEW_SHA] and [u["code_sha"] for u in stamped["units"][4:]] == [OTHER_SHA, OTHER_SHA]
    r = _cli(*other2, "--dry-run")
    assert r.returncode == 0 and "inventory unchanged (already stamped; DRY-RUN: the stamp's receipt is missing" in r.stdout and "another declaration / tree" in r.stdout, r.stdout + r.stderr
    assert len(_receipts(state2)) == 1
    r = _cli(*other2)
    assert r.returncode == 0 and "its receipt was MISSING" in r.stdout and "names superseded" in r.stdout and NEW_SHA[:12] in r.stdout, r.stdout + r.stderr
    recs2 = _receipts(state2)
    assert [x["code_sha"] for x in recs2] == [NEW_SHA, OTHER_SHA] and recs2[1]["supersedes"] == [ST.rerun_receipt_ref(recs2[0])] and recs2[1]["supersedes"][0]["receipt_id"] == recs2[0]["receipt_id"] != recs2[1]["receipt_id"]
    assert json.loads(inv2.read_text()) == stamped
    r = _cli(*other2)
    assert r.returncode == 0 and "inventory unchanged (already stamped)" in r.stdout and len(_receipts(state2)) == 2, r.stdout + r.stderr
    r = _cli(*_stamp_args(inv2, state2))
    assert r.returncode == 2 and "stamped ONCE" in r.stderr and len(_receipts(state2)) == 2, r.stderr


def test_same_second_receipts_of_two_pairs_or_two_rulings_are_told_apart_by_the_receipts_own_id_never_by_write_time_and_tree(monkeypatch):
    """A receipt was named superseded by (utc, code_sha) alone, and grant_ledger.utc_now has one-second
    precision — two receipts written on one tree in one second (two pairs' applies in a row; one pair under two rulings) share both, so
    superseding one retired the other: an unrelated pair's supersession hid THIS pair's orphan, and a receipt superseding a same-second
    orphan retired ITSELF. Now every receipt carries a unique durable `receipt_id`; a `supersedes` entry names a receipt by its FULL
    reference (id, write time, tree, pair, ruling, cells); a partial, drifted or old-shape reference names nothing (the receipt stays visible)."""
    monkeypatch.setattr(ST, "utc_now", lambda: "2026-10-12T00:00:00Z")                         # one second for every receipt below
    decl = _decl(); other = ST.bon_rerun_declaration([OTHER_CELL], SOURCE, MODEL, VIEW)
    corrected = ST.bon_rerun_declaration([OTHER_CELL], "a corrected ruling", MODEL, VIEW)
    a1, b1 = ST.rerun_event(decl, NEW_SHA, "rerun-units"), ST.rerun_event(other, NEW_SHA, "rerun-units")       # two pairs' orphans, one tree, one second
    a2 = ST.rerun_event(decl, OTHER_SHA, "rerun-units", supersedes=[a1])                                          # the first pair's orphan retired
    assert (a1["utc"], a1["code_sha"]) == (b1["utc"], b1["code_sha"])
    assert ST.rerun_receipt_conflicts([a1, b1], corrected, NEW_SHA) == [b1]
    assert ST.rerun_receipt_conflicts([a1, b1, a2], corrected, NEW_SHA) == [b1]              # before the fix: [] — a1's retirement hid b1
    assert ST.rerun_receipt_conflicts([a1, b1, a2], decl, OTHER_SHA) == [] and ST.rerun_receipt_conflicts([a1, b1, a2], decl, NEW_SHA) == [a2]
    assert ST.rerun_receipt_superseded(a1, [a1, b1, a2]) and not ST.rerun_receipt_superseded(b1, [a1, b1, a2]) and not ST.rerun_receipt_superseded(a2, [a1, b1, a2])
    # one pair, two rulings, one tree, one second: the superseding receipt never retires itself
    b2 = ST.rerun_event(corrected, NEW_SHA, "rerun-units", supersedes=[b1])
    third = ST.bon_rerun_declaration([OTHER_CELL], "a third ruling", MODEL, VIEW)
    assert ST.rerun_receipt_conflicts([b1, b2], third, NEW_SHA) == [b2]                      # before the fix: [] — b2's (utc, tree) equal the ref it wrote
    assert ST.rerun_receipt_superseded(b1, [b1, b2]) and not ST.rerun_receipt_superseded(b2, [b1, b2])
    # the id: unique and durable (on the receipt itself), 32 hex; the reference: the id and every field a refusal names
    assert len({r["receipt_id"] for r in (a1, b1, a2, b2)}) == 4 and all(re.fullmatch(r"[0-9a-f]{32}", r["receipt_id"]) for r in (a1, b1, a2, b2))
    assert ST.RERUN_RECEIPT_REF_KEYS == ("receipt_id", "utc", "code_sha", "model", "view", "slug", "source", "cells")
    assert ST.rerun_receipt_ref(b1) == {"receipt_id": b1["receipt_id"], "utc": "2026-10-12T00:00:00Z", "code_sha": NEW_SHA, "model": MODEL, "view": VIEW,
                                        "slug": OTHER_SLUG, "source": SOURCE, "cells": [OTHER_CELL]} == b2["supersedes"][0]
    assert b1["receipt_id"] in ST._rerun_receipt_words([b1]) and b1["utc"] in ST._rerun_receipt_words([b1])
    # a reference is the WHOLE reference: the right id with a drifted field, another receipt's id, or the old (utc, tree) shapes name nothing
    for s in ({**ST.rerun_receipt_ref(b1), "source": "another"}, {**ST.rerun_receipt_ref(b1), "receipt_id": a1["receipt_id"]}, {**ST.rerun_receipt_ref(b1), "receipt_id": None},
              {"utc": b1["utc"], "code_sha": NEW_SHA}, {"utc": b1["utc"], "code_sha": NEW_SHA, "source": SOURCE, "cells": [OTHER_CELL]}):
        bad = {**b2, "supersedes": [s]}
        assert ST.rerun_receipt_conflicts([b1, bad], third, NEW_SHA) == [b1, bad] and not ST.rerun_receipt_superseded(b1, [b1, bad]), s


def test_a_same_second_supersession_of_another_pairs_orphan_hides_no_orphan_of_this_pair_end_to_end(tmp_path, monkeypatch):
    """Through the CLI: two pairs' applies on one tree die between the receipt and the inventory in
    the same second; the first pair's orphan is retired under `--supersede-receipt`; the OTHER pair's retry under a corrected ruling is still
    refused by name (before the fix the unrelated supersession had retired its orphan too: the retry exited 0, stamped the corrected
    declaration and left the conflicting receipt unretired), and its own `--supersede-receipt` names its orphan by the orphan's id."""
    monkeypatch.setenv("SISTEER_PLAN_SKIP_ARTEFACT_CHECK", "1")
    state = tmp_path / "state"; inv = _freeze(state)
    before = json.loads(inv.read_text())
    ST.append_fsync(str(state / "attempts.jsonl"), json.dumps({"event": "attempt", "logical_work_id": STORE_LW, "attempt_id": "x", "job_id": "983259", "cluster": "cluster_a"}))
    other_pair = [a if a != CELL else OTHER_CELL for a in _stamp_args(inv, state)]
    for args in (_stamp_args(inv, state), other_pair):
        r = _cli(*args, env={"SISTEER_STAGE_CRASH_AT": "rerun_after_event"})
        assert r.returncode == 97 and json.loads(inv.read_text()) == before, r.stderr
    # … in the same second: the ledger's two receipts are pinned to one write time (utc_now is second-resolution, so two applies in a row
    # land there on their own; pinning it makes the collision the test's premise rather than its luck)
    led = state / "attempts.jsonl"; T = "2026-10-12T00:00:00Z"
    led.write_text("".join((json.dumps({**json.loads(l), "utc": T}, sort_keys=True) if json.loads(l).get("event") == "rerun_units" else l) + "\n"
                           for l in led.read_text().splitlines() if l.strip()))
    a1, b1 = _receipts(state)
    assert [(x["slug"], x["utc"], x["code_sha"]) for x in (a1, b1)] == [(SLUG, T, NEW_SHA), (OTHER_SLUG, T, NEW_SHA)] and a1["receipt_id"] != b1["receipt_id"]
    # the first pair's orphan is retired under another tree
    other_tree = [a if a != NEW_SHA else OTHER_SHA for a in _stamp_args(inv, state)]
    r = _cli(*other_tree, "--supersede-receipt")
    assert r.returncode == 0 and "it names superseded" in r.stdout and a1["receipt_id"] in r.stdout, r.stdout + r.stderr
    assert _receipts(state)[2]["supersedes"] == [ST.rerun_receipt_ref(a1)] and [u["logical_work_id"] for u in json.loads(inv.read_text())["units"][4:]] == [RS_LW, RC_LW]
    # the OTHER pair's orphan is an orphan still: its retry under a corrected ruling is refused by name, the dry run too, nothing written
    corrected = [a if a != SOURCE else "a corrected ruling" for a in other_pair]
    for extra in ((), ("--dry-run",)):
        r = _cli(*corrected, *extra)
        assert r.returncode == 2, extra + (r.stdout, r.stderr)
        assert ("another declaration / tree" in r.stderr and "an interrupted apply" in r.stderr and b1["receipt_id"] in r.stderr and SOURCE in r.stderr
                and "a corrected ruling" in r.stderr and not r.stdout.strip()), r.stdout + r.stderr
    assert len(_receipts(state)) == 3 and [u["logical_work_id"] for u in json.loads(inv.read_text())["units"][4:]] == [RS_LW, RC_LW]
    # its own declaration completes the apply without a flag … (not taken: the ruling was corrected) … or the flag retires the orphan BY ITS ID
    r = _cli(*corrected, "--supersede-receipt")
    assert r.returncode == 0 and f"rerun-units: stamped {RS2_LW} -> {RC2_LW}" in r.stdout and b1["receipt_id"] in r.stdout, r.stdout + r.stderr
    recs = _receipts(state); after = json.loads(inv.read_text())
    assert len(recs) == 4 and len({x["receipt_id"] for x in recs}) == 4
    assert recs[3]["supersedes"] == [ST.rerun_receipt_ref(b1)] and recs[3]["supersedes"][0]["receipt_id"] == b1["receipt_id"] and recs[3]["source"] == "a corrected ruling"
    assert [u["logical_work_id"] for u in after["units"][4:]] == [RS_LW, RC_LW, RS2_LW, RC2_LW] and after["n_units"] == 8
    assert [u["code_sha"] for u in after["units"][4:]] == [OTHER_SHA, OTHER_SHA, NEW_SHA, NEW_SHA] and [d["slug"] for d in after["reruns"]] == [SLUG, OTHER_SLUG]
    # every receipt is a stamp's or named superseded by its full reference: nothing conflicts, both stamps idempotent, no fifth receipt
    for decl, sha in ((_decl(), OTHER_SHA), (ST.bon_rerun_declaration([OTHER_CELL], "a corrected ruling", MODEL, VIEW), NEW_SHA)):
        assert ST.rerun_receipt_conflicts(recs, decl, sha) == [] and [r["receipt_id"] for r in recs if ST.rerun_receipt_matches(r, decl, sha)] == [recs[2 if sha == OTHER_SHA else 3]["receipt_id"]]
    assert [ST.rerun_receipt_superseded(x, recs) for x in recs] == [True, True, False, False]
    for args in (other_tree, corrected):
        r = _cli(*args)
        assert r.returncode == 0 and "inventory unchanged (already stamped)" in r.stdout, r.stdout + r.stderr
    assert len(_receipts(state)) == 4 and json.loads(inv.read_text()) == after
    # `plan --inventory` plans both pairs for the view
    r = _cli("plan", "--model", MODEL, "--view", VIEW, "--families", "static,bon", "--sha", NEW_SHA, "--inventory", str(inv), "--json")
    assert r.returncode == 0 and {x["logical_work_id"] for x in json.loads(r.stdout) if x["registers"]} == {u["logical_work_id"] for u in after["units"]}, r.stderr


def test_a_half_stamped_or_missing_pair_in_the_inventory_refuses_and_the_fresh_wave_plans_no_rerun(tmp_path, monkeypatch):
    monkeypatch.setenv("SISTEER_PLAN_SKIP_ARTEFACT_CHECK", "1")
    state = tmp_path / "state"; inv = _freeze(state)
    ST.append_fsync(str(state / "attempts.jsonl"), json.dumps({"event": "attempt", "logical_work_id": STORE_LW, "attempt_id": "x", "job_id": "1", "cluster": "cluster_a"}))
    # an inventory frozen without the bon family (an excluding view) carries no store512 to re-run
    doc = json.loads(inv.read_text()); doc["units"] = [u for u in doc["units"] if u["family"] != "bon"]; ST.write_atomic(str(inv), doc)
    r = _cli(*_stamp_args(inv, state))
    assert r.returncode == 2 and "are not units of the inventory" in r.stderr and STORE_LW in r.stderr, r.stderr
    # a hand-edited half pair is refused
    inv2 = _freeze(tmp_path / "state2"); state2 = tmp_path / "state2"
    ST.append_fsync(str(state2 / "attempts.jsonl"), json.dumps({"event": "attempt", "logical_work_id": STORE_LW, "attempt_id": "x", "job_id": "1", "cluster": "cluster_a"}))
    assert _cli(*_stamp_args(inv2, state2)).returncode == 0
    doc = json.loads(inv2.read_text()); doc["units"] = [u for u in doc["units"] if u["logical_work_id"] != RC_LW]; ST.write_atomic(str(inv2), doc)
    r = _cli(*_stamp_args(inv2, state2))
    assert r.returncode == 2 and "a half-stamped pair" in r.stderr, r.stderr
    # the freeze itself never plans a rerun: expected_units without a declaration is the paper's set, with one it is the stamp's superset
    base = ST.expected_units(MODEL, SHA, TABLE, families=["static", "bon"], goals_jsonl=str(goal_fixture(tmp_path)), wave_config=str(wave_config(tmp_path)))
    assert not any(u.get("rerun_of") for u in base) and len(base) == 4
    more = ST.expected_units(MODEL, SHA, TABLE, families=["static", "bon"], goals_jsonl=str(goal_fixture(tmp_path)), wave_config=str(wave_config(tmp_path)), reruns=[_decl()])
    assert [u["logical_work_id"] for u in more] == [u["logical_work_id"] for u in base[:3]] + [RS_LW] + [base[3]["logical_work_id"], RC_LW]
    assert [u for u in more if not u.get("rerun_of")] == base


# ---------------------------------------------------------------------------------------------- (7) end to end: 05 freezes, the stamp lands, 06 submits by name
def test_06_submits_the_pair_by_its_units_name_against_the_stamped_inventory_end_to_end(tmp_path):
    import test_front_door_wave_integration as WI
    w = WI.World(tmp_path)
    # the wave roster view_cells_of reads (05 --step roster renders it in the real flow): the baseline wave's config, committed so the trees stay clean
    (w.repo / "configs" / "sdfbase_full22_roster.yaml").write_text(yaml.safe_dump(
        {"targets": {WI.MODEL: {"views": {VIEW: {"config": f"configs/sdfbase_full22_{WI.MODEL}_{VIEW}.yaml"}}}}}))
    g = ["git", "-c", "user.name=t", "-c", "user.email=t@TESTHOST"]
    subprocess.run([*g, "add", "-A"], cwd=w.repo, check=True); subprocess.run([*g, "commit", "-q", "-m", "the wave roster"], cwd=w.repo, check=True)
    w.sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=w.repo, capture_output=True, text=True, check=True).stdout.strip()
    r = w.stage("05_stage_roster.sh", "--model", WI.MODEL, "--step", "freeze")
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    inv = w.state / "gate" / "expected_units.json"
    assert STORE_LW in {u["logical_work_id"] for u in w.inventory()["units"]}
    # the wave has run its store (the attempt the real funnel records on submission); then the stamp, with the freeze's inputs
    ST.append_fsync(str(w.state / "attempts.jsonl"), json.dumps({"event": "attempt", "logical_work_id": STORE_LW, "attempt_id": "x", "job_id": "983259", "cluster": "cluster_a"}))
    r = _cli("rerun-units", "--out", str(inv), "--cells", CELL, "--source", SOURCE, "--code-sha", w.sha, "--models-table", str(w.repo / "reproduce" / "models.yaml"),
             "--roster", str(w.repo / "configs" / "sdfbase_full22_roster.yaml"), "--clusters", ",".join(AB), "--goals-jsonl", str(w.goals),
             stage_py=w.repo / "reproduce" / "lib" / "stage.py", cwd=w.repo, env=w.env())
    assert r.returncode == 0 and f"stamped {RS_LW} -> {RC_LW}" in r.stdout, r.stdout + r.stderr
    assert {RS_LW, RC_LW} <= {u["logical_work_id"] for u in w.inventory()["units"]}
    # 06, dry: the store under --family static by its `--units` name — the inventory check passes with the pair stamped, the dry submit line carries
    # the rerun's env prefix, the ruling's cell on the cells positional and the `-3` name
    r = w.stage("06_eval_waves.sh", "--model", WI.MODEL, "--view", VIEW, "--family", "static", "--units", f"bon.store512_{SLUG}", "--dry-run")
    out = r.stdout + r.stderr
    assert r.returncode == 0 and "plan == frozen inventory" in out, out[-3000:]
    lines = [l for l in out.splitlines() if l.startswith("DRY-RUN: ssh") and "target_static.sbatch" in l]
    assert len(lines) == 1 and f"F22_BON_RERUN={SLUG}" in lines[0] and f"F22_REGISTRY_CELLS={CELL}" in lines[0] and f"f22-stat-{TAG}-{SLUG}-3" in lines[0], out[-3000:]
    assert f" {WI.MODEL} {VIEW} bon {w.sha} 9 {CELL}" in lines[0], lines[0]
    assert "static.t0" not in " ".join(l for l in out.splitlines() if l.startswith("DRY-RUN: ssh"))           # only the named unit goes out
    # … and the climb under --family bon, afterok on the rerun STORE (unbound in a dry run: resolved at submit time)
    r = w.stage("06_eval_waves.sh", "--model", WI.MODEL, "--view", VIEW, "--family", "bon", "--units", f"bon.climb_{SLUG}", "--dry-run")
    out = r.stdout + r.stderr
    assert r.returncode == 0 and "plan == frozen inventory" in out, out[-3000:]
    lines = [l for l in out.splitlines() if l.startswith("DRY-RUN: ssh") and "bon_climb.sbatch" in l]
    assert len(lines) == 1 and f"F22_BON_RERUN={SLUG}" in lines[0] and f"f22-bonclimb-{TAG}-{SLUG}" in lines[0] and f" {WI.MODEL} {VIEW} {w.sha} 4 {CELL}" in lines[0], out[-3000:]
    assert f"afterok:<{RS_LW}'s bound job>" in out, out[-3000:]
