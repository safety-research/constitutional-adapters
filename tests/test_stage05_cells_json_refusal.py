"""05_stage_roster.sh's `cells_json` refusals are the STAGE's refusals (demo-import blocker 1b, M2): a derived object without its pin, or a table
row without its kind directory, exits 2 with the refusal text and NO traceback. Before M2 the heredoc's SystemExit left `CELLS` empty, the next
line tracebacked on `json.loads('')`, and the stage carried on (exit 0 under --dry-run, a second traceback from stage_objects.py). Fixtures: a
SYNTHETIC copy of the models table whose demo row has no committed source (so the legacy pins path is the one taken, whatever the committed
row declares) and an EMPTY pins file; the site's one cluster is the fake-Slurm seam (`CLUSTER_SSH=local:<tmp>`), so even a real run could touch
no host — and the refusal fires before any staging, so nothing is written under the state dir or the shared root."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO, REPRO, run_stage, write_site  # noqa: E402

KEY = "qwen35_4b_demo"


@pytest.fixture()
def world(tmp_path):
    """(site, env, state) — a one-cluster local site, a table without the committed source, an empty pins file."""
    shared = tmp_path / "shared"; shared.mkdir()
    site = write_site(tmp_path, clusters=("cluster_a",), ssh={"cluster_a": f"local:{shared}"}, shared={"cluster_a": str(shared)})
    table = yaml.safe_load((REPRO / "models.yaml").read_text())
    row = table["models"][KEY]
    row.pop("committed_objects_of", None)
    for o in row["objects"].values():
        if isinstance(o, dict):
            o.pop("committed_objects_of", None)
    tpath = tmp_path / "models.yaml"; tpath.write_text(yaml.safe_dump(table, sort_keys=False))
    pins = tmp_path / "pins.json"; pins.write_text(json.dumps({}))
    env = {"SISTEER_MODELS_TABLE": str(tpath), "SISTEER_PINS_FILE": str(pins)}
    return site, env, tmp_path / "state", shared, table, tpath


def _files(root: Path) -> set:
    return {str(p) for p in root.rglob("*")} if root.exists() else set()


@pytest.mark.parametrize("dry", [False, True])
def test_a_derived_object_without_a_pin_refuses_the_stage_by_name_with_no_traceback(world, dry):
    site, env, state, shared, _table, _tpath = world
    args = ["--model", KEY, "--step", "stage"] + (["--dry-run"] if dry else [])
    r = run_stage("05_stage_roster.sh", *args, site=site, env=env, timeout=600)
    out = r.stdout + r.stderr
    assert r.returncode == 2, out[-2500:]
    assert "has no pin" in r.stderr and "staged by its REGISTRY id" in r.stderr, r.stderr[-2500:]          # the heredoc's own refusal, printed
    assert "REFUSED: 05: cells_json refused (above)" in r.stderr, r.stderr[-2500:]                           # the stage's refusal, by name
    assert "Traceback" not in out and "json.loads" not in out and "JSONDecodeError" not in out, out[-2500:]
    assert "stage_objects.py" not in out and "DRY-RUN:" not in out, out[-2500:]                              # nothing after the refusal
    assert not (state / KEY / "05_cells.json").exists() and _files(shared) == set()                          # nothing written anywhere


def test_a_derived_object_without_its_kind_directory_refuses_the_same_way(world):
    site, env, state, shared, table, tpath = world
    table["models"][KEY]["objects"]["difference_relu"].pop("derived_dir")
    tpath.write_text(yaml.safe_dump(table, sort_keys=False))
    # the ReLU dose view names difference_relu alone (the wave's refusal direction would refuse first, on its pin)
    r = run_stage("05_stage_roster.sh", "--model", KEY, "--step", "stage", "--views", "d0p75_relu", site=site, env=env, timeout=600)
    out = r.stdout + r.stderr
    assert r.returncode == 2 and "has source derived_bank but no derived_dir" in r.stderr and "refusing to guess" in r.stderr, out[-2500:]
    assert "REFUSED: 05: cells_json refused (above)" in r.stderr and "Traceback" not in out, out[-2500:]
    assert _files(shared) == set() and not (state / KEY / "05_cells.json").exists()


def test_the_shell_wires_the_refusal_and_never_parses_an_empty_list():
    """The source-level pin: `CELLS="$(cells_json)"` is guarded by `|| repro_fatal`, and so is the count that reads it — a refusal inside the
    heredoc can no longer reach `json.loads('')`."""
    src = (REPO / "reproduce" / "05_stage_roster.sh").read_text()
    assert 'CELLS="$(cells_json)" || repro_fatal "05: cells_json refused (above)"' in src
    assert '"$CELLS")" || repro_fatal "05: cells_json printed no JSON list (above)"' in src
    assert 'CELLS="$(cells_json)"; n=' not in src
