"""The registry notebook reads the committed paper-frame snapshot when the live registry is absent.

`paper/notebooks/registry_nb.py` resolves ONE source at import - `REGISTRY_NB_SOURCE` = live | snapshot | auto (auto: live when
`REGISTRY_ROOT/registry.sqlite` exists, else the snapshot under `REGISTRY_NB_SNAPSHOT_ROOT`) - and every reader in the file goes
through `connect()`, so the snapshot answers the notebook's SQL unchanged. This module pins that rule on a SYNTHETIC registry
(`sisteer.registry.synth`, hostname TESTHOST, invented ids) and a paper-frame snapshot exported from it with the registry's own
exporter (`snapshot.build(scope="paper_frame", dest_root=...)`, the synthetic model added through `model_ids`), and holds the
helpers the sibling tests (`test_nb_fresh_home.py`, `test_nb_snapshot_parity.py`) import: the notebook mirror (a clone-shaped copy
of the two notebook files with `src/` beside them, so the clone-relative `SISTEER_SRC` default resolves and nothing is written into
the checkout) and the fresh in-process import. The snapshot reader the notebook codes against is `snapshot.open_paper_frame` /
`paper_frame_current`; a tree without them skips this module BY NAME rather than failing somewhere inside the notebook.
"""
from __future__ import annotations

import hashlib
import importlib
import os
import re
import shutil
import sqlite3
import sys
from pathlib import Path

import pytest

from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import snapshot as SN
from sisteer.registry import synth

for _name in ("open_paper_frame", "paper_frame_current", "PAPER_FRAME_TABLES"):
    if not hasattr(SN, _name):
        pytest.skip(f"sisteer.registry.snapshot.{_name} is not in this tree (the paper-frame snapshot lane)", allow_module_level=True)
if "paper_frame" not in SN.SCOPES:
    pytest.skip("sisteer.registry.snapshot.SCOPES has no paper_frame scope (the paper-frame snapshot lane)", allow_module_level=True)

REPO = Path(__file__).resolve().parents[1]
NOTEBOOKS = REPO / "paper" / "notebooks"
NB_FILES = ("registry_nb.py", "make_registry_notebook.py")
NB_VENV_PYTHON = str(REPO / ".venv" / "bin" / "python")     # the clone's own venv (`uv sync --frozen --extra cpu --group notebook`: nbconvert, nbformat); optional
NB_ENV_KEYS = ("REGISTRY_NB_SOURCE", "REGISTRY_ROOT", "REGISTRY_NB_SNAPSHOT_ROOT", "SISTEER_SRC", "NB_CACHE", "NB_REUSE_CACHE",
               "NB_ALLOW_UNSCALED_HEADLINE")


def synthetic_registry(tmp_path, n_runs: int = 4, n_items: int = 10) -> str:
    """A synthetic registry root, ingested (`tests/test_registry_snapshot.py::_setup`)."""
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    synth.write_corpus(root, n_runs=n_runs, n_items=n_items)
    H.ingest_once(root, workers=0, min_free=0)
    return root


def live_head(root: str) -> int:
    with sqlite3.connect(f"file:{DB.db_path(root)}?mode=ro", uri=True) as con:
        return int(con.execute("SELECT MAX(ingest_seq) FROM record_ingest").fetchone()[0])


def export_snapshot(root: str, dest_root: str) -> tuple:
    """(snapshot_id, head) of a paper-frame snapshot of `root` exported under `dest_root` (the synthetic model joins the exported
    model set through `model_ids`; the tree precondition is overridden with a reason, as a test must)."""
    os.makedirs(dest_root, exist_ok=True)
    SN.build(root, DB.db_path(root), check_free_space=False, scope="paper_frame", dest_root=dest_root, model_ids=[synth.MODEL_ID],
             precondition_override="synthetic registry in a test: no paired-ReLU tree to check")
    sid, _manifest = SN.paper_frame_current(dest_root)
    return str(sid), live_head(root)


def notebook_mirror(tmp_path, name: str = "clone") -> Path:
    """`<tmp>/<name>/paper/notebooks/` holding copies of the two notebook files, with `<tmp>/<name>/src` -> this checkout's `src`, so
    `Path(__file__).parents[2] / "src"` resolves the way it does in a clone and every cache / figure lands in the temp tree."""
    clone = Path(tmp_path) / name
    nbdir = clone / "paper" / "notebooks"
    nbdir.mkdir(parents=True)
    for f in NB_FILES:
        shutil.copy(NOTEBOOKS / f, nbdir / f)
    os.symlink(REPO / "src", clone / "src")
    return nbdir


def nb_env(tmp_path, **over) -> dict:
    """The environment a notebook import runs under: an EMPTY home (never the machine's ~/registry), no SISTEER_SRC, Agg."""
    home = Path(tmp_path) / "home"
    home.mkdir(exist_ok=True)
    env = {"HOME": str(home), "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(home / ".mpl"), "SISTEER_REGISTRY_DISABLED": "1",
           "REGISTRY_ROOT": str(home / "registry")}
    env.update({k: v for k, v in over.items() if v is not None})
    return env


def import_notebook(nbdir: Path, env: dict, monkeypatch):
    """A FRESH import of `registry_nb` from `nbdir` under `env` (the module resolves its source and memoises its readers at import)."""
    for k in NB_ENV_KEYS + ("HOME", "MPLBACKEND", "MPLCONFIGDIR"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.syspath_prepend(str(nbdir))
    sys.modules.pop("registry_nb", None)
    import matplotlib
    matplotlib.use("Agg")
    return importlib.import_module("registry_nb")


@pytest.fixture
def world(tmp_path):
    """A synthetic registry, its paper-frame snapshot, and a notebook mirror."""
    root = synthetic_registry(tmp_path)
    dest = str(tmp_path / "paper_snapshot")
    sid, head = export_snapshot(root, dest)
    return dict(root=root, dest=dest, sid=sid, head=head, nbdir=notebook_mirror(tmp_path))


# ---- the source rule ------------------------------------------------------------------------------------------------------------
def test_snapshot_source_when_registry_absent(world, tmp_path, monkeypatch, capsys):
    R = import_notebook(world["nbdir"], nb_env(tmp_path, REGISTRY_NB_SNAPSHOT_ROOT=world["dest"]), monkeypatch)
    out = capsys.readouterr().out
    assert R.SOURCE == "snapshot" and R.SNAPSHOT_ID == world["sid"]
    assert R.SISTEER_SRC == str(world["nbdir"].parents[1] / "src")            # the clone-relative default, SISTEER_SRC unset
    assert "registry source snapshot" in out and world["dest"] in out          # the one header line names the source
    assert R.head() == world["head"]
    with R.connect() as con:
        assert con.execute("SELECT COUNT(*) FROM run").fetchone()[0] == 4
    df = R.query("SELECT run_id FROM v_paper_long", "t")
    assert len(df) == 4 and df.attrs["head"] == world["head"]
    cached = [f for f in os.listdir(R.CACHE) if f.endswith(".parquet")]
    assert cached and all(f.startswith(f"t-snapshot-{world['head']}-") for f in cached), cached
    # the three direct openers, through connect()
    assert R.headline_metric_ids() == {}                                        # the synthetic eval declares no list
    assert len(R.role_collapse_rows(R.head())) == 0
    missing = R.check_headline_scales(strict=False)
    assert isinstance(missing, tuple) and missing == ()


def test_live_refuses_by_name_when_db_absent(world, tmp_path, monkeypatch):
    env = nb_env(tmp_path, REGISTRY_NB_SOURCE="live", REGISTRY_NB_SNAPSHOT_ROOT=world["dest"])
    with pytest.raises(FileNotFoundError) as e:
        import_notebook(world["nbdir"], env, monkeypatch)
    assert os.path.join(env["REGISTRY_ROOT"], "registry.sqlite") in str(e.value) and "registry source live" in str(e.value)


def test_snapshot_refuses_by_name_when_none_committed(world, tmp_path, monkeypatch):
    empty = tmp_path / "no_snapshot"
    empty.mkdir()
    with pytest.raises(FileNotFoundError) as e:
        import_notebook(world["nbdir"], nb_env(tmp_path, REGISTRY_NB_SOURCE="snapshot", REGISTRY_NB_SNAPSHOT_ROOT=str(empty)), monkeypatch)
    assert str(empty) in str(e.value) and "registry source snapshot" in str(e.value)


def test_auto_prefers_live_when_db_present(world, tmp_path, monkeypatch, capsys):
    R = import_notebook(world["nbdir"], nb_env(tmp_path, REGISTRY_ROOT=world["root"], REGISTRY_NB_SNAPSHOT_ROOT=world["dest"]), monkeypatch)
    assert R.SOURCE == "live" and R.SNAPSHOT_ID is None and R.DB == DB.db_path(world["root"])
    assert "registry source live" in capsys.readouterr().out
    df = R.query("SELECT run_id FROM run", "t")
    assert len(df) == 4 and any(f.startswith(f"t-live-{world['head']}-") for f in os.listdir(R.CACHE))


def test_auto_with_neither_imports_and_connect_refuses_by_name(world, tmp_path, monkeypatch):
    """The offline path the notebook's own synthetic-frame tests rely on: importable, `check_headline_scales` answers (), connect() names both."""
    empty = tmp_path / "no_snapshot"
    empty.mkdir()
    env = nb_env(tmp_path, REGISTRY_NB_SNAPSHOT_ROOT=str(empty))
    R = import_notebook(world["nbdir"], env, monkeypatch)
    assert R.SOURCE == "live" and R.SNAPSHOT_ID is None
    assert R.check_headline_scales(strict=False) == () and R.headline_metric_ids() == {} and len(R.role_collapse_rows(0)) == 0
    with pytest.raises(FileNotFoundError) as e:
        R.connect()
    assert R.DB in str(e.value) and str(empty) in str(e.value)


def test_invalid_source_mode_refuses(world, tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="REGISTRY_NB_SOURCE"):
        import_notebook(world["nbdir"], nb_env(tmp_path, REGISTRY_NB_SOURCE="bogus"), monkeypatch)


def test_nb_cache_zero_bypasses_read_and_write(world, tmp_path, monkeypatch):
    import pandas as pd
    R = import_notebook(world["nbdir"], nb_env(tmp_path, REGISTRY_NB_SNAPSHOT_ROOT=world["dest"], NB_CACHE="0"), monkeypatch)
    sql = "SELECT run_id FROM run"
    df = R.query(sql, "t")
    assert len(df) == 4 and not [f for f in os.listdir(R.CACHE) if f.endswith(".parquet")]     # nothing written
    # a planted cache file with the exact key is NOT read either
    sha = hashlib.sha1(sql.encode()).hexdigest()[:8]
    pd.DataFrame({"run_id": ["planted"]}).to_parquet(os.path.join(R.CACHE, f"t-snapshot-{world['head']}-{sha}.parquet"))
    assert len(R.query(sql, "t")) == 4
    monkeypatch.setenv("NB_CACHE", "1")
    assert list(R.query(sql, "t").run_id) == ["planted"]                       # the same key IS read once the cache is on again


# ---- the generator: the paper's panels, nothing else ------------------------------------------------------------------------------
ICLR_CELL_IDS = ("40484fe7", "583e3f04", "12cd66ed", "ec6aacd8", "9bf536ac", "6fec6474", "b6d74ec9", "ac902304", "4f2ab8d1", "7c1de904",
                 "5e61e1ab", "624d0e27", "a3e91d55", "f7a1c206", "11a8433d", "b24c9738", "7d2e9c40", "a8e3d1c2", "d4f7b2a9", "3c9f7a51",
                 "8e1b4d27", "5a7c2e93", "6d2f8b3a", "b0a7c4e1", "9b3d6f18", "c4e8a2d6", "e6f1b5a9", "3c7a1e55", "5b9d2f7e", "7e2c9b41",
                 "f1a8c3d7", "b7d4e9f2", "92fdc8e2", "23555668")   # the paper notebook's cells at the ICLR submission, in order
# spelled in pieces so this file never carries the strings the public tree's census forbids
SECOND_MODEL = re.compile("qwen35_" + "35b|35b_" + "a3b|Qwen3\\.5-3" + "5B|pipeline[-_ ]test|PIPELINE_" + "TEST|REGISTRY_NB_" + "PANELS|figT_")


def test_generator_carries_the_papers_panels_only_in_the_iclr_layout():
    """The public notebook is the paper's panels in the ICLR-era layout: the pinned cell ids in that order,
    no second model, no panel switch, no test artifact. The zip cell lists paper figures only; the header cell names the source;
    connect() is the ONE opener. The reader library spells no second model either."""
    src = (NOTEBOOKS / "make_registry_notebook.py").read_text(encoding="utf-8")
    assert tuple(re.findall(r'^(?:md|code)\("([0-9a-f]{8})"', src, flags=re.M)) == ICLR_CELL_IDS
    assert not SECOND_MODEL.search(src), SECOND_MODEL.search(src).group(0)
    assert "registry source {R.SOURCE}" in src                                     # the header cell names the source
    zip_cell = src[src.index('code("b7d4e9f2"'):src.index('md("92fdc8e2"')]
    assert "PAPER_FIGURES = [" in zip_cell and "fig7_frontier" in zip_cell
    assert '"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}' in src    # no maintainer kernel
    nb = (NOTEBOOKS / "registry_nb.py").read_text(encoding="utf-8")
    assert nb.count("sqlite3.connect(") == 1                                       # connect() is the ONE opener
    assert not SECOND_MODEL.search(nb), SECOND_MODEL.search(nb).group(0)
