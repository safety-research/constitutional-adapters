"""The fresh-clone proof: the registry notebook runs from a clone-shaped tree
with a HOME that holds no `~/registry` and no clone of this repository, `SISTEER_SRC` unset, `PYTHONPATH` empty, caches absent - so
`sisteer.registry` must come from the clone's own `src/` (the clone-relative `SISTEER_SRC` default) and the rows from the committed
paper-frame snapshot. The interpreter is the test venv's, which has pandas / pyarrow / matplotlib and does NOT have `sisteer`
installed, so a wrong default would fail the import rather than fall through to an installed package.

Two proofs. The header and load cells are extracted from the built notebook (`make_registry_notebook.py`, which needs `nbformat`:
the test venv's interpreter first, else the notebook's own venv, else skip) and executed in a subprocess: the header must name
`source snapshot` and the clone's `src`, and the load cell must complete (its frames are empty on a synthetic registry - the paper
selection flags none of its runs - which is the case a fresh clone with a partial snapshot must survive too). Then, where `jupyter
nbconvert` is available, the WHOLE notebook is executed against the synthetic snapshot.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from test_nb_snapshot_fallback import NB_VENV_PYTHON, export_snapshot, notebook_mirror, synthetic_registry

KEEP_ENV = ("PATH", "LANG", "LC_ALL", "TERM", "TMPDIR", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE")
PRELUDE = "import matplotlib; matplotlib.use('Agg')\n"


def _interpreter_with(module: str):
    for py in (sys.executable, NB_VENV_PYTHON):
        if os.path.exists(py) and subprocess.run([py, "-c", f"import {module}"], capture_output=True).returncode == 0:
            return py
    return None


def _scrubbed_env(tmp_path, dest: str, **over) -> dict:
    """An EMPTY home (no registry, no checkout), no SISTEER_SRC, no PYTHONPATH, no REGISTRY_ROOT (the `~/registry` default applies)."""
    home = Path(tmp_path) / "home"
    home.mkdir(exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k in KEEP_ENV}
    env.update({"HOME": str(home), "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(home / ".mpl"), "SISTEER_REGISTRY_DISABLED": "1",
                "REGISTRY_NB_SNAPSHOT_ROOT": dest, "JUPYTER_DATA_DIR": str(home / ".jupyter"), "IPYTHONDIR": str(home / ".ipython")})
    env.update(over)
    return env


@pytest.fixture
def clone(tmp_path):
    py_nb = _interpreter_with("nbformat")
    if py_nb is None:
        pytest.skip("nbformat is in neither the test venv nor the notebook venv; make_registry_notebook.py cannot build the ipynb")
    root = synthetic_registry(tmp_path)
    dest = str(tmp_path / "paper_snapshot")
    sid, head = export_snapshot(root, dest)
    nbdir = notebook_mirror(tmp_path)
    build = subprocess.run([py_nb, "make_registry_notebook.py"], cwd=nbdir, env=_scrubbed_env(tmp_path, dest), capture_output=True, text=True)
    assert build.returncode == 0, build.stderr[-2000:]
    return dict(nbdir=nbdir, dest=dest, sid=sid, head=head, ipynb=nbdir / "registry_qwen_figures.ipynb")


def test_header_and_load_cells_run_from_the_snapshot_in_a_fresh_home(clone, tmp_path):
    nb = json.loads(clone["ipynb"].read_text(encoding="utf-8"))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert [c["id"] for c in code[:2]] == ["583e3f04", "ec6aacd8"]              # the header cell, the load cell
    script = PRELUDE + "".join(code[0]["source"]) + "\n" + "".join(code[1]["source"]) + "\nprint('LOAD CELL COMPLETE', len(paper_all), len(runs_all), HEAD)\n"
    env = _scrubbed_env(tmp_path, clone["dest"])
    assert "SISTEER_SRC" not in env and "PYTHONPATH" not in env and "REGISTRY_ROOT" not in env
    assert not (Path(env["HOME"]) / "registry").exists() and not (Path(env["HOME"]) / "constitutional-adapters").exists()
    proc = subprocess.run([sys.executable, "-c", script], cwd=clone["nbdir"], env=env, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, f"--- stdout\n{proc.stdout[-4000:]}\n--- stderr\n{proc.stderr[-4000:]}"
    out = proc.stdout
    assert f"registry source snapshot: {clone['dest']} @ {clone['sid']}" in out       # the module's header line (the dest lies outside the clone: absolute)
    # the clone's own src, spelled relative to the clone root (a committed render carries no home directory)
    assert f"registry source snapshot · {clone['dest']} @ {clone['sid']} · head {clone['head']} · sisteer.registry from src @" in out
    assert f"LOAD CELL COMPLETE 0 0 {clone['head']}" in out                           # the frames are empty on a synthetic registry; the cell completes
    cache = clone["nbdir"] / "_cache"
    names = sorted(f for f in os.listdir(cache) if f.endswith(".parquet"))
    assert names and all(f"-snapshot-{clone['head']}-" in f for f in names), names


NBCONVERT_CELLS = ("40484fe7", "583e3f04", "ec6aacd8", "b7d4e9f2", "92fdc8e2", "23555668")   # intro, header, load, the figure zip, record


def test_header_load_zip_and_record_cells_execute_under_nbconvert_from_the_snapshot(clone, tmp_path):
    """The kernel path: `jupyter nbconvert --execute` in an interpreter with nbconvert (no `sisteer` installed there either), fresh home,
    snapshot source, and NO kernel override - the notebook's own kernelspec is `python3`, the one ipykernel registers inside whatever
    interpreter runs it (a maintainer kernel name refused on a fresh machine with NoSuchKernel) - over the cells a synthetic snapshot
    can carry: the header, the load, the figure zip (no figure drawn, so it names every missing one) and the record cell. The paper
    cells past Figure 6 need rows the synthetic paper frame does not have (the frontier cell's `rung_eval_table` drops a column an
    empty frame never grew), so the whole-notebook run is a live-snapshot matter, not a synthetic one."""
    py_nb = _interpreter_with("nbconvert")
    if py_nb is None:
        pytest.skip("nbconvert is in neither the test venv nor the clone's venv")
    if subprocess.run([py_nb, "-c", "import ipykernel"], capture_output=True).returncode != 0:
        pytest.skip("the interpreter with nbconvert has no ipykernel to execute the notebook in")
    nb = json.loads(clone["ipynb"].read_text(encoding="utf-8"))
    assert nb["metadata"]["kernelspec"] == {"display_name": "Python 3", "language": "python", "name": "python3"}
    keep = [c for c in nb["cells"] if c.get("id") in NBCONVERT_CELLS]
    assert [c["id"] for c in keep] == list(NBCONVERT_CELLS)
    reduced = clone["nbdir"] / "reduced.ipynb"
    reduced.write_text(json.dumps(dict(nb, cells=keep)), encoding="utf-8")
    env = _scrubbed_env(tmp_path, clone["dest"])
    out_nb = "executed.ipynb"
    proc = subprocess.run([py_nb, "-m", "jupyter", "nbconvert", "--execute", "--to", "notebook", "--output", out_nb,
                           "--ExecutePreprocessor.timeout=1200", reduced.name],
                          cwd=clone["nbdir"], env=env, capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, f"--- stdout\n{proc.stdout[-3000:]}\n--- stderr\n{proc.stderr[-6000:]}"
    executed = json.loads((clone["nbdir"] / out_nb).read_text(encoding="utf-8"))
    stream = lambda cid: "".join("".join(o.get("text", [])) for c in executed["cells"] if c.get("id") == cid
                                 for o in c.get("outputs", []) if o.get("output_type") == "stream")
    assert f"registry source snapshot · {clone['dest']} @ {clone['sid']} · head {clone['head']}" in stream("583e3f04")
    ztext = stream("b7d4e9f2")
    assert "paper_figures.zip: 0 figures" in ztext and "not written at this render: fig3_single_turn_padding" in ztext and "fig7_frontier" in ztext
    assert f"rendered against registry head {clone['head']}" in stream("23555668")
    assert not [f for f in os.listdir(clone["nbdir"] / "figures") if f.startswith("figT_")]      # no test artifact exists any more
