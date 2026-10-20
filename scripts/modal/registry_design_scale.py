"""registry_design_scale — the results registry's design-scale sizing gate on a Modal CPU container.

``tests/test_registry_design_scale.py::test_design_scale_peak_footprint_within_10gb`` writes a synthetic 10 M-row
corpus, replays it with ``registry rebuild`` and builds a snapshot under a footprint watcher (the "concurrency"
and "write performance" constraints). It skips unless ``REGISTRY_DESIGN_SCALE=1``
and writes tens of GB, so it never runs on the grading host and the CPU suite (``cpu_suite.py``) never sets the variable:
this app is the ONE place it executes. It stages THIS checkout's ``src/sisteer/registry`` and the one test module
into a slim image (stdlib + pytest + pyarrow — the registry package needs nothing else; the snapshot's Parquet
export needs pyarrow), runs the module with the variable set on a 16-CPU container (its own disk, no shared volume), and
returns pytest's output together with the figures the test records in ``design_scale_report.json``.

Run from the worktree whose code is under test::

  ~/venvs/modal/bin/python -m modal run -n registry-design-scale-<sha7> \\
      scripts/modal/registry_design_scale.py --workers 12 --out ~/registry-design-scale/<sha7>

The local entrypoint prints ONE line ``REGISTRY_JSON {...}`` with the verdict and the figures (rows per minute the
rebuild sustained, peak footprint against the 10 GB budget, the longest transaction, pytest's exit code) and writes
``pytest.log`` + ``design_scale_report.json`` under ``--out``. Defined here, NEVER run by tests (they import this
module with a stub ``modal`` to check its shape, the way ``registry_volume.py`` is checked).
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import modal


def _repo_root() -> pathlib.Path:
    """The checkout this file lives in (scripts/modal/ -> the repo), resolved ONLY where the mounts are built: Modal
    re-imports this module inside the container as ``/root/registry_design_scale.py`` (no such ancestors)."""
    here = pathlib.Path(__file__).resolve()
    return here.parents[2] if len(here.parents) > 2 else here.parent


SENTINEL = "REGISTRY_JSON "
TEST_MODULE = "tests/test_registry_design_scale.py"
SCRATCH = "/scratch/design-scale"          # container-local disk, never a shared volume
GB = 1024 ** 3
#: pinned to the versions the repo's uv.lock resolves (pytest 9.1.1, pyarrow 25.0.0); the rest of the registry is stdlib
PINS = ("pytest==9.1.1", "pyarrow==25.0.0")

app = modal.App("registry-design-scale")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install(*PINS)
         .env({"PYTHONPATH": "/repo/src", "PYTHONUNBUFFERED": "1"}))
if modal.is_local():
    REPO = _repo_root()
    image = (
        image.add_local_file(str(REPO / "src" / "sisteer" / "__init__.py"), "/repo/src/sisteer/__init__.py")
        .add_local_dir(str(REPO / "src" / "sisteer" / "registry"), remote_path="/repo/src/sisteer/registry",
                       ignore=["__pycache__", "*.pyc"])
        .add_local_file(str(REPO / "pyproject.toml"), "/repo/pyproject.toml")          # the registered `design_scale` mark
        .add_local_file(str(REPO / TEST_MODULE), f"/repo/{TEST_MODULE}")
    )


HEARTBEAT_S = 60
#: the gate's phases are silent from the outside (the test prints its report only at the end); the heartbeat names the
#: phase from what is growing — the record log (corpus generation), the ``.building`` database (rebuild), snapshots/
WATCHED = {"records": "reg/records", "db": "registry.sqlite", "db_building": "registry.sqlite.building", "snapshots": "reg/snapshots"}


def _dir_bytes(path: str) -> int:
    total = 0
    for dp, _, fns in os.walk(path):
        for fn in fns:
            try:
                total += os.path.getsize(os.path.join(dp, fn))
            except OSError:
                pass
    return total


def _sizes() -> dict:
    out = {}
    for name, rel in WATCHED.items():
        p = os.path.join(SCRATCH, rel)
        out[name] = _dir_bytes(p) if os.path.isdir(p) else (os.path.getsize(p) if os.path.exists(p) else 0)
    return out


# no ``ephemeral_disk``: Modal accepts that parameter only for 512 GiB–3 TiB requests (an InvalidError below 524288 MiB);
# the container's default disk holds the gate's ≤ 10 GB footprint, and the test itself asserts the peak against that budget.
# The first run (2026-09-17, 5 h timeout) produced NOTHING: pytest's output was captured only at the end, so the timeout
# left no trace of which phase was slow. Output is streamed line by line now, with a heartbeat of the scratch sizes.
@app.function(image=image, cpu=16, memory=65536, timeout=10 * 3600)
def run_design_scale_fn(workers: int) -> dict:
    """Run the gated module with ``REGISTRY_DESIGN_SCALE=1``; stream its output; return pytest's rc, the log, the report
    and the heartbeat samples (phase sizes over time)."""
    import subprocess
    import threading
    import time
    os.makedirs(SCRATCH, exist_ok=True)
    env = dict(os.environ, REGISTRY_DESIGN_SCALE="1", REGISTRY_DESIGN_SCRATCH=SCRATCH, REGISTRY_WORKERS=str(int(workers)),
               PYTHONPATH="/repo/src")
    argv = [sys.executable, "-m", "pytest", f"/repo/{TEST_MODULE}", "--rootdir=/repo", "-c", "/repo/pyproject.toml",
            "-o", "addopts=", "-p", "no:cacheprovider", "-rA", "-s", "-v", "--tb=short"]
    t0 = time.monotonic()
    lines: list = []
    samples: list = []
    stop = threading.Event()

    def heartbeat() -> None:
        while not stop.wait(HEARTBEAT_S):
            sizes = _sizes()
            samples.append({"t_s": round(time.monotonic() - t0), **sizes})
            print(f"[design-scale] t={round(time.monotonic() - t0)}s " + " ".join(f"{k}={v / 1e9:.2f}GB" for k, v in sizes.items()), flush=True)
    hb = threading.Thread(target=heartbeat, daemon=True)
    hb.start()
    proc = subprocess.Popen(argv, cwd="/repo", env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    rc = proc.wait()
    stop.set()
    elapsed = time.monotonic() - t0
    report = None
    rp = os.path.join(SCRATCH, "design_scale_report.json")
    if os.path.exists(rp):
        with open(rp, "r", encoding="utf-8") as fh:
            report = json.load(fh)
    return {"pytest_rc": rc, "elapsed_s": round(elapsed, 1), "command": argv, "workers": int(workers),
            "container_cpus": os.cpu_count(), "report": report, "log": "".join(lines)[-400_000:], "heartbeat": samples[-600:],
            "final_sizes": _sizes()}


def summarize(res: dict) -> dict:
    """The figures the acceptance table wants, derived from the test's report (None where the test never got there)."""
    rep = res.get("report") or {}
    counts = rep.get("counts") or {}
    rows = sum(int(counts.get(k) or 0) for k in ("item_rows_rev1", "item_rows_rev2", "regrade_rows"))
    wall = rep.get("rebuild_wall_s")
    write_s = rep.get("write_s")
    out = {
        "pytest_rc": res.get("pytest_rc"), "passed": res.get("pytest_rc") == 0, "elapsed_s": res.get("elapsed_s"),
        "workers": res.get("workers"), "container_cpus": res.get("container_cpus"),
        "item_rows": rows or None, "runs": counts.get("runs"),
        "rebuild_wall_s": wall, "parse_s": rep.get("parse_s"), "write_s": write_s,
        "rows_per_min_rebuild_wall": round(rows / (wall / 60.0)) if rows and wall else None,
        "rows_per_min_write_phase": round(rows / (write_s / 60.0)) if rows and write_s else None,
        "longest_transaction_s": rep.get("longest_transaction_s"),
        "peak_total_bytes": rep.get("peak_total_bytes"),
        "peak_total_gb": round(rep["peak_total_bytes"] / GB, 3) if rep.get("peak_total_bytes") is not None else None,
        "budget_gb": 10, "peak_breakdown": rep.get("peak_breakdown"),
        "db_bytes": rep.get("db_bytes"), "db_gb": round(rep["db_bytes"] / GB, 3) if rep.get("db_bytes") is not None else None,
        "runlevel_bytes": rep.get("runlevel_bytes"), "parquet_bytes": rep.get("parquet_bytes"), "parquet_ratio": rep.get("parquet_ratio"),
        "snapshot_s": rep.get("snapshot_s"), "frame_v_paper_long_s": rep.get("frame_v_paper_long_s"),
        "corpus_generation_s": rep.get("corpus_generation_s"), "refused": rep.get("refused"),
        "peak_rss_bytes": rep.get("peak_rss_bytes"), "peak_child_rss_bytes": rep.get("peak_child_rss_bytes"),
        "final_sizes": res.get("final_sizes"),
    }
    return out


def _emit(obj: object) -> None:
    sys.stdout.write(SENTINEL + json.dumps(obj, sort_keys=True, default=str) + "\n")
    sys.stdout.flush()


@app.local_entrypoint()
def main(workers: int = 12, out: str = "") -> None:
    res = run_design_scale_fn.remote(int(workers))
    summary = summarize(res)
    if out:
        d = pathlib.Path(out).expanduser()
        d.mkdir(parents=True, exist_ok=True)
        (d / "pytest.log").write_text(res.get("log") or "", encoding="utf-8")
        if res.get("report") is not None:
            (d / "design_scale_report.json").write_text(json.dumps(res["report"], indent=1, sort_keys=True), encoding="utf-8")
        (d / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True), encoding="utf-8")
        (d / "heartbeat.json").write_text(json.dumps(res.get("heartbeat") or [], indent=1), encoding="utf-8")
        summary["out"] = str(d)
    else:
        sys.stdout.write((res.get("log") or "")[-20_000:] + "\n")
    _emit(summary)
    if res.get("pytest_rc") != 0:
        raise SystemExit(1)
