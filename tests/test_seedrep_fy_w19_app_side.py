"""fy: three app-side defects in ``scripts/modal/seedrep_served.py``, each pinned known-bad -> fixed.

* **F-AB** -- ``LANE`` in the served app is the lane's VOLUME PATH (``lp.LANE`` == ``/vol/seedrep``), but fw's
  ``local_pool_bearer_path(lane=LANE)`` joined it into the runtime-dir key path; pathlib collapses a join onto an absolute
  component, so the spawn gate looked for the bearer at ``/vol/seedrep/api_key`` and refused every pool-pinned row while the
  key sat, 0600, at ``$XDG_RUNTIME_DIR/seedrep-battery/attacker-pool/seedrep/api_key``. Now the default is the pool app's lane
  NAME (``POOL_LANE == seedrep_attacker_pool.LANE``) and a path-like lane refuses by name.
* **F-AA** -- the Modal path never wrote ``exp/serve/standup_jobs.jsonl`` (the cluster sbatch did), which
  ``audit_common.read_standup`` requires, so every in-container audit refused one layer past F-U; and ``mt_serving_up.py``
  unlinks ``endpoint.json`` at teardown, so the audits that run after teardown refused on that too. Now the row writes ONE job
  line in the cluster shape when the standup publishes its record, and ``teardown_standup`` restores the READY ``endpoint.json``
  bytes once the standup process has exited (the volume already held them: the TreeSyncer shipped them at "ready" and never
  deletes).
* **F-AG** -- the metrics scrape outlived teardown and appended ``{"t", "error"}`` rows to ``vllm_metrics.jsonl`` for the whole
  producer-evidence phase; ``utilisation_table.read_metrics`` refuses any row without the running key. Now ``_start_samplers``
  returns a handle that ``teardown_standup`` stops and JOINS before the engine goes away, no row is written after stop, and the
  metrics evidence records ``n_error_rows``.

Every bearer here is assembled at run time (never a credential-shaped literal); the real ``$XDG_RUNTIME_DIR`` is never read.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import os
import secrets
import sys
import threading
import time
import types
from pathlib import Path
from urllib.error import URLError

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "modal", REPO / "scripts" / "sdfbase" / "seedrep"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import audit_common as ac  # noqa: E402
import seedrep_launch_plan as lp  # noqa: E402
from sdfbase.full22.sampled import reconstruct_prev_endpoint as rpe  # noqa: E402

SERVED = REPO / "scripts" / "modal" / "seedrep_served.py"
POOL_APP = REPO / "scripts" / "modal" / "seedrep_attacker_pool.py"
MT_SERVING_UP = REPO / "scripts" / "sdfbase" / "mt_serving_up.py"
ENV = "SEEDREP_ATTACKER_POOL_KEY"
W = "f" * 40


# --------------------------------------------------------------------------- the served app under a stub modal (fw's idiom)

class _Chain:
    def __getattr__(self, name):
        return lambda *a, **k: self


class _FakeDict:
    store: dict = {}

    def __init__(self, name):
        self.name = name
        _FakeDict.store.setdefault(name, {})

    @classmethod
    def from_name(cls, name, create_if_missing=False):
        return cls(name)


def _stub_modal(*, is_local: bool, functions: dict):
    m = types.ModuleType("modal")

    class _App:
        def __init__(self, name=None, **k):
            self.description = None

        def function(self, **kw):
            def deco(f):
                functions[f.__name__] = kw
                return f
            return deco

        def local_entrypoint(self, **kw):
            return lambda f: f

    m.App = _App
    m.Volume = types.SimpleNamespace(from_name=lambda *a, **k: types.SimpleNamespace(commit=lambda: None, reload=lambda: None))
    m.Image = types.SimpleNamespace(from_registry=lambda *a, **k: _Chain(), debian_slim=lambda *a, **k: _Chain())
    m.Dict = _FakeDict
    m.Secret = types.SimpleNamespace(from_dict=lambda d: {"secret": dict(d)})
    m.is_local = lambda: is_local
    m.current_function_call_id = lambda: "fc-test"
    m.forward = None
    m.FunctionCall = None
    return m


def _load_served():
    """Import seedrep_served.py under a stub modal as a CONTAINER import (is_local False), with the runtime dir and the bearer env
    removed for the import so the module-level read never touches the real runtime dir."""
    functions: dict = {}
    prior = sys.modules.get("modal")
    saved = {k: os.environ.get(k) for k in ("XDG_RUNTIME_DIR", ENV)}
    for k in saved:
        os.environ.pop(k, None)
    sys.modules["modal"] = _stub_modal(is_local=False, functions=functions)
    try:
        spec = importlib.util.spec_from_file_location("seedrep_served_fy", SERVED)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        if prior is None:
            sys.modules.pop("modal", None)
        else:
            sys.modules["modal"] = prior
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
    mod._functions = functions
    return mod


@pytest.fixture(scope="module")
def srv():
    return _load_served()


def _write_key(rd: Path, key: str) -> Path:
    d = rd / "seedrep-battery" / "attacker-pool" / "seedrep"
    d.mkdir(parents=True, mode=0o700)
    p = d / "api_key"
    p.write_text(key + "\n")
    p.chmod(0o600)
    return p


def _pool_app_lane_literal() -> str:
    """``seedrep_attacker_pool.LANE`` read from the SOURCE (the module's import reads env / resolves the checkout -- side effects a
    test must not run); the literal is the equality the app's ``POOL_LANE`` spells."""
    tree = ast.parse(POOL_APP.read_text())
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "LANE" for t in n.targets):
            assert isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)
            return n.value.value
    raise AssertionError("seedrep_attacker_pool.py has no module-level LANE literal")


def _row_fn(src: str) -> ast.FunctionDef:
    tree = ast.parse(src)
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_run_standup_row")


# =========================================================================== F-AB: the lane NAME, never the lane's volume path

def test_pool_lane_is_the_pool_apps_lane_name_and_the_volume_path_lane_is_refused(srv, tmp_path):
    assert srv.POOL_LANE == _pool_app_lane_literal() == "seedrep"
    # the path constant stays what it is (the lane's volume root) and is DISTINCT from the name
    assert srv.LANE == lp.LANE and srv.LANE.startswith("/") and srv.POOL_LANE != srv.LANE
    rd = tmp_path / "rt"
    # known-bad at c65b47ba4: the default lane WAS srv.LANE and pathlib collapsed the join onto it -- the gate looked at /vol/seedrep/api_key
    assert Path(rd) / "seedrep-battery" / "attacker-pool" / srv.LANE / "api_key" == Path(srv.LANE) / "api_key" == Path("/vol/seedrep/api_key")
    # fixed: both readers default to the NAME and resolve under the runtime dir at the pool app's spelling
    for fn in (srv.local_pool_bearer_path, srv.read_local_pool_bearer):
        assert inspect.signature(fn).parameters["lane"].default == srv.POOL_LANE, fn.__name__
    assert srv.local_pool_bearer_path(environ={"XDG_RUNTIME_DIR": str(rd)}) == rd / "seedrep-battery" / "attacker-pool" / "seedrep" / "api_key"
    assert srv.local_pool_bearer_path(environ={}) is None
    # a path-like lane REFUSES by name -- the class cannot recur silently
    for bad in (srv.LANE, "/vol/seedrep", "a/b", "seedrep/", "", ".", ".."):
        with pytest.raises(ValueError, match="bare lane NAME"):
            srv.local_pool_bearer_path(bad, {"XDG_RUNTIME_DIR": str(rd)})
        with pytest.raises(ValueError, match="bare lane NAME"):
            srv.read_local_pool_bearer(bad, {"XDG_RUNTIME_DIR": str(rd)})


def test_the_default_lane_reads_the_rendered_bearer_and_the_spawn_gate_accepts_a_pool_pinned_row(srv, tmp_path, capsys):
    rd = tmp_path / "rt"
    rd.mkdir(mode=0o700)
    key = "fy-" + secrets.token_hex(20)
    # before the render: the refusal names the runtime-dir path, never the volume path the known-bad code reported
    _, why = srv.read_local_pool_bearer(environ={"XDG_RUNTIME_DIR": str(rd)})
    assert why.startswith("no bearer file at") and str(rd) in why and "/vol/seedrep" not in why
    _write_key(rd, key)
    bearer, problem = srv.read_local_pool_bearer(environ={"XDG_RUNTIME_DIR": str(rd)})     # the module-level grading host read's call shape
    assert (bearer, problem) == (key, "")
    # the spawn gate, fed by that read: the pool-pinned row spawns (known-bad: all seven refused here)
    row_mt = {"standup_id": "crescendo__s1__base__shard00", "kind": "base", "leg": "crescendo", "pool": "seedrep-attacker-1",
              "engine": "s1", "stage": "base", "shard": 0}
    plan = {"materialisation": "final", "standups": [row_mt], "rungs": []}
    urls = {"seedrep-attacker-1": "http://pool-1.example"}
    srv.POOL_BEARER, srv.POOL_BEARER_PROBLEM = bearer, problem
    try:
        rows, _ = srv.select_launch_rows(plan, go="crescendo__s1__base__shard00", pool_urls=urls, pool_health=lambda u: True,
                                         pool_accepts=lambda u, b: b == key)
    finally:
        srv.POOL_BEARER, srv.POOL_BEARER_PROBLEM = "", ""
    assert [r["standup_id"] for r in rows] == ["crescendo__s1__base__shard00"]
    out = capsys.readouterr().out
    assert hashlib.sha256(key.encode()).hexdigest()[:12] in out and key not in out


# =========================================================================== F-AA: the standup job record + the endpoint record after teardown

def _endpoint() -> dict:
    return {"attested_max_num_seqs": 45, "attested_max_model_len": 11520, "attested_cudagraph_mode": "FULL", "attested_cudagraph_full_capture_max": 45,
            "geometry": {"adjustments": [{"kind": "lora_kernel_index_ceiling", "from_S": 47, "to_S": 45, "D": 2560, "ceiling": 524287}],
                         "window_source": "plan"},
            "model_key": "qwen35_4b", "engine": "lora_native", "serving_path": "lora_native", "thinking_mode": "off", "upstream_timeout_s": 600.0,
            "gpu_memory_utilization": 0.9, "vmodels": {"cell-a": "vm-a"}, "proxy_url": "http://127.0.0.1:1/", "proxy_port": 1, "proxy_token": None,
            "served_models": ["vm-a"], "lora_config": {"max_lora_rank": 64, "max_loras": 4}}


def _row() -> dict:
    return {"standup_id": "pasttense__s1__base__shard00", "kind": "base", "leg": "pasttense", "engine": "s1", "view": "s1", "stage": "base",
            "shard": 0, "geometry": {"S": 45, "L": 11520}}


def _attempt_root(tmp_path: Path):
    root = tmp_path / "attempt"
    serve = root / "exp" / "serve"
    serve.mkdir(parents=True)
    (serve / "endpoint.json").write_text(json.dumps(_endpoint(), indent=2))
    (serve / "serving_attestation.json").write_text(json.dumps({"kind": "serving_attestation", "schema": 2}))
    (serve / "cells_served.json").write_text(json.dumps({"canonical": [{"cell_id": "cell-a", "virtual_model_name": "vm-a"}], "controls": [],
                                                         "cells_skipped": []}))
    aroot = ac.AttemptRoot(path=root, attempt_id="ta-11111111-2222", W=W, leg="pasttense", view="s1", shard="00", stage="base", started_utc=None,
                           attempt={})
    return root, serve, aroot


def test_standup_job_line_is_read_by_the_real_read_standup_and_carries_the_cluster_fields(srv, tmp_path):
    root, serve, aroot = _attempt_root(tmp_path)
    # known-bad (every row): the root without the file refuses exactly where determinism_check / request_evidence_audit did
    with pytest.raises(ac.AuditRefusal, match=r"no '?standup_jobs.jsonl'? under"):
        ac.read_standup(aroot)
    cells_served = json.loads((serve / "cells_served.json").read_text())
    line = srv.standup_job_line(_endpoint(), cells_served, _row(), attempt_id="ta-11111111-2222", call_id="fc-1", W=W, node="TESTHOST",
                                at="2026-01-01T00:00:00Z")
    p = srv.write_standup_jobs(serve, line)
    assert p == serve / ac.STANDUP_JOBS_FILE == root / f"{srv.SERVE_DIR}/standup_jobs.jsonl" and f"{srv.SERVE_DIR}/standup_jobs.jsonl" in srv.ATTEMPT_ROOT_LAYOUT
    assert p.read_text().count("\n") == 1                                       # ONE job line
    st = ac.read_standup(aroot)
    assert st.standup_jobs == [json.loads(json.dumps(line, default=str))] and st.standup_jobs_path == p
    assert st.standup_jobs[0]["sha"] == W == aroot.W and st.S == 45 and st.L == 11520   # engine_identity_table's sha == ATTEMPT.json W check
    # the cluster shape: every field reconstruct_prev_endpoint copies from a cluster standup record is present, spelled the same
    assert set(rpe.COPIED) <= set(line), sorted(set(rpe.COPIED) - set(line))
    assert line["job_id"] == "ta-11111111-2222" and line["node"] == "TESTHOST" and line["key"] == "qwen35_4b" and line["view"] == "s1"
    assert line["mode"] == "pasttense" and line["eval"] == "pasttense" and line["cluster"] == "modal" and line["substrate"] == "modal"
    assert line["max_num_seqs"] == line["attested_max_num_seqs"] == 45 and line["max_model_len"] == line["attested_max_model_len"] == 11520
    assert line["batched_tokens"] == 45 * 11520 and line["lens_reduction_mode"] is None and line["engine"] == "lora_native"
    assert line["geometry_adjustments"][0]["kind"] == "lora_kernel_index_ceiling" and line["upstream_timeout_s"] == 600.0
    assert line["thinking_mode"] == "off" and line["written_at"] == "2026-01-01T00:00:00Z" and line["standup_id"] == "pasttense__s1__base__shard00"
    # a controls container spells its eval legs the way EVAL_LEGS_OF does
    ctrl = srv.standup_job_line(_endpoint(), cells_served, {**_row(), "leg": "controls", "standup_id": "controls__s1__base__shard00"},
                                attempt_id="a", call_id=None, W=W, node="TESTHOST")
    assert ctrl["eval"] == ",".join(lp.EVAL_LEGS_OF["controls"]) and ctrl["mode"] == "controls"


def test_the_ready_endpoint_record_is_restored_after_the_standups_own_unlink(srv, tmp_path):
    root, serve, aroot = _attempt_root(tmp_path)
    cells_served = json.loads((serve / "cells_served.json").read_text())
    srv.write_standup_jobs(serve, srv.standup_job_line(_endpoint(), cells_served, _row(), attempt_id="a", call_id=None, W=W, node="TESTHOST"))
    ep = serve / "endpoint.json"
    ep_bytes = ep.read_bytes()
    assert srv.restore_endpoint_record(ep, ep_bytes) is None and ep.read_bytes() == ep_bytes     # still published: nothing to do
    # what mt_serving_up.py does at teardown (its pollers' down signal) -- pinned at the source so the restore stays load-bearing
    assert '(out / "endpoint.json").unlink()' in MT_SERVING_UP.read_text()
    ep.unlink()
    with pytest.raises(ac.AuditRefusal, match=r"no 'endpoint.json' under"):       # known-bad: the post-teardown audit's refusal
        ac.read_standup(aroot)
    rec = srv.restore_endpoint_record(ep, ep_bytes)
    assert ep.read_bytes() == ep_bytes and rec["sha256"] == hashlib.sha256(ep_bytes).hexdigest() and rec["bytes"] == len(ep_bytes)
    assert ac.read_standup(aroot).endpoint == _endpoint()                                        # byte-identical READY record, read again


def test_the_row_writes_the_job_line_at_publication_and_teardown_stops_samplers_then_restores_the_record(srv):
    src = SERVED.read_text()
    fn = _row_fn(src)
    body = "\n".join(src.splitlines()[fn.lineno - 1:fn.end_lineno])
    # the job line is written once endpoint.json + cells_served.json are read, BEFORE the readiness verdict (a mismatch root still has its record)
    assert body.index("ep_bytes = ep.read_bytes()") < body.index("write_standup_jobs(serve_dir, standup_job_line(") < body.index("readiness_problems(endpoint, cells_served, row")
    # teardown_standup's order: samplers.stop -> STOP file -> wait -> restore_endpoint_record
    td = next(n for n in ast.walk(fn) if isinstance(n, ast.FunctionDef) and n.name == "teardown_standup")
    tsrc = "\n".join(src.splitlines()[td.lineno - 1:td.end_lineno])
    assert tsrc.index("samplers.stop(") < tsrc.index("stop_file.touch()") < tsrc.index("standup.wait(") < tsrc.index("restore_endpoint_record(ep, ep_bytes)")
    assert "n_error_rows=samplers.n_error_rows" in tsrc
    # the handle is bound from the pinned call site (test_proxy_metrics_passthrough keeps the argument shape) and is None before it
    assert 'samplers = _start_samplers(serve_dir, root, stop, {"metrics_url": metrics_url, "metrics_headers": metrics_headers})' in body
    assert body.index("samplers: Optional[Samplers] = None") < body.index("def teardown_standup(") < body.index("samplers = _start_samplers(")
    # the audits that need the record run after teardown and read the RESTORED file (F-AA), determinism still before (needs the live engine)
    assert body.index('evidence["determinism"] = determinism_evidence(') < body.index("teardown_standup(timeout=600)") < body.index("producer_evidence(root, tree=tree")
    # the row's SHARED stop (heartbeat / tree-sync) is still set only after producer-evidence -- the sampler no longer depends on it
    assert body.index("producer_evidence(root, tree=tree") < body.rindex("    stop.set()")


# =========================================================================== F-AG: the metrics scrape stops at teardown

def _metrics_page(srv, running: float) -> str:
    fams = srv.VLLM_METRIC_FAMILIES
    assert {"vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:generation_tokens_total"} <= set(fams)
    return "# HELP\n" + "\n".join(f"{w} {running if w.endswith('running') else 1.0}" for w in fams) + "\n"


def _no_nvidia_smi(*a, **k):
    raise FileNotFoundError("nvidia-smi")


def _wait(pred, timeout=5.0):
    t0 = time.monotonic()
    while not pred():
        assert time.monotonic() - t0 < timeout, "condition not reached"
        time.sleep(0.01)


@pytest.fixture
def fast_sampler(srv, monkeypatch, tmp_path):
    monkeypatch.setattr(srv, "DMON_CADENCE_S", 0.02)
    monkeypatch.setattr(srv, "subprocess", types.SimpleNamespace(Popen=_no_nvidia_smi, STDOUT=-2))
    state = {"up": True, "calls": 0}

    def fake_get(url, timeout=5, headers=None):
        state["calls"] += 1
        if not state["up"]:
            raise URLError("Connection refused")
        return _metrics_page(srv, 3.0)
    monkeypatch.setattr(srv, "_http_get_text", fake_get)
    root = tmp_path / "attempt"
    (root / "exp" / "serve").mkdir(parents=True)
    return state, root


def _rows(root: Path):
    return [json.loads(l) for l in (root / "vllm_metrics.jsonl").read_text().splitlines() if l.strip()]


def test_sampler_is_stopped_and_joined_and_writes_no_row_after_stop(srv, fast_sampler):
    state, root = fast_sampler
    from sdfbase.seedrep import utilisation_table as ut
    stop = threading.Event()
    h = srv._start_samplers(root / "exp" / "serve", root, stop, {"metrics_url": "http://127.0.0.1:1/metrics", "metrics_headers": None})
    assert isinstance(h, srv.Samplers) and h.thread.is_alive()
    _wait(lambda: h.n_rows >= 3)
    # teardown: the handle stops and JOINS the thread; the row's shared stop is untouched (heartbeat / sync keep running after teardown)
    assert h.stop(timeout=2.0) is True and not h.thread.is_alive() and not stop.is_set()
    n_at_stop = len(_rows(root))
    # known-bad: the engine goes away AFTER the sampler stopped -- the old thread kept scraping and appended an error row per cadence
    state["up"] = False
    calls_at_stop = state["calls"]
    time.sleep(0.2)                                                           # ~10 cadences
    rows = _rows(root)
    assert len(rows) == n_at_stop and state["calls"] == calls_at_stop, "a row or a scrape after stop"
    assert all("error" not in r and "vllm:num_requests_running" in r for r in rows) and h.n_error_rows == 0 and h.n_rows == len(rows)
    # and the finished root READS: utilisation_table's reader accepts every row (known-bad: refused on the first post-teardown error row)
    got = ut.read_metrics(root / "vllm_metrics.jsonl")
    assert len(got) == len(rows) and all(r["running"] == 3.0 for r in got)


def test_an_engine_failure_while_up_is_still_recorded_and_counted_but_never_after_stop(srv, fast_sampler):
    state, root = fast_sampler
    stop = threading.Event()
    h = srv._start_samplers(root / "exp" / "serve", root, stop, {"metrics_url": "http://127.0.0.1:1/metrics", "metrics_headers": None})
    _wait(lambda: h.n_rows >= 2)
    state["up"] = False                                                       # a live-engine failure: an error row IS evidence
    _wait(lambda: h.n_error_rows >= 2)
    assert h.stop(timeout=2.0) is True
    rows = _rows(root)
    n_err = sum(1 for r in rows if "error" in r)
    assert n_err == h.n_error_rows >= 2 and all("Connection refused" in r["error"] for r in rows if "error" in r)
    time.sleep(0.2)
    assert len(_rows(root)) == len(rows)                                      # nothing after stop, error or otherwise


def test_the_rows_shared_stop_still_ends_the_scrape_on_a_fail_path(srv, fast_sampler):
    state, root = fast_sampler
    stop = threading.Event()
    h = srv._start_samplers(root / "exp" / "serve", root, stop, {"metrics_url": "http://127.0.0.1:1/metrics", "metrics_headers": None})
    _wait(lambda: h.n_rows >= 1)
    stop.set()                                                                # `fail(...)` sets the shared stop
    state["up"] = False
    _wait(lambda: not h.thread.is_alive(), timeout=2.0)
    n = len(_rows(root))
    time.sleep(0.1)
    assert len(_rows(root)) == n and all("error" not in r for r in _rows(root))
