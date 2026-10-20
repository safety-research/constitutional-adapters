"""qwen-seedrep-native-battery — the compile-cache persist's guards (inode_footprint.md): `persist_compile_caches`
in scripts/modal/seedrep_served.py copies the container's vLLM + FlashInfer compile caches file-for-file onto the volume (10k-35k files
on a volume with ~40k inodes free). It used to run at every container exit unless `SEEDREP_PERSIST_COMPILE_CACHES=0` reached the worker;
wave 1's 11 DRIVERS_FAILED seed-1 rungs each persisted a full tree (+22,834 inodes). Known-bad, pinned here: the persist is OFF by
default (unset -> `skipped_by_default`; set to anything but "1" -> `skipped_by_env`), "1" is the opt-in, and even then a container whose
RESULT is not DONE never persists (`skipped_not_done`). The launcher's setting rides spawn_ctx verbatim ("" when unset), the worker
decides on the ctx value and its own row status, and every skip is recorded in the attempt's evidence."""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "modal"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DRIVER = REPO / "scripts" / "modal" / "seedrep_served.py"
ENV = "SEEDREP_PERSIST_COMPILE_CACHES"


class _Chain:
    def __getattr__(self, name):
        return lambda *a, **k: self


class _App:
    def __init__(self, *a, **k):
        pass

    def function(self, *a, **k):
        return lambda f: f

    def local_entrypoint(self, **kw):
        return lambda f: f


def _stub_modal():
    m = types.ModuleType("modal")
    m.App = _App
    m.Volume = types.SimpleNamespace(from_name=lambda *a, **k: types.SimpleNamespace(commit=lambda: None, reload=lambda: None))
    m.Image = types.SimpleNamespace(from_registry=lambda *a, **k: _Chain(), debian_slim=lambda *a, **k: _Chain())
    m.Dict = types.SimpleNamespace(from_name=lambda *a, **k: {})
    m.is_local = lambda: False
    m.current_function_call_id = lambda: "fc-test"
    return m


_ABSENT = object()


@pytest.fixture(scope="module")
def srv():
    spec = importlib.util.spec_from_file_location("seedrep_served_f5_under_test", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    prior = sys.modules.get("modal", _ABSENT)
    sys.modules["modal"] = _stub_modal()
    try:
        spec.loader.exec_module(mod)
    finally:
        if prior is _ABSENT:
            sys.modules.pop("modal", None)
        else:
            sys.modules["modal"] = prior
    return mod


def _counting_persist(calls):
    def persist():
        calls.append(1)
        return {"vllm": {"shipped": 3}}
    return persist


def test_known_bad_unset_is_off_and_zero_or_other_values_are_off_and_the_copier_is_never_called(srv):
    calls = []
    persist = _counting_persist(calls)
    assert srv.PERSIST_COMPILE_CACHES_ENV == ENV
    # the DEFAULT: unset (a launcher that predates the key, or one that left it alone) -> nothing copied, recorded as the default
    assert srv.compile_cache_persist_evidence(persist, env={}, status="DONE") == "skipped_by_default" == srv.COMPILE_CACHE_SKIPPED_DEFAULT
    assert srv.compile_cache_persist_evidence(persist, env={ENV: ""}, status="DONE") == "skipped_by_default"
    assert srv.compile_cache_persist_evidence(persist, env={ENV: "  "}, status="DONE") == "skipped_by_default"
    # set to anything but "1": off, recorded as the operator's choice
    assert srv.compile_cache_persist_evidence(persist, env={ENV: "0"}, status="DONE") == "skipped_by_env" == srv.COMPILE_CACHE_SKIPPED
    assert srv.compile_cache_persist_evidence(persist, env={ENV: " 0 "}, status="DONE") == "skipped_by_env"
    assert srv.compile_cache_persist_evidence(persist, env={ENV: "yes"}, status="DONE") == "skipped_by_env"       # only "1" turns it on
    assert calls == []                                                          # the copier was NOT called


def test_known_bad_a_non_done_container_never_persists_even_when_asked(srv):
    """The wave-1 shape: `SEEDREP_PERSIST_COMPILE_CACHES=1` reaching a container that ends DRIVERS_FAILED (or any non-DONE status, or a
    caller that does not say) -> nothing copied, recorded as `skipped_not_done`."""
    calls = []
    persist = _counting_persist(calls)
    for status in ("DRIVERS_FAILED", "PETRI_FAILED", "STANDUP_FAILED", "", None):
        assert srv.compile_cache_persist_evidence(persist, env={ENV: "1"}, status=status) == "skipped_not_done" == srv.COMPILE_CACHE_SKIPPED_NOT_DONE, status
    assert srv.compile_cache_persist_evidence(persist, env={ENV: "1"}) == "skipped_not_done"                       # status not given = not DONE
    assert calls == []


def test_opt_in_one_and_a_done_container_persist(srv, monkeypatch):
    calls = []
    persist = _counting_persist(calls)
    assert srv.compile_cache_persist_evidence(persist, env={ENV: "1"}, status="DONE") == {"vllm": {"shipped": 3}}
    assert srv.compile_cache_persist_evidence(persist, env={ENV: " 1 "}, status=srv.RESULT_STATUS_DONE) == {"vllm": {"shipped": 3}}
    assert calls == [1, 1]
    # `env` defaults to the PROCESS environment (what a grading host caller exports), and the copier to the real one
    monkeypatch.delenv(ENV, raising=False)
    assert srv.compile_cache_persist_evidence(status="DONE") == "skipped_by_default"
    monkeypatch.setenv(ENV, "1")
    monkeypatch.setattr(srv, "persist_compile_caches", lambda: {"probe": True})
    assert srv.compile_cache_persist_evidence(status="DONE") == {"probe": True}
    assert srv.compile_cache_persist_evidence(status="DRIVERS_FAILED") == "skipped_not_done"


PLAN_STUB = {"W": "f" * 40, "wave_uuid": "11111111-2222-4333-8444-555555555555", "dicts": {"accepted": "a"}, "plans_root": "/vol/p", "ladder_root": "/vol/l",
             "results_root": "/vol/r", "launch_plan_sha": "0123456789abcdef"}


def test_the_setting_is_read_in_the_launcher_and_rides_spawn_ctx_verbatim_to_the_worker(srv, monkeypatch):
    """The REMOTE worker's environ is the image's, so the launcher / steward reads $SEEDREP_PERSIST_COMPILE_CACHES in ITS process and
    transports the value through spawn_ctx (the channel every operator setting rides) VERBATIM -- "" when unset, so the worker's evidence
    can tell the default from an explicit choice; the worker decides on the ctx value plus its own row status, never on its own environ."""
    ctx0 = srv.spawn_ctx(PLAN_STUB, "app", None, env={ENV: "0"})
    ctxu = srv.spawn_ctx(PLAN_STUB, "app", None, env={})
    ctx1 = srv.spawn_ctx(PLAN_STUB, "app", None, env={ENV: " 1 "})
    assert ctx0[srv.PERSIST_COMPILE_CACHES_CTX] == "0" and ctxu[srv.PERSIST_COMPILE_CACHES_CTX] == "" and ctx1[srv.PERSIST_COMPILE_CACHES_CTX] == "1"
    remote0, remoteu, remote1 = (json.loads(json.dumps(c)) for c in (ctx0, ctxu, ctx1))     # the local -> remote boundary
    assert (remote0[srv.PERSIST_COMPILE_CACHES_CTX], remoteu[srv.PERSIST_COMPILE_CACHES_CTX], remote1[srv.PERSIST_COMPILE_CACHES_CTX]) == ("0", "", "1")
    calls = []
    persist = _counting_persist(calls)
    # the worker's decision is made on the ctx value with its OWN environ saying the opposite
    monkeypatch.setenv(ENV, "1")
    assert srv.compile_cache_persist_evidence(persist, env={ENV: remoteu[srv.PERSIST_COMPILE_CACHES_CTX]}, status="DONE") == "skipped_by_default"
    assert srv.compile_cache_persist_evidence(persist, env={ENV: remote0[srv.PERSIST_COMPILE_CACHES_CTX]}, status="DONE") == "skipped_by_env"
    monkeypatch.setenv(ENV, "0")
    assert srv.compile_cache_persist_evidence(persist, env={ENV: remote1[srv.PERSIST_COMPILE_CACHES_CTX]}, status="DONE") == {"vllm": {"shipped": 3}}
    assert srv.compile_cache_persist_evidence(persist, env={ENV: remote1[srv.PERSIST_COMPILE_CACHES_CTX]}, status="DRIVERS_FAILED") == "skipped_not_done"
    assert calls == [1]
    # a launcher that predates the key carries nothing -> the worker reads "" -> OFF
    assert srv.compile_cache_persist_evidence(persist, env={ENV: str({}.get(srv.PERSIST_COMPILE_CACHES_CTX, ""))}, status="DONE") == "skipped_by_default"
    # the default reads the LAUNCHER's process environment
    monkeypatch.setenv(ENV, "0")
    assert srv.spawn_ctx(PLAN_STUB, "app", None)[srv.PERSIST_COMPILE_CACHES_CTX] == "0"
    monkeypatch.delenv(ENV)
    assert srv.spawn_ctx(PLAN_STUB, "app", None)[srv.PERSIST_COMPILE_CACHES_CTX] == ""


def test_the_row_function_decides_on_the_ctx_value_and_its_own_status_and_records_the_effective_setting(srv):
    """The exit path: the row's `status` is computed BEFORE the evidence is finalised and handed to the persist decision, which reads the
    ctx value with "" (off) as the fallback; the known-bad literals (a bare persist, a "1" fallback) are gone."""
    src = DRIVER.read_text()
    assert "compile_cache_persist=persist_compile_caches()" not in src and "compile_cache_persist=compile_cache_persist_evidence()" not in src
    assert 'ctx.get(PERSIST_COMPILE_CACHES_CTX, "1")' not in src and 'e.get(PERSIST_COMPILE_CACHES_ENV, "1")' not in src
    call = 'compile_cache_persist=compile_cache_persist_evidence(env={PERSIST_COMPILE_CACHES_ENV: str(ctx.get(PERSIST_COMPILE_CACHES_CTX, ""))}, status=status)'
    assert call in src
    assert '"compile_cache_persist_setting": {PERSIST_COMPILE_CACHES_ENV: str(ctx.get(PERSIST_COMPILE_CACHES_CTX, ""))' in src
    body = src[src.index("def _run_standup_row("):]
    i_status = body.index('status = "DONE" if not failed else')
    i_call = body.index(call)
    i_result = body.index("(root / RESULT_FILE).write_text(")
    assert i_status < i_call < i_result                                      # status first, the decision on it, then the RESULT line
    assert srv.RESULT_STATUS_DONE == "DONE"
