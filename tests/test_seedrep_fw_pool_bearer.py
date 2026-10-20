"""fw: every multi-turn child exited rc 0 with ``n_target_forwards_total=0`` while
the attacker pools logged ``POST /v1/chat/completions -> 401`` on every call. Two halves, both pinned here against a LOOPBACK
bearer-guarded fake pool (401 without ``Authorization: Bearer <pool key>``, 200 with it):

1. the DRIVER containers never carried the lane pool bearer, so ``jailbreak_run._vllm_api_key`` fell back to ``$VLLM_API_KEY``
   -- the TARGET proxy's bearer in that container -- and the pools rejected it. Now the bearer rides a DISTINCT env name
   (``attacker_pool.POOL_BEARER_ENV``) delivered by the pool app's own transport (a Modal secret built from the runtime-dir key,
   ``seedrep_served.POOL_BEARER_SECRETS``), handed to the ``--attacker-endpoint`` children only, and checked at spawn time
   (absent -> LaunchRefused; rejected by a pool -> LaunchRefused naming the pool, never the bearer).
2. a driver whose every goal failed attacker-side exited 0 (PyRIT swallowed the cause). Now the child log names the exception
   chain's ROOT (``HTTP 401`` + the endpoint), the first pool 401 aborts the run, and ``generate`` exits non-zero by name
   (``NO_TARGET_FORWARDS``) when goals ran and no target forward happened.

Every bearer here is assembled at run time (never a credential-shaped literal); the real ``$XDG_RUNTIME_DIR`` is never read.
"""
from __future__ import annotations

import hashlib
import http.server
import importlib.util
import json
import os
import secrets
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "modal"))

import jailbreak_run  # noqa: E402
from sisteer.eval.attacks import attacker_pool as contract  # noqa: E402
from sisteer.eval.attacks import multiturn  # noqa: E402
from sisteer.eval.attacks import pyrit_adapters as pa  # noqa: E402
from sisteer.eval.attacks import served_target as st  # noqa: E402
from sisteer.eval.attacks import vllm_attacker as va  # noqa: E402

SERVED = REPO / "scripts" / "modal" / "seedrep_served.py"
POOL_APP = REPO / "scripts" / "modal" / "seedrep_attacker_pool.py"
ENV = "SEEDREP_ATTACKER_POOL_KEY"
POOL_KEY = "fwpool-" + secrets.token_hex(20)        # the lane bearer the fake pool guards with
PROXY_KEY = "fwproxy-" + secrets.token_hex(20)      # the TARGET proxy's bearer ($VLLM_API_KEY in a driver container)
ATTACKER = "the-attacker"


# --------------------------------------------------------------------------- the loopback bearer-guarded pool

class _Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):                           # quiet
        pass

    def _send(self, code, body):
        raw = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _authed(self):
        return self.headers.get("Authorization") == f"Bearer {self.server.key}"

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, {"status": "ok"})           # vLLM's /health is not bearer-guarded
        self.server.seen.append(("GET", self.path, self.headers.get("Authorization")))
        if self.path == "/v1/models":
            return self._send(200, {"data": [{"id": ATTACKER}]}) if self._authed() else self._send(401, {"error": "Unauthorized"})
        self._send(404, {})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(n)
        self.server.seen.append(("POST", self.path, self.headers.get("Authorization")))
        if self.path == "/v1/chat/completions":
            if not self._authed():
                return self._send(401, {"error": "Unauthorized"})
            return self._send(200, {"choices": [{"message": {"content": "ok-from-pool"}}]})
        self._send(404, {})


@pytest.fixture
def pool():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    srv.daemon_threads = True
    srv.key = POOL_KEY
    srv.seen = []
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield SimpleNamespace(url=f"http://127.0.0.1:{srv.server_address[1]}", key=POOL_KEY, seen=srv.seen)
    finally:
        srv.shutdown()
        srv.server_close()


def _bearers(seen, method, path):
    return [auth for m, p, auth in seen if m == method and p == path]


# --------------------------------------------------------------------------- 0. ONE env name, ONE path spelling

def test_the_pool_bearer_env_name_is_one_literal_across_the_contract_the_driver_app_and_the_pool_app(tmp_path):
    srv = _load_served(is_local=False, environ={})
    assert contract.POOL_BEARER_ENV == ENV == srv.POOL_BEARER_ENV
    assert contract.POOL_BEARER_ENV != contract.API_KEY_ENV                # DISTINCT from the pools' / proxy's VLLM_API_KEY
    pool_src = POOL_APP.read_text()
    assert "DRIVER_POOL_KEY_ENV = contract.POOL_BEARER_ENV" in pool_src   # the pool app re-exports the contract's ONE name
    # the served app's read-only reader spells the pool app's runtime-dir path (<rd>/seedrep-battery/attacker-pool/<lane>/api_key)
    assert srv.local_pool_bearer_path("seedrep", {"XDG_RUNTIME_DIR": str(tmp_path)}) == tmp_path / "seedrep-battery" / "attacker-pool" / "seedrep" / "api_key"
    assert 'Path(rd) / "seedrep-battery" / "attacker-pool" / LANE' in pool_src and 'return _runtime_key_dir() / "api_key"' in pool_src
    assert srv.local_pool_bearer_path("seedrep", {}) is None


# --------------------------------------------------------------------------- 1. the driver sends the POOL bearer to the pool

def _seams(pool_url, monkeypatch, *, proxy_key=PROXY_KEY, pool_key=None):
    monkeypatch.setenv("VLLM_API_KEY", proxy_key)          # what a driver container's environ carries for the TARGET proxy
    if pool_key is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, pool_key)
    cfg = {"attacker": {"model": ATTACKER}, "attacks": {}}
    args = SimpleNamespace(attack="actorattack", attacker_engine="vllm", attacker_endpoint=pool_url,
                           attacker_model=None, scorer_endpoint=None, scorer_model=None)
    return jailbreak_run._build_attacker_seams(cfg, args), args


def test_known_bad_without_the_lane_env_the_driver_sends_the_proxys_bearer_and_the_pool_401s(pool, monkeypatch):
    """The known-bad mechanism, reproduced through the REAL chain (_build_attacker_seams -> _vllm_api_key -> VLLMAttackerClient -> the
    loopback pool): with only $VLLM_API_KEY in the environ the attacker call carries the TARGET proxy's bearer and 401s."""
    (scorer_fn, factory, attacker_fn), _ = _seams(pool.url, monkeypatch, pool_key=None)
    with pytest.raises(va.VLLMEndpointUnauthorized) as ei:
        attacker_fn([{"role": "user", "content": "hi"}])
    assert _bearers(pool.seen, "POST", "/v1/chat/completions") == [f"Bearer {PROXY_KEY}"]       # the WRONG bearer, once (no retries)
    msg = str(ei.value)
    assert "401" in msg and pool.url in msg and PROXY_KEY not in msg and POOL_KEY not in msg


def test_fixed_with_the_lane_env_the_driver_sends_the_pool_bearer_to_the_attacker_endpoint_only(pool, monkeypatch):
    (scorer_fn, factory, attacker_fn), args = _seams(pool.url, monkeypatch, pool_key=POOL_KEY)
    assert attacker_fn([{"role": "user", "content": "hi"}]) == "ok-from-pool"
    assert scorer_fn("rate it") == "ok-from-pool"                              # the in-loop scorer shares the attacker endpoint
    assert set(_bearers(pool.seen, "POST", "/v1/chat/completions")) == {f"Bearer {POOL_KEY}"}
    # the resolution rule itself: the attacker endpoint -> the lane bearer; a scorer endpoint elsewhere -> $VLLM_API_KEY (unchanged)
    assert jailbreak_run._vllm_api_key(args, pool.url) == POOL_KEY
    assert jailbreak_run._vllm_api_key(args, pool.url + "/") == POOL_KEY
    assert jailbreak_run._vllm_api_key(args, "http://127.0.0.1:9/") == PROXY_KEY
    # an --attacker-pool record's key still wins for the attacker endpoint (the record path keeps working)
    args._attacker_pool_key = "record-" + secrets.token_hex(8)
    assert jailbreak_run._vllm_api_key(args, pool.url) == args._attacker_pool_key
    assert jailbreak_run._vllm_api_key(args, "http://127.0.0.1:9") == PROXY_KEY


def test_vllm_api_key_legacy_per_job_server_path_is_unchanged(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    args = SimpleNamespace(attacker_endpoint="http://127.0.0.1:8000")
    assert jailbreak_run._vllm_api_key(args, "http://127.0.0.1:8000") is None            # a loopback per-job server: no header
    monkeypatch.setenv("VLLM_API_KEY", PROXY_KEY)
    assert jailbreak_run._vllm_api_key(args, "http://127.0.0.1:8000") == PROXY_KEY        # a guarded per-job server, as before


# --------------------------------------------------------------------------- 2a. the child log names the ROOT; the first 401 aborts

def test_pyrit_wrapper_chain_names_the_attacker_401_and_never_the_bearer(capsys):
    inner = va.VLLMEndpointUnauthorized.for_endpoint("http://pool.example:8000", 401, api_key_set=True)
    try:
        try:
            raise inner
        except Exception as ex:
            raise Exception("Error sending prompt with conversation ID: 1111") from ex      # pyrit's prompt_normalizer wrapper
    except Exception as wrapped:
        exc = wrapped
    assert pa._error_type_name(exc) == "VLLMEndpointUnauthorized"        # was `Exception` (the wrapper) before fw
    assert va.attacker_unauthorized_in(exc) is inner
    assert va.attacker_unauthorized_in(RuntimeError("x")) is None
    root = va.root_cause_line(exc)
    assert root.startswith("VLLMEndpointUnauthorized:") and "401" in root and "http://pool.example:8000" in root
    line = pa.log_goal_error("crescendo", "g7", exc)
    err = capsys.readouterr().err
    assert line in err and "401" in err and "http://pool.example:8000" in err and "g7" in err


class _FakeServedTarget:
    """A served target whose chat is never reached (the attacker 401s first) -- counts what served_target counts."""
    serving_engine = st.NATIVE if hasattr(st, "NATIVE") else "lora_native"

    def __init__(self):
        self.n_target_forwards = 0
        self.last_decode = None

    def chat(self, messages, *, max_new_tokens, **kw):
        self.n_target_forwards += 1
        return "target reply"


def test_actorattack_aborts_on_the_first_pool_401_and_the_log_names_the_endpoint_not_the_bearer(pool, tmp_path, capsys):
    """The REAL chain (run_actorattack -> attacker_fn -> VLLMAttackerClient -> the loopback pool) with the WRONG bearer: the
    first goal records error:VLLMEndpointUnauthorized and the run ABORTS (one POST, no retries, no second goal) instead of
    burning every goal on 401s; stderr names 401 + the endpoint and never either bearer."""
    client = va.VLLMAttackerClient(pool.url, ATTACKER, api_key=PROXY_KEY, max_retries=2)
    attacker_fn = va.make_attacker_fn_vllm(client, temperature=0.5, top_p=0.9, max_new_tokens=64)
    goals = [SimpleNamespace(goal_id=f"g{i}", text=f"goal {i}", category="c") for i in range(3)]
    defense = _FakeServedTarget()
    with pytest.raises(va.VLLMEndpointUnauthorized):
        multiturn.run_actorattack(defense, goals, attacker_fn=attacker_fn, scorer_fn=lambda p: "1", out_dir=tmp_path,
                                  max_retries=3, discovery_retries=2, target_temperature=None)
    posts = _bearers(pool.seen, "POST", "/v1/chat/completions")
    assert posts == [f"Bearer {PROXY_KEY}"]                                     # ONE call: the 401 is never retried
    rows = [json.loads(l) for l in (tmp_path / "progress.jsonl").read_text().splitlines()]
    assert [r["goal_id"] for r in rows] == ["g0"] and rows[0]["outcome"] == "error:VLLMEndpointUnauthorized"
    assert defense.n_target_forwards == 0
    out = capsys.readouterr()
    log = out.out + out.err
    assert "401" in log and pool.url in log and "g0" in log
    assert POOL_KEY not in log and PROXY_KEY not in log


# --------------------------------------------------------------------------- 2b. generate exits non-zero on zero forwards

def test_no_target_forwards_rule_scope():
    ok = {"n_goals_run": 3, "n_goals_total": 3, "n_target_forwards_total": 7}
    assert jailbreak_run.no_target_forwards_problem("crescendo", ok) is None
    resumed = {"n_goals_run": 0, "n_goals_total": 3, "n_target_forwards_total": 0}     # every goal already covered: legit
    assert jailbreak_run.no_target_forwards_problem("crescendo", resumed) is None
    bad = {"n_goals_run": 3, "n_goals_total": 3, "n_target_forwards_total": 0,
           "per_goal": [{"outcome": "error:VLLMEndpointUnauthorized"}] * 3}
    msg = jailbreak_run.no_target_forwards_problem("crescendo", bad)
    assert msg and msg.startswith("[crescendo] NO_TARGET_FORWARDS: goals_run=3 but n_target_forwards_total=0")
    assert "error:VLLMEndpointUnauthorized x3" in msg


class _FakeServed:
    def __init__(self):
        self.virtual_model = "petri__si_constitution__psr_kl_all"
        self.route_info = {"intervention_hash": "cc" * 8, "virtual_model": self.virtual_model}
        self.evidence = {"steering_bites": {"applicable": True, "verified": True}}
        self.timing_log = None
        self.n_target_forwards = 0
        self.mode = "si_constitution__psr_kl_all"
        self.max_model_len = None
        self.n_clamped = 0
        self.n_prompt_over_window = 0
        self.n_window_mismatch = 0
        self.window_mismatches = []
        self.n_tokenize_resolved = 0
        self.n_floor_only_unresolvable = 0
        self.n_all_turns_refused = {}
        self.n_timing_rows = 0
        self.cell_id = self.mode
        self.route_info.update(tt_cell_id=self.mode, harness_cell_id=self.mode, serve_variant=None)


def _generate(monkeypatch, tmp_path, *, forwards, attack="crescendo"):
    """cmd_generate on the served multi-turn path with the gated runner faked to report ``forwards`` target forwards."""
    monkeypatch.setattr(st, "build_served_target", lambda **kw: _FakeServed())
    monkeypatch.setattr(jailbreak_run, "_valid_defense_ids", lambda cfg: {"none", "si_constitution__psr_kl_all"})
    goals = [SimpleNamespace(goal_id=f"g{i}", text=f"goal {i}", category="c") for i in range(3)]
    monkeypatch.setattr(jailbreak_run, "_eval_goals", lambda cfg, *, confirmatory: goals)
    monkeypatch.setattr(jailbreak_run, "_build_attacker_seams",
                        lambda cfg, args, log_dir=None: ((lambda p: "5"), (lambda lock: None), (lambda chat: "x")))

    def fake_run(attack_, defense, goals_arg, **kw):
        return {"n_goals_run": len(goals_arg), "n_goals_total": len(goals_arg), "n_target_forwards_total": forwards,
                "per_goal": [{"goal_id": g.goal_id, "outcome": ("error:Exception" if forwards == 0 else "failure")} for g in goals_arg]}
    monkeypatch.setattr(jailbreak_run, "_run_gated_attack", fake_run)
    cfg = {"model": "m", "dtype": "bfloat16", "private_run": "jb", "attacker": {"model": "qwen"},
           "attacks": {"forward_cap": 150, "target_max_new_tokens": 2048, "target_decode": "generation_config"}}
    args = SimpleNamespace(attack=attack, defense="si_constitution__psr_kl_all", shard=0, nshards=1, xstest=False,
                           confirmatory=True, trial=0, private_root=str(tmp_path / "state"), target_engine="served",
                           target_proxy_url="http://127.0.0.1:7/v1", target_routing_table=str(tmp_path / "rt.json"),
                           positions="response", scale=1.0, goal_ids_file=None, expect_pair_n_streams=None,
                           attacker_engine="vllm", attacker_endpoint="http://127.0.0.1:8000")
    return jailbreak_run.cmd_generate(args, cfg)


def test_generate_exits_non_zero_by_name_when_goals_ran_and_no_target_forward_happened(monkeypatch, tmp_path, capsys):
    """At the base tip this returned None (rc 0) -- the false pass: `done; goals_run=3/3; n_target_forwards_total=0`."""
    with pytest.raises(SystemExit) as ei:
        _generate(monkeypatch, tmp_path, forwards=0)
    assert "[crescendo] NO_TARGET_FORWARDS: goals_run=3 but n_target_forwards_total=0" in str(ei.value)
    assert ei.value.code != 0
    assert "n_target_forwards_total=0" in capsys.readouterr().out                 # the done line still prints first
    # the manifests were still written (the finally ran) so the store says what the run saw
    store = next((tmp_path / "state").rglob("served_target.json"))
    assert json.loads(store.read_text())["manifest_stage"] == "final"


def test_generate_with_target_forwards_still_returns_normally(monkeypatch, tmp_path):
    assert _generate(monkeypatch, tmp_path, forwards=3) is None


def test_generate_turns_the_attacker_401_abort_into_a_named_exit(monkeypatch, tmp_path):
    monkeypatch.setattr(st, "build_served_target", lambda **kw: _FakeServed())
    monkeypatch.setattr(jailbreak_run, "_valid_defense_ids", lambda cfg: {"none", "si_constitution__psr_kl_all"})
    monkeypatch.setattr(jailbreak_run, "_eval_goals", lambda cfg, *, confirmatory: [SimpleNamespace(goal_id="g0", text="t", category="c")])
    monkeypatch.setattr(jailbreak_run, "_build_attacker_seams",
                        lambda cfg, args, log_dir=None: ((lambda p: "5"), (lambda lock: None), (lambda chat: "x")))

    def fake_run(*a, **kw):
        raise va.VLLMEndpointUnauthorized.for_endpoint("http://127.0.0.1:8000", 401, api_key_set=True)
    monkeypatch.setattr(jailbreak_run, "_run_gated_attack", fake_run)
    cfg = {"model": "m", "dtype": "bfloat16", "private_run": "jb", "attacker": {"model": "qwen"},
           "attacks": {"forward_cap": 150, "target_max_new_tokens": 2048, "target_decode": "generation_config"}}
    args = SimpleNamespace(attack="pair", defense="si_constitution__psr_kl_all", shard=0, nshards=1, xstest=False,
                           confirmatory=True, trial=0, private_root=str(tmp_path / "state"), target_engine="served",
                           target_proxy_url="http://127.0.0.1:7/v1", target_routing_table=str(tmp_path / "rt.json"),
                           positions="response", scale=1.0, goal_ids_file=None, expect_pair_n_streams=None,
                           attacker_engine="vllm", attacker_endpoint="http://127.0.0.1:8000")
    with pytest.raises(SystemExit) as ei:
        jailbreak_run.cmd_generate(args, cfg)
    assert str(ei.value).startswith("[pair] ATTACKER_UNAUTHORIZED:") and "401" in str(ei.value) and "http://127.0.0.1:8000" in str(ei.value)


# --------------------------------------------------------------------------- 3. the served app: secret, child env, spawn gate

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
    m.Secret = types.SimpleNamespace(from_dict=lambda d: {"secret": dict(d)})      # records what the app would ship
    m.is_local = lambda: is_local
    m.current_function_call_id = lambda: "fc-test"
    m.forward = None
    m.FunctionCall = None
    return m


def _load_served(*, is_local: bool, environ: dict):
    """Import seedrep_served.py under a stub modal, with XDG_RUNTIME_DIR / the bearer env swapped for ``environ`` for the
    import (so the module-level read never touches the real runtime dir); returns the module with `_functions` = the
    recorded @app.function kwargs by name."""
    functions: dict = {}
    prior = sys.modules.get("modal")
    saved = {k: os.environ.get(k) for k in ("XDG_RUNTIME_DIR", ENV)}
    for k in saved:
        os.environ.pop(k, None)
    os.environ.update(environ)
    sys.modules["modal"] = _stub_modal(is_local=is_local, functions=functions)
    try:
        spec = importlib.util.spec_from_file_location(f"seedrep_served_fw_{is_local}_{len(environ)}", SERVED)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        if prior is None:
            sys.modules.pop("modal", None)
        else:
            sys.modules["modal"] = prior
        for k in saved:
            os.environ.pop(k, None)
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
    mod._functions = functions
    return mod


def _write_key(rd: Path, key: str, *, mode=0o600) -> Path:
    d = rd / "seedrep-battery" / "attacker-pool" / "seedrep"
    d.mkdir(parents=True, mode=0o700)
    p = d / "api_key"
    p.write_text(key + "\n")
    p.chmod(mode)
    return p


def _final_plan():
    row_mt = {"standup_id": "crescendo__s1__base__shard00", "kind": "base", "leg": "crescendo", "pool": "seedrep-attacker-1",
              "engine": "s1", "stage": "base", "shard": 0}
    row_static = {"standup_id": "pasttense__s1__base__shard00", "kind": "base", "leg": "pasttense", "engine": "s1", "stage": "base", "shard": 0}
    return {"materialisation": "final", "standups": [row_mt, row_static], "rungs": []}


def test_read_local_pool_bearer_is_read_only_and_refuses_by_name(tmp_path):
    srv = _load_served(is_local=False, environ={})
    rd = tmp_path / "rt"
    rd.mkdir(mode=0o700)
    env = {"XDG_RUNTIME_DIR": str(rd)}
    bearer, why = srv.read_local_pool_bearer("seedrep", env)
    assert bearer == "" and "no bearer file at" in why
    assert not (rd / "seedrep-battery").exists()                                  # NEVER minted (the pool app's job)
    p = _write_key(rd, POOL_KEY)
    assert srv.read_local_pool_bearer("seedrep", env) == (POOL_KEY, "")
    p.chmod(0o644)
    bearer, why = srv.read_local_pool_bearer("seedrep", env)
    assert bearer == "" and "not private" in why
    p.chmod(0o600)
    p.write_text("\n")
    assert srv.read_local_pool_bearer("seedrep", env)[1].endswith("is empty")
    assert srv.read_local_pool_bearer("seedrep", {})[1].startswith("XDG_RUNTIME_DIR is unset")
    # the pool app's own read-only twin spells the same file (text pin: it cannot be imported without pool params + the mint)
    assert "def read_local_api_key() -> str:" in POOL_APP.read_text()


def test_the_grading_host_import_reads_the_runtime_dir_key_into_the_secret_and_the_container_import_reads_only_its_env(tmp_path):
    rd = tmp_path / "rt"
    rd.mkdir(mode=0o700)
    _write_key(rd, POOL_KEY)
    # the container side (is_local False): the runtime dir is never read; the env var (the Modal secret) is the source
    srv_c = _load_served(is_local=False, environ={"XDG_RUNTIME_DIR": str(rd)})
    assert srv_c.POOL_BEARER == "" and srv_c.POOL_BEARER_SECRETS == [{"secret": {}}]         # an EMPTY secret: no env var exported
    srv_c2 = _load_served(is_local=False, environ={"XDG_RUNTIME_DIR": str(rd), ENV: POOL_KEY})
    assert srv_c2.POOL_BEARER == POOL_KEY
    # the grading host side is pinned at the source (importing under is_local=True runs the whole local bootstrap): the read is the
    # runtime-dir reader and the secret is built from it, by the pool app's own transport
    src = SERVED.read_text()
    assert "if modal.is_local():\n    POOL_BEARER, POOL_BEARER_PROBLEM = read_local_pool_bearer()\n" in src
    assert "POOL_BEARER_SECRETS = pool_bearer_secrets(POOL_BEARER)" in src
    assert srv_c.pool_bearer_secrets(POOL_KEY) == [{"secret": {ENV: POOL_KEY}}]
    # the three DRIVER-running functions carry the secret; the Petri proxy (no in-container driver) does not
    for fn in ("run_view_leg", "run_arditi_leg", "run_rung"):
        assert srv_c._functions[fn]["secrets"] is srv_c.POOL_BEARER_SECRETS, fn
    assert "secrets" not in srv_c._functions["run_petri_proxy"]


def test_spawn_gate_refuses_multi_turn_rows_without_a_bearer_and_with_a_bearer_the_pools_reject(tmp_path, capsys):
    srv = _load_served(is_local=False, environ={})
    plan = _final_plan()
    urls = {"seedrep-attacker-1": "http://pool-1.example"}
    healthy = lambda u: True  # noqa: E731
    # (d) no runtime-dir key on the grading host -> the multi-turn row refuses BEFORE any spawn, naming the reader's problem
    rd = tmp_path / "rt"
    rd.mkdir(mode=0o700)
    srv.POOL_BEARER, srv.POOL_BEARER_PROBLEM = srv.read_local_pool_bearer("seedrep", {"XDG_RUNTIME_DIR": str(rd)})
    with pytest.raises(srv.LaunchRefused) as ei:
        srv.select_launch_rows(plan, go="crescendo__s1__base__shard00", pool_urls=urls, pool_health=healthy)
    msg = str(ei.value)
    assert "need the lane attacker-pool bearer" in msg and ENV in msg and "no bearer file at" in msg and "D99" in msg
    # a static row needs no bearer (unchanged)
    rows, _ = srv.select_launch_rows(plan, go="pasttense__s1__base__shard00", pool_urls=urls, pool_health=healthy)
    assert [r["standup_id"] for r in rows] == ["pasttense__s1__base__shard00"]
    # the key rendered -> the row passes when every pinned pool accepts the bearer; the log carries the sha12, never the bearer
    _write_key(rd, POOL_KEY)
    srv.POOL_BEARER, srv.POOL_BEARER_PROBLEM = srv.read_local_pool_bearer("seedrep", {"XDG_RUNTIME_DIR": str(rd)})
    asked = []

    def accepts(url, bearer):
        asked.append((url, bearer))
        return bearer == POOL_KEY
    rows, _ = srv.select_launch_rows(plan, go="crescendo__s1__base__shard00", pool_urls=urls, pool_health=healthy, pool_accepts=accepts)
    assert [r["standup_id"] for r in rows] == ["crescendo__s1__base__shard00"] and asked == [("http://pool-1.example", POOL_KEY)]
    out = capsys.readouterr().out
    sha12 = hashlib.sha256(POOL_KEY.encode()).hexdigest()[:12]
    assert sha12 in out and POOL_KEY not in out
    # a pool that rejects the bearer refuses the spawn NAMING THE POOL, never the bearer
    with pytest.raises(srv.LaunchRefused) as ei:
        srv.select_launch_rows(plan, go="crescendo__s1__base__shard00", pool_urls=urls, pool_health=healthy, pool_accepts=lambda u, b: False)
    msg = str(ei.value)
    assert "'seedrep-attacker-1' at http://pool-1.example rejects the lane bearer" in msg and POOL_KEY not in msg and sha12 in msg
    # an explicit bearer argument overrides the module's read (the pure-function seam)
    with pytest.raises(srv.LaunchRefused) as ei:
        srv.select_launch_rows(plan, go="crescendo__s1__base__shard00", pool_urls=urls, pool_health=healthy, pool_bearer="", pool_accepts=accepts)
    assert "no bearer was given" in str(ei.value)
    # health is probed BEFORE the bearer: a down pool is still the down-pool refusal
    with pytest.raises(srv.LaunchRefused) as ei:
        srv.select_launch_rows(plan, go="crescendo__s1__base__shard00", pool_urls=urls, pool_health=lambda u: False, pool_bearer=POOL_KEY, pool_accepts=accepts)
    assert "is not healthy" in str(ei.value)


def test_pool_accepts_bearer_is_the_authenticated_models_probe(pool):
    srv = _load_served(is_local=False, environ={})
    assert srv.pool_accepts_bearer(pool.url, POOL_KEY) is True
    assert srv.pool_accepts_bearer(pool.url + "/", POOL_KEY) is True
    assert srv.pool_accepts_bearer(pool.url, PROXY_KEY) is False               # the proxy's bearer: 401
    assert srv.pool_accepts_bearer("http://127.0.0.1:1", POOL_KEY) is False    # nothing listening
    assert _bearers(pool.seen, "GET", "/v1/models") == [f"Bearer {POOL_KEY}", f"Bearer {POOL_KEY}", f"Bearer {PROXY_KEY}"]


def test_the_bearer_reaches_only_the_attacker_endpoint_children_and_a_bare_container_fails_the_row_by_name():
    srv = _load_served(is_local=False, environ={})
    base = {"PATH": "/usr/bin", ENV: POOL_KEY, "HF_HOME": "/x"}
    # container_env scrubs the bearer from every child's base env (alongside the five keys strength scrubs)
    cenv = srv.container_env(bank_view="/vol/bank", tree="/root/tree", thinking=False, base=base)
    assert ENV not in cenv and "HF_HOME" in cenv
    mt_argv = ["/venv/bin/python", "/root/tree/scripts/jailbreak_run.py", "generate", "--attack", "crescendo",
               "--attacker-engine", "vllm", "--attacker-endpoint", "https://pool-1.modal.run", "--target-engine", "served"]
    static_argv = ["/venv/bin/python", "/root/tree/scripts/jailbreak_run.py", "generate", "--attack", "pasttense", "--target-engine", "served"]
    container_environ = {ENV: POOL_KEY, "MODAL_TASK_ID": "ta-1"}
    got = srv.child_env_with_pool_bearer(mt_argv, cenv, container_environ)
    assert got[ENV] == POOL_KEY and got["PATH"] == "/usr/bin"
    assert ENV not in srv.child_env_with_pool_bearer(static_argv, cenv, container_environ)
    assert ENV not in srv.child_env_with_pool_bearer(static_argv, {**cenv, ENV: POOL_KEY}, container_environ)   # scrubbed even if leaked in
    assert srv.child_env_with_pool_bearer(mt_argv + ["--x"], cenv, {ENV: POOL_KEY})[ENV] == POOL_KEY
    with pytest.raises(LookupError) as ei:                                   # a pool child in a container with no secret: fail-closed
        srv.child_env_with_pool_bearer(mt_argv, cenv, {"MODAL_TASK_ID": "ta-1"})
    assert ENV in str(ei.value) and "--attacker-endpoint" in str(ei.value)
    # the container's spawn loop turns that into the row's own named failure (source pin: the hunk sits right after resolve_spawn)
    src = SERVED.read_text()
    i = src.index("k_argv, k_env = lp.resolve_spawn(k, spawn_context)")
    hunk = src[i:i + 1200]
    assert "k_env = child_env_with_pool_bearer(k_argv, k_env, os.environ)" in hunk and 'return fail("POOL_BEARER_MISSING"' in hunk
    # ... and a failed child (rc != 0, the driver's NO_TARGET_FORWARDS / ATTACKER_UNAUTHORIZED exits) still makes the row DRIVERS_FAILED
    assert 'failed = [c for c in children_out if c.get("rc") not in (0, None)]' in src
    assert 'status = "DONE" if not failed else ("PETRI_" + petri_outcome["state"].upper() if is_petri else "DRIVERS_FAILED")' in src
