"""qwen-seedrep-native-battery fs4 (F-AS): the GSM8K child's Inspect provider gets the ONE bearer every client of this proxy sends.

The finding (both gsm8k smokes, every child at once, right after attestation): each GSM8K child (`served_capability_run.py generate`
-> `inspect_served.run_inspect_tasks` -> `build_model` -> our `tinycap` provider -> inspect_ai's `OpenAICompatibleAPI.__init__`) died with
`PrerequisiteError: Unable to initialise tinycap client -- No TINYCAP_API_KEY defined in the environment`. The driver handed the runner
`api_key=os.environ.get(TINYCAP_API_KEY)`; the seedrep app composes no bearer variable of ANY name into a child's env (its lora-native
standup records `proxy_token: None`: the routing proxy checks no token, and the ENGINE's per-launch `VLLM_API_KEY` never leaves the standup
process), so the key was None and inspect_ai -- which wants SOME non-empty key whether or not the server checks one -- refused. The controls
children on the SAME standup passed because their client (`ProxyClientEngine`, `_make_client`) never reads the env: it sends the proxy's
default bearer (`proxy.proxy_client_token`: the record's `proxy_token` when it names one, else `PROXY_CLIENT_BEARER_DEFAULT`).

The fix matches that working path exactly, in the driver where both legs live: `_inspect_api_key(standup_endpoint)` = `$TINYCAP_API_KEY`
when the environment names one (the cluster sbatch / tinycap_served spelling, byte-identical), else `proxy_client_token` over the standup
record the run already read -- one bearer source, one rule, for the Inspect AND EvalPlus legs. No child env gains a name, nothing rides
argv, nothing is logged.

RE-VALIDATED through the REAL composer (`build_plan`), the REAL app child-argv/env composition (`gsm8k_child_up` -> `child_argv_env`) and
the REAL provider construction (`inspect_served.build_model` with inspect_ai's own `get_model`). Synthetic ids, hostname-free, no secret
values (a fake spelled bearer is a test string, not a credential)."""
from __future__ import annotations

import importlib.util
import inspect
import json
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("inspect_ai")  # the Modal cpu-suite image does not install inspect_ai; the contract is unchanged where it exists

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "modal"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import seedrep_launch_plan as lp  # noqa: E402
from seedrep_fixes_common import load_served_capability_run  # noqa: E402
from sisteer.capeval.serving import native_standup as ns  # noqa: E402
from sisteer.capeval.serving import proxy as px  # noqa: E402
from sisteer.capeval.serving import vllm_auth  # noqa: E402
from sisteer.capeval.tiers import inspect_served as isv  # noqa: E402
from sisteer.capeval.tinycap import contracts as tc  # noqa: E402

DRIVER = REPO / "scripts" / "modal" / "seedrep_served.py"
W = "2" * 40
WAVE = "22222222-3333-4444-8444-555555555555"
CONTAINER_ATTEMPT = "ta-11111111-2222"          # the app's attempt id (`mint_attempt_id`: <task id>-<uuid4[:12]>), synthetic
PROXY = "http://127.0.0.1:9/v1"
PASSTHROUGH = "proxy_vllm+passthrough+interleaved"
#: the `none` control's route as the routing table spells it (test_inspect_served's fixture shape): what `build_model` verifies stamps against
ROUTE_NONE = {"cell_id": "none", "virtual_model": "petri__base", "serving_identity": PASSTHROUGH, "cell_class": "control",
              "intervention_hash": "0" * 16}
_ABSENT = object()


class _Chain:
    def __getattr__(self, _name):
        return lambda *a, **k: self

    def __call__(self, *a, **k):
        return self


class _App:
    def __init__(self, name=None, **k):
        self.description = None

    def function(self, **kw):
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


@pytest.fixture(scope="module")
def srv():
    spec = importlib.util.spec_from_file_location("seedrep_served_w25_fs4", DRIVER)
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


@pytest.fixture(scope="module")
def draft():
    inputs = lp.load_inputs(REPO)
    return lp.build_plan(W=W, wave_uuid=WAVE, reqs_sha256=inputs["reqs_sha256"], roster=inputs["roster"], pins=inputs["pins"],
                         arditi_pins=inputs["arditi_pins"], arms=inputs["arms"], materialisation="draft", generated_at="2026-09-18T00:00:00Z")


@pytest.fixture(scope="module")
def srun():
    return load_served_capability_run()


#: every name a bearer has ridden in this lane's processes: the Inspect provider's, the serving stack's (the engine's per-launch token,
#: `determinism_check.PROXY_BEARER_ENV`), the attacker pools' (a Modal secret the app hands --attacker-endpoint children alone)
def _bearer_names(srv) -> set:
    return {tc.INSPECT_API_KEY_ENV, vllm_auth.TOKEN_ENV, srv.POOL_BEARER_ENV}


class _FakeLedgerPopen:
    """The usage ledger as `gsm8k_child_up` sees it: records its argv, writes the endpoint record the ledger writes once its probe passed."""
    started: list = []

    def __init__(self, argv, **kw):
        self.argv, self.kw, self.pid, self.rc = list(argv), kw, 4_000_000, None
        type(self).started.append(self)
        ep = Path(argv[argv.index("--endpoint-json") + 1])
        ep.parent.mkdir(parents=True, exist_ok=True)
        ep.write_text(json.dumps({"upstream": argv[argv.index("--upstream") + 1], "ledger_path": argv[argv.index("--ledger-path") + 1], "pid": self.pid,
                                  "probe": {"ok": True}, "attempt_id": argv[argv.index("--attempt-id") + 1],
                                  "ledger_url": f"http://127.0.0.1:{argv[argv.index('--port') + 1]}/v1"}))

    def poll(self):
        return self.rc

    def wait(self, timeout=None):
        self.rc = 0 if self.rc is None else self.rc
        return self.rc


def _serve_dir(tmp_path: Path) -> Path:
    """The standup's serve dir as the lane's native standup leaves it: the record names NO proxy token (`native_standup` writes the literal
    `proxy_token: None` -- pinned below), so every client of this proxy sends the default bearer and the proxy checks none."""
    serve = tmp_path / "serve"
    serve.mkdir(parents=True)
    (serve / "routing_table.json").write_text(json.dumps({"routes": {}}))
    (serve / "endpoint.json").write_text(json.dumps({"proxy_url": PROXY, "routing_table": str(serve / "routing_table.json"), "proxy_token": None}))
    return serve


def _spawn(srv, k, sid, *, root: Path, serve: Path, child_subs=None, base_env=None):
    """THE app's composition for one child: `child_argv_env` (= `lp.resolve_spawn` over the app's `SpawnContext`)."""
    return srv.child_argv_env(k, attempt_root=str(root), plans_root="/vol/p", proxy_url=PROXY, routing_table=str(serve / "routing_table.json"),
                              driver_python="/py", client_python="/cpy", S=31, ncells=8, bank_view="/b", base_env=dict(base_env or {}),
                              attempt_id=CONTAINER_ATTEMPT, standup_id=sid, child_subs=child_subs)


def _gsm8k_picks(draft):
    """The casualties by class: the `none` child (`gsm8k__none__s1`, the fetched log), a `prompted__*` child and a fitted cell of the s1 BASE
    row; an Arditi base child; an s1 RUNG child."""
    base = [r for r in draft["standups"] if r["leg"] == "gsm8k" and r["engine"] == "s1"][0]
    picks = [(base, [k for k in base["children"] if k["cell"] == "none"][0]),
             (base, [k for k in base["children"] if k["cell"] in lp.PROMPTED_CELLS][0]),
             (base, [k for k in base["children"] if k["cell"] == lp.FITTED_CELLS[0]][0])]
    ard = [r for r in draft["standups"] if r["leg"] == "gsm8k" and r["engine"] == lp.ARDITI][0]
    picks.append((ard, ard["children"][0]))
    rung = [r for r in draft["rungs"] if r["leg"] == "gsm8k" and r["engine"] == "s1"][0]
    picks.append((rung, rung["children"][0]))
    return picks


def _compose_gsm8k(srv, srun, row, k, *, root: Path, serve: Path, monkeypatch):
    """`gsm8k_child_up` (the per-store ledger seam, the child's serve-dir copy of the standup record) -> `child_argv_env`: the FINAL
    (argv, env, the record the child's `--routing-table` sits beside -- what the driver's `_standup_endpoint` reads)."""
    monkeypatch.setattr(srv.os, "killpg", lambda pid, sig: None)
    _FakeLedgerPopen.started = []
    st = srv.gsm8k_child_up(k, root=root, serve_dir=serve, proxy_url=PROXY, driver_python="/py", tree=REPO, env={"PATH": "/usr/bin"},
                            popen=_FakeLedgerPopen, wait_s=5, sleep=lambda s: None)
    argv, env = _spawn(srv, k, row["standup_id"], root=root, serve=serve, child_subs={lp.child_key(k): st["subs"]}, base_env={"PATH": "/usr/bin"})
    table = Path(lp._flag_value(argv, "--routing-table"))
    record = json.loads((table.parent / srun.STANDUP_ENDPOINT_FILENAME).read_text())
    return argv, env, record


def _build(argv, env, *, api_key):
    """The REAL provider construction the driver runs for the target (`run_inspect_tasks` -> `build_model` -> inspect_ai `get_model` -> our
    `tinycap` provider -> `OpenAICompatibleAPI.__init__`), on the child's OWN ledger URL (its `--ledger-url`, the driver's `base_url`)."""
    return isv.build_model(ROUTE_NONE["virtual_model"], base_url=lp._flag_value(argv, "--ledger-url"), api_key=api_key, si=None,
                           thinking=tc.THINKING_OFF, concurrency=int(lp._flag_value(argv, "--concurrency")), route=ROUTE_NONE)


def test_before_w24_the_composed_gsm8k_env_carried_no_bearer_and_the_provider_refused_exactly_as_every_child_did(srv, srun, draft, tmp_path, monkeypatch):
    """BEFORE, re-validated on the casualty (`gsm8k__none__s1`) and every other GSM8K child class: the app's composed env carries NO
    bearer under ANY name (there is no proxy secret to carry: the standup record names `proxy_token: None`), so the pre-fix resolution
    `os.environ.get(TINYCAP_API_KEY)` is None -- and `build_model(api_key=None)` under that env is EXACTLY the smokes' refusal,
    inspect_ai's `PrerequisiteError` naming TINYCAP_API_KEY, raised before any request."""
    from inspect_ai._util.error import PrerequisiteError

    for name in _bearer_names(srv):
        monkeypatch.delenv(name, raising=False)
    for i, (row, k) in enumerate(_gsm8k_picks(draft)):
        argv, env, record = _compose_gsm8k(srv, srun, row, k, root=tmp_path / f"attempt{i}", serve=_serve_dir(tmp_path / f"s{i}"), monkeypatch=monkeypatch)
        assert set(env) & _bearer_names(srv) == set(), (k["cell"], k["s"], sorted(set(env) & _bearer_names(srv)))
        assert record["proxy_token"] is None and lp.unresolved_placeholders(argv) == []
        assert env.get(tc.INSPECT_API_KEY_ENV) is None                          # the pre-fix driver's `api_key=os.environ.get(TINYCAP_API_KEY)`
        with pytest.raises(PrerequisiteError, match=r"tinycap client[\s\S]*TINYCAP_API_KEY") as ei:
            _build(argv, env, api_key=None)
        assert "TINYCAP_API_KEY" in str(ei.value) and "defined in the environment" in str(ei.value), (k["cell"], k["s"])
    # why THAT name: the provider declares it as its key variable (inspect_ai reads `api_key_var` when no key is passed) and the contract spells it
    assert tc.INSPECT_API_KEY_ENV == "TINYCAP_API_KEY" and "api_key_var=C.INSPECT_API_KEY_ENV" in inspect.getsource(isv)
    # why the env was empty: the lane's native standup records no proxy token (the routing proxy checks none) and the app never
    # composes the engine's own per-launch bearer into a child (it never leaves the standup process)
    assert '"proxy_token": None' in inspect.getsource(ns)
    for src in (inspect.getsource(lp.resolve_spawn), inspect.getsource(srv.container_env), inspect.getsource(srv.child_argv_env)):
        assert not any(name in src for name in _bearer_names(srv) - {srv.POOL_BEARER_ENV})


def test_after_the_inspect_legs_bearer_is_the_controls_legs_own_rule_and_the_provider_initialises(srv, srun, draft, tmp_path, monkeypatch):
    """AFTER: for every GSM8K child class, the driver's `_inspect_api_key(standup record)` under the composed env (no TINYCAP_API_KEY) is
    `proxy_client_token(record)` = `PROXY_CLIENT_BEARER_DEFAULT` = the bearer the controls leg's `ProxyClientEngine` (`_make_client`, no
    api_key kwarg) sends the SAME proxy -- and the REAL provider construction with it initialises on the child's own ledger URL, carrying that
    key and the `tinycap` service. Never None; nothing on argv; the composed env is untouched."""
    for name in _bearer_names(srv):
        monkeypatch.delenv(name, raising=False)
    for i, (row, k) in enumerate(_gsm8k_picks(draft)):
        argv, env, record = _compose_gsm8k(srv, srun, row, k, root=tmp_path / f"attempt{i}", serve=_serve_dir(tmp_path / f"s{i}"), monkeypatch=monkeypatch)
        key = srun._inspect_api_key(record, environ=env)
        controls_client = px.ProxyClientEngine(PROXY, identity=ROUTE_NONE["serving_identity"])         # `_make_client`'s construction
        assert key == px.proxy_client_token(record) == px.PROXY_CLIENT_BEARER_DEFAULT == controls_client.api_key and key
        assert key not in argv and not any(key in a for a in argv) and set(env) & _bearer_names(srv) == set()
        model = _build(argv, env, api_key=key)
        assert model.api.api_key == key and model.api.service == tc.INSPECT_PROVIDER_NAME
        assert model.api.base_url == lp._flag_value(argv, "--ledger-url") and model.api.service_model_name() == ROUTE_NONE["virtual_model"]
    # the controls leg's client is built with the default bearer, by construction (no api_key kwarg at the one call site)
    assert "ProxyClientEngine(proxy_url, identity=route.serving_identity)" in inspect.getsource(srun._make_client)
    assert inspect.signature(px.ProxyClientEngine.__init__).parameters["api_key"].default == px.PROXY_CLIENT_BEARER_DEFAULT
    # the rule's three arms, names only: the environment's spelling wins (the cluster sbatch / tinycap_served path, byte-identical); an
    # EMPTY spelling falls through; a record that names a token hands it over; no record at all is still the default, never None
    assert srun._inspect_api_key({"proxy_token": None}, environ={tc.INSPECT_API_KEY_ENV: "spelled-by-the-launcher"}) == "spelled-by-the-launcher"
    assert srun._inspect_api_key({"proxy_token": None}, environ={tc.INSPECT_API_KEY_ENV: ""}) == px.PROXY_CLIENT_BEARER_DEFAULT
    assert srun._inspect_api_key({"proxy_token": "named-by-the-record"}, environ={}) == "named-by-the-record" == px.proxy_client_token({"proxy_token": "named-by-the-record"})
    assert srun._inspect_api_key(None, environ={}) == px.PROXY_CLIENT_BEARER_DEFAULT
    assert srun._inspect_api_key({}, environ={}) == px.PROXY_CLIENT_BEARER_DEFAULT


def test_the_driver_hands_the_one_rule_to_both_inspect_side_legs_and_reads_the_env_nowhere_else(srun):
    """`cmd_generate` passes `api_key=_inspect_api_key(standup_endpoint)` to the Inspect runner AND the EvalPlus runner (the two clients that
    demand a key), and the driver reads TINYCAP_API_KEY through the helper alone -- the pre-fix `os.environ.get(tc.INSPECT_API_KEY_ENV)` is gone."""
    generate = inspect.getsource(srun.cmd_generate)
    assert generate.count("api_key=_inspect_api_key(standup_endpoint)") == 2
    assert "inspect_served.run_inspect_tasks(" in generate and "evalplus_runner.run_served(" in generate
    module = inspect.getsource(srun)
    assert "os.environ.get(tc.INSPECT_API_KEY_ENV)" not in module
    helper = inspect.getsource(srun._inspect_api_key)
    assert "proxy_client_token(standup_endpoint)" in helper and "environ.get(tc.INSPECT_API_KEY_ENV)" in helper


def test_no_legs_child_env_gains_a_bearer_name_and_gsm8k_and_controls_carry_the_same_none(srv, draft, tmp_path):
    """The fix adds NOTHING to any child's env: through the real composition, no child of any leg (static, controls, misalign, bon, gsm8k --
    base and rung) carries a bearer under any name, and the GSM8K child's env has EXACTLY the controls child's variable set (names only) --
    the bearer's one source for both is the driver's rule over the standup record. Byte-identity beyond that: every non-placeholder argv
    token is the generated one, and the plan validates unchanged."""
    serve = _serve_dir(tmp_path)
    fake_subs = {lp.LEDGER_URL_PLACEHOLDER: "http://127.0.0.1:8/v1", lp.CHILD_ROUTING_TABLE_PLACEHOLDER: str(serve / "routing_table.json"),
                 lp.GSM8K_ATTEMPT_ID_PLACEHOLDER: "a1"}
    envs_by_leg: dict = {}
    n = 0
    for row in list(draft["standups"]) + list(draft["rungs"]):
        for k in row["children"]:
            if not k.get("argv"):
                continue                                                          # a Petri child is driven from the grading host (`grading_host_drive`), never spawned here
            if any(str(a).startswith("{pool_url") for a in k["argv"]):
                continue                                                          # a multi-turn child needs the pool URLs; not this defect's leg
            subs = {lp.child_key(k): fake_subs} if row["leg"] == "gsm8k" else None
            argv, env = _spawn(srv, k, row["standup_id"], root=tmp_path / "attempt", serve=serve, child_subs=subs)
            assert set(env) & _bearer_names(srv) == set(), (row["leg"], k.get("cell"), k.get("s"))
            assert all(g == r for g, r in zip(k["argv"], argv) if "{" not in g), (row["leg"], k["argv"], argv)
            envs_by_leg.setdefault(row["leg"], set(env))
            n += 1
    assert n > 0 and {"gsm8k", "controls"} <= set(envs_by_leg)
    assert envs_by_leg["gsm8k"] == envs_by_leg["controls"]
    assert lp.validate_plan(draft) == [] and draft["launch_plan_sha"] == lp.plan_sha(draft)
    assert not hasattr(lp, "INSPECT_API_KEY_ENV") and "TINYCAP" not in inspect.getsource(lp)
