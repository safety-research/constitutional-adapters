"""qwen-seedrep-native-battery fs3 (F-AQ): a GSM8K child's request-evidence attempt identity is ONE value.

The finding (both gsm8k rows failed on first exercise): every GSM8K child -- `none`, `prompted__*`, the fitted cells -- died at
`[served-cap] request evidence: the driver's request-evidence writer runs as attempt 'a1' but this client's extra names '<attempt id>'
(SISTEER_REQUEST_EVIDENCE_EXTRA) -- one run has one attempt id; refusing to record under two` (`request_evidence_client.adopt_attempt_id`).
The composer's GSM8K children carry `--attempt-id {gsm8k_attempt_id}` -> the per-store `a<k>`, which `gsm8k_child_up` mints into
`ladder_state.json` and starts the child's usage ledger with, while `resolve_spawn` stamped the CONTAINER's attempt id into
every child's `SISTEER_REQUEST_EVIDENCE_EXTRA`. The driver's `check_attempt_id_arg` admits `a<k>` alone (the ledger's parser too), so the
container's id can never ride the argv; dropping the flag re-fails `producer_problem` (test_seedrep_review_fixes pins both). The extra
therefore follows the argv (`seedrep_launch_plan.evidence_attempt_id`); every other leg carries no `--attempt-id` and keeps the container's id.

RE-VALIDATED first, through the REAL composer (`build_plan`), the REAL app child-argv/env composition (`gsm8k_child_up` -> `child_argv_env`)
and the REAL predicate (`RequestEvidenceClient.take_over_request_sink`, `inspect_served.request_evidence_sinks`) over PRODUCTION's
`_RequestEvidence` writer (`scripts/served_capability_run.py`). Synthetic ids throughout (a `ta-` container attempt id, hostname-free)."""
from __future__ import annotations

import importlib.util
import inspect
import json
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "modal"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import seedrep_launch_plan as lp  # noqa: E402
from seedrep_fixes_common import GSM8K_TASK, load_served_capability_run, wire  # noqa: E402
from sisteer.capeval.tiers import inspect_served as isv  # noqa: E402
from sisteer.capeval.tinycap import contracts as tc  # noqa: E402
from sisteer.eval import request_evidence_client as rev  # noqa: E402

DRIVER = REPO / "scripts" / "modal" / "seedrep_served.py"
W = "2" * 40
WAVE = "22222222-3333-4444-8444-555555555555"
CONTAINER_ATTEMPT = "ta-11111111-2222"          # the app's attempt id (`mint_attempt_id`: <task id>-<uuid4[:12]>), synthetic
PROXY = "http://127.0.0.1:9/v1"
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
    spec = importlib.util.spec_from_file_location("seedrep_served_w24_fs3", DRIVER)
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
    serve = tmp_path / "serve"
    serve.mkdir(parents=True)
    (serve / "routing_table.json").write_text(json.dumps({"routes": {}}))
    (serve / "endpoint.json").write_text(json.dumps({"proxy_url": PROXY, "routing_table": str(serve / "routing_table.json")}))
    return serve


def _spawn(srv, k, sid, *, root: Path, serve: Path, child_subs=None):
    """THE app's composition for one child: `child_argv_env` (= `lp.resolve_spawn` over the app's `SpawnContext`)."""
    return srv.child_argv_env(k, attempt_root=str(root), plans_root="/vol/p", proxy_url=PROXY, routing_table=str(serve / "routing_table.json"),
                              driver_python="/py", client_python="/cpy", S=31, ncells=8, bank_view="/b", base_env={}, attempt_id=CONTAINER_ATTEMPT,
                              standup_id=sid, child_subs=child_subs)


def _gsm8k_picks(draft):
    """The casualties by class: the `none` child, a `prompted__*` child and a fitted cell of the s1 BASE row; an Arditi base child; an s1 RUNG child."""
    base = [r for r in draft["standups"] if r["leg"] == "gsm8k" and r["engine"] == "s1"][0]
    picks = [(base, [k for k in base["children"] if k["cell"] == "none"][0]),
             (base, [k for k in base["children"] if k["cell"] in lp.PROMPTED_CELLS][0]),
             (base, [k for k in base["children"] if k["cell"] == lp.FITTED_CELLS[0]][0])]
    ard = [r for r in draft["standups"] if r["leg"] == "gsm8k" and r["engine"] == lp.ARDITI][0]
    picks.append((ard, ard["children"][0]))
    rung = [r for r in draft["rungs"] if r["leg"] == "gsm8k" and r["engine"] == "s1"][0]
    picks.append((rung, rung["children"][0]))
    return picks


def test_gsm8k_children_run_as_the_one_attempt_id_the_app_minted_and_the_lane_client_accepts(srv, draft, tmp_path, monkeypatch):
    """After: for every GSM8K child class, the resolved argv's `--attempt-id`, the extra's `attempt_id`, the ledger's `--attempt-id` and the id
    `gsm8k_child_up` minted are ONE value (`a1`); the REAL predicate accepts production's writer running as it, and the record it then writes is
    admitted by the driver's `--attempt-id` scoping. Before (re-validated): the old extra -- the container's id -- against the same argv is
    refused by name, `one run has one attempt id`."""
    srun = load_served_capability_run()
    monkeypatch.setattr(srv.os, "killpg", lambda pid, sig: None)
    for i, (row, k) in enumerate(_gsm8k_picks(draft)):
        root, serve = tmp_path / f"attempt{i}", _serve_dir(tmp_path / f"s{i}")
        _FakeLedgerPopen.started = []
        st = srv.gsm8k_child_up(k, root=root, serve_dir=serve, proxy_url=PROXY, driver_python="/py", tree=REPO, env={"PATH": "/usr/bin"},
                                popen=_FakeLedgerPopen, wait_s=5, sleep=lambda s: None)
        led = _FakeLedgerPopen.started[-1].argv
        argv, env = _spawn(srv, k, row["standup_id"], root=root, serve=serve, child_subs={lp.child_key(k): st["subs"]})
        own = lp._flag_value(argv, "--attempt-id")
        extra = json.loads(env[lp.REQUEST_EVIDENCE_EXTRA_ENV])
        # ONE value, from ONE minting point (the app's per-child ledger standup), on the argv, the ledger and the extra alike
        assert own == extra["attempt_id"] == st["attempt_id"] == led[led.index("--attempt-id") + 1] == st["subs"][lp.GSM8K_ATTEMPT_ID_PLACEHOLDER] == "a1", (k["cell"], k["s"])
        assert tc.check_attempt_id_arg(own) == own and lp.evidence_attempt_id(argv, CONTAINER_ATTEMPT) == own
        assert extra == {"leg": "gsm8k", "arm": k["arm"], "s": k["s"], "attempt_id": own, "standup_id": row["standup_id"]}
        assert env[lp.REQUEST_EVIDENCE_ENV] == lp._flag_value(argv, "--out-root") == str(root / k["out_rel"]) and env[lp.REQUEST_EVIDENCE_SEED_BASE_ENV] == str(k["s"])
        assert argv[argv.index("--attempt-id") + 2] == "--bank-view" and lp.unresolved_placeholders(argv) == []      # the line, untouched
        # the REAL predicate over the composed pair: the lane client built from THIS env, production's writer running as THIS argv's id
        client = rev.from_env(env)
        prod = srun._RequestEvidence(Path(env[lp.REQUEST_EVIDENCE_ENV]) / tc.REQUEST_EVIDENCE_FILENAME, attempt_id=tc.check_attempt_id_arg(own))
        assert client.take_over_request_sink(prod.record_wire) == client.record_request and client.attempt_id == own
    # the `none` child -- the one that passed the render certification and died here -- end to end through the provider's function
    row, k = _gsm8k_picks(draft)[0]
    root, serve = tmp_path / "attempt_none", _serve_dir(tmp_path / "s_none")
    st = srv.gsm8k_child_up(k, root=root, serve_dir=serve, proxy_url=PROXY, driver_python="/py", tree=REPO, env={}, popen=_FakeLedgerPopen, wait_s=5, sleep=lambda s: None)
    argv, env = _spawn(srv, k, row["standup_id"], root=root, serve=serve, child_subs={lp.child_key(k): st["subs"]})
    for var in lp.REQUEST_EVIDENCE_ENVS:
        monkeypatch.setenv(var, env[var])
    rev.reset_env_cache()
    prod = srun._RequestEvidence(Path(env[lp.REQUEST_EVIDENCE_ENV]) / tc.REQUEST_EVIDENCE_FILENAME, attempt_id=tc.check_attempt_id_arg(lp._flag_value(argv, "--attempt-id")))
    req, resp = isv.request_evidence_sinks(prod.record_wire, None)
    client = rev.from_env()
    assert req == client.record_request and resp == client.response_sink
    rec = req(*wire(GSM8K_TASK, "q1", 0))
    assert rec[tc.REQUEST_EVIDENCE_ATTEMPT_FIELD] == "a1" == rec["extra"]["attempt_id"] and rec["extra"]["s"] == k["s"] and rec["extra"]["leg"] == "gsm8k"
    assert prod.records() == [rec] and tc.records_of_attempts(prod.records(), {"a1"}) == [rec]           # the driver's --attempt-id scoping admits it
    # BEFORE, re-validated: the old composition -- the container's attempt id in the extra (the pre-fix `request_evidence_env(attempt_id=ctx.attempt_id)`)
    # against the argv's a1 -- is exactly the refusal both gsm8k rows died on
    old_env = {**env, **lp.request_evidence_env(evidence_dir=env[lp.REQUEST_EVIDENCE_ENV], attempt_id=CONTAINER_ATTEMPT, standup_id=row["standup_id"],
                                                eval_leg="gsm8k", arm=k["arm"], s=int(k["s"]))}
    assert json.loads(old_env[lp.REQUEST_EVIDENCE_EXTRA_ENV])["attempt_id"] == CONTAINER_ATTEMPT != lp._flag_value(argv, "--attempt-id")
    with pytest.raises(rev.RequestEvidenceError, match="one run has one attempt id; refusing to record under two"):
        rev.from_env(old_env).take_over_request_sink(prod.record_wire)
    # why the extra follows the argv and not the other way round: the driver (and the ledger) admit a<k> alone -- the container's id never parses
    with pytest.raises(tc.AttemptScopeError):
        tc.check_attempt_id_arg(CONTAINER_ATTEMPT)
    assert lp.evidence_attempt_id(["/py", "x.py", "generate", "--cell", "none"], CONTAINER_ATTEMPT) == CONTAINER_ATTEMPT
    assert "evidence_attempt_id(argv, ctx.attempt_id)" in inspect.getsource(lp.resolve_spawn)


def test_every_other_legs_argv_and_extra_are_byte_identical_to_before(srv, draft, tmp_path):
    """The other legs carry no `--attempt-id` -- their extra keeps the container's attempt id, their resolved argv adds no flag and every
    non-placeholder token is the generated one; the plan carries `--attempt-id` on GSM8K children ONLY (so the rule reaches nothing else)."""
    serve = _serve_dir(tmp_path)
    pins = [("pasttense", "s1", "base"), ("controls", "s1", "base"), ("misalign", "s1", "base")]      # a static, the controls, a misalign row
    for leg, engine, stage in pins:
        row = [r for r in draft["standups"] if r["leg"] == leg and r["engine"] == engine and r["stage"] == stage][0]
        k = row["children"][0]
        argv, env = _spawn(srv, k, row["standup_id"], root=tmp_path / "attempt", serve=serve)
        assert "--attempt-id" not in argv and lp._flag_value(argv, "--attempt-id") is None
        extra = json.loads(env[lp.REQUEST_EVIDENCE_EXTRA_ENV])
        assert extra == {"leg": k["eval_leg"], "arm": k["arm"], "s": k["s"], "attempt_id": CONTAINER_ATTEMPT, "standup_id": row["standup_id"]}
        assert lp.evidence_attempt_id(argv, CONTAINER_ATTEMPT) == CONTAINER_ATTEMPT
        assert len(argv) == len(k["argv"]) and [a for a in argv if a.startswith("--")] == [a for a in k["argv"] if a.startswith("--")]
        assert all(g == r for g, r in zip(k["argv"], argv) if "{" not in g), (leg, k["argv"], argv)
    assert all(("--attempt-id" in (k.get("argv") or [])) == (k["eval_leg"] == "gsm8k") for _sid, k in lp.plan_children(draft))
    assert lp.validate_plan(draft) == [] and draft["launch_plan_sha"] == lp.plan_sha(draft)
