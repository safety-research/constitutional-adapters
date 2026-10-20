"""Qwen seed-replicate native battery — integration F2 (the app ↔ check argv seam): the exact argv `scripts/modal/seedrep_served.py`
builds for the in-container determinism spot-check is parsed by `determinism_check.py`'s REAL parser (the module's `main`, never a
copy of its parser), runs to completion on a synthetic attempt root, and a non-zero rc is CONSULTED and recorded in the attempt's
evidence the way `producer_evidence` records the audit (sha256 / ABSENT by name + rc).

Known-bad, kept: the argv the app used to build (`--leg <container leg> --attempt-root … --routing-table … --out …`, no `--metrics-url`)
is rejected by the real parser with argparse's exit 2 — the spot-check never ran and `determinism.json` was never written.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "modal", REPO / "scripts" / "sdfbase" / "seedrep", REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import audit_common as ac  # noqa: E402
import determinism_check as dc  # noqa: E402
import seedrep_audit_fixtures as fx  # noqa: E402
import seedrep_launch_plan as lp  # noqa: E402

DRIVER = REPO / "scripts" / "modal" / "seedrep_served.py"


# --------------------------------------------------------------------------- the app under a stub `modal` (nothing launched)
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
    spec = importlib.util.spec_from_file_location("seedrep_served_f2_under_test", DRIVER)
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


class _Parsed(Exception):
    """Raised by the injected `read_attempt_root`: argparse ACCEPTED the argv and `main` reached its first read."""


def _real_parser_accepts(argv_after_python_and_script):
    """Run the check's REAL `main` on the argv; True when argparse accepted it (the injected first read fires), False on argparse's exit 2."""
    saved = ac.read_attempt_root

    def _first_read(path, **kw):
        raise _Parsed(str(path))
    ac.read_attempt_root = _first_read
    try:
        dc.main(list(argv_after_python_and_script))
    except _Parsed:
        return True
    except SystemExit as e:
        assert e.code == 2, e.code
        return False
    finally:
        ac.read_attempt_root = saved
    raise AssertionError("main returned without reading the attempt root")


def _faithful_post(raw: bytes, headers):
    body = json.loads(raw)
    item = body["messages"][0]["content"][len("prompt "):]
    return {"id": "chatcmpl-replay", "choices": [{"message": {"content": f"response for {item} under {body['model']}"}, "finish_reason": "stop"}]}


@pytest.fixture
def hooks(monkeypatch):
    monkeypatch.setattr(dc, "POST_FN", _faithful_post)
    monkeypatch.setattr(dc, "METRICS_FN", lambda: {"num_requests_running": 0, "num_requests_waiting": 0, "generation_tokens_total": 10})
    monkeypatch.setattr(dc, "TOKENIZE_FN", None)


def _row(leg: str, **more) -> dict:
    return {"leg": leg, "kind": "base", "engine": "s1", "stage": "base", "shard": 0, **more}


# --------------------------------------------------------------------------- the seam
def test_the_old_argv_is_rejected_by_the_real_parser_known_bad(tmp_path):
    """The argv seedrep_served.py built before F2 (integration_report F2): `--attempt-root` / `--routing-table` are not parser flags, `--root` is
    required, `--leg controls` is not a table leg, `--metrics-url` is missing — argparse exits 2 before anything runs."""
    old = ["--leg", "controls", "--attempt-root", str(tmp_path), "--proxy-url", "http://127.0.0.1:18080/v1",
           "--routing-table", str(tmp_path / "routing_table.json"), "--out", str(tmp_path / "determinism.json")]
    assert _real_parser_accepts(old) is False
    # and the single-leg spelling of the same old argv (a static container) fails on the unknown flags alone
    old_static = ["--leg", "pasttense", "--attempt-root", str(tmp_path), "--proxy-url", "u", "--routing-table", "t", "--out", str(tmp_path / "d.json")]
    assert _real_parser_accepts(old_static) is False


def test_every_container_legs_argv_is_accepted_by_the_real_parser(srv, tmp_path):
    """For EVERY container leg the plan stands up, each table leg's argv the app now builds parses (the `--leg` choices are
    `audit_common.LEGS`; the controls container's three legs are spelled as the check discovers their stores: `or_bench_hard_1k`)."""
    seen = []
    for container in lp.STANDUP_LEGS:
        legs = srv.determinism_table_legs(container)
        assert legs, container
        for leg in legs:
            assert leg in ac.LEGS, (container, leg)
            assert ac.CONTAINER_LEG_OF[leg] == container, (container, leg)
            argv = srv.determinism_check_argv(driver_python="/venv/bin/python", tree=REPO, table_leg=leg, root=tmp_path / "root",
                                              proxy_url="http://127.0.0.1:18080/v1", metrics_url="http://127.0.0.1:8000/metrics",
                                              out=tmp_path / "audits" / "determinism" / f"{leg}.json",
                                              copy_to=(tmp_path / "determinism.json") if len(legs) == 1 else None,
                                              petri_behaviours=["self_preservation"] if leg == "petri" else ())
            assert argv[0] == "/venv/bin/python" and argv[1] == str(REPO / srv.DETERMINISM_REL)
            assert "--attempt-root" not in argv and "--routing-table" not in argv and "--tokenize-url" not in argv
            assert argv[argv.index("--root") + 1] == str(tmp_path / "root") and argv[argv.index("--leg") + 1] == leg
            assert argv[argv.index("--metrics-url") + 1] == "http://127.0.0.1:8000/metrics"
            assert _real_parser_accepts(argv[2:]) is True, (container, leg, argv)
            seen.append(leg)
    assert sorted(seen) == sorted(ac.LEGS)                       # the 15 table legs, each exactly once across the containers
    assert srv.determinism_table_legs("controls") == ("xstest", "or_bench_hard_1k", "mmlu")
    assert srv.determinism_table_legs("misalign") == ("misalign_leaking", "misalign_murder")


def test_the_apps_argv_runs_the_real_check_to_completion_on_a_synthetic_root(srv, tmp_path, hooks):
    """End to end through the REAL `main`: the app's argv for a static (single-leg) container on a synthetic root writes the
    per-leg record AND the layout's `<root>/determinism.json` copy, rc 0, the record naming the leg."""
    root = fx.make_root(tmp_path / "stores", leg="pasttense", view="s1", arms=("none",))
    fx.populate_root(root, s=1, arms=("none",), n_items=25)
    ar = ac.read_attempt_root(root.path)
    rec_path = root.path / srv.DETERMINISM_COPY_REL / "pasttense.json"
    argv = srv.determinism_check_argv(driver_python=sys.executable, tree=REPO, table_leg="pasttense", root=root.path,
                                      proxy_url="http://127.0.0.1:18080/v1", metrics_url="http://127.0.0.1:8000/metrics",
                                      out=rec_path, copy_to=root.path / srv.DETERMINISM_OUT)
    rc = dc.main(argv[2:])
    assert rc == ac.EXIT_OK
    rec = json.loads(rec_path.read_text())
    assert rec["leg"] == "pasttense" and rec["attempt_id"] == ar.attempt_id and rec["n_items"] == 20 and rec["reproduces"] is True
    assert rec["token_basis"] == "whitespace"                    # declared, never a proxy /tokenize the check cannot call
    assert (root.path / srv.DETERMINISM_OUT).read_bytes() == rec_path.read_bytes()


# --------------------------------------------------------------------------- the rc is consulted and recorded
def test_a_failed_spot_check_is_recorded_absent_by_name_with_its_rc(srv, tmp_path):
    """The record mirrors producer_evidence's binding: a tool that exits 2 (REFUSED) leaves no file -> {absent, why, writer, tree_sha, rc: 2};
    one that writes its record -> {sha256, writer, tree_sha, rc: 0}. One entry per TABLE leg of the container; the controls container
    runs the check three times with `--leg xstest / or_bench_hard_1k / mmlu` and NO layout copy (three records cannot share one file)."""
    root = tmp_path / "attempt"
    (root / "children").mkdir(parents=True)
    calls = []

    def refusing_run(argv, **kw):
        calls.append(list(argv))
        return types.SimpleNamespace(returncode=2)

    out = srv.determinism_evidence(root, tree=REPO, W="f" * 40, row=_row("controls"), driver_python="/venv/bin/python", env={"PATH": "/usr/bin"},
                                   proxy_url="http://127.0.0.1:18080/v1", metrics_url="http://127.0.0.1:8000/metrics", run=refusing_run)
    assert sorted(out) == ["mmlu", "or_bench_hard_1k", "xstest"]
    for leg, rec in out.items():
        assert rec["absent"] is True and rec["rc"] == 2 and rec["writer"] == srv.DETERMINISM_REL and rec["tree_sha"] == "f" * 40
        assert "rc 2" in rec["why"] and rec["argv"][argv_index(rec["argv"], "--leg")] == leg
        assert "--copy-to" not in rec["argv"] and "copy" not in rec
    assert [c[c.index("--leg") + 1] for c in calls] == ["xstest", "or_bench_hard_1k", "mmlu"]
    assert all(c[c.index("--root") + 1] == str(root) for c in calls)
    assert (root / "children" / "determinism.log").is_file()      # the tool's argv + output land beside the children's logs

    # the writing case: the record is bound by sha256 and, for a single-leg container, the layout copy is named
    def writing_run(argv, **kw):
        outp = Path(argv[argv.index("--out") + 1]); outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps({"leg": argv[argv.index("--leg") + 1], "identical": 40, "n": 40}))
        if "--copy-to" in argv:
            Path(argv[argv.index("--copy-to") + 1]).write_bytes(outp.read_bytes())
        return types.SimpleNamespace(returncode=0)

    root2 = tmp_path / "attempt2"; (root2 / "children").mkdir(parents=True)
    out2 = srv.determinism_evidence(root2, tree=REPO, W="f" * 40, row=_row("pasttense"), driver_python="/venv/bin/python", env={}, proxy_url="u",
                                    metrics_url="m", run=writing_run)
    rec = out2["pasttense"]
    assert "absent" not in rec and rec["rc"] == 0 and rec["copy"] == srv.DETERMINISM_OUT
    assert rec["sha256"] == hashlib.sha256((root2 / srv.DETERMINISM_COPY_REL / "pasttense.json").read_bytes()).hexdigest()
    assert (root2 / srv.DETERMINISM_OUT).is_file()

    # a tree without the tool: ABSENT by name, rc None, no argv (nothing ran)
    out3 = srv.determinism_evidence(tmp_path / "attempt3", tree=tmp_path / "no-such-tree", W="f" * 40, row=_row("bon"), driver_python="py", env={},
                                    proxy_url="u", metrics_url="m", run=refusing_run)
    assert out3["bon"]["absent"] is True and out3["bon"]["rc"] is None and out3["bon"]["argv"] is None and "not on the tree" in out3["bon"]["why"]


def argv_index(argv, flag):
    return argv.index(flag) + 1


def test_the_hook_in_the_row_function_binds_the_record_under_modal_evidence(srv):
    """The app's row function calls `determinism_evidence` (its record under modal_evidence.json["determinism"]) and no longer spawns the
    check fire-and-forget: the old flags are gone from the source, the rc-consulting helper is the one call site."""
    src = DRIVER.read_text()
    assert '"--leg", row["leg"], "--attempt-root"' not in src and '"--routing-table", table, "--out"' not in src
    assert 'evidence["determinism"] = determinism_evidence(' in src
    assert src.count("determinism_check.py") >= 1 and "DETERMINISM_REL = \"scripts/sdfbase/seedrep/determinism_check.py\"" in src
