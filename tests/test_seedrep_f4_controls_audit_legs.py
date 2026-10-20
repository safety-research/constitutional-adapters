"""Qwen seed-replicate native battery — integration F4 (`producer_evidence` / certificate ↔ `request_evidence_audit`, controls container):
the app passed `--legs xstest,or_bench_hard,mmlu` (EVAL_LEGS_OF's registry eval id) to an audit whose parser only knows the TABLE legs
(`or_bench_hard_1k`) → exit 2, no controls verdict at all; and the certificate bound the controls child to ONE `audits/request_log/controls/…`
file nobody writes, while the audit writes one verdict per TABLE leg (`audit_root`: `<out>/request_log/<table leg>/<arm>__s<s>…json`, verdict
`leg` = the table leg). Fixed at the callers: `producer_evidence` composes the table legs and binds one verdict per (child, table leg) through
`lp.audit_verdict_rels`; `smoke_evidence_paths` / the certificate's two check loops read the same aligned sources (`lp.smoke_audit_sources`)
and hold each verdict's `leg` to the TABLE leg. The audit is untouched (its output is the authority).

Known-bad: the old `--legs` value is refused by the REAL audit parser (exit 2) and every value the app now composes is accepted; after a
synthetic audit run every path the certificate expects for a controls row EXISTS.
"""
from __future__ import annotations

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
import request_evidence_audit as rea  # noqa: E402
import seedrep_audit_fixtures as fx  # noqa: E402
import seedrep_launch_plan as lp  # noqa: E402

DRIVER = REPO / "scripts" / "modal" / "seedrep_served.py"


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
    spec = importlib.util.spec_from_file_location("seedrep_served_f4_under_test", DRIVER)
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
    """argparse accepted the argv and `main` reached its `--legs` vocabulary check."""


def _audit_parser_accepts_legs(legs: str) -> bool:
    """The REAL audit `main` on `--roots <x> --out <y> --legs <legs>`: True when it gets past the legs vocabulary (the first refusal after
    parsing names the roots), False when `--legs` itself is refused (exit 2 'unknown leg(s)')."""
    try:
        rea.main(["--roots", "/nonexistent-root", "--out", "/nonexistent-out", "--legs", legs] + (["--petri-behaviours", "b1"] if "petri" in legs else []))
    except ac.AuditRefusal as e:
        if "unknown leg" in str(e):
            return False
        return True                                              # refused later (no such root): the legs were accepted
    except SystemExit as e:
        assert e.code == 2, e.code
        return False
    raise AssertionError("the audit ran on a nonexistent root")


def _row(leg: str, children):
    return {"leg": leg, "kind": "base", "engine": "s1", "stage": "base", "shard": 0, "children": children}


def test_the_old_controls_legs_value_is_refused_by_the_real_audit_and_every_composed_value_is_accepted(srv):
    old = ",".join(lp.EVAL_LEGS_OF["controls"])
    assert old == "xstest,or_bench_hard,mmlu" and _audit_parser_accepts_legs(old) is False      # F4 reproduced: exit 2 'unknown leg(s)'
    for container in lp.STANDUP_LEGS:
        legs = ",".join(srv.determinism_table_legs(container))                                  # what producer_evidence now composes
        assert _audit_parser_accepts_legs(legs) is True, (container, legs)
    assert srv.determinism_table_legs("controls") == lp.CONTROLS_ALL == ("xstest", "or_bench_hard_1k", "mmlu")


def test_the_controls_child_is_bound_by_three_per_table_leg_verdicts_never_a_controls_file():
    rels = lp.audit_verdict_rels("controls", "none", 1, stage="base", shard=0)
    assert rels == [("xstest", "audits/request_log/xstest/none__s1.json"), ("or_bench_hard_1k", "audits/request_log/or_bench_hard_1k/none__s1.json"),
                    ("mmlu", "audits/request_log/mmlu/none__s1.json")]
    assert lp.audit_table_legs("controls") == lp.CONTROLS_ALL and lp.audit_table_legs("misalign_leaking") == ("misalign_leaking",)
    # every other child: one verdict, the eval leg IS the table leg, the spelling of audit_verdict_rel unchanged
    assert lp.audit_verdict_rels("gsm8k", "none", 2, stage="16384", shard=0) == [("gsm8k", "audits/request_log/gsm8k/none__s2__cap16384.json")]
    assert lp.audit_verdict_rels("bon", "none", 1, stage="base", shard=3) == [("bon", "audits/request_log/bon/none__s1__shard03.json")]
    row = _row("controls", [{"eval_leg": "controls", "arm": "none", "s": 1}, {"eval_leg": "controls", "arm": "prompted__si_constitution", "s": 1}])
    paths = lp.smoke_evidence_paths(row, "/vol/root")["request_evidence_audit"]
    assert len(paths) == 6 and not any("/request_log/controls/" in p for p in paths)
    src = lp.smoke_audit_sources(row, "/vol/root")
    assert [(p.rsplit("/", 2)[-2], k["arm"], leg) for p, k, leg in src] == [(leg, k["arm"], leg) for k in row["children"] for leg in lp.CONTROLS_ALL]
    assert paths == [p for p, _k, _l in src]                       # the paths and the aligned sources are the same list
    # every table leg the certificate names is one the audit's parser knows
    assert all(leg in ac.LEGS for _p, _k, leg in src)


def test_after_a_real_audit_run_every_path_the_certificate_expects_for_a_controls_row_exists(srv, tmp_path):
    """The REAL audit over a synthetic controls root with the app's `--legs`, then producer_evidence's bindings and the
    certificate's paths: every expected file exists and its `leg` is the table leg the binding names."""
    root = fx.make_root(tmp_path / "stores", leg="controls", view="s1", arms=("none",))
    fx.populate_root(root, s=1, arms=("none",), n_items=3)
    row = _row("controls", [{"eval_leg": "controls", "arm": "none", "s": 1}])
    expected = lp.smoke_evidence_paths(row, str(root.path))["request_evidence_audit"]
    assert len(expected) == 3 and not any(Path(p).exists() for p in expected)
    # producer_evidence runs the REAL audit tool (a subprocess through _run_tool, the app's own argv) -- the ONE run (the audit refuses a second
    # run over an existing audits/request_log tree, so nothing pre-runs it here)
    ev = srv.producer_evidence(root.path, tree=REPO, W=fx.W_FIXTURE, row=row, serve_dir=root.path / "exp" / "serve", driver_python=sys.executable,
                               env={"PATH": "/usr/bin:/bin", "PYTHONPATH": f"{REPO / 'src'}:{REPO / 'scripts'}"})
    log = (root.path / "children" / "request_evidence_audit.log").read_text()
    assert f"--legs {','.join(lp.CONTROLS_ALL)}" in log and "REFUSED:" not in log and "3 AGREES, 0 REFUSED" in log, log[-600:]
    for path, child, table_leg in lp.smoke_audit_sources(row, str(root.path)):
        assert Path(path).is_file(), path
        doc = json.loads(Path(path).read_text())
        assert doc["leg"] == table_leg and doc["arm"] == child["arm"] and int(doc["s"]) == child["s"] and doc["attempt_id"] == ac.read_attempt_root(root.path).attempt_id
        assert all(k in doc for k in lp.AUDIT_VERDICT_KEYS), sorted(set(lp.AUDIT_VERDICT_KEYS) - set(doc))
    # producer_evidence binds exactly those three (beside the recorder's record and cells.json), by sha256, rc 0
    bound = {rel: rec for rel, rec in ev.items() if rel.startswith("audits/request_log/")}
    assert sorted(bound) == sorted(rel for _leg, rel in lp.audit_verdict_rels("controls", "none", 1, stage="base", shard=0))
    assert all("sha256" in rec and rec["rc"] == 0 and rec["writer"] == lp.PRODUCER_WRITERS["request_evidence_audit"] for rec in bound.values()), bound
    assert "audits/request_log/controls/none__s1.json" not in ev
    # the OLD binding (one controls file) is what nobody writes
    assert not (root.path / "audits" / "request_log" / "controls").exists()
