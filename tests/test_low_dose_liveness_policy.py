"""The low-dose numeric-liveness seam: the policy module, and that the two refusal branches consult it.

The whole safety of this change is that it can ONLY turn a refusal into a pass, and only when a numeric record for THIS
cell and THIS standup says so. So the tests are mostly refusals: unset env, missing file, wrong verdict, wrong cell,
another standup's routing table. The last two tests read the patched call sites as text — the branches they guard are
reached only with a live server, and a test that needed one would never run.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MOD = REPO / "scripts" / "sdfbase" / "full22" / "low_dose_liveness_policy.py"


def _load():
    spec = importlib.util.spec_from_file_location("low_dose_liveness_policy", MOD)
    m = importlib.util.module_from_spec(spec)
    sys.modules["low_dose_liveness_policy"] = m
    spec.loader.exec_module(m)
    return m


pol = _load()


def _rec(tmp_path, **over):
    rec = {"verdict": "live_numeric", "cell": "alphasteer", "routing_table": str(tmp_path / "routing_table.json"),
           "tag": "nemotron3nano4b_as_d0p3", "nominal_dose": 0.3, "basis": "numeric: …",
           "continuous_effect": {"diag": {"steered_divergence_median": 0.0421, "noise_ceiling": 0.0011, "bar": 0.0022}},
           "text_divergence_observation": {"n_probes": 5, "n_differing": 0, "rows": [{"prompt": "p"}]},
           "realised_dose": {"available": False}, "written_at": "2026-09-11T00:00:00+00:00"}
    rec.update(over)
    p = tmp_path / "liveness.json"
    p.write_text(json.dumps(rec))
    return p, rec


def test_unset_env_admits_nothing_and_says_so(tmp_path):
    rt = tmp_path / "routing_table.json"
    got, why = pol.consult("alphasteer", rt, env={})
    assert got is None and "is not set" in why and "unchanged" in why


def test_a_matching_record_admits_and_the_summary_carries_the_numbers(tmp_path):
    p, rec = _rec(tmp_path)
    got, why = pol.consult("alphasteer", tmp_path / "routing_table.json", env={pol.POLICY_ENV: str(p)})
    assert got == rec and "ADMITS alphasteer" in why and "OBSERVATION" in why
    s = pol.summary(got)
    assert s["policy"] == "D87_low_dose_numeric_liveness" and s["record_verdict"] == "live_numeric"
    assert s["steered_divergence_median"] == 0.0421 and s["noise_ceiling"] == 0.0011 and s["bar"] == 0.0022
    assert s["nominal_dose"] == 0.3 and s["text_divergence_observation"] == {"n_probes": 5, "n_differing": 0}
    assert s["realised_dose"] is False          # recorded as unavailable, never invented


@pytest.mark.parametrize("over,needle", [
    ({"verdict": "not_live"}, "verdict"),
    ({"verdict": "unevaluable"}, "verdict"),
    ({"cell": "constv2-mix-full__lora_r4_ce"}, "names cell"),
    ({"routing_table": "/somewhere/else/routing_table.json"}, "another standup"),
])
def test_every_mismatch_admits_nothing(tmp_path, over, needle):
    p, _ = _rec(tmp_path, **over)
    got, why = pol.consult("alphasteer", tmp_path / "routing_table.json", env={pol.POLICY_ENV: str(p)})
    assert got is None and needle in why


def test_a_missing_or_malformed_record_admits_nothing(tmp_path):
    got, why = pol.consult("alphasteer", tmp_path / "routing_table.json", env={pol.POLICY_ENV: str(tmp_path / "nope.json")})
    assert got is None and "does not exist" in why
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    got, why = pol.consult("alphasteer", tmp_path / "routing_table.json", env={pol.POLICY_ENV: str(bad)})
    assert got is None and "unreadable" in why
    lst = tmp_path / "list.json"
    lst.write_text("[]")
    got, why = pol.consult("alphasteer", tmp_path / "routing_table.json", env={pol.POLICY_ENV: str(lst)})
    assert got is None and "not a JSON object" in why


def test_the_driver_consults_the_policy_before_its_text_divergence_abort_and_records_it():
    t = (REPO / "scripts" / "served_capability_run.py").read_text()
    assert "import low_dose_liveness_policy as _low_dose_liveness" in t
    consult = t.index("_low_dose_liveness.consult(cell.id, args.routing_table)")
    abort = t.index("POSITIVE CONTROL FAILED for")
    assert consult < abort, "the policy must be consulted BEFORE the abort it can lift"
    assert t.index('evidence["steering_bites"]["low_dose_liveness"]') < abort
    assert 'evidence["steering_bites"]["text_divergence_is_observation"] = True' in t
    assert "bites_error = None" in t[consult:abort]
    # the abort itself, and the logprob abort, are untouched
    assert "an inert serve; nothing was generated." in t and "LOGLIKELIHOOD POSITIVE CONTROL FAILED" in t


def test_the_smoke_consults_the_policy_on_its_steering_bites_failure_only():
    t = (REPO / "scripts" / "sdfbase" / "full22" / "smoke_positive_control.py").read_text()
    assert "import low_dose_liveness_policy as _low_dose_liveness" in t
    assert '_low_dose_liveness.consult(args.cell, args.routing_table)' in t
    assert 'record("steering_bites", True, twin=twin.virtual_model, liveness_basis="numeric (D87)"' in t
    assert 'record("steering_bites", False, twin=twin.virtual_model, error=repr(e)[:800], policy=_ld_why)' in t
    # the other checks keep their own verdicts: the seam is scoped to the text-divergence probe
    for other in ("proxy_identity_mode", "salt_isolation", "determinism", "noise_floor", "intervention_hash_distinct"):
        assert f'record("{other}"' in t
    assert t.count("_low_dose_liveness.consult") == 1


def test_nothing_under_src_serving_moved_so_serving_impl_commit_stays_true():
    """The manifests pin serving_impl_commit = the last commit touching src/sisteer/capeval/serving/*.py +
    tiers/petri_runner.py. This change lives in scripts/ precisely so that pin is not made a lie (the
    --max-logprobs fix made the same choice)."""
    # On main the diff against the wave sha is every lane's work, so the invariant is stated on THIS change's own files:
    # the policy module and every hook that consults it live under scripts/, and the module imports nothing that would
    # make it part of the serving stack (it reads records; it never touches the engine).
    files = ["scripts/sdfbase/full22/low_dose_liveness_policy.py", "scripts/sdfbase/full22/sampled/lora_liveness.py",
             "scripts/sdfbase/full22/smoke_positive_control.py", "scripts/jailbreak_run.py",
             "scripts/sdfbase/full22/sampled/escalate_static.py", "scripts/served_capability_run.py"]
    for f in files:
        assert (REPO / f).is_file(), f
        assert not f.startswith("src/"), f
    # the module READS the serving stack's own instrument (fidelity's logprob bites check) and never patches it: no
    # assignment into an imported serving module, no engine import — a consumer of the stack, not a part of it
    import re
    policy = (REPO / files[0]).read_text()
    assert "import vllm" not in policy
    assert not re.search(r"^\s*_fid\.[A-Za-z_]+\s*=", policy, re.M), "the policy module must not monkeypatch the serving stack"
    assert not re.search(r"^\s*(sisteer|fidelity|geometry)\.[A-Za-z_.]+\s*=", policy, re.M)


# --------------------------------------------------------------------------- #
# The STATIC path: SteeringBitesGuard around bon_served.assert_steering_bites  #
# (jailbreak_run.py generate, escalate_static.py). CPU only, no server.        #
# --------------------------------------------------------------------------- #

import dataclasses  # noqa: E402
from types import SimpleNamespace  # noqa: E402

sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from sisteer.capeval.serving.fidelity import FidelityError  # noqa: E402
from sisteer.capeval.serving.proxy import RouteEntry  # noqa: E402
from sisteer.eval.attacks import bon_served as bs  # noqa: E402
from sisteer.eval.attacks import served_target as st  # noqa: E402

LENS, PASS = "proxy_vllm+lens+interleaved", "proxy_vllm+passthrough+interleaved"
CELL = "constv2-mix-full__lora_r4_ce_minus_ctl"                 # the cell id the launcher passes (--defense / --cell)
HID = "si_constv2_mix_full__lora_r4_ce_minus_ctl"              # its harness route id
ARM = "lora_r4_ce_minus_ctl"
VM, VM_TWIN, VM_ANCHOR = f"petri__{HID}", "petri__none__lens0", "petri__base"


def _routes():
    return [RouteEntry(virtual_model=VM_ANCHOR, cell_id="none", cell_class="base", serving_identity=PASS,
                       intervention_hash="aa" * 8),
            RouteEntry(virtual_model=VM_TWIN, cell_id="none__lens0", cell_class="control", serving_identity=LENS,
                       intervention_hash="bb" * 8, control_of="none"),
            RouteEntry(virtual_model=VM, cell_id=HID, cell_class="class_a", serving_identity=LENS, intervention_hash="cc" * 8)]


def _write_table(path, routes=None):
    path.write_text(json.dumps({"version": 1, "base_model": "org/model", "serving_mode": "interleaved",
                                "exact_replay_strategy": None, "base_revision": "r",
                                "routes": [dataclasses.asdict(r) for r in (routes or _routes())], "prompted_prefixes": []}))
    return path


def _probe(steered="same", twin="same"):
    """A bon_served.SteeringProbe: five byte-id rows per leg, as served_target.probe_fn builds them."""
    return lambda arm: ([list(steered.encode()) for _ in range(5)], [list(twin.encode()) for _ in range(5)])


def _admitting_record(tmp_path, table, cell=CELL, **over):
    rec = {"verdict": "live_numeric", "cell": cell, "routing_table": str(Path(table).resolve()),
           "tag": "llama8b_it_d0p1", "nominal_dose": 0.1, "basis": "numeric (D87): …",
           "continuous_effect": {"diag": {"steered_divergence_median": 0.0421, "noise_ceiling": 0.0011, "bar": 0.0022}},
           "text_divergence_observation": {"n_probes": 5, "n_differing": 0}, "realised_dose": {"available": False},
           "written_at": "2026-09-13T00:00:00+00:00"}
    rec.update(over)
    p = tmp_path / "liveness_low_dose.json"
    p.write_text(json.dumps(rec))
    return p


def test_guard_installs_nothing_with_the_env_unset(tmp_path):
    original = bs.assert_steering_bites
    logs = []
    g = pol.SteeringBitesGuard(CELL, tmp_path / "rt.json", env={}, log=logs.append)
    assert g.active() is False
    with g.installed(bs):
        assert bs.assert_steering_bites is original and g.installed_now is False
        with pytest.raises(FidelityError, match="steering_bites: arm"):
            bs.assert_steering_bites((ARM,), _probe())            # the real refusal, untouched
    assert g.admission is None and logs == []
    g.fold_into(ev := {"steering_bites": {"verified": False}})
    assert ev == {"steering_bites": {"verified": False}}          # nothing folded: byte-identical evidence


def test_guard_lifts_the_text_divergence_refusal_only_when_a_record_admits_this_cell_and_table(tmp_path):
    table = _write_table(tmp_path / "routing_table.json")
    rec_path = _admitting_record(tmp_path, table)
    env = {pol.POLICY_ENV: str(rec_path)}
    original = bs.assert_steering_bites
    logs = []
    g = pol.SteeringBitesGuard(CELL, table, env=env, log=logs.append)
    with g.installed(bs):
        assert bs.assert_steering_bites is not original and g.installed_now is True
        # 0/5 differing -> the real instrument refuses -> the guard consults, is admitted, returns the instrument's own
        # per-arm diagnostic (what served_evidence records for a passing probe)
        diag = bs.assert_steering_bites((ARM,), _probe())
    assert bs.assert_steering_bites is original and g.installed_now is False   # restored
    assert diag == {"arms": {ARM: {"n": 5, "n_differing": 0, "first_divergences": []}}}
    a = g.admission
    assert a["text_divergence_is_observation"] is True and a["liveness_basis"] == "numeric (D87)"
    assert a["text_divergence_refusal"].startswith("steering_bites: arm(s) ['lora_r4_ce_minus_ctl'] produce NO difference")
    assert a["low_dose_liveness"]["policy"] == "D87_low_dose_numeric_liveness"
    assert a["low_dose_liveness"]["steered_divergence_median"] == 0.0421 and a["low_dose_liveness"]["bar"] == 0.0022
    assert "ADMITS " + CELL in a["policy"]
    assert len(logs) == 1 and logs[0].startswith("LOW-DOSE LIVENESS (D87)")
    ev = {"steering_bites": {"applicable": True, "verified": True, "n_differing": 0}}
    g.fold_into(ev)
    assert ev["steering_bites"]["text_divergence_is_observation"] is True and ev["steering_bites"]["n_differing"] == 0
    assert ev["steering_bites"]["low_dose_liveness"]["record_verdict"] == "live_numeric"
    # a probe that DOES diverge passes straight through the original: no consultation, no admission
    g2 = pol.SteeringBitesGuard(CELL, table, env=env, log=logs.append)
    with g2.installed(bs):
        d2 = bs.assert_steering_bites((ARM,), _probe(steered="steered"))
    assert d2["arms"][ARM]["n_differing"] == 5 and g2.admission is None and len(logs) == 1


@pytest.mark.parametrize("over,needle", [({"cell": "constv2-mix-full__lora_r4_ce"}, "names cell"),
                                         ({"verdict": "not_live"}, "verdict"),
                                         ({"routing_table": "/elsewhere/routing_table.json"}, "another standup")])
def test_guard_reraises_the_same_refusal_when_the_record_admits_nothing(tmp_path, over, needle):
    table = _write_table(tmp_path / "routing_table.json")
    env = {pol.POLICY_ENV: str(_admitting_record(tmp_path, table, **over))}
    logs = []
    g = pol.SteeringBitesGuard(CELL, table, env=env, log=logs.append)
    with g.installed(bs), pytest.raises(FidelityError) as ei:
        bs.assert_steering_bites((ARM,), _probe())
    assert str(ei.value).startswith("steering_bites: arm(s)") and g.admission is None
    assert len(logs) == 1 and logs[0].startswith("text-divergence refusal stands:") and needle in logs[0]


def test_guard_touches_no_other_exception(tmp_path):
    table = _write_table(tmp_path / "routing_table.json")
    env = {pol.POLICY_ENV: str(_admitting_record(tmp_path, table))}
    logs = []
    fake_bs = SimpleNamespace(assert_steering_bites=None)
    for exc in (FidelityError("engine_determinism: repeat differs"), FidelityError("steering_bites arm x: paired lists of unequal length"),
                bs.BonServedError("assert_steering_bites: no arms to probe"), RuntimeError("connection reset")):
        def raising(arms, probe, _e=exc):
            probe(arms[0])
            raise _e
        fake_bs.assert_steering_bites = raising
        g = pol.SteeringBitesGuard(CELL, table, env=env, log=logs.append)
        with g.installed(fake_bs), pytest.raises(type(exc)) as ei:
            fake_bs.assert_steering_bites((ARM,), _probe())
        assert ei.value is exc and g.admission is None
    assert fake_bs.assert_steering_bites is raising and logs == []
    # a `steering_bites:` refusal whose probe evidence cannot be re-read as a diagnostic (the original never ran the
    # probe) is NOT lifted either: numeric liveness lifts the text-divergence refusal, not a broken probe
    def raising_without_probe(arms, probe):
        raise FidelityError("steering_bites: zero arms probed — an empty probe set proves nothing")
    fake_bs.assert_steering_bites = raising_without_probe
    g = pol.SteeringBitesGuard(CELL, table, env=env, log=logs.append)
    with g.installed(fake_bs), pytest.raises(FidelityError, match="zero arms"):
        fake_bs.assert_steering_bites((ARM,), _probe())
    assert g.admission is None and len(logs) == 1 and "could not be re-read" in logs[0]


class _StubChat:
    def __init__(self, replies):
        self.replies, self.payloads = replies, []

    def __call__(self, payload):
        self.payloads.append(dict(payload))
        text = self.replies.get(payload["model"], f"reply from {payload['model']}")
        return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}], "petri_intervention_hash": "cc" * 8,
                "usage": {"prompt_tokens": 10, "completion_tokens": 2}}


def _fake_client(chat, routes):
    class FakeClient:
        def __init__(self, base_url, *, identity):
            self.base_url = base_url.rstrip("/") + ("" if base_url.endswith("/v1") else "/v1")
            self.identity, self.served_models, self.proxy_identity = identity, (), {"impl": "fake"}
            self.chat_completions = chat

        def load(self, *, require_virtual_models=()):
            self.served_models = tuple(r.virtual_model for r in routes)
    return FakeClient


def test_build_served_target_end_to_end_through_the_guard(monkeypatch, tmp_path):
    """The REAL build_served_target -> served_evidence -> bs.assert_steering_bites -> fidelity._check_steering_bites chain,
    with a stub transport whose steered and twin legs answer identically (the low-dose case): refused as before with the
    env unset; built, with the probe recorded as an observation, when a record admits this cell on this table."""
    routes = _routes()
    (tmp_path / "standup").mkdir()
    table = _write_table(tmp_path / "standup" / "routing_table.json")
    chat = _StubChat({VM: "same", VM_TWIN: "same"})
    monkeypatch.setattr(st, "ProxyClientEngine", _fake_client(chat, routes))
    build = lambda: st.build_served_target(proxy_url="http://127.0.0.1:9/v1", routing_table=str(table), cell_id=CELL)  # noqa: E731
    # env unset: the guard is inert and the real refusal stands
    monkeypatch.delenv(pol.POLICY_ENV, raising=False)
    g0 = pol.SteeringBitesGuard(CELL, str(table))
    with g0.installed(), pytest.raises(FidelityError, match="steering_bites: arm"):
        build()
    assert g0.admission is None
    # a record for this cell on THIS table: built, evidence verified, the text probe an observation
    rec_path = _admitting_record(tmp_path, table)
    monkeypatch.setenv(pol.POLICY_ENV, str(rec_path))
    logs = []
    g = pol.SteeringBitesGuard(CELL, str(table), log=logs.append)
    with g.installed():
        t = build()
    g.fold_into(t.evidence)
    sb = t.evidence["steering_bites"]
    assert sb["applicable"] is True and sb["verified"] is True and sb["n_probes"] == 5 and sb["n_differing"] == 0
    assert sb["arm"] == ARM and sb["twin"] == "none__lens0" and sb["first_divergences"] == []
    assert sb["text_divergence_is_observation"] is True and sb["liveness_basis"] == "numeric (D87)"
    assert sb["low_dose_liveness"]["record_verdict"] == "live_numeric" and sb["low_dose_liveness"]["nominal_dose"] == 0.1
    assert sb["text_divergence_refusal"].startswith("steering_bites: arm(s)")
    assert t.evidence["salt_isolation"]["ok"] is True                    # the other control ran and stands on its own
    doc = json.loads(st.write_target_manifest(tmp_path / "out", t).read_text())
    assert doc["evidence"]["steering_bites"]["text_divergence_is_observation"] is True
    assert doc["evidence"]["steering_bites"]["low_dose_liveness"]["bar"] == 0.0022
    assert len(logs) == 1 and "LOW-DOSE LIVENESS (D87)" in logs[0]
    # the same record admits nothing on ANOTHER standup's table: the refusal stands there
    (tmp_path / "other").mkdir()
    other = _write_table(tmp_path / "other" / "routing_table.json")
    g2 = pol.SteeringBitesGuard(CELL, str(other))
    with g2.installed(), pytest.raises(FidelityError, match="steering_bites: arm"):
        st.build_served_target(proxy_url="http://127.0.0.1:9/v1", routing_table=str(other), cell_id=CELL)
    # a missing twin is a DIFFERENT refusal (ServedTargetError) and passes through the guard untouched
    no_twin = _write_table(tmp_path / "no_twin.json", [r for r in routes if r.cell_id != "none__lens0"])
    g3 = pol.SteeringBitesGuard(CELL, str(no_twin))
    _admitting_record(tmp_path, no_twin)
    with g3.installed(), pytest.raises(st.ServedTargetError, match="no scale-0 twin"):
        st.build_served_target(proxy_url="http://127.0.0.1:9/v1", routing_table=str(no_twin), cell_id=CELL)
    assert bs.assert_steering_bites.__name__ == "assert_steering_bites"   # restored after every context


def test_jailbreak_run_generate_builds_the_served_target_through_the_guard(monkeypatch, tmp_path):
    """The static driver end to end (cmd_generate --target-engine served) on the low-dose case: served_target.json beside
    the store carries the observation and the numeric summary when admitted; the driver dies as before when not."""
    import jailbreak_run
    import sisteer.models as models_mod

    routes = _routes()
    (tmp_path / "standup").mkdir()
    table = _write_table(tmp_path / "standup" / "routing_table.json")
    chat = _StubChat({VM: "same", VM_TWIN: "same"})
    monkeypatch.setattr(st, "ProxyClientEngine", _fake_client(chat, routes))
    monkeypatch.setattr(models_mod, "load_model_and_tokenizer",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no HF load on the served path")))
    monkeypatch.setattr(jailbreak_run, "_valid_defense_ids", lambda cfg: {"none", CELL})
    goals = [SimpleNamespace(goal_id=f"g{i}", text=f"goal {i}", category="c") for i in range(3)]
    monkeypatch.setattr(jailbreak_run, "_eval_goals", lambda cfg, *, confirmatory: goals)
    monkeypatch.setattr(jailbreak_run, "_build_attacker_seams",
                        lambda cfg, args, log_dir=None: ((lambda p: "5"), (lambda lock: None), (lambda c: "x")))
    ran = {}

    def fake_run(attack, defense, goals_arg, **kw):
        ran.update(defense=defense, out_dir=Path(kw["out_dir"]))
        return {"n_goals_run": len(goals_arg), "n_goals_total": len(goals_arg), "n_target_forwards_total": len(goals_arg)}   # fw
    monkeypatch.setattr(jailbreak_run, "_run_gated_attack", fake_run)
    # synthetic model "m" has no vendor-preset row (the 2026-09-16 campaign default refuses one); the decode
    # is STATED so this test keeps exercising the liveness policy, not the decode default
    cfg = {"model": "m", "dtype": "bfloat16", "private_run": "jb", "attacker": {"model": "qwen"},
           "attacks": {"forward_cap": 150, "target_max_new_tokens": 2048, "target_decode": "generation_config"}}
    args = SimpleNamespace(attack="crescendo", defense=CELL, shard=0, nshards=1, xstest=False, confirmatory=True,
                           trial=0, private_root=str(tmp_path / "state"), target_engine="served",
                           target_proxy_url="http://127.0.0.1:7/v1", target_routing_table=str(table),
                           positions="response", scale=1.0, goal_ids_file=None, expect_pair_n_streams=None,
                           attacker_engine="vllm", attacker_endpoint="http://127.0.0.1:8000")
    # env unset: the text-divergence refusal kills the driver exactly as on f09206a3
    monkeypatch.delenv(pol.POLICY_ENV, raising=False)
    with pytest.raises(FidelityError, match="steering_bites: arm"):
        jailbreak_run.cmd_generate(args, cfg)
    assert not ran
    # the launcher's per-cell env names this cell's record for this standup: the driver builds and generates
    monkeypatch.setenv(pol.POLICY_ENV, str(_admitting_record(tmp_path, table)))
    jailbreak_run.cmd_generate(args, cfg)
    assert ran["out_dir"].as_posix().endswith(f"jb/crescendo/{CELL}/shard00")
    doc = json.loads((ran["out_dir"] / "served_target.json").read_text())
    sb = doc["evidence"]["steering_bites"]
    assert sb["verified"] is True and sb["n_differing"] == 0 and sb["text_divergence_is_observation"] is True
    assert sb["liveness_basis"] == "numeric (D87)" and sb["low_dose_liveness"]["record_verdict"] == "live_numeric"
    assert doc["manifest_stage"] == "final" and doc["harness_cell_id"] == HID and doc["tt_cell_id"] == CELL
    assert bs.assert_steering_bites.__name__ == "assert_steering_bites"
    # a record for the SAME cell but another standup's table admits nothing: the driver dies as before
    (tmp_path / "other").mkdir()
    other = _write_table(tmp_path / "other" / "routing_table.json")
    args2 = SimpleNamespace(**{**vars(args), "target_routing_table": str(other)})
    with pytest.raises(FidelityError, match="steering_bites: arm"):
        jailbreak_run.cmd_generate(args2, cfg)


def test_the_static_entry_points_install_the_guard_around_build_served_target():
    for rel, cell_expr, table_expr, prefix in (
            ("scripts/jailbreak_run.py", "args.defense", "args.target_routing_table", "[target]"),
            ("scripts/sdfbase/full22/sampled/escalate_static.py", "args.defense", "args.routing_table", "[escalate-static]")):
        t = (REPO / rel).read_text()
        assert "import low_dose_liveness_policy as _low_dose_liveness" in t, rel
        assert t.count("_low_dose_liveness.SteeringBitesGuard(") == 1, rel
        i_guard = t.index("_low_dose_liveness.SteeringBitesGuard(")
        assert cell_expr in t[i_guard:i_guard + 200] and table_expr in t[i_guard:i_guard + 200], rel
        i_with = t.index("with _ld_guard.installed():")
        i_build = t.index("build_served_target(", i_with)
        i_fold = t.index("_ld_guard.fold_into(", i_build)
        assert i_guard < i_with < i_build < i_fold, rel
        # the ONE build_served_target call each entry point makes sits inside the context
        calls = [m.start() for m in re.finditer(r"(?<![\w.])build_served_target\(", t)]
        assert len(calls) == 1 and calls[0] == i_build, (rel, calls)
        assert f'print(f"{prefix} {{m}}"' in t, rel
    # the module-level import rides the scripts/ path, never a src/ move (serving_impl_commit stays true)
    t = (REPO / "scripts/jailbreak_run.py").read_text()
    assert 'sys.path.insert(0, str(Path(__file__).resolve().parent / "sdfbase" / "full22"))' in t
    t = (REPO / "scripts/sdfbase/full22/sampled/escalate_static.py").read_text()
    assert 'REPO / "scripts" / "sdfbase" / "full22"' in t
