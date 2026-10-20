"""The served geometry on Modal is derived from the KV capacity the container
actually reports. The per-model budgets (``geometry.BUDGET_BY_MODEL``) were measured on
the cluster's H200; a Modal H200 container reported 782,688 KV tokens at L=16,384 for
glm46vflash against the 786,432-token budget (48 x 16,384) and readiness refused
(correctly). Now: (1) ``vllm_server.validate_scheduler_reports`` raises the
``KvCoverageShortfall`` SUBCLASS when — and only when — every pin held and every problem
is a KV shortfall, carrying the numbers; (2) the standup (``--kv-derived-seqs on``)
re-launches ONCE at S = floor(kv_backed_concurrency), same window, records the shrink
(``geometry.kv_shrink``) and serves that geometry; (3) the launcher composes the flag on
every standup, reads the record back through ONE fail-closed validator, re-derives the
expectation / REC geometry / ``--concurrency`` at the served S (which is store
identity) and trusts a shrunk store before the standup only through the same
validator. The invariant still refuses a clamp / threshold / speculative config — a
shortfall beside one of those is the plain refusal, nothing is derived from it.
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_tinycap_launcher as TL  # noqa: E402
from test_petri_serving_gate import _FAKE_EXPECTED, _sched_report  # noqa: E402
from test_serve_declarations_attestation import (  # noqa: E402
    NEMOTRON, NEMOTRON_CONFIG, _FakeServer, _RecordingBackend, _cached_tok, _load)

from sisteer.capeval.serving import geometry as sg  # noqa: E402
from sisteer.capeval.serving import proxy as px  # noqa: E402
from sisteer.capeval.serving import vllm_server as vs  # noqa: E402
from sisteer.capeval.tinycap import contracts as C  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
GLM_L, GLM_S, GLM_KV = 16384, 48, 782_688          # the Modal container's report, 2026-09-11
GLM_EXPECTED = dict(_FAKE_EXPECTED, max_num_seqs=GLM_S, max_num_batched_tokens=GLM_S * GLM_L,
                    effective_max_num_scheduled_tokens=GLM_S * GLM_L, max_model_len=GLM_L)


def _glm_report(**over):
    rep = _sched_report(**GLM_EXPECTED)
    rep.update(kv_num_gpu_blocks=GLM_KV // 16, kv_block_size=16, kv_num_cache_groups=1)
    rep.update(over)
    return rep


@pytest.fixture(scope="module")
def L():
    return TL._load_launcher(TL._stub_modal(), app_name=TL.TEST_APP_NAME, name="tinycap_served_kv_under_test")


@pytest.fixture
def capctrl():
    return _load("_kv_capctrl_serving_up", "scripts/capctrl_serving_up.py")


# --------------------------------------------------------------------------- #
# (1) the serving lane: a shortfall is ONE named refusal carrying the numbers  #
# --------------------------------------------------------------------------- #


def test_a_pure_kv_shortfall_raises_the_subclass_with_the_containers_numbers():
    with pytest.raises(vs.KvCoverageShortfall) as ei:
        vs.validate_scheduler_reports([_glm_report()], GLM_EXPECTED)
    e = ei.value
    assert isinstance(e, vs.VllmServerError)                        # the SAME refusal to every other catcher
    assert "KV cache holds 782688 tokens < the 786432-token budget (= 48 x 16384)" in str(e)
    assert e.kv_capacity_tokens == GLM_KV and e.kv_backed_concurrency == pytest.approx(GLM_KV / GLM_L)
    assert e.basis == vs.KV_BASIS_UNIFORM and (e.max_model_len, e.max_num_seqs) == (GLM_L, GLM_S)
    assert sg.kv_derived_seqs(e.kv_backed_concurrency) == 47                # 47.77 -> 47
    # the tightest worker decides when several fall short
    with pytest.raises(vs.KvCoverageShortfall) as ei2:
        vs.validate_scheduler_reports([_glm_report(), _glm_report(kv_num_gpu_blocks=(GLM_KV - 16384) // 16)],
                                      GLM_EXPECTED)
    assert ei2.value.kv_capacity_tokens == GLM_KV - 16384
    # a capacity that fits passes unchanged
    assert vs.validate_scheduler_reports([_glm_report(kv_num_gpu_blocks=GLM_S * GLM_L // 16)], GLM_EXPECTED)
    # a hybrid cache: the allocator's own verdict, floor'd the same way
    hybrid = _glm_report(kv_num_cache_groups=2, kv_max_concurrency=47.5,
                         kv_cache_groups=[{"spec": "FullAttentionSpec", "num_layers": 4, "block_size": 16},
                                          {"spec": "MambaSpec", "num_layers": 38, "block_size": 528}])
    with pytest.raises(vs.KvCoverageShortfall) as ei3:
        vs.validate_scheduler_reports([hybrid], GLM_EXPECTED)
    assert ei3.value.basis == vs.KV_BASIS_HYBRID and sg.kv_derived_seqs(ei3.value.kv_backed_concurrency) == 47
    assert ei3.value.kv_capacity_tokens == int(47.5 * GLM_L)
    # kv_coverage_problem's verdicts and messages are byte-identical to kv_coverage's problem half
    for rep in (_glm_report(), hybrid, _glm_report(kv_num_gpu_blocks=None)):
        assert vs.kv_coverage_problem(rep, max_num_seqs=GLM_S, max_model_len=GLM_L) == \
            vs.kv_coverage(rep, max_num_seqs=GLM_S, max_model_len=GLM_L)[0]


def test_the_invariant_still_refuses_a_clamp_a_threshold_or_an_unreadable_report_as_the_plain_error():
    """Nothing is ever derived from a report the invariant refused: a shortfall BESIDE
    a pin problem, an unreadable KV structure, a clamp alone, a long-prefill
    threshold, a speculative config — the plain VllmServerError, never the subclass."""
    cases = [
        _glm_report(max_num_batched_tokens=8192),                         # shortfall + silent clamp
        _glm_report(long_prefill_token_threshold=655),                    # shortfall + chunking threshold
        _glm_report(speculative="SpeculativeConfig(num_speculative_tokens=3)"),
        _glm_report(kv_num_gpu_blocks=None),                              # unreadable KV facts
        _glm_report(kv_num_cache_groups=0),
        _glm_report(kv_num_gpu_blocks=GLM_S * GLM_L // 16, max_num_seqs=47),  # fits, clamped S
    ]
    for rep in cases:
        with pytest.raises(vs.VllmServerError) as ei:
            vs.validate_scheduler_reports([rep], GLM_EXPECTED)
        assert not isinstance(ei.value, vs.KvCoverageShortfall), rep
    # one worker short and one worker clamped on the same launch: plain
    with pytest.raises(vs.VllmServerError) as ei:
        vs.validate_scheduler_reports([_glm_report(), _glm_report(kv_num_gpu_blocks=GLM_S * GLM_L // 16,
                                                                  max_num_seqs=47)], GLM_EXPECTED)
    assert not isinstance(ei.value, vs.KvCoverageShortfall)


# --------------------------------------------------------------------------- #
# the record and its ONE validator                                             #
# --------------------------------------------------------------------------- #


def _record(**over):
    rec = sg.kv_shrink_record(configured_max_num_seqs=GLM_S, max_model_len=GLM_L,
                              kv_backed_concurrency=GLM_KV / GLM_L, kv_capacity_tokens=GLM_KV,
                              basis=vs.KV_BASIS_UNIFORM, first_launch_log="vllm_server.kv_shortfall.log")
    rec.update(over)
    return rec


def test_the_shrink_record_holds_the_rule_and_the_validator_refuses_every_deviation():
    rec = _record()
    assert rec["served_max_num_seqs"] == 47 and rec["configured_max_num_seqs"] == 48 and rec["rule"] == sg.KV_SHRINK_RULE
    ok = dict(configured_max_num_seqs=GLM_S, max_model_len=GLM_L, served_max_num_seqs=47)
    assert sg.kv_shrink_problem(rec, **ok) is None
    assert sg.kv_shrink_problem(json.loads(json.dumps(rec)), **ok) is None       # survives the JSON round trip
    bad = {
        "another configured S": (_record(), dict(ok, configured_max_num_seqs=36)),
        "another window": (_record(), dict(ok, max_model_len=17664)),
        "served S is not what is served": (_record(), dict(ok, served_max_num_seqs=48)),
        "served S != floor": (_record(served_max_num_seqs=46), ok),
        "capacity does not divide to S": (_record(kv_capacity_tokens=46 * GLM_L), ok),
        "another rule": (_record(rule="S = ceil"), ok),
        "not a shrink": (_record(served_max_num_seqs=48, kv_backed_concurrency=48.5,
                                 kv_capacity_tokens=int(48.5 * GLM_L)), dict(ok, served_max_num_seqs=48)),
        "missing field": ({k: v for k, v in _record().items() if k != "basis"}, ok),
        "not a mapping": ("kv_shrink", ok),
        "a bool": (_record(served_max_num_seqs=True), ok),
    }
    for why, (shrink, kw) in bad.items():
        problem = sg.kv_shrink_problem(shrink, **kw)
        assert problem and sg.KV_SHRINK_KEY in problem, (why, problem)
    # the record builder refuses numbers that do not shrink or back nothing
    with pytest.raises(ValueError):
        sg.kv_shrink_record(configured_max_num_seqs=GLM_S, max_model_len=GLM_L, kv_backed_concurrency=48.2,
                            kv_capacity_tokens=int(48.2 * GLM_L), basis="x")
    with pytest.raises(ValueError):
        sg.kv_shrink_record(configured_max_num_seqs=GLM_S, max_model_len=GLM_L, kv_backed_concurrency=0.7,
                            kv_capacity_tokens=int(0.7 * GLM_L), basis="x")
    with pytest.raises(ValueError):
        sg.kv_derived_seqs(float("nan"))


def test_kv_shrunk_geometry_restates_the_resolution_at_the_served_s_honestly():
    geom = sg.resolve_geometry(None, max_model_len=GLM_L, max_num_seqs=GLM_S, fallback=(10240, 64),
                               allow_over_budget=True)
    new = sg.kv_shrunk_geometry(geom, _record())
    assert (new.max_model_len, new.max_num_seqs, new.seqs_source) == (GLM_L, 47, sg.KV_DERIVED_SOURCE)
    assert new.batched_tokens == 47 * GLM_L and new.window_source == geom.window_source
    assert new.as_dict()["max_num_seqs"] == 47 and new.as_dict()["seqs_source"] == "kv_derived"
    assert "NOT identity" in new.describe()                      # the identity S is the budget's; said so
    with pytest.raises(ValueError):
        sg.kv_shrunk_geometry(geom, _record(served_max_num_seqs=46))


# --------------------------------------------------------------------------- #
# (2) the standup: ONE re-launch at floor(the reported concurrency), recorded  #
# --------------------------------------------------------------------------- #


def _bring_up_with_shortfalls(capctrl, tmp_path, monkeypatch, *, shortfalls: int, **kw):
    """The REAL bring_up on the pinned Nemotron config with a REAL ResolvedGeometry
    (4096 x 8) and a launch stand-in that refuses the first ``shortfalls`` launches
    with a KvCoverageShortfall backing 7.3 sequences (the shape readiness half 3
    raises), then comes up. Returns (Standup, [the launches attempted])."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    launches = []

    def fake_launch(launch, *, log_path, **_):
        launches.append(launch)
        Path(log_path).write_text(f"engine log of launch {len(launches)}\n")
        if len(launches) <= shortfalls:
            raise vs.KvCoverageShortfall(
                "resolved scheduler state does not match ...:\n- worker 0: KV cache holds 29900 tokens",
                kv_backed_concurrency=29900 / 4096, kv_capacity_tokens=29900, basis=vs.KV_BASIS_UNIFORM,
                max_model_len=4096, max_num_seqs=launch.max_num_seqs)
        return _FakeServer()
    monkeypatch.setattr(vs, "launch_server", fake_launch)
    monkeypatch.setattr(capctrl, "_start_proxy_thread",
                        lambda app, host, port: types.SimpleNamespace(call_soon_threadsafe=lambda f: None,
                                                                      stop=lambda: None))
    built = {}
    monkeypatch.setattr(px, "make_app", lambda proxy: built.setdefault("proxy", proxy))
    monkeypatch.setattr(px, "HTTPBackend", _RecordingBackend)
    geom = sg.resolve_geometry(None, max_model_len=4096, max_num_seqs=8, fallback=(10240, 64),
                               allow_over_budget=True)
    # the LENS standup (native-serving-default flipped the library default to lora_native)
    kw.setdefault("serving_path", "lens")
    st = capctrl.bring_up(repo=REPO, config_path=REPO / NEMOTRON_CONFIG, cell="none",
                          defenses_dir=tmp_path / "cells", out=tmp_path / "out", geom=geom, **kw)
    return st, launches


def test_the_standup_relaunches_once_at_the_kv_derived_s_and_records_the_shrink(capctrl, tmp_path, monkeypatch, capsys):
    _cached_tok(*NEMOTRON)
    st, launches = _bring_up_with_shortfalls(capctrl, tmp_path, monkeypatch, shortfalls=1, kv_derived_seqs=True)
    assert [l.max_num_seqs for l in launches] == [8, 7]                      # floor(7.30) = 7, ONE re-launch
    assert launches[1].max_num_batched_tokens == 7 * 4096 == launches[1].max_num_seqs * launches[1].max_model_len
    cmd = list(launches[1].cmd)
    assert cmd[cmd.index("--max-num-seqs") + 1] == "7" and cmd[cmd.index("--max-num-batched-tokens") + 1] == str(7 * 4096)
    # the composed trailing flags ride the re-launch too
    assert "--gpu-memory-utilization" in cmd and "--max-logprobs" in cmd
    # the standup SERVES the derived geometry and says so
    assert (st.geometry.max_model_len, st.geometry.max_num_seqs, st.geometry.seqs_source) == (4096, 7, sg.KV_DERIVED_SOURCE)
    ep = st.endpoint()
    assert ep["max_num_seqs"] == 7 and ep["geometry"]["max_num_seqs"] == 7
    shrink = ep["geometry"][sg.KV_SHRINK_KEY]
    assert shrink == ep["kv_shrink"] and shrink["configured_max_num_seqs"] == 8 and shrink["served_max_num_seqs"] == 7
    assert shrink["kv_capacity_tokens"] == 29900 and shrink["basis"] == vs.KV_BASIS_UNIFORM
    assert sg.kv_shrink_problem(shrink, configured_max_num_seqs=8, max_model_len=4096, served_max_num_seqs=7) is None
    # the first launch's engine log is kept aside, the served launch's is clean
    out = tmp_path / "out"
    assert (out / capctrl.KV_SHORTFALL_LOG).read_text().startswith("engine log of launch 1")
    assert (out / "vllm_server.log").read_text().startswith("engine log of launch 2")
    assert shrink["first_launch_log"] == capctrl.KV_SHORTFALL_LOG
    text = capsys.readouterr().out
    assert "KV-DERIVED SHRINK" in text and "re-launching ONCE at S=7" in text


def test_without_the_opt_in_the_refusal_stands_and_a_second_shortfall_is_never_answered(capctrl, tmp_path, monkeypatch):
    _cached_tok(*NEMOTRON)
    with pytest.raises(vs.KvCoverageShortfall):
        _bring_up_with_shortfalls(capctrl, tmp_path / "off", monkeypatch, shortfalls=1)
    with pytest.raises(vs.KvCoverageShortfall):                              # ONE re-launch, never a loop
        _bring_up_with_shortfalls(capctrl, tmp_path / "twice", monkeypatch, shortfalls=2, kv_derived_seqs=True)
    st, launches = _bring_up_with_shortfalls(capctrl, tmp_path / "fits", monkeypatch, shortfalls=0, kv_derived_seqs=True)
    assert len(launches) == 1 and st.geometry.max_num_seqs == 8 and st.kv_shrink is None
    assert sg.KV_SHRINK_KEY not in st.endpoint()["geometry"] and st.endpoint()["kv_shrink"] is None
    # a shortfall that backs no full-length sequence is a refusal, not S=0
    short = vs.KvCoverageShortfall("x", kv_backed_concurrency=0.4, kv_capacity_tokens=1638, basis="u",
                                   max_model_len=4096, max_num_seqs=8)
    geom = sg.resolve_geometry(None, max_model_len=4096, max_num_seqs=8, fallback=(10240, 64), allow_over_budget=True)
    with pytest.raises(vs.VllmServerError, match="impossible"):
        capctrl.kv_derived_geometry(geom, short, tmp_path / "none")
    with pytest.raises(TypeError):
        capctrl.kv_derived_geometry(types.SimpleNamespace(max_num_seqs=8, max_model_len=4096), short, tmp_path / "ns")


def test_main_wires_the_flag_off_by_default_and_the_launcher_spells_it_once(capctrl, L):
    p = capctrl.build_parser()
    base = ["--repo-dir", "r", "--config", "c", "--cell", "none", "--defenses-dir", "d", "--out", "o", "--stop-file", "s"]
    assert p.parse_args(base).kv_derived_seqs == "off"
    assert p.parse_args(base + [capctrl.KV_DERIVED_SEQS_FLAG, "on"]).kv_derived_seqs == "on"
    with pytest.raises(SystemExit):
        p.parse_args(base + [capctrl.KV_DERIVED_SEQS_FLAG, "maybe"])
    assert capctrl.KV_DERIVED_SEQS_FLAG == L.STANDUP_KV_DERIVED_SEQS_FLAG == "--kv-derived-seqs"
    import inspect
    src = inspect.getsource(capctrl.main)
    assert 'kv_derived_seqs=(args.kv_derived_seqs == "on")' in src


# --------------------------------------------------------------------------- #
# (3) the launcher: every standup opts in; the record is read back through the #
# one validator; the identity follows the SERVED S                             #
# --------------------------------------------------------------------------- #


def test_every_standup_argv_opts_in_as_a_valued_pair_in_the_fixed_part(L, tmp_path):
    up = TL._load_standup()
    for cfg in L.plan_configurations():
        geom, _ = L.served_geometry(cfg, 1)
        argv = L._standup_cmd("/tree", "/tree/cfg.yaml", cfg["cell"], "d", tmp_path, tmp_path / "STOP",
                              geom, 64, cfg["standup_extra"])
        i = argv.index(L.STANDUP_KV_DERIVED_SEQS_FLAG)
        assert argv[i - 2:i + 4] == ["--gpu-mem-util", "0.90", L.STANDUP_KV_DERIVED_SEQS_FLAG, "on",
                                     "--ready-timeout-s", "2700"], cfg["name"]
        assert argv.count(L.STANDUP_KV_DERIVED_SEQS_FLAG) == 1
        assert up.build_parser().parse_args(argv[2:]).kv_derived_seqs == "on"


def test_served_geometry_of_endpoint_accepts_the_configured_s_or_a_holding_record_and_refuses_the_rest(L):
    geom = sg.resolve_geometry(None, max_model_len=GLM_L, max_num_seqs=GLM_S, fallback=(10240, 64), allow_over_budget=True)
    same = {"geometry": {"max_model_len": GLM_L, "max_num_seqs": GLM_S}}
    g, shrink, problem = L.served_geometry_of_endpoint(same, geom)
    assert (g, shrink, problem) == (geom, None, None)
    rec = _record()
    shrunk = {"geometry": {"max_model_len": GLM_L, "max_num_seqs": 47, sg.KV_SHRINK_KEY: rec}}
    g, shrink, problem = L.served_geometry_of_endpoint(shrunk, geom)
    assert problem is None and shrink == rec and (g.max_num_seqs, g.seqs_source) == (47, sg.KV_DERIVED_SOURCE)
    # another S with no record; a record that does not hold; another window; no block
    for ep, needle in ((({"geometry": {"max_model_len": GLM_L, "max_num_seqs": 47}}), "records no kv_shrink"),
                       ({"geometry": {"max_model_len": GLM_L, "max_num_seqs": 46, sg.KV_SHRINK_KEY: rec}}, "does not hold"),
                       ({"geometry": {"max_model_len": GLM_L, "max_num_seqs": 47, sg.KV_SHRINK_KEY: _record(served_max_num_seqs=46)}}, "does not hold"),
                       ({"geometry": {"max_model_len": 17664, "max_num_seqs": GLM_S}}, "max_model_len"),
                       ({}, "no geometry block")):
        g, _, problem = L.served_geometry_of_endpoint(ep, geom)
        assert problem and needle in problem and g is geom, (ep, problem)
    # the REC's geometry block re-stated at the served S, the asked S kept beside it
    record = {**geom.as_dict(), C.GEOMETRY_PASS_FIELD: 1, "model_budget": {"x": 1}}
    out = L.served_geometry_record(record, sg.kv_shrunk_geometry(geom, rec), rec)
    assert out["max_num_seqs"] == 47 and out["configured_max_num_seqs"] == 48 and out["batched_tokens"] == 47 * GLM_L
    assert out["identity"] is False and out["seqs_source"] == sg.KV_DERIVED_SOURCE and out[sg.KV_SHRINK_KEY] == rec
    assert out[C.GEOMETRY_PASS_FIELD] == 1 and out["model_budget"] == {"x": 1} and out["max_model_len"] == GLM_L
    assert L.served_lxs(shrunk, geom) == {"max_num_seqs": 47, "max_model_len": GLM_L, "configured_max_num_seqs": 48,
                                          "kv_shrink": rec}
    assert C.REC_SERVED_GEOMETRY_MISMATCH == "SERVED_GEOMETRY_MISMATCH" and C.is_refusal_status(C.REC_SERVED_GEOMETRY_MISMATCH)
    assert C.REC_SERVED_GEOMETRY_MISMATCH in C.REFUSAL_REC_NAMES     # this tree's held set (no FULL22_REC_NAMES here)


def test_a_shrunk_store_is_trusted_before_the_standup_only_through_the_validator_and_exactly_after(L, tmp_path):
    """PW5-4: the served S is store identity. A store this configuration wrote at the
    KV-derived S carries the record in its manifest geometry; BEFORE the standup the
    launcher (expectation at the configured S) trusts it only when the record shrinks
    THIS configured S at THIS window by the rule; AFTER the standup the expectation is
    the served S and every store is held to it exactly."""
    cfg = L.plan_configurations()[0]
    geom, _ = L.served_geometry(cfg, 1)
    S, Lw = geom.max_num_seqs, geom.max_model_len
    served = S - 1
    conc = S - 0.23
    rec = sg.kv_shrink_record(configured_max_num_seqs=S, max_model_len=Lw, kv_backed_concurrency=conc,
                              kv_capacity_tokens=int(conc * Lw), basis=vs.KV_BASIS_UNIFORM)
    assert rec["served_max_num_seqs"] == served
    exp_conf = L.expected_identity(cfg, geom, pass_no=1, controls=cfg["controls"], samples=4, tree_sha=TL.FULL_SHA)
    geom_served = sg.kv_shrunk_geometry(geom, rec)
    exp_served = L.expected_identity(cfg, geom_served, pass_no=1, controls=cfg["controls"], samples=4, tree_sha=TL.FULL_SHA)
    assert exp_served.served["max_num_seqs"] == exp_served.served["concurrency"] == served
    assert exp_served._replace(served=exp_conf.served) == exp_conf          # S (and the width) is the ONLY difference
    assert L.expected_at_served_seqs(exp_conf, served) == exp_served         # the ONE substitution both callers make
    store = TL._store(tmp_path, exp_served, mutate=lambda m: m[C.MANIFEST_KEY_GEOMETRY].update({sg.KV_SHRINK_KEY: rec}))
    # after the standup (exact): the served expectation trusts it, the configured one does not
    assert L.untrusted_store_reason(store, expected=exp_served) is None
    why = L.untrusted_store_reason(store, expected=exp_conf)
    assert why and "max_num_seqs" in why
    # before the standup (tolerated): trusted at ITS served S through the record ...
    assert L.untrusted_store_reason(store, expected=exp_conf, kv_shrink_tolerated=True) is None
    adjusted, problem = L.kv_shrink_adjusted_expectation(json.loads((store / "manifest.json").read_text()), exp_conf)
    assert problem is None and adjusted.served["max_num_seqs"] == adjusted.served["concurrency"] == served
    # ... never around a record that does not hold (another configured S, a wrong floor) ...
    bad = TL._store(tmp_path / "bad", exp_served,
                    mutate=lambda m: m[C.MANIFEST_KEY_GEOMETRY].update({sg.KV_SHRINK_KEY: dict(rec, served_max_num_seqs=served - 1)}))
    why = L.untrusted_store_reason(bad, expected=exp_conf, kv_shrink_tolerated=True)
    assert why and sg.KV_SHRINK_KEY in why
    other = TL._store(tmp_path / "other", exp_served,
                      mutate=lambda m: m[C.MANIFEST_KEY_GEOMETRY].update({sg.KV_SHRINK_KEY: dict(rec, configured_max_num_seqs=S + 1)}))
    assert sg.KV_SHRINK_KEY in (L.untrusted_store_reason(other, expected=exp_conf, kv_shrink_tolerated=True) or "")
    # ... and a store WITHOUT a record is held to the configured identity as before
    plain = TL._store(tmp_path / "plain", exp_conf)
    assert L.untrusted_store_reason(plain, expected=exp_conf, kv_shrink_tolerated=True) is None
    assert L.untrusted_store_reason(plain, expected=exp_served, kv_shrink_tolerated=True) is not None
    # the resume report carries the tolerance through; the exact call does not
    tol = L.resume_report(tmp_path, cfg["controls"], exp_conf, kv_shrink_tolerated=True)
    assert tol["stores"][store.name] is None
    exact = L.resume_report(tmp_path, cfg["controls"], exp_conf)
    assert exact["stores"][store.name] is not None


def test_the_cell_flow_rederives_the_identity_and_the_concurrency_at_the_served_s(L):
    """Source pins on the one cell flow: the served geometry is read off the endpoint
    through served_geometry_of_endpoint, a problem is the SERVED_GEOMETRY_MISMATCH
    refusal (torn down), the expectation / REC geometry / resume decision are
    re-derived at the served S, and the driver's --concurrency is the served S
    (it takes `geom`, which is rebound to the served geometry)."""
    import inspect
    # tinycap-shared-standup §1.1.1: the cell flow is composed of the reusable pieces the row runner shares
    # (the identity stage, the standup + attestation, the drive) — the pins read their sources in flow order
    src = "".join(inspect.getsource(f) for f in (L._run_configuration, L._cell_identity, L._standup_and_attest, L._retake_resume, L._drive_cell))
    # the call wraps since the native ladder passes `ladder_record=`: pin the assignment + callee,
    # then the first argument on the next line
    i_ep = src.index("geom_served, kv_shrink, geometry_problem = served_geometry_of_endpoint(")
    assert src[i_ep:i_ep + 200].split("(", 1)[1].lstrip().startswith("endpoint, geom")
    assert "C.REC_SERVED_GEOMETRY_MISMATCH" in src[i_ep:i_ep + 900] and "_tear_down_standup(stop, standup)" in src[i_ep:i_ep + 900]
    i_re = src.index("geom = geom_served")
    assert i_ep < i_re < src.index("cmd = _driver_cmd(tree, cfg, endpoint, ledger_url, geom,")
    assert 'rec["geometry"] = served_geometry_record(rec["geometry"], geom_served, kv_shrink)' in src
    assert src.count("resume = resume_report(out_dir, wanted, expected, kv_shrink_tolerated=True)") == 1   # BEFORE
    # AFTER, exact (this tree's flow keeps main's ladder arm beside the plain one on both sides)
    assert 'rec["resume_prestandup"] = resume' in src
    assert ("resume = resume_report(out_dir, wanted, expected) if not rung else "
            "ladder_resume_report(out_dir, wanted, expected)\n") in src
    assert "producer_tree_conflicts(out_dir, expected=expected, kv_shrink_tolerated=True)" in src
