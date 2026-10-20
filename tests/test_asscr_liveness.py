"""The low-dose numeric-liveness gate (scripts/alphasteer_f22/screen/asscr_liveness.py) — its verdicts, driven
against a STUB serving path so the decision logic is tested rather than only exercised live.

The gate decides whether a cell is allowed to generate at all, so the cases that matter are the ones where the greedy
text probe and the numeric evidence DISAGREE:
  * live and SMALL — zero text divergence, a measured effect above noise -> `live_numeric`, and the text result is
    recorded as an observation (the case this exists for)
  * genuinely dead — an effect indistinguishable from the engine's own repeat noise -> `not_live`, whatever the text says
  * a spec-identical intervention -> `not_live` even with a large effect (the silently-unchanged-intervention failure)
  * an unreadable instrument -> `unevaluable`, never a pass.

The RULE under test is the instrument's own `fidelity.check_logprob_steering_bites`, loaded from the wave checkout by
path — only the transport (the proxy client, the routing table, the driver module) is stubbed.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GATE = REPO / "scripts" / "alphasteer_f22" / "screen" / "asscr_liveness.py"
sys.path.insert(0, str(REPO / "scripts" / "alphasteer_f22" / "screen"))
import as22_screen as scr  # noqa: E402

PROBES = ("List the first five prime numbers.", "Explain briefly why the sky is blue.", "Write a haiku about a quiet library.")


class _Route:
    def __init__(self, cell_id, vmodel, ihash, identity="proxy_vllm+lens+interleaved"):
        self.cell_id, self.virtual_model, self.intervention_hash, self.serving_identity = cell_id, vmodel, ihash, identity
        self.si_prefix_hash = None


def _fake_wave_repo(tmp_path: Path) -> Path:
    """A wave checkout with only what the gate reads by path: the driver's two symbols. The RULE comes from the real
    fidelity module, registered in sys.modules by the fixture below."""
    wave = tmp_path / "wave"
    (wave / "src").mkdir(parents=True)
    (wave / "scripts" / "sdfbase" / "full22").mkdir(parents=True)
    (wave / "scripts" / "served_capability_run.py").write_text(
        "MMLU_TOP_LOGPROBS = 64\n"
        "def _token_id_of(token):\n"
        "    try:\n"
        "        return int(str(token).split(':')[-1])\n"
        "    except Exception:\n"
        "        return None\n")
    return wave


def _install_stubs(monkeypatch, *, legs, texts, route_hash="HASH_STEER", anchor_hash="HASH_ANCHOR", twin=True,
                   real_fidelity=True, break_client=False):
    """Register stub sisteer modules so the gate's imports resolve to them. `legs` maps (vmodel, prompt) -> {token_id:
    logprob} for the one-token probe; `texts` maps (vmodel, prompt) -> the greedy text."""
    route = _Route("alphasteer", "petri__alphasteer", route_hash)
    anchor = _Route("none", "petri__base", anchor_hash, identity="proxy_vllm+bare")
    twin_r = _Route("none__lens0", "petri__none__lens0", "HASH_TWIN")
    routes = [route, anchor] + ([twin_r] if twin else [])

    class _Client:
        def __init__(self, url, identity=None):
            if break_client:
                raise RuntimeError("proxy unreachable")
            self.url = url

        def load(self, require_virtual_models=None):
            return None

        def chat_completions(self, body):
            vm, prompt = body["model"], body["messages"][-1]["content"]
            if body.get("logprobs"):
                d = legs[(vm, prompt)]
                items = sorted(d.items())
                return {"choices": [{"logprobs": {"content": [{"token": f"tok:{items[0][0]}", "logprob": items[0][1],
                                                               "top_logprobs": [{"token": f"tok:{t}", "logprob": lp}
                                                                                for t, lp in items]}]},
                                     "message": {"content": ""}}]}
            return {"choices": [{"message": {"content": texts.get((vm, prompt), "")}}],
                    "petri_intervention_hash": {route.virtual_model: route_hash, anchor.virtual_model: anchor_hash,
                                                twin_r.virtual_model: "HASH_TWIN"}[vm]}

    proxy = types.ModuleType("sisteer.capeval.serving.proxy")
    proxy.ProxyClientEngine = _Client
    proxy.load_routing_table = lambda p: {"routes": routes}
    bs = types.ModuleType("sisteer.eval.attacks.bon_served")
    bs._is_lens = lambda r: "lens" in r.serving_identity
    stgt = types.ModuleType("sisteer.eval.attacks.served_target")
    stgt.PROBE_PROMPTS = PROBES

    def _route_for(cell, rts):
        for r in rts:
            if r.cell_id == cell:
                return r
        raise RuntimeError(f"no route for {cell}")

    stgt.route_for = _route_for
    mods = {"sisteer.capeval.serving.proxy": proxy, "sisteer.eval.attacks.bon_served": bs,
            "sisteer.eval.attacks.served_target": stgt}
    if real_fidelity:
        wave_src = None
        for cand in (REPO / "src",):
            if (cand / "sisteer" / "capeval" / "serving" / "fidelity.py").is_file():
                wave_src = cand
                break
        if wave_src is None:
            pytest.skip("no fidelity.py to load the real rule from")
        spec = importlib.util.spec_from_file_location("sisteer.capeval.serving.fidelity",
                                                      wave_src / "sisteer" / "capeval" / "serving" / "fidelity.py")
        fid = importlib.util.module_from_spec(spec)
        # in sys.modules BEFORE exec: its dataclasses resolve annotations through sys.modules[cls.__module__]
        monkeypatch.setitem(sys.modules, "sisteer.capeval.serving.fidelity", fid)
        spec.loader.exec_module(fid)
        mods["sisteer.capeval.serving.fidelity"] = fid
    for name, m in mods.items():
        monkeypatch.setitem(sys.modules, name, m)
        # ensure the parent packages EXIST and carry the attribute, without replacing a real one already imported
        parent, _, leaf = name.rpartition(".")
        child = m
        while parent:
            pm = sys.modules.get(parent)
            if pm is None:
                pm = types.ModuleType(parent)
                pm.__path__ = []          # a package, so `from x.y import z` resolves
                monkeypatch.setitem(sys.modules, parent, pm)
            monkeypatch.setattr(pm, leaf, child, raising=False)
            child, (parent, _, leaf) = pm, parent.rpartition(".")
    return routes


def _run(tmp_path, monkeypatch, **kw):
    wave = _fake_wave_repo(tmp_path)
    _install_stubs(monkeypatch, **kw)
    spec = importlib.util.spec_from_file_location("asscr_liveness", GATE)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    out = tmp_path / "liveness.json"
    rt = tmp_path / "routing_table.json"
    rt.write_text("{}")
    rc = gate.main(["--wave-repo", str(wave), "--proxy-url", "http://127.0.0.1:1/v1", "--routing-table", str(rt),
                    "--cell", "alphasteer", "--tag", "nemotron3nano4b_as_d0p3", "--nominal-dose", "0.3",
                    "--out", str(out)])
    return rc, (json.loads(out.read_text()) if out.is_file() else None)


def _legs(steer_delta: float, noise: float = 0.0005, identical_text: bool = True):
    """Three legs over a common 4-token support: the steered leg offset by `steer_delta`, the twin's repeat by `noise`."""
    legs, texts = {}, {}
    base = {10: -0.5, 11: -1.5, 12: -2.5, 13: -3.5}
    for p in PROBES:
        legs[("petri__alphasteer", p)] = {t: v + steer_delta for t, v in base.items()}
        legs[("petri__none__lens0", p)] = dict(base)
        texts[("petri__alphasteer", p)] = "SAME" if identical_text else "DIFFERENT steered text"
        texts[("petri__none__lens0", p)] = "SAME"
        texts[("petri__base", p)] = "SAME"
    # the twin is asked twice; _TwinNoise (below) returns the engine's own noise on the SECOND read
    return legs, texts


class _TwinNoise(dict):
    """(vmodel, prompt) -> logprobs, returning a slightly different vector on the twin's SECOND read (the noise leg)."""

    def __init__(self, legs, noise):
        super().__init__(legs)
        self._noise, self._seen = noise, {}

    def __getitem__(self, key):
        v = dict(super().__getitem__(key))
        if key[0] == "petri__none__lens0":
            n = self._seen.get(key, 0)
            self._seen[key] = n + 1
            if n == 1:
                v = {t: lp + self._noise for t, lp in v.items()}
        return v


def test_live_and_small_is_live_numeric_with_the_text_probe_recorded_as_an_observation(tmp_path, monkeypatch):
    """The case this exists for: NO greedy-text divergence (0/3) but a measured effect far above the repeat noise."""
    legs, texts = _legs(steer_delta=0.05, identical_text=True)
    rc, rec = _run(tmp_path, monkeypatch, legs=_TwinNoise(legs, 0.0005), texts=texts)
    assert rc == 0 and rec["verdict"] == "live_numeric"
    checks = {c["check"]: c["status"] for c in rec["checks"]}
    assert checks["intervention_hash_distinct"] == "PASS" and checks["continuous_effect"] == "PASS"
    assert checks["text_divergence"] == "OBSERVED"
    obs = rec["text_divergence_observation"]
    assert obs["n_differing"] == 0 and obs["n_probes"] == len(PROBES)
    d = rec["continuous_effect"]["diag"]
    assert d["steered_divergence_median"] == pytest.approx(0.05, rel=1e-6)
    assert d["bar"] == pytest.approx(max(1e-3, 2 * 0.0005), rel=1e-6) and d["steered_divergence_median"] > d["bar"]
    # with no dose artifact, fact (B) falls to the serving-time effect and the record says so, never substituting silently
    d = rec["realised_dose"]
    assert d["available"] is False and "no --dose-artifact given" in d["why"]
    assert d["serving_time_available"] is False and "rho record" in d["serving_time_why"]
    assert d["effect_size_proxy"] == pytest.approx(0.05, rel=1e-6) and "not linear in the dose" in d["effect_size_proxy_note"]
    assert rec["facts"]["offline_realised_dose_ok"] is None and "PROXY" in rec["facts"]["fact_b_basis"]


def test_an_effect_inside_the_engines_own_noise_is_not_live_however_the_text_reads(tmp_path, monkeypatch):
    legs, texts = _legs(steer_delta=0.0002, identical_text=False)   # text DIFFERS, but the effect is under the bar
    rc, rec = _run(tmp_path, monkeypatch, legs=_TwinNoise(legs, 0.0005), texts=texts)
    assert rc == 2 and rec["verdict"] == "not_live"
    checks = {c["check"]: c["status"] for c in rec["checks"]}
    assert checks["continuous_effect"] == "FAIL" and checks["intervention_hash_distinct"] == "PASS"
    assert "does not clear the bar" in rec["continuous_effect"]["error"]
    assert "realised_dose_or_effect" in rec["basis"] and "genuinely not live" in rec["basis"]


def test_bit_identical_logprobs_are_not_live(tmp_path, monkeypatch):
    legs, texts = _legs(steer_delta=0.0, identical_text=True)
    rc, rec = _run(tmp_path, monkeypatch, legs=_TwinNoise(legs, 0.0), texts=texts)
    assert rc == 2 and rec["verdict"] == "not_live"
    assert "BIT-IDENTICAL" in rec["continuous_effect"]["error"]


def test_an_intervention_identical_to_the_anchors_is_not_live_even_with_a_large_effect(tmp_path, monkeypatch):
    legs, texts = _legs(steer_delta=0.5, identical_text=False)
    rc, rec = _run(tmp_path, monkeypatch, legs=_TwinNoise(legs, 0.0005), texts=texts,
                   route_hash="SAME_HASH", anchor_hash="SAME_HASH")
    assert rc == 2 and rec["verdict"] == "not_live"
    checks = {c["check"]: c["status"] for c in rec["checks"]}
    assert checks["intervention_hash_distinct"] == "FAIL" and checks["continuous_effect"] == "PASS"
    assert "intervention_hash_distinct" in rec["basis"]


def test_no_twin_route_or_an_unreachable_proxy_is_unevaluable_never_a_pass(tmp_path, monkeypatch):
    legs, texts = _legs(steer_delta=0.05)
    rc, rec = _run(tmp_path, monkeypatch, legs=_TwinNoise(legs, 0.0005), texts=texts, twin=False)
    assert rc == 3 and rec["verdict"] == "unevaluable"
    assert any("scale-0 twin" in str(c.get("error", "")) for c in rec["checks"])
    rc2, rec2 = _run(tmp_path / "b", monkeypatch, legs=_TwinNoise(legs, 0.0005), texts=texts, break_client=True)
    assert rc2 == 3 and rec2["verdict"] == "unevaluable"


def test_the_record_is_what_the_policy_binds_on(tmp_path, monkeypatch):
    """The admitting record must carry the fields the policy module checks: verdict, cell, routing_table."""
    legs, texts = _legs(steer_delta=0.05)
    rc, rec = _run(tmp_path, monkeypatch, legs=_TwinNoise(legs, 0.0005), texts=texts)
    assert rc == 0
    assert rec["cell"] == "alphasteer" and rec["verdict"] == "live_numeric"
    assert Path(rec["routing_table"]) == (tmp_path / "routing_table.json").resolve()
    assert rec["tag"] == "nemotron3nano4b_as_d0p3" and rec["nominal_dose"] == 0.3
    assert rec["thresholds"]["rule_source"].startswith("sisteer.capeval.serving.fidelity.check_logprob_steering_bites")
    assert rec["schema"] == "asscr_low_dose_liveness/1"


def test_the_gate_and_the_policy_agree_on_the_env_var_name():
    pol = importlib.util.spec_from_file_location(
        "pol", REPO / "scripts" / "sdfbase" / "full22" / "low_dose_liveness_policy.py")
    if pol is None or not Path(pol.origin).is_file():
        pytest.skip("low_dose_liveness_policy.py is not in this checkout")
    m = importlib.util.module_from_spec(pol)
    pol.loader.exec_module(m)
    spec = importlib.util.spec_from_file_location("asscr_liveness_names", GATE)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    assert gate.POLICY_ENV == m.POLICY_ENV == "SISTEER_LOW_DOSE_LIVENESS_RECORD"
    assert gate.VERDICTS[0] == m.REQUIRED_VERDICT == "live_numeric"


# -------------------------------------------------------------------------------------------------------------------- #
# The OFFLINE realised dose is fact (B), bound to the served object by sha                                             #
# -------------------------------------------------------------------------------------------------------------------- #

NEMO_RUNGS = {          # the committed capture's real numbers (dose_nemotron3nano4b.json, schema strength_dose/1)
    "as_dm0p5__alphasteer": (-0.5, 0.08152889460325241, "a154867cddc34fec903418aecd95d7106b5c573896a6710684a70c859ef10e07"),
    "as_d0p1__alphasteer": (0.1, 0.016305778175592422, "75891684edddeda3a9a8836114a2b9c4101649a42579f1f50237d5ed2c41ea6c"),
    "as_d0p3__alphasteer": (0.3, 0.048917338252067566, "27710270acb886f68818f21194f7688886503d26d0fb6a0b559f22d4286c0e03"),
    "as_d0p5__alphasteer": (0.5, 0.08152889460325241, "ccdd9731a35800bcb0a2ae1999cdf40e304814c4b9de7082c3f89656f7a5e9a0"),
    "as_d1__alphasteer": (1.0, 0.16305778920650482, "99f335e7867ec14bb7d93408362179140555a927224e118d5ee776e1d3670f0f"),
}
D0P3_SHA = NEMO_RUNGS["as_d0p3__alphasteer"][2]


def _dose_artifact(tmp_path: Path, *, rungs=None, gate=True, schema="strength_dose/1", target="nemotron3nano4b") -> Path:
    rungs = NEMO_RUNGS if rungs is None else rungs
    art = {"schema": schema, "target": target,
           "alphasteer": {"reference": "as_d1__alphasteer",
                          "rungs": {k: {"view": k.split("__")[0], "nominal_dose": n, "realised_dose_median": m,
                                        "object_sha256": s, "staged_id": k,
                                        "effective_layers": {"effective_layer_count": 19}}
                                    for k, (n, m, s) in rungs.items()}},
           "gate_summary": {"all_pass": bool(gate), "rungs": {f"alphasteer/{k}": bool(gate) for k in rungs}},
           "provenance": {"code": {"code_sha": "8103fde1", "dirty": False}, "gate": {"rule": "D51/D55 signed factor"},
                          "geometry": {}, "h_definition": "post-norm residual"}}
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = tmp_path / "dose.json"
    p.write_text(json.dumps(art))
    return p


def _run_dose(tmp_path, monkeypatch, *, artifact, staged="as_d0p3__alphasteer", obj_sha=D0P3_SHA, nominal="0.3",
              steer_delta=0.05, noise=0.0005):
    legs, texts = _legs(steer_delta=steer_delta)
    tmp_path.mkdir(parents=True, exist_ok=True)
    wave = _fake_wave_repo(tmp_path)
    _install_stubs(monkeypatch, legs=_TwinNoise(legs, noise), texts=texts)
    spec = importlib.util.spec_from_file_location("asscr_liveness_dose", GATE)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    out, rt = tmp_path / "liveness.json", tmp_path / "routing_table.json"
    rt.write_text("{}")
    argv = ["--wave-repo", str(wave), "--proxy-url", "http://127.0.0.1:1/v1", "--routing-table", str(rt),
            "--cell", "alphasteer", "--tag", "nemotron3nano4b_as_d0p3", "--nominal-dose", nominal, "--out", str(out),
            "--staged-id", staged, "--arm", "alphasteer"]
    if artifact:
        argv += ["--dose-artifact", str(artifact)]
    if obj_sha:
        argv += ["--object-sha256", obj_sha]
    rc = gate.main(argv)
    return rc, json.loads(out.read_text())


def test_the_offline_dose_satisfies_fact_b_and_is_labelled_offline(tmp_path, monkeypatch):
    rc, rec = _run_dose(tmp_path, monkeypatch, artifact=_dose_artifact(tmp_path))
    assert rc == 0 and rec["verdict"] == "live_numeric"
    d = rec["realised_dose"]
    assert d["available"] is True and d["source"] == "offline_capture"
    assert d["realised_dose_median"] == pytest.approx(0.048917338252067566) and d["artifact_ratio_gate"] is True
    assert d["object_sha256"] == D0P3_SHA and d["staged_id"] == "as_d0p3__alphasteer"
    assert "NOT a serving-time attestation" in d["label"]
    # the independent cross-check: realised / |nominal| is constant across the arm's rungs (0.163058), sign excluded
    assert d["ratio_consistent_rtol_1e-5"] is True
    assert set(d["realised_over_abs_nominal_by_rung"]) == set(NEMO_RUNGS)
    assert all(v == pytest.approx(0.16305778, rel=1e-6) for v in d["realised_over_abs_nominal_by_rung"].values())
    # the serving-time proxy is kept ALONGSIDE, not replaced
    assert d["effect_size_proxy"] == pytest.approx(0.05, rel=1e-6) and "not linear in the dose" in d["effect_size_proxy_note"]
    assert rec["facts"]["offline_realised_dose_ok"] is True and "OFFLINE realised dose" in rec["facts"]["fact_b_basis"]
    assert rec["text_divergence_observation"]["n_differing"] == 0      # live and small, still
    assert "disagreement" not in rec


def test_the_negative_rung_is_not_failed_by_its_unsigned_median(tmp_path, monkeypatch):
    """realised_dose_median is a MAGNITUDE — as_dm0p5 and as_d0p5 read the same — so the ratio cross-check divides by
    |nominal| and the sign stays the artifact's own signed gate's business."""
    rc, rec = _run_dose(tmp_path, monkeypatch, artifact=_dose_artifact(tmp_path), staged="as_dm0p5__alphasteer",
                        obj_sha=NEMO_RUNGS["as_dm0p5__alphasteer"][2], nominal="-0.5")
    assert rc == 0 and rec["verdict"] == "live_numeric"
    assert rec["realised_dose"]["available"] is True and rec["realised_dose"]["ratio_consistent_rtol_1e-5"] is True


def test_a_sha_mismatch_refuses_to_borrow_and_names_the_rebake_prerequisite(tmp_path, monkeypatch):
    """The re-bake case: different bytes, different layer sets. The artifact must NOT be used for them."""
    rc, rec = _run_dose(tmp_path, monkeypatch, artifact=_dose_artifact(tmp_path), obj_sha="f" * 64)
    d = rec["realised_dose"]
    assert d["available"] is False and "REFUSING to borrow" in d["why"] and "RE-BAKED" in d["why"]
    # fact (B) falls back to the serving-time effect, which passes here, so the cell is still live — as a PROXY
    assert rc == 0 and rec["verdict"] == "live_numeric"
    assert rec["facts"]["offline_realised_dose_ok"] is None and "PROXY" in rec["facts"]["fact_b_basis"]


def test_a_zero_dose_or_a_failed_artifact_gate_is_not_live(tmp_path, monkeypatch):
    zero = dict(NEMO_RUNGS); zero["as_d0p3__alphasteer"] = (0.3, 0.0, D0P3_SHA)
    rc, rec = _run_dose(tmp_path, monkeypatch, artifact=_dose_artifact(tmp_path / "z", rungs=zero))
    assert rc == 2 and rec["verdict"] == "not_live" and "ZERO realised dose" in rec["realised_dose"]["why"]
    rc2, rec2 = _run_dose(tmp_path / "g", monkeypatch, artifact=_dose_artifact(tmp_path / "g", gate=False))
    assert rc2 == 2 and rec2["verdict"] == "not_live" and "ratio gate" in rec2["realised_dose"]["why"]


def test_offline_passing_while_the_serving_time_effect_is_inside_the_noise_is_live_but_FLAGGED(tmp_path, monkeypatch):
    """The offline fact stands, and the disagreement — the shape a serving bug makes — is surfaced, never a stop."""
    rc, rec = _run_dose(tmp_path, monkeypatch, artifact=_dose_artifact(tmp_path), steer_delta=0.0002, noise=0.0005)
    assert rc == 0 and rec["verdict"] == "live_numeric"
    dis = rec["disagreement"]
    assert "serving-time effect is not" in dis["what"] and "SERVING bug" in dis["why_it_matters"]
    assert dis["offline_realised_dose_median"] == pytest.approx(0.048917338252067566)
    assert rec["facts"]["serving_time_effect_above_noise"] is False and rec["facts"]["realised_dose_or_effect"] is True
    assert any(c["check"] == "offline_vs_serving_time" and c["status"] == "DISAGREE" for c in rec["checks"])


def test_a_wrong_schema_a_missing_rung_or_a_nominal_mismatch_falls_back_to_the_proxy(tmp_path, monkeypatch):
    rc, rec = _run_dose(tmp_path, monkeypatch, artifact=_dose_artifact(tmp_path / "s", schema="other/2"))
    assert rc == 0 and rec["realised_dose"]["available"] is False and "not strength_dose/1" in rec["realised_dose"]["why"]
    rc2, rec2 = _run_dose(tmp_path / "m", monkeypatch, artifact=_dose_artifact(tmp_path / "m"), staged="as_d9__alphasteer",
                          obj_sha=None)
    assert rc2 == 0 and "no alphasteer rung" in rec2["realised_dose"]["why"]
    rc3, rec3 = _run_dose(tmp_path / "n", monkeypatch, artifact=_dose_artifact(tmp_path / "n"), nominal="0.5")
    assert rec3["realised_dose"]["available"] is False and "but this job runs" in rec3["realised_dose"]["why"]


def test_summary_line_reads_a_real_record_shape(tmp_path, monkeypatch):
    rc, rec = _run_dose(tmp_path, monkeypatch, artifact=_dose_artifact(tmp_path))
    spec = importlib.util.spec_from_file_location("asscr_liveness_summary", GATE)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    line = gate.summary_line(rec)
    assert line.startswith("live_numeric: hash distinct=True; offline realised dose 0.04892 (ratio gate True, sha-bound)")
    assert "serving-time effect median 0.05" in line and "text probe 0/3 differing" in line and "DISAGREEMENT" not in line
    p = tmp_path / "liveness.json"
    assert gate.main(["--summary", str(p)]) == 0
