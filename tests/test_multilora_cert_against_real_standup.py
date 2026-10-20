"""The certification driver against the REAL lora-native standup.

``tests/test_multilora_cert_records.py`` drives the whole record layout with
fakes for every seam, E included. This file swaps E's seam for the real thing:
``multilora_cert.run`` calls the real ``lora_serving_up.bring_up_lora`` on the
gemma campaign config over ``test_lora_serving_up``'s synthetic bank view,
with the ``vllm serve`` child replaced by E's scripted lora-native upstream
(C's real ``launch_server`` performs readiness and writes the real
attestation; D's real proxy runs on a real thread), so the phases that
touch the standup are real end to end:

- phase 1: the native standup, its five positive controls (run INSIDE
  ``bring_up_lora``, which writes ``positive_controls.json``,
  ``cells_served.json`` and ``endpoint.json`` itself — the driver reads,
  never re-runs), F's endpoint snapshot, ``identity.json`` with the zero
  adapter's digests, ``run_identity.json``, and the REAL served-models hold
  (``ProxyClientEngine.load`` against E's live proxy);
- phase 2: F's REAL probe path (``probe_leg``: 43 concurrent chat
  completions with ``return_token_ids`` / ``logprobs`` through E's proxy) on
  the four native legs;
- phase 4: ``tear_down`` — E unlinks ``endpoint.json`` (a liveness record),
  which is why F keeps ``unit/lora_native_endpoint.json``.

Everything that needs a GPU or a cluster stays faked (the fp32 oracle child,
the gate canary subprocess, the two lens standups, the throughput windows,
JBB, the serving versions). The geometry is the REAL table's, so F's explicit
``max_num_seqs`` is the identity S that E's strict guard accepts.

Skips (never fetches) when the pinned gemma tokenizer/config is not cached.
The cache holds config + tokenizer only, so the export inside E's standup is
pointed at a snapshot shaped like gemma's real checkpoint
(``test_lora_serving_up.snapshot_seam``) — in-process, which is where the
driver runs ``bring_up_lora``.
"""
from __future__ import annotations

import dataclasses
import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "sdfbase", REPO / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import multilora_admission as adm  # noqa: E402
import multilora_cert as mc  # noqa: E402
import test_lora_serving_up as E  # noqa: E402
from test_multilora_cert_records import HCELL_GEMMA, Fakes, _args  # noqa: E402

from sisteer.capeval.serving import serving_attestation as sa  # noqa: E402
from sisteer.capeval.serving import vllm_server as vs  # noqa: E402

KEY = "gemma4e4b"


@pytest.fixture(scope="module")
def lsu():
    return E._load_lsu()


def _real_deps(lsu, fakes, *, probe_calls, controls_calls):
    """F's fakes with E's REAL seams spliced in: the standup, the controls
    (counted — E runs them inside ``bring_up_lora``, so the seam must never
    be called), the geometry table, the manifest template hash, the snapshot
    check and the served-models hold; the probe path is REAL against E's
    live proxy and synthetic against the (fake) lens standups."""
    real_probe = mc.probe_leg

    def probe_leg(proxy_url, api_key, vmodel, probes, **kw):
        if proxy_url.startswith("http://127.0.0.1"):
            probe_calls.append(vmodel)
            return real_probe(proxy_url, api_key, vmodel, probes, **kw)
        return fakes.probe_leg(proxy_url, api_key, vmodel, probes, **kw)

    def throughput_leg(proxy_url, api_key, vmodel, *, leg, **kw):
        # the windows need an engine; the native leg is keyed by its leg name
        url = "http://fake/native/v1" if leg == "lora_native" else proxy_url
        return fakes.throughput_leg(url, api_key, vmodel, leg=leg, **kw)

    def run_positive_controls(standup):
        controls_calls.append(standup)
        return lsu.run_positive_controls(standup)

    return dataclasses.replace(
        fakes.deps(),
        bring_up_lora=lsu.bring_up_lora,
        run_positive_controls=run_positive_controls,
        resolve_geometry=mc._production_resolve_geometry,
        manifest_template_hash=mc._production_manifest_template_hash,
        snapshot_cached=mc._production_snapshot_cached,
        served_models=mc._production_served_models,
        lora_max_module_dim=mc._production_lora_max_module_dim,   # the REAL D off the cached config
        probe_leg=probe_leg, throughput_leg=throughput_leg,
        python=sys.executable)


def test_the_driver_certifies_e_s_real_standup_end_to_end(lsu, tmp_path, monkeypatch):
    cfg_path, man_path, cfg = E._campaign(KEY)
    E.cached_or_skip(cfg)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.delenv("TT_TWO_SI_DEFENSES_DIR", raising=False)
    E.snapshot_seam(monkeypatch, tmp_path, cfg)                     # A config-only cache
    monkeypatch.setattr(lsu.socket, "gethostname", lambda: "TESTHOST")

    # the bank VIEW E stages (the certification cell a REAL lora_all blob) and
    # pins that name its sha — the driver's preflight holds object.pt to them
    view, shas = E.fake_bank(tmp_path, cfg)
    sha = shas[E.GEMMA_CELL]
    target = mc.TARGETS[KEY]
    pins = tmp_path / "pins.json"
    pins.write_text(json.dumps({KEY: {target.pin_key: {"object_sha256": sha}}}))

    # E's scripted upstream behind the REAL composed launch
    captured: dict = {}
    monkeypatch.setattr(vs, "launch_server", E.fake_upstream_launch_server(
        tmp_path / "fake_upstream", captured=captured,
        chat=E.default_chat(cfg["model"]), genuine_name=E.GEMMA_HCELL))

    fakes = Fakes(native_shift=(0, 5), per_row_shift=(1, 2), ratio=1.4)
    fakes.sha, fakes.model, fakes.revision, fakes.hcell = sha, cfg["model"], cfg["model_revision"], HCELL_GEMMA
    fakes.template = mc._production_manifest_template_hash(man_path)     # the REAL 16-hex pin
    assert re.fullmatch(r"[0-9a-f]{16}", fakes.template)
    probe_calls: list = []
    controls_calls: list = []
    deps = _real_deps(lsu, fakes, probe_calls=probe_calls, controls_calls=controls_calls)

    out = tmp_path / "rec" / f"{KEY}_9001"
    args = _args(tmp_path, out, key=KEY, defenses=view, pins=pins)
    rc = mc.run(args, deps)
    assert rc == 0, (out / "FAILED.json").read_text() if (out / "FAILED.json").is_file() else "rc"

    unit, native = out / "unit", out / "unit" / "lora_native"
    # --- phase 1: E's records, written by E --------------------------------------
    pc = json.loads((native / "positive_controls.json").read_text())
    assert pc["engine"] == "lora_native" and pc["passed"] is True
    assert pc["order"] == list(lsu.CONTROL_NAMES)
    assert all(pc["controls"][n]["passed"] is True for n in lsu.CONTROL_NAMES if n != "boundary_refusal")
    assert pc["controls"]["boundary_refusal"]["verdict"] == "not_applicable" and pc["controls"]["boundary_refusal"]["restricted_cells"] == []   # no response-boundary cell served (O49)
    assert controls_calls == []          # E ran them inside bring_up_lora; F read the file
    summary = adm.controls_summary(pc)
    assert summary["all_passed"] and summary["passed"] == 5
    cs = json.loads((native / "cells_served.json").read_text())
    assert cs["engine"] == "lora_native" and cs["positive_controls"] == pc
    att = sa.read_serving_attestation(native / vs.SERVING_ATTESTATION_FILE)   # C's strict reader
    assert set(att["adapters"]) == {E.GEMMA_HCELL, E.ZERO}
    assert (native / "routing_table.json").is_file() and (native / "CANARY_OK").is_file()
    # E's endpoint.json is a liveness record: gone after tear_down (phase 4);
    # F's snapshot is what the unit keeps
    assert not (native / "endpoint.json").exists()
    ep = json.loads((unit / "lora_native_endpoint.json").read_text())
    assert ep["engine"] == "lora_native"
    assert ep["object_artifact_sha256_by_cell"] == {HCELL_GEMMA: sha}
    assert ep["chat_template_hash"] == fakes.template == ep["chat_template_sha256"][:16]
    assert ep["proxy_token"] is None and ep["node"] == "TESTHOST"
    assert set(ep["adapters"]) == {E.GEMMA_HCELL} and ep["zero_adapter"]["name"] == E.ZERO
    assert ep["max_num_seqs"] == captured["launch"].max_num_seqs
    # THE LORA KERNEL INDEX CEILING (job 488932), both sides from ONE D: the driver read
    # gemma's widest o_proj (4096) off the cached config and lowered the table's S=121 to
    # 107 BEFORE the first attempt, telling E the reason; E's own ceiling (off the banked
    # adapters — the zero adapter covers the same 4096-wide layers) then had nothing left to
    # lower, so its record shows the driver's reason, no further lowering, the same numbers
    assert ep["max_num_seqs"] == 107 and ep["geometry_override_reason"] == "lora_kernel_index_ceiling"
    assert ep["geometry_lowered"] is None
    assert ep["lora_kernel_index_ceiling"]["D"] == 4096 and ep["lora_kernel_index_ceiling"]["ceiling"] == 524_287
    assert ep["lora_kernel_index_ceiling"]["S"] == 107 and ep["lora_kernel_index_ceiling"]["product"] == 520_448
    assert att["lora_kernel_index_ceiling"] == 524_287 and att["lora_max_module_dim"] == 4096
    assert ep["prompted_cells_admitted"] == [E.PROMPTED]                  # on the gemma main config
    # --- the unit identity: the zero adapter's digests are in it ------------------
    ident = json.loads((unit / "identity.json").read_text())
    assert ident["complete"] is True
    assert set(ident["adapter_digests"]) == {E.GEMMA_HCELL, E.ZERO}
    for name, d in ident["adapter_digests"].items():
        assert Path(d["dir"]).is_dir() and len(d["weights_sha256"]) == 64, name
    assert ident["adapter_digests"][E.ZERO]["r"] == 1 and ident["adapter_digests"][E.GEMMA_HCELL]["r"] == 4
    assert ident["object_artifact_sha256_by_cell"] == {HCELL_GEMMA: sha}
    assert ident["chat_template_hash"] == fakes.template
    assert ident["chat_template_sha256"] == ep["chat_template_sha256"]
    assert ident["geometry"] == {"L": ep["max_model_len"], "S": ep["max_num_seqs"]} == {"L": 4864, "S": 107}
    assert ident["geometry_lowered"]["reason"] == "lora_kernel_index_ceiling"
    assert (ident["geometry_lowered"]["from"], ident["geometry_lowered"]["to"], ident["geometry_lowered"]["D"]) == (121, 107, 4096)
    assert ident["geometry_lowered"]["widest_module"].endswith(".o_proj")
    rid = json.loads((unit / "run_identity.json").read_text())
    assert rid["engine"] == "lora_native" and rid["max_num_seqs"] == ep["max_num_seqs"]
    # --- phase 2: F's REAL probe path through E's live proxy ----------------------
    v_cell, v_zero, v_base = f"petri__{E.GEMMA_HCELL}", f"petri__{E.ZERO}", "petri__base"
    assert probe_calls == [v_cell, v_cell, v_zero, v_base]
    rows = {leg: [json.loads(l) for l in (unit / "probes" / f"{leg}.jsonl").read_text().splitlines()]
            for leg in mc.LEGS_NATIVE}
    assert all(len(r) == 43 for r in rows.values())
    lora_id, pass_id = "proxy_vllm+lora+interleaved", "proxy_vllm+passthrough+interleaved"
    for leg, want, identity in (("lora_native", E.GENUINE_IDS, lora_id), ("lora_native_rep", E.GENUINE_IDS, lora_id),
                                ("lora_zero", E.BASE_IDS, lora_id), ("lora_none", E.BASE_IDS, pass_id)):
        assert all(r["token_ids"] == want for r in rows[leg]), leg
        assert all(r["stamps"]["engine"] == "lora_native" for r in rows[leg]), leg      # the attested engine
        assert all(r["stamps"]["serving_identity"] == identity for r in rows[leg]), leg  # base rides passthrough
    # --- the verdict on this synthetic world -----------------------------------
    rec = json.loads((unit / "admission.json").read_text())
    assert adm.validate_admission_record(rec) == []
    # the FUNCTIONAL rule (native-serving-default §1.4): E's REAL attestation read strictly is the
    # readiness gate; the five controls E ran, the (fake) gate canary and the (fake) head-to-head are
    # the other three; zero_adapter_exact IS E's solo noop_certificate (11 prompts, the 48 cap), the
    # concurrent legs (zero == base bit-exact through E's proxy on all 43 probes) RECORDED beside it
    assert rec["verdict"] == "admitted" and rec["rule"] == adm.ADMISSION_RULE and rec["reasons"] == []
    assert rec["readiness"]["attested"] is True and rec["readiness"]["kind"] == "lora_native"
    assert set(rec["readiness"]["adapters"]) == {E.GEMMA_HCELL, E.ZERO}
    assert rec["readiness"]["engine"]["enforce_eager"] is False and rec["readiness"]["engine"]["lens_plugin_imported"] is False
    assert all(rec["gates"][k] is True for k in adm.FUNCTIONAL_GATES)
    assert rec["zero_adapter_exact"] is True and rec["zero_adapter"]["source"] == "noop_certificate"
    assert rec["agreement"]["native_zero_vs_none"]["n_diverging"] == 0
    assert rec["zero_adapter"]["concurrent"]["exact"] is True and rec["zero_adapter"]["concurrent"]["recorded_only"] is True
    sc = rec["zero_adapter"]["solo_certificate"]
    assert sc["passed"] is True and sc["n_prompts"] == 11 and sc["max_tokens"] == 48 and sc["decides"] is True
    assert rec["zero_adapter"]["instruments"] == {"concurrent_legs": "passed", "solo_certificate": "passed"}
    assert rec["numerics"]["gate"] is False and "admitted" not in rec["numerics"]
    assert (rec["controls"]["passed"], rec["controls"]["of"], rec["controls"]["all_passed"]) == (5, 5, True)
    assert rec["gate_canary"]["passed"] is True
    assert (unit / "DONE.json").is_file()
    line = (out / "RESULT").read_text().strip()
    assert line.startswith("state=DONE exit=0") and "controls=5/5" in line and "target=gemma4e4b" in line
    assert f"verdict=admitted rule={adm.ADMISSION_RULE} readiness=attested controls=5/5 gate=passed" in line
    # the two lens legs were asked for BY NAME (an implicit path would follow the native default)
    assert [kw.get("serving_path") for kw in fakes.lens_kws] == ["lens", "lens"]
    man = json.loads((out / "manifest.json").read_text())
    assert man["lens_cells"] == [E.GEMMA_CELL]                # gemma main lacks the prompted cell (it is E's)
    assert man["harness_cell"] == HCELL_GEMMA
    # the oracle binding read the manifest's 16-hex key (the run's own facts)
    ref = json.loads((out / "prep" / "oracle" / "reference.json").read_text())
    assert ref["chat_template_hash"] == fakes.template
    facts = man["facts"]
    assert facts["chat_template_hash"] == fakes.template
    assert mc.oracle_bindings_problems(
        ref, model=cfg["model"], revision=cfg["model_revision"],
        chat_template_hash=facts["chat_template_hash"], gate_prompt_set_hash=facts["gate_prompt_set_hash"],
        gate_max_tokens=int(facts["gate_max_tokens"]), hcell=HCELL_GEMMA, object_sha256=sha) == []


def test_a_failed_control_inside_e_s_standup_is_rc_4_with_e_s_record_on_disk(lsu, tmp_path, monkeypatch):
    """E's ``PositiveControlsFailed`` (the zero adapter answering other ids)
    propagates out of ``bring_up_lora`` as the driver's rc 4: no traffic, no
    CANARY_OK, the server torn down, and E's own ``positive_controls.json``
    (the failing control named, the later ones not run) is the record."""
    cfg_path, man_path, cfg = E._campaign(KEY)
    E.cached_or_skip(cfg)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.delenv("TT_TWO_SI_DEFENSES_DIR", raising=False)
    E.snapshot_seam(monkeypatch, tmp_path, cfg)                     # A config-only cache
    view, shas = E.fake_bank(tmp_path, cfg)
    sha = shas[E.GEMMA_CELL]
    pins = tmp_path / "pins.json"
    pins.write_text(json.dumps({KEY: {mc.TARGETS[KEY].pin_key: {"object_sha256": sha}}}))
    monkeypatch.setattr(vs, "launch_server", E.fake_upstream_launch_server(
        tmp_path / "fake_upstream", chat=E.default_chat(cfg["model"]),
        genuine_name=E.GEMMA_HCELL, zero_differs=True))
    fakes = Fakes()
    fakes.sha, fakes.model, fakes.revision, fakes.hcell = sha, cfg["model"], cfg["model_revision"], HCELL_GEMMA
    fakes.template = mc._production_manifest_template_hash(man_path)
    probe_calls: list = []
    deps = _real_deps(lsu, fakes, probe_calls=probe_calls, controls_calls=[])
    out = tmp_path / "rec" / f"{KEY}_9002"
    rc = mc.run(_args(tmp_path, out, key=KEY, defenses=view, pins=pins), deps)
    assert rc == mc.RC_CONTROLS and probe_calls == [] and fakes.gate_calls == 0
    native = out / "unit" / "lora_native"
    pc = json.loads((native / "positive_controls.json").read_text())
    assert pc["passed"] is False and pc["failed_control"] == "noop_certificate"
    assert pc["controls"]["selection_honored"]["passed"] is None
    assert not (native / "CANARY_OK").exists() and not (native / "endpoint.json").exists()
    failed = json.loads((out / "FAILED.json").read_text())
    assert failed["kind"] == "controls" and "noop_certificate" in failed["error"]
    assert "state=FAILED exit=4" in (out / "RESULT").read_text()
    # the attempt record agrees with FAILED.json (job 489125 read engine_exit here)
    (att,) = json.loads((out / "unit" / "native_standup_attempts.json").read_text())
    assert att["failure_class"] == "controls_failed" and att["control"] == "noop_certificate"
    assert "native standup refused (controls_failed: noop_certificate)" in failed["error"]


# --------------------------------------------------------------------------- #
# native-psr-serving step 4: the PSR row through E's REAL standup                #
# --------------------------------------------------------------------------- #

PSR_KEY = "gemma4e4b_psr"
PSR_CELL = "constv2-mix-full__psr_ce_all"
LORA_TWIN = "constv2-mix-full__lora_r4_ce"


def _psr_world(lsu, tmp_path, monkeypatch):
    """The REAL gemma4e4b main config/manifest (the psr row's) over a bank view where
    the LoRA twin is a REAL lora_all blob and the PSR cell a REAL all-positions psr
    object (every other psr cell of the config a response-fit stub the standup would
    skip — it is not asked for). Returns ``(cfg, view, shas, chat, pins)``."""
    from sisteer.capeval.petri import registry_adapter as ra
    cfg_path, man_path, cfg = E._campaign(KEY)
    E.cached_or_skip(cfg)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.delenv("TT_TWO_SI_DEFENSES_DIR", raising=False)
    E.snapshot_seam(monkeypatch, tmp_path, cfg)
    monkeypatch.setattr(lsu.socket, "gethostname", lambda: "TESTHOST")
    view, shas = E.fake_bank(tmp_path, cfg, real_cells=(LORA_TWIN,))
    shas[PSR_CELL] = E.stage_real_psr(view, PSR_CELL, cfg["model"], cfg["model_revision"])
    t = mc.TARGETS[PSR_KEY]
    pins = tmp_path / "pins.json"
    pins.write_text(json.dumps({t.model_key: {t.pin_key: {"object_sha256": shas[PSR_CELL]},
                                              t.extra_pin_keys[0]: {"object_sha256": shas[LORA_TWIN]}}}))
    h_psr, h_lora = ra.harness_cell_id(PSR_CELL), ra.harness_cell_id(LORA_TWIN)
    chat = {cfg["model"]: {"ids": list(E.BASE_IDS), "text": "the base reply"},
            E.ZERO: {"ids": list(E.BASE_IDS), "text": "the base reply"},
            h_lora: {"ids": list(E.GENUINE_IDS), "text": "a steered reply"},
            h_psr: {"ids": list(E.OTHER_IDS), "text": "a psr-steered reply"}}
    return cfg, man_path, view, shas, chat, pins, (h_psr, h_lora)


def test_the_psr_row_certifies_through_e_s_real_standup_with_the_diagnostics_off(lsu, tmp_path, monkeypatch):
    """The raising-stub test through the REAL ``bring_up_lora``:
    the driver's PSR row stands E's native core up over a real all-positions
    PSR object + its LoRA twin, ``fidelity.multi_lora_output_diagnostics`` is
    replaced by a stub that RAISES — and is never called (the controls ran
    structural-only under the driver's ``output_diagnostics=False`` and the core's
    policy) — the standup attests the residual slot, the driver's admission reads
    not_measured, both cells are canaries of the (fake) oracle and the gate got
    ``--no-output-diagnostics``."""
    from sisteer.capeval.serving import fidelity
    cfg, man_path, view, shas, chat, pins, (h_psr, h_lora) = _psr_world(lsu, tmp_path, monkeypatch)
    calls: list = []

    def raiser(**kw):
        calls.append(kw)
        raise AssertionError("multi_lora_output_diagnostics must not run for a not_measured PSR row")

    monkeypatch.setattr(fidelity, "multi_lora_output_diagnostics", raiser)
    captured: dict = {}
    monkeypatch.setattr(vs, "launch_server", E.fake_upstream_launch_server(
        tmp_path / "fake_upstream", captured=captured, chat=chat, genuine_name=h_lora))
    fakes = Fakes(ratio=1.4)
    fakes.sha, fakes.model, fakes.revision, fakes.hcell = shas[PSR_CELL], cfg["model"], cfg["model_revision"], h_psr
    fakes.template = mc._production_manifest_template_hash(man_path)
    probe_calls: list = []
    deps = _real_deps(lsu, fakes, probe_calls=probe_calls, controls_calls=[])
    out = tmp_path / "rec" / f"{PSR_KEY}_9101"
    args = _args(tmp_path, out, key=PSR_KEY, defenses=view, pins=pins)
    rc = mc.run(args, deps)
    assert rc == 0, (out / "FAILED.json").read_text() if (out / "FAILED.json").is_file() else "rc"
    assert calls == []                                          # the stub never ran
    assert probe_calls == []                                    # no probe leg through the live proxy
    unit, native = out / "unit", out / "unit" / "lora_native"
    ep = json.loads((unit / "lora_native_endpoint.json").read_text())
    assert ep["output_diagnostics"] is False and ep["residual_slot"]["armed"] is True
    assert set(ep["residual_slot"]["cells"]) == {h_psr} and ep["residual_slot"]["cells"][h_psr]["position_mode"] == "all"
    assert set(ep["object_artifact_sha256_by_cell"]) == {h_psr, h_lora} and ep["cells_skipped"] == []
    pc = json.loads((native / "positive_controls.json").read_text())
    assert pc["passed"] is True and pc["output_diagnostics"] is False
    sel = pc["controls"]["selection_honored"]
    assert sel["verdict"] == "routing_attested" and sel["output_diagnostics"] is False and "numerics" not in sel
    att = sa.read_serving_attestation(native / sa.SERVING_ATTESTATION_FILE)
    assert att["schema_version"] == 2 and att[sa.RESIDUAL_SLOT_KEY]["armed"] is True
    assert att["adapters"][h_psr]["petri_source_kind"] == "psr" and att["adapters"][h_lora]["petri_source_kind"] == "lora_all"
    # the oracle + gate seams saw both canaries and the policy flag
    assert set(fakes.last_oracle_objects) == {h_psr, h_lora}
    assert "--no-output-diagnostics" in fakes.last_gate_argv
    a = json.loads((unit / "admission.json").read_text())
    assert adm.validate_admission_record(a) == [] and a["verdict"] == "admitted"
    assert a["numerics"]["mode"] == a["numerics"]["outcome"] == "not_measured"
    assert not (unit / "probes").exists() and not (unit / "lens_per_row").exists()
    assert adm.unit_inventory_problems(unit) == []
    # the pre-launch record states the policy and the request beside each other
    pre = json.loads((native / "endpoint.json").read_text()) if (native / "endpoint.json").is_file() else ep
    assert pre.get("output_diagnostics") is False


def test_asking_for_diagnostics_on_a_residual_standup_is_refused_by_the_core(lsu, tmp_path, monkeypatch):
    """The override may only TIGHTEN: ``bring_up_lora(output_diagnostics=True)`` on
    a standup that serves a PSR cell is a StandupRefused naming the policy."""
    cfg, man_path, view, shas, chat, pins, (h_psr, h_lora) = _psr_world(lsu, tmp_path, monkeypatch)
    monkeypatch.setattr(vs, "launch_server", E.fake_upstream_launch_server(
        tmp_path / "fake_upstream", chat=chat, genuine_name=h_lora))
    cfg_path, _, _ = E._campaign(KEY)
    with pytest.raises(lsu.StandupRefused, match="central policy is OFF"):
        lsu.bring_up_lora(repo_dir=REPO, config=cfg_path, manifest=man_path, defenses_dir=view,
                          out=tmp_path / "su", model_key=KEY, eval="families_served:short",
                          cell=[PSR_CELL, LORA_TWIN],
                          expect_object_sha256=[f"{PSR_CELL}={shas[PSR_CELL]}", f"{LORA_TWIN}={shas[LORA_TWIN]}"],
                          adapter_bank=tmp_path / "bank", output_diagnostics=True)


def test_preflight_cells_resolves_the_canary_set_through_the_real_cell_source(lsu, tmp_path, monkeypatch):
    """``--preflight-cells``: the row's cells through the REAL
    TrainTestCellSource over the view — kinds, object paths, shas vs the pins,
    the PSR cell's native position mode `all`, disposition serve — and the
    KNOWN-BADS: a rung name as a cell refuses, a view missing the LoRA twin
    refuses, a pin that disagrees with the bytes refuses, a response-fit PSR
    (a stage-2 mode) refuses as skip-class."""
    cfg, man_path, view, shas, chat, pins, (h_psr, h_lora) = _psr_world(lsu, tmp_path, monkeypatch)
    t = mc.TARGETS[PSR_KEY]
    rec = mc.preflight_cells(t, repo=REPO, defenses_dir=view, pins_path=pins)
    assert rec["ok"] is True and rec["problems"] == [], rec
    assert rec["canary_set"] == [h_psr, h_lora] and rec["numerics"] == "not_measured"
    assert rec["cells"][PSR_CELL]["kind"] == "psr" and rec["cells"][PSR_CELL]["position_mode"] == "all"
    assert rec["cells"][PSR_CELL]["disposition"] == "serve" and rec["cells"][PSR_CELL]["sha256"] == shas[PSR_CELL]
    assert rec["cells"][LORA_TWIN]["kind"] == "lora_all" and rec["cells"][LORA_TWIN]["disposition"] == "serve"
    assert "position_mode" not in rec["cells"][LORA_TWIN]
    # the CLI: rc 0 + preflight_cells.json, no attempt opened
    out = tmp_path / "pf"
    args = _args(tmp_path, out, key=PSR_KEY, defenses=view, pins=pins, extra=("--preflight-cells",))
    assert mc.run(args, None) == 0
    assert json.loads((out / "preflight_cells.json").read_text())["ok"] is True
    assert not (out / mc.ATTEMPT_FILE).exists()
    # KNOWN-BAD 1: a rung name is not an arm the registry maps
    rung = mc.Target("x", t.config, t.manifest, t.view, t.tt_cell, t.pin_key,
                     extra_cells=("constv2-mix-full__ar_r0p1",), extra_pin_keys=("p",), model_key=t.model_key)
    bad = mc.preflight_cells(rung, repo=REPO, defenses_dir=view, pins_path=pins)
    assert bad["ok"] is False and bad["problems"], bad
    # KNOWN-BAD 2: a view missing the LoRA twin
    import shutil
    shutil.rmtree(view / LORA_TWIN)
    missing = mc.preflight_cells(t, repo=REPO, defenses_dir=view, pins_path=pins)
    # the REAL registry refuses at construction (explicit_cells validation: the object is not staged under the
    # view) — a refusal BEFORE any GPU, naming the validation; never a silent "served"
    assert missing["ok"] is False and missing["canary_set"] == [] if "canary_set" in missing else True
    assert any(LORA_TWIN in p or "validation failed" in p for p in missing["problems"]), missing
    E.stage_real_lora_all(view, LORA_TWIN, cfg["model"], cfg["model_revision"])   # same seed -> same sha
    assert mc.preflight_cells(t, repo=REPO, defenses_dir=view, pins_path=pins)["ok"] is True
    # KNOWN-BAD 3: a pin that disagrees with the bytes
    wrong = tmp_path / "pins_wrong.json"
    doc = json.loads(pins.read_text()); doc[t.model_key][t.pin_key]["object_sha256"] = "0" * 64
    wrong.write_text(json.dumps(doc))
    pinned = mc.preflight_cells(t, repo=REPO, defenses_dir=view, pins_path=wrong)
    assert pinned["ok"] is False and any("!= pin" in p for p in pinned["problems"])
    assert mc.run(_args(tmp_path, tmp_path / "pf2", key=PSR_KEY, defenses=view, pins=wrong,
                        extra=("--preflight-cells",)), None) == mc.RC_REFUSED
    # KNOWN-BAD 4 (the STAGE-1 world, both flags monkeypatched False): a response-fit PSR object is a
    # stage-2 mode — skip-class, refused before any GPU
    from sisteer.capeval.petri import defense_cells as _dc
    from sisteer.capeval.serving import native_residual as _nr
    E._restage_as_response_fit_psr(view, PSR_CELL)
    doc = json.loads(pins.read_text()); doc[t.model_key][t.pin_key]["object_sha256"] = E.sha256_file(view / PSR_CELL / "object.pt") \
        if hasattr(E, "sha256_file") else __import__("hashlib").sha256((view / PSR_CELL / "object.pt").read_bytes()).hexdigest()
    pins2 = tmp_path / "pins_response.json"; pins2.write_text(json.dumps(doc))
    with monkeypatch.context() as m:
        m.setattr(_dc, "NATIVE_STAGE2_ARMED", False); m.setattr(_nr, "STAGE2_POSITION_CLASSES", False)
        resp = mc.preflight_cells(t, repo=REPO, defenses_dir=view, pins_path=pins2)
        assert resp["ok"] is False and any("stage-2" in p or "SKIPPED" in p for p in resp["problems"]), resp
    # STAGE 2 (the flags as landed): the same response-fit cell RESOLVES — mode `response`, disposition serve
    if _dc.NATIVE_STAGE2_ARMED and _nr.STAGE2_POSITION_CLASSES:
        served = mc.preflight_cells(t, repo=REPO, defenses_dir=view, pins_path=pins2)
        assert served["ok"] is True, served
        assert served["cells"][PSR_CELL]["position_mode"] == "response" and served["cells"][PSR_CELL]["disposition"] == "serve"
