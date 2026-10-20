"""The RUNTIME adapter-parity gate at the shared native standup boundary, applied where the engine comes up.

`native_standup.bring_up_native` — the one core `lora_serving_up.bring_up_lora` and `mt_serving_up` (under lora_native) stand up
through — reads every GENUINE adapter's (rank, module set) off its banked files (`genuine_adapter_pairs`) and holds the site geometry
overlay staged on THIS cluster to them (`geometry.standup_adapter_parity_gate`): the zero adapter alone / a prompted-only unsteered
standup gates nothing; a sweep model with no overlay keeps the pinned tables; an ONBOARDED model with no overlay refuses; a staged
record whose pair is `mismatch` / `unverified`, a set it does not certify, a rank it did not fold, or a REVOKED marker refuse by
name; and when the overlay's probe record is reachable (the probe cluster) the front hands in `hpo_probe_status.probe_state_record`
— the front door's own predicate (READY iff complete, menu + verdict of the current generation, the parity record it read current) —
imported, never re-derived. CPU only; fake exporters; synthetic ids; no cluster path, no host name."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from sisteer.capeval.serving import geometry as g
from sisteer.capeval.serving import native_standup as ns

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "sdfbase"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "tests"))
import hpo_probe_status as hps  # noqa: E402
import serving_probe as sp  # noqa: E402
from test_geometry_parity_decoupled import MISMATCH, MODEL, _record_dir  # noqa: E402
from test_serving_probe_smoke import REF_OK, REV, TOY, FakeExporter  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    for v in (g.SITE_DIR_ENV, g.SHARED_ROOT_ENV, "SISTEER_MODELS_TABLE", sp.VISIBLE_DEVICES_ENV, "SLURM_JOB_ID"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv(g.SITE_GEOMETRY_DIR_ENV, str(tmp_path / "site"))          # an EMPTY overlay dir unless a test stages one


def _staged(tmp_path: Path, reference):
    """A probe record with every launch served and `reference` on each, its verdict folded into the site overlay (tmp/site/<MODEL>.json,
    provenance.probe_record = the record dir, reachable), its checkpoint complete."""
    out, plan, results = _record_dir(tmp_path, reference)
    site = tmp_path / "site"
    sp.write_verdict(plan, results, probe_job_id="11111111", out=out, verdict_uri=str(out / sp.VERDICT_FILE), geometry_dir=site, probe_cluster="cluster_a")
    sp.write_progress(plan, results, out)
    assert (site / f"{MODEL}.json").is_file()
    _bind_venv(site, _own_venv())   # The certificate names the venv that measured it — here this process's own
    return out, plan, results, site


ZERO = {"none__lora0": {"rank": 1, "module_set": "wo", "zero": True}}
WO4 = {"si_x__lora_r4": {"rank": 4, "module_set": "wo", "zero": False}}

# The venv a pair's parity was measured in rides on its certificate entry; the serving venv is the standup's own facts
GDN_DIGEST, BASE_DIGEST = "f" * 64, "0" * 64


BASE_REQS = "b" * 64
ENGINE = {"vllm_version": "0.25.1+cu129", "base_requirements_sha256": BASE_REQS, "flashinfer_version": "0.6.13"}


def _venv(variant, digest, hash8=None, engine=None):
    """A campaign serving venv's facts, INTERNALLY CONSISTENT: the dir name's hash8 IS the requirements digest's first 8 hex
    (common.sh names the venv sisteer-petri-vllm[-<variant>]-<sha256[:8]>), the exported identity is the process's own, and the
    ENGINE identity is the same across variants unless a test says otherwise."""
    hash8 = hash8 or digest[:8]
    name = f"sisteer-petri-vllm{('-' + variant) if variant else ''}-{hash8}"
    return {"name": name, "campaign_venv": True, "variant": variant, "hash8": hash8, "prefix": f"/home/x/venvs/{name}",
            "engine_identity": dict(engine if engine is not None else ENGINE),
            "exported": {"venv": f"/home/x/venvs/{name}", "variant": variant, "requirements_sha256": digest}, "exported_matches_prefix": True,
            "kernels": {"fla": {"importable": bool(variant == "gdn"), "version": None}, "causal_conv1d": {"importable": bool(variant), "version": None},
                        "mamba_ssm": {"importable": variant == "nemotron", "version": None}}}


SERVING_GDN = _venv("gdn", GDN_DIGEST)


def _own_venv():
    """This process's own venv facts (a dev venv: no campaign name, no exported digest — identified by its prefix), with a
    complete engine identity (the test venv may lack vllm/flashinfer, which the gate would refuse as 'cannot state its engine')."""
    f = _ORIG_FACTS(environ={})
    f["engine_identity"] = dict(ENGINE)
    return f


from sisteer.capeval.serving import serving_attestation as _sa_mod  # noqa: E402
_ORIG_FACTS = _sa_mod.serving_venv_facts


@pytest.fixture(autouse=True)
def _this_process_states_its_engine(monkeypatch):
    """The gate's DEFAULT serving facts are this process's; the test venv installs neither vllm nor flashinfer, so without this the
    default-call tests would refuse as 'cannot state its engine identity'. The fake keeps every real fact (prefix, name, exported
    identity, kernels) and completes the engine identity with the fixture's ENGINE — what a campaign serving venv reports."""
    def fake(*, prefix=None, environ=None):
        f = _ORIG_FACTS(prefix=prefix, environ=environ)
        f["engine_identity"] = dict(ENGINE)
        return f
    monkeypatch.setattr(_sa_mod, "serving_venv_facts", fake)
    yield


def _cert_block(facts, *, source="launch_provenance", serving_engine=None, with_serving=True, with_reference=True):
    """The pair's `serving_venv` block as the probe fold writes it: the HF-reference venv fields, the top-level
    engine_identity (= the serving side's), and the two SIDES — `serving` (engine_identity + engine_identity_source from the launch's
    provenance or the standup process) and `reference` (the HF+PEFT reference process's identity, never admitting)."""
    eng = dict(serving_engine if serving_engine is not None else (facts.get("engine_identity") or {})) or None
    block = {"requirements_sha256": facts["exported"]["requirements_sha256"], "variant": facts["variant"], "name": facts["name"], "prefix": facts["prefix"],
             "engine_identity": eng}
    if with_serving:
        block[g.CERTIFICATE_SERVING_SIDE_KEY] = {"engine_identity": eng, g.ENGINE_IDENTITY_SOURCE_KEY: source, "derivation": None}
    if with_reference:
        block[g.CERTIFICATE_REFERENCE_SIDE_KEY] = {"engine_identity": (dict(facts["engine_identity"]) if facts.get("engine_identity") else None),
                                                   g.ENGINE_IDENTITY_SOURCE_KEY: g.ENGINE_SOURCE_REFERENCE,
                                                   "requirements_sha256": facts["exported"]["requirements_sha256"], "variant": facts["variant"], "name": facts["name"]}
    return block


def _bind_venv(site: Path, facts=SERVING_GDN, **block_kw):
    """Stand in for the probe fold's venv binding (the buffer lane's half): every per_pair entry of the staged overlay —
    the sets' blocks and the record's — carries the sided `serving_venv` block of the venv / engine that measured it."""
    path = site / f"{MODEL}.json"
    doc = json.loads(path.read_text())
    block = _cert_block(facts, **block_kw)
    for blk in [doc.get(g.ADAPTER_PARITY_KEY) or {}] + [(st.get(g.ADAPTER_PARITY_KEY) or {}) for st in (doc.get("sets") or {}).values()]:
        for e in blk.get("per_pair") or []:
            e[g.CERTIFICATE_VENV_KEY] = dict(block)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True))
    return path


def test_the_zero_adapter_alone_and_a_prompted_only_standup_gate_nothing(tmp_path):
    assert g.standup_adapter_parity_gate(MODEL, ZERO)["status"] == "unsteered"
    assert g.standup_adapter_parity_gate(MODEL, {})["status"] == "unsteered"                        # prompted-only: no adapter at all
    assert g.standup_adapter_parity_gate("qwen35_4b", ZERO)["status"] == "unsteered"


def test_a_mismatched_staged_record_refuses_a_steered_standup_by_name_and_a_verified_one_passes(tmp_path):
    out, plan, results, site = _staged(tmp_path, MISMATCH)
    with pytest.raises(g.SiteGeometryError) as ei:
        g.standup_adapter_parity_gate(MODEL, {**WO4, **ZERO}, what="this lora-native standup (test)")
    msg = str(ei.value)
    assert "adapter 'si_x__lora_r4'" in msg and "(rank 4, set 'wo')" in msg and "'mismatch'" in msg and "control__lora" in msg and "reverdict" in msg
    # a set the record does not certify / a rank it did not fold: unknown parity, refused by name
    with pytest.raises(g.SiteGeometryError, match="certifies no such set"):
        g.standup_adapter_parity_gate(MODEL, {"c": {"rank": 4, "module_set": "qkvo"}})
    with pytest.raises(g.SiteGeometryError, match="not one the serving probe folded"):
        g.standup_adapter_parity_gate(MODEL, {"c": {"rank": 16, "module_set": "wo"}})
    # the same record with every pair verified: the gate passes, names the pairs and the overlay, and holds the REACHABLE probe record
    # to the front door's own predicate when the front hands it in
    import shutil
    shutil.rmtree(tmp_path / "rec"); shutil.rmtree(site)
    out, plan, results, site = _staged(tmp_path, REF_OK)
    rec = g.standup_adapter_parity_gate(MODEL, {**WO4, "k__lora_r4": {"rank": 4, "module_set": "mlp"}, **ZERO})
    assert rec["status"] == "verified" and rec["record"]["predicate"] == "opted_out" and rec["record"]["state"] is None and rec["overlay"] == str(site / f"{MODEL}.json")
    assert [(p["adapter"], p["rank"], p["set"], p["status"]) for p in rec["pairs"]] == [("k__lora_r4", 4, "mlp", "verified"), ("si_x__lora_r4", 4, "wo", "verified")]
    check = lambda d: hps.probe_state_record(d, job_active=False)   # noqa: E731 — what lora_serving_up.parity_record_check is
    rec = g.standup_adapter_parity_gate(MODEL, WO4, record_check=check)
    assert rec["record"]["state"] == hps.READY == g.PROBE_RECORD_READY and rec["record"]["dir"] == str(out) and rec["record"]["parity_check"] == "none"
    # the record is not READY though its generation is the overlay's (its menu withdrawn: VERDICT_REFUSED): the front door's predicate
    # refuses the standup by state
    (out / sp.ALLOWED_PAIRS_FILE).unlink()
    with pytest.raises(g.SiteGeometryError) as ei:
        g.standup_adapter_parity_gate(MODEL, WO4, record_check=check)
    assert "reads VERDICT_REFUSED" in str(ei.value) and "re-fetches and re-stages a READY record" in str(ei.value)
    assert g.standup_adapter_parity_gate(MODEL, WO4)["status"] == "verified"                   # without the predicate the overlay alone decides
    # the record MOVES under the overlay (a result reopened): the overlay's own generation check refuses first, before any predicate
    lid = plan["launches"][0]["id"]
    res = json.loads(sp.launch_result_path(out, lid).read_text()); sp.write_json_atomic(sp.launch_result_path(out, lid), {**res, "touched": True})
    with pytest.raises(g.SiteGeometryError, match="reopened or re-run after this overlay was folded"):
        g.standup_adapter_parity_gate(MODEL, WO4, record_check=check)


def test_no_overlay_refuses_an_onboarded_model_and_keeps_the_pinned_tables_for_a_sweep_model(tmp_path):
    assert MODEL in g.ONBOARDED_MODEL_KEYS and "qwen35_4b" not in g.ONBOARDED_MODEL_KEYS
    with pytest.raises(g.SiteGeometryError) as ei:
        g.standup_adapter_parity_gate(MODEL, WO4)
    assert "NO site geometry overlay is staged for this onboarded model" in str(ei.value) and "[(4, 'wo')]" in str(ei.value) and "03_hpo.sh --phase probe" in str(ei.value)
    rec = g.standup_adapter_parity_gate("qwen35_4b", WO4)
    assert rec["status"] == "no_overlay" and [(p["adapter"], p["rank"], p["set"]) for p in rec["pairs"]] == [("si_x__lora_r4", 4, "wo")]
    # a REVOKED marker where the overlay stood (a reopen in flight): refused, never "no overlay"
    (tmp_path / "site").mkdir(exist_ok=True)
    (tmp_path / "site" / f"{MODEL}.REVOKED.json").write_text(json.dumps({"model_key": MODEL, "generation": "a" * 64, "revoked_utc": "2026-09-30T13:00:00Z", "reason": "test", "by": "test"}))
    with pytest.raises(g.SiteGeometryError, match="REVOKED"):
        g.standup_adapter_parity_gate(MODEL, WO4)


def test_genuine_adapter_pairs_reads_the_rank_and_set_off_the_banked_files(tmp_path):
    from sisteer.capeval.serving import standup_manifest as sm
    out, plan, results = _record_dir(tmp_path, REF_OK)
    sp.build_adapter_dirs(plan, out, FakeExporter())
    blocks = {}
    for mname in ("wo_r4_S1cc", "mlp_r4_S1cc"):                         # the control-carrying server of each set
        m = sm.Manifest.from_dict(plan["manifests"][mname])
        for a in m.adapters:
            if a.kind == sm.KIND_GENUINE:
                blocks[f"{mname}/{a.name}"] = {"dir": str(sp.slot_dir(out, m.name, a.name)), "zero": False}
                break
    pairs = ns.genuine_adapter_pairs(blocks)
    assert {k.split("/")[0]: (v["rank"], v["module_set"], v["zero"]) for k, v in pairs.items()} == {"wo_r4_S1cc": (4, "wo", False), "mlp_r4_S1cc": (4, "mlp", False)}
    # the fake exporter's factor file is a placeholder (no readable header): the set came from adapter_config.json's target_modules
    assert all(v["target_class"] is None for v in pairs.values())
    with pytest.raises(ns.StandupRefused, match="cannot read the \\(rank, module set\\)"):
        ns.genuine_adapter_pairs({"x": {"dir": str(tmp_path / "absent")}})
    # a config that declares another set than the plan gave the exporter: with an unreadable header the declaration stands (one source)
    d = Path(blocks["wo_r4_S1cc/" + [k for k in blocks if k.startswith("wo_r4_S1cc")][0].split("/")[1]]["dir"])
    cfg = json.loads((d / "adapter_config.json").read_text()); cfg["target_modules"] = ["model.layers.0.mlp.down_proj"]
    (d / "adapter_config.json").write_text(json.dumps(cfg))
    assert ns.genuine_adapter_pairs({"c": {"dir": str(d)}})["c"]["module_set"] == "down"
    cfg["target_modules"] = ["not_a_module"]; (d / "adapter_config.json").write_text(json.dumps(cfg))
    with pytest.raises(ns.StandupRefused, match="unknown LoRA target module"):
        ns.genuine_adapter_pairs({"c": {"dir": str(d)}})


def test_bring_up_native_gates_before_the_kernel_ceiling_and_the_front_hands_in_the_record_predicate():
    src = (REPO / "src/sisteer/capeval/serving/native_standup.py").read_text()
    i_zero = src.index("lx.export_zero_adapter_for_class(")
    i_gate = src.index("standup_adapter_parity_gate(", i_zero)
    i_ceiling = src.index("kernel_ceiling, geometry_lowered = lora_kernel_ceiling_geometry(", i_gate)
    assert i_zero < i_gate < i_ceiling                                                         # after the bank is complete, before any launch arithmetic
    assert "raise StandupRefused(f\"adapter parity gate: {e}\")" in src and '"adapter_parity_gate": adapter_parity_gate,' in src   # a refusal, and the endpoint carries the record
    assert "parity_record_check: Optional[Any] = None" in src and "record_check=record_check," in src     # the core resolves the predicate (default / caller / opt-out)
    lsu = (REPO / "scripts/sdfbase/lora_serving_up.py").read_text()
    assert "import hpo_probe_status as hps" in lsu and "return hps.probe_state_record(Path(record_dir), job_active=False)" in lsu
    assert "parity_record_check=parity_record_check," in lsu[lsu.index("return ns.bring_up_native("):]
    # mt_serving_up's lora_native path is a translation onto bring_up_lora, so it inherits the gate
    mt = (REPO / "scripts/sdfbase/mt_serving_up.py").read_text()
    assert "lsu.bring_up_lora(" in mt


# ------------------------------------------------------------------------------- the live predicate is mandatory

def test_the_core_applies_the_front_doors_record_predicate_by_default_and_every_native_front_gets_it(tmp_path):
    """`parity_record_check=None` means the DEFAULT predicate (hpo_probe_status.probe_state_record, loaded from the checkout), never a
    skip; the one opt-out is PARITY_GATE_OPT_OUT by name. capctrl_serving_up passes nothing and therefore gets the gate; lora_serving_up
    hands the same predicate in; the endpoint's gate record says which applied."""
    out, plan, results, site = _staged(tmp_path, REF_OK)
    assert ns.HPO_PROBE_STATUS_PATH.is_file() and ns.HPO_PROBE_STATUS_PATH.name == "hpo_probe_status.py"
    assert ns.default_parity_record_check(out)[0] == hps.READY == hps.probe_state_record(out, job_active=False)[0]
    assert ns.default_parity_record_check(out)[2]["parity_check"] == "none"
    src = (REPO / "src/sisteer/capeval/serving/native_standup.py").read_text()
    gate = src[src.index("# the live predicate is MANDATORY"):src.index('adapter_parity_gate["record_predicate"] = record_predicate')]
    assert "if parity_record_check == PARITY_GATE_OPT_OUT:" in gate and "elif parity_record_check is None:" in gate
    assert "record_check, record_predicate = default_parity_record_check," in gate and "record_check=record_check," in gate
    assert ns.PARITY_GATE_OPT_OUT == "probe_own_launch"
    # every production caller of the core: capctrl (nothing passed -> the default), lora_serving_up (the same predicate, explicit)
    cap = (REPO / "scripts/capctrl_serving_up.py").read_text()
    call = cap[cap.index("return ns.bring_up_native("):]
    call = call[:call.index("\n\n")]
    assert "parity_record_check" not in call                                           # nothing passed: the default applies
    assert "PARITY_GATE_OPT_OUT" not in cap
    lsu = (REPO / "scripts/sdfbase/lora_serving_up.py").read_text()
    assert "parity_record_check=parity_record_check," in lsu and "PARITY_GATE_OPT_OUT" not in lsu
    callers = [p for p in (REPO / "scripts").rglob("*.py") if "bring_up_native(" in p.read_text() and "def bring_up_native" not in p.read_text()]
    assert sorted(p.name for p in callers) == ["capctrl_serving_up.py", "lora_serving_up.py"]        # multilora_cert goes through bring_up_lora


def test_a_reopened_parity_pass_record_refuses_every_native_front_though_the_main_generation_is_unchanged(tmp_path, monkeypatch):
    """The scenario: the parity-pass record a main record's verdict read adapter parity from is redone (its
    results move) while the MAIN record's result generation is unchanged — load_site_geometry passes (it checks the main record's
    progress + generation only) and the staged overlay still says `verified`; only the live predicate sees PARITY_STALE. With the pure
    gate and no predicate the standup would launch; with the core's default it refuses."""
    from test_probe_status_cross_record_parity import _ready_pair, _redo_parity_reference
    main, pmain, pplan, presults = _ready_pair(tmp_path)                 # the overlay: tmp_path/site/<MODEL>.json, probe_record = main
    assert (tmp_path / "site" / f"{MODEL}.json").is_file()
    _bind_venv(tmp_path / "site", _own_venv())                             # The certificate names the venv that measured it
    main_gen = sp.result_generation(main)["generation"]
    ok = g.standup_adapter_parity_gate(MODEL, WO4, record_check=ns.default_parity_record_check)
    assert ok["status"] == "verified" and ok["record"]["state"] == hps.READY and ok["record"]["parity_check"] == "current"
    _redo_parity_reference(pmain, pplan, presults)                        # the parity record moves; the main does not
    assert sp.result_generation(main)["generation"] == main_gen
    assert g.standup_adapter_parity_gate(MODEL, WO4)["status"] == "verified"          # the overlay alone cannot see it (why None must never mean skip)
    with pytest.raises(g.SiteGeometryError) as ei:
        g.standup_adapter_parity_gate(MODEL, WO4, record_check=ns.default_parity_record_check)
    assert "reads PARITY_STALE" in str(ei.value) and "parity_check 'mismatch'" in str(ei.value) and "re-fetches and re-stages a READY record" in str(ei.value)


# --------------------------------------------------------- the unreachable record and the fenced staging stamp

def _unreachable_staged_copy(tmp_path: Path, site: Path, *, block):
    """The site overlay re-written as the OTHER cluster holds it: provenance.probe_record pointing at a dir that does not exist here, and
    `block` (a staging_verification mapping, or None for a bare copy) — through the one serializer."""
    doc = json.loads((site / f"{MODEL}.json").read_text())
    doc["provenance"]["probe_record"] = str(tmp_path / "elsewhere" / "record")
    doc.pop(g.STAGING_VERIFICATION_KEY, None)
    if block is not None:
        doc[g.STAGING_VERIFICATION_KEY] = block
    (site / f"{MODEL}.json").write_bytes(g.site_geometry_bytes(doc))
    return doc["provenance"][g.RESULT_GENERATION_KEY]


def test_an_unreachable_record_is_verified_only_by_a_staging_stamp_that_carries_the_predicates_reading(tmp_path):
    """The default predicate was selected but never CALLED when the overlay's probe_record is not a directory on
    the serving machine — the normal cross-cluster case — and the gate returned verified with record=None while the log said
    `record_predicate=default:…`. Now: reachable -> the live predicate; unreachable -> ONLY the staging stamp the front door wrote after
    running the predicate at staging time (record_state READY, parity_check none|current, the generation, the utc, not older than the
    unit's submission when known), labelled `staging_stamp:<utc>`; anything less refuses by name."""
    out, plan, results, site = _staged(tmp_path, REF_OK)
    gen = _unreachable_staged_copy(tmp_path, site, block=None)
    # the full stamp, as slurm_stage_geometry writes it now: verified, labelled by the stamp, the predicate never claimed to have run
    full = g.staged_site_geometry_bytes(json.loads((site / f"{MODEL}.json").read_text()), record_generation=gen, verifier_host="TESTHOST",
                                        verified_utc="2026-09-30T18:00:00Z", record_state="READY", parity_check="none", parity_record=None)
    (site / f"{MODEL}.json").write_bytes(full)
    # the fence is REQUIRED: no submission time -> refused by name, never an unfenced stamp
    with pytest.raises(g.SiteGeometryError) as ei:
        g.standup_adapter_parity_gate(MODEL, WO4, record_check=ns.default_parity_record_check)
    assert "carries no submission time" in str(ei.value) and "SISTEER_UNIT_SUBMITTED_UTC" in str(ei.value) and "cannot be fenced" in str(ei.value)
    rec = g.standup_adapter_parity_gate(MODEL, WO4, record_check=ns.default_parity_record_check, stamp_not_before="2026-09-30T17:00:00Z")
    assert rec["status"] == "verified" and rec["record"]["predicate"] == "staging_stamp:2026-09-30T18:00:00Z" and rec["record"]["state"] == "READY"
    assert rec["record"]["parity_check"] == "none" and rec["record"]["generation"] == gen and rec["record"]["stamp_not_before"] == "2026-09-30T17:00:00Z"
    # bounded by the unit's submission: a stamp older than it refuses; one at or after it passes
    with pytest.raises(g.SiteGeometryError, match="predates this unit's submission"):
        g.standup_adapter_parity_gate(MODEL, WO4, record_check=ns.default_parity_record_check, stamp_not_before="2026-09-30T19:00:00Z")
    assert g.standup_adapter_parity_gate(MODEL, WO4, record_check=ns.default_parity_record_check, stamp_not_before="2026-09-30T18:00:00Z")["record"]["stamp_not_before"] == "2026-09-30T18:00:00Z"
    # the OLD three-field stamp (staged before the predicate fields existed): no proof of the predicate -> refused, naming the fields
    _unreachable_staged_copy(tmp_path, site, block={"verified_utc": "2026-09-30T18:00:00Z", "record_generation": gen, "verifier_host": "TESTHOST"})
    with pytest.raises(g.SiteGeometryError) as ei:
        g.standup_adapter_parity_gate(MODEL, WO4, record_check=ns.default_parity_record_check)
    assert "lacks ['record_state', 'parity_check']" in str(ei.value) and "re-stages the overlay" in str(ei.value)
    # a stamp written while the record was not READY / its parity basis not current: refused
    _unreachable_staged_copy(tmp_path, site, block={"verified_utc": "2026-09-30T18:00:00Z", "record_generation": gen, "verifier_host": "TESTHOST",
                                                    "record_state": "PARITY_STALE", "parity_check": "mismatch"})
    with pytest.raises(g.SiteGeometryError, match="not a READY record with a current parity basis"):
        g.standup_adapter_parity_gate(MODEL, WO4, record_check=ns.default_parity_record_check)
    # a stamp for another generation than the copy's: refused
    _unreachable_staged_copy(tmp_path, site, block={"verified_utc": "2026-09-30T18:00:00Z", "record_generation": "f" * 64, "verifier_host": "TESTHOST",
                                                    "record_state": "READY", "parity_check": "none"})
    with pytest.raises(g.SiteGeometryError, match="the stamp is another copy's|is not the overlay's own result_generation"):   # the overlay loader refuses it first
        g.standup_adapter_parity_gate(MODEL, WO4, record_check=ns.default_parity_record_check)
    # no stamp at all with an unreachable record: refused before the gate (never staged through the verifying step)
    _unreachable_staged_copy(tmp_path, site, block=None)
    with pytest.raises(g.SiteGeometryError, match="never staged through the verifying step|no staging_verification block"):
        g.standup_adapter_parity_gate(MODEL, WO4, record_check=ns.default_parity_record_check)
    # the core's label follows what ran: live / staging_stamp:<utc> / not_applicable, never "default" for a predicate that did not run
    src = (REPO / "src/sisteer/capeval/serving/native_standup.py").read_text()
    assert 'if rec_pred == "live":' in src and 'adapter_parity_gate["record_predicate"] = rec_pred' in src and "UNIT_SUBMITTED_UTC_ENV" in src
    assert "stamp_not_before=(os.environ.get(UNIT_SUBMITTED_UTC_ENV) or None)" in src and ns.UNIT_SUBMITTED_UTC_ENV == "SISTEER_UNIT_SUBMITTED_UTC"
    assert g.STAGING_PREDICATE_FIELDS == ("record_state", "parity_check")


# ------------------------------------------------------- the residual-slot carve-out on the residual path (2026-09-30)

def test_a_residual_slot_carrier_is_not_gated_as_a_lora_pair_and_a_genuine_rank_1_lora_still_refuses(tmp_path):
    """A residual carrier (lora_export._export_residual_slot: r=1, target o_proj, petri_source_kind psr / vector / … , NOT the zero adapter)
    was gated as the LoRA pair (1, 'wo') and refused by name ("not one the serving probe folded (folded ranks [4, 16, 64])") — steering
    vectors could never be served on this model. Its certificate is the residual op's own (residual_slot.armed, the noop certificate,
    the positive controls after launch), not LoRA parity: the gate lists it under `residual` and holds only the genuine LoRA pairs to
    the record; a genuine rank-1 LoRA on `wo` is still refused."""
    out, plan, results, site = _staged(tmp_path, REF_OK)                                          # wo folded at rank 4 only
    carrier = {"si_x__psr_ce_all": {"rank": 1, "module_set": "wo", "zero": False, "kind": "psr", "residual": True}}
    rec = g.standup_adapter_parity_gate(MODEL, {**carrier, **ZERO})
    assert rec["status"] == "residual_only" and rec["pairs"] == []
    assert rec["residual"]["carriers"] == [{"adapter": "si_x__psr_ce_all", "kind": "psr", "rank": 1, "set": "wo"}]
    assert "noop certificate" in rec["residual"]["certificate"] and "positive controls" in rec["residual"]["certificate"] and "not LoRA parity" in rec["residual"]["certificate"]
    # beside a genuine verified LoRA pair: the LoRA is held to the record, the carrier rides along listed, never as a pair
    rec = g.standup_adapter_parity_gate(MODEL, {**carrier, **WO4, **ZERO})
    assert rec["status"] == "verified" and [(p["adapter"], p["rank"], p["set"]) for p in rec["pairs"]] == [("si_x__lora_r4", 4, "wo")]
    assert rec["residual"]["carriers"][0]["adapter"] == "si_x__psr_ce_all"
    # a GENUINE rank-1 LoRA on wo (residual False): the parity record never folded that rank — refused by name, as before
    with pytest.raises(g.SiteGeometryError, match="not one the serving probe folded"):
        g.standup_adapter_parity_gate(MODEL, {"si_x__lora_r1": {"rank": 1, "module_set": "wo", "zero": False, "kind": "lora", "residual": False}})
    # every residual kind is carved out; a LoRA kind is not
    for kind in ("psr", "vector", "vector_all", "diffmean", "static_r", "psr_paired_write"):
        assert g.standup_adapter_parity_gate(MODEL, {"c": {"rank": 1, "module_set": "wo", "kind": kind, "residual": True}})["status"] == "residual_only"
    # the pairs reader marks a carrier from the banked block's petri_source_kind (what _adapter_block records off the export's digests)
    from sisteer.capeval.serving import standup_manifest as sm
    sp.build_adapter_dirs(plan, out, FakeExporter())
    m = sm.Manifest.from_dict(plan["manifests"]["wo_r4_S1cc"])
    d = str(sp.slot_dir(out, m.name, next(a.name for a in m.adapters if a.kind == sm.KIND_GENUINE)))
    pairs = ns.genuine_adapter_pairs({"carrier": {"dir": d, "zero": False, "petri_source_kind": "psr"},
                                      "lora": {"dir": d, "zero": False, "petri_source_kind": "lora"},
                                      "zero": {"dir": d, "zero": True, "petri_source_kind": None}})
    assert pairs["carrier"]["residual"] is True and pairs["carrier"]["kind"] == "psr" and pairs["carrier"]["module_set"] == "wo"
    assert pairs["lora"]["residual"] is False and pairs["zero"]["residual"] is False and pairs["zero"]["zero"] is True
    assert ns.RESIDUAL_SLOT_KINDS == ("psr", "vector", "vector_all", "diffmean", "static_r", "psr_paired_write")


# ------------------------------------------------------- the pair is served only in the venv its certificate was measured in
def test_the_certificate_venv_gates_the_pair(tmp_path):
    """The buffer lane binds `serving_venv` into every per_pair entry; the gate holds the standup's own venv to it: equal digests
    pass; a certificate with NO venv is UNPROVEN (refused by name — older certificates are not exempt); differing digests, or
    differing campaign names without digests, refuse naming both venvs; neither side identifiable refuses naming the fix."""
    out, plan, results, site = _staged(tmp_path, REF_OK)
    _bind_venv(site, SERVING_GDN)
    rec = g.standup_adapter_parity_gate(MODEL, {**WO4, **ZERO}, serving_venv=SERVING_GDN)
    assert rec["status"] == "verified" and rec["pairs"][0][g.CERTIFICATE_VENV_KEY]["variant"] == "gdn"
    assert rec["pairs"][0][g.CERTIFICATE_VENV_KEY]["requirements_sha256"] == GDN_DIGEST and rec["serving_venv"]["variant"] == "gdn"
    # the base venv about to serve a pair certified on GDN is ADMITTED — same engine (vLLM vendors its kernels; the
    # overlay reaches only the HF reference); the certificate's venv is recorded as provenance
    rec_b = g.standup_adapter_parity_gate(MODEL, {**WO4, **ZERO}, serving_venv=_venv(None, BASE_DIGEST))
    assert rec_b["status"] == "verified" and rec_b["pairs"][0][g.CERTIFICATE_VENV_KEY]["variant"] == "gdn" and rec_b["serving_venv"]["variant"] is None
    # (e) another ENGINE (a different vllm, base requirements digest or flashinfer) refuses naming both
    for field, other in (("vllm_version", "0.26.0+cu129"), ("base_requirements_sha256", "c" * 64), ("flashinfer_version", "0.7.0")):
        with pytest.raises(g.SiteGeometryError) as ei:
            g.standup_adapter_parity_gate(MODEL, {**WO4, **ZERO}, serving_venv=_venv(None, BASE_DIGEST, engine={**ENGINE, field: other}))
        msg = str(ei.value)
        assert "another ENGINE" in msg and "serving side from launch_provenance" in msg and f"{field}: certificate {ENGINE[field]!r} vs this standup {other!r}" in msg and "adapter 'si_x__lora_r4'" in msg
    # (b) a certificate whose serving side carries no engine identity: unproven engine, refused by name
    _bind_venv(site, dict(SERVING_GDN, engine_identity=None))
    with pytest.raises(g.SiteGeometryError, match="no complete engine identity.*unproven, not exempt"):
        g.standup_adapter_parity_gate(MODEL, {**WO4, **ZERO}, serving_venv=SERVING_GDN)
    # WHERE the serving-side identity came from decides — the standup process admits like launch provenance
    _bind_venv(site, SERVING_GDN, source="standup_process")
    assert g.standup_adapter_parity_gate(MODEL, {**WO4, **ZERO}, serving_venv=SERVING_GDN)["status"] == "verified"
    # the reference process alone attests the HF side only -> unproven
    _bind_venv(site, SERVING_GDN, source=g.ENGINE_SOURCE_REFERENCE)
    with pytest.raises(g.SiteGeometryError, match="reference process alone.*attests the HF side only"):
        g.standup_adapter_parity_gate(MODEL, {**WO4, **ZERO}, serving_venv=SERVING_GDN)
    # an unknown source, a missing source, and no serving side at all (top-level engine identity + reference side only): refused by name
    _bind_venv(site, SERVING_GDN, source="hand_typed")
    with pytest.raises(g.SiteGeometryError, match="unknown or missing source.*'hand_typed'"):
        g.standup_adapter_parity_gate(MODEL, {**WO4, **ZERO}, serving_venv=SERVING_GDN)
    _bind_venv(site, SERVING_GDN, source=None)
    with pytest.raises(g.SiteGeometryError, match="unknown or missing source.*None"):
        g.standup_adapter_parity_gate(MODEL, {**WO4, **ZERO}, serving_venv=SERVING_GDN)
    _bind_venv(site, SERVING_GDN, with_serving=False)
    with pytest.raises(g.SiteGeometryError, match="names no SERVING-side identity.*reference-side block does not stand in"):
        g.standup_adapter_parity_gate(MODEL, {**WO4, **ZERO}, serving_venv=SERVING_GDN)
    _bind_venv(site, SERVING_GDN)
    # (f) a standup that cannot state its engine refuses
    with pytest.raises(g.SiteGeometryError, match="cannot state its engine identity"):
        g.standup_adapter_parity_gate(MODEL, {**WO4, **ZERO}, serving_venv=dict(SERVING_GDN, engine_identity={**ENGINE, "flashinfer_version": None}))
    # (a) a certificate without the venv field (today's writer): unproven, refused by name
    path = site / f"{MODEL}.json"
    doc = json.loads(path.read_text())
    for blk in [doc[g.ADAPTER_PARITY_KEY]] + [st[g.ADAPTER_PARITY_KEY] for st in doc["sets"].values()]:
        for e in blk["per_pair"]:
            e.pop(g.CERTIFICATE_VENV_KEY, None)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True))
    with pytest.raises(g.SiteGeometryError, match="names NO serving venv.*unproven, not exempt"):
        g.standup_adapter_parity_gate(MODEL, {**WO4, **ZERO}, serving_venv=SERVING_GDN)
    # a residual-only standup gates no pair, so no venv check applies
    carrier = {"si_x__psr_ce_all": {"rank": 1, "module_set": "wo", "zero": False, "kind": "psr", "residual": True}}
    assert g.standup_adapter_parity_gate(MODEL, {**carrier, **ZERO}, serving_venv=_venv(None, BASE_DIGEST))["status"] == "residual_only"


def test_certificate_venv_problem_rules():
    gdn, base = SERVING_GDN, _venv(None, BASE_DIGEST)
    cert = lambda f, **over: {g.CERTIFICATE_VENV_KEY: {**_cert_block(f), **over}}  # noqa: E731
    # the ENGINE decides — gdn / base venvs on the same engine admit each other both ways
    assert g.certificate_venv_problem(cert(gdn), gdn) is None and g.certificate_venv_problem(cert(gdn), base) is None
    assert g.certificate_venv_problem(cert(base), gdn) is None
    other_engine = _venv(None, BASE_DIGEST, engine={**ENGINE, "vllm_version": "0.26.0+cu129"})
    assert "another ENGINE" in g.certificate_venv_problem(cert(gdn), other_engine)
    # no exported digest on the standup (not stood up through f22_serving_venv) but a campaign venv name: the engine still decides
    nd_gdn = dict(gdn, exported={"venv": gdn["prefix"], "variant": "gdn", "requirements_sha256": None})
    assert g.certificate_venv_problem(cert(gdn), nd_gdn) is None
    # two dev venvs on the same engine admit (serving side from launch provenance); a block with only the top-level engine identity
    # (no serving side) is unproven; a serving side without its engine is unproven
    dev = _own_venv()
    assert g.certificate_venv_problem({g.CERTIFICATE_VENV_KEY: _cert_block(dev)}, dev) is None
    assert "names no SERVING-side identity" in g.certificate_venv_problem({g.CERTIFICATE_VENV_KEY: {"name": dev["name"], "prefix": dev["prefix"], "variant": None, "requirements_sha256": None, "engine_identity": dict(ENGINE)}}, dev)
    assert "no complete engine identity" in g.certificate_venv_problem({g.CERTIFICATE_VENV_KEY: _cert_block(dev, serving_engine={"vllm_version": None, "base_requirements_sha256": None, "flashinfer_version": None})}, dev)
    # the source literals and keys are the agreed contract
    assert g.ENGINE_SOURCES_ADMITTED == ("standup_process", "launch_provenance") and g.ENGINE_SOURCE_REFERENCE == "reference_process"
    assert (g.CERTIFICATE_SERVING_SIDE_KEY, g.CERTIFICATE_REFERENCE_SIDE_KEY, g.ENGINE_IDENTITY_SOURCE_KEY) == ("serving", "reference", "engine_identity_source")
    # no venv field at all, or an empty block: unproven
    for pair in ({}, {g.CERTIFICATE_VENV_KEY: None}, {g.CERTIFICATE_VENV_KEY: {}}, {g.CERTIFICATE_VENV_KEY: {"requirements_sha256": None, "variant": None, "name": None}}):
        assert "unproven, not exempt" in g.certificate_venv_problem(pair, gdn)
    assert g.CERTIFICATE_VENV_KEY == "serving_venv" and "prefix" in g.CERTIFICATE_VENV_FIELDS
    # native_standup hands the gate its own venv facts (the same facts its records carry)
    src = (REPO / "src" / "sisteer" / "capeval" / "serving" / "native_standup.py").read_text()
    assert "serving_venv=sa.serving_venv_facts())" in src


def test_the_serving_identity_must_be_the_process_own(tmp_path):
    """The exported identity is what the SHELL selected; the process that serves runs in sys.prefix. A gdn
    export carried by a process whose prefix is the base venv (the driver python after the shell picked gdn) is refused naming
    both; a digest is believed only when the process's own venv name agrees with it (hash8 == digest[:8], variants equal)."""
    out, plan, results, site = _staged(tmp_path, REF_OK)
    _bind_venv(site, SERVING_GDN)
    base = _venv(None, BASE_DIGEST)
    # gdn exported, base process: exported_matches_prefix False -> refused before any digest is believed
    wrong_process = dict(base, exported=dict(SERVING_GDN["exported"]), exported_matches_prefix=False)
    with pytest.raises(g.SiteGeometryError) as ei:
        g.standup_adapter_parity_gate(MODEL, {**WO4, **ZERO}, serving_venv=wrong_process)
    msg = str(ei.value)
    assert "exported identity is not the process's" in msg and base["prefix"] in msg and SERVING_GDN["prefix"] in msg and "$SPY, never $MPY" in msg
    # the same venv path but an exported digest the process's name does not agree with (hash8 != digest[:8]): inconsistent, refused
    lying = dict(SERVING_GDN, exported=dict(SERVING_GDN["exported"], requirements_sha256="e" * 64))
    with pytest.raises(g.SiteGeometryError, match="inconsistent with its process venv"):
        g.standup_adapter_parity_gate(MODEL, {**WO4, **ZERO}, serving_venv=lying)
    # an exported variant that disagrees with the process venv's name: inconsistent, refused
    wrong_variant = dict(SERVING_GDN, exported=dict(SERVING_GDN["exported"], variant="nemotron"))
    with pytest.raises(g.SiteGeometryError, match="inconsistent with its process venv"):
        g.standup_adapter_parity_gate(MODEL, {**WO4, **ZERO}, serving_venv=wrong_variant)
    # a dev venv (not a campaign venv) carrying an exported campaign digest: an identity it cannot have earned
    dev = dict(_own_venv(), exported=dict(SERVING_GDN["exported"]), exported_matches_prefix=True)
    with pytest.raises(g.SiteGeometryError, match="not a campaign serving venv, yet carries the exported requirements digest"):
        g.standup_adapter_parity_gate(MODEL, {**WO4, **ZERO}, serving_venv=dev)
    # a certificate whose own name and digest disagree certifies nothing
    cert_lie = dict(SERVING_GDN, name="sisteer-petri-vllm-gdn-deadbeef", prefix="/home/x/venvs/sisteer-petri-vllm-gdn-deadbeef")
    _bind_venv(site, cert_lie)
    with pytest.raises(g.SiteGeometryError, match="certificate's venv identity is inconsistent"):
        g.standup_adapter_parity_gate(MODEL, {**WO4, **ZERO}, serving_venv=SERVING_GDN)
    # the consistent pair still passes
    _bind_venv(site, SERVING_GDN)
    assert g.standup_adapter_parity_gate(MODEL, {**WO4, **ZERO}, serving_venv=SERVING_GDN)["status"] == "verified"
    # the fixture is internally consistent
    for f in (SERVING_GDN, base):
        assert f["hash8"] == f["exported"]["requirements_sha256"][:8] and f["name"].endswith("-" + f["hash8"])


def test_a_certificate_asserting_a_digest_must_name_the_campaign_venv_that_earned_it(tmp_path):
    """A probe run from a developer/driver venv that inherited SISTEER_SERVING_REQS_SHA256 records
    {name: '.venv', requirements_sha256: <gdn>}; the gate must refuse it by name before comparing digests, never admit it
    against a genuine gdn standup on digest equality alone."""
    out, plan, results, site = _staged(tmp_path, REF_OK)
    dev_with_gdn_digest = dict(SERVING_GDN, name=".venv", prefix="/home/x/.venv", campaign_venv=False, variant=None, hash8=None)
    _bind_venv(site, dev_with_gdn_digest)                       # certificate: name '.venv', requirements_sha256 = the gdn digest
    with pytest.raises(g.SiteGeometryError) as ei:
        g.standup_adapter_parity_gate(MODEL, {**WO4, **ZERO}, serving_venv=SERVING_GDN)
    msg = str(ei.value)
    assert "asserts the requirements digest" in msg and "'.venv'" in msg and "not a campaign serving venv" in msg and "f22_serving_venv" in msg
    # the pure rule, directly: a digest without a campaign name is never compared
    pair = {g.CERTIFICATE_VENV_KEY: {"requirements_sha256": GDN_DIGEST, "variant": None, "name": ".venv"}}
    assert "cannot have earned" in g.certificate_venv_problem(pair, SERVING_GDN)
    pair_noname = {g.CERTIFICATE_VENV_KEY: {"requirements_sha256": GDN_DIGEST, "variant": "gdn", "name": None}}
    assert "cannot have earned" in g.certificate_venv_problem(pair_noname, SERVING_GDN)
    # a campaign-named certificate agreeing with its digest still passes
    _bind_venv(site, SERVING_GDN)
    assert g.standup_adapter_parity_gate(MODEL, {**WO4, **ZERO}, serving_venv=SERVING_GDN)["status"] == "verified"


def _staged_at(tmp_path, ranks, reference):
    """Like `_staged`, with the probe record planned at `ranks` (the front door's fits are rank 16; the live overlay folds 4 / 16 / 64)."""
    from test_geometry_parity_decoupled import _results
    out = tmp_path / "rec"
    plan = sp.build_plan(TOY, model_key=MODEL, model_id="org/toy", revision=REV, sets=("wo", "mlp"), ranks=tuple(ranks), windows=(4096, 49152), ladder_caps=())
    results = _results(plan, reference)
    sp.write_json_atomic(out / sp.PLAN_FILE, plan)
    for lid, res in results.items():
        sp.write_json_atomic(sp.launch_result_path(out, lid), res)
    site = tmp_path / "site"
    sp.write_verdict(plan, results, probe_job_id="11111111", out=out, verdict_uri=str(out / sp.VERDICT_FILE), geometry_dir=site, probe_cluster="cluster_a")
    sp.write_progress(plan, results, out)
    _bind_venv(site, _own_venv())
    return out, plan, results, site


def test_a_lora_subtraction_is_held_to_the_fit_rank_pair_whose_certificate_includes_its_rank_2r_difference(tmp_path):
    """The serving probe folds ONE pair per fit rank and certifies it only when its single-rank
    AND its rank-2r difference adapter passed (serving_probe.py: the role-keyed rule), so (r, set) verified IS the certificate of the
    rank-2r lora_subtraction of two rank-r fits on that set. The subtraction exports with r = 2r (lora_export._export_lora_subtraction:
    petri_source_kind lora_subtraction, forwarded by genuine_adapter_pairs as `kind`); the gate used to look (2r, set) up as a pair of its
    own and refuse by name — every s1 standup of the front door serving its difference LoRA (rank 32 on 'mlp', the overlay folding
    4 / 16 / 64) died at the gate. Now: held to (r, set), the record says so; a plain rank-2r LoRA is still refused (no such fold); a
    subtraction whose half-rank pair the set did not fold (30 -> 15) or whose set the overlay does not certify is refused by name; an
    odd served rank is refused (not a stacked pair of two equal-rank parents)."""
    out, plan, results, site = _staged_at(tmp_path, (16,), REF_OK)                       # wo + mlp folded at the front door's fit rank
    assert sorted(int(e["rank"]) for e in json.loads((site / f"{MODEL}.json").read_text())["sets"]["mlp"][g.ADAPTER_PARITY_KEY]["per_pair"]) == [16]
    diff = {"si_d__lora_r16_minus_ctl": {"rank": 32, "module_set": "mlp", "zero": False, "kind": g.DIFFERENCE_ADAPTER_KIND}}
    fit = {"si_x__lora_r16": {"rank": 16, "module_set": "wo", "zero": False, "kind": "lora"}}
    rec = g.standup_adapter_parity_gate(MODEL, {**diff, **fit, **ZERO})
    assert rec["status"] == "verified"
    d = next(p for p in rec["pairs"] if p["adapter"] == "si_d__lora_r16_minus_ctl")
    assert d["rank"] == 32 and d["set"] == "mlp" and d["status"] == "verified"
    assert d["certified_as"] == {"pair_rank": 16, "role": "difference",
                                 "why": "the probe certifies the pair (rank 16, set 'mlp') only with its rank-32 difference adapter passed"}
    assert next(p for p in rec["pairs"] if p["adapter"] == "si_x__lora_r16")["certified_as"] is None          # a pair of its own, as before
    # a PLAIN rank-32 LoRA (not a subtraction) is a pair of its own that the record did not fold: refused by name, as before
    with pytest.raises(g.SiteGeometryError, match=r"\(rank 32, set 'mlp'\) is not one the serving probe folded"):
        g.standup_adapter_parity_gate(MODEL, {"si_p__lora_r32": {"rank": 32, "module_set": "mlp", "kind": "lora"}})
    # a subtraction whose half-rank pair (15, 'mlp') the set did not fold: refused naming the adapter, both ranks and the pair
    with pytest.raises(g.SiteGeometryError, match=r"the rank-30 lora_subtraction of two rank-15 parents on set 'mlp'.*\(rank 15, set 'mlp'\) is not one the serving probe folded"):
        g.standup_adapter_parity_gate(MODEL, {"si_d__r30": {"rank": 30, "module_set": "mlp", "kind": g.DIFFERENCE_ADAPTER_KIND}})
    # an odd served rank is no stacked pair of two equal-rank parents
    with pytest.raises(g.SiteGeometryError, match="odd rank 7"):
        g.standup_adapter_parity_gate(MODEL, {"si_d__odd": {"rank": 7, "module_set": "mlp", "kind": g.DIFFERENCE_ADAPTER_KIND}})
    with pytest.raises(g.SiteGeometryError, match="certifies no such set"):
        g.standup_adapter_parity_gate(MODEL, {"si_d__qkvo": {"rank": 32, "module_set": "qkvo", "kind": g.DIFFERENCE_ADAPTER_KIND}})
    # the kind the gate keys on is the one the exporter stamps and genuine_adapter_pairs forwards
    import inspect
    from sisteer.capeval.serving import lora_export as lx
    assert g.DIFFERENCE_ADAPTER_KIND == "lora_subtraction" and 'kind = "lora_subtraction"' in inspect.getsource(lx)
    src = inspect.getsource(ns.genuine_adapter_pairs)
    assert '"kind": kind' in src and 'block.get("petri_source_kind")' in src
