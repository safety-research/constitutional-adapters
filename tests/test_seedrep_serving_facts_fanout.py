"""Serving facts fanned out per logical run through the REAL hook.

``emit_serving_patches`` reads each key of its map as a SERVED key (``endpoint["adapters"][served_key]`` / the positions map),
so the fan-out keys every single-entry map by the route recorded in the plan and refuses a run-id key before the hook is
called. Asserted on the emitted ``run_patch`` records: a fitted run's carries ``adapter_fingerprint`` + ``applied_positions_mode``
(+ the adapter's intervention hash under ``verify``); a ``none`` run's carries neither; three evals on one route produce three
patches with equal serving facts. Known-bad: a map keyed by run id, a fitted route with no adapter block, a passthrough route
with one, a plan whose served key is a run id — each REFUSED, nothing emitted.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "scripts" / "sdfbase" / "seedrep",):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import emit_serving_facts_fanout as FO  # noqa: E402
import registry_plans as RP  # noqa: E402
from sisteer.registry import emit, identity, synth  # noqa: E402
from sisteer.registry import records as R  # noqa: E402
from sisteer.registry import validate as VAL  # noqa: E402

W = synth.code_sha("c")
CAMPAIGN = "synthbattery"
REV = "0123456789abcdef0123456789abcdef01234567"
IH = "a1b2c3d4e5f60718"
CELL = "constv2-mix-full__psr_ce_all"
ENDPOINT = {"engine": "lora_native", "mode": "petri", "model_revision": REV, "attested_max_num_seqs": 45, "attested_max_model_len": 11520,
            "serving_config_sha256": synth.SHA256_A, "serving_identity": "synth-identity", "prefix_caching": "off",
            "adapters": {CELL: {"r": 4, "petri_intervention_hash": IH, "layers": [3, 7, 11, 15]},
                         "arditi_allpos": {"r": 1, "petri_intervention_hash": "b2c3d4e5f6071829", "layers": [12]}},
            "served_positions": {CELL: "all", "arditi_allpos": "all"},
            # the engine identity the standup hands the hook (serving_attestation.engine_identity, schema 2): per attested adapter the
            # exported object's kind, which the hook reads a served `all` against (every token on a ReLU-steer or a native static direction)
            "engine_identity": {"versions": {"vllm": "0.25.1"},
                                "adapters": {CELL: {"petri_intervention_hash": IH, "petri_zero_adapter": False, "petri_source_kind": "psr_ce_all",
                                                    "residual_fingerprint": None, "position_mode": None},
                                             "arditi_allpos": {"petri_intervention_hash": "b2c3d4e5f6071829", "petri_zero_adapter": False,
                                                               "petri_source_kind": "static_r_arditi", "residual_fingerprint": None, "position_mode": None}}}}


def _ident(leg: str, eval_id: str, arm: str, kind: str, served_key: str, s: int = 1, view: str = "s1") -> dict:
    okey = {"fitted": f"base64s{s}__{CELL}", "none": "none", "prompted": arm, "arditi": "ar_c0p3__arditi_allpos"}[kind]
    prk = f"seedrep_battery:{W[:7]}:{leg}:{view if kind != 'arditi' else f'arditi_s{s}'}"
    ident = {"campaign": CAMPAIGN, "producer_run_key_legacy": prk, "producer_object_key_legacy": okey, "model_id": synth.MODEL_ID,
             "object_id": {"fitted": synth.OBJECT_SHA, "none": "none", "prompted": f"si:{synth.SI_SHA}", "arditi": synth.SHA256_B}[kind],
             "transfer_kind": "base_to_instruct", "eval_id": eval_id, "item_set_id": f"{eval_id}_x", "instrument_id": synth.INSTRUMENT_ID,
             "instrument_match": "match", "harness": "bespoke_sisteer", "harness_version": "synthetic", "scoring_rule": "judge_threshold",
             "prompt_template_sha256": synth.SHA256_A, "decode_seed_rule": "per_item", "decode_source": "model_card", "token_cap_tokens": 4096,
             "thinking_mode": "off", "serving_path": "lora_native", "applied_positions_mode": "all" if kind in ("fitted", "arditi") else "none",
             "multi_turn_steering_scope": "single_turn", "code_sha": W, "substrate": "modal", "substrate_source": "recorded", "seed_base": s,
             "extra": {"seed_rule": "literal_training_seed", "leg": leg, "container": RP.container_of_leg(leg), "arm": arm, "arm_kind": kind, "s": s,
                       "view": view if kind != "arditi" else "arditi", "served_key": served_key, "store": "static"}}
    ident["run_id"] = identity.run_id(CAMPAIGN, prk, okey, eval_id, synth.INSTRUMENT_ID)
    return ident


def _plan_file(root: Path, leg: str, arm: str, s: int, content: dict) -> RP.PlanFile:
    p = root / RP.plan_relpath(leg, arm, s)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(content, indent=1, sort_keys=True) + "\n")
    return RP.load_plan_file(p)


def _plans(root: Path):
    """Three evals on the fitted route (three legs served by one engine), a none run, a prompted run, an Arditi run."""
    files = [_plan_file(root, leg, CELL, 1, _ident(leg, leg, CELL, "fitted", CELL)) for leg in ("pasttense", "codeattack", "renellm")]
    files.append(_plan_file(root, "pasttense", "none", 1, _ident("pasttense", "pasttense", "none", "none", "none")))
    files.append(_plan_file(root, "pasttense", "prompted__si_constitution", 1,
                            _ident("pasttense", "pasttense", "prompted__si_constitution", "prompted", "prompted__si_constitution")))
    files.append(_plan_file(root, "pasttense", "arditi_allpos", 2, _ident("pasttense", "pasttense", "arditi_allpos", "arditi", "arditi_allpos", s=2)))
    return files


def _patches(spool: Path):
    out = {}
    for p in R.iter_record_files(str(spool)):
        for _i, rec, sha in R.read_record_file(p):
            assert isinstance(VAL.validate_record(rec, p, sha), VAL.Validated), rec
            assert rec["record_type"] == "run_patch" and rec["producer"]["script"] == FO.SCRIPT
            out.setdefault(rec["payload"]["run_id"], []).append(rec["payload"])
    return out


@pytest.fixture(autouse=True)
def _emission_on(monkeypatch):
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)


def test_fitted_none_prompted_and_arditi_patches_carry_the_right_facts_and_three_evals_share_them(tmp_path):
    files = _plans(tmp_path / "plans")
    runs = FO.runs_for_engine(files, ENDPOINT)
    assert len(runs) == 6 and all(not FO._is_run_id(k) for _pf, _r, k, _run in runs)
    out = FO.fanout(ENDPOINT, runs, spool=tmp_path / "spool")
    assert len(out) == 6 and all(len(e["paths"]) == 1 for e in out)
    patches = _patches(tmp_path / "spool")
    by_run = {pf.identities["*"]["run_id"]: pf for pf in files}
    assert set(patches) == set(by_run) and all(len(v) == 1 for v in patches.values())     # exactly one patch per logical run
    fitted = [patches[rid][0] for rid, pf in by_run.items() if pf.identities["*"]["extra"]["arm_kind"] == "fitted"]
    assert len(fitted) == 3
    for p in fitted:
        f, v = p["fields"], p["verify"]
        assert f["adapter_fingerprint"] == identity.sha256_hex(identity.canonical_json(ENDPOINT["adapters"][CELL]).encode()) and f["applied_positions_mode"] == "all"
        assert f["applied_layers_json"] == [3, 7, 11, 15] and f["served_max_num_seqs"] == 45 and f["served_max_num_seqs_source"] == "attested"
        assert f["context_window_tokens"] == 11520 and f["serving_engine_version"] == "vllm 0.25.1" and f["prefix_cache_policy"] == "off"
        assert f["serving_config_sha256"] == synth.SHA256_A and f["extra_json"]["served_key"] == CELL
        assert v == {"serving_path": "lora_native", "intervention_hash": IH, "model_revision": REV}
    # three evals on one route: three patches (three run ids) with EQUAL serving facts
    assert fitted[0]["fields"] == fitted[1]["fields"] == fitted[2]["fields"] and fitted[0]["verify"] == fitted[1]["verify"] == fitted[2]["verify"]
    assert len({p["run_id"] for p in fitted}) == 3
    none_p = next(patches[rid][0] for rid, pf in by_run.items() if pf.identities["*"]["extra"]["arm_kind"] == "none")
    prompted_p = next(patches[rid][0] for rid, pf in by_run.items() if pf.identities["*"]["extra"]["arm_kind"] == "prompted")
    for p in (none_p, prompted_p):
        assert "adapter_fingerprint" not in p["fields"] and "applied_positions_mode" not in p["fields"] and "applied_layers_json" not in p["fields"]
        assert "intervention_hash" not in p["verify"] and p["verify"]["serving_path"] == "lora_native"
        assert p["fields"]["served_max_num_seqs"] == 45 and p["fields"]["context_window_tokens"] == 11520     # the engine facts still travel
    assert none_p["fields"]["extra_json"]["served_key"] == "none" and prompted_p["fields"]["extra_json"]["served_key"] == "prompted__si_constitution"
    ar = next(patches[rid][0] for rid, pf in by_run.items() if pf.identities["*"]["extra"]["arm_kind"] == "arditi")
    assert ar["fields"]["applied_layers_json"] == [12] and ar["verify"]["intervention_hash"] == "b2c3d4e5f6071829" and ar["fields"]["applied_positions_mode"] == "all"
    # every field the fan-out emits is classified (S7)
    for p in (fitted[0], none_p, ar):
        assert set(p["fields"]) <= set(R.RUN_PATCHABLE_FIELDS) and set(p["verify"]) <= set(R.RUN_VERIFIED_FIELDS)


def test_a_run_keyed_by_run_id_is_refused_before_the_hook_and_mis_keyed_routes_are_refused(tmp_path):
    files = _plans(tmp_path / "plans")
    rid = files[0].identities["*"]["run_id"]
    with pytest.raises(FO.FanoutRefused, match="keyed by run id"):
        FO.check_keyed_by_served_key({rid: {"run_id": rid, "campaign": CAMPAIGN}})
    with pytest.raises(FO.FanoutRefused, match="keyed by run id"):
        FO.fanout(ENDPOINT, [(files[0], "*", rid, {"run_id": rid, "campaign": CAMPAIGN})], spool=tmp_path / "spool")
    assert not (tmp_path / "spool").exists()
    # the served key written into a plan as a run id
    bad = _plan_file(tmp_path / "bad", "pasttense", CELL, 3, dict(_ident("pasttense", "pasttense", CELL, "fitted", CELL, s=3), extra={
        **_ident("pasttense", "pasttense", CELL, "fitted", CELL, s=3)["extra"], "served_key": rid}))
    with pytest.raises(FO.FanoutRefused, match="is a run id"):
        FO.runs_for_engine([bad], ENDPOINT)
    # a fitted route whose adapter block is missing from the endpoint (its fingerprint would be silently absent)
    ep_missing = {**ENDPOINT, "adapters": {k: v for k, v in ENDPOINT["adapters"].items() if k != CELL}}
    with pytest.raises(FO.FanoutRefused, match="has no adapter block"):
        FO.runs_for_engine(files[:1], ep_missing)
    # a passthrough route that HAS an adapter block (a mis-keyed plan)
    ep_none = {**ENDPOINT, "adapters": {**ENDPOINT["adapters"], "none": {"r": 4}}}
    with pytest.raises(FO.FanoutRefused, match="HAS an adapter block"):
        FO.runs_for_engine(files, ep_none)
    # what the hook would do with a run-id key, for the record: it reads the key as a served key and finds no adapter block,
    # so a fitted run would lose its fingerprint — the very failure the refusal above prevents
    from sisteer.registry import hooks
    lost = hooks.serving_patch_payload(ENDPOINT, rid, {"run_id": rid})
    assert "adapter_fingerprint" not in lost["fields"]
    kept = hooks.serving_patch_payload(ENDPOINT, CELL, {"run_id": rid})
    assert "adapter_fingerprint" in kept["fields"]


def test_cli_selects_the_engines_runs_by_leg_and_view_and_dry_run_writes_nothing(tmp_path, capsys):
    plans = tmp_path / "plans"
    _plans(plans)
    ep = tmp_path / "endpoint.json"
    ep.write_text(json.dumps({k: v for k, v in ENDPOINT.items() if k != "engine_identity"}))
    rc = FO.main(["--plans-dir", str(plans), "--leg", "pasttense", "--view", "s1", "--endpoint", str(ep), "--spool", str(tmp_path / "spool"),
                  "--engine-identity", json.dumps(ENDPOINT["engine_identity"]), "--dry-run"])
    assert rc == 0 and not (tmp_path / "spool").exists()
    lines = [json.loads(l) for l in capsys.readouterr().out.strip().splitlines()]
    assert len(lines) == 3 and {l["served_key"] for l in lines} == {CELL, "none", "prompted__si_constitution"}   # s1's pasttense runs; Arditi is another view
    rc = FO.main(["--plans-dir", str(plans), "--leg", "pasttense", "--leg", "codeattack", "--leg", "renellm", "--view", "s1", "--endpoint", str(ep),
                  "--spool", str(tmp_path / "spool"), "--engine-identity", json.dumps(ENDPOINT["engine_identity"])])
    assert rc == 0
    patches = _patches(tmp_path / "spool")
    assert len(patches) == 5 and all(p[0]["fields"]["serving_engine_version"] == "vllm 0.25.1" for p in patches.values())
    rc = FO.main(["--plans-dir", str(plans), "--leg", "pasttense", "--view", "arditi", "--endpoint", str(ep), "--spool", str(tmp_path / "spool2")])
    assert rc == 0 and len(_patches(tmp_path / "spool2")) == 1
    rc = FO.main(["--plans-dir", str(plans), "--leg", "bon", "--view", "s1", "--endpoint", str(ep), "--spool", str(tmp_path / "spool3")])
    assert rc == 2 and "no plan files for leg 'bon'" in capsys.readouterr().err
