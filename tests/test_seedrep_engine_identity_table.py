"""The Qwen seed-replicate battery's engine-identity table on synthetic standups: one row per standup
with the identity fields read off `endpoint.json` / `serving_attestation.json` / `standup_jobs.jsonl`; per leg the four
base engines present and equal on W / model revision / template kwargs / server_default_policy / S × L; rungs equal on
the first four; the Arditi pin verdict; every known-bad case FAILS the leg by name or REFUSES the table.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "scripts" / "sdfbase" / "seedrep", REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import audit_common as ac  # noqa: E402
import engine_identity_table as eit  # noqa: E402
import seedrep_audit_fixtures as fx  # noqa: E402
from sisteer.capeval.serving import vllm_server as VS  # noqa: E402  (the composer's capture-sizes rule — an independent oracle)

PIN = "b" * 64


def _pins(tmp_path: Path, pin: str = PIN) -> Path:
    return fx.pins_file(tmp_path / "arditi_allpos_pins.json", pin=pin)


def _wave(tmp_path: Path, **kw):
    return fx.build_wave(tmp_path / "stores", legs=("pasttense",), views=("s1", "s2", "s3"), arms=("none",), n_items=1, arditi=True,
                         arditi_pin=PIN, **kw)


def _build(tmp_path: Path, pins=None, W=fx.W_FIXTURE):
    roots = ac.discover_attempt_roots(tmp_path / "stores")
    return eit.build(roots, pins=pins if pins is not None else json.loads(_pins(tmp_path).read_text()), pin_key=eit.DEFAULT_PIN_KEY, W_expected=W)


def test_four_base_engines_pass_with_pinned_identity(tmp_path: Path):
    _wave(tmp_path)
    rows, legs = _build(tmp_path)
    assert len(rows) == 4 and legs["pasttense"]["verdict"] == "PASS" and legs["pasttense"]["problems"] == []
    assert legs["pasttense"]["base_views"] == ["arditi", "s1", "s2", "s3"]
    assert legs["pasttense"]["identity"] == {"W": fx.W_FIXTURE, "model_revision": fx.MODEL_REVISION, "template_kwargs": '{"enable_thinking": false}',
                                             "server_default_policy": '"contract_override"', "SxL": "45x11520"}
    by_view = {r["view"]: r for r in rows}
    for r in rows:
        assert r["problems"] == [] and r["S"] == 45 and r["L"] == 11520 and r["capture_ok"] is True and r["capture_max"] == 45
        assert r["cudagraph_mode"] == "FULL" and r["max_lora_rank"] == 8 and r["max_loras"] == 6 and r["engine"] == "lora_native"
        assert r["kernel_to_S"] == 45 and r["kernel_from_S"] == 56 and r["stage"] == "base" and r["leg"] == "pasttense"
        assert r["serving_config_sha256"] == fx._sha("serving-config|pasttense|base")
    assert by_view["arditi"]["arditi_pin_verdict"] == eit.PIN_OK and by_view["s1"]["arditi_pin_verdict"] == eit.NOT_ARDITI
    # fc6: the attestation keys its adapters by the LoRA NAME the engine registered (== the harness cell id), the real record's shape
    assert set(by_view["arditi"]["adapter_fingerprints"]) == {"arditi_allpos", "none__lora0"}
    assert by_view["arditi"]["adapter_fingerprints"]["arditi_allpos"]["petri_source_kind"] == "vector_all"
    assert by_view["s1"]["n_adapters"] == 1  # `none` only: the zero adapter


def test_arditi_absent_fails_the_leg(tmp_path: Path):
    fx.build_wave(tmp_path / "stores", legs=("pasttense",), views=("s1", "s2", "s3"), arms=("none",), n_items=1)
    rows, legs = _build(tmp_path)
    assert len(rows) == 3 and legs["pasttense"]["verdict"] == "FAIL" and legs["pasttense"]["problems"] == ["Arditi absent"]


def test_identity_field_differing_across_base_engines_fails(tmp_path: Path):
    roots = _wave(tmp_path)
    roots[("pasttense", "s2")].rewrite_endpoint(model_revision="deadbeef")
    _, legs = _build(tmp_path)
    assert legs["pasttense"]["verdict"] == "FAIL"
    assert any(p.startswith("model_revision differs across base engines") for p in legs["pasttense"]["problems"])
    assert legs["pasttense"]["identity"] is None


def test_template_kwargs_and_policy_are_identity_fields(tmp_path: Path):
    roots = _wave(tmp_path)
    roots[("pasttense", "s3")].rewrite_endpoint(default_chat_template_kwargs={"enable_thinking": True})
    roots[("pasttense", "s1")].rewrite_endpoint(server_default_policy="engine_defaults")
    _, legs = _build(tmp_path)
    probs = " | ".join(legs["pasttense"]["problems"])
    assert "template_kwargs differs" in probs and "server_default_policy differs" in probs


def test_pin_mismatch_fails_the_leg(tmp_path: Path):
    _wave(tmp_path)
    rows, legs = _build(tmp_path, pins=json.loads(_pins(tmp_path, pin="c" * 64).read_text()))
    ard = next(r for r in rows if r["view"] == "arditi")
    assert ard["arditi_pin_verdict"].startswith("pin_mismatch: ") and "object_artifact_sha256_by_cell[arditi_allpos]" in ard["arditi_pin_verdict"]
    assert legs["pasttense"]["verdict"] == "FAIL" and any("pin_mismatch" in p for p in legs["pasttense"]["problems"])


def test_arditi_without_pins_file_refuses(tmp_path: Path):
    _wave(tmp_path)
    roots = ac.discover_attempt_roots(tmp_path / "stores")
    with pytest.raises(ac.AuditRefusal, match="--arditi-pins was not given"):
        eit.build(roots, pins=None, pin_key=eit.DEFAULT_PIN_KEY, W_expected=None)


def test_missing_identity_field_refuses_the_table(tmp_path: Path):
    roots = _wave(tmp_path)
    r = roots[("pasttense", "s1")]
    r.endpoint.pop("attested_max_num_seqs")
    r._write_serve()
    with pytest.raises(ac.AuditRefusal, match="attested_max_num_seqs missing"):
        _build(tmp_path)


def test_capture_below_S_and_non_full_mode_fail(tmp_path: Path):
    roots = _wave(tmp_path)
    roots[("pasttense", "s2")].rewrite_endpoint(attested_cudagraph_full_capture_max=32, attested_cudagraph_mode="PIECEWISE")
    rows, legs = _build(tmp_path)
    r = next(r for r in rows if r["view"] == "s2")
    assert r["capture_ok"] is False and "FULL capture max 32 < S 45" in r["problems"] and "is not FULL" in " ".join(r["problems"])
    assert legs["pasttense"]["verdict"] == "FAIL"


def test_W_mismatch_and_standup_sha_disagreement(tmp_path: Path):
    roots = _wave(tmp_path)
    rows, legs = _build(tmp_path, W="0" * 40)
    assert all(any("!= --W" in p for p in r["problems"]) for r in rows) and legs["pasttense"]["verdict"] == "FAIL"
    # standup_jobs.jsonl sha disagreeing with ATTEMPT.json.W
    r = roots[("pasttense", "s1")]
    sj = r.serve / ac.STANDUP_JOBS_FILE
    line = json.loads(sj.read_text().splitlines()[0])
    line["sha"] = "1" * 40
    sj.write_text(json.dumps(line) + "\n")
    rows, _ = _build(tmp_path)
    assert any("!= ATTEMPT.json W" in p for p in next(r for r in rows if r["view"] == "s1")["problems"])


def test_rung_engine_identity_rules(tmp_path: Path):
    _wave(tmp_path)
    rung = fx.make_root(tmp_path / "stores", leg="pasttense", view="s1", arms=("none",), stage="8192")
    rung.flush_pipeline()
    rows, legs = _build(tmp_path)
    assert len(rows) == 5 and legs["pasttense"]["verdict"] == "PASS"
    assert legs["pasttense"]["stages"] == {"base": ["45x11520"], "8192": ["31x16384"]}
    rung.rewrite_endpoint(default_chat_template_kwargs={"enable_thinking": True})
    _, legs = _build(tmp_path)
    assert legs["pasttense"]["verdict"] == "FAIL" and any("rung s1/8192/shard00: template_kwargs" in p for p in legs["pasttense"]["problems"])
    # two engines of one rung stage disagreeing on S×L
    rung.rewrite_endpoint(default_chat_template_kwargs={"enable_thinking": False})
    rung2 = fx.make_root(tmp_path / "stores", leg="pasttense", view="s2", arms=("none",), stage="8192")
    rung2.rewrite_endpoint(attested_max_num_seqs=21, attested_cudagraph_full_capture_max=21,
                           composed_cudagraph_capture_sizes=list(VS.cudagraph_capture_sizes_for(21)),  # the composer's list for ITS S
                           geometry={**rung2.endpoint["geometry"], "adjustments": [{"kind": "lora_kernel_index_ceiling", "from_S": 40, "to_S": 21, "D": 4096, "ceiling": 524287}]})
    rung2.flush_pipeline()
    _, legs = _build(tmp_path)
    assert any("stage 8192: S×L differs" in p for p in legs["pasttense"]["problems"])
    # the two rung engines differ in S, so their capture lists rightly differ — no capture-sizes problem is raised for that
    assert not any("composed_cudagraph_capture_sizes" in p for p in legs["pasttense"]["problems"]), legs["pasttense"]["problems"]


def test_cli_exit_codes_and_files(tmp_path: Path):
    _wave(tmp_path)
    out = tmp_path / "tables" / "engine_identity.csv"
    pins = _pins(tmp_path)
    rc = ac.run_main(eit.main, ["--stores", str(tmp_path / "stores"), "--out", str(out), "--arditi-pins", str(pins), "--W", fx.W_FIXTURE])
    assert rc == 0
    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4 and set(rows[0]) == set(eit.COLUMNS) and {r["view"] for r in rows} == {"s1", "s2", "s3", "arditi"}
    legs = json.loads((tmp_path / "tables" / "engine_identity_legs.json").read_text())
    assert legs["pasttense"]["verdict"] == "PASS"
    rc = ac.run_main(eit.main, ["--stores", str(tmp_path / "stores"), "--out", str(out), "--arditi-pins", str(pins), "--W", "9" * 40])
    assert rc == ac.EXIT_FAILED
    assert ac.run_main(eit.main, ["--stores", str(tmp_path / "nope"), "--out", str(out)]) == ac.EXIT_REFUSED
    assert ac.run_main(eit.main, ["--out", str(out)]) == ac.EXIT_REFUSED


# --------------------------------------------------------------------------- #
# capture sizes are a FUNCTION of the attested S, not an identity axis                                                 #
# --------------------------------------------------------------------------- #

def test_capture_sizes_compare_against_each_engines_own_derivation(tmp_path: Path):
    """KNOWN-GOOD: the four base engines (S=45) and a rung engine (S=31) each hand vLLM the composer's list for THEIR S; the
    two lists differ (by construction, no setting changed) and the leg PASSES — a rung/base pairing is never refused for it."""
    _wave(tmp_path)
    rung = fx.make_root(tmp_path / "stores", leg="pasttense", view="s1", arms=("none",), stage="8192")
    rung.flush_pipeline()
    rows, legs = _build(tmp_path)
    assert legs["pasttense"]["verdict"] == "PASS" and legs["pasttense"]["problems"] == []
    by = {(r["view"], r["stage"]): r for r in rows}
    want45, want31 = sorted(VS.cudagraph_capture_sizes_for(45)), sorted(VS.cudagraph_capture_sizes_for(31))
    assert want45 != want31 and 45 in want45 and 31 in want31
    for v in ("s1", "s2", "s3", "arditi"):
        assert by[(v, "base")]["capture_sizes"] == want45 and by[(v, "base")]["capture_sizes_derived_ok"] is True
    assert by[("s1", "8192")]["S"] == 31 and by[("s1", "8192")]["capture_sizes"] == want31 and by[("s1", "8192")]["capture_sizes_derived_ok"] is True


def test_same_S_different_capture_list_is_a_problem(tmp_path: Path):
    """KNOWN-BAD: s2 serves S=45 like its siblings but hands vLLM the list for S=51 — a real setting change: the row is not
    its own S's derivation AND the leg's engines of one window class disagree; the leg FAILS, both named."""
    roots = _wave(tmp_path)
    roots[("pasttense", "s2")].rewrite_endpoint(composed_cudagraph_capture_sizes=list(VS.cudagraph_capture_sizes_for(51)))
    rows, legs = _build(tmp_path)
    assert legs["pasttense"]["verdict"] == "FAIL"
    probs = legs["pasttense"]["problems"]
    assert any(p.startswith("s2/base/shard00: composed_cudagraph_capture_sizes") and "is not the composer's derivation for its own attested S 45" in p
               and "PR #210" in p for p in probs), probs
    assert any(p.startswith("attested S 45: composed_cudagraph_capture_sizes differs across engines of one attested S: ") and "s2/base/shard00" in p
               for p in probs), probs
    s2 = next(r for r in rows if r["view"] == "s2")
    assert s2["capture_sizes_derived_ok"] is False and s2["capture_sizes"] == sorted(VS.cudagraph_capture_sizes_for(51))
    # the composed list absent → the table REFUSES (an identity input is never blank)
    roots[("pasttense", "s2")].endpoint.pop("composed_cudagraph_capture_sizes")
    roots[("pasttense", "s2")]._write_serve()
    with pytest.raises(ac.AuditRefusal, match="composed_cudagraph_capture_sizes missing"):
        _build(tmp_path)
