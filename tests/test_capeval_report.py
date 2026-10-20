"""capeval report tests. Base lane only.

Covers: build() end-to-end on synthetic rows.parquet + the frozen §7.3
fingerprint shape (HTML + final_<what>_<sweep>.png figures + figs-dir
mirror); the wall-clock discipline (HF P50/P95 end-to-end in-job vs vLLM
mean labeled n — r2-F7/r3-4, never merged); MISSING-INPUT entries in the
gate log for absent optional inputs (never silent); flag rows surfacing
verbatim; the rows.parquet refusal; and the R10 free-disk floor.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

import sisteer.capeval.report.build as rb  # the module (package re-exports
                                            # the build() function under the
                                            # same name)
from sisteer.capeval.store import ROW_COLUMNS

REV = "0123456789abcdef0123456789abcdef01234567"


def _row(**kw):
    base = dict(
        sweep="proto", run_id="r0", model_id="test-org/tiny-model",
        model_revision=REV, arm_name="none", kind="none", coeff=None,
        intervention_hash="a1b2c3d4e5f60718", engine="hf", tier=1,
        task="tinyMMLU", task_version="v1", domain="knowledge",
        item_id="i0", trial_id=0, seed=0, attempt=0, position_order=None,
        metric="acc", value=1.0, extra_json=None,
        harness_config_hash="0f1e2d3c4b5a6978", env_digest="99aabbccddeeff00",
        slurm_job_id=None, created_at=1755000000.0, wall_clock_s=None)
    base.update(kw)
    return base


def _ts_rows(arm, trial, engine, ready, tier_ends):
    rows = [_row(arm_name=arm, trial_id=trial, engine=engine,
                 metric="ts_model_ready", value=ready, item_id="-")]
    for tier, end in tier_ends.items():
        rows.append(_row(arm_name=arm, trial_id=trial, engine=engine,
                         metric=f"ts_tier{tier}_end", value=end, item_id="-"))
    return rows


def _dom(delta, lower, upper, status, margin=0.02, gating=True):
    """A domain entry in the WRITER's shape (DomainStatus.as_dict via
    build_fingerprint: harm-oriented upper_bound/lower_bound + the
    resolved margin — simplify-pass #6)."""
    return {"delta": delta, "harm": -delta, "lower_bound": lower,
            "upper_bound": upper, "status": status, "margin": margin,
            "in_gating_set": gating,
            "role": None if gating else "acceptance"}


# The fingerprint.json wire format the DRIVER writes (schema_version 2:
# KL demoted to an annotation; arms wrapped —
# scripts/capeval_run.py cmd_fingerprint).
FINGERPRINT = {
    "schema_version": 2,
    "sweep": "proto",
    "arms": {
        "number_words_bidpo_c2": {
            "domains": {
                "math": _dom(-0.31, 0.22, 0.4, "degraded"),
                "knowledge": _dom(0.01, -0.04, 0.02, "indeterminate"),
                "math_s2parity": _dom(-0.84, 0.78, 0.9, "degraded",
                                      gating=False),
                "knowledge_s2parity": _dom(0.02, -0.05, 0.01,
                                           "indeterminate", gating=False),
            },
            "screen": "SCREEN-FAIL",
            "kl_coarse_lb": {"per_corpus": {"tier0/kl_coarse_wikitext103": 0.19},
                             "mean": 0.19, "unit": "nats_per_position",
                             "note": "annotation/triage only — never a "
                                     "screen or verdict input (D22)"},
            "gate_log": ["domain code: no rows in this cell — status not "
                         "computed"],
        },
        "json_format_vector_rkl_c05": {
            "domains": {
                "math": _dom(-0.01, -0.01, 0.03, "preserved"),
                "knowledge": _dom(0.0, -0.02, 0.02, "preserved"),
            },
            "screen": "SCREEN-PASS",
            "kl_coarse_lb": None,
        },
    },
}

CONTROLS = {"arms": {
    "random_vector": {"coeffs": [0.5, 1.0, 2.0, 4.0], "acceptance": "pass",
                      "tier0_damage": {"kl_coarse_lb": [1e-4, 5e-4, 4e-3, 0.03]}},
    "noise_lora": {"coeffs": [1e-4, 3e-4, 1e-3, 3e-3], "acceptance": "pass",
                   "tier0_damage": {"kl_coarse_lb": [2e-5, 9e-5, 1e-3, 0.01]}},
}}


@pytest.fixture()
def sweep(tmp_path):
    rows = []
    # accuracy rows + one flag row
    for item in range(4):
        rows.append(_row(item_id=f"i{item}"))
        rows.append(_row(item_id=f"i{item}", arm_name="number_words_bidpo_c2",
                         kind="steering", coeff=2.0, value=0.0))
    rows.append(_row(metric="flag_kl_inf_positions", value=3.0,
                     arm_name="number_words_bidpo_c2", task="tier0/kl_coarse_wikitext103",
                     domain="dist_shift", tier=0))
    # HF ts rows: 3 cells, tier1 end-to-end 100/200/300s; tier0 10/20/30s
    for i, (t1, t0) in enumerate([(100, 10), (200, 20), (300, 30)]):
        rows += _ts_rows(f"arm{i}", i, "hf", ready=1000.0,
                         tier_ends={1: 1000.0 + t1, 0: 1000.0 + t0})
    # vLLM ts rows: 2 repeats, tier1 400/600s
    for trial, t1 in enumerate([400, 600]):
        rows += _ts_rows("base_vllm", trial, "vllm", ready=2000.0,
                         tier_ends={1: 2000.0 + t1})
    df = pd.DataFrame(rows, columns=list(ROW_COLUMNS))
    d = tmp_path / "sweep"
    d.mkdir()
    df.to_parquet(d / "rows.parquet")
    (d / "fingerprint.json").write_text(json.dumps(FINGERPRINT))
    (d / "controls_acceptance.json").write_text(json.dumps(CONTROLS))
    return d


def test_build_end_to_end(sweep, tmp_path):
    figs = tmp_path / "figs"
    paths = rb.build(sweep, "proto", figs_dir=figs, wandb_mode="disabled")
    html = paths.html.read_text()
    assert paths.html.name == "index.html" and paths.html.is_file()
    # figures exist with the standing final_<what>_<sweep>.png names
    assert paths.figures["fingerprint_heatmap"].name == "final_fingerprint_heatmap_proto.png"
    assert paths.figures["dose_response"].name == "final_dose_response_proto.png"
    assert paths.figures["instrument_comparison"].name == "final_instrument_comparison_proto.png"
    for p in paths.figures.values():
        assert p.is_file() and p.stat().st_size > 1000
        assert (figs / p.name).is_file()          # mirror copy
    assert (figs / "index.html").is_file()
    # screen statuses + flag row + wall-clock labels all rendered
    assert "SCREEN-FAIL" in html and "SCREEN-PASS" in html
    assert "flag_kl_inf_positions" in html        # gate log, verbatim
    assert "P50/P95" in html and "no P50/P95 from 2 repeats" in html
    # the KL annotation renders beside the screen, labeled as such
    assert "kl_coarse_lb mean 0.19" in html
    assert "annotation, not a screen input" in html


def test_wall_clock_discipline(sweep):
    df = pd.read_parquet(sweep / "rows.parquet")
    gate = rb.GateLog()
    hf, vllm = rb.wall_clock_tables(df, gate)
    hf1 = hf[hf["tier"] == "1"].iloc[0]
    assert hf1["n"] == 3 and hf1["p50_s"] == 200.0
    assert hf1["p95_s"] == pytest.approx(290.0)   # quantile over 100/200/300
    hf0 = hf[hf["tier"] == "0"].iloc[0]
    assert hf0["p50_s"] == 20.0
    v1 = vllm[vllm["tier"] == "1"].iloc[0]
    assert v1["n"] == 2 and v1["mean_s"] == 500.0  # mean, labeled n=2
    assert "p50_s" not in vllm.columns             # never a vLLM P50/P95 (r2-F7)
    # review infra #9: the acc-row cells filed tier-1 measurements but no
    # timestamps — they must land in the gate log and the n_missing
    # column, never silently vanish from the funnel numbers
    assert hf1["n_missing"] >= 2
    assert any(e["kind"] == "MISSING-TS" for e in gate.entries)
    assert any(e["kind"] == "INCOMPLETE-TIMING" for e in gate.entries)


def test_wall_clock_nan_coeff_cells_reconcile(tmp_path):
    """A calibration cell (kind none, coeff=None -> NaN in parquet) with
    BOTH measurement rows and complete timestamps must reconcile clean:
    NaN keys from the two separate groupbys never compare equal, so every
    calibration cell was double-counted as measured AND missing (12 false
    MISSING-TS alarms on the k=12 calibration sweep)."""
    rows = []
    for item in range(3):
        rows.append(_row(item_id=f"i{item}"))          # arm none, coeff None
    rows += _ts_rows("none", 0, "hf", ready=1000.0,
                     tier_ends={1: 1000.0 + 150})
    df = pd.read_parquet(_parquet(tmp_path, rows))
    gate = rb.GateLog()
    hf, _ = rb.wall_clock_tables(df, gate)
    hf1 = hf[hf["tier"] == "1"].iloc[0]
    assert hf1["n"] == 1 and hf1["p50_s"] == 150.0
    assert hf1["n_missing"] == 0
    assert not any(e["kind"] == "MISSING-TS" for e in gate.entries)


def _parquet(tmp_path, rows):
    path = tmp_path / "nan_coeff_rows.parquet"
    pd.DataFrame(rows, columns=list(ROW_COLUMNS)).to_parquet(path)
    return path


def test_wall_clock_flags_vllm_single_repeat(sweep):
    df = pd.read_parquet(sweep / "rows.parquet")
    # drop one vLLM repeat: 2 are required — n=1 must be gate-logged
    df = df[~((df["engine"] == "vllm") & (df["trial_id"] == 1))]
    gate = rb.GateLog()
    _, vllm = rb.wall_clock_tables(df, gate)
    assert vllm[vllm["tier"] == "1"].iloc[0]["n"] == 1
    assert any(e["kind"] == "INCOMPLETE-TIMING" and "D10" in e["detail"]
               for e in gate.entries)


def test_fingerprint_schema_version_checked(sweep, tmp_path):
    """Simplify-pass #6: a fingerprint.json this reader was not written
    for refuses loudly instead of rendering invented field names."""
    (sweep / "fingerprint.json").write_text(json.dumps(
        {"schema_version": 99, "arms": {}}))
    with pytest.raises(SystemExit, match="schema_version"):
        rb.build(sweep, "proto", figs_dir=tmp_path / "f", wandb_mode="disabled")
    # the LEGACY unwrapped shape (no schema_version at all) also refuses
    (sweep / "fingerprint.json").write_text(json.dumps({"arm": {"domains": {}}}))
    with pytest.raises(SystemExit, match="schema_version"):
        rb.build(sweep, "proto", figs_dir=tmp_path / "f", wandb_mode="disabled")


def test_fingerprint_table_reads_writer_field_names():
    arms = rb.load_fingerprint_arms(FINGERPRINT)
    table = rb.fingerprint_table(arms, rb._domains_registry())
    row = table[(table["arm"] == "number_words_bidpo_c2")
                & (table["domain"] == "math")].iloc[0]
    assert row["harm_lower"] == 0.22 and row["harm_upper"] == 0.4
    assert row["margin"] == 0.02 and row["status"] == "degraded"
    assert row["role"] == "gating"
    parity = table[(table["arm"] == "number_words_bidpo_c2")
                   & (table["domain"] == "math_s2parity")].iloc[0]
    assert parity["role"] == "acceptance"


def test_report_reconciles_missing_arms_and_domains(sweep, tmp_path):
    """Review infra #8: an arm with filed rows but no fingerprint entry,
    and a fingerprint entry missing gating domains, are gate-log failures
    — never a silently smaller screen."""
    paths = rb.build(sweep, "proto", figs_dir=tmp_path / "f",
                     wandb_mode="disabled")
    html = paths.html.read_text()
    # the fixture arms cover only math/knowledge of the 9 gating domains
    assert "MISSING-FINGERPRINT" in html
    assert "screen coverage incomplete" in html
    # per-arm build_fingerprint gate notes surface verbatim
    assert "FINGERPRINT-GATE" in html
    # drop the fingerprint entry for an arm that HAS rows -> gate entry
    doc = json.loads((sweep / "fingerprint.json").read_text())
    del doc["arms"]["number_words_bidpo_c2"]
    (sweep / "fingerprint.json").write_text(json.dumps(doc))
    html2 = rb.build(sweep, "proto", figs_dir=tmp_path / "f2",
                     wandb_mode="disabled").html.read_text()
    assert "has filed rows but no fingerprint" in html2


def test_flag_rows_carry_reason_and_identity(sweep, tmp_path):
    """Review infra #16: FLAG-ROW gate entries include extra_json (the
    reason) and the cell identity fields."""
    df = pd.read_parquet(sweep / "rows.parquet")
    df.loc[df["metric"] == "flag_kl_inf_positions", "extra_json"] = (
        json.dumps({"reason": "3 positions with q=0"}))
    df.to_parquet(sweep / "rows.parquet")
    html = rb.build(sweep, "proto", figs_dir=tmp_path / "f",
                    wandb_mode="disabled").html.read_text()
    assert "3 positions with q=0" in html
    assert "trial=0 attempt=0 engine=hf" in html


def test_missing_optional_inputs_are_loud_not_silent(sweep, tmp_path):
    (sweep / "fingerprint.json").unlink()
    (sweep / "controls_acceptance.json").unlink()
    paths = rb.build(sweep, "proto", figs_dir=tmp_path / "f", wandb_mode="disabled")
    html = paths.html.read_text()
    assert "MISSING-INPUT" in html
    assert "fingerprint.json" in html and "controls_acceptance.json" in html
    assert "fingerprint_heatmap" not in paths.figures  # absent, not faked


def test_missing_parquet_refused(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    with pytest.raises(FileNotFoundError, match="consolidate"):
        rb.build(d, "proto", figs_dir=tmp_path / "f", wandb_mode="disabled")


def test_disk_floor_refusal(sweep, tmp_path, monkeypatch):
    import shutil as _shutil

    class Usage:
        free = 100 * 1024 * 1024  # 100MB < 300MB floor (R10)
        total = used = 0

    monkeypatch.setattr(rb.shutil, "disk_usage", lambda p: Usage)
    with pytest.raises(SystemExit, match="R10"):
        rb.build(sweep, "proto", figs_dir=tmp_path / "f", wandb_mode="disabled")


def test_queue_wait_table_skips_loud_without_job_ids(sweep):
    df = pd.read_parquet(sweep / "rows.parquet")
    gate = rb.GateLog()
    table = rb.queue_wait_table(df, gate)
    assert table.empty
    assert any(e["kind"] == "SKIP" and "slurm_job_id" in e["detail"]
               for e in gate.entries)
