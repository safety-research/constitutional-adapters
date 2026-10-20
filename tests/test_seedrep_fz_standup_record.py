"""fz — the standup record is read from the attempt's `exp/serve/` and survives the Modal path's two shapes (F-AA / F-AE).

Known-bad, pinned from the S1 root (`children/determinism.log`, `children/request_evidence_audit.log`):
  * F-AE — a gsm8k row's root holds `exp/serve/endpoint.json` AND five `exp/serve_gsm8k/<cell>__s1/endpoint.json` (the GSM8K
    assembly's per-cell copies), so `audit_common.find_one(root, "endpoint.json")` refused EVERY audit:
    `6 files called 'endpoint.json' … — an ambiguous attempt root is not read`.
  * F-AA — the Modal app never wrote `exp/serve/standup_jobs.jsonl`, which `read_standup` REQUIRED:
    `no 'standup_jobs.jsonl' under the attempt root`; and `request_evidence_audit` runs after teardown, when `mt_serving_up.py` has
    unlinked the live `endpoint.json`: `no 'endpoint.json' under the attempt root`.

Green: the standup record IS `exp/serve/endpoint.json` (the layout `seedrep_served.ATTEMPT_ROOT_LAYOUT` declares); the per-cell copies
are CELL records exposed by name and never mistaken for the standup's; a second candidate UNDER `exp/serve/` still refuses; the
READY `endpoint.json` bytes the app restores at the same path after teardown are read; an absent `standup_jobs.jsonl` derives the standup from the attested records
with the derivation RECORDED and carried on every consuming table; both the cluster's 27-key job line and the one-line Modal shape
are accepted, and a line lacking a required key refuses by name. Synthetic roots only (`seedrep_audit_fixtures`).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "scripts" / "sdfbase" / "seedrep", REPO / "scripts" / "modal", REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import at_cap_table as act  # noqa: E402
import audit_common as ac  # noqa: E402
import determinism_check as dc  # noqa: E402
import engine_identity_table as eit  # noqa: E402
import request_evidence_audit as rea  # noqa: E402
import seedrep_audit_fixtures as fx  # noqa: E402
import utilisation_table as ut  # noqa: E402

GSM8K_CELLS = ("constv2-mix-full__psr_ce_all", "constv2-mix-full__lora_r4_ce", "constv2ctl2-full__psr_ce_all",
               "constv2ctl2-full__lora_r4_ce", "none")
#: the cluster's `f22_standup` job line (`common.sh`), every key, synthetic values
CLUSTER_LINE = {"job_id": "123456", "node": "TESTHOST", "key": ac.MODEL_KEY, "view": "s1", "mode": "attacks", "eval": "families_served",
                "sha": fx.W_FIXTURE, "engine": "lora_native", "serving_path": "lora_native", "cells_skipped": [], "geometry_adjustments": None,
                "lens_reduction_mode": None, "max_num_seqs": 45, "max_model_len": 11520, "attested_max_num_seqs": 45,
                "attested_max_model_len": 11520, "batched_tokens": 518400, "budget_fraction": 0.6988, "model_facts": None,
                "cluster": "cluster_a", "gpu_mem_util": 0.9, "mrope_attestation": None, "thinking_mode": "off",
                "thinking_certification_verdict": None, "upstream_timeout_s": None, "serving_path_selection": None,
                "written_at": "2026-01-01T00:00:00Z"}
#: the one-line Modal shape (fy): exactly the keys `read_standup` requires
MODAL_LINE = {"job_id": "fc-11111111", "key": ac.MODEL_KEY, "view": "s1", "mode": "attacks", "eval": "families_served", "sha": fx.W_FIXTURE}


def _faithful_post(raw: bytes, headers):
    body = json.loads(raw)
    item = body["messages"][0]["content"][len("q "):] if body["messages"][0]["content"].startswith("q ") else body["messages"][0]["content"][len("prompt "):]
    return {"id": "chatcmpl-replay", "choices": [{"message": {"content": f"response for {item} under {body['model']}"}, "finish_reason": "stop"}]}


@pytest.fixture(autouse=True)
def _hooks(monkeypatch):
    monkeypatch.setattr(dc, "POST_FN", _faithful_post)
    monkeypatch.setattr(dc, "METRICS_FN", lambda: {"num_requests_running": 0, "num_requests_waiting": 0, "generation_tokens_total": 10})
    monkeypatch.setattr(dc, "TOKENIZE_FN", None)
    yield


def _gsm8k_root(tmp_path: Path, *, arms=("none",)):
    root = fx.make_root(tmp_path / "stores", leg="gsm8k", view="s1", arms=arms)
    fx.populate_root(root, s=1, arms=arms, n_items=3)
    copies = root.write_gsm8k_cell_copies(GSM8K_CELLS, s=1)
    return root, ac.read_attempt_root(root.path), copies


def _pasttense_root(tmp_path: Path):
    root = fx.make_root(tmp_path / "stores", leg="pasttense", view="s1", arms=("none",))
    fx.populate_root(root, s=1, arms=("none",), n_items=3)
    return root, ac.read_attempt_root(root.path)


def _audit(ar, tmp_path: Path, legs):
    return rea.audit_root(ar, out_dir=tmp_path / "audits", legs=list(legs), policy_file=None,
                          thinking_expect=dict(ac.THINKING_EXPECT_DEFAULT), petri_behaviours=list(fx.PETRI_BEHAVIOURS),
                          model_key=ac.MODEL_KEY, client_status="absent", siblings=[])


def _every_audit(ar, tmp_path: Path, *, leg: str):
    """Every in-container / table reader that opens the standup record, on one root."""
    standup = ac.read_standup(ar)
    det = dc.run_check(ar, leg=leg, arms=["none"], n=3, proxy_url=None, api_key=None, metrics_url=None, tokenize_url=None,
                       allow_busy=False, petri_behaviours=[])
    verdicts = _audit(ar, tmp_path, [leg])
    rows, legs = eit.build([ar], pins=None, pin_key=eit.DEFAULT_PIN_KEY, W_expected=None)
    util = ut.utilisation_row(ar, s5=fx.s5_references(), expected={leg: 3.0}, period_s=10.0)
    roots = ac.discover_attempt_roots(ar.path.parents[2])
    cap = act.row_for(leg, "none", 1, "s1", roots, petri=list(fx.PETRI_BEHAVIOURS), trig_static=act.trigger_share_static(),
                      trig_gsm8k=act.trigger_share_gsm8k())
    return standup, det, verdicts, rows, legs, util, cap


def _only_missing_views(legs, leg: str) -> bool:
    """A one-root wave fails the leg on the four-base-views rule alone (pre-existing); the standup record feeds the ROW."""
    probs = legs[leg]["problems"]
    return len(probs) == 1 and probs[0].startswith("base engine(s) absent for view(s)")


# --------------------------------------------------------------------------- #
# The layout the readers key on IS the app's                                                                      #
# --------------------------------------------------------------------------- #

def test_layout_constants_are_the_launch_plans():
    import seedrep_launch_plan as lp
    assert ac.SERVE_DIR_REL == lp.SERVE_DIR_REL == "exp/serve"
    assert lp.GSM8K_LEDGER_SERVE_REL.startswith(ac.GSM8K_CELL_SERVE_DIR_REL + "/")
    assert ac.STANDUP_JOBS_REQUIRED_KEYS == ("job_id", "key", "view", "mode", "eval", "sha")
    assert set(ac.STANDUP_JOBS_REQUIRED_KEYS) <= set(CLUSTER_LINE) and set(MODAL_LINE) == set(ac.STANDUP_JOBS_REQUIRED_KEYS)


# --------------------------------------------------------------------------- #
# F-AE: the gsm8k root's per-cell copies                                                                           #
# --------------------------------------------------------------------------- #

def test_known_bad_gsm8k_root_the_legacy_walk_refuses_as_ambiguous(tmp_path: Path):
    """The S1 refusal, reproduced on the synthetic root: 1 standup + 5 cell copies → the whole-root walk sees six."""
    root, ar, copies = _gsm8k_root(tmp_path)
    assert len(copies) == 5
    with pytest.raises(ac.AuditRefusal, match=r"6 files called 'endpoint.json'.*an ambiguous attempt root is not read"):
        ac.find_one(root.path, ac.ENDPOINT_FILE)


def test_gsm8k_root_every_audit_reads_the_standups_record(tmp_path: Path):
    root, ar, copies = _gsm8k_root(tmp_path)
    standup, det, verdicts, rows, legs, util, cap = _every_audit(ar, tmp_path, leg="gsm8k")
    want = root.serve / ac.ENDPOINT_FILE
    assert standup.endpoint_path == want and ac.standup_endpoint_path(root.path) == want
    assert ac.CELL_ENDPOINT_COPY_OF_FIELD not in standup.endpoint  # the standup's record, not a copy
    assert standup.provenance["endpoint_file"] == "endpoint.json" and standup.provenance["standup_jobs_jsonl"] == "present"
    assert ac.vmodel_of_arm(root.path, "none") == root.vmodels["none"]
    # the cell records are exposed BY NAME and only there
    cells = ac.gsm8k_cell_files(root.path, ac.ENDPOINT_FILE)
    assert sorted(cells) == sorted(f"{c}__s1" for c in GSM8K_CELLS) and set(cells.values()) == set(copies)
    assert all(ac.read_json(p)[ac.CELL_ENDPOINT_COPY_OF_FIELD] == str(want) for p in cells.values())
    # every consumer succeeded off the standup's record
    # the check got past the standup read (the known-bad refusal point) and drew its plan off the standup's record; replay fidelity is the
    # determinism suite's business (its stub speaks the static legs' prompt shape, not gsm8k's)
    assert det["S"] == 31 and det["L"] == 16640 and det["n_requests"] >= 1
    assert det["endpoint_file"] == "endpoint.json" and det["standup_record"] == ac.STANDUP_JOBS_FILE
    assert [v["verdict"] for v in verdicts] == ["AGREES"] and verdicts[0]["standup_record"] == ac.STANDUP_JOBS_FILE
    assert rows[0]["endpoint_path"] == str(want) and rows[0]["standup_record"] == ac.STANDUP_JOBS_FILE
    assert rows[0]["problems"] == [] and _only_missing_views(legs, "gsm8k")
    assert util["S"] == 31 and util["L"] == 16640 and util["standup_record"] == ac.STANDUP_JOBS_FILE
    assert cap["applies"] is True and cap["n"] == 3


def test_two_candidates_under_exp_serve_still_refuse(tmp_path: Path):
    root, ar, _ = _gsm8k_root(tmp_path)
    nested = root.serve / "nested"
    nested.mkdir()
    (nested / ac.ENDPOINT_FILE).write_bytes((root.serve / ac.ENDPOINT_FILE).read_bytes())
    with pytest.raises(ac.AuditRefusal, match=r"2 files called 'endpoint.json' under exp/serve/.*ambiguous standup dir"):
        ac.read_standup(ar)
    with pytest.raises(ac.AuditRefusal, match="ambiguous standup dir"):
        ac.vmodel_of_arm(root.path, "none")


def test_cell_copies_alone_are_not_the_standup_record(tmp_path: Path):
    root, ar, _ = _gsm8k_root(tmp_path)
    root.teardown_endpoint(restore=False)
    with pytest.raises(ac.AuditRefusal, match=r"no 'endpoint.json' under exp/serve/ \(the standup record\); the 5 under exp/serve_gsm8k/"
                                              r".*per-cell records, not the standup's"):
        ac.read_standup(ar)


def test_non_gsm8k_root_unchanged_and_a_pre_layout_root_falls_back(tmp_path: Path):
    root, ar = _pasttense_root(tmp_path)
    standup = ac.read_standup(ar)
    assert standup.endpoint_path == root.serve / ac.ENDPOINT_FILE and standup.provenance["standup_dir"] == "exp/serve"
    assert standup.record_source == ac.STANDUP_JOBS_FILE and standup.W == fx.W_FIXTURE
    # a root WITHOUT the declared layout (the pre-layout shape) is read by the whole-root walk, exactly as before
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "serve").mkdir()
    for name in (ac.ENDPOINT_FILE, ac.ATTESTATION_FILE, ac.STANDUP_JOBS_FILE):
        (legacy / "serve" / name).write_bytes((root.serve / name).read_bytes())
    assert ac.standup_dir(legacy) is None
    assert ac.standup_file(legacy, ac.ENDPOINT_FILE) == legacy / "serve" / ac.ENDPOINT_FILE
    (legacy / "other").mkdir()
    (legacy / "other" / ac.ENDPOINT_FILE).write_text("{}")
    with pytest.raises(ac.AuditRefusal, match="2 files called 'endpoint.json'.*ambiguous attempt root"):
        ac.standup_file(legacy, ac.ENDPOINT_FILE)


# --------------------------------------------------------------------------- #
# F-AA: after teardown, and without standup_jobs.jsonl                                                             #
# --------------------------------------------------------------------------- #

def test_after_teardown_the_restored_ready_bytes_are_read(tmp_path: Path):
    """The fy shape: the standup's unlink, then the READY bytes restored at the SAME path — the post-teardown audit reads them."""
    root, ar = _pasttense_root(tmp_path)
    before = (root.serve / ac.ENDPOINT_FILE).read_bytes()
    root.teardown_endpoint(restore=True)
    assert (root.serve / ac.ENDPOINT_FILE).read_bytes() == before
    standup, det, verdicts, rows, legs, util, cap = _every_audit(ar, tmp_path, leg="pasttense")
    assert standup.endpoint_path == root.serve / ac.ENDPOINT_FILE and standup.provenance["endpoint_file"] == "endpoint.json"
    assert det["endpoint_file"] == "endpoint.json" and det["standup_provenance"]["endpoint_path"] == str(root.serve / ac.ENDPOINT_FILE)
    assert [v["verdict"] for v in verdicts] == ["AGREES"] and verdicts[0]["standup_provenance"]["endpoint_file"] == "endpoint.json"
    assert rows[0]["endpoint_path"] == str(root.serve / ac.ENDPOINT_FILE) and rows[0]["problems"] == [] and _only_missing_views(legs, "pasttense")
    assert util["S"] == 45 and cap["applies"] is True


def test_known_bad_no_endpoint_after_teardown_refuses_by_name(tmp_path: Path):
    """The request_evidence_audit refusal (teardown unlinked the record and nothing restored it): refused BY NAME, never a copy."""
    root, ar = _pasttense_root(tmp_path)
    root.teardown_endpoint(restore=False)
    with pytest.raises(ac.AuditRefusal, match=r"no 'endpoint.json' under exp/serve/ \(the standup record\)$"):
        ac.read_standup(ar)


def test_absent_standup_jobs_derives_with_the_derivation_recorded_everywhere(tmp_path: Path):
    root, ar = _pasttense_root(tmp_path)
    root.drop_standup_jobs()
    standup, det, verdicts, rows, legs, util, cap = _every_audit(ar, tmp_path, leg="pasttense")
    assert standup.standup_jobs == [] and standup.standup_jobs_path is None
    assert standup.record_source == ac.STANDUP_RECORD_DERIVED == "derived(endpoint.json+serving_attestation.json; W from ATTEMPT.json)"
    prov = standup.provenance
    assert prov["standup_jobs_jsonl"] == "absent" and prov["derived_from"] == ["endpoint.json", "serving_attestation.json", "ATTEMPT.json"]
    assert prov["W_source"].startswith("ATTEMPT.json W") and "not applicable" in prov["W_source"]
    assert standup.W == fx.W_FIXTURE and standup.S == 45 and standup.L == 11520
    # every consuming table carries the provenance; the sha cross-check is recorded as not applicable, never as passed
    r = rows[0]
    assert r["W"] == fx.W_FIXTURE and r["standup_record"] == ac.STANDUP_RECORD_DERIVED and r["W_source"].startswith("ATTEMPT.json W")
    assert r["problems"] == [] and _only_missing_views(legs, "pasttense")
    assert set(r) == set(eit.COLUMNS)
    assert util["standup_record"] == ac.STANDUP_RECORD_DERIVED and set(util) == {c.rstrip("?") for c in ut.COLUMNS}
    assert det["standup_record"] == ac.STANDUP_RECORD_DERIVED and det["standup_provenance"]["standup_jobs_jsonl"] == "absent"
    assert det["W"] == fx.W_FIXTURE and det["S"] == 45 and det["L"] == 11520
    assert verdicts[0]["verdict"] == "AGREES" and verdicts[0]["standup_record"] == ac.STANDUP_RECORD_DERIVED
    assert cap["applies"] is True


def test_derivation_is_grounded_in_the_workers_report(tmp_path: Path):
    root, ar = _pasttense_root(tmp_path)
    root.drop_standup_jobs()
    root.attestation["scheduler_geometry"]["worker_reports"][0]["max_num_seqs"] = 44
    root._write_serve()
    (root.serve / ac.STANDUP_JOBS_FILE).unlink()
    with pytest.raises(ac.AuditRefusal, match=r"no standup_jobs.jsonl under exp/serve/ and the standup cannot be derived.*"
                                              r"attested_max_num_seqs 45 != the workers' max_num_seqs \[44\]"):
        ac.read_standup(ar)
    root.attestation["scheduler_geometry"] = {}
    root._write_serve()
    (root.serve / ac.STANDUP_JOBS_FILE).unlink()
    with pytest.raises(ac.AuditRefusal, match="worker_reports absent or empty"):
        ac.read_standup(ar)


def test_standup_jobs_cluster_shape_and_modal_shape_accepted_missing_key_refused(tmp_path: Path):
    root, ar = _pasttense_root(tmp_path)
    root.write_standup_jobs_line(CLUSTER_LINE)
    st = ac.read_standup(ar)
    assert st.record_source == ac.STANDUP_JOBS_FILE and st.W == fx.W_FIXTURE and st.provenance["W_source"] == "standup_jobs.jsonl[0].sha"
    root.write_standup_jobs_line(MODAL_LINE)
    st = ac.read_standup(ar)
    assert st.record_source == ac.STANDUP_JOBS_FILE and st.W == fx.W_FIXTURE and st.standup_jobs == [MODAL_LINE]
    rows, legs = eit.build([ar], pins=None, pin_key=eit.DEFAULT_PIN_KEY, W_expected=None)
    assert rows[0]["W"] == fx.W_FIXTURE and rows[0]["W_source"] == "standup_jobs.jsonl[0].sha"
    assert rows[0]["problems"] == [] and _only_missing_views(legs, "pasttense")
    # a sha disagreeing with ATTEMPT.json is still the engine table's problem (the cross-check the record exists for)
    root.write_standup_jobs_line({**MODAL_LINE, "sha": "1" * 40})
    rows, _ = eit.build([ar], pins=None, pin_key=eit.DEFAULT_PIN_KEY, W_expected=None)
    assert any("standup_jobs.jsonl sha" in p and "!= ATTEMPT.json W" in p for p in rows[0]["problems"])
    # a line lacking a required key refuses by name, listing the roster
    root.write_standup_jobs_line({k: v for k, v in MODAL_LINE.items() if k != "sha"})
    with pytest.raises(ac.AuditRefusal, match=r"line 0 lacks \['sha'\] — the standup record's required keys are \['job_id', 'key', 'view', "
                                              r"'mode', 'eval', 'sha'\]"):
        ac.read_standup(ar)
    root.write_standup_jobs_line({"sha": fx.W_FIXTURE})
    with pytest.raises(ac.AuditRefusal, match=r"line 0 lacks \['job_id', 'key', 'view', 'mode', 'eval'\]"):
        ac.read_standup(ar)
    (root.serve / ac.STANDUP_JOBS_FILE).write_text("")
    with pytest.raises(ac.AuditRefusal, match="empty — the standup wrote no job line"):
        ac.read_standup(ar)
