"""decode_policy: the stdlib reader of configs/decode_policy.json resolves eval rows over category rows,
sampling MODES through sampling_mode_policy, refuses a key with no policy (the harness must not invent
numbers), and the export/check round-trips through a synthetic table.
"""
from __future__ import annotations

import json
import os
import subprocess

import pytest

from sisteer.registry import db as DB
from sisteer.registry import decode_policy as DP
from sisteer.registry import harvest as H
from sisteer.registry import records as R
from sisteer.registry import synth

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _policies(root):
    recs = [
        synth.rec("decode_policy", {"policy_id": "p_general", "policy_ref": "synthmodel_4b_it/general", "policy_version": 1,
                                    "model_id": synth.MODEL_ID, "eval_category": "static_attack", "thinking_mode": "off",
                                    "decode_source": "generation_config", "temperature": 0.7, "top_p": 0.8, "top_k": 20,
                                    "max_new_tokens_rule": "cap", "seed_rule": "none", "params_not_sent_json": {"min_p": 0.0},
                                    "source_citation": "generation_config.json@c" * 1, "effective_from_utc": synth.t(1)}, synth.t(1)),
        synth.rec("decode_policy", {"policy_id": "p_eval", "policy_ref": "synthmodel_4b_it/syntheval", "policy_version": 1,
                                    "model_id": synth.MODEL_ID, "eval_id": synth.EVAL_ID, "thinking_mode": "off",
                                    "decode_source": "greedy", "temperature": 0.0, "max_new_tokens_rule": "cap", "seed_rule": "none",
                                    "params_not_sent_json": {"top_p": "engine default 1.0"}, "source_citation": "ruling R_greedy",
                                    "effective_from_utc": synth.t(2)}, synth.t(2)),
        synth.rec("sampling_mode_policy", {"model_id": synth.MODEL_ID, "sampling_mode": "greedy", "policy_id": "p_eval"}, synth.t(3)),
        synth.rec("sampling_mode_policy", {"model_id": synth.MODEL_ID, "sampling_mode": "general", "policy_id": "p_general"}, synth.t(3)),
    ]
    R.write_records(os.path.join(root, "records", "dp", "000001-dp.jsonl"), recs)


@pytest.fixture
def exported(tmp_path):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    R.write_records(os.path.join(root, "records", "dims", "000001-dims.jsonl"), synth.seed_dimensions())
    _policies(root)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.refusals
    conn = DB.connect(DB.db_path(root), readonly=True)
    text = DP.export_json(conn)
    out = str(tmp_path / "decode_policy.json")
    with open(out, "w") as fh:
        fh.write(text)
    assert DP.check_export(conn, out)
    conn.close()
    return out


def test_resolve_prefers_eval_row_over_category(exported):
    p = DP.resolve(synth.MODEL_ID, eval_id=synth.EVAL_ID, eval_category="static_attack", path=exported)
    assert p.policy_id == "p_eval" and p.request_params() == {"temperature": 0.0}
    q = DP.resolve(synth.MODEL_ID, eval_id="other_eval", eval_category="static_attack", path=exported)
    assert q.policy_id == "p_general" and q.request_params() == {"temperature": 0.7, "top_p": 0.8, "top_k": 20}
    assert q.params_not_sent == {"min_p": 0.0}


def test_resolve_by_sampling_mode(exported):
    assert DP.resolve(synth.MODEL_ID, mode="greedy", path=exported).policy_id == "p_eval"
    assert DP.resolve(synth.MODEL_ID, mode="general", path=exported).policy_ref == "synthmodel_4b_it/general"
    with pytest.raises(DP.NoPolicy):
        DP.resolve(synth.MODEL_ID, mode="battery", path=exported)


def test_no_policy_refuses(exported):
    with pytest.raises(DP.NoPolicy):
        DP.resolve("unknown_model", eval_id=synth.EVAL_ID, path=exported)
    with pytest.raises(DP.NoPolicy):
        DP.resolve(synth.MODEL_ID, eval_id="x", eval_category="capability", path=exported)
    with pytest.raises(DP.NoPolicy):
        DP.resolve(synth.MODEL_ID, eval_id=synth.EVAL_ID, thinking_mode="on", path=exported)


def test_export_version_checked_and_reader_is_stdlib(exported, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"export_version": 99, "policies": []}))
    with pytest.raises(ValueError):
        DP.PolicyFile.load(str(bad))
    probe = ("import sys, sisteer.registry.decode_policy as d; p = d.resolve(%r, eval_id=%r, path=%r); "
             "print(p.policy_id, sorted(m for m in sys.modules if m.split('.')[0] in ('pandas','numpy','torch','yaml','sqlite3')))"
             % (synth.MODEL_ID, synth.EVAL_ID, exported))
    out = subprocess.run(["python3", "-S", "-c", probe], capture_output=True, text=True, cwd="/",
                         env={"PYTHONPATH": os.path.join(ROOT, "src"), "PATH": os.environ["PATH"]})
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "p_eval []"


def test_check_export_detects_drift(exported, tmp_path):
    root = str(tmp_path / "reg")
    with open(exported) as fh:
        data = json.load(fh)
    data["policies"][0]["temperature"] = 0.99
    with open(exported, "w") as fh:
        json.dump(data, fh, indent=1, sort_keys=True)
        fh.write("\n")
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert not DP.check_export(conn, exported)
    conn.close()
