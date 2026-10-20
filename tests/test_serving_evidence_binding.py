"""serving_evidence_binding: ONE hash, ONE key set, three sites (arditi-qwen-all-positions Phase A review, round 3, item 2).

The standup's persisted serving-evidence record gains an IDENTITY (`stamp`), the drivers fold its POINTER into every
served_target.json they write (`fold_into` -- a no-op with the launcher's env unset, so every production shard is
byte-identical), and the grader FOLLOWS the pointer and re-hashes (`load_record`). This file pins the module's own
contract. Everything synthetic. The two drivers' fold seams (scripts/jailbreak_run.py,
scripts/sdfbase/full22/sampled/escalate_static.py), the shell producer's tests and the grader's land with the
arditi-qwen-all-positions lane; this copy is the serving-side registration PR's, which carries the module alone.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


seb = _load("serving_evidence_binding", REPO / "scripts" / "sdfbase" / "full22" / "serving_evidence_binding.py")

CELL = "arditi_allpos"
OBJ = hashlib.sha256(b"an object").hexdigest()
BODY = {"endpoint": "/synthetic/serving/cell/endpoint.json", "problems": [],
        "facts": {"serving_path_selection.selected": "lora_native", "lens_reduction_mode": None, "expected_served_layer": 12,
                  "residual_slot.op_name": "vllm.sisteer_residual_slot", "residual_adapters": {CELL: {"petri_source_kind": "vector_all", "position_mode": "all"}},
                  "residual_adapter_ops": {CELL: "add_vector"}, "residual_adapter_boundaries": {CELL: [12]},
                  "residual_adapter_fingerprints": {CELL: "fp" * 8}}}
KW = dict(slurm_job_id="123", serve_dir="/synthetic/serving/cell", cell=CELL, object_sha256=OBJ, served_object_sha256=OBJ)


def _stamped(tmp_path: Path, name="rec.json", **over) -> tuple[Path, dict]:
    rec = seb.stamp(BODY, **{**KW, **over})
    p = tmp_path / name
    p.write_text(json.dumps(rec, sort_keys=True, indent=1), encoding="utf-8")
    return p, rec


def test_stamp_writes_the_identity_and_the_hash_covers_everything_but_itself(tmp_path):
    p, rec = _stamped(tmp_path)
    ident = rec[seb.IDENTITY_KEY]
    assert set(ident) == {*seb.IDENTITY_FIELDS, seb.SHA_FIELD}
    assert ident["residual_fingerprint"] == {CELL: "fp" * 8} and ident["expected_served_layer"] == 12 and ident["object_sha256"] == OBJ
    assert ident["served_object_sha256"] == OBJ                                 # round 5: the bytes the bank view observed
    assert ident["slurm_job_id"] == "123" and ident["serve_dir"] == "/synthetic/serving/cell" and ident["cell"] == CELL
    assert {k: v for k, v in rec.items() if k != seb.IDENTITY_KEY} == BODY            # the body is untouched
    # the hash: canonical JSON of the record with ONLY identity.evidence_sha256 absent
    want = dict(BODY); want[seb.IDENTITY_KEY] = {k: v for k, v in ident.items() if k != seb.SHA_FIELD}
    assert ident[seb.SHA_FIELD] == hashlib.sha256(json.dumps(want, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    loaded, sha = seb.load_record(p)
    assert loaded == rec and sha == ident[seb.SHA_FIELD]
    # any edit -- a body fact OR an identity fact -- breaks the re-hash; the sha field itself is the only thing excluded
    for mutate in (lambda r: r["facts"].update(expected_served_layer=11), lambda r: r[seb.IDENTITY_KEY].update(object_sha256="e" * 64),
                   lambda r: r[seb.IDENTITY_KEY].update(slurm_job_id="999")):
        bad = copy.deepcopy(rec); mutate(bad)
        assert seb.canonical_sha256(bad) != sha
    assert seb.canonical_sha256({**rec, seb.IDENTITY_KEY: {**ident, seb.SHA_FIELD: "x"}}) == sha


def test_stamp_refuses_a_body_it_cannot_identify_by_name():
    with pytest.raises(SystemExit, match="a refused evidence body gets no identity"):
        seb.stamp({**BODY, "problems": ["x"]}, **KW)
    with pytest.raises(SystemExit, match="not the ONE body's record"):
        seb.stamp({"facts": 3}, **KW)
    nofp = copy.deepcopy(BODY); del nofp["facts"]["residual_adapter_fingerprints"]
    with pytest.raises(SystemExit, match="records no residual_adapter_fingerprints"):
        seb.stamp(nofp, **KW)
    nolayer = copy.deepcopy(BODY); nolayer["facts"]["expected_served_layer"] = "12"
    with pytest.raises(SystemExit, match="expected_served_layer='12' is not an integer"):
        seb.stamp(nolayer, **KW)
    with pytest.raises(SystemExit, match="object_sha256='abc' is not a 64-hex sha256"):
        seb.stamp(BODY, **{**KW, "object_sha256": "abc"})
    with pytest.raises(SystemExit, match="every identity field is required"):
        seb.stamp(BODY, **{**KW, "cell": ""})
    # round 5: the served sha is required, 64-hex, and EQUAL to the pinned sha -- bytes other than the pinned rung's get no identity
    with pytest.raises(SystemExit, match="served_object_sha256='' is not a 64-hex sha256"):
        seb.stamp(BODY, **{**KW, "served_object_sha256": ""})
    other = hashlib.sha256(b"another rung's object").hexdigest()
    with pytest.raises(SystemExit, match=f"served_object_sha256 {other} != object_sha256 {OBJ} -- the bytes the view served are not the requested rung's pinned bytes"):
        seb.stamp(BODY, **{**KW, "served_object_sha256": other})
    with pytest.raises(TypeError):                                              # the kwarg is REQUIRED: no caller can omit the served bytes
        seb.stamp(BODY, **{k: v for k, v in KW.items() if k != "served_object_sha256"})


def test_load_record_refuses_by_name(tmp_path):
    with pytest.raises(SystemExit, match="no serving-evidence record at"):
        seb.load_record(tmp_path / "absent.json")
    (tmp_path / "notjson.json").write_text("{", encoding="utf-8")
    with pytest.raises(SystemExit, match="is not JSON"):
        seb.load_record(tmp_path / "notjson.json")
    (tmp_path / "bare.json").write_text(json.dumps(BODY), encoding="utf-8")
    with pytest.raises(SystemExit, match="carries no `identity` block"):
        seb.load_record(tmp_path / "bare.json")
    p, rec = _stamped(tmp_path)
    partial = copy.deepcopy(rec); del partial[seb.IDENTITY_KEY]["serve_dir"]
    (tmp_path / "partial.json").write_text(json.dumps(partial), encoding="utf-8")
    with pytest.raises(SystemExit, match=r"identity lacks \['serve_dir'\]"):
        seb.load_record(tmp_path / "partial.json")
    edited = copy.deepcopy(rec); edited["facts"]["expected_served_layer"] = 7
    (tmp_path / "edited.json").write_text(json.dumps(edited), encoding="utf-8")
    with pytest.raises(SystemExit, match="edited after it was stamped"):
        seb.load_record(tmp_path / "edited.json")
    with pytest.raises(SystemExit, match="but the store points at"):
        seb.load_record(p, expect_sha256="0" * 64)
    assert seb.load_record(p, expect_sha256=rec[seb.IDENTITY_KEY][seb.SHA_FIELD])[1] == rec[seb.IDENTITY_KEY][seb.SHA_FIELD]


def test_fold_into_is_a_no_op_with_the_env_unset_and_folds_the_pointer_when_the_launcher_names_a_record(tmp_path):
    p, rec = _stamped(tmp_path)
    ev = {"cell_id": CELL, "intervention_hash": "cc" * 8, "steering_bites": {"verified": True}}
    before = json.dumps(ev, sort_keys=True)
    assert seb.fold_into(ev, env={}) is None and json.dumps(ev, sort_keys=True) == before          # byte-identical: production
    assert seb.fold_into(ev, env={seb.ENV: "   "}) is None and json.dumps(ev, sort_keys=True) == before
    assert seb.fold_into(None, env={seb.ENV: str(p)}) is None
    logs = []
    sha = seb.fold_into(ev, env={seb.ENV: str(p)}, log=logs.append)
    assert sha == rec[seb.IDENTITY_KEY][seb.SHA_FIELD]
    assert ev[seb.EVIDENCE_SHA_KEY] == sha and ev[seb.EVIDENCE_PATH_KEY] == str(p)
    assert set(ev) == {"cell_id", "intervention_hash", "steering_bites", seb.EVIDENCE_SHA_KEY, seb.EVIDENCE_PATH_KEY}
    assert logs and "the grader follows this pointer" in logs[0]
    assert seb.pointer_of({"evidence": ev}) == (sha, str(p)) and seb.pointer_of({"evidence": {"cell_id": CELL}}) is None and seb.pointer_of({}) is None
    # a NAMED record that is absent / edited / unstamped REFUSES -- the launcher meant it; a silently un-bound store is the bug
    with pytest.raises(SystemExit, match="no serving-evidence record at"):
        seb.fold_into({}, env={seb.ENV: str(tmp_path / "absent.json")})
    (tmp_path / "bare.json").write_text(json.dumps(BODY), encoding="utf-8")
    with pytest.raises(SystemExit, match="carries no `identity` block"):
        seb.fold_into({}, env={seb.ENV: str(tmp_path / "bare.json")})
    assert seb.ENV == "SISTEER_SERVING_EVIDENCE_RECORD"

