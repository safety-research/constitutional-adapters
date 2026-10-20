"""validate_fit.py: the post-fit predicate of one
seed-replicate cell — realized_batch.validate_fit_record on the ACTUAL
(n_docs, bs, accum, seed) with the record's resume marker, the fingerprint
match, effective batch 64, the seed, no continuation lineage, the cap-hit
flag with its last-four-epoch delta, ce_ratio <= 1.1 — on synthetic cells.
The vectors are the trainer's own (expected_realized_vector), so the
{50, 63, 64} control shape and the {8, 64} constitution shape are
real, not typed."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from sisteer.extraction.realized_batch import (allowed_group_sizes,
                                               expected_realized_vector)

REPO = Path(__file__).resolve().parents[1]
SEEDREP = REPO / "scripts" / "sdfbase" / "seedrep"
if str(SEEDREP) not in sys.path:
    sys.path.insert(0, str(SEEDREP))

import validate_fit as vf  # noqa: E402

KEY = "qwen35_4b_base_eb64_s1"
FP = "f" * 64
N_CTL, N_MIX = 17_073, 17_352      # the control pool and the constitution pool
BS, ACC = 2, 32


def _cell(tmp_path, *, cell="constv2ctl2-full__psr_ce_all", seed=1, n_docs=N_CTL,
          bs=BS, accum=ACC, effective_batch=64, vec=None, resumed=None,
          marker_present=True, stopped_early=True, curve=None, ce_ratio=1.02,
          continued_from=None, fp=FP, key=KEY, seed_in_meta=True):
    d = tmp_path / "bank" / key / "cells_all" / cell
    d.mkdir(parents=True)
    obj = b"\x01" * 2048 + cell.encode()
    sha = hashlib.sha256(obj).hexdigest()
    (d / "object.pt").write_bytes(obj)
    if vec is None:
        vec = expected_realized_vector(n_docs, bs, accum, seed)
    fr = {"realized_eff_batch": vec, "steps_per_epoch": allowed_group_sizes(n_docs, bs, accum)["G"],
          "stopped_early": stopped_early, "best_epoch": 4, "epochs_run": 6,
          "heldout_curve": curve if curve is not None else [2.3, 2.2, 2.19, 2.18, 2.185, 2.18, 2.181],
          "ce_ratio": ce_ratio}
    if marker_present:
        fr["resumed_from_step"] = resumed
    si, arm = cell.split("__")
    meta = {"cell_id": cell, "si_id": si, "arm": arm, "object_sha256": sha,
            "effective_batch": effective_batch, "resolved_batch_size": bs,
            "accum_steps": accum, "batch_size": bs, "fit_record": fr,
            "model_id": "Qwen/Qwen3.5-4B-Base", "model_revision": "1001bb4d"}
    if seed_in_meta:
        meta["seed"] = seed
    if continued_from:
        meta["optimizer"] = {"continued_from": continued_from}
    (d / "train_budget_meta.json").write_text(json.dumps(meta))
    (d / "COMPLETE").write_text(json.dumps({"cell_id": cell, "object_sha256": sha,
                                            "fit_fingerprint": fp}))
    return d


def _ok(v, *names):
    return all(v["checks"][n]["ok"] for n in names)


def test_control_vector_50_63_64_passes_with_a_mid_epoch_63(tmp_path):
    vec = expected_realized_vector(N_CTL, BS, ACC, 1)
    assert set(vec) <= {64, 63, 50, 49} and 63 in vec
    assert vec.index(63) < len(vec) - 1          # the deficit lands MID-epoch
    d = _cell(tmp_path, vec=vec)
    v = vf.validate_cell(d, seed=1, expect_key=KEY, expected_fingerprint=FP, n_docs=N_CTL)
    assert v["ok"], v["checks"]
    rb = v["checks"]["realized_batch"]
    assert rb["resume_provenance"] == "fresh" and rb["exact_ok"] is True
    assert rb["distinct_sizes"] == sorted(set(vec)) and 63 in rb["distinct_sizes"]
    assert v["cap_hit"] is False and v["heldout_delta_last4"] is None


def test_constitution_vector_8_64_passes(tmp_path):
    vec = expected_realized_vector(N_MIX, BS, ACC, 2)
    assert sorted(set(vec)) == [8, 64] and vec[-1] == 8
    d = _cell(tmp_path, cell="constv2-mix-full__lora_r4_ce", seed=2, n_docs=N_MIX, vec=vec)
    v = vf.validate_cell(d, seed=2, expect_key=KEY, expected_fingerprint=FP, n_docs=N_MIX)
    assert v["ok"], v["checks"]


def test_record_without_resume_marker_refuses(tmp_path):
    d = _cell(tmp_path, marker_present=False)
    v = vf.validate_cell(d, seed=1, expect_key=KEY, expected_fingerprint=FP, n_docs=N_CTL)
    assert not v["ok"] and v["failed_checks"] == ["realized_batch"]
    assert v["checks"]["realized_batch"]["resume_provenance"] == "absent"
    assert any("resumed_from_step" in p for p in v["checks"]["realized_batch"]["problems"])


def test_resumed_suffix_passes_the_floor_and_skips_the_exact_check(tmp_path):
    full = expected_realized_vector(N_CTL, BS, ACC, 1)
    d = _cell(tmp_path, vec=full[100:], resumed=100)
    v = vf.validate_cell(d, seed=1, expect_key=KEY, expected_fingerprint=FP, n_docs=N_CTL)
    assert v["ok"], v["checks"]["realized_batch"]
    rb = v["checks"]["realized_batch"]
    assert rb["resume_provenance"] == "resumed" and rb["exact_ok"] is None


def test_wrong_seed_on_an_unresumed_run_fails_the_exact_check(tmp_path):
    v1, v2 = expected_realized_vector(N_CTL, BS, ACC, 1), expected_realized_vector(N_CTL, BS, ACC, 2)
    assert v1 != v2                              # the deficit position moves with the seed
    d = _cell(tmp_path, vec=v1, seed=2)          # sidecar says seed 2, vector is seed 1's
    v = vf.validate_cell(d, seed=2, expect_key=KEY, expected_fingerprint=FP, n_docs=N_CTL)
    assert not v["ok"] and v["failed_checks"] == ["realized_batch"]
    assert v["checks"]["realized_batch"]["exact_ok"] is False


def test_fingerprint_key_batch_seed_scratch_and_ce_checks(tmp_path):
    d = _cell(tmp_path)
    bad = vf.validate_cell(d, seed=1, expect_key=KEY, expected_fingerprint="0" * 64, n_docs=N_CTL)
    assert bad["failed_checks"] == ["fingerprint"]
    bad = vf.validate_cell(d, seed=1, expect_key="qwen35_4b_base_eb64_s2",
                           expected_fingerprint=FP, n_docs=N_CTL)
    assert bad["failed_checks"] == ["key"]
    bad = vf.validate_cell(d, seed=3, expect_key=KEY, expected_fingerprint=FP, n_docs=N_CTL)
    assert "seed" in bad["failed_checks"]        # the exact check moves too
    d8 = _cell(tmp_path / "b8", effective_batch=8, bs=2, accum=4,
               vec=expected_realized_vector(N_CTL, 2, 4, 1))
    bad = vf.validate_cell(d8, seed=1, expect_key=KEY, expected_fingerprint=FP, n_docs=N_CTL)
    assert bad["failed_checks"] == ["effective_batch"]
    d164 = _cell(tmp_path / "b164", bs=1, accum=64, vec=expected_realized_vector(N_CTL, 1, 64, 1))
    assert vf.validate_cell(d164, seed=1, expect_key=KEY, expected_fingerprint=FP,
                            n_docs=N_CTL)["ok"]            # the memprobe fallback is admitted
    dc = _cell(tmp_path / "cont", continued_from={"fit_fingerprint": "x", "cap": 4})
    bad = vf.validate_cell(dc, seed=1, expect_key=KEY, expected_fingerprint=FP, n_docs=N_CTL)
    assert bad["failed_checks"] == ["scratch"]
    assert "optimizer.continued_from" in bad["checks"]["scratch"]["problems"][0]
    dce = _cell(tmp_path / "ce", ce_ratio=1.2)
    bad = vf.validate_cell(dce, seed=1, expect_key=KEY, expected_fingerprint=FP, n_docs=N_CTL)
    assert bad["failed_checks"] == ["ce_ratio"]
    dns = _cell(tmp_path / "noseed", seed_in_meta=False)
    bad = vf.validate_cell(dns, seed=1, expect_key=KEY, expected_fingerprint=FP, n_docs=N_CTL)
    assert bad["failed_checks"] == ["seed"] and "records no seed" in bad["checks"]["seed"]["problems"][0]


def test_cap_hit_is_flagged_with_the_last_four_delta_not_failed(tmp_path):
    curve = [2.5, 2.3, 2.2, 2.15, 2.12, 2.11, 2.105, 2.10]
    d = _cell(tmp_path, stopped_early=False, curve=curve)
    v = vf.validate_cell(d, seed=1, expect_key=KEY, expected_fingerprint=FP, n_docs=N_CTL)
    assert v["ok"] and v["cap_hit"] is True
    assert v["heldout_delta_last4"] == pytest.approx(curve[-1] - curve[-5])
    assert "CAP HIT" in v["note"] and "never re-fit" in v["note"]
    short = _cell(tmp_path / "short", stopped_early=False, curve=[2.5, 2.4, 2.3])
    v = vf.validate_cell(short, seed=1, expect_key=KEY, expected_fingerprint=FP, n_docs=N_CTL)
    assert v["ok"] and v["cap_hit"] and v["heldout_delta_last4"] is None


def test_object_hash_and_marker_agreement(tmp_path):
    d = _cell(tmp_path)
    (d / "object.pt").write_bytes(b"replaced")
    v = vf.validate_cell(d, seed=1, expect_key=KEY, expected_fingerprint=FP, n_docs=N_CTL)
    assert v["failed_checks"] == ["sidecars"]
    assert any("object.pt hashes to" in p for p in v["checks"]["sidecars"]["problems"])


def test_n_docs_comes_from_the_pool_manifest_in_either_spelling(tmp_path):
    d = _cell(tmp_path)
    pool = tmp_path / "pool"
    pool.mkdir()
    pairs = pool / "pairs_constv2ctl2-full.jsonl"
    pairs.write_text("{}\n")
    (pool / "pairs_constv2ctl2-full.manifest.json").write_text(json.dumps({"n_docs": N_CTL}))
    v = vf.validate_cell(d, seed=1, expect_key=KEY, expected_fingerprint=FP, pairs=pairs)
    assert v["ok"] and v["checks"]["realized_batch"]["n_docs"] == N_CTL
    assert v["checks"]["realized_batch"]["n_docs_source"].endswith("pairs_constv2ctl2-full.manifest.json")
    (pool / "pairs_constv2ctl2-full.manifest.json").unlink()
    (pool / "pairs_constv2ctl2-full.jsonl.manifest.json").write_text(json.dumps({"n_docs": N_CTL}))
    assert vf.validate_cell(d, seed=1, expect_key=KEY, expected_fingerprint=FP, pairs=pairs)["ok"]
    (pool / "pairs_constv2ctl2-full.jsonl.manifest.json").unlink()
    v = vf.validate_cell(d, seed=1, expect_key=KEY, expected_fingerprint=FP, pairs=pairs)
    assert v["failed_checks"] == ["realized_batch"] and "n_docs unknown" in \
        v["checks"]["realized_batch"]["problems"][0]


def test_cli_writes_verdict_and_ledger_with_exit_codes(tmp_path):
    d = _cell(tmp_path)
    ledger = tmp_path / "rs" / "reseed" / "fits_ledger.jsonl"
    base = [sys.executable, str(SEEDREP / "validate_fit.py"), str(d), "--seed", "1",
            "--expect-key", KEY, "--expected-fingerprint", FP, "--n-docs", str(N_CTL),
            "--cluster", "cluster_a", "--bank-dir", "/remote/dir"]
    env = {**os.environ, "PYTHONPATH": str(REPO / "src"), "RS": str(tmp_path / "rs")}
    r = subprocess.run(base, capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    verdict = json.loads((d / "VALIDATE.json").read_text())
    assert verdict["ok"] and verdict["bank_path"] == "/remote/dir" and verdict["cluster"] == "cluster_a"
    rows = [json.loads(l) for l in ledger.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["cell"] == "constv2ctl2-full__psr_ce_all" and rows[0]["seed"] == 1
    r = subprocess.run(base[:-4] + ["--expected-fingerprint", "0" * 64, "--ledger", str(ledger),
                                    "--out", str(tmp_path / "v.json")],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 4 and "fingerprint" in r.stdout
    assert json.loads((tmp_path / "v.json").read_text())["ok"] is False
    assert len(ledger.read_text().splitlines()) == 2      # a failed cell is on record too
    r = subprocess.run([sys.executable, str(SEEDREP / "validate_fit.py"), str(tmp_path / "nope"),
                        "--seed", "1", "--expect-key", KEY], capture_output=True, text=True, env=env)
    assert r.returncode == 3


def test_find_continued_from_walks_everything():
    assert vf.find_continued_from({"a": {"continued_from": {"x": 1}}, "b": [{"continued_from": None}]}) \
        == ["a.continued_from"]
    assert vf.find_continued_from({"continued_from": {}}) == []
