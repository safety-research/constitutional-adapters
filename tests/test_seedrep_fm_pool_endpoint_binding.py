"""F-M: a pool-dependent row's producer SNAPSHOTS its attacker pool's live endpoint record into the attempt root and
binds it (`producer_evidence`, seedrep_served.py); the smoke certificate reads it as the `pool_endpoint` check (S3 / S4) and REFUSES changed
bytes -- so a certificate keeps the pool's deployment shape (model revision, cards, allocator config, window, S, app / server / container
identity), which the plan's pool rows (an endpoint PATH) and `PoolEndpoint.identity()` alone did not preserve. Synthetic records throughout."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from test_seedrep_attempts import load_app  # noqa: E402  (the stub-modal loader every seedrep_served test uses)

REPO = Path(__file__).resolve().parents[1]
W = "2b4b5964" + "0" * 32


def _endpoint_record(pool_index: int = 0, **over) -> dict:
    """The shape seedrep_attacker_pool.endpoint_record_fields writes (a synthetic instance; placeholders throughout)."""
    rec = {
        "schema_version": 2, "source": "endpoint.json", "lane": "seedrep", "pool_name": f"seedrep-{pool_index}", "pool_index": pool_index,
        "attack": "crescendo", "pool_dir": f"/vol/attacker-pool/seedrep-{pool_index}", "url": "https://TESTHOST--seedrep-attacker-0.modal.direct",
        "model": "huihui-ai/Huihui-Qwen3.6-27B-abliterated", "snapshot": "27502c8717fd5a2f8c0c77188c10c243fd4f672e",
        "model_path": "/vol/hf/hub/models--x/snapshots/27502c87", "gpu": "H200", "tensor_parallel_size": 1, "max_model_len": 65536,
        "max_num_seqs": 64, "max_num_batched_tokens": 16384, "gpu_memory_utilization": 0.9, "cpu_offload_gb": 0, "enforce_eager": False,
        "limit_mm_per_prompt": '{"image": 0, "video": 0}', "attacker_quantization": "none", "prefix_caching": True,
        "api_key_env_name": "VLLM_API_KEY", "api_key_file": None, "job_id": "sv-11111111", "node": "ta-22222222",
        "started_at": "2026-09-18T07:00:00Z", "expected_lifetime_h": 24.0, "serve_log": f"/vol/attacker-pool/seedrep-{pool_index}/serve.log",
        "repo_sha": W, "runtime": "modal_server",
        "liveness": {"kind": "modal", "app_name": f"seedrep-attacker-{pool_index}", "app_id": "ap-33333333", "server_name": "qwen36-27b-abliterated-vllm",
                     "server_id": "sv-11111111", "call_id": "fc-44444444", "container_id": "ta-22222222"},
        "geometry": {"max_model_len": 65536, "max_num_seqs": 64, "reserve_gib": 2.0, "kv_rule": "peak_mean", "kv_seq_tokens": 5780, "note": "D116 sizing for crescendo ..."},
        "sizing": {"kv_rule": "peak_mean", "S": 64, "S_peak": 64, "S_kv_mean": 139, "p99_fit_S": 24, "mean_seq_tokens": 5780, "p99_seq_tokens": 42230,
                   "L": 65536, "slots_cover_peak": True, "kv_resident_at": {"mean": {"fits": True, "S_fit": 139}, "p99": {"fits": False, "S_fit": 24}},
                   "recommended": {"S": 64, "kv_rule": "peak_mean"}, "caveat": "peak_mean sizes SLOTS ... (synthetic)"},
        "watch_metrics": [{"metric": "vllm:num_requests_waiting", "read": "gauge", "window_s": 600, "threshold": "...", "action": "..."},
                          {"metric": "vllm:num_preemptions_total", "read": "counter_delta", "window_s": 600, "threshold": "delta > 0", "action": "..."}],
        "kv_coverage": {"kv_tokens_reported": 1060000, "concurrency_window": 65536, "max_concurrency_reported": 16.17, "kv_rule": "peak_mean",
                        "kv_seq_tokens": 5780, "p99_seq_tokens": 42230, "p99_coverage_fraction": 0.3922},
        "chunked_prefill": True, "vllm_version": "0.25.1+cu129", "pytorch_cuda_alloc_conf": "expandable_segments:True",
        "params": {"tensor_parallel": 1}, "w7": W[:7], "wave_uuid8": None,
    }
    rec.update(over)
    return rec


@pytest.fixture(scope="module")
def app():
    return load_app()


def _row(pool: str | None) -> dict:
    return {"standup_id": "crescendo__s1__base__shard00", "leg": "crescendo", "view": "base", "stage": "s1", "shard": 0,
            "children": [], "pool": pool}


def _producer(app, tmp_path: Path, row: dict, record: dict | None):
    vol = tmp_path / "vol"
    if record is not None:
        p = vol / "attacker-pool" / "seedrep-0" / "endpoint.json"
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    root = tmp_path / "attempt"
    (root / "children").mkdir(parents=True)
    saved = app.VOL_PATH
    app.VOL_PATH = str(vol)
    try:
        out = app.producer_evidence(root, tree=tmp_path / "tree-without-tools", W=W, row=row, serve_dir=tmp_path / "serve",
                                    driver_python=sys.executable, env={}, run=lambda *a, **k: pytest.fail("no tool should run"))
    finally:
        app.VOL_PATH = saved
    return root, out


def test_a_pool_row_binds_a_byte_identical_snapshot_of_its_pool_endpoint_and_the_certificate_refuses_changed_bytes(app, tmp_path):
    lp = app.lp
    rec = _endpoint_record()
    row = _row("seedrep-attacker-0")
    root, out = _producer(app, tmp_path, row, rec)
    snap = root / lp.POOL_ENDPOINT_SNAPSHOT_REL
    text = snap.read_text()
    assert json.loads(text) == rec                                                      # the pool's bytes, not a re-serialisation
    b = out[lp.POOL_ENDPOINT_SNAPSHOT_REL]
    assert b == {"sha256": hashlib.sha256(snap.read_bytes()).hexdigest(), "writer": "scripts/modal/seedrep_attacker_pool.py", "tree_sha": W, "rc": 0}
    assert b["writer"] == lp.PRODUCER_WRITERS["pool_endpoint"]
    # the certificate's check: the snapshot is this row's pool's, and its facts carry the deployment shape incl. the allocator config
    path = f"{root}/{lp.POOL_ENDPOINT_SNAPSHOT_REL}"
    assert lp.smoke_evidence_paths(row, str(root))["pool_endpoint"] == [path]
    facts = lp._smoke_check("pool_endpoint", text, path=path, plan={"W": W}, accepted_attempts=["att-1"], row=row, aid="att-1", root=str(root), producer=out)
    assert facts["producer"] == {"writer": "scripts/modal/seedrep_attacker_pool.py", "tree_sha": W, "rc": 0}
    assert (facts["tensor_parallel_size"], facts["gpu"], facts["max_num_seqs"], facts["max_model_len"]) == (1, "H200", 64, 65536)
    assert "pytorch_cuda_alloc_conf" in facts and facts["pytorch_cuda_alloc_conf"] == "expandable_segments:True" and facts["snapshot"].startswith("27502c87")
    assert (facts["sizing_S"], facts["kv_rule"], facts["slots_cover_peak"], facts["recommended_S"]) == (64, "peak_mean", True, 64) and facts["kv_resident_at"]["p99"]["fits"] is False
    assert (facts["app_id"], facts["server_id"], facts["container_id"], facts["vllm_version"], facts["repo_sha"]) == ("ap-33333333", "sv-11111111", "ta-22222222", "0.25.1+cu129", W)
    # KNOWN-BAD: the same record with the allocator flipped (or ANY byte changed) is not the bytes the producer bound
    changed = json.dumps({**rec, "pytorch_cuda_alloc_conf": None}, indent=2, sort_keys=True) + "\n"
    with pytest.raises(ValueError, match="is not the bytes the producer bound"):
        lp._smoke_check("pool_endpoint", changed, path=path, plan={"W": W}, accepted_attempts=["att-1"], row=row, aid="att-1", root=str(root), producer=out)
    with pytest.raises(ValueError, match="is not the bytes the producer bound"):
        lp._smoke_check("pool_endpoint", text.replace('"max_num_seqs": 64', '"max_num_seqs": 44'), path=path, plan={"W": W}, accepted_attempts=["att-1"], row=row, aid="att-1", root=str(root), producer=out)
    # KNOWN-BAD: a snapshot of ANOTHER pool bound under this row is refused by name even when its bytes match the binding
    other = json.dumps(_endpoint_record(1), indent=2, sort_keys=True) + "\n"
    (root / "other.json").write_text(other)
    bound_other = {**out, lp.POOL_ENDPOINT_SNAPSHOT_REL: {**b, "sha256": hashlib.sha256(other.encode()).hexdigest()}}
    with pytest.raises(ValueError, match="not this row's pinned pool 'seedrep-attacker-0'"):
        lp._smoke_check("pool_endpoint", other, path=path, plan={"W": W}, accepted_attempts=["att-1"], row=row, aid="att-1", root=str(root), producer=bound_other)
    # KNOWN-BAD: a record lacking the allocator field (a pre-F-M pool) is not the pool Server's record
    old = dict(rec); old.pop("pytorch_cuda_alloc_conf")
    old_text = json.dumps(old, indent=2, sort_keys=True) + "\n"
    bound_old = {**out, lp.POOL_ENDPOINT_SNAPSHOT_REL: {**b, "sha256": hashlib.sha256(old_text.encode()).hexdigest()}}
    with pytest.raises(ValueError, match="lacks \\['pytorch_cuda_alloc_conf'\\]"):
        lp._smoke_check("pool_endpoint", old_text, path=path, plan={"W": W}, accepted_attempts=["att-1"], row=row, aid="att-1", root=str(root), producer=bound_old)


def test_an_absent_pool_record_is_bound_absent_by_name_and_a_row_without_a_pool_has_no_source(app, tmp_path):
    lp = app.lp
    row = _row("seedrep-attacker-0")
    root, out = _producer(app, tmp_path, row, None)
    b = out[lp.POOL_ENDPOINT_SNAPSHOT_REL]
    assert b["absent"] is True and "seedrep-attacker-0's endpoint record is absent" in b["why"] and b["rc"] is None
    assert not (root / lp.POOL_ENDPOINT_SNAPSHOT_REL).exists()
    with pytest.raises(ValueError, match="the producer bound no pool_endpoint.json"):
        lp._smoke_check("pool_endpoint", "{}", path=f"{root}/{lp.POOL_ENDPOINT_SNAPSHOT_REL}", plan={"W": W}, accepted_attempts=["a"], row=row, aid="a",
                        root=str(root), producer=out)
    # a row that pins no pool (controls, bon, ...) neither snapshots nor is asked for one
    row0 = {**_row(None), "leg": "controls"}
    root0, out0 = _producer(app, tmp_path / "b", row0, _endpoint_record())
    assert lp.POOL_ENDPOINT_SNAPSHOT_REL not in out0 and lp.smoke_evidence_paths(row0, str(root0))["pool_endpoint"] == []


def test_the_check_is_wired_for_the_pool_dependent_smokes_and_the_plan_s_pool_rows_are_unchanged(app):
    lp = app.lp
    assert "pool_endpoint" in lp.SMOKE_CHECKS["S3"] and "pool_endpoint" in lp.SMOKE_CHECKS["S4"] and "pool_endpoint" in lp.SMOKE_CHECK_NAMES
    assert "pool_endpoint" not in lp.SMOKE_CHECKS["S1"] and "pool_endpoint" not in lp.SMOKE_CHECKS["S2"]        # no pool rows there
    assert lp.pool_endpoint_record("seedrep-attacker-3") == f"{lp.VOL_PATH}/attacker-pool/seedrep-3/endpoint.json"
    with pytest.raises(ValueError, match="is not seedrep-attacker-<k>"):
        lp.pool_endpoint_record("campaign-pool")
    rows = lp.pool_rows(lp.S4_DEFAULTS, "2b4b596", "0123456789abcdef")
    assert [r["endpoint_record"] for r in rows] == [f"{lp.VOL_PATH}/attacker-pool/seedrep-{k}/endpoint.json" for k in range(4)]
    # PoolEndpoint.identity() -- the projection run manifests keep -- now carries the deployment shape too
    sys.path.insert(0, str(REPO / "src"))
    from sisteer.eval.attacks import attacker_pool as c
    ident = c.PoolEndpoint.from_dict(_endpoint_record()).identity()
    assert ident["pytorch_cuda_alloc_conf"] == "expandable_segments:True" and ident["tensor_parallel_size"] == 1 and ident["gpu"] == "H200" and ident["vllm_version"] == "0.25.1+cu129"
