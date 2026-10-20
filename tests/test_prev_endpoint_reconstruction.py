"""The sampled ladder's 16,384 rung runs as a LATER job and reads the 8,192 standup's
recorded proxy hop, but mt_serving_up.py unlinks endpoint.json at teardown — eight rung legs refused. Three tiers now:
the live endpoint.json, f22_standup's durable copy (endpoint.recorded.json), else a reconstruction from standup_jobs.jsonl +
the tree's hop default, stamped as reconstructed; a missing or overridden record REFUSES."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SAMPLED = REPO / "scripts" / "sdfbase" / "full22" / "sampled"
SCRIPT = SAMPLED / "reconstruct_prev_endpoint.py"
sys.path.insert(0, str(REPO / "src"))


@pytest.fixture(scope="module")
def R():
    spec = importlib.util.spec_from_file_location("reconstruct_prev_endpoint", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _record(**over):
    rec = {"job_id": "95983", "node": "gpu-8", "key": "llama8b", "view": "main", "mode": "attacks", "eval": "families_served:short",
           "sha": "9635d8a49d25eee19b310cd1efa833fd9d9d00ae", "lens_reduction_mode": "stacked", "max_num_seqs": 32, "max_model_len": 14592,
           "attested_max_num_seqs": 32, "attested_max_model_len": 14592, "batched_tokens": 466944, "cluster": "cluster_b"}
    rec.update(over)
    return rec


def test_reconstruction_carries_the_trees_hop_default_and_says_it_is_reconstructed(R, tmp_path):
    from sisteer.capeval.serving.proxy import HOP_TIMEOUT_S
    sd = tmp_path / "static_esc"; sd.mkdir()
    (sd / "standup_jobs.jsonl").write_text(json.dumps(_record(job_id="1")) + "\n" + json.dumps(_record()) + "\n")
    out = subprocess.run([sys.executable, str(SCRIPT), "--standup-dir", str(sd)], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    p = Path(out.stdout.strip())
    assert p == sd / "endpoint.reconstructed.json" and p.is_file()
    doc = json.loads(p.read_text())
    assert doc["upstream_timeout_s"] == float(HOP_TIMEOUT_S) == 7200.0 and doc["proxy_upstream_timeout_s"] is None
    assert doc["job_id"] == "95983" and doc["max_model_len"] == 14592 and doc["attested_max_num_seqs"] == 32   # the LAST record wins
    assert "HOP_TIMEOUT_S" in doc["reconstructed"]["hop_source"] and doc["reconstructed"]["from_record"].endswith("standup_jobs.jsonl")
    # the transport approval's reader takes it as an endpoint record
    from importlib import util as _u
    spec = _u.spec_from_file_location("hop_timeout_guard", SAMPLED / "hop_timeout_guard.py"); hg = _u.module_from_spec(spec); spec.loader.exec_module(hg)
    got = hg.read_endpoint_upstream_timeout(p)
    assert got["seconds"] == 7200.0 and got["key"] == "upstream_timeout_s"
    # idempotent: a second run returns the same file without --force
    again = subprocess.run([sys.executable, str(SCRIPT), "--standup-dir", str(sd)], capture_output=True, text=True)
    assert again.returncode == 0 and Path(again.stdout.strip()) == p


def test_a_live_or_recorded_endpoint_wins_over_reconstruction(R, tmp_path):
    sd = tmp_path / "static_esc"; sd.mkdir()
    (sd / "standup_jobs.jsonl").write_text(json.dumps(_record()) + "\n")
    (sd / "endpoint.recorded.json").write_text(json.dumps({"upstream_timeout_s": 7200.0}))
    out = subprocess.run([sys.executable, str(SCRIPT), "--standup-dir", str(sd)], capture_output=True, text=True)
    assert out.returncode == 0 and Path(out.stdout.strip()) == sd / "endpoint.recorded.json"
    assert not (sd / "endpoint.reconstructed.json").exists()
    (sd / "endpoint.json").write_text(json.dumps({"upstream_timeout_s": 7200.0}))
    out = subprocess.run([sys.executable, str(SCRIPT), "--standup-dir", str(sd)], capture_output=True, text=True)
    assert Path(out.stdout.strip()) == sd / "endpoint.json"


def test_missing_malformed_or_overridden_records_refuse_and_write_nothing(R, tmp_path):
    sd = tmp_path / "static_esc"; sd.mkdir()
    out = subprocess.run([sys.executable, str(SCRIPT), "--standup-dir", str(sd)], capture_output=True, text=True)
    assert out.returncode == 2 and "no standup record" in out.stderr and not (sd / "endpoint.reconstructed.json").exists()
    (sd / "standup_jobs.jsonl").write_text("{not json\n")
    out = subprocess.run([sys.executable, str(SCRIPT), "--standup-dir", str(sd)], capture_output=True, text=True)
    assert out.returncode == 2 and "not JSON" in out.stderr
    (sd / "standup_jobs.jsonl").write_text(json.dumps(_record(upstream_timeout_override=900.0)) + "\n")
    out = subprocess.run([sys.executable, str(SCRIPT), "--standup-dir", str(sd)], capture_output=True, text=True)
    assert out.returncode == 2 and "override" in out.stderr and not (sd / "endpoint.reconstructed.json").exists()


def test_the_shell_side_keeps_the_copy_and_reads_the_three_tiers():
    common = (REPO / "scripts/sdfbase/full22/common.sh").read_text()
    assert 'cp -f "$ep" "$out/endpoint.recorded.json"' in common                         # f22_standup keeps the READY record
    sampled = (SAMPLED / "common_sampled.sh").read_text()
    fn = sampled[sampled.index("f22s_prev_endpoint(){"):]
    fn = fn[:fn.index("\n}\n") + 3]
    # tier order by INVOCATION (a comment names the helper earlier): live record, durable copy, reconstruction
    assert fn.index('/endpoint.json 2>/dev/null') < fn.index('/endpoint.recorded.json 2>/dev/null') < fn.index('"$F22S_DIR/reconstruct_prev_endpoint.py"')
    assert "return 1" in fn                                                                # nothing found still refuses
    assert subprocess.run(["bash", "-n", str(SAMPLED / "common_sampled.sh")], capture_output=True).returncode == 0
    assert subprocess.run(["bash", "-n", str(REPO / "scripts/sdfbase/full22/common.sh")], capture_output=True).returncode == 0
