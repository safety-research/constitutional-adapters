"""The pyrit-free gated loader in scripts/grade_gated_served.py must reproduce
jailbreak_run._load_gated_responses byte-for-byte: same ordered-JSONL merge
(order via the wave's ``sample_idx`` synonym) and the same progress records —
so the served baseline wave's gated families grade on an identical basis to a
normal aggregate, without importing pyrit on the grading host. Synthetic data only
(placeholder ids/text, per repo privacy rule)."""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))


def _write(shard: Path, rows, progress):
    shard.mkdir(parents=True, exist_ok=True)
    (shard / "responses.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    (shard / "progress.jsonl").write_text(
        "".join(json.dumps(p) + "\n" for p in progress))


def test_load_gated_dir_orders_by_sample_idx_and_merges_shards(tmp_path):
    import grade_gated_served as g

    root = tmp_path / "fam" / "cellX"
    # shard00 owns goal A (out of order sample_idx -> must sort), shard01 owns B
    _write(root / "shard00",
           [{"goal_id": "A", "sample_idx": "1", "response": "a1"},
            {"goal_id": "A", "sample_idx": "0", "response": "a0"}],
           [{"goal_id": "A", "n_target_forwards": 2, "capped": False}])
    _write(root / "shard01",
           [{"goal_id": "B", "sample_idx": "0", "response": "b0"}],
           [{"goal_id": "B", "n_target_forwards": 1, "capped": False}])

    responses, progress = g.load_gated_dir(root)
    assert responses == {"A": ["a0", "a1"], "B": ["b0"]}  # A ordered by idx
    assert set(progress) == {"A", "B"}
    assert progress["A"]["n_target_forwards"] == 2


def test_load_gated_dir_matches_load_persisted_responses(tmp_path):
    """When pyrit IS importable, the grading host loader must equal the harness one."""
    pytest.importorskip("pyrit")
    import grade_gated_served as g
    from sisteer.eval.attacks.pyrit_adapters import load_persisted_responses

    root = tmp_path / "fam" / "cellX"
    _write(root / "shard00",
           [{"goal_id": "A", "sample_idx": "2", "response": "a2"},
            {"goal_id": "A", "sample_idx": "0", "response": "a0"},
            {"goal_id": "A", "sample_idx": "1", "response": "a1"}],
           [{"goal_id": "A", "n_target_forwards": 3, "capped": False}])

    ours, _ = g.load_gated_dir(root)
    theirs = load_persisted_responses(root / "shard00" / "responses.jsonl")
    assert ours == theirs
