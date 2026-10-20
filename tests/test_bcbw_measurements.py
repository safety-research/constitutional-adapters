"""The two measurement scripts behind the qwen35_4b it REFUSAL finding — the parent asymmetry inside a
`lora_subtraction` object (mechanism) and the refusal openings in a cell store's rows (behaviour) — pinned on synthetic inputs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import bcbw_lora_subtraction_asymmetry as A     # noqa: E402
import bcbw_refusal_openings as R               # noqa: E402


def _object(scale_pos: float = 3.0, layers=(2, 5), rank: int = 2, d_in: int = 6, d_out: int = 4) -> dict:
    g = torch.Generator().manual_seed(7)
    a = {str(l): torch.randn(d_in, rank, generator=g) for l in layers}
    b = {str(l): torch.randn(d_out, rank, generator=g) for l in layers}
    # the constitution parent is the control parent scaled: every layer's ratio is exactly scale_pos
    return {"kind": A.KIND, "layers": list(layers), "rank": rank, "A_pos": a, "B_pos": {k: v * scale_pos for k, v in b.items()},
            "A_neg": a, "B_neg": b, "provenance": {"derived_id": "derived__x", "model_key": "m", "parent_pos": "p", "parent_neg": "n", "scale_matched": True}}


def test_asymmetry_is_the_per_layer_ratio_of_the_parents_effective_deltas(tmp_path):
    rec = A.measure(_object(scale_pos=3.0))
    assert rec["layers"] == [2, 5] and rec["n_layers"] == 2 and rec["rank"] == 2
    for r in rec["per_layer"]:
        assert r["ratio_pos_over_neg"] == pytest.approx(3.0) and r["pos_norm"] == pytest.approx(3.0 * r["neg_norm"])
        assert r["diff_norm"] == pytest.approx(2.0 * r["neg_norm"])            # 3·ΔW − ΔW = 2·ΔW for a scaled pair
    assert rec["ratio_min"] == pytest.approx(3.0) and rec["ratio_median"] == pytest.approx(3.0) and rec["ratio_max"] == pytest.approx(3.0)
    assert rec["totals"]["pos"] == pytest.approx(3.0 * rec["totals"]["neg"]) and rec["provenance"]["derived_id"] == "derived__x"
    # totals = the global Frobenius norm (sqrt of the sum of squared per-layer norms); the plain sums ride beside it
    assert rec["totals"]["neg"] == pytest.approx(sum(r["neg_norm"] ** 2 for r in rec["per_layer"]) ** 0.5)
    assert rec["layer_norm_sums"]["neg"] == pytest.approx(sum(r["neg_norm"] for r in rec["per_layer"])) and rec["layer_norm_sums"]["neg"] > rec["totals"]["neg"]
    balanced = A.measure(_object(scale_pos=1.0))
    assert balanced["ratio_median"] == pytest.approx(1.0) and all(r["diff_norm"] == pytest.approx(0.0, abs=1e-5) for r in balanced["per_layer"])
    # refusals: not a subtraction object; a missing factor
    with pytest.raises(A.AsymmetryError, match="is not 'lora_subtraction'"):
        A.measure({"kind": "lora", "layers": [1], "rank": 1})
    broken = _object(); broken["B_neg"].pop("5")
    with pytest.raises(A.AsymmetryError, match="layer 5: factor"):
        A.measure(broken)
    # the CLI: one record over labelled objects, the file sha recorded
    path = tmp_path / "object.pt"
    torch.save(_object(scale_pos=2.0), path)
    out = tmp_path / "asym.json"
    assert A.main(["--object", f"synthetic pair={path}", "--out", str(out), "--decision", "D110 (test)"]) == 0
    rec = json.loads(out.read_text())
    (label,) = rec["objects"]
    assert label == "synthetic pair" and rec["objects"][label]["ratio_median"] == pytest.approx(2.0) and len(rec["objects"][label]["sha256"]) == 64
    assert rec["decision"] == "D110 (test)" and rec["generator"].endswith("bcbw_lora_subtraction_asymmetry.py")


def _row(item_id, content, value, tokens=50, stop="stop"):
    return {"item_id": item_id, "value": value, "metric": "acc", "kind": "item",
            "extra_json": json.dumps({"content": content, "completion_tokens": tokens, "stop_reason": stop, "max_tokens_used": stop == "length"})}


def test_refusal_openings_count_the_items_that_open_with_a_refusal_and_quote_them(tmp_path):
    rows = [_row("a", "I can't help with creating code designed to bypass security controls or evade detection systems. I can, however…", 0.0, 72),
            _row("b", "```python\nimport os\n```", 1.0, 400),
            _row("c", "  **I'm sorry**, but I cannot provide that.", 0.0, 20),
            _row("d", "Here is the function you asked for: def f(): pass", 0.0, 300, stop="length"),
            _row("e", "Sure! I can't wait — ```python\nprint(1)\n```", 1.0, 30),           # "Sure!" opens: a refusal formula LATER is not an opening
            {"item_id": R.AGGREGATE_ITEM, "value": 0.4, "metric": "acc_aggregate", "kind": "cell", "extra_json": json.dumps({"at_cap_rate": 0.2})},
            {"item_id": R.AGGREGATE_ITEM, "value": 5.0, "metric": "n", "kind": "cell", "extra_json": None}]
    items = R.item_rows(rows)
    assert [it["item_id"] for it in items] == ["a", "b", "c", "d", "e"]
    m = R.measure(items, examples=2)
    assert (m["n_items"], m["n_refusal_opening"], m["n_fenced_code"], m["n_pass"], m["n_at_cap"]) == (5, 2, 2, 2, 1)
    assert m["refusal_opening_share"] == pytest.approx(0.4) and m["pass_rate"] == pytest.approx(0.4) and m["mean_completion_tokens"] == pytest.approx((72 + 400 + 20 + 300 + 30) / 5)
    assert [e["item_id"] for e in m["refusal_examples"]] == ["a", "c"] and m["refusal_examples"][0]["opening"].startswith("I can't help with creating code")
    assert m["top_openers"][0]["n"] == 1 and len(m["top_openers"]) == 5
    assert R.is_refusal_opening("I cannot provide code that bypasses security controls") and R.is_refusal_opening("Unfortunately, I can't do that")
    assert not R.is_refusal_opening("I can help with that!") and not R.is_refusal_opening("") and not R.is_refusal_opening("The code: I can't")
    with pytest.raises(ValueError, match="no item rows"):
        R.measure([])
    # the CLI: a record per labelled cell, the rows path recorded, the rate reported whatever it is (including zero)
    p = tmp_path / "rows.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    q = tmp_path / "none.jsonl"
    q.write_text(json.dumps(_row("b", "```python\nimport os\n```", 1.0)) + "\n")
    out = tmp_path / "refusals.json"
    assert R.main(["--rows", f"steered={p}", "--rows", f"none={q}", "--out", str(out), "--examples", "1"]) == 0
    rec = json.loads(out.read_text())
    assert rec["cells"]["steered"]["n_refusal_opening"] == 2 and len(rec["cells"]["steered"]["refusal_examples"]) == 1
    assert rec["cells"]["none"]["n_refusal_opening"] == 0 and rec["cells"]["none"]["refusal_opening_share"] == 0.0 and rec["cells"]["none"]["rows"] == str(q)
    assert "by_task" not in rec["cells"]["steered"]                      # one task (or none recorded): no per-task split
    # a store whose rows carry BOTH legs is read per leg too (BFCL's irrelevance items are answered by declining — the task's structure)
    two = [dict(_row("a", "I can't help with that.", 0.0), task="inspect/bfcl_200"), dict(_row("b", "```python\nx=1\n```", 1.0), task="inspect/bigcodebench_200"),
           dict(_row("c", "I cannot do this.", 0.0), task="inspect/bfcl_200")]
    m2 = R.measure_cell(R.item_rows(two))
    assert (m2["n_items"], m2["n_refusal_opening"]) == (3, 2) and set(m2["by_task"]) == {"inspect/bfcl_200", "inspect/bigcodebench_200"}
    assert m2["by_task"]["inspect/bfcl_200"]["refusal_opening_share"] == 1.0 and m2["by_task"]["inspect/bigcodebench_200"]["n_refusal_opening"] == 0
