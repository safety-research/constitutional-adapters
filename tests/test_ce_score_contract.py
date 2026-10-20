"""ce_score.py's round-2 ANALYSIS output contract (analysis_r2/r2_paths.py):
``--out <x>.json`` writes ONE combined file {ce_active, ce_noop, ratio,
scale, object_sha256, pairs_sha256, pool, docs: [{doc_id, n_target_tokens,
nll_active_sum, nll_noop_sum}]} — no separate summary/jsonl — with ``pool``
from ``--pool`` (the analysis name) or the pairs filename's stem."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cs = _load_module("ce_score_contract", REPO / "scripts/sdfbase/ce_score.py")


def test_pool_name_resolution():
    assert cs.pool_name(None, "/vol/pool/k/pairs_constv2-heldout.jsonl") == \
        "constv2-heldout"
    assert cs.pool_name(None, "pairs_constv2ctl-100k.jsonl") == "constv2ctl-100k"
    assert cs.pool_name("control-heldout", "pairs_constv2ctl-heldout.jsonl") == \
        "control-heldout"
    assert cs.pool_name(None, "other.jsonl") == "other"


def test_combined_json_output(tiny_llama, tmp_path):
    rows = [{"prompt": "" if i % 2 else "Sys\n\n", "pos": f"doc {i} text body "
             * (2 + i), "neg": "", "instruction_id": "constv2ctl-heldout",
             "doc_id": f"ctl-{i}", "family": "stories" if i % 2 else "advice"}
            for i in range(4)]
    pairs = tmp_path / "pairs_constv2ctl-heldout.jsonl"
    pairs.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "ce_score" / "k" / "bare__control-heldout__scale1.json"
    rc = cs.main(["--model", TINY, "--device", "cpu", "--dtype", "float32",
                  "--pairs", str(pairs), "--out", str(out),
                  "--pool", "control-heldout", "--max-len", "64",
                  "--batch-size", "2"])
    assert rc == 0
    assert out.exists() and not Path(str(out) + ".summary.json").exists()
    doc = json.loads(out.read_text())
    for k in ("ce_active", "ce_noop", "ratio", "scale", "object_sha256",
              "pairs_sha256", "pool", "docs"):
        assert k in doc, k
    assert doc["pool"] == "control-heldout" and doc["scale"] == 1.0
    assert doc["object_sha256"] is None  # bare model
    assert len(doc["docs"]) == 4
    for r in doc["docs"]:
        assert {"doc_id", "n_target_tokens", "nll_active_sum",
                "nll_noop_sum"} <= set(r)
    # the aggregate is the sum/sum of the embedded records
    n = sum(r["n_target_tokens"] for r in doc["docs"])
    assert doc["ce_active"] == sum(r["nll_active_sum"] for r in doc["docs"]) / n
    assert doc["ratio"] == 1.0  # bare: active == noop
    # a .jsonl --out keeps the two-file layout (records + summary)
    out2 = tmp_path / "records.jsonl"
    cs.main(["--model", TINY, "--device", "cpu", "--dtype", "float32",
             "--pairs", str(pairs), "--out", str(out2), "--max-len", "64"])
    assert out2.exists() and Path(str(out2) + ".summary.json").exists()
    assert json.loads(Path(str(out2) + ".summary.json").read_text())["pool"] == \
        "constv2ctl-heldout"
