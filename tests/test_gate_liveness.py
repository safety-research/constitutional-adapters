"""scripts/sdfbase/gate_liveness.py: per-layer live-gate fractions for saved
PSR-gated objects, written as gate_liveness.json ({layer: live_frac}) + a
detail sidecar beside each object — the round-2 analysis' liveness input for
the gate_reg r3 trials and the derived V / P_resid objects. CPU tiny model,
synthetic pool; a bare (sidecar-less, tune_arm-shaped) trial dir needs
--allow-bare, a paired-write object reports the any-term-open fraction."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from residual_fixtures import random_psr

REPO = Path(__file__).resolve().parents[1]
TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


gl = _load_module("gate_liveness", REPO / "scripts/sdfbase/gate_liveness.py")


def _pool(path: Path, n=4) -> Path:
    rows = [{"prompt": "" if i % 2 else "Sys\n\n", "pos": f"probe doc {i} text "
             * (3 + i), "neg": "", "instruction_id": "constv2-mix-1m",
             "doc_id": f"d{i}", "family": "stories" if i % 2 else "advice"}
            for i in range(n)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _psr(model, seed, b=0.5):
    # an all-layer random gate: unnormalised 0.05 write, 0.3-scale gate
    return random_psr(model, seed, write=0.05, normalize=False,
                      gate_scale=0.3, b=b)


def test_trial_dir_and_paired_dir(tiny_llama, tmp_path):
    from sisteer.extraction.paired_write import PSRPairedWriteObject

    model, tok = tiny_llama
    pool = _pool(tmp_path / "pairs_constv2-mix-1m.jsonl")
    # a tune_arm-shaped trial dir: object.pt + summary.json, no sidecar
    trial = tmp_path / "r3_t00"
    trial.mkdir()
    _psr(model, 1).save(trial / "object.pt")
    (trial / "summary.json").write_text(json.dumps({"best_heldout_loss": 2.0}))
    # a paired-write dir (bare too, for the test)
    pw = tmp_path / "v"
    pw.mkdir()
    PSRPairedWriteObject(_psr(model, 2), _psr(model, 3)).save(pw / "object.pt")
    base = ["--model", TINY, "--device", "cpu", "--dtype", "float32",
            "--pairs", str(pool), "--n-probe", "3", "--max-len", "64"]
    # bare dirs refuse without --allow-bare (identity first)
    with pytest.raises(SystemExit, match="allow-bare"):
        gl.main([*base, "--dirs", str(trial)])
    assert gl.main([*base, "--dirs", f"{trial},{pw}", "--allow-bare"]) == 0
    for d in (trial, pw):
        flat = json.loads((d / "gate_liveness.json").read_text())
        assert set(flat) == {str(l) for l in range(model.config.num_hidden_layers)}
        assert all(0.0 <= v <= 1.0 for v in flat.values())
        det = json.loads((d / "gate_liveness_detail.json").read_text())
        assert det["n_probe_docs"] == 3 and det["identity_kind"] == "bare"
        assert det["object_sha256"] and det["pairs_sha256"]
        assert det["min_live_frac"] == min(flat.values())
    assert json.loads((pw / "gate_liveness_detail.json").read_text())["kind"] == \
        "psr_paired_write"
    assert set(json.loads((pw / "gate_liveness_detail.json").read_text())["terms"]) \
        == {"0", "1"}
    # the psr object's liveness equals the guard's own function
    from sisteer.data.pairs import Pair, docs_to_ce_examples
    from sisteer.extraction.psr import psr_gate_liveness
    from sisteer.methods import load_object

    rows = [json.loads(l) for l in pool.read_text().splitlines()][:3]
    exs = docs_to_ce_examples(None, [Pair(prompt=r["prompt"], pos=r["pos"],
                                          neg="", instruction_id="x")
                                     for r in rows])
    obj = load_object(trial / "object.pt", model=model, tok=tok)
    want = psr_gate_liveness(model, tok, obj=obj, examples=exs, max_len=64,
                             n_probe=3)
    got = json.loads((trial / "gate_liveness.json").read_text())
    for l, v in want.items():
        assert got[str(l)] == pytest.approx(v)


def test_liveness_refuses_non_gated_object(tiny_llama, tmp_path):
    from sisteer.extraction.multilayer import MultiLayerLoRAObject

    model, tok = tiny_llama
    pool = _pool(tmp_path / "pairs_x.jsonl")
    d = tmp_path / "lora"
    d.mkdir()
    MultiLayerLoRAObject(model, rank=4).save(d / "object.pt")
    with pytest.raises(SystemExit, match="psr gates"):
        gl.main(["--model", TINY, "--device", "cpu", "--dtype", "float32",
                 "--pairs", str(pool), "--dirs", str(d), "--allow-bare",
                 "--max-len", "64"])
