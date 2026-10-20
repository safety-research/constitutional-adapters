"""The real-artifact exact-equality gate's pure logic
(scripts/petri_artifact_gate.py), CPU-only with a tiny model and fakes —
the GPU job only swaps in the real fp32 target model and the real
artifact bytes.

Load-bearing here:

- the equivalence runner reproduces the synthetic per-kind bar
  (bit-exact per layer/phase/prompt) on the generalized harness, REPORTS
  dead-closed gate layers as evidence (never failures — the tolerant
  ``any(live)`` non-vacuity form), and DETECTS a broken adapter and a
  vacuous all-dead object;
- ``adapter_ops_ctx`` / ``AdapterOpsArmHandle`` thread real
  ``HookContext``s (prefill positions, per-step decode positions)
  through ``model.generate``, so adapter-ops generation is token-exact
  with the object's own ``apply()`` and the scale-0 leg is token-exact
  with the base — the same comparisons the GPU job runs on the real
  model;
- the default-deny refusal checks pass on the REAL adapter registry
  (lora's documented unservable reason; vector_all refusing safetensors
  bytes) and fail when a kind is unexpectedly served;
- the token-leg comparator and the unavailable-vs-failed marker
  discipline.

All fixtures are synthetic: a from-config tiny Llama (no hub access),
generator-seeded tensors, invented ids.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from sisteer.capeval.serving import lens_adapters as la

ROOT = Path(__file__).parents[1]

_spec_path = importlib.util.spec_from_file_location(
    "_petri_artifact_gate", ROOT / "scripts" / "petri_artifact_gate.py")
ag = importlib.util.module_from_spec(_spec_path)
sys.modules["_petri_artifact_gate"] = ag
_spec_path.loader.exec_module(ag)

gate = ag.load_gate_module()

HIDDEN = 32
N_LAYERS = 4
PROMPT_LEN = 7


@pytest.fixture(scope="module")
def tiny_model():
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        hidden_size=HIDDEN, intermediate_size=64,
        num_hidden_layers=N_LAYERS, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=97, max_position_embeddings=64,
        tie_word_embeddings=False)
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg)
    model.eval()
    return model


def _encoded_prompts(n=2, vocab=97, seed=7):
    out = []
    g = torch.Generator().manual_seed(seed)
    for _ in range(n):
        ids = torch.randint(3, vocab, (1, PROMPT_LEN), generator=g)
        out.append((ids, torch.ones_like(ids)))
    return out


def _psr(tiny_model, tmp_path, *, dead_layers=()):
    """A synthetic PSRObject through the REAL save path: live gates on
    every layer except ``dead_layers``, whose gate_b is so negative the
    ReLU is exactly zero on every token."""
    from sisteer.extraction.psr import PSRObject

    layers = [0, 2, 3]
    obj = PSRObject(tiny_model, layers, train_direction=True)
    g = torch.Generator().manual_seed(11)
    for l in layers:
        obj.directions[l].data = torch.randn(HIDDEN, generator=g) * 0.05
        obj.gate_w[l].data = torch.randn(HIDDEN, generator=g) * 0.3
        obj.gate_b[l].data = (torch.full((1,), -1e6) if l in dead_layers
                              else torch.full((1,), 2.0))
    path = tmp_path / "object.pt"
    obj.save(path)
    ops = la.get_adapter("psr")(
        {"kind": "psr", "layers": layers}, path.read_bytes(), b"")
    ops = la.materialize_ops(ops, device="cpu", dtype=torch.float32)
    return obj, ops, layers, path


# --------------------------------------------------------------------------
# the equivalence runner
# --------------------------------------------------------------------------


def test_psr_equivalence_reports_dead_gates_as_evidence(tiny_model, tmp_path):
    """Dead-closed gate layers PASS (the adapter reproduces an
    exactly-zero contribution bit-exactly) and land in the dead map —
    the tolerant any(live) bar, exactly what the real psr artifacts
    need (layers 28/30 dead-closed on the cluster objects)."""
    obj, ops, layers, _ = _psr(tiny_model, tmp_path, dead_layers=(2,))
    out = ag.run_equivalence_check(
        tiny_model, lambda s: obj.apply(tiny_model, scale=s), ops,
        _encoded_prompts())
    assert out.passed, out.mismatches
    assert out.live_layers == (0, 3)
    assert out.dead_layers == (2,)
    assert out.scale0_bit_identical
    assert out.per_layer["2"]["gate_active_fraction"] == 0.0
    assert out.per_layer["2"]["max_abs_delta"] == 0.0
    assert out.per_layer["0"]["gate_active_fraction"] > 0.0
    # every decoder layer was checked, not just op layers
    assert out.layers_checked == tuple(range(N_LAYERS))
    assert out.op_layers == (0, 2, 3)


def test_equivalence_detects_a_broken_adapter(tiny_model, tmp_path):
    """A corrupted op (wrong direction vector) must FAIL the bit-exact
    bar, naming the layer — the check can actually fail."""
    from dataclasses import replace

    obj, ops, layers, _ = _psr(tiny_model, tmp_path)
    broken = tuple(
        replace(op, vector=op.vector + 0.01) if op.layer == 2 else op
        for op in ops)
    out = ag.run_equivalence_check(
        tiny_model, lambda s: obj.apply(tiny_model, scale=s), broken,
        _encoded_prompts())
    assert not out.passed
    assert any("layer 2" in m for m in out.mismatches)


def test_equivalence_flags_a_vacuous_all_dead_object(tiny_model, tmp_path):
    """All gates dead-closed = the equivalence proves nothing — refused
    as VACUOUS (and a red flag for a trained artifact), never a quiet
    pass."""
    obj, ops, layers, _ = _psr(tiny_model, tmp_path, dead_layers=(0, 2, 3))
    out = ag.run_equivalence_check(
        tiny_model, lambda s: obj.apply(tiny_model, scale=s), ops,
        _encoded_prompts())
    assert not out.passed
    assert out.dead_layers == (0, 2, 3)
    assert any("VACUOUS" in m for m in out.mismatches)


def test_vector_all_equivalence_through_the_same_harness(tiny_model,
                                                         tmp_path):
    """The harness is kind-generic: a vector_all object (skip_bos=True —
    the ops genuinely need the HookContext) passes the same bar."""
    from sisteer.extraction.multilayer import MultiLayerVectorObject

    obj = MultiLayerVectorObject(tiny_model)
    g = torch.Generator().manual_seed(13)
    for l in obj.layers:
        obj.params[l].data = torch.randn(HIDDEN, generator=g) * 0.05
    path = tmp_path / "object.pt"
    obj.save(path)
    ops = la.get_adapter("vector_all")(
        {"kind": "vector_all"}, path.read_bytes(), b"")
    ops = la.materialize_ops(ops, device="cpu", dtype=torch.float32)
    out = ag.run_equivalence_check(
        tiny_model, lambda s: obj.apply(tiny_model, scale=s), ops,
        _encoded_prompts())
    assert out.passed, out.mismatches
    assert out.live_layers == tuple(obj.layers)


# --------------------------------------------------------------------------
# adapter_ops_ctx: real HookContexts through model.generate
# --------------------------------------------------------------------------


def _greedy(model, ids, ctx=None, max_new=5):
    import contextlib

    with (ctx if ctx is not None else contextlib.nullcontext()):
        with torch.no_grad():
            out = model.generate(
                input_ids=ids, attention_mask=torch.ones_like(ids),
                max_new_tokens=max_new, do_sample=False, pad_token_id=0)
    return out[0, ids.shape[1]:].tolist()


def test_adapter_ops_generation_is_token_exact_with_the_object(tiny_model,
                                                               tmp_path):
    """End-to-end through model.generate: adapter-ops hooks (real
    positions, prefill + per-step decode) == the object's own apply();
    scale-0 == base. The GPU job runs exactly this on the real model."""
    obj, ops, layers, _ = _psr(tiny_model, tmp_path, dead_layers=(2,))
    ids = _encoded_prompts(n=1)[0][0]
    base = _greedy(tiny_model, ids)
    with_obj = _greedy(tiny_model, ids, obj.apply(tiny_model, scale=1.0))
    handle = ag.AdapterOpsArmHandle(model=tiny_model, ops=ops, scale=1.0,
                                    name="t")
    with_ops = _greedy(tiny_model, ids, handle.ctx())
    handle0 = ag.AdapterOpsArmHandle(model=tiny_model, ops=ops, scale=0.0,
                                     name="t0")
    with_ops0 = _greedy(tiny_model, ids, handle0.ctx())
    assert with_ops == with_obj
    assert with_ops0 == base
    assert with_obj != base  # non-vacuous: the object actually steered
    # a SECOND ctx entry starts fresh (per-call position state)
    assert _greedy(tiny_model, ids, handle.ctx()) == with_obj
    assert handle.ctx_entries == 2


def test_adapter_ops_ctx_refuses_batches(tiny_model, tmp_path):
    obj, ops, layers, _ = _psr(tiny_model, tmp_path)
    ids = torch.randint(3, 97, (2, PROMPT_LEN))
    with pytest.raises(la.AdapterError, match="batch-1"):
        with ag.adapter_ops_ctx(tiny_model, ops, 1.0):
            with torch.no_grad():
                tiny_model(input_ids=ids,
                           attention_mask=torch.ones_like(ids))


def test_adapter_ops_ctx_refuses_layers_beyond_the_model(tiny_model,
                                                         tmp_path):
    obj, ops, layers, _ = _psr(tiny_model, tmp_path)
    from dataclasses import replace

    far = tuple(replace(op, layer=N_LAYERS + 5) for op in ops)
    with pytest.raises(la.AdapterError, match="decoder layer"):
        with ag.adapter_ops_ctx(tiny_model, far, 1.0):
            pass


# --------------------------------------------------------------------------
# token-leg comparison (engine-agnostic, tested with fakes)
# --------------------------------------------------------------------------


def _fake_legs(adapter_break_at=None, scale0_break_at=None):
    n = len(gate.GATE_PROMPT_SET)
    base = {i: [1, 2, 3, i] for i in range(n)}
    steered = {i: [9, 8, i] for i in range(n)}
    calls = {"i": -1}

    def leg(table, brk):
        def fn(messages):
            # identify the prompt by its position in the frozen set
            idx = next(i for i, m in enumerate(gate.GATE_PROMPT_SET)
                       if list(m) == list(messages))
            toks = list(table[idx])
            if brk is not None and idx == brk:
                toks[-1] += 100
            return toks
        return fn

    return {
        "base": leg(base, None),
        "object": leg(steered, None),
        "adapter": leg(steered, adapter_break_at),
        "adapter_scale0": leg(base, scale0_break_at),
    }


def test_token_equality_report_passes_on_exact_legs():
    rep = ag.token_equality_report(gate, _fake_legs())
    assert rep["passed"]
    assert rep["prompts"] == len(gate.GATE_PROMPT_SET)
    assert rep["object_differs_from_base_on_n_prompts"] == rep["prompts"]
    assert all(c["passed"] for c in rep["adapter_vs_object"])
    assert all(c["passed"] for c in rep["adapter_scale0_vs_base"])


def test_token_equality_report_fails_with_divergence_position():
    rep = ag.token_equality_report(gate, _fake_legs(adapter_break_at=2))
    assert not rep["passed"]
    bad = rep["adapter_vs_object"][2]
    assert not bad["passed"] and bad["first_divergence"] == 2
    rep = ag.token_equality_report(gate, _fake_legs(scale0_break_at=1))
    assert not rep["passed"]
    assert not rep["adapter_scale0_vs_base"][1]["passed"]


def test_token_equality_report_requires_all_four_legs():
    legs = _fake_legs()
    legs.pop("adapter_scale0")
    with pytest.raises(ValueError, match="adapter_scale0"):
        ag.token_equality_report(gate, legs)


# --------------------------------------------------------------------------
# default-deny refusal checks on the REAL registry
# --------------------------------------------------------------------------


def test_unservable_refusal_check_passes_with_the_documented_reason():
    # "lora" left this check 2026-08-28 (the o_proj site serves it);
    # circuit_breaker still exercises the checker on a real refusal.
    res = ag.check_lens_kind_refusal(
        "circuit_breaker", must_mention=("UNSERVABLE on the lens path",
                                         "PEFT adapter export",
                                         "native multi-LoRA"))
    assert res["passed"] and res["refused"], res


def test_lora_kind_now_serves_and_the_refusal_check_says_so():
    """The inverted expectation, pinned: a gate still configured to
    expect the old lora refusal must FAIL LOUDLY (never silently pass)
    now that the kind serves."""
    res = ag.check_lens_kind_refusal("lora", must_mention=("anything",))
    assert not res["passed"] and not res["refused"]
    assert "UNEXPECTEDLY served" in res["message"]


def test_refusal_check_fails_when_the_kind_is_served():
    res = ag.check_lens_kind_refusal("psr", must_mention=("anything",))
    assert not res["passed"] and not res["refused"]
    assert "UNEXPECTEDLY served" in res["message"]


def test_refusal_check_fails_on_a_wrong_reason():
    res = ag.check_lens_kind_refusal(
        "circuit_breaker",
        must_mention=("a phrase the reason does not contain",))
    assert not res["passed"] and res["refused"]
    assert res["missing_expected_phrases"]


def test_vector_all_adapter_refuses_safetensors_bytes():
    """The real 'vector' artifact is a safetensors container — the
    vector_all adapter must refuse those bytes (they are not a torch
    blob), the exact check the GPU job runs on the real file."""
    from safetensors.torch import save as st_save

    blob = st_save({"15": torch.randn(8)})
    res = ag.check_adapter_refuses_bytes(
        "vector_all", {"kind": "vector_all"}, blob, b"")
    assert res["passed"] and res["refused"], res


def test_adapter_refusal_check_fails_when_bytes_are_accepted(tiny_model,
                                                             tmp_path):
    """The checker is not refusal-happy: genuine vector_all bytes are
    ACCEPTED by the adapter, and the check reports that as a failure of
    the refusal expectation."""
    from sisteer.extraction.multilayer import MultiLayerVectorObject

    obj = MultiLayerVectorObject(tiny_model)
    path = tmp_path / "object.pt"
    obj.save(path)
    res = ag.check_adapter_refuses_bytes(
        "vector_all", {"kind": "vector_all"}, path.read_bytes(), b"")
    assert not res["passed"]
    assert "ACCEPTED" in res["message"]


# --------------------------------------------------------------------------
# artifact table + marker discipline
# --------------------------------------------------------------------------


def test_artifact_spec_parse():
    a = ag.ArtifactSpec.parse("psr_kl_all:psr:950886be")
    assert (a.name, a.expected_kind, a.sha_prefix) == (
        "psr_kl_all", "psr", "950886be")
    with pytest.raises(ValueError, match="expected_kind"):
        ag.ArtifactSpec.parse("x:notakind:aa")
    with pytest.raises(ValueError):
        ag.ArtifactSpec.parse("no-fields")


def test_artifact_paths(tmp_path):
    d = tmp_path / "vector"
    d.mkdir()
    (d / "object.safetensors").write_bytes(b"x")
    (d / "object.json").write_text("{}")
    path, sidecar = ag.artifact_paths(tmp_path, "vector")
    assert path.name == "object.safetensors" and sidecar.name == "object.json"
    e = tmp_path / "empty"
    e.mkdir()
    with pytest.raises(FileNotFoundError):
        ag.artifact_paths(tmp_path, "empty")
    (e / "object.pt").write_bytes(b"x")
    (e / "object.safetensors").write_bytes(b"x")
    with pytest.raises(FileNotFoundError):  # ambiguous is refused too
        ag.artifact_paths(tmp_path, "empty")


class _WordTok:
    """A deterministic word tokenizer for the gate prompts: ids in
    [3, 97) (the tiny model's vocab), no bos_token (so encode_kwargs
    stays default)."""

    def __call__(self, text, **kwargs):
        ids = [3 + (sum(ord(c) for c in w) % 90) for w in text.split()]
        return {"input_ids": (ids or [5])[:16]}


def _fake_shim(model):
    class _Eng:
        pass

    class _Shim:
        def _render(self, messages):
            return " ".join(str(m["content"]) for m in messages)

    shim = _Shim()
    shim.engine = _Eng()
    shim.engine.model = model
    shim.engine.tokenizer = _WordTok()
    return shim


def _vector_artifact_dir(tmp_path, *, metadata=None, layers=(1, 3), seed=3):
    """A REAL SteeringVector artifact through its own save path —
    object.safetensors + the object.json sidecar load_object dispatches
    on."""
    from sisteer.vectors import SteeringVector

    d = tmp_path / "vector"
    d.mkdir(exist_ok=True)
    g = torch.Generator().manual_seed(seed)
    sv = SteeringVector(
        vectors={l: torch.randn(HIDDEN, generator=g) * 0.05 for l in layers},
        method="distill_soft", model_name="tiny",
        instruction_id="obj_vector", metadata=dict(metadata or {}))
    sv.save(d / "object.safetensors")
    return d, sv


def _run_vector_check(tiny_model, d, artifact_bytes=None):
    return ag._check_vector_artifact(
        _fake_shim(tiny_model), gate,
        path=d / "object.safetensors",
        artifact_bytes=(artifact_bytes if artifact_bytes is not None
                        else (d / "object.safetensors").read_bytes()),
        sidecar_bytes=(d / "object.json").read_bytes())


def test_vector_check_runs_semantic_equivalence_and_passes(tiny_model,
                                                           tmp_path):
    """Kind 'vector' IS registered on the lens path now — the gate's old
    expectation (get_adapter('vector') refuses as ABSENT) is stale and
    would fail every run. The updated check adapts the real bytes AND
    proves them semantically: the ops reproduce VectorObject.apply
    bit-exactly per layer/phase/prompt at BOTH scales, scale-0 is
    bit-identical, and vector_all still default-denies the foreign
    container."""
    assert "vector" in la.registered_kinds()  # the premise of the change
    d, sv = _vector_artifact_dir(tmp_path)
    res = _run_vector_check(tiny_model, d)
    assert res["passed"], res
    assert res["loaded_class"] == "VectorObject"
    assert res["kind_vector_adapter"]["passed"]
    assert res["kind_vector_adapter"]["n_ops"] == 2
    assert res["kind_vector_adapter"]["op_layers"] == [1, 3]
    eq = res["vector_equivalence"]
    assert eq["passed"]
    assert sorted(eq["per_scale"]) == sorted(
        str(s) for s in ag.VECTOR_EQUIVALENCE_SCALES)
    for outcome in eq["per_scale"].values():
        assert outcome["passed"], outcome["mismatches"]
        assert outcome["scale0_bit_identical"]
        assert tuple(outcome["op_layers"]) == (1, 3)
        assert outcome["live_layers"]  # non-vacuous: it actually steered
    assert res["vector_all_adapter_refusal"]["passed"]
    assert res["vector_all_equivalence"]["status"] == "not_runnable"


def test_vector_check_catches_a_semantically_wrong_adapter_input(tiny_model,
                                                                 tmp_path):
    """The bar can actually fail: bytes whose vectors are SIGN-FLIPPED
    relative to the authoritative object adapt fine (n_ops right, layers
    right — the old len(ops)>=1 check passed this) but cannot reproduce
    VectorObject.apply — the equivalence names the divergence and the
    check fails."""
    from safetensors.torch import save as st_save

    d, sv = _vector_artifact_dir(tmp_path)
    flipped = st_save({f"layer_{l}": -t for l, t in sv.vectors.items()})
    res = _run_vector_check(tiny_model, d, artifact_bytes=flipped)
    assert res["kind_vector_adapter"]["passed"]      # adapts fine...
    assert not res["vector_equivalence"]["passed"]   # ...but is WRONG
    assert not res["passed"]
    some = next(iter(res["vector_equivalence"]["per_scale"].values()))
    assert any("do not reproduce" in m for m in some["mismatches"])


def test_vector_check_fails_when_the_registered_adapter_refuses(tiny_model,
                                                                tmp_path):
    """A vector artifact the registered adapter cannot serve (projection
    edit in the metadata) fails the check loudly — a lens-served kind
    whose real bytes do not adapt is a gate failure, not a shrug — and
    the equivalence honestly reports it did not run."""
    d, _sv = _vector_artifact_dir(
        tmp_path, metadata={"projection_targets": {"2": 1.0}})
    res = _run_vector_check(tiny_model, d)
    assert not res["passed"]
    assert res["kind_vector_adapter"]["refused"]
    assert "projection" in res["kind_vector_adapter"]["message"]
    assert not res["vector_equivalence"]["passed"]
    assert "not run" in res["vector_equivalence"]["error"]


def test_clear_stale_markers_removes_all_three(tmp_path):
    ag.record_unavailable(tmp_path, "old run")
    ag.record_failure(tmp_path, "old run")
    ag._write_marker(tmp_path, ag.RECORD_MARKER, {"status": "completed"})
    (tmp_path / ag.EVIDENCE_FILE).write_text("{}")
    ag.clear_stale_markers(tmp_path)
    for name in (ag.UNAVAILABLE_MARKER, ag.FAILURE_MARKER, ag.RECORD_MARKER):
        assert not (tmp_path / name).exists()
    # markers only: the evidence file is rewritten by every run anyway
    assert (tmp_path / ag.EVIDENCE_FILE).exists()


def test_gate_cli_clears_stale_markers_at_run_start(tmp_path):
    """A re-run into the same out dir must not leave a previous run's
    verdict beside this run's: a stale artifact_gate_record.json next to
    a fresh failure (or, here, a fresh unavailable) reads as a success
    to anything keying on the success marker's existence."""
    ag._write_marker(tmp_path, ag.RECORD_MARKER, {"status": "completed"})
    ag.record_failure(tmp_path, "an even older run")
    manifest = ROOT / "configs" / "capeval" / "manifests" / "petri_twosi.yaml"
    rc = ag.main([
        "gate", "--manifest", str(manifest),
        "--artifacts-root", str(tmp_path / "does-not-exist"),
        "--artifact", "vector:steering_vector:00",
        "--out-dir", str(tmp_path)])
    assert rc == 0  # bootstrap failure -> recorded unavailable
    assert (tmp_path / ag.UNAVAILABLE_MARKER).is_file()
    # exactly ONE verdict on disk — the stale ones are gone
    assert not (tmp_path / ag.RECORD_MARKER).exists()
    assert not (tmp_path / ag.FAILURE_MARKER).exists()


def test_marker_discipline(tmp_path):
    """Unavailable and FAILED are distinct on-disk markers — a validation
    failure must never look like a recorded skip."""
    m1 = ag.record_unavailable(tmp_path, "no model")
    m2 = ag.record_failure(tmp_path, "equivalence broke")
    assert m1.name == "artifact_gate_unavailable.json"
    assert m2.name == "artifact_gate_failed.json"
    assert m1.name != m2.name != ag.RECORD_MARKER
    assert json.loads(m1.read_text())["status"] == (
        "artifact gate unavailable")
    assert "FAILED" in json.loads(m2.read_text())["status"]
    # the proxy-vLLM served leg is recorded unavailable with the exact
    # deferred-wiring reason, never silently skipped
    assert "NotImplementedError" in ag.PROXY_VLLM_UNAVAILABLE_REASON
