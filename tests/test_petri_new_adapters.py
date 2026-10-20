"""The six newest lens adapters (alphakl / static_r / cosine_gated_r /
lowrank_gated / loreft_both / vector): equivalence against the
AUTHORITATIVE object arithmetic + refusals, CPU-only.

Follows tests/test_petri_lens_adapters.py's load-bearing pattern: build a
synthetic artifact, adapt the exact bytes into hook ops, run a real
tiny-model forward (prefill + one KV-cached decode step) under the
object's own hooks, capture every decoder layer's hidden states
immediately BEFORE and AFTER those hooks, and assert ``apply_hook_ops``
maps pre to post BIT-EXACTLY.

Two evidence tiers, stated per test:

- ROUND-TRIP (alphakl, vector, and the loreft phase-contrast): the real
  class is importable in this worktree, so the artifact goes through the
  class's own ``save`` and the forward runs under the class's own
  ``apply(model, scale)``.
- SCHEMA-PINNED (static_r, cosine_gated_r, lowrank_gated, loreft_both):
  the classes live only on an earlier development branch
  (src/sisteer/extraction/baseline_arms.py ~314-727 and its low-rank
  module, lines ~708-935, there) and cannot be
  imported here. Artifact bytes are constructed to those classes' exact
  documented ``save`` schemas, and the reference forward runs under hooks
  whose bodies are copied VERBATIM from those classes' ``apply`` — cited
  inline — so these tests pin the schema + arithmetic rather than
  round-tripping the class.

Known divergences found while writing this file are left as
strict-xfail tests (the correct expectation stays in the test body):

- loreft_both is not BIT-exact: the shared OP_AFFINE_SUBSPACE body is
  reft.py-verbatim (``lin(h) - rot(h)``), while the both-phase LoReFT object
  computes ``h @ (W - R).T + b`` — same math, different float rounding.
- static_r / cosine_gated_r under a bf16 serving dtype: both classes
  accumulate ``h.float() + mag * r`` in fp32 and cast ONCE, while the
  served ops round the addend to the hidden dtype and add in it.
- static_r serves the rs-mapping's layers without cross-checking the
  blob's own 'layers' list, which StaticRObject.load_blob treats as
  authoritative (extra rs entries are silently dropped by the class).

All fixtures are synthetic: a from-config tiny Llama (no hub access),
generator-seeded random tensors, invented ids.
"""

import inspect
import io
import json
from contextlib import contextmanager

import pytest
import torch

from sisteer.capeval.serving import lens_adapters as la

HIDDEN = 32
N_LAYERS = 4
PROMPT_LEN = 7


# --------------------------------------------------------------------------
# tiny model + the pre/post capture harness
# (mirrors tests/test_petri_lens_adapters.py — copied, not imported, so a
# concurrent edit of that file cannot break this one)
# --------------------------------------------------------------------------


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


def _prompt_batch(vocab=97, seed=7):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(3, vocab, (1, PROMPT_LEN), generator=g)
    mask = torch.ones(1, PROMPT_LEN, dtype=torch.long)
    next_id = torch.randint(3, vocab, (1, 1), generator=g)
    next_mask = torch.ones(1, PROMPT_LEN + 1, dtype=torch.long)
    return ids, mask, next_id, next_mask


@contextmanager
def _null_ctx():
    yield


def _run_pre_post(model, obj_apply_ctx):
    """Run prefill + one KV-cached decode step; capture every decoder
    layer's output hidden states immediately BEFORE (pre) and AFTER
    (post) the object's hooks (forward hooks fire in registration
    order). Returns ({layer: [prefill, decode]} x2)."""
    from sisteer.models import get_decoder_layers

    decoder = get_decoder_layers(model)
    pre = {l: [] for l in range(len(decoder))}
    post = {l: [] for l in range(len(decoder))}

    def mk(store, l):
        def hook(_mod, _args, output):
            h = output[0] if isinstance(output, tuple) else output
            store[l].append(h.detach().clone())
        return hook

    ids, mask, next_id, next_mask = _prompt_batch()
    pre_handles = [decoder[l].register_forward_hook(mk(pre, l))
                   for l in range(len(decoder))]
    try:
        with obj_apply_ctx:
            post_handles = [decoder[l].register_forward_hook(mk(post, l))
                            for l in range(len(decoder))]
            try:
                with torch.no_grad():
                    out = model(input_ids=ids, attention_mask=mask,
                                use_cache=True)
                    model(input_ids=next_id, attention_mask=next_mask,
                          past_key_values=out.past_key_values,
                          use_cache=True)
            finally:
                for h in post_handles:
                    h.remove()
    finally:
        for h in pre_handles:
            h.remove()
    return pre, post


PREFILL_CTX = la.HookContext(la.PHASE_PREFILL, tuple(range(PROMPT_LEN)))
DECODE_CTX = la.HookContext(la.PHASE_DECODE, (PROMPT_LEN,))


def _assert_equivalent(pre, post, ops, op_layers, scale=1.0):
    """apply_hook_ops on each captured pre-tensor must equal the object's
    own post-tensor BIT-EXACTLY, at every layer, both phases."""
    for l in range(N_LAYERS):
        for phase_idx, ctx in ((0, PREFILL_CTX), (1, DECODE_CTX)):
            mine = la.apply_hook_ops(pre[l][phase_idx], ops, scale, layer=l,
                                     ctx=ctx)
            assert torch.equal(mine, post[l][phase_idx]), (
                f"layer {l} phase {ctx.phase}: adapter ops do not "
                "reproduce the object's own transformation")
            if l not in op_layers:
                assert mine is pre[l][phase_idx]  # untouched, same tensor


@pytest.fixture(scope="module")
def base_acts(tiny_model):
    """Every layer's UNHOOKED [prefill, decode] block outputs — the
    pre-steering activations any single-op-layer test can craft gates
    against."""
    pre, _ = _run_pre_post(tiny_model, _null_ctx())
    return pre


def _blob(payload) -> bytes:
    buf = io.BytesIO()
    torch.save(payload, buf)
    return buf.getvalue()


# --------------------------------------------------------------------------
# the coverage map: adding/removing a kind is a deliberate test change
# --------------------------------------------------------------------------


def test_registered_and_documented_unservable_kind_sets_are_exact():
    assert set(la.registered_kinds()) == {
        la.SYNTHETIC_KIND, la.ZERO_KIND,
        "psr", "vector_all", "antidote", "alphasteer", "loreft",
        "alphakl", "static_r", "cosine_gated_r",
        # caa_static_r (E45 H2): the Class B CAA floor, wired end to end. Its
        # OWN kind, not static_r — CaaStaticRObject.save records no
        # steer_positions / site, and static_r's defaults for an absent pair
        # would serve a one-layer last-prompt-token block-OUTPUT edit instead
        # of the multi-layer, all-prompt-position resid_pre one it implements.
        "caa_static_r",
        "lowrank_gated", "loreft_both", "vector",
        # stacked routing: the prompt+object wrapper kind, which
        # delegates to the object leg's own adapter
        # (STACKED_KIND == manifest.PETRI_STACKED_KIND).
        la.STACKED_KIND,
        # 2026-08-28: the o_proj module site (HOOK_SITE_OPROJ) landed,
        # making the two lora kinds servable on the lens path — the
        # mid-block edit is exact at the module, and it is the only
        # committed path that can serve a positions="response" LoRA fit
        # (native multi-LoRA applies an adapter at every position).
        "lora", "lora_all",
        # 2026-09-06: the sdf-base-steering DERIVED difference object
        # (multilayer.LoRASubtractionObject — the corpus LoRA fit minus its
        # control fit). A new blob SCHEMA served through the SAME o_proj op
        # as its lora_all parents, twice per layer at coeff +1 / -1.
        # Registered rather than materialised as one rank-8 lora_all: the
        # two separate rank-4 reductions are what the authoritative class
        # computes, and a concatenated rank-8 GEMM would be a different
        # reduction order over different bytes — i.e. a different object
        # identity than the one staged, pinned and sha-verified.
        "lora_subtraction",
        # 2026-09-17 (relu-differenced-native-qwen): the UNDISTILLED paired write
        # (extraction/paired_write.PSRPairedWriteObject) — both psr parents' gates
        # read the SAME pre-write state and their writes are subtracted in fp32,
        # one op per layer (OP_RELU_GATED_VECTOR_PAIR). Its own kind: a two-sided
        # blob no single-gate adapter reads, and a function two sequential psr
        # hooks would NOT compute. Registered so the certification's lens legs
        # (the not_slower_at_S head-to-head) stand the paired cell up.
        "psr_paired_write",
    }
    assert set(la.UNSERVABLE_KIND_REASONS) == {
        "circuit_breaker", "token", "lofit", "cast",
        "fishback", "gated_fishback",
    }
    # no kind is both adapted and documented-unservable
    assert not set(la.registered_kinds()) & set(la.UNSERVABLE_KIND_REASONS)


def _guard_forwarded_kwargs():
    """The keyword names the fit/serve guard passes to every adapter, READ
    FROM THE GUARD — this file never keeps a copy of them.

    A hardcoded ``("header_ids", "turn_end_ids")`` here is complete only
    until the guard grows a THIRD keyword: the copy would stay green while
    the class the caller pins — a registered adapter the guard calls with a
    keyword it does not accept, i.e. ``TypeError`` on every cell of that
    kind — quietly reopened. So the tuple is read off the wrapper
    ``lens_adapters._fit_serve_guarded`` builds, the same wrapper
    ``register_adapter`` installs on every kind, and it widens when the
    guard widens.

    ``follow_wrapped=False`` is load-bearing: the wrapper is
    ``functools.wraps``-ed onto the adapter it wraps, so the default read
    walks ``__wrapped__`` and hands back the ADAPTER's signature — the very
    thing under test — which would compare each adapter against itself and
    pass vacuously."""
    def _probe(resolved_spec, artifact_bytes, sidecar_bytes=b"", **kw):
        return ()

    params = inspect.signature(
        la._fit_serve_guarded("__guard_kwargs_probe__", _probe),
        follow_wrapped=False).parameters
    assert not any(p.kind is inspect.Parameter.VAR_KEYWORD
                   for p in params.values()), (
        "the guard's wrapper grew a **kwargs, so the keywords it forwards "
        "are no longer enumerable from its signature and this derivation "
        "is no longer exhaustive — give lens_adapters a module-level "
        "constant that the guard's call site BUILDS ITS CALL FROM, and "
        "read that constant here instead")
    return tuple(name for name, p in params.items()
                 if p.kind is inspect.Parameter.KEYWORD_ONLY)


def test_every_registered_adapter_accepts_the_keywords_the_guard_passes():
    """``register_adapter``'s wrapper calls EVERY adapter with both
    ``header_ids=`` and ``turn_end_ids=``, so an adapter that omits one is a
    ``TypeError`` at adapt time and its whole kind is unservable.

    WHY THIS IS A SEPARATE TEST FROM THE COVERAGE MAP ABOVE. The map catches a
    kind that was never registered; this catches a kind that IS registered and
    whose signature fell behind the calling convention — the failure mode a
    merge produces and neither side can see. ``turn_end_ids`` arrived with the
    all-turns mask, which widened every adapter it could see; ``caa_static_r``
    was added on a branch the widening never touched, so the two landed green
    apart and the merged tree raised ``TypeError: _adapt_caa_static_r() got an
    unexpected keyword argument 'turn_end_ids'`` on every caa_static_r cell.
    Checked by SIGNATURE rather than by adapting bytes so a new kind is covered
    the moment it registers, with no synthetic artifact to write.

    The keyword set itself is DERIVED from the guard
    (:func:`_guard_forwarded_kwargs`) and never copied into this file, so a
    third keyword on the guard is covered the moment it is added rather than
    leaving this test green over a hole."""
    guard_kwargs = _guard_forwarded_kwargs()
    # ...and the derivation is itself checked: a read that came back empty,
    # or that lost one of the two keywords the guard passes today, would
    # make the loop below vacuous — the same silent green the derivation
    # exists to prevent.
    assert guard_kwargs and set(guard_kwargs) >= {"header_ids",
                                                  "turn_end_ids"}, (
        "the guard's forwarded-keyword set read back as "
        f"{guard_kwargs!r}: it must be non-empty and still carry both "
        "keywords the guard passes today, or the per-adapter check below "
        "passes on adapters that would raise TypeError when served")
    behind = {}
    for kind in la.registered_kinds():
        fn = la.get_adapter(kind)
        params = inspect.signature(inspect.unwrap(fn)).parameters
        missing = [k for k in guard_kwargs
                   if k not in params
                   and not any(p.kind is inspect.Parameter.VAR_KEYWORD
                               for p in params.values())]
        if missing:
            behind[kind] = missing
    assert not behind, (
        "these registered adapters do not accept a keyword the guard passes "
        "them, so every cell of their kind raises TypeError at adapt time: "
        + repr(behind))


# ==========================================================================
# alphakl — ROUND-TRIP against extraction/alphakl.AlphaKLObject
# ==========================================================================


def _make_alphakl(tiny_model, layers, ws, rs, base_scale, clamp_gate,
                  tmp_path):
    from sisteer.extraction.alphakl import AlphaKLObject

    obj = AlphaKLObject(tiny_model, layers, ws, rs, base_scale=base_scale,
                        clamp_gate=clamp_gate)
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "object.pt"
    obj.save(path)
    return obj, path.read_bytes()


def _alphakl_fixture_tensors(seed=101):
    g = torch.Generator().manual_seed(seed)
    layers = [1, 3]
    # deliberately NOT unit-norm: the gate's magnitude lives in w
    ws = {l: torch.randn(HIDDEN, generator=g) * 0.3 for l in layers}
    rs = {l: torch.randn(HIDDEN, generator=g) * 0.05 for l in layers}
    return layers, ws, rs


def test_alphakl_roundtrip_ops(tiny_model, tmp_path):
    """clamp_gate=True -> the relu op; base_scale baked as coeff; w passed
    through UNMODIFIED (bit-equal to the saved tensor, and visibly not
    unit-norm — this fails if anyone normalizes it); the gate has NO bias
    (an exactly-zero single-element gate_b)."""
    layers, ws, rs = _alphakl_fixture_tensors()
    _, art = _make_alphakl(tiny_model, layers, ws, rs, 1.7, True, tmp_path)

    ops = la.get_adapter("alphakl")(
        {"kind": "alphakl", "layers": layers, "clamp_gate": True,
         "base_scale": 1.7}, art, b"")
    assert [op.layer for op in ops] == layers
    for op, l in zip(ops, layers):
        assert op.op == la.OP_RELU_GATED_VECTOR
        assert op.coeff == 1.7                       # base_scale baked
        assert op.positions == la.POSITIONS_ALL
        assert not op.skip_bos
        assert op.phase == la.PHASE_BOTH
        # w verbatim, and NOT unit-norm — a normalization would trip both
        assert torch.equal(op.gate_w, ws[l])
        assert abs(float(op.gate_w.norm()) - 1.0) > 0.25, \
            "fixture too weak: w's norm must sit away from 1"
        assert torch.equal(op.vector, rs[l])
        # no bias term: exactly one element, exactly zero
        assert op.gate_b.numel() == 1
        assert torch.equal(op.gate_b, torch.zeros(1))


def test_alphakl_clamp_false_maps_to_the_linear_op(tiny_model, tmp_path):
    layers, ws, rs = _alphakl_fixture_tensors()
    _, art = _make_alphakl(tiny_model, layers, ws, rs, 1.7, False, tmp_path)
    ops = la.get_adapter("alphakl")({"kind": "alphakl"}, art, b"")
    assert all(op.op == la.OP_LINEAR_GATED_VECTOR for op in ops)


@pytest.mark.parametrize("clamp_gate", [True, False])
def test_alphakl_adapter_reproduces_alphaklobject_apply(
        tiny_model, tmp_path, base_acts, clamp_gate):
    """Bit-exact equivalence against AlphaKLObject.apply, both gate modes,
    prefill AND decode (the arm steers generated tokens too). The fixture
    guarantees the first op layer sees projections of BOTH signs, so the
    relu/linear distinction is live, not vacuous."""
    layers, ws, rs = _alphakl_fixture_tensors()
    # layer 1 is the first op layer: its pre-steering prefill acts equal
    # the unhooked base acts, so the sign mix can be asserted up front
    proj = base_acts[layers[0]][0].float() @ ws[layers[0]]
    assert bool((proj > 0).any()) and bool((proj < 0).any()), \
        "fixture too weak: need both gate signs at the first op layer"

    obj, art = _make_alphakl(tiny_model, layers, ws, rs, 1.7, clamp_gate,
                             tmp_path)
    ops = la.get_adapter("alphakl")({"kind": "alphakl"}, art, b"")

    pre, post = _run_pre_post(tiny_model, obj.apply(tiny_model, scale=1.0))
    # non-vacuous: the gate fired somewhere, in BOTH phases
    assert any(not torch.equal(pre[l][0], post[l][0]) for l in layers)
    assert any(not torch.equal(pre[l][1], post[l][1]) for l in layers)
    _assert_equivalent(pre, post, ops, set(layers))


def test_alphakl_relu_and_linear_genuinely_differ_on_negative_projection(
        tiny_model, tmp_path, base_acts):
    """The two gate modes are different INTERVENTIONS, not a tolerance:
    on the same forward (whose first op layer sees a negative projection,
    asserted above the run) the clamped and unclamped objects — and their
    adapted ops — produce different streams."""
    layers, ws, rs = _alphakl_fixture_tensors()
    l0 = layers[0]
    proj = base_acts[l0][0].float() @ ws[l0]
    assert bool((proj < 0).any())

    _, art_relu = _make_alphakl(tiny_model, layers, ws, rs, 1.7, True,
                                tmp_path / "relu")
    _, art_lin = _make_alphakl(tiny_model, layers, ws, rs, 1.7, False,
                               tmp_path / "lin")
    ops_relu = la.get_adapter("alphakl")({"kind": "alphakl"}, art_relu, b"")
    ops_lin = la.get_adapter("alphakl")({"kind": "alphakl"}, art_lin, b"")

    hidden = base_acts[l0][0]
    out_relu = la.apply_hook_ops(hidden, ops_relu, 1.0, layer=l0,
                                 ctx=PREFILL_CTX)
    out_lin = la.apply_hook_ops(hidden, ops_lin, 1.0, layer=l0,
                                ctx=PREFILL_CTX)
    assert not torch.equal(out_relu, out_lin)
    # and each matches the class's own arithmetic (alphakl.py:619-637),
    # hand-computed: gate = (w.h) [clamped iff relu], add = mag*g*r, cast
    g = hidden.float() @ ws[l0]
    for out, gate in ((out_relu, g.clamp_min(0.0)), (out_lin, g)):
        want = hidden + (1.7 * gate.unsqueeze(-1) * rs[l0]).to(hidden.dtype)
        assert torch.equal(out, want)


def test_alphakl_selector_scale_multiplies_the_baked_base_scale(
        tiny_model, tmp_path):
    """coeff carries the object's base_scale; the runtime scale is the
    SELECTOR's and multiplies it — apply_hook_ops at selector scale s must
    equal AlphaKLObject.apply(scale=s) (mag = base_scale * s)."""
    layers, ws, rs = _alphakl_fixture_tensors()
    obj, art = _make_alphakl(tiny_model, layers, ws, rs, 2.0, True, tmp_path)
    ops = la.get_adapter("alphakl")({"kind": "alphakl"}, art, b"")
    assert all(op.coeff == 2.0 for op in ops)

    pre, post = _run_pre_post(tiny_model, obj.apply(tiny_model, scale=0.25))
    assert any(not torch.equal(pre[l][0], post[l][0]) for l in layers)
    _assert_equivalent(pre, post, ops, set(layers), scale=0.25)


def test_alphakl_bf16_cast_semantics_match_the_class():
    """Positive control for the fp32 materialization policy: the class
    computes the gate and add in fp32 and casts the ADD once
    (alphakl.py:634-637: ``hidden + add.to(hidden.dtype)``) — exactly the
    op body's structure, so a bf16 hidden state is served bit-exactly."""
    g = torch.Generator().manual_seed(103)
    hidden = (torch.randn(1, PROMPT_LEN, HIDDEN, generator=g)
              .to(torch.bfloat16))
    w = torch.randn(HIDDEN, generator=g) * 0.3
    r = torch.randn(HIDDEN, generator=g) * 0.05
    art = _blob({"kind": "alphakl", "layers": [1], "ws": {1: w},
                 "rs": {1: r}, "base_scale": 1.3, "clamp_gate": True,
                 "stage": None, "metadata": {}})
    ops = la.get_adapter("alphakl")({"kind": "alphakl"}, art, b"")
    mat = la.materialize_ops(ops, device="cpu", dtype=torch.bfloat16)
    assert mat[0].gate_w.dtype == torch.float32   # fp32 gate math
    got = la.apply_hook_ops(hidden, mat, 1.0, layer=1, ctx=PREFILL_CTX)
    # AlphaKLObject's hook, verbatim, on the bf16 hidden:
    gate = (hidden.float() @ w).clamp_min(0.0)
    add = 1.3 * gate.unsqueeze(-1) * r
    want = hidden + add.to(hidden.dtype)
    assert torch.equal(got, want)


def _alphakl_blob(**over):
    g = torch.Generator().manual_seed(107)
    payload = {"kind": "alphakl", "layers": [1],
               "ws": {1: torch.randn(HIDDEN, generator=g) * 0.3},
               "rs": {1: torch.randn(HIDDEN, generator=g) * 0.05},
               "base_scale": 1.0, "clamp_gate": True, "stage": None,
               "metadata": {}}
    payload.update(over)
    return _blob(payload)


def test_alphakl_refusals():
    adapt = la.get_adapter("alphakl")
    with pytest.raises(la.AdapterError, match="sidecar"):
        adapt({"kind": "alphakl"}, _alphakl_blob(), b"stray")
    with pytest.raises(la.AdapterError, match="wrong adapter"):
        adapt({"kind": "alphakl"}, _alphakl_blob(kind="psr"), b"")
    payload_missing = {"kind": "alphakl", "layers": [1],
                       "ws": {1: torch.ones(HIDDEN)}, "base_scale": 1.0}
    with pytest.raises(la.AdapterError, match="lacks 'rs'"):
        adapt({"kind": "alphakl"}, _blob(payload_missing), b"")
    bad_w = torch.ones(HIDDEN)
    bad_w[3] = float("nan")
    with pytest.raises(la.AdapterError, match="non-finite"):
        adapt({"kind": "alphakl"}, _alphakl_blob(ws={1: bad_w}), b"")
    with pytest.raises(la.AdapterError, match="bool"):
        adapt({"kind": "alphakl"}, _alphakl_blob(clamp_gate=1), b"")
    with pytest.raises(la.AdapterError, match="cross-check"):
        adapt({"kind": "alphakl", "clamp_gate": False}, _alphakl_blob(), b"")
    with pytest.raises(la.AdapterError, match="cross-check"):
        adapt({"kind": "alphakl", "layers": [2]}, _alphakl_blob(), b"")
    # a layer named in 'layers' but absent from the ws mapping refuses
    with pytest.raises(la.AdapterError, match="lacks layer"):
        adapt({"kind": "alphakl"}, _alphakl_blob(layers=[1, 2]), b"")


# ==========================================================================
# static_r — SCHEMA-PINNED (StaticRObject lives on the earlier development branch;
# blobs below follow its save() schema, reference hooks copy its apply()
# verbatim: baseline_arms.py ~383-451 there)
# ==========================================================================


def _static_r_rich_blob(layers, rs, base_scale, steer_positions, site,
                        **over):
    """StaticRObject.save's exact schema: normalized layers/rs plus the
    legacy top-level layer/r pair (the tensor when single-layer, None
    when multi)."""
    single = len(layers) == 1
    l0 = layers[0]
    payload = {"kind": "static_r",
               "layer": l0 if single else None,
               "r": rs[l0] if single else None,
               "layers": list(layers),
               "rs": dict(rs),
               "steer_positions": steer_positions,
               "site": site,
               "base_scale": base_scale, "metadata": {}}
    payload.update(over)
    return _blob(payload)


def _static_r_legacy_blob(layer, r, base_scale):
    """The pre-multi-layer artifact format (the landed A1 shape): a bare
    layer/r pair, no steer_positions or site keys — the class's load
    defaults them to prompt_last/post."""
    return _blob({"kind": "static_r", "layer": layer, "r": r,
                  "base_scale": base_scale, "metadata": {}})


@contextmanager
def _static_r_post_reference(model, layers, rs, mag, steer_positions):
    """StaticRObject.apply for site='post', VERBATIM (the earlier development branch's
    baseline_arms.py:383-416 — _edited + make_post): decode forwards
    (seq_len <= 1) are never edited; prompt_last edits row -1 only,
    prefill_all every position; the add runs in fp32, cast once."""
    from sisteer.models import get_decoder_layers

    decoder = get_decoder_layers(model)

    def _edited(hidden, r_master):
        if hidden.shape[1] <= 1:
            return None
        r = r_master.to(device=hidden.device)
        if steer_positions == "prefill_all":
            return (hidden.float() + mag * r).to(hidden.dtype)
        edited = hidden.clone()
        edited[:, -1, :] = (
            hidden[:, -1, :].float() + mag * r).to(hidden.dtype)
        return edited

    def make_post(r_master):
        def hook(_mod, _args, output):
            if mag == 0.0:
                return output
            hidden = output[0] if isinstance(output, tuple) else output
            edited = _edited(hidden, r_master)
            if edited is None:
                return output
            if isinstance(output, tuple):
                return (edited,) + output[1:]
            return edited
        return hook

    handles = [decoder[l].register_forward_hook(make_post(rs[l]))
               for l in layers]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def _full_run(model, install):
    """Prefill + one KV-cached decode step with ``install(decoder)``'s
    hooks live; capture every block's output AFTER those hooks, plus the
    prefill and decode logits."""
    from sisteer.models import get_decoder_layers

    decoder = get_decoder_layers(model)
    caps = {l: [] for l in range(len(decoder))}

    def mk(l):
        def hook(_mod, _args, output):
            h = output[0] if isinstance(output, tuple) else output
            caps[l].append(h.detach().clone())
        return hook

    ids, mask, next_id, next_mask = _prompt_batch()
    hooks = install(decoder)
    cap_handles = [decoder[l].register_forward_hook(mk(l))
                   for l in range(len(decoder))]
    try:
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=mask, use_cache=True)
            dec = model(input_ids=next_id, attention_mask=next_mask,
                        past_key_values=out.past_key_values, use_cache=True)
    finally:
        for h in cap_handles:
            h.remove()
        for h in hooks:
            h.remove()
    return caps, out.logits.detach().clone(), dec.logits.detach().clone()


def _ops_installer(ops, scale=1.0):
    """Block-output hooks driving apply_hook_ops with the right phase
    context per forward — the worker hook body, CPU-shaped."""
    def install(decoder):
        state = {"seen": 0}

        def mk(l):
            def hook(_mod, _args, output):
                h = output[0] if isinstance(output, tuple) else output
                S = h.shape[1]
                if S > 1:
                    ctx = la.HookContext(la.PHASE_PREFILL, tuple(range(S)))
                    state["seen"] = S
                else:
                    ctx = la.HookContext(la.PHASE_DECODE, (state["seen"],))
                    state["seen"] += 1
                new = la.apply_hook_ops(h, ops, scale, layer=l, ctx=ctx)
                if new is h:
                    return None
                if isinstance(output, tuple):
                    return (new,) + output[1:]
                return new
            return hook
        return [decoder[l].register_forward_hook(mk(l))
                for l in range(len(decoder))]
    return install


def _static_r_pre_installer(layers, rs, mag):
    """StaticRObject.apply for site='pre', VERBATIM (the earlier development branch's
    baseline_arms.py:418-446 — make_pre, prefill_all): the edit lands on
    the hidden states ENTERING each steered block, both calling
    conventions handled; decode forwards pass through."""
    def _edited(hidden, r_master):
        if hidden.shape[1] <= 1:
            return None
        r = r_master.to(device=hidden.device)
        return (hidden.float() + mag * r).to(hidden.dtype)

    def install(decoder):
        def make_pre(r_master):
            def hook(_mod, args, kwargs):
                if mag == 0.0:
                    return None
                if "hidden_states" in kwargs:
                    hidden = kwargs["hidden_states"]
                elif args and torch.is_tensor(args[0]):
                    hidden = args[0]
                else:
                    return None
                edited = _edited(hidden, r_master)
                if edited is None:
                    return None
                if "hidden_states" in kwargs:
                    return (args, {**kwargs, "hidden_states": edited})
                return ((edited,) + tuple(args[1:]), kwargs)
            return hook
        return [decoder[l].register_forward_pre_hook(make_pre(rs[l]),
                                                     with_kwargs=True)
                for l in layers]
    return install


def _static_r_dirs(layers, seed=211):
    g = torch.Generator().manual_seed(seed)
    return {l: torch.randn(HIDDEN, generator=g) * 0.05 for l in layers}


@pytest.mark.parametrize("legacy", [True, False],
                         ids=["legacy_blob", "rich_blob"])
def test_static_r_prompt_last_post_roundtrip_and_equivalence(
        tiny_model, legacy):
    """The prompt_last/post shape (arm static_r), from BOTH artifact
    generations: op = add_vector at the object's own layer,
    positions='last', prefill-ONLY, base_scale baked as coeff. Equivalence
    against the class's verbatim hook: only the last prompt row moves at
    prefill; the decode step is bit-untouched (KV-cache persistence)."""
    rs = _static_r_dirs([2])
    if legacy:
        art = _static_r_legacy_blob(2, rs[2], 1.5)
    else:
        art = _static_r_rich_blob([2], rs, 1.5, "prompt_last", "post")

    ops = la.get_adapter("static_r")({"kind": "static_r"}, art, b"")
    (op,) = ops
    assert op.op == la.OP_ADD_VECTOR and op.layer == 2
    assert op.positions == la.POSITIONS_LAST and not op.skip_bos
    assert op.phase == la.PHASE_PREFILL
    assert op.coeff == 1.5
    assert torch.equal(op.vector, rs[2])

    pre, post = _run_pre_post(
        tiny_model,
        _static_r_post_reference(tiny_model, [2], rs, 1.5, "prompt_last"))
    assert not torch.equal(pre[2][0], post[2][0])   # prefill steered
    assert torch.equal(pre[2][0][:, :-1, :], post[2][0][:, :-1, :])
    assert not torch.equal(pre[2][0][:, -1, :], post[2][0][:, -1, :])
    assert torch.equal(pre[2][1], post[2][1])       # decode untouched
    _assert_equivalent(pre, post, ops, {2})
    # decode is NOT INVOKED (the same tensor), never a masked application
    out = la.apply_hook_ops(pre[2][1], ops, 1.0, layer=2, ctx=DECODE_CTX)
    assert out is pre[2][1]


def test_static_r_prefill_all_pre_serves_one_block_earlier(tiny_model):
    """THE SITE SHIFT: a resid_pre object (the static_r_matched shape —
    prefill_all/pre) must serve at layer L-1's OUTPUT, since resid_pre of
    block L is the output of block L-1. Full-forward equivalence: the
    class's verbatim pre-hooks at layers [1, 3] and the adapter's ops at
    [0, 2] must yield byte-identical downstream streams and logits, at
    prefill AND decode."""
    layers = [1, 3]
    rs = _static_r_dirs(layers)
    art = _static_r_rich_blob(layers, rs, 0.9, "prefill_all", "pre")

    ops = la.get_adapter("static_r")(
        {"kind": "static_r", "steer_positions": "prefill_all",
         "site": "pre"}, art, b"")
    assert [op.layer for op in ops] == [0, 2]       # one block EARLIER
    assert all(op.positions == la.POSITIONS_ALL and not op.skip_bos
               and op.phase == la.PHASE_PREFILL and op.coeff == 0.9
               for op in ops)

    base_caps, base_logits, base_dec = _full_run(tiny_model,
                                                 lambda d: [])
    ref_caps, ref_logits, ref_dec = _full_run(
        tiny_model, _static_r_pre_installer(layers, rs, 0.9))
    ops_caps, ops_logits, ops_dec = _full_run(tiny_model,
                                              _ops_installer(ops))
    # non-vacuous: the intervention moved the model
    assert not torch.equal(base_logits, ref_logits)
    # identical logits, prefill and decode — byte for byte
    assert torch.equal(ref_logits, ops_logits)
    assert torch.equal(ref_dec, ops_dec)
    # identical streams at every block whose observation point agrees
    # (layers 0 and 2 differ by WHERE the edit is observed: the reference
    # edits block 1/3's INPUT, the adapter block 0/2's OUTPUT — the same
    # tensor of the residual stream, captured on opposite sides)
    for l in (1, 3):
        for phase in (0, 1):
            assert torch.equal(ref_caps[l][phase], ops_caps[l][phase])


def test_static_r_resid_pre_of_layer_zero_refuses(tiny_model):
    """resid_pre of layer 0 is the embedding output — no preceding block
    can carry the edit; serving it one block later would be a DIFFERENT
    intervention, so the adapter must refuse."""
    rs = _static_r_dirs([0])
    art = _static_r_rich_blob([0], rs, 1.0, "prefill_all", "pre")
    with pytest.raises(la.AdapterError, match="embedding"):
        la.get_adapter("static_r")({"kind": "static_r"}, art, b"")


def test_static_r_refusals():
    adapt = la.get_adapter("static_r")
    rs = _static_r_dirs([1])
    good = _static_r_rich_blob([1], rs, 1.0, "prompt_last", "post")
    with pytest.raises(la.AdapterError, match="sidecar"):
        adapt({"kind": "static_r"}, good, b"stray")
    with pytest.raises(la.AdapterError, match="steer_positions"):
        adapt({"kind": "static_r"},
              _static_r_rich_blob([1], rs, 1.0, "decode_all", "post"), b"")
    with pytest.raises(la.AdapterError, match="unknown hook site"):
        adapt({"kind": "static_r"},
              _static_r_rich_blob([1], rs, 1.0, "prompt_last", "mid"), b"")
    with pytest.raises(la.AdapterError, match="lacks 'base_scale'"):
        adapt({"kind": "static_r"},
              _blob({"kind": "static_r", "layer": 1, "r": rs[1]}), b"")
    with pytest.raises(la.AdapterError, match="no direction"):
        adapt({"kind": "static_r"},
              _blob({"kind": "static_r", "layer": None, "r": None,
                     "layers": [], "rs": {}, "base_scale": 1.0}), b"")
    with pytest.raises(la.AdapterError, match="cross-check"):
        adapt({"kind": "static_r", "steer_positions": "prefill_all"},
              good, b"")
    with pytest.raises(la.AdapterError, match="cross-check"):
        adapt({"kind": "static_r", "site": "pre"}, good, b"")
    bad_r = rs[1].clone()
    bad_r[0] = float("inf")
    with pytest.raises(la.AdapterError, match="non-finite"):
        adapt({"kind": "static_r"}, _static_r_legacy_blob(1, bad_r, 1.0),
              b"")


def test_static_r_rs_inconsistent_with_stored_layers_refuses():
    """load_blob keys the object off 'layers' and silently DROPS rs
    entries outside it, so an artifact whose two records disagree would
    otherwise be served as a WIDER intervention than load_object
    hydrates."""
    rs = _static_r_dirs([1, 2])
    art = _blob({"kind": "static_r", "layer": None, "r": None,
                 "layers": [1],          # says one layer...
                 "rs": dict(rs),         # ...but carries two
                 "steer_positions": "prompt_last", "site": "post",
                 "base_scale": 1.0, "metadata": {}})
    with pytest.raises(la.AdapterError, match="layers"):
        la.get_adapter("static_r")({"kind": "static_r"}, art, b"")


@pytest.mark.xfail(
    strict=True,
    reason="BUG (bf16 serving dtype): StaticRObject accumulates in fp32 "
    "and casts ONCE — edited row = (h.float() + mag*r).to(dtype) "
    "(the earlier development branch's baseline_arms.py:396-399) — while the served op "
    "materializes the vector to the MODEL dtype and adds in it "
    "(OP_ADD_VECTOR's 'model' policy), i.e. h + (r_bf16*eff). Under bf16 "
    "the two double-round differently (3/32 coordinates differ on this "
    "seeded fixture); under fp32 they coincide, so only this test sees "
    "it.")
def test_static_r_bf16_add_matches_the_class_fp32_accumulate():
    torch.manual_seed(0)
    hidden = torch.randn(1, PROMPT_LEN, HIDDEN).to(torch.bfloat16)
    r = torch.randn(HIDDEN) * 0.03
    art = _static_r_legacy_blob(1, r, 1.3)
    ops = la.get_adapter("static_r")({"kind": "static_r"}, art, b"")
    mat = la.materialize_ops(ops, device="cpu", dtype=torch.bfloat16)
    got = la.apply_hook_ops(hidden, mat, 1.0, layer=1, ctx=PREFILL_CTX)
    # the class's verbatim arithmetic on the same bf16 stream:
    want = hidden.clone()
    want[:, -1, :] = (hidden[:, -1, :].float() + 1.3 * r).to(hidden.dtype)
    assert torch.equal(got, want)


# ==========================================================================
# cosine_gated_r — SCHEMA-PINNED (CosineGatedRObject, the earlier development branch's
# baseline_arms.py ~648-727; _cosine ibid. ~203-208)
# ==========================================================================


def _cosine_ref(h, d):
    """baseline_arms._cosine, verbatim: fp32, 1e-12 denominator floor."""
    hf = h.float()
    df = d.float()
    return (hf @ df) / (hf.norm(dim=-1) * df.norm() + 1e-12)


def _cos_blob(layer, r, theta, d_cond, base_scale, **over):
    payload = {"kind": "cosine_gated_r", "layer": layer, "r": r,
               "theta": theta, "d_cond": d_cond,
               "base_scale": base_scale, "metadata": {}}
    payload.update(over)
    return _blob(payload)


@contextmanager
def _cosine_reference(model, layer, r, theta, d_cond, mag):
    """CosineGatedRObject.apply, VERBATIM (the earlier development branch's
    baseline_arms.py:684-707): per-request hard gate on
    cos(h_last, d_cond) > theta (STRICT), firing rows get mag*r on the
    last prompt token in fp32 (cast once); non-firing forwards and decode
    steps return the hook input unchanged."""
    from sisteer.models import get_decoder_layers

    def hook(_mod, _args, output):
        if mag == 0.0:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape[1] <= 1:
            return output  # decode step: KV-cache persistence
        mask = _cosine_ref(hidden[:, -1, :],
                           d_cond.to(hidden.device)) > theta
        if not bool(mask.any()):
            return output  # no row fires: bit-identical to unsteered
        rr = r.to(device=hidden.device)
        edited = hidden.clone()
        edited[mask, -1, :] = (
            hidden[mask, -1, :].float() + mag * rr).to(hidden.dtype)
        if isinstance(output, tuple):
            return (edited,) + output[1:]
        return edited

    h = get_decoder_layers(model)[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()


def _cos_fixture(seed=307):
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(HIDDEN, generator=g) * 0.05
    d_cond = torch.randn(HIDDEN, generator=g)
    return r, d_cond


def test_cosine_gated_r_roundtrip_and_equivalence_when_firing(
        tiny_model, base_acts):
    """Round-trip of the save schema plus bit-exact equivalence against
    the class's verbatim hook on a real forward that FIRES: theta is set
    strictly below the request's actual last-token cosine. Only the last
    prompt token moves; the decode step is untouched."""
    r, d_cond = _cos_fixture()
    # layer 2 is the only op layer, so its pre-steering prefill acts are
    # the unhooked base acts — the real cosine is computable up front
    cos_val = float(_cosine_ref(base_acts[2][0][:, -1, :], d_cond)[0])
    theta = cos_val - 0.25

    art = _cos_blob(2, r, theta, d_cond, 1.5)
    ops = la.get_adapter("cosine_gated_r")(
        {"kind": "cosine_gated_r", "layer": 2, "theta": theta}, art, b"")
    (op,) = ops
    assert op.op == la.OP_PREFILL_LAST_COSINE_GATED and op.layer == 2
    assert op.threshold == theta and op.coeff == 1.5
    assert op.positions == la.POSITIONS_LAST and op.phase == la.PHASE_PREFILL
    assert torch.equal(op.vector, r) and torch.equal(op.cond, d_cond)

    pre, post = _run_pre_post(
        tiny_model, _cosine_reference(tiny_model, 2, r, theta, d_cond, 1.5))
    assert not torch.equal(pre[2][0], post[2][0])   # it fired
    assert torch.equal(pre[2][0][:, :-1, :], post[2][0][:, :-1, :])
    assert torch.equal(pre[2][1], post[2][1])       # decode untouched
    _assert_equivalent(pre, post, ops, {2})
    # decode-phase application is NOT INVOKED (the same tensor)
    out = la.apply_hook_ops(pre[2][1], ops, 1.0, layer=2, ctx=DECODE_CTX)
    assert out is pre[2][1]


def test_cosine_gate_is_strict_and_non_firing_is_bit_identical():
    """A cosine EXACTLY AT theta does not fire (strict '>'), and any
    non-firing request comes back as the SAME tensor — asserted by
    identity, not closeness. One epsilon below theta, it fires."""
    r, d_cond = _cos_fixture()
    g = torch.Generator().manual_seed(311)
    hidden = torch.randn(1, PROMPT_LEN, HIDDEN, generator=g)
    # theta == the op's own fp32 cosine of this exact request
    cos_val = float(_cosine_ref(hidden[:, -1, :], d_cond)[0])

    at = la.get_adapter("cosine_gated_r")(
        {"kind": "cosine_gated_r"}, _cos_blob(1, r, cos_val, d_cond, 1.5),
        b"")
    out_at = la.apply_hook_ops(hidden, at, 1.0, layer=1, ctx=PREFILL_CTX)
    assert out_at is hidden                          # AT theta: no fire

    below = la.get_adapter("cosine_gated_r")(
        {"kind": "cosine_gated_r"},
        _cos_blob(1, r, cos_val - 1e-4, d_cond, 1.5), b"")
    out_below = la.apply_hook_ops(hidden, below, 1.0, layer=1,
                                  ctx=PREFILL_CTX)
    assert not torch.equal(out_below, hidden)        # strictly above: fires
    # ...and matches the class's arithmetic on the fired row
    want = hidden.clone()
    want[:, -1, :] = (hidden[:, -1, :].float() + 1.5 * r).to(hidden.dtype)
    assert torch.equal(out_below, want)

    above = la.get_adapter("cosine_gated_r")(
        {"kind": "cosine_gated_r"}, _cos_blob(1, r, 2.0, d_cond, 1.5), b"")
    out_above = la.apply_hook_ops(hidden, above, 1.0, layer=1,
                                  ctx=PREFILL_CTX)
    assert out_above is hidden                       # can never fire


def test_cosine_gated_r_refusals():
    adapt = la.get_adapter("cosine_gated_r")
    r, d_cond = _cos_fixture()
    good = _cos_blob(1, r, 0.1, d_cond, 1.0)
    with pytest.raises(la.AdapterError, match="sidecar"):
        adapt({"kind": "cosine_gated_r"}, good, b"stray")
    with pytest.raises(la.AdapterError, match="zero norm"):
        adapt({"kind": "cosine_gated_r"},
              _cos_blob(1, r, 0.1, torch.zeros(HIDDEN), 1.0), b"")
    with pytest.raises(la.AdapterError, match="lacks 'theta'"):
        adapt({"kind": "cosine_gated_r"},
              _blob({"kind": "cosine_gated_r", "layer": 1, "r": r,
                     "d_cond": d_cond, "base_scale": 1.0}), b"")
    with pytest.raises(la.AdapterError, match="finite"):
        adapt({"kind": "cosine_gated_r"},
              _cos_blob(1, r, float("inf"), d_cond, 1.0), b"")
    with pytest.raises(la.AdapterError, match="cross-check"):
        adapt({"kind": "cosine_gated_r", "theta": 0.9}, good, b"")
    with pytest.raises(la.AdapterError, match="wrong adapter"):
        adapt({"kind": "cosine_gated_r"},
              _static_r_legacy_blob(1, r, 1.0), b"")


@pytest.mark.xfail(
    strict=True,
    reason="BUG (bf16 serving dtype): CosineGatedRObject writes the fired "
    "row as (h.float() + mag*r).to(dtype) — fp32 accumulate, ONE cast "
    "(the earlier development branch's baseline_arms.py:698-700) — while the served op rounds "
    "the fp32 delta to the hidden dtype FIRST and then adds in bf16 "
    "(out + (eff*vector*mask).to(hidden.dtype)). The two double-round "
    "differently under bf16 (3/32 coordinates on this seeded fixture); "
    "fp32 coincides, so the fp32 equivalence tests stay green.")
def test_cosine_gated_r_bf16_matches_the_class_fp32_accumulate():
    torch.manual_seed(0)
    hidden = torch.randn(1, PROMPT_LEN, HIDDEN).to(torch.bfloat16)
    r = torch.randn(HIDDEN) * 0.03
    d_cond = torch.randn(HIDDEN)
    art = _cos_blob(1, r, -2.0, d_cond, 1.3)   # theta=-2: always fires
    ops = la.get_adapter("cosine_gated_r")({"kind": "cosine_gated_r"},
                                           art, b"")
    mat = la.materialize_ops(ops, device="cpu", dtype=torch.bfloat16)
    got = la.apply_hook_ops(hidden, mat, 1.0, layer=1, ctx=PREFILL_CTX)
    want = hidden.clone()
    want[:, -1, :] = (hidden[:, -1, :].float() + 1.3 * r).to(hidden.dtype)
    assert torch.equal(got, want)


# ==========================================================================
# lowrank_gated — SCHEMA-PINNED (the low-rank gated object; the earlier
# development branch's low-rank module, lines ~708-821)
# ==========================================================================


def _lowrank_gated_blob(layers, rank, As, Bs, base_scale, clamp_gate,
                      **over):
    payload = {"kind": "lowrank_gated", "layers": list(layers), "rank": rank,
               "As": dict(As), "Bs": dict(Bs), "base_scale": base_scale,
               "clamp_gate": clamp_gate, "metadata": {}}
    payload.update(over)
    return _blob(payload)


@contextmanager
def _lowrank_gated_reference(model, layers, As, Bs, mag, clamp_gate):
    """The low-rank gated object's apply, VERBATIM (the earlier development
    branch's low-rank module, lines 765-796): z = sigma(h_fp32 A') per token (ReLU iff
    clamp_gate), add = mag * (z B') in fp32, cast once; EVERY position,
    prefill AND decode."""
    from sisteer.models import get_decoder_layers

    decoder = get_decoder_layers(model)

    def make(layer):
        def hook(_mod, _args, output):
            if mag == 0.0:
                return output
            hidden = output[0] if isinstance(output, tuple) else output
            z = hidden.float() @ As[layer].to(hidden.device).t()
            if clamp_gate:
                z = z.clamp_min(0.0)
            Bl = Bs[layer].to(hidden.device)
            add = mag * (z @ Bl.t())
            steered = hidden + add.to(hidden.dtype)
            if isinstance(output, tuple):
                return (steered,) + output[1:]
            return steered
        return hook

    handles = [decoder[l].register_forward_hook(make(l)) for l in layers]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


RANK = 3


def _lowrank_gated_fixture(seed=401):
    g = torch.Generator().manual_seed(seed)
    layers = [1, 3]
    As = {l: torch.randn(RANK, HIDDEN, generator=g) * 0.1 for l in layers}
    Bs = {l: torch.randn(HIDDEN, RANK, generator=g) * 0.1 for l in layers}
    return layers, As, Bs


@pytest.mark.parametrize("clamp_gate", [True, False])
def test_lowrank_gated_roundtrip_and_equivalence(tiny_model, base_acts,
                                               clamp_gate):
    """Rank-k arithmetic, both gate variants, bit-exact against the
    class's verbatim hook — at every position, prefill AND decode (the
    ladder convention: generated tokens are steered too)."""
    layers, As, Bs = _lowrank_gated_fixture()
    # the first op layer sees latents of both signs (non-vacuity of the
    # relu/linear split, and of the relu arm itself)
    z0 = base_acts[layers[0]][0].float() @ As[layers[0]].t()
    assert bool((z0 > 0).any()) and bool((z0 < 0).any())

    art = _lowrank_gated_blob(layers, RANK, As, Bs, 1.4, clamp_gate)
    ops = la.get_adapter("lowrank_gated")(
        {"kind": "lowrank_gated", "rank": RANK}, art, b"")
    want_op = (la.OP_LOWRANK_RELU_GATED if clamp_gate
               else la.OP_LOWRANK_LINEAR_GATED)
    assert [op.layer for op in ops] == layers
    for op, l in zip(ops, layers):
        assert op.op == want_op
        assert op.coeff == 1.4 and op.phase == la.PHASE_BOTH
        assert op.positions == la.POSITIONS_ALL and not op.skip_bos
        assert torch.equal(op.lowrank_a, As[l])
        assert torch.equal(op.lowrank_b, Bs[l])

    pre, post = _run_pre_post(
        tiny_model,
        _lowrank_gated_reference(tiny_model, layers, As, Bs, 1.4, clamp_gate))
    assert any(not torch.equal(pre[l][0], post[l][0]) for l in layers)
    assert any(not torch.equal(pre[l][1], post[l][1]) for l in layers)
    _assert_equivalent(pre, post, ops, set(layers))


def test_lowrank_gated_gate_variants_genuinely_differ(base_acts):
    """With negative latents present, the relu and linear translations
    are different interventions on the same stream."""
    layers, As, Bs = _lowrank_gated_fixture()
    l0 = layers[0]
    hidden = base_acts[l0][0]
    z = hidden.float() @ As[l0].t()
    assert bool((z < 0).any())
    art_relu = _lowrank_gated_blob(layers, RANK, As, Bs, 1.4, True)
    art_lin = _lowrank_gated_blob(layers, RANK, As, Bs, 1.4, False)
    ops_relu = la.get_adapter("lowrank_gated")({"kind": "lowrank_gated"},
                                             art_relu, b"")
    ops_lin = la.get_adapter("lowrank_gated")({"kind": "lowrank_gated"},
                                            art_lin, b"")
    out_relu = la.apply_hook_ops(hidden, ops_relu, 1.0, layer=l0)
    out_lin = la.apply_hook_ops(hidden, ops_lin, 1.0, layer=l0)
    assert not torch.equal(out_relu, out_lin)
    # each matches the class's hand-computed arithmetic
    for out, zz in ((out_relu, z.clamp_min(0.0)), (out_lin, z)):
        want = hidden + (1.4 * (zz @ Bs[l0].t())).to(hidden.dtype)
        assert torch.equal(out, want)


def test_lowrank_gated_shape_confusion_refuses():
    """lowrank_a is [k, d] and lowrank_b [d, k]; a transposed store would
    silently steer in the wrong subspace, so both slips refuse."""
    layers, As, Bs = _lowrank_gated_fixture()
    adapt = la.get_adapter("lowrank_gated")
    # B stored as [k, d] (the read basis's shape)
    bad_b = {l: Bs[l].t().contiguous() for l in layers}
    with pytest.raises(la.AdapterError, match="expected"):
        adapt({"kind": "lowrank_gated"},
              _lowrank_gated_blob(layers, RANK, As, bad_b, 1.0, True), b"")
    # A stored as [d, k] (the write basis's shape)
    bad_a = {l: As[l].t().contiguous() for l in layers}
    with pytest.raises(la.AdapterError, match=r"must be \[rank"):
        adapt({"kind": "lowrank_gated"},
              _lowrank_gated_blob(layers, RANK, bad_a, Bs, 1.0, True), b"")


def test_lowrank_gated_refusals():
    layers, As, Bs = _lowrank_gated_fixture()
    adapt = la.get_adapter("lowrank_gated")
    good = _lowrank_gated_blob(layers, RANK, As, Bs, 1.0, True)
    with pytest.raises(la.AdapterError, match="sidecar"):
        adapt({"kind": "lowrank_gated"}, good, b"stray")
    with pytest.raises(la.AdapterError, match="lacks 'Bs'"):
        adapt({"kind": "lowrank_gated"},
              _blob({"kind": "lowrank_gated", "layers": layers, "rank": RANK,
                     "As": As, "base_scale": 1.0}), b"")
    with pytest.raises(la.AdapterError, match="rank must be > 0"):
        adapt({"kind": "lowrank_gated"},
              _lowrank_gated_blob(layers, 0, As, Bs, 1.0, True), b"")
    with pytest.raises(la.AdapterError, match="cross-check"):
        adapt({"kind": "lowrank_gated", "rank": RANK + 1}, good, b"")
    with pytest.raises(la.AdapterError, match="bool"):
        adapt({"kind": "lowrank_gated"},
              _lowrank_gated_blob(layers, RANK, As, Bs, 1.0, 1), b"")
    bad_a = {l: As[l].clone() for l in layers}
    bad_a[1][0, 0] = float("nan")
    with pytest.raises(la.AdapterError, match="non-finite"):
        adapt({"kind": "lowrank_gated"},
              _lowrank_gated_blob(layers, RANK, bad_a, Bs, 1.0, True), b"")


# ==========================================================================
# loreft_both — SCHEMA-PINNED (the both-phase LoReFT object; the earlier
# development branch's low-rank module, lines ~824-935)
# ==========================================================================


def _loreft_both_blob(layers, rank, Rs, Ws, bs, base_scale, **over):
    payload = {"kind": "loreft_both", "layers": list(layers),
               "rank": rank, "Rs": dict(Rs), "Ws": dict(Ws),
               "bs": dict(bs), "base_scale": base_scale, "metadata": {}}
    payload.update(over)
    return _blob(payload)


@contextmanager
def _loreft_both_reference(model, layers, Rs, Ws, bs, mag):
    """The both-phase LoReFT object's apply, VERBATIM (the earlier development
    branch's low-rank module, lines 887-901): delta = W - R (pre-subtracted MATRIX),
    phi = h_fp32 delta' + b, add = mag * (phi R), cast once; EVERY
    position, prefill AND decode (no seq_len gate)."""
    from sisteer.models import get_decoder_layers

    decoder = get_decoder_layers(model)

    def make(layer):
        def hook(_mod, _args, output):
            if mag == 0.0:
                return output
            hidden = output[0] if isinstance(output, tuple) else output
            dev = hidden.device
            delta = (Ws[layer].to(dev) - Rs[layer].to(dev))
            phi = hidden.float() @ delta.t() + bs[layer].to(dev)
            add = mag * (phi @ Rs[layer].to(dev))
            steered = hidden + add.to(hidden.dtype)
            if isinstance(output, tuple):
                return (steered,) + output[1:]
            return steered
        return hook

    handles = [decoder[l].register_forward_hook(make(l)) for l in layers]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def _loreft_both_fixture(seed=503):
    g = torch.Generator().manual_seed(seed)
    layers = [1, 3]
    Rs, Ws, bs = {}, {}, {}
    for l in layers:
        Q, _ = torch.linalg.qr(torch.randn(HIDDEN, RANK, generator=g))
        Rs[l] = Q.t().contiguous()
        Ws[l] = Rs[l] + torch.randn(RANK, HIDDEN, generator=g) * 0.2
        bs[l] = torch.randn(RANK, generator=g) * 0.1
    return layers, Rs, Ws, bs


def test_loreft_both_roundtrip_ops():
    layers, Rs, Ws, bs = _loreft_both_fixture()
    art = _loreft_both_blob(layers, RANK, Rs, Ws, bs, 0.8)
    ops = la.get_adapter("loreft_both")(
        {"kind": "loreft_both", "rank": RANK}, art, b"")
    assert [op.layer for op in ops] == layers
    for op, l in zip(ops, layers):
        assert op.op == la.OP_AFFINE_SUBSPACE
        assert op.phase == la.PHASE_BOTH       # unlike loreft: BOTH phases
        assert op.coeff == 0.8
        assert torch.equal(op.rotate, Rs[l])
        assert torch.equal(op.affine_w, Ws[l])
        assert torch.equal(op.affine_b, bs[l])


@pytest.mark.xfail(
    strict=True,
    reason="BUG (bit-exactness): OP_AFFINE_SUBSPACE's body is reft.py's "
    "LoReFTObject VERBATIM — delta = (lin(h32) - rot(h32)) @ R, i.e. TWO "
    "matmuls subtracted — but the both-phase LoReFT object pre-subtracts the "
    "MATRIX: phi = h @ (W - R).T + b (the earlier development branch's "
    "low-rank module, lines 893-895). Same math, different float rounding: "
    "~2.4e-7 max abs diff in fp32 on this fixture, so the adapter does "
    "NOT bit-exactly reproduce the object whose hash the cell serves "
    "under. (test_loreft_both_equivalence_within_float_tolerance pins "
    "that the translation is otherwise correct.)")
def test_loreft_both_adapter_reproduces_object_bit_exactly(tiny_model):
    layers, Rs, Ws, bs = _loreft_both_fixture()
    art = _loreft_both_blob(layers, RANK, Rs, Ws, bs, 0.8)
    ops = la.get_adapter("loreft_both")({"kind": "loreft_both"}, art,
                                          b"")
    pre, post = _run_pre_post(
        tiny_model,
        _loreft_both_reference(tiny_model, layers, Rs, Ws, bs, 0.8))
    assert any(not torch.equal(pre[l][0], post[l][0]) for l in layers)
    _assert_equivalent(pre, post, ops, set(layers))


def test_loreft_both_equivalence_within_float_tolerance(tiny_model):
    """The translation is mathematically the object's edit (same layers,
    same affine-subspace map, both phases): apply_hook_ops matches the
    class's verbatim hook to float tolerance at every op layer, prefill
    AND decode — only the summation ORDER differs (see the strict-xfail
    bit-exactness test above)."""
    layers, Rs, Ws, bs = _loreft_both_fixture()
    art = _loreft_both_blob(layers, RANK, Rs, Ws, bs, 0.8)
    ops = la.get_adapter("loreft_both")({"kind": "loreft_both"}, art,
                                          b"")
    pre, post = _run_pre_post(
        tiny_model,
        _loreft_both_reference(tiny_model, layers, Rs, Ws, bs, 0.8))
    # non-vacuous in BOTH phases (the class has no seq_len gate)
    for l in layers:
        assert not torch.equal(pre[l][0], post[l][0])
        assert not torch.equal(pre[l][1], post[l][1])
    for l in layers:
        for phase_idx, ctx in ((0, PREFILL_CTX), (1, DECODE_CTX)):
            mine = la.apply_hook_ops(pre[l][phase_idx], ops, 1.0, layer=l,
                                     ctx=ctx)
            torch.testing.assert_close(mine, post[l][phase_idx])


def test_loreft_both_steers_decode_unlike_the_loreft_kind(tiny_model,
                                                            tmp_path):
    """The two kinds share OP_AFFINE_SUBSPACE and differ ONLY in phase:
    loreft (reft.py) gates on prefill-vs-decode by seq_len, so its ops are
    prefill-only and a decode forward comes back as the SAME tensor;
    loreft_both's benign penalty covers generation, so its ops fire at
    decode too."""
    from sisteer.extraction.reft import LoReFTObject

    # loreft_both: BOTH phases — a decode forward IS edited
    layers, Rs, Ws, bs = _loreft_both_fixture()
    js_ops = la.get_adapter("loreft_both")(
        {"kind": "loreft_both"},
        _loreft_both_blob(layers, RANK, Rs, Ws, bs, 0.8), b"")
    g = torch.Generator().manual_seed(509)
    dec = torch.randn(1, 1, HIDDEN, generator=g)
    out = la.apply_hook_ops(dec, js_ops, 1.0, layer=layers[0],
                            ctx=DECODE_CTX)
    assert not torch.equal(out, dec)

    # loreft (the real class, importable here): prefill-only — NOT INVOKED
    torch.manual_seed(31)
    obj = LoReFTObject(tiny_model, [1], rank=2)
    with torch.no_grad():   # at init W=R, b=0 is a no-op; perturb
        obj.learned[1].weight.add_(
            torch.randn(2, HIDDEN, generator=g) * 0.2)
    path = tmp_path / "loreft.pt"
    obj.save(path)
    lo_ops = la.get_adapter("loreft")(
        {"kind": "loreft", "layers": [1], "rank": 2}, path.read_bytes(),
        b"")
    assert all(op.phase == la.PHASE_PREFILL for op in lo_ops)
    assert all(op.phase == la.PHASE_BOTH for op in js_ops)
    out_lo = la.apply_hook_ops(dec, lo_ops, 1.0, layer=1, ctx=DECODE_CTX)
    assert out_lo is dec


def test_loreft_both_refusals():
    layers, Rs, Ws, bs = _loreft_both_fixture()
    adapt = la.get_adapter("loreft_both")
    good = _loreft_both_blob(layers, RANK, Rs, Ws, bs, 1.0)
    with pytest.raises(la.AdapterError, match="sidecar"):
        adapt({"kind": "loreft_both"}, good, b"stray")
    with pytest.raises(la.AdapterError, match="lacks 'bs'"):
        adapt({"kind": "loreft_both"},
              _blob({"kind": "loreft_both", "layers": layers,
                     "rank": RANK, "Rs": Rs, "Ws": Ws,
                     "base_scale": 1.0}), b"")
    with pytest.raises(la.AdapterError, match="rank must be > 0"):
        adapt({"kind": "loreft_both"},
              _loreft_both_blob(layers, -1, Rs, Ws, bs, 1.0), b"")
    # W whose leading dim is not the stored rank
    bad_w = {l: torch.cat([Ws[l], Ws[l][:1]]) for l in layers}
    with pytest.raises(la.AdapterError, match="shape"):
        adapt({"kind": "loreft_both"},
              _loreft_both_blob(layers, RANK, Rs, bad_w, bs, 1.0), b"")
    # b of the wrong width
    bad_b = {l: torch.zeros(RANK + 1) for l in layers}
    with pytest.raises(la.AdapterError, match="dim"):
        adapt({"kind": "loreft_both"},
              _loreft_both_blob(layers, RANK, Rs, Ws, bad_b, 1.0), b"")
    with pytest.raises(la.AdapterError, match="cross-check"):
        adapt({"kind": "loreft_both", "rank": RANK + 2}, good, b"")


# ==========================================================================
# vector — ROUND-TRIP against methods.VectorObject over a SteeringVector
# (safetensors + JSON sidecar), deployed through hooks.steering
# ==========================================================================


def _vector_artifacts(tmp_path, method="diffmean", metadata=None,
                      vectors=None, seed=601):
    from sisteer.vectors import SteeringVector

    g = torch.Generator().manual_seed(seed)
    if vectors is None:
        vectors = {1: torch.randn(HIDDEN, generator=g) * 0.05,
                   3: torch.randn(HIDDEN, generator=g) * 0.05}
    sv = SteeringVector(vectors=vectors, method=method,
                        model_name="TESTMODEL", instruction_id="synthetic",
                        metadata=metadata or {})
    path = tmp_path / "object.safetensors"
    sv.save(path)
    return sv, path.read_bytes(), path.with_suffix(".json").read_bytes()


def test_vector_roundtrip_and_equivalence(tiny_model, tmp_path):
    """The plain additive arm: one add_vector per container layer, coeff
    1.0 (the magnitude lives in the vector's norm — nothing baked), the
    load_object defaults positions='all'/skip_bos=True, both phases.
    Bit-exact against VectorObject.apply through hooks.steering, with the
    BOS carve-out doing real work."""
    from sisteer.methods import VectorObject

    sv, art, side = _vector_artifacts(tmp_path)
    obj = VectorObject(sv)               # defaults: all / skip_bos=True
    ops = la.get_adapter("vector")({"kind": "vector"}, art, side)
    assert [op.layer for op in ops] == [1, 3]
    for op, l in zip(ops, (1, 3)):
        assert op.op == la.OP_ADD_VECTOR
        assert op.coeff == 1.0           # NO baked scale on this kind
        assert op.positions == la.POSITIONS_ALL
        assert op.skip_bos               # the stored default
        assert op.phase == la.PHASE_BOTH
        assert torch.equal(op.vector, sv.vectors[l])

    pre, post = _run_pre_post(tiny_model, obj.apply(tiny_model, scale=1.0))
    assert any(not torch.equal(pre[l][0], post[l][0]) for l in (1, 3))
    _assert_equivalent(pre, post, ops, {1, 3})
    # BOS untouched, its neighbor steered; the decode position steered
    assert torch.equal(pre[1][0][:, 0, :], post[1][0][:, 0, :])
    assert not torch.equal(pre[1][0][:, 1, :], post[1][0][:, 1, :])
    assert not torch.equal(pre[1][1], post[1][1])


def test_vector_skip_bos_defaults_true_and_reads_the_spec(tmp_path):
    _, art, side = _vector_artifacts(tmp_path)
    adapt = la.get_adapter("vector")
    assert all(op.skip_bos for op in adapt({}, art, side))
    assert not any(op.skip_bos
                   for op in adapt({"skip_bos": False}, art, side))
    with pytest.raises(la.AdapterError, match="bool"):
        adapt({"skip_bos": "yes"}, art, side)


def test_vector_add_is_in_the_model_dtype_cast_before_multiplying(tmp_path):
    """hooks.SteeringHandle._make_hook casts the vector to the hidden
    dtype BEFORE multiplying by scale (hooks.py:259+286: v = vec.to(...,
    hidden.dtype); addition = v * scale) — which the op's 'model'
    materialization policy reproduces bit-exactly under bf16. The
    alternative order (scale in fp32, cast the product) is a genuinely
    different arithmetic on this fixture, so the test would catch a
    policy flip."""
    g = torch.Generator().manual_seed(607)
    v = torch.randn(HIDDEN, generator=g) * 0.03
    _, art, side = _vector_artifacts(tmp_path, vectors={1: v})
    ops = la.get_adapter("vector")({"skip_bos": False}, art, side)
    mat = la.materialize_ops(ops, device="cpu", dtype=torch.bfloat16)
    assert mat[0].vector.dtype == torch.bfloat16

    hidden = torch.randn(1, PROMPT_LEN, HIDDEN, generator=g).to(
        torch.bfloat16)
    scale = 0.3
    got = la.apply_hook_ops(hidden, mat, scale, layer=1, ctx=PREFILL_CTX)
    # hooks.py's verbatim order: cast v, THEN multiply, then add
    want = hidden + v.to(torch.bfloat16) * scale
    assert torch.equal(got, want)
    # the other order really is different arithmetic here (guards the
    # fixture's strength, and documents why the policy is load-bearing)
    other = hidden + (v * scale).to(torch.bfloat16)
    assert not torch.equal(want, other)


def test_vector_refuses_the_antidote_method(tmp_path):
    _, art, side = _vector_artifacts(tmp_path, method="antidote",
                                     metadata={"base_alpha": 3.5})
    with pytest.raises(la.AdapterError, match="antidote"):
        la.get_adapter("vector")({"kind": "vector"}, art, side)


def test_vector_refuses_projection_targets(tmp_path):
    _, art, side = _vector_artifacts(
        tmp_path, metadata={"projection_targets": {"1": 3.0}})
    with pytest.raises(la.AdapterError, match="projection"):
        la.get_adapter("vector")({"kind": "vector"}, art, side)


def test_vector_refuses_norm_relative(tmp_path):
    _, art, side = _vector_artifacts(tmp_path)
    with pytest.raises(la.AdapterError, match="norm_relative"):
        la.get_adapter("vector")({"norm_relative": True}, art, side)


def test_vector_response_mode_with_header_ids(tmp_path):
    """The cross-phase response mask: ops carry the assistant header's
    exact token ids; at prefill the mask is LOCATED by matching them
    against the request's own tail (never width arithmetic), and every
    decode position is inside the region."""
    g = torch.Generator().manual_seed(613)
    v = torch.randn(HIDDEN, generator=g) * 0.05
    _, art, side = _vector_artifacts(tmp_path, vectors={1: v})
    header = (11, 12, 13)
    ops = la.get_adapter("vector")({"positions": "response"}, art, side,
                                   header_ids=header)
    (op,) = ops
    assert op.positions == la.POSITIONS_RESPONSE
    assert op.header_ids == header

    hidden = torch.randn(1, PROMPT_LEN, HIDDEN, generator=g)
    token_ids = (40, 41, 42, 43) + header          # header at the tail
    ctx = la.HookContext(la.PHASE_PREFILL, tuple(range(PROMPT_LEN)),
                         token_ids=token_ids)
    out = la.apply_hook_ops(hidden, ops, 1.0, layer=1, ctx=ctx)
    start = PROMPT_LEN - len(header)
    assert torch.equal(out[:, :start, :], hidden[:, :start, :])
    for t in range(start, PROMPT_LEN):
        assert not torch.equal(out[:, t, :], hidden[:, t, :])
    # decode: inside the region
    dec = torch.randn(1, 1, HIDDEN, generator=g)
    dctx = la.HookContext(la.PHASE_DECODE, (PROMPT_LEN,),
                          token_ids=(50,))
    assert not torch.equal(
        la.apply_hook_ops(dec, ops, 1.0, layer=1, ctx=dctx), dec)
    # a prompt NOT ending at the header refuses (assistant prefill /
    # template drift), and a context without token ids refuses too
    bad = la.HookContext(la.PHASE_PREFILL, tuple(range(PROMPT_LEN)),
                         token_ids=(40, 41, 42, 43, 11, 12, 99))
    with pytest.raises(la.AdapterError, match="assistant header"):
        la.apply_hook_ops(hidden, ops, 1.0, layer=1, ctx=bad)
    no_ids = la.HookContext(la.PHASE_PREFILL, tuple(range(PROMPT_LEN)))
    with pytest.raises(la.AdapterError, match="no token ids"):
        la.apply_hook_ops(hidden, ops, 1.0, layer=1, ctx=no_ids)
    # and a registry that supplied no header cannot serve the cell
    with pytest.raises(la.AdapterError, match="no assistant header"):
        la.get_adapter("vector")({"positions": "response"}, art, side)


def test_vector_container_and_sidecar_refusals(tmp_path):
    from safetensors.torch import save as st_save

    _, art, side = _vector_artifacts(tmp_path)
    adapt = la.get_adapter("vector")
    # the sidecar is REQUIRED (part of the hashed identity)
    with pytest.raises(la.AdapterError, match="sidecar"):
        adapt({"kind": "vector"}, art, b"")
    # sidecar layers disagreeing with the container refuse
    bad_side = json.loads(side.decode("utf-8"))
    bad_side["layers"] = [1]
    with pytest.raises(la.AdapterError, match="inconsistent"):
        adapt({"kind": "vector"}, art,
              json.dumps(bad_side).encode("utf-8"))
    # a non-layer_<k> entry is not a SteeringVector container
    stray = st_save({"foo": torch.ones(4)})
    with pytest.raises(la.AdapterError, match="unexpected safetensors"):
        adapt({"kind": "vector"}, stray, side)
    # an ActAdd-style [P, d] per-position carrier is a different
    # intervention (front injection) — the 1-D check refuses it
    g = torch.Generator().manual_seed(617)
    _, art2d, side2d = _vector_artifacts(
        tmp_path / "2d", vectors={1: torch.randn(3, HIDDEN, generator=g)})
    with pytest.raises(la.AdapterError, match="1-D"):
        adapt({"kind": "vector"}, art2d, side2d)
    # hooks.py position modes with no lens translation refuse loudly
    for mode in ("prompt", "front", "last"):
        with pytest.raises(la.AdapterError, match="not lens-served"):
            adapt({"positions": mode}, art, side)
