"""LoRA on the vLLM lens path: the o_proj module site (2026-08-28).

The lora kinds ('lora' — RankOneLoRAObject, one layer rank 1; 'lora_all'
— MultiLayerLoRAObject, rank-general, every layer: the scale-12 lora_r4
objects) edit the attention o_proj MID-BLOCK — the delta is computed
from the o_proj INPUT and lands on its OUTPUT before the same block's
MLP — so they were UNSERVABLE while the block output was the lens
path's only hook site, and vLLM's native multi-LoRA cannot serve them
either once the fit is position-masked (an adapter applies at EVERY
position, so a positions='response' fit — token 1 steered from its own
prefill row — is inexpressible there). This file pins the landed o_proj
site end to end on CPU:

- the adapters translate REAL blob schemas (the rank-4 response-fit
  form and the legacy rank-1 form) into OP_LORA_OPROJ ops carrying the
  object's own tensors, and refuse every self-inconsistent artifact the
  authoritative load_blob refuses
- apply_oproj_ops mirrors the authoritative HF hook bodies EXACTLY —
  proven against MultiLayerLoRAObject itself on a real tiny model, not
  against a re-implementation, across prefill AND KV-cached decode
- the response mask is located, not computed: the last prefill row (the
  forward that emits token 1) is steered, rows before the boundary stay
  BIT-identical, and an off-by-one or unmasked boundary fails
- the worker installs the o_proj hooks (plus the positions-stashing
  decoder pre-hook), refuses TP != 1 and an unhookable architecture,
  and serves per-request rows through the same slots as the block site
- the registry shim folds a coefficient into the WRITE basis alone and
  the whole build round-trips through WorkerObjectRegistry.load.
"""

from __future__ import annotations

import io

import pytest
import torch

from sisteer.capeval.serving import lens_adapters as la
from sisteer.capeval.serving import lens_steer as ls
from sisteer.capeval.serving import lens_worker as lw
from test_petri_lens_worker import FakeLayer, FakeView, _forward, _ok_header

D_IN = 10   # deliberately non-square o_proj shapes: gemma-4-E4B is
D_OUT = 6   # 2048 -> 2560, and square shapes hide a transposed basis
RANK = 3
HEADER = (7, 8, 9)   # a 3-token assistant header, like gemma's


def _blob_bytes(payload) -> bytes:
    buf = io.BytesIO()
    torch.save(payload, buf)
    return buf.getvalue()


def _lora_all_blob(layers=(0, 2), rank=RANK, seed=3, positions="response",
                   header_len=len(HEADER), **extra):
    g = torch.Generator().manual_seed(seed)
    blob = {
        "kind": "lora_all",
        "A": {str(l): torch.randn(D_IN, rank, generator=g) for l in layers},
        "B": {str(l): torch.randn(D_OUT, rank, generator=g) * 0.5
              for l in layers},
        "rank": rank,
        "layers": list(layers),
        "positions": positions,
    }
    if positions == "response":
        blob["response_header_len"] = header_len
    blob.update(extra)
    return blob


def _adapt_all(blob, spec=None, header=HEADER):
    return la.get_adapter("lora_all")(
        dict(spec or {"kind": "lora_all"}), _blob_bytes(blob), b"",
        header_ids=tuple(header) if header else None)


def _prefill_ctx(prompt_ids):
    return la.HookContext(la.PHASE_PREFILL,
                          tuple(range(len(prompt_ids))),
                          token_ids=tuple(prompt_ids))


def _decode_ctx(pos, token_id=5):
    return la.HookContext(la.PHASE_DECODE, (pos,), token_ids=(token_id,))


# ---------------------------------------------------------------------------
# adapter translation: the real schemas in, the object's own tensors out
# ---------------------------------------------------------------------------


class TestAdapters:
    def test_rank4_response_blob_translates(self):
        blob = _lora_all_blob()
        ops = _adapt_all(blob)
        assert [op.layer for op in ops] == [0, 2]
        for op in ops:
            assert op.op == la.OP_LORA_OPROJ
            assert la.op_site(op) == la.HOOK_SITE_OPROJ
            assert op.positions == la.POSITIONS_RESPONSE
            assert op.header_ids == HEADER
            assert op.phase == la.PHASE_BOTH and not op.skip_bos
            assert op.coeff == 1.0
            A = blob["A"][str(op.layer)]
            B = blob["B"][str(op.layer)]
            # read basis is Aᵀ ([k, d_in]), write basis is B ([d_out, k])
            # — the exact tensors, no rescale, no swap (the non-square
            # shapes make a silent transpose a shape error here)
            assert torch.equal(op.lowrank_a, A.t())
            assert torch.equal(op.lowrank_b, B)
            assert op.lowrank_a.dtype == torch.float32

    def test_rank1_legacy_pair_canonicalizes_to_k1(self):
        g = torch.Generator().manual_seed(7)
        blob = _lora_all_blob(rank=1)
        blob["A"] = {"0": torch.randn(D_IN, generator=g),
                     "2": torch.randn(D_IN, generator=g)}
        blob["B"] = {"0": torch.randn(D_OUT, generator=g),
                     "2": torch.randn(D_OUT, generator=g)}
        del blob["rank"]
        ops = _adapt_all(blob)
        for op in ops:
            assert op.lowrank_a.shape == (1, D_IN)
            assert op.lowrank_b.shape == (D_OUT, 1)
            assert torch.equal(op.lowrank_a[0], blob["A"][str(op.layer)])
            assert torch.equal(op.lowrank_b[:, 0], blob["B"][str(op.layer)])

    def test_single_layer_lora_kind_translates(self):
        g = torch.Generator().manual_seed(9)
        A, B = torch.randn(D_IN, generator=g), torch.randn(D_OUT, generator=g)
        ops = la.get_adapter("lora")(
            {"kind": "lora"},
            _blob_bytes({"kind": "lora", "A": A, "B": B, "layer": 5}), b"")
        (op,) = ops
        assert op.layer == 5 and op.op == la.OP_LORA_OPROJ
        # no positions key -> the historical all-token fit
        assert op.positions == la.POSITIONS_ALL and op.header_ids is None
        assert torch.equal(op.lowrank_a[0], A)
        assert torch.equal(op.lowrank_b[:, 0], B)

    def test_spec_positions_override_remasks_an_all_fit(self):
        """The cross-phase cell: an all-position artifact served under
        the response mask (the o_proj edit is derived per position from
        that position's own o_proj input, so DERIVATION LOCALITY holds
        and the override is sound)."""
        blob = _lora_all_blob(positions="all", header_len=None)
        blob.pop("response_header_len", None)
        ops = _adapt_all(blob, spec={"kind": "lora_all",
                                     "positions": "response"})
        assert all(op.positions == la.POSITIONS_RESPONSE for op in ops)
        assert all(op.header_ids == HEADER for op in ops)

    @pytest.mark.parametrize("mutate, match", [
        (lambda b: b["B"].__setitem__("0", torch.randn(D_OUT, 2)), "rank"),
        (lambda b: b.__setitem__("rank", 8), "rank"),
        (lambda b: b.__setitem__("layers", [0]), "layers"),
        (lambda b: b["B"].__setitem__("0", torch.randn(D_OUT)), "pair|1-D"),
        (lambda b: b.__setitem__("kind", "lora"), "wrong adapter|kind"),
        (lambda b: b["A"].__setitem__(
            "0", torch.full((D_IN, RANK), float("nan"))), "non-finite"),
        (lambda b: b.__setitem__("response_header_len", 0), "H <= 0|positive"),
    ])
    def test_self_inconsistent_artifacts_refuse(self, mutate, match):
        blob = _lora_all_blob()
        mutate(blob)
        with pytest.raises(la.AdapterError, match=match):
            _adapt_all(blob)

    def test_fit_header_width_must_match_served_header(self):
        """Like with like: header_ids are the id-form
        (chat_markers.generation_header_ids) of the SAME width deriver
        the fit used (hooks.response_header_len), so on the template the
        object was fit against, the two are EQUAL (gemma-4: 3;
        Llama-3.1: 4) and any inequality is template drift — including a
        registry wrongly exported with the terminator-anchored LOCATED
        header (gemma-4: 4 ids, absorbing the prompt-side newline after
        ``<turn|>``), which would steer one token outside the fit
        support while looking like a plausible header."""
        blob = _lora_all_blob(header_len=4)
        with pytest.raises(la.AdapterError, match="4-token"):
            _adapt_all(blob, header=HEADER)  # 3-token served header
        blob = _lora_all_blob(header_len=2)
        with pytest.raises(la.AdapterError, match="2-token"):
            _adapt_all(blob, header=HEADER)

    def test_response_without_header_ids_refuses(self):
        with pytest.raises(la.AdapterError, match="assistant header"):
            _adapt_all(_lora_all_blob(), header=None)

    def test_sidecar_refuses(self):
        with pytest.raises(la.AdapterError, match="sidecar"):
            la.get_adapter("lora_all")(
                {"kind": "lora_all"}, _blob_bytes(_lora_all_blob()),
                b'{"x": 1}', header_ids=HEADER)


# ---------------------------------------------------------------------------
# the op: site routing, the mask, token 1
# ---------------------------------------------------------------------------


def _one_op(**over):
    g = torch.Generator().manual_seed(11)
    kw = dict(op=la.OP_LORA_OPROJ, layer=0,
              lowrank_a=torch.randn(RANK, D_IN, generator=g),
              lowrank_b=torch.randn(D_OUT, RANK, generator=g),
              positions=la.POSITIONS_RESPONSE, header_ids=HEADER)
    kw.update(over)
    return la.HookOp(**kw)


class TestApplyOprojOps:
    def test_block_apply_refuses_an_oproj_op(self):
        with pytest.raises(la.AdapterError, match="oproj-site"):
            la.apply_hook_ops(torch.zeros(2, D_OUT), (_one_op(),), 1.0,
                              layer=0)

    def test_oproj_apply_refuses_a_block_op(self):
        block = la.HookOp(op=la.OP_ADD_VECTOR, layer=0,
                          vector=torch.ones(D_OUT))
        with pytest.raises(la.AdapterError, match="block-output"):
            la.apply_oproj_ops(torch.zeros(2, D_IN), torch.zeros(2, D_OUT),
                               (block,), 1.0, layer=0)

    def test_scale0_and_other_layer_are_the_same_tensor(self):
        x, out = torch.randn(3, D_IN), torch.randn(3, D_OUT)
        assert la.apply_oproj_ops(x, out, (_one_op(),), 0.0, layer=0) is out
        assert la.apply_oproj_ops(
            x, out, (_one_op(layer=3),), 1.0, layer=0) is out

    def test_response_mask_steers_the_token1_row_and_only_from_there(self):
        """THE token-1 assertion, built to fail on an off-by-one: the
        prompt is [t..., h1 h2 h3]; the served region starts AT the
        header, so rows 0..P-H-1 are BIT-identical and rows P-H..P-1 —
        including row P-1, the forward that samples token 1 — move.
        A mask starting one row later would leave row P-H unmoved; one
        starting one row earlier would move row P-H-1."""
        prompt = [3, 4, 5, 6] + list(HEADER)      # P=7, H=3, start=4
        P, H = len(prompt), len(HEADER)
        x, out = torch.randn(P, D_IN), torch.randn(P, D_OUT)
        op = _one_op()
        res = la.apply_oproj_ops(x, out, (op,), 1.0, layer=0,
                                 ctx=_prefill_ctx(prompt))
        assert torch.equal(res[:P - H], out[:P - H])   # bit-identical
        moved = [not torch.equal(res[i], out[i]) for i in range(P - H, P)]
        assert all(moved), f"header rows unmoved: {moved}"
        assert not torch.equal(res[P - 1], out[P - 1])  # token 1's row
        # and the steered rows carry EXACTLY the authoritative delta
        delta = (x.float() @ op.lowrank_a.t()) @ op.lowrank_b.t()
        expect = torch.where(
            torch.tensor([i >= P - H for i in range(P)]).unsqueeze(-1),
            out + delta.to(out.dtype), out)
        assert torch.equal(res, expect)

    def test_decode_rows_are_always_inside_the_region(self):
        x, out = torch.randn(1, D_IN), torch.randn(1, D_OUT)
        res = la.apply_oproj_ops(x, out, (_one_op(),), 1.0, layer=0,
                                 ctx=_decode_ctx(12))
        assert not torch.equal(res, out)

    # NOTE (2026-09-02 merge): the branch's multi-turn o_proj tests
    # (continuing-past-the-header serves; headerless refuses) pinned the
    # superseded unconditional last-occurrence locator — main's declared-only
    # boundary policy won the merge (see lens_adapters._response_start and
    # test_gemma4_petri_multiturn_boundary), so main's refusal pin is
    # restored here.
    def test_prompt_not_ending_at_header_refuses(self):
        prompt = list(HEADER) + [3, 4]   # header NOT at the tail
        with pytest.raises(la.AdapterError, match="does not end"):
            la.apply_oproj_ops(torch.randn(5, D_IN), torch.randn(5, D_OUT),
                               (_one_op(),), 1.0, layer=0,
                               ctx=_prefill_ctx(prompt))

    def test_headerless_prompt_refuses(self):
        """Header ids occurring NOWHERE in the prompt — a foreign chat
        template — refuse through the same tail check as any non-tail
        render, with the declared-only policy's message. Re-pins the
        o_proj-site headerless case whose branch-era test was dropped at
        the 2026-09-02 merge (its match string named the superseded
        locator's refusal); the block-site twin lives in
        test_petri_response_mask.py."""
        prompt = [3, 4, 5, 6, 7]         # header (7, 8, 9) occurs nowhere
        with pytest.raises(la.AdapterError, match="does not end"):
            la.apply_oproj_ops(torch.randn(5, D_IN), torch.randn(5, D_OUT),
                               (_one_op(),), 1.0, layer=0,
                               ctx=_prefill_ctx(prompt))

    def test_response_op_without_ctx_refuses(self):
        with pytest.raises(la.AdapterError, match="HookContext"):
            la.apply_oproj_ops(torch.randn(2, D_IN), torch.randn(2, D_OUT),
                               (_one_op(),), 1.0, layer=0)

    def test_wrong_dims_refuse(self):
        with pytest.raises(la.AdapterError, match="input dim"):
            la.apply_oproj_ops(torch.randn(1, D_IN + 1),
                               torch.randn(1, D_OUT), (_one_op(),), 1.0,
                               layer=0, ctx=_decode_ctx(9, 5))
        with pytest.raises(la.AdapterError, match="output dim"):
            la.apply_oproj_ops(torch.randn(1, D_IN),
                               torch.randn(1, D_OUT + 1), (_one_op(),), 1.0,
                               layer=0, ctx=_decode_ctx(9, 5))

    def test_op_validation_pins_the_mode_envelope(self):
        with pytest.raises(la.AdapterError, match="rank disagreement"):
            _one_op(lowrank_b=torch.randn(D_OUT, RANK + 1))
        with pytest.raises(la.AdapterError, match="'all' or 'response'"):
            _one_op(positions=la.POSITIONS_LAST, header_ids=None)
        with pytest.raises(la.AdapterError, match="skip_bos"):
            _one_op(positions=la.POSITIONS_ALL, header_ids=None,
                    skip_bos=True)
        with pytest.raises(la.AdapterError, match="skip_bos"):
            _one_op(positions=la.POSITIONS_ALL, header_ids=None,
                    phase=la.PHASE_PREFILL)


# ---------------------------------------------------------------------------
# HF equivalence: the ops against MultiLayerLoRAObject ITSELF (tiny model)
# ---------------------------------------------------------------------------


TINY_D = 16  # tiny_llama hidden size


def _tiny_blob(model, layers=(0, 1), rank=4, seed=13, header_len=4):
    g = torch.Generator().manual_seed(seed)
    return {
        "kind": "lora_all",
        "A": {str(l): torch.randn(TINY_D, rank, generator=g)
              for l in layers},
        "B": {str(l): torch.randn(TINY_D, rank, generator=g) * 0.5
              for l in layers},
        "rank": rank,
        "layers": list(layers),
        "positions": "response",
        "response_header_len": header_len,
    }


def test_ops_reproduce_the_authoritative_class_forward(tiny_llama):
    """The load-bearing equivalence: greedy decode under (a) the
    authoritative MultiLayerLoRAObject hooks and (b) this lane's
    OP_LORA_OPROJ ops applied at the same o_proj modules through
    apply_oproj_ops with per-forward HookContexts — same tokens, and
    logits equal to fp32 kernel-order noise, across the prefill AND
    four KV-cached decode steps. Not a re-implementation on either
    side: (a) is the class itself, (b) is the committed serving op."""
    from sisteer.methods import load_object

    model, tok = tiny_llama
    ids = tok("The quick brown fox jumps over", return_tensors="pt")[
        "input_ids"]
    P = ids.shape[1]
    H = 4
    header = tuple(int(t) for t in ids[0, -H:])
    blob = _tiny_blob(model, header_len=H)

    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "object.pt"
        torch.save(blob, path)
        obj = load_object(path, model, tok)

    ops = la.get_adapter("lora_all")(
        {"kind": "lora_all"}, _blob_bytes(blob), b"", header_ids=header)

    def greedy(steer: str, n_new: int = 4):
        """Manual greedy loop so both paths share the decode mechanics."""
        state = {"seen": 0}
        handles = []
        if steer == "ops":
            layer_mods = {op.layer: None for op in ops}
            decoder = model.model.layers
            for l in layer_mods:
                oproj = decoder[l].self_attn.o_proj

                def make(l):
                    def hook(mod, args, output):
                        x = args[0][0]          # [T, d_in] (batch of 1)
                        out = output[0]
                        T = x.shape[0]
                        if T > 1:
                            ctx = la.HookContext(
                                la.PHASE_PREFILL, tuple(range(T)),
                                token_ids=tuple(int(t)
                                                for t in seq[0, :T]))
                        else:
                            pos = state["seen"] - 1
                            ctx = la.HookContext(
                                la.PHASE_DECODE, (pos,),
                                token_ids=(int(seq[0, pos]),))
                        new = la.apply_oproj_ops(x, out, ops, 1.0,
                                                 layer=l, ctx=ctx)
                        return new.unsqueeze(0)
                    return hook
                handles.append(oproj.register_forward_hook(make(l)))
        seq = ids.clone()
        outs = []
        try:
            import contextlib
            cm = obj.apply(model, scale=1.0) if steer == "class" \
                else contextlib.nullcontext()
            with torch.no_grad(), cm:
                past = None
                feed = seq
                for _ in range(n_new + 1):
                    state["seen"] = seq.shape[1]
                    r = model(input_ids=feed, past_key_values=past,
                              use_cache=True)
                    past = r.past_key_values
                    logit = r.logits[0, -1]
                    outs.append(logit.clone())
                    nxt = int(logit.argmax())
                    feed = torch.tensor([[nxt]])
                    seq = torch.cat([seq, feed], dim=1)
        finally:
            for h in handles:
                h.remove()
        return seq, outs

    seq_a, logits_a = greedy("class")
    seq_b, logits_b = greedy("ops")
    seq_base, logits_base = greedy("base")

    assert torch.equal(seq_a, seq_b), (
        "greedy tokens diverge between the authoritative class and the "
        "served ops")
    for i, (x, y) in enumerate(zip(logits_a, logits_b)):
        assert torch.allclose(x, y, rtol=1e-5, atol=1e-6), (
            f"step {i}: logits diverge beyond fp32 kernel-order noise "
            f"(max abs {float((x - y).abs().max())})")
    # and the steering BITES: the very first steered forward (which
    # emits token 1) differs from base — an unmasked or off-by-one
    # boundary that misses the final prefill row fails HERE
    assert not torch.equal(logits_a[0], logits_base[0]), (
        "token 1's logits equal base — the forward emitting token 1 "
        "was not steered")
    assert not torch.equal(logits_b[0], logits_base[0])


# ---------------------------------------------------------------------------
# the worker: install, TP guard, the o_proj hook body
# ---------------------------------------------------------------------------


class OprojModule:
    def __init__(self):
        self.hooks = []

    def register_forward_hook(self, fn):
        self.hooks.append(fn)
        return object()


class OprojLayer(FakeLayer):
    """A decoder layer with a hookable self_attn.o_proj and a pre-hook
    registration surface (the positions stash)."""

    class _Attn:
        def __init__(self):
            self.o_proj = OprojModule()

    def __init__(self):
        super().__init__()
        self.pre_hooks = []
        self.pre_hook_with_kwargs = []
        self.self_attn = self._Attn()

    def register_forward_pre_hook(self, fn, *, with_kwargs=False):
        self.pre_hooks.append(fn)
        self.pre_hook_with_kwargs.append(with_kwargs)
        return object()


def _lora_obj(cell_id="arm-lora", layers=(1,), scale=1.0):
    g = torch.Generator().manual_seed(21)
    ops = tuple(la.HookOp(
        op=la.OP_LORA_OPROJ, layer=l,
        lowrank_a=torch.randn(RANK, D_IN, generator=g),
        lowrank_b=torch.randn(D_OUT, RANK, generator=g),
        positions=la.POSITIONS_RESPONSE, header_ids=HEADER) for l in layers)
    return ls.LoadedObject(cell_id=cell_id, kind="lora_all",
                           expect_hash=f"hash-{cell_id}", ops=ops,
                           effective_scale=float(scale))


def _registry(*objs, header=HEADER):
    return ls.WorkerObjectRegistry({o.cell_id: o for o in objs},
                                   assistant_header_ids=header)


def _sel(obj):
    return {"petri_object": obj.cell_id, "expect_hash": obj.expect_hash}


def _steering(registry, n_layers=4, tp=1):
    view = FakeView(layers=[OprojLayer() for _ in range(n_layers)])
    view.tensor_parallel_size = (lambda: tp) if tp != "missing" else None
    if tp == "missing":
        del view.tensor_parallel_size
    steering = lw.LensWorkerSteering(view=view, registry=registry)
    steering.preflight(object(), header_verifier=_ok_header)
    steering.install()
    return steering, view


class TestWorkerInstall:
    def test_oproj_hooks_installed_with_positions_stash(self):
        arm = _lora_obj(layers=(1, 3))
        steering, view = _steering(_registry(arm))
        rep = steering._report
        assert rep.layers == () and rep.oproj_layers == (1, 3)
        assert rep.plan_layer == 1
        for i, layer in enumerate(view.layers):
            n = 1 if i in (1, 3) else 0
            assert len(layer.self_attn.o_proj.hooks) == n
            assert len(layer.pre_hooks) == n     # the positions stash
            assert len(layer.hooks) == 0         # no block hooks needed

    def test_mixed_object_installs_both_sites(self):
        g = torch.Generator().manual_seed(2)
        block_op = la.HookOp(op=la.OP_ADD_VECTOR, layer=2,
                             vector=torch.ones(8))
        mixed = ls.LoadedObject(
            cell_id="arm-mixed", kind="stacked", expect_hash="hash-m",
            ops=(_lora_obj(layers=(1,)).ops[0], block_op),
            effective_scale=1.0)
        steering, view = _steering(_registry(mixed))
        rep = steering._report
        assert rep.layers == (2,) and rep.oproj_layers == (1,)
        assert rep.plan_layer == 1   # the o_proj hook fires first
        assert len(view.layers[2].hooks) == 1
        assert len(view.layers[1].self_attn.o_proj.hooks) == 1

    def test_tp_over_one_refuses_at_preflight(self):
        with pytest.raises(lw.LensWorkerError, match="tensor_parallel"):
            _steering(_registry(_lora_obj()), tp=2)

    def test_unreadable_tp_refuses_at_preflight(self):
        with pytest.raises(lw.LensWorkerError, match="tensor-parallel"):
            _steering(_registry(_lora_obj()), tp=None)

    def test_block_only_registry_never_consults_tp(self):
        vec = ls.LoadedObject(
            cell_id="arm-v", kind="synthetic_vector", expect_hash="h",
            ops=(la.HookOp(op=la.OP_ADD_VECTOR, layer=1,
                           vector=torch.ones(8)),),
            effective_scale=1.0)
        steering, _ = _steering(_registry(vec), tp="missing")
        assert steering._report.oproj_layers == ()

    def test_missing_oproj_module_refuses_at_preflight(self):
        view = FakeView(layers=[FakeLayer() for _ in range(4)])  # no attn
        view.tensor_parallel_size = lambda: 1
        steering = lw.LensWorkerSteering(view=view,
                                         registry=_registry(_lora_obj()))
        with pytest.raises(lw.LensWorkerError, match="o_proj"):
            steering.preflight(object(), header_verifier=_ok_header)


def _stash_positions(view, steering, pos):
    """Run the installed decoder pre-hooks the way a real forward would
    (with_kwargs shape: (module, args, kwargs) — here the llama-style
    positional call; the keyword-only convention has its own tests)."""
    for layer in view.layers:
        for pre in layer.pre_hooks:
            pre(None, (pos,), {})


class TestOprojHookBody:
    def _prefill_batch(self, view, arm, prompt, filler_q=2):
        qsl, seq_lens, nat, pos = _forward([(len(prompt), 0),
                                            (filler_q, 0)])
        view.set_batch(
            ["req-s", "req-u"],
            {"req-s": _sel(arm)},
            {"req-s": list(prompt), "req-u": list(range(50, 50 + filler_q))},
            (qsl, seq_lens, nat))
        return nat, pos

    def test_prefill_steers_the_masked_tail_rows_only(self):
        arm = _lora_obj(layers=(1,))
        steering, view = _steering(_registry(arm))
        prompt = [3, 4, 5, 6] + list(HEADER)   # P=7, start=4
        P, H = len(prompt), len(HEADER)
        nat, pos = self._prefill_batch(view, arm, prompt)
        _stash_positions(view, steering, pos)
        x = torch.randn(nat, D_IN)
        out = torch.randn(nat, D_OUT)
        out0 = out.clone()
        res = steering.oproj_hook_body(1, (x,), (out, None))
        new, bias = res
        assert bias is None
        assert torch.equal(out, out0)             # original unmodified
        assert torch.equal(new[:P - H], out0[:P - H])
        for i in range(P - H, P):
            assert not torch.equal(new[i], out0[i])
        assert torch.equal(new[P:], out0[P:])      # filler rows untouched
        # exactly the committed op arithmetic on the steered slice
        expect = la.apply_oproj_ops(
            x[:P], out0[:P], arm.ops, 1.0, layer=1,
            ctx=la.HookContext(la.PHASE_PREFILL, tuple(range(P)),
                               token_ids=tuple(prompt)))
        assert torch.equal(new[:P], expect)

    def test_decode_step_steers_its_single_row(self):
        arm = _lora_obj(layers=(1,))
        steering, view = _steering(_registry(arm))
        prompt = [3, 4] + list(HEADER)
        qsl, seq_lens, nat, pos = _forward([(1, len(prompt))])
        view.set_batch(["req-s"], {"req-s": _sel(arm)},
                       {"req-s": list(prompt) + [42]},
                       (qsl, seq_lens, nat),
                       prompt_lens={"req-s": len(prompt)})
        _stash_positions(view, steering, pos)
        x, out = torch.randn(1, D_IN), torch.randn(1, D_OUT)
        new = steering.oproj_hook_body(1, (x,), out)   # plain-tensor form
        assert torch.is_tensor(new) and not torch.equal(new, out)

    def test_scale0_twin_is_a_bitwise_passthrough(self):
        twin = _lora_obj(cell_id="none__lens0", scale=0.0)
        steering, view = _steering(_registry(twin))
        prompt = [3, 4] + list(HEADER)
        nat, pos = self._prefill_batch(view, twin, prompt)
        view.extra = {"req-s": dict(_sel(twin), scale=0.0)}
        _stash_positions(view, steering, pos)
        x, out = torch.randn(nat, D_IN), torch.randn(nat, D_OUT)
        assert steering.oproj_hook_body(1, (x,), (out, None)) is None

    def test_missing_stash_refuses_and_condemns(self):
        arm = _lora_obj(layers=(1,))
        steering, view = _steering(_registry(arm))
        prompt = [3, 4] + list(HEADER)
        nat, _pos = self._prefill_batch(view, arm, prompt)
        with pytest.raises(lw.LensWorkerError, match="stashed positions"):
            steering.oproj_hook_body(
                1, (torch.randn(nat, D_IN),), (torch.randn(nat, D_OUT),
                                               None))
        with pytest.raises(lw.LensWorkerError, match="CONDEMNED"):
            steering.begin_run([4])

    def test_deferred_bias_refuses(self):
        arm = _lora_obj(layers=(1,))
        steering, view = _steering(_registry(arm))
        prompt = [3, 4] + list(HEADER)
        nat, pos = self._prefill_batch(view, arm, prompt)
        _stash_positions(view, steering, pos)
        with pytest.raises(lw.LensWorkerError, match="skip_bias_add"):
            steering.oproj_hook_body(
                1, (torch.randn(nat, D_IN),),
                (torch.randn(nat, D_OUT), torch.zeros(D_OUT)))

    def test_ledger_consumes_once_via_the_oproj_site(self):
        arm = _lora_obj(layers=(1, 3))
        steering, view = _steering(_registry(arm))
        steering.begin_run([7])
        prompt = [3, 4, 5, 6] + list(HEADER)
        nat, pos = self._prefill_batch(view, arm, prompt, filler_q=2)
        _stash_positions(view, steering, pos)
        x, out = torch.randn(nat, D_IN), torch.randn(nat, D_OUT)
        for l in (1, 3):   # both hooked layers see the same forward
            steering.oproj_hook_body(l, (x,), (out, None))
        steering.finish_run()   # exactly one credit drawn -> clean finish


# ---------------------------------------------------------------------------
# the shim: fold + full registry round trip
# ---------------------------------------------------------------------------


def test_shim_folds_write_basis_and_roundtrips(tmp_path):
    from sisteer.capeval import manifest as cm
    from sisteer.capeval.serving import registry_shim as rs

    blob = _lora_all_blob()
    art = tmp_path / "cell.pt"
    art.write_bytes(_blob_bytes(blob))
    cond = {
        "cell_id": "safety_refusal__lora_r4",
        "artifact_path": str(art),
        "resolved_spec": {"kind": "lora_all", "positions": "response",
                          "response_header_len": len(HEADER)},
        "coefficient": 0.5,
        "object_sha256": cm.sha256_hex16(art.read_bytes()),
    }
    res = rs.build_lens_registry([cond], tmp_path / "reg",
                                 assistant_header_ids=HEADER)
    (c,) = res.conditions
    assert c.kind == "lora_all" and c.coefficient_was_folded
    assert c.effective_coefficient == 0.5
    reg = ls.WorkerObjectRegistry.load(__import__("pathlib").Path(
        res.registry_dir))
    (obj,) = reg.objects()
    assert obj.effective_scale == 1.0   # the fold carries the c, not scale
    for op in obj.ops:
        A = blob["A"][str(op.layer)]
        B = blob["B"][str(op.layer)]
        assert torch.equal(op.lowrank_a, A.t())          # read basis fixed
        assert torch.allclose(op.lowrank_b, 0.5 * B)     # write basis folded
        assert op.positions == la.POSITIONS_RESPONSE
        assert op.header_ids == HEADER


def test_shim_expect_served_coeff_refuses_on_lora(tmp_path):
    """The magnitude lives in a tensor, not a scalar — the driver's
    scalar cross-check is undefined here and must refuse, not pretend."""
    from sisteer.capeval import manifest as cm
    from sisteer.capeval.serving import registry_shim as rs

    art = tmp_path / "cell.pt"
    art.write_bytes(_blob_bytes(_lora_all_blob()))
    cond = {
        "cell_id": "c0", "artifact_path": str(art),
        "resolved_spec": {"kind": "lora_all"},
        "coefficient": 2.0,
        "object_sha256": cm.sha256_hex16(art.read_bytes()),
        "expect_served_coeff": 2.0,
    }
    with pytest.raises(rs.RegistryShimError, match="single scalar"):
        rs.build_lens_registry([cond], tmp_path / "reg",
                               assistant_header_ids=HEADER)
