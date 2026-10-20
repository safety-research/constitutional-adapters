"""lens_worker: the in-worker steering wiring — selector extraction,
per-request context construction over mixed forwards, the split-residual
write path, one-registration-only install, the ledger through the
orchestrator, and every preflight refusal. CPU-only: vllm/vllm-lens are
never importable here, so every vLLM-shaped input is a fake driven
through the module's injectable view seam (synthetic metadata, invented
token ids, fake decoder layers returning the (hidden, residual) pair).

The load-bearing tests are the split-residual one (the op must be
computed against hidden+residual and the delta written into hidden
alone — the test constructs a gate that FLIPS between the two streams,
so computing against hidden alone produces a provably different answer,
never a plausible one), the condemned-latch/ledger set (a mid-run
failure must poison every later forward and the run-end check), and the
Part G mixed-batch isolation set (no request's rows may ever be read
from or written by another request's slot — the cross-request
contamination class that only a GPU run could catch before Part G
existed)."""

import pytest
import torch

from sisteer.capeval.serving import lens_adapters as la
from sisteer.capeval.serving import lens_batch as lb
from sisteer.capeval.serving import lens_steer as ls
from sisteer.capeval.serving import lens_worker as lw

HIDDEN = 8


# --------------------------------------------------------------------------
# fakes: the view seam, decoder layers, registries
# --------------------------------------------------------------------------


class FakeLayer:
    """A decoder layer as the installer sees it: hook registration only.
    Mirrors torch's ``register_forward_hook(fn, *, with_kwargs=False)``
    surface — the install registers ``with_kwargs=True`` (qwen3_next-
    style models call their layers keyword-only), and the flag is
    recorded so tests can pin that."""

    def __init__(self):
        self.hooks = []
        self.hook_with_kwargs = []

    def register_forward_hook(self, fn, *, with_kwargs=False):
        self.hooks.append(fn)
        self.hook_with_kwargs.append(with_kwargs)
        return object()


class FakeView:
    """The vLLM-shaped attribute surface lens_worker depends on, faked.
    Mutable so one steering instance can be driven across forwards.

    ``prompt_lens`` mirrors the real view's ``prompt_len`` read
    (``len(CachedRequestState.prompt_token_ids)``): a request's prompt
    length is FIXED at submission, so it is remembered from the first
    forward that names the request (where ``tokens`` is the bare prompt)
    and kept across later ``set_batch`` calls, exactly like the runner
    state. Tests that first show a request mid-decode (or that model a
    recompute) pass ``prompt_lens`` explicitly."""

    def __init__(self, *, layers=None, req_ids=(), extra=None, tokens=None,
                 metadata=None, missing=(), eager=True, v2=False,
                 prompt_lens=None, prefix_caching=False):
        self.layers = list(layers) if layers is not None else []
        self.batch_req_ids = list(req_ids)
        self.extra = dict(extra or {})
        self.tokens = dict(tokens or {})
        self.metadata = metadata
        self.missing = tuple(missing)
        self.eager = eager
        self.v2 = v2
        self.prefix_caching = prefix_caching
        self.prompt_lens = dict(prompt_lens or {})
        for rid, toks in self.tokens.items():
            self.prompt_lens.setdefault(rid, len(toks))

    def set_batch(self, req_ids, extra, tokens, metadata, prompt_lens=None):
        self.batch_req_ids = list(req_ids)
        self.extra = dict(extra)
        self.tokens = dict(tokens)
        self.metadata = metadata
        if prompt_lens:
            self.prompt_lens.update(prompt_lens)
        for rid, toks in self.tokens.items():
            self.prompt_lens.setdefault(rid, len(toks))

    def num_reqs(self):
        return len(self.batch_req_ids)

    def req_ids(self):
        return tuple(self.batch_req_ids)

    def extra_args(self, req_id):
        return self.extra.get(req_id)

    def request_tokens(self, req_id):
        return self.tokens[req_id]

    def prompt_len(self, req_id):
        return self.prompt_lens[req_id]

    def forward_metadata(self):
        return self.metadata

    def decoder_layers(self):
        return self.layers

    def model_device_dtype(self):
        return torch.device("cpu"), torch.float32

    def enforce_eager(self):
        return self.eager

    def prefix_caching_enabled(self):
        # the RESOLVED cache state (None = unreadable, preflight refuses)
        return self.prefix_caching

    def v2_model_runner_active(self):
        return self.v2

    def missing_runner_attrs(self):
        return self.missing


def _row_local_op(layer=1, value=0.5):
    """positions=all, phase=both, no BOS skip — needs no context."""
    return la.HookOp(op=la.OP_ADD_VECTOR, layer=layer,
                     vector=torch.full((HIDDEN,), float(value)))


def _last_op(layer=1):
    """positions='last' — plan-needing."""
    return la.HookOp(op=la.OP_ADD_VECTOR, layer=layer,
                     vector=torch.ones(HIDDEN),
                     positions=la.POSITIONS_LAST)


def _prefill_only_op(layer=1):
    return la.HookOp(op=la.OP_ADD_VECTOR, layer=layer,
                     vector=torch.ones(HIDDEN), phase=la.PHASE_PREFILL)


def _gated_op(layer=1):
    """A ReLU-gated op (row-local: all positions, both phases) whose
    gate READS the hidden state — the discriminator between the true
    residual stream and the hidden_states half."""
    return la.HookOp(op=la.OP_RELU_GATED_VECTOR, layer=layer,
                     vector=torch.ones(HIDDEN),
                     gate_w=torch.ones(HIDDEN),
                     gate_b=torch.zeros(1))


def _obj(cell_id, ops, scale=1.0):
    return ls.LoadedObject(cell_id=cell_id, kind="synthetic_vector",
                           expect_hash=f"hash-{cell_id}", ops=tuple(ops),
                           effective_scale=float(scale))


def _registry(*objs, header=None):
    return ls.WorkerObjectRegistry({o.cell_id: o for o in objs},
                                   assistant_header_ids=header)


def _sel(obj, scale=None):
    out = {"petri_object": obj.cell_id, "expect_hash": obj.expect_hash}
    if scale is not None:
        out["scale"] = float(scale)
    return out


def _forward(segs):
    """(query_start_loc, seq_lens, num_actual_tokens, positions tensor)
    for (q, computed) segment shapes."""
    qsl, seq_lens, pos = [0], [], []
    for q, computed in segs:
        qsl.append(qsl[-1] + q)
        seq_lens.append(computed + q)
        pos.extend(range(computed, computed + q))
    return qsl, seq_lens, qsl[-1], torch.tensor(pos, dtype=torch.long)


def _ok_header(tok, recorded):
    return None


def _make_steering(registry, n_layers=4):
    view = FakeView(layers=[FakeLayer() for _ in range(n_layers)])
    steering = lw.LensWorkerSteering(view=view, registry=registry)
    steering.preflight(object(), header_verifier=_ok_header)
    steering.install()
    return steering, view


HEADER = (900, 901, 902, 903)


def _response_gated_op(layer=1):
    """A response-masked gated op — the Petri fitted-cell shape whose
    boundary location can fail PER REQUEST (header not locatable)."""
    return la.HookOp(op=la.OP_RELU_GATED_VECTOR, layer=layer,
                     vector=torch.ones(HIDDEN),
                     gate_w=torch.ones(HIDDEN),
                     gate_b=torch.zeros(1),
                     positions=la.POSITIONS_RESPONSE, header_ids=HEADER)


# --------------------------------------------------------------------------
# Part A — selector extraction
# --------------------------------------------------------------------------


class TestExtractSelector:
    def test_none_when_no_selector_keys(self):
        assert lw.extract_selector(None) is None
        assert lw.extract_selector({}) is None
        assert lw.extract_selector({"output_residual_stream": True}) is None

    def test_selector_coexists_with_other_subsystem_keys(self):
        got = lw.extract_selector({"petri_object": "cell-a",
                                   "expect_hash": "h1",
                                   "output_residual_stream": [3]})
        assert got == {"petri_object": "cell-a", "expect_hash": "h1"}

    def test_scale_passes_through(self):
        got = lw.extract_selector({"petri_object": "cell-a",
                                   "expect_hash": "h1", "scale": 0})
        assert got == {"petri_object": "cell-a", "expect_hash": "h1",
                       "scale": 0.0}

    @pytest.mark.parametrize("half", [{"expect_hash": "h1"},
                                      {"scale": 0.0},
                                      {"petri_object": "cell-a"}])
    def test_half_selector_refuses(self, half):
        with pytest.raises(ls.LensSteerError):
            lw.extract_selector(half)

    def test_bad_scale_type_refuses(self):
        with pytest.raises(ls.LensSteerError, match="scale"):
            lw.extract_selector({"petri_object": "a", "expect_hash": "h",
                                 "scale": "0"})


# --------------------------------------------------------------------------
# Part B — build_request_slots: contexts over a mixed forward
# --------------------------------------------------------------------------


class TestBuildRequestSlots:
    def test_mixed_batch_contexts_and_token_slicing(self):
        arm = _obj("arm-last", [_last_op()])
        reg = _registry(arm)
        # A: complete prefill q=5; B: decode at position 7; C: unhooked
        # complete prefill q=3.
        qsl, seq_lens, nat, pos = _forward([(5, 0), (1, 7), (3, 0)])
        tokens = {"req-a": [100, 101, 102, 103, 104],
                  "req-b": [200, 201, 202, 203, 204, 205, 206, 207],
                  "req-c": [300, 301, 302]}
        fwd = lw.build_request_slots(
            query_start_loc=qsl, seq_lens=seq_lens, num_actual_tokens=nat,
            positions=pos, req_ids=("req-a", "req-b", "req-c"),
            raw_selectors=(_sel(arm), _sel(arm), None),
            token_lookup=tokens.__getitem__,
            prompt_len_lookup={"req-a": 5, "req-b": 7,
                               "req-c": 3}.__getitem__,
            registry=reg)
        a, b, c = fwd.slots
        assert a.rows == slice(0, 5)
        assert a.ctx.phase == la.PHASE_PREFILL
        assert a.ctx.positions == (0, 1, 2, 3, 4)
        assert a.ctx.token_ids == (100, 101, 102, 103, 104)
        assert b.rows == slice(5, 6)
        assert b.ctx.phase == la.PHASE_DECODE
        assert b.ctx.positions == (7,)
        assert b.ctx.token_ids == (207,)  # the fed token, by position
        assert c.obj is None and c.selector is None
        assert c.ctx is not None  # contexts are built for every segment
        # the tracked projection: A and B only; A's prefill is the one
        # credit-drawing segment.
        assert fwd.plan is not None
        assert [s.q for s in fwd.tracked_plan.segments] == [5, 1]
        assert [s.q for s in fwd.tracked_plan.full_prefills] == [5]
        assert fwd.tracked_plan.n_decode == 1

    def test_token_shortfall_refuses(self):
        arm = _obj("arm-last", [_last_op()])
        reg = _registry(arm)
        qsl, seq_lens, nat, pos = _forward([(1, 7)])
        # the runner knows only 7 tokens but the segment needs position 7
        # — the decode-token timing assumption failed; must be loud.
        with pytest.raises(lw.LensWorkerError, match="timing"):
            lw.build_request_slots(
                query_start_loc=qsl, seq_lens=seq_lens,
                num_actual_tokens=nat, positions=pos, req_ids=("req-b",),
                raw_selectors=(_sel(arm),),
                token_lookup=lambda rid: list(range(7)),
                prompt_len_lookup=lambda rid: 7, registry=reg)

    def test_row_local_path_tolerates_chunked_shape(self):
        """A chunked-prefill continuation (q>=2, computed>0) refuses on
        the plan path but is chunking-safe for row-local ops — the
        asymmetry the ops_need_plan predicate exists for."""
        arm = _obj("arm-a", [_row_local_op()])
        reg = _registry(arm)
        qsl, seq_lens, nat, pos = _forward([(4, 3)])  # lattice-refused shape
        fwd = lw.build_request_slots(
            query_start_loc=qsl, seq_lens=seq_lens, num_actual_tokens=nat,
            positions=pos, req_ids=("req-a",), raw_selectors=(_sel(arm),),
            token_lookup=lambda rid: list(range(7)),
            prompt_len_lookup=lambda rid: 7, registry=reg)
        assert fwd.plan is None and fwd.tracked_plan is None
        assert fwd.slots[0].rows == slice(0, 4)
        assert fwd.slots[0].ctx is None

    def test_scale0_does_not_force_the_plan(self):
        """The no-op leg's donor ops may be position-dependent, but at
        effective scale 0 nothing is applied — mis-segmentation cannot
        change the math, so the lattice/ledger machinery stays off."""
        noop = _obj("none__lens0", [_last_op()], scale=0.0)
        reg = _registry(noop)
        qsl, seq_lens, nat, pos = _forward([(4, 3)])  # lattice-refused shape
        fwd = lw.build_request_slots(
            query_start_loc=qsl, seq_lens=seq_lens, num_actual_tokens=nat,
            positions=pos, req_ids=("req-a",),
            raw_selectors=(_sel(noop, scale=0.0),),
            token_lookup=lambda rid: list(range(7)),
            prompt_len_lookup=lambda rid: 7, registry=reg)
        assert fwd.plan is None
        assert fwd.slots[0].obj.effective_scale == 0.0

    def test_plan_shape_refusals_still_fire_for_tracked_ops(self):
        arm = _obj("arm-last", [_last_op()])
        reg = _registry(arm)
        qsl, seq_lens, nat, pos = _forward([(4, 3)])
        with pytest.raises(lb.LensBatchError):
            lw.build_request_slots(
                query_start_loc=qsl, seq_lens=seq_lens,
                num_actual_tokens=nat, positions=pos, req_ids=("req-a",),
                raw_selectors=(_sel(arm),),
                token_lookup=lambda rid: list(range(7)),
                prompt_len_lookup=lambda rid: 7, registry=reg)

    def test_positions_cross_check_is_wired(self):
        arm = _obj("arm-last", [_last_op()])
        reg = _registry(arm)
        qsl, seq_lens, nat, _ = _forward([(5, 0)])
        bad_pos = torch.tensor([0, 1, 2, 3, 9])  # last row misplaced
        with pytest.raises(lb.LensBatchError, match="positions"):
            lw.build_request_slots(
                query_start_loc=qsl, seq_lens=seq_lens,
                num_actual_tokens=nat, positions=bad_pos,
                req_ids=("req-a",), raw_selectors=(_sel(arm),),
                token_lookup=lambda rid: list(range(5)),
                prompt_len_lookup=lambda rid: 5, registry=reg)

    @pytest.mark.parametrize("make_ops", [
        lambda: [_last_op()],       # plan path
        lambda: [_row_local_op()],  # row-bounds path
    ])
    def test_segment_request_count_mismatch_refuses(self, make_ops):
        arm = _obj("arm", make_ops())
        reg = _registry(arm)
        qsl, seq_lens, nat, pos = _forward([(5, 0), (1, 7)])
        with pytest.raises(lw.LensWorkerError, match="drift"):
            lw.build_request_slots(
                query_start_loc=qsl, seq_lens=seq_lens,
                num_actual_tokens=nat, positions=pos, req_ids=("req-a",),
                raw_selectors=(_sel(arm),),
                token_lookup=lambda rid: list(range(10)),
                prompt_len_lookup=lambda rid: 5, registry=reg)

    def test_selector_count_mismatch_refuses(self):
        reg = _registry(_obj("arm", [_row_local_op()]))
        qsl, seq_lens, nat, pos = _forward([(5, 0)])
        with pytest.raises(lw.LensWorkerError, match="selectors"):
            lw.build_request_slots(
                query_start_loc=qsl, seq_lens=seq_lens,
                num_actual_tokens=nat, positions=pos, req_ids=("req-a",),
                raw_selectors=(), token_lookup=lambda rid: [],
                prompt_len_lookup=lambda rid: 5, registry=reg)

    def test_selector_resolution_refusals(self):
        arm = _obj("arm-a", [_row_local_op()])
        reg = _registry(arm)
        qsl, seq_lens, nat, pos = _forward([(5, 0)])

        def build(sel):
            return lw.build_request_slots(
                query_start_loc=qsl, seq_lens=seq_lens,
                num_actual_tokens=nat, positions=pos, req_ids=("req-a",),
                raw_selectors=(sel,),
                token_lookup=lambda rid: list(range(5)),
                prompt_len_lookup=lambda rid: 5, registry=reg)

        with pytest.raises(ls.LensSteerError, match="unknown petri_object"):
            build({"petri_object": "ghost", "expect_hash": "hash-ghost"})
        with pytest.raises(ls.LensSteerError, match="expect_hash"):
            build({"petri_object": "arm-a", "expect_hash": "wrong"})
        with pytest.raises(ls.LensSteerError, match="scale"):
            build({"petri_object": "arm-a", "expect_hash": "hash-arm-a",
                   "scale": 0.5})


# --------------------------------------------------------------------------
# Part C — preflight and install
# --------------------------------------------------------------------------


class TestPreflight:
    def _steering(self, registry=None, **view_kw):
        view_kw.setdefault("layers", [FakeLayer() for _ in range(4)])
        reg = registry if registry is not None \
            else _registry(_obj("arm", [_row_local_op()]))
        return lw.LensWorkerSteering(view=FakeView(**view_kw), registry=reg)

    def test_missing_runner_attrs_refuse(self):
        s = self._steering(missing=("input_batch.num_reqs",))
        with pytest.raises(lw.LensWorkerError, match="input_batch.num_reqs"):
            s.preflight(object(), header_verifier=_ok_header)

    def test_v2_runner_refuses(self):
        s = self._steering(v2=True)
        with pytest.raises(lw.LensWorkerError, match="V2"):
            s.preflight(object(), header_verifier=_ok_header)

    @pytest.mark.parametrize("eager", [False, None])
    def test_enforce_eager_must_be_verifiably_on(self, eager):
        s = self._steering(eager=eager)
        with pytest.raises(lw.LensWorkerError, match="enforce_eager"):
            s.preflight(object(), header_verifier=_ok_header)

    def test_empty_registry_refuses(self):
        s = self._steering(registry=_registry())
        with pytest.raises(lw.LensWorkerError, match="EMPTY"):
            s.preflight(object(), header_verifier=_ok_header)

    def test_layer_out_of_range_refuses(self):
        s = self._steering(registry=_registry(_obj("arm",
                                                   [_row_local_op(layer=9)])))
        with pytest.raises(lw.LensWorkerError, match="9"):
            s.preflight(object(), header_verifier=_ok_header)

    def test_no_decoder_layers_refuses(self):
        s = self._steering(layers=[])
        with pytest.raises(lw.LensWorkerError, match="decoder layers"):
            s.preflight(object(), header_verifier=_ok_header)

    def test_missing_tokenizer_refuses(self):
        s = self._steering()
        with pytest.raises(lw.LensWorkerError, match="tokenizer"):
            s.preflight(None, header_verifier=_ok_header)

    def test_header_verifier_failure_propagates(self):
        s = self._steering()

        def bad(tok, recorded):
            raise ls.LensSteerError("assistant header mismatch")

        with pytest.raises(ls.LensSteerError, match="header"):
            s.preflight(object(), header_verifier=bad)

    def test_default_header_verifier_is_the_real_check(self):
        assert lw.DEFAULT_HEADER_VERIFIER is ls.verify_assistant_header


class TestInstall:
    def test_one_hook_per_steered_layer_once(self):
        reg = _registry(_obj("arm-a", [_row_local_op(layer=1)]),
                        _obj("arm-b", [_last_op(layer=3)]))
        view = FakeView(layers=[FakeLayer() for _ in range(4)])
        steering = lw.LensWorkerSteering(view=view, registry=reg)
        steering.preflight(object(), header_verifier=_ok_header)
        report = steering.install()
        assert report.registration_count == 1
        assert report.layers == (1, 3)
        assert report.plan_layer == 1
        assert [len(l.hooks) for l in view.layers] == [0, 1, 0, 1]
        assert report.ledger_cells == ("arm-b",)
        with pytest.raises(lw.LensWorkerError, match="twice"):
            steering.install()
        assert [len(l.hooks) for l in view.layers] == [0, 1, 0, 1]
        assert steering.per_request_registrations == 0

    def test_install_before_preflight_refuses(self):
        reg = _registry(_obj("arm", [_row_local_op()]))
        steering = lw.LensWorkerSteering(
            view=FakeView(layers=[FakeLayer() for _ in range(4)]),
            registry=reg)
        with pytest.raises(lw.LensWorkerError, match="preflight"):
            steering.install()

    def test_registered_closure_reaches_hook_body(self):
        reg = _registry(_obj("arm", [_row_local_op(layer=1)]))
        steering, view = _make_steering(reg)
        # a warmup forward (no requests) through the REGISTERED closure
        # (with_kwargs shape: (module, args, kwargs, output)):
        # bit-identical passthrough, and no per-request registration.
        out = view.layers[1].hooks[0](None, (), {}, None)
        assert out is None
        assert steering.per_request_registrations == 0

    def test_cells_needing_ledger_names_only_plan_needing_arms(self):
        reg = _registry(_obj("arm-local", [_row_local_op()]),
                        _obj("arm-last", [_last_op()]),
                        _obj("none__lens0", [_last_op()], scale=0.0))
        assert lw.cells_needing_ledger(reg) == ("arm-last",)


# --------------------------------------------------------------------------
# Part D — the hook body: the split-residual write path
# --------------------------------------------------------------------------


def _batch_one_prefill(view, arm, *, q=4, unhooked_q=3):
    """One steered complete prefill + one unhooked prefill."""
    qsl, seq_lens, nat, pos = _forward([(q, 0), (unhooked_q, 0)])
    view.set_batch(
        ["req-s", "req-u"],
        {"req-s": _sel(arm) if arm is not None else None},
        {"req-s": list(range(100, 100 + q)),
         "req-u": list(range(200, 200 + unhooked_q))},
        (qsl, seq_lens, nat))
    return qsl, seq_lens, nat, pos


class TestHookBody:
    def test_split_residual_write(self):
        """The op runs on the TRUE stream (hidden + residual); the delta
        lands on hidden alone; residual passes through as the SAME
        object; unhooked rows stay bit-identical. The gate is built to
        FLIP between the two streams, so the hidden-only bug produces a
        provably different tensor — asserted directly."""
        arm = _obj("arm-g", [_gated_op(layer=1)])
        steering, view = _make_steering(_registry(arm))
        qsl, seq_lens, nat, pos = _batch_one_prefill(view, arm)
        torch.manual_seed(0)
        # gate_w = ones: h rows sum negative (gate closed on hidden
        # alone), h+resid rows sum positive (gate open on the true
        # stream) — the discriminating construction.
        h = -torch.ones(nat, HIDDEN) + 0.05 * torch.randn(nat, HIDDEN)
        r = 3.0 * torch.ones(nat, HIDDEN) + 0.05 * torch.randn(nat, HIDDEN)
        h0, r0 = h.clone(), r.clone()
        out = steering.hook_body(1, (pos, h, r), (h, r))
        assert out is not None
        new_h, out_r = out
        assert out_r is r                      # residual object untouched
        assert torch.equal(r, r0)              # ... and unmodified
        assert torch.equal(h, h0)              # original hidden unmodified
        assert new_h is not h
        # unhooked request's rows: bit-identical.
        assert torch.equal(new_h[4:7], h0[4:7])
        # steered rows: exactly the true-stream application's delta.
        true = h0 + r0
        expect = la.apply_hook_ops(true[0:4], arm.ops, 1.0, layer=1)
        assert torch.equal(new_h[0:4], h0[0:4] + (expect - true[0:4]))
        # the discriminator: computed against hidden alone, the gate is
        # closed and the "steered" rows would be unchanged — a silently
        # wrong answer this test would catch.
        wrong = la.apply_hook_ops(h0[0:4], arm.ops, 1.0, layer=1)
        assert torch.equal(wrong, h0[0:4])
        assert not torch.equal(expect, true[0:4])

    def test_plain_tensor_output(self):
        arm = _obj("arm-g", [_gated_op(layer=1)])
        steering, view = _make_steering(_registry(arm))
        qsl, seq_lens, nat, pos = _batch_one_prefill(view, arm)
        h = torch.ones(nat, HIDDEN)
        h0 = h.clone()
        out = steering.hook_body(1, (pos, h), h)
        assert torch.is_tensor(out)
        expect = la.apply_hook_ops(h0[0:4], arm.ops, 1.0, layer=1)
        assert torch.equal(out[0:4], expect)
        assert torch.equal(out[4:7], h0[4:7])
        assert torch.equal(h, h0)

    def test_scale0_is_a_bitwise_passthrough(self):
        noop = _obj("none__lens0", [_last_op(layer=1)], scale=0.0)
        steering, view = _make_steering(_registry(noop))
        qsl, seq_lens, nat, pos = _forward([(4, 0)])
        view.set_batch(["req-s"], {"req-s": _sel(noop, scale=0.0)},
                       {"req-s": list(range(4))}, (qsl, seq_lens, nat))
        h, r = torch.randn(nat, HIDDEN), torch.randn(nat, HIDDEN)
        h0, r0 = h.clone(), r.clone()
        assert steering.hook_body(1, (pos, h, r), (h, r)) is None
        assert torch.equal(h, h0) and torch.equal(r, r0)

    def test_phase_skipped_decode_is_a_passthrough(self):
        """A prefill-only op in a decode-only forward: NOT INVOKED, so
        the hook returns None and the output stays the same object."""
        arm = _obj("arm-p", [_prefill_only_op(layer=1)])
        steering, view = _make_steering(_registry(arm))
        # a phase-restricted op is plan-needing; the in-proc harness's
        # run arms the ledger (the decode forward observes zero
        # prefills — fine). The request's PROMPT is 5 tokens; the 6th
        # known token is the fed decode token.
        steering.begin_run([5])
        qsl, seq_lens, nat, pos = _forward([(1, 5)])
        view.set_batch(["req-s"], {"req-s": _sel(arm)},
                       {"req-s": list(range(6))}, (qsl, seq_lens, nat),
                       prompt_lens={"req-s": 5})
        h, r = torch.randn(nat, HIDDEN), torch.randn(nat, HIDDEN)
        assert steering.hook_body(1, (pos, h, r), (h, r)) is None

    def test_warmup_forward_passes_through(self):
        arm = _obj("arm", [_row_local_op(layer=1)])
        steering, view = _make_steering(_registry(arm))
        view.set_batch([], {}, {}, None)  # no requests, no forward ctx
        assert steering.hook_body(1, (), None) is None

    def test_unhooked_only_forward_passes_through(self):
        arm = _obj("arm", [_row_local_op(layer=1)])
        steering, view = _make_steering(_registry(arm))
        # requests in flight, none carrying a selector — passthrough
        # even without metadata (nothing to place).
        view.set_batch(["req-u"], {}, {"req-u": [1, 2, 3]}, None)
        assert steering.hook_body(1, (), (torch.ones(3, HIDDEN),
                                          torch.ones(3, HIDDEN))) is None

    def test_missing_metadata_with_selector_refuses_and_condemns(self):
        arm = _obj("arm", [_row_local_op(layer=1)])
        steering, view = _make_steering(_registry(arm))
        qsl, seq_lens, nat, pos = _forward([(4, 0)])
        view.set_batch(["req-s"], {"req-s": _sel(arm)},
                       {"req-s": list(range(4))}, None)
        h = torch.ones(nat, HIDDEN)
        with pytest.raises(lw.LensWorkerError, match="no attention metadata"):
            steering.hook_body(1, (pos, h, h), (h, h))
        # the latch: a later, perfectly healthy forward still refuses,
        # and so does the run-level surface — the run is condemned.
        view.metadata = (qsl, seq_lens, nat)
        with pytest.raises(lw.LensWorkerError, match="CONDEMNED"):
            steering.hook_body(1, (pos, h, h), (h, h))
        with pytest.raises(lw.LensWorkerError, match="CONDEMNED"):
            steering.finish_run()
        with pytest.raises(lw.LensWorkerError, match="CONDEMNED"):
            steering.begin_run([4])

    def test_empty_args_refuse(self):
        arm = _obj("arm", [_last_op(layer=1)])
        steering, view = _make_steering(_registry(arm))
        _batch_one_prefill(view, arm)
        h = torch.ones(7, HIDDEN)
        with pytest.raises(lw.LensWorkerError, match="positional args"):
            steering.hook_body(1, (), (h, h))

    def test_short_output_refuses(self):
        arm = _obj("arm", [_row_local_op(layer=1)])
        steering, view = _make_steering(_registry(arm))
        qsl, seq_lens, nat, pos = _batch_one_prefill(view, arm)
        h = torch.ones(nat - 1, HIDDEN)  # fewer rows than the metadata
        with pytest.raises(lw.LensWorkerError, match="rows"):
            steering.hook_body(1, (pos, h, h), (h, h))

    @pytest.mark.parametrize("output", [
        {"h": 1},                              # not a tensor or pair
        (torch.ones(7, HIDDEN), "resid"),      # pair with a non-tensor
        (torch.ones(7, HIDDEN),) * 3,          # wrong arity
    ])
    def test_unrecognized_layer_output_refuses(self, output):
        arm = _obj("arm", [_row_local_op(layer=1)])
        steering, view = _make_steering(_registry(arm))
        qsl, seq_lens, nat, pos = _batch_one_prefill(view, arm)
        with pytest.raises(lw.LensWorkerError):
            steering.hook_body(1, (pos,), output)

    def test_reqid_count_mismatch_refuses(self):
        class LyingView(FakeView):
            def num_reqs(self):
                return 2

        arm = _obj("arm", [_row_local_op(layer=1)])
        view = LyingView(layers=[FakeLayer() for _ in range(4)])
        steering = lw.LensWorkerSteering(view=view, registry=_registry(arm))
        steering.preflight(object(), header_verifier=_ok_header)
        steering.install()
        view.set_batch(["req-a"], {"req-a": _sel(arm)},
                       {"req-a": [1, 2]}, None)
        view.batch_req_ids = ["req-a"]  # one id for a claimed 2 requests
        with pytest.raises(lw.LensWorkerError, match="request ids"):
            steering.hook_body(1, (), (torch.ones(2, HIDDEN),) * 2)


# --------------------------------------------------------------------------
# Part E — the ledger through the orchestrator
# --------------------------------------------------------------------------


def _tracked_setup(n_layers=4, extra_objs=()):
    arm = _obj("arm-last", [_last_op(layer=1),
                            _last_op(layer=3)])
    steering, view = _make_steering(_registry(arm, *extra_objs), n_layers)
    return arm, steering, view


def _drive(steering, view, arm, segs, tokens, layers=(1, 3), sel=None):
    """One forward: set the batch, run the hook at each steered layer."""
    qsl, seq_lens, nat, pos = _forward(segs)
    view.set_batch(list(tokens), {rid: (sel or _sel(arm))
                                  for rid in tokens},
                   tokens, (qsl, seq_lens, nat))
    h = torch.zeros(nat, HIDDEN)
    r = torch.zeros(nat, HIDDEN)
    out = None
    for l in layers:
        out = steering.hook_body(l, (pos, h, r), (h, r))
    return out


class TestLedger:
    def test_happy_path_consumes_once_per_forward(self):
        """Two steered layers see the same prefill forward; the single
        credit is drawn ONCE (a double draw would refuse at the second
        layer), and the run finishes clean."""
        arm, steering, view = _tracked_setup()
        steering.begin_run([5])
        _drive(steering, view, arm, [(5, 0)],
               {"req-a": list(range(5))})              # the prefill
        _drive(steering, view, arm, [(1, 5)],
               {"req-a": list(range(6))})              # a decode step
        steering.finish_run()  # no leftover, no over-draw

    def test_recompute_refuses_in_flight_and_condemns(self):
        arm, steering, view = _tracked_setup()
        steering.begin_run([5])
        with pytest.raises(lb.LensBatchError, match="DISCARDED"):
            _drive(steering, view, arm, [(7, 0)],       # 5 + 2 recompute
                   {"req-a": list(range(7))}, layers=(1,))
        with pytest.raises(lw.LensWorkerError, match="CONDEMNED"):
            steering.finish_run()

    def test_leftover_refuses_at_finish(self):
        arm, steering, view = _tracked_setup()
        steering.begin_run([5, 9])
        _drive(steering, view, arm, [(5, 0)], {"req-a": list(range(5))})
        with pytest.raises(lb.LensBatchError, match="9"):
            steering.finish_run()

    def test_unarmed_tracked_forward_serves_fine(self):
        """THE HTTP-SERVING REGRESSION (review defect 1). The serving
        path has no run boundary and never calls begin_run — an unarmed
        ledger must be SKIPPED, not an error: previously the first
        request selecting any plan-needing arm raised 'observe with no
        armed run' inside the hook, which CONDEMNED the worker and
        bricked the whole server for every later request. Unarmed, the
        per-request invariants (checked in build_request_slots) are the
        defense; a well-shaped prefill + decode must serve, steer, and
        leave the worker healthy."""
        arm, steering, view = _tracked_setup()
        out = _drive(steering, view, arm, [(5, 0)],
                     {"req-a": list(range(5))}, layers=(1,))
        assert out is not None                      # it actually steered
        _drive(steering, view, arm, [(1, 5)],       # ... and keeps serving
               {"req-a": list(range(6))}, layers=(1,))
        # the worker is NOT condemned: the run-level surface still
        # accepts an arming (begin_run is _require_alive-gated), and a
        # clean disarm leaves everything healthy.
        steering.begin_run([5])
        steering.abort_run()

    def test_unarmed_overlapping_requests_serve_independently(self):
        """Overlapping HTTP requests: request A prefills, then decodes
        SHARING a forward with request B's fresh prefill, then B decodes
        alone — no arming anywhere, every forward serves, and the
        per-request invariants hold per request (A's decode at
        computed >= len(A's prompt), B's prefill at q == len(B's
        prompt)), which no run-global multiset could bracket."""
        arm, steering, view = _tracked_setup()
        _drive(steering, view, arm, [(5, 0)],
               {"req-a": list(range(5))}, layers=(1,))
        # A decodes while B prefills, one flattened forward:
        qsl, seq_lens, nat, pos = _forward([(1, 5), (7, 0)])
        view.set_batch(["req-a", "req-b"],
                       {"req-a": _sel(arm), "req-b": _sel(arm)},
                       {"req-a": list(range(6)),
                        "req-b": list(range(7))},
                       (qsl, seq_lens, nat))
        h = torch.zeros(nat, HIDDEN)
        out = steering.hook_body(1, (pos, h, h), (h, h))
        assert out is not None
        _drive(steering, view, arm, [(1, 7)],
               {"req-b": list(range(8))}, layers=(1,))
        steering._require_alive()  # never condemned

    def test_untracked_prefill_draws_no_credit(self):
        """A row-local request's prefill in the same forward as a
        tracked one must not consume (or over-draw) a credit."""
        local = _obj("arm-local", [_row_local_op(layer=1)])
        arm, steering, view = _tracked_setup(extra_objs=(local,))
        steering.begin_run([5])  # the tracked request's length ONLY
        qsl, seq_lens, nat, pos = _forward([(5, 0), (7, 0)])
        view.set_batch(
            ["req-t", "req-u"],
            {"req-t": _sel(arm), "req-u": _sel(local)},
            {"req-t": list(range(5)), "req-u": list(range(7))},
            (qsl, seq_lens, nat))
        h = torch.zeros(nat, HIDDEN)
        steering.hook_body(1, (pos, h, h), (h, h))
        steering.finish_run()  # 7 never drew; 5 consumed exactly once

    def test_observe_fires_only_at_the_lowest_installed_layer(self):
        """The mechanism pin: a forward whose hooks somehow ran only at
        a higher layer leaves the credit undrawn — observation is keyed
        to the one layer that runs on every forward, not to every
        steered layer."""
        arm, steering, view = _tracked_setup()
        steering.begin_run([5])
        _drive(steering, view, arm, [(5, 0)],
               {"req-a": list(range(5))}, layers=(3,))
        with pytest.raises(lb.LensBatchError, match="never observed"):
            steering.finish_run()


# --------------------------------------------------------------------------
# Part E' — the per-request prefill invariants through the orchestrator
# (the always-on defense; no arming anywhere in this class)
# --------------------------------------------------------------------------


class TestRequestLocalInvariants:
    def _mk(self):
        arm = _obj("arm-last", [_last_op(layer=1)])
        steering, view = _make_steering(_registry(arm))
        return arm, steering, view

    def _forward_with(self, steering, view, arm, seg, tokens, prompt_len):
        qsl, seq_lens, nat, pos = _forward([seg])
        view.set_batch(["req-a"], {"req-a": _sel(arm)},
                       {"req-a": tokens}, (qsl, seq_lens, nat),
                       prompt_lens={"req-a": prompt_len})
        h = torch.zeros(nat, HIDDEN)
        return steering.hook_body(1, (pos, h, h), (h, h))

    def test_unarmed_recompute_refuses_and_condemns(self):
        """A preemption-recompute (prompt N=5 + 2 generated re-run as
        one fresh 7-token prefill) with NO armed ledger — over HTTP the
        ledger is never armed, so this is the shape that previously
        passed every check. The per-request invariant refuses it in
        flight and condemns the worker."""
        arm, steering, view = self._mk()
        with pytest.raises(lb.LensBatchError,
                           match="recompute.*DISCARDED|DISCARDED"):
            self._forward_with(steering, view, arm, (7, 0),
                               list(range(7)), prompt_len=5)
        with pytest.raises(lw.LensWorkerError, match="CONDEMNED"):
            self._forward_with(steering, view, arm, (5, 0),
                               list(range(5)), prompt_len=5)

    def test_unarmed_first_chunk_refuses(self):
        """The FIRST chunk of a chunked prefill (5 of 9 prompt tokens,
        computed == 0) is complete-prefill-shaped for the lattice; only
        the request's own prompt length exposes it."""
        arm, steering, view = self._mk()
        with pytest.raises(lb.LensBatchError, match="SHORTER"):
            self._forward_with(steering, view, arm, (5, 0),
                               list(range(9)), prompt_len=9)

    def test_unarmed_one_token_final_chunk_refuses(self):
        """The 1-token final chunk (computed == N-1 == 4 for a 5-token
        prompt) is bit-for-bit a decode step; ``computed < prompt_len``
        is what tells them apart — a PROMPT row must never be steered as
        a generated one."""
        arm, steering, view = self._mk()
        with pytest.raises(lb.LensBatchError, match="INSIDE"):
            self._forward_with(steering, view, arm, (1, 4),
                               list(range(5)), prompt_len=5)

    def test_genuine_decode_steps_pass(self):
        arm, steering, view = self._mk()
        # first decode: computed == prompt_len; later: computed > it.
        self._forward_with(steering, view, arm, (1, 5),
                           list(range(6)), prompt_len=5)
        self._forward_with(steering, view, arm, (1, 8),
                           list(range(9)), prompt_len=5)
        steering._require_alive()

    def test_sub_two_token_prompt_refuses(self):
        arm, steering, view = self._mk()
        with pytest.raises(lb.LensBatchError, match="< 2"):
            self._forward_with(steering, view, arm, (1, 1),
                               [7, 8], prompt_len=1)

    def test_untracked_requests_are_out_of_scope(self):
        """A recompute-shaped UNTRACKED prefill (scale-0 twin) beside a
        clean tracked one: the invariant governs tracked requests only,
        mirroring the tracked_plan projection — an untracked shape must
        not condemn anyone (nothing boundary-dependent is applied to
        it)."""
        arm = _obj("arm-last", [_last_op(layer=1)])
        noop = _obj("none__lens0", [_last_op(layer=1)], scale=0.0)
        steering, view = _make_steering(_registry(arm, noop))
        qsl, seq_lens, nat, pos = _forward([(5, 0), (7, 0)])
        view.set_batch(
            ["req-t", "req-z"],
            {"req-t": _sel(arm), "req-z": _sel(noop, scale=0.0)},
            {"req-t": list(range(5)), "req-z": list(range(7))},
            (qsl, seq_lens, nat),
            prompt_lens={"req-t": 5, "req-z": 5})  # z LOOKS recomputed
        h = torch.zeros(nat, HIDDEN)
        assert steering.hook_body(1, (pos, h, h), (h, h)) is not None
        steering._require_alive()

    def test_armed_ledger_still_cross_checks_on_top(self):
        """The in-proc harness's armed ledger is ADDITIONAL, not
        replaced: a run-level leftover (a length armed but never
        prefilled) still refuses at finish_run even though every
        per-request invariant passed."""
        arm, steering, view = self._mk()
        steering.begin_run([5, 9])
        self._forward_with(steering, view, arm, (5, 0),
                           list(range(5)), prompt_len=5)
        with pytest.raises(lb.LensBatchError, match="9"):
            steering.finish_run()


class TestAbortRun:
    def test_abort_clears_an_armed_run_for_the_next_begin(self):
        """The smoke's exception-path leak: generate() dies mid-run for
        a non-steering reason, finish_run is never reached, and without
        a disarm the NEXT run's begin_run failed with a misleading
        'already armed'."""
        arm = _obj("arm-last", [_last_op(layer=1)])
        steering, _view = _make_steering(_registry(arm))
        steering.begin_run([5])
        steering.abort_run()
        steering.begin_run([4])   # would raise "already armed" pre-fix
        steering.abort_run()

    def test_abort_is_a_safe_noop_when_unarmed(self):
        arm = _obj("arm-last", [_last_op(layer=1)])
        steering, _view = _make_steering(_registry(arm))
        steering.abort_run()      # nothing armed — must not raise

    def test_abort_is_callable_on_a_condemned_worker(self):
        """Cleanup must never be refused: after a condemnation the abort
        still disarms (so the harness's except-path RPC cannot raise a
        second, masking error) while the condemnation itself stays."""
        arm = _obj("arm-last", [_last_op(layer=1)])
        steering, view = _make_steering(_registry(arm))
        steering.begin_run([5])
        qsl, seq_lens, nat, pos = _forward([(7, 0)])
        view.set_batch(["req-a"], {"req-a": _sel(arm)},
                       {"req-a": list(range(7))}, (qsl, seq_lens, nat),
                       prompt_lens={"req-a": 5})
        h = torch.zeros(nat, HIDDEN)
        with pytest.raises(lb.LensBatchError):
            steering.hook_body(1, (pos, h, h), (h, h))
        steering.abort_run()      # no raise, despite the condemnation
        with pytest.raises(lw.LensWorkerError, match="CONDEMNED"):
            steering.begin_run([5])   # ... which is preserved


# --------------------------------------------------------------------------
# Part E'' — the resolved-scheduler-geometry report (launch verification)
# --------------------------------------------------------------------------


class TestSchedulerConfigReport:
    """Fakes mirror the REAL pinned 0.25.1 layout exactly:
    scheduler-owned values on scheduler_config, max_model_len on
    model_config ONLY (SchedulerConfig's is an InitVar — consumed by
    __post_init__, never stored), max_num_scheduled_tokens a real
    usually-None SchedulerConfig field, speculative_config a VllmConfig
    attribute that is None when disabled."""

    class _Sched:
        def __init__(self, seqs=8, tokens=131072, chunked=True,
                     threshold=0, partial=1, long_partial=1, mnst=None,
                     scheduler_cls=None, async_scheduling=False):
            self.max_num_seqs = seqs
            self.max_num_batched_tokens = tokens
            self.long_prefill_token_threshold = threshold
            self.max_num_partial_prefills = partial
            self.max_long_partial_prefills = long_partial
            self.max_num_scheduled_tokens = mnst
            self.enable_chunked_prefill = chunked
            # the real 0.25.1 defaults: scheduler_cls is None and
            # get_scheduler_cls resolves it (Scheduler, or AsyncScheduler
            # under async_scheduling).
            self.scheduler_cls = scheduler_cls
            self.async_scheduling = async_scheduling
            # NB deliberately NO max_model_len here — the class-level
            # docstring; test_real_layout_* pins this.

    class _Model:
        def __init__(self, mml=16384):
            self.max_model_len = mml

    class _Cache:
        def __init__(self, enabled=False, blocks=40960, block_size=16,
                     mamba_dtype="auto"):
            self.enable_prefix_caching = enabled
            # the post-profiling KV size gpu_worker.initialize_cache
            # writes (40960 x 16 = 655,360 tokens: covers the fake's
            # 131,072-token budget with room)
            self.num_gpu_blocks = blocks
            self.block_size = block_size
            # the real 0.25.1 CacheConfig field: a plain string, "auto"
            # unless --mamba-ssm-cache-dtype composed one. None here =
            # a surface WITHOUT the attribute (older/foreign config).
            if mamba_dtype is not None:
                self.mamba_ssm_cache_dtype = mamba_dtype

    class _Cfg:
        def __init__(self, sched, model="default", spec=None,
                     cache="default"):
            self.scheduler_config = sched
            if model == "default":
                model = TestSchedulerConfigReport._Model()
            if model is not None:
                self.model_config = model
            self.speculative_config = spec
            if cache == "default":
                cache = TestSchedulerConfigReport._Cache()
            if cache is not None:
                self.cache_config = cache

    _CLEAN = {"max_num_seqs": 8, "max_num_batched_tokens": 131072,
              "long_prefill_token_threshold": 0,
              "max_num_partial_prefills": 1,
              "max_long_partial_prefills": 1,
              "max_model_len": 16384,
              "effective_max_num_scheduled_tokens": 131072,
              "speculative": None,
              "scheduler_cls": "vllm.v1.core.sched.scheduler.Scheduler",
              "enable_chunked_prefill": True,
              "enable_prefix_caching": False,
              "mamba_ssm_cache_dtype": "auto",
              "kv_num_gpu_blocks": 40960, "kv_block_size": 16}

    def test_reports_plain_scalars(self):
        rep = lw.scheduler_config_report(self._Cfg(self._Sched()))
        assert rep == self._CLEAN

    def test_reports_resolved_mamba_ssm_cache_dtype(self):
        """The RESOLVED cache_config.mamba_ssm_cache_dtype rides the report
        as a string for every launch (recorded); the launcher asserts it
        only on a launch that composed the opt-in (expected_scheduler_state
        — see test_nemotron_chat_markers). A surface without the attribute
        reports None rather than refusing: the refusal belongs to the
        launcher's comparison, exactly on the launches that composed it."""
        rep = lw.scheduler_config_report(
            self._Cfg(self._Sched(), cache=self._Cache(mamba_dtype="float32")))
        assert rep["mamba_ssm_cache_dtype"] == "float32"
        rep = lw.scheduler_config_report(
            self._Cfg(self._Sched(), cache=self._Cache(mamba_dtype=None)))
        assert rep["mamba_ssm_cache_dtype"] is None
        # a torch dtype object on the surface is reported by its str form
        rep = lw.scheduler_config_report(
            self._Cfg(self._Sched(), cache=self._Cache(mamba_dtype=torch.float32)))
        assert rep["mamba_ssm_cache_dtype"] == "torch.float32"

    def test_real_layout_scheduler_config_has_no_max_model_len(self):
        """THE FAKE-ONLY-ATTRIBUTE REGRESSION. At the pinned tag,
        SchedulerConfig.max_model_len is an InitVar — NOT a runtime
        attribute — so a report that read it off scheduler_config
        (as an earlier version of this code did, against a fake that
        happened to carry it) would AttributeError-refuse on the REAL
        stack and brick every lens launch at readiness. The fake now
        matches the real layout, this test pins that it does, and the
        report must succeed reading model_config instead."""
        sched = self._Sched()
        assert not hasattr(sched, "max_model_len")  # the real layout
        rep = lw.scheduler_config_report(self._Cfg(sched))
        assert rep["max_model_len"] == 16384        # from model_config

    def test_max_model_len_only_on_scheduler_config_refuses(self):
        """The inverse pin: a config shaped like the OLD fake — a
        max_model_len attribute sitting on scheduler_config, no
        model_config — must refuse loudly, proving the report never
        falls back to the fake-only location."""
        sched = self._Sched()
        sched.max_model_len = 16384                 # the old fake shape
        with pytest.raises(lw.LensWorkerError,
                           match="model_config.max_model_len"):
            lw.scheduler_config_report(self._Cfg(sched, model=None))

    def test_missing_scheduler_config_refuses(self):
        class NoSched:
            pass

        with pytest.raises(lw.LensWorkerError, match="scheduler_config"):
            lw.scheduler_config_report(NoSched())

    def test_missing_field_refuses(self):
        for field in lw.SCHEDULER_CONFIG_FIELDS:
            sched = self._Sched()
            delattr(sched, field)
            with pytest.raises(lw.LensWorkerError, match=field):
                lw.scheduler_config_report(self._Cfg(sched))

    def test_unreadable_field_refuses(self):
        sched = self._Sched(tokens="lots")
        with pytest.raises(lw.LensWorkerError, match="not an integer"):
            lw.scheduler_config_report(self._Cfg(sched))

    def test_effective_ceiling_resolves_none_to_the_budget(self):
        rep = lw.scheduler_config_report(self._Cfg(self._Sched(mnst=None)))
        assert rep["effective_max_num_scheduled_tokens"] == 131072
        # a diverged ceiling (what spec decode does) is reported as-is,
        # for the launcher to refuse against its expectation:
        rep = lw.scheduler_config_report(self._Cfg(self._Sched(mnst=8192)))
        assert rep["effective_max_num_scheduled_tokens"] == 8192

    def test_missing_scheduled_tokens_field_refuses(self):
        sched = self._Sched()
        delattr(sched, "max_num_scheduled_tokens")
        with pytest.raises(lw.LensWorkerError,
                           match="max_num_scheduled_tokens"):
            lw.scheduler_config_report(self._Cfg(sched))

    def test_unreadable_kv_cache_size_refuses(self):
        """The resolved KV size (num_gpu_blocks x block_size, written by
        gpu_worker.initialize_cache after memory profiling) is what the
        launcher holds to the composed budget — max_num_seqs is a
        ceiling, not a grant, and an under-covered geometry silently
        preempts its way down (48x16384 over a 146,118-token cache =
        8.92x delivered, job 232411). Unreadable = refuse, never pass."""
        for kw in ({"blocks": None}, {"blocks": 0}, {"blocks": True},
                   {"block_size": None}, {"block_size": -1}):
            with pytest.raises(lw.LensWorkerError, match="KV cache"):
                lw.scheduler_config_report(
                    self._Cfg(self._Sched(),
                              cache=self._Cache(**kw)))

    def test_speculative_state_is_reported(self):
        class SpecCfg:
            num_speculative_tokens = 3

        rep = lw.scheduler_config_report(
            self._Cfg(self._Sched(), spec=SpecCfg()))
        assert rep["speculative"] == "SpecCfg(num_speculative_tokens=3)"

    def test_missing_speculative_attribute_refuses(self):
        cfg = self._Cfg(self._Sched())
        del cfg.speculative_config
        with pytest.raises(lw.LensWorkerError, match="speculative"):
            lw.scheduler_config_report(cfg)

    def test_scheduler_cls_none_resolves_like_the_pinned_default(self):
        """The pinned runtime state: scheduler_cls is STORED as None and
        get_scheduler_cls resolves it — to the stock synchronous
        Scheduler, or to AsyncScheduler iff async_scheduling. The report
        mirrors exactly that resolution."""
        rep = lw.scheduler_config_report(self._Cfg(self._Sched()))
        assert rep["scheduler_cls"] == \
            "vllm.v1.core.sched.scheduler.Scheduler"
        rep = lw.scheduler_config_report(
            self._Cfg(self._Sched(async_scheduling=True)))
        assert rep["scheduler_cls"] == \
            "vllm.v1.core.sched.async_scheduler.AsyncScheduler"

    def test_custom_scheduler_cls_is_reported_verbatim(self):
        # a custom qualname string reports as-is ...
        rep = lw.scheduler_config_report(
            self._Cfg(self._Sched(scheduler_cls="evil.Scheduler")))
        assert rep["scheduler_cls"] == "evil.Scheduler"

        # ... and a class OBJECT (the other accepted form at the tag)
        # reports as its dotted path.
        class MyScheduler:
            pass

        rep = lw.scheduler_config_report(
            self._Cfg(self._Sched(scheduler_cls=MyScheduler)))
        assert rep["scheduler_cls"].endswith("MyScheduler")
        assert "." in rep["scheduler_cls"]

    def test_missing_scheduler_cls_surface_refuses(self):
        for attr in ("scheduler_cls", "async_scheduling"):
            sched = self._Sched()
            delattr(sched, attr)
            with pytest.raises(lw.LensWorkerError, match=attr):
                lw.scheduler_config_report(self._Cfg(sched))

    def test_chunked_prefill_flag_is_informational(self):
        sched = self._Sched()
        del sched.enable_chunked_prefill
        rep = lw.scheduler_config_report(self._Cfg(sched))
        assert rep["enable_chunked_prefill"] is None  # recorded, not asserted


# --------------------------------------------------------------------------
# Part F — module wiring facts
# --------------------------------------------------------------------------


class TestModuleWiring:
    def test_worker_extension_cls_names_this_module(self):
        assert lw.WORKER_EXTENSION_CLS == (
            "sisteer.capeval.serving.lens_worker.SisteerWorkerExtension")

    def test_extension_absent_without_vllm_lens(self):
        """On this CPU box vllm-lens is unimportable, so the GPU-only
        extension class must not exist — and the module must import
        anyway (the guard the whole file depends on)."""
        if not ls.HAVE_VLLM_LENS:
            assert not hasattr(lw, "SisteerWorkerExtension")
        else:  # pragma: no cover — GPU venv only
            assert issubclass(lw.SisteerWorkerExtension, object)

    def test_required_runner_attrs_pin_the_v1_surface(self):
        assert lw.REQUIRED_RUNNER_ATTRS == (
            "input_batch.num_reqs", "input_batch.req_ids", "requests",
            "model")


# --------------------------------------------------------------------------
# Part G — mixed-batch isolation (cross-request contamination coverage)
# --------------------------------------------------------------------------
#
# The first steered GPU smoke (Slurm 173152) flagged two mixed-batch
# requests as contaminated. The forensics EXONERATED this wiring: every
# multi-request composition agreed bit-for-bit (the flagged scale-0 row's
# mixed-batch output equalled the pure-engine no-selectors baseline; the
# flagged steered row's equalled its all-steered batch), and the outputs
# that moved were the SOLO batch-of-1 references — vLLM 0.25.1's
# VLLM_BATCH_INVARIANT=1 does not pin cuBLASLt kernel selection across
# batch sizes on SM90, so a lone request's GEMMs reduce in a different
# order than any batched request's and greedy flips at near-tie tokens
# (first divergences at 14 and 22, where a wrong-row steer corrupts
# token 0 — fidelity.py's own forensic note). What the investigation
# exposed HERE is a coverage hole: no CPU test mixed SELECTIONS within
# one forward, so a real cross-request write could only ever have been
# caught by a GPU run. This part closes the hole. Every invariant a
# mixed batch must hold is pinned against fakes:
#
#   - scale-0 and unhooked rows BIT-identical beside steered neighbours
#     (untouched, not merely edited-by-zero);
#   - a request's edit invariant to its row offset and its neighbours
#     (the segment-relative-vs-batch-absolute indexing bug is exactly a
#     position-dependent delta: correct solo, wrong once other requests
#     shift the offsets);
#   - distinguished-row ops (the AlphaSteer last-row matmul, the cosine
#     gate) read THEIR OWN segment's last row, never the batch's;
#   - phase restrictions hold per request inside one mixed forward
#     (a decoding request under a prefill-only arm passes through even
#     while a neighbouring prefill is steered);
#   - the pairing of rows to requests follows the LIVE req_ids order on
#     every forward, so a condense/swap reorder between forwards moves
#     the edit with the request.


def _alpha_op(layer=1, coeff=1.0):
    """An AlphaSteer-shaped op: the edit is computed from the LAST ROW
    of the tensor the op is handed and broadcast to every prompt
    position — THE distinguished-row op, and therefore the one most
    sensitive to being handed the wrong slice. Identity matrix, so the
    edit's value IS the row that was read (maximally discriminating)."""
    return la.HookOp(op=la.OP_PREFILL_LAST_MATMUL, layer=layer,
                     matrix=torch.eye(HIDDEN), coeff=float(coeff),
                     phase=la.PHASE_PREFILL)


def _cosine_op(layer=1, threshold=0.0):
    """A cosine-gated op: one hard per-REQUEST decision read off the
    last prompt token — the other distinguished-row read."""
    cond = torch.zeros(HIDDEN)
    cond[0] = 1.0
    return la.HookOp(op=la.OP_PREFILL_LAST_COSINE_GATED, layer=layer,
                     vector=torch.ones(HIDDEN), cond=cond,
                     threshold=float(threshold), phase=la.PHASE_PREFILL,
                     positions=la.POSITIONS_LAST)


def _prefill_ctx(q):
    return la.HookContext(phase=la.PHASE_PREFILL,
                          positions=tuple(range(q)))


def _expected_rows(h0, r0, rows, obj, layer, ctx):
    """One slot's expected post-write hidden rows, computed independently
    of the wiring but mirroring its exact arithmetic: the op runs on the
    true stream (hidden + residual) of THIS slice alone; the delta lands
    on hidden alone."""
    true = h0 + r0
    applied = la.apply_hook_ops(true[rows], obj.ops, obj.effective_scale,
                                layer=layer, ctx=ctx)
    return h0[rows] + (applied - true[rows])


def _slices(qsl):
    return [slice(a, b) for a, b in zip(qsl, qsl[1:])]


class TestMixedBatchIsolation:
    def _drive(self, reg, spec, *, layer, tracked_lens=None, h=None, r=None,
               seed=7):
        """One mixed forward on a fresh steering instance.

        ``spec``: (req_id, (q, computed), selector-or-None) in batch
        order. Returns (steering, new_hidden, h0, r0, slices) with
        ``new_hidden`` the hook's replacement hidden (h0 when the hook
        returned None — a passthrough is bit-identity by construction)."""
        steering, view = _make_steering(reg)
        if tracked_lens is not None:
            steering.begin_run(tracked_lens)
        qsl, seq_lens, nat, pos = _forward([shape for _rid, shape, _s in spec])
        tokens = {rid: list(range(1000 * i, 1000 * i + c + q))
                  for i, (rid, (q, c), _s) in enumerate(spec)}
        # a prefill's prompt IS its q rows; a request first seen at a
        # decode step has its whole context as prompt (this is its first
        # generated token) — the real runner would report exactly that.
        prompt_lens = {rid: (q if c == 0 else c)
                       for rid, (q, c), _s in spec}
        view.set_batch([rid for rid, _shape, _s in spec],
                       {rid: sel for rid, _shape, sel in spec
                        if sel is not None},
                       tokens, (qsl, seq_lens, nat),
                       prompt_lens=prompt_lens)
        if h is None:
            gen = torch.Generator().manual_seed(seed)
            h = torch.randn(nat, HIDDEN, generator=gen)
            r = torch.randn(nat, HIDDEN, generator=gen)
        h0, r0 = h.clone(), r.clone()
        out = steering.hook_body(layer, (pos, h, r), (h, r))
        # invariants of EVERY forward: the layer's own output tensors are
        # never modified in place, and the residual object passes through.
        assert torch.equal(h, h0) and torch.equal(r, r0)
        if out is None:
            return steering, h0, h0, r0, _slices(qsl)
        new_h, out_r = out
        assert out_r is r
        assert new_h is not h
        return steering, new_h, h0, r0, _slices(qsl)

    def test_smoke_shaped_mixed_prefill_isolation(self):
        """The GPU smoke's mixed batch, on CPU: unhooked requests, a
        prefill-only vector arm (static_r-shaped) and a last-row matmul
        arm (alphasteer-shaped), each beside its scale-0 twin, in ONE
        flattened prefill forward. Every steered request's rows must be
        exactly its own solo application; every scale-0 and unhooked
        request's rows must be BIT-identical — 'a delta that is zero for
        its own rows' is not the same claim as 'untouched'."""
        static = _obj("static-arm", [_prefill_only_op(layer=1)])
        static0 = _obj("static-arm__scale0", [_prefill_only_op(layer=1)],
                       scale=0.0)
        alpha = _obj("alpha-arm", [_alpha_op(layer=2, coeff=-0.3)])
        alpha0 = _obj("alpha-arm__scale0", [_alpha_op(layer=2, coeff=-0.3)],
                      scale=0.0)
        reg = _registry(static, static0, alpha, alpha0)
        spec = [("req-n0", (5, 0), None),
                ("req-s1", (4, 0), _sel(static)),
                ("req-s2", (6, 0), _sel(static)),
                ("req-z1", (4, 0), _sel(static0, scale=0.0)),
                ("req-a1", (5, 0), _sel(alpha)),
                ("req-a2", (4, 0), _sel(alpha)),
                ("req-z2", (6, 0), _sel(alpha0, scale=0.0)),
                ("req-n1", (3, 0), None)]
        tracked = [4, 6, 5, 4]  # the scale-1 requests' prompt lengths

        # layer 1 — the static arm's layer (and the plan layer).
        steering, new_h, h0, r0, rows = self._drive(
            reg, spec, layer=1, tracked_lens=tracked)
        for i in (0, 3, 4, 5, 6, 7):  # unhooked, scale-0, other-layer arms
            assert torch.equal(new_h[rows[i]], h0[rows[i]]), f"slot {i}"
        for i in (1, 2):
            q = spec[i][1][0]
            assert torch.equal(
                new_h[rows[i]],
                _expected_rows(h0, r0, rows[i], static, 1, _prefill_ctx(q)))

        # layer 2 — the alpha arm's layer, same forward metadata (fresh
        # tensors: in a real forward layer 2 sees layer 1's output; no
        # ledger arming needed — only the plan layer, 1, consumes).
        steering2, new_h2, h20, r20, rows2 = self._drive(
            reg, spec, layer=2, tracked_lens=None, seed=8)
        for i in (0, 1, 2, 3, 6, 7):
            assert torch.equal(new_h2[rows2[i]], h20[rows2[i]]), f"slot {i}"
        for i in (4, 5):
            q = spec[i][1][0]
            assert torch.equal(
                new_h2[rows2[i]],
                _expected_rows(h20, r20, rows2[i], alpha, 2, _prefill_ctx(q)))

        # the accounting: one forward over two steered layers consumed
        # each tracked prefill exactly once.
        steering.hook_body(2, *self._replay_args(spec, seed=7))
        steering.finish_run()

    @staticmethod
    def _replay_args(spec, *, seed):
        qsl, seq_lens, nat, pos = _forward(
            [shape for _rid, shape, _s in spec])
        gen = torch.Generator().manual_seed(seed)
        h = torch.randn(nat, HIDDEN, generator=gen)
        r = torch.randn(nat, HIDDEN, generator=gen)
        return (pos, h, r), (h, r)

    def test_last_row_op_reads_its_own_segments_last_row(self):
        """Handed the whole flattened batch instead of one request's
        slice, OP_PREFILL_LAST_MATMUL would steer EVERY request from the
        batch's last request's last token — the exact contamination
        shape the GPU smoke was suspected of. Each segment's last row
        carries a distinct value, so own-row and batch-row reads give
        provably different edits."""
        alpha = _obj("alpha-arm", [_alpha_op(layer=1)])
        spec = [("req-a", (4, 0), _sel(alpha)),
                ("req-b", (3, 0), _sel(alpha)),
                ("req-u", (2, 0), None)]
        h = torch.zeros(9, HIDDEN)
        h[3] = 1.0   # req-a's own last prompt row
        h[6] = 2.0   # req-b's own last prompt row
        h[8] = 7.0   # the BATCH's last row (req-u's) — the wrong read
        r = torch.zeros(9, HIDDEN)
        steering, new_h, h0, r0, rows = self._drive(
            _registry(alpha), spec, layer=1, tracked_lens=[4, 3],
            h=h, r=r)
        assert torch.equal(new_h[rows[0]], h0[rows[0]] + 1.0)
        assert torch.equal(new_h[rows[1]], h0[rows[1]] + 2.0)
        assert torch.equal(new_h[rows[2]], h0[rows[2]])  # unhooked
        # the batch-last-row read would have added 7.0 everywhere:
        assert not torch.equal(new_h[rows[0]], h0[rows[0]] + 7.0)
        steering.finish_run()

    def test_cosine_gate_decides_per_request_from_its_own_last_row(self):
        """The cosine gate is one hard decision per REQUEST, read off
        that request's own last prompt token. Constructed so the firing
        request's last row is aligned with the probe and the holding
        request's (which is also the BATCH's last row) is anti-aligned:
        a batch-global read would hold both."""
        gate = _obj("gate-arm", [_cosine_op(layer=1)])
        spec = [("req-fire", (3, 0), _sel(gate)),
                ("req-hold", (4, 0), _sel(gate))]
        h = torch.zeros(7, HIDDEN)
        h[2, 0] = 1.0    # req-fire's last row: cos = +1 > 0
        h[6, 0] = -1.0   # req-hold's last row (batch-last): cos = -1 < 0
        r = torch.zeros(7, HIDDEN)
        steering, new_h, h0, r0, rows = self._drive(
            _registry(gate), spec, layer=1, tracked_lens=[3, 4], h=h, r=r)
        assert torch.equal(
            new_h[rows[0]],
            _expected_rows(h0, r0, rows[0], gate, 1, _prefill_ctx(3)))
        assert not torch.equal(new_h[rows[0]], h0[rows[0]])  # it fired
        assert torch.equal(new_h[rows[1]], h0[rows[1]])      # bit-held
        steering.finish_run()

    def test_request_edit_is_invariant_to_batch_position_and_neighbors(self):
        """One request — same tokens, same row values — served solo,
        first in a mixed batch, and last in a different mixed batch with
        another steered request beside it. The delta written to its rows
        must be BIT-identical in all three. The classic
        segment-relative-vs-batch-absolute indexing bug is exactly a
        position-dependent delta: correct solo, wrong once neighbours
        shift the offsets — which is why the GPU smoke's mixed batch was
        the first place it could ever have shown."""
        alpha = _obj("alpha-arm", [_alpha_op(layer=1)])
        alpha0 = _obj("alpha-arm__scale0", [_alpha_op(layer=1)], scale=0.0)
        reg = _registry(alpha, alpha0)
        gen = torch.Generator().manual_seed(21)
        block_h = torch.randn(4, HIDDEN, generator=gen)
        block_r = torch.randn(4, HIDDEN, generator=gen)
        target = ("req-t", (4, 0), _sel(alpha))

        def run(spec, t_index, tracked_lens, seed):
            qsl = [0]
            for _rid, (q, _c), _s in spec:
                qsl.append(qsl[-1] + q)
            gen2 = torch.Generator().manual_seed(seed)
            h = torch.randn(qsl[-1], HIDDEN, generator=gen2)
            r = torch.randn(qsl[-1], HIDDEN, generator=gen2)
            t_rows = slice(qsl[t_index], qsl[t_index + 1])
            h[t_rows] = block_h
            r[t_rows] = block_r
            steering, new_h, h0, _r0, rows = self._drive(
                reg, spec, layer=1, tracked_lens=tracked_lens, h=h, r=r)
            steering.finish_run()
            return new_h[rows[t_index]] - h0[rows[t_index]]

        d_solo = run([target], 0, [4], seed=31)
        d_first = run([target,
                       ("req-n", (5, 0), None),
                       ("req-z", (3, 0), _sel(alpha0, scale=0.0))],
                      0, [4], seed=32)
        d_last = run([("req-n", (3, 0), None),
                      ("req-o", (5, 0), _sel(alpha)),
                      target],
                     2, [5, 4], seed=33)
        assert torch.equal(d_solo, d_first)
        assert torch.equal(d_solo, d_last)
        assert not torch.equal(d_solo, torch.zeros_like(d_solo))

    def test_mixed_prefill_and_decode_forward_isolation(self):
        """The post-condense shape: decode steps sharing one forward
        with a fresh prefill. A DECODING request under a prefill-only
        arm passes through untouched even though its selector is armed
        and a neighbouring prefill is being steered; a both-phase arm's
        decode row gets exactly its one-row edit; unhooked and scale-0
        decode rows stay bit-identical."""
        pre = _obj("pre-arm", [_prefill_only_op(layer=1)])
        both = _obj("both-arm", [_last_op(layer=1)])
        both0 = _obj("both-arm__scale0", [_last_op(layer=1)], scale=0.0)
        alpha = _obj("alpha-arm", [_alpha_op(layer=1)])
        reg = _registry(pre, both, both0, alpha)
        spec = [("req-d1", (1, 6), _sel(pre)),    # decode, prefill-only arm
                ("req-d2", (1, 9), _sel(both)),   # decode, both-phase arm
                ("req-p", (5, 0), _sel(alpha)),   # the fresh prefill
                ("req-d3", (1, 4), None),         # decode, unhooked
                ("req-dz", (1, 7), _sel(both0, scale=0.0))]
        steering, new_h, h0, r0, rows = self._drive(
            reg, spec, layer=1, tracked_lens=[5])
        for i in (0, 3, 4):
            assert torch.equal(new_h[rows[i]], h0[rows[i]]), f"slot {i}"
        assert torch.equal(
            new_h[rows[1]],
            _expected_rows(h0, r0, rows[1], both, 1,
                           la.HookContext(phase=la.PHASE_DECODE,
                                          positions=(9,))))
        assert not torch.equal(new_h[rows[1]], h0[rows[1]])
        assert torch.equal(
            new_h[rows[2]],
            _expected_rows(h0, r0, rows[2], alpha, 1, _prefill_ctx(5)))
        steering.finish_run()

    def test_all_inert_mixed_forward_returns_none(self):
        """A mixed forward of scale-0 legs and unhooked requests only:
        the hook returns None — the output OBJECT passes through, so
        bit-identity holds by construction (the smoke's scale0_noop
        claim, in its mixed-batch form)."""
        noop = _obj("arm__scale0", [_last_op(layer=1)], scale=0.0)
        steering, view = _make_steering(_registry(noop))
        qsl, seq_lens, nat, pos = _forward([(4, 0), (1, 5), (3, 0)])
        view.set_batch(
            ["req-z1", "req-z2", "req-n"],
            {"req-z1": _sel(noop, scale=0.0),
             "req-z2": _sel(noop, scale=0.0)},
            {"req-z1": list(range(4)), "req-z2": list(range(6)),
             "req-n": list(range(3))},
            (qsl, seq_lens, nat))
        h, r = torch.randn(nat, HIDDEN), torch.randn(nat, HIDDEN)
        h0, r0 = h.clone(), r.clone()
        assert steering.hook_body(1, (pos, h, r), (h, r)) is None
        assert torch.equal(h, h0) and torch.equal(r, r0)

    def test_write_follows_req_ids_across_reordering(self):
        """vLLM's persistent batch swap-removes finished requests, so
        req_ids order changes between forwards. The wiring re-pairs
        rows to requests from the LIVE req_ids on every forward: the
        steered request's one-row edit must follow it to its new
        segment index, with a bit-identical delta."""
        both = _obj("both-arm", [_last_op(layer=1)])
        steering, view = _make_steering(_registry(both))
        steering.begin_run([])  # decode-only forwards draw no credits
        row_val = torch.full((HIDDEN,), 0.25)

        def decode_forward(order, steered_idx, computed):
            segs = [(1, c) for c in computed]
            qsl, seq_lens, nat, pos = _forward(segs)
            tokens = {rid: list(range(c + 1))
                      for rid, c in zip(order, computed)}
            view.set_batch(order,
                           {order[steered_idx]: _sel(both)},
                           tokens, (qsl, seq_lens, nat),
                           prompt_lens={rid: c for rid, c
                                        in zip(order, computed)})
            h = torch.randn(nat, HIDDEN)
            h[steered_idx] = row_val
            r = torch.zeros(nat, HIDDEN)
            h0 = h.clone()
            out = steering.hook_body(1, (pos, h, r), (h, r))
            assert out is not None
            new_h = out[0]
            for i in range(nat):
                if i != steered_idx:
                    assert torch.equal(new_h[i], h0[i]), f"row {i}"
            return new_h[steered_idx] - h0[steered_idx]

        # forward 1: [A, B*, C]; forward 2 (A finished, swapped): [B*, C]
        d1 = decode_forward(["req-a", "req-b", "req-c"], 1, [5, 7, 3])
        d2 = decode_forward(["req-b", "req-c"], 0, [8, 4])
        assert torch.equal(d1, d2)
        assert not torch.equal(d1, torch.zeros_like(d1))
        steering.finish_run()


# --------------------------------------------------------------------------
# Part H — the selection record (DIRECT selection evidence)
# --------------------------------------------------------------------------
#
# `selection_honored` in the serving gate is verified by comparing OUTPUT
# TOKEN IDS between a request's mixed-batch serving and its same-size
# reference serving. That is an INFERENCE: the right object must have
# been loaded because the output looks like that arm's output. It is
# sound only while outputs are composition-stable — which is what
# VLLM_BATCH_INVARIANT=1 was buying — so at BI=0 it can fail for reasons
# that say nothing about serving, and even at BI=1 it is only indirect
# evidence that the INTENDED object was the one bound.
#
# Selection is a discrete fact, so `SelectionRecord` observes it: at the
# apply site, per request per steered layer, the cell id / expect_hash /
# effective_scale of the object the worker actually bound, plus whether
# applying it changed that request's rows. These tests pin the two
# claims that matter — that it NAMES the resolved cell per request in a
# mixed batch, and that a deliberately mis-routed request IS caught by
# the same assertion that passes on a clean one — plus the properties
# that make it safe to leave switched on in a serving worker (bounded,
# resettable, and structurally unable to raise).


def _drive_forward(steering, view, spec, layer, *, seed=7, tracked_lens=None):
    """One flattened forward over ``spec`` ((req_id, (q, computed),
    selector-or-None) in batch order) against an already-installed
    steering instance."""
    if tracked_lens is not None:
        steering.begin_run(tracked_lens)
    qsl, seq_lens, nat, pos = _forward([shape for _rid, shape, _s in spec])
    tokens = {rid: list(range(1000 * i, 1000 * i + c + q))
              for i, (rid, (q, c), _s) in enumerate(spec)}
    view.set_batch([rid for rid, _shape, _s in spec],
                   {rid: sel for rid, _shape, sel in spec if sel is not None},
                   tokens, (qsl, seq_lens, nat))
    gen = torch.Generator().manual_seed(seed)
    h = torch.randn(nat, HIDDEN, generator=gen)
    r = torch.randn(nat, HIDDEN, generator=gen)
    return steering.hook_body(layer, (pos, h, r), (h, r))


def _smoke_shaped_registry():
    """The GPU smoke's own shape: a prefill-only vector arm at layer 1
    and a last-row matmul arm at layer 2, each beside its scale-0 twin."""
    static = _obj("static-arm", [_prefill_only_op(layer=1)])
    static0 = _obj("static-arm__scale0", [_prefill_only_op(layer=1)],
                   scale=0.0)
    alpha = _obj("alpha-arm", [_alpha_op(layer=2, coeff=-0.3)])
    alpha0 = _obj("alpha-arm__scale0", [_alpha_op(layer=2, coeff=-0.3)],
                  scale=0.0)
    objs = {o.cell_id: o for o in (static, static0, alpha, alpha0)}
    spec = [("req-n0", (5, 0), None),
            ("req-s1", (4, 0), _sel(static)),
            ("req-z1", (4, 0), _sel(static0, scale=0.0)),
            ("req-a1", (5, 0), _sel(alpha)),
            ("req-z2", (6, 0), _sel(alpha0, scale=0.0)),
            ("req-n1", (3, 0), None)]
    #: what each row ASKED for — the intent the record is checked against.
    intended = {"req-s1": "static-arm", "req-z1": "static-arm__scale0",
                "req-a1": "alpha-arm", "req-z2": "alpha-arm__scale0"}
    return objs, spec, intended, [4, 5]   # tracked: the scale-1 prompt lens


def _run_smoke_shaped(registry, spec, tracked):
    """Both steered layers of one mixed forward, on ONE instance, so the
    record accumulates exactly as it does in a real forward pass."""
    steering, view = _make_steering(registry)
    _drive_forward(steering, view, spec, 1, tracked_lens=tracked)
    _drive_forward(steering, view, spec, 2, seed=8)
    steering.finish_run()
    return steering


def _assert_every_request_got_what_it_asked_for(record, intended):
    """THE ASSERTION. Used unchanged by the clean case and the
    mis-routed case, so the mis-routed case demonstrates that this exact
    check catches it — the record is not being graded on a weaker bar
    than the one it exists to enforce."""
    snap = record.as_dict()
    assert snap["n_observations"] > 0, "the record observed nothing at all"
    for req_id, want in intended.items():
        entry = snap["requests"].get(req_id)
        assert entry is not None, f"{req_id} was never observed"
        assert entry["applied_cell"] == want, (
            f"{req_id} asked for {want!r} but the worker bound "
            f"{entry['applied_cell']!r}")
        assert entry["matched"] is True
        assert entry["stable"] is True
    assert snap["n_mismatched"] == 0
    assert snap["all_observations_matched"] is True


class TestSelectionRecord:
    def test_names_the_resolved_cell_per_request_in_a_mixed_batch(self):
        """The headline: after a mixed forward, the record says which
        object EACH request was bound to — by cell id, by expect_hash and
        by effective_scale — and it is right."""
        objs, spec, intended, tracked = _smoke_shaped_registry()
        steering = _run_smoke_shaped(_registry(*objs.values()), spec, tracked)
        _assert_every_request_got_what_it_asked_for(
            steering.selection_record, intended)

        snap = steering.selection_record.as_dict()
        # the identity recorded is the BOUND object's own, not an echo of
        # the request: hash and scale come off the LoadedObject.
        for req_id, want in intended.items():
            e = snap["requests"][req_id]
            assert e["applied_expect_hash"] == objs[want].expect_hash
            assert e["applied_effective_scale"] == objs[want].effective_scale
            # every selected request is visited at BOTH steered layers.
            assert e["n_applications"] == 2
            assert e["layers"] == [1, 2]
        # the pair table is the request-id-INDEPENDENT view of the same
        # fact — usable when a caller cannot correlate its rows to the
        # worker's request ids.
        assert snap["pairs"] == {c: {c: 2} for c in intended.values()}

    def test_scale0_twin_is_distinguishable_from_its_live_arm(self):
        """A twin and its arm differ ONLY in effective_scale, so that
        field plus 'did applying it change anything' is the cheapest
        check that one was not served in place of the other."""
        objs, spec, _intended, tracked = _smoke_shaped_registry()
        steering = _run_smoke_shaped(_registry(*objs.values()), spec, tracked)
        req = steering.selection_record.as_dict()["requests"]

        for live, twin in (("req-s1", "req-z1"), ("req-a1", "req-z2")):
            assert req[live]["applied_effective_scale"] == 1.0
            assert req[twin]["applied_effective_scale"] == 0.0
            # the live arm has exactly one op layer, so exactly one of
            # its two applications edits rows; the twin edits none.
            assert req[live]["n_applications_modifying"] == 1
            assert req[twin]["n_applications_modifying"] == 0

    def test_unselected_requests_are_never_recorded(self):
        """A row that carries no selector must resolve NOTHING — the
        record's silence about it is what makes 'intended: none' a
        checkable claim rather than an absence the harness has to guess
        at."""
        objs, spec, intended, tracked = _smoke_shaped_registry()
        steering = _run_smoke_shaped(_registry(*objs.values()), spec, tracked)
        recorded = set(steering.selection_record.as_dict()["requests"])
        assert recorded == set(intended)
        assert "req-n0" not in recorded and "req-n1" not in recorded

    def test_a_misrouted_request_is_caught(self):
        """THE NEGATIVE CASE. One cell id is deliberately re-pointed at a
        different object, so the wrong object's ops are genuinely applied
        — the output-comparison half cannot tell, because the wrong
        object DOES produce different-looking text, which is exactly what
        that half is looking for. The record must catch it, under the
        SAME assertion the clean case passes."""
        objs, spec, intended, tracked = _smoke_shaped_registry()

        class MisroutingRegistry(ls.WorkerObjectRegistry):
            """A routing bug, injected: 'alpha-arm' resolves to the
            static arm. Both ``build_request_slots`` and
            ``registry.apply`` go through ``select``, so this is a real
            mis-route and not a record-only fiction."""

            def select(self, petri_object, expect_hash, scale=None):
                if petri_object == "alpha-arm":
                    wrong = self._objects["static-arm"]
                    return super().select(wrong.cell_id, wrong.expect_hash,
                                          None)
                return super().select(petri_object, expect_hash, scale)

        bad = MisroutingRegistry({o.cell_id: o for o in objs.values()})
        steering = _run_smoke_shaped(bad, spec, tracked)
        record = steering.selection_record

        with pytest.raises(AssertionError, match=r"req-a1 asked for"):
            _assert_every_request_got_what_it_asked_for(record, intended)

        snap = record.as_dict()
        assert snap["n_mismatched"] == 2          # one per steered layer
        assert snap["all_observations_matched"] is False
        assert snap["requests"]["req-a1"]["applied_cell"] == "static-arm"
        assert snap["requests"]["req-a1"]["matched"] is False
        # the mismatch list names the request, what it asked for and what
        # it got — enough to act on without re-running.
        m = [x for x in snap["mismatches"] if x["req_id"] == "req-a1"]
        assert m and m[0]["requested"] == "alpha-arm"
        assert m[0]["applied_cell"] == "static-arm"
        # and the rows that were NOT mis-routed still read clean, so the
        # finding is localized rather than smearing over the batch.
        assert snap["requests"]["req-s1"]["matched"] is True

    def test_a_resolution_that_changes_mid_generation_is_flagged(self):
        """One request, two forwards, same selector: the object bound
        must not change. A record entry that saw two different bindings
        is unstable, and says so."""
        arm = _obj("arm", [_last_op(layer=1)])
        other = _obj("other", [_last_op(layer=1)])
        steering, view = _make_steering(_registry(arm, other))
        spec = [("req-x", (3, 0), _sel(arm))]
        _drive_forward(steering, view, spec, 1, tracked_lens=[3])
        assert steering.selection_record.as_dict()["requests"][
            "req-x"]["stable"] is True

        steering.selection_record.observe(
            req_id="req-x", layer=1, requested="arm",
            requested_hash=arm.expect_hash, requested_scale=None,
            applied=other, modified=True)
        snap = steering.selection_record.as_dict()
        assert snap["requests"]["req-x"]["stable"] is False
        assert snap["n_unstable_requests"] == 1
        steering.finish_run()

    def test_record_is_bounded_and_resettable(self):
        """It must not grow without bound in a long run: per-request
        detail stops at the cap and says how many it declined, while the
        aggregates keep counting. Reset scopes the next read to one
        submitted batch."""
        arm = _obj("arm", [_row_local_op(layer=1)])
        rec = lw.SelectionRecord(max_requests=3, max_pair_keys=1,
                                 max_mismatches=1)
        for i in range(10):
            rec.observe(req_id=f"req-{i}", layer=1, requested="arm",
                        requested_hash=arm.expect_hash, requested_scale=None,
                        applied=arm, modified=True)
        snap = rec.as_dict()
        assert snap["n_observations"] == 10          # aggregates keep going
        assert snap["n_requests_recorded"] == 3      # detail is capped
        assert snap["n_requests_dropped"] == 7
        assert snap["max_requests"] == 3

        rec.reset()
        empty = rec.as_dict()
        assert empty["n_observations"] == 0 and empty["requests"] == {}
        assert empty["all_observations_matched"] is False   # nothing seen

    def test_observing_can_never_raise_into_the_apply_path(self):
        """An instrument that can condemn the worker it instruments is
        worse than no instrument. A broken object counts an error and
        the apply path continues."""
        class Exploding:
            @property
            def cell_id(self):
                raise RuntimeError("boom")

        rec = lw.SelectionRecord()
        rec.observe(req_id="req-x", layer=1, requested="arm",
                    requested_hash="h", requested_scale=None,
                    applied=Exploding(), modified=True)
        snap = rec.as_dict()
        assert snap["n_record_errors"] == 1
        assert "boom" in snap["first_record_error"]
        assert snap["requests"] == {}

    def test_snapshot_is_json_able_and_states_its_own_coverage(self):
        """The snapshot crosses a collective_rpc boundary and lands in a
        JSON record, and it must carry the statement of what it does NOT
        cover with it — a diagnostic that implies binding coverage it
        lacks is the same defect as the output-inference it replaces."""
        import json

        objs, spec, _intended, tracked = _smoke_shaped_registry()
        steering = _run_smoke_shaped(_registry(*objs.values()), spec, tracked)
        snap = steering.selection_record.as_dict()
        json.dumps(snap)   # plain scalars only; no tensors, no objects
        assert "DOES NOT OBSERVE" in snap["coverage"]
        assert "OP-LEVEL BINDING" in snap["coverage"]
        assert snap["coverage"] is lw.SELECTION_RECORD_COVERAGE

    def test_the_instrument_does_not_change_what_is_applied(self):
        """The whole premise: switching the record off must not change a
        single written value. Driven twice — once with the live record,
        once with a record whose observe() is a no-op — the hidden states
        must be bit-identical."""
        objs, spec, _intended, tracked = _smoke_shaped_registry()

        def run(neuter):
            steering, view = _make_steering(_registry(*objs.values()))
            if neuter:
                steering.selection_record.observe = (
                    lambda **kw: None)   # type: ignore[method-assign]
            out = _drive_forward(steering, view, spec, 1,
                                 tracked_lens=tracked)
            steering.finish_run()
            return out[0]

        assert torch.equal(run(False), run(True))


# --------------------------------------------------------------------------
# Part H — per-request adapter failures are contained (the 2026-08-30
# EngineCore incident): an AdapterError attributable to ONE request's
# apply must fail THAT request loudly, never the engine.
# --------------------------------------------------------------------------


def _two_prefills(view, arm, *, ids_a, ids_b):
    """Two selector-carrying complete prefills: req-a with ids_a,
    req-b with ids_b."""
    qsl, seq_lens, nat, pos = _forward([(len(ids_a), 0), (len(ids_b), 0)])
    view.set_batch(
        ["req-a", "req-b"],
        {"req-a": _sel(arm), "req-b": _sel(arm)},
        {"req-a": list(ids_a), "req-b": list(ids_b)},
        (qsl, seq_lens, nat))
    return qsl, seq_lens, nat, pos


class TestPerRequestFailureContainment:

    #: a prompt in which the assistant header occurs nowhere — the
    #: locator refuses it (per request), which used to kill the engine.
    BAD = tuple(range(1, 7))
    #: an ordinary render: content then the trailing header.
    GOOD = (10, 11) + HEADER

    def _steering(self):
        arm = _obj("arm-r", [_response_gated_op(layer=1)])
        return _make_steering(_registry(arm, header=HEADER))[0], arm

    def test_one_bad_request_fails_alone_and_the_batch_serves(self):
        steering, arm = self._steering()
        _, _, nat, pos = _two_prefills(steering._view, arm,
                                       ids_a=self.BAD, ids_b=self.GOOD)
        h = torch.randn(nat, HIDDEN)
        h0 = h.clone()
        out = steering.hook_body(1, (pos, h), h)
        assert torch.is_tensor(out)
        a, b = slice(0, len(self.BAD)), slice(len(self.BAD), nat)
        # the failed request's rows are UNSTEERED (its response will be
        # refused at the proxy — never returned as a normal completion)
        assert torch.equal(out[a], h0[a])
        # the healthy request is served: header + generated region steered
        assert not torch.equal(out[b], h0[b])
        # the worker is NOT condemned — the next forward serves fine
        failure = steering.take_request_failure("req-a")
        assert failure is not None
        assert "req-a" in failure["req_id"]
        assert "assistant header" in failure["error"]
        assert failure["cell_id"] == "arm-r"
        assert steering.take_request_failure("req-b") is None

    def test_the_failed_request_is_skipped_not_retried(self):
        steering, arm = self._steering()
        _, _, nat, pos = _two_prefills(steering._view, arm,
                                       ids_a=self.BAD, ids_b=self.GOOD)
        h = torch.randn(nat, HIDDEN)
        steering.hook_body(1, (pos, h), h)
        # a later decode forward for the same request: skipped silently,
        # still unsteered, no engine involvement
        qsl, seq_lens, nat2, pos2 = _forward([(1, len(self.BAD))])
        steering._view.set_batch(
            ["req-a"], {"req-a": _sel(arm)},
            {"req-a": list(self.BAD) + [77]}, (qsl, seq_lens, nat2),
            prompt_lens={"req-a": len(self.BAD)})
        h2 = torch.randn(nat2, HIDDEN)
        h20 = h2.clone()
        out2 = steering.hook_body(1, (pos2, h2), h2)
        assert out2 is None or torch.equal(out2, h20)
        # exactly ONE failure record, and popping it empties the ledger
        assert steering.take_request_failure("req-a") is not None
        assert steering.take_request_failure("req-a") is None

    def test_the_worker_is_not_condemned_by_a_request_failure(self):
        steering, arm = self._steering()
        _, _, nat, pos = _two_prefills(steering._view, arm,
                                       ids_a=self.BAD, ids_b=self.GOOD)
        h = torch.randn(nat, HIDDEN)
        steering.hook_body(1, (pos, h), h)
        # a fresh forward with only healthy requests serves normally
        qsl, seq_lens, nat2, pos2 = _forward([(len(self.GOOD), 0)])
        steering._view.set_batch(
            ["req-c"], {"req-c": _sel(arm)},
            {"req-c": list(self.GOOD)}, (qsl, seq_lens, nat2))
        h2 = torch.randn(nat2, HIDDEN)
        out2 = steering.hook_body(1, (pos2, h2), h2)
        assert torch.is_tensor(out2)

    def test_non_request_failures_still_condemn_and_raise(self):
        """Only per-slot adapter errors are contained; forward-level
        geometry/metadata breakage stays engine-fatal by design."""
        arm = _obj("arm", [_row_local_op(layer=1)])
        steering, view = _make_steering(_registry(arm))
        view.set_batch(["req-s"], {"req-s": _sel(arm)},
                       {"req-s": [1, 2, 3]}, None)  # metadata missing
        with pytest.raises(lw.LensWorkerError, match="no attention"):
            steering.hook_body(1, (), (torch.ones(3, HIDDEN),
                                       torch.ones(3, HIDDEN)))
        with pytest.raises(lw.LensWorkerError, match="CONDEMNED"):
            steering.hook_body(1, (), (torch.ones(3, HIDDEN),
                                       torch.ones(3, HIDDEN)))

    def test_finish_run_refuses_on_unclaimed_request_failures(self):
        """The in-proc harness's loudness: a run that poisoned a request
        must not finish clean while the failure sits unread."""
        steering, arm = self._steering()
        steering.begin_run([len(self.BAD), len(self.GOOD)])
        _, _, nat, pos = _two_prefills(steering._view, arm,
                                       ids_a=self.BAD, ids_b=self.GOOD)
        h = torch.randn(nat, HIDDEN)
        steering.hook_body(1, (pos, h), h)
        with pytest.raises(lw.LensWorkerError, match="request failure"):
            steering.finish_run()

    def test_the_failure_ledger_is_bounded_and_overflow_condemns(self):
        steering, arm = self._steering()
        steering._request_failures_cap = 2
        for i, rid in enumerate(("r0", "r1")):
            qsl, seq_lens, nat, pos = _forward([(len(self.BAD), 0)])
            steering._view.set_batch(
                [rid], {rid: _sel(arm)},
                {rid: list(self.BAD)}, (qsl, seq_lens, nat))
            steering.hook_body(1, (pos, torch.randn(nat, HIDDEN)),
                               torch.randn(nat, HIDDEN))
        qsl, seq_lens, nat, pos = _forward([(len(self.BAD), 0)])
        steering._view.set_batch(
            ["r2"], {"r2": _sel(arm)},
            {"r2": list(self.BAD)}, (qsl, seq_lens, nat))
        with pytest.raises(lw.LensWorkerError, match="failure ledger"):
            steering.hook_body(1, (pos, torch.randn(nat, HIDDEN)),
                               torch.randn(nat, HIDDEN))

    def test_child_request_ids_are_claimable_by_the_parent_id(self):
        """vLLM parallel sampling suffixes engine child ids; the proxy
        holds the parent (response) id and must still be able to claim
        the failure."""
        steering, arm = self._steering()
        qsl, seq_lens, nat, pos = _forward([(len(self.BAD), 0)])
        steering._view.set_batch(
            ["chatcmpl-abc_0"], {"chatcmpl-abc_0": _sel(arm)},
            {"chatcmpl-abc_0": list(self.BAD)}, (qsl, seq_lens, nat))
        steering.hook_body(1, (pos, torch.randn(nat, HIDDEN)),
                           torch.randn(nat, HIDDEN))
        failure = steering.take_request_failure("chatcmpl-abc")
        assert failure is not None and failure["req_id"] == "chatcmpl-abc_0"


# --------------------------------------------------------------------------
# The install report's serve declarations, the preflight's ON-header gate,
# and the SelectionRecord's per-request facts + masked-row count (the rest
# of the thinking-ON path is tests/test_thinking_on_serving.py)
# --------------------------------------------------------------------------


class TestServeDeclarationsInWorker:
    def test_install_report_carries_an_empty_declarations_map_for_undeclared_cells(self):
        steering, _ = _make_steering(_registry(_obj("a", [_row_local_op()]),
                                               _obj("b", [_response_gated_op()], scale=0.0),
                                               header=HEADER))
        report = steering._report
        assert report.lens_serve_declarations == {}
        payload = lw.install_report_payload(report)
        assert payload[lw.LENS_SERVE_DECLARATIONS_KEY] == {}
        # ... and the strict reader reads it back as an attested empty map
        assert lw.serve_declarations_attested({"worker_reports": [payload]}) == ({}, None)

    def test_install_report_carries_the_declared_cells_under_their_own_ids(self):
        decl = {"serve_out_of_support": ["thinking"]}
        on = (900, 901, 902, 909)
        arm = ls.LoadedObject(cell_id="si_safety__psr_kl_all@thinking", kind="psr", expect_hash="h1",
                              ops=(la.HookOp(op=la.OP_RELU_GATED_VECTOR, layer=1,
                                             vector=torch.ones(HIDDEN), gate_w=torch.ones(HIDDEN),
                                             gate_b=torch.zeros(1), positions=la.POSITIONS_RESPONSE,
                                             header_ids=HEADER, thinking_header_ids=on),),
                              effective_scale=1.0, serve_declarations=decl)
        twin = ls.LoadedObject(cell_id="none__lens0", kind="psr", expect_hash="h0", ops=arm.ops,
                               effective_scale=0.0, serve_declarations=decl)
        reg = ls.WorkerObjectRegistry({o.cell_id: o for o in (arm, twin)},
                                      assistant_header_ids=HEADER,
                                      assistant_header_ids_thinking_on=on)
        assert reg.serves_thinking_on()
        seen = []
        steering = lw.LensWorkerSteering(view=FakeView(layers=[FakeLayer() for _ in range(3)]),
                                         registry=reg)
        steering.preflight(object(), header_verifier=_ok_header,
                           thinking_header_verifier=lambda tok, rec: seen.append(rec))
        assert seen == [on]                      # gated on the LOADED ops, handed the recorded header
        report = steering.install()
        assert report.lens_serve_declarations == {arm.cell_id: decl, "none__lens0": decl}
        assert lw.install_report_payload(report)[lw.LENS_SERVE_DECLARATIONS_KEY] == {
            "none__lens0": decl, arm.cell_id: decl}

    def test_preflight_never_calls_the_on_verifier_for_an_ordinary_registry(self):
        calls = []
        reg = _registry(_obj("a", [_response_gated_op()]), header=HEADER)
        steering = lw.LensWorkerSteering(view=FakeView(layers=[FakeLayer() for _ in range(3)]),
                                         registry=reg)
        steering.preflight(object(), header_verifier=_ok_header,
                           thinking_header_verifier=lambda tok, rec: calls.append(rec))
        assert calls == [] and lw.DEFAULT_THINKING_HEADER_VERIFIER is ls.verify_assistant_header_thinking_on

    def test_selection_record_defaults_without_a_context(self):
        rec = lw.SelectionRecord()
        rec.observe(req_id="r", layer=1, requested="a", requested_hash="h", requested_scale=None,
                    applied=_obj("a", [_row_local_op()]), modified=True)
        entry = rec.as_dict()["requests"]["r"]
        assert entry["facts"] == {"response_prefill": False, "agent_loop": False, "thinking_on": False}
        assert entry["prefill_masked_rows"] is None
        # a decode-group context contributes the two non-tail facts only
        rec.observe(req_id="r2", layer=1, requested="a", requested_hash="h", requested_scale=None,
                    applied=_obj("a", [_row_local_op()]), modified=True,
                    ctx=la.DecodeRows(agent_loop=True))
        assert rec.as_dict()["requests"]["r2"]["facts"] == {"response_prefill": False,
                                                            "agent_loop": True, "thinking_on": False}
        assert rec.n_record_errors == 0


# --------------------------------------------------------------------------
# Part I — the record's diagnostics on the ROW-LOCAL path (code-review
# finding A1, fit-support-transfer): ``facts`` and ``prefill_masked_rows``
# come from the request's own selector and its own prefill, never from
# whether a positional plan happened to be built — so a request's record
# reads the same alone (ctx None) and co-batched with a plan-needing request.
# --------------------------------------------------------------------------

#: the declared ON header: shares HEADER's first three ids, as the real
#: Qwen OFF/ON pair shares its prefix, and differs in width
ON_HEADER = (900, 901, 902, 909)


def _two_header_resp_op(layer=1):
    """The Qwen-shaped response op: the OFF header plus a declared ON header."""
    return la.HookOp(op=la.OP_ADD_VECTOR, layer=layer, vector=torch.ones(HIDDEN),
                     positions=la.POSITIONS_RESPONSE, header_ids=HEADER,
                     thinking_header_ids=ON_HEADER)


def _declared_registry(*objs):
    return ls.WorkerObjectRegistry({o.cell_id: o for o in objs},
                                   assistant_header_ids=HEADER,
                                   assistant_header_ids_thinking_on=ON_HEADER)


def _make_steering_declared(registry, n_layers=3, *, prefix_caching=False):
    view = FakeView(layers=[FakeLayer() for _ in range(n_layers)],
                    prefix_caching=prefix_caching)
    steering = lw.LensWorkerSteering(view=view, registry=registry)
    steering.preflight(object(), header_verifier=_ok_header,
                       thinking_header_verifier=lambda tok, rec: None)
    steering.install()
    return steering, view


def _record_diagnostics(record):
    """(facts, prefill_masked_rows) per recorded request — the two fields A1 is about."""
    return {rid: (e["facts"], e["prefill_masked_rows"])
            for rid, e in record.as_dict()["requests"].items()}


NO_FACTS = {"response_prefill": False, "agent_loop": False, "thinking_on": False}
ON_FACTS = {"response_prefill": False, "agent_loop": False, "thinking_on": True}


class TestFastPathDiagnostics:
    """Finding A1: ``build_request_slots`` returns ``ctx=None`` when no active
    object needs a positional plan — the experiment's ``@all_positions`` cell
    and a scale-0 ``none__lens0`` twin — and the record read both ``facts``
    and ``prefill_masked_rows`` off the HookContext alone. So an isolated
    all-positions prefill filed ``None`` for a mask covering every row, a
    declared twin filed ``thinking_on=False`` against its own selector, and
    both values CHANGED when a positional request shared the forward. Every
    test here runs the REAL slot builder and the REAL hook body; the earlier
    record tests handed a HookContext to the instrument directly, which is
    exactly how they missed it."""

    ALL_IDS = (10, 11, 12, 13, 14)           # a 5-row prompt for the all-positions cell
    TWIN_ON_IDS = (1, 2, 3, *ON_HEADER)      # the declared ON render, ending at the ON header
    TWIN_OFF_IDS = (9, 8, 7, *HEADER)        # the OFF render, ending at the OFF header

    def _objs(self):
        allpos = _obj("si_safety__psr_kl_all@all_positions", [_row_local_op(layer=1)])
        base = _obj("si_safety__psr_kl_all", [_two_header_resp_op(layer=1)])
        twin = ls.LoadedObject(cell_id="none__lens0", kind="synthetic_vector",
                               expect_hash="hash-none__lens0", ops=base.ops,
                               effective_scale=0.0)             # the donor's ops, scale 0
        last = _obj("arm-last", [_last_op(layer=1)])            # a plan-needing neighbour
        return allpos, base, twin, last

    @staticmethod
    def _twin_sel(twin, *, on):
        sel = _sel(twin, scale=0.0)
        if on:
            sel["thinking_on"] = 1
        return sel

    @staticmethod
    def _drive(steering, view, spec, *, layer=1):
        """One forward over ``spec`` ((req_id, ids, selector-or-None) in batch
        order), every request a COMPLETE cold prefill. Returns (hook output,
        hidden, residual)."""
        qsl, seq_lens, nat, pos = _forward([(len(ids), 0) for _rid, ids, _s in spec])
        view.set_batch([rid for rid, _ids, _s in spec],
                       {rid: sel for rid, _ids, sel in spec if sel is not None},
                       {rid: list(ids) for rid, ids, _s in spec}, (qsl, seq_lens, nat))
        gen = torch.Generator().manual_seed(11)
        h = torch.randn(nat, HIDDEN, generator=gen)
        r = torch.randn(nat, HIDDEN, generator=gen)
        return steering.hook_body(layer, (pos, h, r), (h, r)), h, r

    @pytest.mark.parametrize("prefix_caching", [False, True])
    def test_isolated_all_positions_prefill_files_every_prompt_row(self, prefix_caching):
        allpos, base, twin, last = self._objs()
        steering, view = _make_steering_declared(_declared_registry(allpos, base, twin, last),
                                                 prefix_caching=prefix_caching)
        out, _h, _r = self._drive(steering, view, [("req-all", self.ALL_IDS, _sel(allpos))])
        assert out is not None                                  # scale 1: it steered
        entry = steering.selection_record.as_dict()["requests"]["req-all"]
        assert entry["prefill_masked_rows"] == len(self.ALL_IDS)  # P: positions=all, no BOS skip
        assert entry["facts"] == NO_FACTS
        assert entry["n_applications_modifying"] >= 1
        assert steering.selection_record.n_record_errors == 0

    @pytest.mark.parametrize("prefix_caching", [False, True])
    def test_isolated_declared_twin_files_its_selector_fact_and_the_on_width(self, prefix_caching):
        allpos, base, twin, last = self._objs()
        steering, view = _make_steering_declared(_declared_registry(allpos, base, twin, last),
                                                 prefix_caching=prefix_caching)
        out, _h, _r = self._drive(
            steering, view,
            [("req-twin-on", self.TWIN_ON_IDS, self._twin_sel(twin, on=True)),
             ("req-twin-off", self.TWIN_OFF_IDS, self._twin_sel(twin, on=False))])
        assert out is None                                      # scale 0: a bitwise passthrough
        d = steering.selection_record.as_dict()["requests"]
        assert d["req-twin-on"]["facts"] == ON_FACTS
        assert d["req-twin-on"]["prefill_masked_rows"] == len(ON_HEADER)   # the ON header's width
        assert d["req-twin-off"]["facts"] == NO_FACTS
        assert d["req-twin-off"]["prefill_masked_rows"] == len(HEADER)     # the OFF header's width
        assert all(e["n_applications_modifying"] == 0 for e in d.values())
        assert steering.selection_record.n_record_errors == 0

    @pytest.mark.parametrize("prefix_caching", [False, True])
    def test_record_is_identical_isolated_and_co_batched_with_a_positional_request(self, prefix_caching):
        """The reviewer's reproduction: the same request, alone (row-local
        path) and beside a plan-needing request (plan path), must file the
        same facts and the same masked-row count."""
        allpos, base, twin, last = self._objs()
        reqs = [("req-all", self.ALL_IDS, _sel(allpos)),
                ("req-twin-on", self.TWIN_ON_IDS, self._twin_sel(twin, on=True)),
                ("req-twin-off", self.TWIN_OFF_IDS, self._twin_sel(twin, on=False))]

        def run(spec):
            steering, view = _make_steering_declared(_declared_registry(allpos, base, twin, last),
                                                     prefix_caching=prefix_caching)
            self._drive(steering, view, spec)
            assert steering.selection_record.n_record_errors == 0
            return _record_diagnostics(steering.selection_record)

        alone = {}
        for req in reqs:
            alone.update(run([req]))                            # each alone: ctx None
        mixed = run(reqs + [("req-last", [50, 51, 52, 53], _sel(last))])   # the plan path
        assert alone["req-all"] == mixed["req-all"] == (NO_FACTS, len(self.ALL_IDS))
        assert alone["req-twin-on"] == mixed["req-twin-on"] == (ON_FACTS, len(ON_HEADER))
        assert alone["req-twin-off"] == mixed["req-twin-off"] == (NO_FACTS, len(HEADER))
        assert mixed["req-last"] == (NO_FACTS, 1)               # the neighbour: its last row

    def test_slot_builder_carries_the_diagnostics_on_both_paths(self):
        """At the builder: the row-local path keeps ``ctx=None`` (no plan, no
        positions read) and carries the diagnostics beside it; the plan path
        carries the same values next to its contexts."""
        allpos, base, twin, last = self._objs()
        reg = _declared_registry(allpos, base, twin, last)
        toks = {"req-all": self.ALL_IDS, "req-twin": self.TWIN_ON_IDS, "req-last": [50, 51, 52, 53]}
        sels = {"req-all": _sel(allpos), "req-twin": self._twin_sel(twin, on=True),
                "req-last": _sel(last)}

        def build(rids):
            qsl, seq_lens, nat, pos = _forward([(len(toks[r]), 0) for r in rids])
            return lw.build_request_slots(
                query_start_loc=qsl, seq_lens=seq_lens, num_actual_tokens=nat,
                positions=pos, req_ids=tuple(rids), raw_selectors=tuple(sels[r] for r in rids),
                token_lookup=lambda rid: toks[rid], prompt_len_lookup=lambda rid: len(toks[rid]),
                registry=reg)

        alone = build(["req-all", "req-twin"])
        assert alone.plan is None and all(s.ctx is None for s in alone.slots)
        mixed = build(["req-all", "req-twin", "req-last"])
        assert mixed.plan is not None and all(s.ctx is not None for s in mixed.slots)
        for a, m in zip(alone.slots, mixed.slots):
            assert (a.facts, a.prefill_masked_rows) == (m.facts, m.prefill_masked_rows)
        by_id = {s.req_id: s for s in alone.slots}
        assert (by_id["req-all"].facts, by_id["req-all"].prefill_masked_rows) == (NO_FACTS, len(self.ALL_IDS))
        assert (by_id["req-twin"].facts, by_id["req-twin"].prefill_masked_rows) == (ON_FACTS, len(ON_HEADER))

    def test_row_local_diagnostics_never_refuse_and_read_no_positions(self):
        """The diagnostic is an instrument: a shape the row-local path
        tolerates keeps being served (files null), a token source behind the
        forward files null instead of refusing, a decode step files null, an
        unhooked request carries nothing — and the forward's positions tensor
        is never read on this path (an unreadable one is fine)."""
        allpos, *_ = self._objs()
        reg = _declared_registry(allpos)
        sel_all = _sel(allpos)

        def build(segs, *, tokens, prompt_len, sel, positions=None):
            qsl, seq_lens, nat, pos = _forward(segs)
            return lw.build_request_slots(
                query_start_loc=qsl, seq_lens=seq_lens, num_actual_tokens=nat,
                positions=(pos if positions is None else positions), req_ids=("req-a",),
                raw_selectors=(sel,), token_lookup=lambda rid: tokens,
                prompt_len_lookup=lambda rid: prompt_len, registry=reg)

        # a chunked-prefill continuation (lattice-refused; tolerated here): served, null
        fwd = build([(4, 3)], tokens=list(range(7)), prompt_len=7, sel=sel_all)
        assert fwd.plan is None and fwd.slots[0].ctx is None
        assert fwd.slots[0].rows == slice(0, 4) and fwd.slots[0].prefill_masked_rows is None
        assert fwd.slots[0].facts == NO_FACTS
        # the token source behind the forward: the plan path refuses ("timing"); here null
        fwd = build([(5, 0)], tokens=[1, 2, 3], prompt_len=5, sel=sel_all)
        assert fwd.slots[0].rows == slice(0, 5) and fwd.slots[0].prefill_masked_rows is None
        # a decode step: no prefill to count
        fwd = build([(1, 5)], tokens=list(range(6)), prompt_len=5, sel=sel_all)
        assert fwd.slots[0].prefill_masked_rows is None
        # a complete prefill with an UNREADABLE positions tensor: the row-local
        # path never reads positions, and the diagnostic must not start to
        fwd = build([(5, 0)], tokens=self.ALL_IDS, prompt_len=5, sel=sel_all,
                    positions=object())
        assert fwd.slots[0].ctx is None and fwd.slots[0].prefill_masked_rows == 5
        # an unhooked request: no selector, no diagnostics
        fwd = build([(5, 0)], tokens=self.ALL_IDS, prompt_len=5, sel=None)
        assert fwd.slots[0].selector is None and fwd.slots[0].facts is None
        assert fwd.slots[0].prefill_masked_rows is None

    def test_fast_path_arithmetic_is_bit_identical_to_the_reference_apply(self):
        """The diagnostics are computed BESIDE the apply, never inside it: the
        all-positions cell's rows equal ``la.apply_hook_ops`` on the true
        stream with NO context (the row-local reference), bit for bit; the
        scale-0 twin's and the unhooked request's rows are untouched; and
        neutering the diagnostic changes no written value."""
        allpos, base, twin, last = self._objs()
        spec = [("req-n", [70, 71, 72], None),
                ("req-all", self.ALL_IDS, _sel(allpos)),
                ("req-twin-on", self.TWIN_ON_IDS, self._twin_sel(twin, on=True))]

        def run(monkeypatch=None):
            steering, view = _make_steering_declared(_declared_registry(allpos, base, twin, last))
            if monkeypatch is not None:
                monkeypatch.setattr(lw, "row_local_prefill_masked_rows",
                                    lambda *a, **k: None)
            out, h, r = self._drive(steering, view, spec)
            return out[0], h, r, steering

        new_h, h, r, steering = run()
        true = h + r
        rows = slice(3, 8)                                      # req-all's rows
        expect = la.apply_hook_ops(true[rows], allpos.ops, 1.0, layer=1)   # ctx None: the reference
        assert not torch.equal(expect, true[rows])
        assert torch.equal(new_h[rows], h[rows] + (expect - true[rows]))
        assert torch.equal(new_h[0:3], h[0:3]) and torch.equal(new_h[8:], h[8:])
        assert steering.selection_record.as_dict()["requests"]["req-all"]["prefill_masked_rows"] == 5
        with pytest.MonkeyPatch.context() as mp:
            neutered, _h2, _r2, steering2 = run(mp)
            assert steering2.selection_record.as_dict()["requests"]["req-all"]["prefill_masked_rows"] is None
        assert torch.equal(new_h, neutered)
