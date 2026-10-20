"""Classify ONCE per forward (F1) and classify WITHOUT a device sync (F2)
— the lens-vectorize landing's classification half — pinned on the CPU
against the worker's view seam and against fakes shaped like vLLM
0.25.1's real runner surface, one per served FAMILY:

- llama (plain): one FlashAttention metadata object, 1-D positions;
- gemma-4 (sliding + full attention): two bearing metadata objects that
  ALIAS one CommonAttentionMetadata's tensors, plus the nested decoder
  accessor (``model.language_model.layers``);
- Qwen3.5 (hybrid gated-deltanet): a bearing group + a GDN group whose
  ``non_spec_query_start_loc`` IS the bearing group's tensor, and (3, T)
  M-RoPE positions;
- Nemotron-3-Nano (hybrid Mamba2): a bearing group + a Mamba2 group that
  carries ``seq_lens`` only (no ``num_actual_tokens``), and the
  ``backbone.layers`` accessor;
- GLM-4.6V-Flash (dense M-RoPE): (3, T) positions from
  ``mrope_positions.cpu``, the ``language_model`` nesting.

For each, the KV-cache-group cross-check must resolve with ZERO
``torch.equal`` calls when the groups alias one buffer (the real 0.25.1
shape — every builder is handed a shallow copy of the same
CommonAttentionMetadata), and must still REFUSE, with the message it
always had, when a group carries a disagreeing copy. The one-time
GPU-VERIFY latches (the runner's CPU mirrors == the device tensors) are
exercised with planted disagreements."""

import sys
import types

import numpy as np
import pytest
import torch

from sisteer.capeval.serving import lens_adapters as la
from sisteer.capeval.serving import lens_batch as lb
from sisteer.capeval.serving import lens_worker as lw
from test_petri_hybrid_metadata import FlashMeta, GDNMeta, _hybrid_dict, _t
from test_petri_lens_worker import (HEADER, HIDDEN, FakeLayer, FakeView,
                                    _forward, _gated_op, _last_op, _obj,
                                    _ok_header, _registry, _row_local_op, _sel)


# --------------------------------------------------------------------------
# a view with a forward key + CPU mirrors (the real view's F1/F2 surface)
# --------------------------------------------------------------------------


class KeyedView(FakeView):
    """FakeView plus the F1/F2 methods: a forward key that is a fresh
    object per set_batch (as the runner's attn_metadata dict is per
    step), CPU-mirror metadata, and CPU positions handed through."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.key = object()
        self.positions_cpu = None
        self.n_positions_calls = 0

    def set_batch(self, *a, **kw):
        super().set_batch(*a, **kw)
        self.key = object()

    def forward_key(self):
        return self.key if self.metadata is not None else None

    def forward_metadata_cpu(self):
        if self.metadata is None:
            return None
        qsl, seq, nat = self.metadata
        return (torch.tensor(qsl, dtype=torch.int32),
                torch.tensor(seq, dtype=torch.int32), nat)

    def forward_positions(self, device_positions, num_tokens):
        self.n_positions_calls += 1
        return (self.positions_cpu if self.positions_cpu is not None
                else device_positions)


def _make(registry, n_layers=32, view_cls=KeyedView, **kw):
    view = view_cls(layers=[FakeLayer() for _ in range(n_layers)])
    st = lw.LensWorkerSteering(view=view, registry=registry, **kw)
    st.preflight(object(), header_verifier=_ok_header)
    st.install()
    return st, view


def _arm32():
    """A plan-needing arm hooked at all 32 layers (the psr_kl_all shape)."""
    return _obj("arm", [_last_op(layer=l) for l in range(32)])


# --------------------------------------------------------------------------
# Part A — F1: one classification per forward
# --------------------------------------------------------------------------


class TestClassifyOncePerForward:
    def _count(self, monkeypatch):
        n = {"slots": 0, "selectors": 0}
        orig = lw.build_request_slots
        orig_sel = lw.extract_selector

        def counting(**kw):
            n["slots"] += 1
            return orig(**kw)

        def counting_sel(x):
            n["selectors"] += 1
            return orig_sel(x)
        monkeypatch.setattr(lw, "build_request_slots", counting)
        monkeypatch.setattr(lw, "extract_selector", counting_sel)
        return n

    def test_32_hooks_one_classification(self, monkeypatch):
        n = self._count(monkeypatch)
        arm = _arm32()
        st, view = _make(_registry(arm))
        qsl, seq_lens, nat, pos = _forward([(1, 5), (1, 7)])
        view.set_batch(["r0", "r1"], {"r0": _sel(arm), "r1": _sel(arm)},
                       {"r0": list(range(6)), "r1": list(range(8))},
                       (qsl, seq_lens, nat), prompt_lens={"r0": 3, "r1": 3})
        h, r = torch.zeros(nat, HIDDEN), torch.zeros(nat, HIDDEN)
        for layer in range(32):
            assert st.hook_body(layer, (pos, h, r), (h, r)) is not None
        assert n["slots"] == 1
        assert n["selectors"] == 2           # once per request, once per forward
        assert view.n_positions_calls == 1
        # a NEW forward (new key object) classifies again, exactly once
        view.set_batch(["r0", "r1"], {"r0": _sel(arm), "r1": _sel(arm)},
                       {"r0": list(range(7)), "r1": list(range(9))},
                       (*_forward([(1, 6), (1, 8)])[:3],),
                       prompt_lens={"r0": 3, "r1": 3})
        pos2 = _forward([(1, 6), (1, 8)])[3]
        for layer in range(32):
            st.hook_body(layer, (pos2, h, r), (h, r))
        assert n["slots"] == 2

    def test_two_forwards_never_alias(self):
        """The cache holds a STRONG reference to the classified forward's
        key, so a later forward's key can never be the same object."""
        arm = _arm32()
        st, view = _make(_registry(arm))
        qsl, seq_lens, nat, pos = _forward([(1, 5)])
        view.set_batch(["r0"], {"r0": _sel(arm)}, {"r0": list(range(6))},
                       (qsl, seq_lens, nat), prompt_lens={"r0": 3})
        h = torch.zeros(nat, HIDDEN)
        st.hook_body(0, (pos, h, h), (h, h))
        key1 = st._fwd_cache[0]
        assert key1 is view.key
        view.set_batch(["r0"], {"r0": _sel(arm)}, {"r0": list(range(7))},
                       (*_forward([(1, 6)])[:3],), prompt_lens={"r0": 3})
        assert view.key is not key1
        st.hook_body(0, (_forward([(1, 6)])[3], h, h), (h, h))
        assert st._fwd_cache[0] is view.key and st._fwd_cache[0] is not key1

    def test_passthrough_forward_is_cached_too(self, monkeypatch):
        n = self._count(monkeypatch)
        arm = _arm32()
        st, view = _make(_registry(arm))
        qsl, seq_lens, nat, pos = _forward([(1, 5)])
        view.set_batch(["u"], {}, {"u": list(range(6))}, (qsl, seq_lens, nat))
        h = torch.zeros(nat, HIDDEN)
        for layer in range(32):
            assert st.hook_body(layer, (pos, h, h), (h, h)) is None
        assert n["selectors"] == 1 and n["slots"] == 0

    def test_request_count_changing_under_a_live_forward_refuses(self):
        arm = _arm32()
        st, view = _make(_registry(arm))
        qsl, seq_lens, nat, pos = _forward([(1, 5)])
        view.set_batch(["r0"], {"r0": _sel(arm)}, {"r0": list(range(6))},
                       (qsl, seq_lens, nat), prompt_lens={"r0": 3})
        h = torch.zeros(nat, HIDDEN)
        st.hook_body(0, (pos, h, h), (h, h))
        view.batch_req_ids = ["r0", "r1"]          # same key, more requests
        with pytest.raises(lw.LensWorkerError, match="changed under a live"):
            st.hook_body(1, (pos, h, h), (h, h))

    def test_refusal_fires_once_with_the_uncached_message(self):
        """A classification refusal (the token source is behind the
        forward) raises from the first hook with EXACTLY the message the
        per-hook classification produced, condemns, and is not cached."""
        def drive(view_cls):
            arm = _arm32()
            st, view = _make(_registry(arm), view_cls=view_cls)
            qsl, seq_lens, nat, pos = _forward([(1, 5)])
            view.set_batch(["r0"], {"r0": _sel(arm)}, {"r0": list(range(4))},
                           (qsl, seq_lens, nat), prompt_lens={"r0": 3})
            h = torch.zeros(nat, HIDDEN)
            with pytest.raises(lw.LensWorkerError) as e:
                st.hook_body(0, (pos, h, h), (h, h))
            with pytest.raises(lw.LensWorkerError, match="CONDEMNED"):
                st.hook_body(1, (pos, h, h), (h, h))
            return str(e.value), st
        cached_msg, st = drive(KeyedView)
        uncached_msg, _ = drive(FakeView)
        assert cached_msg == uncached_msg
        assert "token source is behind the forward" in cached_msg
        assert st._fwd_cache is None

    def test_view_without_a_key_classifies_at_every_hook(self, monkeypatch):
        n = self._count(monkeypatch)
        arm = _arm32()
        st, view = _make(_registry(arm), view_cls=FakeView)
        qsl, seq_lens, nat, pos = _forward([(1, 5)])
        view.set_batch(["r0"], {"r0": _sel(arm)}, {"r0": list(range(6))},
                       (qsl, seq_lens, nat), prompt_lens={"r0": 3})
        h = torch.zeros(nat, HIDDEN)
        for layer in range(32):
            st.hook_body(layer, (pos, h, h), (h, h))
        assert n["slots"] == 32       # the pre-F1 behavior, kept for fakes

    def test_cached_groups_are_reused_across_layers(self):
        arm = _arm32()
        st, view = _make(_registry(arm))
        qsl, seq_lens, nat, pos = _forward([(1, 5), (1, 6)])
        view.set_batch(["r0", "r1"], {"r0": _sel(arm), "r1": _sel(arm)},
                       {"r0": list(range(6)), "r1": list(range(7))},
                       (qsl, seq_lens, nat), prompt_lens={"r0": 3, "r1": 3})
        h = torch.zeros(nat, HIDDEN)
        st.hook_body(0, (pos, h, h), (h, h))
        cf = st._fwd_cache[2]
        assert len(cf.groups) == 1 and cf.groups[0].rows == slice(0, 2)
        st.hook_body(1, (pos, h, h), (h, h))
        assert st._fwd_cache[2] is cf


# --------------------------------------------------------------------------
# Part B — F2: sync-free cross-checks (aliasing tensors) per family
# --------------------------------------------------------------------------


@pytest.fixture
def no_torch_equal(monkeypatch):
    """Make any device-comparison a loud failure — the aliasing fast path
    must resolve every cross-check on the real metadata shapes."""
    def boom(*a, **k):
        raise AssertionError("torch.equal was called (a device sync on the "
                             "real stack)")
    monkeypatch.setattr(torch, "equal", boom)


class MambaMeta:
    """Shaped like vllm v1/attention/backends/mamba_attn.py's
    BaseMambaAttentionMetadata at 0.25.1: seq_lens (aliasing the common
    metadata's), num_reqs and the prefill/decode counts — NO
    num_actual_tokens, NO query_start_loc, NO non_spec_* names."""

    def __init__(self, seq_lens, num_reqs):
        self.seq_lens = seq_lens
        self.num_reqs = num_reqs
        self.num_prefills = 0
        self.num_decodes = num_reqs
        self.query_start_loc_p = None
        self.query_start_loc_d = None


def _common(qsl=(0, 1, 2, 3), seq=(6, 8, 5)):
    return _t(*qsl), _t(*seq), len(seq)


def _family_md(family):
    qsl, seq, nat = _common()
    fa = FlashMeta(qsl, seq, nat)
    if family == "llama":
        return {f"model.layers.{l}.self_attn.attn": fa for l in range(4)}
    if family == "gemma4":
        sliding = FlashMeta(qsl, seq, nat)       # a second object, same tensors
        return {"model.layers.0.self_attn.attn": fa,
                "model.layers.1.self_attn.attn": sliding,
                "model.layers.2.self_attn.attn": fa,
                "model.layers.3.self_attn.attn": sliding}
    if family == "qwen35":
        gdn = GDNMeta(num_actual_tokens=nat, non_spec_query_start_loc=qsl)
        return _hybrid_dict(fa, gdn)
    if family == "nemotron":
        mamba = MambaMeta(seq, len(seq))
        return {"model.layers.0.mixer": mamba, "model.layers.1.mixer": mamba,
                "model.layers.2.mixer.attn": fa, "model.layers.3.mixer": mamba}
    if family == "glm":
        return {f"language_model.model.layers.{l}.self_attn.attn": fa
                for l in range(4)}
    raise AssertionError(family)


FAMILIES = ("llama", "gemma4", "qwen35", "nemotron", "glm")


class TestSyncFreeCrossChecks:
    @pytest.mark.parametrize("family", FAMILIES)
    def test_aliasing_groups_need_no_device_compare(self, family, no_torch_equal):
        got = lw.resolve_attn_metadata(_family_md(family))
        assert got is not None
        qsl, seq, nat = got
        assert qsl.tolist() == [0, 1, 2, 3] and seq.tolist() == [6, 8, 5]
        assert nat == 3

    def test_gemma_distinct_but_equal_copies_still_pass(self):
        qsl, seq, nat = _common()
        md = {"a": FlashMeta(qsl, seq, nat),
              "b": FlashMeta(qsl.clone(), seq.clone(), nat)}
        assert lw.resolve_attn_metadata(md) is not None

    def test_gemma_disagreeing_group_refuses_as_before(self):
        qsl, seq, nat = _common()
        bad = FlashMeta(qsl, _t(6, 9, 5), nat)
        with pytest.raises(lw.LensWorkerError, match="disagree pairwise"):
            lw.resolve_attn_metadata({"a": FlashMeta(qsl, seq, nat), "b": bad})

    def test_qwen_gdn_disagreeing_copy_refuses_as_before(self):
        qsl, seq, nat = _common()
        fa = FlashMeta(qsl, seq, nat)
        gdn = GDNMeta(num_actual_tokens=nat,
                      non_spec_query_start_loc=_t(0, 1, 3, 3))
        with pytest.raises(lw.LensWorkerError,
                           match="non_spec_query_start_loc disagrees"):
            lw.resolve_attn_metadata(_hybrid_dict(fa, gdn))
        # an equal COPY (not an alias) still passes, through the real compare
        gdn_ok = GDNMeta(num_actual_tokens=nat, non_spec_query_start_loc=qsl.clone())
        assert lw.resolve_attn_metadata(_hybrid_dict(fa, gdn_ok)) is not None

    def test_nemotron_mamba_seq_lens_disagreement_refuses(self):
        qsl, seq, nat = _common()
        fa = FlashMeta(qsl, seq, nat)
        bad = MambaMeta(_t(6, 8, 4), 3)
        with pytest.raises(lw.LensWorkerError, match="seq_lens that disagree"):
            lw.resolve_attn_metadata({"m": bad, "a": fa})
        # the mamba group alone is never authoritative
        with pytest.raises(lw.LensWorkerError, match="no KV cache group"):
            lw.resolve_attn_metadata({"m": MambaMeta(seq, 3)})

    def test_tensors_equal_fast_path_and_fallthrough(self, monkeypatch):
        base = torch.arange(10, dtype=torch.int32)
        a, b = base[:5], base[:5]                 # two views, one memory
        calls = []
        real = torch.equal

        def spy(x, y):
            calls.append(1)
            return real(x, y)
        monkeypatch.setattr(torch, "equal", spy)
        assert lw._tensors_equal(a, a) and lw._tensors_equal(a, b)
        assert calls == []                        # no compare needed
        assert lw._tensors_equal(a, base[:5].clone())
        assert not lw._tensors_equal(a, base[5:10])
        assert len(calls) == 2                    # the real compare, when needed
        # same memory, different SHAPE is not the same tensor
        assert not lw._tensors_equal(base[:4], base[:5])

    def test_mrope_rows_reduce_on_cpu_without_a_device_compare(self, no_torch_equal):
        pos = torch.arange(6, dtype=torch.long)
        stack = torch.stack([pos, pos.clone(), pos.clone()])   # 3 copies
        # copies, not aliases: _tensors_equal falls to torch.equal — which
        # on CPU tensors is what the plan path does; the fixture forbids it
        # here to show the alias shortcut is NOT what makes M-RoPE cheap
        # (the CPU mirror is), so use aliases to pass the fixture:
        stack_alias = pos.unsqueeze(0).expand(3, -1)
        assert torch.is_tensor(lw._normalize_positions(stack_alias))
        with pytest.raises(AssertionError):
            lw._normalize_positions(stack)      # copies -> a real compare

    def test_mrope_disagreeing_rows_still_refuse(self):
        pos = torch.arange(6, dtype=torch.long)
        stack = torch.stack([pos, pos.clone(), pos.clone()])
        stack[2, 3] += 1
        with pytest.raises(lw.LensWorkerError, match="DISAGREE"):
            lw._normalize_positions(stack)


# --------------------------------------------------------------------------
# Part C — F2 on the REAL view over a runner-shaped fake worker
# --------------------------------------------------------------------------


class _Buf:
    """A CpuGpuBuffer stand-in: the CPU side is the buffer, the 'device'
    side is a separate tensor the fake keeps equal (or plants unequal)."""

    def __init__(self, cpu):
        self.cpu = cpu
        self.gpu = cpu.clone()


def _fake_worker(*, qsl, seq, computed, req_ids, extra, tokens, mrope=None,
                 max_reqs=16, max_tokens=64, model=None):
    """A vLLM-0.25.1-shaped worker: runner.query_start_loc (CpuGpuBuffer),
    runner.optimistic_seq_lens_cpu (pinned CPU), runner.seq_lens (device),
    runner.positions (device, int64), input_batch.num_computed_tokens_cpu
    (numpy), requests[req].prompt/output token lists, and — for an
    M-RoPE model — runner.mrope_positions (CpuGpuBuffer, (3, T))."""
    n = len(req_ids)
    T = int(qsl[-1])
    q = _Buf(torch.zeros(max_reqs + 1, dtype=torch.int32))
    q.cpu[:n + 1] = torch.tensor(qsl, dtype=torch.int32)
    q.cpu[n + 1:] = int(qsl[-1])
    q.gpu = q.cpu.clone()
    seq_cpu = torch.zeros(max_reqs, dtype=torch.int32)
    seq_cpu[:n] = torch.tensor(seq, dtype=torch.int32)
    seq_dev = seq_cpu.clone()
    pos_vals = []
    for i in range(n):
        pos_vals.extend(range(int(computed[i]), int(computed[i]) + qsl[i + 1] - qsl[i]))
    positions = torch.zeros(max_tokens, dtype=torch.int64)
    positions[:T] = torch.tensor(pos_vals, dtype=torch.int64)
    input_batch = types.SimpleNamespace(
        num_reqs=n, req_ids=list(req_ids),
        num_computed_tokens_cpu=np.array(list(computed) + [0] * (max_reqs - n),
                                         dtype=np.int32))
    requests = {}
    for rid in req_ids:
        prompt, output = tokens[rid]
        requests[rid] = types.SimpleNamespace(
            prompt_token_ids=list(prompt), output_token_ids=list(output),
            sampling_params=types.SimpleNamespace(extra_args=extra.get(rid)))
    runner = types.SimpleNamespace(
        query_start_loc=q, optimistic_seq_lens_cpu=seq_cpu, seq_lens=seq_dev,
        positions=positions, input_batch=input_batch, requests=requests,
        model=model if model is not None else types.SimpleNamespace())
    if mrope is not None:
        m = _Buf(torch.zeros(3, max_tokens + 1, dtype=torch.int64))
        m.cpu[:, :T] = mrope
        m.gpu = m.cpu.clone()
        runner.mrope_positions = m
    worker = types.SimpleNamespace(model_runner=runner)
    md = FlashMeta(q.gpu[:n + 1], seq_dev[:n], T)
    return worker, md, positions[:T], pos_vals


def _fake_forward_context(monkeypatch, attn_metadata):
    mod = types.ModuleType("vllm.forward_context")
    mod.is_forward_context_available = lambda: attn_metadata is not None
    ctx = types.SimpleNamespace(attn_metadata=attn_metadata)
    mod.get_forward_context = lambda: ctx
    pkg = types.ModuleType("vllm")
    pkg.forward_context = mod
    monkeypatch.setitem(sys.modules, "vllm", pkg)
    monkeypatch.setitem(sys.modules, "vllm.forward_context", mod)
    return ctx


class TestRealViewCpuMirrors:
    QSL, SEQ, COMPUTED = (0, 1, 2, 5), (6, 8, 3), (5, 7, 0)
    RIDS = ("r0", "r1", "p")
    TOKENS = {"r0": (list(range(3)), [10, 11, 12]),
              "r1": (list(range(3)), [20, 21, 22, 23, 24]),
              "p": ([30, 31, 32], [])}

    def _view(self, monkeypatch, **kw):
        worker, md, pos_dev, pos_vals = _fake_worker(
            qsl=self.QSL, seq=self.SEQ, computed=self.COMPUTED,
            req_ids=self.RIDS, extra={}, tokens=self.TOKENS, **kw)
        _fake_forward_context(monkeypatch, {"l0": md, "l1": md})
        return lw.RealWorkerView(worker), worker, md, pos_dev, pos_vals

    def test_forward_key_is_the_live_attn_metadata_object(self, monkeypatch):
        view, worker, md, _, _ = self._view(monkeypatch)
        ctx = sys.modules["vllm.forward_context"].get_forward_context()
        assert view.forward_key() is ctx.attn_metadata
        _fake_forward_context(monkeypatch, None)
        assert view.forward_key() is None

    def test_cpu_mirrors_are_the_runner_buffers(self, monkeypatch):
        view, worker, md, _, _ = self._view(monkeypatch)
        qsl, seq, nat = view.forward_metadata_cpu()
        assert qsl.tolist() == list(self.QSL) and seq.tolist() == list(self.SEQ)
        assert nat == 5
        # the SAME memory as the runner's CPU buffers, not a copy of the
        # device tensors
        assert qsl.data_ptr() == worker.model_runner.query_start_loc.cpu.data_ptr()
        assert seq.data_ptr() == worker.model_runner.optimistic_seq_lens_cpu.data_ptr()
        assert view._cpu_mirror_verified
        # the plain device read is unchanged
        got = view.forward_metadata()
        assert got[0] is md.query_start_loc

    def test_cpu_mirror_mismatch_refuses_on_the_first_forward(self, monkeypatch):
        view, worker, md, _, _ = self._view(monkeypatch)
        worker.model_runner.optimistic_seq_lens_cpu[1] += 1   # plant a drift
        with pytest.raises(lw.LensWorkerError, match="seq_lens mirror disagrees"):
            view.forward_metadata_cpu()
        assert not view._cpu_mirror_verified

    def test_verification_is_once(self, monkeypatch):
        """After the first classified forward the mirrors are trusted —
        a later disagreement is NOT re-detected (the check is of the
        pinned runner's buffer pairing, not of each step): stated so a
        reader knows exactly what 'once' buys and costs."""
        view, worker, md, _, _ = self._view(monkeypatch)
        view.forward_metadata_cpu()
        worker.model_runner.optimistic_seq_lens_cpu[1] += 1
        qsl, seq, nat = view.forward_metadata_cpu()      # no refusal
        assert seq.tolist()[1] == self.SEQ[1] + 1

    def test_mirror_shape_drift_refuses(self, monkeypatch):
        view, worker, md, _, _ = self._view(monkeypatch)
        worker.model_runner.input_batch.num_reqs = 2       # fewer than the metadata
        with pytest.raises(lw.LensWorkerError, match="scheduled requests"):
            view.forward_metadata_cpu()

    def test_missing_mirror_attrs_are_reported_at_preflight(self, monkeypatch):
        view, worker, md, _, _ = self._view(monkeypatch)
        assert view.missing_runner_attrs() == ()
        del worker.model_runner.optimistic_seq_lens_cpu
        assert "optimistic_seq_lens_cpu" in view.missing_runner_attrs()
        assert lw.CPU_MIRROR_RUNNER_ATTRS == (
            "query_start_loc.cpu", "optimistic_seq_lens_cpu",
            "input_batch.num_computed_tokens_cpu")

    def test_derived_positions_match_the_device_and_verify_once(self, monkeypatch):
        view, worker, md, pos_dev, pos_vals = self._view(monkeypatch)
        got = view.forward_positions(pos_dev, 5)
        assert got.tolist() == pos_vals == [5, 7, 0, 1, 2]
        assert got.dtype == torch.long and view._positions_verified
        # a longer device tensor (a padded buffer) is read to T
        got2 = view.forward_positions(worker.model_runner.positions, 5)
        assert got2.tolist() == pos_vals

    def test_derived_positions_disagreeing_with_the_device_refuse(self, monkeypatch):
        view, worker, md, pos_dev, _ = self._view(monkeypatch)
        bad = pos_dev.clone()
        bad[3] += 1
        with pytest.raises(lw.LensWorkerError, match="disagree with the positions"):
            view.forward_positions(bad, 5)

    def test_mrope_positions_come_from_the_cpu_buffer(self, monkeypatch):
        pos = torch.tensor([5, 7, 0, 1, 2], dtype=torch.int64)
        stack = torch.stack([pos, pos, pos])
        view, worker, md, pos_dev, _ = self._view(monkeypatch, mrope=stack)
        dev_stack = worker.model_runner.mrope_positions.gpu[:, :5]
        got = view.forward_positions(dev_stack, 5)
        assert got.shape == (3, 5) and torch.equal(got, stack)
        assert got.data_ptr() == worker.model_runner.mrope_positions.cpu.data_ptr()
        assert torch.equal(lw._normalize_positions(got), pos)
        # a multimodal batch (rows disagree) is mirrored faithfully and
        # refused by the plan path's reduction, as before
        worker.model_runner.mrope_positions.cpu[2, 3] += 4
        worker.model_runner.mrope_positions.gpu[2, 3] += 4
        view2 = lw.RealWorkerView(worker)
        got = view2.forward_positions(worker.model_runner.mrope_positions.gpu[:, :5], 5)
        with pytest.raises(lw.LensWorkerError, match="DISAGREE"):
            lw._normalize_positions(got)

    def test_mrope_without_a_cpu_buffer_refuses(self, monkeypatch):
        view, worker, md, pos_dev, _ = self._view(monkeypatch)
        with pytest.raises(lw.LensWorkerError, match="mrope_positions.cpu"):
            view.forward_positions(torch.stack([pos_dev] * 3), 5)

    def test_request_tokens_is_a_ranged_view(self, monkeypatch):
        view, worker, md, _, _ = self._view(monkeypatch)
        toks = view.request_tokens("r1")
        full = list(range(3)) + [20, 21, 22, 23, 24]
        assert isinstance(toks, lw.RequestTokens)
        assert len(toks) == 8 and list(toks) == full
        assert toks[7] == 24 and toks[-1] == 24 and toks[2] == 2
        assert toks[7:8] == [24]                 # the decode read: one token
        assert toks[0:3] == [0, 1, 2]            # a complete prefill
        assert toks[2:5] == [2, 20, 21]          # across the boundary
        assert toks[1:7:2] == full[1:7:2]
        with pytest.raises(IndexError):
            toks[8]
        # a plain list behaves identically for the lookup's consumer
        n = {"slices": []}

        class Spy(lw.RequestTokens):
            def __getitem__(self, item):
                n["slices"].append(item)
                return super().__getitem__(item)
        spy = Spy(*self.TOKENS["r1"])
        plan = lb.derive_batch_plan([0, 1], [8], 1)
        fwd = lw.build_request_slots(
            query_start_loc=[0, 1], seq_lens=[8], num_actual_tokens=1,
            positions=[7], req_ids=["r1"], raw_selectors=[_sel(_last_arm())],
            token_lookup=lambda rid: spy, prompt_len_lookup=lambda rid: 3,
            registry=_registry(_last_arm()))
        assert n["slices"] == [slice(7, 8)]      # ONE ranged read, no copy
        assert fwd.slots[0].ctx.token_ids == (24,)
        assert plan.segments[0].seq_len == 8

    def test_end_to_end_hook_over_the_real_view(self, monkeypatch):
        """The whole F1/F2 surface driven through LensWorkerSteering on a
        runner-shaped fake: the classification reads the CPU mirrors (the
        tensors handed to build_request_slots share the runner's CPU
        memory), one classification serves every layer, and the steered
        rows equal the plain-view answer."""
        arm = _obj("arm", [_last_op(layer=0), _last_op(layer=1)])
        extra = {"r0": _sel(arm), "r1": _sel(arm)}
        worker, md, pos_dev, _ = _fake_worker(
            qsl=self.QSL, seq=self.SEQ, computed=self.COMPUTED,
            req_ids=self.RIDS, extra=extra, tokens=self.TOKENS)
        _fake_forward_context(monkeypatch, {"l0": md, "l1": md})
        view = lw.RealWorkerView(worker)
        seen = {}
        orig = lw.build_request_slots

        def capture(**kw):
            seen["qsl"] = kw["query_start_loc"]
            seen["positions"] = kw["positions"]
            seen.setdefault("calls", 0)
            seen["calls"] += 1
            return orig(**kw)
        monkeypatch.setattr(lw, "build_request_slots", capture)
        st = lw.LensWorkerSteering(view=view, registry=_registry(arm))
        st._preflighted = True                    # decoder_layers needs vllm-lens
        st._plan_site = (0, lw._SITE_RANK[la.HOOK_SITE_BLOCK])
        st._plan_layer = 0
        h = torch.arange(5 * HIDDEN, dtype=torch.float32).reshape(5, HIDDEN)
        outs = [st.hook_body(l, (pos_dev, h, h), (h, h)) for l in (0, 1)]
        assert seen["calls"] == 1
        assert seen["qsl"].data_ptr() == worker.model_runner.query_start_loc.cpu.data_ptr()
        assert seen["positions"].tolist() == [5, 7, 0, 1, 2]
        # r0, r1: decode rows steered; p: an UNHOOKED prefill, untouched
        for out in outs:
            assert out is not None
            new_h = out[0]
            assert not torch.equal(new_h[0], h[0]) and not torch.equal(new_h[1], h[1])
            assert torch.equal(new_h[2:5], h[2:5])
        # the plain (uncached, device-read) view gives the same bits
        plain = FakeView(layers=[])
        plain.set_batch(list(self.RIDS), extra,
                        {r: list(p) + list(o) for r, (p, o) in self.TOKENS.items()},
                        (list(self.QSL), list(self.SEQ), 5),
                        prompt_lens={r: 3 for r in self.RIDS})
        st2 = lw.LensWorkerSteering(view=plain, registry=_registry(arm))
        st2._preflighted = True
        st2._plan_site = (0, lw._SITE_RANK[la.HOOK_SITE_BLOCK])
        st2._plan_layer = 0
        ref = st2.hook_body(0, (pos_dev, h, h), (h, h))
        assert torch.equal(ref[0], outs[0][0])


def _last_arm():
    return _obj("arm-last", [_last_op(layer=1)])


# --------------------------------------------------------------------------
# Part D — the decoder-layer accessor per family (vllm-lens first, ours next)
# --------------------------------------------------------------------------


def _vllm_lens_get_layers(model):
    """vllm-lens 1.2.1's _get_layers, verbatim (read from the wheel)."""
    m = model
    if hasattr(m, "language_model") and hasattr(m.language_model, "model"):
        return m.language_model.model.layers
    if (hasattr(m, "model") and hasattr(m.model, "decoder")
            and hasattr(m.model.decoder, "layers")):
        return m.model.decoder.layers
    if hasattr(m, "model") and hasattr(m.model, "layers"):
        return m.model.layers
    raise AttributeError(
        f"Cannot find decoder layers on {type(model).__name__}. "
        "Expected model.language_model.model.layers, "
        "model.model.decoder.layers, or model.model.layers")


class TestDecoderLayerAccessor:
    @pytest.fixture(autouse=True)
    def _lens(self, monkeypatch):
        ext = types.ModuleType("vllm_lens._worker_ext")
        ext._get_layers = _vllm_lens_get_layers
        pkg = types.ModuleType("vllm_lens")
        pkg._worker_ext = ext
        monkeypatch.setitem(sys.modules, "vllm_lens", pkg)
        monkeypatch.setitem(sys.modules, "vllm_lens._worker_ext", ext)

    @staticmethod
    def _tree(shape):
        import torch.nn as nn
        layers = nn.ModuleList([nn.Identity() for _ in range(3)])

        class Wrapper(nn.Module):
            pass
        root = Wrapper()
        if shape == "llama":                       # llama / qwen3_next / nemotron_h (vLLM)
            root.model = Wrapper()
            root.model.layers = layers
        elif shape == "glm":                       # Glm4vForConditionalGeneration
            root.language_model = Wrapper()
            root.language_model.model = Wrapper()
            root.language_model.model.layers = layers
        elif shape == "gemma_nested":              # HF gemma-4 wrapper nesting
            root.model = Wrapper()
            root.model.language_model = Wrapper()
            root.model.language_model.layers = layers
        elif shape == "nemotron_backbone":         # the checkpoint's naming
            root.backbone = Wrapper()
            root.backbone.layers = layers
        elif shape == "none":
            pass
        return root, layers

    @pytest.mark.parametrize("shape,via", [
        ("llama", "lens"), ("glm", "lens"),
        ("gemma_nested", "ours"), ("nemotron_backbone", "ours")])
    def test_each_family_tree_resolves(self, shape, via):
        root, layers = self._tree(shape)
        if via == "lens":
            assert _vllm_lens_get_layers(root) is layers
        else:
            with pytest.raises(AttributeError):
                _vllm_lens_get_layers(root)      # vllm-lens cannot; ours can
        worker = types.SimpleNamespace(model_runner=types.SimpleNamespace(model=root))
        got = lw.RealWorkerView(worker).decoder_layers()
        assert list(got) == list(layers)

    def test_unresolvable_tree_refuses_loudly(self):
        root, _ = self._tree("none")
        worker = types.SimpleNamespace(model_runner=types.SimpleNamespace(model=root))
        with pytest.raises(lw.LensWorkerError, match="cannot locate the decoder"):
            lw.RealWorkerView(worker).decoder_layers()


# --------------------------------------------------------------------------
# Part E — per family: the vectorized hook on the family's metadata shape
# --------------------------------------------------------------------------


class TestFamiliesEndToEnd:
    """The same mixed decode batch driven through the worker for each
    family's classification branch — the metadata shape from Part B, the
    positions shape the family hands its layers (1-D, or the M-RoPE
    (3, T) stack) — grouped vs per-row: equal on the CPU's exact integer
    fixtures (the values cannot tell the reduction modes apart here), and
    the branch is the one named."""

    @pytest.mark.parametrize("family", FAMILIES)
    def test_branch(self, family, monkeypatch):
        arm = _obj("arm", [_last_op(layer=0), _gated_op(layer=1)])
        qsl, seq, computed = (0, 1, 2, 3), (6, 8, 5), (5, 7, 4)
        rids = ("r0", "u", "r2")
        extra = {"r0": _sel(arm), "r2": _sel(arm)}
        tokens = {"r0": (list(range(3)), [1, 2, 3]),
                  "u": (list(range(3)), [4, 5, 6, 7, 8]),
                  "r2": (list(range(3)), [9, 10])}
        outs = {}
        for path in (None, "1"):
            mrope = None
            pos1d = torch.tensor([5, 7, 4], dtype=torch.int64)
            if family in ("qwen35", "glm"):
                mrope = torch.stack([pos1d, pos1d, pos1d])
            worker, md, pos_dev, _ = _fake_worker(
                qsl=qsl, seq=seq, computed=computed, req_ids=rids, extra=extra,
                tokens=tokens, mrope=mrope)
            fam_md = _family_md(family)
            # re-point the family's bearing objects at THIS worker's tensors
            for m in {id(v): v for v in fam_md.values()}.values():
                if isinstance(m, FlashMeta):
                    m.query_start_loc, m.seq_lens = md.query_start_loc, md.seq_lens
                    m.num_actual_tokens = 3
                elif isinstance(m, GDNMeta):
                    m.non_spec_query_start_loc = md.query_start_loc
                    m.num_actual_tokens = 3
                elif isinstance(m, MambaMeta):
                    m.seq_lens = md.seq_lens
            _fake_forward_context(monkeypatch, fam_md)
            view = lw.RealWorkerView(worker)
            st = lw.LensWorkerSteering(view=view, registry=_registry(arm),
                                       per_row_apply=path)
            st._preflighted = True
            st._plan_site = (0, lw._SITE_RANK[la.HOOK_SITE_BLOCK])
            st._plan_layer = 0
            layer_pos = (worker.model_runner.mrope_positions.gpu[:, :3]
                         if mrope is not None else pos_dev)
            h = torch.arange(3 * HIDDEN, dtype=torch.float32).reshape(3, HIDDEN)
            r = torch.ones(3, HIDDEN)
            res = []
            for layer in (0, 1):
                out = st.hook_body(layer, (), (h, r),
                                   kwargs={"positions": layer_pos,
                                           "hidden_states": h, "residual": r})
                assert out is not None
                res.append(out[0])
            outs[path] = res
            cf = st._fwd_cache[2]
            if path is None:
                assert st.apply_path == lw.APPLY_PATH_VECTORIZED_STACKED
                assert [g.rows for g in cf.groups] == [(0, 2)]  # scattered pair
                assert cf.per_slot == ()
            else:
                assert st.apply_path == "per_row"
                assert cf.groups == () and len(cf.per_slot) == 2
        for layer in range(2):
            assert torch.equal(outs[None][layer], outs["1"][layer])
            assert torch.equal(outs[None][layer][1], torch.arange(
                HIDDEN, 2 * HIDDEN, dtype=torch.float32))   # the unhooked row


class TestCoScheduleOncePerForward:
    def test_record_counts_forwards_not_hooks_under_the_cache(self):
        """With a forward key the classification (and so the record) runs
        once per forward whatever the layer count; a plain FakeView counts
        per hook — the documented CPU-fake caveat."""
        a = _obj("cell-a", [_last_op(layer=l) for l in range(8)])
        b = _obj("cell-b", [_last_op(layer=l) for l in range(8)])
        for view_cls, expect in ((KeyedView, 1), (FakeView, 8)):
            st, view = _make(_registry(a, b), n_layers=8, view_cls=view_cls)
            qsl, seq_lens, nat, pos = _forward([(1, 5), (1, 7)])
            view.set_batch(["r0", "r1"], {"r0": _sel(a), "r1": _sel(b)},
                           {"r0": list(range(6)), "r1": list(range(8))},
                           (qsl, seq_lens, nat), prompt_lens={"r0": 3, "r1": 3})
            h = torch.zeros(nat, HIDDEN)
            for layer in range(8):
                st.hook_body(layer, (pos, h, h), (h, h))
            rec = st.coschedule_record.as_dict()
            assert rec["n_forwards"] == expect, view_cls.__name__
            assert rec["n_forwards_with_same_render_pair"] == expect
            assert rec["max_distinct_nonzero_groups_in_one_forward"] == 2
