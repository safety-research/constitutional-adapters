"""resolve_attn_metadata: the forward-metadata normalization behind
``RealWorkerView.forward_metadata`` — the seam that decides which
per-layer attention metadata the position logic may trust for the
batch's row segmentation. CPU-only: every metadata object here is a
fake built from the REAL vLLM field names (FlashAttention's canonical
``query_start_loc``/``seq_lens``/``num_actual_tokens``; the
gated-delta-net ``GDNAttentionMetadata`` shape from
v1/attention/backends/gdn_attn.py, which carries ``num_actual_tokens``
but expresses the segmentation under split ``non_spec_*`` names), so no
GPU or vLLM install is needed.

The load-bearing property: on a hybrid-attention model (Qwen3.5 —
8 full-attention + 24 GDN layers) the segmentation must be TAKEN only
from the canonical-field-bearing group, with every non-bearing group
CROSS-CHECKED wherever its fields permit, and every provable
disagreement — or unverifiable state — must still refuse. A mask
computed under a wrong segmentation is silently wrong; these tests pin
that the hybrid fix made the check *correct*, not lenient.

Part two pins the OTHER hybrid enabler: the decoder-layer call
conventions. llama/gemma-style models call ``layer(positions, hidden,
residual)`` positionally; qwen3_next/qwen3_5-style hybrids call
``layer(positions=..., hidden_states=..., residual=...)`` keyword-only
(positions THIRD in the signature), so the hooks register
``with_kwargs=True`` and read ``kwargs["positions"]`` first, falling
back to ``args[0]`` — and a call carrying neither still refuses. Both
conventions must steer identically, and the keyword must win whenever
the call names positions (args[0] on such a call is hidden_states).

Part three pins the KV-COVERAGE arithmetic for hybrids — the one
member of this stack that fails SILENTLY rather than loudly. For a
uniform cache the readiness verdict stays byte-identical
(num_gpu_blocks x block_size against the budget); for a hybrid cache
that formula misstates capacity (LCM scheduler block size; Mamba/GDN
state pages), so the worker reports the group structure plus the
allocator's own max-concurrency verdict (kv_group_report) and the
launcher holds THAT to max_num_seqs (kv_coverage_problem). Refusals
are reserved for unreadable structure — never for "hybrid detected".

Part four pins the M-RoPE positions layout — the layer hardware found
behind the first three. Qwen3.5 is an M-RoPE model, so every decoder
layer receives positions as a ``(3, num_tokens)`` temporal/height/width
STACK (not 1-D, and 3 is NOT the KV-group count). Text-only batches
carry three identical rows by construction, so the plan path reduces
the stack to row 0 after proving the rows equal; disagreeing rows (a
multimodal batch, where absolute-position accounting does not hold)
refuse. The row-local path never reads positions and is untouched —
row-local ops are position-independent and keep serving under any
positions shape."""

import sys
import types

import pytest
import torch

from sisteer.capeval.serving import lens_batch as lb
from sisteer.capeval.serving import lens_worker as lw
from sisteer.capeval.serving import vllm_server as vs
from test_lora_lens_serving import (D_IN, D_OUT, HEADER as LORA_HEADER,
                                    _lora_obj, _registry as _lora_registry,
                                    _sel as _lora_sel,
                                    _steering as _oproj_steering)
from test_petri_lens_worker import (HIDDEN, _forward, _last_op,
                                    _make_steering, _obj, _registry,
                                    _row_local_op, _sel)


# --------------------------------------------------------------------------
# fakes: metadata shapes, from the real field names
# --------------------------------------------------------------------------


class FlashMeta:
    """The canonical V1 flash-attention metadata surface the wiring
    reads (the real class carries more fields; only these matter)."""

    def __init__(self, qsl, seq_lens, nat):
        self.query_start_loc = qsl
        self.seq_lens = seq_lens
        self.num_actual_tokens = nat


class GDNMeta:
    """Shaped like vllm v1/attention/backends/gdn_attn.py's
    ``GDNAttentionMetadata``: ``num_actual_tokens`` but NO top-level
    ``query_start_loc`` / ``seq_lens`` — the segmentation lives under
    split names, with ``non_spec_query_start_loc`` equal to the batch
    ``query_start_loc`` whenever no speculative decode is in flight.
    Built from the real field names on purpose, so the tests run
    without a GPU or any particular vLLM version installed."""

    def __init__(self, *, num_actual_tokens, non_spec_query_start_loc,
                 num_spec_decodes=0, num_spec_decode_tokens=0,
                 num_prefills=0, num_prefill_tokens=0,
                 num_decodes=0, num_decode_tokens=0):
        self.num_prefills = num_prefills
        self.num_prefill_tokens = num_prefill_tokens
        self.num_decodes = num_decodes
        self.num_decode_tokens = num_decode_tokens
        self.num_spec_decodes = num_spec_decodes
        self.num_spec_decode_tokens = num_spec_decode_tokens
        self.num_actual_tokens = num_actual_tokens
        self.non_spec_query_start_loc = non_spec_query_start_loc
        self.spec_query_start_loc = None
        self.spec_sequence_masks = None
        self.spec_token_indx = None
        self.non_spec_token_indx = None
        self.prefill_query_start_loc = None
        self.has_initial_state = None


def _t(*vals):
    return torch.tensor(vals, dtype=torch.int32)


def _flash(qsl=(0, 5, 6), seq_lens=(5, 9), nat=6):
    return FlashMeta(_t(*qsl), _t(*seq_lens), nat)


def _gdn(qsl=(0, 5, 6), nat=6, **kw):
    return GDNMeta(num_actual_tokens=nat,
                   non_spec_query_start_loc=_t(*qsl), **kw)


def _hybrid_dict(fa, gdn):
    """Qwen3.5-shaped per-layer dict: a few full-attention layers
    sharing one metadata object, many GDN layers sharing another."""
    d = {f"model.layers.{i}.self_attn": fa for i in range(2)}
    d.update({f"model.layers.{i}.linear_attn": gdn for i in range(2, 8)})
    return d


# --------------------------------------------------------------------------
# the uniform (pre-hybrid) paths, unchanged
# --------------------------------------------------------------------------


def test_uniform_single_group_dict_passes():
    fa = _flash()
    got = lw.resolve_attn_metadata({f"l{i}": fa for i in range(4)})
    assert got is not None
    qsl, seq_lens, nat = got
    assert qsl is fa.query_start_loc
    assert seq_lens is fa.seq_lens
    assert nat == 6


def test_uniform_distinct_but_agreeing_objects_pass():
    got = lw.resolve_attn_metadata({"a": _flash(), "b": _flash()})
    assert got is not None
    assert torch.equal(got[0], _t(0, 5, 6))


def test_bare_single_object_passes():
    fa = _flash()
    qsl, seq_lens, nat = lw.resolve_attn_metadata(fa)
    assert qsl is fa.query_start_loc and nat == 6


def test_none_and_empty_are_no_forward():
    assert lw.resolve_attn_metadata(None) is None
    assert lw.resolve_attn_metadata({}) is None
    assert lw.resolve_attn_metadata([None]) is None


def test_single_element_list_unwraps():
    fa = _flash()
    got = lw.resolve_attn_metadata([{"a": fa}])
    assert got is not None and got[0] is fa.query_start_loc


def test_multi_element_list_refuses():
    with pytest.raises(lw.LensWorkerError, match="element list"):
        lw.resolve_attn_metadata([{"a": _flash()}, {"a": _flash()}])


def test_bare_object_without_canonical_fields_refuses():
    # A GDN-only view arriving as a single object: never authoritative.
    with pytest.raises(lw.LensWorkerError, match="has no"):
        lw.resolve_attn_metadata(_gdn())


def test_none_entry_in_dict_refuses():
    with pytest.raises(lw.LensWorkerError, match="None"):
        lw.resolve_attn_metadata({"a": _flash(), "b": None})


# --------------------------------------------------------------------------
# the hybrid partition (Qwen3.5: full-attention + gated-delta-net)
# --------------------------------------------------------------------------


def test_hybrid_agreeing_takes_segmentation_from_bearing_group():
    fa, gdn = _flash(), _gdn()
    qsl, seq_lens, nat = lw.resolve_attn_metadata(_hybrid_dict(fa, gdn))
    # The canonical group's own tensors, not a reconstruction.
    assert qsl is fa.query_start_loc
    assert seq_lens is fa.seq_lens
    assert nat == 6


def test_hybrid_gdn_num_actual_tokens_disagreement_refuses():
    with pytest.raises(lw.LensWorkerError, match="num_actual_tokens"):
        lw.resolve_attn_metadata(_hybrid_dict(_flash(), _gdn(nat=5)))


def test_hybrid_gdn_split_qsl_disagreement_refuses():
    with pytest.raises(lw.LensWorkerError,
                       match="non_spec_query_start_loc"):
        lw.resolve_attn_metadata(
            _hybrid_dict(_flash(), _gdn(qsl=(0, 4, 6))))


def test_hybrid_spec_decode_in_flight_refuses():
    # With speculative decode active the non_spec split no longer
    # describes the whole batch — unverifiable, so it must refuse, not
    # skip the check.
    with pytest.raises(lw.LensWorkerError, match="speculative"):
        lw.resolve_attn_metadata(_hybrid_dict(
            _flash(), _gdn(num_spec_decodes=1, num_spec_decode_tokens=2)))


def test_hybrid_unrecognized_split_shape_refuses():
    # non_spec_query_start_loc without num_spec_decodes: not the GDN
    # shape the cross-check reads — cannot prove the split covers the
    # whole batch; must refuse, not pass unchecked.
    odd = types.SimpleNamespace(num_actual_tokens=6,
                                non_spec_query_start_loc=_t(0, 5, 6))
    with pytest.raises(lw.LensWorkerError, match="num_spec_decodes"):
        lw.resolve_attn_metadata({"a": _flash(), "b": odd})


def test_bearing_groups_internal_disagreement_refuses():
    with pytest.raises(lw.LensWorkerError, match="disagrees across"):
        lw.resolve_attn_metadata(
            {"a": _flash(qsl=(0, 5, 6)), "b": _flash(qsl=(0, 4, 6),
                                                     seq_lens=(4, 9))})
    with pytest.raises(lw.LensWorkerError, match="disagrees across"):
        lw.resolve_attn_metadata(
            {"a": _flash(seq_lens=(5, 9)), "b": _flash(seq_lens=(5, 8))})
    with pytest.raises(lw.LensWorkerError, match="disagrees across"):
        lw.resolve_attn_metadata({"a": _flash(nat=6), "b": _flash(nat=7)})


def test_bearing_set_empty_refuses_even_when_split_is_valid():
    # An all-GDN dict whose split fields WOULD form a perfectly valid
    # segmentation: the non-bearing group alone must never be silently
    # treated as authoritative.
    with pytest.raises(lw.LensWorkerError, match="canonical"):
        lw.resolve_attn_metadata({"a": _gdn(), "b": _gdn()})


def test_partial_canonical_fields_still_cross_checked():
    # A group carrying SOME canonical fields (qsl but no seq_lens) is
    # non-bearing, but what it does carry must still agree.
    partial_ok = types.SimpleNamespace(query_start_loc=_t(0, 5, 6),
                                       num_actual_tokens=6)
    got = lw.resolve_attn_metadata({"a": _flash(), "b": partial_ok})
    assert got is not None and got[2] == 6

    partial_bad = types.SimpleNamespace(query_start_loc=_t(0, 3, 6),
                                        num_actual_tokens=6)
    with pytest.raises(lw.LensWorkerError, match="query_start_loc"):
        lw.resolve_attn_metadata({"a": _flash(), "b": partial_bad})

    seqlens_bad = types.SimpleNamespace(seq_lens=_t(5, 8),
                                        num_actual_tokens=6)
    with pytest.raises(lw.LensWorkerError, match="seq_lens"):
        lw.resolve_attn_metadata({"a": _flash(), "b": seqlens_bad})


def test_bearing_requires_populated_fields():
    # Canonical attrs present but None do not bear the segmentation —
    # and alone they must refuse, not crash or pass.
    hollow = types.SimpleNamespace(query_start_loc=None, seq_lens=None,
                                   num_actual_tokens=6)
    with pytest.raises(lw.LensWorkerError, match="canonical"):
        lw.resolve_attn_metadata({"a": hollow})
    # Next to a bearing group, the hollow object's num_actual_tokens is
    # still comparable — and must agree.
    assert lw.resolve_attn_metadata({"a": _flash(), "b": hollow})
    hollow_bad = types.SimpleNamespace(query_start_loc=None,
                                       seq_lens=None, num_actual_tokens=9)
    with pytest.raises(lw.LensWorkerError, match="num_actual_tokens"):
        lw.resolve_attn_metadata({"a": _flash(), "b": hollow_bad})


# --------------------------------------------------------------------------
# the RealWorkerView delegation (fake vllm.forward_context module)
# --------------------------------------------------------------------------


def _fake_forward_context(monkeypatch, available, attn_metadata=None):
    mod = types.ModuleType("vllm.forward_context")
    mod.is_forward_context_available = lambda: available
    mod.get_forward_context = lambda: types.SimpleNamespace(
        attn_metadata=attn_metadata)
    pkg = types.ModuleType("vllm")
    pkg.forward_context = mod
    monkeypatch.setitem(sys.modules, "vllm", pkg)
    monkeypatch.setitem(sys.modules, "vllm.forward_context", mod)


def test_real_view_no_context_is_none(monkeypatch):
    _fake_forward_context(monkeypatch, available=False)
    assert lw.RealWorkerView(object()).forward_metadata() is None


def test_real_view_delegates_hybrid_to_resolver(monkeypatch):
    fa = _flash()
    _fake_forward_context(monkeypatch, available=True,
                          attn_metadata=_hybrid_dict(fa, _gdn()))
    got = lw.RealWorkerView(object()).forward_metadata()
    assert got is not None and got[0] is fa.query_start_loc

    _fake_forward_context(monkeypatch, available=True,
                          attn_metadata=_hybrid_dict(fa, _gdn(nat=5)))
    with pytest.raises(lw.LensWorkerError, match="num_actual_tokens"):
        lw.RealWorkerView(object()).forward_metadata()


# --------------------------------------------------------------------------
# the decoder-layer call conventions (positions: keyword first, args[0] next)
# --------------------------------------------------------------------------


def _last_arm_steering():
    """A plan-NEEDING arm (positions='last'), so the positions read is
    load-bearing in every test below: the row-local path never consumes
    positions, and a convention test built on it would pass vacuously."""
    arm = _obj("arm-last", [_last_op(layer=1)])
    steering, view = _make_steering(_registry(arm))
    return arm, steering, view


def _prefill_batch(view, arm, q=5):
    qsl, seq_lens, nat, pos = _forward([(q, 0)])
    view.set_batch(["req-s"], {"req-s": _sel(arm)},
                   {"req-s": list(range(100, 100 + q))},
                   (qsl, seq_lens, nat))
    return nat, pos


def _hr(nat, seed=11):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(nat, HIDDEN, generator=g),
            torch.randn(nat, HIDDEN, generator=g))


def test_positional_call_convention_unchanged():
    # llama/gemma shape: layer(positions, hidden, residual) positionally.
    arm, steering, view = _last_arm_steering()
    nat, pos = _prefill_batch(view, arm)
    h, r = _hr(nat)
    out = steering.hook_body(1, (pos, h, r), (h, r))
    assert out is not None
    new_h, new_r = out
    assert torch.equal(new_r, r)
    assert torch.equal(new_h[:nat - 1], h[:nat - 1])   # positions='last'
    assert not torch.equal(new_h[nat - 1], h[nat - 1])


def test_keyword_call_convention_steers_identically():
    # qwen3_next shape: layer(positions=..., hidden_states=...,
    # residual=...) keyword-only — torch hands the hook args=() and the
    # kwargs dict. The SAME forward must resolve the same positions and
    # produce the SAME steered output as the positional convention.
    results = []
    for mode in ("positional", "keyword"):
        arm, steering, view = _last_arm_steering()
        nat, pos = _prefill_batch(view, arm)
        h, r = _hr(nat)
        if mode == "positional":
            out = steering.hook_body(1, (pos, h, r), (h, r))
        else:
            out = steering.hook_body(
                1, (), (h, r),
                kwargs={"positions": pos, "hidden_states": h,
                        "residual": r})
        assert out is not None, mode
        results.append(out)
    assert torch.equal(results[0][0], results[1][0])
    assert torch.equal(results[0][1], results[1][1])


def test_keyword_positions_win_over_positional_args():
    # qwen3_next's SIGNATURE is forward(hidden_states, residual,
    # positions=None): whenever such a model is called with positions
    # as a keyword and anything positional, args[0] is hidden_states —
    # the keyword must win or the positions read is a float matrix.
    arm, steering, view = _last_arm_steering()
    nat, pos = _prefill_batch(view, arm)
    h, r = _hr(nat)
    out = steering.hook_body(1, (h, r), (h, r), kwargs={"positions": pos})
    assert out is not None
    new_h, _ = out
    assert torch.equal(new_h[:nat - 1], h[:nat - 1])
    assert not torch.equal(new_h[nat - 1], h[nat - 1])


def test_positional_fallback_on_a_keyword_shaped_call_refuses_loudly():
    # The counterfactual for the test above: with NO positions keyword,
    # a hidden_states tensor at args[0] reaches the positions
    # validation and refuses — the wrong-convention read is loud,
    # never a silent mis-row.
    arm, steering, view = _last_arm_steering()
    nat, _pos = _prefill_batch(view, arm)
    h, r = _hr(nat)
    with pytest.raises((lw.LensWorkerError, lb.LensBatchError)):
        steering.hook_body(1, (h, r), (h, r))


def test_call_with_neither_convention_refuses_and_condemns():
    arm, steering, view = _last_arm_steering()
    nat, pos = _prefill_batch(view, arm)
    h, r = _hr(nat)
    with pytest.raises(lw.LensWorkerError, match="neither a"):
        steering.hook_body(1, (), (h, r), kwargs={})
    # the refusal condemned the worker, like every hook refusal
    with pytest.raises(lw.LensWorkerError, match="CONDEMNED"):
        steering.hook_body(1, (pos, h, r), (h, r))
    # and the kwargs=None default (a bare no-args call) refuses the same
    arm2, steering2, view2 = _last_arm_steering()
    _prefill_batch(view2, arm2)
    with pytest.raises(lw.LensWorkerError, match="neither a"):
        steering2.hook_body(1, (), (h, r))


def test_warmup_keyword_call_is_still_a_passthrough():
    # a warmup forward (no requests in flight) never reads positions,
    # whatever the call convention — provable passthrough, not a refusal.
    arm, steering, view = _last_arm_steering()
    view.set_batch([], {}, {}, None)
    assert steering.hook_body(1, (), None, kwargs={}) is None


def test_install_registers_block_hooks_with_kwargs():
    arm, steering, view = _last_arm_steering()
    assert [i for i, l in enumerate(view.layers) if l.hooks] == [1]
    assert view.layers[1].hook_with_kwargs == [True]
    # end-to-end through the REGISTERED closure, keyword-shaped call
    nat, pos = _prefill_batch(view, arm)
    h, r = _hr(nat)
    out = view.layers[1].hooks[0](
        None, (), {"positions": pos, "hidden_states": h, "residual": r},
        (h, r))
    assert out is not None


def test_pos_stash_reads_both_conventions():
    arm = _lora_obj(layers=(1,))
    steering, view = _oproj_steering(_lora_registry(arm))
    layer = view.layers[1]
    assert layer.pre_hook_with_kwargs == [True]
    (pre,) = layer.pre_hooks
    pos = torch.arange(4)
    pre(None, (pos,), {})                                   # positional
    assert steering._fwd_positions is pos
    pos2 = torch.arange(5)
    pre(None, (), {"positions": pos2, "hidden_states": None,
                   "residual": None})                       # keyword-only
    assert steering._fwd_positions is pos2
    pre(None, (), {})                # neither: stash None, o_proj refuses
    assert steering._fwd_positions is None


def test_oproj_steers_after_a_keyword_stash():
    # the full o_proj path under the qwen3_next-style layer call: the
    # pre-hook stashes positions from the KEYWORD call and the o_proj
    # hook steers off that stash exactly as under the positional call.
    arm = _lora_obj(layers=(1,))
    steering, view = _oproj_steering(_lora_registry(arm))
    prompt = [3, 4] + list(LORA_HEADER)
    qsl, seq_lens, nat, pos = _forward([(1, len(prompt))])
    view.set_batch(["req-s"], {"req-s": _lora_sel(arm)},
                   {"req-s": list(prompt) + [42]},
                   (qsl, seq_lens, nat),
                   prompt_lens={"req-s": len(prompt)})
    for layer in view.layers:
        for pre in layer.pre_hooks:
            pre(None, (), {"positions": pos, "hidden_states": None,
                           "residual": None})
    x, out = torch.randn(1, D_IN), torch.randn(1, D_OUT)
    new = steering.oproj_hook_body(1, (x,), out)
    assert torch.is_tensor(new) and not torch.equal(new, out)


# --------------------------------------------------------------------------
# KV coverage on hybrids: kv_group_report (worker) + kv_coverage_problem
# --------------------------------------------------------------------------


class FullAttentionSpec:
    """Named like the real vLLM spec on purpose — the report carries
    ``type(spec).__name__``."""

    def __init__(self, block_size=16, page_size_bytes=4096):
        self.block_size = block_size
        self.page_size_bytes = page_size_bytes


class MambaSpec:
    def __init__(self, block_size=64, page_size_bytes=1 << 20):
        self.block_size = block_size
        self.page_size_bytes = page_size_bytes


class _Group:
    def __init__(self, spec, n_layers):
        self.kv_cache_spec = spec
        self.layer_names = [f"model.layers.{i}" for i in range(n_layers)]


class _KVConfig:
    def __init__(self, groups, num_blocks=8192):
        self.kv_cache_groups = groups
        self.num_blocks = num_blocks


class _Runner:
    def __init__(self, kv_cache_config):
        if kv_cache_config is not None:
            self.kv_cache_config = kv_cache_config


def _mc_fn(value):
    calls = []

    def fn(vllm_config, kv_cfg):
        calls.append((vllm_config, kv_cfg))
        return value
    fn.calls = calls
    return fn


def test_kv_group_report_uniform_single_group():
    cfg = object()
    kv = _KVConfig([_Group(FullAttentionSpec(), 32)])
    fn = _mc_fn(48.0)
    rep = lw.kv_group_report(cfg, _Runner(kv), fn)
    assert rep["kv_num_cache_groups"] == 1
    assert rep["kv_cache_groups"] == [{"spec": "FullAttentionSpec",
                                       "block_size": 16, "num_layers": 32,
                                       "page_size_bytes": 4096}]
    assert rep["kv_max_concurrency"] == 48.0
    # the allocator's own function, fed the REAL config objects
    assert fn.calls == [(cfg, kv)]


def test_kv_group_report_hybrid_two_groups():
    # the Qwen3.5 shape: a full-attention group + a GDN/Mamba group
    kv = _KVConfig([_Group(FullAttentionSpec(block_size=16), 8),
                    _Group(MambaSpec(block_size=64), 24)])
    rep = lw.kv_group_report(object(), _Runner(kv), _mc_fn(9.5))
    assert rep["kv_num_cache_groups"] == 2
    assert [g["spec"] for g in rep["kv_cache_groups"]] == [
        "FullAttentionSpec", "MambaSpec"]
    assert [g["num_layers"] for g in rep["kv_cache_groups"]] == [8, 24]
    assert rep["kv_max_concurrency"] == 9.5


def test_kv_group_report_unreadable_structure_refuses():
    fn = _mc_fn(9.5)
    with pytest.raises(lw.LensWorkerError, match="kv_cache_config"):
        lw.kv_group_report(object(), _Runner(None), fn)      # attr missing
    with pytest.raises(lw.LensWorkerError, match="kv_cache_config"):
        lw.kv_group_report(object(), None, fn)               # no runner
    with pytest.raises(lw.LensWorkerError, match="no KV cache groups"):
        lw.kv_group_report(object(), _Runner(_KVConfig([])), fn)
    with pytest.raises(lw.LensWorkerError, match="no KV cache groups"):
        lw.kv_group_report(object(), _Runner(_KVConfig(None)), fn)


def test_kv_group_report_unaccountable_spec_refuses():
    class WeirdSpec:  # no block_size / page_size_bytes at all
        pass

    class RaisingSpec:
        block_size = 16

        @property
        def page_size_bytes(self):
            raise RuntimeError("no page size for you")

    fn = _mc_fn(9.5)
    kv = _KVConfig([_Group(FullAttentionSpec(), 8),
                    _Group(WeirdSpec(), 24)])
    with pytest.raises(lw.LensWorkerError, match="cannot account for"):
        lw.kv_group_report(object(), _Runner(kv), fn)
    kv = _KVConfig([_Group(RaisingSpec(), 8)])
    with pytest.raises(lw.LensWorkerError, match="cannot account for"):
        lw.kv_group_report(object(), _Runner(kv), fn)


def test_kv_group_report_bad_verdict_refuses():
    kv = _KVConfig([_Group(FullAttentionSpec(), 8)])

    def raising(vllm_config, kv_cfg):
        raise ValueError("allocator says no")
    with pytest.raises(lw.LensWorkerError, match="allocator"):
        lw.kv_group_report(object(), _Runner(kv), raising)
    for bad in (0.0, -3.0, float("inf"), float("nan")):
        with pytest.raises(lw.LensWorkerError, match="max concurrency"):
            lw.kv_group_report(object(), _Runner(kv), _mc_fn(bad))
    with pytest.raises(lw.LensWorkerError, match="allocator"):
        lw.kv_group_report(object(), _Runner(kv), _mc_fn("not-a-number"))


def test_kv_group_report_default_fn_import_refuses_without_vllm():
    # the lazy default import path: no vLLM on this box, so the report
    # must refuse ("could not be imported"), never guess a coverage.
    kv = _KVConfig([_Group(FullAttentionSpec(), 8)])
    with pytest.raises(lw.LensWorkerError, match="could not be imported"):
        lw.kv_group_report(object(), _Runner(kv), None)


# ---- the launcher half: kv_coverage_problem over the reported halves ----


def _uniform_rep(**over):
    rep = {"kv_num_gpu_blocks": 8192, "kv_block_size": 16,
           "kv_num_cache_groups": 1}
    rep.update(over)
    return rep


def _hybrid_rep(mc=9.5, **over):
    rep = {"kv_num_gpu_blocks": 8192, "kv_block_size": 16,
           "kv_num_cache_groups": 2,
           "kv_cache_groups": [
               {"spec": "FullAttentionSpec", "block_size": 16,
                "num_layers": 8, "page_size_bytes": 4096},
               {"spec": "MambaSpec", "block_size": 64,
                "num_layers": 24, "page_size_bytes": 1 << 20}],
           "kv_max_concurrency": mc}
    rep.update(over)
    return rep


def _cov(rep, max_num_seqs=8, max_model_len=16384):
    return vs.kv_coverage_problem(rep, max_num_seqs=max_num_seqs,
                                   max_model_len=max_model_len)


def test_uniform_coverage_verdicts_are_unchanged():
    # exactly covering: 8192 x 16 == 8 x 16384 — passes, and the
    # allocator fields are NOT consulted on the uniform path (an absurd
    # kv_max_concurrency cannot perturb the verdict in either direction)
    assert _cov(_uniform_rep()) is None
    assert _cov(_uniform_rep(kv_max_concurrency=0.001)) is None
    # one block short: the ORIGINAL message, numbers and all
    msg = _cov(_uniform_rep(kv_num_gpu_blocks=8191))
    assert "131056 tokens < the 131072-token budget" in msg
    assert "argv but not in memory" in msg
    # unreadable KV size: the ORIGINAL refusal
    msg = _cov(_uniform_rep(kv_num_gpu_blocks=None))
    assert "resolved KV cache size could not be read" in msg


def test_missing_group_count_refuses():
    for rep in ({"kv_num_gpu_blocks": 8192, "kv_block_size": 16},
                _uniform_rep(kv_num_cache_groups=0),
                _uniform_rep(kv_num_cache_groups=True),
                _uniform_rep(kv_num_cache_groups="2")):
        msg = _cov(rep)
        assert msg and "group structure could not be read" in msg


def test_hybrid_coverage_uses_the_allocator_verdict():
    # healthy hybrid passes — NOT refused for being hybrid
    assert _cov(_hybrid_rep(mc=9.5)) is None
    assert _cov(_hybrid_rep(mc=8.0)) is None      # exactly covering
    # under-covered hybrid refuses on the allocator's number, naming
    # the group structure and both numbers
    msg = _cov(_hybrid_rep(mc=7.99))
    assert "7.99-way" in msg and "max_num_seqs=8" in msg
    assert "FullAttentionSpec" in msg and "MambaSpec" in msg
    assert "argv but not in memory" in msg


def test_hybrid_ignores_the_uniform_formula():
    # THE QUIET-MISCOUNT SHAPE, both directions:
    # (a) uniform arithmetic looks HEALTHY (8192 x 16 covers the budget)
    #     but the allocator says 2.5-way — must refuse on the allocator
    msg = _cov(_hybrid_rep(mc=2.5))
    assert msg is not None and "2.50-way" in msg
    # (b) uniform arithmetic looks STARVED (tiny block count) but the
    #     allocator says the width is backed — must pass
    assert _cov(_hybrid_rep(mc=64.0, kv_num_gpu_blocks=1)) is None


def test_hybrid_unreadable_fields_refuse():
    msg = _cov(_hybrid_rep(kv_cache_groups=None))
    assert "per-group structure could not be read" in msg
    msg = _cov(_hybrid_rep(kv_cache_groups=[{"spec": "FullAttentionSpec"}]))
    assert "per-group structure could not be read" in msg  # count mismatch
    msg = _cov(_hybrid_rep(kv_cache_groups=[{"no-spec": 1}, {"spec": ""}]))
    assert "per-group structure could not be read" in msg
    for bad in (None, True, float("inf"), float("nan"), 0, -1, "9.5"):
        msg = _cov(_hybrid_rep(mc=bad))
        assert msg and "max-concurrency verdict could not be read" in msg


def test_hybrid_flows_through_validate_scheduler_reports():
    # end-to-end through the readiness validator, the way the launcher
    # actually consumes the report (expected keys compared, kv_* fields
    # consumed by the coverage check, extras passed through)
    expected = {"max_num_seqs": 8, "max_model_len": 16384}
    good = dict(_hybrid_rep(mc=9.5), **expected)
    assert vs.validate_scheduler_reports([good], expected)
    with pytest.raises(vs.VllmServerError, match="2.50-way"):
        vs.validate_scheduler_reports(
            [dict(_hybrid_rep(mc=2.5), **expected)], expected)


# --------------------------------------------------------------------------
# M-RoPE positions: the (3, num_tokens) stack (Qwen3.5's fourth layer)
# --------------------------------------------------------------------------


def _stack(pos, rows=3):
    return torch.stack([pos.clone() for _ in range(rows)])


def test_mrope_stacked_positions_steer_identically_to_1d():
    # qwen3_5 hands every layer positions as a (3, T) t/h/w stack; for
    # a text-only batch the rows are identical and the plan path must
    # resolve the SAME rows as the classic 1-D vector.
    results = []
    for stacked in (False, True):
        arm, steering, view = _last_arm_steering()
        nat, pos = _prefill_batch(view, arm)
        h, r = _hr(nat)
        p = _stack(pos) if stacked else pos
        out = steering.hook_body(
            1, (), (h, r),
            kwargs={"positions": p, "hidden_states": h, "residual": r})
        assert out is not None, f"stacked={stacked}"
        results.append(out)
    assert torch.equal(results[0][0], results[1][0])
    assert torch.equal(results[0][1], results[1][1])
    # and the steered row is the LAST one (positions='last') — the
    # reduction fed real positions, not a guess. _hr is seed-pinned, so
    # the un-steered h is recomputable for the comparison.
    new_h = results[1][0]
    nat = new_h.shape[0]
    h, _ = _hr(nat)
    assert not torch.equal(new_h[nat - 1], h[nat - 1])
    assert torch.equal(new_h[:nat - 1], h[:nat - 1])


def test_mrope_disagreeing_rows_refuse_and_condemn():
    # rows disagree exactly when image/video tokens are in the batch —
    # there the absolute-position accounting does not hold, so the
    # plan path must refuse, never take row 0 on faith.
    arm, steering, view = _last_arm_steering()
    nat, pos = _prefill_batch(view, arm)
    h, r = _hr(nat)
    bad = _stack(pos)
    bad[2, 2:] += 7   # an image-grid-like divergence in the w row
    with pytest.raises(lw.LensWorkerError, match="DISAGREE"):
        steering.hook_body(
            1, (), (h, r),
            kwargs={"positions": bad, "hidden_states": h, "residual": r})
    with pytest.raises(lw.LensWorkerError, match="CONDEMNED"):
        steering.hook_body(1, (pos, h, r), (h, r))


def test_row_local_ops_never_read_positions_whatever_the_shape():
    # the row-local path is position-INDEPENDENT and never consumes
    # positions — a disagreeing stack must not refuse it (no
    # over-refusal), exactly as the pre-M-RoPE code behaved.
    arm = _obj("arm-local", [_row_local_op(layer=1)])
    steering, view = _make_steering(_registry(arm))
    nat, pos = _prefill_batch(view, arm)
    h, r = _hr(nat)
    bad = _stack(pos)
    bad[2, 2:] += 7
    out = steering.hook_body(
        1, (), (h, r),
        kwargs={"positions": bad, "hidden_states": h, "residual": r})
    assert out is not None   # steered, all rows, positions untouched


def test_normalize_positions_unit():
    pos = torch.arange(6, dtype=torch.long)
    # 1-D: passthrough, object-identical (the uniform path untouched)
    assert lw._normalize_positions(pos) is pos
    # non-tensor: passthrough (the CPU tests' list-shaped inputs)
    as_list = [0, 1, 2]
    assert lw._normalize_positions(as_list) is as_list
    # (3, T) agreeing: reduced to the single row
    assert torch.equal(lw._normalize_positions(_stack(pos)), pos)
    # generic row counts reduce too — the safety is rows-equal, not "3"
    assert torch.equal(lw._normalize_positions(_stack(pos, rows=2)), pos)
    assert torch.equal(lw._normalize_positions(pos.unsqueeze(0)), pos)
    # no rows at all: refuse
    with pytest.raises(lw.LensWorkerError, match="no"):
        lw._normalize_positions(torch.zeros((0, 6), dtype=torch.long))
    # disagreeing rows: refuse
    bad = _stack(pos)
    bad[1, 3] += 1
    with pytest.raises(lw.LensWorkerError, match="DISAGREE"):
        lw._normalize_positions(bad)


def test_oproj_steers_after_a_keyword_stash_of_stacked_positions():
    # the full o_proj path under qwen3_5's real call shape: keyword-only
    # layer call AND (3, T) M-RoPE positions in the stash.
    arm = _lora_obj(layers=(1,))
    steering, view = _oproj_steering(_lora_registry(arm))
    prompt = [3, 4] + list(LORA_HEADER)
    qsl, seq_lens, nat, pos = _forward([(1, len(prompt))])
    view.set_batch(["req-s"], {"req-s": _lora_sel(arm)},
                   {"req-s": list(prompt) + [42]},
                   (qsl, seq_lens, nat),
                   prompt_lens={"req-s": len(prompt)})
    for layer in view.layers:
        for pre in layer.pre_hooks:
            pre(None, (), {"positions": _stack(pos),
                           "hidden_states": None, "residual": None})
    x, out = torch.randn(1, D_IN), torch.randn(1, D_OUT)
    new = steering.oproj_hook_body(1, (x,), out)
    assert torch.is_tensor(new) and not torch.equal(new, out)
