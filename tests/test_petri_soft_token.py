"""Soft-token serving via prompt embeddings (serving/soft_token.py).

THE claim under test: the embedding sequence the module builds, run with
NO hook, equals the real SoftTokenObject's own hooked forward
BIT-EXACTLY — logits, every hidden state, and a KV-cached decode step —
on a real tiny Llama. Plus: the sentinel is LOCATED (never computed),
every malformed input refuses loudly, scale 0 means what the class means
by it (the unsteered MODEL on the INJECTED prompt — not the raw prompt),
and the two vLLM 0.25.1 request shapes are built with the engine flag
checked rather than assumed.

All fixtures are synthetic: a from-config tiny Llama (no hub access) and
a deterministic fake tokenizer. Nothing imports vllm — the request
formats are read-not-executed (module docstring's GPU list).
"""

import base64
import io

import pytest
import torch

from sisteer.capeval.serving import lens_adapters as la
from sisteer.capeval.serving import soft_token as st
from sisteer.extraction.objects import SoftTokenObject
from sisteer.extraction.tokens import AppendedSoftTokenObject

HIDDEN = 32
N_LAYERS = 4

BOS = "<s>"
SENT = "<SENT>"
BOS_ID = 1
SENT_ID = 5
WORDS = {"alpha": 10, "beta": 11, "gamma": 12, "delta": 13,
         "epsilon": 14, "zeta": 15}
SPECIALS = {BOS: BOS_ID, SENT: SENT_ID, "<bos>": 3,
            "<|begin_of_text|>": 4}

ENGINE_ON = {"enable_prompt_embeds": True}


# --------------------------------------------------------------------------
# fixtures: tiny model + deterministic fake tokenizer
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


class FakeTokenizer:
    """Deterministic word-level tokenizer with single-token specials —
    the HF calling convention (``tok(text, add_special_tokens=...)
    ["input_ids"]``), including BOS prepending under
    ``add_special_tokens=True`` (Llama-style: unconditional)."""

    def __init__(self, specials=SPECIALS, words=WORDS, bos_id=BOS_ID):
        self.specials = dict(specials)
        self.words = dict(words)
        self.bos_id = bos_id

    def _encode(self, text):
        for s, i in self.specials.items():
            if s in text:
                left, right = text.split(s, 1)
                return self._encode(left) + [i] + self._encode(right)
        return [self.words[w] for w in text.split()]

    def __call__(self, text, add_special_tokens=False):
        ids = self._encode(text)
        if add_special_tokens:
            ids = [self.bos_id] + ids
        return {"input_ids": ids}


def _blob(obj, tmp_path, name="object.pt"):
    path = tmp_path / name
    obj.save(path)
    return path.read_bytes()


def _spec(kind, sentinel=SENT, **extra):
    # the shape registry_adapter._resolve_torch_blob produces: the blob's
    # non-tensor fields + kind + arm.
    spec = {"kind": kind, "sentinel": sentinel, "arm": "token_bos"}
    spec.update(extra)
    return spec


def _mk_token_obj(model, init_seed=3, cls=SoftTokenObject):
    g = torch.Generator().manual_seed(init_seed)
    init = torch.randn(HIDDEN, generator=g)
    return cls(model, FakeTokenizer(), sentinel=(SENT, SENT_ID), init=init)


# --------------------------------------------------------------------------
# pins against the real classes
# --------------------------------------------------------------------------


def test_kind_strings_pin_the_real_classes():
    assert st.SOFT_TOKEN_KINDS == (SoftTokenObject.kind,
                                   AppendedSoftTokenObject.kind)
    assert st.KIND_TOKEN == "token"
    assert st.KIND_TOKEN_APPENDED == "token_appended"


def test_inject_delegates_to_the_real_classes(tiny_model):
    """The serving-side splice IS the training-side splice: for every
    BOS text the class knows, and for the no-BOS prepend, and for the
    appended variant, inject_sentinel equals the real object's inject."""
    obj = _mk_token_obj(tiny_model)
    app = _mk_token_obj(tiny_model, cls=AppendedSoftTokenObject)
    for prompt in ("<|begin_of_text|>alpha beta", "<bos>alpha beta",
                   "<s>alpha beta", "alpha beta gamma"):
        assert st.inject_sentinel("token", SENT, prompt) == \
            obj.inject(prompt)
        assert st.inject_sentinel("token_appended", SENT, prompt) == \
            app.inject(prompt)
    # and the shapes are the documented ones
    assert st.inject_sentinel("token", SENT, "<s>alpha") == "<s>" + SENT + "alpha"
    assert st.inject_sentinel("token", SENT, "alpha") == SENT + "alpha"
    assert st.inject_sentinel("token_appended", SENT, "alpha") == "alpha" + SENT
    with pytest.raises(st.SoftTokenError, match="not a soft-token kind"):
        st.inject_sentinel("vector", SENT, "alpha")


# --------------------------------------------------------------------------
# artifact loading + validation
# --------------------------------------------------------------------------


def test_blob_roundtrip_for_both_kinds(tiny_model, tmp_path):
    for cls, kind in ((SoftTokenObject, "token"),
                      (AppendedSoftTokenObject, "token_appended")):
        obj = _mk_token_obj(tiny_model, cls=cls)
        art = st.load_soft_token_artifact(
            kind, _spec(kind), _blob(obj, tmp_path, f"{kind}.pt"),
            embedding_dim=HIDDEN)
        assert art.kind == kind
        assert art.sentinel_text == SENT
        assert art.param.dtype == torch.float32
        assert torch.equal(art.param, obj.param.detach().float())


def test_loader_refusals(tiny_model, tmp_path):
    obj = _mk_token_obj(tiny_model)
    blob = _blob(obj, tmp_path)

    with pytest.raises(st.SoftTokenError, match="not a soft-token kind"):
        st.load_soft_token_artifact("vector", {"kind": "vector"}, blob)
    with pytest.raises(st.SoftTokenError, match="no sidecar"):
        st.load_soft_token_artifact("token", _spec("token"), blob,
                                    b'{"stray": 1}')
    with pytest.raises(st.SoftTokenError, match="empty artifact"):
        st.load_soft_token_artifact("token", _spec("token"), b"")
    with pytest.raises(st.SoftTokenError, match="did not load"):
        st.load_soft_token_artifact("token", _spec("token"), b"not torch")
    # the wrong kind's artifact routed here (mis-mapped cell)
    with pytest.raises(st.SoftTokenError, match="stores kind"):
        st.load_soft_token_artifact("token_appended",
                                    _spec("token_appended"), blob)
    # F9-style cross-checks against the resolved spec
    with pytest.raises(st.SoftTokenError, match="cross-check"):
        st.load_soft_token_artifact("token", _spec("token", sentinel="<X>"),
                                    blob)
    with pytest.raises(st.SoftTokenError, match="cross-check"):
        st.load_soft_token_artifact("token", _spec("token", site="post"),
                                    blob)
    # a fit-lane site stamp is tolerated when it is the embedding site
    art = st.load_soft_token_artifact(
        "token", _spec("token", site=st.EMBEDDING_SITE), blob)
    assert art.sentinel_text == SENT
    # dimension check against the served model
    with pytest.raises(st.SoftTokenError, match="embedding dim"):
        st.load_soft_token_artifact("token", _spec("token"), blob,
                                    embedding_dim=HIDDEN + 1)


def _raw_blob(payload):
    buf = io.BytesIO()
    torch.save(payload, buf)
    return buf.getvalue()


def test_loader_refuses_malformed_blobs():
    ok = {"kind": "token", "param": torch.randn(HIDDEN), "sentinel": SENT}
    with pytest.raises(st.SoftTokenError, match="save schema"):
        st.load_soft_token_artifact("token", _spec("token"),
                                    _raw_blob({**ok, "extra": 1}))
    with pytest.raises(st.SoftTokenError, match="save schema"):
        st.load_soft_token_artifact(
            "token", _spec("token"),
            _raw_blob({"kind": "token", "sentinel": SENT}))
    with pytest.raises(st.SoftTokenError, match="'kind' field"):
        st.load_soft_token_artifact("token", _spec("token"),
                                    _raw_blob(torch.randn(3)))
    with pytest.raises(st.SoftTokenError, match="site"):
        st.load_soft_token_artifact(
            "token", _spec("token"), _raw_blob({**ok, "site": "post"}))
    # a blob-stamped embedding site is accepted
    art = st.load_soft_token_artifact(
        "token", _spec("token"),
        _raw_blob({**ok, "site": st.EMBEDDING_SITE}))
    assert torch.equal(art.param, ok["param"].float())
    with pytest.raises(st.SoftTokenError, match="non-empty string"):
        st.load_soft_token_artifact("token", _spec("token"),
                                    _raw_blob({**ok, "sentinel": ""}))
    with pytest.raises(st.SoftTokenError, match="1-D"):
        st.load_soft_token_artifact(
            "token", _spec("token"),
            _raw_blob({**ok, "param": torch.randn(2, HIDDEN)}))
    bad = torch.randn(HIDDEN)
    bad[3] = float("nan")
    with pytest.raises(st.SoftTokenError, match="non-finite"):
        st.load_soft_token_artifact("token", _spec("token"),
                                    _raw_blob({**ok, "param": bad}))
    with pytest.raises(st.SoftTokenError, match="non-floating"):
        st.load_soft_token_artifact(
            "token", _spec("token"),
            _raw_blob({**ok, "param": torch.ones(HIDDEN,
                                                 dtype=torch.int64)}))


def test_sentinel_must_be_a_single_token():
    class MultiIdSentinel(FakeTokenizer):
        def __call__(self, text, add_special_tokens=False):
            if text == SENT:
                return {"input_ids": [SENT_ID, 6]}
            return super().__call__(text, add_special_tokens)

    with pytest.raises(st.SoftTokenError, match="single token"):
        st.resolve_sentinel_id(MultiIdSentinel(), SENT)
    assert st.resolve_sentinel_id(FakeTokenizer(), SENT) == SENT_ID


# --------------------------------------------------------------------------
# THE equivalence: hooked forward == hook-free embeds forward, bit-exact
# --------------------------------------------------------------------------


def _forwards(model, *, input_ids=None, inputs_embeds=None, ctx=None,
              decode_id=42):
    """Prefill (+ hidden states) and one KV-cached decode step, under an
    optional object ctx (the hooked reference) or with no hook at all
    (the embeds path)."""
    from contextlib import nullcontext

    if input_ids is not None:
        T = input_ids.shape[1]
        kwargs = {"input_ids": input_ids}
    else:
        T = inputs_embeds.shape[1]
        kwargs = {"inputs_embeds": inputs_embeds}
    mask = torch.ones(1, T, dtype=torch.long)
    next_id = torch.tensor([[decode_id]])
    next_mask = torch.ones(1, T + 1, dtype=torch.long)
    with (ctx or nullcontext()), torch.no_grad():
        out = model(**kwargs, attention_mask=mask,
                    output_hidden_states=True, use_cache=True)
        dec = model(input_ids=next_id, attention_mask=next_mask,
                    past_key_values=out.past_key_values, use_cache=True)
    return out, dec


def _built_for(model, obj, raw, scale, tmp_path, *, tok=None,
               add_special_tokens=False):
    tok = tok or FakeTokenizer()
    art = st.load_soft_token_artifact(
        obj.kind, _spec(obj.kind), _blob(obj, tmp_path, f"{obj.kind}-eq.pt"),
        embedding_dim=HIDDEN)
    W = model.get_input_embeddings().weight.detach()
    return st.build_soft_token_embeds(art, tok, W, raw, scale=scale,
                                      add_special_tokens=add_special_tokens)


def _assert_bit_exact(model, obj, raw, scale, tmp_path, *,
                      add_special_tokens=False, expect_index):
    tok = FakeTokenizer()
    injected = obj.inject(raw)
    ids = torch.tensor(
        [tok(injected, add_special_tokens=add_special_tokens)["input_ids"]])
    out_h, dec_h = _forwards(model, input_ids=ids,
                             ctx=obj.apply(model, scale=scale))
    built = _built_for(model, obj, raw, scale, tmp_path, tok=tok,
                       add_special_tokens=add_special_tokens)
    assert built.token_ids == tuple(ids[0].tolist())  # located bookkeeping
    assert built.sentinel_index == expect_index
    out_e, dec_e = _forwards(model, inputs_embeds=built.embeds.unsqueeze(0))
    # non-vacuity (the fidelity discipline: vacuous evidence never
    # passes): the hooked forward must actually differ from the plain
    # model on the same ids, or this equality would prove nothing.
    out_plain, _ = _forwards(model, input_ids=ids)
    assert not torch.equal(out_h.logits, out_plain.logits), (
        "the hook did not fire — the equivalence below would be vacuous")
    assert torch.equal(out_h.logits, out_e.logits), (
        "prefill logits diverge: the embeds path is NOT the hooked "
        "forward")
    for i, (hh, he) in enumerate(zip(out_h.hidden_states,
                                     out_e.hidden_states)):
        assert torch.equal(hh, he), f"hidden state {i} diverges"
    assert torch.equal(dec_h.logits, dec_e.logits), (
        "KV-cached decode step diverges")
    return built


def test_after_bos_variant_matches_hooked_forward_bit_exactly(
        tiny_model, tmp_path):
    """kind 'token' (sentinel right after the BOS text), scale 1.3: the
    module's whole claim, on a real tiny Llama."""
    obj = _mk_token_obj(tiny_model)
    _assert_bit_exact(tiny_model, obj, BOS + "alpha beta gamma delta epsilon",
                      1.3, tmp_path, expect_index=1)


def test_appended_variant_matches_hooked_forward_bit_exactly(
        tiny_model, tmp_path):
    """kind 'token_appended' (Radevski [E_x, e_b]): sentinel at the FINAL
    prompt position, same bit-exact equivalence."""
    obj = _mk_token_obj(tiny_model, init_seed=11,
                        cls=AppendedSoftTokenObject)
    raw = BOS + "alpha beta gamma delta"
    built = _assert_bit_exact(tiny_model, obj, raw, 2.0, tmp_path,
                              expect_index=5)
    assert built.sentinel_index == len(built.token_ids) - 1


def test_tokenizer_added_bos_matches_hooked_forward_bit_exactly(
        tiny_model, tmp_path):
    """No BOS text in the prompt; the TOKENIZER adds one
    (add_special_tokens=True). inject prepends the sentinel to the
    string, so it lands right after the added BOS — located, cross-
    checked against tok('')'s prefix, and still bit-exact."""
    obj = _mk_token_obj(tiny_model, init_seed=17)
    _assert_bit_exact(tiny_model, obj, "alpha beta gamma zeta", 0.7,
                      tmp_path, add_special_tokens=True, expect_index=1)


# --------------------------------------------------------------------------
# scale semantics: the replacement, the zero pass-through, the honest no-op
# --------------------------------------------------------------------------


def test_scale_zero_is_the_unsteered_model_on_the_injected_prompt(
        tiny_model, tmp_path):
    """The class's scale semantics, reproduced exactly: scale 0 keeps
    the sentinel's NATIVE embedding (never zeros it), so the forward is
    bitwise the plain model on the INJECTED ids — and is NOT the raw
    prompt's forward, because the sentinel is still a real token in the
    context. No scale value reaches the raw-prompt baseline."""
    tok = FakeTokenizer()
    obj = _mk_token_obj(tiny_model)
    raw = BOS + "alpha beta gamma delta epsilon"
    injected = obj.inject(raw)
    ids = torch.tensor([tok(injected)["input_ids"]])
    raw_ids = torch.tensor([tok(raw)["input_ids"]])
    W = tiny_model.get_input_embeddings().weight.detach()

    built0 = _built_for(tiny_model, obj, raw, 0.0, tmp_path, tok=tok)
    # the sentinel row IS the embedding matrix's native row
    assert torch.equal(built0.embeds[built0.sentinel_index], W[SENT_ID])

    out_h0, _ = _forwards(tiny_model, input_ids=ids,
                          ctx=obj.apply(tiny_model, scale=0.0))
    out_e0, _ = _forwards(tiny_model, inputs_embeds=built0.embeds.unsqueeze(0))
    out_plain, _ = _forwards(tiny_model, input_ids=ids)  # no hook at all
    assert torch.equal(out_h0.logits, out_e0.logits)
    assert torch.equal(out_e0.logits, out_plain.logits)

    # the honest no-op fact: scale 0 != the raw (un-injected) prompt
    out_raw, _ = _forwards(tiny_model, input_ids=raw_ids)
    assert not torch.equal(out_e0.logits[0, -1], out_raw.logits[0, -1]), (
        "scale 0 unexpectedly equals the raw-prompt forward — the "
        "sentinel token should perturb attention; if this ever holds, "
        "the honest-no-op documentation is wrong and must be revisited")


def test_near_zero_scale_is_a_replacement_not_the_native_row(
        tiny_model, tmp_path):
    """The cliff at zero is the class's own: scale=1e-6 REPLACES the row
    with a near-zero vector — nothing like scale=0's native row."""
    obj = _mk_token_obj(tiny_model)
    raw = BOS + "alpha beta"
    built0 = _built_for(tiny_model, obj, raw, 0.0, tmp_path)
    built_eps = _built_for(tiny_model, obj, raw, 1e-6, tmp_path)
    k = built0.sentinel_index
    assert not torch.equal(built_eps.embeds[k], built0.embeds[k])
    assert torch.allclose(built_eps.embeds[k],
                          torch.zeros(HIDDEN), atol=1e-4)
    # and the exact replacement arithmetic at an ordinary scale
    built = _built_for(tiny_model, obj, raw, 1.3, tmp_path)
    art_param = obj.param.detach().float()
    assert torch.equal(built.embeds[k],
                       (1.3 * art_param).to(built.embeds.dtype))


def test_dtype_cast_order_matches_the_hook_exactly():
    """The hook computes scale*param IN FP32 and casts the PRODUCT to the
    embedding dtype. In bf16 the other order gives different bits — the
    built row must match the hooked embedding layer's output and the
    fp32-product cast, bitwise."""
    class _EmbOnly:
        def __init__(self, emb):
            self._emb = emb

        def get_input_embeddings(self):
            return self._emb

    torch.manual_seed(1)
    emb = torch.nn.Embedding(20, 8).to(torch.bfloat16)
    stub = _EmbOnly(emb)
    scale = 1.0 + 2**-9
    param = torch.full((8,), 1.0 + 2**-9, dtype=torch.float32)
    obj = SoftTokenObject(stub, FakeTokenizer(), sentinel=(SENT, SENT_ID),
                          init=param)

    art = st.SoftTokenArtifact(kind="token", sentinel_text=SENT,
                               param=param.clone())
    built = st.build_soft_token_embeds(
        art, FakeTokenizer(), emb.weight.detach(), BOS + "alpha beta",
        scale=scale, add_special_tokens=False)
    assert built.embeds.dtype == torch.bfloat16

    idx = torch.tensor([list(built.token_ids)])
    with obj.apply(stub, scale=scale), torch.no_grad():
        hooked = emb(idx)[0]
    assert torch.equal(hooked, built.embeds)

    right_order = (scale * param).to(torch.bfloat16)
    wrong_order = scale * param.to(torch.bfloat16)
    assert torch.equal(built.embeds[built.sentinel_index], right_order)
    assert not torch.equal(right_order, wrong_order), (
        "this pin needs a scale/param pair where the cast order matters")


# --------------------------------------------------------------------------
# the sentinel is LOCATED — and every corrupted splice refuses
# --------------------------------------------------------------------------


def _art(tiny_model, kind="token", cls=SoftTokenObject):
    obj = _mk_token_obj(tiny_model, cls=cls)
    return st.SoftTokenArtifact(kind=kind, sentinel_text=SENT,
                                param=obj.param.detach().float())


def _build(tiny_model, art, tok, raw=BOS + "alpha beta gamma", scale=1.0):
    W = tiny_model.get_input_embeddings().weight.detach()
    return st.build_soft_token_embeds(art, tok, W, raw, scale=scale,
                                      add_special_tokens=False)


def test_absent_sentinel_refuses(tiny_model):
    class DroppingTokenizer(FakeTokenizer):
        """Simulates a normalizer that strips the reserved token from
        running text (while still resolving it standalone)."""
        def __call__(self, text, add_special_tokens=False):
            out = super().__call__(text, add_special_tokens)
            if text != SENT:
                out["input_ids"] = [i for i in out["input_ids"]
                                    if i != SENT_ID]
            return out

    with pytest.raises(st.SoftTokenError, match="ABSENT"):
        _build(tiny_model, _art(tiny_model), DroppingTokenizer())


def test_duplicate_sentinel_refuses(tiny_model):
    # string level: the raw rendered prompt already carries the sentinel
    with pytest.raises(st.SoftTokenError, match="already contains the "
                                                "sentinel text"):
        _build(tiny_model, _art(tiny_model), FakeTokenizer(),
               raw=BOS + "alpha " + SENT + " beta")

    # id level: a tokenizer that emits the sentinel id twice
    class DuplicatingTokenizer(FakeTokenizer):
        def __call__(self, text, add_special_tokens=False):
            out = super().__call__(text, add_special_tokens)
            if SENT_ID in out["input_ids"] and text != SENT:
                out["input_ids"] = out["input_ids"] + [SENT_ID]
            return out

    with pytest.raises(st.SoftTokenError, match="appears 2 times"):
        _build(tiny_model, _art(tiny_model), DuplicatingTokenizer())


def test_sentinel_id_in_raw_tokenization_refuses(tiny_model):
    # a tokenizer that maps a natural word onto the sentinel's id
    words = dict(WORDS, weird=SENT_ID)
    tok = FakeTokenizer(words=words)
    with pytest.raises(st.SoftTokenError, match="already contains the "
                                                "sentinel id"):
        _build(tiny_model, _art(tiny_model), tok,
               raw=BOS + "alpha weird beta")


def test_mangled_neighbour_refuses(tiny_model):
    class MergingTokenizer(FakeTokenizer):
        """Simulates a splice that changes a neighbouring token."""
        def __call__(self, text, add_special_tokens=False):
            out = super().__call__(text, add_special_tokens)
            ids = out["input_ids"]
            if SENT_ID in ids and text != SENT:
                j = ids.index(SENT_ID)
                if j + 1 < len(ids):
                    ids[j + 1] = 96
            return out

    with pytest.raises(st.SoftTokenError, match="neighbouring tokens "
                                                "changed"):
        _build(tiny_model, _art(tiny_model), MergingTokenizer())


class _MovingTokenizer(FakeTokenizer):
    """Moves the sentinel id to a chosen end — a located position that
    CONTRADICTS the object's own expectation."""

    def __init__(self, to_end):
        super().__init__()
        self.to_end = to_end

    def __call__(self, text, add_special_tokens=False):
        out = super().__call__(text, add_special_tokens)
        ids = out["input_ids"]
        if SENT_ID in ids and text != SENT:
            ids = [i for i in ids if i != SENT_ID]
            ids = ids + [SENT_ID] if self.to_end else [SENT_ID] + ids
            out["input_ids"] = ids
        return out


def test_position_contradiction_refuses_for_the_after_bos_kind(tiny_model):
    with pytest.raises(st.SoftTokenError, match="not immediately after "
                                                "the BOS region"):
        _build(tiny_model, _art(tiny_model), _MovingTokenizer(to_end=True))


def test_position_contradiction_refuses_for_the_appended_kind(tiny_model):
    art = _art(tiny_model, kind="token_appended",
               cls=AppendedSoftTokenObject)
    with pytest.raises(st.SoftTokenError, match="FINAL prompt position"):
        _build(tiny_model, art, _MovingTokenizer(to_end=False),
               raw="alpha beta gamma")


def test_build_input_refusals(tiny_model):
    art = _art(tiny_model)
    tok = FakeTokenizer()
    W = tiny_model.get_input_embeddings().weight.detach()
    raw = BOS + "alpha beta"
    for bad_scale in (float("nan"), float("inf"), True, "1.0"):
        with pytest.raises(st.SoftTokenError, match="finite number"):
            st.build_soft_token_embeds(art, tok, W, raw, scale=bad_scale,
                                       add_special_tokens=False)
    with pytest.raises(st.SoftTokenError, match="2-D"):
        st.build_soft_token_embeds(art, tok, W[0], raw, scale=1.0,
                                   add_special_tokens=False)
    with pytest.raises(st.SoftTokenError, match="param dim"):
        st.build_soft_token_embeds(art, tok, W[:, : HIDDEN - 1], raw,
                                   scale=1.0, add_special_tokens=False)
    with pytest.raises(st.SoftTokenError, match="non-floating"):
        st.build_soft_token_embeds(
            art, tok, torch.ones(97, HIDDEN, dtype=torch.int64), raw,
            scale=1.0, add_special_tokens=False)
    with pytest.raises(st.SoftTokenError, match="non-empty string"):
        st.build_soft_token_embeds(art, tok, W, "", scale=1.0,
                                   add_special_tokens=False)
    # token ids past the matrix's vocab refuse (tokenizer/matrix mismatch)
    with pytest.raises(st.SoftTokenError, match="out of range"):
        st.build_soft_token_embeds(art, tok, W[:6], raw, scale=1.0,
                                   add_special_tokens=False)
    with pytest.raises(st.SoftTokenError, match="SoftTokenArtifact"):
        st.build_soft_token_embeds({"param": art.param}, tok, W, raw,
                                   scale=1.0, add_special_tokens=False)


# --------------------------------------------------------------------------
# submission: the engine flag is CHECKED, and both request shapes are exact
# --------------------------------------------------------------------------


def test_engine_flag_is_checked_not_assumed():
    st.require_prompt_embeds_enabled(ENGINE_ON)
    st.require_prompt_embeds_enabled(["--model", "m",
                                      "--enable-prompt-embeds"])
    st.require_prompt_embeds_enabled(["--enable-prompt-embeds=true"])
    for bad in ({}, {"enable_prompt_embeds": False},
                {"enable_prompt_embeds": "yes"},  # exactly True, not truthy
                ["--model", "m"], ["--enable-prompt-embeds=false"]):
        with pytest.raises(st.SoftTokenError,
                           match="enable.prompt.embeds"):
            st.require_prompt_embeds_enabled(bad)
    with pytest.raises(st.SoftTokenError, match="engine-arg mapping"):
        st.require_prompt_embeds_enabled("--enable-prompt-embeds")


def test_offline_pure_prompt_shape(tiny_model, tmp_path):
    obj = _mk_token_obj(tiny_model)
    built = _built_for(tiny_model, obj, BOS + "alpha beta gamma", 1.3,
                       tmp_path)
    with pytest.raises(st.SoftTokenError, match="enable.prompt.embeds"):
        st.to_embeds_prompt(built, engine_args={})
    prompt = st.to_embeds_prompt(built, engine_args=ENGINE_ON)
    assert set(prompt) == {"prompt_embeds"}
    assert torch.equal(prompt["prompt_embeds"], built.embeds)
    salted = st.to_embeds_prompt(built, engine_args=ENGINE_ON,
                                 cache_salt="arm-x")
    assert salted["cache_salt"] == "arm-x"
    with pytest.raises(st.SoftTokenError, match="cache_salt"):
        st.to_embeds_prompt(built, engine_args=ENGINE_ON, cache_salt="")


@pytest.mark.parametrize("scale", [1.3, 0.0])
def test_offline_mixed_prompt_recombines_to_the_exact_sequence(
        tiny_model, tmp_path, scale):
    """The mixed form (ids + mask + sentinel row): recombined the way
    the v0.25.1 runner does (engine-side embeddings at True positions,
    client rows at False ones — gpu_model_runner.py:3540-3548), it must
    equal the pure sequence bit-exactly, at a served scale AND at the
    scale-0 leg (where the client row is the native one)."""
    obj = _mk_token_obj(tiny_model)
    built = _built_for(tiny_model, obj, BOS + "alpha beta gamma delta",
                       scale, tmp_path)
    prompt = st.to_embeds_prompt(built, engine_args=ENGINE_ON, mixed=True)
    assert set(prompt) == {"prompt_embeds", "prompt_token_ids",
                           "prompt_is_token_ids"}
    assert prompt["prompt_token_ids"] == list(built.token_ids)
    k = built.sentinel_index
    mask = prompt["prompt_is_token_ids"]
    assert mask.count(False) == 1 and mask[k] is False
    # non-sentinel rows are deterministic zero filler (stable cache hashes)
    zeros = torch.zeros(HIDDEN, dtype=built.embeds.dtype)
    for j in range(len(mask)):
        if j != k:
            assert torch.equal(prompt["prompt_embeds"][j], zeros)
    # the engine's documented recombination
    W = tiny_model.get_input_embeddings().weight.detach()
    ids_t = torch.tensor(prompt["prompt_token_ids"])
    mask_t = torch.tensor(mask)
    engine_side = torch.where(mask_t.unsqueeze(-1), W[ids_t],
                              prompt["prompt_embeds"])
    assert torch.equal(engine_side, built.embeds)


def test_openai_payload_roundtrips_the_exact_bytes(tiny_model, tmp_path):
    obj = _mk_token_obj(tiny_model)
    built = _built_for(tiny_model, obj, BOS + "alpha beta", 1.3, tmp_path)
    with pytest.raises(st.SoftTokenError, match="enable.prompt.embeds"):
        st.encode_prompt_embeds_for_openai(built, engine_args=[])
    payload = st.encode_prompt_embeds_for_openai(built,
                                                 engine_args=ENGINE_ON)
    assert isinstance(payload, str)
    # the server's decode (renderers/embed_utils.py at v0.25.1):
    # b64decode -> torch.load(weights_only=True)
    decoded = torch.load(io.BytesIO(base64.b64decode(payload,
                                                     validate=True)),
                         weights_only=True)
    assert torch.equal(decoded, built.embeds)
    assert decoded.dim() == 2 and decoded.shape[1] == HIDDEN


# --------------------------------------------------------------------------
# the lens path still refuses these kinds — the dict now names this module
# --------------------------------------------------------------------------


def test_lens_path_refusal_names_the_committed_embeds_path():
    """token stays UNSERVABLE on the LENS path (it is still not a
    block-output op) — but its refusal now points at the committed
    prompt-embeds path, the same shape as lora's entry pointing at the
    adapter export."""
    with pytest.raises(la.AdapterError, match="soft_token"):
        la.get_adapter("token")
    with pytest.raises(la.AdapterError, match="prompt-embeds"):
        la.get_adapter("token")
    # and an adapter registration for it is still refused outright
    with pytest.raises(la.AdapterError, match="UNSERVABLE"):
        la.register_adapter("token", lambda *a, **k: ())
