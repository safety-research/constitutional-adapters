import pytest
import torch

from sisteer.hooks import steering
from sisteer.models import hidden_size, num_layers
from sisteer.vectors import SteeringVector


def _logits(model, tok, text, **steer_kwargs):
    enc = tok(text, return_tensors="pt")
    if steer_kwargs.get("sv") is not None or "sv" in steer_kwargs:
        sv = steer_kwargs.pop("sv")
        with steering(model, sv, **steer_kwargs) as h:
            if h is not None and steer_kwargs.get("positions") in (
                "response", "prompt", "front",
            ):
                h.reset()
            return model(**enc).logits.detach()
    return model(**enc).logits.detach()


def test_zero_vector_is_noop(tiny_llama):
    model, tok = tiny_llama
    d = hidden_size(model)
    base = _logits(model, tok, "The capital of France is")
    steered = _logits(
        model, tok, "The capital of France is", sv={1: torch.zeros(d)}
    )
    assert torch.allclose(base, steered)


def test_nonzero_vector_changes_logits(tiny_llama):
    model, tok = tiny_llama
    d = hidden_size(model)
    torch.manual_seed(0)
    vec = torch.randn(d)
    base = _logits(model, tok, "The capital of France is")
    steered = _logits(model, tok, "The capital of France is", sv={1: vec}, scale=5.0)
    assert not torch.allclose(base, steered)


def test_scale_zero_is_noop(tiny_llama):
    model, tok = tiny_llama
    d = hidden_size(model)
    vec = torch.randn(d)
    base = _logits(model, tok, "hello world")
    steered = _logits(model, tok, "hello world", sv={1: vec}, scale=0.0)
    assert torch.allclose(base, steered)


def test_sv_none_passthrough(tiny_llama):
    model, tok = tiny_llama
    base = _logits(model, tok, "hello")
    with steering(model, None) as h:
        assert h is None
        enc = tok("hello", return_tensors="pt")
        out = model(**enc).logits
    assert torch.allclose(base, out)


def test_prompt_mode_masks_response_positions(tiny_llama):
    """prompt mode: positions >= prompt_len are NOT steered, so the logits
    at later positions differ only via attention to steered prefix."""
    model, tok = tiny_llama
    d = hidden_size(model)
    text = "one two three four five six"
    enc = tok(text, return_tensors="pt")
    n = enc["input_ids"].shape[1]
    vec = torch.randn(d)

    with steering(model, {1: vec}, scale=3.0, positions="response",
                  prompt_len=n) as h:
        h.reset()
        # every position < prompt_len -> nothing steered at all
        steered = model(**enc).logits
    base = model(**enc).logits
    assert torch.allclose(base, steered)

    with steering(model, {1: vec}, scale=3.0, positions="response",
                  prompt_len=2) as h:
        h.reset()
        steered2 = model(**enc).logits
    # positions before prompt_len untouched (causal: logits at pos 0,1
    # depend only on unsteered positions)
    assert torch.allclose(base[:, :2], steered2[:, :2])
    assert not torch.allclose(base[:, 2:], steered2[:, 2:])


def test_skip_bos(tiny_llama):
    model, tok = tiny_llama
    d = hidden_size(model)
    vec = torch.randn(d)
    enc = tok("alpha beta gamma", return_tensors="pt")
    base = model(**enc).logits
    with steering(model, {1: vec}, scale=3.0, skip_bos=True) as h:
        h.reset()
        steered = model(**enc).logits
    # BOS position output depends only on BOS residual (causal), which is
    # unsteered under skip_bos.
    assert torch.allclose(base[:, 0], steered[:, 0])
    assert not torch.allclose(base[:, 1:], steered[:, 1:])


def test_position_tracking_across_incremental_forwards(tiny_llama):
    """Simulate KV-cached generation: full-prefix forward then 1-token
    forwards; response-mode must start steering exactly at prompt_len."""
    model, tok = tiny_llama
    d = hidden_size(model)
    vec = torch.randn(d) * 5
    enc = tok("a b c d", return_tensors="pt")
    n = enc["input_ids"].shape[1]

    with steering(model, {1: vec}, positions="response", prompt_len=n) as h:
        h.reset()
        out = model(**enc, use_cache=True)
        prefill_logits = out.logits
        past = out.past_key_values
        # next token: global position n -> steered
        nxt = torch.tensor([[enc["input_ids"][0, -1]]])
        step_logits = model(input_ids=nxt, past_key_values=past).logits

    base_out = model(**enc, use_cache=True)
    assert torch.allclose(prefill_logits, base_out.logits)  # prefill unsteered
    base_step = model(
        input_ids=nxt, past_key_values=base_out.past_key_values
    ).logits
    assert not torch.allclose(step_logits, base_step)  # decode step steered


def test_position_tracker_at_128k_prompt_lens(tiny_llama):
    """The KV-cache position tracker is a pure token counter, so a
    131k-token prefill under positions='response' must (a) mask the WHOLE
    prefill and (b) steer the first decode steps at global index
    prompt_len — asserted on the tracker + mask directly (a real 131k CPU
    forward would prove nothing more, at silly cost); the real-forward
    semantics are covered at small n by
    test_position_tracking_across_incremental_forwards."""
    from sisteer.hooks import SteeringHandle

    model, _tok = tiny_llama
    d = hidden_size(model)
    n = 131072
    handle = SteeringHandle(model, {1: torch.randn(d)}, positions="response")
    try:
        handle.reset(prompt_len=n)
        # the prefill forward reaching the layer-0 tracker
        handle._track_positions(None, (),
                                {"hidden_states": torch.zeros(1, n, 1)})
        mask = handle._position_mask(1, n, torch.device("cpu"))
        assert mask.shape == (1, n)
        assert float(mask.sum()) == 0.0  # entire 131k prefill unsteered
        # decode steps: global positions n, n+1 -> steered
        for _step in range(2):
            handle._track_positions(None, (),
                                    {"hidden_states": torch.zeros(1, 1, 1)})
            step_mask = handle._position_mask(1, 1, torch.device("cpu"))
            assert float(step_mask.sum()) == 1.0
        assert handle._seen == n + 2
    finally:
        handle.remove()


def test_front_per_position_injection(tiny_llama):
    """[P, d] vectors touch only the first P positions (ActAdd)."""
    model, tok = tiny_llama
    d = hidden_size(model)
    P = 3
    mat = torch.randn(P, d) * 3
    enc = tok("one two three four five six seven", return_tensors="pt")
    base = model(**enc).logits
    with steering(model, {1: mat}, positions="front") as h:
        h.reset()
        steered = model(**enc).logits
    # causal: outputs strictly before the first steered position... position 0
    # IS steered here, so all outputs may change; instead verify decode-step
    # behavior: after prefill, incremental steps get nothing added.
    assert not torch.allclose(base, steered)
    with steering(model, {1: mat}, positions="front") as h:
        h.reset()
        out = model(**enc, use_cache=True)
        nxt = torch.tensor([[enc["input_ids"][0, -1]]])
        step = model(input_ids=nxt, past_key_values=out.past_key_values).logits
    with steering(model, {1: mat}, positions="front") as h:
        h.reset()
        out2 = model(**enc, use_cache=True)
    base_step = model(
        input_ids=nxt, past_key_values=out2.past_key_values
    ).logits
    assert torch.allclose(step, base_step)  # offsets >= P: no addition


def test_projection_targeting(tiny_llama):
    """Stolfo Eq. 2 mode: projections onto u are moved to the target."""
    model, tok = tiny_llama
    from sisteer.extraction.capture import capture_layer_outputs

    d = hidden_size(model)
    torch.manual_seed(1)
    vec = torch.randn(d)
    u = vec / vec.norm()
    target = 7.0
    enc = tok("check the projection please", return_tensors="pt")
    with steering(model, {1: vec}, project_targets={1: target}):
        with capture_layer_outputs(model, [1]) as cap:
            model(**enc)
    proj = (cap[1][0].float() @ u).squeeze()
    assert torch.allclose(proj, torch.full_like(proj, target), atol=1e-3)


def test_norm_relative_scaling(tiny_llama):
    model, tok = tiny_llama
    from sisteer.extraction.capture import capture_layer_outputs

    d = hidden_size(model)
    vec = torch.randn(d)
    enc = tok("norm relative test", return_tensors="pt")
    with capture_layer_outputs(model, [1]) as cap_base:
        model(**enc)
    with steering(model, {1: vec}, scale=0.5, norm_relative=True):
        with capture_layer_outputs(model, [1]) as cap:
            model(**enc)
    diff = (cap[1] - cap_base[1])[0]
    h_norm = cap_base[1][0].norm(dim=-1)
    # addition magnitude per position == scale * ||h||
    assert torch.allclose(diff.norm(dim=-1), 0.5 * h_norm, rtol=0.15)


def test_position_modes_include_last():
    from sisteer.hooks import POSITION_MODES

    assert POSITION_MODES == ("all", "response", "prompt", "front", "last")


def test_last_mode_prefill_touches_only_final_prompt_token(tiny_llama):
    """Jailbreak Antidote: during a multi-token prefill, "last" steers
    exactly the final prompt token — every earlier position's logits are
    BIT-IDENTICAL to the unsteered forward (causal attention: positions
    before the last never see it; the mask zeroes the addition exactly)."""
    model, tok = tiny_llama
    d = hidden_size(model)
    torch.manual_seed(0)
    vec = torch.randn(d) * 5
    enc = tok("one two three four five", return_tensors="pt")
    base = model(**enc).logits.detach()
    with steering(model, {1: vec}, positions="last") as h:
        h.reset()
        steered = model(**enc).logits.detach()
    assert torch.equal(base[:, :-1], steered[:, :-1])  # bit-identical
    assert not torch.allclose(base[:, -1], steered[:, -1])


def test_last_mode_decode_steers_every_incremental_step(tiny_llama):
    """Under KV-cached generation every 1-token forward's single position
    IS the current last token, so every decode step is steered."""
    model, tok = tiny_llama
    d = hidden_size(model)
    torch.manual_seed(0)
    vec = torch.randn(d) * 5
    enc = tok("a b c d", return_tensors="pt")
    nxt = torch.tensor([[enc["input_ids"][0, -1]]])

    with steering(model, {1: vec}, positions="last") as h:
        h.reset()
        out = model(**enc, use_cache=True)
        step_steered = model(
            input_ids=nxt, past_key_values=out.past_key_values
        ).logits
    # identical steered prefill, but the STEP itself unsteered: the
    # difference isolates the per-step addition
    with steering(model, {1: vec}, positions="last") as h:
        h.reset()
        out2 = model(**enc, use_cache=True)
    step_plain = model(
        input_ids=nxt, past_key_values=out2.past_key_values
    ).logits
    assert not torch.allclose(step_steered, step_plain)


def test_last_mode_mask_uses_attention_mask_per_row(tiny_llama):
    """Batched prefill: the per-row last REAL token comes from the 2D
    padding mask — right padding lands on each row's own length-1, left
    padding on seq_len-1; without a usable mask, seq_len-1."""
    from sisteer.hooks import SteeringHandle

    model, _tok = tiny_llama
    d = hidden_size(model)
    handle = SteeringHandle(model, {1: torch.randn(d)}, positions="last")
    try:
        dev = torch.device("cpu")
        # multi-token forward reaching the layer-0 tracker
        handle._track_positions(None, (),
                                {"hidden_states": torch.zeros(2, 4, 1)})
        # right padding: row0 has 3 real tokens, row1 has 4
        handle._attn_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
        mask = handle._position_mask(2, 4, dev)
        assert mask.tolist() == [[0, 0, 1, 0], [0, 0, 0, 1]]
        # left padding: both rows end at index 3
        handle._attn_mask = torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]])
        mask = handle._position_mask(2, 4, dev)
        assert mask.tolist() == [[0, 0, 0, 1], [0, 0, 0, 1]]
        # no usable mask -> seq_len - 1 fallback
        handle._attn_mask = None
        mask = handle._position_mask(2, 4, dev)
        assert mask.tolist() == [[0, 0, 0, 1], [0, 0, 0, 1]]
        # decode step: the single position is steered
        handle._track_positions(None, (),
                                {"hidden_states": torch.zeros(2, 1, 1)})
        mask = handle._position_mask(2, 1, dev)
        assert mask.tolist() == [[1.0], [1.0]]
    finally:
        handle.remove()


def test_last_mode_rejects_full_sequence_training(tiny_llama):
    """positions='last' has no meaning for a full-sequence training
    forward — set_prompt_lens must fail LOUDLY."""
    from sisteer.hooks import SteeringHandle

    model, _tok = tiny_llama
    d = hidden_size(model)
    handle = SteeringHandle(model, {1: torch.randn(d)}, positions="last")
    try:
        with pytest.raises(RuntimeError, match="full-sequence training"):
            handle.set_prompt_lens(torch.tensor([3]))
        # the belt inside the mask path fires too
        handle._prompt_lens = torch.tensor([3])
        with pytest.raises(RuntimeError, match="per-sample prompt"):
            handle._position_mask(1, 4, torch.device("cpu"))
    finally:
        handle.remove()


def test_last_mode_rejects_per_position_matrix(tiny_llama):
    """A [P, d] ActAdd-style matrix bypasses the position mask — 'last'
    must refuse it rather than silently front-inject."""
    from sisteer.hooks import SteeringHandle

    model, _tok = tiny_llama
    d = hidden_size(model)
    with pytest.raises(ValueError, match="per-position"):
        SteeringHandle(model, {1: torch.randn(3, d)}, positions="last")


def test_vector_roundtrip(tmp_path, tiny_llama):
    model, _ = tiny_llama
    d = hidden_size(model)
    sv = SteeringVector(
        vectors={1: torch.randn(d), 2: torch.randn(d)},
        method="caa",
        model_name="tiny",
        instruction_id="french",
        metadata={"note": "test", "diagnostics": {"1": {"d_prime": 1.0}}},
    )
    sv.save(tmp_path / "v.safetensors")
    loaded = SteeringVector.load(tmp_path / "v.safetensors")
    assert loaded.layers == [1, 2]
    assert torch.allclose(loaded.vectors[1], sv.vectors[1])
    assert loaded.metadata["note"] == "test"
    assert loaded.instruction_id == "french"


def test_num_layers_resolution(tiny_llama):
    model, _ = tiny_llama
    assert num_layers(model) >= 2
