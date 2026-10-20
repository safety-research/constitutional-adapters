"""T12 — model-onboarding accessors and the hybrid-cache contract.

No downloads anywhere in this file: accessor dispatch runs on SYNTHETIC
wrapper modules shaped like the real architectures, and the prefix-cache
checks run on TINY random-weight instances of the REAL transformers
classes built from in-code configs (from_config fetches nothing). The two
GLM-4.6V-Flash checks at the bottom that touch the REAL tokenizer/config
read the pinned revision from the local HF cache only and skip loudly
when it is absent (the tests/test_glm_chat_markers.py convention).

The prefix-cache tests pin a CONTRACT, not today's exception messages:
TeacherPrefixCache must never SILENTLY return wrong logits on a hybrid
cache layout — either it raises (today's measured behavior: qwen3_5's
LinearAttentionLayer entries at any prefix length; gemma4's sliding-window
crop once the prefix exceeds the window) or its logits match the uncached
forward. If hybrid support lands in extraction/prefix_cache.py, these
tests flip to enforcing the equivalence with no edit.
"""

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from sisteer.models import get_decoder_layers, hidden_size

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

GEMMA_CFG = REPO / "configs" / "two_si_gemma4e4b.yaml"

# GLM-4.6V-Flash pin (must equal MODEL_SPECS["glm46vflash"] — asserted
# below, so these literals cannot drift from the registry).
GLM_MODEL = "zai-org/GLM-4.6V-Flash"
GLM_REV = "411bb4d77144a3f03accbf4b780f5acb8b7cde4e"


# ---- synthetic wrapper dispatch (the pure accessor contract) -------------------


class _Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _layers(n=4):
    return nn.ModuleList([nn.Linear(8, 8) for _ in range(n)])


def test_conditional_generation_nesting():
    """gemma4/qwen3_5 ConditionalGeneration shape: the text decoder sits at
    model.<model>.language_model.layers and the dims under
    config.text_config — models.py must find both with no model-specific
    branch."""
    wrapper = nn.Module()
    wrapper.model = nn.Module()
    wrapper.model.language_model = nn.Module()
    wrapper.model.language_model.layers = _layers(42)
    wrapper.config = _Cfg(text_config=_Cfg(hidden_size=2560))
    assert len(get_decoder_layers(wrapper)) == 42
    assert hidden_size(wrapper) == 2560


def test_text_causal_lm_nesting():
    """qwen3_5 ForCausalLM shape (what AutoModelForCausalLM actually
    resolves qwen3_5 to at transformers 5.14.1): plain model.layers +
    top-level hidden_size."""
    wrapper = nn.Module()
    wrapper.model = nn.Module()
    wrapper.model.layers = _layers(64)
    wrapper.config = _Cfg(hidden_size=5120)
    assert len(get_decoder_layers(wrapper)) == 64
    assert hidden_size(wrapper) == 5120


def test_legacy_language_model_nestings():
    """The older multimodal spellings stay dispatchable (they are in
    _LAYER_PATHS on purpose)."""
    a = nn.Module()
    a.language_model = nn.Module()
    a.language_model.layers = _layers(3)
    assert len(get_decoder_layers(a)) == 3
    b = nn.Module()
    b.language_model = nn.Module()
    b.language_model.model = nn.Module()
    b.language_model.model.layers = _layers(5)
    assert len(get_decoder_layers(b)) == 5


def test_unknown_structure_fails_loudly():
    stranger = nn.Module()
    stranger.config = _Cfg()
    with pytest.raises(ValueError, match="decoder layers"):
        get_decoder_layers(stranger)
    with pytest.raises(ValueError, match="hidden_size"):
        hidden_size(stranger)


# ---- real-class tiny instances (from_config — no downloads) --------------------


def _tiny_qwen35():
    from transformers import AutoModelForCausalLM
    from transformers.models.qwen3_5 import Qwen3_5Config

    text_cls = type(Qwen3_5Config().get_text_config())
    cfg = text_cls(hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                   num_attention_heads=4, num_key_value_heads=2,
                   vocab_size=128, max_position_embeddings=512)
    m = AutoModelForCausalLM.from_config(cfg)
    m.eval()
    return m, cfg


def _tiny_gemma4(sliding_window: int):
    from transformers import AutoModelForCausalLM
    from transformers.models.gemma4 import Gemma4Config

    text_cls = type(Gemma4Config().get_text_config())
    cfg = text_cls(hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                   num_attention_heads=4, num_key_value_heads=2,
                   vocab_size=128, max_position_embeddings=1024,
                   sliding_window=sliding_window)
    m = AutoModelForCausalLM.from_config(cfg)
    m.eval()
    return m, cfg


def test_real_class_accessors_tiny():
    """The dispatch holds on the REAL classes, not just the fakes."""
    q, _ = _tiny_qwen35()
    assert type(q).__name__ == "Qwen3_5ForCausalLM"
    assert len(get_decoder_layers(q)) == 4
    assert hidden_size(q) == 32
    g, gcfg = _tiny_gemma4(sliding_window=64)
    assert len(get_decoder_layers(g)) == 4
    assert hidden_size(g) == 32
    # the hybrid facts the cache contract below rests on
    assert "linear_attention" in getattr(
        _tiny_qwen35()[1], "layer_types", [])
    assert "sliding_attention" in getattr(gcfg, "layer_types", [])


def _cache_never_silently_wrong(model, P, S=6, B=2, must_pass=False):
    """Either TeacherPrefixCache raises (loud), or its suffix logits match
    the uncached forward. must_pass=True additionally requires the
    equivalence branch (no exception tolerated)."""
    from sisteer.extraction.prefix_cache import TeacherPrefixCache

    torch.manual_seed(0)
    prefix = torch.randint(5, 128, (P,))
    suffix = torch.randint(5, 128, (B, S))
    ids = torch.cat([prefix.unsqueeze(0).expand(B, -1), suffix], dim=1)
    mask = torch.ones_like(ids)
    try:
        cache = TeacherPrefixCache.build(model, prefix)
        got = cache.forward_ids(model, ids, mask, P)
    except Exception as e:  # noqa: BLE001 — loud IS the accepted branch
        if must_pass:
            raise AssertionError(
                f"prefix cache must be exact in this regime but raised "
                f"{type(e).__name__}: {e}") from e
        return "raised"
    with torch.no_grad():
        want = model(input_ids=ids, attention_mask=mask).logits[:, P:]
    assert torch.allclose(got, want, atol=1e-4), (
        "prefix cache returned well-formed but WRONG logits — the silent "
        "failure mode this contract exists to forbid "
        f"(max|diff| {(got - want).abs().max().item():.3e})")
    return "equivalent"


def test_prefix_cache_qwen35_hybrid_contract():
    """qwen3_5's cache mixes LinearAttentionLayer (recurrent state, no
    K/V) with DynamicLayer — replay must refuse or be exact, never
    silently wrong. Measured 2026-08-25: raises AttributeError('keys')."""
    m, _ = _tiny_qwen35()
    _cache_never_silently_wrong(m, P=20)


def test_prefix_cache_gemma4_beyond_window_contract():
    """gemma4 sliding layers store only the last `sliding_window` K/V, so
    a prefix past the window (the real E4B regime: SIs render ~6k tokens
    against window 512, 35/42 layers sliding) cannot replay from the plain
    crop. Refuse or be exact — measured 2026-08-25: raises RuntimeError
    (size mismatch)."""
    m, _ = _tiny_gemma4(sliding_window=8)
    _cache_never_silently_wrong(m, P=20)


def test_prefix_cache_gemma4_within_window_exact():
    """Below the window every layer holds full K/V, and the replay must be
    EXACT (the safety_refusal regime: a ~101-token prefix under window
    512). This is the branch that must not regress if hybrid support
    lands."""
    m, _ = _tiny_gemma4(sliding_window=64)
    assert _cache_never_silently_wrong(m, P=20, must_pass=True) == "equivalent"


# ---- GLM-4.6V-Flash (glm4v): a full-attention VLM wrapper, text-only ------------


def _tiny_glm4v():
    """A tiny Glm4vForConditionalGeneration from in-code configs (no
    downloads). M-RoPE constrains the shape: mrope_section must sum to the
    rotary-frequency width (head_dim x partial_rotary_factor / 2), so the
    text config pins head_dim 32 (hidden 64 / 2 heads) and partial factor
    0.5 -> 16 rotary dims -> 8 frequencies = [2, 3, 3]. The default
    section ([8, 12, 12] = the real model's 128-dim heads) cannot fit a
    tiny head."""
    from transformers import AutoModelForImageTextToText
    from transformers.models.glm4v import Glm4vConfig

    cfg = Glm4vConfig(
        text_config=dict(hidden_size=64, intermediate_size=128,
                         num_hidden_layers=4, num_attention_heads=2,
                         num_key_value_heads=1, vocab_size=160,
                         max_position_embeddings=512,
                         rope_parameters={"rope_type": "default",
                                          "rope_theta": 10000.0,
                                          "mrope_section": [2, 3, 3],
                                          "partial_rotary_factor": 0.5}),
        vision_config=dict(hidden_size=32, depth=1, num_heads=2,
                           intermediate_size=64, out_hidden_size=64,
                           patch_size=14, image_size=28,
                           spatial_merge_size=2),
    )
    m = AutoModelForImageTextToText.from_config(cfg)
    m.eval()
    return m, cfg


def test_glm4v_real_class_accessors_tiny():
    """glm4v loads ONLY as Glm4vForConditionalGeneration (no causal-LM
    auto-mapping in transformers 5.14.1 — models.auto_model_class's whole
    reason to exist); its text decoder sits at model.language_model.layers
    with the dims under text_config, and EVERY layer carries
    self_attn.o_proj (full attention — no hybrid subset, unlike qwen3_5).
    Text-only use: a plain input_ids forward (no pixel inputs) runs — the
    M-RoPE positions for pure text are the ordinary arange."""
    from sisteer.models import oproj_layers

    m, _ = _tiny_glm4v()
    assert type(m).__name__ == "Glm4vForConditionalGeneration"
    assert len(get_decoder_layers(m)) == 4
    assert hidden_size(m) == 64
    assert oproj_layers(m) == [0, 1, 2, 3]
    ids = torch.randint(5, 150, (2, 12))
    with torch.no_grad():
        out = m(input_ids=ids, attention_mask=torch.ones_like(ids))
    assert tuple(out.logits.shape) == (2, 12, 160)


def test_prefix_cache_glm4v_full_attention_exact():
    """GLM-4.6V-Flash is full attention on every layer with a standard
    DynamicCache — neither measured hybrid limitation applies — so the
    prefix replay must be EXACT (measured 2026-09-02: max|dlogit| ~2e-7
    on this tiny instance). This is the CPU evidence a per-model
    teacher_prefix_cache: true flip would rest on; the derived config
    keeps the generator's uniform false until the onboarding equivalence
    gate confirms it on the real weights (the standing rule in
    scripts/derive_model_config.py's docstring)."""
    m, _ = _tiny_glm4v()
    assert _cache_never_silently_wrong(m, P=20, must_pass=True) == "equivalent"


def _divergent_prefix_cache(monkeypatch, delta: float = 0.5):
    """Stand in for TeacherPrefixCache with a replay that returns the
    uncached suffix logits PLUS a constant offset on one vocab slot — a
    well-formed but non-equivalent replay, the shape of what the real
    GLM-4.6V-Flash weights measured on the H200."""
    from sisteer.extraction import prefix_cache as pc

    class Divergent:
        @classmethod
        def build(cls, model, prefix):
            return cls()

        def forward_ids(self, model, ids, mask, P):
            with torch.no_grad():
                want = model(input_ids=ids, attention_mask=mask).logits[:, P:]
            got = want.clone()
            got[..., 7] += delta
            return got

    monkeypatch.setattr(pc, "TeacherPrefixCache", Divergent)


def test_prefix_cache_nonequivalence_known_limit_only_for_glm4v(monkeypatch):
    """The real-weights onboarding of GLM-4.6V-Flash (Modal H200, bf16,
    2026-09-02) measured a NON-equivalent replay (KL 1.19e-2 > 2e-3)
    while the tiny fp32 instance above replays exactly — cause not
    isolated. With teacher_prefix_cache: false already in the config
    that is a KNOWN-LIMIT scoped to the glm4v family; the same
    non-equivalence on a glm4v config that ENABLES the cache, or on any
    other family, stays a gate FAILURE."""
    from onboard_model_smoke import check_prefix_cache

    _divergent_prefix_cache(monkeypatch)
    glm, _ = _tiny_glm4v()
    got = check_prefix_cache(glm, None, None, {"teacher_prefix_cache": False})
    assert got[0] == "KNOWN-LIMIT" and "glm4v family" in got[1]
    with pytest.raises(AssertionError, match="NOT equivalent"):
        check_prefix_cache(glm, None, None, {"teacher_prefix_cache": True})
    q, _ = _tiny_qwen35()
    with pytest.raises(AssertionError, match="NOT equivalent"):
        check_prefix_cache(q, None, None, {"teacher_prefix_cache": False})


# ---- NemotronH (Nemotron-3-Nano-4B) hybrid-cache contract ---------------------
#
# The architecture seams themselves (module tree, attention-mixer o_proj
# probe, LoRA shapes, the use_cache pin, hook gradients) are pinned in
# tests/test_nemotron_arch.py; this row is the SMOKE side of the contract —
# the failure signature the gate classifies as KNOWN-LIMIT.


def test_prefix_cache_nemotron_hybrid_contract():
    """Same contract as the qwen3_5 / gemma4 rows: refuse or be exact.
    Measured 2026-09-02: TeacherPrefixCache.build's explicit use_cache=True
    forward raises KeyError('mlp') at DynamicCache construction — and that
    signature is in the smoke's KNOWN-LIMIT table, so the onboarding gate
    reports it as the documented limitation rather than a failure."""
    from sisteer.extraction.prefix_cache import TeacherPrefixCache

    from onboard_model_smoke import _KNOWN_CACHE_LIMITS
    from test_nemotron_arch import _tiny_nemotron

    m, _ = _tiny_nemotron()
    verdict = _cache_never_silently_wrong(m, P=20)
    assert verdict in ("raised", "equivalent")
    if verdict == "raised":
        with pytest.raises(Exception) as ei:
            TeacherPrefixCache.build(m, torch.randint(5, 128, (20,)))
        assert any(sig in str(ei.value) for sig in _KNOWN_CACHE_LIMITS), \
            str(ei.value)


# ---- cross-file pin consistency -------------------------------------------------


def test_model_specs_match_committed_configs():
    import yaml

    from derive_model_config import MODEL_SPECS, SDFBASE_SLUGS

    for slug, spec in MODEL_SPECS.items():
        # The sdf-base-steering entries' committed configs are the
        # GENERATED sdfbase_<slug>_fits.yaml (make_sdfbase_configs.py),
        # not a two_si derivation — same cross-file pin, different home.
        path = (REPO / "configs" / f"sdfbase_{slug}_fits.yaml"
                if slug in SDFBASE_SLUGS else spec.config_path)
        with open(path) as fh:
            cfg = yaml.safe_load(fh)
        assert cfg["model"] == spec.model
        assert cfg["model_revision"] == spec.revision
        assert cfg["layer"] == spec.layer == round(0.469 * spec.n_layers)


def test_smoke_gate_collector_semantics():
    """The smoke's Gate: a raise is FAIL (never a crash of the gate), a
    soft tuple is SKIP/KNOWN-LIMIT/INFO, and only FAIL makes the gate
    fail."""
    from onboard_model_smoke import Gate

    g = Gate()
    g.run("ok", lambda: "fine")
    g.run("soft", lambda: ("SKIP", "not installed"))
    g.run("limit", lambda: ("KNOWN-LIMIT", "documented"))
    g.run("fact", lambda: ("INFO", "recorded, neither pass nor fail"))
    assert not g.failed
    g.run("boom", lambda: (_ for _ in ()).throw(RuntimeError("x")))
    assert g.failed
    assert [s for s, _, _ in g.rows] == ["PASS", "SKIP", "KNOWN-LIMIT",
                                         "INFO", "FAIL"]


def test_check_mamba_kernel_path_verdicts(monkeypatch):
    """The full tier's kernel-path row: PASS iff the modeling module
    latched the fused path (is_fast_path_available), INFO — never FAIL —
    on the pure-torch fallback (naming what resolved), SKIP without Mamba2
    mixers. The tiny real-class nemotron is built kernels-off, so it IS the
    fallback case; the fused verdict is exercised by flipping the latch."""
    import onboard_model_smoke as oms
    from test_nemotron_arch import _tiny_nemotron
    from transformers.models.nemotron_h import modeling_nemotron_h as mnh

    m, _ = _tiny_nemotron()
    got = oms.check_mamba_kernel_path(m)
    assert got[0] == "INFO", got
    assert got[1].startswith("pure-torch SSD scan (torch_forward)")
    for fact in ("use_mamba_kernels=False", "kernels-hub=", "2 Mamba2 blocks"):
        assert fact in got[1], fact
    if hasattr(mnh, "is_fast_path_available"):
        for fact in ("is_fast_path_available=False", "mamba_ssm=",
                     "causal_conv1d="):
            assert fact in got[1], fact

    if hasattr(mnh, "is_fast_path_available"):
        # transformers <= 5.14: the module-level latch decides
        monkeypatch.setattr(mnh, "is_fast_path_available", True)
        got = oms.check_mamba_kernel_path(m)
        assert isinstance(got, str), got
        assert got.startswith("fused kernels (cuda_kernels_forward)")
        assert "is_fast_path_available=True" in got
        monkeypatch.setattr(mnh, "is_fast_path_available", False)
    else:
        # transformers >= 5.15: per-op wrappers resolved at import; the
        # verdict is read out of each wrapper's implementation
        assert "dispatch=per-op wrappers" in got[1]
        assert "=torch" in got[1]
        import types as _types
        fake_pkg = _types.ModuleType("mamba_ssm"); fake_pkg.__version__ = "9.9"
        monkeypatch.setitem(sys.modules, "mamba_ssm", fake_pkg)

        def fused(*a, **k):
            pass
        fused.__module__ = "mamba_ssm.ops.triton"
        for name, fn in list(vars(mnh).items()):
            if oms._kernel_impl(fn) is not None:
                code = fn.__code__ if "implementation" in fn.__code__.co_freevars \
                    else fn.__wrapped__.__code__
                target = fn if "implementation" in fn.__code__.co_freevars \
                    else fn.__wrapped__
                cell = target.__closure__[code.co_freevars.index("implementation")]
                monkeypatch.setattr(cell, "cell_contents", fused)
        got = oms.check_mamba_kernel_path(m)
        assert isinstance(got, str), got
        assert got.startswith("fused kernels (cuda_kernels_forward)")
        assert "pip mamba_ssm 9.9" in got
        monkeypatch.undo()
        m, _ = _tiny_nemotron()

    # through the Gate: the fallback is recorded and never fails the gate
    g = oms.Gate()
    g.run("Mamba2 kernel path", lambda: oms.check_mamba_kernel_path(m))
    assert [s for s, _, _ in g.rows] == ["INFO"] and not g.failed

    plain = nn.Module()
    plain.config = _Cfg()
    got = oms.check_mamba_kernel_path(plain)
    assert got[0] == "SKIP" and "no Mamba2 mixers" in got[1]


# ---- the CLI: --model-config / --mode (the pilot's spelling; review fix F10) ----
#
# No downloads, no GPU: every model/tokenizer-touching check is stubbed, so
# these tests drive the REAL argparse + spec resolution + tier dispatch end
# to end — the layer that was broken (the pilot called a CLI the script did
# not define, and only the printed --dry-run declarations were tested).


class _FakeTok:
    chat_template = "{{ messages }}"

    def __len__(self):
        return 42


def mock_onboard_checks(monkeypatch):
    """Stub every model/tokenizer op in onboard_model_smoke. Returns
    ``(module, calls, specs_seen)``: ``calls`` records which checks ran (in
    order), ``specs_seen`` every ModelSpec a check received. Also imported
    by tests/test_pilot_order.py's parser-level integration test."""
    import onboard_model_smoke as oms

    calls: list[str] = []
    specs: list = []

    def note(name, ret="ok"):
        def fn(*a, **k):
            calls.append(name)
            specs.extend(x for x in a if isinstance(x, oms.ModelSpec))
            return ret
        return fn

    monkeypatch.setattr(oms, "check_transformers_version",
                        note("transformers"))
    monkeypatch.setattr(oms, "check_config", note("config"))
    monkeypatch.setattr(oms, "check_meta_accessors", note("meta"))
    monkeypatch.setattr(oms, "_tok", note("tok", ret=_FakeTok()))
    monkeypatch.setattr(oms, "check_system_role", note("system_role"))
    monkeypatch.setattr(oms, "check_header_len", note("header_len"))
    monkeypatch.setattr(oms, "check_thinking_pin", note("thinking_pin"))
    monkeypatch.setattr(oms, "check_si_counts", note("si_counts"))
    monkeypatch.setattr(oms, "check_sentinel", note("sentinel"))
    monkeypatch.setattr(oms, "check_kv_cache_allocatable",
                        note("kv_cache_allocatable"))
    monkeypatch.setattr(oms, "check_generation_stop_ids",
                        note("generation_stop_ids"))
    monkeypatch.setattr(oms, "check_model_load",
                        note("model_load", ret=(object(), _FakeTok(), "ok")))
    monkeypatch.setattr(oms, "check_mamba_kernel_path",
                        note("mamba_kernel_path"))
    monkeypatch.setattr(oms, "check_prefix_cache", note("prefix_cache"))
    monkeypatch.setattr(oms, "check_soft_token", note("soft_token"))
    monkeypatch.setattr(oms, "check_vllm", note("vllm"))
    return oms, calls, specs


def test_cli_model_config_full_runs_the_weights_tier(monkeypatch):
    oms, calls, specs = mock_onboard_checks(monkeypatch)
    rc = oms.main(["--model-config", str(GEMMA_CFG), "--mode", "full"])
    assert rc == 0
    assert "model_load" in calls and "prefix_cache" in calls
    assert "vllm" not in calls
    assert specs and all(s.slug == "gemma4e4b" for s in specs), (
        "--model-config must resolve the spec from the config's "
        "model/model_revision pins")


def test_cli_full_records_the_kernel_path_after_the_weight_load(monkeypatch):
    """The Mamba2 kernel-path row is a FULL-tier row: it needs the loaded
    model (the mixers resolve their kernels at construction), runs right
    after the load and before the heavy checks, and never runs in the
    weight-free tiers."""
    oms, calls, _ = mock_onboard_checks(monkeypatch)
    assert oms.main(["--model-config", str(GEMMA_CFG), "--mode", "full"]) == 0
    assert (calls.index("model_load") < calls.index("mamba_kernel_path")
            < calls.index("prefix_cache"))
    calls.clear()
    assert oms.main(["--model-config", str(GEMMA_CFG),
                     "--mode", "tokenizer-only"]) == 0
    assert "mamba_kernel_path" not in calls
    calls.clear()
    assert oms.main(["--model-config", str(GEMMA_CFG), "--mode", "vllm"]) == 0
    assert "mamba_kernel_path" not in calls


def test_cli_mode_vllm_runs_only_the_vllm_gate(monkeypatch):
    # Step 3 runs inside the DEDICATED vllm venv, which overlays its own
    # torch/transformers: the lockfile-pin checks would fail spuriously
    # there, and full mode would double-load the weights.
    oms, calls, _ = mock_onboard_checks(monkeypatch)
    rc = oms.main(["--model-config", str(GEMMA_CFG), "--mode", "vllm"])
    assert rc == 0
    assert calls == ["vllm"]


def test_cli_positional_slug_still_works(monkeypatch):
    oms, calls, specs = mock_onboard_checks(monkeypatch)
    rc = oms.main(["gemma4e4b", "--tokenizer-only"])
    assert rc == 0
    assert "sentinel" in calls
    # the two weight-free architecture checks ride the tokenizer-only tier
    assert "kv_cache_allocatable" in calls and "generation_stop_ids" in calls
    assert "model_load" not in calls and "vllm" not in calls
    assert specs and specs[0].slug == "gemma4e4b"


def test_cli_nemotron_slugs_resolve(monkeypatch, tmp_path):
    """Both Nemotron keys pin the SAME checkpoint (no base variant exists),
    so --model-config cannot resolve on the pins alone: each lane's config
    resolves to ITS spec by canonical file name, and a config with the
    shared pin under a lane-less name refuses rather than gating the
    documents config against the two-SI spec (the first-match bug this
    replaces)."""
    oms, calls, specs = mock_onboard_checks(monkeypatch)
    assert oms.main(["nemotron3nano4b", "--tokenizer-only"]) == 0
    assert specs[0].slug == "nemotron3nano4b"
    specs.clear()
    twosi = REPO / "configs" / "two_si_nemotron3nano4b.yaml"
    assert oms.main(["--model-config", str(twosi), "--mode",
                     "tokenizer-only"]) == 0
    assert specs and all(s.slug == "nemotron3nano4b" for s in specs)
    ambiguous = tmp_path / "scratch.yaml"
    ambiguous.write_text(
        "model: nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16\n"
        "model_revision: dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f\n")
    with pytest.raises(SystemExit) as ei:
        oms.main(["--model-config", str(ambiguous), "--mode",
                  "tokenizer-only"])
    # two lanes share the pin: two_si and the full-scale Slurm lane
    # (nemotron3nano4b_doc_full; the data-scaling ladder lane was retired 2026-10-08)
    assert "matches 2 ModelSpecs" in str(ei.value)
    assert "nemotron3nano4b_doc_full" in str(ei.value)
    specs.clear()
    full = REPO / "configs" / "sdfbase_nemotron3nano4b_doc_full_fits.yaml"
    assert oms.main(["--model-config", str(full), "--mode",
                     "tokenizer-only"]) == 0
    assert specs and all(s.slug == "nemotron3nano4b_doc_full" for s in specs)


def test_check_si_counts_documents_lane_is_informational():
    """A documents-lane config (sdfbase_*_fits.yaml) carries no
    max_len_by_si: the SI counts are reported, not pinned — asserting
    None == {...} failed every sdfbase-config smoke run before this."""
    import onboard_model_smoke as oms
    from test_nemotron_chat_markers import FakeNemoTok

    tok = FakeNemoTok()
    spec = oms.MODEL_SPECS["nemotron3nano4b_doc_full"]
    detail = oms.check_si_counts(tok, spec, {"data_mode": "documents",
                                             "max_len": 8192})
    assert "no max_len_by_si" in detail and "safety_refusal=" in detail
    with pytest.raises(AssertionError, match="regenerate"):
        oms.check_si_counts(tok, spec, {"max_len_by_si": {"wrong": 1}})


def test_check_kv_cache_allocatable_verdicts(monkeypatch):
    """Config-only verdicts: allocatable -> PASS detail; unmappable types
    with teacher_prefix_cache: false -> KNOWN-LIMIT; with true/absent ->
    FAIL (the cache the config would build cannot exist)."""
    import onboard_model_smoke as oms

    class _AC:
        types = ["full_attention"]

        @staticmethod
        def from_pretrained(*a, **k):
            return _Cfg(layer_types=list(_AC.types))

    monkeypatch.setattr("transformers.AutoConfig", _AC)
    spec = oms.MODEL_SPECS["nemotron3nano4b"]
    assert isinstance(oms.check_kv_cache_allocatable(spec, {}), str)
    # the verdict is about the cache MAPPING, not about any one type: 5.14.1
    # could not map 'mlp' (Nemotron's trap), 5.15.0 can; an invented type is
    # unmappable under every version
    _AC.types = ["linear_attention", "no_such_block", "full_attention",
                 "no_such_block"]
    got = oms.check_kv_cache_allocatable(spec, {"teacher_prefix_cache": False})
    assert got[0] == "KNOWN-LIMIT" and "['no_such_block'] (2 layers)" in got[1]
    with pytest.raises(AssertionError, match="teacher_prefix_cache: false"):
        oms.check_kv_cache_allocatable(spec, {"teacher_prefix_cache": True})
    with pytest.raises(AssertionError):
        oms.check_kv_cache_allocatable(spec, {})
    assert oms.check_kv_cache_allocatable(spec, None)[0] == "KNOWN-LIMIT"


def test_cli_mode_tokenizer_only_matches_the_legacy_flag(monkeypatch):
    oms, calls, _ = mock_onboard_checks(monkeypatch)
    assert oms.main(["gemma4e4b", "--tokenizer-only"]) == 0
    legacy = list(calls)
    calls.clear()
    assert oms.main(["gemma4e4b", "--mode", "tokenizer-only"]) == 0
    assert calls == legacy


def test_cli_vllm_flag_stays_additive_on_full(monkeypatch):
    oms, calls, _ = mock_onboard_checks(monkeypatch)
    rc = oms.main(["--model-config", str(GEMMA_CFG), "--vllm"])
    assert rc == 0
    assert "model_load" in calls and "vllm" in calls


def test_cli_requires_exactly_one_model_selector(monkeypatch):
    oms, _, _ = mock_onboard_checks(monkeypatch)
    with pytest.raises(SystemExit):
        oms.main([])
    with pytest.raises(SystemExit):
        oms.main(["gemma4e4b", "--model-config", str(GEMMA_CFG)])


def test_cli_tokenizer_only_conflicts_with_another_mode(monkeypatch):
    oms, _, _ = mock_onboard_checks(monkeypatch)
    with pytest.raises(SystemExit):
        oms.main(["gemma4e4b", "--tokenizer-only", "--mode", "full"])


def test_cli_unknown_config_pin_fails_loudly(monkeypatch, tmp_path):
    # The onboarding facts (layer/hidden counts, cache limits) exist only
    # per MODEL_SPECS entry — an unknown pin must refuse, never guess.
    oms, _, _ = mock_onboard_checks(monkeypatch)
    cfg = tmp_path / "two_si_unknown.yaml"
    cfg.write_text("model: nobody/unknown-model\nmodel_revision: deadbeef\n")
    with pytest.raises(SystemExit) as ei:
        oms.main(["--model-config", str(cfg)])
    assert "matches no known ModelSpec" in str(ei.value)


GLM_TWO_SI_CFG = REPO / "configs" / "two_si_glm46vflash.yaml"
GLM_DOC_CFG = REPO / "configs" / "sdfbase_glm46vflash_doc_fits.yaml"


def test_resolve_spec_disambiguates_the_shared_glm_pin(tmp_path):
    """glm46vflash (two_si) and glm46vflash_doc (doc-CE) pin the SAME
    checkpoint. --model-config resolves to the lane whose committed config
    was handed in — the banner and slug name the lane being gated; an
    off-tree copy that KEPT the canonical lane file name still resolves
    by name; an off-tree copy under a lane-less name REFUSES (the same
    contract test_cli_nemotron_slugs_resolve pins for the Nemotron pair —
    a first-match fallback would gate the doc config against the two_si
    spec). Single-match lanes are untouched."""
    import onboard_model_smoke as oms
    from derive_model_config import MODEL_SPECS

    assert oms._spec_committed_config(MODEL_SPECS["glm46vflash"]) == \
        GLM_TWO_SI_CFG
    assert oms._spec_committed_config(MODEL_SPECS["glm46vflash_doc"]) == \
        GLM_DOC_CFG
    assert oms._spec_committed_config(MODEL_SPECS["gemma4e4b_base"]) == \
        REPO / "configs" / "sdfbase_gemma4e4b_base_fits.yaml"
    assert oms._spec_committed_config(MODEL_SPECS["gemma4e4b"]) == GEMMA_CFG
    assert oms._resolve_spec_from_config(GLM_TWO_SI_CFG).slug == "glm46vflash"
    assert oms._resolve_spec_from_config(GLM_DOC_CFG).slug == "glm46vflash_doc"
    named_copy = tmp_path / GLM_DOC_CFG.name
    named_copy.write_text(GLM_DOC_CFG.read_text())
    assert oms._resolve_spec_from_config(named_copy).slug == "glm46vflash_doc"
    copy = tmp_path / "elsewhere.yaml"
    copy.write_text(GLM_DOC_CFG.read_text())
    with pytest.raises(SystemExit, match="matches 4 ModelSpecs"):
        oms._resolve_spec_from_config(copy)
    assert oms._resolve_spec_from_config(GEMMA_CFG).slug == "gemma4e4b"
    assert oms._resolve_spec_from_config(
        REPO / "configs" / "sdfbase_gemma4e4b_base_fits.yaml").slug == \
        "gemma4e4b_base"


def test_cli_glm_doc_lane_by_slug_and_by_config(monkeypatch):
    """Both spellings gate the doc lane under ITS slug: the positional slug
    loads the committed sdfbase config as the derived config (sdfbase slugs
    have no committed two_si file and previously gated against None),
    and --model-config on that file resolves to glm46vflash_doc, not to
    the two_si sibling sharing the pin."""
    oms, calls, specs = mock_onboard_checks(monkeypatch)
    assert oms.main(["glm46vflash_doc", "--tokenizer-only"]) == 0
    assert specs and all(s.slug == "glm46vflash_doc" for s in specs)
    assert "si_counts" in calls and "model_load" not in calls
    calls.clear()
    specs.clear()
    assert oms.main(["--model-config", str(GLM_DOC_CFG),
                     "--mode", "tokenizer-only"]) == 0
    assert specs and all(s.slug == "glm46vflash_doc" for s in specs)
    calls.clear()
    specs.clear()


def test_cli_glm_by_slug_and_by_config(monkeypatch):
    """Both spellings gate the GLM two-SI lane under ITS slug, with the
    committed two_si_glm46vflash.yaml loaded as the derived config."""
    oms, calls, specs = mock_onboard_checks(monkeypatch)
    assert oms.main(["glm46vflash", "--tokenizer-only"]) == 0
    assert specs and all(s.slug == "glm46vflash" for s in specs)
    assert "si_counts" in calls and "model_load" not in calls
    calls.clear()
    specs.clear()
    assert oms.main(["--model-config", str(GLM_TWO_SI_CFG),
                     "--mode", "tokenizer-only"]) == 0
    assert specs and all(s.slug == "glm46vflash" for s in specs)


def test_si_counts_check_handles_doc_mode_configs(monkeypatch):
    """check_si_counts compares the live re-derivation against a two_si
    config's max_len_by_si and FAILS on a mismatch. A doc-CE config
    (data_mode: documents) has no per-SI teacher budget by design, so the
    counts are reported with an explicit doc-mode note instead of the
    category-error failure that made every sdfbase --model-config
    onboarding exit 1 on that row (gemma base and GLM doc alike, measured
    2026-09-02). A config with NEITHER key still fails hard — nothing is
    skipped silently."""
    import yaml

    import onboard_model_smoke as oms

    counts = {"safety_refusal": 100, "si_constitution": 5883,
              "si_irrelevant": 6419}
    monkeypatch.setattr(oms, "measure_si_prompt_tokens",
                        lambda tok: dict(counts))
    spec = oms.MODEL_SPECS["glm46vflash"]
    two_si = yaml.safe_load(GLM_TWO_SI_CFG.read_text())
    ok = oms.check_si_counts(object(), spec, two_si)
    assert ok.endswith("[matches committed config]")
    assert "si_irrelevant=6419 (-> 7956 -> max_len 8192)" in ok
    doc = yaml.safe_load(GLM_DOC_CFG.read_text())
    detail = oms.check_si_counts(object(), oms.MODEL_SPECS["glm46vflash_doc"],
                                 doc)
    assert detail.endswith("[doc-mode config: no max_len_by_si to compare; "
                           "sequence_max_len=8192]")
    assert "safety_refusal=100" in detail
    with pytest.raises(AssertionError, match="regenerate"):
        oms.check_si_counts(object(), spec,
                            {"max_len_by_si": {"safety_refusal": 4096}})
    with pytest.raises(AssertionError):
        oms.check_si_counts(object(), spec, {})       # neither key: hard
    with pytest.raises(AssertionError):
        oms.check_si_counts(object(), spec, {"data_mode": "pairs"})
    assert "[" not in oms.check_si_counts(object(), spec, None)  # counts only


# ---- GLM-4.6V-Flash against the REAL tokenizer/config (local cache only) --------


def _skip_unless_glm_cached():
    """Loud skip naming what is missing — the tests/test_glm_chat_markers.py
    convention. Never a download."""
    transformers = pytest.importorskip("transformers")
    try:
        transformers.AutoTokenizer.from_pretrained(
            GLM_MODEL, revision=GLM_REV, local_files_only=True)
        transformers.AutoConfig.from_pretrained(
            GLM_MODEL, revision=GLM_REV, local_files_only=True)
    except Exception as e:  # cache miss, offline resolution failure, ...
        pytest.skip(f"pinned {GLM_MODEL} tokenizer/config @ {GLM_REV[:8]} "
                    f"not in the local HF cache ({type(e).__name__})")


def test_glm_spec_pin_matches_test_constants():
    from derive_model_config import MODEL_SPECS

    spec = MODEL_SPECS["glm46vflash"]
    assert (spec.model, spec.revision) == (GLM_MODEL, GLM_REV)
    assert (spec.n_layers, spec.hidden_size, spec.layer) == (40, 4096, 19)


def test_meta_accessor_check_dispatches_through_auto_model_class():
    """check_meta_accessors builds its meta instance through
    models.auto_model_class — the loader's own dispatch — never a
    hard-coded AutoModelForCausalLM, which raises 'Unrecognized
    configuration class' on glm4v (measured 2026-09-02) and would fail
    the gate on a model load_model_and_tokenizer handles. Source-level
    pin, plus a real run on the cached GLM config (meta device: the full
    40-layer structure, zero weight bytes)."""
    import ast
    import inspect
    import textwrap

    import onboard_model_smoke as oms

    tree = ast.parse(textwrap.dedent(inspect.getsource(oms.check_meta_accessors)))
    imported = {a.name for n in ast.walk(tree)
                if isinstance(n, ast.ImportFrom) for a in n.names}
    assert "auto_model_class" in imported
    assert "AutoModelForCausalLM" not in imported
    from_config = [n for n in ast.walk(tree)
                   if isinstance(n, ast.Call)
                   and isinstance(n.func, ast.Attribute)
                   and n.func.attr == "from_config"]
    assert from_config, "no from_config call — the meta probe is gone"
    for call in from_config:
        # the instance comes off the auto_model_class result, not a class
        # named inline
        assert isinstance(call.func.value, ast.Name), ast.dump(call)
        assert call.func.value.id == "auto_cls", ast.dump(call)

    _skip_unless_glm_cached()
    from derive_model_config import MODEL_SPECS

    detail = oms.check_meta_accessors(MODEL_SPECS["glm46vflash"])
    assert detail.startswith(
        "Glm4vForConditionalGeneration (via AutoModelForImageTextToText)")
    assert detail.endswith("40 layers, hidden 4096")


def test_glm_tokenizer_tier_checks_against_committed_config():
    """The grading-host-runnable half of the onboarding gate on the REAL
    GLM-4.6V-Flash tokenizer: system role rendered directly, H = 5 via the
    add_generation_prompt diff (<|assistant|>\\n<think></think>\\n under
    the thinking-off pin), enable_thinking pinned off, [sMASK] as the
    token-arm sentinel, and the per-SI teacher-prompt counts re-deriving
    EXACTLY the committed two_si_glm46vflash.yaml max_len_by_si
    (100 / 5883 / 6419 -> 2048 / 8192 / 8192). A tokenizer or SI-text
    change that moves any bucket fails here, on the grading host, before a pod
    sees it."""
    _skip_unless_glm_cached()
    import onboard_model_smoke as oms

    spec = oms.MODEL_SPECS["glm46vflash"]
    tok = oms._tok(spec)
    derived = oms._load_derived_config(spec.config_path)
    assert derived is not None, f"{spec.config_path} missing"
    assert oms.check_system_role(tok).startswith("direct")
    assert oms.check_header_len(tok) == "H = 5 (via diff)"
    assert oms.check_thinking_pin(tok, spec) == \
        "enable_thinking present in template; pinned False"
    assert oms.check_sentinel(tok) == "sentinel '[sMASK]' -> id 151332"
    detail = oms.check_si_counts(tok, spec, derived)
    assert detail.endswith("[matches committed config]")
    for frag in ("safety_refusal=100 (-> 1637 -> max_len 2048)",
                 "si_constitution=5883 (-> 7420 -> max_len 8192)",
                 "si_irrelevant=6419 (-> 7956 -> max_len 8192)"):
        assert frag in detail, detail
    assert derived["max_len_by_si"] == {"safety_refusal": 2048,
                                        "si_constitution": 8192,
                                        "si_irrelevant": 8192}
    assert derived["max_len"] == 8192
