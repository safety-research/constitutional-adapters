"""The Nemotron GSM8K decode ran CACHE-FREE (a full prefix recompute
per token) because transformers 5.14.1 cannot build a DynamicCache for a
NemotronH config whose ``layer_types`` carry "mlp", and the loader's rescue
was to pin ``use_cache`` off. The fix registers the placeholder cache layer
transformers 5.15.0 ships ("mlp" -> LinearAttentionLayer) at load time;
generate() then decodes with its cache and the outputs are unchanged.

Pinned here, on the tiny random-weight NemotronH of tests/test_nemotron_arch
(the REAL class, CPU, pure-torch SSD path) and the CPU tiny llama:

* ``register_missing_cache_layer_types`` registers exactly the placeholder
  types (mlp / moe) the installed transformers lacks — in BOTH mappings, to
  LinearAttentionLayer — and never an unknown type; idempotent;
* after it, ``DynamicCache(config=...)`` allocates the hybrid's cache with
  the layer classes 5.15.0 would give it, and ``prepare_kv_cache`` leaves
  ``use_cache`` ON (the pin fires only for a genuinely unmappable type);
* cached greedy generate == cache-free generate, token for token, on a
  LEFT-padded batch; a cached forward's logits equal the cache-free ones;
* the evaluator's ``decode_runtime`` reads the cache flags and the Mamba
  kernel path; ``assert_cached_decode`` refuses a cache-free runtime unless
  told otherwise and merely warns about the torch scan;
* the aggregate carries ``runtime`` OUTSIDE the identity (the identity, and
  so the label merge keys, are byte-identical to the published r2 cells);
* generate_answers logs one pace line per chunk.
"""

from __future__ import annotations

import json
import sys
import types
import warnings
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts" / "sdfbase") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts" / "sdfbase"))
if str(REPO / "tests") not in sys.path:
    sys.path.insert(0, str(REPO / "tests"))

import base_gsm8k as bg  # noqa: E402
import base_mmlu as bm  # noqa: E402
import sisteer.models as sm  # noqa: E402
from test_nemotron_arch import _tiny_nemotron  # noqa: E402

cache_utils = pytest.importorskip("transformers.cache_utils")
DYN = cache_utils.DYNAMIC_LAYER_TYPE_MAPPING
STA = cache_utils.STATIC_LAYER_TYPE_MAPPING


def _left_padded_batch():
    torch.manual_seed(1)
    ids = torch.randint(5, 128, (3, 12))
    am = torch.ones_like(ids)
    ids[0, :4] = 0
    am[0, :4] = 0
    ids[1, :1] = 0
    am[1, :1] = 0
    return ids, am


# --------------------------------------------------------------------------- #
# registration                                                                 #
# --------------------------------------------------------------------------- #


def test_register_mirrors_transformers_5_15_placeholder_entries(monkeypatch):
    # Start from the 5.14.1 world whatever the installed version or prior
    # tests did: no "mlp" in either mapping (restored by monkeypatch).
    monkeypatch.delitem(DYN, "mlp", raising=False)
    monkeypatch.delitem(STA, "mlp", raising=False)
    m, cfg = _tiny_nemotron()
    assert sm.kv_cache_unmappable_layer_types(cfg) == ["mlp"]
    with pytest.raises(KeyError, match="mlp"):
        cache_utils.DynamicCache(config=cfg)

    assert sm.register_missing_cache_layer_types(cfg) == ["mlp"]
    assert DYN["mlp"] is cache_utils.LinearAttentionLayer
    assert STA["mlp"] is cache_utils.LinearAttentionLayer
    assert sm.kv_cache_unmappable_layer_types(cfg) == []
    # idempotent
    assert sm.register_missing_cache_layer_types(cfg) == []
    # the hybrid's cache now allocates with 5.15.0's layer classes: empty
    # placeholders for mamba/mlp, a real KV layer for attention
    c = cache_utils.DynamicCache(config=cfg)
    assert [type(layer).__name__ for layer in c.layers] == [
        "LinearAttentionLayer", "LinearAttentionLayer", "DynamicLayer",
        "LinearAttentionLayer"]
    assert list(cfg.layer_types) == ["linear_attention", "mlp",
                                     "full_attention", "linear_attention"]


def test_register_never_invents_an_entry_for_an_unknown_type():
    fake = types.SimpleNamespace(layer_types=["linear_attention",
                                              "no_such_block"])
    assert sm.register_missing_cache_layer_types(fake) == []
    assert "no_such_block" not in DYN and "no_such_block" not in STA
    assert sm.kv_cache_unmappable_layer_types(fake) == ["no_such_block"]
    # a config without layer_types (Llama-style) registers nothing
    assert sm.register_missing_cache_layer_types(types.SimpleNamespace()) == []


def test_prepare_kv_cache_registers_then_leaves_the_cache_on(monkeypatch):
    monkeypatch.delitem(DYN, "mlp", raising=False)
    monkeypatch.delitem(STA, "mlp", raising=False)
    m, cfg = _tiny_nemotron()
    assert m.config.use_cache is True
    with pytest.warns(RuntimeWarning, match="registered as empty placeholder"):
        out = sm.prepare_kv_cache(m)
    assert out == {"registered": ["mlp"], "pinned_off": [],
                   "pre_fix_cache_free": ["mlp"]}
    assert m.kv_cache_preparation == out
    assert m.config.use_cache is True
    assert m.generation_config.use_cache is not False
    # second call: nothing to register, no warning — the stable marker stays
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert sm.prepare_kv_cache(m) == {"registered": [], "pinned_off": [],
                                          "pre_fix_cache_free": ["mlp"]}
    # nested text_config is covered too
    nested = types.SimpleNamespace(
        config=types.SimpleNamespace(
            use_cache=True,
            text_config=types.SimpleNamespace(layer_types=["mlp"],
                                              use_cache=True)),
        generation_config=types.SimpleNamespace(use_cache=True))
    monkeypatch.delitem(DYN, "mlp", raising=False)
    with pytest.warns(RuntimeWarning):
        assert sm.prepare_kv_cache(nested)["registered"] == ["mlp"]
    assert nested.config.text_config.use_cache is True


def test_prepare_kv_cache_still_pins_off_a_genuinely_unmappable_type():
    bad = types.SimpleNamespace(
        config=types.SimpleNamespace(use_cache=True,
                                     layer_types=["linear_attention",
                                                  "no_such_block"]),
        generation_config=types.SimpleNamespace(use_cache=True))
    with pytest.warns(RuntimeWarning, match="cannot allocate"):
        out = sm.prepare_kv_cache(bad)
    assert out == {"registered": [], "pinned_off": ["no_such_block"],
                   "pre_fix_cache_free": []}
    assert bad.config.use_cache is False
    assert bad.generation_config.use_cache is False
    # a model without a config is a no-op
    assert sm.prepare_kv_cache(types.SimpleNamespace()) == {
        "registered": [], "pinned_off": [], "pre_fix_cache_free": []}


# --------------------------------------------------------------------------- #
# the cached decode is the same decode                                         #
# --------------------------------------------------------------------------- #


def test_cached_generate_equals_cache_free_on_tiny_nemotron():
    m, cfg = _tiny_nemotron()
    sm.prepare_kv_cache(m)
    ids, am = _left_padded_batch()
    with torch.no_grad():
        g_free = m.generate(ids, attention_mask=am, max_new_tokens=8,
                            do_sample=False, pad_token_id=0, use_cache=False)
        g_cached = m.generate(ids, attention_mask=am, max_new_tokens=8,
                              do_sample=False, pad_token_id=0, use_cache=True)
        free = m(input_ids=ids, attention_mask=am, use_cache=False)
        cached = m(input_ids=ids, attention_mask=am, use_cache=True)
    assert tuple(g_free.shape) == (3, 20)
    assert torch.equal(g_free, g_cached)
    assert torch.equal(free.logits, cached.logits)
    assert free.past_key_values is None
    assert cached.past_key_values is not None
    assert cached.past_key_values.get_seq_length() == 12
    # a cached second step reads the primed state, not a recompute: the
    # attention layer's KV grew by one token
    with torch.no_grad():
        nxt = g_cached[:, 12:13]
        am2 = torch.cat([am, torch.ones_like(nxt)], dim=1)
        step = m(input_ids=nxt, attention_mask=am2,
                 past_key_values=cached.past_key_values, use_cache=True)
    assert step.past_key_values.get_seq_length() == 13
    assert tuple(step.logits.shape) == (3, 1, 128)


def test_loader_applies_prepare_kv_cache(monkeypatch):
    """load_model_and_tokenizer's post-load step is prepare_kv_cache (not
    the bare pin): a NemotronH loaded through it keeps use_cache ON."""
    seen: dict = {}
    real = sm.prepare_kv_cache

    def spy(model):
        seen["out"] = real(model)
        return seen["out"]

    monkeypatch.setattr(sm, "prepare_kv_cache", spy)
    m, cfg = _tiny_nemotron()

    class _Tok:
        eos_token_id = 1
        chat_template = None

    monkeypatch.setattr(sm, "load_tokenizer",
                        lambda name, revision=None, **kw: _Tok())
    monkeypatch.setattr(sm, "auto_model_class",
                        lambda name, revision=None, **kw: types.SimpleNamespace(
                            from_pretrained=lambda *a, **k: m))
    monkeypatch.setattr(sm, "check_chat_template", lambda tok, **kw: None)
    monkeypatch.setattr(sm, "pin_generation_stop_ids", lambda model, tok: [])
    monkeypatch.setattr(sm, "_enforce_llama31_mirror", lambda name: None)
    # main's loader refuses a Mamba hybrid on the pure-torch fallback
    # (assert_fused_mamba_kernels, user directive 2026-09-09) and this CPU env
    # has neither kernel package, so the guard is stubbed HERE: what this test
    # pins is the prepare_kv_cache step that follows it. The guard's own
    # contract (refuses, names the packages, propagates from the loader) is
    # tests/test_nemotron_arch.py's.
    monkeypatch.setattr(sm, "assert_fused_mamba_kernels", lambda model: None)
    model, tok = sm.load_model_and_tokenizer("fake/nemo", device="cpu",
                                             dtype="float32")
    assert model is m and "out" in seen
    assert seen["out"]["pinned_off"] == []
    assert model.config.use_cache is True
    assert model.generation_config.use_cache is not False


# --------------------------------------------------------------------------- #
# evaluator: runtime facts, refusal, aggregate block, pace line               #
# --------------------------------------------------------------------------- #


def test_decode_runtime_reads_the_flags_and_kernel_path(tiny_llama):
    model, _tok = tiny_llama
    # tiny_llama is SESSION-scoped and shared: decode_runtime is a READER of
    # the two cache flags, so this test sets the flags it then reads rather
    # than asserting that no earlier test left them pinned off -- in the
    # reseed merge tree's Modal CPU suite (2026-09-15) one had, and the first
    # assertion read False on a flag this test never set. Restored after.
    cfg, gc = model.config, model.generation_config
    prev = (getattr(cfg, "use_cache", None), getattr(gc, "use_cache", None))
    try:
        cfg.use_cache, gc.use_cache = True, True
        rt = bg.decode_runtime(model)
        assert rt["kv_cache"] is True
        assert rt["mamba_kernel_path"] is None  # no Mamba2 mixer on a llama
        assert rt["transformers"] and rt["torch"]
        gc.use_cache = False
        assert bg.decode_runtime(model)["kv_cache"] is False
    finally:
        cfg.use_cache, gc.use_cache = prev
    # kernels off -> "torch" under either dispatch generation (the <= 5.14
    # latch, or 5.15's per-op wrappers read through models.resolved_mamba_kernels)
    m, _cfg = _tiny_nemotron()
    assert bg.mamba_kernel_path(m) == "torch"
    assert bg.decode_runtime(m)["mamba_kernel_path"] == "torch"


def test_assert_cached_decode_refuses_cache_free_unless_allowed(capsys):
    ok = {"kv_cache": True, "mamba_kernel_path": "fused"}
    bg.assert_cached_decode(ok)
    bad = {"kv_cache": False, "config_use_cache": False,
           "generation_config_use_cache": False, "mamba_kernel_path": "fused"}
    with pytest.raises(SystemExit, match="CACHE-FREE"):
        bg.assert_cached_decode(bad)
    bg.assert_cached_decode(bad, allow_cache_free=True)  # diagnosis path
    bg.assert_cached_decode({"kv_cache": True, "mamba_kernel_path": "torch"})
    assert "pure-torch SSD scan" in capsys.readouterr().out


def test_cli_exposes_allow_cache_free_outside_the_identity():
    src = (REPO / "scripts/sdfbase/base_gsm8k.py").read_text()
    assert "--allow-cache-free" in src
    # the run identity is built from these keys only; no cache key joins it
    assert '"allow_cache_free"' not in src and '"kv_cache":' not in \
        src.split("run_identity = {")[1].split("}")[0]


def test_aggregate_records_runtime_outside_the_identity(tmp_path):
    identity = {"run_label": "r2", "model": "m", "cell": "none",
                "object_sha256": None, "batch_size": 128}
    per_q = [{"qidx": 0, "qid": "a", "correct": True, "n_new_tokens": 3,
              "terminated": True, "hit_cap": False, "empty": False,
              "rule": "answer_is", "stop_reason": "stop_string",
              "extracted": 1.0, "gold": 1.0, "response": "The answer is 1."}]
    agg = bg.aggregate(per_q)
    runtime = {"kv_cache": True, "mamba_kernel_path": "fused",
               "transformers": "5.14.1"}
    agg_path, pq_path = bg.write_cell_results(tmp_path, "r2", "none", agg,
                                              per_q, identity, runtime=runtime)
    rec = json.loads(agg_path.read_text())
    assert rec["runtime"] == runtime
    assert rec["identity"] == identity and "runtime" not in rec["identity"]
    header, rows = bg.read_per_question(pq_path)
    assert header == identity and len(rows) == 1
    # without a runtime the record is byte-for-byte the pre-fix shape
    agg_path2, _ = bg.write_cell_results(tmp_path, "r2b", "none", agg, per_q,
                                         identity)
    assert "runtime" not in json.loads(agg_path2.read_text())


# --------------------------------------------------------------------------- #
# decode-path parity gate                                                        #
# --------------------------------------------------------------------------- #

_IDENT = {"run_label": "r2", "model": "m", "model_revision": "rev",
          "config_sha256": "c" * 8, "batch_size": 128,
          "ordered_qid_hash": "q" * 8, "n": 3, "shots": 8,
          "exemplars_sha256": "e" * 8, "max_new_tokens": 512,
          "stop_strings": ["\nQuestion:"], "dataset": "d",
          "dataset_config": "main", "dataset_split": "test",
          "dataset_revision": "dr", "dtype": "bfloat16",
          "attn_implementation": "sdpa", "decoding": "greedy",
          "extraction": "answer_is_first|hash_last|last_number", "scale": 1.0}


def _rows3(flip: dict | None = None):
    rows = []
    for i, (ext, stop) in enumerate(((6.0, "stop_string"), (5.0, "eos"),
                                     (None, "cap"))):
        rows.append({"qidx": i, "qid": f"q{i}", "gold": 6.0 if i == 0 else 7.0,
                     "extracted": ext, "rule": "answer_is" if ext else "none",
                     "correct": ext == 6.0, "terminated": stop != "cap",
                     "stop_reason": stop, "stopped_on_string": stop == "stop_string",
                     "hit_cap": stop == "cap", "n_new_tokens": 10 + i,
                     "empty": False, "response": f"The answer is {ext}."})
    for i, patch in (flip or {}).items():
        rows[i].update(patch)
    return rows


def _publish(run_dir, label, cell, rows, identity, runtime=None,
             scale=1.0):
    ident = {**identity, "run_label": label, "cell": cell, "scale": scale,
             "object_path": None if cell == "none" else f"/o/{cell}.pt",
             "object_sha256": None if cell == "none" else "s" * 8}
    return bg.write_cell_results(run_dir, label, cell, bg.aggregate(rows),
                                 rows, ident, runtime=runtime)


def _attest(tmp_path, pub_pq, new_pq, name="att.json"):
    import gsm8k_decode_parity as gp

    att = gp.compare(pub_pq, new_pq)
    p = tmp_path / name
    p.write_text(json.dumps(att))
    return p, att


CACHED_RT = {"kv_cache": True, "pre_fix_cache_free": ["mlp"],
             "cache_placeholder_registered": ["mlp"],
             "transformers": "5.14.1", "mamba_kernel_path": "fused"}


def test_pre_fix_cache_free_marker_is_stable_across_loads(monkeypatch):
    """The gate must key on a property of the ARCHITECTURE, not
    on whether this load mutated the global mapping — a second in-process
    load (or a transformers that ships 'mlp') registers nothing."""
    monkeypatch.delitem(DYN, "mlp", raising=False)
    m1, cfg = _tiny_nemotron()
    with pytest.warns(RuntimeWarning):
        p1 = sm.prepare_kv_cache(m1)
    assert p1["registered"] == ["mlp"] and p1["pre_fix_cache_free"] == ["mlp"]
    m2, _ = _tiny_nemotron()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p2 = sm.prepare_kv_cache(m2)  # mapping already carries 'mlp'
    assert p2["registered"] == [] and p2["pre_fix_cache_free"] == ["mlp"]
    for m in (m1, m2):
        rt = bg.decode_runtime(m)
        assert rt["pre_fix_cache_free"] == ["mlp"]
    assert bg.decode_runtime(m1)["cache_placeholder_registered"] == ["mlp"]
    assert bg.decode_runtime(m2)["cache_placeholder_registered"] == []
    # read off the config even without a loader record (a model that did
    # not go through prepare_kv_cache)
    m3, _ = _tiny_nemotron()
    assert bg.decode_runtime(m3)["pre_fix_cache_free"] == ["mlp"]
    assert sm.pre_fix_cache_free_layer_types(cfg) == ["mlp"]
    # architectures the pinned transformers cached natively stay exempt
    for lt in (["sliding_attention", "full_attention"],          # gemma-like
               ["linear_attention", "full_attention"],           # qwen3.5-like
               ["full_attention"], None):                        # GLM / llama
        c = types.SimpleNamespace(layer_types=lt) if lt is not None \
            else types.SimpleNamespace()
        assert sm.pre_fix_cache_free_layer_types(c) == []
    nested = types.SimpleNamespace(text_config=types.SimpleNamespace(
        layer_types=["mlp", "full_attention"]))
    assert sm.pre_fix_cache_free_layer_types(nested) == ["mlp"]
    assert sm.pre_fix_cache_free_layer_types(None) == []
    # 'moe' was mappable on 5.14.1 -> a MoE hybrid was never cache-free
    assert sm.pre_fix_cache_free_layer_types(
        types.SimpleNamespace(layer_types=["moe", "full_attention"])) == []


def test_rows_outcome_digest_ignores_text_and_order():
    a = _rows3()
    b = [dict(r, response="something else") for r in reversed(a)]
    assert bg.rows_outcome_digest(a) == bg.rows_outcome_digest(b)
    assert bg.rows_outcome_digest(a) != bg.rows_outcome_digest(
        _rows3({2: {"stop_reason": "stop_string"}}))


def test_legacy_cells_are_the_runtime_less_aggregates(tmp_path):
    d = tmp_path / "r2"
    d.mkdir()
    _publish(d, "r2", "none", _rows3(), _IDENT)
    _publish(d, "r2", "c1__psr", _rows3(), _IDENT)
    _publish(d, "r2", "c2__psr", _rows3(), _IDENT, runtime=CACHED_RT)
    # an --allow-cache-free re-score records kv_cache False: cache-free too
    _publish(d, "r2", "c3__psr", _rows3(), _IDENT,
             runtime={**CACHED_RT, "kv_cache": False})
    bm.write_summary(bg.summary_path(d, "r2"), {**_IDENT}, ["none"], {},
                     merge_keys=bg.MERGE_KEYS)
    assert bg.legacy_cells(d, "r2") == ["c1__psr", "c3__psr", "none"]
    assert bg.legacy_cells(tmp_path / "empty", "r2") == []


def test_decode_parity_attestation_round_trip(tmp_path):
    import gsm8k_decode_parity as gp

    d = tmp_path / "r2"
    d.mkdir()
    _agg, pub = _publish(d, "r2", "none", _rows3(), _IDENT)
    s = tmp_path / "scratch"
    s.mkdir()
    # same outcomes, different text on one row -> still identical outcomes
    _agg2, new = _publish(s, "parity_x1", "none",
                          _rows3({0: {"response": "So 6. The answer is 6."}}),
                          _IDENT, runtime=CACHED_RT)
    att = gp.compare(pub, new)
    assert att["kind"] == bg.ATTESTATION_KIND
    assert att["verdict"]["outcomes_identical"] is True
    assert att["identical_response_text"] == 2 and att["n"] == 3
    assert att["published"]["rows_digest"] == bg.rows_outcome_digest(_rows3())
    assert att["new"]["runtime"] == CACHED_RT
    assert att["shared_identity"]["cell"] == "none"
    assert att["mcnemar"] == {"both": 1, "a_only": 0, "b_only": 0,
                              "neither": 2, "p_exact_two_sided": 1.0}
    # a flipped extracted answer or stop reason breaks the verdict, named
    _a3, new2 = _publish(s, "parity_x2", "none",
                         _rows3({1: {"extracted": 7.0, "correct": True},
                                 2: {"stop_reason": "stop_string",
                                     "terminated": True}}),
                         _IDENT, runtime=CACHED_RT)
    att2 = gp.compare(pub, new2)
    assert att2["verdict"]["outcomes_identical"] is False
    assert att2["identical_outcomes"] == 1
    assert [r["qidx"] for r in att2["extracted_differs"]] == [1]
    assert [r["qidx"] for r in att2["stop_reason_differs"]] == [2]
    assert att2["mcnemar"]["b_only"] == 1
    # a different run (batch size) is refused outright
    _a4, other = _publish(s, "parity_x3", "none", _rows3(),
                          {**_IDENT, "batch_size": 32}, runtime=CACHED_RT)
    with pytest.raises(SystemExit, match="batch_size differs"):
        gp.compare(pub, other)
    assert gp.mcnemar_exact_p(0, 0) == 1.0 and gp.mcnemar_exact_p(0, 8) < 0.01
    # the sides must be the two IMPLEMENTATIONS — a cached
    # file cannot stand as the published side, and the new side must be a
    # cached re-score with its aggregate
    with pytest.raises(SystemExit, match="must be the CACHE-FREE side"):
        gp.compare(new, new2)
    _a5, uncached_new = _publish(s, "parity_x4", "none", _rows3(), _IDENT)
    with pytest.raises(SystemExit, match="must be the CACHED re-score"):
        gp.compare(pub, uncached_new)
    _a6, off_new = _publish(s, "parity_x5", "none", _rows3(), _IDENT,
                            runtime={**CACHED_RT, "kv_cache": False})
    with pytest.raises(SystemExit, match="must be the CACHED re-score"):
        gp.compare(pub, off_new)
    # an explicitly cache-free published side (a --allow-cache-free run
    # that recorded its runtime) is accepted as the legacy side
    (tmp_path / "r2off").mkdir()
    _a7, pub_off = _publish(tmp_path / "r2off", "r2off", "none", _rows3(),
                            _IDENT, runtime={**CACHED_RT, "kv_cache": False})
    assert gp.compare(pub_off, new)["verdict"]["outcomes_identical"] is True


def test_gate_is_silent_when_no_mixing_can_occur(tmp_path):
    d = tmp_path / "r2"
    d.mkdir()
    _publish(d, "r2", "none", _rows3(), _IDENT)
    # (a) no legacy cell left un-rescored: the invocation re-scores `none`
    assert bg.require_decode_parity(d, "r2", ["none", "c1__psr"], _IDENT,
                                    CACHED_RT, None) is None
    # (b) a model the pinned transformers cached natively (gemma/qwen/GLM):
    #     its legacy cells were cached too — nothing to attest. This is
    #     the STABLE marker; what this load registered is irrelevant
    plain = {**CACHED_RT, "pre_fix_cache_free": [],
             "cache_placeholder_registered": []}
    assert bg.require_decode_parity(d, "r2", ["c1__psr"], _IDENT, plain,
                                    None) is None
    # (b') ...and a Nemotron load that registered nothing (second load in
    #     the process, or transformers >= 5.15) still trips the gate
    second = {**CACHED_RT, "cache_placeholder_registered": []}
    with pytest.raises(SystemExit, match="refusing to mix"):
        bg.require_decode_parity(d, "r2", ["c1__psr"], _IDENT, second, None)
    # (c) a cache-free run (diagnosis) mixes nothing new
    off = {**CACHED_RT, "kv_cache": False}
    assert bg.require_decode_parity(d, "r2", ["c1__psr"], _IDENT, off,
                                    None) is None
    # (d) an empty label
    assert bg.require_decode_parity(tmp_path / "r9", "r9", ["c1__psr"],
                                    _IDENT, CACHED_RT, None) is None


def test_gate_refuses_mixing_without_or_with_a_bad_attestation(tmp_path):
    d = tmp_path / "r2"
    d.mkdir()
    _agg, pub = _publish(d, "r2", "none", _rows3(), _IDENT)
    _publish(d, "r2", "c1__psr", _rows3(), _IDENT)
    with pytest.raises(SystemExit, match="refusing to mix"):
        bg.require_decode_parity(d, "r2", ["c2__psr"], _IDENT, CACHED_RT,
                                 None)
    with pytest.raises(SystemExit, match="does not exist"):
        bg.require_decode_parity(d, "r2", ["c2__psr"], _IDENT, CACHED_RT,
                                 str(tmp_path / "missing.json"))
    s = tmp_path / "scratch"
    s.mkdir()
    # verdict False -> refused, naming the counts
    _a, bad_new = _publish(s, "p1", "none",
                           _rows3({1: {"stop_reason": "cap",
                                       "terminated": False}}),
                           _IDENT, runtime=CACHED_RT)
    p_bad, _ = _attest(tmp_path, pub, bad_new, "bad.json")
    with pytest.raises(SystemExit, match="do NOT agree on every row"):
        bg.require_decode_parity(d, "r2", ["c2__psr"], _IDENT, CACHED_RT,
                                 str(p_bad))
    # good attestation, wrong run identity -> refused on the key
    _a, good_new = _publish(s, "p2", "none", _rows3(), _IDENT,
                            runtime=CACHED_RT)
    p_good, _ = _attest(tmp_path, pub, good_new, "good.json")
    with pytest.raises(SystemExit, match="batch_size = 128"):
        bg.require_decode_parity(d, "r2", ["c2__psr"],
                                 {**_IDENT, "batch_size": 64}, CACHED_RT,
                                 str(p_good))
    # attests a cell that is not one of THIS label's legacy cells
    other = tmp_path / "other"
    other.mkdir()
    _a, opub = _publish(other, "r7", "c9__psr", _rows3(), _IDENT)
    _a, onew = _publish(s, "p3", "c9__psr", _rows3(), _IDENT,
                        runtime=CACHED_RT)
    p_other, _ = _attest(tmp_path, opub, onew, "other.json")
    with pytest.raises(SystemExit, match="not one of label 'r2'"):
        bg.require_decode_parity(d, "r2", ["c2__psr"], _IDENT, CACHED_RT,
                                 str(p_other))
    # an attestation from a FITTED legacy cell is refused even
    # though every other check passes — it is a legacy cell of this label, its
    # sha and digest bind to this label's file, its published side is
    # cache-free, and the two paths agree ON THAT CELL. Agreement on one
    # fitted object says nothing about the others, and accepting it would
    # stamp the whole label as proven. Only the object-less bare cell counts.
    _a, fpub = _publish(d, "r2", "c1f__psr", _rows3(), _IDENT)
    _a, fnew = _publish(s, "p_fit", "c1f__psr", _rows3(), _IDENT,
                        runtime=CACHED_RT)
    p_fit, att_fit = _attest(tmp_path, fpub, fnew, "fitted.json")
    assert att_fit["verdict"]["outcomes_identical"] is True
    assert att_fit["shared_identity"]["cell"] == "c1f__psr"
    assert att_fit["shared_identity"]["object_sha256"] is not None
    # c1f__psr is NOT in the re-scored set, so it stays a legacy cell of this
    # label and the attestation clears every earlier check
    with pytest.raises(SystemExit, match="object-less 'none' cell"):
        bg.require_decode_parity(d, "r2", ["c2__psr"], _IDENT,
                                 CACHED_RT, str(p_fit))
    # and a hand-edited attestation that renames itself to `none` while
    # keeping an object sha is refused too (the object check is independent)
    forged_none = json.loads(p_good.read_text())
    forged_none["shared_identity"]["object_sha256"] = "s" * 8
    p_forged_none = tmp_path / "forged_none.json"
    p_forged_none.write_text(json.dumps(forged_none))
    with pytest.raises(SystemExit, match="object-less 'none' cell"):
        bg.require_decode_parity(d, "r2", ["c2__psr"], _IDENT, CACHED_RT,
                                 str(p_forged_none))
    # attests a same-named file with DIFFERENT outcomes than this label's
    d2 = tmp_path / "r2b"
    d2.mkdir()
    _publish(d2, "r2b", "none", _rows3({0: {"extracted": 1.0,
                                             "correct": False}}), _IDENT)
    _publish(d2, "r2b", "c1__psr", _rows3(), _IDENT)
    with pytest.raises(SystemExit, match="attestations are per label"):
        bg.require_decode_parity(d2, "r2b", ["c2__psr"],
                                 {**_IDENT, "run_label": "r2b"}, CACHED_RT,
                                 str(p_good))
    # a hand-edited attestation whose published side claims a cached
    # runtime: sha and digest match this label's file, still refused —
    # the proof must have exercised the cache-free path
    forged = json.loads(p_good.read_text())
    forged["published"]["runtime"] = dict(CACHED_RT)
    p_forged = tmp_path / "forged.json"
    p_forged.write_text(json.dumps(forged))
    with pytest.raises(SystemExit, match="never exercised the cache-free"):
        bg.require_decode_parity(d, "r2", ["c2__psr"], _IDENT, CACHED_RT,
                                 str(p_forged))
    # a digest edited to match while the bytes do not: the sha binding wins
    forged2 = json.loads(p_good.read_text())
    forged2["published"]["per_question_sha256"] = "0" * 64
    p_forged2 = tmp_path / "forged2.json"
    p_forged2.write_text(json.dumps(forged2))
    with pytest.raises(SystemExit, match="attestations are per label"):
        bg.require_decode_parity(d, "r2", ["c2__psr"], _IDENT, CACHED_RT,
                                 str(p_forged2))
    # attested cached run on another transformers -> refused
    with pytest.raises(SystemExit, match="is not this runtime"):
        bg.require_decode_parity(d, "r2", ["c2__psr"], _IDENT,
                                 {**CACHED_RT, "transformers": "5.15.0"},
                                 str(p_good))
    # wrong kind
    junk = tmp_path / "junk.json"
    junk.write_text(json.dumps({"kind": "other"}))
    with pytest.raises(SystemExit, match="is not a"):
        bg.require_decode_parity(d, "r2", ["c2__psr"], _IDENT, CACHED_RT,
                                 str(junk))


def test_gate_admits_and_stamps_a_good_attestation(tmp_path):
    d = tmp_path / "r2"
    d.mkdir()
    _agg, pub = _publish(d, "r2", "none", _rows3(), _IDENT)
    _publish(d, "r2", "c1__psr", _rows3(), _IDENT)
    s = tmp_path / "scratch"
    s.mkdir()
    _a, new = _publish(s, "parity_x1", "none",
                       _rows3({2: {"response": "different text, same cap"}}),
                       _IDENT, runtime=CACHED_RT)
    p, att = _attest(tmp_path, pub, new)
    got = bg.require_decode_parity(d, "r2", ["c2__psr"], _IDENT, CACHED_RT,
                                   str(p))
    assert got["cell"] == "none" and got["legacy_cells"] == ["c1__psr", "none"]
    assert got["n"] == 3 and got["identical_outcomes"] == 3
    assert got["attestation"] == str(p) and len(got["attestation_sha256"]) == 64
    # an explicitly cache-free legacy side (a runtime block with kv_cache
    # False) is admitted too
    d3 = tmp_path / "r2c"
    d3.mkdir()
    _a, pub3 = _publish(d3, "r2c", "none", _rows3(), _IDENT,
                        runtime={**CACHED_RT, "kv_cache": False})
    _publish(d3, "r2c", "c1__psr", _rows3(), _IDENT)
    _a, new3 = _publish(s, "parity_c", "none", _rows3(), _IDENT,
                        runtime=CACHED_RT)
    p3, _ = _attest(tmp_path, pub3, new3, "att3.json")
    assert bg.require_decode_parity(d3, "r2c", ["c2__psr"],
                                    {**_IDENT, "run_label": "r2c"},
                                    CACHED_RT, str(p3))["cell"] == "none"
    # NO cross-label reuse: the r2_x2 twin of the object-less
    # none has the same OUTCOMES, but it is a different file — an
    # attestation written from r2's file is refused for it; write one from
    # r2_x2's own published file instead
    d2 = tmp_path / "r2_x2"
    d2.mkdir()
    _a, pub2 = _publish(d2, "r2_x2", "none", _rows3(), _IDENT, scale=2.0)
    ident2 = {**_IDENT, "run_label": "r2_x2", "scale": 2.0}
    with pytest.raises(SystemExit, match="attestations are per label"):
        bg.require_decode_parity(d2, "r2_x2", ["c2__psr"], ident2, CACHED_RT,
                                 str(p))
    _a, new2 = _publish(s, "parity_x2", "none", _rows3(), _IDENT,
                        runtime=CACHED_RT, scale=2.0)
    p2, _ = _attest(tmp_path, pub2, new2, "att_x2.json")
    assert bg.require_decode_parity(d2, "r2_x2", ["c2__psr"], ident2,
                                    CACHED_RT, str(p2))["legacy_cells"] == \
        ["none"]


def test_cli_wires_decode_parity_outside_the_identity():
    src = (REPO / "scripts/sdfbase/base_gsm8k.py").read_text()
    assert "--decode-parity" in src
    main_src = src.split("def main(", 1)[1]
    assert "require_decode_parity(run_dir, run_label, cells, run_identity," \
        in main_src
    assert 'runtime["decode_parity"] = parity' in main_src
    ident = main_src.split("run_identity = {")[1].split("}")[0]
    assert "decode_parity" not in ident and "runtime" not in ident


def test_generate_answers_logs_a_pace_line_per_chunk(tiny_llama, capsys):
    model, tok = tiny_llama
    prompts = [bg.render_prompt(f"Q{i}?", 1) for i in range(5)]
    out = bg.generate_answers(model, tok, None, prompts, batch_size=2,
                              max_new_tokens=3)
    assert len(out) == 5
    lines = [ln for ln in capsys.readouterr().out.splitlines()
             if "s/step" in ln]
    assert len(lines) == 3  # ceil(5 / 2) chunks
    assert lines[0].startswith("[base-gsm8k] chunk 1/3:")
    assert "3 steps" in lines[0]
