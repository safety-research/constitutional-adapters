"""The twosi-model-sweeps FIT configs are generator-pinned.

scripts/make_fit_configs.py is the single source of the four committed
configs/two_si_*_fits*.yaml files; these tests regenerate and byte-compare,
so the files cannot drift from the generator, from each other (the dp3
variant differs from its sibling ONLY in data_parallel), or from the
transform's invariants: max_len 16384 for the clean-v3 (4096-token
completion) pools, fixed HPO-winner optimizer settings where a model's HPO
has landed, fail-loud sentinels/placeholders where it has not.
"""
import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "make_fit_configs", REPO / "scripts/make_fit_configs.py")
mfc = importlib.util.module_from_spec(spec)
sys.modules["make_fit_configs"] = mfc
spec.loader.exec_module(mfc)

FILES = ["two_si_llama_fits.yaml", "two_si_llama_fits_dp3.yaml",
         "two_si_qwen38_27b_fits.yaml", "two_si_gemma4e4b_fits.yaml"]


@pytest.fixture(scope="module")
def built():
    return mfc.build_all(None)


def test_committed_files_match_generator(built):
    for name in FILES:
        header, cfg = built[name]
        want = header + yaml.safe_dump(
            cfg, sort_keys=False, width=79, default_flow_style=False,
            allow_unicode=True)
        got = (REPO / "configs" / name).read_text()
        assert got == want, (
            f"configs/{name} drifted from scripts/make_fit_configs.py — "
            "regenerate with `python scripts/make_fit_configs.py` (plus "
            "--llama-winners once the Llama HPO lands) and commit both.")


def test_max_len_is_16384_with_per_si_floors(built):
    for name in FILES:
        cfg = built[name][1]
        assert cfg["max_len"] == 16384, name
        by_si = cfg["max_len_by_si"]
        # 4096-token v3 completions: the long SIs need 16384; the short
        # safety SI fits 8192 (prompt ~105 + 1024 + 4096 + 1 < 8192).
        assert by_si["si_constitution"] == 16384, name
        assert by_si["si_irrelevant"] == 16384, name
        assert by_si["safety_refusal"] == 8192, name


def test_winners_folded_where_hpo_landed(built):
    qwen = built["two_si_qwen38_27b_fits.yaml"][1]["train_kwargs"]
    e4b = built["two_si_gemma4e4b_fits.yaml"][1]["train_kwargs"]
    for arm in mfc.FIT_ARMS:
        for cfg_tk, winners in ((qwen, mfc.QWEN_INTERIM_WINNERS),
                                (e4b, mfc.E4B_WINNERS_V4)):
            assert cfg_tk[arm]["lr"] == winners[arm]["lr"]
            assert cfg_tk[arm]["warmup_frac"] == winners[arm]["warmup_frac"]
            assert (cfg_tk[arm]["max_grad_norm"]
                    == winners[arm]["max_grad_norm"])
            assert cfg_tk[arm]["lr_schedule"] == "warmup_cosine"


def test_llama_waits_on_its_hpo_loudly(built):
    """The sentinel discipline: an arm WITHOUT a landed winner carries the
    fail-loud sentinel; a landed arm carries real finite floats. As of
    2026-08-27 all three Llama fit arms have landed (re-tune@32,
    user-authorized), so the committed config must carry real values —
    the sentinel branch is exercised via build with emptied winners."""
    tk = built["two_si_llama_fits.yaml"][1]["train_kwargs"]
    for arm in mfc.FIT_ARMS:
        assert arm in mfc.LLAMA_LANDED_WINNERS
        assert float(tk[arm]["lr"]) > 0
        assert tk[arm]["weight_decay"] == 0.01
        assert tk[arm]["warmup_frac"] == 0.15
        assert tk[arm]["accum_steps"] == 16
    orig = mfc.LLAMA_LANDED_WINNERS
    mfc.LLAMA_LANDED_WINNERS = {}
    try:
        bare = mfc.build_all(None)["two_si_llama_fits.yaml"][1]["train_kwargs"]
        for arm in mfc.FIT_ARMS:
            assert bare[arm]["lr"] == mfc.LLAMA_PENDING
            with pytest.raises(ValueError):
                float(bare[arm]["lr"])
    finally:
        mfc.LLAMA_LANDED_WINNERS = orig


def test_llama_winner_fold_replaces_sentinels():
    winners = {a: {"lr": 1e-4 * (i + 1), "warmup_frac": 0.05,
                   "max_grad_norm": 0.5}
               for i, a in enumerate(mfc.FIT_ARMS)}
    out = mfc.build_all(winners)
    for name in ("two_si_llama_fits.yaml", "two_si_llama_fits_dp3.yaml"):
        tk = out[name][1]["train_kwargs"]
        for i, arm in enumerate(mfc.FIT_ARMS):
            assert tk[arm]["lr"] == pytest.approx(1e-4 * (i + 1))


def test_dp3_variant_differs_only_in_data_parallel(built):
    base = built["two_si_llama_fits.yaml"][1]
    dp3 = built["two_si_llama_fits_dp3.yaml"][1]
    assert dp3["data_parallel"] == {"gpus": 3}
    assert "data_parallel" not in base
    stripped = {k: v for k, v in dp3.items() if k != "data_parallel"}
    assert stripped == base


def test_llama_bs_measured_qwen_e4b_fail_loud_until_memprobe(built):
    llama = built["two_si_llama_fits.yaml"][1]["batch_size_by_arm"]
    for arm in mfc.FIT_ARMS:
        assert llama[arm] == 2  # H200/16384 worst-case probe, 0.85 guard
    e4b = built["two_si_gemma4e4b_fits.yaml"][1]["batch_size_by_arm"]
    for arm in mfc.FIT_ARMS:
        # 2026-08-27 offline-teacher re-probe: bs2 all arms (the old bs1
        # was measured with the ONLINE teacher resident; hpomem measured
        # 110.1/95.8/110.8 GiB reserved at bs2, bs4 OOM).
        assert e4b[arm] == 2
    for name in ("two_si_qwen38_27b_fits.yaml",
                 "two_si_gemma4e4b_fits.yaml"):
        table = built[name][1]["batch_size_by_arm"]
        for arm, v in table.items():
            if isinstance(v, str):
                with pytest.raises(ValueError):
                    int(v)  # resolve_batch_size int()s it -> loud


def test_fit_class_lists_are_the_sweeps(built):
    for name in FILES:
        cfg = built[name][1]
        assert cfg["class_a_arms"] == mfc.CLASS_A_ARMS, name
        assert cfg["class_c_arms"] == mfc.CLASS_C_ARMS, name


def test_gradient_checkpointing_is_qwen_only_per_arm(built):
    """All four Qwen fit arms checkpoint (needed to fit H200 at 16384;
    the hook arms since the backward-inside-apply() fix, parity-pinned by
    tests/test_ckpt_grad_parity.py). Per-arm, never the default block —
    a future incompatible arm should fail one cell, not the config. The
    Qwen config also runs teacher_prefix_cache: false, which the enable
    requires. Llama/E4B (measured without checkpointing) carry the flag
    nowhere."""
    qcfg = built["two_si_qwen38_27b_fits.yaml"][1]
    q = qcfg["train_kwargs"]
    for arm in mfc.FIT_ARMS:
        assert q[arm]["gradient_checkpointing"] is True
    assert "gradient_checkpointing" not in q["default"]
    assert qcfg["teacher_prefix_cache"] is False
    for name in ("two_si_llama_fits.yaml", "two_si_llama_fits_dp3.yaml",
                 "two_si_gemma4e4b_fits.yaml"):
        tk = built[name][1]["train_kwargs"]
        for block in tk.values():
            assert "gradient_checkpointing" not in block, name


def test_maybe_enable_gradient_checkpointing_driver_hook():
    """The driver enable: off by default (no config key -> False, model
    untouched), on when the config asks (use_reentrant=False + use_cache
    off), and a LOUD refusal for a model that cannot checkpoint."""
    spec2 = importlib.util.spec_from_file_location(
        "_t2_fitcfg", REPO / "scripts/train_two_si_objects.py")
    t2 = importlib.util.module_from_spec(spec2)
    sys.modules["_t2_fitcfg"] = t2
    spec2.loader.exec_module(t2)

    class _Cfg:
        def __init__(self, drop=0.0):
            self.use_cache = True
            self.attention_dropout = drop

    class _Model:
        supports_gradient_checkpointing = True

        def __init__(self, drop=0.0):
            self.config = _Cfg(drop)
            self.calls = []
            self.mode = "eval"

        def gradient_checkpointing_enable(self, **kw):
            self.calls.append(kw)

        def train(self):
            self.mode = "train"

    m = _Model()
    assert t2.maybe_enable_gradient_checkpointing(m, {}) is False
    assert m.calls == [] and m.config.use_cache is True

    on = {"teacher_prefix_cache": False,
          "train_kwargs": {"default": {"gradient_checkpointing": True}}}
    # per-arm override beats the default, both directions
    mixed = {"teacher_prefix_cache": False,
             "train_kwargs": {
                 "default": {"gradient_checkpointing": False},
                 "token_faithful": {"gradient_checkpointing": True}}}
    assert t2.maybe_enable_gradient_checkpointing(
        _Model(), mixed, arms=["psr_kl_all"]) is False
    with pytest.raises(SystemExit):  # mixed values in one process refuse
        t2.maybe_enable_gradient_checkpointing(
            _Model(), mixed, arms=["token_faithful", "psr_kl_all"])
    assert t2.maybe_enable_gradient_checkpointing(
        _Model(), mixed, arms=["token_faithful"]) is True
    assert t2.maybe_enable_gradient_checkpointing(m, on) is True
    assert m.calls == [{"gradient_checkpointing_kwargs":
                        {"use_reentrant": False}}]
    assert m.config.use_cache is False
    # THE engagement half: HF layers only consult the flag in train-mode
    # (the first probe wave "enabled" it in eval and changed nothing).
    assert m.mode == "train"

    # train-mode is refused when the forward would become stochastic
    with pytest.raises(SystemExit):
        t2.maybe_enable_gradient_checkpointing(_Model(drop=0.1), on)

    class _NoCkpt:
        supports_gradient_checkpointing = False

    with pytest.raises(SystemExit):
        t2.maybe_enable_gradient_checkpointing(_NoCkpt(), on)
    # a stub/None model (routing tests) stands down silently
    assert t2.maybe_enable_gradient_checkpointing(None, on) is False


def test_gradient_checkpointing_engages_on_a_real_hf_model():
    """In-memory tiny Llama (no download): the helper must leave the model
    ACTUALLY checkpointing — is_gradient_checkpointing AND training both
    true — since HF's per-layer wrapper is gated on both."""
    transformers = pytest.importorskip("transformers")
    spec2 = importlib.util.spec_from_file_location(
        "_t2_fitcfg2", REPO / "scripts/train_two_si_objects.py")
    t2 = importlib.util.module_from_spec(spec2)
    sys.modules["_t2_fitcfg2"] = t2
    spec2.loader.exec_module(t2)
    cfgm = transformers.LlamaConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, vocab_size=128)
    model = transformers.LlamaForCausalLM(cfgm).eval()
    on = {"teacher_prefix_cache": False,
          "train_kwargs": {"default": {"gradient_checkpointing": True}}}
    assert t2.maybe_enable_gradient_checkpointing(model, on) is True
    assert model.is_gradient_checkpointing
    assert model.training
    assert model.config.use_cache is False


def test_lockstep_guard_accepts_per_arm_gradient_checkpointing():
    """token_faithful is lockstep-compatible, so its per-arm
    gradient_checkpointing key must be registered driver-consumed — the
    guard refused the first Qwen tf canary before it was."""
    spec2 = importlib.util.spec_from_file_location(
        "_t2_fitcfg3", REPO / "scripts/train_two_si_objects.py")
    t2 = importlib.util.module_from_spec(spec2)
    sys.modules["_t2_fitcfg3"] = t2
    spec2.loader.exec_module(t2)
    cfg = {"train_kwargs": {"token_faithful":
                            {"gradient_checkpointing": True}}}
    assert t2.lockstep_dropped_keys(cfg, "token_faithful") == []
    t2.refuse_silently_dropped_kwargs(cfg, ["token_faithful"])  # no raise
    # an actually-unknown key still refuses
    cfg2 = {"train_kwargs": {"token_faithful": {"not_a_real_knob": 1}}}
    with pytest.raises(SystemExit):
        t2.refuse_silently_dropped_kwargs(cfg2, ["token_faithful"])


def test_solo_fit_consumes_per_arm_gradient_checkpointing(tiny_llama,
                                                          tmp_path):
    """End-to-end mini-fit: a SOLO arm (psr_mse_all) with the per-arm
    ckpt flag trains through the production driver on a CHECKPOINTED
    tiny model — the key must reach the model-load enable and NEVER the
    trainer signature (train_psr TypeError'd on it before the driver
    filter existed)."""
    torch = pytest.importorskip("torch")
    spec2 = importlib.util.spec_from_file_location(
        "_t2_fitcfg4", REPO / "scripts/train_two_si_objects.py")
    t2 = importlib.util.module_from_spec(spec2)
    sys.modules["_t2_fitcfg4"] = t2
    spec2.loader.exec_module(t2)

    from sisteer.data.pairs import Pair
    model, tok = tiny_llama
    pairs = [Pair(prompt=p, pos=pos, neg=neg,
                  instruction_id="safety_refusal")
             for p, pos, neg in
             [("Say something nice.", "Bonjour", "Hello"),
              ("Describe a tree.", "Grand arbre", "Tall tree"),
              ("Greet the reader.", "Salut", "Hi")]]
    cfg = {
        "layer": 1, "completion_tokens": 16, "max_len": 256,
        "max_len_by_si": {"safety_refusal": 256},
        # the real checkpointing consumers (the Qwen fits) run with the
        # teacher prefix cache OFF; ckpt + tpc refuses loudly (HF kills
        # use_cache under train-mode ckpt -> teacher KV build gets None)
        "teacher_prefix_cache": False,
        "train_kwargs": {
            "default": {"epochs": 1, "seed": 0, "holdout_frac": 0.0,
                        "floor_ratio_max": None, "probe_n": 2,
                        "monitors": False},
            "psr_mse_all": {"gradient_checkpointing": True},
        },
        "si_ids": ["safety_refusal"],
    }
    was_training = model.training
    try:
        with pytest.raises(SystemExit, match="teacher_prefix_cache"):
            t2.maybe_enable_gradient_checkpointing(
                model, {**cfg, "teacher_prefix_cache": True},
                arms=["psr_mse_all"])
        t2.maybe_enable_gradient_checkpointing(model, cfg,
                                               arms=["psr_mse_all"])
        assert model.is_gradient_checkpointing and model.training
        outcome = t2.train_si_objects(model, tok, "safety_refusal", pairs,
                                      ["psr_mse_all"], cfg=cfg,
                                      save_dir=tmp_path)
        assert outcome.ok, {a: (r.error_type, r.error)
                            for a, r in outcome.failed.items()}
    finally:  # session-scoped fixture: leave the model as found
        model.gradient_checkpointing_disable()
        model.train(was_training)
