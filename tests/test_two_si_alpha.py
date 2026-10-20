"""The eval-time serving-coefficient (alpha / dose) grid for the two-SI matrix.

One fitted object, scored at several doses, with NO refit anywhere: the
``--alpha`` (alias ``--coeff``) override on ``two_si_run.py generate`` /
``aggregate``. CPU-only, no network beyond the cached tiny test model, no
harmful text (the synthetic-prompt conventions of test_two_si_run.py).

Covers, per the change's contract:

- DEFAULT-PATH INVARIANCE: no ``--alpha`` (including every pre-existing
  caller whose args namespace has no ``alpha`` attribute at all), and an
  explicit ``--alpha`` equal to the config's pre-registered ``scale``, both
  produce today's private paths and today's serving coefficient exactly;
- the override reaches the APPLICATION-TIME coefficient for both hook
  shapes in ``sisteer.hooks`` ([d] broadcast and [P, d] per-position), for
  PSR's gated edit (``h + scale*ReLU(w.h+b)*d`` — the serving ``scale`` is
  the same coefficient, applied outside the gate), and for both Class B
  arms (antidote / alphasteer, whose ``apply`` composes ``base_* * scale``)
  — verified as an exact 0.25x delta on a tiny synthetic model;
- PATH/KEY SEPARATION: a non-default dose writes to its own
  ``<private_run>_alpha<A>`` namespace, so doses never collide, while the
  default dose keeps today's tree byte-identically (no migration);
- METADATA: every cell_meta sidecar and the aggregate record BOTH
  ``fit_scale`` (the pre-registered fit-time operating point) and
  ``served_alpha`` (the dose the transcripts were generated under), so a
  transcript can never be read as served at the dose it was fit at;
- the DOSE-CONSISTENCY GATE: the aggregate refuses — before any grading
  spend — transcripts whose recorded served_alpha contradicts the dose it
  is about to grade (including unrecorded doses in an off-fit tree, and
  off-fit records sitting in the default tree), while legacy sidecars
  without the record stay acceptable at the pre-registered dose only.

The bound (also in the code): serving an object fit at scale 1.0 at
alpha=0.25 is an UNDER-APPLICATION; a win there says the deployed dose is
too strong for THIS object, never that 0.25 is the right fit target.
"""

from __future__ import annotations

import json
import stat
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import two_si_run  # noqa: E402
from sisteer.eval.defense_cell import (  # noqa: E402
    alpha_run_namespace,
    format_alpha,
    registry_from_cfg,
)
from sisteer.eval.jailbreak_data import Goal  # noqa: E402
from sisteer.methods import ARMS  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TWO_SI_CONFIG = REPO / "configs" / "two_si.yaml"


def _goals():
    return [Goal("g0", "benign goal zero text", "illegal", "harmbench", "rev"),
            Goal("g1", "benign goal one text", "illegal", "harmbench", "rev")]


def _xs_goals():
    return [Goal("xs0", "a benign safe prompt zero", "homonyms", "xstest",
                 "rev", split="safe")]


@pytest.fixture()
def cfg(tmp_path):
    c = yaml.safe_load(TWO_SI_CONFIG.read_text())
    c["defenses_dir"] = str(tmp_path / "defenses" / "cells")
    c["out_dir"] = str(tmp_path / "out")
    c["pool_manifest"] = str(tmp_path / "pool_manifest.json")
    c["private_run"] = "two_si_test"
    c["bon"] = {**c["bon"], "n_max": 2, "max_new_tokens": 8}
    return c


def _registry(cfg):
    return registry_from_cfg(cfg, repo_root=REPO)


def _gen_args(root, cell_id, **over):
    base = dict(cell=cell_id, attack="bon", shard=0, nshards=1, trial=0,
                confirmatory=False, private_root=str(root), xstest=False)
    base.update(over)
    return SimpleNamespace(**base)


@pytest.fixture()
def gen_seams(monkeypatch):
    """The offline generate seams of test_two_si_run.py: no model, fake
    defense build (recording the serving kwargs), fake bon_generate
    (recording the out dir)."""
    import sisteer.eval.attacks.bon as bon_mod
    import sisteer.eval.jailbreak as jb
    import sisteer.models as models_mod

    monkeypatch.setattr(models_mod, "load_model_and_tokenizer",
                        lambda *a, **k: (None, None))
    monkeypatch.setattr(two_si_run, "_eval_goals",
                        lambda cfg, *, confirmatory, attack=None:
                        _goals())
    built: list = []
    ran: list = []

    def fake_build_defense_cell(model, tok, cell, **kw):
        built.append((cell, kw))
        return SimpleNamespace(mode=cell.arm or "none", n_target_forwards=7)

    def fake_bon_generate(defense, goals, *, out_dir, n_max, sigma, trial,
                          max_new_tokens, batch_size=None):
        ran.append(Path(out_dir))

    monkeypatch.setattr(jb, "build_defense_cell", fake_build_defense_cell)
    monkeypatch.setattr(bon_mod, "bon_generate", fake_bon_generate)
    return built, ran


# --------------------------------------------------------------------------- #
# keying + parsing helpers                                                     #
# --------------------------------------------------------------------------- #


def test_alpha_namespace_default_is_identity_and_doses_separate():
    # the pre-registered dose keeps TODAY'S namespace — byte-identical path
    assert alpha_run_namespace("two_si_v1", 1.0, 1.0) == "two_si_v1"
    assert alpha_run_namespace("two_si_v1", 0.5, 0.5) == "two_si_v1"
    # any other dose gets its own tree, distinct per dose
    a = alpha_run_namespace("two_si_v1", 0.25, 1.0)
    b = alpha_run_namespace("two_si_v1", 0.5, 1.0)
    assert a == "two_si_v1_alpha0.25" and b == "two_si_v1_alpha0.5"
    assert len({a, b, "two_si_v1"}) == 3
    # canonical short formatting
    assert format_alpha(1.0) == "1" and format_alpha(0.25) == "0.25"


def test_served_alphas_parsing(cfg):
    # a pre-existing caller has NO alpha attribute at all -> the fit scale
    assert two_si_run._served_alphas(SimpleNamespace(), cfg) == [cfg["scale"]]
    assert two_si_run._served_alphas(SimpleNamespace(alpha=None), cfg) == \
        [cfg["scale"]]
    # repeatable + comma-separated + order-preserving dedupe
    got = two_si_run._served_alphas(
        SimpleNamespace(alpha=["0.25", "0.5,1", "1.0"]), cfg)
    assert got == [0.25, 0.5, 1.0]
    with pytest.raises(SystemExit, match="not a number"):
        two_si_run._served_alphas(SimpleNamespace(alpha=["lots"]), cfg)


def test_cli_accepts_alpha_and_coeff_alias(monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(two_si_run, "cmd_generate",
                        lambda args, cfg: seen.update(alpha=args.alpha))
    monkeypatch.setattr(sys, "argv", [
        "two_si_run.py", "generate", "--cell", "none",
        "--alpha", "0.25", "--coeff", "0.5,1"])
    two_si_run.main()
    assert seen["alpha"] == ["0.25", "0.5,1"]  # both spellings, one dest


# --------------------------------------------------------------------------- #
# generate — default-path invariance, per-dose keying, metadata                #
# --------------------------------------------------------------------------- #


def test_generate_default_path_unchanged(cfg, tmp_path, gen_seams):
    """No --alpha == today's behaviour: today's private dir, the
    pre-registered serving coefficient — and an EXPLICIT --alpha equal to
    the config scale lands in the SAME tree (the dose, not the flag, keys
    the path)."""
    built, ran = gen_seams
    root = tmp_path / "state"
    for args in (_gen_args(root, "none"),                       # no attr edits
                 _gen_args(root, "none", alpha=[format_alpha(cfg["scale"])])):
        two_si_run.cmd_generate(args, cfg)
        assert str(ran[-1]).endswith(f"{cfg['private_run']}/none")
        _, kw = built[-1]
        assert kw["scale"] == cfg["scale"]
    assert ran[0] == ran[1]  # explicitly-default dose == flag-less path
    meta = json.loads((ran[-1] / "cell_meta_bon_shard00.json").read_text())
    # both coefficients recorded, equal at the default dose
    assert meta["fit_scale"] == cfg["scale"]
    assert meta["served_alpha"] == cfg["scale"]


def test_generate_alpha_keys_paths_and_records_both_coefficients(
        cfg, tmp_path, gen_seams):
    built, ran = gen_seams
    root = tmp_path / "state"
    arm = next(a for a in cfg["class_a_arms"] if a in ARMS)
    cell_id = f"{cfg['si_ids'][0]}__{arm}"
    two_si_run.cmd_generate(_gen_args(root, cell_id, alpha=["0.25"]), cfg)

    # the serving coefficient is the dose, NOT the config scale
    _, kw = built[-1]
    assert kw["scale"] == 0.25 != cfg["scale"]
    # its own namespace, still keyed on cell.id inside, still mode-0700
    assert str(ran[-1]).endswith(f"{cfg['private_run']}_alpha0.25/{cell_id}")
    assert stat.S_IMODE(ran[-1].stat().st_mode) == 0o700
    assert stat.S_IMODE(ran[-1].parent.stat().st_mode) == 0o700
    # the sidecar records BOTH numbers, and they differ
    meta = json.loads((ran[-1] / "cell_meta_bon_shard00.json").read_text())
    assert meta["served_alpha"] == 0.25
    assert meta["fit_scale"] == cfg["scale"] == 1.0
    assert meta["private_run"] == f"{cfg['private_run']}_alpha0.25"


def test_generate_multi_alpha_runs_each_dose_once(cfg, tmp_path, gen_seams):
    """A comma-separated grid runs one full pass per dose against the SAME
    loaded model, deduplicated, with the fit-scale dose landing in today's
    tree and the off-fit dose in its own."""
    built, ran = gen_seams
    root = tmp_path / "state"
    two_si_run.cmd_generate(
        _gen_args(root, "none", alpha=["0.25,1", "1.0"]), cfg)
    assert [kw["scale"] for _, kw in built] == [0.25, 1.0]  # deduped, in order
    assert str(ran[0]).endswith(f"{cfg['private_run']}_alpha0.25/none")
    assert str(ran[1]).endswith(f"{cfg['private_run']}/none")  # today's tree
    assert ran[0] != ran[1]  # doses never collide
    metas = [json.loads((d / "cell_meta_bon_shard00.json").read_text())
             for d in ran]
    assert [m["served_alpha"] for m in metas] == [0.25, 1.0]
    assert all(m["fit_scale"] == 1.0 for m in metas)


def test_generate_xstest_alpha_keys_and_meta(cfg, tmp_path, monkeypatch):
    """The xstest pass (read by the aggregate from the same tree) is
    dose-keyed and dose-recorded like the attack passes."""
    import sisteer.eval.jailbreak as jb
    import sisteer.eval.jailbreak_data as jd
    import sisteer.models as models_mod

    monkeypatch.setattr(models_mod, "load_model_and_tokenizer",
                        lambda *a, **k: (None, None))
    monkeypatch.setattr(jd, "load_xstest_safe", lambda **kw: _xs_goals())
    scales: list = []

    def fake_build_defense_cell(model, tok, cell, **kw):
        scales.append(kw["scale"])
        return SimpleNamespace(
            mode="none", n_target_forwards=0,
            generate=lambda texts, max_new_tokens:
                ["a benign completion"] * len(texts))

    monkeypatch.setattr(jb, "build_defense_cell", fake_build_defense_cell)
    root = tmp_path / "state"
    two_si_run.cmd_generate(
        _gen_args(root, "none", xstest=True, alpha=["0.25"]), cfg)
    d = root / f"{cfg['private_run']}_alpha0.25" / "none"
    assert scales == [0.25]
    assert (d / "xstest_responses.jsonl").exists()
    meta = json.loads((d / "cell_meta_xstest.json").read_text())
    assert meta["served_alpha"] == 0.25 and meta["fit_scale"] == 1.0


def test_generate_gated_attacker_loaded_once_across_doses(cfg, tmp_path,
                                                          monkeypatch):
    """The gated tiers' attacker seams are built ONCE per invocation and
    shared across the dose grid — never re-loaded per alpha."""
    import jailbreak_run
    import sisteer.eval.jailbreak as jb
    import sisteer.models as models_mod

    monkeypatch.setattr(models_mod, "load_model_and_tokenizer",
                        lambda *a, **k: (None, None))
    monkeypatch.setattr(two_si_run, "_eval_goals",
                        lambda cfg, *, confirmatory, attack=None:
                        _goals())
    monkeypatch.setattr(jb, "build_defense_cell",
                        lambda model, tok, cell, **kw:
                        SimpleNamespace(mode="none", n_target_forwards=1))
    loads = {"n": 0}

    def fake_load_attacker(cfg):
        loads["n"] += 1
        return (None, None)

    monkeypatch.setattr(jailbreak_run, "_load_attacker", fake_load_attacker)
    monkeypatch.setattr(jailbreak_run, "_attacker_sampling",
                        lambda cfg, attack: {"temperature": 0.0, "top_p": 1.0})
    for seam in ("_make_scorer_fn", "_make_attacker_factory",
                 "_make_attacker_fn"):
        monkeypatch.setattr(jailbreak_run, seam, lambda *a, **k: object())
    out_dirs: list = []

    def fake_gated(attack, defense, goals, **kw):
        out_dirs.append(Path(kw["out_dir"]))
        return {"n_goals_run": len(goals), "n_goals_total": len(goals),
                "n_target_forwards_total": 5}

    monkeypatch.setattr(jailbreak_run, "_run_gated_attack", fake_gated)

    root = tmp_path / "state"
    two_si_run.cmd_generate(
        _gen_args(root, "none", attack="pair", alpha=["0.25,1"]), cfg)
    assert loads["n"] == 1                      # one attacker, two doses
    assert len(out_dirs) == 2
    assert str(out_dirs[0]).endswith(
        f"{cfg['private_run']}_alpha0.25/pair/none/shard00")
    assert str(out_dirs[1]).endswith(f"{cfg['private_run']}/pair/none/shard00")
    for d, a in zip(out_dirs, (0.25, 1.0)):
        meta = json.loads((d / "cell_meta.json").read_text())
        assert meta["served_alpha"] == a and meta["fit_scale"] == 1.0


# --------------------------------------------------------------------------- #
# the coefficient actually scales the applied delta (tiny synthetic model)     #
# --------------------------------------------------------------------------- #


def _delta_after_layer(model, enc, layer, ctx_factory):
    """Hidden-state delta at the steered layer's OUTPUT, between a steered
    forward and the plain forward — captured as the INPUT of layer+1 via a
    forward PRE-hook, i.e. the exact tensor the rest of the network
    consumes. (output_hidden_states is NOT used: this transformers version
    records each layer's output before externally-registered forward hooks
    replace it, so it cannot see the intervention.)"""
    import torch

    from sisteer.models import get_decoder_layers

    captured: dict = {}

    def pre(_mod, args, kwargs):
        h = kwargs.get("hidden_states", args[0] if args else None)
        captured["h"] = h.detach().clone()
        return None

    handle = get_decoder_layers(model)[layer + 1].register_forward_pre_hook(
        pre, with_kwargs=True)
    try:
        with torch.no_grad():
            model(**enc)
            base = captured["h"]
            with ctx_factory():
                model(**enc)
            steered = captured["h"]
    finally:
        handle.remove()
    return steered - base


def test_alpha_quarter_scales_broadcast_hook_delta(tiny_llama):
    """hooks.py's [d] broadcast shape ((v * self.scale).view(1,1,-1)):
    alpha=0.25 applies EXACTLY 0.25x the alpha=1.0 delta."""
    import torch

    from sisteer.hooks import steering

    model, tok = tiny_llama
    layer = 0  # tiny model: layer 1's hidden_states slot is post-final-norm
    torch.manual_seed(0)
    vec = torch.randn(model.config.hidden_size)
    enc = tok("a tiny synthetic prompt", return_tensors="pt")
    d100 = _delta_after_layer(
        model, enc, layer,
        lambda: steering(model, {layer: vec}, scale=1.0, skip_bos=False))
    d025 = _delta_after_layer(
        model, enc, layer,
        lambda: steering(model, {layer: vec}, scale=0.25, skip_bos=False))
    assert d100.abs().max() > 0
    assert torch.allclose(d025, 0.25 * d100, atol=1e-5)


def test_alpha_quarter_scales_per_position_hook_delta(tiny_llama):
    """hooks.py's [P, d] per-position shape (add[...] = v[lo:hi] *
    self.scale, the ActAdd front-injection carrier): same exact 0.25x."""
    import torch

    from sisteer.hooks import steering

    model, tok = tiny_llama
    layer, p = 0, 3
    torch.manual_seed(0)
    mat = torch.randn(p, model.config.hidden_size)
    enc = tok("a tiny synthetic prompt with several tokens",
              return_tensors="pt")
    assert enc["input_ids"].shape[1] > p
    d100 = _delta_after_layer(
        model, enc, layer,
        lambda: steering(model, {layer: mat}, scale=1.0, positions="front"))
    d025 = _delta_after_layer(
        model, enc, layer,
        lambda: steering(model, {layer: mat}, scale=0.25, positions="front"))
    assert d100[:, :p].abs().max() > 0
    assert torch.allclose(d100[:, p:], torch.zeros_like(d100[:, p:]),
                          atol=1e-6)  # front rows only
    assert torch.allclose(d025, 0.25 * d100, atol=1e-5)


def test_alpha_quarter_scales_psr_gated_edit(tiny_llama):
    """PSR's gated edit (h + scale*ReLU(w.h+b)*d) routes the SERVING scale
    through apply(scale=...) — there is no separate PSR coefficient, and the
    gate reads the pre-steering hidden, so the delta is exactly linear."""
    import torch

    from sisteer.extraction.psr import PSRObject

    model, tok = tiny_llama
    layer = 0  # tiny model: layer 1's hidden_states slot is post-final-norm
    obj = PSRObject(model, [layer])
    torch.manual_seed(0)
    obj.directions[layer].data = torch.randn_like(obj.directions[layer]) * 0.1
    obj.gate_w[layer].data = torch.randn_like(obj.gate_w[layer]) * 0.05
    # gate_b stays at its init of 1.0, so ReLU(w.h + 1) is mostly active
    enc = tok("a tiny synthetic prompt", return_tensors="pt")
    d100 = _delta_after_layer(model, enc, layer,
                              lambda: obj.apply(model, scale=1.0))
    d025 = _delta_after_layer(model, enc, layer,
                              lambda: obj.apply(model, scale=0.25))
    assert d100.abs().max() > 0
    assert torch.allclose(d025, 0.25 * d100, atol=1e-5)


def test_alpha_quarter_scales_class_b_alphasteer(tiny_llama):
    """Class B alphasteer: its own prefill hook composes base_scale * alpha,
    so alpha=0.25 is 0.25x the calibrated deployment coefficient."""
    import torch

    from sisteer.extraction.alphasteer import AlphaSteerObject

    model, tok = tiny_llama
    layer = 0  # tiny model: layer 1's hidden_states slot is post-final-norm
    d = model.config.hidden_size
    torch.manual_seed(0)
    obj = AlphaSteerObject(model, layer=layer, M=torch.randn(d, d) * 0.05,
                           base_scale=2.0)
    enc = tok("a tiny synthetic prompt", return_tensors="pt")
    d100 = _delta_after_layer(model, enc, layer,
                              lambda: obj.apply(model, scale=1.0))
    d025 = _delta_after_layer(model, enc, layer,
                              lambda: obj.apply(model, scale=0.25))
    assert d100.abs().max() > 0
    assert torch.allclose(d025, 0.25 * d100, atol=1e-5)


def test_alpha_quarter_scales_class_b_antidote(tiny_llama):
    """Class B antidote: steering()-backed at positions='last' with
    base_alpha * alpha — the last-token delta scales exactly."""
    import torch

    from sisteer.extraction.antidote import AntidoteObject
    from sisteer.vectors import SteeringVector

    model, tok = tiny_llama
    layer = 0  # tiny model: layer 1's hidden_states slot is post-final-norm
    torch.manual_seed(0)
    sv = SteeringVector(
        vectors={layer: torch.randn(model.config.hidden_size)},
        method="antidote", model_name="tiny", instruction_id="synthetic",
        metadata={"base_alpha": 2.0})
    obj = AntidoteObject(sv)
    enc = tok("a tiny synthetic prompt", return_tensors="pt")
    d100 = _delta_after_layer(model, enc, layer,
                              lambda: obj.apply(model, scale=1.0))
    d025 = _delta_after_layer(model, enc, layer,
                              lambda: obj.apply(model, scale=0.25))
    assert d100[:, -1].abs().max() > 0            # the current last token
    assert torch.allclose(d100[:, :-1], torch.zeros_like(d100[:, :-1]),
                          atol=1e-6)              # ... and ONLY it
    assert torch.allclose(d025, 0.25 * d100, atol=1e-5)


def test_alpha_reaches_obj_apply_through_the_serving_chain(tiny_llama,
                                                           monkeypatch):
    """End to end through the lane's serving surface: build_defense(scale=A)
    -> LLMHFSteered -> generate_with_object -> obj.apply(scale=A)."""
    import sisteer.eval.jailbreak as jb

    model, tok = tiny_llama
    seen: list = []

    class StubObj:
        kind = "vector"

        def inject(self, p):
            return p

        @contextmanager
        def apply(self, model, scale=1.0):
            seen.append(scale)
            yield

    monkeypatch.setattr(jb, "load_object", lambda path, m, t: StubObj())
    defense = jb.build_defense(model, tok, mode="vector_all",
                               object_path="unused", scale=0.25, batch_size=2,
                               positions="all")
    defense.generate(["a tiny synthetic prompt"], max_new_tokens=2)
    assert seen == [0.25]


# --------------------------------------------------------------------------- #
# aggregate — per-dose tree, recorded dose, and the consistency gate           #
# --------------------------------------------------------------------------- #


def _write_manifest(cfg):
    Path(cfg["pool_manifest"]).write_text(json.dumps(
        {"regime": "matched_intersection",
         "pool_sizes": {si: 657 for si in cfg["si_ids"]}}))


def _store(cfg, root, registry, *, namespace=None, sidecar=None):
    """Per-cell BoN + xstest transcripts under ``namespace`` (default:
    the config private_run). ``sidecar``: None writes NO cell_meta (a
    pre-alpha-seam legacy store); "legacy" writes one without served_alpha;
    a float writes one recording that served dose."""
    ns = namespace or cfg["private_run"]
    for cell in registry:
        d = Path(root) / ns / cell.id
        d.mkdir(parents=True, exist_ok=True)
        with (d / "bon_responses.jsonl").open("w") as f:
            for g in _goals():
                for k in range(cfg["bon"]["n_max"]):
                    f.write(json.dumps({
                        "goal_id": g.goal_id, "sample_idx": k, "trial": 0,
                        "variant": f"variant {k}",
                        "response": f"benign response {k} for {g.goal_id}",
                    }) + "\n")
        with (d / "xstest_responses.jsonl").open("w") as f:
            for g in _xs_goals():
                f.write(json.dumps({"goal_id": g.goal_id,
                                    "response": "benign compliance text"})
                        + "\n")
        if sidecar is not None:
            meta = {"cell_id": cell.id, "attack": "bon", "fit_scale": 1.0}
            if sidecar != "legacy":
                meta["served_alpha"] = float(sidecar)
            (d / "cell_meta_bon_shard00.json").write_text(json.dumps(meta))


def _run_aggregate(cfg, root, monkeypatch, **over):
    import sisteer.eval.jailbreak_data as jd

    monkeypatch.setattr(two_si_run, "_eval_goals",
                        lambda cfg, *, confirmatory, attack=None:
                        _goals())
    monkeypatch.setattr(jd, "load_xstest_safe", lambda **kw: _xs_goals())
    base = dict(attack="bon", trial=0, confirmatory=False,
                private_root=str(root), transport="live", concurrency=1,
                poll_interval=0, out=None)
    base.update(over)
    two_si_run.cmd_aggregate(
        SimpleNamespace(**base), cfg,
        grade_fn=lambda items: {it: 0.1 for it in items},
        xstest_grade_fn=lambda items: {it: "C" for it in items})


def test_aggregate_off_fit_dose_reads_its_tree_and_states_both_numbers(
        cfg, tmp_path, monkeypatch):
    _write_manifest(cfg)
    root = tmp_path / "state"
    registry = _registry(cfg)
    _store(cfg, root, registry, namespace=f"{cfg['private_run']}_alpha0.25",
           sidecar=0.25)
    _run_aggregate(cfg, root, monkeypatch, alpha=["0.25"])

    # per-dose output file — never overwrites the default-dose aggregate
    out = Path(cfg["out_dir"]) / "two_si_bon_alpha0.25_aggregate.json"
    assert out.exists()
    agg = json.loads(out.read_text())
    # BOTH coefficients at top level, different numbers, plus the bound
    assert agg["served_alpha"] == 0.25
    assert agg["fit_scale"] == 1.0
    assert "refit" in agg["dose_note"]  # the under-application bound travels
    # ... and the dose next to every score: per-cell too
    assert all(entry["served_alpha"] == 0.25
               for entry in agg["cells"].values())
    assert set(agg["cells"]) == {c.id for c in registry}


def test_aggregate_default_dose_keeps_todays_output_and_marks_dose(
        cfg, tmp_path, monkeypatch):
    """Default dose: today's filename and tree (a LEGACY store with no
    sidecars is still aggregable), served_alpha == fit_scale recorded, no
    dose_note."""
    _write_manifest(cfg)
    root = tmp_path / "state"
    _store(cfg, root, _registry(cfg), sidecar=None)   # pre-seam legacy store
    _run_aggregate(cfg, root, monkeypatch)            # no alpha attr at all
    out = Path(cfg["out_dir"]) / "two_si_bon_aggregate.json"
    assert out.exists()
    agg = json.loads(out.read_text())
    assert agg["served_alpha"] == agg["fit_scale"] == 1.0
    assert "dose_note" not in agg
    assert all(entry["served_alpha"] == 1.0 for entry in agg["cells"].values())


def test_aggregate_gate_refuses_a_recorded_dose_mismatch(cfg, tmp_path,
                                                         monkeypatch):
    """Transcripts RECORDED as served at 1.0 sitting in the alpha-0.25 tree
    (e.g. copied between dose trees) refuse loudly, before any grading."""
    _write_manifest(cfg)
    root = tmp_path / "state"
    _store(cfg, root, _registry(cfg),
           namespace=f"{cfg['private_run']}_alpha0.25", sidecar=1.0)
    with pytest.raises(SystemExit, match="SERVED at alpha=1"):
        _run_aggregate(cfg, root, monkeypatch, alpha=["0.25"])


def test_aggregate_gate_refuses_unverifiable_off_fit_transcripts(
        cfg, tmp_path, monkeypatch):
    """At an off-fit dose, a sidecar WITHOUT served_alpha — or no sidecar at
    all — is itself a mismatch: pre-seam code could only serve the config
    scale, so an unrecorded dose tree cannot be trusted as 0.25."""
    _write_manifest(cfg)
    root = tmp_path / "state"
    ns = f"{cfg['private_run']}_alpha0.25"
    _store(cfg, root, _registry(cfg), namespace=ns, sidecar="legacy")
    with pytest.raises(SystemExit, match="no served_alpha recorded"):
        _run_aggregate(cfg, root, monkeypatch, alpha=["0.25"])
    _store(cfg, root, _registry(cfg), namespace=ns, sidecar=None)
    for p in Path(root, ns).glob("*/cell_meta_bon_shard00.json"):
        p.unlink()
    with pytest.raises(SystemExit, match="no cell_meta sidecar"):
        _run_aggregate(cfg, root, monkeypatch, alpha=["0.25"])


def test_aggregate_gate_refuses_off_fit_records_in_the_default_tree(
        cfg, tmp_path, monkeypatch):
    """The reverse copy: transcripts recorded at 0.25 sitting in the DEFAULT
    tree refuse a default-dose aggregate — the gate is not a rubber stamp
    for the default path either."""
    _write_manifest(cfg)
    root = tmp_path / "state"
    _store(cfg, root, _registry(cfg), sidecar=0.25)
    with pytest.raises(SystemExit, match="SERVED at alpha=0.25"):
        _run_aggregate(cfg, root, monkeypatch)


def test_aggregate_accepts_matching_recorded_default_dose(cfg, tmp_path,
                                                          monkeypatch):
    """New-code default-dose runs record served_alpha == fit scale; the gate
    accepts them (regression against over-tight matching)."""
    _write_manifest(cfg)
    root = tmp_path / "state"
    _store(cfg, root, _registry(cfg), sidecar=1.0)
    _run_aggregate(cfg, root, monkeypatch)
    assert (Path(cfg["out_dir"]) / "two_si_bon_aggregate.json").exists()


def test_aggregate_refuses_a_multi_dose_invocation(cfg, tmp_path, monkeypatch):
    _write_manifest(cfg)
    root = tmp_path / "state"
    _store(cfg, root, _registry(cfg), sidecar=1.0)
    with pytest.raises(SystemExit, match="ONE served dose"):
        _run_aggregate(cfg, root, monkeypatch, alpha=["0.25,1"])
