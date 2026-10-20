"""Class C — the decode-only (positions="response") FIT wave, and the
fit x serve 2x2 it opens.

Class A/B/C is a FIT-TIME taxonomy: an object's class is how it was FITTED;
the serve phase is the eval cell's own explicit choice
(jailbreak._set_response_positions), so all four fit x serve cells are in
play. Class C reuses the Class A driver's existing
mask leg (no new fit machinery). This file pins:

- ``default_class_c_arms`` mirrors the A/B helpers (config-first, module
  fallback, registry-filtered), and the config's membership + exclusions;
- the ``--positions`` CLI override reaches the builders and the provenance
  record (``positions`` + ``positions_at_fit`` in train_budget_meta.json),
  and a masked wave REQUIRES its own ``--out`` (Class C cell ids collide
  with the Class A wave's in the shared defenses dir);
- FIT-LEG PERSISTENCE per member (the 2026-08-21 fixes): a response-fit
  artifact must be distinguishable from an all-fit one after the fact —
  the torch-blob objects (psr*/lora*/lofit/diffmean) persist AND restore
  the leg (the DiffMeanObject/LoReFT precedent), while the SteeringVector
  path (vector/bidpo) records it in the sidecar metadata as provenance and
  hydrates all-position by that path's pinned convention (test_bidpo.py);
- the serve-side seam's 2x2: an all fit served response actually masks, a
  response fit served all is ACTIVELY reset (the previously unreachable
  cell), all/all is unchanged, never-maskable objects keep their no-op
  under "all" and refuse "response" with their real reason;
- backward compatibility: an artifact saved WITHOUT a positions key loads
  as its historical all-token fit.

CPU-only: the tiny_llama fixture and synthetic pairs; no network, no GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import train_two_si_objects as t2

from sisteer.data.pairs import Pair
from sisteer.eval.defense_cell import object_path_for
from sisteer.eval.jailbreak import _set_response_positions
from sisteer.extraction.diffmean import DiffMeanObject
from sisteer.extraction.lofit import LoFiTObject
from sisteer.extraction.multilayer import (MultiLayerLoRAObject,
                                           MultiLayerVectorObject)
from sisteer.extraction.objects import RankOneLoRAObject, SoftTokenObject
from sisteer.extraction.psr import PSRObject
from sisteer.hooks import SteeringHandle
from sisteer.methods import (ARMS, VectorObject, load_object,
                             object_artifact_name)
from sisteer.vectors import SteeringVector

REPO = Path(__file__).resolve().parents[1]
TWO_SI_CONFIG = REPO / "configs" / "two_si.yaml"

SI_ID = "safety_refusal"
H = 2  # synthetic assistant-header allowance — the machinery is convention-
       # agnostic (the production H=4 comes from the pinned tokenizer)

# The recorded exclusions, each with its distinct reason (mirrors the
# config's class_c_arms comment block, so a silent re-add fails a test).
USER_RULED_OUT = ("alphakl_null", "alphakl_soft")       # stay Class A
PRINCIPLED_REFUSALS = ("token_bos", "token_faithful")   # prompt sentinel
JUDGEMENT_CALL_OUT = ("loreft",)                        # identity inversion
FIT_INDEPENDENT_OUT = ("diffmean",)   # capture hardcodes position="last" on
                                      # prompt-only renderings: a response
                                      # "fit" is a relabel of the bitwise-
                                      # identical Class A vector (2026-08-25)

CLASS_C_EXPECTED = ["psr_kl_all", "psr_mse_all", "lora_r1", "lofit",
                    "bidpo", "vector", "bidpo_lora"]


def _cfg() -> dict:
    return yaml.safe_load(TWO_SI_CONFIG.read_text())


def _pairs():
    data = [
        ("Say something nice.", "Bonjour le monde", "Hello world"),
        ("Describe a tree.", "C'est un grand arbre", "It is a tall tree"),
        ("Greet the reader.", "Salut a tous", "Hi everyone"),
    ]
    return [Pair(prompt=p, pos=pos, neg=neg, instruction_id=SI_ID)
            for p, pos, neg in data]


def _with_header(monkeypatch, h: int = H):
    """The serve seam derives H from the tokenizer; the tiny fixture has no
    chat template, so patch the derivation (the test_positions_steering
    convention — never mutate the shared fixture's tokenizer)."""
    monkeypatch.setattr(
        "sisteer.eval.jailbreak.response_header_len", lambda tok: h)


# --------------------------------------------------------------------------- #
# the config record + the helper                                               #
# --------------------------------------------------------------------------- #


def test_shipped_config_class_c_members_and_exclusions():
    """The shipped class_c_arms holds every maskable-fit Class A arm whose
    artifact persists its fit leg, in class_a order; every recorded
    exclusion stays out; and (fit-time taxonomy) every member is a Class A
    arm being re-fit on the decode-only leg."""
    cfg = _cfg()
    assert cfg["class_c_arms"] == CLASS_C_EXPECTED
    for arm in (USER_RULED_OUT + PRINCIPLED_REFUSALS + JUDGEMENT_CALL_OUT
                + FIT_INDEPENDENT_OUT):
        assert arm not in cfg["class_c_arms"], (
            f"{arm} is a recorded class_c exclusion — see the class_c_arms "
            "comment block in configs/two_si.yaml before re-adding it")
    assert set(cfg["class_c_arms"]) <= set(cfg["class_a_arms"]), (
        "Class C is the decode-only FIT of Class A arms (fit-time "
        "taxonomy) — a member outside class_a_arms is a different arm set, "
        "not a mask leg")
    # class_a (arm-major) order, restricted — the submission-time record
    a_order = [a for a in cfg["class_a_arms"] if a in set(cfg["class_c_arms"])]
    assert cfg["class_c_arms"] == a_order


def test_default_class_c_arms_mirrors_the_a_and_b_helpers():
    # config wins (the submission-time record) and is registry-filtered
    assert t2.default_class_c_arms(
        {"class_c_arms": ["diffmean", "not_an_arm"]}) == ["diffmean"]
    # absent/None falls back to the module tuple, filtered to the registry —
    # the exact derivation of default_class_a_arms / default_class_b_arms
    arms = t2.default_class_c_arms({"class_c_arms": None})
    assert arms == [a for a in t2.CLASS_C_ARMS if a in ARMS]
    assert arms  # non-empty
    # and the shipped config resolves to its own list
    assert t2.default_class_c_arms(_cfg()) == CLASS_C_EXPECTED


# --------------------------------------------------------------------------- #
# the --positions override (the Class C submission path)                       #
# --------------------------------------------------------------------------- #


def test_cfg_with_positions_overrides_a_copy_and_validates():
    cfg = {"train_kwargs": {"default": {"epochs": 2, "positions": "all"},
                            "alphakl_soft": {"beta": 1.0}}}
    out = t2._cfg_with_positions(cfg, "response")
    assert out["train_kwargs"]["default"]["positions"] == "response"
    # other defaults and per-arm blocks survive the merge
    assert out["train_kwargs"]["default"]["epochs"] == 2
    assert out["train_kwargs"]["alphakl_soft"] == {"beta": 1.0}
    # a COPY — the caller's cfg (and anything derived from it) never shifts
    assert cfg["train_kwargs"]["default"]["positions"] == "all"
    with pytest.raises(SystemExit, match="--positions must be one of"):
        t2._cfg_with_positions(cfg, "prompt")  # not a FIT position mode


def test_effective_positions_flag_over_config():
    cfg = {"train_kwargs": {"default": {"positions": "all"}}}
    ns = argparse.Namespace(positions="response")
    assert t2._effective_positions(ns, cfg) == "response"
    ns = argparse.Namespace(positions="")
    assert t2._effective_positions(ns, cfg) == "all"
    # a config already carrying the response leg is honoured without a flag
    cfg_r = {"train_kwargs": {"default": {"positions": "response"}}}
    assert t2._effective_positions(ns, cfg_r) == "response"


def test_plan_positions_response_defaults_to_class_c_arms(capsys):
    """`plan --positions response` with no --arms plans the config's
    class_c_arms — the one-command Class C submission path, plan side."""
    cfg = {"layer": 1, "max_len": 256,
           "train_kwargs": {"default": {"epochs": 1, "seed": 0}},
           "class_a_arms": ["lora_r1", "diffmean"],
           "class_c_arms": ["diffmean"]}
    args = argparse.Namespace(si=SI_ID, arms="", pairs="", with_model=False,
                              positions="response")
    t2.cmd_plan(args, cfg)
    out = capsys.readouterr().out
    assert "positions=response" in out
    assert "1 arms" in out           # class_c_arms, not the 2 Class A arms
    assert "diffmean" in out


def test_train_masked_wave_requires_its_own_out(tmp_path):
    """The collision guard fires in milliseconds, BEFORE any pairs/model
    load: a masked wave without --out would resume against (or clobber) the
    Class A artifacts at the same cell ids."""
    args = argparse.Namespace(
        si=SI_ID, arms="", pairs=str(tmp_path / "does_not_exist.jsonl"),
        out="", overwrite=False, positions="response",
        bidpo_harmful_pairs="", bidpo_benign_pairs="")
    with pytest.raises(SystemExit) as ei:
        t2.cmd_train(args, _cfg())
    # the --out guard, not the missing-pairs error: it must fire first
    assert "--out" in str(ei.value)
    assert "Class A" in str(ei.value)


# --------------------------------------------------------------------------- #
# --positions response reaches the builder AND the provenance record           #
# --------------------------------------------------------------------------- #


def test_positions_response_reaches_builder_and_provenance(tiny_llama,
                                                           tmp_path):
    """One masked train_si_objects run, one lockstep arm (lora_r1 — the
    build_lockstep_object path) + one solo arm (diffmean — the registry-
    trainer path): every artifact's train_budget_meta.json records
    positions="response" AND positions_at_fit="response" (the fit-phase
    provenance the evals read to assert or deliberately cross the
    serve phase), and both artifacts re-hydrate still on their fit leg."""
    model, tok = tiny_llama
    cfg = {
        "layer": 1, "completion_tokens": 16, "max_len": 128,
        "max_len_by_si": {SI_ID: 128},
        "train_kwargs": {"default": {
            "epochs": 1, "seed": 0, "positions": "response",
            # explicit H: the tiny tokenizer has no chat template to derive
            # it from (production leaves this null and derives H=4)
            "response_header_len": H,
            # the no-op-floor guard is a training-quality gate, not what
            # this test measures — off, so a tiny random model can't flake
            "floor_ratio_max": None,
        }},
        "si_ids": [SI_ID],
    }
    cfg = t2._cfg_with_positions(cfg, "response")  # the CLI flag's exact path
    outcome = t2.train_si_objects(model, tok, SI_ID, _pairs(),
                                  ["lora_r1", "diffmean"], cfg=cfg,
                                  save_dir=tmp_path)
    assert outcome.ok, {a: (r.error_type, r.error)
                        for a, r in outcome.failed.items()}

    for arm in ("lora_r1", "diffmean"):
        meta = json.loads(
            (tmp_path / f"{SI_ID}__{arm}" / "train_budget_meta.json")
            .read_text())
        assert meta["positions"] == "response", arm
        assert meta["positions_at_fit"] == "response", arm
        assert meta["response_header_len"] == H, arm
        # fit-leg persistence, live on this run's real artifacts (the
        # 2026-08-21 fixes: before them lora_r1 re-hydrated ALL-POSITION
        # and was byte-indistinguishable from a Class A fit)
        loaded = load_object(object_path_for(tmp_path, f"{SI_ID}__{arm}"),
                             model=model, tok=tok)
        assert loaded.fit_positions == "response", arm
        assert loaded.fit_response_header_len == H, arm


def test_all_token_wave_records_positions_at_fit_all(tiny_llama, tmp_path):
    """The default (Class A) wave stamps positions_at_fit="all" — the field
    exists on EVERY artifact, so distinguishing the classes after the fact
    never depends on a missing-key convention."""
    model, tok = tiny_llama
    cfg = {"layer": 1, "completion_tokens": 16, "max_len": 128,
           "max_len_by_si": {SI_ID: 128},
           "train_kwargs": {"default": {"epochs": 1, "seed": 0,
                                        "floor_ratio_max": None}},
           "si_ids": [SI_ID]}
    outcome = t2.train_si_objects(model, tok, SI_ID, _pairs(), ["diffmean"],
                                  cfg=cfg, save_dir=tmp_path)
    assert outcome.ok
    meta = json.loads(
        (tmp_path / f"{SI_ID}__diffmean" / "train_budget_meta.json")
        .read_text())
    assert meta["positions"] == "all"
    assert meta["positions_at_fit"] == "all"


# --------------------------------------------------------------------------- #
# fit-leg persistence, per member                                              #
# --------------------------------------------------------------------------- #

# Per-arm response-mode object factories: each builds the arm's REAL
# artifact class the way its trainer does, on the fit's mask leg. Adding an
# arm to class_c_arms without adding its factory here fails the persistence
# test with instructions — the audit can never silently lapse.
def _masked(obj):
    obj.set_fit_positions("response", H)
    return obj


def _masked_sv(method: str, model):
    """The SteeringVector-path artifact (vector / bidpo): the fit leg rides
    the sidecar metadata — exactly what train_distill, train_bidpo and the
    lockstep _VectorArmObject stamp."""
    return VectorObject(SteeringVector(
        vectors={1: torch.randn(model.config.hidden_size)}, method=method,
        model_name="", instruction_id=SI_ID,
        metadata={"positions": "response", "response_header_len": H}))


_ARTIFACT_FACTORIES = {
    "psr_kl_all": lambda model, tok: _masked(
        PSRObject(model, [0, 1], train_direction=True)),
    "psr_mse_all": lambda model, tok: _masked(
        PSRObject(model, [0, 1], train_direction=False)),
    "lora_r1": lambda model, tok: _masked(RankOneLoRAObject(model, layer=1)),
    # lora_r1_all: the all-layer sibling joins Class C —
    # MultiLayerLoRAObject mirrors RankOneLoRAObject's
    # PositionMaskedFit tracker and persists the fit leg in its blob.
    "lora_r1_all": lambda model, tok: _masked(
        MultiLayerLoRAObject(model, layers=[0, 1])),
    # lora_r4 (2026-08-28): the lora arm's post-re-tune alias — the same
    # MultiLayerLoRAObject machinery at the re-tuned rank, so the same
    # persistence contract holds at rank 4.
    "lora_r4": lambda model, tok: _masked(
        MultiLayerLoRAObject(model, layers=[0, 1], rank=4)),
    "bidpo_lora": lambda model, tok: _masked(
        RankOneLoRAObject(model, layer=1)),
    "lofit": lambda model, tok: _masked(
        LoFiTObject(model, [(0, 0), (1, 1)], base_scale=1.0)),
    "diffmean": lambda model, tok: _masked(DiffMeanObject(
        model, {1: torch.randn(model.config.hidden_size)})),
    "vector": lambda model, tok: _masked_sv("distill_soft", model),
    "bidpo": lambda model, tok: _masked_sv("bidpo", model),
}

# vector/bidpo: hydration stays a plain all-position VectorObject and the
# serve seam sets the phase (the SteeringVector-path convention pinned by
# tests/test_bidpo.py) — the fit leg is still persisted, as sidecar
# provenance, so the artifact is never byte-indistinguishable from Class A.
_SIDECAR_PROVENANCE_ARMS = {"vector", "bidpo"}


def test_fit_leg_persists_for_every_class_c_member(tiny_llama, tmp_path,
                                                   monkeypatch):
    _with_header(monkeypatch)
    model, tok = tiny_llama
    for arm in _cfg()["class_c_arms"]:
        if arm not in _ARTIFACT_FACTORIES:
            pytest.fail(
                f"class_c_arms gained {arm!r} with no artifact factory in "
                "tests/test_class_c_arms.py::_ARTIFACT_FACTORIES — fit-leg "
                "persistence must be verified per member (a non-persisting "
                "artifact is byte-indistinguishable from a Class A fit); "
                "add a factory, don't drop the check")
        obj = _ARTIFACT_FACTORIES[arm](model, tok)
        p = tmp_path / arm / object_artifact_name(arm)
        p.parent.mkdir(parents=True, exist_ok=True)
        obj.save(p)
        loaded = load_object(p, model=model, tok=tok)
        if arm in _SIDECAR_PROVENANCE_ARMS:
            assert loaded.sv.metadata["positions"] == "response", arm
            assert loaded.sv.metadata["response_header_len"] == H, arm
            assert loaded.positions == "all", arm  # pinned convention
            # the serve seam arms it explicitly (matched Class C serving)
            _set_response_positions(loaded, "response", tok)
            assert loaded.positions == "response", arm
            assert loaded.response_header_len == H, arm
        else:
            assert loaded.fit_positions == "response", (
                f"{arm}: the artifact did not restore its fit leg — it "
                "would re-hydrate as an ALL-POSITION object and be "
                "evaluated as a decode-only cell while running the "
                "all-token method")
            assert loaded.fit_response_header_len == H, arm


def test_legacy_artifacts_without_positions_key_load_as_all(tiny_llama,
                                                            tmp_path):
    """Backward compatibility: an artifact saved BEFORE the persistence fix
    (no positions key in the blob) loads as its historical all-token fit —
    no previously-fit object changes behaviour."""
    model, tok = tiny_llama

    legacy = {
        "lora": {"kind": "lora",
                 "A": torch.randn(model.config.hidden_size),
                 "B": torch.zeros(model.config.hidden_size), "layer": 1},
        "psr": {"kind": "psr", "layers": [1], "train_direction": True,
                "directions": {"1": torch.zeros(model.config.hidden_size)},
                "gate_w": {"1": torch.zeros(model.config.hidden_size)},
                "gate_b": {"1": torch.ones(1)}},
    }
    for name, blob in legacy.items():
        p = tmp_path / f"{name}.pt"
        torch.save(blob, p)
        loaded = load_object(p, model=model, tok=tok)
        assert loaded.fit_positions == "all", name
        assert loaded.fit_response_header_len is None, name


# --------------------------------------------------------------------------- #
# the serve seam: all four fit x serve cells                                   #
# --------------------------------------------------------------------------- #


def test_serve_seam_2x2_on_a_position_masked_fit_object(tiny_llama, tmp_path,
                                                        monkeypatch):
    """RankOneLoRAObject (the PositionMaskedFit family): all four cells."""
    _with_header(monkeypatch)
    model, tok = tiny_llama

    def _save(obj):
        p = tmp_path / "o.pt"
        obj.save(p)
        return p

    # fit=all / serve=all — unchanged (the standard Class A leg)
    obj = load_object(_save(RankOneLoRAObject(model, layer=1)),
                      model=model, tok=tok)
    _set_response_positions(obj, "all", tok)
    assert obj.fit_positions == "all"
    with obj.apply(model, scale=1.0) as handle:
        assert handle is None  # no mask tracker: every position steered

    # fit=all / serve=response — cross-phase: the serve seam MASKS it
    obj = load_object(_save(RankOneLoRAObject(model, layer=1)),
                      model=model, tok=tok)
    _set_response_positions(obj, "response", tok)
    assert obj.fit_positions == "response"
    assert obj.fit_response_header_len == H
    with obj.apply(model, scale=1.0) as handle:
        assert isinstance(handle, SteeringHandle)
        assert handle.positions == "response"
        assert handle.response_header_len == H

    # fit=response / serve=response — matched Class C
    obj = load_object(_save(_masked(RankOneLoRAObject(model, layer=1))),
                      model=model, tok=tok)
    _set_response_positions(obj, "response", tok)
    assert obj.fit_positions == "response"

    # fit=response / serve=all — the previously UNREACHABLE cell: an
    # explicit "all" ACTIVELY resets (the old early-return silently left
    # the object in response mode — the wrong leg)
    obj = load_object(_save(_masked(RankOneLoRAObject(model, layer=1))),
                      model=model, tok=tok)
    assert obj.fit_positions == "response"  # hydrated on its fit leg
    _set_response_positions(obj, "all", tok)
    assert obj.fit_positions == "all"
    with obj.apply(model, scale=1.0) as handle:
        assert handle is None  # every position steered again


def test_serve_seam_2x2_on_the_vector_deployment_path(tiny_llama, tmp_path,
                                                      monkeypatch):
    """VectorObject (vector / bidpo): the deployment-attr seam."""
    _with_header(monkeypatch)
    model, tok = tiny_llama
    p = tmp_path / object_artifact_name("vector")
    _masked_sv("distill_soft", model).save(p)

    obj = load_object(p, model=model, tok=tok)      # hydrates all-position
    _set_response_positions(obj, "response", tok)   # serve=response
    assert obj.positions == "response" and obj.response_header_len == H
    _set_response_positions(obj, "all", tok)        # serve=all resets
    assert obj.positions == "all" and obj.response_header_len is None


def test_serve_seam_routes_multilayer_vector_through_its_fit_leg(
        tiny_llama, tmp_path, monkeypatch):
    """MultiLayerVectorObject holds BOTH seams; a persisted response fit
    re-hydrates masked and an explicit all-serve resets both legs."""
    _with_header(monkeypatch)
    model, tok = tiny_llama
    p = tmp_path / "o.pt"
    _masked(MultiLayerVectorObject(model, layers=[0, 1])).save(p)
    loaded = load_object(p, model=model, tok=tok)
    assert loaded.fit_positions == "response"       # persisted + restored
    _set_response_positions(loaded, "all", tok)
    assert loaded.fit_positions == "all"
    assert loaded.positions == "all"

    # MultiLayerLoRAObject: same blob convention as RankOneLoRAObject
    p2 = tmp_path / "o2.pt"
    _masked(MultiLayerLoRAObject(model, layers=[0, 1])).save(p2)
    assert load_object(p2, model=model, tok=tok).fit_positions == "response"


def test_alphakl_serve_side_mask_works_and_fit_side_still_refuses(
        tiny_llama, monkeypatch):
    """The two halves of the 2026-08-21 alphakl ruling, pinned together:

    SERVE half — AlphaKLObject masks at serve time through the same seam as
    every maskable arm, so its decode-only cell runs under the POSITION
    MASK (first generated token steered), never the engine-phase gate:
    masked-out prompt positions stay BIT-identical, in-mask real positions
    are steered.

    FIT half — the principled refusal STAYS: positions="response" is never
    an accepted FIT for the ladder (a masked fit solves for the same (w, r)
    and merely relabels the artifact), and the arm stays out of
    class_c_arms. If this half fails, serve-side masking leaked into the
    fit — a regression, not progress."""
    _with_header(monkeypatch)
    model, tok = tiny_llama
    d = model.config.hidden_size
    torch.manual_seed(0)
    from sisteer.extraction.alphakl import AlphaKLObject, train_alphakl

    obj = AlphaKLObject(model, [0, 1],
                        ws={0: torch.randn(d), 1: torch.randn(d)},
                        rs={0: torch.randn(d), 1: torch.randn(d)},
                        base_scale=1.0,
                        clamp_gate=False)  # raw gate: nonzero a.s., so every
                                           # in-mask position visibly steers
    _set_response_positions(obj, "response", tok)
    assert obj.fit_positions == "response"
    assert obj.fit_response_header_len == H

    enc = tok(["Say something nice to the reader today.",
               "Describe a very tall tree."],
              return_tensors="pt", padding=True)
    with torch.no_grad():
        base = model(**enc, output_hidden_states=True).hidden_states
    with torch.no_grad(), obj.apply(model, scale=1.0) as handle:
        assert isinstance(handle, SteeringHandle)
        assert handle.positions == "response"
        # full-sequence forward: per-row boundaries, the training-forward
        # convention (generation derives them from H per prefill instead)
        bounds = enc["attention_mask"].sum(dim=1) - H
        assert bool((bounds > 0).all())
        handle.set_prompt_lens(bounds)
        steered = model(**enc, output_hidden_states=True).hidden_states
        b, s = enc["input_ids"].shape
        mask = handle._position_mask(b, s, enc["input_ids"].device).bool()
    on_real = mask & enc["attention_mask"].bool()
    assert bool((~mask).any()) and bool(on_real.any())
    after = base[2], steered[2]  # output of steered layer 1
    assert torch.equal(after[1][~mask], after[0][~mask]), (
        "masked-out positions must stay BIT-identical")
    assert bool((after[1][on_real] != after[0][on_real]).any()), (
        "in-mask real positions must actually steer")

    # ...and the serve-side mask must not have opened a fit-side door
    with pytest.raises(ValueError, match="cannot honor a positions"):
        train_alphakl(model, tok, None, _pairs(), layer=1, stage=1,
                      positions="response", response_header_len=H)
    for arm in USER_RULED_OUT:
        assert arm not in _cfg()["class_c_arms"]


def test_serve_seam_preserves_no_op_and_refuses_with_the_real_reason(
        tiny_llama, monkeypatch):
    """Objects with NEITHER seam: "all" stays the historical no-op (nothing
    about plain Class A/B serving changes — antidote's intrinsic last-token
    deployment is untouched), and "response" refuses with the object's own
    reason (FIT_MASK_REFUSAL for the soft-token arms)."""
    _with_header(monkeypatch)
    model, tok = tiny_llama

    # a soft token: "all" no-op, "response" the sentinel-embedding refusal
    st = SoftTokenObject(model, tok, sentinel=(tok.unk_token,
                                               tok.unk_token_id))
    _set_response_positions(st, "all", tok)         # no-op, no error
    assert not hasattr(st, "positions")             # no stray attr grown
    with pytest.raises(ValueError, match="soft PROMPT token"):
        _set_response_positions(st, "response", tok)

    # antidote: intrinsic positions="last" lives inside apply(), not an
    # attribute — the seam must not (and cannot) touch it
    from sisteer.methods import AntidoteObject
    anti = AntidoteObject(SteeringVector(
        vectors={1: torch.randn(model.config.hidden_size)},
        method="antidote", model_name="", instruction_id="",
        metadata={"base_alpha": 1.0}))
    _set_response_positions(anti, "all", tok)       # no-op, no error
    assert not hasattr(anti, "positions")
    with pytest.raises(ValueError, match="cannot honour a position mask"):
        _set_response_positions(anti, "response", tok)


def test_default_build_silently_unmasks_a_response_fit_artifact(
        tiny_llama, tmp_path, monkeypatch):
    """PINS THE HAZARD'S PRESENCE, not just the correct call.

    Three lanes independently tripped over this and two of us gave a third
    lane OPPOSITE answers about it, because it is genuinely two stages:

      STAGE 1 (load): a response-fit artifact DOES carry its mask and DOES
        re-arm on load — diffmean saves a kind-tagged torch blob, so it
        routes through _KIND_LOADERS and DiffMeanObject.load_blob's
        set_fit_positions, NOT the SteeringVector safetensors path that
        deliberately drops the fit leg.
      STAGE 2 (serve): `_set_response_positions` ACTIVELY SETS rather than
        early-returning on "all", so a DEFAULT caller silently unmasks it.

    Net: the phase survives loading and is then thrown away unless the
    caller passes "response" explicitly. Testing only the explicit call
    would not notice the default changing again — and the failure is
    invisible, because an all-position object serving under a
    response-phase label produces perfectly well-formed output. This is
    the third near-miss from a default reset; assert the trap is still
    where we think it is."""
    _with_header(monkeypatch)
    model, tok = tiny_llama

    obj = _masked(DiffMeanObject(
        model, {1: torch.randn(model.config.hidden_size)}))
    p = tmp_path / "dm" / "object.pt"
    p.parent.mkdir(parents=True, exist_ok=True)
    obj.save(p)

    # STAGE 1: the artifact carries the fit leg and re-arms on load.
    loaded = load_object(p, model=model, tok=tok)
    assert loaded.fit_positions == "response", (
        "a response-fit diffmean artifact must re-hydrate MASKED; if this "
        "fails the artifact is byte-indistinguishable from a Class A fit")
    assert loaded.fit_response_header_len == H

    # STAGE 2: a DEFAULT "all" build throws that away, silently.
    _set_response_positions(loaded, "all", tok)
    assert loaded.fit_positions == "all", (
        "the default-'all' serve path no longer resets a masked object. "
        "That may be an improvement — but it CHANGES what every default "
        "caller serves, so update this test deliberately rather than "
        "letting the behaviour drift")

    # And the explicit call preserves it — the only safe way to serve these.
    again = load_object(p, model=model, tok=tok)
    _set_response_positions(again, "response", tok)
    assert again.fit_positions == "response"
    assert again.fit_response_header_len == H


# --------------------------------------------------------------------------- #
# T6 (twosi-model-sweeps): lora_r1_all is a REAL Class C member on this base   #
# (masked fit honoured end to end + fit leg persisted), token_faithful's      #
# masked fit refuses with the FIT_MASK_REFUSAL rationale — and the driver's   #
# fallback tuple + the derived per-model configs keep class_c inside class_a  #
# order.                                                                       #
# --------------------------------------------------------------------------- #

# The driver's fallback tuple, pinned (lora_r1_all after its single-layer
# sibling, then lora_r4 — the arm's 2026-08-28 post-re-tune alias, same
# machinery — after IT; the arm-major order, restricted).
CLASS_C_FALLBACK_EXPECTED = ("psr_kl_all", "psr_mse_all", "lora_r1",
                             "lora_r1_all", "lora_r4", "lofit",
                             "bidpo", "vector", "bidpo_lora")


def test_driver_fallback_class_c_gains_lora_r1_all():
    assert t2.CLASS_C_ARMS == CLASS_C_FALLBACK_EXPECTED
    assert "lora_r1_all" in _ARTIFACT_FACTORIES  # the persistence audit rows
    # every fallback member has a factory, so a config naming any of them
    # (the per-model files include lora_r1_all) is covered by the
    # persistence test above
    missing = [a for a in t2.CLASS_C_ARMS if a not in _ARTIFACT_FACTORIES]
    assert not missing, missing


def test_derived_model_configs_keep_class_c_inside_class_a_order():
    """Every per-model sweep config (configs/two_si_*.yaml — a peer derives
    them; skipped until they land) must keep class_c ⊆ class_a in class_a
    order; wherever the sweep carries lora_r1_all as a Class A arm it must
    also be Class C (it satisfies both criteria on this base).
    The membership pin is conditional because the glob has since gained
    configs derived from the BASE arm lists, which never carried
    lora_r1_all at all (two_si_llama_hpo.yaml, 2026-08-26) — those are not
    the per-model sweep world this pin is about."""
    derived = [p for p in (REPO / "configs").glob("two_si_*.y*ml")]
    if not derived:
        pytest.skip("no derived per-model configs yet (WS3 lands them)")
    seen_lora_r1_all = 0
    for p in derived:
        cfg = yaml.safe_load(p.read_text())
        if "class_c_arms" not in cfg or "class_a_arms" not in cfg:
            continue  # not a sweep config
        assert set(cfg["class_c_arms"]) <= set(cfg["class_a_arms"]), p.name
        a_order = [a for a in cfg["class_a_arms"]
                   if a in set(cfg["class_c_arms"])]
        assert cfg["class_c_arms"] == a_order, p.name
        if "lora_r1_all" in cfg["class_a_arms"]:
            assert "lora_r1_all" in cfg["class_c_arms"], p.name
            seen_lora_r1_all += 1
    # the pin must still BITE: the per-model sweep configs carry the arm
    assert seen_lora_r1_all >= 2, "no config exercises the lora_r1_all pin"


def test_lora_r1_all_masked_fit_round_trips_through_the_driver(tiny_llama,
                                                               tmp_path):
    """The full T6 round-trip on the REAL object through the production
    driver: a response-leg lora_r1_all fit trains (no documented-hole
    branch), stamps positions_at_fit='response' in its sidecar, and the
    reloaded artifact HONOURS the fit leg (fit_positions + H restored)."""
    model, tok = tiny_llama
    cfg = {
        "layer": 1, "completion_tokens": 16, "max_len": 128,
        "max_len_by_si": {SI_ID: 128},
        "train_kwargs": {"default": {
            "epochs": 1, "seed": 0, "positions": "response",
            "response_header_len": H, "floor_ratio_max": None,
        }},
        "si_ids": [SI_ID],
    }
    cfg = t2._cfg_with_positions(cfg, "response")
    outcome = t2.train_si_objects(model, tok, SI_ID, _pairs(),
                                  ["lora_r1_all"], cfg=cfg,
                                  save_dir=tmp_path)
    assert outcome.ok, {a: (r.error_type, r.error)
                        for a, r in outcome.failed.items()}
    cell = tmp_path / f"{SI_ID}__lora_r1_all"
    meta = json.loads((cell / "train_budget_meta.json").read_text())
    assert meta["positions_at_fit"] == "response"
    assert meta["response_header_len"] == H
    loaded = load_object(object_path_for(tmp_path, f"{SI_ID}__lora_r1_all"),
                         model=model, tok=tok)
    assert loaded.fit_positions == "response"
    assert loaded.fit_response_header_len == H
    # and the published cell carries the COMPLETE marker (the Class C store
    # gets the same atomic-publish contract as Class A)
    assert (cell / t2.COMPLETE_MARKER_NAME).exists()


def test_token_faithful_masked_fit_refuses_with_fit_mask_refusal(
        tiny_llama, tmp_path):
    """token_faithful has NO response-only cell, by design (spec Fixed 5):
    the driver records the arm FAILED with the soft-prompt-token rationale
    (never a silent all-token fit under a decode-only label), and the
    object class itself carries FIT_MASK_REFUSAL for the kl-trainer path
    (objects.py) — both halves of the refusal, pinned together."""
    from sisteer.extraction.tokens import AppendedSoftTokenObject

    model, tok = tiny_llama
    cfg = {
        "layer": 1, "completion_tokens": 16, "max_len": 128,
        "max_len_by_si": {SI_ID: 128},
        "train_kwargs": {"default": {
            "epochs": 1, "seed": 0, "positions": "response",
            "response_header_len": H, "floor_ratio_max": None,
        }},
        "si_ids": [SI_ID],
    }
    cfg = t2._cfg_with_positions(cfg, "response")
    outcome = t2.train_si_objects(
        model, tok, SI_ID, _pairs(), ["token_faithful"], cfg=cfg,
        save_dir=tmp_path, sentinel=(tok.unk_token, tok.unk_token_id))
    rec = outcome.arms["token_faithful"]
    assert rec.status == "failed"
    assert "soft PROMPT token" in (rec.error or "")
    assert "decode-only" in (rec.error or "")
    # nothing was published for the hole
    assert not (tmp_path / f"{SI_ID}__token_faithful").exists()
    # the object-level constant exists for the kl-trainer refusal path
    assert "soft PROMPT token" in AppendedSoftTokenObject.FIT_MASK_REFUSAL
