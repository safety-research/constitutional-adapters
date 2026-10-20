"""Class B fit-once driver (scripts/fit_class_b.py): the content assembly
(the AlphaSteer authors' verbatim prompt-list seam + the min_prompts floor),
the single-layer-15 comparability pin, and the fit-and-save loop — each
object written the instant it completes, each arm isolated, resume by skip.
CPU-only, tiny-llama, synthetic content (no network, no cluster files, no
real mode-0700 store).

circuit_breaker left the registry ENTIRELY on 2026-08-20:
it is no longer a Class B arm, its store-reading reroute/retain machinery
left this driver, and the tests here pin the removal — while its fitted
artifact stays loadable (tests/test_circuit_breaker.py pins the loader).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import fit_class_b as fcb  # noqa: E402

from sisteer.data.content_sets import ContentContrast  # noqa: E402
from sisteer.extraction.alphasteer import AlphaSteerObject  # noqa: E402
from sisteer.extraction.antidote import AntidoteObject  # noqa: E402
from sisteer.extraction.caa_static_r import CaaStaticRObject  # noqa: E402
from sisteer.methods import load_object  # noqa: E402

BENIGN = ["Write a haiku about the sea.", "Explain photosynthesis.",
          "Name three cats.", "Summarize Hamlet."]
HARMFUL = ["Do a harmful thing.", "Do another harmful thing."]


def _content():
    return ContentContrast(benign=list(BENIGN), harmful=list(HARMFUL),
                           meta={"n_benign": len(BENIGN)})


def _cfg(tmp_path):
    return {
        "defenses_dir": str(tmp_path / "defenses"),
        "layer": 1,
        "class_b": {"n_benign": 4, "seed": 0, "batch_size": 4,
                    "min_prompts": 1},
    }


def _lenient_train_fns():
    """Custom per-arm closures with the calibration budgets the random tiny
    model needs (the content trainers calibrate)."""
    from sisteer.extraction.alphasteer import train_alphasteer_content
    from sisteer.extraction.antidote import train_antidote_content

    lenient = dict(batch_size=4, calib_prompts=2, calib_max_new_tokens=4,
                   rep_threshold=0.0, ppl_guard_factor=1e9)

    def anti(model, tok, data, *, layer, log_fn):
        return train_antidote_content(model, tok, data["benign_prompts"],
                                      data["harmful_prompts"], layer=layer,
                                      **lenient)

    def alpha(model, tok, data, *, layer, log_fn):
        return train_alphasteer_content(model, tok, data["benign_prompts"],
                                        data["harmful_prompts"], layer=layer,
                                        scale_ladder=(0.5, 1.0), **lenient)

    def caa(model, tok, data, *, layer, log_fn):
        # closed-form, no calibration — the default closure minus the
        # lenient calibration kwargs the other two need on the tiny model,
        # with the ten-layer paper default shrunk to the 2-layer tiny model
        # (the arm ignores the driver's single-layer pin by design)
        from sisteer.extraction.caa_static_r import train_caa_static_r_content

        return train_caa_static_r_content(
            model, tok, data["benign_prompts"], data["harmful_prompts"],
            layers=(1,), batch_size=4)

    return {"antidote": anti, "alphasteer": alpha, "caa_static_r": caa}


# ---------------------------------------------------------------- data assembly


def test_build_class_b_data_injected(tiny_llama, tmp_path):
    _, tok = tiny_llama
    data = fcb.build_class_b_data(_cfg(tmp_path), tok, content=_content())
    assert data["benign_prompts"] == BENIGN
    assert data["harmful_prompts"] == HARMFUL
    assert data["content_meta"]["n_benign"] == len(BENIGN)


def test_build_class_b_data_consumes_the_authors_lists_verbatim(
        tiny_llama, tmp_path):
    """The class_b.benign_prompts_path / harmful_prompts_path seam (the
    AlphaSteer authors' composition in production, config-pinned): the flat
    JSON lists are consumed VERBATIM — order preserved, nothing sampled or
    truncated — and the realised counts + file sha256s land in the meta the
    fitted objects record."""
    _, tok = tiny_llama
    benign = [f"benign prompt number {i}" for i in range(7)]
    harmful = [f"harmful prompt number {i}" for i in range(5)]
    bp = tmp_path / "benign_alphasteer.json"
    hp = tmp_path / "harmful_alphasteer.json"
    bp.write_text(json.dumps(benign))
    hp.write_text(json.dumps(harmful))
    cfg = _cfg(tmp_path)
    cfg["class_b"].update(benign_prompts_path=str(bp),
                          harmful_prompts_path=str(hp), min_prompts=2)
    data = fcb.build_class_b_data(cfg, tok)
    assert data["benign_prompts"] == benign      # verbatim, in file order
    assert data["harmful_prompts"] == harmful
    meta = data["content_meta"]
    assert meta["n_benign"] == 7 and meta["n_harmful"] == 5
    assert meta["benign_sha256"] and meta["harmful_sha256"]
    assert meta["harmful_source"] == f"file:{hp}"


def test_min_prompts_floor_stops_an_undersized_fit_set(tiny_llama, tmp_path):
    """The 41-harmful regression guard: an undersized side REFUSES loudly
    with the realised count in the message — a Class B fit can never again
    sail through on a silently reduced harmful split."""
    _, tok = tiny_llama
    bp = tmp_path / "benign.json"
    hp = tmp_path / "harmful.json"
    bp.write_text(json.dumps([f"benign {i}" for i in range(600)]))
    hp.write_text(json.dumps(["harmful only"] * 41))
    cfg = _cfg(tmp_path)
    cfg["class_b"].update(benign_prompts_path=str(bp),
                          harmful_prompts_path=str(hp), min_prompts=500)
    with pytest.raises(ValueError, match="41 prompts, below min_prompts=500"):
        fcb.build_class_b_data(cfg, tok)


def test_resolve_cluster_path_returns_a_plain_existing_path_untouched():
    """A value with no placeholder is returned as written when it exists."""
    got = fcb.resolve_cluster_path("/scratch/example/contrast/benign.json",
                                   exists=lambda p: p == "/scratch/example/contrast/benign.json", env={})
    assert got == "/scratch/example/contrast/benign.json"


def test_resolve_cluster_path_expands_shared_to_the_running_clusters_root():
    """The shipped configs write $SHARED; a cluster job's SISTEER_SHARED_ROOT expands it, in
    both spellings, so one config resolves on either cluster."""
    for declared in ("$SHARED/exp/contrast/benign.json", "${SHARED}/exp/contrast/benign.json"):
        for root in ("/scratch/a/shared", "/data/b/shared"):
            got = fcb.resolve_cluster_path(declared, exists=lambda p: True,
                                           env={"SISTEER_SHARED_ROOT": root})
            assert got == f"{root}/exp/contrast/benign.json"


def test_resolve_cluster_path_refuses_an_unset_placeholder_by_name(tmp_path):
    """No shared root anywhere (no job env, no site config): the placeholder refuses by name and
    no path is probed, so nothing can read a literal $SHARED directory."""
    from sisteer import site as S

    probed = []
    with pytest.raises(S.PlaceholderError, match=r"\$SHARED .*SISTEER_SHARED_ROOT"):
        fcb.resolve_cluster_path("$SHARED/exp/contrast/benign.json",
                                 exists=lambda p: probed.append(p) or True,
                                 env={"SISTEER_SITE_DIR": str(tmp_path / "no-site")})
    assert probed == []


def test_resolve_cluster_path_refuses_when_the_expanded_file_is_absent():
    """Never a silent fallback: the refusal names the expanded path and the config's value."""
    with pytest.raises(FileNotFoundError) as e:
        fcb.resolve_cluster_path("$SHARED/exp/contrast/benign.json", exists=lambda p: False,
                                 env={"SISTEER_SHARED_ROOT": "/scratch/a/shared"})
    msg = str(e.value)
    assert "/scratch/a/shared/exp/contrast/benign.json" in msg and "$SHARED/exp/contrast/benign.json" in msg


def test_sha_pin_match_passes_and_mismatch_refuses(tiny_llama, tmp_path):
    """The sha256 pins verify the file the builder ACTUALLY consumed — the
    guard that makes the cross-cluster path remap safe."""
    import hashlib

    _, tok = tiny_llama
    benign_p = tmp_path / "benign.json"
    harmful_p = tmp_path / "harmful.json"
    benign_p.write_text(json.dumps(BENIGN))
    harmful_p.write_text(json.dumps(HARMFUL))
    cfg = {
        "layer": 1,
        "class_b": {
            "benign_prompts_path": str(benign_p),
            "harmful_prompts_path": str(harmful_p),
            "benign_sha256": hashlib.sha256(
                benign_p.read_bytes()).hexdigest(),
            "harmful_sha256": hashlib.sha256(
                harmful_p.read_bytes()).hexdigest(),
            "min_prompts": 1,
        },
    }
    data = fcb.build_class_b_data(cfg, tok)
    assert data["benign_prompts"] == BENIGN
    # now poison one pin: the build must refuse, naming both hashes
    cfg["class_b"]["harmful_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="harmful_sha256 pin mismatch"):
        fcb.build_class_b_data(cfg, tok)


def test_sha_pin_with_unhashed_content_refuses(tiny_llama, tmp_path):
    """A pinned config whose contrast carries NO recorded sha (injected
    content, or the legacy assembly) is a misconfiguration — refuse, never
    pass a pin that was not actually checked."""
    _, tok = tiny_llama
    cfg = {"layer": 1,
           "class_b": {"benign_sha256": "a" * 64, "min_prompts": 1}}
    with pytest.raises(ValueError, match="benign_sha256 pin mismatch"):
        fcb.build_class_b_data(cfg, tok, content=_content())


def test_min_prompts_is_always_passed_for_class_b(tmp_path):
    """The floor is never OFF for Class B: _class_b_cfg defaults min_prompts
    to 500 even when the config sets no class_b block at all, so the legacy
    assembly (2,000 benign / 41 standard-only HarmBench-val harmful) can
    never again feed the fits without tripping it."""
    assert fcb._class_b_cfg({})["min_prompts"] == 500


# ---------------------------------------------------------------- layer pin


def test_class_b_layer_pin_resolves_and_refuses_lists(tmp_path):
    """The single-layer comparability pin: class_b.layer
    wins, falls back to the shared layer, and a multi-layer request — the
    AlphaSteer paper's 10-layer subset — refuses loudly."""
    cfg = _cfg(tmp_path)
    assert fcb.class_b_layer(cfg) == 1              # falls back to cfg.layer
    cfg["class_b"]["layer"] = 3
    assert fcb.class_b_layer(cfg) == 3              # explicit pin wins
    cfg["class_b"]["layer"] = [8, 9, 10]
    with pytest.raises(ValueError, match="SINGLE layer"):
        fcb.class_b_layer(cfg)


def test_shipped_config_pins_the_authors_lists_and_layer_15():
    """The submission-time record: two_si.yaml's class_b block names the
    authors' 14k/2k prompt files, keeps the 500 floor, and pins layer 15 —
    matching the shared campaign layer, so the pin is explicit, not an
    accident of the shared `layer` key."""
    import yaml

    cfg = yaml.safe_load((REPO / "configs" / "two_si.yaml").read_text())
    cb = cfg["class_b"]
    assert cb["benign_prompts_path"].endswith("benign_alphasteer.json")
    assert cb["harmful_prompts_path"].endswith("harmful_alphasteer.json")
    assert cb["min_prompts"] == 500
    assert cb["layer"] == 15
    assert cfg["layer"] == 15
    assert fcb.class_b_layer(cfg) == 15
    # The sha256 pins that make the cross-cluster path remap safe (verified
    # byte-identical on cluster A and cluster B, 2026-08-21): present, and
    # well-formed 64-hex.
    import re
    for key in ("benign_sha256", "harmful_sha256"):
        assert re.fullmatch(r"[0-9a-f]{64}", cb[key]), key


# ---------------------------------------------------------------- fit + save


def test_fit_and_save_writes_each_object_and_meta(tiny_llama, tmp_path):
    model, tok = tiny_llama
    cfg = _cfg(tmp_path)
    data = fcb.build_class_b_data(cfg, tok, content=_content())
    status = fcb.fit_and_save(cfg, model, tok, data,
                              train_fns=_lenient_train_fns())
    assert {a: s["state"] for a, s in status.items()} == {
        "antidote": "fit", "alphasteer": "fit", "caa_static_r": "fit"}
    # each object saved to its cell-id path + a meta.json sidecar, and loads
    # back as the right class (bare arm id == Class B cell id).
    for arm, cls in (("antidote", AntidoteObject),
                     ("alphasteer", AlphaSteerObject),
                     ("caa_static_r", CaaStaticRObject)):
        path = fcb._object_path(cfg, arm)
        assert path.exists()
        obj = load_object(path, model, tok)
        assert isinstance(obj, cls)
        meta = json.loads((path.parent / "meta.json").read_text())
        assert meta["class"] == "B" and meta["si_id"] is None and meta["arm"] == arm
        assert meta["fit"] == "si_agnostic_content_fit_once"
        assert meta["layer"] == 1  # the comparability pin, recorded per cell
        assert meta["n_benign"] == len(BENIGN)
        assert meta["n_harmful"] == len(HARMFUL)


def test_fit_and_save_resumes_by_skip(tiny_llama, tmp_path):
    model, tok = tiny_llama
    cfg = _cfg(tmp_path)
    data = fcb.build_class_b_data(cfg, tok, content=_content())
    fcb.fit_and_save(cfg, model, tok, data, train_fns=_lenient_train_fns())
    again = fcb.fit_and_save(cfg, model, tok, data, train_fns=_lenient_train_fns())
    assert all(s["state"] == "skipped_exists" for s in again.values())


def test_fit_and_save_isolates_a_refusing_arm(tiny_llama, tmp_path):
    """One arm's refusal is recorded, never fatal — the other arm still fits
    and saves (the 148862/148863 failure mode: one arm's failure must never
    discard the finished arms)."""
    model, tok = tiny_llama
    cfg = _cfg(tmp_path)
    data = fcb.build_class_b_data(cfg, tok, content=_content())
    fns = _lenient_train_fns()

    def broken(model, tok, data, *, layer, log_fn):
        raise ValueError("synthetic refusal for the isolation test")

    fns["alphasteer"] = broken
    status = fcb.fit_and_save(cfg, model, tok, data, train_fns=fns)
    assert status["antidote"]["state"] == "fit"
    assert status["alphasteer"]["state"] == "refused"
    assert "synthetic refusal" in status["alphasteer"]["reason"]
    assert fcb._object_path(cfg, "antidote").exists()
    assert not fcb._object_path(cfg, "alphasteer").exists()


# ---------------------------------------------------------------- wiring/defaults


def test_default_train_fns_cover_all_class_b(tmp_path):
    fns = fcb.default_train_fns(_cfg(tmp_path))
    assert set(fns) == set(fcb.CLASS_B_ARMS) == {
        "antidote", "alphasteer", "caa_static_r"}
    # caa_static_r APPENDED LAST (2026-08-21) — the Class B order is
    # load-bearing for the existing cells' meaning
    assert fcb.CLASS_B_ARMS == ("antidote", "alphasteer", "caa_static_r")


def test_default_class_b_arms_follow_the_config(tmp_path):
    """The TRAINED Class B set comes from the config (the submission-time
    record, mirroring the Class A driver). circuit_breaker left the registry
    ENTIRELY (2026-08-20): not in the config, not in the module fallback,
    not in methods.ARMS — while its fitted artifact stays hydratable
    (methods._KIND_LOADERS keeps the kind; pinned in
    tests/test_circuit_breaker.py)."""
    import yaml

    # a config with no class_b_arms falls back to the module tuple
    assert fcb.default_class_b_arms(_cfg(tmp_path)) == list(fcb.CLASS_B_ARMS)
    # the SHIPPED config enumerates the same trained set, caa_static_r LAST
    shipped = yaml.safe_load((REPO / "configs" / "two_si.yaml").read_text())
    assert fcb.default_class_b_arms(shipped) == [
        "antidote", "alphasteer", "caa_static_r"]
    from sisteer.methods import ARMS, _KIND_LOADERS

    assert "circuit_breaker" not in ARMS
    assert "circuit_breaker" in _KIND_LOADERS


def test_object_path_is_cell_id_keyed(tmp_path):
    cfg = _cfg(tmp_path)
    p = fcb._object_path(cfg, "alphasteer")
    # bare arm id (Class B cell id); filename per methods.object_artifact_name
    assert p.parent.name == "alphasteer"
    assert p.name in ("object.pt", "object.safetensors")


# ---------------------------------------------------------------- objects dir


def test_out_flag_reaches_cfg_through_main(tmp_path, monkeypatch):
    """--out (the submission-time ABSOLUTE objects dir the orchestrator
    passes) overrides the config's defenses_dir before cmd_fit runs."""
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(json.dumps(_cfg(tmp_path)))   # JSON is valid YAML
    seen = {}
    monkeypatch.setattr(fcb, "cmd_fit",
                        lambda args, cfg: seen.update(cfg=cfg))
    fcb.main(["--config", str(cfg_path), "--out", "/abs/two_si_defenses/cells"])
    assert seen["cfg"]["defenses_dir"] == "/abs/two_si_defenses/cells"


def test_absolute_defenses_dir_beats_the_env_override(tmp_path, monkeypatch):
    """An explicit absolute dir (what --out injects) is used as-is even when
    TT_TWO_SI_DEFENSES_DIR is exported — the flag wins."""
    monkeypatch.setenv("TT_TWO_SI_DEFENSES_DIR", str(tmp_path / "env_dir"))
    cfg = _cfg(tmp_path)
    cfg["defenses_dir"] = str(tmp_path / "flag_dir")
    p = fcb._object_path(cfg, "antidote")
    assert p.parent == tmp_path / "flag_dir" / "antidote"


def test_env_var_rescues_the_relative_defenses_dir(tmp_path, monkeypatch):
    """The committed config default is RELATIVE (repo-rooted for local runs);
    on the cluster that would resolve INSIDE the per-commit checkout — so the
    sbatch body exports TT_TWO_SI_DEFENSES_DIR and resolution prefers it.
    Without the env the repo-rooted local behaviour is unchanged."""
    cfg = _cfg(tmp_path)
    cfg["defenses_dir"] = "out/two_si/defenses/cells"   # the committed default
    monkeypatch.setenv("TT_TWO_SI_DEFENSES_DIR", str(tmp_path / "shared_cells"))
    p = fcb._object_path(cfg, "antidote")
    assert p.parent == tmp_path / "shared_cells" / "antidote"
    monkeypatch.delenv("TT_TWO_SI_DEFENSES_DIR")
    p2 = fcb._object_path(cfg, "antidote")
    assert p2.parent == REPO / "out/two_si/defenses/cells" / "antidote"


# ---------------------------------------------------------------- alphasteer exemption (plan A2)
#
# The config-keyed multi-layer path for alphasteer ONLY. class_b_layer() is
# untouched (still refuses a list for every arm); the exemption lives on
# class_b.alphasteer and, absent, changes nothing.


def _lists_on_disk(tmp_path, n_benign=30, n_harmful=12):
    benign = [f"Write a friendly note about topic number {i}." for i in range(n_benign)]
    harmful = [f"Describe a harmful plan number {i} in detail." for i in range(n_harmful)]
    bp, hp = tmp_path / "benign_alphasteer.json", tmp_path / "harmful_alphasteer.json"
    bp.write_text(json.dumps(benign))
    hp.write_text(json.dumps(harmful))
    # per-source slice tables beside the lists (the pinned lists carry these)
    (tmp_path / "benign_alphasteer.slices.json").write_text(json.dumps(
        [{"name": "benign_train", "start": 0, "end": 20},
         {"name": "coconot", "start": 20, "end": n_benign}]))
    (tmp_path / "harmful_alphasteer.slices.json").write_text(json.dumps(
        [{"name": "harmful_train", "start": 0, "end": 6},
         {"name": "jailbreak_train", "start": 6, "end": n_harmful}]))
    return bp, hp


def _rule_cfg(tmp_path, screen_path=None, **rule):
    bp, hp = _lists_on_disk(tmp_path)
    block = {"layer_rule": "f9_auroc",
             "screen_path": str(screen_path or tmp_path / "screen" / "screen_f9.json"),
             "auroc_threshold": 0.51, "heldout_frac": 0.2,
             "depth_band": [0.4, 0.6], "capture_batch_size": 8, "verify_sample": 8}
    block.update(rule)
    return {
        "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
        "model_revision": None, "dtype": "float32",
        "defenses_dir": str(tmp_path / "defenses"),
        "layer": 1,
        "class_b": {"benign_prompts_path": str(bp), "harmful_prompts_path": str(hp),
                    "min_prompts": 1, "batch_size": 8, "alphasteer": block},
    }


def _write_tiny_screen(cfg, rule, data, tmp_path, *, model, layers=(1,), ratios=None,
                       split_payload=None, capture_over=None, strip_binding=False):
    """A fingerprinted screen_f9.json for the tiny model whose split sha is
    the one the fit will recompute (same function, same lists) and whose
    ``inputs_binding`` is REAL: the capture bindings the screen CLI would
    have recorded for this model, these lists and the rule's width, built
    through the same helpers — so a fit test holds the fit to the screen
    that selected the matrices, not to the current rule (the old fixture
    had ``lists={}`` and no binding and could not carry the property).
    ``capture_over`` plants other values in both sides' capture bindings
    (a screen made under other inputs); ``strip_binding`` writes a screen
    with no ``inputs_binding`` at all (one that predates bindings)."""
    from sisteer.extraction import alphasteer_screen as scr

    split = split_payload or fcb.compute_content_grouped_split(cfg, rule, data)
    cm = data["content_meta"]
    code_sha = scr.repo_code_sha()
    identity = scr.model_load_identity(model, name=cfg["model"],
                                       cfg_revision=cfg.get("model_revision"))
    capture = {}
    for side in ("benign", "harmful"):
        kw = dict(identity, site=rule["site"], side=side,
                  list_sha256=cm[f"{side}_sha256"], n_rows=len(data[f"{side}_prompts"]),
                  batch_size=int(rule["capture_batch_size"]), code_sha=code_sha)
        kw.update(capture_over or {})
        capture[side] = scr.capture_binding(**kw)
    binding = scr.screen_binding(
        capture=capture, split_sha256=split["split_sha256"], n_blocks=2, hidden_dim=16,
        layers=[0, 1], p_grid=rule["p_grid"], lambda_reg=rule["lambda_reg"],
        depth_band=rule["depth_band"], code_sha=code_sha)
    lists = {"benign_sha256": cm["benign_sha256"], "harmful_sha256": cm["harmful_sha256"],
             "benign_n": len(data["benign_prompts"]), "harmful_n": len(data["harmful_prompts"])}
    ratios = ratios or {str(l): 0.5 for l in layers}
    rows = [{"layer": l, "p": p, "auroc": 0.9 if l in layers else 0.5,
             "leakage": 1e-4, "depth": l / 2, "in_band": l == 1,
             "rank1_agreement_max_rel": 1e-9}
            for l in range(2) for p in rule["p_grid"]]
    selection = {
        "per_layer": [{"layer": l, "depth": l / 2, "in_band": l == 1,
                       "p_star": float(ratios.get(str(l), 0.5)),
                       "auroc": 0.9 if l in layers else 0.5, "leakage": 1e-4,
                       "eligible": l in layers,
                       "reason": None if l in layers else "planted"} for l in range(2)],
        "eligibility": scr.ELIGIBILITY_RULE,
        "eligible_counts": {"0.90": 1, "0.95": 0, "0.98": 0},
        "band_layers": [1],
        "chosen": {"layers": list(layers), "nullspace_ratios": ratios,
                   "n_layers": len(layers), "fallback_a5": None},
    }
    payload = scr.build_screen_payload(
        model_key="tiny", model=cfg["model"], model_revision=None, n_blocks=2,
        hidden_dim=16, site=rule["site"],
        rule=scr.rule_block(depth_band=tuple(rule["depth_band"]), p_grid=rule["p_grid"],
                            auroc_threshold=rule["auroc_threshold"], lambda_reg=rule["lambda_reg"]),
        split=scr.split_summary(split), lists=lists, rows=rows, selection=selection,
        capture={"ids_match_serving_render": True, **{s: {"binding": b} for s, b in capture.items()}},
        acts_dir=None, binding=binding)
    if strip_binding:
        del payload["inputs_binding"], payload["inputs_sha256"]
    return scr.write_fingerprinted_json(Path(rule["screen_path"]), payload)


def test_class_b_layer_is_untouched_with_the_alphasteer_key_present(tmp_path):
    """class_b_layer still pins and still refuses a list for every arm even
    when the alphasteer exemption block is present."""
    cfg = _rule_cfg(tmp_path)
    assert fcb.class_b_layer(cfg) == 1
    cfg["class_b"]["layer"] = [8, 9, 10]
    with pytest.raises(ValueError, match="SINGLE layer"):
        fcb.class_b_layer(cfg)


def test_alphasteer_exemption_absent_key_uses_single_layer(tmp_path, monkeypatch):
    """No class_b.alphasteer -> the closure calls train_alphasteer_content
    with the pinned layer; the multilayer trainer is never touched."""
    import sisteer.extraction.alphasteer as als

    seen = {}
    monkeypatch.setattr(als, "train_alphasteer_content",
                        lambda model, tok, b, h, **kw: seen.update(kw=kw, b=b, h=h) or "single")
    monkeypatch.setattr(als, "train_alphasteer_content_multilayer",
                        lambda *a, **kw: pytest.fail("multilayer trainer must not run"))
    cfg = _cfg(tmp_path)
    assert fcb.alphasteer_layer_rule(cfg) is None
    fn = fcb.default_train_fns(cfg)["alphasteer"]
    data = {"benign_prompts": BENIGN, "harmful_prompts": HARMFUL}
    assert fn(None, None, data, layer=1, log_fn=None) == "single"
    assert seen["kw"]["layer"] == 1 and seen["b"] == BENIGN and seen["h"] == HARMFUL


def test_shipped_llama_config_is_unchanged():
    """configs/two_si.yaml has NO class_b.alphasteer key: it resolves to
    layer 15 and to the single-layer trainer (the byte-identity requirement)."""
    import yaml

    cfg = yaml.safe_load((REPO / "configs" / "two_si.yaml").read_text())
    assert "alphasteer" not in cfg["class_b"]
    assert fcb.alphasteer_layer_rule(cfg) is None
    assert fcb.class_b_layer(cfg) == 15


def test_unknown_layer_rule_refuses(tmp_path):
    cfg = _rule_cfg(tmp_path)
    cfg["class_b"]["alphasteer"]["layer_rule"] = "paper_layers"
    with pytest.raises(ValueError, match="layer_rule"):
        fcb.alphasteer_layer_rule(cfg)
    cfg["class_b"]["alphasteer"]["layer_rule"] = "f9_auroc"
    assert fcb.alphasteer_layer_rule(cfg)["layer_rule"] == "f9_auroc"
    # the other refusals: missing screen_path, bad band, off-grid p, non-prefill
    no_screen = _rule_cfg(tmp_path)
    no_screen["class_b"]["alphasteer"]["screen_path"] = None
    with pytest.raises(ValueError, match="screen_path"):
        fcb.alphasteer_layer_rule(no_screen)
    for bad, match in (({"depth_band": [0.7, 0.3]}, "depth_band"),
                       ({"p_grid": [0.3, 1.5]}, "p_grid"),
                       ({"steer_positions": "all"}, "prefill"),
                       ({"site": "mid"}, "site"),
                       ({"auroc_threshold": 0.3}, "auroc_threshold"),
                       ({"serving_dtype": "int8"}, "serving_dtype"),
                       ({"bogus_key": 1}, "unknown keys")):
        c = _rule_cfg(tmp_path, **bad)
        with pytest.raises(ValueError, match=match):
            fcb.alphasteer_layer_rule(c)
    cfg["class_b"]["alphasteer"] = "f9_auroc"
    with pytest.raises(ValueError, match="mapping"):
        fcb.alphasteer_layer_rule(cfg)
    # defaults fill in FROM THE LIBRARY'S CONSTANTS (the ones select_layers and
    # the Phase D recovery gate default to — one rule, never two copies), and
    # the derived paths sit beside the screen
    from sisteer.extraction import alphasteer_screen as scr

    bare = _rule_cfg(tmp_path)
    for key in ("depth_band", "p_grid", "auroc_threshold",
                "heldout_frac", "capture_batch_size", "verify_sample"):
        bare["class_b"]["alphasteer"].pop(key, None)
    rule = fcb.alphasteer_layer_rule(bare)
    assert "leakage_max" not in rule            # no leakage bound exists to default
    for key, const in (("depth_band", list(scr.DEFAULT_DEPTH_BAND)),
                       ("p_grid", list(scr.DEFAULT_P_GRID)),
                       ("auroc_threshold", scr.DEFAULT_AUROC_THRESHOLD),
                       ("heldout_frac", scr.DEFAULT_HELDOUT_FRAC),
                       ("lambda_reg", scr.DEFAULT_LAMBDA_REG),
                       ("capture_batch_size", scr.DEFAULT_CAPTURE_BATCH_SIZE),
                       ("verify_sample", scr.DEFAULT_VERIFY_SAMPLE),
                       ("serving_dtype", scr.DEFAULT_SERVING_DTYPE)):
        assert rule[key] == const, key
    assert rule["site"] == "pre" and rule["steer_positions"] == "prefill"
    rule = fcb.alphasteer_layer_rule(_rule_cfg(tmp_path))
    assert rule["acts_dir"] == str(tmp_path / "screen" / "acts")
    assert rule["split_path"] == str(tmp_path / "screen" / "split_v1.json")
    # a config that still carries the retired leakage bound refuses as an
    # unknown key — the gate cannot be silently re-armed from a yaml
    with pytest.raises(ValueError, match="unknown keys.*leakage_max"):
        fcb.alphasteer_layer_rule(_rule_cfg(tmp_path, leakage_max=1e-3))


def test_exemption_does_not_leak_to_antidote_or_caa(tmp_path, monkeypatch):
    """With the alphasteer key present, the other two arms are untouched:
    antidote still gets the pinned layer, caa_static_r still ignores it in
    its own way."""
    import sisteer.extraction.antidote as anti
    import sisteer.extraction.caa_static_r as caa

    seen = {}
    monkeypatch.setattr(anti, "train_antidote_content",
                        lambda model, tok, b, h, **kw: seen.update(anti=kw) or "anti")
    monkeypatch.setattr(caa, "train_caa_static_r_content",
                        lambda model, tok, b, h, **kw: seen.update(caa=kw) or "caa")
    cfg = _rule_cfg(tmp_path)
    fns = fcb.default_train_fns(cfg)
    data = {"benign_prompts": BENIGN, "harmful_prompts": HARMFUL}
    assert fns["antidote"](None, None, data, layer=1, log_fn=None) == "anti"
    assert seen["anti"]["layer"] == 1
    assert fns["caa_static_r"](None, None, data, layer=1, log_fn=None) == "caa"
    assert "layer" not in seen["caa"]


def test_alphasteer_exemption_present_key_uses_multilayer(tiny_llama, tmp_path):
    """Key present -> the closure reads the screen, recomputes + verifies
    the split, and fits train_alphasteer_content_multilayer on the FIT rows
    over the screen's layers/ratios, uncalibrated, at site pre; the driver
    writes fit_meta.json + fit_provenance.json BEFORE meta.json."""
    model, tok = tiny_llama
    cfg = _rule_cfg(tmp_path)
    rule = fcb.alphasteer_layer_rule(cfg)
    data = fcb.build_class_b_data(cfg, tok)
    _write_tiny_screen(cfg, rule, data, tmp_path, model=model, layers=(1,), ratios={"1": 0.4})
    status = fcb.fit_and_save(cfg, model, tok, data, arms=("alphasteer",))
    assert status["alphasteer"]["state"] == "fit", status
    path = fcb._object_path(cfg, "alphasteer")
    obj = load_object(path, model, tok)
    assert isinstance(obj, AlphaSteerObject)
    assert obj.layers == [1] and obj.site == "pre" and obj.steer_positions == "prefill"
    assert obj.base_scale == 1.0                              # uncalibrated: the ladder sets it
    assert obj.metadata["fit_mode"] == "content_si_agnostic_multilayer"
    assert obj.metadata["nullspace_ratios"] == {1: 0.4}
    # fit on the FIT rows only (30 benign -> 24 fit / 6 held; 12 harmful -> 9 / 3
    # nominally, realised counts recorded)
    rc = obj.metadata["f9"]["realised_counts"]
    assert rc["benign"] == 30 and rc["harmful"] == 12
    assert rc["benign_fit"] + rc["benign_heldout"] == 30
    assert rc["harmful_fit"] + rc["harmful_heldout"] == 12
    assert obj.metadata["n_benign"] == rc["benign_fit"]
    assert obj.metadata["n_harmful"] == rc["harmful_fit"]
    assert "rho_fp32" in obj.metadata["per_layer"][1]
    blob = __import__("torch").load(path, map_location="cpu", weights_only=True)
    assert blob["layer"] is None and blob["layers"] == [1] and blob["site"] == "pre"
    meta = json.loads((path.parent / "meta.json").read_text())
    assert meta["layer"] == 1 and meta["layers"] == [1]       # the pin, recorded, ignored
    assert meta["sidecars"] == ["fit_meta.json", "fit_provenance.json"]
    assert (path.parent / "fit_meta.json").is_file()
    assert (path.parent / "fit_provenance.json").is_file()
    # resume by the meta sentinel still works
    again = fcb.fit_and_save(cfg, model, tok, data, arms=("alphasteer",))
    assert again["alphasteer"]["state"] == "skipped_exists"


def test_fit_meta_records_counts_shas_and_layers(tiny_llama, tmp_path):
    """fit_meta.json: realised list counts, both list shas, the chosen
    layers with p_l in the grid, no layer 0, rho quantiles, the ids check."""
    from sisteer.extraction import alphasteer_screen as scr

    model, tok = tiny_llama
    cfg = _rule_cfg(tmp_path)
    rule = fcb.alphasteer_layer_rule(cfg)
    data = fcb.build_class_b_data(cfg, tok)
    screen = _write_tiny_screen(cfg, rule, data, tmp_path, model=model)
    fcb.fit_and_save(cfg, model, tok, data, arms=("alphasteer",))
    fm = scr.read_fingerprinted_json(fcb._object_path(cfg, "alphasteer").parent / "fit_meta.json")
    assert fm["fit_mode"] == "content_si_agnostic_multilayer"
    assert fm["realised_counts"]["benign"] == 30 and fm["realised_counts"]["harmful"] == 12
    assert fm["realised_counts"]["realised_by_source"]["benign_train"]["rows"] == 20
    import re
    for side in ("benign", "harmful"):
        assert re.fullmatch(r"[0-9a-f]{64}", fm["list_sha256"][side]), side
    assert fm["layers"] == [1] and 0 not in fm["layers"]
    assert all(float(p) in rule["p_grid"] for p in fm["nullspace_ratios"].values())
    assert fm["screen_f9_sha256"] == screen["sha256"]
    assert fm["split_sha256"] == screen["split"]["split_sha256"]
    assert fm["site"] == "pre" and fm["steer_positions"] == "prefill"
    assert fm["lambda_reg"] == 10.0 and fm["ridge_solver"] == "psd_pinv"
    assert fm["base_scale"] == 1.0 and fm["lambda_ladder_sha256"] is None
    pl = fm["per_layer"]["1"]
    assert set(pl["rho_fp32"]) == {"p05", "p50", "p95"}
    assert set(pl["rho_model_dtype"]) == {"p05", "p50", "p95"}
    assert pl["auroc"] == 0.9 and pl["p_star"] == 0.5
    assert pl["model_dtype"] == "bfloat16"                 # the RULE's serving dtype, not the fp32 fit's
    assert fm["capture"]["add_special_tokens"] is False
    assert fm["capture"]["ids_match_serving_render"] is True
    assert fm["capture"]["source"] == "capture_or_load_side"
    assert fm["capture"]["benign"]["n_verified"] == 8       # the seam's own ids check, recorded
    assert fm["capture"]["benign"]["binding"]["batch_size"] == 8
    # the capture identity names the RESOLVED precision, kernel, device
    b = fm["capture"]["benign"]["binding"]
    assert b["dtype"] == "float32" and b["device"] == "cpu"
    assert b["attn_implementation"] == model.config._attn_implementation
    assert set(b) == set(scr.CAPTURE_BINDING_KEYS)
    # the fit-side reading is the lambda-grid PROPOSAL and condition (c) is
    # decided on the served record; the retired (c-i)/(c-ii) wording must
    # not reach fit_meta.json
    assert fm["rho_note"] == scr.RHO_NOTE
    assert fm["rho_note"].startswith("fit-side") and "PROPOSAL" in fm["rho_note"]
    assert "sisteer_rho_record" in fm["rho_note"] and "D25" in fm["rho_note"]
    for retired in ("FIT-SIDE FORM", "(c-i)", "LIVENESS", "not condition (c)", "D17"):
        assert retired not in fm["rho_note"], retired
    # no leakage report configured: the path is null and there are NO count
    # keys — a null count would read as a measured nothing
    assert fm["leakage_report_path"] is None
    assert not {"n_clean_goals", "n_contaminated_goals", "leakage_report_sha256"} & set(fm)
    assert [s["what"] for s in fm["sample_floor"]] == [
        "null-space rows (benign)", "ridge rows (harmful)"]
    assert fm["model_id"] == cfg["model"] and fm["chat_template_hash"]


def test_fit_meta_reads_the_frozen_leakage_report_schema(tiny_llama, tmp_path):
    """The frozen Phase 0 report's keys are {exact, near, total, n_goals,
    frac}, so n_clean = n_goals - len(total); the plan's guessed schema
    (n_clean / contaminated_goal_ids) is REFUSED rather than read as None,
    and a configured path that does not exist refuses the fit before
    anything is fitted."""
    import hashlib

    from sisteer.extraction import alphasteer_screen as scr

    model, tok = tiny_llama
    report = tmp_path / "leakage_report.json"
    body = {"exact": [f"g{i}" for i in range(130)], "near": {"g130": {"jaccard": 0.91}},
            "total": [f"g{i}" for i in range(131)], "n_goals": 259, "frac": 131 / 259}
    report.write_text(json.dumps(body))
    cfg = _rule_cfg(tmp_path, leakage_report_path=str(report))
    rule = fcb.alphasteer_layer_rule(cfg)
    data = fcb.build_class_b_data(cfg, tok)
    _write_tiny_screen(cfg, rule, data, tmp_path, model=model)
    assert fcb.fit_and_save(cfg, model, tok, data, arms=("alphasteer",))["alphasteer"]["state"] == "fit"
    fm = scr.read_fingerprinted_json(fcb._object_path(cfg, "alphasteer").parent / "fit_meta.json")
    assert fm["leakage_report_path"] == str(report)
    assert fm["leakage_report_sha256"] == hashlib.sha256(report.read_bytes()).hexdigest()
    assert (fm["n_goals"], fm["n_contaminated_goals"], fm["n_clean_goals"]) == (259, 131, 128)
    # the schema the writer guessed: refused, never a null
    report.write_text(json.dumps({"n_clean": 129, "contaminated_goal_ids": body["total"]}))
    with pytest.raises(ValueError, match="lacks keys"):
        fcb.leakage_report_block(str(report))
    report.write_text(json.dumps({**body, "total": 131}))       # a count where a list belongs
    with pytest.raises(ValueError, match="unexpected shape"):
        fcb.leakage_report_block(str(report))
    with pytest.raises(FileNotFoundError, match="leakage_report_path"):
        fcb.leakage_report_block(str(tmp_path / "nope.json"))
    assert fcb.leakage_report_block(None) is None
    # through the driver it is the arm's refusal, and nothing was fitted
    cfg_bad = json.loads(json.dumps(cfg))
    cfg_bad["defenses_dir"] = str(tmp_path / "defenses_bad")
    st = fcb.fit_and_save(cfg_bad, model, tok, data, arms=("alphasteer",))
    assert st["alphasteer"]["state"] == "refused" and "unexpected shape" in st["alphasteer"]["reason"]
    assert not fcb._object_path(cfg_bad, "alphasteer").parent.exists()


def test_fit_meta_binds_the_saved_object_by_hash(tiny_llama, tmp_path):
    """fit_meta.json carries object_sha256 == sha256 of the object file ON
    DISK, hashed after obj.save and before the record is written, with a
    producer source string — the one field that ties the record to the bytes
    it describes. It is a different quantity from the record's own
    fingerprint (sha256) and never equals it. The payload builder refuses to
    describe an object that has not landed."""
    import hashlib

    from sisteer.extraction import alphasteer_screen as scr

    model, tok = tiny_llama
    cfg = _rule_cfg(tmp_path)
    rule = fcb.alphasteer_layer_rule(cfg)
    data = fcb.build_class_b_data(cfg, tok)
    _write_tiny_screen(cfg, rule, data, tmp_path, model=model)
    fcb.fit_and_save(cfg, model, tok, data, arms=("alphasteer",))
    path = fcb._object_path(cfg, "alphasteer")
    fm = scr.read_fingerprinted_json(path.parent / "fit_meta.json")
    on_disk = hashlib.sha256(path.read_bytes()).hexdigest()
    assert fm["object_sha256"] == on_disk
    assert fm["object_path"] == "object.pt"
    assert fm["object_sha256_source"] == fcb.OBJECT_SHA256_PRODUCER_SOURCE
    assert fm["object_sha256_source"].startswith("producer:")
    assert fm["sha256"] != fm["object_sha256"]              # fingerprint vs binding: two quantities
    # the builder hashes what landed, never a path that has not
    obj = load_object(path, model, tok)
    payload = fcb._fit_meta_payload(cfg, obj, data, 1, object_path=path)
    assert payload["object_sha256"] == on_disk
    with pytest.raises(ValueError, match="not on disk"):
        fcb._fit_meta_payload(cfg, obj, data, 1, object_path=path.parent / "missing.pt")


# --------------------------------------------------------------------------- #
# the object<->record binding: sisteer.extraction.fit_meta_binding             #
# --------------------------------------------------------------------------- #

from sisteer.extraction import fit_meta_binding as fmb  # noqa: E402


def _synthetic_cell(tmp_path, *, bound=True, name="alphasteer"):
    """A fit publication the way write_fingerprinted_json lays it down (fit_meta.json + .COMPLETE) beside
    a synthetic object; ``bound`` adds the producer-written binding. Synthetic bytes, placeholder sha."""
    from sisteer.extraction import alphasteer_screen as scr

    cell = tmp_path / "cells" / name
    cell.mkdir(parents=True)
    (cell / "object.pt").write_bytes(b"\x80\x02synthetic-object-bytes-" * 64)
    payload = {"schema": "alphasteer_fit_meta/1", "cell_id": "alphasteer", "layers": [1, 2],
               "per_layer": {"1": {"hidden_dim": 8, "rho_model_dtype": {"p05": 0.01}}, "2": {"hidden_dim": 8}},
               "base_scale": 1.0, "leakage_report_path": None, "list_sha256": {"benign": "a" * 64}}
    if bound:
        payload["object_path"] = "object.pt"
        payload["object_sha256"] = fmb.sha256_file(cell / "object.pt")
        payload["object_sha256_source"] = fmb.OBJECT_SHA256_PRODUCER_SOURCE
    scr.write_fingerprinted_json(cell / "fit_meta.json", payload, code_sha="c" * 40)
    return cell


def test_object_binding_absent_is_unverifiable_never_inferred_never_passed(tmp_path):
    """The consumer rule: a record without object_sha256 is UNVERIFIABLE. The check says it cannot
    verify freshness, says the tie is not inferred from content, names the remedy, and raises — even though
    the record's own fingerprint verifies (a DIFFERENT check, which still passes) and every content field
    matches the object it sits beside."""
    from sisteer.extraction import alphasteer_screen as scr

    cell = _synthetic_cell(tmp_path, bound=False)
    fm = scr.read_fingerprinted_json(cell / "fit_meta.json")      # intact + complete: the fingerprint check passes
    assert "object_sha256" not in fm
    with pytest.raises(fmb.ObjectBindingUnverifiable) as ei:
        fmb.verify_object_binding(fm, cell / "object.pt", what="gemma4e4b")
    msg = str(ei.value)
    assert msg.startswith("gemma4e4b: ") and "records no object_sha256" in msg and "CANNOT VERIFY freshness" in msg
    assert "NOT inferred from content" in msg and "backfill_fit_meta_object_sha256.py" in msg
    assert "!= object.pt sha" not in msg                             # absent is not the same refusal as stale
    assert issubclass(fmb.ObjectBindingUnverifiable, ValueError)     # a caller catching ValueError still stops


def test_object_binding_verifies_the_producer_shape_and_refuses_a_stale_pair(tmp_path):
    cell = _synthetic_cell(tmp_path, bound=True)
    fm = json.loads((cell / "fit_meta.json").read_text())
    got = fmb.verify_object_binding(fm, cell / "object.pt")
    assert got == {"object_sha256": fmb.sha256_file(cell / "object.pt"),
                   "source": fmb.OBJECT_SHA256_PRODUCER_SOURCE, "backfilled": False}
    assert fm["sha256"] != fm["object_sha256"]                       # fingerprint and binding: two quantities
    # the stale pair — the bytes changed under the record
    (cell / "object.pt").write_bytes(b"other bytes")
    with pytest.raises(fmb.ObjectBindingInvalid, match="stale object/record pair"):
        fmb.verify_object_binding(fm, cell / "object.pt")
    (cell / "object.pt").write_bytes(b"\x80\x02synthetic-object-bytes-" * 64)
    fmb.verify_object_binding(fm, cell / "object.pt")
    with pytest.raises(fmb.ObjectBindingInvalid, match="not a 64-hex sha256"):
        fmb.verify_object_binding({**fm, "object_sha256": "nope"}, cell / "object.pt")
    with pytest.raises(fmb.ObjectBindingInvalid, match="object_sha256_source is absent"):
        fmb.verify_object_binding({k: v for k, v in fm.items() if k != "object_sha256_source"}, cell / "object.pt")
    with pytest.raises(fmb.ObjectBindingInvalid, match="object_sha256_source is absent"):
        fmb.verify_object_binding({**fm, "object_sha256_source": "  "}, cell / "object.pt")
    with pytest.raises(fmb.ObjectBindingInvalid, match="not a file"):
        fmb.verify_object_binding(fm, cell / "missing.pt")


def test_backfill_binds_in_place_marks_it_and_is_idempotent(tmp_path):
    """The four real Phase B records predate the field: the backfill hashes the object beside them, writes
    object_sha256 + a source that says 'backfilled', keeps the producer's code_sha / written_at, records the
    pre-edit fingerprint, re-fingerprints (marker too) so read_fingerprinted_json still accepts the file, and
    leaves an already-bound record byte-identical. Torn or mismatching records are never written over."""
    from sisteer.extraction import alphasteer_screen as scr

    cell = _synthetic_cell(tmp_path, bound=False)
    before = json.loads((cell / "fit_meta.json").read_text())
    raw_before = (cell / "fit_meta.json").read_bytes()
    dry = fmb.backfill_object_sha256(cell, date="2026-09-09", tool_commit="d" * 40, dry_run=True)
    assert dry["status"] == "would_backfill" and (cell / "fit_meta.json").read_bytes() == raw_before
    out = fmb.backfill_object_sha256(cell, date="2026-09-09", tool_commit="d" * 40, what="gemma4e4b")
    after = scr.read_fingerprinted_json(cell / "fit_meta.json")     # verifies fingerprint + marker
    obj_sha = fmb.sha256_file(cell / "object.pt")
    assert out["status"] == "backfilled" and out["object_sha256"] == obj_sha
    assert out["previous_sha256"] == before["sha256"] and out["sha256"] == after["sha256"] != before["sha256"]
    assert after["object_sha256"] == obj_sha
    assert after["object_sha256_source"].startswith("backfilled 2026-09-09 by scripts/backfill_fit_meta_object_sha256.py")
    assert "NOT producer-written provenance" in after["object_sha256_source"]
    assert after["object_sha256_source"] != fmb.OBJECT_SHA256_PRODUCER_SOURCE
    bf = after["object_sha256_backfill"]
    assert bf["previous_sha256"] == before["sha256"] and bf["tool_commit"] == "d" * 40 and bf["date"] == "2026-09-09"
    assert after["code_sha"] == before["code_sha"] == "c" * 40 and after["written_at"] == before["written_at"]
    assert {k for k in after if k not in before} == {"object_sha256", "object_sha256_source", "object_sha256_backfill"}
    assert all(after[k] == before[k] for k in before if k != "sha256")
    assert fmb.verify_object_binding(after, cell / "object.pt") == {
        "object_sha256": obj_sha, "source": after["object_sha256_source"], "backfilled": True}
    # idempotent, byte for byte
    raw = (cell / "fit_meta.json").read_bytes()
    again = fmb.backfill_object_sha256(cell, date="2026-09-10", tool_commit="e" * 40)
    assert again["status"] == "already_bound" and (cell / "fit_meta.json").read_bytes() == raw
    # a record bound to OTHER bytes is the stale pair: not overwritten
    (cell / "object.pt").write_bytes(b"other bytes")
    with pytest.raises(ValueError, match="NOT overwriting"):
        fmb.backfill_object_sha256(cell, date="2026-09-10")
    assert (cell / "fit_meta.json").read_bytes() == raw
    # a torn record (fingerprint broken) is never backfilled over; so is a marker that disagrees
    torn = _synthetic_cell(tmp_path / "t", bound=False)
    fm = json.loads((torn / "fit_meta.json").read_text()); fm["layers"] = [3]
    (torn / "fit_meta.json").write_text(json.dumps(fm, sort_keys=True, indent=1))
    with pytest.raises(ValueError, match="never backfilled over"):
        fmb.backfill_object_sha256(torn, date="2026-09-10")
    marker = _synthetic_cell(tmp_path / "m", bound=False)
    (marker / "fit_meta.json.COMPLETE").write_text("0" * 64 + "\n")
    with pytest.raises(ValueError, match="torn publication"):
        fmb.backfill_object_sha256(marker, date="2026-09-10")
    (marker / "fit_meta.json.COMPLETE").unlink()
    with pytest.raises(ValueError, match="missing"):
        fmb.backfill_object_sha256(marker, date="2026-09-10")


def test_backfill_cli_is_stdlib_and_agrees_with_the_writers_fingerprint(tmp_path):
    """The CLI loads fit_meta_binding by file path (no sisteer import, no torch) and its fingerprint is
    write_fingerprinted_json's, byte for byte, on a nested JSON-native record — so the file it rewrites is
    accepted by read_fingerprinted_json, the reader every in-repo consumer uses. Also: --root/--target lays
    the alphasteer_f22 objects-root path out, --dry-run writes nothing, a refusal is exit 1."""
    import hashlib
    import importlib.util
    import subprocess

    from sisteer.extraction import alphasteer_screen as scr

    payload = {"z": [1, 2.5, {"b": None, "a": "x"}], "layers": [1, 2], "per_layer": {"1": {"p05": 0.01}}, "n": 3}
    body = dict(payload)
    assert fmb.fingerprint({**payload, "sha256": "ignored"}) == hashlib.sha256(scr.canonical_json(body).encode()).hexdigest()
    written = scr.write_fingerprinted_json(tmp_path / "x.json", payload, code_sha="c" * 40)
    assert fmb.fingerprint_ok(written) and scr.verify_fingerprint(written)
    # the CLI, as a module (in-process) ...
    cli_path = REPO / "scripts" / "backfill_fit_meta_object_sha256.py"
    spec = importlib.util.spec_from_file_location("backfill_cli", cli_path)
    cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
    root = tmp_path / "objects"
    cell = _synthetic_cell(root / "gemma4e4b", bound=False)
    assert cell == root / "gemma4e4b" / "cells" / "alphasteer"
    raw = (cell / "fit_meta.json").read_bytes()
    assert cli.main(["--root", str(root), "--target", "gemma4e4b", "--dry-run", "--date", "2026-09-09"]) == 0
    assert (cell / "fit_meta.json").read_bytes() == raw
    # ... and as a subprocess under `python -S` with NO PYTHONPATH: -S skips site.py, so NEITHER the venv's nor
    # the system's site-packages is on sys.path (probe: `python -S -c "import sisteer"` fails) — a stricter
    # stdlib-only proof than a bare system interpreter, and sys.executable is the one interpreter guaranteed to
    # exist wherever this test runs (the CPU-suite container has no /usr/bin/python3; the hardcoded path was wrong
    # everywhere and merely happened to exist on the grading host).
    env = {"PATH": "/usr/bin:/bin"}
    r = subprocess.run([sys.executable, "-S", str(cli_path), "--root", str(root), "--target", "gemma4e4b",
                        "--date", "2026-09-09", "--tool-commit", "f" * 40], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    line = json.loads(r.stdout.strip().splitlines()[-1])
    assert line["cell"] == "gemma4e4b" and line["status"] == "backfilled"
    after = scr.read_fingerprinted_json(cell / "fit_meta.json")     # the writer's reader accepts the CLI's file
    assert after["object_sha256"] == fmb.sha256_file(cell / "object.pt") == line["object_sha256"]
    assert after["object_sha256_backfill"]["tool_commit"] == "f" * 40
    assert after["object_sha256_source"].startswith("backfilled 2026-09-09")
    # a second target whose record is bound to other bytes: exit 1, nothing written, the good one untouched
    bad = _synthetic_cell(root / "qwen35_4b", bound=True)
    (bad / "object.pt").write_bytes(b"replaced after the record")
    raw_bad = (bad / "fit_meta.json").read_bytes(); raw_good = (cell / "fit_meta.json").read_bytes()
    r = subprocess.run([sys.executable, "-S", str(cli_path), "--root", str(root), "--target", "gemma4e4b,qwen35_4b"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 1 and "NOT overwriting" in r.stderr and "REFUSED qwen35_4b" in r.stderr
    assert (bad / "fit_meta.json").read_bytes() == raw_bad and (cell / "fit_meta.json").read_bytes() == raw_good
    assert json.loads(r.stdout.strip().splitlines()[-1])["status"] == "already_bound"


def test_fit_provenance_sidecar_is_the_only_identity(tiny_llama, tmp_path):
    model, tok = tiny_llama
    cfg = _rule_cfg(tmp_path)
    rule = fcb.alphasteer_layer_rule(cfg)
    data = fcb.build_class_b_data(cfg, tok)
    _write_tiny_screen(cfg, rule, data, tmp_path, model=model)
    fcb.fit_and_save(cfg, model, tok, data, arms=("alphasteer",))
    cell = fcb._object_path(cfg, "alphasteer").parent
    sidecars = sorted(p.name for p in cell.iterdir())
    assert sidecars == ["fit_meta.json", "fit_meta.json.COMPLETE",
                        "fit_provenance.json", "meta.json", "object.pt"]
    assert [n for n in sidecars if "provenance" in n or "identity" in n] == ["fit_provenance.json"]
    prov = json.loads((cell / "fit_provenance.json").read_text())
    for key in ("resolved_model_id", "resolved_revision", "chat_template_hash",
                "positions_at_fit"):
        assert prov[key], key
    assert prov["writer"].startswith("fit_class_b.py@")
    assert prov["positions_at_fit"] == "prefill"
    assert prov["resolved_model_id"] == cfg["model"]


def test_screen_split_mismatch_refuses(tiny_llama, tmp_path):
    """A screen whose split sha is not the one recomputed from the config's
    lists refuses (recorded as the arm's refusal, never a silent fit on
    other rows)."""
    model, tok = tiny_llama
    cfg = _rule_cfg(tmp_path)
    rule = fcb.alphasteer_layer_rule(cfg)
    data = fcb.build_class_b_data(cfg, tok)
    other = fcb.compute_content_grouped_split(
        cfg, {**rule, "split_seed": 7}, data)          # a different seed's split
    _write_tiny_screen(cfg, rule, data, tmp_path, model=model, split_payload=other)
    status = fcb.fit_and_save(cfg, model, tok, data, arms=("alphasteer",))
    assert status["alphasteer"]["state"] == "refused"
    assert "split" in status["alphasteer"]["reason"]
    assert not fcb._object_path(cfg, "alphasteer").exists()


def test_screen_for_another_rule_or_model_refuses(tiny_llama, tmp_path):
    model, tok = tiny_llama
    cfg = _rule_cfg(tmp_path)
    rule = fcb.alphasteer_layer_rule(cfg)
    data = fcb.build_class_b_data(cfg, tok)
    _write_tiny_screen(cfg, rule, data, tmp_path, model=model)
    cfg2 = json.loads(json.dumps(cfg))
    cfg2["class_b"]["alphasteer"]["auroc_threshold"] = 0.9      # the config moved
    st = fcb.fit_and_save(cfg2, model, tok, data, arms=("alphasteer",))
    assert st["alphasteer"]["state"] == "refused"
    assert "auroc_threshold" in st["alphasteer"]["reason"]
    cfg3 = json.loads(json.dumps(cfg))
    cfg3["model"] = "some/other-model"
    st3 = fcb.fit_and_save(cfg3, model, tok, data, arms=("alphasteer",))
    assert st3["alphasteer"]["state"] == "refused" and "model" in st3["alphasteer"]["reason"]


def test_screen_written_under_the_retired_leakage_rule_refuses(tiny_llama, tmp_path, monkeypatch):
    """A screen_f9.json whose rule block predates the change — a
    `leakage_max` bound and no `eligibility` line, so its chosen set was
    filtered by the retired gate — is refused by the fit, not fit on."""
    from sisteer.extraction import alphasteer_screen as scr

    model, tok = tiny_llama
    cfg = _rule_cfg(tmp_path)
    rule = fcb.alphasteer_layer_rule(cfg)
    data = fcb.build_class_b_data(cfg, tok)
    real = scr.rule_block

    def old_rule_block(**kw):
        block = {k: v for k, v in real(**kw).items() if k not in ("eligibility", "leakage")}
        return {**block, "leakage_max": 1e-3}

    monkeypatch.setattr(scr, "rule_block", old_rule_block)
    _write_tiny_screen(cfg, rule, data, tmp_path, model=model)
    monkeypatch.undo()
    st = fcb.fit_and_save(cfg, model, tok, data, arms=("alphasteer",))
    assert st["alphasteer"]["state"] == "refused"
    assert "eligibility" in st["alphasteer"]["reason"] and "D38" in st["alphasteer"]["reason"]
    assert "leakage eligibility bound" in st["alphasteer"]["reason"]
    assert not fcb._object_path(cfg, "alphasteer").exists()


def test_require_alphasteer_rule_refuses_a_config_without_the_block(tmp_path, monkeypatch, capsys):
    """With the exemption key ABSENT the alphasteer arm silently takes
    the legacy single-layer path and exits 0 — indistinguishable from
    success. `--require-alphasteer-rule` turns that into a refusal (exit 2)
    BEFORE any model is loaded; with the block present it is a no-op; the
    shipped Llama config (no block, legacy path intended) never passes it."""
    def boom(*a, **k):
        raise AssertionError("cmd_fit must not run after the refusal")

    monkeypatch.setattr(fcb, "cmd_fit", boom)
    plain = tmp_path / "plain.yaml"
    plain.write_text(json.dumps(_cfg(tmp_path)))
    with pytest.raises(SystemExit) as e:
        fcb.main(["--config", str(plain), "--require-alphasteer-rule"])
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert "no class_b.alphasteer block" in err and "layer 1" in err and "D39" in err
    # the same config WITHOUT the flag proceeds to the fit (the legacy path)
    ran = {}
    monkeypatch.setattr(fcb, "cmd_fit", lambda args, cfg: ran.setdefault("cfg", cfg))
    fcb.main(["--config", str(plain)])
    assert ran["cfg"]["layer"] == 1 and "alphasteer" not in ran["cfg"]["class_b"]
    # with the block present the flag is satisfied and the fit proceeds
    ruled = tmp_path / "ruled.yaml"
    ruled.write_text(json.dumps(_rule_cfg(tmp_path)))
    ran.clear()
    fcb.main(["--config", str(ruled), "--require-alphasteer-rule"])
    assert "cfg" in ran
    # the shipped Llama sbatch does not pass the flag (legacy path intended),
    # the Phase B wrapper always does
    assert "--require-alphasteer-rule" not in (REPO / "scripts/sbatch/fit_class_b.sbatch").read_text()
    assert "--require-alphasteer-rule" in (REPO / "scripts/sbatch/alphasteer_fit_f22.sbatch").read_text()


def test_missing_or_torn_screen_refuses(tiny_llama, tmp_path):
    model, tok = tiny_llama
    cfg = _rule_cfg(tmp_path)
    data = fcb.build_class_b_data(cfg, tok)
    st = fcb.fit_and_save(cfg, model, tok, data, arms=("alphasteer",))
    assert st["alphasteer"]["state"] == "refused"
    assert "FileNotFoundError" in st["alphasteer"]["reason"]


def test_multilayer_fit_captures_through_the_one_seam_at_the_screens_width(
        tiny_llama, tmp_path, monkeypatch):
    """The fit's activations come through capture_or_load_side over
    the FULL lists at the RULE's batch width in BOTH branches — acts dir
    absent (captured, sentinel written) and present (reused, no forward) —
    and the trainer ALWAYS receives precaptured=. The two fits are
    bit-identical, and the fixture's widths DIFFER (class_b.batch_size 3 vs
    capture_batch_size 8) so a trainer-side capture at the fit width would
    be caught by the spy (the d=16 fixture's bits do not depend on width, so
    the pin is on the call, not on inequality). A sentinel under another
    binding refuses."""
    import torch

    from sisteer.extraction import alphasteer, alphasteer_screen as scr

    model, tok = tiny_llama
    cfg = _rule_cfg(tmp_path, capture_batch_size=8)
    cfg["class_b"]["batch_size"] = 3
    rule = fcb.alphasteer_layer_rule(cfg)
    data = fcb.build_class_b_data(cfg, tok)
    _write_tiny_screen(cfg, rule, data, tmp_path, model=model)          # acts_dir=None -> rule["acts_dir"]
    widths, precaptured = [], []
    real_collect = alphasteer._collect_last_acts
    monkeypatch.setattr(scr, "_collect_last_acts",
                        lambda m, tk, texts, layers, site, bs, **kw: (
                            widths.append(bs), real_collect(m, tk, texts, layers, site, bs, **kw))[1])
    real_train = alphasteer.train_alphasteer_content_multilayer
    monkeypatch.setattr(alphasteer, "train_alphasteer_content_multilayer",
                        lambda *a, **kw: (precaptured.append(kw.get("precaptured") is not None),
                                          real_train(*a, **kw))[1])
    # (a) absent: captured through the seam at the RULE's width, sentinel bound
    st = fcb.fit_and_save(cfg, model, tok, data, arms=("alphasteer",))
    assert st["alphasteer"]["state"] == "fit", st
    assert widths == [8, 8] and precaptured == [True]        # never class_b.batch_size (3)
    captured = load_object(fcb._object_path(cfg, "alphasteer"), model, tok)
    acts_dir = Path(rule["acts_dir"])
    assert captured.metadata["f9"]["acts_source"] == {
        "acts_dir": str(acts_dir), "reused": {"benign": False, "harmful": False},
        "capture_batch_size": {"benign": 8, "harmful": 8},
        "capture_batch_size_source": "screen.inputs_binding.capture",
        "rule_capture_batch_size": 8}
    sent = json.loads((acts_dir / "benign" / "capture_done.json").read_text())
    assert sent["binding"]["batch_size"] == 8 and sent["binding"]["model"] == cfg["model"]
    assert sent["binding"]["n_rows"] == 30 and sent["binding"]["site"] == "pre"
    assert sent["binding"]["list_sha256"] == data["content_meta"]["benign_sha256"]
    # the sentinel's binding IS the screen's recorded one (P1-a), precision included (P1-b)
    screen = scr.read_fingerprinted_json(Path(rule["screen_path"]))
    assert sent["binding"] == screen["inputs_binding"]["capture"]["benign"]
    assert sent["binding"]["dtype"] == "float32" and sent["binding"]["device"] == "cpu"
    assert sent["binding"]["attn_implementation"] == model.config._attn_implementation
    # (b) present: reused under the matching binding, NO forward, same bits
    cfg2 = json.loads(json.dumps(cfg))
    cfg2["defenses_dir"] = str(tmp_path / "defenses_reused")
    forwards = []
    h = model.register_forward_pre_hook(lambda *_: forwards.append(1))
    try:
        st2 = fcb.fit_and_save(cfg2, model, tok, data, arms=("alphasteer",))
    finally:
        h.remove()
    assert st2["alphasteer"]["state"] == "fit", st2
    assert forwards == [] and widths == [8, 8] and precaptured == [True, True]
    reused = load_object(fcb._object_path(cfg2, "alphasteer"), model, tok)
    assert reused.metadata["f9"]["acts_source"]["reused"] == {"benign": True, "harmful": True}
    assert torch.equal(reused.Ms[1], captured.Ms[1])
    assert reused.metadata["capture"]["benign"] == sent
    # (c) a cache under another binding than the screen's: refused, never
    # captured over. (The RULE's width moving no longer reaches the fit — the
    # screen's recorded width is adopted, see the P1-a test below — so the
    # foreign cache is planted in the sentinel itself.)
    cfg3 = json.loads(json.dumps(cfg))
    cfg3["defenses_dir"] = str(tmp_path / "defenses_foreign")
    sp = acts_dir / "benign" / "capture_done.json"
    foreign = json.loads(sp.read_text())
    foreign["binding"]["batch_size"] = 4
    sp.write_text(json.dumps(foreign))
    st3 = fcb.fit_and_save(cfg3, model, tok, data, arms=("alphasteer",))
    assert st3["alphasteer"]["state"] == "refused"
    assert "CheckpointMismatch" in st3["alphasteer"]["reason"]
    assert "batch_size" in st3["alphasteer"]["reason"]
    assert widths == [8, 8] and not fcb._object_path(cfg3, "alphasteer").exists()


def test_fit_holds_the_screens_capture_binding_before_it_captures(tiny_llama, tmp_path, monkeypatch):
    """The fit's capture inputs are the SCREEN's, held against
    ``inputs_binding.capture`` BEFORE anything is loaded or captured. With
    the cache ABSENT (the screen artifact moved alone), a screen bound to
    batch 32 is recaptured at 32 — the screen's width, not the rule's 8 —
    and the object records where the width came from (ADOPTED: a recorded
    throughput override the fit must reproduce). A screen bound to another
    precision, another attention kernel, another code sha, another list, or
    to nothing at all REFUSES with no forward pass, no sentinel written and
    nothing fitted (REQUIRED equal: the fit cannot produce the screen's
    matrices under any of them)."""
    from sisteer.extraction import alphasteer, alphasteer_screen as scr

    model, tok = tiny_llama
    cfg = _rule_cfg(tmp_path, capture_batch_size=8)
    rule = fcb.alphasteer_layer_rule(cfg)
    data = fcb.build_class_b_data(cfg, tok)
    widths = []
    real_collect = alphasteer._collect_last_acts
    monkeypatch.setattr(scr, "_collect_last_acts",
                        lambda m, tk, texts, layers, site, bs, **kw: (
                            widths.append(bs), real_collect(m, tk, texts, layers, site, bs, **kw))[1])
    # ADOPTED: the screen ran with --batch-size 32; the fit follows the screen
    _write_tiny_screen(cfg, rule, data, tmp_path, model=model, capture_over={"batch_size": 32})
    assert not Path(rule["acts_dir"]).exists()                 # the cache is absent
    st = fcb.fit_and_save(cfg, model, tok, data, arms=("alphasteer",))
    assert st["alphasteer"]["state"] == "fit", st
    assert widths == [32, 32]                                  # never the rule's 8
    obj = load_object(fcb._object_path(cfg, "alphasteer"), model, tok)
    src = obj.metadata["f9"]["acts_source"]
    assert src["capture_batch_size"] == {"benign": 32, "harmful": 32}
    assert src["capture_batch_size_source"] == "screen.inputs_binding.capture"
    assert src["rule_capture_batch_size"] == 8 and src["reused"] == {"benign": False, "harmful": False}
    sent = json.loads((Path(rule["acts_dir"]) / "benign" / "capture_done.json").read_text())
    assert sent["binding"]["batch_size"] == 32
    assert sent["binding"] == scr.read_fingerprinted_json(
        Path(rule["screen_path"]))["inputs_binding"]["capture"]["benign"]
    # REQUIRED equal: every other key. Each foreign screen refuses before any
    # forward, writes no sentinel, fits nothing.
    forwards = []
    h = model.register_forward_pre_hook(lambda *_: forwards.append(1))
    try:
        for k, (over, key) in enumerate((
                ({"dtype": "bfloat16"}, "dtype"),                  # captured at another precision
                ({"attn_implementation": "eager"}, "attn_implementation"),
                ({"code_sha": "f" * 40}, "code_sha"),              # a recapture here is not the screen's capture
                ({"list_sha256": "0" * 64}, "list_sha256"),
                (None, "inputs_binding"))):                        # a screen that predates bindings
            c = json.loads(json.dumps(cfg))
            c["defenses_dir"] = str(tmp_path / f"defenses_{k}")
            c["class_b"]["alphasteer"]["screen_path"] = str(tmp_path / f"screen_{k}" / "screen_f9.json")
            r = fcb.alphasteer_layer_rule(c)
            _write_tiny_screen(c, r, data, tmp_path, model=model, capture_over=over,
                               strip_binding=over is None)
            st = fcb.fit_and_save(c, model, tok, data, arms=("alphasteer",))
            assert st["alphasteer"]["state"] == "refused", (over, st)
            assert key in st["alphasteer"]["reason"], (over, st)
            if over is not None:
                assert "CheckpointMismatch" in st["alphasteer"]["reason"], (over, st)
            assert not Path(r["acts_dir"]).exists(), over              # nothing captured
            assert not fcb._object_path(c, "alphasteer").exists()       # nothing fitted
    finally:
        h.remove()
    assert forwards == [] and widths == [32, 32]


def test_compute_content_grouped_split_uses_the_slices_beside_the_lists(tiny_llama, tmp_path):
    _, tok = tiny_llama
    cfg = _rule_cfg(tmp_path)
    rule = fcb.alphasteer_layer_rule(cfg)
    data = fcb.build_class_b_data(cfg, tok)
    split = fcb.compute_content_grouped_split(cfg, rule, data)
    assert set(split["realised_by_source"]) == {"benign_train", "coconot",
                                                "harmful_train", "jailbreak_train"}
    assert split["slices_source"]["benign"].endswith("benign_alphasteer.slices.json")
    assert split["content_disjoint_verified"] is True
    # a list without a slices file is one source, said so
    (tmp_path / "harmful_alphasteer.slices.json").unlink()
    split2 = fcb.compute_content_grouped_split(cfg, rule, data)
    assert "all" in split2["realised_by_source"]
    assert split2["slices_source"]["harmful"].startswith("absent")


def test_sbatch_is_parameterised_positionally_and_keeps_the_hard_rules():
    """scripts/sbatch/fit_class_b.sbatch: positional config/arms/out, the
    numeric-$1 resubmit contract, the screen step before the fit, --time
    raised to 6h, and every hard-rule flag intact; no --export anywhere."""
    import subprocess

    text = (REPO / "scripts" / "sbatch" / "fit_class_b.sbatch").read_text()
    # directive text only: the part after #SBATCH and before any trailing
    # "# ..." comment (the header annotates hard rule 2 by naming --mem=0)
    header = [l.split("#SBATCH", 1)[1].split("#", 1)[0].strip()
              for l in text.splitlines() if l.startswith("#SBATCH")]
    flags = " ".join(header)
    for want in ("--qos=high", "--gres=gpu:1", "--mem=128G", "--time=0-06:00:00",
                 "--no-requeue", "--signal=B:TERM@120", "--open-mode=append",
                 "--partition=general"):
        assert want in flags, want
    # the hard rules are about DIRECTIVES and COMMANDS, not the prose that
    # warns about them: check the #SBATCH lines and the non-comment body
    code = "\n".join(l for l in text.splitlines()
                     if l.strip() and not l.lstrip().startswith("#"))
    assert "--export" not in flags and "--export" not in code
    assert "--mem=0" not in flags and "--mem=0" not in code
    assert "--qos=dev" not in flags and "--qos=dev" not in code
    assert 'FITB_CONFIG="${1:-configs/two_si.yaml}"' in text
    assert 'FITB_ARMS="${2:-}"' in text and 'FITB_OUT="${3:-}"' in text
    assert '=~ ^[0-9]+$' in text                     # a numeric $1 is a resubmit
    body = text.split("set -uo pipefail", 1)[1]
    # the bindings are written (atomically) and read BEFORE s2_init is entered
    # (positions in the non-comment CODE, so prose about s2_init does not count)
    assert code.index("fit_class_b.bindings") < code.index("s2_init")
    assert code.index('mv -f "$FITB_BINDINGS.tmp" "$FITB_BINDINGS"') < code.index("s2_init 1 5")
    assert code.index('source "$FITB_BINDINGS"') < code.index('s2_init "$1"')
    assert body.index("alphasteer_screen.py") < body.index("fit_class_b.py")
    assert "--if-configured" in body and "--resume" in body
    assert '--out "$FITB_OUT_DIR"' in body
    assert "s2_run" in body and "s2_init" in body
    subprocess.run(["bash", "-n", str(REPO / "scripts" / "sbatch" / "fit_class_b.sbatch")],
                   check=True)


def test_phase_b_sbatch_owns_its_root_guards_on_the_gate_and_requires_the_rule():
    """scripts/sbatch/alphasteer_fit_f22.sbatch (alphasteer-full22 Phase B):
    the as22 checkout root (never the peers' shared checkout), the in-body
    llama_gate.json passed:true guard BEFORE any GPU work, the screen step
    WITHOUT --if-configured (a config with no rule block refuses loudly),
    the fit WITH --require-alphasteer-rule, the bindings-before-s2_init
    resubmit contract verbatim, the campaign figure push, and every
    hard-rule flag intact."""
    import subprocess

    path = REPO / "scripts" / "sbatch" / "alphasteer_fit_f22.sbatch"
    text = path.read_text()
    header = [l.split("#SBATCH", 1)[1].split("#", 1)[0].strip()
              for l in text.splitlines() if l.startswith("#SBATCH")]
    flags = " ".join(header)
    for want in ("--job-name=as22-fit", "--qos=high", "--gres=gpu:1", "--mem=128G",
                 "--time=0-08:00:00", "--no-requeue", "--signal=B:TERM@120",
                 "--open-mode=append", "--partition=general"):
        assert want in flags, want
    code = "\n".join(l for l in text.splitlines()
                     if l.strip() and not l.lstrip().startswith("#"))
    assert "--export" not in flags and "--export" not in code
    assert "--mem=0" not in flags and "--mem=0" not in code
    assert "--qos=dev" not in flags and "--qos=dev" not in code
    assert not any(re.search(r"(^|\s)(export\s+)?CUDA_VISIBLE_DEVICES=", l) for l in code.splitlines())
    assert 'S2_ROOT="${SISTEER_PROJECT_ROOT:?set SISTEER_PROJECT_ROOT, the cluster project directory (the front door exports it from the site config)}-as22"' in code            # own checkout, never the peers'
    assert code.index('S2_ROOT="${SISTEER_PROJECT_ROOT:?set SISTEER_PROJECT_ROOT, the cluster project directory (the front door exports it from the site config)}-as22"') < code.index("stage2_common.sh")
    assert not re.search(r"\$\{?SISTEER_PROJECT_ROOT(?::\?[^}]*)?\}?/repo", code)
    # the resubmit contract, verbatim from fit_class_b.sbatch
    ref = (REPO / "scripts" / "sbatch" / "fit_class_b.sbatch").read_text()
    for line in ('=~ ^[0-9]+$', 'source "$FITB_BINDINGS"', 's2_init "$1" "${2:-5}"',
                 'mv -f "$FITB_BINDINGS.tmp" "$FITB_BINDINGS"', "s2_init 1 5",
                 '[ "$FITB_BINDINGS" != "$STOP_DIR/fit_class_b.bindings" ]'):
        assert line in ref and line in text, line
    assert code.index("fit_class_b.bindings") < code.index("s2_init")
    assert code.index('mv -f "$FITB_BINDINGS.tmp" "$FITB_BINDINGS"') < code.index("s2_init 1 5")
    # the venv extras come from the CONFIG (`env_extras:` -- `mamba`, the fused
    # kernels REQUIRED on the Mamba-hybrid target), derived in BOTH branches
    # right before their s2_init: never hardcoded (that would sync the kernels
    # into every target's venv), never via --export (hard rule 1)
    assert "as22_env_extras()" in code and "S2_UV_EXTRAS=mamba" not in code
    derive = 'S2_UV_EXTRAS="$(as22_env_extras "$S2_ROOT/repo/$FITB_CONFIG")"'
    assert code.count(derive) == 2
    first = code.index(derive)
    second = code.index(derive, first + 1)
    assert first < code.index('s2_init "$1" "${2:-5}"') < second < code.index("s2_init 1 5")
    # the helper, run for real against the shipped configs: the Mamba-hybrid
    # target names the extra, the other three name none (their syncs are bare)
    fn = re.search(r"as22_env_extras\(\) \{\n.*?\n\}", text, re.S).group(0)
    for key, want in (("nemotron3nano4b", "mamba"), ("gemma4e4b", ""),
                      ("qwen35_4b", ""), ("glm46vflash", "")):
        got = subprocess.run(
            ["bash", "-c", fn + '\nas22_env_extras "$1"', "_",
             str(REPO / f"configs/alphasteer_f22_{key}.yaml")],
            capture_output=True, text=True, check=True).stdout.strip()
        assert got == want, (key, got)
    # and a flow-list spelling parses to the same word list
    got = subprocess.run(["bash", "-c", fn + '\nas22_env_extras "$1"', "_", "/dev/stdin"],
                         input="model: x\nenv_extras: [mamba, longctx]  # note\n",
                         capture_output=True, text=True, check=True).stdout.strip()
    assert got == "mamba longctx"
    # the gate guard sits after s2_init and before the screen and the fit
    assert "llama_gate.json" in code and "read_fingerprinted_json" in code
    assert code.index("llama_gate.json") < code.index("alphasteer_screen.py") < code.index("fit_class_b.py")
    assert 'get("passed") is True' in code and 'gate_reading") == "reading_released"' in code
    assert "--if-configured" not in code                             # absence must refuse, not no-op
    assert "--require-alphasteer-rule" in code
    assert "--resume" in code and "--arms" in code and "--out" in code
    assert "as22_finish" in code and "alphasteer-full22" in code and "mirror-figs.sh" in code
    assert "final_auroc_vs_depth_by_p_" in code
    subprocess.run(["bash", "-n", str(path)], check=True)


def test_phase_b_configs_mirror_each_target_pin_and_carry_the_rule_block():
    """configs/alphasteer_f22_<target>.yaml exist for all four Phase B
    targets, take model / revision / dtype / attn and the class_b lists from
    the target's two_si_<target>.yaml, and carry the f9_auroc rule block
    with its own screen_path — the absence of which is what would silently
    have fit four single-layer L15 objects under the multi-layer arm's name."""
    for key in ("gemma4e4b", "qwen35_4b", "glm46vflash", "nemotron3nano4b"):
        cfg = yaml.safe_load((REPO / f"configs/alphasteer_f22_{key}.yaml").read_text())
        ship = yaml.safe_load((REPO / f"configs/two_si_{key}.yaml").read_text())
        for k in ("model", "model_revision", "dtype", "attn_implementation", "layer"):
            assert cfg[k] == ship[k], (key, k)
        for k in ("benign_prompts_path", "harmful_prompts_path", "benign_sha256",
                  "harmful_sha256", "min_prompts", "layer"):
            assert cfg["class_b"][k] == ship["class_b"][k], (key, k)
        assert "alphasteer" not in ship["class_b"]           # the shipped configs stay single-layer
        assert cfg["class_b_arms"] == ["alphasteer"]
        rule = fcb.alphasteer_layer_rule(cfg)
        assert rule is not None and rule["layer_rule"] == "f9_auroc"
        assert rule["model_key"] == key and fcb.alphasteer_model_key(cfg, rule) == key
        assert rule["screen_path"].endswith(f"alphasteer_f22/{key}/screen_f9.json")
        assert rule["leakage_report_path"].endswith("alphasteer_f22/leakage_report.json")
        assert rule["auroc_threshold"] == 0.95 and "leakage_max" not in rule
        assert "llama_gate" not in cfg                        # Phase D's block is the gate's alone
        # the fused-kernel venv extra is declared by the Mamba-hybrid target ONLY
        # (the other three sync bare, so their venvs -- hence their captures --
        # are untouched by another target's kernels); the shipped configs never
        assert cfg.get("env_extras") == ("mamba" if key == "nemotron3nano4b" else None), key
        assert "env_extras" not in ship


def test_screen_cli_noops_without_the_block_or_arm(tmp_path, capsys):
    """scripts/alphasteer_screen.py --if-configured / --arms exit 0 doing
    nothing on a config without the exemption (the shipped Llama config
    path), and refuse loudly without --if-configured."""
    import alphasteer_screen as cli

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(json.dumps(_cfg(tmp_path)))
    assert cli.main(["--config", str(cfg_path), "--if-configured"]) == 0
    assert "no class_b.alphasteer" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="no class_b.alphasteer"):
        cli.main(["--config", str(cfg_path)])
    rule_cfg = tmp_path / "rule.yaml"
    rule_cfg.write_text(json.dumps(_rule_cfg(tmp_path)))
    assert cli.main(["--config", str(rule_cfg), "--arms", "antidote,caa_static_r"]) == 0
    assert "do not include alphasteer" in capsys.readouterr().out


# ---------------------------------------------------------------- the sbatch, executed


STUB_SHA = "cafecafe" * 5
FITB_FAKE_USER = "fitbxuser"     # no such account: no real path is ever touched

# The stubbed cluster (test_capeval_stage2_common's pattern): git answers HEAD,
# scontrol renders the pinned job record (Comment pin with the cluster's
# trailing space, the Command= the resubmit re-reads, TimeLimit/StdOut it
# carries forward), srun/sbatch record their argv, uv delivers TERM to the
# script during `uv sync` when asked — INSIDE s2_init, after the handler is
# armed and before it returns.
_SBATCH_STUBS = r"""
git() { echo "$STUB_SHA"; }
scontrol() {
  printf '   Comment=sha:%s \n   Command=%s\n   TimeLimit=01:00:00\n   StdOut=%s/job.out\n' \
    "$STUB_SHA" "$FITB_SCRIPT" "$FITB_TMP"
}
srun() { printf '%s\n' "$*" >> "$SRUN_RECORD"; }
sbatch() { printf '%s\n' "$*" >> "$SBATCH_RECORD"; }
uv() { if [ "${1:-}" = sync ] && [ "${FITB_TERM_IN_UV_SYNC:-0}" = 1 ]; then kill -TERM $$; fi; }
export -f git scontrol srun sbatch uv
exec bash "$FITB_SCRIPT" "$@"
"""


def _sbatch_world(tmp_path):
    """fit_class_b.sbatch and stage2_common.sh under a tmp $SHARED: both
    scripts read the cluster's shared root from the job environment
    (SISTEER_SHARED_ROOT, set below, as the front door passes it) and
    s2_init reads the escalation QoS from the job's ESCALATE_QOS, so the
    copies are byte for byte; the job script's one site line is counted so
    a drift fails here instead of silently testing nothing."""
    import os

    shared = tmp_path / "shared"
    repo = shared / "proj" / "repo"
    (repo / "scripts" / "sbatch").mkdir(parents=True)
    common = (REPO / "scripts" / "sbatch" / "stage2_common.sh").read_text()
    assert 'SHARED="${SISTEER_SHARED_ROOT:?' in common and '[ -d "/' not in common
    (repo / "scripts" / "sbatch" / "stage2_common.sh").write_text(common)
    (shared / "proj" / "EXPECTED_SHA").write_text(STUB_SHA + "\n")
    (shared / "scripts").mkdir()
    (shared / "scripts" / "hf_env.sh").write_text("")
    text = (REPO / "scripts" / "sbatch" / "fit_class_b.sbatch").read_text()
    assert text.count('SHARED="${SISTEER_SHARED_ROOT:?') == 1     # the job environment names the shared root (set below)
    script = tmp_path / "fit_class_b.sbatch"
    script.write_text(text)
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("S2_", "SLURM_", "STUB_", "FITB_", "UV_"))}
    env.update(USER=FITB_FAKE_USER, SLURM_JOB_ID="4242",
               SLURM_JOB_NAME="tt-twosi-fitb-test", STUB_SHA=STUB_SHA,
               FITB_SCRIPT=str(script), FITB_TMP=str(tmp_path),
               S2_UV_ENV=str(tmp_path / "venv"), SISTEER_PROJECT_ROOT=str(shared / "proj"), SISTEER_SHARED_ROOT=str(shared),
               ESCALATE_QOS="qos_nonpreempt",
               SRUN_RECORD=str(tmp_path / "srun.log"),
               SBATCH_RECORD=str(tmp_path / "sbatch.log"))
    return shared, env


def _run_sbatch(env, *args):
    import subprocess

    return subprocess.run(["bash", "-c", _SBATCH_STUBS, "harness", *args],
                          capture_output=True, text=True, env=env, timeout=120)


def test_sbatch_term_during_init_resubmits_a_successor_that_finds_its_bindings(tmp_path):
    """Executed: attempt 1 is TERMed INSIDE s2_init (during `uv sync`,
    after the escalation handler is armed) -> it resubmits itself with the
    numeric contract, and the bindings file ALREADY exists, so the successor
    (attempt 2) reloads config, arms and out dir and runs the screen step and
    the fit with exactly them. A successor without a bindings file is FATAL."""
    shared, env = _sbatch_world(tmp_path)
    cfg, arms, out = "configs/alphasteer_f22_x.yaml", "alphasteer", str(tmp_path / "cells")
    p1 = _run_sbatch({**env, "FITB_TERM_IN_UV_SYNC": "1"}, cfg, arms, out)
    assert p1.returncode == 143, p1.stdout + p1.stderr
    assert "SIGTERM (preemption or cancel)" in p1.stdout
    assert "config=" not in p1.stdout            # never got past s2_init: TERM landed inside it
    bindings = shared / "exp" / "training" / "tt-twosi-fitb-test" / "fit_class_b.bindings"
    assert bindings.read_text() == f"FITB_CONFIG={cfg}\nFITB_ARMS={arms}\nFITB_OUT={out}\n"
    assert not bindings.with_name(bindings.name + ".tmp").exists()
    resubmits = (tmp_path / "sbatch.log").read_text().splitlines()
    assert len(resubmits) == 1 and resubmits[0].endswith(f"{tmp_path / 'fit_class_b.sbatch'} 2 5")
    assert "--job-name=tt-twosi-fitb-test" in resubmits[0]
    assert f"--comment=sha:{STUB_SHA}" in resubmits[0]
    # the successor, exactly as s2_escalate invokes it: numeric $1 $2
    (tmp_path / "srun.log").unlink(missing_ok=True)
    p2 = _run_sbatch(env, "2", "5")
    assert p2.returncode == 0, p2.stdout + p2.stderr
    assert "resubmission: bindings reloaded" in p2.stdout
    assert f"config={cfg} arms={arms} out={out}" in p2.stdout
    sruns = (tmp_path / "srun.log").read_text().splitlines()
    screen = [s for s in sruns if "alphasteer_screen.py" in s]
    fit = [s for s in sruns if "fit_class_b.py" in s]
    assert len(screen) == 1 and len(fit) == 1 and sruns.index(screen[0]) < sruns.index(fit[0])
    assert screen[0].endswith(f"python scripts/alphasteer_screen.py --config {cfg} "
                              f"--if-configured --resume --arms {arms}")
    assert fit[0].endswith(f"python scripts/fit_class_b.py --config {cfg} --arms {arms} --out {out}")
    # no bindings file -> a numeric resubmission is FATAL, before s2_init
    bindings.unlink()
    p3 = _run_sbatch(env, "2", "5")
    assert p3.returncode == 2 and "FATAL: resubmission" in p3.stdout
    assert "escalation qos" not in p3.stdout       # refused before s2_init ran at all
