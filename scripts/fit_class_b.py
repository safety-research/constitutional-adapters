"""Class B (SI-agnostic) fit-once driver for the two-SI matrix.

Fits the SI-agnostic baselines — ``antidote``, ``alphasteer`` and
``caa_static_r`` (the CAA / diff-in-means floor, 2026-08-21) — ONCE on
harmful/benign CONTENT (no system instruction), and writes each object to the
cell-id-keyed defenses dir the moment it completes. This is the missing half
of the two-SI fit pipeline: the per-SI Class A objects are trained by
``train_two_si_objects.py``; the SI-agnostic Class B objects are trained here.

Data: both Class B arms fit on the AlphaSteer
AUTHORS' exact harmful-vs-benign prompt composition — 14,000 benign / 2,000
harmful — materialised as template-agnostic flat JSON prompt lists (the
an earlier development branch's ``materialize_alphasteer_contrast.py`` replay of
the authors' calc_steering_matrix.py, seed and RNG order included) and
consumed VERBATIM through ``build_content_contrast``'s
``benign_prompts_path`` / ``harmful_prompts_path`` seam (design lifted from
that peer's commit 9d95cb5). The shipped ``configs/two_si.yaml`` pins the two
files (cluster paths + recorded sha256s); each fitted object's meta.json
records the realised counts and file hashes.

This replaces the previous default assembly for Class B, which drew 2,000
benign (Alpaca + OR-Bench-80k) against 41 harmful — HarmBench-val's 80 rows
silently reduced to the standard-only functional subset by
``load_harmbench``'s default — for ``alphasteer``'s d = 4096 ridge:
catastrophically underdetermined, and nothing surfaced the count. Two
guards now stop a recurrence, LOUDLY:

- ``min_prompts`` (default 500) is ALWAYS passed to the contrast builder
  here, so an undersized side refuses before any capture forward;
- the trainers themselves label any n < d fit with the realised n and d
  (``extraction.fit_checks.check_sample_floor``) — the authors' own
  2,000-harmful side is still n < d at d = 4096, legal under the ridge but
  never again silent.

Layer pin: both Class B arms fit at the SINGLE
campaign layer — 15 on the pinned Llama-3.1-8B — for comparability with every
other arm. The AlphaSteer paper steers a 10-layer subset; that divergence is
deliberate and documented in extraction/alphasteer.py. The pin is enforced
here: ``class_b.layer`` (falling back to the config's shared ``layer``) must
be a single int, and a list/tuple refuses loudly rather than quietly fitting
a multi-layer variant.

circuit_breaker left the registry ENTIRELY on 2026-08-20, alongside cast:
it is no longer a Class B arm here, nothing trains or enumerates it, and its
store-reading reroute/retain machinery left this driver with it. Its
implementation (extraction/circuit_breaker.py) and its ``load_object``
hydration survive, so the already-fitted on-disk artifact — which a peer lane
uses as a serving/kind-coverage target — stays readable.

Robustness (the failure mode that killed the overnight Class A jobs
148862/148863, applied here as a design rule): **each object is saved the
instant it completes**, and each arm is try/except-isolated — one arm's
refusal is recorded, never fatal to the others. Re-running skips any arm
whose object already exists.

Every outside touch (model load, prompt-file read, the trainers) is
injectable, so the assembly + fit-loop + save/resume logic is unit-tested on
a tiny CPU model with synthetic content (tests/test_fit_class_b.py) with no
network and no cluster files.

MULTI-LAYER EXEMPTION FOR ``alphasteer`` ONLY — the single-layer pin does not
apply to this one arm. ``class_b_layer`` is UNTOUCHED and still refuses a
list for every arm. The exemption is a SEPARATE function on a SEPARATE
config key::

    class_b:
      alphasteer:                 # ABSENT on configs/two_si.yaml -> nothing changes
        layer_rule: f9_auroc      # the ONLY accepted value; anything else refuses
        depth_band: [0.15, 0.65]
        p_grid: [0.3, 0.4, 0.5, 0.6]
        auroc_threshold: 0.95     # eligibility is AUROC ONLY; leakage is a reported diagnostic
        heldout_frac: 0.10
        split_seed: 0
        screen_path: <abs path to screen_f9.json>   # produced by scripts/alphasteer_screen.py
        lambda_reg: 10.0
        site: pre
        steer_positions: prefill

``alphasteer_layer_rule(cfg)`` returns the validated block or None. With the
key absent the ``_alphasteer`` closure calls ``train_alphasteer_content``
exactly as before (byte-identical Llama behaviour); with it present the
closure reads the screen artifact, recomputes the content-grouped split and
checks its sha against the screen's, and fits
``train_alphasteer_content_multilayer`` on the FIT rows over the screen's
chosen layers and per-layer ratios, uncalibrated (the strength is the served
lambda ladder's, plan Phase C). The multi-layer arm ignores the driver-level
pin exactly as ``caa_static_r`` does; the pin is still resolved and recorded.
Beside ``meta.json`` (the resume sentinel, still written LAST) it writes
``fit_meta.json`` and ``fit_provenance.json`` — for this path
only, so no other arm's directory changes. ``fit_meta.json`` carries
``object_sha256``, the hash of the saved object file: the ONE field
that binds the record to the bytes it describes, so a consumer (the Phase C
bake) can refuse a stale object/record pair instead of inferring the tie.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling scripts

REPO = Path(__file__).resolve().parents[1]

# The object<->record binding lives in sisteer.extraction.fit_meta_binding (pure stdlib, shared
# with the cluster-side backfill CLI and adoptable by the Phase C bake): OBJECT_SHA256_PRODUCER_SOURCE
# is what `object_sha256_source` says when THIS writer computed the hash; a record whose source starts
# with "backfilled" was hashed after the fact by scripts/backfill_fit_meta_object_sha256.py (the four
# Phase B fits predate the field) and must never be read as producer-written provenance.
from sisteer.extraction.fit_meta_binding import (  # noqa: E402
    OBJECT_SHA256_PRODUCER_SOURCE, sha256_file as _sha256_file)

# The SI-agnostic baselines, in matrix order (defense_cell.CLASS_B_ARMS).
# The TRAINED set for a run comes from the config (default_class_b_arms below,
# mirroring train_two_si_objects.default_class_a_arms): configs/two_si.yaml is
# the submission-time record of the matrix. circuit_breaker was removed from
# the registry ENTIRELY on 2026-08-20 and left this tuple
# with it — its fitted artifact stays loadable via methods.load_object.
# caa_static_r is the CAA / diff-in-means floor —
# an earlier development branch's static_r fit, deployment-matched to
# alphasteer (extraction/caa_static_r.py) — APPENDED LAST so the existing
# Class B order keeps its meaning.
CLASS_B_ARMS = ("antidote", "alphasteer", "caa_static_r")


def default_class_b_arms(cfg: dict) -> list:
    """The Class B arms the next fit wave trains: config ``class_b_arms``
    (the submission-time record), falling back to the module tuple —
    identical derivation to the Class A driver's ``default_class_a_arms``."""
    return list(cfg.get("class_b_arms") or CLASS_B_ARMS)


# --------------------------------------------------------------------------- #
# config                                                                       #
# --------------------------------------------------------------------------- #


def load_cfg(path: str) -> dict:
    import yaml

    p = Path(path)
    if not p.is_absolute():
        p = REPO / path
    return yaml.safe_load(p.read_text())


def _class_b_cfg(cfg: dict) -> dict:
    """The Class B knobs, with defaults. An optional ``class_b:`` config block
    overrides any of them (absent keys keep the default).

    ``benign_prompts_path`` / ``harmful_prompts_path`` are the explicit
    prompt-list files (the AlphaSteer authors' composition in production —
    the shipped config pins them); when unset, the legacy
    build_content_contrast assembly applies. ``min_prompts`` is ALWAYS
    passed through (never None), so the undersized-fit floor is on for
    every Class B fit regardless of which path supplied the data."""
    d = {
        "n_benign": 2000,
        "alpaca_frac": 0.5,
        "n_harmful": None,          # all of HarmBench-val
        "harmbench_split": "val",   # fit/eval-disjoint (test is the eval split)
        "orbench_config": "or-bench-80k",
        "benign_prompts_path": None,
        "harmful_prompts_path": None,
        # Optional sha256 pins for the two prompt files (the shipped config
        # sets both): checked against the sha build_content_contrast records
        # for what it ACTUALLY read, so a path remap (below) or a silently
        # edited file can never swap the fit set unnoticed.
        "benign_sha256": None,
        "harmful_sha256": None,
        "min_prompts": 500,
        "layer": None,              # None -> the config's shared `layer`
        "seed": 0,
        "batch_size": 8,
    }
    d.update(cfg.get("class_b") or {})
    return d


def resolve_cluster_path(path: str, *, exists=None, env=None) -> str:
    """A config-pinned path on the cluster that is running. The configs write the prompt lists as
    ``$SHARED/...``: the placeholder expands through ``sisteer.site`` to this job's shared root
    (SISTEER_SHARED_ROOT in a cluster job's environment, else the site config's cluster block), so
    the same config resolves on either cluster with no remapping. An unset or unknown placeholder
    raises ``sisteer.site.PlaceholderError`` naming the variable to set; an expanded path that does
    not exist raises FileNotFoundError naming it. The class_b sha256 pins then verify the bytes.
    ``exists`` and ``env`` are injectable for tests."""
    from sisteer import site as _site

    exists = exists or (lambda p: Path(p).is_file())
    resolved = str(_site.expand_path(path, env=env, key="class_b prompt list"))
    if exists(resolved):
        return resolved
    shown = resolved if resolved == path else f"{resolved} (the config's {path})"
    raise FileNotFoundError(
        f"no file at {shown} -- stage the file under this cluster's shared root")


def _check_sha_pins(cb: dict, meta: dict) -> None:
    """Refuse LOUDLY when a configured sha256 pin does not match the sha of
    the file the contrast builder actually consumed (recorded in its meta).
    A pin with NO recorded sha refuses too: a pinned config means the
    authors'-list path is in force, so content from anywhere else (the
    legacy assembly, an injected contrast) is a misconfiguration."""
    for side in ("benign", "harmful"):
        pin = cb.get(f"{side}_sha256")
        if not pin:
            continue
        got = meta.get(f"{side}_sha256")
        if got != pin:
            raise ValueError(
                f"class_b.{side}_sha256 pin mismatch: config pins {pin}, the "
                f"consumed {side} prompt list hashed to {got} -- the fit set "
                "is not the pinned authors' list. Re-stage the file (its "
                "sha256 must match on every cluster) or update the pin "
                "deliberately.")


def class_b_layer(cfg: dict) -> int:
    """The SINGLE layer the antidote/alphasteer Class B arms fit at
    (pinned 2026-08-20: layer 15 on the pinned Llama-3.1-8B, matching
    every other arm's comparability pin). ``class_b.layer`` wins when
    set; otherwise the config's shared ``layer``. A list/tuple — someone
    reaching for the AlphaSteer paper's 10-layer subset — refuses LOUDLY:
    the single-layer pin is the comparability decision, not an accident of
    plumbing. (caa_static_r is the ONE documented exemption: its identity
    IS the paper's ten-layer matched deployment, fixed inside its own
    trainer — extraction/caa_static_r.py — never through this knob.)"""
    cb = _class_b_cfg(cfg)
    layer = cb["layer"] if cb["layer"] is not None else cfg["layer"]
    if isinstance(layer, (list, tuple)):
        raise ValueError(
            f"Class B fits are pinned to a SINGLE layer (got {layer!r}). "
            "The AlphaSteer paper steers a 10-layer subset, but this "
            "campaign pins every arm to the one comparability layer "
            "(15 on the pinned Llama-3.1-8B) — a multi-layer Class B fit "
            "would be a different experiment; see extraction/alphasteer.py "
            "documented divergence 1.")
    return int(layer)


# The one accepted layer rule for the alphasteer exemption. Anything
# else in `class_b.alphasteer.layer_rule` refuses.
ALPHASTEER_LAYER_RULES = ("f9_auroc",)

def _alphasteer_rule_defaults() -> dict:
    """The exemption block's defaults. The LIBRARY owns every number: the
    screen's ``select_layers`` and the released-Llama recovery gate (plan
    Phase D) default to the same constants this dict reads, so the gate can
    never validate a rule the fits do not run. Lazy on purpose — the
    no-block path must not import torch."""
    from sisteer.extraction import alphasteer_screen as scr

    return {
        "depth_band": list(scr.DEFAULT_DEPTH_BAND),
        "p_grid": list(scr.DEFAULT_P_GRID),
        "auroc_threshold": scr.DEFAULT_AUROC_THRESHOLD,
        # NO leakage bound: eligibility is AUROC only; leakage is reported per
        # layer as a diagnostic. A config carrying `leakage_max` refuses
        # as an unknown key rather than silently re-arming the retired gate.
        "heldout_frac": scr.DEFAULT_HELDOUT_FRAC,
        "split_seed": 0,
        "screen_path": None,        # REQUIRED: where scripts/alphasteer_screen.py wrote screen_f9.json
        "acts_dir": None,           # default: <dirname(screen_path)>/acts
        "split_path": None,         # default: <dirname(screen_path)>/split_v1.json
        "lambda_reg": scr.DEFAULT_LAMBDA_REG,
        "site": "pre",
        "steer_positions": "prefill",
        "capture_batch_size": scr.DEFAULT_CAPTURE_BATCH_SIZE,
        "verify_sample": scr.DEFAULT_VERIFY_SAMPLE,
        "serving_dtype": scr.DEFAULT_SERVING_DTYPE,   # rho's second reading: the WAVE's serving dtype
        "model_key": None,          # roster key for the artifact; default: slug of the model id
        "leakage_report_path": None,  # optional: the frozen Phase 0 report, recorded into fit_meta.json
    }


def alphasteer_layer_rule(cfg: dict) -> dict | None:
    """The config-keyed multi-layer exemption for ``alphasteer`` ONLY.
    Returns the validated ``class_b.alphasteer`` block with defaults
    filled, or None when the key is absent — in which case NOTHING about the
    driver changes (``class_b_layer`` still pins the single layer and
    ``train_alphasteer_content`` runs as today). A block whose
    ``layer_rule`` is anything but ``f9_auroc`` refuses; so does a block
    without a ``screen_path``, a band/grid/threshold outside (0, 1), or a
    deployment the served adapter cannot express (``steer_positions`` other
    than ``prefill``)."""
    cb = cfg.get("class_b") or {}
    block = cb.get("alphasteer")
    if block is None:
        return None
    if not isinstance(block, dict):
        raise ValueError("class_b.alphasteer must be a mapping (the "
                         "multi-layer exemption block), got "
                         f"{type(block).__name__}")
    rule = block.get("layer_rule")
    if rule not in ALPHASTEER_LAYER_RULES:
        raise ValueError(
            f"class_b.alphasteer.layer_rule must be one of "
            f"{ALPHASTEER_LAYER_RULES}, got {rule!r} — the F.9-AUROC rule is "
            "the only multi-layer selection this driver knows; a different "
            "rule is a different experiment")
    defaults = _alphasteer_rule_defaults()
    unknown = sorted(set(block) - set(defaults) - {"layer_rule"})
    if unknown:
        raise ValueError(f"class_b.alphasteer has unknown keys {unknown}")
    out = dict(defaults)
    out.update(block)
    out["layer_rule"] = rule
    if not out["screen_path"]:
        raise ValueError("class_b.alphasteer.screen_path is required: the "
                         "path scripts/alphasteer_screen.py writes screen_f9.json to")
    band = out["depth_band"]
    if (not isinstance(band, (list, tuple)) or len(band) != 2
            or not 0.0 <= float(band[0]) < float(band[1]) <= 1.0):
        raise ValueError(f"class_b.alphasteer.depth_band must be [lo, hi] "
                         f"with 0 <= lo < hi <= 1, got {band!r}")
    out["depth_band"] = [float(band[0]), float(band[1])]
    grid = out["p_grid"]
    if (not isinstance(grid, (list, tuple)) or not grid
            or any(not 0.0 < float(p) < 1.0 for p in grid)
            or len(set(float(p) for p in grid)) != len(grid)):
        raise ValueError(f"class_b.alphasteer.p_grid must be distinct ratios "
                         f"in (0, 1), got {grid!r}")
    out["p_grid"] = sorted(float(p) for p in grid)
    for key in ("auroc_threshold",):
        if not 0.5 < float(out[key]) <= 1.0:
            raise ValueError(f"class_b.alphasteer.{key} must be in (0.5, 1], got {out[key]!r}")
    if not 0.0 < float(out["heldout_frac"]) < 1.0:
        raise ValueError(f"class_b.alphasteer.heldout_frac must be in (0, 1), got {out['heldout_frac']!r}")
    if float(out["lambda_reg"]) <= 0.0:
        raise ValueError(f"class_b.alphasteer.lambda_reg must be > 0, got {out['lambda_reg']!r}")
    from sisteer.extraction.alphasteer import SITES
    from sisteer.extraction.alphasteer_screen import SERVING_DTYPES

    if out["site"] not in SITES:
        raise ValueError(f"class_b.alphasteer.site must be one of {SITES}, got {out['site']!r}")
    if out["serving_dtype"] not in SERVING_DTYPES:
        raise ValueError(f"class_b.alphasteer.serving_dtype must be one of "
                         f"{SERVING_DTYPES}, got {out['serving_dtype']!r}")
    if out["steer_positions"] != "prefill":
        raise ValueError(
            "class_b.alphasteer.steer_positions must be 'prefill': the served "
            "adapter (_adapt_alphasteer) refuses every other mode, got "
            f"{out['steer_positions']!r}")
    out["auroc_threshold"] = float(out["auroc_threshold"])
    out["heldout_frac"] = float(out["heldout_frac"])
    out["lambda_reg"] = float(out["lambda_reg"])
    out["split_seed"] = int(out["split_seed"])
    screen = Path(out["screen_path"])
    if out["acts_dir"] is None:
        out["acts_dir"] = str(screen.parent / "acts")
    if out["split_path"] is None:
        out["split_path"] = str(screen.parent / "split_v1.json")
    return out


def rule_path(rule: dict, key: str, override=None) -> Path:
    """One path of the validated ``class_b.alphasteer`` block (``screen_path``, ``acts_dir``,
    ``split_path``, ``leakage_report_path``) or its command-line ``override``, with any site
    placeholder expanded through ``sisteer.site`` (an unset one refuses by name). The block keeps the
    config's spelling, which is what fit_meta records; the expansion happens where a path is opened
    or written, so nothing reads or writes a literal ``$SHARED`` directory."""
    from sisteer import site as _site

    return _site.expand_path(override or rule[key], key=f"class_b.alphasteer.{key}")


def alphasteer_model_key(cfg: dict, rule: dict | None = None) -> str:
    """The roster key the screen/fit artifacts are filed under: the rule
    block's ``model_key`` if set, else a slug of the config's model id."""
    rule = rule or alphasteer_layer_rule(cfg) or {}
    if rule.get("model_key"):
        return str(rule["model_key"])
    return "".join(c if c.isalnum() else "_" for c in str(cfg["model"])).strip("_").lower()


# --------------------------------------------------------------------------- #
# data assembly                                                                #
# --------------------------------------------------------------------------- #


def build_class_b_data(cfg: dict, tok, *, content=None) -> dict:
    """Assemble the Class B fit input: one harmful/benign ContentContrast
    shared by both arms. ``content`` (a ContentContrast) is injectable for
    tests; absent, it is built by ``build_content_contrast`` — from the
    config-pinned verbatim prompt lists when ``class_b.benign_prompts_path``
    / ``harmful_prompts_path`` are set (production: the AlphaSteer authors'
    14,000/2,000 composition), else the legacy assembly. ``min_prompts`` is
    always in force here (see _class_b_cfg)."""
    del tok  # kept in the signature: every driver data hook takes (cfg, tok)
    cb = _class_b_cfg(cfg)
    if content is None:
        from sisteer.data.content_sets import build_content_contrast

        hb = cfg.get("datasets", {}).get("harmbench", {})
        # The config writes the lists under $SHARED; expand onto whichever
        # cluster is running (the sha256 pin check below verifies the bytes).
        benign_path = cb["benign_prompts_path"]
        harmful_path = cb["harmful_prompts_path"]
        if benign_path is not None:
            benign_path = resolve_cluster_path(benign_path)
        if harmful_path is not None:
            harmful_path = resolve_cluster_path(harmful_path)
        content = build_content_contrast(
            n_benign=cb["n_benign"], alpaca_frac=cb["alpaca_frac"],
            n_harmful=cb["n_harmful"], harmbench_split=cb["harmbench_split"],
            orbench_config=cb["orbench_config"], seed=cb["seed"],
            harmbench_path=hb.get("path"),
            benign_prompts_path=benign_path,
            harmful_prompts_path=harmful_path,
            min_prompts=cb["min_prompts"])
    _check_sha_pins(cb, getattr(content, "meta", {}) or {})

    return {
        "benign_prompts": content.benign,
        "harmful_prompts": content.harmful,
        "content_meta": getattr(content, "meta", {}),
    }


# --------------------------------------------------------------------------- #
# the fit trainers (one closure per arm; injectable for tests)                 #
# --------------------------------------------------------------------------- #


def class_b_batch_size(cb: dict, arm: str) -> int:
    """The capture batch size for ONE Class B arm: ``batch_size_by_arm[arm]``
    if present, else the block's shared ``batch_size``.

    Per-arm because the three arms capture very differently — caa_static_r
    harvests TEN layers, antidote and alphasteer one — so a single width
    either wastes the cheap arms' headroom or overruns the expensive one.
    Safe to vary: all three are CLOSED-FORM captures (pooled diff-in-means /
    PCA / ridge over last-prompt-token activations), so the batch size is a
    pure throughput knob — grouping prompts differently cannot change a mean
    — unlike the gradient arms, where batch size moves the optimisation
    trajectory. Not bitwise across widths (float reduction order), which is
    why a change here is worth recording rather than assuming invisible."""
    by_arm = cb.get("batch_size_by_arm") or {}
    return int(by_arm.get(arm, cb["batch_size"]))


def default_train_fns(cfg: dict) -> dict:
    """arm -> ``fit(model, tok, data, layer, log_fn) -> object``. Each closure
    calls the arm's SI-agnostic content trainer with the assembled data."""
    cb = _class_b_cfg(cfg)

    def _antidote(model, tok, data, *, layer, log_fn):
        from sisteer.extraction.antidote import train_antidote_content

        return train_antidote_content(
            model, tok, data["benign_prompts"], data["harmful_prompts"],
            layer=layer, batch_size=class_b_batch_size(cb, "antidote"),
            log_fn=log_fn)

    def _alphasteer(model, tok, data, *, layer, log_fn):
        rule = alphasteer_layer_rule(cfg)
        if rule is None:
            # The landed single-layer path, byte-identical (spec: existing
            # Llama Class B behaviour is unchanged; configs/two_si.yaml has
            # no class_b.alphasteer key).
            from sisteer.extraction.alphasteer import train_alphasteer_content

            return train_alphasteer_content(
                model, tok, data["benign_prompts"], data["harmful_prompts"],
                layer=layer, batch_size=class_b_batch_size(cb, "alphasteer"),
                log_fn=log_fn)
        # The exemption: the F.9 screen owns the layer set and the per-layer
        # ratios; the driver-level pin is deliberately ignored here, the
        # caa_static_r precedent (it is still resolved and recorded).
        del layer
        return fit_alphasteer_multilayer_from_screen(
            cfg, rule, model, tok, data,
            batch_size=class_b_batch_size(cb, "alphasteer"), log_fn=log_fn)

    def _caa_static_r(model, tok, data, *, layer, log_fn):
        # The CAA / diff-in-means floor (added 2026-08-21): the
        # an earlier development branch's MATCHED implementation
        # (static_r_matched), vendored verbatim from that branch @ 9c63c0d
        # in extraction/caa_static_r.py — the paper's TEN layers, resid_pre
        # (captured AND applied through the shared harvest helper),
        # prefill-only, RAW per-layer r broadcast to all prompt positions,
        # with the per-layer fitted-sign assertion. Same contrast and
        # sha-pin path as the other two arms; closed-form, no calibration.
        # The driver's single-layer pin is DELIBERATELY ignored here (the
        # lofit precedent): the ten-layer set IS the arm's identity — the
        # withdrawn single-layer/last-token form measured below the
        # unsteered baseline and is not a floor. meta.json records the
        # realised layers from the object itself.
        # DEPLOYED OPERATING POINT: the trainer's base_scale default is
        # caa_static_r.DEPLOYED_LAM = 0.05 — a FIXED operating point
        # (2026-08-21), deliberately below the suggested lam grid in that
        # branch, NOT a swept default. It is baked into the fitted object's
        # base_scale, so the pre-registered serving dose (eval alpha ==
        # config scale == 1.0) deploys lam 0.05 and the eval --alpha seam
        # re-serves the same bytes at multiples of it.
        del layer
        from sisteer.extraction.caa_static_r import train_caa_static_r_content

        return train_caa_static_r_content(
            model, tok, data["benign_prompts"], data["harmful_prompts"],
            batch_size=class_b_batch_size(cb, "caa_static_r"),
            log_fn=log_fn)

    return {"antidote": _antidote, "alphasteer": _alphasteer,
            "caa_static_r": _caa_static_r}


# --------------------------------------------------------------------------- #
# the alphasteer multi-layer fit from the F.9 screen                          #
# --------------------------------------------------------------------------- #


def _slices_path_for(list_path: str | None) -> Path | None:
    """``<list>.slices.json`` beside a pinned list (the per-source
    boundaries the content-grouped split stratifies by)."""
    if not list_path:
        return None
    p = str(list_path)
    return Path(p[:-5] + ".slices.json") if p.endswith(".json") else None


def load_list_slices(list_path: str | None, n_rows: int) -> tuple[list[dict], str]:
    """The slice table for a pinned list, resolved on this cluster like the
    list itself; a list with no ``.slices.json`` is one source ``all``
    (recorded as such)."""
    from sisteer.extraction.alphasteer_screen import whole_list_slices

    sp = _slices_path_for(list_path)
    if sp is None:
        return whole_list_slices(n_rows), "none (single source 'all')"
    try:
        resolved = resolve_cluster_path(str(sp))
    except FileNotFoundError:
        return whole_list_slices(n_rows), f"absent: {sp} (single source 'all')"
    return json.loads(Path(resolved).read_text()), resolved


def compute_content_grouped_split(cfg: dict, rule: dict, data: dict) -> dict:
    """The ``split_v1.json`` payload for this config's lists — the SAME
    function the screen used, so the fit's rows are the screen's rows."""
    from sisteer.extraction.alphasteer_screen import (
        split_side_content_grouped, split_v1_payload)

    cb = _class_b_cfg(cfg)
    b_slices, b_src = load_list_slices(cb.get("benign_prompts_path"),
                                       len(data["benign_prompts"]))
    h_slices, h_src = load_list_slices(cb.get("harmful_prompts_path"),
                                       len(data["harmful_prompts"]))
    kw = dict(heldout_frac=rule["heldout_frac"], seed=rule["split_seed"])
    benign = split_side_content_grouped(data["benign_prompts"], side="benign",
                                        slices=b_slices, **kw)
    harmful = split_side_content_grouped(data["harmful_prompts"], side="harmful",
                                         slices=h_slices, **kw)
    payload = split_v1_payload(benign, harmful)
    payload["slices_source"] = {"benign": str(b_src), "harmful": str(h_src)}
    return payload


def _check_screen_matches_config(screen: dict, cfg: dict, rule: dict) -> None:
    """The screen artifact must have been produced under THIS config's rule
    and for THIS model — a screen for another band, grid, threshold or
    model refuses rather than selecting layers the config never asked for."""
    from sisteer.extraction.alphasteer_screen import SCHEMA

    if screen.get("schema") != SCHEMA:
        raise ValueError(f"screen schema {screen.get('schema')!r} != {SCHEMA!r}")
    if screen.get("model") != cfg["model"]:
        raise ValueError(f"screen was computed for model {screen.get('model')!r}, "
                         f"config pins {cfg['model']!r}")
    rev_cfg = cfg.get("model_revision")
    if rev_cfg is not None and screen.get("model_revision") not in (None, rev_cfg):
        raise ValueError(f"screen model_revision {screen.get('model_revision')!r} "
                         f"!= config's {rev_cfg!r}")
    from sisteer.extraction.alphasteer_screen import ELIGIBILITY_RULE

    srule = screen["rule"]
    # The screen must have been SELECTED under the rule this driver fits under:
    # AUROC only, leakage a reported diagnostic. An older screen carries a
    # `leakage_max` bound and no `eligibility` line — its chosen set was
    # filtered by the retired gate — and refuses rather than being fit on.
    if srule.get("eligibility") != ELIGIBILITY_RULE or "leakage_max" in srule:
        raise ValueError(
            "screen rule.eligibility is not the AUROC-only rule this driver fits "
            f"under (got {srule.get('eligibility')!r}"
            + ("; the screen's rule still carries a leakage eligibility bound"
               if "leakage_max" in srule else "")
            + ") — the screen predates D38; refit it under this code")
    want = {
        "depth_band": [float(x) for x in rule["depth_band"]],
        "p_grid": [float(p) for p in rule["p_grid"]],
        "auroc_threshold": float(rule["auroc_threshold"]),
        "lambda_reg": float(rule["lambda_reg"]),
    }
    for k, v in want.items():
        got = srule.get(k)
        got_n = [float(x) for x in got] if isinstance(got, list) else (
            None if got is None else float(got))
        if got_n != v:
            raise ValueError(f"screen rule.{k} = {got!r} but the config's "
                             f"class_b.alphasteer.{k} = {v!r} — refit the "
                             "screen under this config or fix the config")
    if screen.get("site") != rule["site"]:
        raise ValueError(f"screen site {screen.get('site')!r} != config site "
                         f"{rule['site']!r}")


# The frozen Phase 0 artifact's keys: ``total`` is the contaminated
# goal-id list (exact ∪ near), ``n_goals`` the goal universe.
LEAKAGE_REPORT_KEYS = ("exact", "near", "total", "n_goals", "frac")


def leakage_report_block(path: str | None) -> dict | None:
    """``fit_meta.json``'s leakage block from the frozen report at ``path``
    (None when the config sets no path). Read by the report's ACTUAL schema
    (``n_clean = n_goals − len(total)``); a missing file, a missing key or a
    wrong shape RAISES — a provenance record with a silent null is how a
    writer and a reader guessing different schemas survives."""
    if not path:
        return None
    from sisteer import site as _site

    p = _site.expand_path(path, key="class_b.alphasteer.leakage_report_path")
    if not p.is_file():
        raise FileNotFoundError(
            f"class_b.alphasteer.leakage_report_path {p} does not exist")
    raw = p.read_bytes()
    rep = json.loads(raw)
    missing = [k for k in LEAKAGE_REPORT_KEYS if k not in rep]
    if missing:
        raise ValueError(
            f"{path}: leakage report lacks keys {missing} — expected the frozen "
            f"Phase 0 schema {list(LEAKAGE_REPORT_KEYS)}; refusing to record a "
            "half-populated provenance block")
    total, n_goals, frac = rep["total"], rep["n_goals"], rep["frac"]
    if (not isinstance(total, list) or not isinstance(n_goals, int)
            or isinstance(n_goals, bool) or not isinstance(frac, (int, float))
            or len(total) > n_goals):
        raise ValueError(
            f"{path}: leakage report has an unexpected shape (total: "
            f"{type(total).__name__}[{len(total) if hasattr(total, '__len__') else '?'}], "
            f"n_goals: {n_goals!r}, frac: {frac!r}); refusing")
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
            "n_goals": n_goals, "n_contaminated_goals": len(total),
            "n_clean_goals": n_goals - len(total), "frac_contaminated": float(frac)}


def fit_alphasteer_multilayer_from_screen(
    cfg: dict, rule: dict, model, tok, data: dict, *, batch_size: int,
    log_fn=None,
):
    """Fit the multi-layer alphasteer object the F.9 screen selected:
    read + fingerprint-verify ``screen_f9.json``, check it was made
    under this config's rule and model, recompute the content-grouped split
    and REFUSE unless its sha equals the screen's, then run
    ``train_alphasteer_content_multilayer`` on the FIT rows over
    ``chosen.layers`` / ``chosen.nullspace_ratios`` at the rule's site,
    prefill-only, ``lambda_reg``, UNCALIBRATED (``base_scale = 1.0`` — the
    served ladder sets the strength). The activations come through the ONE
    capture seam the screen used (``capture_or_load_side``) under the
    SCREEN'S recorded capture binding (``inputs_binding.capture[side]``),
    which the fit rebuilds from the model it loaded and the lists it read and
    holds equal BEFORE anything is loaded or captured — so the fit's matrices
    are the screen's bit for bit whether the cache was reused or recaptured,
    and a screen this fit cannot reproduce (another precision, kernel,
    revision, list or code sha) refuses instead of receiving matrices it
    never scored. The one ADOPTED key is the batch width: the screen CLI's
    ``--batch-size`` is a legitimate throughput override of the rule's
    ``capture_batch_size``, recorded by the screen, and the screen's width
    is what reproduces its bytes, so it wins over the rule's (logged when
    they differ). Fit-side rho quantiles over the held-out harmful rows (fp32
    and the rule's serving dtype) land in ``metadata["per_layer"]`` as the
    lambda-grid PROPOSAL."""
    import torch

    from sisteer.extraction import alphasteer_screen as scr
    from sisteer.extraction.alphasteer import train_alphasteer_content_multilayer
    from sisteer.models import chat_template_hash

    t0 = time.time()
    leakage_report = leakage_report_block(rule.get("leakage_report_path"))
    screen = scr.read_fingerprinted_json(rule_path(rule, "screen_path"))
    scr.check_screen_payload(screen)
    _check_screen_matches_config(screen, cfg, rule)
    chosen = screen["chosen"]
    layers = [int(l) for l in chosen["layers"]]
    if not layers:
        raise ValueError("screen chose no layers (chosen.layers is empty)")
    if 0 in layers:
        raise ValueError("screen chose layer 0 — never eligible; refusing")
    ratios = {int(l): float(chosen["nullspace_ratios"][str(l)]) for l in layers}
    grid = [float(p) for p in rule["p_grid"]]
    bad = {l: p for l, p in ratios.items() if p not in grid}
    if bad:
        raise ValueError(f"chosen ratios not in the grid {grid}: {bad}")

    # The split: recomputed here with the SAME function/seed/frac/slices and
    # pinned to the screen's sha — the fit runs on the rows the screen scored.
    split = compute_content_grouped_split(cfg, rule, data)
    if split["split_sha256"] != screen["split"]["split_sha256"]:
        raise ValueError(
            "the content-grouped split recomputed from the config's lists "
            f"({split['split_sha256'][:16]}...) differs from the screen's "
            f"({screen['split']['split_sha256'][:16]}...) — the lists, the "
            "seed, the held-out fraction or the slices changed since the "
            "screen ran; refusing to fit on rows the screen did not score")
    b_fit = split["sides"]["benign"]["fit_idx"]
    h_fit = split["sides"]["harmful"]["fit_idx"]
    h_held = split["sides"]["harmful"]["heldout_idx"]

    # The activations: the screen's acts dir through the one capture seam,
    # under the SCREEN'S capture binding. The screen records, per side,
    # every input its captures were a function of
    # (inputs_binding.capture[side]); the fit rebuilds the same binding from
    # the model it loaded and the lists it read and holds the two equal
    # BEFORE anything is loaded or captured. An existing cache under another
    # binding was already refused by the seam — but an ABSENT cache used to
    # be recaptured under the RULE's width and THIS code's sha, whatever the
    # screen had used, and the screen's AUROC/leakage were then attached to
    # matrices it never scored. Two kinds of key:
    #   ADOPTED from the screen: batch_size. `--batch-size` on the screen CLI
    #   is a legitimate throughput override of class_b.alphasteer.
    #   capture_batch_size (a pure padding-width knob, recorded), and the
    #   fit's job is to reproduce the screen's bytes, so the screen's width
    #   wins over the rule's; logged when they differ.
    #   REQUIRED equal: everything else — model, resolved revision, dtype,
    #   attention kernel, device family, site, list sha + rows, tokenization,
    #   code sha. Any difference means this fit cannot produce the screen's
    #   matrices, and it refuses (CheckpointMismatch) rather than fit
    #   different ones under the screen's numbers.
    content_meta = data.get("content_meta", {}) or {}
    site = rule["site"]
    code_sha = scr.repo_code_sha()
    identity = scr.model_load_identity(model, name=cfg["model"],
                                       cfg_revision=cfg.get("model_revision"))
    recorded = (screen.get("inputs_binding") or {}).get("capture")
    if not isinstance(recorded, dict) or set(recorded) != {"benign", "harmful"}:
        raise scr.CheckpointMismatch(
            f"{rule['screen_path']}: the screen records no capture binding "
            "(inputs_binding.capture) — it predates capture bindings, so the "
            "inputs its matrices were captured under are unknown; re-run the "
            "screen rather than fit under its numbers")
    acts_dir = Path(screen.get("acts_dir") or rule_path(rule, "acts_dir"))
    acts, sentinels, reused, widths = {}, {}, {}, {}

    def log(m):
        print(f"[fit_class_b] alphasteer: {m}")

    for side in ("benign", "harmful"):
        prompts = data[f"{side}_prompts"]
        widths[side] = int(recorded[side]["batch_size"])
        binding = scr.capture_binding(
            **identity, site=site, side=side,
            list_sha256=content_meta.get(f"{side}_sha256"), n_rows=len(prompts),
            batch_size=widths[side], code_sha=code_sha)
        if binding != recorded[side]:
            raise scr.CheckpointMismatch(
                f"the screen's {side} matrices were captured under different "
                "inputs than this fit would capture under — "
                f"{scr.binding_diff(binding, recorded[side])}. Refusing to fit "
                "matrices the screen never scored under its AUROC rule; "
                "re-run the screen under these inputs, or fit under the screen's.")
        if widths[side] != int(rule["capture_batch_size"]):
            log(f"{side}: capturing at the SCREEN's batch width {widths[side]} "
                f"(class_b.alphasteer.capture_batch_size is "
                f"{rule['capture_batch_size']}; the screen's recorded width "
                "wins -- its bytes are the target)")
        reused[side] = scr.read_capture_sentinel(acts_dir / side) is not None
        acts[side], sentinels[side] = scr.capture_or_load_side(
            model, tok, prompts, side=side, acts_dir=acts_dir, binding=binding,
            verify_sample=int(rule["verify_sample"]), layers=layers, log_fn=log)
    bi, hi, hh = (torch.tensor(ix, dtype=torch.long) for ix in (b_fit, h_fit, h_held))

    obj = train_alphasteer_content_multilayer(
        model, tok, [data["benign_prompts"][i] for i in b_fit],
        [data["harmful_prompts"][i] for i in h_fit],
        layers=layers, nullspace_ratios=ratios,
        steer_positions=rule["steer_positions"], site=site,
        lambda_reg=rule["lambda_reg"], batch_size=batch_size, log_fn=log_fn,
        calibrate=False,
        precaptured=({l: acts["benign"][l][bi] for l in layers},
                     {l: acts["harmful"][l][hi] for l in layers}))
    # the capture record is the seam's, not the trainer's "precaptured" note
    obj.metadata["capture"] = {
        "source": "capture_or_load_side", "acts_dir": str(acts_dir),
        "reused": reused, "benign": sentinels["benign"],
        "harmful": sentinels["harmful"],
        "add_special_tokens": scr.CAPTURE_ADD_SPECIAL_TOKENS,
        "ids_match_serving_render": True}
    rho = scr.rho_by_layer({l: acts["harmful"][l][hh] for l in layers},
                           {l: obj.Ms[l] for l in layers}, scale=1.0,
                           model_dtype=getattr(torch, rule["serving_dtype"]))
    screen_by_layer = {int(pl["layer"]): pl for pl in screen["per_layer"]}
    for l in layers:
        obj.metadata["per_layer"][l].update({
            "auroc": screen_by_layer[l]["auroc"],
            "leakage": screen_by_layer[l]["leakage"],
            "p_star": screen_by_layer[l]["p_star"],
            **rho[l],
        })
    obj.metadata["rho_note"] = scr.RHO_NOTE
    obj.metadata["f9"] = {
        "layer_rule": rule["layer_rule"],
        "screen_path": str(rule["screen_path"]),
        "screen_f9_sha256": screen["sha256"],
        "split_sha256": split["split_sha256"],
        "split_convention": split["convention"],
        "acts_source": {"acts_dir": str(acts_dir), "reused": reused,
                        "capture_batch_size": widths,
                        "capture_batch_size_source": "screen.inputs_binding.capture",
                        "rule_capture_batch_size": int(rule["capture_batch_size"])},
        "realised_counts": {
            "benign": len(data["benign_prompts"]),
            "harmful": len(data["harmful_prompts"]),
            "benign_fit": len(b_fit), "harmful_fit": len(h_fit),
            "benign_heldout": split["n_heldout_benign"],
            "harmful_heldout": len(h_held),
            "realised_by_source": split["realised_by_source"],
            "note": "REALISED — content-grouped split, see split_v1.json",
        },
        "list_sha256": {side: content_meta.get(f"{side}_sha256")
                        for side in ("benign", "harmful")},
        "eligible_counts": screen["eligible_counts"],
        "chosen": chosen,
        "capture_screen": screen.get("capture"),
        "leakage_report": leakage_report,
        "fit_seconds": time.time() - t0,
    }
    obj.metadata["provenance"] = {
        "resolved_model_id": cfg["model"],
        "resolved_revision": (getattr(model.config, "_commit_hash", None)
                              or cfg.get("model_revision")),
        "chat_template_hash": chat_template_hash(tok),
        "writer": f"fit_class_b.py@{scr.repo_code_sha()}",
        "positions_at_fit": rule["steer_positions"],
    }
    return obj


def _fit_meta_payload(cfg: dict, obj, data: dict, layer: int, *, object_path: Path) -> dict:
    """``fit_meta.json`` from the fitted object's metadata, PLUS
    the one field that binds the record to the object it describes:
    ``object_sha256``, the sha256 of the object file ON DISK. It is
    hashed here, after ``obj.save`` and before the record is written, so it
    names the bytes that actually landed — never a pre-save in-memory copy.
    Every other sha the record carries (leakage report, screen, split, the
    lists) pins an INPUT; without this one nothing ties the record to the
    object, and a stale object/record pair is undetectable. ``sha256`` (the
    record's own fingerprint, ``write_fingerprinted_json``) is a different
    quantity and never equals it; the bake refuses a record without
    ``object_sha256`` rather than inferring the tie from content."""
    object_path = Path(object_path)
    if not object_path.is_file():
        raise ValueError(f"fit_meta.json would describe {object_path}, which is not on disk — "
                         "the object is hashed AFTER it lands, never before")
    md = obj.metadata
    f9 = md.get("f9", {})
    prov = md.get("provenance", {})
    lr = f9.get("leakage_report")
    # configured -> every leakage field, from the report's actual schema
    # (leakage_report_block); not configured -> the path alone, null, and
    # NO count keys — a null count would read as a measured nothing
    leakage = ({"leakage_report_path": lr["path"],
                "leakage_report_sha256": lr["sha256"],
                "n_goals": lr["n_goals"],
                "n_clean_goals": lr["n_clean_goals"],
                "n_contaminated_goals": lr["n_contaminated_goals"]}
               if lr else {"leakage_report_path": None})
    return {
        "schema": "alphasteer_fit_meta/1",
        "cell_id": "alphasteer", "class": "B", "arm": "alphasteer",
        "fit_mode": md.get("fit_mode"),
        "driver_layer_pin_ignored": layer,
        "layers": [int(l) for l in md.get("layers", [])],
        "nullspace_ratios": {str(k): v for k, v in md.get("nullspace_ratios", {}).items()},
        "site": md.get("site"), "steer_positions": md.get("steer_positions"),
        "lambda_reg": md.get("lambda_reg"), "ridge_solver": md.get("ridge_solver"),
        "base_scale": obj.base_scale,
        "base_scale_source": md.get("base_scale_source"),
        "lambda_ladder_sha256": None,      # set by the ladder when it bakes the strength (plan C)
        "screen_f9_sha256": f9.get("screen_f9_sha256"),
        "split_sha256": f9.get("split_sha256"),
        "split_convention": f9.get("split_convention"),
        "realised_counts": f9.get("realised_counts"),
        "list_sha256": f9.get("list_sha256"),
        "sample_floor": md.get("sample_floor"),
        "per_layer": {str(k): v for k, v in md.get("per_layer", {}).items()},
        "rho_note": md.get("rho_note"),
        "capture": md.get("capture"),
        "capture_screen": f9.get("capture_screen"),
        "eligible_counts": f9.get("eligible_counts"),
        "chosen": f9.get("chosen"),
        "acts_source": f9.get("acts_source"),
        **leakage,
        "model_id": prov.get("resolved_model_id"),
        "model_revision": prov.get("resolved_revision"),
        "chat_template_hash": prov.get("chat_template_hash"),
        "n_params": obj.n_params(),
        "fit_seconds": f9.get("fit_seconds"),
        # the object<->record binding: the bytes on disk, hashed after the save
        "object_path": object_path.name,
        "object_sha256": _sha256_file(object_path),
        "object_sha256_source": OBJECT_SHA256_PRODUCER_SOURCE,
    }


def _write_alphasteer_sidecars(cfg: dict, arm: str, obj, data: dict,
                               layer: int, out_dir: Path, *, object_path: Path) -> list[Path]:
    """``fit_meta.json`` + ``fit_provenance.json`` beside the object — for the
    alphasteer exemption path ONLY (config-keyed like the exemption itself;
    every other arm's directory is unchanged). Both are written BEFORE
    ``meta.json`` so the sentinel still means "everything is here", and
    AFTER the object (``object_path``, already on disk) so ``fit_meta.json``
    can record the hash of the bytes that landed."""
    from sisteer.extraction.alphasteer_screen import write_fingerprinted_json

    written = []
    if arm != "alphasteer" or alphasteer_layer_rule(cfg) is None:
        return written
    if obj.metadata.get("fit_mode") != "content_si_agnostic_multilayer":
        raise ValueError("alphasteer exemption active but the fitted object is "
                         f"not multi-layer (fit_mode={obj.metadata.get('fit_mode')!r})")
    fm = out_dir / "fit_meta.json"
    write_fingerprinted_json(fm, _fit_meta_payload(cfg, obj, data, layer, object_path=object_path))
    written.append(fm)
    prov = dict(obj.metadata.get("provenance") or {})
    missing = [k for k in ("resolved_model_id", "resolved_revision",
                           "chat_template_hash", "writer", "positions_at_fit")
               if not prov.get(k)]
    if missing:
        raise ValueError(f"fit_provenance.json would carry null identity keys "
                         f"{missing} — refusing to write an identity sidecar "
                         "that identifies nothing")
    fp = out_dir / "fit_provenance.json"
    fp.write_text(json.dumps(prov, indent=1, sort_keys=True))
    written.append(fp)
    return written


# --------------------------------------------------------------------------- #
# fit + save (each object saved the instant it completes)                      #
# --------------------------------------------------------------------------- #


def _object_path(cfg: dict, arm: str) -> Path:
    from sisteer.eval.defense_cell import object_path_for, resolve_defenses_dir

    # the ONE shared resolution (absolute/flagged > TT_TWO_SI_DEFENSES_DIR
    # > repo-rooted relative default), so this writer can never split from the
    # eval readers across per-commit checkouts.
    return object_path_for(resolve_defenses_dir(cfg, repo_root=REPO),
                           arm)  # bare arm id == Class B cell id


def _meta_path(cfg: dict, arm: str) -> Path:
    return _object_path(cfg, arm).parent / "meta.json"


def _save_object(cfg: dict, arm: str, obj, data: dict, layer: int) -> Path:
    """Write the object then a meta.json COMPLETION SENTINEL (last). The object
    is saved directly (not via a .tmp rename): the SteeringVector container
    (antidote) writes a sidecar keyed on the object's OWN filename, which a
    .tmp suffix would misplace (object.safetensors.json instead of object.json).
    meta.json is written strictly after the object and is the resume sentinel —
    a torn object write leaves no meta, so the arm re-fits rather than being
    skipped as done."""
    path = _object_path(cfg, arm)
    path.parent.mkdir(parents=True, exist_ok=True)
    obj.save(path)
    meta = {
        "cell_id": arm, "class": "B", "si_id": None, "arm": arm,
        "fit": "si_agnostic_content_fit_once",
        # The comparability pin (2026-08-20): single layer, 15 in production.
        # (caa_static_r deliberately ignores it — its realised ten-layer set
        # is recorded under "layers" below, from the object itself.)
        "layer": layer,
        "n_benign": len(data["benign_prompts"]),
        "n_harmful": len(data["harmful_prompts"]),
        "content_meta": data.get("content_meta", {}),
        "n_params": obj.n_params(),
    }
    obj_layers = getattr(obj, "layers", None)
    if obj_layers is not None:
        # Multi-layer objects record the layers they ACTUALLY fit — the
        # driver-level pin above is not the whole truth for them.
        meta["layers"] = [int(l) for l in obj_layers]
    # The alphasteer exemption's fit_meta.json / fit_provenance.json go
    # BEFORE meta.json, which stays the sentinel written LAST. A
    # no-op for every other arm and for alphasteer without the config key.
    sidecars = _write_alphasteer_sidecars(cfg, arm, obj, data, layer, path.parent, object_path=path)
    if sidecars:
        meta["sidecars"] = [p.name for p in sidecars]
    _meta_path(cfg, arm).write_text(json.dumps(meta, indent=1, sort_keys=True))
    _registry_object(cfg, arm, path.parent)
    return path


def _registry_object(cfg: dict, arm: str, obj_dir: Path) -> None:
    """Results-registry native emission (additive; a no-op when emission is off): the Class B
    object's ``object`` + ``artifact`` records after every sidecar and the ``meta.json`` sentinel
    are on disk. The writer states what it fitted — a contrastive direction (diffmean /
    CAA / Arditi / AlphaSteer) from the benign-vs-harmful prompt sets; an arm the registry's
    table does not name is not recorded, never guessed."""
    try:
        from sisteer.registry import hooks as _rh
        kind = _rh.object_kind_for_legacy_kind(arm)
        if kind is None:
            _rh.notice(f"fit_class_b: arm {arm!r} has no declared object kind; object not recorded")
            return
        objective = {"alphasteer": "alphasteer", "static_r_arditi": "arditi_search"}.get(kind, "diffmean")
        _rh.emit_object_from_sidecar(
            str(obj_dir), campaign=str(cfg.get("registry_campaign") or os.environ.get("REGISTRY_CAMPAIGN", "class_b")),
            object_kind=kind, paper_label=_rh.paper_label_for(kind, "safety"), object_role="safety",
            fit_target_kind="contrastive_direction", fit_objective=objective, fit_thinking_render="off",
            status="reference", script="scripts/fit_class_b.py")
    except Exception as e:  # noqa: BLE001 — a lost record never costs a fit
        print(f"[registry] fit_class_b: object not recorded ({type(e).__name__}: {e})", flush=True)


def fit_and_save(
    cfg: dict,
    model,
    tok,
    data: dict,
    *,
    arms=CLASS_B_ARMS,
    train_fns=None,
    log_fn=None,
    force: bool = False,
) -> dict:
    """Fit each Class B arm and SAVE IT THE INSTANT IT COMPLETES. Each arm is
    isolated: a refusal/error is recorded and the run carries on (a late arm can
    never discard an earlier finished one — the 148862/148863 failure mode).
    Resumes by skipping arms whose object already exists (``force`` re-fits)."""
    train_fns = train_fns or default_train_fns(cfg)
    layer = class_b_layer(cfg)
    status: dict[str, dict] = {}
    for arm in arms:
        path = _object_path(cfg, arm)
        if path is None:
            status[arm] = {"state": "error", "reason": "no object path (not a Class B arm)"}
            continue
        # Resume on the meta.json sentinel (written strictly AFTER the object),
        # never the object file alone — a torn object write has no meta, so it
        # re-fits instead of being trusted as complete.
        if _meta_path(cfg, arm).exists() and not force:
            status[arm] = {"state": "skipped_exists", "path": str(path)}
            print(f"[fit_class_b] {arm}: complete (meta present), skipping -> {path}")
            continue
        try:
            obj = train_fns[arm](model, tok, data, layer=layer, log_fn=log_fn)
            saved = _save_object(cfg, arm, obj, data, layer)
            status[arm] = {"state": "fit", "path": str(saved),
                           "n_params": obj.n_params()}
            print(f"[fit_class_b] {arm}: fit + saved -> {saved} "
                  f"(n_params={obj.n_params()})")
        except Exception as e:  # noqa: BLE001 — isolate the arm, never fatal
            status[arm] = {"state": "refused", "reason": f"{type(e).__name__}: {e}"}
            print(f"[fit_class_b] {arm}: REFUSED — {status[arm]['reason']}")
    return status


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def load_class_b_model(cfg: dict):
    """The ONE model load both drivers make (this fit and
    scripts/alphasteer_screen.py), so the screen and the fit cannot load the
    same config two different ways. Every keyword this passes is a capture
    input: ``alphasteer_screen.CAPTURE_INPUTS`` must name it, and
    ``test_capture_binding_names_every_model_load_and_capture_input`` records
    this call to hold the two together — add a keyword here and that test
    fails until the binding carries it or exempts it with a reason."""
    from sisteer.models import load_model_and_tokenizer

    return load_model_and_tokenizer(
        cfg["model"], revision=cfg.get("model_revision"), dtype=cfg["dtype"],
        attn_implementation=cfg.get("attn_implementation", "sdpa"))


def cmd_fit(args, cfg) -> None:
    model, tok = load_class_b_model(cfg)

    data = build_class_b_data(cfg, tok)
    rule = alphasteer_layer_rule(cfg)
    print(f"[fit_class_b] data: {len(data['benign_prompts'])} benign, "
          f"{len(data['harmful_prompts'])} harmful "
          f"(layer pin: {class_b_layer(cfg)}; alphasteer layer rule: "
          f"{rule['layer_rule'] + ' <- ' + str(rule['screen_path']) if rule else 'none (single-layer pin)'})")
    arms = args.arms.split(",") if args.arms else default_class_b_arms(cfg)
    status = fit_and_save(cfg, model, tok, data, arms=arms, force=args.force)
    print("[fit_class_b] summary: " + json.dumps(status, indent=1))
    if any(s["state"] == "refused" for s in status.values()):
        raise SystemExit("[fit_class_b] one or more arms refused — see summary")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/two_si.yaml")
    ap.add_argument("--arms", default="", help="comma list; default = the "
                    "config's class_b_arms (falling back to all known Class B)")
    ap.add_argument("--out", default="", help=(
        "ABSOLUTE defenses_dir override (D58): the submission-time, "
        "commit-independent objects dir under the private remote root — the "
        "overnight orchestrator always passes it. Without it the config's "
        "defenses_dir resolves via TT_TWO_SI_DEFENSES_DIR (cluster) or the "
        "repo root (local)"))
    ap.add_argument("--force", action="store_true", help="re-fit existing objects")
    ap.add_argument("--require-alphasteer-rule", action="store_true", help=(
        "refuse (exit 2, before any model load) unless the config carries a "
        "class_b.alphasteer block. The multi-layer exemption is opt-in by "
        "config key, so with the key ABSENT the alphasteer arm silently fits "
        "the legacy single-layer object at the inherited pin and exits 0 — "
        "indistinguishable from success. Every alphasteer-full22 Phase B "
        "submission passes this flag (decisions D39); the shipped Llama "
        "config, which is meant to take the legacy path, never does."))
    args = ap.parse_args(argv)
    cfg = load_cfg(args.config)
    if args.require_alphasteer_rule and alphasteer_layer_rule(cfg) is None:
        print(f"[fit_class_b] REFUSING: --require-alphasteer-rule but {args.config} has "
              "no class_b.alphasteer block — the alphasteer arm would silently fit the "
              f"legacy single-layer object at layer {class_b_layer(cfg)} under the "
              "multi-layer arm's name (D39). Add the block or drop the flag.",
              file=sys.stderr, flush=True)
        raise SystemExit(2)
    if args.out:
        cfg["defenses_dir"] = args.out   # the flag wins over env + config
    cmd_fit(args, cfg)


if __name__ == "__main__":
    main()
