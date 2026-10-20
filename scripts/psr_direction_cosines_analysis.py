#!/usr/bin/env python3
"""Cross-layer cosine similarity of fitted PSR steering directions.

CPU-only, read-only analysis of the scale-12 response-leg ``psr_kl_all``
bank (plus the promoted Qwen3.5 winners and the gemma4e2b mirror object).
Per model, layer x layer cosine matrices between the fitted steering
directions of the three SIs (safety_refusal / si_constitution /
si_irrelevant), the gate_w appendix blocks, and the gauge-invariant
rank-one-Jacobian product blocks, plus the derived headline contrasts
dL = CI - (CS + SI)/2 and dS = CS - SI.

Run (no venv of its own -- use the repository's venv):

    PYTHONPATH=src .venv/bin/python \
      scripts/psr_direction_cosines_analysis.py [--self-check] ...

Key discipline (authoritative):

* The fit leg is read from ``blob["positions"]`` (the raw
  artifact) and assert-matched against ``obj.fit_positions`` after the
  canonical hydration. The banned leg attribute (the unrelated serve-time
  attr that shadows the blob key's name and reads ``'all'`` on every
  response-leg object) is never accessed; a test greps this module for it.
* SI metadata resolves through :func:`si_meta`, never the registry's
  ``resolve_si_id`` (which raises on ``safety_refusal``).
* Sidecar flavour is keyed off which files exist, never the bank
  root; the ``COMPLETE`` marker is an observation, never a validity gate.
* Raw ``||d||``, ``||gate_w||`` and ``gate_b`` are gauge-dependent
  ((w, b, d) -> (cw, cb, d/c) leaves the intervention exactly unchanged);
  cross-object comparisons use ``||d||*||w||``, ``b/||w||`` and the
  factorised Jacobian cosine cos(d_A, d_B) * cos(w_A, w_B).
* An empirical null (shuffled layer correspondences, per-entry
  region off-diagonals, distance-matched off-diagonals at TRUE layer
  distance, circular shifts) is recorded per block AND per derived
  contrast, for the all-layer aggregate and each fractional-depth region,
  each with the observed diagonal mean's percentile. The dL/dS contrast
  nulls apply ONE shared layer mapping to all three pair blocks (the
  contrasts are linear in the blocks, so permuting/shifting the combined
  contrast matrix is identical to applying the same mapping to CI, CS and
  SI simultaneously). The isotropic scale (1/sqrt(d) for direction and
  gate-vector cosine blocks, 1/d for the rank-one Jacobian PRODUCT blocks
  -- the product of two independent cosines) is a theoretical reference
  only and every sigma-score is DESCRIPTIVE.
* Depth regions are FRACTIONAL depth on ORIGINAL layer indices
  (early: l/(L-1) < 0.25; late: l/(L-1) >= 0.5), never positional
  quarters of a (possibly masked-compressed) layer list; masked layers
  are excluded pairwise, never compressed out first.
* Finiteness -- every directions/gate_w/gate_b tensor is asserted finite
  at load; NaN/Inf is a hard error naming object, tensor and layer (a
  non-finite direction otherwise sails through: its norm goes Inf/NaN,
  no dead-layer flag fires, and the diagonal quietly turns None).
* SI lengths are reported in TOKENS under each model's
  recorded tokenizer id + pinned revision, loaded with
  ``local_files_only=True``; a tokenizer that will not load offline is a
  hard error naming the model (never a silent word-count fallback).
* Load discipline -- each object's bytes are read ONCE:
  the recorded sha256 and the tensors come from those same bytes, parsed
  with ``weights_only=True`` (no pickle execution before validation). The
  default manifest is FROZEN: exactly 10 pinned (model_key, si, arm)
  cells with sha256 pins; missing/extra/duplicate cells, a pin mismatch,
  a same-model identity mismatch (HF id / revision / d_model / layer
  list), a contradicted identity-bearing sidecar alias, or a required
  object on the wrong fit leg all exit non-zero. ``--allow-partial`` is
  the test-only escape for synthetic manifests.

Determinism: float64 numpy matmul under one BLAS is run-to-run
deterministic on this box; outputs carry no timestamps. If determinism
ever flakes, pin OMP_NUM_THREADS=1.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import sys
import types
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

import sisteer
from sisteer.data import instructions as _instructions
from sisteer.extraction.psr import (FROZEN_LAYER_FLOOR, PSRObject,
                                    psr_frozen_gate_b)

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

_REPO_SRC = Path(__file__).resolve().parent.parent / "src"

EXPECTED_LEG = "response"
ARM = "psr_kl_all"
SI_ORDER = ["safety_refusal", "si_constitution", "si_irrelevant"]

# Sidecar model fields are full HF ids, never bank keys. This map is
# asserted, never guessed; identity comes FROM the sidecar and the map
# is the cross-check that must agree.
MODEL_KEY_TO_HF = {
    "llama8b": "NousResearch/Meta-Llama-3.1-8B-Instruct",
    "gemma4e4b": "google/gemma-4-E4B-it",
    "llama3b": "unsloth/Llama-3.2-3B-Instruct",
    "qwen35_9b": "Qwen/Qwen3.5-9B",
    "qwen35_4b": "Qwen/Qwen3.5-4B",
    "gemma4e2b": "google/gemma-4-E2B-it",
}

_SCALE12 = "$GRADE/objects/scale-12"   # the grade root's placeholder form; load_entry expands it through sisteer.site

DEFAULT_OBJECTS = (
    [{"model_key": "llama8b", "si": si, "arm": ARM, "expect_leg": EXPECTED_LEG,
      "path": f"{_SCALE12}/llama8b/{si}__{ARM}/object.pt"} for si in SI_ORDER]
    + [{"model_key": "gemma4e4b", "si": si, "arm": ARM,
        "expect_leg": EXPECTED_LEG,
        "path": f"{_SCALE12}/gemma4e4b/{si}__{ARM}/object.pt"}
       for si in SI_ORDER]
    + [{"model_key": "llama3b", "si": "safety_refusal", "arm": ARM,
        "expect_leg": EXPECTED_LEG,
        "path": f"{_SCALE12}/llama3b/safety_refusal__{ARM}/object.pt"},
       {"model_key": "qwen35_9b", "si": "safety_refusal", "arm": ARM,
        "expect_leg": EXPECTED_LEG,
        "path": "$STAGE/qwen35_promoted/9b/object.pt"},
       {"model_key": "qwen35_4b", "si": "safety_refusal", "arm": ARM,
        "expect_leg": EXPECTED_LEG,
        "path": "$STAGE/qwen35_promoted/4b/object.pt"},
       {"model_key": "gemma4e2b", "si": "safety_refusal", "arm": ARM,
        "expect_leg": EXPECTED_LEG,
        "path": f"$STAGE/fits/mirror/gemma4e2b/"
                f"safety_refusal__{ARM}/object.pt"}]
)

# Frozen sha256 pins for the 10-object default manifest (review fix: the
# default run must not be able to exit 0 with the wrong object set).
# Measured from the real banks 2026-08-29; a refit changes the bytes and
# trips this DELIBERATELY -- re-pin only on a deliberate promotion.
_DEFAULT_SHA256 = {
    ("llama8b", "safety_refusal"):
        "7afa6c9f7fecf6171deea92fb23daee968da8d3d65e7ad6bc3ba15848dc69d38",
    ("llama8b", "si_constitution"):
        "9fecde7e5a9dd3c1d4fa684435403e5b30569e3c2c5c85b84c4164827c0492d3",
    ("llama8b", "si_irrelevant"):
        "cca94b1e5a6c085865bfe14251c2947da27abaaa0d8773f8d965299606c55d6c",
    ("gemma4e4b", "safety_refusal"):
        "6abb92c84237e05a2107f1d318abdd728181fb01d6c513d23f5f9bb118e7be29",
    ("gemma4e4b", "si_constitution"):
        "c4de7681269335f7653a8694d69803c07cb047cb864981b0b1425275ceb37495",
    ("gemma4e4b", "si_irrelevant"):
        "d13a956f7d9749b25070aedb319309beb60ec3a5516f071cf96b8a47cc3dfbb3",
    ("llama3b", "safety_refusal"):
        "901509d4ce8a3debfa960e3e37c5b6752759f2dfc586c7b29ec257774d6bd2b2",
    ("qwen35_9b", "safety_refusal"):
        "015f0ded1e21337cc1b7671435b0dd0935f70f2a871dc334a8f7d41c712a0879",
    ("qwen35_4b", "safety_refusal"):
        "75625f59809498899db9efcd2b19e9ce8c2f95593e7c7c1a109a03e29fe5b5c8",
    ("gemma4e2b", "safety_refusal"):
        "7b3ef39db8e63b93ba91228fa59156f4351128809ba269ad39bea4305666793a",
}
for _e in DEFAULT_OBJECTS:
    _e["sha256"] = _DEFAULT_SHA256[(_e["model_key"], _e["si"])]
del _e

# Empirical-null machinery -- fixed seed for determinism.
# Distance-matched offsets are TRUE layer-number distances, 1..5.
NULL_SEED = 12345
NULL_PERMS = 1000
NULL_OFFSETS = (1, 2, 3, 4, 5)

# --------------------------------------------------------------------------
# Regression gate: two-path-verified llama8b values. The
# implementation must reproduce this table exactly -- a deviation is a bug,
# not a finding. Pair keys are canonicalised to SI_ORDER order.
# --------------------------------------------------------------------------

REGRESSION_MODEL = "llama8b"
REGRESSION_TOL = 5e-5  # exact at the reported 4-decimal precision

# The published derived contrasts were computed from the
# 4-decimal-ROUNDED pairwise table, so their compounded rounding error
# against a full-precision computation is bounded by 5e-5 * (1 + 1/2 + 1/2)
# for dL and 5e-5 * 2 for dS -- i.e. 1e-4. Full-precision values measured
# here: dL mean +0.0745477, dS mean +0.0294239, dL L0 +0.1892073,
# dS L0 +0.1076984.
REGRESSION_TOL_DERIVED = 1.01e-4

REGRESSION_PAIRS = {
    ("si_constitution", "si_irrelevant"): {
        "l0": 0.4220, "mean": 0.1530, "late_mean": 0.0198,
        "max": 0.5241, "argmax_layer": 11},
    ("safety_refusal", "si_constitution"): {
        "l0": 0.2866, "mean": 0.0932, "late_mean": 0.0297,
        "max": 0.3015, "argmax_layer": 2},
    ("safety_refusal", "si_irrelevant"): {
        "l0": 0.1789, "mean": 0.0637, "late_mean": 0.0433,
        "max": 0.3458, "argmax_layer": 28},
}
REGRESSION_MIN_NORMS = {
    "safety_refusal": 0.2519,
    "si_constitution": 0.3934,
    "si_irrelevant": 0.4352,
}
REGRESSION_CONTRASTS = {
    "dL_mean": 0.0746, "dS_mean": 0.0295,
    "dL_l0": 0.1893, "dS_l0": 0.1077,
}

# Pre-registered two-sign check protocol, frozen before gemma4e4b was
# computed. Recorded in the outputs so the artifact carries its own
# commitment. The outcome is reported as meeting (or not meeting) the
# preregistered two-sign check, never as an unqualified replication --
# the criterion is the signs of two artifact-level means on ONE held-out
# model, so meeting it identifies no mechanism.
TWO_SIGN_CHECK_PROTOCOL = (
    "Frozen before gemma4e4b was computed (D14f): primary contrasts are "
    "dL = CI - (CS+SI)/2 and dS = CS - SI on the same-layer diagonal; "
    "aggregation is the mean over live layers of the same-layer diagonal; "
    "depth regions are first-quarter and second-half by fractional depth; "
    "the check is met iff the SIGN of both contrasts reproduces on the "
    "held-out model (magnitude reported, not required to match). llama8b "
    "is exploratory (unblinded during planning); gemma4e4b is the held-out "
    "confirmatory model; the four safety-only models are descriptive "
    "geometry only. Scope (D20d): the criterion is the signs of two "
    "artifact-level means on one held-out model -- one fit per cell, no "
    "seed replication, no effect-size floor, no control for the length or "
    "fit-quality confounds -- so a met check identifies no mechanism."
)


class IdentityError(SystemExit):
    """Fail-loud identity/validation failure: exits non-zero, names the
    disagreeing field and both values."""


def assert_import_provenance() -> None:
    """Guard against PYTHONPATH shadowing drift: sisteer must resolve
    inside THIS repo checkout's src (the worktree when run from one)."""
    got = Path(sisteer.__file__).resolve()
    if _REPO_SRC not in got.parents:
        raise IdentityError(
            f"sisteer imported from {got}, expected under {_REPO_SRC} -- "
            "PYTHONPATH shadowing drift; refusing to analyse against the "
            "wrong code")


# --------------------------------------------------------------------------
# Stub loading seam (proven byte-identical to the raw blob):
# PSRObject.load_blob needs exactly config.hidden_size and a decoder
# ModuleList; tok is unused (verified). psr.py is not touched.
# --------------------------------------------------------------------------

class _StubInner(nn.Module):
    def __init__(self, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(nn.Identity() for _ in range(n_layers))


class _StubModel(nn.Module):
    def __init__(self, d_model: int, n_layers: int):
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=d_model)
        self.model = _StubInner(n_layers)


# --------------------------------------------------------------------------
# SI metadata: BY_ID first, then module-level Instruction attributes
# (safety_refusal lives at instructions.SAFETY_SI, deliberately absent from
# the registry). Loud failure on a genuinely unknown id; never a silent
# placeholder. resolve_si_id is never called -- it raises on the single
# most important SI in the study.
# --------------------------------------------------------------------------

def si_meta(si_id: str) -> dict:
    inst = _instructions.BY_ID.get(si_id)
    source = "BY_ID"
    if inst is None:
        for name in sorted(dir(_instructions)):
            val = getattr(_instructions, name)
            if isinstance(val, _instructions.Instruction) and val.id == si_id:
                inst, source = val, "module_attribute"
                break
    if inst is None:
        raise IdentityError(
            f"unknown SI id {si_id!r}: not in instructions.BY_ID and no "
            "module-level Instruction carries that id -- refusing to "
            "substitute a placeholder")
    return {"text": inst.text, "words": len(inst.text.split()),
            "chars": len(inst.text), "registry_source": source}


def _f(x):
    """JSON-safe float: finite -> float, non-finite/None -> None."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _tilde(path: Path) -> str:
    try:
        return "~/" + str(path.relative_to(Path.home()))
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------
# Loading + identity resolution
# --------------------------------------------------------------------------

class LoadedObject:
    """One hydrated object: JSON-ready identity record + numpy tensors."""

    def __init__(self, record: dict, D: np.ndarray, W: np.ndarray,
                 B: np.ndarray, layers: list[int],
                 live_dir: np.ndarray, live_gw: np.ndarray):
        self.record = record
        self.D = D          # [L, d] float64 directions
        self.W = W          # [L, d] float64 gate_w
        self.B = B          # [L] float64 gate_b
        self.layers = layers
        self.live_dir = live_dir  # direction-based live mask (dead+near-dead)
        self.live_gw = live_gw    # gate_w-based live mask (same rules)
        self.model_key = record["model_key"]
        self.si = record["si"]
        self.sha256 = record["sha256"]
        self.d_model = record["d_model"]
        self.leg = record["leg"]


def _read_sidecars(obj_dir: Path) -> tuple[str, dict, dict | None, list[str]]:
    """Detect the sidecar flavour by WHICH FILES EXIST (never the root):
    provenance.json -> flavour B authoritative; else
    train_budget_meta.json -> flavour A; both -> B authoritative with A
    cross-checked; neither -> hard error. Returns
    (flavour, authoritative_json, secondary_json_or_None, files_present)."""
    files_present = sorted(p.name for p in obj_dir.iterdir() if p.is_file())
    prov_p = obj_dir / "provenance.json"
    tbm_p = obj_dir / "train_budget_meta.json"
    prov = json.loads(prov_p.read_text()) if prov_p.is_file() else None
    tbm = json.loads(tbm_p.read_text()) if tbm_p.is_file() else None
    if prov is None and tbm is None:
        raise IdentityError(
            f"{obj_dir}: neither provenance.json nor train_budget_meta.json "
            "present -- object identity cannot be established from the "
            "artifact, refusing to guess from the path")
    if prov is not None:
        return "provenance", prov, tbm, files_present
    return "train_budget_meta", tbm, None, files_present


def _sidecar_identity(flavour: str, side: dict, secondary: dict | None,
                      where: str) -> dict:
    """Normalise the two sidecar flavours to one identity dict. The model
    id key name DIFFERS by flavour: ``model_id`` in flavour A
    (train_budget_meta), ``model`` in flavour B (provenance)."""
    if flavour == "provenance":
        ident = {
            "si": side.get("si"),
            "model_id": side.get("model"),
            "model_revision": side.get("model_revision"),
            "arm": side.get("arm"),
            "leg": side.get("positions"),
            "response_header_len": side.get("response_header_len"),
            "fit_fingerprint": side.get("fit_fingerprint"),
            "target_layers": None,
            "fit_stats": {k: side.get(k) for k in
                          ("floor_ratio", "final_residual_ratio")
                          if k in side},
            "hpo": {k: side.get(k) for k in
                    ("lr", "clip", "effective_batch", "epochs") if k in side},
            "selection": side.get("selection"),
            "source": side.get("source"),
        }
        if secondary is not None:
            # Both sidecars present: B is authoritative, A's overlapping
            # fields must agree.
            for a_key, b_val, label in (
                    ("si_id", ident["si"], "si"),
                    ("model_id", ident["model_id"], "model id"),
                    ("model_revision", ident["model_revision"],
                     "model_revision"),
                    ("arm", ident["arm"], "arm"),
                    ("positions", ident["leg"], "fit leg"),
                    ("response_header_len", ident["response_header_len"],
                     "response_header_len")):
                if a_key in secondary and secondary[a_key] != b_val:
                    raise IdentityError(
                        f"{where}: sidecar flavours disagree on {label}: "
                        f"train_budget_meta.{a_key}={secondary[a_key]!r} "
                        f"vs provenance={b_val!r}")
            ident["target_layers"] = secondary.get("target_layers")
        return ident
    fit_record = side.get("fit_record") or {}
    return {
        "si": side.get("si_id"),
        "model_id": side.get("model_id"),
        "model_revision": side.get("model_revision"),
        "arm": side.get("arm"),
        "leg": side.get("positions"),
        "response_header_len": side.get("response_header_len"),
        "fit_fingerprint": side.get("fit_fingerprint"),
        "target_layers": side.get("target_layers"),
        "fit_stats": {k: fit_record.get(k) for k in
                      ("floor_ratio", "best_heldout_loss", "epochs_run",
                       "final_residual_ratio") if k in fit_record},
        "hpo": None,
        "selection": None,
        "source": None,
    }


def _check_identity_aliases(obj_dir: Path, ident: dict, flavour: str,
                            side: dict, secondary: dict | None) -> list[str]:
    """Review fix: the sidecars carry additional identity-BEARING alias
    fields that were recorded but never verified. Normalise exactly those
    aliases onto the canonical identity fields and require agreement --
    a contradiction is a fail-loud exit. Everything else in these files
    (metrics, selection prose, COMPLETE, writer notes) stays a pure
    observation, so cosmetic writer differences never gate the run.
    Returns the list of alias fields checked (recorded as an observation).
    """
    checked: list[str] = []

    def req(src: str, field: str, alias_val, canon_key: str) -> None:
        if alias_val != ident[canon_key]:
            raise IdentityError(
                f"{obj_dir}: {src} alias {field}={alias_val!r} contradicts "
                f"the authoritative identity ({canon_key}="
                f"{ident[canon_key]!r})")
        checked.append(f"{src}:{field}")

    # flavour A: train_budget_meta carries positions_at_fit (either as the
    # authoritative sidecar or as the flavour-B secondary).
    tbm = side if flavour == "train_budget_meta" else secondary
    if tbm is not None and "positions_at_fit" in tbm:
        req("train_budget_meta.json", "positions_at_fit",
            tbm["positions_at_fit"], "leg")

    # flavour B extras: fit_provenance.json and summary.json, checked
    # whenever present (the real flavour-B bank always writes both).
    fp_path = obj_dir / "fit_provenance.json"
    if fp_path.is_file():
        fp = json.loads(fp_path.read_text())
        for field, canon_key in (("positions_at_fit", "leg"),
                                 ("resolved_model_id", "model_id"),
                                 ("resolved_revision", "model_revision")):
            if field in fp:
                req("fit_provenance.json", field, fp[field], canon_key)

    su_path = obj_dir / "summary.json"
    if su_path.is_file():
        su = json.loads(su_path.read_text())
        for field, canon_key in (("position", "leg"),
                                 ("model", "model_id"),
                                 ("model_revision", "model_revision"),
                                 ("si", "si"),
                                 ("arm", "arm")):
            if field in su:
                req("summary.json", field, su[field], canon_key)
    return checked


def load_entry(entry: dict, eps_dead: float, near_dead_rel: float
               ) -> LoadedObject:
    from sisteer import site as _site
    path = _site.expand_path(entry["path"], key="manifest path")
    model_key = entry["model_key"]
    if model_key not in MODEL_KEY_TO_HF:
        raise IdentityError(
            f"manifest model_key {model_key!r} absent from MODEL_KEY_TO_HF "
            f"(known: {sorted(MODEL_KEY_TO_HF)}) -- the mapping is asserted, "
            "never guessed")
    if not path.is_file():
        raise IdentityError(f"{model_key}/{entry['si']}: object file "
                            f"missing: {path}")

    # ---- 1. blob: authoritative for geometry and the fit leg ----
    # Read the file bytes ONCE: the recorded sha256 and the parsed tensors
    # come from those exact bytes (no time-of-check/time-of-use seam
    # between hashing and loading), and weights_only=True keeps arbitrary
    # pickle execution out of the load path (verified to hold on all ten
    # real objects).
    raw_bytes = path.read_bytes()
    sha256 = hashlib.sha256(raw_bytes).hexdigest()
    pin = entry.get("sha256")
    if pin is not None and pin != sha256:
        raise IdentityError(
            f"{model_key}/{entry['si']}: object sha256 mismatch -- file has "
            f"{sha256}, manifest pins {pin} (the bank changed under the "
            "frozen manifest; a refit must be re-pinned deliberately)")
    blob = torch.load(io.BytesIO(raw_bytes), map_location="cpu",
                      weights_only=True)
    if blob.get("kind") != "psr":
        raise IdentityError(
            f"{path}: blob kind={blob.get('kind')!r}, expected 'psr'")
    layers = [int(l) for l in blob["layers"]]
    if layers != sorted(set(layers)):
        raise IdentityError(f"{path}: blob layers not sorted unique: "
                            f"{blob['layers']}")
    # The leg comes from the raw artifact -- unambiguous, unlike the banned
    # leg attribute on the hydrated object. Missing key predates the
    # persist and means the historical all-token fit.
    leg = blob.get("positions", "all")
    header_len = blob.get("response_header_len")

    dirs, gws, gbs = blob["directions"], blob["gate_w"], blob["gate_b"]
    d_model = None
    for l in layers:
        k = str(l)
        for name, dct in (("directions", dirs), ("gate_w", gws),
                          ("gate_b", gbs)):
            if k not in dct:
                raise IdentityError(
                    f"{path}: layer {l} missing from blob[{name!r}]")
            # Finiteness is validated, not just shape: a non-finite
            # direction otherwise stays 'live' (Inf norm trips no dead
            # flag), normalisation turns it NaN, the within-SI unit-
            # diagonal assertion cannot see it (NaN comparisons are
            # False), and the run exits 0 with a silently invalid matrix.
            if not bool(torch.isfinite(dct[k]).all()):
                raise IdentityError(
                    f"{model_key}/{entry['si']}: non-finite values "
                    f"(NaN/Inf) in blob[{name!r}] at layer {l} ({path}) "
                    "-- an invalid tensor poisons cosines silently; "
                    "refusing to analyse")
        if dirs[k].dim() != 1 or gws[k].dim() != 1:
            raise IdentityError(
                f"{path}: layer {l} direction/gate_w is not a vector")
        if d_model is None:
            d_model = int(dirs[k].shape[0])
        if int(dirs[k].shape[0]) != d_model or int(gws[k].shape[0]) != d_model:
            raise IdentityError(
                f"{path}: inconsistent d_model at layer {l} "
                f"(expected {d_model})")
        if gbs[k].numel() != 1:
            raise IdentityError(f"{path}: gate_b at layer {l} is not scalar")

    # ---- 2. canonical hydration through the stub seam ----
    stub = _StubModel(d_model, max(layers) + 1)
    obj = PSRObject.load_blob(blob, stub, None)  # tok unused, verified
    # the hydrated fit leg must equal the raw artifact's; a mismatch
    # means load_blob's restore path changed.
    if obj.fit_positions != leg:
        raise IdentityError(
            f"{path}: load_blob leg restore drifted -- hydrated "
            f"fit_positions={obj.fit_positions!r} != blob {leg!r}")
    if leg != "all" and obj.fit_response_header_len != header_len:
        raise IdentityError(
            f"{path}: hydrated fit_response_header_len="
            f"{obj.fit_response_header_len!r} != blob {header_len!r}")
    # Byte-identity of the seam, checked at every load.
    for l in layers:
        k = str(l)
        if not (torch.equal(obj.directions[l].data, dirs[k])
                and torch.equal(obj.gate_w[l].data, gws[k])
                and torch.equal(obj.gate_b[l].data, gbs[k])):
            raise IdentityError(
                f"{path}: stub-seam drift -- hydrated tensors differ from "
                f"the raw blob at layer {l}")

    # ---- 3. sidecars: authoritative for si/model identity ----
    flavour, side, secondary, files_present = _read_sidecars(path.parent)
    ident = _sidecar_identity(flavour, side, secondary, str(path.parent))
    for field, side_val, other_val, other_src in (
            ("si", ident["si"], entry["si"], "manifest"),
            ("arm", ident["arm"], entry["arm"], "manifest"),
            ("fit leg", ident["leg"], leg, "blob"),
            ("response_header_len", ident["response_header_len"],
             header_len, "blob")):
        if side_val != other_val:
            raise IdentityError(
                f"{path.parent}: sidecar {field}={side_val!r} != "
                f"{other_src} {field}={other_val!r}")
    expected_hf = MODEL_KEY_TO_HF[model_key]
    if ident["model_id"] != expected_hf:
        raise IdentityError(
            f"{path.parent}: sidecar model id {ident['model_id']!r} != "
            f"MODEL_KEY_TO_HF[{model_key!r}] = {expected_hf!r} (comparison "
            "through the asserted map; sidecar fields are full HF ids)")
    if ident["target_layers"] is not None and \
            [int(l) for l in ident["target_layers"]] != layers:
        raise IdentityError(
            f"{path.parent}: sidecar target_layers != blob layers: "
            f"{ident['target_layers']} vs {layers}")
    # Path is a CROSS-CHECK only: the <si>__<arm> dirname scheme where it
    # exists; the qwen promoted dirs carry no si in the path, so there the
    # manifest expectation IS the path-level check (no parsing).
    dirname = path.parent.name
    if "__" in dirname:
        parts = dirname.split("__")
        if len(parts) == 2 and (parts[0] != ident["si"]
                                or parts[1] != ident["arm"]):
            raise IdentityError(
                f"{path.parent}: dirname {dirname!r} disagrees with sidecar "
                f"si={ident['si']!r} arm={ident['arm']!r}")
    # Identity-bearing sidecar aliases must agree with the (already
    # blob-verified) authoritative identity; run after the blob/manifest
    # loop so ident['leg'] is known equal to the blob's leg.
    alias_checks = _check_identity_aliases(path.parent, ident, flavour,
                                           side, secondary)

    # ---- 4. numpy views + health flags ----
    D = np.stack([dirs[str(l)].detach().numpy().astype(np.float64)
                  for l in layers])
    W = np.stack([gws[str(l)].detach().numpy().astype(np.float64)
                  for l in layers])
    B = np.array([float(gbs[str(l)]) for l in layers], dtype=np.float64)
    norms = np.linalg.norm(D, axis=1)
    wnorms = np.linalg.norm(W, axis=1)
    dead_d = norms < eps_dead
    near_d = ~dead_d & (norms < near_dead_rel * np.median(norms))
    dead_w = wnorms < eps_dead
    near_w = ~dead_w & (wnorms < near_dead_rel * np.median(wnorms))
    live_dir = ~dead_d & ~near_d
    live_gw = ~dead_w & ~near_w

    # Gate health: forward-free signals only; observations, never
    # exclusions -- no gate flag masks a direction.
    frozen_groups = psr_frozen_gate_b(obj)
    n_frozen = sum(len(ls) for ls in frozen_groups.values())
    # 'Provably' means provably (review fix): only an EXACTLY zero gate_w
    # makes ReLU(w.h+b) input-independent -- for ANY nonzero w, however
    # small, some h makes the preactivation positive. Epsilon-small-but-
    # nonzero weights get a separate, explicitly HEURISTIC label instead.
    zero_w = ~np.any(W != 0.0, axis=1)       # count_nonzero(w) == 0, exact
    provably_dead_gate = zero_w & (B <= 0)   # ReLU(0.h+b) == 0 for every h
    constant_gate = zero_w & (B > 0)         # ReLU(0.h+b) == b for every h
    tiny_gate_w = ~zero_w & dead_w           # 0 < ||w|| < eps: no proof

    # Gauge-invariant per-layer quantities. Raw norms/gate_b are recorded
    # too but are gauge-DEPENDENT: never compare them across objects.
    jac_norms = norms * wnorms
    with np.errstate(divide="ignore", invalid="ignore"):
        thresholds = np.where(wnorms > 0, B / wnorms, np.nan)

    idx = {l: i for i, l in enumerate(layers)}
    record = {
        "model_key": model_key,
        "si": ident["si"],
        "arm": ident["arm"],
        "path": _tilde(path),
        "sha256": sha256,  # hash of the SAME bytes the tensors came from
        "sidecar_flavor": flavour,
        "sidecar_files_present": files_present,
        "alias_identity_checks": alias_checks,  # observation: what agreed
        "complete_marker_present": (path.parent / "COMPLETE").is_file(),
        "model_id": ident["model_id"],
        "model_revision": ident["model_revision"],
        "leg": leg,
        "response_header_len": header_len,
        "layers": layers,
        "d_model": d_model,
        # theoretical isotropic scales (F4): vector cosines ~ N(0, 1/d);
        # the rank-one Jacobian product of two INDEPENDENT such cosines
        # has sd 1/sqrt(d) * 1/sqrt(d) = 1/d (the rank-one matrices live
        # in a d^2-dimensional ambient space)
        "chance_sigma": _f(1.0 / math.sqrt(d_model)),
        "chance_sigma_jacobian": _f(1.0 / d_model),
        "per_layer_norms": {str(l): _f(norms[idx[l]]) for l in layers},
        "gate_w_norms": {str(l): _f(wnorms[idx[l]]) for l in layers},
        "gate_b": {str(l): _f(B[idx[l]]) for l in layers},
        "jacobian_norms": {str(l): _f(jac_norms[idx[l]]) for l in layers},
        "gate_thresholds": {str(l): _f(thresholds[idx[l]]) for l in layers},
        "min_norm": _f(norms.min()),
        "max_norm": _f(norms.max()),
        "dead_layers": [l for l in layers if dead_d[idx[l]]],
        "near_dead_layers": [l for l in layers if near_d[idx[l]]],
        "dead_gate_w_layers": [l for l in layers if dead_w[idx[l]]],
        "near_dead_gate_w_layers": [l for l in layers if near_w[idx[l]]],
        "frozen_gate_b_groups": {str(v): ls
                                 for v, ls in frozen_groups.items()},
        "frozen_fingerprint_fires": bool(n_frozen >= FROZEN_LAYER_FLOOR),
        "provably_dead_gate_layers": [l for l in layers
                                      if provably_dead_gate[idx[l]]],
        "constant_gate_layers": [l for l in layers if constant_gate[idx[l]]],
        # heuristic flag ONLY -- epsilon-small but nonzero gate_w proves
        # nothing about the gate's behaviour on real hidden states
        "tiny_nonzero_gate_w_layers": [l for l in layers
                                       if tiny_gate_w[idx[l]]],
        "sidecar_fit_fingerprint": ident["fit_fingerprint"],  # never asserted
        "fit_stats": ident["fit_stats"],
        "hpo": ident["hpo"],
        "selection": ident["selection"],
        "source": ident["source"],
    }
    return LoadedObject(record, D, W, B, layers, live_dir, live_gw)


# --------------------------------------------------------------------------
# Cosine machinery, aggregates, empirical null
# --------------------------------------------------------------------------

def cosine_matrix(A: np.ndarray, Bm: np.ndarray, live_a: np.ndarray,
                  live_b: np.ndarray) -> np.ndarray:
    """Signed cosine block: M[i, j] = cos(a_i, b_j), float64, NaN at
    masked (dead/near-dead) rows/cols. No abs(), no sign fixing -- the
    gate is ReLU, so direction sign is meaningful."""
    with np.errstate(invalid="ignore", divide="ignore"):
        An = A / np.linalg.norm(A, axis=1, keepdims=True)
        Bn = Bm / np.linalg.norm(Bm, axis=1, keepdims=True)
    M = An @ Bn.T
    M[~live_a, :] = np.nan
    M[:, ~live_b] = np.nan
    return M


def region_masks(layers: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Fractional-depth regions from ORIGINAL layer indices: the
    depth of layer l is l / max(layers) (= l/(L-1) on the full 0..L-1
    stacks the real banks fit); early = depth < 0.25, late = depth >= 0.5.
    Positional quarters (n // 4 of the layer list) are WRONG whenever L is
    not divisible by 4 -- for L=42, layer 10 (depth 10/41 = 0.244) and for
    L=35, layer 8 (depth 0.235) belong to the first quarter but n//4
    excludes them -- and compressing masked layers out before splitting
    shifts survivors across region boundaries. For L=32 these masks
    reproduce L0-7 and L16-31 exactly."""
    arr = np.asarray(layers, dtype=np.float64)
    denom = float(arr.max()) if arr.size and arr.max() > 0 else 1.0
    depth = arr / denom
    return depth < 0.25, depth >= 0.5


def chance_sigma(kind: str, d_model: int) -> float:
    """Theoretical isotropic chance scale per block kind: a
    cosine of two independent isotropic vectors in R^d is ~N(0, 1/d), so
    sigma = 1/sqrt(d) for the direction and gate_w cosine blocks. The
    rank-one Jacobian cosine is the PRODUCT of two such independent
    cosines -- cos(d_A w_A^T, d_B w_B^T) = cos(d_A,d_B) * cos(w_A,w_B) --
    with variance (1/d)*(1/d), so its sigma is 1/d, not 1/sqrt(d) (the
    rank-one matrices live in a d^2-dimensional ambient space)."""
    if kind == "jacobian":
        return 1.0 / d_model
    if kind in ("direction", "gate_w"):
        return 1.0 / math.sqrt(d_model)
    raise IdentityError(f"chance_sigma: unknown block kind {kind!r}")


def _stats(vals: np.ndarray) -> dict:
    if vals.size == 0:
        return {"mean": None, "sd": None, "n": 0}
    return {"mean": _f(vals.mean()),
            "sd": _f(vals.std(ddof=1)) if vals.size > 1 else None,
            "n": int(vals.size)}


def _null_dist(vals: np.ndarray, observed: float | None) -> dict:
    """Distribution record + the observed value's percentile in it."""
    st = _stats(vals)
    st["p025"] = _f(np.quantile(vals, 0.025)) if vals.size else None
    st["p975"] = _f(np.quantile(vals, 0.975)) if vals.size else None
    st["observed"] = _f(observed)
    st["observed_percentile"] = (_f((vals <= observed).mean())
                                 if vals.size and observed is not None
                                 else None)
    return st


def empirical_null(M: np.ndarray, live: np.ndarray, layers: list[int]
                   ) -> dict | None:
    """Empirical null: shuffled layer correspondences, individual
    off-diagonal entries, distance-matched off-diagonals, circular shifts
    (position-preserving rotations), each with the observed diagonal
    mean's percentile -- for the all-layer aggregate AND each fractional-
    depth region. Regions and pair distances use ORIGINAL layer positions:
    masked layers are excluded pairwise, never compressed out first, so
    'distance 1' is true layer adjacency and a masked layer cannot shift
    its survivors across a region boundary. Plus off-diagonal maxima
    (layer-shift detector). All quantities DESCRIPTIVE, never
    inferential."""
    live = np.asarray(live, dtype=bool)
    lay = np.asarray(layers, dtype=int)
    n = len(layers)
    if int(live.sum()) < 3:
        return None
    early_m, late_m = region_masks(layers)

    def region_rec(mask: np.ndarray) -> dict:
        li = np.where(mask & live)[0]
        k = int(li.size)
        observed = float(M[li, li].mean()) if k else None
        rec = {"region_layers": [int(lay[i]) for i in li],
               "n_live": k,
               "observed_diag_mean": _f(observed)}
        # shuffled layer correspondences within the region (a null for
        # the REGION MEAN -- never to be drawn behind per-layer traces)
        if k >= 2:
            sub = M[np.ix_(li, li)]
            ar = np.arange(k)
            rng = np.random.default_rng(NULL_SEED)
            perm_means = np.array([sub[ar, rng.permutation(k)].mean()
                                   for _ in range(NULL_PERMS)])
            rec["shuffled"] = _null_dist(perm_means, observed)
        else:
            rec["shuffled"] = None
        # individual off-diagonal entries (the per-entry null)
        A, Bi = np.meshgrid(li, li, indexing="ij")
        offsel = A != Bi
        rec["offdiag_entries"] = _null_dist(M[A[offsel], Bi[offsel]],
                                            observed)
        # distance-matched off-diagonals at TRUE layer-number distance
        dist = np.abs(lay[A] - lay[Bi])
        rec["distance_matched"] = {
            str(o): _null_dist(M[A[dist == o], Bi[dist == o]], observed)
            for o in NULL_OFFSETS}
        # circular shifts over the region's ORIGINAL position list (live
        # pairs only; preserves local layer autocorrelation)
        q = np.where(mask)[0]
        shift_means = []
        for s in range(1, q.size):
            rows, cols = q, q[(np.arange(q.size) + s) % q.size]
            ok = live[rows] & live[cols]
            if ok.any():
                shift_means.append(float(M[rows[ok], cols[ok]].mean()))
        if shift_means:
            cs = _null_dist(np.array(shift_means), observed)
            cs["n_shifts"] = len(shift_means)
            rec["circular_shift"] = cs
        else:
            rec["circular_shift"] = None
        return rec

    regions = {"all": region_rec(np.ones(n, dtype=bool)),
               "early": region_rec(early_m),
               "late": region_rec(late_m)}

    def absmax_rec(li: np.ndarray) -> dict | None:
        if li.size < 2:
            return None
        A, Bi = np.meshgrid(li, li, indexing="ij")
        offsel = A != Bi
        vals = M[A[offsel], Bi[offsel]]
        t = int(np.abs(vals).argmax())
        i, j = int(A[offsel][t]), int(Bi[offsel][t])
        return {"value": _f(M[i, j]),
                "layer_i": int(lay[i]), "layer_j": int(lay[j])}

    return {"seed": NULL_SEED, "n_perm": NULL_PERMS,
            "position_convention": (
                "regions and pair distances use ORIGINAL layer indices; "
                "masked layers are excluded pairwise, never compressed "
                "out of the axis"),
            "regions": regions,
            "late_offdiag_max": absmax_rec(np.where(late_m & live)[0]),
            "offdiag_max": absmax_rec(np.where(live)[0])}


def pooled_offdiag_band(pair_mats: list[np.ndarray],
                        pair_lives: list[np.ndarray],
                        layers: list[int]) -> dict:
    """F2: the per-entry null band for diagonal-trace figures. For each
    fractional-depth region, POOL the individual off-diagonal entries of
    the given blocks by CONCATENATION (never an envelope of per-block
    quantiles, and never a distribution of block MEANS -- a mean-null band
    is ~sqrt(k) narrower than the per-entry spread and makes ordinary
    layers look exceptional), then take the 2.5/97.5% quantiles of the
    pooled sample. Returns {region: (p025, p975, n_entries) | None}."""
    early_m, late_m = region_masks(layers)
    n = len(layers)
    out = {}
    for name, rmask in (("all", np.ones(n, dtype=bool)),
                        ("early", early_m), ("late", late_m)):
        pool = []
        for M, lv in zip(pair_mats, pair_lives):
            li = np.where(rmask & np.asarray(lv, dtype=bool))[0]
            if li.size < 2:
                continue
            A, Bi = np.meshgrid(li, li, indexing="ij")
            sel = A != Bi
            pool.append(M[A[sel], Bi[sel]])
        vals = np.concatenate(pool) if pool else np.empty(0)
        out[name] = ((float(np.quantile(vals, 0.025)),
                      float(np.quantile(vals, 0.975)), int(vals.size))
                     if vals.size >= 2 else None)
    return out


def block_record(model_key: str, si_a: str, si_b: str, M: np.ndarray,
                 layers: list[int], live: np.ndarray, sigma: float,
                 sha_a: str, sha_b: str, matrix_file: str) -> dict:
    diag = np.diag(M).copy()
    live_idx = np.where(live)[0]
    n_live = int(live_idx.size)
    vals = diag[live_idx]
    early_m, late_m = region_masks(layers)
    early_live = np.where(early_m & np.asarray(live, dtype=bool))[0]
    late_live = np.where(late_m & np.asarray(live, dtype=bool))[0]

    def agg(idxs):
        if idxs.size == 0:
            return None
        return _f(diag[idxs].mean())

    diag_mean = _f(vals.mean()) if n_live else None
    if n_live:
        arg = live_idx[int(np.argmax(vals))]
        diag_max, diag_max_layer = _f(diag[arg]), int(layers[arg])
    else:
        diag_max, diag_max_layer = None, None
    mean_early, mean_late = agg(early_live), agg(late_live)
    su = (lambda v: _f(v / sigma) if v is not None else None)
    return {
        "model_key": model_key, "si_a": si_a, "si_b": si_b,
        "matrix_file": matrix_file,
        "sha256_a": sha_a, "sha256_b": sha_b,
        "sigma": _f(sigma),
        "same_layer_diagonal": {str(l): _f(diag[i])
                                for i, l in enumerate(layers)},
        "same_layer_diagonal_sigma_units": {str(l): su(_f(diag[i]))
                                            for i, l in enumerate(layers)},
        "diag_mean": diag_mean, "diag_max": diag_max,
        "diag_max_layer": diag_max_layer,
        "diag_mean_sigma_units": su(diag_mean),
        "diag_mean_se": _f(sigma / math.sqrt(n_live)) if n_live else None,
        "diag_mean_early": mean_early, "diag_mean_late": mean_late,
        "diag_early_sigma_units": su(mean_early),
        "diag_late_sigma_units": su(mean_late),
        "n_live_layers": n_live,
        "masked_layers": [int(layers[i]) for i in range(len(layers))
                          if not live[i]],
        "null": empirical_null(M, live, layers),
    }


# --------------------------------------------------------------------------
# Derived contrasts: the headline quantities
# --------------------------------------------------------------------------

def compute_contrasts(model_key: str, blocks: dict, layers: list[int],
                      sigma: float, pair_mats: dict | None = None
                      ) -> dict | None:
    """dL = CI - (CS+SI)/2 (long-pair excess) and dS = CS - SI
    (mixed-pair difference), per layer on the same-layer diagonals.
    Neither contrast identifies a mechanism: the two mixed pairs do NOT
    hold coarse length constant -- the constitution is CLOSER in token
    length to safety than the irrelevant control is, so a purely
    length-graded similarity model yields dS > 0 with no content component
    -- and dL/dS are two contrasts of three cosines, not a component
    decomposition. Requires all three cross-SI blocks. sigma-units use the
    independence-propagated sigmas (dL: sigma*sqrt(1.5), dS: sigma*sqrt(2))
    and are DESCRIPTIVE only.

    Empirical nulls (F1): given the three full pair matrices, the contrast
    MATRICES MdL = CI - (CS+SI)/2 and MdS = CS - SI are formed elementwise
    and run through the same empirical_null machinery as every block.
    Because the contrasts are linear in the blocks, every permutation /
    shift / off-diagonal draw on the combined matrix is EXACTLY one shared
    layer mapping applied to CI, CS and SI simultaneously -- never three
    independent nulls combined."""
    try:
        cs = blocks[("safety_refusal", "si_constitution")]
        si = blocks[("safety_refusal", "si_irrelevant")]
        ci = blocks[("si_constitution", "si_irrelevant")]
    except KeyError:
        return None
    d_cs = np.array([_nan(cs["same_layer_diagonal"][str(l)]) for l in layers])
    d_si = np.array([_nan(si["same_layer_diagonal"][str(l)]) for l in layers])
    d_ci = np.array([_nan(ci["same_layer_diagonal"][str(l)]) for l in layers])
    dL = d_ci - (d_cs + d_si) / 2.0
    dS = d_cs - d_si
    live = np.isfinite(dL) & np.isfinite(dS)
    live_idx = np.where(live)[0]
    n_live = int(live_idx.size)
    early_m, late_m = region_masks(layers)
    sigma_dL, sigma_dS = sigma * math.sqrt(1.5), sigma * math.sqrt(2.0)

    def side(series, s):
        vals = series[live_idx]
        mean = _f(vals.mean()) if n_live else None
        el = np.where(early_m & live)[0]
        ll = np.where(late_m & live)[0]
        return {
            "l0": _f(series[0]) if live[0] else None,
            "mean": mean,
            "mean_early": _f(series[el].mean()) if el.size else None,
            "mean_late": _f(series[ll].mean()) if ll.size else None,
            "sigma": _f(s),
            "mean_sigma_units": _f(mean / s) if mean is not None else None,
            "mean_se": _f(s / math.sqrt(n_live)) if n_live else None,
            "sign_of_mean": (None if mean is None
                             else ("+" if mean > 0 else
                                   "-" if mean < 0 else "0")),
        }

    null = None
    if pair_mats is not None and all(
            pair_mats.get(key) is not None for key in ("cs", "si", "ci")):
        MdL = pair_mats["ci"] - (pair_mats["cs"] + pair_mats["si"]) / 2.0
        MdS = pair_mats["cs"] - pair_mats["si"]
        null = {
            "dL": empirical_null(MdL, live, layers),
            "dS": empirical_null(MdS, live, layers),
            "mapping_note": (
                "one SHARED layer mapping across the three pair blocks: "
                "the null is computed on the elementwise contrast matrix, "
                "which for every permutation/shift/entry draw equals the "
                "same mapping applied to CI, CS and SI simultaneously"),
        }

    return {
        "model_key": model_key,
        "n_live_layers": n_live,
        "per_layer": {str(l): {"dL": _f(dL[i]), "dS": _f(dS[i])}
                      for i, l in enumerate(layers)},
        "dL": side(dL, sigma_dL),
        "dS": side(dS, sigma_dS),
        "null": null,
        "sigma_note": ("contrast sigmas assume independence of the three "
                       "same-layer diagonals; descriptive only (D12/D14e)"),
    }


def _nan(x):
    return float("nan") if x is None else float(x)


# --------------------------------------------------------------------------
# SI token metadata: tokens, not words, under each model's
# RECORDED tokenizer id + pinned revision, offline only.
# --------------------------------------------------------------------------

def compute_token_meta(models: dict[str, tuple[str, str | None]],
                       si_texts: dict[str, str]) -> dict:
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoTokenizer

    from sisteer.models import apply_chat_template_folding

    out = {}
    for model_key in sorted(models):
        hf_id, revision = models[model_key]
        if not revision:
            raise IdentityError(
                f"{model_key}: no model_revision recorded in the sidecar -- "
                "cannot pin the tokenizer (D16); refusing to count tokens "
                "under an unpinned tokenizer")
        try:
            tok = AutoTokenizer.from_pretrained(
                hf_id, revision=revision, local_files_only=True)
        except Exception as e:  # noqa: BLE001 -- any load failure is fatal
            raise IdentityError(
                f"tokenizer for {model_key} ({hf_id} @ {revision}) failed "
                f"to load offline (local_files_only=True): {e!r}. This is a "
                "hard error -- no word-count fallback, no network (D15).")
        si_tokens = {}
        for si_id in sorted(si_texts):
            text = si_texts[si_id]
            raw = len(tok(text, add_special_tokens=False)["input_ids"])
            rendered_str = apply_chat_template_folding(
                tok,
                [{"role": "system", "content": text},
                 {"role": "user", "content": ""}],
                add_generation_prompt=False)
            rendered = len(tok(rendered_str,
                               add_special_tokens=False)["input_ids"])
            if rendered < raw:
                raise IdentityError(
                    f"{model_key}/{si_id}: rendered chat-template count "
                    f"({rendered}) < raw count ({raw}) -- the rendered "
                    "prefill contains the SI verbatim plus template "
                    "overhead, so this means the probe is wrong (D15)")
            si_tokens[si_id] = {"raw": raw, "rendered": rendered}
        entry = {"tokenizer_id": hf_id, "revision": revision,
                 "si_tokens": si_tokens,
                 "rendered_convention": (
                     "SI as system message + empty user turn through "
                     "sisteer.models.apply_chat_template_folding, "
                     "add_generation_prompt=False, tokenized with "
                     "add_special_tokens=False")}
        if all(s in si_tokens for s in SI_ORDER):
            c = si_tokens["si_constitution"]["raw"]
            entry["ratios"] = {
                "irrelevant_over_constitution_raw":
                    _f(si_tokens["si_irrelevant"]["raw"] / c),
                "constitution_over_safety_raw":
                    _f(c / si_tokens["safety_refusal"]["raw"]),
            }
        out[model_key] = entry
    return out


# --------------------------------------------------------------------------
# Regression gate + self-check (the cannot-validate rule)
# --------------------------------------------------------------------------

def run_regression_check(blocks: list[dict], objects: list[dict],
                         contrasts: list[dict]) -> dict:
    """Compare the computed llama8b table against the two-path-verified
    expected values. Returns a record with per-quantity expected/computed/
    pass; overall 'passed' is False on ANY mismatch or missing quantity
    (inability to validate must not read as success)."""
    checks = []

    def add(name, expected, computed, tol=REGRESSION_TOL, exact=False):
        if computed is None:
            ok = False
        elif exact:
            ok = computed == expected
        else:
            ok = abs(computed - expected) <= tol
        checks.append({"name": name, "expected": expected,
                       "computed": computed, "tolerance": (None if exact
                                                           else tol),
                       "passed": bool(ok)})

    by_pair = {(b["si_a"], b["si_b"]): b for b in blocks
               if b["model_key"] == REGRESSION_MODEL}
    for pair, exp in sorted(REGRESSION_PAIRS.items()):
        b = by_pair.get(pair)
        tag = f"{pair[0]}x{pair[1]}"
        if b is None:
            checks.append({"name": tag, "expected": "block present",
                           "computed": "MISSING", "tolerance": None,
                           "passed": False})
            continue
        add(f"{tag}.l0", exp["l0"],
            b["same_layer_diagonal"].get("0"))
        add(f"{tag}.mean", exp["mean"], b["diag_mean"])
        add(f"{tag}.late_mean", exp["late_mean"], b["diag_mean_late"])
        add(f"{tag}.max", exp["max"], b["diag_max"])
        add(f"{tag}.argmax_layer", exp["argmax_layer"],
            b["diag_max_layer"], exact=True)

    by_si = {o["si"]: o for o in objects
             if o["model_key"] == REGRESSION_MODEL}
    for si_id, exp in sorted(REGRESSION_MIN_NORMS.items()):
        o = by_si.get(si_id)
        add(f"min_norm.{si_id}", exp,
            None if o is None else o["min_norm"])

    con = next((c for c in contrasts
                if c["model_key"] == REGRESSION_MODEL), None)
    # Published derived values carry compounded 4-decimal rounding from the
    # pairwise table -- see REGRESSION_TOL_DERIVED.
    add("contrast.dL_mean", REGRESSION_CONTRASTS["dL_mean"],
        None if con is None else con["dL"]["mean"], tol=REGRESSION_TOL_DERIVED)
    add("contrast.dS_mean", REGRESSION_CONTRASTS["dS_mean"],
        None if con is None else con["dS"]["mean"], tol=REGRESSION_TOL_DERIVED)
    add("contrast.dL_l0", REGRESSION_CONTRASTS["dL_l0"],
        None if con is None else con["dL"]["l0"], tol=REGRESSION_TOL_DERIVED)
    add("contrast.dS_l0", REGRESSION_CONTRASTS["dS_l0"],
        None if con is None else con["dS"]["l0"], tol=REGRESSION_TOL_DERIVED)

    return {"model": REGRESSION_MODEL, "tolerance": REGRESSION_TOL,
            "tolerance_derived": REGRESSION_TOL_DERIVED,
            "checks": checks,
            "passed": bool(checks) and all(c["passed"] for c in checks)}


def print_regression(rec: dict) -> None:
    print(f"\nRegression gate ({rec['model']}, tol={rec['tolerance']:g}, "
          f"derived tol={rec['tolerance_derived']:g}):")
    for c in rec["checks"]:
        status = "PASS" if c["passed"] else "FAIL"
        print(f"  [{status}] {c['name']}: expected={c['expected']} "
              f"computed={c['computed']}")
    print(f"  => {'PASSED' if rec['passed'] else 'FAILED'}")


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

SURFACE, TXT, TXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e5e4e0"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
NEUTRAL, POLE_RED, MASK_GRAY = "#f0efec", "#e34948", "#b3b2ad"
PAIR_COLORS = {
    ("si_constitution", "si_irrelevant"): BLUE,    # CI -- co-headline first
    ("safety_refusal", "si_constitution"): ORANGE,  # CS
    ("safety_refusal", "si_irrelevant"): AQUA,      # SI
}
PAIR_LABELS = {
    ("si_constitution", "si_irrelevant"): "constitution x irrelevant (CI)",
    ("safety_refusal", "si_constitution"): "constitution x safety (CS)",
    ("safety_refusal", "si_irrelevant"): "safety x irrelevant (SI)",
}
# The whole-model diffmean cos(const, irrelevant) overlay is withdrawn as
# apples-to-oranges (the "factor" ranges 0.61x-5.6x purely by choice of
# aggregation); the qualified comparison lives in prose only, never on a
# figure (F6).


def _shared_scale(mats: dict, kind: str) -> dict:
    """Shared symmetric colour ENDPOINTS for one heatmap kind: V is the
    TRUE maximum |value| pooled over every drawn cell of that kind. The
    within-SI unit diagonal (1.0 by construction -- a sanity check, not
    a finding) is masked from the drawing, so it is excluded here too.
    Under the previous shared LINEAR scale V had to be a clipped
    percentile to keep the bulk visible at all, and the cross-SI
    0.02-0.15 range -- which carries the ordering the study is about --
    still rendered near-white next to the ~0.6 within-SI adjacent-layer
    band. The asinh norm (_heat_norm) compresses the top of the range
    instead, so the true max costs nothing and NO drawn cell is
    clipped."""
    pool = []
    for (mk, sa, sb, kd), M in mats.items():
        if kd != kind:
            continue
        vals = M.copy()
        if sa == sb:
            np.fill_diagonal(vals, np.nan)  # trivial unit diagonal: masked
        pool.append(np.abs(vals[np.isfinite(vals)]))
    allv = np.concatenate(pool) if pool else np.empty(0)
    if allv.size == 0:
        return {"V": 0.05}
    return {"V": float(allv.max())}


def _heat_norm(V: float, sigma: float):
    """Symmetric NONLINEAR diverging norm for every heatmap
    (matplotlib.colors.AsinhNorm): near-linear inside +/-linear_width,
    logarithmic-like compression beyond. One shared linear scale cannot
    show the ~0.6 within-SI adjacent-layer band and the 0.02-0.15
    cross-SI bulk (the whole point of the study) at once -- the bulk
    rendered indistinguishable from zero. linear_width = the block
    kind's +/-2 sigma isotropic chance half-width for this model, so the
    region indistinguishable from chance is exactly the near-white part
    of the map and everything above chance gets real colour separation
    (descriptive). Nothing is clipped: vmax is the drawn data's
    true max (_shared_scale)."""
    from matplotlib.colors import AsinhNorm
    return AsinhNorm(linear_width=2.0 * sigma, vmin=-V, vmax=V)


def _fmt_tick(t: float) -> str:
    return "0" if t == 0 else f"{t:+.3g}"


def _asinh_ticks(V: float, linear_width: float, chance: float
                 ) -> list[float]:
    """Colorbar ticks for the asinh scale, at the values a reader cares
    about: 0, +/-chance (the hatched 2 sigma band edge), a 1-2-5 decade
    ladder between chance and V, and the TRUE +/-V endpoints. The bar
    axis is asinh-scaled (matplotlib draws it under the image norm), so
    each tick lands at its true NONLINEAR position -- the compression is
    visible on the bar rather than hidden. Ladder ticks that would
    overprint the chance tick, an already-kept tick, or the endpoint
    (closer than 8% of the half-bar in display space) are dropped."""
    if not 0.0 < chance < V:
        return [-V, 0.0, V]
    denom = math.asinh(V / linear_width)

    def pos(v: float) -> float:
        return math.asinh(v / linear_width) / denom

    kept = [chance, V]
    for v in (c * 10.0 ** k for k in range(-4, 1) for c in (1.0, 2.0, 5.0)):
        if chance < v < V and all(abs(pos(v) - pos(u)) >= 0.08
                                  for u in kept):
            kept.append(v)
    side = sorted(kept)
    return [-t for t in reversed(side)] + [0.0] + side


def _display_matrix(M: np.ndarray, same_si: bool) -> np.ndarray:
    """Copy of a block for DRAWING only. Within-SI blocks get their
    trivial unit diagonal (cos = 1.0 by construction; still asserted at
    analysis time) masked to NaN so it renders as the masked colour and
    cannot dominate the colour scale. Never mutates the saved matrix."""
    out = M.copy()
    if same_si:
        np.fill_diagonal(out, np.nan)
    return out


def _si_token_footer(model_key: str, token_meta: dict, si_meta_rec: dict,
                     present_sis: list[str]) -> str:
    tm = token_meta.get(model_key)
    if tm is None:
        parts = [f"{s} {si_meta_rec[s]['words']:,} w" for s in present_sis]
        return ("SI lengths (WORDS -- token counts not computed this run): "
                + " / ".join(parts))
    toks = tm["si_tokens"]
    parts = [f"{s} {toks[s]['raw']:,} tok" for s in present_sis
             if s in toks]
    line = (f"SI tokens ({tm['tokenizer_id']} @ {tm['revision'][:8]}): "
            + " / ".join(parts))
    r = tm.get("ratios")
    if r and len(present_sis) > 1:
        line += (f" -- irrelevant/constitution x"
                 f"{r['irrelevant_over_constitution_raw']:.3f} "
                 f"(approximately length-matched, "
                 f"{100 * (r['irrelevant_over_constitution_raw'] - 1):.1f}% "
                 f"in tokens); constitution/safety x"
                 f"{r['constitution_over_safety_raw']:.1f} vs "
                 f"irrelevant/safety x"
                 f"{r['constitution_over_safety_raw'] * r['irrelevant_over_constitution_raw']:.1f}"
                 f" -- the two MIXED pairs are not length-matched")
    return line


def _mixed_pair_length_line(model_keys: list[str], token_meta: dict
                            ) -> str | None:
    """Caption line: the two mixed pairs (CS and SI) do NOT hold
    coarse length constant -- the constitution sits closer in token length
    to safety than the irrelevant control does -- so a purely length-graded
    similarity model produces dS > 0 with no content component. Stated on
    the headline figure so dS is never read as length-controlled. Returns
    None when token counts were not computed (--no-tokens)."""
    parts = []
    for mk in model_keys:
        toks = (token_meta.get(mk) or {}).get("si_tokens") or {}
        try:
            s = toks["safety_refusal"]["raw"]
            c = toks["si_constitution"]["raw"]
            i = toks["si_irrelevant"]["raw"]
        except KeyError:
            continue
        parts.append(f"{mk} constitution/safety x{c / s:.1f} vs "
                     f"irrelevant/safety x{i / s:.1f}")
    if not parts:
        return None
    return ("the mixed pairs are NOT length-matched: " + "; ".join(parts)
            + " (raw tokens) -- the constitution is closer in length to "
              "safety than the control is, so a purely length-graded "
              "similarity model yields dS > 0 with no content component "
              "(D20a).")


def make_figures(figs_dir: Path, loaded: list[LoadedObject], mats: dict,
                 results: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    figs_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    cmap = LinearSegmentedColormap.from_list("cosdiv",
                                             [BLUE, NEUTRAL, POLE_RED])
    cmap.set_bad(MASK_GRAY)
    token_meta = results.get("token_meta", {})
    si_meta_rec = results["si_meta"]
    by_model: dict[str, dict[str, LoadedObject]] = {}
    for lo in loaded:
        by_model.setdefault(lo.model_key, {})[lo.si] = lo
    model_order = list(dict.fromkeys(lo.model_key for lo in loaded))
    blocks_by = {(b["model_key"], b["si_a"], b["si_b"]): b
                 for b in results["blocks"]}
    contrasts_by = {c["model_key"]: c for c in results["contrasts"]}
    scale_by_kind = {kind: _shared_scale(mats, kind)
                     for kind in ("direction", "gate_w", "jacobian")}

    def style_axes(ax):
        ax.set_facecolor(SURFACE)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.tick_params(colors=TXT2, labelsize=8)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)

    def depth_axis(ax, n_layers):
        L = n_layers - 1 if n_layers > 1 else 1
        sec = ax.secondary_xaxis(
            "top", functions=(lambda x, L=L: x / L, lambda x, L=L: x * L))
        sec.set_xlabel("fractional depth l/(L-1)", fontsize=7, color=TXT2)
        sec.tick_params(colors=TXT2, labelsize=7)
        return sec

    def footer(fig, lines):
        import textwrap
        width = max(60, int(fig.get_figwidth() * 18))
        wrapped = []
        for ln in lines:
            wrapped.extend(textwrap.wrap(ln, width=width,
                                         subsequent_indent="      ")
                           or [""])
        fig.text(0.01, 0.005, "\n".join(wrapped), fontsize=6.5, color=TXT2,
                 va="bottom", ha="left")

    def series_legend(ax, title=None, **kw):
        """One real legend per figure (drawn on the first panel; every
        panel plots the same series). Inline end-of-line labels collided
        on crowded panels and never explained the bands/reference lines;
        the legend carries every colour-coded artist, spelled out."""
        leg = ax.legend(fontsize=7, frameon=False, labelcolor=TXT2,
                        title=title, **kw)
        if title is not None:
            leg.get_title().set_fontsize(6.5)
            leg.get_title().set_color(TXT2)
        return leg

    def sha_line(model_key):
        objs = by_model[model_key]
        return " ".join(f"{s}={objs[s].sha256[:8]}"
                        for s in SI_ORDER if s in objs)

    def save(fig, name):
        fig.savefig(figs_dir / name, dpi=160, facecolor=SURFACE)
        plt.close(fig)
        written.append(name)

    grid_models = [m for m in model_order
                   if all(s in by_model[m] for s in SI_ORDER)]
    safety_models = [m for m in model_order if m not in grid_models
                     and "safety_refusal" in by_model[m]]

    # ---- 1. headline: derived contrasts ----
    if grid_models:
        fig, axes = plt.subplots(1, len(grid_models),
                                 figsize=(5.6 * len(grid_models), 4.6),
                                 sharey=True, squeeze=False)
        fig.patch.set_facecolor(SURFACE)
        for ax, mk in zip(axes[0], grid_models):
            style_axes(ax)
            lo = next(iter(by_model[mk].values()))
            layers = lo.layers
            con = contrasts_by.get(mk)
            if con is None:
                continue
            dL = np.array([_nan(con["per_layer"][str(l)]["dL"])
                           for l in layers])
            dS = np.array([_nan(con["per_layer"][str(l)]["dS"])
                           for l in layers])
            sigma = chance_sigma("direction", lo.d_model)
            band = 2 * sigma * math.sqrt(2.0)
            ax.axhspan(-band, band, color=GRID, alpha=0.5, zorder=0,
                       label="isotropic +/-2*sigma*sqrt(2)\nchance band "
                             "(descriptive)")
            ax.axhline(0, color=TXT2, linewidth=0.8)
            ax.plot(layers, dL, color=BLUE, linewidth=2,
                    label="dL = CI - (CS+SI)/2\n(long-pair excess)")
            ax.plot(layers, dS, color=ORANGE, linewidth=2,
                    label="dS = CS - SI\n(mixed-pair difference)")
            role = ("exploratory" if mk == "llama8b"
                    else "held-out confirmatory"
                    if mk == "gemma4e4b" else "descriptive")
            ax.set_title(
                f"{mk} ({role}) -- mean dL {con['dL']['mean']:+.4f}, "
                f"dS {con['dS']['mean']:+.4f}", fontsize=9, color=TXT)
            ax.set_xlabel("layer", fontsize=8, color=TXT2)
            depth_axis(ax, len(layers))
        series_legend(
            axes[0][0], loc="upper right",
            title="CI = constitution x irrelevant\n"
                  "CS = constitution x safety\n"
                  "SI = safety x irrelevant\n(same-layer cosines)")
        axes[0][0].set_ylabel("contrast (cosine units)", fontsize=8,
                              color=TXT2)
        rep = results.get("replication") or {}
        lines = ["Headline contrasts (D14a, interpretation per D20): "
                 "dL = long-pair excess (CI - (CS+SI)/2), "
                 "dS = mixed-pair difference (CS - SI); same-layer "
                 "diagonals, response leg, psr_kl_all.",
                 _mixed_pair_length_line(grid_models, token_meta),
                 "shaded: +/-2*sigma*sqrt(2) isotropic reference (the wider "
                 "contrast sigma; descriptive, not inferential -- D12/D14e)."]
        lines = [ln for ln in lines if ln]
        if rep.get("verdict"):
            met = rep["verdict"] == "two_sign_check_met"
            lines.append("preregistered two-sign check on the held-out "
                         "model (sign of both contrasts): "
                         + ("met" if met else "NOT met")
                         + " -- a sign criterion only; it does not "
                           "identify a mechanism (D20d)")
        for mk in grid_models:
            lines.append(f"{mk}: "
                         + _si_token_footer(mk, token_meta, si_meta_rec,
                                            SI_ORDER)
                         + f" | sha {sha_line(mk)}")
        footer(fig, lines)
        fig.subplots_adjust(bottom=0.30, top=0.82, left=0.07, right=0.98)
        save(fig, "final_contrasts_llama8b_gemma4e4b.png")

    # ---- 2. cross-SI same-layer diagonals ----
    if grid_models:
        fig, axes = plt.subplots(1, len(grid_models),
                                 figsize=(5.6 * len(grid_models), 4.6),
                                 sharey=True, squeeze=False)
        fig.patch.set_facecolor(SURFACE)
        for ax, mk in zip(axes[0], grid_models):
            style_axes(ax)
            lo = next(iter(by_model[mk].values()))
            layers = lo.layers
            sigma = chance_sigma("direction", lo.d_model)
            # F2: the traces are INDIVIDUAL per-layer entries, so the band
            # must be a per-entry null -- a distribution of block MEANS is
            # ~sqrt(k) narrower and makes ordinary layers look
            # exceptional. Pool = CONCATENATE the three cross-SI blocks'
            # off-diagonal entries per depth region, then take quantiles;
            # mid-depth layers (neither early nor late) wear the all-layer
            # pool.
            early_m, late_m = region_masks(layers)
            pms, plv = [], []
            for pair in PAIR_COLORS:
                key = (mk,) + pair + ("direction",)
                if key in mats:
                    pms.append(mats[key])
                    plv.append(by_model[mk][pair[0]].live_dir
                               & by_model[mk][pair[1]].live_dir)
            band = pooled_offdiag_band(pms, plv, layers) if pms else {}
            lo_band, hi_band, have_band = [], [], False
            for i in range(len(layers)):
                nm = ("early" if early_m[i]
                      else "late" if late_m[i] else "all")
                bd = band.get(nm) or band.get("all")
                if bd is None:
                    lo_band.append(np.nan)
                    hi_band.append(np.nan)
                else:
                    lo_band.append(bd[0])
                    hi_band.append(bd[1])
                    have_band = True
            if have_band:
                ax.fill_between(layers, lo_band, hi_band, color=GRID,
                                alpha=0.7, linewidth=0, step="mid",
                                zorder=0,
                                label="empirical null band: 2.5-97.5% of "
                                      "pooled off-diagonal entries")
            ax.axhline(0, color=TXT2, linewidth=0.8)
            ax.axhline(2 * sigma, color=TXT2, linewidth=0.7,
                       linestyle="--", alpha=0.6,
                       label="isotropic +/-2 sigma reference "
                             "(descriptive)")
            ax.axhline(-2 * sigma, color=TXT2, linewidth=0.7,
                       linestyle="--", alpha=0.6)
            for pair, color in PAIR_COLORS.items():
                b = blocks_by.get((mk,) + pair)
                if not b:
                    continue
                d = np.array([_nan(b["same_layer_diagonal"][str(l)])
                              for l in layers])
                ax.plot(layers, d, color=color, linewidth=2,
                        label=PAIR_LABELS[pair])
            ax.set_title(mk, fontsize=9, color=TXT)
            ax.set_xlabel("layer", fontsize=8, color=TXT2)
            depth_axis(ax, len(layers))
        series_legend(axes[0][0], loc="upper right")
        axes[0][0].set_ylabel("same-layer cosine", fontsize=8, color=TXT2)
        lines = ["Cross-SI same-layer diagonals: CI = constitution x "
                 "irrelevant (co-headline -- approximately length-matched "
                 "pair), CS = constitution x safety, SI = safety x "
                 "irrelevant. response leg, psr_kl_all.",
                 "shaded: per-entry empirical null -- 2.5-97.5% of "
                 "INDIVIDUAL off-diagonal entries, pooled by concatenating "
                 "the three cross-SI blocks, per fractional-depth region "
                 "(early/late; mid-depth layers wear the all-layer pool); "
                 "dashed: isotropic +/-2 sigma reference (descriptive)."]
        for mk in grid_models:
            lines.append(f"{mk}: "
                         + _si_token_footer(mk, token_meta, si_meta_rec,
                                            SI_ORDER)
                         + f" | sha {sha_line(mk)}")
        footer(fig, lines)
        fig.subplots_adjust(bottom=0.30, top=0.86, left=0.07, right=0.98)
        save(fig, "final_diag_cross_si_llama8b_gemma4e4b.png")

    # ---- heatmap helpers ----
    def draw_heatmap(ax, M, norm, layers):
        im = ax.imshow(np.ma.masked_invalid(M), origin="lower", cmap=cmap,
                       norm=norm, interpolation="nearest",
                       extent=(-0.5, len(layers) - 0.5,
                               -0.5, len(layers) - 0.5))
        ax.tick_params(colors=TXT2, labelsize=7)
        for s in ax.spines.values():
            s.set_color(GRID)
        return im

    KIND_QTY = {"direction": "direction cosine similarity",
                "gate_w": "gate_w cosine similarity",
                "jacobian": "rank-one Jacobian cosine similarity",
                "offleg": "direction cosine similarity"}

    def draw_cbar(fig, im, V, sigma, kind, cax=None, ax=None):
        """The colour legend, complete on the bar itself: the bar axis is
        asinh-scaled (matplotlib draws it under the image's norm), so the
        nonlinear spacing is shown honestly -- ticks at the values a
        reader cares about (0, +/-2 sigma chance, a decade ladder, the
        true +/-V endpoints) land at their true compressed positions.
        Plus an axis label naming the quantity and the sign convention,
        ONE labelled hatched band marking the +/-2 sigma chance region
        (near-linear and near-white by construction of the norm), and a
        swatch for the masked colour (trivial unit diagonal / dead
        layer)."""
        from matplotlib.patches import Rectangle
        cb = (fig.colorbar(im, cax=cax) if cax is not None
              else fig.colorbar(im, ax=ax))
        ticks = _asinh_ticks(V, im.norm.linear_width, 2 * sigma)
        cb.set_ticks(ticks)
        cb.set_ticklabels([_fmt_tick(t) for t in ticks])
        cb.ax.minorticks_off()
        cb.ax.tick_params(colors=TXT2, labelsize=7)
        cb.outline.set_edgecolor(GRID)
        cb.set_label(f"{KIND_QTY[kind]} (red = aligned, blue = opposed)",
                     fontsize=7.5, color=TXT2)
        cb.ax.axhspan(-2 * sigma, 2 * sigma, facecolor="none",
                      edgecolor=TXT2, hatch="/////", linewidth=0.7)
        cb.ax.text(-0.55, 0.0,
                   f"chance band +/-2 sigma = +/-{2 * sigma:.4g}",
                   rotation=90, va="center", ha="center", color=TXT2,
                   fontsize=6.5, transform=cb.ax.get_yaxis_transform())
        cb.ax.add_patch(Rectangle((0.0, -0.045), 1.0, 0.03,
                                  transform=cb.ax.transAxes,
                                  facecolor=MASK_GRAY, edgecolor=GRID,
                                  linewidth=0.6, clip_on=False))
        cb.ax.text(0.5, -0.055, "masked:\ntrivial unit\ndiag / dead\nlayer",
                   transform=cb.ax.transAxes, ha="center", va="top",
                   color=TXT2, fontsize=6.5)
        return cb

    def jac_norm_panel(ax, mk, sis):
        style_axes(ax)
        for s in sis:
            lo = by_model[mk][s]
            jn = np.array([lo.record["jacobian_norms"][str(l)]
                           if lo.record["jacobian_norms"][str(l)] is not None
                           else np.nan for l in lo.layers])
            color = {"safety_refusal": AQUA, "si_constitution": ORANGE,
                     "si_irrelevant": BLUE}[s]
            ax.plot(lo.layers, jn, color=color, linewidth=1.6, label=s)
        ax.set_ylabel("||d||*||w||", fontsize=7, color=TXT2)
        ax.set_xlabel("layer", fontsize=7, color=TXT2)
        ax.legend(fontsize=6.5, frameon=False, ncol=len(sis),
                  labelcolor=TXT2)

    def heatmap_kind(kind, prefix):
        V = scale_by_kind[kind]["V"]
        for mk in grid_models:
            objs = by_model[mk]
            layers = objs[SI_ORDER[0]].layers
            sigma = chance_sigma(kind, objs[SI_ORDER[0]].d_model)
            norm = _heat_norm(V, sigma)
            fig = plt.figure(figsize=(11.5, 13.2))
            fig.patch.set_facecolor(SURFACE)
            gs = fig.add_gridspec(4, 4, width_ratios=[1, 1, 1, 0.05],
                                  height_ratios=[1, 1, 1, 0.4],
                                  hspace=0.32, wspace=0.28)
            im = None
            for r, sa in enumerate(SI_ORDER):
                for c, sb in enumerate(SI_ORDER):
                    ax = fig.add_subplot(gs[r, c])
                    key = (mk, sa, sb, kind)
                    tkey = (mk, sb, sa, kind)
                    M = (mats[key] if key in mats else mats[tkey].T)
                    im = draw_heatmap(ax, _display_matrix(M, sa == sb),
                                      norm, layers)
                    if r == 0:
                        ax.set_title(sb, fontsize=8, color=TXT)
                    if c == 0:
                        ax.set_ylabel(f"{sa}\nlayer", fontsize=8, color=TXT2)
                    if r == 2:
                        ax.set_xlabel("layer", fontsize=8, color=TXT2)
            cax = fig.add_subplot(gs[0:3, 3])
            draw_cbar(fig, im, V, sigma, kind, cax=cax)
            jac_norm_panel(fig.add_subplot(gs[3, 0:3]), mk, SI_ORDER)
            fig.suptitle(f"{mk} -- {kind} cosine blocks (3x3 SI grid), "
                         f"response leg, psr_kl_all", fontsize=11,
                         color=TXT, y=0.995)
            lines = [
                f"colour scale: NONLINEAR symmetric asinh -- near-linear "
                f"inside +/-{norm.linear_width:.4g} (= this model's "
                f"isotropic 2 sigma chance scale for {kind} blocks), "
                "compressive above -- so at-chance cells stay near-white "
                "and the above-chance range gets real colour separation; "
                f"endpoints V=+/-{V:.3f} = the true max |value| over "
                f"every drawn {kind} cell (shared across every {kind} "
                "heatmap), so no drawn cell is clipped.",
                "the within-SI unit diagonal (the grid's own diagonal "
                "blocks at i == j) is 1.0 by construction -- a sanity "
                "check asserted in code, not a finding -- and is masked "
                "from the drawing and the scale; mid-gray = masked "
                "(trivial unit diagonal or dead layer).",
                "colorbar: asinh-scaled, ticks at their "
                "true nonlinear positions; hatched band marks this "
                f"model's isotropic +/-2 sigma = +/-{2 * sigma:.4g} "
                "(descriptive); red = aligned, blue = opposed.",
                "beneath: gauge-invariant ||d_l||*||w_l|| (raw norms are "
                "gauge-dependent and never compared across objects -- "
                "D14b).",
                _si_token_footer(mk, token_meta, si_meta_rec, SI_ORDER)
                + f" | sha {sha_line(mk)}"]
            if kind == "jacobian":
                lines.insert(0, "rank-one Jacobian cosine: cos(d_A w_A^T, "
                             "d_B w_B^T) = cos(d_A, d_B) * cos(w_A, w_B) "
                             "(exact factorisation, D14b); as a product of "
                             "two independent cosines its isotropic chance "
                             "scale is 1/d, not 1/sqrt(d).")
            footer(fig, lines)
            fig.subplots_adjust(bottom=0.11)
            save(fig, f"final_{prefix}_grid_{mk}.png")
        for mk in safety_models:
            lo = by_model[mk]["safety_refusal"]
            layers = lo.layers
            sigma = chance_sigma(kind, lo.d_model)
            norm = _heat_norm(V, sigma)
            fig = plt.figure(figsize=(6.8, 8.2))
            fig.patch.set_facecolor(SURFACE)
            gs = fig.add_gridspec(2, 2, width_ratios=[1, 0.05],
                                  height_ratios=[1, 0.35],
                                  hspace=0.34, wspace=0.22)
            ax = fig.add_subplot(gs[0, 0])
            M = mats[(mk, "safety_refusal", "safety_refusal", kind)]
            im = draw_heatmap(ax, _display_matrix(M, True), norm, layers)
            ax.set_xlabel("layer", fontsize=8, color=TXT2)
            ax.set_ylabel("layer", fontsize=8, color=TXT2)
            draw_cbar(fig, im, V, sigma, kind,
                      cax=fig.add_subplot(gs[0, 1]))
            jac_norm_panel(fig.add_subplot(gs[1, 0]), mk, ["safety_refusal"])
            fig.suptitle(f"{mk} -- safety_refusal x safety_refusal "
                         f"({kind}), response leg, psr_kl_all",
                         fontsize=10, color=TXT, y=0.99)
            footer(fig, [
                f"colour scale: NONLINEAR symmetric asinh -- near-linear "
                f"inside +/-{norm.linear_width:.4g} (= this model's "
                f"isotropic 2 sigma chance scale for {kind} blocks), "
                f"compressive above; endpoints V=+/-{V:.3f} = the true "
                f"max |value| over every drawn {kind} cell (shared with "
                f"every {kind} heatmap), so no drawn cell is clipped; "
                "the unit diagonal (1.0 by construction, asserted -- a "
                "sanity check, not a finding) is masked from the drawing "
                "and the scale.",
                "colorbar: asinh-scaled, ticks at their "
                "true nonlinear positions; hatched band = isotropic "
                f"+/-2 sigma = +/-{2 * sigma:.4g} (descriptive); red = "
                "aligned, blue = opposed; mid-gray = masked (trivial "
                "unit diagonal or dead layer).",
                _si_token_footer(mk, token_meta, si_meta_rec,
                                 ["safety_refusal"])
                + f" | sha {lo.sha256[:8]}"])
            fig.subplots_adjust(bottom=0.14)
            save(fig, f"final_{prefix}_safety_{mk}.png")

    # ---- 3/4. direction heatmaps; 7. gate_w appendix; 8. jacobian ----
    heatmap_kind("direction", "heatmap")
    heatmap_kind("gate_w", "gatew")
    heatmap_kind("jacobian", "jac")

    # ---- 6. per-model gauge-invariant norm/gate figures ----
    from matplotlib.lines import Line2D
    for mk in model_order:
        sis = [s for s in SI_ORDER if s in by_model[mk]]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.6), sharex=True)
        fig.patch.set_facecolor(SURFACE)
        fired_flags: dict[str, str] = {}
        for ax, field, ylab in ((ax1, "jacobian_norms",
                                 "||d_l|| * ||w_l|| (gauge-invariant)"),
                                (ax2, "gate_thresholds",
                                 "b_l / ||w_l|| (gauge-invariant)")):
            style_axes(ax)
            for s in sis:
                lo = by_model[mk][s]
                vals = np.array([lo.record[field][str(l)]
                                 if lo.record[field][str(l)] is not None
                                 else np.nan for l in lo.layers])
                color = {"safety_refusal": AQUA, "si_constitution": ORANGE,
                         "si_irrelevant": BLUE}[s]
                ax.plot(lo.layers, vals, color=color, linewidth=1.8, label=s)
                frozen = sorted({l for ls in
                                 lo.record["frozen_gate_b_groups"].values()
                                 for l in ls})
                flag_sets = (
                    (frozen, "o", "frozen gate_b"),
                    (lo.record["dead_layers"], "x", "dead"),
                    (lo.record["near_dead_layers"], "P", "near-dead"),
                    (lo.record["provably_dead_gate_layers"], "^",
                     "provably-dead gate"),
                    (lo.record["constant_gate_layers"], "s",
                     "constant gate"))
                for lset, marker, flag_label in flag_sets:
                    if lset:
                        li = [lo.layers.index(l) for l in lset]
                        ax.plot(np.array(lo.layers)[li], vals[li], marker,
                                color=color, markersize=7,
                                markerfacecolor="none")
                        fired_flags[flag_label] = marker
            ax.set_ylabel(ylab, fontsize=8, color=TXT2)
        # Every colour/marker encoding gets a legend entry: series always;
        # flag markers whenever any actually fired (none on today's bank).
        handles, _ = ax1.get_legend_handles_labels()
        handles += [Line2D([], [], linestyle="none", marker=m, color=TXT2,
                           markerfacecolor="none", markersize=7,
                           label=f"{lab} (flagged layer)")
                    for lab, m in sorted(fired_flags.items())]
        ax1.legend(handles=handles, fontsize=7, frameon=False,
                   ncol=len(sis), labelcolor=TXT2)
        ax2.set_xlabel("layer", fontsize=8, color=TXT2)
        depth_axis(ax1, len(by_model[mk][sis[0]].layers))
        fig.suptitle(f"{mk} -- gauge-invariant direction/gate health "
                     "(D11/D14b), response leg, psr_kl_all",
                     fontsize=10, color=TXT)
        footer(fig, [
            "raw ||d||, ||gate_w|| and gate_b are gauge-dependent under "
            "(w,b,d) -> (cw,cb,d/c) and are never compared across objects "
            "(D14b); open markers = flagged layers (none expected on "
            "today's bank).",
            "b/||w|| is the scale-invariant gate bias, NOT a signed "
            "threshold: the gate fires where w.h + b > 0, so the decision "
            "boundary along w-hat sits at -b/||w|| (D20f); what it means "
            "for firing depends on the distribution of w.h on real hidden "
            "states, unseen here.",
            _si_token_footer(mk, token_meta, si_meta_rec, sis)
            + f" | sha {sha_line(mk)}"])
        fig.subplots_adjust(bottom=0.17, top=0.88)
        save(fig, f"final_norms_{mk}.png")

    # ---- 9. gate_b departure small multiples (raw-parameter freeze
    # diagnostics -- the ONE deliberately gauge-dependent figure) ----
    n_models = len(model_order)
    ncols = 3
    nrows = math.ceil(n_models / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.2 * ncols, 3.2 * nrows),
                             squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for i, mk in enumerate(model_order):
        ax = axes[i // ncols][i % ncols]
        style_axes(ax)
        any_frozen = False
        for s in [s for s in SI_ORDER if s in by_model[mk]]:
            lo = by_model[mk][s]
            dep = lo.B - 1.0
            color = {"safety_refusal": AQUA, "si_constitution": ORANGE,
                     "si_irrelevant": BLUE}[s]
            ax.plot(lo.layers, dep, color=color, linewidth=1.5, label=s)
            frozen = sorted({l for ls in
                             lo.record["frozen_gate_b_groups"].values()
                             for l in ls})
            if frozen:
                li = [lo.layers.index(l) for l in frozen]
                ax.plot(np.array(lo.layers)[li], dep[li], "o", color=color,
                        markersize=7, markerfacecolor="none")
                any_frozen = True
        ax.axhline(0, color=TXT2, linewidth=0.7)
        ax.set_title(mk, fontsize=9, color=TXT)
        ax.set_xlabel("layer", fontsize=7, color=TXT2)
        ax.set_ylabel("gate_b - 1.0", fontsize=7, color=TXT2)
        handles, _ = ax.get_legend_handles_labels()
        if any_frozen:
            handles.append(Line2D([], [], linestyle="none", marker="o",
                                  color=TXT2, markerfacecolor="none",
                                  markersize=7,
                                  label="frozen-gate_b group layer"))
        ax.legend(handles=handles, fontsize=6, frameon=False,
                  labelcolor=TXT2)
    for j in range(n_models, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle("gate_b departure from init (raw parameter -- freeze "
                 "diagnostics ONLY; gauge-dependent, not comparable across "
                 "objects, D14b)", fontsize=10, color=TXT)
    footer(fig, ["open circles = layers in a frozen-gate_b group "
                 "(psr_frozen_gate_b); none fire on today's bank (D11)."])
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    save(fig, "final_gateb_departure_all.png")

    # ---- off-leg quarantine figures (separately labelled) ----
    for rec in results.get("off_leg_objects", []):
        key = (rec["model_key"], rec["si"], rec["si"], "offleg")
        if key not in mats:
            continue
        M = mats[key]
        layers = rec["layers"]
        sigma = rec["chance_sigma"]
        fig, ax = plt.subplots(figsize=(6.4, 6.2))
        fig.patch.set_facecolor(SURFACE)
        offd = _display_matrix(M, True)
        finite = offd[np.isfinite(offd)]
        V = float(np.abs(finite).max()) if finite.size else 0.05
        norm = _heat_norm(V, sigma)
        im = draw_heatmap(ax, offd, norm, layers)
        draw_cbar(fig, im, V, sigma, "offleg", ax=ax)
        ax.set_title(f"QUARANTINED (off-leg: {rec['leg']!r}, not "
                     f"{EXPECTED_LEG!r}) -- {rec['model_key']} {rec['si']}",
                     fontsize=9, color=POLE_RED)
        footer(fig, [f"excluded from every response-leg matrix and summary; "
                     f"sha {rec['sha256'][:8]}; isotropic sigma "
                     f"{sigma:.4f}; NONLINEAR symmetric asinh colour "
                     f"scale (near-linear inside the +/-2 sigma chance "
                     "width, compressive above), endpoints at this "
                     "block's max |off-diagonal cos| -- nothing clipped; "
                     "the unit diagonal (1.0 by construction) is masked "
                     "from the drawing and the scale."])
        fig.subplots_adjust(bottom=0.12)
        save(fig, f"final_offleg_{rec['model_key']}_{rec['si']}.png")

    return written


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

def analyse(loaded: list[LoadedObject], out_dir: Path, si_meta_by_id: dict
            ) -> tuple[dict, dict]:
    """Compute every block/contrast record + matrices. Returns
    (results-dict-without-token-meta, matrices dict)."""
    mats: dict = {}
    blocks: list[dict] = []
    blocks_gw: list[dict] = []
    blocks_jac: list[dict] = []
    contrasts: list[dict] = []

    by_model: dict[str, dict[str, LoadedObject]] = {}
    model_order: list[str] = []
    for lo in loaded:
        if lo.model_key not in by_model:
            model_order.append(lo.model_key)
        by_model.setdefault(lo.model_key, {})[lo.si] = lo

    for mk in model_order:
        objs = by_model[mk]
        present = [s for s in SI_ORDER if s in objs]
        # Same-model-group identity invariant (review fix): every cross-SI
        # "same-layer" statistic is POSITIONAL (np.diag pairs row indices,
        # not layer numbers), so before any cross-SI block is formed the
        # whole group must share one identity frame -- identical HF model
        # id, identical pinned revision, identical d_model, identical layer
        # list. A mismatch on the full-grid target is a real defect, never
        # a case to accommodate: objects with layers [0,2,5] vs [0,3,5]
        # would silently report layer 2 against layer 3 as "same-layer".
        ref = objs[present[0]]
        for s in present[1:]:
            o = objs[s]
            for field, va, vb in (
                    ("model_id", ref.record["model_id"],
                     o.record["model_id"]),
                    ("model_revision", ref.record["model_revision"],
                     o.record["model_revision"]),
                    ("d_model", ref.d_model, o.d_model),
                    ("layers", ref.layers, o.layers)):
                if va != vb:
                    raise IdentityError(
                        f"{mk}: same-model group identity mismatch on "
                        f"{field}: {present[0]}={va!r} vs {s}={vb!r} -- "
                        "cross-SI same-layer statistics are positional and "
                        "would silently pair different layers/coordinates")
        layers = objs[present[0]].layers
        # direction-kind sigma; the contrasts are direction-block derived
        sigma = chance_sigma("direction", objs[present[0]].d_model)
        for a, b in [(x, y) for i, x in enumerate(present)
                     for y in present[i:]]:
            la, lb = objs[a], objs[b]
            live_dir = la.live_dir & lb.live_dir
            live_gw = la.live_gw & lb.live_gw
            M_dir = cosine_matrix(la.D, lb.D, la.live_dir, lb.live_dir)
            M_gw = cosine_matrix(la.W, lb.W, la.live_gw, lb.live_gw)
            M_jac = M_dir * M_gw  # exact factorisation
            if a == b:
                # Within-SI unit diagonal: sanity check, not a finding.
                for kindM, klive in ((M_dir, la.live_dir), (M_gw, la.live_gw)):
                    d = np.diag(kindM)[klive]
                    if d.size and np.abs(d - 1.0).max() >= 1e-9:
                        raise IdentityError(
                            f"{mk}/{a}: within-SI diagonal deviates from 1 "
                            f"by {np.abs(d - 1.0).max():.3e}")
            for kind, M, live, blist in (
                    ("direction", M_dir, live_dir, blocks),
                    ("gate_w", M_gw, live_gw, blocks_gw),
                    ("jacobian", M_jac, live_dir & live_gw, blocks_jac)):
                mats[(mk, a, b, kind)] = M
                fname = f"{MATRIX_PREFIX[kind]}{mk}__{a}__x__{b}.npy"
                blist.append(block_record(
                    mk, a, b, M, layers, live,
                    chance_sigma(kind, objs[present[0]].d_model),
                    la.sha256, lb.sha256, f"matrices/{fname}"))
        cross = {(b["si_a"], b["si_b"]): b for b in blocks
                 if b["model_key"] == mk and b["si_a"] != b["si_b"]}
        pair_mats = {
            "cs": mats.get((mk, "safety_refusal", "si_constitution",
                            "direction")),
            "si": mats.get((mk, "safety_refusal", "si_irrelevant",
                            "direction")),
            "ci": mats.get((mk, "si_constitution", "si_irrelevant",
                            "direction")),
        }
        con = compute_contrasts(mk, cross, layers, sigma, pair_mats)
        if con is not None:
            contrasts.append(con)

    # Two-sign check verdict: does the SIGN of both contrasts on the
    # held-out confirmatory model match the exploratory one? A sign
    # criterion only -- it identifies no mechanism, so the verdict enum
    # names the criterion, never a "replication".
    replication = None
    con_by = {c["model_key"]: c for c in contrasts}
    if "llama8b" in con_by and "gemma4e4b" in con_by:
        ex, cf = con_by["llama8b"], con_by["gemma4e4b"]
        signs_ok = (ex["dL"]["sign_of_mean"] == cf["dL"]["sign_of_mean"]
                    and ex["dS"]["sign_of_mean"] == cf["dS"]["sign_of_mean"]
                    and None not in (ex["dL"]["sign_of_mean"],
                                     ex["dS"]["sign_of_mean"]))
        replication = {
            "protocol": TWO_SIGN_CHECK_PROTOCOL,
            "exploratory_model": "llama8b",
            "confirmatory_model": "gemma4e4b",
            "llama8b": {"dL_mean": ex["dL"]["mean"],
                        "dS_mean": ex["dS"]["mean"],
                        "signs": [ex["dL"]["sign_of_mean"],
                                  ex["dS"]["sign_of_mean"]]},
            "gemma4e4b": {"dL_mean": cf["dL"]["mean"],
                          "dS_mean": cf["dS"]["mean"],
                          "signs": [cf["dL"]["sign_of_mean"],
                                    cf["dS"]["sign_of_mean"]]},
            "verdict": ("two_sign_check_met" if signs_ok
                        else "two_sign_check_not_met"),
        }

    summary = {}
    for mk in model_order:
        cross_b = [b for b in blocks if b["model_key"] == mk
                   and b["si_a"] != b["si_b"]]
        entry = {f"{b['si_a']}__x__{b['si_b']}": {
            "diag_mean": b["diag_mean"],
            "diag_mean_sigma_units": b["diag_mean_sigma_units"],
            "diag_mean_se": b["diag_mean_se"],
            "diag_max": b["diag_max"],
            "diag_max_layer": b["diag_max_layer"]} for b in cross_b}
        if mk in con_by:
            entry["contrasts"] = {
                "dL_mean": con_by[mk]["dL"]["mean"],
                "dS_mean": con_by[mk]["dS"]["mean"]}
        summary[mk] = entry

    results = {
        "leg": EXPECTED_LEG,
        "arm": ARM,
        "si_order": SI_ORDER,
        "si_meta": {s: {k: v for k, v in m.items() if k != "text"}
                    for s, m in si_meta_by_id.items()},
        "model_key_to_hf": dict(sorted(MODEL_KEY_TO_HF.items())),
        "objects": [lo.record for lo in loaded],
        "blocks": blocks,
        "blocks_gate_w": blocks_gw,
        "blocks_jacobian": blocks_jac,
        "contrasts": contrasts,
        "replication": replication,
        "summary": summary,
        "notes": {
            "sigma": ("all sigma-units are DESCRIPTIVE, not inferential: "
                      "the isotropic baseline assumes isotropy and layer "
                      "independence, and it is KIND-SPECIFIC -- N(0, 1/d) "
                      "(sigma 1/sqrt(d)) for direction and gate_w cosine "
                      "blocks, sigma 1/d for the rank-one Jacobian PRODUCT "
                      "blocks (the product of two independent cosines; "
                      "F4). The empirical null in each block record -- "
                      "all-layer and per fractional-depth region, with "
                      "observed percentiles -- is the primary reference, "
                      "and the derived contrasts carry their own shared-"
                      "mapping nulls (D12/D14e)"),
            "gauge": ("per_layer_norms, gate_w_norms and gate_b are "
                      "gauge-dependent under (w,b,d) -> (cw,cb,d/c) and "
                      "must never be compared across objects; use "
                      "jacobian_norms, gate_thresholds and the jacobian "
                      "blocks (D14b)"),
            "n": ("n = 1 fitted object per model x SI; layers are not "
                  "independent replicates -- every conclusion is "
                  "artifact-level (D14h)"),
        },
    }
    return results, mats


MATRIX_PREFIX = {"direction": "", "gate_w": "gatew__",
                 "jacobian": "jac__", "offleg": "offleg__"}


def write_outputs(out_dir: Path, results: dict, mats: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "matrices").mkdir(exist_ok=True)
    for (mk, a, b, kind), M in sorted(mats.items()):
        np.save(out_dir / "matrices"
                / f"{MATRIX_PREFIX[kind]}{mk}__{a}__x__{b}.npy", M)
    diagonals = {}
    for label, blist in (("direction", results["blocks"]),
                         ("gate_w", results["blocks_gate_w"]),
                         ("jacobian", results["blocks_jacobian"])):
        for b in blist:
            key = f"{b['model_key']}|{b['si_a']}|x|{b['si_b']}|{label}"
            diagonals[key] = {
                "raw": b["same_layer_diagonal"],
                "sigma_units": b["same_layer_diagonal_sigma_units"]}
    for name, payload in (("results.json", results),
                          ("diagonals.json", diagonals)):
        (out_dir / name).write_text(
            json.dumps(payload, sort_keys=True, indent=2, allow_nan=False)
            + "\n")


def print_identity_table(loaded: list[LoadedObject],
                         off_leg: list[LoadedObject]) -> None:
    print(f"\n{'model':<10} {'si':<16} {'leg':<9} {'L':>3} {'d':>5} "
          f"{'sha8':<9} {'flavour':<18} {'COMPLETE':<9} {'si words':>8}")
    for lo in loaded + off_leg:
        r = lo.record
        tag = "" if lo in loaded else "  [QUARANTINED off-leg]"
        print(f"{r['model_key']:<10} {r['si']:<16} {r['leg']:<9} "
              f"{len(r['layers']):>3} {r['d_model']:>5} "
              f"{r['sha256'][:8]:<9} {r['sidecar_flavor']:<18} "
              f"{str(r['complete_marker_present']):<9} "
              f"{si_meta(r['si'])['words']:>8}{tag}")


def validate_manifest_cells(manifest: list[dict],
                            expected: list[dict]) -> None:
    """Frozen-manifest gate (review fix): in default/pinned mode the
    manifest must cover EXACTLY the expected unique (model_key, si, arm)
    cells -- missing, extra or duplicate is fail-loud -- and every entry
    must carry the frozen sha256 pin for its cell. A partial or unpinned
    manifest is test-only, behind the explicit --allow-partial flag."""
    cell = lambda e: (e["model_key"], e["si"], e["arm"])  # noqa: E731
    got = Counter(cell(e) for e in manifest)
    want = {cell(e): e.get("sha256") for e in expected}
    problems = []
    dup = sorted(c for c, n in got.items() if n > 1)
    missing = sorted(set(want) - set(got))
    extra = sorted(set(got) - set(want))
    if dup:
        problems.append(f"duplicate cells: {dup}")
    if missing:
        problems.append(f"missing cells: {missing}")
    if extra:
        problems.append(f"extra cells: {extra}")
    unpinned = sorted({cell(e) for e in manifest if not e.get("sha256")})
    if unpinned:
        problems.append(f"unpinned entries (no sha256): {unpinned}")
    wrong_pin = sorted(
        {cell(e) for e in manifest
         if e.get("sha256") and want.get(cell(e))
         and e["sha256"] != want[cell(e)]})
    if wrong_pin:
        problems.append(
            f"entries pinned to a different sha256 than the frozen "
            f"manifest: {wrong_pin}")
    if problems:
        raise IdentityError(
            "frozen-manifest validation failed (the default run must not "
            "exit 0 with the wrong object set): " + "; ".join(problems)
            + " -- partial/unpinned manifests are test-only and require "
              "--allow-partial")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--manifest", type=Path, default=None,
                   help="JSON list of object entries; default = the "
                        "built-in 10-object manifest")
    p.add_argument("--out", type=Path,
                   default=Path("out/psr_direction_cosines"))
    p.add_argument("--figs", type=Path,
                   default=Path.home() / "figs" / "psr-direction-cosines")
    p.add_argument("--eps-dead", type=float, default=1e-6,
                   help="absolute dead-layer norm epsilon")
    p.add_argument("--near-dead-rel", type=float, default=1e-3,
                   help="relative near-dead flag: norm < rel * median")
    p.add_argument("--no-figs", action="store_true",
                   help="matrices/JSON only (test fast path)")
    p.add_argument("--no-tokens", action="store_true",
                   help="skip tokenizer-based SI token counting "
                        "(synthetic manifests only; the real run never "
                        "uses it)")
    p.add_argument("--self-check", action="store_true",
                   help="run ONLY the llama8b regression check and exit "
                        "non-zero on mismatch OR inability to validate")
    p.add_argument("--skip-regression", action="store_true",
                   help="disable the in-run regression gate -- synthetic "
                        "manifests only; the real run never uses it")
    p.add_argument("--allow-partial", action="store_true",
                   help="permit a partial/unpinned manifest and demote an "
                        "off-leg object from fail-loud to quarantine-and-"
                        "continue -- synthetic/test manifests only; the "
                        "real run never uses it")
    return p.parse_args(argv)


def run(argv=None) -> int:
    args = parse_args(argv)
    assert_import_provenance()
    import matplotlib
    print(f"sisteer={sisteer.__file__}")
    print(f"torch={torch.__version__} numpy={np.__version__} "
          f"matplotlib={matplotlib.__version__}")

    if args.self_check:
        manifest = [e for e in DEFAULT_OBJECTS
                    if e["model_key"] == REGRESSION_MODEL]
    elif args.manifest is not None:
        manifest = json.loads(Path(args.manifest).read_text())
    else:
        manifest = DEFAULT_OBJECTS

    if not args.allow_partial:
        expected = (manifest if args.self_check else DEFAULT_OBJECTS)
        validate_manifest_cells(manifest, expected)

    loaded, off_leg = [], []
    for entry in manifest:
        lo = load_entry(entry, args.eps_dead, args.near_dead_rel)
        if lo.leg != entry.get("expect_leg", EXPECTED_LEG):
            warnings.warn(
                f"QUARANTINED: {lo.model_key}/{lo.si} is leg {lo.leg!r}, "
                f"not {entry.get('expect_leg', EXPECTED_LEG)!r} -- excluded "
                "from the response-leg matrix set, recorded in "
                "off_leg_objects (never silently dropped, never mixed)",
                RuntimeWarning, stacklevel=1)
            off_leg.append(lo)
        else:
            loaded.append(lo)
    if not loaded:
        raise IdentityError("no in-leg objects loaded -- nothing to analyse")

    print_identity_table(loaded, off_leg)
    si_ids = sorted({lo.si for lo in loaded + off_leg})
    si_meta_by_id = {s: si_meta(s) for s in si_ids}

    results, mats = analyse(loaded, args.out, si_meta_by_id)
    results["off_leg_objects"] = [lo.record for lo in off_leg]
    # Off-leg objects with a computable within-SI block get a separately
    # labelled matrix (offleg__ prefix -- quarantine, not silence).
    for lo in off_leg:
        M = cosine_matrix(lo.D, lo.D, lo.live_dir, lo.live_dir)
        mats[(lo.model_key, lo.si, lo.si, "offleg")] = M

    if args.self_check:
        rec = run_regression_check(results["blocks"], results["objects"],
                                   results["contrasts"])
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "self_check.json").write_text(
            json.dumps(rec, sort_keys=True, indent=2, allow_nan=False)
            + "\n")
        print_regression(rec)
        return 0 if rec["passed"] else 1

    self_check_rec = None
    have_llama8b_grid = all(
        any(lo.model_key == REGRESSION_MODEL and lo.si == s for lo in loaded)
        for s in SI_ORDER)
    if have_llama8b_grid and not args.skip_regression:
        self_check_rec = run_regression_check(
            results["blocks"], results["objects"], results["contrasts"])
        print_regression(self_check_rec)

    if not args.no_tokens:
        models = {}
        for lo in loaded:
            models[lo.model_key] = (lo.record["model_id"],
                                    lo.record["model_revision"])
        results["token_meta"] = compute_token_meta(
            models, {s: m["text"] for s, m in si_meta_by_id.items()})
    else:
        results["token_meta"] = {}

    write_outputs(args.out, results, mats)
    if self_check_rec is not None:
        (args.out / "self_check.json").write_text(
            json.dumps(self_check_rec, sort_keys=True, indent=2,
                       allow_nan=False) + "\n")
    print(f"\noutputs -> {args.out}")

    if not args.no_figs:
        written = make_figures(args.figs, loaded, mats, results)
        print(f"figures -> {args.figs}: {len(written)} files")
        for w in written:
            print(f"  {w}")

    if self_check_rec is not None and not self_check_rec["passed"]:
        print("REGRESSION GATE FAILED -- outputs written for diagnosis; "
              "exiting non-zero (a deviation is a bug, not a finding)",
              file=sys.stderr)
        return 1
    if off_leg and not args.allow_partial:
        # Quarantine-and-continue is right for an unexpected extra; it is
        # WRONG for a required object (review fix). Diagnostics (off-leg
        # record, offleg__ matrix, quarantine figure) are already written
        # above -- now fail loudly.
        names = ", ".join(f"{lo.model_key}/{lo.si} (leg={lo.leg!r})"
                          for lo in off_leg)
        print(f"REQUIRED OBJECT(S) ON THE WRONG LEG: {names} -- quarantine "
              "diagnostics written; exiting non-zero (a required object on "
              "the wrong fit leg is a defect, not a case to accommodate)",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(run())
