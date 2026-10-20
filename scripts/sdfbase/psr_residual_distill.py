#!/usr/bin/env python
"""Build the round-2 PSR decomposition objects:

  V        — the coherent virtual teacher ``psr_paired_write`` = +P_mix − P_ctl
             (compose-v: CPU, model-free, parents identity-gated);
  P_resid  — a fresh single-gate all-layer ``psr`` object DISTILLED to
             reproduce V's closed-loop trajectory (distill: GPU), with the
             CALIBRATED recovery gate (calibrate: GPU).

Subcommands (``all`` runs them in one process, one model load — the
derived Modal app's in-container command):

  compose-v     parents <root>/<key>/cells_all/{<mix_si>,<ctl_si>}__psr_ce_all
                -> <out_root>/psr_paired_write/<key>/<mix_si>__psr_ce_all_minus_ctl/
                   {object.pt, derived_meta.json}
  capture-set   the stratified union of the two parents' pools (source ×
                family, ~10%/stratum doc-level holdout, thin strata flagged)
                -> capture_manifest.json
  calibrate     numeric floor (bare/bare, two micro-batch compositions),
                planted control C at the parents' write norms, the real-model
                teacher identity check R(mix ⊕ C, mix) ≡ +C, then the two
                recoverable controls (a) R(mix, ∅) and (b) R(mix ⊕ C, mix)
                distilled with the PRODUCTION trainer/budget/lr
                -> calibration_controls.json (ED_attain per stratum)
  identity-check the identity-only preflight (no floor, no distillation):
                planted C -> the real-model identity check R(mix ⊕ C, mix)
                ≡ +C (write identity AND hook contract, composite and
                reference) -> the hook contract of the PERSISTED V reloaded
                from disk -> identity_check_report.json; exit 2 on any problem
  distill       teacher pre-check (the identity check again, PLUS the hook
                contract of V itself, refusing to start on disagreement) ->
                production distillation of V ->
                ED table with doc-level bootstrap CIs -> recovery verdict
                -> <out_root>/psr_residual/<key>/<mix_si>__psr_ce_all_minus_ctl_distilled/
                   {object.pt (kind psr), derived_meta.json, capture_manifest.json,
                    report.json, doc_records.jsonl, calibration.json}
  verify-v      CPU, model-free: the persisted V re-admitted against the
                live bank through the shared derived-object verdict (a
                torn publish or a parent refit under the same cell id is a
                problem; exit 2) — what the derived app's compose stage asks
                before letting an existing V stand.
  merge-calibration  CPU, model-free: the per-multiplier sweep files that
                separate ``calibrate --lr-mult <m>`` runs wrote (one
                container per multiplier in the derived Modal app) merged
                into ONE calibration_controls.json — identities must agree
                on everything but trainer.lr / trainer.lr_mult, the union of
                multipliers must be exactly the requested sweep — with the
                SAME selection rule the in-process sweep applies.

Identity discipline: parents are ADMITTED through the ONE gate the
derived-object builders share (scripts/sdfbase/parent_admission.py — the
lora_subtract.py gate: COMPLETE marker present, sidecar names the cell, the
shared staged-cell gate ``train_budget.staged_cell_problems`` passes, the
model revision resolves against the bank's store identity with its source
recorded, kind ``psr``); the two parents must agree on model id/revision,
target layers, fit leg and template treatment; every derived artifact this
script writes is re-validated through the shared derived-object verdict
(``sisteer.eval.object_identity``) with the parents re-verified on disk
before it is left on disk.

The recovery gate is only meaningful if attainable and production scores
are measured under ONE identity, so ``calibrate`` stamps a full identity
fingerprint on ``calibration_controls.json`` (model + revision + dtype, both
parents' object/sidecar shas + fit fingerprints, the source pools' shas, the
capture split's holdout fraction / seed / manifest sha, the layer set, every
trainer knob and every measurement setting — :func:`calibration_identity`)
and ``distill`` / ``all`` REFUSE unless the production run's identity equals
it field for field (:func:`require_same_identity`; the pre-review check
compared lr alone). The distilled object's blob
stays ``kind: "psr"`` (loadable everywhere with zero loader changes) and
carries its ``provenance`` block inside the blob; its ``derived_meta.json``
records ``derivation: "psr_residual_mse"`` next to that kind.

Hyperparameters come from the model's fits config (the PSR winner:
``train_kwargs.psr_ce_all.{lr, max_grad_norm, warmup_frac, gate_reg,
accum_steps}`` + ``batch_size_by_arm.psr_ce_all`` + ``max_len``). The
DISTILLATION BUDGET is explicit: ``--distill-epochs-cap`` (default
60) and ``--distill-patience`` (default 5) — the earlier 24 / 2 under-
budgeted the planted controls on both round-2 models (qwen's control (b)
early-stopped at epoch 11 at ED 0.29 / 0.33, under the pre-registered 0.5
floor; gemma's hit the 24-epoch cap at 0.63 with the held-out still
creeping), which made the gate's attainable ED a statement about the
budget, not the teacher. ``--lr-mult`` is the pre-stated 3-point
mini-sweep {0.3, 1, 3}: ``calibrate`` / ``all`` run the controls at EVERY
listed multiplier (one container, sequentially; the sweep file is rewritten
after each multiplier so a late crash keeps the finished ones), record all
of them under ``calibration_controls.json: sweep.per_mult``, and SELECT the
multiplier with the best planted recovery (:data:`SELECTION_RULE`) for the
production distillation — whose ``derived_meta.json`` stamps the selected
multiplier and the whole sweep table. The file's top level IS the selected
multiplier's calibration (the analysis shape is unchanged). The controls
always re-run at the same lr as the production run, so attainable and
production scores are same-trainer by construction; ``distill`` takes the
selected multiplier by default, or ONE explicit ``--lr-mult`` that must be
in the sweep.

Orchestration durability (the production path used to re-run every
calibration fit serially on any retry, against an unchanged 8 h function
timeout):

* ``calibrate`` / ``all`` RESUME a sweep: before running a multiplier they
  read the existing ``calibration_controls.json`` and reuse a finished
  ``sweep.per_mult`` payload whose recorded identity equals this run's
  identity at that multiplier FIELD FOR FIELD (:func:`identity_diffs` — the
  trainer knobs including the budget, both parents, the capture split, the
  measurement settings); a payload of another identity is re-run and
  replaced, and says why. The lr-independent floor / planted C / identity
  check are recomputed every time (minutes; they are the gate).
* every control and the production fit take ``--checkpoint-every N``
  (optimizer steps): the trainer's resumable checkpoints, keyed by the
  identity fingerprint (+ the control's name) so a checkpoint can only ever
  resume the fit it was written by, under ``<out dir>/ckpt/``.
* ``merge-calibration`` is the aggregation step of the derived app's
  one-container-per-multiplier fan-out; ``distill`` is its production-only
  step and REFUSES without a complete, identity-matching sweep on disk.

The word "inexpressible" never appears in anything this script writes
(``residual.assert_no_banned_words`` guards every artifact).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parents[1]
for _p in (REPO / "src", _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
# Sibling scripts own what they own (the lora_subtract.py sibling-import
# convention): gate_liveness.py the gate_liveness.json shapes and writer,
# parent_admission.py the ONE parent-admission gate + the cross-parent match
# + the pool-budget reader. All torch-free at import.
from gate_liveness import paired_liveness_detail, write_liveness  # noqa: E402
from parent_admission import (PARENT_MATCH_FIELDS, admit_parent,  # noqa: E402,F401
                              pool_budget)
from parent_admission import require_parents_match as _parents_match  # noqa: E402
# The two derived-object builders file their registry records the same way:
# lora_subtract.py owns the registry model-id spelling (torch-free at import,
# like the siblings above).
from lora_subtract import registry_model_id  # noqa: E402

CTL_SI = "constv2ctl-100k"
ARM = "psr_ce_all"
KIND_V = "psr_paired_write"
DERIVATION_RESID = "psr_residual_mse"
DEFAULT_ROOT = Path("$STAGE/sdfbase/out")   # under the staging directory; expanded through sisteer.site where used
#: Where the derived app's one-container-per-multiplier calibrate runs put
#: their sweep files: <resid dir>/calib_parts/lrmult<m>/calibration_controls.json
CALIB_PARTS_DIRNAME = "calib_parts"
DEFAULT_POOLS = Path("$STAGE/sdfbase/pools")


def site_path(value, key: str) -> Path:
    """``value`` with its site placeholder (``$STAGE``, the staging directory) expanded through sisteer.site; an unset one
    refuses by name, and a value without one comes back as a path unchanged."""
    from sisteer import site as _site
    return _site.expand_path(value, key=key)
#: the round-2 Modal banks (gemma / qwen) and the Slurm fit lane's newmodels_full bank keys
#: (<bank>/<key>/cells_all/<cell> is the same layout; `_doc` because these targets have no base variant),
#: + the Llama-3.1-8B lane: base + instruct-direct, + the instruct-direct batch-64 lanes,
#: + the qwen-base batch-64 SEED REPLICATES (2026-09-15): the mix-full - ctl2-full pair per seed, from SAME-seed parents
#: + the paper-companion-repo worked example: the FIT checkpoint's registry id (reproduce/models.yaml
#: fit.checkpoint.model_id — every fit-side key carries _base), named by reproduce/04_fit_objects.sh
MODEL_KEYS = ("gemma4e4b_base", "qwen35_4b_base", "nemotron3nano4b_doc", "glm46vflash_doc",
              "llama8b_base", "llama8b_it_doc",
              "gemma4e4b_it_doc_eb64", "qwen35_4b_it_doc_eb64",
              "qwen35_4b_demo_base",
              "qwen35_4b_base_eb64_s1", "qwen35_4b_base_eb64_s2", "qwen35_4b_base_eb64_s3")

#: Revised distillation budget (was 24 / 2, set for tiny data and an
#: expected early stop — it under-budgeted the planted controls on both
#: round-2 models; see the module docstring).
EPOCHS_CAP = 60
PATIENCE = 5
#: The plan's pre-stated lr-multiplier mini-sweep for the calibration controls.
DEFAULT_LR_MULTS = (0.3, 1.0, 3.0)
#: The planted gate must be active on >= 30% of positions.
PLANTED_ACTIVE_FRAC = 0.3


class ResidualError(RuntimeError):
    """Loud failure — never emit a derived object off unproven parents."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path) -> str:
    from sisteer.eval.defense_cell import sha256_file as _s
    return _s(path)


def v_derived_id(mix_si: str) -> str:
    return f"{mix_si}__{ARM}_minus_ctl"


def resid_derived_id(mix_si: str) -> str:
    return f"{mix_si}__{ARM}_minus_ctl_distilled"


def _emission():
    """scripts/sdfbase/registry_emission.py — the ONE owner of the emission evidence:
    sibling file, loaded by path when not importable."""
    mod = sys.modules.get("registry_emission")
    if mod is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "registry_emission", Path(__file__).resolve().parent / "registry_emission.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["registry_emission"] = mod
        spec.loader.exec_module(mod)
    return mod


def _derived_hook_call(obj_dir, meta: dict, *, object_kind: str, derivation: str,
                       script: str, extra: dict | None, emitter=None, fit_provenance: dict | None = None):
    """The call into ``hooks.emit_object_from_sidecar`` for a derived ReLU-steer object (raises on its own
    failures; returns the hook's paths / [] when off / None when the guarded hook swallowed a failure).
    ``emitter``: a callable with the hook's signature writing ONE of its two
    records (``registry_emission.component_emitter(<component>)``) for the partial replay; default the hook."""
    from sisteer.registry import hooks as _rh
    fn = emitter if emitter is not None else _rh.emit_object_from_sidecar
    parent_ids = [(meta.get("parent_pos") or {}).get("object_sha256"),
                  (meta.get("parent_neg") or {}).get("object_sha256")]
    # A DERIVED object's record (vocab.FIT_TARGET_RULES["derived"], the DB CHECK object_fit_target_kind_pairing): objective
    # `not_applicable` (nothing was fit here — the parents carry the ce objective), the derivation record built from the
    # sidecar exactly as the adapter-filed in-paper rows carry it, the writer stated. Before 2026-10-02 this call passed
    # fit_objective="ce" and no params: every record it ever wrote was REFUSED at the registry's DB CHECK after the
    # cluster-side evidence had said "emitted" (the front door's paired write, record_refusal 2026-10-02T01:11Z) — no
    # native derived emission had ever folded.
    stated = {"fit_model_id": registry_model_id(meta.get("model_id")),
              "fit_layers_json": meta.get("layers"),
              # the sidecar's `layers` are the PARENTS' target_layers (the trainers index decoder blocks from 0, model.model.layers[i]; the
              # convention the hook files for every fit it reads): stated beside the explicit list, which the hook never guesses a convention
              # for (demo-import M1b) — before this the record carried the list and NO fit_layer_index_convention
              "fit_layer_index_convention": "zero_based_decoder_block" if isinstance(meta.get("layers"), list) else None,
              "parent_object_ids_json": parent_ids if all(parent_ids) else None,
              "derivation_params_json": _rh.derivation_params_from_derived_meta(meta),
              "fit_writer": script,
              # a replay restates the builder's provenance (the prior record's), never this job's context
              "fit_provenance": dict(fit_provenance) if fit_provenance else None}
    more = {"parent_object_ids": parent_ids,
            "derived_id": meta.get("derived_id"),
            "scale_matched": meta.get("scale_matched"),
            **(extra or {})}
    kwargs = {k: v for k, v in stated.items() if v is not None}
    more = {k: v for k, v in more.items() if v is not None}
    # the campaign: the parents' recorded one (both must agree), an explicit REGISTRY_CAMPAIGN only when it agrees, never a
    # module default (registry_emission.campaign_for_derived; CampaignUnset is recorded as `failed` evidence by the wrappers)
    return fn(
        str(obj_dir), campaign=_emission().campaign_for_derived_or_off(meta),
        object_kind=object_kind,
        paper_label=_rh.paper_label_for(object_kind, "difference"),
        object_role="difference", fit_target_kind="derived",
        fit_objective="not_applicable", derivation=derivation, script=script,
        extra=more or None, **kwargs)


def registry_emit_derived(obj_dir, meta: dict, *, object_kind: str,
                          derivation: str, script: str,
                          extra: dict | None = None, fit_provenance: dict | None = None) -> dict:
    """The results-registry ``object`` + ``artifact`` records for a derived
    ReLU-steer object this script just published: the paired write V is kind
    ``composed`` / ``derivation="composition"`` (paper label
    ``difference_relu``), the distilled object kind
    ``relu_steer_distilled_difference`` /
    ``derivation="residual_distillation"`` (``difference_relu_distilled``);
    both role ``difference``, target kind ``derived``, objective ``ce`` (the
    parents'); the parents' ids go to the hook's own ``parent_object_ids_json``
    column when both are known (a distilled meta names no flat parent list)
    and ride ``extra`` as well. Never raises and never silent: returns the
    EVIDENCE written beside the artefact (``registry_emission.json``:
    emitted / not_emitted / failed + the REGISTRY_EMIT_FAILED marker);
    prints ``registry: object <sha256>`` when a record was filed."""
    obj_dir = Path(obj_dir)
    mod = _emission()
    try:
        campaign = mod.campaign_for_derived_or_off(meta)
    except mod.CampaignUnset as e:
        return mod.failed(obj_dir, meta.get("object_sha256"), script, "derived object", f"CampaignUnset: {e}")
    ev = mod.emit_with_evidence(
        obj_dir, object_id=meta.get("object_sha256"), writer=script, campaign=campaign,
        emit=lambda: _derived_hook_call(obj_dir, meta, object_kind=object_kind, derivation=derivation,
                                        script=script, extra=extra, fit_provenance=fit_provenance),
        what="derived object")
    if ev.get("status") == "emitted":
        print(f"registry: object {meta.get('object_sha256')}", flush=True)
    return ev


def ensure_registry_emission_derived(obj_dir, *, object_kind: str, derivation: str,
                                     script: str, extra: dict | None = None) -> dict:
    """A KEPT derived object (its ``derived_meta.json`` beside it) is not a recorded one: verify the emission
    evidence (keyed by the sidecar's object id; else the object AND artifact records already in the spool) and
    replay what is missing — the whole hook, or only the missing record — never by
    rebuilding or re-distilling (registry_emission.ensure_emitted)."""
    obj_dir = Path(obj_dir)
    meta = json.loads((obj_dir / "derived_meta.json").read_text(encoding="utf-8"))
    oid = meta.get("object_sha256")
    mod = _emission()
    try:
        campaign = mod.campaign_for_derived_or_off(meta)
    except mod.CampaignUnset as e:
        ev = mod.failed(obj_dir, oid, script, "derived object", f"CampaignUnset: {e}")
        return ev
    # a kept-object replay restates the BUILDER's provenance from the prior record (the campaign the stale evidence names, else this
    # one), never this job's context; the dir's fit_provenance.json, when present, wins inside the hook
    prev = mod.read_evidence(obj_dir) or {}
    prior = mod.fit_provenance_from_prior_record(oid, prev.get("campaign") or campaign) or mod.fit_provenance_from_prior_record(oid, campaign)
    ev = mod.ensure_emitted(
        obj_dir, object_id=oid, writer=script, campaign=campaign,
        emit=lambda: _derived_hook_call(obj_dir, meta, object_kind=object_kind, derivation=derivation,
                                        script=script, extra=extra, fit_provenance=prior), what="derived object",
        # Replay only the record the spool lacks, never a second object row
        emit_component=lambda c: _derived_hook_call(obj_dir, meta, object_kind=object_kind, derivation=derivation,
                                                    script=script, extra=extra, emitter=mod.component_emitter(c), fit_provenance=prior))
    if ev.get("status") == "emitted":
        print(f"registry: object {oid}", flush=True)
    return ev


def distilled_extra_from_meta(meta: dict) -> dict:
    """The distilled record's ``extra`` reconstructed from the published sidecar (what the fresh publish stated
    from its live values): the teacher V's id, the selected lr multiplier + how it was selected, the recovery
    verdict."""
    trainer = meta.get("trainer") or {}
    out = {"teacher_object_id": (meta.get("teacher") or {}).get("object_sha256"),
           "lr_mult": trainer.get("lr_mult"),
           "lr_mult_selected_by": trainer.get("lr_mult_selected_by"),
           "recovery_verdict": meta.get("recovery_verdict"),
           "replayed_from_sidecar": True}
    return {k: v for k, v in out.items() if v is not None}


# --------------------------------------------------------------------------- #
# Parents                                                                      #
# --------------------------------------------------------------------------- #


def load_parent_record(root: Path, key: str, si: str) -> tuple[Path, dict]:
    """(object path, provenance block) for one fitted PSR parent through
    the shared admission gate (``parent_admission.admit_parent``, kind
    ``psr``; every refusal a :class:`ResidualError`), plus this builder's
    own requirement — an all-position fit (the paired write is defined on
    all-position parents) — and the parent's fit record for the sidecar."""
    cell_id = f"{si}__{ARM}"
    obj_path, blob, meta, prov = admit_parent(root, key, cell_id, kind="psr",
                                              error=ResidualError)
    if blob.get("positions", "all") != "all":
        raise ResidualError(f"{obj_path} is a masked fit "
                            f"(positions={blob.get('positions')!r}) — the "
                            "paired write is defined on all-position parents")
    prov = {
        **prov,
        "si_id": si,
        "arm": ARM,
        "fit_record": {k: (meta.get("fit_record") or {}).get(k)
                       for k in ("best_epoch", "epochs_run", "best_heldout_ce",
                                 "heldout_noop_ce", "ce_ratio")},
    }
    return obj_path, prov


def require_parents_match(pos_prov: dict, neg_prov: dict) -> None:
    """The shared cross-parent match, named for this builder's object."""
    _parents_match(pos_prov, neg_prov, what="a paired write",
                   error=ResidualError)


def parents_provenance(root, key, mix_si, ctl_si, pools_dir) -> tuple[dict, dict, dict]:
    """(pos_prov, neg_prov, shared) for the pair — identity-gated and
    matched; ``shared`` carries the lifted common identity + scale label."""
    _p, pos = load_parent_record(root, key, mix_si)
    _n, neg = load_parent_record(root, key, ctl_si)
    require_parents_match(pos, neg)
    pos["data_scale"] = pool_budget(pools_dir, mix_si)
    neg["data_scale"] = pool_budget(pools_dir, ctl_si)
    pt, nt = pos["data_scale"]["official_tokens"], neg["data_scale"]["official_tokens"]
    matched = pt is not None and pt == nt
    shared = {
        "model_key": key,
        "model_id": pos["model_id"],
        "model_revision": pos["model_revision"],
        "positions": pos["positions"],
        "response_header_len": pos["response_header_len"],
        "target_layers": pos["target_layers"],
        "template_hash": pos["template_hash"],
        "scale_matched": bool(matched),
        "comparison": (
            "scale_matched (primary: both parents trained at the same "
            "official-token budget)" if matched else
            f"budget_mismatched (descriptive: parents trained at {pt} vs {nt} "
            "official tokens — bounds nothing; the D23 scale limitation)"),
    }
    return pos, neg, shared


# --------------------------------------------------------------------------- #
# compose-v                                                                    #
# --------------------------------------------------------------------------- #


LIVENESS_RECEIPT = "gate_liveness_receipt.json"   # binds <vdir>/gate_liveness.json to V's bytes (sha256 of object.pt); emit-v requires it


def _admission_liveness():
    """The admission's OWN liveness criterion (seedrep/derived_admission.py, the rule the paper-side stager applies to a paired
    write): ``(dead_floor, dead_layers, liveness_min)`` — read from the same place, never re-typed here."""
    try:
        from seedrep.derived_admission import DEFAULT_DEAD_FLOOR, dead_layers, liveness_min
    except ImportError:
        import importlib.util
        spec = importlib.util.spec_from_file_location("sisteer_seedrep_derived_admission", HERE / "seedrep" / "derived_admission.py")
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)  # type: ignore[union-attr]
        DEFAULT_DEAD_FLOOR, dead_layers, liveness_min = mod.DEFAULT_DEAD_FLOOR, mod.dead_layers, mod.liveness_min
    return DEFAULT_DEAD_FLOOR, dead_layers, liveness_min


def liveness_verdict(gl: dict) -> dict:
    """The admission's verdict over a liveness map: ``not_fully_dead`` (every layer's live fraction > the dead floor), the dead layers,
    the floor, the min and its layer — what the receipt records and what emit-v / the front door require."""
    floor, dead_layers, liveness_min = _admission_liveness()
    dead = list(dead_layers(gl, floor))
    mn, layer = liveness_min(gl)
    return {"dead_floor": floor, "dead_layers": dead, "not_fully_dead": not dead, "liveness_min": mn, "liveness_min_layer": layer,
            "liveness_rule": "D69 (seedrep/derived_admission.py): admitted only when no layer's live fraction is <= the dead floor"}


def compose_v(root: Path, out_root: Path, key: str, mix_si: str,
              ctl_si: str = CTL_SI, pools_dir: Path = DEFAULT_POOLS, emit: bool = True) -> Path:
    """Write V's artifact + sidecar; validate through the shared derived
    verdict with the parents re-verified on disk. ``emit=False`` DEFERS V's
    registry record to :func:`emit_v` (the direct-object build files it only
    after V's liveness is bound to its bytes: a composed row in the registry
    must imply an admission-ready V)."""
    import torch

    from sisteer.eval.object_identity import derived_object_problems
    from sisteer.extraction.paired_write import (APPLICATION_ORDER,
                                                 PSRPairedWriteObject)

    pools_dir = site_path(pools_dir, "pools_dir")

    pos_prov, neg_prov, shared = parents_provenance(root, key, mix_si, ctl_si,
                                                    pools_dir)
    pos_blob = torch.load(pos_prov["object_path"], map_location="cpu",
                          weights_only=True)
    neg_blob = torch.load(neg_prov["object_path"], map_location="cpu",
                          weights_only=True)
    derived_id = v_derived_id(mix_si)
    provenance = {
        "kind": KIND_V,
        "derived_id": derived_id,
        "derivation": ("u_pos(z) - u_neg(z) on the SHARED pre-write layer "
                       "output z, one combined hook per layer (both parents' "
                       "parameters stored bit-exact)"),
        "application_order": APPLICATION_ORDER,
        **shared,
        "parent_pos": pos_prov,
        "parent_neg": neg_prov,
        "built_at": _now(),
        "builder": "scripts/sdfbase/psr_residual_distill.py compose-v",
        # the code that built it — the derived Modal app bakes its commit
        # pin into the image env, so a sidecar names the code as well as
        # the parents (a V left on the volume by an earlier commit is
        # distinguishable from one the current code wrote)
        "builder_commit": os.environ.get("SDFBASE_DERIVED_SHA16"),
    }
    blob = PSRPairedWriteObject.compose_blob(pos_blob, neg_blob,
                                             provenance=provenance)
    out_dir = Path(out_root) / KIND_V / key / derived_id
    out_dir.mkdir(parents=True, exist_ok=True)
    obj_path = out_dir / "object.pt"
    meta_path = out_dir / "derived_meta.json"
    # V's id names the corpus cell only: a sidecar here naming OTHER parents
    # is another pair's published object and is never overwritten
    other = sidecar_pair_problems(meta_path, mix_si, ctl_si)
    if other:
        raise ResidualError(
            f"{out_dir} already holds a V of another pair — refusing to "
            "overwrite it:\n  - " + "\n  - ".join(other))
    # ATOMIC publish: the blob and its sidecar are written to temp names,
    # round-tripped and verdict-checked there, and only then moved into
    # place (sidecar last) — a crash mid-compose can never leave an
    # object.pt without its sidecar, or a sidecar naming another object,
    # at the path every consumer admits V from.
    tmp_obj = out_dir / "object.pt.tmp"
    tmp_meta = out_dir / "derived_meta.json.tmp"
    torch.save(blob, tmp_obj)
    back = torch.load(tmp_obj, map_location="cpu", weights_only=True)
    for side in ("pos", "neg"):
        for tag in ("directions", "gate_w", "gate_b"):
            for l, t in blob[side][tag].items():
                if not torch.equal(back[side][tag][l], t):
                    tmp_obj.unlink(missing_ok=True)
                    raise ResidualError(
                        f"{obj_path}: {side}.{tag}[{l}] did not round-trip "
                        "bit-exact")
    v_meta = {**provenance, "object_sha256": sha256_file(tmp_obj),
              "layers": blob["layers"]}
    tmp_meta.write_text(json.dumps(v_meta, indent=1, default=str) + "\n",
                        encoding="utf-8")
    os.replace(tmp_obj, obj_path)
    os.replace(tmp_meta, meta_path)
    problems = derived_object_problems(obj_path, derived_id=derived_id,
                                       parents_root=root)
    if problems:
        raise ResidualError(
            f"{obj_path} failed the derived-object verdict right after being "
            "written:\n  - " + "\n  - ".join(problems))
    # the registry record for the paired write — AFTER the verdict; deferred to emit-v when the caller gates it
    # on the liveness probe (the direct-object build)
    if emit:
        registry_emit_derived(out_dir, v_meta, object_kind="composed",
                              derivation="composition",
                              script="scripts/sdfbase/psr_residual_distill.py compose-v",
                              extra={"application_order": APPLICATION_ORDER})
    else:
        print(f"[psr-residual] V's registry record DEFERRED to emit-v (after the liveness probe binds "
              f"{LIVENESS_RECEIPT} to {obj_path})", flush=True)
    return obj_path


def liveness_binding_problems(vdir) -> list[str]:
    """Why V's liveness is NOT admission-ready and bound to its bytes (empty = ready): ``gate_liveness.json`` present, a JSON
    mapping of numbers; ``gate_liveness_receipt.json`` present with ``object_sha256`` == sha256(object.pt) and
    ``gate_liveness_sha256`` == sha256(gate_liveness.json). The receipt is what 04_fit_objects.sh's done-ness reads and what emit-v
    requires, so a composed row in the registry implies a V the admission (seedrep/derived_admission.py) will not refuse for a
    missing probe."""
    vdir = Path(vdir)
    problems: list[str] = []
    obj, gl, rc = vdir / "object.pt", vdir / "gate_liveness.json", vdir / LIVENESS_RECEIPT
    if not obj.is_file():
        return [f"no object.pt at {vdir}"]
    doc = None
    if not gl.is_file() or gl.stat().st_size == 0:
        problems.append(f"gate_liveness.json absent or empty at {vdir}")
    else:
        try:
            doc = json.loads(gl.read_text(encoding="utf-8"))
            if not isinstance(doc, dict) or not doc or not all(isinstance(v, (int, float)) for v in doc.values()):
                problems.append("gate_liveness.json is not a non-empty mapping of numbers"); doc = None
        except ValueError as e:
            problems.append(f"gate_liveness.json unreadable: {e}")
    if doc:
        # the admission's criterion, applied HERE so a fully dead V never gets a composed row the front door could read as done:
        # the probe writes measured fractions INCLUDING zero and exits 0
        v = liveness_verdict(doc)
        if not v["not_fully_dead"]:
            problems.append(f"fully dead layer(s) {', '.join(v['dead_layers'])} (live fraction <= dead floor {v['dead_floor']}; "
                            f"the admission's D69 criterion, seedrep/derived_admission.py) — the admission refuses this V; NOT published")
    if not rc.is_file():
        problems.append(f"{LIVENESS_RECEIPT} absent at {vdir} (the probe's receipt binding the liveness to V's bytes)")
        return problems
    try:
        receipt = json.loads(rc.read_text(encoding="utf-8"))
    except ValueError as e:
        return problems + [f"{LIVENESS_RECEIPT} unreadable: {e}"]
    if receipt.get("object_sha256") != sha256_file(obj):
        problems.append(f"{LIVENESS_RECEIPT} names object {str(receipt.get('object_sha256'))[:16]}, V's bytes are {sha256_file(obj)[:16]}")
    if gl.is_file() and receipt.get("gate_liveness_sha256") != sha256_file(gl):
        problems.append(f"{LIVENESS_RECEIPT} names a gate_liveness.json other than the one on disk")
    if receipt.get("not_fully_dead") is not True:
        problems.append(f"{LIVENESS_RECEIPT} does not attest not-fully-dead (not_fully_dead={receipt.get('not_fully_dead')!r})")
    return problems


def write_liveness_receipt(vdir, *, writer: str, context: dict | None = None) -> Path:
    """Write ``gate_liveness_receipt.json`` beside V after a successful probe: V's object sha, the probe file's sha, the writer and
    its context (job, node, code sha, probe settings). Refuses (ResidualError) when object.pt or gate_liveness.json is missing."""
    vdir = Path(vdir)
    obj, gl = vdir / "object.pt", vdir / "gate_liveness.json"
    if not obj.is_file() or not gl.is_file() or gl.stat().st_size == 0:
        raise ResidualError(f"liveness receipt refused: {vdir} must hold object.pt and a non-empty gate_liveness.json")
    try:
        gl_doc = json.loads(gl.read_text(encoding="utf-8"))
        if not isinstance(gl_doc, dict) or not gl_doc or not all(isinstance(x, (int, float)) for x in gl_doc.values()):
            raise ValueError("not a non-empty mapping of numbers")
    except ValueError as e:
        raise ResidualError(f"liveness receipt refused: {gl} is not a liveness map ({e})")
    verdict = liveness_verdict(gl_doc)
    doc = {"object_sha256": sha256_file(obj), "gate_liveness_sha256": sha256_file(gl), "writer": writer,
           "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), **verdict, **(context or {})}
    tmp = vdir / (LIVENESS_RECEIPT + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, vdir / LIVENESS_RECEIPT)
    print(f"[psr-residual] liveness receipt: object {doc['object_sha256'][:16]} -> {vdir / LIVENESS_RECEIPT}; "
          f"{'not fully dead' if verdict['not_fully_dead'] else 'FULLY DEAD at layer(s) ' + ', '.join(verdict['dead_layers'])} "
          f"(min {verdict['liveness_min']} at layer {verdict['liveness_min_layer']}, dead floor {verdict['dead_floor']})", flush=True)
    return vdir / LIVENESS_RECEIPT


def emit_v(root: Path, out_root: Path, key: str, mix_si: str, ctl_si: str = CTL_SI) -> dict:
    """File V's registry record ONLY once its liveness is bound to its bytes. Refuses
    (ResidualError, exit 2) on any :func:`liveness_binding_problems`; otherwise the kept-object replay
    (:func:`ensure_registry_emission_derived`): the whole hook when nothing is recorded, only the missing record, or nothing at
    all when the evidence is terminal under the current campaign + contract — idempotent across re-runs."""
    from sisteer.extraction.paired_write import APPLICATION_ORDER

    vdir = Path(out_root) / KIND_V / key / v_derived_id(mix_si)
    if not (vdir / "object.pt").is_file() or not (vdir / "derived_meta.json").is_file():
        raise ResidualError(f"emit-v: no persisted V (object.pt + derived_meta.json) at {vdir}")
    problems = liveness_binding_problems(vdir)
    if problems:
        raise ResidualError("emit-v refused — V's liveness is not admission-ready / not bound to its bytes:\n  - " + "\n  - ".join(problems))
    return ensure_registry_emission_derived(vdir, object_kind="composed", derivation="composition",
                                            script="scripts/sdfbase/psr_residual_distill.py emit-v",
                                            extra={"application_order": APPLICATION_ORDER})


def sidecar_pair_problems(meta_path, mix_si: str, ctl_si: str) -> list[str]:
    """Problems (empty = same pair, or no sidecar) when an existing derived
    sidecar names parents other than ``<mix_si>__psr_ce_all`` /
    ``<ctl_si>__psr_ce_all``."""
    meta_path = Path(meta_path)
    if not meta_path.is_file():
        return []
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except ValueError as e:
        return [f"{meta_path} unreadable: {e}"]
    out = []
    for role, cell in (("parent_pos", f"{mix_si}__{ARM}"),
                       ("parent_neg", f"{ctl_si}__{ARM}")):
        got = (meta.get(role) or {}).get("cell_id")
        if got != cell:
            out.append(f"{meta_path}: {role}.cell_id={got!r}, this pair needs "
                       f"{cell!r}")
    return out


def verify_v(root: Path, out_root: Path, key: str, mix_si: str,
             ctl_si: str | None = None) -> list[str]:
    """The persisted V re-admitted against the LIVE bank (the shared
    derived-object verdict with the parents re-verified on disk), model-
    free: problems (empty = valid), a missing object or sidecar being a
    problem too. The derived app's compose stage calls this before it
    decides an existing V may stand (a torn publish or a parent refit under
    the same cell id must trigger a rebuild, never a loop of downstream
    refusals). With ``ctl_si`` the sidecar must ALSO name that control as
    ``parent_neg`` — V's id carries the corpus cell only, so a same-id V
    composed against another control is a problem for this pair, not a
    reusable teacher."""
    from sisteer.eval.object_identity import derived_object_problems

    derived_id = v_derived_id(mix_si)
    out_dir = Path(out_root) / KIND_V / key / derived_id
    obj_path = out_dir / "object.pt"
    meta_path = out_dir / "derived_meta.json"
    if not obj_path.is_file():
        return [f"{obj_path} missing"]
    if not meta_path.is_file():
        return [f"{meta_path} missing (torn publish)"]
    try:
        problems = list(derived_object_problems(obj_path, derived_id=derived_id,
                                                parents_root=root))
        if ctl_si is not None:
            problems += sidecar_pair_problems(meta_path, mix_si, ctl_si)
        return problems
    except Exception as e:  # noqa: BLE001 — an unreadable blob is a problem, not a crash
        return [f"{obj_path}: {type(e).__name__}: {e}"]


# --------------------------------------------------------------------------- #
# capture set                                                                  #
# --------------------------------------------------------------------------- #


#: The capture-budget draw, stated once (recorded in every subsampled
#: capture manifest under ``capture_subsample.rule``).
CAPTURE_SUBSAMPLE_RULE = (
    "per source pool and per family (source x family stratum): the family's "
    "token quota is budget x share[family] with ONE declared share table "
    "applied to both sources (the mix pool's realized official-token share "
    "per family); the stratum's docs are sorted by doc_id, shuffled by "
    "random.Random(f'{seed}|capture|{si}|{family}') and taken in that order "
    "until the running sum of official_tokens reaches the quota AND at least "
    "CAPTURE_MIN_STRATUM_DOCS docs are taken (the doc that crosses the quota "
    "is kept, so a stratum carries >= its quota); kept in pool order; the "
    "holdout carve then runs on the selection exactly as on a whole pool")
#: Every capture stratum must be able to hold one doc out (the 10% carve
#: keeps >= 1 per stratum), so a stratum draws at least this many docs.
CAPTURE_MIN_STRATUM_DOCS = 2


def family_token_shares(docs: list) -> dict[str, float]:
    """A pool's realized official-token share per family (the ONE share
    table the budgeted draw applies to BOTH sources — a draw balanced by
    source but not by family gave the two sources materially different
    genre mixtures, 6.5% vs 19% advice tokens)."""
    tot: dict[str, int] = {}
    for d in docs:
        if d.official_tokens is None:
            raise ResidualError(f"doc {d.doc_id!r} records no official_tokens — "
                                "the capture budget is defined on them")
        tot[d.family] = tot.get(d.family, 0) + int(d.official_tokens)
    n = sum(tot.values())
    if n <= 0:
        raise ResidualError("pool carries no official tokens")
    return {f: tot[f] / n for f in sorted(tot)}


def capture_subsample(docs: list, budget_tokens: int, *, seed: int, si: str,
                      family_shares: dict[str, float]) -> tuple[list, dict]:
    """A seeded, budget-bounded, family-stratified subsample of ONE source
    pool for the capture set (:data:`CAPTURE_SUBSAMPLE_RULE`). The 100k
    pair's pools ARE the capture set (81 docs); the full − full pair's pools
    hold 17k docs each (42.8M official tokens), so an all-docs capture would
    make one distillation epoch ~470x longer than the measured 100k one —
    the capture set is the distillation's training data for matching V's
    writes, not the parents' training data, and a budget matched to the
    100k pools (100,000 official tokens per source) holds the distillation
    step fixed across the two decompositions. Refuses a non-positive
    budget, a doc without ``official_tokens``, a family in the share table
    the pool lacks (or the reverse), and a stratum that cannot meet its
    quota with >= CAPTURE_MIN_STRATUM_DOCS docs (never a silent whole-pool
    or thinner fallback)."""
    import random

    budget = int(budget_tokens)
    if budget <= 0:
        raise ResidualError(f"--capture-budget-tokens must be > 0, got {budget}")
    by_family: dict[str, list] = {}
    for d in docs:
        if d.official_tokens is None:
            raise ResidualError(
                f"pool {si}: doc {d.doc_id!r} records no official_tokens — the "
                "capture budget is defined on them")
        by_family.setdefault(d.family, []).append(d)
    if set(by_family) != set(family_shares):
        raise ResidualError(
            f"pool {si} families {sorted(by_family)} != the declared share "
            f"table's {sorted(family_shares)}")
    picked: set[str] = set()
    per_family = {}
    for fam in sorted(by_family):
        quota = int(round(budget * float(family_shares[fam])))
        order = sorted(by_family[fam], key=lambda d: d.doc_id)
        rng = random.Random(f"{seed}|capture|{si}|{fam}")
        rng.shuffle(order)
        cum, n = 0, 0
        for d in order:
            if cum >= quota and n >= CAPTURE_MIN_STRATUM_DOCS:
                break
            picked.add(d.doc_id)
            cum += int(d.official_tokens)
            n += 1
        if cum < quota or n < CAPTURE_MIN_STRATUM_DOCS:
            raise ResidualError(
                f"pool {si} family {fam!r}: {n} docs / {cum} official tokens in "
                f"total, below the quota {quota} with >= "
                f"{CAPTURE_MIN_STRATUM_DOCS} docs — a quota the stratum cannot "
                "meet is not a subsample")
        per_family[fam] = {"quota_tokens": quota, "n_pool": len(by_family[fam]),
                           "n_selected": n, "official_tokens_selected": int(cum),
                           "seed_key": f"{seed}|capture|{si}|{fam}"}
    selected = [d for d in docs if d.doc_id in picked]
    record = {"si": si, "n_pool": len(docs), "n_selected": len(selected),
              "official_tokens_selected": int(sum(int(d.official_tokens)
                                                  for d in selected)),
              "official_tokens_pool": int(sum(int(d.official_tokens) for d in docs)),
              "per_family": per_family}
    return selected, record


def build_capture_set(pools_dir: Path, mix_si: str, ctl_si: str, *,
                      frac: float = 0.1, seed: int = 0,
                      budget_tokens: int | None = None):
    """The capture set: every doc of both parent pools (the 100k pair's
    design; the manifest is then byte-identical to what it always was), or
    — with ``budget_tokens`` — a seeded per-source subsample of that many
    official tokens (:func:`capture_subsample`; recorded in the manifest
    under ``capture_subsample`` and therefore in the calibration identity's
    capture sha)."""
    from sisteer.extraction.residual import (load_capture_docs,
                                             stratified_holdout)

    mix_path = Path(pools_dir) / f"pairs_{mix_si}.jsonl"
    ctl_path = Path(pools_dir) / f"pairs_{ctl_si}.jsonl"
    for p in (mix_path, ctl_path):
        if not p.is_file():
            raise ResidualError(f"pool {p} missing")
    mix_docs = load_capture_docs(mix_path, "mix")
    ctl_docs = load_capture_docs(ctl_path, "ctl")
    subsample = None
    if budget_tokens is not None:
        shares = family_token_shares(mix_docs)   # ONE table for both sources
        mix_docs, mix_rec = capture_subsample(mix_docs, budget_tokens, seed=seed,
                                              si=mix_si, family_shares=shares)
        ctl_docs, ctl_rec = capture_subsample(ctl_docs, budget_tokens, seed=seed,
                                              si=ctl_si, family_shares=shares)
        subsample = {"budget_tokens": int(budget_tokens),
                     "rule": CAPTURE_SUBSAMPLE_RULE,
                     "min_stratum_docs": CAPTURE_MIN_STRATUM_DOCS,
                     "family_shares": shares,
                     "family_shares_source": f"realized official-token shares of the mix pool {mix_si}",
                     "per_source": {"mix": mix_rec, "ctl": ctl_rec}}
    docs = mix_docs + ctl_docs
    train, hold, manifest = stratified_holdout(docs, frac=frac, seed=seed)
    if subsample is not None:
        manifest["capture_subsample"] = subsample
    sources = {"mix": {"si": mix_si, "pairs_path": str(mix_path),
                       "sha256": sha256_file(mix_path)},
               "ctl": {"si": ctl_si, "pairs_path": str(ctl_path),
                       "sha256": sha256_file(ctl_path)}}
    return docs, train, hold, manifest, sources


# --------------------------------------------------------------------------- #
# Trainer defaults + model                                                     #
# --------------------------------------------------------------------------- #


#: The trainer knobs a fits config supplies and an explicit CLI value may
#: override.
HP_KEYS = ("lr", "max_grad_norm", "warmup_frac", "reg_coeff", "accum_steps",
           "batch_size", "max_len")
#: The ``trainer`` block every artifact records (calibration_controls.json,
#: derived_meta.json, calibration.json) — one spelling, this key order.
TRAINER_HP_KEYS = ("lr", "lr_mult", "max_grad_norm", "warmup_frac", "reg_coeff",
                   "accum_steps", "batch_size", "max_len", "weight_decay",
                   "epochs", "patience", "seed", "config", "config_sha256")


# --------------------------------------------------------------------------- #
# Calibration identity                                                        #
# --------------------------------------------------------------------------- #

IDENTITY_KIND = "psr_residual_calibration_identity"

#: The parent provenance fields the identity pins per parent.
_IDENTITY_PARENT_FIELDS = ("cell_id", "object_sha256", "budget_meta_sha256",
                           "fit_fingerprint", "model_revision")


def calibration_identity(args, hp: dict, model, *, pos_prov: dict,
                         neg_prov: dict, manifest: dict, sources: dict,
                         layers) -> dict:
    """Everything the recovery gate's attainable EDs depend on, in ONE block
    with a sha256 fingerprint — stamped on ``calibration_controls.json`` by
    ``calibrate`` and compared IN FULL by ``distill`` / ``all`` before a
    production run consumes it (production used to compare lr alone, so a
    calibration built on other parents, another capture split, another seed
    or other measurement settings would have gated a production run it says
    nothing about). Fields: the model (id, revision, model key, dtype,
    device type), both parents' object + sidecar shas and fit fingerprints,
    the source pools' shas, the capture split (holdout fraction, seed, the
    capture manifest's own sha — the exact bytes ``capture_manifest.json``
    is written with), the layer set, every trainer knob
    (:data:`TRAINER_HP_KEYS`) and every measurement setting (n_boot, the
    planted control's seed and target active fraction)."""
    from sisteer.extraction.residual import capture_manifest_text

    par = next(model.parameters())
    ident = {
        "kind": IDENTITY_KIND,
        "model": {"id": args.model, "revision": args.revision,
                  "model_key": args.model_key, "dtype": str(par.dtype),
                  "device_type": par.device.type},
        "parents": {role: {k: prov.get(k) for k in _IDENTITY_PARENT_FIELDS}
                    for role, prov in (("pos", pos_prov), ("neg", neg_prov))},
        "source_pools": {k: {"si": v["si"], "sha256": v["sha256"]}
                         for k, v in sources.items()},
        "capture": {"holdout_frac": float(args.holdout_frac),
                    "seed": int(args.seed),
                    "manifest_sha256": hashlib.sha256(
                        capture_manifest_text(manifest, sources=sources)
                        .encode("utf-8")).hexdigest(),
                    "n_train": manifest.get("n_train"),
                    "n_holdout": manifest.get("n_holdout"),
                    # the budget is named only when one is set, so every
                    # whole-pool identity (the 100k pair's) keeps its
                    # fingerprint byte for byte
                    **({"budget_tokens": int(args.capture_budget_tokens),
                        "family_shares": (manifest.get("capture_subsample") or {}).get("family_shares"),
                        "min_stratum_docs": CAPTURE_MIN_STRATUM_DOCS}
                       if getattr(args, "capture_budget_tokens", None) is not None
                       else {})},
        "layers": [int(l) for l in layers],
        "trainer": {k: hp[k] for k in TRAINER_HP_KEYS},
        "measurement": {"n_boot": int(args.n_boot),
                        "planted_seed": int(args.seed) + 1,
                        "planted_active_frac": PLANTED_ACTIVE_FRAC},
    }
    ident["fingerprint"] = hashlib.sha256(
        json.dumps(ident, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return ident


def identity_diffs(recorded: dict, current: dict, prefix: str = "") -> list[str]:
    """Dotted paths (with both values) where two identity blocks differ;
    nested dicts are walked, ``fingerprint`` itself is skipped (the fields
    are what explain a fingerprint mismatch)."""
    out = []
    for k in sorted(set(recorded) | set(current)):
        if k == "fingerprint" and not prefix:
            continue
        path = f"{prefix}{k}"
        a, b = recorded.get(k), current.get(k)
        if isinstance(a, dict) and isinstance(b, dict):
            out += identity_diffs(a, b, prefix=path + ".")
        elif json.dumps(a, sort_keys=True, default=str) != \
                json.dumps(b, sort_keys=True, default=str):
            out.append(f"{path} (calibration {a!r} != production {b!r})")
    return out


def require_same_identity(calibration: dict, identity: dict) -> None:
    """Refuse a production run whose identity differs ANYWHERE from the
    calibration's — attainable and production scores must be same-trainer,
    same-parents, same-capture, same-measurement, or the recovery gate
    compares two different experiments."""
    recorded = calibration.get("identity")
    if not isinstance(recorded, dict) or not recorded.get("fingerprint"):
        raise ResidualError(
            "calibration_controls.json records no identity fingerprint — a "
            "pre-fingerprint calibration cannot be proven same-trainer / "
            "same-parents / same-capture as this production run; re-run "
            "calibrate")
    diffs = identity_diffs(recorded, identity)
    if diffs:
        raise ResidualError(
            "calibration identity != production identity on "
            + "; ".join(diffs)
            + " — attainable and production scores must be same-trainer, "
              "same-parents, same-capture and same-measurement (the recovery "
              "gate compares them); re-run calibrate under this run's settings")


def trainer_defaults(cfg_path) -> dict:
    """The model's PSR winner from its fits config."""
    import yaml

    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    tk = (cfg.get("train_kwargs") or {}).get(ARM) or {}
    bs = (cfg.get("batch_size_by_arm") or {}).get(ARM)
    for k in ("lr", "max_grad_norm", "warmup_frac", "gate_reg", "accum_steps"):
        if tk.get(k) is None:
            raise ResidualError(f"{cfg_path}: train_kwargs.{ARM}.{k} missing")
    if bs is None:
        raise ResidualError(f"{cfg_path}: batch_size_by_arm.{ARM} missing")
    return {"lr": float(tk["lr"]), "max_grad_norm": float(tk["max_grad_norm"]),
            "warmup_frac": float(tk["warmup_frac"]),
            "reg_coeff": float(tk["gate_reg"]),
            "accum_steps": int(tk["accum_steps"]), "batch_size": int(bs),
            "max_len": int(cfg.get("max_len", 8192)),
            "weight_decay": float(tk.get("weight_decay", 0.01)),
            "config": str(cfg_path), "config_sha256": sha256_file(cfg_path)}


def format_mult(m: float) -> str:
    """The ONE spelling of an lr multiplier in file keys and tags."""
    return f"{float(m):g}"


def parse_lr_mults(spec) -> list[float]:
    """``--lr-mult``: a comma-separated list (``0.3,1,3``), a single value,
    or None → the default sweep. Positive, de-duplicated, order kept."""
    if spec is None:
        return list(DEFAULT_LR_MULTS)
    if isinstance(spec, (int, float)):
        vals = [float(spec)]
    else:
        try:
            vals = [float(x) for x in str(spec).replace(";", ",").split(",")
                    if x.strip()]
        except ValueError as e:
            raise ResidualError(f"--lr-mult {spec!r}: {e}") from e
    if not vals:
        raise ResidualError("--lr-mult: empty list")
    if any(not math.isfinite(v) or v <= 0 for v in vals):
        raise ResidualError(f"--lr-mult {spec!r}: every multiplier must be a "
                            "finite number > 0")
    out: list[float] = []
    for v in vals:
        if v not in out:
            out.append(v)
    return out


def lr_mults_from_args(args) -> list[float]:
    return parse_lr_mults(getattr(args, "lr_mult", None))


def hp_from_args(args, lr_mult: float | None = None) -> dict:
    """The trainer hyperparameters at ONE lr multiplier: ``lr_mult`` when
    given, else the single value of ``--lr-mult`` (a list is an error here
    — the sweep callers resolve one multiplier at a time), else 1.0."""
    hp = trainer_defaults(args.config) if args.config else {
        **dict.fromkeys(HP_KEYS), "weight_decay": 0.01, "config": None,
        "config_sha256": None}
    for k in HP_KEYS:
        v = getattr(args, k, None)
        if v is not None:
            hp[k] = v  # an explicit CLI value wins over the config's
    if hp["lr"] is None:
        raise ResidualError("no lr: pass --config or --lr")
    if lr_mult is None:
        spec = getattr(args, "lr_mult", None)
        mults = parse_lr_mults(spec) if spec is not None else [1.0]
        if len(mults) != 1:
            raise ResidualError(
                f"one lr multiplier at a time here, got --lr-mult {spec!r}")
        lr_mult = mults[0]
    hp["lr"] = float(hp["lr"]) * float(lr_mult)
    hp["lr_mult"] = float(lr_mult)
    hp["epochs"] = int(args.epochs)
    hp["patience"] = int(args.patience)
    hp["seed"] = int(args.seed)
    return hp


def stop_reason(diag: dict, hp: dict) -> str:
    """Why a distillation stopped, in words the report can carry: the
    epoch cap (and whether the held-out was still improving when it hit),
    or the patience rule (after which best epoch), with the counts."""
    cap, pat = int(hp["epochs"]), int(hp["patience"])
    run, best = int(diag.get("epochs_run") or 0), diag.get("best_epoch")
    curve = [float(v) for v in (diag.get("heldout_curve") or [])]
    if run >= cap:
        tail = ""
        if len(curve) >= 3:
            still = curve[-1] < min(curve[:-1])
            tail = (" — held-out still improving at the cap" if still
                    else f" — held-out flat since best epoch {best}")
        also = " (early-stop criterion also met)" if diag.get("stopped_early") else ""
        return f"epoch cap {cap} reached{also}{tail}"
    if diag.get("stopped_early"):
        return (f"early stop: no held-out improvement for {pat} epochs after "
                f"best epoch {best} (ran {run}/{cap})")
    return f"stopped after {run}/{cap} epochs"


#: How the production multiplier is picked from the sweep — pre-stated.
#: (2026-09-09, the rule after the GLM residual: the previous rule
#: was a discrete argmax of the same score, which on a monotone ladder returns
#: the top rung by construction and can never refuse.)
SELECTION_RULE = (
    "a concave quadratic in log(lr_mult) is least-squares fitted to the "
    "worst-stratum attainable content ED (min over mix/ctl of the min over "
    "controls (a) R(mix, ∅) and (b) R(mix ⊕ C, mix) — the number the "
    "recovery gate compares against) over at least 4 scored rungs; the rung "
    "nearest the vertex (in log lr_mult) is SELECTED only when the fit is "
    "concave, the vertex lies strictly inside the ladder, that rung is not "
    "an edge rung, and its ED bootstrap CI lies wholly above the CI of BOTH "
    "edge rungs; otherwise the sweep is REFUSED with a box-edge verdict and "
    "no multiplier is promoted — a decision for the user, never a re-roll "
    "(an explicit --lr-mult IS that decision and is stamped as an override). "
    "Jointly, the selected rung's planted-control held-out losses are "
    "compared with each control's own ladder minimum: a rung outside that "
    "control's plateau noise is FILED as a held-out divergence finding, "
    "never a refusal")
#: fewer scored rungs than this cannot support the quadratic → refused
MIN_FIT_RUNGS = 4
#: the smallest held-out-loss noise band credited to a control: 1% of its
#: minimum held-out loss (its plateau's epoch-to-epoch range when wider)
HELDOUT_NOISE_REL_FLOOR = 0.01
VERDICT_INTERIOR = "interior"
VERDICT_BOX_EDGE = "box_edge"
#: ONE multiplier requested: there is no ladder and nothing is selected BY
#: the rule — the multiplier is the user's explicit choice, stamped as such
VERDICT_SINGLE_RUNG = "single_rung"


def _attain_score(payload: dict) -> float:
    vals = [v for v in (payload.get("attainable_ed_content") or {}).values()
            if v is not None]
    return min(vals) if vals else float("-inf")


def _score_ci(payload: dict) -> tuple[float, float] | None:
    """The bootstrap CI of the number :func:`_attain_score` IS: the pooled
    content-class ED of the control attaining the minimum on the stratum
    attaining the minimum (``controls[*].ed[<stratum>].content.{lo,hi}``).
    None when the payload records no per-stratum CI — the selector then
    cannot establish separation and refuses."""
    attain = {g: v for g, v in (payload.get("attainable_ed_content") or {}).items()
              if v is not None}
    if not attain:
        return None
    g = min(attain, key=attain.get)
    best = None
    for ctl in (payload.get("controls") or {}).values():
        e = (((ctl.get("ed") or {}).get(g) or {}).get("content") or {})
        if e.get("ed") is None or e.get("lo") is None or e.get("hi") is None:
            continue
        if best is None or float(e["ed"]) < best[0]:
            best = (float(e["ed"]), float(e["lo"]), float(e["hi"]))
    return None if best is None else (best[1], best[2])


def _fit_log_quadratic(points: list[tuple[float, float]]) -> dict:
    """Least-squares ``y = c2·x² + c1·x + c0`` with ``x = ln(lr_mult)`` over
    ``(lr_mult, score)`` points; concavity, the vertex and whether it lies
    strictly inside the ladder."""
    import numpy as np
    xs = np.array([math.log(m) for m, _ in points], dtype=float)
    ys = np.array([y for _, y in points], dtype=float)
    c2, c1, c0 = (float(c) for c in np.polyfit(xs, ys, 2))
    concave = c2 < -1e-12
    vx = (-c1 / (2.0 * c2)) if abs(c2) > 1e-12 else None
    pred = c2 * xs ** 2 + c1 * xs + c0
    return {"x": "ln(lr_mult)", "n_points": len(points),
            "coefficients": {"c2": c2, "c1": c1, "c0": c0},
            "concave": bool(concave),
            "vertex_lr_mult": (float(math.exp(vx)) if vx is not None else None),
            "vertex_inside_ladder": (bool(xs.min() < vx < xs.max())
                                     if vx is not None else False),
            "rmse": float(np.sqrt(np.mean((pred - ys) ** 2)))}


def _heldout_divergence(per_mult: dict[str, dict], selected: str) -> dict:
    """Per planted control: the rung minimising its best held-out loss, that
    rung's plateau noise (the epoch-to-epoch range of its held-out curve from
    the best epoch on, floored at :data:`HELDOUT_NOISE_REL_FLOOR` × the
    minimum) and whether the selected rung's loss sits within that band.
    Outside = ``divergent`` (a finding the artifacts carry), never a
    refusal: the ED is the recovery gate's own quantity (direction), the
    held-out loss the distillation MSE (magnitude included); when they
    disagree the reader must know."""
    names: set[str] = set()
    for pl in per_mult.values():
        names.update((pl.get("controls") or {}).keys())
    out: dict = {}
    for name in sorted(names):
        losses: dict[str, float] = {}
        for key, pl in per_mult.items():
            c = (pl.get("controls") or {}).get(name) or {}
            if c.get("best_heldout_loss") is not None:
                losses[key] = float(c["best_heldout_loss"])
        if not losses or selected not in losses:
            out[name] = {"divergent": None,
                         "reason": "no held-out loss recorded at the selected rung"}
            continue
        k_min = min(losses, key=losses.get)
        c_min = (per_mult[k_min].get("controls") or {}).get(name) or {}
        curve = [float(v) for v in (c_min.get("heldout_curve") or [])]
        be = c_min.get("best_epoch")
        plateau = (curve[int(be) - 1:] if (be is not None and int(be) >= 1
                                           and len(curve) >= int(be)) else curve)
        spread = (max(plateau) - min(plateau)) if len(plateau) >= 2 else 0.0
        noise = max(spread, HELDOUT_NOISE_REL_FLOOR * abs(losses[k_min]))
        gap = losses[selected] - losses[k_min]
        out[name] = {
            "ladder_min_lr_mult": float(per_mult[k_min]["trainer"]["lr_mult"]),
            "min_loss": losses[k_min], "selected_loss": losses[selected],
            "gap": gap, "noise": noise, "plateau_epochs": len(plateau),
            "divergent": bool(gap > noise),
            "losses_by_lr_mult": {k: losses[k] for k in sorted(losses, key=float)}}
    return out


def select_lr_mult(per_mult: dict[str, dict]) -> tuple[str | None, dict, dict]:
    """:data:`SELECTION_RULE` over ``{format_mult(m): calibration payload}``.
    Returns ``(selected key or None, table, selection)``: the per-rung
    comparison table every artifact records (``selected`` True on at most
    one rung) and the selection record — verdict (:data:`VERDICT_INTERIOR`
    / :data:`VERDICT_BOX_EDGE`), the reason, the fit, the CI separation,
    the held-out divergence check — plus ``reference_lr_mult_key``, the rung
    a refused sweep's document is written around (for reading, never
    promotion). This selector can genuinely fail: a monotone ladder, a
    convex or flat one, a vertex off the ladder, a ladder of fewer than
    :data:`MIN_FIT_RUNGS` scored rungs, or a top rung whose CI the edges
    overlap all return ``None``. A sweep of exactly ONE multiplier is not a
    ladder: it passes through as :data:`VERDICT_SINGLE_RUNG` — the
    operator's explicit choice, stamped as such, never as a selection by
    the rule."""
    if not per_mult:
        raise ResidualError("select_lr_mult: empty sweep")
    table: dict = {}
    scored: list[tuple[float, str, float]] = []
    for key, pl in per_mult.items():
        m = float(pl["trainer"]["lr_mult"])
        score = _attain_score(pl)
        ci = _score_ci(pl)
        table[key] = {
            "lr_mult": m, "lr": pl["trainer"]["lr"],
            "score_worst_stratum_attainable_ed": (
                score if math.isfinite(score) else None),
            "score_ci95": (list(ci) if ci is not None else None),
            "attainable_ed_content": pl.get("attainable_ed_content"),
            "controls": {name: {k: c.get(k) for k in (
                "epochs_run", "best_epoch", "stopped_early", "stop_reason",
                "best_heldout_loss", "wall_s")}
                for name, c in (pl.get("controls") or {}).items()},
            "selected": False,
        }
        if math.isfinite(score):
            scored.append((m, key, score))
    scored.sort()
    sel: dict = {"rule": SELECTION_RULE, "verdict": VERDICT_BOX_EDGE,
                 "reason": None, "selected_lr_mult": None, "promoted": False,
                 "n_rungs": len(per_mult), "n_scored": len(scored),
                 "min_fit_rungs": MIN_FIT_RUNGS,
                 "ladder_lr_mults": [m for m, _, _ in scored],
                 "edge_lr_mults": None, "argmax_lr_mult": None,
                 "argmax_on_edge": None, "fit": None, "ci_separation": None,
                 "heldout_loss": None, "heldout_divergence_finding": None,
                 "reference_lr_mult_key": next(iter(per_mult))}
    if not scored:
        sel["reason"] = "no rung has an attainable ED (unfinished sweep)"
        return None, table, sel
    if len(per_mult) == 1 and len(scored) == 1:
        key = scored[0][1]
        table[key]["selected"] = True
        sel.update(verdict=VERDICT_SINGLE_RUNG, promoted=True,
                   selected_lr_mult=scored[0][0], reference_lr_mult_key=key,
                   edge_lr_mults=[scored[0][0], scored[0][0]],
                   argmax_lr_mult=scored[0][0], argmax_on_edge=True,
                   reason=("one multiplier requested: no ladder to select "
                           "from; the multiplier is the user's explicit "
                           "choice, not a selection by the rule"))
        return key, table, sel
    edges = (scored[0][1], scored[-1][1])
    sel["edge_lr_mults"] = [scored[0][0], scored[-1][0]]
    # the raw argmax (ties toward lr_mult 1) is RECORDED for the reader; it
    # selects nothing — it is what the previous rule promoted blindly
    am = max(scored, key=lambda t: (t[2], -abs(math.log(t[0]))))
    sel["argmax_lr_mult"], sel["argmax_on_edge"] = am[0], am[1] in edges
    sel["reference_lr_mult_key"] = am[1]

    def refuse(reason: str, ref_key: str):
        sel["reason"], sel["reference_lr_mult_key"] = reason, ref_key
        return None, table, sel

    if len(scored) < MIN_FIT_RUNGS:
        return refuse(f"too few scored rungs for a quadratic fit "
                      f"({len(scored)} < {MIN_FIT_RUNGS})", am[1])
    fit = _fit_log_quadratic([(m, sc) for m, _, sc in scored])
    sel["fit"] = fit
    if not fit["concave"]:
        return refuse("the quadratic in log(lr_mult) is not concave — no "
                      "interior maximum (a monotone or convex ladder)", am[1])
    if not fit["vertex_inside_ladder"]:
        return refuse(f"the vertex lr_mult {fit['vertex_lr_mult']:.4g} lies "
                      f"outside the ladder [{scored[0][0]:g}, {scored[-1][0]:g}]",
                      am[1])
    vx = math.log(fit["vertex_lr_mult"])
    near = min(scored, key=lambda t: abs(math.log(t[0]) - vx))
    fit["nearest_rung_lr_mult"] = near[0]
    nk = near[1]
    if nk in edges:
        return refuse(f"the rung nearest the vertex ({near[0]:g}) is an edge "
                      "rung", nk)
    ci_sel = table[nk]["score_ci95"]
    edge_rows: dict = {}
    separated = ci_sel is not None
    for ek in edges:
        eci = table[ek]["score_ci95"]
        ok = bool(ci_sel is not None and eci is not None and ci_sel[0] > eci[1])
        edge_rows[ek] = {"lr_mult": table[ek]["lr_mult"], "ci95": eci,
                         "separated": ok}
        separated = separated and ok
    sel["ci_separation"] = {"selected_lr_mult": near[0], "selected_ci95": ci_sel,
                            "edges": edge_rows,
                            "separated_from_both_edges": bool(separated)}
    if not separated:
        return refuse("the ED bootstrap CI of the rung nearest the vertex "
                      f"({near[0]:g}) does not clear both edge rungs' CIs"
                      + (" (no CI recorded)" if ci_sel is None else ""), nk)
    table[nk]["selected"] = True
    hd = _heldout_divergence(per_mult, nk)
    sel.update(verdict=VERDICT_INTERIOR, promoted=True, selected_lr_mult=near[0],
               reference_lr_mult_key=nk, heldout_loss=hd,
               heldout_divergence_finding=bool(any(v.get("divergent")
                                                   for v in hd.values())),
               reason=("concave fit with its vertex inside the ladder; the "
                       "nearest rung is interior and its ED CI clears both "
                       "edge rungs' CIs"))
    return nk, table, sel


def calibration_for_mult(doc: dict, lr_mult: float | None) -> tuple[dict, float]:
    """The calibration payload a production run at ``lr_mult`` gates on:
    from a SWEEP file, the recorded per-multiplier payload (the selected one
    when ``lr_mult`` is None; an explicit multiplier must be in the sweep
    and the sweep must be complete); from a pre-sweep single file, the file
    itself at its own multiplier. Returns (payload, lr_mult)."""
    sweep = doc.get("sweep")
    if not sweep:
        rec = ((doc.get("identity") or {}).get("trainer") or {}).get("lr_mult")
        m = float(lr_mult) if lr_mult is not None else float(rec or 1.0)
        return doc, m
    if not sweep.get("complete", True):
        raise ResidualError(
            "calibration_controls.json holds an INCOMPLETE lr-mult sweep "
            f"(finished {list((sweep.get('per_mult') or {}))} of "
            f"{sweep.get('lr_mults')}) — re-run calibrate to completion before "
            "distilling")
    if lr_mult is None:
        if sweep.get("selected_lr_mult") is None:
            sel = sweep.get("selection") or {}
            raise ResidualError(
                "the lr-mult sweep was REFUSED "
                f"({sel.get('verdict')}: {sel.get('reason')}) — no multiplier "
                "was selected and nothing is promoted; building anyway is the "
                "user's decision: pass --lr-mult <m> explicitly (stamped as an "
                "override in derived_meta.json)")
        m = float(sweep["selected_lr_mult"])
    else:
        m = float(lr_mult)
    key = format_mult(m)
    per = sweep.get("per_mult") or {}
    if key not in per:
        raise ResidualError(
            f"--lr-mult {key} is not in the calibration sweep "
            f"{list(per)} (selected "
            f"{format_mult(sweep['selected_lr_mult']) if sweep.get('selected_lr_mult') is not None else 'nothing — the sweep refused'}) — "
            "the recovery gate needs a calibration at THIS lr; re-run "
            "calibrate with it in the list")
    return per[key], m


def sweep_summary(sweep: dict | None) -> dict | None:
    """The sweep block without the heavy per-multiplier payloads — what
    derived_meta.json / calibration.json stamp on the distilled object."""
    if not sweep:
        return None
    return {k: sweep.get(k) for k in ("lr_mults", "selected_lr_mult",
                                      "selection", "selection_rule", "budget",
                                      "table", "complete")}


def _selection_sentence(sweep: dict) -> str:
    """One line for the console: what the sweep selected, or why it refused."""
    sel = sweep.get("selection") or {}
    if sweep.get("selected_lr_mult") is not None:
        tail = (" — HELD-OUT DIVERGENCE FINDING: the selected rung is outside "
                "a planted control's loss noise of that control's own minimum "
                "(see sweep.selection.heldout_loss)"
                if sel.get("heldout_divergence_finding") else "")
        return (f"selected lr_mult {format_mult(sweep['selected_lr_mult'])} "
                f"({sel.get('verdict')}: {sel.get('reason')}){tail}")
    return (f"REFUSED ({sel.get('verdict')}: {sel.get('reason')}); nothing "
            "promoted — a decision for the user (`distill --lr-mult <m>` "
            "builds anyway and is stamped as an override)")


def _load_model(args):
    from sisteer.extraction.fitmon import pin_attention_backends
    from sisteer.models import load_model_and_tokenizer

    # device/dtype None -> the loader's own defaults (cuda when available;
    # bf16 on cuda, fp32 otherwise); the loader also puts the model in
    # eval mode.
    model, tok = load_model_and_tokenizer(args.model, device=args.device,
                                          dtype=args.dtype,
                                          revision=args.revision)
    pin_attention_backends()
    return model, tok


def hydrate_parents(model, tok, root, key, mix_si, ctl_si, pools_dir):
    from sisteer.methods import load_object

    pos_prov, neg_prov, shared = parents_provenance(root, key, mix_si, ctl_si,
                                                    pools_dir)
    mix = load_object(pos_prov["object_path"], model=model, tok=tok)
    ctl = load_object(neg_prov["object_path"], model=model, tok=tok)
    return mix, ctl, pos_prov, neg_prov, shared


class MonotoneSteps:
    """W&B accepts a non-decreasing ``step`` per run and silently DROPS any
    point logged below the run's high-water mark. One process here logs
    several trainings in sequence — control (a), control (b), per lr
    multiplier, then production — and each restarts its optimizer step at
    0, so everything after the first training was lost (control (b)'s
    curves never reached W&B). Every section maps its local step s to
    ``base + s`` with ``base`` = the run's high-water mark + 1 at the
    section's start; the local step is logged alongside as
    ``<tag>/local_step`` so per-section x-axes are recoverable."""

    def __init__(self):
        self.base = 0
        self.high = -1

    def new_section(self) -> int:
        self.base = self.high + 1
        return self.base

    def step(self, local) -> int:
        g = self.base + int(local or 0)
        self.high = max(self.high, g)
        return g


WANDB_STEPS = MonotoneSteps()


def _log_fn_factory(tag: str, wandb_run=None, steps: MonotoneSteps | None = None):
    steps = WANDB_STEPS if steps is None else steps

    def log_fn(rec: dict, step=None):
        line = {"tag": tag, "step": step, **{k: (float(v) if isinstance(v, (int, float)) else v)
                                            for k, v in rec.items()}}
        print(json.dumps(line, default=str), flush=True)
        if wandb_run is not None:
            try:
                payload = {f"{tag}/{k}": v for k, v in rec.items()
                           if isinstance(v, (int, float))}
                payload[f"{tag}/local_step"] = int(step or 0)
                wandb_run.log(payload, step=steps.step(step))
            except Exception:  # noqa: BLE001 — telemetry never kills a fit
                pass
    return log_fn


def _distill(model, tok, teacher, train, hold, hp, *, tag, log_fn, n_boot,
             teacher_precheck=None, checkpoint=None, steps: MonotoneSteps | None = None):
    from sisteer.extraction.distill import EarlyStopSpec
    from sisteer.extraction.psr import PSRObject
    from sisteer.extraction.residual import train_psr_residual
    from sisteer.models import num_layers

    (WANDB_STEPS if steps is None else steps).new_section()  # W&B steps stay monotone
    student = PSRObject(model, list(range(num_layers(model))),
                        train_direction=True)
    diag: dict = {}
    t0 = time.time()
    ck = checkpoint or {}
    history = train_psr_residual(
        model, tok, student, teacher, train, hold,
        lr=hp["lr"], epochs=hp["epochs"], batch_size=hp["batch_size"],
        accum_steps=hp["accum_steps"], max_len=hp["max_len"],
        weight_decay=hp["weight_decay"], reg_coeff=hp["reg_coeff"],
        warmup_frac=hp["warmup_frac"], max_grad_norm=hp["max_grad_norm"],
        seed=hp["seed"], log_fn=log_fn, log_tag=tag,
        early_stop=EarlyStopSpec(patience=hp["patience"]),
        diagnostics=diag, teacher_precheck=teacher_precheck, n_boot=n_boot,
        boot_seed=hp["seed"], **ck)
    diag["wall_s"] = round(time.time() - t0, 1)
    diag["n_steps"] = len(history)
    return student, diag


def _strip_records(diag: dict) -> tuple[dict, list]:
    d = dict(diag)
    recs = d.pop("doc_records", [])
    return d, recs


# --------------------------------------------------------------------------- #
# Analysis-contract blocks (analysis_r2/r2_paths.py CALIBRATION_SCHEMA)        #
# --------------------------------------------------------------------------- #

def analysis_calibration_block(production_ed: dict, calibration: dict,
                               verdict: dict, t_arith: dict) -> dict:
    """The fields the round-2 analysis reads off ``calibration.json``:
    ``strata["<source>/<class>"] = {ed, ed_ci95, ed_attain,
    ed_attain_components, teacher_bare_mse, floor_mse}`` (the source strata
    mix / ctl AND the source×family strata, both position classes),
    ``gate.{verdict, sentence}`` with the pipeline's tokens, ``t_arith_rms``
    (RMS over layers of the per-layer arithmetic-residual inconsistency),
    ``n_holdout_docs``. Derived from the SAME numbers the richer blocks
    carry — one truth, two spellings."""
    from sisteer.extraction.residual import (POSITION_CLASSES,
                                             VERDICT_NOT_RECOVERED,
                                             VERDICT_RECOVERED,
                                             VERDICT_UNRESOLVABLE)

    # The analysis pipeline's gate tokens (r2_stats.recovery_gate recomputes
    # the verdict from ``strata`` and compares against these), keyed by the
    # trainer's own verdict vocabulary so a wording change cannot desync them.
    gate_tokens = {VERDICT_RECOVERED: "recovered",
                   VERDICT_NOT_RECOVERED: "not_recovered",
                   VERDICT_UNRESOLVABLE: "unresolvable"}
    controls = calibration.get("controls") or {}
    floor = calibration.get("floor") or {}
    strata = {}
    n_docs = {}
    for g, by_cls in production_ed.items():
        src = g.split("/", 1)[0]
        n_docs[g] = (by_cls.get("content") or {}).get("n_docs")
        for cls in POSITION_CLASSES:
            e = by_cls.get(cls) or {}
            comps = {}
            for name, ctl in controls.items():
                v = ((ctl.get("ed") or {}).get(g) or {}).get(cls) or {}
                comps[name] = v.get("ed")
            vals = [v for v in comps.values() if v is not None]
            fl = ((floor.get(g) or floor.get(src) or {}).get(cls) or {}
                  ).get("mse_sum_layers")
            strata[f"{g}/{cls}"] = {
                "ed": e.get("ed"),
                "ed_ci95": ([e.get("lo"), e.get("hi")]
                            if e.get("lo") is not None else None),
                "ed_attain": min(vals) if vals else None,
                "ed_attain_components": comps,
                "teacher_bare_mse": e.get("teacher_deviation"),
                "floor_mse": fl,
                "n_docs": e.get("n_docs"),
                "n_positions": e.get("n_positions"),
            }
    fails = [r["sentence"] for r in verdict["per_stratum"].values()
             if r["verdict"] != VERDICT_RECOVERED]
    incs = [v["rms_inconsistency"] for v in t_arith.values()
            if v.get("rms_inconsistency") is not None]
    return {
        "strata": strata,
        "gate": {"verdict": gate_tokens[verdict["verdict"]],
                 "sentence": "; ".join(fails) if fails else None,
                 "per_stratum_verdict": {s: gate_tokens[r["verdict"]]
                                         for s, r in verdict["per_stratum"].items()}},
        "t_arith_rms": (math.sqrt(sum(x * x for x in incs) / len(incs))
                        if incs else None),
        "t_arith_rms_definition": "RMS over layers of the per-layer RMS "
                                  "distance ||h^V - T_arith|| on the capture "
                                  "holdout (diagnostic, never a gate)",
        "n_holdout_docs": n_docs,
    }


# --------------------------------------------------------------------------- #
# Shared by calibrate + distill: the reference batches, the planted control  #
# and the real-model teacher identity check                                    #
# --------------------------------------------------------------------------- #


def _lk(d: dict, l):
    """One per-layer lookup for both key spellings: tables come back from
    JSON with string keys, in-process with int keys."""
    return d[str(l)] if str(l) in d else d.get(l)


def _reference_batches(tok, train, hp, device, label: str):
    """The first 4–8 training docs as the reference batches the planted
    control is calibrated on and the identity check is measured on."""
    from sisteer.extraction.residual import make_cap_batches

    return make_cap_batches(tok, train[:max(4, min(len(train), 8))],
                            batch_size=hp["batch_size"], max_len=hp["max_len"],
                            device=device, label=label)


def _floor_per_layer(floor_table: dict, layers) -> dict:
    """{layer: per-layer floor MSE} of the mix/content stratum — the
    identity check's numeric-floor tolerance — for the layers that have
    one. ``floor_table`` is ``numeric_floor()["floor"]``, in-process or
    re-read from calibration_controls.json."""
    per = ((floor_table.get("mix") or {}).get("content") or {}
           ).get("per_layer_mse") or {}
    out = {}
    for l in layers:
        v = _lk(per, l)
        if v is not None:
            out[l] = v
    return out


#: The FULL per-layer identity report, written to the run's out dir BEFORE
#: any refusal (the Modal runner keeps a 3000-char stderr tail, which lost
#: gemma's layer<17 lines) — one file per stage.
IDENTITY_REPORT_NAME = "calibration_identity_report.json"
PRECHECK_REPORT_NAME = "precheck_identity_report.json"
IDENTITY_CHECK_REPORT_NAME = "identity_check_report.json"


def _plant_and_check(model, mix, layers, ref_batches, *, seed: int,
                     write_rms: dict):
    """Plant the control C at ``write_rms`` and run the real-model teacher
    identity check R(mix ⊕ C, mix) ≡ +C on the WRITE, plus
    the bf16 perturbation floor on the same batches (diagnostic). Returns
    (C, the planted report, the composite teacher, problems, per-layer
    report, per-layer perturbation floor); an empty ``problems`` is a
    pass."""
    from sisteer.extraction.paired_write import SharedStateWriteObject
    from sisteer.extraction.residual import (plant_control_object,
                                             stream_perturbation_floor,
                                             teacher_identity_problems)

    C, planted = plant_control_object(model, layers, ref_batches, seed=seed,
                                      write_rms=write_rms,
                                      active_frac=PLANTED_ACTIVE_FRAC)
    composite = SharedStateWriteObject([(1.0, mix), (1.0, C), (-1.0, mix)])
    problems, ident = teacher_identity_problems(
        model, ref_batches, composite=composite, reference=C, layers=layers)
    pert = stream_perturbation_floor(model, ref_batches, layers=layers,
                                     ctx_factory=lambda: C.apply(model))
    return C, planted, composite, problems, ident, pert


def write_identity_report(path, *, stage: str, identity: dict, seed: int,
                          planted: dict, write_rms: dict, ident: dict,
                          pert: dict, floor_mse: dict | None,
                          problems: list[str],
                          teacher_contract: dict | None = None,
                          teacher_object: str | None = None) -> Path:
    """The complete identity-check record (every layer, pass or fail):
    the rule, the planted control, the parents' write RMS, per layer the
    write-level check + BOTH hook contracts (composite, reference) + the
    trajectory diagnostics + the bf16 perturbation floor + the
    padded-vs-solo floor; and, when given, the hook contract of the
    persisted teacher V itself (``teacher_hook_contract``). Written
    through the banned-word guard; returns the path (the CLI prints it, a
    refusal names it)."""
    from sisteer.extraction.residual import (IDENTITY_ATOL, IDENTITY_RTOL,
                                             write_calibration)

    per_layer = {}
    for l, row in ident.items():
        fl = _lk(floor_mse, l) if floor_mse else None
        per_layer[int(l)] = {
            **row,
            "bf16_perturbation_floor": _lk(pert, l),
            "padded_vs_solo_floor_rms": (math.sqrt(max(fl, 0.0))
                                         if fl is not None else None),
        }
    payload = {
        "kind": "psr_residual_identity_report",
        "stage": stage,
        "built_at": _now(),
        "passed": not problems,
        "problems": list(problems),
        "rule": (
            "(1) write identity: per layer, on the composite's own captured "
            "pre-write z, the RMS over valid positions of the per-position L2 "
            "‖u_composite(z) − u_reference(z)‖ (both writes evaluated in fp32 "
            f"off the forward) must be <= max({IDENTITY_ATOL:g}, "
            f"{IDENTITY_RTOL:g} · cancellation scale [the largest write RMS "
            "among the composite's signed terms and the reference]); "
            "(2) hook contract, composite AND reference (and the persisted "
            "teacher V when present): on the same captured forward, the "
            "actual post-hook output must equal z + cast_to_stream_dtype(u(z)) "
            "at every target layer, per element, within half an ulp of the "
            "stream dtype at |z + cast(u)| — the final add's rounding only, "
            "never a tolerance relative to the write; trajectory_rms_divergence "
            "and bf16_perturbation_floor are diagnostics (the model's bf16 "
            "chaos floor on these batches), never the gate"),
        "identity": identity,
        "planted_control": {"seed": seed,
                            "active_frac_target": PLANTED_ACTIVE_FRAC,
                            "per_layer": {int(l): v for l, v in planted.items()}},
        "parents_write_rms": {k: {int(l): v for l, v in d.items()}
                              for k, d in write_rms.items()},
        "per_layer": per_layer,
    }
    if teacher_contract is not None:
        payload["teacher_hook_contract"] = {
            "object": teacher_object,
            "passed": all(r["pass"] for r in teacher_contract.values()),
            "per_layer": {int(l): r for l, r in teacher_contract.items()},
        }
    write_calibration(path, payload)
    return Path(path)


def _v_hook_contract(model, V, layers, ref_batches, v_path) -> tuple[list[str], dict]:
    """The hook contract of the teacher that is actually distilled — the
    PERSISTED V reloaded from disk (the derived app's torch.save →
    load_object path), not the in-process planted composite the identity
    check runs on."""
    from sisteer.extraction.residual import hook_contract_problems

    return hook_contract_problems(
        model, ref_batches, obj=V, layers=layers,
        label=f"teacher V ({KIND_V}, reloaded from {v_path})")


# --------------------------------------------------------------------------- #
# calibrate                                                                    #
# --------------------------------------------------------------------------- #


def _sweep_payloads_on_disk(out_path) -> dict[str, dict]:
    """The finished per-multiplier payloads an existing sweep file holds
    (``sweep.per_mult``; a pre-sweep single file counts as its own one
    multiplier), or ``{}`` when there is no readable file. Never raises:
    an unreadable file is reported and treated as absent (a fresh run)."""
    path = Path(out_path)
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as e:
        print(f"[psr-residual] {path} is unreadable ({type(e).__name__}: {e}) "
              "— ignoring it; the sweep starts fresh", flush=True)
        return {}
    if not isinstance(doc, dict):
        return {}
    sweep = doc.get("sweep")
    if isinstance(sweep, dict) and isinstance(sweep.get("per_mult"), dict):
        return {str(k): v for k, v in sweep["per_mult"].items()
                if isinstance(v, dict)}
    m = ((doc.get("identity") or {}).get("trainer") or {}).get("lr_mult")
    if m is not None and isinstance(doc.get("controls"), dict):
        return {format_mult(float(m)): doc}
    return {}


def reusable_payload(prior: dict | None, identity: dict) -> tuple[bool, str]:
    """Whether a finished calibration payload stands in for running this
    multiplier again: it must be a complete controls payload (both
    controls with their ED tables, the attainable numbers) AND its recorded
    identity must equal ``identity`` on every field (:func:`identity_diffs`
    — trainer incl. lr / budget / seed, parents, source pools, capture
    split, layers, measurement). Returns (ok, why)."""
    if not isinstance(prior, dict):
        return False, "no finished payload on disk"
    if prior.get("kind") != "psr_residual_calibration_controls":
        return False, f"payload kind {prior.get('kind')!r} is not a controls payload"
    controls = prior.get("controls") or {}
    for name in ("R_mix_empty", "R_mixC_mix"):
        if not isinstance((controls.get(name) or {}).get("ed"), dict):
            return False, f"control {name} has no ED table (unfinished)"
    if not isinstance(prior.get("attainable_ed_content"), dict):
        return False, "no attainable_ed_content (unfinished)"
    rec = prior.get("identity")
    if not isinstance(rec, dict) or not rec.get("fingerprint"):
        return False, "payload records no identity fingerprint"
    diffs = identity_diffs(rec, identity)
    if diffs:
        return False, ("recorded identity differs from this run's on "
                       + "; ".join(diffs))
    return True, f"identity fingerprint {identity['fingerprint'][:16]} matches"


DOC_RECORDS_NAME = "calibration_doc_records.jsonl"


def _prior_doc_records(out_path) -> dict[float, list[dict]]:
    """The rows of an existing ``calibration_doc_records.jsonl`` by
    multiplier — read ONCE before the sweep rewrites it, so a reused
    multiplier's records (not regenerated) ride along; a missing / torn
    file → nothing."""
    recs_path = Path(out_path).with_name(DOC_RECORDS_NAME)
    if not recs_path.is_file():
        return {}
    out: dict[float, list[dict]] = {}
    try:
        for line in recs_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            out.setdefault(float(r["lr_mult"]), []).append(r)
    except (ValueError, TypeError, KeyError, OSError) as e:
        print(f"[psr-residual] {recs_path} is unreadable ({e}); the reused "
              "multipliers' doc records are not carried over", flush=True)
        return {}
    return out


def _write_doc_records(out_path, mults: list[float],
                       records_by_mult: dict[float, list[dict]]) -> Path:
    """``calibration_doc_records.jsonl`` for the multipliers finished SO FAR,
    in sweep order — rewritten after every multiplier (like the sweep file)
    so an interrupted sweep keeps its finished multipliers' records."""
    recs_path = Path(out_path).with_name(DOC_RECORDS_NAME)
    with open(recs_path, "w", encoding="utf-8") as f:
        for m in mults:
            for r in records_by_mult.get(float(m), []):
                f.write(json.dumps(r) + "\n")
    return recs_path


def checkpoint_spec(checkpoint_every, out_dir, subdir: str, key: str, *,
                    fresh: bool = False) -> dict | None:
    """The trainer's resumable-checkpoint kwargs for ONE fit, or None when
    checkpointing is off (``checkpoint_every`` falsy). ``key`` binds the
    checkpoint to the fit's full identity (the calibration identity
    fingerprint, plus the control's name) so a stale file can never resume
    a different fit — the fits app's convention (its key is the cell fit
    fingerprint). ``fresh`` (``--no-resume``) DISCARDS an existing
    checkpoint dir first, so the fit really starts from zero and writes new
    checkpoints — a forced fresh run that silently resumed the old
    trajectory would not be one."""
    if not checkpoint_every:
        return None
    ck_dir = Path(out_dir) / "ckpt" / subdir
    if fresh and ck_dir.exists():
        import shutil

        shutil.rmtree(ck_dir)
        print(f"[psr-residual] --no-resume: discarded the existing checkpoint "
              f"dir {ck_dir} — this fit starts from zero", flush=True)
    return {"checkpoint_every": int(checkpoint_every),
            "checkpoint_dir": str(ck_dir),
            "checkpoint_key": key}


def _write_sweep_file(out_path, per_mult: dict, mults: list[float], *,
                      budget: dict, complete: bool) -> dict:
    """The sweep document: top level = the SELECTED multiplier's calibration
    (the analysis shape, unchanged), plus ``sweep`` with every multiplier's
    payload, the comparison table and the selection. Rewritten after each
    multiplier finishes (``complete`` False until the last) so a late crash
    keeps the finished ones; ``distill`` refuses an incomplete sweep."""
    from sisteer.extraction.residual import write_calibration

    selected, table, selection = select_lr_mult(per_mult)
    ref = selected if selected is not None else selection["reference_lr_mult_key"]
    doc = {**per_mult[ref],
           "sweep": {"lr_mults": [float(m) for m in mults],
                     "finished_lr_mults": [float(per_mult[k]["trainer"]["lr_mult"])
                                           for k in per_mult],
                     "selected_lr_mult": (
                         float(per_mult[selected]["trainer"]["lr_mult"])
                         if selected is not None else None),
                     "selection": selection,
                     "top_level_is": (
                         "the selected multiplier's calibration"
                         if selected is not None else
                         f"REFERENCE ONLY: lr_mult {format_mult(per_mult[ref]['trainer']['lr_mult'])} "
                         "(the rung the refusal names), NOT selected — the sweep "
                         "was refused; `distill` needs an explicit --lr-mult"),
                     "selection_rule": SELECTION_RULE,
                     "budget": dict(budget),
                     "table": table,
                     "per_mult": per_mult,
                     "complete": bool(complete)}}
    write_calibration(out_path, doc)
    return doc


def run_calibration(model, tok, mix, ctl, train, hold, hp_by_mult: dict, *,
                    layers, out_path, n_boot, seed, identity_by_mult: dict,
                    log_prefix="calib", wandb_run=None, V=None, v_path=None,
                    checkpoint_every=None, resume: bool = True) -> dict:
    """Calibration over the lr-multiplier sweep: the floor, the
    planted C, the identity check (write identity + hook contracts, plus
    the persisted V's contract when ``V`` is given) ONCE — none of them
    depends on lr — then controls (a) and (b) at EVERY multiplier in
    ``hp_by_mult`` (``{lr_mult: hp}``, matched by ``identity_by_mult``),
    each stamped with its own identity so a production run can prove it is
    same-everything at that lr. Writes/rewrites ``out_path`` as the sweep
    document (:func:`_write_sweep_file`) and returns it.

    ``resume`` (default): a multiplier whose finished payload is already in
    ``out_path`` under THIS identity (:func:`reusable_payload`) is reused,
    not re-run — an interrupted sweep continues where it stopped, and a
    production retry never repeats finished controls. Payloads of other
    identities, and of multipliers not in this sweep, are dropped (loudly).
    ``checkpoint_every`` (optimizer steps) makes each control's fit
    resumable through the trainer's checkpoints (:func:`checkpoint_spec`,
    keyed by that multiplier's identity fingerprint + the control name)."""
    from sisteer.extraction.paired_write import (SharedStateWriteObject,
                                                 paired_gate_liveness)
    from sisteer.extraction.residual import (attainable_from_controls,
                                             measure_write_rms, numeric_floor)

    mults = [float(m) for m in hp_by_mult]
    hp0 = hp_by_mult[mults[0]]
    budget = {"epochs_cap": int(hp0["epochs"]), "patience": int(hp0["patience"])}
    for m in mults:  # the budget is one number per sweep, by construction
        assert (int(hp_by_mult[m]["epochs"]), int(hp_by_mult[m]["patience"])) == \
            (budget["epochs_cap"], budget["patience"])
    device = next(model.parameters()).device
    floor = numeric_floor(model, tok, hold, layers=layers,
                          batch_size=hp0["batch_size"], max_len=hp0["max_len"])
    ref_batches = _reference_batches(tok, train, hp0, device, "calib reference")
    write_rms = {"mix": measure_write_rms(model, mix, ref_batches),
                 "ctl": measure_write_rms(model, ctl, ref_batches)}
    target = {l: max(write_rms["mix"][l], write_rms["ctl"][l]) for l in layers}
    _C, planted, composite, problems, ident, pert = _plant_and_check(
        model, mix, layers, ref_batches, seed=seed + 1, write_rms=target)
    v_contract = None
    if V is not None:
        v_problems, v_contract = _v_hook_contract(model, V, layers, ref_batches,
                                                  v_path)
        problems = list(problems) + list(v_problems)
    # the FULL per-layer record lands on disk before any verdict is acted on
    # (the identity check does not depend on lr: stamped with the identity
    # of the first multiplier; the sweep's other identities differ only in
    # trainer.lr / trainer.lr_mult)
    report_path = write_identity_report(
        Path(out_path).with_name(IDENTITY_REPORT_NAME), stage="calibration",
        identity=identity_by_mult[mults[0]], seed=seed + 1, planted=planted,
        write_rms=write_rms, ident=ident, pert=pert,
        floor_mse=_floor_per_layer(floor["floor"], layers), problems=problems,
        teacher_contract=v_contract,
        teacher_object=str(v_path) if v_path is not None else None)
    print(f"[psr-residual] identity report -> {report_path}", flush=True)
    if problems:
        raise ResidualError(
            "real-model teacher identity check FAILED — R(mix ⊕ C, mix) does "
            f"not reproduce the +C write, or a hook contract is violated (full "
            f"per-layer report: {report_path}):\n  - " + "\n  - ".join(problems))
    live = paired_gate_liveness(model, tok, obj=composite,
                                examples=[d.example() for d in train],
                                max_len=hp0["max_len"])
    planted_live = {int(l): live["terms"][1][l] for l in layers}
    thin = [l for l, f in planted_live.items() if f < PLANTED_ACTIVE_FRAC]
    identity_block = {
        "passed": True, "per_layer": ident, "bf16_perturbation_floor": pert,
        "report": str(report_path),
        "teacher_hook_contract_passed": (
            all(r["pass"] for r in v_contract.values())
            if v_contract is not None else None),
        "teacher_object": str(v_path) if v_path is not None else None,
    }

    prior = _sweep_payloads_on_disk(out_path) if resume else {}
    prior_records = _prior_doc_records(out_path) if resume else {}
    wanted = {format_mult(m) for m in mults}
    for extra in sorted(set(prior) - wanted):
        print(f"[psr-residual] {out_path}: dropping the finished lr_mult="
              f"{extra} payload — not in this sweep {sorted(wanted)}",
              flush=True)
    per_mult: dict[str, dict] = {}
    reused: set[float] = set()
    records_by_mult: dict[float, list[dict]] = {}
    doc: dict = {}
    for m in mults:
        hp = hp_by_mult[m]
        key = format_mult(m)
        tag = f"{log_prefix}/lrmult{key}"
        ok, why = reusable_payload(prior.get(key), identity_by_mult[m])
        if ok:
            per_mult[key] = prior[key]
            reused.add(float(m))
            records_by_mult[float(m)] = prior_records.get(float(m), [])
            print(f"[psr-residual] calibration controls at lr_mult={key}: "
                  f"REUSING the finished payload in {out_path} ({why}) — "
                  "not re-run", flush=True)
            doc = _write_sweep_file(out_path, per_mult, mults, budget=budget,
                                    complete=(m == mults[-1]))
            _write_doc_records(out_path, mults, records_by_mult)
            continue
        if key in prior:
            print(f"[psr-residual] calibration controls at lr_mult={key}: "
                  f"the payload in {out_path} is NOT reusable ({why}) — "
                  "re-running", flush=True)
        print(f"[psr-residual] calibration controls at lr_mult={key} "
              f"(lr={hp['lr']:.4g}, cap {budget['epochs_cap']} epochs, "
              f"patience {budget['patience']})", flush=True)
        fp = identity_by_mult[m]["fingerprint"]
        ck = {name: checkpoint_spec(checkpoint_every, Path(out_path).parent,
                                    f"lrmult{key}/{name}",
                                    f"psr_residual_calib|{fp}|{name}",
                                    fresh=not resume)
              for name in ("R_mix_empty", "R_mixC_mix")}
        # (a) R(mix, ∅): the mix parent alone, in-class by construction.
        teacher_a = SharedStateWriteObject([(1.0, mix)])
        _s_a, diag_a = _distill(model, tok, teacher_a, train, hold, hp,
                                tag=f"{tag}/R_mix_empty",
                                log_fn=_log_fn_factory(f"{tag}/R_mix_empty",
                                                       wandb_run), n_boot=n_boot,
                                checkpoint=ck["R_mix_empty"])
        # (b) R(mix ⊕ C, mix): the coherent teacher is exactly +C.
        _s_b, diag_b = _distill(model, tok, composite, train, hold, hp,
                                tag=f"{tag}/R_mixC_mix",
                                log_fn=_log_fn_factory(f"{tag}/R_mixC_mix",
                                                       wandb_run), n_boot=n_boot,
                                checkpoint=ck["R_mixC_mix"])
        diag_a, recs_a = _strip_records(diag_a)
        diag_b, recs_b = _strip_records(diag_b)
        diag_a["stop_reason"] = stop_reason(diag_a, hp)
        diag_b["stop_reason"] = stop_reason(diag_b, hp)
        attain = attainable_from_controls([diag_a["ed"], diag_b["ed"]])
        per_mult[key] = {
            "kind": "psr_residual_calibration_controls",
            "built_at": _now(),
            "identity": identity_by_mult[m],
            "trainer": {k: hp[k] for k in TRAINER_HP_KEYS},
            "floor": floor["floor"],
            "floor_compositions": {"a": floor["composition_a"],
                                   "b": floor["composition_b"]},
            "parents_write_rms": {k: {int(l): v for l, v in d.items()}
                                  for k, d in write_rms.items()},
            "planted_control": {"seed": seed + 1,
                                "active_frac_target": PLANTED_ACTIVE_FRAC,
                                "per_layer": planted,
                                "realized_active_frac_under_teacher": planted_live,
                                "layers_below_target": thin},
            "teacher_identity_check": identity_block,
            "controls": {"R_mix_empty": diag_a, "R_mixC_mix": diag_b},
            "attainable_ed_content": attain,
            "attainable_rule": "min over controls (a) R(mix, ∅) and (b) "
                               "R(mix ⊕ C, mix) of the pooled content-class ED "
                               "per source stratum (plan §2.5)",
        }
        records_by_mult[float(m)] = [{"lr_mult": m, "control": name, **r}
                                     for name, recs in (("R_mix_empty", recs_a),
                                                        ("R_mixC_mix", recs_b))
                                     for r in recs]
        print(f"[psr-residual] lr_mult={key}: attainable ED {attain}; "
              f"(a) {diag_a['stop_reason']}; (b) {diag_b['stop_reason']}",
              flush=True)
        doc = _write_sweep_file(out_path, per_mult, mults, budget=budget,
                                complete=(m == mults[-1]))
        _write_doc_records(out_path, mults, records_by_mult)
        if wandb_run is not None:
            try:
                wandb_run.summary[f"calib/lrmult{key}/attainable_ed_content"] = attain
                for name, d_ in (("R_mix_empty", diag_a), ("R_mixC_mix", diag_b)):
                    wandb_run.summary[f"calib/lrmult{key}/{name}/epochs_run"] = d_["epochs_run"]
                    wandb_run.summary[f"calib/lrmult{key}/{name}/stop_reason"] = d_["stop_reason"]
            except Exception:  # noqa: BLE001 — telemetry only
                pass
    print(f"[psr-residual] lr-mult sweep {mults}: {_selection_sentence(doc['sweep'])} — "
          f"{json.dumps({k: v['score_worst_stratum_attainable_ed'] for k, v in doc['sweep']['table'].items()})}",
          flush=True)
    if wandb_run is not None:
        try:
            wandb_run.summary["calib/selected_lr_mult"] = doc["sweep"]["selected_lr_mult"]
        except Exception:  # noqa: BLE001
            pass
    return doc


def _identity_diff_paths(a: dict, b: dict) -> list[str]:
    return [d.split(" ", 1)[0] for d in identity_diffs(a, b)]


#: The only identity fields two multipliers of ONE sweep may differ on.
_SWEEP_VARIABLE_FIELDS = frozenset({"trainer.lr", "trainer.lr_mult"})


def merge_calibration_parts(parts: list, mults: list[float], out_path,
                            *, doc_records: bool = True) -> dict:
    """Merge the sweep files ``parts`` (each written by ``calibrate`` for
    one or more multipliers — the derived app runs one multiplier per
    container) into ONE complete sweep document at ``out_path`` covering
    EXACTLY ``mults``, selecting with the same :data:`SELECTION_RULE` the
    in-process sweep applies. Refuses (``ResidualError``): a part that is
    not a complete sweep, a multiplier finished twice, a multiplier missing
    or not requested, identities that differ anywhere but ``trainer.lr`` /
    ``trainer.lr_mult``, or budgets that differ. The parts' sibling
    ``calibration_doc_records.jsonl`` files are concatenated beside the
    merged file in sweep order."""
    mults = [float(m) for m in mults]
    wanted = [format_mult(m) for m in mults]
    per_mult: dict[str, dict] = {}
    origin: dict[str, Path] = {}
    budgets: dict[str, dict] = {}
    missing_files: list[str] = []
    for part in parts:
        path = Path(part)
        if not path.is_file():
            missing_files.append(str(path))
            continue
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as e:
            raise ResidualError(f"merge-calibration: {path} unreadable: {e}") from e
        sweep = doc.get("sweep") if isinstance(doc, dict) else None
        if not isinstance(sweep, dict):
            raise ResidualError(
                f"merge-calibration: {path} is not a sweep file (no `sweep`)")
        if not sweep.get("complete", False):
            raise ResidualError(
                f"merge-calibration: {path} holds an INCOMPLETE sweep "
                f"(finished {list(sweep.get('per_mult') or {})} of "
                f"{sweep.get('lr_mults')})")
        for key, payload in (sweep.get("per_mult") or {}).items():
            key = str(key)
            if key not in wanted:
                raise ResidualError(
                    f"merge-calibration: {path} carries lr_mult={key}, which "
                    f"is not in the requested sweep {wanted}")
            if key in per_mult:
                raise ResidualError(
                    f"merge-calibration: lr_mult={key} finished twice "
                    f"({origin[key]} and {path}) — one payload per multiplier")
            ok, why = reusable_payload(payload, payload.get("identity") or {})
            if not ok:
                raise ResidualError(
                    f"merge-calibration: {path} lr_mult={key}: {why}")
            per_mult[key] = payload
            origin[key] = path
            budgets[key] = dict(sweep.get("budget") or {})
    missing = [k for k in wanted if k not in per_mult]
    if missing or missing_files:
        raise ResidualError(
            f"merge-calibration: no finished payload for lr_mult={missing} "
            f"among {[str(p) for p in parts]}"
            + (f" (missing part file(s): {missing_files})" if missing_files else "")
            + f" — the sweep {wanted} is not complete; run calibrate for the "
              "missing multiplier(s) first")
    ref_key = wanted[0]
    ref_ident = per_mult[ref_key]["identity"]
    for key in wanted[1:]:
        bad = [d for d in _identity_diff_paths(ref_ident,
                                               per_mult[key]["identity"])
               if d not in _SWEEP_VARIABLE_FIELDS]
        if bad:
            raise ResidualError(
                f"merge-calibration: lr_mult={ref_key} ({origin[ref_key]}) and "
                f"lr_mult={key} ({origin[key]}) were calibrated under DIFFERENT "
                f"identities on {bad} — one sweep is one identity except for "
                "the lr; these parts do not form a sweep")
        if budgets[key] != budgets[ref_key]:
            raise ResidualError(
                f"merge-calibration: budget differs between lr_mult={ref_key} "
                f"{budgets[ref_key]} and lr_mult={key} {budgets[key]}")
        if float(per_mult[key]["trainer"]["lr_mult"]) != float(key):
            raise ResidualError(
                f"merge-calibration: {origin[key]} lr_mult={key} payload "
                f"records trainer.lr_mult={per_mult[key]['trainer']['lr_mult']}")
    ordered = {k: per_mult[k] for k in wanted}
    budget = budgets[ref_key] or {
        "epochs_cap": int(ref_ident["trainer"]["epochs"]),
        "patience": int(ref_ident["trainer"]["patience"])}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = _write_sweep_file(out_path, ordered, mults, budget=budget,
                            complete=True)
    if doc_records:
        recs_out = out_path.with_name(DOC_RECORDS_NAME)
        n = 0
        with open(recs_out, "w", encoding="utf-8") as f:
            for key in wanted:
                src = origin[key].with_name(DOC_RECORDS_NAME)
                if not src.is_file():
                    continue
                for line in src.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    if format_mult(float(r["lr_mult"])) == key:
                        f.write(json.dumps(r) + "\n")
                        n += 1
        print(f"[psr-residual] merge-calibration: {n} doc records -> {recs_out}",
              flush=True)
    print(f"[psr-residual] merge-calibration: {wanted} from {len(parts)} "
          f"part(s) -> {out_path}; {_selection_sentence(doc['sweep'])} — "
          f"{json.dumps({k: v['score_worst_stratum_attainable_ed'] for k, v in doc['sweep']['table'].items()})}",
          flush=True)
    return doc


# --------------------------------------------------------------------------- #
# distill                                                                      #
# --------------------------------------------------------------------------- #


def _selected_by(sweep: dict | None, lr_mult: float) -> str | None:
    """What derived_meta.json says about the production multiplier: the
    :data:`SELECTION_RULE` when it is the sweep's selection; an explicit
    user override otherwise (after a refusal, or against a different
    selection); None for a pre-sweep calibration."""
    if not sweep:
        return None
    chosen = sweep.get("selected_lr_mult")
    sel = sweep.get("selection") or {}
    if chosen is not None and format_mult(chosen) == format_mult(lr_mult):
        if sel.get("verdict") == VERDICT_SINGLE_RUNG:
            return (f"explicit single --lr-mult {format_mult(lr_mult)}: no "
                    "ladder was swept, nothing was selected by the rule")
        return SELECTION_RULE
    if chosen is None:
        return (f"user override: explicit --lr-mult {format_mult(lr_mult)} after "
                f"the sweep's {sel.get('verdict')} refusal ({sel.get('reason')})")
    return (f"user override: explicit --lr-mult {format_mult(lr_mult)} (the "
            f"sweep selected {format_mult(chosen)})")


def run_production(model, tok, V, mix, ctl, train, hold, hp, calibration, *,
                   layers, out_dir: Path, capture_manifest: dict, sources: dict,
                   pos_prov: dict, neg_prov: dict, shared: dict, mix_si: str,
                   v_path: Path, n_boot: int, seed: int, identity: dict,
                   wandb_run=None, checkpoint: dict | None = None,
                   sweep: dict | None = None) -> dict:
    """The production distillation of V at ``hp`` (ONE lr multiplier),
    gated by ``calibration`` — the per-multiplier payload of the sweep at
    that multiplier (:func:`calibration_for_mult`); ``sweep`` (the sweep
    block) is stamped on the artifacts so the selection is auditable."""
    import torch

    # The calibration gates THIS run only if it was measured under the same
    # identity — every field, not just lr.
    require_same_identity(calibration, identity)

    from sisteer.eval.object_identity import derived_object_problems
    from sisteer.extraction.paired_write import paired_gate_liveness
    from sisteer.extraction.psr import psr_gate_liveness
    from sisteer.extraction.residual import (
        arithmetic_residual_inconsistency, assert_no_banned_words,
        make_cap_batches, recovery_verdict, write_calibration,
        write_capture_manifest)

    device = next(model.parameters()).device
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_batches = _reference_batches(tok, train, hp, device,
                                     "precheck reference")
    planted_seed = int(calibration["planted_control"]["seed"])
    wr = calibration["parents_write_rms"]
    target = {int(l): max(float(_lk(wr["mix"], l)), float(_lk(wr["ctl"], l)))
              for l in layers}
    floor_mse = _floor_per_layer(calibration["floor"], layers)
    precheck_report = out_dir / PRECHECK_REPORT_NAME

    def precheck() -> list[str]:
        # the identity check again, on the calibration's own planted C —
        # PLUS the hook contract of V itself, the object the trainer is
        # about to capture targets from — and the FULL per-layer record on
        # disk before the trainer rules
        _c, planted_p, _comp, problems, ident, pert = _plant_and_check(
            model, mix, layers, ref_batches, seed=planted_seed,
            write_rms=target)
        v_problems, v_contract = _v_hook_contract(model, V, layers,
                                                  ref_batches, v_path)
        problems = list(problems) + list(v_problems)
        write_identity_report(
            precheck_report, stage="production_precheck", identity=identity,
            seed=planted_seed, planted=planted_p,
            write_rms={k: {int(l): float(v) for l, v in wr[k].items()}
                       for k in ("mix", "ctl")},
            ident=ident, pert=pert, floor_mse=floor_mse, problems=problems,
            teacher_contract=v_contract, teacher_object=str(v_path))
        print(f"[psr-residual] identity report -> {precheck_report}",
              flush=True)
        return problems

    from sisteer.extraction.residual import TeacherPrecheckError

    tag = "psr_residual/production"
    try:
        student, diag = _distill(model, tok, V, train, hold, hp, tag=tag,
                                 log_fn=_log_fn_factory(tag, wandb_run),
                                 n_boot=n_boot, teacher_precheck=precheck,
                                 checkpoint=checkpoint)
    except TeacherPrecheckError as e:
        raise ResidualError(
            f"{e}\n  (full per-layer report: {precheck_report})") from e
    diag, records = _strip_records(diag)
    diag["stop_reason"] = stop_reason(diag, hp)
    verdict = recovery_verdict(diag["ed"],
                               calibration["attainable_ed_content"],
                               calibration["floor"])
    t_arith = arithmetic_residual_inconsistency(
        model, make_cap_batches(tok, hold, batch_size=hp["batch_size"],
                                max_len=hp["max_len"], device=device,
                                label="t_arith"),
        pos=mix, neg=ctl, paired=V, layers=layers)
    train_examples = [d.example() for d in train]
    liveness = psr_gate_liveness(model, tok, obj=student,
                                 examples=train_examples,
                                 max_len=hp["max_len"],
                                 n_probe=len(train_examples))
    v_live = paired_gate_liveness(model, tok, obj=V, examples=train_examples,
                                  max_len=hp["max_len"])

    derived_id = resid_derived_id(mix_si)
    cm_path = out_dir / "capture_manifest.json"
    write_capture_manifest(cm_path, capture_manifest, sources=sources)
    v_sha = sha256_file(v_path)
    live_by_layer = {int(l): v for l, v in liveness.items()}
    provenance = {
        "kind": "psr",
        "derivation": DERIVATION_RESID,
        "derived_id": derived_id,
        **{k: shared[k] for k in ("model_key", "model_id", "model_revision",
                                  "positions", "response_header_len",
                                  "target_layers", "template_hash",
                                  "scale_matched", "comparison")},
        "parent_pos": pos_prov,
        "parent_neg": neg_prov,
        "teacher": {"kind": KIND_V, "derived_id": v_derived_id(mix_si),
                    "object_path": str(v_path),
                    "object_sha256": v_sha,
                    "application_order": V.application_order},
        "capture_manifest_sha256": sha256_file(cm_path),
        "built_at": _now(),
        "builder": "scripts/sdfbase/psr_residual_distill.py distill",
    }
    obj_path = out_dir / "object.pt"
    tmp = out_dir / "object.pt.tmp"
    student.save(tmp)
    blob = torch.load(tmp, map_location="cpu", weights_only=True)
    blob["provenance"] = provenance
    torch.save(blob, obj_path)
    tmp.unlink(missing_ok=True)
    Path(str(tmp) + ".provenance.json").unlink(missing_ok=True)
    trainer_cfg = {k: hp[k] for k in TRAINER_HP_KEYS}
    trainer_cfg.update({
        "source_weights": diag["source_weights"],
        "position_classes": diag["position_classes"],
        "loss_layers": diag["loss_layers"],
        "epochs_run": diag["epochs_run"], "best_epoch": diag["best_epoch"],
        "stopped_early": diag["stopped_early"],
        "stop_reason": diag["stop_reason"],
        "optimizer_steps_run": diag["optimizer_steps_run"],
        "lr_mult_selected_by": _selected_by(sweep, float(hp["lr_mult"])),
    })
    calibration_summary = {
        "identity": calibration["identity"],
        "attainable_ed_content": calibration["attainable_ed_content"],
        "floor": calibration["floor"],
        "planted_control": calibration["planted_control"],
        "controls": {k: {"ed": v["ed"], "epochs_run": v["epochs_run"],
                         "best_epoch": v["best_epoch"],
                         "best_heldout_loss": v["best_heldout_loss"],
                         "stop_reason": v.get("stop_reason")}
                     for k, v in calibration["controls"].items()},
        "lr_mult": float(hp["lr_mult"]),
        "lr_mult_sweep": sweep_summary(sweep),
        "budget": {"epochs_cap": int(hp["epochs"]),
                   "patience": int(hp["patience"])},
        "production_ed": diag["ed"],
        "production_stop_reason": diag["stop_reason"],
        "recovery": verdict,
        "arithmetic_residual_inconsistency": t_arith,
    }
    meta = {**provenance, "object_sha256": sha256_file(obj_path),
            "layers": [int(l) for l in student.layers],
            "trainer": trainer_cfg,
            "calibration": calibration_summary,
            "recovery_verdict": verdict["verdict"],
            "gate_liveness": live_by_layer}
    (out_dir / "derived_meta.json").write_text(
        json.dumps(meta, indent=1, default=str) + "\n", encoding="utf-8")
    report = {**diag, "gate_liveness": live_by_layer,
              "teacher_gate_liveness": v_live, "recovery": verdict,
              "arithmetic_residual_inconsistency": t_arith}
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")
    with open(out_dir / "doc_records.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    write_calibration(out_dir / "calibration.json", {
        "kind": "psr_residual_calibration",
        "derived_id": derived_id,
        "built_at": _now(),
        **calibration_summary,
        **analysis_calibration_block(diag["ed"], calibration, verdict, t_arith),
        "sentences": [r["sentence"] for r in verdict["per_stratum"].values()],
        "trainer": trainer_cfg,
    })
    # gate_liveness.json ({layer: live_frac}) beside the distilled object AND
    # beside its teacher V — the analysis' liveness column for both, in the
    # telemetry script's own shapes (gate_liveness.py owns them).
    write_liveness(out_dir, live_by_layer,
                   detail={"kind": "psr", "object_sha256": meta["object_sha256"],
                           "n_probe_docs": len(train_examples)})
    v_flat, v_detail = paired_liveness_detail(v_live, KIND_V)
    write_liveness(Path(v_path).parent, v_flat,
                   detail={**v_detail, "object_sha256": v_sha,
                           "n_probe_docs": len(train_examples)})
    for name in ("derived_meta.json", "report.json", "calibration.json"):
        assert_no_banned_words((out_dir / name).read_text(encoding="utf-8"),
                               where=str(out_dir / name))
    problems = derived_object_problems(obj_path, derived_id=derived_id,
                                       parents_root=None)
    if problems:
        raise ResidualError(
            f"{obj_path} failed the derived-object verdict right after being "
            "written:\n  - " + "\n  - ".join(problems))
    # the registry record for the distilled difference — AFTER the verdict:
    # the teacher V's id, the selected lr multiplier and the
    # recovery verdict ride extra
    registry_emit_derived(
        out_dir, meta, object_kind="relu_steer_distilled_difference",
        derivation="residual_distillation",
        script="scripts/sdfbase/psr_residual_distill.py distill",
        extra={"teacher_object_id": v_sha, "lr_mult": float(hp["lr_mult"]),
               "lr_mult_selected_by": trainer_cfg.get("lr_mult_selected_by"),
               "recovery_verdict": verdict["verdict"],
               "epochs_run": diag.get("epochs_run"),
               "best_epoch": diag.get("best_epoch"),
               "stop_reason": diag.get("stop_reason")})
    return {"object": str(obj_path), "verdict": verdict["verdict"],
            "ed": {s: diag["ed"][s]["content"]["ed"] for s in ("mix", "ctl")
                   if s in diag["ed"]},
            "epochs_run": diag["epochs_run"], "stop_reason": diag["stop_reason"],
            "lr_mult": float(hp["lr_mult"]), "wall_s": diag["wall_s"]}


# --------------------------------------------------------------------------- #
# identity-check (the identity-only preflight)                                 #
# --------------------------------------------------------------------------- #


def run_identity_check(model, tok, mix, ctl, V, train, hp, *, layers,
                       out_dir: Path, seed: int, identity: dict,
                       v_path) -> dict:
    """Everything the calibration's teacher gate checks, and nothing else:
    the parents' write RMS on the reference batches, the planted C, the
    real-model identity check R(mix ⊕ C, mix) ≡ +C (write identity + both
    hook contracts) with the bf16 perturbation floor, and the hook
    contract of the persisted V reloaded from ``v_path``. Writes
    ``identity_check_report.json`` (stage ``identity_check``) to
    ``out_dir`` BEFORE any verdict is acted on and returns a compact
    summary; the caller refuses on problems."""
    from sisteer.extraction.residual import measure_write_rms

    device = next(model.parameters()).device
    ref_batches = _reference_batches(tok, train, hp, device,
                                     "identity reference")
    write_rms = {"mix": measure_write_rms(model, mix, ref_batches),
                 "ctl": measure_write_rms(model, ctl, ref_batches)}
    target = {l: max(write_rms["mix"][l], write_rms["ctl"][l]) for l in layers}
    _C, planted, _composite, problems, ident, pert = _plant_and_check(
        model, mix, layers, ref_batches, seed=seed + 1, write_rms=target)
    v_problems, v_contract = _v_hook_contract(model, V, layers, ref_batches,
                                              v_path)
    problems = list(problems) + list(v_problems)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = write_identity_report(
        out_dir / IDENTITY_CHECK_REPORT_NAME, stage="identity_check",
        identity=identity, seed=seed + 1, planted=planted, write_rms=write_rms,
        ident=ident, pert=pert, floor_mse=None, problems=problems,
        teacher_contract=v_contract, teacher_object=str(v_path))
    print(f"[psr-residual] identity report -> {report_path}", flush=True)
    summary = {
        int(l): {
            "write_disc": ident[l]["rms_write_discrepancy"],
            "write_tol": ident[l]["tolerance"],
            "composite_max_half_ulps":
                ident[l]["hook_contract_composite"]["max_err_half_ulps"],
            "composite_violations":
                ident[l]["hook_contract_composite"]["n_violations"],
            "reference_max_half_ulps":
                ident[l]["hook_contract_reference"]["max_err_half_ulps"],
            "reference_violations":
                ident[l]["hook_contract_reference"]["n_violations"],
            "v_max_half_ulps": v_contract[l]["max_err_half_ulps"],
            "v_violations": v_contract[l]["n_violations"],
            "v_write_active_frac": v_contract[l]["write_active_frac"],
            "stream_dtype": v_contract[l]["stream_dtype"],
        } for l in layers}
    return {"report": str(report_path), "passed": not problems,
            "problems": list(problems), "per_layer": summary,
            "n_reference_positions": v_contract[layers[0]]["n_positions"]}


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def _resid_dir(args) -> Path:
    return Path(args.out_root) / "psr_residual" / args.model_key / \
        resid_derived_id(args.mix_si)


def _capture(args):
    return build_capture_set(args.pools, args.mix_si, args.ctl_si,
                             frac=args.holdout_frac, seed=args.seed,
                             budget_tokens=getattr(args, "capture_budget_tokens",
                                                   None))


def cmd_compose_v(args) -> int:
    p = compose_v(args.root, args.out_root, args.model_key, args.mix_si,
                  args.ctl_si, pools_dir=args.pools, emit=not getattr(args, "no_emit", False))
    print(f"[psr-residual] V -> {p}")
    return 0


def cmd_emit_v(args) -> int:
    ev = emit_v(args.root, args.out_root, args.model_key, args.mix_si, args.ctl_si)
    print(f"[psr-residual] emit-v: {ev.get('status')} ({ev.get('campaign')}; {len(ev.get('records') or [])} record file(s))")
    return 0 if ev.get("status") == "emitted" else 2


def cmd_liveness_receipt(args) -> int:
    vdir = Path(args.out_root) / KIND_V / args.model_key / v_derived_id(args.mix_si)
    ctx = {k: v for k, v in (("slurm_job_id", args.slurm_job_id), ("node", args.node), ("code_sha", args.code_sha),
                             ("n_probe", args.n_probe), ("pairs", args.pairs)) if v not in (None, "")}
    write_liveness_receipt(vdir, writer=args.writer, context=ctx)
    return 0


def cmd_verify_v(args) -> int:
    problems = verify_v(args.root, args.out_root, args.model_key, args.mix_si,
                        args.ctl_si)
    v_path = Path(args.out_root) / KIND_V / args.model_key / \
        v_derived_id(args.mix_si) / "object.pt"
    if problems:
        raise ResidualError(
            f"persisted V {v_path} does not pass the derived-object verdict "
            "against the live bank:\n  - " + "\n  - ".join(problems))
    print(f"[psr-residual] V verified against the live bank: {v_path}")
    return 0


def cmd_capture_set(args) -> int:
    from sisteer.extraction.residual import write_capture_manifest

    _d, train, hold, manifest, sources = _capture(args)
    out = Path(args.out) if args.out else _resid_dir(args) / "capture_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_capture_manifest(out, manifest, sources=sources)
    sub = manifest.get("capture_subsample")
    if sub:
        print("[psr-residual] capture subsample: budget "
              f"{sub['budget_tokens']} official tokens per source, family shares "
              f"{ {k: round(v, 4) for k, v in sub['family_shares'].items()} } -> "
              + ", ".join(f"{k}: {r['n_selected']}/{r['n_pool']} docs, "
                          f"{r['official_tokens_selected']} tokens "
                          + str({f: fr['n_selected'] for f, fr in r['per_family'].items()})
                          for k, r in sub["per_source"].items()))
    print(f"[psr-residual] capture set: {len(train)} train / {len(hold)} "
          f"holdout docs; thin strata {manifest['thin_strata']}; no-holdout "
          f"strata {manifest['no_holdout_strata']} -> {out}")
    return 0


def _wandb(args, name: str):
    """The ONE W&B entry point of this script, called FIRST in every
    command that takes ``--wandb`` (before any model load). ``--wandb``
    means live curves in the operator's own entity or nothing runs: the
    shared validator (``sisteer.wandb_entity`` — the same one the Modal
    wrappers call at container start) refuses when WANDB_ENTITY is unset or
    empty or the key's default entity differs, with bounded retries on API
    errors, and ``wandb.init`` is then called with THAT entity explicitly.
    It never degrades to "telemetry off": the incident behind it was a key
    whose default entity was another account, and the old
    ``try/except → None`` here would have logged there, or nowhere,
    silently."""
    if not args.wandb:
        return None
    from sisteer.wandb_entity import wandb_init_checked, wandb_project

    # WANDB_PROJECT first (the Modal W&B secret carries it), else the site's distill project (wandb_project("distill"))
    return wandb_init_checked(
        project=os.environ.get("WANDB_PROJECT") or wandb_project("distill"),
        name=name, config={"mode": "psr_residual"})


def cmd_calibrate(args) -> int:
    from sisteer.models import num_layers

    mults = lr_mults_from_args(args)
    # one W&B run per calibrate invocation, named by its multiplier(s): the
    # derived app runs one multiplier per container, so three runs per model
    wb = _wandb(args, f"resid-calib-{args.model_key}-{args.mix_si}-lrmult"
                      + ",".join(format_mult(m) for m in mults))  # refuses first
    hp_by_mult = {m: hp_from_args(args, m) for m in mults}
    model, tok = _load_model(args)
    mix, ctl, pos_prov, neg_prov, _sh = hydrate_parents(
        model, tok, args.root, args.model_key, args.mix_si, args.ctl_si,
        args.pools)
    V, v_path = _load_or_compose_v(args, model, tok)
    _d, train, hold, manifest, sources = _capture(args)
    layers = list(range(num_layers(model)))
    identity_by_mult = {
        m: calibration_identity(args, hp_by_mult[m], model, pos_prov=pos_prov,
                                neg_prov=neg_prov, manifest=manifest,
                                sources=sources, layers=layers)
        for m in mults}
    out = Path(args.out) if args.out else _resid_dir(args) / "calibration_controls.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = run_calibration(model, tok, mix, ctl, train, hold, hp_by_mult,
                          layers=layers, out_path=out, n_boot=args.n_boot,
                          seed=args.seed, identity_by_mult=identity_by_mult,
                          wandb_run=wb, V=V, v_path=v_path,
                          checkpoint_every=args.checkpoint_every,
                          resume=not args.no_resume)
    print(f"[psr-residual] calibration controls -> {out} (lr-mult sweep "
          f"{mults}, {_selection_sentence(doc['sweep'])})")
    return 0


def cmd_distill(args) -> int:
    from sisteer.methods import load_object
    from sisteer.models import num_layers

    wb = _wandb(args, f"resid-prod-{args.model_key}-{args.mix_si}")  # refuses first
    explicit = (parse_lr_mults(args.lr_mult) if args.lr_mult is not None
                else None)
    if explicit is not None and len(explicit) != 1:
        raise ResidualError(
            f"distill takes ONE --lr-mult (got {args.lr_mult!r}); the sweep "
            "belongs to calibrate — omit it to distill at the calibration's "
            "selected multiplier")
    out_dir = _resid_dir(args)
    # A published, valid distilled object under this id is KEPT: the re-run verifies / replays its registry
    # record and never re-distils — unless --no-resume asks for a fresh production explicitly. The `finish`
    # stage re-submitted over a published object (04_fit_objects.sh, "no emission evidence") lands here before
    # any model load.
    kept_obj, kept_meta = out_dir / "object.pt", out_dir / "derived_meta.json"
    if kept_obj.is_file() and kept_meta.is_file() and not getattr(args, "no_resume", False):
        from sisteer.eval.object_identity import derived_object_problems
        problems = derived_object_problems(kept_obj, derived_id=resid_derived_id(args.mix_si),
                                           parents_root=None)
        if not problems:
            meta = json.loads(kept_meta.read_text(encoding="utf-8"))
            print(f"[psr-residual] {out_dir} already holds a valid distilled object "
                  f"(object_sha256 {str(meta.get('object_sha256'))[:16]}…) — kept; verifying its registry record",
                  flush=True)
            ev = ensure_registry_emission_derived(
                out_dir, object_kind="relu_steer_distilled_difference",
                derivation="residual_distillation",
                script="scripts/sdfbase/psr_residual_distill.py distill",
                extra=distilled_extra_from_meta(meta))
            print(f"[psr-residual] {json.dumps({'object': str(kept_obj), 'kept': True, 'registry_emission': ev.get('status')})}")
            return _emission().EMIT_FAILED_RC if ev.get("status") == "failed" else 0
        print(f"[psr-residual] {out_dir} holds an object that FAILS the derived verdict — re-distilling:\n  - "
              + "\n  - ".join(problems), file=sys.stderr)
    calib_path = Path(args.calibration) if args.calibration else \
        out_dir / "calibration_controls.json"
    if not calib_path.is_file():
        raise ResidualError(
            f"{calib_path} missing — run `calibrate` first (the recovery gate "
            "needs the attainable EDs and the floor at THIS trainer/lr)")
    calib_doc = json.loads(calib_path.read_text(encoding="utf-8"))
    calibration, lr_mult = calibration_for_mult(
        calib_doc, explicit[0] if explicit else None)
    hp = hp_from_args(args, lr_mult)
    model, tok = _load_model(args)
    mix, ctl, pos_prov, neg_prov, shared = hydrate_parents(
        model, tok, args.root, args.model_key, args.mix_si, args.ctl_si,
        args.pools)
    V, v_path = _load_or_compose_v(args, model, tok)
    _d, train, hold, manifest, sources = _capture(args)
    layers = list(range(num_layers(model)))
    identity = calibration_identity(args, hp, model, pos_prov=pos_prov,
                                    neg_prov=neg_prov, manifest=manifest,
                                    sources=sources, layers=layers)
    # refuse BEFORE any training: the full identity, not just lr
    require_same_identity(calibration, identity)
    ck = checkpoint_spec(args.checkpoint_every, out_dir, "production",
                         f"psr_residual_prod|{identity['fingerprint']}",
                         fresh=args.no_resume)
    res = run_production(model, tok, V, mix, ctl, train, hold, hp, calibration,
                         layers=layers, out_dir=out_dir,
                         capture_manifest=manifest, sources=sources,
                         pos_prov=pos_prov, neg_prov=neg_prov, shared=shared,
                         mix_si=args.mix_si, v_path=v_path, n_boot=args.n_boot,
                         seed=args.seed, identity=identity, wandb_run=wb,
                         checkpoint=ck, sweep=calib_doc.get("sweep"))
    print(f"[psr-residual] {json.dumps(res)}")
    ev = _emission().read_evidence(out_dir)
    if ev is not None and ev.get("status") == "failed":
        # the object stands; the JOB fails visibly (registry_emission.EMIT_FAILED_RC); a re-run replays the record
        return _emission().EMIT_FAILED_RC
    return 0


def cmd_all(args) -> int:
    from sisteer.methods import load_object
    from sisteer.models import num_layers

    wb = _wandb(args, f"resid-{args.model_key}-{args.mix_si}")  # refuses first
    mults = lr_mults_from_args(args)
    hp_by_mult = {m: hp_from_args(args, m) for m in mults}
    v_path = compose_v(args.root, args.out_root, args.model_key, args.mix_si,
                       args.ctl_si, pools_dir=args.pools)
    print(f"[psr-residual] V -> {v_path}")
    _d, train, hold, manifest, sources = _capture(args)
    out_dir = _resid_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    from sisteer.extraction.residual import write_capture_manifest
    write_capture_manifest(out_dir / "capture_manifest.json", manifest,
                           sources=sources)
    model, tok = _load_model(args)
    mix, ctl, pos_prov, neg_prov, shared = hydrate_parents(
        model, tok, args.root, args.model_key, args.mix_si, args.ctl_si,
        args.pools)
    V, v_path = _load_or_compose_v(args, model, tok)   # the persisted V, reloaded
    layers = list(range(num_layers(model)))
    identity_by_mult = {
        m: calibration_identity(args, hp_by_mult[m], model, pos_prov=pos_prov,
                                neg_prov=neg_prov, manifest=manifest,
                                sources=sources, layers=layers)
        for m in mults}
    doc = run_calibration(
        model, tok, mix, ctl, train, hold, hp_by_mult, layers=layers,
        out_path=out_dir / "calibration_controls.json", n_boot=args.n_boot,
        seed=args.seed, identity_by_mult=identity_by_mult, wandb_run=wb,
        V=V, v_path=v_path, checkpoint_every=args.checkpoint_every,
        resume=not args.no_resume)
    if doc["sweep"]["selected_lr_mult"] is None:
        print(f"[psr-residual] lr-mult sweep {_selection_sentence(doc['sweep'])}",
              flush=True)
        if args.no_production:
            return 0
        print("[psr-residual] no production distillation: the sweep refused "
              "to select a multiplier (rc 4)", flush=True)
        return 4
    selected = float(doc["sweep"]["selected_lr_mult"])
    print(f"[psr-residual] attainable ED (selected lr_mult "
          f"{format_mult(selected)}): {doc['attainable_ed_content']}")
    if args.no_production:
        return 0
    calibration, _m = calibration_for_mult(doc, selected)
    ck = checkpoint_spec(args.checkpoint_every, out_dir, "production",
                         f"psr_residual_prod|"
                         f"{identity_by_mult[selected]['fingerprint']}",
                         fresh=args.no_resume)
    res = run_production(model, tok, V, mix, ctl, train, hold,
                         hp_by_mult[selected], calibration,
                         layers=layers, out_dir=out_dir,
                         capture_manifest=manifest, sources=sources,
                         pos_prov=pos_prov, neg_prov=neg_prov, shared=shared,
                         mix_si=args.mix_si, v_path=v_path, n_boot=args.n_boot,
                         seed=args.seed, identity=identity_by_mult[selected],
                         wandb_run=wb, checkpoint=ck, sweep=doc["sweep"])
    print(f"[psr-residual] {json.dumps(res)}")
    return 0


def _load_or_compose_v(args, model, tok):
    """The persisted V (composed if absent), admitted through the shared
    derived-object verdict and reloaded through ``methods.load_object`` —
    the exact object the derived app distills."""
    from sisteer.eval.object_identity import assert_derived_object
    from sisteer.methods import load_object

    v_path = Path(args.out_root) / KIND_V / args.model_key / \
        v_derived_id(args.mix_si) / "object.pt"
    if not v_path.is_file():
        v_path = compose_v(args.root, args.out_root, args.model_key,
                           args.mix_si, args.ctl_si, pools_dir=args.pools)
    other = sidecar_pair_problems(v_path.with_name("derived_meta.json"),
                                  args.mix_si, args.ctl_si)
    if other:
        raise ResidualError(f"persisted V {v_path} belongs to another pair — "
                            "refusing to distill it:\n  - " + "\n  - ".join(other))
    assert_derived_object(v_path, consumer="psr-residual",
                          derived_id=v_derived_id(args.mix_si),
                          parents_root=args.root)
    return load_object(v_path, model=model, tok=tok), v_path


def cmd_merge_calibration(args) -> int:
    mults = parse_lr_mults(args.lr_mult)
    if args.parts:
        parts = [Path(p) for p in str(args.parts).split(",") if p.strip()]
    else:
        parts_dir = Path(args.parts_dir) if args.parts_dir else \
            _resid_dir(args) / CALIB_PARTS_DIRNAME
        parts = [parts_dir / f"lrmult{format_mult(m)}" /
                 "calibration_controls.json" for m in mults]
    out = Path(args.out) if args.out else \
        _resid_dir(args) / "calibration_controls.json"
    merge_calibration_parts(parts, mults, out)
    return 0


def cmd_identity_check(args) -> int:
    from sisteer.models import num_layers

    hp = hp_from_args(args, lr_mults_from_args(args)[0])  # lr is irrelevant here
    model, tok = _load_model(args)
    mix, ctl, pos_prov, neg_prov, _sh = hydrate_parents(
        model, tok, args.root, args.model_key, args.mix_si, args.ctl_si,
        args.pools)
    V, v_path = _load_or_compose_v(args, model, tok)
    _d, train, _hold, manifest, sources = _capture(args)
    layers = list(range(num_layers(model)))
    identity = calibration_identity(args, hp, model, pos_prov=pos_prov,
                                    neg_prov=neg_prov, manifest=manifest,
                                    sources=sources, layers=layers)
    res = run_identity_check(model, tok, mix, ctl, V, train, hp,
                             layers=layers, out_dir=_resid_dir(args),
                             seed=args.seed, identity=identity, v_path=v_path)
    print(f"[psr-residual] identity-summary {json.dumps(res, default=str)}")
    if not res["passed"]:
        raise ResidualError(
            "real-model teacher identity check FAILED (full per-layer report: "
            f"{res['report']}):\n  - " + "\n  - ".join(res["problems"]))
    print(f"[psr-residual] identity check PASSED on {len(layers)} layers -> "
          f"{res['report']}")
    return 0


def _common(p, *, model: bool):
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                   help="bank root holding <key>/cells_all/")
    p.add_argument("--out-root", type=Path, default=None,
                   help="defaults to <root>/derived")
    p.add_argument("--pools", type=Path, default=DEFAULT_POOLS)
    p.add_argument("--model-key", required=True, choices=list(MODEL_KEYS) + ["tinybase"])
    p.add_argument("--mix-si", default="constv2-mix-100k")
    p.add_argument("--ctl-si", default=CTL_SI)
    p.add_argument("--holdout-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--capture-budget-tokens", type=int, default=None,
                   help="capture set = a seeded per-source subsample of this "
                        "many official tokens from each parent pool (the "
                        "full - full pair: 100000, matching the 100k pools' "
                        "size); default: every doc of both pools")
    if model:
        p.add_argument("--model", required=True, help="HF model id")
        p.add_argument("--revision", default=None)
        p.add_argument("--device", default=None)
        p.add_argument("--dtype", default=None)
        p.add_argument("--config", default=None,
                       help="fits config yaml (the D22 PSR winner hp)")
        p.add_argument("--lr", type=float, default=None)
        p.add_argument("--lr-mult", default=None,
                       help="calibrate/all: the comma-separated lr-multiplier "
                            "sweep for the controls (default 0.3,1,3 — all run, "
                            "the best planted recovery is selected for "
                            "production); distill: ONE multiplier (default: the "
                            "calibration's selected one)")
        p.add_argument("--max-grad-norm", type=float, default=None)
        p.add_argument("--warmup-frac", type=float, default=None)
        p.add_argument("--reg-coeff", type=float, default=None)
        p.add_argument("--accum-steps", type=int, default=None)
        p.add_argument("--batch-size", type=int, default=None)
        p.add_argument("--max-len", type=int, default=None)
        p.add_argument("--distill-epochs-cap", "--epochs", dest="epochs",
                       type=int, default=EPOCHS_CAP,
                       help=f"epoch cap of every distillation (controls and "
                            f"production; default {EPOCHS_CAP}, R2-D19)")
        p.add_argument("--distill-patience", "--patience", dest="patience",
                       type=int, default=PATIENCE,
                       help=f"early-stop patience in epochs (default "
                            f"{PATIENCE}, R2-D19)")
        p.add_argument("--n-boot", type=int, default=1000)
        p.add_argument("--wandb", action="store_true")
        p.add_argument("--checkpoint-every", type=int, default=0,
                       help="resumable trainer checkpoints every N optimizer "
                            "steps for every control and the production fit "
                            "(under <out dir>/ckpt/, keyed by the identity "
                            "fingerprint); 0 = off")
        p.add_argument("--no-resume", action="store_true",
                       help="a FRESH run: calibrate/all re-run every multiplier "
                            "even when calibration_controls.json already holds "
                            "its finished payload under this identity, and every "
                            "fit (controls, production) discards an existing "
                            "checkpoint dir instead of resuming it")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("compose-v", help="V = +mix − ctl (CPU)")
    _common(p, model=False)
    p.add_argument("--no-emit", action="store_true",
                   help="defer V's registry record to `emit-v` (the direct-object build files it only after the liveness "
                        "probe binds gate_liveness_receipt.json to V's bytes)")
    p.set_defaults(fn=cmd_compose_v)
    p = sub.add_parser("emit-v", help="file V's registry record once gate_liveness.json + its receipt are bound to V's bytes "
                                      "(refuses otherwise; idempotent replay)")
    _common(p, model=False)
    p.set_defaults(fn=cmd_emit_v)
    p = sub.add_parser("liveness-receipt", help="write gate_liveness_receipt.json beside V after a successful probe")
    _common(p, model=False)
    p.add_argument("--writer", default="scripts/sdfbase/psr_residual_distill.py liveness-receipt")
    p.add_argument("--slurm-job-id", default=None); p.add_argument("--node", default=None); p.add_argument("--code-sha", default=None)
    p.add_argument("--n-probe", type=int, default=None); p.add_argument("--pairs", default=None)
    p.set_defaults(fn=cmd_liveness_receipt)
    p = sub.add_parser("verify-v", help="the persisted V re-admitted against "
                                        "the live bank (CPU; exit 2 on any problem)")
    _common(p, model=False)
    p.set_defaults(fn=cmd_verify_v)
    p = sub.add_parser("capture-set", help="stratified capture set manifest")
    _common(p, model=False)
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_capture_set)
    p = sub.add_parser("calibrate", help="floor + planted controls (GPU)")
    _common(p, model=True)
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_calibrate)
    p = sub.add_parser("identity-check",
                       help="identity-only preflight: write identity + hook "
                            "contracts + the persisted V's contract (GPU)")
    _common(p, model=True)
    p.set_defaults(fn=cmd_identity_check)
    p = sub.add_parser("distill", help="production distillation (GPU)")
    _common(p, model=True)
    p.add_argument("--calibration", default=None,
                   help="calibration_controls.json (default: in the out dir)")
    p.set_defaults(fn=cmd_distill)
    p = sub.add_parser("merge-calibration",
                       help="merge per-multiplier calibrate outputs into ONE "
                            "complete sweep file (CPU, model-free)")
    _common(p, model=False)
    p.add_argument("--lr-mult", required=True,
                   help="the FULL sweep the merged file must cover, e.g. 0.3,1,3")
    p.add_argument("--parts", default=None,
                   help="comma-separated part files (default: <out dir>/"
                        f"{CALIB_PARTS_DIRNAME}/lrmult<m>/calibration_controls.json "
                        "per multiplier)")
    p.add_argument("--parts-dir", default=None,
                   help=f"the parts root (default <out dir>/{CALIB_PARTS_DIRNAME})")
    p.add_argument("--out", default=None,
                   help="the merged file (default <out dir>/calibration_controls.json)")
    p.set_defaults(fn=cmd_merge_calibration)
    p = sub.add_parser("all", help="compose-v + capture-set + calibrate + distill")
    _common(p, model=True)
    p.add_argument("--no-production", action="store_true",
                   help="calibration only")
    p.set_defaults(fn=cmd_all)
    return ap


def main(argv=None) -> int:
    from sisteer.extraction import tokcache

    args = build_parser().parse_args(argv)
    args.root, args.pools = site_path(args.root, "--root"), site_path(args.pools, "--pools")
    if args.out_root is None:
        args.out_root = Path(args.root) / "derived"
    # One in-process token memo for the run: the capture set is encoded many
    # times over (reference batches, both controls, the production fit, the
    # liveness probes), and a cache hit is a copy of what the tokenizer
    # returned (byte-neutral — the fit scripts' convention).
    prev = tokcache.active()
    tokcache.set_active(tokcache.TokCache())
    from sisteer.wandb_entity import WandbEntityError

    try:
        return args.fn(args)
    except (ResidualError, WandbEntityError) as e:
        print(f"[psr-residual] REFUSED: {e}", file=sys.stderr)
        return 2
    finally:
        tokcache.set_active(prev)


if __name__ == "__main__":
    sys.exit(main())
