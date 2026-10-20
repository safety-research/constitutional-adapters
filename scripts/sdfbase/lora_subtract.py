#!/usr/bin/env python
"""Build LoRA task-vector SUBTRACTION objects (sdf-base-steering round-2):
ΔW(mix cell) − ΔW(control cell), per model, from the fitted ``lora_r4_ce``
artifacts — the values-specific remainder (the corpus-minus-control
decomposition) as a SERVABLE object. The control cell is ``--ctl-si``
(default constv2ctl-100k, the round-1 pair); the full − full pair is
``--cells constv2-mix-full --ctl-si constv2ctl2-full``, same derived id
template — the control rides in the provenance (``parent_neg.cell_id``),
never in the id.

The PRIMARY pair is SCALE-MATCHED: constv2-mix-100k − constv2ctl-100k —
both parents trained at the same 100k-official-token budget, so the
difference isolates values from register at matched data scale. The
mix-1m − ctl-100k object is ALSO built but labeled
``budget_mismatched (descriptive)`` — its parents' training budgets
differ 10×, so it bounds nothing (the scale limitation, now stamped into
the artifact itself). Both parents' data scales (official token budgets,
read from the staged pool manifests) ride in the provenance.

Pure CPU tensor bookkeeping: parent blobs are read straight off the local
harvest bank (never a model load), admitted through the ONE parent-admission
gate the derived-object builders share (scripts/sdfbase/parent_admission.py:
COMPLETE marker present — a torn publish is never a parent — the budget
sidecar names exactly the parent cell, the SHARED staged-cell gate
``train_budget.staged_cell_problems`` passes, the model revision resolves
against the bank's store identity with its source recorded, and the two
parents agree on model id/revision, target layers, fit-position leg and
template treatment: a difference across any of those is not a task vector),
and composed structurally — both parents' (A, B) factors ride into the
artifact bit-exact, so dropping the control term recovers the corpus fit's
deltas bitwise (see extraction/multilayer.LoRASubtractionObject, kind
``lora_subtraction``, registered in methods._KIND_LOADERS: the artifacts
hydrate through the ordinary ``load_object`` path, e.g.
``probe_score.py score --object <derived>/object.pt [--scale 2.0]``).

Outputs land in a ``derived/`` namespace, NEVER the fits banks (these are
derived, untrained objects — placing them beside COMPLETE fit cells would
let a bank walker mistake one for a fitted cell):

  <root>/derived/lora_subtraction/<model_key>/
      <mix_si>__lora_r4_ce_minus_ctl/{object.pt, derived_meta.json}

Default root is the local harvest mirror ($STAGE/sdfbase/out);
default cells are the scale-matched primary + the labeled 1M pair; the
default MODEL sweep is the round-2 pair (gemma4e4b_base, qwen35_4b_base —
``DEFAULT_MODEL_KEYS``). ``MODEL_KEYS`` is the SUPPORTED-key registry (the
two Llama-3.1-8B keys included) and is deliberately not the default: an
invocation that omits ``--models`` must keep building exactly what it
always built, and the Llama keys' parents live in neither Llama config's
default cells (a registry doubling as a default selection is the "guard
doing double duty" shape). Name the Llama keys explicitly. The
provenance block (parent cell ids + object.pt sha256s + budget-sidecar
sha256s + parent fit fingerprints + model id/revision + layer set + fit
leg + data scales) rides both inside the blob and in derived_meta.json.
The sidecar is named derived_meta.json, NOT train_budget_meta.json, so
the staged-cell identity gate never mistakes a derived object for a fit —
and every derived-object consumer validates it through the shared
``train_budget.derived_object_problems`` verdict instead (object bytes ↔
sidecar sha, sidecar ↔ embedded provenance, parent identities complete
and mutually consistent; with a bank root, parents re-verified on disk).
This builder runs that same verdict on every artifact it writes.

Writes are ATOMIC (temp file + ``os.replace``): a re-run over an existing
publication — even one the volume holds read-only — lands whole or not at
all, and ``--keep-valid`` makes a re-run idempotent: an existing object
that passes the shared verdict against THIS bank AND names exactly the
requested parents is kept (its sha unchanged); a torn or tampered one of
the SAME pair is rebuilt. An existing object whose sidecar names OTHER
parents is another pair's publication under this id and is REFUSED — never
rebuilt in place, with or without ``--keep-valid``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parents[1]
for _p in (REPO / "src", _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
# The sdfbase siblings' streaming file sha (build_pools imports it the same
# way) — the sisteer library copy (eval.defense_cell) costs a torch import.
from build_heldout_split import sha256_file  # noqa: E402
# The ONE parent-admission gate both derived-object builders share
# (parent_admission.py: COMPLETE + sidecar + shared staged-cell gate +
# store-identity revision resolution + kind), re-exported under the names
# this module always had.
from parent_admission import (STORE_IDENT, admit_parent,  # noqa: E402,F401
                              pool_budget, resolve_model_revision)
from parent_admission import require_parents_match as _parents_match  # noqa: E402

CTL_SI = "constv2ctl-100k"
ARM = "lora_r4_ce"
#: the fit outputs and the pools under the staging directory ($STAGE), expanded through sisteer.site where they are used
DEFAULT_ROOT = Path("$STAGE/sdfbase/out")
DEFAULT_POOLS = Path("$STAGE/sdfbase/pools")


def site_path(value, key: str) -> Path:
    """``value`` with its site placeholder (``$STAGE``, the staging directory) expanded through sisteer.site; an unset one
    refuses by name, and a value without one comes back as a path unchanged."""
    from sisteer import site as _site
    return _site.expand_path(value, key=key)
#: The SUPPORTED model keys (a registry, not a selection): the round-2 Modal
#: banks (gemma / qwen), the Slurm fit lane's newmodels_full bank keys (the GLM
#: lane, 2026-09-08: <bank>/<key>/cells_all/<cell> is the same layout; `_doc`
#: because these targets have no base variant), and the Llama-3.1-8B lane
#: (2026-09-09: base + instruct-direct).
MODEL_KEYS = ("gemma4e4b_base", "qwen35_4b_base",
              "glm46vflash_doc", "nemotron3nano4b_doc",
              "llama8b_base", "llama8b_it_doc",
              # the instruct-direct batch-64 lanes: their pair is
              # constv2-mix-full - constv2ctl2-full, named explicitly,
              # never swept by default
              "gemma4e4b_it_doc_eb64", "qwen35_4b_it_doc_eb64",
              # the paper-companion-repo worked example: the FIT
              # checkpoint's registry id (reproduce/models.yaml
              # fit.checkpoint.model_id — every fit-side key carries _base;
              # the served id qwen35_4b_demo keys models.yaml), its pair named
              # explicitly by reproduce/04_fit_objects.sh, never swept by
              # default
              "qwen35_4b_demo_base",
              # the qwen-base batch-64 SEED REPLICATES (2026-09-15): the
              # pair is constv2-mix-full - constv2ctl2-full per seed, from
              # SAME-seed parents, named explicitly, never swept by default
              "qwen35_4b_base_eb64_s1", "qwen35_4b_base_eb64_s2",
              "qwen35_4b_base_eb64_s3")
#: The DEFAULT ``--models`` sweep: the round-2 pair, exactly what the tool
#: swept before the Llama keys were registered. The Llama keys are swept
#: only when named (their default-cell parents do not exist).
DEFAULT_MODEL_KEYS = ("gemma4e4b_base", "qwen35_4b_base")
#: Default pairs: the scale-matched PRIMARY plus the
#: budget-mismatched 1M version, explicitly labeled.
DEFAULT_CELLS = ("constv2-mix-100k", "constv2-mix-1m")


class SubtractError(RuntimeError):
    """Loud failure — never emit a derived object off mismatched parents."""


def load_parent(root: Path, key: str, si: str) -> tuple[dict, dict]:
    """(blob, provenance) for one fitted ``lora_all`` parent cell through
    the shared admission gate (``parent_admission.admit_parent``: COMPLETE
    marker present, sidecar names the cell, staged-cell gate, identity
    fields recorded, model revision resolved against the bank's store
    identity with its source recorded, kind ``lora_all``); every refusal
    is a :class:`SubtractError`."""
    _obj, blob, _meta, prov = admit_parent(root, key, f"{si}__{ARM}",
                                           kind="lora_all", error=SubtractError)
    # the parent's path rides its provenance into derived_meta.json so the emitter can read the parent's OWN emission
    # evidence (registry_emission.campaign_for_derived: a derived object carries its parents' campaign)
    prov.setdefault("object_path", str(_obj))
    return blob, prov


def require_parents_match(pos_prov: dict, neg_prov: dict) -> None:
    """The shared cross-parent match, named for this builder's object."""
    _parents_match(pos_prov, neg_prov, what="a task-vector difference",
                   error=SubtractError)


#: The general ``lora_all`` shape's architecture keys (D contract §2): a
#: legacy o_proj-only blob carries none of them and reads as the defaults.
_ARCH_DEFAULTS = {"lora_modules": ("o_proj",), "lora_set_id": "wo",
                  "init_scheme": "zero"}


def architecture_of(blob: dict) -> dict:
    """The ARCHITECTURE a ``lora_all`` blob states about itself (D contract
    §2): ``lora_modules`` (sorted tuple), ``lora_set_id``, ``per_expert_rank``
    (the {shared_expert, routed_expert} map or None), ``init_scheme``,
    ``rank`` and ``layers`` — legacy o_proj-only blobs read as
    ``("o_proj",)`` / ``wo`` / None / ``zero``. Read off the blob, never the
    config: what was fitted is what the artifact says."""
    out = {}
    mods = blob.get("lora_modules")
    out["lora_modules"] = (tuple(sorted(str(m) for m in mods)) if mods
                           else _ARCH_DEFAULTS["lora_modules"])
    out["lora_set_id"] = blob.get("lora_set_id") or _ARCH_DEFAULTS["lora_set_id"]
    per = blob.get("per_expert_rank")
    out["per_expert_rank"] = ({str(k): int(v) for k, v in per.items()}
                              if isinstance(per, dict) else None)
    out["init_scheme"] = blob.get("init_scheme", _ARCH_DEFAULTS["init_scheme"])
    out["rank"] = int(blob["rank"]) if blob.get("rank") is not None else None
    out["layers"] = ([int(l) for l in blob["layers"]]
                     if isinstance(blob.get("layers"), list) else None)
    out["targets"] = (dict(blob["targets"])
                      if isinstance(blob.get("targets"), dict) else None)
    return out


def is_general_shape(blob: dict) -> bool:
    """True for the general module-set shape (any key of D contract §2's
    general block present), False for the legacy o_proj-only blob."""
    return any(k in blob for k in ("lora_modules", "lora_set_id",
                                   "per_expert_rank", "targets"))


def require_architecture_match(pos_blob: dict, neg_blob: dict) -> dict:
    """Refuse a pair whose parents were fitted under different LoRA
    ARCHITECTURES — module set, per-expert rank, init scheme, per-module
    targets — each NAMED, before any tensor is touched (D contract §2: the
    difference is only defined on the same factor keys). ``compose_blob``
    re-checks the same axes with its own ``lora_subtraction:`` wording once
    the tensors are in hand; this is the by-name refusal the builder and its
    test speak. Returns the (shared) architecture."""
    a, b = architecture_of(pos_blob), architecture_of(neg_blob)
    for f in ("lora_modules", "lora_set_id", "per_expert_rank", "init_scheme"):
        if a[f] != b[f]:
            raise SubtractError(
                f"parents disagree on {f}: pos has {a[f]!r}, neg has {b[f]!r} "
                "— a task-vector difference across LoRA architectures is not "
                "a defined object (refit the parents on one architecture)")
    if a["targets"] != b["targets"]:
        ka = sorted(a["targets"] or {})
        kb = sorted(b["targets"] or {})
        raise SubtractError(
            f"parents disagree on targets: pos adapts {ka}, neg adapts {kb} "
            "(or their sites / per-module ranks differ) — same-architecture "
            "parents only")
    return a


def trainable_params(blob: dict) -> int:
    """Σ numel over every factor tensor a blob carries — a parent's ``A``/``B``
    or a difference's ``A_pos``/``B_pos``/``A_neg``/``B_neg`` (= 2 × parent)."""
    n = 0
    for key in ("A", "B", "A_pos", "B_pos", "A_neg", "B_neg"):
        d = blob.get(key)
        if isinstance(d, dict):
            n += sum(int(t.numel()) for t in d.values() if hasattr(t, "numel"))
    return n


def registry_model_id(name):
    """The registry ``model_id`` for a fit checkpoint (an HF repo) through
    decode_policy.json's ``model_spellings`` — the seedrep emitter's rule;
    None when no export is present or nothing maps it (the hook then keeps
    the sidecar's spelling; nothing is guessed). Shared with
    psr_residual_distill.py (the sibling-import convention)."""
    if not name:
        return None
    try:
        from sisteer.registry.decode_policy import PolicyFile
        pf = PolicyFile.load()
    except Exception:  # noqa: BLE001 — no export on this host
        return None
    mid = pf.model_id(str(name))
    return mid if (mid != str(name) or mid in pf.models) else None


SCRIPT_NAME = "scripts/sdfbase/lora_subtract.py"


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


class RegistryEmissionFailed(SubtractError):
    """The difference is published and valid but its registry ``object`` record could not be written
    (REGISTRY_EMIT_FAILED beside it): the run exits registry_emission.EMIT_FAILED_RC; a --keep-valid re-run
    replays the record without rebuilding."""


def _hook_call(obj_dir: Path, meta: dict, emitter=None, fit_provenance: dict | None = None):
    """The call into ``hooks.emit_object_from_sidecar`` for a difference (raises on its own failures; returns
    the hook's paths / [] when off / None when the guarded hook swallowed a failure). ``emitter``: a callable
    with the hook's signature writing ONE of its two records
    (``registry_emission.component_emitter(<component>)``) for the partial replay; default the hook."""
    from sisteer.registry import hooks as _rh
    fn = emitter if emitter is not None else _rh.emit_object_from_sidecar
    rank = meta.get("rank")
    # a DERIVED object's record (vocab.FIT_TARGET_RULES["derived"]; the DB CHECK object_fit_target_kind_pairing): objective
    # `not_applicable` (the parents carry the ce objective), the derivation record from the sidecar as the adapter-filed
    # in-paper rows carry it, the writer stated — before 2026-10-02 this passed fit_objective="ce" and no params, and every
    # record it wrote died at the registry's CHECK after the cluster evidence said "emitted"
    stated = {
        "fit_model_id": registry_model_id(meta.get("model_id")),
        "fit_rank": int(rank) if rank is not None else None,
        "fit_alpha": float(rank) if rank is not None else None,
        "fit_target_modules_json": list(meta["lora_modules"]) if meta.get("lora_modules") else None,
        "fit_layers_json": meta.get("layers"),
        # the sidecar's `layers` are the PARENTS' target_layers (the trainers index decoder blocks from 0): stated beside the explicit list,
        # which the hook never guesses a convention for (demo-import M1b) — before this the record carried the list and NO convention
        "fit_layer_index_convention": "zero_based_decoder_block" if isinstance(meta.get("layers"), list) else None,
        "derivation_params_json": _rh.derivation_params_from_derived_meta(meta),
        "fit_writer": SCRIPT_NAME,
        # a replay restates the builder's provenance (the prior record's), never this job's context
        "fit_provenance": dict(fit_provenance) if fit_provenance else None,
    }
    extra = {
        "parent_object_ids": meta.get("parent_object_ids"),
        "trainable_params": meta.get("trainable_params"),
        "export_rank": 2 * int(rank) if rank is not None else None,
        "lora_set_id": meta.get("lora_set_id"),
        "per_expert_rank": meta.get("per_expert_rank"),
        "init_scheme": meta.get("init_scheme"),
        "scale_matched": meta.get("scale_matched"),
        "derived_id": meta.get("derived_id"),
    }
    kwargs = {k: v for k, v in stated.items() if v is not None}
    extra = {k: v for k, v in extra.items() if v is not None}
    # the campaign: the parents' recorded one (both must agree), an explicit REGISTRY_CAMPAIGN only when it agrees, never a
    # module default (registry_emission.campaign_for_derived; CampaignUnset is recorded as `failed` evidence by the wrappers)
    return fn(
        str(obj_dir), campaign=_emission().campaign_for_derived_or_off(meta),
        object_kind="lora_subtraction",
        paper_label=_rh.paper_label_for("lora_subtraction", "difference"),
        object_role="difference", fit_target_kind="derived",
        fit_objective="not_applicable", derivation="lora_subtraction",
        script=SCRIPT_NAME, extra=extra or None, **kwargs)


def registry_emit(obj_dir: Path, meta: dict) -> dict:
    """The results-registry ``object`` + ``artifact`` records for a freshly
    built difference: kind ``lora_subtraction``, role ``difference``, target
    kind ``derived``, objective ``ce`` (the parents'), ``derivation``
    ``lora_subtraction``; the parents' rank / module set / layers as the fit
    columns (``hooks.object_from_sidecar``'s own keyword arguments; the hook
    also fills ``parent_object_ids_json`` from derived_meta.json), the
    difference's own facts (``parent_object_ids``, ``trainable_params`` = 2 ×
    parent, ``export_rank`` = 2r, ``lora_set_id``, ``per_expert_rank``,
    ``init_scheme``) in ``extra``. Never raises and never silent: returns
    the EVIDENCE written beside the artefact
    (``registry_emission.json``: emitted / not_emitted / failed + the
    REGISTRY_EMIT_FAILED marker), prints ``registry: object <sha256>`` when a
    record was filed."""
    obj_dir = Path(obj_dir)
    mod = _emission()
    try:
        campaign = mod.campaign_for_derived_or_off(meta)
    except mod.CampaignUnset as e:
        return mod.failed(obj_dir, meta.get("object_sha256"), SCRIPT_NAME, "difference object", f"CampaignUnset: {e}")
    ev = mod.emit_with_evidence(
        obj_dir, object_id=meta.get("object_sha256"), writer=SCRIPT_NAME, campaign=campaign,
        emit=lambda: _hook_call(obj_dir, meta), what="difference")
    if ev.get("status") == "emitted":
        print(f"registry: object {meta['object_sha256']}", flush=True)
    return ev


def ensure_registry_emission(obj_dir: Path) -> dict:
    """A KEPT (--keep-valid) difference is not a recorded one: verify its emission evidence (keyed by the
    sidecar's object id; else the object AND artifact records already in the spool) and replay what is missing —
    the whole hook, or only the missing record — never by rebuilding
    (registry_emission.ensure_emitted). Prints ``registry: object <sha256>`` when the row
    is (or was) emitted, so 04_fit_objects.sh reads the same line as for a fresh build."""
    obj_dir = Path(obj_dir)
    meta = json.loads((obj_dir / "derived_meta.json").read_text(encoding="utf-8"))
    oid = meta.get("object_sha256")
    mod = _emission()
    try:
        campaign = mod.campaign_for_derived_or_off(meta)
    except mod.CampaignUnset as e:
        return mod.failed(obj_dir, oid, SCRIPT_NAME, "difference object", f"CampaignUnset: {e}")
    prev = mod.read_evidence(obj_dir) or {}
    prior = mod.fit_provenance_from_prior_record(oid, prev.get("campaign") or campaign) or mod.fit_provenance_from_prior_record(oid, campaign)
    ev = mod.ensure_emitted(
        obj_dir, object_id=oid, writer=SCRIPT_NAME, campaign=campaign,
        emit=lambda: _hook_call(obj_dir, meta, fit_provenance=prior), what="difference",
        # replay only the record the spool lacks, never a second object row
        emit_component=lambda c: _hook_call(obj_dir, meta, emitter=mod.component_emitter(c), fit_provenance=prior))
    if ev.get("status") == "emitted":
        print(f"registry: object {oid}", flush=True)
    return ev


def derived_id_for(mix_si: str) -> str:
    """The derived id template (analysis DERIVED_KINDS ``LoRA_sub``): the
    corpus cell names it; the control is provenance."""
    return f"{mix_si}__{ARM}_minus_ctl"


def other_pair_problems(meta_path, mix_si: str, ctl_si: str) -> list[str]:
    """Problems (empty = same pair, or no sidecar) when an existing sidecar
    names parents other than ``<mix_si>__lora_r4_ce`` / ``<ctl_si>__lora_r4_ce``."""
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


def existing_valid(root: Path, out_root: Path, key: str, mix_si: str,
                   ctl_si: str) -> tuple[Path, list[str]]:
    """(object path, problems) for an existing publication of this pair:
    empty problems = the object passes the shared derived-object verdict
    against ``root`` (parents re-verified on disk) AND its sidecar names
    exactly the requested parents — the derived id carries the corpus cell
    only, so a same-id object built against ANOTHER control must never be
    kept for this pair."""
    from sisteer.eval.train_budget import derived_object_problems

    derived_id = derived_id_for(mix_si)
    out_dir = out_root / key / derived_id
    obj_path = out_dir / "object.pt"
    meta_path = out_dir / "derived_meta.json"
    if not obj_path.is_file():
        return obj_path, [f"{obj_path} missing"]
    if not meta_path.is_file():
        return obj_path, [f"{meta_path} missing (torn publish)"]
    try:
        problems = list(derived_object_problems(obj_path, derived_id=derived_id,
                                                parents_root=root))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001 — an unreadable artifact is a problem, not a crash
        return obj_path, [f"{obj_path}: {type(e).__name__}: {e}"]
    want = {"parent_pos": f"{mix_si}__{ARM}", "parent_neg": f"{ctl_si}__{ARM}"}
    for role, cell in want.items():
        got = (meta.get(role) or {}).get("cell_id")
        if got != cell:
            problems.append(f"{meta_path}: {role}.cell_id={got!r}, this pair "
                            f"needs {cell!r}")
    return obj_path, problems


def build_one(root: Path, out_root: Path, key: str, mix_si: str,
              pools_dir: Path = DEFAULT_POOLS, *, ctl_si: str = CTL_SI,
              keep_valid: bool = False) -> Path:
    import torch

    from sisteer.eval.train_budget import derived_object_problems
    from sisteer.extraction.multilayer import LoRASubtractionObject

    pools_dir = site_path(pools_dir, "pools_dir")

    # an existing publication under this id that names OTHER parents is
    # another pair's object: REFUSED, never rebuilt in place (the derived id
    # carries the corpus cell only)
    other = other_pair_problems(out_root / key / derived_id_for(mix_si)
                                / "derived_meta.json", mix_si, ctl_si)
    if other:
        raise SubtractError(
            f"{out_root / key / derived_id_for(mix_si)} already holds an object "
            "of another pair — refusing to overwrite it:\n  - "
            + "\n  - ".join(other))
    if keep_valid:
        obj_path, problems = existing_valid(root, out_root, key, mix_si, ctl_si)
        if not problems:
            print(f"[lora-subtract] {key}: {mix_si} - {ctl_si} already "
                  f"published and valid against {root} — kept {obj_path}")
            # kept is not recorded: verify / replay the registry row, never rebuild
            ev = ensure_registry_emission(obj_path.parent)
            if ev.get("status") == "failed":
                raise RegistryEmissionFailed(
                    f"{obj_path.parent}: the kept difference's registry emission FAILED "
                    f"({ev.get('error')}); REGISTRY_EMIT_FAILED written beside it — the object stands, "
                    "fix the spool / push and re-run --keep-valid to replay the record")
            return obj_path
        if obj_path.is_file():
            print(f"[lora-subtract] {key}: existing {obj_path} FAILS the "
                  "derived-object verdict for this pair — rebuilding:\n  - "
                  + "\n  - ".join(problems), file=sys.stderr)
    pos_blob, pos_prov = load_parent(root, key, mix_si)
    neg_blob, neg_prov = load_parent(root, key, ctl_si)
    require_parents_match(pos_prov, neg_prov)
    # the LoRA architecture axes (module set, per-expert rank, init scheme,
    # targets) — by name, BEFORE compose_blob's layer / rank / shape checks
    arch = require_architecture_match(pos_blob, neg_blob)
    pos_scale = pool_budget(pools_dir, mix_si)
    neg_scale = pool_budget(pools_dir, ctl_si)
    pos_prov["data_scale"] = pos_scale
    neg_prov["data_scale"] = neg_scale
    pos_tok, neg_tok = pos_scale["official_tokens"], neg_scale["official_tokens"]
    matched = pos_tok is not None and pos_tok == neg_tok
    derived_id = derived_id_for(mix_si)
    provenance = {
        "kind": "lora_subtraction",
        "derived_id": derived_id,
        "derivation": "delta(parent_pos) - delta(parent_neg), structural "
                      "(both parents' factors stored bit-exact)",
        "model_key": key,
        # The parents' SHARED identity, lifted to the top so a consumer
        # reads the model this object serves against without diving into
        # the parent blocks (require_parents_match proved them equal).
        "model_id": pos_prov["model_id"],
        "model_revision": pos_prov["model_revision"],
        "positions": pos_prov["positions"],
        "response_header_len": pos_prov["response_header_len"],
        # The parents' SHARED LoRA architecture (D contract §2; equal by
        # require_architecture_match): the module set and its id, the MoE
        # per-expert rank map (None on a dense set), the init scheme — a
        # legacy o_proj-only pair reads as ["o_proj"] / wo / null / zero.
        "lora_modules": list(arch["lora_modules"]),
        "lora_set_id": arch["lora_set_id"],
        "per_expert_rank": arch["per_expert_rank"],
        "init_scheme": arch["init_scheme"],
        # the two parents' object ids (object_id == sha256(object.pt))
        "parent_object_ids": [pos_prov["object_sha256"],
                              neg_prov["object_sha256"]],
        "parent_pos": pos_prov,
        "parent_neg": neg_prov,
        # the round-2 labeling contract: scale-matched pairs are
        # the PRIMARY comparison; mismatched pairs are descriptive only
        # (the scale limitation, stamped into the artifact).
        "scale_matched": bool(matched),
        "comparison": (
            "scale_matched (primary: both parents trained at the same "
            "official-token budget)" if matched else
            "budget_mismatched (descriptive: parents trained at "
            f"{pos_tok} vs {neg_tok} official tokens — bounds nothing; "
            "the D23 scale limitation)"),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "builder": "scripts/sdfbase/lora_subtract.py",
    }
    try:
        blob = LoRASubtractionObject.compose_blob(pos_blob, neg_blob,
                                                  provenance=provenance)
    except ValueError as e:
        if is_general_shape(pos_blob) and "lora_subtraction:" not in str(e):
            # compose_blob keys the general shape on the `targets` block
            # (MultiLayerLoRAObject.blob_is_general) and reads anything else
            # as the legacy o_proj shape with integer layer keys — a parent
            # that names module-set keys but no `targets` fails inside its
            # int() of a "<layer>.<module>" key. Name the malformed parent
            # instead of surfacing that traceback.
            raise SubtractError(
                f"parent blob names the module set {arch['lora_set_id']!r} "
                f"({list(arch['lora_modules'])}) but carries no `targets` "
                "block — not a complete general-shape lora_all artifact "
                f"(compose_blob read it as the legacy shape): {e}") from e
        raise
    out_dir = out_root / key / derived_id
    out_dir.mkdir(parents=True, exist_ok=True)
    obj_path = out_dir / "object.pt"
    meta_path = out_dir / "derived_meta.json"
    # ATOMIC publish: the blob is written and round-tripped as a temp file,
    # then renamed over the target (a rename needs a writable DIRECTORY,
    # not a writable target — a read-only earlier publication is replaced,
    # never opened for writing: the EACCES re-run trap); the sidecar follows
    # the same way, so a reader never sees a new object beside an old meta.
    tmp_obj = out_dir / "object.pt.tmp"
    tmp_meta = out_dir / "derived_meta.json.tmp"
    torch.save(blob, tmp_obj)
    # Round-trip sanity: the artifact must reload under weights_only=True
    # (that reload is what refuses any non-tensor leakage into the blob)
    # and every retained factor must come back bit-exact.
    back = torch.load(tmp_obj, map_location="cpu", weights_only=True)
    for side in ("A_pos", "B_pos", "A_neg", "B_neg"):
        for l, t in blob[side].items():
            if not torch.equal(back[side][l], t):
                tmp_obj.unlink(missing_ok=True)
                raise SubtractError(
                    f"{obj_path}: {side}[{l}] did not round-trip bit-exact")
    meta = {**provenance, "object_sha256": sha256_file(tmp_obj),
            "layers": blob["layers"], "rank": blob["rank"],
            # the difference's own parameter count: every retained factor
            # (A_pos, B_pos, A_neg, B_neg) = 2 × one parent's
            "trainable_params": trainable_params(blob)}
    if blob.get("targets") is not None:
        meta["targets"] = blob["targets"]      # the general shape's per-module sites
    tmp_meta.write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp_obj, obj_path)
    os.replace(tmp_meta, meta_path)
    # The shared derived-object verdict on what was just written, parents
    # re-verified against THIS bank: the builder never publishes an
    # artifact a consumer would refuse.
    problems = derived_object_problems(obj_path, derived_id=derived_id,
                                       parents_root=root)
    if problems:
        raise SubtractError(
            f"{obj_path} failed the derived-object verdict right after "
            "being written:\n  - " + "\n  - ".join(problems))
    # the registry record — AFTER the verdict (the publication event); a kept
    # --keep-valid object is VERIFIED / replayed above, never re-filed twice
    ev = registry_emit(out_dir, meta)
    if ev.get("status") == "failed":
        raise RegistryEmissionFailed(
            f"{out_dir}: the difference is published and valid but its registry emission FAILED "
            f"({ev.get('error')}); REGISTRY_EMIT_FAILED written beside it — re-run with --keep-valid "
            "to replay the record without rebuilding")
    return obj_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                    help="harvest bank root holding <key>/cells_all/")
    ap.add_argument("--out-root", type=Path, default=None,
                    help="defaults to <root>/derived/lora_subtraction")
    ap.add_argument("--models", nargs="+", default=list(DEFAULT_MODEL_KEYS),
                    help="model keys to sweep (default: the round-2 pair "
                         f"{list(DEFAULT_MODEL_KEYS)}; the Llama keys "
                         f"{[k for k in MODEL_KEYS if k not in DEFAULT_MODEL_KEYS]} "
                         "are supported but must be named)")
    ap.add_argument("--cells", nargs="+", default=list(DEFAULT_CELLS),
                    help="mix cell SIs (default: the scale-matched primary "
                         "constv2-mix-100k plus the budget-mismatched "
                         "constv2-mix-1m, both labeled in provenance)")
    ap.add_argument("--pools", type=Path, default=DEFAULT_POOLS,
                    help="staged pool manifests dir (parents' data scales "
                         "for the provenance)")
    ap.add_argument("--ctl-si", default=CTL_SI,
                    help="the control cell SI subtracted from every --cells "
                         "entry (default constv2ctl-100k; the full - full "
                         "pair is constv2ctl2-full)")
    ap.add_argument("--keep-valid", action="store_true",
                    help="keep an existing publication of a pair when it "
                         "passes the shared derived-object verdict against "
                         "--root and names exactly these parents (idempotent "
                         "re-runs); anything else is rebuilt")
    args = ap.parse_args(argv)
    args.root, args.pools = site_path(args.root, "--root"), site_path(args.pools, "--pools")
    out_root = args.out_root or (args.root / "derived/lora_subtraction")
    rc = 0
    for key in args.models:
        for si in args.cells:
            try:
                path = build_one(args.root, out_root, key, si,
                                 pools_dir=args.pools, ctl_si=args.ctl_si,
                                 keep_valid=args.keep_valid)
            # compose_blob's structural refusals (layer sets, rank, shapes,
            # fit legs) are ValueErrors — the same REFUSED verdict as the
            # builder's own, never a traceback.
            except RegistryEmissionFailed as e:
                # the artefact stands; the JOB fails visibly (registry_emission.EMIT_FAILED_RC), never silently
                print(f"[lora-subtract] EMISSION FAILED {key}/{si}: {e}", file=sys.stderr)
                rc = max(rc, _emission().EMIT_FAILED_RC)
                continue
            except (SubtractError, ValueError) as e:
                print(f"[lora-subtract] REFUSED {key}/{si}: {e}",
                      file=sys.stderr)
                rc = 2
                continue
            print(f"[lora-subtract] {key}: {si} - {args.ctl_si} -> {path}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
