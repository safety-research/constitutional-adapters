#!/usr/bin/env python3
"""The Qwen seed-replicate battery — file the wave's objects ONCE.

The 15 lane objects (the grading host's bank ``$STAGE/sdfbase/full22_objects/qwen35_4b/cells/<staged id>/`` — 12 fitted
``base64s<s>__<corpus>__<arm>`` + 3 derived ``derived_base64s<s>__…_minus_ctl``) and the Arditi operating point
(``ar_c0p3__arditi_allpos``, read-only from the grading host's copy) go through ``hooks.emit_object_from_sidecar`` — the
fit writers' own hook — which reads ``fit_seed`` / the fit record from the sidecars and hashes the artifact. Before
anything is emitted the artifact's sha256 is verified against the PIN (``configs/sdfbase_full22_pins.json`` /
``configs/arditi_allpos_pins.json``); a mismatch refuses that object.

What the WRITER must state (the hook never guesses kind, label, role, target kind, objective) is derived from the
registry's OWN declared spellings — ``configs/registry/legacy_spellings.yaml`` through ``fit_sidecars.load_seeds``
(``object_kinds`` for the arm spelling → ``object_kind`` / ``fit_objective`` / ``derivation``; ``paper_labels`` for the
cell spelling → ``paper_label`` / ``object_role`` / ``fit_corpus_id``) — and from the sidecars (``derived_meta.json`` →
``derived``; a document fit → ``corpus_document_fit``); an unknown spelling refuses. Every object row carries
``extra.fit_replicate_label = s<s>`` and ``extra.training_seed = s`` beside the hook's own ``fit_seed``.

Campaigns: the object rows are filed under the FITS' campaign (``reseed`` for the lane objects — the spelling the
registry's own sidecar adapter assigns them; ``arditi_strength`` for the Arditi object), not the battery's run campaign:
an object is campaign-independent and a run's own ``campaign`` names where it was drawn. Both are flags.

  emit_objects.py --bank $STAGE/sdfbase/full22_objects/qwen35_4b/cells --pins configs/sdfbase_full22_pins.json \\
                  --arditi-dir $HOME/arditi-stage/objects/arditi_allpos/qwen35_4b/ladder/cells/ar_c0p3__arditi_allpos \\
                  --arditi-pins configs/arditi_allpos_pins.json [--campaign reseed] [--arditi-campaign arditi_strength] \\
                  [--spool $HOME/registry/records] [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "src") not in sys.path and not any(p.endswith("/src") and "sisteer" in os.listdir(p) for p in sys.path if os.path.isdir(p)):
    sys.path.insert(0, str(REPO / "src"))

from sisteer.registry import hooks  # noqa: E402
from sisteer.registry import vocab as V  # noqa: E402
from sisteer.registry.adapters import fit_sidecars as FS  # noqa: E402
from sisteer.registry.decode_policy import PolicyFile  # noqa: E402

SCRIPT = "scripts/sdfbase/seedrep/emit_objects.py"
MODEL_KEY = "qwen35_4b"
LANE = "seedrep_battery"
LANE_ID_RE = re.compile(r"^(?P<derived>derived_)?base64s(?P<s>[1-9])__(?P<cell>[^_].+)$")
ARDITI_STAGED_ID = "ar_c0p3__arditi_allpos"
ARDITI_KIND = "vector_all"
#: the bank mirror under the staging directory ($STAGE), expanded through sisteer.site where it is read
DEFAULT_BANK = f"$STAGE/sdfbase/full22_objects/{MODEL_KEY}/cells"
DEFAULT_ARDITI_DIR = Path.home() / "arditi-stage" / "objects" / "arditi_allpos" / MODEL_KEY / "ladder" / "cells" / ARDITI_STAGED_ID
DEFAULT_PINS = REPO / "configs" / "sdfbase_full22_pins.json"
DEFAULT_ARDITI_PINS = REPO / "configs" / "arditi_allpos_pins.json"
DEFAULT_SEEDS_DIR = REPO / "configs" / "registry"
DEFAULT_POLICY = REPO / "configs" / "registry" / "decode_policy.json"
DEFAULT_OBJECT_CAMPAIGN = "reseed"            # fit_sidecars: a base64s*/derived_base64s* id is filed under `reseed`
DEFAULT_ARDITI_CAMPAIGN = "arditi_strength"   # legacy_spellings.yaml campaigns: "arditi_strength | arditi"
_OBJECTIVE_BY_SUFFIX = {"ce": "ce", "kl": "kl"}


class ObjectsRefused(ValueError):
    """An object that cannot be described honestly (pin mismatch, unknown spelling, missing sidecar): not emitted."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ObjectsRefused(f"{path}: not a JSON object")
    return data


@dataclass
class ObjectSpec:
    staged_id: str
    obj_dir: Path
    pin_sha: str
    object_kind: str
    paper_label: str
    object_role: str
    fit_target_kind: str
    fit_objective: str
    derivation: str
    fit_corpus_id: Optional[str]
    fit_model_id: Optional[str]
    fit_thinking_render: str
    training_seed: Optional[int]
    extra: Dict[str, object]
    # a DERIVED object's columns (vocab.FIT_TARGET_RULES["derived"]: parent_object_ids_json NOT NULL; the registry's DB CHECK and, since
    # 2026-10-02, the writer-side validator refuse a derived record without them): the two parents by object id, the derivation record
    parent_object_ids: Optional[List[str]] = None
    derivation_params: Optional[Dict[str, object]] = None
    # a CONTRASTIVE direction's columns (FIT_TARGET_RULES["contrastive_direction"]: fit_contrast_sets_json + fit_layers_json NOT NULL): the
    # fit layer(s) from the pins, and the contrast sets as the staged record can honestly state them (the adapters' convention: a
    # self-describing mapping carrying its confidence when the capture-set ids / shas were never recorded)
    fit_layers: Optional[List[int]] = None
    #: the convention of an EXPLICIT fit_layers list (vocab.LAYER_INDEX_CONVENTIONS): the hook states it or refuses (demo-import M1b)
    fit_layer_index_convention: Optional[str] = None
    fit_contrast_sets: Optional[Dict[str, object]] = None


def lane_pins(pins: Mapping[str, Any]) -> Dict[str, str]:
    """``{staged_id: pin sha}`` of the lane's objects (``base64s<s>__`` / ``derived_base64s<s>__`` ids under the model key)."""
    block = pins.get(MODEL_KEY)
    if not isinstance(block, dict):
        raise ObjectsRefused(f"pins carry no {MODEL_KEY!r} block")
    out: Dict[str, str] = {}
    for sid, ent in block.items():
        if not LANE_ID_RE.match(sid):
            continue
        sha = str((ent or {}).get("object_sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise ObjectsRefused(f"pins entry {sid!r} carries no 64-hex object_sha256")
        out[sid] = sha
    if not out:
        raise ObjectsRefused(f"pins carry no lane objects under {MODEL_KEY!r}")
    return out


def verify_pin(obj_dir: Path, pin_sha: str) -> Path:
    art = hooks.find_object_artifact(str(obj_dir))
    if art is None:
        raise ObjectsRefused(f"{obj_dir}: no object artifact")
    got = sha256_file(Path(art))
    if got != pin_sha:
        raise ObjectsRefused(f"{obj_dir}: sha256({os.path.basename(art)}) = {got[:12]}… but the pin says {pin_sha[:12]}…; refusing to describe it")
    return Path(art)


def _fit_model_id(hf_repo: Optional[str], pf: Optional[PolicyFile]) -> Optional[str]:
    if not hf_repo:
        return None
    if pf is None:
        return None
    mid = pf.model_id(str(hf_repo))
    if mid == str(hf_repo) and mid not in pf.models:
        raise ObjectsRefused(f"fit model {hf_repo!r} maps to no registry model_id (decode_policy.json model_spellings)")
    return mid


def describe_lane_object(staged_id: str, bank: Path, pin_sha: str, seeds: FS.Seeds, pf: Optional[PolicyFile]) -> ObjectSpec:
    m = LANE_ID_RE.match(staged_id)
    if not m:
        raise ObjectsRefused(f"{staged_id!r} is not a lane object id")
    obj_dir = bank / staged_id
    if not obj_dir.is_dir():
        raise ObjectsRefused(f"{obj_dir}: missing from the bank")
    verify_pin(obj_dir, pin_sha)
    cell = m.group("cell")
    corpus, _, arm = cell.partition("__")
    if not arm:
        raise ObjectsRefused(f"{staged_id!r}: cell {cell!r} has no <corpus>__<arm> shape")
    kind_entry = seeds.object_kind_for(arm) or seeds.object_kind_for(cell)
    if not kind_entry or kind_entry.get("object_kind") not in V.OBJECT_KINDS:
        raise ObjectsRefused(f"{staged_id!r}: no legacy_spellings.yaml object_kinds entry for arm {arm!r} / cell {cell!r}")
    label_entry = seeds.paper_label_for(cell)
    if not label_entry or label_entry.get("paper_label") not in V.PAPER_LABELS or label_entry.get("object_role") not in V.OBJECT_ROLES:
        raise ObjectsRefused(f"{staged_id!r}: no legacy_spellings.yaml paper_labels entry for cell {cell!r}")
    side = hooks.read_sidecars(str(obj_dir))
    tbm = side.get("train_budget_meta.json") or {}
    dm = side.get("derived_meta.json") or {}
    derived = bool(m.group("derived"))
    if derived != bool(dm):
        raise ObjectsRefused(f"{staged_id!r}: id says derived={derived} but derived_meta.json present={bool(dm)}")
    if derived:
        derivation = str(kind_entry.get("derivation") or dm.get("kind") or "")
        if derivation not in V.DERIVATIONS:
            raise ObjectsRefused(f"{staged_id!r}: derivation {derivation!r} not in DERIVATIONS")
        parents = [(dm.get("parent_pos") or {}).get("object_sha256"), (dm.get("parent_neg") or {}).get("object_sha256")]
        if not all(isinstance(x, str) and len(x) == 64 for x in parents):
            raise ObjectsRefused(f"{staged_id!r}: derived_meta.json names no complete parents (parent_pos/parent_neg object_sha256 = {parents}) — "
                                 "a derived record carries both parents by object id (FIT_TARGET_RULES['derived']); nothing emitted")
        spec = ObjectSpec(staged_id, obj_dir, pin_sha, str(kind_entry["object_kind"]), str(label_entry["paper_label"]), str(label_entry["object_role"]),
                          "derived", "not_applicable", derivation, label_entry.get("fit_corpus_id") or None,
                          _fit_model_id(dm.get("model_id"), pf), "not_applicable", int(m.group("s")),
                          {"parent_object_sha256s": [dm.get("parent_pos", {}).get("object_sha256"), dm.get("parent_neg", {}).get("object_sha256")],
                           "parent_cell_ids": [dm.get("parent_pos", {}).get("cell_id"), dm.get("parent_neg", {}).get("cell_id")],
                           "derived_id": dm.get("derived_id"), "derivation_note": dm.get("derivation")},
                          parent_object_ids=parents, derivation_params=hooks.derivation_params_from_derived_meta(dm))
    else:
        if not tbm:
            raise ObjectsRefused(f"{staged_id!r}: a fitted object without train_budget_meta.json")
        if tbm.get("training_budget_kind") != "document_tokens" and tbm.get("data_mode") != "documents":
            raise ObjectsRefused(f"{staged_id!r}: train_budget_meta.json is not a document fit (training_budget_kind={tbm.get('training_budget_kind')!r}, "
                                 f"data_mode={tbm.get('data_mode')!r}); fit_target_kind undecidable")
        seed = tbm.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed != int(m.group("s")):
            raise ObjectsRefused(f"{staged_id!r}: sidecar seed {seed!r} disagrees with the id's training seed s{m.group('s')}")
        obj_text = str(kind_entry.get("fit_objective") or "")
        objective = obj_text if obj_text in V.FIT_OBJECTIVES else None
        if objective is None:
            for suf, val in _OBJECTIVE_BY_SUFFIX.items():
                if re.search(rf"(^|_){suf}(_|$)", arm):
                    objective = val
                    break
        if objective is None:
            raise ObjectsRefused(f"{staged_id!r}: fit objective undecidable from arm {arm!r} ({obj_text!r})")
        render = "not_applicable" if tbm.get("chat_template_bypassed") is True else "unverified"
        spec = ObjectSpec(staged_id, obj_dir, pin_sha, str(kind_entry["object_kind"]), str(label_entry["paper_label"]), str(label_entry["object_role"]),
                          "corpus_document_fit", objective, "none", label_entry.get("fit_corpus_id") or corpus,
                          _fit_model_id(tbm.get("model_id"), pf), render, seed,
                          {"fit_fingerprint": (tbm.get("fit_record") or {}).get("fit_fingerprint"), "effective_batch": tbm.get("effective_batch"),
                           "corpus": corpus})
    spec.extra.update({"fit_replicate_label": f"s{m.group('s')}", "training_seed": int(m.group("s")), "lane": LANE, "staged_id": staged_id,
                       "pin_sha256": pin_sha, "arm_legacy": arm, "cell_id_legacy": cell})
    return spec


def describe_arditi(cell_dir: Path, arditi_pins: Mapping[str, Any]) -> ObjectSpec:
    ent = (arditi_pins.get(MODEL_KEY) or {}).get(ARDITI_STAGED_ID)
    if not isinstance(ent, dict):
        raise ObjectsRefused(f"arditi pins carry no {MODEL_KEY}/{ARDITI_STAGED_ID}")
    sha = str(ent.get("object_sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise ObjectsRefused("arditi pin carries no 64-hex object_sha256")
    if ent.get("kind") != ARDITI_KIND:
        raise ObjectsRefused(f"arditi pin kind {ent.get('kind')!r} is not {ARDITI_KIND!r}")
    if not cell_dir.is_dir():
        raise ObjectsRefused(f"{cell_dir}: missing")
    verify_pin(cell_dir, sha)
    prov = hooks.read_sidecars(str(cell_dir)).get("fit_provenance.json") or {}
    blk = prov.get("arditi_allpos") if isinstance(prov.get("arditi_allpos"), dict) else {}
    extra = {"dose": {"c": ent.get("c"), "unit": ent.get("dose_unit") or "raw_coefficient"}, "alpha_equiv": ent.get("alpha_equiv"),
             "served_layer": ent.get("served_layer"), "fit_layer": ent.get("fit_layer"), "direction_sha256": ent.get("direction_sha256"),
             "positions": ent.get("positions"), "phase": ent.get("phase"), "staged_id": ARDITI_STAGED_ID, "pin_sha256": sha, "lane": LANE,
             "producer": blk.get("producer"), "producer_commit": blk.get("producer_commit"), "fit_positions_declared": ent.get("fit_positions"),
             "cross_model_caveat": blk.get("cross_model_caveat")}
    fit_layer = ent.get("fit_layer")
    if not isinstance(fit_layer, int):
        raise ObjectsRefused(f"arditi pin carries no integer fit_layer (got {fit_layer!r}) — a contrastive_direction record requires fit_layers_json")
    # the layer column is the SERVED block (a zero-based decoder block: `_adapt_vector_all` serves the stored block-output layer as stored),
    # the statement the arditi_allpos adapter files from cell_facts.json:served_layer — never `fit_layer` (13: the fit-time residual index,
    # another convention; it rides `extra.fit_layer`). demo-import (Q14's corollary, 2026-10-10): the hook states an explicit list's
    # convention or refuses, so the one stated here is the served block's; before this the tool filed [13]. The layer is READ FROM THE
    # OBJECT'S OWN PROVENANCE (fit_provenance.json:arditi_allpos.served_layer AND .stored_layer, which must agree), and the pin's served_layer
    # must agree with it — the pin's object_sha256 protects the bytes, not the pin's layer metadata, so an edited pin (served_layer 13 over the
    # same bytes) is refused here by name rather than filed as the writer's statement (the registry owner's review of the hook diff).
    pin_served = ent.get("served_layer")
    if not isinstance(pin_served, int) or isinstance(pin_served, bool):
        raise ObjectsRefused(f"arditi pin carries no integer served_layer (got {pin_served!r}) — the contrastive_direction record states the SERVED "
                             "block as its fit_layers_json (zero_based_decoder_block), never a conversion of fit_layer")
    prov_served, prov_stored = blk.get("served_layer"), blk.get("stored_layer")
    for name, val in (("served_layer", prov_served), ("stored_layer", prov_stored)):
        if not isinstance(val, int) or isinstance(val, bool):
            raise ObjectsRefused(f"{cell_dir}: fit_provenance.json:arditi_allpos.{name} is not an integer (got {val!r}) — the served block is read from the "
                                 "object's own provenance, never from the pin alone")
    if prov_served != prov_stored:
        raise ObjectsRefused(f"{cell_dir}: fit_provenance.json:arditi_allpos.served_layer {prov_served} disagrees with .stored_layer {prov_stored} — a vector_all "
                             "blob stores the served block-output layer and is served as stored; the object's provenance does not describe one layer")
    if pin_served != prov_served:
        raise ObjectsRefused(f"{cell_dir}: the arditi pin's served_layer {pin_served} disagrees with fit_provenance.json:arditi_allpos.served_layer {prov_served} "
                             f"(= stored_layer) — the pin's sha protects the bytes, not its layer metadata; the object's own provenance is the statement")
    served_layer = prov_served
    # the staged Arditi direction records no capture-set ids or shas (its producer composed it from Arditi's recipe); the column the
    # pairing rule requires is stated as the adapters state it — a mapping that carries its own confidence, never a guess at the sets
    sets = {"confidence": "unverified", "source": "arditi pins + the cell's fit_provenance.json",
            "reason": "the staged Arditi direction records no capture-set ids or shas" + (f" (producer {blk.get('producer')})" if blk.get("producer") else "")}
    return ObjectSpec(ARDITI_STAGED_ID, cell_dir, sha, ARDITI_KIND, "arditi", "baseline_published", "contrastive_direction", "arditi_search",
                      "none", None, f"{MODEL_KEY}_it", "unverified", None, extra, fit_layers=[served_layer], fit_contrast_sets=sets,
                      fit_layer_index_convention="zero_based_decoder_block")


def emit_spec(spec: ObjectSpec, *, campaign: str, spool: Optional[str], status: str = "approved", dry_run: bool = False,
              code_sha: Optional[str] = None) -> Dict[str, object]:
    if status not in V.OBJECT_STATUSES:
        raise ObjectsRefused(f"status {status!r} not in OBJECT_STATUSES")
    kwargs: Dict[str, object] = dict(campaign=campaign, object_kind=spec.object_kind, paper_label=spec.paper_label, object_role=spec.object_role,
                                     fit_target_kind=spec.fit_target_kind, fit_objective=spec.fit_objective, fit_thinking_render=spec.fit_thinking_render,
                                     status=status, derivation=spec.derivation, extra=dict(spec.extra, code_sha_at_emission=code_sha))
    if spec.fit_model_id:
        kwargs["fit_model_id"] = spec.fit_model_id
    if spec.fit_corpus_id:
        kwargs["fit_corpus_id"] = spec.fit_corpus_id
    if spec.parent_object_ids:                       # the derived columns the pairing rule requires
        kwargs["parent_object_ids_json"] = list(spec.parent_object_ids)
    if spec.derivation_params:
        kwargs["derivation_params_json"] = dict(spec.derivation_params)
    if spec.fit_layers:                              # the contrastive columns the pairing rule requires
        kwargs["fit_layers_json"] = list(spec.fit_layers)
        if spec.fit_layer_index_convention:
            kwargs["fit_layer_index_convention"] = spec.fit_layer_index_convention
    if spec.fit_contrast_sets:
        kwargs["fit_contrast_sets_json"] = dict(spec.fit_contrast_sets)
    entry: Dict[str, object] = {"staged_id": spec.staged_id, "object_id": spec.pin_sha, "object_kind": spec.object_kind, "paper_label": spec.paper_label,
                                "object_role": spec.object_role, "fit_target_kind": spec.fit_target_kind, "fit_objective": spec.fit_objective,
                                "campaign": campaign, "dir": str(spec.obj_dir), "paths": []}
    if dry_run:
        # the payload the hook WOULD write, built through the same function (it raises on a sidecar/artifact sha disagreement),
        # so a dry run refuses exactly what a real run would; nothing is written
        payload, _artifact = hooks.object_from_sidecar(str(spec.obj_dir), producer=hooks.emit.producer_block(SCRIPT, code_sha=code_sha), **kwargs)
        entry["would_emit"] = {k: payload.get(k) for k in ("object_id", "object_kind", "paper_label", "object_role", "fit_target_kind", "fit_objective",
                                                           "fit_positions", "fit_seed", "fit_seed_confidence", "fit_model_id", "campaign", "status",
                                                           "derivation", "parent_object_ids_json", "fit_layers_json", "fit_contrast_sets_json")}
        return entry
    paths = hooks.emit_object_from_sidecar(str(spec.obj_dir), script=SCRIPT, spool=spool, registry_emit=True, **kwargs)
    if not paths:
        raise ObjectsRefused(f"{spec.staged_id}: emit_object_from_sidecar wrote nothing (see the [registry] notice)")
    entry["paths"] = list(paths)
    return entry


def emit_all(*, bank: Path, pins_path: Path, arditi_dir: Optional[Path], arditi_pins_path: Optional[Path], campaign: str, arditi_campaign: str,
             seeds_dir: Path, policy_path: Optional[Path], spool: Optional[str], dry_run: bool, status: str = "approved",
             code_sha: Optional[str] = None, only: Optional[Sequence[str]] = None) -> List[Dict[str, object]]:
    seeds = FS.load_seeds(str(seeds_dir))
    pf = PolicyFile.load(str(policy_path)) if policy_path else None
    pins = lane_pins(_read_json(pins_path))
    specs: List[ObjectSpec] = []
    for sid in sorted(pins):
        if only and sid not in only:
            continue
        specs.append(describe_lane_object(sid, bank, pins[sid], seeds, pf))
    out = [emit_spec(s, campaign=campaign, spool=spool, status=status, dry_run=dry_run, code_sha=code_sha) for s in specs]
    if arditi_dir is not None and (not only or ARDITI_STAGED_ID in only):
        if arditi_pins_path is None:
            raise ObjectsRefused("--arditi-dir needs --arditi-pins")
        out.append(emit_spec(describe_arditi(arditi_dir, _read_json(arditi_pins_path)), campaign=arditi_campaign, spool=spool, status=status,
                             dry_run=dry_run, code_sha=code_sha))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bank", default=str(DEFAULT_BANK))
    ap.add_argument("--pins", default=str(DEFAULT_PINS))
    ap.add_argument("--arditi-dir", default=str(DEFAULT_ARDITI_DIR))
    ap.add_argument("--arditi-pins", default=str(DEFAULT_ARDITI_PINS))
    ap.add_argument("--no-arditi", action="store_true")
    ap.add_argument("--campaign", default=DEFAULT_OBJECT_CAMPAIGN, help="the lane objects' campaign (the fits', not the battery's)")
    ap.add_argument("--arditi-campaign", default=DEFAULT_ARDITI_CAMPAIGN)
    ap.add_argument("--seeds-dir", default=str(DEFAULT_SEEDS_DIR))
    ap.add_argument("--policy", default=str(DEFAULT_POLICY), help="decode_policy.json (model spellings for fit_model_id)")
    ap.add_argument("--spool", default=None, help="the registry spool (default: the resolved grading host spool)")
    ap.add_argument("--status", default="approved")
    ap.add_argument("--code-sha", default=None)
    ap.add_argument("--only", nargs="*", default=None, help="a subset of staged ids")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    try:
        from sisteer import site as _site
        out = emit_all(bank=_site.expand_path(a.bank, key="--bank"), pins_path=Path(a.pins), arditi_dir=None if a.no_arditi else Path(a.arditi_dir),
                       arditi_pins_path=Path(a.arditi_pins), campaign=a.campaign, arditi_campaign=a.arditi_campaign, seeds_dir=Path(a.seeds_dir),
                       policy_path=Path(a.policy) if a.policy else None, spool=a.spool, dry_run=a.dry_run, status=a.status, code_sha=a.code_sha,
                       only=a.only)
    except ObjectsRefused as exc:
        print(f"emit_objects: REFUSED: {exc}", file=sys.stderr)
        return 2
    for e in out:
        print(json.dumps(e, sort_keys=True))
    print(f"emit_objects: {len(out)} object(s) {'described (dry run)' if a.dry_run else 'emitted'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
