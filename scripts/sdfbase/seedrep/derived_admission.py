#!/usr/bin/env python3
"""derived_admission — the stager's PREDICATE for one derived object of the
seed-replicate lane: ``<derived_dir>/ADMISSION.json``. A pure read of what
is on disk; re-runnable any time.

    derived_admission.py <derived_dir> --parents <pos_cell_dir> <neg_cell_dir> \\
        [--floor <f>] [--dead-floor <d>] [--key <key>] [--seed <s>] [--out ADMISSION.json]

ADMITTED iff every reason below holds; otherwise ``admitted: false`` with
every failing reason named, the artifact PRESERVED on disk (never deleted or
moved), excluded from staging, the roster and the battery, and reported.

Common to both derived kinds (``derived_meta.json`` ``kind``: ``psr`` = the
residual distillation, ``lora_subtraction`` = the task-vector difference):

* both parents' ``train_budget_meta.json`` + ``COMPLETE`` exist and agree
  with the derived object's ``parent_pos`` / ``parent_neg`` blocks on
  ``cell_id``, ``object_sha256`` AND ``fit_fingerprint`` (identity by sha
  and fingerprint, never by name alone);
* both parents record the SAME ``seed`` (top-level in the sidecar) and,
  when ``--seed`` is given, that seed; the derived ``model_key`` equals
  both parent blocks' and, when ``--key`` is given, that key (the parent
  dirs must also sit under ``…/<key>/cells_all/``);
* the parents agree on every :data:`PARENT_MATCH_FIELDS` entry (the shared
  ``require_parents_match`` tuple — pinned equal to
  ``sisteer.eval.train_budget.DERIVED_PARENT_MATCH_FIELDS`` by the tests).

Distilled (``psr``): ``gate_liveness.json`` present, numeric, and NOT FULLY
DEAD ANYWHERE — every layer's live fraction ``> --dead-floor`` (default
``0.0``; a layer at or below it names the object refused). ``report.json``
present with ``recovery.verdict`` equal to ``derived_meta.recovery_verdict``.
``sweep.json`` present with every part of ``lr_mults ∪ extensions`` complete;
``assess.json`` present with ``selected_lr_mult == trainer.lr_mult`` and the
selection INTERIOR — or on an edge only after the one extension, in which
case it is admitted with the alarm stated (``alarms``); an edge with NO
extension is not admitted.

Two facts that were refusal grounds until 2026-09-18 are now RECORDED ALARMS
— a distilled object is admitted as long as it is not fully dead anywhere: a
``recovery_verdict`` other than ``"recovered"`` (recorded verbatim), and a
min over layers below the published Qwen derived object's own min (``--floor``
/ :data:`DEFAULT_FLOOR`, kept as ``liveness_reference``; every layer below it
is listed in ``layers_below_reference``). Nothing measured leaves the record:
``admission_rule`` and ``superseded_rule`` name both rules and the date.

Subtraction (``lora_subtraction``): ``scale_matched: true``.

Exit 0 admitted, 4 not admitted (ADMISSION.json written either way), 3 when
the derived dir cannot be read at all.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for _p in (REPO / "src",):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

#: The published Qwen derived object's (``derived__constv2-mix-full__psr_ce_all_
#: minus_ctl_distilled``, object sha 55c08d97) gate liveness: min over ALL 32
#: layers of its ``gate_liveness.json``, taken in layer-index order — layer 9;
#: its ``gate_liveness_detail.json`` records the same ``min_live_frac``. Written
#: as the exact float, not rounded: 0.8135 rounds UP and would refuse the very
#: object the floor is named after (0.81347 < 0.8135), and the pin test asserts
#: that object passes at equality. The earlier constant 0.933 was a misread:
#: it is layer 10, the first sub-1.0 entry when the file's string-sorted keys
#: ("0", "1", "10", "11", …) are read in file order; under it the published
#: object refused itself on 4 of its 32 layers (2026-09-17, liveness_refusal.md).
#: ``tests/fixtures/seedrep/published_qwen_derived_gate_liveness.json`` is the
#: 32-value file; ``tests/test_seedrep_derived_admission.py`` pins this constant
#: equal to its min.
DEFAULT_FLOOR = 0.8134719699019494
#: Since 2026-09-18 the published object's min is a REFERENCE that is
#: recorded (``liveness_reference``, ``layers_below_reference``, an alarm), not
#: a refusal ground; the only liveness refusal is a FULLY DEAD layer, live
#: fraction ``<= DEFAULT_DEAD_FLOOR``.
PUBLISHED_REFERENCE = DEFAULT_FLOOR
DEFAULT_DEAD_FLOOR = 0.0
ADMISSION_RULE = ("distilled: not fully dead anywhere (every layer's live fraction > "
                  "dead_floor); recovery verdict and the published-reference liveness "
                  "are recorded alarms — user ruling 2026-09-18, "
                  "qwen-seedrep-native-battery D69")
SUPERSEDED_RULE = ("distilled: recovery_verdict == 'recovered' AND liveness min over "
                   "layers >= the published Qwen derived object's min (until 2026-09-18)")
#: sisteer.eval.train_budget.DERIVED_PARENT_MATCH_FIELDS, spelled out so this
#: file runs under a bare python3 too; tests/test_seedrep_derived_admission.py
#: pins the two equal.
PARENT_MATCH_FIELDS = ("model_id", "model_revision", "target_layers",
                       "positions", "response_header_len", "template_hash")
KIND_DISTILLED = "psr"
KIND_SUBTRACTION = "lora_subtraction"
#: the COMPOSED paired write (scripts/sdfbase/psr_residual_distill.py compose-v: u_pos(z) - u_neg(z) on the shared
#: pre-write layer output, both parents' parameters stored bit-exact) — the difference_relu arm since 2026-09-19. It has
#: NO distillation trainer, so no recovery / sweep / assess artifacts exist for it: recovery is recorded as NOT
#: APPLICABLE with the reason, never omitted. Its liveness is judged as the distilled object's is (fully dead refuses;
#: the published reference is a recorded alarm that travels on the record, the store rows and every caption), and
#: scale_matched must hold as for a subtraction.
KIND_PAIRED_WRITE = "psr_paired_write"
KINDS = (KIND_DISTILLED, KIND_SUBTRACTION, KIND_PAIRED_WRITE)
PAIRED_WRITE_RULE = ("paired_write: parents' identity by sha + fingerprint, same seed, the shared tuple; scale_matched "
                     "true; not fully dead anywhere (every layer's live fraction > dead_floor); the published-reference "
                     "liveness is a recorded alarm that travels with the object (D69 rule, applied 2026-09-20); recovery "
                     "NOT APPLICABLE (no distillation trainer for this kind)")
RECOVERY_NOT_APPLICABLE = "not_applicable"
RECOVERY_NOT_APPLICABLE_REASON = ("a psr_paired_write composes u_pos(z) - u_neg(z) on the shared pre-write state with both "
                                  "parents stored bit-exact; there is no distillation trainer, so no recovery verdict, "
                                  "sweep or assessment exists for this kind — checked: does not apply (not: passed)")

__all__ = ["admission", "liveness_min", "dead_layers", "PARENT_MATCH_FIELDS", "DEFAULT_FLOOR",
           "PUBLISHED_REFERENCE", "DEFAULT_DEAD_FLOOR", "ADMISSION_RULE", "SUPERSEDED_RULE",
           "KIND_PAIRED_WRITE", "PAIRED_WRITE_RULE", "RECOVERY_NOT_APPLICABLE",
           "RECOVERY_NOT_APPLICABLE_REASON", "main"]


class InputError(RuntimeError):
    """The derived dir cannot be read at all (exit 3)."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path, what: str, *, required: bool = True) -> dict | None:
    if not path.is_file():
        if required:
            raise InputError(f"{what} missing at {path}")
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as e:
        raise InputError(f"{what} at {path} unreadable: {e}")
    if not isinstance(doc, dict):
        raise InputError(f"{what} at {path} is not a JSON object")
    return doc


def _fmt(m) -> str:
    return f"{float(m):g}"


def _parent(cell_dir: Path, role: str, block: dict, reasons: list[str],
            *, key: str | None) -> dict:
    """One parent's identity off its own sidecars, compared to the derived
    object's parent block."""
    rec = {"role": role, "dir": str(cell_dir), "cell_id": block.get("cell_id"),
           "object_sha256": None, "fit_fingerprint": None, "seed": None,
           "key": None, "sidecars_ok": False}
    meta = _read_json(cell_dir / "train_budget_meta.json",
                      f"{role} parent train_budget_meta.json", required=False)
    marker = _read_json(cell_dir / "COMPLETE", f"{role} parent COMPLETE",
                        required=False)
    if meta is None or marker is None:
        reasons.append(f"{role} parent at {cell_dir} lacks "
                       f"{'train_budget_meta.json' if meta is None else 'COMPLETE'}")
        return rec
    rec["sidecars_ok"] = True
    rec["object_sha256"] = meta.get("object_sha256")
    rec["fit_fingerprint"] = marker.get("fit_fingerprint")
    rec["seed"] = meta.get("seed")
    try:
        rec["key"] = cell_dir.resolve().parents[1].name \
            if cell_dir.resolve().parent.name == "cells_all" else None
    except IndexError:
        rec["key"] = None
    rec["match_fields"] = {f: meta.get(f) for f in PARENT_MATCH_FIELDS}
    if meta.get("cell_id") != block.get("cell_id"):
        reasons.append(f"{role} parent sidecar is cell {meta.get('cell_id')!r}, "
                       f"derived_meta names {block.get('cell_id')!r}")
    if marker.get("cell_id") not in (None, meta.get("cell_id")):
        reasons.append(f"{role} parent COMPLETE names {marker.get('cell_id')!r}, "
                       f"its sidecar {meta.get('cell_id')!r}")
    for field, mine in (("object_sha256", meta.get("object_sha256")),
                        ("fit_fingerprint", marker.get("fit_fingerprint"))):
        theirs = block.get(field)
        if not mine or not theirs:
            reasons.append(f"{role} parent {field} absent "
                           f"({'sidecar' if not mine else 'derived_meta block'})")
        elif mine != theirs:
            reasons.append(f"{role} parent {field} {mine[:12]}… != derived_meta's "
                           f"{theirs[:12]}… — the derived object was built on a "
                           "different publication of this cell")
    if marker.get("object_sha256") not in (None, meta.get("object_sha256")):
        reasons.append(f"{role} parent COMPLETE object_sha256 "
                       f"{str(marker.get('object_sha256'))[:12]}… != sidecar "
                       f"{str(meta.get('object_sha256'))[:12]}…")
    if rec["seed"] is None:
        reasons.append(f"{role} parent sidecar records no seed")
    if key is not None and rec["key"] != key:
        reasons.append(f"{role} parent dir {cell_dir} is not under …/{key}/cells_all/")
    # the parent's BYTES, not only its sidecar: the bank must hold the object this derivation consumed — verify the
    # actual parent identities
    p_obj = cell_dir / "object.pt"
    if not p_obj.is_file():
        reasons.append(f"{role} parent has no object.pt at {cell_dir}")
    elif meta.get("object_sha256") and _sha256(p_obj) != meta.get("object_sha256"):
        reasons.append(f"{role} parent object.pt hashes to {_sha256(p_obj)[:12]}… != its sidecar's "
                       f"{str(meta.get('object_sha256'))[:12]}… — the bank's bytes are not the object this derivation consumed")
    rec["object_bytes_verified"] = p_obj.is_file() and _sha256(p_obj) == meta.get("object_sha256")
    return rec


def _bound_paired_write(derived: Path, meta: dict, reasons: list[str]) -> dict:
    """The paired-write branch is selected and judged from the BYTES, never from the sidecar's labels (the object hash
    binds bytes, not the sidecar's separate claims; a record under validation cannot supply its own inputs):
    (1) the shared derived-object validator (``sisteer.eval.train_budget.derived_object_problems``) — sidecar sha == bytes,
    derived_id recorded, parent blocks complete + matched, the artifact's own ``kind`` == the sidecar's, the artifact's embedded
    provenance names the SAME parents (cell_id + object_sha256); (2) the blob's ``kind`` must be ``psr_paired_write`` — a plain
    ``psr`` object relabelled in its sidecar is refused here, before any recovery exemption applies; (3) ``scale_matched`` is
    DERIVED from the embedded parents' ``data_scale.official_tokens`` (the bound evidence compose-v wrote) and every claim of it
    (sidecar, embedded flag) must agree with the derivation. Returns the record block; every failure is a refusal reason."""
    rec = {"shared_validator": "sisteer.eval.train_budget.derived_object_problems", "shared_validator_problems": None,
           "blob_kind": None, "embedded_official_tokens": {"pos": None, "neg": None}, "scale_matched_derived": None,
           "scale_matched_sidecar": meta.get("scale_matched"), "scale_matched_embedded": None}
    try:
        from sisteer.eval.train_budget import derived_object_problems
    except Exception as e:  # noqa: BLE001
        reasons.append(f"bytes: the shared derived-object validator cannot be imported ({type(e).__name__}: {e}) — refusing to admit on sidecar claims alone")
        return rec
    probs = derived_object_problems(derived / "object.pt", derived_id=meta.get("derived_id"))
    rec["shared_validator_problems"] = len(probs)
    reasons.extend(f"bytes: {pr}" for pr in probs)
    try:
        import torch
        blob = torch.load(derived / "object.pt", map_location="cpu", weights_only=True)
    except Exception as e:  # noqa: BLE001
        reasons.append(f"bytes: object.pt does not load as a weights-only torch blob ({type(e).__name__}: {e})")
        return rec
    if not isinstance(blob, dict):
        reasons.append("bytes: object.pt is not a dict blob"); return rec
    rec["blob_kind"] = blob.get("kind")
    if blob.get("kind") != KIND_PAIRED_WRITE:
        reasons.append(f"bytes: the object IS a {blob.get('kind')!r}, the sidecar says {KIND_PAIRED_WRITE!r} — the recovery "
                       "exemption is selected from the bytes, never from the sidecar's label")
    prov = blob.get("provenance") or {}
    toks = {}
    for role in ("pos", "neg"):
        block = prov.get(f"parent_{role}") or {}
        toks[role] = ((block.get("data_scale") or {}).get("official_tokens")) if isinstance(block, dict) else None
    rec["embedded_official_tokens"] = toks
    rec["scale_matched_embedded"] = prov.get("scale_matched")
    if toks["pos"] is None or toks["neg"] is None:
        reasons.append("bytes: the embedded provenance carries no parent official_tokens — scale matching cannot be derived from bound evidence")
        return rec
    matched = toks["pos"] == toks["neg"]
    rec["scale_matched_derived"] = matched
    if meta.get("scale_matched") is not matched:
        reasons.append(f"scale_matched: the sidecar claims {meta.get('scale_matched')!r} but the embedded parent budgets are "
                       f"{toks['pos']} vs {toks['neg']} official tokens (derived {matched}) — the claim is not bound to the object")
    if prov.get("scale_matched") is not None and bool(prov.get("scale_matched")) is not matched:
        reasons.append(f"scale_matched: the embedded flag {prov.get('scale_matched')!r} contradicts the embedded budgets "
                       f"{toks['pos']} vs {toks['neg']} (derived {matched})")
    if not matched:
        reasons.append(f"scale_matched False by the bound evidence: parents trained at {toks['pos']} vs {toks['neg']} official tokens")
    return rec


def _sweep_and_assess(derived: Path, meta: dict, reasons: list[str],
                      alarms: list[str]) -> dict:
    from derived_sweep import (ASSESS_NAME, SWEEP_NAME, SweepError,  # noqa: E402
                               ladder_of, load_sweep, parts_report)

    out = {"sweep": None, "assess": None, "parts_complete": None,
           "selected_lr_mult": None}
    try:
        sweep = load_sweep(derived)
    except SweepError as e:
        reasons.append(f"{SWEEP_NAME}: {e}")
        return out
    if sweep is None:
        reasons.append(f"no {SWEEP_NAME} under {derived} — the ladder was never "
                       "declared (a tree built before the declaration, or not "
                       "through the sbatch)")
        return out
    ladder = ladder_of(sweep)
    exts = [float(x) for x in (sweep.get("extensions") or [])]
    parts = parts_report(derived, ladder)
    incomplete = {k: v["why"] for k, v in parts.items() if not v["complete"]}
    out["sweep"] = {"lr_mults": [float(x) for x in sweep["lr_mults"]],
                    "extensions": exts, "ladder": ladder}
    out["parts_complete"] = not incomplete
    if incomplete:
        reasons.append(f"calibration parts incomplete for lr_mult "
                       f"{sorted(incomplete)}: {incomplete}")
    a = _read_json(derived / ASSESS_NAME, ASSESS_NAME, required=False)
    if a is None:
        reasons.append(f"no {ASSESS_NAME} under {derived} — the selection was "
                       "never recorded before distillation")
        return out
    out["assess"] = {k: a.get(k) for k in ("selected_lr_mult", "ladder", "on_edge",
                                           "edge_side", "recommended_extension",
                                           "alarm", "assessed_at")}
    sel = a.get("selected_lr_mult")
    out["selected_lr_mult"] = sel
    prod = (meta.get("trainer") or {}).get("lr_mult")
    if sel is None:
        reasons.append("assess.json records no selection (the sweep refused)")
    elif prod is None or _fmt(prod) != _fmt(sel):
        reasons.append(f"distilled trainer.lr_mult {prod} != assessed selection "
                       f"{_fmt(sel)}")
    if sorted(float(x) for x in (a.get("ladder") or [])) != ladder:
        reasons.append(f"assess.json ladder {a.get('ladder')} != declared "
                       f"{ladder} — assessed on another ladder")
    if a.get("on_edge") is True:
        if exts:
            alarms.append(f"edge selection ({_fmt(sel) if sel is not None else '?'}, "
                          f"{a.get('edge_side')}) persists after the one D7 "
                          f"extension {[_fmt(x) for x in exts]} — published "
                          "with the alarm stated (decisions D7)")
        else:
            reasons.append(f"selection {_fmt(sel) if sel is not None else '?'} "
                           f"is on the {a.get('edge_side')} edge of "
                           f"{ladder} and the D7 extension rung "
                           f"({a.get('recommended_extension')}) was never run")
    elif a.get("on_edge") is None and sel is not None:
        reasons.append("assess.json carries no edge decision")
    return out


def _liveness_checks(derived: Path, reasons: list[str], alarms: list[str], *, floor: float,
                     dead_floor: float):
    """gate_liveness.json judged the admission way: absent / non-numeric refuses; a FULLY DEAD layer (live fraction <=
    dead_floor) refuses; layers below the published reference are a recorded alarm. Returns
    (live_min, live_layer, dead, below_ref, n_layers), each None when unreadable."""
    live_min = live_layer = dead = below_ref = n_layers = None
    gl = _read_json(derived / "gate_liveness.json", "gate_liveness.json", required=False)
    if not gl:
        reasons.append("gate_liveness.json absent or empty")
        return live_min, live_layer, dead, below_ref, n_layers
    try:
        live_min, live_layer = liveness_min(gl)
        dead = dead_layers(gl, dead_floor)
        below_ref = layers_below(gl, floor)
        n_layers = len(gl)
    except (TypeError, ValueError, AttributeError):
        reasons.append("gate_liveness.json values are not all numbers")
        return live_min, live_layer, dead, below_ref, n_layers
    if dead:
        # The ONE liveness refusal ground: a fully dead layer.
        reasons.append(f"gate liveness fully dead (live fraction <= {dead_floor}) "
                       f"at layer(s) {', '.join(dead)} — not admitted: the object is "
                       f"dead somewhere")
    elif below_ref:
        alarms.append(f"gate liveness min over layers {live_min:.4f} (layer "
                      f"{live_layer}) < published reference {floor} on "
                      f"{len(below_ref)} of {n_layers} layers (recorded; not a "
                      f"refusal ground since 2026-09-18, D69)")
    return live_min, live_layer, dead, below_ref, n_layers


def liveness_min(gl: dict) -> tuple[float, str]:
    """``(min, layer)`` over EVERY entry of a ``gate_liveness.json`` mapping
    (``{"<layer>": live_frac}``), whatever order the keys arrive in — the file
    is written with string-sorted keys (``"0", "1", "10", …``), so the first
    sub-1.0 entry in file order is NOT the min (that misread produced the
    0.933 floor). Raises ``TypeError``/``ValueError`` when a value is not a
    number, ``ValueError`` on an empty mapping."""
    vals = {str(k): float(v) for k, v in gl.items()}
    if not vals:
        raise ValueError("gate_liveness.json has no layers")
    by_layer = sorted(vals, key=lambda k: (0, int(k)) if k.isdigit() else (1, k))
    layer = min(by_layer, key=vals.__getitem__)    # ties: the lowest layer index
    return vals[layer], layer


def dead_layers(gl: dict, dead_floor: float = DEFAULT_DEAD_FLOOR) -> list[str]:
    """Layers whose live fraction is ``<= dead_floor`` — FULLY DEAD under the
    admission rule — in layer-index order. Same numeric contract as
    :func:`liveness_min` (raises on a non-numeric value)."""
    vals = {str(k): float(v) for k, v in gl.items()}
    by_layer = sorted(vals, key=lambda k: (0, int(k)) if k.isdigit() else (1, k))
    return [k for k in by_layer if vals[k] <= dead_floor]


def layers_below(gl: dict, reference: float) -> list[str]:
    """Layers whose live fraction is ``< reference`` (the published object's
    min), in layer-index order — recorded, never a refusal ground."""
    vals = {str(k): float(v) for k, v in gl.items()}
    by_layer = sorted(vals, key=lambda k: (0, int(k)) if k.isdigit() else (1, k))
    return [k for k in by_layer if vals[k] < reference]


def admission(derived_dir, *, pos_dir, neg_dir, floor: float = DEFAULT_FLOOR,
              dead_floor: float = DEFAULT_DEAD_FLOOR,
              key: str | None = None, seed: int | None = None) -> dict:
    derived = Path(derived_dir)
    meta = _read_json(derived / "derived_meta.json", "derived_meta.json")
    kind = meta.get("kind")
    reasons: list[str] = []
    alarms: list[str] = []
    if kind not in KINDS:
        raise InputError(f"derived_meta.json kind {kind!r} is none of {KINDS!r}")
    if not (derived / "object.pt").is_file():
        reasons.append("object.pt absent")
    obj_sha = meta.get("object_sha256")
    if (derived / "object.pt").is_file() and obj_sha and \
            _sha256(derived / "object.pt") != obj_sha:
        reasons.append("object.pt does not hash to derived_meta.object_sha256")

    # --- parents ----------------------------------------------------------------
    blocks = {"pos": meta.get("parent_pos") or {}, "neg": meta.get("parent_neg") or {}}
    parents = {"pos": _parent(Path(pos_dir), "pos", blocks["pos"], reasons, key=key),
               "neg": _parent(Path(neg_dir), "neg", blocks["neg"], reasons, key=key)}
    p, n = parents["pos"], parents["neg"]
    if p["seed"] is not None and n["seed"] is not None and p["seed"] != n["seed"]:
        reasons.append(f"parents' seeds differ: pos {p['seed']}, neg {n['seed']} — "
                       "a difference object needs SAME-seed parents")
    if seed is not None:
        for role, rec in parents.items():
            if rec["seed"] is not None and int(rec["seed"]) != int(seed):
                reasons.append(f"{role} parent seed {rec['seed']} != expected {seed}")
    dkey = meta.get("model_key")
    for role in ("pos", "neg"):
        bk = blocks[role].get("model_key")
        if bk is not None and dkey is not None and bk != dkey:
            reasons.append(f"{role} parent block model_key {bk!r} != derived "
                           f"model_key {dkey!r}")
    if key is not None and dkey is not None and dkey != key:
        reasons.append(f"derived model_key {dkey!r} != expected key {key!r}")
    if p["sidecars_ok"] and n["sidecars_ok"]:
        for f in PARENT_MATCH_FIELDS:
            if p["match_fields"].get(f) != n["match_fields"].get(f):
                reasons.append(f"parents disagree on {f}: pos "
                               f"{p['match_fields'].get(f)!r}, neg "
                               f"{n['match_fields'].get(f)!r} (require_parents_match)")

    # --- kind-specific ----------------------------------------------------------
    rec_verdict, live_min, live_layer, sweep_block = None, None, None, None
    bound: dict | None = None
    below_ref: list[str] | None = None
    dead: list[str] | None = None
    n_layers: int | None = None
    if kind == KIND_DISTILLED:
        rec_verdict = meta.get("recovery_verdict")
        report = _read_json(derived / "report.json", "report.json", required=False)
        rep_verdict = ((report or {}).get("recovery") or {}).get("verdict")
        if rec_verdict != "recovered":
            # recorded, not a refusal ground.
            alarms.append(f"recovery_verdict {rec_verdict!r} != 'recovered' (recorded; "
                          f"not a refusal ground since 2026-09-18, D69)")
        if report is None:
            reasons.append("report.json absent")
        elif rep_verdict != rec_verdict:
            reasons.append(f"report.json recovery.verdict {rep_verdict!r} != "
                           f"derived_meta recovery_verdict {rec_verdict!r}")
        live_min, live_layer, dead, below_ref, n_layers = _liveness_checks(
            derived, reasons, alarms, floor=floor, dead_floor=dead_floor)
        sweep_block = _sweep_and_assess(derived, meta, reasons, alarms)
    elif kind == KIND_PAIRED_WRITE:
        # recovery is stated NOT APPLICABLE with the reason, never omitted.
        rec_verdict = RECOVERY_NOT_APPLICABLE
        # the branch is BOUND to the bytes — the shared validator, the blob's own kind, and scale matching derived from
        # the embedded parent budgets — before anything the sidecar claims is believed
        bound = _bound_paired_write(derived, meta, reasons)
        # Condition 2: the same liveness judgement as the distilled object's — a dead layer refuses, the published
        # reference is a recorded alarm.
        live_min, live_layer, dead, below_ref, n_layers = _liveness_checks(
            derived, reasons, alarms, floor=floor, dead_floor=dead_floor)
    else:
        if meta.get("scale_matched") is not True:
            reasons.append(f"scale_matched {meta.get('scale_matched')!r} is not true")

    doc = {
        "kind": "seedrep_derived_admission",
        "derived_kind": kind,
        "derived_id": meta.get("derived_id"),
        "derived_dir": str(derived),
        "object_sha256": obj_sha,
        "key": dkey, "expected_key": key, "expected_seed": seed,
        "seed": p["seed"] if p["seed"] == n["seed"] else None,
        "admitted": not reasons,
        "reasons": reasons,
        "alarms": alarms,
        "parents": {r: {k: v for k, v in rec.items() if k != "match_fields"}
                    for r, rec in parents.items()},
        "recovery_verdict": rec_verdict,
        # "checked and passed" vs "does not apply" survive as different values (condition 1): a distilled object records
        # its verdict + applies True; a paired write records not_applicable + applies False + the reason; a subtraction None.
        "recovery_applicability": ({"applies": True, "reason": None} if kind == KIND_DISTILLED else
                                   {"applies": False, "reason": RECOVERY_NOT_APPLICABLE_REASON} if kind == KIND_PAIRED_WRITE
                                   else None),
        "gate_liveness_min": live_min,
        "gate_liveness_min_layer": live_layer,
        # ``floor`` is kept for readers of older records; it is now the
        # published-object REFERENCE (advisory), duplicated as liveness_reference.
        "floor": floor if kind in (KIND_DISTILLED, KIND_PAIRED_WRITE) else None,
        "liveness_reference": floor if kind in (KIND_DISTILLED, KIND_PAIRED_WRITE) else None,
        "layers_below_reference": below_ref,
        "n_layers_below_reference": len(below_ref) if below_ref is not None else None,
        "n_layers": n_layers,
        "dead_floor": dead_floor if kind in (KIND_DISTILLED, KIND_PAIRED_WRITE) else None,
        "dead_layers": dead,
        "admission_rule": (ADMISSION_RULE if kind == KIND_DISTILLED else
                           PAIRED_WRITE_RULE if kind == KIND_PAIRED_WRITE else None),
        "superseded_rule": SUPERSEDED_RULE if kind == KIND_DISTILLED else None,
        "scale_matched": (meta.get("scale_matched") if kind == KIND_SUBTRACTION else
                          bound["scale_matched_derived"] if kind == KIND_PAIRED_WRITE else None),
        "bytes_bound": bound if kind == KIND_PAIRED_WRITE else None,
        "selected_lr_mult": (sweep_block or {}).get("selected_lr_mult"),
        "sweep": (sweep_block or {}).get("sweep"),
        "assess": (sweep_block or {}).get("assess"),
        "parts_complete": (sweep_block or {}).get("parts_complete"),
        "checked_at": _now(),
        "checker": "scripts/sdfbase/seedrep/derived_admission.py",
    }
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("derived_dir")
    ap.add_argument("--parents", nargs=2, required=True,
                    metavar=("POS_CELL_DIR", "NEG_CELL_DIR"))
    ap.add_argument("--floor", type=float, default=DEFAULT_FLOOR,
                    help="gate-liveness REFERENCE (default: the published Qwen derived "
                         "object's own min over layers, DEFAULT_FLOOR); since D69 "
                         "(2026-09-18) recorded as an alarm, not a refusal ground")
    ap.add_argument("--dead-floor", type=float, default=DEFAULT_DEAD_FLOOR,
                    help="a layer whose live fraction is <= this is FULLY DEAD and refuses "
                         "the object (default 0.0) — the one liveness refusal ground (D69)")
    ap.add_argument("--key", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None,
                    help="default <derived_dir>/ADMISSION.json")
    a = ap.parse_args(argv)
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    try:
        doc = admission(a.derived_dir, pos_dir=a.parents[0], neg_dir=a.parents[1],
                        floor=a.floor, dead_floor=a.dead_floor, key=a.key, seed=a.seed)
    except InputError as e:
        print(f"[derived-admission] REFUSED: {e}", file=sys.stderr)
        return 3
    out = Path(a.out) if a.out else Path(a.derived_dir) / "ADMISSION.json"
    out.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n",
                   encoding="utf-8")
    state = "ADMITTED" if doc["admitted"] else "NOT ADMITTED"
    print(f"[derived-admission] {doc['derived_id']} ({doc['derived_kind']}): {state}"
          f" -> {out}")
    for r in doc["reasons"]:
        print(f"  - {r}")
    for al in doc["alarms"]:
        print(f"  ! ALARM: {al}")
    return 0 if doc["admitted"] else 4


if __name__ == "__main__":
    sys.exit(main())
