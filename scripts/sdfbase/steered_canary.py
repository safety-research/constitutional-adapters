#!/usr/bin/env python
"""The STEERED SERVING CANARY of an onboarded model (paper-companion-repo, 2026-09-30):
the first time a steering vector is served natively on a model, stand one ReLU object up as a
residual-slot carrier beside one certified LoRA adapter and `none`, read the residual slot's own
certificate off the AUTHORITATIVE records, then measure the engine's throughput three ways.

Three subcommands, each a step of scripts/sdfbase/steered_canary.sbatch:

  stage   — a CANARY VIEW (the explicit-cells layout defense_cell.registry_from_explicit_cells
            reads): per served cell `<view>/<cell>/object.pt` + `train_budget_meta.json` (the
            legacy identity schema: cell_id + object_sha256 bind the bytes) + `staged_from.json`
            (where the bytes came from, every provenance fact the trial recorded). The trial's own
            records are copied beside the view (`<view>/_provenance/<cell>/`), never into the cell
            dir (lora_export reads an `object.json` sidecar when present and refuses one on a
            residual kind; stray files must not become identity). Idempotent: a present object.pt
            with the same digest is kept, a different one refuses — nothing is ever overwritten.
            The object's stored kind must be the arm's (registry_adapter.ARM_OBJECT_KINDS) and its
            fit leg `all` (native multi-LoRA / the residual slot have no position mask).
  derive  — the canary's config + manifest pair from a roster view's pair: the SAME data except
            `explicit_cells` / `defenses_dir` / the run namespace (config) and `petri.cells` (the
            harness ids) / `petri.audit` (manifest). Written OUTSIDE the repo (the generated
            full22 pairs are byte-identity tested), with a derivation record naming both sides.
  run     — ONE native standup through lora_serving_up.bring_up_lora (the certification driver's
            seam) serving [relu cell, lora cell] (+ none, + the zero adapter) at the requested
            L x S; the attestation facts (residual_slot.armed, the worker's n_layers_hooked /
            hooked_boundaries over every block, the carrier's kind + rank), the six positive
            controls (the noop certificate and steering_bites in full), a live greedy completion
            per arm; then the load check three ways — requests naming the base model, the LoRA
            adapter, the residual carrier — at each B of `--batches`, `max_tokens == min_tokens`
            exact-length decodes through the proxy, tok/s from the ENGINE's generation counter
            (closed_loop.closed_loop_via_proxy, the gate's loop). A check that fails is recorded
            and the run exits non-zero; nothing is measured around.

The record `<out>/<target>.json` carries every fact; `checks` is the list a reader judges by.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import hashlib
import json
import os
import shutil
import signal
import socket
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
for _p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "sdfbase"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SCHEMA = "steered_canary/1"
TAG = "[steered_canary]"
EVAL_NAME = "families_served:short"
ARMS = ("base", "lora", "relu")
#: the §7.2 window shape (native_timing_72.PROTOCOL): 10 s warm-up, 45 s window, exact length
DEFAULT_WINDOW_S, DEFAULT_WARMUP_S = 45.0, 10.0
DEFAULT_BATCHES = (32, 128)
DEFAULT_MAX_TOKENS = 64
GEOMETRY_OVERRIDE_REASON = (
    "steered serving canary: the probe's certified short window (L 4096) at the requested width; the "
    "LoRA kernel index ceiling of the banked adapters lowers S when it binds, and the record says so")
BUDGET_META_NAME = "train_budget_meta.json"
STAGED_FROM_NAME = "staged_from.json"
PROVENANCE_DIR = "_provenance"
RC_REFUSED, RC_CHECKS_FAILED, RC_TERMINATED = 2, 3, 143


class CanaryRefused(RuntimeError):
    """A precondition the canary cannot certify — refused, never measured around."""


def log(msg: str) -> None:
    print(f"{TAG} {msg}", flush=True)


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json_atomic(path: Path, obj: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


# --------------------------------------------------------------------------- #
# pure helpers (unit-tested without a GPU)                                      #
# --------------------------------------------------------------------------- #
def window_plan(batches: Sequence[int], arms: Sequence[str] = ARMS) -> List[Tuple[int, str]]:
    """B outer, arm inner: the three arms at one B are ADJACENT windows of the same run."""
    return [(int(B), str(a)) for B in batches for a in arms]


def parse_batches(text: str) -> List[int]:
    out = [int(x) for x in str(text).split(",") if x.strip()]
    if not out or any(b < 1 for b in out):
        raise CanaryRefused(f"--batches needs positive integers, got {text!r}")
    return out


def arm_of_cell(cell_id: str) -> str:
    if "__" not in cell_id:
        raise CanaryRefused(f"cell {cell_id!r} is not a <si>__<arm> object cell")
    return cell_id.split("__", 1)[1]


def derive_pair(cfg: Mapping[str, Any], man: Mapping[str, Any], *, cells: Sequence[str],
                harness_cells: Sequence[str], defenses_dir: str, namespace: str
                ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """``(config, manifest, changes)``: the view pair's data with ONLY the keys the canary
    owns replaced — config `explicit_cells` / `defenses_dir` / `out_dir` / `private_run`,
    manifest `petri.cells` / `petri.audit`. Everything else is the source's, as data."""
    if list(cells)[:1] != ["none"]:
        raise CanaryRefused(f"the first cell must be 'none' (the same-wave bare anchor), got {list(cells)}")
    if len(cells) != len(harness_cells):
        raise CanaryRefused("cells and harness cells differ in length")
    if "explicit_cells" not in cfg or "petri" not in man or "cells" not in man["petri"]:
        raise CanaryRefused("the source pair is not a full22 explicit-cells config + petri manifest")
    # the canary MARKER rides first: common.sh's registry guard (f22_canary_identity) refuses a config carrying a top-level `canary:`
    # key by name, so a launcher that resolves THIS pair can never register its cells
    new_cfg = {"canary": {"harness": "scripts/sdfbase/steered_canary.py", "namespace": namespace,
                          "note": "a serving canary's derived pair — never a wave's; the results registry refuses this config's identity"}}
    new_cfg.update(json.loads(json.dumps(cfg)))
    new_man = json.loads(json.dumps(man))
    changes = {"config": {"canary": {"from": cfg.get("canary"), "to": new_cfg["canary"]}}, "manifest": {}}
    for key, value in (("explicit_cells", list(cells)), ("defenses_dir", str(defenses_dir)),
                       ("out_dir", f"out/{namespace}"), ("private_run", namespace)):
        changes["config"][key] = {"from": new_cfg.get(key), "to": value}
        new_cfg[key] = value
    for key, value in (("cells", list(harness_cells)), ("audit", namespace)):
        changes["manifest"][f"petri.{key}"] = {"from": new_man["petri"].get(key), "to": value}
        new_man["petri"][key] = value
    return new_cfg, new_man, changes


def attestation_checks(*, endpoint: Mapping[str, Any], attestation: Mapping[str, Any],
                       controls: Mapping[str, Any], relu_h: str, lora_h: str, n_layers: int,
                       relu_kind: str = "psr", cells_served: Optional[Mapping[str, Any]] = None
                       ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """The facts the canary claims, each read from the AUTHORITATIVE record it lives in and
    judged ONCE here: ``(checks, facts)`` — a check is ``{"name", "ok", "detail"}``."""
    checks: List[Dict[str, Any]] = []
    facts: Dict[str, Any] = {}

    def check(name: str, ok: bool, detail: Any) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    ep = endpoint
    facts["engine"] = {"engine": ep.get("engine"), "serving_engine": ep.get("serving_engine"),
                       "attested_enforce_eager": ep.get("attested_enforce_eager"),
                       "attested_cudagraph_mode": ep.get("attested_cudagraph_mode")}
    check("engine_lora_native", ep.get("engine") == "lora_native" and ep.get("serving_engine") == "lora_native", facts["engine"])
    check("graphs_on", ep.get("attested_enforce_eager") is False and ep.get("attested_cudagraph_mode") not in (None, "NONE"), facts["engine"])
    facts["geometry"] = {"L": ep.get("max_model_len"), "S": ep.get("max_num_seqs"), "attested_S": ep.get("attested_max_num_seqs"),
                         "adjustments": (ep.get("geometry") or {}).get("adjustments")}
    facts["lora_cells"] = ep.get("lora_cells")
    facts["cells_skipped"] = ep.get("cells_skipped")
    check("no_cell_skipped", not ep.get("cells_skipped"), ep.get("cells_skipped"))
    check("both_steered_cells_served", set(ep.get("lora_cells") or []) == {relu_h, lora_h}, ep.get("lora_cells"))
    # the gate's record rides the standup's records (endpoint.json / cells_served.json share the block)
    gate = ep.get("adapter_parity_gate") or (cells_served or {}).get("adapter_parity_gate") or {}
    facts["adapter_parity_gate"] = gate
    residual_block = gate.get("residual") or {}
    carriers = [c.get("adapter") for c in (residual_block.get("carriers") or [])]
    check("parity_gate_lists_the_carrier", relu_h in carriers and gate.get("status") in ("verified", "residual_only"),
          {"status": gate.get("status"), "carriers": carriers,
           "pairs": [(p.get("adapter"), p.get("rank"), p.get("set"), p.get("status")) for p in (gate.get("pairs") or [])]})
    # the residual slot: endpoint block, attestation block, the worker's report
    rs = ep.get("residual_slot") or {}
    facts["endpoint_residual_slot"] = rs
    check("endpoint_residual_slot_armed_for_the_carrier", rs.get("armed") is True and set(rs.get("cells") or {}) == {relu_h}, rs)
    ad = ep.get("adapters") or {}
    relu_block, lora_block = ad.get(relu_h) or {}, ad.get(lora_h) or {}
    facts["adapter_blocks"] = {c: {k: (b or {}).get(k) for k in ("name", "r", "petri_source_kind", "petri_zero_adapter", "target_modules")}
                               for c, b in ad.items()}
    check("carrier_is_a_rank1_residual_entry", relu_block.get("petri_source_kind") == relu_kind and relu_block.get("r") == 1,
          facts["adapter_blocks"].get(relu_h))
    check("lora_block_is_a_plain_adapter", lora_block.get("petri_source_kind") in ("lora", "lora_all") and lora_block.get("petri_zero_adapter") in (None, False),
          facts["adapter_blocks"].get(lora_h))
    blk = attestation.get("residual_slot") or {}
    facts["attested_residual_slot"] = {k: blk.get(k) for k in ("armed", "rpc_method", "n_workers")}
    facts["attested_residual_expected"] = {k: v for k, v in (blk.get("expected") or {}).items() if k != "adapters"}
    check("attestation_residual_slot_armed", blk.get("armed") is True, facts["attested_residual_slot"])
    w0 = (blk.get("worker_reports") or [{}])[0]
    hooked = w0.get("hooked_boundaries")
    facts["worker_residual"] = {k: w0.get(k) for k in ("armed", "op_name", "op_registered", "runner_bound", "n_layers_hooked",
                                                        "n_layers_model", "hooked_boundaries", "ledger_size", "ledger_fault", "error")}
    facts["patched_classes"] = w0.get("patched_classes")
    facts["worker_bank"] = w0.get("bank")
    check("worker_hooked_every_block",
          w0.get("n_layers_hooked") == n_layers and w0.get("n_layers_model") == n_layers
          and isinstance(hooked, list) and sorted(int(x) for x in hooked) == list(range(n_layers)),
          {"n_layers_hooked": w0.get("n_layers_hooked"), "n_layers_model": w0.get("n_layers_model"),
           "n_hooked_boundaries": (len(hooked) if isinstance(hooked, list) else None), "expected": n_layers})
    check("worker_op_registered_and_wrapped",
          bool(w0.get("op_registered")) and any(e.get("forward_wrapped") for e in (w0.get("patched_classes") or {}).values()),
          {"op_registered": w0.get("op_registered"), "patched_classes": list((w0.get("patched_classes") or {}))})
    check("worker_op_ran", int((w0.get("bank") or {}).get("op_invocations") or 0) > 0, (w0.get("bank") or {}).get("op_invocations"))
    check("worker_ledger_clean", not w0.get("ledger_fault") and not w0.get("ledger_size"),
          {"ledger_size": w0.get("ledger_size"), "ledger_fault": w0.get("ledger_fault")})
    relu_att = (attestation.get("adapters") or {}).get(relu_block.get("name")) or {}
    facts["carrier_attested"] = {k: relu_att.get(k) for k in ("petri_source_kind", "residual_fingerprint", "position_mode", "r", "lora_int_id")}
    check("carrier_attested_residual_all_positions",
          relu_att.get("residual_fingerprint") is not None and relu_att.get("position_mode") == "all" and relu_att.get("petri_source_kind") == relu_kind,
          facts["carrier_attested"])
    slots = w0.get("slots") or {}
    sid = str(relu_att.get("lora_int_id"))
    check("worker_slot_fingerprint_matches", sid in slots and slots[sid].get("fingerprint") == relu_att.get("residual_fingerprint"),
          {"lora_int_id": sid, "slots": list(slots)})
    # the six positive controls, in full for the two the canary is about
    pc = controls
    facts["controls"] = {n: {"passed": c.get("passed"), "verdict": c.get("verdict")} for n, c in (pc.get("controls") or {}).items()}
    facts["controls_order"] = pc.get("order")
    facts["noop_certificate"] = (pc.get("controls") or {}).get("noop_certificate")
    facts["steering_bites"] = (pc.get("controls") or {}).get("steering_bites")
    check("positive_controls_passed", pc.get("passed") is True,
          {"passed": pc.get("passed"), "failed_control": pc.get("failed_control"), "error": str(pc.get("error"))[:400] if pc.get("error") else None,
           "controls": facts["controls"]})
    noop = facts["noop_certificate"] or {}
    check("noop_certificate_passed", noop.get("passed") is True, {k: noop.get(k) for k in ("passed", "verdict")})
    bites = facts["steering_bites"] or {}
    relu_bite = (bites.get("cells") or {}).get(relu_h) or {}
    check("steering_bites_on_the_carrier", bool(relu_bite) and bool(relu_bite.get("n_differing")), relu_bite or bites.get("verdict"))
    return checks, facts


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """``{"<arm>@B<B>": {"tok_s", "running_mean", "errors"}}`` — the headline per window."""
    return {f"{r['arm']}@B{r['B']}": {"tok_s": r.get("tok_s"), "client_tok_s": r.get("client_tok_s"),
                                      "running_mean": r.get("running_mean"), "running_max": r.get("running_max"),
                                      "waiting_max": r.get("waiting_max"), "requests_done": r.get("requests_done"),
                                      "errors": r.get("errors")} for r in rows}


def rows_problems(rows: Sequence[Mapping[str, Any]], plan: Sequence[Tuple[int, str]]) -> List[str]:
    """Every planned window present, error-free, with an engine tok/s reading."""
    problems: List[str] = []
    have = {(int(r["B"]), r["arm"]): r for r in rows}
    for B, arm in plan:
        r = have.get((B, arm))
        if r is None:
            problems.append(f"window {arm}@B{B} missing")
            continue
        if r.get("errors"):
            problems.append(f"window {arm}@B{B}: {r['errors']} request errors ({r.get('error_sample')})")
        if r.get("tok_s") is None:
            problems.append(f"window {arm}@B{B}: no engine tok/s (counter unreadable)")
        if not r.get("requests_done"):
            problems.append(f"window {arm}@B{B}: no request completed inside the window")
    return problems


# --------------------------------------------------------------------------- #
# stage                                                                          #
# --------------------------------------------------------------------------- #
def _load_json(p: Path) -> Dict[str, Any]:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def parse_pin(text: str) -> Dict[str, str]:
    """``<object_sha256>:<fit_fingerprint>:<search_id>/<trial_id>`` — the FROZEN HPO CHOICE, independent of the
    trial path the operator typed: the digest obtained elsewhere, and the registry's fit_trial
    record's fingerprint + ids. Every field is required."""
    parts = str(text).split(":")
    if len(parts) < 3:
        raise CanaryRefused(f"pin {text!r} is not <object_sha256>:<fit_fingerprint>:<search_id>/<trial_id>")
    sha, fp, rest = parts[0].strip(), parts[1].strip(), ":".join(parts[2:]).strip()
    if "/" not in rest:
        raise CanaryRefused(f"pin {text!r}: the third field must be <search_id>/<trial_id>")
    search_id, trial_id = rest.rsplit("/", 1)
    if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise CanaryRefused(f"pin {text!r}: object_sha256 must be 64 hex chars")
    if len(fp) != 64 or any(c not in "0123456789abcdef" for c in fp):
        raise CanaryRefused(f"pin {text!r}: fit_fingerprint must be 64 hex chars")
    if not search_id or not trial_id:
        raise CanaryRefused(f"pin {text!r}: search_id and trial_id must be non-empty")
    return {"object_sha256": sha, "fit_fingerprint": fp, "search_id": search_id, "trial_id": trial_id}


def provenance_problems(summary: Mapping[str, Any], trial_rec: Mapping[str, Any], *, pin: Mapping[str, str],
                        object_sha256: str, arm: str, fit_model: Optional[str], fit_revision: Optional[str]) -> List[str]:
    """Why a trial directory is NOT the frozen HPO choice ``pin`` names (empty = it is): the bytes' digest, the
    fit fingerprint (summary AND fit_trial_record, which must agree), the search and trial ids, the arm, and the
    fit checkpoint (``fit_model`` / ``fit_revision``, the front door's models.yaml) are each held to the pin."""
    p: List[str] = []
    if object_sha256 != pin["object_sha256"]:
        p.append(f"object.pt sha256 {object_sha256[:12]} != the pin's {pin['object_sha256'][:12]}")
    for label, rec in (("summary.json", summary), ("fit_trial_record.json", trial_rec)):
        if rec.get("fit_fingerprint") != pin["fit_fingerprint"]:
            p.append(f"{label} fit_fingerprint {str(rec.get('fit_fingerprint'))[:12]} != the pin's {pin['fit_fingerprint'][:12]}")
        if rec.get("search_id") != pin["search_id"]:
            p.append(f"{label} search_id {rec.get('search_id')!r} != the pin's {pin['search_id']!r}")
    if trial_rec.get("trial_id") != pin["trial_id"]:
        p.append(f"fit_trial_record.json trial_id {trial_rec.get('trial_id')!r} != the pin's {pin['trial_id']!r}")
    if trial_rec.get("status") not in (None, "complete"):
        p.append(f"fit_trial_record.json status {trial_rec.get('status')!r} is not complete")
    if summary.get("arm") != arm:
        p.append(f"summary.json arm {summary.get('arm')!r} != the cell's {arm!r}")
    if fit_model is not None and summary.get("model") != fit_model:
        p.append(f"summary.json model {summary.get('model')!r} != the fit checkpoint {fit_model!r}")
    if fit_revision is not None and summary.get("model_revision") != fit_revision:
        p.append(f"summary.json model_revision {str(summary.get('model_revision'))[:12]} != the fit checkpoint's {fit_revision[:12]}")
    return p


def stage_cell(view_dir: Path, cell_id: str, src_dir: Path, *, target: str, completion_tokens: int, pin: Mapping[str, str],
               fit_model: Optional[str] = None, fit_revision: Optional[str] = None,
               expected_n_layers: Optional[int] = None, load_blob: Optional[Any] = None) -> Dict[str, Any]:
    """Stage ONE trial directory (object.pt + summary.json + fit_trial_record.json) as the view cell
    ``cell_id`` — only when it IS the frozen HPO choice ``pin`` names (:func:`provenance_problems`, before any
    byte is copied). Returns the staged_from record. ``load_blob`` (tests) replaces the torch load."""
    from sisteer.capeval.petri import registry_adapter as ra

    src_dir, view_dir = Path(src_dir), Path(view_dir)
    obj = src_dir / "object.pt"
    if not obj.is_file():
        raise CanaryRefused(f"{src_dir} has no object.pt")
    for name in ("summary.json", "fit_trial_record.json"):
        if not (src_dir / name).is_file():
            raise CanaryRefused(f"{src_dir} has no {name} — a trial without its provenance record is never staged")
    arm = arm_of_cell(cell_id)
    summary0 = _load_json(src_dir / "summary.json")
    trial0 = _load_json(src_dir / "fit_trial_record.json")
    sha0 = sha256_file(obj)
    problems = provenance_problems(summary0, trial0, pin=pin, object_sha256=sha0, arm=arm, fit_model=fit_model, fit_revision=fit_revision)
    if problems:
        raise CanaryRefused(f"{src_dir} is not the frozen HPO choice {pin['search_id']}/{pin['trial_id']}: " + "; ".join(problems))
    want_kind = ra.ARM_OBJECT_KINDS.get(arm)
    if want_kind is None:
        raise CanaryRefused(f"arm {arm!r} of {cell_id!r} has no object kind in registry_adapter.ARM_OBJECT_KINDS")
    if load_blob is None:
        import torch

        def load_blob(p: Path) -> Any:  # noqa: E306
            return torch.load(p, map_location="cpu", weights_only=False)
    blob = load_blob(obj)
    if not isinstance(blob, dict):
        raise CanaryRefused(f"{obj} is not a dict blob ({type(blob).__name__})")
    if blob.get("kind") != want_kind:
        raise CanaryRefused(f"{obj} stores kind {blob.get('kind')!r}; the arm {arm!r} of {cell_id!r} serves kind {want_kind!r}")
    if blob.get("positions") != "all":
        raise CanaryRefused(f"{obj} was fit at positions={blob.get('positions')!r}; the native paths serve all-position fits only")
    layers = list(blob.get("layers") or [])
    if expected_n_layers is not None and sorted(int(x) for x in layers) != list(range(int(expected_n_layers))):
        raise CanaryRefused(f"{obj} covers layers {layers[:3]}..{layers[-3:]} ({len(layers)}), not every one of {expected_n_layers} blocks")
    sha, summary, trial_rec = sha0, summary0, trial0
    fit_si = summary.get("si")
    cell_si = cell_id.split("__", 1)[0]
    cell_dir = view_dir / cell_id
    cell_dir.mkdir(parents=True, exist_ok=True)
    dst = cell_dir / "object.pt"
    if dst.is_file():
        have = sha256_file(dst)
        if have != sha:
            raise CanaryRefused(f"{dst} exists with sha256 {have[:12]} != the source's {sha[:12]} — refusing to overwrite a staged object")
    else:
        tmp = cell_dir / "object.pt.tmp"
        shutil.copyfile(obj, tmp)
        if sha256_file(tmp) != sha:
            tmp.unlink(missing_ok=True)
            raise CanaryRefused(f"copy of {obj} did not reproduce its digest")
        os.replace(tmp, dst)
    for stray in ("object.json",):
        if (cell_dir / stray).exists():
            raise CanaryRefused(f"{cell_dir / stray} exists — an `object.json` sidecar enters the export's identity; the canary view carries none")
    provenance = {
        "fit_si": fit_si, "served_cell_si": cell_si,
        "relabel": (None if fit_si in (None, cell_si) else
                    f"the object was fit at SI {fit_si!r} (an HPO trial budget); it is served under the cell id {cell_id!r} because "
                    f"the harness vocabulary (registry_adapter.HARNESS_TO_TRAINTEST_SI) has no entry for {fit_si!r} — a SERVING label "
                    "for the canary only, never a result row; the bytes are pinned by object_sha256"),
        "pin": dict(pin), "fit_checkpoint_expected": {"model": fit_model, "revision": fit_revision},
        "trial_id": trial_rec.get("trial_id"), "search_id": summary.get("search_id") or trial_rec.get("search_id"),
        "fit_fingerprint": summary.get("fit_fingerprint") or trial_rec.get("fit_fingerprint"),
        "record_path": trial_rec.get("record_path"), "campaign": trial_rec.get("campaign"),
        "arm": summary.get("arm"), "round": summary.get("round"), "trial": summary.get("trial"),
        "fit_model": summary.get("model"), "fit_model_revision": summary.get("model_revision"),
        "template_hash": summary.get("template_hash"), "writer": summary.get("writer"), "position": summary.get("position"),
        "object_desc": summary.get("object_desc"), "best_heldout_loss": summary.get("best_heldout_loss"),
        "floor_ratio": summary.get("floor_ratio"), "floor_ok": summary.get("floor_ok"),
        "object_kind": blob.get("kind"), "object_layers": layers, "object_positions": blob.get("positions"),
    }
    meta = {"cell_id": cell_id, "object_sha256": sha, "completion_tokens": int(completion_tokens),
            "note": ("steered canary staging (scripts/sdfbase/steered_canary.py stage): the legacy identity schema — cell_id + "
                     "object_sha256 bind the bytes; completion_tokens is the view config's cap, carried because the schema requires "
                     f"it; the fit's own budget record is summary.json under {PROVENANCE_DIR}/{cell_id}/"),
            "canary": provenance}
    write_json_atomic(cell_dir / BUDGET_META_NAME, meta)
    prov_dir = view_dir / PROVENANCE_DIR / cell_id
    prov_dir.mkdir(parents=True, exist_ok=True)
    copied = {}
    for name in ("summary.json", "fit_trial_record.json", "object.pt.provenance.json"):
        p = src_dir / name
        if p.is_file():
            shutil.copyfile(p, prov_dir / name)
            copied[name] = sha256_file(p)
    staged_shas = {"object.pt": sha, BUDGET_META_NAME: sha256_file(cell_dir / BUDGET_META_NAME)}
    rec = {"target": target, "lane": "canary", "cell_id": cell_id, "staged_id": f"canary__{src_dir.name}",
           "source": str(src_dir), "pulled_at": utc_now(), "object_sha256": sha, "staged_shas": staged_shas,
           "provenance_copies": {f"{PROVENANCE_DIR}/{cell_id}/{k}": v for k, v in copied.items()},
           "staged_by": "scripts/sdfbase/steered_canary.py stage", "canary": provenance}
    write_json_atomic(cell_dir / STAGED_FROM_NAME, rec)
    # the registry's own verdict on the staged dir (CPU, model-free): refuse here, never at the standup
    from sisteer.eval.train_budget import staged_cell_problems
    problems = staged_cell_problems(dst, cell_id, require_sidecar=True)
    if problems:
        raise CanaryRefused(f"staged cell {cell_id!r} does not validate: " + "; ".join(problems))
    return rec


def cmd_stage(args: argparse.Namespace) -> int:
    view_dir = Path(args.view_dir)
    pins: Dict[str, Dict[str, str]] = {}
    for spec in args.pin:
        if "=" not in spec:
            raise CanaryRefused(f"--pin needs CELL=<sha256>:<fit_fingerprint>:<search_id>/<trial_id>, got {spec!r}")
        cell, text = spec.split("=", 1)
        pins[cell.strip()] = parse_pin(text)
    records = []
    for spec in args.cell:
        if "=" not in spec:
            raise CanaryRefused(f"--cell needs CELL=SRC_DIR, got {spec!r}")
        cell, src = spec.split("=", 1)
        cell = cell.strip()
        if cell not in pins:
            raise CanaryRefused(f"cell {cell!r} has no --pin (the frozen HPO choice is required for every staged cell)")
        rec = stage_cell(view_dir, cell, Path(src), target=args.target, completion_tokens=args.completion_tokens, pin=pins[cell],
                         fit_model=args.expect_fit_model, fit_revision=args.expect_fit_revision, expected_n_layers=args.expect_n_layers)
        log(f"staged {cell} <- {src} (object.pt sha256 {rec['object_sha256'][:12]}; fit SI {rec['canary'].get('fit_si')!r})")
        records.append(rec)
    write_json_atomic(view_dir / "CANARY_VIEW.json", {"schema": SCHEMA, "target": args.target, "written_at": utc_now(),
                                                        "cells": records, "staged_by": "scripts/sdfbase/steered_canary.py stage"})
    return 0


# --------------------------------------------------------------------------- #
# derive                                                                         #
# --------------------------------------------------------------------------- #
def cmd_derive(args: argparse.Namespace) -> int:
    import yaml
    from sisteer.capeval.petri import registry_adapter as ra

    cfg_src, man_src = Path(args.from_config), Path(args.from_manifest)
    cfg = yaml.safe_load(cfg_src.read_text(encoding="utf-8"))
    man = yaml.safe_load(man_src.read_text(encoding="utf-8"))
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    harness = [ra.harness_cell_id(c) for c in cells]
    namespace = f"steered_canary_{args.target}"
    new_cfg, new_man, changes = derive_pair(cfg, man, cells=cells, harness_cells=harness,
                                            defenses_dir=str(Path(args.defenses_dir).resolve()), namespace=namespace)
    header = (f"# DERIVED by scripts/sdfbase/steered_canary.py derive at {utc_now()} from {{src}} — the same data except the keys "
              "the canary owns ({keys}); a serving canary's pair, never a wave's (see derivation.json beside it)\n")
    out_cfg, out_man = Path(args.out_config), Path(args.out_manifest)
    out_cfg.parent.mkdir(parents=True, exist_ok=True); out_man.parent.mkdir(parents=True, exist_ok=True)
    out_cfg.write_text(header.format(src=cfg_src, keys=", ".join(changes["config"])) + yaml.safe_dump(new_cfg, sort_keys=False), encoding="utf-8")
    out_man.write_text(header.format(src=man_src, keys=", ".join(changes["manifest"])) + yaml.safe_dump(new_man, sort_keys=False), encoding="utf-8")
    rec = {"schema": SCHEMA, "written_at": utc_now(), "target": args.target, "cells": cells, "harness_cells": harness,
           "source": {"config": str(cfg_src), "config_sha256": sha256_file(cfg_src), "manifest": str(man_src), "manifest_sha256": sha256_file(man_src)},
           "derived": {"config": str(out_cfg), "config_sha256": sha256_file(out_cfg), "manifest": str(out_man), "manifest_sha256": sha256_file(out_man)},
           "changes": changes}
    write_json_atomic(out_cfg.parent / "derivation.json", rec)
    log(f"derived {out_cfg} and {out_man}: cells {cells} -> harness {harness}")
    return 0


# --------------------------------------------------------------------------- #
# run                                                                            #
# --------------------------------------------------------------------------- #
class _Terminated(Exception):
    pass


def _on_term(signum: int, _frame: Any) -> None:
    raise _Terminated(f"signal {signum}")


def live_completion(proxy_url: str, api_key: str, vmodel: str, *, max_tokens: int = 48, timeout_s: float = 300.0) -> Dict[str, Any]:
    body = json.dumps({"model": vmodel, "messages": [{"role": "user", "content": "Give one sentence of advice about learning to program."}],
                       "max_tokens": int(max_tokens), "temperature": 0.0, "seed": 0}).encode()
    req = urllib.request.Request(proxy_url.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
    resp = json.load(urllib.request.urlopen(req, timeout=timeout_s))
    return {"vmodel": vmodel, "text": resp["choices"][0]["message"]["content"][:400],
            "usage": resp.get("usage"), "petri": {k: (resp.get("petri") or {}).get(k) for k in ("virtual_model", "cell_id", "engine")}}


def cmd_run(args: argparse.Namespace) -> int:
    import lora_serving_up as lsu  # noqa: E402  (the standup seam; the serving venv)
    import native_timing_72 as nt  # noqa: E402  (the §7.2 helpers: tokenizer count, exact-length probe)
    from sisteer.capeval.petri import registry_adapter as ra
    from sisteer.capeval.serving import closed_loop as cl
    from sisteer.capeval.serving import proxy as px
    from sisteer.capeval.serving import serving_attestation as sa
    from sisteer.capeval.serving.run_identity import serving_versions, tree_sha

    repo = Path(args.repo_dir).resolve()
    cfg, man = Path(args.config), Path(args.manifest)
    for p in (cfg, man, Path(args.defenses_dir)):
        if not p.exists():
            raise CanaryRefused(f"missing {p}")
    out = Path(args.out).resolve()
    (out / "windows").mkdir(parents=True, exist_ok=True)
    batches = parse_batches(args.batches)
    plan = window_plan(batches)
    relu_h, lora_h = ra.harness_cell_id(args.relu_cell), ra.harness_cell_id(args.lora_cell)
    node, sha, started = socket.gethostname(), tree_sha(repo), time.monotonic()
    record: Dict[str, Any] = {
        "schema": SCHEMA, "target": args.target, "eval": EVAL_NAME, "status": "starting", "date": utc_now(), "node": node, "sha": sha,
        "cells": {"relu": args.relu_cell, "lora": args.lora_cell}, "harness_cells": {"relu": relu_h, "lora": lora_h},
        "config": str(cfg), "manifest": str(man), "defenses_dir": str(args.defenses_dir), "out": str(out), "standup_out": str(out / "standup"),
        "protocol": {"L": int(args.max_model_len), "S_requested": int(args.max_num_seqs), "batches": batches, "arms": list(ARMS),
                     "max_tokens": int(args.max_tokens), "min_tokens": int(args.max_tokens), "exact_length": True,
                     "window_s": float(args.window_s), "warmup_s": float(args.warmup_s), "prompt_tokens": int(args.prompt_tokens),
                     "windows_order": [f"{a}@B{B}" for B, a in plan]},
        "harness_env": {"SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID"), "SLURM_JOB_NAME": os.environ.get("SLURM_JOB_NAME")},
        "checks": [], "rows": [],
    }
    target_json = out / f"{args.target}.json"
    write_json_atomic(target_json, record)
    kw: Dict[str, Any] = dict(
        repo_dir=str(repo), config=str(cfg), manifest=str(man), defenses_dir=str(args.defenses_dir), out=str(out / "standup"),
        model_key=str(args.target), eval=EVAL_NAME, cell=[args.relu_cell, args.lora_cell],
        expect_object_sha256=[str(d) for d in args.expect_object_sha256],
        max_model_len=int(args.max_model_len), max_num_seqs=int(args.max_num_seqs), geometry_override_reason=GEOMETRY_OVERRIDE_REASON,
        allow_over_budget=bool(args.allow_over_budget), gpu_mem_util=str(args.gpu_mem_util),
        ready_timeout_s=int(args.ready_timeout_s), probe_max_tokens=int(args.probe_max_tokens),
    )
    if args.adapter_bank:
        kw["adapter_bank"] = str(args.adapter_bank)
    record["standup_kwargs"] = dict(kw)
    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)
    su: Any = None
    rc = 1
    try:
        log(f"native standup: {args.target} cells=[{args.relu_cell}, {args.lora_cell}] at L {args.max_model_len} x S {args.max_num_seqs} (eval {EVAL_NAME})")
        t0 = time.monotonic()
        su = lsu.bring_up_lora(**kw)
        record["standup_ready_s"] = time.monotonic() - t0
        endpoint = json.loads(Path(su.endpoint_path).read_text())
        attestation = sa.read_serving_attestation(Path(su.attestation_path))
        controls = json.loads(Path(endpoint["positive_controls"]).read_text()) if endpoint.get("positive_controls") else {}
        cells_served = json.loads(Path(endpoint["cells_served"]).read_text()) if endpoint.get("cells_served") else {}
        record["endpoint"] = endpoint
        record["versions"] = serving_versions()
        record["kernel_ceiling"] = getattr(su, "kernel_ceiling", None)
        record["geometry_lowered"] = getattr(su, "geometry_lowered", None)
        record["geometry_adjustments"] = list(getattr(su, "geometry_adjustments", ()) or ())
        checks, facts = attestation_checks(endpoint=endpoint, attestation=attestation, controls=controls,
                                           relu_h=relu_h, lora_h=lora_h, n_layers=int(args.expect_n_layers),
                                           cells_served=cells_served)
        record["checks"], record["facts"] = checks, facts
        write_json_atomic(target_json, record)
        failed = [c["name"] for c in checks if not c["ok"]]
        log(f"standup ready in {record['standup_ready_s']:.0f}s; attestation checks: {len(checks) - len(failed)}/{len(checks)} ok"
            + (f" — FAILED {failed}" if failed else ""))
        vmodels = {"base": su.base_vmodel, "lora": su.vmodels[lora_h], "relu": su.vmodels[relu_h]}
        record["vmodels"] = dict(vmodels, zero=su.zero_vmodel)
        api_key = px.proxy_client_token(endpoint)
        # a live greedy completion per arm (a smoke of the served route beyond the controls' probes)
        live = {a: live_completion(su.proxy_url, api_key, vm) for a, vm in vmodels.items()}
        live["relu_differs_from_base"] = live["relu"]["text"] != live["base"]["text"]
        live["lora_differs_from_base"] = live["lora"]["text"] != live["base"]["text"]
        record["live"] = live
        write_json_atomic(target_json, record)
        # the timing prompt through the SERVED tokenizer's render; exact length certified on the base first
        prompt = cl.build_timing_prompt(nt.tokenizer_count_fn(su.tokenizer), target_tokens=int(args.prompt_tokens))
        prompt["sha256"] = hashlib.sha256(prompt["text"].encode()).hexdigest()
        usage = nt.exact_length_probe(su.proxy_url, api_key, vmodels["base"], prompt["text"], max_tokens=int(args.max_tokens))
        prompt["served_prompt_tokens"] = usage.get("prompt_tokens")
        prompt["exact_length_probe_usage"] = usage
        record["prompt"] = {k: v for k, v in prompt.items() if k != "text"}
        (out / "timing_prompt.txt").write_text(prompt["text"], encoding="utf-8")
        log(f"timing prompt: {prompt['tokens']} tokens by the served tokenizer, {usage.get('prompt_tokens')} served; "
            f"exact length {usage.get('completion_tokens')} == {args.max_tokens}")
        metrics_url = f"{su.launch.base_url}/metrics"
        headers = dict(su.launch.auth_headers)
        record["status"] = "partial"
        rows: List[Dict[str, Any]] = []
        for B, arm in plan:
            vm = vmodels[arm]
            log(f"window {arm}@B{B}: {vm} — warm-up {args.warmup_s}s, window {args.window_s}s, max_tokens = min_tokens = {args.max_tokens}")
            sampler = cl.StepCounterSampler(metrics_url, headers, sample_s=1.0).start()
            with cl.prompt_bound(prompt["text"]):
                stats, gauges = asyncio.run(cl.closed_loop_via_proxy(
                    su.proxy_url, api_key, vm, metrics_url=metrics_url, metrics_headers=headers, B=int(B),
                    max_tokens=int(args.max_tokens), warmup_s=float(args.warmup_s), window_s=float(args.window_s), exact_length=True))
            samples = sampler.stop()
            counter = cl.counter_span_over_gauges(samples, gauges)
            write_json_atomic(out / "windows" / f"gauges_{arm}_B{B}.json", {"gauges": gauges, "counter_samples": samples, "stats": stats})
            ms = cl.ms_per_step_all(stats, int(B), counter_steps=counter.get("steps"), counter_span_s=counter.get("span_s"))
            row = {"arm": arm, "B": int(B), "vmodel": vm, "harness_cell": {"base": "none", "lora": lora_h, "relu": relu_h}[arm],
                   "tok_s": stats.get("server_tok_s"),
                   "tok_s_source": ("engine vllm:generation_tokens_total" if stats.get("server_tok_s") is not None else "UNREADABLE (engine counter missing)"),
                   "client_tok_s": stats.get("client_tok_s"), "running_mean": stats.get("running_mean"), "running_max": stats.get("running_max"),
                   "waiting_max": stats.get("waiting_max"), "requests_done": stats.get("requests_done"), "errors": stats.get("errors"),
                   "error_sample": stats.get("error_sample"), "latency_p50_s": stats.get("latency_p50_s"), "latency_p95_s": stats.get("latency_p95_s"),
                   "server_tokens": stats.get("server_tokens"), "client_tokens": stats.get("client_tokens"), "measured_window_s": stats.get("window_s"),
                   "ms_per_step": ms, "counter": dict(counter), "geometry": {"L": int(args.max_model_len), "S_attested": endpoint.get("attested_max_num_seqs")},
                   "max_tokens": int(args.max_tokens), "window_s": float(args.window_s), "warmup_s": float(args.warmup_s), "date": utc_now()}
            rows.append(row)
            record["rows"] = rows
            record["summary"] = summarize_rows(rows)
            write_json_atomic(target_json, record)
            log(f"  {arm}@B{B}: engine tok/s {row['tok_s'] and round(row['tok_s'], 1)} (client {row['client_tok_s'] and round(row['client_tok_s'], 1)}); "
                f"running mean {row['running_mean'] and round(row['running_mean'], 1)} max {row['running_max']} waiting max {row['waiting_max']}; "
                f"requests {row['requests_done']}; errors {row['errors']}; ms/step gate {ms['gate'] and round(ms['gate'], 2)}")
        problems = rows_problems(rows, plan)
        record["rows_problems"] = problems
        record["status"] = "complete"
        record["elapsed_s"] = time.monotonic() - started
        write_json_atomic(target_json, record)
        if failed or problems:
            rc = RC_CHECKS_FAILED
            log(f"complete with FAILED checks (rc {rc}): checks {failed}; windows {problems} -> {target_json}")
        else:
            rc = 0
            log(f"complete: every check ok, {len(rows)} windows -> {target_json}")
    except _Terminated as e:
        record["status"], record["terminated"] = "partial", str(e)
        write_json_atomic(target_json, record)
        log(f"terminated ({e}); partial record written")
        rc = RC_TERMINATED
    except (CanaryRefused, lsu.LoraStandupError) as e:
        record["status"], record["refused"] = "refused", f"{type(e).__name__}: {e}"
        write_json_atomic(target_json, record)
        log(f"REFUSED: {type(e).__name__}: {e}")
        rc = RC_REFUSED
    finally:
        if su is not None:
            try:
                su.tear_down()
                log("standup torn down")
            except Exception as e:  # noqa: BLE001
                log(f"tear_down raised {type(e).__name__}: {e}")
    return rc


# --------------------------------------------------------------------------- #
# CLI                                                                            #
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage", help="stage trial objects as a canary view")
    s.add_argument("--target", required=True)
    s.add_argument("--view-dir", required=True, type=Path)
    s.add_argument("--cell", action="append", default=[], metavar="CELL=SRC_DIR", help="a served cell and the trial dir holding its object.pt")
    s.add_argument("--pin", action="append", default=[], metavar="CELL=SHA256:FIT_FINGERPRINT:SEARCH_ID/TRIAL_ID",
                   help="the frozen HPO choice per cell, independent of the path: the object digest obtained elsewhere and the registry's fit_trial fingerprint + ids")
    s.add_argument("--expect-fit-model", default=None, help="the fit checkpoint's hf_repo (reproduce/models.yaml <key>.fit.checkpoint)")
    s.add_argument("--expect-fit-revision", default=None, help="the fit checkpoint's revision")
    s.add_argument("--completion-tokens", type=int, required=True, help="the view config's completion cap (the legacy sidecar schema's field)")
    s.add_argument("--expect-n-layers", type=int, default=None, help="every object must cover exactly these blocks (0..n-1)")
    s.set_defaults(fn=cmd_stage)
    d = sub.add_parser("derive", help="derive the canary config + manifest from a roster view's pair")
    d.add_argument("--target", required=True)
    d.add_argument("--from-config", required=True)
    d.add_argument("--from-manifest", required=True)
    d.add_argument("--cells", required=True, help="comma-separated TrainTest cell ids, `none` first")
    d.add_argument("--defenses-dir", required=True)
    d.add_argument("--out-config", required=True)
    d.add_argument("--out-manifest", required=True)
    d.set_defaults(fn=cmd_derive)
    r = sub.add_parser("run", help="the standup, the attestation checks and the three-way load check")
    r.add_argument("--target", required=True)
    r.add_argument("--config", required=True)
    r.add_argument("--manifest", required=True)
    r.add_argument("--defenses-dir", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--repo-dir", default=str(REPO))
    r.add_argument("--relu-cell", required=True, help="the residual carrier's TrainTest cell id")
    r.add_argument("--lora-cell", required=True, help="the certified LoRA cell's TrainTest cell id")
    r.add_argument("--expect-object-sha256", action="append", default=[], metavar="CELL=SHA")
    r.add_argument("--adapter-bank", default=None)
    r.add_argument("--expect-n-layers", type=int, required=True, help="the model's block count the worker must hook in full")
    r.add_argument("--batches", default=",".join(str(b) for b in DEFAULT_BATCHES))
    r.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    r.add_argument("--window-s", type=float, default=DEFAULT_WINDOW_S)
    r.add_argument("--warmup-s", type=float, default=DEFAULT_WARMUP_S)
    r.add_argument("--prompt-tokens", type=int, default=800)
    r.add_argument("--max-model-len", type=int, default=4096)
    r.add_argument("--max-num-seqs", type=int, default=64)
    r.add_argument("--allow-over-budget", action="store_true")
    r.add_argument("--gpu-mem-util", default="0.9")
    r.add_argument("--ready-timeout-s", type=int, default=2400)
    r.add_argument("--probe-max-tokens", type=int, default=48)
    r.set_defaults(fn=cmd_run)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.fn(args))
    except CanaryRefused as e:
        log(f"REFUSED: {e}")
        return RC_REFUSED


if __name__ == "__main__":
    sys.exit(main())
