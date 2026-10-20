#!/usr/bin/env python
"""Capture the reference-pool activations for ONE target and compute the realised dose of every
LoRA and AlphaSteer arm from the staged object bytes (module docstring of dose_common.py for the
arithmetic). Runs under dose_capture.sbatch on one GPU; `--dry-run` does everything that needs no
model (pool, objects, tokenizer render, template hash, provenance) and is what the CPU preflight
runs.

  python dose_capture.py --target nemotron3nano4b --out-dir $SHARED/exp/strength_dose/nemotron3nano4b
      [--lora-views dm1 d0p1 ...] [--include-it-views] [--no-alphasteer] [--batch-size 8] [--dry-run]

Outputs (all under --out-dir):
  dose_<target>.json          the per-arm realised doses, the gate, the three reference norms, provenance
  capture_<target>.pt         per-row tensors (norms, signed factors, reads) — the report re-aggregates from these
  capture_sample_<target>.pt  a few prompts' raw o_proj inputs / h_last rows — the tests' oracle input
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping, Any, Dict, List, Optional, Tuple

import torch
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
for _p in (REPO / "src", REPO / "scripts" / "sdfbase" / "full22", REPO / "scripts" / "sdfbase", REPO / "scripts", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import dose_common as dc  # noqa: E402

LORA_BANK_REL = "exp/objects/sdfbase/full22"
AS_BANK_REL = "exp/objects/alphasteer_f22"
SERVING_PATHS = ("src/sisteer/capeval/serving/", "src/sisteer/capeval/tiers/petri_runner.py")
LEG_GEOMETRY = {  # the campaign's served max_model_len per leg (rendering agent; prompt side identical across legs)
    "static_families_and_bon": 4864, "multiturn": 8704, "controls": 2048}


def log(*a: Any) -> None:
    print(f"[dose {time.strftime('%H:%M:%S')}]", *a, flush=True)


def detect_shared() -> Path:
    """The cluster's shared root, $SHARED through sisteer.site (the job's SISTEER_SHARED_ROOT, else CLUSTER_SHARED_ROOT, else
    the site config's cluster block); refused by name when none is set or it is not a directory here."""
    from sisteer import site as _site
    try:
        root = _site.expand_path("$SHARED", key="--shared")
    except (_site.PlaceholderError, _site.SiteConfigError) as exc:
        raise SystemExit(f"no shared root: {exc}; or pass --shared")
    if not root.is_dir():
        raise SystemExit(f"no shared root: {root} ($SHARED) is not a directory on this host; pass --shared")
    return root


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()


def code_provenance() -> dict:
    try:
        head = git("rev-parse", "HEAD")
        dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
        serving = git("log", "-1", "--format=%H", "HEAD", "--", *SERVING_PATHS)
    except Exception as e:  # noqa: BLE001
        return {"error": f"git unavailable: {e}"}
    return {"code_sha": head, "dirty": dirty, "serving_impl_commit": serving,
            "serving_impl_commit_rule": "the last commit touching " + " + ".join(SERVING_PATHS),
            "script": str(Path(__file__).resolve().relative_to(REPO))}


def env_provenance() -> dict:
    import transformers
    rec = {"python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__,
           "cuda": torch.version.cuda, "hostname": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
           "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"), "executable": sys.executable}
    for mod in ("mamba_ssm", "causal_conv1d"):
        try:
            m = __import__(mod)
            rec[mod] = getattr(m, "__version__", "present")
        except Exception as e:  # noqa: BLE001
            rec[mod] = f"absent ({type(e).__name__})"
    return rec


# --- arms --------------------------------------------------------------------------------------
def resolve_lora_arms(target: str, shared: Path, views: List[str]) -> Tuple[Dict[str, dict], dict]:
    """{cell_id: {"rungs": {view: {...}}, "reference_view": "d1"}}, plus a record of the resolution.
    Every object's sha is checked against configs/sdfbase_full22_pins.json before it is read."""
    import f22_roster
    from make_full22_configs import view_dose
    roster = f22_roster.load_roster(REPO / "configs" / "sdfbase_full22_roster.yaml")
    pins = json.loads((REPO / "configs" / "sdfbase_full22_pins.json").read_text())
    bank = shared / LORA_BANK_REL / target
    cells: Dict[str, dict] = {}
    record = {"bank": str(bank), "views": {}}
    for view in views:
        dose = view_dose(roster, target, view)
        if dose is None:
            raise SystemExit(f"{target}/{view} is not a dose view")
        staged = f22_roster.staged_cells(roster, target, view)
        record["views"][view] = {"dose": dose, "cells": dict(staged)}
        for cell_id, staged_id in staged.items():
            obj = bank / "cells" / staged_id / "object.pt"
            if not obj.is_file():
                raise SystemExit(f"missing object {obj}")
            pin = (pins.get(target) or {}).get(staged_id)      # pins are nested per target
            want = pin["object_sha256"] if isinstance(pin, dict) else pin
            got = dc.sha256_file(obj)
            if want is None:
                raise SystemExit(f"{staged_id}: no pin in sdfbase_full22_pins.json")
            if got != want:
                raise SystemExit(f"{obj}: sha256 {got} != pinned {want}")
            cells.setdefault(cell_id, {"rungs": {}})
            cells[cell_id]["rungs"][view] = {"view": view, "dose": float(dose), "staged_id": staged_id,
                                             "object_path": str(obj), "object_sha256": got}
    # The reference rung is the one whose DECLARED DOSE is 1.0, resolved per cell rather than spelled. The
    # instruct-direct lane names it `it_d1`, so hard-coding dc.LORA_REFERENCE_VIEW ("d1") made
    # --include-it-views ALWAYS refuse on gemma4e4b / qwen35_4b even with every object correctly staged:
    # the spelling it looked for cannot occur in that lane. Requiring EXACTLY one keeps the old refusal's
    # strength — a cell with no 1.0 rung, or somehow two, still refuses and says which.
    for cell_id, c in cells.items():
        c["reference_view"] = reference_rung(f"{target}/{cell_id}", c["rungs"])
    return cells, record


def reference_rung(where: str, rungs: Mapping[str, dict]) -> str:
    """The view name of the rung at dose 1.0 — the unscaled parent the gate measures against.

    Selected BY DOSE, never by spelling. The instruct-direct lane names this rung `it_d1`, so looking for
    the literal "d1" made --include-it-views refuse on gemma4e4b / qwen35_4b however correctly their
    objects were staged: the spelling it searched for cannot occur in that lane. Exactly one is required,
    so a cell with no 1.0 rung (or, impossibly, two) still refuses and says which."""
    refs = sorted(v for v, r in rungs.items() if float(r["dose"]) == 1.0)
    if len(refs) != 1:
        raise SystemExit(f"{where}: expected exactly ONE rung at dose 1.0 among {sorted(rungs)}, "
                         f"found {refs or 'none'} — the gate needs the unscaled parent")
    return refs[0]


def log_effective_layer_summary(target: str, summary: Mapping[str, Any]) -> None:
    """Emit the NARROWED and ZERO-EFFECTIVE-LAYERS warnings, scoped to what the statistic establishes.

    `effective_layer_count` counts a layer active when the MEDIAN prompt's changed-coordinate fraction
    exceeds the threshold, at the captured position and phase. A layer whose edit survives on a MINORITY
    of prompts therefore reads INACTIVE: two layers each changing 2 of 5 prompts give a count of 0 while
    4 of those 5 prompts changed somewhere. So a zero count is a SUSPECTED near-no-op, never a
    demonstrated one, and this emitter must not say otherwise.

    It is a function so the EMITTED TEXT can be asserted under that counterexample. The same claim was
    corrected in the report first and left here, where a reader who never opens the report still met the
    categorical version — which is why the wording lives in one testable place now."""
    for k, (eff_n, nom_n) in (summary.get("narrowed") or {}).items():
        log(f"  NARROWED: {k} steers {eff_n} of its {nom_n} nominal layers on the MEDIAN prompt at the captured "
            "position/phase — the rest round away there in the served dtype")
    for k in (summary.get("zero_effective_layers") or []):
        log("!" * 100)
        log(f"!! ZERO EFFECTIVE LAYERS ON THE MEDIAN PROMPT: {k} ({target}) — at the captured position/phase, every")
        log("!! nominally-steered layer's edit rounds away in the served numeric format. This is a SUSPECTED")
        log("!! near-no-op, not a demonstrated one: the statistic is a per-layer MEDIAN over prompts, so an edit")
        log("!! surviving on a minority of prompts reads as zero here, and other positions and phases are not")
        log("!! measured. Confirm over all prompts/positions/phases before reporting this as a base-model number.")
        log("!" * 100)


def resolve_alphasteer_arm(target: str, shared: Path) -> Tuple[Optional[dict], dict]:
    bank = shared / AS_BANK_REL / target / "strength"
    rec_path = bank / "strength_bake.json"
    if not rec_path.is_file():
        # A bank with staged cells but NO bake record is incomplete staging, not an absent arm. Returning
        # None there silently drops the arm, and on an instruct target (whose LoRA views are excluded by
        # default) that leaves NO arms at all — after which the capture loads the model, writes output and
        # passes a gate over an empty rung set. Refuse instead; --no-alphasteer is how you opt out on purpose.
        if (bank / "cells").is_dir():
            raise SystemExit(f"{target}: {bank / 'cells'} holds staged cells but {rec_path.name} is MISSING — "
                             "the AlphaSteer arm is staged but unrecorded (incomplete staging). Refusing rather "
                             "than dropping the arm; pass --no-alphasteer to run without it deliberately.")
        return None, {"bank": str(bank), "status": "no strength_bake.json"}
    bake = json.loads(rec_path.read_text())
    rungs = {}
    for c in bake["cells"]:
        obj = bank / "cells" / c["staged_id"] / "object.pt"      # the record's path carries the baking cluster's prefix
        if not obj.exists():
            raise SystemExit(f"missing object {obj}")
        got = dc.sha256_file(obj)
        if got != c["object_sha256"]:
            raise SystemExit(f"{obj}: sha256 {got} != bake record {c['object_sha256']}")
        fp = bank / "cells" / c["staged_id"] / "fit_provenance.json"
        fitp = json.loads(fp.read_text()) if fp.is_file() else {}
        rungs[c["staged_id"]] = {"view": c["staged_id"].split("__")[0], "strength": float(c["strength"]),
                                 "staged_id": c["staged_id"], "object_path": str(obj), "object_sha256": got,
                                 "layers_recorded": list(c["layers"]), "is_parent": bool(c.get("is_parent")),
                                 "fit_chat_template_hash": fitp.get("chat_template_hash"),
                                 "fit_positions": fitp.get("positions_at_fit")}
    ref = [k for k, v in rungs.items() if v["strength"] == dc.ALPHASTEER_REFERENCE_STRENGTH]
    if len(ref) != 1:
        raise SystemExit(f"{target}: expected exactly one strength-1.0 AlphaSteer rung, got {ref}")
    return {"rungs": rungs, "reference": ref[0]}, {"bank": str(bank), "bake_record": str(rec_path),
                                                    "anchor_strength": bake.get("anchor_strength")}


# --- the capture ---------------------------------------------------------------------------------
#: every attribute a decoder block might hang an RMSNorm off between attention and the residual add —
#: probed on BOTH the norm-bearing and the direct-add architectures, because the discriminating
#: evidence for "this one adds directly" is that NONE of these receives the o_proj output
_NORM_CANDIDATE_ATTRS = ("post_attention_layernorm", "post_self_attn_layernorm", "pre_feedforward_layernorm")


class Capture:
    """Forward hooks that record, per batch: the o_proj INPUT and OUTPUT rows and the block's
    resid_pre/resid_post at every LoRA layer; the block OUTPUT at every AlphaSteer hook layer; and the
    INPUT of every candidate post-attention norm, which is what verifies where the o_proj edit lands."""

    def __init__(self, model, lora_layers: List[int], as_hook_layers: List[int]):
        from sisteer.models import attention_oproj, get_decoder_layers
        self.decoder = get_decoder_layers(model)
        self.lora_layers = sorted(set(lora_layers))
        self.as_layers = sorted(set(as_hook_layers))
        self.oproj = {}
        self.norm_mod: Dict[int, Any] = {}
        self.norm_attr: Dict[int, Optional[str]] = {}
        self.block_class = {}
        for l in self.lora_layers:
            m = attention_oproj(self.decoder[l])
            if m is None:
                raise SystemExit(f"decoder layer {l} has no attention o_proj — the LoRA object cannot be served here")
            self.oproj[l] = m
            mod, attr = dc.post_attention_norm(self.decoder[l])
            self.norm_mod[l], self.norm_attr[l] = mod, attr
            self.block_class[l] = type(self.decoder[l]).__name__
        self.x: Dict[int, torch.Tensor] = {}
        self.o_out: Dict[int, torch.Tensor] = {}
        self.pre: Dict[int, torch.Tensor] = {}
        self.post: Dict[int, torch.Tensor] = {}
        self.norm_in: Dict[Tuple[int, str], torch.Tensor] = {}
        self.handles = []

    def __enter__(self):
        for l in self.lora_layers:
            self.handles.append(self.oproj[l].register_forward_hook(self._oproj_hook(l)))
            self.handles.append(self.decoder[l].register_forward_pre_hook(self._pre_hook(l), with_kwargs=True))
            for attr in _NORM_CANDIDATE_ATTRS:
                mod = getattr(self.decoder[l], attr, None)
                if mod is not None:
                    self.handles.append(mod.register_forward_pre_hook(self._norm_hook(l, attr)))
        for l in sorted(set(self.lora_layers) | set(self.as_layers)):
            self.handles.append(self.decoder[l].register_forward_hook(self._post_hook(l)))
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()
        self.handles = []

    def verify_norm_placement(self) -> dict:
        """MEASURE where the o_proj edit lands and hold the architecture table to it, by REPLAYING THE
        BLOCK'S OWN RESIDUAL ADD on the captured tensors:
          * norm-bearing (table names a norm N): N's captured input must be the o_proj output, AND some
            downstream norm's input (the residual after attention, `resid_mid`) must equal
            `resid_pre + N(o_proj_out)` bit-for-bit;
          * direct-add (table says None): `resid_pre + o_proj_out` must equal either the block output
            (NemotronH, whose MLP is its own block) or some downstream norm's input (Llama/Qwen-style
            blocks, whose `post_attention_layernorm` receives the residual on its way to the MLP) — and
            no candidate norm may receive the raw o_proj output.
        Equality is NaN-aware (left padding leaves fully-masked attention rows that can be NaN). Run on
        the first batch; a table entry that is wrong here is a wrong quantity everywhere."""
        rec = {}
        for l in self.lora_layers:
            o = self.o_out[l]; pre = self.pre[l]; post = self.post[l]
            cands = {a: self.norm_in.get((l, a)) for a in _NORM_CANDIDATE_ATTRS if self.norm_in.get((l, a)) is not None}
            fed_o = sorted(a for a, t in cands.items() if dc.nan_equal(o, t))
            want = self.norm_attr[l]
            with torch.no_grad():
                contrib = o if self.norm_mod[l] is None else self.norm_mod[l](o)
                mid_expected = pre + contrib
            fed_mid = sorted(a for a, t in cands.items() if dc.nan_equal(mid_expected, t))
            block_out_is_mid = dc.nan_equal(mid_expected, post)
            r = {"block_class": self.block_class[l], "declared_post_attention_norm": want,
                 "candidates_present": sorted(cands), "candidates_fed_the_oproj_output": fed_o,
                 "candidates_fed_resid_pre_plus_contribution": fed_mid, "block_output_is_resid_pre_plus_contribution": block_out_is_mid,
                 "nan_rows_in_oproj_output": int(o.isnan().any(dim=-1).sum()),
                 "diag_max_abs_diff_vs_candidates": {a: float((o.float() - t.float()).nan_to_num(0).abs().max()) for a, t in cands.items()}}
            rec[l] = r
            if want is None:
                if fed_o:
                    raise SystemExit(f"layer {l} ({self.block_class[l]}): the table says the attention output is added to "
                                     f"the residual DIRECTLY, but {fed_o} receives the o_proj output — the edit is re-normalised "
                                     f"on this architecture and the realised dose would be the wrong quantity. {json.dumps(r)}")
                if not (block_out_is_mid or fed_mid):
                    raise SystemExit(f"layer {l} ({self.block_class[l]}): direct add claimed, but resid_pre + o_proj_out matches "
                                     f"neither the block output nor any downstream norm's input — the path is not the one "
                                     f"the table describes; refusing. {json.dumps(r)}")
            else:
                if fed_o != [want]:
                    raise SystemExit(f"layer {l} ({self.block_class[l]}): the table says `{want}` normalises the attention output, "
                                     f"but the module(s) actually fed the o_proj output are {fed_o or 'none'}. {json.dumps(r)}")
                if not fed_mid:
                    raise SystemExit(f"layer {l} ({self.block_class[l]}): `{want}` receives the o_proj output, but "
                                     f"resid_pre + {want}(o_proj_out) matches no downstream norm's input — the residual add is not "
                                     f"the one the table describes; refusing. {json.dumps(r)}")
        return rec

    def _oproj_hook(self, l):
        def hook(module, args, output):
            self.x[l] = args[0].detach()
            out = output[0] if isinstance(output, tuple) else output
            self.o_out[l] = out.detach()
        return hook

    def _norm_hook(self, l, attr):
        def hook(module, args):
            if args:
                self.norm_in[(l, attr)] = args[0].detach()
            return None
        return hook

    def _pre_hook(self, l):
        def hook(module, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            self.pre[l] = hidden.detach()
            return None
        return hook

    def _post_hook(self, l):
        def hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self.post[l] = hidden.detach()
        return hook

    def clear(self):
        self.x.clear(); self.o_out.clear(); self.pre.clear(); self.post.clear(); self.norm_in.clear()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--target", required=True, choices=dc.TARGETS)
    ap.add_argument("--shared", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--pool", type=Path, default=None, help=f"default $SHARED/{dc.POOL_REL}")
    ap.add_argument("--pool-sha256", default=dc.POOL_SHA256)
    ap.add_argument("--lora-views", nargs="*", default=None,
                    help="LoRA dose views (default: the roster's dose views of this target, `it_*` excluded)")
    ap.add_argument("--include-it-views", action="store_true", help="admit the instruct-direct `it_*` LoRA views")
    ap.add_argument("--no-alphasteer", action="store_true")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--sample-prompts", type=int, default=6, help="raw rows kept for the tests' oracle (first k of each half)")
    ap.add_argument("--dry-run", action="store_true", help="everything but the model: pool, objects, render, provenance")
    args = ap.parse_args(argv)

    t0 = time.time()
    shared = args.shared or detect_shared()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    target = args.target
    cfg = yaml.safe_load((REPO / "configs" / f"sdfbase_full22_{target}.yaml").read_text())
    model_id, revision = cfg["model"], cfg["model_revision"]
    model_dtype = getattr(torch, cfg.get("dtype", "bfloat16"))

    # 1. the pool — the shared artefact, refused unless byte-identical to the pin
    pool_path = args.pool or (shared / dc.POOL_REL)
    rows = dc.load_pool(pool_path, args.pool_sha256)
    texts = [r["text"] for r in rows]
    is_harmful = torch.tensor([r["half"] == "harmful" for r in rows])
    log(f"pool {pool_path}: {len(rows)} rows ({int(is_harmful.sum())} harmful / {int((~is_harmful).sum())} benign), sha ok")

    # 2. the arms
    import f22_roster
    roster = f22_roster.load_roster(REPO / "configs" / "sdfbase_full22_roster.yaml")
    all_views = list(roster["targets"][target]["views"])
    dose_views = [v for v in all_views if "dose" in roster["targets"][target]["views"][v]]
    if args.lora_views is None:
        lora_views = [v for v in dose_views if args.include_it_views or not v.startswith("it_")]
    else:
        lora_views = list(args.lora_views)
        bad = [v for v in lora_views if v.startswith("it_") and not args.include_it_views]
        if bad:
            raise SystemExit(f"{bad}: `it_*` views are excluded by the 2026-09-11 scope cut; pass --include-it-views")
    excluded = sorted(set(dose_views) - set(lora_views))
    lora_cells, lora_record = (resolve_lora_arms(target, shared, lora_views) if lora_views else ({}, {"views": {}}))
    lora_record["excluded_views"] = excluded
    log(f"LoRA views {lora_views} -> cells {sorted(lora_cells)}; excluded {excluded}")
    as_arm, as_record = (None, {"status": "disabled"}) if args.no_alphasteer else resolve_alphasteer_arm(target, shared)
    log(f"AlphaSteer: {sorted(as_arm['rungs']) if as_arm else as_record}")
    # An empty arm set must never reach the model. `all([])` is True, so a run with nothing resolved writes
    # output and reports gate_summary.all_pass over zero rungs -- a vacuous pass that reads exactly like a
    # real one, and contributes no rows to a combined report while not appearing under missing_targets.
    if not lora_cells and as_arm is None:
        raise SystemExit(f"{target}: NO arms resolved — LoRA views {lora_views or '[]'} (excluded "
                         f"{excluded or '[]'}) and AlphaSteer {as_record.get('status', 'absent')!r}. "
                         "There is nothing to capture, and a gate over zero rungs passes vacuously. "
                         "Refusing before the model is loaded.")

    # LoRA blobs -> ops (small; all rungs in memory)
    lora_ops: Dict[str, Dict[str, List[dc.LoraOp]]] = {}
    lora_layers: List[int] = []
    for cell_id, c in lora_cells.items():
        lora_ops[cell_id] = {}
        for view, r in c["rungs"].items():
            blob = dc.load_blob(Path(r["object_path"]))
            ops = dc.lora_ops_from_blob(blob)
            r["kind"] = blob["kind"]
            r["layers"] = sorted({op.layer for op in ops})
            r["rank"] = int(ops[0].A.shape[1])
            r["d_in"], r["d_out"] = int(ops[0].A.shape[0]), int(ops[0].B.shape[0])
            lora_ops[cell_id][view] = ops
            lora_layers.extend(r["layers"])
        layer_sets = {tuple(r["layers"]) for r in c["rungs"].values()}
        if len(layer_sets) != 1:
            raise SystemExit(f"{cell_id}: rungs disagree on layers {layer_sets}")
        # the read basis is UNCHANGED across rungs (the producer's rule); assert it, so a rung whose A moved is refused
        ref_ops = lora_ops[cell_id][c["reference_view"]]
        for view, ops in lora_ops[cell_id].items():
            for o, oref in zip(ops, ref_ops):
                if not torch.equal(o.A, oref.A) or o.coeff != oref.coeff or o.layer != oref.layer:
                    raise SystemExit(f"{cell_id}/{view}: read basis A (or op order) differs from the reference rung — "
                                     "the dose copy is not `parent with B scaled`; refusing")
    lora_layers = sorted(set(lora_layers))

    # AlphaSteer blobs: layer geometry now (mmap; the matrices are read later, per rung, on the GPU)
    as_objs: Dict[str, dc.AlphaSteerObject] = {}
    as_hook_layers: List[int] = []
    if as_arm:
        for sid, r in as_arm["rungs"].items():
            obj = dc.alphasteer_from_blob(dc.load_blob(Path(r["object_path"]), mmap=True))
            if obj.base_scale != r["strength"]:
                raise SystemExit(f"{sid}: blob base_scale {obj.base_scale} != bake record strength {r['strength']}")
            if list(obj.fit_layers) != sorted(r["layers_recorded"]):
                raise SystemExit(f"{sid}: blob fit layers {obj.fit_layers} != recorded {r['layers_recorded']}")
            if obj.site != "pre":
                raise SystemExit(f"{sid}: site={obj.site!r}; the strength lane's objects are site=pre (hook = fit - 1)")
            r.update(fit_layers=list(obj.fit_layers), hook_layers=list(obj.hook_layers), d=obj.d, site=obj.site,
                     base_scale=obj.base_scale)
            if r["fit_chat_template_hash"] and r["fit_chat_template_hash"] != dc.EXPECTED_TEMPLATE_HASH[target]:
                raise SystemExit(f"{sid}: fit_provenance chat_template_hash {r['fit_chat_template_hash']} != served "
                                 f"{dc.EXPECTED_TEMPLATE_HASH[target]} — the object was fit under another template")
            as_objs[sid] = obj
            as_hook_layers = list(obj.hook_layers)
        geo = {(tuple(o.fit_layers), o.d) for o in as_objs.values()}
        if len(geo) != 1:
            raise SystemExit(f"AlphaSteer rungs disagree on geometry {geo}")

    # 3. tokenizer + render (the pinned render; refuses the wrong template)
    from sisteer.models import load_tokenizer
    tok = load_tokenizer(model_id, revision=revision)
    thash = dc.assert_template_hash(target, tok)
    rendered = dc.render_texts(tok, texts)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    lens = [len(tok(r, add_special_tokens=False)["input_ids"]) for r in rendered]
    log(f"render ok: template {thash}, prompt tokens min/median/max {min(lens)}/{sorted(lens)[len(lens)//2]}/{max(lens)}")

    provenance = {
        "schema": dc.SCHEMA, "target": target, "model": model_id, "model_revision": revision, "dtype": str(model_dtype),
        "pool": {"path": str(pool_path), "sha256": args.pool_sha256, "n_rows": len(rows),
                 "n_harmful": int(is_harmful.sum()), "n_benign": int((~is_harmful).sum()),
                 "halves": "idx < 259 harmful (HarmBench standard/test 159 + stratified JBB 100 = two_si_run._eval_goals(confirmatory=True)); then load_xstest_safe()[:120]",
                 "sidecars": ["pool_v1.sha256", "pool_v1_provenance.json"]},
        "render": {"call": "sisteer.models.load_tokenizer(model, revision) -> build_chat_prompt(tok, text, system=None) -> apply_chat_template(add_generation_prompt=True, **tok._sisteer_template_kwargs)",
                   "template_kwargs": dict(getattr(tok, "_sisteer_template_kwargs", {}) or {}),
                   "add_special_tokens": False, "padding_side": "left", "system": None,
                   "chat_template_hash": thash, "chat_template_hash_expected": dc.EXPECTED_TEMPLATE_HASH[target],
                   "prompt_tokens": {"min": min(lens), "median": sorted(lens)[len(lens) // 2], "max": max(lens)}},
        "geometry": {"note": "prompt-side render identical across legs; the capture has no completion, so geometry only bounds the prompt, which fits every leg",
                     "served_max_model_len_by_leg": LEG_GEOMETRY, "max_prompt_tokens_in_pool": max(lens)},
        "positions": {"captured": "every real (unmasked) prompt token; `last` = the final prompt row = the generation-prompt position",
                      "lora_headline": "last prompt row (D54); all-rows median reported beside it",
                      "alphasteer": "h_last = last prompt row (the row the served op reads); edit broadcast to every prompt row"},
        "h_definition": {"lora": "||resid_pre(L)||: the residual stream entering the stored layer's block, fp32 norm (D54); resid_post, the o_proj input and the o_proj output norms recorded beside it",
                         "alphasteer": "||h_last|| at the block OUTPUT of hook layer = fit layer - 1 (site=pre), fp32 norm"},
        "edit_definition": {"lora_at_the_oproj_site": "sum_ops coeff * ((x.float() @ A) @ B^T), x = o_proj input; fp32 (gated) and served-dtype composition (cast per op, added in the model dtype, exactly as apply_oproj_ops does)",
                            "lora_at_the_residual": "what the o_proj edit DELIVERS to the residual stream: the edit itself on a direct-add block (NemotronHBlock, Qwen3_5DecoderLayer), or N(attn_out + edit) - N(attn_out) through the block's OWN post-attention norm module where the architecture re-normalises the attention output first (Gemma4TextDecoderLayer post_attention_layernorm, Glm4vTextDecoderLayer post_self_attn_layernorm). THE HEADLINE realised dose is this one over ||resid_pre||; the o_proj-site ratio is reported beside it and they are NOT the same quantity on a norm-bearing block. The placement is MEASURED per layer (norm_placement), never inferred from an attribute name.",
                            "alphasteer": "base_scale * (h_last @ M): fp32 (gated) and model-dtype matmul with M cast to the model dtype (reported)"},
        "effective_layers": {"question": "of the layers an object nominally steers, how many contribute an edit that survives the served numeric format rather than rounding away",
                             "criterion": "elementwise (base + edit) != base in the SERVING dtype — the served arithmetic itself, so it is right at subnormals, exponent boundaries and ties-to-even; per (prompt, layer) the fraction of hidden-dim coordinates that move",
                             "site": {"lora": "the residual stream at the last prompt row (both the o_proj-site cast and, where present, the post-attention norm must leave the sum changed; the o_proj-site fraction is reported separately)",
                                      "alphasteer": "the residual stream at the hook layer, last prompt row (the row whose h defines the edit)"},
                             "active_rule": "median prompt's fraction > 0; `_half` requires >= 0.5",
                             "recompute": "median_frac_changed_by_layer and per_layer_realised_dose_median are shipped per rung, at EVERY rung including the negative one, so a reader who disagrees with the threshold recomputes the count without re-running anything",
                             "ulp_unit": "|edit| in ULPs of the base at each element's magnitude, via lens_adapters.ulp_spacing (the serving tree's own definition, imported not copied)"},
        "gate": {"rule": "D51/D55: per (prompt, layer) signed factor c = <e_s, e_ref>/<e_ref, e_ref> on fp32 edits; median == s/s_ref (rtol %g), every row within %g, residual <= %g; norm-ratio median == |s/s_ref|" % (dc.GATE_RTOL_MEDIAN, dc.GATE_ATOL_MAX, dc.GATE_RESID_MAX)},
        "lora": lora_record, "alphasteer": as_record, "code": code_provenance(), "env": env_provenance(),
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if args.dry_run:
        out = {"dry_run": True, "provenance": provenance, "lora_cells": dc.json_ready(lora_cells),
               "alphasteer": dc.json_ready(as_arm), "lora_layers": lora_layers, "alphasteer_hook_layers": as_hook_layers,
               "elapsed_s": round(time.time() - t0, 1)}
        p = args.out_dir / f"dryrun_{target}.json"
        p.write_text(json.dumps(out, indent=1) + "\n")
        log(f"DRY-RUN OK -> {p}")
        return 0

    # 4. the model
    from sisteer.models import load_model_and_tokenizer
    model, _tok2 = load_model_and_tokenizer(model_id, revision=revision, dtype=model_dtype,
                                            attn_implementation=cfg.get("attn_implementation", "sdpa"))
    model.eval()
    dev = next(model.parameters()).device
    if target == "nemotron3nano4b":
        import transformers.models.nemotron_h.modeling_nemotron_h as nh
        if not getattr(nh, "is_fast_path_available", False):
            raise SystemExit("NemotronH fused Mamba fast path is NOT available (mamba_ssm/causal_conv1d) — refusing "
                             "to run the pure-torch SSM fallback (the standing Nemotron rule)")
        provenance["env"]["nemotron_fast_path"] = True
    cap = Capture(model, lora_layers, as_hook_layers)
    for cell_id, c in lora_cells.items():
        r0 = c["rungs"][c["reference_view"]]
        for l in r0["layers"]:
            m = cap.oproj[l]
            d_in, d_out = int(m.weight.shape[1]), int(m.weight.shape[0])
            if (d_in, d_out) != (r0["d_in"], r0["d_out"]):
                raise SystemExit(f"{cell_id} layer {l}: o_proj is {d_in}->{d_out} but the object's A/B are "
                                 f"{r0['d_in']}->{r0['d_out']}")
    from sisteer.extraction.capture import _last_logit_kwargs
    logit_kw = _last_logit_kwargs(model)

    # per-row stores (LoRA) and per-prompt stores (AlphaSteer)
    N = len(rows)
    norm_placement: Dict[int, dict] = {}
    row_prompt: List[int] = []; row_pos: List[int] = []; row_is_last: List[bool] = []
    h_pre_norm = {l: [] for l in lora_layers}; h_post_norm = {l: [] for l in lora_layers}; x_norm = {l: [] for l in lora_layers}
    o_out_norm = {l: [] for l in lora_layers}
    reads_ref = {cell: {l: [] for l in lora_layers} for cell in lora_cells}     # [rows, n_ops*k] fp32, the reference rung's reads
    _STORE = ("fp32", "served", "c", "resid", "e_res_fp32", "e_res_served",
              "surv_oproj", "surv_resid", "ulps_resid_max")
    lora_edit = {cell: {view: {l: {k: [] for k in _STORE} for l in lora_layers}
                        for view in c["rungs"]} for cell, c in lora_cells.items()}
    as_h_last = {l: [] for l in as_hook_layers}                 # [N, d] model dtype
    as_h_norm_rows = {l: [] for l in as_hook_layers}            # per row fp32 norms of the block output
    as_row_prompt: List[int] = []
    sample_idx = set(range(args.sample_prompts)) | set(range(dc.POOL_N_HARMFUL, dc.POOL_N_HARMFUL + args.sample_prompts))
    sample = {"prompt_idx": sorted(sample_idx), "lora_x": {l: {} for l in lora_layers}, "lora_resid_pre": {l: {} for l in lora_layers},
              "lora_oproj_out": {l: {} for l in lora_layers},
              "alphasteer_h_last": {l: {} for l in as_hook_layers}, "alphasteer_h_rows": {l: {} for l in as_hook_layers}}

    with cap, torch.no_grad():
        for i in range(0, N, args.batch_size):
            batch = rendered[i:i + args.batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
            mask = enc["attention_mask"].bool()
            cap.clear()
            model(**enc, use_cache=False, **logit_kw)
            B, S = mask.shape
            for b in range(B):
                pidx = i + b
                m = mask[b]
                n_real = int(m.sum())
                if n_real == 0 or not bool(m[-1]):
                    raise SystemExit(f"prompt {pidx}: left padding expected the last column real (n_real={n_real})")
                if lora_layers:
                    row_prompt.extend([pidx] * n_real); row_pos.extend(range(n_real))
                    row_is_last.extend([False] * (n_real - 1) + [True])
                for l in lora_layers:
                    x = cap.x[l][b][m]                       # [T, d_in] model dtype
                    attn_out = cap.o_out[l][b][m]            # [T, d_out] model dtype — what the edit is added to
                    pre_m = cap.pre[l][b][m]                 # the residual stream entering the block
                    if not (torch.isfinite(x).all() and torch.isfinite(attn_out).all() and torch.isfinite(pre_m).all()):
                        raise SystemExit(f"prompt {pidx} layer {l}: non-finite values on REAL (unmasked) rows — refusing to "
                                         "measure a norm ratio on them (NaN is expected only on padded rows)")
                    pre = pre_m.to(torch.float32)
                    post = cap.post[l][b][m].to(torch.float32)
                    h_pre_norm[l].append(pre.norm(dim=-1).cpu()); h_post_norm[l].append(post.norm(dim=-1).cpu())
                    x_norm[l].append(x.to(torch.float32).norm(dim=-1).cpu())
                    o_out_norm[l].append(attn_out.to(torch.float32).norm(dim=-1).cpu())
                    if pidx in sample_idx:
                        sample["lora_x"][l][pidx] = x.cpu(); sample["lora_resid_pre"][l][pidx] = pre.cpu()
                        sample["lora_oproj_out"][l][pidx] = attn_out.cpu()
                    for cell_id, c in lora_cells.items():
                        ref_view = c["reference_view"]
                        edits = {}
                        for view, ops in lora_ops[cell_id].items():
                            lops = [op for op in ops if op.layer == l]
                            reads = [dc.lora_read(x, op) for op in lops]
                            if view == ref_view:
                                reads_ref[cell_id][l].append(torch.cat(reads, dim=-1).cpu())
                            fp32, steered = dc.lora_edit_from_reads(reads, lops, model_dtype, base=attn_out)
                            edits[view] = (fp32, steered)
                        e_ref = edits[ref_view][0]
                        for view, (fp32, steered) in edits.items():
                            cfac, resid = dc.signed_factor(fp32, e_ref)
                            served = steered.to(torch.float32) - attn_out.to(torch.float32)   # what the hook actually added
                            # what reaches the RESIDUAL STREAM: the edit itself on a direct-add block,
                            # N(attn+e) - N(attn) where the attention output is re-normalised first
                            e_res32, e_res_sv, mid_st, mid = dc.residual_edit_through_norm(
                                attn_out, fp32, steered, pre_m, cap.norm_mod[l], model_dtype)
                            sv_o = dc.survival_between(attn_out, steered, model_dtype)     # survives the o_proj-site cast?
                            sv_r = dc.survival_between(mid, mid_st, model_dtype)           # ... and still moves the residual?
                            st = lora_edit[cell_id][view][l]
                            st["fp32"].append(fp32.norm(dim=-1).cpu()); st["served"].append(served.norm(dim=-1).cpu())
                            st["c"].append(cfac.cpu()); st["resid"].append(resid.cpu())
                            st["e_res_fp32"].append(e_res32.norm(dim=-1).cpu())
                            st["e_res_served"].append(e_res_sv.norm(dim=-1).cpu())
                            st["surv_oproj"].append(sv_o["frac_changed"].cpu())
                            st["surv_resid"].append(sv_r["frac_changed"].cpu())
                            st["ulps_resid_max"].append(sv_r["edit_ulps_max"].cpu())
                if as_hook_layers:
                    as_row_prompt.extend([pidx] * n_real)
                for l in as_hook_layers:
                    h = cap.post[l][b][m]                    # [T, d] model dtype — the block OUTPUT
                    as_h_last[l].append(h[-1].unsqueeze(0).cpu())
                    as_h_norm_rows[l].append(h.to(torch.float32).norm(dim=-1).cpu())
                    if pidx in sample_idx:
                        sample["alphasteer_h_last"][l][pidx] = h[-1].cpu(); sample["alphasteer_h_rows"][l][pidx] = h.to(torch.float32).norm(dim=-1).cpu()
            if i == 0:
                if lora_layers:
                    norm_placement = cap.verify_norm_placement()
                    for l, r in norm_placement.items():
                        if r["declared_post_attention_norm"]:
                            how = (f"re-normalised by `{r['declared_post_attention_norm']}` before the residual add (VERIFIED: it "
                                   f"receives the o_proj output and resid_pre + norm(o_proj_out) reappears at {r['candidates_fed_resid_pre_plus_contribution']})")
                        else:
                            how = ("added to the residual DIRECTLY (VERIFIED: resid_pre + o_proj_out reappears "
                                   + ("as the block output" if r["block_output_is_resid_pre_plus_contribution"] else f"at {r['candidates_fed_resid_pre_plus_contribution']}")
                                   + (f"; candidate norms present {r['candidates_present']}, none fed the o_proj output" if r["candidates_present"] else "; this block has no candidate norm module") + ")")
                        log(f"  layer {l} {r['block_class']}: attention output {how}"
                            + (f" [{r['nan_rows_in_oproj_output']} NaN pad rows in the batch]" if r["nan_rows_in_oproj_output"] else ""))
                log(f"first batch: {B} prompts, S={S}, lora layers {lora_layers}, alphasteer hook layers {as_hook_layers[:3]}...")
    log(f"forward done in {time.time() - t0:.0f}s; {len(row_prompt)} LoRA rows")
    del cap
    torch.cuda.empty_cache()

    # 5. AlphaSteer edits per rung, from each rung's OWN bytes, on the captured h_last
    as_result = None
    if as_arm:
        H = {l: torch.cat(as_h_last[l], 0).to(dev) for l in as_hook_layers}      # [N, d]
        ref_sid = as_arm["reference"]
        as_edits: Dict[str, Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = {}
        for sid in sorted(as_arm["rungs"], key=lambda s: as_arm["rungs"][s]["strength"]):
            t1 = time.time()
            obj = dc.alphasteer_from_blob(dc.load_blob(Path(as_arm["rungs"][sid]["object_path"]), mmap=True))
            per_layer = {}
            for fl, hl in zip(obj.fit_layers, obj.hook_layers):
                M = obj.Ms[fl].to(dev)
                fp32, served = dc.alphasteer_edit(H[hl], M, obj.base_scale, model_dtype)
                sv = dc.survival(H[hl], served, model_dtype)   # the edit is added straight to the residual stream
                per_layer[fl] = (fp32.norm(dim=-1).cpu(), served.to(torch.float32).norm(dim=-1).cpu(), fp32.cpu(),
                                 sv["frac_changed"].cpu(), sv["edit_ulps_max"].cpu())
                del M
            as_edits[sid] = per_layer
            torch.cuda.empty_cache()
            log(f"alphasteer {sid} (base_scale {obj.base_scale}): edits on {len(per_layer)} layers in {time.time() - t1:.0f}s")
        as_result = {"reference": ref_sid, "rungs": {}}
        h_last_norm = {l: H[l].to(torch.float32).norm(dim=-1).cpu() for l in as_hook_layers}
        as_rows_prompt = torch.tensor(as_row_prompt)
        for sid, per_layer in as_edits.items():
            r = as_arm["rungs"][sid]
            s = r["strength"]; s_ref = as_arm["rungs"][ref_sid]["strength"]
            by_layer = {}; cs = []; resids = []; nrs = []; ratio_last = {}; ratio_rows_med = {}; surv = {}
            for fl, hl in zip(r["fit_layers"], r["hook_layers"]):
                n32, nsv, e, frac, ulps = per_layer[fl]
                e_ref = as_edits[ref_sid][fl][2]
                cfac, resid = dc.signed_factor(e, e_ref)
                cs.append(cfac); resids.append(resid); nrs.append(n32 / as_edits[ref_sid][fl][0])
                ratio_last[fl] = n32 / h_last_norm[hl]
                surv[fl] = frac
                # against the per-prompt median token-row norm at this hook layer (the peer's pooled convention)
                med_rows = torch.stack([as_h_norm_rows[hl][p].median() for p in range(N)])
                ratio_rows_med[fl] = n32 / med_rows
                by_layer[fl] = {"hook_layer": hl, "realised_dose_last": dc.three_way(ratio_last[fl], is_harmful),
                                "realised_dose_vs_row_median": dc.three_way(ratio_rows_med[fl], is_harmful),
                                "edit_norm_fp32": dc.quantiles(n32), "edit_norm_served_dtype": dc.quantiles(nsv),
                                "served_over_fp32_norm_ratio": dc.quantiles(nsv / n32),
                                "h_last_norm": dc.three_way(h_last_norm[hl], is_harmful),
                                "signed_factor_vs_reference": dc.quantiles(cfac),
                                "frac_changed_served": dc.quantiles(frac), "edit_ulps_max": dc.quantiles(ulps)}
            agg = dc.per_prompt_over_layers(ratio_last)
            eff = dc.effective_layers(surv)
            eff["per_layer_realised_dose_median"] = {int(fl): float(v.to(torch.float64).median()) for fl, v in ratio_last.items()}
            eff["site"] = "the residual stream at the hook layer (the AlphaSteer edit is added there directly)"
            gate = dc.gate_verdict(torch.stack(cs), torch.stack(resids), torch.stack(nrs), s, s_ref)
            as_result["rungs"][sid] = {
                "view": r["view"], "nominal_dose": s, "dose_unit": "alphasteer_base_scale", "object_sha256": r["object_sha256"],
                "staged_id": sid, "fit_layers": r["fit_layers"], "hook_layers": r["hook_layers"],
                "realised_dose_median": float(agg["median_over_layers"].median()),
                "realised_dose": dc.three_way(agg["median_over_layers"], is_harmful),
                "realised_dose_max_layer": dc.three_way(agg["max_over_layers"], is_harmful),
                "signed_realised_dose_median": float(agg["median_over_layers"].median()) * (1 if s >= 0 else -1),
                "effective_layers": eff, "gate": gate, "by_layer": by_layer}
        as_result["reference_norms_by_hook_layer"] = {
            hl: {"token_rows": dc.three_way(torch.cat(as_h_norm_rows[hl]), is_harmful[as_rows_prompt]),
                 "last_row": dc.three_way(h_last_norm[hl], is_harmful)} for hl in as_hook_layers}
        del H
        torch.cuda.empty_cache()

    # 6. LoRA aggregation
    lora_result = {}
    rp = torch.tensor(row_prompt) if row_prompt else None
    is_last = torch.tensor(row_is_last) if row_is_last else None
    for cell_id, c in lora_cells.items():
        ref_view = c["reference_view"]
        lora_result[cell_id] = {"reference_view": ref_view, "kind": c["rungs"][ref_view]["kind"], "rungs": {}}
        for view, r in c["rungs"].items():
            s, s_ref = r["dose"], c["rungs"][ref_view]["dose"]
            by_layer = {}; cs = []; resids = []; nrs = []; ratio_last = {}; ratio_rows = {}; surv = {}
            for l in r["layers"]:
                st = lora_edit[cell_id][view][l]
                n32 = torch.cat(st["fp32"]); nsv = torch.cat(st["served"]); cfac = torch.cat(st["c"]); resid = torch.cat(st["resid"])
                nres32 = torch.cat(st["e_res_fp32"]); nres_sv = torch.cat(st["e_res_served"])
                so = torch.cat(st["surv_oproj"]); sr = torch.cat(st["surv_resid"]); ulps = torch.cat(st["ulps_resid_max"])
                nref = torch.cat(lora_edit[cell_id][ref_view][l]["e_res_fp32"])
                hpre = torch.cat(h_pre_norm[l]); hpost = torch.cat(h_post_norm[l]); hattn = torch.cat(o_out_norm[l])
                # THE HEADLINE: what reaches the residual stream, over the residual stream's own norm
                ratio_rows_all = nres32 / hpre
                ratio_last[l] = ratio_rows_all[is_last]
                surv[l] = sr[is_last]
                # per-prompt median over that prompt's rows
                med = torch.zeros(N, dtype=torch.float64)
                for p in range(N):
                    med[p] = ratio_rows_all[rp == p].to(torch.float64).median()
                ratio_rows[l] = med
                cs.append(cfac); resids.append(resid); nrs.append(n32 / torch.cat(lora_edit[cell_id][ref_view][l]["fp32"]))
                by_layer[l] = {"block_class": norm_placement.get(l, {}).get("block_class"),
                               "post_attention_norm": norm_placement.get(l, {}).get("declared_post_attention_norm"),
                               "realised_dose_last": dc.three_way(ratio_last[l], is_harmful),
                               "realised_dose_row_median_per_prompt": dc.three_way(med, is_harmful),
                               "realised_dose_token_rows": dc.three_way(ratio_rows_all, is_harmful[rp]),
                               "realised_dose_vs_resid_post_last": dc.three_way((nres32 / hpost)[is_last], is_harmful),
                               "oproj_site_ratio_last": dc.three_way((n32 / hattn)[is_last], is_harmful),
                               "oproj_site_edit_over_resid_pre_last": dc.three_way((n32 / hpre)[is_last], is_harmful),
                               "residual_edit_norm_fp32_last": dc.quantiles(nres32[is_last]),
                               "residual_edit_norm_served_dtype_last": dc.quantiles(nres_sv[is_last]),
                               "edit_norm_fp32_last": dc.quantiles(n32[is_last]), "edit_norm_served_dtype_last": dc.quantiles(nsv[is_last]),
                               "served_over_fp32_norm_ratio": dc.quantiles(nsv / n32),
                               "frac_changed_served_oproj_site": dc.quantiles(so[is_last]),
                               "frac_changed_served_residual": dc.quantiles(sr[is_last]),
                               "edit_ulps_max_residual": dc.quantiles(ulps[is_last]),
                               "signed_factor_vs_reference": dc.quantiles(cfac)}
            agg = dc.per_prompt_over_layers(ratio_last)
            agg_rows = dc.per_prompt_over_layers(ratio_rows)
            eff = dc.effective_layers(surv)
            eff["per_layer_realised_dose_median"] = {int(l): float(v.to(torch.float64).median()) for l, v in ratio_last.items()}
            eff["site"] = ("the residual stream at the last prompt row: the served o_proj-site cast AND, where the "
                           "architecture re-normalises the attention output, the norm, must both leave the sum changed")
            gate_all = dc.gate_verdict(torch.cat(cs), torch.cat(resids), torch.cat(nrs), s, s_ref)
            gate_last = dc.gate_verdict(torch.cat([x[is_last] for x in cs]), torch.cat([x[is_last] for x in resids]),
                                        torch.cat([x[is_last] for x in nrs]), s, s_ref)
            lora_result[cell_id]["rungs"][view] = {
                "view": view, "nominal_dose": s, "dose_unit": "lora_write_basis_multiplier", "staged_id": r["staged_id"],
                "object_sha256": r["object_sha256"], "layers": r["layers"], "rank": r["rank"],
                "realised_dose_median": float(agg["median_over_layers"].median()),
                "realised_dose": dc.three_way(agg["median_over_layers"], is_harmful),
                "realised_dose_max_layer": dc.three_way(agg["max_over_layers"], is_harmful),
                "realised_dose_all_rows": dc.three_way(agg_rows["median_over_layers"], is_harmful),
                "signed_realised_dose_median": float(agg["median_over_layers"].median()) * (1 if s >= 0 else -1),
                "effective_layers": eff, "gate": gate_all, "gate_last_row_only": gate_last, "by_layer": by_layer}
        lora_result[cell_id]["norm_placement"] = dc.json_ready(norm_placement)
        lora_result[cell_id]["reference_norms_by_layer"] = {
            l: {"resid_pre_token_rows": dc.three_way(torch.cat(h_pre_norm[l]), is_harmful[rp]),
                "resid_pre_last_row": dc.three_way(torch.cat(h_pre_norm[l])[is_last], is_harmful),
                "resid_post_token_rows": dc.three_way(torch.cat(h_post_norm[l]), is_harmful[rp]),
                "oproj_input_norm_token_rows": dc.three_way(torch.cat(x_norm[l]), is_harmful[rp]),
                "oproj_output_norm_token_rows": dc.three_way(torch.cat(o_out_norm[l]), is_harmful[rp])} for l in lora_layers}

    # 7. write
    provenance["elapsed_s"] = round(time.time() - t0, 1)
    capture = {"schema": dc.SCHEMA, "target": target, "pool_sha256": args.pool_sha256, "model_dtype": str(model_dtype),
               "is_harmful": is_harmful, "prompt_tokens": torch.tensor(lens),
               "lora": {"layers": lora_layers, "row_prompt": rp, "row_pos": torch.tensor(row_pos) if row_pos else None,
                        "row_is_last": is_last,
                        "h_pre_norm": {l: torch.cat(h_pre_norm[l]) for l in lora_layers},
                        "h_post_norm": {l: torch.cat(h_post_norm[l]) for l in lora_layers},
                        "x_norm": {l: torch.cat(x_norm[l]) for l in lora_layers},
                        "oproj_out_norm": {l: torch.cat(o_out_norm[l]) for l in lora_layers},
                        "norm_placement": dc.json_ready(norm_placement),
                        "reads_reference": {cell: {l: torch.cat(reads_ref[cell][l]) for l in lora_layers} for cell in lora_cells},
                        "edits": {cell: {view: {l: {k: torch.cat(v) for k, v in st.items()} for l, st in per.items()}
                                         for view, per in views.items()} for cell, views in lora_edit.items()},
                        "cells": dc.json_ready(lora_cells)},
               "alphasteer": {"hook_layers": as_hook_layers, "h_last": {l: torch.cat(as_h_last[l], 0) for l in as_hook_layers},
                              "h_norm_rows": {l: torch.cat(as_h_norm_rows[l]) for l in as_hook_layers},
                              "row_prompt": torch.tensor(as_row_prompt) if as_row_prompt else None,
                              "rungs": dc.json_ready(as_arm["rungs"]) if as_arm else None},
               "provenance": provenance}
    cap_path = args.out_dir / f"capture_{target}.pt"
    torch.save(capture, cap_path)
    smp_path = args.out_dir / f"capture_sample_{target}.pt"
    torch.save({"schema": dc.SCHEMA, "target": target, "model_dtype": str(model_dtype), "pool_sha256": args.pool_sha256,
                "lora_layers": lora_layers, "alphasteer_hook_layers": as_hook_layers, **sample,
                "lora_cells": dc.json_ready(lora_cells), "alphasteer_rungs": dc.json_ready(as_arm["rungs"]) if as_arm else None},
               smp_path)
    provenance["capture_sha256"] = dc.sha256_file(cap_path)
    provenance["capture_sample_sha256"] = dc.sha256_file(smp_path)
    out = {"schema": dc.SCHEMA, "target": target, "lora": dc.json_ready(lora_result), "alphasteer": dc.json_ready(as_result),
           "provenance": provenance}
    gates = [(f"lora/{c}/{v}", r["gate"]["pass"]) for c, cr in lora_result.items() for v, r in cr["rungs"].items()]
    if as_result:
        gates += [(f"alphasteer/{s}", r["gate"]["pass"]) for s, r in as_result["rungs"].items()]
    eff_all = {f"lora/{c}/{v}": r["effective_layers"] for c, cr in lora_result.items() for v, r in cr["rungs"].items()}
    if as_result:
        eff_all.update({f"alphasteer/{s}": r["effective_layers"] for s, r in as_result["rungs"].items()})
    zero = sorted(k for k, e in eff_all.items() if e["effective_layer_count"] == 0)
    narrowed = {k: (e["effective_layer_count"], e["nominal_layer_count"]) for k, e in eff_all.items()
                if 0 < e["effective_layer_count"] < e["nominal_layer_count"]}
    out["gate_summary"] = {"all_pass": all(p for _, p in gates), "rungs": dict(gates)}
    out["effective_layer_summary"] = {
        "by_arm": {k: {kk: e[kk] for kk in ("nominal_layer_count", "effective_layer_count",
                                            "effective_layer_count_half", "active_layers", "inactive_layers")}
                   for k, e in eff_all.items()},
        "zero_effective_layers": zero, "narrowed": narrowed}
    p = args.out_dir / f"dose_{target}.json"
    p.write_text(json.dumps(out, indent=1) + "\n")
    log(f"wrote {p} ({cap_path.stat().st_size / 1e6:.1f} MB capture); gates: {out['gate_summary']}")
    def _line(tag, r):
        e = r["effective_layers"]
        log(f"  {tag}: nominal {r['nominal_dose']:+.3g} realised_median {r['realised_dose_median']:.4g} "
            f"layers {e['effective_layer_count']}/{e['nominal_layer_count']} effective "
            f"({e['effective_layer_count_half']} at >=half) gate {'PASS' if r['gate']['pass'] else 'FAIL'} exact={r['gate']['exact']}")
    for c, cr in lora_result.items():
        for v, r in cr["rungs"].items():
            _line(f"lora {c} {v}", r)
    if as_result:
        for sid, r in as_result["rungs"].items():
            _line(f"alphasteer {sid}", r)
    log_effective_layer_summary(target, out["effective_layer_summary"])
    for c, cr in lora_result.items():
        for l, r in (cr.get("norm_placement") or {}).items():
            if r.get("declared_post_attention_norm"):
                log(f"  note: layer {l} ({r['block_class']}) re-normalises the attention output through "
                    f"`{r['declared_post_attention_norm']}` before the residual add — the headline dose is the "
                    "post-norm residual perturbation, not the o_proj-site edit")
        break
    return 0 if out["gate_summary"]["all_pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
