"""MATH-500 (thinking ON, 16,384) reports -> registry (``adapters/math500.py``; source ``evalcheck/math500``).

Reproduces ``paperlib.lane_math500_r4.load`` (pin 40997227, ``lane_rule="view"``) over the steer-pins r1 report
``$GRADE/tinycap/math500/final_steer_pins_r1/report.json`` (published pin 55e3cbdd + ext1 ac67e138 + llama + steer waves,
admitted per leg): one row per primary leg of each primary comparison (accuracy = the leg's conditional accuracy over its scored
draws; the contrast = the producer's percentile bootstrap over items, B = 10,000 seed 0, exact McNemar p — every number the
PRODUCER's), the steer legs (AlphaSteer / Arditi at their operating points) on the paper lane by object identity, and the
``design.status_by_row`` cells the record files as not analysed as ``pending`` dispositions. The superseded revision-4 report
(``final_ext1_r4/report_evalcheck_published.json``) is ingested with the same reader and its runs marked ``superseded`` by the r1
record (the paper's ``_superseded_crosscheck``: identical points on every shared leg, checked here too).

``at_cap_share`` is the UNSCORED share ``n_censored / (n_scored + n_censored)`` (revision 5's correction); loops from the leg's
``at_cap.n_loop``. Sensitivity sampling rows (``general`` / ``battery``) are separate instruments and are not primary here.
"""
from __future__ import annotations

import os
import re
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from .base import AdapterContext, AdapterRefusal, Input, RecordBatch, SourceStateRow, check_input, site_root
from . import declared_common as DC
from .declared_common import ARMS, EVAL_LABEL, HOME, MODELS, blank_row, fnum, read_json

M_ROOT = site_root("$GRADE/tinycap/math500")
REPORT = os.path.join(M_ROOT, "final_steer_pins_r1", "report.json")
SUPERSEDED_REPORT = os.path.join(M_ROOT, "final_ext1_r4", "report_evalcheck_published.json")
STORE_PIN = "55e3cbdd0bfb5d53f8d62c50a099a3c1c105100b"
EXT1_PIN = "ac67e138b90aaefb3d7b8b9adeb5ae520404158a"
SOURCE, EVAL_ID, CAP, MODE, TASK = "evalcheck/math500", "math500", 16384, "thinking_card", "inspect/math"
RESOLVED_WORDS = ("lower than bare", "higher than bare")
CHAT_CKPT = ("nemotron3nano4b", "glm46vflash")
SIDE_KINDS = ("bare", "prompted", "prompted_constitution", "lora_ctl2", "lora_mix", "object", "alphasteer", "arditi")
STEER_KINDS = ("alphasteer", "arditi")
MODE_BY_MODEL = {"llama8b": "card"}
#: The pinned reader's judge string (`paper/paperlib/lane_math500_r4.py` at c53ae173, the `judge=` of its measured row: the grader,
#: its framework, and the denominator rule spelled in full). The paper keys `judge` in its identity, so the adapter's park must
#: be this string byte-for-byte — the short spelling it carried before matched none of the pin's 27 keys.
MATH500_JUDGE_LEGACY = ("inspect/math grader ({framework}, sympy exact match); an unscored draw (cap inside an unfinished think block, "
                        "or otherwise unterminated) leaves the denominator (primary branch)")
W4_PAPER_LABEL = {"alphasteer": "alphasteer_rebake", "arditi": "arditi"}
#: where a served store's entry (`legs[*]`) carries a path on the producing cluster's shared root, in preference order (the gsm8k reader's list)
SUBSTRATE_WITNESS_KEYS = ("ledger.binding.routing_table", "launch.serving_config", "launch.standup_dir", "launch.excluded.chat_template.value")


def _pp(x) -> float:
    return float("nan") if x is None else 100.0 * float(x)


def _tag(model: str, view_legacy: str) -> str:
    spec = MODELS.get(model)
    if view_legacy == "it":
        return spec["view"] if spec else f"{model}_it"
    if view_legacy != "main":
        raise AdapterRefusal(f"{model}", f"leg view {view_legacy!r} is neither 'main' nor 'it'")
    return (spec or {}).get("transfer_view") or model


def _lane_kind_legacy(model: str, view_legacy: str) -> str:
    tag, spec = _tag(model, view_legacy), MODELS.get(model)
    if spec is not None:
        if tag == spec["view"]:
            return spec["lane_kind"]
        if tag == spec.get("transfer_view"):
            return "base→instruct"
    if tag.endswith("_it"):
        return "instruct-direct"
    if model in CHAT_CKPT:
        return "chat-ckpt"
    return "base→instruct"


def _decode(model: str, sampling: dict, mode: str = MODE) -> str:
    row = sampling.get(model) or {}
    names = [("temperature", "T"), ("top_p", "top_p"), ("top_k", "top_k"), ("min_p", "min_p"), ("presence_penalty", "presence_penalty"), ("repetition_penalty", "repetition_penalty")]
    parts = [f"{short}={row[k]}" for k, short in names if row.get(k) is not None]
    head = "thinking ON" if mode == MODE else "no thinking switch (card row)"
    return f"{head}, k=1, cap {CAP} (one pass, no rung); {mode} sampling row: " + (", ".join(parts) or "unrecorded")


def _gate_failures(d: dict) -> List[str]:
    bad = []
    for name, g in (d.get("gates") or {}).items():
        if isinstance(g, dict) and "status" in g and str(g["status"]).upper() != "PASSED":
            bad.append(f"{name}={g['status']}")
    pg = (d.get("publication_gate") or {}).get("status")
    if pg is not None and str(pg).upper() != "PASS":
        bad.append(f"publication_gate={pg}")
    return bad


class Math500Adapter:
    name = "math500"
    #: 2 = `judge` spelled in full (`MATH500_JUDGE_LEGACY`); the shared emitter's `paper_dose` park rides in
    EMISSION_VERSION = DC.emission_version(3)   # 3: the substrate derived per served store from its ledger binding / launch paths, never a constant (2026-09-21)
    covers_sources: Tuple[str, ...] = (SOURCE,)
    rng_scope = "none"
    allow_symlinks = False
    campaign = "sdfbase_full22"

    def __init__(self, report: str = REPORT, superseded_report: Optional[str] = SUPERSEDED_REPORT, store_pin: str = STORE_PIN, ext1_pin: str = EXT1_PIN):
        self.report, self.superseded_report, self.store_pin, self.ext1_pin = report, superseded_report, store_pin, ext1_pin
        self.roots = (M_ROOT,)
        self.last_report: Optional[DC.EmitReport] = None
        self.last_facts: Dict[str, object] = {}

    def discover(self) -> List[Input]:
        if not os.path.lexists(self.report):
            raise AdapterRefusal(self.report, "input missing")
        out = [check_input(self.report, role="report", allow_symlink=self.allow_symlinks)]
        if self.superseded_report and os.path.lexists(self.superseded_report):
            out.append(check_input(self.superseded_report, role="report", allow_symlink=self.allow_symlinks))
        return out

    def comparison_group(self, inp: Input) -> str:
        # the current report and the superseded revision share one group (the superseded rows name the current runs)
        return os.path.basename(os.path.dirname(self.report))

    def long_rows(self, report: Optional[str] = None, superseded_by: Optional[str] = None) -> Tuple[List[dict], Dict[str, object]]:
        report = report or self.report
        d = read_json(report)
        pin = str((d.get("provenance") or {}).get("store_pin") or d.get("sha") or "")
        if not pin or not self.store_pin.startswith(pin):
            raise AdapterRefusal(report, f"the report's store pin is {pin!r}, not {self.store_pin[:7]}")
        failed = _gate_failures(d)
        if failed:
            raise AdapterRefusal(report, f"the report records gates not PASSED ({failed})")
        if d.get("task") != TASK:
            raise AdapterRefusal(report, f"task is {d.get('task')!r}, not {TASK}")
        tip = str(d.get("analysis_code_sha") or "")
        legs = d["legs"]
        sampling_all = d.get("sampling_rows") or {}
        leg_pins_legacy = (d.get("store_pins") or {}).get("pins") or {}
        masked_by_leg_legacy = ((d.get("masked_primary") or {}).get("by_leg") or {})
        when = DC.mtime_utc(report)
        rows: List[dict] = []
        words: Dict[str, str] = {}
        for comp in d.get("comparisons", []):
            if not (comp.get("primary") and comp.get("usable")):
                continue
            model = comp["model_key"]
            mode = MODE_BY_MODEL.get(model, MODE)
            if comp.get("mode") != mode:
                raise AdapterRefusal(report, f"{comp.get('name')}: a PRIMARY comparison in sampling mode {comp.get('mode')!r}; the paper's instrument for {model} is the {mode} row")
            sampling = sampling_all.get(mode) or {}
            prim = comp["estimands"]["primary"]
            stats, verdicts, mcn = prim["stats"], comp.get("verdicts") or {}, comp.get("mcnemar") or {}
            n_items_design = int(prim["n_items"])
            bare_key = comp["leg_keys"]["bare"]
            for leg_kind, lk in comp["leg_keys"].items():
                if leg_kind not in SIDE_KINDS:
                    raise AdapterRefusal(report, f"{lk}: leg kind {leg_kind!r} unknown; refusing to guess its arm")
                a = prim["acc_by_leg"].get(leg_kind)
                leg_legacy = legs.get(lk)
                if a is None or leg_legacy is None:
                    raise AdapterRefusal(report, f"{lk}: named in leg_keys but missing from acc_by_leg / legs")
                cell_legacy = leg_legacy["cell"]
                arm_legacy = ARMS.get(cell_legacy) or (DC.ARMS_BY_KEY[W4_PAPER_LABEL[cell_legacy]] if cell_legacy in W4_PAPER_LABEL and leg_kind in STEER_KINDS else None)
                if arm_legacy is None:
                    raise AdapterRefusal(report, f"{lk}: cell id {cell_legacy!r} is not in the paper's arm map")
                view_legacy = leg_legacy.get("view")
                if leg_kind in STEER_KINDS:
                    lane, tag = (MODELS.get(model) or {}).get("lane_kind") or ("chat-ckpt" if model in CHAT_CKPT else "instruct-direct"), (MODELS.get(model) or {}).get("view", model)
                else:
                    lane, tag = _lane_kind_legacy(model, view_legacy), _tag(model, view_legacy)
                n_scored, n_cens, n_corr = int(a["n_scored"]), int(a["n_censored"]), int(a["n_correct"])
                n_items = n_scored + n_cens
                n_masked = len(masked_by_leg_legacy.get(lk) or [])
                if n_items != n_items_design - n_masked:
                    raise AdapterRefusal(report, f"{lk}: scored + unscored = {n_items}, population {n_items_design} minus {n_masked} masked")
                if abs(n_corr / n_scored - float(a["point"])) > 1e-12:
                    raise AdapterRefusal(report, f"{lk}: n_correct/n_scored disagrees with the filed point")
                s, ac = leg_legacy["summary"], leg_legacy.get("at_cap") or {}
                n_loop = int(ac.get("n_loop") or 0)
                unscored_share = n_cens / n_items
                ident = leg_legacy.get("identity") or {}
                served = ident.get("served") or {}
                thinking = (leg_legacy.get("outcomes_header") or {}).get("thinking") or ident.get("thinking_mode")
                thinking = thinking if thinking in ("on", "off", "not_available") else ("not_available" if model == "llama8b" else "on")
                sha = ident.get("object_artifact_sha256")
                sha = sha.lower() if isinstance(sha, str) and re.fullmatch(r"[0-9a-fA-F]{64}", sha) else None
                tree = str(leg_legacy.get("producer_tree") or leg_pins_legacy.get(lk) or ident.get("producer_tree_sha") or "")
                row = blank_row(source=SOURCE, model=model, lane_kind=lane, view=tag, dose=1.0, cell_id=cell_legacy, arm=arm_legacy.key, family=arm_legacy.family, object_type=arm_legacy.object_type,
                                component="capability", eval_id=EVAL_ID, eval=EVAL_LABEL[EVAL_ID], metric="acc", threshold=float("nan"), bon_n=float("nan"),
                                value=float(a["point"]), k=n_corr, n=n_scored, ci_lo=float(a["ci_low"]), ci_hi=float(a["ci_high"]),
                                at_cap_share=unscored_share, loop_share=n_loop / n_items, degenerate_share=n_loop / n_items, decode_cap_tokens=CAP,
                                decode=_decode(model, sampling, mode), judge=MATH500_JUDGE_LEGACY.format(framework=s.get("framework")),
                                status="measured", degenerate_win=False, anchor_degenerate=False, thinking_mode=thinking, object_sha256=sha if sha else float("nan"),
                                serving_window_tokens=(int(served["max_model_len"]) if served.get("max_model_len") is not None else float("nan")),
                                provenance=f"{report} ›comparisons[{comp['name']}].estimands.primary.acc_by_leg[{leg_kind}] + legs[{lk}]; producer tree {tree[:8]}")
                row.update(_when=when, _code_sha=tree if len(tree) == 40 else "unverified", _code_sha_reason="the leg names no producer tree sha", _model_revision=ident.get("model_revision"),
                           _object_id=sha, _store_uri=leg_legacy.get("cell_dir"), _substrate_path="",
                           # the substrate PER ENTRY from the standup's routing table / serving config the entry's ledger binding and launch
                           # fields carry on a cluster's shared root (one llama entry of r1 ran on cluster B); `cell_dir` is the grading host mirror
                           _substrate_derived=DC.D.substrate_derived_from_record(
                               {"ledger": leg_legacy.get("ledger") or {}, "launch": leg_legacy.get("launch") or {}}, "report.json:legs[*]",
                               prefer=SUBSTRATE_WITNESS_KEYS),
                           _substrate_reason="the entry's ledger binding / launch fields name no path on a cluster's shared root",
                           _producer_key=DC.store_run_key(leg_legacy.get("cell_dir"), str(leg_legacy.get("cell_key") or lk)), _object_key=("none" if cell_legacy == "none" else (f"{cell_legacy}@{ident.get('intervention_hash')}" if ident.get("intervention_hash") else cell_legacy)),
                           _instrument={"harness": "inspect_ai", "harness_version": tip or "unverified", "harness_config_hash": ident.get("harness_config_hash"), "token_cap_tokens": CAP,
                                        "decode": {k: v for k, v in (sampling.get(model) or {}).items() if k in DC.D.DECODE_KEYS}, "decode_source": "model_card", "decode_seed_rule": "per_item",
                                        "thinking_mode": thinking, "serving_path": ("lens" if served.get("reduction_mode") == "stacked" else "unverified"), "serving_identity": f"reduction_mode={served.get('reduction_mode')}",
                                        "intervention_hash": ident.get("intervention_hash"), "scoring_rule": "sympy_equiv", "instrument_config_hash": f"math500_{mode}_16384_onepass", "fit_positions": "all"},
                           _judge=None, _n_items_planned=n_items_design, _key=f"{SOURCE}|{os.path.basename(os.path.dirname(report))}|{model}|{leg_kind}",
                           _superseded_by_key=(f"{SOURCE}|{os.path.basename(os.path.dirname(self.report))}|{model}|{leg_kind}" if superseded_by else None),
                           _extra={"leg_kind": leg_kind, "wave": leg_legacy.get("wave"), "store_pin": leg_pins_legacy.get(lk), "analysis_tip": tip, "n_censored": n_cens, "raw_at_cap": s.get("n_at_cap"), "mode": mode},
                           _degeneration={"unit": "items", "counts": {"at_token_cap": (n_cens, n_items), "classifier": (n_loop, n_items), "union5": (n_loop, n_items)}},
                           _strength_coefficient=(1.0 if arm_legacy.object_kind not in ("none", "system_instruction") else None),
                           _is_operating_point=(1 if arm_legacy.object_kind not in ("none", "system_instruction") else None),
                           _superseded=superseded_by)
                if leg_kind in STEER_KINDS:
                    row["_strength_nominal_source"] = "derived"
                if leg_kind == "bare":
                    rows.append(row)
                    continue
                key = f"{leg_kind}_minus_bare"
                st_, v = stats.get(key), verdicts.get(key)
                if st_ is None or v is None:
                    raise AdapterRefusal(report, f"{lk}: no {key} contrast / verdict in the comparison")
                words[f"{model}/{arm_legacy.key}"] = v["word"]
                p = ((mcn.get(key) or {}).get("primary") or {}).get("p_value")
                row.update(delta=_pp(st_["point"]), delta_lo=_pp(st_["ci_low"]), delta_hi=_pp(st_["ci_high"]), delta_p=float(p) if p is not None else float("nan"),
                           delta_method="percentile bootstrap over items (B=10,000, seed 0) vs the same comparison's bare leg, primary (conditional) branch; delta_p = exact two-sided McNemar on complete cases",
                           suggestive=(v["word"] not in RESOLVED_WORDS), note=f"verdict (D5 P2-2): {v['word']}; UNSCORED {n_cens}/{n_items}",
                           _anchor_key=f"{SOURCE}|{os.path.basename(os.path.dirname(report))}|{model}|bare", _anchor_kind="same_standup_none", _n_pairs=n_items_design)
                rows.append(row)
        # steer-wave cells the record files as not analysed -> pending dispositions
        pending = []
        for rk, srow in ((d.get("design") or {}).get("status_by_row") or {}).items():
            stt = str(srow.get("status"))
            if stt == "required":
                continue
            cell_legacy, model, mode = srow.get("cell"), srow.get("model_key"), srow.get("mode")
            if cell_legacy not in W4_PAPER_LABEL or mode != MODE_BY_MODEL.get(model, MODE):
                continue
            arm_legacy = DC.ARMS_BY_KEY[W4_PAPER_LABEL[cell_legacy]]
            spec = MODELS.get(model) or {}
            r = blank_row(source=SOURCE, model=model, lane_kind=spec.get("lane_kind"), view=spec.get("view", model), dose=1.0, cell_id=cell_legacy, arm=arm_legacy.key, family=arm_legacy.family,
                          object_type=arm_legacy.object_type, component="capability", eval_id=EVAL_ID, eval=EVAL_LABEL[EVAL_ID], metric="acc", threshold=float("nan"), bon_n=float("nan"),
                          decode_cap_tokens=CAP, decode=_decode(model, sampling_all.get(mode) or {}, mode), status=f"{'held' if stt == 'held' else 'pending'}: the record files {rk} as {stt} ({srow.get('reason')})",
                          note=str(srow.get("reason")), thinking_mode=("not_available" if model == "llama8b" else "on"), provenance=f"{report} ›design.status_by_row[{rk}]")
            rows.append(r)
            pending.append(rk)
        facts = dict(store_pin=pin, analysis_tip=tip, words=words, pending=pending, n_rows=len(rows))
        return rows, facts

    def read(self, ctx: AdapterContext, group: str, inputs: Sequence[Input]) -> Iterator[RecordBatch]:
        # the current report first, the superseded revision after it (its rows name the current run they are superseded by); one batch
        rows: List[dict] = []
        facts: Dict[str, object] = {}
        for inp in sorted(inputs, key=lambda i: (i.uri == self.superseded_report, i.uri)):
            superseded = None
            if inp.uri == self.superseded_report:
                superseded = f"revision-4 report superseded by {self.report} (steer-pins r1, EvalCheck 2026-09-15 02:17Z): identical points on every shared leg"
            r, f = self.long_rows(inp.path, superseded_by=superseded)
            rows += r
            facts[inp.uri] = f
        self.last_facts[group] = facts
        batch = RecordBatch(f"{self.name}:{group}", group)
        rep = DC.emit_rows(batch, ctx, rows, adapter=self.name, campaign=self.campaign, reader="lane_math500_r4.load",
                           seed_provenance="producer statistics read (B=10000 seed 0 per the report), none recomputed", code_sha="unverified",
                           code_sha_reason="the producer tree sha rides each leg")
        self.last_report = rep
        yield batch

    def source_states(self, ctx: AdapterContext, inputs: Sequence[Input]) -> List[SourceStateRow]:
        out = []
        for inp in inputs:
            d = read_json(inp.path)
            out.append(SourceStateRow(f"{self.name}:{os.path.basename(os.path.dirname(inp.uri))}", inp.uri, "$.legs", 1, 1, sorted((d.get("legs") or {}).keys()), inp.sha256))
        return out


def register() -> None:
    from .base import register as _register
    _register(Math500Adapter.name, lambda: Math500Adapter())
