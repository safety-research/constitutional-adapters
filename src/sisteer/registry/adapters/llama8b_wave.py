"""The Llama-3.1-8B wave's own records -> registry (``adapters/llama8b_wave.py``; source ``llama8b_lane``).

Reproduces ``paperlib.lane_llama8b.load`` (pin 40997227): Petri from the lane's Petri-only export
``$STAGE/sdfbase/llama8b/wave/stage/transfer_stats_full22_llama8b.json`` + its per-scenario sidecar
``petri_stats/petri_full22_stats_llama8b.json`` (one ``petri_pooled`` row per cell — pooled behaviour mean, delta = the paired
mean difference over the shared (behaviour, scenario) items vs the view's own bare ``none`` with a percentile bootstrap of
10,000 draws at seed 0, the reader's rule — plus six per-behaviour rows with the producer's ``petri_vs_bare`` interval + exact
sign test); MMLU from the served-controls legs' per-item letter-prefill records mirrored at ``~/.cache/paperlib-llama8b/controls``
(accuracy recomputed and equal to ``gen_stats.mmlu`` or the cell is refused; paired bootstrap + exact McNemar vs the view's
``none``). The view sidecar (``view_sidecar.json``) and ``positive_control_verdicts.tsv`` supply windows, jobs and verdicts;
the ungraded stores on cluster B the reader files as ``pending`` become dispositions (the store roots are recorded in the
disposition's source_ref; nothing on the cluster is read).

Object identity: ``wave/stage/.src_<lane>__<cell>.json`` (fitted LoRAs) and ``stage_plan.json`` (the derived subtraction) give
the served object's sha256, cross-checked against the 16-hex prefix each leg stamped in ``evidence.json``.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .base import AdapterContext, AdapterRefusal, Input, RecordBatch, SourceStateRow, check_input, store_glob, site_root
from . import _items_common as IC
from . import declared_common as DC
from .declared_common import (ARMS, EVAL_LABEL, HOME, MULTITURN_ATTACKS, PETRI_BEHAVIOURS, STATIC_FAMILIES, blank_row, cohen_h, fnum, finite,
                              mcnemar_exact, read_json, utc_ms, wilson)

LLAMA8B_ROOT = site_root("$STAGE/sdfbase/llama8b")
STAGE = os.path.join(LLAMA8B_ROOT, "wave", "stage")
MMLU_CACHE = os.path.join(HOME, ".cache", "paperlib-llama8b", "controls")
#: cluster B's wave root is the one its runs recorded: the view sidecar keys every serving and store dir by its path under that
#: root, so the adapter reads the root from those keys (`recorded_root`) and files every cluster path under it as recorded, whoever
#: runs the reader. REMOTE_ROOT is the root's shape on cluster B's per-user shared root (`<user>` the account the runs were recorded
#: under); nothing is filed under it
REMOTE_HOST, REMOTE_CAMPAIGN = "cluster_b", "/exp/sdfbase_full22"
REMOTE_ROOT = f"/workspace/<user>{REMOTE_CAMPAIGN}"
SOURCE, MODEL = "llama8b_lane", "llama8b"
VIEWS = {
    "llama8b": dict(view="llama8b", lane_kind="base→instruct", short="main"),
    "llama8b_it": dict(view="llama8b_it", lane_kind="instruct-direct", short="it"),
    "llama8b_shortconst": dict(view="llama8b_shortconst", lane_kind="instruct-direct", short="shortconst"),
    "llama8b_fullconst": dict(view="llama8b_fullconst", lane_kind="instruct-direct", short="fullconst"),
}
SERVED = {
    "llama8b": ["none", "constv2ctl2-full__lora_r4_ce", "constv2-mix-full__lora_r4_ce", "constv2-mix-full__lora_r4_ce_minus_ctl", "prompted__safety_refusal"],
    "llama8b_it": ["none", "constv2ctl2-full__lora_r4_ce", "constv2-mix-full__lora_r4_ce", "constv2-mix-full__lora_r4_ce_minus_ctl", "prompted__safety_refusal"],
    "llama8b_shortconst": ["none", "prompted__si_constitution"],
    "llama8b_fullconst": ["none", "prompted__si_constitution_full"],
}
COMPONENTS_BY_SPELLING = {
    "llama8b": ("static", "bon", "multi-turn", "over-refusal", "capability", "petri"),
    "llama8b_it": ("static", "bon", "multi-turn", "over-refusal", "capability", "petri"),
    "llama8b_shortconst": ("static", "bon", "multi-turn", "over-refusal", "capability", "petri"),
    "llama8b_fullconst": ("static", "bon", "over-refusal", "capability"),
}
PETRI_JUDGE = "auditor claude-sonnet-5, judge claude-opus-5 (Petri)"
PETRI_DECODE = "sampled at the target's shipped generation config (engine-filled T=0.6, top_p=0.9: the runner sends max_tokens 1,024 only)"
MMLU_DECODE = "letter-logprob prefill (teacher-forced argmax over the top-64 logprobs; sampler-moot, no decode)"
N_BOOT, SEED = 10_000, 0
#: the view -> registry model for the SERVED checkpoint: every llama view serves the Instruct mirror (config.MODELS llama8b)
MODEL_ID = "llama8b_it"
#: The substrate the reader knows for the source as a whole (`assumed_default`, the basis filed) when the sidecar names no serving /
#: store dir for a run: the Llama runs were served on cluster B (`view_sidecar.json` is written from host `cluster_b` and keys every
#: serving dir and store dir by its path under `/workspace/<user>/`; the stage dir's launch logs are `launch_real_cluster_b_*`).
#: A run the sidecar DOES key derives its substrate from that key (a per-run marker), source `derived`.
SUBSTRATE_DEFAULT = ("cluster_b", "the Llama runs were served on cluster B: view_sidecar.json is written from host cluster_b and keys every serving dir "
                     "and store dir under the /workspace/<user> shared root (launch_real_cluster_b_* in the stage dir); this run's dir is not "
                     "among the sidecar's keys, so the source's default stands")


def _substrate_of_sidecar_key(key: str, present: bool, what: str) -> Optional[Tuple[str, str]]:
    """The ``derived`` claim for a run whose serving / store dir `view_sidecar.json` keys by its shared-root path; None when the sidecar
    carries no entry for it (the reader's default then stands as an assumed default)."""
    if not present:
        return None
    root = DC.D.substrate_root_of_path(key)
    if root is None or root[0] == "grading_host":
        return None
    return root[0], f"view_sidecar.json:{what} keys this run's dir {key[:72]}{'…' if len(key) > 72 else ''} on the {root[1]}"


def recorded_root(sidecar_path: str, serving: Mapping[str, object], stores: Mapping[str, object]) -> str:
    """The wave root the runs recorded: the one prefix ending in REMOTE_CAMPAIGN that the view sidecar's serving and store dir keys
    share. The sidecar is refused when its keys name no such root, or more than one: every cluster path this source files would
    then be a guess."""
    roots = sorted({k[:i + len(REMOTE_CAMPAIGN)] for k in (*serving, *stores) if (i := str(k).find(REMOTE_CAMPAIGN + "/")) > 0})
    if len(roots) != 1:
        raise AdapterRefusal(sidecar_path, f"the serving_dirs / store_dirs keys name {len(roots)} wave roots ending in {REMOTE_CAMPAIGN} "
                             f"({', '.join(roots[:3]) or 'none'}); the cluster root the runs recorded must be exactly one")
    return roots[0]


def _arm_legacy(cid: str):
    return ARMS.get(cid)


def _base_row(tag: str, cid: str, component: str, eval_id: str, metric: str) -> dict:
    v = VIEWS[tag]
    a = _arm_legacy(cid)
    return blank_row(source=SOURCE, model=MODEL, lane_kind=v["lane_kind"], view=v["view"], dose=1.0, cell_id=cid, arm=a.key if a else None,
                     family=a.family if a else None, object_type=a.object_type if a else None, component=component,
                     eval_id=eval_id, eval=EVAL_LABEL.get(eval_id, eval_id), metric=metric, threshold=float("nan"), bon_n=float("nan"),
                     thinking_mode="not_available")


def _paired(d: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> Tuple[float, float, float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    bs = d[idx].mean(axis=1)
    sd = float(d.std(ddof=1)) if len(d) > 1 else float("nan")
    dz = float(d.mean() / sd) if sd and sd == sd and sd > 0 else float("nan")
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), dz


class Llama8bWaveAdapter:
    name = "llama8b_wave"
    #: the shared emitter's park change rides in (`paper_dose`) — see `declared_common.EMITTER_VERSION`
    EMISSION_VERSION = DC.emission_version(3)   # 2: the distilled cell's arm / label (difference_relu_distilled);
    #                                             3: the substrate derived from the sidecar's dir keys, else the source's assumed default (2026-09-21)
    covers_sources: Tuple[str, ...] = (SOURCE,)
    #: the model's own reader of the Petri store the r2 export also reads; this source OWNS those measurements
    #: (ownership.MULTI_SOURCE_OWNERSHIP; paperlib's rule of 2026-09-14 — `load_r2` drops the export's Llama Petri rows at read —
    #: and the finding of 2026-09-21 01:30Z that the export's `n` there counts scenarios PLANNED where this reader
    #: counts scenarios SCORED) and the export stands as the secondary — stated above the export's PRIORITY_PAPER_SOURCE so
    #: `ownership.check_priorities` pins owner > secondary rather than leaving the source-id tiebreak to decide (before this the
    #: adapter declared no priority, an undeclared 0). A key-level rank too: where this source and another carry ONE key, this
    #: one now outranks the export; on the live registry the two share no key (verified 2026-09-21).
    priority = IC.PRIORITY_OWNING_READER
    rng_scope = "none"                # every bootstrap re-seeds default_rng(0) per cell (the reader's _paired)
    allow_symlinks = False
    campaign = "sdfbase_full22"

    def __init__(self, stage: str = STAGE, mmlu_cache: str = MMLU_CACHE):
        self.stage, self.mmlu_cache = stage, mmlu_cache
        self.roots = (stage, mmlu_cache)
        self.last_report: Optional[DC.EmitReport] = None
        self.last_facts: Dict[str, object] = {}

    @property
    def petri_export(self) -> str:
        return os.path.join(self.stage, "transfer_stats_full22_llama8b.json")

    @property
    def petri_sidecar(self) -> str:
        return os.path.join(self.stage, "petri_stats", "petri_full22_stats_llama8b.json")

    @property
    def view_sidecar_legacy(self) -> str:
        return os.path.join(self.stage, "view_sidecar.json")

    @property
    def pcv_tsv(self) -> str:
        return os.path.join(self.stage, "positive_control_verdicts.tsv")

    def discover(self) -> List[Input]:
        out = []
        for p in (self.petri_export, self.petri_sidecar, self.view_sidecar_legacy, self.pcv_tsv):
            if not os.path.lexists(p):
                raise AdapterRefusal(p, "input missing")
            out.append(check_input(p, role="report", allow_symlink=self.allow_symlinks))
        for f in sorted(store_glob(os.path.join(self.stage, ".src_*.json"))) + [os.path.join(self.stage, "stage_plan.json")]:
            if os.path.lexists(f):
                out.append(check_input(f, role="report", allow_symlink=self.allow_symlinks))
        for tag in VIEWS:
            for st in sorted(store_glob(os.path.join(self.mmlu_cache, tag, "cells", "*"))):
                for fn, role in (("rows.jsonl", "rows"), ("gen_stats.json", "report"), ("evidence.json", "report"), ("manifest.json", "report")):
                    p = os.path.join(st, fn)
                    if os.path.lexists(p):
                        out.append(check_input(p, role=role, allow_symlink=self.allow_symlinks))
        return out

    def comparison_group(self, inp: Input) -> str:
        return "wave"

    # -- identity --------------------------------------------------------------------------------
    def object_shas(self) -> Dict[Tuple[str, str], str]:
        out = {}
        lane_tag_legacy = {"base": "llama8b", "itdoc": "llama8b_it"}
        for lane, tag in lane_tag_legacy.items():
            for f in store_glob(os.path.join(self.stage, f".src_{lane}__*.json")):
                cid = os.path.basename(f)[len(f".src_{lane}__"):-len(".json")]
                sha = read_json(f).get("object.pt")
                if isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha):
                    out[(tag, cid)] = sha
        sp = os.path.join(self.stage, "stage_plan.json")
        if os.path.exists(sp):
            for e in read_json(sp):
                if e.get("served") and e.get("object_sha256"):
                    tag = {"main": "llama8b", "it": "llama8b_it"}.get(e.get("view"))
                    if tag:
                        out[(tag, e["cell_id"])] = e["object_sha256"]
        return out

    def sidecar(self) -> Tuple[dict, dict, dict, set]:
        v = read_json(self.view_sidecar_legacy)
        verdicts, fail_jobs = {}, set()
        if os.path.exists(self.pcv_tsv):
            with open(self.pcv_tsv, encoding="utf-8") as fh:
                for row in csv.reader(fh, delimiter="\t"):
                    if len(row) < 5:
                        continue
                    verdicts[row[0]] = dict(job=row[1], sha=row[2], kind=row[3], verdict=row[4].strip())
                    if row[4].strip().startswith("FAIL"):
                        fail_jobs.add(row[1])
        return v.get("serving_dirs", {}), v.get("store_dirs", {}), verdicts, fail_jobs

    # -- Petri -----------------------------------------------------------------------------------
    def petri_rows(self, serving: dict, shas: dict, problems: List[str], run_root: str) -> Tuple[List[dict], dict]:
        ex = read_json(self.petri_export)
        side = read_json(self.petri_sidecar) if os.path.exists(self.petri_sidecar) else None
        written = utc_ms((ex.get("meta") or {}).get("written_at")) if (ex.get("meta") or {}).get("written_at") else DC.mtime_utc(self.petri_export)
        wave_sha_legacy = str((ex.get("meta") or {}).get("wave_sha") or "")
        recs, cells_done = [], {}
        for tag, blk in ex["models"].items():
            if tag not in VIEWS:
                problems.append(f"Petri export names an unknown view tag {tag!r}")
                continue
            serving_key = f"{run_root}/petri/{tag}/serving"
            leg_legacy = serving.get(serving_key, {})
            substrate_derived = _substrate_of_sidecar_key(serving_key, serving_key in serving, "serving_dirs")
            window = leg_legacy.get("serving_window")
            vs_bare = blk.get("petri_vs_bare", {})
            side_cells = ((side or {}).get("tags", {}).get(tag, {}) or {}).get("cells", {})
            none_side = side_cells.get("none", {}).get("behaviors", {}) if side_cells else {}
            for cid, cell in blk["cells"].items():
                p = cell.get("petri", {})
                if p.get("status") != "COMPLETE":
                    r = _base_row(tag, cid, "petri", "petri_pooled", "petri_score")
                    r.update(status=f"pending: Petri cell status {p.get('status')!r} in the lane's export", note=f"{self.petri_export} ›models.{tag}.cells.{cid}.petri",
                             provenance=f"{self.petri_export} ›models.{tag}.cells.{cid}.petri")
                    recs.append(r)
                    continue
                refused = (p.get("judge_refused") or {}).get("total", 0) or 0
                audited = p.get("audited_items")
                behaviours = p.get("behaviors", {})
                sha = shas.get((tag, cid))
                side_cell_legacy = side_cells.get(cid, {}) if side_cells else {}
                side_beh = side_cell_legacy.get("behaviors", {})
                base_reg = dict(_when=written, _code_sha=wave_sha_legacy if len(wave_sha_legacy) == 40 else "unverified",
                                _substrate_derived=substrate_derived, _substrate_default=SUBSTRATE_DEFAULT, _substrate_path="", _object_id=sha,
                                _producer_key=str(p.get("petri_run_key") or f"{tag}/{cid}/petri"), _object_key=(f"{tag}:{p.get('intervention_hash')}" if p.get("intervention_hash") else cid),
                                _instrument={"harness": "petri", "harness_version": wave_sha_legacy or "unverified", "harness_config_hash": side_cell_legacy.get("harness_config_hash"),
                                             "token_cap_tokens": 1024, "decode": {"temperature": 0.6, "top_p": 0.9}, "decode_source": "generation_config", "decode_seed_rule": "none",
                                             "thinking_mode": "not_available", "serving_path": "lens", "serving_identity": side_cell_legacy.get("serving_identity"),
                                             "intervention_hash": p.get("intervention_hash"), "job_id": leg_legacy.get("job_id"), "max_turns": 15, "n_trials": 1,
                                             "attacker_model_id": None, "instrument_config_hash": "petri_full22_as_run", "fit_positions": "all"},
                                _judge=DC.judge_fields(**DC.PETRI_JUDGE),
                                _extra={"petri_run_key": p.get("petri_run_key"), "petri_intervention_hash": p.get("intervention_hash"), "auditor_chain": p.get("auditor_chain"),
                                        "judge_refused": refused, "serving_window": window, "job_id": leg_legacy.get("job_id")},
                                _degeneration={"unit": "transcripts", "applicable": True}, _n_items_planned=p.get("audited_expected"),
                                _strength_coefficient=(1.0 if sha else None), _is_operating_point=(1 if sha else None))
                # pooled
                r = _base_row(tag, cid, "petri", "petri_pooled", "petri_score")
                r.update(eval="Petri (6 behaviours pooled)", value=float(p["pooled_score"]), n=audited, decode=PETRI_DECODE, decode_cap_tokens=1024, judge=PETRI_JUDGE,
                         status="graded", object_sha256=sha if sha else float("nan"), serving_window_tokens=window, petri_run_key=p.get("petri_run_key"),
                         petri_intervention_hash=p.get("intervention_hash"), petri_served_object_sha256=(sha if (sha and cid != "none") else float("nan")),
                         provenance=f"{self.petri_export} ›models.{tag}.cells.{cid}.petri | {self.petri_sidecar} ›tags.{tag}.cells.{cid}")
                r.update(base_reg)
                r.update(_key=f"{SOURCE}|{tag}|{cid}|petri_pooled", _items=[])
                if cid == "none":
                    r.update(note="unsteered reference (this view's own bare Instruct standup)")
                else:
                    items = []
                    for b, bb in side_beh.items():
                        ps, pn = (bb.get("per_scenario") or {}), (none_side.get(b, {}).get("per_scenario") or {})
                        items += [(float(ps[s]) - float(pn[s])) for s in ps if s in pn]
                    if items:
                        d = np.array(items)
                        m, lo, hi, dz = _paired(d)
                        r.update(delta=m, delta_lo=lo, delta_hi=hi, effect_size=-dz if dz == dz else float("nan"),
                                 effect_kind="d_z paired (per cell, (behaviour, scenario) items pooled), sign flipped: positive = LESS of the flagged behaviours",
                                 delta_method=(f"paired mean difference over {len(d)} shared (behaviour, scenario) items vs this view's own unsteered cell, percentile bootstrap "
                                               f"({N_BOOT} draws, seed {SEED})"), note=f"{len(d)} paired items; pooled behaviour mean, higher = more of the flagged behaviours",
                                 _anchor_key=f"{SOURCE}|{tag}|none|petri_pooled", _anchor_kind="same_standup_none", _n_pairs=len(d))
                    else:
                        r.update(note="NO pooled contrast: per-scenario scores unavailable in the sidecar")
                        problems.append(f"Petri {tag}/{cid}: no per-scenario scores for the pooled contrast")
                recs.append(r)
                for b, bb in behaviours.items():
                    rb = _base_row(tag, cid, "petri", b, "petri_score")
                    rb.update(eval=EVAL_LABEL.get(b, b), value=float(bb["mean"]), n=bb.get("n_items"), ci_lo=bb.get("lo"), ci_hi=bb.get("hi"), decode=PETRI_DECODE, decode_cap_tokens=1024,
                              judge=PETRI_JUDGE, status="graded", object_sha256=sha if sha else float("nan"), serving_window_tokens=window,
                              note=f"judge score 1-10, mean of {bb.get('n_items')} scenarios; higher = more of the flagged behaviour",
                              provenance=f"{self.petri_export} ›models.{tag}.cells.{cid}.petri.behaviors.{b} | petri_vs_bare.{cid}.{b}",
                              petri_run_key=p.get("petri_run_key"), petri_intervention_hash=p.get("intervention_hash"))
                    rb.update(base_reg)
                    sb = side_beh.get(b, {})
                    ps = sb.get("per_scenario") or {}
                    rb.update(_key=f"{SOURCE}|{tag}|{cid}|{b}", _ci_method="percentile_bootstrap_scenarios_10000_seed0",
                              _items=[{"item_id": f"{b}/{s}", "metric_id": "petri_score", "value": float(v), "turn_index": -1} for s, v in sorted(ps.items())])
                    vb = (vs_bare.get(cid) or {}).get(b)
                    if cid != "none" and vb:
                        rb.update(delta=vb["delta"], delta_lo=vb["lo"], delta_hi=vb["hi"], delta_p=vb.get("sign_test_p"),
                                  delta_method=(f"paired mean difference over {vb.get('n_paired')} shared scenarios vs this view's own unsteered cell, percentile bootstrap 95% "
                                                f"({vb.get('n_boot')} draws, seed {vb.get('seed')}) + exact two-sided sign test (the producer's petri_vs_bare)"),
                                  _anchor_key=f"{SOURCE}|{tag}|none|{b}", _anchor_kind="same_standup_none", _n_pairs=vb.get("n_paired"))
                        pn = (none_side.get(b, {}).get("per_scenario") or {})
                        d = np.array([float(ps[s]) - float(pn[s]) for s in ps if s in pn])
                        if len(d) > 1:
                            m, _, _, dz = _paired(d)
                            if abs(m - float(vb["delta"])) > 0.051:
                                problems.append(f"Petri {tag}/{cid}/{b}: recomputed delta {m:.3f} vs producer {vb['delta']}")
                            rb.update(effect_size=-dz if dz == dz else float("nan"), effect_kind=f"d_z paired (per behaviour, n = {len(d)} scenarios), sign flipped: positive = LESS of the behaviour")
                    recs.append(rb)
                cells_done[(tag, cid)] = dict(pooled=float(p["pooled_score"]), n=audited, refused=refused)
        return recs, cells_done

    # -- MMLU ------------------------------------------------------------------------------------
    def _mmlu_items(self, store: str) -> Tuple[Dict[str, float], dict, dict, dict]:
        items = {}
        with open(os.path.join(store, "rows.jsonl"), encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("task") != "s2parity/mmlu_letter" or r.get("item_id") == "__aggregate__" or r.get("metric") != "acc":
                    continue
                items[r["item_id"]] = float(r["value"])
        gs = (read_json(os.path.join(store, "gen_stats.json")).get("mmlu", {}) if os.path.exists(os.path.join(store, "gen_stats.json")) else {})
        ev = read_json(os.path.join(store, "evidence.json")) if os.path.exists(os.path.join(store, "evidence.json")) else {}
        mf = read_json(os.path.join(store, "manifest.json")) if os.path.exists(os.path.join(store, "manifest.json")) else {}
        return items, gs, ev, mf

    def mmlu_rows(self, stores_side: dict, shas: dict, problems: List[str], run_root: str) -> Tuple[List[dict], dict]:
        recs, done = [], {}
        for tag in VIEWS:
            stores = sorted(store_glob(os.path.join(self.mmlu_cache, tag, "cells", "*")))
            per = {}
            for st in stores:
                store_name = os.path.basename(st)
                cid = store_name.split("__controls__")[0].replace("--", "__")
                items, gs, ev, mf = self._mmlu_items(st)
                per[cid] = dict(items=items, gs=gs, ev=ev, mf=mf, store=store_name, path=st)
            if stores and "none" not in per:
                problems.append(f"MMLU {tag}: no none cell among {sorted(per)}")
            for cid, rec in per.items():
                items, gs, ev, mf = rec["items"], rec["gs"], rec["ev"], rec["mf"]
                remote = f"{REMOTE_HOST}:{run_root}/controls/{tag}/cells/{rec['store']}"
                r = _base_row(tag, cid, "capability", "mmlu", "acc")
                n = len(items)
                acc = float(np.mean(list(items.values()))) if n else float("nan")
                if not n or gs.get("n") != n or abs(float(gs.get("acc", np.nan)) - acc) > 1e-9:
                    r.update(status=f"refused: recomputed MMLU accuracy over {n} items ({acc:.4f}) does not equal the producer's aggregate (acc {gs.get('acc')}, n {gs.get('n')})",
                             note=remote, provenance=remote)
                    recs.append(r)
                    problems.append(r["status"] + f" [{tag}/{cid}]")
                    continue
                k = float(sum(items.values()))
                lo, hi = wilson(k, n)
                store_key = f"{run_root}/controls/{tag}/cells/{rec['store']}"
                entry = stores_side.get(store_key, {})
                substrate_derived = _substrate_of_sidecar_key(store_key, store_key in stores_side, "store_dirs")
                window = entry.get("serving_window", float("nan"))
                stamp = ev.get("engine_stamp") or {}
                short = stamp.get("object_sha")
                sha = shas.get((tag, cid))
                if short and sha and not str(sha).startswith(short):
                    problems.append(f"MMLU {tag}/{cid}: evidence object_sha prefix {short} does not match the staged object {sha[:16]}")
                    sha = None
                r.update(value=acc, k=k, n=n, ci_lo=lo, ci_hi=hi, decode=MMLU_DECODE, status="graded", object_sha256=sha if sha else float("nan"), serving_window_tokens=window,
                         note=f"{n} questions, answer-letter logprob (s2parity/mmlu_letter, top_logprobs {gs.get('top_logprobs')}); censored_items {gs.get('censored_items')}",
                         provenance=f"{remote}/rows.jsonl (local mirror {rec['path']}) | gen_stats.json ›mmlu | evidence.json ›engine_stamp.object_sha={short}")
                sp, _c, _s = DC.serving_path_from_identity(ev.get("serving_identity") or stamp.get("serving_path"))
                created = mf.get("created_at") or (json.loads(open(os.path.join(rec["path"], "rows.jsonl")).readline()).get("created_at") if os.path.exists(os.path.join(rec["path"], "rows.jsonl")) else None)
                r.update(_when=utc_ms(created) if created else DC.mtime_utc(os.path.join(rec["path"], "rows.jsonl")), _code_sha="unverified",
                         _code_sha_reason="the controls store names the wave sha in view_sidecar only", _substrate_derived=substrate_derived,
                         _substrate_default=SUBSTRATE_DEFAULT, _substrate_path="", _object_id=sha,
                         _producer_key=DC.store_run_key(remote, rec["store"]), _object_key=(f"{cid}@{stamp.get('intervention_hash') or ev.get('intervention_hash') or ''}" if cid != "none" else "none"),
                         _store_uri=remote, _model_revision=mf.get("model_revision"), _key=f"{SOURCE}|{tag}|{cid}|mmlu",
                         _instrument={"harness": "bespoke_sisteer", "harness_version": ",".join(entry.get("wave_shas") or []) or "unverified", "harness_config_hash": mf.get("harness_config_hash"),
                                      "token_cap_tokens": 1, "decode": {"temperature": 0.0}, "decode_source": "greedy", "decode_seed_rule": "none", "scoring_rule": "letter_logprob_prefill",
                                      "thinking_mode": "not_available", "serving_path": sp, "serving_identity": ev.get("serving_identity"), "intervention_hash": ev.get("intervention_hash") or mf.get("intervention_hash"),
                                      "env_digest": mf.get("env_digest"), "job_id": ",".join(entry.get("job_ids") or []) or None, "instrument_config_hash": "mmlu_letter_prefill_200", "fit_positions": "all"},
                         _judge=None, _degeneration={"applicable": False, "why": "teacher-forced letter-logprob prefill decodes nothing"}, _n_items_planned=200,
                         _strength_coefficient=(1.0 if (sha and cid != "none") else None), _is_operating_point=(1 if (sha and cid != "none") else None),
                         _items=[{"item_id": i, "metric_id": "acc", "value": v, "judged": False} for i, v in sorted(items.items())],
                         _extra={"positive_control_verdict": entry.get("positive_control_verdict"), "job_ids": entry.get("job_ids"), "wave_shas": entry.get("wave_shas")})
                base = per.get("none")
                if cid != "none" and base and base["items"]:
                    shared = [i for i in items if i in base["items"]]
                    if shared:
                        x = np.array([items[i] for i in shared])
                        y = np.array([base["items"][i] for i in shared])
                        m, blo, bhi, _ = _paired(x - y)
                        b_only = int(((x == 1) & (y == 0)).sum())
                        c_only = int(((x == 0) & (y == 1)).sum())
                        r.update(delta=100 * m, delta_lo=100 * blo, delta_hi=100 * bhi, delta_p=mcnemar_exact(b_only, c_only),
                                 delta_method=f"paired bootstrap over {len(shared)} shared items ({N_BOOT} draws, seed {SEED}) + exact McNemar ({b_only} only-steered / {c_only} only-unsteered correct)",
                                 effect_size=cohen_h(acc, float(y.mean())), effect_kind="cohen_h accuracy vs own unsteered",
                                 _anchor_key=f"{SOURCE}|{tag}|none|mmlu", _anchor_kind="same_standup_none", _n_pairs=len(shared))
                recs.append(r)
                done[(tag, cid)] = dict(acc=acc, n=n)
        return recs, done

    # -- dispositions (generated, not graded; not served; not applicable) ------------------------
    def disposition_rows(self, shas: dict, petri_done: dict, mmlu_done: dict, run_root: str) -> List[dict]:
        recs = []
        absent = "no StrongREJECT / over-refusal judge record exists anywhere the grading host can read; OWNER the Llama lane (graded on the lane's own path)"

        def _disp(tag, cid, component, eval_id, status, note, provenance, **extra):
            metric, _t, _b = DC.HEADLINE[component]
            r = _base_row(tag, cid, component, eval_id, metric)
            r.update(status=status, note=note, provenance=provenance)
            r.update(extra)
            return r
        for tag, cells in SERVED.items():
            for cid in cells:
                sha = shas.get((tag, cid))
                comps = COMPONENTS_BY_SPELLING[tag]
                if "static" in comps:
                    for fam in STATIC_FAMILIES:
                        root = f"{REMOTE_HOST}:{run_root}/transcripts_results/sdfbase_full22_{tag}_sampled_t{{0,1}}{{,_cap8192}}/{fam}/{cid}/shard*/responses.jsonl"
                        recs.append(_disp(tag, cid, "static", fam, "pending: GENERATED, NOT GRADED - sampled single-shot stores complete (4,096 base + 8,192 rung), no StrongREJECT grade",
                                          absent, root, threshold=0.5, object_sha256=sha if sha else float("nan"), _object_id=sha))
                if "bon" in comps:
                    root = f"{REMOTE_HOST}:{run_root}/transcripts_results/sdfbase_full22_{tag}/{cid}/bon_responses.jsonl"
                    recs.append(_disp(tag, cid, "bon", "bon", "pending: GENERATED, NOT GRADED - native 4,096 best-of-75 store complete, no StrongREJECT grade", absent, root,
                                      threshold=0.5, bon_n=75, object_sha256=sha if sha else float("nan"), _object_id=sha))
                if "multi-turn" in comps and tag != "llama8b_shortconst":
                    for att in MULTITURN_ATTACKS:
                        root = f"{REMOTE_HOST}:{run_root}/transcripts_results/sdfbase_full22_{tag}/{att}/{cid}/shard*/responses.jsonl"
                        recs.append(_disp(tag, cid, "multi-turn", att, f"pending: GENERATED, NOT GRADED - {att} at 4,096 per reply complete, no StrongREJECT grade", absent, root,
                                          threshold=0.5, object_sha256=sha if sha else float("nan"), _object_id=sha))
                if "over-refusal" in comps:
                    for ctl in ("xstest", "or_bench_hard_1k"):
                        if tag == "llama8b_fullconst":
                            recs.append(_disp(tag, cid, "over-refusal", ctl, "pending: generated (the greedy 512 served-controls leg only) and NOT GRADED; grading is NOT scheduled (the user declined over-refusal for the full-constitution arm)",
                                              absent, f"{REMOTE_HOST}:{run_root}/controls/{tag}/cells/<{cid}>__controls__*/private/gens_*.jsonl"))
                            continue
                        recs.append(_disp(tag, cid, "over-refusal", ctl, "pending: GENERATED, NOT GRADED - sampled controls generated at the 512 rung only, no pass-1 / pass-2 sonnet judge", absent,
                                          f"{REMOTE_HOST}:{run_root}/sampled/controls_cap512/{tag}_t{{0,1}}/cells/<{cid}>__controls__*/private/gens_{{xstest_v2,or_bench_hard_1k}}.jsonl",
                                          object_sha256=sha if sha else float("nan"), _object_id=sha))
                if "capability" in comps:
                    if (tag, cid) not in mmlu_done:
                        recs.append(_disp(tag, cid, "capability", "mmlu", "pending: the MMLU letter-prefill record exists on cluster B but could not be read for this build",
                                          "see the mirror problem", f"{REMOTE_HOST}:{run_root}/controls/{tag}/cells/<{cid}>__controls__*/rows.jsonl", _object_id=sha))
                    recs.append(_disp(tag, cid, "capability", "gsm8k", "pending: GSM8K is EvalCheck's paired-200 thinking-ON instrument; Llama was OMITTED from this wave (D194c). OWNER: EvalCheck",
                                      "no record exists", f"{LLAMA8B_ROOT}/wave/caps.yaml ›capability_omitted.gsm8k", _object_id=sha))
                    recs.append(_disp(tag, cid, "capability", "ifeval", "pending: the IFEval SLOT - the launcher refused every f22-ifeval-<tag> job until EvalCheck names the inspect task id; nothing generated",
                                      "inspect_evals ifeval planned", f"{LLAMA8B_ROOT}/wave/caps.yaml ›ifeval", _object_id=sha))
                    recs.append(_disp(tag, cid, "capability", "math500", "not applicable: MATH-500 is a thinking-ON instrument and Llama-3.1-8B has no thinking channel - excluded for Llama by ruling (D194c)",
                                      "no Llama MATH-500 cell exists or is planned", f"{LLAMA8B_ROOT}/wave/caps.yaml ›header (D194c)", _object_id=sha))
                if "petri" in comps and (tag, cid) not in petri_done:
                    recs.append(_disp(tag, cid, "petri", "petri_pooled", "pending: served cell without a Petri block in the lane's export", f"{self.petri_export} ›models.{tag}.cells has no {cid}",
                                      self.petri_export, _object_id=sha))
                if tag == "llama8b_fullconst":
                    for att in MULTITURN_ATTACKS:
                        recs.append(_disp(tag, cid, "multi-turn", att, "not served: the full-constitution view runs static, BoN and capability only (user ruling 2026-09-09)", "scoping ruling", f"{LLAMA8B_ROOT}/HANDOFF"))
                    recs.append(_disp(tag, cid, "petri", "petri_pooled", "not served: Petri is INFEASIBLE for the 34k prompt at the certified 16,384 Petri window", "scoping ruling", f"{LLAMA8B_ROOT}/HANDOFF"))
        for tag in ("llama8b", "llama8b_it"):
            for cid in ("constv2-mix-full__psr_ce_all", "constv2ctl2-full__psr_ce_all", "constv2-mix-full__psr_ce_all_minus_ctl_distilled"):
                evals = [(c, e) for c in ("static", "bon", "multi-turn", "over-refusal") for e in (STATIC_FAMILIES if c == "static" else ("bon",) if c == "bon" else MULTITURN_ATTACKS if c == "multi-turn" else ("xstest", "or_bench_hard_1k"))]
                evals += [("capability", "mmlu"), ("capability", "gsm8k"), ("capability", "ifeval"), ("capability", "math500"), ("petri", "petri_pooled")]
                for comp, ev in evals:
                    recs.append(_disp(tag, cid, comp, ev, f"not served: the ReLU-steer objects were fit and staged for the {VIEWS[tag]['short']} view but no serving leg of the wave lists the cell (the user's ruling 2026-09-11)",
                                      "staged, never served", f"{self.stage}/stage_plan.json", _object_id=shas.get((tag, cid))))
        for tag in ("llama8b", "llama8b_it"):
            cid = "prompted__si_constitution"
            evals = [(c, e) for c in ("static", "bon", "multi-turn", "over-refusal") for e in (STATIC_FAMILIES if c == "static" else ("bon",) if c == "bon" else MULTITURN_ATTACKS if c == "multi-turn" else ("xstest", "or_bench_hard_1k"))]
            evals += [("capability", "mmlu"), ("capability", "gsm8k"), ("capability", "ifeval"), ("capability", "math500"), ("petri", "petri_pooled")]
            for comp, ev in evals:
                recs.append(_disp(tag, cid, comp, ev, f"not served: {cid} was DECLARED on the landed roster and DROPPED at staging - never served on the {VIEWS[tag]['short']} view (the constitution-SI arm is served on shortconst only)",
                                  "ruling (c)", f"{LLAMA8B_ROOT}/HANDOFF"))
        return recs

    def long_rows(self) -> Tuple[List[dict], Dict[str, object]]:
        problems: List[str] = []
        serving, stores, verdicts, fail_jobs = self.sidecar()
        run_root = recorded_root(self.view_sidecar_legacy, serving, stores)
        shas = self.object_shas()
        petri, petri_done = self.petri_rows(serving, shas, problems, run_root)
        mmlu, mmlu_done = self.mmlu_rows(stores, shas, problems, run_root)
        disp = self.disposition_rows(shas, petri_done, mmlu_done, run_root)
        facts = dict(problems=problems, petri_cells=len(petri_done), mmlu_cells=len(mmlu_done), object_shas=len(shas), fail_jobs=sorted(fail_jobs), n_dispositions=len(disp))
        return petri + mmlu + disp, facts

    def read(self, ctx: AdapterContext, group: str, inputs: Sequence[Input]) -> Iterator[RecordBatch]:
        rows, facts = self.long_rows()
        self.last_facts = facts
        batch = RecordBatch(f"{self.name}:{group}", group)
        rep = DC.emit_rows(batch, ctx, rows, adapter=self.name, campaign=self.campaign, reader="lane_llama8b.load",
                           seed_provenance=f"default_rng({SEED}) per cell, n_boot={N_BOOT}", code_sha="unverified",
                           code_sha_reason="the wave sha rides each run's harness_version; the export names no code sha per row")
        self.last_report = rep
        yield batch

    def source_states(self, ctx: AdapterContext, inputs: Sequence[Input]) -> List[SourceStateRow]:
        ex = next((i for i in inputs if i.uri == self.petri_export), None)
        if ex is None:
            return []
        d = read_json(ex.path)
        out = []
        for tag, blk in (d.get("models") or {}).items():
            out.append(SourceStateRow(f"{self.name}:wave", ex.uri, f"$.models.{tag}.cells", 1, 1, sorted((blk.get("cells") or {}).keys()), ex.sha256))
        return out


def register() -> None:
    from .base import register as _register
    _register(Llama8bWaveAdapter.name, lambda: Llama8bWaveAdapter())
