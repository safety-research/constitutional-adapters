"""The completion-cap MERGES of the agentic-misalignment ladder (`em_misalign` versions 7 and 9; source `misalign_ladder`, em_misalign's fourth root).

A SYNTHETIC world (placeholder shas, TESTHOST, invented scenario ids): a seed merged root (`qwen35_4b_s1`: a prompted key and a fitted key) and a
paired merged root (`qwen35_4b_s1p`: a fitted key with a judge-blocked row and a key whose leaking rows were never re-drawn), each in the merge
tool's shape -- `ladder_rung` on every row, `meta.ladder`, `ladder_report.json`, `ladder_merge.log`, a header-only `paired.csv`, a row store whose
`run_id` collides with the base root's -- plus an inert `agentic.POISONED-conc200/` copy, a lens-merge key (by sha, by `rerun_dir`, by route), a
canon root on a lens base (refused) and on a lora_native base (admitted), and the tool pin. Version 9: a canon root that carries its OWN none with
a `paired.csv` that has rows (the producer's pairs against that none) is admitted and files them as run_contrasts; a root with no none still refuses
a non-empty `paired.csv`; a passthrough identity on a lora_native standup is admitted; every other merged root binds its parked block to the canon
none run by identity (no delta across roots) when the canon root is admitted, and parks `no_merged_baseline` when it is absent or refused."""
from __future__ import annotations

import hashlib
import json
import os
import sys

import pytest
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
sys.path.insert(0, HERE)

from sisteer.registry import validate as VAL  # noqa: E402
from sisteer import site as SITE  # noqa: E402
from sisteer.registry.adapters import declared_common as DC  # noqa: E402
from sisteer.registry.adapters import em_misalign as EM  # noqa: E402
from sisteer.registry.adapters.base import AdapterContext, AdapterRefusal, RecordBatch  # noqa: E402
from sisteer.registry import synth

import registry_declared_fixtures as F  # noqa: E402
import _anchor_frame_fixtures as AF  # noqa: E402  (emitter 4: in-memory reads need a primed anchor frame)

NOW = "2026-09-22T03:00:00.000Z"
PRODUCER = {"script": "tests.em_misalign_ladder", "code_sha": synth.code_sha("a"), "session": None, "host": "TESTHOST", "substrate": "grading_host"}
LORA, PASS, LENS = "proxy_vllm+lora+interleaved", "proxy_vllm+passthrough+interleaved", "proxy_vllm+lens+interleaved"
MODEL = "qwen35_4b"
TREE_S, TREE_P = synth.code_sha("4"), synth.code_sha("5")
S1, S1P, CANON = f"{MODEL}_s1", f"{MODEL}_s1p", MODEL
LORA_H, LORA_R = "si_constv2_mix_full__lora_r4_ce", "constv2-mix-full__lora_r4_ce"
LDIFF_H, LDIFF_R = "si_constv2_mix_full__lora_r4_ce_minus_ctl", "constv2-mix-full__lora_r4_ce_minus_ctl"
CTL2_H, CTL2_R = "si_constv2ctl2_full__psr_ce_all", "constv2ctl2-full__psr_ce_all"
CONST_H, CONST_R = "prompted__constitution", "prompted__si_constitution"
SHA = {LORA_R: hashlib.sha256(b"synthetic lora s1 merged").hexdigest(), LDIFF_R: hashlib.sha256(b"synthetic lora diff s1 merged").hexdigest(),
       CTL2_R: hashlib.sha256(b"synthetic ctl2 relu s1 merged").hexdigest()}
STAGED = {LORA_R: f"base64s1__{LORA_R}", LDIFF_R: f"derived_base64s1__{LDIFF_R}", CTL2_R: f"base64s1__{CTL2_R}"}
ITEMS = {"leaking": ["leaking_explicit-alpha_none", "leaking_explicit-beta_none"], "murder": ["murder_explicit-alpha_none", "murder_explicit-beta_none"]}
N_SAMPLES = 2
OVERRIDE = {"min_p": 0.0, "repetition_penalty": 1.0, "temperature": 0.7, "top_k": 20, "top_p": 0.8}
FILL = {"temperature": 1.0, "min_p": 0.0, "repetition_penalty": 1.0, "top_k": 20, "top_p": 0.8}

# per key: behaviour -> per slot (4 = 2 items x 2 samples) the tuple (harmful, rung, at_cap_after) -- rung 16384 = a re-drawn pair, rung 3000 = a
# base row that completed (at_cap False) or looped (at_cap True); `blocked` names slots whose verdict is judge-blocked
S1_KEYS = {CONST_H: dict(roster=CONST_R, ident=PASS, kind="prompted", si="constitution", si_tt="si_constitution",
                         rows={"leaking": [(1, 3000, False), (0, 16384, True), (1, 16384, False), (0, 3000, False)], "murder": [(0, 16384, True), (0, 16384, True), (0, 3000, False), (1, 16384, False)]}),
           CTL2_H: dict(roster=CTL2_R, staged=STAGED[CTL2_R], ident=LORA, kind="psr",
                        rows={"leaking": [(1, 3000, False), (1, 3000, True), (0, 16384, False), (0, 3000, False)], "murder": [(1, 16384, False), (0, 3000, False), (0, 3000, False), (1, 16384, True)]})}
S1P_KEYS = {LORA_H: dict(roster=LORA_R, staged=STAGED[LORA_R], ident=LORA, kind="lora_all",
                         rows={"leaking": [(0, 3000, False), (0, 16384, False), (1, 16384, True), (0, 3000, False)], "murder": [(0, 3000, False), (0, 3000, False), (1, 16384, False), (0, 16384, False)]},
                         blocked={"murder": [1]}),
            LDIFF_H: dict(roster=LDIFF_R, staged=STAGED[LDIFF_R], ident=LORA, kind="lora_subtraction",
                          rows={"leaking": [(0, 3000, False), (1, 3000, True), (0, 3000, False), (0, 3000, True)], "murder": [(0, 16384, False), (0, 3000, False), (1, 3000, False), (0, 3000, False)]})}


def _w(path: str, obj, jsonl: bool = False) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        if jsonl:
            for r in obj:
                fh.write(json.dumps(r) + "\n")
        else:
            json.dump(obj, fh, indent=1)
    return path


def _ctx() -> AdapterContext:
    return AdapterContext(root="", now=NOW, producer=dict(PRODUCER))


def _si_prefix(si_id: str) -> str:
    return str(DC.load_seeds().si_object_id(si_id)).split(":", 1)[1][:16]


def _sha(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _merged_rows(key: str, spec, judge_model="claude-sonnet-5"):
    """``(responses, judged, report_counts)`` for one merged key: every slot of every behaviour, each row carrying its generating cap."""
    resp, judged, rep = [], [], {}
    blocked = spec.get("blocked", {})
    for b, ids in ITEMS.items():
        slots = [(item, s) for item in ids for s in range(N_SAMPLES)]
        c = {"n": 0, "at_cap_src": 0, "selected": 0, "rerun_present": 0, "at_cap_after": 0, "at_cap_after_all": 0, "degenerate": 0}
        for si, (item, s) in enumerate(slots):
            h, rung, capped = spec["rows"][b][si]
            r = {"family": "agentic", "cell_id": key, "virtual_model": f"petri__{key}", "item_id": item, "sample_idx": s, "behavior": b,
                 "prompt_sha": hashlib.sha256(f"{item}|{s}".encode()).hexdigest(), "response": f"placeholder response {key} {b} {si}",
                 "finish_reason": ("length" if capped else "stop"), "usage": {"prompt_tokens": 100, "completion_tokens": (rung if capped else 40), "total_tokens": 140},
                 "model_reported": f"petri__{key}", "ladder_rung": rung}
            resp.append(r)
            blk = si in blocked.get(b, [])
            judged.append(dict(r, judge_tag=EM.SEEDREP_JUDGE_TAGS[b], judge_model=judge_model, judge_pool="lowprio", judged_at="2026-09-21T22:00:00Z",
                               structural_gate=(0 if blk else 1), judge_blocked=blk, classifier_harmful=(None if blk else h), harmful=(None if blk else h)))
            c["n"] += 1
            if rung == 16384:
                c["at_cap_src"] += 1
                c["selected"] += 1
                c["rerun_present"] += 1
                if capped:
                    c["at_cap_after"] += 1
            elif capped:
                c["at_cap_src"] += 1
                c["degenerate"] += 1
            if capped:
                c["at_cap_after_all"] += 1
        rep[b] = {k: v for k, v in c.items() if v or k in ("n", "at_cap_src", "selected", "rerun_present", "at_cap_after", "at_cap_after_all")}
        if rep[b].get("degenerate") == 0:
            rep[b].pop("degenerate", None)
    return resp, judged, rep


def _summary(target: str, keys: dict, rows_by_key: dict) -> list:
    out = []
    for key, judged in rows_by_key.items():
        for b in ITEMS:
            rows = [r for r in judged if r["behavior"] == b]
            scored = [r for r in rows if not r["judge_blocked"]]
            n, k = len(scored), sum(int(r["harmful"]) for r in scored)
            cap_k = sum(1 for r in rows if r["finish_reason"] == "length")
            for metric, kk, est in (("harmful", k, k / n), ("at_cap", cap_k, cap_k / n), ("loop", 0, 0.0), ("degenerate", 0, 0.0), ("harmful_reasoned", k, k / n),
                                    ("harmful_bare", 0, 0.0), ("harmful_fuzzy", k, k / n)):
                out.append(dict(target=target, arm=key, family="agentic", behavior=b, metric=metric, n=n, k=kk, estimate=est, ci_low=max(0.0, est - 0.1),
                                ci_high=min(1.0, est + 0.1), ci_method="wilson95", boot_low=None, boot_high=None, note=None, engine=keys[key]["ident"],
                                standup_engine=None, sampler="t1", sampling=json.dumps({"temperature": 1.0})))
            out.append(dict(target=target, arm=key, family="agentic", behavior=b, metric="response_chars", n=n, k=None, estimate=1234.0, ci_low=None, ci_high=None,
                            ci_method=None, boot_low=None, boot_high=None, note=None, engine=keys[key]["ident"], standup_engine=None, sampler="t1",
                            sampling=json.dumps({"temperature": 1.0})))
    return out


PAIRED_COLS = ["target", "arm", "ref", "family", "behavior", "metric", "n_pairs", "mean_arm", "mean_none", "diff", "boot_low", "boot_high", "arm_only", "none_only",
               "mcnemar_p", "note", "engine", "anchor_engine", "standup_engine", "anchor_standup_engine", "cross_engine"]


def _paired_against_none(target: str, keys: dict, rows_by_key: dict, *, ref: str = "none", omit=()) -> list:
    """The producer's pairs of every non-none key against the root's none over the slots both scored: one agentic/harmful row per key x behaviour
    (``mean_arm``, ``mean_none``, ``diff``, a placeholder bootstrap interval and McNemar p, the two idents, ``cross_engine`` False)."""
    out = []
    none_rows = {(r["behavior"], r["item_id"], r["sample_idx"]): r for r in rows_by_key.get("none", []) if not r["judge_blocked"]}
    for key, judged in rows_by_key.items():
        if key == "none" or key in omit:
            continue
        for b in ITEMS:
            pairs = [(int(r["harmful"]), int(none_rows[(b, r["item_id"], r["sample_idx"])]["harmful"])) for r in judged
                     if r["behavior"] == b and not r["judge_blocked"] and (b, r["item_id"], r["sample_idx"]) in none_rows]
            n_pairs = len(pairs)
            mean_key, mean_none = sum(a for a, _ in pairs) / n_pairs, sum(x for _, x in pairs) / n_pairs
            diff = mean_key - mean_none
            out.append(dict(zip(PAIRED_COLS, [target, key, ref, "agentic", b, "harmful", n_pairs, mean_key, mean_none, diff, diff - 0.1, diff + 0.1,
                                              sum(1 for a, x in pairs if a and not x), sum(1 for a, x in pairs if x and not a), 0.5, None, keys[key]["ident"],
                                              keys["none"]["ident"], None, None, False])))
    return out


def _root(rsl: str, target: str, *, kind: str, tree: str, job: str, keys: dict, rerun_root: str = "/shared/tester/exp/misalign_16k", src_root: str = "/shared/tester/exp/misalign",
          engine_kind: str = "lora_native", paired_rows: bool = False, report_override=None, unrerun: int = 0, base_cap: int = 3000, rerun_cap: int = 16384,
          poisoned: bool = False, paired_against_none: bool = False, paired_ref: str = "none", paired_omit=()) -> str:
    """One merged root in the merge tool's shape. ``keys``: harness id -> {roster, ident, kind, rows[, staged, si, blocked]}. ``paired_rows`` writes ONE
    invented pair row (the version-7 refusal probe); ``paired_against_none`` writes the producer's real pairs of every key against the root's none."""
    import pandas as pd
    tdir = os.path.join(rsl, target)
    suffix = target[len(MODEL):]
    selection = "main" if suffix in ("", "_bl") else suffix[1:]
    bank, legs, harness_cells, roster_cells = {}, [], ["none"], ["none"]
    legs.append({"cell_class": "base", "cell_id": "none", "control_of": None, "harness_config_hash": "4269a5c2d945d1c4", "intervention_hash": "2ff9cdff091aef6f", "kind": "none",
                 "object_path": None, "petri_run_key": "0e9f26004a079ff8", "serving_identity": PASS, "si_id": None, "si_prefix_hash": None, "virtual_model_name": "petri__base",
                 "si_id_for_runner": None, "measured": kind == "canon"})
    for i, (h, c) in enumerate(keys.items()):
        if h == "none":
            continue
        harness_cells.append(h)
        roster_cells.append(c["roster"])
        if c.get("staged"):
            sha = SHA[c["roster"]]
            bank[c["roster"]] = {"identity": "staged", "object_sha256": sha, "positions_at_fit": "all", "sidecar_object_sha256": sha,
                                 "staged_from": {"cell_id": c["roster"], "effective_batch": 64, "key": f"{MODEL}_base_eb64_s1", "kind": c["kind"], "lane": c["staged"].split("__")[0],
                                                 "object_sha256": sha, "pulled_at": "2026-09-21T06:30:00Z", "source": "synthetic", "staged_by": "synthetic"}}
        legs.append({"cell_class": ("prompted" if c.get("si") else "class_a"), "cell_id": h, "control_of": None, "harness_config_hash": "f411a1c5a34edb57",
                     "intervention_hash": f"{i + 1:016x}", "kind": c["kind"], "object_path": None, "petri_run_key": hashlib.sha256(f"{target}|{h}".encode()).hexdigest()[:16],
                     "serving_identity": c["ident"], "si_id": c.get("si"), "si_prefix_hash": (_si_prefix(c["si_tt"]) if c.get("si") else None), "virtual_model_name": f"petri__{h}",
                     "si_id_for_runner": c.get("si"), "measured": True})
    measured = [h for h in keys] if kind != "canon" else ["none"] + [h for h in keys if h != "none"]
    policy = {"measure_none": kind == "canon", "measured_cells_declared": None, "lane": ("paired" if kind == "paired" else ("reseed" if kind == "seed" else None)),
              "pairing_anchor": (f"exp/misalign/{MODEL}/agentic/none" if kind in ("seed", "paired") else None), "source": f"misalign_full22_{target}.yaml misalign_policy",
              "served_legs": ["none"] + [h for h in keys if h != "none"] + ["none__lora0"], "measured_legs": measured, "unmeasured_served": ([] if kind == "canon" else ["none", "none__lora0"])}
    override = json.dumps(OVERRIDE, separators=(",", ":"))
    ep = {"model": "Qwen/Qwen3.5-4B", "model_revision": F.REVISION, "model_key": MODEL, "proxy_url": "http://127.0.0.1:1/v1", "max_model_len": 16384, "max_num_seqs": 31,
          "attested_max_model_len": 16384, "attested_max_num_seqs": 31, "thinking_mode": "off", "node": "gpu-7", "slurm_job_id": job, "cells": roster_cells,
          "harness_cells": harness_cells, "server_default_policy": "contract_override", "override_generation_config": override, "bank_objects": bank}
    _w(os.path.join(tdir, "cells_full.json"), {"target": target, "model_key": MODEL, "wave_view": selection, "wave_sha": tree, "written_at": "2026-09-21T08:01:39Z",
                                              "measured_policy": policy, "endpoint": ep,
                                              "cells_served": {"manifest": f"/shared/tester/repo/{tree}/configs/capeval/manifests/misalign_full22_{target}.yaml"}, "legs": legs})
    _w(os.path.join(tdir, "cells.json"), {"cells": [{"cell_id": h} for h in keys]})
    serving = {"substrate": "slurm", "slurm_job_id": job, "baseline_pins": None, "served_cluster": "cluster_a", "shard": None, "attested_max_num_seqs": 31,
               "attested_max_model_len": 16384, "max_num_seqs": 31, "max_model_len": 16384, "node": "gpu-7", "measured_policy": policy,
               "server_default_policy": "contract_override", "override_generation_config": override}
    _w(os.path.join(tdir, "agentic", "cells.json"), {"cells": [{"cell_id": h, "virtual_model": f"petri__{h}", "serving_identity": c["ident"], "intervention_hash": "x",
                                                                 "si_id": c.get("si"), "kind": c["kind"]} for h, c in keys.items()], "serving": serving})
    report, rows_by_key, pq = {}, {}, []
    art = {h: SHA[c["roster"]] for h, c in keys.items() if c.get("staged")}
    for h, c in keys.items():
        resp, judged, rep = _merged_rows(h, c)
        rows_by_key[h] = judged
        report[f"agentic/{h}"] = rep
        cd = os.path.join(tdir, "agentic", h)
        _w(os.path.join(cd, "responses.jsonl"), resp, jsonl=True)
        _w(os.path.join(cd, "judged.jsonl"), judged, jsonl=True)
        n_sel = sum(v.get("selected", 0) for v in rep.values())
        n_deg = sum(v.get("degenerate", 0) for v in rep.values())
        meta = {"behaviors": list(ITEMS), "cell_id": h, "cell": {"cell_id": h, "intervention_hash": "x", "kind": c["kind"], "serving_identity": c["ident"]}, "family": "agentic",
                "hostname": "gpu-7", "items": sum(len(v) for v in ITEMS.values()), "item_ids": [i for v in ITEMS.values() for i in v], "max_tokens": base_cap, "n_samples": N_SAMPLES,
                "sampler": "t1", "sampling": {"temperature": 1.0}, "temperature": 1.0, "slurm_job_id": job,
                "params": {"concurrency": 44, "max_tokens": base_cap, "sampler": "t1", "sampling": {"temperature": 1.0}, "temperature": 1.0},
                "present": len(resp), "generated_this_run": len(resp), "suite_revision": "agentic.v2@synthetic", "run": {"only_cells": sorted(keys), "serving": serving},
                "endpoint_identity": {"base_model": "Qwen/Qwen3.5-4B", "engine": {"kind": engine_kind}, "server_default_policy": "contract_override", "thinking_mode": "off",
                                      "object_artifact_sha256_by_cell": art},
                "ladder": {"src_dir": f"{src_root}/{target}/agentic/{h}", "rerun_dir": f"{rerun_root}/{target}/agentic/{h}", "src_cap": base_cap, "rerun_cap": rerun_cap, "min_zlib": 0.2,
                           "selected": n_sel, "degenerate": n_deg, "rerun_present": n_sel - unrerun, "unrerun": unrerun}}
        _w(os.path.join(cd, "meta.json"), meta)
        for r in judged:
            for metric in ("harmful", "harmful_fuzzy", "at_cap", "response_chars", "zlib_ratio"):
                if metric in ("harmful", "harmful_fuzzy") and r["judge_blocked"]:
                    continue
                pq.append(dict(run_id=f"misalign_{target}.{h}.0e63221d", arm_name=h, task=f"agentic/{r['behavior']}", item_id=r["item_id"], metric=metric, value=float(r["harmful"] or 0)))
    if report_override:
        report_override(report)
    _w(os.path.join(tdir, "ladder_report.json"), report)
    with open(os.path.join(tdir, "ladder_merge.log"), "w", encoding="utf-8") as fh:
        fh.write("cell/behavior n cap@src loop sel rerun unrerun cap@rung residual%\n")
    adir = os.path.join(tdir, "analysis", target)
    os.makedirs(adir, exist_ok=True)
    pd.DataFrame(_summary(target, keys, rows_by_key)).to_csv(os.path.join(adir, "summary.csv"), index=False)
    pr = pd.DataFrame(columns=PAIRED_COLS)
    if paired_rows:
        pr = pd.DataFrame([dict(zip(PAIRED_COLS, [target, list(keys)[0], "none", "agentic", "leaking", "harmful", 4, 0.5, 0.25, 0.25, 0.0, 0.5, 1, 0, 0.5, None, LORA, PASS, None, None, False]))])
    elif paired_against_none:
        pr = pd.DataFrame(_paired_against_none(target, keys, rows_by_key, ref=paired_ref, omit=paired_omit), columns=PAIRED_COLS)
    pr.to_csv(os.path.join(adir, "paired.csv"), index=False)
    os.makedirs(os.path.join(tdir, "store"), exist_ok=True)
    pd.DataFrame(pq).to_parquet(os.path.join(tdir, "store", "rows.parquet"), index=False)
    os.makedirs(os.path.join(tdir, "em8q", list(keys)[0]), exist_ok=True)
    os.makedirs(os.path.join(tdir, "serving"), exist_ok=True)
    if poisoned:
        _w(os.path.join(tdir, "agentic.POISONED-conc200", list(keys)[0], "judged.jsonl"), [{"item_id": "x", "sample_idx": 0, "behavior": "leaking"}], jsonl=True)
    return tdir


NONE_KEY = dict(roster="none", ident=PASS, kind="none",
                rows={"leaking": [(1, 3000, False), (1, 16384, True), (0, 16384, False), (0, 3000, False)], "murder": [(0, 16384, True), (0, 16384, False), (0, 3000, False), (0, 16384, False)]})
CANON_KEYS = {"none": NONE_KEY, CONST_H: dict(S1_KEYS[CONST_H]), CTL2_H: dict(S1_KEYS[CTL2_H])}


def _roster(tmp_path) -> dict:
    roster = {"targets": {MODEL: {"views": {"s1": {"cells": {CTL2_R: STAGED[CTL2_R], LORA_R: STAGED[LORA_R], LDIFF_R: STAGED[LDIFF_R], CONST_R: None}},
                                            "s1p": {"paired_of": "s1", "cells": {LORA_R: STAGED[LORA_R], LDIFF_R: STAGED[LDIFF_R]}},
                                            "main": {"cells": {CTL2_R: STAGED[CTL2_R], CONST_R: None}}}}}}
    rp = str(tmp_path / "roster.yaml")
    with open(rp, "w", encoding="utf-8") as fh:
        yaml.safe_dump(roster, fh)
    pins = str(tmp_path / "pins.json")
    _w(pins, {MODEL: {STAGED[r]: {"object_sha256": SHA[r]} for r in STAGED}})
    return {"roster": rp, "pins": pins}


def _world(tmp_path, *, s1_kw=None, s1p_kw=None, with_s1p=True, canon_kw=None):
    figs, rs = str(tmp_path / "figs"), str(tmp_path / "rs")
    rsl = os.path.join(rs, "misalign_ladder")
    F.build_em_panel(figs)
    os.makedirs(rsl, exist_ok=True)
    s1_args = dict(keys=S1_KEYS)
    s1_args.update(s1_kw or {})
    _root(rsl, S1, kind="seed", tree=TREE_S, job="586157", **s1_args)
    if with_s1p:
        s1p_args = dict(keys=S1P_KEYS, poisoned=True)
        s1p_args.update(s1p_kw or {})
        _root(rsl, S1P, kind="paired", tree=TREE_P, job="586027", **s1p_args)
    if canon_kw is not None:
        canon_args = dict(keys=dict(S1_KEYS))
        canon_args.update(canon_kw)
        _root(rsl, CANON, kind="canon", tree=TREE_S, job="592837", **canon_args)
    os.makedirs(os.path.join(rsl, "_scratch_never_read", "agentic"), exist_ok=True)
    return dict(figs=figs, rsl=rsl, grade=str(tmp_path / "grade"), **_roster(tmp_path))


def _adapter(w, **kw) -> EM.EmMisalignAdapter:
    args = dict(root=w["figs"], full22_pins=w["pins"], baseline_pins=None, behaviour_source={}, grade_root=w["grade"], seedrep_root=None, m16k_root=None, roster_path=w["roster"],
                ladder_root=w["rsl"])
    args.update(kw)
    return EM.EmMisalignAdapter(**args)


def _groups(ad):
    groups = {}
    for i in ad.discover():
        groups.setdefault(ad.comparison_group(i), []).append(i)
    return groups


def _read(ad, group, groups=None):
    groups = groups if groups is not None else _groups(ad)
    recs = []
    ctx = AF.primed_ctx(ad, now=NOW, producer=PRODUCER, groups=groups)     # the frame of the pass AFTER the fold: every own anchor current
    for batch in ad.read(ctx, group, groups.get(group, [])):
        recs.extend(batch.records)
    bad = [str(v) for v in (VAL.validate_record(r, path="<test>") for r in recs) if isinstance(v, VAL.Malformed)]
    assert not bad, bad[:4]
    out = {}
    for r in recs:
        out.setdefault(r["record_type"], []).append(r["payload"])
    return out, ad.last_facts[group], ad.last_report


G1, G1P, GC = f"{EM.LADDER_GROUP_PREFIX}{S1}", f"{EM.LADDER_GROUP_PREFIX}{S1P}", f"{EM.LADDER_GROUP_PREFIX}{CANON}"


def _run(recs, key: str, ev: str):
    return next(r for r in recs["run"] if r["campaign_cell_id_legacy"] == key and r["eval_id"] == ev)


# ============================================================================================
# declarations, the tool pin, discovery
# ============================================================================================

def test_the_declarations_the_version_and_the_merge_tool_pin():
    # em_misalign 11 (2026-10-03): the anchor currency decides every contrast through the shared emitter (`extra.anchor_selection`)
    assert EM.EmMisalignAdapter.EMISSION_VERSION == DC.emission_version(12)      # 12: a row whose object the registry does not carry is parked by name (registry follow-ups-4 item 8)
    assert EM.LADDER_SOURCE == "misalign_ladder" and EM.LADDER_SOURCE in EM.EmMisalignAdapter.covers_sources
    # version 9: four anchor states, all distinct; the same-root kind is the step reader's same_standup_none, the cross-root kind sibling_none
    states = (EM.LADDER_ANCHOR_STATE, EM.LADDER_ANCHOR_STATE_OWN, EM.LADDER_ANCHOR_STATE_CANON, EM.LADDER_ANCHOR_STATE_IS_BASELINE)
    assert states == ("no_merged_baseline", "own_merged_baseline", "canon_merged_baseline", "is_merged_baseline") and len(set(states)) == 4
    assert EM.LADDER_ANCHOR_KIND_SAME == "same_standup_none" and EM.LADDER_ANCHOR_KIND_CANON == "sibling_none"
    assert "lora_native canon base" in EM.LADDER_ANCHOR_NEEDED and "{engine}" in EM.LADDER_CANON_ANCHOR_WHY and "{decode}" in EM.LADDER_CANON_ANCHOR_WHY
    assert EM.LADDER_ROOT == os.path.join(EM.LC.RS_DEFAULT, "misalign_ladder") and str(SITE.expand_path("$GRADE")) not in EM.LADDER_ROOT
    assert EM.LADDER_READ_AT in EM.V.READ_AT and EM.LADDER_READ_AT == "ladder_final"
    assert EM.LADDER_INSTRUMENT_LABEL.endswith("_merged") and EM.LADDER_INSTRUMENT_LABEL != EM.M16K_INSTRUMENT_LABEL
    # the merge tool's current bytes are pinned (LADDER_MERGE_TOOL_FILE_SHA256): a change to the tool forces a conscious decision about the identity
    # it names. The identity (LADDER_MERGE_TOOL: the commit, blob and sha256 the merged roots were written under) moves only with the merge rule; the
    # 2026-10-19 edit (refusals before any write, a usage text naming `python` and `$GRADE`) left the rule, and so the identity, as it was
    tool = os.path.join(os.path.dirname(HERE), EM.LADDER_MERGE_TOOL["path"])
    assert os.path.isfile(tool) and _sha(tool) == EM.LADDER_MERGE_TOOL_FILE_SHA256, (
        "misalign_ladder_merge.py changed: re-pin LADDER_MERGE_TOOL_FILE_SHA256, and move LADDER_MERGE_TOOL (commit + blob + sha256) too if the merge rule changed")
    assert len(EM.LADDER_MERGE_TOOL["commit"]) == 40 and len(EM.LADDER_MERGE_TOOL["blob_sha1"]) == 40 and len(EM.LADDER_MERGE_TOOL["sha256"]) == 64
    assert len(EM.LADDER_SUPERSEDED_LENS_RESPONSES_SHA256) == 13 and "none__lens0" in EM.LADDER_SUPERSEDED_LENS_RESPONSES_SHA256.values()
    assert not set(EM.LADDER_SUPERSEDED_LENS_RESPONSES_SHA256) & set(EM.M16K_SUPERSEDED_LENS_RESPONSES_SHA256)
    assert EM.EmMisalignAdapter(root="/nonexistent", full22_pins=None, baseline_pins=None, behaviour_source={}).ladder_root is None


def test_discovery_one_group_per_merged_root_and_the_inert_copies_are_never_inputs(tmp_path):
    w = _world(tmp_path)
    ad = _adapter(w)
    groups = _groups(ad)
    assert {g for g in groups if g.startswith(EM.LADDER_GROUP_PREFIX)} == {G1, G1P}
    assert ad.last_facts["<ladder-discover>"]["skipped"] == {"_scratch_never_read": "a leading-underscore directory (a producer's quarantine or scratch); never read"}
    every = [i.uri for gs in groups.values() for i in gs]
    assert not any("POISONED" in u or "/em8q/" in u or "/serving/" in u or "/store/cells/" in u for u in every), every
    assert any(u.endswith("ladder_report.json") for u in every) and any(u.endswith("ladder_merge.log") for u in every) and any(u.endswith("store/rows.parquet") for u in every)
    assert ad.status_sources(G1) == (EM.LADDER_SOURCE,)


# ============================================================================================
# the merged reading
# ============================================================================================

def test_the_seed_root_files_the_merged_whole_battery_value_with_the_rung_shares_and_the_residual(tmp_path):
    w = _world(tmp_path)
    recs, facts, rep = _read(_adapter(w), G1)
    assert rep.runs == 4 and rep.contrasts == 0 and rep.contrasts_parked == 4 and rep.dispositions == 0 and rep.refusals == []
    assert facts["refused"] == [] and facts["kind"] == "seed" and facts["drawn_keys"] == [CONST_H, CTL2_H] and facts["engine_kinds"] == ["lora_native"]
    r = _run(recs, CONST_R, "agentic_leaking")
    assert r["lane_source_legacy"] == "misalign_ladder" and r["campaign"] == "sdfbase_full22" and r["model_id"] == "qwen35_4b_it" and r["view_legacy"] == S1
    assert r["producer_run_key_legacy"].startswith("misalign_ladder/qwen35_4b_s1/")
    assert (r["token_cap_tokens"], r["token_cap_ladder_json"], r["token_cap_rung_index"]) == (16384, [3000, 16384], None)
    assert r["harness_version"] == EM.LADDER_MERGE_TOOL["commit"] == r["code_sha"] and r["n_items_judged"] == 4
    m = next(x for x in recs["run_metric"] if x["run_id"] == r["run_id"])
    assert (m["k"], m["n"], m["value"]) == (2, 4, 50.0)
    inst = next(i for i in recs["instrument"] if i["instrument_id"] == r["instrument_id"])
    assert (inst["read_at"], inst["token_cap_tokens"], inst["token_cap_ladder_json"], inst["harness_config_hash"]) == ("ladder_final", 16384, [3000, 16384], EM.LADDER_INSTRUMENT_LABEL)
    assert inst["ladder_promotion_rule"] == EM.LADDER_MERGE_RULE and inst["decode_source"] == "runner_forced_T1" and inst["decode_temperature"] == 1.0 and inst["decode_top_p"] is None
    assert facts["canonical_seed_instrument_ids"]["agentic_leaking"] not in {i["instrument_id"] for i in recs["instrument"]}
    ex = r["extra"]
    assert ex["ladder_rung_shares"]["3000"] == dict(ex["ladder_rung_shares"]["3000"], n_rows=2, n_scored=2, k_harmful=1, n_blocked=0, n_at_cap=0)
    assert ex["ladder_rung_shares"]["16384"] == dict(ex["ladder_rung_shares"]["16384"], n_rows=2, n_scored=2, k_harmful=1, n_blocked=0, n_at_cap=1)
    assert ex["unresolved_truncation"] == dict(ex["unresolved_truncation"], n_at_ceiling=1, n_redrawn=2, share=0.25, n_at_cap_all=1, n_loops_kept_at_source_cap=0, n_unrerun=0, n=4)
    assert ex["final_at_ceiling_share"] == 0.25 and ex["read_at"] == "ladder_final" and ex["merge"]["tool"] == EM.LADDER_MERGE_TOOL and ex["merge"]["base_tree_sha"] == TREE_S
    deg = next(d for d in recs["run_degeneration"] if d["run_id"] == r["run_id"] and d["measure"] == "at_token_cap")
    assert (deg["numerator"], deg["denominator"]) == (1, 4)
    # the fitted key: the ctl2 relu with a loop kept at 3,000 (residual counts it, not as a ceiling row)
    r2 = _run(recs, CTL2_R, "agentic_leaking")
    assert r2["object_id"] == SHA[CTL2_R] and r2["producer_object_key_legacy"] == STAGED[CTL2_R]
    assert r2["extra"]["unresolved_truncation"] == dict(r2["extra"]["unresolved_truncation"], n_at_ceiling=0, n_at_cap_all=1, n_loops_kept_at_source_cap=1, n_redrawn=1)
    # decode provenance as in v6
    fc = r["field_confidence_json"]
    assert fc["decode_sent_json"]["confidence"] == "recorded" and "max_tokens 3000" in fc["decode_sent_json"]["source"] and fc["decode_resolved_json"]["confidence"] == "derived"
    assert r["decode_sent_json"] == {"temperature": 1.0} and r["decode_resolved_json"] == FILL
    assert (r["substrate"], r["substrate_source"]) == ("cluster_a", "recorded")
    assert len(recs["item_result"]) == 3 * 16 and len(recs["item_flags"]) == 16
    assert ex["rows_parquet_run_ids_legacy"] == sorted(f"misalign_{S1}.{h}.0e63221d" for h in S1_KEYS) and "COLLIDES" in ex["rows_parquet_run_id_note"]


def test_no_contrast_is_computed_the_block_is_parked_and_the_headline_reference_is_unresolved_without_the_panels(tmp_path):
    w = _world(tmp_path)
    recs, facts, rep = _read(_adapter(w), G1)
    assert rep.contrasts == 0 and rep.contrasts_parked == 4 and "run_contrast" not in recs
    r = _run(recs, CONST_R, "agentic_murder")
    m = next(x for x in recs["run_metric"] if x["run_id"] == r["run_id"])
    blk = m["extra"]["delta_block"]
    assert blk["anchor"]["anchor_state"] == "no_merged_baseline" and blk["delta"] is None and blk["anchor"]["anchor_run_id"] is None
    assert "lora_native canon base" in blk["anchor"]["needed"] and len(blk["anchor"]["never_used"]) == 2
    assert blk["anchor"]["canon_root_state"] == "absent" and "no canon root" in blk["anchor"]["canon_root_why"]
    assert r["extra"]["anchor_state"] == "no_merged_baseline" and r["extra"]["anchor_filed_as"] is None and r["extra"]["anchor_run_id"] is None
    assert r["extra"]["anchor_needed"] == EM.LADDER_ANCHOR_NEEDED and r["extra"]["in_root_none_measured"] is False
    assert facts["canon_anchor"]["state"] == "absent" and facts["anchor_runs"] is None and facts["anchors_bound"] == {"run_contrast": 0, "parked_binding": 0}
    href = r["extra"]["headline_none_reference"]
    assert href["state"] == "unresolved" and href["instrument_mismatch"] is True and href["run_id"] is None and href["lane_source_legacy"] == "em/agentic/qwen35_4b"
    assert facts["headline_none_reference"]["agentic_leaking"]["lane_source_legacy"] == "em/agentic/qwen35_4b_vendor_leak"


def test_a_blocked_verdict_is_no_row_and_a_key_never_redrawn_for_a_behaviour_still_files_the_whole_battery(tmp_path):
    w = _world(tmp_path)
    recs, facts, rep = _read(_adapter(w), G1P)
    assert rep.runs == 4 and rep.dispositions == 0 and rep.refusals == [] and facts["n_blocked_total"] == 1
    assert [s["path"].split(os.sep)[-1] for s in facts["skipped"]] == ["agentic.POISONED-conc200"]
    r = _run(recs, LORA_R, "agentic_murder")
    m = next(x for x in recs["run_metric"] if x["run_id"] == r["run_id"])
    assert (m["k"], m["n"], m["n_abstentions"]) == (1, 3, 1) and r["n_items_judged"] == 3 and r["extra"]["judge_blocked"] == 1 and r["extra"]["lower_bound"] is True
    items = [i for i in recs["item_result"] if i["run_id"] == r["run_id"] and i["metric_id"] == "harmful"]
    assert len(items) == 4 and sum(1 for i in items if i["label"] == "judge_blocked") == 1 and all(i["value"] is None for i in items if i["label"] == "judge_blocked")
    r2 = _run(recs, LDIFF_R, "agentic_leaking")
    assert r2["extra"]["ladder_rung_shares"]["16384"]["n_rows"] == 0 and r2["extra"]["unresolved_truncation"]["n_redrawn"] == 0 and r2["extra"]["unresolved_truncation"]["n_loops_kept_at_source_cap"] == 2
    assert r2["n_items_judged"] == 4 and r2["extra"]["paired_of"] == "s1"


# ============================================================================================
# refusals by name: the lens merge, the canon root, the report cross-checks
# ============================================================================================

def test_a_lens_merge_key_refuses_by_sha_by_rerun_dir_and_by_route_and_a_paired_csv_with_rows_refuses(tmp_path, monkeypatch):
    w = _world(tmp_path)
    lens_sha = _sha(os.path.join(w["rsl"], S1, "agentic", CTL2_H, "responses.jsonl"))
    with monkeypatch.context() as mp_:
        mp_.setattr(EM, "LADDER_SUPERSEDED_LENS_RESPONSES_SHA256", dict(EM.LADDER_SUPERSEDED_LENS_RESPONSES_SHA256, **{lens_sha: CTL2_H}))
        recs, facts, rep = _read(_adapter(w), G1)
    assert rep.runs == 2 and rep.dispositions == 2 and all("superseded lens generation" in x and lens_sha[:12] in x for x in facts["refused"])
    w2 = _world(tmp_path / "b")
    # the marker is the path shape under the grade root, whoever's home it is: one key's rerun_dir points at the lens tree
    mp = os.path.join(w2["rsl"], S1, "agentic", CONST_H, "meta.json")
    m = json.load(open(mp)); m["ladder"]["rerun_dir"] = f"/home/TESTUSER/si-steering-grade/misalign_16k/qwen35_4b/agentic/{CONST_H}"; _w(mp, m)
    recs2, facts2, rep2 = _read(_adapter(w2), G1)
    assert rep2.runs == 2 and any("meta.ladder.rerun_dir" in x and "lens tree" in x for x in facts2["refused"])
    w3 = _world(tmp_path / "c", s1_kw=dict(keys={CONST_H: dict(S1_KEYS[CONST_H], ident=LENS)}))
    recs3, facts3, rep3 = _read(_adapter(w3), G1)
    assert rep3.runs == 0 and all("lens route" in x for x in facts3["refused"])
    w4 = _world(tmp_path / "d", s1_kw=dict(paired_rows=True))
    with pytest.raises(AdapterRefusal, match="paired.csv carries 1 row"):
        _read(_adapter(w4), G1)


def test_a_canon_merged_root_is_refused_on_a_lens_base_and_admitted_on_a_lora_native_one(tmp_path):
    w = _world(tmp_path, canon_kw=dict(keys={"none": dict(roster="none", ident=PASS, kind="none", rows=S1_KEYS[CONST_H]["rows"]), CTL2_H: dict(S1_KEYS[CTL2_H], ident=LENS)}, engine_kind="lens"))
    with pytest.raises(AdapterRefusal, match="admitted only on a lora_native base provenance"):
        _read(_adapter(w), GC)
    w2 = _world(tmp_path / "b", canon_kw=dict(keys={"none": dict(roster="none", ident=PASS, kind="none", rows=S1_KEYS[CONST_H]["rows"]), CTL2_H: dict(S1_KEYS[CTL2_H])}))
    recs, facts, rep = _read(_adapter(w2), GC)
    assert rep.runs == 4 and facts["kind"] == "canon" and _run(recs, "none", "agentic_leaking")["object_id"] == "none"


def test_the_merge_report_and_the_meta_ladder_block_must_describe_the_record(tmp_path):
    def bump(rep):
        rep[f"agentic/{CONST_H}"]["leaking"]["at_cap_after_all"] += 1
    w = _world(tmp_path, s1_kw=dict(report_override=bump))
    recs, facts, rep = _read(_adapter(w), G1)
    assert rep.runs == 3 and any("at_cap_after_all 2 != the record's finish_reason=length count 1" in x for x in facts["refused"])
    w2 = _world(tmp_path / "b", s1_kw=dict(unrerun=1))
    recs2, facts2, rep2 = _read(_adapter(w2), G1)
    assert rep2.runs == 0 and all("ladder.unrerun 1" in x for x in facts2["refused"])
    w3 = _world(tmp_path / "c", s1_kw=dict(rerun_cap=8192))
    recs3, facts3, rep3 = _read(_adapter(w3), G1)
    assert rep3.runs == 0 and all("rerun_cap 8192" in x for x in facts3["refused"])
    w4 = _world(tmp_path / "d")
    os.remove(os.path.join(w4["rsl"], S1, "ladder_report.json"))
    with pytest.raises(AdapterRefusal, match="ladder_report.json is absent"):
        _read(_adapter(w4), G1)
    w5 = _world(tmp_path / "e")
    mp = os.path.join(w5["rsl"], S1, "agentic", CONST_H, "judged.jsonl")
    rows = [json.loads(l) for l in open(mp)]
    rows[0].pop("ladder_rung")
    _w(mp, rows, jsonl=True)
    recs5, facts5, rep5 = _read(_adapter(w5), G1)
    assert any("carries no ladder_rung" in x for x in facts5["refused"])


def test_the_instrument_guard_refuses_a_step_or_vendor_shaped_identity(tmp_path):
    w = _world(tmp_path)
    ad = _adapter(w)
    good = {"eval_id": "agentic_leaking", "decode_source": "runner_forced_T1", "decode_temperature": 1.0, "decode_seed_rule": "none", "token_cap_tokens": 16384,
            "token_cap_ladder_json": [3000, 16384], "read_at": "ladder_final", "harness_version": EM.LADDER_MERGE_TOOL["commit"], "harness_config_hash": EM.LADDER_INSTRUMENT_LABEL,
            "instrument_id": "0" * 64, "serving_path": "lora_native", "applied_positions_mode": "all"}
    for bad_field, msg in ((dict(read_at="fixed"), "not the merged identity"), (dict(harness_config_hash=EM.M16K_INSTRUMENT_LABEL), "not the merged identity"),
                           (dict(decode_top_p=0.8), "vendor-shaped"), (dict(token_cap_ladder_json=[16384]), "is not 16384 on")):
        batch = RecordBatch("x", G1)
        batch.add("instrument", dict(good, **bad_field), PRODUCER, NOW)
        with pytest.raises(AdapterRefusal, match=msg):
            ad._ladder_guard_instruments(batch, {"group": G1, "model": MODEL, "instrument_ids": {}})


# ============================================================================================
# version 9: a root with its own none, the paired.csv it may carry, the passthrough route, the canon binding of every other root
# ============================================================================================

def _canon_world(tmp_path, **canon_kw):
    kw = dict(keys=dict(CANON_KEYS), paired_against_none=True)
    kw.update(canon_kw)
    return _world(tmp_path, canon_kw=kw)


def test_a_root_with_its_own_none_admits_paired_rows_and_files_the_producers_paired_difference_against_its_none(tmp_path):
    # the fitted key has NO producer pair: its binding stands alone (parked with the none run bound, delta null)
    w = _canon_world(tmp_path, paired_omit={CTL2_H})
    ad = _adapter(w)
    recs, facts, rep = _read(ad, GC)
    assert rep.runs == 6 and rep.contrasts == 2 and rep.contrasts_parked == 2 and rep.dispositions == 0 and rep.refusals == []
    assert facts["in_root_none"] is True and facts["kind"] == "canon" and facts["engine_kinds"] == ["lora_native"] and "canon_anchor" not in facts
    assert facts["anchors_bound"] == {"run_contrast": 2, "parked_binding": 2} and sorted(facts["anchor_runs"]) == ["leaking", "murder"]
    assert len(facts["no_paired_row"]) == 2 and all(CTL2_H in x and "the binding stands" in x for x in facts["no_paired_row"])
    none_run = {b: _run(recs, "none", f"agentic_{b}") for b in ITEMS}
    assert {r["run_id"] for r in none_run.values()} == {v["run_id"] for v in facts["anchor_runs"].values()}
    # the none rows: the baseline itself, no block
    for b, nr in none_run.items():
        assert nr["extra"]["anchor_state"] == "is_merged_baseline" and nr["extra"]["anchor_run_id"] is None and nr["extra"]["in_root_none_measured"] is True
        assert next(m for m in recs["run_metric"] if m["run_id"] == nr["run_id"])["extra"].get("delta_block") is None
    # the prompted key: the producer's paired difference as a run_contrast against the none run of the same behaviour
    paired = {(rw["arm"], rw["behavior"]): rw for rw in _paired_against_none(CANON, CANON_KEYS, {h: [json.loads(l) for l in open(os.path.join(w["rsl"], CANON, "agentic", h, "judged.jsonl"))] for h in CANON_KEYS})}
    for b in ITEMS:
        r = _run(recs, CONST_R, f"agentic_{b}")
        c = next(c for c in recs["run_contrast"] if c["run_id"] == r["run_id"])
        prow = paired[(CONST_H, b)]
        assert c["anchor_run_id"] == none_run[b]["run_id"] and c["anchor_kind"] == "same_standup_none" and c["anchor_judgment_id"] == none_run[b]["judgment_id"] if "judgment_id" in none_run[b] else True
        assert abs(c["delta"] - 100.0 * prow["diff"]) < 1e-9 and abs(c["delta_lo"] - 100.0 * prow["boot_low"]) < 1e-9 and c["delta_p"] == 0.5 and c["n_pairs"] == prow["n_pairs"] == 4
        assert c["delta_method"] == EM.LADDER_SAME_ROOT_DELTA_METHOD and c["extra"]["paired"] is True and c["extra"]["cross_root"] is False and c["extra"]["cross_engine"] is False
        assert c["extra"]["engine"] == PASS and c["extra"]["anchor_engine"] == PASS and c["extra"]["anchor_run_id"] == none_run[b]["run_id"] and c["extra"]["n_pairs"] == 4
        ex = r["extra"]
        assert ex["anchor_state"] == "own_merged_baseline" and ex["anchor_filed_as"] == "run_contrast" and ex["anchor_run_id"] == none_run[b]["run_id"] and ex["anchor_needed"] is None
        assert ex["external_anchor"]["filed_as"] == "run_contrast" and ex["external_anchor"]["kind_legacy"] == "same_standup_none" and ex["external_anchor"]["cross_root"] is False
        assert "lora_native" in ex["anchor_why"] and '"temperature": 1.0' in ex["anchor_why"]
        assert r["delta"] == c["delta"] if "delta" in r else True
        assert next(m for m in recs["run_metric"] if m["run_id"] == r["run_id"])["extra"].get("delta_block") is None
        # the fitted key: bound, no producer pair, no contrast
        r2 = _run(recs, CTL2_R, f"agentic_{b}")
        assert not [c for c in recs["run_contrast"] if c["run_id"] == r2["run_id"]]
        blk = next(m for m in recs["run_metric"] if m["run_id"] == r2["run_id"])["extra"]["delta_block"]
        assert blk["anchor"]["anchor_state"] == "own_merged_baseline" and blk["anchor"]["anchor_run_id"] == none_run[b]["run_id"] and blk["paired"] is False
        assert blk["delta"] is None and "no agentic/harmful row" in blk["delta_method"] and blk["anchor"]["anchor_key"] == f"misalign_ladder|{CANON}|{b}|none"
        assert r2["extra"]["anchor_filed_as"] == "parked_binding" and r2["extra"]["anchor_run_id"] == none_run[b]["run_id"] and r2["extra"]["external_anchor"]["filed_as"] == "parked_binding"
    # the pure derivation (no emission) already names the states and the producer's delta
    rows, f2 = ad.ladder_rows(CANON)
    st = {(r["cell_id"], r["eval_id"]): r["_extra"]["anchor_state"] for r in rows}
    assert st[("none", "agentic_leaking")] == "is_merged_baseline" and st[(CONST_R, "agentic_murder")] == "own_merged_baseline" and st[(CTL2_R, "agentic_murder")] == "own_merged_baseline"
    assert f2["readings"][f"{CONST_H}/leaking"]["n_pairs"] == 4 and f2["readings"][f"{CTL2_H}/leaking"]["n_pairs"] is None


def test_a_root_with_no_none_still_refuses_a_non_empty_paired_csv_and_a_pair_against_another_ref_refuses(tmp_path):
    # the version-7 contract stands: a seed root (no none) with rows in paired.csv refuses by name
    w = _world(tmp_path, s1_kw=dict(paired_rows=True))
    with pytest.raises(AdapterRefusal, match="paired.csv carries 1 row"):
        _read(_adapter(w), G1)
    # so does a seed root carrying the producer's own pair shape (rows against a none it does not draw)
    w2 = _world(tmp_path / "b", s1_kw=dict(keys=dict(S1_KEYS, none=NONE_KEY)))
    os.remove(os.path.join(w2["rsl"], S1, "analysis", S1, "paired.csv"))
    import pandas as pd
    rows_by_key = {h: [json.loads(l) for l in open(os.path.join(w2["rsl"], S1, "agentic", h, "judged.jsonl"))] for h in dict(S1_KEYS, none=NONE_KEY)}
    pd.DataFrame(_paired_against_none(S1, dict(S1_KEYS, none=NONE_KEY), rows_by_key), columns=PAIRED_COLS).to_csv(os.path.join(w2["rsl"], S1, "analysis", S1, "paired.csv"), index=False)
    # ... that root DOES draw a none now, so it is admitted with its pairs (the rule is about the none, not the root's name)
    recs, facts, rep = _read(_adapter(w2), G1)
    assert facts["in_root_none"] is True and rep.contrasts == 4 and rep.contrasts_parked == 0 and rep.runs == 6
    # a pair against any ref but none refuses the root
    w3 = _canon_world(tmp_path / "c", paired_ref=CONST_H)
    with pytest.raises(AdapterRefusal, match="pairs against ref"):
        _read(_adapter(w3), GC)


def test_a_passthrough_identity_on_a_lora_native_standup_is_admitted(tmp_path):
    # the canon root records `proxy_vllm+passthrough+interleaved` on every key while the standup's engine is lora_native: the ROUTE, not a lens marker
    w = _canon_world(tmp_path, keys={"none": NONE_KEY, CONST_H: dict(S1_KEYS[CONST_H], ident=PASS), CTL2_H: dict(S1_KEYS[CTL2_H], ident=PASS)})
    recs, facts, rep = _read(_adapter(w), GC)
    assert rep.runs == 6 and rep.refusals == [] and facts["refused"] == [] and facts["engine_kinds"] == ["lora_native"]
    r = _run(recs, CTL2_R, "agentic_leaking")
    assert r["extra"]["serving_engine"] == PASS and r["extra"]["standup_engine"] == "lora_native" and r["extra"]["superseded_lens_generation_check"]["is_lens_generation"] is False
    meta = json.load(open(os.path.join(w["rsl"], CANON, "agentic", CTL2_H, "meta.json")))
    assert EM.EmMisalignAdapter._ladder_lens_check(meta, "0" * 64, PASS) is None and EM.EmMisalignAdapter._ladder_lens_check(meta, "0" * 64, LENS) is not None
    # the same root with a lens ENGINE is still refused, whatever the route says
    w2 = _canon_world(tmp_path / "b", keys={"none": NONE_KEY, CONST_H: dict(S1_KEYS[CONST_H], ident=PASS)}, engine_kind="lens")
    with pytest.raises(AdapterRefusal, match="admitted only on a lora_native base provenance"):
        _read(_adapter(w2), GC)


def test_a_root_with_no_none_binds_its_block_to_the_canon_none_run_when_the_canon_root_is_admitted(tmp_path):
    w = _canon_world(tmp_path)
    ad = _adapter(w)
    recs_c, facts_c, rep_c = _read(ad, GC)
    none_run = {b: _run(recs_c, "none", f"agentic_{b}")["run_id"] for b in ITEMS}
    recs, facts, rep = _read(ad, G1)
    assert rep.runs == 4 and rep.contrasts == 0 and rep.contrasts_parked == 4 and "run_contrast" not in recs and rep.refusals == []
    assert facts["canon_anchor"]["state"] == "read" and facts["canon_anchor"]["root"] == os.path.join(w["rsl"], CANON) and facts["canon_anchor"]["engine_kinds"] == ["lora_native"]
    assert facts["anchors_bound"] == {"run_contrast": 0, "parked_binding": 4} and {b: v["run_id"] for b, v in facts["anchor_runs"].items()} == none_run
    for cell in (CONST_R, CTL2_R):
        for b in ITEMS:
            r = _run(recs, cell, f"agentic_{b}")
            m = next(x for x in recs["run_metric"] if x["run_id"] == r["run_id"])
            blk = m["extra"]["delta_block"]
            an = blk["anchor"]
            assert an["anchor_state"] == "canon_merged_baseline" and an["anchor_run_id"] == none_run[b] and an["kind_legacy"] == "sibling_none" and an["cross_root"] is True
            assert an["needed"] is None and an["anchor_root"] == os.path.join(w["rsl"], CANON) and an["own_root"] == os.path.join(w["rsl"], S1)
            assert an["anchor_engine_kind"] == "lora_native" and an["anchor_decode_sent"] == {"temperature": 1.0} and an["anchor_is_common_term"] is True
            assert "lora_native on both roots" in an["why"] and '"temperature": 1.0' in an["why"] and "no delta is computed" in an["why"]
            assert blk["delta"] is None and blk["paired"] is False and blk["n_pairs"] is None and "canon_merged_baseline" in blk["delta_method"]
            assert an["instrument_match"]["decode_match"] is True and an["instrument_match"]["cap_matched"] is True
            assert an["instrument_match"]["fields_differing"] == ([] if cell == CONST_R else ["serving_path"])
            assert (an["k"], an["n"]) == (2, 4) if b == "leaking" else (an["k"], an["n"]) == (0, 4)
            ex = r["extra"]
            assert ex["anchor_state"] == "canon_merged_baseline" and ex["anchor_filed_as"] == "parked_binding" and ex["anchor_run_id"] == none_run[b] and ex["anchor_needed"] is None
            assert ex["external_anchor"]["anchor_run_id"] == none_run[b] and ex["external_anchor"]["cross_root"] is True and ex["external_anchor"]["resolution"]["state"] == "resolved"
            assert "bound by identity" in r["note"] if "note" in r else True
    # the pure derivation already decides the state from the canon root (the run id comes at read time)
    rows, f2 = ad.ladder_rows(S1)
    assert {r["_extra"]["anchor_state"] for r in rows} == {"canon_merged_baseline"} and all(r["_delta_block"]["anchor"]["anchor_run_id"] is None for r in rows)
    assert {r["_delta_block"]["anchor"]["anchor_key"] for r in rows} == {f"misalign_ladder|{CANON}|{b}|none" for b in ITEMS}
    # a canon root that REFUSES (lens engine) leaves every other root parked no_merged_baseline, the refusal named
    w2 = _canon_world(tmp_path / "b", engine_kind="lens")
    recs2, facts2, rep2 = _read(_adapter(w2), G1)
    assert rep2.contrasts_parked == 4 and facts2["canon_anchor"]["state"] == "refused" and "admitted only on a lora_native base provenance" in facts2["canon_anchor"]["why"]
    blk2 = next(x for x in recs2["run_metric"] if x["run_id"] == _run(recs2, CONST_R, "agentic_leaking")["run_id"])["extra"]["delta_block"]
    assert blk2["anchor"]["anchor_state"] == "no_merged_baseline" and blk2["anchor"]["anchor_run_id"] is None and blk2["anchor"]["canon_root_state"] == "refused"
    assert facts2["anchor_runs"] is None and "refused" in facts2["anchor_runs_why"]


def test_the_canon_binding_is_declined_on_an_engine_or_decode_mismatch_and_a_canon_root_never_binds_to_itself(tmp_path):
    w = _canon_world(tmp_path)
    ad = _adapter(w)
    anchor = ad._ladder_canon_anchor(MODEL)
    assert anchor["state"] == "read" and sorted(anchor["per_behaviour"]) == ["leaking", "murder"]
    anchor["per_behaviour"]["leaking"]["engine_kind"] = "lens"
    anchor["per_behaviour"]["murder"]["decode_sent"] = {"temperature": 0.7}
    rows, facts = ad.ladder_rows(S1)
    by = {(r["cell_id"], r["eval_id"]): r["_delta_block"]["anchor"] for r in rows}
    assert by[(CONST_R, "agentic_leaking")]["anchor_state"] == "no_merged_baseline" and "standup engine 'lora_native' here vs 'lens'" in by[(CONST_R, "agentic_leaking")]["why"]
    assert by[(CONST_R, "agentic_murder")]["anchor_state"] == "no_merged_baseline" and "decode sent" in by[(CONST_R, "agentic_murder")]["why"] and "NOT bound" in by[(CONST_R, "agentic_murder")]["why"]
    assert by[(CONST_R, "agentic_leaking")]["mismatch"] and by[(CONST_R, "agentic_leaking")]["anchor_run_id"] is None
    # a canon root without a none of its own has no baseline and consults no canon anchor (no recursion): its keys park no_merged_baseline
    w2 = _world(tmp_path / "b", canon_kw=dict(keys={CONST_H: dict(S1_KEYS[CONST_H]), CTL2_H: dict(S1_KEYS[CTL2_H])}))
    rows2, facts2 = _adapter(w2).ladder_rows(CANON)
    assert facts2["in_root_none"] is False and "canon_anchor" not in facts2 and {r["_extra"]["anchor_state"] for r in rows2} == {"no_merged_baseline"}


def test_the_shape_equals_em_misaligns_misalignment_rows(tmp_path):
    w = _world(tmp_path)
    ad = _adapter(w)
    rows, _facts = ad.ladder_rows(S1)
    panel_rows, _f = ad.long_rows("gemma4e4b")
    public = lambda r: {k for k in r if not k.startswith("_")}  # noqa: E731
    assert public(rows[0]) == public(panel_rows[0]) and {r["status"] for r in rows} == {"measured"}
