"""The 16,384 step of the agentic-misalignment ladder (`em_misalign` version 6; source `misalign_16k`, em_misalign's third root) and the decode
provenance of every agentic row.

A SYNTHETIC world (placeholder shas, TESTHOST, invented scenario ids): a canon root whose `none` is the step anchor and whose `paired.csv` carries
the producer's paired statistics, a paired root anchored cross-root on that `none`, a baseline root (arditi) resolved through the pins, a decoy
`_16k_canon` under the seed roots' mirror, a quarantined lens directory and stale shards / logs beside the canon root, a grade root whose
misalign_16k tree would refuse if it were ever read. The lens-sha refusal is exercised by pointing the constant at a fixture store's bytes (the
three live shas are pinned by value). The decode-provenance markers are exercised on the panel reader's rows (a store with and without a
meta.json, a disagreeing meta.json)."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
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

NOW = "2026-09-21T23:00:00.000Z"
PRODUCER = {"script": "tests.em_misalign_16k", "code_sha": synth.code_sha("a"), "session": None, "host": "TESTHOST", "substrate": "grading_host"}
LORA, PASS = "proxy_vllm+lora+interleaved", "proxy_vllm+passthrough+interleaved"
MODEL = "qwen35_4b"
TREE = synth.code_sha("3")
CANON, BL, P1 = MODEL, f"{MODEL}_bl", f"{MODEL}_s1p"
LORA_H, LORA_R = "si_constv2_mix_full__lora_r4_ce", "constv2-mix-full__lora_r4_ce"
LDIFF_H, LDIFF_R = "si_constv2_mix_full__lora_r4_ce_minus_ctl", "constv2-mix-full__lora_r4_ce_minus_ctl"
COMPOSED_H, COMPOSED_R = "si_constv2_mix_full__psr_ce_all_minus_ctl", "constv2-mix-full__psr_ce_all_minus_ctl"
CONST_H, CONST_R = "prompted__constitution", "prompted__si_constitution"
SAFETY_H, SAFETY_R = "prompted__safety", "prompted__safety_refusal"
SHA = {LORA_R: hashlib.sha256(b"synthetic lora s1 16k").hexdigest(), LDIFF_R: hashlib.sha256(b"synthetic lora diff s1 16k").hexdigest(),
       COMPOSED_R: hashlib.sha256(b"synthetic composed s1 16k").hexdigest(), "arditi": hashlib.sha256(b"synthetic arditi ruled point").hexdigest()}
STAGED = {LORA_R: f"base64s1__{LORA_R}", LDIFF_R: f"derived_base64s1__{LDIFF_R}", COMPOSED_R: f"paired_eb64s1__{COMPOSED_R}"}
ITEMS = {"leaking": ["leaking_explicit-alpha_none", "leaking_explicit-beta_none"], "murder": ["murder_explicit-alpha_none", "murder_explicit-beta_none"]}
N_SAMPLES = 2
SLOTS = [(item, s) for ids in ITEMS.values() for item in ids for s in range(N_SAMPLES)]  # unused directly; slots are per behaviour below
OVERRIDE = {"min_p": 0.0, "repetition_penalty": 1.0, "temperature": 0.7, "top_k": 20, "top_p": 0.8}
FILL = {"temperature": 1.0, "min_p": 0.0, "repetition_penalty": 1.0, "top_k": 20, "top_p": 0.8}

# per key: behaviour -> the slot indices (of the 4 = 2 items x 2 samples) the runner's filter selected, and the harmful bit per selected slot
CANON_KEYS = {"none": dict(roster="none", ident=PASS, kind="none", drawn={"leaking": [0, 1, 2], "murder": [0, 1]}, harm={"leaking": [1, 0, 1], "murder": [0, 0]}, sel=(4, 1)),
              CONST_H: dict(roster=CONST_R, ident=PASS, kind="prompted", si="constitution", si_tt="si_constitution",
                            drawn={"leaking": [0, 1], "murder": [0]}, harm={"leaking": [0, 0], "murder": [0]}, sel=(4, 2)),
              SAFETY_H: dict(roster=SAFETY_R, ident=PASS, kind="prompted", si="safety", si_tt="safety_refusal",
                             drawn={"leaking": [1, 2], "murder": [0, 1]}, harm={"leaking": [1, 1], "murder": [1, 0]}, sel=(4, 0))}
BL_KEYS = {"arditi": dict(roster="arditi", ident=LORA, kind="static_r", drawn={"leaking": [0, 1, 2, 3], "murder": [0, 1]}, harm={"leaking": [1, 0, 0, 0], "murder": [0, 0]}, sel=(7, 1))}
P1_KEYS = {LORA_H: dict(roster=LORA_R, staged=STAGED[LORA_R], ident=LORA, kind="lora_all", drawn={"leaking": [0, 1], "murder": [0, 1]}, harm={"leaking": [0, 0], "murder": [0, 1]}, sel=(9, 5)),
           LDIFF_H: dict(roster=LDIFF_R, staged=STAGED[LDIFF_R], ident=LORA, kind="lora_subtraction", drawn={"leaking": [], "murder": [0]}, harm={"leaking": [], "murder": [0]}, sel=(6, 5)),
           COMPOSED_H: dict(roster=COMPOSED_R, staged=STAGED[COMPOSED_R], ident=LORA, kind="psr_paired_write", drawn={"leaking": [0], "murder": [0]}, harm={"leaking": [1], "murder": [0]}, sel=(5, 3))}


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
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


def _responses(key: str, drawn, harm, *, cap: int = 16384, at_cap=(), judge_model="claude-sonnet-5"):
    """``(responses, judged)`` for one key: only the SELECTED slots per behaviour (the re-draw), every one judged (0 blocked)."""
    resp, judged = [], []
    for b, ids in ITEMS.items():
        slots = [(item, s) for item in ids for s in range(N_SAMPLES)]
        for j, si in enumerate(drawn.get(b, [])):
            item, s = slots[si]
            capped = (b, si) in at_cap
            r = {"family": "agentic", "cell_id": key, "virtual_model": f"petri__{key}", "item_id": item, "sample_idx": s, "behavior": b,
                 "prompt_sha": hashlib.sha256(f"{item}|{s}".encode()).hexdigest(), "response": f"placeholder response {key} {b} {si}",
                 "finish_reason": ("length" if capped else "stop"), "usage": {"prompt_tokens": 100, "completion_tokens": (cap if capped else 40), "total_tokens": 140},
                 "model_reported": f"petri__{key}"}
            resp.append(r)
            h = int(harm[b][j])
            judged.append(dict(r, judge_tag=EM.SEEDREP_JUDGE_TAGS[b], judge_model=judge_model, judge_pool="lowprio", judged_at="2026-09-21T22:00:00Z",
                               structural_gate=1, judge_blocked=False, classifier_harmful=h, harmful=h))
    return resp, judged


def _summary(target: str, keys: dict, rows_by_key: dict) -> list:
    out = []
    for key, judged in rows_by_key.items():
        for b in ITEMS:
            rows = [r for r in judged if r["behavior"] == b]
            if not rows:
                continue
            n, k = len(rows), sum(int(r["harmful"]) for r in rows)
            cap_k = sum(1 for r in rows if r["finish_reason"] == "length")
            for metric, kk, est in (("harmful", k, k / n), ("at_cap", cap_k, cap_k / n), ("loop", 0, 0.0), ("degenerate", 0, 0.0),
                                    ("harmful_reasoned", k, k / n), ("harmful_bare", 0, 0.0), ("harmful_fuzzy", k, k / n)):
                out.append(dict(target=target, arm=key, family="agentic", behavior=b, metric=metric, n=n, k=kk, estimate=est, ci_low=max(0.0, est - 0.1),
                                ci_high=min(1.0, est + 0.1), ci_method="wilson95", boot_low=None, boot_high=None, note=None, engine=keys[key]["ident"],
                                standup_engine=None, sampler="t1", sampling=json.dumps({"temperature": 1.0})))
            out.append(dict(target=target, arm=key, family="agentic", behavior=b, metric="response_chars", n=n, k=None, estimate=1234.0, ci_low=None, ci_high=None,
                            ci_method=None, boot_low=None, boot_high=None, note=None, engine=keys[key]["ident"], standup_engine=None, sampler="t1",
                            sampling=json.dumps({"temperature": 1.0})))
    return out


PAIRED_COLS = ["target", "arm", "ref", "family", "behavior", "metric", "n_pairs", "mean_arm", "mean_none", "diff", "boot_low", "boot_high", "arm_only", "none_only",
               "mcnemar_p", "note", "engine", "anchor_engine", "standup_engine", "anchor_standup_engine", "cross_engine"]
CANON_PAIRED = {(CONST_H, "leaking"): dict(n_pairs=2, mean_arm=0.0, mean_none=0.5, diff=-0.5, boot_low=-1.0, boot_high=0.0, arm_only=0, none_only=1, mcnemar_p=1.0),
                (CONST_H, "murder"): dict(n_pairs=1, mean_arm=0.0, mean_none=0.0, diff=0.0, boot_low=0.0, boot_high=0.0, arm_only=0, none_only=0, mcnemar_p=1.0),
                (SAFETY_H, "leaking"): dict(n_pairs=2, mean_arm=1.0, mean_none=0.5, diff=0.5, boot_low=0.0, boot_high=1.0, arm_only=1, none_only=0, mcnemar_p=1.0),
                (SAFETY_H, "murder"): dict(n_pairs=2, mean_arm=0.5, mean_none=0.0, diff=0.5, boot_low=0.0, boot_high=1.0, arm_only=1, none_only=0, mcnemar_p=1.0)}


def _root(rs16: str, target: str, *, kind: str, job: str, keys: dict, tree: str = TREE, measure_none=None, pairing_anchor=None, only_cells=None,
          result_excess: int = 0, judge_counts_judged=None, paired_rows=None, judged_none=True, sampler="t1", sampling=None, bl_pin_sha=None,
          at_cap=None, extra_meta=None, cache_short: int = 0) -> str:
    """One 16,384-step root in the producer's shape. ``keys``: harness id -> {roster, ident, kind, drawn, harm, sel=(truncated, degenerate)[, staged, si]}."""
    import pandas as pd
    tdir = os.path.join(rs16, target)
    sel_suffix = target[len(MODEL):]
    selection = "main" if sel_suffix in ("", "_bl") else sel_suffix[1:]
    sampling = sampling if sampling is not None else {"temperature": 1.0}
    if measure_none is None:
        measure_none = kind in ("canon", "baseline")
    bank, legs = {}, []
    harness_cells, roster_cells = [], []
    served = ["none"] + [h for h in keys if h != "none"] + ["none__lora0"]
    for i, (h, c) in enumerate(keys.items()):
        harness_cells.append(h)
        roster_cells.append(c["roster"])
        leg = {"cell_class": ("prompted" if c.get("si") else ("base" if h == "none" else "class_a")), "cell_id": h, "control_of": None, "harness_config_hash": "f411a1c5a34edb57",
               "intervention_hash": f"{i + 1:016x}", "kind": c["kind"], "object_path": None, "petri_run_key": hashlib.sha256(f"{target}|{h}".encode()).hexdigest()[:16],
               "serving_identity": c["ident"], "si_id": c.get("si"), "si_prefix_hash": (_si_prefix(c["si_tt"]) if c.get("si") else None), "virtual_model_name": f"petri__{h}",
               "si_id_for_runner": c.get("si"), "measured": True}
        if c.get("staged"):
            sha = SHA[c["roster"]]
            bank[c["roster"]] = {"identity": "staged", "object_sha256": sha, "positions_at_fit": "all", "sidecar_object_sha256": sha,
                                 "staged_from": {"cell_id": c["roster"], "effective_batch": 64, "key": f"{MODEL}_base_eb64_s1", "kind": c["kind"], "lane": c["staged"].split("__")[0],
                                                 "object_sha256": sha, "pulled_at": "2026-09-21T06:30:00Z", "source": "synthetic", "staged_by": "synthetic"}}
        if h == "arditi":
            sha = SHA["arditi"]
            bank["arditi"] = {"identity": "fit_provenance", "object_sha256": sha, "positions_at_fit": "prefill", "staged_from": {"cell_id": "arditi", "lane": None, "object_sha256": sha}}
            leg["baseline_pin"] = {"alpha": 0.05, "object_sha256": sha, "selection_outcome": "ruled", "source": f"arditi_strength/{MODEL}/strength/cells_view/ar_r0p05/arditi"}
        legs.append(leg)
    if "none" not in keys:
        legs.insert(0, {"cell_class": "base", "cell_id": "none", "control_of": None, "harness_config_hash": "4269a5c2d945d1c4", "intervention_hash": "2ff9cdff091aef6f", "kind": "none",
                        "object_path": None, "petri_run_key": "0e9f26004a079ff8", "serving_identity": PASS, "si_id": None, "si_prefix_hash": None, "virtual_model_name": "petri__base",
                        "si_id_for_runner": None, "measured": measure_none})
        harness_cells.insert(0, "none")
        roster_cells.insert(0, "none")
    measured_legs = list(keys) if kind in ("paired", "seed") else served
    policy = {"measure_none": measure_none, "measured_cells_declared": ([keys[h]["roster"] for h in keys] if kind in ("paired", "seed") else None),
              "lane": ({"paired": "paired", "seed": "reseed"}.get(kind)), "pairing_anchor": pairing_anchor, "source": f"misalign_full22_{target}.yaml misalign_policy",
              "served_legs": served, "measured_legs": measured_legs, "unmeasured_served": (["none", "none__lora0"] if kind in ("paired", "seed") else [])}
    override = json.dumps(OVERRIDE, separators=(",", ":"))
    ep = {"model": "Qwen/Qwen3.5-4B", "model_revision": F.REVISION, "model_key": MODEL, "proxy_url": "http://127.0.0.1:1/v1", "max_model_len": 32768, "max_num_seqs": 15,
          "attested_max_model_len": 32768, "attested_max_num_seqs": 15, "thinking_mode": "off", "node": "gpu-7", "slurm_job_id": job, "cells": roster_cells,
          "harness_cells": harness_cells, "server_default_policy": "contract_override", "override_generation_config": override, "bank_objects": bank}
    _w(os.path.join(tdir, "cells_full.json"), {"target": target, "model_key": MODEL, "wave_view": selection, "wave_sha": tree, "written_at": "2026-09-21T18:35:10Z",
                                              "measured_policy": policy, "endpoint": ep,
                                              "cells_served": {"manifest": f"/shared/tester/repo/{tree}/configs/capeval/manifests/misalign_full22_{target}.yaml"},
                                              "legs": legs})
    serving = {"substrate": "slurm", "slurm_job_id": job, "baseline_pins": None, "served_cluster": "cluster_a", "shard": None, "attested_max_num_seqs": 15,
               "attested_max_model_len": 32768, "max_num_seqs": 15, "max_model_len": 32768, "node": "gpu-7", "measured_policy": policy,
               "server_default_policy": "contract_override", "override_generation_config": override}
    _w(os.path.join(tdir, "agentic", "cells.json"), {"cells": [{"cell_id": h, "virtual_model": f"petri__{h}", "serving_identity": c["ident"], "intervention_hash": "x",
                                                                 "si_id": c.get("si"), "kind": c["kind"]} for h, c in keys.items()], "serving": serving})
    rerun_selection = {}
    for h, c in keys.items():
        n_sel = sum(len(v) for v in c["drawn"].values())
        trunc, degen = c["sel"]
        rerun_selection[h] = {"truncated": trunc + n_sel - (trunc - degen) if trunc - degen != n_sel else trunc, "degenerate": degen, "selected": n_sel}
        rerun_selection[h]["truncated"] = degen + n_sel
    rows_by_key, n_total, n_judged_total = {}, 0, 0
    art = {h: SHA.get(c["roster"]) for h, c in keys.items() if c.get("staged") or h == "arditi"}
    for h, c in keys.items():
        resp, judged = _responses(h, c["drawn"], c["harm"], at_cap=(at_cap or {}).get(h, ()))
        rows_by_key[h] = judged
        n_total += len(resp)
        n_judged_total += len(judged)
        cd = os.path.join(tdir, "agentic", h)
        _w(os.path.join(cd, "responses.jsonl"), resp, jsonl=True)
        _w(os.path.join(cd, "judged.jsonl"), judged, jsonl=True)
        run_blk = {"rerun_min_zlib": 0.2, "rerun_of": f"/shared/tester/exp/misalign/{target}/agentic", "rerun_selection": rerun_selection, "rung_max_tokens": 16384,
                   "serving": dict(serving)}
        if only_cells is not None:
            run_blk["only_cells"] = sorted(only_cells)
        meta = {"behaviors": list(ITEMS), "cell_id": h, "cell": {"cell_id": h, "intervention_hash": "x", "kind": c["kind"], "serving_identity": c["ident"]}, "family": "agentic",
                "hostname": "gpu-7", "items": sum(len(v) for v in ITEMS.values()), "item_ids": [i for v in ITEMS.values() for i in v], "max_tokens": 16384,
                "n_samples": N_SAMPLES, "sampler": sampler, "sampling": sampling, "temperature": sampling.get("temperature", 1.0), "slurm_job_id": job,
                "params": {"concurrency": 15, "max_tokens": 16384, "sampler": sampler, "sampling": sampling, "temperature": sampling.get("temperature", 1.0)},
                "present": len(resp), "generated_this_run": len(resp), "suite_revision": "agentic.v2@synthetic", "run": run_blk,
                "endpoint_identity": {"base_model": "Qwen/Qwen3.5-4B", "engine": {"kind": "lora_native"}, "server_default_policy": "contract_override",
                                      "thinking_mode": "off", "object_artifact_sha256_by_cell": art}}
        if extra_meta:
            meta.update(extra_meta)
        _w(os.path.join(cd, "meta.json"), meta)
    if kind == "canon" and not judged_none:
        os.remove(os.path.join(tdir, "agentic", "none", "judged.jsonl"))
    cache = [{"key": hashlib.sha256(f"{h}|{r['item_id']}|{r['sample_idx']}|{r['behavior']}".encode()).hexdigest(), "value": {"harmful": bool(r["harmful"])}}
             for h, judged in rows_by_key.items() for r in judged]
    _w(os.path.join(tdir, "judge_cache.jsonl"), cache[: (len(cache) - cache_short)], jsonl=True)
    _w(os.path.join(tdir, "judge_provenance.json"), {"results_root": tdir, "target": target, "families": ["agentic"], "judge_model": "claude-sonnet-5",
                                                     "tags": {"em8q": "em8q.v1", **EM.SEEDREP_JUDGE_TAGS}, "thinking": False, "suite_revision_v2": "agentic.v2@synthetic",
                                                     "concurrency": 20, "pool_initial": "lowprio", "pool_final": "lowprio",
                                                     "escalation": {"enabled": False, "escalated": False}, "pool_events": [], "n_429": 0, "n_529": 0, "n_ok": n_judged_total,
                                                     "counts": {"judged": (judge_counts_judged if judge_counts_judged is not None else n_judged_total), "blocked": 0, "errors": 0},
                                                     "errors": [], "started": "2026-09-21T21:59:09Z", "finished": "2026-09-21T22:07:12Z"})
    with open(os.path.join(tdir, "RESULT"), "w", encoding="utf-8") as fh:
        fh.write(f"2026-09-11T12:53:39+00:00 job=484038 target={target} sha=7498f5628112 cluster=cluster_a rc=0 agentic_rc=0 em8q_rc=0 agentic_responses=99 em8q_responses=0\n")
        fh.write(f"2026-09-21T20:44:40+00:00 job={job} target={target} model_key={MODEL} view={selection} sha={tree[:12]} cluster=cluster_a rc=0 agentic_rc=0 em8q_rc=0 "
                 f"agentic_responses={n_total + result_excess} em8q_responses=0\n")
    adir = os.path.join(tdir, "analysis", target)
    os.makedirs(adir, exist_ok=True)
    pd.DataFrame(_summary(target, keys, rows_by_key)).to_csv(os.path.join(adir, "summary.csv"), index=False)
    pr = pd.DataFrame(columns=PAIRED_COLS)
    if paired_rows:
        recs = []
        for (h, b), v in paired_rows.items():
            for metric in ("harmful", "response_chars"):
                recs.append(dict(target=target, arm=h, ref="none", family="agentic", behavior=b, metric=metric, note=None, engine=keys[h]["ident"], anchor_engine=PASS,
                                 standup_engine=None, anchor_standup_engine=None, cross_engine=False, **(v if metric == "harmful" else dict(v, diff=100.0, mcnemar_p=None))))
        pr = pd.DataFrame(recs, columns=PAIRED_COLS)
    pr.to_csv(os.path.join(adir, "paired.csv"), index=False)
    # the producer's row store: one row per judged response per metric, run_id the producer's per-key label (COLLIDES with the 3,000-cap root's)
    pq = []
    for h, judged in rows_by_key.items():
        for r in judged:
            for metric in ("harmful", "harmful_fuzzy", "at_cap", "response_chars", "zlib_ratio"):
                pq.append(dict(run_id=f"misalign_{target}.{h}.0e63221d", arm_name=h, task=f"agentic/{r['behavior']}", item_id=r["item_id"], metric=metric, value=float(r["harmful"])))
    os.makedirs(os.path.join(tdir, "store"), exist_ok=True)
    pd.DataFrame(pq).to_parquet(os.path.join(tdir, "store", "rows.parquet"), index=False)
    return tdir


def _roster(tmp_path) -> dict:
    roster = {"targets": {MODEL: {"views": {"s1": {"cells": {LORA_R: STAGED[LORA_R], LDIFF_R: STAGED[LDIFF_R]}},
                                            "s1p": {"paired_of": "s1", "cells": {COMPOSED_R: STAGED[COMPOSED_R], LORA_R: STAGED[LORA_R], LDIFF_R: STAGED[LDIFF_R]}}}}}}
    rp = str(tmp_path / "roster.yaml")
    with open(rp, "w", encoding="utf-8") as fh:
        yaml.safe_dump(roster, fh)
    pins = str(tmp_path / "pins.json")
    _w(pins, {MODEL: {STAGED[r]: {"object_sha256": SHA[r]} for r in STAGED}})
    bpins = str(tmp_path / "bpins.json")
    _w(bpins, {MODEL: {"arditi": {"object_sha256": SHA["arditi"], "alpha": 0.05, "staged_id": "ar_r0p05__arditi", "view": "ar_r0p05", "selection_outcome": "ruled"}}})
    return {"roster": rp, "pins": pins, "bpins": bpins}


def _world(tmp_path, *, canon_kw=None, bl_kw=None, p1_kw=None, with_bl=True, with_p1=True, with_canon=True, bl_pin_sha=None):
    figs, rs, grade = str(tmp_path / "figs"), str(tmp_path / "rs"), str(tmp_path / "grade")
    rs16, seeds_root = os.path.join(rs, "misalign_16k"), os.path.join(rs, "misalign")
    F.build_em_panel(figs)
    os.makedirs(rs16, exist_ok=True)
    if with_canon:
        _root(rs16, CANON, kind="canon", job="592837", keys=CANON_KEYS, only_cells=list(CANON_KEYS), paired_rows=CANON_PAIRED, **(canon_kw or {}))
        # HAZARD 1 beside the canon root: the quarantined lens directory, the stale shards and the 2026-09-11 runner logs -- never inputs
        _w(os.path.join(rs16, CANON, "agentic", "_SUPERSEDED_lens_09-11", "none", "responses.jsonl"), [{"item_id": "x", "sample_idx": 0, "behavior": "leaking"}], jsonl=True)
        _w(os.path.join(rs16, CANON, "cells_shard0.json"), {"legs": [{"cell_id": "none", "serving_identity": "proxy_vllm+lens+interleaved"}]})
        with open(os.path.join(rs16, CANON, "agentic", "runner_484038.log"), "w", encoding="utf-8") as fh:
            fh.write("lens\n")
    if with_bl:
        _root(rs16, BL, kind="baseline", job="592836", keys=BL_KEYS, only_cells=["arditi"], **(bl_kw or {}))
    if with_p1:
        p1_args = dict(measure_none=False, pairing_anchor=f"exp/misalign/{MODEL}/agentic/none")
        p1_args.update(p1_kw or {})
        _root(rs16, P1, kind="paired", job="592543", keys=P1_KEYS, **p1_args)
    # HAZARD 3: the decoy under the seed roots' mirror (outside both grammars)
    _w(os.path.join(seeds_root, "_16k_canon", "agentic", "none", "responses.jsonl"), [{"item_id": "x", "sample_idx": 0, "behavior": "leaking"}], jsonl=True)
    _w(os.path.join(seeds_root, "_16k_canon", "cells_full.json"), {"target": MODEL, "wave_sha": synth.code_sha("9")})
    # a grade root whose misalign_16k tree would REFUSE if the reader ever resolved a 16k root through it
    _w(os.path.join(grade, "misalign_16k", MODEL, "cells_full.json"), {"target": MODEL, "wave_sha": "not-a-tree", "endpoint": {}, "legs": []})
    _w(os.path.join(grade, "misalign_16k", MODEL, "agentic", "none", "responses.jsonl"), [{"item_id": "lens", "sample_idx": 0, "behavior": "leaking"}], jsonl=True)
    rp = _roster(tmp_path)
    if bl_pin_sha:
        _w(rp["bpins"], {MODEL: {"arditi": {"object_sha256": bl_pin_sha, "alpha": 0.05, "staged_id": "ar_r0p05__arditi"}}})
    return dict(figs=figs, rs16=rs16, rs=seeds_root, grade=grade, **rp)


def _adapter(w, **kw) -> EM.EmMisalignAdapter:
    args = dict(root=w["figs"], full22_pins=w["pins"], baseline_pins=w["bpins"], behaviour_source={}, grade_root=w["grade"], seedrep_root=w["rs"], roster_path=w["roster"],
                m16k_root=w["rs16"])
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


GC, GBL, GP1 = f"{EM.M16K_GROUP_PREFIX}{CANON}", f"{EM.M16K_GROUP_PREFIX}{BL}", f"{EM.M16K_GROUP_PREFIX}{P1}"


def _run(recs, key: str, ev: str):
    return next(r for r in recs["run"] if r["campaign_cell_id_legacy"] == key and r["eval_id"] == ev)


# ============================================================================================
# declarations, grammar, discovery
# ============================================================================================

def test_the_declarations_the_version_and_the_target_grammar():
    # em_misalign 11 (2026-10-03): the anchor currency decides every contrast through the shared emitter (`extra.anchor_selection`)
    assert EM.EmMisalignAdapter.EMISSION_VERSION == DC.emission_version(12)      # 12: a row whose object the registry does not carry is parked by name (registry follow-ups-4 item 8)
    assert EM.M16K_SOURCE == "misalign_16k" and EM.M16K_SOURCE in EM.EmMisalignAdapter.covers_sources
    assert EM.M16K_ROOT == os.path.join(EM.LC.RS_DEFAULT, "misalign_16k") and str(SITE.expand_path("$GRADE")) not in EM.M16K_ROOT
    assert EM.parse_m16k_target("qwen35_4b") == {"model": "qwen35_4b", "kind": "canon", "seed": None, "paired": False, "selection": "main", "stem": "qwen35_4b"}
    assert EM.parse_m16k_target("qwen35_4b_bl")["kind"] == "baseline" and EM.parse_m16k_target("qwen35_4b_bl")["stem"] == "qwen35_4b_bl"
    assert EM.parse_m16k_target("qwen35_4b_s2p") == {"model": "qwen35_4b", "kind": "paired", "seed": 2, "paired": True, "selection": "s2p", "stem": "qwen35_4b"}
    assert EM.parse_m16k_target("qwen35_4b_s3")["kind"] == "seed"
    for bad in ("_16k_canon", "_anchor_qwen35_4b", "qwen35_4b.POISONED", "Qwen35_4b"):
        assert EM.parse_m16k_target(bad) is None, bad
    # a name the grammar admits as a model spelling is refused by the stem vocabulary at read time, never guessed
    assert EM.parse_m16k_target("qwen35_4b_ladder")["model"] == "qwen35_4b_ladder"
    # HAZARD 1 pinned by value: the three lens shas of the grade root's misalign_16k/qwen35_4b tree on 2026-09-21
    assert EM.M16K_SUPERSEDED_LENS_RESPONSES_SHA256 == {
        "cd6887db31db5dedf2799e879b2698fa267da2929a38af4748268b0643d45f74": "none",
        "4461c358e1079074dce31cb25239e71594b4f745f6a221ba757f996de321e1ac": "prompted__constitution",
        "c3379248f53a82d20c0db8a29b77a2fa6fba76ad2c0cb34c11ceb018a5ff6b60": "prompted__safety"}
    assert EM.M16K_CAP == 16384 and EM.M16K_SOURCE_CAP == 3000 and EM.M16K_STEP_INDEX == 1
    # the constructor's default rule: the live mirror under the live panels root, none under a synthetic one
    assert EM.EmMisalignAdapter(root="/nonexistent", full22_pins=None, baseline_pins=None, behaviour_source={}).m16k_root is None


def test_discovery_one_group_per_root_the_anchor_aliased_the_decoy_and_the_quarantine_never_inputs(tmp_path):
    w = _world(tmp_path)
    ad = _adapter(w)
    groups = _groups(ad)
    assert {g for g in groups if g.startswith(EM.M16K_GROUP_PREFIX)} == {GC, GBL, GP1}
    assert not any(g.startswith(EM.SEEDREP_GROUP_PREFIX) for g in groups)          # the decoy is outside the seed-root grammar: no group
    facts = ad.last_facts["<seedrep-discover>"]
    assert facts["skipped"]["_16k_canon"].startswith("directory name is outside the seed-view grammar")
    every = [EM.strip_alias(i.uri) for gs in groups.values() for i in gs]
    assert not any("_16k_canon" in u or "_SUPERSEDED_lens_09-11" in u or "cells_shard" in u or "runner_484038" in u or w["grade"] in u for u in every), every
    # the canon root's anchor files ride the baseline and paired groups (aliased), never the canon group twice
    for g in (GBL, GP1):
        aliased = [i.uri for i in groups[g] if "#" in os.path.basename(i.uri)]
        assert len(aliased) == 4 and all(os.path.join(w["rs16"], CANON) in u for u in aliased), aliased
        assert all(EM.group_of_alias(u.rsplit("#", 1)[1]) == g for u in aliased)
    assert not any("#" in os.path.basename(i.uri) for i in groups[GC])
    assert any(i.uri.endswith("store/rows.parquet") for i in groups[GC])
    assert ad.status_sources(GBL) == (EM.M16K_SOURCE,)


# ============================================================================================
# the canon root: the step anchor, the same-root contrasts, the instrument, the decode provenance
# ============================================================================================

def test_the_canon_root_files_the_step_over_the_redrawn_items_with_the_producers_paired_contrasts(tmp_path):
    w = _world(tmp_path)
    ad = _adapter(w)
    recs, facts, rep = _read(ad, GC)
    assert rep.runs == 6 and rep.contrasts == 4 and rep.contrasts_parked == 0 and rep.refusals == [] and rep.dispositions == 0
    assert facts["refused"] == [] and facts["not_drawn"] == [] and facts["kind"] == "canon" and facts["drawn_rule"] == "meta.json run.only_cells"
    assert facts["result_count_matches_record"] is True and facts["n_judged_total"] == 12
    none_l = _run(recs, "none", "agentic_leaking")
    assert (none_l["n_items_judged"], none_l["token_cap_tokens"], none_l["token_cap_rung_index"], none_l["token_cap_ladder_json"]) == (3, 16384, 1, [3000, 16384])
    assert none_l["lane_source_legacy"] == EM.M16K_SOURCE and none_l["view_legacy"] == CANON and none_l["campaign"] == "sdfbase_full22"
    assert none_l["producer_run_key_legacy"].startswith(f"{EM.M16K_SOURCE}/{CANON}/") and none_l["object_id"] == "none"
    assert none_l["harness_version"] == TREE and none_l["code_sha"] == TREE and none_l["context_window_tokens"] == 32768
    assert (none_l["substrate"], none_l["substrate_source"]) == ("cluster_a", "recorded") and none_l["cluster_job_id"] == "592837"
    metric = next(m for m in recs["run_metric"] if m["run_id"] == none_l["run_id"])
    assert (metric["k"], metric["n"], round(metric["value"], 6)) == (2, 3, round(200.0 / 3, 6))
    inst = next(i for i in recs["instrument"] if i["instrument_id"] == none_l["instrument_id"])
    assert (inst["token_cap_tokens"], inst["token_cap_ladder_json"], inst["read_at"], inst["harness_config_hash"]) == (16384, [3000, 16384], "fixed", EM.M16K_INSTRUMENT_LABEL)
    assert inst["ladder_promotion_rule"] == EM.M16K_STEP_RULE and inst["decode_source"] == "runner_forced_T1" and inst["decode_temperature"] == 1.0
    assert inst["decode_top_p"] is None and inst["decode_seed_rule"] == "none" and inst["thinking_mode"] == "off" and inst["context_window_tokens"] == 32768
    assert inst["judge_id"] == DC.judge_fields(**DC.AGENTIC_JUDGE)["judge_id"]
    assert facts["canonical_seed_instrument_ids"]["agentic_leaking"] not in {i["instrument_id"] for i in recs["instrument"]}
    # the decode provenance: the wire body recorded, the engine fill derived onto decode_resolved_json, the policy match derived
    fc = none_l["field_confidence_json"]
    assert fc["decode_sent_json"]["confidence"] == "recorded" and "meta.json" in fc["decode_sent_json"]["source"] and "max_tokens 16384" in fc["decode_sent_json"]["source"]
    assert fc["decode_source"]["confidence"] == "derived" and "runner_forced_T1" in fc["decode_source"]["source"]
    assert fc["decode_resolved_json"]["confidence"] == "derived" and "contract_override" in fc["decode_resolved_json"]["source"]
    assert none_l["decode_sent_json"] == {"temperature": 1.0} and none_l["decode_resolved_json"] == FILL and none_l["decode_source"] == "runner_forced_T1"
    ex = none_l["extra"]
    assert ex["decode_record"]["recorded"] is True and ex["decode_record"]["max_tokens_sent"] == 16384 and ex["decode_record"]["sampler"] == "t1"
    # the redraw filter, to the row: the three counts and the rule by its threshold (never item_filtered_v1 nor zlib_collapse)
    rs = ex["redraw_selection"]
    assert rs["rule"] == "misalign runner zlib_ratio < 0.2 excluded as loops" and rs["counts"] == "recorded"
    assert (rs["n_at_cap_source"], rs["n_loops_excluded"], rs["n_redrawn"]) == (6, 1, 5) and "item_filtered_v1" not in rs["rule"] and "zlib_collapse" not in rs["rule"]
    assert ex["token_cap_rung_index"] == 1 and ex["source_cap_tokens"] == 3000 and ex["in_root_none_measured"] is True and ex["step_anchor"]["kind_legacy"] == "same_standup_none"
    assert ex["rows_parquet_run_ids_legacy"] == sorted(f"misalign_{CANON}.{h}.0e63221d" for h in CANON_KEYS) and "COLLIDES" in ex["rows_parquet_run_id_note"]
    assert ex["superseded_lens_generation_check"]["is_lens_generation"] is False and ex["coexists_with"]["lane_source_legacy"] == f"em/agentic/{MODEL}"
    assert ex["canonical_seed_instrument"]["adopted"] is False
    assert ex["final_at_ceiling_share"] == 0.0 and ex["unresolved_truncation"] == dict(ex["unresolved_truncation"], n_at_ceiling=0, n_redrawn=3, share=0.0, ceiling_tokens=16384)
    # the same-root contrasts are the producer's paired.csv statistics against the canon none run and its judgment
    const_l = _run(recs, CONST_R, "agentic_leaking")
    c = next(x for x in recs["run_contrast"] if x["run_id"] == const_l["run_id"])
    assert c["anchor_run_id"] == none_l["run_id"] and c["anchor_kind"] == "same_standup_none" and c["n_pairs"] == 2
    assert (c["delta"], c["delta_lo"], c["delta_hi"], c["delta_p"]) == (-50.0, -100.0, 0.0, 1.0) and c["extra"]["paired"] is True
    assert c["anchor_judgment_id"] == next(j["judgment_id"] for j in recs["judgment"] if j["run_id"] == none_l["run_id"])
    assert c["extra"]["anchor_root"] == c["extra"]["own_root"] == os.path.join(w["rs16"], CANON)
    # the prompted objects are the declared texts; item rows are judged x 3 metrics; the not-drawn slots are simply absent
    assert const_l["object_id"] == str(DC.load_seeds().si_object_id("si_constitution")) and const_l["n_items_judged"] == 2
    assert len(recs["item_result"]) == 3 * 12 and len(recs["item_flags"]) == 12
    # the grade root's misalign_16k tree was never read (it would have refused: its wave_sha is no tree sha)
    assert all(r["extra"]["judged_store"].startswith(w["rs16"]) for r in recs["run"])


def test_a_result_count_above_the_record_is_a_fact_and_below_it_a_refusal_and_the_judge_counts_of_record_are_the_rows_and_the_cache(tmp_path):
    w = _world(tmp_path, canon_kw=dict(result_excess=2521))
    recs, facts, rep = _read(_adapter(w), GC)
    assert rep.runs == 6 and facts["result_count_matches_record"] is False and facts["result_agentic_responses"] == 12 + 2521
    assert "counted the agentic tree as it stood" in facts["result_count_note"]
    assert facts["judge_counts_of_record"] == dict(facts["judge_counts_of_record"], judged=12, blocked=0, judge_cache_lines=12, judge_cache_blocked=0, judge_provenance_describes_record=True)
    w2 = _world(tmp_path / "b", canon_kw=dict(result_excess=-1))
    with pytest.raises(AdapterRefusal, match="the record holds more than the standup ever counted"):
        _read(_adapter(w2), GC)
    # judge_provenance.json is overwritten per judge pass (a one-item retry reads counts.judged 1): a fact beside the record's counts, never a gate
    w3 = _world(tmp_path / "c", canon_kw=dict(judge_counts_judged=1))
    recs3, facts3, rep3 = _read(_adapter(w3), GC)
    assert rep3.runs == 6 and facts3["judge_counts_of_record"]["judge_provenance_describes_record"] is False and facts3["judge_counts_of_record"]["judged"] == 12
    assert facts3["judge_counts_of_record"]["judge_provenance_counts"] == {"judged": 1, "blocked": 0, "errors": 0}
    assert "overwritten per judge pass" in recs3["run"][0]["extra"]["judge_provenance"]["counts_are_the_last_pass"]
    # a cache with fewer entries than the judged rows is a verdict the cache of record lacks
    w4 = _world(tmp_path / "d", canon_kw=dict(cache_short=1))
    with pytest.raises(AdapterRefusal, match="judge_cache.jsonl holds 11 entries, fewer than the 12 judged rows"):
        _read(_adapter(w4), GC)


def test_the_unresolved_truncation_residual_rides_every_row_from_the_at_cap_rows(tmp_path):
    """Re-drawn items at the 16,384 ceiling AGAIN (ReLU 23:01Z: 40-63 % of the prompted keys' murder items on the seed roots) are long-thinkers the
    runner's loop filter passed; the share is filed beside the at_token_cap degeneration measure so a merged number carries it."""
    w = _world(tmp_path, canon_kw=dict(at_cap={CONST_H: (("leaking", 0), ("murder", 0))}))
    recs, facts, rep = _read(_adapter(w), GC)
    r = _run(recs, CONST_R, "agentic_murder")
    assert r["extra"]["final_at_ceiling_share"] == 1.0 and r["extra"]["unresolved_truncation"]["n_at_ceiling"] == 1 and "long-thinkers" in r["extra"]["unresolved_truncation"]["note"]
    deg = [d for d in recs["run_degeneration"] if d["run_id"] == r["run_id"] and d["measure"] == "at_token_cap"]
    assert len(deg) == 1 and (deg[0]["numerator"], deg[0]["denominator"], deg[0]["measurement_state"]) == (1, 1, "measured")
    assert facts["readings"][f"{CONST_H}/leaking"]["final_at_ceiling_share"] == 0.5
    fl = [f for f in recs["item_flags"] if f["run_id"] == r["run_id"]]
    assert len(fl) == 1 and fl[0]["at_token_cap"] == 1 and fl[0]["response_tokens"] == 16384


def test_a_seed_root_reads_its_five_keys_from_the_redraw_selection_and_anchors_cross_root(tmp_path):
    """The seed roots (s1 / s3, ReLU 23:01Z): the seed selection's keys drawn per `run.rerun_selection` (no `only_cells`), the prompted keys on the
    passthrough route (the canon none's own instrument), the fitted keys through the roster's seed selection, all anchored on the canon 16,384 none."""
    w = _world(tmp_path)
    s1 = f"{MODEL}_s1"
    keys = {SAFETY_H: dict(roster=SAFETY_R, ident=PASS, kind="prompted", si="safety", si_tt="safety_refusal", drawn={"leaking": [0, 1, 2], "murder": [0, 1, 2, 3]},
                           harm={"leaking": [1, 0, 0], "murder": [0, 0, 1, 0]}, sel=(3, 0)),
            LORA_H: dict(roster=LORA_R, staged=STAGED[LORA_R], ident=LORA, kind="lora_all", drawn={"leaking": [0], "murder": [0, 1]}, harm={"leaking": [0], "murder": [0, 0]}, sel=(4, 1))}
    _root(w["rs16"], s1, kind="seed", job="592540", keys=keys, measure_none=False, pairing_anchor=f"exp/misalign/{MODEL}/agentic/none",
          at_cap={SAFETY_H: (("murder", 0), ("murder", 1))})
    ad = _adapter(w)
    groups = _groups(ad)
    g = f"{EM.M16K_GROUP_PREFIX}{s1}"
    assert g in groups and len([i for i in groups[g] if "#" in os.path.basename(i.uri)]) == 4
    canon, _f, _r = _read(ad, GC, groups)
    recs, facts, rep = _read(ad, g, groups)
    assert rep.runs == 4 and rep.contrasts == 4 and rep.dispositions == 0 and facts["kind"] == "seed" and facts["drawn_rule"] == "meta.json run.rerun_selection (selected > 0)"
    saf_m = _run(recs, SAFETY_R, "agentic_murder")
    assert saf_m["instrument_id"] == _run(canon, "none", "agentic_murder")["instrument_id"]       # the same instrument (passthrough, T1, this tree); another producer key
    assert saf_m["run_id"] != _run(canon, "none", "agentic_murder")["run_id"] and saf_m["extra"]["final_at_ceiling_share"] == 0.5
    c = next(x for x in recs["run_contrast"] if x["run_id"] == saf_m["run_id"])
    assert c["anchor_kind"] == "sibling_none" and c["anchor_run_id"] == _run(canon, "none", "agentic_murder")["run_id"] and c["extra"]["instrument_match"]["fields_differing"] == []
    lora = _run(recs, LORA_R, "agentic_leaking")
    assert lora["object_id"] == SHA[LORA_R] and lora["producer_object_key_legacy"] == STAGED[LORA_R] and lora["extra"]["selection"] == "s1" and lora["extra"]["paired_of"] is None
    assert lora["extra"]["rows_parquet_run_ids_legacy"] == sorted(f"misalign_{s1}.{h}.0e63221d" for h in keys)


def test_a_canon_root_with_an_unjudged_none_or_a_measure_none_false_policy_refuses_by_name(tmp_path):
    w = _world(tmp_path, canon_kw=dict(judged_none=False))
    with pytest.raises(AdapterRefusal, match="the canon root's none is not judged"):
        _read(_adapter(w), GC)
    w2 = _world(tmp_path / "b", canon_kw=dict(measure_none=False))
    with pytest.raises(AdapterRefusal, match="measure_none is False, not true"):
        _read(_adapter(w2), GC)


# ============================================================================================
# the paired root: cross-root sibling_none contrasts, the objects from the roster, the not-drawn disposition
# ============================================================================================

def test_the_paired_root_anchors_cross_root_on_the_canon_none_and_files_a_not_applicable_for_an_undrawn_behaviour(tmp_path):
    w = _world(tmp_path)
    ad = _adapter(w)
    canon, _cf, _r = _read(ad, GC)
    recs, facts, rep = _read(ad, GP1)
    assert rep.runs == 5 and rep.contrasts == 5 and rep.contrasts_parked == 0 and rep.dispositions == 1 and rep.refusals == []
    assert facts["not_drawn"] == [f"{P1}/{LDIFF_H}/leaking"] and facts["contrasts_from_blocks"] == 5 and facts["drawn_rule"] == "meta.json run.rerun_selection (selected > 0)"
    d = recs["disposition"][0]
    assert d["state"] == "not_applicable" and d["reason"].startswith("not applicable: no item of this behaviour was re-drawn at 16,384") and d["eval_id"] == "agentic_leaking"
    assert "1 of 6 at-cap rows" in d["reason"] and "5 excluded as loops" in d["reason"]
    none_l = _run(canon, "none", "agentic_leaking")
    lora_l = _run(recs, LORA_R, "agentic_leaking")
    c = next(x for x in recs["run_contrast"] if x["run_id"] == lora_l["run_id"])
    assert c["anchor_kind"] == "sibling_none" and c["anchor_run_id"] == none_l["run_id"]
    assert c["anchor_judgment_id"] == next(j["judgment_id"] for j in canon["judgment"] if j["run_id"] == none_l["run_id"])
    d_, lo, hi, p = DC.two_prop(0.0, 2, 2 / 3, 3)
    assert c["delta"] == d_ and c["extra"]["paired"] is False and c["extra"]["cross_root"] is True and c["n_pairs"] == 2
    assert c["extra"]["anchor_root"] == os.path.join(w["rs16"], CANON) and c["extra"]["own_root"] == os.path.join(w["rs16"], P1)
    assert c["extra"]["instrument_match"]["decode_and_cap"] == "match" and c["extra"]["instrument_match"]["fields_differing"] == ["serving_path"]
    assert c["extra"]["lane_policy_pairing_anchor"] == f"exp/misalign/{MODEL}/agentic/none"
    ex = lora_l["extra"]
    assert ex["external_anchor"]["filed_as"] == "run_contrast" and ex["external_anchor"]["anchor_run_id"] == none_l["run_id"] and ex["anchor_is_common_term"] is True
    assert ex["step_anchor"]["kind_legacy"] == "sibling_none" and ex["step_anchor"]["cross_root"] is True and ex["in_root_none_measured"] is False
    assert ex["paired_of"] == "s1" and ex["object_also_in_seed_selection"] is True and ex["small_redraw_note"]
    # the objects: the roster's staged ids, the pinned sha == the store's served sha == the runner's artifact sha
    assert lora_l["object_id"] == SHA[LORA_R] and lora_l["producer_object_key_legacy"] == STAGED[LORA_R] and lora_l["strength_coefficient"] == 1.0
    comp = _run(recs, COMPOSED_R, "agentic_murder")
    assert comp["object_id"] == SHA[COMPOSED_R] and comp["extra"]["object_also_in_seed_selection"] is False
    assert lora_l["decode_resolved_json"] == FILL and lora_l["field_confidence_json"]["decode_sent_json"]["confidence"] == "recorded"


def test_a_paired_root_that_measures_an_in_root_none_or_carries_paired_rows_refuses_by_name(tmp_path):
    w = _world(tmp_path, p1_kw=dict(measure_none=True))
    with pytest.raises(AdapterRefusal, match="measure_none is True, not false"):
        _read(_adapter(w), GP1)
    w2 = _world(tmp_path / "b", p1_kw=dict(paired_rows={(LORA_H, "leaking"): dict(n_pairs=1, mean_arm=0.0, mean_none=0.0, diff=0.0, boot_low=0.0, boot_high=0.0, arm_only=0, none_only=0, mcnemar_p=1.0)}))
    with pytest.raises(AdapterRefusal, match="paired.csv carries 2 row"):
        _read(_adapter(w2), GP1)


def test_without_a_canon_root_the_paired_rows_park_the_anchor_on_the_metric(tmp_path):
    w = _world(tmp_path, with_canon=False)
    recs, facts, rep = _read(_adapter(w), GP1)
    assert rep.runs == 5 and rep.contrasts == 0 and rep.contrasts_parked == 5 and facts["anchor_runs"] is None and "no such target directory" in facts["anchor_runs_why"]
    m = next(x for x in recs["run_metric"] if x["run_id"] == _run(recs, LORA_R, "agentic_leaking")["run_id"])
    blk = m["extra"]["delta_block"]
    assert blk["anchor"]["kind_legacy"] == "sibling_none" and blk["anchor"]["state"] == "absent" and blk["delta"] is None
    assert _run(recs, LORA_R, "agentic_leaking")["extra"]["external_anchor"]["anchor_run_id"] is None


# ============================================================================================
# the baseline root: arditi through the pins, cross-root contrasts
# ============================================================================================

def test_the_baseline_root_resolves_arditi_through_the_pins_and_anchors_on_the_canon_none(tmp_path):
    w = _world(tmp_path)
    ad = _adapter(w)
    recs, facts, rep = _read(ad, GBL)
    assert rep.runs == 2 and rep.contrasts == 2 and rep.refusals == [] and facts["drawn_rule"] == "meta.json run.only_cells"
    r = _run(recs, "arditi", "agentic_leaking")
    assert r["object_id"] == SHA["arditi"] and r["producer_object_key_legacy"] == "ar_r0p05__arditi" and r["operating_point_id_legacy"] == "ar_r0p05__arditi"
    assert (r["strength_coefficient"], r["strength_unit"], r["is_operating_point"]) == (0.05, "alpha", 1)
    assert r["extra"]["baseline_pin"]["alpha"] == 0.05 and r["extra"]["baseline_pin_source"] == "store leg baseline_pin"
    c = next(x for x in recs["run_contrast"] if x["run_id"] == r["run_id"])
    canon, _f, _r = _read(ad, GC)
    assert c["anchor_kind"] == "sibling_none" and c["anchor_run_id"] == _run(canon, "none", "agentic_leaking")["run_id"] and c["n_pairs"] == 4
    assert r["extra"]["redraw_selection"] == dict(r["extra"]["redraw_selection"], n_at_cap_source=7, n_loops_excluded=1, n_redrawn=6)
    # a pins-file sha that disagrees with the served / attested / leg-pin sha refuses the key by name (never a guess)
    w2 = _world(tmp_path / "b", bl_pin_sha="f" * 64)
    recs2, facts2, rep2 = _read(_adapter(w2), GBL)
    assert rep2.runs == 0 and rep2.dispositions == 2 and any("names 2 object shas for arditi" in x for x in facts2["refused"])


# ============================================================================================
# HAZARD 1: the lens-sha refusal
# ============================================================================================

def test_a_store_whose_responses_hash_to_a_lens_generation_refuses_by_name(tmp_path, monkeypatch):
    w = _world(tmp_path)
    lens_sha = _sha(os.path.join(w["rs16"], CANON, "agentic", SAFETY_H, "responses.jsonl"))
    monkeypatch.setattr(EM, "M16K_SUPERSEDED_LENS_RESPONSES_SHA256", dict(EM.M16K_SUPERSEDED_LENS_RESPONSES_SHA256, **{lens_sha: SAFETY_H}))
    recs, facts, rep = _read(_adapter(w), GC)
    assert rep.runs == 4 and rep.dispositions == 2
    assert all("superseded lens generation" in x and lens_sha[:12] in x for x in facts["refused"]) and len(facts["refused"]) == 2
    assert all(SAFETY_H not in r["extra"]["raw_arm"] for r in recs["run"])
    # the same bytes as the canon root's none refuse the anchor itself, so every root of the model parks
    ad2 = _adapter(w)
    none_sha = _sha(os.path.join(w["rs16"], CANON, "agentic", "none", "responses.jsonl"))
    monkeypatch.setattr(EM, "M16K_SUPERSEDED_LENS_RESPONSES_SHA256", dict(EM.M16K_SUPERSEDED_LENS_RESPONSES_SHA256, **{none_sha: "none"}))
    recs2, facts2, rep2 = _read(ad2, GP1)
    assert rep2.contrasts == 0 and rep2.contrasts_parked == 5 and facts2["anchor"]["state"] == "refused" and "superseded lens generation" in facts2["anchor"]["why"]


# ============================================================================================
# the instrument guard, a vendor key, the RESULT / meta cross-checks
# ============================================================================================

def test_a_vendor_sampler_key_is_refused_and_the_guard_refuses_a_vendor_shaped_or_merged_ladder_instrument(tmp_path):
    w = _world(tmp_path, bl_kw=dict(sampler="vendor", sampling=OVERRIDE))
    recs, facts, rep = _read(_adapter(w), GBL)
    assert rep.runs == 0 and rep.dispositions == 2 and all("not the runner's default t1" in x for x in facts["refused"])
    ad = _adapter(w)
    batch = RecordBatch("x", GBL)
    good = {"eval_id": "agentic_leaking", "decode_source": "runner_forced_T1", "decode_temperature": 1.0, "decode_seed_rule": "none", "token_cap_tokens": 16384,
            "token_cap_ladder_json": [3000, 16384], "instrument_id": "0" * 64, "serving_path": "lora_native", "applied_positions_mode": "all"}
    batch.add("instrument", dict(good, decode_top_p=0.8), PRODUCER, NOW)
    with pytest.raises(AdapterRefusal, match="vendor-shaped decode"):
        ad._m16k_guard_instruments(batch, {"group": GBL, "model": MODEL, "instrument_ids": {}})
    batch2 = RecordBatch("y", GBL)
    batch2.add("instrument", dict(good, token_cap_ladder_json=[16384]), PRODUCER, NOW)
    with pytest.raises(AdapterRefusal, match=r"token_cap_ladder_json \[16384\] is not \[3000, 16384\]"):
        ad._m16k_guard_instruments(batch2, {"group": GBL, "model": MODEL, "instrument_ids": {}})


def test_a_key_sent_at_another_cap_or_with_a_disagreeing_selection_count_is_refused_by_name(tmp_path):
    w = _world(tmp_path, bl_kw=dict(extra_meta={"max_tokens": 3000}))
    recs, facts, rep = _read(_adapter(w), GBL)
    assert rep.runs == 0 and all("is not the step's cap 16384" in x for x in facts["refused"])
    w2 = _world(tmp_path / "b")
    mp = os.path.join(w2["rs16"], BL, "agentic", "arditi", "meta.json")
    m = json.load(open(mp))
    m["run"]["rerun_selection"]["arditi"]["selected"] = 99
    _w(mp, m)
    recs2, facts2, rep2 = _read(_adapter(w2), GBL)
    assert rep2.runs == 0 and all("rerun_selection[arditi].selected 99 != the responses generated 6" in x for x in facts2["refused"])


# ============================================================================================
# decode provenance on the PANEL reader's rows: the declaration's value never moves; its provenance is the key's meta.json
# ============================================================================================

EM_PANEL = [("gemma4e4b", "gemma4e4b", "base→instruct", "gemma4e4b", "base")]


def _panel_world(tmp_path, tag: str, *, meta=None, with_meta=True):
    root, grade = str(tmp_path / f"figs_{tag}"), str(tmp_path / f"grade_{tag}")
    F.build_em_panel(root)
    pins = F.build_em_pins(str(tmp_path / f"pins_{tag}.json"))
    F.build_em_store(grade, "gemma4e4b", legs=[("none", PASS, "aaaa000011112222"), (LORA_H, "proxy_vllm+lens+interleaved", "bbbb000011112222")],
                     bank_objects={LORA_R: F.SHA_A})
    if with_meta:
        for key in ("none", LORA_H):
            _w(os.path.join(grade, "misalign", "gemma4e4b", "agentic", key, "meta.json"),
               meta if meta is not None else {"max_tokens": 3000, "temperature": 1.0, "params": {"concurrency": 36, "max_tokens": 3000, "temperature": 1.0}, "slurm_job_id": "424242"})
    return EM.EmMisalignAdapter(root=root, full22_pins=pins, baseline_pins=None, behaviour_source={}, panels=EM_PANEL, grade_root=grade, seedrep_root=None, m16k_root=None)


def _panel_runs(ad):
    recs, facts, rep = _read(ad, "gemma4e4b")
    return recs.get("run", []), rep


def test_panel_rows_read_their_decode_provenance_from_the_keys_meta_json_and_keep_the_value(tmp_path):
    runs, rep = _panel_runs(_panel_world(tmp_path, "rec"))
    assert rep.runs == 2 and rep.refusals == []
    for r in runs:
        fc = r["field_confidence_json"]
        assert r["decode_sent_json"] == {"temperature": 1.0} and r["decode_source"] == "runner_forced_T1" and r["decode_resolved_json"] == {}
        assert fc["decode_sent_json"]["confidence"] == "recorded" and "the pre-sampler runner wrote temperature alone" in fc["decode_sent_json"]["source"]
        assert fc["decode_source"]["confidence"] == "derived" and fc["decode_resolved_json"]["confidence"] == "unverified"
        assert "no server default policy" in fc["decode_resolved_json"]["reason"]
        assert r["extra"]["decode_record"]["recorded"] is True and r["extra"]["decode_record"]["max_tokens_sent"] == 3000
    # no meta.json beside the store: the declaration stands, marked as the reader's assumed default -- never recorded
    runs2, rep2 = _panel_runs(_panel_world(tmp_path, "nometa", with_meta=False))
    assert rep2.runs == 2
    for r in runs2:
        fc = r["field_confidence_json"]
        assert r["decode_sent_json"] == {"temperature": 1.0} and fc["decode_sent_json"]["confidence"] == "assumed_default" and fc["decode_source"]["confidence"] == "assumed_default"
        assert r["extra"]["decode_record"]["recorded"] is False
    # a meta.json that disagrees with the T1 declaration refuses the key rather than filing a decode the record contradicts
    runs3, rep3 = _panel_runs(_panel_world(tmp_path, "dis", meta={"max_tokens": 3000, "sampler": "vendor", "sampling": OVERRIDE, "temperature": 0.7, "slurm_job_id": "424242"}))
    assert rep3.runs == 0 and rep3.dispositions == 2


def test_the_decode_record_helper_by_shape():
    rec = EM.agentic_decode_record({"sampler": "t1", "sampling": {"temperature": 1.0}, "max_tokens": 16384}, expected_source="runner_forced_T1", declared={"temperature": 1.0},
                                   meta_path="/x/agentic/none/meta.json", server_policy="contract_override", override=OVERRIDE, override_source="cells_full.json:endpoint")
    assert rec["recorded"] and rec["disagreement"] is None and rec["resolved"] == FILL and rec["confidence"]["decode_resolved_json"][0] == "derived"
    rec = EM.agentic_decode_record({"temperature": 1.0, "max_tokens": 3000, "params": {"temperature": 1.0}}, expected_source="runner_forced_T1", declared={"temperature": 1.0},
                                   meta_path="/x/agentic/none/meta.json", server_policy=None, override=None, override_source=None)
    assert rec["sent"] == {"temperature": 1.0} and rec["resolved"] is None and rec["confidence"]["decode_resolved_json"][0] == "unverified"
    rec = EM.agentic_decode_record({"sampler": "t1", "sampling": {"temperature": 1.0, "top_p": 0.9}, "max_tokens": 3000}, expected_source="runner_forced_T1", declared={"temperature": 1.0},
                                   meta_path="/x/agentic/none/meta.json", server_policy=None, override=None, override_source=None)
    assert rec["disagreement"] and "top_p" in rec["disagreement"]
    vend = EM.agentic_decode_record({"sampler": "vendor", "sampling": OVERRIDE | {"presence_penalty": 1.5}, "max_tokens": 16384}, expected_source="vendor_recommendation",
                                    declared=EM.VENDOR_DECODE, meta_path="/x/agentic/none/meta.json", server_policy=None, override=None, override_source=None)
    assert vend["disagreement"] is None and vend["extra_recorded_keys"] == ["min_p", "repetition_penalty"] and vend["confidence"]["decode_sent_json"][0] == "derived"
    none_rec = EM.agentic_decode_record(None, expected_source="runner_forced_T1", declared={"temperature": 1.0}, meta_path=None, server_policy=None, override=None, override_source=None)
    assert none_rec["recorded"] is False and none_rec["confidence"]["decode_sent_json"][0] == "assumed_default"


# ============================================================================================
# the shape
# ============================================================================================

def test_the_shape_equals_em_misaligns_misalignment_rows(tmp_path):
    w = _world(tmp_path)
    ad = _adapter(w)
    rows, _facts = ad.m16k_rows(CANON)
    seed_rows, _f = ad.long_rows("gemma4e4b")
    public = lambda r: {k for k in r if not k.startswith("_")}  # noqa: E731
    assert public(rows[0]) == public(seed_rows[0])
    assert {r["status"] for r in rows} == {"measured"}
    p_rows, _pf = ad.m16k_rows(P1)
    assert {r["status"].split(":")[0] for r in p_rows} == {"measured", "not applicable"}
