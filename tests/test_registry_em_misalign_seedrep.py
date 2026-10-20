"""The three-seed family's agentic-misalignment stores through the ``em_misalign`` legacy-store reader (source ``misalign_seedrep``; ReLU
2026-09-21), on a SYNTHETIC world: one SEED view (``qwen35_4b_s1``: a fitted ReLU object at its ``base64s1__`` staged id and a prompted arm,
with the LoRA legs the policy names RESERVED to the paired view), one PAIRED view (``qwen35_4b_s1p``: the composed paired write at
``paired_eb64s1__`` and the LoRA object re-served from the seed view's staged id -- a standup replicate), the anchor's re-judge store beside them
and a grade-root copy of the paper's plain ``none`` responses it is byte-identical to. Invented shas and trees, a handful of placeholder
responses per behaviour (no generation text), hostname ``TESTHOST``.

What it pins:

* the DECLARATIONS: the provisional source in ``sources_r2_v1.yaml`` and in ``covers_sources``, the version bump, the view grammar, one
  comparison group per view under the source, the anchor store aliased to every view, a poisoned copy skipped BY NAME.
* the IDENTITY: harness cell -> roster cell (``petri_stats.PCELL_REV`` + the composed paired write) -> the roster's staged id (the object key)
  -> the pinned sha, attested against the store's ``bank_objects`` and the runner's per-cell artifact sha (a disagreement refuses the cell);
  a prompted arm's ``si_prefix_hash`` against the declared prompt; producer key ``misalign/<view>/<petri run key>``; campaign
  ``sdfbase_full22``, view ``qwen35_4b_s<N>[p]``, source ``misalign_seedrep``.
* the STANDUP-REPLICATE markers on the paired view (``extra.standup_replicate_of_view`` / ``extra.paired_of``) and their absence on the
  composed object and on the seed view; the legs the seed view did not draw filed as ``reserved_not_run``.
* the COUNTS: ``n`` = judged rows minus blocked, ``k`` = harmful (= classifier AND gate on every scored row), a missing judged row and a blocked
  row filed as facts (``judged_rows_missing`` / ``judge_blocked``, the reading a lower bound), three item rows per response.
* NO IN-ROOT NONE: an in-root judged ``none``, a ``paired.csv`` with rows, or a policy with ``measure_none`` true refuses the view by name.
* the INSTRUMENT: the runner's T1 decode as sent (temperature 1.0, nothing else, seed rule none, ``runner_forced_T1``), cap 3,000 read fixed,
  never a vendor shape -- a vendor-sampler cell is refused, and the post-emission guard refuses a vendor-shaped instrument.
* the ANCHOR (version 5): the stem's plain 3,000-cap ``none`` -- judged in its own store and named as the lane's pairing anchor -- is REGISTERED as
  a run from the stem's panel group (rung 0 of the ladder, its own instrument, ``extra.superseded_for_headline_by_run_id`` the ladder's none, no
  superseded state; the plain arms stay skipped), the re-judge files as a SECOND judgment on it (the same judge id, a session-told judgment id),
  and every seed row files a ``run_contrast`` against that run and that judgment (``sibling_none``, unpaired, ``cap_matched``, the common-term flag)
  with the block's numbers; an UNJUDGED plain none stays skipped and the seed rows park; a plain none whose panel statistic is not its record refuses.
* the SUBSTRATE recorded from the serving block's ``served_cluster``; the SHAPE equals em_misalign's misalignment rows.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil

import pytest
import yaml

from sisteer.registry import validate as VAL
from sisteer.registry.adapters import declared_common as DC
from sisteer.registry.adapters import em_misalign as EM
from sisteer.registry.adapters.base import AdapterContext, AdapterRefusal, RecordBatch
from sisteer.registry import synth

import registry_declared_fixtures as F  # noqa: E402
import _anchor_frame_fixtures as AF  # noqa: E402  (emitter 4: in-memory reads need a primed anchor frame)

NOW = "2026-09-21T18:00:00.000Z"
PRODUCER = {"script": "tests.em_misalign_seedrep", "code_sha": synth.code_sha("a"), "session": None, "host": "TESTHOST", "substrate": "grading_host"}
LORA, PASS = "proxy_vllm+lora+interleaved", "proxy_vllm+passthrough+interleaved"
MODEL = "qwen35_4b"
TREE_S, TREE_P = synth.code_sha("1"), synth.code_sha("2")
PSR_H, PSR_R = "si_constv2_mix_full__psr_ce_all", "constv2-mix-full__psr_ce_all"
LORA_H, LORA_R = "si_constv2_mix_full__lora_r4_ce", "constv2-mix-full__lora_r4_ce"
LDIFF_H, LDIFF_R = "si_constv2_mix_full__lora_r4_ce_minus_ctl", "constv2-mix-full__lora_r4_ce_minus_ctl"
COMPOSED_H, COMPOSED_R = "si_constv2_mix_full__psr_ce_all_minus_ctl", "constv2-mix-full__psr_ce_all_minus_ctl"
SAFETY_H, SAFETY_R = "prompted__safety", "prompted__safety_refusal"
SHA = {PSR_R: hashlib.sha256(b"synthetic psr s1").hexdigest(), LORA_R: hashlib.sha256(b"synthetic lora s1").hexdigest(),
       LDIFF_R: hashlib.sha256(b"synthetic lora diff s1").hexdigest(), COMPOSED_R: hashlib.sha256(b"synthetic composed s1").hexdigest()}
STAGED = {PSR_R: f"base64s1__{PSR_R}", LORA_R: f"base64s1__{LORA_R}", LDIFF_R: f"derived_base64s1__{LDIFF_R}", COMPOSED_R: f"paired_eb64s1__{COMPOSED_R}"}
ITEMS = {"leaking": ["leaking_explicit-alpha_none", "leaking_explicit-beta_none"], "murder": ["murder_explicit-alpha_none", "murder_explicit-beta_none"]}
N_SAMPLES = 2
ANCHOR_HARM = {"leaking": [1, 0, 1, 0], "murder": [0, 0, 1, 0]}     # per (item, sample) in order: k = 2 of 4 and 1 of 4
PLAIN_HARM = {"leaking": [1, 1, 1, 0], "murder": [0, 0, 0, 0]}      # the plain store's OWN (first) judgment of the same responses: k = 3 of 4 and 0 of 4
PLAIN_KEY = "8d364001a7b20bbb"


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


def _responses(cell: str, harm_by_beh, *, blocked=(), missing=(), judge_model="claude-sonnet-5", judge_pool="lowprio", at_cap=()):
    """``(responses rows, judged rows)`` for one cell: ITEMS x N_SAMPLES per behaviour; ``harm_by_beh[b]`` the harmful bit per slot (classifier
    = harmful, gate = 1 on every scored row), ``blocked`` / ``missing`` / ``at_cap`` name (behaviour, slot index)."""
    resp, judged = [], []
    for b, ids in ITEMS.items():
        slot = 0
        for item in ids:
            for s in range(N_SAMPLES):
                cap = (b, slot) in at_cap
                r = {"family": "agentic", "cell_id": cell, "virtual_model": f"petri__{cell}", "item_id": item, "sample_idx": s, "behavior": b,
                     "prompt_sha": hashlib.sha256(f"{item}|{s}".encode()).hexdigest(), "response": "placeholder response text",
                     "finish_reason": ("length" if cap else "stop"), "usage": {"prompt_tokens": 100, "completion_tokens": (3000 if cap else 40), "total_tokens": 140},
                     "model_reported": f"petri__{cell}"}
                resp.append(r)
                if (b, slot) not in missing:
                    blk = (b, slot) in blocked
                    h = None if blk else int(harm_by_beh[b][slot])
                    judged.append(dict(r, judge_tag=EM.SEEDREP_JUDGE_TAGS[b], judge_model=judge_model, judge_pool=judge_pool, judged_at="2026-09-21T11:00:00Z",
                                       structural_gate=(0 if blk else 1), judge_blocked=blk, classifier_harmful=h, harmful=h))
                slot += 1
    return resp, judged


def _summary(view: str, cells: dict, rows_by_cell: dict, sampler="t1") -> list:
    """The producer's summary.csv rows from the judged rows: harmful (n = scored, k = harmful), at_cap (k = finish_reason length), loop /
    degenerate (0), the continuous metrics blank."""
    out = []
    for cell, judged in rows_by_cell.items():
        for b in ITEMS:
            rows = [r for r in judged if r["behavior"] == b]
            scored = [r for r in rows if not r["judge_blocked"]]
            n, k = len(scored), sum(int(r["harmful"]) for r in scored)
            cap_k = sum(1 for r in rows if r["finish_reason"] == "length")
            for metric, kk, est in (("harmful", k, k / n if n else 0.0), ("at_cap", cap_k, cap_k / n if n else 0.0), ("loop", 0, 0.0), ("degenerate", 0, 0.0),
                                    ("harmful_reasoned", k, k / n if n else 0.0), ("harmful_bare", 0, 0.0), ("harmful_fuzzy", k, k / n if n else 0.0)):
                out.append(dict(target=view, arm=cell, family="agentic", behavior=b, metric=metric, n=n, k=kk, estimate=est, ci_low=max(0.0, est - 0.1),
                                ci_high=min(1.0, est + 0.1), ci_method="wilson95", boot_low=None, boot_high=None, note=None, engine=cells[cell]["ident"],
                                standup_engine=None, sampler=sampler, sampling=json.dumps({"temperature": 1.0})))
            out.append(dict(target=view, arm=cell, family="agentic", behavior=b, metric="response_chars", n=n, k=None, estimate=1234.0, ci_low=None, ci_high=None,
                            ci_method=None, boot_low=None, boot_high=None, note=None, engine=cells[cell]["ident"], standup_engine=None, sampler=sampler,
                            sampling=json.dumps({"temperature": 1.0})))
    return out


def _view(root: str, view: str, *, tree: str, job: str, cells: dict, measured_legs, only_cells, harm, blocked=None, missing=None, at_cap=None,
          measure_none=False, pairing_anchor=f"exp/misalign/{MODEL}/agentic/none", judged_none=False, paired_rows=False, sampler="t1",
          sampling=None, bank_sha_override=None, judge_model="claude-sonnet-5", meta_sha_override=None) -> str:
    """One view directory in the producer's shape. ``cells``: harness id -> {roster, staged, ident, kind, si}."""
    import pandas as pd
    vdir = os.path.join(root, view)
    sel = view.split("_")[-1]
    sampling = sampling if sampling is not None else {"temperature": 1.0}
    bank = {}
    legs = [{"cell_class": "base", "cell_id": "none", "control_of": None, "harness_config_hash": "d68387c7669966f7", "intervention_hash": "2ff9cdff091aef6f", "kind": "none",
             "object_path": None, "petri_run_key": "0000aaaa1111bbbb", "serving_identity": PASS, "si_id": None, "si_prefix_hash": None, "virtual_model_name": "petri__base",
             "si_id_for_runner": None, "measured": False}]
    harness_cells, roster_cells = ["none"], ["none"]
    for i, (h, c) in enumerate(cells.items()):
        harness_cells.append(h)
        roster_cells.append(c["roster"])
        if c.get("staged"):
            sha = (bank_sha_override or {}).get(h, SHA[c["roster"]])
            bank[c["roster"]] = {"identity": "staged", "object_sha256": sha, "positions_at_fit": "all", "sidecar_object_sha256": sha,
                                 "staged_from": {"cell_id": c["roster"], "effective_batch": 64, "key": f"{MODEL}_base_eb64_s1", "kind": c["kind"], "lane": c["staged"].split("__")[0],
                                                 "object_sha256": sha, "pulled_at": "2026-09-21T06:30:00Z", "source": "synthetic", "staged_by": "synthetic"}}
        legs.append({"cell_class": ("prompted" if c.get("si") else "class_a"), "cell_id": h, "control_of": None, "harness_config_hash": "b7f3c3ea6599e45c",
                     "intervention_hash": f"{i + 1:016x}", "kind": c["kind"], "object_path": None, "petri_run_key": hashlib.sha256(f"{view}|{h}".encode()).hexdigest()[:16],
                     "serving_identity": c["ident"], "si_id": c.get("si"), "si_prefix_hash": (_si_prefix(c["si_tt"]) if c.get("si") else None), "virtual_model_name": f"petri__{h}",
                     "si_id_for_runner": c.get("si"), "measured": True})
    policy = {"measure_none": measure_none, "measured_cells_declared": [cells[h]["roster"] for h in measured_legs], "lane": ("paired" if sel.endswith("p") else "reseed"),
              "pairing_anchor": pairing_anchor, "source": f"misalign_full22_{view}.yaml misalign_policy", "served_legs": ["none"] + list(measured_legs) + ["none__lora0"],
              "measured_legs": list(measured_legs), "unmeasured_served": ["none", "none__lora0"]}
    override = json.dumps({"min_p": 0.0, "repetition_penalty": 1.0, "temperature": 0.7, "top_k": 20, "top_p": 0.8})
    ep = {"model": "Qwen/Qwen3.5-4B", "model_revision": F.REVISION, "model_key": MODEL, "proxy_url": "http://127.0.0.1:1/v1", "max_model_len": 16384, "max_num_seqs": 31,
          "attested_max_model_len": 16384, "attested_max_num_seqs": 31, "thinking_mode": "off", "node": "compute-7", "slurm_job_id": job, "cells": roster_cells,
          "harness_cells": harness_cells, "server_default_policy": "contract_override", "override_generation_config": override, "bank_objects": bank}
    _w(os.path.join(vdir, "cells_full.json"), {"target": view, "model_key": MODEL, "wave_view": sel, "wave_sha": tree, "written_at": "2026-09-21T08:01:39Z",
                                              "measured_policy": policy, "endpoint": ep,
                                              "cells_served": {"manifest": f"/shared/tester/repo/{tree}/configs/capeval/manifests/misalign_full22_{view}.yaml"},
                                              "legs": legs})
    _w(os.path.join(vdir, "agentic", "cells.json"), {"cells": [{"cell_id": h, "virtual_model": f"petri__{h}", "serving_identity": c["ident"], "intervention_hash": "x",
                                                                 "si_id": c.get("si"), "kind": c["kind"]} for h, c in cells.items()],
                                                       "serving": {"substrate": "slurm", "slurm_job_id": job, "baseline_pins": None, "served_cluster": "cluster_a", "shard": None,
                                                                   "attested_max_num_seqs": 31, "attested_max_model_len": 16384, "max_num_seqs": 31, "max_model_len": 16384,
                                                                   "node": "compute-7", "measured_policy": policy, "server_default_policy": "contract_override",
                                                                   "override_generation_config": override}})
    rows_by_cell = {}
    n_total = 0
    art = {h: (meta_sha_override or {}).get(h, SHA.get(c["roster"])) for h, c in cells.items() if c.get("staged")}
    for h, c in cells.items():
        if h not in only_cells:
            continue
        resp, judged = _responses(h, harm[h], blocked=(blocked or {}).get(h, ()), missing=(missing or {}).get(h, ()), at_cap=(at_cap or {}).get(h, ()), judge_model=judge_model)
        rows_by_cell[h] = judged
        n_total += len(resp)
        cd = os.path.join(vdir, "agentic", h)
        _w(os.path.join(cd, "responses.jsonl"), resp, jsonl=True)
        _w(os.path.join(cd, "judged.jsonl"), judged, jsonl=True)
        run_blk = {"serving": {"substrate": "slurm", "served_cluster": "cluster_a", "slurm_job_id": job, "measured_policy": policy}}
        if sorted(only_cells) != sorted(measured_legs):
            run_blk["only_cells"] = sorted(only_cells)
        _w(os.path.join(cd, "meta.json"), {"behaviors": list(ITEMS), "cell_id": h, "family": "agentic", "hostname": "compute-7", "items": sum(len(v) for v in ITEMS.values()),
                                           "item_ids": [i for v in ITEMS.values() for i in v], "max_tokens": 3000, "n_samples": N_SAMPLES, "sampler": sampler, "sampling": sampling,
                                           "temperature": 1.0, "slurm_job_id": job, "suite_revision": "agentic.v2@synthetic", "run": run_blk,
                                           "endpoint_identity": {"base_model": "Qwen/Qwen3.5-4B", "engine": {"kind": "lora_native"}, "server_default_policy": "contract_override",
                                                                 "thinking_mode": "off", "object_artifact_sha256_by_cell": art}})
    if judged_none:
        resp, judged = _responses("none", {b: [0] * (len(ITEMS[b]) * N_SAMPLES) for b in ITEMS})
        _w(os.path.join(vdir, "agentic", "none", "judged.jsonl"), judged, jsonl=True)
    _w(os.path.join(vdir, "judge_provenance.json"), {"results_root": vdir, "target": view, "families": ["agentic"], "judge_model": judge_model,
                                                     "tags": {"em8q": "em8q.v1", **EM.SEEDREP_JUDGE_TAGS}, "thinking": False, "suite_revision_v2": "agentic.v2@synthetic",
                                                     "concurrency": 20, "pool_initial": "lowprio", "pool_final": "lowprio",
                                                     "escalation": {"enabled": True, "escalated": False}, "pool_events": [], "n_429": 0, "n_529": 0,
                                                     "n_ok": sum(len(v) for v in rows_by_cell.values()),
                                                     "counts": {"judged": sum(len(v) for v in rows_by_cell.values()), "blocked": sum(1 for v in rows_by_cell.values() for r in v if r["judge_blocked"]), "errors": 0},
                                                     "errors": [], "started": "2026-09-21T10:54:28Z", "finished": "2026-09-21T11:20:54Z"})
    with open(os.path.join(vdir, "RESULT"), "w", encoding="utf-8") as fh:
        fh.write(f"2026-09-21T09:29:25+00:00 job={job} target={view} model_key={MODEL} view={sel} sha={tree[:12]} cluster=cluster_a rc=0 agentic_rc=0 em8q_rc=0 "
                 f"agentic_responses={n_total} em8q_responses=0\n")
    adir = os.path.join(vdir, "analysis", view)
    os.makedirs(adir, exist_ok=True)
    pd.DataFrame(_summary(view, cells, rows_by_cell, sampler=sampler)).to_csv(os.path.join(adir, "summary.csv"), index=False)
    cols = ["target", "arm", "ref", "family", "behavior", "metric", "n_pairs", "mean_arm", "mean_none", "diff", "boot_low", "boot_high", "arm_only", "none_only", "mcnemar_p", "note"]
    pr = pd.DataFrame(columns=cols)
    if paired_rows:
        pr = pd.DataFrame([dict(zip(cols, [view, PSR_H, "none", "agentic", "leaking", "harmful", 4, 0.5, 0.25, 0.25, 0.0, 0.5, 1, 0, 0.5, None]))])
    pr.to_csv(os.path.join(adir, "paired.csv"), index=False)
    return vdir


def _anchor(root: str, grade: str) -> str:
    adir = os.path.join(root, EM.SEEDREP_ANCHOR_DIR.format(model=MODEL))
    resp, judged = _responses("none", ANCHOR_HARM)
    cd = os.path.join(adir, "agentic", "none")
    _w(os.path.join(cd, "responses.jsonl"), resp, jsonl=True)
    _w(os.path.join(cd, "judged.jsonl"), judged, jsonl=True)
    _w(os.path.join(cd, "meta.json"), {"max_tokens": 3000, "temperature": 1.0, "slurm_job_id": "90552"})
    _w(os.path.join(adir, "judge_provenance.json"), {"target": f"{MODEL}_anchor", "families": ["agentic"], "judge_model": "claude-sonnet-5", "tags": EM.SEEDREP_JUDGE_TAGS,
                                                     "thinking": False, "suite_revision_v2": "agentic.v2@synthetic", "concurrency": 20, "pool_initial": "lowprio",
                                                     "pool_final": "lowprio", "n_429": 0, "n_529": 0, "counts": {"judged": len(judged), "blocked": 0, "errors": 0},
                                                     "started": "2026-09-21T16:08:52Z", "finished": "2026-09-21T16:13:21Z"})
    _w(os.path.join(adir, "cells_full.json"), {"target": MODEL, "wave_sha": synth.code_sha("8"), "endpoint": {"slurm_job_id": "90552", "node": "compute-0"}, "legs": []})
    # the paper's PLAIN none store under the grade root: byte-identical responses (the same generation); the ladder's differ
    plain = os.path.join(grade, "misalign", MODEL, "agentic", "none", "responses.jsonl")
    os.makedirs(os.path.dirname(plain), exist_ok=True)
    shutil.copyfile(os.path.join(cd, "responses.jsonl"), plain)
    ladder = os.path.join(grade, "misalign_ladder", MODEL, "agentic", "none", "responses.jsonl")
    _w(ladder, resp[:-1], jsonl=True)
    return adir


def _paper_world(figs: str, grade: str, *, plain_judged: bool = True, plain_none_kn=None) -> None:
    """The stem's plain + ladder PANELS and judged stores as em_misalign reads them: the plain store's own (first) judgment of its none, its
    summary.csv statistic for none (n scored, k harmful -- PLAIN_HARM unless overridden), the ladder tree beside it as the live panel."""
    import pandas as pd
    _, judged_plain = _responses("none", PLAIN_HARM, judge_pool="highprio")
    pcell = os.path.join(grade, "misalign", MODEL, "agentic", "none")
    if plain_judged:
        _w(os.path.join(pcell, "judged.jsonl"), judged_plain, jsonl=True)
    _w(os.path.join(pcell, "meta.json"), {"max_tokens": 3000, "temperature": 1.0, "params": {"max_tokens": 3000, "temperature": 1.0}, "slurm_job_id": "90552"})
    for sub in ("misalign", "misalign_ladder"):
        F.build_em_store(grade, MODEL, legs=[("none", PASS, PLAIN_KEY)], bank_objects={}, subdir=sub, wave_sha=synth.code_sha("8"))
    for tree, kn in ((MODEL, plain_none_kn or {b: (4, sum(PLAIN_HARM[b])) for b in ITEMS}), (f"{MODEL}_ladder", {b: (4, 1) for b in ITEMS})):
        rows = []
        for b, (n, k) in kn.items():
            for metric, kk, est in (("harmful", k, k / n), ("harmful_reasoned", k, k / n), ("harmful_bare", 0, 0.0), ("degenerate", 0, 0.0), ("at_cap", 0, 0.0), ("loop", 0, 0.0)):
                rows.append(dict(target=tree, arm="none", family="agentic", behavior=b, metric=metric, n=n, k=kk, estimate=est, ci_low=max(0.0, est - 0.1), ci_high=min(1.0, est + 0.1),
                                 ci_method="wilson95", boot_low=None, boot_high=None, note=None))
        rows.append(dict(target=tree, arm="none", family="em8q", behavior="betley", metric="coherence", n=4, k=None, estimate=90.0, ci_low=88.0, ci_high=92.0, ci_method="bootstrap95",
                         boot_low=88.0, boot_high=92.0, note=None))
        d = os.path.join(figs, tree)
        os.makedirs(d, exist_ok=True)
        pd.DataFrame(rows).to_csv(os.path.join(d, "summary.csv"), index=False)
        pd.DataFrame(columns=["target", "arm", "ref", "family", "behavior", "metric", "n_pairs", "mean_arm", "mean_none", "diff", "boot_low", "boot_high", "arm_only", "none_only",
                              "mcnemar_p", "note"]).to_csv(os.path.join(d, "paired.csv"), index=False)


SEED_CELLS = {PSR_H: dict(roster=PSR_R, staged=STAGED[PSR_R], ident=LORA, kind="psr"),
              SAFETY_H: dict(roster=SAFETY_R, staged=None, ident=PASS, kind="prompted", si="safety", si_tt="safety_refusal"),
              LORA_H: dict(roster=LORA_R, staged=STAGED[LORA_R], ident=LORA, kind="lora_all"),
              LDIFF_H: dict(roster=LDIFF_R, staged=STAGED[LDIFF_R], ident=LORA, kind="lora_subtraction")}
PAIRED_CELLS = {COMPOSED_H: dict(roster=COMPOSED_R, staged=STAGED[COMPOSED_R], ident=LORA, kind="psr_paired_write"),
                LORA_H: dict(roster=LORA_R, staged=STAGED[LORA_R], ident=LORA, kind="lora_all")}
SEED_HARM = {PSR_H: {"leaking": [0, 0, 1, 0], "murder": [0, 0, 0, 0]}, SAFETY_H: {"leaking": [1, 1, 0, 1], "murder": [0, 1, 0, 0]}}
PAIRED_HARM = {COMPOSED_H: {"leaking": [1, 0, 0, 0], "murder": [0, 0, 1, 1]}, LORA_H: {"leaking": [0, 0, 0, 0], "murder": [0, 1, 0, 0]}}


def _roster(tmp_path) -> dict:
    roster = {"targets": {MODEL: {"views": {"s1": {"cells": {PSR_R: STAGED[PSR_R], LORA_R: STAGED[LORA_R], LDIFF_R: STAGED[LDIFF_R], SAFETY_R: None,
                                                                "prompted__si_constitution": None}},
                                            "s1p": {"cells": {COMPOSED_R: STAGED[COMPOSED_R], LORA_R: STAGED[LORA_R], LDIFF_R: STAGED[LDIFF_R]}}}}}}
    rp = str(tmp_path / "roster.yaml")
    with open(rp, "w", encoding="utf-8") as fh:
        yaml.safe_dump(roster, fh)
    pins = str(tmp_path / "pins.json")
    _w(pins, {MODEL: {STAGED[r]: {"object_sha256": SHA[r]} for r in SHA}})
    return {"roster": rp, "pins": pins}


def _world(tmp_path, *, seed_kw=None, paired_kw=None, with_anchor=True, poisoned=True, paper=True, plain_judged=True, plain_none_kn=None):
    """The panels root (one EM panel so discovery has a panel, plus the stem's plain + ladder panels), the seed-view root with both views (+ the
    anchor, + a poisoned copy), a grade root with the stem's plain + ladder stores."""
    figs, seeds_root, grade = str(tmp_path / "figs"), str(tmp_path / "rs" / "misalign"), str(tmp_path / "grade")
    F.build_em_panel(figs)
    if paper:
        _paper_world(figs, grade, plain_judged=plain_judged, plain_none_kn=plain_none_kn)
    os.makedirs(seeds_root, exist_ok=True)
    _view(seeds_root, f"{MODEL}_s1", tree=TREE_S, job="586157", cells=SEED_CELLS, measured_legs=[PSR_H, LORA_H, LDIFF_H, SAFETY_H], only_cells=[PSR_H, SAFETY_H], harm=SEED_HARM,
          **(seed_kw or {}))
    _view(seeds_root, f"{MODEL}_s1p", tree=TREE_P, job="586027", cells=PAIRED_CELLS, measured_legs=[COMPOSED_H, LORA_H], only_cells=[COMPOSED_H, LORA_H], harm=PAIRED_HARM,
          **(paired_kw or {}))
    if poisoned:
        os.makedirs(os.path.join(seeds_root, f"{MODEL}_s1p", "agentic.POISONED-conc200", COMPOSED_H), exist_ok=True)
        os.makedirs(os.path.join(seeds_root, "qwen35_4b_s9.POISONED"), exist_ok=True)
    if with_anchor:
        _anchor(seeds_root, grade)
    rp = _roster(tmp_path)
    return dict(figs=figs, rs=seeds_root, grade=grade, **rp)


def _adapter(w, **kw) -> EM.EmMisalignAdapter:
    args = dict(root=w["figs"], full22_pins=w["pins"], baseline_pins=None, behaviour_source={}, grade_root=w["grade"], seedrep_root=w["rs"], roster_path=w["roster"])
    args.update(kw)
    return EM.EmMisalignAdapter(**args)


def _read(ad, group):
    inputs = ad.discover()
    groups = {}
    for i in inputs:
        groups.setdefault(ad.comparison_group(i), []).append(i)
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


S1, S1P = f"{EM.SEEDREP_GROUP_PREFIX}{MODEL}_s1", f"{EM.SEEDREP_GROUP_PREFIX}{MODEL}_s1p"


# ============================================================================================
# declarations, discovery, grouping
# ============================================================================================

def test_the_declarations_the_version_and_the_view_grammar():
    assert EM.SEEDREP_SOURCE == "misalign_seedrep" and EM.SEEDREP_SOURCE in EM.EmMisalignAdapter.covers_sources
    # em_misalign 11 (2026-10-03): the anchor currency decides every contrast through the shared emitter (`extra.anchor_selection`)
    assert EM.EmMisalignAdapter.EMISSION_VERSION == DC.emission_version(12)      # 12: a row whose object the registry does not carry is parked by name (registry follow-ups-4 item 8)
    with open(os.path.join(DC.REPO_ROOT, "configs", "registry", "sources_r2_v1.yaml"), encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    assert {r["source"]: r["owner"] for r in data["sources"]}[EM.SEEDREP_SOURCE] == "the authors"
    add = {a["source"]: a for a in data["meta"]["additions"]}[EM.SEEDREP_SOURCE]
    assert add["provisional"] is True and add["paper_build_commit"] is None and "em_misalign" in add["why"]
    assert EM.parse_seedrep_target("qwen35_4b_s2") == {"model": "qwen35_4b", "seed": 2, "paired": False, "seed_view": "qwen35_4b_s2", "selection": "s2"}
    assert EM.parse_seedrep_target("qwen35_4b_s3p")["paired"] is True and EM.parse_seedrep_target("qwen35_4b_s3p")["seed_view"] == "qwen35_4b_s3"
    assert EM.parse_seedrep_target("_anchor_qwen35_4b") is None and EM.parse_seedrep_target("qwen35_4b") is None and EM.parse_seedrep_target("qwen35_4b_s0") is None
    assert EM.EmMisalignAdapter(root="/nonexistent", full22_pins=None, baseline_pins=None, behaviour_source={}, seedrep_root=None).status_sources(S1) == (EM.SEEDREP_SOURCE,)
    # the composed paired write is mapped HERE (these stores name its sha) while the shared EM map deliberately leaves it unmapped
    assert EM.SEEDREP_HARNESS_TO_ROSTER[COMPOSED_H] == COMPOSED_R and COMPOSED_H not in DC.EM_SPELLING_TO_ARMS_KEY


def test_discovery_groups_one_view_per_group_aliases_the_anchor_and_skips_a_poisoned_copy_by_name(tmp_path):
    w = _world(tmp_path)
    ad = _adapter(w)
    inputs = ad.discover()
    groups = {}
    for i in inputs:
        groups.setdefault(ad.comparison_group(i), []).append(i)
    assert {S1, S1P} <= set(groups) and "gemma4e4b" in groups
    disc = ad.last_facts["<seedrep-discover>"]
    assert disc["views"] == [f"{MODEL}_s1", f"{MODEL}_s1p"]
    assert "qwen35_4b_s9.POISONED" in disc["skipped"] and "POISONED" in disc["skipped"]["qwen35_4b_s9.POISONED"]
    assert EM.SEEDREP_ANCHOR_DIR.format(model=MODEL) in disc["skipped"]
    for g in (S1, S1P, MODEL):
        uris = [i.uri for i in groups[g]]
        anchor_uris = [u for u in uris if f"_anchor_{MODEL}" in u]
        assert anchor_uris and all(u.endswith(f"#{EM.alias_of_group(g)}") for u in anchor_uris) and "/" not in EM.alias_of_group(g)
        assert not any("POISONED" in u for u in uris)
    assert EM.group_of_alias(EM.alias_of_group(S1)) == S1
    assert any(u.endswith(f"{COMPOSED_H}/judged.jsonl") for u in (i.uri for i in groups[S1P]))
    # the panel path is untouched: an adapter with no seed-view root discovers the panels alone
    plain = EM.EmMisalignAdapter(root=w["figs"], full22_pins=w["pins"], baseline_pins=None, behaviour_source={}, grade_root=w["grade"], seedrep_root=None)
    assert {plain.comparison_group(i) for i in plain.discover()} == {"gemma4e4b", MODEL}       # the stem's plain + ladder panels are one group
    absent = _adapter(w, seedrep_root=str(tmp_path / "nowhere"))
    assert {absent.comparison_group(i) for i in absent.discover()} == {"gemma4e4b", MODEL}


# ============================================================================================
# the seed view: identity, values, the reserved legs, the anchor parked on the metric
# ============================================================================================

def test_the_seed_view_files_the_identity_from_the_store_and_the_values_from_the_record(tmp_path):
    w = _world(tmp_path)
    out, facts, rep = _read(_adapter(w), S1)
    assert rep.refusals == [] and facts["refused"] == [] and rep.dispositions == 0
    runs = {(p["extra"]["raw_arm"], p["extra"]["behaviour"]): p for p in out["run"]}
    assert set(runs) == {(PSR_H, "leaking"), (PSR_H, "murder"), (SAFETY_H, "leaking"), (SAFETY_H, "murder")}
    psr = runs[(PSR_H, "leaking")]
    assert psr["campaign"] == "sdfbase_full22" and psr["lane_source_legacy"] == EM.SEEDREP_SOURCE and psr["view_legacy"] == f"{MODEL}_s1"
    assert psr["producer_run_key_legacy"] == f"misalign/{MODEL}_s1/{hashlib.sha256(f'{MODEL}_s1|{PSR_H}'.encode()).hexdigest()[:16]}"
    assert psr["producer_object_key_legacy"] == STAGED[PSR_R] and psr["object_id"] == SHA[PSR_R] and psr["campaign_cell_id_legacy"] == PSR_R
    assert psr["extra"]["paper_arm"] == "const_psr" and psr["extra"]["roster_cell_id"] == PSR_R and psr["extra"]["staged_lane"] == "base64s1"
    assert psr["code_sha"] == TREE_S and psr["harness_version"] == TREE_S and psr["cluster_job_id"] == "586157"
    assert (psr["substrate"], psr["substrate_source"]) == ("cluster_a", "recorded") and "served_cluster" in psr["field_confidence_json"]["substrate"]["source"]
    assert psr["is_operating_point"] == 1 and psr["strength_coefficient"] == 1.0 and psr["model_id"] == "qwen35_4b_it"
    # the value: k harmful of n scored, in percent, Wilson 95 %
    m = {(runs_p["extra"]["raw_arm"], runs_p["extra"]["behaviour"]): mm for mm in out["run_metric"] for runs_p in [next(r for r in out["run"] if r["run_id"] == mm["run_id"])]}
    assert (m[(PSR_H, "leaking")]["k"], m[(PSR_H, "leaking")]["n"], m[(PSR_H, "leaking")]["value"]) == (1, 4, 25.0)
    assert (m[(SAFETY_H, "leaking")]["k"], m[(SAFETY_H, "leaking")]["n"]) == (3, 4) and m[(SAFETY_H, "leaking")]["ci_method"] == "wilson_95"
    # the prompted arm: the roster cell id is the object key, the object the declared prompt's text sha
    saf = runs[(SAFETY_H, "murder")]
    assert saf["producer_object_key_legacy"] == SAFETY_R and saf["object_id"] == str(DC.load_seeds().si_object_id("safety_refusal")) and saf["serving_path"] == "passthrough"
    # the legs the policy named and this standup did not draw are a fact, never a refusal
    assert facts["reserved_not_run"] == sorted([LORA_H, LDIFF_H]) and psr["extra"]["reserved_not_run"] == sorted([LORA_H, LDIFF_H])
    assert psr["extra"]["standup_replicate_of_view"] is None and psr["extra"]["paired_of"] is None and psr["extra"]["paired_view"] is False
    # the instrument: the runner's T1 as SENT, cap 3,000 read fixed, seed rule none; never the vendor shape, never the canonical ladder row
    for inst in out["instrument"]:
        assert (inst["decode_source"], inst["decode_temperature"], inst["decode_top_p"], inst["decode_seed_rule"]) == ("runner_forced_T1", 1.0, None, "none")
        assert inst["token_cap_tokens"] == 3000 and inst["token_cap_ladder_json"] == [3000] and inst["read_at"] == "fixed" and inst["harness"] == "agentic_misalignment"
    assert set(facts["instrument_ids"]) == {"agentic_leaking", "agentic_murder"}
    assert all(cid != facts["instrument_ids"][ev] for ev, cid in facts["canonical_seed_instrument_ids"].items() if isinstance(cid, str) and len(cid) == 64)
    assert psr["decode_sent_json"] == {"temperature": 1.0} and psr["extra"]["server_default_policy"] == "contract_override"
    assert psr["extra"]["server_default_generation_config"]["top_k"] == 20 and psr["extra"]["decode_policy_ref"] == EM.SEEDREP_DECODE_POLICY_REF
    # the judgment: the seeded Sonnet-5 classifier, the rows' pool, no refusals here
    jud = {j["run_id"]: j for j in out["judgment"]}
    assert jud[psr["run_id"]]["judge_id"] == DC.judge_fields(**DC.AGENTIC_JUDGE)["judge_id"] and jud[psr["run_id"]]["api_pool"] == "lowprio"
    # THE ANCHOR (v5): a run_contrast per row against the REGISTERED plain none run and the SECOND judgment on it; nothing parked
    assert rep.contrasts == 4 and rep.contrasts_parked == 0 and len(out["run_contrast"]) == 4
    assert all(mm["extra"].get("delta_block") is None for mm in out["run_metric"])
    panel, pfacts, _ = _read(_adapter(w), MODEL)
    plain = {p["eval_id"]: p for p in panel["run"] if p["producer_run_key_legacy"] == f"misalign/{MODEL}/{PLAIN_KEY}"}
    second = {v["run_id"]: v["second_judgment_id"] for v in pfacts["anchor_second_judgments"].values() if v["filed"]}
    ct = {c["run_id"]: c for c in out["run_contrast"]}[psr["run_id"]]
    assert ct["anchor_run_id"] == plain["agentic_leaking"]["run_id"] and ct["anchor_judgment_id"] == second[plain["agentic_leaking"]["run_id"]]
    assert ct["anchor_kind"] == "sibling_none" and ct["n_pairs"] == 4 and ct["effect_kind"] == "cohens_h" and ct["computed_by"] == "producer"
    d, lo, hi, p = DC.two_prop(0.25, 4, 0.5, 4)
    assert ct["delta"] == pytest.approx(d) and ct["delta"] == pytest.approx(-25.0) and ct["delta_lo"] == pytest.approx(lo) and ct["delta_p"] == pytest.approx(p)
    assert ct["effect_size"] == pytest.approx(DC.cohen_h(0.25, 0.5))
    cx = ct["extra"]
    assert cx["anchor_is_common_term"] is True and cx["paired"] is False and cx["cap_matched"] is True and (cx["arm_cap"], cx["anchor_cap"]) == (3000, 3000)
    assert (cx["k_anchor"], cx["n_anchor"], cx["anchor_rate_pct"]) == (2, 4, 50.0) and cx["instrument_match"]["decode_and_cap"] == "match"
    assert cx["instrument_match"]["anchor_instrument_id"] == plain["agentic_leaking"]["instrument_id"] and "harness_version" in cx["instrument_match"]["fields_differing"]
    assert cx["anchor_responses_sha256"] == F.sha256_file(os.path.join(w["grade"], "misalign", MODEL, "agentic", "none", "responses.jsonl"))
    assert psr["extra"]["external_anchor"]["anchor_run_id"] == plain["agentic_leaking"]["run_id"] and psr["extra"]["external_anchor"]["filed_as"] == "run_contrast"
    assert psr["extra"]["anchor_is_common_term"] is True and facts["contrasts_from_blocks"] == 4 and set(facts["anchor_runs"]) == {"leaking", "murder"}
    # the seed rows' instruments never equal the anchor run's: different ids, no shared collapse key
    assert set(facts["instrument_ids"].values()).isdisjoint({p["instrument_id"] for p in plain.values()})
    # three item rows per response with the flags on the harmful row; the degeneration measures from the producer's counts
    assert len(out["item_result"]) == 3 * 16 and len(out["item_flags"]) == 16
    assert {it["metric_id"] for it in out["item_result"]} == {"harmful", "classifier_harmful", "structural_gate"}
    assert {d_["measure"] for d_ in out["run_degeneration"]} >= {"at_token_cap", "classifier", "union5"}


def test_the_paired_view_marks_the_standup_replicate_and_the_composed_object(tmp_path):
    w = _world(tmp_path)
    out, facts, rep = _read(_adapter(w), S1P)
    assert rep.refusals == [] and facts["refused"] == [] and facts["paired"] is True and facts["seed_view"] == f"{MODEL}_s1"
    runs = {(p["extra"]["raw_arm"], p["extra"]["behaviour"]): p for p in out["run"]}
    lora = runs[(LORA_H, "leaking")]
    assert lora["producer_object_key_legacy"] == STAGED[LORA_R] and lora["object_id"] == SHA[LORA_R] and lora["view_legacy"] == f"{MODEL}_s1p"
    assert lora["extra"]["standup_replicate_of_view"] == f"{MODEL}_s1" and lora["extra"]["paired_of"] == "s1" and "STANDUP REPLICATE" in lora["extra"]["standup_replicate_note"]
    assert lora["producer_run_key_legacy"].startswith(f"misalign/{MODEL}_s1p/") and lora["code_sha"] == TREE_P
    comp = runs[(COMPOSED_H, "murder")]
    assert comp["producer_object_key_legacy"] == STAGED[COMPOSED_R] and comp["object_id"] == SHA[COMPOSED_R] and comp["extra"]["paper_arm"] == "diff_psr"
    assert comp["extra"]["paired_of"] == "s1" and comp["extra"]["standup_replicate_of_view"] is None and comp["extra"]["staged_lane"] == "paired_eb64s1"
    assert facts["reserved_not_run"] == [] and facts["only_cells"] is None
    # the paired view's instruments are its own tree's (harness_version), distinct from the seed view's
    seed_out, seed_facts, _ = _read(_adapter(w), S1)
    assert set(facts["instrument_ids"].values()).isdisjoint(seed_facts["instrument_ids"].values())
    # the paired view's skipped list names the poisoned copy beside its cells
    assert any("POISONED" in s["path"] for s in facts["skipped"])


# ============================================================================================
# counts: missing and blocked rows are facts; the value is judged - blocked
# ============================================================================================

def test_a_missing_judged_row_and_a_blocked_row_are_counted_and_the_reading_is_a_lower_bound(tmp_path):
    w = _world(tmp_path, seed_kw=dict(blocked={PSR_H: (("leaking", 0),)}, missing={SAFETY_H: (("murder", 3),)}))
    out, facts, rep = _read(_adapter(w), S1)
    assert rep.refusals == [] and facts["refused"] == []
    runs = {(p["extra"]["raw_arm"], p["extra"]["behaviour"]): p for p in out["run"]}
    psr = runs[(PSR_H, "leaking")]
    assert psr["extra"]["judge_blocked"] == 1 and psr["extra"]["lower_bound"] is True and psr["n_items_judged"] == 3 and psr["n_items_generated"] == 4
    jud = {j["run_id"]: j for j in out["judgment"]}
    assert jud[psr["run_id"]]["n_judge_refusals"] == 1
    m = next(mm for mm in out["run_metric"] if mm["run_id"] == psr["run_id"])
    assert (m["n"], m["k"], m["n_abstentions"]) == (3, 1, 1) and m["extra"]["suggestive"] is True
    saf = runs[(SAFETY_H, "murder")]
    assert saf["extra"]["judged_rows_missing"] == 1 and saf["n_items_generated"] == 4 and saf["n_items_judged"] == 3
    blocked_items = [it for it in out["item_result"] if it["run_id"] == psr["run_id"] and it["label"] == "judge_blocked"]
    assert len(blocked_items) == 3 and all(it["value"] is None for it in blocked_items)
    assert facts["judge_blocked"] == {f"{PSR_H}/leaking": 1, f"{PSR_H}/murder": 0, f"{SAFETY_H}/leaking": 0, f"{SAFETY_H}/murder": 0}


# ============================================================================================
# no in-root none; refusals by name
# ============================================================================================

def test_an_in_root_none_refuses_the_view_by_name(tmp_path):
    w = _world(tmp_path, seed_kw=dict(judged_none=True))
    with pytest.raises(AdapterRefusal, match="in-root unsteered leg was judged"):
        _read(_adapter(w), S1)
    w = _world(tmp_path / "b", paired_kw=dict(paired_rows=True))
    with pytest.raises(AdapterRefusal, match="paired.csv carries 1 row"):
        _read(_adapter(w), S1P)
    w = _world(tmp_path / "c", seed_kw=dict(measure_none=True))
    with pytest.raises(AdapterRefusal, match="measure_none is True"):
        _read(_adapter(w), S1)
    w = _world(tmp_path / "d", seed_kw=dict(pairing_anchor="exp/misalign/qwen35_4b_s1/agentic/none"))
    with pytest.raises(AdapterRefusal, match="pairing_anchor"):
        _read(_adapter(w), S1)


def test_a_vendor_sampler_cell_is_refused_and_the_guard_refuses_a_vendor_shaped_instrument(tmp_path):
    w = _world(tmp_path, paired_kw=dict(sampler="vendor", sampling={"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0}))
    out, facts, rep = _read(_adapter(w), S1P)
    assert "run" not in out and len(facts["refused"]) == 4 and all("not the runner's default t1" in r for r in facts["refused"])
    assert rep.dispositions == 4 and all(d["state"] == "held" for d in out["disposition"])
    # the guard: an instrument record shaped like the vendor preset refuses the view outright
    ad = _adapter(w)
    batch = RecordBatch("x", S1P)
    vendor = dict(DC.instrument_fields(eval_id="agentic_leaking", item_set_id="agentic_leaking_scenarios", harness="agentic_misalignment", harness_version="x",
                                       harness_config_hash="x", scoring_rule="rate_over_scenarios", token_cap_tokens=3000, decode_source="vendor_recommendation",
                                       decode={"temperature": 0.7, "top_p": 0.8, "top_k": 20}, decode_seed_rule="per_response", thinking_mode="off",
                                       serving_path="lora_native", applied_positions_mode="all", judge_id=None))
    batch.add("instrument", vendor, PRODUCER, NOW, observe=False)
    with pytest.raises(AdapterRefusal, match="vendor-shaped decode"):
        ad._seedrep_guard_instruments(batch, {"group": S1P, "model": MODEL, "cap": 3000, "instrument_ids": {}})


def test_a_served_sha_that_differs_from_the_pin_and_an_unmapped_harness_spelling_refuse_by_name(tmp_path):
    w = _world(tmp_path, seed_kw=dict(bank_sha_override={PSR_H: "f" * 64}))
    out, facts, rep = _read(_adapter(w), S1)
    assert any(f"{PSR_H}/leaking" in r and "!= the pinned sha" in r for r in facts["refused"])
    assert {p["extra"]["raw_arm"] for p in out["run"]} == {SAFETY_H}
    w = _world(tmp_path / "b", seed_kw=dict(meta_sha_override={PSR_H: "e" * 64}))
    out, facts, rep = _read(_adapter(w), S1)
    assert any("object_artifact_sha256_by_cell" in r for r in facts["refused"]) and {p["extra"]["raw_arm"] for p in out["run"]} == {SAFETY_H}
    # a directory the policy does not name refuses the view (a judged cell the lane never declared)
    w = _world(tmp_path / "c")
    os.makedirs(os.path.join(w["rs"], f"{MODEL}_s1", "agentic", "si_unknown__thing"), exist_ok=True)
    with pytest.raises(AdapterRefusal, match="not in the lane's measured_legs"):
        _read(_adapter(w), S1)


def test_an_absent_anchor_store_parks_the_binding_with_no_delta_and_the_rows_still_file(tmp_path):
    w = _world(tmp_path, with_anchor=False)
    out, facts, rep = _read(_adapter(w), S1)
    assert rep.refusals == [] and len(out["run"]) == 4 and facts["anchor"]["state"] == "absent"
    blk = out["run_metric"][0]["extra"]["delta_block"]
    assert blk["delta"] is None and blk["anchor"]["anchor_run_id"] is None and blk["anchor_is_common_term"] is True and "not readable" in blk["delta_method"]


# ============================================================================================
# v5: the plain none registered as a run, the second judgment, the guard
# ============================================================================================

def test_the_judged_plain_none_of_the_anchor_stem_is_registered_as_a_run_with_relus_second_judgment(tmp_path):
    w = _world(tmp_path)
    out, facts, rep = _read(_adapter(w), MODEL)
    runs = {p["run_id"]: p for p in out["run"]}
    plain = {p["eval_id"]: p for p in runs.values() if p["producer_run_key_legacy"] == f"misalign/{MODEL}/{PLAIN_KEY}"}
    ladder = {p["eval_id"]: p for p in runs.values() if p["producer_run_key_legacy"] == f"misalign_ladder/{MODEL}/{PLAIN_KEY}"}
    assert set(plain) == set(ladder) == {"agentic_leaking", "agentic_murder"} and rep.superseded_unmeasured_skipped == 0 and rep.refusals == []
    for ev, p in plain.items():
        assert p["token_cap_tokens"] == 3000 and p["token_cap_rung_index"] == 0 and p["token_cap_ladder_json"] == [3000, 16384]
        assert p["lifecycle_state"] == "generated" and p["superseded_by_run_id"] is None and p["status_ruling_id"] is None
        assert p["extra"]["superseded_for_headline_by_run_id"] == ladder[ev]["run_id"] and p["extra"]["registered_as_run"] is True
        assert p["instrument_id"] != ladder[ev]["instrument_id"] and p["run_id"] != ladder[ev]["run_id"]
        assert (p["decode_sent_json"], p["decode_source"], p["decode_seed_rule"]) == ({"temperature": 1.0}, "runner_forced_T1", "none")
        assert p["extra"]["plain_none_first_judgment"]["judge_pools"] == ["highprio"]
    states = [s for s in out["run_state"] if s["run_id"] in plain_ids(plain)]
    assert {s["state"] for s in states} == {"judged"}       # judged, never superseded: the generated state is the run row's own lifecycle_state
    # two judgments per plain run: the store's own, then the re-judge (same judge id, another judgment id, the producing session on the row)
    jud = {}
    for j in out["judgment"]:
        jud.setdefault(j["run_id"], []).append(j)
    em_judge = DC.judge_fields(**DC.AGENTIC_JUDGE)["judge_id"]
    for ev, p in plain.items():
        js = sorted(jud[p["run_id"]], key=lambda j: (j.get("producer_session") or ""))      # the store's own first (no session), the re-judge second
        assert len(js) == 2 and {j["judge_id"] for j in js} == {em_judge} and js[0]["judgment_id"] != js[1]["judgment_id"]
        first, second = js
        assert first["api_pool"] == "highprio" and first.get("producer_session") is None and first["n_items_judged"] == 4
        assert second["api_pool"] == "lowprio" and second["producer_session"] == EM.SEEDREP_ANCHOR_SESSION.format(model=MODEL) and second["extra"]["second_judgment_of_run"] is True
        assert second["extra"]["first_judgment_id"] == first["judgment_id"] and second["judgment_id"] == EM.EmMisalignAdapter.anchor_second_judgment_id(p["run_id"], em_judge, MODEL)
        b = ev.split("_", 1)[1]
        ms = {m["judgment_id"]: m for m in out["run_metric"] if m["run_id"] == p["run_id"]}
        assert ms[first["judgment_id"]]["k"] == sum(PLAIN_HARM[b]) and ms[second["judgment_id"]]["k"] == sum(ANCHOR_HARM[b]) and ms[second["judgment_id"]]["n"] == 4
        assert ms[second["judgment_id"]]["extra"]["anchor_is_common_term"] is True
        items = [i for i in out["item_result"] if i["run_id"] == p["run_id"]]
        assert len(items) == 3 * 4 and {i["judgment_id"] for i in items} == {second["judgment_id"]}
    assert len(jud[ladder["agentic_leaking"]["run_id"]]) == 1
    assert {k: v["filed"] for k, v in facts["anchor_second_judgments"].items()} == {f"{MODEL}/leaking": True, f"{MODEL}/murder": True}


def plain_ids(plain):
    return {p["run_id"] for p in plain.values()}


def test_guard_an_unjudged_plain_none_stays_skipped_and_the_seed_rows_park(tmp_path):
    w = _world(tmp_path, plain_judged=False)
    out, facts, rep = _read(_adapter(w), MODEL)
    assert not any(p["producer_run_key_legacy"].startswith(f"misalign/{MODEL}/") for p in out["run"]) and rep.superseded_unmeasured_skipped == 2
    assert "anchor_second_judgments" not in facts and "registered_plain_none" not in facts
    sout, sfacts, srep = _read(_adapter(w), S1)
    assert srep.contrasts == 0 and srep.contrasts_parked == 4 and "run_contrast" not in sout and sfacts["anchor_runs"] is None
    assert "not registered" in sfacts["anchor_runs_why"]
    psr_run = next(p["run_id"] for p in sout["run"] if p["extra"]["raw_arm"] == PSR_H and p["extra"]["behaviour"] == "leaking")
    blk = next(m for m in sout["run_metric"] if m["run_id"] == psr_run)["extra"]["delta_block"]
    assert blk["anchor"]["anchor_run_id"] is None and blk["delta"] == pytest.approx(-25.0) and blk["anchor_is_common_term"] is True


def test_a_plain_none_whose_panel_statistic_is_not_its_record_is_refused_not_registered(tmp_path):
    w = _world(tmp_path, plain_none_kn={"leaking": (4, 1), "murder": (4, 0)})       # the record says 3 of 4 leaking
    out, facts, rep = _read(_adapter(w), MODEL)
    plain = [p for p in out["run"] if p["producer_run_key_legacy"].startswith(f"misalign/{MODEL}/")]
    assert {p["eval_id"] for p in plain} == {"agentic_murder"}
    assert any("leaking/none" in r and "is not registered on a statistic" in r for r in facts["refused"])
    assert any(d["state"] == "held" and d["eval_id"] == "agentic_leaking" for d in out["disposition"])


def test_without_a_seed_view_lane_naming_the_anchor_no_plain_none_is_registered(tmp_path):
    w = _world(tmp_path)
    ad = _adapter(w, seedrep_root=None)
    out, facts, rep = _read(ad, MODEL)
    assert not any(p["producer_run_key_legacy"].startswith(f"misalign/{MODEL}/") for p in out["run"]) and rep.superseded_unmeasured_skipped == 2
    assert ad._anchor_stems() == set()


# ============================================================================================
# the shape equals em_misalign's misalignment rows
# ============================================================================================

def test_the_shape_equals_em_misaligns_misalignment_rows(tmp_path):
    w = _world(tmp_path)
    out, facts, rep = _read(_adapter(w), S1)
    metric, threshold, bon_n = DC.HEADLINE["misalignment"]
    assert {m["metric_id"] for m in out["run_metric"]} == {metric} == {"misaligned_rate"} and {m["threshold"] for m in out["run_metric"]} == {-1.0}
    em_judge = DC.judge_fields(**DC.AGENTIC_JUDGE)["judge_id"]
    assert {j["judge_id"] for j in out["judgment"]} == {em_judge} and {p["judge_id"] for p in out["run"]} == {em_judge}
    for p in out["run"]:
        assert p["item_set_id"] == f"{p['eval_id']}_scenarios" and p["scoring_rule"] == "rate_over_scenarios" and p["harness"] == "agentic_misalignment"
        assert p["extra"]["paper_component"] == "misalignment" and p["eval_id"] in ("agentic_leaking", "agentic_murder") and p["thinking_mode"] == "off"
    assert {d["denominator_unit"] for d in out["run_degeneration"] if d["measurement_state"] == "measured"} == {"scenarios"}
    assert {m["ci_method"] for m in out["run_metric"]} == {"wilson_95"} and all(0.0 <= m["value"] <= 100.0 for m in out["run_metric"])
    # the source_state rows name the view's entries and its agentic entries (the poisoned copy visible as a member the reader did not read)
    ad = _adapter(w)
    inputs = ad.discover()
    rows = ad.source_states(_ctx(), [i for i in inputs if ad.comparison_group(i) == S1P])
    ents = [r for r in rows if r.json_path == "$.entries"]
    assert any(any("POISONED" in m for m in r.current_members) for r in ents)
    assert any(r.json_path == "$.arm" and set(r.current_members) == {COMPOSED_H, LORA_H} for r in rows)
