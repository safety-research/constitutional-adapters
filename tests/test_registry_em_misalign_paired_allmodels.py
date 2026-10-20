"""The paired-write difference-ReLU agentic roots of the other model families (`em_misalign` version 8; source `misalign_paired_allmodels`,
em_misalign's fifth root) and the judge-pool escalation chains every em_misalign root accepts since version 8.

A SYNTHETIC world (placeholder shas, TESTHOST, invented scenario ids): one gemma paired root under the seed roots' mirror in the runner's shape --
one steered key (the composed difference-ReLU object), no none, a judge-blocked leaking row, `lowprio>highprio` escalation chains, a header-only
`paired.csv`, a row store, an inert `agentic.POISONED-conc200/` copy -- plus an in-memory registry connection carrying the object row and the
model's unsteered agentic runs (the frame's anchors) and the staging manifest."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys

import pytest
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
sys.path.insert(0, HERE)

from sisteer.registry import db as DB  # noqa: E402
from sisteer import site as SITE  # noqa: E402
from sisteer.registry import harvest as H  # noqa: E402
from sisteer.registry import identity as I  # noqa: E402
from sisteer.registry import reader_state as RS  # noqa: E402
from sisteer.registry import records as R  # noqa: E402
from sisteer.registry import schema_def as S  # noqa: E402
from sisteer.registry import synth  # noqa: E402
from sisteer.registry import validate as VAL  # noqa: E402
from sisteer.registry.adapters import anchor_current as AC  # noqa: E402
from sisteer.registry.adapters import declared_common as DC  # noqa: E402
from sisteer.registry.adapters import em_misalign as EM  # noqa: E402
from sisteer.registry.adapters.base import AdapterContext, AdapterRefusal, RecordBatch  # noqa: E402
from sisteer.registry import synth

import registry_declared_fixtures as F  # noqa: E402

NOW = "2026-09-22T09:00:00.000Z"
PRODUCER = {"script": "tests.em_misalign_paired_allmodels", "code_sha": synth.code_sha("a"), "session": None, "host": "TESTHOST", "substrate": "grading_host"}
LORA, PASS = "proxy_vllm+lora+interleaved", "proxy_vllm+passthrough+interleaved"
MODEL, MODEL_ID = "gemma4e4b", "gemma4e4b_it"
TARGET, VIEW = f"{MODEL}_paired_d1_{MODEL}_s1", f"paired_d1_{MODEL}_s1"
TREE, JOB = synth.code_sha("6"), "597492"
KEY_H, KEY_R = EM.PAIRED_KEY_H, EM.PAIRED_KEY_R
STAGED = f"paired_eb8__{KEY_R}"
SHA = hashlib.sha256(b"synthetic gemma paired-write difference relu").hexdigest()
OTHER_SHA = hashlib.sha256(b"another synthetic object").hexdigest()
PETRI_KEY = "5034c0d5719d41aa"
ITEMS = {"leaking": ["leaking_explicit-alpha_none", "leaking_explicit-beta_none"], "murder": ["murder_explicit-alpha_none", "murder_explicit-beta_none"]}
N_SAMPLES = 2
# per behaviour: per slot (2 items x 2 samples) the harmful verdict; `blocked` = judge-blocked slots; `pools` = the judge_pool spelling per slot
ROWS = {"leaking": dict(h=[1, 0, 1, 1], blocked=[1], pools=["highprio", "lowprio>highprio", "highprio", "lowprio>highprio"]),
        "murder": dict(h=[0, 1, 0, 0], blocked=[], pools=["highprio", "highprio", "highprio", "highprio"])}
GROUP = f"{EM.PAIRED_GROUP_PREFIX}{TARGET}"
ANCHOR_RUNS = {"agentic_leaking": [("run:none:leaking:16k", 16384, 101, 510, 1), ("run:none:leaking:3k", 3000, 99, 510, 1), ("run:none:leaking:old", 3000, 80, 510, 0)],
               "agentic_murder": [("run:none:murder:16k", 16384, 58, 490, 1)]}


def _w(path: str, obj, jsonl: bool = False) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        if jsonl:
            for r in obj:
                fh.write(json.dumps(r) + "\n")
        else:
            json.dump(obj, fh, indent=1)
    return path


def _sha(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


#: a computed twin's computation id beside a producer's row (computation '') at one (run, judgment, metric, threshold)
TWIN = "comp:twin"


def _reading_key(rid: str, computation_id: str = "") -> str:
    return json.dumps({"run_id": rid, "judgment_id": "j:" + rid, "metric_id": "misaligned_rate", "threshold": -1.0, "computation_id": computation_id}, sort_keys=True)


def _conn(*, object_row=("composed", "difference_relu", "gemma4e4b_base", "approved", STAGED, "composition"), with_anchors: bool = True,
          observed: bool = False, withdrawn=(), twins=None, rejudged=None, abandoned=()):
    """An in-memory registry with the tables the reader consults: the object row, the model's unsteered agentic runs and the frame's flags --
    a TABLE named `v_run_current` in the reader-state sidecar's shape (reader_state/2), which is what the
    reader's `_paired_in_frame_anchors` reads on the grading host (the fold) and on a shard (the shipped table) alike -- and `run_metric` in its
    real identity shape (judgment, threshold, computation). `observed` adds a grading-host-shaped `observation` table: every anchor reading
    observed current under one revision, except the readings of the runs in `withdrawn` (current under no revision -- the paper frame's
    withdrawn-reading predicate drops them); `twins` maps a run id to a computed twin row at computation `TWIN` with another value, kept
    (1) or withdrawn (0). Every anchor reading sits under a complete judgment `j:<run>`; `rejudged` maps a run id to its re-judge --
    the judgment re-filed `superseded` by a new complete one `j2:<run>` under the same judge, carrying a replacement reading `(k, n)` or,
    with None, none; the runs in `abandoned` have their only judgment released (`abandoned`)."""
    twins = dict(twins or {})
    rejudged = dict(rejudged or {})
    assert observed or not withdrawn and all(twins.values()), "a withdrawn reading needs the observation table"
    con = sqlite3.connect(":memory:")
    con.executescript("""
        CREATE TABLE object (object_id TEXT PRIMARY KEY, object_kind TEXT, paper_label TEXT, fit_model_id TEXT, status TEXT, campaign_cell_id_legacy TEXT, derivation TEXT);
        CREATE TABLE run (run_id TEXT PRIMARY KEY, model_id TEXT, object_id TEXT, eval_id TEXT, view_legacy TEXT, lane_source_legacy TEXT, instrument_id TEXT,
                          token_cap_tokens INTEGER, decode_source TEXT);
        CREATE TABLE v_run_current (run_id TEXT PRIMARY KEY, is_current INTEGER, why TEXT, held_out INTEGER, in_current_paper INTEGER,
                                    selection_state TEXT, roster_entry_ids_json TEXT);
        CREATE TABLE run_metric (run_id TEXT, judgment_id TEXT, metric_id TEXT, threshold REAL, computation_id TEXT, k REAL, n INTEGER, value REAL);
        CREATE TABLE judgment (judgment_id TEXT PRIMARY KEY, run_id TEXT, judge_id TEXT, status TEXT, superseded_by_judgment_id TEXT);
    """)
    if observed:
        con.execute("CREATE TABLE observation (source_id TEXT, revision_id TEXT, table_name TEXT, observation_key TEXT, current INTEGER)")
    if object_row is not None:
        con.execute("INSERT INTO object VALUES (?,?,?,?,?,?,?)", (SHA,) + tuple(object_row))
    if with_anchors:
        for ev, runs in ANCHOR_RUNS.items():
            for rid, cap, k, n, icp in runs:
                con.execute("INSERT INTO run VALUES (?,?,?,?,?,?,?,?,?)", (rid, MODEL_ID, "none", ev, MODEL_ID, "em/agentic/gemma4e4b_it", "inst:" + rid, cap, "runner_forced_T1"))
                con.execute("INSERT INTO v_run_current VALUES (?,?,?,?,?,?,?)",
                            (rid, icp, (None if icp else "selection_state superseded (a later run at the same key)"), 0, icp,
                             ("selected" if icp else "superseded"), ('["re:1"]' if icp else "[]")))
                con.execute("INSERT INTO run_metric VALUES (?,?,?,?,?,?,?,?)", (rid, "j:" + rid, "misaligned_rate", -1.0, "", k, n, 100.0 * k / n))
                status = "abandoned" if rid in abandoned else ("superseded" if rid in rejudged else "complete")
                con.execute("INSERT INTO judgment VALUES (?,?,?,?,?)", ("j:" + rid, rid, "judge:em", status, ("j2:" + rid if rid in rejudged else None)))
                if rid in rejudged:
                    con.execute("INSERT INTO judgment VALUES (?,?,?,?,?)", ("j2:" + rid, rid, "judge:em", "complete", None))
                    if rejudged[rid] is not None:
                        k2, n2 = rejudged[rid]
                        con.execute("INSERT INTO run_metric VALUES (?,?,?,?,?,?,?,?)", (rid, "j2:" + rid, "misaligned_rate", -1.0, "", k2, n2, 100.0 * k2 / n2))
                if observed:
                    con.execute("INSERT INTO observation VALUES (?,?,?,?,?)", ("src:t", "rev1", "run_metric", _reading_key(rid), 0 if rid in withdrawn else 1))
                if rid in twins:
                    con.execute("INSERT INTO run_metric VALUES (?,?,?,?,?,?,?,?)", (rid, "j:" + rid, "misaligned_rate", -1.0, TWIN, k + 1, n, 100.0 * (k + 1) / n))
                    if observed:
                        con.execute("INSERT INTO observation VALUES (?,?,?,?,?)", ("src:t", "rev1", "run_metric", _reading_key(rid, TWIN), int(twins[rid])))
        # another model's none never enters
        con.execute("INSERT INTO run VALUES (?,?,?,?,?,?,?,?,?)", ("run:qwen:none", "qwen35_4b_it", "none", "agentic_leaking", "qwen35_4b_it", "x", "i", 3000, "runner_forced_T1"))
        con.execute("INSERT INTO v_run_current VALUES (?,?,?,?,?,?,?)", ("run:qwen:none", 1, None, 0, 1, "selected", '["re:2"]'))
    con.commit()
    return con


def _ctx(conn=None) -> AdapterContext:
    return AdapterContext(root="", now=NOW, producer=dict(PRODUCER), conn=conn)


def _rows(key: str, rows_spec=ROWS, judge_model=EM.SEEDREP_JUDGE_MODEL, cap: int = 3000, at_cap_slots=()):
    resp, judged = [], []
    for b, ids in ITEMS.items():
        spec = rows_spec[b]
        slots = [(item, s) for item in ids for s in range(N_SAMPLES)]
        for si, (item, s) in enumerate(slots):
            capped = (b, si) in at_cap_slots
            r = {"family": "agentic", "cell_id": key, "virtual_model": f"petri__{key}", "item_id": item, "sample_idx": s, "behavior": b,
                 "prompt_sha": hashlib.sha256(f"{item}|{s}".encode()).hexdigest(), "response": f"placeholder response {key} {b} {si}",
                 "finish_reason": ("length" if capped else "stop"), "usage": {"prompt_tokens": 100, "completion_tokens": (cap if capped else 40), "total_tokens": 140},
                 "model_reported": f"petri__{key}"}
            resp.append(r)
            blk = si in spec["blocked"]
            h = spec["h"][si]
            judged.append(dict(r, judge_tag=EM.SEEDREP_JUDGE_TAGS[b], judge_model=judge_model, judge_pool=spec["pools"][si], judged_at="2026-09-22T07:00:00Z",
                               structural_gate=(0 if blk else 1), judge_blocked=blk, classifier_harmful=(None if blk else h), harmful=(None if blk else h)))
    return resp, judged


def _summary(target: str, key: str, judged: list, ident: str) -> list:
    out = []
    for b in ITEMS:
        rows = [r for r in judged if r["behavior"] == b]
        scored = [r for r in rows if not r["judge_blocked"]]
        n, k = len(scored), sum(int(r["harmful"]) for r in scored)
        cap_k = sum(1 for r in rows if r["finish_reason"] == "length")
        for metric, kk, est in (("harmful", k, k / n), ("at_cap", cap_k, cap_k / n), ("loop", 0, 0.0), ("degenerate", 0, 0.0), ("harmful_reasoned", k, k / n),
                                ("harmful_bare", 0, 0.0), ("harmful_fuzzy", k, k / n)):
            out.append(dict(target=target, arm=key, family="agentic", behavior=b, metric=metric, n=n, k=kk, estimate=est, ci_low=max(0.0, est - 0.1),
                            ci_high=min(1.0, est + 0.1), ci_method="wilson95", boot_low=None, boot_high=None, note=None, engine=ident, standup_engine=None,
                            sampler="t1", sampling=json.dumps({"temperature": 1.0})))
        out.append(dict(target=target, arm=key, family="agentic", behavior=b, metric="response_chars", n=n, k=None, estimate=1234.0, ci_low=None, ci_high=None,
                        ci_method=None, boot_low=None, boot_high=None, note=None, engine=ident, standup_engine=None, sampler="t1", sampling=json.dumps({"temperature": 1.0})))
    return out


PAIRED_COLS = ["target", "arm", "ref", "family", "behavior", "metric", "n_pairs", "mean_arm", "mean_none", "diff", "boot_low", "boot_high", "arm_only", "none_only",
               "mcnemar_p", "note", "engine", "anchor_engine", "standup_engine", "anchor_standup_engine", "cross_engine"]


def _root(rs: str, target: str = TARGET, *, tree: str = TREE, job: str = JOB, sha: str = SHA, artifact_sha=None, source_sha=None, rows_spec=ROWS, measure_none: bool = False,
          none_judged: bool = False, paired_rows: bool = False, view=None, engine_kind: str = "lora_native", cap: int = 3000, at_cap_slots=(), poisoned: bool = True,
          judge_model: str = EM.SEEDREP_JUDGE_MODEL, kind: str = EM.PAIRED_STAGING_KIND, result_ok: bool = True, staged_id: str = STAGED) -> str:
    """One paired all-models root in the runner's shape (the seed roots' file set)."""
    import pandas as pd
    tdir = os.path.join(rs, target)
    artifact_sha = sha if artifact_sha is None else artifact_sha
    source_sha = sha if source_sha is None else source_sha
    view = VIEW if view is None else view
    bank = {KEY_R: {"identity": "staged", "object_sha256": sha, "positions_at_fit": "all", "sidecar_object_sha256": sha,
                    "staged_from": {"cell_id": KEY_R, "kind": kind, "lane": "paired_eb8", "key": staged_id, "object_sha256": sha, "pulled_at": "2026-09-22T06:30:00Z",
                                    "source": f"$STAGE/sdfbase/relu_paired/MANIFEST_paired_d1_{MODEL}.json object_sha256 {source_sha}", "staged_by": "synthetic"}}}
    legs = [{"cell_class": "base", "cell_id": "none", "control_of": None, "harness_config_hash": "4269a5c2d945d1c4", "intervention_hash": "2ff9cdff091aef6f", "kind": "none",
             "object_path": None, "petri_run_key": "6dfb68eb01e3681f", "serving_identity": PASS, "si_id": None, "si_prefix_hash": None, "virtual_model_name": "petri__base",
             "si_id_for_runner": None, "measured": measure_none},
            {"cell_class": "class_a", "cell_id": KEY_H, "control_of": None, "harness_config_hash": "f411a1c5a34edb57", "intervention_hash": "5923ecb7916846e9", "kind": kind,
             "object_path": None, "petri_run_key": PETRI_KEY, "serving_identity": LORA, "si_id": None, "si_prefix_hash": None, "virtual_model_name": f"petri__{KEY_H}",
             "si_id_for_runner": None, "measured": True},
            {"cell_class": "base", "cell_id": "none__lora0", "control_of": None, "harness_config_hash": "4269a5c2d945d1c4", "intervention_hash": "2ff9cdff091aef6f", "kind": "none",
             "object_path": None, "petri_run_key": "6ed3a09ba4360e3f", "serving_identity": LORA, "si_id": None, "si_prefix_hash": None, "virtual_model_name": "petri__none__lora0",
             "si_id_for_runner": None, "measured": False}]
    measured = ([KEY_H] + (["none"] if measure_none else []))
    policy = {"measure_none": measure_none, "measured_cells_declared": None, "lane": "paired", "pairing_anchor": f"exp/misalign/{MODEL}/agentic/none",
              "source": f"misalign_full22_{target}.yaml misalign_policy", "served_legs": ["none", KEY_H, "none__lora0"], "measured_legs": measured,
              "unmeasured_served": [x for x in ("none", "none__lora0") if x not in measured]}
    ep = {"model": "google/gemma-4-E4B-it", "model_revision": F.REVISION, "model_key": MODEL, "proxy_url": "http://127.0.0.1:1/v1", "max_model_len": 16384, "max_num_seqs": 31,
          "attested_max_model_len": 16384, "attested_max_num_seqs": 31, "thinking_mode": "off", "node": "gpu-1", "slurm_job_id": job, "cells": ["none", KEY_R],
          "harness_cells": ["none", KEY_H], "server_default_policy": "contract_override", "override_generation_config": None, "bank_objects": bank}
    _w(os.path.join(tdir, "cells_full.json"), {"target": target, "model_key": MODEL, "wave_view": view, "wave_sha": tree, "written_at": "2026-09-22T06:40:00Z",
                                              "measured_policy": policy, "endpoint": ep,
                                              "cells_served": {"manifest": f"/shared/tester/repo/{tree}/configs/capeval/manifests/misalign_full22_{target}.yaml"}, "legs": legs})
    _w(os.path.join(tdir, "cells.json"), {"cells": [{"cell_id": h} for h in measured]})
    serving = {"substrate": "slurm", "slurm_job_id": job, "baseline_pins": None, "served_cluster": "cluster_a", "shard": None, "attested_max_num_seqs": 31,
               "attested_max_model_len": 16384, "max_num_seqs": 31, "max_model_len": 16384, "node": "gpu-1", "measured_policy": policy,
               "server_default_policy": "contract_override", "override_generation_config": None}
    _w(os.path.join(tdir, "agentic", "cells.json"), {"cells": [{"cell_id": KEY_H, "virtual_model": f"petri__{KEY_H}", "serving_identity": LORA, "intervention_hash": "5923ecb7916846e9",
                                                                 "si_id": None, "kind": kind}], "serving": serving})
    resp, judged = _rows(KEY_H, rows_spec, judge_model=judge_model, cap=cap, at_cap_slots=at_cap_slots)
    cd = os.path.join(tdir, "agentic", KEY_H)
    _w(os.path.join(cd, "responses.jsonl"), resp, jsonl=True)
    _w(os.path.join(cd, "judged.jsonl"), judged, jsonl=True)
    meta = {"behaviors": list(ITEMS), "cell_id": KEY_H, "cell": {"cell_id": KEY_H, "intervention_hash": "5923ecb7916846e9", "kind": kind, "serving_identity": LORA}, "family": "agentic",
            "hostname": "gpu-1", "items": sum(len(v) for v in ITEMS.values()), "item_ids": [i for v in ITEMS.values() for i in v], "max_tokens": cap, "n_samples": N_SAMPLES,
            "sampler": "t1", "sampling": {"temperature": 1.0}, "temperature": 1.0, "slurm_job_id": job,
            "params": {"concurrency": 44, "max_tokens": cap, "sampler": "t1", "sampling": {"temperature": 1.0}, "temperature": 1.0},
            "present": len(resp), "generated_this_run": len(resp), "suite_revision": "agentic.v2@synthetic", "run": {"serving": serving},
            "endpoint_identity": {"base_model": "google/gemma-4-E4B-it", "engine": {"kind": engine_kind}, "server_default_policy": "contract_override", "thinking_mode": "off",
                                  "object_artifact_sha256_by_cell": {KEY_H: artifact_sha}}}
    _w(os.path.join(cd, "meta.json"), meta)
    if none_judged:
        nresp, njudged = _rows("none", rows_spec, judge_model=judge_model)
        _w(os.path.join(tdir, "agentic", "none", "responses.jsonl"), nresp, jsonl=True)
        _w(os.path.join(tdir, "agentic", "none", "judged.jsonl"), njudged, jsonl=True)
    n_blocked = sum(1 for r in judged if r["judge_blocked"])
    _w(os.path.join(tdir, "judge_provenance.json"), {"judge_model": judge_model, "tags": dict(EM.SEEDREP_JUDGE_TAGS), "thinking": False, "families": ["agentic"],
                                                     "suite_revision_v2": "agentic.v2@synthetic", "concurrency": 100, "pool_initial": "lowprio", "pool_final": "highprio",
                                                     "escalation": {"escalated": True}, "n_429": 0, "n_529": 189, "n_ok": len(judged) - n_blocked,
                                                     "counts": {"judged": len(judged), "blocked": n_blocked}, "started": "2026-09-22T06:50:00Z", "finished": "2026-09-22T07:20:00Z", "wall_s": 1800})
    with open(os.path.join(tdir, "RESULT"), "w", encoding="utf-8") as fh:
        fh.write(f"2026-09-22T06:39:00Z rc=1 agentic_rc=1 job={job} sha={tree[:12]} agentic_responses=0\n")
        if result_ok:
            fh.write(f"2026-09-22T06:45:00Z rc=0 agentic_rc=0 job={job} sha={tree[:12]} cluster=cluster_a agentic_responses={len(resp)} em8q_responses=4\n")
    adir = os.path.join(tdir, "analysis", target)
    os.makedirs(adir, exist_ok=True)
    pd.DataFrame(_summary(target, KEY_H, judged, LORA)).to_csv(os.path.join(adir, "summary.csv"), index=False)
    pr = pd.DataFrame(columns=PAIRED_COLS)
    if paired_rows:
        pr = pd.DataFrame([dict(zip(PAIRED_COLS, [target, KEY_H, "none", "agentic", "leaking", "harmful", 4, 0.5, 0.25, 0.25, 0.0, 0.5, 1, 0, 0.5, None, LORA, PASS, None, None, False]))])
    pr.to_csv(os.path.join(adir, "paired.csv"), index=False)
    pq = []
    for r in judged:
        for metric in ("harmful", "harmful_fuzzy", "at_cap", "response_chars", "zlib_ratio"):
            if metric in ("harmful", "harmful_fuzzy") and r["judge_blocked"]:
                continue
            pq.append(dict(run_id=f"misalign_{target}.{KEY_H}.0e63221d", arm_name=KEY_H, task=f"agentic/{r['behavior']}", item_id=r["item_id"], metric=metric, value=float(r["harmful"] or 0)))
    os.makedirs(os.path.join(tdir, "store"), exist_ok=True)
    pd.DataFrame(pq).to_parquet(os.path.join(tdir, "store", "rows.parquet"), index=False)
    os.makedirs(os.path.join(tdir, "em8q", KEY_H), exist_ok=True)
    os.makedirs(os.path.join(tdir, "serving"), exist_ok=True)
    if poisoned:
        _w(os.path.join(tdir, "agentic.POISONED-conc200", KEY_H, "judged.jsonl"), [{"item_id": "x", "sample_idx": 0, "behavior": "leaking"}], jsonl=True)
    return tdir


def _manifest(staging: str, *, sha: str = SHA, kind: str = EM.PAIRED_STAGING_KIND, cell_id: str = KEY_R, model: str = MODEL) -> str:
    return _w(os.path.join(staging, EM.PAIRED_STAGING_MANIFEST.format(model=model)),
              {"_schema": "relu_paired_manifest/1", "_written_at": "2026-09-22T06:20:00Z", "cluster": "cluster_a",
               "object": {"staged_id": STAGED, "bank": f"paired_d1_{model}_s1", "view": f"paired_d1_{model}_s1", "seed": 1, "cell_id": cell_id, "kind": kind, "object_sha256": sha,
                          "sidecar_object_sha256": sha}})


def _world(tmp_path, *, root_kw=None, with_manifest: bool = True, manifest_kw=None):
    figs, rs = str(tmp_path / "figs"), str(tmp_path / "rs")
    rsm = os.path.join(rs, "misalign")
    F.build_em_panel(figs)
    os.makedirs(rsm, exist_ok=True)
    _root(rsm, **(root_kw or {}))
    os.makedirs(os.path.join(rsm, "_scratch_never_read", "agentic"), exist_ok=True)
    staging = str(tmp_path / "staging")
    os.makedirs(staging, exist_ok=True)
    if with_manifest:
        _manifest(staging, **(manifest_kw or {}))
    roster = {"targets": {"qwen35_4b": {"views": {"s1": {"cells": {}}}}}}
    rp = str(tmp_path / "roster.yaml")
    with open(rp, "w", encoding="utf-8") as fh:
        yaml.safe_dump(roster, fh)
    return dict(figs=figs, rsm=rsm, staging=staging, grade=str(tmp_path / "grade"), roster=rp)


def _adapter(w, **kw) -> EM.EmMisalignAdapter:
    args = dict(root=w["figs"], full22_pins=None, baseline_pins=None, behaviour_source={}, grade_root=w["grade"], seedrep_root=w["rsm"], m16k_root=None, ladder_root=None,
                roster_path=w["roster"], staging_dir=w["staging"])
    args.update(kw)
    return EM.EmMisalignAdapter(**args)


def _groups(ad):
    groups = {}
    for i in ad.discover():
        groups.setdefault(ad.comparison_group(i), []).append(i)
    return groups


def _read(ad, group, conn=None, groups=None):
    groups = groups if groups is not None else _groups(ad)
    recs = []
    for batch in ad.read(_ctx(conn), group, groups.get(group, [])):
        recs.extend(batch.records)
    bad = [str(v) for v in (VAL.validate_record(r, path="<test>") for r in recs) if isinstance(v, VAL.Malformed)]
    assert not bad, bad[:4]
    out = {}
    for r in recs:
        out.setdefault(r["record_type"], []).append(r["payload"])
    return out, ad.last_facts[group], ad.last_report


def _run(recs, ev: str):
    return next(r for r in recs["run"] if r["eval_id"] == ev)


# ============================================================================================
# declarations, the grammar, discovery
# ============================================================================================

def test_the_declarations_the_version_and_the_grammar():
    # em_misalign 11 (2026-10-03): the anchor currency decides every contrast through the shared emitter (`extra.anchor_selection`)
    assert EM.EmMisalignAdapter.EMISSION_VERSION == DC.emission_version(12)      # 12: a row whose object the registry does not carry is parked by name (registry follow-ups-4 item 8)
    assert EM.PAIRED_SOURCE == "misalign_paired_allmodels" and EM.PAIRED_SOURCE in EM.EmMisalignAdapter.covers_sources
    assert EM.PAIRED_CAP == 3000 and EM.PAIRED_ANCHOR_STATE == "pooled_in_frame_anchors" and EM.PAIRED_STAGING_KIND == "psr_paired_write"
    assert EM.PAIRED_STAGING_DIR == str(SITE.expand_path("$STAGE/sdfbase/relu_paired")) and str(SITE.expand_path("$GRADE")) not in EM.PAIRED_STAGING_DIR
    assert EM.parse_paired_target(TARGET) == {"model": MODEL, "dose": 1.0, "dose_token": "d1", "seed": 1}
    assert EM.parse_paired_target("llama8b_paired_d0p5_llama8b_s2") == {"model": "llama8b", "dose": 0.5, "dose_token": "d0p5", "seed": 2}
    assert EM.parse_paired_target("nemotron3nano4b_paired_d1_nemotron3nano4b_s1")["model"] == "nemotron3nano4b"
    for bad in ("gemma4e4b_paired_d1_llama8b_s1", "gemma4e4b_paired_d1_gemma4e4b_s0", "gemma4e4b_paired_d1_gemma4e4b", "qwen35_4b_s1", "qwen35_4b_s1p", "qwen35_4b", "gemma4e4b_paired_gemma4e4b_s1"):
        assert EM.parse_paired_target(bad) is None, bad
    # the escalation-chain helpers every root shares
    assert EM.pool_chain_final("lowprio>highprio") == "highprio" and EM.pool_chain_final("lowprio") == "lowprio" and EM.pool_chain_final(None) is None
    assert EM.pool_chain_bad_elements(["lowprio>highprio", "highprio", "batch"]) == [] and sorted(EM.pool_chain_bad_elements(["lowprio>foo", "bar"])) == ["bar", "foo"]
    fact = EM.pool_chains_fact([{"judge_pool": "lowprio>highprio"}, {"judge_pool": "highprio"}, {"judge_pool": "lowprio>highprio"}])
    assert fact["judge_pool_chains"] == {"lowprio>highprio": 2, "highprio": 1} and fact["resolved_pools"] == {"highprio": 3} and fact["api_pool"] == "highprio"
    mixed = EM.pool_chains_fact([{"judge_pool": "lowprio"}, {"judge_pool": "lowprio>highprio"}])
    assert mixed["api_pool"] is None and mixed["resolved_pools"] == {"lowprio": 1, "highprio": 1}


def test_the_three_spellings_are_legacy_model_keys():
    with open(os.path.join(os.path.dirname(HERE), "configs", "registry", "legacy_spellings.yaml"), encoding="utf-8") as fh:
        rows = {r["legacy"]: r for r in yaml.safe_load(fh)["model_keys"]}
    assert rows[TARGET]["model_id"] == MODEL_ID and rows[TARGET]["fit_model_id"] == "gemma4e4b_base" and rows[TARGET]["transfer_kind"] == "base_to_instruct"
    assert rows["llama8b_paired_d1_llama8b_s1"]["model_id"] == "llama8b_it" and rows["llama8b_paired_d1_llama8b_s1"]["transfer_kind"] == "base_to_instruct"
    assert rows["nemotron3nano4b_paired_d1_nemotron3nano4b_s1"]["model_id"] == "nemotron3nano4b" == rows["nemotron3nano4b_paired_d1_nemotron3nano4b_s1"]["fit_model_id"]
    assert rows["nemotron3nano4b_paired_d1_nemotron3nano4b_s1"]["transfer_kind"] == "chat_checkpoint"
    for legacy in (TARGET, "llama8b_paired_d1_llama8b_s1", "nemotron3nano4b_paired_d1_nemotron3nano4b_s1"):
        assert EM.parse_paired_target(legacy)["model"] in DC.MODELS


def test_discovery_one_group_per_paired_root_skipped_by_the_seed_replicate_discovery_and_the_inert_copies_are_never_inputs(tmp_path):
    w = _world(tmp_path)
    ad = _adapter(w)
    groups = _groups(ad)
    paired = {g for g in groups if g.startswith(EM.PAIRED_GROUP_PREFIX)}
    assert paired == {GROUP} and not any(g.startswith(EM.SEEDREP_GROUP_PREFIX) and TARGET in g for g in groups), sorted(groups)
    assert ad.last_facts["<paired-discover>"]["targets"] == [TARGET]
    skipped = ad.last_facts["<seedrep-discover>"]["skipped"]
    assert TARGET in skipped and "misalign_paired_allmodels" in skipped[TARGET], skipped
    every = [i.uri for i in groups[GROUP]]
    assert not any("POISONED" in u or "/em8q/" in u or "/serving/" in u or "_scratch_never_read" in u for u in every), every
    for tail in ("cells_full.json", "judge_provenance.json", "RESULT", "summary.csv", "paired.csv", "store/rows.parquet", "agentic/cells.json", f"{KEY_H}/judged.jsonl",
                 f"{KEY_H}/responses.jsonl", f"{KEY_H}/meta.json"):
        assert any(u.endswith(tail) for u in every), tail
    assert ad.status_sources(GROUP) == (EM.PAIRED_SOURCE,)
    states = [r for r in ad.source_states(_ctx(), groups[GROUP]) if r.source_id == f"{ad.name}:{GROUP}"]
    assert states and all(TARGET in r.input_uri for r in states) and all(r.found == 1 for r in states), states
    assert "agentic.POISONED-conc200" in next(r for r in states if r.json_path == "$.entries" and r.input_uri.endswith(TARGET)).current_members
    assert next(r for r in states if r.json_path == "$.arm").current_members == [KEY_H]


# ============================================================================================
# the reading
# ============================================================================================

def test_the_root_files_two_runs_with_both_denominators_the_chains_the_terminal_reason_and_the_parked_frame_anchors(tmp_path):
    w = _world(tmp_path)
    recs, facts, rep = _read(_adapter(w), GROUP, conn=_conn())
    assert rep.runs == 2 and rep.contrasts == 0 and rep.contrasts_parked == 2 and rep.dispositions == 0 and rep.refusals == []
    assert facts["refused"] == [] and facts["tree_sha"] == TREE and facts["slurm_job_id"] == JOB and facts["model"] == MODEL
    lk, mu = _run(recs, "agentic_leaking"), _run(recs, "agentic_murder")
    for r in (lk, mu):
        assert r["lane_source_legacy"] == EM.PAIRED_SOURCE and r["campaign"] == "sdfbase_full22" and r["model_id"] == MODEL_ID and r["view_legacy"] == TARGET
        assert r["object_id"] == SHA and r["producer_object_key_legacy"] == STAGED and r["transfer_kind"] == "base_to_instruct" and r["campaign_cell_id_legacy"] == KEY_R
        assert r["producer_run_key_legacy"] == f"{EM.PAIRED_SOURCE}/{TARGET}/{PETRI_KEY}" and r["code_sha"] == TREE
        assert r["token_cap_tokens"] == 3000 and r["decode_source"] == "runner_forced_T1" and r["strength_coefficient"] == 1.0 and r["is_operating_point"] == 1
        ex = r["extra"]
        assert ex["terminal"] is True and ex["terminal_reason"] == EM.PAIRED_TERMINAL_REASON and ex["cap_independent"] is True and ex["n_at_cap"] == 0
        assert ex["decode_justification"] == EM.PAIRED_DECODE_JUSTIFICATION and ex["decode_record"]["justification"] == EM.PAIRED_DECODE_JUSTIFICATION
        assert ex["decode_record"]["recorded"] is True and ex["decode_record"]["sent"] == {"temperature": 1.0} and ex["decode_record"]["resolved"] is None
        assert ex["anchor_state"] == EM.PAIRED_ANCHOR_STATE and ex["in_root_none_measured"] is False and ex["object_resolution"]["resolved_by"] == "the connected registry's object row"
        assert ex["object_resolution"]["registry_row"]["paper_label"] == "difference_relu" and ex["object_resolution"]["staging_manifest"]["object_sha256"] == SHA
        assert ex["self_registration"]["refused"] is True and ex["rows_parquet_run_ids_legacy"] == [f"misalign_{TARGET}.{KEY_H}.0e63221d"]
        fc = r["field_confidence_json"]
        assert fc["decode_sent_json"]["confidence"] == "recorded" and fc["decode_source"]["confidence"] == "derived" and fc["decode_resolved_json"]["confidence"] == "unverified"
        assert r["decode_resolved_json"] == {}
    # leaking: 3 scored of 4 (one judge-blocked), harmful 3 -> 100 % over the scored rows, 75 % over the planned rows; the chains resolve to highprio
    m_lk = next(m for m in recs["run_metric"] if m["run_id"] == lk["run_id"])
    assert (m_lk["k"], m_lk["n"], m_lk["n_abstentions"]) == (3, 3, 1) and m_lk["value"] == pytest.approx(100.0)
    assert lk["extra"]["harmful_over_planned"] == {"k": 3, "n_planned": 4, "rate_pct": 75.0, "n_unscored": 1, "note": lk["extra"]["harmful_over_planned"]["note"]}
    assert lk["extra"]["lower_bound"] is True and lk["extra"]["judge_blocked"] == 1 and "LOWER BOUND" in lk["extra"]["note"] if "note" in lk["extra"] else True
    j_lk = next(j for j in recs["judgment"] if j["run_id"] == lk["run_id"])
    assert j_lk["api_pool"] == "highprio" and j_lk["extra"]["judge_pool_chains"] == {"highprio": 2, "lowprio>highprio": 2} and j_lk["extra"]["resolved_pools"] == {"highprio": 4}
    # murder: 4 scored, harmful 1
    m_mu = next(m for m in recs["run_metric"] if m["run_id"] == mu["run_id"])
    assert (m_mu["k"], m_mu["n"], m_mu["n_abstentions"]) == (1, 4, None) and m_mu["value"] == pytest.approx(25.0)
    j_mu = next(j for j in recs["judgment"] if j["run_id"] == mu["run_id"])
    assert j_mu["api_pool"] == "highprio" and j_mu["extra"]["judge_pool_chains"] == {"highprio": 4}
    # the instrument: cap 3,000 on [3000] read fixed, the runner's T1 as sent, the fixed label, never the seeded canonical id
    insts = {i["instrument_id"]: i for i in recs["instrument"]}
    inst = insts[lk["instrument_id"]]
    assert inst["token_cap_tokens"] == 3000 and inst["token_cap_ladder_json"] == [3000] and inst["read_at"] == "fixed" and inst["harness_version"] == TREE
    assert inst["decode_temperature"] == 1.0 and inst["decode_top_p"] is None and inst["decode_source"] == "runner_forced_T1"
    assert EM.SEEDREP_INSTRUMENT_LABEL.format(cap=3000) in json.dumps(inst)
    assert facts["instrument_ids"]["agentic_leaking"] and facts["instrument_ids"]["agentic_murder"]
    # the parked contrast names the frame's unsteered runs of THIS model (never another model's, never a run the frame does not flag)
    blk = m_lk["extra"]["delta_block"]
    assert blk["paired"] is False and blk["delta"] is None and blk["anchor"]["anchor_state"] == EM.PAIRED_ANCHOR_STATE and blk["anchor"]["object_id"] == "none"
    assert blk["anchor"]["in_frame_unsteered_run_ids"] == ["run:none:leaking:16k", "run:none:leaking:3k"] and blk["anchor"]["state"] == "derived"
    assert [a["rate_pct"] for a in blk["anchor"]["in_frame_unsteered"]] == pytest.approx([100.0 * 101 / 510, 100.0 * 99 / 510])
    assert blk["anchor"]["lane_policy_pairing_anchor"] == f"exp/misalign/{MODEL}/agentic/none"
    assert m_mu["extra"]["delta_block"]["anchor"]["in_frame_unsteered_run_ids"] == ["run:none:murder:16k"]
    assert lk["extra"]["in_frame_unsteered_reference"]["n_candidates"] == 3
    # 8 judged rows -> 8 x 3 item rows
    assert len(recs["item_result"]) == 24 and sum(1 for i in recs["item_result"] if i["label"] == "judge_blocked") == 3
    assert facts["readings"][f"{KEY_H}/leaking"]["rate_over_planned_pct"] == 75.0 and facts["readings"][f"{KEY_H}/murder"]["api_pool"] == "highprio"


def test_without_a_registry_connection_the_staging_manifest_resolves_the_object_and_the_frame_anchors_are_unresolved(tmp_path):
    w = _world(tmp_path)
    recs, facts, rep = _read(_adapter(w), GROUP, conn=None)
    assert rep.runs == 2 and facts["refused"] == []
    ex = _run(recs, "agentic_leaking")["extra"]
    assert ex["object_resolution"]["resolved_by"] == "ReLU's staging manifest (no registry connection)" and ex["object_resolution"]["registry_row"] is None
    assert ex["object_resolution"]["staged_id"] == STAGED and _run(recs, "agentic_leaking")["producer_object_key_legacy"] == STAGED
    ref = ex["in_frame_unsteered_reference"]
    assert ref["state"] == "unresolved" and "no registry connection" in ref["why"] and ref["in_frame_unsteered_run_ids"] == []


def test_a_root_with_at_cap_rows_is_not_terminal_and_a_mixed_pool_judgment_files_api_pool_null(tmp_path):
    rows = {"leaking": dict(h=[1, 0, 1, 1], blocked=[], pools=["lowprio", "lowprio>highprio", "lowprio", "highprio"]),
            "murder": dict(h=[0, 1, 0, 0], blocked=[], pools=["highprio"] * 4)}
    w = _world(tmp_path, root_kw=dict(rows_spec=rows, at_cap_slots=(("leaking", 1),)))
    recs, facts, rep = _read(_adapter(w), GROUP, conn=_conn())
    assert rep.runs == 2 and facts["refused"] == []
    lk = _run(recs, "agentic_leaking")
    assert lk["extra"]["terminal"] is False and lk["extra"]["terminal_reason"] is None and lk["extra"]["n_at_cap"] == 1 and lk["extra"]["comparability_16k"] is None
    deg = next(d for d in recs["run_degeneration"] if d["run_id"] == lk["run_id"] and d["measure"] == "at_token_cap")
    assert deg["numerator"] == 1
    j = next(j for j in recs["judgment"] if j["run_id"] == lk["run_id"])
    assert j["api_pool"] is None and j["extra"]["resolved_pools"] == {"lowprio": 2, "highprio": 2} and j["extra"]["judge_pool_chains"] == {"lowprio": 2, "lowprio>highprio": 1, "highprio": 1}
    assert next(j for j in recs["judgment"] if j["run_id"] == _run(recs, "agentic_murder")["run_id"])["api_pool"] == "highprio"


# ============================================================================================
# refusals by name
# ============================================================================================

def test_an_unknown_pool_element_refuses_the_key_and_the_seed_replicate_reader_shares_the_rule(tmp_path):
    rows = {"leaking": dict(h=[1, 0, 1, 1], blocked=[], pools=["lowprio>frontdoor", "highprio", "highprio", "highprio"]), "murder": dict(h=[0, 1, 0, 0], blocked=[], pools=["highprio"] * 4)}
    w = _world(tmp_path, root_kw=dict(rows_spec=rows))
    recs, facts, rep = _read(_adapter(w), GROUP, conn=_conn())
    assert rep.runs == 1 and rep.dispositions == 1
    d = recs["disposition"][0]
    assert d["state"] == "held" and d["eval_id"] == "agentic_leaking" and "not API pools" in d["reason"] and "frontdoor" in d["reason"] and "lowprio>frontdoor" in d["reason"]
    assert any("frontdoor" in x for x in facts["refused"])
    # a chain of API pools is never that refusal (the version-7 defect that held the bl merged leaking key and the s2 murder key)
    assert EM.pool_chain_bad_elements({"highprio", "lowprio>highprio"}) == [] and EM.pool_chain_bad_elements({"lowprio", "lowprio>highprio"}) == []


@pytest.mark.parametrize("case", ["kind", "label", "absent_row", "manifest_sha", "artifact_sha", "source_sha", "no_authority"])
def test_the_object_is_refused_by_name_unless_it_is_the_models_composed_difference_relu_object(tmp_path, case):
    root_kw, manifest_kw, with_manifest, conn = {}, {}, True, _conn()
    if case == "kind":
        conn = _conn(object_row=("relu_steer", "difference_relu", "gemma4e4b_base", "approved", STAGED, "fit"))
    elif case == "label":
        conn = _conn(object_row=("composed", "control_relu", "gemma4e4b_base", "approved", STAGED, "composition"))
    elif case == "absent_row":
        conn = _conn(object_row=None)
    elif case == "manifest_sha":
        manifest_kw = dict(sha=OTHER_SHA)
    elif case == "artifact_sha":
        root_kw = dict(artifact_sha=OTHER_SHA)
    elif case == "source_sha":
        root_kw = dict(source_sha=OTHER_SHA)
    elif case == "no_authority":
        conn, with_manifest = None, False
    w = _world(tmp_path, root_kw=root_kw, with_manifest=with_manifest, manifest_kw=manifest_kw)
    recs, facts, rep = _read(_adapter(w), GROUP, conn=conn)
    assert rep.runs == 0 and rep.dispositions == 2 and len(facts["refused"]) == 2, (case, facts["refused"])
    reason = recs["disposition"][0]["reason"]
    if case == "kind":
        assert "kind 'relu_steer'" in reason and "not the composed difference_relu object" in reason
    elif case == "label":
        assert "label 'control_relu'" in reason
    elif case == "absent_row":
        assert "not in the connected registry's object table" in reason
    elif case == "manifest_sha":
        assert "staging manifest" in reason and OTHER_SHA[:16] in reason
    elif case == "artifact_sha":
        assert "object_artifact_sha256_by_cell" in reason
    elif case == "source_sha":
        assert "staged_from.source names object" in reason
    else:
        assert "resolves to neither" in reason and "no staging manifest" in reason
    assert "run" not in recs


@pytest.mark.parametrize("case", ["measure_none", "none_judged", "paired_rows", "view", "result", "engine_keys_present"])
def test_the_root_refuses_whole_by_name(tmp_path, case):
    kw = {"measure_none": dict(measure_none=True), "none_judged": dict(none_judged=True), "paired_rows": dict(paired_rows=True), "view": dict(view="paired_d1_gemma4e4b_s2"),
          "result": dict(result_ok=False), "engine_keys_present": {}}[case]
    w = _world(tmp_path, root_kw=kw)
    if case == "engine_keys_present":
        os.makedirs(os.path.join(w["rsm"], TARGET, "agentic", "si_constv2_mix_full__lora_r4_ce"), exist_ok=True)
    ad = _adapter(w)
    with pytest.raises(AdapterRefusal) as ei:
        _read(ad, GROUP, conn=_conn())
    msg = str(ei.value)
    expect = {"measure_none": "measure_none is True", "none_judged": "unsteered key was judged", "paired_rows": "paired.csv carries 1 row", "view": "wave_view",
              "result": "RESULT carries no line with rc=0", "engine_keys_present": "not the policy's measured keys"}[case]
    assert expect in msg, msg


def test_the_instrument_guard_refuses_a_vendor_shaped_or_laddered_instrument(tmp_path):
    w = _world(tmp_path)
    ad = _adapter(w)
    batch = RecordBatch("x", GROUP)
    batch.add("instrument", {"instrument_id": "i1", "eval_id": "agentic_leaking", "decode_source": "runner_forced_T1", "decode_temperature": 1.0, "decode_seed_rule": "none",
                             "token_cap_tokens": 3000, "token_cap_ladder_json": [3000, 16384], "read_at": "ladder_final"}, dict(PRODUCER), NOW)
    with pytest.raises(AdapterRefusal) as ei:
        ad._paired_guard_instruments(batch, {"group": GROUP, "model": MODEL, "instrument_ids": {}})
    assert "not 3,000 on [3000] read fixed" in str(ei.value)
    batch2 = RecordBatch("y", GROUP)
    batch2.add("instrument", {"instrument_id": "i2", "eval_id": "agentic_leaking", "decode_source": "vendor_recommendation", "decode_temperature": 1.0, "decode_seed_rule": "none",
                              "decode_top_p": 0.95, "token_cap_tokens": 3000, "token_cap_ladder_json": [3000], "read_at": "fixed"}, dict(PRODUCER), NOW)
    with pytest.raises(AdapterRefusal) as ei2:
        ad._paired_guard_instruments(batch2, {"group": GROUP, "model": MODEL, "instrument_ids": {}})
    assert "not the runner's T1 as sent" in str(ei2.value) and "vendor-shaped" in str(ei2.value)


def test_a_none_root_of_the_seed_replicate_shape_under_the_same_mirror_is_not_a_paired_group(tmp_path):
    """The seed roots and the paired roots share one mirror directory: a name outside the paired grammar is never a paired group, and a paired name
    is never a seed-replicate group (each reader's discovery owns its grammar)."""
    w = _world(tmp_path)
    ad = _adapter(w)
    assert ad.comparison_group(next(i for i in ad.discover() if TARGET in i.uri)) == GROUP
    assert ad.status_sources(f"{EM.SEEDREP_GROUP_PREFIX}qwen35_4b_s1") == (EM.SEEDREP_SOURCE,)
    with pytest.raises(AdapterRefusal) as ei:
        ad.paired_rows("qwen35_4b_s1")
    assert "outside the grammar" in str(ei.value)
    with pytest.raises(AdapterRefusal) as ei2:
        ad.paired_rows("llama8b_paired_d1_llama8b_s1")
    assert "no judged store" in str(ei2.value)


# ============================================================================================
# the in-frame readings follow the paper frame: a withdrawn reading is absent, on the grading host as on a shard
# ============================================================================================

def test_a_withdrawn_unsteered_reading_is_absent_from_the_parked_block_and_a_twin_never_supplies_it(tmp_path):
    """The reading the block carries for an in-frame unsteered run is the one the paper frame counts (the anchor loader's withdrawn-observation
    rule, `anchor_current.withdrawn_metric_cte`): a revision that withdrew the `misaligned_rate` row of a run that stays current and in the paper
    leaves the run NAMED with k / n / rate_pct null -- never the withdrawn value, which the table still holds -- a withdrawn computed twin beside
    a counted producer row never supplies the reading, and a kept twin loses to the producer's row by the primary-key order."""
    w = _world(tmp_path)
    conn = _conn(observed=True, withdrawn=("run:none:leaking:3k",), twins={"run:none:leaking:16k": 0, "run:none:murder:16k": 1})
    recs, facts, rep = _read(_adapter(w), GROUP, conn=conn)
    assert rep.runs == 2 and rep.contrasts_parked == 2 and rep.refusals == []
    lk, mu = _run(recs, "agentic_leaking"), _run(recs, "agentic_murder")
    blk = next(m for m in recs["run_metric"] if m["run_id"] == lk["run_id"])["extra"]["delta_block"]["anchor"]
    assert blk["state"] == "derived" and blk["in_frame_unsteered_run_ids"] == ["run:none:leaking:16k", "run:none:leaking:3k"]
    by = {a["run_id"]: a for a in blk["in_frame_unsteered"]}
    assert (by["run:none:leaking:3k"]["k"], by["run:none:leaking:3k"]["n"], by["run:none:leaking:3k"]["rate_pct"]) == (None, None, None)
    assert by["run:none:leaking:3k"]["selection_state"] == "selected"
    assert (by["run:none:leaking:16k"]["k"], by["run:none:leaking:16k"]["n"]) == (101, 510) and by["run:none:leaking:16k"]["rate_pct"] == pytest.approx(100.0 * 101 / 510)
    mblk = next(m for m in recs["run_metric"] if m["run_id"] == mu["run_id"])["extra"]["delta_block"]["anchor"]
    assert [(a["run_id"], a["k"], a["n"]) for a in mblk["in_frame_unsteered"]] == [("run:none:murder:16k", 58, 490)]
    assert lk["extra"]["in_frame_unsteered_reference"]["anchors"][1]["rate_pct"] is None
    assert conn.execute("SELECT COUNT(*) FROM run_metric WHERE run_id = 'run:none:leaking:3k'").fetchone()[0] == 1       # the predicate, not the table
    assert facts["in_frame_anchors"]["agentic_leaking"]["anchors"] == ["run:none:leaking:16k", "run:none:leaking:3k"]
    # the same connection without an observation table reads every row as a reading: nothing can be withdrawn where nothing was observed
    raw = EM.EmMisalignAdapter._paired_frame_readings(_conn(twins={"run:none:murder:16k": 1}), ["run:none:leaking:3k", "run:none:murder:16k", "run:none:absent"])
    assert raw == {"run:none:leaking:3k": (99, 510, pytest.approx(100.0 * 99 / 510)), "run:none:murder:16k": (58, 490, pytest.approx(100.0 * 58 / 490))}
    assert EM.EmMisalignAdapter._paired_frame_readings(conn, []) == {}


# a real registry (the fold, `observation`, `v_run_current`): one roster-matched unsteered agentic run of the synthetic model
AG_EVAL, AG_ITEMS = "agentic_leaking", "agentic_leaking_items"
AG_INST = synth.instrument_fields(eval_id=AG_EVAL, item_set_id=AG_ITEMS, harness="agentic_misalignment", token_cap_tokens=3000, token_cap_ladder_json="[3000]")
AG_INST_ID = str(AG_INST["instrument_id"])
WORLD_SRC, WORLD_ADAPTER = "src:agentic_world", "synthetic"


def _world_rev(rev: str, i: int):
    return synth.rec("source_revision", {"revision_id": rev, "source_id": WORLD_SRC, "adapter": WORLD_ADAPTER, "observed_utc": synth.t(i),
                                         "inputs_json": {"a": "%064x" % i}, "accepted": 1}, synth.t(i))


def _world_obs(rev: str, keys, i: int, table: str = "run"):
    return synth.rec("observation", {"source_id": WORLD_SRC, "revision_id": rev, "table_name": table,
                                     "observation_keys": [I.canonical_json(k) for k in keys]}, synth.t(i))


def _agentic_run(run_key: str, i0: int, object_key: str, object_id: str, k: int, n: int):
    """A judged agentic run of the synthetic model with ONE `misaligned_rate` reading (the producer's row; threshold the sentinel)."""
    rid = I.run_id(synth.CAMPAIGN, run_key, object_key, AG_EVAL, AG_INST_ID)
    rid, recs = synth.run_records(run_key, i0, object_key, object_id, n_items=n, run_id=rid, eval_id=AG_EVAL, item_set_id=AG_ITEMS,
                                  instrument_id=AG_INST_ID, harness="agentic_misalignment", token_cap_tokens=3000)
    jid = I.judgment_id(rid, synth.JUDGE_ID)
    recs = [r for r in recs if r["record_type"] != "run_metric"]
    recs.append(synth.rec("run_metric", {"run_id": rid, "judgment_id": jid, "metric_id": "misaligned_rate", "value": 100.0 * k / n, "k": k, "n": n,
                                         "computed_by": "producer"}, synth.t(i0 + 9)))
    return rid, jid, recs


def _agentic_registry(root: str) -> dict:
    """The world: the agentic eval, its item set, an instrument and a roster entry selecting the unsteered run (so the fold flags it
    `in_current_paper`); the unsteered run with its reading and a steered run with a contrast at `misaligned_rate` (so the sidecar ships the
    metric); revision rev1 of a synthetic source observing the unsteered run's key and its reading."""
    H.ensure_layout(root)
    dims = synth.seed_dimensions(0) + [
        synth.rec("eval", {"eval_id": AG_EVAL, "eval_category": "agentic", "headline_metric_id": "misaligned_rate", "metric_direction": "lower_is_better",
                           "attacker_required": 0, "paper_status": "in_paper"}, synth.t(12)),
        synth.rec("item_set", {"item_set_id": AG_ITEMS, "eval_id": AG_EVAL, "dataset_version": "synthetic/agentic@v1", "n_items": 5, "items_sha256": synth.SHA256_B,
                               "source_uri": "synthetic://items/agentic.jsonl", "item_id_scheme": "q<index>", "subsample_rule": "full"}, synth.t(13)),
        synth.rec("instrument", AG_INST, synth.t(14)),
        synth.rec("roster_entry", {"roster_entry_id": "re_ag_none", "round_id": "r_synth", "comparison_id": "fig", "model_id": synth.MODEL_ID, "eval_id": AG_EVAL,
                                   "instrument_id": AG_INST_ID, "judge_id": synth.JUDGE_ID, "object_selector_json": {"paper_label": "none"},
                                   "strength_rule": "operating_point", "source_citation": "synthetic", "inferred": 0}, synth.t(51)),
    ]
    R.write_records(os.path.join(root, "records", "dimensions", "seed", "000001-dimensions.jsonl"), dims)
    none, j_none, recs_none = _agentic_run("ag_none", 100, "none", "none", 1, 5)
    lora, j_lora, recs_lora = _agentic_run("ag_lora", 140, "constitution_lora@deadbeef", synth.OBJECT_SHA, 3, 5)
    for rid, recs in ((none, recs_none), (lora, recs_lora)):
        synth.write_run_dir(root, synth.CAMPAIGN, rid, [(str(r["record_type"]), [r]) for r in recs], gz_items=False)
    k_rate = {"run_id": none, "judgment_id": j_none, "metric_id": "misaligned_rate", "threshold": -1.0, "computation_id": ""}
    contrast = synth.rec("run_contrast", {"run_id": lora, "anchor_run_id": none, "judgment_id": j_lora, "metric_id": "misaligned_rate", "threshold": -1.0,
                                          "anchor_kind": "same_standup_none", "delta": 40.0, "delta_method": "exact McNemar", "n_pairs": 5,
                                          "computed_by": "registry"}, synth.t(8992))
    R.write_records(os.path.join(root, "records", "a", "000001-rev1.jsonl"),
                    [_world_rev("rev1", 9000), _world_obs("rev1", [{"run_id": none}], 9001), _world_obs("rev1", [k_rate], 9002, table="run_metric"), contrast])
    _ingest_world(root)
    return {"none": none, "lora": lora, "k_rate": k_rate, "j_none": j_none}


def _rejudge(root: str, w: dict, *, replacement=None, i: int = 700) -> str:
    """The unsteered run re-judged under the SAME judge (the roster entry's, so the run stays selected): its judgment re-filed `superseded`
    naming a new complete judgment as its successor; `replacement=(k, n)` also files the successor's `misaligned_rate` row (no revision
    observes it: a never-observed row is a reading). Returns the successor's id -- the same id each call, so a later call files under it."""
    j_new = I.judgment_id(w["none"], synth.JUDGE_ID, "s", synth.t(600))
    recs = [synth.rec("judgment", {"judgment_id": w["j_none"], "run_id": w["none"], "judge_id": synth.JUDGE_ID, "n_items_judged": 5, "status": "superseded",
                                   "superseded_by_judgment_id": j_new}, synth.t(i)),
            # the successor states the form its id was minted in (main's identity rule: `claimed_by_session` + `claimed_utc` reproduce it)
            synth.rec("judgment", {"judgment_id": j_new, "run_id": w["none"], "judge_id": synth.JUDGE_ID, "n_items_judged": 5, "status": "complete",
                                   "claimed_by_session": "s", "claimed_utc": synth.t(600),
                                   "started_utc": synth.t(i - 100), "finished_utc": synth.t(i - 99), "api_pool": "lowprio"}, synth.t(i + 1))]
    if replacement is not None:
        k, n = replacement
        recs.append(synth.rec("run_metric", {"run_id": w["none"], "judgment_id": j_new, "metric_id": "misaligned_rate", "value": 100.0 * k / n, "k": k, "n": n,
                                             "computed_by": "producer"}, synth.t(i + 2)))
    R.write_records(os.path.join(root, "records", "c", f"{i:06d}-rejudge.jsonl"), recs)
    _ingest_world(root)
    return j_new


def _withdraw_reading(root: str, w: dict) -> None:
    """rev2: the unsteered run's key stays observed (the run stays current); its reading is withdrawn."""
    R.write_records(os.path.join(root, "records", "a", "000002-rev2.jsonl"),
                    [_world_rev("rev2", 9010), _world_obs("rev2", [{"run_id": w["none"]}], 9011),
                     synth.rec("withdrawal", {"source_id": WORLD_SRC, "observation_key": I.canonical_json(w["k_rate"]), "from_revision_id": "rev1",
                                              "to_revision_id": "rev2", "table_name": "run_metric", "reason": "removed_measurement"}, synth.t(9012))])
    _ingest_world(root)


def _ingest_world(root: str) -> None:
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, (rep.refused, rep.blocked, rep.refusals)


def test_on_a_withdrawn_unsteered_reading_the_grading_host_read_and_the_shard_read_agree_and_the_anchor_is_absent_on_both(tmp_path):
    """Sidecar parity for the paired reader: on the live connection (the grading host) and on a `reader_state` sidecar (a shard) the in-frame
    anchors of the synthetic model are the same run with the same reading; after a revision withdraws that reading while the run stays
    current and in the paper, BOTH name the run with k / n / rate_pct null -- the live table still holds the row and the sidecar does
    not, so only the frame's predicate applied on the grading host keeps the two reads identical (the anchor is absent, never numeric)."""
    root = str(tmp_path / "reg")
    w = _agentic_registry(root)
    world = _world(tmp_path / "paired")

    def reads(tag: str):
        with DB.reading(DB.db_path(root)) as live:
            vrc = tuple(live.execute("SELECT is_current, in_current_paper, selection_state FROM v_run_current WHERE run_id = ?", (w["none"],)).fetchone())
            held = live.execute("SELECT COUNT(*) FROM run_metric WHERE run_id = ? AND metric_id = 'misaligned_rate'", (w["none"],)).fetchone()[0]
            grading_host = _adapter(world)._paired_in_frame_anchors(live, synth.MODEL_ID, NOW)
        out = str(tmp_path / f"rs_{tag}.sqlite")
        rep = RS.export(root, ["em_misalign"], out, uri_home=str(tmp_path / "home"))
        conn = RS.open_reader_state(out, ["em_misalign"])
        try:
            shipped = conn.execute("SELECT COUNT(*) FROM run_metric WHERE run_id = ?", (w["none"],)).fetchone()[0]
            shard = _adapter(world)._paired_in_frame_anchors(conn, synth.MODEL_ID, NOW)
        finally:
            conn.close()
        return vrc, held, rep, shipped, grading_host, shard

    vrc, held, rep, shipped, grading_host, shard = reads("rev1")
    assert vrc == (1, 1, "selected") and held == 1 and shipped == 1 and rep.withdrawn_metric_rows == 0 and rep.non_current_judgment_metric_rows == 0
    assert grading_host == shard and set(grading_host) == {f"agentic_{b}" for b in EM.SEEDREP_BEHAVIOURS}
    assert grading_host[AG_EVAL]["state"] == "derived" and grading_host[AG_EVAL]["why"] is None and grading_host[AG_EVAL]["in_frame_unsteered_run_ids"] == [w["none"]]
    (a,) = grading_host[AG_EVAL]["anchors"]
    assert (a["k"], a["n"], a["rate_pct"], a["selection_state"], a["token_cap_tokens"]) == (1, 5, 20.0, "selected", 3000)
    assert grading_host["agentic_murder"]["state"] == "derived" and grading_host["agentic_murder"]["anchors"] == [] and grading_host["agentic_murder"]["n_candidates"] == 0
    _withdraw_reading(root, w)
    vrc, held, rep, shipped, grading_host, shard = reads("rev2")
    assert vrc == (1, 1, "selected") and held == 1 and shipped == 0 and rep.withdrawn_metric_rows == 1
    assert grading_host == shard and grading_host[AG_EVAL]["state"] == "derived" and grading_host[AG_EVAL]["in_frame_unsteered_run_ids"] == [w["none"]]
    (a,) = grading_host[AG_EVAL]["anchors"]
    assert (a["k"], a["n"], a["rate_pct"]) == (None, None, None) and a["selection_state"] == "selected" and a["run_id"] == w["none"]


def test_a_rejudged_unsteered_run_contributes_only_its_current_judgments_reading(tmp_path):
    """The judgment-currency rule (`schema_def.non_current_judgment_sql`) in the paired reader: a selected unsteered run whose judgment a
    re-judge displaced is NAMED with k / n / rate_pct null when the successor filed no `misaligned_rate` row -- never the displaced judgment's
    number, which the table still holds -- and with the successor's row when it did, although the displaced judgment's row sorts first by key;
    a run whose only judgment a claim released (`abandoned`) is named with nulls too."""
    w = _world(tmp_path)
    conn = _conn(rejudged={"run:none:leaking:3k": None, "run:none:leaking:16k": (204, 510)}, abandoned=("run:none:murder:16k",))
    recs, facts, rep = _read(_adapter(w), GROUP, conn=conn)
    assert rep.runs == 2 and rep.contrasts_parked == 2 and rep.refusals == []
    lk, mu = _run(recs, "agentic_leaking"), _run(recs, "agentic_murder")
    blk = next(m for m in recs["run_metric"] if m["run_id"] == lk["run_id"])["extra"]["delta_block"]["anchor"]
    assert blk["state"] == "derived" and blk["in_frame_unsteered_run_ids"] == ["run:none:leaking:16k", "run:none:leaking:3k"]
    by = {a["run_id"]: a for a in blk["in_frame_unsteered"]}
    assert (by["run:none:leaking:3k"]["k"], by["run:none:leaking:3k"]["n"], by["run:none:leaking:3k"]["rate_pct"]) == (None, None, None)
    assert (by["run:none:leaking:16k"]["k"], by["run:none:leaking:16k"]["n"]) == (204, 510) and by["run:none:leaking:16k"]["rate_pct"] == pytest.approx(40.0)
    mblk = next(m for m in recs["run_metric"] if m["run_id"] == mu["run_id"])["extra"]["delta_block"]["anchor"]
    assert [(a["run_id"], a["k"], a["n"], a["rate_pct"]) for a in mblk["in_frame_unsteered"]] == [("run:none:murder:16k", None, None, None)]
    assert conn.execute("SELECT COUNT(*) FROM run_metric WHERE run_id = 'run:none:leaking:3k'").fetchone()[0] == 1            # the rule, not the table
    assert [tuple(r) for r in conn.execute("SELECT judgment_id, k FROM run_metric WHERE run_id = 'run:none:leaking:16k' ORDER BY judgment_id")] == \
        [("j2:run:none:leaking:16k", 204), ("j:run:none:leaking:16k", 101)]
    # the predicate text is schema_def's (one rule, three readers: the loader, the sidecar export, this reader)
    assert AC.CURRENT_JUDGMENT_SQL == EM.AC.CURRENT_JUDGMENT_SQL == S.non_current_judgment_sql("rm.judgment_id", "jm.status")
    assert all(f"'{s}'" in AC.CURRENT_JUDGMENT_SQL for s in S.NON_CURRENT_JUDGMENT_STATUSES)


def test_on_a_rejudged_selected_baseline_the_grading_host_read_and_the_shard_read_agree_numeric_then_null_then_the_successors_row(tmp_path):
    """Sidecar parity for the re-judge: the agentic registry's selected unsteered run (its reading counted) is re-judged under the same judge
    -- the run stays selected and in the paper, the displaced judgment's row stays in `run_metric`, and `v_paper_long` counts nothing for it
    -- so the grading host read and the shard read both name the run with nulls (the export ships no row and reports the dropped one); once the
    successor files its `misaligned_rate` row, both carry the successor's number, not the displaced judgment's, which sorts first by key."""
    root = str(tmp_path / "reg")
    w = _agentic_registry(root)
    world = _world(tmp_path / "paired")

    def reads(tag: str):
        with DB.reading(DB.db_path(root)) as live:
            vrc = tuple(live.execute("SELECT is_current, in_current_paper, selection_state FROM v_run_current WHERE run_id = ?", (w["none"],)).fetchone())
            held = [tuple(r) for r in live.execute("SELECT judgment_id, k FROM run_metric WHERE run_id = ? AND metric_id = 'misaligned_rate' ORDER BY judgment_id", (w["none"],))]
            frame = [tuple(r) for r in live.execute("SELECT judgment_id, metric_id FROM v_paper_long WHERE run_id = ? AND in_current_paper = 1", (w["none"],))]
            grading_host = _adapter(world)._paired_in_frame_anchors(live, synth.MODEL_ID, NOW)
        out = str(tmp_path / f"rs_{tag}.sqlite")
        rep = RS.export(root, ["em_misalign"], out, uri_home=str(tmp_path / "home"))
        conn = RS.open_reader_state(out, ["em_misalign"])
        try:
            shipped = [tuple(r) for r in conn.execute("SELECT judgment_id, k FROM run_metric WHERE run_id = ?", (w["none"],))]
            shard = _adapter(world)._paired_in_frame_anchors(conn, synth.MODEL_ID, NOW)
        finally:
            conn.close()
        assert grading_host == shard and vrc == (1, 1, "selected") and grading_host[AG_EVAL]["in_frame_unsteered_run_ids"] == [w["none"]]
        (a,) = grading_host[AG_EVAL]["anchors"]
        return held, frame, rep, shipped, (a["k"], a["n"], a["rate_pct"])

    held, frame, rep, shipped, reading = reads("judged")
    assert held == [(w["j_none"], 1)] and frame == [(w["j_none"], "misaligned_rate")] and shipped == held and reading == (1, 5, 20.0)
    assert (rep.withdrawn_metric_rows, rep.non_current_judgment_metric_rows) == (0, 0)
    j_new = _rejudge(root, w)
    held, frame, rep, shipped, reading = reads("rejudged")
    assert held == [(w["j_none"], 1)] and frame == [] and shipped == [] and reading == (None, None, None)
    assert (rep.withdrawn_metric_rows, rep.non_current_judgment_metric_rows) == (0, 1) and rep.as_dict()["non_current_judgment_metric_rows"] == 1
    assert _rejudge(root, w, replacement=(2, 5), i=800) == j_new                                   # the same successor id: the row is filed under it
    held, frame, rep, shipped, reading = reads("successor-row")
    assert sorted(held) == sorted([(w["j_none"], 1), (j_new, 2)]) and frame == [(j_new, "misaligned_rate")] and shipped == [(j_new, 2)]
    assert reading == (2, 5, 40.0) and (rep.withdrawn_metric_rows, rep.non_current_judgment_metric_rows) == (0, 1)
