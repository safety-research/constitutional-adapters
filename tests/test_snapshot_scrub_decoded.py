"""The paper frame's export-only scrub, checked on the DECODED export.

Absolute paths, a hostname and a user token are planted in every URI / location column and JSON path the live schema has
(``run.store_uri`` / ``store_completion_artifact_uri`` / ``config_path`` / ``notes`` / ``extra_json``, ``object.artifact_locations_json``
/ ``object_sidecars_json`` / ``fit_config_path`` / ``fit_heldout_curve_uri`` / ``fit_wandb_url`` / ``extra_json``,
``fit_search.trials_uri`` / ``grad_norm_reconnaissance_uri`` / ``wandb_sweep_url``, ``judgment.store_uri``, ``run_metric.source_uri``,
``run_degeneration.source_uri``, ``run_state_event.evidence_uri``, ``item_flags.extra_json`` (the live schema has no
``*_artifact_uri`` column on item_flags: the artifact ids are sha256s), ``ruling.*_json`` guards' ``artifact_uri``, ``si_text.source_uri``
with a literal ``$HOME``, ``model.extra_json``), the frame is exported, and EVERY exported table is decoded both ways (Parquet via
pyarrow, SQLite via sqlite3), every string and every JSON cell walked recursively: no ``/home/``, ``/workspace``, ``/vol/``,
``/tmp/``, ``~/``, ``$HOME``, the hostname or the user token survives; a location with a known hash reads ``sha256:<hex>``;
a path under this repo reads ``repo:<rel>``; ``MANIFEST.scrubbed`` names every column whose value changed.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3

import pytest

from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import records as R
from sisteer.registry import schema_def as S
from sisteer.registry import snapshot as SN
from sisteer.registry import synth

HOST = synth.TESTHOST          # "TESTHOST": synth stamps it as node_hostname on every running run
USER = "TESTUSER"
TOKENS = (HOST, USER)
LEAK = re.compile(r"/home/|/workspace|/scratch/|/vol/|/tmp/|~/|\$HOME|" + HOST + "|" + USER)
N_ITEMS = 4
CONFIG_PATH = f"/home/{USER}/si-steering/configs/sdfbase_x.yaml"                     # under a checkout of this repo → repo:configs/...
EVAL_REPO_CONFIG = f"/scratch/{USER}/si-steering-eval-repo/{'a' * 40}/configs/fits.yaml"


def _model(mid, family, kind, i):
    return synth.rec("model", {"model_id": mid, "family": family, "size_label": "4b", "checkpoint_kind": kind, "hf_repo": "org/x",
                               "hf_revision": "d" * 40, "chat_template_sha256": synth.SHA256_A, "thinking_channel": "available",
                               "paper_status": "in_paper", "bank_presence": f"NOT in ~/stage/fits on {HOST}"}, synth.t(i))


def _leaky_object():
    o = synth.object_lora()
    o.update({
        "artifact_locations_json": [{"uri": f"/home/{USER}/objects/x/object.pt", "bytes": 1234, "verified_utc": synth.t(3)},
                                    {"uri": f"cluster_a:/scratch/{USER}/exp/objects/x", "bytes": None, "source": "_records/x.json"}],
        "object_sidecars_json": [{"kind": "train_budget_meta.json", "sha256": synth.SHA256_A, "uri": f"/home/{USER}/objects/x/train_budget_meta.json"},
                                 {"kind": "COMPLETE", "sha256": synth.SHA256_B, "uri": f"/home/{USER}/objects/x/COMPLETE"}],
        "fit_config_path": EVAL_REPO_CONFIG, "fit_config_sha256": synth.SHA256_B,
        "fit_heldout_curve_uri": f"/home/{USER}/curves/x.json", "fit_wandb_url": f"https://wandb.example/{USER}/runs/1",
        "notes": f"fit on {HOST}; see /home/{USER}/fit.log",
        # unknown keys park in extra_json
        "bank_dirs": [f"/home/{USER}/stage/x", f"/vol/out/x"], "blob_keys": ["kind", "layers"],
    })
    return o


def _leaky_run(key, i0, mid, objk, oid):
    """A judged run's records, every location leaking (mirrors synth.run_records)."""
    p = synth.run_payload(key, objk, oid, i0, model_id=mid, store_uri=f"/home/{USER}/stores/{key}", config_path=CONFIG_PATH,
                          config_sha256=synth.SHA256_B, notes=f"queued from {HOST}; log at /scratch/{USER}/logs/{key}.out",
                          # unknown keys park in extra_json
                          config=CONFIG_PATH, generation_complete_uri=f"/vol/registry/{key}.json",
                          bank_dirs=[f"/home/{USER}/a", "/tmp/b"], note=f"ran on {HOST} from /home/{USER}/x",
                          nested={"lane_file": f"/workspace/{USER}/f.json", "keep": "plain"},
                          wandb_url=f"https://wandb.example/{USER}/runs/1", keep_me="plain",
                          **{f"/home/{USER}/weird-key": "a leaking KEY"})
    rid = str(p["run_id"])
    jid = I.judgment_id(rid, synth.JUDGE_ID)
    t = synth.t
    out = [synth.rec("run", p, t(i0)),
           synth.rec("run_state", {"run_id": rid, "state": "queued", "event_utc": t(i0), "evidence_kind": "submit_wrapper"}, t(i0)),
           synth.rec("run_state", {"run_id": rid, "state": "running", "event_utc": t(i0 + 1), "evidence_kind": "job_log", "scheduler_state": "RUNNING",
                                   "evidence_uri": f"/scratch/{USER}/logs/{key}.out"}, t(i0 + 1)),
           synth.rec("run_patch", {"run_id": rid, "fields": {"node_hostname": HOST, "cluster_job_id": "4242", "gpu_count": 1},
                                   "verify": {"serving_path": "lora_native"}}, t(i0 + 2)),
           synth.rec("run_state", {"run_id": rid, "state": "generated", "event_utc": t(i0 + 3), "evidence_kind": "completion_artifact",
                                   "evidence_uri": f"/home/{USER}/stores/{key}/results.json"}, t(i0 + 3)),
           synth.rec("run_patch", {"run_id": rid, "fields": {"n_items_generated": N_ITEMS, "store_completion_artifact_uri": f"/home/{USER}/stores/{key}/completion.json",
                                                             "store_completion_sha256": synth.SHA256_A}}, t(i0 + 4)),
           synth.rec("judgment", {"judgment_id": jid, "run_id": rid, "judge_id": synth.JUDGE_ID, "started_utc": t(i0 + 5), "finished_utc": t(i0 + 6),
                                  "n_items_judged": N_ITEMS, "status": "complete", "api_pool": "lowprio", "store_uri": f"/home/{USER}/grades/{key}",
                                  "grades_root": f"/workspace/{USER}/grades/{key}"}, t(i0 + 6)),
           synth.rec("run_state", {"run_id": rid, "state": "judged", "event_utc": t(i0 + 7), "evidence_kind": "completion_artifact"}, t(i0 + 7)),
           synth.rec("run_patch", {"run_id": rid, "fields": {"n_items_judged": N_ITEMS, "judge_id": synth.JUDGE_ID}}, t(i0 + 8)),
           synth.rec("run_metric", {"run_id": rid, "judgment_id": jid, "metric_id": "asr", "threshold": 0.5, "value": 0.25, "k": 1, "n": N_ITEMS,
                                    "computed_by": "producer",
                                    "source_uri": f"/home/{USER}/grade/curves.json#targets.x; /home/{USER}/grade/aggregate.json (sha256 197f5f4ece52)"}, t(i0 + 9)),
           ]
    # item-scale records carry the WHOLE row (records.ITEM_WHOLE_ROW_COLUMNS): every column named, unknown keys parked in extra_json
    degen = dict.fromkeys(R.ITEM_WHOLE_ROW_COLUMNS["run_degeneration"], None)
    degen.update({"run_id": rid, "judgment_id": jid, "measure": "union5", "scope": "all", "measurement_state": "measured", "numerator": 1,
                  "denominator": N_ITEMS, "denominator_unit": "items", "pooling_json": {}, "aggregation": "union", "rate": 0.25,
                  "definition_version": "1", "threshold_params_json": {}, "computed_by": "producer", "computed_utc": t(i0 + 10),
                  "source_uri": f"/home/{USER}/degen/{key}/judge_summary.json", "notes": f"judged on {HOST}", "source_revision_id": ""})
    out.append(synth.rec("run_degeneration", degen, t(i0 + 10)))
    items = synth.item_records(rid, jid, i0 + 11, N_ITEMS, seed=1)
    flags = []
    for k in range(N_ITEMS):
        f = dict.fromkeys(R.ITEM_WHOLE_ROW_COLUMNS["item_flags"], None)
        f.update({"run_id": rid, "item_id": f"q{k:05d}", "trial_index": 0, "sample_index": 0, "turn_index": -1, "judgment_id": jid,
                  "at_token_cap": 0, "finish_reason": "stop", "response_tokens": 10, "source_revision_id": "",
                  "generation_artifact_uri": f"/home/{USER}/gen/{key}.jsonl", "at_token_cap_source": f"finish_reason ({HOST})"})
        flags.append(synth.rec("item_flags", f, t(i0 + 12)))
    return rid, out, items, flags


def _roster_entry(i, mid, object_id=None, selector=None):
    p = {"roster_entry_id": f"re{i}", "round_id": "r_synth", "comparison_id": "fig_synth", "model_id": mid, "eval_id": synth.EVAL_ID,
         "instrument_id": synth.INSTRUMENT_ID, "judge_id": synth.JUDGE_ID, "strength_rule": "operating_point",
         "source_citation": f"paperlib at /home/{USER}/si-steering/paper/paperlib/config.py", "inferred": 0}
    if object_id:
        p["object_id"] = object_id
    else:
        p["object_selector_json"] = selector
    return synth.rec("roster_entry", p, synth.t(50 + i))


def build_leaky_corpus(root):
    H.ensure_layout(root)
    dims = synth.seed_dimensions(0)
    dims = [d for d in dims if not (d["record_type"] == "si_text")]
    dims += [
        synth.rec("si_text", {"si_id": "synthetic_si", "si_text_sha256": synth.SI_SHA, "si_kind": "constitution_short",
                              "source_uri": f"$HOME/.local/state/x/candidate_sis.json::system_instructions[id=A10].text (file sha256 {synth.SHA256_A})"}, synth.t(3)),
        _model("gemma4e4b_it", "gemma4", "instruct", 11), _model(SN.PAPER_FRAME_TEST_MODEL_IDS[0], "qwen35", "instruct", 12),
        synth.rec("object", _leaky_object(), synth.t(13)),
        synth.rec("fit_search", {"search_id": "hpo:abc123def4567890", "campaign": synth.CAMPAIGN, "target_object_class_json": {"kind": "lora"},
                                 "method": "sobol", "n_trials": 3, "n_trials_completed": 3, "search_space_json": {"lr": [1e-4, 1e-3]},
                                 "objective": "heldout_loss", "objective_direction": "minimize", "substrate": "grading_host", "code_sha": synth.CODE_SHA,
                                 "started_utc": synth.t(1), "trials_uri": f"/home/{USER}/hpo/lora_r1", "grad_norm_reconnaissance_uri": "/tmp/gn.json",
                                 "wandb_sweep_url": f"https://wandb.example/{USER}/sweeps/1", "search_dir": f"/home/{USER}/hpo/lora_r1"}, synth.t(14)),
        synth.rec("ruling", {"ruling_id": "rl-synth-exclude", "round_id": "r_synth", "action": "exclude", "scope_kind": "model",
                             "scope_json": {"model_id": synth.MODEL_ID, "note": f"runs on {HOST} under /home/{USER}/x"},
                             "target_json": {"artifact_uri": f"/home/{USER}/t.json", "disposition": "declined"},
                             "expires_when_json": {"conditions": [], "guards": [{"artifact_uri": "~/figs/x/summary.csv", "sha256": synth.SHA256_A}]},
                             "reason": f"synthetic; evidence at /scratch/{USER}/e.json", "ruled_utc": synth.t(60), "ruled_by": "user",
                             "source_ref": "test"}, synth.t(60)),
    ]
    R.write_records(os.path.join(root, "records", "dimensions", "seed", "000001-dimensions.jsonl"), dims)
    run_ids = {}
    for k, (key, mid, objk, oid) in enumerate([("g_none", "gemma4e4b_it", "none", "none"),
                                                ("g_lora", "gemma4e4b_it", "constitution_lora@deadbeef", synth.OBJECT_SHA),
                                                ("t_none", SN.PAPER_FRAME_TEST_MODEL_IDS[0], "none", "none")]):
        rid, recs, items, flags = _leaky_run(key, 100 + 20 * k, mid, objk, oid)
        files = [(str(r["record_type"]), [r]) for r in recs] + [("item_result", items), ("item_flags", flags)]
        synth.write_run_dir(root, synth.CAMPAIGN, rid, files)
        run_ids[key] = rid
    R.write_records(os.path.join(root, "records", "roster", "000001-roster.jsonl"),
                    [_roster_entry(1, "gemma4e4b_it", object_id="none"), _roster_entry(2, "gemma4e4b_it", selector={"paper_label": "constitution_lora"})])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.refusals[:8]
    return run_ids


@pytest.fixture
def export(tmp_path, monkeypatch):
    root = str(tmp_path / "reg")
    dest = str(tmp_path / "dest")
    run_ids = build_leaky_corpus(root)
    monkeypatch.setattr(SN, "_private_tokens", lambda: TOKENS)
    with DB.connect(DB.db_path(root), readonly=True) as live:
        assert live.execute("SELECT COUNT(*) FROM blocked_records").fetchone()[0] == 0
        # the leaks ARE in the live database (the scrub is export-only)
        assert live.execute("SELECT store_uri FROM run LIMIT 1").fetchone()[0].startswith(f"/home/{USER}/")
        assert LEAK.search(live.execute("SELECT artifact_locations_json FROM object WHERE object_id = ?", (synth.OBJECT_SHA,)).fetchone()[0])
        assert live.execute("SELECT COUNT(*) FROM item_flags WHERE extra_json LIKE '%generation_artifact_uri%'").fetchone()[0] == 3 * N_ITEMS
        assert live.execute("SELECT COUNT(*) FROM run_degeneration").fetchone()[0] == 3
    info = SN.build(root, DB.db_path(root), check_free_space=False, scope="paper_frame", dest_root=dest,
                    precondition_override="scrub test: the fixture tree is not the #417 tree")
    return {"root": root, "dest": dest, "info": info, "run_ids": run_ids}


def _walk_strings(value, path=""):
    """Every string in a decoded value, with its JSON path (keys AND values)."""
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for k, v in value.items():
            if isinstance(k, str):
                yield f"{path}.<key>", k
            yield from _walk_strings(v, f"{path}.{k}")
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from _walk_strings(v, f"{path}[{i}]")


def _decoded_rows(info, table):
    """(parquet rows, sqlite rows) as lists of dicts, JSON cells decoded."""
    import pyarrow.parquet as pq
    pq_rows = pq.read_table(os.path.join(info.dir, f"{table}.parquet")).to_pylist()
    path, _, _ = SN.paper_frame_sqlite_path(os.path.dirname(os.path.dirname(info.dir)))
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    sq_rows = [dict(r) for r in conn.execute(f"SELECT * FROM {table}")]
    conn.close()
    return pq_rows, sq_rows


def _cells(rows):
    for i, row in enumerate(rows):
        for col, val in row.items():
            if isinstance(val, bytes):
                continue
            if isinstance(val, str) and col.endswith("_json"):
                try:
                    obj = json.loads(val)
                except ValueError:
                    obj = val
                for p, s in _walk_strings(obj, f"[{i}].{col}"):
                    yield col, p, s
            elif isinstance(val, str):
                yield col, f"[{i}].{col}", val


def test_no_location_hostname_or_user_token_survives_in_either_format(export):
    info = export["info"]
    for table in SN.PAPER_FRAME_TABLES:
        pq_rows, sq_rows = _decoded_rows(info, table)
        assert len(pq_rows) == len(sq_rows) == info.manifest["tables"][table], table
        for fmt, rows in (("parquet", pq_rows), ("sqlite", sq_rows)):
            for col, path, s in _cells(rows):
                assert not LEAK.search(s), (table, fmt, col, path, s[:200])


def test_known_hashes_read_sha256_and_repo_paths_read_relative(export):
    info, run_ids = export["info"], export["run_ids"]
    _, runs = _decoded_rows(info, "run")
    assert {r["run_id"] for r in runs} == set(run_ids.values())
    for r in runs:
        assert r["store_uri"] == f"sha256:{synth.SHA256_A}" == r["store_completion_artifact_uri"]      # store_completion_sha256
        assert r["config_path"] == f"sha256:{synth.SHA256_B}"                                            # config_sha256
        assert r["node_hostname"] == "<redacted>" and "<path>" in r["notes"] and "<redacted>" in r["notes"] and "queued from" in r["notes"]
        assert re.fullmatch(r"sha256:[0-9a-f]{64}", r["record_uri"])
        extra = json.loads(r["extra_json"])
        assert extra["config"] == "repo:configs/sdfbase_x.yaml"                    # a path under a checkout of this repo → relative
        assert "generation_complete_uri" not in extra and "wandb_url" not in extra   # no hash, not under the repo → key removed
        assert extra["bank_dirs"] == [] and extra["nested"] == {"keep": "plain"} and extra["keep_me"] == "plain"
        assert extra["note"] == "ran on <redacted> from <path>"
        assert "<path>" in "".join(k for k in extra if "weird-key" in k) or not any("weird-key" in k for k in extra)
    _, objs = _decoded_rows(info, "object")
    o = next(x for x in objs if x["object_id"] == synth.OBJECT_SHA)
    locs = json.loads(o["artifact_locations_json"])
    assert [l["uri"] for l in locs] == [f"sha256:{synth.OBJECT_SHA}"] * 2 and locs[0]["bytes"] == 1234 and locs[1]["source"] == "_records/x.json"
    side = json.loads(o["object_sidecars_json"])
    assert [s["uri"] for s in side] == ["train_budget_meta.json", "COMPLETE"] and side[0]["sha256"] == synth.SHA256_A
    assert o["fit_config_path"] == f"sha256:{synth.SHA256_B}" and o["fit_heldout_curve_uri"] is None and o["fit_wandb_url"] is None
    assert json.loads(o["extra_json"])["bank_dirs"] == [] and json.loads(o["extra_json"])["blob_keys"] == ["kind", "layers"]
    _, fs = _decoded_rows(info, "fit_search")
    assert fs[0]["trials_uri"] is None and fs[0]["grad_norm_reconnaissance_uri"] is None and fs[0]["wandb_sweep_url"] is None
    assert "search_dir" not in json.loads(fs[0]["extra_json"])
    _, js = _decoded_rows(info, "judgment")
    assert all(j["store_uri"] is None for j in js) and all("grades_root" not in json.loads(j["extra_json"]) for j in js)
    _, rm = _decoded_rows(info, "run_metric")
    assert all(m["source_uri"] == "<path>#targets.x; <path> (sha256 197f5f4ece52)" for m in rm)   # a provenance sentence keeps its wording
    _, rd = _decoded_rows(info, "run_degeneration")
    assert len(rd) == 3 and all(d["source_uri"] is None for d in rd)
    _, rse = _decoded_rows(info, "run_state_event")
    leaky_events = [e for e in rse if e["state"] in ("running", "generated")]
    assert leaky_events and all(re.fullmatch(r"scrubbed:[0-9a-f]{16}", e["evidence_uri"]) for e in leaky_events)   # NOT NULL key column
    assert len({e["evidence_uri"] for e in leaky_events}) == len(leaky_events)                                    # ... stays unique
    _, flags = _decoded_rows(info, "item_flags")
    assert flags and all("generation_artifact_uri" not in json.loads(f["extra_json"]) for f in flags)
    assert all(json.loads(f["extra_json"])["at_token_cap_source"] == "finish_reason (<redacted>)" for f in flags)
    _, si = _decoded_rows(info, "si_text")
    assert si[0]["source_uri"] == f"<path> (file sha256 {synth.SHA256_A})"      # a locator sentence keeps its wording; the $HOME path (with its [selector]) goes
    _, ru = _decoded_rows(info, "ruling")
    guards = json.loads(ru[0]["expires_when_json"])["guards"]
    assert guards == [{"sha256": synth.SHA256_A}]                                                                    # artifact_uri key removed
    assert "artifact_uri" not in json.loads(ru[0]["target_json"]) and json.loads(ru[0]["target_json"])["disposition"] == "declined"
    assert json.loads(ru[0]["scope_json"])["note"] == "runs on <redacted> under <path>" and ru[0]["reason"] == "synthetic; evidence at <path>"
    _, models = _decoded_rows(info, "model")
    g = next(m for m in models if m["model_id"] == "gemma4e4b_it")
    assert json.loads(g["extra_json"])["bank_presence"] == "NOT in <path> on <redacted>"
    _, ri = _decoded_rows(info, "record_ingest")
    assert ri[0]["file_uri"] == f"sha256:{ri[0]['file_sha256']}"
    _, roster = _decoded_rows(info, "roster_entry")
    assert all(r["source_citation"] == "paperlib at <path>" for r in roster)
    # the materialised long view carries the run table's scrubbed store_uri, not its own NULL
    _, long = _decoded_rows(info, "v_paper_long")
    assert long and all(r["store_uri"] == f"sha256:{synth.SHA256_A}" for r in long)
    assert all("<redacted>" in r["notes"] for r in long)


def test_manifest_scrubbed_names_every_column_touched(export):
    root, info = export["root"], export["info"]
    scrubbed = info.manifest["scrubbed"]
    live = DB.connect(DB.db_path(root), readonly=True)
    touched = {}
    for table in SN.PAPER_FRAME_TABLES:
        _, rows = _decoded_rows(info, table)
        if table in S.TABLE_BY_NAME:
            pk = list(S.TABLE_BY_NAME[table].pk)
            cols = list(S.TABLE_BY_NAME[table].column_names)
            source = f"v_{table}_current" if table in S.ITEM_SCALE_TABLES else table
        else:
            cols = [d[0] for d in live.execute(f"SELECT * FROM {table} LIMIT 0").description]
            pk = [c for c in ("run_id", "object_id", "roster_entry_id", "comparison_id", "judgment_id", "metric_id", "round_id") if c in cols]
            source = table
        key_cols = [c for c in pk if c not in ("evidence_uri",)]        # a scrubbed key column cannot align itself
        live_rows = {tuple(r[c] for c in key_cols): dict(r) for r in live.execute(f"SELECT {', '.join(cols)} FROM {source}")}
        diff = set()
        for r in rows:
            k = tuple(r[c] for c in key_cols)
            if k not in live_rows:
                continue
            for c in cols:
                a, b = r.get(c), live_rows[k].get(c)
                if c.endswith("_json") and isinstance(a, str) and isinstance(b, str):
                    try:
                        if json.loads(a) == json.loads(b):
                            continue
                    except ValueError:
                        pass
                if a != b:
                    diff.add(c)
        if diff:
            touched[table] = sorted(diff)
    live.close()
    for table, cols in touched.items():
        assert table in scrubbed, (table, cols)
        assert set(cols) <= set(scrubbed[table]), (table, sorted(set(cols) - set(scrubbed[table])))
    # the columns the brief names are all listed
    assert {"store_uri", "store_completion_artifact_uri", "config_path", "extra_json", "notes", "node_hostname", "record_uri"} <= set(scrubbed["run"])
    assert {"artifact_locations_json", "object_sidecars_json", "fit_config_path", "fit_heldout_curve_uri", "fit_wandb_url", "extra_json", "notes"} <= set(scrubbed["object"])
    assert {"trials_uri", "grad_norm_reconnaissance_uri", "wandb_sweep_url", "extra_json"} <= set(scrubbed["fit_search"])
    assert {"store_uri", "extra_json"} <= set(scrubbed["judgment"]) and "source_uri" in scrubbed["run_metric"] and "source_uri" in scrubbed["run_degeneration"]
    assert "evidence_uri" in scrubbed["run_state_event"] and "extra_json" in scrubbed["item_flags"] and "source_uri" in scrubbed["si_text"]
    assert {"expires_when_json", "target_json", "scope_json", "reason"} <= set(scrubbed["ruling"]) and scrubbed["record_ingest"] == ["file_uri"]
    assert "store_uri" in scrubbed["v_paper_long"] and "notes" in scrubbed["v_paper_long"]
    # a table with nothing to scrub is absent from the map
    assert "computation" not in scrubbed or scrubbed["computation"] == ["record_uri"]


def test_scrub_frame_is_pure_and_keyed_by_column_kind():
    """The pure function on hand-made rows: the three forms, NOT NULL handling, JSON keys, prose, a clean row untouched."""
    rows = [{"run_id": "r1", "store_uri": f"/home/{USER}/s", "store_completion_sha256": synth.SHA256_A, "config_path": f"/tmp/{USER}/c.yaml",
             "config_sha256": None, "notes": f"on {HOST}", "extra_json": json.dumps({"config": f"/home/{USER}/si-steering/worktrees/w/configs/a.yaml",
                                                                                     "uris": [f"/vol/x", "keep"], "plain": "ok"}),
             "record_uri": "records/x/000001-run.jsonl", SN.RECORD_SHA_HELPER: synth.SHA256_B, "campaign": "c"},
            {"run_id": "r2", "store_uri": "synthetic:///store/x", "store_completion_sha256": None, "config_path": None, "config_sha256": None,
             "notes": "clean", "extra_json": "{}", "record_uri": "records/y/000001-run.jsonl", SN.RECORD_SHA_HELPER: None, "campaign": "c"}]
    out, cols = SN.scrub_frame("run", rows, tokens=TOKENS, repo_roots=("/nonexistent/repo",))
    assert cols == ["config_path", "extra_json", "notes", "record_uri", "store_uri"]
    a, b = out
    assert a["store_uri"] == f"sha256:{synth.SHA256_A}" and a["config_path"] is None and a["notes"] == "on <redacted>"
    assert a["record_uri"] == f"sha256:{synth.SHA256_B}" and SN.RECORD_SHA_HELPER not in a
    assert json.loads(a["extra_json"]) == {"config": "repo:configs/a.yaml", "uris": ["keep"], "plain": "ok"}
    assert b["store_uri"] == "synthetic:///store/x" and b["record_uri"] == "records/y/000001-run.jsonl" and b["notes"] == "clean"
    # a NOT NULL location column without a hash takes the unique scrubbed form; a leaking KEY is redacted
    rows = [{"run_id": "r", "state": "generated", "event_utc": synth.t(1), "evidence_uri": f"/home/{USER}/e.json", "evidence_kind": "completion_artifact",
             "extra_json": json.dumps({f"/home/{USER}/k": 1}), "record_uri": "records/z"}]
    out, cols = SN.scrub_frame("run_state_event", rows, tokens=TOKENS, repo_roots=())
    assert re.fullmatch(r"scrubbed:[0-9a-f]{16}", out[0]["evidence_uri"]) and json.loads(out[0]["extra_json"]) == {"<path>": 1}
    assert cols == ["evidence_uri", "extra_json"]
    # the sidecar basename rule and the object's artifact hash
    rows = [{"object_id": "o", "artifact_sha256": synth.SHA256_A,
             "artifact_locations_json": json.dumps([{"uri": f"/home/{USER}/o.pt", "bytes": 1}]),
             "object_sidecars_json": json.dumps([{"kind": "COMPLETE", "sha256": synth.SHA256_B, "uri": f"/home/{USER}/x/COMPLETE"}]),
             "record_uri": "records/o"}]
    out, cols = SN.scrub_frame("object", rows, tokens=TOKENS, repo_roots=())
    assert json.loads(out[0]["artifact_locations_json"]) == [{"uri": f"sha256:{synth.SHA256_A}", "bytes": 1}]
    assert json.loads(out[0]["object_sidecars_json"]) == [{"kind": "COMPLETE", "sha256": synth.SHA256_B, "uri": "COMPLETE"}]
    assert cols == ["artifact_locations_json", "object_sidecars_json"]
    # URLs with a hostname are locations; a scheme with no host (synthetic:///) is not
    out, cols = SN.scrub_frame("fit_search", [{"search_id": "s", "wandb_sweep_url": "https://wandb.example/e/sweeps/1", "trials_uri": "synthetic:///t",
                                              "record_uri": "r"}], tokens=TOKENS, repo_roots=())
    assert out[0]["wandb_sweep_url"] is None and out[0]["trials_uri"] == "synthetic:///t" and cols == ["wandb_sweep_url"]
    # $HOME and ~ name a home directory
    out, cols = SN.scrub_frame("si_text", [{"si_id": "s", "source_uri": "$HOME/.local/x.json::text", "record_uri": "r"},
                                           {"si_id": "t", "source_uri": "~/x.txt", "record_uri": "r"}], tokens=TOKENS, repo_roots=())
    assert all(re.fullmatch(r"scrubbed:[0-9a-f]{16}", o["source_uri"]) for o in out) and cols == ["source_uri"]
