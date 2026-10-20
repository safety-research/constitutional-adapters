"""The job-side closer (`scripts/sdfbase/full22/registry_close_stores.py`): a full22 job's REGISTERED runs (a REAL job-start
registration over the real seeds, in a registry seeded the real way) are closed from the stores the job wrote — `run_patch` with the
store_uri + completion evidence and `run_state generated`, through the real emitter — judged against the registration's PLAN
(`n_items_planned`, the item set's goals) and the job's shard plan, never the directories present: shard00 alone of a nine-shard job is
refused, a progress row not done is refused, a count off the plan is refused; full coverage files `generated` with n_items_generated ==
n_items_planned. BoN: a cell whose every slot is a decided no-fit refusal is complete (0 generated items); an undecided slot refuses.
Controls: the object's cell at the FINAL rung (escalate false) is the store; an unfinished ladder refuses. The records fold into a real
registry and the closed runs enter the pending-judgment worklist with the store path the grading daemon's layouts expect."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from sisteer.registry import db as DB
from sisteer.registry import emit, emit_cli, harvest as H, hooks, pending as P, synth
from sisteer.registry import records as R

REPO = Path(__file__).resolve().parents[1]
CLOSER = REPO / "scripts" / "sdfbase" / "full22" / "registry_close_stores.py"
SEEDS = REPO / "configs" / "registry"
TAG = "qwen35_4b"
OBJECTS = ("none", "prompted__safety_refusal")
#: decode-recording: the request sampler the synthetic stores record (the vendor six, as the demo smoke's stores carry them) and the
#: producers' own statements beside it; every fixture writes them by default (`decode=None` / `target_decode=None` / `sampling=None` is the
#: legacy store, which files the store facts plus the `unverified` entry)
VENDOR_SIX = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0}
VENDOR_SIX_F = {k: (int(v) if k == "top_k" else float(v)) for k, v in VENDOR_SIX.items()}
STATIC_DECODE = dict(VENDOR_SIX, do_sample=True, cap=4096, arm="vendor_recommended", seed_base=20260909, trial=0)
BON_DECODE = {"temperature": 1.0}
BON_POLICY_ID = f"{TAG}_demo|bon|off|grandfathered_T1@v1"
MT_POLICY_ID = f"{TAG}_demo|multi_turn|off|vendor_preset@v1"


def _seeded_registry(root: Path) -> None:
    from sisteer.registry.adapters import dimensions as DIM
    H.ensure_layout(str(root))
    bundle = DIM.build_seed_bundle(str(root), str(SEEDS), verify_si=False)
    assert DIM.publish_seed_records(str(root), bundle)
    assert H.ingest_once(str(root), workers=0, min_free=0).refused == 0


def _register(tmp_path: Path, root: Path, *, job: str, job_name: str, evals: str, producer_key: "str | None" = None, code_sha=None, extra=()) -> Path:
    """The REAL job-start registration over the real seeds (one run per eval x object), spooled into the registry's records. `producer_key`:
    the job-independent key common.sh forms; default jobplan's `<job_name>@<job_id>` — two job ids, two run ids. `extra`:
    further register flags (a rung registration's `--token-cap / --rung-index / --ladder / --pass-label`)."""
    co = tmp_path / f"checkout_{job}"; (co / "configs").mkdir(parents=True)
    os.symlink(SEEDS, co / "configs" / "registry")
    cfg = co / "configs" / f"sdfbase_full22_{TAG}.yaml"; cfg.write_text(yaml.safe_dump({"explicit_cells": list(OBJECTS)}))
    pend = tmp_path / "pending"
    rc = emit_cli.main(["register", "--from-config", str(cfg), "--model-key", TAG, "--evals", evals, "--serving-path", "lora_native", "--campaign", "sdfbase_full22",
                        "--job-id", job, "--job-name", job_name, "--pending-dir", str(pend), "--spool", str(root / "records"), "--code-sha", code_sha or synth.code_sha("a"),
                        "--substrate", "cluster_a", "--no-push", *(["--producer-key", producer_key] if producer_key else []), *extra])
    assert rc == 0
    return pend / f"{job}.json"


def _runs(pending: Path) -> dict:
    return {(r["route"]["eval_id"], r["route"]["object"]): r for r in json.loads(pending.read_text())["runs"]}


def _static_store(wave: Path, trial: int, eval_id: str, obj: str, *, nshards: int, goals: int, variants: int, shards=None, fail_goal_in_shard=None,
                  decode=STATIC_DECODE, row_decode=None) -> Path:
    """`decode`: served_target.json#decode and every response row's `decode` (the sampled static layout); None = a legacy / greedy store with
    neither; `row_decode` overrides the rows' block alone (a store whose rows disagree with served_target.json)."""
    d = wave / "transcripts_results" / f"sdfbase_full22_{TAG}_sampled_t{trial}" / eval_id / obj
    per = goals // nshards; extra = goals - per * nshards
    for s in (shards if shards is not None else range(nshards)):
        sd = d / f"shard{s:02d}"; sd.mkdir(parents=True, exist_ok=True)
        n = per + (1 if s < extra else 0)
        prog, resp = [], []
        for g in range(n):
            gid = f"g{s:02d}_{g}"; outcome = "failed" if (fail_goal_in_shard == s and g == 0) else "done"
            prog.append({"goal_id": gid, "family": eval_id, "n_variants": variants, "outcome": outcome, "capped": False})
            for v in range(variants):
                rd = row_decode if row_decode is not None else decode
                resp.append({"goal_id": gid, "variant": v, "response": "x", **({"decode": dict(rd)} if rd is not None else {})})
        (sd / "progress.jsonl").write_text("".join(json.dumps(r) + "\n" for r in prog))
        (sd / "responses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in resp))
        (sd / "served_target.json").write_text(json.dumps({"n_sampled_requests": len(resp), **({"decode": dict(decode)} if decode is not None else {})}))
    return d


def _nofit_row(g, s, t, **over):
    """A typed no-fit row as served_window.PromptOverWindow.record() + bon_served_run write it (the canonical key set)."""
    row = {"goal_id": g, "sample_idx": s, "trial": t, "engine": "vllm_served", "serving_engine": "lora_native", "cell_id": "x", "error": "prompt_over_window",
           "prompt_tokens": 6017, "prompt_tokens_source": "tokenize_endpoint", "prompt_floor": 4353, "requested_max_tokens": 512, "completion_tokens": 0,
           "window": 4864, "server_window": 4864, "effective_window": 4864, "window_mismatch": None,
           "window_reason": "prompt 6017 leaves -1153 of window 4864 (< 64 minimum completion)", "window_reason_detail": "", "overflow_excerpt": "maximum context length"}
    row.update(over); return row


def _bon_plan_config(tmp_path: Path, *, goals: int, n_max: int) -> Path:
    """A wave config the store job's own goal resolver (two_si_run._eval_goals, attack='bon') accepts offline: a JBB fixture CSV of `goals`
    harmful behaviours (`Index`, `Goal`, `Category`, `Source`), the confirmatory goal count, the bon block (`n_max`, `goal_subset: jbb`)."""
    csv = tmp_path / "jbb_fixture.csv"
    csv.write_text("Index,Goal,Category,Source\n" + "".join(f"{i},goal {i},cat{i % 10},synthetic\n" for i in range(goals)))
    cfg = tmp_path / "plan_config.yaml"
    cfg.write_text(yaml.safe_dump({"explicit_cells": list(OBJECTS),
                                   "datasets": {"jbb": {"revision": "fixture", "path": str(csv), "harmful_subset": goals, "stratify_seed": 0},
                                                "harmbench": {"revision": "fixture", "path": str(tmp_path / "absent.parquet")}},
                                   "prereg": {"confirmatory_goals": goals}, "bon": {"n_max": n_max, "goal_subset": "jbb"}}))
    return cfg


def _bon_store(wave: Path, obj: str, *, goals: int, n_max: int, responded: int, refused: int, manifest: bool = True, config: Path | None = None,
               decode=BON_DECODE, preset: str = "harness_default", policy_id: str = BON_POLICY_ID) -> Path:
    """A BoN cell as bon_served_run.py leaves it: `responded` slots with a validated response row, `refused` slots with a TYPED no-fit row
    (prompt_over_window with the exact prompt length from the tokenize endpoint against the effective window: `nofit_proven`), the rest
    undecided; the end-of-run marker `bon_served_run.json` (its n_target_forwards is the LAST invocation's call counter — here deliberately
    small, as after a resume) and the durable `bon_served_manifest.json` whose `units` the shared predicate reads."""
    d = wave / "transcripts_results" / f"sdfbase_full22_{TAG}" / obj; d.mkdir(parents=True, exist_ok=True)
    slots = [(f"jbb_harmful_{g}", s, 0) for g in range(goals) for s in range(n_max)]                                   # the resolver's goal ids
    sampler = {"decode": dict(decode), "bon_decode_preset": preset, "decode_policy_id": policy_id} if decode is not None else {}   # bon_served_run.py's record per row and on both manifests
    (d / "bon_served_run.json").write_text(json.dumps({"cell_id": obj, "n_goals": goals, "n_goals_selected": goals, "n_max": n_max, "n_target_forwards": 3,
                                                       "config": str(config) if config else "", "confirmatory": True,
                                                       **({**sampler, "bon_decode_preset_declared": "canonical"} if decode is not None else {})}))
    (d / "bon_responses.jsonl").write_text("".join(json.dumps({"goal_id": g, "sample_idx": s, "trial": t, "response": "y", **sampler}) + "\n" for g, s, t in slots[:responded]))
    (d / "bon_errors.jsonl").write_text("".join(json.dumps(_nofit_row(g, s, t)) + "\n" for g, s, t in slots[responded:responded + refused]))
    if manifest:
        n_planned = goals * n_max
        (d / "bon_served_manifest.json").write_text(json.dumps({"cell_id": obj, "units": {"n_planned": n_planned, "n_work": n_planned, "n_written": responded,
                                                                                       "n_prompt_over_window": refused, "n_done": responded, "n_nofit": refused,
                                                                                       "n_uncovered": n_planned - responded, "n_undecided": n_planned - responded - refused},
                                                                **sampler, "seed_rule": "bon_seed_v1", "n_legacy_unrecorded_rows": 0}))
    return d


def _misalign_cell(root: Path, cell: str, *, items: dict, n_samples: int, finished: bool = True, drop: int = 0, sampling=BON_DECODE) -> Path:
    """The misalign runner's agentic cell: `<root>/agentic/<cell>/responses.jsonl` (one row per (item, sample), `behavior` on every row; both
    behaviours in one file) + meta.json; `drop` removes the last rows (a cell cut short)."""
    d = root / "agentic" / cell; d.mkdir(parents=True, exist_ok=True)
    rows = [{"item_id": i, "behavior": b, "sample_idx": s, "cell_id": cell, "family": "agentic", "response": "r"} for b, ids in items.items() for i in ids for s in range(n_samples)]
    rows = rows[: len(rows) - drop] if drop else rows
    (d / "responses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    all_ids = [i for ids in items.values() for i in ids]
    (d / "meta.json").write_text(json.dumps({"cell_id": cell, "n_samples": n_samples, "items": len(all_ids), "item_ids": all_ids, "behaviors": sorted(items),
                                             "finished": 1.0 if finished else None, "present": len(rows),
                                             **({"sampling": dict(sampling), "sampler": "t1", "params": {"max_tokens": 16384, "temperature": 1.0, "sampler": "t1", "sampling": dict(sampling)}}
                                                if sampling is not None else {})}))
    return d


def _controls_cell(wave: Path, trial: int, obj: str, cap: int, *, escalate: dict, n: dict, rows: dict, decode=VENDOR_SIX) -> Path:
    """`decode`: the sampled cell's resolved_config.gen sampler keys (+ the driver's other gen keys) and gen_stats.json[<control>].decode; None =
    the fixture's former empty manifest (no gen: the closer files the facts plus the `unverified` entry)."""
    cell = wave / "sampled" / f"controls_cap{cap}" / f"{TAG}_t{trial}" / "cells" / f"{obj.replace('__', '--')}__controls__t0__a0__deadbeef__{cap:08x}"
    (cell / "private").mkdir(parents=True, exist_ok=True)
    (cell / "cap_ladder.json").write_text(json.dumps({"cap": cap, "cell_id": obj, "controls": {k: {"escalate": escalate[k], "exit": "escalate" if escalate[k] else "clean", "n": n[k]} for k in n}}))
    for k, r in rows.items():
        (cell / "private" / f"gens_{k}{'_v2' if k == 'xstest' else ''}.jsonl").write_text("".join(json.dumps({"i": i}) + "\n" for i in range(r)))
    if decode is None:
        (cell / "manifest.json").write_text("{}")
    else:
        gen = dict(decode, arm="vendor_recommended", batch_size=8, do_sample=True, max_new_tokens=cap, seed_base=20260909, trial=trial, escalated_from={"cap": cap // 4, "root": str(wave)})
        (cell / "manifest.json").write_text(json.dumps({"cell_id": obj, "controls": ["xstest", "or_bench_hard_1k"], "resolved_config": {"driver": "served_capability_run_v2", "gen": gen}}))
        (cell / "gen_stats.json").write_text(json.dumps({k: {"n": n[k], "decode": dict(decode, do_sample=True, cap=cap, arm="vendor_recommended", seed_base=20260909, trial=trial)} for k in n}))
    (cell / "evidence.json").write_text("{}")
    return cell


def _close(pending: Path, spool: Path, wave: Path, *extra: str, job_name=None, bare: bool = False) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=str(REPO / "src")); env.pop("SISTEER_REGISTRY_DISABLED", None)
    # no `-S` by default: the closer imports bon_climb.store_completion / registry_adapter.harness_cell_id, whose packages need PyYAML from
    # site-packages; `bare=True` IS `python -S` — the cluster job's interpreter, which cannot import the multi-turn goal resolver
    argv = [sys.executable, *(["-S"] if bare else []), str(CLOSER), "--pending", str(pending), "--wave-root", str(wave), "--tag", TAG, "--spool", str(spool), "--code-sha", synth.code_sha("b"), *extra]
    if job_name:
        argv += ["--job-name", job_name]
    return subprocess.run(argv, capture_output=True, text=True, env=env, timeout=300)


def _records(spool: Path, run_id: str):
    from sisteer.registry import identity as I
    d = spool / "sdfbase_full22" / I.run_dir_name(run_id); out = []
    for f in sorted(d.glob("0*.jsonl")):
        for ln in f.read_text().splitlines():
            rec = json.loads(ln); out.append((rec.get("record_type"), rec.get("payload") or {}, rec.get("producer") or {}))
    return out


def _generated(spool: Path, run_id: str):
    recs = _records(spool, run_id)
    st = [p for t, p, _pr in recs if t == "run_state" and p.get("state") == "generated"]
    pa = [p["fields"] for t, p, _pr in recs if t == "run_patch" and "store_uri" in (p.get("fields") or {})]
    return (st[-1] if st else None), (pa[-1] if pa else None)


def test_static_completeness_is_judged_against_the_plan_and_the_shard_count(tmp_path, monkeypatch):
    for k in (emit.DISABLED_ENV, emit.SPOOL_ENV, emit.PUSH_SCRIPT_ENV, hooks.EMIT_ENV, hooks.RUN_IDENTITY_ENV):
        monkeypatch.delenv(k, raising=False)
    root = tmp_path / "reg"; _seeded_registry(root); spool = root / "records"; wave = tmp_path / "wave"
    pending = _register(tmp_path, root, job="11111", job_name=f"f22s-{TAG}-static_base_t0", evals="pasttense,grok420")
    runs = _runs(pending); assert len(runs) == 4
    plan = runs[("pasttense", "none")]["plan"]; planned, iset = plan["n_items_planned"], plan["item_set_id"]
    goals, variants = (int(x) for x in iset.rsplit("_", 1)[1].split("x")); assert goals * variants == planned
    # pasttense/none: shard00 ONLY of the nine planned shards; pasttense/prompted: full but one goal FAILED; grok420/none: full and clean;
    # grok420/prompted: full but one response short of the plan
    _static_store(wave, 0, "pasttense", "none", nshards=9, goals=goals, variants=variants, shards=[0])
    _static_store(wave, 0, "pasttense", "prompted__safety_refusal", nshards=9, goals=goals, variants=variants, fail_goal_in_shard=3)
    good = _static_store(wave, 0, "grok420", "none", nshards=9, goals=goals, variants=variants)
    short = _static_store(wave, 0, "grok420", "prompted__safety_refusal", nshards=9, goals=goals, variants=variants)
    p = short / "shard08" / "responses.jsonl"; p.write_text("".join(p.read_text().splitlines(keepends=True)[:-1]))
    assert H.ingest_once(str(root), workers=0, min_free=0).refused == 0
    r = _close(pending, spool, wave, "--nshards", "9", "--dry-run", job_name=f"f22s-{TAG}-static_base_t0")
    assert r.returncode == 2 and r.stdout.count("DRY-RUN ") == 1 and r.stdout.count("REFUSED ") == 3, r.stdout + r.stderr
    assert "8 of 9 expected shard(s) missing or incomplete: shard01, shard02" in r.stdout
    assert "1 progress row(s) not done ({'failed': 1}" in r.stdout
    assert f"{planned - 1} response row(s), the registration plans {planned}" in r.stdout
    assert all(_generated(spool, r_["run_id"]) == (None, None) for r_ in runs.values())
    r = _close(pending, spool, wave, "--nshards", "9", job_name=f"f22s-{TAG}-static_base_t0")
    assert r.returncode == 2 and "1 filed, 0 skipped, 3 refused of 4" in r.stdout, r.stdout + r.stderr
    st, pa = _generated(spool, runs[("grok420", "none")]["run_id"])
    assert pa["store_uri"] == str(good) and pa["n_items_generated"] == planned and pa["store_completion_artifact_uri"] == str(good / "shard00" / "progress.jsonl")
    # decode-recording: the SAME patch carries the request's observed sampler -- served_target.json#decode, every response row agreeing
    assert pa["decode_sent_json"] == VENDOR_SIX_F and (pa["decode_temperature_sent"], pa["decode_top_p_sent"], pa["decode_top_k_sent"]) == (0.7, 0.8, 20) and "decode_resolved_json" not in pa
    assert pa["field_confidence_json"]["decode_sent_json"]["confidence"] == "recorded" and pa["field_confidence_json"]["decode_sent_json"]["source"].startswith(str(good / "shard00" / "served_target.json") + "#decode; sha256 ")
    assert pa["extra_json"]["decode_evidence"]["family"] == "static" and pa["extra_json"]["decode_evidence"]["witnesses"]["response_rows"] == planned
    assert pa["extra_json"]["decode_evidence"]["unmodelled_keys"] == {"shard00/served_target.json#decode": ["arm", "cap", "do_sample", "seed_base", "trial"]}
    assert set(pa) <= set(R.RUN_PATCHABLE_FIELDS), set(pa) - set(R.RUN_PATCHABLE_FIELDS)
    g_iset = runs[("grok420", "none")]["plan"]["item_set_id"]
    assert st["evidence_kind"] == "completion_artifact" and f"{planned} planned, item set {g_iset}" in st["note"] and '"shards_missing": []' in st["note"]
    for k in (("pasttense", "none"), ("pasttense", "prompted__safety_refusal"), ("grok420", "prompted__safety_refusal")):
        assert _generated(spool, runs[k]["run_id"]) == (None, None)
    # the missing shards arrive: the SAME command now files that run (nothing else re-filed: idempotent)
    _static_store(wave, 0, "pasttense", "none", nshards=9, goals=goals, variants=variants, shards=range(1, 9))
    r = _close(pending, spool, wave, "--nshards", "9", job_name=f"f22s-{TAG}-static_base_t0")
    assert "1 filed, 1 skipped, 2 refused of 4" in r.stdout and r.stdout.count("already generated") == 1
    # the fold: the two complete runs enter the pending-judgment worklist with their stores; the refused two stay `running`
    assert H.ingest_once(str(root), workers=0, min_free=0).refused == 0
    with DB.reading(DB.db_path(str(root))) as conn:
        pend = {x["run_id"]: x for x in P.pending(conn, None, "sdfbase_full22", run_id=tuple(v["run_id"] for v in runs.values()))}
        assert set(pend) == {runs[("grok420", "none")]["run_id"], runs[("pasttense", "none")]["run_id"]}
        assert pend[runs[("grok420", "none")]["run_id"]]["pending_reason"] == "no_judgment" and pend[runs[("grok420", "none")]["run_id"]]["store_uri"] == str(good)
        assert conn.execute("SELECT lifecycle_state, n_items_generated FROM run WHERE run_id=?", (runs[("grok420", "none")]["run_id"],)).fetchone()[:2] == ("generated", planned)
        assert conn.execute("SELECT lifecycle_state FROM run WHERE run_id=?", (runs[("grok420", "prompted__safety_refusal")]["run_id"],)).fetchone()[0] == "running"
    sys.path.insert(0, str(REPO / "scripts" / "registry"))
    import grade_daemon as GD
    assert GD.store_root_of("static", str(good)) == (str(wave / "transcripts_results"), f"sdfbase_full22_{TAG}_sampled_t0")
    # explicit static form (--private-root/--private-run) and a job name without a trial are handled by name
    r = _close(pending, spool, wave, "--nshards", "9", "--private-root", str(wave / "transcripts_results"), "--private-run", f"sdfbase_full22_{TAG}_sampled_t0")
    assert "2 skipped, 2 refused" in r.stdout
    r = _close(pending, spool, wave, "--nshards", "9", job_name="f22s-no-trial")
    assert r.returncode == 2 and "needs --wave-root, --tag and a trial" in r.stdout
    assert "allow-partial" not in CLOSER.read_text()


def test_bon_completeness_is_the_stores_durable_manifest_through_the_shared_predicate(tmp_path, monkeypatch):
    """A BoN cell is complete when `bon_climb.store_completion` over `bon_served_manifest.json` says every planned
    slot is DECIDED (n_undecided == 0, n_done + n_nofit == n_planned) — never the marker's `n_target_forwards`, the last invocation's call
    counter. An all-no-fit cell with typed `nofit_proven` rows (the live si_constitution shape) files with 0 generated items; a torn line
    and a stray goal id are reported as ledger integrity, never a shortfall; a resumed store missing a draw — an intact marker, one
    response row short — refuses; a store without the durable manifest refuses (completeness cannot be certified)."""
    for k in (emit.DISABLED_ENV, emit.SPOOL_ENV, emit.PUSH_SCRIPT_ENV, hooks.EMIT_ENV, hooks.RUN_IDENTITY_ENV):
        monkeypatch.delenv(k, raising=False)
    root = tmp_path / "reg"; _seeded_registry(root); spool = root / "records"; wave = tmp_path / "wave"
    pending = _register(tmp_path, root, job="808185", job_name=f"f22-stat-{TAG}-3", evals="strongreject_bon")
    runs = _runs(pending); assert set(runs) == {("strongreject_bon", o) for o in OBJECTS}
    goals = int(json.loads(pending.read_text())["runs"][0]["plan"]["n_items_planned"]); assert goals > 0
    cfg = _bon_plan_config(tmp_path, goals=goals, n_max=3)
    full = _bon_store(wave, "none", goals=goals, n_max=3, responded=goals * 3, refused=0, config=cfg)                        # every slot answered
    nofit = _bon_store(wave, "prompted__safety_refusal", goals=goals, n_max=3, responded=0, refused=goals * 3, config=cfg)   # every slot a decided no-fit
    with open(nofit / "bon_errors.jsonl", "a") as fh:                                                              # a torn line and a stray goal: ledger integrity, not a shortfall
        fh.write('{"goal_id": "jbb_\n'); fh.write(json.dumps({"goal_id": "extra_goal", "sample_idx": 0, "trial": 0, "error": "prompt_over_window", "prompt_tokens": 6017,
                                                             "prompt_tokens_source": "tokenize_endpoint", "effective_window": 4864}) + "\n")
    r = _close(pending, spool, wave, job_name=f"f22-stat-{TAG}-3")
    assert r.returncode == 0, r.stdout + r.stderr
    assert H.ingest_once(str(root), workers=0, min_free=0).refused == 0
    st1, pa1 = _generated(spool, runs[("strongreject_bon", "none")]["run_id"])
    st2, pa2 = _generated(spool, runs[("strongreject_bon", "prompted__safety_refusal")]["run_id"])
    assert pa1["store_uri"] == str(full) and pa1["n_items_generated"] == goals * 3 and pa1["store_completion_artifact_uri"] == str(full / "bon_served_manifest.json")
    assert pa2["store_uri"] == str(nofit) and pa2["n_items_generated"] == 0
    # decode-recording: the manifest's `decode` (T=1.0), the rows agreeing (and the all-no-fit store with no row), the producer's own policy id as a statement
    for pa_, n_rows in ((pa1, goals * 3), (pa2, 0)):
        assert pa_["decode_sent_json"] == {"temperature": 1.0} and pa_["decode_temperature_sent"] == 1.0 and "decode_top_p_sent" not in pa_
        dev = pa_["extra_json"]["decode_evidence"]
        assert dev["family"] == "bon" and dev["producer_policy_id"] == BON_POLICY_ID and dev["witnesses"]["rows_recorded"] == n_rows and dev["witnesses"]["bon_decode_preset_declared"] == "canonical"
        assert pa_["field_confidence_json"]["decode_sent_json"]["source"].startswith(pa_["store_completion_artifact_uri"] + "#decode; sha256 " + pa_["store_completion_sha256"][:12])
    assert '"n_nofit": ' + str(goals * 3) in st2["note"] and '"n_undecided": 0' in st2["note"] and '"n_foreign_slots": 1' in st2["note"] and '"n_unparsable_rows": 1' in st2["note"]
    assert '"n_target_forwards_last_invocation": 3' in st2["note"]                                                # read for the record, never for completeness
    assert f'"n_covered": {goals * 3}' in st1["note"] and '"n_gaps": 0' in st1["note"] and f'"plan_goals": {goals}' in st1["note"] and '"plan_n_max": 3' in st1["note"]
    with DB.reading(DB.db_path(str(root))) as conn:                                                             # the all-no-fit cell is ALSO certified (judged -> ingested by the fold)
        assert conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (runs[("strongreject_bon", "none")]["run_id"],)).fetchone()[0] == "generated"
        assert conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (runs[("strongreject_bon", "prompted__safety_refusal")]["run_id"],)).fetchone()[0] in ("judged", "ingested")
    # a resumed store missing one draw: intact marker, the manifest says one slot undecided -> refused; no durable manifest -> refused
    pend2 = _register(tmp_path, root, job="808186", job_name=f"f22-stat-{TAG}-4", evals="strongreject_bon"); wave2 = tmp_path / "wave2"
    short = _bon_store(wave2, "none", goals=goals, n_max=3, responded=goals * 3 - 1, refused=0, config=cfg)
    _bon_store(wave2, "prompted__safety_refusal", goals=goals, n_max=3, responded=goals * 3, refused=0, manifest=False, config=cfg)
    r2 = _close(pend2, spool, wave2, job_name=f"f22-stat-{TAG}-4")
    assert r2.returncode == 2, r2.stdout + r2.stderr
    assert f"REFUSED strongreject_bon none" in r2.stdout and "1 of " in r2.stdout and "UNDECIDED" in r2.stdout
    assert "REFUSED strongreject_bon prompted__safety_refusal" in r2.stdout and "no `units` accounting" in r2.stdout
    assert "0 filed, 0 skipped, 2 refused" in r2.stdout
    # a STALE manifest — every manifest and marker field complete, ONE response line deleted from the live ledger
    pend3 = _register(tmp_path, root, job="808187", job_name=f"f22-stat-{TAG}-5", evals="strongreject_bon"); wave3 = tmp_path / "wave3"
    stale = _bon_store(wave3, "none", goals=goals, n_max=3, responded=goals * 3, refused=0, config=cfg)
    lines = (stale / "bon_responses.jsonl").read_text().splitlines(); (stale / "bon_responses.jsonl").write_text("".join(l + "\n" for l in lines[:-1]))
    noplan = _bon_store(wave3, "prompted__safety_refusal", goals=goals, n_max=3, responded=goals * 3, refused=0)   # a marker naming no config: no universe
    r3 = _close(pend3, spool, wave3, job_name=f"f22-stat-{TAG}-5")
    assert r3.returncode == 2, r3.stdout + r3.stderr
    assert "REFUSED strongreject_bon none" in r3.stdout and "1 planned slot(s) have neither a response row nor a proven no-fit" in r3.stdout and "manifest's units are stale" in r3.stdout
    assert "REFUSED strongreject_bon prompted__safety_refusal" in r3.stdout and "names no `config`" in r3.stdout
    assert "0 filed, 0 skipped, 2 refused" in r3.stdout
    assert H.ingest_once(str(root), workers=0, min_free=0).refused == 0                                          # the registration folds; nothing else was filed
    with DB.reading(DB.db_path(str(root))) as conn:
        assert {conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (r["run_id"],)).fetchone()[0] for r in _runs(pend3).values()} == {"running"}


def _judgments(spool: Path, run_id: str):
    return [(p, pr) for t, p, pr in _records(spool, run_id) if t == "judgment"]


def test_an_all_typed_no_fit_bon_cell_is_filed_as_a_no_fit_by_design_judgment_a_mixed_cell_grades_and_an_untyped_empty_cell_refuses(tmp_path, monkeypatch):
    """A BoN cell whose EVERY planned slot is a typed `nofit_proven` refusal (exact prompt length from the tokenize
    endpoint + the minimum completion > the effective window) and which holds no response is a NO-FIT-BY-DESIGN cell: `generated` with 0
    items AND a complete zero-item judgment for the instrument's judge (minted from the seeds) carrying the certificate, so the run is
    judged once and never dispatched (a grader refuses an empty store as incomplete every tick). A cell with SOME no-fits and rows files
    `generated` with its rows only (the grader grades them). A cell with 0 rows whose errors are NOT typed no-fits is refused — a genuine
    failure, never a certificate. Re-running the closer over an already-generated all-no-fit cell files the certificate once."""
    for k in (emit.DISABLED_ENV, emit.SPOOL_ENV, emit.PUSH_SCRIPT_ENV, hooks.EMIT_ENV, hooks.RUN_IDENTITY_ENV):
        monkeypatch.delenv(k, raising=False)
    root = tmp_path / "reg"; _seeded_registry(root); spool = root / "records"; wave = tmp_path / "wave"
    pending = _register(tmp_path, root, job="808185", job_name=f"f22-stat-{TAG}-3", evals="strongreject_bon")
    runs = _runs(pending); goals = int(json.loads(pending.read_text())["runs"][0]["plan"]["n_items_planned"])
    cfg = _bon_plan_config(tmp_path, goals=goals, n_max=3)
    nofit = _bon_store(wave, "prompted__safety_refusal", goals=goals, n_max=3, responded=0, refused=goals * 3, config=cfg)       # every slot a typed no-fit
    with open(nofit / "bon_errors.jsonl", "a") as fh:                            # the live ledger's shape: a torn duplicate of a PROVEN slot with a misspelt source
        fh.write(json.dumps({"goal_id": "jbb_harmful_0", "sample_idx": 0, "trial": 0, "error": "prompt_over_window", "prompt_tokens": 6017, "prompt_tokens_source": "tkenize_endpoint",
                             "effective_window": 4864}) + "\n")                  # (an undecided KIND on a slot another row proves: reported, not fatal)
        fh.write(json.dumps({"goal_id": "jbb_harmful_1", "sample_idx": 0, "trial": 0, "error": "promt_over_window"}) + "\n")   # a torn error spelling
    mixed = _bon_store(wave, "none", goals=goals, n_max=3, responded=goals * 3 - 4, refused=4, config=cfg)                       # rows + four typed no-fits
    r = _close(pending, spool, wave, job_name=f"f22-stat-{TAG}-3")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "1 no-fit-by-design certificate(s)" in r.stdout and "judged strongreject_bon" not in r.stdout and "no-fit-by-design (" in r.stdout
    assert H.ingest_once(str(root), workers=0, min_free=0).refused == 0
    rid_nofit = runs[("strongreject_bon", "prompted__safety_refusal")]["run_id"]; rid_mixed = runs[("strongreject_bon", "none")]["run_id"]
    st, pa = _generated(spool, rid_nofit); assert pa["n_items_generated"] == 0 and pa["store_uri"] == str(nofit)
    js = _judgments(spool, rid_nofit)
    assert [p["status"] for p, _ in js] == ["in_progress", "complete"] and js[-1][0]["n_items_judged"] == 0 and js[-1][0]["store_uri"] == str(nofit)
    assert js[-1][1]["script"].endswith("registry_close_stores")                # the closer's producer script (its dotted module name)
    with DB.reading(DB.db_path(str(root))) as conn:
        state, notes, extra, jid = conn.execute("SELECT lifecycle_state, notes, extra_json, judge_id FROM run WHERE run_id = ?", (rid_nofit,)).fetchone()
        assert state in ("judged", "ingested") and "no-fit-by-design cell" in notes and "prompt 6017-6017 tokens" in notes and '"nofit_certificate"' in extra
        cert = json.loads(extra)["nofit_certificate"]
        assert cert["n_slots"] == goals * 3 and cert["n_planned"] == goals * 3 and cert["n_malformed_rows"] == 2 and cert["windows"] == [4864]   # the torn duplicate (misspelt source, short shape) + torn spelling: counted, harmless
        req = conn.execute("SELECT DISTINCT required_judge_id FROM v_pending_judgment WHERE run_id = ? AND is_instrument_judge = 1", (rid_nofit,)).fetchall()
        assert req == [], req                                                        # graded once: no longer pending for its instrument judge
        assert conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (rid_mixed,)).fetchone()[0] == "generated"
    st2, pa2 = _generated(spool, rid_mixed); assert pa2["n_items_generated"] == goals * 3 - 4 and _judgments(spool, rid_mixed) == []
    assert '"n_nofit": 4' in st2["note"]
    # idempotent: the same closer over the same stores files nothing new
    r2 = _close(pending, spool, wave, job_name=f"f22-stat-{TAG}-3"); assert r2.returncode == 0, r2.stdout + r2.stderr
    assert "0 filed, 2 skipped, 0 refused" in r2.stdout and "no-fit-by-design (" not in r2.stdout and len(_judgments(spool, rid_nofit)) == 2
    # 0 rows whose errors are NOT typed no-fits (no prompt length from the tokenize endpoint): refused, never certified
    pend2 = _register(tmp_path, root, job="808186", job_name=f"f22-stat-{TAG}-4", evals="strongreject_bon"); wave2 = tmp_path / "wave2"
    untyped = _bon_store(wave2, "none", goals=goals, n_max=3, responded=0, refused=goals * 3, config=cfg)
    rows = [json.loads(l) for l in (untyped / "bon_errors.jsonl").read_text().splitlines()]
    (untyped / "bon_errors.jsonl").write_text("".join(json.dumps({k: v for k, v in r.items() if k not in ("prompt_tokens", "prompt_tokens_source")}) + "\n" for r in rows))
    _bon_store(wave2, "prompted__safety_refusal", goals=goals, n_max=3, responded=goals * 3, refused=0, config=cfg)
    r3 = _close(pend2, spool, wave2, job_name=f"f22-stat-{TAG}-4")
    assert r3.returncode == 2 and "REFUSED strongreject_bon none" in r3.stdout and "UNDECIDED" in r3.stdout and "no-fit-by-design (" not in r3.stdout, r3.stdout + r3.stderr
    assert "1 filed, 0 skipped, 1 refused" in r3.stdout
    # a planned slot with an old proof AND a later VALID contradicting row (a differently-stamped retry, or a row of
    # another kind) is UNDECIDED — never "first proof wins": the cell is refused by name and stays running, no certificate, not even generated
    pend3 = _register(tmp_path, root, job="808188", job_name=f"f22-stat-{TAG}-6", evals="strongreject_bon"); wave3 = tmp_path / "wave3"
    contra = _bon_store(wave3, "none", goals=goals, n_max=3, responded=0, refused=goals * 3, config=cfg)
    with open(contra / "bon_errors.jsonl", "a") as fh:
        fh.write(json.dumps(_nofit_row("jbb_harmful_0", 1, 0, effective_window=6000, window=6000, server_window=6000)) + "\n")   # a valid PROVEN retry under another window: another geometry stamp
        fh.write(json.dumps(_nofit_row("jbb_harmful_1", 2, 0, window_reason="inconsistent rejection")) + "\n")                 # a valid row of another kind on a proven slot
        torn = _nofit_row("jbb_harmful_2", 0, 0, server_window=None); del torn["window"]; torn["prompt_tokns"] = 6017          # a torn duplicate: canonical keys missing/misspelt
        fh.write(json.dumps(torn) + "\n")
        fh.write(json.dumps(_nofit_row("jbb_harmful_3", 0, 0, server_window=None)) + "\n")                                    # a VALID row with an optional window field None vs 4864: another stamp, no exception
        fh.write(json.dumps(_nofit_row("jbb_harmful_4", 0, 0, requested_max_tokens=4096)) + "\n")                              # the climb's refusal of the SAME prompt / window at another completion budget: the same stamp, certified
    _bon_store(wave3, "prompted__safety_refusal", goals=goals, n_max=3, responded=goals * 3, refused=0, config=cfg)
    r4 = _close(pend3, spool, wave3, job_name=f"f22-stat-{TAG}-6")
    assert r4.returncode == 2 and "REFUSED strongreject_bon none" in r4.stdout and "3 of " in r4.stdout and "UNDECIDED under the per-slot rule" in r4.stdout, r4.stdout + r4.stderr
    assert "first jbb_harmful_0/1: 2 geometry stamps" in r4.stdout and "no-fit-by-design (" not in r4.stdout and "1 filed, 0 skipped, 1 refused" in r4.stdout and "3 of " in r4.stdout, r4.stdout
    assert H.ingest_once(str(root), workers=0, min_free=0).refused == 0
    with DB.reading(DB.db_path(str(root))) as conn:
        assert conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (_runs(pend3)[("strongreject_bon", "none")]["run_id"],)).fetchone()[0] == "running"
    # the read-only audit names the same two slots
    r5 = subprocess.run([sys.executable, str(CLOSER), "--audit-nofit", str(contra), "--pending", str(pend3), "--spool", str(spool)], capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=str(REPO / "src")))
    assert r5.returncode == 0, r5.stdout + r5.stderr
    audit = json.loads(r5.stdout)["audit"]; assert audit["n_undecided"] == 3 and audit["n_certified"] == goals * 3 - 3 and {u["goal_id"] for u in audit["undecided"]} == {"jbb_harmful_0", "jbb_harmful_1", "jbb_harmful_3"}
    assert audit["n_malformed_rows"] == 1                                            # the torn duplicate decided nothing (jbb_harmful_2/0 stays certified)
    whys = [u["why"] for u in audit["undecided"]]                                    # geometry (wider window), kinds (inconsistent), geometry (None vs 4864 server_window)
    assert whys[0].startswith("2 geometry stamps") and whys[1].startswith("row kinds ['inconsistent_rejection', 'nofit_proven']") and whys[2].startswith("2 geometry stamps") and "None" in whys[2], whys
    assert not any(u["goal_id"] == "jbb_harmful_4" for u in audit["undecided"])      # a different completion budget is not a different geometry


def test_padded_static_units_file_under_the_padding_lanes_root(tmp_path, monkeypatch):
    """The padded static ladder (809290 p16384 / 809293 p32768 / p65536): one job per level, the static layout under the lane's own root —
    `<wave root>/static_padding/transcripts_results/pad_static_<model>_p<pad>/<eval>/<object>/shardNN` — registered with the level's padding
    block; the closer reads the level from the registration (extra.padding / the padstat-<key>-p<pad> job name) and judges the store with
    the static predicate (three shards here, as the job ran). A level whose store is one response short refuses."""
    for k in (emit.DISABLED_ENV, emit.SPOOL_ENV, emit.PUSH_SCRIPT_ENV, hooks.EMIT_ENV, hooks.RUN_IDENTITY_ENV):
        monkeypatch.delenv(k, raising=False)
    root = tmp_path / "reg"; _seeded_registry(root); spool = root / "records"; wave = tmp_path / "wave"
    pending = _register(tmp_path, root, job="809290", job_name=f"padstat-{TAG}-p16384", evals="pasttense")
    # the GENERATED padding config the job registers (make_padding_configs: its own private_run; the key, not the registry model id)
    key = "qwen35_4b_it"                                              # deliberately NOT the registry model id (qwen35_4b)
    cfgdir = tmp_path / "configs" / "sdfbase" / "static_padding"; cfgdir.mkdir(parents=True)   # absolute paths: nothing is written into the worktree
    cfg_path = cfgdir / f"sdfbase_full22_{key}.padstatic_p16384.yaml"; cfg_rel = str(cfg_path)
    cfg_path.write_text(yaml.safe_dump({"private_run": f"pad_static_{key}_p16384", "out_dir": f"pad_static_{key}_p16384"}))
    doc = json.loads(pending.read_text())
    for r in doc["runs"]:                                            # what jobplan.padding_stamp writes under extra.padding: `tokens` + the contract keys
        r["plan"]["extra"]["padding"] = str({"tokens": 16384, "filler_sha256": "0" * 64, "cap": 16384}); r["plan"]["extra"]["padding_identity_sha256"] = "1" * 64
        r["plan"]["extra"]["config"] = cfg_rel
    pending.write_text(json.dumps(doc))
    try:
        runs = _runs(pending); plan = runs[("pasttense", "none")]["plan"]; planned, iset = plan["n_items_planned"], plan["item_set_id"]
        goals, variants = (int(x) for x in iset.rsplit("_", 1)[1].split("x"))
        pad_root = wave / "static_padding" / "transcripts_results"
        def pad_store(obj, **kw):
            d = pad_root / f"pad_static_{key}_p16384" / "pasttense" / obj
            tmp = _static_store(tmp_path / "scratch" / obj, 0, "pasttense", obj, nshards=3, goals=goals, variants=variants, **kw)
            import shutil; shutil.copytree(tmp, d, dirs_exist_ok=True); return d
        full = pad_store("none")
        short = pad_store("prompted__safety_refusal")
        lines = (short / "shard02" / "responses.jsonl").read_text().splitlines(); (short / "shard02" / "responses.jsonl").write_text("".join(l + "\n" for l in lines[:-1]))
        r = _close(pending, spool, wave, "--nshards", "3", job_name=f"padstat-{TAG}-p16384")
        assert r.returncode == 2, r.stdout + r.stderr
        assert "generated pasttense none" in r.stdout and "(padding)" in r.stdout and f"store_uri={full}" in r.stdout
        assert "REFUSED pasttense prompted__safety_refusal" in r.stdout and "1 filed, 0 skipped, 1 refused" in r.stdout
        assert H.ingest_once(str(root), workers=0, min_free=0).refused == 0
        st, pa = _generated(spool, runs[("pasttense", "none")]["run_id"])
        assert pa["store_uri"] == str(full) and pa["n_items_generated"] == planned and '"layout": "padding"' in st["note"] and '"padding_tokens": 16384' in st["note"]
        assert f'"private_run": "pad_static_{key}_p16384"' in st["note"]
        with DB.reading(DB.db_path(str(root))) as conn:
            assert conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (runs[("pasttense", "none")]["run_id"],)).fetchone()[0] == "generated"
            assert conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (runs[("pasttense", "prompted__safety_refusal")]["run_id"],)).fetchone()[0] == "running"
        # the STEERED shape (stage.py: padobj-<tag>-p<pad>; the config's own namespace pad_static_<key>_obj_p<pad>) under the same lane root
        pend2 = _register(tmp_path, root, job="809300", job_name=f"padobj-{TAG}-p16384", evals="pasttense")
        cfg2_rel = str(cfgdir / f"sdfbase_full22_{key}.padstatic_obj_p16384.yaml")
        Path(cfg2_rel).write_text(yaml.safe_dump({"private_run": f"pad_static_{key}_obj_p16384"}))
        doc2 = json.loads(pend2.read_text())
        for r_ in doc2["runs"]:
            r_["plan"]["extra"]["padding"] = str({"tokens": 16384, "filler_sha256": "0" * 64}); r_["plan"]["extra"]["config"] = cfg2_rel
        pend2.write_text(json.dumps(doc2)); runs2 = _runs(pend2)
        import shutil
        for obj in OBJECTS:
            shutil.copytree(_static_store(tmp_path / "scratch2" / obj, 0, "pasttense", obj, nshards=3, goals=goals, variants=variants), pad_root / f"pad_static_{key}_obj_p16384" / "pasttense" / obj, dirs_exist_ok=True)
        r2 = _close(pend2, spool, wave, "--nshards", "3", job_name=f"padobj-{TAG}-p16384")
        assert r2.returncode == 0 and "2 filed, 0 skipped, 0 refused" in r2.stdout, r2.stdout + r2.stderr
        st2, pa2 = _generated(spool, runs2[("pasttense", "none")]["run_id"]); assert pa2["store_uri"] == str(pad_root / f"pad_static_{key}_obj_p16384" / "pasttense" / "none")
        # a registration whose config cannot be read refuses by name (never a guessed directory) — a fresh registration (its own run ids)
        pend3 = _register(tmp_path, root, job="809301", job_name=f"padobj-{TAG}-p32768", evals="pasttense")
        doc3 = json.loads(pend3.read_text())
        for r_ in doc3["runs"]:
            r_["plan"]["extra"]["padding"] = str({"tokens": 32768, "filler_sha256": "0" * 64}); r_["plan"]["extra"]["config"] = str(cfgdir / "absent.yaml")
        pend3.write_text(json.dumps(doc3))
        r3 = _close(pend3, spool, wave, "--nshards", "3", job_name=f"padobj-{TAG}-p32768")
        assert r3.returncode == 2 and f"the registered config {cfgdir / 'absent.yaml'} is not on disk (tried {cfgdir / 'absent.yaml'})" in r3.stdout, r3.stdout + r3.stderr
        assert "0 filed, 0 skipped, 2 refused" in r3.stdout and "could not be read" not in r3.stdout   # closer-fix-2: an ABSENT config names the paths tried; "could not be read" is the unreadable-file branch
    finally:
        pass                                                                     # every fixture lives under tmp_path


def test_misalign_agentic_cells_file_one_run_per_behaviour_against_the_stores_frozen_plan(tmp_path, monkeypatch):
    """The agentic unit's layout: the misalign runner writes ONE cell file per Petri harness cell (`prompted__safety` for the wave
    object `prompted__safety_refusal`) holding both behaviours; the registration has one run per (agentic_<behaviour>, object) and no
    n_items_planned (the agentic item sets record no size). A finished cell files each behaviour's run with that behaviour's rows as
    n_items_generated, judged against the store's frozen plan (meta: items x n_samples, every item of the behaviour with n_samples rows);
    a cell cut one draw short refuses both; an unfinished cell refuses."""
    for k in (emit.DISABLED_ENV, emit.SPOOL_ENV, emit.PUSH_SCRIPT_ENV, hooks.EMIT_ENV, hooks.RUN_IDENTITY_ENV):
        monkeypatch.delenv(k, raising=False)
    root = tmp_path / "reg"; _seeded_registry(root); spool = root / "records"; wave = tmp_path / "exp" / "sdfbase_full22_native"; wave.mkdir(parents=True)
    mis = tmp_path / "exp" / "misalign" / TAG
    pending = _register(tmp_path, root, job="808433", job_name=f"f22-misalign-{TAG}", evals="agentic_leaking,agentic_murder")
    runs = _runs(pending); assert set(runs) == {(e, o) for e in ("agentic_leaking", "agentic_murder") for o in OBJECTS}
    items = {"leaking": ["L1", "L2", "L3"], "murder": ["M1", "M2"]}
    _misalign_cell(mis, "none", items=items, n_samples=4)
    _misalign_cell(mis, "prompted__safety", items=items, n_samples=4)
    r = _close(pending, spool, wave, job_name=f"f22-misalign-{TAG}")
    assert r.returncode == 0, r.stdout + r.stderr
    assert H.ingest_once(str(root), workers=0, min_free=0).refused == 0
    st, pa = _generated(spool, runs[("agentic_leaking", "prompted__safety_refusal")]["run_id"])
    assert pa["store_uri"] == str(mis / "agentic" / "prompted__safety") and pa["n_items_generated"] == 12 and pa["store_completion_artifact_uri"] == str(mis / "agentic" / "prompted__safety" / "meta.json")
    assert pa["decode_sent_json"] == {"temperature": 1.0} and pa["decode_temperature_sent"] == 1.0 and pa["extra_json"]["decode_evidence"]["family"] == "misalign"      # decode-recording: meta.json#sampling
    assert pa["extra_json"]["decode_evidence"]["witnesses"] == {"sampler": "t1", "finished": True, "params_sampling_agrees": True} and pa["field_confidence_json"]["decode_sent_json"]["source"].startswith(str(mis / "agentic" / "prompted__safety" / "meta.json") + "#sampling; sha256 ")
    assert '"behaviour": "leaking"' in st["note"] and '"behaviour_items": 3' in st["note"] and "store meta" in st["note"]
    st2, pa2 = _generated(spool, runs[("agentic_murder", "none")]["run_id"]); assert pa2["n_items_generated"] == 8 and pa2["store_uri"] == str(mis / "agentic" / "none")
    with DB.reading(DB.db_path(str(root))) as conn:
        assert {conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (runs[k]["run_id"],)).fetchone()[0] for k in runs} == {"generated"}
    # cut short / unfinished: refused by name
    pend2 = _register(tmp_path, root, job="808434", job_name=f"f22-misalign-{TAG}", evals="agentic_leaking,agentic_murder"); mis2 = tmp_path / "exp2" / "misalign" / TAG
    wave2 = tmp_path / "exp2" / "sdfbase_full22_native"; wave2.mkdir(parents=True)
    _misalign_cell(mis2, "none", items=items, n_samples=4, drop=1)
    _misalign_cell(mis2, "prompted__safety", items=items, n_samples=4, finished=False)
    r2 = _close(pend2, spool, wave2, job_name=f"f22-misalign-{TAG}")
    assert r2.returncode == 2 and "0 filed, 0 skipped, 4 refused" in r2.stdout, r2.stdout + r2.stderr
    assert "19 row(s) / 19 distinct slot(s) / present 19 vs items 5 x n_samples 4 = 20" in r2.stdout and "has not finished" in r2.stdout


def test_controls_store_is_the_final_rung_and_an_unfinished_ladder_refuses(tmp_path, monkeypatch):
    for k in (emit.DISABLED_ENV, emit.SPOOL_ENV, emit.PUSH_SCRIPT_ENV, hooks.EMIT_ENV, hooks.RUN_IDENTITY_ENV):
        monkeypatch.delenv(k, raising=False)
    root = tmp_path / "reg"; _seeded_registry(root); spool = root / "records"; wave = tmp_path / "wave"
    pending = _register(tmp_path, root, job="33333", job_name=f"f22s-{TAG}-{TAG}-controls_t1", evals="xstest,or_bench_hard")
    runs = _runs(pending); assert len(runs) == 4
    px = runs[("xstest", "none")]["plan"]["n_items_planned"]; po = runs[("or_bench_hard", "none")]["plan"]["n_items_planned"]
    n = {"xstest": max(px, 450), "or_bench_hard_1k": po}
    # none: 512 escalated, 2048 final; prompted: only 512, still escalating (unfinished)
    _controls_cell(wave, 1, "none", 512, escalate={"xstest": False, "or_bench_hard_1k": True}, n=n, rows=n)
    final = _controls_cell(wave, 1, "none", 2048, escalate={"xstest": False, "or_bench_hard_1k": False}, n=n, rows=n)
    _controls_cell(wave, 1, "prompted__safety_refusal", 512, escalate={"xstest": True, "or_bench_hard_1k": True}, n=n, rows=n)
    r = _close(pending, spool, wave, job_name=f"f22s-{TAG}-{TAG}-controls_t1")
    assert r.returncode == 2 and "2 filed, 0 skipped, 2 refused of 4" in r.stdout, r.stdout + r.stderr
    for e in ("xstest", "or_bench_hard"):
        st, pa = _generated(spool, runs[(e, "none")]["run_id"])
        assert pa["store_uri"] == str(final) and pa["store_completion_artifact_uri"] == str(final / "cap_ladder.json") and pa["n_items_generated"] == runs[(e, "none")]["plan"]["n_items_planned"]
        assert '"final_rung": 2048' in st["note"]
        # decode-recording: the final-rung cell's manifest, read for the run's OWN control (xstest -> xstest, or_bench_hard -> or_bench_hard_1k), gen_stats agreeing
        assert pa["decode_sent_json"] == VENDOR_SIX_F and pa["decode_top_k_sent"] == 20 and pa["field_confidence_json"]["decode_sent_json"]["source"].startswith(str(final / "manifest.json") + "#resolved_config.gen; sha256 ")
        dev = pa["extra_json"]["decode_evidence"]; w = dev["witnesses"]
        assert dev["family"] == "controls" and w["spelling"] == {"xstest": "xstest", "or_bench_hard": "or_bench_hard_1k"}[e] and w["gen_stats_decode_agrees"] is True and w["escalated_from_cap"] == 512
        assert dev["unmodelled_keys"] == {"resolved_config.gen": ["arm", "batch_size", "do_sample", "escalated_from", "max_new_tokens", "seed_base", "trial"]}
        assert _generated(spool, runs[(e, "prompted__safety_refusal")]["run_id"]) == (None, None)
    assert "escalate=false" in r.stdout
    sys.path.insert(0, str(REPO / "scripts" / "registry"))
    import grade_daemon as GD
    assert GD.store_root_of("controls", str(final))[0] == str(wave / "sampled" / "controls_cap2048")


def _multiturn_store(wave: Path, private_run: str, attack: str, obj: str, *, nshards: int, goals: int, shards=None, error_goal_in_shard=None,
                     drop_response_goal_in_shard=None, duplicate_goal_into_shard=None, goal_ids=None, target_decode=VENDOR_SIX, second_decode_in_shard=None) -> Path:
    """A multi-turn cell as jailbreak_run's drivers leave it: `<wave>/transcripts_results/<private_run>/<attack>/<cell>/shardNN/` with
    progress.jsonl (one row per goal: goal_id, outcome `AttackOutcome.SUCCESS|FAILURE`, served_window), responses.jsonl (one row per persisted
    target reply: goal_id, order, response), served_target.json; the 259 goals dealt round-robin over the shards (29/29/29/29/29/29/29/28/28)."""
    d = wave / "transcripts_results" / private_run / attack / obj
    all_ids = list(goal_ids) if goal_ids is not None else [f"goal_{g:03d}" for g in range(goals)]
    for s in (shards if shards is not None else range(nshards)):
        sd = d / f"shard{s:02d}"; sd.mkdir(parents=True, exist_ok=True)
        gids = [gid for g, gid in enumerate(all_ids) if g % nshards == s]
        if duplicate_goal_into_shard == s:
            gids.append([gid for g, gid in enumerate(all_ids) if g % nshards == (s + 1) % nshards][0])   # a goal the next shard owns
        prog, resp = [], []
        for i, gid in enumerate(gids):
            outcome = "error:PromptOverWindow" if (error_goal_in_shard == s and i == 0) else ("AttackOutcome.SUCCESS" if i % 3 else "AttackOutcome.FAILURE")
            td = (dict(target_decode, temperature=1.0) if (second_decode_in_shard == s and i == 0) else dict(target_decode)) if target_decode is not None else None
            prog.append({"goal_id": gid, "category": "x", "n_target_forwards": 3, "outcome": outcome, "turns": 3, "capped": False, "served_window": 49152,
                         **({"target_decode": td} if td is not None else {})})
            if not (drop_response_goal_in_shard == s and i == 0):
                for k in range(3):
                    resp.append({"goal_id": gid, "order": k, "response": f"reply {k}", "serving_engine": "lora_native"})
        (sd / "progress.jsonl").write_text("".join(json.dumps(r) + "\n" for r in prog))
        (sd / "responses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in resp))
        (sd / "served_target.json").write_text(json.dumps({"serving_engine": "lora_native", "max_model_len": 49152, "n_sampled_requests": len(resp),
                                                            **({"decode": dict(target_decode, do_sample=True, cap=4096, arm="vendor_recommended", seed_base=20260909, trial=0,
                                                                              source=MT_POLICY_ID, source_kind="decode_policy_registry")} if target_decode is not None else {})}))
        (sd / "attacker_engine.json").write_text(json.dumps({"attack": attack, "attacker_engine": "vllm",
                                                             **({"target_decode": {"mode": "vendor_preset", "source": MT_POLICY_ID, "preset": dict(target_decode)}} if target_decode is not None else {})}))
    return d


def test_multiturn_cell_files_the_full_goal_set_from_one_nine_shard_store_and_refuses_anything_short(tmp_path, monkeypatch):
    """A multi-turn (attack, cell) store is complete when every one of the job plan's shards holds the three files, no goal's outcome is
    one the grader refuses (`error:*`) or excludes (`refused:*`), every goal sits in ONE shard with a response row, and the distinct goals are the
    registration's 259. The private run is the REGISTERED config's; a consolidated cell's provenance (consolidation.json) rides the note."""
    for k in (emit.DISABLED_ENV, emit.SPOOL_ENV, emit.PUSH_SCRIPT_ENV, hooks.EMIT_ENV, hooks.RUN_IDENTITY_ENV):
        monkeypatch.delenv(k, raising=False)
    root = tmp_path / "reg"; _seeded_registry(root); spool = root / "records"; wave = tmp_path / "exp" / "reproduce" / TAG / "waves" / "mt-u0-cluster_a"; wave.mkdir(parents=True)
    pending = _register(tmp_path, root, job="810070", job_name=f"f22-att-{TAG}-g012-crescendo", evals="crescendo,pair")
    runs = _runs(pending); assert set(runs) == {(e, o) for e in ("crescendo", "pair") for o in OBJECTS}
    plan = runs[("crescendo", "none")]["plan"]; planned = plan["n_items_planned"]; assert planned == 259 and plan["item_set_id"] == "multiturn_goals_259"
    assert plan["extra"]["seed_instrument_id"].startswith("multiturn_crescendo_")
    # the private run is the REGISTERED config's `private_run` (extra.config = configs/sdfbase_full22_<TAG>.yaml, the repo's real wave config,
    # which names it; target_attacks.sbatch's PRIVATE_RUN is the same key)
    prun = yaml.safe_load((REPO / "configs" / f"sdfbase_full22_{TAG}.yaml").read_text())["private_run"]; assert prun == f"sdfbase_full22_{TAG}"
    # the EXACT goal set: the closer recomputes the registered config's eval goal set with
    # the ACTIVE checkout's resolver (jailbreak_run._eval_goals — the twin grader's own ids; the grading host resolves them from the HF cache, as the
    # twin's tests require) and holds the store's ids equal to it; an ids file is a cross-check only
    import hashlib
    sys.path.insert(0, str(REPO / "scripts")); import jailbreak_run as jr
    real_ids = [g.goal_id for g in jr._eval_goals(jr._load_cfg(str(REPO / "configs" / f"sdfbase_full22_{TAG}.yaml")), confirmatory=True)]
    assert len(real_ids) == 259 == len(set(real_ids))
    cfg_sha = hashlib.sha256((REPO / "configs" / f"sdfbase_full22_{TAG}.yaml").read_bytes()).hexdigest()
    ids_file = tmp_path / "expected_goal_ids.json"
    r = _close(pending, spool, wave, "--emit-expected-goal-ids", str(ids_file))
    assert r.returncode == 0 and "259 eval goal ids" in r.stdout, r.stdout + r.stderr
    emitted = json.loads(ids_file.read_text())
    assert emitted["config"] == f"configs/sdfbase_full22_{TAG}.yaml" and emitted["config_sha256"] == cfg_sha and set(emitted["goal_ids"]) == set(real_ids)
    assert len(emitted["resolver_sha256"]) == 64 and emitted["resolver_files"] == ["scripts/jailbreak_run.py", "src/sisteer/eval/jailbreak_data.py"] and len(emitted["goal_ids_sha256"]) == 64
    assert emitted["datasets"]["jbb_revision"] and emitted["datasets"]["harmbench_revision"]
    IDS = ("--expected-goal-ids", str(ids_file))
    good = _multiturn_store(wave, prun, "crescendo", "none", nshards=9, goals=259, goal_ids=real_ids)              # complete, one cell from nine shards
    (good / "consolidation.json").write_text(json.dumps({"copied_utc": "2026-10-01T05:56:03Z", "note": "synthetic", "shards": {
        f"shard{s:02d}": {"cluster": "cluster_b" if s in (3, 4, 7, 8) else "cluster_a", "job_id": str(100 + s), "serve_dir": f"/x/serving/attacks_{s}"} for s in range(9)}}))
    _multiturn_store(wave, prun, "crescendo", "prompted__safety_refusal", nshards=9, goals=259, shards=[0, 1, 2, 5, 6], goal_ids=real_ids)  # one cluster's half only
    _multiturn_store(wave, prun, "pair", "none", nshards=9, goals=259, error_goal_in_shard=4, goal_ids=real_ids)                      # an errored goal
    _multiturn_store(wave, prun, "pair", "prompted__safety_refusal", nshards=9, goals=259, drop_response_goal_in_shard=7, goal_ids=real_ids)   # a goal with no reply
    g4 = real_ids[4 * 1 + 0] if False else [gid for g, gid in enumerate(real_ids) if g % 9 == 4][0]   # pair/none's errored goal: the first id dealt to shard04
    g7 = [gid for g, gid in enumerate(real_ids) if g % 9 == 7][0]                                      # pair/safety's reply-less goal: shard07's first
    assert H.ingest_once(str(root), workers=0, min_free=0).refused == 0
    r = _close(pending, spool, wave, "--nshards", "9", "--dry-run", *IDS, job_name=f"f22-att-{TAG}-g012-crescendo")
    assert r.returncode == 2 and r.stdout.count("DRY-RUN ") == 1 and r.stdout.count("REFUSED ") == 3, r.stdout + r.stderr
    assert "4 of 9 expected shard(s) missing or incomplete: shard03, shard04, shard07, shard08" in r.stdout
    assert f"1 progress row(s) the grader would not grade (shard04:{g4}=error:PromptOverWindow)" in r.stdout
    assert f"1 goal(s) carry no response row (e.g. {g7})" in r.stdout
    assert all(_generated(spool, r_["run_id"]) == (None, None) for r_ in runs.values())
    # an interpreter that cannot import the resolver (the cluster's bare python3 -S: no site-packages) refuses as UNTRUSTED, ids file or not
    r = _close(pending, spool, wave, "--nshards", "9", "--dry-run", *IDS, job_name=f"f22-att-{TAG}-g012-crescendo", bare=True)
    assert r.returncode == 2 and r.stdout.count("REFUSED ") == 4 and r.stdout.count("UNTRUSTED: the eval goal set cannot be recomputed by this interpreter") == 4, r.stdout + r.stderr
    assert "cannot stand in for the resolver" in r.stdout and "DRY-RUN" not in r.stdout
    r = _close(pending, spool, wave, "--nshards", "9", *IDS, job_name=f"f22-att-{TAG}-g012-crescendo")
    assert r.returncode == 2 and "1 filed, 0 skipped, 3 refused of 4" in r.stdout, r.stdout + r.stderr
    st, pa = _generated(spool, runs[("crescendo", "none")]["run_id"])
    assert pa["store_uri"] == str(good) and pa["n_items_generated"] == 259 and pa["store_completion_artifact_uri"] == str(good / "shard00" / "progress.jsonl")
    assert st["evidence_kind"] == "completion_artifact" and "259 planned, item set multiturn_goals_259" in st["note"]
    # decode-recording: the nine shards' progress rows' one target_decode block, served_target.json agreeing, the attacker's registry row as the producer's statement
    assert pa["decode_sent_json"] == VENDOR_SIX_F and pa["decode_temperature_sent"] == 0.7 and pa["field_confidence_json"]["decode_sent_json"]["source"].startswith(str(good / "shard00" / "progress.jsonl") + "#target_decode; sha256 ")
    dev = pa["extra_json"]["decode_evidence"]
    assert dev["family"] == "multiturn" and dev["producer_policy_id"] == MT_POLICY_ID and dev["witnesses"]["shards"] == [f"shard{i:02d}" for i in range(9)]
    assert dev["witnesses"]["rows_with_target_decode"] == 259 and dev["witnesses"]["distinct_blocks"] == 1 and dev["witnesses"]["served_target_decode_agrees"] is True
    assert "decode_sent" not in st["note"] and "target_decode" not in st["note"]                     # the evidence rides the patch, never the note (its length is pinned below)
    assert '"goals_distinct": 259' in st["note"] and '"layout": "multiturn"' in st["note"] and '"shard03": "cluster_b job 103"' in st["note"], st["note"]
    # the private run is the REGISTERED config's (the repo's configs/sdfbase_full22_<TAG>.yaml names it), read by name, not the tag spelling
    assert f'"private_run": "{prun}"' in st["note"] and f'"private_run_source": "config configs/sdfbase_full22_{TAG}.yaml"' in st["note"], st["note"]
    assert '"AttackOutcome.SUCCESS": 169' in st["note"] and '"goals_per_shard": [29, 29, 29, 29, 29, 29, 29, 28, 28]' in st["note"], st["note"]
    assert '"goals_expected": 259' in st["note"] and '"goals_missing": 0' in st["note"] and '"source": "resolver in-process"' in st["note"] and f'"config_sha256": "{cfg_sha[:12]}"' in st["note"], st["note"]
    assert '"cross_check": {"agrees": true' in st["note"] and '"resolver_sha256": "' in st["note"], st["note"]
    assert len(st["note"]) < 2000, len(st["note"])                                              # the whole fact block survives the note cap
    # the second cluster's shards arrive (the one-time copy): the SAME command files that run; the two defective pair cells still refuse
    _multiturn_store(wave, prun, "crescendo", "prompted__safety_refusal", nshards=9, goals=259, shards=[3, 4, 7, 8], goal_ids=real_ids)
    r = _close(pending, spool, wave, "--nshards", "9", job_name=f"f22-att-{TAG}-g012-crescendo")                    # no file: the resolver alone
    assert "1 filed, 1 skipped, 2 refused of 4" in r.stdout and r.stdout.count("already generated") == 1, r.stdout + r.stderr
    # a RESUMED goal: the latest progress row decides (jailbreak_run._load_progress) — pair/none's errored goal regenerated appends a row
    resumed = wave / "transcripts_results" / prun / "pair" / "none" / "shard04" / "progress.jsonl"
    with resumed.open("a") as fh:
        fh.write(json.dumps({"goal_id": g4, "outcome": "AttackOutcome.FAILURE", "turns": 3, "capped": False, "served_window": 49152}) + "\n")
    r = _close(pending, spool, wave, "--nshards", "9", "--evals", "pair", *IDS, job_name=f"f22-att-{TAG}-g012-crescendo")
    assert "1 filed, 2 skipped, 1 refused of 4" in r.stdout, r.stdout + r.stderr
    st_p, _pa_p = _generated(spool, runs[("pair", "none")]["run_id"])
    assert '"progress_rows_superseded_by_a_later_row": 1' in st_p["note"] and '"goals_not_graded": 0' in st_p["note"], st_p["note"]
    # a goal present in two shards refuses by name (an ambiguous store, never a silent merge)
    _multiturn_store(wave, prun, "pair", "prompted__safety_refusal", nshards=9, goals=259, duplicate_goal_into_shard=2, goal_ids=real_ids)
    g3 = [gid for g, gid in enumerate(real_ids) if g % 9 == 3][0]
    r = _close(pending, spool, wave, "--nshards", "9", "--evals", "pair", "--dry-run", job_name=f"f22-att-{TAG}-g012-crescendo")
    assert f"1 goal(s) appear in two shards (e.g. {g3} in shard02 and shard03)" in r.stdout and r.stdout.count("already generated") == 1, r.stdout + r.stderr
    # 259 ids with ONE substituted (same count, every file and reply in place): refused by the SET, naming the missing and the foreign id
    sub = real_ids[:100] + ["goal_from_another_revision"] + real_ids[101:]
    _multiturn_store(wave, prun, "pair", "prompted__safety_refusal", nshards=9, goals=259, goal_ids=sub)
    r = _close(pending, spool, wave, "--nshards", "9", "--evals", "pair", "--dry-run", job_name=f"f22-att-{TAG}-g012-crescendo")
    assert f"the goal set differs from the registered config's eval set (1 missing: {real_ids[100]}; 1 foreign: goal_from_another_revision)" in r.stdout, r.stdout + r.stderr
    # the cross-check file: another config's sha, a STALE emitter (another resolver implementation), an EDITED id
    # list, and a list that disagrees with the recomputed set each refuse by name — the resolver's recomputation is the authority
    base = json.loads(ids_file.read_text())
    bad = tmp_path / "other_cfg.json"; bad.write_text(json.dumps({**base, "config_sha256": "0" * 64}))
    r = _close(pending, spool, wave, "--nshards", "9", "--evals", "pair", "--dry-run", "--expected-goal-ids", str(bad), job_name=f"f22-att-{TAG}-g012-crescendo")
    assert f"certifies config 'configs/sdfbase_full22_{TAG}.yaml' @ 000000000000, the registration's is 'configs/sdfbase_full22_{TAG}.yaml' @ {cfg_sha[:12]}" in r.stdout, r.stdout
    stale = tmp_path / "stale.json"; stale.write_text(json.dumps({**base, "resolver_sha256": "f" * 64}))
    r = _close(pending, spool, wave, "--nshards", "9", "--evals", "pair", "--dry-run", "--expected-goal-ids", str(stale), job_name=f"f22-att-{TAG}-g012-crescendo")
    assert "was written by resolver 'jailbreak_run._eval_goals(confirmatory=True)' @ ffffffffffff, the active checkout's is" in r.stdout and "stale or foreign emitter" in r.stdout, r.stdout
    edited = tmp_path / "edited.json"; edited.write_text(json.dumps({**base, "goal_ids": base["goal_ids"][:100] + ["goal_from_another_revision"] + base["goal_ids"][101:]}))
    r = _close(pending, spool, wave, "--nshards", "9", "--evals", "pair", "--dry-run", "--expected-goal-ids", str(edited), job_name=f"f22-att-{TAG}-g012-crescendo")
    assert "goal_ids do not hash to the file's goal_ids_sha256 — an edited list" in r.stdout, r.stdout
    import hashlib as _h
    sub_ids = base["goal_ids"][:100] + ["goal_from_another_revision"] + base["goal_ids"][101:]
    rehashed = tmp_path / "rehashed.json"; rehashed.write_text(json.dumps({**base, "goal_ids": sub_ids, "goal_ids_sha256": _h.sha256(json.dumps(sub_ids, separators=(",", ":")).encode()).hexdigest()}))
    r = _close(pending, spool, wave, "--nshards", "9", "--evals", "pair", "--dry-run", "--expected-goal-ids", str(rehashed), job_name=f"f22-att-{TAG}-g012-crescendo")
    assert "disagrees with the recomputed eval set (1 missing, 1 foreign: ['goal_from_another_revision'])" in r.stdout, r.stdout
    # the fold: the three complete runs enter the pending-judgment worklist with the cell store, which the daemon's multiturn layout resolves
    assert H.ingest_once(str(root), workers=0, min_free=0).refused == 0
    with DB.reading(DB.db_path(str(root))) as conn:
        pend = {x["run_id"]: x for x in P.pending(conn, None, "sdfbase_full22", run_id=tuple(v["run_id"] for v in runs.values()))}
        assert set(pend) == {runs[("crescendo", "none")]["run_id"], runs[("crescendo", "prompted__safety_refusal")]["run_id"], runs[("pair", "none")]["run_id"]}
        assert pend[runs[("crescendo", "none")]["run_id"]]["pending_reason"] == "no_judgment" and pend[runs[("crescendo", "none")]["run_id"]]["store_uri"] == str(good)
        assert conn.execute("SELECT lifecycle_state, n_items_generated FROM run WHERE run_id=?", (runs[("crescendo", "none")]["run_id"],)).fetchone()[:2] == ("generated", 259)
        assert conn.execute("SELECT lifecycle_state FROM run WHERE run_id=?", (runs[("pair", "prompted__safety_refusal")]["run_id"],)).fetchone()[0] == "running"
    sys.path.insert(0, str(REPO / "scripts" / "registry"))
    import grade_daemon as GD
    assert GD.store_root_of("multiturn", str(good)) == (str(wave / "transcripts_results"), prun)
    assert GD.target_for("multiturn", prun, TAG) == TAG and GD.config_for("multiturn", configs_root="/c", private_run=prun) == f"/c/{prun}.yaml"
    # a registration naming no config falls back to the tag spelling; one naming an unreadable config refuses by name
    sys.path.insert(0, str(CLOSER.parent)); import registry_close_stores as RCS
    assert RCS.multiturn_private_run({"plan": {"extra": {}}}, TAG, repo_root=str(REPO)) == (None, prun, "sdfbase_full22_<tag>")
    why, got, _src = RCS.multiturn_private_run({"plan": {"extra": {"config": "configs/absent.yaml"}}}, TAG, repo_root=str(REPO))
    assert got is None and "configs/absent.yaml is not on disk (tried " in why and str(REPO / "configs" / "absent.yaml") in why   # closer-fix-2: absent = the paths tried, by name


def test_the_jobs_exit_trap_closer_refuses_untrusted_under_the_nodes_bare_python_and_files_under_the_driver_venv(tmp_path, monkeypatch):
    """The EXIT trap's REAL cleanup command — common.sh's f22_registry_close_stores — runs `${F22_PY:-python3}`; on a
    compute node the bare python3 cannot import the multi-turn goal resolver (pandas, the dataset loaders), so the closer must REFUSE as
    UNTRUSTED, visibly, and file nothing — never a silent success; under the job's driver venv (F22_PY) the same command files `generated`.
    The bare interpreter is emulated by a python3 on PATH that runs this interpreter with -S (no site-packages: the resolver's imports
    missing, which is what the node's python3 lacks); the venv case is this interpreter itself."""
    for k in (emit.DISABLED_ENV, emit.SPOOL_ENV, emit.PUSH_SCRIPT_ENV, hooks.EMIT_ENV, hooks.RUN_IDENTITY_ENV):
        monkeypatch.delenv(k, raising=False)
    root = tmp_path / "reg"; _seeded_registry(root); spool = root / "records"
    shared = tmp_path / "shared"; wave = shared / "exp" / "reproduce" / TAG / "waves" / "mt-u0-cluster_a"; wave.mkdir(parents=True)
    job = "810071"
    pending = _register(tmp_path, root, job=job, job_name=f"f22-att-{TAG}-g012-crescendo", evals="crescendo")
    (shared / "registry" / "pending").mkdir(parents=True); (shared / "registry" / "pending" / f"{job}.json").write_text(pending.read_text())
    runs = _runs(pending)
    sys.path.insert(0, str(REPO / "scripts")); import jailbreak_run as jr
    real_ids = [g.goal_id for g in jr._eval_goals(jr._load_cfg(str(REPO / "configs" / f"sdfbase_full22_{TAG}.yaml")), confirmatory=True)]
    prun = yaml.safe_load((REPO / "configs" / f"sdfbase_full22_{TAG}.yaml").read_text())["private_run"]
    for obj in OBJECTS:
        _multiturn_store(wave, prun, "crescendo", obj, nshards=9, goals=259, goal_ids=real_ids)
    assert H.ingest_once(str(root), workers=0, min_free=0).refused == 0
    bare = tmp_path / "bare_bin"; bare.mkdir()
    (bare / "python3").write_text(f"#!/bin/bash\nexec {sys.executable} -S \"$@\"\n"); (bare / "python3").chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k not in ("F22_PY", "PYTHONPATH", "SISTEER_REGISTRY_DISABLED")}
    figs = tmp_path / "figs"; figs.mkdir()                       # the trap's RESULT file lives here (the loud-refusal contract appends to it)
    env.update({"SHARED": str(shared), "SLURM_JOB_ID": job, "SLURM_JOB_NAME": f"f22-att-{TAG}-g012-crescendo", "WAVE_ROOT": str(wave), "TAG": TAG,
                "NSHARDS": "9", "WAVE_SHA": "c" * 40, "REPO_DIR": str(REPO), "SISTEER_REGISTRY_SPOOL": str(spool), "FIGS_DIR": str(figs)})
    cmd = ["bash", "-c", f"source {REPO}/scripts/sdfbase/full22/common.sh && f22_registry_close_stores"]
    # (a) the node's bare python3 first on PATH, no F22_PY: the real command refuses UNTRUSTED, visibly, files nothing, and RETURNS the closer's rc
    # (closer-fix-2: f22_result turns it into the job's exit 75; before it the trap's `|| true` returned 0)
    r = subprocess.run(cmd, capture_output=True, text=True, env={**env, "PATH": f"{bare}:{env['PATH']}"}, cwd=str(REPO), timeout=600)
    out = r.stdout + r.stderr
    assert r.returncode == 2, out[-2000:]
    assert out.count("[registry] REFUSED crescendo") == 2 and out.count("UNTRUSTED: the eval goal set cannot be recomputed by this interpreter") >= 2, out[-3000:]
    assert "run the closer where the resolver imports" in out and "generated crescendo" not in out
    # the shell half (26b81a206): ONE loud line naming the interpreter and the cause, the RESULT surface, a registry notice — never `|| true` silence
    assert out.count("[closer] REFUSED rc=2 under python3 (F22_PY unset — the bare interpreter, not the driver venv): job 810071's registered runs stay as they were — REFUSED crescendo") == 1, out[-3000:]
    assert (figs / "RESULT").exists() and f"job={job} name=f22-att-{TAG}-g012-crescendo CLOSER_REFUSED rc=2 interpreter=python3" in (figs / "RESULT").read_text()
    # the notice rides the closer's refused-run list — both runs were refused (UNTRUSTED), so two `--run-id` statements, never the pending file's whole set
    assert out.count("[registry] state running: wrote") >= 2 and "state running: stated for 2 refused run(s) of job 810071" in out and "applied to 2 runs" not in out, out[-3000:]
    side = json.loads((shared / "registry" / "pending" / f"{job}.json.refused_run_ids.json").read_text())
    assert sorted(side["refused_run_ids"]) == sorted(r_["run_id"] for r_ in runs.values()) and side["job_id"] == job
    assert all(_generated(spool, r_["run_id"]) == (None, None) for r_ in runs.values())
    # (b) the driver venv (F22_PY): the same command files generated for both cells
    r = subprocess.run(cmd, capture_output=True, text=True, env={**env, "F22_PY": sys.executable}, cwd=str(REPO), timeout=600)
    out = r.stdout + r.stderr
    assert r.returncode == 0 and out.count("[registry] generated crescendo") == 2 and "REFUSED" not in out and "[closer]" not in out, out[-3000:]
    for obj in OBJECTS:
        st, pa = _generated(spool, runs[("crescendo", obj)]["run_id"])
        assert pa["n_items_generated"] == 259 and pa["store_uri"] == str(wave / "transcripts_results" / prun / "crescendo" / obj) and '"source": "resolver in-process"' in st["note"]


def test_decode_evidence_contradictions_refuse_the_close_by_name_and_an_absent_block_files_the_facts_plus_the_unverified_entry(tmp_path, monkeypatch):
    """decode-recording (the registry owner, 2026-10-13): the closer files the request's observed sampler in the SAME run_patch as the store
    facts, from the store it inspected. A store drawn under two samplers (a multi-turn store whose progress rows state two target_decode
    blocks; a static store whose response rows disagree with served_target.json) is REFUSED by name with nothing filed -- a mixture is never a
    store; a store that records no sampler at all (a legacy static store, a legacy BoN manifest, a controls cell with an empty manifest) files the
    store facts exactly as before plus the `unverified` entry naming why, and no decode column."""
    for k in (emit.DISABLED_ENV, emit.SPOOL_ENV, emit.PUSH_SCRIPT_ENV, hooks.EMIT_ENV, hooks.RUN_IDENTITY_ENV):
        monkeypatch.delenv(k, raising=False)
    root = tmp_path / "reg"; _seeded_registry(root); spool = root / "records"; wave = tmp_path / "wave"
    pending = _register(tmp_path, root, job="12121", job_name=f"f22s-{TAG}-static_base_t0", evals="pasttense,grok420")
    runs = _runs(pending); plan = runs[("pasttense", "none")]["plan"]; planned, iset = plan["n_items_planned"], plan["item_set_id"]
    goals, variants = (int(x) for x in iset.rsplit("_", 1)[1].split("x"))
    # pasttense/none: the rows disagree with served_target.json (a mixture); pasttense/prompted: a legacy store with no decode block at all;
    # grok420/none: a greedy-looking served_target (no decode) but rows that carry one -- served_target states none, the rows state one: absence
    # of the primary block files the facts + the unverified entry (the rows alone are never the evidence); grok420/prompted: the ordinary sampled store
    mixed = _static_store(wave, 0, "pasttense", "none", nshards=9, goals=goals, variants=variants, row_decode=dict(STATIC_DECODE, temperature=0.9))
    legacy = _static_store(wave, 0, "pasttense", "prompted__safety_refusal", nshards=9, goals=goals, variants=variants, decode=None)
    _static_store(wave, 0, "grok420", "none", nshards=9, goals=goals, variants=variants, decode=None, row_decode=STATIC_DECODE)
    good = _static_store(wave, 0, "grok420", "prompted__safety_refusal", nshards=9, goals=goals, variants=variants)
    assert H.ingest_once(str(root), workers=0, min_free=0).refused == 0
    r = _close(pending, spool, wave, "--nshards", "9", "--dry-run", job_name=f"f22s-{TAG}-static_base_t0")
    assert r.returncode == 2 and r.stdout.count("DRY-RUN ") == 3 and r.stdout.count("REFUSED ") == 1, r.stdout + r.stderr
    assert f"REFUSED pasttense none" in r.stdout and "decode evidence:" in r.stdout and "response row(s) in shard00 were sent" in r.stdout and "while served_target.json states" in r.stdout
    assert all(_generated(spool, r_["run_id"]) == (None, None) for r_ in runs.values())
    r = _close(pending, spool, wave, "--nshards", "9", job_name=f"f22s-{TAG}-static_base_t0")
    assert r.returncode == 2 and "3 filed, 0 skipped, 1 refused of 4" in r.stdout, r.stdout + r.stderr
    assert _generated(spool, runs[("pasttense", "none")]["run_id"]) == (None, None)                     # the mixture: nothing filed, the run stays running
    refused = json.loads((pending.parent / f"{pending.stem}.refused_run_ids.json").read_text()) if (pending.parent / f"{pending.stem}.refused_run_ids.json").exists() else None
    if refused is not None:
        assert runs[("pasttense", "none")]["run_id"] in json.dumps(refused)
    for key, store in ((("pasttense", "prompted__safety_refusal"), legacy), (("grok420", "none"), wave / "transcripts_results" / f"sdfbase_full22_{TAG}_sampled_t0" / "grok420" / "none")):
        st, pa = _generated(spool, runs[key]["run_id"])
        assert pa["store_uri"] == str(store) and pa["n_items_generated"] == planned and "decode_sent_json" not in pa and "decode_temperature_sent" not in pa and "extra_json" not in pa, pa
        assert pa["field_confidence_json"] == {"decode_sent_json": {"confidence": "unverified", "reason": pa["field_confidence_json"]["decode_sent_json"]["reason"]}}
        assert "no shard's served_target.json states a decode block" in pa["field_confidence_json"]["decode_sent_json"]["reason"]
    st, pa = _generated(spool, runs[("grok420", "prompted__safety_refusal")]["run_id"])
    assert pa["store_uri"] == str(good) and pa["decode_sent_json"] == VENDOR_SIX_F
    # the fold: the legacy rows read `unrecorded` with the reason in their provenance; the sampled row carries its sent set
    assert H.ingest_once(str(root), workers=0, min_free=0).refused == 0
    with DB.reading(DB.db_path(str(root))) as conn:
        leg = conn.execute("SELECT decode_sent_json, decode_policy_match, field_confidence_json FROM run WHERE run_id=?", (runs[("pasttense", "prompted__safety_refusal")]["run_id"],)).fetchone()
        assert leg[0] == "{}" and leg[1] == "unrecorded" and json.loads(leg[2])["decode_sent_json"]["confidence"] == "unverified"
        smp = conn.execute("SELECT decode_sent_json, decode_temperature_sent, decode_top_k_sent, decode_policy_match FROM run WHERE run_id=?", (runs[("grok420", "prompted__safety_refusal")]["run_id"],)).fetchone()
        assert json.loads(smp[0]) == VENDOR_SIX_F and (smp[1], smp[2]) == (0.7, 20) and smp[3] in ("exact", "no_match")   # the synthetic key's policy rows decide the verdict, never `unrecorded`
    # a multi-turn store whose progress rows state TWO target_decode blocks: refused by name, nothing filed; a legacy unseeded store files the facts + the entry
    wave_mt = tmp_path / "exp" / "reproduce" / TAG / "waves" / "mt-u0-cluster_a"; wave_mt.mkdir(parents=True)
    pend_mt = _register(tmp_path, root, job="12122", job_name=f"f22-att-{TAG}-g012-crescendo", evals="crescendo")
    runs_mt = _runs(pend_mt)
    sys.path.insert(0, str(REPO / "scripts")); import jailbreak_run as jr
    real_ids = [g.goal_id for g in jr._eval_goals(jr._load_cfg(str(REPO / "configs" / f"sdfbase_full22_{TAG}.yaml")), confirmatory=True)]
    prun = yaml.safe_load((REPO / "configs" / f"sdfbase_full22_{TAG}.yaml").read_text())["private_run"]
    _multiturn_store(wave_mt, prun, "crescendo", "none", nshards=9, goals=259, goal_ids=real_ids, second_decode_in_shard=4)
    unseeded = _multiturn_store(wave_mt, prun, "crescendo", "prompted__safety_refusal", nshards=9, goals=259, goal_ids=real_ids, target_decode=None)
    assert H.ingest_once(str(root), workers=0, min_free=0).refused == 0
    r = _close(pend_mt, spool, wave_mt, "--nshards", "9", job_name=f"f22-att-{TAG}-g012-crescendo")
    assert r.returncode == 2 and "1 filed, 0 skipped, 1 refused of 2" in r.stdout, r.stdout + r.stderr
    assert "REFUSED crescendo none" in r.stdout and "the progress rows state 2 distinct target_decode blocks" in r.stdout and "one store, one sampler" in r.stdout
    assert _generated(spool, runs_mt[("crescendo", "none")]["run_id"]) == (None, None)
    st, pa = _generated(spool, runs_mt[("crescendo", "prompted__safety_refusal")]["run_id"])
    assert pa["store_uri"] == str(unseeded) and pa["n_items_generated"] == 259 and "decode_sent_json" not in pa
    assert "no progress row states target_decode" in pa["field_confidence_json"]["decode_sent_json"]["reason"] and "legacy unseeded" in pa["field_confidence_json"]["decode_sent_json"]["reason"]
