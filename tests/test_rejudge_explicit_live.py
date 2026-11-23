"""An explicit run list judged live by MiniMax-M3 end to end, through the public commands: the scope from the runs'
stores, the enumeration, the never-submitted source batch directory, the live run under a root and ceiling of its own, the engine (against
a local mock of OpenRouter's chat completions), and the filing under the judge rows already on record. What it pins:

  * the judge ids such a filing mints are the four batch_or_live rows on record (``filing.MINIMAX_M3_JUDGE_IDS``), recomputed from the
    code's protocol, and -- where the registry is on this machine -- equal to the registry's rows field by field (read only);
  * the per-unit continuous StrongREJECT score lands in ``strongreject`` and the per-(goal, trial) maximum in ``strongreject_best``;
  * a job with no Claude judgment files with null ``re_measures_*`` and its own scope text; C/P/I runs name the judged rung;
  * ``--expect-pinned-judges`` refuses a filing that would stage a judge record or another judge, and the on-disk staging check refuses a
    tampered staging and a filing that left a run it was asked to file with nothing staged; ``--only-run`` delivers one run's files alone;
  * under that flag every explicit-run-list job is held to its run in the registry BEFORE anything is staged (the store it reads, its
    files, its kind), so a scope built without the registry and never verified cannot be filed; a staging whose recorded check found
    a problem is never delivered, with or without the flag;
  * ``--global-root`` makes the engine's global check and the dry run walk the run list's root, and is refused unless it is an existing
    absolute directory; ``init`` (and a C/P/I ``init --refresh``) states the root and both ceilings, none of which has a default.
Offline: synthetic stores and prompts, a mock on 127.0.0.1, a scratch registry root (no database)."""
from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import os
import pathlib
import re
import shutil
import sqlite3
import sys
import time

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from aiohttp import web  # noqa: E402

from sisteer.registry import records as RR  # noqa: E402

from rejudge.common import filing as F  # noqa: E402
from rejudge.common import harvester as H  # noqa: E402
from rejudge.common import live as LV  # noqa: E402
from rejudge.common import merge as M  # noqa: E402
from rejudge.strongreject import build_manifests as SB  # noqa: E402
from rejudge.strongreject import enumerate_units as E  # noqa: E402
from rejudge.strongreject import file_records as SF  # noqa: E402
from rejudge.strongreject import live as SRL  # noqa: E402
from rejudge.strongreject import request as R  # noqa: E402
from rejudge.strongreject import scope_runs as SS  # noqa: E402
from rejudge.terra import build_manifests as TB  # noqa: E402
from rejudge.terra import cpi as C  # noqa: E402
from rejudge.terra import file_records as TF  # noqa: E402
from rejudge.terra import harvest as TH  # noqa: E402
from rejudge.terra import live as TL  # noqa: E402
from rejudge.terra import scope_cpi_runs as SC  # noqa: E402

PINS = F.MINIMAX_M3_JUDGE_IDS
REGISTRY = os.path.join(os.path.expanduser("~"), "registry", "registry.sqlite")
FAST = {"concurrency": 4, "min_concurrency": 1, "flush_n": 50, "flush_s": 0.05, "control_s": 0.05, "progress_s": 0.1,
        "global_check_s": 3600.0, "transport_retries": 1, "transport_backoff_s": 0.001, "transport_backoff_max_s": 0.01,
        "rate_limit_min_s": 0.0, "rate_limit_backoff_s": 0.001, "rate_limit_backoff_max_s": 0.01, "request_timeout_s": 20.0,
        "cooldown_s": 0.05, "cooldown_max_s": 0.2}
GOALS = {"goals": {f"g{i}": {"dataset": "synthetic/goals", "revision": "0" * 40, "text": f"Synthetic goal text {i}"} for i in range(4)},
         "goal_sets": {"static": [f"g{i}" for i in range(4)], "mt": [f"g{i}" for i in range(4)], "bon": [f"g{i}" for i in range(2)]}}
#: per goal, the verdicts (refused, convincing, specific) its replies draw, so the per-goal maximum differs from every unit's score
VERDICTS = [(1, 1, 1), (0, 2, 3), (0, 5, 5), (0, 1, 1)]


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")


def jl(path: pathlib.Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def reply(i: int, j: int) -> str:
    r, c, s = VERDICTS[(i + j) % len(VERDICTS)]
    return f"synthetic reply {i}.{j} MOCK:r{r}c{c}s{s} #{i * 10 + j}"


# ---------------------------------------------------------------------------------------------------------------------------------------
# a mock of OpenRouter's chat completions: the verdict (or C/P/I label) a reply asks for, from its MOCK marker
# ---------------------------------------------------------------------------------------------------------------------------------------
def completion(args, name, cost=0.0001):
    return {"id": f"gen-{time.time_ns()}", "model": "minimax/minimax-m3", "provider": "GMICloud", "object": "chat.completion",
            "choices": [{"index": 0, "finish_reason": "tool_calls", "native_finish_reason": "tool_calls",
                         "message": {"role": "assistant", "content": None, "refusal": None, "tool_calls": [
                             {"id": "c1", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}}],
            "usage": {"prompt_tokens": 400, "completion_tokens": 60, "total_tokens": 460, "cost": cost,
                      "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0}, "completion_tokens_details": {"reasoning_tokens": 0}}}


class Mock:
    def __init__(self):
        self.payloads = []

    async def handle(self, request):
        payload = await request.json()
        self.payloads.append(payload)
        content = " ".join(str(m.get("content")) for m in payload["messages"])
        m = re.search(r"MOCK:r(\d)c(\d)s(\d) #\d+", content)
        if m:
            args = {"reasoning": "synthetic", "refused": int(m.group(1)), "convincing": int(m.group(2)), "specific": int(m.group(3))}
            return web.json_response(completion(args, "StrongREJECTRating"))
        m = re.search(r"MOCKLABEL:(\w+) #\d+", content)
        return web.json_response(completion({"reasoning": "synthetic", "label": m.group(1)}, C.TOOL_NAME))


def run_engine(make_runner, mock):
    async def go():
        app = web.Application()
        app.router.add_post("/api/v1/chat/completions", mock.handle)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            r = make_runner(dict(FAST, base_url=f"http://127.0.0.1:{port}/api/v1"))
            return await r.run_async(), r
        finally:
            await runner.cleanup()
    return asyncio.run(go())


def zero_walker(root, exclude=()):
    return {"usd": 0.0, "per_dir": {}, "errors": []}


#: the model and object every synthetic run names (the run list of a build without the registry states them; the database agrees)
MODEL, OBJECT = "qwen_synth", "o" * 64


def registry_db(root: pathlib.Path, runs) -> None:
    """A synthetic registry database under a registry root (``registry.sqlite``: what the filing's check of explicit-run-list jobs
    reads). ``runs``: ``{run_id: (eval_id, store_uri)}``; each run's instrument names the Haiku v2 judge for strongreject_bon and the
    v1 judge otherwise."""
    root.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(root / "registry.sqlite"))
    con.executescript("""
      CREATE TABLE run (run_id TEXT PRIMARY KEY, campaign TEXT, eval_id TEXT, model_id TEXT, object_id TEXT, lane_source_legacy TEXT,
                        pass_label TEXT, trial_index INTEGER, token_cap_tokens INTEGER, store_uri TEXT, instrument_id TEXT);
      CREATE TABLE instrument (instrument_id TEXT PRIMARY KEY, eval_id TEXT, judge_id TEXT);
      CREATE TABLE judge (judge_id TEXT PRIMARY KEY, judge_model TEXT, rubric_name TEXT);
      CREATE TABLE judgment (judgment_id TEXT PRIMARY KEY, run_id TEXT, judge_id TEXT, status TEXT, finished_utc TEXT, n_items_judged INTEGER,
                             n_judge_refusals INTEGER, store_uri TEXT);
    """)
    for rid, (ev, store) in runs.items():
        con.execute("INSERT OR IGNORE INTO instrument VALUES (?,?,?)", (f"inst-{ev}", ev, SS.HAIKU_V2 if ev == "strongreject_bon" else SS.HAIKU_V1))
        con.execute("INSERT INTO run VALUES (?,?,?,?,?,?,?,?,?,?,?)", (rid, "synthcamp", ev, MODEL, OBJECT, None, None, 0, 4096, str(store), f"inst-{ev}"))
    con.commit()
    con.close()


def registry_root_with(tmp: pathlib.Path, track: str, judges, runs=None, name: str = "registry_root") -> str:
    """A scratch registry root whose ``records/`` holds these judge records (what ``Filer.judge_on_record`` reads) and, with ``runs``,
    a database of those runs (:func:`registry_db`); nothing else."""
    root = tmp / name
    if runs is not None:
        registry_db(root, runs)
    for j in judges:
        st = tmp / f"{name}_judge_staging_{j['judge_id'][:8]}"
        f = F.Filer(str(st), track, j, "synthetic", "0" * 40)
        p = f.stage_judge()
        dest = root / "records" / os.path.relpath(p, st)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
    return str(root)


def staged_rows(staging: pathlib.Path):
    out = {}
    for p in sorted(staging.rglob("*.jsonl*")):
        op = gzip.open if str(p).endswith(".gz") else open
        with op(p, "rt") as fh:
            out[str(p.relative_to(staging))] = [json.loads(l) for l in fh if l.strip()]
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------
# StrongREJECT: three runs with no Claude judgment (static, PAIR, best-of-N), the commands in order
# ---------------------------------------------------------------------------------------------------------------------------------------
@pytest.fixture
def sr(tmp_path):
    lm = tmp_path / "runlist"
    lm.mkdir(mode=0o700)
    st = tmp_path / "stores"
    for s in range(2):
        jl(st / "static" / f"shard{s:02d}" / "responses.jsonl",
           [{"goal_id": f"g{g}", "trial": 0, "sample_idx": v, "response": reply(g, v)} for g in (2 * s, 2 * s + 1) for v in range(3)])
    jl(st / "pair" / "shard00" / "responses.jsonl", [{"goal_id": f"g{g}", "order": o, "response": reply(g, 3 + o)} for g in range(4) for o in range(2)])
    jl(st / "pair" / "shard00" / "progress.jsonl", [{"goal_id": f"g{g}", "outcome": "fail"} for g in range(4)])
    jl(st / "bon" / "bon_responses.jsonl", [{"goal_id": f"g{g}", "trial": 0, "sample_idx": d, "response": reply(g, 5 + d)} for g in range(2) for d in range(4)])
    goals = tmp_path / "goals_src.json"
    goals.write_text(json.dumps(GOALS))
    runs = {"synth:codeattack:1": ("codeattack", st / "static"), "synth:pair:1": ("pair", st / "pair"),
            "synth:strongreject_bon:1": ("strongreject_bon", st / "bon")}
    tsv = tmp_path / "g.runs.tsv"
    tsv.write_text("run_id\teval_id\tstore\tcampaign\tmodel_id\tobject_id\n"
                   + "".join(f"{r}\t{e}\t{p}\tsynthcamp\t{MODEL}\t{OBJECT}\n" for r, (e, p) in runs.items()))
    g = lm / "sr" / "g"
    site = "local" if str(tmp_path).startswith(SS.HOME + "/") else "synth"
    assert SS.main(["--runs", str(tsv), "--goals", str(goals), "--out", str(g / "scope"), "--no-registry", "--site", f"synth={tmp_path}",
                    "--scope-note", "synthetic explicit run list"]) == 0
    assert E.main(["--jobs", str(g / "scope" / "jobs.jsonl"), "--goals", str(g / "scope" / "goals.json"), "--site", site,
                   "--out", str(g / "enum"), "--workers", "1"]) == 0
    rep = json.load(open(g / "enum" / "report.json"))["summary"]
    assert rep["jobs"] == 3 and rep["units"] == 12 + 8 + 8 and rep["jobs_with_missing"] == 0
    assert SB.main(["--judge", "minimax-m3", "--enum", str(g / "enum"), "--goals", str(g / "scope" / "goals.json"), "--run-dir", str(g / "full"),
                    "--require-file", str(lm / "NEVER_SUBMIT"), "--gate-note", "live slot only; this batch dir never submits", "--workers", "1"]) == 0
    assert H.main(["--run-dir", str(g / "full"), "--dry-run"]) == 0
    assert SRL.main(["init", "--live-dir", str(g / "live"), "--source-run-dir", str(g / "full"), "--max-spend-usd", "5", "--concurrency", "4",
                     "--global-ceiling-usd", "400", "--global-root", str(lm), "--reasoning-reroute"]) == 0
    return {"tmp": tmp_path, "lm": lm, "g": g, "runs": runs, "stores": st}


def sr_engine(sr):
    d = str(sr["g"] / "live")
    code, r = run_engine(lambda ov: LV.LiveRunner(d, overrides=ov, log=lambda m: None, spend_walker=zero_walker), Mock())
    assert code == LV.EXIT_DONE
    return r


def sr_file(sr, staging, *extra, root=None):
    g = sr["g"]
    argv = ["--judge", "minimax-m3", "--jobs", str(g / "scope" / "jobs.jsonl"), "--units", str(g / "full" / "units.tsv.gz"),
            "--run-dir", str(g / "full"), "--run-dir", str(g / "live"), "--staging", str(staging), "--code-sha", "1" * 40, "--dry-run"]
    if root:
        argv += ["--registry-root", root]
    return SF.main(argv + list(extra))


def test_the_live_run_keeps_its_root_routing_and_engine_settings(sr):
    g, lm = sr["g"], sr["lm"]
    tr = json.load(open(g / "live" / "track.json"))
    assert tr["live"]["global_root"] == str(lm) and tr["live"]["global_ceiling_usd"] == 400.0 and tr["live"]["reasoning_reroute"] is True
    assert tr["live"]["routing"] == SRL.routing() and tr["live"]["providers"] == SRL.PROVIDERS and tr["max_spend_usd"] == 5.0
    src = json.load(open(g / "full" / "track.json"))
    assert src["submit_gate"]["requires_files"] == [str(lm / "NEVER_SUBMIT")] and not (lm / "NEVER_SUBMIT").exists()
    # the engine's global check and the dry run's global line walk the run list's root
    r = LV.LiveRunner(str(g / "live"), log=lambda m: None)
    assert r.cfg["global_root"] == str(lm) and LV.global_root_of(str(g / "live")) == str(lm)
    other = r.global_other()
    assert not other["errors"] and all(k.startswith(str(lm)) for k in other["per_dir"])
    assert SRL.main(["dry-run", "--live-dir", str(g / "live")]) == 0
    dry = json.load(open(g / "live" / "dry_run.json"))
    assert dry["global"]["ceiling_usd"] == 400.0 and all(k.startswith(str(lm)) for k in dry["global"]["top_dirs"])


def test_init_states_its_root_and_ceilings_and_refuses_a_root_that_is_not_an_existing_absolute_directory(sr, capsys):
    g, lm = sr["g"], sr["lm"]
    base = ["init", "--live-dir", str(g / "live2"), "--source-run-dir", str(g / "full"), "--concurrency", "4"]
    both = ["--global-ceiling-usd", "400", "--max-spend-usd", "5"]
    for argv in (base, base + ["--global-root", str(lm)], base + ["--global-root", str(lm), "--max-spend-usd", "5"],
                 base + ["--global-root", str(lm), "--global-ceiling-usd", "400"], base + both):
        with pytest.raises(SystemExit) as e:
            SRL.main(argv)
        assert e.value.code == 2
    capsys.readouterr()
    with pytest.raises(SystemExit, match="not an existing directory"):
        SRL.main(base + both + ["--global-root", str(lm / "no_such_root")])
    with pytest.raises(SystemExit, match="not an absolute path"):
        SRL.main(base + both + ["--global-root", "runlist"])
    assert not (g / "live2").exists()                               # every refusal came before anything was written
    assert LV.check_global_root(str(lm) + "/sr/..") == str(lm)
    assert SRL.main(base + both + ["--global-root", str(lm) + "/sr/.."]) == 0
    tr = json.load(open(g / "live2" / "track.json"))
    assert tr["max_spend_usd"] == 5.0 and tr["live"]["global_ceiling_usd"] == 400.0 and tr["live"]["global_root"] == str(lm)


def test_the_judge_ids_are_the_pinned_rows_and_match_the_registry(sr):
    g = sr["g"]
    live = M.live_protocol([str(g / "full"), str(g / "live")])
    assert live["with_batch"] is True
    specs = {k: SF.judge_spec(k, judge="minimax-m3", live=live) for k in ("sr", "sr2")}
    assert {k: v["judge_id"] for k, v in specs.items()} == PINS["strongreject_minimax_m3"]
    assert all(v["judge_backend"] == "batch_or_live" for v in specs.values())
    if not os.path.exists(REGISTRY):
        pytest.skip("no registry on this machine")
    con = sqlite3.connect(f"file:{REGISTRY}?mode=ro", uri=True, timeout=30)
    try:
        for k, spec in specs.items():
            row = con.execute("SELECT judge_model, rubric_name, rubric_version, rubric_sha256, protocol_sha256, threshold_rule, judge_backend, "
                              "extra_json FROM judge WHERE judge_id = ?", (spec["judge_id"],)).fetchone()
            assert row is not None, f"judge {spec['judge_id']} is not on record"
            assert row[:7] == tuple(spec[c] for c in ("judge_model", "rubric_name", "rubric_version", "rubric_sha256", "protocol_sha256",
                                                      "threshold_rule", "judge_backend"))
            assert json.loads(row[7])["protocol"] == spec["extra"]["protocol"]
    finally:
        con.close()


def test_a_run_with_no_claude_judgment_files_its_full_scores_under_the_pinned_judges(sr):
    mock_r = sr_engine(sr)
    assert mock_r.cfg["global_root"] == str(sr["lm"])
    tmp = sr["tmp"]
    # no judge on record: --expect-pinned-judges refuses before anything is staged
    with pytest.raises(SystemExit, match="not on record"):
        sr_file(sr, tmp / "st0", "--expect-pinned-judges", root=str(tmp / "empty_root"))
    assert not (tmp / "st0").exists()
    live = M.live_protocol([str(sr["g"] / "full"), str(sr["g"] / "live")])
    root = registry_root_with(tmp, "strongreject_minimax_m3", [SF.judge_spec(k, judge="minimax-m3", live=live) for k in ("sr", "sr2")],
                              runs=sr["runs"])
    ids = tmp / "filing_ids.json"
    assert sr_file(sr, tmp / "st", "--expect-pinned-judges", "--ids-out", str(ids), root=root) == 0
    rep = json.load(open(tmp / "st" / "filing_report.json"))
    # the three explicit-run-list jobs were held to their runs in the registry before anything was staged
    assert rep["registry_check"] == {"registry": os.path.join(root, "registry.sqlite"), "jobs": 3, "problems": []}
    assert rep["judges"] == PINS["strongreject_minimax_m3"] and rep["totals"]["runs_complete"] == 3 and not rep["gate"]["problems"]
    assert rep["staging_check"]["problems"] == [] and set(rep["staging_check"]["runs"]) == set(sr["runs"])
    assert rep["staging_check"]["expected_runs"] == sorted(sr["runs"])       # held to the runs of --jobs
    # the logical sha256 the check read in its one pass is the registry's own (plain and gzip files)
    staged_files = [f for v in rep["staging_check"]["runs"].values() for f in v["files"]]
    assert {f["rel"].endswith(".gz") for f in staged_files} == {True, False}
    assert all(f["logical_sha256"] == RR.file_logical_sha(str(tmp / "st" / f["rel"])) for f in staged_files)
    files = staged_rows(tmp / "st")
    assert not [p for p in files if "/judge/" in p]                      # the judges are on record: no judge record staged
    judgments = {r["payload"]["run_id"]: r["payload"] for recs in files.values() for r in recs if r["record_type"] == "judgment"}
    items = [r["payload"] for recs in files.values() for r in recs if r["record_type"] == "item_result"]
    assert {r["record_type"] for recs in files.values() for r in recs} == {"judgment", "item_result"}
    for rid, (ev, store) in sr["runs"].items():
        j = judgments[rid]
        kind = "sr2" if ev == "strongreject_bon" else "sr"
        assert j["judge_id"] == PINS["strongreject_minimax_m3"][kind] and j["status"] == "complete" and j["grading_code_sha"] == "1" * 40
        assert j["judgment_id"] == F.identity.judgment_id(rid, j["judge_id"])
        assert j["extra"]["re_measures_judgment_id"] is None and j["extra"]["re_measures_judge_id"] is None
        assert j["extra"]["scope"] == "synthetic explicit run list" and j["extra"]["transport"] == "openrouter_live"
        rows = [r for r in items if r["run_id"] == rid]
        per_unit = [r for r in rows if r["metric_id"] == "strongreject"]
        best = {(r["item_id"], r["trial_index"]): r["value"] for r in rows if r["metric_id"] == "strongreject_best"}
        assert len(per_unit) == j["extra"]["n_units"] and not [r for r in rows if r["metric_id"] == "strongreject_unscored"]
        # the continuous score of every unit is the Haiku formula over its verdict; the goal's best is the max over its units
        for r in per_unit:
            x = r["extra"]
            assert r["value"] == R.strongreject_score(x["refused"], x["convincing"], x["specific"]) and x["transport"] == "openrouter_live"
        for (goal, trial), v in best.items():
            assert v == max(r["value"] for r in per_unit if (r["item_id"], r["trial_index"]) == (goal, trial))
        assert len({r["value"] for r in per_unit}) > 1 and set(best) == {(r["item_id"], r["trial_index"]) for r in per_unit}
        assert not {"asr", "run_metric", "run_state"} & {r["metric_id"] for r in rows}
    assert judgments["synth:pair:1"]["extra"]["judged_stores"] == [str(sr["runs"]["synth:pair:1"][1])]
    # the ids file: what an attribution after delivery reads
    doc = json.load(open(ids))
    assert doc["judgments"] == {rid: judgments[rid]["judgment_id"] for rid in sr["runs"]} and doc["judge_ids"] == sorted(set(PINS["strongreject_minimax_m3"].values()))
    assert all(u.startswith("records/rejudge/strongreject_minimax_m3__") and not u.endswith(".gz") for u in doc["record_files"])
    assert doc["runs"]["synth:pair:1"]["item_rows"]["strongreject"] == 8

    # one run first: only its files reach the inbox
    first = "synth:pair:1"
    assert SF.main(["--judge", "minimax-m3", "--jobs", str(sr["g"] / "scope" / "jobs.jsonl"), "--units", str(sr["g"] / "full" / "units.tsv.gz"),
                    "--run-dir", str(sr["g"] / "full"), "--run-dir", str(sr["g"] / "live"), "--staging", str(tmp / "st"), "--registry-root", root,
                    "--deliver", "runs", "--deliver-only", "--only-run", first, "--expect-pinned-judges"]) == 0
    inbox = pathlib.Path(root) / "inbox" / "rejudge"
    delivered = sorted(str(p.relative_to(inbox)) for p in inbox.rglob("*.jsonl*"))
    assert delivered and all(f"/{F.identity.run_dir_name(first)}/" in p for p in delivered) and len(delivered) == 2
    with pytest.raises(SystemExit, match="not a run of this staging"):
        SF.main(["--judge", "minimax-m3", "--jobs", str(sr["g"] / "scope" / "jobs.jsonl"), "--units", str(sr["g"] / "full" / "units.tsv.gz"),
                 "--run-dir", str(sr["g"] / "full"), "--run-dir", str(sr["g"] / "live"), "--staging", str(tmp / "st"), "--registry-root", root,
                 "--deliver", "runs", "--deliver-only", "--only-run", "synth:not-staged"])
    with pytest.raises(SystemExit, match="goes with --deliver runs --deliver-only"):
        sr_file(sr, tmp / "st9", "--only-run", first, root=root)

    # a tampered staging: a judge record and a judgment under another judge are refused before any delivery
    tampered = tmp / "st_tampered"
    shutil.copytree(tmp / "st", tampered)
    jdir = tampered / "rejudge" / f"strongreject_minimax_m3__{PINS['strongreject_minimax_m3']['sr'][:12]}" / "judge"
    jdir.mkdir()
    src_judge = next((pathlib.Path(root) / "records").rglob("*-judge.jsonl"))
    shutil.copyfile(src_judge, jdir / "000001-judge.jsonl")
    chk = F.check_staging(str(tampered), PINS["strongreject_minimax_m3"].values())
    assert any("a judge record is staged" in p for p in chk["problems"])
    with pytest.raises(SystemExit, match="staging check"):
        SF.main(["--judge", "minimax-m3", "--jobs", str(sr["g"] / "scope" / "jobs.jsonl"), "--units", str(sr["g"] / "full" / "units.tsv.gz"),
                 "--run-dir", str(sr["g"] / "full"), "--run-dir", str(sr["g"] / "live"), "--staging", str(tampered), "--registry-root", root,
                 "--deliver", "runs", "--deliver-only", "--expect-pinned-judges"])
    assert F.check_staging(str(tmp / "st"), ["f" * 64])["problems"]           # another judge's ids: every judgment refused
    assert any("not in the filing's run list" in p for p in F.check_staging(str(tmp / "st"), PINS["strongreject_minimax_m3"].values(),
                                                                                run_ids=["synth:pair:1"])["problems"])


def test_a_run_the_filing_was_asked_to_file_with_nothing_staged_refuses_the_filing(sr, capsys):
    """The filer skips a run with no unit in the unit map, or with too few terminal units, without a word. Under --expect-pinned-judges
    the staging check is held to the runs the filing was asked to file, so the skip refuses the filing and its delivery."""
    sr_engine(sr)
    tmp, g = sr["tmp"], sr["g"]
    live = M.live_protocol([str(g / "full"), str(g / "live")])
    root = registry_root_with(tmp, "strongreject_minimax_m3", [SF.judge_spec(k, judge="minimax-m3", live=live) for k in ("sr", "sr2")],
                              runs=sr["runs"])
    # a job of the scope whose run has no unit in the unit map
    jobs = [json.loads(l) for l in open(g / "scope" / "jobs.jsonl")]
    ghost = dict(jobs[0], run_id="synth:codeattack:ghost")
    plus = g / "scope" / "jobs_plus.jsonl"
    plus.write_text("".join(json.dumps(j) + "\n" for j in jobs + [ghost]))
    argv = ["--judge", "minimax-m3", "--units", str(g / "full" / "units.tsv.gz"), "--run-dir", str(g / "full"), "--run-dir", str(g / "live"),
            "--code-sha", "1" * 40, "--registry-root", root, "--expect-pinned-judges"]
    assert SF.main(argv + ["--jobs", str(plus), "--staging", str(tmp / "st_ghost"), "--dry-run"]) == 2
    assert "REFUSED (staging check)" in capsys.readouterr().out
    chk = json.load(open(tmp / "st_ghost" / "filing_report.json"))["staging_check"]
    assert len(chk["problems"]) == 1 and "1 runs of the list have nothing staged" in chk["problems"][0] and "synth:codeattack:ghost" in chk["problems"][0]
    assert set(chk["runs"]) == set(sr["runs"]) and chk["expected_runs"] == sorted(list(sr["runs"]) + ["synth:codeattack:ghost"])
    # the three staged runs are whole on disk, but the staging was refused: a delivery reads the recorded check and refuses too
    assert not F.check_staging(str(tmp / "st_ghost"), PINS["strongreject_minimax_m3"].values())["problems"]
    assert "nothing staged" in F.staged_check_refusal(str(tmp / "st_ghost"))
    with pytest.raises(SystemExit, match="staging check.*nothing staged"):
        SF.main(argv + ["--jobs", str(plus), "--staging", str(tmp / "st_ghost"), "--deliver", "runs", "--deliver-only"])
    # ... and so does a delivery WITHOUT the flag: a recorded refusal blocks the staging whatever the delivery's flags
    plain = [a for a in argv if a != "--expect-pinned-judges"]
    with pytest.raises(SystemExit, match="staging check.*nothing staged"):
        SF.main(plain + ["--jobs", str(plus), "--staging", str(tmp / "st_ghost"), "--deliver", "runs", "--deliver-only"])
    assert not (pathlib.Path(root) / "inbox").exists()
    # --runs narrows what the filing is asked to file: the same scope with the ghost left out files clean
    keep = tmp / "keep.txt"
    keep.write_text("".join(r + "\n" for r in sr["runs"]))
    assert SF.main(argv + ["--jobs", str(plus), "--runs", str(keep), "--staging", str(tmp / "st_keep"), "--dry-run"]) == 0
    assert F.staged_check_refusal(str(tmp / "st_keep")) is None and F.staged_check_refusal(str(tmp / "no_such_staging")) is None
    # ... and a run named in --runs with nothing staged is refused the same way
    keep.write_text("".join(r + "\n" for r in list(sr["runs"]) + ["synth:codeattack:ghost"]))
    assert SF.main(argv + ["--jobs", str(plus), "--runs", str(keep), "--staging", str(tmp / "st_named"), "--dry-run"]) == 2
    # a run below --min-terminal is skipped: refused, not silently left out (every run here has fewer than 9 terminal units but one)
    assert SF.main(argv + ["--jobs", str(g / "scope" / "jobs.jsonl"), "--staging", str(tmp / "st_min"), "--min-terminal", "9", "--dry-run"]) == 2
    chk = json.load(open(tmp / "st_min" / "filing_report.json"))["staging_check"]
    assert set(chk["runs"]) == {"synth:codeattack:1"} and "2 runs of the list have nothing staged" in chk["problems"][0]
    # a staging made WITHOUT the flag records no check, so its delivery is what it always was
    assert SF.main(plain + ["--jobs", str(g / "scope" / "jobs.jsonl"), "--staging", str(tmp / "st_plain"), "--dry-run"]) == 0
    assert "staging_check" not in json.load(open(tmp / "st_plain" / "filing_report.json")) and F.staged_check_refusal(str(tmp / "st_plain")) is None
    assert SF.main(plain + ["--jobs", str(g / "scope" / "jobs.jsonl"), "--staging", str(tmp / "st_plain"), "--deliver", "runs", "--deliver-only"]) == 0
    assert len(list((pathlib.Path(root) / "inbox" / "rejudge").rglob("*.jsonl*"))) == 6


def test_an_explicit_run_list_job_is_held_to_its_run_in_the_registry_before_staging(sr):
    """A scope built without the registry is only as good as its list. Under --expect-pinned-judges the filing runs the registry half
    itself (what ``scope_runs.py --verify-jobs`` runs), before anything is staged: a job that reads another run's store -- by its path
    or by its files -- is refused, and so is a registry root with no database to check against."""
    sr_engine(sr)
    tmp, g, st = sr["tmp"], sr["g"], sr["stores"]
    live = M.live_protocol([str(g / "full"), str(g / "live")])
    judges = [SF.judge_spec(k, judge="minimax-m3", live=live) for k in ("sr", "sr2")]
    root = registry_root_with(tmp, "strongreject_minimax_m3", judges, runs=sr["runs"])
    jobs = {j["run_id"]: j for j in (json.loads(l) for l in open(g / "scope" / "jobs.jsonl"))}
    pair, static = "synth:pair:1", "synth:codeattack:1"
    # ANOTHER run's store, as valid as this one's
    jl(st / "pair_b" / "shard00" / "responses.jsonl", [{"goal_id": f"g{i}", "order": 0, "response": reply(i, 9)} for i in range(4)])
    jl(st / "pair_b" / "shard00" / "progress.jsonl", [{"goal_id": f"g{i}", "outcome": "fail"} for i in range(4)])
    theirs = str(st / "pair_b" / "shard00" / "responses.jsonl")

    def file(name, edits=None, *extra, root=root, flag=True):
        path = tmp / f"jobs_{name}.jsonl"
        path.write_text("".join(json.dumps((edits or {}).get(r, j)) + "\n" for r, j in jobs.items()))
        return SF.main(["--judge", "minimax-m3", "--jobs", str(path), "--units", str(g / "full" / "units.tsv.gz"), "--run-dir", str(g / "full"),
                        "--run-dir", str(g / "live"), "--staging", str(tmp / f"st_{name}"), "--code-sha", "1" * 40, "--dry-run",
                        "--registry-root", root, *(["--expect-pinned-judges"] if flag else []), *extra])

    def unit(run_id, **changes):
        return {run_id: dict(jobs[run_id], units=[dict(jobs[run_id]["units"][0], **changes)])}

    assert file("ok") == 0
    refused = {"store": (unit(pair, path=str(st / "pair_b"), files=[theirs]), "are not the one the registry records for the run"),
               "files": (unit(pair, files=[theirs]), "reads 1 file.s. that are not under its path"),
               "kind": ({static: dict(jobs[static], kind="sr2")}, "the run's instrument is rubric sr, the list says kind 'sr2'"),
               "fields": ({static: dict(jobs[static], object_id="x" * 64)}, "the registry's object_id is")}
    for name, (edits, why) in refused.items():
        with pytest.raises(SystemExit, match=r"REFUSED \(--expect-pinned-judges\): the jobs do not verify against the registry: .*" + why):
            file(name, edits)
        assert not (tmp / f"st_{name}").exists()                                  # refused before anything was staged
    # the registry is the authority: the same jobs against a registry that ties the run to another store are refused
    moved = dict(sr["runs"], **{pair: ("pair", st / "pair_b")})
    with pytest.raises(SystemExit, match="are not the one the registry records for the run"):
        file("moved", root=registry_root_with(tmp, "strongreject_minimax_m3", judges, runs=moved, name="root_moved"))
    # no database under the registry root: nothing to hold the jobs to
    with pytest.raises(SystemExit, match="no registry database at .*registry.sqlite: the 3 explicit-run-list jobs"):
        file("nodb", root=registry_root_with(tmp, "strongreject_minimax_m3", judges, name="root_nodb"))
    assert not (tmp / "st_nodb").exists()
    # a registry that records the stores where they were generated: the scope built from a mirror verifies through --path-rewrite
    cluster = {r: (ev, "/synth-cluster/stores/" + pathlib.Path(p).name) for r, (ev, p) in sr["runs"].items()}
    root_c = registry_root_with(tmp, "strongreject_minimax_m3", judges, runs=cluster, name="root_cluster")
    with pytest.raises(SystemExit, match="are not the one the registry records for the run"):
        file("mirror_plain", root=root_c)
    assert file("mirror", None, "--path-rewrite", f"/synth-cluster/stores/={st}/", root=root_c) == 0
    # a second filing of a run is the filer's own business: a MiniMax judgment on record does not refuse the check
    con = sqlite3.connect(os.path.join(root, "registry.sqlite"))
    con.execute("INSERT INTO judge VALUES (?,?,?)", (PINS["strongreject_minimax_m3"]["sr"], "minimax-m3", "strongreject"))
    con.execute("INSERT INTO judgment VALUES (?,?,?,?,?,?,?,?)", ("mm-1", pair, PINS["strongreject_minimax_m3"]["sr"], "complete", None, 8, 0, None))
    con.commit()
    con.close()
    assert file("follow_up") == 0
    # without the flag the filing checks no registry (the paper-row path): the same wrong job stages
    assert file("unflagged", refused["store"][0], flag=False) == 0


def test_the_paper_row_jobs_still_file_with_their_haiku_ids_and_scope(sr):
    """A job that names its Haiku judgment (the paper-row scope's shape) keeps the re_measures ids and the paper-row scope text."""
    sr_engine(sr)
    g = sr["g"]
    jobs = [json.loads(l) for l in open(g / "scope" / "jobs.jsonl")]
    for j in jobs:
        j.update({"haiku_judge_id": SF.HAIKU_JUDGES[j["kind"]], "haiku_judgment_id": "h-" + j["run_id"], "mapping_method": "exact_path"})
        j.pop("scope")
    (g / "scope" / "jobs_paper_shape.jsonl").write_text("".join(json.dumps(j) + "\n" for j in jobs))
    assert SF.main(["--judge", "minimax-m3", "--jobs", str(g / "scope" / "jobs_paper_shape.jsonl"), "--units", str(g / "full" / "units.tsv.gz"),
                    "--run-dir", str(g / "full"), "--run-dir", str(g / "live"), "--staging", str(sr["tmp"] / "st_paper"), "--code-sha", "1" * 40,
                    "--dry-run"]) == 0
    files = staged_rows(sr["tmp"] / "st_paper")
    js = [r["payload"] for recs in files.values() for r in recs if r["record_type"] == "judgment"]
    assert js and all(j["extra"]["scope"] == SF.PAPER_ROW_SCOPE and j["extra"]["re_measures_judgment_id"] == "h-" + j["run_id"]
                      and "judged_stores" not in j["extra"] for j in js)


# ---------------------------------------------------------------------------------------------------------------------------------------
# C/P/I: an XSTest run (both halves) and an OR-Bench run with no Sonnet judgment
# ---------------------------------------------------------------------------------------------------------------------------------------
LABELS = ("full_compliance", "partial_refusal", "full_refusal")


def pinned():
    xs = {f"xst{i:03d}": {"item_id": f"xst{i:03d}", "prompt": f"Synthetic xstest prompt {i}", "subset": "safe" if i < 4 else "unsafe"} for i in range(6)}
    ob = {f"orb{i:03d}": {"item_id": f"orb{i:03d}", "prompt": f"Synthetic or-bench prompt {i}"} for i in range(5)}
    return {"or_bench_hard": ob, "xstest": xs}


@pytest.fixture
def cpi(tmp_path):
    lm = tmp_path / "runlist"
    lm.mkdir(mode=0o700)
    P = pinned()
    cells = {}
    for ev, task, cap in (("xstest", "xstest_v2", 2048), ("or_bench_hard", "or_bench_hard_1k", 8192)):
        cell = tmp_path / "cells" / f"{ev}_cap{cap}"
        rows = [{"__spool_header__": {"task": f"judged/{task}", "gen_cfg": {"max_new_tokens": cap}}}]
        rows += [{"item_id": iid, "text": f"synthetic reply to {iid} MOCKLABEL:{LABELS[n % 3]} #{n}"} for n, iid in enumerate(P[ev])]
        jl(cell / "private" / f"gens_{task}.jsonl", rows)
        cells[ev] = (cell, cap)
    runs = {"synth:xstest:1": "xstest", "synth:or_bench_hard:1": "or_bench_hard"}
    tsv = tmp_path / "g4.runs.tsv"
    tsv.write_text("run_id\teval_id\tcell\tcap\tcampaign\n" + "".join(f"{r}\t{ev}\t{cells[ev][0]}\t{cells[ev][1]}\tsynthcamp\n" for r, ev in runs.items()))
    td = lm / "terra" / "g4"
    assert SC.main(["--runs", str(tsv), "--out", str(td / "scope"), "--no-registry", "--scope-note", "synthetic explicit run list"],
                   pinned_fn=pinned) == 0
    assert TB.main(["--judge", "minimax-m3", "--family", "cpi", "--scope", str(td / "scope"), "--run-dir", str(td / "full" / "cpi"),
                    "--max-spend-usd", "20", "--require-file", str(lm / "NEVER_SUBMIT"), "--gate-note", "live slot only", "--workers", "1"]) == 0
    assert TH.main(["--run-dir", str(td / "full" / "cpi"), "--dry-run"]) == 0
    assert TL.main(["init", "--terra-dir", str(td), "--runs", "cpi", "--max-spend-usd", "20", "--concurrency", "4", "--global-ceiling-usd", "400",
                    "--global-root", str(lm)]) == 0
    return {"tmp": tmp_path, "lm": lm, "td": td, "runs": runs, "cells": cells}


def test_cpi_live_run_files_both_xstest_halves_at_the_named_rung_under_the_pinned_judges(cpi):
    td, lm, tmp = cpi["td"], cpi["lm"], cpi["tmp"]
    paths = TL.Paths(str(td))
    tr = json.load(open(td / "live" / "cpi" / "track.json"))
    assert tr["live"]["global_root"] == str(lm) and tr["live"]["global_ceiling_usd"] == 400.0 and tr["live"]["routing"] == SRL.routing()
    assert json.load(open(td / "full" / "cpi" / "track.json"))["submit_gate"] == {"requires_files": [str(lm / "NEVER_SUBMIT")],
                                                                                  "note": "live slot only"}
    lives = TF.live_protocols([paths.source("cpi"), paths.live("cpi")])
    specs = {r: TF.judge_spec(r, "minimax-m3", live=lives["cpi"]) for r in ("or_bench_cpi", "xstest_cpi")}
    assert {r: s["judge_id"] for r, s in specs.items()} == PINS["cpi_minimaxm3"]
    if os.path.exists(REGISTRY):
        con = sqlite3.connect(f"file:{REGISTRY}?mode=ro", uri=True, timeout=30)
        try:
            for r, spec in specs.items():
                row = con.execute("SELECT judge_model, rubric_name, rubric_version, rubric_sha256, protocol_sha256, threshold_rule, judge_backend, "
                                  "extra_json FROM judge WHERE judge_id = ?", (spec["judge_id"],)).fetchone()
                assert row is not None and row[:7] == tuple(spec[c] for c in ("judge_model", "rubric_name", "rubric_version", "rubric_sha256",
                                                                              "protocol_sha256", "threshold_rule", "judge_backend"))
                assert json.loads(row[7])["protocol"] == spec["extra"]["protocol"]
        finally:
            con.close()
    code, _ = run_engine(lambda ov: TL.TerraLiveRunner(paths.live("cpi"), gate_file=paths.gate_file, overrides=ov, log=lambda m: None,
                                                       spend_walker=zero_walker), Mock())
    assert code == LV.EXIT_DONE
    root = registry_root_with(tmp, "cpi_minimaxm3", list(specs.values()))
    ids = tmp / "cpi_ids.json"
    base = ["--judge", "minimax-m3", "--scope-cpi", str(td / "scope"), "--run-dir", str(td / "full" / "cpi"), "--run-dir", str(td / "live" / "cpi"),
            "--registry-root", root, "--code-sha", "1" * 40]
    assert TF.main(base + ["--staging", str(tmp / "cst"), "--dry-run", "--expect-pinned-judges", "--ids-out", str(ids)]) == 0
    rep = json.load(open(tmp / "cst" / "filing_report.json"))
    assert rep["judges"] == PINS["cpi_minimaxm3"] and rep["totals"]["runs_complete"] == 2 and not rep["refused"]
    files = staged_rows(tmp / "cst")
    assert not [p for p in files if "/judge/" in p]
    js = {r["payload"]["run_id"]: r["payload"] for recs in files.values() for r in recs if r["record_type"] == "judgment"}
    items = [r["payload"] for recs in files.values() for r in recs if r["record_type"] == "item_result"]
    x = js["synth:xstest:1"]
    assert x["judge_id"] == PINS["cpi_minimaxm3"]["xstest_cpi"] and x["status"] == "complete"
    assert x["extra"]["re_measures_judgment_id"] is None and x["extra"]["scope"] == "synthetic explicit run list"
    assert x["extra"]["judged_store"] == str(cpi["cells"]["xstest"][0]) and x["extra"]["judged_cap"] == 2048
    # the judgment names the cell where it was generated and the sha256 of the spool its texts came from, as its scope states them
    scoped = {r["run_id"]: r for r in (json.loads(l) for l in open(td / "scope" / "runs.jsonl"))}
    spool = cpi["cells"]["xstest"][0] / "private" / "gens_xstest_v2.jsonl"
    assert x["extra"]["judged_spool_sha256"] == hashlib.sha256(spool.read_bytes()).hexdigest() == scoped["synth:xstest:1"]["judged_spool_sha256"]
    assert x["extra"]["judged_store_origin"] == scoped["synth:xstest:1"]["judged_store_origin"] == str(cpi["cells"]["xstest"][0])
    scoped["synth:xstest:1"]["judged_store_origin"] = "/synth-cluster/exp/cells/xstest_cap2048"      # a cell read through a mirror
    xr = [r for r in items if r["run_id"] == "synth:xstest:1"]
    assert len(xr) == 6 and {r["metric_id"] for r in xr} == {"xst_label"} and sorted(r["extra"]["subset"] for r in xr).count("unsafe") == 2
    assert all(r["value"] == TF.LABEL_VALUE[r["label"]] for r in xr)
    orr = [r for r in items if r["run_id"] == "synth:or_bench_hard:1"]
    assert len(orr) == 5 and {r["metric_id"] for r in orr} == {"orb_label"} and js["synth:or_bench_hard:1"]["extra"]["judged_cap"] == 8192
    assert not set(TF.HEADLINE_IDS_NEVER_FILED["xstest"]) & {r["metric_id"] for r in items}
    doc = json.load(open(ids))
    assert doc["judgments"] == {rid: js[rid]["judgment_id"] for rid in cpi["runs"]}
    # one run first
    assert TF.main(base + ["--staging", str(tmp / "cst"), "--deliver", "runs", "--deliver-only", "--accept-run-repoint", "--only-run",
                           "synth:xstest:1", "--expect-pinned-judges"]) == 0
    inbox = pathlib.Path(root) / "inbox" / "rejudge"
    delivered = sorted(str(p.relative_to(inbox)) for p in inbox.rglob("*.jsonl*"))
    assert len(delivered) == 2 and all(f"/{F.identity.run_dir_name('synth:xstest:1')}/" in p for p in delivered)
    # a judge not on record refuses before staging
    with pytest.raises(SystemExit, match="not on record"):
        TF.main(["--judge", "minimax-m3", "--scope-cpi", str(td / "scope"), "--run-dir", str(td / "full" / "cpi"), "--run-dir",
                 str(td / "live" / "cpi"), "--registry-root", str(tmp / "empty_root"), "--staging", str(tmp / "cst2"), "--dry-run",
                 "--expect-pinned-judges"])
    # a scope whose cell was read through a mirror: the judgment carries the cell's origin beside the path read
    mirrored = tmp / "scope_mirrored"
    shutil.copytree(td / "scope", mirrored)
    (mirrored / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in scoped.values()))
    assert TF.main(["--judge", "minimax-m3", "--scope-cpi", str(mirrored), "--run-dir", str(td / "full" / "cpi"), "--run-dir",
                    str(td / "live" / "cpi"), "--code-sha", "1" * 40, "--staging", str(tmp / "cst_mirrored"), "--dry-run"]) == 0
    jm = {r["payload"]["run_id"]: r["payload"]["extra"] for recs in staged_rows(tmp / "cst_mirrored").values() for r in recs
          if r["record_type"] == "judgment"}["synth:xstest:1"]
    assert jm["judged_store_origin"] == "/synth-cluster/exp/cells/xstest_cap2048" and jm["judged_store"] == str(cpi["cells"]["xstest"][0])
    # a run the filing was asked to file with nothing staged (here: every run below --min-terminal) refuses the filing and its delivery
    assert TF.main(base + ["--staging", str(tmp / "cst_min"), "--dry-run", "--expect-pinned-judges", "--min-terminal", "100"]) == 2
    rep_min = json.load(open(tmp / "cst_min" / "filing_report.json"))
    assert "2 runs of the list have nothing staged" in rep_min["staging_check"]["problems"][0] and rep_min["refused"]
    assert rep_min["staging_check"]["expected_runs"] == sorted(cpi["runs"])
    with pytest.raises(SystemExit, match="the staging was refused"):
        TF.main(base + ["--staging", str(tmp / "cst_min"), "--deliver", "runs", "--deliver-only", "--accept-run-repoint", "--expect-pinned-judges"])
    only = tmp / "only_xstest.txt"
    only.write_text("synth:xstest:1\n")
    assert TF.main(base + ["--staging", str(tmp / "cst_only"), "--dry-run", "--expect-pinned-judges", "--runs", str(only)]) == 0
    assert json.load(open(tmp / "cst_only" / "filing_report.json"))["staging_check"]["expected_runs"] == ["synth:xstest:1"]


def test_a_cpi_init_states_its_root_and_ceilings_and_refuses_a_root_that_is_not_an_existing_absolute_directory(cpi, capsys):
    td, lm = cpi["td"], cpi["lm"]
    base = ["init", "--terra-dir", str(td), "--live-root", str(td / "live2"), "--runs", "cpi", "--concurrency", "4"]
    for argv in (base + ["--max-spend-usd", "20", "--global-root", str(lm)], base + ["--global-ceiling-usd", "400", "--global-root", str(lm)],
                 base + ["--max-spend-usd", "20", "--global-ceiling-usd", "400"]):
        with pytest.raises(SystemExit) as e:
            TL.main(argv)
        assert e.value.code == 2
    capsys.readouterr()
    both = ["--max-spend-usd", "20", "--global-ceiling-usd", "400"]
    with pytest.raises(SystemExit, match="not an existing directory"):
        TL.main(base + both + ["--global-root", str(lm / "no_such_root")])
    with pytest.raises(SystemExit, match="not an absolute path"):
        TL.main(base + both + ["--global-root", "runlist"])
    assert not (td / "live2" / "cpi" / "track.json").exists()
    assert TL.main(base + both + ["--global-root", str(lm) + "/terra/.."]) == 0
    tr = json.load(open(td / "live2" / "cpi" / "track.json"))
    assert tr["live"]["global_root"] == str(lm) and tr["live"]["global_ceiling_usd"] == 400.0 and tr["max_spend_usd"] == 20.0


def test_a_refresh_restates_the_root_and_ceilings(cpi, capsys):
    """``init --refresh`` re-registers a live directory: the global root and both ceilings are restated at every init, so a refresh that
    left any unstated is refused before anything moves, and the function refuses what the CLI would."""
    td, lm = cpi["td"], cpi["lm"]
    track = td / "live" / "cpi" / "track.json"
    before = track.read_bytes()
    refresh = ["init", "--terra-dir", str(td), "--runs", "cpi", "--refresh"]
    for extra in ([], ["--global-ceiling-usd", "400"], ["--max-spend-usd", "20"], ["--global-root", str(lm), "--global-ceiling-usd", "400"]):
        with pytest.raises(SystemExit) as e:
            TL.main(refresh + extra)
        assert e.value.code == 2
    capsys.readouterr()
    paths = TL.Paths(str(td))
    with pytest.raises(SystemExit, match="none has a default"):
        TL.init_run("cpi", paths, None, None, None, refresh=True, log=lambda m: None)
    assert track.read_bytes() == before and not (td / "live" / "cpi" / "superseded").exists()   # every refusal left the directory as it was
    assert TL.main(refresh + ["--max-spend-usd", "20", "--global-root", str(lm), "--global-ceiling-usd", "400"]) == 0
    tr = json.loads(track.read_text())
    assert tr["live"]["global_root"] == str(lm) and tr["live"]["global_ceiling_usd"] == 400.0 and tr["max_spend_usd"] == 20.0