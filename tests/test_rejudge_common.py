"""The shared OpenRouter batch library (scripts/rejudge/common): body serialization and chunking, the submit-all-up-front submitter's
refusal handling (413 split, 429 backoff, 402 balance queue), the harvester's parse / retry / pause loop against a fake Batch API,
and the registry filing module's records (the registry's own validator; nothing existing restated). Offline throughout."""
from __future__ import annotations

import gzip
import json
import os
import pathlib
import sys
import time

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from rejudge.common import batchfile as BF  # noqa: E402
from rejudge.common import filing as F  # noqa: E402
from rejudge.common import submitter as SUB  # noqa: E402
from rejudge.common.harvester import EXIT_DONE, EXIT_PAUSED, EXIT_UNFINISHED, Harvester  # noqa: E402
from rejudge.common.openrouter import OpenRouterError, classify_error, limit_from_message, redact  # noqa: E402
from rejudge.common.state import Ledger  # noqa: E402
from rejudge.strongreject import file_records as FR  # noqa: E402
from rejudge.strongreject import judges as J  # noqa: E402
from rejudge.strongreject import request as R  # noqa: E402
from rejudge.strongreject import track as T  # noqa: E402
from sisteer.registry import identity  # noqa: E402
from sisteer.registry import records as RR  # noqa: E402


def _inputs(n, kind="sr"):
    return [{"kind": kind, "dataset": "harmbench", "revision": "rev", "goal": f"goal {i % 7}", "response": f"response number {i} " * (i % 5 + 1)}
            for i in range(n)]


# --------------------------------------------------------------------------------------------
# bodies, chunking, manifests
# --------------------------------------------------------------------------------------------
def test_batch_body_puts_requests_last_and_round_trips():
    hdr = BF.batch_header("minimax/minimax-m3", "/v1/chat/completions", {"only": ["deepinfra"]})
    lines = [BF.serialize_request("a", {"messages": [{"role": "user", "content": "x—y"}]}), BF.serialize_request("b", {"m": 1})]
    body = BF.serialize_batch(hdr, lines)
    d = json.loads(body)
    assert list(d.keys()) == ["endpoint", "model", "provider", "completion_window", "requests"]
    assert [r["custom_id"] for r in d["requests"]] == ["a", "b"]
    assert d["requests"][0]["body"]["messages"][0]["content"] == "x—y"
    assert body.isascii()


def test_chunk_respects_both_caps_and_refuses_an_oversized_request():
    hdr = BF.batch_header("m", "/v1/chat/completions")
    reqs = [BF.PackedRequest(str(i), b"x" * 100) for i in range(50)]
    over = BF.header_overhead(hdr)
    parts = list(BF.chunk(reqs, hdr, max_bytes=over + 1000, max_requests=7))
    assert sum(len(p) for p in parts) == 50
    for p in parts:
        assert len(p) <= 7
        assert len(BF.serialize_batch(hdr, [r.line for r in p])) <= over + 1000
    with pytest.raises(ValueError):
        list(BF.chunk([BF.PackedRequest("big", b"x" * 5000)], hdr, max_bytes=over + 1000))


def test_manifest_body_is_immutable_and_hash_checked(tmp_path):
    rd = str(tmp_path / "run")
    T.init_run(rd)
    (m,) = T.add_inputs(rd, _inputs(20), prefix="t")
    assert m["n_requests"] == len(set(R.cache_key(x["kind"], x["dataset"], x["revision"], x["goal"], x["response"]) for x in _inputs(20)))
    body = BF.read_body(rd, m)
    assert BF.sha256_bytes(body) == m["body_sha256"]
    with pytest.raises(FileExistsError):
        BF.write_manifest(rd, m["batch_key"], BF.batch_header("m", "e"), [], "t")
    # tamper with the body: the submit-side read refuses it
    p = os.path.join(rd, m["body_path"])
    with gzip.open(p, "wb") as fh:
        fh.write(body.replace(b"response number 1 ", b"response number X "))
    with pytest.raises(ValueError):
        BF.read_body(rd, m)


def test_add_inputs_refuses_a_key_that_no_longer_recomputes(tmp_path):
    rd = str(tmp_path / "run")
    T.init_run(rd)
    x = dict(_inputs(1)[0], k="0" * 64)
    with pytest.raises(ValueError):
        T.add_inputs(rd, [x])


def test_error_classification_and_redaction():
    assert classify_error(413, "more than 50,000 requests") == "too_large"
    assert limit_from_message("Batch has more than 50,000 requests") == 50000
    assert classify_error(429, "too many batches in progress") == "rate_limited"
    assert classify_error(402, "insufficient credits") == "balance"
    assert classify_error(403, "Key limit exceeded") == "balance"
    assert classify_error(403, "Input flagged by moderation") == "moderation"
    assert classify_error(403, "forbidden") == "auth"
    assert classify_error(503, "") == "transient"
    probe = "-".join(["sk", "or", "probe"]) + "".join(chr(97 + i) for i in range(16))     # assembled at run time: no key-shaped literal
    assert probe not in redact(f"echo Bearer {probe} and {probe}", probe)


# --------------------------------------------------------------------------------------------
# a fake OpenRouter Batch API
# --------------------------------------------------------------------------------------------
class FakeAPI:
    """Accepts batches (or refuses per a script), completes them on the next GET with synthetic results. ``outcome(cid, i)`` names
    the result of the i-th request of a batch: "verdict", "moderation", "missing", "truncated", "malformed", "error" (a 500), or a dict -- the
    ``error`` object of the result entry itself. ``fail_message`` fails every batch with that message; ``fail_first`` only the first
    N batches (``fail_message`` may then be a function of the batch's custom ids)."""

    def __init__(self, refuse=None, outcome=None, fail_message=None, fail_first=None):
        self.refuse = list(refuse or [])      # a list of OpenRouterError to raise on successive submits
        self.outcome = outcome or (lambda cid, i: "verdict")
        self.fail_message = fail_message
        self.fail_first = fail_first
        self.n_failed = 0
        self.batches = {}
        self.submits = 0
        self.bodies = []

    def submit(self, body):
        self.submits += 1
        if self.refuse:
            raise self.refuse.pop(0)
        d = json.loads(body)
        self.bodies.append(d)
        bid = f"batch-{len(self.batches)}"
        self.batches[bid] = {"id": bid, "status": "validating", "model": d["model"], "provider": "OpenAI",
                             "request_counts": {"total": len(d["requests"]), "completed": 0, "failed": 0}, "created_at": time.time(),
                             "_reqs": d["requests"]}
        return {k: v for k, v in self.batches[bid].items() if not k.startswith("_")}

    def iter_batches(self, statuses=None, created_after=None, max_pages=1000):
        for b in self.batches.values():
            self._advance(b)
            yield {k: v for k, v in b.items() if not k.startswith("_") and k != "results"}

    def get(self, bid):
        return {k: v for k, v in self.batches[bid].items() if not k.startswith("_")}

    def _advance(self, b):
        """one listing = one step of the provider's clock: validating -> in_progress -> a final status with its results"""
        if b["status"] == "validating":
            b["status"] = "in_progress"
        elif b["status"] == "in_progress":
            fail = self.fail_message and (self.fail_first is None or self.n_failed < self.fail_first)
            if fail:
                self.n_failed += 1
                msg = self.fail_message([r["custom_id"] for r in b["_reqs"]]) if callable(self.fail_message) else self.fail_message
                b.update(status="failed", error={"message": msg}, results=None)
            else:
                res = []
                for i, r in enumerate(b["_reqs"]):
                    o = self.outcome(r["custom_id"], i)
                    if isinstance(o, dict):
                        res.append({"custom_id": r["custom_id"], "response": None, "error": o})
                    elif o == "verdict":
                        args = json.dumps({"reasoning": "r", "refused": i % 2, "convincing": 3, "specific": 4})
                        res.append({"custom_id": r["custom_id"], "error": None, "response": {"status_code": 200, "body": {
                            "choices": [{"finish_reason": "stop", "message": {"tool_calls": [{"function": {"name": "StrongREJECTRating", "arguments": args}}]}}],
                            "usage": {"prompt_tokens": 800, "completion_tokens": 100, "prompt_tokens_details": {"cached_tokens": 0}}}}})
                    elif o == "truncated":
                        res.append({"custom_id": r["custom_id"], "error": None, "response": {"status_code": 200, "body": {
                            "choices": [{"finish_reason": "length", "message": {"tool_calls": [{"function": {"name": "StrongREJECTRating", "arguments": '{"reasoning": "cut'}}]}}],
                            "usage": {"prompt_tokens": 800, "completion_tokens": 600}}}})
                    elif o == "malformed":
                        res.append({"custom_id": r["custom_id"], "error": None, "response": {"status_code": 200, "body": {
                            "choices": [{"finish_reason": "stop", "message": {"tool_calls": [{"function": {"name": "StrongREJECTRating", "arguments": "{not json"}}]}}],
                            "usage": {"prompt_tokens": 800, "completion_tokens": 10}}}})
                    elif o == "moderation":
                        res.append({"custom_id": r["custom_id"], "response": None, "error": {"code": 403, "message": "flagged by moderation"}})
                    elif o == "missing":
                        continue
                    else:
                        res.append({"custom_id": r["custom_id"], "response": None, "error": {"code": 500, "message": "upstream"}})
                b.update(status="completed", results=res, usage={"cost": 0.001 * len(res)})

    def key_info(self):
        return {"limit": 2000, "limit_remaining": 1999}


def _run(tmp_path, n=30, max_requests=10, extra=None, init=None):
    rd = str(tmp_path / "run")
    T.init_run(rd, extra=dict({"submit_spacing_s": 0}, **(extra or {})), **(init or {}))
    T.add_inputs(rd, _inputs(n), max_requests=max_requests)
    return rd


def test_harvester_submits_everything_up_front_then_harvests(tmp_path):
    rd = _run(tmp_path)
    api = FakeAPI(outcome=lambda cid, i: "moderation" if i == 3 else "verdict")
    h = Harvester(rd, api, log=lambda m: None)
    out = h.cycle()
    assert out["submit"]["submitted"] == len(BF.list_manifests(rd)) == 3      # every batch out before any result came back
    for _ in range(3):
        h.cycle()
    counts = h.ledger.outcome_counts()
    assert counts.get("moderation") == 3 and counts.get("verdict") == 27       # one moderated request per batch, recorded per item
    assert not h.busy()
    mod = list(h.ledger.iter_requests("moderation"))
    assert all(r["terminal"] == 1 and r["verdict"] is None for r in mod)
    prog = h.progress(done=True)
    assert prog["requests"]["outcomes"]["moderation"] == 3 and prog["spend_usd"]["batches_reported"] > 0
    h.ledger.close()


def test_transient_failures_are_re_asked_and_missing_results_counted(tmp_path):
    rd = _run(tmp_path, n=10)
    seen = {}

    def outcome(cid, i):
        seen[cid] = seen.get(cid, 0) + 1
        if i == 0 and seen[cid] == 1:
            return "provider_error"
        if i == 1 and seen[cid] == 1:
            return "missing"
        return "verdict"
    api = FakeAPI(outcome=outcome)
    h = Harvester(rd, api, log=lambda m: None)
    for _ in range(8):
        h.cycle()
    counts = h.ledger.outcome_counts()
    assert counts == {"verdict": 10}
    attempts = [r[0] for r in h.ledger.conn.execute("SELECT outcome FROM attempts")]
    assert attempts.count("provider_error") == 1 and attempts.count("missing") == 1   # history kept
    h.ledger.close()


def test_413_splits_429_backs_off_402_queues(tmp_path):
    rd = _run(tmp_path, n=10, max_requests=10)
    api = FakeAPI(refuse=[OpenRouterError("too_large", 413, "more than 4 requests")])
    led = Ledger(rd).writer()
    s = SUB.submit_pending(rd, led, api, {"submit_spacing_s": 0}, log=lambda m: None)
    assert s["split"] == 1
    states = led.state_counts()
    assert states.get("split") == 1 and states.get("pending") == 3          # 10 requests -> pieces of <= 4
    api.refuse = [OpenRouterError("rate_limited", 429, "too many batches in progress", retry_after_s=120)]
    s = SUB.submit_pending(rd, led, api, {"submit_spacing_s": 0}, log=lambda m: None)
    assert s["stopped"] == "rate_limited" and s["submitted"] == 0
    assert all(b["next_attempt_ts"] > time.time() + 60 for b in led.batches(("pending",)) if b["last_error_kind"] == "rate_limited")
    for b in led.batches(("pending",)):
        led.update_batch(b["batch_key"], next_attempt_ts=0)
    api.refuse = [OpenRouterError("balance", 402, "Insufficient credits: the estimated cost exceeds your balance")]
    s = SUB.submit_pending(rd, led, api, {"submit_spacing_s": 0}, log=lambda m: None)
    assert s["stopped"] == "balance"
    assert len(led.batches(("balance_queued",))) == 1
    ev = [r["kind"] for r in led.conn.execute("SELECT kind FROM events")]
    assert "BALANCE" in ev and "SPLIT" in ev and "RATE_LIMITED" in ev
    for b in led.batches(("pending", "balance_queued")):
        led.update_batch(b["batch_key"], next_attempt_ts=0)
    s = SUB.submit_pending(rd, led, api, {"submit_spacing_s": 0}, log=lambda m: None)
    assert s["submitted"] == 3                                               # nothing dropped: everything goes out once the refusal clears
    led.close()


def test_a_provider_account_failure_pauses_the_run(tmp_path):
    rd = _run(tmp_path, n=10)
    api = FakeAPI(fail_message="HTTP 401: Incorrect API key provided: sk-proj-****abcd.")
    h = Harvester(rd, api, log=lambda m: None)
    for _ in range(4):
        h.cycle()
    assert h.paused()
    assert h.ledger.outcome_counts() == {"batch_failed_auth": 10}
    assert all(r["terminal"] == 0 for r in h.ledger.iter_requests())        # nothing given up on
    assert api.submits == 1                                                  # and nothing re-sent into the same failure
    assert not h.busy()
    assert h.final_state()[0] == EXIT_PAUSED
    h.ledger.close()


def test_one_writer_per_run_directory(tmp_path):
    rd = _run(tmp_path, n=3)
    a = Ledger(rd).writer()
    from rejudge.common.state import WriterLocked
    with pytest.raises(WriterLocked):
        Ledger(rd).writer()
    a.close()


# --------------------------------------------------------------------------------------------
# filing
# --------------------------------------------------------------------------------------------
EXISTING_JUDGES = {"50095c69789db8b367cd38cb167ed9a768606bfac3aa843d035a4cfa5a233dfa", "2bad2e6b4fc42a66b2a8d833ecbd08ac7bec8f05ddc8e812c156fab3ef626fc4",
                   "c10c253f156c334234682c802c7b412132b5878861d5b992eb129beddd0d8539"}


def test_strongreject_judges_are_new_rows_with_the_haiku_rubric():
    for kind, rubric_sha in (("sr", "282b79aa0a40d816506b70290e878c68d775da11a8c380a38b8569f273ba1778"),
                             ("sr2", "c8c98abbac0df19e782466c9d2ffec1a37be2e2f0e37698106b351e52c9d484f")):
        j = FR.judge_spec(kind)
        assert j["rubric_sha256"] == rubric_sha
        assert j["judge_backend"] == "batch" and j["judge_model"] == "minimax-m3" and j["rubric_name"] == "strongreject"
        assert j["judge_id"] not in EXISTING_JUDGES
        assert j["judge_id"] == identity.judge_id(j)
    assert FR.judge_spec("sr", max_tokens=1500)["judge_id"] == FR.judge_spec("sr", max_tokens=4000)["judge_id"]


def test_filing_writes_valid_records_and_never_touches_run_state(tmp_path):
    run_id = "sdfbase_full22:codeattack:7c6751db529f:1913d3f96e3676ae"
    units = [("goal_a", 0, 0, -1, "k1", "sr"), ("goal_a", 0, 1, -1, "k2", "sr"), ("goal_b", 0, 0, -1, "k3", "sr"), ("goal_c", 0, 0, -1, "k4", "sr")]
    outcomes = {"k1": {"outcome": "verdict", "verdict": {"refused": 0, "convincing": 4, "specific": 5}, "batch_id": "b1", "attempts": 1,
                       "harvested_utc": "2026-11-10T05:00:00Z", "provider": "OpenAI"},
                "k2": {"outcome": "verdict", "verdict": {"refused": 1, "convincing": 2, "specific": 1}, "batch_id": "b1", "attempts": 1,
                       "harvested_utc": "2026-11-10T05:00:00Z"},
                "k3": {"outcome": "moderation", "verdict": None, "error": {"kind": "moderation", "message": "flagged"}, "status_code": 403,
                       "batch_id": "b1", "attempts": 1, "harvested_utc": "2026-11-10T05:00:00Z"}}
    judge = FR.judge_spec("sr")
    f = F.Filer(str(tmp_path / "staging"), "strongreject_minimax_m3", judge, "test", "0" * 39 + "1")
    f.stage_judge()
    jid = identity.judgment_id(run_id, judge["judge_id"])
    rows, info = FR.run_rows(run_id, jid, units, outcomes, F.utc_ms())
    by_metric = {}
    for r in rows:
        by_metric.setdefault(r["metric_id"], []).append(r)
    assert len(by_metric["strongreject"]) == 2 and len(by_metric["strongreject_unscored"]) == 1
    assert by_metric["strongreject_unscored"][0]["label"] == "moderation" and by_metric["strongreject_unscored"][0]["value"] is None
    assert [r["item_id"] for r in by_metric["strongreject_best"]] == ["goal_a"]   # goal_b moderated, goal_c unjudged: no best row
    assert by_metric["strongreject_best"][0]["value"] == pytest.approx(7 / 8)
    assert info["counts"] == {"verdict": 2, "moderation": 1, "unjudged": 1}
    judgment = F.judgment_payload(run_id=run_id, judge_id=judge["judge_id"], status="partial", n_items_judged=2, n_judge_refusals=0,
                                  n_errors=1, started_utc="2026-11-10T05:00:00Z", finished_utc="2026-11-10T05:00:00Z", batch_ids=["b1"],
                                  grading_code_sha="0" * 39 + "1", rubric_sha256=judge["rubric_sha256"], judge_cache_kind="sr", store_uri=None,
                                  producer_session="test", extra={})
    f.stage_run(run_id, judgment, rows)
    rep = f.validate()
    assert rep["refusals"] == []
    assert rep["records"] == {"judge": 1, "judgment": 1, "item_result": len(rows)}
    kinds = set()
    for p in f.files:
        for _, rec, _ in RR.read_record_file(p):
            kinds.add(rec["record_type"])
            if rec["record_type"] == "judgment":
                assert identity.judgment_id_form(rec["payload"]) == "run_judge"
            if rec["record_type"] == "item_result":
                assert rec["payload"]["judgment_id"] == jid and rec["payload"]["source_revision_id"] == ""
    assert kinds == {"judge", "judgment", "item_result"}                     # no run_state, run_patch, run_metric, supersede
    # every file lives in the rejudge namespace, the judge first
    rels = [rel for _, rel in f.staged_files()]
    assert all(rel.startswith(f"rejudge/strongreject_minimax_m3__{judge['judge_id'][:12]}/") for rel in rels)
    assert [rel for _, rel in f.staged_files("judge")] == [r for r in rels if "/judge/" in r]
    # delivery: into the inbox, refused on any name collision
    root = tmp_path / "root"
    (root / "records").mkdir(parents=True)
    dests = f.deliver(str(root), only="judge")
    assert len(dests) == 1 and dests[0].startswith(str(root / "inbox" / F.DEFAULT_HOST_LABEL))
    # delivering again is a no-op: the identical bytes already stand (a delivery interrupted part-way resumes the same way)
    assert f.deliver(str(root), only="judge") == [] and f.last_delivery == {"written": 0, "already_standing_identical": 1}
    collide = root / "records" / [rel for _, rel in f.staged_files("runs")][0]
    collide.parent.mkdir(parents=True)
    collide.write_text("{}")
    with pytest.raises(F.FilingRefused):
        f.deliver(str(root), only="runs")
    assert not (root / "inbox" / F.DEFAULT_HOST_LABEL / [rel for _, rel in f.staged_files("runs")][0]).exists()


# --------------------------------------------------------------------------------------------
# the scratch proof's comparison
# --------------------------------------------------------------------------------------------
def _toy_db(path):
    import sqlite3
    con = sqlite3.connect(path)
    con.executescript("""
    CREATE TABLE run (run_id TEXT PRIMARY KEY, judge_id TEXT, record_uri TEXT, ingest_seq INTEGER);
    CREATE TABLE judgment (judgment_id TEXT PRIMARY KEY, run_id TEXT, judge_id TEXT, status TEXT, record_uri TEXT);
    CREATE TABLE item_result (run_id TEXT, item_id TEXT, judgment_id TEXT, value REAL, record_file_id INTEGER);
    CREATE TABLE record_ingest (file_uri TEXT, record_file_id INTEGER);
    CREATE VIEW v_paper_current AS SELECT r.run_id, r.judge_id, r.ingest_seq, (SELECT COUNT(*) FROM judgment j WHERE j.run_id = r.run_id AND j.judge_id = r.judge_id) AS n FROM run r;
    INSERT INTO run VALUES ('r1', 'haiku', 'records/c/r1/000001-run.jsonl', 1), ('r2', 'haiku', 'records/c/r2/000001-run.jsonl', 2);
    INSERT INTO judgment VALUES ('j1', 'r1', 'haiku', 'complete', 'records/c/r1/000002-judgment.jsonl');
    INSERT INTO item_result VALUES ('r1', 'g', 'j1', 0.5, 7);
    INSERT INTO record_ingest VALUES ('records/c/r1/000001-run.jsonl', 5);
    """)
    con.commit()
    return con


def test_scratch_proof_passes_a_side_by_side_filing_and_fails_an_overwrite(tmp_path):
    from rejudge.common import scratch_proof as SP
    db = str(tmp_path / "r.sqlite")
    con = _toy_db(db)
    before = str(tmp_path / "before.json")
    SP.main(["digest", "--db", db, "--out", before, "--views"])
    # the filing: a new judgment + its items + its file, and the run row re-pointed (bookkeeping only)
    con.executescript("""
    INSERT INTO judgment VALUES ('jl', 'r1', 'newjudge', 'complete', 'records/rejudge/t__newjudge/r1/000001-judgment.jsonl');
    INSERT INTO item_result VALUES ('r1', 'g', 'jl', 0.9, 9);
    INSERT INTO record_ingest VALUES ('records/rejudge/t__newjudge/r1/000001-judgment.jsonl', 9);
    UPDATE run SET record_uri = 'records/rejudge/t__newjudge/r1/000001-judgment.jsonl', ingest_seq = 3 WHERE run_id = 'r1';
    """)
    con.commit()
    after = str(tmp_path / "after.json")
    SP.main(["digest", "--db", db, "--out", after, "--new-judge", "newjudge", "--views"])
    assert SP.main(["compare", before, after]) == 0
    rep_after = json.load(open(after))
    assert rep_after["tables"]["run"]["repointed"] == 1
    # an overwrite of an existing judge's row is caught
    con.execute("UPDATE item_result SET value = 0.1 WHERE judgment_id = 'j1'")
    con.commit()
    after2 = str(tmp_path / "after2.json")
    SP.main(["digest", "--db", db, "--out", after2, "--new-judge", "newjudge", "--views"])
    assert SP.main(["compare", before, after2]) == 1
    # so is a change of the run's primary judge
    con.execute("UPDATE item_result SET value = 0.5 WHERE judgment_id = 'j1'")
    con.execute("UPDATE run SET judge_id = 'newjudge' WHERE run_id = 'r2'")
    con.commit()
    after3 = str(tmp_path / "after3.json")
    SP.main(["digest", "--db", db, "--out", after3, "--new-judge", "newjudge", "--views"])
    assert SP.main(["compare", before, after3]) == 1


# --------------------------------------------------------------------------------------------
# account and credential failures, floods, refusals, spend
# --------------------------------------------------------------------------------------------
#: two account-level failures a provider can return (a blocked account, a dead own-provider key); the key fragment is masked and synthetic
BLOCK = "Policy Violation: this user has been blocked for a previous policy violation"
DEAD_KEY = "HTTP 401: Incorrect API key provided: sk-proj-****abcd. You can find your API key at https://platform.openai.com/account/api-keys."


def test_a_mid_run_account_block_returned_per_request_pauses_and_records_nothing_against_the_items(tmp_path):
    """Batch 0 answers, batches 1-2 return the account block per request: every blocked request is a non-terminal `account_error` (never
    a terminal `moderation` that would let the loop report done), the run pauses, and once the account is fixed they are re-asked
    without having spent any re-ask budget."""
    rd = _run(tmp_path)
    first = set(BF.read_manifest(rd, BF.list_manifests(rd)[0])["custom_ids"])
    blocked = {"on": True}

    def outcome(cid, i):
        if cid in first or not blocked["on"]:
            return "verdict"
        return {"code": 403, "message": BLOCK, "metadata": {"provider_name": "OpenAI", "error_type": "refusal"}}
    api = FakeAPI(outcome=outcome)
    h = Harvester(rd, api, log=lambda m: None)
    for _ in range(4):
        h.cycle()
    assert h.ledger.outcome_counts() == {"verdict": 10, "account_error": 20}
    assert not any(r["terminal"] for r in h.ledger.iter_requests("account_error"))
    assert "account/credential" in h.paused() and BLOCK in h.paused()
    assert not h.busy() and h.final_state()[0] == EXIT_PAUSED
    assert api.submits == 3 and not [k for k in BF.list_manifests(rd) if k.startswith("retry")]     # nothing re-sent into the block
    # the person fixes the account and removes PAUSED: the 20 are re-asked and judged
    blocked["on"] = False
    os.remove(os.path.join(rd, "PAUSED"))
    for _ in range(4):
        h.cycle()
    assert h.ledger.outcome_counts() == {"verdict": 30}
    assert h.final_state()[0] == EXIT_DONE
    h.ledger.close()


def test_an_identical_error_flood_is_held_until_a_person_rules_on_it(tmp_path):
    """An unrecognised failure that looks like moderation (or anything else) on most of a batch is HELD, not recorded: the run pauses;
    acknowledging the signature as a real per-item outcome re-asks the held requests once and records them as such."""
    rd = _run(tmp_path, n=20, max_requests=20)
    api = FakeAPI(outcome=lambda cid, i: "moderation" if i < 15 else "verdict")
    h = Harvester(rd, api, log=lambda m: None)
    for _ in range(3):
        h.cycle()
    assert h.ledger.outcome_counts() == {"held": 15, "verdict": 5}
    assert not any(r["terminal"] for r in h.ledger.iter_requests("held"))
    held = list(h.ledger.iter_requests("held"))[0]["error"]
    assert held["parsed_outcome"] == "moderation" and "identical-error flood" in h.paused() and held["signature"] in h.paused()
    assert h.final_state()[0] == EXIT_PAUSED
    with open(os.path.join(rd, "ACKNOWLEDGED"), "w") as fh:
        fh.write(f"{held['signature']}  # genuine moderation of jailbreak transcripts\n")
    os.remove(os.path.join(rd, "PAUSED"))
    for _ in range(4):
        h.cycle()
    assert h.ledger.outcome_counts() == {"moderation": 15, "verdict": 5}
    assert all(r["terminal"] for r in h.ledger.iter_requests("moderation"))
    assert h.final_state()[0] == EXIT_DONE
    h.ledger.close()


def test_truncation_pauses_and_a_re_ask_raises_the_cap(tmp_path):
    rd = _run(tmp_path, n=40, max_requests=40)
    seen = {}

    def outcome(cid, i):
        seen[cid] = seen.get(cid, 0) + 1
        return "truncated" if i < 10 and seen[cid] == 1 else "verdict"
    api = FakeAPI(outcome=outcome)
    h = Harvester(rd, api, log=lambda m: None)
    for _ in range(3):
        h.cycle()
    assert "truncation" in h.paused() and h.ledger.outcome_counts() == {"truncated": 10, "verdict": 30}
    assert len(api.bodies) == 1                                             # no re-ask into a paused run
    os.remove(os.path.join(rd, "PAUSED"))
    for _ in range(4):
        h.cycle()
    retry = api.bodies[1]
    raised = J.get()["truncated_retry_max_tokens"]
    assert {r["body"]["max_tokens"] for r in retry["requests"]} == {raised} and raised > J.get()["request"]["max_tokens"]
    assert len(retry["requests"]) == 10
    assert h.ledger.outcome_counts() == {"verdict": 40} and h.final_state()[0] == EXIT_DONE
    h.ledger.close()


def test_a_truncated_call_with_no_higher_cap_is_terminal_at_once(tmp_path):
    rd = _run(tmp_path, n=10, max_requests=10, extra={"truncated_retry_max_tokens": 0})
    api = FakeAPI(outcome=lambda cid, i: "truncated" if i == 0 else "verdict")
    h = Harvester(rd, api, log=lambda m: None)
    for _ in range(4):
        h.cycle()
    (t,) = list(h.ledger.iter_requests("truncated"))
    assert t["terminal"] == 1 and "no higher cap" in t["error"]["finalized"]
    assert api.submits == 1 and h.final_state()[0] == EXIT_DONE
    h.ledger.close()


def test_batch_failures_do_not_spend_the_items_re_asks(tmp_path):
    """Two unnamed batch failures, then a malformed call: the parse error still gets its one re-ask (before the fix one shared counter
    made it terminal on the third attempt)."""
    rd = _run(tmp_path, n=5, max_requests=5)
    seen = {}

    def outcome(cid, i):
        seen[cid] = seen.get(cid, 0) + 1
        return "malformed" if i == 0 and seen[cid] == 1 else "verdict"
    api = FakeAPI(outcome=outcome, fail_message="internal validation error", fail_first=2)
    h = Harvester(rd, api, log=lambda m: None)
    for _ in range(12):
        h.cycle()
    assert h.ledger.outcome_counts() == {"verdict": 5}
    first = BF.read_manifest(rd, "b00000")["custom_ids"]
    assert h.ledger.attempt_outcomes(first[0]) == ["batch_failed", "batch_failed", "parse_error", "verdict"]
    h.ledger.close()


def test_a_failure_naming_one_request_requeues_the_others_uncounted(tmp_path):
    rd = _run(tmp_path, n=6, max_requests=6)
    ids = BF.read_manifest(rd, "b00000")["custom_ids"]
    api = FakeAPI(fail_message=lambda cids: f"Batch request '{cids[0]}' contains prohibited content.", fail_first=1)
    h = Harvester(rd, api, log=lambda m: None)
    for _ in range(6):
        h.cycle()
    assert h.ledger.get(ids[0])["outcome"] == "moderation" and h.ledger.get(ids[0])["terminal"] == 1
    assert all(h.ledger.attempt_outcomes(c) == ["batch_failed_other", "verdict"] for c in ids[1:])
    assert h.final_state()[0] == EXIT_DONE
    h.ledger.close()


def test_a_balance_error_inside_a_batch_is_re_asked_after_a_backoff(tmp_path):
    rd = _run(tmp_path, n=10, max_requests=10)
    seen = {}

    def outcome(cid, i):
        seen[cid] = seen.get(cid, 0) + 1
        if i == 0 and seen[cid] == 1:
            return {"code": 402, "message": "Insufficient credits", "metadata": {"limit_source": "openrouter_key_limit"}}
        return "verdict"
    api = FakeAPI(outcome=outcome)
    h = Harvester(rd, api, log=lambda m: None)
    for _ in range(3):
        h.cycle()
    assert h.ledger.outcome_counts() == {"balance_error": 1, "verdict": 9} and not h.paused()
    (b,) = [b for b in h.ledger.batches(("pending",)) if b["batch_key"].startswith("retry")]
    assert b["batch_key"].startswith("retry1b_") and b["next_attempt_ts"] > time.time() + 200
    assert h.busy()                                                          # waiting on the backoff, not done
    h.ledger.update_batch(b["batch_key"], next_attempt_ts=0)
    for _ in range(3):
        h.cycle()
    assert h.ledger.outcome_counts() == {"verdict": 10} and h.final_state()[0] == EXIT_DONE
    h.ledger.close()


@pytest.mark.parametrize("status,message", [(403, BLOCK), (401, "Invalid API key"), (404, "No eligible :batch endpoint for this model"),
                                            (400, "requests must come last")])
def test_a_refused_submit_pauses_the_run_and_stops_the_pass(tmp_path, status, message):
    """Before the fix a 403 / 401 / 404 refusal was re-POSTed three times per batch and then left 'refused' while the loop exited 0."""
    rd = _run(tmp_path)
    api = FakeAPI(refuse=[OpenRouterError(classify_error(status, message), status, message)])
    h = Harvester(rd, api, log=lambda m: None)
    out = h.cycle()
    assert api.submits == 1 and out["submit"]["submitted"] == 0 and out["submit"]["stopped"]
    assert h.ledger.state_counts() == {"pending": 3} and message in h.paused()
    assert not h.busy() and h.final_state()[0] == EXIT_PAUSED
    prog = h.progress(done=True)
    assert prog["done"] is False and prog["readiness"]["paused"]
    h.ledger.close()


def test_a_set_aside_batch_leaves_the_run_unfinished_not_done(tmp_path):
    rd = _run(tmp_path)
    h = Harvester(rd, FakeAPI(), log=lambda m: None)
    SUB.register_all(rd, h.ledger)
    h.set_aside("b00002", "operator: a body the API cannot take")
    for _ in range(4):
        h.cycle()
    assert h.ledger.state_counts() == {"harvested": 2, "refused": 1}
    assert not h.busy()
    code, line = h.final_state()
    assert code == EXIT_UNFINISHED and "b00002" in line
    assert h.progress(done=True)["done"] is False
    h.ledger.close()


def test_the_spend_ceiling_pauses_before_a_submit_would_pass_it(tmp_path):
    rd = _run(tmp_path, init={"max_spend_usd": 1e-9})
    api = FakeAPI()
    h = Harvester(rd, api, log=lambda m: None)
    out = h.cycle()
    assert out["submit"]["stopped"] == "spend_ceiling" and api.submits == 0 and "spend ceiling" in h.paused()
    h.ledger.close()


def test_the_full_run_waits_for_the_pilot_report(tmp_path):
    report = tmp_path / "pilot" / "pilot_report.json"
    rd = _run(tmp_path, init={"requires_files": [str(report)]})
    api = FakeAPI()
    h = Harvester(rd, api, log=lambda m: None)
    assert h.cycle()["submit"]["stopped"] == "gated" and api.submits == 0 and "pilot" in h.paused()
    report.parent.mkdir()
    report.write_text("{}")
    os.remove(os.path.join(rd, "PAUSED"))
    assert h.cycle()["submit"]["submitted"] == 3
    h.ledger.close()


def test_retry_manifests_carry_a_cost_estimate(tmp_path):
    rd = _run(tmp_path, n=10, max_requests=10)
    seen = {}

    def outcome(cid, i):
        seen[cid] = seen.get(cid, 0) + 1
        return "error" if i == 0 and seen[cid] == 1 else "verdict"
    h = Harvester(rd, FakeAPI(outcome=outcome), log=lambda m: None)
    for _ in range(3):
        h.cycle()
    (key,) = [k for k in BF.list_manifests(rd) if k.startswith("retry")]
    assert BF.read_manifest(rd, key)["est"]["cost_usd_with_fee"] > 0
    h.ledger.close()


# --------------------------------------------------------------------------------------------
# the filing gate, infrastructure outcomes, follow-up filings
# --------------------------------------------------------------------------------------------
RUN = "sdfbase_full22:codeattack:7c6751db529f:1913d3f96e3676ae"
KEYS = [f"{i:064x}" for i in range(1, 5)]


def _filing_world(tmp_path):
    """A run directory whose ledger holds four keys of one paper run, the jobs and units files, an empty registry root."""
    rd = str(tmp_path / "rundir")
    T.init_run(rd)
    led = Ledger(rd).writer()
    led.register_manifest({"batch_key": "b00000", "n_requests": 4, "body_bytes": 1, "custom_ids": KEYS})
    led.update_batch("b00000", state="harvested")
    for k in KEYS[:3]:
        led.record_outcome(k, "b00000", "batch-1", "verdict", True, {"refused": 0, "convincing": 4, "specific": 3}, None, 200, "stop",
                           None, None, "OpenAI")
    led.close()
    jobs = tmp_path / "jobs.jsonl"
    jobs.write_text(json.dumps({"run_id": RUN, "kind": "sr", "haiku_judgment_id": "h" * 36, "haiku_judge_id": FR.HAIKU_JUDGES["sr"]}) + "\n")
    units = tmp_path / "units.tsv.gz"
    with gzip.open(units, "wt") as fh:
        fh.write("run_id\titem_id\ttrial\tsample\tturn\tcustom_id\tkind\tresponse_chars\n")
        for i, k in enumerate(KEYS):
            fh.write(f"{RUN}\tgoal_{i % 2}\t0\t{i}\t-1\t{k}\tsr\t10\n")
    root = tmp_path / "root"
    (root / "records").mkdir(parents=True)
    return rd, str(jobs), str(units), str(root)


def _record(rd, k, outcome, terminal):
    led = Ledger(rd).writer()
    led.record_outcome(k, "b00000", "batch-2", outcome, terminal, {"refused": 1, "convincing": 1, "specific": 1} if outcome == "verdict" else None,
                       None if outcome == "verdict" else {"kind": outcome, "message": "upstream"}, 500 if outcome != "verdict" else 200, None, None,
                       None, "OpenAI")
    led.close()


def _file(tmp_path, rd, jobs, units, root, staging, *extra):
    return FR.main(["--jobs", jobs, "--units", units, "--run-dir", rd, "--staging", str(tmp_path / staging), "--code-sha", "0" * 39 + "1",
                    "--registry-root", root, *extra])


def _inbox(root):
    return sorted(str(p.relative_to(root)) for p in pathlib.Path(root, "inbox").rglob("*.jsonl*")) if os.path.isdir(os.path.join(root, "inbox")) else []


def _merge(root):
    """What the registry harvester's inbox merge does: move inbox/<host>/<rel> to records/<rel>."""
    ib = pathlib.Path(root, "inbox", F.DEFAULT_HOST_LABEL)
    for p in sorted(ib.rglob("*.jsonl*")):
        dest = pathlib.Path(root, "records", p.relative_to(ib))
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.replace(p, dest)


def test_infrastructure_outcomes_are_not_filed_as_item_rows_by_default():
    units = [("g", 0, 0, -1, "k1", "sr"), ("g", 0, 1, -1, "k2", "sr"), ("h", 0, 0, -1, "k3", "sr")]
    outcomes = {"k1": {"outcome": "verdict", "verdict": {"refused": 0, "convincing": 4, "specific": 5}, "harvested_utc": "2026-11-10T05:00:00Z"},
                "k2": {"outcome": "provider_error", "error": {"kind": "provider_error", "message": "upstream"}, "harvested_utc": "2026-11-10T05:00:00Z"},
                "k3": {"outcome": "moderation", "error": {"kind": "moderation", "message": "flagged"}, "harvested_utc": "2026-11-10T05:00:00Z"}}
    jid = identity.judgment_id(RUN, FR.judge_spec("sr")["judge_id"])
    rows, info = FR.run_rows(RUN, jid, units, outcomes, F.utc_ms())
    assert sorted((r["metric_id"], r["label"]) for r in rows) == [("strongreject", None), ("strongreject_unscored", "moderation")]
    assert info["counts"]["infra_unfiled"] == 1 and info["unsettled"] == 1 and info["filed_unscored"] == {"moderation": 1}
    rows2, info2 = FR.run_rows(RUN, jid, units, outcomes, F.utc_ms(), file_infra=True)
    assert ("strongreject_unscored", "provider_error") in {(r["metric_id"], r["label"]) for r in rows2} and info2["unsettled"] == 0


def test_every_unit_row_names_its_provider_and_attempt_with_one_spelling():
    """Every unit row, a verdict or an unscored one, from a batch or a live ledger, names the provider that served its outcome and its
    attempt under the verdict rows' key ``attempt``. The paper's first StrongREJECT filing (192df392 at dad98c586) named the provider on a
    live unscored row only, so its 8 parse_error and 4 truncated batch rows carry none, and its unscored rows spell the key
    ``attempts``; the C/P/I filings name DeepInfra on the same rows."""
    h = "2026-11-10T05:00:00Z"
    units = [("g", 0, 0, -1, "kv", "sr"), ("g", 0, 1, -1, "kp", "sr"), ("g", 0, 2, -1, "kt", "sr"), ("h", 0, 0, -1, "km", "sr"),
             ("h", 0, 1, -1, "klv", "sr"), ("h", 0, 2, -1, "klt", "sr"), ("i", 0, 0, -1, "kx", "sr")]
    outcomes = {
        "kv": {"outcome": "verdict", "verdict": {"refused": 0, "convincing": 4, "specific": 5}, "batch_id": "b1", "attempts": 1,
               "provider": "DeepInfra", "backend": "batch", "harvested_utc": h},
        "kp": {"outcome": "parse_error", "error": {"kind": "parse_error", "message": "no tool call"}, "batch_id": "b1", "attempts": 2,
               "provider": "DeepInfra", "backend": "batch", "harvested_utc": h},
        "kt": {"outcome": "truncated", "error": {"kind": "truncated", "message": "finish_reason length"}, "batch_id": "b2", "attempts": 3,
               "provider": "DeepInfra", "backend": "batch", "harvested_utc": h},
        "km": {"outcome": "moderation", "error": {"kind": "moderation", "message": "flagged"}, "status_code": 403, "batch_id": "b1",
               "attempts": 1, "provider": None, "backend": "batch", "harvested_utc": h},      # rejected before any provider served it
        "klv": {"outcome": "verdict", "verdict": {"refused": 1, "convincing": 1, "specific": 1}, "batch_id": None, "attempts": 1,
                "provider": "GMICloud", "backend": "live", "quantization": "fp8", "generation_id": "gen-1", "harvested_utc": h},
        "klt": {"outcome": "truncated", "error": {"kind": "truncated", "message": "finish_reason length"}, "batch_id": None, "attempts": 2,
                "provider": "Parasail", "backend": "live", "quantization": "fp8", "generation_id": "gen-2", "harvested_utc": h},
        "kx": {"outcome": "provider_error", "error": {"kind": "provider_error", "message": "upstream"}, "batch_id": "b2", "attempts": 4,
               "provider": "DeepInfra", "backend": "batch", "harvested_utc": h}}
    jid = identity.judgment_id(RUN, FR.judge_spec("sr")["judge_id"])
    rows, _ = FR.run_rows(RUN, jid, units, outcomes, F.utc_ms(), file_infra=True)
    by_key = {r["extra"]["request_key"]: r for r in rows if r["metric_id"] in ("strongreject", "strongreject_unscored")}
    assert sorted(by_key) == sorted(outcomes)
    for cid, r in by_key.items():
        assert "provider" in r["extra"] and r["extra"]["provider"] == outcomes[cid]["provider"], (cid, r["metric_id"], r["extra"])
        assert r["extra"].get("attempt") == outcomes[cid]["attempts"] and "attempts" not in r["extra"], (cid, r["metric_id"], r["extra"])
    assert {k: by_key[k]["extra"]["transport"] for k in ("kp", "kt", "klt")} == {"kp": "openrouter_batch", "kt": "openrouter_batch",
                                                                                  "klt": "openrouter_live"}


def test_the_delivery_gate_refuses_a_moving_or_partial_run(tmp_path):
    rd, jobs, units, root = _filing_world(tmp_path)
    # the fourth key is waiting on a re-ask: nothing is delivered, whatever the flags
    _record(rd, KEYS[3], "provider_error", False)
    assert _file(tmp_path, rd, jobs, units, root, "s1", "--deliver", "all", "--allow-partial") == 3 and _inbox(root) == []
    # PAUSED refuses too
    _record(rd, KEYS[3], "provider_error", True)
    open(os.path.join(rd, "PAUSED"), "w").write("account\n")
    assert _file(tmp_path, rd, jobs, units, root, "s2", "--deliver", "all", "--allow-partial") == 3 and _inbox(root) == []
    os.remove(os.path.join(rd, "PAUSED"))
    # the fourth key ended on an infrastructure outcome: the run is partial -> refused unless --allow-partial
    assert _file(tmp_path, rd, jobs, units, root, "s3", "--deliver", "all") == 3 and _inbox(root) == []
    assert _file(tmp_path, rd, jobs, units, root, "s4", "--deliver", "all", "--allow-partial") == 0
    rep = json.load(open(tmp_path / "s4" / "filing_report.json"))
    assert rep["runs"][RUN]["status"] == "partial" and rep["runs"][RUN]["infra_unfiled"] == 1
    assert len(_inbox(root)) == 3                                            # judge, judgment, item_result


def test_deliver_only_refuses_when_the_ledger_changed_since_staging(tmp_path):
    rd, jobs, units, root = _filing_world(tmp_path)
    _record(rd, KEYS[3], "verdict", True)
    assert _file(tmp_path, rd, jobs, units, root, "st", "--dry-run") == 0
    base = ["--jobs", jobs, "--units", units, "--run-dir", rd, "--staging", str(tmp_path / "st"), "--registry-root", root, "--deliver-only"]
    FR.main(base + ["--deliver", "judge"])
    _record(rd, KEYS[3], "verdict", True)                                    # one more attempt recorded after staging
    with pytest.raises(SystemExit) as e:
        FR.main(base + ["--deliver", "runs"])
    assert "changed since" in str(e.value)
    assert len(_inbox(root)) == 1                                            # the judge only


def test_a_follow_up_filing_is_staged_above_the_registrys_names_and_delivers(tmp_path):
    rd, jobs, units, root = _filing_world(tmp_path)
    _record(rd, KEYS[3], "provider_error", True)
    assert _file(tmp_path, rd, jobs, units, root, "first", "--deliver", "all", "--allow-partial") == 0
    _merge(root)
    first = sorted(str(p.relative_to(root)) for p in pathlib.Path(root, "records").rglob("*.jsonl*"))
    assert sorted(os.path.basename(p) for p in first) == ["000001-judge.jsonl", "000001-judgment.jsonl", "000002-item_result.jsonl.gz"]
    # the infrastructure failure is re-asked later and judged: a follow-up filing of the same run
    _record(rd, KEYS[3], "verdict", True)
    assert _file(tmp_path, rd, jobs, units, root, "second", "--deliver", "all") == 0
    rep = json.load(open(tmp_path / "second" / "filing_report.json"))
    assert rep["runs"][RUN]["status"] == "complete" and rep["gate"]["follow_up_runs"] == [RUN]
    second = _inbox(root)
    assert sorted(os.path.basename(p) for p in second) == ["000003-judgment.jsonl", "000004-item_result.jsonl.gz"]   # no judge again, no name reused
    # the follow-up judgment names the same judgment id (the registry folds it over the first)
    recs = [rec for p in second for _, rec, _ in RR.read_record_file(os.path.join(root, p))]
    (jg,) = [r for r in recs if r["record_type"] == "judgment"]
    assert jg["payload"]["judgment_id"] == identity.judgment_id(RUN, FR.judge_spec("sr")["judge_id"]) and jg["payload"]["status"] == "complete"
    assert jg["payload"]["extra"]["follow_up_of_seq"] == 2


def test_scratch_proof_compares_a_follow_up_filing_against_a_before_that_holds_the_first(tmp_path):
    from rejudge.common import scratch_proof as SP
    db = str(tmp_path / "r.sqlite")
    con = _toy_db(db)
    con.executescript("""
    INSERT INTO judgment VALUES ('jl', 'r1', 'newjudge', 'partial', 'records/rejudge/t__newjudge/r1/000001-judgment.jsonl');
    INSERT INTO item_result VALUES ('r1', 'g', 'jl', 0.9, 9);
    INSERT INTO record_ingest VALUES ('records/rejudge/t__newjudge/r1/000001-judgment.jsonl', 9);
    UPDATE run SET record_uri = 'records/rejudge/t__newjudge/r1/000001-judgment.jsonl', ingest_seq = 3 WHERE run_id = 'r1';
    """)
    con.commit()
    before = str(tmp_path / "before.json")
    SP.main(["digest", "--db", db, "--out", before, "--new-judge", "newjudge", "--views"])
    # the follow-up: the same judgment refolded complete, its item row replaced, one more row, the run re-pointed again
    con.executescript("""
    UPDATE judgment SET status = 'complete', record_uri = 'records/rejudge/t__newjudge/r1/000003-judgment.jsonl' WHERE judgment_id = 'jl';
    UPDATE item_result SET value = 0.7, record_file_id = 11 WHERE judgment_id = 'jl';
    INSERT INTO item_result VALUES ('r1', 'h', 'jl', 0.2, 11);
    INSERT INTO record_ingest VALUES ('records/rejudge/t__newjudge/r1/000003-judgment.jsonl', 11);
    UPDATE run SET record_uri = 'records/rejudge/t__newjudge/r1/000003-judgment.jsonl', ingest_seq = 4 WHERE run_id = 'r1';
    """)
    con.commit()
    after = str(tmp_path / "after.json")
    SP.main(["digest", "--db", db, "--out", after, "--new-judge", "newjudge", "--views"])
    assert SP.main(["compare", before, after]) == 0
    con.execute("UPDATE judgment SET status = 'superseded' WHERE judgment_id = 'j1'")      # an existing judge's row: caught
    con.commit()
    after2 = str(tmp_path / "after2.json")
    SP.main(["digest", "--db", db, "--out", after2, "--new-judge", "newjudge", "--views"])
    assert SP.main(["compare", before, after2]) == 1


# --------------------------------------------------------------------------------------------
# run control: a POST is recorded before it is made, the gates are re-read before every POST, a 413's children keep the estimate,
# spend is counted on the safe side, a batch that ends expired / failed pauses, and the ramp holds the first pass to a canary and then
# to a limit set by measured turnaround
# --------------------------------------------------------------------------------------------
class Died(BaseException):
    """The process ending mid-POST (a hard stop, a crash): not an error the submitter can catch."""


class DyingAPI(FakeAPI):
    """``die_after_accept``: the server takes the batch, then the process dies before the reply is read. ``die_before_accept``: the
    process dies while the bytes go out (the server never sees a batch)."""

    def __init__(self, die_after_accept=0, die_before_accept=0, **kw):
        super().__init__(**kw)
        self.die_after_accept = die_after_accept
        self.die_before_accept = die_before_accept

    def submit(self, body):
        if self.die_before_accept:
            self.die_before_accept -= 1
            self.submits += 1
            raise Died()
        out = super().submit(body)
        if self.die_after_accept:
            self.die_after_accept -= 1
            raise Died()
        return out


def test_a_batch_whose_post_was_cut_off_after_the_server_took_it_is_adopted_not_posted_again(tmp_path):
    rd = _run(tmp_path, n=20, max_requests=10)
    api = DyingAPI(die_after_accept=1)
    led = Ledger(rd).writer()
    with pytest.raises(Died):
        SUB.submit_pending(rd, led, api, {"submit_spacing_s": 0}, log=lambda m: None)
    assert led.state_counts() == {"submit_unknown": 1, "pending": 1} and len(api.batches) == 1
    led.close()
    h = Harvester(rd, api, log=lambda m: None)          # the restarted unit
    for _ in range(4):
        h.cycle()
    assert api.submits == 2 and len(api.batches) == 2                       # one POST per batch: the first was adopted
    assert h.ledger.outcome_counts() == {"verdict": 20} and h.final_state()[0] == EXIT_DONE
    assert any(r["kind"] == "ADOPTED" for r in h.ledger.conn.execute("SELECT kind FROM events"))
    h.ledger.close()


def test_a_batch_whose_post_never_reached_the_server_is_re_queued_after_the_unknown_window(tmp_path):
    rd = _run(tmp_path, n=10, max_requests=10, extra={"unknown_resubmit_s": 0})
    api = DyingAPI(die_before_accept=1)
    led = Ledger(rd).writer()
    with pytest.raises(Died):
        SUB.submit_pending(rd, led, api, {"submit_spacing_s": 0}, log=lambda m: None)
    assert led.state_counts() == {"submit_unknown": 1} and api.batches == {}
    led.close()
    time.sleep(1.1)
    h = Harvester(rd, api, log=lambda m: None)
    for _ in range(4):
        h.cycle()
    assert len(api.batches) == 1 and h.ledger.outcome_counts() == {"verdict": 10} and h.final_state()[0] == EXIT_DONE
    h.ledger.close()


def test_removing_the_go_file_mid_pass_stops_the_next_post(tmp_path):
    gate = tmp_path / "GO"
    gate.write_text("go")
    rd = _run(tmp_path, n=30, max_requests=10, init={"requires_files": [str(gate)]})

    class API(FakeAPI):
        def submit(self, body):
            out = super().submit(body)
            if os.path.exists(gate):
                os.remove(gate)              # the soft stop, while the first POST is in flight
            return out
    api = API()
    led = Ledger(rd).writer()
    s = SUB.submit_pending(rd, led, api, BF.read_track(rd), log=lambda m: None)
    assert api.submits == 1 and s["stopped"] == "gated" and "submit gate" in open(os.path.join(rd, "PAUSED")).read()
    assert led.state_counts() == {"submitted": 1, "pending": 2}
    led.close()


def test_a_pause_written_mid_pass_stops_the_next_post(tmp_path):
    rd = _run(tmp_path, n=30, max_requests=10)

    class API(FakeAPI):
        def submit(self, body):
            out = super().submit(body)
            open(os.path.join(rd, "PAUSED"), "w").write("2026-11-11T00:00:00Z a person\n")
            return out
    api = API()
    led = Ledger(rd).writer()
    s = SUB.submit_pending(rd, led, api, {"submit_spacing_s": 0}, log=lambda m: None)
    assert api.submits == 1 and s["stopped"] == "paused" and led.state_counts() == {"submitted": 1, "pending": 2}
    led.close()


def test_a_refused_post_goes_back_to_pending_not_left_unknown(tmp_path):
    rd = _run(tmp_path, n=10, max_requests=10)
    api = FakeAPI(refuse=[OpenRouterError("rate_limited", 429, "slow down", retry_after_s=60)])
    led = Ledger(rd).writer()
    SUB.submit_pending(rd, led, api, {"submit_spacing_s": 0}, log=lambda m: None)
    (b,) = led.batches()
    assert b["state"] == "pending" and b["last_error_kind"] == "rate_limited" and b["submit_attempts"] == 1
    led.close()


def test_413_children_carry_the_parents_estimate(tmp_path):
    rd = _run(tmp_path, n=10, max_requests=10)
    parent = BF.read_manifest(rd, "b00000")["est"]["cost_usd_with_fee"]
    api = FakeAPI(refuse=[OpenRouterError("too_large", 413, "more than 4 requests")])
    led = Ledger(rd).writer()
    SUB.submit_pending(rd, led, api, {"submit_spacing_s": 0}, log=lambda m: None, max_submits=0)
    SUB.submit_pending(rd, led, api, {"submit_spacing_s": 0}, log=lambda m: None, max_submits=1)
    kids = [b for b in led.batches() if b["parent"] == "b00000"]
    assert len(kids) == 3 and all(b["est_cost"] > 0 for b in kids)
    assert sum(b["est_cost"] for b in kids) == pytest.approx(parent, rel=0.05)
    led.close()


def test_committed_spend_counts_an_unreported_batch_at_its_estimate_and_scales_in_flight_by_the_measured_ratio(tmp_path):
    rd = _run(tmp_path, n=30, max_requests=10)
    led = Ledger(rd).writer()
    SUB.register_all(rd, led)
    a, b, c = (r["batch_key"] for r in led.batches())
    est = {r["batch_key"]: r["est_cost"] for r in led.batches()}
    led.update_batch(a, state="harvested", cost=2 * est[a] / (1 + BF.CREDIT_FEE))     # ran at twice its estimate
    led.update_batch(b, state="harvested", cost=None)                                # an expired batch that reported nothing
    led.update_batch(c, state="submitted")
    assert SUB.cost_ratio(led) == pytest.approx(2.0)
    assert SUB.committed_spend(led) == pytest.approx(2 * est[a] + est[b] + 2 * est[c])
    led.close()


class EndingAPI(FakeAPI):
    """The first ``n_end`` batches end with ``status`` (expired / failed), returning results for their first ``keep`` requests only."""

    def __init__(self, status="expired", n_end=1, keep=3, **kw):
        super().__init__(**kw)
        self.end_status, self.n_end, self.keep = status, n_end, keep

    def _advance(self, b):
        before = b["status"]
        super()._advance(b)
        if before == "in_progress" and b["status"] == "completed" and self.n_end:
            self.n_end -= 1
            b.update(status=self.end_status, results=b["results"][:self.keep], error={"message": f"batch {self.end_status}"})


@pytest.mark.parametrize("status", ["expired", "failed"])
def test_a_batch_that_ends_expired_or_failed_pauses_the_run_and_is_re_asked_only_after_a_person(tmp_path, status):
    rd = _run(tmp_path, n=10, max_requests=10, extra={"health": {"pause_on_batch_end": True}})
    api = EndingAPI(status=status)
    h = Harvester(rd, api, log=lambda m: None)
    for _ in range(4):
        h.cycle()
    assert status in h.paused() and "unanswered" in h.paused()
    oc = h.ledger.outcome_counts()
    assert oc["verdict"] == 3 and oc[f"batch_{status}"] == 7                         # the partial results kept, the rest not terminal
    assert api.submits == 1 and not h.busy() and h.final_state()[0] == EXIT_PAUSED   # nothing re-sent on its own
    os.remove(os.path.join(rd, "PAUSED"))
    for _ in range(4):
        h.cycle()
    assert api.submits == 2 and h.ledger.outcome_counts() == {"verdict": 10} and h.final_state()[0] == EXIT_DONE
    h.ledger.close()


def test_without_the_setting_an_expired_batch_is_re_asked_as_before(tmp_path):
    rd = _run(tmp_path, n=10, max_requests=10)
    h = Harvester(rd, EndingAPI(), log=lambda m: None)
    for _ in range(6):
        h.cycle()
    assert not h.paused() and h.ledger.outcome_counts() == {"verdict": 10}
    h.ledger.close()


RAMP = {"canary_batches": 1, "canary_min_verdict_share": 0.85, "canary_cost_margin": 1.0}


class PricedAPI(FakeAPI):
    """Reports ``cost_of(batch requests)`` as the batch's cost."""

    def __init__(self, cost_of=None, **kw):
        super().__init__(**kw)
        self.cost_of = cost_of

    def _advance(self, b):
        super()._advance(b)
        if b["status"] == "completed" and self.cost_of is not None:
            b["usage"] = {"cost": self.cost_of([r["custom_id"] for r in b["_reqs"]])}


def _ramp_run(tmp_path, n=60, max_requests=10, ramp=None, cap=1e6):
    rd = _run(tmp_path, n=n, max_requests=max_requests, extra={"ramp": dict(RAMP, **(ramp or {}))}, init={"max_spend_usd": cap})
    return rd


def test_the_canary_goes_alone_then_the_limit_its_turnaround_sets(tmp_path):
    rd = _ramp_run(tmp_path, n=55)                                   # 5 batches of 10 and one of 5
    api = FakeAPI()
    h = Harvester(rd, api, log=lambda m: None)
    out = h.cycle()
    assert out["submit"]["submitted"] == 1 and api.bodies[0]["requests"].__len__() == 5     # the smallest batch first
    assert out["submit"]["stopped"] == "ramp_canary" and h.progress(done=False)["ramp"]["phase"] == "canary"
    h.cycle()                                                        # harvested
    (can,) = [b for b in h.ledger.batches() if b["state"] == "harvested"]
    t0 = time.time()
    stamp = lambda x: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(x))      # noqa: E731 -- the canary took half an hour
    h.ledger.update_batch(can["batch_key"], submitted_utc=stamp(t0 - 3600), harvested_utc=stamp(t0 - 1800))
    h.cycle()                                                        # the next pass reads the gate, then the ramp's limit
    ev = [r["kind"] for r in h.ledger.conn.execute("SELECT kind FROM events")]
    assert "CANARY_PASSED" in ev
    rs = h.progress(done=False)["ramp"]
    assert rs["phase"] == "ramp" and rs["limit_requests"] == 20      # peak 5 x the growth cap 4 (half an hour of the 24-hour window)
    assert api.submits == 3                                          # largest first under the limit: 10 + 10 in flight, then held
    for _ in range(12):
        h.cycle()
    assert h.ledger.outcome_counts() == {"verdict": 55} and h.final_state()[0] == EXIT_DONE
    h.ledger.close()


def test_re_asks_are_never_held_by_the_canary(tmp_path):
    rd = _ramp_run(tmp_path, n=30, ramp={"canary_min_settled_share": 0.0})
    seen = {}

    def outcome(cid, i):
        seen[cid] = seen.get(cid, 0) + 1
        return "error" if i == 0 and seen[cid] == 1 else "verdict"
    api = FakeAPI(outcome=outcome)
    led = Ledger(rd).writer()
    SUB.register_all(rd, led)
    led.close()
    h = Harvester(rd, api, log=lambda m: None)
    for _ in range(3):
        h.cycle()
    retry = [b for b in h.ledger.batches() if int(b["attempt"]) > 0]
    assert retry and all(b["state"] in ("submitted", "harvested") for b in retry)
    h.ledger.close()


def test_a_canary_under_the_verdict_share_pauses_and_a_person_removing_paused_accepts_it(tmp_path):
    rd = _ramp_run(tmp_path, n=25)                                   # the smallest batch (5) is the canary
    api = FakeAPI(outcome=lambda cid, i: "malformed" if i < 2 else "verdict")   # 2 of 5 unparseable on the first try: 60% verdicts
    h = Harvester(rd, api, log=lambda m: None)
    for _ in range(4):
        h.cycle()
    p = h.paused()
    assert p and "canary gate" in p and "60.0%" in p
    assert [b["state"] for b in h.ledger.batches() if int(b["attempt"]) == 0].count("pending") == 2      # the rest of the pass held
    os.remove(os.path.join(rd, "PAUSED"))
    for _ in range(12):
        h.cycle()
    ev = [r["kind"] for r in h.ledger.conn.execute("SELECT kind FROM events")]
    assert "CANARY_FAILED" in ev and "CANARY_ACCEPTED" in ev and h.final_state()[0] == EXIT_DONE
    h.ledger.close()


def test_a_canary_whose_measured_cost_projects_over_the_ceiling_pauses(tmp_path):
    rd = _ramp_run(tmp_path, n=25)
    led = Ledger(rd).writer()
    SUB.register_all(rd, led)
    total = sum(b["est_cost"] for b in led.batches())
    led.close()
    tr = BF.read_track(rd)
    tr["max_spend_usd"] = 2.0 * total                               # room for the estimate, not for three times it
    BF.write_track(rd, tr)
    est = {}
    for k in BF.list_manifests(rd):
        m = BF.read_manifest(rd, k)
        for c in m["custom_ids"]:
            est[c] = m["est"]["cost_usd_with_fee"] / m["n_requests"]
    api = PricedAPI(cost_of=lambda ids: 3.0 * sum(est[c] for c in ids) / (1 + BF.CREDIT_FEE))
    h = Harvester(rd, api, log=lambda m: None)
    for _ in range(4):
        h.cycle()
    assert "canary gate" in h.paused() and "projects to" in h.paused() and api.submits == 1
    h.ledger.close()


def test_an_inflight_limit_file_overrides_the_ramp_and_zero_holds_the_first_pass(tmp_path):
    rd = _ramp_run(tmp_path, n=40, ramp={"canary_batches": 1})
    open(os.path.join(rd, SUB.INFLIGHT_FILE), "w").write("0\n")
    api = FakeAPI()
    h = Harvester(rd, api, log=lambda m: None)
    assert h.cycle()["submit"]["stopped"] == "ramp_limit" and api.submits == 0
    open(os.path.join(rd, SUB.INFLIGHT_FILE), "w").write("1000\n")   # a person lifts it (the canary's hold too)
    assert h.cycle()["submit"]["submitted"] == 4
    h.ledger.close()


def test_the_limit_holds_a_serial_provider_inside_the_window_and_doubles_a_parallel_one():
    ramp = dict(SUB.RAMP_DEFAULTS)
    H, N = 3600.0, 10_000
    # serial: four batches of 10k all sent at 0, done one after another at 1 h each
    serial = [(0.0, (j + 1) * H, N) for j in range(4)]
    lim, hist = SUB.growth_limit(ramp, serial, serial[-3:], now=10 * H)
    rate = N / H
    assert lim / rate <= ramp["target_share"] * ramp["window_hours"] * H + 1      # the whole limit drains within 14.4 h at that rate
    assert lim > 4 * N                                                          # and it still grows from 40k
    # parallel: the same four, each done in 2 h side by side -> x4 (capped)
    par = [(0.0, 2 * H, N) for _ in range(4)]
    lim, _ = SUB.growth_limit(ramp, par, par[-3:], now=10 * H)
    assert lim == 4 * 4 * N
    # slow: a batch that took 20 h of the 24 -> the limit shrinks below what was in flight
    slow = [(0.0, 20 * H, N), (0.0, 20 * H, N)]
    lim, _ = SUB.growth_limit(ramp, slow, slow, now=21 * H)
    assert lim < 2 * N
