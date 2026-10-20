"""BatchJudge (scripts/judge_batch.py) against a fully mocked client.

Plan 7 test row: custom_id <-> cache-key round trip (keys include model +
template version), schema-as-tool encoding in the submitted
MessageCreateParams + tool_use-block parsing in harvest,
chunking at the request-count AND byte caps (unified merge defaults:
25k requests — the shared-Batch-pool headroom — and 48MB — the measured
413 ceiling; the merge took the tighter bound on each axis) with
wave-bounded in-flight batches + 429 backoff on create (the Batch org's
in-flight pool is shared with other campaigns),
cache short-circuit, and NO cross-model seeding -- sonnet-5 keys must MISS
for a haiku judge.
"""

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import anthropic

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import judge_batch  # noqa: E402
import judge_sonnet  # noqa: E402
from judge_batch import BatchJudge, build_request, chunk_requests  # noqa: E402
from judge_sonnet import HyperSteerScore, QualityQuestions  # noqa: E402

HAIKU = "claude-haiku-4-5"


# ---------------------------------------------------------------- mocks


class FakeBatches:
    def __init__(self):
        self.created: list[tuple[str, list[dict]]] = []
        self.results_by_id: dict[str, list] = {}
        self.retrieve_calls: list[str] = []

    def create(self, requests):
        bid = f"msgbatch_{len(self.created)}"
        self.created.append((bid, list(requests)))
        return SimpleNamespace(id=bid, processing_status="in_progress")

    def retrieve(self, bid):
        self.retrieve_calls.append(bid)
        return SimpleNamespace(
            id=bid, processing_status="ended",
            request_counts=SimpleNamespace(processing=0),
        )

    def results(self, bid):
        return iter(self.results_by_id.get(bid, []))


class FakeClient:
    def __init__(self):
        self.messages = SimpleNamespace(batches=FakeBatches())


def make_judge(tmp_path, **kw):
    client = FakeClient()
    judge = BatchJudge(tmp_path / "judge_cache.jsonl", model=HAIKU,
                       client=client, **kw)
    return judge, client.messages.batches


def succeeded(custom_id, tool_name, tool_input, stop_reason="tool_use"):
    return SimpleNamespace(
        custom_id=custom_id,
        result=SimpleNamespace(
            type="succeeded",
            message=SimpleNamespace(
                stop_reason=stop_reason,
                content=[SimpleNamespace(type="tool_use", name=tool_name,
                                         input=tool_input)],
            ),
        ),
    )


HS_INPUT = {"justification": "ok", "follows_base_prompt": 2,
            "follows_steering_prompt": 1, "fluency": 2}


# ---------------------------------------------------------------- keys


def test_key_scheme_matches_judge_sonnet_exactly():
    """One scheme, one implementation: sha256 over \\x00-terminated parts."""
    parts = ("hs.v1", HAIKU, "steer text", "a query", "a response")
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode())
        h.update(b"\x00")
    assert judge_batch.cache_key(*parts) == h.hexdigest()
    assert judge_batch.cache_key(*parts) == judge_sonnet.Judge._key(None, *parts)


def test_keys_include_model_and_template_version(tmp_path):
    judge, _ = make_judge(tmp_path)
    key = judge.collect_hypersteer("steer", "query", "resp")
    # model is a key component
    assert key != judge.key("hs.v1", "claude-sonnet-5", "steer", "query", "resp")
    # template version tag is a key component
    assert key != judge.key("hs.v2", HAIKU, "steer", "query", "resp")
    # and the key matches the documented construction
    assert key == judge.key("hs.v1", HAIKU, "steer", "query", "resp")


def test_sonnet_keys_miss_no_cross_model_seeding(tmp_path):
    """A sonnet-5-keyed cache (stage-1/1.5) can never seed a haiku run."""
    sonnet_key = judge_batch.cache_key(
        "hs.v1", judge_sonnet.JUDGE_MODEL, "steer", "query", "resp")
    cache = tmp_path / "judge_cache.jsonl"
    cache.write_text(json.dumps(
        {"key": sonnet_key, "value": {"justification": "x",
                                      "follows_base_prompt": 2,
                                      "follows_steering_prompt": 2,
                                      "fluency": 2}}) + "\n")
    judge, _ = make_judge(tmp_path)
    assert judge_sonnet.JUDGE_MODEL == "claude-sonnet-5"  # the seed's model
    haiku_key = judge.collect_hypersteer("steer", "query", "resp")
    assert haiku_key != sonnet_key
    # the sonnet entry did NOT short-circuit: the haiku request is queued
    assert haiku_key in judge.pending
    assert len(judge.pending) == 1
    # and reading the score back for haiku finds nothing yet
    assert judge.hypersteer("steer", "query", "resp") is None


def test_cache_short_circuit(tmp_path):
    judge, batches = make_judge(tmp_path)
    key = judge.key("hs.v1", HAIKU, "s", "q", "r")
    judge._write_cache(key, HS_INPUT)
    assert judge.collect_hypersteer("s", "q", "r") == key
    assert judge.pending == {}          # cached -> nothing queued
    assert judge.submit() == []         # nothing to submit
    assert batches.created == []
    # duplicate collects of the same uncached item queue exactly once
    judge.collect_hypersteer("s2", "q2", "r2")
    judge.collect_hypersteer("s2", "q2", "r2")
    assert len(judge.pending) == 1


# ---------------------------------------------------------------- tools


def test_schema_as_tool_encoding_in_submitted_params(tmp_path):
    judge, batches = make_judge(tmp_path)
    key = judge.collect_hypersteer("steer text", "the query", "the response")
    [bid] = judge.submit()
    (bid_created, requests), = batches.created
    assert bid == bid_created
    [req] = requests
    # custom_id IS the cache key
    assert req["custom_id"] == key
    params = req["params"]
    assert params["model"] == HAIKU
    assert params["messages"] == [{
        "role": "user",
        "content": judge_sonnet.HYPERSTEER_SCORE.format(
            steering="steer text", query="the query", response="the response"),
    }]
    # pydantic schema encoded as a TOOL, choice forced (review fix 4)
    [tool] = params["tools"]
    assert tool["name"] == "HyperSteerScore"
    assert tool["input_schema"] == HyperSteerScore.model_json_schema()
    assert params["tool_choice"] == {"type": "tool", "name": "HyperSteerScore"}
    # batch ids persisted for resume
    state = json.loads(judge.batches_path.read_text())
    assert state["batches"] == [{"id": bid, "harvested": False}]
    assert judge.pending == {}


def test_genq_request_uses_quality_questions_schema(tmp_path):
    judge, batches = make_judge(tmp_path)
    judge.collect_questions("some query")
    judge.submit()
    [req] = batches.created[0][1]
    assert req["params"]["tool_choice"]["name"] == "QualityQuestions"
    assert req["params"]["tools"][0]["input_schema"] == \
        QualityQuestions.model_json_schema()


# ---------------------------------------------------------------- chunking


def test_chunking_defaults_25k_requests_48mb_bytes():
    """The chunker caps at BOTH the request count and the byte budget, and the
    merge deliberately kept the TIGHTER bound on each axis because the two
    lanes hit a real 413 from opposite directions:
      - requests 25k (capeval/petri, 2026-08-14): smaller batches are smaller
        in-flight steps against the SHARED Batch org pool.
      - bytes 48MB (attackgen/jailbreak, 2026-08-15): a ~200MB /
        40460-request json.dumps estimate was rejected with 'request body
        exceeds max-request-body-bytes', so the effective limit is far under
        the documented 256MB and the json.dumps estimate undercounts.
    Pins both defaults, and proves the split fires on EACH cap."""
    assert judge_batch.MAX_REQUESTS_PER_BATCH == 25_000
    assert judge_batch.MAX_BYTES_PER_BATCH == 48 * 1024 * 1024
    reqs = [{"custom_id": str(i), "params": {}} for i in range(7)]
    # request-count limit (byte cap slack so ONLY the count binds)
    chunks = chunk_requests(reqs, max_requests=3, max_bytes=10**9)
    assert [len(c) for c in chunks] == [3, 3, 1]
    assert [r for c in chunks for r in c] == reqs  # order + completeness
    # a single default-limit chunk stays whole
    assert len(chunk_requests(reqs)) == 1
    # BYTE cap splits BELOW the request-count limit: 4 fat requests, a byte
    # budget that fits only 2, and a count limit (1000) that never binds —
    # so the split here can ONLY come from the byte cap, not the count cap.
    fat = [{"custom_id": str(i), "params": {"messages": "x" * 500}}
           for i in range(4)]
    per = len(json.dumps(fat[0]).encode())
    chunks = chunk_requests(fat, max_requests=1000, max_bytes=2 * per + 1)
    assert [len(c) for c in chunks] == [2, 2]
    assert [r for c in chunks for r in c] == fat  # order + completeness


def test_chunking_respects_byte_cap():
    big = [{"custom_id": str(i), "params": {"messages": "x" * 100}}
           for i in range(4)]
    per = len(json.dumps(big[0]).encode())
    chunks = chunk_requests(big, max_requests=100, max_bytes=2 * per + 1)
    assert [len(c) for c in chunks] == [2, 2]
    # an oversized single request still ships (alone) rather than looping
    chunks = chunk_requests(big[:1], max_requests=100, max_bytes=1)
    assert [len(c) for c in chunks] == [1]


def test_submit_splits_into_multiple_batches(tmp_path):
    judge, batches = make_judge(tmp_path, max_requests_per_batch=2)
    for i in range(5):
        judge.collect_hypersteer("s", f"q{i}", "r")
    ids = judge.submit()
    assert len(ids) == 3
    assert [len(reqs) for _bid, reqs in batches.created] == [2, 2, 1]
    assert len(judge.unharvested_batches()) == 3


def _raise_413():
    import httpx
    return anthropic.RequestTooLargeError(
        "request body exceeds max-request-body-bytes",
        response=httpx.Response(
            413, request=httpx.Request(
                "POST", "https://api.anthropic.com/v1/messages/batches")),
        body=None,
    )


def test_submit_selfheals_on_413_by_splitting(tmp_path):
    """A chunk whose real body exceeds the API cap (413) despite the byte
    estimate is split in half and retried until every sub-batch fits — the
    exact cap never has to be guessed (2026-08-14 judge fix)."""
    judge, batches = make_judge(tmp_path)  # default caps: one chunk of 5
    for i in range(5):
        judge.collect_hypersteer("s", f"q{i}", "r")
    original_create = batches.create

    def too_large(requests):
        if len(requests) > 2:                       # anything over 2 -> 413
            raise _raise_413()
        return original_create(requests)

    batches.create = too_large
    ids = judge.submit()
    assert all(len(reqs) <= 2 for _bid, reqs in batches.created)
    assert sum(len(reqs) for _bid, reqs in batches.created) == 5
    assert len(ids) == len(batches.created)         # every sub-batch created
    state = json.loads(judge.batches_path.read_text())
    assert [b["id"] for b in state["batches"]] == ids   # all recorded
    assert state.get("creating", []) == []              # no false orphan left


def test_submit_reraises_413_on_single_request(tmp_path):
    """A single request over the limit is unfixable -> fail loudly, never
    loop forever splitting."""
    judge, batches = make_judge(tmp_path)
    judge.collect_hypersteer("s", "q", "r")
    batches.create = lambda requests: (_ for _ in ()).throw(_raise_413())
    with pytest.raises(anthropic.RequestTooLargeError):
        judge.submit()


# ------------------------------------------------ 429 backoff + waves
# The Batch org's in-flight-request limit is SHARED with other campaigns:
# a 429 on batches.create backs off and retries (the pool drains as
# in-flight batches finish), and run_pending submits in waves of
# max_inflight_batches, harvesting each wave before the next.


def _raise_429():
    import httpx
    return anthropic.RateLimitError(
        "Number of Message Batch requests in the middle of processing has "
        "exceeded your available limit",
        response=httpx.Response(
            429, request=httpx.Request(
                "POST", "https://api.anthropic.com/v1/messages/batches")),
        body=None,
    )


def test_create_backs_off_on_429_then_succeeds(tmp_path, monkeypatch):
    """A 429 on create sleeps and retries with exponential backoff until
    the shared in-flight pool drains — the batch is eventually created
    with NO work lost. The pre-create journal entry REMAINS across the
    retries (we are still trying to create this exact chunk) and clears
    on success."""
    sleeps: list[float] = []
    monkeypatch.setattr(judge_batch.time, "sleep",
                        lambda s: sleeps.append(s))
    judge, batches = make_judge(tmp_path)
    key = judge.collect_hypersteer("s", "q", "r")
    original_create = batches.create
    fails = {"n": 0}

    def rate_limited_thrice(requests):
        if fails["n"] < 3:
            fails["n"] += 1
            # the journal entry for this chunk survives every 429 retry
            assert len(judge._batch_state().get("creating", [])) == 1
            raise _raise_429()
        return original_create(requests)

    batches.create = rate_limited_thrice
    [bid] = judge.submit()
    assert sleeps == [30, 60, 120]                  # exponential schedule
    # the full chunk was created — no work lost, id recorded, journal clear
    [(created_bid, reqs)] = batches.created
    assert created_bid == bid
    assert [r["custom_id"] for r in reqs] == [key]
    state = json.loads(judge.batches_path.read_text())
    assert [b["id"] for b in state["batches"]] == [bid]
    assert state.get("creating", []) == []
    assert judge.pending == {}


def test_exhausted_429_backoff_reraises(tmp_path, monkeypatch):
    """A 429 on every attempt exhausts the backoff and re-raises loudly —
    work is never silently dropped, and the sleeps follow the capped
    exponential schedule (30s doubling, ceiling 300s)."""
    sleeps: list[float] = []
    monkeypatch.setattr(judge_batch.time, "sleep",
                        lambda s: sleeps.append(s))
    judge, batches = make_judge(tmp_path)
    judge.collect_hypersteer("s", "q", "r")
    batches.create = lambda requests: (_ for _ in ()).throw(_raise_429())
    with pytest.raises(anthropic.RateLimitError):
        judge.submit()
    assert len(sleeps) == judge_batch.BATCH_CREATE_429_MAX_ATTEMPTS - 1
    assert sleeps[:4] == [30, 60, 120, 240]
    assert set(sleeps[4:]) == {judge_batch.BATCH_CREATE_429_MAX_SLEEP_S}


def test_413_still_splits_while_429_backs_off(tmp_path, monkeypatch):
    """The two create-time failure modes stay independent: a 413 splits
    the chunk (journal entry dropped, halves journal themselves) while a
    429 on a half backs off and retries THAT half."""
    monkeypatch.setattr(judge_batch.time, "sleep", lambda s: None)
    judge, batches = make_judge(tmp_path)
    keys = [judge.collect_hypersteer("s", f"q{i}", "r") for i in range(4)]
    original_create = batches.create
    rate_limited_once = {"done": False}

    def too_large_then_rate_limited(requests):
        if len(requests) > 2:                       # whole chunk -> 413
            raise _raise_413()
        if not rate_limited_once["done"]:           # first half -> one 429
            rate_limited_once["done"] = True
            raise _raise_429()
        return original_create(requests)

    batches.create = too_large_then_rate_limited
    ids = judge.submit()
    assert len(ids) == 2                            # both halves created
    assert sorted(len(reqs) for _b, reqs in batches.created) == [2, 2]
    assert {r["custom_id"] for _b, reqs in batches.created
            for r in reqs} == set(keys)
    state = json.loads(judge.batches_path.read_text())
    assert state.get("creating", []) == []          # no journal leftovers


def test_submit_creates_at_most_one_wave(tmp_path):
    """submit() bounds itself to max_inflight_batches chunks per call; the
    remainder stays pending, in order, for the next wave — chunking is
    deterministic across waves."""
    judge, batches = make_judge(tmp_path, max_requests_per_batch=1,
                                max_inflight_batches=2)
    keys = [judge.collect_hypersteer("s", f"q{i}", "r") for i in range(5)]
    assert len(judge.submit()) == 2
    assert list(judge.pending) == keys[2:]          # only the wave left
    assert len(judge.submit()) == 2
    assert len(judge.submit()) == 1
    assert judge.pending == {}
    submitted = [r["custom_id"] for _b, reqs in batches.created for r in reqs]
    assert submitted == keys                        # same order, no loss


class AutoHSBatches(FakeBatches):
    """create() files a succeeded HyperSteerScore result for every request,
    so run_pending's wave loop harvests each wave to completion."""

    def create(self, requests):
        ns = super().create(requests)
        self.results_by_id[ns.id] = [
            succeeded(r["custom_id"], "HyperSteerScore", HS_INPUT)
            for r in requests
        ]
        return ns


def test_wave_submission_bounds_inflight_batches(tmp_path, capsys):
    """With max_inflight_batches=2 and enough requests for 5 batches, no
    more than 2 batches are ever in flight (created-but-unharvested):
    run_pending submits a wave, polls+harvests it to completion, then
    submits the next — until all 5 are created AND harvested. The
    coordination-courtesy line is emitted once, with the run totals."""
    client = FakeClient()
    client.messages.batches = AutoHSBatches()
    batches = client.messages.batches
    judge = BatchJudge(tmp_path / "judge_cache.jsonl", model=HAIKU,
                       client=client,
                       max_requests_per_batch=1, max_inflight_batches=2)
    keys = [judge.collect_hypersteer("s", f"q{i}", "r") for i in range(5)]

    inflight_at_create: list[int] = []
    original_create = batches.create

    def counting_create(requests):
        # batches already created-but-unharvested + the one being created
        inflight_at_create.append(len(judge.unharvested_batches()) + 1)
        return original_create(requests)

    batches.create = counting_create
    judge.run_pending(interval_s=0)

    assert len(batches.created) == 5                # all 5 ultimately created
    assert max(inflight_at_create) <= 2             # never more than a wave
    assert inflight_at_create == [1, 2, 1, 2, 1]    # waves of 2 + 2 + 1
    assert judge.unharvested_batches() == []        # every wave harvested
    assert judge.pending == {}
    for k in keys:
        assert judge.cache[k] == HS_INPUT           # no work lost
    text = capsys.readouterr().out
    assert text.count("[coordination courtesy]") == 1
    assert "~5 claude-haiku-4-5 requests in 5 batch(es)" in text


# ---------------------------------------------------------------- harvest


def test_harvest_roundtrip_by_custom_id(tmp_path):
    judge, batches = make_judge(tmp_path)
    key = judge.collect_hypersteer("steer", "query", "resp")
    [bid] = judge.submit()
    batches.results_by_id[bid] = [succeeded(key, "HyperSteerScore", HS_INPUT)]
    judge.poll([bid], interval_s=0)
    counts = judge.harvest([bid])
    assert counts["succeeded"] == 1 and counts["blocked"] == 0
    # keyed back into the cache by custom_id, validated through the schema
    assert judge.cache[key] == HS_INPUT
    assert judge.hypersteer("steer", "query", "resp") == {
        "base": 2, "steer": 1, "fluency": 2,
        "harmonic": 3.0 / (1 / 2 + 1 / 1 + 1 / 2),
    }
    # written in judge_sonnet's jsonl format; a fresh judge re-reads it
    judge2, _ = make_judge(tmp_path)
    assert judge2.cache[key] == HS_INPUT
    assert judge2.collect_hypersteer("steer", "query", "resp") == key
    assert judge2.pending == {}
    # sidecar marks the batch harvested (resume never re-polls it)
    assert judge.unharvested_batches() == []


def test_harvest_failure_modes_record_blocked(tmp_path):
    judge, batches = make_judge(tmp_path)
    keys = [judge.collect_hypersteer("s", f"q{i}", "r") for i in range(5)]
    [bid] = judge.submit()
    bad_input = dict(HS_INPUT, follows_base_prompt=7)  # not in Literal[0,1,2]
    batches.results_by_id[bid] = [
        SimpleNamespace(custom_id=keys[0],
                        result=SimpleNamespace(type="errored")),
        SimpleNamespace(custom_id=keys[1],
                        result=SimpleNamespace(type="expired")),
        succeeded(keys[2], "HyperSteerScore", bad_input),
        succeeded(keys[3], "HyperSteerScore", HS_INPUT, stop_reason="refusal"),
        succeeded(keys[4], "SomeUnknownTool", {}),
    ]
    counts = judge.harvest([bid])
    assert counts["blocked"] == 5 and counts["succeeded"] == 0
    for k in keys:
        assert judge.cache[k].get("__blocked__") is True
    # blocked entries are per-item outcomes: score readers return None
    assert judge.hypersteer("s", "q2", "r") is None


def test_harvest_skips_already_cached(tmp_path):
    judge, batches = make_judge(tmp_path)
    key = judge.collect_hypersteer("s", "q", "r")
    [bid] = judge.submit()
    judge._write_cache(key, HS_INPUT)
    batches.results_by_id[bid] = [
        succeeded(key, "HyperSteerScore", dict(HS_INPUT, fluency=0))]
    counts = judge.harvest([bid])
    assert counts["skipped_cached"] == 1
    assert judge.cache[key] == HS_INPUT  # first write wins


def test_batch_ids_persisted_immediately_per_create(tmp_path):
    """Each batch id hits the sidecar the moment create returns
    — dying between creates must not orphan the earlier batches."""
    judge, batches = make_judge(tmp_path, max_requests_per_batch=1)
    judge.collect_hypersteer("s", "q1", "r")
    judge.collect_hypersteer("s", "q2", "r")

    recorded_at_create: list[list[str]] = []
    original_create = batches.create

    def tracking_create(requests):
        result = original_create(requests)
        return result

    batches.create = tracking_create

    original_record = judge._record_batches

    def tracking_record(ids, **kw):
        original_record(ids, **kw)
        recorded_at_create.append(judge.unharvested_batches())

    judge._record_batches = tracking_record
    judge.submit()
    # after the FIRST create, the first id was already on disk
    assert recorded_at_create[0] == ["msgbatch_0"]
    assert recorded_at_create[1] == ["msgbatch_0", "msgbatch_1"]


# ---------------------------------------------------------------- stolfo


def test_stolfo_wave_depends_on_harvested_questions(tmp_path):
    judge, batches = make_judge(tmp_path)
    # before genq is harvested, answering requests cannot be assembled
    assert judge.collect_stolfo("query", "resp", rep=0) is None
    genq_key = judge.collect_questions("query")
    [bid] = judge.submit()
    questions = [f"Q{i}?" for i in range(5)]
    batches.results_by_id[bid] = [
        succeeded(genq_key, "QualityQuestions", {"questions": questions})]
    judge.harvest([bid])
    assert judge.questions_for("query") == questions
    # wave 2: answering key embeds the question text + repeat index
    key0 = judge.collect_stolfo("query", "resp", rep=0)
    key1 = judge.collect_stolfo("query", "resp", rep=1)
    qtext = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))
    assert key0 == judge.key("stolfo.v1", HAIKU, "query", "resp", qtext, "0")
    assert key0 != key1
    [bid2] = judge.submit()
    answers = {"answers": [{"rationale": "r", "answer": a}
                           for a in ("yes", "yes", "no", "na", "yes")]}
    batches.results_by_id[bid2] = [
        succeeded(key0, "QualityAnswers", answers),
        succeeded(key1, "QualityAnswers", answers),
    ]
    judge.harvest([bid2])
    assert judge.stolfo_quality("query", "resp", 0) == pytest.approx(3 / 4)


# ----------------------------------------------------------------------- transactionality


def test_truncated_final_cache_line_loads_and_repairs(tmp_path):
    """A SIGKILL mid-append leaves a torn final line: loading must keep
    every complete entry, drop the torn tail, and REPAIR the file so the
    next append starts on a clean line instead of gluing onto the tear."""
    cache = tmp_path / "judge_cache.jsonl"
    good = json.dumps({"key": "k1", "value": HS_INPUT}) + "\n"
    cache.write_text(good + '{"key": "k2", "va')  # torn, no newline
    judge, _ = make_judge(tmp_path)
    assert judge.cache == {"k1": HS_INPUT}
    assert cache.read_text() == good               # repaired: tail truncated
    judge._write_cache("k3", HS_INPUT)             # append lands cleanly
    judge2, _ = make_judge(tmp_path)
    assert set(judge2.cache) == {"k1", "k3"}


def test_torn_terminated_final_line_also_repaired(tmp_path):
    cache = tmp_path / "judge_cache.jsonl"
    good = json.dumps({"key": "k1", "value": HS_INPUT}) + "\n"
    cache.write_text(good + '{"half a li\n')       # torn but newline-terminated
    judge, _ = make_judge(tmp_path)
    assert set(judge.cache) == {"k1"}
    assert cache.read_text() == good


def test_corrupt_non_final_cache_line_raises(tmp_path):
    """Mid-file corruption is NOT the torn-tail case: silently dropping
    those entries would re-spend their judgments — fail loudly instead."""
    cache = tmp_path / "judge_cache.jsonl"
    cache.write_text('{"broken\n'
                     + json.dumps({"key": "k1", "value": HS_INPUT}) + "\n")
    with pytest.raises(ValueError, match="non-final"):
        make_judge(tmp_path)


def test_batches_state_atomic_with_precreate_journal(tmp_path):
    """batches.json is published atomically (temp+fsync+replace) and every
    create is journaled BEFORE the API call: dying inside batches.create
    leaves an auditable 'creating' entry instead of a silent orphan."""
    judge, batches = make_judge(tmp_path, max_requests_per_batch=1)
    judge.collect_hypersteer("s", "q1", "r")
    judge.collect_hypersteer("s", "q2", "r")
    original_create = batches.create

    def dies_on_second_create(requests):
        if batches.created:
            raise RuntimeError("network death mid-create")
        return original_create(requests)

    batches.create = dies_on_second_create
    with pytest.raises(RuntimeError, match="mid-create"):
        judge.submit()
    # state is intact JSON (never a torn file), no temp leftover
    state = json.loads(judge.batches_path.read_text())
    assert not judge.batches_path.with_name(
        judge.batches_path.name + ".tmp").exists()
    # the first batch is recorded; the second's pre-create journal remains
    assert [b["id"] for b in state["batches"]] == ["msgbatch_0"]
    assert len(state["creating"]) == 1
    assert state["creating"][0]["n_requests"] == 1

    # a later resume surfaces the possible orphan LOUDLY, parks it under
    # possible_orphans (warn once), and still harvests the recorded batch
    judge2, _ = make_judge(tmp_path)
    judge2.resume(interval_s=0)
    state2 = json.loads(judge2.batches_path.read_text())
    assert state2["creating"] == []
    assert len(state2["possible_orphans"]) == 1
    assert judge2.unharvested_batches() == []      # msgbatch_0 harvested


def test_concurrent_collect_dedup_two_judges_one_cache(tmp_path):
    """Two judge processes on one cache namespace can never
    double-submit the same uncached keys — the second's locked round
    re-reads the cache and drops every now-answered pending key."""
    judge_a, batches_a = make_judge(tmp_path)
    judge_b, batches_b = make_judge(tmp_path)
    key = judge_a.collect_hypersteer("s", "q", "r")
    assert judge_b.collect_hypersteer("s", "q", "r") == key
    assert key in judge_b.pending                  # B queued it too

    batches_a.results_by_id["msgbatch_0"] = [
        succeeded(key, "HyperSteerScore", HS_INPUT)]
    judge_a.run_pending(interval_s=0)              # A's full locked round
    assert judge_a.cache[key] == HS_INPUT

    judge_b.run_pending(interval_s=0)              # B refreshes + dedupes
    assert batches_b.created == []                 # nothing double-submitted
    assert judge_b.pending == {}
    assert judge_b.cache[key] == HS_INPUT          # B sees A's result


def test_lock_held_across_the_submit_round(tmp_path):
    """The namespace flock is actually HELD while the round runs: a
    would-be concurrent locker (probed non-blocking from inside the fake
    create call) finds it taken."""
    import fcntl

    judge, batches = make_judge(tmp_path)
    key = judge.collect_hypersteer("s", "q", "r")
    observed = {}
    original_create = batches.create

    def probing_create(requests):
        with judge.lock_path.open("a") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(f, fcntl.LOCK_UN)
                observed["held"] = False
            except BlockingIOError:
                observed["held"] = True
        return original_create(requests)

    batches.create = probing_create
    batches.results_by_id["msgbatch_0"] = [
        succeeded(key, "HyperSteerScore", HS_INPUT)]
    judge.run_pending(interval_s=0)
    assert observed["held"] is True
