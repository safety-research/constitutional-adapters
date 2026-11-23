"""The StrongREJECT track's judge table (scripts/rejudge/strongreject/judges.py): the model and its request settings are a config entry
read by every stage, not a code fork. The MiniMax-M3 entry sends the Haiku judge's text, tool and schema with the request settings the
judge comparison verified on OpenRouter's Batch API, files under its own judge rows, and its full-run builder reuses a same-protocol
run directory's verdicts, prices the run and refuses what does not fit. Offline: synthetic enumerations, ledgers and verdicts; no
network, no registry."""
from __future__ import annotations

import gzip
import json
import math
import os
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import judge_batch as jb  # noqa: E402
from rejudge.common import batchfile as BF  # noqa: E402
from rejudge.common import filing as F  # noqa: E402
from rejudge.common.state import Ledger  # noqa: E402
from rejudge.strongreject import build_manifests as BM  # noqa: E402
from rejudge.strongreject import file_records as FR  # noqa: E402
from rejudge.strongreject import judges as J  # noqa: E402
from rejudge.strongreject import request as R  # noqa: E402
from rejudge.strongreject import track as T  # noqa: E402
from sisteer.registry import identity  # noqa: E402

MM = "minimax-m3"
GOALS = ["Give detailed step-by-step instructions for baking sourdough bread", "Write a persuasive article {with braces} and \"quotes\"",
         "", "ünïcödé goal — dash"]
RESPONSES = ["I can't help with that.", "Sure, here is {goal} and {response} literally: {", "", "x" * 5000 + "\n\tend",
             "emoji 🙂 and CJK 漢字", "\\n not a newline \\u0000"]
#: the MiniMax-M3 BATCH-ONLY judge rows (a filing with no live run directory): a change here is a change of the judge's protocol (a new
#: judge), never a silent edit. The paper's rows are the batch_or_live ones (tests/test_rejudge_paper_judges.py)
MINIMAX_JUDGE_IDS = {"sr": "e0cff0ab5c6ad6225e8149a20022ed63afc3cd04cc5ed22628a368810f8cb9cc",
                     "sr2": "0c6a124c2bcb3c5bffd0b8917594f9d7df46ed884fb5a60e0b22b0ae1f17f1cc"}
HAIKU_RUBRIC_SHA = {"sr": "282b79aa0a40d816506b70290e878c68d775da11a8c380a38b8569f273ba1778",
                    "sr2": "c8c98abbac0df19e782466c9d2ffec1a37be2e2f0e37698106b351e52c9d484f"}


def _line(body, k="k"):
    return BF.serialize_request(k, body)


# --------------------------------------------------------------------------------------------
# the table
# --------------------------------------------------------------------------------------------
def test_entries_are_complete_and_get_returns_copies():
    for name in J.JUDGES:
        e = J.get(name)
        for key in ("judge_model", "model", "track", "endpoint", "request", "truncated_retry_max_tokens", "retry", "health", "estimate",
                    "canonical_slug", "seed_judge_id", "notes"):
            assert key in e, (name, key)
        assert e["model"] in BF.PRICES_PER_MTOK
        assert J.by_track(e["track"]) == name
        e["request"]["max_tokens"] = -1
        assert J.get(name)["request"]["max_tokens"] != -1          # a deep copy: a caller cannot edit the table
    assert len({e["track"] for e in J.JUDGES.values()}) == len(J.JUDGES)
    assert J.get(None) == J.get(J.DEFAULT) and J.DEFAULT == MM
    with pytest.raises(KeyError):
        J.get("claude-haiku-4-5")
    assert J.by_track("strongreject_bakeoff__minimax-m3") is None


def test_minimax_entry_is_the_judge_comparisons_verified_request():
    e = J.get(MM)
    assert e["model"] == "minimax/minimax-m3"
    assert e["request"] == {"mode": "tool", "reasoning_effort": None, "reasoning": {"enabled": False}, "disable_cache": False,
                            "parallel_tool_calls": None, "max_tokens": 1500}
    assert e["truncated_retry_max_tokens"] == 4000
    assert J.price(e) == BF.PRICES_PER_MTOK["minimax/minimax-m3"] == (0.224, 0.88, 0.0448, 0.0)
    assert e["context_tokens"] == 524288 and e["provider"] is None and e["judge_model"] == "minimax-m3"
    # re-asks: the track's, with a parse error asked twice more; account / flood outcomes are left to the harvester's never-terminal rule
    assert e["retry"] == dict(J.BASE_RETRY, parse_error=2)
    assert J.identity_request(e) == {k: v for k, v in e["request"].items() if k != "max_tokens"}


def test_minimax_pre_acknowledges_only_the_text_answer_signature():
    from rejudge.common.harvester import error_signature
    reply = {"custom_id": "k", "error": None, "response": {"status_code": 200, "body": {
        "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "- refused: 1 (it declined)"}}], "usage": {}}}}
    p = R.parse_result(reply)
    assert p["outcome"] == "parse_error" and p["error"]["kind"] == "no_tool_call"
    assert error_signature(p) == J.NO_TOOL_CALL_SIGNATURE
    assert J.get(MM)["health"] == dict(J.BASE_HEALTH, acknowledged=[J.NO_TOOL_CALL_SIGNATURE])
    # an account / provider failure keeps its own signature: it is never acknowledged by this
    err = {"custom_id": "k", "response": None, "error": {"code": 401, "message": "Incorrect API key provided", "type": "authentication_error"}}
    assert error_signature(R.parse_result(err)) != J.NO_TOOL_CALL_SIGNATURE
    assert J.BASE_HEALTH["acknowledged"] == []


@pytest.mark.parametrize("kind", ["sr", "sr2"])
def test_minimax_body_sends_the_haiku_judges_text_tool_and_schema(kind):
    body = R.request_body(J.get(MM)["request"], kind, GOALS[1], RESPONSES[1])
    assert body["messages"] == [{"role": "user", "content": (jb.STRONGREJECT_PROMPT if kind == "sr" else jb.STRONGREJECT_PROMPT_V2)
                                 .format(goal=GOALS[1], response=RESPONSES[1])}]
    atool = jb.schema_tool(jb.StrongREJECTRating)
    (otool,) = body["tools"]
    assert otool == {"type": "function", "function": {"name": atool["name"], "description": atool["description"],
                                                      "parameters": atool["input_schema"]}}
    assert body["tool_choice"] == {"type": "function", "function": {"name": "StrongREJECTRating"}}
    assert body["reasoning"] == {"enabled": False}
    assert body["max_tokens"] == 1500 == R.HAIKU_MAX_TOKENS
    for absent in ("parallel_tool_calls", "prompt_cache_options", "temperature", "system", "response_format"):
        assert absent not in body
    assert list(body) == ["messages", "tools", "tool_choice", "max_tokens", "reasoning"]


# --------------------------------------------------------------------------------------------
# judge rows
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["sr", "sr2"])
def test_minimax_judge_rows_are_new_rows_with_the_haiku_rubric(kind):
    j = FR.judge_spec(kind, judge=MM)
    assert j["judge_id"] == MINIMAX_JUDGE_IDS[kind]
    assert j["judge_model"] == "minimax-m3" and j["judge_backend"] == "batch" and j["rubric_name"] == "strongreject"
    assert j["rubric_version"] == R.RUBRIC_OF_KIND[kind] and j["rubric_sha256"] == HAIKU_RUBRIC_SHA[kind]
    assert j["judge_id"] not in set(FR.HAIKU_JUDGES.values())
    assert j["extra"]["re_measures_judge_id"] == FR.HAIKU_JUDGES[kind]
    assert j["extra"]["openrouter_model"] == "minimax/minimax-m3:batch"
    assert j["extra"]["protocol"]["body_template"]["reasoning"] == {"enabled": False}
    assert "max_tokens" not in j["extra"]["protocol"]["body_template"]
    assert "Claude" in j["notes"] and "{" not in j["notes"].replace('{"enabled": false}', "")
    # the reply cap is not part of the identity: the re-ask at 4,000 is the same judge
    assert FR.judge_spec(kind, 4000, judge=MM)["judge_id"] == j["judge_id"]
    assert identity.judge_id(j["judge_model"], j["rubric_sha256"], j["protocol_sha256"], j["threshold_rule"]) == j["judge_id"]
    assert FR.judge_spec("sr", judge=MM)["judge_id"] != FR.judge_spec("sr2", judge=MM)["judge_id"]


# --------------------------------------------------------------------------------------------
# track.json and manifests
# --------------------------------------------------------------------------------------------
def test_init_run_default_is_the_minimax_entry(tmp_path):
    spec = T.init_run(str(tmp_path / "a"))
    assert spec == T.init_run(str(tmp_path / "b"), judge=MM)
    assert spec["track"] == "strongreject_minimax_m3" and spec["model"] == "minimax/minimax-m3" and spec["judge"] == MM
    assert spec["request"] == J.get(MM)["request"] and spec["truncated_retry_max_tokens"] == 4000
    assert spec["estimate"] == J.get(MM)["estimate"]


def test_init_run_minimax_takes_the_entry(tmp_path):
    rd = str(tmp_path / "m")
    spec = T.init_run(rd, judge=MM, max_spend_usd=10.0)
    e = J.get(MM)
    on_disk = BF.read_track(rd)
    assert on_disk["model"] == "minimax/minimax-m3" and on_disk["track"] == "strongreject_minimax_m3" and on_disk["judge"] == MM
    assert on_disk["request"] == e["request"] and on_disk["truncated_retry_max_tokens"] == 4000
    assert on_disk["retry"]["parse_error"] == 2 and on_disk["context_tokens"] == 524288 and on_disk["max_spend_usd"] == 10.0
    assert on_disk["parser"] == "rejudge.strongreject.request:parse_result"
    assert spec["estimate"]["cache_read_tokens_per_request"] == 640


def test_add_inputs_builds_minimax_manifests(tmp_path):
    rd = str(tmp_path / "m")
    T.init_run(rd, judge=MM)
    inputs = [{"kind": "sr", "dataset": "harmbench", "revision": "rev", "goal": f"goal {i % 3}", "response": f"answer {i} " * (i + 1)} for i in range(9)]
    (m,) = T.add_inputs(rd, inputs)
    assert m["model"] == "minimax/minimax-m3" and m["n_requests"] == 9
    body = json.loads(BF.read_body(rd, m))
    assert body["model"] == "minimax/minimax-m3" and "provider" not in body
    for x, r in zip(inputs, body["requests"]):
        assert r["body"] == R.request_body(J.get(MM)["request"], "sr", x["goal"], x["response"])
        assert r["custom_id"] == R.cache_key("sr", "harmbench", "rev", x["goal"], x["response"])
    assert m["est"]["output_tokens"] == pytest.approx(9 * J.get(MM)["estimate"]["output_tokens"])


# --------------------------------------------------------------------------------------------
# cost bands
# --------------------------------------------------------------------------------------------
def test_cost_bands_by_hand():
    est = J.get(MM)["estimate"]
    n, prompt, out_c, out_h = 1000, 1_700_000.0, 340_000.0, 375_000.0
    b = BM.cost_bands("minimax/minimax-m3", est, n, prompt, out_c, out_h, 1500)
    fee = 1.055
    cached = 640 * n
    low = ((prompt - cached) * 0.224 + cached * 0.0448 + out_c * 0.88) / 1e6
    reask = 1 + est["parse_error_rate"] + est["parse_error_rate"] * est["reask_fail_rate"]
    assert b["low_usd_with_fee"] == pytest.approx(low * fee, abs=1e-4)
    assert b["central_usd_with_fee"] == pytest.approx(low * reask * fee, abs=1e-4)
    assert b["high_usd_with_fee"] == pytest.approx((prompt * 0.224 + out_h * 0.88) / 1e6 * reask * fee, abs=1e-4)
    assert b["low_usd_with_fee"] <= b["central_usd_with_fee"] <= b["high_usd_with_fee"]
    assert b["submit_reservation_usd"] == pytest.approx((prompt * 0.224 + n * 1500 * 0.88) / 1e6, abs=1e-4)
    assert BM.output_tokens(est, "codeattack") == est["output_tokens_by_eval"]["codeattack"]
    assert BM.output_tokens(est, "codeattack", "high") == est["output_tokens_by_eval_high"]["codeattack"]
    assert BM.output_tokens(est, "no_such_eval") == est["output_tokens"]


# --------------------------------------------------------------------------------------------
# the full-run builder: reuse, context, tokenizer pin
# --------------------------------------------------------------------------------------------
RUN = "full22:pasttense:aaaaaaaaaaaa:bbbbbbbbbbbbbbbb"


def _texts(n=6):
    goals = {f"goal_{i}": {"text": f"Synthetic goal number {i}"} for i in range(3)}
    texts = []
    for i in range(n):
        g = f"goal_{i % 3}"
        resp = f"Synthetic response {i} " * (i + 1)
        k = R.cache_key("sr", "harmbench", "rev", goals[g]["text"], resp)
        texts.append({"k": k, "kind": "sr", "dataset": "harmbench", "revision": "rev", "goal_id": g, "response": resp})
    return goals, texts


def _enum(tmp_path, goals, texts):
    scope = tmp_path / "scope"
    scope.mkdir()
    (scope / "goals.json").write_text(json.dumps({"goals": goals}))
    (scope / "jobs.jsonl").write_text(json.dumps({"run_id": RUN, "kind": "sr", "category": "static", "campaign": "full22", "eval_id": "pasttense",
                                                  "haiku_judgment_id": "h" * 36, "haiku_judge_id": FR.HAIKU_JUDGES["sr"]}) + "\n")
    enum = tmp_path / "enum"
    enum.mkdir()
    with gzip.open(enum / "texts.jsonl.gz", "wt") as fh:
        for x in texts:
            fh.write(json.dumps(x) + "\n")
    with gzip.open(enum / "units.tsv.gz", "wt") as fh:
        fh.write("run_id\titem_id\ttrial\tsample\tturn\tcustom_id\tkind\tresponse_chars\n")
        for i, x in enumerate(texts):
            for trial in (0, 1):            # two units per key: the map keeps every unit
                fh.write(f"{RUN}\t{x['goal_id']}\t{trial}\t{i}\t-1\t{x['k']}\tsr\t{len(x['response'])}\n")
    return str(scope), str(enum)


def _arm(tmp_path, goals, texts, outcomes, max_tokens=1500, request=None):
    """A bake-off-arm-like run directory (the arm's track.json shape) whose ledger holds ``outcomes`` {index: (outcome, terminal)}."""
    rd = str(tmp_path / "arm")
    e = J.get(MM)
    BF.write_track(rd, {"track": "strongreject_bakeoff__minimax-m3", "model": e["model"], "endpoint": R.ENDPOINT, "provider": None,
                        "parser": "rejudge.strongreject.request:parse_result", "retry": dict(J.BASE_RETRY), "max_requests": 2000,
                        "request": dict(request or e["request"]), "truncated_retry_max_tokens": 4000})
    hdr = BF.batch_header(e["model"], R.ENDPOINT)
    reqs = [BF.PackedRequest(x["k"], BF.serialize_request(x["k"], R.request_body(e["request"], "sr", goals[x["goal_id"]]["text"], x["response"],
                                                                                 max_tokens=max_tokens)))
            for i, x in enumerate(texts) if i in outcomes]
    man = BF.write_manifest(rd, "b00000", hdr, reqs, "strongreject_bakeoff__minimax-m3")
    led = Ledger(rd).writer()
    led.register_manifest(man)
    led.update_batch("b00000", state="harvested")
    for i, (outcome, terminal) in outcomes.items():
        if outcome is None:
            continue
        led.record_outcome(texts[i]["k"], "b00000", "batch-1", outcome, terminal,
                           {"refused": 1, "convincing": 2, "specific": 1, "score": 0.0} if outcome == "verdict" else None,
                           None if outcome == "verdict" else {"kind": outcome}, 200, "tool_calls", None, None, "DeepInfra")
    led.close()
    return rd


def _build(scope, enum, rd, *extra):
    return BM.main(["--enum", enum, "--goals", os.path.join(scope, "goals.json"), "--run-dir", rd, "--judge", MM, "--workers", "1",
                    "--no-submit-gate", *extra])


def test_build_reuses_the_arms_terminal_verdicts_and_keeps_their_units(tmp_path):
    goals, texts = _texts(6)
    scope, enum = _enum(tmp_path, goals, texts)
    # 0, 1 verdicts; 2 a terminal parse error after ONE re-ask (this run allows two, so it is re-sent); 3 moderation (final either way);
    # 4 still waiting on a re-ask; 5 never in the arm
    arm = _arm(tmp_path, goals, texts, {0: ("verdict", True), 1: ("verdict", True), 2: ("parse_error", True), 3: ("moderation", True),
                                        4: ("parse_error", False)})
    rd = str(tmp_path / "full")
    assert _build(scope, enum, rd, "--reuse-run-dir", arm, "--max-spend-x-high", "2") == 0
    sent = {r["custom_id"] for k in BF.list_manifests(rd) for r in BF.body_requests(BF.read_body(rd, BF.read_manifest(rd, k)))}
    assert sent == {texts[i]["k"] for i in (2, 4, 5)}
    reuse = json.load(open(os.path.join(rd, "reuse.json")))
    assert set(reuse["keys"]) == {texts[i]["k"] for i in (0, 1, 3)} and reuse["reused_outcomes"] == {"verdict": 2, "moderation": 1}
    assert reuse["dirs"][0]["not_candidates"] == {"terminal_parse_error_fewer_reasks_there": 1, "not_terminal": 1}
    with gzip.open(os.path.join(rd, "units.tsv.gz"), "rt") as fh:
        assert sum(1 for _ in fh) - 1 == 12                     # every unit, the reused keys' included
    spec = BF.read_track(rd)
    assert spec["reuse_run_dirs"] == [os.path.abspath(arm)] and spec["model"] == "minimax/minimax-m3"
    rep = json.load(open(os.path.join(rd, "dry_run.json")))
    assert rep["groups"]["all"]["requests"] == 3 and rep["groups"]["all"]["reused"] == 3
    b = rep["bands"]["all"]
    assert 0 < b["low_usd_with_fee"] <= b["central_usd_with_fee"] <= b["high_usd_with_fee"]
    assert spec["max_spend_usd"] == 10 * math.ceil(2 * b["high_usd_with_fee"] / 10) == 10.0       # never 0 (no ceiling)
    assert rep["context"]["over_context"] == 0 and "turnaround" in rep
    for k in BF.list_manifests(rd):
        for r in BF.body_requests(BF.read_body(rd, BF.read_manifest(rd, k))):
            assert r["body"]["reasoning"] == {"enabled": False} and r["body"]["max_tokens"] == 1500


def test_build_re_sends_a_key_whose_arm_body_differs(tmp_path):
    goals, texts = _texts(3)
    scope, enum = _enum(tmp_path, goals, texts)
    arm = _arm(tmp_path, goals, texts, {0: ("verdict", True), 1: ("verdict", True)}, max_tokens=1200)
    rd = str(tmp_path / "full")
    assert _build(scope, enum, rd, "--reuse-run-dir", arm) == 0
    rep = json.load(open(os.path.join(rd, "dry_run.json")))
    assert rep["reuse"]["reused"] == 0 and rep["reuse"]["body_mismatch"] == 2 and rep["groups"]["all"]["requests"] == 3


def test_build_refuses_a_reuse_dir_of_another_protocol(tmp_path):
    goals, texts = _texts(2)
    scope, enum = _enum(tmp_path, goals, texts)
    arm = _arm(tmp_path, goals, texts, {0: ("verdict", True)}, request=dict(J.get(MM)["request"], reasoning=None, reasoning_effort="none"))
    with pytest.raises(SystemExit, match="another protocol"):
        _build(scope, enum, str(tmp_path / "full"), "--reuse-run-dir", arm)


def test_build_refuses_a_request_over_the_context(tmp_path, monkeypatch):
    goals, texts = _texts(3)
    texts.append(dict(texts[0]))
    big = "word " * 4000
    texts[-1].update(response=big, k=R.cache_key("sr", "harmbench", "rev", goals["goal_0"]["text"], big))
    scope, enum = _enum(tmp_path, goals, texts)
    monkeypatch.setitem(J.JUDGES[MM], "context_tokens", 4000)
    with pytest.raises(SystemExit, match="do not fit"):
        _build(scope, enum, str(tmp_path / "full"))
    over = json.load(open(tmp_path / "full" / "over_context.json"))
    assert [k for k, _ in over] == [texts[-1]["k"]]


def test_build_refuses_a_tokenizer_other_than_the_pinned_one(tmp_path):
    goals, texts = _texts(2)
    scope, enum = _enum(tmp_path, goals, texts)
    fake = tmp_path / "tokenizer.json"
    fake.write_text("{}")
    with pytest.raises(SystemExit, match="sha256"):
        _build(scope, enum, str(tmp_path / "full"), "--tokenizer", str(fake))


def test_a_full_scope_build_needs_a_gate_or_an_explicit_no_gate(tmp_path):
    goals, texts = _texts(2)
    scope, enum = _enum(tmp_path, goals, texts)
    with pytest.raises(SystemExit, match="needs"):
        BM.main(["--enum", enum, "--goals", os.path.join(scope, "goals.json"), "--run-dir", str(tmp_path / "f"), "--judge", MM])
    rd = str(tmp_path / "g")
    assert BM.main(["--enum", enum, "--goals", os.path.join(scope, "goals.json"), "--run-dir", rd, "--judge", MM, "--workers", "1",
                    "--require-file", str(tmp_path / "GO"), "--gate-note", "the go-ahead file"]) == 0
    gate = BF.read_track(rd)["submit_gate"]
    assert gate == {"requires_files": [str(tmp_path / "GO")], "note": "the go-ahead file"}


# --------------------------------------------------------------------------------------------
# filing
# --------------------------------------------------------------------------------------------
def test_filing_refuses_run_dirs_of_another_protocol_or_a_missing_reuse_dir(tmp_path):
    goals, texts = _texts(3)
    scope, enum = _enum(tmp_path, goals, texts)
    arm = _arm(tmp_path, goals, texts, {0: ("verdict", True)})
    full = str(tmp_path / "full")
    assert _build(scope, enum, full, "--reuse-run-dir", arm) == 0
    other = str(tmp_path / "other")
    T.init_run(other, mode="json_schema")                         # the same model under another request form: another protocol
    assert FR.judge_of_run_dirs([full, arm]) == MM and FR.judge_of_run_dirs([str(tmp_path / "missing")]) == MM
    assert FR.ledger_check([full, arm], MM) == []
    assert any("reuses the verdicts" in p for p in FR.ledger_check([full], MM))
    assert any("not the minimax-m3 entry" in p for p in FR.ledger_check([full, arm, other], MM))
    bare = str(tmp_path / "bare")
    Ledger(bare).close()
    assert any("no track.json" in p for p in FR.ledger_check([bare], MM))      # a bare ledger never passes


def test_minimax_filing_stages_new_rows_and_reads_the_arms_verdicts(tmp_path):
    goals, texts = _texts(4)
    scope, enum = _enum(tmp_path, goals, texts)
    arm = _arm(tmp_path, goals, texts, {0: ("verdict", True), 1: ("moderation", True)})
    full = str(tmp_path / "full")
    assert _build(scope, enum, full, "--reuse-run-dir", arm) == 0
    led = Ledger(full).writer()
    for x in texts[2:]:
        led.record_outcome(x["k"], "b00000", "batch-2", "verdict", True, {"refused": 0, "convincing": 4, "specific": 5, "score": 0.875}, None, 200,
                           "tool_calls", None, None, "DeepInfra")
    led.update_batch("b00000", state="harvested")
    led.close()
    staging = tmp_path / "staging"
    with pytest.raises(SystemExit, match="reuses the verdicts"):
        FR.main(["--jobs", os.path.join(scope, "jobs.jsonl"), "--units", os.path.join(full, "units.tsv.gz"), "--run-dir", full,
                 "--staging", str(staging), "--code-sha", "0" * 40, "--dry-run"])
    assert FR.main(["--jobs", os.path.join(scope, "jobs.jsonl"), "--units", os.path.join(full, "units.tsv.gz"), "--run-dir", full,
                    "--run-dir", arm, "--staging", str(staging), "--code-sha", "0" * 40, "--dry-run"]) == 0
    rep = json.load(open(staging / "filing_report.json"))
    assert rep["judge_entry"] == MM and rep["judge_model"] == "minimax-m3" and rep["judges"]["sr"] == MINIMAX_JUDGE_IDS["sr"]
    assert rep["runs"][RUN]["status"] == "complete" and rep["runs"][RUN]["verdict"] == 6 and rep["runs"][RUN]["moderation"] == 2
    assert all(len(v["refusals"]) == 0 for v in rep["validation"].values())
    rows = []
    for p in staging.rglob("*.jsonl*"):
        op = gzip.open if str(p).endswith(".gz") else open
        with op(p, "rt") as fh:
            rows += [json.loads(l) for l in fh if l.strip()]
    judges = [r for r in rows if r.get("record_type") == "judge" or "judge_model" in (r.get("payload") or {})]
    assert judges and all((r.get("payload") or r)["judge_model"] == "minimax-m3" for r in judges)
    jid = identity.judgment_id(RUN, MINIMAX_JUDGE_IDS["sr"])
    text = json.dumps(rows)
    assert jid in text and FR.HAIKU_JUDGES["sr"] in text          # keyed by the new judgment; the Haiku judgment only referenced
    assert '"judge_model": "minimax-m3"' in text
    # nothing but judge / judgment / item_result records: no run_state, run_patch, metric, supersede or withdrawal
    kinds = {r.get("record_type") or r.get("kind") or r.get("type") for r in rows}
    assert kinds <= {"judge", "judgment", "item_result", None}
    assert F.DEFAULT_HOST_LABEL
