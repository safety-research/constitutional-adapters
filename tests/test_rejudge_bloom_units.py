"""The Petri track's dedup, batching, the real harvester over a fake Batch API, and the registry rows it files
(scripts/rejudge/petri/track.py, file_records.py, synthetic_proof.py). Offline: invented transcripts, a fake API, a temp run directory;
the record validator is the registry's own."""
from __future__ import annotations

import gzip
import json
import os
import pathlib
import sqlite3
import sys
import time

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from rejudge.common import batchfile as BF  # noqa: E402
from rejudge.common.harvester import Harvester  # noqa: E402
from rejudge.petri import judges as J  # noqa: E402
from rejudge.petri import request as R  # noqa: E402
from rejudge.petri import track as T  # noqa: E402

#: the paper judge's profile (GLM-5.3)
PROF = J.profile()

sys.path.insert(0, str(ROOT / "tests"))
from test_rejudge_bloom_request import synthetic_prompt, synthetic_tool  # noqa: E402

BEHAVIORS = {"self_preservation": "self_preservation", "defer_to_users": "defer_to_users",
             "cyber_bio_chem_capabilities": "cyber_bio_chem_operational_uplift"}


def make_inputs(n_per=3):
    """Rows shaped like enumerate_inputs' inputs.jsonl.gz, plus the tools table."""
    rows, tools = [], {}
    for beh, own in BEHAVIORS.items():
        tool = R.canonical_tool(synthetic_tool(own))
        ts = R.tool_sha(tool)
        tools[ts] = tool
        for i in range(n_per):
            prompt = synthetic_prompt(f"[M1] USER:\n{beh} transcript {i}\n\n[M2] ASSISTANT:\nreply {i}")
            head, rest = R.split_prompt(prompt)
            rows.append({"phys": f"petri/TESTHOST_audit|00000000|{beh}|scenario_{i}|1", "key": R.judge_input_key(prompt, tool),
                         "prefix_sha": R.prefix_sha(prompt, tool), "tool_sha": ts, "behavior": beh, "own_dimension": own, "prompt": prompt,
                         "tokens": {"head": len(head) // 4, "rest": len(rest) // 4, "tool": 2000}})
    return rows, tools


def test_one_request_per_distinct_input_one_behaviour_per_batch(tmp_path):
    rows, tools = make_inputs()
    rd = str(tmp_path / "run")
    T.init_run(rd)
    mans = T.add_inputs(rd, rows + rows[:2], tools)                       # duplicates dedup to one request each
    assert len(mans) == len(BEHAVIORS)
    ids = [c for m in mans for c in m["custom_ids"]]
    assert sorted(ids) == sorted(r["key"] for r in rows) and len(ids) == len(set(ids))
    for m in mans:
        beh = m["extra"]["behavior"]
        body = BF.read_body(rd, m)
        reqs = BF.body_requests(body)
        assert {R.own_dimension(R.body_tool(q["body"])) for q in reqs} == {BEHAVIORS[beh]}
        for q in reqs:
            r = next(x for x in rows if x["key"] == q["custom_id"])
            assert R.body_text(q["body"]) == r["prompt"] and R.judge_input_key(R.body_text(q["body"]), R.body_tool(q["body"])) == q["custom_id"]
        assert json.loads(body)["model"] == PROF["model"] and json.loads(body)["endpoint"] == R.ENDPOINT
        assert json.loads(body)["provider"] == {"only": ["deepinfra"], "allow_fallbacks": False}
        assert m["extra"]["cache_estimate"]["n_prefixes"] == 1
    tr = BF.read_track(rd)
    assert tr["parser"] == "rejudge.petri.bakeoff.models:parse_result" and tr["retry"]["moderation"] == 0 and tr["retry"]["parse_error"] == 1


def test_a_changed_text_is_refused(tmp_path):
    rows, tools = make_inputs(1)
    rows[0] = dict(rows[0], prompt=rows[0]["prompt"] + "x")
    rd = str(tmp_path / "run")
    T.init_run(rd)
    with pytest.raises(ValueError):
        T.add_inputs(rd, rows, tools)


def test_cost_range_orders_best_nocache_worst():
    rows, _ = make_inputs()
    est = T.cache_estimate(rows)
    c = T.cost_range(PROF["model"], est, 3000)
    assert c["best"] < c["no_cache"] <= c["worst"]          # the model's catalogue lists no cache-write price (billed as input)


# --------------------------------------------------------------------------------------------
# the real harvester over the fake API, then the rows
# --------------------------------------------------------------------------------------------
@pytest.fixture()
def harvested(tmp_path):
    from rejudge.petri.synthetic_proof import SyntheticAPI
    rows, tools = make_inputs(12)
    rd = str(tmp_path / "run")
    T.init_run(rd, extra={"submit_spacing_s": 0})
    T.add_inputs(rd, rows, tools)
    h = Harvester(rd, SyntheticAPI(), log=lambda m: None)
    for _ in range(8):
        h.cycle()
        if not h.busy():
            break
    h.progress(done=True)
    h.ledger.close()
    return rd, rows, tools


def test_every_request_reaches_a_terminal_outcome(harvested):
    rd, rows, _ = harvested
    con = sqlite3.connect(os.path.join(rd, "state.sqlite"))
    got = dict(con.execute("SELECT custom_id, outcome FROM requests WHERE terminal = 1").fetchall())
    assert set(got) == {r["key"] for r in rows}
    assert set(got.values()) <= {"verdict", "moderation", "judge_refusal", "parse_error"}
    v = json.loads(con.execute("SELECT verdict_json FROM requests WHERE outcome = 'verdict' LIMIT 1").fetchone()[0])
    assert {"petri_score", "eval_awareness", "scenario_realism", "own_dimension", "highlights", "summary", "justification"} <= set(v)


def test_rows_map_one_verdict_to_every_unit_and_record_every_rejection(harvested):
    from rejudge.petri import file_records as FR
    rd, rows, tools = harvested
    keys = {r["phys"]: {"key": r["key"], "tool_sha": r["tool_sha"], "own_dimension": r["own_dimension"]} for r in rows}
    outcomes = FR.read_outcomes([rd])
    # two runs (a pooled run and a per-behaviour twin) list the same transcripts
    units = [{"phys": r["phys"], "item_id": f"{r['behavior']}/{r['phys'].split('|')[3]}", "trial_index": "0", "sample_index": "0", "turn_index": "-1"}
             for r in rows]
    a, ia = FR.run_rows("run_pooled", "J1", units, keys, outcomes, tools, "OPUS1", "2026-11-10T00:00:00.000Z")
    b, ib = FR.run_rows("run_twin", "J2", units, keys, outcomes, tools, "OPUS2", "2026-11-10T00:00:00.000Z")
    assert len(a) == len(b) and ia["counts"] == ib["counts"]
    assert sum(ia["counts"].values()) == len(units)
    for r in rows:
        o = outcomes[r["key"]]
        mine = [x for x in a if x["extra"]["request_key"] == r["key"]]
        if o["outcome"] == "verdict":
            assert sorted(x["metric_id"] for x in mine) == ["eval_awareness", "petri_score", "scenario_realism"]
            ps = next(x for x in mine if x["metric_id"] == "petri_score")
            assert ps["value"] == o["verdict"]["petri_score"] and ps["extra"]["own_dimension"] == r["own_dimension"] and ps["label"] is None
        else:
            assert [(x["metric_id"], x["value"], x["label"]) for x in mine] == [("petri_score", None, o["outcome"])]
        assert all(x["judgment_id"] == "J1" and x["source_revision_id"] == "" and x["extra"]["re_measures_judgment_id"] == "OPUS1" for x in mine)
    # a unit whose input has no terminal outcome gets no row; one whose verdict scores another dimension files as parse_error
    first = rows[0]
    v = next(o for o in outcomes.values() if o["outcome"] == "verdict")
    bad = {first["key"]: dict(v, verdict=dict(v["verdict"], own_dimension="defer_to_users" if first["own_dimension"] != "defer_to_users" else "self_preservation"))}
    c, ic = FR.run_rows("r", "J", units[:2], keys, bad, tools, "O", "2026-11-10T00:00:00.000Z")
    assert ic["counts"] == {"parse_error": 1, "unjudged": 1}
    assert c[0]["label"] == "parse_error" and c[0]["extra"]["error_kind"] == "own_dimension_mismatch"


def test_the_staged_records_pass_the_registry_validator(harvested, tmp_path):
    pytest.importorskip("inspect_petri")
    from rejudge.common import filing as F
    from rejudge.petri import file_records as FR
    rd, rows, tools = harvested
    judge = FR.judge_spec(tools)
    assert judge["judge_model"] == "glm-5.3" and judge["rubric_name"] == "petri_bloom_judge" and judge["judge_backend"] == "batch"
    assert judge["judge_id"] != FR.PETRI_JUDGE and judge["threshold_rule"] == FR.OPUS_THRESHOLD_RULE
    assert FR.judge_spec(dict(reversed(list(tools.items()))))["judge_id"] == judge["judge_id"]      # independent of the table's order
    keys = {r["phys"]: {"key": r["key"], "tool_sha": r["tool_sha"], "own_dimension": r["own_dimension"]} for r in rows}
    units = [{"phys": r["phys"], "item_id": f"{r['behavior']}/{r['phys'].split('|')[3]}", "trial_index": "0", "sample_index": "0", "turn_index": "-1"}
             for r in rows]
    filer = F.Filer(str(tmp_path / "staging"), PROF["track"], judge, "tests", "0" * 40, session="TESTHOST")
    filer.stage_judge()
    jid = F.identity.judgment_id("TESTRUN", filer.judge_id)
    items, info = FR.run_rows("TESTRUN", jid, units, keys, FR.read_outcomes([rd]), tools, "OPUS", F.utc_ms())
    jm = F.judgment_payload(run_id="TESTRUN", judge_id=filer.judge_id, status="complete", n_items_judged=info["counts"].get("verdict", 0),
                            n_judge_refusals=info["counts"].get("judge_refusal", 0), n_errors=0, started_utc=info["first_harvest"],
                            finished_utc=info["last_harvest"], batch_ids=info["batch_ids"], grading_code_sha="0" * 40,
                            rubric_sha256=judge["rubric_sha256"], judge_cache_kind="petri_judge_input_sha256", store_uri=None,
                            producer_session="TESTHOST", extra={})
    filer.stage_run("TESTRUN", jm, items)
    v = filer.validate()
    assert v["refusals"] == [] and v["records"] == {"judge": 1, "judgment": 1, "item_result": len(items)}


def test_protocol_carries_no_text_and_no_max_tokens():
    from rejudge.petri import file_records as FR
    p = FR.protocol()
    s = json.dumps(p)
    assert "max_tokens" not in p["body_template"] and "You are an impartial judge" not in s
    assert p["body_template"]["reasoning"] == {"effort": "low"} and p["body_template"]["tool_choice"]["function"]["name"] == "answer"
    assert "prompt_cache_options" not in p["body_template"] and isinstance(p["body_template"]["messages"][0]["content"], str)
    assert p["served_by"] == ["DeepInfra"] and p["model"] == PROF["model"] and FR.protocol("glm-5.3") == p


def test_synthetic_run_directories_are_never_filed_live(tmp_path, monkeypatch):
    from rejudge.petri import file_records as FR
    rd = tmp_path / "run"
    rd.mkdir()
    (rd / "SYNTHETIC").write_text("x")
    with pytest.raises(SystemExit):
        FR.main(["--scope", str(tmp_path), "--enum", str(tmp_path), "--run-dir", str(rd), "--staging", str(tmp_path / "s")])
    with pytest.raises(SystemExit):
        FR.main(["--scope", str(tmp_path), "--enum", str(tmp_path), "--run-dir", str(rd), "--staging", str(tmp_path / "s"),
                 "--allow-synthetic", "--registry-root", os.path.expanduser("~/registry")])
    st = tmp_path / "staged"
    st.mkdir()
    (st / "SYNTHETIC").write_text("x")
    with pytest.raises(SystemExit):
        FR.main(["--staging", str(st), "--registry-root", os.path.expanduser("~/registry"), "--deliver", "judge", "--deliver-only"])



# --------------------------------------------------------------------------------------------
# the library's guards in the Petri track
# --------------------------------------------------------------------------------------------
DEAD_KEY = "HTTP 401: Incorrect API key provided: sk-proj-****abcd. You can find your API key at https://platform.openai.com/account/api-keys."


def verdict_entry(req, attempt=0, finish="tool_calls"):
    """A valid answer to a request, scored from its custom_id (no synthetic outcome mix)."""
    tool = R.canonical_tool(req["body"]["tools"][0])
    own = R.own_dimension(tool)
    h = int(req["custom_id"][:8], 16)
    args = {"highlights": "h", "summary": "s", own: 1 + h % 10, "eval_awareness": 1 + (h >> 4) % 10, "scenario_realism": 1 + (h >> 8) % 10,
            "justification": "j"}
    msg = {"role": "assistant", "content": None, "tool_calls": [{"id": "c", "type": "function", "function": {"name": "answer", "arguments": json.dumps(args)}}]}
    return {"custom_id": req["custom_id"], "error": None, "response": {"status_code": 200, "body": {
        "usage": {"prompt_tokens": 9000, "completion_tokens": 2500, "cost": 0.02}, "choices": [{"finish_reason": finish, "message": msg}]}}}


def _harvest(rd, entry_fn, cycles=4):
    from rejudge.petri.synthetic_proof import SyntheticAPI
    api = SyntheticAPI(entry_fn=entry_fn, provider="DeepInfra")            # the profile's serving provider (the filing refuses another)
    h = Harvester(rd, api, log=lambda m: None)
    for _ in range(cycles):
        h.cycle()
    return h, api


def _run_dir(tmp_path, name, rows, tools, **init):
    rd = str(tmp_path / name)
    T.init_run(rd, extra={"submit_spacing_s": 0}, **init)
    T.add_inputs(rd, rows, tools)
    return rd


def test_track_json_carries_the_librarys_guards(tmp_path):
    rd = str(tmp_path / "run")
    T.init_run(rd, max_spend_usd=12.5, requires_files=[str(tmp_path / "smoke" / "GO")], est_reasoning_tokens=3000)
    tr = BF.read_track(rd)
    assert tr["truncated_retry_max_tokens"] == 32000 > tr["request"]["max_tokens"] == 16000
    assert tr["health"] == T.HEALTH and tr["max_spend_usd"] == 12.5
    assert tr["submit_gate"]["requires_files"] == [str(tmp_path / "smoke" / "GO")]
    assert tr["request"]["cache_mode"] is None and T.output_per_request(tr) == T.EST_ANSWER_TOKENS + 3000
    assert T.output_per_request(BF.read_track(T.init_run(str(tmp_path / "dflt")) and str(tmp_path / "dflt"))) == T.EST_ANSWER_TOKENS + 310.0
    T.init_run(str(tmp_path / "plain"))
    assert "submit_gate" not in BF.read_track(str(tmp_path / "plain")) and BF.read_track(str(tmp_path / "plain"))["max_spend_usd"] == 0.0


def test_a_per_request_account_failure_in_a_one_request_batch_pauses_and_records_nothing(tmp_path):
    """One request per batch answered per request with a dead own-provider key: a non-terminal account_error, never a provider_error
    row; the run pauses at once, nothing is re-sent, the filing gate refuses, and the fixed account resumes it without spent re-asks."""
    from rejudge.common.harvester import EXIT_DONE, EXIT_PAUSED
    from rejudge.common.state import readiness
    rows, tools = make_inputs(1)                                          # 3 inputs, 3 behaviours: three one-request batches
    rd = _run_dir(tmp_path, "smoke", rows, tools)
    dead = {"on": True}

    def entry(req, attempt):
        if dead["on"]:
            return {"custom_id": req["custom_id"], "error": None,
                    "response": {"status_code": 401, "body": {"error": {"code": 401, "message": DEAD_KEY}}}}
        return verdict_entry(req)
    h, api = _harvest(rd, entry)
    assert h.ledger.outcome_counts() == {"account_error": 3}
    assert not any(r["terminal"] for r in h.ledger.iter_requests("account_error"))
    assert "account/credential" in h.paused() and h.final_state()[0] == EXIT_PAUSED
    assert api.submits == 3 and not [k for k in BF.list_manifests(rd) if k.startswith("retry")]
    assert not readiness(rd)["ready"]
    dead["on"] = False
    os.remove(os.path.join(rd, "PAUSED"))
    for _ in range(4):
        h.cycle()
    assert h.ledger.outcome_counts() == {"verdict": 3} and h.final_state()[0] == EXIT_DONE
    h.ledger.close()


def test_a_per_request_balance_failure_is_held_back_and_never_terminal(tmp_path):
    rows, tools = make_inputs(1)
    rd = _run_dir(tmp_path, "run", rows, tools)
    h, _ = _harvest(rd, lambda req, a: {"custom_id": req["custom_id"], "response": None, "error": {"code": 402, "message": "Insufficient credits"}}, cycles=3)
    assert h.ledger.outcome_counts() == {"balance_error": 3}
    assert not any(r["terminal"] for r in h.ledger.iter_requests("balance_error"))
    held = [b for b in h.ledger.batches(("pending",)) if b["batch_key"].startswith("retry")]
    assert held and all(float(b["next_attempt_ts"] or 0) > time.time() + 200 for b in held)    # a backoff, not an immediate re-POST
    h.ledger.close()


def test_a_truncated_reply_is_re_asked_once_with_the_opus_cap(tmp_path):
    rows, tools = make_inputs(4)                                          # 12 inputs: 4 per behaviour
    rd = _run_dir(tmp_path, "run", rows, tools)
    cut = rows[0]["key"]
    h, _ = _harvest(rd, lambda req, a: verdict_entry(req, finish="length" if req["custom_id"] == cut and a == 0 else "tool_calls"), cycles=6)
    retry = [k for k in BF.list_manifests(rd) if k.startswith("retry")]
    assert len(retry) == 1
    (q,) = BF.body_requests(BF.read_body(rd, BF.read_manifest(rd, retry[0])))
    assert q["custom_id"] == cut and q["body"]["max_tokens"] == R.OPUS_MAX_TOKENS == 32000
    assert h.ledger.outcome_counts() == {"verdict": 12}
    h.ledger.close()


def _enum_dir(tmp_path, rows, tools):
    d = tmp_path / "enum"
    d.mkdir(exist_ok=True)
    with gzip.open(d / "inputs.jsonl.gz", "wt") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    (d / "tools.json").write_text(json.dumps(tools))
    return str(d)


def _keys_file(tmp_path, name, keys):
    p = tmp_path / name
    p.write_text("\n".join(keys) + "\n")
    return str(p)


def test_build_manifests_excludes_keys_and_gates_a_full_scope_run(tmp_path):
    """The full run never re-sends the pilot's inputs, and a full-scope run must name its submit gate."""
    from rejudge.petri import build_manifests as BM
    rows, tools = make_inputs(4)
    enum = _enum_dir(tmp_path, rows, tools)
    allk = [r["key"] for r in rows]
    smoke, pilot = allk[:1], allk[:5]
    with pytest.raises(SystemExit) as e:
        BM.main(["--enum", enum, "--run-dir", str(tmp_path / "nogate")])
    assert "--require-file" in str(e.value) and "--no-submit-gate" in str(e.value)
    sk, pk = _keys_file(tmp_path, "smoke.txt", smoke), _keys_file(tmp_path, "pilot.txt", pilot)
    assert BM.main(["--enum", enum, "--run-dir", str(tmp_path / "pilot"), "--keys", pk, "--exclude-keys", sk, "--max-spend-usd", "12",
                    "--require-file", str(tmp_path / "smoke_report.json")]) == 0
    report = str(tmp_path / "pilot_report.json")
    assert BM.main(["--enum", enum, "--run-dir", str(tmp_path / "full"), "--exclude-keys", sk, "--exclude-keys", pk,
                    "--require-file", report, "--max-spend-usd", "400"]) == 0
    ids = lambda rd: {c for k in BF.list_manifests(rd) for c in BF.read_manifest(rd, k)["custom_ids"]}  # noqa: E731
    assert ids(str(tmp_path / "pilot")) == set(pilot) - set(smoke)
    assert ids(str(tmp_path / "full")) == set(allk) - set(pilot)
    tr = BF.read_track(str(tmp_path / "full"))
    assert tr["submit_gate"]["requires_files"] == [report] and tr["max_spend_usd"] == 400.0
    dr = json.load(open(tmp_path / "full" / "dry_run.json"))
    assert dr["excluded_keys"] == 5 and dr["requests"] == len(allk) - 5 and dr["ceiling_estimate_usd_with_fee"] > 0
    # the gate holds the full run: nothing is submitted while the pilot report is missing
    h, api = _harvest(str(tmp_path / "full"), verdict_entry, cycles=1)
    assert api.submits == 0 and "submit gate" in h.paused()
    h.ledger.close()


def test_a_key_terminal_in_two_ledgers_is_refused(tmp_path):
    """A verdict in one ledger and a moderation in another never resolve by ledger order."""
    from rejudge.common.state import Ledger
    from rejudge.petri import file_records as FR
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    for rd, outcome in ((a, "verdict"), (b, "moderation")):
        led = Ledger(rd).writer()
        led.register_manifest({"batch_key": "b0", "n_requests": 1, "body_bytes": 1, "custom_ids": ["k" * 64]})
        led.record_outcome("k" * 64, "b0", "batch-1", outcome, True, {"petri_score": 3} if outcome == "verdict" else None,
                           None if outcome == "verdict" else {"kind": "moderation"}, 200, None, None, None, "OpenAI")
        led.close()
    assert set(FR.read_outcomes([a])) == {"k" * 64}
    with pytest.raises(FR.DuplicateOutcomes):
        FR.read_outcomes([a, b])


def test_ledger_settings_derive_the_judge_and_refuse_probes_and_other_protocols(tmp_path):
    """The judge row's protocol comes from the profile, and every ledger must carry the profile's request settings (``track.json``): a
    ledger built with other settings or a probe is never filed under this judge. max_tokens is outside the identity."""
    pytest.importorskip("inspect_petri")
    from rejudge.petri import file_records as FR
    _, tools = make_inputs(1)
    base = str(tmp_path / "base")
    T.init_run(base)
    st, per = FR.ledger_settings([base])
    assert st == {k: v for k, v in J.arm_request(PROF).items() if k != "max_tokens"} | J.arm_shape(PROF) and per[base]["max_tokens"] == 16000
    j0 = FR.judge_spec(tools)["judge_id"]
    assert j0 == FR.judge_spec(tools, [16000, 32000])["judge_id"]                 # the reply cap is not identity
    other = str(tmp_path / "other")
    T.init_run(other)
    tr = BF.read_track(other)
    tr["request"]["reasoning_effort"] = "high"
    BF.write_track(other, tr)
    with pytest.raises(ValueError, match="request settings"):
        FR.ledger_settings([base, other])
    probe = str(tmp_path / "probe")
    T.init_run(probe, extra={"id_suffix": ":high"})
    with pytest.raises(ValueError):
        FR.ledger_settings([probe])
    with pytest.raises(SystemExit) as e:                                         # a run directory never harvested: a clean refusal
        FR.main(["--scope", str(tmp_path), "--enum", str(tmp_path), "--run-dir", base, "--staging", str(tmp_path / "st"), "--dry-run"])
    assert "never harvested" in str(e.value)


# --------------------------------------------------------------------------------------------
# the filing driver end to end: the delivery gate and follow-up filings
# --------------------------------------------------------------------------------------------
RUN = "testcamp:petri_self_preservation:000000000000:0000000000000000"


def _filing_world(tmp_path):
    """Three self_preservation transcripts of one paper run, the scope and enumeration files, a ledger built by the track, an empty
    registry root."""
    rows, tools = make_inputs(3)
    rows = [r for r in rows if r["behavior"] == "self_preservation"]
    tools = {r["tool_sha"]: tools[r["tool_sha"]] for r in rows}
    enum = _enum_dir(tmp_path, rows, tools)
    scope = tmp_path / "scope"
    scope.mkdir()
    (scope / "jobs.jsonl").write_text(json.dumps({"run_id": RUN, "campaign": "testcamp", "opus_judgment_id": "o" * 36}) + "\n")
    with gzip.open(scope / "units.tsv.gz", "wt") as fh:
        fh.write("run_id\titem_id\ttrial_index\tsample_index\tturn_index\tphys\n")
        for i, r in enumerate(rows):
            fh.write(f"{RUN}\tself_preservation/scenario_{i}\t0\t0\t-1\t{r['phys']}\n")
    rd = _run_dir(tmp_path, "ledger", rows, tools)
    root = tmp_path / "root"
    (root / "records").mkdir(parents=True)
    return rd, rows, enum, str(scope), str(root)


def _file(tmp_path, rd, enum, scope, root, staging, *extra):
    from rejudge.petri import file_records as FR
    rds = rd if isinstance(rd, list) else [rd]
    return FR.main(["--scope", scope, "--enum", enum, *[x for r in rds for x in ("--run-dir", r)], "--staging", str(tmp_path / staging),
                    "--code-sha", "0" * 39 + "1", "--registry-root", root, *extra])


def _inbox(root):
    return sorted(str(p.relative_to(root)) for p in pathlib.Path(root, "inbox").rglob("*.jsonl*")) if os.path.isdir(os.path.join(root, "inbox")) else []


def _merge(root):
    """What the registry harvester's inbox merge does: move inbox/<host>/<rel> to records/<rel>."""
    from rejudge.common import filing as F
    ib = pathlib.Path(root, "inbox", F.DEFAULT_HOST_LABEL)
    for p in sorted(ib.rglob("*.jsonl*")):
        dest = pathlib.Path(root, "records", p.relative_to(ib))
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.replace(p, dest)


def test_the_delivery_gate_refuses_a_paused_moving_or_partial_ledger(tmp_path):
    pytest.importorskip("inspect_petri")
    rd, rows, enum, scope, root = _filing_world(tmp_path)
    last = rows[2]["key"]
    mode = {"last": "dead"}

    def entry(req, attempt):
        if req["custom_id"] == last and mode["last"] == "dead":
            return {"custom_id": last, "response": None, "error": {"code": 401, "message": DEAD_KEY}}
        if req["custom_id"] == last and mode["last"] == "infra":
            return {"custom_id": last, "response": None, "error": {"code": 500, "message": "upstream"}}
        return verdict_entry(req)
    h, _ = _harvest(rd, entry)
    assert h.paused() and h.ledger.outcome_counts() == {"verdict": 2, "account_error": 1}
    # paused, one request waiting on a person: nothing is delivered, whatever the flags
    assert _file(tmp_path, rd, enum, scope, root, "s1", "--deliver", "all", "--allow-partial") == 3 and _inbox(root) == []
    rep = json.load(open(tmp_path / "s1" / "filing_report.json"))
    assert any("PAUSED" in p for p in rep["gate"]["problems"]) and rep["runs"][RUN]["status"] == "partial"
    assert "account_error" not in rep["runs"][RUN] and rep["runs"][RUN]["unjudged"] == 1          # never an item outcome
    # the account is fixed; the request then fails on the provider on every attempt: terminal, NOT filed as a row, the run partial
    os.remove(os.path.join(rd, "PAUSED"))
    mode["last"] = "infra"
    for _ in range(12):
        h.cycle()
    assert h.ledger.outcome_counts() == {"verdict": 2, "provider_error": 1} and not h.busy()
    h.ledger.close()
    assert _file(tmp_path, rd, enum, scope, root, "s2", "--deliver", "all") == 3 and _inbox(root) == []
    assert _file(tmp_path, rd, enum, scope, root, "s3", "--deliver", "all", "--allow-partial") == 0
    rep = json.load(open(tmp_path / "s3" / "filing_report.json"))
    assert rep["runs"][RUN]["status"] == "partial" and rep["runs"][RUN]["infra_unfiled"] == 1 and rep["runs"][RUN]["rows"] == 6
    assert len(_inbox(root)) == 3                                                 # judge, judgment, item_result


def test_a_follow_up_filing_is_staged_above_the_registrys_names_and_delivers(tmp_path):
    """A run filed partial (a pilot first, or a run whose re-ask finished later) can be filed again: the second staging
    numbers its files above the registry's, the judge is not staged again, and the delivery is not refused."""
    pytest.importorskip("inspect_petri")
    from rejudge.common.state import Ledger
    from rejudge.petri import file_records as FR
    from sisteer.registry import identity
    from sisteer.registry import records as RR
    rd, rows, enum, scope, root = _filing_world(tmp_path)
    late = rows[2]["key"]
    h, _ = _harvest(rd, lambda req, a: verdict_entry(req) if req["custom_id"] != late else
                    {"custom_id": late, "response": None, "error": {"code": 500, "message": "upstream"}}, cycles=12)
    assert h.ledger.outcome_counts() == {"verdict": 2, "provider_error": 1}
    h.ledger.close()
    assert _file(tmp_path, rd, enum, scope, root, "first", "--deliver", "all", "--allow-partial") == 0
    _merge(root)
    first = sorted(os.path.basename(str(p)) for p in pathlib.Path(root, "records").rglob("*.jsonl*"))
    assert first == ["000001-judge.jsonl", "000001-judgment.jsonl", "000002-item_result.jsonl.gz"]
    # the infrastructure failure is judged later (an operator's re-run): a follow-up filing of the same run
    man = BF.read_manifest(rd, next(k for k in BF.list_manifests(rd) if late in BF.read_manifest(rd, k)["custom_ids"]))
    req = next(q for q in BF.body_requests(BF.read_body(rd, man)) if q["custom_id"] == late)
    led = Ledger(rd).writer()
    led.record_outcome(late, man["batch_key"], "batch-late", "verdict", True, R.parse_result(verdict_entry(req))["verdict"], None, 200,
                       "tool_calls", None, None, "DeepInfra")
    led.close()
    assert _file(tmp_path, rd, enum, scope, root, "second", "--deliver", "all") == 0
    rep = json.load(open(tmp_path / "second" / "filing_report.json"))
    assert rep["runs"][RUN]["status"] == "complete" and rep["gate"]["follow_up_runs"] == [RUN]
    second = _inbox(root)
    assert sorted(os.path.basename(p) for p in second) == ["000003-judgment.jsonl", "000004-item_result.jsonl.gz"]   # no judge again, no name reused
    recs = [rec for p in second for _, rec, _ in RR.read_record_file(os.path.join(root, p))]
    (jg,) = [r for r in recs if r["record_type"] == "judgment"]
    jid = FR.judge_spec(json.load(open(os.path.join(enum, "tools.json"))))["judge_id"]
    assert jg["payload"]["judgment_id"] == identity.judgment_id(RUN, jid) and jg["payload"]["status"] == "complete"
    assert jg["payload"]["extra"]["follow_up_of_seq"] == 2


def test_deliver_only_refuses_another_ledger_set_or_a_changed_ledger(tmp_path):
    pytest.importorskip("inspect_petri")
    from rejudge.common.state import Ledger
    from rejudge.petri import file_records as FR
    rd, rows, enum, scope, root = _filing_world(tmp_path)
    h, _ = _harvest(rd, verdict_entry)
    h.ledger.close()
    assert _file(tmp_path, rd, enum, scope, root, "st", "--dry-run") == 0
    base = ["--staging", str(tmp_path / "st"), "--registry-root", root, "--deliver-only"]
    FR.main(base + ["--deliver", "judge"])
    with pytest.raises(SystemExit) as e:
        FR.main(base + ["--deliver", "runs"])                                   # no --run-dir: the ledgers it was staged from are unknown
    assert "exactly the ledgers" in str(e.value)
    led = Ledger(rd).writer()                                                    # one more attempt recorded after staging
    led.record_outcome(rows[0]["key"], "b00000_self_preservation", "batch-x", "moderation", True, None, {"kind": "moderation"}, 403, None,
                       None, None, "DeepInfra")
    led.close()
    with pytest.raises(SystemExit) as e:
        FR.main(base + ["--run-dir", rd, "--deliver", "runs"])
    assert "changed since" in str(e.value)
    assert len(_inbox(root)) == 1                                                # the judge only


def test_check_coverage_names_overlaps_and_gaps(tmp_path):
    """The smoke, pilot and full run directories must partition the inputs: a pilot rebuilt with another cut and a stale full run
    either sends inputs twice or never."""
    from rejudge.petri import build_manifests as BM
    from rejudge.petri import check_coverage as CC
    rows, tools = make_inputs(2)
    enum = _enum_dir(tmp_path, rows, tools)
    ks = [r["key"] for r in rows]
    for name, sel in (("a", ks[:2]), ("b", ks[2:]), ("c", ks[1:3])):
        assert BM.main(["--enum", enum, "--run-dir", str(tmp_path / name), "--keys", _keys_file(tmp_path, name + ".txt", sel)]) == 0
    assert CC.main(["--enum", enum, "--run-dir", str(tmp_path / "a"), "--run-dir", str(tmp_path / "b")]) == 0
    rep = CC.coverage(CC.enum_keys(enum), [str(tmp_path / "a"), str(tmp_path / "b"), str(tmp_path / "c")])
    assert not rep["partition"] and rep["in_two_or_more"] == 2 and rep["in_none"] == 0
    rep = CC.coverage(CC.enum_keys(enum), [str(tmp_path / "a")])
    assert not rep["partition"] and rep["in_none"] == len(ks) - 2
    assert CC.main(["--enum", enum, "--run-dir", str(tmp_path / "a")]) == 1
