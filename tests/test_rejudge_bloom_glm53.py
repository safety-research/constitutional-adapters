"""The Petri track's GLM-5.3 judge profile (scripts/rejudge/petri/judges.py and its use in track.py, build_manifests.py and
file_records.py): the full run sends exactly the judge comparison's request, pinned to DeepInfra; the comparison ledger's verdicts are
reused only when every body it sent is the one the full run would send; the filing names a new judge (the Opus judge's rubric, the
candidate's protocol) and refuses verdicts served by another provider. Offline: invented transcripts, a fake Batch API, temp
directories."""
from __future__ import annotations

import gzip
import json
import os
import pathlib
import sqlite3
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from rejudge.common import batchfile as BF  # noqa: E402
from rejudge.common.harvester import Harvester  # noqa: E402
from rejudge.petri import judges as J  # noqa: E402
from rejudge.petri import request as R  # noqa: E402
from rejudge.petri import track as T  # noqa: E402
from rejudge.petri.bakeoff import models as M  # noqa: E402
from test_rejudge_bloom_units import _enum_dir, make_inputs, verdict_entry  # noqa: E402

GLM = "glm-5.3"
PIN = {"only": ["deepinfra"], "allow_fallbacks": False}


def _usage_entry(req, attempt=0):
    e = verdict_entry(req, attempt)
    e["response"]["body"]["usage"] = {"prompt_tokens": 12000, "completion_tokens": 1500, "cost": 0.007,
                                      "prompt_tokens_details": {"cached_tokens": 4000, "cache_write_tokens": 0},
                                      "completion_tokens_details": {"reasoning_tokens": 150}}
    return e


def _harvest(rd, provider="DeepInfra", entry_fn=_usage_entry, cycles=6):
    from rejudge.petri.synthetic_proof import SyntheticAPI
    h = Harvester(rd, SyntheticAPI(entry_fn=entry_fn, provider=provider), log=lambda m: None)
    for _ in range(cycles):
        h.cycle()
        if not h.busy():
            break
    h.progress(done=not h.busy())
    h.ledger.close()


def _retrack(run_dir, track):
    """Rewrite a run directory's ``track.json`` under another track name (``write_track`` refuses a change, by design)."""
    tr = BF.read_track(run_dir)
    tr["track"] = track
    with open(os.path.join(run_dir, "track.json"), "w") as fh:
        json.dump(tr, fh)


def _bakeoff_ledger(tmp_path, rows, tools, name="bakeoff", cand=None):
    """The judge comparison's own run directory for the candidate (``track`` ``petri_bakeoff``, unpinned), as the comparison wrote it:
    its ``track.json`` fields and one batch per behaviour of ``bakeoff.models.openrouter_body`` (optionally from an altered candidate
    entry, to prove a body difference is caught)."""
    cand = dict(cand or M.CANDIDATES[GLM])
    rd = str(tmp_path / name)
    keys_file = tmp_path / f"{name}_keys.txt"
    keys_file.write_text("\n".join(r["key"] for r in rows) + "\n")
    spec = {"track": "petri_bakeoff", "model": cand["model"], "endpoint": R.ENDPOINT, "provider": None,
            "parser": "rejudge.petri.bakeoff.models:parse_result", "retry": T.RETRY, "max_bytes": BF.DEFAULT_MAX_BYTES,
            "max_requests": BF.DEFAULT_MAX_REQUESTS, "submit_spacing_s": 0,
            "request": {"max_tokens": cand["max_tokens"], "reasoning_effort": cand["reasoning_effort"], "split": False, "cache_prefix": False,
                        "cache_mode": None, "parallel_tool_calls": None if cand.get("drop_parallel_tool_calls") else False},
            "truncated_retry_max_tokens": cand["truncated_retry_max_tokens"], "health": dict(T.HEALTH), "max_spend_usd": 4.0,
            "estimate": {"answer_tokens": T.EST_ANSWER_TOKENS, "reasoning_tokens": 310.0, "framing_tokens": T.EST_FRAMING_TOKENS},
            "arm": GLM, "tool_choice": "forced", "output": "tool", "enum_dir": str(tmp_path / "enum"), "keys_file": str(keys_file),
            "id_suffix": "", "bakeoff": True}
    BF.write_track(rd, spec)
    header = BF.batch_header(spec["model"], spec["endpoint"], None)
    by_beh = {}
    for r in rows:
        by_beh.setdefault(r["behavior"], []).append(r)
    for i, beh in enumerate(sorted(by_beh)):
        packed = [BF.PackedRequest(r["key"], BF.serialize_request(r["key"], M.openrouter_body(r["prompt"], tools[r["tool_sha"]], cand)), 1.0, 1.0)
                  for r in sorted(by_beh[beh], key=lambda r: r["key"])]
        BF.write_manifest(rd, f"x{i:02d}00_{beh}", header, packed, "petri_bakeoff", extra={"behavior": beh, "arm": GLM})
    return rd


# --------------------------------------------------------------------------------------------
# the request: the judge comparison's, exactly
# --------------------------------------------------------------------------------------------
def test_the_glm_run_sends_the_bakeoff_arms_body_pinned_to_deepinfra(tmp_path):
    rows, tools = make_inputs(2)
    rd = str(tmp_path / "full")
    T.init_run(rd, judge=GLM, max_spend_usd=150)
    mans = T.add_inputs(rd, rows, tools)
    tr = BF.read_track(rd)
    cand = M.CANDIDATES[GLM]
    assert (tr["track"], tr["model"], tr["parser"], tr["provider"]) == ("petri_glm53", "z-ai/glm-5.3", "rejudge.petri.bakeoff.models:parse_result", PIN)
    assert tr["request"] == {"max_tokens": 16000, "reasoning_effort": "low", "split": False, "cache_prefix": False, "cache_mode": None,
                             "parallel_tool_calls": None}
    assert tr["truncated_retry_max_tokens"] == 32000 == cand["truncated_retry_max_tokens"] and tr["max_spend_usd"] == 150
    assert (tr["judge"], tr["arm"], tr["tool_choice"], tr["output"]) == (GLM, GLM, "forced", "tool")
    by_key = {r["key"]: r for r in rows}
    for m in mans:
        assert m["track"] == "petri_glm53" and m["provider"] == PIN
        body = BF.read_body(rd, m)
        head = json.loads(body)
        assert list(head) == ["endpoint", "model", "provider", "completion_window", "requests"] and head["provider"] == PIN
        for q in head["requests"]:
            r = by_key[q["custom_id"]]
            assert q["body"] == M.openrouter_body(r["prompt"], tools[r["tool_sha"]], cand)
            assert set(q["body"]) == {"messages", "tools", "tool_choice", "max_tokens", "reasoning"}
            assert q["body"]["reasoning"] == {"effort": "low"} and q["body"]["max_tokens"] == 16000
            assert q["body"]["tool_choice"] == {"type": "function", "function": {"name": "answer"}}
            assert R.body_text(q["body"]) == r["prompt"] and R.judge_input_key(R.body_text(q["body"]), R.body_tool(q["body"])) == q["custom_id"]


def test_the_glm_runs_request_lines_are_the_bakeoff_builds_byte_for_byte(tmp_path):
    rows, tools = make_inputs(2)
    _enum_dir(tmp_path, rows, tools)
    bake = _bakeoff_ledger(tmp_path, rows, tools)
    full = str(tmp_path / "full")
    T.init_run(full, judge=GLM)
    T.add_inputs(full, rows, tools)

    def lines(rd):
        return {q["custom_id"]: BF.serialize_request(q["custom_id"], q["body"])
                for k in BF.list_manifests(rd) for q in BF.body_requests(BF.read_body(rd, BF.read_manifest(rd, k)))}
    assert lines(bake) == lines(full) and len(lines(full)) == len(rows)
    bt, ft = BF.read_track(bake), BF.read_track(full)
    assert bt["request"] == ft["request"] == J.arm_request(J.profile(GLM))
    prof = J.profile(GLM)
    assert J.ledger_identity(bt, prof) == J.ledger_identity(ft, prof)
    assert bt["provider"] is None and ft["provider"] == PIN                     # the pin is routing, not identity


def test_the_profile_is_the_default_and_refuses_a_claude_candidate(tmp_path, monkeypatch):
    assert J.DEFAULT == GLM and T.init_run(str(tmp_path / "a")) == T.init_run(str(tmp_path / "b"), judge=GLM)
    from rejudge.petri import file_records as FR
    assert FR.protocol() == FR.protocol(judge=GLM) and FR.protocol()["served_by"] == ["DeepInfra"]
    with pytest.raises(ValueError):
        J.profile("opus-5-5-low")                                                # not a profile
    monkeypatch.setitem(M.CANDIDATES, "claude-x", dict(M.CANDIDATES[GLM], model="anthropic/claude-x"))
    monkeypatch.setitem(J.PROFILES, "claude-x", dict(J.PROFILES[GLM], arm="claude-x", model="anthropic/claude-x"))
    with pytest.raises(ValueError, match="Claude"):
        J.profile("claude-x")                                                    # a Claude model is never the re-judge
    assert all(not str(J.profile(n)["model"]).startswith(("claude", "anthropic/")) for n in J.PROFILES if n != "claude-x")


# --------------------------------------------------------------------------------------------
# build_manifests --judge glm-5.3 --reuse-ledger
# --------------------------------------------------------------------------------------------
def _bm(*a):
    from rejudge.petri import build_manifests as BM
    return BM.main(list(a))


def test_build_manifests_reuses_a_byte_identical_ledger_and_calibrates_on_it(tmp_path):
    rows, tools = make_inputs(4)
    enum = _enum_dir(tmp_path, rows, tools)
    bake = _bakeoff_ledger(tmp_path, rows[:5], tools)
    _harvest(bake)
    full = str(tmp_path / "full")
    assert _bm("--judge", GLM, "--enum", enum, "--run-dir", full, "--reuse-ledger", bake, "--no-submit-gate", "--max-spend-usd", "150") == 0
    from rejudge.petri import check_coverage as CC
    rep = CC.coverage(CC.enum_keys(enum), [bake, full])
    assert rep["partition"] and rep["per_run_dir"] == {bake: 5, full: len(rows) - 5}
    tr = BF.read_track(full)
    assert tr["reused_ledgers"] == [{"ledger": bake, "n_keys": 5, "bodies_byte_identical": 5, "outcomes": {"verdict": 5},
                                     "served_by": {"DeepInfra": 5}, "infra_terminal_resent": 0}] and tr["provider"] == PIN
    assert "submit_gate" not in tr and tr["estimate"]["reasoning_tokens"] == J.profile(GLM)["est_reasoning_tokens"]
    dr = json.load(open(os.path.join(full, "dry_run.json")))
    assert dr["requests"] == len(rows) - 5 and dr["excluded_keys"] == 5 and dr["calibration"]["n"] == 5
    assert dr["calibration"]["providers"] == {"DeepInfra": 5} and dr["calibration"]["completion_mean"] == 1500
    c = dr["cost_usd_with_fee"]
    assert 0 < c["low"]["usd_with_fee"] <= c["high"]["usd_with_fee"] and 0 < c["central"]["usd_with_fee"] <= c["high"]["usd_with_fee"]
    assert "Petri judge on glm-5.3" in open(os.path.join(full, "dry_run.md")).read()


def test_build_manifests_refuses_a_ledger_whose_bodies_differ_or_that_still_moves_and_writes_nothing(tmp_path):
    rows, tools = make_inputs(2)
    enum = _enum_dir(tmp_path, rows, tools)
    other = dict(M.CANDIDATES[GLM], max_tokens=12000)                         # same identity settings, another body
    bad = _bakeoff_ledger(tmp_path, rows[:2], tools, name="bad", cand=other)
    _harvest(bad)
    with pytest.raises(SystemExit) as e:
        _bm("--judge", GLM, "--enum", enum, "--run-dir", str(tmp_path / "f1"), "--reuse-ledger", bad, "--no-submit-gate")
    assert "differs from this build's" in str(e.value) and not os.path.exists(tmp_path / "f1")
    fresh = _bakeoff_ledger(tmp_path, rows[:2], tools, name="fresh")           # never harvested: no verdicts
    with pytest.raises(SystemExit) as e:
        _bm("--judge", GLM, "--enum", enum, "--run-dir", str(tmp_path / "f2"), "--reuse-ledger", fresh, "--no-submit-gate")
    assert "never harvested" in str(e.value) and not os.path.exists(tmp_path / "f2")
    paused = _bakeoff_ledger(tmp_path, rows[:2], tools, name="paused")         # harvested, then paused: not stopped moving
    _harvest(paused)
    pathlib.Path(paused, "PAUSED").write_text("a person is looking\n")
    with pytest.raises(SystemExit) as e:
        _bm("--judge", GLM, "--enum", enum, "--run-dir", str(tmp_path / "f2"), "--reuse-ledger", paused, "--no-submit-gate")
    assert "stopped moving" in str(e.value) and not os.path.exists(tmp_path / "f2")
    other = _bakeoff_ledger(tmp_path, rows[:2], tools, name="other")         # another track: never a ledger of this judge
    _harvest(other)
    _retrack(other, "petri_other")
    with pytest.raises(SystemExit) as e:
        _bm("--judge", GLM, "--enum", enum, "--run-dir", str(tmp_path / "f3"), "--reuse-ledger", other, "--no-submit-gate")
    assert "not a ledger of the glm-5.3 judge" in str(e.value)
    for flag in (["--effort", "high"], ["--no-split"], ["--max-tokens", "12000"], ["--id-suffix", ":x"]):   # the request is the profile's
        with pytest.raises(SystemExit):
            _bm("--judge", GLM, "--enum", enum, "--run-dir", str(tmp_path / "f4"), "--no-submit-gate", *flag)
    with pytest.raises(SystemExit):
        _bm("--enum", enum, "--run-dir", str(tmp_path / "f5"), "--no-submit-gate", "--reuse-ledger", bad)    # the default profile refuses it too


# --------------------------------------------------------------------------------------------
# filing
# --------------------------------------------------------------------------------------------
RUN = "testcamp:petri_pooled:000000000000:0000000000000000"


def _world(tmp_path, provider_full="DeepInfra"):
    rows, tools = make_inputs(3)
    enum = _enum_dir(tmp_path, rows, tools)
    scope = tmp_path / "scope"
    scope.mkdir()
    (scope / "jobs.jsonl").write_text(json.dumps({"run_id": RUN, "campaign": "testcamp", "opus_judgment_id": "o" * 36}) + "\n")
    with gzip.open(scope / "units.tsv.gz", "wt") as fh:
        fh.write("run_id\titem_id\ttrial_index\tsample_index\tturn_index\tphys\n")
        for r in rows:
            fh.write(f"{RUN}\t{r['behavior']}/{r['phys'].split('|')[3]}\t0\t0\t-1\t{r['phys']}\n")
    bake = _bakeoff_ledger(tmp_path, rows[:3], tools)
    _harvest(bake)
    full = str(tmp_path / "full")
    assert _bm("--judge", GLM, "--enum", enum, "--run-dir", full, "--reuse-ledger", bake, "--no-submit-gate", "--max-spend-usd", "5") == 0
    tr = BF.read_track(full)
    tr["submit_spacing_s"] = 0
    BF.write_track(full, tr)
    _harvest(full, provider=provider_full)
    root = tmp_path / "root"
    (root / "records").mkdir(parents=True)
    return rows, tools, enum, str(scope), bake, full, str(root)


def _fr(*a):
    from rejudge.petri import file_records as FR
    return FR.main(list(a))


def test_the_glm_filing_is_a_new_judge_with_the_opus_rubric_and_validates(tmp_path):
    pytest.importorskip("inspect_petri")
    from rejudge.petri import file_records as FR
    rows, tools, enum, scope, bake, full, root = _world(tmp_path)
    st, per = FR.ledger_settings([bake, full], GLM)
    assert per[bake]["track"] == "petri_bakeoff" and per[bake]["provider"] is None and per[full]["provider"] == PIN
    glm = FR.judge_spec(tools, [16000], GLM)
    assert glm["judge_model"] == "glm-5.3" and glm["judge_backend"] == "batch" and glm["rubric_name"] == "petri_bloom_judge"
    assert glm["rubric_sha256"] == FR.rubric_sha256(tools) and glm["judge_id"] != FR.PETRI_JUDGE
    assert glm["threshold_rule"] == FR.OPUS_THRESHOLD_RULE and glm["extra"]["protocol"]["served_by"] == ["DeepInfra"]
    assert FR.ledger_settings([bake], GLM)[0] == st                             # the routing pin is not identity
    assert _fr("--judge", GLM, "--scope", scope, "--enum", enum, "--run-dir", bake, "--run-dir", full, "--staging", str(tmp_path / "st"),
               "--registry-root", root, "--code-sha", "0" * 40, "--dry-run") == 0
    rep = json.load(open(tmp_path / "st" / "filing_report.json"))
    assert rep["judge_id"] == glm["judge_id"] and rep["track"] == "petri_glm53" and rep["validation"]["refusals"] == []
    assert rep["runs"][RUN]["status"] == "complete" and rep["served_by_counts"] == {"DeepInfra": len(rows)}
    assert rep["validation"]["records"] == {"judge": 1, "judgment": 1, "item_result": 3 * len(rows)}
    staged = list(pathlib.Path(tmp_path / "st").rglob("*-item_result.jsonl.gz"))
    assert len(staged) == 1 and f"petri_glm53__{glm['judge_id'][:12]}" in str(staged[0])
    with gzip.open(staged[0], "rt") as fh:
        recs = [json.loads(l) for l in fh]
    assert {r["payload"]["extra"]["judge_model"] for r in recs} == {"glm-5.3"} and {r["payload"]["extra"]["provider"] for r in recs} == {"DeepInfra"}


def test_the_glm_filing_refuses_another_provider_and_mixed_judges(tmp_path):
    pytest.importorskip("inspect_petri")
    from rejudge.petri import file_records as FR
    rows, tools, enum, scope, bake, full, root = _world(tmp_path, provider_full="Morph")
    with pytest.raises(SystemExit) as e:
        _fr("--judge", GLM, "--scope", scope, "--enum", enum, "--run-dir", bake, "--run-dir", full, "--staging", str(tmp_path / "st"),
            "--registry-root", root, "--dry-run")
    assert "served by 'Morph'" in str(e.value) and not os.path.exists(tmp_path / "st")
    other = str(tmp_path / "other")
    T.init_run(other)
    _retrack(other, "petri_other")
    with pytest.raises(ValueError):
        FR.ledger_settings([bake, other], GLM)                                 # a ledger of another track is never this judge's
    con = sqlite3.connect(os.path.join(full, "state.sqlite"))
    assert {p for (p,) in con.execute("SELECT DISTINCT provider FROM requests")} == {"Morph"}
    con.close()


# --------------------------------------------------------------------------------------------
# infrastructure outcomes settled by a follow-up; reuse reads the outcomes; the pin is the profile's; the protocol pinned
# --------------------------------------------------------------------------------------------
def _ledger_with(tmp_path, name, outcome, harvested="2026-11-11T00:00:00Z", provider="DeepInfra"):
    from rejudge.common.state import Ledger
    rd = str(tmp_path / name)
    led = Ledger(rd).writer()
    led.register_manifest({"batch_key": "b0", "n_requests": 1, "body_bytes": 1, "custom_ids": ["k" * 64]})
    led.record_outcome("k" * 64, "b0", f"batch-{name}", outcome, True, {"petri_score": 3} if outcome == "verdict" else None,
                       None if outcome == "verdict" else {"kind": outcome}, 200, None, None, None, provider)
    led.close()
    con = sqlite3.connect(os.path.join(rd, "state.sqlite"))
    con.execute("UPDATE requests SET harvested_utc = ?", (harvested,))
    con.commit()
    con.close()
    return rd


def test_an_infrastructure_outcome_is_settled_by_another_ledgers_verdict_in_either_order(tmp_path):
    from rejudge.petri import file_records as FR
    failed = _ledger_with(tmp_path, "full", "batch_failed")
    follow = _ledger_with(tmp_path, "followup", "verdict", harvested="2026-11-12T00:00:00Z")
    for order in ([failed, follow], [follow, failed]):
        o = FR.read_outcomes(order)["k" * 64]
        assert o["outcome"] == "verdict" and o["run_dir"] == follow and o["settled_infra"] == [failed]
    late = _ledger_with(tmp_path, "late", "provider_error", harvested="2026-11-13T00:00:00Z")
    for order in ([failed, late], [late, failed]):                         # two infrastructure outcomes: the later harvest
        assert FR.read_outcomes(order)["k" * 64]["run_dir"] == late
    rejected = _ledger_with(tmp_path, "rejected", "moderation")
    with pytest.raises(FR.DuplicateOutcomes):                              # two fileable outcomes: still refused
        FR.read_outcomes([follow, failed, rejected])


def test_reuse_excludes_only_fileable_deepinfra_outcomes_and_refuses_bad_ledgers(tmp_path):
    rows, tools = make_inputs(2)
    enum = _enum_dir(tmp_path, rows, tools)
    flaky = rows[0]["key"]

    def entry(req, attempt=0):
        if req["custom_id"] == flaky:
            return {"custom_id": flaky, "response": None, "error": {"code": 500, "message": "upstream"}}
        return _usage_entry(req, attempt)
    bake = _bakeoff_ledger(tmp_path, rows[:3], tools)
    _harvest(bake, entry_fn=entry, cycles=30)
    con = sqlite3.connect(os.path.join(bake, "state.sqlite"))
    assert dict(con.execute("SELECT custom_id, outcome FROM requests"))[flaky] == "provider_error"
    con.close()
    full = str(tmp_path / "full")
    assert _bm("--judge", GLM, "--enum", enum, "--run-dir", full, "--reuse-ledger", bake, "--no-submit-gate") == 0
    sent = {c for k in BF.list_manifests(full) for c in BF.read_manifest(full, k)["custom_ids"]}
    assert flaky in sent and not (set(r["key"] for r in rows[1:3]) & sent) and len(sent) == len(rows) - 2
    assert BF.read_track(full)["reused_ledgers"][0]["infra_terminal_resent"] == 1
    morph = _bakeoff_ledger(tmp_path, rows[:2], tools, name="morph")
    _harvest(morph, provider="Morph")
    with pytest.raises(SystemExit) as e:
        _bm("--judge", GLM, "--enum", enum, "--run-dir", str(tmp_path / "f1"), "--reuse-ledger", morph, "--no-submit-gate")
    assert "served by 'Morph'" in str(e.value) and not os.path.exists(tmp_path / "f1")
    synth = _bakeoff_ledger(tmp_path, rows[:2], tools, name="synth")
    _harvest(synth)
    pathlib.Path(synth, "SYNTHETIC").write_text("synthetic\n")
    with pytest.raises(SystemExit) as e:
        _bm("--judge", GLM, "--enum", enum, "--run-dir", str(tmp_path / "f2"), "--reuse-ledger", synth, "--no-submit-gate")
    assert "SYNTHETIC" in str(e.value)
    extra = _bakeoff_ledger(tmp_path, rows[:2], tools, name="extra")
    _harvest(extra)
    tr = BF.read_track(extra)
    header = BF.batch_header(tr["model"], tr["endpoint"], tr.get("provider"))
    r = rows[3]
    BF.write_manifest(extra, "zz_never_submitted", header, [BF.PackedRequest(
        r["key"], BF.serialize_request(r["key"], M.openrouter_body(r["prompt"], tools[r["tool_sha"]], M.CANDIDATES[GLM])), 1.0, 1.0)], "petri_bakeoff")
    with pytest.raises(SystemExit) as e:
        _bm("--judge", GLM, "--enum", enum, "--run-dir", str(tmp_path / "f3"), "--reuse-ledger", extra, "--no-submit-gate")
    assert "not in its ledger" in str(e.value)


def test_an_arm_profiles_pin_and_fields_cannot_be_overridden(tmp_path):
    with pytest.raises(ValueError):
        T.init_run(str(tmp_path / "a"), judge=GLM, provider={"only": ["morph"]})
    for clash in ({"provider": None}, {"model": "z-ai/glm-5.3-flash"}, {"request": {}}, {"parser": "x:y"}):
        with pytest.raises(ValueError):
            T.init_run(str(tmp_path / "b"), judge=GLM, extra=clash)
    assert not os.path.exists(tmp_path / "a") and not os.path.exists(tmp_path / "b")


def test_the_glm_protocol_hash_is_pinned():
    """A change to the body builders or the candidate's entry would mint another judge id between filings: this literal is the protocol
    half of the judge row on record (d4dc6939; tests/test_rejudge_paper_judges.py checks the whole row)."""
    from rejudge.common import filing as F
    from rejudge.petri import file_records as FR
    assert F.canonical_sha256(FR.protocol(judge=GLM)) == "b6ee14ea0a88884f17c50ff4ea0847a15a6cc48c9c1b3ed88a01321c367dd65c"
