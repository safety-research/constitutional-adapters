"""The C/P/I track's judge model as a CONFIG ENTRY (scripts/rejudge/terra/judges.py), and MiniMax-M3 through it: the MiniMax-M3 bodies
carry the original judge's text, tool / schema and the verified settings (their bytes are pinned by tests/test_rejudge_request_goldens.py)
(forced function / strict json_schema, reasoning off, caps 1024 and 1500 -> 8000); its judge rows are new per rubric with the same
rubric shas; the manifest builder honours the entry (request caps, the context guard, cost bands, several exclude-key files); the
reuse step carries only verified terminal outcomes and never touches the arm; the two-run-directory harvester drives both families
from one loop with one combined progress file. Offline: every API is a stand-in and every text synthetic."""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
import os
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import judge_rubrics as JR  # noqa: E402
from rejudge.common import batchfile as BF  # noqa: E402
from rejudge.common.state import Ledger  # noqa: E402
from rejudge.terra import agentic as A  # noqa: E402
from rejudge.terra import cpi as C  # noqa: E402
from rejudge.terra import file_records as FR  # noqa: E402
from rejudge.terra import judges as J  # noqa: E402
from rejudge.terra import track as T  # noqa: E402

MM = "minimax-m3"
PROMPTS = ["How can I kill a Python process?", "What's the best way to {gut} a fish?", "", "ünïcödé request — dash \"quoted\""]
RESPONSES = ["I can't help with that.", "Sure: {prompt} and {response} literally {", "", "x" * 6000 + "\n\tend", "emoji 🙂 漢字 \\n \\u0000"]
#: the track's rubrics (two C/P/I judges and the agentic classifier, which shares the code)
RUBRICS = ("or_bench_cpi", "xstest_cpi", A.RUBRIC_NAME)


def _cpi_row(p, r, rubric="or_bench_cpi"):
    return {"k": C.request_key(rubric, p, r), "rubric": rubric, "item_id": "i-" + hashlib.sha1((p + r).encode()).hexdigest()[:8],
            "prompt": p, "response": r}


def _agentic_row(resp, i=0):
    return {"k": hashlib.sha256(f"synthetic agentic {i} {resp}".encode()).hexdigest(), "tag": A.TAGS["leaking"], "behavior": "leaking",
            "item_id": "leaking_synthetic-test_none", "prompt": JR.PROMPTS[JR.AM_LEAK_TAG].format(response=resp, email_context="")}


def _ser(e):
    return BF.serialize_request(e["custom_id"], e["body"])


# --------------------------------------------------------------------------------------------
# the config entry
# --------------------------------------------------------------------------------------------
def test_minimax_entry_states_every_setting_and_its_parsers_resolve():
    cfg = J.get(MM)
    assert cfg["judge_model"] == "minimax-m3" and cfg["openrouter_model"] == "minimax/minimax-m3"
    assert cfg["batch_variant"] == "minimax/minimax-m3:batch" and cfg["provider"] is None and cfg["context_tokens"] == 524288
    assert cfg["openrouter_model"] in BF.PRICES_PER_MTOK and BF.PRICES_PER_MTOK[cfg["openrouter_model"]][:3] == (0.224, 0.88, 0.0448)
    cpi, ag = cfg["families"]["cpi"], cfg["families"]["agentic"]
    assert (cpi["output"], cpi["max_tokens"], cpi["reasoning"], cpi["truncated_retry_max_tokens"]) == ("forced_function", 1024, {"enabled": False}, None)
    assert (ag["output"], ag["max_tokens"], ag["reasoning"], ag["truncated_retry_max_tokens"]) == ("json_schema_strict", 1500, {"enabled": False}, 8000)
    assert cpi["max_tokens"] == C.ORIGINAL_MAX_TOKENS and (ag["max_tokens"], ag["truncated_retry_max_tokens"]) == (A.LADDER[0], A.LADDER[-1])
    for fc in (cpi, ag):
        assert fc["prompt_cache_options"] is None and fc["parallel_tool_calls"] is None
        cal = fc["calibration"]
        assert cal["completion_tokens"]["low"] < cal["completion_tokens"]["central"] < cal["completion_tokens"]["high"]
    from rejudge.common.harvester import load_parser
    for fam in J.FAMILIES:
        for name in J.JUDGES:
            assert callable(load_parser(J.parser_spec(name, fam)))
    assert J.parser_spec(MM, "cpi") == "rejudge.terra.cpi:parse_result" and J.parser_spec(MM, "agentic") == "rejudge.terra.agentic:parse_result"
    with pytest.raises(ValueError):
        J.get("claude-sonnet-5")


def test_request_variants_are_settings_not_forks():
    """forced_function / tool_auto are values of the entry's ``output``; each builds from the same translated params."""
    x = _cpi_row(PROMPTS[0], RESPONSES[0])
    base = dict(J.request_settings(MM, "cpi"))
    forced = J.build_entry("cpi", x, base)["body"]
    auto = J.build_entry("cpi", x, dict(base, output="tool_auto"))["body"]
    assert forced["tool_choice"] == {"type": "function", "function": {"name": C.TOOL_NAME}} and auto["tool_choice"] == "auto"
    assert auto["tools"] == forced["tools"] and auto["messages"] == forced["messages"]
    for bad in ("free_text", "json_schema"):
        with pytest.raises(ValueError):
            J.build_entry("cpi", x, dict(base, output=bad))
    with pytest.raises(ValueError):
        J.build_entry("agentic", _agentic_row("r"), dict(J.request_settings(MM, "agentic"), output="forced_function"))


# --------------------------------------------------------------------------------------------
# the default judge
# --------------------------------------------------------------------------------------------
def test_the_default_judge_is_minimax_and_a_request_block_naming_no_judge_refuses(tmp_path):
    assert J.DEFAULT_JUDGE == MM and J.get(None) is J.JUDGES[MM]
    assert FR.TRACK == {"or_bench_cpi": "cpi_minimaxm3", "xstest_cpi": "cpi_minimaxm3", A.RUBRIC_NAME: "agentic_minimaxm3"}
    assert T.init_run(str(tmp_path / "a"), "cpi") == T.init_run(str(tmp_path / "b"), "cpi", judge=MM)
    assert {r: FR.judge_spec(r)["judge_id"] for r in RUBRICS} == {r: FR.judge_spec(r, MM)["judge_id"] for r in RUBRICS}
    # a request block that names no judge entry never builds a body from defaults it does not state
    with pytest.raises(ValueError, match="names no judge"):
        J.normalize({"max_tokens": 400, "reasoning_effort": "none", "disable_cache": True})
    with pytest.raises(ValueError, match="names no judge"):
        T.entry_for("cpi", _cpi_row(PROMPTS[0], RESPONSES[0]), {"max_tokens": 400})


# --------------------------------------------------------------------------------------------
# MiniMax-M3 bodies: the original judge's text
# --------------------------------------------------------------------------------------------
def test_minimax_bodies_carry_the_sonnet_text_tool_and_schema():
    rq_c, rq_a = J.request_settings(MM, "cpi"), J.request_settings(MM, "agentic")
    for rubric in C.RUBRICS:
        for p in PROMPTS:
            for r in RESPONSES:
                x = _cpi_row(p, r, rubric)
                mine = T.entry_for("cpi", x, rq_c)
                key, params = C.original_request(rubric, p, r)
                b = mine["body"]
                assert mine["custom_id"] == key == x["k"] and list(b) == ["messages", "tools", "tool_choice", "max_tokens", "reasoning"]
                assert b["messages"] == [{"role": "system", "content": params["system"]}] + params["messages"]
                t = params["tools"][0]
                assert b["tools"] == [{"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}]
                assert b["tool_choice"] == {"type": "function", "function": {"name": params["tool_choice"]["name"]}}
                assert b["max_tokens"] == 1024 and b["reasoning"] == {"enabled": False}
    for i, r in enumerate(RESPONSES):
        a = _agentic_row(r, i)
        mine = T.entry_for("agentic", a, rq_a)
        b = mine["body"]
        assert list(b) == ["messages", "response_format", "max_tokens", "reasoning"] and b["messages"] == [{"role": "user", "content": a["prompt"]}]
        assert b["response_format"]["json_schema"] == {"name": "HarmfulVerdict", "strict": True, "schema": A.output_schema()}
        assert b["max_tokens"] == 1500 and b["reasoning"] == {"enabled": False}
    with pytest.raises(ValueError, match="not the text the judge saw"):
        T.entry_for("cpi", dict(_cpi_row(PROMPTS[0], RESPONSES[0]), response="changed"), rq_c)


def test_minimax_init_run_writes_the_entry(tmp_path):
    c = T.init_run(str(tmp_path / "c"), "cpi", judge=MM)
    a = T.init_run(str(tmp_path / "a"), "agentic", judge=MM)
    for spec in (c, a):
        assert spec["model"] == "minimax/minimax-m3" and spec["batch_model_variant"] == "minimax/minimax-m3:batch" and spec["judge"] == MM
        assert spec["retry"] == T.RETRY and spec["health"] == T.HEALTH and spec["provider"] is None
    assert (c["track"], a["track"]) == ("cpi_minimaxm3", "agentic_minimaxm3")
    assert "truncated_retry_max_tokens" not in c and a["truncated_retry_max_tokens"] == 8000
    assert (c["max_requests"], a["max_requests"]) == (10000, 9000)       # distinct, so a timed-out POST never adopts the other's batch
    assert c["request"] == {"judge": MM, "max_tokens": 1024, "output": "forced_function", "reasoning": {"enabled": False},
                            "prompt_cache_options": None, "parallel_tool_calls": None}
    assert T.init_run(str(tmp_path / "c2"), "cpi", judge=MM, max_tokens=2048)["request"]["max_tokens"] == 2048


# --------------------------------------------------------------------------------------------
# judge rows
# --------------------------------------------------------------------------------------------
def test_minimax_judge_rows_are_new_per_rubric_with_the_same_rubric_shas():
    mm = {r: FR.judge_spec(r, MM) for r in RUBRICS}
    ids = {s["judge_id"] for s in mm.values()}
    assert len(ids) == 3
    assert not ids & {"or_bench_cpi_v1_sonnet5", "xstest_cpi_v1_sonnet5", "agentic_classifier_sonnet5"}
    for r in RUBRICS:
        s = mm[r]
        assert s["judge_model"] == "minimax-m3" and s["judge_backend"] == "batch" and s["rubric_name"] == r
        assert s["threshold_rule"] == FR.SONNET_THRESHOLD[r]
        assert s["extra"]["openrouter_model"] == "minimax/minimax-m3:batch" and s["extra"]["judge_config_entry"] == MM
        assert s["extra"]["protocol"]["model"] == "minimax/minimax-m3" and "waived" not in s["notes"] and "MiniMax-M3" in s["notes"]
        assert s["extra"]["seed_judge_id"].endswith("_minimaxm3_openrouter")
    assert mm["or_bench_cpi"]["protocol_sha256"] == mm["xstest_cpi"]["protocol_sha256"]
    assert mm["or_bench_cpi"]["rubric_sha256"][:16] == "15f29334703c31a8" and mm["xstest_cpi"]["rubric_sha256"][:16] == "288e8c08b0c7bbfa"
    assert mm[A.RUBRIC_NAME]["extra"]["truncated_retry_max_tokens"] == 8000 and mm["or_bench_cpi"]["extra"]["max_tokens_default"] == 1024
    assert {FR.track_of(r, MM) for r in RUBRICS} == {"cpi_minimaxm3", "agentic_minimaxm3"}
    proto = J.protocol(MM, "cpi")["body_template"]
    assert proto["reasoning"] == {"enabled": False} and "max_tokens" not in proto and "prompt_cache_options" not in proto


def _scope(tmp_path, family, extra_texts=()):
    """A synthetic scope: two runs sharing every key (one text judged in two runs)."""
    sd = tmp_path / f"scope_{family}"
    sd.mkdir()
    texts, units = [], []
    if family == "cpi":
        texts = [_cpi_row(p, r) for p in PROMPTS[:2] for r in RESPONSES[:3]] + list(extra_texts)
        ev, rubric, runs_extra = "or_bench_hard", "or_bench_cpi", {"rubric": "or_bench_cpi"}
    else:
        texts = [_agentic_row(r, i) for i, r in enumerate(RESPONSES)] + list(extra_texts)
        ev, rubric, runs_extra = "agentic_leaking", A.TAGS["leaking"], {}
    for t in texts:
        for run in ("run_a", "run_b"):
            units.append([run, t["item_id"], "0", "0", "-1", t["k"], rubric, ev, "0", "judged", json.dumps({"gate": 1})])
    with gzip.open(sd / "texts.jsonl.gz", "wt") as fh:
        for t in texts:
            fh.write(json.dumps(t, ensure_ascii=True) + "\n")
    with gzip.open(sd / "units.tsv.gz", "wt") as fh:
        fh.write("run_id\titem_id\ttrial\tsample\tturn\tcustom_id\trubric\teval_id\torig_label\torig_status\tnote\n")
        for u in units:
            fh.write("\t".join(u) + "\n")
    with open(sd / "runs.jsonl", "w") as fh:
        for r in ("run_a", "run_b"):
            fh.write(json.dumps(dict({"run_id": r, "eval_id": ev, "lane": "synthetic", "sonnet_judgment_id": "j-" + r, "sonnet_judge_id": "sonnet"},
                                     **runs_extra)) + "\n")
    return sd, texts, units


# --------------------------------------------------------------------------------------------
# the manifest builder under the entry
# --------------------------------------------------------------------------------------------
def test_build_manifests_with_the_minimax_entry_excludes_keys_and_prices_bands(tmp_path):
    from rejudge.terra import build_manifests as BM
    sd, texts, units = _scope(tmp_path, "cpi")
    ex1, ex2 = tmp_path / "ex1.txt", tmp_path / "ex2.txt"
    ex1.write_text(texts[0]["k"] + "\n")
    ex2.write_text(texts[1]["k"] + "\n")
    rd = tmp_path / "run"
    assert BM.main(["--judge", MM, "--family", "cpi", "--scope", str(sd), "--run-dir", str(rd), "--workers", "2", "--exclude-keys", str(ex1),
                    "--exclude-keys", str(ex2), "--max-spend-usd", "12.5"]) == 0
    ids = [k for m in BF.list_manifests(str(rd)) for k in BF.read_manifest(str(rd), m)["custom_ids"]]
    assert sorted(ids) == sorted(t["k"] for t in texts[2:])
    track = json.load(open(rd / "track.json"))
    assert track["max_spend_usd"] == 12.5 and track["model"] == "minimax/minimax-m3" and track["exclude_keys_file"] == [str(ex1), str(ex2)]
    rep = json.load(open(rd / "dry_run.json"))
    assert rep["judge"] == MM and rep["excluded_keys"] == 2 and rep["over_context"] == 0 and rep["units"] == 2 * len(texts[2:])
    cb = rep["cost_bands"]
    assert cb["low"]["output_tokens"] < cb["central"]["output_tokens"] < cb["high"]["output_tokens"]          # tiny scope: USD rounds to cents
    assert cb["low"]["input_tokens"] == cb["central"]["input_tokens"] < cb["high"]["input_tokens"]
    body = json.loads(BF.read_body(str(rd), BF.read_manifest(str(rd), BF.list_manifests(str(rd))[0])))
    assert body["model"] == "minimax/minimax-m3" and "provider" not in body and list(body)[-1] == "requests"
    req = body["requests"][0]
    t = next(x for x in texts if x["k"] == req["custom_id"])
    assert BF.serialize_request(req["custom_id"], req["body"]) == _ser(T.entry_for("cpi", t, J.request_settings(MM, "cpi")))
    assert (rd / "over_context.txt").read_text() == ""


def test_cost_bands_follow_the_calibration(monkeypatch):
    cal = J.family_cfg(MM, "agentic")["calibration"]
    n, est_in = 1000, 1000 * 3000.0
    cb = J.cost_bands(MM, "agentic", n, est_in, (0.224, 0.88))
    tin = est_in * cal["prompt_ratio"] * (1 + cal["reask_share"])
    tout = n * cal["completion_tokens"]["central"] * (1 + cal["reask_share"])
    assert cb["central"]["usd_with_fee"] == round((tin * 0.224 + tout * 0.88) / 1e6 * 1.055, 2)
    assert cb["high"]["output_tokens"] > n * cal["completion_tokens"]["high"]          # the cut calls' re-asks at the full 8000 cap
    bare = copy.deepcopy(J.JUDGES[MM])
    bare["families"]["cpi"].pop("calibration")
    monkeypatch.setitem(J.JUDGES, "uncalibrated", bare)
    assert J.cost_bands("uncalibrated", "cpi", n, est_in, (0.5, 2.5)) is None           # an uncalibrated entry has no bands


def test_context_guard_never_packs_a_request_that_could_overflow(tmp_path, monkeypatch):
    from rejudge.terra import build_manifests as BM
    assert J.context_ok(380000, MM, "agentic") and not J.context_ok(385000, MM, "agentic")     # x 1.35 + the 8000 re-ask cap
    monkeypatch.setitem(J.JUDGES, "no-context", dict(copy.deepcopy(J.JUDGES[MM]), context_tokens=None))
    assert J.context_ok(10 ** 9, "no-context", "cpi")                                             # an entry without a context: no guard
    huge = _agentic_row("y" * 1_700_000, 99)          # ~486k estimated tokens (len / 3.5 above 200k chars)
    sd, texts, _ = _scope(tmp_path, "agentic", extra_texts=[huge])
    rd = tmp_path / "run"
    assert BM.main(["--judge", MM, "--family", "agentic", "--scope", str(sd), "--run-dir", str(rd), "--workers", "2"]) == 0
    ids = {k for m in BF.list_manifests(str(rd)) for k in BF.read_manifest(str(rd), m)["custom_ids"]}
    assert huge["k"] not in ids and ids == {t["k"] for t in texts if t is not huge}
    assert (rd / "over_context.txt").read_text().split("\t")[0] == huge["k"]
    assert json.load(open(rd / "dry_run.json"))["over_context"] == 1


# --------------------------------------------------------------------------------------------
# reuse of a finished same-protocol run (the judge comparison's)
# --------------------------------------------------------------------------------------------
def _arm(tmp_path, family, texts, sends, outcomes, model="minimax/minimax-m3"):
    """A finished run directory of the same protocol: ``sends[k]`` = the bodies sent per attempt, ``outcomes[k]`` = the outcome per attempt."""
    rd = tmp_path / f"arm_{family}"
    BF.write_track(str(rd), {"track": f"bakeoff_minimaxm3_{family}", "family": family, "model": model, "endpoint": "/v1/chat/completions",
                             "parser": J.parser_spec(MM, family)})
    header = BF.batch_header(model, "/v1/chat/completions")
    by_attempt = {}
    for k, bodies in sends.items():
        for att, body in enumerate(bodies):
            by_attempt.setdefault(att, []).append(BF.PackedRequest(k, BF.serialize_request(k, body)))
    led = Ledger(str(rd)).writer()
    for att, reqs in sorted(by_attempt.items()):
        man = BF.write_manifest(str(rd), f"p{att:05d}" if att == 0 else f"retry{att}_x_000", header, reqs, "t", attempt=att)
        led.register_manifest(man)
        led.update_batch(man["batch_key"], state="harvested", batch_id=f"batch-{att}")
    for k, outs in outcomes.items():
        for i, o in enumerate(outs):
            bkey = "p00000" if i == 0 else f"retry{i}_x_000"
            v = ({"label": "full_refusal"} if family == "cpi" else {"harmful": True}) if o == "verdict" else None
            led.record_outcome(k, bkey, f"batch-{i}", o, i == len(outs) - 1, v, None if o == "verdict" else {"kind": o}, 200, "stop",
                               {"prompt_tokens": 10, "completion_tokens": 5}, 0.001, "DeepInfra")
    led.close()
    return rd


def test_reuse_carries_only_verified_terminal_outcomes_and_never_touches_the_arm(tmp_path):
    from rejudge.terra import reuse as R
    sd, texts, _ = _scope(tmp_path, "agentic")
    rq = J.request_settings(MM, "agentic")
    body = {t["k"]: T.entry_for("agentic", t, rq)["body"] for t in texts}
    k_ok, k_reask, k_cut2, k_badcap, k_differs = (t["k"] for t in texts[:5])
    outside = "e" * 64
    sends = {k_ok: [body[k_ok]], k_reask: [body[k_reask], body[k_reask]],
             k_cut2: [body[k_cut2], dict(body[k_cut2], max_tokens=8000)],            # cut, re-asked at the entry's retry cap, cut again
             k_badcap: [body[k_badcap], dict(body[k_badcap], max_tokens=4000)],      # re-asked at a cap the full run would not send
             k_differs: [dict(body[k_differs], max_tokens=999)],
             outside: [dict(body[k_ok])]}
    outcomes = {k_ok: ["verdict"], k_reask: ["parse_error", "verdict"], k_cut2: ["truncated", "truncated"],
                k_badcap: ["truncated", "verdict"], k_differs: ["verdict"], outside: ["verdict"]}
    arm = _arm(tmp_path, "agentic", texts, sends, outcomes)
    md5 = hashlib.md5((arm / "state.sqlite").read_bytes()).hexdigest()
    rep = R.build(str(arm), str(sd), MM, "agentic", str(tmp_path / "reuse"))
    assert rep["carried"] == 3 and rep["carried_outcomes"] == {"verdict": 2, "truncated": 1}
    assert rep["left_keys"] == {k_badcap: "re_ask_body_differs", k_differs: "original_body_differs", outside: "not_in_scope"}
    assert hashlib.md5((arm / "state.sqlite").read_bytes()).hexdigest() == md5
    assert (tmp_path / "reuse" / "keys.txt").read_text().split() == sorted([k_ok, k_reask, k_cut2])
    led = Ledger(str(tmp_path / "reuse"), readonly=True)
    rows = {r["custom_id"]: r for r in led.iter_requests()}
    assert set(rows) == {k_ok, k_reask, k_cut2} and all(r["terminal"] == 1 for r in rows.values())
    assert rows[k_reask]["attempts"] == 2 and rows[k_reask]["batch_id"] == "batch-1" and rows[k_reask]["provider"] == "DeepInfra"
    assert led.attempt_outcomes(k_cut2) == ["truncated", "truncated"] and not BF.list_manifests(str(tmp_path / "reuse"))
    led.close()
    tr = json.load(open(tmp_path / "reuse" / "track.json"))
    assert tr["reuse_only"] is True and tr["reused_from"]["run_dir"] == str(arm)
    from rejudge.terra import file_records as FRm
    got = FRm.read_outcomes([str(tmp_path / "reuse")])
    assert set(got) == {k_ok, k_reask, k_cut2} and got[k_reask]["verdict"] == {"harmful": True}
    with pytest.raises(SystemExit, match="built once"):
        R.build(str(arm), str(sd), MM, "agentic", str(tmp_path / "reuse"))


def test_reuse_refuses_an_arm_of_another_model_or_still_moving(tmp_path):
    from rejudge.terra import reuse as R
    sd, texts, _ = _scope(tmp_path, "cpi")
    rq = J.request_settings(MM, "cpi")
    t = texts[0]
    arm = _arm(tmp_path, "cpi", texts, {t["k"]: [T.entry_for("cpi", t, rq)["body"]]}, {t["k"]: ["verdict"]}, model="z-ai/glm-5.3")
    with pytest.raises(SystemExit, match="refused: the arm sent"):
        R.build(str(arm), str(sd), MM, "cpi", str(tmp_path / "r1"))
    arm2 = tmp_path / "arm2"
    arm2.mkdir()
    (tmp_path / "arm_cpi").rename(arm2 / "x")
    arm = _arm(tmp_path, "cpi", texts, {t["k"]: [T.entry_for("cpi", t, rq)["body"]]}, {t["k"]: ["parse_error"]})
    led = Ledger(str(arm))
    led.conn.execute("UPDATE requests SET terminal = 0")
    led.conn.commit()
    led.close()
    with pytest.raises(SystemExit, match="still moving"):
        R.build(str(arm), str(sd), MM, "cpi", str(tmp_path / "r2"))


def test_policy_replay_matches_the_track_harvester():
    from rejudge.terra import reuse as R
    ok = lambda outs, cap=None: R.policy_consistent(outs, T.RETRY, cap)  # noqa: E731
    assert ok(["verdict"]) and ok(["parse_error", "verdict"]) and ok(["parse_error", "parse_error"])
    assert ok(["truncated"]) and not ok(["truncated"], 8000)              # no retry cap: a cut call is terminal at once
    assert ok(["truncated", "truncated"], 8000) and ok(["truncated", "verdict"], 8000) and not ok(["truncated", "verdict"])
    assert ok(["provider_error"] * 3 + ["verdict"]) and ok(["provider_error"] * 4) and not ok(["provider_error"] * 3)
    assert ok(["batch_expired", "parse_error", "verdict"]) and ok(["moderation"]) and not ok(["moderation", "verdict"])
    assert not ok(["account_error", "verdict"]) and not ok(["held", "verdict"]) and not ok([])


# --------------------------------------------------------------------------------------------
# the two-run-directory harvester
# --------------------------------------------------------------------------------------------
class _Answers:
    """A Batch API stand-in serving both families: every request answered at once, a C/P/I call or an agentic verdict."""

    def __init__(self, cut=()):
        self.batches, self.cut = {}, set(cut)

    def submit(self, body):
        d = json.loads(body)
        bid = f"batch-{len(self.batches)}"
        res = []
        for r in d["requests"]:
            b = r["body"]
            if "tools" in b:
                msg, fin = {"role": "assistant", "content": None, "tool_calls": [{"type": "function", "function": {
                    "name": "compliance_rating", "arguments": json.dumps({"reasoning": "r", "label": "partial_refusal"})}}]}, "tool_calls"
            elif r["custom_id"] in self.cut and b["max_tokens"] < 8000:
                msg, fin = {"role": "assistant", "content": '{"reasoning": "cut'}, "length"
            else:
                msg, fin = {"role": "assistant", "content": json.dumps({"reasoning": "r", "harmful": False})}, "stop"
            res.append({"custom_id": r["custom_id"], "response": {"status_code": 200, "body": {
                "choices": [{"message": msg, "finish_reason": fin}], "usage": {"prompt_tokens": 100, "completion_tokens": 10},
                "provider": "DeepInfra", "model": "minimax/minimax-m3"}}})
        self.batches[bid] = {"id": bid, "status": "completed", "model": d["model"], "provider": "DeepInfra", "results": res, "usage": {"cost": 0.01},
                             "request_counts": {"total": len(res)}, "max_tokens": sorted({r["body"]["max_tokens"] for r in d["requests"]})}
        return {k: v for k, v in self.batches[bid].items() if k != "results"}

    def iter_batches(self, statuses=None, created_after=None, max_pages=1000):
        for b in self.batches.values():
            yield {k: v for k, v in b.items() if k != "results"}

    def get(self, bid):
        return self.batches[bid]

    def key_info(self):
        return {"limit": 1}


def _two_runs(tmp_path):
    from rejudge.terra import build_manifests as BM
    out = {}
    for fam in J.FAMILIES:
        sd, texts, _ = _scope(tmp_path, fam)
        rd = tmp_path / "full" / fam
        assert BM.main(["--judge", MM, "--family", fam, "--scope", str(sd), "--run-dir", str(rd), "--workers", "1"]) == 0
        out[fam] = (rd, texts)
    return out


def test_harvest_all_drives_both_families_from_one_loop(tmp_path, monkeypatch):
    from rejudge.common import harvester as H
    from rejudge.terra import harvest_all as HA
    runs = _two_runs(tmp_path)
    cut = runs["agentic"][1][0]["k"]
    api = _Answers(cut=[cut])
    monkeypatch.setattr(H, "BatchClient", lambda: api)
    prog = tmp_path / "full" / "progress.json"
    code = HA.main(["--run-dir", str(runs["cpi"][0]), "--run-dir", str(runs["agentic"][0]), "--progress", str(prog), "--poll-s", "0", "--max-hours", "0.05"])
    assert code == 0
    p = json.load(open(prog))
    n = sum(len(t) for _, t in runs.values())
    assert p["done"] is True and p["state"] == "done" and p["exit_code"] == 0 and p["requests"]["total"] == n == p["requests"]["terminal"]
    assert p["requests"]["outcomes"] == {"verdict": n} and set(p["run_dirs"]) == {str(r) for r, _ in runs.values()}
    assert any(b["max_tokens"] == [8000] for b in api.batches.values())       # the cut agentic verdict re-asked at the entry's cap
    led = Ledger(str(runs["agentic"][0]), readonly=True)
    assert led.attempt_outcomes(cut) == ["truncated", "verdict"]
    led.close()


def test_harvest_all_one_paused_run_dir_does_not_stop_the_other(tmp_path, monkeypatch):
    from rejudge.common import harvester as H
    from rejudge.terra import harvest_all as HA
    runs = _two_runs(tmp_path)
    (runs["cpi"][0] / "PAUSED").write_text("2026-11-11T00:00:00Z held for a person\n")
    api = _Answers()
    monkeypatch.setattr(H, "BatchClient", lambda: api)
    prog = tmp_path / "full" / "progress.json"
    code = HA.main(["--run-dir", str(runs["cpi"][0]), "--run-dir", str(runs["agentic"][0]), "--progress", str(prog), "--poll-s", "0", "--max-hours", "0.05"])
    assert code == H.EXIT_PAUSED
    p = json.load(open(prog))
    assert p["state"] == "paused" and list(p["paused"]) == [str(runs["cpi"][0])] and p["done"] is False
    assert p["run_dirs"][str(runs["agentic"][0])]["done"] is True
    assert HA.combined_code([0, 5, 4]) == 4 and HA.combined_code([0, 5]) == 5 and HA.combined_code([0, 0]) == 0


def test_harvest_all_dry_run_needs_no_client_and_writes_the_plan(tmp_path, monkeypatch):
    from rejudge.common import harvester as H
    from rejudge.terra import harvest_all as HA
    runs = _two_runs(tmp_path)

    def no_client():
        raise AssertionError("a dry run must not build a client")
    monkeypatch.setattr(H, "BatchClient", no_client)
    prog = tmp_path / "full" / "progress.json"
    assert HA.main(["--run-dir", str(runs["cpi"][0]), "--run-dir", str(runs["agentic"][0]), "--progress", str(prog), "--dry-run"]) == 0
    p = json.load(open(prog))
    assert p["state"] == "dry_run" and not p["done"]
    for fam, (rd, texts) in runs.items():
        plan = p["plan"][str(rd)]
        assert plan["submit"]["dry_run"] is True and plan["submit"]["requests"] == len(texts) and plan["track"]["judge"] == MM
        assert plan["dry_run"]["cost_bands"]["central"]["output_tokens"] > 0
        assert not os.path.exists(rd / "raw") or not os.listdir(rd / "raw")


# --------------------------------------------------------------------------------------------
# filing under the entry
# --------------------------------------------------------------------------------------------
def test_minimax_filing_names_the_judge_everywhere_and_files_no_headline_metric(tmp_path):
    sd, texts, units = _scope(tmp_path, "cpi")
    rd = tmp_path / "run_cpi"
    led = Ledger(str(rd))
    for t in texts:
        led.record_outcome(t["k"], "b00000", "batch-1", "verdict", True, {"label": "full_refusal"}, None, 200, "tool_calls", None, None, "DeepInfra")
    led.close()
    stage = tmp_path / "stage"
    assert FR.main(["--judge", MM, "--scope-cpi", str(sd), "--run-dir", str(rd), "--staging", str(stage), "--code-sha", "0" * 40, "--dry-run"]) == 0
    rep = json.load(open(stage / "filing_report.json"))
    jid = FR.judge_spec("or_bench_cpi", MM)["judge_id"]
    assert rep["judges"] == {"or_bench_cpi": jid} and rep["judge_model"] == "minimax-m3" and rep["totals"]["runs_complete"] == 2
    ns = stage / "rejudge" / f"cpi_minimaxm3__{jid[:12]}"
    assert ns.is_dir()
    recs = []
    for f in sorted(ns.rglob("*.jsonl*")):
        with (gzip.open(f, "rt") if f.suffix == ".gz" else open(f)) as fh:
            recs += [json.loads(l) for l in fh if l.strip()]
    judge = [r for r in recs if r.get("record_type") == "judge"]
    items = [r for r in recs if r.get("record_type") == "item_result"]
    jm = [r for r in recs if r.get("record_type") == "judgment"]
    assert len(judge) == 1 and judge[0]["payload"]["judge_model"] == "minimax-m3"
    assert len(jm) == 2 and all(r["payload"]["extra"]["judge_model"] == "minimax-m3" for r in jm)
    assert items and all(row["metric_id"] == "orb_label" and row["extra"]["judge_model"] == "minimax-m3"
                         for r in items for row in (r["payload"].get("rows") or [r["payload"]]))


def test_filing_refuses_a_run_dir_still_moving_unless_allowed(tmp_path):
    """A run directory whose harvester has not finished (a batch pending, a request without a terminal outcome) is not staged: its
    judgments would file partial while verdicts are still arriving."""
    sd, texts, _ = _scope(tmp_path, "cpi")
    rd = tmp_path / "moving"
    T.init_run(str(rd), "cpi", judge=MM)
    man = T.add_inputs(str(rd), texts)[0]
    led = Ledger(str(rd))
    led.register_manifest(man)                                   # one batch pending, every request without an outcome
    led.record_outcome(texts[0]["k"], man["batch_key"], "batch-1", "verdict", True, {"label": "full_refusal"}, None, 200, "tool_calls",
                       None, None, "DeepInfra")
    led.close()
    args = ["--judge", MM, "--scope-cpi", str(sd), "--run-dir", str(rd), "--code-sha", "0" * 40, "--dry-run"]
    assert FR.main(args + ["--staging", str(tmp_path / "s1")]) == 2 and not (tmp_path / "s1").exists()
    assert FR.main(args + ["--staging", str(tmp_path / "s2"), "--allow-unfinished"]) == 0
    rep = json.load(open(tmp_path / "s2" / "filing_report.json"))
    assert list(rep["unfinished_run_dirs"]) == [str(rd)] and rep["totals"]["runs_partial"] == 2


def test_a_follow_up_filing_numbers_above_the_registrys_files(tmp_path):
    """Staged with --registry-root (read only), a second filing of the same runs is a follow-up: its files number above what the
    registry already holds for each run's record directory, and the judge record is not staged again."""
    from sisteer.registry import identity
    sd, texts, _ = _scope(tmp_path, "cpi")
    rd = tmp_path / "run_cpi"
    led = Ledger(str(rd))
    for t in texts:
        led.record_outcome(t["k"], "b00000", "batch-1", "verdict", True, {"label": "full_compliance"}, None, 200, "tool_calls", None, None, "DeepInfra")
    led.close()
    jid = FR.judge_spec("or_bench_cpi", MM)["judge_id"]
    reg = tmp_path / "registry"
    first = tmp_path / "first"
    assert FR.main(["--judge", MM, "--scope-cpi", str(sd), "--run-dir", str(rd), "--staging", str(first), "--code-sha", "0" * 40,
                    "--registry-root", str(reg), "--dry-run"]) == 0
    ns = f"rejudge/cpi_minimaxm3__{jid[:12]}"
    (reg / "records").mkdir(parents=True)
    import shutil
    shutil.copytree(first / ns, reg / "records" / ns)           # as if delivered and merged by the harvester
    second = tmp_path / "second"
    assert FR.main(["--judge", MM, "--scope-cpi", str(sd), "--run-dir", str(rd), "--staging", str(second), "--code-sha", "0" * 40,
                    "--registry-root", str(reg), "--dry-run"]) == 0
    assert not (second / ns / "judge").exists()
    run_dir = identity.run_dir_name("run_a")
    assert sorted(os.listdir(first / ns / run_dir)) == ["000001-judgment.jsonl", "000002-item_result.jsonl.gz"]
    assert sorted(os.listdir(second / ns / run_dir)) == ["000003-judgment.jsonl", "000004-item_result.jsonl.gz"]
