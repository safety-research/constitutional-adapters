"""``scripts/sdfbase/record_resolved_sampler.py`` — the sampler the ENGINE resolved, read off
vLLM's own ``Received request <id>: params: SamplingParams(...)`` lines (emitted only under
``--enable-log-requests``) and compared WITHIN REQUEST CLASSES against what each class was sent — and
the ``--enable-log-requests`` opt-in on ``scripts/sdfbase/mt_serving_up.py`` that makes those lines exist.

What is pinned here, in order of why the tool exists:

  1. silence is NEVER agreement: zero params lines is UNREADABLE, not AGREES — unless the server's own
     start-up attestation of the flag is present (FLAG_SET_NO_REQUESTS_YET: nothing to read yet), and a
     flag attested MUTED (log level above INFO) is UNREADABLE again
  2. request CLASSES (job 487306, 2026-09-12): the standup's greedy probes at max_tokens 48 are their
     own class and never a divergence of the eval class; only a WITHIN-class mismatch is DIVERGED; the
     neutral-default eval tuple (the real "never reached the engine" failure) is still caught, apart
     from the probes; an undeclared cap is reported, never folded; a client cap equal to a probe cap
     is REFUSED
  3. every DISTINCT resolved tuple is reported, never averaged into the modal one
  4. a key that landed on vLLM's neutral default is flagged as such
  5. the two stated limits: max_tokens is the class KEY and is tallied apart, never a compared key; a
     key outside SamplingParams.__repr__ is reported as unreadable for that key, never as agreeing
  6. the exit code is recorded, not propagated: 0 for every recorded verdict, --strict opts into a gate;
     --require-wired opts into the strict SUBSET that fails ONLY when the instrument the caller declared was
     never wired (UNREADABLE 3, FLAG_SET_NO_REQUESTS_YET 4) and keeps every finding about the sampler at 0,
     DIVERGED and PROBE_NOT_INSTRUMENTED included; the two flags together are refused (exit 2)
  7. the opt-in is opt-in: without it the standup's launch command is byte-identical, and the shell
     helper forwards it only through F22_STANDUP_EXTRA (never hardcoded on the argv).

The probe producer is ``sisteer.eval.attacks.served_target`` (``_greedy``), not bites_policy.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "sdfbase", REPO / "scripts" / "sdfbase" / "full22"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import record_resolved_sampler as rrs  # noqa: E402
from sisteer.eval.attacks.served_target import PROBE_MAX_TOKENS  # noqa: E402

F22 = REPO / "scripts/sdfbase/full22"

_SENT = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
         "presence_penalty": 1.5, "repetition_penalty": 1.0}
_GREEDY_REF = {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0,
               "presence_penalty": 0.0, "repetition_penalty": 1.0}
# the 0.25.1 request_logger.py:64 INFO line, verbatim shape: params only, no prompt
_SP = ("INFO [request_logger.py:64] Received request chatcmpl-{i}: params: SamplingParams(n=1, "
       "presence_penalty={pp}, frequency_penalty=0.0, repetition_penalty=1.0, temperature={t}, "
       "top_p={tp}, top_k={tk}, min_p=0.0, seed={seed}, stop=[], max_tokens={cap}), lora_request: None.")
_VENDOR = dict(t=0.7, tp=0.8, tk=20, pp=1.5)
_GREEDY = dict(t=0.0, tp=1.0, tk=0, pp=0.0)
_NEUTRAL = dict(t=1.0, tp=1.0, tk=0, pp=0.0)          # what a key that never reached the engine becomes
# vLLM 0.25.1 entrypoints/serve/utils/request_logger.py:21-33, verbatim
_ATTEST_INFO = ("(APIServer pid=1) INFO 09-12 02:05:04 [request_logger.py:28] `--enable-log-requests` "
                "is set but the minimum log level is higher than DEBUG. Only limited information will "
                "be logged to minimize overhead. To view more details, set `VLLM_LOGGING_LEVEL=DEBUG`.")
_ATTEST_MUTED = ("(APIServer pid=1) WARNING 09-12 02:05:04 [request_logger.py:22] `--enable-log-requests` "
                 "is set but the minimum log level is higher than INFO. No request information will be logged.")
_STARTED = "(APIServer pid=1) INFO 09-12 02:05:04 [api_server.py:616] Starting vLLM server"


def _sp(i, cap=512, seed=None, **kw):
    return _SP.format(i=i, cap=cap, seed=(1000 + i if seed is None else seed), **{**_VENDOR, **kw})


def _static_row(i, cap=512, sampler=_SENT):
    return {"goal_id": f"g{i}", "decode": {"do_sample": True, **sampler, "cap": cap, "arm": "x"},
            "max_tokens": cap, "response": "x"}


def _run(tmp_path, lines, *, static_rows=None, controls_rows=None, gen_stats=None,
         expect=None, extra=(), log_name="vllm_server.log"):
    out = tmp_path / "rec.json"
    lg = tmp_path / log_name
    if lines is not None:
        lg.write_text("\n".join(lines) + "\n")
    argv = ["--vllm-log", str(lg), "--out", str(out)]
    if static_rows is not None:
        f = tmp_path / "responses.jsonl"
        f.write_text("\n".join(json.dumps(r) for r in static_rows) + "\n")
        argv += ["--sent-static", str(f)]
    if controls_rows is not None:
        f = tmp_path / "meta.jsonl"
        f.write_text("\n".join(json.dumps(r) for r in controls_rows) + "\n")
        argv += ["--sent-controls", str(f)]
    if gen_stats is not None:
        f = tmp_path / "gen_stats.json"
        f.write_text(json.dumps(gen_stats))
        argv += ["--sent-controls", str(f)]
    for k, v in (expect or {}).items():
        argv += ["--expect", f"{k}={v}"]
    argv += list(extra)
    rc = rrs.main(argv)
    return rc, (json.loads(out.read_text()) if out.exists() else None)


_VENDOR_ROWS = [_static_row(i) for i in range(3)]


# ------------------------------------------------- 1. silence, and the three readings of it
def test_silence_without_the_attestation_is_unreadable_never_agrees(tmp_path):
    rc, rec = _run(tmp_path, [_STARTED], static_rows=_VENDOR_ROWS)
    assert rec["verdict"] == "UNREADABLE" and rec["verdict"] != "AGREES"
    assert "NOT agreement" in rec["why"] and "--enable-log-requests" in rec["why"]
    assert rec["n_resolved_requests"] == 0
    assert rec["logs"][0]["log_requests_flag_attested"] is False
    assert rc == 0, "recorded, not propagated"
    assert _run(tmp_path, [_STARTED], static_rows=_VENDOR_ROWS, extra=["--strict"])[0] == 3


def test_a_missing_log_is_unreadable_and_named(tmp_path):
    rc, rec = _run(tmp_path, None, static_rows=_VENDOR_ROWS, log_name="never_written.log")
    assert rc == 0 and rec["verdict"] == "UNREADABLE"
    assert rec["missing_logs"] == [str(tmp_path / "never_written.log")] and rec["n_logs"] == 0


@pytest.mark.parametrize("attest", [_ATTEST_INFO, _ATTEST_INFO.replace("`", "")])
def test_the_flag_attested_before_any_request_is_not_silence(tmp_path, attest):
    rc, rec = _run(tmp_path, [attest, _STARTED])
    assert rec["verdict"] == "FLAG_SET_NO_REQUESTS_YET", rec["why"]
    assert rec["logs"][0]["log_requests_flag_attested"] is True
    assert rec["logs"][0]["log_requests_muted"] is False
    assert "says nothing about the sampler" in rec["why"]
    assert rc == 0
    assert _run(tmp_path, [attest, _STARTED], extra=["--strict"])[0] == 4


def test_a_muted_attestation_is_unreadable_because_nothing_will_ever_be_logged(tmp_path):
    rc, rec = _run(tmp_path, [_ATTEST_MUTED, _STARTED], static_rows=_VENDOR_ROWS)
    assert rec["verdict"] == "UNREADABLE" and "MUTED" in rec["why"] and "above INFO" in rec["why"]
    assert rec["logs"][0]["log_requests_muted"] is True
    assert rec["logs"][0]["log_requests_flag_attested"] is True


def test_an_attested_server_that_then_served_requests_is_read_normally(tmp_path):
    rc, rec = _run(tmp_path, [_ATTEST_INFO, _STARTED, _sp(0), _sp(1)], static_rows=_VENDOR_ROWS)
    assert rc == 0 and rec["verdict"] == "AGREES" and rec["n_resolved_requests"] == 2


# ------------------------------------------- 2. request classes: the job 487306 lesson
def test_probes_are_their_own_class_and_both_tuples_present_is_agrees(tmp_path):
    """Job 487306: 5,180 eval requests at the vendor row + 180 greedy standup probes at max_tokens 48
    were read as one class and the probes filed as four diverged keys. FAILS on the single-class code."""
    lines = [_sp(i) for i in range(3)]
    lines += [_sp(10 + i, cap=PROBE_MAX_TOKENS, seed=0, **_GREEDY) for i in range(2)]
    rc, rec = _run(tmp_path, lines, static_rows=_VENDOR_ROWS)
    assert rec["verdict"] == "AGREES", rec["why"]
    assert rc == 0
    c, p = rec["classes"]["client"], rec["classes"]["probe"]
    assert c["n"] == 3 and c["verdict"] == "AGREES" and c["distinct_resolved"][0]["sampler"] == _SENT
    assert p["n"] == 2 and p["verdict"] == "AGREES"
    assert p["reference"] == _GREEDY_REF
    assert "served_target" in p["reference_source"] and "temperature=0.0" in p["reference_source"]
    assert rec["divergences"] == []                       # the top-level block is the CLIENT class
    assert rec["probe_caps"] == [str(PROBE_MAX_TOKENS)]
    assert rec["probe_cap_source"] == "sisteer.eval.attacks.served_target.PROBE_MAX_TOKENS"
    assert str(PROBE_MAX_TOKENS) in rec["class_rule"] and "512" in rec["class_rule"]
    assert rec["n_probe_requests"] == 2 and rec["n_unclassified_requests"] == 0


def test_a_probe_that_is_not_greedy_is_a_within_class_divergence(tmp_path):
    lines = [_sp(i) for i in range(3)]
    lines += [_sp(10, cap=PROBE_MAX_TOKENS, seed=0, **{**_GREEDY, "t": 0.7})]
    rc, rec = _run(tmp_path, lines, static_rows=_VENDOR_ROWS)
    assert rec["verdict"] == "DIVERGED" and "probe" in rec["why"]
    assert rec["classes"]["client"]["verdict"] == "AGREES"
    assert [d["key"] for d in rec["classes"]["probe"]["divergences"]] == ["temperature"]
    assert rc == 0, "a finding, not a job failure"
    assert _run(tmp_path, lines, static_rows=_VENDOR_ROWS, extra=["--strict"])[0] == 1


def test_a_within_class_mismatch_in_the_eval_class_is_diverged_with_the_probes_agreeing(tmp_path):
    lines = [_sp(i) for i in range(3)] + [_sp(3, tp=0.95)]           # one eval request off the row
    lines += [_sp(10, cap=PROBE_MAX_TOKENS, seed=0, **_GREEDY)]
    rc, rec = _run(tmp_path, lines, static_rows=_VENDOR_ROWS)
    assert rec["verdict"] == "DIVERGED" and "client" in rec["why"] and "probe" not in rec["why"].split(":")[0]
    assert rec["classes"]["probe"]["verdict"] == "AGREES"
    c = rec["classes"]["client"]
    assert c["verdict"] == "DIVERGED" and len(c["distinct_resolved"]) == 2
    (d,) = c["divergences"]
    assert d["key"] == "top_p" and d["resolved"] == 0.95 and d["n_requests"] == 1
    assert d["equals_vllm_neutral_default"] is False


def test_the_neutral_default_eval_tuple_is_still_caught_apart_from_the_probes(tmp_path):
    """The real failure the tool exists for: the vendor keys never reached the engine, so the eval
    requests resolved to vLLM's neutral defaults. It resembles the greedy probe tuple; the class key
    (max_tokens) keeps the two worlds apart and the flag names it."""
    lines = [_sp(i, **_NEUTRAL) for i in range(3)]
    lines += [_sp(10, cap=PROBE_MAX_TOKENS, seed=0, **_GREEDY)]
    rc, rec = _run(tmp_path, lines, static_rows=_VENDOR_ROWS)
    assert rec["verdict"] == "DIVERGED"
    assert rec["classes"]["probe"]["verdict"] == "AGREES"
    keys = {d["key"]: d for d in rec["classes"]["client"]["divergences"]}
    assert set(keys) == {"temperature", "top_p", "top_k", "presence_penalty"}
    assert all(d["equals_vllm_neutral_default"] for d in keys.values()), (
        "a key that resolved to vLLM's neutral default is the signature of a key that never reached "
        "the engine")


def test_a_greedy_request_at_a_client_cap_is_a_client_divergence_not_a_probe(tmp_path):
    lines = [_sp(i) for i in range(3)] + [_sp(10, seed=0, **_GREEDY)]     # looks like a probe; is not
    rc, rec = _run(tmp_path, lines, static_rows=_VENDOR_ROWS)
    assert rec["verdict"] == "DIVERGED" and rec["classes"]["client"]["verdict"] == "DIVERGED"
    assert rec["classes"]["probe"]["n"] == 0 and rec["classes"]["probe"]["verdict"] == "EMPTY"
    keys = {d["key"]: d for d in rec["classes"]["client"]["divergences"]}
    assert set(keys) == {"temperature", "top_p", "top_k", "presence_penalty"}
    assert keys["temperature"]["equals_vllm_neutral_default"] is False        # 0.0 is not the neutral 1.0
    assert all(keys[k]["equals_vllm_neutral_default"] for k in ("top_p", "top_k", "presence_penalty"))


def test_a_client_cap_equal_to_a_probe_cap_is_refused_with_a_record(tmp_path):
    rows = [_static_row(i, cap=PROBE_MAX_TOKENS) for i in range(2)]
    lines = [_sp(i, cap=PROBE_MAX_TOKENS) for i in range(2)]
    rc, rec = _run(tmp_path, lines, static_rows=rows)
    assert rc == 2 and rec["verdict"] == "REFUSED"
    assert str(PROBE_MAX_TOKENS) in rec["why"] and "cannot be told apart" in rec["why"]
    assert "classes" not in rec


def test_a_request_at_an_undeclared_cap_is_reported_not_folded(tmp_path):
    lines = [_sp(i) for i in range(3)] + [_sp(10, cap=999)] + [_sp(11, cap=300, seed=0, **_GREEDY)]
    rc, rec = _run(tmp_path, lines, static_rows=_VENDOR_ROWS)
    assert rec["verdict"] == "UNCLASSIFIED_REQUESTS" and rc == 0
    assert rec["classes"]["client"]["verdict"] == "AGREES" and rec["classes"]["client"]["n"] == 3
    u = rec["classes"]["unclassified"]
    assert u["n"] == 2 and u["verdict"] == "UNCLASSIFIED"
    by_match = {b["matches_client_reference"]: b for b in u["distinct_resolved"]}
    assert by_match[True]["sampler"] == _SENT            # a clamped client request: same tuple, other cap
    assert by_match[False]["sampler"]["temperature"] == 0.0
    assert "1 of 2 do" in rec["why"]
    assert _run(tmp_path, lines, static_rows=_VENDOR_ROWS, extra=["--strict"])[0] == 3


def test_declared_caps_classify_a_controls_legs_scaffolding(tmp_path):
    """served_capability_run.py fires MMLU scoring greedy at max_tokens 1, the thinking-ON bite probe
    greedy at 256, and a render probe carrying the controls' own sampler at 512; the leg declares them."""
    gs = {"mmlu": {"n_total": 5, "decode": {"do_sample": True, **_SENT, "cap": 2048, "arm": "a",
                                              "seed_base": 1, "trial": 0}}}
    lines = [_sp(i, cap=2048) for i in range(5)] + [_sp(20, cap=512)]
    lines += [_sp(30 + i, cap=1, seed=0, **_GREEDY) for i in range(4)] + [_sp(40, cap=256, seed=0, **_GREEDY)]
    lines += [_sp(50, cap=PROBE_MAX_TOKENS, seed=0, **_GREEDY)]
    extra = ["--probe-cap", str(PROBE_MAX_TOKENS), "--probe-cap", "1", "--probe-cap", "256",
             "--client-cap", "512"]
    rc, rec = _run(tmp_path, lines, gen_stats=gs, extra=extra)
    assert rec["verdict"] == "AGREES", rec["why"]
    assert rec["classes"]["client"]["n"] == 6 and rec["classes"]["probe"]["n"] == 6
    assert rec["n_unclassified_requests"] == 0
    assert rec["probe_cap_source"] == "--probe-cap" and rec["probe_caps"] == ["1", str(PROBE_MAX_TOKENS), "256"]
    assert rec["client_caps"] == ["512", "2048"] and "--client-cap" in rec["client_cap_source"]
    # the same log without the declarations is not wrong, it is loud
    rc2, rec2 = _run(tmp_path, lines, gen_stats=gs)
    assert rec2["verdict"] == "UNCLASSIFIED_REQUESTS" and rec2["n_unclassified_requests"] == 6


def test_with_no_client_cap_known_every_non_probe_request_is_client(tmp_path):
    rows = [{"item_id": f"i{i}", "sampler": _SENT} for i in range(2)]      # a legacy meta row: no cap
    lines = [_sp(i) for i in range(3)] + [_sp(10, cap=PROBE_MAX_TOKENS, seed=0, **_GREEDY)]
    rc, rec = _run(tmp_path, lines, controls_rows=rows)
    assert rec["verdict"] == "AGREES" and rec["classes"]["client"]["n"] == 3 and rec["classes"]["probe"]["n"] == 1
    assert rec["client_caps"] == [] and "ANY non-probe cap" in rec["class_rule"]


def test_a_greedy_controls_leg_sent_temperature_zero_and_nothing_else(tmp_path):
    gs = {"gsm8k": {"n_total": 4, "decode": {"do_sample": False, "cap": 2048, "arm": "greedy"}}}
    lines = [_sp(i, cap=2048, seed=0, **_GREEDY) for i in range(4)]
    rc, rec = _run(tmp_path, lines, gen_stats=gs)
    assert rec["verdict"] == "AGREES", rec["why"]
    assert rec["classes"]["client"]["reference"] == {"temperature": 0.0}
    assert rec["sent"]["distinct_max_tokens"] == [{"max_tokens": "2048", "n": 4}]


# ------------------------------------------------- 3. the comparison within a class
def test_a_distinct_second_sampler_is_reported_not_averaged(tmp_path):
    lines = [_sp(0, pp=1.5), _sp(1, pp=0.0), _sp(2, pp=1.5)]
    rc, rec = _run(tmp_path, lines, static_rows=_VENDOR_ROWS)
    assert rec["verdict"] == "DIVERGED"
    assert len(rec["distinct_resolved"]) == 2, (
        "one control decoding differently from the rest is precisely the defect this record exists to surface")
    assert sorted(b["n"] for b in rec["distinct_resolved"]) == [1, 2]
    (d,) = rec["divergences"]
    assert d["key"] == "presence_penalty" and d["n_requests"] == 1 and d["resolved"] == 0.0


def test_max_tokens_is_the_class_key_and_never_a_compared_key(tmp_path):
    rows = [_static_row(0, cap=512), _static_row(1, cap=300)]        # two declared client caps
    lines = [_sp(0, cap=512), _sp(1, cap=300), _sp(2, cap=512)]
    rc, rec = _run(tmp_path, lines, static_rows=rows)
    assert rec["verdict"] == "AGREES" and rec["divergences"] == []
    assert rec["distinct_max_tokens"] == [{"max_tokens": "512", "n": 2}, {"max_tokens": "300", "n": 1}]
    assert len(rec["distinct_resolved"]) == 1, "a different cap is not a second sampler"


def test_a_key_outside_the_repr_is_reported_unreadable_for_that_key_never_as_agreeing(tmp_path):
    line = ("INFO Received request chatcmpl-0: params: SamplingParams(n=1, repetition_penalty=1.0, "
            "temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, seed=0, max_tokens=512), lora_request: None.")
    rc, rec = _run(tmp_path, [line], static_rows=_VENDOR_ROWS)      # no presence_penalty stated
    assert rec["verdict"] == "DIVERGED"
    (d,) = rec["divergences"]
    assert d["key"] == "presence_penalty" and d["resolved"] is None and d["sent"] == 1.5
    assert "SamplingParams.__repr__" in d["note"]


def test_client_rows_that_disagree_leave_the_class_uncompared_not_agreeing(tmp_path):
    rows = [_static_row(0), _static_row(1, sampler={**_SENT, "top_p": 0.9})]
    rc, rec = _run(tmp_path, [_sp(0)], static_rows=rows)
    assert rec["verdict"] == "UNCOMPARED" and rec["reference"] is None and rc == 0
    assert _run(tmp_path, [_sp(0)], static_rows=rows, extra=["--strict"])[0] == 3


def test_expect_is_the_client_reference_when_given_and_wins_over_the_rows(tmp_path):
    rc, rec = _run(tmp_path, [_sp(0)], static_rows=_VENDOR_ROWS, expect={"top_p": 1.0})
    assert rec["verdict"] == "DIVERGED" and rec["reference_source"] == "--expect"
    (d,) = rec["divergences"]
    assert d["key"] == "top_p" and d["resolved"] == 0.8 and d["sent"] == 1.0


def test_expect_alone_suffices_without_any_client_rows(tmp_path):
    rc, rec = _run(tmp_path, [_sp(0), _sp(1)], expect={"temperature": 0.7, "top_k": 20})
    assert rc == 0 and rec["verdict"] == "AGREES" and rec["sent"]["n_rows"] == 0


def test_expect_outside_the_sampler_keys_is_refused(tmp_path, capsys):
    rc, rec = _run(tmp_path, [_sp(0)], expect={"max_tokens": 512})
    assert rc == 2 and rec is None and "REFUSED" in capsys.readouterr().err


# ------------------------------------------------- 4. the sources
def test_the_generation_config_override_line_is_read_when_present(tmp_path):
    lines = ["WARNING [model.py:1528] Default vLLM sampling parameters have been overridden by the "
             "model's `generation_config.json`: `{'temperature': 1.0, 'top_p': 0.95}`. If this is not intended",
             _sp(0)]
    rc, rec = _run(tmp_path, lines, static_rows=_VENDOR_ROWS)
    assert rc == 0 and rec["verdict"] == "AGREES"
    ov = rec["generation_config_override"][0]
    assert ov["present"] is True and ov["lines"][0]["params"] == "{'temperature': 1.0, 'top_p': 0.95}"


def test_gen_stats_blocks_are_read_weighted_by_n_total_with_their_caps(tmp_path):
    gs = tmp_path / "gen_stats.json"
    gs.write_text(json.dumps({
        "mmlu": {"n_total": 5, "decode": {"do_sample": True, **_SENT, "cap": 2048, "arm": "a", "seed_base": 1, "trial": 0}},
        "gsm8k": {"n_total": 7, "decode": {"do_sample": True, **_SENT, "cap": 2048, "arm": "a", "seed_base": 1, "trial": 0}},
        "xstest": {"n_total": 2, "decode": {"do_sample": False, "cap": 512, "arm": "greedy"}},
        "meta": {"note": "no decode block here"}}))
    sent = rrs.read_sent([gs])
    assert sent["n_rows"] == 14
    assert sent["distinct"] == [{"sampler": _SENT, "n": 12}, {"sampler": {"temperature": 0.0}, "n": 2}]
    assert sent["distinct_max_tokens"] == [{"max_tokens": "2048", "n": 12}, {"max_tokens": "512", "n": 2}]


def test_static_rows_carry_their_decode_block_and_cap_and_junk_is_skipped(tmp_path):
    rows = tmp_path / "responses.jsonl"
    rows.write_text(json.dumps({"goal_id": "g0", "decode": {**_SENT, "max_new_tokens": 512, "cap": 512}}) + "\n"
                    + "not json\n" + json.dumps({"goal_id": "g1", "response": "no decode block"}) + "\n"
                    + json.dumps(["not", "a", "row"]) + "\n")
    sent = rrs.read_sent([rows])
    assert sent["n_rows"] == 1 and sent["distinct"] == [{"sampler": _SENT, "n": 1}]
    assert sent["distinct_max_tokens"] == [{"max_tokens": "512", "n": 1}]


def test_the_params_parser_keeps_a_non_literal_as_text_and_skips_unknown_keys():
    p = rrs._parse_params("n=1, temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, "
                          "stop=['a, b=c'], logits_processors=<fn>, seed=3, max_tokens=64")
    assert p == {"n": 1, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
                 "seed": 3, "max_tokens": 64}


def test_an_empty_log_beside_a_populated_one_still_reads_the_populated_one(tmp_path):
    (tmp_path / "a.log").write_text(_sp(0) + "\n")
    (tmp_path / "b.log").write_text(_STARTED + "\n")
    rows = tmp_path / "responses.jsonl"
    rows.write_text(json.dumps(_static_row(0)) + "\n")
    out = tmp_path / "rec.json"
    rc = rrs.main(["--vllm-log", str(tmp_path / "a.log"), "--vllm-log", str(tmp_path / "b.log"),
                   "--sent-static", str(rows), "--out", str(out)])
    rec = json.loads(out.read_text())
    assert rc == 0 and rec["verdict"] == "AGREES" and rec["n_logs"] == 2
    assert [lg["n_resolved_lines"] for lg in rec["logs"]] == [1, 0]


# ------------------------------------------------- 5. the exit policy, stated in the record
def test_the_exit_policy_is_recorded_and_names_the_active_gate(tmp_path):
    rc, rec = _run(tmp_path, [_sp(0)], static_rows=_VENDOR_ROWS)
    assert rc == 0 and rec["exit_policy"].startswith("recorded, not propagated")
    rc, rec = _run(tmp_path, [_sp(0)], static_rows=_VENDOR_ROWS, extra=["--strict"])
    assert rc == 0 and rec["exit_policy"].startswith("--strict")
    assert rrs._EXIT_STRICT == {"AGREES": 0, "DIVERGED": 1, "UNREADABLE": 3, "UNCOMPARED": 3,
                                "UNCLASSIFIED_REQUESTS": 3, "FLAG_SET_NO_REQUESTS_YET": 4,
                                "PROBE_NOT_INSTRUMENTED": 0}     # the zero-tag remainder is descriptive, never a failure


# ------------------------------------------------- 6. the opt-in on the standup
def test_enable_log_requests_is_opt_in_and_the_launch_command_is_byte_identical_without_it():
    import mt_serving_up as m

    base = m.launch_cmd_extras("0.9")
    assert m.launch_cmd_extras("0.9", None, False) == base
    assert m.launch_cmd_extras("0.9", enable_log_requests=False) == base
    assert m.launch_cmd_extras("0.9", enable_log_requests=True) == (*base, "--enable-log-requests")
    assert m.launch_cmd_extras("0.9", 64, True) == (
        "--gpu-memory-utilization", "0.9", "--max-logprobs", "64", "--enable-log-requests")
    src = (REPO / "scripts/sdfbase/mt_serving_up.py").read_text()
    assert 'ap.add_argument("--enable-log-requests", action="store_true"' in src
    assert ("cmd=tuple(launch.cmd) + launch_cmd_extras(args.gpu_mem_util, args.max_logprobs, "
            "args.enable_log_requests)") in src
    # recorded beside max_logprobs in BOTH cells_served.json and endpoint.json, so an UNREADABLE
    # verdict is explainable from the standup's own record
    assert src.count('"enable_log_requests": args.enable_log_requests,') == 2
    assert "prompt text is a separate DEBUG record" in src
    assert "OUT of ``src/sisteer/capeval/serving/``" in src


def test_the_shell_helper_forwards_the_opt_in_only_through_f22_standup_extra():
    common = (F22 / "common.sh").read_text()
    # the verbatim forward is still there, and $7 is still --max-logprobs (no positional was added)
    assert '${F22_STANDUP_EXTRA[@]+"${F22_STANDUP_EXTRA[@]}"}' in common
    assert 'gLP="${7:-}"' in common and '${gLP:+--max-logprobs "$gLP"}' in common
    mentions = [ln for ln in common.splitlines() if "--enable-log-requests" in ln]
    assert mentions, "the helper's header must name the opt-in and how it is forwarded"
    # never HARDCODED on the standup argv: every non-comment mention is the F22_LOG_REQUESTS scalar
    # opt-in beside F22_STANDUP_EXTRA (a scalar rides through
    # sbatch where a Bash array does not) -- default OFF (`${F22_LOG_REQUESTS:-0}` must equal 1), any
    # other value refused before the standup -- so the command line stays byte-identical unless a lane
    # opts in through one of the two doors
    wired = [ln for ln in mentions if not ln.strip().startswith("#")]
    assert wired and all("F22_LOG_REQUESTS" in ln for ln in wired), (
        "--enable-log-requests must never be hardcoded on the standup argv: the default is OFF and "
        "the command line stays byte-identical unless a lane opts in (F22_STANDUP_EXTRA or "
        f"F22_LOG_REQUESTS=1); found {wired}")
    assert '[ "${F22_LOG_REQUESTS:-0}" = 1 ] && log_requests_flag="--enable-log-requests"' in common
    assert 'case "${F22_LOG_REQUESTS:-}" in ""|0|1) ;; *)' in common      # 1, 0 or unset; anything else refuses
    assert '${log_requests_flag:+"$log_requests_flag"}' in common         # the only path onto the argv
    assert "record_resolved_sampler.py" in common


def test_the_readback_lives_outside_the_frozen_serving_tree():
    assert (REPO / "scripts/sdfbase/record_resolved_sampler.py").is_file()
    frozen = REPO / "src/sisteer/capeval/serving"
    assert not any(p.name == "record_resolved_sampler.py" for p in frozen.rglob("*.py"))
    assert not any("enable-log-requests" in p.read_text(errors="replace")
                   for p in frozen.rglob("*.py")), (
        "the opt-in is spliced by the standup, never by the frozen serving tree")


# ------------------------------------------------- 7. the probe class's reference: --probe-expect {greedy,card}
# The GSM8K driver's no-switch render probe decodes under the row's CARD (0.6 / 0.9) by design, so on the
# Llama wave the greedy reference filed the instrument's own probe as a divergence on every leg while the
# client class agreed. The recorder admits a probe class whose sampler equals
# the PINNED card row — the reference is the launcher's --expect values (the same the client class is checked
# against), NEVER the driver's runtime declaration of what it decoded under; the record says what it compared
# against; `card` without --expect is refused.
_CARD = dict(t=0.6, tp=0.9, tk=0, pp=0.0)
_CARD_EXPECT = {"temperature": 0.6, "top_p": 0.9}
_CARD_CLIENT = [_sp(i, cap=8192, **_CARD) for i in range(3)]
_CARD_ARGS = ["--client-cap", "8192"]


def _probe_lines(cap=PROBE_MAX_TOKENS, n=2, **tuple_):
    return [_sp(50 + i, cap=cap, seed=0, **tuple_) for i in range(n)]


def test_card_mode_agrees_when_the_probe_class_carries_the_pinned_card_row(tmp_path):
    lines = _CARD_CLIENT + _probe_lines(**_CARD)
    rc, rec = _run(tmp_path, lines, expect=_CARD_EXPECT, extra=[*_CARD_ARGS, "--probe-expect", "card"])
    assert rc == 0 and rec["verdict"] == "AGREES", rec["why"]
    p = rec["classes"]["probe"]
    assert p["n"] == 2 and p["verdict"] == "AGREES" and p["divergences"] == []
    assert p["reference"] == _CARD_EXPECT and p["reference_kind"] == "card"
    assert "--expect" in p["reference_source"] and "pinned card row" in p["reference_source"]
    assert rec["probe_reference"] == {"kind": "card", "expect": _CARD_EXPECT, "source": rrs.PROBE_CARD_REFERENCE_SOURCE}
    assert "the pinned card row" in rec["why"] and "--probe-expect card" in rec["why"]
    c = rec["classes"]["client"]
    assert c["verdict"] == "AGREES" and c["reference"] == _CARD_EXPECT and c["reference_kind"] == "expect"
    assert rec["n_client_requests"] == 3 and rec["n_probe_requests"] == 2, "card does NOT fold the probes into the client class"
    # the SAME log under the default reference is today's Llama-wave reading: DIVERGED in the probe class only
    rc2, rec2 = _run(tmp_path, lines, expect=_CARD_EXPECT, extra=_CARD_ARGS)
    assert rec2["verdict"] == "DIVERGED" and rec2["classes"]["client"]["verdict"] == "AGREES"
    assert rec2["classes"]["probe"]["verdict"] == "DIVERGED" and rec2["probe_reference"]["kind"] == "greedy"
    assert {d["key"] for d in rec2["classes"]["probe"]["divergences"]} == {"temperature", "top_p"}


def test_card_mode_diverges_when_the_probe_class_carries_greedy(tmp_path):
    lines = _CARD_CLIENT + _probe_lines(**_GREEDY)
    rc, rec = _run(tmp_path, lines, expect=_CARD_EXPECT, extra=[*_CARD_ARGS, "--probe-expect", "card"])
    assert rc == 0 and rec["verdict"] == "DIVERGED" and "probe" in rec["why"]
    assert rec["classes"]["client"]["verdict"] == "AGREES"
    p = rec["classes"]["probe"]
    assert p["verdict"] == "DIVERGED" and p["reference_kind"] == "card" and p["reference"] == _CARD_EXPECT
    keys = {d["key"]: d for d in p["divergences"]}
    assert set(keys) == {"temperature", "top_p"}
    assert keys["temperature"]["sent"] == 0.6 and keys["temperature"]["resolved"] == 0.0
    assert keys["top_p"]["sent"] == 0.9 and keys["top_p"]["resolved"] == 1.0 and keys["top_p"]["equals_vllm_neutral_default"]
    assert rrs.PROBE_CARD_REFERENCE_SOURCE in rec["why"], "the verdict names what the probe class was compared against"
    assert _run(tmp_path, lines, expect=_CARD_EXPECT, extra=[*_CARD_ARGS, "--probe-expect", "card", "--strict"])[0] == 1
    # the mirror: the same greedy probes AGREE under the default reference
    assert _run(tmp_path, lines, expect=_CARD_EXPECT, extra=_CARD_ARGS)[1]["verdict"] == "AGREES"


def test_card_mode_with_a_mixed_probe_class_diverges_on_the_greedy_tuple_and_shows_the_split(tmp_path):
    """The Llama wave's declaration (--probe-cap 48 512 2048) puts the driver's card-row render probe (512)
    and its greedy bites probes (48) in ONE class; `card` compares that class against the card row, so the
    greedy tuple is the divergence now and the record shows which tuple, at which cap, with how many requests."""
    lines = _CARD_CLIENT + _probe_lines(cap=48, n=10, **_GREEDY) + _probe_lines(cap=512, n=1, **_CARD)
    extra = [*_CARD_ARGS, "--probe-cap", "48", "--probe-cap", "512", "--probe-cap", "2048", "--probe-expect", "card"]
    rc, rec = _run(tmp_path, lines, expect=_CARD_EXPECT, extra=extra)
    assert rec["verdict"] == "DIVERGED" and rec["classes"]["client"]["verdict"] == "AGREES"
    p = rec["classes"]["probe"]
    assert p["n"] == 11 and p["verdict"] == "DIVERGED" and p["reference_kind"] == "card"
    assert sorted(b["n"] for b in p["distinct_resolved"]) == [1, 10]
    assert all(d["n_requests"] == 10 for d in p["divergences"]), "only the greedy tuple diverges from the card row"
    assert p["distinct_max_tokens"] == [{"max_tokens": "48", "n": 10}, {"max_tokens": "512", "n": 1}]


def test_card_mode_without_any_expect_is_refused_even_with_agreeing_client_rows(tmp_path, capsys):
    rows = [_static_row(i, cap=8192, sampler={**_CARD_EXPECT, "top_k": 0}) for i in range(2)]
    lines = _CARD_CLIENT + _probe_lines(**_CARD)
    rc, rec = _run(tmp_path, lines, static_rows=rows, extra=["--probe-expect", "card"])
    assert rc == 2 and rec is None
    err = capsys.readouterr().err
    assert "REFUSED" in err and "--probe-expect card" in err and "--expect" in err
    assert "never the driver" in err
    # the rows alone DO serve the client class under the default — the refusal is card-specific
    rc2, rec2 = _run(tmp_path, lines, static_rows=rows)
    assert rc2 == 0 and rec2["classes"]["client"]["verdict"] == "AGREES" and rec2["classes"]["client"]["reference_kind"] == "rows"


def test_greedy_is_the_default_and_the_record_names_the_probe_reference_either_way(tmp_path):
    lines = [_sp(i) for i in range(3)] + _probe_lines(**_GREEDY)
    rc, rec = _run(tmp_path, lines, static_rows=_VENDOR_ROWS)
    assert rec["verdict"] == "AGREES"
    assert rec["probe_reference"] == {"kind": "greedy", "expect": _GREEDY_REF, "source": rrs.PROBE_REFERENCE_SOURCE}
    assert rec["classes"]["probe"]["reference_kind"] == "greedy" and rec["classes"]["client"]["reference_kind"] == "rows"
    assert rec["classes"]["unclassified"]["reference_kind"] is None
    assert rrs.PROBE_EXPECT_CHOICES == ("greedy", "card", "kinds")
    # an explicit --probe-expect greedy is the default, byte for byte
    _rc, explicit = _run(tmp_path, lines, static_rows=_VENDOR_ROWS, extra=["--probe-expect", "greedy"])
    assert json.dumps(explicit, sort_keys=True) == json.dumps(rec, sort_keys=True)
    # the kinds-mode keys never appear outside kinds mode: greedy and card records keep their shape
    for extra in ([], ["--probe-expect", "greedy"]):
        _rc, r = _run(tmp_path, lines, static_rows=_VENDOR_ROWS, extra=extra)
        assert not ({"probe_tags", "n_probe_requests_by_kind", "probe_verdict_by_kind"} & set(r))
        assert "kinds" not in r["classes"]["probe"] and "kinds" not in r["probe_reference"]


# ------------------------------------------------- 8. the probe class by KIND: --probe-expect kinds + --probe-tags
# The ruling on the Llama 17-leg re-read (decisions-recorder R1-R5): the wave's probe class is MIXED — 10 greedy
# steering-bites probes at cap 48 + ONE render probe at cap 512 decoding under the card row — so ONE reference reads one
# half as a divergence whichever is chosen. Probes are therefore classified by KIND, each kind against ITS OWN pinned
# reference (render -> the pinned card row = --expect; bites / lens_on_bite -> greedy at their pinned caps). THE TAG
# CLASSIFIES: the driver stamps every probe attempt `X-Tinycap-Task: probe/<kind>` and writes it with the attempt's
# max_tokens into request_evidence.jsonl (contracts.REQUEST_EVIDENCE_FIELDS) — read back through --probe-tags. THE
# SHAPE CROSS-CHECKS and must agree (the tagged caps are the kind's pinned caps; per kind the tagged and logged cap
# multisets are equal), else the probe is UNCLASSIFIED: a third outcome that FAILS the probe class, never AGREES, never a
# DIVERGED against a guess. vLLM mints its own request ids and the proxy forwards no client id, so the join is per kind
# by cap multiset, never per line — stated in the record.
_KINDS_ARGS = [*_CARD_ARGS, "--probe-expect", "kinds"]
_CELL, _TWIN = "petri__si_x__lora", "none__lora0"


def _tag_row(kind, cap, vmodel=_CELL, n=0):
    """One request_evidence.jsonl record (contracts.request_evidence_record's shape) of a labelled probe attempt."""
    return {"request_evidence_version": 2, "ts": "2026-09-12T00:00:00Z", "kind": "probe", "task": f"probe/{kind}",
            "sample": f"{vmodel}/{n}", "vmodel": vmodel, "petri_thinking_on_present": False,
            "petri_thinking_on_value": None, "enable_thinking_present": False, "enable_thinking_value": None,
            "body_sha256": "0" * 64, "request_id": f"{kind}{n}".ljust(32, "0"), "body_bytes": 181, "max_tokens": cap,
            "attempt_id": "a1"}


def _task_row(i, cap=8192):
    return {**_tag_row("x", cap, vmodel=_CELL, n=i), "kind": "task", "task": "inspect/gsm8k_paired200", "sample": f"g{i}/0"}


def _bites_tags(n=10, cap=PROBE_MAX_TOKENS):
    """5 prompts on the cell + 5 on its scale-0 twin, as _RequestEvidence.labelled numbers them (per kind and route)."""
    return [_tag_row("bites", cap, vmodel=(_CELL if i < n // 2 else _TWIN), n=i % max(n // 2, 1)) for i in range(n)]


_LLAMA_LOG = _CARD_CLIENT + _probe_lines(cap=PROBE_MAX_TOKENS, n=10, **_GREEDY) + _probe_lines(cap=512, n=1, **_CARD)
_LLAMA_TAGS = _bites_tags() + [_tag_row("render", 512)]


def _run_kinds(tmp_path, lines, tag_rows, *, expect=_CARD_EXPECT, extra=(), tags_name="request_evidence.jsonl",
               write_tags=True, ledger=False):
    f = tmp_path / tags_name
    if write_tags:
        rows = tag_rows
        if ledger:      # the usage ledger's spelling of the same tag: task + max_tokens_requested (+ rejected)
            rows = [{"task": r["task"], "sample": r["sample"], "vmodel": r["vmodel"], "status": 200,
                     "rejected": bool(r.get("rejected", False)), "max_tokens_requested": r["max_tokens"],
                     "request_id": r["request_id"], "body_sha256": r["body_sha256"]} for r in tag_rows]
        f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return _run(tmp_path, lines, expect=expect, extra=[*_KINDS_ARGS, "--probe-tags", str(f), *extra])


def test_kinds_mode_classifies_the_mixed_llama_probe_class_by_tag_and_agrees(tmp_path):
    """The 17-leg shape: 3 client draws at the card row, 10 greedy bites at 48, ONE card-row render probe at 512 — the
    class that read DIVERGED under `greedy` (the render probe) and under `card` (the bites) AGREES by kind."""
    rc, rec = _run_kinds(tmp_path, _LLAMA_LOG, _LLAMA_TAGS, extra=["--probe-cap", "48", "--probe-cap", "512", "--probe-cap", "2048"])
    assert rc == 0 and rec["verdict"] == "AGREES", rec["why"]
    p = rec["classes"]["probe"]
    assert p["n"] == 11 and p["verdict"] == "AGREES" and p["reference_kind"] == "kinds" and p["reference"] is None
    assert p["n_by_kind"] == {"render": 1, "bites": 10, "lens_on_bite": 0, "logprob_bites": 0, "load_check": 0, "positive_control": 0, "standup_controls": 0} and p["n_unclassified"] == 0
    assert p["n_unclassified_tags"] == 0 and p["unclassified"]["n"] == 0 and p["divergences"] == []
    b, r, lo = p["kinds"]["bites"], p["kinds"]["render"], p["kinds"]["lens_on_bite"]
    assert b["verdict"] == "AGREES" and b["n"] == 10 and b["n_tagged"] == 10 and b["reference"] == _GREEDY_REF
    assert b["reference_kind"] == "greedy" and "served_target.probe_fn" in b["reference_source"] and b["caps_pinned"] == [48]
    assert b["tag_check"] == {"agrees": True, "tagged": {"48": 10}, "logged": {"48": 10}, "problems": []}
    assert r["verdict"] == "AGREES" and r["n"] == 1 and r["n_tagged"] == 1 and r["reference"] == _CARD_EXPECT
    assert r["reference_kind"] == "card" and "_render_probe" in r["reference_source"] and r["caps_pinned"] == [512, 2048]
    assert "never the driver's own sampling_row" in r["reference_source"]
    assert lo["verdict"] == "EMPTY" and lo["n"] == 0 and lo["n_tagged"] == 0 and lo["reference_kind"] is None
    # the ruling's counts, at the top level, and a verdict per kind
    assert rec["n_probe_requests_by_kind"] == {"render": 1, "bites": 10, "lens_on_bite": 0, "logprob_bites": 0, "load_check": 0, "positive_control": 0, "standup_controls": 0}
    assert rec["n_probe_unclassified_requests"] == 0 and rec["n_probe_unclassified_tags"] == 0
    assert rec["probe_verdict_by_kind"] == {"render": "AGREES", "bites": "AGREES", "lens_on_bite": "EMPTY", "logprob_bites": "EMPTY",
                                            "load_check": "EMPTY", "positive_control": "EMPTY", "standup_controls": "EMPTY"}
    assert rec["probe_remainder_verdict"] == "EMPTY" and rec["n_probe_remainder_requests"] == 0 and rec["n_carved_from_client"] == {}
    assert rec["n_probe_requests"] == 11 and rec["n_unclassified_requests"] == 0 and rec["n_client_requests"] == 3
    # the record says what each kind was compared against, and where the tag came from
    pr = rec["probe_reference"]
    assert pr["kind"] == "kinds" and pr["expect"] == _CARD_EXPECT and pr["source"] == rrs.PROBE_KINDS_REFERENCE_SOURCE
    assert pr["kinds"]["render"] == {"caps": [512, 2048], "reference_kind": "card", "reference": _CARD_EXPECT,
                                     "source": rrs.PROBE_KIND_TABLE["render"]["source"]}
    assert pr["kinds"]["bites"]["reference"] == _GREEDY_REF and pr["kinds"]["lens_on_bite"]["caps"] == [256]
    t = rec["probe_tags"]
    assert t["sources"] == [str(tmp_path / "request_evidence.jsonl")] and t["missing"] == [] and "rows" not in t
    assert t["n_rows"] == 11 and t["n_probe_rows"] == 11 and t["n_rejected_skipped"] == 0
    assert t["by_kind"] == {"bites": [{"max_tokens": "48", "n": 10}], "render": [{"max_tokens": "512", "n": 1}]}
    assert "X-Tinycap-Task" in t["tag_field"] and t["join"] == rrs.PROBE_TAG_JOIN and "chatcmpl-<hex>" in t["join"]
    assert "by kind" in rec["why"] and "render 1 = the pinned card row" in rec["why"] and "bites 10 = greedy" in rec["why"]
    assert rec["probe_cap_source"] == "--probe-cap" and rec["probe_caps"] == ["48", "512", "2048"]
    # the client class is untouched by the mode
    c = rec["classes"]["client"]
    assert c["verdict"] == "AGREES" and c["reference"] == _CARD_EXPECT and c["reference_kind"] == "expect"
    # the SAME log under the two single-reference modes: DIVERGED either way — the reason the kinds exist
    assert _run(tmp_path, _LLAMA_LOG, expect=_CARD_EXPECT, extra=[*_CARD_ARGS, "--probe-cap", "48", "--probe-cap", "512"])[1]["verdict"] == "DIVERGED"
    assert _run(tmp_path, _LLAMA_LOG, expect=_CARD_EXPECT, extra=[*_CARD_ARGS, "--probe-cap", "48", "--probe-cap", "512", "--probe-expect", "card"])[1]["verdict"] == "DIVERGED"


def test_kinds_mode_compares_each_kind_against_its_own_pinned_reference(tmp_path):
    # the bites carry the CARD tuple: a divergence of kind bites (against greedy), the render probe still agrees
    lines = _CARD_CLIENT + _probe_lines(cap=48, n=10, **_CARD) + _probe_lines(cap=512, n=1, **_CARD)
    rc, rec = _run_kinds(tmp_path, lines, _LLAMA_TAGS)
    assert rc == 0 and rec["verdict"] == "DIVERGED" and "kind bites" in rec["why"] and "kind render" not in rec["why"]
    p = rec["classes"]["probe"]
    assert p["verdict"] == "DIVERGED" and p["kinds"]["bites"]["verdict"] == "DIVERGED" and p["kinds"]["render"]["verdict"] == "AGREES"
    keys = {d["key"]: d for d in p["kinds"]["bites"]["divergences"]}
    assert set(keys) == {"temperature", "top_p"} and all(d["kind"] == "bites" for d in keys.values())
    assert keys["temperature"]["sent"] == 0.0 and keys["temperature"]["resolved"] == 0.6
    assert all(d["kind"] == "bites" for d in p["divergences"]) and len(p["divergences"]) == 2
    assert rec["probe_verdict_by_kind"] == {"render": "AGREES", "bites": "DIVERGED", "lens_on_bite": "EMPTY", "logprob_bites": "EMPTY",
                                            "load_check": "EMPTY", "positive_control": "EMPTY", "standup_controls": "EMPTY"}
    assert rec["divergences"] == [], "the top-level divergences are the CLIENT class's"
    assert _run_kinds(tmp_path, lines, _LLAMA_TAGS, extra=["--strict"])[0] == 1
    # the render probe GREEDY: a divergence of kind render (against the pinned card row), the bites agree
    lines = _CARD_CLIENT + _probe_lines(cap=48, n=10, **_GREEDY) + _probe_lines(cap=512, n=1, **_GREEDY)
    rc, rec = _run_kinds(tmp_path, lines, _LLAMA_TAGS)
    assert rec["verdict"] == "DIVERGED" and "kind render" in rec["why"] and "kind bites" not in rec["why"]
    r = rec["classes"]["probe"]["kinds"]["render"]
    assert r["verdict"] == "DIVERGED" and {d["key"] for d in r["divergences"]} == {"temperature", "top_p"}
    assert {d["kind"] for d in r["divergences"]} == {"render"}
    assert next(d for d in r["divergences"] if d["key"] == "top_p")["equals_vllm_neutral_default"] is True
    assert rec["classes"]["probe"]["kinds"]["bites"]["verdict"] == "AGREES"


def test_kinds_mode_a_zero_tag_run_is_not_instrumented_and_a_partially_tagged_run_is_unclassified(tmp_path):
    """Two readings of a probe request no tag names, decided by a count the recorder already has: ZERO tagged requests
    in the whole run = a driver that cannot tag (a frozen sha) -> the remainder is NOT_INSTRUMENTED (descriptive, exit 0,
    compared against NOTHING); SOME requests tagged and these not -> UNCLASSIFIED (a genuine gap). Never classified by
    shape alone either way — the shape is a cross-check, never the classifier."""
    # (a) the tag file does not exist at all: zero tags
    rc, rec = _run_kinds(tmp_path, _LLAMA_LOG, _LLAMA_TAGS, write_tags=False)
    assert rc == 0 and rec["verdict"] == "PROBE_NOT_INSTRUMENTED", rec["why"]
    p = rec["classes"]["probe"]
    assert p["verdict"] == "NOT_INSTRUMENTED" and p["n"] == 11 and p["n_unclassified"] == 0 and p["unclassified"]["n"] == 0
    assert p["n_by_kind"] == {"render": 0, "bites": 0, "lens_on_bite": 0, "logprob_bites": 0, "load_check": 0, "positive_control": 0, "standup_controls": 0}
    assert all(p["kinds"][k]["verdict"] == "EMPTY" for k in rrs.PROBE_KINDS)
    r = p["remainder"]
    assert r["verdict"] == "NOT_INSTRUMENTED" and r["n"] == 11 and r["reason"] == "untagged_driver"
    assert r["consequence"] == "probe-class attestation unavailable on this driver; client class unaffected"
    assert r["compared_against"] is None, "descriptive only: no shape-based fallback, no verdict at lower assurance"
    assert r["shape_counts"] == [{"max_tokens": "48", "shape": "greedy", "n": 10}, {"max_tokens": "512", "shape": "sampled", "n": 1}]
    assert sorted(b["n"] for b in r["distinct_resolved"]) == [1, 10] and r["tag_source"]["missing"] == [str(tmp_path / "request_evidence.jsonl")]
    assert rec["probe_tags"]["missing"] == [str(tmp_path / "request_evidence.jsonl")] and rec["probe_tags"]["sources"] == []
    assert rec["probe_tags"]["named"] is True
    assert rec["n_unclassified_requests"] == 0 and rec["probe_remainder_verdict"] == "NOT_INSTRUMENTED" and rec["n_probe_remainder_requests"] == 11
    assert rec["probe_verdict_by_kind"] == {k: "EMPTY" for k in rrs.PROBE_KINDS}
    assert "probe-class attestation unavailable on this driver; client class unaffected" in rec["why"]
    assert "compared against nothing" in rec["why"] and "untagged_driver" in rec["why"] and "tag source missing" in rec["why"]
    assert rec["classes"]["client"]["verdict"] == "AGREES" and rec["n_client_requests"] == 3, "the client class is unaffected"
    assert _run_kinds(tmp_path, _LLAMA_LOG, _LLAMA_TAGS, write_tags=False, extra=["--strict"])[0] == 0, "exit-0 side in BOTH modes"
    # (b) the file exists but carries only Inspect TASK rows — no probe tag at all: the same zero-tag reading
    rc, rec = _run_kinds(tmp_path, _LLAMA_LOG, [_task_row(i) for i in range(3)])
    assert rec["verdict"] == "PROBE_NOT_INSTRUMENTED" and rec["classes"]["probe"]["remainder"]["n"] == 11
    assert rec["probe_tags"]["n_rows"] == 3 and rec["probe_tags"]["n_probe_rows"] == 0 and rec["probe_tags"]["by_kind"] == {}
    assert "carries no probe/ row" in rec["why"]
    # (c) the render probe is tagged, the bites are not: SOME tags -> the 10 bites are UNCLASSIFIED, a genuine gap
    rc, rec = _run_kinds(tmp_path, _LLAMA_LOG, [_tag_row("render", 512)])
    assert rec["verdict"] == "UNCLASSIFIED_REQUESTS", rec["why"]
    p = rec["classes"]["probe"]
    assert p["verdict"] == "UNCLASSIFIED" and p["kinds"]["render"]["verdict"] == "AGREES" and p["kinds"]["bites"]["verdict"] == "EMPTY"
    assert p["n_unclassified"] == 10 and p["unclassified"]["reasons"] == [{"reason": "no tag or declaration names max_tokens 48", "n": 10}]
    assert (p["remainder"]["verdict"], p["remainder"]["n"], p["remainder"]["reason"], p["remainder"]["consequence"]) == (
        "UNCLASSIFIED", 10, "no tag or declaration names these requests", None)
    assert p["n_by_kind"]["render"] == 1 and "matched no kind" in rec["why"] and "NOT_INSTRUMENTED" not in rec["why"]
    assert _run_kinds(tmp_path, _LLAMA_LOG, [_tag_row("render", 512)], extra=["--strict"])[0] == 3
    # (d) one untagged request among tagged ones (11 logged at 48, 10 tagged): the bites kind is UNCLASSIFIED by count,
    # never NOT_INSTRUMENTED — the driver tags, so a missing tag is a gap
    rc, rec = _run_kinds(tmp_path, _LLAMA_LOG + _probe_lines(cap=48, n=1, **_GREEDY), _LLAMA_TAGS)
    assert rec["verdict"] == "UNCLASSIFIED_REQUESTS" and rec["classes"]["probe"]["kinds"]["bites"]["verdict"] == "UNCLASSIFIED"
    assert rec["classes"]["probe"]["remainder"]["verdict"] == "EMPTY" and rec["n_unclassified_requests"] == 11


def test_kinds_mode_a_tag_shape_disagreement_is_unclassified(tmp_path):
    # (a) COUNT: the tag says 10 bites at 48, the log has 9 -> the bites kind is not classified; the render kind still is
    lines = _CARD_CLIENT + _probe_lines(cap=48, n=9, **_GREEDY) + _probe_lines(cap=512, n=1, **_CARD)
    rc, rec = _run_kinds(tmp_path, lines, _LLAMA_TAGS)
    assert rc == 0 and rec["verdict"] == "UNCLASSIFIED_REQUESTS", rec["why"]
    p = rec["classes"]["probe"]
    b = p["kinds"]["bites"]
    assert b["verdict"] == "UNCLASSIFIED" and b["n"] == 9 and b["n_tagged"] == 10 and b["reference"] is None
    assert b["reference_kind"] is None and b["reference_source"] is None and b["divergences"] == []
    assert b["tag_check"]["agrees"] is False and b["tag_check"]["tagged"] == {"48": 10} and b["tag_check"]["logged"] == {"48": 9}
    assert "disagree in count" in b["tag_check"]["problems"][0]
    assert p["kinds"]["render"]["verdict"] == "AGREES"
    assert p["n_unclassified"] == 9 and p["n_unclassified_tags"] == 10 and p["unclassified"]["distinct_max_tokens"] == [{"max_tokens": "48", "n": 9}]
    assert rec["n_unclassified_requests"] == 9 and rec["n_probe_unclassified_tags"] == 10
    assert rec["probe_verdict_by_kind"]["bites"] == "UNCLASSIFIED"
    # (b) CAP: the tag says render at 300 and the log has a render-looking line at 300 — the render code sends 512 / 2048
    lines = _CARD_CLIENT + _probe_lines(cap=48, n=10, **_GREEDY) + _probe_lines(cap=300, n=1, **_CARD)
    rc, rec = _run_kinds(tmp_path, lines, _bites_tags() + [_tag_row("render", 300)],
                         extra=["--probe-cap", "48", "--probe-cap", "300"])
    assert rec["verdict"] == "UNCLASSIFIED_REQUESTS"
    r = rec["classes"]["probe"]["kinds"]["render"]
    assert r["verdict"] == "UNCLASSIFIED" and r["n"] == 1 and "code sends [512, 2048]" in r["tag_check"]["problems"][0]
    assert rec["classes"]["probe"]["kinds"]["bites"]["verdict"] == "AGREES"
    # (c) a tag whose KIND the table does not know: every log line found a kind, and the class still fails
    rc, rec = _run_kinds(tmp_path, _LLAMA_LOG, _LLAMA_TAGS + [_tag_row("hook_attest", 48)])
    assert rec["verdict"] == "UNCLASSIFIED_REQUESTS"
    p = rec["classes"]["probe"]
    assert p["verdict"] == "UNCLASSIFIED" and p["n_unclassified"] == 0 and p["n_unclassified_tags"] == 1
    assert p["unclassified"]["unknown_kind_tags"] == {"hook_attest@48": 1}
    assert p["kinds"]["bites"]["verdict"] == "AGREES" and p["kinds"]["render"]["verdict"] == "AGREES"
    assert "unknown kinds" in rec["why"] and "hook_attest@48" in rec["why"]
    # (d) one cap tagged under TWO kinds, one of them off-table (lens_on_bite's code sends 256, not 48): two kinds at one cap
    # are UNCLASSIFIED only when tag-count plus shape cannot separate them — here the off-table tag is lens_on_bite's own
    # problem and the valid claimant (bites, 10 tagged, 10 logged) classifies; the leg still fails on the stray tag
    rc, rec = _run_kinds(tmp_path, _LLAMA_LOG, _LLAMA_TAGS + [_tag_row("lens_on_bite", 48)])
    assert rec["verdict"] == "UNCLASSIFIED_REQUESTS"
    p = rec["classes"]["probe"]
    assert p["kinds"]["bites"]["verdict"] == "AGREES" and p["kinds"]["lens_on_bite"]["verdict"] == "UNCLASSIFIED"
    assert "code sends [256]" in p["kinds"]["lens_on_bite"]["tag_check"]["problems"][0]
    assert p["kinds"]["render"]["verdict"] == "AGREES" and p["n_unclassified"] == 0 and p["n_unclassified_tags"] == 1
    # two TAGGED card-reference kinds at one cap cannot be separated by shape: both UNCLASSIFIED
    rc, rec = _run_kinds(tmp_path, _LLAMA_LOG, _LLAMA_TAGS + [_tag_row("render", 48)])
    assert rec["verdict"] == "UNCLASSIFIED_REQUESTS" and rec["classes"]["probe"]["kinds"]["render"]["verdict"] == "UNCLASSIFIED"
    assert "code sends [512, 2048]" in rec["classes"]["probe"]["kinds"]["render"]["tag_check"]["problems"][0]
    # (e) a divergence AND an unclassified kind: DIVERGED wins the verdict and the why still names the other
    lines = _CARD_CLIENT + _probe_lines(cap=48, n=9, **_GREEDY) + _probe_lines(cap=512, n=1, **_GREEDY)
    rc, rec = _run_kinds(tmp_path, lines, _LLAMA_TAGS)
    assert rec["verdict"] == "DIVERGED" and "kind render" in rec["why"] and "ALSO" in rec["why"] and "matched no kind" in rec["why"]


def test_kinds_mode_an_unknown_shape_is_unclassified(tmp_path):
    """A probe-class request at a cap no kind pins: UNCLASSIFIED whether untagged or tagged with a kind that never sends it."""
    lines = _CARD_CLIENT + _probe_lines(cap=48, n=10, **_GREEDY) + _probe_lines(cap=512, n=1, **_CARD) + _probe_lines(cap=999, n=1, **_GREEDY)
    caps = ["--probe-cap", "48", "--probe-cap", "512", "--probe-cap", "999"]
    rc, rec = _run_kinds(tmp_path, lines, _LLAMA_TAGS, extra=caps)
    assert rec["verdict"] == "UNCLASSIFIED_REQUESTS"
    p = rec["classes"]["probe"]
    assert p["n_unclassified"] == 1 and p["unclassified"]["reasons"] == [{"reason": "no tag or declaration names max_tokens 999", "n": 1}]
    assert p["kinds"]["bites"]["verdict"] == "AGREES" and p["kinds"]["render"]["verdict"] == "AGREES"
    rc, rec = _run_kinds(tmp_path, lines, _LLAMA_TAGS + [_tag_row("bites", 999)], extra=caps)
    assert rec["verdict"] == "UNCLASSIFIED_REQUESTS"
    b = rec["classes"]["probe"]["kinds"]["bites"]
    assert b["verdict"] == "UNCLASSIFIED" and b["n"] == 11 and "code sends [48]" in b["tag_check"]["problems"][0]
    # and WITHOUT the extra --probe-cap the 999 line is a stray request (the class rule), named as such beside the kinds
    rc, rec = _run_kinds(tmp_path, lines, _LLAMA_TAGS)
    assert rec["verdict"] == "UNCLASSIFIED_REQUESTS" and rec["classes"]["unclassified"]["n"] == 1
    assert rec["classes"]["probe"]["verdict"] == "AGREES" and rec["n_unclassified_requests"] == 1
    assert "999" in rec["why"] or "neither a client cap" in rec["why"]


def test_kinds_mode_default_probe_caps_are_the_kind_table_and_every_kind_can_appear(tmp_path):
    """No --probe-cap: the probe class is every cap the kind table pins; the render retry (2048) and the lens-ON bite (256)
    classify under their kinds when tagged."""
    lines = (_CARD_CLIENT + _probe_lines(cap=48, n=10, **_GREEDY) + _probe_lines(cap=512, n=1, **_CARD)
             + _probe_lines(cap=2048, n=1, **_CARD) + _probe_lines(cap=256, n=2, **_GREEDY))
    tags = _bites_tags() + [_tag_row("render", 512), _tag_row("render", 2048, n=1),
                            _tag_row("lens_on_bite", 256), _tag_row("lens_on_bite", 256, vmodel="none", n=0)]
    rc, rec = _run_kinds(tmp_path, lines, tags)
    assert rc == 0 and rec["verdict"] == "AGREES", rec["why"]
    assert rec["probe_caps"] == ["1", "48", "64", "256", "512", "2048"] and rec["probe_cap_source"].startswith("the probe-kind table")
    assert "load_check [the per-leg cap]" in rec["probe_cap_source"] and "positive_control [64]" in rec["probe_cap_source"]
    p = rec["classes"]["probe"]
    assert p["n_by_kind"] == {"render": 2, "bites": 10, "lens_on_bite": 2, "logprob_bites": 0, "load_check": 0, "positive_control": 0, "standup_controls": 0} and p["n"] == 14
    assert p["kinds"]["render"]["tag_check"]["tagged"] == {"512": 1, "2048": 1} and p["kinds"]["render"]["verdict"] == "AGREES"
    lo = p["kinds"]["lens_on_bite"]
    assert lo["verdict"] == "AGREES" and lo["reference"] == _GREEDY_REF and "_lens_on_bite" in lo["reference_source"]
    assert "lens_on_bite 2 = greedy" in rec["why"] and "render 2 = the pinned card row" in rec["why"]


def test_kinds_mode_reads_the_usage_ledger_spelling_and_skips_rejected_rows(tmp_path):
    tags = _LLAMA_TAGS + [{**_tag_row("bites", 48, n=9), "rejected": True}]      # an 11th attempt the proxy refused: never logged
    rc, rec = _run_kinds(tmp_path, _LLAMA_LOG, tags, tags_name="usage_ledger.jsonl", ledger=True)
    assert rc == 0 and rec["verdict"] == "AGREES", rec["why"]
    t = rec["probe_tags"]
    assert t["n_probe_rows"] == 12 and t["n_rejected_skipped"] == 1 and t["by_kind"]["bites"] == [{"max_tokens": "48", "n": 10}]
    assert rec["classes"]["probe"]["kinds"]["bites"]["n_tagged"] == 10


def test_kinds_mode_refuses_without_its_inputs_and_tags_are_refused_outside_it(tmp_path, capsys):
    f = tmp_path / "request_evidence.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in _LLAMA_TAGS))
    # no --probe-tags: the kind is the driver's tag, never inferred from the line alone
    rc, rec = _run(tmp_path, _LLAMA_LOG, expect=_CARD_EXPECT, extra=_KINDS_ARGS)
    assert rc == 2 and rec is None
    err = capsys.readouterr().err
    assert "REFUSED" in err and "--probe-tags" in err and "never" in err and "classification never defaults" in err
    # no --expect: the render kind's reference is the pinned card row, never the driver's declaration
    rc, rec = _run(tmp_path, _LLAMA_LOG, extra=[*_KINDS_ARGS, "--probe-tags", str(f)])
    assert rc == 2 and rec is None
    err = capsys.readouterr().err
    assert "REFUSED" in err and "render kind's reference" in err and "never the driver" in err
    # --probe-tags under another mode: refused, never silently ignored
    for mode in ([], ["--probe-expect", "greedy"], ["--probe-expect", "card"]):
        rc, rec = _run(tmp_path, _LLAMA_LOG, expect=_CARD_EXPECT, extra=[*_CARD_ARGS, *mode, "--probe-tags", str(f)])
        assert rc == 2 and rec is None
        assert "silently ignored" in capsys.readouterr().err


def test_the_kind_table_is_pinned_to_the_code_that_fires_each_kind_and_to_the_drivers_tag():
    import load_check as lc
    import served_capability_run as srun
    import smoke_positive_control as spc
    import unified_sweep_run as usr
    from sisteer.capeval.tinycap import contracts as tc

    # the kind NAMES are the driver's tag vocabulary, byte for byte; the tag prefix too
    assert rrs.PROBE_KINDS == tc.PROBE_KINDS == ("render", "bites", "lens_on_bite", "logprob_bites", "load_check", "positive_control", "standup_controls")
    assert tc.PROBE_KINDS == (*tc.PROBE_KINDS_DRIVER, *tc.PROBE_KINDS_STANDUP)
    # the driver's logprob liveness probes are their OWN kind at cap 1 (a `bites` tag at cap 1
    # would be UNCLASSIFIED against that kind's pinned 48), greedy-referenced, pinned to the driver's constants
    assert rrs.PROBE_KIND_LOGPROB_BITES == tc.PROBE_KIND_LOGPROB_BITES == "logprob_bites" and tc.PROBE_KIND_LOGPROB_BITES in tc.PROBE_KINDS_DRIVER
    assert rrs.PROBE_KIND_TABLE["logprob_bites"]["caps"] == (rrs.LOGPROB_BITES_MAX_TOKENS,) == (tc.LOGPROB_BITES_MAX_TOKENS,) == (1,)
    assert rrs.PROBE_KIND_TABLE["logprob_bites"]["reference"] == "greedy"
    assert '"max_tokens": tc.LOGPROB_BITES_MAX_TOKENS, "seed": 0,' in (REPO / "scripts/served_capability_run.py").read_text(), (
        "the logprob liveness probe must send the named cap the table pins")
    assert tc.PROBE_KINDS_STANDUP == (tc.PROBE_KIND_LOAD_CHECK, tc.PROBE_KIND_POSITIVE_CONTROL, tc.PROBE_KIND_STANDUP_CONTROLS) == ("load_check", "positive_control", "standup_controls")
    # fg1 / F-AV: the native standup's own positive controls -- greedy at the STANDUP's probe cap (endpoint.json.probe_max_tokens), no constant pins it
    assert rrs.PROBE_KIND_STANDUP_CONTROLS == tc.PROBE_KIND_STANDUP_CONTROLS == "standup_controls"
    assert rrs.PROBE_KIND_TABLE["standup_controls"]["caps"] == rrs.CAP_PER_STANDUP == "per-standup" and rrs._pinned_caps("standup_controls") is None
    assert rrs._caps_label("standup_controls") == "per-standup" and rrs.PROBE_KIND_TABLE["standup_controls"]["reference"] == "greedy"
    nsrc = (REPO / "src/sisteer/capeval/serving/native_standup.py").read_text()
    assert '"max_tokens": int(self.probe_max_tokens), "temperature": 0,' in nsrc, "the standup's greedy_ids sends its own probe cap, greedy, untagged"
    assert (rrs.REQUEST_EVIDENCE_FILENAME, rrs.LEDGER_FILENAME) == (tc.REQUEST_EVIDENCE_FILENAME, tc.LEDGER_FILENAME) == ("request_evidence.jsonl", "usage_ledger.jsonl")
    assert (rrs.PROBE_KIND_RENDER, rrs.PROBE_KIND_BITES, rrs.PROBE_KIND_LENS_ON_BITE) == (
        tc.PROBE_KIND_RENDER, tc.PROBE_KIND_BITES, tc.PROBE_KIND_LENS_ON_BITE)
    assert (rrs.PROBE_KIND_LOAD_CHECK, rrs.PROBE_KIND_POSITIVE_CONTROL) == (tc.PROBE_KIND_LOAD_CHECK, tc.PROBE_KIND_POSITIVE_CONTROL)
    assert (lc.LOAD_CHECK_KIND, spc.POSITIVE_CONTROL_KIND, spc.BITES_KIND) == ("load_check", "positive_control", "bites")
    assert rrs.PROBE_TAG_PREFIX == tc.PROBE_TASK_PREFIX == "probe/"
    assert all(tc.probe_task_label(k) == rrs.PROBE_TAG_PREFIX + k for k in rrs.PROBE_KINDS)
    # the CAPS are what each kind's code sends
    assert rrs.PROBE_KIND_TABLE["bites"]["caps"] == (PROBE_MAX_TOKENS,) == (usr._SERVED_PROBE_MAX_TOKENS,) == (48,)
    assert rrs.PROBE_KIND_TABLE["render"]["caps"] == (srun.RENDER_PROBE_MAX_TOKENS, srun.RENDER_PROBE_RETRY_MAX_TOKENS) == (512, 2048)
    assert rrs.PROBE_KIND_TABLE["lens_on_bite"]["caps"] == (srun.LENS_ON_BITE_MAX_TOKENS,) == (256,)
    assert rrs.PROBE_KIND_TABLE["positive_control"]["caps"] == (spc.POSITIVE_CONTROL_MAX_TOKENS,) == (
        tc.POSITIVE_CONTROL_MAX_TOKENS,) == (rrs.POSITIVE_CONTROL_MAX_TOKENS,) == (64,)
    csrc = (REPO / "src/sisteer/capeval/tinycap/contracts.py").read_text()
    assert rrs.PROBE_KIND_TABLE["load_check"]["caps"] == rrs.CAP_PER_LEG == "per-leg", (
        "the load check fires at the RUN's cap: the client cap on the static / sampled lanes, the controls cap (512) on MATH-500")
    assert rrs._pinned_caps("load_check") is None and rrs._caps_label("load_check") == "per-leg" and rrs._caps_label("render") == [512, 2048]
    # the identity BASIS vocabulary: a tag binds, a declaration counts
    assert (rrs.KIND_BASIS_TAG, rrs.KIND_BASIS_DECLARED_COUNT, rrs.KIND_BASIS_TAG_AND_DECLARED) == tc.KIND_BASES == ("tag", "declared_count", "tag+declared")
    assert tc.KIND_JOIN_AMBIGUOUS_FIELD == "join_ambiguous" and "binds" in csrc and "counts" in csrc
    for name in ("KIND_BASIS_TAG", "KIND_BASIS_DECLARED_COUNT", "KIND_BASIS_TAG_AND_DECLARED", "KIND_BASES", "KIND_JOIN_AMBIGUOUS_FIELD"):
        assert name in tc.__all__, name
    src = (REPO / "scripts/served_capability_run.py").read_text()
    assert '"temperature": 0.0, "max_tokens": LENS_ON_BITE_MAX_TOKENS, "seed": 0,' in src, (
        "the lens-ON bite must send the named cap the table pins")
    assert 'prompt_id, max_tokens, seed = RENDER_PROBE_PROMPT_ARITHMETIC, RENDER_PROBE_MAX_TOKENS, 0' in src
    lsrc = (F22 / "load_check.py").read_text()
    assert '"max_tokens": int(cap), "temperature": 0.0,' in lsrc and '"seed": i}' in lsrc, "the load check sends the run's cap, greedy"
    ssrc = (F22 / "smoke_positive_control.py").read_text()
    assert 'ap.add_argument("--max-tokens", type=int, default=POSITIVE_CONTROL_MAX_TOKENS)' in ssrc
    assert '"max_tokens": args.max_tokens, "seed": 0})' in ssrc and 'resp = chat_pc({"model": vmodel, "messages": msgs, "temperature": 0.0,' in ssrc
    # the REFERENCES: render is the card row (never the driver's sampling_row), the greedy kinds greedy
    assert rrs.PROBE_KIND_TABLE["render"]["reference"] == "card"
    assert all(rrs.PROBE_KIND_TABLE[k]["reference"] == "greedy" for k in ("bites", "lens_on_bite", "logprob_bites", "load_check", "positive_control", "standup_controls"))
    assert rrs.kind_reference("render", {"temperature": 0.6}) == {"temperature": 0.6}
    assert rrs.kind_reference("bites", {"temperature": 0.6}) == rrs.probe_reference() == _GREEDY_REF == dict(tc.GREEDY_REFERENCE_RESOLVED)
    assert rrs.kind_reference("load_check", _CARD_EXPECT) == rrs.kind_reference("positive_control", _CARD_EXPECT) == _GREEDY_REF
    # the greedy reference's top_p is 1.0 even under a generation_config that declares 0.9 — said beside the constant, twice
    rsrc = (REPO / "scripts/sdfbase/record_resolved_sampler.py").read_text()
    for text in (csrc, rsrc):
        assert "EVEN WHEN" in text and "top_p 0.9" in text and "97376" in text and "looks" in text and "sampler drift" in text
    # the TAG the recorder reads is the one the driver writes: the labelled wrapper stamps the task header with the
    # kind, and the request-evidence record carries `task` + `max_tokens` (contracts.REQUEST_EVIDENCE_FIELDS)
    assert "task" in tc.REQUEST_EVIDENCE_FIELDS and "max_tokens" in tc.REQUEST_EVIDENCE_FIELDS
    assert "label = tc.probe_task_label(kind)" in src and "headers = {tc.HEADER_TASK: label," in src
    rec = tc.request_evidence_record(json.dumps({"model": "m", "messages": [], "temperature": 0.0, "max_tokens": 48}).encode(),
                                     headers={tc.HEADER_TASK: tc.probe_task_label("bites"), tc.HEADER_SAMPLE: "m/0",
                                              tc.HEADER_REQUEST_ID: "0" * 32})
    assert rec["kind"] == "probe" and rec["task"] == "probe/bites" and rec["max_tokens"] == 48
    # no two kinds pin one cap (the shape cross-check would be ambiguous by construction)
    caps = [c for k in rrs.PROBE_KINDS for c in (rrs._pinned_caps(k) or ())]
    assert len(caps) == len(set(caps)) and sorted(caps) == [1, 48, 64, 256, 512, 2048]
    # the VOCABULARY lives in contracts, beside the kinds, and the recorder speaks it
    assert set(rrs._EXIT_STRICT) | {"REFUSED"} == set(tc.RESOLVED_SAMPLER_VERDICTS)
    assert "PROBE_NOT_INSTRUMENTED" in tc.RESOLVED_SAMPLER_VERDICTS and "NOT_INSTRUMENTED" in tc.RESOLVED_SAMPLER_CLASS_VERDICTS
    assert rrs.VERDICT_NOT_INSTRUMENTED == tc.PROBE_VERDICT_NOT_INSTRUMENTED == "NOT_INSTRUMENTED"
    assert rrs.NOT_INSTRUMENTED_REASON_UNTAGGED_DRIVER == tc.NOT_INSTRUMENTED_REASON_UNTAGGED_DRIVER == "untagged_driver"
    assert rrs.NOT_INSTRUMENTED_CONSEQUENCE == tc.NOT_INSTRUMENTED_CONSEQUENCE
    assert rrs.PROBE_VERDICT_PRECEDENCE == tc.RESOLVED_SAMPLER_PROBE_VERDICT_PRECEDENCE == ("DIVERGED", "UNCLASSIFIED", "AGREES", "NOT_INSTRUMENTED", "EMPTY")
    assert rrs.DECLARE_KIND_GRAMMAR == tc.DECLARE_KIND_GRAMMAR == "<kind>:cap=<C>[:count=<N>|:count=<N>!]" and tc.DECLARE_KIND_FLAG == "--declare-kind"
    for name in ("PROBE_KIND_LOAD_CHECK", "PROBE_KIND_POSITIVE_CONTROL", "PROBE_KIND_STANDUP_CONTROLS", "PROBE_KINDS_STANDUP", "POSITIVE_CONTROL_MAX_TOKENS",
                 "GREEDY_REFERENCE_RESOLVED", "DECLARE_KIND_GRAMMAR", "RESOLVED_SAMPLER_VERDICTS", "RESOLVED_SAMPLER_CLASS_VERDICTS",
                 "RESOLVED_SAMPLER_PROBE_VERDICT_PRECEDENCE", "NOT_INSTRUMENTED_CONSEQUENCE"):
        assert name in tc.__all__, name


# ------------------------------------------------- 9. the standup's two harness kinds, declarations, NOT_INSTRUMENTED
# Job 97376 (the Llama lane's 8,192 rung): the client class held 632 sampled draws PLUS 46 greedy load-check requests at
# the SAME cap (a false DIVERGED), and 48 greedy positive-control requests at 64 sat UNCLASSIFIED. The load check fires at
# the RUN's cap by design, so no cap can separate it — identity must come from the SOURCE: the standup scripts now stamp
# `probe/load_check` / `probe/positive_control` (+ `probe/bites` for probe_fn) and write <out stem>.request_evidence.jsonl
# (harness_evidence.py), and a standup that stamps no tag (a frozen sha) DECLARES the same identity: --declare-kind
# <kind>:cap=<C>[:count=<N>|:count=<N>!], first-class, cross-checked exactly against the log. The untagged, undeclared
# remainder on a ZERO-tag driver is NOT_INSTRUMENTED (descriptive, exit 0); on a tagging driver it is UNCLASSIFIED.
_LC_LOG = _CARD_CLIENT + _probe_lines(cap=8192, n=2, **_GREEDY)     # 3 sampled client draws + 2 greedy load checks at the client cap
_STANDUP_LOG = _LC_LOG + _probe_lines(cap=48, n=10, **_GREEDY) + _probe_lines(cap=64, n=6, **_GREEDY)


def _lc_tags(n=2, cap=8192):
    return [_tag_row("load_check", cap, vmodel="petri__base", n=i) for i in range(n)]


def _decl(*specs):
    return [x for spec in specs for x in ("--declare-kind", spec)]


def test_a_tagged_load_check_at_the_client_cap_is_carved_by_its_tag_and_agrees(tmp_path):
    lines = _LC_LOG + _probe_lines(cap=48, n=10, **_GREEDY) + _probe_lines(cap=512, n=1, **_CARD)
    rc, rec = _run_kinds(tmp_path, lines, _LLAMA_TAGS + _lc_tags())
    assert rc == 0 and rec["verdict"] == "AGREES", rec["why"]
    lc = rec["classes"]["probe"]["kinds"]["load_check"]
    assert lc["verdict"] == "AGREES" and lc["n"] == 2 and lc["n_tagged"] == 2 and lc["n_declared"] == 0
    assert lc["reference"] == _GREEDY_REF and lc["reference_kind"] == "greedy" and lc["caps_pinned"] == "per-leg"
    assert lc["basis"] == "tag" and lc["join_ambiguous"] is False and lc["competing_kinds"] == [], "a tag BINDS the request to the kind"
    assert "load_check.py" in lc["reference_source"] and lc["identity"]["source"] == "tag" and lc["identity"]["regime"] == ["client_cap"]
    assert "greedy shape separates" in lc["identity"]["selection"][0]
    assert lc["tag_check"] == {"agrees": True, "tagged": {"8192": 2}, "logged": {"8192": 2}, "problems": []}
    c = rec["classes"]["client"]
    assert c["verdict"] == "AGREES" and c["n"] == 3 and rec["n_client_requests"] == 3, "the client class holds ONLY the untagged requests at the client cap"
    assert rec["n_probe_requests"] == 13 and rec["n_carved_from_client"] == {"load_check": 2}
    assert rec["probe_remainder_verdict"] == "EMPTY" and rec["n_unclassified_requests"] == 0
    assert "load_check 2 = greedy + neutral defaults (tag)" in rec["why"] and rec["probe_basis_by_kind"]["load_check"] == "tag"
    assert _run_kinds(tmp_path, lines, _LLAMA_TAGS + _lc_tags(), extra=["--strict"])[0] == 0
    # the SAME log under greedy / card: the load check reads as a client DIVERGENCE — the reading this kind removes
    for mode in ([], ["--probe-expect", "card"]):
        r2 = _run(tmp_path, lines, expect=_CARD_EXPECT, extra=[*_CARD_ARGS, "--probe-cap", "48", "--probe-cap", "512", *mode])[1]
        assert r2["verdict"] == "DIVERGED" and r2["classes"]["client"]["verdict"] == "DIVERGED" and r2["classes"]["client"]["n"] == 5
    # a tag count that disagrees with the greedy-shaped requests at the client cap: UNCLASSIFIED, never a partial carve
    rc, rec = _run_kinds(tmp_path, lines, _LLAMA_TAGS + _lc_tags(n=3))
    assert rec["verdict"] == "UNCLASSIFIED_REQUESTS"
    lc = rec["classes"]["probe"]["kinds"]["load_check"]
    assert lc["verdict"] == "UNCLASSIFIED" and lc["n"] == 2 and lc["n_tagged"] == 3 and "disagree in count" in lc["tag_check"]["problems"][0]
    assert lc["reference"] is None and lc["divergences"] == []
    assert rec["classes"]["client"]["n"] == 3 and rec["n_unclassified_requests"] == 2 and rec["n_probe_unclassified_tags"] == 3


def test_a_declared_load_check_at_the_client_cap_under_a_sampled_client_is_accepted_cap_only(tmp_path):
    """The greedy shape separates the load check from a SAMPLED client class, so count is optional / a maximum."""
    for decl in ("load_check:cap=8192", "load_check:cap=8192:count=2", "load_check:cap=8192:count=5", "load_check:cap=8192:count=2!"):
        rc, rec = _run(tmp_path, _LC_LOG, expect=_CARD_EXPECT, extra=[*_KINDS_ARGS, *_decl(decl)])
        assert rc == 0 and rec["verdict"] == "AGREES", (decl, rec["why"])
        lc = rec["classes"]["probe"]["kinds"]["load_check"]
        assert lc["verdict"] == "AGREES" and lc["n"] == 2 and lc["n_declared"] == 1 and lc["n_tagged"] == 0 and lc["reference"] == _GREEDY_REF
        assert lc["identity"]["source"] == "declaration" and lc["identity"]["claims"] == [
            {"cap": "8192", "count": (None if decl.count("count") == 0 else int(decl.split("count=")[1].rstrip("!"))),
             "exact": decl.endswith("!"), "source": "declaration"}]
        assert "greedy shape separates" in lc["identity"]["selection"][0] and lc["identity"]["regime"] == ["client_cap"]
        assert rec["classes"]["client"]["n"] == 3 and rec["classes"]["client"]["verdict"] == "AGREES"
        assert rec["n_carved_from_client"] == {"load_check": 2} and rec["probe_remainder_verdict"] == "EMPTY"
        assert rec["declarations"][0]["raw"] == decl and rec["probe_tags"]["named"] is False and rec["declare_kind_grammar"] == rrs.DECLARE_KIND_GRAMMAR
        assert "load_check 2 = greedy + neutral defaults (declared_count)" in rec["why"]
        assert lc["basis"] == "declared_count" and rec["probe_basis_by_kind"] == {"load_check": "declared_count"}, "a declaration COUNTS"
        assert lc["join_ambiguous"] is False and rec["probe_join_ambiguous_kinds"] == []
        assert rec["classes"]["probe"]["client_regime"] == "sampled"
    # a declared load check with nothing greedy at the cap under a maximum: the kind is EMPTY and the run still AGREES
    rc, rec = _run(tmp_path, _CARD_CLIENT, expect=_CARD_EXPECT, extra=[*_KINDS_ARGS, *_decl("load_check:cap=8192:count=46")])
    assert rc == 0 and rec["verdict"] == "AGREES" and rec["classes"]["probe"]["kinds"]["load_check"]["verdict"] == "EMPTY"
    assert rec["classes"]["probe"]["verdict"] == "EMPTY" and rec["n_probe_requests"] == 0


def test_a_declaration_whose_count_or_cap_disagrees_with_the_log_is_unclassified(tmp_path):
    # exact count mismatch at the client cap: the greedy requests are carved OUT and left UNCLASSIFIED — never folded back
    rc, rec = _run(tmp_path, _LC_LOG, expect=_CARD_EXPECT, extra=[*_KINDS_ARGS, *_decl("load_check:cap=8192:count=1!")])
    assert rc == 0 and rec["verdict"] == "UNCLASSIFIED_REQUESTS", rec["why"]
    lc = rec["classes"]["probe"]["kinds"]["load_check"]
    assert lc["verdict"] == "UNCLASSIFIED" and lc["n"] == 2 and lc["reference"] is None and lc["divergences"] == []
    assert "1! (exact)" in lc["tag_check"]["problems"][0] and "2 logged" in lc["tag_check"]["problems"][0]
    assert rec["n_unclassified_requests"] == 2 and rec["n_probe_unclassified_declarations"] == 1 and rec["n_probe_unclassified_tags"] == 0
    assert rec["classes"]["client"]["n"] == 3 and rec["classes"]["client"]["verdict"] == "AGREES"
    assert "declaration(s) bound to no classified kind" in rec["why"] and "2 x load_check:" in rec["why"]
    assert _run(tmp_path, _LC_LOG, expect=_CARD_EXPECT, extra=[*_KINDS_ARGS, *_decl("load_check:cap=8192:count=1!"), "--strict"])[0] == 3
    # a maximum exceeded: unexpected extra traffic stays loud
    rc, rec = _run(tmp_path, _LC_LOG, expect=_CARD_EXPECT, extra=[*_KINDS_ARGS, *_decl("load_check:cap=8192:count=1")])
    assert rec["verdict"] == "UNCLASSIFIED_REQUESTS" and "at most 1" in rec["classes"]["probe"]["kinds"]["load_check"]["tag_check"]["problems"][0]
    # positive_control at its probe cap: a maximum exceeded; an exact count with nothing fired; a maximum with nothing fired is EMPTY
    pc = _CARD_CLIENT + _probe_lines(cap=64, n=5, **_GREEDY)
    rc, rec = _run(tmp_path, pc, expect=_CARD_EXPECT, extra=[*_KINDS_ARGS, *_decl("positive_control:cap=64:count=3")])
    assert rec["verdict"] == "UNCLASSIFIED_REQUESTS" and rec["classes"]["probe"]["kinds"]["positive_control"]["n"] == 5
    assert "unexpected extra traffic" in rec["classes"]["probe"]["kinds"]["positive_control"]["tag_check"]["problems"][0]
    rc, rec = _run(tmp_path, _CARD_CLIENT, expect=_CARD_EXPECT, extra=[*_KINDS_ARGS, *_decl("positive_control:cap=64:count=5!")])
    assert rec["verdict"] == "UNCLASSIFIED_REQUESTS" and rec["classes"]["probe"]["kinds"]["positive_control"]["verdict"] == "UNCLASSIFIED"
    assert rec["n_unclassified_requests"] == 0 and rec["n_probe_unclassified_declarations"] == 1
    rc, rec = _run(tmp_path, _CARD_CLIENT, expect=_CARD_EXPECT, extra=[*_KINDS_ARGS, *_decl("positive_control:cap=64:count=5")])
    assert rec["verdict"] == "AGREES" and rec["classes"]["probe"]["kinds"]["positive_control"]["verdict"] == "EMPTY"
    assert "none fired" in rec["classes"]["probe"]["kinds"]["positive_control"]["identity"]["selection"][0]
    # a cap the kind's code does not send: identity and shape disagree; the declared cap still becomes a probe cap
    rc, rec = _run(tmp_path, _CARD_CLIENT + _probe_lines(cap=128, n=2, **_GREEDY), expect=_CARD_EXPECT,
                   extra=[*_KINDS_ARGS, *_decl("positive_control:cap=128")])
    assert rec["verdict"] == "UNCLASSIFIED_REQUESTS"
    k = rec["classes"]["probe"]["kinds"]["positive_control"]
    assert k["verdict"] == "UNCLASSIFIED" and k["n"] == 2 and "code sends [64]" in k["tag_check"]["problems"][0]
    assert rec["probe_caps"] == ["1", "48", "64", "128", "256", "512", "2048"] and "--declare-kind caps ['128']" in rec["probe_cap_source"]
    # a declared cap whose requests are NOT greedy-shaped: a declaration claims a cap for the standup's greedy traffic only
    rc, rec = _run(tmp_path, _CARD_CLIENT + _probe_lines(cap=64, n=2, **_CARD), expect=_CARD_EXPECT,
                   extra=[*_KINDS_ARGS, *_decl("positive_control:cap=64")])
    assert rec["verdict"] == "UNCLASSIFIED_REQUESTS" and "not greedy-shaped" in rec["classes"]["probe"]["kinds"]["positive_control"]["tag_check"]["problems"][0]


def test_a_leftover_untagged_undeclared_greedy_request_at_the_client_cap_is_unclassified_never_folded_back(tmp_path):
    # zero tags, no declaration: the two greedy requests at 8192 are neither client (the reference is sampled) nor any kind
    rc, rec = _run_kinds(tmp_path, _LC_LOG, [], write_tags=False)
    assert rc == 0 and rec["verdict"] == "UNCLASSIFIED_REQUESTS", rec["why"]
    c = rec["classes"]["client"]
    assert c["n"] == 3 and c["verdict"] == "AGREES", "the client class is the untagged SAMPLED requests; the greedy ones are not folded back as a divergence"
    p = rec["classes"]["probe"]
    assert p["verdict"] == "UNCLASSIFIED" and p["n"] == 2 and p["n_unclassified"] == 2 and rec["n_carved_from_client"] == {"unclassified": 2}
    reason = p["unclassified"]["reasons"][0]["reason"]
    assert reason.startswith("greedy-shaped at the client cap 8192 with no tag and no declaration") and "load_check:cap=8192" in reason
    assert "not folded" in reason and p["unclassified"]["distinct_max_tokens"] == [{"max_tokens": "8192", "n": 2}]
    assert p["remainder"]["verdict"] == "EMPTY", "nothing at a probe cap: the client-cap leftover is UNCLASSIFIED, never NOT_INSTRUMENTED"
    assert all(k["verdict"] == "EMPTY" for k in p["kinds"].values())
    assert _run_kinds(tmp_path, _LC_LOG, [], write_tags=False, extra=["--strict"])[0] == 3
    # with SOME tags present (a tagging driver) the same leftover is still UNCLASSIFIED
    lines = _LC_LOG + _probe_lines(cap=48, n=10, **_GREEDY)
    rc, rec = _run_kinds(tmp_path, lines, _bites_tags())
    assert rec["verdict"] == "UNCLASSIFIED_REQUESTS" and rec["classes"]["probe"]["kinds"]["bites"]["verdict"] == "AGREES"
    assert rec["classes"]["client"]["n"] == 3 and rec["n_unclassified_requests"] == 2
    # under a GREEDY client reference nothing is carved by shape: greedy at the client cap IS client-shaped (the record says so)
    glog = [_sp(i, cap=8192, **_GREEDY) for i in range(3)] + _probe_lines(cap=8192, n=2, **_GREEDY)
    rc, rec = _run_kinds(tmp_path, glog, [], write_tags=False, expect={"temperature": 0.0}, tags_name="never_written.jsonl")
    assert rc == 0 and rec["verdict"] == "AGREES" and rec["classes"]["client"]["n"] == 5 and rec["n_carved_from_client"] == {}
    assert "indistinguishable from client traffic" in rec["why"] and rec["classes"]["probe"]["client_regime"] == "greedy"


def test_the_positive_control_at_64_classifies_by_tag_or_by_declaration(tmp_path):
    lines = _CARD_CLIENT + _probe_lines(cap=64, n=6, **_GREEDY)
    tags = [_tag_row("positive_control", 64, vmodel=vm, n=i) for vm in ("petri__si_x", "none", "none__lens0") for i in range(2)]
    rc, rec = _run_kinds(tmp_path, lines, tags)
    assert rc == 0 and rec["verdict"] == "AGREES", rec["why"]
    k = rec["classes"]["probe"]["kinds"]["positive_control"]
    assert k["verdict"] == "AGREES" and k["n"] == 6 and k["n_tagged"] == 6 and k["caps_pinned"] == [64] and k["reference"] == _GREEDY_REF
    assert "smoke_positive_control.py" in k["reference_source"] and k["identity"]["regime"] == ["probe_cap"] and k["identity"]["source"] == "tag"
    assert "positive_control 6 = greedy + neutral defaults (tag)" in rec["why"]
    # declared, cap-only: the cap plus the verified greedy shape is the identity (the count is decided at run time)
    rc, rec = _run(tmp_path, lines, expect=_CARD_EXPECT, extra=[*_KINDS_ARGS, *_decl("positive_control:cap=64")])
    assert rec["verdict"] == "AGREES" and rec["classes"]["probe"]["kinds"]["positive_control"]["identity"]["source"] == "declaration"
    assert "every request at max_tokens 64 (6), greedy-shaped" in rec["classes"]["probe"]["kinds"]["positive_control"]["identity"]["selection"][0]
    # a TAGGED positive control that decoded under the card row is a DIVERGENCE of its kind (the tag classifies, the shape is compared)
    rc, rec = _run_kinds(tmp_path, _CARD_CLIENT + _probe_lines(cap=64, n=6, **_CARD), tags)
    assert rec["verdict"] == "DIVERGED" and "kind positive_control" in rec["why"]
    d = {x["key"]: x for x in rec["classes"]["probe"]["kinds"]["positive_control"]["divergences"]}
    assert set(d) == {"temperature", "top_p"} and d["top_p"]["resolved"] == 0.9 and d["temperature"]["resolved"] == 0.6
    assert all(x["kind"] == "positive_control" for x in d.values())
    # greedy / card modes on the same log: 64 is not a probe cap there, so the requests are stray UNCLASSIFIED (today's reading)
    r2 = _run(tmp_path, lines, expect=_CARD_EXPECT, extra=[*_CARD_ARGS, "--probe-cap", "48"])[1]
    assert r2["verdict"] == "UNCLASSIFIED_REQUESTS" and r2["classes"]["unclassified"]["n"] == 6


_PRE_KINDS_TOP_KEYS = ["verdict", "why", "classes", "n_client_requests", "n_probe_requests", "n_unclassified_requests", "n_logs",
                       "missing_logs", "n_resolved_requests", "class_rule", "client_caps", "client_cap_source", "probe_caps",
                       "probe_cap_source", "probe_reference", "vllm_neutral_defaults", "exit_policy", "generation_config_override",
                       "sent", "logs", "distinct_resolved", "distinct_max_tokens", "reference", "reference_source", "divergences"]
_PRE_KINDS_CLASS_KEYS = ["n", "verdict", "distinct_resolved", "distinct_max_tokens", "reference", "reference_source", "divergences",
                         "reference_kind"]


def test_greedy_and_card_records_keep_the_pre_kinds_shape_byte_for_byte(tmp_path):
    """Pinned: the two single-reference modes are untouched by the standup kinds — the record's key order, the class
    blocks' key order, the probe reference's shape, and today's reading of a log that carries a load check and a
    positive control (a client DIVERGENCE and a stray UNCLASSIFIED). (The same records were byte-compared against the
    pre-change recorder on the job-97376 log at the time of the change; this pins the shape they must keep.)"""
    lines = _STANDUP_LOG
    for mode in ([], ["--probe-expect", "greedy"], ["--probe-expect", "card"]):
        rc, rec = _run(tmp_path, lines, expect=_CARD_EXPECT, extra=[*_CARD_ARGS, "--probe-cap", "48", *mode])
        assert rc == 0 and list(rec) == _PRE_KINDS_TOP_KEYS, mode
        assert list(rec["classes"]["probe"]) == _PRE_KINDS_CLASS_KEYS and list(rec["classes"]["client"]) == _PRE_KINDS_CLASS_KEYS
        assert list(rec["probe_reference"]) == ["kind", "expect", "source"] and rec["probe_caps"] == ["48"]
        assert rec["verdict"] == "DIVERGED" and rec["classes"]["client"]["n"] == 5 and rec["classes"]["unclassified"]["n"] == 6
        blob = json.dumps(rec)
        assert "load_check" not in blob and "NOT_INSTRUMENTED" not in blob and "positive_control" not in blob
        assert "declarations" not in rec and "remainder" not in rec["classes"]["probe"] and "kinds" not in rec["classes"]["probe"]
    # the declaration flag is refused outside kinds mode — never silently ignored (a refused run writes no record)
    (tmp_path / "rec.json").unlink()
    for mode in ([], ["--probe-expect", "card"]):
        rc, rec = _run(tmp_path, lines, expect=_CARD_EXPECT, extra=[*_CARD_ARGS, *mode, *_decl("positive_control:cap=64")])
        assert rc == 2 and rec is None


def test_a_zero_tag_run_with_a_load_check_declaration_carves_the_client_class_and_reports_the_rest_not_instrumented(tmp_path):
    lines = _LLAMA_LOG + _probe_lines(cap=8192, n=2, **_GREEDY)
    rc, rec = _run(tmp_path, lines, expect=_CARD_EXPECT, extra=[*_KINDS_ARGS, *_decl("load_check:cap=8192")])
    assert rc == 0 and rec["verdict"] == "PROBE_NOT_INSTRUMENTED", rec["why"]
    p = rec["classes"]["probe"]
    assert p["verdict"] == "AGREES", "precedence: per-kind AGREES, with the remainder NOT_INSTRUMENTED beside it"
    assert p["kinds"]["load_check"]["verdict"] == "AGREES" and p["kinds"]["load_check"]["n"] == 2
    assert rec["classes"]["client"]["n"] == 3 and rec["classes"]["client"]["verdict"] == "AGREES", "the client class excludes the declared greedy requests"
    assert rec["n_carved_from_client"] == {"load_check": 2}
    assert p["remainder"]["verdict"] == "NOT_INSTRUMENTED" and p["remainder"]["n"] == 11 and p["remainder"]["reason"] == "untagged_driver"
    assert "load_check 2 = greedy + neutral defaults (declared_count)" in rec["why"] and "remainder NOT_INSTRUMENTED: 11" in rec["why"]
    assert rec["probe_tags"]["named"] is False and rec["probe_tags"]["sources"] == [] and "no --probe-tags named" in rec["why"]
    assert _run(tmp_path, lines, expect=_CARD_EXPECT, extra=[*_KINDS_ARGS, *_decl("load_check:cap=8192"), "--strict"])[0] == 0


def test_a_fully_declared_zero_tag_run_agrees_per_kind_with_an_empty_remainder(tmp_path):
    decl = _decl("load_check:cap=8192", "bites:cap=48:count=10!", "positive_control:cap=64:count=6")
    rc, rec = _run(tmp_path, _STANDUP_LOG, expect=_CARD_EXPECT, extra=[*_KINDS_ARGS, *decl])
    assert rc == 0 and rec["verdict"] == "AGREES", rec["why"]
    p = rec["classes"]["probe"]
    assert p["verdict"] == "AGREES" and p["remainder"]["verdict"] == "EMPTY" and p["remainder"]["n"] == 0
    assert rec["probe_verdict_by_kind"] == {"render": "EMPTY", "bites": "AGREES", "lens_on_bite": "EMPTY", "logprob_bites": "EMPTY", "load_check": "AGREES", "positive_control": "AGREES",
                                            "standup_controls": "EMPTY"}
    assert rec["n_probe_requests_by_kind"] == {"render": 0, "bites": 10, "lens_on_bite": 0, "logprob_bites": 0, "load_check": 2, "positive_control": 6, "standup_controls": 0}
    assert rec["n_client_requests"] == 3 and rec["n_probe_requests"] == 18 and rec["n_unclassified_requests"] == 0
    assert all(p["kinds"][k]["identity"]["source"] == "declaration" for k in ("bites", "load_check", "positive_control"))
    assert "bites 10 = greedy + neutral defaults (declared_count)" in rec["why"] and "positive_control 6 = greedy" in rec["why"]
    assert rec["probe_basis_by_kind"] == {"bites": "declared_count", "load_check": "declared_count", "positive_control": "declared_count"}
    assert len(rec["declarations"]) == 3
    assert rec["declarations"][1] == {"kind": "bites", "cap": "48", "count": 10, "exact": True, "raw": "bites:cap=48:count=10!"}
    assert p["verdict_precedence"] == ["DIVERGED", "UNCLASSIFIED", "AGREES", "NOT_INSTRUMENTED", "EMPTY"]
    assert _run(tmp_path, _STANDUP_LOG, expect=_CARD_EXPECT, extra=[*_KINDS_ARGS, *decl, "--strict"])[0] == 0
    # precedence: a declared kind that resolved temperature 0.0 but top_p 0.9 is a DIVERGENCE of its kind — the greedy
    # path resolves top_p 1.0 on this engine even under a generation_config that declares 0.9, so the reference is 1.0
    # (a greedy reference pinned at the card's top_p would read the real world the other way round)
    odd = _LC_LOG + _probe_lines(cap=64, n=6, t=0.0, tp=0.9, tk=0, pp=0.0)
    rc, rec = _run(tmp_path, odd, expect=_CARD_EXPECT, extra=[*_KINDS_ARGS, *_decl("load_check:cap=8192", "positive_control:cap=64")])
    assert rec["verdict"] == "DIVERGED" and rec["classes"]["probe"]["verdict"] == "DIVERGED" and "kind positive_control" in rec["why"]
    d = rec["classes"]["probe"]["kinds"]["positive_control"]["divergences"]
    assert [x["key"] for x in d] == ["top_p"] and d[0]["sent"] == 1.0 and d[0]["resolved"] == 0.9 and d[0]["kind"] == "positive_control"
    assert rec["classes"]["probe"]["kinds"]["load_check"]["verdict"] == "AGREES"
    assert _run(tmp_path, odd, expect=_CARD_EXPECT, extra=[*_KINDS_ARGS, *_decl("load_check:cap=8192", "positive_control:cap=64"), "--strict"])[0] == 1


def test_a_partially_declared_zero_tag_run_agrees_on_the_declared_kinds_beside_a_not_instrumented_remainder(tmp_path):
    decl = _decl("load_check:cap=8192:count=2!", "positive_control:cap=64:count=6!")
    rc, rec = _run(tmp_path, _STANDUP_LOG, expect=_CARD_EXPECT, extra=[*_KINDS_ARGS, *decl])
    assert rc == 0 and rec["verdict"] == "PROBE_NOT_INSTRUMENTED", rec["why"]
    p = rec["classes"]["probe"]
    assert p["verdict"] == "AGREES" and p["remainder"]["verdict"] == "NOT_INSTRUMENTED" and p["remainder"]["n"] == 10
    assert p["remainder"]["shape_counts"] == [{"max_tokens": "48", "shape": "greedy", "n": 10}]
    assert rec["probe_verdict_by_kind"]["load_check"] == "AGREES" and rec["probe_verdict_by_kind"]["positive_control"] == "AGREES"
    assert rec["probe_verdict_by_kind"]["bites"] == "EMPTY"
    assert "load_check 2 = greedy" in rec["why"] and "positive_control 6 = greedy" in rec["why"]
    assert "remainder NOT_INSTRUMENTED: 10 untagged, undeclared request(s) (10 greedy at max_tokens 48)" in rec["why"]
    assert rec["n_client_requests"] == 3 and rec["n_unclassified_requests"] == 0
    assert _run(tmp_path, _STANDUP_LOG, expect=_CARD_EXPECT, extra=[*_KINDS_ARGS, *decl, "--strict"])[0] == 0


def test_a_load_check_declaration_at_the_client_cap_under_a_greedy_client_needs_an_exact_count(tmp_path):
    glog = [_sp(i, cap=8192, **_GREEDY) for i in range(3)] + _probe_lines(cap=8192, n=2, **_GREEDY)
    gexp = {"temperature": 0.0}
    for decl in ("load_check:cap=8192", "load_check:cap=8192:count=2"):
        rc, rec = _run(tmp_path, glog, expect=gexp, extra=[*_KINDS_ARGS, *_decl(decl)])
        assert rc == 2 and rec is not None and rec["verdict"] == "REFUSED", decl
        assert "exact count=N!" in rec["why"] and "GREEDY client reference" in rec["why"] and decl in rec["why"]
    # a client reference that states no temperature cannot tell the regime either
    rc, rec = _run(tmp_path, glog, expect={"top_p": 1.0}, extra=[*_KINDS_ARGS, *_decl("load_check:cap=8192")])
    assert rc == 2 and rec["verdict"] == "REFUSED" and "states no temperature" in rec["why"]
    # with count=N!: N identical greedy-shaped requests are taken by count — the choice is immaterial
    rc, rec = _run(tmp_path, glog, expect=gexp, extra=[*_KINDS_ARGS, *_decl("load_check:cap=8192:count=2!")])
    assert rc == 0 and rec["verdict"] == "AGREES", rec["why"]
    lc = rec["classes"]["probe"]["kinds"]["load_check"]
    assert lc["verdict"] == "AGREES" and lc["n"] == 2 and "by count" in lc["identity"]["selection"][0] and "immaterial" in lc["identity"]["selection"][0]
    assert rec["classes"]["client"]["n"] == 3 and rec["classes"]["client"]["verdict"] == "AGREES" and rec["classes"]["probe"]["client_regime"] == "greedy"
    # more declared than logged: which requests are the load check cannot be told — nothing stays in the client class by guess
    rc, rec = _run(tmp_path, glog, expect=gexp, extra=[*_KINDS_ARGS, *_decl("load_check:cap=8192:count=6!")])
    assert rec["verdict"] == "UNCLASSIFIED_REQUESTS" and "only 5 greedy-shaped" in rec["classes"]["probe"]["kinds"]["load_check"]["tag_check"]["problems"][0]
    assert rec["classes"]["client"]["n"] == 0 and rec["classes"]["probe"]["kinds"]["load_check"]["n"] == 5
    # the greedy-shaped requests at the cap are NOT identical: which N are the load check cannot be told -> UNCLASSIFIED
    mixed = glog[:4] + [_sp(9, cap=8192, t=0.0, tp=0.9, tk=0, pp=0.0)]
    rc, rec = _run(tmp_path, mixed, expect=gexp, extra=[*_KINDS_ARGS, *_decl("load_check:cap=8192:count=2!")])
    assert rec["verdict"] == "UNCLASSIFIED_REQUESTS" and "cannot be told" in rec["classes"]["probe"]["kinds"]["load_check"]["tag_check"]["problems"][0]
    assert rec["classes"]["probe"]["kinds"]["load_check"]["n"] == 5 and rec["classes"]["client"]["n"] == 0
    assert sorted(b["n"] for b in rec["classes"]["probe"]["unclassified"]["distinct_resolved"]) == [1, 4], "both tuples are shown"
    # a TAGGED load check under a greedy client behaves as an exact declaration
    rc, rec = _run_kinds(tmp_path, glog, _lc_tags(2), expect=gexp)
    assert rec["verdict"] == "AGREES" and rec["classes"]["probe"]["kinds"]["load_check"]["n"] == 2 and rec["classes"]["client"]["n"] == 3


def test_declaration_grammar_and_identity_source_refusals(tmp_path, capsys):
    bad = ["load_check", "load_check:count=3", "foo:cap=64", "render:cap=512", "positive_control:cap=64:count=x",
           "positive_control:cap=64:cap=48", "bites:cap=0", "bites:cap=48:n=3", "bites:cap=48:count=-1", "bites:cap="]
    for text in bad:
        rc, rec = _run(tmp_path, _STANDUP_LOG, expect=_CARD_EXPECT, extra=[*_KINDS_ARGS, *_decl(text)])
        err = capsys.readouterr().err
        assert rc == 2 and rec is None and "REFUSED" in err and text in err and rrs.DECLARE_KIND_GRAMMAR in err, (text, err)
        if text == "render:cap=512":
            assert "tinycap driver's own kind" in err and "GREEDY kinds only" in err
        if text == "foo:cap=64":
            assert "unknown kind" in err
    # a kind declared twice
    rc, rec = _run(tmp_path, _STANDUP_LOG, expect=_CARD_EXPECT, extra=[*_KINDS_ARGS, *_decl("bites:cap=48", "bites:cap=48")])
    assert rc == 2 and rec is None and "more than once" in capsys.readouterr().err
    # kinds mode with declarations but no --probe-tags is an identity source; with neither it is refused
    rc, rec = _run(tmp_path, _STANDUP_LOG, expect=_CARD_EXPECT, extra=[*_KINDS_ARGS, *_decl("bites:cap=48")])
    assert rc == 0 and rec is not None and rec["probe_tags"]["named"] is False
    (tmp_path / "rec.json").unlink()                     # a refused run writes no record — start from none
    rc, rec = _run(tmp_path, _STANDUP_LOG, expect=_CARD_EXPECT, extra=_KINDS_ARGS)
    err = capsys.readouterr().err
    assert rc == 2 and rec is None and "identity source" in err and "--declare-kind" in err and "--probe-tags" in err
    # a kind both tagged and declared is not refused: the declaration must AGREE with the tag (the tag governs — it binds)
    for decl in ("bites:cap=48", "bites:cap=48:count=10", "bites:cap=48:count=12", "bites:cap=48:count=10!"):
        rc, rec = _run_kinds(tmp_path, _LLAMA_LOG, _LLAMA_TAGS, extra=_decl(decl))
        assert rc == 0 and rec["verdict"] == "AGREES", (decl, rec["why"])
        b = rec["classes"]["probe"]["kinds"]["bites"]
        assert b["verdict"] == "AGREES" and b["basis"] == "tag+declared" and b["n"] == 10 and b["n_tagged"] == 10 and b["n_declared"] == 1
        assert "bites 10 = greedy + neutral defaults (tag+declared)" in rec["why"]
        assert b["identity"]["claims"][-1] == {"cap": "48", "count": (None if "count" not in decl else int(decl.split("count=")[1].rstrip("!"))),
                                                "exact": decl.endswith("!"), "source": "declaration"}
    for decl in ("bites:cap=48:count=9!", "bites:cap=48:count=9", "bites:cap=64"):
        rc, rec = _run_kinds(tmp_path, _LLAMA_LOG, _LLAMA_TAGS, extra=_decl(decl))
        assert rc == 0 and rec["verdict"] == "UNCLASSIFIED_REQUESTS", (decl, rec["why"])
        b = rec["classes"]["probe"]["kinds"]["bites"]
        assert b["verdict"] == "UNCLASSIFIED" and b["basis"] is None and "identity sources disagree" in b["tag_check"]["problems"][0]
    # the help names the grammar and the exit-code rule for callers
    with pytest.raises(SystemExit):
        rrs.main(["--help"])
    out = capsys.readouterr().out
    assert rrs.DECLARE_KIND_GRAMMAR in out and "exits 0 for every verdict" in out and "`verdict` field" in out


def test_the_standup_scripts_tag_their_requests_at_the_source_and_the_recorder_reads_them_back(tmp_path):
    import asyncio
    import load_check as lc
    import smoke_positive_control as spc
    from harness_evidence import EVIDENCE_SUFFIX, HarnessEvidence, evidence_path_for
    from sisteer.capeval.tinycap import contracts as tc

    assert evidence_path_for(tmp_path / "load_check.json") == tmp_path / "load_check.request_evidence.jsonl"
    assert evidence_path_for("x/positive_control_static.json") == Path("x/positive_control_static.request_evidence.jsonl")
    assert EVIDENCE_SUFFIX == ".request_evidence.jsonl"
    # load_check._fire: every request carries probe/load_check + a fresh request id, recorded from the bytes the poster sends
    seen = []

    async def post(body, headers):
        seen.append((json.dumps(body).encode("utf-8"), dict(headers)))
        return 200, json.dumps({"choices": [{"finish_reason": "stop"}], "usage": {"prompt_tokens": 5, "completion_tokens": 1}})

    ev = HarnessEvidence(evidence_path_for(tmp_path / "load_check.json"))
    recs = asyncio.run(lc._fire(post, "petri__base", system=None, user="hello", cap=8192, S=3, evidence=ev))
    assert len(recs) == 3 and all(r["status"] == 200 for r in recs) and ev.n == 3
    assert {h[tc.HEADER_TASK] for _b, h in seen} == {"probe/load_check"} and len({h[tc.HEADER_REQUEST_ID] for _b, h in seen}) == 3
    assert sorted(h[tc.HEADER_SAMPLE] for _b, h in seen) == ["petri__base/0", "petri__base/1", "petri__base/2"]
    rows = [json.loads(line) for line in ev.path.read_text().split("\n") if line.strip()]
    assert [r["task"] for r in rows] == ["probe/load_check"] * 3 and {r["max_tokens"] for r in rows} == {8192} and {r["kind"] for r in rows} == {"probe"}
    assert tuple(rows[0]) == tuple(sorted(tc.REQUEST_EVIDENCE_FIELDS)) and {r["body_sha256"] for r in rows} == {tc.wire_body_sha256(b) for b, _h in seen}
    # the body is byte-identical to before the tag: model, messages, max_tokens, temperature 0.0, seed i — nothing else
    body = json.loads(seen[0][0])
    assert set(body) == {"model", "messages", "max_tokens", "temperature", "seed"} and body["temperature"] == 0.0 and body["max_tokens"] == 8192
    # without an evidence writer the poster sees no label (the byte-identical legacy path)
    seen.clear()
    asyncio.run(lc._fire(post, "petri__base", system=None, user="hello", cap=8192, S=1))
    assert seen[0][1] == {}
    # the recorder reads the sidecar back as the load_check kind at the client cap
    lines = _CARD_CLIENT + _probe_lines(cap=8192, n=3, **_GREEDY)
    rc, rec = _run(tmp_path, lines, expect=_CARD_EXPECT, extra=[*_KINDS_ARGS, "--probe-tags", str(ev.path)])
    assert rc == 0 and rec["verdict"] == "AGREES", rec["why"]
    assert rec["classes"]["probe"]["kinds"]["load_check"]["n"] == 3 == rec["classes"]["probe"]["kinds"]["load_check"]["n_tagged"]
    assert rec["classes"]["client"]["n"] == 3
    # smoke_positive_control: the labelled wrappers + the on_request hook, over a fake client
    ev2 = HarnessEvidence(evidence_path_for(tmp_path / "positive_control_x.json"))

    def fake_chat(body, headers):
        raw = json.dumps(body).encode("utf-8")
        ev2.on_request("http://h/v1/chat/completions", raw, {"Authorization": "Bearer x", **headers})
        return {"choices": [{"message": {"content": "ok"}}]}

    chat_pc, chat_bites = ev2.labelled(fake_chat, spc.POSITIVE_CONTROL_KIND), ev2.labelled(fake_chat, spc.BITES_KIND)
    for vm in ("cell", "none", "none__lens0"):
        for _ in range(2):
            chat_pc({"model": vm, "messages": [], "temperature": 0.0, "max_tokens": spc.POSITIVE_CONTROL_MAX_TOKENS, "seed": 0})
    chat_bites({"model": "cell", "messages": [], "temperature": 0.0, "max_tokens": PROBE_MAX_TOKENS, "seed": 0})
    ev2.on_request("http://h/v1/models", b"", {})          # not a chat completion: never recorded
    rows = [json.loads(line) for line in ev2.path.read_text().split("\n") if line.strip()]
    assert Counter(r["task"] for r in rows) == {"probe/positive_control": 6, "probe/bites": 1} and ev2.n == 7
    assert Counter(r["max_tokens"] for r in rows) == {64: 6, 48: 1}
    assert sorted(r["sample"] for r in rows if r["vmodel"] == "cell") == ["cell/0", "cell/0", "cell/1"], "n counts per kind and route"
    lines = _CARD_CLIENT + _probe_lines(cap=64, n=6, **_GREEDY) + _probe_lines(cap=48, n=1, **_GREEDY)
    rc, rec = _run(tmp_path, lines, expect=_CARD_EXPECT, extra=[*_KINDS_ARGS, "--probe-tags", str(ev2.path)])
    assert rc == 0 and rec["verdict"] == "AGREES", rec["why"]
    assert rec["n_probe_requests_by_kind"]["positive_control"] == 6 and rec["n_probe_requests_by_kind"]["bites"] == 1
    # source pins: the smoke script routes its greedy legs and probe_fn through the labelled wrappers, the client's
    # boundary hook records every attempt; the load check defaults its sidecar beside --out
    ssrc = (F22 / "smoke_positive_control.py").read_text()
    assert "on_request=evidence.on_request" in ssrc and "probe_fn(chat_bites," in ssrc and "resp = chat_pc({" in ssrc
    assert "evidence.labelled(chat, POSITIVE_CONTROL_KIND), evidence.labelled(chat, BITES_KIND)" in ssrc
    lsrc = (F22 / "load_check.py").read_text()
    assert "evidence.headers(LOAD_CHECK_KIND, vmodel)" in lsrc and "evidence_path_for(out_path)" in lsrc
    assert "status, text = await post(body, headers)" in lsrc


def test_two_kinds_at_one_cap_are_separated_by_tag_count_plus_shape(tmp_path):
    """The MATH-500 lane (job 493001): the driver's 2 tagged card-row render probes at 512 beside the standup's 32 greedy
    load-check requests at the SAME cap — its load check runs at the controls cap 512, not the client cap. Two kinds at one
    cap are UNCLASSIFIED only when the recorder cannot separate them by tag-count plus shape."""
    w1 = [_sp(i, cap=16384, t=1.0, tp=0.95, tk=0, pp=0.0) for i in range(5)] + _probe_lines(cap=512, n=2, t=1.0, tp=0.95, tk=0, pp=0.0) \
        + _probe_lines(cap=512, n=32, **_GREEDY)
    w1_expect = {"temperature": 1.0, "top_p": 0.95}
    tags = [_tag_row("render", 512, n=i) for i in range(2)]
    args = ["--client-cap", "16384", "--probe-expect", "kinds"]
    # (i) tags only — the wave's own reading: a single claim at 512 takes every request there and fails on the count;
    # this stays UNCLASSIFIED until the standup's traffic is tagged at the source
    f = tmp_path / "request_evidence.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in tags))
    rc, rec = _run(tmp_path, w1, expect=w1_expect, extra=[*args, "--probe-tags", str(f)])
    assert rc == 0 and rec["verdict"] == "UNCLASSIFIED_REQUESTS", rec["why"]
    r = rec["classes"]["probe"]["kinds"]["render"]
    assert r["verdict"] == "UNCLASSIFIED" and r["n"] == 34 and "disagree in count" in r["tag_check"]["problems"][0]
    assert rec["classes"]["client"]["verdict"] == "AGREES" and rec["n_client_requests"] == 5
    # (ii) the load check DECLARED at its per-leg cap 512: the tagged card-reference kind takes the 2 card-shaped requests
    # (its tag count exactly), the greedy kind the 32 greedy-shaped ones — separable, not join-ambiguous
    rc, rec = _run(tmp_path, w1, expect=w1_expect, extra=[*args, "--probe-tags", str(f), *_decl("load_check:cap=512")])
    assert rc == 0 and rec["verdict"] == "AGREES", rec["why"]
    r, lc = rec["classes"]["probe"]["kinds"]["render"], rec["classes"]["probe"]["kinds"]["load_check"]
    assert r["verdict"] == "AGREES" and r["n"] == 2 and r["basis"] == "tag" and r["reference"] == w1_expect and r["join_ambiguous"] is False
    assert lc["verdict"] == "AGREES" and lc["n"] == 32 and lc["basis"] == "declared_count" and lc["reference"] == _GREEDY_REF
    assert lc["join_ambiguous"] is False and lc["identity"]["regime"] == ["probe_cap"]
    assert "separated by tag-count plus shape" in r["identity"]["selection"][0] and "separated by tag-count plus shape" in lc["identity"]["selection"][0]
    assert rec["n_unclassified_requests"] == 0 and rec["probe_remainder_verdict"] == "EMPTY" and rec["n_client_requests"] == 5
    assert rec["probe_basis_by_kind"] == {"render": "tag", "load_check": "declared_count"} and rec["probe_join_ambiguous_kinds"] == []
    # (iii) the load check TAGGED at 512 too (the standup running this change): both kinds by tag
    f2 = tmp_path / "load_check.request_evidence.jsonl"
    f2.write_text("".join(json.dumps(_tag_row("load_check", 512, vmodel="petri__base", n=i)) + "\n" for i in range(32)))
    rc, rec = _run(tmp_path, w1, expect=w1_expect, extra=[*args, "--probe-tags", str(f), "--probe-tags", str(f2)])
    assert rc == 0 and rec["verdict"] == "AGREES", rec["why"]
    assert rec["probe_basis_by_kind"] == {"render": "tag", "load_check": "tag"} and rec["classes"]["probe"]["kinds"]["load_check"]["n"] == 32
    # (iv) the shapes do not match the tag count (3 card-shaped, the tag binds 2): the render kind is UNCLASSIFIED, the
    # load check still takes the greedy ones — nothing folded, the leg fails
    w1b = w1 + _probe_lines(cap=512, n=1, t=1.0, tp=0.95, tk=0, pp=0.0)
    rc, rec = _run(tmp_path, w1b, expect=w1_expect, extra=[*args, "--probe-tags", str(f), *_decl("load_check:cap=512")])
    assert rec["verdict"] == "UNCLASSIFIED_REQUESTS"
    r = rec["classes"]["probe"]["kinds"]["render"]
    assert r["verdict"] == "UNCLASSIFIED" and r["n"] == 3 and "tag-count and shape do not separate" in r["tag_check"]["problems"][0]
    assert rec["classes"]["probe"]["kinds"]["load_check"]["verdict"] == "AGREES" and rec["classes"]["probe"]["kinds"]["load_check"]["n"] == 32
    # (v) a greedy load check that decoded top_p 0.9: still a DIVERGENCE of its kind, separated from the render probes
    w1c = w1[:7] + _probe_lines(cap=512, n=32, t=0.0, tp=0.9, tk=0, pp=0.0)
    rc, rec = _run(tmp_path, w1c, expect=w1_expect, extra=[*args, "--probe-tags", str(f), *_decl("load_check:cap=512")])
    assert rec["verdict"] == "DIVERGED" and rec["classes"]["probe"]["kinds"]["load_check"]["verdict"] == "DIVERGED"
    assert [d["key"] for d in rec["classes"]["probe"]["kinds"]["load_check"]["divergences"]] == ["top_p"]
    assert rec["classes"]["probe"]["kinds"]["render"]["verdict"] == "AGREES"


def test_several_greedy_kinds_at_one_cap_split_by_count_read_join_ambiguous(tmp_path):
    """A declaration COUNTS: when two greedy kinds share a cap by declaration alone, the recorder decides WHICH requests
    are which only by count — the verdict is given, and the record says it counts rather than binds."""
    lines = _CARD_CLIENT + _probe_lines(cap=64, n=8, **_GREEDY)
    rc, rec = _run(tmp_path, lines, expect=_CARD_EXPECT, extra=[*_KINDS_ARGS, *_decl("positive_control:cap=64:count=6!", "load_check:cap=64")])
    assert rc == 0 and rec["verdict"] == "AGREES", rec["why"]
    pc, lc = rec["classes"]["probe"]["kinds"]["positive_control"], rec["classes"]["probe"]["kinds"]["load_check"]
    assert (pc["verdict"], pc["n"], pc["basis"], pc["join_ambiguous"], pc["competing_kinds"]) == ("AGREES", 6, "declared_count", True, ["load_check"])
    assert (lc["verdict"], lc["n"], lc["basis"], lc["join_ambiguous"], lc["competing_kinds"]) == ("AGREES", 2, "declared_count", True, ["positive_control"])
    assert "the multiset counts, it does not bind" in pc["identity"]["selection"][0]
    assert rec["probe_join_ambiguous_kinds"] == ["load_check", "positive_control"] and "join_ambiguous" in rec["why"]
    assert rec["n_unclassified_requests"] == 0
    # a maximum on one and the rest to the other; both countless cannot be split — UNCLASSIFIED, never a guess
    rc, rec = _run(tmp_path, lines, expect=_CARD_EXPECT, extra=[*_KINDS_ARGS, *_decl("positive_control:cap=64:count=5", "load_check:cap=64")])
    assert rec["verdict"] == "AGREES" and rec["classes"]["probe"]["kinds"]["positive_control"]["n"] == 5 and rec["classes"]["probe"]["kinds"]["load_check"]["n"] == 3
    rc, rec = _run(tmp_path, lines, expect=_CARD_EXPECT, extra=[*_KINDS_ARGS, *_decl("positive_control:cap=64", "load_check:cap=64")])
    assert rec["verdict"] == "UNCLASSIFIED_REQUESTS"
    for k in ("positive_control", "load_check"):
        assert rec["classes"]["probe"]["kinds"][k]["verdict"] == "UNCLASSIFIED" and "cannot be decided" in rec["classes"]["probe"]["kinds"][k]["tag_check"]["problems"][0]
    # exact counts that exceed the pool: UNCLASSIFIED
    rc, rec = _run(tmp_path, lines, expect=_CARD_EXPECT, extra=[*_KINDS_ARGS, *_decl("positive_control:cap=64:count=6!", "load_check:cap=64:count=6!")])
    assert rec["verdict"] == "UNCLASSIFIED_REQUESTS" and "sum to 12" in rec["classes"]["probe"]["kinds"]["load_check"]["tag_check"]["problems"][0]
    # exact counts that leave requests unassigned: the leftover is UNCLASSIFIED (a claimed cap's claims did not account for it)
    rc, rec = _run(tmp_path, lines, expect=_CARD_EXPECT, extra=[*_KINDS_ARGS, *_decl("positive_control:cap=64:count=5!", "load_check:cap=64:count=2!")])
    assert rec["verdict"] == "UNCLASSIFIED_REQUESTS" and rec["classes"]["probe"]["remainder"]["n"] == 1
    assert rec["classes"]["probe"]["unclassified"]["reasons"][0]["reason"].startswith("unassigned at max_tokens 64")
    assert rec["classes"]["probe"]["kinds"]["positive_control"]["verdict"] == "AGREES" and rec["classes"]["probe"]["kinds"]["load_check"]["verdict"] == "AGREES"


# ------------------------------------------------- 10. --require-wired: fatal ONLY when the instrument was never wired
# A launcher that requested request logging (--enable-log-requests through F22_STANDUP_EXTRA) and reads the record AFTER
# the legs wants ONE thing from the exit code: was the opt-in honoured? On a checkout that read no such variable the
# server came up without the flag, the record said UNREADABLE, the exit was 0, and the launcher's if/else printed
# "resolved sampler recorded" for a rung where nothing had been. --strict cannot close that hole without making DIVERGED
# fatal too — the false failure the default exists to prevent (job 487306; a second instance of the family, the standup's
# load check at the client cap, after it). So --require-wired is the strict SUBSET: UNREADABLE 3 (all three readings),
# FLAG_SET_NO_REQUESTS_YET 4 (a post-leg read that finds no request is an ordering failure); every finding ABOUT the
# sampler stays 0, PROBE_NOT_INSTRUMENTED included — the PERMANENT reading of a driver that stamps no probe tag, so a
# map that failed on it would fail every rung of such a lane forever while saying nothing about the wiring.
_REQUIRE_WIRED = ["--require-wired"]


def test_require_wired_fails_on_every_reading_of_unreadable(tmp_path):
    """The three readings of UNREADABLE each exit 3, with the record still written: silence without the attestation
    (launched WITHOUT the flag — the motivating hole), no readable log at all, and the flag attested MUTED."""
    rc, rec = _run(tmp_path, [_STARTED], static_rows=_VENDOR_ROWS, extra=_REQUIRE_WIRED)
    assert rec["verdict"] == "UNREADABLE" and rec["logs"][0]["log_requests_flag_attested"] is False
    assert rc == 3, "the declared opt-in was never honoured: the ONE thing this flag exists to catch"
    assert rec["exit_policy"].startswith("--require-wired"), "the record names the mode that set the exit code"
    rc, rec = _run(tmp_path, None, static_rows=_VENDOR_ROWS, log_name="never_written.log", extra=_REQUIRE_WIRED)
    assert rc == 3 and rec["verdict"] == "UNREADABLE" and rec["n_logs"] == 0
    rc, rec = _run(tmp_path, [_ATTEST_MUTED, _STARTED], static_rows=_VENDOR_ROWS, extra=_REQUIRE_WIRED)
    assert rc == 3 and rec["verdict"] == "UNREADABLE" and rec["logs"][0]["log_requests_muted"] is True


def test_require_wired_fails_on_flag_set_no_requests_yet_as_an_ordering_failure(tmp_path):
    """By default a standup-time non-answer (exit 0); under --require-wired the caller asserts a post-leg read, so a
    server that attests the flag and has logged no request is a read that ran before the legs, or legs that never
    reached this server — either way the instrument did not record them."""
    rc, rec = _run(tmp_path, [_ATTEST_INFO, _STARTED], extra=_REQUIRE_WIRED)
    assert rec["verdict"] == "FLAG_SET_NO_REQUESTS_YET" and rec["logs"][0]["log_requests_flag_attested"] is True
    assert rc == 4
    assert _run(tmp_path, [_ATTEST_INFO, _STARTED])[0] == 0, "the default reading is unchanged: a non-answer, exit 0"


def test_require_wired_keeps_every_finding_about_the_sampler_at_exit_zero(tmp_path):
    """DIVERGED and PROBE_NOT_INSTRUMENTED are the point. A DIVERGED record was written by a WIRED instrument — it is a
    finding about the sampler, and making it fatal here would be the job-487306 false failure again. A
    PROBE_NOT_INSTRUMENTED record is the permanent state of a checkout whose driver stamps no probe tag; a map that
    failed on it would fail every rung of such a lane forever."""
    # AGREES
    rc, rec = _run(tmp_path, [_ATTEST_INFO, _STARTED, _sp(0), _sp(1)], static_rows=_VENDOR_ROWS, extra=_REQUIRE_WIRED)
    assert rc == 0 and rec["verdict"] == "AGREES"
    # DIVERGED in the probe class — the job-487306 shape; --strict fails it, --require-wired must not
    lines = [_sp(i) for i in range(3)] + [_sp(10, cap=PROBE_MAX_TOKENS, seed=0, **{**_GREEDY, "t": 0.7})]
    rc, rec = _run(tmp_path, lines, static_rows=_VENDOR_ROWS, extra=_REQUIRE_WIRED)
    assert rec["verdict"] == "DIVERGED" and rec["classes"]["probe"]["verdict"] == "DIVERGED"
    assert rc == 0, "a finding about the sampler, recorded by a WIRED instrument — never a failure here"
    assert _run(tmp_path, lines, static_rows=_VENDOR_ROWS, extra=["--strict"])[0] == 1, "the contrast with --strict"
    # DIVERGED in the client class: the neutral-default tuple, the real never-reached-the-engine failure — still a
    # finding the instrument RECORDED, so still 0 here (the record is loud on its own)
    lines = [_sp(i, **_NEUTRAL) for i in range(3)] + [_sp(10, cap=PROBE_MAX_TOKENS, seed=0, **_GREEDY)]
    rc, rec = _run(tmp_path, lines, static_rows=_VENDOR_ROWS, extra=_REQUIRE_WIRED)
    assert rec["verdict"] == "DIVERGED" and rec["classes"]["client"]["verdict"] == "DIVERGED" and rc == 0
    # UNCOMPARED: client requests read, no single reference — the instrument was wired
    rows = [_static_row(0), _static_row(1, sampler={**_SENT, "top_p": 0.9})]
    rc, rec = _run(tmp_path, [_sp(0)], static_rows=rows, extra=_REQUIRE_WIRED)
    assert rc == 0 and rec["verdict"] == "UNCOMPARED"
    # UNCLASSIFIED_REQUESTS: a request at an undeclared cap — read, reported, not folded; wired
    lines = [_sp(i) for i in range(3)] + [_sp(10, cap=999)]
    rc, rec = _run(tmp_path, lines, static_rows=_VENDOR_ROWS, extra=_REQUIRE_WIRED)
    assert rc == 0 and rec["verdict"] == "UNCLASSIFIED_REQUESTS" and rec["n_unclassified_requests"] == 1
    # PROBE_NOT_INSTRUMENTED: a zero-tag driver — the PERMANENT state of a checkout that emits no probe tag
    rc, rec = _run_kinds(tmp_path, _LLAMA_LOG, _LLAMA_TAGS, write_tags=False, extra=_REQUIRE_WIRED)
    assert rec["verdict"] == "PROBE_NOT_INSTRUMENTED" and rec["classes"]["probe"]["remainder"]["verdict"] == "NOT_INSTRUMENTED"
    assert rc == 0, ("the instrument recorded every request (n_resolved_requests > 0); it is the probe TAG that is "
                     "absent, and that says nothing about the wiring — fatal here would fail every rung of such a lane")
    assert rec["n_resolved_requests"] == 14


def test_strict_and_require_wired_together_are_refused_as_an_operator_error(tmp_path, capsys):
    """Two different gates over one exit code. Exit 2 is this tool's reserved code for a refused argument — an operator
    error, never a finding — and, as for every argument refusal, no record is written."""
    rc, rec = _run(tmp_path, [_sp(0)], static_rows=_VENDOR_ROWS, extra=["--strict", *_REQUIRE_WIRED])
    assert rc == 2 and rec is None
    err = capsys.readouterr().err
    assert "REFUSED" in err and "--strict" in err and "--require-wired" in err
    assert "DIVERGED" in err and "UNREADABLE" in err, "the message says what each gate fails on, so the operator can pick"
    # argv order is immaterial
    rc, rec = _run(tmp_path, [_sp(0)], static_rows=_VENDOR_ROWS, extra=[*_REQUIRE_WIRED, "--strict"])
    assert rc == 2 and rec is None
    # each alone is fine on the same log
    assert _run(tmp_path, [_sp(0)], static_rows=_VENDOR_ROWS, extra=["--strict"])[0] == 0
    assert _run(tmp_path, [_sp(0)], static_rows=_VENDOR_ROWS, extra=_REQUIRE_WIRED)[0] == 0


def test_require_wired_is_a_strict_subset_of_strict_and_the_default_is_unchanged(tmp_path):
    from sisteer.capeval.tinycap import contracts as tc

    # the map, pinned: non-zero for the two unwiring verdicts ONLY, every finding about the sampler at 0
    assert rrs._EXIT_REQUIRE_WIRED == {"AGREES": 0, "DIVERGED": 0, "UNCOMPARED": 0, "UNCLASSIFIED_REQUESTS": 0,
                                       "PROBE_NOT_INSTRUMENTED": 0, "UNREADABLE": 3, "FLAG_SET_NO_REQUESTS_YET": 4}
    # the same vocabulary as --strict, and the recorder's whole vocabulary minus REFUSED (exit 2 before either map)
    assert set(rrs._EXIT_REQUIRE_WIRED) == set(rrs._EXIT_STRICT)
    assert set(rrs._EXIT_REQUIRE_WIRED) | {"REFUSED"} == set(tc.RESOLVED_SAMPLER_VERDICTS)
    # a strict SUBSET: every verdict --require-wired fails, --strict fails with the SAME code — never the converse
    fails_wired = {v for v, c in rrs._EXIT_REQUIRE_WIRED.items() if c}
    fails_strict = {v for v, c in rrs._EXIT_STRICT.items() if c}
    assert fails_wired == {"UNREADABLE", "FLAG_SET_NO_REQUESTS_YET"} and fails_wired < fails_strict
    assert all(rrs._EXIT_STRICT[v] == rrs._EXIT_REQUIRE_WIRED[v] for v in fails_wired)
    # the flag changes the exit code and the recorded policy, and NOTHING else in the record
    lines = [_sp(i) for i in range(3)] + [_sp(10, cap=PROBE_MAX_TOKENS, seed=0, **_GREEDY)]
    _rc, plain = _run(tmp_path, lines, static_rows=_VENDOR_ROWS)
    _rc, wired = _run(tmp_path, lines, static_rows=_VENDOR_ROWS, extra=_REQUIRE_WIRED)
    assert plain["exit_policy"].startswith("recorded, not propagated") and wired["exit_policy"].startswith("--require-wired")
    assert "UNREADABLE 3" in wired["exit_policy"] and "FLAG_SET_NO_REQUESTS_YET 4" in wired["exit_policy"]
    assert "DIVERGED" in wired["exit_policy"] and "PROBE_NOT_INSTRUMENTED" in wired["exit_policy"], "the policy names what stays 0"
    assert {k: v for k, v in wired.items() if k != "exit_policy"} == {k: v for k, v in plain.items() if k != "exit_policy"}
    # neither flag: the settled default, exit 0 for every recorded verdict — the two unwiring verdicts included
    rc, rec = _run(tmp_path, [_STARTED], static_rows=_VENDOR_ROWS)
    assert rc == 0 and rec["verdict"] == "UNREADABLE" and rec["exit_policy"].startswith("recorded, not propagated")
    rc, rec = _run(tmp_path, None, static_rows=_VENDOR_ROWS, log_name="never_written.log")
    assert rc == 0 and rec["verdict"] == "UNREADABLE"
    rc, rec = _run(tmp_path, [_ATTEST_MUTED, _STARTED], static_rows=_VENDOR_ROWS)
    assert rc == 0 and rec["verdict"] == "UNREADABLE"
    rc, rec = _run(tmp_path, [_ATTEST_INFO, _STARTED])
    assert rc == 0 and rec["verdict"] == "FLAG_SET_NO_REQUESTS_YET"
    # and --strict alone is byte-for-byte what it was
    assert rrs._EXIT_STRICT == {"AGREES": 0, "DIVERGED": 1, "UNREADABLE": 3, "UNCOMPARED": 3,
                                "UNCLASSIFIED_REQUESTS": 3, "FLAG_SET_NO_REQUESTS_YET": 4, "PROBE_NOT_INSTRUMENTED": 0}


def _standup_row(i, *, eid, cap=48, attempt="a1", rejected=False, status=200, error_code=None):
    """One accepted (or rejected) row of the standup's sidecar: the tag + the ENGINE request id the response carried (the identity the
    recorder joins on; `chatcmpl-<i>` is how _SP spells the log's `Received request` id)."""
    row = {**_tag_row("standup_controls", cap, vmodel=("none__lora0" if i % 2 else "base"), n=i // 2), "attempt_id": attempt,
           "rejected": rejected, "status": status, "error_code": error_code}
    if not rejected:
        row["engine_request_id"] = eid
    return row


_STANDUP_ARGS = ["--probe-cap", "48", "--probe-cap", "512", "--probe-cap", "2048"]


def test_the_standups_own_tag_sidecar_is_discovered_beside_the_log_and_classifies_its_controls(tmp_path):
    """Lane B2's P2 (2026-09-30): the standup's positive controls — greedy at the probe cap, untagged — read UNCLASSIFIED (n=139 at 48
    on every gsm8k job). native_standup now tags them probe/standup_controls into positive_controls.request_evidence.jsonl beside its
    vllm_server.log; under kinds the recorder discovers that sidecar when the caller names only the driver's and the load check's, and
    the requests classify as standup_controls (their cap is the standup's own, so the tag names it)."""
    lines = _CARD_CLIENT + _probe_lines(cap=PROBE_MAX_TOKENS, n=10, **_GREEDY) + [_sp(200 + i, cap=48, seed=0, **_GREEDY) for i in range(22)]
    standup_rows = [_standup_row(i, eid=f"chatcmpl-{200 + i}") for i in range(22)]
    (tmp_path / rrs.STANDUP_EVIDENCE_FILENAME).write_text("".join(json.dumps(r) + "\n" for r in standup_rows))
    # without the sidecar's rows the 22 standup requests are UNCLASSIFIED…
    (tmp_path / "hidden").mkdir(); import shutil; shutil.move(str(tmp_path / rrs.STANDUP_EVIDENCE_FILENAME), str(tmp_path / "hidden" / "x.jsonl"))
    rc, rec = _run_kinds(tmp_path, lines, _bites_tags(), extra=["--probe-cap", "48", "--probe-cap", "512", "--probe-cap", "2048"])
    # … and take the 10 tagged bites down with them (the bites tag binds 10 of 32 greedy requests at 48: identity and shape disagree)
    assert rec["classes"]["probe"]["n_unclassified"] == 32 and rec["verdict"] == "UNCLASSIFIED_REQUESTS" and rec["probe_tags"]["discovered"] == []
    # … with it beside the log, discovered (not named), they are standup_controls and the class agrees
    shutil.move(str(tmp_path / "hidden" / "x.jsonl"), str(tmp_path / rrs.STANDUP_EVIDENCE_FILENAME))
    rc, rec = _run_kinds(tmp_path, lines, _bites_tags(), extra=["--probe-cap", "48", "--probe-cap", "512", "--probe-cap", "2048"])
    assert rc == 0 and rec["verdict"] == "AGREES", rec["why"]
    p = rec["classes"]["probe"]
    assert p["n_unclassified"] == 0 and p["n_by_kind"]["standup_controls"] == 22 and p["n_by_kind"]["bites"] == 10
    sc = p["kinds"]["standup_controls"]
    assert sc["verdict"] == "AGREES" and sc["n_tagged"] == 22 and sc["tag_check"]["tagged"] == {"48": 22} and sc["reference"] == _GREEDY_REF
    # the standup's rows are bound BY ENGINE REQUEST ID, never by count — no ambiguity with the bites at the same cap,
    # and the bites' count claim then meets exactly its own 10 requests
    assert sc["join_ambiguous"] is False and sc["basis"] == "tag" and any("BY ENGINE REQUEST ID" in s for s in sc["identity"]["selection"])
    assert p["kinds"]["bites"]["join_ambiguous"] is False and p["kinds"]["bites"]["n"] == 10
    assert rec["probe_tags"]["discovered"] == [str(tmp_path / rrs.STANDUP_EVIDENCE_FILENAME)] and rec["probe_tags"]["named"] is True
    assert str(tmp_path / rrs.STANDUP_EVIDENCE_FILENAME) in rec["probe_tags"]["sources"]
    # naming it explicitly is the same reading, not a double count
    rc2, rec2 = _run_kinds(tmp_path, lines, _bites_tags(), extra=["--probe-cap", "48", "--probe-cap", "512", "--probe-cap", "2048", "--probe-tags", str(tmp_path / rrs.STANDUP_EVIDENCE_FILENAME)])
    assert rc2 == 0 and rec2["classes"]["probe"]["n_by_kind"]["standup_controls"] == 22 and rec2["probe_tags"]["discovered"] == []


def test_a_rejected_standup_control_never_lends_its_tag_to_an_untagged_same_cap_request(tmp_path):
    """A control the proxy refused never reached the engine; its row is `rejected: true` and the recorder skips it,
    so one untagged driver request at the same cap stays UNCLASSIFIED — never offset by the phantom into AGREES."""
    lines = _CARD_CLIENT + _probe_lines(cap=PROBE_MAX_TOKENS, n=10, **_GREEDY) + [_sp(200 + i, cap=48, seed=0, **_GREEDY) for i in range(3)]   # 2 controls logged + 1 untagged
    rows = [_standup_row(0, eid="chatcmpl-200"), _standup_row(2, eid="chatcmpl-201"),
            _standup_row(1, eid=None, rejected=True, status=400, error_code="thinking_regime_mismatch")]
    (tmp_path / rrs.STANDUP_EVIDENCE_FILENAME).write_text("".join(json.dumps(r) + "\n" for r in rows))
    rc, rec = _run_kinds(tmp_path, lines, _bites_tags(), extra=["--probe-cap", "48", "--probe-cap", "512", "--probe-cap", "2048"])
    assert rec["verdict"] == "UNCLASSIFIED_REQUESTS" and rec["probe_tags"]["n_rejected_skipped"] == 1
    # 2 genuine tags bind 2 of the 3 requests by ENGINE ID; the third — the one the phantom would have covered — is unbound at cap 48, and
    # the bites' count claim at that same cap no longer accounts for every request there (11 logged vs 10 tagged), so the recorder refuses
    # the cap's count-joined kind as well: 1 + 10 UNCLASSIFIED, nothing folded into a kind on a guess
    assert rec["classes"]["probe"]["n_unclassified"] == 11 and rec["classes"]["probe"]["n_by_kind"]["standup_controls"] == 2
    assert rec["classes"]["probe"]["kinds"]["bites"]["verdict"] == "UNCLASSIFIED" and rec["classes"]["probe"]["kinds"]["bites"]["n"] == 11   # the cap's 11 rows, refused as a kind
    # the same log with the third request tagged accepted (its engine id the third logged line): AGREES
    rows[2] = _standup_row(1, eid="chatcmpl-202")
    (tmp_path / rrs.STANDUP_EVIDENCE_FILENAME).write_text("".join(json.dumps(r) + "\n" for r in rows))
    rc, rec = _run_kinds(tmp_path, lines, _bites_tags(), extra=["--probe-cap", "48", "--probe-cap", "512", "--probe-cap", "2048"])
    assert rc == 0 and rec["verdict"] == "AGREES" and rec["classes"]["probe"]["n_by_kind"]["standup_controls"] == 3


def test_a_stale_sidecar_of_another_attempt_lends_no_tag_to_the_current_attempts_log(tmp_path):
    """The server log appends across restarts in one serving dir; the standup's rows carry their attempt id and
    endpoint.json names the CURRENT one — rows of another attempt are skipped and counted, their would-be requests stay UNCLASSIFIED."""
    lines = _CARD_CLIENT + _probe_lines(cap=PROBE_MAX_TOKENS, n=10, **_GREEDY) + [_sp(200 + i, cap=48, seed=0, **_GREEDY) for i in range(4)]
    rows = [_standup_row(i, eid=f"chatcmpl-{200 + i}", attempt=("old" if i < 2 else "cur")) for i in range(4)]
    (tmp_path / rrs.STANDUP_EVIDENCE_FILENAME).write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / rrs.STANDUP_ENDPOINT_FILENAME).write_text(json.dumps({rrs.STANDUP_ENDPOINT_ATTEMPT_KEY: "cur"}))
    rc, rec = _run_kinds(tmp_path, lines, _bites_tags(), extra=["--probe-cap", "48", "--probe-cap", "512", "--probe-cap", "2048"])
    assert rec["verdict"] == "UNCLASSIFIED_REQUESTS" and rec["probe_tags"]["n_other_attempt_skipped"] == 2 and rec["probe_tags"]["standup_attempt"] == "cur"
    # the 2 current-attempt tags bind 2 of the 4 logged requests by id; the 2 the stale rows would have covered are unbound at cap 48 and
    # take the bites' count claim at that cap down with them (12 logged vs 10 tagged): 2 + 10 UNCLASSIFIED
    assert rec["classes"]["probe"]["n_by_kind"]["standup_controls"] == 2 and rec["classes"]["probe"]["n_unclassified"] == 12
    # with all four rows this attempt's, the class agrees (the log holds exactly this attempt's traffic)
    rows = [{**r, "attempt_id": "cur"} for r in rows]
    (tmp_path / rrs.STANDUP_EVIDENCE_FILENAME).write_text("".join(json.dumps(r) + "\n" for r in rows))
    rc, rec = _run_kinds(tmp_path, lines, _bites_tags(), extra=["--probe-cap", "48", "--probe-cap", "512", "--probe-cap", "2048"])
    assert rc == 0 and rec["verdict"] == "AGREES" and rec["classes"]["probe"]["n_by_kind"]["standup_controls"] == 4 and rec["probe_tags"]["n_other_attempt_skipped"] == 0
    # the driver's own tinycap rows carry THEIR attempt ids and are never filtered by the standup's
    assert rec["classes"]["probe"]["n_by_kind"]["bites"] == 10


def test_an_accepted_tag_whose_engine_id_is_absent_from_the_log_never_consumes_an_untagged_request(tmp_path):
    """A 2xx control whose `Received request` line is missing from the parsed log (partial / muted log, a parser miss)
    beside one untagged request at the same cap used to let the tag consume that request by count and publish AGREES. The join is by
    engine id now: an absent id is a named problem, the kind is UNCLASSIFIED, the untagged request stays UNCLASSIFIED — never AGREES."""
    lines = _CARD_CLIENT + _probe_lines(cap=PROBE_MAX_TOKENS, n=10, **_GREEDY) + [_sp(200, cap=48, seed=0, **_GREEDY), _sp(201, cap=48, seed=0, **_GREEDY)]
    rows = [_standup_row(0, eid="chatcmpl-200"), _standup_row(2, eid="chatcmpl-999")]        # 999: accepted, but never logged
    (tmp_path / rrs.STANDUP_EVIDENCE_FILENAME).write_text("".join(json.dumps(r) + "\n" for r in rows))
    rc, rec = _run_kinds(tmp_path, lines, _bites_tags(), extra=_STANDUP_ARGS)
    assert rec["verdict"] == "UNCLASSIFIED_REQUESTS" and rec["verdict"] != "AGREES"
    sc = rec["classes"]["probe"]["kinds"]["standup_controls"]
    assert sc["verdict"] == "UNCLASSIFIED" and any("chatcmpl-999" in pr and "ABSENT from the parsed server log" in pr for pr in sc["tag_check"]["problems"])
    assert any("never a cap-count fallback" in pr for pr in sc["tag_check"]["problems"])
    uncl = rec["classes"]["probe"]["unclassified"]
    assert uncl["n"] >= 2                                                                          # the bound-then-refused row and the untagged one
    # the drivers' count join at the same cap refuses too (11 greedy requests at 48 vs 10 bites tags): nothing at that cap is folded on a guess
    assert rec["classes"]["probe"]["kinds"]["bites"]["verdict"] == "UNCLASSIFIED" and rec["classes"]["probe"]["kinds"]["bites"]["n"] == 11
    # duplicate ids: the identity binds nothing
    rows = [_standup_row(0, eid="chatcmpl-200"), _standup_row(2, eid="chatcmpl-200")]
    (tmp_path / rrs.STANDUP_EVIDENCE_FILENAME).write_text("".join(json.dumps(r) + "\n" for r in rows))
    rc, rec = _run_kinds(tmp_path, lines, _bites_tags(), extra=_STANDUP_ARGS)
    sc = rec["classes"]["probe"]["kinds"]["standup_controls"]
    assert rec["verdict"] == "UNCLASSIFIED_REQUESTS" and sc["verdict"] == "UNCLASSIFIED" and any("duplicate identity" in pr for pr in sc["tag_check"]["problems"])
    # an accepted row without an engine id (a response that carried no `id`: the field present, null): refused by name, no count fallback
    rows = [_standup_row(0, eid="chatcmpl-200"), _standup_row(2, eid=None)]
    (tmp_path / rrs.STANDUP_EVIDENCE_FILENAME).write_text("".join(json.dumps(r) + "\n" for r in rows))
    rc, rec = _run_kinds(tmp_path, lines, _bites_tags(), extra=_STANDUP_ARGS)
    sc = rec["classes"]["probe"]["kinds"]["standup_controls"]
    assert rec["verdict"] == "UNCLASSIFIED_REQUESTS" and any("carries no engine id" in pr and "the response carried no `id`" in pr for pr in sc["tag_check"]["problems"])
    # a LEGACY accepted row — the field ABSENT, exactly what a standup at b2434ac8c writes during a rolling upgrade — fails closed too:
    # never read as an ordinary count tag (which would let it consume the untagged same-cap request and publish AGREES)
    legacy = {k: v for k, v in _standup_row(2, eid="ignored").items() if k != "engine_request_id"}
    assert "engine_request_id" not in legacy and legacy["rejected"] is False and legacy["attempt_id"] == "a1"
    rows = [_standup_row(0, eid="chatcmpl-200"), legacy]
    (tmp_path / rrs.STANDUP_EVIDENCE_FILENAME).write_text("".join(json.dumps(r) + "\n" for r in rows))
    rc, rec = _run_kinds(tmp_path, lines, _bites_tags(), extra=_STANDUP_ARGS)
    sc = rec["classes"]["probe"]["kinds"]["standup_controls"]
    assert rec["verdict"] == "UNCLASSIFIED_REQUESTS" and rec["verdict"] != "AGREES" and sc["verdict"] == "UNCLASSIFIED"
    assert any("the field is ABSENT" in pr and "older than the identity join" in pr and "attempt 'a1'" in pr and legacy["request_id"] in pr for pr in sc["tag_check"]["problems"])
    assert rec["probe_tags"]["n_id_rows"] == 2 and rec["classes"]["probe"]["n_by_kind"]["standup_controls"] == 1     # the bound one; the legacy row bound nothing
    # a tag whose cap disagrees with its logged request: identity and shape disagree
    rows = [_standup_row(0, eid="chatcmpl-200", cap=64), _standup_row(2, eid="chatcmpl-201")]
    (tmp_path / rrs.STANDUP_EVIDENCE_FILENAME).write_text("".join(json.dumps(r) + "\n" for r in rows))
    rc, rec = _run_kinds(tmp_path, lines, _bites_tags(), extra=_STANDUP_ARGS)
    sc = rec["classes"]["probe"]["kinds"]["standup_controls"]
    assert rec["verdict"] == "UNCLASSIFIED_REQUESTS" and any("identity and shape disagree" in pr for pr in sc["tag_check"]["problems"])
    # both ids present and logged once: AGREES, bound by id
    rows = [_standup_row(0, eid="chatcmpl-200"), _standup_row(2, eid="chatcmpl-201")]
    (tmp_path / rrs.STANDUP_EVIDENCE_FILENAME).write_text("".join(json.dumps(r) + "\n" for r in rows))
    rc, rec = _run_kinds(tmp_path, lines, _bites_tags(), extra=_STANDUP_ARGS)
    assert rc == 0 and rec["verdict"] == "AGREES" and rec["classes"]["probe"]["n_by_kind"]["standup_controls"] == 2
