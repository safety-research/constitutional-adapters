"""Coherence-canary tests (indicator metrics on
synthetic degenerate/healthy generations; threshold derivation from 6
synthetic base runs; the >=2-indicator rule; the rerun-confirmation
state machine). Base lane, no model needed — everything here is pure.

Thresholds are calibration-owned (§4.4/§7.1): derivation delegates to
stats.calibrate.canary_thresholds_from_values; these tests pin the rule
(worst base value + p99 null spread in the harmful direction) through
tier0's consuming adapter.

canary_values consumes GenOuts and shares ONE per-prompt/aggregate
implementation with the production ``_run_canary`` path
(canary_gen_stats + canary_values_from_stats — simplify item 4); the
raw-token-id / ended_with_eos semantics are the r1 finding-9 fixes."""

import math

import pytest

from sisteer.capeval.serving import GenOut
from sisteer.capeval.tiers import tier0 as t0

IND = t0.CANARY_INDICATORS


#: the first id a word is given by :func:`_word_ids`; above every explicit id the degenerate cases pass (3, 7) and the EOS (99)
_WORD_ID_BASE = 100


def _word_ids(words):
    """Deterministic token ids for ``words``: a word -> id table built in order of FIRST APPEARANCE, so two words collide only
    when they are the same word. This used to be ``hash(w) % 1000``, and CPython randomizes str hashing
    per process, so two adjacent healthy words collided mod 1000 in ~2.75 % of fresh processes (11 of 400 PYTHONHASHSEED values,
    e.g. 3, 18, 38) and ``max_rep_run_p95`` read 2.0 on prose with no repeat -- a gate flake that was blamed on an unrelated landing
    before it was traced here. A per-process hash is not a property of the text; the table is."""
    table = {}
    return [table.setdefault(w, _WORD_ID_BASE + len(table)) for w in words]


def _word_gen(text, lp=-1.0, ended=True, ids=None, eos=99):
    """A GenOut whose token ids stand in for the words of ``text`` (any
    hashables work for n-gram runs), terminal EOS appended when ended."""
    toks = ids if ids is not None else _word_ids(text.split())
    toks = list(toks) + ([eos] if ended else [])
    return GenOut(text=text, gen_token_logprobs=[lp] * len(toks),
                  ended_with_eos=ended, gen_token_ids=toks)


# ---------------------------------------------------------------------------
# indicator metrics on synthetic generations
# ---------------------------------------------------------------------------


def test_word_ids_are_a_property_of_the_text_not_of_the_process():
    """Distinct words never share an id (no spurious n-gram run), the same word always does (a real repeat still
    reads as one), and the ids do not depend on the interpreter's hash seed -- the gate flaked on PYTHONHASHSEED 3 / 18 / 38."""
    ids = _word_ids("the river runs east past the mill and the bridge".split())
    assert ids == [100, 101, 102, 103, 104, 100, 105, 106, 100, 107]
    assert len(set(ids)) == len(set("the river runs east past the mill and the bridge".split()))
    healthy = _word_gen("The river runs east past the mill and the bridge.")
    assert t0.max_consecutive_ngram_run(healthy.gen_token_ids[:-1]) == 1
    assert _word_gen("e e e e", ended=False).gen_token_ids == [100, 100, 100, 100]
    # the explicit ids the degenerate cases pass sit below the table's first id, so a word never aliases one of them or the EOS
    assert _WORD_ID_BASE > 99


def test_max_rep_run_degenerate_vs_healthy():
    # unigram degeneration: 'e e e e e e' -> run 6
    assert t0.max_consecutive_ngram_run(["e"] * 6) == 6
    # bigram cycle degeneration: 'a b a b a b' -> the (a,b) gram thrice
    assert t0.max_consecutive_ngram_run(list("ababab")) == 3
    # trigram cycle: 'x y z x y z x y z' -> 3
    assert t0.max_consecutive_ngram_run(["x", "y", "z"] * 3) == 3
    # healthy prose: no consecutive repeats at any n
    assert t0.max_consecutive_ngram_run(
        "the quick brown fox jumps over the lazy dog".split()) == 1
    assert t0.max_consecutive_ngram_run([]) == 0


def test_canary_values_on_degenerate_and_healthy_runs():
    healthy = [
        _word_gen("The river runs east past the mill and the bridge."),
        _word_gen("A good ledger tells a story if you read it slowly."),
    ]
    v = t0.canary_values(healthy, abrupt_min_tokens=3)
    assert set(v) == set(IND)
    assert v["max_rep_run_p95"] == 1.0
    assert v["distinct3"] > 0.9
    assert v["nonascii_frac"] == 0.0
    assert v["abrupt_eos_rate"] == 0.0
    assert v["mean_tok_logprob"] == pytest.approx(-1.0)

    degen = [
        _word_gen("e e e e e e e e e e e e", lp=-0.01, ids=[7] * 12,
                  ended=False),
        _word_gen("ééé", lp=-9.0, ids=[3], ended=True),  # 1 content token
    ]
    d = t0.canary_values(degen, abrupt_min_tokens=3)
    assert d["max_rep_run_p95"] >= 12
    assert d["distinct3"] < 0.2
    assert d["nonascii_frac"] > 0.0
    assert d["abrupt_eos_rate"] == 0.5  # the 1-content-token gen is abrupt


def test_canary_values_refuses_empty_run():
    with pytest.raises(t0.Tier0Error, match="no per-prompt stats"):
        t0.canary_values([])


def test_nonascii_frac():
    assert t0.nonascii_frac("") == 0.0
    assert t0.nonascii_frac("abcd") == 0.0
    assert t0.nonascii_frac("abéé") == 0.5


def test_nonascii_aggregate_is_unweighted_per_prompt_mean():
    """ONE aggregation semantics (simplify item 4): the unweighted mean
    of per-prompt fractions — a char-weighted pool would say 2/6 here."""
    outs = [_word_gen("aaaa"), _word_gen("éé")]
    assert t0.canary_values(outs)["nonascii_frac"] == pytest.approx(0.5)


def test_repetition_counted_from_raw_ids_not_decoded_text():
    """A special-token degeneration (e.g. 256 BOS tokens)
    decodes to EMPTY text — re-tokenizing the text hides the repetition
    entirely; the raw generated ids must drive max_rep_run."""
    bos_storm = GenOut(text="", gen_token_logprobs=[-0.01] * 30,
                       ended_with_eos=False, gen_token_ids=[128000] * 30)
    stats = t0.canary_gen_stats(bos_storm)
    assert stats["max_rep_run"] == 30
    assert stats["n_tokens"] == 30
    v = t0.canary_values([bos_storm])
    assert v["max_rep_run_p95"] == 30.0


def test_abrupt_requires_ended_with_eos_and_excludes_terminal_eos():
    """7 content tokens + EOS is SEVEN content tokens (an
    <8 threshold must catch it), and a short generation that merely hit
    max_new_tokens without an EOS is not an abrupt EOS."""
    seven_plus_eos = GenOut(text="a b c d e f g",
                            gen_token_logprobs=[-1.0] * 8,
                            ended_with_eos=True,
                            gen_token_ids=[1, 2, 3, 4, 5, 6, 7, 99])
    s = t0.canary_gen_stats(seven_plus_eos, abrupt_min_tokens=8)
    assert s["n_tokens"] == 7  # terminal EOS excluded
    assert s["abrupt"] == 1
    short_no_eos = GenOut(text="a b c", gen_token_logprobs=[-1.0] * 3,
                          ended_with_eos=False, gen_token_ids=[1, 2, 3])
    assert t0.canary_gen_stats(short_no_eos, abrupt_min_tokens=8)["abrupt"] == 0


def test_canary_values_matches_production_aggregation_exactly():
    """simplify item 4: canary_values IS canary_gen_stats +
    canary_values_from_stats — the pair the production path uses."""
    outs = [_word_gen("one two three"), _word_gen("éé", ids=[5, 5, 5]),
            _word_gen("x y x y", ended=False)]
    stats = [t0.canary_gen_stats(o, abrupt_min_tokens=4) for o in outs]
    assert t0.canary_values(outs, abrupt_min_tokens=4) == \
        t0.canary_values_from_stats(stats)


# ---------------------------------------------------------------------------
# threshold derivation from 6 synthetic base runs (calibration-owned rule)
# ---------------------------------------------------------------------------


def _base_runs():
    """Six synthetic base self-runs: trial 0 is the reference."""
    runs = []
    for i, (rep, d3, lp, na, ab) in enumerate([
            (2.0, 0.80, -1.00, 0.010, 0.00),   # trial 0 (reference)
            (2.0, 0.79, -1.05, 0.012, 0.00),
            (3.0, 0.81, -0.98, 0.010, 0.01),
            (2.0, 0.78, -1.10, 0.011, 0.00),
            (2.0, 0.80, -1.02, 0.013, 0.00),
            (2.0, 0.82, -1.01, 0.010, 0.00)]):
        runs.append({"max_rep_run_p95": rep, "distinct3": d3,
                     "mean_tok_logprob": lp, "nonascii_frac": na,
                     "abrupt_eos_rate": ab})
    return runs


def test_derive_thresholds_worst_plus_p99_spread():
    th = t0.derive_canary_thresholds(_base_runs())
    assert isinstance(th, t0.CanaryThresholds)
    # Hand-computed against the calibration-owned rule (worst base value
    # + p99 of the harmful-oriented reference-contrast deltas, np.quantile
    # linear interpolation over the 5 contrasts — with 5 values p99 sits
    # just inside the max):
    # max_rep_run_p95 (up): worst 3.0, deltas [0,1,0,0,0] -> p99 0.96
    assert th.fail_lines["max_rep_run_p95"] == pytest.approx(3.96, abs=1e-12)
    # distinct3 (down): worst 0.78, harmful deltas [.01,-.01,.02,0,-.02]
    # -> p99 0.0196 -> 0.78 - 0.0196
    assert th.fail_lines["distinct3"] == pytest.approx(0.7604, abs=1e-12)
    # mean_tok_logprob (down): worst -1.10, p99 spread 0.098
    assert th.fail_lines["mean_tok_logprob"] == pytest.approx(-1.198, abs=1e-12)
    # abrupt_eos_rate (up): worst 0.01, p99 spread 0.0096
    assert th.fail_lines["abrupt_eos_rate"] == pytest.approx(0.0196, abs=1e-12)
    # nonascii_frac (up): worst 0.013, deltas [.002,0,.001,.003,0] ->
    # p99 0.00296
    assert th.fail_lines["nonascii_frac"] == pytest.approx(0.01596, abs=1e-12)


def test_derive_thresholds_refuses_missing_indicator():
    runs = _base_runs()
    del runs[2]["distinct3"]
    with pytest.raises(t0.Tier0Error, match="distinct3"):
        t0.derive_canary_thresholds(runs)


def test_thresholds_dataclass_refuses_missing_lines():
    with pytest.raises(t0.Tier0Error, match="missing indicators"):
        t0.CanaryThresholds(fail_lines={"distinct3": 0.5})


def test_from_calibration_adapter_shape():
    lines = {ind: {"fail_line": 1.0, "worst_base": 0.5,
                   "null_spread_p99": 0.5} for ind in IND}
    th = t0.CanaryThresholds.from_calibration(lines)
    assert th.fail_lines == {ind: 1.0 for ind in IND}
    assert th.nonascii_base == 0.0


# ---------------------------------------------------------------------------
# tripping (harmful direction only) and the >=2-indicator rule
# ---------------------------------------------------------------------------


def _thresholds():
    return t0.CanaryThresholds(fail_lines={
        "max_rep_run_p95": 4.0, "distinct3": 0.76,
        "mean_tok_logprob": -1.20, "nonascii_frac": 0.05,
        "abrupt_eos_rate": 0.02})


def _healthy_values():
    return {"max_rep_run_p95": 2.0, "distinct3": 0.80,
            "mean_tok_logprob": -1.0, "nonascii_frac": 0.01,
            "abrupt_eos_rate": 0.0}


def test_tripped_harmful_direction_only():
    th = _thresholds()
    assert t0.tripped_indicators(_healthy_values(), th) == ()
    # BETTER than the line in the harmless direction never trips
    great = {**_healthy_values(), "distinct3": 0.99, "mean_tok_logprob": -0.1}
    assert t0.tripped_indicators(great, th) == ()
    bad = {**_healthy_values(), "max_rep_run_p95": 40.0, "distinct3": 0.10}
    assert set(t0.tripped_indicators(bad, th)) == {"max_rep_run_p95",
                                                   "distinct3"}
    # down-direction trip
    low = {**_healthy_values(), "mean_tok_logprob": -5.0}
    assert t0.tripped_indicators(low, th) == ("mean_tok_logprob",)


def test_nan_indicator_is_a_loud_refusal_never_a_pass():
    """NaN compares false against every fail line, so a
    numerically destroyed run would trip NOTHING — it must refuse."""
    v = {**_healthy_values(), "mean_tok_logprob": float("nan")}
    with pytest.raises(t0.Tier0Error, match="NaN"):
        t0.tripped_indicators(v, _thresholds())


def test_nonascii_is_base_relative():
    th = t0.CanaryThresholds(fail_lines=_thresholds().fail_lines,
                             nonascii_base=0.30)
    # absolute 0.32 but only +0.02 over the base reference: under the
    # 0.05 delta line -> no trip (the French-vector case, §4.4)
    v = {**_healthy_values(), "nonascii_frac": 0.32}
    assert "nonascii_frac" not in t0.tripped_indicators(v, th)
    v2 = {**_healthy_values(), "nonascii_frac": 0.40}
    assert "nonascii_frac" in t0.tripped_indicators(v2, th)


# ---------------------------------------------------------------------------
# the confirmation state machine (§4.4/F16 — the driver owns the fresh
# subprocess; this owns the states)
# ---------------------------------------------------------------------------


def test_state_machine_pass_on_zero_or_one_indicator():
    assert t0.canary_outcome([]) == t0.CANARY_PASS
    assert t0.canary_outcome(["distinct3"]) == t0.CANARY_PASS  # 1 < 2


def test_state_machine_two_indicators_need_confirmation():
    out = t0.canary_outcome(["distinct3", "max_rep_run_p95"])
    assert out == t0.CANARY_NEEDS_CONFIRMATION


def test_state_machine_confirmed_only_when_reproduced():
    t1 = ["distinct3", "max_rep_run_p95"]
    assert t0.canary_outcome(t1, ["distinct3", "abrupt_eos_rate"]) == \
        t0.CANARY_CONFIRMED_FAIL  # >=2 again (indicator set may differ)
    assert t0.canary_outcome(t1, ["distinct3"]) == t0.CANARY_NOT_REPRODUCED
    assert t0.canary_outcome(t1, []) == t0.CANARY_NOT_REPRODUCED


def test_state_machine_attempt1_result_ignored_when_attempt0_passed():
    # a clean attempt 0 never consults attempt 1
    assert t0.canary_outcome([], ["distinct3", "max_rep_run_p95"]) == \
        t0.CANARY_PASS
