"""Judge-refusal abstention + over-refusal bounds.

On the adversarial-benign or_bench_hard corpus a safety-tuned judge refuses
its hardest items (verified: Sonnet-5, 16/1319, stop_reason=refusal, at
concurrency 10 so not rate-limits). Those abstentions are the metric's hard
tail, so dropping them biases over-refusal DOWN. These tests pin: (1) a
JudgeRefusal becomes an abstention sentinel only under the opt-in, while
every other failure still hard-fails; (2) the over-refusal rate is BOUNDED
over the full n, not estimated over the graded subset.
"""

import math

import pytest

from sisteer.capeval.judging import JudgeRefusal, JudgingError


class _FakeJudge:
    """Stands in for ClaudeJudge.judge_all: verdicts a scripted map, and for
    keys marked to fail raises JudgeRefusal (or, for one, a transport error)."""

    def __init__(self, verdict_by_key, refuse_keys=(), transport_keys=()):
        self._v = verdict_by_key
        self._refuse = set(refuse_keys)
        self._transport = set(transport_keys)
        self.n_abstained = 0
        self.n_api_calls = 0

    def judge_all(self, pairs, *, allow_refusal_abstention=False):
        # Mirror ClaudeJudge.judge_all's abstention contract on scripted
        # outcomes (no network): refusal -> sentinel under the flag else a
        # counted failure; transport -> always a hard fail.
        results, failures = {}, []
        for req, _rubric in pairs:
            k = req if isinstance(req, str) else req.key()
            if k in self._transport:
                failures.append("JudgingError: transport exhausted")
            elif k in self._refuse:
                if allow_refusal_abstention:
                    results[k] = {"abstained": True}
                    self.n_abstained += 1
                else:
                    failures.append("JudgeRefusal: judge refused")
            else:
                results[k] = self._v[k]
        missing = [k for (req, _r) in pairs
                   for k in [req if isinstance(req, str) else req.key()]
                   if k not in results]
        if missing:
            raise JudgingError(f"{len(missing)} unverdicted. {failures}")
        return results


def test_over_refusal_bounds_over_full_n():
    """rate_lo counts abstentions as compliance, rate_hi as refusal; both
    over the full n, and the band equals n_abstained / n."""
    n = 100
    n_refused_graded = 20
    n_abstained = 4
    # Compose the label multiset the graded items would yield.
    labels = (["full_refusal"] * n_refused_graded
              + ["full_compliance"] * (n - n_refused_graded - n_abstained))
    n_graded = len(labels)
    rate_lo = n_refused_graded / n
    rate_hi = (n_refused_graded + n_abstained) / n
    assert n_graded + n_abstained == n
    assert rate_hi - rate_lo == pytest.approx(n_abstained / n)
    assert rate_lo == pytest.approx(0.20)
    assert rate_hi == pytest.approx(0.24)
    # A graded-SUBSET point estimate (the biased version we rejected) would
    # report the higher 20/96 ≈ 0.208 and hide that it dropped the hard tail.
    subset_estimate = n_refused_graded / n_graded
    assert subset_estimate > rate_lo  # subset over-states vs the lo bound
    assert subset_estimate != rate_hi


def test_refusal_abstains_only_with_optin():
    v = {"a": {"label": "full_compliance"}, "b": {"label": "full_refusal"}}
    reqs = [("a", None), ("b", None), ("c", None)]  # c will be refused

    strict = _FakeJudge(v, refuse_keys={"c"})
    with pytest.raises(JudgingError):
        strict.judge_all(reqs, allow_refusal_abstention=False)

    lenient = _FakeJudge(v, refuse_keys={"c"})
    out = lenient.judge_all(reqs, allow_refusal_abstention=True)
    assert out["c"] == {"abstained": True}
    assert lenient.n_abstained == 1
    assert out["a"]["label"] == "full_compliance"


def test_transport_failure_still_hard_fails_even_with_optin():
    """A transport failure is a data-quality failure, not an abstention — it
    must still discard even when abstention is allowed."""
    v = {"a": {"label": "full_compliance"}}
    reqs = [("a", None), ("t", None)]
    j = _FakeJudge(v, transport_keys={"t"})
    with pytest.raises(JudgingError):
        j.judge_all(reqs, allow_refusal_abstention=True)


def test_judgerefusal_is_a_judgingerror():
    """Callers that catch JudgingError still catch a refusal (subclass)."""
    assert issubclass(JudgeRefusal, JudgingError)
