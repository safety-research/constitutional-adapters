"""Engine-internal fidelity gate LOGIC (serving/fidelity.py) with
SYNTHETIC token-id evidence — CPU-only, no real recorded data. The GPU
evidence gathering lives in ``scripts/petri_serving_gate.py``.

Covers: first_divergence_index; the five within-engine exact-equality
checks each failing independently (scale-0 no-op, engine determinism as
INVALIDATING, per-arm steering-bites, both batch-invariance directions,
per-request selection); the selection check's two certification
refusals (fewer than two selections; indistinguishable reference outputs)
and that a real mismatch outranks them; failure accumulation (never
short-circuiting on the first); missing / vacuous / malformed evidence
each failing loudly and being contained to their own check; and the
numbers-and-verdicts-only report (no evidence text can leak — ids must
be integers, foreign exception texts are withheld, non-identifier
names are placeholders, as_dict refuses non-scalar leaves).
"""

import json

import pytest

from sisteer.capeval.serving import fidelity as fid

N, LEN = 6, 10


def _seqs(n=N, length=LEN, base=0):
    return [[base + 100 * i + j for j in range(length)] for i in range(n)]


def _mut(seqs, pair, at):
    out = [list(s) for s in seqs]
    out[pair][at] += 7777
    return out


def _good_evidence():
    return {
        "engine_determinism": {"run1_ids": _seqs(), "run2_ids": _seqs()},
        "scale0_noop": {"scale0_ids": _seqs(), "unsteered_ids": _seqs()},
        "steering_bites": {"arms": {
            "si_safety__cast": {"steered_ids": _seqs(base=1000),
                                "scale0_ids": _seqs()},
        }},
        "batch_invariance": {
            "steered_reference_ids": _seqs(base=1000),
            "steered_batched_ids": _seqs(base=1000),
            "unsteered_reference_ids": _seqs(),
            "unsteered_batched_ids": _seqs()},
        "selection_honored": {"requests": [
            {"arm": None, "prompt_index": 0,
             "reference_ids": [1, 2, 3], "batched_ids": [1, 2, 3]},
            {"arm": "si_safety__cast", "prompt_index": 0,
             "reference_ids": [4, 5, 6], "batched_ids": [4, 5, 6]},
            {"arm": "si_constitution__cast", "prompt_index": 1,
             "reference_ids": [7, 8], "batched_ids": [7, 8]},
        ]},
    }


def _failed(report):
    """The check names whose verdict is a failure (not passed, not
    skipped/indeterminate)."""
    return sorted(k for k, v in report.checks.items()
                  if v != "passed" and not v.startswith("indeterminate"))


# --------------------------------------------------------------------------
# the comparison primitive
# --------------------------------------------------------------------------


def test_first_divergence_index():
    assert fid.first_divergence_index([1, 2, 3], [1, 2, 3]) is None
    assert fid.first_divergence_index([], []) is None
    assert fid.first_divergence_index([1, 9], [1, 2]) == 1
    assert fid.first_divergence_index([5, 5, 5], [9, 5, 5]) == 0
    # a pure length difference (prefix relation) diverges at min(len)
    assert fid.first_divergence_index([1, 2], [1, 2, 3]) == 2
    assert fid.first_divergence_index([1, 2, 3], [1, 2]) == 2
    assert fid.first_divergence_index([], [7]) == 0


def test_check_names_are_frozen_and_evidence_keys_match():
    assert fid.FIDELITY_CHECKS == (
        "engine_determinism", "scale0_noop", "steering_bites",
        "batch_invariance", "selection_honored")
    assert fid.EVIDENCE_KEYS == fid.FIDELITY_CHECKS


# --------------------------------------------------------------------------
# the happy path — and the report's shape
# --------------------------------------------------------------------------


def test_all_checks_pass():
    rep = fid.evaluate_fidelity(_good_evidence())
    assert rep.passed and not rep.invalidated
    assert rep.failures == ()
    assert set(rep.checks) == set(fid.FIDELITY_CHECKS)
    assert all(v == "passed" for v in rep.checks.values())
    # diagnostics recorded even on a clean pass — and diagnostics ONLY
    assert rep.diagnostics["engine_determinism"] == {
        "n": N, "n_divergent": 0, "first_divergences": []}
    arms = rep.diagnostics["steering_bites"]["arms"]
    assert arms["si_safety__cast"]["n_differing"] == N  # bites everywhere
    assert rep.diagnostics["batch_invariance"]["steered"]["n"] == N
    assert rep.diagnostics["selection_honored"] == {
        "n_requests": 3, "n_selections": 3, "n_mismatched": 0,
        "mismatches": []}
    # JSON-ready, scalars only
    json.loads(json.dumps(rep.as_dict()))


def test_report_as_dict_is_scalars_only():
    d = fid.evaluate_fidelity(_good_evidence()).as_dict()

    def walk(o):
        if o is None or isinstance(o, (bool, int, float, str)):
            return
        if isinstance(o, dict):
            for k, v in o.items():
                assert isinstance(k, str)
                walk(v)
            return
        assert isinstance(o, list)
        for v in o:
            walk(v)

    walk(d)


# --------------------------------------------------------------------------
# each check failing independently
# --------------------------------------------------------------------------


def test_scale0_noop_failure_is_named_and_diagnosed():
    ev = _good_evidence()
    ev["scale0_noop"]["scale0_ids"] = _mut(_seqs(), 2, 4)
    rep = fid.evaluate_fidelity(ev)
    assert not rep.passed and not rep.invalidated
    assert _failed(rep) == ["scale0_noop"]
    assert "NOT a no-op" in rep.checks["scale0_noop"]
    # first-divergence indices are recorded as a DIAGNOSTIC
    assert rep.diagnostics["scale0_noop"]["first_divergences"] == [[2, 4]]


def test_determinism_failure_invalidates_the_whole_gate():
    ev = _good_evidence()
    ev["engine_determinism"]["run2_ids"] = _mut(_seqs(), 1, 4)
    rep = fid.evaluate_fidelity(ev)
    assert not rep.passed
    assert rep.invalidated
    # reported as invalidating, not as one failure among several:
    assert len(rep.failures) == 1
    assert rep.failures[0].startswith("engine_determinism")
    assert "INVALIDATES" in rep.failures[0]
    # the other checks' comparisons found no divergence, but a pass on
    # a nondeterministic engine evidences nothing
    for name in fid.FIDELITY_CHECKS[1:]:
        assert rep.checks[name].startswith("indeterminate")


def test_determinism_plus_real_failure_both_reported():
    ev = _good_evidence()
    ev["engine_determinism"]["run2_ids"] = _mut(_seqs(), 1, 4)
    ev["scale0_noop"]["scale0_ids"] = _mut(_seqs(), 0, 0)
    rep = fid.evaluate_fidelity(ev)
    assert rep.invalidated
    assert len(rep.failures) == 2
    assert rep.failures[0].startswith("engine_determinism")  # first
    assert rep.failures[1].startswith("scale0_noop")
    # an observed failure stays a failure, never "indeterminate"
    assert "NOT a no-op" in rep.checks["scale0_noop"]
    assert rep.checks["steering_bites"].startswith("indeterminate")


def test_steering_bites_names_the_silent_arm_only():
    ev = _good_evidence()
    ev["steering_bites"]["arms"] = {
        "biter": {"steered_ids": _seqs(base=1000), "scale0_ids": _seqs()},
        "silent_arm": {"steered_ids": _seqs(base=1000),
                       "scale0_ids": _seqs(base=1000)},  # no difference
    }
    rep = fid.evaluate_fidelity(ev)
    assert _failed(rep) == ["steering_bites"]
    msg = rep.checks["steering_bites"]
    assert "silent_arm" in msg and "biter" not in msg.split("arm(s)")[1]
    assert "no-op" in msg.lower()
    arms = rep.diagnostics["steering_bites"]["arms"]
    assert arms["silent_arm"]["n_differing"] == 0
    assert arms["biter"]["n_differing"] == N


def test_batch_invariance_fails_per_direction_and_names_both():
    for direction, label in (("steered", "steered_batched_ids"),
                             ("unsteered", "unsteered_batched_ids")):
        ev = _good_evidence()
        base = 1000 if direction == "steered" else 0
        ev["batch_invariance"][label] = _mut(_seqs(base=base), 3, 1)
        rep = fid.evaluate_fidelity(ev)
        assert _failed(rep) == ["batch_invariance"]
        assert f"{direction} requests changed when batched" \
            in rep.checks["batch_invariance"]
        assert "contamination" in rep.checks["batch_invariance"]
    # both directions broken -> ONE failure naming both
    ev = _good_evidence()
    ev["batch_invariance"]["steered_batched_ids"] = _mut(
        _seqs(base=1000), 0, 0)
    ev["batch_invariance"]["unsteered_batched_ids"] = _mut(_seqs(), 5, 9)
    rep = fid.evaluate_fidelity(ev)
    msg = rep.checks["batch_invariance"]
    assert "steered requests changed" in msg
    assert "unsteered requests changed" in msg
    assert rep.diagnostics["batch_invariance"]["steered"][
        "first_divergences"] == [[0, 0]]
    assert rep.diagnostics["batch_invariance"]["unsteered"][
        "first_divergences"] == [[5, 9]]


def test_selection_honored_mismatch_names_request_and_selection():
    ev = _good_evidence()
    ev["selection_honored"]["requests"][1]["batched_ids"] = [4, 9, 6]
    rep = fid.evaluate_fidelity(ev)
    assert _failed(rep) == ["selection_honored"]
    msg = rep.checks["selection_honored"]
    assert "request 1" in msg and "si_safety__cast" in msg
    assert "first diverges at 1" in msg
    assert rep.diagnostics["selection_honored"]["mismatches"] == [[1, 1]]


# --------------------------------------------------------------------------
# selection_honored's certification refusals
# --------------------------------------------------------------------------


def test_selection_needs_two_distinct_selections():
    ev = _good_evidence()
    ev["selection_honored"]["requests"] = [
        {"arm": None, "prompt_index": 0,
         "reference_ids": [1, 2], "batched_ids": [1, 2]},
        {"arm": None, "prompt_index": 1,
         "reference_ids": [3, 4], "batched_ids": [3, 4]},
    ]
    rep = fid.evaluate_fidelity(ev)
    assert _failed(rep) == ["selection_honored"]
    assert "TWO distinct selections" in rep.checks["selection_honored"]


def test_selection_refuses_indistinguishable_reference_outputs():
    # two selections share prompt 0 and produce IDENTICAL reference output:
    # a server routing both to one object would pass — refuse.
    ev = _good_evidence()
    ev["selection_honored"]["requests"] = [
        {"arm": None, "prompt_index": 0,
         "reference_ids": [1, 2], "batched_ids": [1, 2]},
        {"arm": "armA", "prompt_index": 0,
         "reference_ids": [1, 2], "batched_ids": [1, 2]},
    ]
    rep = fid.evaluate_fidelity(ev)
    assert _failed(rep) == ["selection_honored"]
    assert "indistinguishable" in rep.checks["selection_honored"]


def test_selection_refuses_when_no_prompt_is_shared():
    ev = _good_evidence()
    ev["selection_honored"]["requests"] = [
        {"arm": None, "prompt_index": 0,
         "reference_ids": [1, 2], "batched_ids": [1, 2]},
        {"arm": "armA", "prompt_index": 1,
         "reference_ids": [3, 4], "batched_ids": [3, 4]},
    ]
    rep = fid.evaluate_fidelity(ev)
    assert _failed(rep) == ["selection_honored"]
    assert "no probe prompt is served under more than one selection" \
        in rep.checks["selection_honored"]


def test_selection_mismatch_outranks_the_certification_refusals():
    # indistinguishable evidence AND an observed mismatch: the mismatch
    # is the real failure and must be the one reported.
    ev = _good_evidence()
    ev["selection_honored"]["requests"] = [
        {"arm": None, "prompt_index": 0,
         "reference_ids": [1, 2], "batched_ids": [1, 2]},
        {"arm": "armA", "prompt_index": 0,
         "reference_ids": [1, 2], "batched_ids": [9, 2]},
    ]
    rep = fid.evaluate_fidelity(ev)
    msg = rep.checks["selection_honored"]
    assert "NOT honored" in msg
    assert "indistinguishable" not in msg


def test_selection_field_validation():
    for bad, match in (
            ({"arm": 7, "prompt_index": 0, "reference_ids": [1],
              "batched_ids": [1]}, "string or None"),
            ({"arm": None, "prompt_index": -1, "reference_ids": [1],
              "batched_ids": [1]}, "non-negative"),
            ({"arm": None, "prompt_index": 0, "reference_ids": [],
              "batched_ids": []}, "vacuous"),
            ("not-a-mapping", "mapping"),
    ):
        ev = _good_evidence()
        ev["selection_honored"]["requests"] = [bad]
        rep = fid.evaluate_fidelity(ev)
        assert _failed(rep) == ["selection_honored"]
        assert match in rep.checks["selection_honored"]


# --------------------------------------------------------------------------
# failure accumulation — every failure reported, never just the first
# --------------------------------------------------------------------------


def test_every_failure_reported_not_just_the_first():
    ev = _good_evidence()
    ev["scale0_noop"]["scale0_ids"] = _mut(_seqs(), 0, 0)
    ev["steering_bites"]["arms"] = {
        "silent": {"steered_ids": _seqs(), "scale0_ids": _seqs()}}
    ev["batch_invariance"]["steered_batched_ids"] = _mut(
        _seqs(base=1000), 2, 2)
    ev["selection_honored"]["requests"][0]["batched_ids"] = [1, 2, 9]
    rep = fid.evaluate_fidelity(ev)
    assert not rep.passed and not rep.invalidated  # determinism held
    assert _failed(rep) == sorted(["scale0_noop", "steering_bites",
                                   "batch_invariance",
                                   "selection_honored"])
    assert len(rep.failures) == 4
    for name in _failed(rep):
        assert any(f.startswith(name) for f in rep.failures)


# --------------------------------------------------------------------------
# missing / vacuous / malformed evidence
# --------------------------------------------------------------------------


def test_missing_probes_all_fail_and_invalidate():
    rep = fid.evaluate_fidelity({})
    assert not rep.passed
    assert len(rep.failures) == 5
    assert all("no evidence gathered" in f for f in rep.failures)
    # determinism evidence missing -> nothing else can be certified
    assert rep.invalidated
    assert rep.failures[0].startswith("engine_determinism")


def test_unknown_evidence_key_refuses():
    ev = _good_evidence()
    ev["scale0noop"] = {"typo": True}  # the classic orphaned probe
    rep = fid.evaluate_fidelity(ev)
    assert not rep.passed
    assert all(v == "passed" for v in rep.checks.values())
    assert len(rep.failures) == 1
    assert "scale0noop" in rep.failures[0]
    assert "unknown evidence key" in rep.failures[0]


@pytest.mark.parametrize("check,mutation,match", [
    ("engine_determinism",
     lambda ev: ev["engine_determinism"].update(run1_ids=[]),
     "vacuous"),
    ("engine_determinism",
     lambda ev: ev["engine_determinism"].update(
         run1_ids=[[]] + _seqs()[1:], run2_ids=[[]] + _seqs()[1:]),
     "generated no tokens on either side"),
    ("scale0_noop",
     lambda ev: ev["scale0_noop"].update(scale0_ids=_seqs(n=4)),
     "unequal length"),
    ("scale0_noop",
     lambda ev: ev["scale0_noop"].pop("unsteered_ids"),
     "lacks"),
    ("steering_bites",
     lambda ev: ev["steering_bites"].update(arms={}),
     "zero arms"),
    ("steering_bites",
     lambda ev: ev["steering_bites"].update(arms=[["a", 1]]),
     "must be a mapping"),
    ("batch_invariance",
     lambda ev: ev["batch_invariance"].update(steered_reference_ids=[]),
     "vacuous"),
    ("selection_honored",
     lambda ev: ev["selection_honored"].update(requests=[]),
     "zero requests"),
    ("selection_honored",
     lambda ev: ev["selection_honored"].update(requests="nope"),
     "must be a list"),
])
def test_vacuous_or_missing_evidence_fails(check, mutation, match):
    ev = _good_evidence()
    mutation(ev)
    rep = fid.evaluate_fidelity(ev)
    assert check in _failed(rep)
    assert match in rep.checks[check]


def test_evidence_value_none_or_non_mapping_fails_that_check():
    ev = _good_evidence()
    ev["scale0_noop"] = None
    rep = fid.evaluate_fidelity(ev)
    assert "no evidence gathered" in rep.checks["scale0_noop"]
    ev = _good_evidence()
    ev["scale0_noop"] = [1, 2]
    rep = fid.evaluate_fidelity(ev)
    assert "must be a mapping" in rep.checks["scale0_noop"]


def test_token_ids_must_be_integers():
    for bad in ("text-instead-of-an-id", 3.5, True):
        ev = _good_evidence()
        ids = _seqs()
        ids[1][2] = bad
        ev["scale0_noop"]["scale0_ids"] = ids
        rep = fid.evaluate_fidelity(ev)
        assert _failed(rep) == ["scale0_noop"]
        msg = rep.checks["scale0_noop"]
        assert "non-integer element" in msg
        assert type(bad).__name__ in msg
        # the VALUE never appears, only its type name
        assert str(bad) not in msg


class _Boom(list):
    """A list whose iteration raises with evidence text in the message —
    the malformed-evidence + leak-vector case in one object."""

    def __iter__(self):
        raise RuntimeError("SECRET EVIDENCE TEXT do not leak")


def test_malformed_evidence_is_contained_and_exception_text_withheld():
    ev = _good_evidence()
    ev["batch_invariance"]["steered_reference_ids"] = _Boom([[1]])
    rep = fid.evaluate_fidelity(ev)
    assert _failed(rep) == ["batch_invariance"]  # the rest still ran
    msg = rep.checks["batch_invariance"]
    assert "could not be decoded" in msg and "RuntimeError" in msg
    assert "SECRET EVIDENCE TEXT" not in json.dumps(rep.as_dict())
    for name in ("engine_determinism", "scale0_noop", "steering_bites",
                 "selection_honored"):
        assert rep.checks[name] == "passed"


def test_evaluate_fidelity_refuses_non_mapping_evidence():
    with pytest.raises(fid.FidelityError, match="mapping"):
        fid.evaluate_fidelity([("engine_determinism", {})])


# --------------------------------------------------------------------------
# privacy: no evidence text can enter the report
# --------------------------------------------------------------------------


def test_no_text_leaks_into_the_report():
    marker = "SECRET PROMPT TEXT 93af"  # free text: spaces => never a name
    ev = _good_evidence()
    # 1. text where token ids belong -> refused by type name
    ids = _seqs()
    ids[0][0] = marker
    ev["scale0_noop"]["scale0_ids"] = ids
    # 2. text as an arm name (made silent so the check fails and would
    #    have echoed the name) -> shown as a typed placeholder
    ev["steering_bites"]["arms"] = {
        marker: {"steered_ids": _seqs(), "scale0_ids": _seqs()}}
    # 3. text as an unknown evidence key -> placeholder, value ignored
    ev[marker + " ?"] = {"x": marker}
    # 4. text inside a foreign exception message -> withheld
    ev["batch_invariance"]["unsteered_reference_ids"] = _Boom([[1]])
    # 5. text as a selection arm id -> placeholder in the mismatch line
    ev["selection_honored"]["requests"] = [
        {"arm": None, "prompt_index": 0,
         "reference_ids": [1, 2], "batched_ids": [1, 2]},
        {"arm": marker, "prompt_index": 0,
         "reference_ids": [3, 4], "batched_ids": [9, 4]},
    ]
    rep = fid.evaluate_fidelity(ev)
    dump = json.dumps(rep.as_dict())
    assert marker not in dump
    assert "non-identifier" in dump  # the placeholders did the echoing
    assert not rep.passed


def test_identifier_like_arm_ids_are_echoed():
    ev = _good_evidence()
    ev["steering_bites"]["arms"] = {
        "si_safety__cast-v2.1": {"steered_ids": _seqs(),
                                 "scale0_ids": _seqs()}}
    rep = fid.evaluate_fidelity(ev)
    assert "si_safety__cast-v2.1" in rep.checks["steering_bites"]


def test_report_walker_refuses_non_scalar_leaves():
    rep = fid.evaluate_fidelity(_good_evidence())
    bad = fid.FidelityReport(
        passed=rep.passed, invalidated=rep.invalidated,
        failures=rep.failures, checks=rep.checks,
        diagnostics={"engine_determinism": {"obj": object()}})
    with pytest.raises(fid.FidelityError, match="numbers and verdicts"):
        bad.as_dict()
