"""The steered smoke certifies the PRODUCTION serving configuration
(scripts/petri_steered_smoke.py) — CPU-only, no vllm.

exact_replay was dropped for the Petri audit:
``VLLM_BATCH_INVARIANT=1`` costs 2.97x and on H200/SM90 delivers only
same-size composition invariance, never batch-size-1-versus-N, so the
manifest now pins ``serving_mode: interleaved`` — whose composed launch
sets no VLLM_BATCH_INVARIANT — and re-running the audit is knowingly no
longer bit-for-bit reproducible. The contracts under test hold the gate
and production pointing at the same configuration:

1. THE PIN: the committed manifest's serving_mode, the smoke's
   ``CERTIFIED_SERVING_MODE``/``CERTIFIED_BATCH_INVARIANT``, and the
   composer's env for that mode all agree — a gate certifying a
   configuration production does not run is exactly the failure mode
   the PETRI-geometry phase closed for geometry, re-opened for the
   determinism mode.
2. THE WAIVER (``split_gating_failures``): at BI=0, failures of the
   composition-dependent checks (batch_invariance, selection_honored's
   output half) are waived — recorded, never gating — because
   composition invariance is not a property production claims; every
   other failure, every instrument failure, and everything at BI=1
   gates. The waiver must never suppress: waived failures come back in
   the second half of the split, verbatim.
3. THE LOSS IS RECORDED: the note that rides the record/marker/console
   states plainly that the audit is not exactly replayable.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "_petri_steered_smoke_bi0", ROOT / "scripts" / "petri_steered_smoke.py")
smoke = importlib.util.module_from_spec(_spec)
sys.modules["_petri_steered_smoke_bi0"] = smoke
_spec.loader.exec_module(smoke)

from sisteer.capeval import manifest as cm  # noqa: E402

MANIFEST = ROOT / "configs" / "capeval" / "manifests" / "petri_twosi.yaml"


# --- 1. the gate certifies what production runs ---------------------------------


def test_manifest_serving_mode_is_the_certified_serving_mode():
    spec = cm.load_petri_spec(MANIFEST)
    assert spec.serving_mode == smoke.CERTIFIED_SERVING_MODE, (
        "the committed manifest's serving_mode and the smoke gate's "
        "certified mode disagree — the gate would certify a "
        "configuration production does not run (or vice versa); move "
        "them together, with the decision recorded in both")
    assert spec.serving_mode == "interleaved"  # exact_replay dropped


def test_certified_batch_invariant_matches_the_interleaved_launch_env():
    """interleaved composes NO VLLM_BATCH_INVARIANT, i.e. the engine
    resolves BI=0 — so the certified BI setting must be "0". (The
    composer sets the env var only for the exact_replay
    batch_invariant strategy: vllm_server.build_server_launch.)"""
    assert smoke.CERTIFIED_SERVING_MODE == "interleaved"
    assert smoke.CERTIFIED_BATCH_INVARIANT == "0"


def test_the_certified_phase_keys_name_bi0():
    # the record's fixed keys must say what they certify: a reader of
    # an old record (bi1_certified) and a new one (bi0_certified) must
    # never wonder which configuration passed.
    assert smoke.PHASE_CERTIFIED == "bi0_certified"
    assert smoke.PHASE_PETRI == "bi0_certified_petri16k"
    assert smoke.PHASE_BI1 == "bi1_probe"


def test_composer_only_sets_batch_invariant_for_exact_replay():
    """The env-var side of contract 1, read off the composer's source:
    BATCH_INVARIANT_ENV is set exactly on the exact_replay
    batch_invariant strategy, so an interleaved production launch runs
    BI=0."""
    import inspect

    from sisteer.capeval.serving import vllm_server as vs

    src = inspect.getsource(vs.build_server_launch)
    assert 'if exact_replay_strategy == "batch_invariant":' in src
    assert src.count("BATCH_INVARIANT_ENV") == 1, (
        "build_server_launch sets BATCH_INVARIANT_ENV somewhere new — "
        "re-verify that an interleaved launch still resolves BI=0, then "
        "update this pin")


# --- 2. the waiver ---------------------------------------------------------------


def _failures():
    return [
        "batch_invariance: steered rows diverged between compositions "
        "(3/8; first divergences [[0, 17]])",
        "selection_honored: 2/8 rows differ across compositions over 3 "
        "distinct selections",
        "engine_determinism: 1/16 pairs differ; first divergences "
        "[[4, 0]]",
    ]


def test_composition_failures_are_waived_at_bi0_and_only_at_bi0():
    gating, waived = smoke.split_gating_failures(
        _failures(), batch_invariant="0")
    assert [f.split(":")[0] for f in waived] == ["batch_invariance",
                                                 "selection_honored"]
    assert [f.split(":")[0] for f in gating] == ["engine_determinism"]
    # nothing suppressed: the two halves partition the input verbatim
    assert sorted(gating + waived) == sorted(_failures())


def test_nothing_is_waived_at_bi1():
    gating, waived = smoke.split_gating_failures(
        _failures(), batch_invariant="1")
    assert waived == []
    assert gating == _failures()


def test_structural_failures_always_gate():
    extra = ("evidence: unknown evidence key(s) ['typo'] — allowed keys "
             "are [...]")
    gating, waived = smoke.split_gating_failures(
        [extra], batch_invariant="0")
    assert gating == [extra] and waived == []


def test_instrument_failures_gate_even_on_waivable_checks():
    """A waivable check whose PROBE broke (no evidence, malformed
    evidence) is the harness failing, not outputs moving — it must
    gate at any BI. The strings are evaluate_fidelity's own
    normalizations; this test pins the coupling."""
    for msg in (
            "batch_invariance: no evidence gathered — a missing probe "
            "is a failure, not a pass",
            "selection_honored: evidence must be a mapping, got type "
            "list",
            "batch_invariance: evidence could not be decoded "
            "(TypeError) — malformed evidence fails this check alone"):
        gating, waived = smoke.split_gating_failures(
            [msg], batch_invariant="0")
        assert gating == [msg] and waived == [], msg


def test_instrument_marks_match_evaluate_fidelity_texts():
    """The never-waive marks must actually occur in fidelity.py's
    failure normalizations, or the instrument carve-out silently
    stops matching anything."""
    src = (ROOT / "src" / "sisteer" / "capeval" / "serving"
           / "fidelity.py").read_text()
    for mark in smoke._INSTRUMENT_FAILURE_MARKS:
        assert mark in src, (
            f"{mark!r} no longer appears in fidelity.py — the "
            "instrument-failure carve-out went stale; update "
            "_INSTRUMENT_FAILURE_MARKS with the new wording")


def test_phase_check_split_tags_expected_failures_for_gating_bi0_phases():
    """Pre-2026-08-27 the expected/unexpected split applied only to the
    non-gating BI=0 probe; with BI=0 now the gating configuration, a
    GATING phase's report must carry the same tags (they feed the
    waiver's console/record annotations)."""
    report = {"checks": {
        "engine_determinism": "passed",
        "scale0_noop": "passed",
        "steering_bites": "passed",
        "batch_invariance": "batch_invariance: rows diverged",
        "selection_honored": "passed",
    }}
    split = smoke.phase_check_split(report, batch_invariant="0",
                                    gating=True)
    assert split["expected_failures"] == ["batch_invariance"]
    assert split["unexpected_failures"] == []
    assert split["all_expected"] is True
    # ... and at BI=1 the same failure is NOT expected, gating or not
    split1 = smoke.phase_check_split(report, batch_invariant="1",
                                     gating=True)
    assert split1["expected_failures"] == []
    assert split1["unexpected_failures"] == ["batch_invariance"]


def test_selection_direct_still_gates_at_bi0():
    """The waiver covers selection_honored's OUTPUT half only; the
    DIRECT half (the worker's own routing record) is composition-
    independent and a FAILED verdict must still produce a gating
    failure at BI=0."""
    msg = smoke.selection_direct_failure({
        "verdict": "FAILED", "n_rows": 12, "n_mismatched": 2,
        "aggregate_n_observations": 24, "aggregate_n_mismatched": 3})
    assert msg is not None
    # and split_gating_failures never waives it (its text does not
    # start with a waivable check name)
    gating, waived = smoke.split_gating_failures([msg],
                                                 batch_invariant="0")
    assert gating == [msg] and waived == []


# --- 3. the loss is recorded -----------------------------------------------------


def test_the_dropped_guarantee_is_stated_plainly():
    note = smoke.EXACT_REPLAY_DROPPED_NOTE
    for needle in ("BIT-FOR-BIT", "2026-08-27", "2.97x",
                   "interleaved", "repeat-determinism"):
        assert needle.lower() in note.lower(), needle
    # the manifest carries the same record next to the pin itself
    manifest_text = MANIFEST.read_text()
    assert "EXACT_REPLAY DROPPED" in manifest_text
    assert "bit-for-bit" in manifest_text


# --- the 32-way re-measure flag --------------------------------------------------


def test_parse_probe_utils_skip_and_lists():
    assert smoke.parse_probe_utils("skip") == ()
    assert smoke.parse_probe_utils("") == ()
    assert smoke.parse_probe_utils(None) == ()
    assert smoke.parse_probe_utils("0.90,0.95,0.97") == (0.90, 0.95, 0.97)
    assert smoke.parse_probe_utils("0.9") == (0.9,)


def test_parse_probe_utils_refuses_malformed_values():
    for bad in ("0.9,zero", "1.5", "0.2", "0.9,0.9"):
        with pytest.raises(SystemExit):
            smoke.parse_probe_utils(bad)
