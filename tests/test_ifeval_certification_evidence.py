"""ifeval-canonical-battery — ANALYSIS revision 4: the evidence flags are authoritative at CERTIFICATION (code + tests
only, no number moved).

The defect. Revision 3 closed the GENERATION route (``--no-eval-logs`` folds I4 / I5 to PENDING, never waivable), but report-only
certification still read the two log-derived gates' STATUS alone: ``verdict_of`` declared ``production: true`` /
``eval_logs_validated: true`` whenever the statuses were PASSED, ``production_refusals`` had no line for a PASSED whose evidence
flag was false, and ``gate_report`` (``--gate-report``) trusted both. So a report FILE carrying ``I4 PASSED, eval_logs_read false``
and ``I5 PASSED, texts_reclassified false`` — the shape of the preserved revision-2 no-log artifact — passed ``--gate-report`` with
rc 0, and the genuine revision-3 production report with only those two flags flipped still read ``production: true``.

The rule pinned here: a PASSED I4 requires ``eval_logs_read is True``; a PASSED I5 requires ``texts_reclassified is True``; false
or MISSING evidence is an unwaivable refusal (``verdict_of`` / ``production_refusals`` / ``gate_report``), named by flag and value;
``--gate-report`` also re-derives the verdict and refuses a stored ``production: true`` the gates do not support. The four
regression shapes, each SYNTHETIC (a minimal report dict with the gates' real field names and the real detail strings; no store):

1. the old-artifact shape (revision 2's no-log pass: PASSED + false evidence, a stored ``production: true``) is refused;
2. the current production shape with ONLY the two evidence flags flipped (or removed) is refused;
3. the genuine production shape (PASSED + true evidence) still certifies: ``production true``, ``eval_logs_validated true``, rc 0;
4. the comparison-only route (PENDING + false evidence + the ``comparison_only`` block) is unchanged: the same three refusals,
   ``production false``, ``comparison_only true``, and nothing beyond the unread logs refuses it.

The same four are then run against the PRESERVED artifacts when this machine has them (read-only; flipped copies go to scratch;
skipped elsewhere, e.g. the CPU suite image) — the counterexamples above, end to end through the CLI.

Revision 5. The defect: certification still failed OPEN when a required gate was ABSENT or carried no
recognised status — ``gates_failed`` / ``gates_pending_*`` /
``gates_passed_without_evidence`` matched explicit FAILED / PENDING / PASSED only, and ``verdict_of`` required nothing of a gate it
could not find, so three malformed copies of the genuine production report (I4 / I5 deleted; their ``status`` keys deleted; their
statuses set to "UNKNOWN" with the evidence flags removed) passed ``--gate-report`` with rc 0. The rule pinned below: every
``GATE_NAMES`` entry must be PRESENT with a status in ``GATE_STATUSES`` before any waiver rule is read (``gates_malformed``, an
unwaivable refusal naming the gate and the value the file carries); ``production`` requires the positive ``eval_logs_validated``
(I4 / I5 present, PASSED, evidence True); ``--gate-report`` also refuses a stored ``eval_logs_validated: true`` the gates do not
support. Two further points are about the record's Section 10 and are pinned at the end: the matrix ``--check`` note (disclosed at revision 5
as a refusal with the frozen inputs; RE-FILED as the PASSING check from the frozen
registry copies, the refusal named as history, the record's matrix pin moved to the regenerated file, and the one difference from the file
the analysis read — the four deferred Arditi rows' harness-cell label — named and checked against git history), and
steps 3-6 rendered from the same as-run pieces as the As-run table's runner command.

Revision 6 (the closing review's two residuals; record only): Section 10's step 4 wrote no file while step 6 read
``<out>/diff_r2_r3_full.json`` — step 4 now writes both filed diff files as run (the CLI print redirected; the full ``report_match``
written whole) and states that it exits 4 BY CONSTRUCTION (leaves added, none changed), the counts read from the filed match; the
acceptance line names the clean-main attribution exception instead of a bare "union-clean". Pinned at the end.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from sisteer import site as SITE  # noqa: E402


def _load(name, rel):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


A = _load("tinycap_ifeval_battery", "scripts/tinycap_ifeval_battery.py")
P = _load("tinycap_ifeval_record_prose", "scripts/tinycap_ifeval_record_prose.py")

I4, I5 = "I4_scoring_path", "I5_at_cap"
MISSING = object()          # "the key is absent from the file", as distinct from False

#: the preserved artifacts named above (this machine only; the CLI tests below skip without them). Read-only: never modified, copies
#: for the flipped-flag fixtures go to the test's tmp_path.
ARTIFACTS = Path(os.environ.get("IFEVAL_ARTIFACTS") or SITE.expand_path("$GRADE/tinycap/ifeval")).expanduser()
R2_NO_LOG = ARTIFACTS / "final_e43f81f_r2" / "ifeval_report_match_venv2.json"
R3_PRODUCTION = ARTIFACTS / "final_e43f81f" / "ifeval_report.json"
R3_COMPARISON_ONLY = ARTIFACTS / "final_e43f81f" / "ifeval_report_match_venv2.json"


# --------------------------------------------------------------------------- synthetic report shapes (no store, no producer)
def _log_gate(gate: str, status: str, evidence) -> dict:
    """One log-derived gate as the report file carries it: PASSED with `pending []`, or PENDING with the real detail string; the
    evidence flag set to `evidence`, or left out when `evidence is MISSING`."""
    detail = None if status == A.GATE_PASS else (A.EVAL_LOGS_PENDING_I4 if gate == I4 else A.EVAL_LOGS_PENDING_I5)
    g = A._gate(status, detail, legs={}, failures=[], pending=[] if detail is None else [detail])
    if evidence is not MISSING:
        g[A.LOG_DERIVED_EVIDENCE[gate]] = evidence
    return g


def _report(*, i4=(A.GATE_PASS, True), i5=(A.GATE_PASS, True), comparison_only=False, allow_pending_gates=False, verdict=None) -> dict:
    """A minimal report: every other gate PASSED, I9 PENDING (the main pass runs alone by design), every designed contrast filed,
    no unusable leg — so the two log-derived gates are the ONLY thing that can refuse it."""
    gates = {g: A._gate(A.GATE_PASS, None) for g in A.GATE_NAMES}
    gates[I4] = _log_gate(I4, *i4)
    gates[I5] = _log_gate(I5, *i5)
    gates["I9_report_match"] = A._gate(A.GATE_PENDING, "the two-venv match is run as its own pass", waivable=True)
    A.refresh_gate_aggregates(gates)
    rep = {"gates": gates, "allow_pending_gates": allow_pending_gates, "unusable_legs": [], "contrasts": [],
           "contrast_summary": {"words": {}, "n_filed": 0, "n_designed": 0}, "cells": {}}
    if comparison_only:
        rep["comparison_only"] = {"mode": A.COMPARISON_ONLY_MODE, "eval_logs_read": False}
    rep["verdict"] = A.verdict_of(rep) if verdict is None else verdict
    return rep


def _old_artifact_shape() -> dict:
    """The revision-2 no-log artifact's shape: I4 / I5 PASSED, `eval_logs_read` / `texts_reclassified` False, no `comparison_only`,
    and a STORED verdict of the revision-2 vintage — `production: true`, none of the revision-3 keys."""
    return _report(i4=(A.GATE_PASS, False), i5=(A.GATE_PASS, False),
                   verdict={"production": True, "gates_failed": [], "gates_pending_unwaivable": [], "words": {}, "n_filed": 0,
                            "n_promoted_cells": 0, "n_usable_cells": 0})


def _write(tmp_path: Path, name: str, rep: dict) -> Path:
    out = tmp_path / name
    out.write_text(json.dumps(rep, default=str), encoding="utf-8")
    return out


def _evidence_lines(refusals: list[str]) -> list[str]:
    return [r for r in refusals if "PASSED without its evidence (never waivable)" in r]


# --------------------------------------------------------------------------- 1. the old-artifact shape is refused
def test_the_old_no_log_artifact_shape_is_refused_by_flag_and_value(tmp_path, capsys):
    rep = _old_artifact_shape()
    assert rep["gates"][I4]["status"] == rep["gates"][I5]["status"] == A.GATE_PASS       # the statuses alone said PASSED
    assert A.gates_pending_log_derived(rep["gates"]) == []                                 # r3's PENDING rule does not see it
    assert A.gates_passed_without_evidence(rep["gates"]) == [I4, I5]
    v = A.verdict_of(rep)
    assert v["production"] is False and v["eval_logs_validated"] is False and v["gates_passed_without_evidence"] == [I4, I5]
    assert v["gates_pending_log_derived"] == [] and v["gates_failed"] == [] and v["comparison_only"] is False
    refusals = A.production_refusals(rep)
    ev = _evidence_lines(refusals)
    assert len(refusals) == 2 == len(ev)
    assert f"gate {I4} PASSED without its evidence (never waivable): `eval_logs_read` is False, not True" in ev[0]
    assert f"gate {I5} PASSED without its evidence (never waivable): `texts_reclassified` is False, not True" in ev[1]
    # not the honest no-log state: it refuses a comparison-only pass too, and the non-log gates are NOT clear
    assert A.refusals_beyond_eval_logs(rep) == refusals and A.non_log_gates_clear(rep) is False
    # `--allow-pending-gates` waives nothing here
    waived = copy.deepcopy(rep)
    waived["allow_pending_gates"] = True
    assert A.verdict_of(waived)["production"] is False and len(A.production_refusals(waived)) == 2
    # the CLI: rc 3, the two evidence lines AND the stored verdict's claim named on stderr
    out = _write(tmp_path, "old_shape.json", rep)
    assert A.gate_report(out) == A.EXIT_NOT_PRODUCTION == 3 and A.main(["--gate-report", str(out)]) == 3
    err = capsys.readouterr().err
    assert err.count("REFUSED: gate I4_scoring_path PASSED without its evidence") == 2          # two runs
    assert "REFUSED: the stored verdict claims production: true, but re-derived from the gates and their evidence flags it is false" in err


# --------------------------------------------------------------------------- 2. the current shape with only the flags flipped
@pytest.mark.parametrize("evidence", [False, MISSING], ids=["false", "missing"])
def test_the_production_shape_with_only_the_evidence_flags_flipped_is_refused(tmp_path, evidence):
    genuine = _report()
    assert genuine["verdict"]["production"] is True
    flipped = copy.deepcopy(genuine)
    for g in A.LOG_DERIVED_GATES:
        flag = A.LOG_DERIVED_EVIDENCE[g]
        if evidence is MISSING:
            del flipped["gates"][g][flag]
        else:
            flipped["gates"][g][flag] = evidence
    # nothing else moved: the statuses, the tallies, every other gate
    assert flipped["gates"][I4]["status"] == flipped["gates"][I5]["status"] == A.GATE_PASS
    assert A.refresh_gate_aggregates(copy.deepcopy(flipped["gates"]))["n_passed"] == genuine["gates"]["n_passed"]
    v = A.verdict_of(flipped)
    assert v["production"] is False and v["eval_logs_validated"] is False and v["gates_passed_without_evidence"] == [I4, I5]
    refusals = A.production_refusals(flipped)
    shown = "None" if evidence is MISSING else "False"
    assert refusals == [f"gate {I4} PASSED without its evidence (never waivable): `eval_logs_read` is {shown}, not True — the status alone does not certify that the .eval logs were read",
                        f"gate {I5} PASSED without its evidence (never waivable): `texts_reclassified` is {shown}, not True — the status alone does not certify that the .eval logs were read"]
    # the file still carries the genuine verdict (production true): --gate-report refuses it and says the stored verdict is not trusted
    out = _write(tmp_path, f"flipped_{'missing' if evidence is MISSING else 'false'}.json", flipped)
    assert A.gate_report(out) == A.EXIT_NOT_PRODUCTION
    # one flag alone is enough
    one = copy.deepcopy(genuine)
    one["gates"][I5]["texts_reclassified"] = False
    assert A.verdict_of(one)["production"] is False and A.gates_passed_without_evidence(one["gates"]) == [I5]
    assert len(A.production_refusals(one)) == 1 and "`texts_reclassified` is False" in A.production_refusals(one)[0]


# --------------------------------------------------------------------------- 3. the genuine production shape still certifies
def test_the_genuine_production_shape_still_certifies(tmp_path):
    rep = _report()
    v = rep["verdict"]
    assert v["production"] is True and v["eval_logs_validated"] is True and v["comparison_only"] is False
    assert v["gates_passed_without_evidence"] == [] and v["gates_pending_log_derived"] == [] and v["gates_failed"] == []
    assert A.production_refusals(rep) == [] and A.refusals_beyond_eval_logs(rep) == [] and A.non_log_gates_clear(rep)
    out = _write(tmp_path, "genuine.json", rep)
    assert A.gate_report(out) == 0 and A.main(["--gate-report", str(out)]) == 0
    # a FAILED log-derived gate stays a FAILED refusal (the evidence rule is about PASSED only) and is not double-counted
    failed = copy.deepcopy(rep)
    failed["gates"][I4] = A._gate(A.GATE_FAIL, "leg x: stock score differs", legs={}, failures=["leg x: stock score differs"], eval_logs_read=True)
    A.refresh_gate_aggregates(failed["gates"])
    assert A.gates_passed_without_evidence(failed["gates"]) == [] and A.production_refusals(failed) == [f"gate {I4} FAILED: leg x: stock score differs"]


# --------------------------------------------------------------------------- 4. the comparison-only route is unchanged
def test_the_comparison_only_route_is_unchanged():
    rep = _report(i4=(A.GATE_PENDING, False), i5=(A.GATE_PENDING, False), comparison_only=True)
    v = rep["verdict"]
    assert v["production"] is False and v["comparison_only"] is True and v["eval_logs_validated"] is False
    assert v["gates_pending_log_derived"] == [I4, I5] == list(A.LOG_DERIVED_GATES) and v["gates_passed_without_evidence"] == []
    refusals = A.production_refusals(rep)
    assert refusals == [f"gate {I4} PENDING (never waivable): {A.EVAL_LOGS_PENDING_I4}",
                        f"gate {I5} PENDING (never waivable): {A.EVAL_LOGS_PENDING_I5}",
                        f"a {A.COMPARISON_ONLY_MODE} report: it claims no production (its purpose is the two-venv numeric-leaf match, I9)"]
    assert _evidence_lines(refusals) == [] and A.refusals_beyond_eval_logs(rep) == [] and A.non_log_gates_clear(rep)
    # the honest no-log state WITHOUT the mode block (a production attempt that read no log): the same two PENDING lines, still no evidence line
    honest = _report(i4=(A.GATE_PENDING, False), i5=(A.GATE_PENDING, False))
    assert honest["verdict"]["production"] is False and len(A.production_refusals(honest)) == 2 and _evidence_lines(A.production_refusals(honest)) == []


# --------------------------------------------------------------------------- the record note, verbatim from the filler


# --------------------------------------------------------------------------- the preserved artifacts, end to end (this machine only)
needs_artifacts = pytest.mark.skipif(not (R2_NO_LOG.is_file() and R3_PRODUCTION.is_file() and R3_COMPARISON_ONLY.is_file()),
                                     reason="the preserved revision-2 / revision-3 artifacts are not on this machine")


@needs_artifacts
def test_d69_counterexamples_on_the_preserved_artifacts(tmp_path, capsys):
    # (1) the revision-2 no-log artifact: I4 / I5 PASSED with false evidence and a stored production: true -> rc 3 (was 0)
    r2 = json.loads(R2_NO_LOG.read_text(encoding="utf-8"))
    assert r2["gates"][I4]["status"] == r2["gates"][I5]["status"] == A.GATE_PASS
    assert r2["gates"][I4]["eval_logs_read"] is False and r2["gates"][I5]["texts_reclassified"] is False and r2["verdict"]["production"] is True
    assert A.main(["--gate-report", str(R2_NO_LOG)]) == A.EXIT_NOT_PRODUCTION
    err = capsys.readouterr().err
    assert f"REFUSED: gate {I4} PASSED without its evidence (never waivable): `eval_logs_read` is False" in err
    assert f"REFUSED: gate {I5} PASSED without its evidence (never waivable): `texts_reclassified` is False" in err
    assert "REFUSED: the stored verdict claims production: true" in err
    # (3) the genuine revision-3 production report: rc 0, and its re-derived verdict equals the stored one key for key
    r3 = json.loads(R3_PRODUCTION.read_text(encoding="utf-8"))
    assert r3["gates"][I4]["eval_logs_read"] is True and r3["gates"][I5]["texts_reclassified"] is True
    assert A.main(["--gate-report", str(R3_PRODUCTION)]) == 0 and capsys.readouterr().err == ""
    derived = A.verdict_of(r3)
    # the stored r3 verdict predates the r4 / r5 keys; both re-derive empty, and every other key is equal
    assert derived.pop("gates_passed_without_evidence") == [] and derived.pop("gates_malformed") == [] and derived == r3["verdict"]
    # (2) the same report with ONLY the two flags flipped (a scratch copy; the artifact untouched) -> rc 3
    before = R3_PRODUCTION.read_bytes()
    flipped = copy.deepcopy(r3)
    flipped["gates"][I4]["eval_logs_read"] = False
    flipped["gates"][I5]["texts_reclassified"] = False
    out = _write(tmp_path, "r3_flipped.json", flipped)
    assert A.main(["--gate-report", str(out)]) == A.EXIT_NOT_PRODUCTION and A.verdict_of(flipped)["production"] is False
    assert R3_PRODUCTION.read_bytes() == before
    # (4) the revision-3 comparison-only artifact: the same three refusals as before, nothing beyond the unread logs
    cmp = json.loads(R3_COMPARISON_ONLY.read_text(encoding="utf-8"))
    refusals = A.production_refusals(cmp)
    assert len(refusals) == 3 and _evidence_lines(refusals) == [] and refusals[-1].startswith(f"a {A.COMPARISON_ONLY_MODE} report")
    assert A.refusals_beyond_eval_logs(cmp) == [] and A.non_log_gates_clear(cmp) and A.main(["--gate-report", str(R3_COMPARISON_ONLY)]) == 3
    assert cmp["verdict"]["production"] is False and cmp["verdict"]["comparison_only"] is True


# =========================================================================== revision 5: a gate the file does not carry
def _malformed_lines(refusals: list[str]) -> list[str]:
    return [r for r in refusals if "ABSENT from the report's gates" in r or "carries no recognised status" in r or "is not a gate record" in r]


def _assert_refused_by_name(rep: dict, gates: list[str], *, lines: int, validated: bool = False) -> list[str]:
    """The r5 verdict shape for a report whose `gates` are malformed at ``gates``: no other rule sees them, production False, the
    malformed lines FIRST and exactly ``lines`` of them, unwaivable, unfiltered by the diagnostic. ``eval_logs_validated`` is a
    statement about I4 / I5 only: False when either is malformed, ``validated`` (True) when another gate is the malformed one and
    the logs WERE read — production is refused all the same."""
    assert A.gates_malformed(rep["gates"]) == gates
    assert A.gates_failed(rep["gates"]) == [] and A.gates_pending_log_derived(rep["gates"]) == [] and A.gates_passed_without_evidence(rep["gates"]) == []
    v = A.verdict_of(rep)
    assert v["production"] is False and v["eval_logs_validated"] is validated and v["gates_malformed"] == gates
    assert v["gates_failed"] == [] and v["gates_pending_log_derived"] == [] and v["gates_passed_without_evidence"] == [] and v["gates_pending_unwaivable"] == []
    refusals = A.production_refusals(rep)
    assert refusals[:lines] == _malformed_lines(refusals) and len(refusals) == lines
    assert all(g in refusals[i] and "(never waivable)" in refusals[i] for i, g in enumerate(gates))
    # not the honest no-log state: the diagnostic does not filter it, and `--allow-pending-gates` waives nothing
    assert A.refusals_beyond_eval_logs(rep) == refusals and A.non_log_gates_clear(rep) is False
    waived = copy.deepcopy(rep)
    waived["allow_pending_gates"] = True
    assert A.verdict_of(waived)["production"] is False and A.production_refusals(waived) == refusals
    return refusals


def test_deleting_the_i4_and_i5_gate_entries_is_refused_by_name(tmp_path, capsys):
    """Variant (i): the genuine production shape with the two log-derived gate ENTRIES deleted. Before r5 this re-derived
    production True with every refusal list empty; now each absent gate is named, first, and nothing waives it."""
    rep = _report()
    assert rep["verdict"]["production"] is True
    del rep["gates"][I4], rep["gates"][I5]                 # the stored verdict still claims production: true
    refusals = _assert_refused_by_name(rep, [I4, I5], lines=2)
    assert refusals[0].startswith(f"gate {I4} ABSENT from the report's gates (never waivable): every one of the 10 gates must be present with a status in ['PASSED', 'FAILED', 'PENDING']")
    assert refusals[1].startswith(f"gate {I5} ABSENT from the report's gates (never waivable)")
    out = _write(tmp_path, "i4_i5_deleted.json", rep)
    assert A.gate_report(out) == A.EXIT_NOT_PRODUCTION == 3 and A.main(["--gate-report", str(out)]) == 3
    err = capsys.readouterr().err
    assert err.count(f"REFUSED: gate {I4} ABSENT") == 2 and err.count(f"REFUSED: gate {I5} ABSENT") == 2
    assert "REFUSED: the stored verdict claims production: true" in err and "REFUSED: the stored verdict claims eval_logs_validated: true" in err


def test_a_gate_entry_with_no_status_key_is_refused_by_name(tmp_path):
    """Variant (ii): the entries kept (evidence flags and all) but their `status` keys deleted — `status` is None, not PASSED,
    so neither the r4 evidence rule nor any status rule sees them; r5 names them."""
    rep = _report()
    del rep["gates"][I4]["status"], rep["gates"][I5]["status"]
    assert rep["gates"][I4]["eval_logs_read"] is True and rep["gates"][I5]["texts_reclassified"] is True     # the evidence alone certifies nothing
    refusals = _assert_refused_by_name(rep, [I4, I5], lines=2)
    assert refusals[0] == (f"gate {I4} carries no recognised status (never waivable): `status` is None, not one of ['PASSED', 'FAILED', 'PENDING'] — "
                           "an unrecognised or missing status certifies nothing and no flag waives it")
    assert "`status` is None" in refusals[1] and refusals[1].startswith(f"gate {I5} carries no recognised status")
    assert A.gate_report(_write(tmp_path, "status_deleted.json", rep)) == 3


@pytest.mark.parametrize("bad", ["UNKNOWN", "passed", "", True, 1, None], ids=["UNKNOWN", "lowercase", "empty", "bool", "int", "null"])
def test_an_unrecognised_status_is_refused_by_name_and_not_read_as_any_other_state(tmp_path, bad):
    """Variant (iii): statuses set to a value outside {PASSED, FAILED, PENDING}, the evidence flags removed. An unrecognised status
    is its own refusal — not a PENDING, not a FAILED, not a PASSED-without-evidence — and the value the file carries is quoted."""
    rep = _report()
    for g in A.LOG_DERIVED_GATES:
        rep["gates"][g]["status"] = bad
        rep["gates"][g].pop(A.LOG_DERIVED_EVIDENCE[g], None)
    refusals = _assert_refused_by_name(rep, [I4, I5], lines=2)
    assert f"`status` is {bad!r}, not one of ['PASSED', 'FAILED', 'PENDING']" in refusals[0]
    assert A.gate_report(_write(tmp_path, "unknown_status.json", rep)) == 3
    # the tally of the file's claims (`n_passed`) simply does not count it; certification refuses above it
    assert A.refresh_gate_aggregates(copy.deepcopy(rep["gates"]))["n_passed"] == len(A.GATE_NAMES) - 3       # I4, I5 and the PENDING I9


def test_any_absent_gate_refuses_including_a_waivable_one_a_non_record_entry_and_a_missing_gates_block(tmp_path, capsys):
    # the waivable I9: PENDING is waived, ABSENCE is not — every gate must exist before any waiver rule is read
    rep = _report()
    del rep["gates"]["I9_report_match"]
    refusals = _assert_refused_by_name(rep, ["I9_report_match"], lines=1, validated=True)
    assert "I9_report_match ABSENT" in refusals[0]
    # a non-log gate deleted: refused by name, I4 / I5 untouched (their evidence still True), production False all the same
    rep = _report()
    del rep["gates"]["I0_completeness"]
    _assert_refused_by_name(rep, ["I0_completeness"], lines=1, validated=True)
    assert A.eval_logs_validated(rep["gates"]) is True                       # the logs WERE read; the report is still not production
    # an entry that is not a record at all (a bare string where the gate record should be)
    rep = _report()
    rep["gates"][I4] = A.GATE_PASS
    refusals = _assert_refused_by_name(rep, [I4], lines=1)
    assert refusals[0].startswith(f"gate {I4} is not a gate record (never waivable): the file carries str 'PASSED'")
    # the whole `gates` block missing, or not a mapping: all ten ABSENT, no KeyError, rc 3 through the CLI
    for shape, name in ((None, "no_gates"), ([], "gates_list")):
        rep = _report()
        if shape is None:
            del rep["gates"]
        else:
            rep["gates"] = shape
        assert A.gates_malformed(A._gates_of(rep)) == list(A.GATE_NAMES)
        v = A.verdict_of(rep)
        assert v["production"] is False and v["eval_logs_validated"] is False and v["gates_malformed"] == list(A.GATE_NAMES)
        refusals = A.production_refusals(rep)
        assert len(refusals) == len(A.GATE_NAMES) and all("ABSENT" in r for r in refusals)
        assert A.main(["--gate-report", str(_write(tmp_path, f"{name}.json", rep))]) == 3
    err = capsys.readouterr().err
    assert err.count("REFUSED: gate I0_completeness ABSENT") == 2 and "REFUSED: the stored verdict claims production: true" in err


def test_eval_logs_validated_is_a_positive_requirement_and_production_implies_it():
    genuine = _report()
    assert A.eval_logs_validated(genuine["gates"]) is True and genuine["verdict"]["production"] is True
    shapes = {
        "i4 pending": _report(i4=(A.GATE_PENDING, False)),
        "i5 failed": _report(i5=(A.GATE_FAIL, True)),
        "i4 passed, evidence false": _report(i4=(A.GATE_PASS, False)),
        "i5 passed, evidence missing": _report(i5=(A.GATE_PASS, MISSING)),
    }
    absent = _report(); del absent["gates"][I4]; shapes["i4 absent"] = absent
    unknown = _report(); unknown["gates"][I5]["status"] = "UNKNOWN"; shapes["i5 unrecognised"] = unknown
    for label, rep in shapes.items():
        v = A.verdict_of(rep)
        assert A.eval_logs_validated(rep["gates"]) is False and v["eval_logs_validated"] is False and v["production"] is False, label
        assert A.production_refusals(rep), label
    # production is impossible without it, whatever else the file says: every constructed shape obeys production => validated
    for rep in (genuine, *shapes.values()):
        v = A.verdict_of(rep)
        assert (not v["production"]) or v["eval_logs_validated"]


def test_gate_report_refuses_a_stored_eval_logs_validated_claim_the_gates_do_not_support(tmp_path, capsys):
    """An honest no-log shape (I4 / I5 PENDING) whose STORED verdict nevertheless claims eval_logs_validated: true (production
    false): rc 3 with the eval_logs_validated line, and no production-claim line (none was made)."""
    rep = _report(i4=(A.GATE_PENDING, False), i5=(A.GATE_PENDING, False))
    assert rep["verdict"]["production"] is False and rep["verdict"]["eval_logs_validated"] is False
    rep["verdict"]["eval_logs_validated"] = True
    out = _write(tmp_path, "claims_validated.json", rep)
    assert A.main(["--gate-report", str(out)]) == 3
    err = capsys.readouterr().err
    assert "REFUSED: the stored verdict claims eval_logs_validated: true, but re-derived from the gates and their evidence flags it is false" in err
    assert "claims production: true" not in err and err.count("PENDING (never waivable)") == 2


def test_the_genuine_and_comparison_only_shapes_are_unchanged_by_revision_5(tmp_path):
    genuine = _report()
    assert A.gates_malformed(genuine["gates"]) == [] and genuine["verdict"]["gates_malformed"] == [] and genuine["verdict"]["production"] is True
    assert A.production_refusals(genuine) == [] and A.gate_report(_write(tmp_path, "genuine5.json", genuine)) == 0
    cmp = _report(i4=(A.GATE_PENDING, False), i5=(A.GATE_PENDING, False), comparison_only=True)
    assert cmp["verdict"]["gates_malformed"] == [] and cmp["verdict"]["production"] is False and cmp["verdict"]["comparison_only"] is True
    refusals = A.production_refusals(cmp)
    assert len(refusals) == 3 and _malformed_lines(refusals) == [] and refusals[-1].startswith(f"a {A.COMPARISON_ONLY_MODE} report")
    assert A.refusals_beyond_eval_logs(cmp) == [] and A.non_log_gates_clear(cmp)
    # the r4 shapes: still their own state, never re-read as malformed
    old = _old_artifact_shape()
    assert A.gates_malformed(old["gates"]) == [] and A.gates_passed_without_evidence(old["gates"]) == [I4, I5] and len(A.production_refusals(old)) == 2


# --------------------------------------------------------------------------- the record: revision 5's note and Section 10 (the two MEDIUMs)


def _record_commit(md: str) -> str:
    return re.search(r"analysis tip `([0-9a-f]{40})`", md).group(1)


def _section_10(md: str) -> str:
    return md[md.index("## 10. Reproduce"):md.index("## Revision 2 after review")]


#: the SHAPE of the filed r2 -> r3 match (`final_e43f81f/diff_r2_r3_full.json`, read into the record's Revision 3 block): four added leaves,
#: the one promoted pair's recorded promotion chains, nothing changed — what the step-4 exit note reads its counts from (never types)
FILED_R2_R3_MATCH = {"shared_numeric": 15193, "differing": [], "type_changed": [], "missing": [],
                     "added": [["noise_floor", 3, "pairs", 2, "attestation", "ladder_promotions", view, i] for view in ("main", "it") for i in (0, 1)],
                     "equal": False}






#: main at the roster-shadow remedy's base: the last tree carrying the matrix file the revision-3 analysis read — the anchor for
#: the git cross-check of the note's claim about that file (a wrong anchor fails loudly: the blob must hash to P.ANALYSIS_MATRIX_SHA256)
ANALYSIS_MATRIX_COMMIT = "78e0fd9064c856d9d617064818e13519e734c849"


#: the tree the public release rewrote: since then the matrix's frozen-input shas moved with the release's re-pin and one
#: policy reason lost a working-session name, so the note's claim about this tree is checked against the matrix as the
#: release found it
PRE_RELEASE_SHA = "6801151ed3ba5a62e9fb2ad5d6d01adc8ef82eac"


def _tree_matrix():
    r = subprocess.run(["git", "show", f"{PRE_RELEASE_SHA}:{P.MATRIX_REL}"], cwd=REPO, capture_output=True, check=False)
    if r.returncode != 0:
        pytest.skip(f"shallow clone: {PRE_RELEASE_SHA[:8]} is not in this checkout")
    return hashlib.sha256(r.stdout).hexdigest(), json.loads(r.stdout)




def test_the_analysis_matrix_the_record_names_differs_from_this_trees_file_in_the_four_deferred_rows_label_only():
    """The note's claim about the file the revision-3 analysis read (`ANALYSIS_MATRIX_SHA256`), checked against the BYTES rather than the
    prose: main carried that file at the remedy's base commit; it hashes to the named digest, its rows 0-35 equal this tree's, its counts
    and contrasts too, and rows 36-39 differ in exactly harness_cell / standup_harness_cell / deferred.pending (null -> arditi; the
    vocabulary item dropped). Skips by name on a shallow clone (the repo's precedent for git cross-checks)."""
    spec = f"{ANALYSIS_MATRIX_COMMIT}:{P.MATRIX_REL}"
    if subprocess.run(["git", "cat-file", "-e", spec], cwd=REPO, capture_output=True, check=False).returncode != 0:
        pytest.skip(f"shallow clone: {ANALYSIS_MATRIX_COMMIT[:8]} is not in this checkout")
    old_bytes = subprocess.run(["git", "show", spec], cwd=REPO, capture_output=True, check=True).stdout
    assert hashlib.sha256(old_bytes).hexdigest() == P.ANALYSIS_MATRIX_SHA256
    old = json.loads(old_bytes)
    digest, new = _tree_matrix()
    assert digest != P.ANALYSIS_MATRIX_SHA256
    assert old["rows"][:36] == new["rows"][:36] and old["counts"] == new["counts"] and old["contrasts"] == new["contrasts"] and old["policy"] == new["policy"]
    assert [i for i in range(len(old["rows"])) if old["rows"][i] != new["rows"][i]] == [36, 37, 38, 39] and len(new["rows"]) == 40
    for a, b in zip(old["rows"][36:40], new["rows"][36:40]):
        assert {k for k in a if a[k] != b[k]} == {"harness_cell", "standup_harness_cell", "deferred"}
        assert (a["harness_cell"], a["standup_harness_cell"], b["harness_cell"], b["standup_harness_cell"]) == (None, None, "arditi", "arditi")
        assert {k for k in a["deferred"] if a["deferred"][k] != b["deferred"][k]} == {"pending"}
        assert any(p.startswith("harness vocabulary:") and "'arditi'" in p for p in a["deferred"]["pending"])
        assert b["deferred"]["pending"] == [f"record {b['deferred']['record']}"] and a["status"] == b["status"] == "deferred"
    # the source block moved only where the remedy says: the overlay's sha (its header) and the frozen-inputs record
    assert {k for k in set(old["source"]) | set(new["source"]) if old["source"].get(k) != new["source"].get(k)} == {"roster_overlay_sha256", "frozen_inputs"}
    assert old["source"].get("frozen_inputs") is None and new["source"]["frozen_inputs"]["roster_sha256"] == old["source"]["roster_sha256"]




# --------------------------------------------------------------------------- the preserved artifact, end to end (this machine only)
@needs_artifacts
def test_d74_malformed_variants_of_the_preserved_production_report(tmp_path, capsys):
    before = R3_PRODUCTION.read_bytes()
    r3 = json.loads(before)
    assert A.main(["--gate-report", str(R3_PRODUCTION)]) == 0 and capsys.readouterr().err == ""
    variants = {}
    v = copy.deepcopy(r3); del v["gates"][I4], v["gates"][I5]; variants["i_deleted"] = (v, "ABSENT from the report's gates")
    v = copy.deepcopy(r3); del v["gates"][I4]["status"], v["gates"][I5]["status"]; variants["ii_no_status"] = (v, "`status` is None")
    v = copy.deepcopy(r3)
    for g in A.LOG_DERIVED_GATES:
        v["gates"][g]["status"] = "UNKNOWN"; del v["gates"][g][A.LOG_DERIVED_EVIDENCE[g]]
    variants["iii_unknown"] = (v, "`status` is 'UNKNOWN'")
    for name, (rep, needle) in variants.items():
        assert A.verdict_of(rep)["production"] is False and A.verdict_of(rep)["gates_malformed"] == [I4, I5], name
        assert A.main(["--gate-report", str(_write(tmp_path, f"r3_{name}.json", rep))]) == A.EXIT_NOT_PRODUCTION, name
        err = capsys.readouterr().err
        assert f"REFUSED: gate {I4} " in err and f"REFUSED: gate {I5} " in err and needle in err, name
        assert "REFUSED: the stored verdict claims production: true" in err and "claims eval_logs_validated: true" in err, name
    assert R3_PRODUCTION.read_bytes() == before
    # the standing shapes keep their rcs and lines: the r2 no-log artifact 3 (evidence lines), the r3 comparison-only 3 (PENDING lines)
    assert A.main(["--gate-report", str(R2_NO_LOG)]) == 3 and "PASSED without its evidence" in capsys.readouterr().err
    assert A.main(["--gate-report", str(R3_COMPARISON_ONLY)]) == 3
    err = capsys.readouterr().err
    assert err.count("PENDING (never waivable)") == 2 and _malformed_lines(err.splitlines()) == []


# --------------------------------------------------------------------------- revision 6: step 4 writes what step 6 reads; the exit explained; the acceptance wording




def test_the_exit_note_reads_the_match_never_types_it():
    commit = "0" * 40
    assert "ADDED 4 leaves (the promotion chains" in P.diff_exit_note(FILED_R2_R3_MATCH)
    two = {**FILED_R2_R3_MATCH, "added": FILED_R2_R3_MATCH["added"][:2]}
    assert "ADDED 2 leaves (the promotion chains" in P.diff_exit_note(two) and "ADDED 2 leaves" in P.reproduce_steps(commit, r2_diff=two)
    other = {**FILED_R2_R3_MATCH, "added": [["verdict", "n_filed"]], "differing": [(["contrasts", 0, "point"], 0.1, 0.2)]}
    note = P.diff_exit_note(other)
    assert "ADDED 1 leaves while 1 shared numeric leaves changed" in note and "promotion chains" not in note
    # without the match: no count typed, the reader is sent to the Revision 3 block, and the steps still render with the exit explained
    bare = P.reproduce_steps(commit)
    assert "EXITS 4 BY CONSTRUCTION" in bare and "ADDED leaves (the count and the paths are in the Revision 3 block" in bare and "ADDED 4" not in bare
    assert bare.index("# 4. ") < bare.index("# 5. ") < bare.index("# 6. ")


def test_diff_reports_exits_4_when_a_leaf_was_added_and_the_documented_full_match_command_writes_the_file(tmp_path, capsys):
    import shlex
    import subprocess
    a = {"sha": "x", "cells": {"c": {"n": 3, "score": 0.5}}, "gates": {"I1": {"status": "PASSED"}}}
    b = {"sha": "x", "cells": {"c": {"n": 3, "score": 0.5, "extra": 7}}, "gates": {"I1": {"status": "PASSED"}}}
    pa, pb = _write(tmp_path, "a.json", a), _write(tmp_path, "b.json", b)
    # identical leaves -> 0; one ADDED leaf and nothing changed -> EXIT_REPORTS_DIFFER (4), by construction of `equal`
    assert A.diff_reports(pa, pa) == 0 and json.loads(capsys.readouterr().out)["equal"] is True
    assert A.diff_reports(pa, pb) == A.EXIT_REPORTS_DIFFER == 4
    printed = json.loads(capsys.readouterr().out)
    assert printed["equal"] is False and printed["added"] == [["cells", "c", "extra"]] and printed["differing"] == []
    # the documented one-liner, run as the record shows it (PYTHONPATH=src:scripts; <out> substituted), writes the full match whole
    cmd = P.AS_RUN_DIFF_FULL_CMD.replace("<out>_r2/ifeval_report.json", str(pa)).replace("<out>/ifeval_report.json", str(pb))
    out = tmp_path / "diff_full.json"
    argv = shlex.split(cmd.replace("<out>/diff_r2_r3_full.json", str(out)))
    assert argv[0] == "python" and argv[1] == "-c" and argv[-3:] == [str(pa), str(pb), str(out)]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(REPO / "src"), str(REPO / "scripts")])}
    run = subprocess.run([sys.executable, *argv[1:]], env=env, cwd=REPO, capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stderr[-2000:]
    assert out.read_text(encoding="utf-8") == json.dumps(A.report_match(a, b), indent=1)
    assert json.loads(out.read_text(encoding="utf-8")) == printed          # under the 20-path cut the print IS the full match




def test_the_matrix_digest_refusals_are_live_on_a_synthetic_record():
    """The prose filler's two matrix refusals — frozen_matrix_source (a tree whose frozen matrix is not the one the report / record
    pins) and record_matrix_sha256 (a record without the `| matrix | <rel> sha256 <hex>` row) — on a synthetic record string: their
    only coverage left with the results/ record."""
    row = "| matrix | `" + P.MATRIX_REL + "` sha256 `" + "0" * 64 + "` | the frozen matrix |"
    assert P.record_matrix_sha256("# a record\n\n" + row + "\n") == "0" * 64
    with pytest.raises(SystemExit, match="pins no matrix file digest"):
        P.record_matrix_sha256("# a record without the row\n")
    with pytest.raises(SystemExit, match="is not the matrix the report / record pins"):
        P.frozen_matrix_source({"matrix": {"sha256": "0" * 64}})
    real = hashlib.sha256((REPO / P.MATRIX_REL).read_bytes()).hexdigest()
    assert P.frozen_matrix_source({"matrix": {"sha256": real}})["sha256"] == real
    assert P.RESULTS_RECORD_REL == A.RESULTS_RECORD_REL == "out/ifeval_canonical/results.md"   # the record's OUTPUT path, one spelling
