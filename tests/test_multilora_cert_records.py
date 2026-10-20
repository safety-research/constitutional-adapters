"""The probe arithmetic pinned with EXACT values, the ONE admission function
under the FUNCTIONAL rule (admitted iff readiness attested + five controls +
gate canary + not slower at S; the greedy-token numerics — the three-way
comparison against the primary floor with both one-sided bounds of a
paired-over-prompts bootstrap — RECORDED with ``gate: false`` and never read;
the zero adapter decided by the SOLO certificate with the concurrent legs
recorded), the schema-dispatched validator (schema-4 sidecars keep
validating), the offline ``readmit`` / ``reevaluate`` that verify every
input's bytes, the record schema and its resume rules, the driver's phase
ordering, its native run_id and its ``Target.model_key`` (the two llama
rows on one model), the lens legs' explicit ``serving_path="lens"``, the
report (no numerics-verdict column; a MIXED historical/new directory), and
the sbatch's syntax + hard-rule pins.

Every external seam of the driver (E's ``bring_up_lora`` /
``run_positive_controls``, capctrl's lens standup, the two gate subprocesses,
the HTTP probes and windows, the GPU wait, JBB, versions) is a fake shaped
like the real return value — nothing here touches a GPU, a network, or E's
code. Synthetic data only: invented token ids, placeholder shas, ``TESTHOST``.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import time
import types
from dataclasses import asdict
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "sdfbase"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import multilora_admission as adm  # noqa: E402
import multilora_cert as mc  # noqa: E402
import multilora_cert_report as rep  # noqa: E402
from sisteer.capeval.serving import proxy as px  # noqa: E402
from sisteer.capeval.serving import serving_attestation as sa  # noqa: E402


def _real_run_state(rel: str):
    """``rel`` under this host's run-state tree (``$RUNSTATE``, sisteer.site: SISTEER_RUN_STATE_ROOT from the environment, else the
    site config), or None where the site names none -- the real records these checks read are on the grading host only."""
    from sisteer import site as _site
    try:
        return _site.expand_path(f"$RUNSTATE/{rel}")
    except (_site.PlaceholderError, _site.SiteConfigError):
        return None

SBATCH = REPO / "scripts/sdfbase/multilora_cert.sbatch"
BASE_REQS = REPO / "configs/capeval/vllm-requirements.txt"
NEMOTRON_OVERLAY = REPO / "configs/capeval/vllm-requirements-nemotron.txt"
#: the hash8 in the overlay venv's name for the CURRENT requirement files (sha256 of
#: base + nemotron overlay; a node builds that venv on first use after the files change);
#: the sbatch's hash formula must reproduce it, or every nemotron job rebuilds
#: a second venv — when the requirement files change on purpose, update this pin
#: together with them
NEMOTRON_VENV_HASH8 = "c24e9f1a"


# --------------------------------------------------------------------------- #
# probe arithmetic — exact values                                             #
# --------------------------------------------------------------------------- #

def test_probe_agreement_first_position_divergence_is_zero_not_full():
    pa = adm.probe_agreement([9, 5, 5], [5, 5, 5])
    assert (pa.d, pa.frac, pa.exact, pa.agree_prefix, pa.denom) == (0, 0.0, False, 0, 3)


def test_probe_agreement_later_divergence():
    pa = adm.probe_agreement([1, 2, 9, 4], [1, 2, 3, 4])
    assert (pa.d, pa.frac, pa.exact) == (2, 0.5, False)


def test_probe_agreement_identical_short_outputs_are_full_agreement_not_over_128():
    pa = adm.probe_agreement([7, 8], [7, 8])
    assert pa.exact and pa.frac == 1.0 and pa.d is None and pa.agree_prefix == 2
    assert pa.frac != 2 / 128


def test_probe_agreement_unequal_lengths_diverge_at_min_len_over_max_len():
    pa = adm.probe_agreement([1, 2], [1, 2, 3])
    assert (pa.d, pa.denom) == (2, 3) and abs(pa.frac - 2 / 3) < 1e-12 and not pa.exact


def test_probe_agreement_both_empty_is_exact_and_counted_empty():
    pa = adm.probe_agreement([], [])
    assert pa.exact and pa.frac == 1.0 and pa.empty
    assert adm.pooled_agreement([([], [])])["n_empty"] == 1


def test_probe_agreement_one_empty_is_a_first_token_divergence():
    assert adm.probe_agreement([1], []).d == 0 and adm.probe_agreement([], [1]).frac == 0.0


def test_first_divergence_index_matches_fidelity_on_every_case():
    from sisteer.capeval.serving import fidelity as fd
    for a, b in [([9, 5, 5], [5, 5, 5]), ([1, 2, 9, 4], [1, 2, 3, 4]), ([7, 8], [7, 8]),
                 ([1, 2], [1, 2, 3]), ([], []), ([1], []), ([], [1]), ([3, 3, 3], [3, 3, 3, 3, 3])]:
        assert adm.first_divergence_index(a, b) == fd.first_divergence_index(a, b), (a, b)


def test_pooled_agreement_exact_values():
    p = adm.pooled_agreement([([7, 8], [7, 8]), ([9, 5, 5, 5], [5, 5, 5, 5])])
    assert abs(p["token_agreement"] - 2 / 6) < 1e-12
    assert abs(p["disagreement"] - 4 / 6) < 1e-12
    assert p["exact_match_rate"] == 0.5 and p["n_diverging"] == 1
    assert p["first_token_disagreement_rate"] == 0.5 and p["n"] == 2 and p["n_empty"] == 0


def test_pooled_agreement_empty_input_has_no_number():
    p = adm.pooled_agreement([])
    assert p["n"] == 0 and p["token_agreement"] is None and p["disagreement"] is None


def _f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def test_float32_ulps_is_the_int32_bit_distance():
    a, b = _f32(-2.5), _f32(-2.5000005)
    ia, ib = (struct.unpack("<i", struct.pack("<f", v))[0] for v in (a, b))
    # both negative: the monotone map is 0x80000000 - unsigned == distance in signed magnitude
    assert adm.float32_ulps(a, b) == abs(ia - ib) == 2
    assert adm.float32_ulps(-1.0, -1.0000001) == 1
    assert adm.float32_ulps(0.0, -0.0) == 0
    assert adm.float32_ulps(-1e-45, 1e-45) == 2      # sign crossing counted through zero
    with pytest.raises(adm.AdmissionError):
        adm.float32_ulps(float("-inf"), -1.0)


def _row(idx, ids, *, lp=-0.5, top5=None, token=None, top5_tokens=None):
    ft = {"id": ids[0] if ids else None, "logprob": lp, "token": token,
          "top5": top5 if top5 is not None else ([[ids[0], lp]] if ids else []),
          "top5_tokens": top5_tokens or []}
    return {"probe_index": idx, "token_ids": list(ids), "first_token": ft}


def test_first_token_ulps_missing_token_counts_missing_and_leaves_token_metrics():
    stacked = {0: _row(0, [11, 2, 3], lp=-0.25)}
    native = {0: _row(0, [12, 2, 3], lp=-0.30, top5=[[12, -0.30], [13, -1.0]])}   # 11 absent
    u = adm.first_token_ulps(native, stacked)
    assert u == {"max": None, "p50": None, "n_present": 0, "n_missing": 1, "match_basis": {"id": 0, "token": 0}}
    # the id comparison already registers d == 0; the token metrics are unchanged by the ulps
    p = adm.pooled_agreement([(native[0]["token_ids"], stacked[0]["token_ids"])])
    assert p["first_token_disagreement_rate"] == 1.0 and abs(p["token_agreement"] - 0.0) < 1e-12


def test_first_token_ulps_present_token_is_the_hand_picked_bit_distance():
    a, b = _f32(-0.75), _f32(-0.7500002)
    stacked = {0: _row(0, [11, 2], lp=a)}
    native = {0: _row(0, [11, 2], lp=b, top5=[[11, b], [5, -3.0]])}
    u = adm.first_token_ulps(native, stacked)
    ia, ib = (struct.unpack("<i", struct.pack("<f", v))[0] for v in (a, b))
    assert u["n_present"] == 1 and u["n_missing"] == 0 and u["max"] == u["p50"] == abs(ia - ib) > 0


def test_first_token_ulps_falls_back_to_token_strings_when_ids_are_unparsed():
    stacked = {0: _row(0, [11], lp=-0.5, token="Hello")}
    native = {0: _row(0, [11], lp=-0.5, top5=[[None, -0.5]], top5_tokens=["Hello"])}
    u = adm.first_token_ulps(native, stacked)
    assert u["n_present"] == 1 and u["match_basis"] == {"id": 0, "token": 1} and u["max"] == 0


def test_rows_by_probe_refuses_duplicates_and_malformed_rows():
    with pytest.raises(adm.AdmissionError):
        adm.rows_by_probe([_row(0, [1]), _row(0, [2])])
    with pytest.raises(adm.AdmissionError):
        adm.rows_by_probe([{"probe_index": 0, "token_ids": "12"}])


# --------------------------------------------------------------------------- #
# the admission function                                                        #
# --------------------------------------------------------------------------- #

BASE = [[1, 2, 3, 4, 5, 6, 7, 8]] * 4 + [[9, 9, 9]] * 2                     # 6 probes, base output
CELL = [[1, 2, 3, 4, 5, 6, 7, 8]] * 3 + [[1, 2, 3, 4, 5, 6, 7, 21]] + [[9, 9, 9], [9, 9, 10, 10]]


def _legs(ids_by_leg):
    return {leg: [_row(i, ids) for i, ids in enumerate(seq)] for leg, seq in ids_by_leg.items() if seq is not None}


def _perturb(seq, probe, pos, new=777):
    out = [list(x) for x in seq]
    out[probe][pos] = new
    return out


def _gate(passed=True, status="completed"):
    if status == "missing":
        return {"ran": False, "status": "missing", "passed": False, "lora_canary_passed": False,
                "noop_lora_passed": False, "noop_passthrough_passed": False, "multi_lora_checked": False,
                "margin_atol_nats": None, "record": None, "reason": "none", "bound": False, "attempt": None}
    return {"ran": True, "status": status, "passed": passed, "lora_canary_passed": passed,
            "noop_lora_passed": passed, "noop_passthrough_passed": passed, "multi_lora_checked": passed,
            "margin_atol_nats": 0.05, "record": "/synthetic/gate_record.json", "reason": None,
            "bound": True, "attempt": {"nonce": "0" * 32, "rc": 0}}


def _selection_record(**over):
    """A PASSED ``selection_honored`` in E's record shape: the structural
    verdict off the engine's per-request LoRA bookkeeping plus the non-gating
    output numerics (``fidelity.run_multi_lora_routing_control``)."""
    rec = {"passed": True, "verdict": "routing_attested", "n_requests": 231, "n_observed": 231,
           "n_missing": 0, "n_ambiguous": 0, "n_foreign": 0, "n_inconsistent": 0, "n_misrouted": 0,
           "epoch": "e" * 32, "runner_kind": "v2",
           "steps": {"n_steps": 231, "n_steps_mixed": 99, "max_distinct_selections_per_step": 3},
           "per_selection": {}, "id_name_map": {"1": "none__lora0", "2": "cell"},
           "numerics": {"gating": False, "rule": "equal-size", "n_mismatched_rows": 0},
           "wall_s": 0.1}
    rec.update(over)
    return rec


def _e_controls(*, ok=True, failed=None):
    """``positive_controls.json`` as E's ``run_positive_controls`` writes it:
    ``controls`` keyed by E's CONTROL_NAMES, each ``{"passed": bool|None}``
    (``selection_honored`` in its structural shape), plus the record-level
    ``passed`` / ``failed_control``."""
    names = adm.CONTROL_NAMES
    controls = {n: {"passed": True, "wall_s": 0.1} for n in names}
    controls["selection_honored"] = _selection_record()
    if failed is not None:
        controls[failed] = {"passed": False, "error": "synthetic"}
        for later in names[names.index(failed) + 1:]:
            controls[later] = {"passed": None, "skipped": "an earlier control failed"}
    return {"engine": "lora_native", "order": list(names), "controls": controls,
            "passed": bool(ok and failed is None), "failed_control": failed, "error": None}


def _controls(n=5, *, noop=True):
    """The controls summary as ``adm.controls_summary`` returns it, with the
    FIVE named verdicts: the rule reads ``verdicts["noop_certificate"]`` as
    ``zero_adapter_exact`` (``noop=False`` -> inexact; ``noop=None`` -> not run
    / absent -> ``zero_adapter_unmeasured``). ``n < 5`` fails the LAST
    ``5 - n`` controls other than the certificate."""
    verdicts = {name: True for name in adm.CONTROL_NAMES}
    for name in [c for c in adm.CONTROL_NAMES if c != adm.ZERO_ADAPTER_CONTROL][::-1][: 5 - n]:
        verdicts[name] = False
    verdicts[adm.ZERO_ADAPTER_CONTROL] = noop
    return {"passed": n, "of": 5, "all_passed": (n == 5 and noop is True), "verdicts": verdicts,
            "problems": ([] if n == 5 else ["x"])}


def _h2h(ratio=1.4, S=121):
    return {"not_slower_at_S": ratio is not None and ratio >= 1.0, "ratio_at_S": ratio, "headline_B": S}


def _cert(passed=True, n_prompts=11):
    """E's ``noop_certificate`` record as ``run_positive_controls`` writes it."""
    return {"passed": passed, "n_prompts": n_prompts, "all_exact": passed, "n_tokens": [48, 19, 5] + [48] * (n_prompts - 3),
            "wall_s": 0.1, "rule": "D1: the zero adapter is exact (choices[0].token_ids equal on every prompt)"}


def _readiness(attested=True, problem=None):
    """The readiness input as the driver records it off the strict-read attestation
    (``adm.readiness_record`` over the fake attestation), or a refused read."""
    if not attested:
        return adm.readiness_record(None, problem=problem or "synthetic: the strict reader refused", path="/synthetic")
    return adm.readiness_record(_fake_attestation("si_cell"), path="/synthetic/serving_attestation.json")


def _inputs(*, native=None, stacked=None, per_row=None, zero=None, none=None, native_rep=None,
            stacked_rep=None, compA=None, compB=None, gate=None, controls=None, h2h=None,
            noop_certificate="default", control_max_tokens=48, readiness="default", attempt_id=None):
    stacked = CELL if stacked is None else stacked
    native = stacked if native is None else native
    per_row = _perturb(stacked, 1, 3) if per_row is None else per_row      # a non-degenerate floor by default
    legs = _legs({"lora_native": native, "lora_native_rep": native if native_rep is None else native_rep,
                  "lora_zero": BASE if zero is None else zero, "lora_none": BASE if none is None else none,
                  "lens_stacked": stacked, "lens_stacked_rep": stacked if stacked_rep is None else stacked_rep,
                  "lens_stacked_compA": stacked if compA is None else compA,
                  "lens_stacked_compB": stacked if compB is None else compB, "lens_per_row": per_row})
    if noop_certificate == "default":
        noop_certificate = _cert()
    if readiness == "default":
        readiness = _readiness()
    return adm.AdmissionInputs(legs=legs, gate_canary=gate or _gate(), controls=controls or _controls(),
                               head_to_head=h2h or _h2h(), n_probes=len(stacked), max_tokens=128,
                               noop_certificate=noop_certificate, control_max_tokens=control_max_tokens,
                               readiness=readiness, attempt_id=attempt_id)


def test_admitted_on_the_four_functional_gates_with_the_numerics_recorded():
    """The FUNCTIONAL rule: readiness attested, 5/5 controls, the gate canary
    passed and not slower at S -> ``admitted``, the two-way verdict following
    the ``gates`` block, no reason. The greedy-token numerics are RECORDED
    whole with ``gate: false`` and the rule's note — here native == stacked on
    every prompt while the primary floor diverges on one, so the recorded
    three-way ``outcome`` is ``admitted`` too (Δ <= 0 on every resample) —
    and no ``numerics.admitted`` verdict exists any more."""
    v = adm.admission_verdict(_inputs(native=CELL, per_row=_perturb(CELL, 1, 3)))
    assert v["verdict"] == "admitted" and v["reasons"] == [] and v["rule"] == adm.ADMISSION_RULE == "functional-2026-09-13"
    assert v["schema_version"] == adm.SCHEMA_VERSION == 5
    assert v["gates"] == {"readiness_attested": True, "controls_all_passed": True, "gate_canary_passed": True,
                          "not_slower_at_S": True, "rule": v["gates"]["rule"]}
    assert tuple(k for k in v["gates"] if k != "rule") == adm.FUNCTIONAL_GATES
    assert v["readiness"]["attested"] is True and v["readiness"]["source"] == adm.READINESS_SOURCE
    assert v["readiness"]["engine"]["enforce_eager"] is False and v["readiness"]["kind"] == "lora_native"
    nm = v["numerics"]
    assert nm["gate"] is False and nm["note"] == adm.NUMERICS_NOTE and "admitted" not in nm
    assert nm["outcome"] == "admitted" and nm["within_floor"] is True and nm["reasons"] == []
    assert v["floor"]["source"] == "per_row_vs_stacked" and not v["floor"]["degenerate"]
    prs = nm["prompt_resampling_sensitivity"]
    assert prs["floor_pair"] == "per_row_vs_stacked" and prs["n_boot"] == 2000 and prs["margin_m"] == 0.0
    assert prs["delta_point"] < 0 and prs["bounds"]["upper95_one_sided"] <= 0.0 and prs["delta_ci95_high"] <= 0.0
    assert prs["bounds"]["lower95_one_sided"] == prs["delta_ci95_low"] and len(prs["caveats"]) >= 3
    assert nm["primary_point_inequality"] is True
    assert v["floor"]["margin"] == adm.DIAGNOSTIC_MARGIN == nm["margin"] and adm.DIAGNOSTIC_MARGIN["m"] == 0.0
    assert "functional" in v["verdict_note"] and adm.validate_admission_record(v) == []


def test_an_inconclusive_greedy_comparison_is_recorded_and_the_standup_is_still_admitted():
    """The shape the schema-4 rule read INCONCLUSIVE (native differs from
    stacked at ONE late token, the floor at an earlier one: Δ < 0 at the point
    but a resample without the floor's prompt has Δ > 0): under the functional
    rule the comparison is a DIAGNOSTIC — ``numerics.outcome == inconclusive``
    with its finding under ``numerics.reasons`` — and the verdict, which never
    reads it, is ``admitted`` with no reason."""
    v = adm.admission_verdict(_inputs(native=_perturb(CELL, 0, 7), per_row=_perturb(CELL, 1, 3)))
    prs = v["numerics"]["prompt_resampling_sensitivity"]
    assert prs["delta_point"] < 0 < prs["delta_ci95_high"] and prs["delta_ci95_low"] <= 0
    assert v["numerics"]["outcome"] == "inconclusive" and v["numerics"]["reasons"] == ["native_disagreement_inconclusive"]
    assert v["verdict"] == "admitted" and v["reasons"] == []
    assert v["numerics"]["primary_point_inequality"] is True and v["numerics"]["within_floor"] is False


def test_the_verdict_is_two_way_and_names_every_failed_gate():
    """Every functional gate, failed one at a time and all at once: the reason
    names the gate, the ``gates`` block carries the bit, the verdict is
    ``not_admitted`` — never ``inconclusive`` (the vocabulary is two-way)."""
    base = dict(native=CELL, per_row=_perturb(CELL, 1, 3))
    r = adm.admission_verdict(_inputs(**base, readiness=_readiness(attested=False, problem="kind='lens'")))
    assert r["verdict"] == "not_admitted" and r["reasons"] == ["readiness_unattested"]
    assert r["gates"]["readiness_attested"] is False and r["readiness"]["attested"] is False
    assert r["readiness"]["problem"] == "kind='lens'" and r["numerics"]["outcome"] == "admitted"
    none = adm.admission_verdict(_inputs(**base, readiness=None))
    assert none["reasons"] == ["readiness_unattested"] and "not recorded" in none["readiness"]["problem"]
    c = adm.admission_verdict(_inputs(**base, controls=_controls(4)))
    assert c["verdict"] == "not_admitted" and c["reasons"] == ["controls_failed"] and c["gates"]["controls_all_passed"] is False
    g = adm.admission_verdict(_inputs(**base, gate=_gate(passed=False, status="failed")))
    assert g["reasons"] == ["gate_canary_failed"] and g["gates"]["gate_canary_passed"] is False
    m = adm.admission_verdict(_inputs(**base, gate=_gate(status="missing")))
    assert m["reasons"] == ["gate_canary_missing"]
    s = adm.admission_verdict(_inputs(**base, h2h=_h2h(ratio=0.97)))
    assert s["reasons"] == ["slower_at_S"] and s["gates"]["not_slower_at_S"] is False
    h = adm.admission_verdict(_inputs(**base, h2h=_h2h(ratio=None)))
    assert h["reasons"] == ["head_to_head_missing"]
    every = adm.admission_verdict(_inputs(**base, readiness=None, controls=_controls(3),
                                          gate=_gate(status="missing"), h2h=_h2h(ratio=0.5)))
    assert every["verdict"] == "not_admitted"
    assert every["reasons"] == ["readiness_unattested", "controls_failed", "gate_canary_missing", "slower_at_S"]
    assert all(every["gates"][k] is False for k in adm.FUNCTIONAL_GATES)
    assert adm.VERDICTS == ("admitted", "not_admitted") and "inconclusive" not in adm.VERDICTS
    for rec in (r, none, c, g, m, s, h, every):
        assert rec["verdict"] in adm.VERDICTS and adm.validate_admission_record(rec) == [], rec["reasons"]


def test_no_margin_option_exists_under_the_functional_rule():
    """The schema-4 rule's admission margin is RETIRED: the numerics gate
    nothing, so there is nothing to amend. ``AdmissionInputs`` has no
    ``margin`` field, the module keeps no ``admission_margin`` /
    ``DEFAULT_MARGIN`` / ``ZERO_CONCURRENT_INEXACT_IS_DEFINITE``, the readmit
    CLI and the driver's parser refuse ``--admission-margin``, and the
    recorded threshold is the literal m = 0 with a source that says what
    it is."""
    with pytest.raises(TypeError):
        adm.AdmissionInputs(legs={}, gate_canary=_gate(), controls=_controls(), head_to_head=_h2h(),
                            n_probes=6, max_tokens=128, margin={"m": 0.5, "source": "x"})
    for gone in ("admission_margin", "DEFAULT_MARGIN", "ZERO_CONCURRENT_INEXACT_IS_DEFINITE"):
        assert not hasattr(adm, gone), gone
    assert adm.DIAGNOSTIC_MARGIN["m"] == 0.0 and "no admission margin exists" in adm.DIAGNOSTIC_MARGIN["source"]
    assert "admission_margin" not in mc.IDENTITY_OPTIONS and "admission_margin_source" not in mc.IDENTITY_OPTIONS
    assert "admission_margin" not in mc.parser_option_dests()
    with pytest.raises(SystemExit):
        mc.build_parser().parse_args(["--model-key", "gemma4e4b", "--repo-dir", "/r", "--out", "/o",
                                      "--admission-margin", "0.05"])
    import io, contextlib
    with contextlib.redirect_stderr(io.StringIO()), pytest.raises(SystemExit):
        adm.main(["readmit", "/nowhere", "--admission-margin", "0.05"])


def _seqs(n=43, L=128):
    """``n`` distinct probe outputs of ``L`` tokens (the certification's shape)."""
    return [[i * 1000 + t + 1 for t in range(L)] for i in range(n)]


def _diverge(seqs, at):
    """``at`` = ``{probe: position}``: each named probe's output differs from
    position ``position`` on (so ``agree_prefix == position``)."""
    out = [list(x) for x in seqs]
    for probe, pos in at.items():
        for t in range(pos, len(out[probe])):
            out[probe][t] = 900_000 + probe * 1000 + t
    return out


def test_a_resolved_greedy_excess_is_recorded_and_refuses_nothing():
    """The shape the schema-4 rule REFUSED (native diverges at token 1 on
    EVERY prompt while the primary floor diverges at token 7 — Δ = 6/8 on
    every resample, the one-sided LOWER bound far above 0): under the
    functional rule the excess is a RECORDED finding (``numerics.outcome ==
    not_admitted``, ``native_disagreement_exceeds_floor`` under
    ``numerics.reasons``) and the standup, whose four gates hold, is
    ``admitted`` with no reason — production decodes by sampling."""
    n = 43
    base = [[1, 2, 3, 4, 5, 6, 7, 8]] * n
    native = _diverge(base, {i: 1 for i in range(n)})
    per_row = _diverge(base, {i: 7 for i in range(n)})
    v = adm.admission_verdict(_inputs(stacked=base, native=native, per_row=per_row, zero=base, none=base))
    assert v["verdict"] == "admitted" and v["reasons"] == []
    assert v["numerics"]["outcome"] == "not_admitted" and v["numerics"]["reasons"] == ["native_disagreement_exceeds_floor"]
    prs = v["numerics"]["prompt_resampling_sensitivity"]
    assert abs(prs["delta_point"] - 6 / 8) < 1e-12 and prs["bounds"]["lower95_one_sided"] > 0.5
    assert v["numerics"]["within_floor"] is False and v["numerics"]["primary_point_inequality"] is False
    assert v["numerics"]["gate"] is False and adm.validate_admission_record(v) == []


def test_sparse_single_prompt_excess_is_a_recorded_inconclusive_outcome():
    """A single-prompt excess on 6 probes: the point inequality is False, a
    resample without that prompt has Δ <= 0 and one with it twice has Δ > 0 —
    neither bound clears 0, so the RECORDED three-way outcome is
    ``inconclusive``; the verdict never reads it."""
    v = adm.admission_verdict(_inputs(native=_perturb(CELL, 0, 1), per_row=_perturb(CELL, 1, 7)))
    prs = v["numerics"]["prompt_resampling_sensitivity"]
    assert v["numerics"]["primary_point_inequality"] is False and prs["delta_point"] > 0
    assert prs["delta_ci95_low"] <= 0 < prs["delta_ci95_high"]
    assert v["numerics"]["outcome"] == "inconclusive" and v["numerics"]["reasons"] == ["native_disagreement_inconclusive"]
    assert v["verdict"] == "admitted" and v["reasons"] == []


def test_degenerate_floor_with_gate_canary_passed_records_the_oracle_source():
    """The degenerate-floor fallback, RECORDED: the PRIMARY has 0 flips -> the
    gate canary is the diagnostic yardstick, no bootstrap runs (``skipped``
    names why), outcome ``degenerate_gate_canary``; the verdict reads the gate
    canary as a GATE in every case anyway. A noisy native repeat changes
    nothing: it is a recorded diagnostic, not a floor candidate."""
    v = adm.admission_verdict(_inputs(native=_perturb(CELL, 0, 7), per_row=CELL))
    assert v["floor"]["degenerate"] and v["floor"]["source"] == "gate_canary_neartie_oracle"
    assert v["verdict"] == "admitted" and v["reasons"] == [] and v["numerics"]["within_floor"] is None
    assert v["numerics"]["outcome"] == "degenerate_gate_canary"
    assert "degenerate" in v["numerics"]["prompt_resampling_sensitivity"]["skipped"]
    assert adm.validate_admission_record(v) == []
    native = _perturb(CELL, 0, 7)
    noisy = adm.admission_verdict(_inputs(native=native, per_row=CELL, native_rep=_perturb(native, 2, 2)))
    assert noisy["floor"]["degenerate"] and noisy["floor"]["source"] == "gate_canary_neartie_oracle"
    assert noisy["verdict"] == "admitted" and noisy["numerics"] == v["numerics"]
    assert noisy["floor"]["diagnostics"]["pairs"]["native_rep_vs_native"]["n_diverging"] == 1
    assert noisy["floor"]["diagnostics"]["native_variability_exceeds_lens"] is True


def test_degenerate_floor_with_gate_canary_missing_is_not_admitted_by_the_gate():
    v = adm.admission_verdict(_inputs(per_row=CELL, gate=_gate(status="missing")))
    assert v["verdict"] == "not_admitted" and v["reasons"] == ["gate_canary_missing"]
    assert "floor_degenerate_and_gate_canary_missing" in v["numerics"]["reasons"]     # the recorded finding


def test_degenerate_floor_with_gate_canary_failed_is_not_admitted_by_the_gate():
    v = adm.admission_verdict(_inputs(per_row=CELL, gate=_gate(passed=False, status="failed")))
    assert v["reasons"] == ["gate_canary_failed"] and v["verdict"] == "not_admitted"
    assert "floor_degenerate_and_gate_canary_failed" in v["numerics"]["reasons"]


def test_gate_canary_failed_with_passing_non_degenerate_floor_is_not_admitted():
    v = adm.admission_verdict(_inputs(native=CELL, per_row=_perturb(CELL, 1, 3),
                                      gate=_gate(passed=False, status="failed")))
    assert v["verdict"] == "not_admitted" and v["reasons"] == ["gate_canary_failed"]
    assert v["numerics"]["within_floor"] is True and v["numerics"]["outcome"] == "admitted"   # recorded; the gate decided


def test_zero_adapter_exactness_is_the_solo_certificate_and_the_concurrent_legs_are_recorded():
    """``zero_adapter_exact`` IS the standup's SOLO
    certificate (``noop_certificate``, one request at a time, N x T recorded);
    the concurrent lora_zero vs lora_none legs (43 x 128) are RECORDED under
    ``zero_adapter.concurrent`` with their disagreement and whether the two
    instruments disagree — and refuse nothing. Solo passed but concurrent
    inexact -> exact, ``instruments_disagree`` True, the concurrent finding
    under ``numerics.reasons``, the verdict ``admitted``. A FAILED solo
    certificate fails the ``noop_certificate`` control and reaches the verdict
    as ``controls_failed``. An unmeasured concurrent leg is a recorded
    finding, not a refusal (the unit inventory, not the verdict, requires
    every leg's rows)."""
    v = adm.admission_verdict(_inputs(native=CELL, per_row=_perturb(CELL, 1, 3), zero=_perturb(BASE, 2, 0)))
    assert v["zero_adapter_exact"] is True and v["zero_adapter"]["source"] == adm.ZERO_ADAPTER_SOURCE == "noop_certificate"
    assert v["verdict"] == "admitted" and v["reasons"] == []
    za = v["zero_adapter"]
    assert za["instruments"] == {"concurrent_legs": "failed", "solo_certificate": "passed"}
    assert za["failed_instruments"] == ["concurrent_legs"] and za["instruments_disagree"] is True
    assert za["concurrent"]["n_diverging"] == 1 and za["concurrent"]["n"] == 6 and za["concurrent"]["max_tokens"] == 128
    assert za["concurrent"]["exact"] is False and za["concurrent"]["measured"] is True and za["concurrent"]["recorded_only"] is True
    assert "recorded" in za["concurrent"]["label"]
    assert za["solo_certificate"]["passed"] is True and za["solo_certificate"]["n_prompts"] == 11
    assert za["solo_certificate"]["max_tokens"] == 48 and za["solo_certificate"]["decides"] is True
    assert za["solo_certificate"]["label"] == "solo zero-adapter token-exact on 11 probes, cap 48: passed"
    assert v["numerics"]["reasons"] == ["zero_adapter_inexact:concurrent_legs"] and v["numerics"]["outcome"] == "admitted"
    assert "definite_refusal" not in za and "contract_question_open" not in za       # the question is ruled
    # an inconclusive primary comparison beside the concurrent zero finding: both recorded, still admitted
    both = adm.admission_verdict(_inputs(native=_perturb(CELL, 0, 7), per_row=_perturb(CELL, 1, 3),
                                         zero=_perturb(BASE, 2, 0)))
    assert both["numerics"]["outcome"] == "inconclusive" and both["verdict"] == "admitted"
    assert both["numerics"]["reasons"] == ["zero_adapter_inexact:concurrent_legs", "native_disagreement_inconclusive"]
    # the solo certificate FAILED (E's control failed, so controls are 4/5): controls_failed, once
    bad = adm.admission_verdict(_inputs(native=CELL, per_row=_perturb(CELL, 1, 3), noop_certificate=_cert(passed=False),
                                        controls=_controls(4, noop=False)))
    assert bad["reasons"] == ["controls_failed"] and bad["verdict"] == "not_admitted"
    assert bad["zero_adapter_exact"] is False and bad["gates"]["controls_all_passed"] is False
    assert bad["zero_adapter"]["instruments"] == {"concurrent_legs": "passed", "solo_certificate": "failed"}
    assert bad["zero_adapter"]["instruments_disagree"] is True
    # a certificate record that says FAILED beside a controls summary that says 5/5: still controls_failed (fail closed)
    lied = adm.admission_verdict(_inputs(native=CELL, per_row=_perturb(CELL, 1, 3), noop_certificate=_cert(passed=False)))
    assert lied["reasons"] == ["controls_failed"] and lied["gates"]["controls_all_passed"] is False
    # no certificate record at all: its verdict falls back to the controls summary; scope unknown
    nocert = adm.admission_verdict(_inputs(native=CELL, per_row=_perturb(CELL, 1, 3), noop_certificate=None,
                                           control_max_tokens=None))
    sc = nocert["zero_adapter"]["solo_certificate"]
    assert sc["passed"] is True and sc["n_prompts"] is None and sc["max_tokens"] is None and "? probes, cap ?" in sc["label"]
    assert nocert["verdict"] == "admitted" and nocert["zero_adapter_exact"] is True
    # the cap falls back to the observed max n_tokens when the control's cap is not known
    obs = adm.admission_verdict(_inputs(native=CELL, per_row=_perturb(CELL, 1, 3), control_max_tokens=None))
    assert obs["zero_adapter"]["solo_certificate"]["max_tokens"] == 48
    assert obs["zero_adapter"]["solo_certificate"]["max_tokens_basis"].startswith("observed")
    # both exact: exact, both instruments passed, no disagreement
    ok = adm.admission_verdict(_inputs(native=CELL, per_row=_perturb(CELL, 1, 3)))
    assert ok["zero_adapter_exact"] is True and ok["zero_adapter"]["instruments"] == {"concurrent_legs": "passed",
                                                                                      "solo_certificate": "passed"}
    assert ok["zero_adapter"]["instruments_disagree"] is False
    # the concurrent legs unmeasured (lora_zero missing): a recorded finding; the four gates decide
    inp = _inputs(native=CELL, per_row=_perturb(CELL, 1, 3))
    legs = {k: v for k, v in inp.legs.items() if k != "lora_zero"}
    un = adm.admission_verdict(adm.AdmissionInputs(legs=legs, gate_canary=_gate(), controls=_controls(),
                                                   head_to_head=_h2h(), n_probes=6, max_tokens=128,
                                                   noop_certificate=_cert(), readiness=_readiness()))
    assert "zero_adapter_unmeasured:concurrent_legs" in un["numerics"]["reasons"]
    assert "leg_missing:lora_zero" in un["numerics"]["reasons"] and un["numerics"]["structural_ok"] is False
    assert un["zero_adapter"]["instruments"]["concurrent_legs"] == "unmeasured" and un["zero_adapter"]["instruments_disagree"] is False
    assert un["verdict"] == "admitted" and un["reasons"] == []


@pytest.mark.parametrize("config", ["admitted", "not_admitted"])
def test_determinism_is_recorded_and_never_read_by_the_verdict_D27(config):
    """Determinism: the two determinism flags are recorded and no reason ever
    names them. Both repeat pairs are DIAGNOSTICS now
    (``floor.diagnostics.pairs``, with their own bands against the primary):
    a flip in either leaves ``numerics`` byte-identical and changes only the
    recorded diagnostics — no reason, no verdict change."""
    kw = dict(native=CELL, per_row=_perturb(CELL, 1, 3))
    if config == "not_admitted":
        kw["h2h"] = _h2h(ratio=0.8)
    base = adm.admission_verdict(_inputs(**kw))
    assert base["diagnostics"]["native_deterministic"] is True and base["diagnostics"]["stacked_deterministic"] is True
    assert (base["verdict"] == "admitted") == (config == "admitted")
    flip_n = adm.admission_verdict(_inputs(**kw, native_rep=_perturb(kw["native"], 2, 2)))
    flip_s = adm.admission_verdict(_inputs(**kw, stacked_rep=_perturb(CELL, 3, 0)))
    assert flip_n["diagnostics"]["native_deterministic"] is False and flip_n["diagnostics"]["stacked_deterministic"] is True
    assert flip_s["diagnostics"]["stacked_deterministic"] is False and flip_s["diagnostics"]["native_deterministic"] is True
    for flipped, pair in ((flip_n, "native_rep_vs_native"), (flip_s, "stacked_rep_vs_stacked")):
        assert flipped["numerics"] == base["numerics"]
        assert flipped["verdict"] == base["verdict"] and flipped["reasons"] == base["reasons"]
        d = flipped["floor"]["diagnostics"]["pairs"][pair]
        assert d["n_diverging"] == 1 and d["delta_vs_primary"]["delta_point"] is not None
        assert base["floor"]["diagnostics"]["pairs"][pair]["n_diverging"] == 0
        assert not any("determin" in r for r in flipped["reasons"])
    assert flip_n["floor"]["diagnostics"]["native_variability_exceeds_lens"] is True
    assert flip_s["floor"]["diagnostics"]["native_variability_exceeds_lens"] is False


def test_ulps_arbitrarily_large_never_gate_D11():
    stacked_rows = [_row(i, ids, lp=-0.001) for i, ids in enumerate(CELL)]
    inp = _inputs(native=CELL, per_row=_perturb(CELL, 1, 3))
    legs = dict(inp.legs)
    legs["lens_stacked"], legs["lora_native"] = stacked_rows, [_row(i, ids, lp=-20.0, top5=[[ids[0], -20.0]])
                                                             for i, ids in enumerate(CELL)]
    v = adm.admission_verdict(adm.AdmissionInputs(legs=legs, gate_canary=_gate(), controls=_controls(),
                                                  head_to_head=_h2h(), n_probes=6, max_tokens=128,
                                                  noop_certificate=_cert(), readiness=_readiness()))
    assert v["first_token_ulps"]["native_vs_stacked"]["max"] > 1_000_000
    assert v["verdict"] == "admitted" and v["reasons"] == []


def test_slower_at_S_is_not_admitted_whatever_the_numerics_say():
    v = adm.admission_verdict(_inputs(native=CELL, per_row=_perturb(CELL, 1, 3), h2h=_h2h(ratio=0.97)))
    assert v["numerics"]["outcome"] == "admitted" and v["verdict"] == "not_admitted" and v["reasons"] == ["slower_at_S"]
    assert v["head_to_head"] == {"not_slower_at_S": False, "ratio_at_S": 0.97, "headline_B": 121}


def test_four_of_five_controls_is_not_admitted_whatever_the_numerics_say():
    v = adm.admission_verdict(_inputs(native=CELL, per_row=_perturb(CELL, 1, 3), controls=_controls(4)))
    assert v["numerics"]["outcome"] == "admitted" and v["verdict"] == "not_admitted" and v["reasons"] == ["controls_failed"]
    assert v["controls"] == {"passed": 4, "of": 5, "all_passed": False, "problems": ["x"]}


def test_missing_leg_is_a_recorded_finding_not_a_gate():
    """A missing probe leg makes the recorded comparison ``unmeasured``
    (``leg_missing:…`` under ``numerics.reasons``, ``structural_ok`` False);
    the four gates still decide. (A unit cannot COMPLETE without every leg —
    the evidence inventory requires them — so this is the verdict function's
    contract, not a way to skip a leg.)"""
    inp = _inputs()
    legs = {k: v for k, v in inp.legs.items() if k != "lens_per_row"}
    v = adm.admission_verdict(adm.AdmissionInputs(legs=legs, gate_canary=_gate(), controls=_controls(),
                                                  head_to_head=_h2h(), n_probes=6, max_tokens=128,
                                                  readiness=_readiness()))
    assert "leg_missing:lens_per_row" in v["numerics"]["reasons"] and v["numerics"]["structural_ok"] is False
    assert v["numerics"]["outcome"] == "unmeasured" and v["numerics"]["floor_measured"] is False
    assert v["verdict"] == "admitted" and v["reasons"] == []


def _as_schema4_d111(v5):
    """A schema-5 record re-shaped into the schema-4 record the same legs
    would have produced (the blocks that rule wrote, its zero source, its
    three-way verdict) — the shape of the three historical sidecars."""
    v4 = json.loads(json.dumps(v5))
    v4["schema_version"], v4["rule"] = 4, "D111"
    v4.pop("readiness"); v4.pop("gates")
    nm = v4["numerics"]
    nm.pop("gate"); nm.pop("note")
    nm["admitted"] = bool(v5["zero_adapter"]["concurrent"]["exact"] and v5["gate_canary"]["passed"]
                          and nm["outcome"] in ("admitted", "degenerate_gate_canary"))
    za = v4["zero_adapter"]
    za["source"] = adm.ZERO_ADAPTER_SOURCE_V4
    za["exact"] = bool(v5["zero_adapter"]["concurrent"]["exact"])
    za.pop("instruments_disagree"); za["contract_question_open"] = False; za["definite_refusal"] = not za["exact"]
    v4["zero_adapter_exact"] = za["exact"]
    v4["verdict"] = ("admitted" if nm["admitted"] and v5["controls"]["all_passed"] and v5["head_to_head"]["not_slower_at_S"]
                     else ("inconclusive" if nm["outcome"] == "inconclusive" else "not_admitted"))
    return v4


def test_admission_record_schema_v5_keys_and_the_validator_dispatches_by_schema():
    """Schema 5 (the functional rule): the schema-4 blocks kept whole
    (agreement, ulps, floor with its margin and diagnostics, the numerics'
    three-way ``outcome`` and ``prompt_resampling_sensitivity``, the
    zero-adapter block with both instruments), plus ``readiness`` and
    ``gates``, ``numerics.gate`` false with NO ``numerics.admitted``, the
    two-way verdict following the gates. The validator judges a record
    against ITS OWN schema: a schema-4 record claiming the schema-4 rule
    validates (the three historical sidecars keep validating); a schema-4
    record claiming the functional rule refuses; a schema-5 record claiming
    the schema-4 rule refuses; an unknown schema refuses by name; a
    schema-5 record whose verdict does not follow its gates, or that
    carries a numerics verdict, is named."""
    v = adm.admission_verdict(_inputs())
    assert v["schema_version"] == 5 and v["rule"] == adm.ADMISSION_RULE
    assert adm.SCHEMA_RULES == {4: ("D111",), 5: (adm.ADMISSION_RULE,)}
    assert adm.RULE_ORDER == ("D102", "D111", adm.ADMISSION_RULE)
    for k in adm.REQUIRED_ADMISSION_KEYS:
        assert k in v, k
    assert set(adm.REQUIRED_ADMISSION_KEYS) == set(adm.REQUIRED_ADMISSION_KEYS_V4) | {"readiness", "gates"}
    assert set(v["agreement"]) == set(adm.PAIRS) == {"native_vs_stacked", "per_row_vs_stacked", "stacked_compA_vs_compB",
                                                     "native_rep_vs_native", "stacked_rep_vs_stacked", "native_zero_vs_none"}
    assert set(v["first_token_ulps"]) == set(adm.ULPS_PAIRS)
    for pair in v["agreement"].values():
        for k in ("token_agreement", "disagreement", "exact_match_rate", "n_diverging",
                  "first_token_disagreement_rate", "n_empty", "sum_agree_prefix", "sum_denom"):
            assert k in pair
    assert set(v["floor"]) >= set(adm.REQUIRED_FLOOR_KEYS) and "effective" not in v["floor"]
    assert set(v["floor"]["diagnostics"]["pairs"]) == set(adm.FLOOR_DIAGNOSTIC_PAIRS)
    for d in v["floor"]["diagnostics"]["pairs"].values():
        assert set(d) >= {"disagreement", "n_diverging", "n", "delta_vs_primary"}
    assert set(v["numerics"]) >= set(adm.REQUIRED_NUMERICS_KEYS) and "admitted" not in v["numerics"]
    assert v["numerics"]["gate"] is False and v["numerics"]["note"] == adm.NUMERICS_NOTE
    prs = v["numerics"]["prompt_resampling_sensitivity"]
    assert set(prs) >= set(adm.REQUIRED_SENSITIVITY_KEYS) and set(prs["bounds"]) == {"lower95_one_sided", "upper95_one_sided"}
    assert prs["caveats"] == list(adm.BOOTSTRAP_CAVEATS) and prs["planning"]["n_prompts_for_margin"]
    assert set(v["zero_adapter"]) >= set(adm.REQUIRED_ZERO_KEYS) and v["zero_adapter"]["source"] == "noop_certificate"
    assert set(v["readiness"]) >= set(adm.REQUIRED_READINESS_KEYS) and set(v["gates"]) >= set(adm.FUNCTIONAL_GATES)
    assert v["verdict"] in adm.VERDICTS and v["numerics"]["outcome"] in adm.OUTCOMES
    assert adm.validate_admission_record(v) == []
    # --- dispatch by the record's OWN schema ---
    v4 = _as_schema4_d111(v)
    assert adm.validate_admission_record(v4) == [], adm.validate_admission_record(v4)
    assert adm.validate_admission_record(dict(v4, rule=adm.ADMISSION_RULE)) == [
        f"rule '{adm.ADMISSION_RULE}' is not a schema-4 rule (schema 4 names ['D111'])"]
    assert adm.validate_admission_record(dict(v, rule="D111")) == [
        f"rule 'D111' is not a schema-5 rule (schema 5 names ['{adm.ADMISSION_RULE}'])"]
    for sv in (1, 2, 3, 6, "5", None):
        probs = adm.validate_admission_record(dict(v, schema_version=sv))
        assert len(probs) == 1 and "not a known schema" in probs[0], (sv, probs)
    assert adm.validate_admission_record("nope") == ["record is str, not a JSON object"]
    # a schema-4 D111 record is held to D111's vocabulary, a schema-5 one to the functional
    assert "verdict 'inconclusive' not in ['admitted', 'not_admitted']" in adm.validate_admission_record(dict(v, verdict="inconclusive"))
    assert adm.validate_admission_record(dict(v4, verdict="inconclusive")) == []
    assert any("zero_adapter.source" in p for p in adm.validate_admission_record(dict(v4, zero_adapter=v["zero_adapter"])))
    # schema-5 specifics: the numerics gate, no numerics verdict, readiness and gates present, the verdict follows the gates
    bad = json.loads(json.dumps(v)); bad["numerics"]["gate"] = True; bad["numerics"]["admitted"] = True
    probs = adm.validate_admission_record(bad)
    assert any("numerics.gate True != False" in p for p in probs) and any("numerics carries an `admitted`" in p for p in probs)
    lie = json.loads(json.dumps(v)); lie["gates"]["not_slower_at_S"] = False
    assert any("does not follow the gates" in p for p in adm.validate_admission_record(lie))
    gone = json.loads(json.dumps(v)); gone.pop("readiness"); gone.pop("gates")
    probs = adm.validate_admission_record(gone)
    assert "missing key readiness" in probs and "missing key gates" in probs and "readiness missing" in probs and "gates missing" in probs
    # a D102 (schema 3) record: unknown to this validator, refused by name (its D111 sidecar is what the report reads)
    v3 = json.loads(json.dumps(v4))
    v3["schema_version"], v3["rule"] = 3, "D102"
    assert adm.validate_admission_record(v3) == ["schema_version 3 is not a known schema (known: [4, 5])"]
    # the REAL historical sidecar, when this machine holds it: schema 4, D111, valid
    real = _real_run_state("multilora-cert/gemma4e4b_490087/unit/admission.readmitted.json")
    if real is not None and real.is_file():
        side = json.loads(real.read_text())
        assert (side["schema_version"], side["rule"]) == (4, "D111") and adm.validate_admission_record(side) == []


# --------------------------------------------------------------------------- #
# the D111 rule on shaped legs (the two ceba8874 records, synthetically)        #
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# the D102 rule on shaped legs (the two ceba8874 records, synthetically)        #
# --------------------------------------------------------------------------- #

def _shaped(*, per_row_at, native_at, native_rep_at=None, zero_at=None, n=43, L=128):
    """Legs shaped like a real unit: ``n`` probes of ``L`` tokens; the lens
    per_row leg diverges from stacked at ``per_row_at`` (``{probe: pos}``),
    native at ``native_at``, native_rep from NATIVE at ``native_rep_at``, the
    concurrent zero leg from base at ``zero_at``."""
    stacked = _seqs(n, L)
    native = _diverge(stacked, native_at)
    base = [[7_000_000 + i * 1000 + t for t in range(L)] for i in range(n)]
    return dict(stacked=stacked, native=native, per_row=_diverge(stacked, per_row_at),
                native_rep=(_diverge(native, native_rep_at) if native_rep_at else native),
                zero=(_diverge(base, zero_at) if zero_at else base), none=base)


def test_gemma_shaped_excess_over_the_primary_is_recorded_as_resolved_and_the_standup_admitted():
    """The gemma unit's shape (job 489675 / 490087): primary floor ~0.16
    (10/43), native vs stacked ~0.24 (13/43), the native engine's own repeat
    ~0.25 (12/43) — native diverging EARLIER on a SUPERSET of the prompts the
    floor diverges on, so Δ >= 0 on every resample: the schema-4 rule refused
    this (``native_disagreement_exceeds_floor``, the one-sided LOWER bound
    above 0). Under the functional rule the SAME numbers are recorded — the
    same table, the same band, the same three-way outcome ``not_admitted`` as
    a finding, the native variability that exceeds the lens's — and the
    verdict is ``admitted`` on the four gates (the real record re-evaluates
    the same way: see the mixed report test)."""
    legs = _shaped(per_row_at={i: 40 for i in range(10)}, native_at={i: 26 for i in range(13)},
                   native_rep_at={i: 14 for i in range(23, 35)})
    v = adm.admission_verdict(_inputs(**legs, h2h=_h2h(ratio=5.3, S=107)))
    ag = v["agreement"]
    assert ag["per_row_vs_stacked"]["n_diverging"] == 10 and abs(ag["per_row_vs_stacked"]["disagreement"] - 0.160) < 0.002
    assert ag["native_vs_stacked"]["n_diverging"] == 13 and abs(ag["native_vs_stacked"]["disagreement"] - 0.241) < 0.002
    assert ag["native_rep_vs_native"]["n_diverging"] == 12 and abs(ag["native_rep_vs_native"]["disagreement"] - 0.249) < 0.002
    prs = v["numerics"]["prompt_resampling_sensitivity"]
    assert prs["floor_pair"] == "per_row_vs_stacked" and 0.07 < prs["delta_point"] < 0.09
    assert prs["bounds"]["lower95_one_sided"] > 0 and prs["frac_boot_delta_positive"] > 0.99
    assert v["numerics"]["outcome"] == "not_admitted" and v["numerics"]["primary_point_inequality"] is False
    assert v["numerics"]["reasons"] == ["native_disagreement_exceeds_floor"]
    assert v["verdict"] == "admitted" and v["reasons"] == []
    dg = v["floor"]["diagnostics"]
    assert dg["native_variability"]["pair"] == "native_rep_vs_native" and dg["native_variability_exceeds_lens"] is True
    assert dg["pairs"]["native_rep_vs_native"]["delta_vs_primary"]["delta_point"] > 0
    assert "effective" not in v["floor"] and v["floor"]["source"] == "per_row_vs_stacked"
    assert adm.validate_admission_record(v) == []


def test_nemotron_shaped_near_tie_is_a_recorded_inconclusive_outcome():
    """The nemotron unit's shape (job 489693 / 490088): primary floor ~0.275
    (18/43), native vs stacked ~0.32 (17/43) on DISJOINT prompts. The point
    inequality is False, the lower bound below 0 and the upper above it: the
    RECORDED three-way outcome is ``inconclusive`` (``native_disagreement_
    inconclusive`` under ``numerics.reasons``); the verdict is ``admitted``."""
    legs = _shaped(per_row_at={i: 44 for i in range(18)}, native_at={i: 24 for i in range(20, 37)},
                   native_rep_at={i: 100 for i in range(5)})
    v = adm.admission_verdict(_inputs(**legs, h2h=_h2h(ratio=1.23, S=86)))
    ag = v["agreement"]
    assert ag["per_row_vs_stacked"]["n_diverging"] == 18 and ag["native_vs_stacked"]["n_diverging"] == 17
    assert ag["native_vs_stacked"]["disagreement"] > ag["per_row_vs_stacked"]["disagreement"]
    prs = v["numerics"]["prompt_resampling_sensitivity"]
    assert prs["delta_point"] > 0 and prs["delta_ci95_low"] <= 0 < prs["delta_ci95_high"]
    assert prs["n_prompts"] == 43 and prs["n_prompts_differing"] == 35 and prs["frac_boot_delta_positive"] > 0.5
    assert v["numerics"]["primary_point_inequality"] is False and v["numerics"]["outcome"] == "inconclusive"
    assert v["numerics"]["reasons"] == ["native_disagreement_inconclusive"]
    assert v["verdict"] == "admitted" and v["reasons"] == []
    # the planning number is computed from the paired influence variance and cross-checked by the bootstrap sd
    pl = prs["planning"]
    assert pl["n_prompts_now"] == 43 and pl["sd_influence"] > 0
    assert abs(pl["se_delta_linearised"] - pl["se_delta_bootstrap"]) / pl["se_delta_bootstrap"] < 0.25
    assert pl["n_prompts_for_margin"]["0.05"] > pl["n_prompts_for_margin"]["0.1"] > 43


def test_bootstrap_is_deterministic_in_the_attempt_id_and_its_seed_survives_the_rule_change():
    """The seed is a pure function of the attempt id — the same inputs give
    byte-identical comparisons, and ``paired_bootstrap_delta`` with one seed
    returns the same bounds twice; another attempt id draws other resamples
    (the bounds move, the recorded outcome on decisive data does not). The
    seed's namespace is ``BOOTSTRAP_SEED_NAMESPACE``, the rule the bootstrap
    was specified under, NOT the current admission rule, so a functional
    re-evaluation of a record written under the earlier rule reproduces its
    sidecar's bands exactly (pinned on the real record in the mixed report
    test)."""
    assert adm.BOOTSTRAP_SEED_NAMESPACE == "D111" and adm.ADMISSION_RULE not in ("D111",)
    assert adm.bootstrap_seed("a" * 32) == int.from_bytes(
        hashlib.sha256(f"multilora-admission:D111:{'a' * 32}".encode()).digest()[:8], "big")
    legs = _shaped(per_row_at={i: 44 for i in range(18)}, native_at={i: 24 for i in range(20, 37)})
    kw = dict(_inputs(**legs).__dict__)
    a = adm.admission_verdict(adm.AdmissionInputs(**dict(kw, attempt_id="a" * 32)))
    b = adm.admission_verdict(adm.AdmissionInputs(**dict(kw, attempt_id="a" * 32)))
    c = adm.admission_verdict(adm.AdmissionInputs(**dict(kw, attempt_id="b" * 32)))
    key = "prompt_resampling_sensitivity"
    assert a["numerics"][key] == b["numerics"][key] and a["floor"]["diagnostics"] == b["floor"]["diagnostics"]
    assert a["numerics"][key]["seed"] == adm.bootstrap_seed("a" * 32) != c["numerics"][key]["seed"]
    assert a["numerics"][key]["delta_ci95_low"] != c["numerics"][key]["delta_ci95_low"]
    assert a["numerics"][key]["delta_point"] == c["numerics"][key]["delta_point"]
    assert a["numerics"]["outcome"] == c["numerics"]["outcome"] == "inconclusive"
    assert a["verdict"] == c["verdict"] == "admitted"
    # the function itself, twice with one seed
    rows = {leg: adm.rows_by_probe(r, leg=leg) for leg, r in _inputs(**legs).legs.items()}
    per_n, _ = adm.per_probe_agreements(rows, "lora_native", "lens_stacked")
    per_f, _ = adm.per_probe_agreements(rows, "lens_per_row", "lens_stacked")
    x = adm.paired_bootstrap_delta(per_n, per_f, seed=7, n_boot=300)
    y = adm.paired_bootstrap_delta(per_n, per_f, seed=7, n_boot=300)
    assert x == y and x["n_boot"] == 300 and x["seed"] == 7 and x["n_prompts"] == 43 and x["sd_boot"] > 0
    # the resampled statistic is the TABLE's statistic: the full-sample delta equals the table's difference
    table = adm.admission_verdict(_inputs(**legs))["agreement"]
    assert x["delta_point"] == pytest.approx(table["native_vs_stacked"]["disagreement"]
                                             - table["per_row_vs_stacked"]["disagreement"])
    # a paired set with no differing prompt has delta identically 0
    z = adm.paired_bootstrap_delta(per_n, per_n, seed=1, n_boot=50)
    assert z["delta_point"] == 0.0 == z["delta_ci95_low"] == z["delta_ci95_high"] and z["n_prompts_differing"] == 0
    # an empty paired set is unmeasured, never a number
    e = adm.paired_bootstrap_delta({}, per_f, seed=1)
    assert e["n_prompts"] == 0 and e["delta_point"] is None and e["delta_ci95_low"] is None


def test_pooled_agreement_is_the_pool_of_per_probe_agreements():
    """One arithmetic seam: ``pooled_agreement`` over pairs == ``pool_probe_agreements``
    over the per-probe objects == ``1 - pooled_token_agreement`` for the disagreement."""
    pairs = list(zip(CELL, _perturb(CELL, 0, 3)))
    per = [adm.probe_agreement(a, b) for a, b in pairs]
    assert adm.pooled_agreement(pairs) == adm.pool_probe_agreements(per)
    assert adm.pooled_agreement(pairs)["disagreement"] == 1.0 - adm.pooled_token_agreement(per)
    assert adm.pooled_token_agreement([]) == 1.0




# --------------------------------------------------------------------------- #
# the evidence summaries                                                        #
# --------------------------------------------------------------------------- #

def _gate_record(cell, *, lora_passed=True, certs=("passthrough", "lora"), multi=True, status="completed",
                 recorded_at=None):
    return {"status": status, "phase0_passed": [], "tp_note": "not_applicable",
            "kinds": {cell: {"passed_paths": (["lora"] if lora_passed else []),
                             "failed_paths": ([] if lora_passed else ["lora"])}},
            "certificates": [{"backend": "proxy_vllm", "path": p, "mode": "interleaved", "model_revision": "0" * 40,
                              "serving_impl_commit": "1" * 40, "gate_prompt_set_hash": "ab" * 8} for p in certs],
            "multi_lora_checked": multi, "gate_prompt_set_hash": "ab" * 8,
            # the real gate stamps time.time(); a synthetic 0 would predate every attempt
            "recorded_at": (time.time() if recorded_at is None else recorded_at)}


def _bound_gate_dir(gate_dir, record=None, *, rc=0, evidence=True, markers=(), unit=None):
    """A gate dir exactly as the driver leaves it: ``ATTEMPT.json``
    BEFORE the child's outputs, ``canary_run.json`` AFTER — returns the attempt."""
    d = Path(gate_dir)
    attempt = adm.begin_gate_attempt(d, unit=unit or {"harness_cell": "si_cell"})
    if record is not None:
        (d / adm.GATE_RECORD).write_text(json.dumps(record))
    if evidence:
        (d / adm.GATE_EVIDENCE).write_text(json.dumps({"cross_engine_comparator": {"margin_atol_nats": 0.05}}))
    for name, doc in markers:
        (d / name).write_text(json.dumps(doc))
    adm.finish_gate_attempt(d, attempt, rc=rc)
    return attempt


def test_gate_canary_summary_reads_completed_record_and_evidence(tmp_path):
    attempt = _bound_gate_dir(tmp_path, _gate_record("si_cell"))
    s = adm.gate_canary_summary(tmp_path, cell="si_cell")
    assert s["passed"] and s["status"] == "completed" and s["margin_atol_nats"] == 0.05 and s["ran"]
    assert s["lora_canary_passed"] and s["noop_lora_passed"] and s["noop_passthrough_passed"] and s["multi_lora_checked"]
    assert s["bound"] and s["attempt"]["nonce"] == attempt["nonce"] and s["attempt"]["rc"] == 0
    assert s["attempt"]["unit"] == {"harness_cell": "si_cell"}


@pytest.mark.parametrize("kw,why", [
    (dict(certs=("passthrough",)), "no lora no-op certificate"),
    (dict(lora_passed=False), "lora canary not passed"),
    (dict(multi=False), "multi_lora not checked"),
])
def test_gate_canary_summary_completed_but_short_is_not_passed(tmp_path, kw, why):
    _bound_gate_dir(tmp_path, _gate_record("si_cell", **kw))
    s = adm.gate_canary_summary(tmp_path, cell="si_cell")
    assert not s["passed"] and why in s["reason"] and s["bound"]


def test_gate_canary_summary_failed_unavailable_and_missing(tmp_path):
    f = tmp_path / "failed"
    _bound_gate_dir(f, None, rc=4, evidence=False,
                    markers=[("gate_failed.json", {"status": "gate FAILED", "reason": "canary lora refused"})])
    s = adm.gate_canary_summary(f, cell="c")
    assert s["status"] == "failed" and not s["passed"] and s["ran"] and "refused" in s["reason"]
    u = tmp_path / "unavail"
    _bound_gate_dir(u, None, rc=3, evidence=False,
                    markers=[("gate_unavailable.json", {"status": "phase-0 unavailable", "reason": "no venv"})])
    assert adm.gate_canary_summary(u, cell="c")["status"] == "unavailable"
    m = adm.gate_canary_summary(tmp_path / "nothing", cell="c")
    assert m["status"] == "missing" and not m["ran"] and not m["passed"] and not m["bound"]
    assert adm.gate_canary_summary(None, cell="c")["status"] == "missing"


def test_gate_canary_summary_is_bound_to_the_current_attempt(tmp_path):
    """THE FINDING: a rerun into the same out dir kept an earlier attempt's
    successful gate_record.json beside the current attempt's failure /
    timeout, and the reader returned the record first — a failed canary
    supplied the degenerate-floor fallback's evidence from an earlier
    attempt. Now the record counts only under the current attempt's
    binding."""
    # (a) an UNBOUND record (no ATTEMPT.json / canary_run.json) never passes
    d = tmp_path / "unbound"
    d.mkdir()
    (d / adm.GATE_RECORD).write_text(json.dumps(_gate_record("si_cell")))
    s = adm.gate_canary_summary(d, cell="si_cell")
    assert not s["passed"] and s["status"] == "unbound" and not s["bound"]
    assert "not bound to a current attempt" in s["reason"]
    # (b) pass -> FAILURE in the same dir: the old record is moved aside, the
    # current failure is what the reader sees
    d = tmp_path / "pass_then_fail"
    _bound_gate_dir(d, _gate_record("si_cell"))
    assert adm.gate_canary_summary(d, cell="si_cell")["passed"]
    attempt2 = adm.begin_gate_attempt(d, unit={"harness_cell": "si_cell", "attempt": 2})
    assert not (d / adm.GATE_RECORD).exists() and attempt2["stale_moved_to"]
    stale = Path(attempt2["stale_moved_to"])
    assert stale.parent == d and (stale / adm.GATE_RECORD).is_file()          # renamed, never deleted
    assert (stale / adm.GATE_RUN).is_file() and (stale / adm.GATE_ATTEMPT).is_file()
    (d / adm.GATE_FAILED).write_text(json.dumps({"status": "gate FAILED", "reason": "canary refused now"}))
    adm.finish_gate_attempt(d, attempt2, rc=4)
    s = adm.gate_canary_summary(d, cell="si_cell")
    assert not s["passed"] and s["status"] == "failed" and "refused now" in s["reason"]
    assert s["attempt"]["nonce"] == attempt2["nonce"] and s["attempt"]["rc"] == 4
    # (c) pass -> TIMEOUT (rc 124, nothing written): not passed, the exit named
    d = tmp_path / "pass_then_timeout"
    _bound_gate_dir(d, _gate_record("si_cell"))
    attempt2 = adm.begin_gate_attempt(d, unit={"harness_cell": "si_cell"})
    adm.finish_gate_attempt(d, attempt2, rc=124)
    s = adm.gate_canary_summary(d, cell="si_cell")
    assert not s["passed"] and s["status"] == "failed" and "rc=124" in s["reason"] and s["ran"]
    # (d) a record beside a NON-ZERO exit of the current child is not a completion
    d = tmp_path / "record_but_rc"
    _bound_gate_dir(d, _gate_record("si_cell"), rc=1)
    s = adm.gate_canary_summary(d, cell="si_cell")
    assert not s["passed"] and "exited rc=1" in s["reason"]
    # (e) a record REPLACED after the run (sha differs from what the run left)
    d = tmp_path / "replaced"
    _bound_gate_dir(d, _gate_record("si_cell"))
    (d / adm.GATE_RECORD).write_text(json.dumps(_gate_record("si_cell", multi=True, recorded_at=time.time() + 1)))
    s = adm.gate_canary_summary(d, cell="si_cell")
    assert not s["passed"] and "not the record this attempt left" in s["reason"]
    # (f) a record whose recorded_at predates the attempt
    d = tmp_path / "old_stamp"
    _bound_gate_dir(d, _gate_record("si_cell", recorded_at=1.0))
    s = adm.gate_canary_summary(d, cell="si_cell")
    assert not s["passed"] and "predates the attempt" in s["reason"]
    # (g) a run record from ANOTHER attempt (nonce mismatch)
    d = tmp_path / "nonce"
    _bound_gate_dir(d, _gate_record("si_cell"))
    att = json.loads((d / adm.GATE_ATTEMPT).read_text())
    att["nonce"] = "f" * 32
    (d / adm.GATE_ATTEMPT).write_text(json.dumps(att))
    s = adm.gate_canary_summary(d, cell="si_cell")
    assert not s["passed"] and "nonce" in s["reason"] and not s["bound"]
    # the verdict carries the binding through
    rec = adm.admission_verdict(_inputs(gate=s))
    assert rec["gate_canary"]["bound"] is False and "gate_canary_failed" in rec["reasons"]


def test_the_lens_leg_leaves_the_reduction_mode_env_as_it_found_it(tmp_path, monkeypatch):
    """The Modal merge-tree suite on 20a6064e: bring_up_lens set SISTEER_LENS_STACKED_REDUCTIONS for the lens
    leg's child and never restored it, so the in-process certification tests left "1" in the shared pytest
    process and 45 later tinycap / sampling-protocol tests refused REDUCTION_MODE_MISMATCH before reaching the
    refusal they test. The pin is scoped to the leg: unset stays unset, a prior value comes back."""
    monkeypatch.delenv(mc.REDUCTION_ENV_VAR, raising=False)
    rc, out = _run(tmp_path / "w", tmp_path / "gemma4e4b_1", Fakes(native_shift=None, per_row_shift=(1, 2), ratio=1.3))
    assert rc == 0 and mc.REDUCTION_ENV_VAR not in os.environ
    monkeypatch.setenv(mc.REDUCTION_ENV_VAR, "0")
    rc, out = _run(tmp_path / "w2", tmp_path / "gemma4e4b_2", Fakes(native_shift=None, per_row_shift=(1, 2), ratio=1.3))
    assert rc == 0 and os.environ[mc.REDUCTION_ENV_VAR] == "0"


def test_control_names_are_the_producers_and_the_summary_is_bound_to_them():
    """Item 9: ``controls_summary`` read ANY five booleans as the five
    controls, so a record with the wrong five names passed. It is bound to
    E's ``lora_serving_up.CONTROL_NAMES`` (one tuple, pinned equal here) and
    reads E's record shape."""
    import lora_serving_up as lsu
    assert adm.CONTROL_NAMES == lsu.CONTROL_NAMES
    assert len(adm.CONTROL_NAMES) == adm.N_CONTROLS == 6 and adm.CONTROL_NAMES[-1] == adm.BOUNDARY_CONTROL == "boundary_refusal"
    ok = adm.controls_summary(_e_controls())
    assert ok["all_passed"] and ok["passed"] == ok["of"] == 6 and ok["problems"] == [] and ok["not_applicable"] == []
    assert set(ok["verdicts"]) == set(adm.CONTROL_NAMES)
    # O49: boundary_refusal recorded not_applicable is acceptable ONLY with the EMPTY restricted-cell set stated
    # (an all-positions / LoRA-only standup): `of` then counts the five applicable controls
    na = _e_controls()
    na["controls"]["boundary_refusal"] = {"passed": None, "verdict": adm.NOT_APPLICABLE, "reason": "no restricted-mode adapter",
                                          "restricted_cells": []}
    s_na = adm.controls_summary(na)
    assert s_na["all_passed"] and s_na["passed"] == s_na["of"] == 5 and s_na["not_applicable"] == ["boundary_refusal"]
    for cells in (["si_safety__psr_kl_all"], None):                 # served a restricted cell, or silent about it
        bad = _e_controls()
        bad["controls"]["boundary_refusal"] = {"passed": None, "verdict": adm.NOT_APPLICABLE, "restricted_cells": cells}
        s_bad = adm.controls_summary(bad)
        assert not s_bad["all_passed"] and any("does not state an EMPTY restricted-cell set" in p for p in s_bad["problems"]), cells
    # a record produced BEFORE the sixth control existed (the five-name run order, no boundary entry — the three
    # pulled records of 2026-09-13/14) reads not_applicable (legacy) with its verdicts unchanged, so its own recorded
    # summary block re-derives byte-equal; a five-name order WITH a boundary entry, or a six-name order missing it, does not
    legacy = _e_controls(); legacy["order"] = list(adm.CONTROL_NAMES[:5]); del legacy["controls"]["boundary_refusal"]
    s_leg = adm.controls_summary(legacy)
    assert s_leg["all_passed"] and s_leg["passed"] == s_leg["of"] == 5 and s_leg["not_applicable"] == ["boundary_refusal"]
    assert set(s_leg["verdicts"]) == set(adm.CONTROL_NAMES[:5])
    six_missing = _e_controls(); del six_missing["controls"]["boundary_refusal"]
    assert any("'boundary_refusal' is missing" in p for p in adm.controls_summary(six_missing)["problems"])
    # a FAILED boundary_refusal fails the summary like any other control
    s_fail = adm.controls_summary(_e_controls(failed="boundary_refusal"))
    assert not s_fail["all_passed"] and "boundary_refusal" in " ".join(s_fail["problems"])
    # a bare {name: verdict} mapping in E's names is accepted for the four output/digest
    # controls, but NEVER for selection_honored: its passed verdict must be the structural
    # record (a bare `selection_honored: true` bypassed the rule and admission_verdict
    # read `admitted` with no reasons)
    bare = {n: True for n in adm.CONTROL_NAMES}
    s_bool = adm.controls_summary(bare)
    assert not s_bool["all_passed"] and s_bool["passed"] == 6
    assert any("selection_honored is bool, not the control's record" in p for p in s_bool["problems"])
    assert adm.controls_summary({**bare, "selection_honored": _selection_record()})["all_passed"]
    bare_maps = {n: {"passed": True} for n in adm.CONTROL_NAMES}
    s_bare = adm.controls_summary(bare_maps)
    assert not s_bare["all_passed"] and any("expected 'routing_attested'" in p for p in s_bare["problems"])
    assert adm.controls_summary({**bare_maps, "selection_honored": _selection_record()})["all_passed"]
    # ... and through the admission verdict: the bypass is closed
    v = adm.admission_verdict(_inputs(controls=adm.controls_summary(bare)))
    assert v["verdict"] != "admitted" and v["reasons"]
    # THE FINDING: five booleans under the WRONG names are not the five controls
    wrong = {"zero_adapter_noop": True, "salt_isolation": True, "steering_bites": True,
             "selection_honored": True, "adapter_digest_attestation": True}
    s = adm.controls_summary(wrong)
    assert not s["all_passed"] and s["passed"] == 3
    assert any("'noop_certificate' is missing" in p for p in s["problems"])
    assert any("'adapter_digests' is missing" in p for p in s["problems"])
    assert any("unexpected control verdicts ['adapter_digest_attestation', 'zero_adapter_noop']" in p
               for p in s["problems"])
    # THE STRUCTURAL SHAPE (gemma canary job 489308): a passed selection_honored in
    # E's record shape must carry the routing_attested verdict, >= 1 mixed step, every
    # request observed and the non-gating numerics block — the older output-based
    # record (rule: equal-size at the top, no verdict) fails closed
    pre_d88 = _e_controls()
    pre_d88["controls"]["selection_honored"] = {"passed": True, "rule": "equal-size", "n_layouts": 3,
                                                "selection_first_divergences": []}
    s_old = adm.controls_summary(pre_d88)
    assert not s_old["all_passed"] and s_old["passed"] == 6          # six True verdicts, but problems
    assert any("verdict=None, expected 'routing_attested'" in p for p in s_old["problems"])
    assert any("carries no steps block" in p for p in s_old["problems"])
    assert any("carries no numerics block" in p for p in s_old["problems"])
    for over, match in (({"verdict": "output_equality"}, "verdict='output_equality'"),
                        ({"steps": {"n_steps": 9, "n_steps_mixed": 0, "max_distinct_per_step": 1}},
                         "n_steps_mixed=0, expected >= 1"),
                        ({"n_observed": 230}, "n_observed=230 != n_requests=231"),
                        ({"n_missing": 1}, "n_missing=1, expected 0"),
                        ({"n_requests": 0, "n_observed": 0}, "n_requests=0, expected >= 1"),
                        ({"numerics": {"gating": True}}, "numerics.gating=True"),
                        ({"numerics": None}, "carries no numerics block")):
        bad = _e_controls()
        bad["controls"]["selection_honored"] = _selection_record(**over)
        summ = adm.controls_summary(bad)
        assert not summ["all_passed"] and any(match in p for p in summ["problems"]), (over, summ["problems"])
        assert adm.selection_control_problems(_selection_record(**over))
    assert adm.selection_control_problems(_selection_record()) == []
    assert adm.selection_control_problems("x") == [
        "selection_honored is str, not the control's record — a bare verdict carries no structural "
        "evidence and fails closed"]
    assert adm.selection_control_problems(True)[0].startswith("selection_honored is bool")
    # a FAILED selection_honored is named as failed, not re-judged on shape
    s_f = adm.controls_summary(_e_controls(failed="selection_honored"))
    assert any("failed: selection_honored" in p for p in s_f["problems"])
    assert not any("verdict=" in p for p in s_f["problems"])
    assert adm.SELECTION_VERDICT_ROUTING_ATTESTED == sa.SELECTION_VERDICT_ROUTING_ATTESTED == "routing_attested"
    # one failed control in E's shape: the later ones are "not run" and named
    s2 = adm.controls_summary(_e_controls(failed="steering_bites"))
    assert not s2["all_passed"] and s2["passed"] == 2
    assert any("failed: steering_bites" in p for p in s2["problems"])
    assert any("'selection_honored' carries no verdict" in p for p in s2["problems"])
    assert any("failed_control='steering_bites'" in p for p in s2["problems"])
    assert any("passed=False" in p for p in s2["problems"])
    # a missing control, a record-level passed False, a non-mapping
    four = _e_controls()
    four["controls"].pop("salt_isolation")
    s3 = adm.controls_summary(four)
    assert not s3["all_passed"] and s3["passed"] == 5 and any("'salt_isolation' is missing" in p for p in s3["problems"])
    lied = _e_controls()
    lied["passed"] = False
    assert not adm.controls_summary(lied)["all_passed"]
    assert not adm.controls_summary(["not", "a", "mapping"])["all_passed"]


def _points(leg, tok_by_B, **over):
    """Valid windows as the driver's ``throughput_leg`` records them: the
    ENGINE's evidence present (``running_mean`` off the gauge, ``server_tokens``
    off the counter, ``throughput_source == "server"``; F2)."""
    pts = [{"leg": leg, "B": B, "rep": r, "tok_s": t * (1 + 0.01 * r), "ms_per_step": 1000.0 * B / t,
            "sm_util_mean": 50.0, "requests_done": 40, "errors": 0, "running_mean": float(B) * 0.95,
            "server_tokens": int(t * 60), "throughput_source": adm.THROUGHPUT_SOURCE_SERVER}
           for B, t in tok_by_B.items() for r in range(2)]
    for p in pts:
        p.update(over)
    return pts


def test_head_to_head_invalidates_errored_and_short_windows():
    """THE FINDING (code review 2026-09-11, P1): windows were averaged without
    reading ``errors`` / ``requests_done`` / ``running_mean``, so a lens leg
    whose requests the proxy rejected after the engine counted their tokens
    (positive engine counters, errored windows) manufactured a passing ratio.
    Invalid windows enter no mean; a headline leg with none valid cannot
    establish ``not_slower_at_S`` and the verdict names why."""
    lens = _points("lens_stacked", {121: 800.0})
    native = _points("lora_native", {121: 1000.0})
    ok = adm.head_to_head({"lens_stacked": lens, "lora_native": native}, headline_B=121)
    assert ok["not_slower_at_S"] and ok["headline_invalid"] == {} and ok["per_B"]["121"]["n_valid"] == {
        "lens_stacked": 2, "lora_native": 2, "lens_per_row": 0}
    # every lens window errored (the proxy post-check rejected them): positive
    # tok_s on the counter, yet the leg is INVALID -> no ratio, not "not slower"
    bad_lens = _points("lens_stacked", {121: 100.0}, errors=3, error_sample="HTTP 400: refused")
    h = adm.head_to_head({"lens_stacked": bad_lens, "lora_native": native}, headline_B=121)
    assert h["ratio_at_S"] is None and not h["not_slower_at_S"] and h["headline_missing"]
    assert h["per_B"]["121"]["lens_stacked_tok_s"] is None
    assert h["headline_invalid"] == {"lens_stacked": ["errors=3 (HTTP 400: refused)"]}
    assert h["per_B"]["121"]["invalid"]["lens_stacked"]["0"] == ["errors=3 (HTTP 400: refused)"]
    rec = adm.admission_verdict(_inputs(h2h=h))
    assert rec["verdict"] == "not_admitted" and rec["numerics"]["outcome"] == "admitted"
    assert rec["reasons"] == ["head_to_head_invalid:lens_stacked"] and rec["gates"]["not_slower_at_S"] is False
    # an errored NATIVE leg is just as invalid
    bad_native = _points("lora_native", {121: 5000.0}, errors=1)
    h = adm.head_to_head({"lens_stacked": lens, "lora_native": bad_native}, headline_B=121)
    assert not h["not_slower_at_S"] and set(h["headline_invalid"]) == {"lora_native"}
    # too few completions, concurrency never reached, fields unrecorded, tok_s missing
    for over, why in ((dict(requests_done=0), "requests_done=0 < 1"),
                      (dict(running_mean=10.0), "running_mean=10.0 < 0.5 x B=121"),
                      (dict(errors=None), "errors unrecorded"),
                      (dict(requests_done=None), "requests_done unrecorded"),
                      (dict(tok_s=None), "tok_s missing")):
        reasons = adm.point_invalid_reasons(_points("lens_stacked", {121: 800.0}, **over)[0])
        assert any(why in r for r in reasons), (over, reasons)
    # one valid rep beside one invalid rep: the valid one carries the mean
    mixed = _points("lens_stacked", {121: 800.0})
    mixed[1].update(errors=2, tok_s=99999.0)
    h = adm.head_to_head({"lens_stacked": mixed, "lora_native": native}, headline_B=121)
    assert h["per_B"]["121"]["n_valid"]["lens_stacked"] == 1 and h["headline_invalid"] == {}
    assert abs(h["per_B"]["121"]["lens_stacked_tok_s"] - 800.0) < 1e-9 and h["not_slower_at_S"]
    # the legacy point shape (no validity fields) is fail-closed: unrecorded == invalid
    legacy = [{"leg": "lens_stacked", "B": 121, "rep": 0, "tok_s": 800.0}]
    h = adm.head_to_head({"lens_stacked": legacy, "lora_native": native}, headline_B=121)
    assert not h["not_slower_at_S"] and "lens_stacked" in h["headline_invalid"]
    assert h["validity"]["min_requests_done"] == adm.MIN_REQUESTS_DONE == 1
    assert h["validity"]["min_running_mean_frac"] == adm.MIN_RUNNING_MEAN_FRAC == 0.5


def test_head_to_head_ratios_and_headline():
    pts = {"lens_stacked": _points("lens_stacked", {8: 100.0, 32: 300.0, 121: 800.0}),
           "lora_native": _points("lora_native", {8: 220.0, 32: 540.0, 121: 1000.0}),
           "lens_per_row": _points("lens_per_row", {121: 700.0})}
    h = adm.head_to_head(pts, headline_B=121, ready_s={"lora_native": 200.0})
    assert h["not_slower_at_S"] and abs(h["ratio_at_S"] - 1.25) < 1e-9 and h["headline_B"] == 121
    assert abs(h["per_B"]["8"]["ratio"] - 2.2) < 1e-9 and h["lens_per_row_tok_s_at_S"] > 0
    slow = adm.head_to_head({"lens_stacked": pts["lens_stacked"],
                             "lora_native": _points("lora_native", {121: 700.0})}, headline_B=121)
    assert not slow["not_slower_at_S"] and slow["ratio_at_S"] < 1.0
    missing = adm.head_to_head({"lens_stacked": pts["lens_stacked"]}, headline_B=121)
    assert missing["ratio_at_S"] is None and not missing["not_slower_at_S"] and missing["headline_missing"]


# --------------------------------------------------------------------------- #
# the driver: fakes for every seam                                              #
# --------------------------------------------------------------------------- #

HCELL_GEMMA = "si_constv2ctl2_full__lora_r4_ce"
FAKE_TREE = "f" * 40


class FakeLaunch:
    base_url = "http://127.0.0.1:1"
    auth_headers = {"Authorization": "Bearer synthetic-vllm-token"}
    api_key = "synthetic-vllm-token"


SYNTH_MODEL, SYNTH_REV = "synthetic/model", "0" * 40
GENUINE_IH, NONE_IH = "cc" * 8, "aa" * 8
PASS_ID, LORA_ID = "proxy_vllm+passthrough+interleaved", "proxy_vllm+lora+interleaved"


def _hx(label):
    """A synthetic 64-hex digest (the loaders require lowercase hex)."""
    return hashlib.sha256(label.encode()).hexdigest()


def _hcells(hcell):
    """One genuine harness cell or a list of them (native-psr-serving: a PSR row
    serves two fitted cells) -> the list."""
    return list(hcell) if isinstance(hcell, (list, tuple)) else [hcell]


def _fake_blocks(hcell):
    """The version-2 adapter blocks (``proxy.ADAPTER_BLOCK_KEYS``) a native
    standup writes: every genuine cell and the zero adapter."""
    out = {}
    for i, h in enumerate(_hcells(hcell)):
        out[h] = {"name": h, "dir": f"/synthetic/bank/{h}", "config_sha256": _hx(f"cfg:{h}"),
                  "weights_sha256": _hx(f"w:{h}"), "r": 4, "petri_intervention_hash": _hx(f"ih:{h}")[:16] if i else GENUINE_IH,
                  "petri_zero_adapter": False, "petri_source_kind": "lora_all"}
    out["none__lora0"] = {"name": "none__lora0", "dir": "/synthetic/bank/zero", "config_sha256": _hx("cfg:zero"),
                          "weights_sha256": _hx("w:zero"), "r": 1, "petri_intervention_hash": NONE_IH,
                          "petri_zero_adapter": True, "petri_source_kind": None}
    return out


def _fake_v2_table(hcell):
    """A routing table ``proxy.load_routing_table`` LOADS (item 11: the fake
    writes the shapes the readers require, so a schema drift fails these
    tests instead of hiding behind a lenient reader)."""
    blocks = _fake_blocks(hcell)
    routes = [
        px.RouteEntry(virtual_model="petri__base", cell_id="none", cell_class="base",
                      serving_identity=PASS_ID, intervention_hash=NONE_IH),
        *[px.RouteEntry(virtual_model=f"petri__{h}", cell_id=h, cell_class="class_a",
                        serving_identity=LORA_ID, intervention_hash=blocks[h]["petri_intervention_hash"])
          for h in _hcells(hcell)],
        px.RouteEntry(virtual_model="petri__none__lora0", cell_id="none__lora0", cell_class="control",
                      serving_identity=LORA_ID, intervention_hash=NONE_IH, control_of="none"),
    ]
    return {"version": 2, "base_model": SYNTH_MODEL, "serving_mode": "interleaved",
            "exact_replay_strategy": None, "base_revision": SYNTH_REV,
            "routes": [asdict(r) for r in routes], "prompted_prefixes": [],
            px.LENS_SERVED_POSITIONS_KEY: {}, px.ADAPTERS_KEY: _fake_blocks(hcell)}


def _fake_attestation(hcell):
    """A ``serving_attestation.json`` C's STRICT reader accepts, built from the
    reader's own key constants (item 11) over the same blocks."""
    blocks = _fake_blocks(hcell)
    names = sorted(blocks)                       # the composer's sorted --lora-modules order
    adapters = {}
    for i, n in enumerate(names):
        b = blocks[n]
        adapters[n] = {"dir": b["dir"], "config_sha256": b["config_sha256"],
                       "weights_sha256": b["weights_sha256"],
                       "fingerprint_bf16": hashlib.sha256(f"fp:{n}".encode()).hexdigest(), "r": b["r"],
                       "lora_int_id": i + 1, "module_keys": ["L0.self_attn.o_proj"],
                       "petri_intervention_hash": (None if b["petri_zero_adapter"] else b["petri_intervention_hash"]),
                       "petri_zero_adapter": b["petri_zero_adapter"], "models_root": b["dir"],
                       # native-psr-serving (schema 2): the kind and the residual identity (a LoRA: None)
                       "petri_source_kind": b.get("petri_source_kind"),
                       "residual_fingerprint": None, "position_mode": None}
        assert set(adapters[n]) == set(sa.ADAPTER_KEYS)
    engine = {"enforce_eager": False, "cudagraph_mode": "FULL_AND_PIECEWISE", "cudagraph_specialize_lora": True,
              "cudagraph_capture_sizes": [1, 2, 4, 8], "max_cudagraph_capture_size": 8,
              "lens_plugin_imported": False, "enable_prefix_caching": True,
              # the FULL capture surface (2026-09-16, additive): the dispatcher's FULL keys
              "cudagraph_full_capture_sizes": [1, 2, 4, 8], "cudagraph_full_capture_max": 8}
    assert set(engine) == set(sa.ENGINE_KEYS) | set(sa.ENGINE_CAPTURE_COVERAGE_KEYS)
    lora_config = {"max_loras": len(names), "max_lora_rank": 8, "max_cpu_loras": len(names),
                   "lora_dtype": "torch.bfloat16", "fully_sharded_loras": False, "target_modules": ["o_proj"]}
    assert set(lora_config) == set(sa.LORA_CONFIG_KEYS)
    worker = [{"id": a["lora_int_id"], "rank": a["r"], "n_modules": 1, "module_keys": a["module_keys"],
               "tensor_dtype": "torch.bfloat16", "fingerprint_bf16": a["fingerprint_bf16"],
               "execution_bindings": {"model.layers.0.self_attn.o_proj": "model.layers.0.self_attn.o_proj"}}
              for a in adapters.values()]
    assert all(set(w) == set(sa.WORKER_ADAPTER_KEYS) for w in worker)
    att = {"kind": sa.SERVING_ATTESTATION_KIND, "schema_version": sa.SERVING_ATTESTATION_SCHEMA_VERSION,
           "model": SYNTH_MODEL, "revision": SYNTH_REV, "attested_at": 0.0, "server_pid": 1,
           "hostname": "TESTHOST", "slurm_job_id": None, "argv_sha256": "0" * 16, "host": "127.0.0.1",
           "port": 1, "nonce": "0" * 32, "expected": {},
           "expected_native_lora": {"n_adapters": len(names), "adapters": {
               n: {k: adapters[n][k] for k in ("dir", "fingerprint_bf16", "module_keys", "r", "lora_int_id")}
               for n in names}},
           "scheduler_geometry": {"rpc_method": "sisteer_scheduler_config", "n_workers": 1, "worker_reports": [{}]},
           "engine": engine, "lora_config": lora_config, "adapters": adapters, "worker_adapters": [worker],
           "id_name_map": {str(a["lora_int_id"]): n for n, a in adapters.items()},
           "name_binding": {"asserted": True, "basis": "synthetic"},
           "execution_modules": {"module_keys": ["L0.self_attn.o_proj"],
                                 "module_names": ["model.layers.0.self_attn.o_proj"],
                                 "n_modules": 1, "other": [],
                                 "strides": {"model.layers.0.self_attn.o_proj": {"in": 24, "out": 32}},
                                 "stride_max": 32},
           # native-psr-serving (schema 2): the residual slot block — unarmed on a LoRA-only unit
           sa.RESIDUAL_SLOT_KEY: {"armed": False, "rpc_method": sa.RESIDUAL_SLOT_RPC_METHOD, "n_workers": 1,
                                  "expected": {"armed": False, "op_name": sa.RESIDUAL_SLOT_OP_NAME,
                                               "n_layers_model": None, "hidden_size": None,
                                               "adapters": {n: {"lora_int_id": adapters[n]["lora_int_id"],
                                                                "residual_fingerprint": None, "kind": None,
                                                                "op": None, "position_mode": None,
                                                                "layers": None, "boundaries": None}
                                                            for n in names}},
                                  "worker_reports": [{"armed": False, "slots": {}, "ledger_size": 0,
                                                      "ledger_fault": False}]}}
    assert set(att["execution_modules"]) == set(sa.EXECUTION_MODULES_KEYS)
    assert set(sa.REQUIRED_TOP_KEYS_V2) <= set(att)
    return att


class FakeStandup:
    def __init__(self, kind, out, *, hcell, sha, S, mode=None, objmap=None, output_diagnostics=True):
        self.kind, self.out, self.hcell, self.sha, self.S, self.mode = kind, Path(out), hcell, sha, S, mode
        #: {harness cell: sha} for EVERY genuine cell (native-psr-serving: a PSR row serves two);
        #: the primary ``hcell`` first
        self.objmap = dict(objmap) if objmap else {hcell: sha}
        self.output_diagnostics = output_diagnostics
        self.proxy_url = f"http://fake/{kind}/v1"
        self.launch = FakeLaunch()
        self.torn_down = False
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "vllm_server.log").write_text("Graph capturing finished in 12 secs\n")
        if kind == "native":
            # LOADABLE shapes (item 11): the cert reads both through the strict
            # readers, so these must be what the readers require
            (self.out / "serving_attestation.json").write_text(json.dumps(_fake_attestation(list(self.objmap))))
            (self.out / "routing_table.json").write_text(json.dumps(_fake_v2_table(list(self.objmap))))
            (self.out / "cells_served.json").write_text(json.dumps({"engine": "lora_native",
                                                                    "output_diagnostics": self.output_diagnostics}))
        else:
            (self.out / "steering_attestation.json").write_text(json.dumps({"lens_reduction_mode": mode}))

    def endpoint(self, eval_name=None):
        if self.kind == "native":
            blocks = _fake_blocks(list(self.objmap))
            return {"engine": "lora_native", "model": SYNTH_MODEL, "revision": SYNTH_REV,
                    "object_artifact_sha256_by_cell": dict(self.objmap),
                    "max_model_len": 4864, "max_num_seqs": self.S,
                    # E's shape: the GENUINE blocks under `adapters`, the zero block apart
                    "adapters": {h: blocks[h] for h in self.objmap}, "zero_adapter": blocks["none__lora0"],
                    "attested_enforce_eager": False, "attested_cudagraph_mode": "FULL_AND_PIECEWISE",
                    "proxy_token": None, "output_diagnostics": self.output_diagnostics,
                    "vmodels": {**{h: f"petri__{h}" for h in self.objmap},
                                "none__lora0": "petri__none__lora0", "none": "petri__base"}}
        return {"proxy_url": self.proxy_url, "lens_reduction_mode": self.mode}

    def tear_down(self):
        self.torn_down = True


class Geom:
    def __init__(self, S, L=4864, strict=True):
        self.max_model_len, self.max_num_seqs = L, S
        self.seqs_source, self.window_source = ("model_budget" if strict else "override"), "table"
        self.budget_tokens, self.budget_source = L * S, "model_budget"


class Fakes:
    """One configurable set of seams; ``deps()`` builds the Deps."""

    #: ``native_shift=None`` is the ADMITTED happy path (native == stacked on every prompt, so
    #: Δ vs the primary floor is <= 0 on every resample and the upper bound clears m = 0);
    #: ``native_shift=(probe, pos)`` shifts ONE native prompt, which the schema-4 rule reads
    #: INCONCLUSIVE (a resample drawing that prompt twice puts Δ above 0) — the earlier tests
    #: called that shape admitted
    def __init__(self, *, S=121, native_shift=None, per_row_shift=None, zero_break=False, ratio=1.4,
                 gate="passed", controls_ok=True, mamba_once=False, stacked_mode="stacked",
                 oracle_key="harness", native_rep_shift=None, tree=FAKE_TREE, probe_max=128,
                 throughput_errors=None, kernel_rc=0, oracle_rc=0, metrics_unreadable=(),
                 module_dim=32, fail_with_log=None, fail_controls=None):
        self.S, self.native_shift, self.per_row_shift = S, native_shift, per_row_shift
        self.zero_break, self.ratio, self.gate, self.controls_ok = zero_break, ratio, gate, controls_ok
        self.mamba_once, self.stacked_mode, self.oracle_key = mamba_once, stacked_mode, oracle_key
        self.native_rep_shift, self.tree, self.probe_max = native_rep_shift, tree, probe_max
        #: {kind: errors} — every window of that leg records that many request errors
        self.throughput_errors = dict(throughput_errors or {})
        #: the fused-kernel preflight child's exit code (nemotron)
        self.kernel_rc = kernel_rc
        #: the oracle child's exit code when non-zero: it dies and writes no reference (F1)
        self.oracle_rc = oracle_rc
        #: {kind}: the engine's /metrics was unreadable for that leg — the driver's
        #: client-completion fallback shape (F2): no running_mean, no server_tokens,
        #: tok_s None, throughput_source client_fallback, client_tok_s kept
        self.metrics_unreadable = set(metrics_unreadable)
        #: the base's widest o_proj in/out feature (D): 32 stays far under the LoRA kernel
        #: index ceiling; 4096 is gemma-4-E4B / qwen3.5-4B, 5120 nemotron (job 488932)
        self.module_dim = module_dim
        #: a server log the FIRST native standup dies with (no Mamba text: no retry)
        self.fail_with_log = fail_with_log
        #: a control name: the FIRST native standup comes up, then E's run_positive_controls
        #: refuses it (PositiveControlsFailed out of bring_up_lora, the record attached)
        self.fail_controls = fail_controls
        self.hcell = HCELL_GEMMA          # _run sets the target's harness id
        self.native_attempts = 0
        self.oracle_calls = 0
        self.gate_calls = 0
        self.kernel_preflights = 0
        self.subprocess_order = []
        self.lens_geoms = []
        #: every kwargs dict the lens legs handed to capctrl_bring_up (the serving_path pin)
        self.lens_kws = []
        #: every model_key resolve_geometry was asked for (the Target.model_key pin)
        self.geometry_model_keys = []
        self.standups = []
        self.env_at_lens = []

    # -- seams -----------------------------------------------------------------
    def bring_up_lora(self, **kw):
        self.native_attempts += 1
        self.last_native_kw = kw
        if self.mamba_once and self.native_attempts == 1:
            out = Path(kw["out"])
            out.mkdir(parents=True, exist_ok=True)
            (out / "vllm_server.log").write_text("ValueError: Mamba cache block count 12 < required 24 at max_num_seqs\n")
            raise RuntimeError("vLLM engine init failed (synthetic)")
        if self.fail_with_log is not None and self.native_attempts == 1:
            out = Path(kw["out"])
            out.mkdir(parents=True, exist_ok=True)
            (out / "vllm_server.log").write_text(self.fail_with_log)
            raise RuntimeError("VllmServerError: vllm serve exited with 1 before becoming ready (synthetic)")
        if self.fail_controls is not None and self.native_attempts == 1:
            out = Path(kw["out"])
            out.mkdir(parents=True, exist_ok=True)
            (out / "vllm_server.log").write_text("INFO kernel_warmup finished; graphs captured; READY\n")
            record = _e_controls(ok=False, failed=self.fail_controls)
            (out / "positive_controls.json").write_text(json.dumps(record))
            raise mc.lsu.PositiveControlsFailed(
                f"positive control {self.fail_controls!r} FAILED: FidelityError: synthetic", record)
        objmap = {mc.harness_cell_id(spec.split("=", 1)[0]): spec.split("=", 1)[1]
                  for spec in (kw.get("expect_object_sha256") or [])} or {self.hcell: self.sha}
        s = FakeStandup("native", kw["out"], hcell=self.hcell, sha=self.sha, S=kw["max_num_seqs"], objmap=objmap,
                        output_diagnostics=(kw.get("output_diagnostics") is not False))
        self.standups.append(s)
        return s

    def run_positive_controls(self, standup):
        # E's record shape under E's names (item 9)
        return _e_controls(ok=self.controls_ok, failed=(None if self.controls_ok else adm.CONTROL_NAMES[-1]))

    def capctrl_bring_up(self, **kw):
        self.lens_geoms.append(kw["geom"])
        self.lens_kws.append(dict(kw))
        self.env_at_lens.append(os.environ.get(mc.REDUCTION_ENV_VAR))
        mode = "stacked" if os.environ.get(mc.REDUCTION_ENV_VAR) == "1" else "per_row"
        attested = self.stacked_mode if mode == "stacked" else "per_row"
        s = FakeStandup("stacked" if mode == "stacked" else "per_row", kw["out"], hcell=self.hcell, sha=self.sha,
                        S=kw["geom"].max_num_seqs, mode=attested)
        self.standups.append(s)
        return s

    def attested_reduction_mode_at(self, path):
        d = json.loads(Path(path).read_text())
        return d.get("lens_reduction_mode"), None

    def resolve_geometry(self, eval_name, *, model_key, max_model_len, max_num_seqs, strict_identity):
        self.geometry_model_keys.append(model_key)
        return Geom(max_num_seqs or self.S, max_model_len or 4864, strict=strict_identity)

    def run_subprocess(self, argv, log_path, timeout_s):
        argv = [str(a) for a in argv]
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text(" ".join(argv))
        if len(argv) >= 3 and argv[1] == "-c" and "mamba_ssm" in argv[2]:
            self.kernel_preflights += 1
            self.subprocess_order.append("kernel_preflight")
            return self.kernel_rc
        if "gate-worker-reference" in argv:
            self.subprocess_order.append("oracle")
        elif "gate" in argv:
            self.subprocess_order.append("gate")
        if "gate-worker-reference" in argv:
            self.oracle_calls += 1
            if self.oracle_rc:
                return self.oracle_rc          # the child died; no reference written
            objects = json.loads(Path(argv[argv.index("--objects-json") + 1]).read_text())
            expect = [argv[i + 1] for i, a in enumerate(argv) if a == "--expect-object-sha256"]
            out = Path(argv[argv.index("--out") + 1])
            self.last_oracle_objects = dict(objects)
            def _key(cell):
                return cell if self.oracle_key == "harness" else "constv2ctl2-full__lora_r4_ce"
            ref = {"stage": "completed", "cell_id_space": ("harness" if self.oracle_key == "harness" else "traintest"),
                   "model": self.model, "revision": self.revision,
                   # E's producer records BOTH spellings: the 64-hex over the served
                   # template bytes and its 16-hex prefix under the manifest's key
                   "chat_template_sha256": self.template + "f" * (64 - len(self.template)),
                   "chat_template_hash": self.template,
                   "gate_prompt_set_hash": "gh" * 8, "gate_max_tokens": 64,
                   "objects": {_key(cell): {"object_pt": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
                               for cell, path in objects.items()},
                   "prompt_ids": [[1, 2]] * 6, "base": {"k": [1, 2, 3]},
                   "arms": {_key(cell): {"k": [1, 2, 4]} for cell in objects},
                   "tree_sha": self.tree, "torch_version": "2.11", "transformers_version": "5.14.1",
                   "produced_at": 0, "expect": (expect[0] if len(expect) == 1 else expect)}
            out.write_text(json.dumps(ref))
            return 0
        if "gate" in argv:
            self.gate_calls += 1
            out_dir = Path(argv[argv.index("--out-dir") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            self.last_gate_argv = argv
            if self.gate == "passed":
                (out_dir / "gate_record.json").write_text(json.dumps(_gate_record(self.hcell)))
                (out_dir / "gate_evidence.json").write_text(json.dumps({"cross_engine_comparator": {"margin_atol_nats": 0.05}}))
                return 0
            if self.gate == "timeout":
                return 124          # run_subprocess's TimeoutExpired code: nothing written
            (out_dir / "gate_failed.json").write_text(json.dumps({"status": "gate FAILED", "reason": "synthetic canary failure"}))
            return 4
        raise AssertionError(argv)

    def probe_leg(self, proxy_url, api_key, vmodel, probes, *, max_tokens, concurrency, fillers=0, filler_vmodel=None):
        kind = proxy_url.split("/")[3]
        rows = []
        for i, msgs in enumerate(probes):
            n = 6 + (i % 5)
            ids = [1000 + i] + [(i * 7 + k) % 50 for k in range(1, n)]
            if vmodel in ("petri__base", "petri__none__lora0"):
                ids = [2000 + i] + ids[1:]
                if vmodel == "petri__none__lora0" and self.zero_break and i == 2:
                    ids[3] = 999
            else:
                shift = {"native": self.native_shift, "per_row": self.per_row_shift}.get(kind)
                if shift and i == shift[0]:
                    ids[shift[1]] = 888
            lp = -0.5 - 0.001 * i
            rows.append({"probe_index": i, "vmodel": vmodel, "token_ids": ids, "n_tokens": len(ids),
                         "finish_reason": "stop", "first_token": {"id": ids[0], "logprob": lp, "token": f"t{ids[0]}",
                                                                  "top5": [[ids[0], lp], [5, -3.0]], "top5_tokens": [f"t{ids[0]}", "t5"]},
                         "prompt_token_ids": [1, 2, 3], "stamps": {"engine": kind, "serving_identity": "proxy_vllm+lora+interleaved"}})
        return rows, []

    def throughput_leg(self, proxy_url, api_key, vmodel, *, metrics_url, metrics_headers, batches, repeats, max_tokens,
                       warmup_s, window_s, out_dir, leg):
        kind = proxy_url.split("/")[3]
        base = {"native": 800.0 * self.ratio, "stacked": 800.0, "per_row": 760.0}[kind]
        errors = int(self.throughput_errors.get(kind, 0))
        unreadable = kind in self.metrics_unreadable
        pts = []
        for B in batches:
            for r in range(repeats):
                tok = base * (B / 121) ** 0.5 * (1 + 0.005 * r)
                pt = {"leg": leg, "vmodel": vmodel, "B": int(B), "rep": r, "max_tokens": max_tokens,
                      "ms_per_step": 1.0, "sm_util_mean": 55.0, "errors": errors,
                      "error_sample": ("HTTP 400: the proxy refused the request" if errors else None),
                      "requests_done": 24, "window_s": float(window_s), "client_tok_s": tok}
                if unreadable:
                    pt.update(tok_s=None, throughput_source=adm.THROUGHPUT_SOURCE_CLIENT_FALLBACK,
                              running_mean=None, server_tokens=None)
                else:
                    pt.update(tok_s=tok, throughput_source=adm.THROUGHPUT_SOURCE_SERVER,
                              running_mean=float(B), server_tokens=max(1, int(tok * float(window_s))))
                pts.append(pt)
                # the raw window records the REAL throughput_leg writes (A2: inventoried evidence)
                Path(out_dir).mkdir(parents=True, exist_ok=True)
                (Path(out_dir) / f"gauges_{leg}_B{B}_r{r}.json").write_text(json.dumps([{"t": 0.0, "running": B}]))
                (Path(out_dir) / f"util_{leg}_B{B}_r{r}.csv").write_text("ts, 55, 1000\n")
        return {"leg": leg, "vmodel": vmodel, "B_list": list(batches), "points": pts,
                "metrics_source": ("engine /metrics unreadable" if unreadable
                                   else "vllm:generation_tokens_total via /metrics (Bearer)"),
                "throughput_source": (adm.THROUGHPUT_SOURCE_CLIENT_FALLBACK if unreadable
                                      else adm.THROUGHPUT_SOURCE_SERVER),
                "exact_length_decode": True}

    def wait_gpu_free(self):
        return {"drained": True, "waited_s": 0.0, "used_mib": 0}

    def load_jbb_first(self, n):
        goals = sorted((f"jbb_harmful_{i}", f"synthetic goal {i}") for i in range(1, 101))[:n]
        return goals, "1" * 40

    def gate_constants(self):
        prompts = tuple(({"role": "user", "content": f"gate prompt {i}"},) for i in range(6))
        return prompts, "gh" * 8, 64

    def probe_prompts(self):
        return tuple(f"probe prompt {i}" for i in range(5))

    def manifest_template_hash(self, path):
        return self.template

    def versions(self):
        return {"vllm": "0.25.1", "torch": "2.11.0", "transformers": "5.14.1", "vllm-lens": "1.2.1"}

    def tree_sha(self, repo):
        return self.tree

    def snapshot_cached(self, model, rev):
        return True

    def verify_adapter_dir(self, d, expected):
        return None

    def served_models(self, proxy_url, api_key):
        genuine = [f"petri__{h}" for h in (self.standups[-1].objmap if self.standups else {self.hcell: None})]
        return ["petri__base", "petri__none__lora0", *genuine, "petri__prompted__safety"]

    def lora_max_module_dim(self, model, revision):
        return int(self.module_dim), "synthetic.layers.0.self_attn.o_proj"

    def deps(self):
        return mc.Deps(bring_up_lora=self.bring_up_lora, run_positive_controls=self.run_positive_controls,
                       capctrl_bring_up=self.capctrl_bring_up, attested_reduction_mode_at=self.attested_reduction_mode_at,
                       resolve_geometry=self.resolve_geometry, run_subprocess=self.run_subprocess,
                       probe_leg=self.probe_leg, throughput_leg=self.throughput_leg, wait_gpu_free=self.wait_gpu_free,
                       load_jbb_first=self.load_jbb_first, gate_constants=self.gate_constants,
                       probe_prompts=self.probe_prompts, manifest_template_hash=self.manifest_template_hash,
                       versions=self.versions, tree_sha=self.tree_sha, snapshot_cached=self.snapshot_cached,
                       verify_adapter_dir=self.verify_adapter_dir, served_models=self.served_models,
                       lora_max_module_dim=self.lora_max_module_dim,
                       hostname=lambda: "TESTHOST", python="/synthetic/python")


def _world(tmp_path, key="gemma4e4b", *, object_bytes=b"synthetic object bytes"):
    """A defenses view + pins for the target ROW ``key`` with a synthetic
    object whose sha the pins name — under the row's MODEL key, as the real
    pins.json is keyed (both llama rows live under ``llama8b``); the
    config/manifest are the REAL repo files (read only).

    A QUALIFIED pin key (native-psr-serving §1.6: the stage-2 row's Arditi
    canary, pinned by the misalignment authority) is written into a SECOND file
    BESIDE this one, which is exactly where ``resolve_pin_key`` looks — so a
    test relocates both authorities with ``--pins`` and never mixes a synthetic
    wave pin with the committed misalignment one."""
    t = mc.TARGETS[key]
    defenses = tmp_path / "views" / t.model_key / t.view
    pins = tmp_path / "pins.json"
    pins_doc = {t.model_key: {}}
    extra_docs: dict = {}
    sha = None
    for i, (tt, pin_key) in enumerate(zip(t.cells, t.pin_keys)):
        (defenses / tt).mkdir(parents=True, exist_ok=True)
        # every fitted cell of the row (native-psr-serving: the PSR row's LoRA twin too) gets ITS OWN bytes
        payload = object_bytes if i == 0 else object_bytes + f" [{tt}]".encode()
        (defenses / tt / "object.pt").write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        if mc.PIN_SOURCE_SEP in pin_key:
            rel, _, ref = pin_key.partition(mc.PIN_SOURCE_SEP)
            model, key_name = ref.split(".", 1)
            extra_docs.setdefault(rel, {}).setdefault(model, {})[key_name] = {"object_sha256": digest}
        else:
            pins_doc[t.model_key][pin_key] = {"object_sha256": digest}
        sha = sha if sha is not None else digest          # the PRIMARY cell's sha
    pins.write_text(json.dumps(pins_doc))
    for rel, doc in extra_docs.items():
        (pins.parent / rel).write_text(json.dumps(doc))
    return defenses, pins, sha


def _args(tmp_path, out, *, key="gemma4e4b", defenses=None, pins=None, resume=None, extra=()):
    argv = ["--model-key", key, "--repo-dir", str(REPO), "--out", str(out),
            "--defenses-dir", str(defenses), "--pins", str(pins), "--adapter-bank", str(tmp_path / "bank"),
            "--window-s", "1", "--warmup-s", "0", "--repeats", "2", *extra]
    if resume:
        argv += ["--resume-from", str(resume)]
    return mc.build_parser().parse_args(argv)


def _run(tmp_path, out, fakes, **kw):
    import yaml
    key = kw.pop("key", "gemma4e4b")
    defenses, pins, sha = _world(tmp_path, key, object_bytes=kw.pop("object_bytes", b"synthetic object bytes"))
    cfg = yaml.safe_load((REPO / mc.TARGETS[key].config).read_text())
    fakes.sha, fakes.model, fakes.revision = sha, cfg["model"], cfg["model_revision"]
    fakes.hcell = mc.harness_cell_id(mc.TARGETS[key].tt_cell)
    fakes.template = "0a2c8073c878ab1d"
    args = _args(tmp_path, out, key=key, defenses=defenses, pins=pins, **kw)
    rc = mc.run(args, fakes.deps())
    return rc, Path(out)


def test_driver_end_to_end_record_layout_and_verdict(tmp_path, monkeypatch):
    written = []
    real = mc.write_json_atomic
    monkeypatch.setattr(mc, "write_json_atomic", lambda p, o: (written.append(Path(p)), real(p, o))[1])
    fakes = Fakes(native_shift=None, per_row_shift=(1, 2), ratio=1.4)
    rc, out = _run(tmp_path, tmp_path / "rec" / "gemma4e4b_1001", fakes)
    assert rc == 0
    prep, unit = out / "prep", out / "unit"
    assert (prep / "preflight" / "DONE.json").is_file() and (prep / "oracle" / "DONE.json").is_file()
    assert json.loads((prep / "oracle" / "objects.json").read_text()) == {HCELL_GEMMA: str(tmp_path / "views/gemma4e4b/main/constv2ctl2-full__lora_r4_ce/object.pt")}
    # identity.json FIRST under unit/, DONE.json LAST
    under_unit = [p for p in written if unit in p.parents]
    assert under_unit[0] == unit / "identity.json" and under_unit[-1] == unit / "DONE.json"
    ident = json.loads((unit / "identity.json").read_text())
    assert ident["complete"] and ident["object_artifact_sha256_by_cell"] == {HCELL_GEMMA: fakes.sha}
    assert set(ident["adapter_digests"]) == {HCELL_GEMMA, "none__lora0"} and ident["geometry"] == {"L": 4864, "S": 121}
    assert (unit / "lora_native" / "CANARY_OK").is_file() and (unit / "lora_native" / "positive_controls.json").is_file()
    assert (unit / "lora_native" / "endpoint.json").is_file()
    for leg in adm.LEGS:
        assert (unit / "probes" / f"{leg}.jsonl").is_file() and (unit / "probes" / f"{leg}.DONE.json").is_file(), leg
    done = json.loads((unit / "probes" / "lora_native.DONE.json").read_text())
    assert done["n_rows"] == 43 and done["sha256_of_jsonl"] == hashlib.sha256((unit / "probes" / "lora_native.jsonl").read_bytes()).hexdigest()
    ps = json.loads((unit / "probes" / "probe_set.json").read_text()) if (unit / "probes" / "probe_set.json").is_file() else None
    assert ps is None or ps["n"] == 43
    for leg in ("lora_native", "lens_stacked", "lens_per_row"):
        assert (unit / "throughput" / f"{leg}.json").is_file() and (unit / "throughput" / f"{leg}.DONE.json").is_file()
    assert json.loads((unit / "throughput" / "lens_per_row.json").read_text())["B_list"] == [121]
    assert json.loads((unit / "throughput" / "lora_native.json").read_text())["B_list"] == [8, 32, 121]
    h2h = json.loads((unit / "head_to_head.json").read_text())
    assert h2h["not_slower_at_S"] and h2h["headline_B"] == 121 and abs(h2h["ratio_at_S"] - 1.4) < 1e-6
    adm_rec = json.loads((unit / "admission.json").read_text())
    assert adm.validate_admission_record(adm_rec) == []
    assert adm_rec["verdict"] == "admitted" and adm_rec["floor"]["source"] == "per_row_vs_stacked"
    assert adm_rec["zero_adapter_exact"] and adm_rec["gate_canary"]["passed"] and adm_rec["controls"]["passed"] == adm_rec["controls"]["of"] == 6
    # the FUNCTIONAL rule in the job's own record: readiness = the strict-read attestation of phase 1
    # (what it attested copied), the four gates, no numerics verdict; the solo certificate decides
    # zero-exactness with its scope (11 control prompts at the driver's 48-token cap) and the concurrent
    # legs are recorded; the bootstrap is seeded from THIS attempt, the six pairs are in the table, the
    # margin is the recorded diagnostic's literal m = 0
    assert adm_rec["rule"] == adm.ADMISSION_RULE and adm_rec["schema_version"] == 5
    assert adm_rec["readiness"]["attested"] is True and adm_rec["readiness"]["kind"] == "lora_native"
    assert adm_rec["readiness"]["path"] == str(unit / "lora_native" / "serving_attestation.json")
    assert adm_rec["readiness"]["engine"]["enforce_eager"] is False
    assert set(adm_rec["readiness"]["adapters"]) == {HCELL_GEMMA, "none__lora0"}
    assert all(adm_rec["gates"][k] is True for k in adm.FUNCTIONAL_GATES) and adm_rec["reasons"] == []
    assert adm_rec["numerics"]["gate"] is False and "admitted" not in adm_rec["numerics"]
    assert adm_rec["zero_adapter"]["source"] == "noop_certificate"
    sc = adm_rec["zero_adapter"]["solo_certificate"]
    assert sc["passed"] is True and sc["max_tokens"] == 48 and sc["max_tokens_basis"].startswith("control_probe_max_tokens")
    assert adm_rec["zero_adapter"]["concurrent"]["exact"] is True and adm_rec["zero_adapter"]["instruments_disagree"] is False
    prs = adm_rec["numerics"]["prompt_resampling_sensitivity"]
    assert prs["seed"] == adm.bootstrap_seed(adm_rec["attempt_id"]) and prs["n_prompts"] == 43
    assert adm_rec["numerics"]["outcome"] == "admitted" and adm_rec["numerics"]["within_floor"] is True
    assert set(adm_rec["agreement"]) == set(adm.PAIRS) and adm_rec["floor"]["margin"] == adm.DIAGNOSTIC_MARGIN
    cs = json.loads((unit / "controls_summary.json").read_text())
    assert cs["readiness"] == adm_rec["readiness"] and cs["summary"]["all_passed"] and cs["gate_canary"]["passed"]
    assert (unit / "DONE.json").is_file()
    assert (unit / "gate_canary" / "gate_record.json").is_file()
    assert "--lora-native-standup" in fakes.last_gate_argv and "--reference" in fakes.last_gate_argv
    assert fakes.last_gate_argv[fakes.last_gate_argv.index("--reference") + 1] == str(prep / "oracle" / "reference.json")
    line = (out / "RESULT").read_text().strip()
    assert re.fullmatch(r"state=DONE exit=0 elapsed=\d+ host=TESTHOST target=gemma4e4b verdict=admitted "
                        rf"rule={adm.ADMISSION_RULE} readiness=attested controls=6/6 gate=passed not_slower=True "
                        r"ratio_at_S=1\.400 numerics=diagnostic outcome=admitted floor=per_row_vs_stacked "
                        r"unit_reused_from=none", line), line
    # the lens legs ran at stacked then per_row, env pinned before each bring-up, BOTH asked for
    # serving_path=lens BY NAME (the standup default is native since 2026-09-13), both torn down
    assert fakes.env_at_lens == ["1", "0"] and all(s.torn_down for s in fakes.standups)
    assert [kw["serving_path"] for kw in fakes.lens_kws] == ["lens", "lens"]
    man = json.loads((out / "manifest.json").read_text())
    assert man["harness_cell"] == HCELL_GEMMA and man["lens_cells"] == ["constv2ctl2-full__lora_r4_ce"]
    assert man["prep_reused"] == {"oracle": None} and man["unit_reused_from"] is None


def test_probe_set_is_43_and_hashed(tmp_path):
    fakes = Fakes()
    goals, rev = fakes.load_jbb_first(32)
    ps = mc.build_probe_set(goals, fakes.gate_constants()[0], fakes.probe_prompts(), max_tokens=128, jbb_revision=rev)
    assert ps["n"] == 43 and len(ps["sources"]["jbb_first32"]) == 32 and ps["sources"]["jbb_first32"] == sorted(ps["sources"]["jbb_first32"])
    assert ps["sources"]["gate_prompt_set"] == 6 and ps["sources"]["probe_prompts"] == 5 and ps["max_tokens"] == 128
    assert len(ps["hash"]) == 64
    ps2 = mc.build_probe_set(goals, fakes.gate_constants()[0], fakes.probe_prompts(), max_tokens=64, jbb_revision=rev)
    assert ps2["hash"] != ps["hash"]
    with pytest.raises(mc.CertRefusal):
        mc.build_probe_set(list(reversed(goals)), fakes.gate_constants()[0], fakes.probe_prompts(), max_tokens=128, jbb_revision=rev)


def test_run_identity_is_minted_with_engine_lora_native(tmp_path):
    from sisteer.capeval.serving.run_identity import RunIdentity
    rc, out = _run(tmp_path, tmp_path / "rec" / "gemma4e4b_1002", Fakes(native_shift=None, per_row_shift=(1, 2)))
    assert rc == 0
    rid = json.loads((out / "unit" / "run_identity.json").read_text())
    assert rid["engine"] == "lora_native" and len(rid["run_id"]) == 16
    fields = {k: v for k, v in rid.items() if k not in ("run_id", "geometry_id", "written_at")}
    with_engine = RunIdentity(**fields)
    without = RunIdentity(**dict(fields, engine=None))
    assert with_engine.run_id == rid["run_id"] and with_engine.geometry_id == rid["geometry_id"]
    assert without.run_id != rid["run_id"] and without.geometry_id == rid["geometry_id"]


def test_refuses_to_enter_the_unit_without_the_oracle_done(tmp_path, monkeypatch):
    fakes = Fakes()
    monkeypatch.setattr(mc, "phase_oracle", lambda ctx: None)      # the oracle phase stubbed away
    rc, out = _run(tmp_path, tmp_path / "rec" / "gemma4e4b_1003", fakes)
    assert rc == mc.RC_REFUSED and fakes.native_attempts == 0
    assert not (out / "unit" / "identity.json").exists()
    assert "no complete oracle" in json.loads((out / "FAILED.json").read_text())["error"]


def test_oracle_is_keyed_by_the_harness_id_and_a_traintest_keyed_reference_refuses(tmp_path):
    fakes = Fakes(oracle_key="traintest")
    rc, out = _run(tmp_path, tmp_path / "rec" / "gemma4e4b_1004", fakes)
    assert rc == mc.RC_REFUSED and fakes.native_attempts == 0
    err = json.loads((out / "FAILED.json").read_text())["error"]
    assert "objects keys" in err and HCELL_GEMMA in err and "cell_id_space" in err
    # the request the driver made WAS keyed by the harness id
    assert list(json.loads((out / "prep" / "oracle" / "objects.json").read_text())) == [HCELL_GEMMA]


def test_oracle_bindings_problems_name_each_field():
    # the reference carries BOTH template spellings (E's producer); the binding
    # reads the 16-hex chat_template_hash — the manifest's key — never the 64-hex
    ok = {"stage": "completed", "cell_id_space": "harness", "model": "m", "revision": "r",
          "chat_template_hash": "t" * 16, "chat_template_sha256": "t" * 64,
          "gate_prompt_set_hash": "g", "gate_max_tokens": 64, "objects": {"hc": {"object_pt": "/p", "sha256": "s"}},
          "arms": {"hc": {}}, "base": {"k": [1]}, "tree_sha": "T"}
    kw = dict(model="m", revision="r", chat_template_hash="t" * 16, gate_prompt_set_hash="g", gate_max_tokens=64,
              hcell="hc", object_sha256="s")
    assert mc.oracle_bindings_problems(ok, **kw) == []
    assert mc.oracle_bindings_problems(ok, tree_sha="T", **kw) == []
    assert any("tree_sha" in p for p in mc.oracle_bindings_problems(ok, tree_sha="U", **kw))
    # a reference whose 64-hex differs but whose 16-hex matches still binds (the
    # canary binds the 64-hex to the standup's endpoint, not this reader), and a
    # reference that mismatches the 16-hex refuses by that name
    assert mc.oracle_bindings_problems(dict(ok, chat_template_sha256="u" * 64), **kw) == []
    assert any("chat_template_hash" in p
               for p in mc.oracle_bindings_problems(dict(ok, chat_template_hash="u" * 16), **kw))
    for field, bad in (("model", "x"), ("revision", "x"), ("chat_template_hash", "x"), ("gate_prompt_set_hash", "x"),
                       ("gate_max_tokens", 32), ("cell_id_space", "traintest")):
        probs = mc.oracle_bindings_problems(dict(ok, **{field: bad}), **kw)
        assert any(field in p for p in probs), (field, probs)
    assert any("sha256" in p for p in mc.oracle_bindings_problems(dict(ok, objects={"hc": {"sha256": "z"}}), **kw))
    assert any("arms" in p for p in mc.oracle_bindings_problems(dict(ok, arms={"other": {}}), **kw))
    assert any("error" in p for p in mc.oracle_bindings_problems(dict(ok, error="boom"), **kw))


def test_resume_reuses_a_bound_oracle_and_reruns_it_when_the_object_differs(tmp_path):
    f1 = Fakes(native_shift=None, per_row_shift=(1, 2))
    rc, first = _run(tmp_path / "a", tmp_path / "a" / "rec" / "gemma4e4b_2001", f1)
    assert rc == 0 and f1.oracle_calls == 1
    # same object -> the oracle is reused, the child never runs
    f2 = Fakes(native_shift=None, per_row_shift=(1, 2))
    rc, second = _run(tmp_path / "a", tmp_path / "a" / "rec" / "gemma4e4b_2002", f2, resume=first)
    assert rc == 0 and f2.oracle_calls == 0
    assert json.loads((second / "manifest.json").read_text())["prep_reused"] == {"oracle": str(first / "prep" / "oracle")}
    # a different object sha -> NOT reused, the oracle re-runs
    f3 = Fakes(native_shift=None, per_row_shift=(1, 2))
    rc, third = _run(tmp_path / "b", tmp_path / "b" / "rec" / "gemma4e4b_2003", f3, resume=first,
                     object_bytes=b"a different object")
    assert rc == 0 and f3.oracle_calls == 1
    assert json.loads((third / "manifest.json").read_text())["prep_reused"] == {"oracle": None}
    assert (third / "prep" / "oracle" / "reuse_refused.json").is_file()
    # a different tree sha -> not reused either
    f4 = Fakes(native_shift=None, per_row_shift=(1, 2), tree="e" * 40)
    rc, fourth = _run(tmp_path / "a", tmp_path / "a" / "rec" / "gemma4e4b_2004", f4, resume=first)
    assert rc == 0 and f4.oracle_calls == 1


def test_resume_complete_unit_is_reused_whole_under_its_provenance(tmp_path):
    f1 = Fakes(native_shift=None, per_row_shift=(1, 2))
    rc, first = _run(tmp_path, tmp_path / "rec" / "gemma4e4b_3001", f1)
    assert rc == 0
    f2 = Fakes(native_shift=None, per_row_shift=(1, 2))
    rc, second = _run(tmp_path, tmp_path / "rec" / "gemma4e4b_3002", f2, resume=first)
    assert rc == 0 and f2.native_attempts == 0 and f2.gate_calls == 0 and f2.standups == []
    man = json.loads((second / "manifest.json").read_text())
    assert man["unit_reused_from"] == str(first / "unit") and not (second / "unit").exists()
    assert (second / "RESULT").read_text().strip().endswith(f"unit_reused_from={first / 'unit'}")
    reuse = json.loads((second / "unit_reuse.json").read_text())
    assert reuse["unit_reused_from"] == str(first / "unit") and reuse["reasons"] == []
    # the reuse is SELF-CONTAINED: a verbatim copy under reused_unit/, bound in unit_reuse.json
    copy = second / adm.REUSED_UNIT_DIR
    inventory = json.loads((first / "unit" / adm.UNIT_INVENTORY_FILE).read_text())
    assert inventory["missing"] == [] and inventory["n_files"] == len(inventory["files"]) > 30
    for name in list(inventory["files"]) + [adm.UNIT_INVENTORY_FILE, "DONE.json"]:
        assert (copy / name).read_bytes() == (first / "unit" / name).read_bytes(), name
    assert adm.unit_inventory_problems(copy) == [] and adm.unit_inventory_problems(first / "unit") == []
    src = reuse["source"]
    assert src["unit_dir"] == str(first / "unit") and src["n_files"] == len(inventory["files"]) + 2
    assert src["attempt_id"] == json.loads((first / "unit" / "DONE.json").read_text())["attempt_id"]
    assert src["sha256"]["admission.json"] == src["admission_sha256"] == hashlib.sha256((copy / "admission.json").read_bytes()).hexdigest()
    assert set(src["sha256"]) == set(src["files"]) == set(inventory["files"]) | {adm.UNIT_INVENTORY_FILE, "DONE.json"}
    assert reuse["attempt_id"] == man["attempt_id"] != src["attempt_id"]
    assert adm.reused_unit_binding_problems(second, record_attempt_id=man["attempt_id"], unit_reused_from=str(first / "unit")) == []


def test_resume_partial_unit_contributes_nothing(tmp_path):
    f1 = Fakes(native_shift=None, per_row_shift=(1, 2))
    rc, first = _run(tmp_path, tmp_path / "rec" / "gemma4e4b_3101", f1)
    (first / "unit" / "DONE.json").unlink()                 # a job killed inside the unit
    f2 = Fakes(native_shift=None, per_row_shift=(1, 2))
    rc, second = _run(tmp_path, tmp_path / "rec" / "gemma4e4b_3102", f2, resume=first)
    assert rc == 0 and f2.native_attempts == 1 and (second / "unit" / "DONE.json").is_file()
    reuse = json.loads((second / "unit_reuse.json").read_text())
    assert reuse["unit_reused_from"] is None and any("no DONE.json" in r for r in reuse["reasons"])
    # nothing under the prior unit was copied: this run's legs are its own rows
    assert json.loads((second / "unit" / "probes" / "lora_native.DONE.json").read_text())["n_rows"] == 43


def test_resume_complete_unit_with_a_different_identity_is_not_reused(tmp_path):
    f1 = Fakes(native_shift=None, per_row_shift=(1, 2))
    rc, first = _run(tmp_path, tmp_path / "rec" / "gemma4e4b_3201", f1)
    # sampling differs (max_tokens 64 vs 128)
    f2 = Fakes(native_shift=None, per_row_shift=(1, 2))
    rc, second = _run(tmp_path, tmp_path / "rec" / "gemma4e4b_3202", f2, resume=first, extra=("--probe-max-tokens", "64"))
    assert rc == 0 and f2.native_attempts == 1
    reasons = json.loads((second / "unit_reuse.json").read_text())["reasons"]
    assert any("max_tokens" in r or "probe_set_hash" in r for r in reasons)
    # a prior adapter dir that no longer verifies on disk
    f3 = Fakes(native_shift=None, per_row_shift=(1, 2))
    f3.verify_adapter_dir = lambda d, e: "weights_sha256 differs"
    rc, third = _run(tmp_path, tmp_path / "rec" / "gemma4e4b_3203", f3, resume=first)
    assert rc == 0 and f3.native_attempts == 1
    assert any("weights_sha256 differs" in r for r in json.loads((third / "unit_reuse.json").read_text())["reasons"])
    # a different tree
    f4 = Fakes(native_shift=None, per_row_shift=(1, 2), tree="e" * 40)
    rc, fourth = _run(tmp_path, tmp_path / "rec" / "gemma4e4b_3204", f4, resume=first)
    assert rc == 0 and f4.native_attempts == 1


def test_mamba_refusal_retries_once_at_half_S_and_records_it(tmp_path):
    fakes = Fakes(native_shift=None, per_row_shift=(1, 2), mamba_once=True, S=161)
    rc, out = _run(tmp_path, tmp_path / "rec" / "gemma4e4b_4001", fakes)
    assert rc == 0 and fakes.native_attempts == 2
    assert fakes.last_native_kw["max_num_seqs"] == 80 and fakes.last_native_kw["geometry_override_reason"] == "mamba_cache_blocks_under_graphs"
    ident = json.loads((out / "unit" / "identity.json").read_text())
    assert ident["geometry"] == {"L": 4864, "S": 80} and ident["geometry_lowered"]["from"] == 161 and ident["geometry_lowered"]["to"] == 80
    man = json.loads((out / "manifest.json").read_text())
    assert man["geometry_lowered"]["reason"] == "mamba_cache_blocks_under_graphs"
    # every later leg follows the served width: lens standups at S=80, headline B = 80
    assert [g.max_num_seqs for g in fakes.lens_geoms] == [80, 80]
    assert json.loads((out / "unit" / "head_to_head.json").read_text())["headline_B"] == 80
    assert json.loads((out / "unit" / "throughput" / "lora_native.json").read_text())["B_list"] == [8, 32, 80]
    attempts = json.loads((out / "unit" / "native_standup_attempts.json").read_text())
    assert attempts[0]["mamba_refusal"] and attempts[1]["ok"]


def test_controls_failure_stops_before_traffic_rc4(tmp_path):
    fakes = Fakes(controls_ok=False)
    rc, out = _run(tmp_path, tmp_path / "rec" / "gemma4e4b_5001", fakes)
    assert rc == mc.RC_CONTROLS and fakes.gate_calls == 0
    assert not (out / "unit" / "probes").exists() and not (out / "unit" / "lora_native" / "CANARY_OK").exists()
    assert json.loads((out / "FAILED.json").read_text())["kind"] == "controls"
    assert all(s.torn_down for s in fakes.standups)
    assert "verdict=none" in (out / "RESULT").read_text() and "state=FAILED exit=4" in (out / "RESULT").read_text()


def test_native_standup_refusal_without_mamba_is_rc4_not_retried(tmp_path):
    fakes = Fakes()

    def refuse(**kw):
        fakes.native_attempts += 1
        raise RuntimeError("engine died (synthetic)")
    fakes.bring_up_lora = refuse
    rc, out = _run(tmp_path, tmp_path / "rec" / "gemma4e4b_5002", fakes)
    assert rc == mc.RC_CONTROLS and fakes.native_attempts == 1


def test_gate_canary_failure_is_recorded_and_the_unit_continues(tmp_path):
    fakes = Fakes(native_shift=None, per_row_shift=(1, 2), gate="failed")
    rc, out = _run(tmp_path, tmp_path / "rec" / "gemma4e4b_6001", fakes)
    assert rc == 0 and (out / "unit" / "DONE.json").is_file()
    rec = json.loads((out / "unit" / "admission.json").read_text())
    assert rec["gate_canary"]["status"] == "failed" and rec["verdict"] == "not_admitted"
    assert "gate_canary_failed" in rec["reasons"]
    # degenerate floor + failed canary: the gate refuses by name; the recorded numerics name the fallback finding
    fakes2 = Fakes(native_shift=None, per_row_shift=None, gate="failed")
    rc, out2 = _run(tmp_path, tmp_path / "rec" / "gemma4e4b_6002", fakes2)
    rec2 = json.loads((out2 / "unit" / "admission.json").read_text())
    assert rec2["floor"]["degenerate"] and rec2["reasons"] == ["gate_canary_failed"]
    assert "floor_degenerate_and_gate_canary_failed" in rec2["numerics"]["reasons"]


def test_errored_throughput_windows_cannot_establish_not_slower(tmp_path):
    """Finding 5 through the driver: every lens window errored (the proxy
    rejected the requests after the engine counted their tokens), the native
    leg is clean and 1.4x faster on the counter — the verdict is
    ``not_admitted`` with ``head_to_head_invalid:lens_stacked``, never
    ``admitted`` off a manufactured ratio."""
    fakes = Fakes(native_shift=None, per_row_shift=(1, 2), ratio=1.4, throughput_errors={"stacked": 2})
    rc, out = _run(tmp_path, tmp_path / "rec" / "gemma4e4b_7101", fakes)
    assert rc == 0
    h2h = json.loads((out / "unit" / "head_to_head.json").read_text())
    assert h2h["ratio_at_S"] is None and not h2h["not_slower_at_S"]
    assert h2h["headline_invalid"] == {"lens_stacked": ["errors=2 (HTTP 400: the proxy refused the request)"]}
    assert h2h["per_B"]["121"]["n_valid"]["lens_stacked"] == 0 and h2h["per_B"]["121"]["n_valid"]["lora_native"] == 2
    rec = json.loads((out / "unit" / "admission.json").read_text())
    assert rec["numerics"]["outcome"] == "admitted" and rec["verdict"] == "not_admitted"
    assert rec["reasons"] == ["head_to_head_invalid:lens_stacked"] and rec["gates"]["not_slower_at_S"] is False
    assert (f"verdict=not_admitted rule={adm.ADMISSION_RULE} readiness=attested controls=6/6 gate=passed "
            "not_slower=False ratio_at_S=None numerics=diagnostic outcome=admitted") in (out / "RESULT").read_text()


def test_gate_pass_then_failure_rerun_in_place_reads_not_passed(tmp_path):
    """Finding 6 through the driver: a first run passes its gate canary; a
    RERUN into the same --out fails it. The rerun's admission reads the gate
    as failed — the earlier attempt's whole unit was moved aside (renamed,
    never deleted), and the gate record is bound to the current attempt."""
    out = tmp_path / "rec" / "gemma4e4b_8001"
    rc, out = _run(tmp_path, out, Fakes(native_shift=None, per_row_shift=(1, 2), gate="passed"))
    assert rc == 0
    first = json.loads((out / "unit" / "admission.json").read_text())
    assert first["verdict"] == "admitted" and first["gate_canary"]["passed"] and first["gate_canary"]["bound"]
    rc, out2 = _run(tmp_path, out, Fakes(native_shift=None, per_row_shift=(1, 2), gate="failed"))
    assert rc == 0 and out2 == out
    second = json.loads((out / "unit" / "admission.json").read_text())
    assert second["gate_canary"]["status"] == "failed" and not second["gate_canary"]["passed"]
    assert second["verdict"] == "not_admitted" and "gate_canary_failed" in second["reasons"]
    assert second["gate_canary"]["bound"] and second["gate_canary"]["attempt"]["rc"] == 4
    # the first attempt's unit sits aside, its record intact; the live unit has no record
    stale_units = sorted(out.glob("stale-*/unit"))
    assert len(stale_units) == 1 and (stale_units[0] / "gate_canary" / "gate_record.json").is_file()
    assert (stale_units[0] / "DONE.json").is_file() and (stale_units[0] / "admission.json").is_file()
    assert not (out / "unit" / "gate_canary" / "gate_record.json").exists()
    assert (out / "unit" / "gate_canary" / "gate_failed.json").is_file()
    assert (out / "unit" / "gate_canary" / adm.GATE_ATTEMPT).is_file() and (out / "unit" / "gate_canary" / adm.GATE_RUN).is_file()
    man = json.loads((out / "manifest.json").read_text())
    assert man["prior_unit_moved_to"] == str(stale_units[0].parent)


def test_gate_pass_then_timeout_rerun_in_place_reads_not_passed(tmp_path):
    """Finding 6, the timeout face: the rerun's gate child times out (rc 124,
    nothing written). The earlier success cannot stand in: the summary is a
    bound failure naming the exit code."""
    out = tmp_path / "rec" / "gemma4e4b_8002"
    rc, out = _run(tmp_path, out, Fakes(native_shift=None, per_row_shift=(1, 2), gate="passed"))
    assert rc == 0
    rc, _ = _run(tmp_path, out, Fakes(native_shift=None, per_row_shift=(1, 2), gate="timeout"))
    assert rc == 0
    rec = json.loads((out / "unit" / "admission.json").read_text())
    assert not rec["gate_canary"]["passed"] and rec["gate_canary"]["status"] == "failed"
    assert "rc=124" in rec["gate_canary"]["reason"] and rec["gate_canary"]["attempt"]["rc"] == 124
    assert rec["verdict"] == "not_admitted" and "gate_canary_failed" in rec["reasons"]
    assert not (out / "unit" / "gate_canary" / "gate_record.json").exists()
    # the degenerate-floor face: the gate refuses by name, the recorded numerics name the fallback finding
    rc, out3 = _run(tmp_path, tmp_path / "rec" / "gemma4e4b_8003",
                    Fakes(native_shift=None, per_row_shift=None, gate="timeout"))
    rec3 = json.loads((out3 / "unit" / "admission.json").read_text())
    assert rec3["floor"]["degenerate"] and rec3["reasons"] == ["gate_canary_failed"]
    assert "floor_degenerate_and_gate_canary_failed" in rec3["numerics"]["reasons"]


def test_nemotron_preflights_the_fused_kernels_before_the_oracle(tmp_path):
    """Finding 7 through the driver: on the Mamba-hybrid target the fused-kernel
    bindings are imported in the oracle interpreter BEFORE the oracle child
    runs; a failing preflight is a refusal (rc 2) naming the overlay venv;
    the other targets never run it."""
    fakes = Fakes(native_shift=None, per_row_shift=(1, 2))
    rc, out = _run(tmp_path, tmp_path / "rec" / "nemotron3nano4b_9001", fakes, key="nemotron3nano4b")
    assert rc == 0, (out / "FAILED.json").read_text() if (out / "FAILED.json").exists() else rc
    assert fakes.kernel_preflights == 1
    assert fakes.subprocess_order[:2] == ["kernel_preflight", "oracle"]
    pre = json.loads((out / "prep" / "oracle" / "kernel_preflight.json").read_text())
    assert pre["rc"] == 0 and pre["python"] == "/synthetic/python" and pre["target"] == "nemotron3nano4b"
    assert "import mamba_ssm, causal_conv1d" in pre["check"]
    assert json.loads((out / "manifest.json").read_text())["facts"]["fused_kernel_preflight_rc"] == 0
    # a venv without the kernels: refused before any model loads, the fix named
    fakes = Fakes(native_shift=None, per_row_shift=(1, 2), kernel_rc=1)
    rc, out = _run(tmp_path, tmp_path / "rec" / "nemotron3nano4b_9002", fakes, key="nemotron3nano4b")
    assert rc == mc.RC_REFUSED and fakes.oracle_calls == 0 and fakes.native_attempts == 0
    failed = json.loads((out / "FAILED.json").read_text())
    assert failed["kind"] == "refusal" and "mamba_ssm / causal_conv1d" in failed["error"]
    assert "sisteer-petri-vllm-nemotron-" in failed["error"] and "not weakened" in failed["error"]
    # gemma never runs it
    fakes = Fakes(native_shift=None, per_row_shift=(1, 2), kernel_rc=1)
    rc, _ = _run(tmp_path, tmp_path / "rec" / "gemma4e4b_9003", fakes)
    assert rc == 0 and fakes.kernel_preflights == 0 and fakes.subprocess_order[0] == "oracle"


def test_the_fake_standups_records_load_through_the_strict_readers(tmp_path):
    """Item 11: the fake native standup writes a routing table
    ``load_routing_table`` loads and an attestation ``read_serving_attestation``
    accepts — and the cert READS them that way, so a schema drift in either
    reader fails these tests instead of hiding behind a lenient ``json.loads``."""
    su = FakeStandup("native", tmp_path / "su", hcell=HCELL_GEMMA, sha="0" * 64, S=8)
    table = px.load_routing_table(su.out / "routing_table.json")
    assert table["version"] == 2 and set(table[px.ADAPTERS_KEY]) == {HCELL_GEMMA, "none__lora0"}
    att = sa.read_serving_attestation(su.out / "serving_attestation.json")
    assert set(att["adapters"]) == {HCELL_GEMMA, "none__lora0"}
    assert set(sa.engine_identity(att)["adapters"][HCELL_GEMMA]) == set(px.ENGINE_ADAPTER_KEYS)
    digests = mc._adapter_digests_from(su.endpoint(), su.out)
    assert set(digests) == {HCELL_GEMMA, "none__lora0"} and digests[HCELL_GEMMA]["r"] == 4
    # drift in any of the three records is a ControlsFailed naming it
    ep = su.endpoint()
    ep["adapters"][HCELL_GEMMA] = dict(ep["adapters"][HCELL_GEMMA], weights_sha256="9" * 64)
    with pytest.raises(mc.ControlsFailed, match="endpoint weights_sha256"):
        mc._adapter_digests_from(ep, su.out)
    doc = json.loads((su.out / "serving_attestation.json").read_text())
    doc["engine"]["enforce_eager"] = True
    (su.out / "serving_attestation.json").write_text(json.dumps(doc))
    with pytest.raises(mc.ControlsFailed, match="strict reader"):
        mc._adapter_digests_from(su.endpoint(), su.out)
    (su.out / "serving_attestation.json").write_text(json.dumps({"kind": "lora_native"}))
    with pytest.raises(mc.ControlsFailed, match="strict reader"):
        mc._read_native_attestation(su.out)
    (su.out / "routing_table.json").write_text(json.dumps({"version": 2, "adapters": {}}))
    with pytest.raises(mc.ControlsFailed, match="does not load"):
        mc._load_native_table(su.out)


def test_the_cert_and_the_gate_share_one_proxy_bearer_rule():
    """Item 10: the gate's bootstrap sent no Authorization header when E's
    endpoint carried ``proxy_token: None`` while the cert always sent the
    ProxyClientEngine default to the same proxy. Both now go through
    ``proxy.proxy_client_token`` / ``proxy_client_headers``; the default is
    named ONCE in proxy.py."""
    assert mc._proxy_api_key({"proxy_token": None}) == px.PROXY_CLIENT_BEARER_DEFAULT
    assert mc._proxy_api_key({}) == px.PROXY_CLIENT_BEARER_DEFAULT
    assert mc._proxy_api_key({"proxy_token": "minted-at-standup"}) == "minted-at-standup"
    assert px.proxy_client_headers({"proxy_token": None}) == {
        "Content-Type": "application/json", "Authorization": f"Bearer {px.PROXY_CLIENT_BEARER_DEFAULT}"}
    assert px.proxy_client_headers({"proxy_token": "t0k"})["Authorization"] == "Bearer t0k"
    assert px.proxy_client_headers(None)["Authorization"] == f"Bearer {px.PROXY_CLIENT_BEARER_DEFAULT}"
    import inspect
    assert inspect.signature(px.ProxyClientEngine.__init__).parameters["api_key"].default == \
        px.PROXY_CLIENT_BEARER_DEFAULT
    # the cert carries no bearer literal of its own; the gate reads the same rule
    assert not hasattr(mc, "PROXY_API_KEY_DEFAULT")
    gate_src = (REPO / "scripts" / "petri_serving_gate.py").read_text()
    assert "headers = px.proxy_client_headers(endpoint)" in gate_src


def test_slower_target_reports_not_admitted_with_numerics_admitted(tmp_path):
    fakes = Fakes(native_shift=None, per_row_shift=(1, 2), ratio=0.9)
    rc, out = _run(tmp_path, tmp_path / "rec" / "gemma4e4b_7001", fakes)
    assert rc == 0
    rec = json.loads((out / "unit" / "admission.json").read_text())
    assert rec["numerics"]["outcome"] == "admitted" and rec["verdict"] == "not_admitted" and rec["reasons"] == ["slower_at_S"]
    assert (f"verdict=not_admitted rule={adm.ADMISSION_RULE} readiness=attested controls=6/6 gate=passed "
            "not_slower=False ratio_at_S=0.900 numerics=diagnostic outcome=admitted") in (out / "RESULT").read_text()


def test_lens_leg_requires_the_attested_reduction_mode(tmp_path):
    fakes = Fakes(native_shift=None, per_row_shift=(1, 2), stacked_mode="per_row")
    rc, out = _run(tmp_path, tmp_path / "rec" / "gemma4e4b_8001", fakes)
    assert rc == mc.RC_FAILED
    assert "required 'stacked'" in json.loads((out / "FAILED.json").read_text())["error"]
    assert (out / "unit" / "lens_stacked" / "reduction_mode_check.json").is_file()
    assert all(s.torn_down for s in fakes.standups)


def test_lens_cells_include_the_prompted_baseline_only_when_the_config_lists_it(tmp_path):
    for key, want in (("gemma4e4b", ["constv2ctl2-full__lora_r4_ce"]),
                      ("nemotron3nano4b", ["constv2-mix-full__lora_r4_ce_minus_ctl", "prompted__safety_refusal"]),
                      # the llama config lists the prompted baseline: both rows serve it on the lens legs
                      ("llama8b", ["constv2-mix-full__lora_r4_ce", "prompted__safety_refusal"]),
                      ("llama8b_subtraction", ["constv2-mix-full__lora_r4_ce_minus_ctl", "prompted__safety_refusal"])):
        defenses, pins, sha = _world(tmp_path / key, key)
        fakes = Fakes()
        ctx = mc.build_context(_args(tmp_path / key, tmp_path / key / "out", key=key, defenses=defenses, pins=pins), fakes.deps())
        assert ctx.lens_cells == want, key
        assert ctx.hcell.startswith("si_") and ctx.object_sha256 == sha


def test_the_stage_two_row_resolves_all_three_pins_including_the_qualified_one(tmp_path):
    """native-psr-serving §1.6 (including the T5b cell): ``gemma4e4b_psr2cert`` serves FOUR
    fitted cells and the Arditi one is pinned by ANOTHER authority. ``build_context`` resolves every pin
    through the one rule — the qualified key from the file beside ``--pins`` — and the
    per-cell maps (objects, harness ids, object paths) carry all three, with the PRIMARY
    cell keeping the single-cell semantics (the run identity's harness cell, the sha)."""
    key = "gemma4e4b_psr2cert"
    defenses, pins, sha = _world(tmp_path, key)
    # the qualified pin landed in its OWN file beside the wave's, which is where the driver
    # looks — never in the wave's pins doc
    assert (pins.parent / "misalign_baseline_pins.json").is_file()
    assert "arditi" not in json.dumps(json.loads(pins.read_text()))
    ctx = mc.build_context(_args(tmp_path, tmp_path / "out", key=key, defenses=defenses, pins=pins),
                           Fakes().deps())
    assert ctx.cells == ["safety_refusal__psr_kl_all", "si_constitution__psr_kl_all", "arditi", "constv2-mix-full__lora_r4_ce"]
    assert ctx.hcells == ["si_safety__psr_kl_all", "si_constitution__psr_kl_all", "arditi", "si_constv2_mix_full__lora_r4_ce"]
    assert ctx.hcell == "si_safety__psr_kl_all" and ctx.object_sha256 == sha
    assert set(ctx.objmap_tt) == set(ctx.cells) and set(ctx.objmap_h) == set(ctx.hcells)
    assert len(set(ctx.objmap_tt.values())) == 4          # every cell its own bytes
    mis = json.loads((pins.parent / "misalign_baseline_pins.json").read_text())
    assert ctx.objmap_tt["arditi"] == mis["gemma4e4b"]["arditi"]["object_sha256"]
    assert ctx.object_pt_by_tt["arditi"] == defenses / "arditi" / "object.pt"
    # a pin file the row's directory does not carry refuses by name, before any GPU
    (pins.parent / "misalign_baseline_pins.json").unlink()
    with pytest.raises(mc.CertRefusal, match="no pins file at"):
        mc.build_context(_args(tmp_path, tmp_path / "out2", key=key, defenses=defenses, pins=pins),
                         Fakes().deps())


def test_parse_batches_and_native_kwargs():
    assert mc.parse_batches("8,32,S", 121) == [8, 32, 121]
    assert mc.parse_batches("8,32,S", 74) == [8, 32, 74]
    assert mc.parse_batches("8,32,S", 20) == [8, 20]          # 32 > S dropped
    assert mc.parse_batches("S,S,8", 40) == [8, 40]


def test_probe_body_and_response_parsing():
    body = mc.probe_body("petri__x", [{"role": "user", "content": "hi"}], max_tokens=128)
    assert body["return_token_ids"] is True and body["return_tokens_as_token_ids"] is True
    assert body["logprobs"] is True and body["top_logprobs"] == 5 and body["temperature"] == 0.0
    resp = {"choices": [{"token_ids": [11, 12, 13], "finish_reason": "stop",
                         "logprobs": {"content": [{"token": "token_id:11", "logprob": -0.1,
                                                   "top_logprobs": [{"token": "token_id:11", "logprob": -0.1},
                                                                    {"token": "token_id:4", "logprob": -2.0},
                                                                    {"token": "weird", "logprob": -3.0}]}]}}],
            "prompt_token_ids": [1, 2], "petri": {"engine": "lora_native"}}
    row = mc.parse_probe_response(resp, probe_index=3, vmodel="petri__x")
    assert row["token_ids"] == [11, 12, 13] and row["first_token"]["id"] == 11
    assert row["first_token"]["top5"] == [[11, -0.1], [4, -2.0], [None, -3.0]] and row["stamps"] == {"engine": "lora_native"}
    with pytest.raises(RuntimeError):
        mc.parse_probe_response({"choices": [{"message": {"content": "x"}}]}, probe_index=0, vmodel="v")


def test_result_line_format():
    """The RESULT line (§1.4): verdict, rule, the four gates by name (readiness /
    controls / gate / not_slower + ratio_at_S), then the numerics as a diagnostic
    (``numerics=diagnostic outcome=<three-way>``). A legacy schema-4 admission
    still spells its numerics verdict; no admission at all spells ``none`` in
    every field — exactly the sbatch's fallback line."""
    F = adm.ADMISSION_RULE
    legacy = mc.result_line(state="DONE", rc=0, elapsed=61.9, host="TESTHOST", target="qwen35_4b",
                            admission={"verdict": "not_admitted", "numerics": {"admitted": True},
                                       "controls": {"passed": 5, "of": 5}, "head_to_head": {"not_slower_at_S": False},
                                       "floor": {"source": "gate_canary_neartie_oracle"}}, unit_reused_from=None)
    assert legacy == ("state=DONE exit=0 elapsed=61 host=TESTHOST target=qwen35_4b verdict=not_admitted rule=none "
                      "readiness=none controls=5/5 gate=none not_slower=False ratio_at_S=None numerics=True outcome=none "
                      "floor=gate_canary_neartie_oracle unit_reused_from=none")
    line = mc.result_line(state="DONE", rc=0, elapsed=1.0, host="TESTHOST", target="llama8b_subtraction",
                          admission={"verdict": "admitted", "rule": F, "readiness": {"attested": True},
                                     "numerics": {"gate": False, "outcome": "inconclusive"},
                                     "controls": {"passed": 5, "of": 5}, "gate_canary": {"passed": True, "status": "completed"},
                                     "head_to_head": {"not_slower_at_S": True, "ratio_at_S": 1.25},
                                     "floor": {"source": "per_row_vs_stacked"}}, unit_reused_from=None)
    assert line == (f"state=DONE exit=0 elapsed=1 host=TESTHOST target=llama8b_subtraction verdict=admitted rule={F} "
                    "readiness=attested controls=5/5 gate=passed not_slower=True ratio_at_S=1.250 numerics=diagnostic "
                    "outcome=inconclusive floor=per_row_vs_stacked unit_reused_from=none")
    line = mc.result_line(state="DONE", rc=0, elapsed=1.0, host="TESTHOST", target="gemma4e4b",
                          admission={"verdict": "not_admitted", "rule": F, "readiness": {"attested": False, "problem": "x"},
                                     "numerics": {"gate": False, "outcome": "admitted"},
                                     "controls": {"passed": 4, "of": 5}, "gate_canary": {"passed": False, "status": "failed"},
                                     "head_to_head": {"not_slower_at_S": True, "ratio_at_S": 2.0},
                                     "floor": {"source": "per_row_vs_stacked"}}, unit_reused_from=None)
    assert ("verdict=not_admitted rule=" + F + " readiness=unattested controls=4/5 gate=failed not_slower=True "
            "ratio_at_S=2.000 numerics=diagnostic outcome=admitted") in line
    none = mc.result_line(state="FAILED", rc=2, elapsed=3, host="TESTHOST", target="gemma4e4b", admission=None,
                          unit_reused_from=None)
    assert none == ("state=FAILED exit=2 elapsed=3 host=TESTHOST target=gemma4e4b verdict=none rule=none readiness=none "
                    "controls=0/6 gate=none not_slower=None ratio_at_S=None numerics=None outcome=none floor=none "
                    "unit_reused_from=none")
    # the sbatch's fallback line carries the SAME fields in the same order
    fallback = re.search(r'echo "state=\$1 exit=\$2 elapsed=\$SECONDS host=\$\(hostname\) target=\$KEY ([^"]*)"',
                         SBATCH.read_text()).group(1)
    assert fallback == none.split(" target=gemma4e4b ")[1]


# --------------------------------------------------------------------------- #
# the report                                                                    #
# --------------------------------------------------------------------------- #

def test_report_on_a_synthetic_pair_produces_both_pngs_and_labels_reuse(tmp_path):
    pytest.importorskip("matplotlib")
    records = tmp_path / "records"
    f1 = Fakes(native_shift=None, per_row_shift=(1, 2), ratio=1.6)
    rc, a = _run(tmp_path / "w1", records / "gemma4e4b_9001", f1)
    assert rc == 0
    f2 = Fakes(native_shift=None, per_row_shift=None, ratio=1.1, gate="failed")
    rc, b = _run(tmp_path / "w2", records / "nemotron3nano4b_9002", f2, key="nemotron3nano4b")
    assert rc == 0
    # a third record that REUSED gemma's unit (§5) — drawn under the original job id
    f3 = Fakes(native_shift=None, per_row_shift=(1, 2), ratio=1.6)
    rc, c = _run(tmp_path / "w1", records / "gemma4e4b_9003", f3, resume=a)
    assert rc == 0 and json.loads((c / "manifest.json").read_text())["unit_reused_from"] == str(a / "unit")
    out = rep.build(records, tmp_path / "figs")
    names = {p.name for p in out["figures"]}
    assert names == {"head_to_head_tok_s.png", "admission_disagreement.png"}
    for p in out["figures"]:
        assert p.stat().st_size > 10_000, p
    md = out["table"]
    assert "gemma4e4b" in md and "nemotron3nano4b" in md and "summary_table.md" in os.listdir(tmp_path / "figs")
    recs = {r.path.name: r for r in out["records"]}
    assert set(recs) == {"gemma4e4b_9003", "nemotron3nano4b_9002"}       # newest per target
    assert recs["gemma4e4b_9003"].reused_from == str(a / "unit") and "reused by job 9003" in recs["gemma4e4b_9003"].provenance
    assert "job 9001" in recs["gemma4e4b_9003"].provenance and "measured" in recs["gemma4e4b_9003"].provenance
    assert recs["nemotron3nano4b_9002"].verdict == "not_admitted" and recs["gemma4e4b_9003"].verdict == "admitted"
    assert "gate_canary_neartie_oracle" in md
    everything = rep.load_records(records, all_dirs=True)
    assert [r.path.name for r in everything] == ["gemma4e4b_9001", "gemma4e4b_9003", "nemotron3nano4b_9002"]
    assert all(r.problems == [] for r in everything)


# --------------------------------------------------------------------------- #
# native-serving-default (2026-09-13): Target.model_key, the llama rows, the    #
# lens legs' explicit path, the readiness input, the mixed report               #
# --------------------------------------------------------------------------- #

REAL_RECORD = _real_run_state("multilora-cert/gemma4e4b_490087")


def test_llama_rows_resolve_their_pins_config_and_manifest_from_the_real_repo():
    """§1.4: two rows on ONE model. ``llama8b`` (main view, the lora_all cell)
    and ``llama8b_subtraction`` (the subtraction cell) share config, manifest,
    view and ``model_key == "llama8b"``; their pin keys resolve in the REAL
    pins.json to the spec's digests (504b6651…, 4375b67a…); the config and
    manifest exist and name the pinned model; every other row's model key IS
    its row key (the default); ``--target`` is an alias of ``--model-key``."""
    import yaml
    pins = json.loads((REPO / mc.PINS_REL).read_text())
    a, s = mc.TARGETS["llama8b"], mc.TARGETS["llama8b_subtraction"]
    assert a.model_key == s.model_key == "llama8b" and a.key == "llama8b" and s.key == "llama8b_subtraction"
    assert a.config == s.config == "configs/sdfbase_full22_llama8b.yaml"
    assert a.manifest == s.manifest == "configs/capeval/manifests/petri_full22_llama8b.yaml"
    assert a.view == s.view == "main" and not a.fused_mamba_kernels and not s.fused_mamba_kernels
    assert (a.tt_cell, a.pin_key) == ("constv2-mix-full__lora_r4_ce", "base__constv2-mix-full__lora_r4_ce")
    assert (s.tt_cell, s.pin_key) == ("constv2-mix-full__lora_r4_ce_minus_ctl", "derived__constv2-mix-full__lora_r4_ce_minus_ctl")
    assert pins["llama8b"][a.pin_key]["object_sha256"] == "504b6651facd296608af8e7441b71a51ec5f1062666e4d21f79bf7f91fa8f9b0"
    assert pins["llama8b"][s.pin_key]["object_sha256"] == "4375b67a0fb0621e9ee296041d0842cb24294112e18424a803e8e54e9538a740"
    assert "llama8b_subtraction" not in pins
    for rel in (a.config, a.manifest):
        assert (REPO / rel).is_file(), rel
    cfg = yaml.safe_load((REPO / a.config).read_text())
    assert cfg["model"] == "NousResearch/Meta-Llama-3.1-8B-Instruct" and cfg["model_revision"] == "d10aef7999a2b5ba950ab3974312feeedbfe0b77"
    assert a.tt_cell in cfg["explicit_cells"] and s.tt_cell in cfg["explicit_cells"] and mc.PROMPTED_CELL in cfg["explicit_cells"]
    for key, t in mc.TARGETS.items():
        # a row's MODEL key: llama8b for every llama row; a certification row (native-psr-serving)
        # names its model with the row's `_psr` / `_psr2cert` suffix stripped; every other row's
        # model key IS its row key
        want = ("llama8b" if key.startswith("llama8b")
                else key.removesuffix("_psr2cert").removesuffix("_psr").removesuffix("_paired"))
        assert t.key == key and t.model_key == want
    assert mc.Target("x", "c", "m", "v", "cell", "pin").model_key == "x"
    assert sorted(mc.TARGETS) == ["gemma4e4b", "gemma4e4b_psr", "gemma4e4b_psr2cert", "llama8b",
                                  "llama8b_paired", "llama8b_psr", "llama8b_subtraction", "nemotron3nano4b",
                                  "nemotron3nano4b_psr", "qwen35_4b", "qwen35_4b_paired", "qwen35_4b_psr"]
    args = mc.build_parser().parse_args(["--target", "llama8b_subtraction", "--repo-dir", "/r", "--out", "/o"])
    assert args.model_key == "llama8b_subtraction"


def test_model_key_is_read_at_every_model_site_and_the_row_key_names_the_record(tmp_path, monkeypatch):
    """Every use of the target key that means the MODEL reads
    ``Target.model_key``: the geometry table, the bank view under the shared
    root, the adapter bank, pins.json, the standup's ``model_key``, the run
    identity — while the out dir, the manifest's ``target`` and the RESULT
    keep the ROW key. Driven on the subtraction row (its two keys differ),
    with ``defenses_dir`` and the adapter bank left to their defaults under a
    monkeypatched shared root. A pins file keyed by the ROW key refuses by
    name."""
    import yaml
    key, t = "llama8b_subtraction", mc.TARGETS["llama8b_subtraction"]
    root = tmp_path / "shared"
    view = root / mc.DEFENSES_REL / "llama8b" / "cells_view" / "main"
    (view / t.tt_cell).mkdir(parents=True)
    (view / t.tt_cell / "object.pt").write_bytes(b"llama synthetic object")
    sha = hashlib.sha256(b"llama synthetic object").hexdigest()
    pins = tmp_path / "pins.json"
    pins.write_text(json.dumps({"llama8b": {t.pin_key: {"object_sha256": sha}}}))
    monkeypatch.setattr(mc.lsu, "shared_root", lambda: root)
    # the adapter-bank default is E's seam (its resolution is E's business, native-serving-default §1.9);
    # what is pinned here is the KEY the driver hands it
    bank_calls = []

    def default_adapter_bank(model_key):
        bank_calls.append(model_key)
        return root / "exp" / "multilora_adapters" / model_key
    monkeypatch.setattr(mc.lsu, "default_adapter_bank", default_adapter_bank)
    cfg = yaml.safe_load((REPO / t.config).read_text())
    # D = 4096 (llama's o_proj width): the kernel ceiling LOWERS S, so every geometry read site fires —
    # build_context's strict + lowered resolutions and the two lens legs' re-resolution at the lowered S
    f = Fakes(native_shift=None, per_row_shift=(1, 2), module_dim=4096)
    f.sha, f.model, f.revision, f.hcell = sha, cfg["model"], cfg["model_revision"], mc.harness_cell_id(t.tt_cell)
    f.template = "0a2c8073c878ab1d"
    out = tmp_path / "rec" / "llama8b_subtraction_7001"
    argv = ["--model-key", key, "--repo-dir", str(REPO), "--out", str(out), "--pins", str(pins),
            "--window-s", "1", "--warmup-s", "0", "--repeats", "2"]
    ctx = mc.build_context(mc.build_parser().parse_args(argv), f.deps())
    assert ctx.defenses_dir == view and ctx.bank_dir == root / "exp" / "multilora_adapters" / "llama8b"
    assert bank_calls == ["llama8b"]
    assert ctx.object_sha256 == sha and ctx.target is t and f.geometry_model_keys == ["llama8b", "llama8b"]
    assert ctx.S == 107 and ctx.geometry_lowered["reason"] == "lora_kernel_index_ceiling"
    assert ctx.lens_cells == [t.tt_cell, mc.PROMPTED_CELL]
    rc = mc.run(mc.build_parser().parse_args(argv), f.deps())
    assert rc == 0, (out / "FAILED.json").read_text() if (out / "FAILED.json").is_file() else rc
    assert set(bank_calls) == {"llama8b"}
    assert f.last_native_kw["model_key"] == "llama8b" and f.last_native_kw["defenses_dir"] == str(view)
    assert f.last_native_kw["adapter_bank"] == str(root / "exp" / "multilora_adapters" / "llama8b")
    assert f.last_native_kw["cell"] == [t.tt_cell, mc.PROMPTED_CELL] and f.last_native_kw["max_num_seqs"] == 107
    # 2 (the explicit build_context) + 2 (run's build_context) + 2 (the lens legs at the lowered S): all the MODEL key
    assert f.geometry_model_keys == ["llama8b"] * 6 and [g.max_num_seqs for g in f.lens_geoms] == [107, 107]
    man = json.loads((out / "manifest.json").read_text())
    assert man["target"] == key and man["model_key"] == "llama8b" and man["defenses_dir"] == str(view)
    assert man["adapter_bank"] == str(root / "exp" / "multilora_adapters" / "llama8b")
    ident = json.loads((out / "unit" / "identity.json").read_text())
    assert ident["target"] == key and ident["model_key"] == "llama8b" and ident["harness_cell"] == "si_constv2_mix_full__lora_r4_ce_minus_ctl"
    assert json.loads((out / "unit" / "run_identity.json").read_text())["model_key"] == "llama8b"
    assert "target=llama8b_subtraction verdict=admitted" in (out / "RESULT").read_text()
    r = rep.TargetRecord(out)
    assert r.target == key and r.verdict == "admitted" and r.problems == []
    wrong = tmp_path / "wrong_pins.json"
    wrong.write_text(json.dumps({key: {t.pin_key: {"object_sha256": sha}}}))
    with pytest.raises(mc.CertRefusal, match="no llama8b/derived__constv2-mix-full__lora_r4_ce_minus_ctl"):
        mc.build_context(mc.build_parser().parse_args(["--model-key", key, "--repo-dir", str(REPO), "--out", str(out),
                                                       "--pins", str(wrong)]), f.deps())


def test_the_lens_legs_name_their_serving_path_and_the_production_binding_requires_it(tmp_path, monkeypatch):
    """§1.4: the standup entry points default to the native engine since
    2026-09-13, so the head-to-head's two lens legs pass
    ``serving_path="lens"`` EXPLICITLY — the fake deps record it on both legs
    — and the PRODUCTION binding forwards it to ``capctrl_serving_up.bring_up``
    and REFUSES a call without it, with another path, or on a tree whose
    ``bring_up`` has no ``serving_path`` parameter (never dropping the kwarg
    to make the call go, which would make the comparison native-vs-native)."""
    fakes = Fakes(native_shift=None, per_row_shift=(1, 2))
    rc, out = _run(tmp_path, tmp_path / "rec" / "gemma4e4b_7301", fakes)
    assert rc == 0 and mc.LENS_SERVING_PATH == "lens"
    assert [kw["serving_path"] for kw in fakes.lens_kws] == ["lens", "lens"] and fakes.env_at_lens == ["1", "0"]
    for kw in fakes.lens_kws:                       # the rest of the call is what it always was
        # the digests ride as a LIST (one `TT_CELL=SHA` per fitted cell — a PSR row serves two)
        assert kw["cell"] == ["constv2ctl2-full__lora_r4_ce"] and len(kw["expect_object_sha256"]) == 1 \
            and kw["expect_object_sha256"][0].startswith("constv2ctl2-full__lora_r4_ce=")
    calls = []

    def bring_up_with(*, serving_path="lora_native", **kw):
        calls.append(dict(kw, serving_path=serving_path))
        return "standup"

    def bring_up_without(**kw):
        calls.append(kw)
        return "standup"
    fake_cs = types.SimpleNamespace(bring_up=bring_up_with)
    monkeypatch.setitem(sys.modules, "capctrl_serving_up", fake_cs)
    assert mc._production_capctrl_bring_up(repo="/r", serving_path="lens") == "standup"
    assert calls == [{"repo": "/r", "serving_path": "lens"}]
    with pytest.raises(mc.CertRefusal, match="serving_path='lens', got None"):
        mc._production_capctrl_bring_up(repo="/r")                       # implicit: would follow the native default
    with pytest.raises(mc.CertRefusal, match="native-vs-native"):
        mc._production_capctrl_bring_up(repo="/r", serving_path="lora_native")
    fake_cs.bring_up = bring_up_without
    with pytest.raises(mc.CertRefusal, match="accepts no serving_path"):
        mc._production_capctrl_bring_up(repo="/r", serving_path="lens")
    assert len(calls) == 1                                               # never called past a failed guard


def test_readiness_is_the_strict_read_of_the_attestation(tmp_path):
    """The functional rule's first gate: ``readiness_record`` copies WHAT the
    strict reader accepted (kind, engine bits, adapters, name binding); a
    refused read is ``attested: False`` with the reader's problem verbatim;
    ``readiness_from_unit`` re-reads a unit's attestation through the REAL
    strict reader — a graphs-off file, a foreign file or an absent one is
    unattested BY NAME; a reader that cannot be imported is an ENVIRONMENT
    problem the caller refuses on, never a false 'unattested'; the verdict
    reads ``readiness_unattested``."""
    att = _fake_attestation("si_cell")
    rd = adm.readiness_record(att, path="/p")
    assert rd["attested"] is True and rd["kind"] == "lora_native" and rd["problem"] is None and rd["path"] == "/p"
    assert rd["engine"] == {"enforce_eager": False, "cudagraph_mode": "FULL_AND_PIECEWISE",
                            "cudagraph_specialize_lora": True, "lens_plugin_imported": False}
    assert rd["adapters"] == ["none__lora0", "si_cell"] and rd["name_binding"]["asserted"] is True
    assert rd["source"] == adm.READINESS_SOURCE and "strict" in rd["source"]
    refused = adm.readiness_record(None, problem="ServingAttestationError: kind='lens'")
    assert refused["attested"] is False and refused["problem"] == "ServingAttestationError: kind='lens'" and refused["engine"] is None
    assert adm.readiness_record(None)["problem"] == "no serving attestation was read"
    assert adm.readiness_record("junk")["attested"] is False
    unit = tmp_path / "unit"
    (unit / "lora_native").mkdir(parents=True)
    (unit / adm.UNIT_ATTESTATION_REL).write_text(json.dumps(att))
    rd, env = adm.readiness_from_unit(unit)
    assert env is None and rd["attested"] is True and rd["adapters"] == ["none__lora0", "si_cell"]
    assert rd["path"] == str(unit / "lora_native" / "serving_attestation.json")
    doc = dict(att)
    doc["engine"] = dict(att["engine"], enforce_eager=True)
    (unit / adm.UNIT_ATTESTATION_REL).write_text(json.dumps(doc))
    rd, env = adm.readiness_from_unit(unit)
    assert env is None and rd["attested"] is False and "enforce_eager" in rd["problem"]
    (unit / adm.UNIT_ATTESTATION_REL).write_text(json.dumps({"lens_reduction_mode": "stacked"}))
    rd, env = adm.readiness_from_unit(unit)
    assert rd["attested"] is False and "lacks required keys" in rd["problem"]
    (unit / adm.UNIT_ATTESTATION_REL).unlink()
    rd, env = adm.readiness_from_unit(unit)
    assert rd["attested"] is False and "no serving attestation" in rd["problem"]

    class Boom(Exception):
        pass

    def reader(path):
        raise Boom("synthetic refusal")
    rd, env = adm.readiness_from_unit(unit, reader=(reader, Boom))
    assert rd["attested"] is False and rd["problem"] == "Boom: synthetic refusal" and env is None
    import unittest.mock as um
    with um.patch.object(adm, "_strict_attestation_reader", lambda: (None, None, "no package")):
        rd, env = adm.readiness_from_unit(unit)
    assert env == "no package" and rd["attested"] is False and rd["problem"] == "strict reader unavailable"
    v = adm.admission_verdict(_inputs(readiness=refused))
    assert v["verdict"] == "not_admitted" and v["reasons"] == ["readiness_unattested"] and v["readiness"] == refused
    # an offline re-evaluation without the reader REFUSES rather than filing a false 'unattested'
    fakes = Fakes(native_shift=None, per_row_shift=(1, 2))
    rc, out = _run(tmp_path, tmp_path / "rec" / "gemma4e4b_7401", fakes)
    assert rc == 0
    with um.patch.object(adm, "_strict_attestation_reader", lambda: (None, None, "no package")):
        ev = adm.reevaluate(out)
    assert ev["new"] is None and any("cannot re-evaluate without the strict reader" in q for q in ev["problems"])
    # ... and the job's recorded readiness is held to the re-read: an attestation swapped after the job is named
    (out / "unit" / adm.UNIT_ATTESTATION_REL).write_text(json.dumps(doc))
    ev = adm.reevaluate(out)
    assert ev["new"] is None and any("readiness re-read from the attestation on disk differs" in q for q in ev["problems"])


def test_report_over_a_mixed_directory_renders_the_historical_d111_sidecar_beside_a_functional_record(tmp_path, monkeypatch):
    """On REAL evidence: the pulled gemma record (in-job schema-3
    verdict; a schema-4 sidecar that read NOT ADMITTED; the server logs
    absent) copied beside a synthetic schema-5 llama record. The report
    renders BOTH — the historical one under its readmitted heading with ITS
    verdict, its bound sidecar (not the in-job verdict) and its readiness
    "not an input"; the new one under the functional rule — with no problem
    on either, both in the figures. Then the SAME real record re-evaluated
    under the functional rule IN MEMORY reads ``admitted`` on all four gates,
    its recorded diagnostics byte-equal to what the earlier rule computed
    (the same agreement table, the same bands, the same three-way outcome),
    and NOT ONE BYTE of the record changes. The three existing records are
    not readmitted; this proves the rule on them without rewriting them."""
    pytest.importorskip("matplotlib")
    if REAL_RECORD is None or not (REAL_RECORD / "unit" / adm.READMITTED_ADMISSION_FILE).is_file():
        pytest.skip(f"the pulled record {REAL_RECORD} is not on this machine")
    import shutil
    _clean_tree(monkeypatch)
    F = adm.ADMISSION_RULE
    records = tmp_path / "records"
    records.mkdir()
    hist_dir = records / REAL_RECORD.name
    shutil.copytree(REAL_RECORD, hist_dir)
    rc, new = _run(tmp_path / "w", records / "llama8b_9201", Fakes(native_shift=None, per_row_shift=(1, 2), ratio=1.3),
                   key="llama8b")
    assert rc == 0
    snapshot = lambda: {p.relative_to(hist_dir).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
                        for p in hist_dir.rglob("*") if p.is_file()}
    before = snapshot()
    out = rep.build(records, tmp_path / "figs", all_dirs=True)
    recs = {r.path.name: r for r in out["records"]}
    hist, fresh = recs[REAL_RECORD.name], recs["llama8b_9201"]
    assert hist.readmitted is not None and hist.rule.startswith("D111 readmitted (module ") and hist.rule_name == "D111"
    assert not hist.is_functional and hist.verdict == "not_admitted" and hist.in_job_rule == "D102" and hist.in_job_verdict == "admitted"
    assert hist.readiness_txt == "— (not an input under D111)" and hist.controls_txt == "5/5"
    assert hist.gate_canary_txt.startswith("passed (") and hist.not_slower_txt.startswith("yes (4.20x @ S=107")
    assert hist.zero_verdict == "INEXACT (concurrent_legs) -> not admitted"
    assert hist.problems == [] and hist.absent_diagnostic_logs and hist.complete
    assert fresh.is_functional and fresh.rule == f"{F} in-job" and fresh.verdict == "admitted" and fresh.problems == []
    assert fresh.readiness_txt == "attested" and fresh.controls_txt == "6/6"
    md = out["table"]
    assert "| gemma4e4b | D111 readmitted (module " in md
    assert "| **not_admitted** | — (not an input under D111) | 5/5 | passed (" in md
    assert f"| llama8b | {F} in-job | **admitted** | attested | 6/6 | passed (" in md
    assert [r.path.name for r in out["figure_records"]] == [REAL_RECORD.name, "llama8b_9201"]
    assert {p.name for p in out["figures"]} == {"head_to_head_tok_s.png", "admission_disagreement.png"}
    # the SAME real record, re-evaluated under the functional rule IN MEMORY
    ev = adm.reevaluate(hist_dir)
    assert ev["problems"] == [] and ev["new"] is not None
    v = ev["new"]
    side = json.loads((hist_dir / "unit" / adm.READMITTED_ADMISSION_FILE).read_text())
    assert v["verdict"] == "admitted" and v["rule"] == F and v["reasons"] == [] and v["schema_version"] == 5
    assert all(v["gates"][k] is True for k in adm.FUNCTIONAL_GATES)
    assert v["readiness"]["attested"] is True and v["readiness"]["engine"]["enforce_eager"] is False
    assert v["readiness"]["engine"]["lens_plugin_imported"] is False
    assert v["readiness"]["adapters"] == ["none__lora0", "si_constv2ctl2_full__lora_r4_ce"]
    assert v["zero_adapter_exact"] is True and v["zero_adapter"]["source"] == "noop_certificate"
    assert v["zero_adapter"]["instruments"] == {"concurrent_legs": "failed", "solo_certificate": "passed"}
    assert v["zero_adapter"]["instruments_disagree"] is True
    assert v["zero_adapter"]["concurrent"]["n_diverging"] == side["zero_adapter"]["concurrent"]["n_diverging"] == 8
    assert v["zero_adapter"]["solo_certificate"]["passed"] is True and v["zero_adapter"]["solo_certificate"]["n_prompts"] == 11
    # the diagnostics are the numbers the schema-4 rule computed, exactly (same legs, same
    # seed namespace); only the block's `rule` stamp names the rule that recorded it
    assert v["agreement"] == side["agreement"] and v["first_token_ulps"] == side["first_token_ulps"]
    prs_new, prs_old = v["numerics"]["prompt_resampling_sensitivity"], side["numerics"]["prompt_resampling_sensitivity"]
    assert {k: x for k, x in prs_new.items() if k != "rule"} == {k: x for k, x in prs_old.items() if k != "rule"}
    assert (prs_new["rule"], prs_old["rule"]) == (F, "D111")
    assert v["floor"]["diagnostics"] == side["floor"]["diagnostics"]
    assert {k: x for k, x in v["floor"].items() if k not in ("rule", "margin")} == \
        {k: x for k, x in side["floor"].items() if k not in ("rule", "margin")}
    assert v["numerics"]["outcome"] == side["numerics"]["outcome"] == "not_admitted"
    assert v["numerics"]["reasons"] == side["reasons"] == ["zero_adapter_inexact:concurrent_legs",
                                                           "native_disagreement_exceeds_floor"]
    assert side["verdict"] == "not_admitted" and v["numerics"]["gate"] is False
    assert adm.validate_admission_record(v) == [] and adm.validate_admission_record(side) == []
    assert snapshot() == before                             # nothing written by the report or the re-evaluation
    assert not (hist_dir / "unit" / "admission.readmitted.json.tmp").exists()


# --------------------------------------------------------------------------- #
# the sbatch                                                                    #
# --------------------------------------------------------------------------- #

def test_sbatch_parses_and_keeps_the_hard_rules():
    subprocess.run(["bash", "-n", str(SBATCH)], check=True)
    src = SBATCH.read_text()
    assert "--export=" not in src and "--mem=0" not in src
    for pin in ("#SBATCH --qos=high", "#SBATCH --gres=gpu:1", "#SBATCH --time=02:00:00", "#SBATCH --no-requeue",
                "#SBATCH --signal=B:TERM@120", "#SBATCH --open-mode=append", "#SBATCH --cpus-per-task=16",
                "#SBATCH --mem=128G", "hf_env.sh", "HF_HUB_OFFLINE=1", "mirror-figs.sh", "srun --ntasks=1",
                "cleanup(){ trap '' SIGTERM; kill -TERM -$$ 2>/dev/null; wait; }", "trap on_exit EXIT",
                'sisteer-petri-vllm$TAG-$H', "flock -x 9", "--exclude-newer 2026-08-03T00:00:00Z",
                "vllm.general_plugins", "multilora_cert.py --model-key", "RESULT", "lora_serving_up.py"):
        assert pin in src, pin
    # positional args only, three of them, and the checkout-sha guard
    assert 'KEY="${1:?' in src and 'REPO_DIR="${2:?' in src and 'RESUME="${3:-}"' in src
    assert "rev-parse HEAD" in src and "diff --quiet HEAD" in src
    # the object pin is held on the job side too — under the MODEL key, like the bank view and the venv
    assert "sha256sum \"$OBJ\"" in src and "sdfbase_full22_pins.json" in src
    # ONE pin resolution for the script (native-psr-serving §1.6): bare or qualified, read
    # beside $PINS — the driver's resolve_pin_key in the sbatch's spelling
    assert 'pin_sha(){' in src and 'pin_sha "$PINKEY"' in src and 'pin_sha "${_PINKEYS[$i]}"' in src
    assert '"$PINS" "$MODEL_KEY" "$1"' in src
    assert 'DEFENSES="$SHARED/exp/objects/sdfbase/full22/$MODEL_KEY/cells_view/$VIEW"' in src
    assert 'case "$MODEL_KEY" in\n  nemotron3nano4b) OVERLAY=' in src
    # the two llama rows, ONE model (§1.4): MODEL_KEY=llama8b on both, the pinned snapshot, --time on the submit line
    assert re.search(r"^\s+llama8b\)\s+MODEL_KEY=llama8b;", src, re.M) and re.search(r"^\s+llama8b_subtraction\)\s+MODEL_KEY=llama8b;", src, re.M)
    # llama8b, llama8b_subtraction, llama8b_psr (psrcert view), llama8b_paired (pairedcert view: the paired write's teacher)
    assert src.count("MODEL_KEY=llama8b;") == 4
    assert src.count("models--NousResearch--Meta-Llama-3.1-8B-Instruct/snapshots/d10aef7999a2b5ba950ab3974312feeedbfe0b77") == 4
    assert "llama8b | llama8b_subtraction" in src and "--time=03:00:00" in src
    assert ('KEY="${1:?REQUIRED: gemma4e4b | qwen35_4b | nemotron3nano4b | llama8b | llama8b_subtraction | '
            'gemma4e4b_psr | qwen35_4b_psr | nemotron3nano4b_psr | llama8b_psr | gemma4e4b_psr2cert | '
            'qwen35_4b_paired | llama8b_paired}"') in src           # the two paired-write rows (relu-differenced-native-qwen)


def _sbatch_fragment(start, end_marker):
    src = SBATCH.read_text()
    i = src.index(start)
    j = src.index(end_marker, i) + len(end_marker)
    return src[i:j]


def _sbatch_row(key):
    """Run the sbatch's OWN row-selection block (``case "$KEY"`` through ``esac``)
    under bash for ``key``: ``(rc, "MODEL_KEY|VIEW|CELL|PINKEY|<snapshot>")``."""
    frag = _sbatch_fragment('case "$KEY" in\n  gemma4e4b)', "esac")
    script = (f'KEY={key}; HF_HOME=/hf; log(){{ echo "LOG: $*"; }}; FINAL_RC=""\n{frag}\n'
              'echo "$MODEL_KEY|$VIEW|$CELL|$PINKEY|${SNAP#/hf/hub/}"')
    p = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    return p.returncode, p.stdout.strip()


def _sbatch_venv_name(key):
    """Run the sbatch's OWN row block, then its venv-selection lines (the
    ``case "$MODEL_KEY"`` block through the VENV assignment) under bash for the
    target row ``key`` and print the venv path — the formula the job uses, not a
    re-implementation of it."""
    row = _sbatch_fragment('case "$KEY" in\n  gemma4e4b)', "esac")
    src = SBATCH.read_text()
    start = src.index('case "$MODEL_KEY" in\n  nemotron3nano4b) OVERLAY=')
    end = src.index("VENV=", start)
    end = src.index("\n", end)
    venv = src[start:end]
    script = (f'KEY={key}; HF_HOME=/hf; REPO_DIR={REPO}; REQS="$REPO_DIR/configs/capeval/vllm-requirements.txt"; '
              f'log(){{ :; }}; FINAL_RC=""; USER=u\n{row}\n{venv}\necho "$VENV"')
    return subprocess.run(["bash", "-c", script], check=True, capture_output=True, text=True).stdout.strip()


def test_sbatch_rows_resolve_the_two_llama_rows_to_one_model():
    """The sbatch's ``case "$KEY"`` rows, run under bash: every row's MODEL_KEY /
    VIEW / CELL / PINKEY equal the driver's ``Target`` (one table, two spellings
    pinned equal), the two llama rows share MODEL_KEY=llama8b and the pinned
    snapshot (the config's ``model_revision``), an unknown key exits 2 by name."""
    import yaml
    snaps = {"gemma4e4b": "models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2",
             "qwen35_4b": "models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
             "nemotron3nano4b": "models--nvidia--NVIDIA-Nemotron-3-Nano-4B-BF16/snapshots/dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f",
             "llama8b": "models--NousResearch--Meta-Llama-3.1-8B-Instruct/snapshots/d10aef7999a2b5ba950ab3974312feeedbfe0b77"}
    for key, t in mc.TARGETS.items():
        rc, got = _sbatch_row(key)
        assert rc == 0, (key, got)
        assert tuple(got.split("|")) == (t.model_key, t.view, t.tt_cell, t.pin_key, snaps[t.model_key]), (key, got)
        cfg = yaml.safe_load((REPO / t.config).read_text())
        assert got.endswith(cfg["model_revision"])                     # the sbatch's snapshot IS the config's revision
    assert _sbatch_row("llama8b")[1].split("|")[0] == _sbatch_row("llama8b_subtraction")[1].split("|")[0] == "llama8b"
    rc, got = _sbatch_row("bogus")
    assert rc == 2 and "unknown target key bogus" in got


def test_sbatch_selects_the_overlay_venv_for_nemotron_and_the_base_for_the_rest():
    """The nemotron job runs from the lanes' KERNEL-OVERLAY venv
    (``common.sh f22_serving_venv nemotron``: base requirements, then the
    overlay with ``--no-build-isolation --no-deps`` under the FORCE_BUILD
    flags and the fit env's CUDA recipe, through common.sh's ONE overlay
    command ``f22_overlay_install``), named by the sha256 of base+overlay
    bytes — and the name REPRODUCES the venv already on the login host for
    the current files. gemma / qwen keep the shared base venv,
    byte-identical."""
    src = SBATCH.read_text()
    for pin in ('nemotron3nano4b) OVERLAY="$REPO_DIR/configs/capeval/vllm-requirements-nemotron.txt"; TAG="-nemotron";;',
                '*)               OVERLAY=""; TAG="";;',
                'SERVE_REQS_SHA=$(cat "$REQS" ${OVERLAY:+"$OVERLAY"} | sha256sum | cut -c1-64)',
                'F22_COMMON="$REPO_DIR/scripts/sdfbase/full22/common.sh"', 'source "$F22_COMMON"',
                'f22_overlay_install "$VENV/bin/python" "$OVERLAY" || exit 1',
                'export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"', 'TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0;10.0}"',
                '[ -x "$CUDA_HOME/bin/nvcc" ] ||',
                # the fused-kernel BINDINGS are preflighted in the venv before any model loads
                "import mamba_ssm, causal_conv1d; from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined; "
                "from causal_conv1d import causal_conv1d_fn"):
        assert pin in src, pin
    # no second spelling of the overlay install: the job's only kernel install is the shared one
    assert "--no-build-isolation" not in src
    base_h = hashlib.sha256(BASE_REQS.read_bytes()).hexdigest()[:8]
    overlay_h = hashlib.sha256(BASE_REQS.read_bytes() + NEMOTRON_OVERLAY.read_bytes()).hexdigest()[:8]
    assert base_h != overlay_h
    assert _sbatch_venv_name("gemma4e4b") == f"/home/u/venvs/sisteer-petri-vllm-{base_h}"
    assert _sbatch_venv_name("qwen35_4b") == f"/home/u/venvs/sisteer-petri-vllm-{base_h}"
    assert _sbatch_venv_name("nemotron3nano4b") == f"/home/u/venvs/sisteer-petri-vllm-nemotron-{overlay_h}"
    # both llama rows: the base venv (no kernel overlay), selected by the MODEL key
    assert _sbatch_venv_name("llama8b") == _sbatch_venv_name("llama8b_subtraction") == f"/home/u/venvs/sisteer-petri-vllm-{base_h}"
    assert not mc.TARGETS["llama8b"].fused_mamba_kernels and not mc.TARGETS["llama8b_subtraction"].fused_mamba_kernels
    # the formula reproduces the venv already built on the cluster from these
    # files (a changed requirements file changes the name AND this pin, together)
    assert overlay_h == NEMOTRON_VENV_HASH8, (
        f"base+overlay now hash to {overlay_h}: the nemotron venv name changed with the requirement files; "
        f"update NEMOTRON_VENV_HASH8 (was {NEMOTRON_VENV_HASH8}) deliberately")
    # the driver's own preflight is the same import line, and the target says so
    assert mc.TARGETS["nemotron3nano4b"].fused_mamba_kernels is True
    assert not mc.TARGETS["gemma4e4b"].fused_mamba_kernels and not mc.TARGETS["qwen35_4b"].fused_mamba_kernels
    assert "import mamba_ssm, causal_conv1d; from mamba_ssm.ops.triton.ssd_combined import " \
           "mamba_chunk_scan_combined; from causal_conv1d import causal_conv1d_fn" in mc.FUSED_KERNEL_PREFLIGHT


# --------------------------------------------------------------------------- #
# attempt isolation and throughput evidence                                   #
# --------------------------------------------------------------------------- #

def _fakes_for(tmp_path, key, defenses, pins, sha, **kw):
    """A ``Fakes`` bound to a world ``_world`` built (the same binding ``_run``
    does), for tests that run the driver TWICE into one out dir."""
    import yaml
    cfg = yaml.safe_load((REPO / mc.TARGETS[key].config).read_text())
    f = Fakes(**kw)
    f.sha, f.model, f.revision = sha, cfg["model"], cfg["model_revision"]
    f.hcell = mc.harness_cell_id(mc.TARGETS[key].tt_cell)
    f.template = "0a2c8073c878ab1d"
    return f


FAILED_RESULT_RE = (r"state=FAILED exit=2 elapsed=\d+ host=TESTHOST target=gemma4e4b verdict=none rule=none readiness=none "
                    r"controls=0/6 gate=none not_slower=None ratio_at_S=None numerics=None outcome=none floor=none "
                    r"unit_reused_from=none")


def _assert_earlier_attempt_quarantined(out, first_attempt_id, first_admission):
    """The rerun's out dir: NO unit, NO earlier admission in any reader's
    path; the earlier attempt's records intact under one ``stale-*`` dir; the
    rerun's ATTEMPT.json / manifest / RESULT name the NEW attempt only."""
    assert not (out / "unit").exists()
    stale = sorted(p for p in out.glob("stale-*") if p.is_dir())
    assert len(stale) == 1, stale
    for name in ("unit/admission.json", "unit/DONE.json", "unit/identity.json", "RESULT", "manifest.json",
                 mc.ATTEMPT_FILE, "prep/oracle/DONE.json", "prep/preflight/DONE.json"):
        assert (stale[0] / name).is_file(), name
    assert json.loads((stale[0] / "unit" / "admission.json").read_text()) == first_admission
    assert "verdict=admitted" in (stale[0] / "RESULT").read_text()
    att2 = json.loads((out / mc.ATTEMPT_FILE).read_text())
    assert re.fullmatch(r"[0-9a-f]{32}", att2["attempt_id"]) and att2["attempt_id"] != first_attempt_id
    assert att2["prior_attempt_moved_to"] == str(stale[0])
    assert set(att2["prior_records_moved"]) >= {"unit", "manifest.json", "RESULT", "prep", mc.ATTEMPT_FILE}
    man = json.loads((out / "manifest.json").read_text())
    assert man["attempt_id"] == att2["attempt_id"] and man["prior_unit_moved_to"] == str(stale[0])
    assert man["prior_attempt_moved_to"] == str(stale[0]) and "unit" not in man["phases"]
    assert man["unit_reused_from"] is None
    line = (out / "RESULT").read_text().strip()
    assert re.fullmatch(FAILED_RESULT_RE, line), line
    # the report on this dir presents NO verdict off the earlier unit
    r = rep.TargetRecord(out)
    assert r.verdict == "incomplete" and r.admission is None and not r.done
    assert "no admission.json" in r.problems and r.attempt_id == att2["attempt_id"]
    return stale[0]


def test_success_then_preflight_failure_rerun_publishes_no_earlier_admission(tmp_path):
    """A successful out dir reused by a rerun whose PREFLIGHT refuses
    (object.pt no longer hashes to its pin). The earlier unit's admission.json
    / DONE.json used to stay in place and the FAILED attempt's RESULT
    (``state=FAILED exit=2 ... verdict=admitted``) and the report both read
    them as the rerun's. Now every earlier record is quarantined at attempt
    start, BEFORE the preflight can fail."""
    key, t = "gemma4e4b", mc.TARGETS["gemma4e4b"]
    defenses, pins, sha = _world(tmp_path, key)
    out = tmp_path / "rec" / "gemma4e4b_8101"
    f1 = _fakes_for(tmp_path, key, defenses, pins, sha, native_shift=None, per_row_shift=(1, 2))
    assert mc.run(_args(tmp_path, out, key=key, defenses=defenses, pins=pins), f1.deps()) == 0
    first = json.loads((out / "unit" / "admission.json").read_text())
    first_attempt = json.loads((out / mc.ATTEMPT_FILE).read_text())["attempt_id"]
    assert first["verdict"] == "admitted" and first["attempt_id"] == first_attempt
    done = json.loads((out / "unit" / "DONE.json").read_text())
    assert done["attempt_id"] == first_attempt == json.loads((out / "manifest.json").read_text())["attempt_id"]
    assert done[adm.DONE_ADMISSION_SHA_KEY] == hashlib.sha256((out / "unit" / "admission.json").read_bytes()).hexdigest()
    assert json.loads((out / "unit" / "identity.json").read_text())["attempt_id"] == first_attempt
    assert adm.unit_binding_problems(out / "unit", expected_attempt_id=first_attempt) == []
    assert rep.TargetRecord(out).verdict == "admitted"
    # the object is replaced on disk, the pin stands: the rerun's preflight refuses
    (defenses / t.tt_cell / "object.pt").write_bytes(b"a swapped object")
    f2 = _fakes_for(tmp_path, key, defenses, pins, sha, native_shift=None, per_row_shift=(1, 2))
    rc = mc.run(_args(tmp_path, out, key=key, defenses=defenses, pins=pins), f2.deps())
    assert rc == mc.RC_REFUSED and f2.oracle_calls == 0 and f2.native_attempts == 0
    failed = json.loads((out / "FAILED.json").read_text())
    assert failed["kind"] == "refusal" and "sha256" in failed["error"] and "preflight" in failed["error"]
    stale = _assert_earlier_attempt_quarantined(out, first_attempt, first)
    assert (out / "prep" / "preflight" / "FAILED.json").is_file()
    assert not (out / "prep" / "oracle").exists()          # the rerun never reached the oracle
    assert (stale / "prep" / "oracle" / "reference.json").is_file()


def test_success_then_oracle_failure_rerun_publishes_no_earlier_admission(tmp_path):
    """F1, the oracle face: the rerun passes preflight and its fp32 oracle
    child dies (no reference written) — a CertRefusal after the preflight,
    still BEFORE the unit. Same contract: the FAILED RESULT carries no
    verdict, the earlier unit sits aside, the report reads incomplete."""
    key = "gemma4e4b"
    defenses, pins, sha = _world(tmp_path, key)
    out = tmp_path / "rec" / "gemma4e4b_8102"
    f1 = _fakes_for(tmp_path, key, defenses, pins, sha, native_shift=None, per_row_shift=(1, 2))
    assert mc.run(_args(tmp_path, out, key=key, defenses=defenses, pins=pins), f1.deps()) == 0
    first = json.loads((out / "unit" / "admission.json").read_text())
    first_attempt = first["attempt_id"]
    f2 = _fakes_for(tmp_path, key, defenses, pins, sha, native_shift=None, per_row_shift=(1, 2), oracle_rc=1)
    rc = mc.run(_args(tmp_path, out, key=key, defenses=defenses, pins=pins), f2.deps())
    assert rc == mc.RC_REFUSED and f2.oracle_calls == 1 and f2.native_attempts == 0 and f2.gate_calls == 0
    failed = json.loads((out / "FAILED.json").read_text())
    assert failed["kind"] == "refusal" and "oracle child rc=1" in failed["error"]
    _assert_earlier_attempt_quarantined(out, first_attempt, first)
    assert (out / "prep" / "preflight" / "DONE.json").is_file()      # this attempt's own preflight
    assert not (out / "prep" / "oracle" / "DONE.json").exists()      # and NO oracle completion
    # a third attempt succeeds again: its records are its own, both earlier attempts sit aside
    f3 = _fakes_for(tmp_path, key, defenses, pins, sha, native_shift=None, per_row_shift=(1, 2))
    assert mc.run(_args(tmp_path, out, key=key, defenses=defenses, pins=pins), f3.deps()) == 0
    third = json.loads((out / "unit" / "admission.json").read_text())
    assert third["verdict"] == "admitted" and third["attempt_id"] not in (first_attempt, None)
    assert third["attempt_id"] == json.loads((out / mc.ATTEMPT_FILE).read_text())["attempt_id"]
    assert len([p for p in out.glob("stale-*") if p.is_dir()]) == 2
    assert rep.TargetRecord(out).verdict == "admitted" and rep.TargetRecord(out).problems == []
    assert "state=DONE exit=0" in (out / "RESULT").read_text()


def test_run_unit_refuses_a_populated_unit_dir_inside_an_attempt(tmp_path, monkeypatch):
    """The quarantine lives in ``begin_attempt`` now; ``run_unit`` no longer
    moves a unit aside itself — a populated unit dir at unit start is a
    contract breach it refuses by name, never something it tidies away.
    (The oracle gate, which ``run_unit`` checks FIRST, is stubbed here.)"""
    key = "gemma4e4b"
    defenses, pins, sha = _world(tmp_path, key)
    f = _fakes_for(tmp_path, key, defenses, pins, sha)
    out = tmp_path / "rec" / "gemma4e4b_8103"
    ctx = mc.build_context(_args(tmp_path, out, key=key, defenses=defenses, pins=pins), f.deps())
    assert re.fullmatch(r"[0-9a-f]{32}", ctx.attempt_id)
    monkeypatch.setattr(mc, "require_oracle_done", lambda ctx: None)
    (ctx.unit / "lora_native").mkdir(parents=True)
    (ctx.unit / "lora_native" / "endpoint.json").write_text("{}")
    with pytest.raises(RuntimeError, match="already populated inside this attempt"):
        mc.run_unit(ctx)
    assert f.native_attempts == 0


def test_unit_binding_problems_names_every_mismatch(tmp_path):
    """``adm.unit_binding_problems`` — the ONE rule the driver's RESULT, the
    whole-unit reuse and the report read a unit through (F1)."""
    unit = tmp_path / "unit"
    unit.mkdir()
    aid = "ab" * 16

    def write(identity=aid, done=aid, admission=aid, sha=True):
        (unit / "identity.json").write_text(json.dumps({"attempt_id": identity, "complete": True}))
        (unit / "admission.json").write_text(json.dumps({"attempt_id": admission, "verdict": "admitted"}))
        digest = hashlib.sha256((unit / "admission.json").read_bytes()).hexdigest()
        (unit / "DONE.json").write_text(json.dumps({"attempt_id": done,
                                                    adm.DONE_ADMISSION_SHA_KEY: (digest if sha else "0" * 64)}))
    write()
    assert adm.unit_binding_problems(unit) == [] and adm.unit_binding_problems(unit, expected_attempt_id=aid) == []
    probs = adm.unit_binding_problems(unit, expected_attempt_id="cd" * 16)
    assert len(probs) == 1 and "!= the expected attempt" in probs[0]
    write(done="cd" * 16)
    assert any("attempt ids disagree" in p for p in adm.unit_binding_problems(unit))
    write(sha=False)
    assert any(adm.DONE_ADMISSION_SHA_KEY in p for p in adm.unit_binding_problems(unit))
    write(admission=None)
    assert any("admission.json carries no attempt_id" in p for p in adm.unit_binding_problems(unit))
    (unit / "DONE.json").unlink()
    assert adm.unit_binding_problems(unit) == [f"no DONE.json under {unit}"]
    (unit / "DONE.json").write_text("{not json")
    assert any("DONE.json unreadable" in p for p in adm.unit_binding_problems(unit))


def test_report_and_reuse_refuse_a_unit_not_bound_to_the_records_attempt(tmp_path):
    """F1's reader side: the report binds manifest, DONE.json and
    admission.json to ONE attempt and presents a mismatch as ``unbound``
    (never the admission's verdict); the driver's whole-unit reuse refuses a
    prior unit whose records do not bind."""
    import shutil
    records = tmp_path / "records"
    f1 = Fakes(native_shift=None, per_row_shift=(1, 2))
    rc, good = _run(tmp_path / "w", records / "gemma4e4b_8201", f1)
    assert rc == 0
    r = rep.TargetRecord(good)
    assert r.verdict == "admitted" and r.binding_problems == [] and r.problems == [] and r.admission_unbound is None

    def variant(name, mutate):
        d = records / name
        shutil.copytree(good, d)
        mutate(d)
        return rep.TargetRecord(d)

    def other_attempts_admission(d):
        a = json.loads((d / "unit" / "admission.json").read_text())
        a["attempt_id"] = "ee" * 16
        (d / "unit" / "admission.json").write_text(json.dumps(a))

    def other_attempts_manifest(d):
        m = json.loads((d / "manifest.json").read_text())
        m["attempt_id"] = "ee" * 16
        (d / "manifest.json").write_text(json.dumps(m))

    def manifest_from_before_the_binding(d):
        m = json.loads((d / "manifest.json").read_text())
        del m["attempt_id"]
        (d / "manifest.json").write_text(json.dumps(m))

    def reuse_claimed_over_this_records_own_unit(d):
        m = json.loads((d / "manifest.json").read_text())
        m["unit_reused_from"] = str(d / "unit")
        (d / "manifest.json").write_text(json.dumps(m))

    # (a) an admission from another attempt beside this attempt's DONE.json
    ra = variant("gemma4e4b_8202", other_attempts_admission)
    assert ra.verdict == "unbound" and ra.admission is None and ra.admission_unbound["verdict"] == "admitted"
    assert any("attempt ids disagree" in p for p in ra.binding_problems)
    assert any(adm.DONE_ADMISSION_SHA_KEY in p for p in ra.binding_problems)
    assert ra.problems[0].startswith("unbound:") and "NOT this record's attempt's" in " ".join(ra.problems)
    assert "UNBOUND" in ra.provenance
    # (b) a manifest naming another attempt than the unit's
    rb = variant("gemma4e4b_8203", other_attempts_manifest)
    assert rb.verdict == "unbound" and any("!= the expected attempt" in p for p in rb.binding_problems)
    # (c) a record from before the binding (no attempt_id in the manifest)
    rcd = variant("gemma4e4b_8204", manifest_from_before_the_binding)
    assert rcd.verdict == "unbound" and any("carries no attempt_id" in p for p in rcd.binding_problems)
    # (d) a manifest claiming a reuse the record never made: no unit_reuse.json, no copy
    rd = variant("gemma4e4b_8205", reuse_claimed_over_this_records_own_unit)
    assert rd.verdict == "unbound" and rd.admission is None
    assert any("the reuse was never recorded" in p for p in rd.binding_problems)
    # the table presents `unbound`, never `admitted`, for those records
    out = rep.build(records, tmp_path / "figs", all_dirs=True)
    md = out["table"]
    by_name = {r.path.name: r for r in out["records"]}
    assert by_name["gemma4e4b_8201"].verdict == "admitted"
    assert {by_name[n].verdict for n in ("gemma4e4b_8202", "gemma4e4b_8203", "gemma4e4b_8204", "gemma4e4b_8205")} == {"unbound"}
    assert md.count("**unbound**") == 4 and md.count("**admitted**") == 1
    # the driver's whole-unit reuse refuses the unbound prior (and reuses the bound one)
    f2 = Fakes(native_shift=None, per_row_shift=(1, 2))
    rc, second = _run(tmp_path / "w", records / "gemma4e4b_8206", f2, resume=records / "gemma4e4b_8202")
    assert rc == 0 and f2.native_attempts == 1
    reasons = json.loads((second / "unit_reuse.json").read_text())["reasons"]
    # refused by the inventory digest (A2) before the three-record binding; either names admission.json
    assert any(r.startswith(("prior unit inventory:", "prior unit binding:")) and "admission.json" in r
               for r in reasons), reasons
    f3 = Fakes(native_shift=None, per_row_shift=(1, 2))
    rc, third = _run(tmp_path / "w", records / "gemma4e4b_8207", f3, resume=good)
    assert rc == 0 and f3.native_attempts == 0
    assert json.loads((third / "manifest.json").read_text())["unit_reused_from"] == str(good / "unit")
    assert rep.TargetRecord(third).verdict == "admitted" and rep.TargetRecord(third).problems == []


def test_a_window_without_engine_evidence_is_invalid_by_name():
    """The concurrency guard ran only when ``running_mean`` was present, so a
    window with NO gauge sample (the driver's client-count fallback after a
    failed first ``/metrics`` read) passed with ``invalid reasons: []``.
    Missing or non-finite ``running_mean``, a missing ``server_tokens`` and a
    non-``server`` ``throughput_source`` each invalidate the window by name;
    the client count is a diagnostic."""
    good = _points("lens_stacked", {121: 800.0})[0]
    assert adm.point_invalid_reasons(good) == []
    for over, why in ((dict(running_mean=None), "running_mean unrecorded (None)"),
                      (dict(running_mean=float("nan")), "running_mean unreadable"),
                      (dict(running_mean=float("inf")), "running_mean unreadable"),
                      (dict(running_mean="121"), "running_mean unreadable"),
                      (dict(server_tokens=None), "server_tokens unrecorded (None)"),
                      (dict(server_tokens=-5), "server_tokens unreadable"),
                      (dict(throughput_source=adm.THROUGHPUT_SOURCE_CLIENT_FALLBACK),
                       "throughput_source='client_fallback'"),
                      (dict(throughput_source=None), "throughput_source=None"),
                      (dict(B=None), "B unrecorded")):
        reasons = adm.point_invalid_reasons(dict(good, **over))
        assert any(why in r for r in reasons), (over, reasons)
    # the reproduced production shape: client count present, no engine evidence at all
    fallback = dict(good, tok_s=None, running_mean=None, server_tokens=None,
                    throughput_source=adm.THROUGHPUT_SOURCE_CLIENT_FALLBACK, client_tok_s=812.0)
    reasons = adm.point_invalid_reasons(fallback)
    assert [r.split(" ")[0].split("=")[0] for r in reasons] == ["running_mean", "server_tokens", "throughput_source", "tok_s"]
    h = adm.head_to_head({"lens_stacked": _points("lens_stacked", {121: 800.0}),
                          "lora_native": [dict(fallback, leg="lora_native", rep=0),
                                          dict(fallback, leg="lora_native", rep=1)]}, headline_B=121)
    assert h["ratio_at_S"] is None and not h["not_slower_at_S"]
    assert set(h["headline_invalid"]) == {"lora_native"} and h["per_B"]["121"]["n_valid"]["lora_native"] == 0
    assert any("running_mean unrecorded" in r for r in h["headline_invalid"]["lora_native"])
    assert h["validity"]["throughput_source_required"] == adm.THROUGHPUT_SOURCE_SERVER == "server"
    rec = adm.admission_verdict(_inputs(h2h=h))
    assert rec["verdict"] == "not_admitted" and rec["numerics"]["outcome"] == "admitted"
    assert rec["reasons"] == ["head_to_head_invalid:lora_native"]


def test_driver_metrics_failure_on_the_native_leg_is_head_to_head_invalid(tmp_path):
    """F2 through the driver: the native leg's engine ``/metrics`` was
    unreadable (every window in the client-count fallback shape, a 2.0x ratio
    on client counts) — ``headline_invalid`` names the leg, the verdict is
    ``not_admitted`` with ``head_to_head_invalid:lora_native``, and the leg
    file keeps the client numbers as a diagnostic."""
    fakes = Fakes(native_shift=None, per_row_shift=(1, 2), ratio=2.0, metrics_unreadable={"native"})
    rc, out = _run(tmp_path, tmp_path / "rec" / "gemma4e4b_7201", fakes)
    assert rc == 0
    leg = json.loads((out / "unit" / "throughput" / "lora_native.json").read_text())
    assert leg["throughput_source"] == adm.THROUGHPUT_SOURCE_CLIENT_FALLBACK
    assert all(p["tok_s"] is None and p["client_tok_s"] > 0 and p["running_mean"] is None for p in leg["points"])
    h2h = json.loads((out / "unit" / "head_to_head.json").read_text())
    assert h2h["ratio_at_S"] is None and not h2h["not_slower_at_S"] and set(h2h["headline_invalid"]) == {"lora_native"}
    assert h2h["per_B"]["121"]["lora_native_tok_s"] is None and h2h["per_B"]["121"]["n_valid"]["lora_native"] == 0
    assert h2h["per_B"]["121"]["n_valid"]["lens_stacked"] == 2
    rec = json.loads((out / "unit" / "admission.json").read_text())
    assert rec["numerics"]["outcome"] == "admitted" and rec["verdict"] == "not_admitted"
    assert rec["reasons"] == ["head_to_head_invalid:lora_native"]
    assert (f"verdict=not_admitted rule={adm.ADMISSION_RULE} readiness=attested controls=6/6 gate=passed "
            "not_slower=False ratio_at_S=None numerics=diagnostic outcome=admitted") in (out / "RESULT").read_text()


def _fake_proxy_for_windows(*, metrics_ok):
    """E's proxy + the engine's ``/metrics`` on ONE local port: ``POST
    /v1/chat/completions`` answers a fixed 8-token completion (so the
    client-side count is always positive); ``GET /metrics`` answers a
    vLLM-shaped body (2 running, the generation counter advancing 8 per
    completion) when ``metrics_ok``, 404 otherwise — the production failure
    F2 reproduces (the first read fails and the leg fell back to client
    counts)."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    state = {"gen": 0, "posts": 0, "metrics_gets": 0}
    lock = threading.Lock()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body: bytes, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            with lock:
                state["metrics_gets"] += 1
                gen = state["gen"]
            if self.path == "/metrics" and metrics_ok:
                body = ('vllm:num_requests_running{model_name="m"} 2.0\n'
                        'vllm:num_requests_waiting{model_name="m"} 0.0\n'
                        f'vllm:generation_tokens_total{{model_name="m"}} {float(gen)}\n').encode()
                self._send(200, body, "text/plain")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            with lock:
                state["posts"] += 1
                state["gen"] += 8
            body = json.dumps({"id": "x", "object": "chat.completion",
                               "choices": [{"index": 0, "finish_reason": "length",
                                            "message": {"role": "assistant", "content": "eight tokens"}}],
                               "usage": {"completion_tokens": 8, "prompt_tokens": 3, "total_tokens": 11}}).encode()
            self._send(200, body)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, state


def test_production_throughput_leg_records_a_metrics_failure_as_a_diagnostic_never_as_evidence(tmp_path):
    """F2 on the REAL ``throughput_leg`` / ``closed_loop_via_proxy`` against a
    local stand-in for the proxy: with the engine's ``/metrics`` unreadable
    every window carries ``tok_s None``, ``running_mean None``,
    ``server_tokens None`` and ``throughput_source == "client_fallback"``
    (the client count kept under ``client_tok_s``) and is INVALID for the
    head-to-head — the reproduced ``ratio_at_S: 2.0, not_slower_at_S: True``
    cannot recur. With ``/metrics`` readable the same path records the
    engine's evidence and the window is valid."""
    pytest.importorskip("aiohttp")
    srv, state = _fake_proxy_for_windows(metrics_ok=False)
    try:
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        rec = mc.throughput_leg(f"{base}/v1", "synthetic-bearer", "petri__x", metrics_url=f"{base}/metrics",
                                metrics_headers={"Authorization": "Bearer engine"}, batches=[2], repeats=1,
                                max_tokens=8, warmup_s=0.0, window_s=0.6, out_dir=tmp_path / "raw",
                                leg="lora_native")
    finally:
        srv.shutdown()
    assert state["posts"] > 2 and state["metrics_gets"] >= 3      # probe + m0 + m1: sampled, not disabled
    assert "unreadable at the probe (HTTP 404)" in rec["metrics_probe"]
    assert rec["throughput_source"] == adm.THROUGHPUT_SOURCE_CLIENT_FALLBACK and rec["n_client_fallback_windows"] == 1
    assert "DIAGNOSTIC" in rec["metrics_source"] and "invalid" in rec["metrics_source"]
    (pt,) = rec["points"]
    assert pt["tok_s"] is None and pt["ms_per_step"] is None
    assert pt["throughput_source"] == adm.THROUGHPUT_SOURCE_CLIENT_FALLBACK
    assert pt["running_mean"] is None and pt["server_tokens"] is None and pt["server_tok_s"] is None
    assert pt["client_tok_s"] > 0 and pt["client_tokens"] == 8 * pt["requests_done"] and pt["requests_done"] >= 1
    assert pt["errors"] == 0 and "tok_s_source" not in pt
    reasons = adm.point_invalid_reasons(pt)
    assert any("running_mean unrecorded" in r for r in reasons)
    assert any("server_tokens unrecorded" in r for r in reasons)
    assert any("throughput_source='client_fallback'" in r for r in reasons)
    lens = _points("lens_stacked", {2: 100.0})
    h = adm.head_to_head({"lens_stacked": lens, "lora_native": rec["points"]}, headline_B=2)
    assert h["ratio_at_S"] is None and not h["not_slower_at_S"] and set(h["headline_invalid"]) == {"lora_native"}
    verdict = adm.admission_verdict(_inputs(h2h=h))
    assert verdict["verdict"] == "not_admitted" and "head_to_head_invalid:lora_native" in verdict["reasons"]
    # the readable face: the engine's evidence is recorded and the window counts
    srv2, state2 = _fake_proxy_for_windows(metrics_ok=True)
    try:
        base = f"http://127.0.0.1:{srv2.server_address[1]}"
        rec2 = mc.throughput_leg(f"{base}/v1", "synthetic-bearer", "petri__x", metrics_url=f"{base}/metrics",
                                 metrics_headers={}, batches=[2], repeats=1, max_tokens=8, warmup_s=0.0,
                                 window_s=2.6, out_dir=tmp_path / "raw2", leg="lora_native")
    finally:
        srv2.shutdown()
    (pt2,) = rec2["points"]
    assert rec2["throughput_source"] == adm.THROUGHPUT_SOURCE_SERVER and rec2["n_client_fallback_windows"] == 0
    assert pt2["throughput_source"] == adm.THROUGHPUT_SOURCE_SERVER and pt2["server_tokens"] > 0
    assert pt2["running_mean"] == 2.0 and pt2["tok_s"] == pt2["server_tok_s"] > 0 and pt2["client_tok_s"] > 0
    assert adm.point_invalid_reasons(pt2) == []
    h2 = adm.head_to_head({"lens_stacked": lens, "lora_native": rec2["points"]}, headline_B=2)
    assert h2["headline_invalid"] == {} and h2["ratio_at_S"] is not None


def test_reused_unit_is_self_contained_and_never_follows_the_source_path(tmp_path):
    """A reproduced regression: whole-unit reuse recorded only the mutable
    SOURCE DIRECTORY and the report resolved a reused record by that path with
    no expected attempt. A completes ``not_admitted``; B resumes from A and
    reuses it (``not_admitted``); A is rerun in its own out dir (begin_attempt
    archives A's unit, the new attempt writes an ``admitted`` unit at the same
    path, here with a different probe cap too); the report on B then followed
    the path and read ``admitted`` with no binding problem while B's RESULT
    said ``not_admitted``. Now B holds a verbatim copy under ``reused_unit/``
    bound by ``unit_reuse.json`` (source attempt_id + per-file sha256), and
    every reader resolves the reuse THERE."""
    import shutil
    records = tmp_path / "records"
    # A: slower at S -> not_admitted
    rc, a = _run(tmp_path / "w", records / "gemma4e4b_8401", Fakes(native_shift=None, per_row_shift=(1, 2), ratio=0.9))
    assert rc == 0
    a_first = json.loads((a / "unit" / "admission.json").read_text())
    a_first_attempt = a_first["attempt_id"]
    assert a_first["verdict"] == "not_admitted" and "slower_at_S" in a_first["reasons"]
    a_first_bytes = (a / "unit" / "admission.json").read_bytes()
    # B: resumes from A, reuses A's unit whole
    fb = Fakes(native_shift=None, per_row_shift=(1, 2), ratio=0.9)
    rc, b = _run(tmp_path / "w", records / "gemma4e4b_8402", fb, resume=a)
    assert rc == 0 and fb.native_attempts == 0
    assert "verdict=not_admitted" in (b / "RESULT").read_text() and f"unit_reused_from={a / 'unit'}" in (b / "RESULT").read_text()
    rb = rep.TargetRecord(b)
    assert rb.verdict == "not_admitted" and rb.binding_problems == [] and rb.problems == []
    assert rb.unit_dir == b / adm.REUSED_UNIT_DIR and (b / adm.REUSED_UNIT_DIR / "admission.json").read_bytes() == a_first_bytes
    assert "job 8401" in rb.provenance and "reused by job 8402" in rb.provenance
    b_attempt = json.loads((b / "manifest.json").read_text())["attempt_id"]
    # A is RERUN in place with different settings: faster (admitted) and another probe cap (identity differs)
    rc, a2 = _run(tmp_path / "w", records / "gemma4e4b_8401", Fakes(native_shift=None, per_row_shift=(1, 2), ratio=1.4),
                  extra=("--probe-max-tokens", "64"))
    assert rc == 0 and a2 == a
    a_new = json.loads((a / "unit" / "admission.json").read_text())
    assert a_new["verdict"] == "admitted" and a_new["attempt_id"] != a_first_attempt and a_new["max_tokens"] == 64
    assert json.loads((a / "unit" / "identity.json").read_text())["max_tokens"] == 64
    assert len([p for p in a.glob("stale-*") if p.is_dir()]) == 1          # A's original sits archived
    assert rep.TargetRecord(a).verdict == "admitted"
    # B is UNCHANGED: its own copy, its original verdict, the ORIGINAL source attempt — never A's new unit
    rb2 = rep.TargetRecord(b)
    assert rb2.verdict == "not_admitted" and rb2.binding_problems == [] and rb2.problems == []
    assert rb2.admission["attempt_id"] == a_first_attempt and rb2.admission["max_tokens"] == 128
    assert rb2.admission == a_first and rb2.identity["attempt_id"] == a_first_attempt
    assert rb2.head_to_head["ratio_at_S"] < 1.0
    assert "verdict=not_admitted" in (b / "RESULT").read_text()
    reuse = json.loads((b / adm.UNIT_REUSE_FILE).read_text())
    assert reuse["source"]["attempt_id"] == a_first_attempt and reuse["attempt_id"] == b_attempt
    out = rep.build(records, tmp_path / "figs", all_dirs=True)
    by_name = {r.path.name: r for r in out["records"]}
    assert by_name["gemma4e4b_8401"].verdict == "admitted" and by_name["gemma4e4b_8402"].verdict == "not_admitted"
    assert out["table"].count("**admitted**") == 1 and out["table"].count("**not_admitted**") == 1
    # the "resume from B" chain: B has no unit/ of its own, so C runs its own unit (never a reuse of a copy)
    fc = Fakes(native_shift=None, per_row_shift=(1, 2), ratio=0.9)
    rc, c = _run(tmp_path / "w", records / "gemma4e4b_8403", fc, resume=b)
    assert rc == 0 and fc.native_attempts == 1
    assert any("no DONE.json" in r for r in json.loads((c / adm.UNIT_REUSE_FILE).read_text())["reasons"])

    # --- tampering with B's copy or its reuse record is UNBOUND, never A's verdict ---
    def variant(name, mutate):
        d = records / name
        shutil.copytree(b, d)
        mutate(d)
        return rep.TargetRecord(d)

    def copy_replaced_by_the_sources_new_admission(d):
        shutil.copy(a / "unit" / "admission.json", d / adm.REUSED_UNIT_DIR / "admission.json")

    def copy_gone(d):
        shutil.rmtree(d / adm.REUSED_UNIT_DIR)

    def reuse_record_from_before_the_copy(d):
        r = json.loads((d / adm.UNIT_REUSE_FILE).read_text())
        del r["source"]
        (d / adm.UNIT_REUSE_FILE).write_text(json.dumps(r))

    def reuse_record_of_another_attempt(d):
        r = json.loads((d / adm.UNIT_REUSE_FILE).read_text())
        r["attempt_id"] = "ee" * 16
        (d / adm.UNIT_REUSE_FILE).write_text(json.dumps(r))

    def head_to_head_swapped(d):
        shutil.copy(a / "unit" / "head_to_head.json", d / adm.REUSED_UNIT_DIR / "head_to_head.json")

    v1 = variant("gemma4e4b_8404", copy_replaced_by_the_sources_new_admission)
    assert v1.verdict == "unbound" and v1.admission is None and v1.admission_unbound["verdict"] == "admitted"
    assert any("attempt ids disagree" in p for p in v1.binding_problems)
    assert any("admission.json sha256" in p and "!= the accepted" in p for p in v1.binding_problems)
    v2 = variant("gemma4e4b_8405", copy_gone)
    assert v2.verdict == "unbound" and any("no self-contained copy" in p for p in v2.binding_problems)
    assert "no admission.json" in v2.problems
    v3 = variant("gemma4e4b_8406", reuse_record_from_before_the_copy)
    assert v3.verdict == "unbound" and any("records no accepted source" in p for p in v3.binding_problems)
    v4 = variant("gemma4e4b_8407", reuse_record_of_another_attempt)
    assert v4.verdict == "unbound" and any("is not this record's attempt" in p for p in v4.binding_problems)
    v5 = variant("gemma4e4b_8408", head_to_head_swapped)
    assert v5.verdict == "unbound" and any("head_to_head.json sha256" in p for p in v5.binding_problems)
    # the acceptance itself is held to the same rule: a snapshot that does not bind is refused, no copy is kept
    fd = Fakes(native_shift=None, per_row_shift=(1, 2), ratio=1.4)
    a_done = json.loads((a / "unit" / "DONE.json").read_text())
    a_done["attempt_id"] = "dd" * 16
    (a / "unit" / "DONE.json").write_text(json.dumps(a_done))
    rc, d = _run(tmp_path / "w", records / "gemma4e4b_8409", fd, resume=a, extra=("--probe-max-tokens", "64"))
    assert rc == 0 and fd.native_attempts == 1 and not (d / adm.REUSED_UNIT_DIR).exists()
    reasons = json.loads((d / adm.UNIT_REUSE_FILE).read_text())["reasons"]
    # refused by the inventory (A2: INVENTORY.json's attempt != the tampered DONE.json's) before the
    # three-record binding; either names the attempt disagreement
    assert any(r.startswith(("prior unit inventory:", "prior unit binding:")) and "attempt" in r
               for r in reasons), reasons


# --------------------------------------------------------------------------- #
# evidence inventory and config fingerprint                                    #
# --------------------------------------------------------------------------- #

def test_a_complete_unit_writes_its_evidence_inventory_and_done_binds_it(tmp_path):
    """A2: unit/INVENTORY.json lists every file under unit/ with its digest
    and the required-evidence list; DONE.json carries its sha256; the
    required set covers the gate record + evidence, the native attestation /
    endpoint / cells_served / positive_controls, every probe leg's rows, every
    throughput leg's windows + gauges + util, the consumed oracle reference."""
    rc, out = _run(tmp_path, tmp_path / "rec" / "gemma4e4b_8501", Fakes(native_shift=None, per_row_shift=(1, 2)))
    assert rc == 0
    unit = out / "unit"
    inv = json.loads((unit / adm.UNIT_INVENTORY_FILE).read_text())
    done = json.loads((unit / "DONE.json").read_text())
    assert done[adm.DONE_INVENTORY_SHA_KEY] == hashlib.sha256((unit / adm.UNIT_INVENTORY_FILE).read_bytes()).hexdigest()
    assert inv["attempt_id"] == done["attempt_id"] and inv["missing"] == [] and done["n_files"] == inv["n_files"]
    assert adm.UNIT_INVENTORY_FILE not in inv["files"] and "DONE.json" not in inv["files"]
    for entry in ("gate_canary/gate_record.json", "gate_canary/gate_evidence.json", "lora_native/serving_attestation.json",
                  "lora_native_endpoint.json", "lora_native/cells_served.json", "lora_native/positive_controls.json",
                  "oracle_reference.json", "probes/lens_per_row.jsonl", "throughput/lens_per_row.json"):
        assert entry in inv["required"] and entry in inv["files"], entry
    assert "throughput/raw/gauges_lora_native_*.json" in inv["required"]
    assert any(n.startswith("throughput/raw/util_lens_stacked_") for n in inv["files"])
    assert (unit / "oracle_reference.json").read_bytes() == (out / "prep" / "oracle" / "reference.json").read_bytes()
    for rel, meta in inv["files"].items():
        assert hashlib.sha256((unit / rel).read_bytes()).hexdigest() == meta["sha256"], rel
    assert adm.unit_inventory_problems(unit) == []
    assert rep.TargetRecord(out).problems == []
    # a file dropped after completion, a byte changed, a planted file: each named
    (unit / "probes" / "lens_stacked.jsonl").unlink()
    assert any("inventoried file missing: probes/lens_stacked.jsonl" in p for p in adm.unit_inventory_problems(unit))
    (unit / "probes" / "lens_stacked.jsonl").write_text("{}\n")
    assert any("probes/lens_stacked.jsonl: sha256" in p for p in adm.unit_inventory_problems(unit))
    (unit / "planted.json").write_text("{}")
    assert any("does not list: ['planted.json']" in p for p in adm.unit_inventory_problems(unit))
    assert rep.TargetRecord(out).problems != []
    # the required list is a function of the admission: a failed canary needs no record
    assert "gate_canary/gate_record.json" not in adm.unit_required_evidence({"gate_canary": {"passed": False}})
    assert adm.evidence_missing(["a", "b/*.csv"], ["a", "b/x.csv"]) == []
    assert adm.evidence_missing(["a", "b/*.csv"], ["b/x.txt"]) == ["a", "b/*.csv"]


def test_a_summaries_only_source_cannot_be_reused_and_a_complete_one_reuses_whole(tmp_path):
    """A2 (reproduced): a source holding only identity / DONE / admission /
    head_to_head reused with zero standups, exit 0, `admitted`, report
    problems == []. Now the reuse snapshots the WHOLE inventory: a source with
    no INVENTORY.json, one missing an inventoried probe leg, and one whose
    probe rows were altered each refuse by name and the unit reruns; a
    complete source reuses whole, every reference the copied documents name
    resolves under reused_unit/, and the report sees no problem."""
    import shutil
    records = tmp_path / "records"
    rc, a = _run(tmp_path / "w", records / "gemma4e4b_8601", Fakes(native_shift=None, per_row_shift=(1, 2)))
    assert rc == 0
    # (a) the reproduction: summaries only
    summaries = records / "gemma4e4b_8602"
    shutil.copytree(a, summaries)
    for p in list((summaries / "unit").rglob("*")):
        if p.is_file() and p.relative_to(summaries / "unit").as_posix() not in (
                "identity.json", "DONE.json", "admission.json", "head_to_head.json"):
            p.unlink()
    f = Fakes(native_shift=None, per_row_shift=(1, 2))
    rc, b = _run(tmp_path / "w", records / "gemma4e4b_8603", f, resume=summaries)
    assert rc == 0 and f.native_attempts == 1 and not (b / adm.REUSED_UNIT_DIR).exists()
    reasons = json.loads((b / adm.UNIT_REUSE_FILE).read_text())["reasons"]
    assert any("prior unit inventory:" in r and adm.UNIT_INVENTORY_FILE in r for r in reasons), reasons
    # (b) an inventoried probe leg missing from the source
    partial = records / "gemma4e4b_8604"
    shutil.copytree(a, partial)
    (partial / "unit" / "probes" / "lens_stacked.jsonl").unlink()
    f = Fakes(native_shift=None, per_row_shift=(1, 2))
    rc, c = _run(tmp_path / "w", records / "gemma4e4b_8605", f, resume=partial)
    assert rc == 0 and f.native_attempts == 1
    reasons = json.loads((c / adm.UNIT_REUSE_FILE).read_text())["reasons"]
    assert any("inventoried file missing: probes/lens_stacked.jsonl" in r for r in reasons), reasons
    # (c) altered probe rows in the source
    altered = records / "gemma4e4b_8606"
    shutil.copytree(a, altered)
    (altered / "unit" / "probes" / "lora_native.jsonl").write_text("{}\n")
    f = Fakes(native_shift=None, per_row_shift=(1, 2))
    rc, d = _run(tmp_path / "w", records / "gemma4e4b_8607", f, resume=altered)
    assert rc == 0 and f.native_attempts == 1
    assert any("probes/lora_native.jsonl: sha256" in r for r in json.loads((d / adm.UNIT_REUSE_FILE).read_text())["reasons"])
    # (d) the complete source reuses WHOLE; every reference resolves locally
    f = Fakes(native_shift=None, per_row_shift=(1, 2))
    rc, e = _run(tmp_path / "w", records / "gemma4e4b_8608", f, resume=a)
    assert rc == 0 and f.native_attempts == 0 and f.gate_calls == 0
    copy = e / adm.REUSED_UNIT_DIR
    inv = json.loads((a / "unit" / adm.UNIT_INVENTORY_FILE).read_text())
    assert all((copy / rel).read_bytes() == (a / "unit" / rel).read_bytes() for rel in inv["files"])
    assert (copy / "gate_canary" / "gate_record.json").is_file() and (copy / "lora_native" / "serving_attestation.json").is_file()
    assert (copy / "probes" / "lens_per_row.jsonl").is_file() and (copy / "oracle_reference.json").is_file()
    assert adm.unit_inventory_problems(copy) == []
    r = rep.TargetRecord(e)
    assert r.verdict == "admitted" and r.binding_problems == [] and r.problems == []
    rec_path = r.admission["gate_canary"]["record"]
    assert rec_path.startswith(str(a / "unit"))                          # the copied document still names the SOURCE path
    assert r.gate_record_path == copy / "gate_canary" / "gate_record.json" and r.gate_record_path.is_file()
    assert r.resolve_path("/elsewhere/x.json") is None
    assert json.loads(r.gate_record_path.read_text())["status"] == "completed"
    # the copy is bound whole: dropping a copied probe leg is `unbound`
    (copy / "probes" / "lens_stacked_rep.jsonl").unlink()
    r2 = rep.TargetRecord(e)
    assert r2.verdict == "unbound" and any("probes/lens_stacked_rep.jsonl" in p for p in r2.binding_problems)


def test_every_argparse_option_is_classified_and_the_fingerprint_binds_them(tmp_path):
    """A3: every option of build_parser is IDENTITY, PATH or OPERATIONAL —
    exactly one, and nothing unclassified — so a new option cannot be added
    without deciding whether it changes what must be certified; the
    fingerprint covers the IDENTITY options and changes with any of them."""
    dests = set(mc.parser_option_dests())
    classes = (set(mc.IDENTITY_OPTIONS), set(mc.PATH_OPTIONS), set(mc.OPERATIONAL_OPTIONS))
    assert classes[0] | classes[1] | classes[2] == dests, sorted(dests - (classes[0] | classes[1] | classes[2]))
    assert not (classes[0] & classes[1]) and not (classes[0] & classes[2]) and not (classes[1] & classes[2])
    for must in ("gpu_mem_util", "max_logprobs", "control_probe_max_tokens", "probe_max_tokens",
                 "throughput_max_tokens", "batches", "window_s", "warmup_s", "repeats", "eval", "model_key",
                 "max_model_len", "max_num_seqs"):
        assert must in mc.IDENTITY_OPTIONS, must
    for path in ("out", "resume_from", "repo_dir", "defenses_dir", "adapter_bank", "pins"):
        assert path in mc.PATH_OPTIONS, path
    defenses, pins, sha = _world(tmp_path, "gemma4e4b")
    a1 = _args(tmp_path, tmp_path / "o1", key="gemma4e4b", defenses=defenses, pins=pins)
    a2 = _args(tmp_path, tmp_path / "o2", key="gemma4e4b", defenses=defenses, pins=pins, resume=tmp_path / "o1")
    assert mc.config_fingerprint(a1) == mc.config_fingerprint(a2)         # paths never enter the fingerprint
    a3 = _args(tmp_path, tmp_path / "o1", key="gemma4e4b", defenses=defenses, pins=pins, extra=("--gpu-mem-util", "0.3"))
    fp1, fp3 = mc.config_fingerprint(a1), mc.config_fingerprint(a3)
    assert fp1["sha256"] != fp3["sha256"] and fp1["options"]["gpu_mem_util"] == "0.9" and fp3["options"]["gpu_mem_util"] == "0.3"
    assert set(fp1["options"]) == set(mc.IDENTITY_OPTIONS)
    a4 = _args(tmp_path, tmp_path / "o1", key="gemma4e4b", defenses=defenses, pins=pins, extra=("--ready-timeout-s", "60"))
    assert mc.config_fingerprint(a4) == fp1                                 # a timeout never does
    # an option added to the parser without a classification refuses
    extra = mc.build_parser()
    extra.add_argument("--brand-new-knob", default=None)
    monkey = list(mc.parser_option_dests(extra))
    assert "brand_new_knob" in monkey
    import unittest.mock as um
    with um.patch.object(mc, "build_parser", lambda: extra):
        with pytest.raises(mc.CertRefusal, match="brand_new_knob.*not classified"):
            mc.config_fingerprint(a1)


def test_resume_after_a_standup_setting_change_reruns_the_unit(tmp_path):
    """A3 (reproduced): --gpu-mem-util 0.9 -> 0.3 between two runs reused the
    old unit with zero native launches and read `admitted`. The fingerprint
    is in the identity: the resume names `identity.config` and reruns; the
    same for --max-logprobs and --control-probe-max-tokens; an unchanged
    configuration still reuses."""
    records = tmp_path / "records"
    rc, first = _run(tmp_path / "w", records / "gemma4e4b_8701", Fakes(native_shift=None, per_row_shift=(1, 2)))
    assert rc == 0
    ident = json.loads((first / "unit" / "identity.json").read_text())
    assert ident["config"]["gpu_mem_util"] == "0.9" and len(ident["config_fingerprint"]) == 64
    for i, extra in enumerate((("--gpu-mem-util", "0.3"), ("--max-logprobs", "8"),
                               ("--control-probe-max-tokens", "16"))):
        f = Fakes(native_shift=None, per_row_shift=(1, 2))
        rc, out = _run(tmp_path / "w", records / f"gemma4e4b_871{i}", f, resume=first, extra=extra)
        assert rc == 0 and f.native_attempts == 1, extra
        reasons = json.loads((out / adm.UNIT_REUSE_FILE).read_text())["reasons"]
        assert any(r.startswith("identity.config:") for r in reasons), (extra, reasons)
        assert any(r.startswith("identity.config_fingerprint:") for r in reasons), (extra, reasons)
        assert extra[0].lstrip("-").replace("-", "_") in " ".join(reasons)
        assert f.last_native_kw[{"--gpu-mem-util": "gpu_mem_util", "--max-logprobs": "max_logprobs",
                                 "--control-probe-max-tokens": "probe_max_tokens"}[extra[0]]] == \
            (extra[1] if extra[0] == "--gpu-mem-util" else int(extra[1]))
    f = Fakes(native_shift=None, per_row_shift=(1, 2))
    rc, same = _run(tmp_path / "w", records / "gemma4e4b_8720", f, resume=first)
    assert rc == 0 and f.native_attempts == 0


# --------------------------------------------------------------------------- #
# the LoRA kernel index ceiling (gemma canary job 488932)                       #
# --------------------------------------------------------------------------- #

#: the failed job's server log, trimmed to the lines that classify it (unit/lora_native/vllm_server.log:335-583)
ILLEGAL_ACCESS_LOG = """(EngineCore pid=3140933) INFO 09-12 09:50:14 [kernel_warmup.py:170] Using FlashInfer autotune cache file
(EngineCore pid=3140933) ERROR 09-12 09:50:35 [core.py:1231] EngineCore failed to start.
(EngineCore pid=3140933) ERROR 09-12 09:50:35 [core.py:1231]     kernel_warmup(self)
(EngineCore pid=3140933) ERROR 09-12 09:50:35 [core.py:1231]     flashinfer_autotune(worker.model_runner)
(EngineCore pid=3140933) ERROR 09-12 09:50:35 [core.py:1231]     runner._dummy_run(**dummy_run_kwargs)
(EngineCore pid=3140933) ERROR 09-12 09:50:35 [core.py:1231]     buf15 = empty_strided_cuda((s59, 10240), (10240, 1), torch.bfloat16)
(EngineCore pid=3140933) ERROR 09-12 09:50:35 [core.py:1231] torch.AcceleratorError: CUDA error: an illegal memory access was encountered
(EngineCore pid=3140933) ERROR 09-12 09:50:35 [core.py:1231] Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
terminate called after throwing an instance of 'c10::AcceleratorError'
  what():  CUDA error: an illegal memory access was encountered
(APIServer pid=3140821)     raise RuntimeError(
(APIServer pid=3140821) RuntimeError: Engine core initialization failed. See root cause above.
""" + "x" * 5000 + "\n"     # the API server's own long tail, so the root cause is NOT in the last 4000 bytes


def test_the_driver_lowers_S_to_the_lora_kernel_index_ceiling_before_the_first_attempt_and_runs_all_three_legs_there(tmp_path):
    """Job 488932: the gemma table's 121 x 4864 = 588,544 batched tokens
    overflowed the pinned LoRA Triton kernels' int32 offsets (x 4096) at
    warmup. The driver now computes the ceiling at build_context — BEFORE
    the first attempt, no GPU minute lost — lowers S to 524,287 // 4864 =
    107, tells the standup the reason (its strict identity guard would
    refuse a bare 107), runs ALL THREE legs at 107 (like-for-like), and the
    identity / manifest / head_to_head / admission / report all label it.
    The per-eval table is untouched; the same configuration reuses."""
    fakes = Fakes(native_shift=None, per_row_shift=(1, 2), module_dim=4096)
    rc, out = _run(tmp_path, tmp_path / "rec" / "gemma4e4b_8801", fakes)
    assert rc == 0 and fakes.native_attempts == 1
    # the standup was asked for the LOWERED S with the reason; both lens legs ran at it
    assert fakes.last_native_kw["max_num_seqs"] == 107
    assert fakes.last_native_kw["geometry_override_reason"] == sa.LORA_KERNEL_CEILING_REASON == "lora_kernel_index_ceiling"
    assert [g.max_num_seqs for g in fakes.lens_geoms] == [107, 107]
    assert all(g.seqs_source == "override" for g in fakes.lens_geoms)
    unit = out / "unit"
    ident = json.loads((unit / "identity.json").read_text())
    assert ident["geometry"] == {"L": 4864, "S": 107}
    low = ident["geometry_lowered"]
    assert (low["reason"], low["from"], low["to"], low["L"], low["D"], low["ceiling"]) == \
        ("lora_kernel_index_ceiling", 121, 107, 4864, 4096, 524_287)
    assert low["product_table"] == 588_544 and low["product"] == 520_448
    man = json.loads((out / "manifest.json").read_text())
    assert man["geometry"]["S"] == 107 and man["geometry"]["L"] == 4864 and man["geometry_lowered"] == low
    assert man["facts"]["lora_kernel_index_ceiling"] == {"D": 4096, "widest_module": "synthetic.layers.0.self_attn.o_proj",
                                                         "ceiling": 524_287, "L": 4864, "S": 107, "product": 520_448, "lowered": True}
    # like-for-like: every throughput leg's headline B is the lowered S
    for leg, want in (("lora_native", [8, 32, 107]), ("lens_stacked", [8, 32, 107]), ("lens_per_row", [107])):
        assert json.loads((unit / "throughput" / f"{leg}.json").read_text())["B_list"] == want, leg
    h2h = json.loads((unit / "head_to_head.json").read_text())
    assert h2h["headline_B"] == 107 and set(h2h["per_B"]) == {"8", "32", "107"} and h2h["not_slower_at_S"]
    assert h2h["geometry"] == {"L": 4864, "S": 107, "lowered": low}
    rec = json.loads((unit / "admission.json").read_text())
    assert rec["geometry"] == {"L": 4864, "S": 107, "lowered": low} and rec["verdict"] == "admitted"
    assert rec["head_to_head"]["headline_B"] == 107
    assert "verdict=admitted" in (out / "RESULT").read_text()
    # the report labels the lowering (table line + the admission facts)
    r = rep.TargetRecord(out)
    assert r.problems == []
    md = rep.table([r])
    assert "LOWERED S 121 -> 107 (lora_kernel_index_ceiling)" in md
    # the same configuration REUSES the lowered unit (the lowering is part of the identity)
    f2 = Fakes(native_shift=None, per_row_shift=(1, 2), module_dim=4096)
    rc, second = _run(tmp_path, tmp_path / "rec" / "gemma4e4b_8802", f2, resume=out)
    assert rc == 0 and f2.native_attempts == 0
    # a narrow base never lowers: the table's S serves, nothing is labelled
    f3 = Fakes(native_shift=None, per_row_shift=(1, 2), module_dim=32)
    rc, third = _run(tmp_path, tmp_path / "rec" / "gemma4e4b_8803", f3)
    assert rc == 0 and f3.last_native_kw["max_num_seqs"] == 121 and f3.last_native_kw.get("geometry_override_reason") is None
    ident3 = json.loads((third / "unit" / "identity.json").read_text())
    assert ident3["geometry"] == {"L": 4864, "S": 121} and ident3["geometry_lowered"] is None
    assert json.loads((third / "manifest.json").read_text())["facts"]["lora_kernel_index_ceiling"]["lowered"] is False
    # nemotron's D=5120 at its table S=161: 786,432 -> 419,430 // 4864 = 86
    f4 = Fakes(native_shift=None, per_row_shift=(1, 2), module_dim=5120, S=161)
    rc, fourth = _run(tmp_path, tmp_path / "rec" / "nemotron3nano4b_8804", f4, key="nemotron3nano4b")
    assert rc == 0 and f4.last_native_kw["max_num_seqs"] == 86
    low4 = json.loads((fourth / "unit" / "identity.json").read_text())["geometry_lowered"]
    assert (low4["from"], low4["to"], low4["ceiling"], low4["product"]) == (161, 86, 419_430, 418_304)
    # ... and the Mamba retry still lowers FROM the ceiling S, recording the prior lowering
    f5 = Fakes(native_shift=None, per_row_shift=(1, 2), module_dim=5120, S=161, mamba_once=True)
    rc, fifth = _run(tmp_path, tmp_path / "rec" / "nemotron3nano4b_8805", f5, key="nemotron3nano4b")
    assert rc == 0 and f5.native_attempts == 2 and f5.last_native_kw["max_num_seqs"] == 43
    low5 = json.loads((fifth / "unit" / "identity.json").read_text())["geometry_lowered"]
    assert low5["reason"] == mc.MAMBA_LOWER_REASON and (low5["from"], low5["to"]) == (86, 43)
    assert low5["after"]["reason"] == "lora_kernel_index_ceiling" and low5["after"]["to"] == 86


def test_an_engine_init_failure_is_classified_by_name_for_the_next_reader(tmp_path):
    """Job 488932's record said only `exited with 1` (`mamba_refusal: false`).
    The classifier reads a wide window of the server log (the root cause sits
    hundreds of lines above the API server's final traceback) and names it;
    the attempt record, the ControlsFailed error and FAILED.json carry the
    class and its hint; no retry fires for a non-Mamba failure."""
    cls = mc.classify_engine_failure(ILLEGAL_ACCESS_LOG)
    assert cls["failure_class"] == "illegal_memory_access_at_warmup"
    assert cls["illegal_memory_access"] and cls["at_warmup"] and not cls["mamba_refusal"] and not cls["cuda_out_of_memory"]
    assert "int32" in cls["hint"] and "lora_kernel_index_ceiling" in cls["hint"] and "job 488932" in cls["hint"]
    assert mc.classify_engine_failure("torch.AcceleratorError: CUDA error: an illegal memory access was encountered")["failure_class"] == "illegal_memory_access"
    assert mc.classify_engine_failure("ValueError: Mamba cache block count 12 < required 24")["failure_class"] == "mamba_cache_blocks"
    assert mc.classify_engine_failure("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate")["failure_class"] == "cuda_out_of_memory"
    assert mc.classify_engine_failure("vllm serve exited with 1 before becoming ready")["failure_class"] == "engine_exit"
    assert mc.classify_engine_failure("")["failure_class"] == "engine_exit"
    assert set(mc.ENGINE_FAILURE_HINTS) == set(mc.ENGINE_FAILURE_CLASSES)
    # through the driver
    fakes = Fakes(fail_with_log=ILLEGAL_ACCESS_LOG)
    rc, out = _run(tmp_path, tmp_path / "rec" / "gemma4e4b_8901", fakes)
    assert rc == mc.RC_CONTROLS and fakes.native_attempts == 1
    (att,) = json.loads((out / "unit" / "native_standup_attempts.json").read_text())
    assert att["failure_class"] == "illegal_memory_access_at_warmup" and att["mamba_refusal"] is False
    assert att["illegal_memory_access"] is True and att["at_warmup"] is True and "int32" in att["hint"]
    failed = json.loads((out / "FAILED.json").read_text())
    assert failed["kind"] == "controls" and "native standup refused (illegal_memory_access_at_warmup)" in failed["error"]
    assert "lora_kernel_index_ceiling" in failed["error"]
    assert "state=FAILED exit=4" in (out / "RESULT").read_text()


def test_a_failed_positive_control_is_classified_as_controls_failed_not_engine_exit(tmp_path):
    """Job 489125: the engine warmed up, attested and the standup then refused
    `selection_honored`, but native_standup_attempts.json said
    `failure_class: engine_exit` with the "before readiness" hint while
    FAILED.json said kind `controls`. A PositiveControlsFailed out of
    bring_up_lora is `controls_failed` naming the control, whatever the log
    says (the warmup lines are in it — the engine DID warm up)."""
    rec = {"failed_control": "noop_certificate"}
    cls = mc.classify_engine_failure("INFO kernel_warmup done", mc.lsu.PositiveControlsFailed("x", rec))
    assert cls["failure_class"] == "controls_failed" and cls["control"] == "noop_certificate"
    assert "AFTER readiness" in cls["hint"] and "positive_controls.json" in cls["hint"]
    assert cls["at_warmup"] is True and cls["illegal_memory_access"] is False      # the log bits stay informational
    assert mc.classify_engine_failure("INFO kernel_warmup done")["failure_class"] == "engine_exit"
    assert mc.classify_engine_failure("", RuntimeError("x"))["control"] is None
    assert "controls_failed" in mc.ENGINE_FAILURE_CLASSES and set(mc.ENGINE_FAILURE_HINTS) == set(mc.ENGINE_FAILURE_CLASSES)
    # through the driver: the record, the error and FAILED.json agree; no retry; rc 4
    fakes = Fakes(fail_controls="selection_honored")
    rc, out = _run(tmp_path, tmp_path / "rec" / "gemma4e4b_9101", fakes)
    assert rc == mc.RC_CONTROLS and fakes.native_attempts == 1
    (att,) = json.loads((out / "unit" / "native_standup_attempts.json").read_text())
    assert att["failure_class"] == "controls_failed" and att["control"] == "selection_honored"
    assert att["mamba_refusal"] is False and "before readiness" not in att["hint"]
    failed = json.loads((out / "FAILED.json").read_text())
    assert failed["kind"] == "controls"
    assert "native standup refused (controls_failed: selection_honored)" in failed["error"]
    assert "AFTER readiness" in failed["error"]
    assert "state=FAILED exit=4" in (out / "RESULT").read_text()


# --------------------------------------------------------------------------- #
# the offline readmit and the report's preference for it                         #
# --------------------------------------------------------------------------- #

def _clean_tree(monkeypatch, sha="c" * 40):
    """Pin ``adm._git_tree`` to a CLEAN tree whose committed module is the one
    on disk: the readmit happy paths must not depend on whether this checkout
    happens to be dirty (the real ``_git_tree`` is exercised for its shape in
    ``test_readmit_writes_beside…`` and end to end by the record readmits)."""
    module = hashlib.sha256(Path(adm.__file__).read_bytes()).hexdigest()
    monkeypatch.setattr(adm, "_git_tree", lambda mp, claimed_sha=None: {
        "sha": claimed_sha or sha, "dirty": False, "module_sha256": module, "module_sha256_at_tree": module,
        "module_matches_tree": True, "bind_skipped": None})
    return module


def test_readmit_writes_beside_the_admission_never_over_it_and_carries_provenance(tmp_path, monkeypatch):
    """``readmit`` on a complete record verifies every numerical input's bytes,
    rebuilds the inputs through the job's loader, cross-checks the rebuilt
    agreement table against the original, writes ``unit/admission.readmitted.
    json`` and leaves ``admission.json`` byte-identical (the DONE.json binding
    and the evidence inventory still hold — the post-hoc file is neither
    inventoried nor "unlisted"). The output carries the original's sha256, its
    verdict and reasons, the verified input digests, the tree sha, the rule
    and a timestamp. A second readmit under the SAME rule regenerates the
    derived file (``superseded`` None) and still touches nothing else. The CLI
    returns 0, prints the JSON and then ``verdict=… outcome=… written=…`` as
    its last line; ``--require-admitted`` returns 3 when the verdict is not
    admitted; a sourced margin is recorded, an unsourced one refuses (2)."""
    # the REAL _git_tree: its shape, bound to this module's bytes (clean or dirty is the checkout's business)
    real = adm._git_tree(Path(adm.__file__))
    assert real["module_sha256"] == hashlib.sha256(Path(adm.__file__).read_bytes()).hexdigest()
    if real["bind_skipped"] is None:
        assert re.fullmatch(r"[0-9a-f]{40}", real["sha"]) and isinstance(real["dirty"], bool)
        assert re.fullmatch(r"[0-9a-f]{64}", real["module_sha256_at_tree"]) and isinstance(real["module_matches_tree"], bool)
    _clean_tree(monkeypatch)
    fakes = Fakes(native_shift=None, per_row_shift=(1, 2), ratio=1.4)
    rc, out = _run(tmp_path, tmp_path / "rec" / "gemma4e4b_1001", fakes)
    assert rc == 0
    unit = out / "unit"
    before = (unit / "admission.json").read_bytes()
    done_before = (unit / "DONE.json").read_bytes()
    assert adm.unit_binding_problems(unit) == [] and adm.unit_inventory_problems(unit) == []
    res = adm.readmit(out)
    assert res["problems"] == [] and res["written"] == str(unit / adm.READMITTED_ADMISSION_FILE), res
    assert res["verdict"] == "admitted" and res["outcome"] == "admitted" and res["superseded"] is None
    assert (unit / "admission.json").read_bytes() == before and (unit / "DONE.json").read_bytes() == done_before
    assert adm.unit_binding_problems(unit) == [] and adm.unit_inventory_problems(unit) == []
    doc = json.loads((unit / adm.READMITTED_ADMISSION_FILE).read_text())
    assert adm.validate_admission_record(doc) == [] and doc["verdict"] == "admitted" and doc["rule"] == adm.ADMISSION_RULE
    # the readiness was RE-READ from the attestation on disk through the strict reader, and equals the job's
    assert doc["readiness"]["attested"] is True and doc["readiness"] == json.loads(before)["readiness"]
    assert res["gates"] == doc["gates"] and all(doc["gates"][k] is True for k in adm.FUNCTIONAL_GATES)
    prov = doc["readmitted_from"]
    assert prov["sha256_of_original_admission_json"] == hashlib.sha256(before).hexdigest()
    assert prov["rule"] == adm.ADMISSION_RULE and prov["original_verdict"] == "admitted" and prov["original_reasons"] == []
    assert prov["original_schema_version"] == 5 and prov["admission_json"] == str(unit / "admission.json")
    assert isinstance(prov["readmitted_at"], float) and "tree_sha_of_readmit" in prov and "module_sha256" in prov
    assert prov["module_sha256"] == hashlib.sha256(Path(adm.__file__).read_bytes()).hexdigest()
    assert prov["tree_dirty"] is False and prov["module_matches_tree"] is True and prov["provenance_note"] is None
    assert prov["tree_sha_of_readmit"] == "c" * 40 and prov["tree_bind_skipped"] is None
    assert "margin" not in prov and prov["superseded"] is None
    # the RESULT sidecar beside the job's RESULT, the original line preserved, in the driver's vocabulary
    assert res["sidecar"] == str(out / adm.RESULT_SIDECAR)
    side = (out / adm.RESULT_SIDECAR).read_text().strip()
    fields = dict(kv.split("=", 1) for kv in side.split())
    assert fields["rule"] == adm.ADMISSION_RULE and fields["verdict"] == "admitted" and fields["outcome"] == "admitted"
    assert fields["readiness"] == "attested" and fields["controls"] == "6/6" and fields["gate"] == "passed"
    assert fields["not_slower"] == "True" and fields["numerics"] == "diagnostic"
    assert fields["zero"] == "exact" and fields["original_verdict"] == "admitted"
    assert fields["module"] == prov["module_sha256"][:12] and len(fields["tree"]) == 8 and "dirty" not in fields
    assert (out / "RESULT").read_text().startswith("state=DONE exit=0")
    assert f"rule={adm.ADMISSION_RULE} readiness=attested controls=6/6 gate=passed" in (out / "RESULT").read_text()
    iv = prov["inputs_verified"]
    assert set(iv["legs"]) == set(adm.LEGS) and iv["inventory_verified"] == iv["inventory_listed"] > 40
    assert iv["absent_diagnostic_logs"] == [] and iv["done_binds_inventory"] is True
    for leg, meta in iv["legs"].items():
        assert meta["sha256"] == hashlib.sha256((unit / "probes" / f"{leg}.jsonl").read_bytes()).hexdigest()
        assert meta["n_rows"] == 43
    assert doc["attempt_id"] == json.loads(before)["attempt_id"]
    # the rebuilt inputs ARE the job's inputs: the table and the seeded bootstrap agree exactly
    assert doc["agreement"] == json.loads(before)["agreement"]
    assert (doc["numerics"]["prompt_resampling_sensitivity"]
            == json.loads(before)["numerics"]["prompt_resampling_sensitivity"])
    # the derived file is regenerated, the admission still untouched; the unit dir path works too
    res2 = adm.readmit(unit)
    assert res2["written"] and res2["superseded"] is None and (unit / "admission.json").read_bytes() == before
    # the readmitted file is never part of the unit's evidence walk
    assert adm.READMITTED_ADMISSION_FILE not in adm._walk_unit_files(unit)
    # CLI: JSON then the verdict line; --require-admitted; NO margin flag exists any more
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc_cli = adm.main(["readmit", str(out)])
    lines = buf.getvalue().rstrip("\n").split("\n")
    assert rc_cli == 0 and lines[-1].startswith("verdict=admitted outcome=admitted written=")
    assert json.loads("\n".join(lines[:-1]))["verdict"] == "admitted"
    with contextlib.redirect_stdout(io.StringIO()):
        assert adm.main(["readmit", str(out), "--require-admitted"]) == 0
    with contextlib.redirect_stderr(io.StringIO()), pytest.raises(SystemExit):
        adm.main(["readmit", str(out), "--admission-margin", "0.05", "--margin-source", "x"])
    doc = json.loads((unit / adm.READMITTED_ADMISSION_FILE).read_text())
    assert doc["floor"]["margin"] == adm.DIAGNOSTIC_MARGIN == doc["numerics"]["margin"]
    # reevaluate() is the readmit without the write: the same record, nothing touched
    listing = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    digests = {p: hashlib.sha256((out / p).read_bytes()).hexdigest() for p in listing}
    ev = adm.reevaluate(out)
    assert ev["problems"] == [] and ev["new"]["verdict"] == "admitted" and ev["new"]["rule"] == adm.ADMISSION_RULE
    assert ev["original_sha256"] == hashlib.sha256(before).hexdigest() and ev["unit_dir"] == str(unit)
    assert {k: v for k, v in ev["new"].items() if k != "readmitted_from"} == {k: v for k, v in doc.items() if k != "readmitted_from"}
    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()) == listing
    assert {p: hashlib.sha256((out / p).read_bytes()).hexdigest() for p in listing} == digests


def test_readmit_refuses_an_aggregate_preserving_tamper_of_a_leg_at_the_bytes(tmp_path, monkeypatch):
    """The aggregate-preserving tamper: flipping a verdict in memory by
    RE-ALLOCATING per-prompt contributions while preserving every aggregate
    (the same divergence moved from one probe to another of equal length)
    passes the marginal cross-checks. Here the same tamper is written to
    ``probes/lora_native.jsonl``: the agreement table and the ulps still
    reproduce the original exactly, but the JSONL no longer hashes to its
    DONE.json ``sha256_of_jsonl`` (nor to the inventory), so the readmit
    refuses at the BYTES and writes nothing. Forging the DONE.json sha too is
    caught by the inventory; forging the inventory is caught by DONE.json's
    ``inventory_sha256``. A record dir with no admission refuses (exit 2)."""
    _clean_tree(monkeypatch)
    fakes = Fakes(native_shift=(0, 5), per_row_shift=(1, 2), ratio=1.4)     # native diverges on probe 0
    rc, out = _run(tmp_path, tmp_path / "rec" / "gemma4e4b_1002", fakes)
    assert rc == 0
    unit = out / "unit"
    original = json.loads((unit / "admission.json").read_text())
    leg = unit / "probes" / "lora_native.jsonl"
    stacked = {r["probe_index"]: r for r in (json.loads(l) for l in (unit / "probes" / "lens_stacked.jsonl").read_text().split("\n") if l.strip())}
    # probes 0 and 5 have equal length in the fake (6 + i % 5); move probe 0's divergence to probe 5
    # in BOTH native legs (the repeat leg equals the native leg), so every pair's aggregates are preserved
    for name in ("lora_native", "lora_native_rep"):
        path = unit / "probes" / f"{name}.jsonl"
        rows = [json.loads(l) for l in path.read_text().split("\n") if l.strip()]
        assert len(rows[0]["token_ids"]) == len(rows[5]["token_ids"]) and rows[0]["token_ids"] != stacked[0]["token_ids"]
        assert rows[5]["token_ids"] == stacked[5]["token_ids"]
        rows[0]["token_ids"] = list(stacked[0]["token_ids"])
        rows[5]["token_ids"] = list(stacked[5]["token_ids"])
        rows[5]["token_ids"][5] = 888
        path.write_text("".join(json.dumps(r, sort_keys=True, default=str) + "\n" for r in rows))
    # every aggregate the old readmit pinned is preserved by construction
    tampered = adm.admission_verdict(adm.admission_inputs_from_unit(
        unit, gate_canary=original["gate_canary"], controls={"all_passed": True, "passed": 5, "of": 5, "verdicts": {}},
        head_to_head=original["head_to_head"], n_probes=43, max_tokens=128, attempt_id=original["attempt_id"],
        geometry=None))
    for pair in original["agreement"]:
        assert {k: tampered["agreement"][pair][k] for k in ("n", "sum_agree_prefix", "sum_denom", "n_diverging", "n_empty")} == \
            {k: original["agreement"][pair][k] for k in ("n", "sum_agree_prefix", "sum_denom", "n_diverging", "n_empty")}
    assert tampered["first_token_ulps"]["native_vs_stacked"]["max"] == original["first_token_ulps"]["native_vs_stacked"]["max"]
    # ...and on this fake the moved-to prompt is exchangeable with the original (same length, neither
    # is the floor's diverging prompt), so the point Δ is identical and the paired band moves only by
    # resampling noise: NO check on the derived numbers can be trusted to notice the tamper — only the
    # bytes can (the review's re-allocation landed on the floor's own prompts and flipped a verdict;
    # either way the marginal checks were blind to it)
    o, tt = original["numerics"]["prompt_resampling_sensitivity"], tampered["numerics"]["prompt_resampling_sensitivity"]
    assert tt["delta_point"] == o["delta_point"] and abs(tt["delta_ci95_high"] - o["delta_ci95_high"]) < 0.005
    res = adm.readmit(out)
    assert res["written"] is None, res
    for name in ("lora_native", "lora_native_rep"):
        assert any(f"probes/{name}.jsonl sha256" in q and "sha256_of_jsonl" in q for q in res["problems"]), res["problems"]
        assert any(f"probes/{name}.jsonl: sha256" in q and "inventoried" in q for q in res["problems"])
    assert not (unit / adm.READMITTED_ADMISSION_FILE).exists()
    # forge the DONE.json shas as well: the inventory still names the files
    for name in ("lora_native", "lora_native_rep"):
        done_p = unit / "probes" / f"{name}.DONE.json"
        d = json.loads(done_p.read_text())
        d["sha256_of_jsonl"] = hashlib.sha256((unit / "probes" / f"{name}.jsonl").read_bytes()).hexdigest()
        done_p.write_text(json.dumps(d))
    res = adm.readmit(out)
    assert res["written"] is None and not any("sha256_of_jsonl" in q for q in res["problems"])
    assert any("probes/lora_native.jsonl: sha256" in q for q in res["problems"])
    assert any("probes/lora_native.DONE.json: sha256" in q for q in res["problems"])
    # forge the inventory too: DONE.json's inventory_sha256 no longer binds it
    inv_p = unit / adm.UNIT_INVENTORY_FILE
    inv = json.loads(inv_p.read_text())
    for rel in ("probes/lora_native.jsonl", "probes/lora_native.DONE.json",
                "probes/lora_native_rep.jsonl", "probes/lora_native_rep.DONE.json"):
        inv["files"][rel]["sha256"] = hashlib.sha256((unit / rel).read_bytes()).hexdigest()
        inv["files"][rel]["size"] = (unit / rel).stat().st_size
    inv_p.write_text(json.dumps(inv))
    res = adm.readmit(out)
    assert res["written"] is None and any("inventory_sha256 does not bind" in q for q in res["problems"]), res["problems"]
    assert not (unit / adm.READMITTED_ADMISSION_FILE).exists()
    empty = tmp_path / "rec" / "gemma4e4b_1003"
    empty.mkdir(parents=True)
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        assert adm.readmit(empty)["written"] is None and adm.main(["readmit", str(empty)]) == adm.RC_READMIT_REFUSED


def test_readmit_provenance_binds_the_module_to_a_clean_tree(tmp_path, monkeypatch):
    """A DIRTY tree refuses to record a tree sha (the sha would not
    name the code that ran) unless ``allow_dirty``, and then the file and the
    sidecar are MARKED; a tree claimed clean whose committed module does not
    hash to the module on disk refuses; a skipped bind (no git) is recorded,
    not a refusal; ``--tree-sha`` binds against the named tree. The report
    renders ``rule (module <sha12> @ tree <sha8>[, dirty])``."""
    fakes = Fakes(native_shift=None, per_row_shift=(1, 2), ratio=1.4)
    rc, out = _run(tmp_path, tmp_path / "rec" / "gemma4e4b_1005", fakes)
    assert rc == 0
    unit = out / "unit"
    module = hashlib.sha256(Path(adm.__file__).read_bytes()).hexdigest()
    real = adm._git_tree

    def fake_tree(module_path, *, claimed_sha=None, dirty=True, matches=True, skipped=None):
        sha = claimed_sha or "a" * 40
        at_tree = module if matches else "0" * 64
        if skipped:
            return {"sha": claimed_sha, "dirty": None, "module_sha256": module, "module_sha256_at_tree": None,
                    "module_matches_tree": None, "bind_skipped": skipped}
        return {"sha": sha, "dirty": dirty, "module_sha256": module, "module_sha256_at_tree": at_tree,
                "module_matches_tree": matches, "bind_skipped": None}

    monkeypatch.setattr(adm, "_git_tree", lambda mp, claimed_sha=None: fake_tree(mp, claimed_sha=claimed_sha, dirty=True))
    res = adm.readmit(out)
    assert res["written"] is None and any("DIRTY" in q for q in res["problems"]) and res["tree"]["dirty"] is True
    assert not (unit / adm.READMITTED_ADMISSION_FILE).exists() and not (out / adm.RESULT_SIDECAR).exists()
    res = adm.readmit(out, allow_dirty=True)
    assert res["written"]
    prov = json.loads((unit / adm.READMITTED_ADMISSION_FILE).read_text())["readmitted_from"]
    assert prov["tree_dirty"] is True and prov["provenance_note"].startswith("DIRTY TREE")
    assert "dirty=true" in (out / adm.RESULT_SIDECAR).read_text()
    r = rep.TargetRecord(out)
    assert r.rule == f"{adm.ADMISSION_RULE} readmitted (module {module[:12]} @ tree aaaaaaaa, dirty)" and r.problems == []
    # clean tree, module does not match the committed one -> refuse
    monkeypatch.setattr(adm, "_git_tree", lambda mp, claimed_sha=None: fake_tree(mp, claimed_sha=claimed_sha, dirty=False, matches=False))
    res = adm.readmit(out)
    assert res["written"] is None and any("not the code that ran" in q for q in res["problems"])
    # clean and matching: recorded, unmarked; the report renders module @ tree
    monkeypatch.setattr(adm, "_git_tree", lambda mp, claimed_sha=None: fake_tree(mp, claimed_sha=claimed_sha, dirty=False))
    res = adm.readmit(out, tree_sha="b" * 40)
    prov = json.loads((unit / adm.READMITTED_ADMISSION_FILE).read_text())["readmitted_from"]
    assert res["written"] and prov["tree_sha_of_readmit"] == "b" * 40 and prov["tree_dirty"] is False
    assert prov["module_matches_tree"] is True and prov["provenance_note"] is None
    assert rep.TargetRecord(out).rule == f"{adm.ADMISSION_RULE} readmitted (module {module[:12]} @ tree bbbbbbbb)"
    assert (out / adm.RESULT_SIDECAR).read_text().split()[-2] == "tree=bbbbbbbb"
    # git unavailable: the bind is skipped and recorded, the file written
    monkeypatch.setattr(adm, "_git_tree", lambda mp, claimed_sha=None: fake_tree(mp, claimed_sha=claimed_sha, skipped="no git"))
    res = adm.readmit(out)
    prov = json.loads((unit / adm.READMITTED_ADMISSION_FILE).read_text())["readmitted_from"]
    assert res["written"] and prov["tree_bind_skipped"] == "no git" and prov["tree_sha_of_readmit"] is None
    assert rep.TargetRecord(out).rule == f"{adm.ADMISSION_RULE} readmitted (module {module[:12]} @ tree unknown)"
    monkeypatch.setattr(adm, "_git_tree", real)
    # a stale sidecar (another verdict) is a named problem and never quoted
    (out / adm.RESULT_SIDECAR).write_text(f"rule={adm.ADMISSION_RULE} verdict=not_admitted readiness=attested controls=4/5 "
                                          "gate=passed not_slower=True numerics=diagnostic outcome=admitted zero=exact "
                                          "original_verdict=x module=y tree=z\n")
    r = rep.TargetRecord(out)
    assert r.result_line == r.result and any(adm.RESULT_SIDECAR in q for q in r.problems)


def test_a_pulled_record_reads_clean_logs_absent_gate_record_remapped(tmp_path, monkeypatch):
    """Shaped like the real pulled records: the record dir is COPIED
    elsewhere (so every absolute path its documents name — the admission's
    ``gate_canary.record`` — points at the original, now-absent
    ``<out>/unit/``) and the inventoried server logs are dropped. The report
    reads it with no problem: the absent logs are ``absent_diagnostic_logs``
    (reported, never a pass), the gate record resolves under the local
    ``unit/``, and ``readmit`` re-evaluates it (the logs named as absent, the
    sidecar written). A missing NON-log inventoried file is still a problem
    for both."""
    import shutil
    _clean_tree(monkeypatch)
    fakes = Fakes(native_shift=None, per_row_shift=(1, 2), ratio=1.4)
    rc, out = _run(tmp_path, tmp_path / "cluster" / "gemma4e4b_1006", fakes)
    assert rc == 0
    unit = out / "unit"
    logs = sorted(p.relative_to(unit).as_posix() for p in unit.rglob("*.log"))
    assert logs, "the fake unit writes at least one server/gate log"
    inv = json.loads((unit / adm.UNIT_INVENTORY_FILE).read_text())
    assert all(l in inv["files"] for l in logs)
    rec_abs = json.loads((unit / "admission.json").read_text())["gate_canary"]["record"]
    assert rec_abs.startswith(str(unit))                  # the ORIGINAL absolute path
    pulled = tmp_path / "pulled" / "gemma4e4b_1006"
    shutil.copytree(out, pulled)
    for l in logs:
        (pulled / "unit" / l).unlink()
    shutil.rmtree(out)                                     # the original is gone, as on this machine
    r = rep.TargetRecord(pulled)
    assert r.problems == [], r.problems
    assert r.absent_diagnostic_logs == logs and r.complete
    assert r.gate_record_path == pulled / "unit" / "gate_canary" / "gate_record.json" and r.gate_record_path.is_file()
    assert adm.unit_inventory_problems(pulled / "unit", allow_absent_logs=True) == []
    assert any("inventoried file missing" in q for q in adm.unit_inventory_problems(pulled / "unit"))   # the strict default
    res = adm.readmit(pulled)
    assert res["written"] and res["inputs_verified"]["absent_diagnostic_logs"] == logs
    assert res["inputs_verified"]["inventory_verified"] == res["inputs_verified"]["inventory_listed"] - len(logs)
    assert (pulled / adm.RESULT_SIDECAR).is_file()
    r = rep.TargetRecord(pulled)
    assert r.problems == [] and r.readmitted is not None and r.result_line == (pulled / adm.RESULT_SIDECAR).read_text().strip()
    md = rep.table([r])
    assert f"absent_diagnostic_logs={logs}" in md and f"RESULT='rule={adm.ADMISSION_RULE}" in md
    # a NON-log inventoried file missing is a problem for the report and a refusal for the readmit
    (pulled / "unit" / "throughput" / "lens_per_row.json").unlink()
    assert any("inventoried file missing: throughput/lens_per_row.json" in q for q in rep.TargetRecord(pulled).problems)
    assert adm.readmit(pulled)["written"] is None


def test_readmit_supersedes_an_older_rule_and_never_a_newer_or_unknown_one(tmp_path, monkeypatch):
    """An existing ``admission.readmitted.json`` under an OLDER rule of
    :data:`RULE_ORDER` (D102, or the D111 sidecar the three historical records
    carry) is overwritten and named under ``readmitted_from.superseded``; one
    under a NEWER rule is never overwritten; one under an UNKNOWN rule
    (``D999``, ``not-a-rule`` — the earlier ``D<n>`` parse would have read
    ``D999`` as newer) refuses by name. An absent server log is reported under
    ``absent_diagnostic_logs`` — and never counts as verified — while an absent
    numerical input refuses."""
    _clean_tree(monkeypatch)
    fakes = Fakes(native_shift=None, per_row_shift=(1, 2), ratio=1.4)
    rc, out = _run(tmp_path, tmp_path / "rec" / "gemma4e4b_1004", fakes)
    assert rc == 0
    unit = out / "unit"
    target = unit / adm.READMITTED_ADMISSION_FILE
    assert [adm.rule_rank(r) for r in ("D102", "D111", adm.ADMISSION_RULE)] == [0, 1, 2]
    assert adm.rule_rank("D999") is None and adm.rule_rank("not-a-rule") is None and adm.rule_rank(None) is None
    for older in ("D102", "D111"):
        target.write_text(json.dumps({"rule": older, "verdict": "not_admitted", "readmitted_from": {"rule": older}}))
        res = adm.readmit(out)
        assert res["written"] and res["superseded"] == older, older
        doc = json.loads(target.read_text())
        assert doc["readmitted_from"]["superseded"] == older and doc["rule"] == adm.ADMISSION_RULE and doc["verdict"] == "admitted"
    # the same rule: regenerated, nothing superseded
    res = adm.readmit(out)
    assert res["written"] and res["superseded"] is None
    # a NEWER rule (one this module does not know yet, appended to the order): never overwritten
    monkeypatch.setattr(adm, "RULE_ORDER", adm.RULE_ORDER + ("functional-2099-01-01",))
    target.write_text(json.dumps({"rule": "functional-2099-01-01", "verdict": "admitted",
                                  "readmitted_from": {"rule": "functional-2099-01-01"}}))
    res = adm.readmit(out)
    assert res["written"] is None and any("NEWER" in q for q in res["problems"])
    assert json.loads(target.read_text())["rule"] == "functional-2099-01-01"
    monkeypatch.setattr(adm, "RULE_ORDER", ("D102", "D111", adm.ADMISSION_RULE))
    # an UNKNOWN rule: refused by name, never overwritten (D999 is not "newer"; it is unknown)
    for unknown in ("D999", "not-a-rule"):
        target.write_text(json.dumps({"rule": unknown, "verdict": "admitted", "readmitted_from": {"rule": unknown}}))
        res = adm.readmit(out)
        assert res["written"] is None and any("unknown rule" in q and unknown in q for q in res["problems"]), (unknown, res["problems"])
        assert json.loads(target.read_text())["rule"] == unknown
    target.unlink()
    # an inventoried server log absent: reported, not a pass; an inventoried numerical input absent: refusal
    (unit / "lora_native" / "vllm_server.log").write_text("synthetic log\n")
    inv = adm.build_unit_inventory(unit, attempt_id=json.loads((unit / "DONE.json").read_text())["attempt_id"])
    inv_bytes = json.dumps(inv, indent=2, sort_keys=True).encode()
    (unit / adm.UNIT_INVENTORY_FILE).write_bytes(inv_bytes)
    done = json.loads((unit / "DONE.json").read_text())
    done[adm.DONE_INVENTORY_SHA_KEY] = hashlib.sha256(inv_bytes).hexdigest()
    (unit / "DONE.json").write_text(json.dumps(done, indent=2, sort_keys=True))
    (unit / "lora_native" / "vllm_server.log").unlink()
    res = adm.readmit(out)
    assert res["written"] and res["inputs_verified"]["absent_diagnostic_logs"] == ["lora_native/vllm_server.log"]
    assert res["inputs_verified"]["inventory_verified"] == res["inputs_verified"]["inventory_listed"] - 1
    (unit / "throughput" / "lens_per_row.json").rename(unit / "throughput" / "lens_per_row.json.moved")
    res = adm.readmit(out)
    assert res["written"] is None and any("inventoried input missing: throughput/lens_per_row.json" in q
                                          for q in res["problems"])


#: the table's header cells (native-serving-default §1.4): NO column reads as a numerics verdict
TABLE_HEADER_CELLS = ["target", "rule", "verdict", "readiness", "controls", "gate canary",
                      "not slower at S (ratio; B=32, B=8)",
                      "diagnostics: native vs stacked Δ [90% band] · primary floor · zero concurrent n_div/n · "
                      "solo certificate N × T · first-token ulps", "provenance"]


def test_report_prefers_a_bound_readmit_and_renders_the_functional_verdict(tmp_path, monkeypatch):
    """With ``admission.readmitted.json`` beside a record's admission, the
    report reads IT (``rule`` = ``functional-… readmitted (module … @ tree …)``),
    renders the verdict with the four gates as columns and ONE diagnostics
    column (Δ vs the primary with its band, the primary floor, the zero
    adapter's concurrent n_div/43, the solo certificate's N × T, the ulps) —
    no column reads as a numerics verdict — and both figures. A record
    WITHOUT a readmit reads ``… in-job``. A record whose greedy comparison
    D111 read INCONCLUSIVE (one native prompt shifted) is ADMITTED with the
    outcome recorded in its detail line. A readmit whose recorded original sha
    is not the admission on disk is STALE: named under problems, never
    preferred. A reused record is readmitted at its own copy."""
    pytest.importorskip("matplotlib")
    _clean_tree(monkeypatch)
    F = adm.ADMISSION_RULE
    records = tmp_path / "records"
    rc, a = _run(tmp_path / "w1", records / "gemma4e4b_9101", Fakes(native_shift=None, per_row_shift=(1, 2), ratio=1.6))
    assert rc == 0
    rc, b = _run(tmp_path / "w2", records / "nemotron3nano4b_9102",
                 Fakes(native_shift=None, per_row_shift=None, ratio=1.1, gate="failed"), key="nemotron3nano4b")
    assert rc == 0
    rc, q = _run(tmp_path / "w3", records / "qwen35_4b_9103", Fakes(native_shift=(0, 5), per_row_shift=(1, 2), ratio=2.0),
                 key="qwen35_4b")
    assert rc == 0
    assert adm.readmit(a)["written"]
    out = rep.build(records, tmp_path / "figs")
    recs = {r.target: r for r in out["records"]}
    g, n, qw = recs["gemma4e4b"], recs["nemotron3nano4b"], recs["qwen35_4b"]
    assert g.readmitted is not None and g.rule.startswith(f"{F} readmitted (module ") and " @ tree " in g.rule
    assert g.rule_short == "readmitted" and ", dirty" not in g.rule and g.rule_name == F and g.is_functional
    assert g.verdict == "admitted" == g.in_job_verdict and g.in_job_rule == F and g.admission is g.readmitted
    assert g.problems == [] and g.outcome == "admitted" and g.primary_point_inequality is True
    assert g.gate_scope.startswith("6 frozen prompts, 64 tokens, first fork, 0.05 nats; completed")
    assert g.readiness_txt == "attested" and g.controls_txt == "6/6"
    assert g.gate_canary_txt == "passed (6 frozen prompts, 64 tokens, first fork, 0.05 nats; completed)"
    assert g.not_slower_txt == "yes (1.60x @ S=121; B=32 1.60x, B=8 1.60x)"
    assert g.zero_verdict == "exact (solo certificate)"
    assert n.readmitted is None and n.rule == f"{F} in-job" and n.verdict == "not_admitted" == n.in_job_verdict
    assert n.admission is n.admission_in_job and n.outcome == "degenerate_gate_canary"
    assert n.admission["reasons"] == ["gate_canary_failed"] and n.gate_canary_txt.startswith("FAILED (")
    assert n.primary_point_inequality is True         # native == stacked == per_row: 0 <= 0
    # the greedy comparison D111 called INCONCLUSIVE is a recorded diagnostic: the standup is admitted
    assert qw.verdict == "admitted" == qw.in_job_verdict and qw.outcome == "inconclusive"
    assert qw.admission["reasons"] == [] and qw.admission["numerics"]["reasons"] == ["native_disagreement_inconclusive"]
    assert qw.zero_verdict == "exact (solo certificate)"
    # an INCOMPLETE record (a failed standup: manifest, no unit) is a table row, never in the figures
    failed = records / "qwen35_4b_9104"
    failed.mkdir()
    (failed / "manifest.json").write_text(json.dumps({"target": "qwen35_4b", "slurm_job_id": "9104", "attempt_id": "f" * 32}))
    (failed / "RESULT").write_text("state=FAILED exit=4 elapsed=1 host=TESTHOST target=qwen35_4b verdict=none\n")
    out = rep.build(records, tmp_path / "figs", all_dirs=True)
    names = [r.path.name for r in out["records"]]
    assert "qwen35_4b_9104" in names and "qwen35_4b_9103" in names
    assert [r.path.name for r in out["figure_records"]] == ["gemma4e4b_9101", "nemotron3nano4b_9102", "qwen35_4b_9103"]
    incomplete = next(r for r in out["records"] if r.path.name == "qwen35_4b_9104")
    assert not incomplete.complete and incomplete.verdict == "incomplete" and incomplete.zero_verdict == "—"
    assert incomplete.readiness_txt == incomplete.controls_txt == incomplete.diagnostics_txt == "—"
    md = out["table"]
    # THE HEADER: the verdict, its rule, the four gates, one diagnostics column — nothing that reads as a numerics verdict
    header = [c.strip() for c in md.split("\n")[0].strip().strip("|").split("|")]
    assert header == TABLE_HEADER_CELLS
    for banned in ("in-job verdict", "re-evaluated", "outcome", "numerics", "zero verdict", "three-way", "primary comparison"):
        assert not any(banned in h for h in header), (banned, header)
    assert "| qwen35_4b | — | **incomplete** | — | — | — | — | — |" in md
    assert md.count("**admitted**") == 2 and md.count("**not_admitted**") == 1 and "**inconclusive**" not in md
    assert f"{F} readmitted (module " in md and " @ tree " in md and f"{F} in-job" in md
    assert "primary floor 0.0000 (n_div 0) DEGENERATE -> gate_canary_neartie_oracle" in md
    assert "| FAILED (6 frozen prompts, 64 tokens, first fork, ? nats; failed) |" in md   # a failed canary leaves no evidence file
    assert "solo certificate yes (? × 48)" in md      # the fake certificate records no n_prompts; the cap is the driver's
    assert "zero concurrent exact (0/43 diverge" in md and "passed (6 frozen prompts, 64 tokens, first fork, 0.05 nats; completed)" in md
    # the rule, the reversal history, the band's coverage limitation and the planning qualification are RENDERED
    assert rep.FUNCTIONAL_NOTE in md and "Admission is FUNCTIONAL" in rep.FUNCTIONAL_NOTE
    assert "reversed by the analysis review (D111)" in md and rep.D102_NOTE in md
    assert adm.BOOTSTRAP_COVERAGE_NOTE in md and adm.PLANNING_NOTE in md
    assert "nominal coverage is unvalidated" in adm.BOOTSTRAP_COVERAGE_NOTE
    assert "80% power to DETECT a 5-pp excess" in adm.PLANNING_NOTE and "zero-adapter" in adm.PLANNING_NOTE
    assert "planning (80% power to DETECT an excess of m" in md and "n(5 pp)=" in md and "not noninferiority" in md
    # the detail lines carry the recorded outcome and the numerics' findings AS diagnostics
    assert "| DIAGNOSTIC numerics (gate=false): outcome=inconclusive numerics.reasons=['native_disagreement_inconclusive']" in md
    assert "gates={readiness_attested=yes, controls_all_passed=yes, gate_canary_passed=yes, not_slower_at_S=yes}" in md
    foot = " ".join(rep.figure_footnote().split())            # textwrap.fill wraps inside the phrases
    assert "Admission is FUNCTIONAL" in foot and "reversed by the analysis review (D111)" in foot
    assert "nominal coverage is unvalidated" in foot
    assert foot.count("Bands measure sensitivity") == 1 and rep.figure_footnote().count("\n") >= 2
    assert "coverage unvalidated" in Path(rep.__file__).read_text()          # the legend label carries it too
    assert "Greedy-token disagreement (diagnostic)" in Path(rep.__file__).read_text()   # the figure's title
    assert qw.admission["numerics"]["prompt_resampling_sensitivity"]["planning"]["reads"] == adm.PLANNING_NOTE
    row = next(l for l in md.split("\n") if l.startswith("| qwen35_4b |") and "**admitted**" in l)
    cells = [c.strip() for c in row.strip().strip("|").split("|")]
    assert cells[1] == f"{F} in-job" and cells[2] == "**admitted**" and cells[3] == "attested" and cells[4] == "6/6"
    assert cells[5].startswith("passed (") and cells[6] == "yes (2.00x @ S=121; B=32 2.00x, B=8 2.00x)"
    assert cells[7].startswith("native vs stacked Δ ") and " pp [" in cells[7] and "solo certificate yes (? × 48)" in cells[7]
    assert "first-token ulps max" in cells[7]
    assert {p.name for p in out["figures"]} == {"head_to_head_tok_s.png", "admission_disagreement.png"}
    assert all(p.stat().st_size > 10_000 for p in out["figures"])
    # stale: the recorded original sha no longer matches -> not preferred, named
    p = a / "unit" / adm.READMITTED_ADMISSION_FILE
    doc = json.loads(p.read_text())
    doc["readmitted_from"]["sha256_of_original_admission_json"] = "0" * 64
    p.write_text(json.dumps(doc))
    stale = rep.TargetRecord(a)
    assert stale.readmitted is None and stale.rule == f"{F} in-job" and stale.admission is stale.admission_in_job
    assert any("stale" in x for x in stale.readmit_problems) and any("readmit:" in x for x in stale.problems)
    # a reused record (§5) is readmitted at ITS OWN copy, never at the source path
    rc, c = _run(tmp_path / "w1", records / "gemma4e4b_9104", Fakes(native_shift=None, per_row_shift=(1, 2), ratio=1.6),
                 resume=a)
    assert rc == 0 and json.loads((c / "manifest.json").read_text())["unit_reused_from"] == str(a / "unit")
    res = adm.readmit(c)
    assert res["written"] == str(c / adm.REUSED_UNIT_DIR / adm.READMITTED_ADMISSION_FILE), res
    reused = rep.TargetRecord(c)
    assert reused.readmitted is not None and reused.binding_problems == [] and reused.verdict == "admitted"
    assert adm.reused_unit_binding_problems(c, record_attempt_id=reused.attempt_id,
                                            unit_reused_from=reused.reused_from) == []


# --------------------------------------------------------------------------- #
# native-psr-serving step 4: the not_measured numerics mode + multi-cell rows #
# --------------------------------------------------------------------------- #

def test_psr_row_not_measured_end_to_end_runs_no_probe_and_no_per_row_leg(tmp_path):
    """The stage-1 PSR row through the driver (fakes at every seam): TWO fitted
    cells served (the PSR cell + its LoRA twin), the oracle armed for both, the
    standup asked with ``output_diagnostics=False``, the gate with
    ``--no-output-diagnostics``, NO probe leg, the lens STACKED leg + throughput
    kept, the lens PER_ROW leg DROPPED, the admission ``numerics.mode`` /
    ``outcome`` = ``not_measured`` under the functional rule, the inventory
    complete WITHOUT numerics evidence, the RESULT line stating it."""
    fakes = Fakes(ratio=1.4)
    rc, out = _run(tmp_path, tmp_path / "rec" / "gemma4e4b_psr_2001", fakes, key="gemma4e4b_psr")
    assert rc == 0, (out / "FAILED.json").read_text() if (out / "FAILED.json").is_file() else "rc"
    t = mc.TARGETS["gemma4e4b_psr"]
    h_psr, h_lora = (mc.harness_cell_id(c) for c in t.cells)
    unit = out / "unit"
    # --- the standup was asked for BOTH cells (+ the prompted baseline), diagnostics OFF
    kw = fakes.last_native_kw
    assert kw["cell"] == [t.tt_cell, *t.extra_cells, mc.PROMPTED_CELL]
    assert [s.split("=")[0] for s in kw["expect_object_sha256"]] == list(t.cells)
    assert kw["output_diagnostics"] is False
    # --- the oracle was armed for both canaries (objects.json keyed by harness ids, one --expect per cell)
    assert set(fakes.last_oracle_objects) == {h_psr, h_lora}
    ref = json.loads((out / "prep" / "oracle" / "reference.json").read_text())
    assert set(ref["objects"]) == set(ref["arms"]) == {h_psr, h_lora} and len(ref["expect"]) == 2
    # --- the gate ran with the policy flag
    assert "--no-output-diagnostics" in fakes.last_gate_argv
    # --- the lens legs: ONE stacked standup serving both cells, NO per_row
    assert len(fakes.lens_kws) == 1 and fakes.env_at_lens == ["1"]
    assert fakes.lens_kws[0]["cell"] == list(t.cells)
    assert [s.split("=")[0] for s in fakes.lens_kws[0]["expect_object_sha256"]] == list(t.cells)
    assert not (unit / "lens_per_row").exists()
    # --- no probe leg at all; throughput for the two head-to-head legs only
    assert not (unit / "probes").exists()
    assert sorted(p.name for p in (unit / "throughput").glob("*.json")) == [
        "lens_stacked.DONE.json", "lens_stacked.json", "lora_native.DONE.json", "lora_native.json"]
    # --- identity + endpoint carry every cell
    ident = json.loads((unit / "identity.json").read_text())
    assert ident["cells"] == list(t.cells) and ident["harness_cells"] == [h_psr, h_lora]
    assert set(ident["object_artifact_sha256_by_cell"]) == {h_psr, h_lora} and ident["numerics"] == "not_measured"
    assert set(ident["adapter_digests"]) == {h_psr, h_lora, "none__lora0"}
    ep = json.loads((unit / "lora_native_endpoint.json").read_text())
    assert ep["output_diagnostics"] is False and set(ep["object_artifact_sha256_by_cell"]) == {h_psr, h_lora}
    # --- the admission: functional gates decide; the numerics block states not_measured
    a = json.loads((unit / "admission.json").read_text())
    assert adm.validate_admission_record(a) == [], adm.validate_admission_record(a)
    assert a["verdict"] == "admitted" and a["rule"] == adm.ADMISSION_RULE
    assert a["numerics"]["mode"] == "not_measured" == a["numerics"]["outcome"] and a["numerics"]["gate"] is False
    assert a["numerics"]["reasons"] == [] and a["numerics"]["floor_measured"] is False
    assert a["head_to_head"]["not_slower_at_S"] is True and a["zero_adapter"]["exact"] is True
    assert a["zero_adapter"]["instruments"]["concurrent_legs"] == "unmeasured"
    assert all(pair["n"] == 0 for pair in a["agreement"].values())
    # --- the inventory: complete without numerics evidence; DONE bound to it
    inv = json.loads((unit / adm.UNIT_INVENTORY_FILE).read_text())
    assert inv["missing"] == [] and not any(r.startswith("probes/") or "lens_per_row" in r for r in inv["required"])
    assert "lens_stacked/steering_attestation.json" in inv["required"] and "throughput/lens_stacked.json" in inv["required"]
    assert adm.unit_inventory_problems(unit) == []
    assert adm.unit_binding_problems(unit, expected_attempt_id=a["attempt_id"]) == []
    # --- the RESULT line
    line = (out / "RESULT").read_text()
    assert "verdict=admitted" in line and "numerics=not_measured outcome=not_measured" in line
    assert "target=gemma4e4b_psr" in line and "controls=6/6" in line
    # --- the manifest names every cell and the mode
    man = json.loads((out / "manifest.json").read_text())
    assert man["cells"] == list(t.cells) and man["numerics"] == "not_measured"
    # --- the offline re-evaluation reads the record as not_measured too (no leg is "not complete on disk")
    re_eval = adm.reevaluate(out)
    assert re_eval["problems"] == [] and re_eval["new"]["numerics"]["mode"] == "not_measured", re_eval["problems"]
    assert re_eval["new"]["verdict"] == "admitted"
    # the structural control's shape under the policy: no numerics block by design, and one PRESENT refuses
    sel = json.loads((unit / "lora_native" / "positive_controls.json").read_text())["controls"]["selection_honored"] \
        if (unit / "lora_native" / "positive_controls.json").is_file() else None
    off = {"verdict": "routing_attested", "steps": {"n_steps_mixed": 3}, "n_requests": 9, "n_observed": 9,
           "n_missing": 0, "output_diagnostics": False}
    assert adm.selection_control_problems(off) == []
    assert any("two claims" in p for p in adm.selection_control_problems({**off, "numerics": {"gating": False}}))
    assert any("no numerics block" in p for p in adm.selection_control_problems({k: v for k, v in off.items() if k != "output_diagnostics"}))
    del sel


def test_a_not_measured_unit_carrying_probe_evidence_is_refused_by_name(tmp_path):
    """The known-bad, at both seams: an admission that says not_measured beside a
    probe leg on disk is two claims about one unit — the inventory names it, and
    the verdict refuses inputs that hand in legs under that mode."""
    fakes = Fakes(ratio=1.4)
    rc, out = _run(tmp_path, tmp_path / "rec" / "gemma4e4b_psr_2002", fakes, key="gemma4e4b_psr")
    assert rc == 0
    unit = out / "unit"
    assert adm.unit_inventory_problems(unit) == []
    (unit / "probes").mkdir()
    (unit / "probes" / "lora_native.jsonl").write_text('{"probe_index": 0}\n')
    probs = adm.unit_inventory_problems(unit)
    assert any("numerics 'not_measured' but the unit carries numerics evidence" in p for p in probs), probs
    # the verdict: legs handed in under not_measured refuse
    inputs = _inputs()
    with pytest.raises(adm.AdmissionError, match="not_measured"):
        adm.admission_verdict(dataclasses_replace(inputs, numerics_mode=adm.NUMERICS_NOT_MEASURED))
    with pytest.raises(adm.AdmissionError, match="numerics_mode"):
        adm.admission_verdict(dataclasses_replace(inputs, numerics_mode="sometimes"))


def dataclasses_replace(obj, **over):
    import dataclasses
    return dataclasses.replace(obj, **over)


def test_not_measured_verdict_and_validator_vocabulary():
    """A not_measured verdict with NO legs: admitted on the four functional gates,
    outcome not_measured, no leg named missing; the validator holds the literal to
    its outcome (a not_measured record whose outcome says unmeasured, or a
    diagnostic record whose outcome says not_measured, is named); a record written
    before native-psr-serving (no ``mode``) validates as diagnostic."""
    base = _inputs()
    inputs = dataclasses_replace(base, legs={}, numerics_mode=adm.NUMERICS_NOT_MEASURED)
    v = adm.admission_verdict(inputs)
    assert v["verdict"] == "admitted" and v["numerics"]["outcome"] == "not_measured" == v["numerics"]["mode"]
    assert v["numerics"]["reasons"] == [] and v["numerics"]["structural_ok"] is True
    assert v["numerics"]["prompt_resampling_sensitivity"]["skipped"].startswith("not measured by design")
    assert adm.validate_admission_record(v) == []
    assert "not_measured" in adm.OUTCOMES and adm.NUMERICS_MODES == ("diagnostic", "not_measured")
    # the same inputs WITHOUT the mode: every leg is a named finding, outcome unmeasured (attempted and missing)
    d = adm.admission_verdict(dataclasses_replace(base, legs={}))
    assert d["numerics"]["outcome"] == "unmeasured" and d["numerics"]["mode"] == "diagnostic"
    assert any(r.startswith("leg_missing:") for r in d["numerics"]["reasons"])
    assert adm.validate_admission_record(d) == []
    # known-bads of the vocabulary
    lie = json.loads(json.dumps(v)); lie["numerics"]["outcome"] = "unmeasured"
    assert any("mode 'not_measured' but outcome 'unmeasured'" in p for p in adm.validate_admission_record(lie))
    lie2 = json.loads(json.dumps(d)); lie2["numerics"]["outcome"] = "not_measured"
    assert any("outcome 'not_measured' under mode 'diagnostic'" in p for p in adm.validate_admission_record(lie2))
    lie3 = json.loads(json.dumps(v)); lie3["numerics"]["reasons"] = ["leg_missing:lora_native"]
    assert any("names missing legs as findings" in p for p in adm.validate_admission_record(lie3))
    lie4 = json.loads(json.dumps(v)); lie4["numerics"]["mode"] = "sometimes"
    assert any("numerics.mode 'sometimes'" in p for p in adm.validate_admission_record(lie4))
    legacy = json.loads(json.dumps(d)); legacy["numerics"].pop("mode"); legacy["numerics"].pop("mode_note")
    assert adm.validate_admission_record(legacy) == [] and adm.numerics_mode_of(legacy) == "diagnostic"
    # the evidence split: a not_measured record requires the CORE only
    req_nm = adm.unit_required_evidence(v)
    req_d = adm.unit_required_evidence(d)
    assert set(req_nm) < set(req_d)
    assert not any(r.startswith("probes/") or "lens_per_row" in r for r in req_nm)
    assert all(r in req_nm for r in adm.UNIT_EVIDENCE_CORE) and set(adm.UNIT_EVIDENCE_FIXED) == set(adm.UNIT_EVIDENCE_CORE) | set(adm.UNIT_EVIDENCE_NUMERICS)
    assert adm.THROUGHPUT_LEGS_CORE == ("lora_native", "lens_stacked")


def test_oracle_bindings_hold_the_row_s_whole_canary_set():
    """The oracle binding is a SET equality over every harness cell of the row:
    a reference armed for one of two cells, or for a stray cell, refuses by
    name; the legacy single-cell call keeps its shape."""
    want = {"h1": "a" * 64, "h2": "b" * 64}
    ok = {"cell_id_space": "harness", "model": "m", "revision": "r", "chat_template_hash": "t",
          "gate_prompt_set_hash": "g", "gate_max_tokens": 64, "base": {"k": [1]},
          "objects": {h: {"sha256": s} for h, s in want.items()}, "arms": {h: {} for h in want}}
    common = dict(model="m", revision="r", chat_template_hash="t", gate_prompt_set_hash="g", gate_max_tokens=64,
                  hcell="h1", object_sha256="a" * 64)
    assert mc.oracle_bindings_problems(ok, objmap=want, **common) == []
    one = dict(ok, objects={"h1": {"sha256": "a" * 64}}, arms={"h1": {}})
    probs = mc.oracle_bindings_problems(one, objmap=want, **common)
    assert any("objects keys ['h1'] != ['h1', 'h2']" in p for p in probs) and any("arms keys" in p for p in probs)
    stray = dict(ok, objects={**ok["objects"], "h3": {"sha256": "c" * 64}})
    assert any("h3" in p for p in mc.oracle_bindings_problems(stray, objmap=want, **common))
    wrong = json.loads(json.dumps(ok)); wrong["objects"]["h2"]["sha256"] = "d" * 64
    assert any("objects[h2].sha256" in p for p in mc.oracle_bindings_problems(wrong, objmap=want, **common))
    # legacy single-cell shape (no objmap): exactly {hcell}
    assert mc.oracle_bindings_problems(one, **common) == []
    assert mc.oracle_bindings_problems(ok, **common)          # two cells where one was expected
