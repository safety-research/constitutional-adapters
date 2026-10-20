"""Every FITTED GSM8K child of a
native multi-LoRA standup refused its own correctly-bound standup —

    [served-cap] OOS_DECLARATION_MISMATCH: object_artifact_sha256 = None but the
    launcher held the standup to '8be1e1a7…'

— while the ``none`` child (no digest to compare) ran. Root cause, pinned here
against the REAL S1 record bytes (``tests/fixtures/seedrep_f_l/S1``):

* ``contracts.check_object_artifact_digest`` was experiment-1's compare: the
  legacy scalar ``endpoint.json.object_artifact_sha256`` PRESENT had to equal
  the pin (a present ``null`` refused everywhere) and the per-cell map had to be
  EXACTLY ``{harness_key: pin}`` — one fitted cell per standup.
* the view engine serves FIVE fitted cells, so the writer
  (``mt_serving_up`` -> ``capctrl_serving_up.legacy_object_sha256``) writes the
  scalar ``null`` ("several") and a five-entry ``object_artifact_sha256_by_cell``
  keyed by the HARNESS id (``si_constv2_mix_full__lora_r4_ce``) — the id form the
  child already derives through ``cell_spellings(--cell).harness``; the TrainTest
  spelling the child is passed (``constv2-mix-full__lora_r4_ce``) is not a key.

The fixed compare looks the cell up by the harness id in the map, holds it to
``--expect-object-sha256``, and holds the scalar to the writer's derivation
(single distinct digest, else null) instead of to the pin. A tampered digest, a
sibling's pin, a map keyed by another spelling, a torn scalar and a bare null
scalar all still refuse by name. The ``none`` path and the controls children's
posture are unchanged.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import served_capability_run as srun  # noqa: E402

from sisteer.capeval.tinycap import contracts as C  # noqa: E402

FIX = REPO / "tests" / "fixtures" / "seedrep_f_l" / "S1"
MODEL_KEY = "qwen35_4b"
#: the fitted GSM8K child's posture (`lp._gsm8k_children`): --thinking on
#: --allow-out-of-support-thinking --expect-object-sha256 <bare pin>; an all-position
#: native adapter is header_located=False (undeclared), so the native path admits it
GSM8K_POSTURE = dict(arm=C.THINKING_ON, out_of_support=True, header_located=False)
#: the controls children's posture (`lp._controls_children`): thinking OFF, in support
CONTROLS_POSTURE = dict(arm="off", out_of_support=False)


def _load(name: str) -> dict:
    return json.loads((FIX / name).read_text())


@pytest.fixture(scope="module")
def s1():
    """The real S1 records + the plan's per-cell pins (the standup's own
    ``--expect-object-sha256 <TrainTest id>=<pin>`` argv, recorded in
    modal_evidence.json — the launcher composes each child's bare pin from it)."""
    ep, cs, me = _load("endpoint.json"), _load("cells_served.json"), _load("modal_evidence.json")
    argv = me["standup_argv"]
    pins = dict(argv[i + 1].split("=", 1) for i, x in enumerate(argv) if x == "--expect-object-sha256")
    assert len(pins) == 5 and all(C.is_sha256_hex(v) for v in pins.values())
    return ep, cs, pins


def _budget(ep: dict) -> dict:
    return {"chat_template_hash": ep["chat_template_hash"]}


def _harness(tt: str) -> str:
    return C.cell_spellings(tt, declared=False).harness


def _flip(sha: str) -> str:
    return ("0" if sha[0] != "0" else "1") + sha[1:]


def test_real_s1_record_binds_by_harness_id_and_attests_several(s1):
    """The root cause as facts of the real bytes: the scalar the old compare
    read first is ``null``; the map is keyed by HARNESS ids, five of them, so the
    old EXACT ``{harness: pin}`` reading is false for every cell and the
    TrainTest spelling the child is passed names nothing; cells_served.json's
    ``expected_object_sha256_by_cell`` (the pins the standup was TOLD) equals
    the endpoint's map (the artifacts it LOADED)."""
    ep, cs, pins = s1
    assert C.endpoint_is_native(ep)
    assert ep[C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256] is None                      # "several"
    by_cell = ep[C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL]
    assert len(by_cell) == 5 and set(by_cell) == set(ep["lora_cells"])
    assert cs["expected_object_sha256_by_cell"] == by_cell == cs["object_artifact_sha256_by_cell"]
    for tt, pin in pins.items():
        harness = _harness(tt)
        assert by_cell.get(tt) is None, f"the TrainTest spelling {tt!r} must not be a key"
        assert by_cell[harness] == pin                                              # the writer's key form
        assert dict(by_cell) != {harness: pin}                                      # the old exact reading: False


def test_every_fitted_gsm8k_child_passes_the_digest_compare_on_the_real_record(s1):
    """Through contracts AND the child's wrapper (`srun._check_serve_attestation`,
    the exact call `cmd_generate` makes): no refusal, the cell's digest read under
    the harness id, the WHOLE map recorded into the route block."""
    ep, _cs, pins = s1
    for tt, pin in pins.items():
        harness = _harness(tt)
        out = C.check_serve_attestation(ep, cell=tt, model_key=MODEL_KEY, expected_object_sha256=pin,
                                        budget=_budget(ep), **GSM8K_POSTURE)
        assert out["object_digest"] == {"scalar": None, "by_cell": ep[C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL]}
        assert out["object_digest"]["by_cell"][harness] == pin
        assert out["declarations"]["own_key"] == harness
        # the driver's door reaches the same verdict through the same code
        via_child = srun._check_serve_attestation(ep, cell=tt, expected_object_sha256=pin, budget=_budget(ep),
                                                  model_key=MODEL_KEY, **GSM8K_POSTURE)
        assert via_child["object_digest"] == out["object_digest"]
        route = srun._attestation_route_block(via_child)
        assert route[C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL][harness] == pin
        assert route[C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256] is None


def test_direct_compare_required_passes_and_records_the_whole_map(s1):
    ep, _cs, pins = s1
    for tt, pin in pins.items():
        rec = C.check_object_artifact_digest(ep, harness_key=_harness(tt), expected_sha256=pin, required=True)
        assert rec == {"scalar": None, "by_cell": ep[C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL]}


def test_tampered_or_sibling_digest_is_refused_by_name(s1):
    """A real mismatch still refuses: a tampered pin, and a SIBLING cell's pin
    (the cross-cell mix-up a five-cell standup makes possible) — each naming the
    harness key, the digest the standup bound and the one the launcher held it to,
    through the child's wrapper as ``[served-cap] OOS_DECLARATION_MISMATCH``."""
    ep, _cs, pins = s1
    tts = sorted(pins)
    for i, tt in enumerate(tts):
        harness, pin = _harness(tt), pins[tt]
        sibling = pins[tts[(i + 1) % len(tts)]]
        assert sibling != pin
        for wrong in (_flip(pin), sibling):
            with pytest.raises(SystemExit) as e:
                srun._check_serve_attestation(ep, cell=tt, expected_object_sha256=wrong, budget=_budget(ep),
                                              model_key=MODEL_KEY, **GSM8K_POSTURE)
            msg = str(e.value)
            assert msg.startswith(f"[served-cap] {C.REC_OOS_DECLARATION_MISMATCH}")
            assert repr(harness) in msg and pin in msg and wrong in msg
            assert "held the standup to" in msg


def test_a_map_keyed_by_the_traintest_spelling_is_refused_naming_the_harness_id(s1):
    """The namespace fault made explicit: a record whose map is keyed by the
    TrainTest ids names no harness id, so the cell is unbound — refused by name,
    listing the map's keys, never read through a second spelling."""
    ep, _cs, pins = s1
    by_cell = ep[C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL]
    rekeyed = {**ep, C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL: {tt: by_cell[_harness(tt)] for tt in pins}}
    tt = "constv2-mix-full__lora_r4_ce"
    with pytest.raises(C.OutOfSupportAttestationRefusal) as e:
        C.check_serve_attestation(rekeyed, cell=tt, model_key=MODEL_KEY, expected_object_sha256=pins[tt],
                                  budget=_budget(ep), **GSM8K_POSTURE)
    assert e.value.rec == C.REC_OOS_DECLARATION_MISMATCH
    assert f"not {_harness(tt)!r}" in str(e.value) and repr(tt) in str(e.value)


def test_torn_scalar_and_bare_null_scalar_still_refuse(s1):
    """The legacy scalar is held to the writer's derivation, not dropped: a
    non-null scalar beside a several-digest map is torn; a null scalar with NO map
    binds nothing (the pre-F-V refusal, still right when there is no map)."""
    ep, _cs, pins = s1
    tt = "constv2-mix-full__lora_r4_ce"
    torn = {**ep, C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256: pins[tt]}
    with pytest.raises(C.OutOfSupportAttestationRefusal, match="torn record"):
        C.check_object_artifact_digest(torn, harness_key=_harness(tt), expected_sha256=pins[tt])
    no_map = {k: v for k, v in ep.items() if k != C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL}
    assert no_map[C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256] is None
    for required in (True, False):
        with pytest.raises(C.OutOfSupportAttestationRefusal) as e:
            C.check_object_artifact_digest(no_map, harness_key=_harness(tt), expected_sha256=pins[tt],
                                           required=required)
        assert e.value.rec == C.REC_OOS_DECLARATION_MISMATCH and pins[tt] in str(e.value)


def test_none_child_compares_no_digest(s1):
    """The `none` child's argv carries neither --allow-out-of-support-thinking nor
    a pin: the base route is not a fitted cell, so step 2 never runs — unchanged."""
    ep, _cs, _pins = s1
    assert not C.is_lens_cell("none")
    out = C.check_serve_attestation(ep, cell="none", arm=C.THINKING_ON, out_of_support=False,
                                    model_key=MODEL_KEY, expected_object_sha256=None, budget=None)
    assert out["object_digest"] is None and out["declarations"]["own_key"] == "none"


def test_controls_children_posture_passes_on_the_real_record(s1):
    """The controls children (xstest / or_bench_hard_1k / mmlu; thinking OFF, in
    support) go through the same `_check_serve_attestation`: today's composer
    passes no pin (no compare); a pin, if one is added, compares against the real
    record in support (``required=False``, the map still compared) and passes."""
    ep, _cs, pins = s1
    for tt, pin in pins.items():
        without = srun._check_serve_attestation(ep, cell=tt, expected_object_sha256=None, budget=None,
                                                model_key=MODEL_KEY, **CONTROLS_POSTURE)
        assert without["object_digest"] is None
        with_pin = srun._check_serve_attestation(ep, cell=tt, expected_object_sha256=pin, budget=None,
                                                 model_key=MODEL_KEY, **CONTROLS_POSTURE)
        assert with_pin["object_digest"]["by_cell"][_harness(tt)] == pin
    # the prompted cells serve no object: no compare, whatever the posture
    for prompted in ("prompted__si_constitution", "prompted__safety_refusal"):
        out = srun._check_serve_attestation(ep, cell=prompted, expected_object_sha256=None, budget=None,
                                            model_key=MODEL_KEY, **CONTROLS_POSTURE)
        assert out["object_digest"] is None
