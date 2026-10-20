"""The ruled-operating-point gate on the steer admission record (scripts/sdfbase/steer_admission_ruling.py).

Synthetic ladders and a synthetic record throughout (no real session data): four targets, six rungs each, invented shas.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


R = _load("steer_admission_ruling_under_test", "scripts/sdfbase/steer_admission_ruling.py")
OP = _load("operating_points_under_test", "scripts/alphasteer_f22/operating_points.py")


def _sha(*parts) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


def _ladder_root(tmp_path: Path) -> Path:
    root = tmp_path / "ladder_canon"
    for target in OP.RULING["rungs"]:
        rungs = [{"i": i, "lambda": 0.05 * (2 ** (i - 1)) + (0.001 if target == "qwen35_4b" else 0.0),
                  "object_sha256": _sha(target, i)} for i in range(1, 7)]
        body = json.dumps(
            {"rungs": rungs, "chosen": {"lambda": rungs[0]["lambda"], "i": 1}, "outcome": "chosen"})
        (root / target).mkdir(parents=True)
        (root / target / "lambda_ladder.json").write_text(body, encoding="utf-8")
        # THE REAL TWO-SCREEN SHAPE. DECODE_POLICY reads some targets' XSTest half off a SIBLING
        # screen of `ladder_canon` (nemotron: `ladder_canon_topp1`), and `_ladder_root_for` REFUSES
        # when that sibling is absent rather than fall back — falling back would print a top_p 0.95
        # number under a top_p 1.0 label, which is the exact defect the decode ruling removed. This
        # fixture minted only `ladder_canon`, so every oracle call here refused; the FIXTURE was
        # wrong, not the guard, and the guard is deliberately left intact (its refusal is pinned by
        # tests/test_operating_points_overrefusal.py::
        # test_missing_named_screen_is_a_refusal_never_a_silent_fallback).
        # The two screens hold the SAME six objects, byte-identical, measured under different
        # samplers — only the reported over-refusal differs — so the sibling carries the same rungs,
        # which is also what keeps `_bind` below able to read lambda/object_sha256 off either screen.
        sub = OP.LADDER_ROOT_BY_TARGET.get(target)
        if sub:
            (root.parent / sub / target).mkdir(parents=True)
            (root.parent / sub / target / "lambda_ladder.json").write_text(body, encoding="utf-8")
    return root


def _entries(rec: dict) -> dict:
    out = {}
    for target, t in rec["targets"].items():
        view = "main" if target == "nemotron3nano4b" else "it"
        out[(target, "alphasteer")] = {
            "model_key": target, "cell": "alphasteer", "view": view, "rung": t["rung_i"], "lambda": t["lambda"],
            "object_sha256": t["object_sha256"], "selection_outcome": t["selection_outcome"],
            "bank_view_rel": {"primary": f"exp/objects/alphasteer_f22_rebake/{target}/ladder_canon/lam{t['rung_i']}/cells_view/{view}",
                              "overflow": f"exp/objects/alphasteer_f22_rebake/{target}/ladder_canon/lam{t['rung_i']}/cells_view/{view}"}}
        out[(target, "arditi")] = {"model_key": target, "cell": "arditi", "selection_outcome": None, "object_sha256": _sha("ar", target)}
    return out


def test_default_ladder_root_mirrors_the_oracle():
    assert R.DEFAULT_LADDER_ROOT == OP.DEFAULT_LADDER


def test_absent_ladder_root_skips_never_guesses(tmp_path):
    assert R.check_entries({("gemma4e4b", "alphasteer"): {"rung": 99}}, where="x.json",
                           ladder_root=tmp_path / "nowhere", curves=tmp_path / "no_curves.json") is None


def test_binding_record_passes_and_returns_the_oracle_record(tmp_path):
    root = _ladder_root(tmp_path)
    rec = OP.build(root, tmp_path / "no_curves.json")
    got = R.check_entries(_entries(rec), where="synthetic.json", ladder_root=root, curves=tmp_path / "no_curves.json")
    assert got is not None and got["ruling"]["rungs"] == OP.RULING["rungs"]
    # the rung comes from the RULING, lambda + sha from THAT rung's table row — never from the ladder's own `chosen`
    for target, i in OP.RULING["rungs"].items():
        assert got["targets"][target]["rung_i"] == i
        assert got["targets"][target]["object_sha256"] == _sha(target, i)
        assert (got["targets"][target]["selection_outcome"].startswith("ruled_")) is (i != 1)


@pytest.mark.parametrize("field,value", [("rung", 1), ("lambda", 0.123456), ("object_sha256", "0" * 64),
                                         ("selection_outcome", "chosen")])
def test_one_drifted_field_refuses_naming_the_entry(tmp_path, field, value):
    root = _ladder_root(tmp_path)
    rec = OP.build(root, tmp_path / "no_curves.json")
    entries = _entries(rec)
    entries[("gemma4e4b", "alphasteer")][field] = value      # gemma is a MOVED target (ruled rung 3)
    with pytest.raises(SystemExit) as ei:
        R.check_entries(entries, where="synthetic.json", ladder_root=root, curves=tmp_path / "no_curves.json")
    msg = str(ei.value)
    assert f"entries.gemma4e4b.alphasteer.{field}" in msg and "DRIFTS" in msg and "superseded dose" in msg


def test_a_bank_path_bound_to_another_rung_refuses(tmp_path):
    root = _ladder_root(tmp_path)
    rec = OP.build(root, tmp_path / "no_curves.json")
    entries = _entries(rec)
    entries[("qwen35_4b", "alphasteer")]["bank_view_rel"]["overflow"] = "exp/objects/alphasteer_f22_rebake/qwen35_4b/ladder_canon/lam3/cells_view/it"
    with pytest.raises(SystemExit, match=r"entries\.qwen35_4b\.alphasteer\.bank_view_rel\.overflow"):
        R.check_entries(entries, where="synthetic.json", ladder_root=root, curves=tmp_path / "no_curves.json")


def test_a_target_the_ruling_does_not_name_is_a_refusal(tmp_path):
    root = _ladder_root(tmp_path)
    rec = OP.build(root, tmp_path / "no_curves.json")
    entries = _entries(rec)
    entries[("glm46vflash", "alphasteer")] = dict(entries[("gemma4e4b", "alphasteer")], model_key="glm46vflash")
    with pytest.raises(SystemExit, match="names no target 'glm46vflash'"):
        R.check_entries(entries, where="synthetic.json", ladder_root=root, curves=tmp_path / "no_curves.json")


def test_arditi_entries_are_not_the_ladders(tmp_path):
    root = _ladder_root(tmp_path)
    rec = OP.build(root, tmp_path / "no_curves.json")
    entries = {k: v for k, v in _entries(rec).items() if k[1] == "arditi"}
    assert R.check_entries(entries, where="synthetic.json", ladder_root=root, curves=tmp_path / "no_curves.json") is not None


def test_the_committed_record_binds_to_the_ruling_where_the_ladder_is_present():
    """The live guard: on a machine with the grade dir the committed record must bind; elsewhere this skips."""
    if not (R._on_this_host(R.DEFAULT_LADDER_ROOT) or Path("/nonexistent")).is_dir():
        pytest.skip("no ladder record on this machine — the gate is skipped here by design")
    doc = json.loads((REPO / "configs/sdfbase/steer_admission.json").read_text(encoding="utf-8"))
    entries = {(k, c): e for k, by in doc["entries"].items() for c, e in by.items()}
    assert R.check_entries(entries, where="steer_admission.json") is not None


#: gemma's rungs READ FROM THE ORACLE, never typed here. These cases used to hardcode 3 as the
#: ruled rung and 2 as "a candidate". The SECOND ruling of 2026-09-14 moved gemma to 2, which
#: silently inverted every one of them: the ruled-rung refusal stopped firing (3 became an ordinary
#: rung, so the case asserting a refusal got none), and the drift cases kept passing for the WRONG
#: REASON — they were tripping the ruled-rung contradiction rather than the field drift they name,
#: which is a green test that has stopped testing its subject. Deriving all three from
#: ``OP.RULING`` is what stops the next ruling doing this again.
_GEMMA_RULED = OP.RULING["rungs"]["gemma4e4b"]
_GEMMA_CANDIDATE = next(i for i in range(1, 7) if i != _GEMMA_RULED)
_GEMMA_OTHER_RUNG = next(i for i in range(1, 7) if i not in (_GEMMA_RULED, _GEMMA_CANDIDATE))


def _candidate(entries: dict, rec: dict, root: Path, target: str, rung: int) -> dict:
    lad = json.loads((root / target / "lambda_ladder.json").read_text(encoding="utf-8")); row = lad["rungs"][rung - 1]
    e = entries[(target, "alphasteer")]
    e.update({"rung": rung, "lambda": row["lambda"], "object_sha256": row["object_sha256"], "selection_outcome": "candidate_2026-09-14",
              "bank_view_rel": {r: f"exp/objects/alphasteer_f22_rebake/{target}/ladder_canon/lam{rung}/cells_view/{e['view']}" for r in ("primary", "overflow")}})
    return e


def test_a_candidate_bound_to_its_ladder_row_passes(tmp_path):
    root = _ladder_root(tmp_path); rec = OP.build(root, tmp_path / "no_curves.json"); entries = _entries(rec)
    _candidate(entries, rec, root, "gemma4e4b", _GEMMA_CANDIDATE)   # a rung the ruling does NOT name, bound to its own row
    got = R.check_entries(entries, where="synthetic.json", ladder_root=root, curves=tmp_path / "no_curves.json")
    assert got is not None and got["targets"]["gemma4e4b"]["rung_i"] == _GEMMA_RULED   # a candidate never moves the ruling


@pytest.mark.parametrize("field,value", [("lambda", 0.999), ("object_sha256", "f" * 64), ("rung", _GEMMA_OTHER_RUNG)])
def test_a_candidate_that_disagrees_with_its_ladder_row_refuses(tmp_path, field, value):
    root = _ladder_root(tmp_path); rec = OP.build(root, tmp_path / "no_curves.json"); entries = _entries(rec)
    # the candidate names a NON-ruled rung, so the only thing that can refuse is the field drift
    e = _candidate(entries, rec, root, "gemma4e4b", _GEMMA_CANDIDATE); e[field] = value
    with pytest.raises(SystemExit) as ei:
        R.check_entries(entries, where="synthetic.json", ladder_root=root, curves=tmp_path / "no_curves.json")
    assert "entries.gemma4e4b.alphasteer" in str(ei.value) and "DRIFTS" in str(ei.value)


def test_a_candidate_out_of_the_ladder_refuses(tmp_path):
    root = _ladder_root(tmp_path); rec = OP.build(root, tmp_path / "no_curves.json"); entries = _entries(rec)
    e = _candidate(entries, rec, root, "gemma4e4b", _GEMMA_CANDIDATE); e["rung"] = 9
    with pytest.raises(SystemExit, match=r"entries\.gemma4e4b\.alphasteer\.rung"):
        R.check_entries(entries, where="synthetic.json", ladder_root=root, curves=tmp_path / "no_curves.json")


def test_a_candidate_naming_the_ruled_rung_itself_refuses(tmp_path):
    root = _ladder_root(tmp_path); rec = OP.build(root, tmp_path / "no_curves.json"); entries = _entries(rec)
    _candidate(entries, rec, root, "gemma4e4b", _GEMMA_RULED)   # THE ruled rung: a ruled point is not a candidate
    with pytest.raises(SystemExit, match="a ruled point is not a candidate"):
        R.check_entries(entries, where="synthetic.json", ladder_root=root, curves=tmp_path / "no_curves.json")


# --------------------------------------------------------------------------- #
# the KNOWN-BAD input this record was re-stamped from (2026-09-15): an interim gemma entry labelled `candidate_2026-09-14`
# at rung 2 — a bound candidate under the FIRST ruling (lam3), a contradiction under the SECOND (lam2 is the ruled rung). The corrected
# oracle spells gemma's word `ruled_2026-09-14_second`; the gate must refuse the candidate label at the ruled rung BY NAME, and
# a ruled entry carrying the plain `ruled_2026-09-14` word at that rung must refuse too (the word is part of the pin).
# --------------------------------------------------------------------------- #

def test_the_oracle_spells_gemmas_second_ruling_and_the_gate_binds_the_word(tmp_path):
    root = _ladder_root(tmp_path); rec = OP.build(root, tmp_path / "no_curves.json")
    assert OP.RULING["rungs"]["gemma4e4b"] == 2 and "gemma_second_ruling_2026_09_14" in OP.RULING
    assert rec["targets"]["gemma4e4b"]["selection_outcome"] == "ruled_2026-09-14_second"
    # the `_second` spelling is gemma's alone; every other target carries the plain ruled word where it moved off the synthetic
    # ladder's `chosen` (rung 1 in this fixture) and the confirmed word where it did not — never `_second`
    for t, v in rec["targets"].items():
        if t != "gemma4e4b":
            assert v["selection_outcome"] == ("ruled_2026-09-14" if OP.RULING["rungs"][t] != 1 else "chosen_confirmed_2026-09-14"), (t, v["selection_outcome"])
    entries = _entries(rec)
    entries[("gemma4e4b", "alphasteer")]["selection_outcome"] = "ruled_2026-09-14"      # the day's FIRST word at the SECOND ruling's rung
    with pytest.raises(SystemExit, match=r"entries\.gemma4e4b\.alphasteer\.selection_outcome: record='ruled_2026-09-14' ruled='ruled_2026-09-14_second'"):
        R.check_entries(entries, where="synthetic.json", ladder_root=root, curves=tmp_path / "no_curves.json")


def test_a_candidate_label_at_the_ruled_rung_refuses_by_name_under_the_corrected_oracle(tmp_path):
    """The exact shape of the record this commit re-stamped: rung / lambda / sha / bank paths bound to the ladder's row 2 (a candidate the
    gate ADMITTED under the first ruling), labelled `candidate_2026-09-14` — refused, naming the entry and the field, now that 2 is ruled."""
    root = _ladder_root(tmp_path); rec = OP.build(root, tmp_path / "no_curves.json"); entries = _entries(rec)
    e = _candidate(entries, rec, root, "gemma4e4b", OP.RULING["rungs"]["gemma4e4b"])
    assert R.is_candidate(e["selection_outcome"]) and R.entry_problems("gemma4e4b", "alphasteer", e, rec["targets"]["gemma4e4b"],
                                                                        json.loads((root / "gemma4e4b/lambda_ladder.json").read_text())) == [
        "entries.gemma4e4b.alphasteer.selection_outcome: 'candidate_2026-09-14' names the RULED rung 2 itself — a ruled point is not a candidate"]
    with pytest.raises(SystemExit) as ei:
        R.check_entries(entries, where="steer_admission.json", ladder_root=root, curves=tmp_path / "no_curves.json")
    msg = str(ei.value)
    assert "entries.gemma4e4b.alphasteer.selection_outcome" in msg and "names the RULED rung 2 itself" in msg and "DRIFTS" in msg


def test_the_committed_gemma_entry_relabelled_candidate_refuses_where_the_ladder_is_present():
    """The live known-bad: the COMMITTED record with gemma's word put back to the interim candidate label (everything else as committed,
    bound to the real ladder's row 2) must refuse by name on the workstation; the committed word itself is the oracle's `_second`."""
    if not (R._on_this_host(R.DEFAULT_LADDER_ROOT) or Path("/nonexistent")).is_dir():
        pytest.skip("no ladder record on this machine — the gate is skipped here by design")
    doc = json.loads((REPO / "configs/sdfbase/steer_admission.json").read_text(encoding="utf-8"))
    entries = {(k, c): e for k, by in doc["entries"].items() for c, e in by.items()}
    gem = entries[("gemma4e4b", "alphasteer")]
    assert gem["selection_outcome"] == "ruled_2026-09-14_second" and gem["rung"] == 2 and "ruled_disqualified_pending_user" not in gem
    assert gem["superseded"]["rung"] == 3 and gem["superseded"]["superseded_on"] == "2026-09-14" and [h["rung"] for h in gem["superseded_earlier"]] == [1]
    assert doc["ruling"]["rungs"] == OP.RULING["rungs"]
    bad = {k: (dict(v, selection_outcome="candidate_2026-09-14") if k == ("gemma4e4b", "alphasteer") else v) for k, v in entries.items()}
    with pytest.raises(SystemExit) as ei:
        R.check_entries(bad, where="steer_admission.json")
    assert "entries.gemma4e4b.alphasteer.selection_outcome: 'candidate_2026-09-14' names the RULED rung 2 itself" in str(ei.value)


# --------------------------------------------------------------------------- #
# the Arditi entries: null, or EXACTLY the ruled word bound to `ruling` + `superseded`, the view spelled from the alpha, the battery
# --------------------------------------------------------------------------- #

ARDITI_SHA_NEW, ARDITI_SHA_OLD = _sha("ar", "llama8b", 0.05), _sha("ar", "llama8b", 0.1)


def _battery(tmp_path: Path, alphas=(0.05, 0.1, 0.2, 0.4, -0.1)) -> Path:
    """A synthetic battery record in the real one's shape: targets.<key>.cells.<view>.alpha (no object shas, as the real one)."""
    p = tmp_path / "battery_curves.json"
    p.write_text(json.dumps({"budget_pp": 20.0, "targets": {k: {"cells": {R.arditi_view(a): {"alpha": a} for a in alphas}}
                                                          for k in ("gemma4e4b", "qwen35_4b", "nemotron3nano4b", "llama8b")}}), encoding="utf-8")
    return p


def _ruled_arditi(key: str = "llama8b") -> dict:
    rel = f"exp/objects/arditi_strength/{key}/strength/cells_view/ar_r0p05"
    return {"model_key": key, "cell": "arditi", "alpha": 0.05, "view": "ar_r0p05", "staged_id": "ar_r0p05__arditi", "object_sha256": ARDITI_SHA_NEW,
            "bank_view_rel": {"primary": rel, "overflow": rel}, "selection_outcome": R.ARDITI_RULED_OUTCOME,
            "ruling": {"ruled_on": "2026-09-14", "alpha": 0.05, "view": "ar_r0p05", "object_sha256": ARDITI_SHA_NEW,
                       "supersedes": {"alpha": 0.1, "view": "ar_r0p1", "object_sha256": ARDITI_SHA_OLD}},
            "superseded": {"alpha": 0.1, "view": "ar_r0p1", "object_sha256": ARDITI_SHA_OLD, "superseded_on": "2026-09-14",
                           "superseded_by": "a synthetic ruling"}}


def test_arditi_view_is_the_alpha_spelled():
    assert [R.arditi_view(a) for a in (0.05, 0.1, 0.2, 0.4, -0.1)] == ["ar_r0p05", "ar_r0p1", "ar_r0p2", "ar_r0p4", "ar_rm0p1"]


def test_a_ruled_arditi_entry_that_binds_passes_with_and_without_the_battery(tmp_path):
    root = _ladder_root(tmp_path); rec = OP.build(root, tmp_path / "no_curves.json"); entries = _entries(rec)
    entries[("llama8b", "arditi")] = _ruled_arditi()
    kw = dict(where="synthetic.json", ladder_root=root, curves=tmp_path / "no_curves.json")
    assert R.check_entries(entries, arditi_record=_battery(tmp_path), **kw) is not None
    assert R.check_entries(entries, arditi_record=tmp_path / "no_battery.json", **kw) is not None      # the battery half skips, never guesses
    # the AlphaSteer half skipped (no ladder) still runs the Arditi half
    assert R.check_entries({("llama8b", "arditi"): _ruled_arditi()}, where="x.json", ladder_root=tmp_path / "nowhere",
                           curves=tmp_path / "no_curves.json", arditi_record=_battery(tmp_path)) is None


@pytest.mark.parametrize("mutate,needle", [
    (lambda e: e.update(selection_outcome="chosen"), "arditi.selection_outcome"),                       # a ladder word
    (lambda e: e.update(selection_outcome="candidate_2026-09-14"), "arditi.selection_outcome"),
    (lambda e: e.pop("superseded"), "supersedes nothing"),                                              # a ruling with no superseded block
    (lambda e: e["ruling"].update(alpha=0.1), "arditi.ruling.alpha"),                                   # the ruling names another alpha
    (lambda e: e["ruling"].update(object_sha256=ARDITI_SHA_OLD), "arditi.ruling.object_sha256"),        # the ruling names another object
    (lambda e: e["superseded"].update(object_sha256="0" * 64), "arditi.superseded.object_sha256"),      # superseded != ruling.supersedes
    (lambda e: e["ruling"]["supersedes"].update(object_sha256=ARDITI_SHA_NEW, alpha=0.05), "names the ACTIVE point itself"),
    (lambda e: e.update(view="ar_r0p1"), "arditi.view"),                                                # the view is not the alpha spelled
    (lambda e: e.update(staged_id="ar_r0p1__arditi"), "arditi.staged_id"),
    (lambda e: e["bank_view_rel"].update(overflow="exp/objects/arditi_strength/llama8b/strength/cells_view/ar_r0p1"), "bank_view_rel.overflow"),
    (lambda e: e["superseded"].pop("superseded_by"), "superseded.superseded_by"),
])
def test_a_ruled_arditi_entry_that_does_not_bind_refuses_naming_the_field(tmp_path, mutate, needle):
    root = _ladder_root(tmp_path); rec = OP.build(root, tmp_path / "no_curves.json"); entries = _entries(rec)
    e = _ruled_arditi(); mutate(e); entries[("llama8b", "arditi")] = e
    with pytest.raises(SystemExit) as ei:
        R.check_entries(entries, where="synthetic.json", ladder_root=root, curves=tmp_path / "no_curves.json", arditi_record=_battery(tmp_path))
    assert needle in str(ei.value) and "does not bind" in str(ei.value)


def test_an_arditi_alpha_the_battery_never_measured_or_disagrees_with_refuses(tmp_path):
    root = _ladder_root(tmp_path); rec = OP.build(root, tmp_path / "no_curves.json"); entries = _entries(rec)
    entries[("llama8b", "arditi")] = _ruled_arditi()
    with pytest.raises(SystemExit, match="never measured 'ar_r0p05'"):
        R.check_entries(entries, where="synthetic.json", ladder_root=root, curves=tmp_path / "no_curves.json",
                        arditi_record=_battery(tmp_path, alphas=(0.1, 0.2)))
    bat = tmp_path / "b2.json"; doc = json.loads(_battery(tmp_path).read_text()); doc["targets"]["llama8b"]["cells"]["ar_r0p05"]["alpha"] = 0.06
    bat.write_text(json.dumps(doc))
    with pytest.raises(SystemExit, match=r"arditi\.alpha: record=0\.05 battery\[ar_r0p05\]=0\.06"):
        R.check_entries(entries, where="synthetic.json", ladder_root=root, curves=tmp_path / "no_curves.json", arditi_record=bat)
    # an unruled entry naming alpha + view is held to the same spelling / battery checks; one naming neither (a fixture) is left alone
    plain = {"model_key": "qwen35_4b", "cell": "arditi", "selection_outcome": None, "alpha": 0.05, "view": "ar_r0p1", "object_sha256": _sha("ar", "qwen35_4b")}
    with pytest.raises(SystemExit, match=r"entries\.qwen35_4b\.arditi\.view"):
        R.check_entries({("qwen35_4b", "arditi"): plain}, where="synthetic.json", ladder_root=root, curves=tmp_path / "no_curves.json", arditi_record=_battery(tmp_path))
    assert R.arditi_entry_problems("qwen35_4b", "arditi", {"model_key": "qwen35_4b", "cell": "arditi", "selection_outcome": None}, json.loads(_battery(tmp_path).read_text())) == []


def test_the_committed_arditi_entries_bind_to_the_battery_where_it_is_present():
    """The live guard for the Arditi half: on the analysis workstation the committed record's four Arditi entries bind to the battery
    record (llama8b RULED to alpha 0.05 on 2026-09-14; the other three at the battery's own 0.05 pick); elsewhere the spelling / block
    checks still run and the battery half skips."""
    doc = json.loads((REPO / "configs/sdfbase/steer_admission.json").read_text(encoding="utf-8"))
    entries = {(k, c): e for k, by in doc["entries"].items() for c, e in by.items() if c == "arditi"}
    battery = R.load_arditi_record()
    for (k, c), e in entries.items():
        assert R.arditi_entry_problems(k, c, e, battery) == [], (k, c)
    llama = entries[("llama8b", "arditi")]
    assert llama["selection_outcome"] == R.ARDITI_RULED_OUTCOME and llama["alpha"] == 0.05 and llama["view"] == "ar_r0p05"
    assert llama["ruling"]["supersedes"]["alpha"] == 0.1 and llama["superseded"]["view"] == "ar_r0p1"
    assert {k for (k, _), e in entries.items() if e["selection_outcome"] is None} == {"gemma4e4b", "qwen35_4b", "nemotron3nano4b"}


# --------------------------------------------------------------------------- #
# the ALL-POSITIONS Arditi entry (`arditi_allpos`, qwen35_4b only — never `arditi`, the prefill-only arm): a RAW-COEFFICIENT dose on a
# vector_all object in BOTH phases on the native residual slot; the outcome EXACTLY the declared over-budget override, with both
# readings and the ladder cap on the entry; a complete fit provenance; the view the coefficient spelled (ar_c0p3), never an alpha (ar_r…)
# --------------------------------------------------------------------------- #

ALLPOS_SHA = _sha("ap", "qwen35_4b", 0.3)
ALLPOS_REL = "exp/objects/arditi_allpos/qwen35_4b/ladder/cells_view/ar_c0p3"


def _allpos(key: str = "qwen35_4b") -> dict:
    """A synthetic all-positions entry in the committed one's SHAPE (invented shas and readings — a fixture, not a measurement)."""
    return {
        "arm": "arditi_allpos", "cell": "arditi_allpos", "model_key": key, "object_sha256": ALLPOS_SHA, "view": "ar_c0p3",
        "bank_cell_dir": "arditi_allpos", "staged_id": "ar_c0p3__arditi_allpos", "bank_view_rel": {"primary": ALLPOS_REL, "overflow": ALLPOS_REL},
        "lane_root": "exp/objects/arditi_allpos", "coefficient": 0.3, "dose_unit": "raw_coefficient", "kind": "vector_all", "phase": "both",
        "served_layer": 12, "serving_path": "lora_native", "selection_outcome": R.ALLPOS_OUTCOME,
        "selection": {"outcome": R.ALLPOS_OUTCOME, "rule_returns": "ar_c0p2", "rule_reading_pp": 5.0, "override_reading_pp": 10.3, "budget_pp": 10,
                      "ladder_cap": 2048, "single_trial": True, "dose": {"value": 0.3, "unit": "raw_coefficient"}},
        "fit_provenance": {"fitted_on_model": "synthetic/model", "fitted_on_revision": _sha("rev")[:40], "rendering": "chat template, synthetic",
                           # its OWN key, never prose the export must parse: the positions the fit activations were captured at
                           "capture_positions": "9 post-instruction positions",
                           "item_source": "data/synthetic/fit_prompts.json", "item_source_repo_sha": _sha("tree")[:40],
                           "realised_counts": {"harmful_train": 120, "harmful_val": 30, "harmless_train": 121, "harmless_val": 31},
                           "direction_sha256": _sha("dir"), "direction_pt_sha256": _sha("dir.pt"), "fit_layer": 13, "fit_position": -5,
                           "direction_norm": 3.0, "served_layer": 12,
                           "fit_quality_gate": {"harmful_mean": 5.0, "harmless_mean": -10.0, "harmful_frac_positive": 0.95, "harmless_frac_negative": 0.97},
                           "fit_record_sha256": _sha("fit_record")},
        "attestation_expect": {"serving_path": "lora_native", "op": "add_vector", "coeff": 1.0, "positions": "all", "phase": "both",
                               "served_layer": 12, "served_positions_expected": "all"},
    }


def test_c_view_is_the_coefficient_spelled_never_the_alpha_spelling():
    assert [R.c_view(c) for c in (0.3, 0.2, 1.0, 0.5, 0.05)] == ["ar_c0p3", "ar_c0p2", "ar_c1", "ar_c0p5", "ar_c0p05"]
    assert R.c_view(0.3) != R.arditi_view(0.3) == "ar_r0p3"          # same style, a different unit letter: c is never r


def test_an_all_positions_entry_that_binds_passes_with_and_without_the_ladder(tmp_path):
    e = _allpos()
    assert R.arditi_allpos_entry_problems("qwen35_4b", "arditi_allpos", e) == []
    kw = dict(where="synthetic.json", curves=tmp_path / "no_curves.json")
    # the AlphaSteer half skipped (no ladder) still runs this gate — None is the skip, not a pass on nothing
    assert R.check_entries({("qwen35_4b", "arditi_allpos"): e}, ladder_root=tmp_path / "nowhere", **kw) is None
    root = _ladder_root(tmp_path); rec = OP.build(root, tmp_path / "no_curves.json"); entries = _entries(rec)
    entries[("qwen35_4b", "arditi_allpos")] = e
    assert R.check_entries(entries, ladder_root=root, arditi_record=_battery(tmp_path), **kw) is not None


def _served_layer_13_everywhere(e: dict) -> None:
    """THE FOOTGUN: the FIT layer (13) written as this arm's served layer on every copy — only the fit - 1 rule can catch it."""
    e["served_layer"] = 13; e["fit_provenance"]["served_layer"] = 13; e["attestation_expect"]["served_layer"] = 13


_ALLPOS_BAD = [
    # alpha-shaped: the prefill-only arm's dose shape on the all-positions arm
    (lambda e: e.update(alpha=0.1054), "arditi_allpos.alpha: present"),
    (lambda e: e.update(view="ar_r0p3"), "arditi_allpos.view: 'ar_r0p3' is the ALPHA spelling"),
    # the dose unit / kind / phase
    (lambda e: e.update(dose_unit="alpha"), "arditi_allpos.dose_unit: record='alpha'"),
    (lambda e: e.update(dose_unit="residual_norm_fraction"), "arditi_allpos.dose_unit: record='residual_norm_fraction'"),
    (lambda e: e.update(kind="static_r"), "arditi_allpos.kind: record='static_r'"),
    (lambda e: e.update(phase="prefill"), "arditi_allpos.phase: record='prefill'"),
    # the selection block and the outcome word (on the entry AND in the block)
    (lambda e: e.pop("selection"), "arditi_allpos.selection: missing"),
    (lambda e: e["selection"].pop("outcome"), "arditi_allpos.selection.outcome: missing"),
    (lambda e: e["selection"].update(outcome=None), "arditi_allpos.selection.outcome: missing"),
    (lambda e: e["selection"].update(outcome=R.ARDITI_RULED_OUTCOME), "arditi_allpos.selection.outcome: record='ruled_2026-09-14'"),
    (lambda e: e.update(selection_outcome=R.ARDITI_RULED_OUTCOME), "arditi_allpos.selection_outcome: record='ruled_2026-09-14'"),   # the Arditi word
    (lambda e: e.update(selection_outcome="chosen"), "arditi_allpos.selection_outcome: record='chosen'"),                            # the ladder words
    (lambda e: e.update(selection_outcome="ruled_2026-09-14_second"), "arditi_allpos.selection_outcome: record='ruled_2026-09-14_second'"),
    (lambda e: e.update(selection_outcome="candidate_2026-09-14"), "arditi_allpos.selection_outcome: record='candidate_2026-09-14'"),
    (lambda e: e.update(selection_outcome="user_override_2026-09-16"), "arditi_allpos.selection_outcome: record='user_override_2026-09-16'"),  # a near miss
    (lambda e: e.update(selection_outcome=None), "arditi_allpos.selection_outcome: record=None"),
    # the dose spelled: view <-> dose, staged id, bank paths, the top-level mirror
    (lambda e: e["selection"]["dose"].update(value=0.2), "arditi_allpos.view: 'ar_c0p3' is not the dose c=0.2 spelled ('ar_c0p2')"),
    (lambda e: e.update(view="ar_c0p2"), "arditi_allpos.view: 'ar_c0p2' is not the dose c=0.3 spelled ('ar_c0p3')"),
    (lambda e: e["selection"]["dose"].update(unit="alpha"), "arditi_allpos.selection.dose:"),
    (lambda e: e["selection"].pop("dose"), "arditi_allpos.selection.dose:"),
    (lambda e: e.update(coefficient=0.2), "arditi_allpos.coefficient: record=0.2 selection.dose.value=0.3"),
    (lambda e: e.update(staged_id="ar_c0p2__arditi_allpos"), "arditi_allpos.staged_id: 'ar_c0p2__arditi_allpos' does not bind the view 'ar_c0p3'"),
    (lambda e: e.update(staged_id="ar_c0p3__arditi"), "arditi_allpos.staged_id: 'ar_c0p3__arditi' does not bind"),            # the prefill arm's suffix
    (lambda e: e["bank_view_rel"].update(overflow="exp/objects/arditi_allpos/qwen35_4b/ladder/cells_view/ar_c0p2"), "arditi_allpos.bank_view_rel.overflow"),
    # the reading's identity
    (lambda e: e["selection"].pop("ladder_cap"), "arditi_allpos.selection.ladder_cap: missing"),
    (lambda e: e["selection"].update(ladder_cap=None), "arditi_allpos.selection.ladder_cap: missing"),
    (lambda e: e["selection"].update(ladder_cap="2048"), "arditi_allpos.selection.ladder_cap: '2048' is not a positive token cap"),
    (lambda e: e["selection"].pop("single_trial"), "arditi_allpos.selection.single_trial: None"),
    # the override supersedes a DIFFERENT point
    (lambda e: e["selection"].pop("rule_returns"), "arditi_allpos.selection.rule_returns: missing"),
    (lambda e: e["selection"].update(rule_returns=None), "arditi_allpos.selection.rule_returns: missing"),
    (lambda e: e["selection"].update(rule_returns="ar_c0p3"), "arditi_allpos.selection.rule_returns: 'ar_c0p3' is the selected view itself"),
    # the readings: the override OVER budget, the rule's return WITHIN it
    (lambda e: e["selection"].update(override_reading_pp=9.5), "arditi_allpos.selection.override_reading_pp: 9.5 is within the budget 10"),
    (lambda e: e["selection"].update(override_reading_pp=10), "arditi_allpos.selection.override_reading_pp: 10 is within the budget 10"),   # AT budget is not over it
    (lambda e: e["selection"].update(rule_reading_pp=10.5), "arditi_allpos.selection.rule_reading_pp: 10.5 is over the budget 10"),
    (lambda e: e["selection"].pop("budget_pp"), "arditi_allpos.selection.budget_pp: None is not a number"),
    (lambda e: e["selection"].update(override_reading_pp="10.3"), "arditi_allpos.selection.override_reading_pp: '10.3' is not a number"),
    # fit provenance: present, complete, non-null, shas 64-hex
    (lambda e: e.pop("fit_provenance"), "arditi_allpos.fit_provenance: missing"),
    (lambda e: e["fit_provenance"].pop("fitted_on_revision"), "arditi_allpos.fit_provenance.fitted_on_revision: missing"),
    (lambda e: e["fit_provenance"].pop("fitted_on_model"), "arditi_allpos.fit_provenance.fitted_on_model: missing"),
    (lambda e: e["fit_provenance"].pop("rendering"), "arditi_allpos.fit_provenance.rendering: missing"),
    (lambda e: e["fit_provenance"].pop("item_source_repo_sha"), "arditi_allpos.fit_provenance.item_source_repo_sha: missing"),
    (lambda e: e["fit_provenance"].update(direction_sha256=None), "arditi_allpos.fit_provenance.direction_sha256: missing"),
    (lambda e: e["fit_provenance"].pop("direction_pt_sha256"), "arditi_allpos.fit_provenance.direction_pt_sha256: missing"),
    (lambda e: e["fit_provenance"].pop("fit_position"), "arditi_allpos.fit_provenance.fit_position: missing"),
    (lambda e: e["fit_provenance"].pop("direction_norm"), "arditi_allpos.fit_provenance.direction_norm: missing"),
    (lambda e: e["fit_provenance"].pop("fit_record_sha256"), "arditi_allpos.fit_provenance.fit_record_sha256: missing"),
    (lambda e: e["fit_provenance"].update(fit_record_sha256="deadbeef"), "arditi_allpos.fit_provenance.fit_record_sha256: 'deadbeef' is not a 64-hex"),
    (lambda e: e["fit_provenance"].pop("realised_counts"), "arditi_allpos.fit_provenance.realised_counts: missing"),
    (lambda e: e["fit_provenance"]["realised_counts"].pop("harmless_val"), "arditi_allpos.fit_provenance.realised_counts.harmless_val: None"),
    (lambda e: e["fit_provenance"]["fit_quality_gate"].pop("harmful_frac_positive"), "arditi_allpos.fit_provenance.fit_quality_gate.harmful_frac_positive: None"),
    (lambda e: e["fit_provenance"].update(fit_quality_gate="passed"), "arditi_allpos.fit_provenance.fit_quality_gate: 'passed' is not a block"),
    # the served layer: one value written once, and fit - 1 (the vector_all footgun)
    (_served_layer_13_everywhere, "arditi_allpos.served_layer: 13 is not fit_layer - 1 (13 - 1 = 12)"),
    (lambda e: e["fit_provenance"].update(served_layer=13), "arditi_allpos.served_layer: record=12 fit_provenance.served_layer=13"),
    # the native slot: serving_path and the attestation block, no lens key
    (lambda e: e.update(serving_path="lens"), "arditi_allpos.serving_path: record='lens'"),
    (lambda e: e.pop("serving_path"), "arditi_allpos.serving_path: record=None"),
    (lambda e: e.pop("attestation_expect"), "arditi_allpos.attestation_expect: missing"),
    (lambda e: e["attestation_expect"].pop("serving_path"), "arditi_allpos.attestation_expect.serving_path: record=None native-slot='lora_native'"),
    (lambda e: e["attestation_expect"].update(serving_path="lens"), "arditi_allpos.attestation_expect.serving_path: record='lens'"),
    (lambda e: e["attestation_expect"].update(coeff=0.3), "arditi_allpos.attestation_expect.coeff: record=0.3 native-slot=1.0"),
    (lambda e: e["attestation_expect"].update(op="relu_gated_vector"), "arditi_allpos.attestation_expect.op: record='relu_gated_vector'"),
    (lambda e: e["attestation_expect"].update(phase="prefill"), "arditi_allpos.attestation_expect.phase: record='prefill' native-slot='both'"),
    (lambda e: e["attestation_expect"].update(positions="prefill / all"), "arditi_allpos.attestation_expect.positions: record='prefill / all'"),
    (lambda e: e["attestation_expect"].update(served_positions_expected="response"), "arditi_allpos.attestation_expect.served_positions_expected"),
    (lambda e: e["attestation_expect"].update(lens_reduction_mode="stacked"), "arditi_allpos.attestation_expect.lens_reduction_mode: 'stacked' is a LENS key"),
    (lambda e: e["attestation_expect"].update(served_layer=13), "arditi_allpos.attestation_expect.served_layer: record=13 entry=12"),
]


@pytest.mark.parametrize("mutate,needle", _ALLPOS_BAD, ids=[n.split(":")[0].rsplit(".", 1)[-1] + f"#{i}" for i, (_, n) in enumerate(_ALLPOS_BAD)])
def test_an_all_positions_entry_that_does_not_bind_refuses_naming_the_field(tmp_path, mutate, needle):
    """One KNOWN-BAD twin per refusal: the gate names the entry and the field, and check_entries carries it as the arm's own refusal."""
    e = _allpos(); mutate(e)
    problems = R.arditi_allpos_entry_problems("qwen35_4b", "arditi_allpos", e)
    assert any(needle in p for p in problems), (needle, problems)
    with pytest.raises(SystemExit) as ei:
        R.check_entries({("qwen35_4b", "arditi_allpos"): e}, where="synthetic.json", ladder_root=tmp_path / "nowhere", curves=tmp_path / "no_curves.json")
    msg = str(ei.value)
    assert needle in msg and "an all-positions Arditi entry does not bind" in msg and "never file it as an alpha" in msg


def test_an_all_positions_entry_filed_under_the_prefill_arms_cell_refuses_and_vice_versa(tmp_path):
    """`arditi_allpos` is NEVER `arditi`: each arm's gate refuses the other's shape by name."""
    e = dict(_allpos(), cell="arditi")
    with pytest.raises(SystemExit, match=r"entries\.qwen35_4b\.arditi\.selection_outcome: record='user_override_over_budget_2026-09-16'"):
        R.check_entries({("qwen35_4b", "arditi"): e}, where="synthetic.json", ladder_root=tmp_path / "nowhere", curves=tmp_path / "no_curves.json")
    ruled = dict(_ruled_arditi("qwen35_4b"), cell="arditi_allpos")
    problems = R.arditi_allpos_entry_problems("qwen35_4b", "arditi_allpos", ruled)
    assert any("arditi_allpos.alpha: present" in p for p in problems) and any("is the ALPHA spelling" in p for p in problems)
    assert any("arditi_allpos.selection_outcome: record='ruled_2026-09-14'" in p for p in problems)


def test_the_dispatch_is_total_a_cell_no_gate_knows_is_refused_by_name(tmp_path):
    """A steer cell outside the gated three used to fall through UNCHECKED (a silent admission); now it is named and refused, with or
    without the ladder — before any half runs."""
    assert set(R.GATED_CELLS) == {"alphasteer", "arditi", "arditi_allpos"}
    stray = dict(_allpos(), cell="arditi_decode_only", staged_id="ar_c0p3__arditi_decode_only")
    kw = dict(where="synthetic.json", curves=tmp_path / "no_curves.json")
    with pytest.raises(SystemExit, match=r"no admission gate knows the steer cell\(s\) \['entries\.qwen35_4b\.arditi_decode_only'\]"):
        R.check_entries({("qwen35_4b", "arditi_decode_only"): stray}, ladder_root=tmp_path / "nowhere", **kw)
    root = _ladder_root(tmp_path); rec = OP.build(root, tmp_path / "no_curves.json"); entries = _entries(rec)
    entries[("qwen35_4b", "arditi_decode_only")] = stray
    with pytest.raises(SystemExit, match="no admission gate knows") as ei:
        R.check_entries(entries, ladder_root=root, arditi_record=_battery(tmp_path), **kw)
    assert "never admitted unchecked" in str(ei.value)


def test_the_committed_all_positions_entry_binds_and_is_filed_as_the_declared_override():
    """The live entry: qwen35_4b's `arditi_allpos` binds by itself (no ladder, no battery needed) and says what it is — a raw-coefficient
    dose on a vector_all object, the OVER-budget override with both readings and the cap, a complete fit provenance, the native
    slot; never an alpha, never a claim that the rule chose it, never a significance word."""
    doc = json.loads((REPO / "configs/sdfbase/steer_admission.json").read_text(encoding="utf-8"))
    entries = {(k, c): e for k, by in doc["entries"].items() for c, e in by.items() if c == R.CELL_ARDITI_ALLPOS}
    assert set(entries) == {("qwen35_4b", "arditi_allpos")}                       # the arm is qwen35_4b ONLY
    e = entries[("qwen35_4b", "arditi_allpos")]
    assert R.arditi_allpos_entry_problems("qwen35_4b", "arditi_allpos", e) == []
    assert R.check_entries(entries, where="steer_admission.json", ladder_root=Path("/nonexistent"), curves=Path("/nonexistent.json")) is None
    assert "alpha" not in e and e["dose_unit"] == "raw_coefficient" and e["kind"] == "vector_all" and e["phase"] == "both"
    assert e["view"] == R.c_view(e["selection"]["dose"]["value"]) == "ar_c0p3" and e["staged_id"] == "ar_c0p3__arditi_allpos"
    assert e["coefficient"] == e["selection"]["dose"]["value"] == 0.3 and e["cluster_constraint"] == "primary"
    assert e["bank_view_rel"]["primary"] == e["bank_view_rel"]["overflow"] == "exp/objects/arditi_allpos/qwen35_4b/ladder/cells_view/ar_c0p3"
    sel = e["selection"]
    assert e["selection_outcome"] == sel["outcome"] == R.ALLPOS_OUTCOME == "user_override_over_budget_2026-09-16"
    assert sel["rule_returns"] == "ar_c0p2" != e["view"] and sel["ladder_cap"] == 2048 and sel["single_trial"] is True
    assert sel["rule_reading_pp"] == 5.04 and sel["override_reading_pp"] == 10.26 and sel["budget_pp"] == 10
    assert sel["rule_reading_pp"] <= sel["budget_pp"] < sel["override_reading_pp"]
    assert sel["dose"] == {"value": 0.3, "unit": "raw_coefficient"} and sel["alpha_equiv"]["informational"] is True
    fp = e["fit_provenance"]
    assert fp["fit_layer"] - 1 == fp["served_layer"] == e["served_layer"] == e["attestation_expect"]["served_layer"] == 12
    assert all(fp.get(k) is not None for k in R.FIT_PROVENANCE_KEYS)
    assert {k: fp["realised_counts"][k] for k in R.FIT_REALISED_COUNT_KEYS} == {"harmful_train": 123, "harmful_val": 31, "harmless_train": 125, "harmless_val": 31}
    assert fp["realised_counts"]["harmful_train"] != 128 and fp["realised_counts"]["harmful_val"] != 32          # realised, never the nominal
    assert e["serving_path"] == e["attestation_expect"]["serving_path"] == "lora_native" and e["attestation_expect"]["coeff"] == 1.0
    assert e["attestation_expect"]["op"] == "add_vector" and e["attestation_expect"]["phase"] == "both" and "lens_reduction_mode" not in e["attestation_expect"]
    # the prose: an override the rule did NOT make, on a single trial — no significance word anywhere on the entry
    assert "OVERRIDE" in e["selection_label"] and "returns c = 0.2" in e["selection_label"] and "10.26" in e["selection_label"]
    text = json.dumps(e).lower()
    for word in ("p-value", "p value", "significant", "confidence interval", "wilson"):
        assert word not in text, word
    # and the record's own documentation names the new class
    assert "arditi_allpos" in doc["what"] and "user_override_over_budget_2026-09-16" in doc["what"]
