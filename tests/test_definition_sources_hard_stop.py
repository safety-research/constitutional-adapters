"""``degeneracy_union.definition_sources_or_refuse`` -- the ONE hard stop every producer of degeneration rows applies to
``verify_definition_sources``: ``ok`` False refuses by name whether the source is PRESENT and drifted ("do not match their pins") or an
in-repo pin that is ABSENT (the historical grading driver's committed evidence copy, configs/evidence/arditi_driver_3fed461f.py.txt:
"not verified"); ``ok`` None (a source outside the repository that is not on this host) is recorded, never refused.

Each producer's front door is exercised with the evidence pinned at an ABSENT in-repo path, the real ``verify_definition_sources``
running: ``degeneracy_gap_fill.load_union``, ``sampled/static_degeneracy_union.load_union``, ``mt_degeneracy_union.load_union``,
``degeneration_measures.verify_definitions_once`` (the closer's) and ``static_padding/degeneracy_items`` (its gate) -- every one refuses
naming the entry; none filters its verdicts on ``present`` any more (a filter on ``present`` is what let the gap-fill proceed without
the evidence file while the others merely stamped the verdict). A union tree whose module has no shared rule is refused, never judged
by a weaker rule of the producer's own."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from sisteer.eval import degeneracy_union as du

REPO = Path(__file__).resolve().parents[1]
FULL22 = REPO / "scripts" / "sdfbase" / "full22"
ABSENT = "configs/evidence/absent.py.txt"
ENTRY = "zlib_collapse.extracted_from"
PRODUCERS = {
    "degeneracy_gap_fill": FULL22 / "degeneracy_gap_fill.py",
    "static_degeneracy_union": FULL22 / "sampled" / "static_degeneracy_union.py",
    "mt_degeneracy_union": FULL22 / "mt_degeneracy_union.py",
}


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_hard_stop_under_test", PRODUCERS[name])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def absent_evidence(monkeypatch):
    """The historical driver's evidence copy pinned at an in-repo path this tree does not carry: `present` False, `ok` False."""
    monkeypatch.setitem(du.DEFINITION_SOURCES["zlib_collapse"], "extracted_from", dict(du._ARDITI_DRIVER, path=ABSENT))
    if str(REPO / "src") not in sys.path:
        sys.path.insert(0, str(REPO / "src"))
    return ABSENT


def test_the_rule_refuses_an_absent_in_repo_pin_by_name_with_the_entries_and_the_record_on_the_exception(absent_evidence):
    with pytest.raises(du.DefinitionSourcesUnverified) as ei:
        du.definition_sources_or_refuse()
    assert ei.value.failed == [ENTRY] and ei.value.record[ENTRY]["present"] is False and ei.value.record[ENTRY]["ok"] is False
    msg = str(ei.value)
    assert f"in-repo definition sources not verified: ['{ENTRY}']" in msg and "an absent in-repo pin is a failed verdict" in msg, msg
    assert "do not match their pins" not in msg and ABSENT in msg and "refusing to file degeneration rows" in msg, msg
    assert isinstance(ei.value, RuntimeError)                                                 # every producer catches it as one


def test_the_rule_on_a_handed_in_record_drift_absence_and_an_in_repo_entry_without_a_verdict_refuse_and_an_outside_source_passes():
    real = du.verify_definition_sources()
    assert du.definition_sources_or_refuse(real) == dict(real)                                # this tree verifies
    outside = {k: dict(v) for k, v in real.items()}
    outside["rep_ratio_loop.definition_origin"] = {**outside["rep_ratio_loop.definition_origin"], "present": False, "ok": None, "actual": None}
    assert du.definition_sources_or_refuse(outside)["rep_ratio_loop.definition_origin"]["ok"] is None
    drift = {k: dict(v) for k, v in real.items()}
    drift["classifier"] = {**drift["classifier"], "ok": False, "actual": "0" * 64}
    with pytest.raises(du.DefinitionSourcesUnverified, match=r"do not match their pins: \['classifier'\]") as ei:
        du.definition_sources_or_refuse(drift)
    assert ei.value.failed == ["classifier"] and "not verified" not in str(ei.value)
    absent = {k: dict(v) for k, v in real.items()}
    absent[ENTRY] = {"path": str(REPO / ABSENT), "expected": "0" * 64, "actual": None, "present": False, "ok": False}
    with pytest.raises(du.DefinitionSourcesUnverified, match=rf"in-repo definition sources not verified: \['{ENTRY}'\]"):
        du.definition_sources_or_refuse(absent)
    both = dict(drift)
    both[ENTRY] = absent[ENTRY]
    with pytest.raises(du.DefinitionSourcesUnverified) as ei:
        du.definition_sources_or_refuse(both)
    assert ei.value.failed == ["classifier", ENTRY] and "do not match their pins: ['classifier']" in str(ei.value) and f"not verified: ['{ENTRY}']" in str(ei.value)
    # an in-repo entry whose verdict is None (never produced by verify_definition_sources, but a stamped record could carry it): refused, never recorded
    weird = {k: dict(v) for k, v in real.items()}
    weird["classifier"] = {**weird["classifier"], "ok": None}
    with pytest.raises(du.DefinitionSourcesUnverified, match=r"not verified: \['classifier'\]"):
        du.definition_sources_or_refuse(weird)
    # the `module` descriptor carries no verdict and is never judged
    assert "module" in du.definition_sources_or_refuse(real) and "ok" not in real["module"]


def _refused(mod, capsys) -> str:
    """``load_union`` on this repository's own src with the evidence pinned absent: the producer's Refuse, naming the entry and the path --
    on stderr for the gap-fill and sampled runners (their Refuse prints and exits 2), in the exit message for the multi-turn runner."""
    with pytest.raises(SystemExit) as ei:
        mod.load_union(REPO / "src")
    err = capsys.readouterr().err + str(ei.value)
    assert "union definition pins do not verify (absent in-repo source, or drift)" in err and ENTRY in err and ABSENT in err, err
    assert "not verified" in err and "definition pins drifted" not in err, err
    return err


@pytest.mark.parametrize("name", sorted(PRODUCERS))
def test_each_producers_front_door_refuses_an_absent_evidence_file_and_admits_this_tree(name, absent_evidence, capsys, monkeypatch):
    mod = _load(name)
    _refused(mod, capsys)
    src = PRODUCERS[name].read_text()
    assert "definition_sources_or_refuse" in src and 'r.get("present") and r.get("ok") is False' not in src, name
    # the pin restored: this tree is admitted and the verdict record is stamped, the evidence entry ok True
    monkeypatch.setitem(du.DEFINITION_SOURCES["zlib_collapse"], "extracted_from", du._ARDITI_DRIVER)
    out = mod.load_union(REPO / "src")
    stamp = out[-1]
    assert stamp["verify_definition_sources"][ENTRY]["ok"] is True and stamp["verify_definition_sources"][ENTRY]["present"] is True
    assert stamp["verify_definition_sources"]["module"]["definition_sha"] == du.DEFINITION_SHA


@pytest.mark.parametrize("name", sorted(PRODUCERS))
def test_a_union_tree_without_the_shared_rule_is_refused_by_name_never_judged_by_a_weaker_rule(name, capsys, monkeypatch):
    mod = _load(name)
    monkeypatch.delattr(du, "definition_sources_or_refuse")
    with pytest.raises(SystemExit) as ei:
        mod.load_union(REPO / "src")
    err = capsys.readouterr().err + str(ei.value)
    assert "has no definition_sources_or_refuse" in err and "shared hard stop" in err, err


def test_the_closers_degeneration_measures_and_the_padded_items_writer_apply_the_same_rule(absent_evidence):
    for p in (str(FULL22), str(FULL22 / "static_padding")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import degeneration_measures as DM
    DM._VERIFIED.clear()
    with pytest.raises(RuntimeError, match=rf"in-repo definition sources not verified: \['{ENTRY}'\]") as ei:
        DM.verify_definitions_once()
    assert isinstance(ei.value, du.DefinitionSourcesUnverified) and not DM._VERIFIED
    DM._VERIFIED.clear()
    dm_src = (FULL22 / "degeneration_measures.py").read_text()
    assert "du.definition_sources_or_refuse()" in dm_src and 'r.get("present") is not False' not in dm_src
    di_src = (FULL22 / "static_padding" / "degeneracy_items.py").read_text()
    assert "du.definition_sources_or_refuse(" in di_src and 'v.get("ok") is False]' not in di_src
    # the shared rule is the only judge of `ok` left in the producers: no producer reads `ok` False on its own any more
    for name, path in PRODUCERS.items():
        src = path.read_text()
        assert '"ok") is False' not in src, name
