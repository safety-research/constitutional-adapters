"""DOSEPETRIFIX (chain 368): the steered-view TAG grammar's two suffix copies read a FRONT-DOOR dose view as steered
through the models table, keep the s-family by name, and leave every paper target's dose-named view alone.

Pins: (1) the helper -- name-only reading False for d0p75_relul and d0p1; target-aware: the tree's front-door model (qwen35_4b_demo) runs the
SINGLE-PHASE recipe (one wave s1 + two dose rungs that carry their own `none`, no u0), so ITS dose arms are NOT steered waves
and borrow no anchor, while a SYNTHETIC row that runs in phases (u0 + the dose views, on a tmp copy of the table) classifies its dose arms as steered
and borrows `<key>_u0`; False for the five paper targets' d0p1/d0p25/d0p5/d1p5, s1 / s2rl as before, negatives (_u0, none, main, a d-less
bare tag, an unknown key), and a missing reproduce/lib/models.py REFUSED BY NAME; (2) petri_run.steered_wave_audit and emit_contrasts.u0_view_of
over the same tags, with the s-family regex literals unchanged; (3) agreement: on every front-door dose arm the models table declares (what 05 --step
roster prints into the roster block) the two call sites, the helper and f22_roster.steered_wave_view / models.steered_wave_view classify identically;
(4) copies (3)-(5) -- the three STEERED_WAVE_VIEW_RE view-name grammars -- are unchanged by name; (5) the table is
read ONCE per revision and never for a tag that is not dose-shaped (the daemon's contrasts tick classifies every complete steered run), a changed
table is re-read, and a missing TABLE refuses by name like a missing reader."""
from __future__ import annotations

import importlib.util
import os
import re
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "sdfbase" / "full22"))
sys.path.insert(0, str(REPO / "reproduce" / "lib"))

from sisteer import front_door_views as fdv  # noqa: E402
import petri_run as prun  # noqa: E402
import f22_roster as FR  # noqa: E402
import models as M  # noqa: E402

KEY = "qwen35_4b_demo"                                   # the front-door model of reproduce/models.yaml (configured, not run): single-phase, no u0
PHASED = "phased_demo"                                   # a SYNTHETIC row that runs in PHASES (u0 + the dose views) on a tmp copy of the table
PAPER = ("qwen35_4b", "llama8b", "gemma4e4b", "nemotron3nano4b", "glm46vflash")
PAPER_DOSES = ("d0p1", "d0p25", "d0p5", "d1p5")
FD_DOSES = ("d0p75", "d0p75_relu")                  # the table's dose views of the front-door model (and of the synthetic phased row)


def _phased_table(d: Path) -> str:
    """A copy of the models table plus PHASED: the demo's row under its own key and view_legacy_prefix with u0 declared beside the two dose views."""
    import copy, yaml
    t = yaml.safe_load((REPO / "reproduce" / "models.yaml").read_text())
    row = copy.deepcopy(t["models"][KEY]); row["view_legacy_prefix"] = PHASED
    row["views"] = {"u0": {"cells": ["none", "prompted__safety_refusal", "prompted__si_constitution"]}, **{v: row["views"][v] for v in FD_DOSES}}
    t["models"][PHASED] = row
    d.mkdir(parents=True, exist_ok=True); p = d / "models.yaml"; p.write_text(yaml.safe_dump(t, sort_keys=False))
    return str(p)


@pytest.fixture
def phased(tmp_path, monkeypatch) -> str:
    """The phased table bound as THE models table (every copy reads SISTEER_MODELS_TABLE: the helper, the roster rule, the reader, the two call sites)."""
    p = _phased_table(tmp_path / "phased")
    monkeypatch.setenv("SISTEER_MODELS_TABLE", p); fdv._CACHE.clear(); FR._FRONT_DOOR_CACHE.clear(); FR._SINGLE_PHASE_CACHE.clear()
    yield p
    fdv._CACHE.clear(); FR._FRONT_DOOR_CACHE.clear(); FR._SINGLE_PHASE_CACHE.clear()


def _ec():
    spec = importlib.util.spec_from_file_location("emit_contrasts_under_test", REPO / "scripts" / "registry" / "emit_contrasts.py")
    ec = importlib.util.module_from_spec(spec); spec.loader.exec_module(ec)
    return ec


def _S(audit):
    return type("S", (), {"audit": audit})()


# ------------------------------------------------------------------------------------------------- (1) the helper

def test_the_helper_reads_the_front_door_keys_from_the_models_table_and_splits_a_tag_longest_key_first(phased):
    keys = fdv.front_door_keys()
    assert KEY in keys and PHASED in keys and not (set(PAPER) & set(keys))
    assert fdv.split_key_view(f"petri_full22_{KEY}_d0p75_relu") == (KEY, "d0p75_relu")
    assert fdv.split_key_view(f"{KEY}_d0p75") == (KEY, "d0p75") and fdv.split_key_view(f"{KEY}_s2") == (KEY, "s2")
    assert fdv.split_key_view("qwen35_4b_d0p1") is None and fdv.split_key_view("unknownmodel_d0p75_relul") is None
    assert fdv.split_key_view("d0p75_relul") is None and fdv.split_key_view(f"{KEY}") is None and fdv.split_key_view("") is None
    assert fdv.front_door_dose_tag(f"petri_full22_{PHASED}_d0p75_relu") == PHASED and fdv.front_door_dose_tag(f"{PHASED}_d0p75") == PHASED   # a phased row's dose arm: steered, a front-door dose tag
    assert fdv.front_door_dose_tag(f"petri_full22_{KEY}_d0p75_relu") is None and fdv.front_door_dose_tag(f"{KEY}_d0p75") is None             # the single-phase demo's: not one (its own none)
    for t in (f"{KEY}_u0", f"{KEY}_battery", f"{KEY}_s2rl", f"{PHASED}_u0", "qwen35_4b_d0p1", "llama8b_d1p5", "unknownmodel_d0p75_relul", "d0p75_relul", KEY, "", None):
        assert fdv.front_door_dose_tag(t) is None, t


def test_a_front_door_dose_view_is_steered_a_paper_dose_view_is_not_and_the_name_alone_decides_nothing(phased):
    for v in FD_DOSES:
        assert fdv.front_door_dose_view(v, PHASED) and fdv.steered_wave_view(v, PHASED), v           # a row that runs in phases: its dose arms are steered
        assert not fdv.front_door_dose_view(v, KEY) and not fdv.steered_wave_view(v, KEY), v          # the single-phase demo: its dose rungs carry their own none
    for key in PAPER:
        for v in PAPER_DOSES:
            assert not fdv.front_door_dose_view(v, key), (key, v)
    assert not fdv.front_door_dose_view("d0p75_relul", "unknownmodel") and not fdv.front_door_dose_view("d0p1", "unknownmodel")
    # the name alone: the s-family regex does not admit a dose tag, and a dose name without a front-door key is nothing
    assert not prun._STEERED_AUDIT_RE.search(f"petri_full22_{KEY}_d0p75_relu") and not prun._STEERED_AUDIT_RE.search("petri_full22_qwen35_4b_d0p1")
    assert fdv.DOSE_VIEW_RE.match("d0p75_relul") and fdv.DOSE_VIEW_RE.match("d0p1") and not fdv.DOSE_VIEW_RE.match("s2rl")
    # negatives: u0, none, main, a d-less bare tag
    for v in ("u0", "none", "main", "it", "foo", "s1", "s2rl"):
        assert not fdv.front_door_dose_view(v, KEY) and not fdv.front_door_dose_view(v, PHASED), v


def test_a_missing_models_table_reader_is_refused_by_name_never_read_as_not_a_front_door_view(tmp_path, monkeypatch):
    monkeypatch.setattr(fdv, "MODELS_PY", tmp_path / "reproduce" / "lib" / "models.py")
    fdv._CACHE.clear()
    with pytest.raises(fdv.FrontDoorTableMissing, match="reproduce/lib/models.py is absent"):
        fdv.front_door_dose_view("d0p75_relu", KEY)
    with pytest.raises(fdv.FrontDoorTableMissing):
        prun.steered_wave_audit(_S(f"petri_full22_{KEY}_d0p75_relu"))
    with pytest.raises(fdv.FrontDoorTableMissing):
        _ec().u0_view_of(f"{KEY}_d0p75_relu")
    # the s-family clause needs no table: it still answers by name (its anchor is the `_u0` literal: an s<N>r view is the phased recipe's)
    assert prun.steered_wave_audit(_S(f"petri_full22_{KEY}_s2rl")) and _ec().u0_view_of(f"{KEY}_s2rl") == f"{KEY}_u0"


# ------------------------------------------------------------------------------------------------- (2) the two suffix copies

def test_steered_wave_audit_admits_the_front_door_dose_arms_and_keeps_every_other_reading(phased):
    for v in FD_DOSES:
        assert prun.steered_wave_audit(_S(f"petri_full22_{PHASED}_{v}")), v                        # the phased row's dose audits drive only the declared cells
        assert not prun.steered_wave_audit(_S(f"petri_full22_{KEY}_{v}")), v                       # the single-phase demo's drive their own none too
    for v in ("s2rl", "s2r", "s3r"):
        assert prun.steered_wave_audit(_S(f"petri_full22_{KEY}_{v}")), v
    for key in PAPER:
        for v in PAPER_DOSES:
            assert not prun.steered_wave_audit(_S(f"petri_full22_{key}_{v}")), (key, v)
    for a in (f"petri_full22_{KEY}_u0", f"petri_full22_{KEY}_battery", f"petri_full22_{PHASED}_u0", f"petri_full22_{KEY}_none", f"petri_full22_{KEY}_main", f"petri_full22_{KEY}",
              "petri_full22_synth", "petri_full22_unknownmodel_d0p75_relul", f"petri_full22_{KEY}_s2rlx", "d0p75_relul"):
        assert not prun.steered_wave_audit(_S(a)), a
    assert prun._STEERED_AUDIT_RE.pattern == r"_s[0-9]+rl?$"        # the s-family literal is unchanged (the grammar sweep + S2RLFIX5 pins hold)


def test_the_contrast_emitter_borrows_the_u0_anchor_for_a_front_door_dose_view_and_for_no_paper_dose_view(phased):
    ec = _ec()
    for v in FD_DOSES:
        assert ec.u0_view_of(f"{PHASED}_{v}") == f"{PHASED}_u0", v                                   # the phased row: the baseline wave the table declares
        assert ec.u0_view_of(f"{KEY}_{v}") is None, v                                               # the single-phase demo: no borrowed anchor (its own none)
    assert ec.u0_view_of(f"{KEY}_s2rl") == f"{KEY}_u0" and ec.u0_view_of(f"{KEY}_s2r") == f"{KEY}_u0"
    for key in PAPER:
        for v in PAPER_DOSES:
            assert ec.u0_view_of(f"{key}_{v}") is None, (key, v)
    for t in (f"{KEY}_u0", f"{KEY}_battery", f"{PHASED}_u0", f"{KEY}_none", f"{KEY}_main", KEY, "unknownmodel_d0p75_relul", "d0p75_relul", None, ""):
        assert ec.u0_view_of(t) is None, t
    assert ec._U0_VIEW_RE.pattern == r"_s[0-9]+rl?$"


# ------------------------------------------------------------------------------------------------- (3) agreement across the copies

def test_every_front_door_dose_arm_classifies_identically_in_the_two_call_sites_the_helper_and_the_roster_rules(phased):
    # the arms are the models table's declared dose views: the roster block 05 --step roster prints is read off the table (reproduce/lib/models.py
    # roster_block), and the production roster carries no front-door block (the front-door model is configured, not run). On the phased table both
    # kinds of row sit side by side: the demo (single-phase: not steered anywhere) and the synthetic phased row (steered everywhere, anchored on its u0)
    ec = _ec(); seen = 0
    for key in fdv.front_door_keys():
        for view in M.views(key):
            if not fdv.DOSE_VIEW_RE.match(str(view)):
                continue
            seen += 1
            expect = FR.steered_wave_view(view, key)
            assert expect is (not M.single_phase(key)) and expect is (key == PHASED)
            assert M.steered_wave_view(view, key) is expect
            assert fdv.front_door_dose_view(view, key) is expect
            assert prun.steered_wave_audit(_S(f"petri_full22_{key}_{view}")) is expect
            assert ec.u0_view_of(f"{key}_{view}") == (f"{key}_{M.baseline_view(key)}" if expect else None)
    assert seen >= 4, f"the table names only {seen} front-door dose arm(s)"
    # the paper targets' dose-named views: not steered waves anywhere (same-wave none anchor, the collision case)
    for key in PAPER:
        for v in PAPER_DOSES:
            assert not FR.steered_wave_view(v, key) and not M.steered_wave_view(v, key)


# ------------------------------------------------------------------------------------------------- (4) copies (3)-(5) unchanged by name

def test_the_three_view_name_grammars_are_unchanged_and_the_s_family_literal_appears_in_exactly_the_two_suffix_copies():
    assert FR.STEERED_WAVE_VIEW_RE.pattern == M.STEERED_WAVE_VIEW_RE.pattern == r"^s[0-9]+rl?$"
    mm = (REPO / "scripts" / "sdfbase" / "full22" / "misalign" / "make_misalign_manifest.py").read_text()
    assert 'STEERED_WAVE_VIEW_RE = re.compile(r"^s[0-9]+rl?$")' in mm
    helper = (REPO / "src" / "sisteer" / "front_door_views.py").read_text()
    assert not re.search(r"s\[0-9\]\+r", helper)                    # the helper duplicates no s-family grammar: the table decides


# ------------------------------------------------------------------------------------------------- (5) one table read per revision, none for a non-dose tag

def test_the_table_is_read_once_per_revision_and_never_for_a_tag_that_is_not_dose_shaped(monkeypatch):
    fdv._CACHE.clear(); calls = []; real = fdv.models_module
    monkeypatch.setattr(fdv, "models_module", lambda path=None: (calls.append(str(path)), real(path))[1])
    ec = _ec()
    tags = ([f"{KEY}_{v}" for v in FD_DOSES] + [f"{k}_{v}" for k in PAPER for v in PAPER_DOSES]
            + [f"{KEY}_s2rl", f"{KEY}_u0", f"{KEY}_battery", "qwen35_4b_it", "llama8b_main", "glm46vflash_s2p"])
    tags = (tags * 12)[:300]
    c0 = time.process_time()
    for t in tags:
        ec.u0_view_of(t); prun.steered_wave_audit(_S("petri_full22_" + t))
    cpu = time.process_time() - c0
    assert len(calls) == 1, calls                                   # one reader load + one table parse for 600 classifications
    # CPU time, not wall-clock: a per-call yaml reparse (the first cut: 0.7-1.9 s EACH) would cost hundreds of CPU-seconds here; the shared
    # box's load and a `nice 10` worker inflate wall-clock several-fold (3.7 s measured on the pre-run scratch) without touching this
    assert cpu < 2.0, f"600 classifications took {cpu:.2f}s of CPU on a warm table"
    # a tag that is not dose-shaped never opens the table: with the reader unreachable it still answers by name
    fdv._CACHE.clear()
    def refusing(path=None):
        raise AssertionError("the table was opened for a tag that is not dose-shaped")
    monkeypatch.setattr(fdv, "models_module", refusing)
    assert ec.u0_view_of(f"{KEY}_u0") is None and ec.u0_view_of("qwen35_4b_it") is None and ec.u0_view_of("llama8b_main") is None
    assert ec.u0_view_of(f"{KEY}_s2rl") == f"{KEY}_u0" and ec.u0_view_of(f"{KEY}_battery") is None
    assert not prun.steered_wave_audit(_S(f"petri_full22_{KEY}_u0")) and not prun.steered_wave_audit(_S(f"petri_full22_{KEY}_battery"))
    assert prun.steered_wave_audit(_S(f"petri_full22_{KEY}_s2rl")) and not prun.steered_wave_audit(_S("petri_full22_qwen35_4b_it"))
    with pytest.raises(AssertionError, match="not dose-shaped"):         # a dose-shaped tag DOES need the table
        ec.u0_view_of(f"{KEY}_d0p75_relu")


def test_a_changed_table_is_re_read_and_a_missing_table_is_refused_by_name(monkeypatch, tmp_path):
    tbl = Path(_phased_table(tmp_path / "t"))                                                      # the demo (single-phase) beside the synthetic phased row
    monkeypatch.setenv("SISTEER_MODELS_TABLE", str(tbl)); fdv._CACHE.clear(); calls = []; real = fdv.models_module
    monkeypatch.setattr(fdv, "models_module", lambda path=None: (calls.append(1), real(path))[1])
    assert fdv.front_door_dose_tag(f"{PHASED}_d0p75_relu") == PHASED and fdv.front_door_dose_tag(f"{PHASED}_d0p75") == PHASED and fdv.front_door_dose_tag("qwen35_4b_d0p1") is None
    assert fdv.front_door_dose_tag(f"{KEY}_d0p75_relu") is None                                     # the single-phase demo's dose rung: no front-door dose tag
    assert len(calls) == 1
    tbl.write_text(tbl.read_text() + "\n# a revision\n"); os.utime(tbl, ns=(time.time_ns() + 5_000_000_000,) * 2)   # a new mtime: the table changed
    assert fdv.front_door_dose_tag(f"{PHASED}_d0p75_relu") == PHASED and len(calls) == 2
    assert fdv.front_door_dose_tag(f"{PHASED}_d0p75") == PHASED and len(calls) == 2
    monkeypatch.setenv("SISTEER_MODELS_TABLE", str(tmp_path / "absent.yaml"))
    with pytest.raises(fdv.FrontDoorTableMissing, match="models table is absent"):
        fdv.front_door_dose_tag(f"{KEY}_d0p75_relu")
    with pytest.raises(fdv.FrontDoorTableMissing):
        _ec().u0_view_of(f"{KEY}_d0p75_relu")
    assert _ec().u0_view_of(f"{KEY}_s2rl") == f"{KEY}_u0" and _ec().u0_view_of(f"{KEY}_u0") is None      # by name, no table
    fdv._CACHE.clear()
