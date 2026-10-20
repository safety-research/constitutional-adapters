"""build_strength.py's staged-id grammar per (target, family). The eff-batch-64 instruct-direct refits
(2026-09-13) give gemma4e4b / qwen35_4b `itdoc64__` / `derived_itdoc64__` PARENTS and `d<dose>it64__` COPIES,
served by the unchanged `it_d<dose>` views — never the `d<dose>it__` ids whose batch-8 shas historical stores carry.
llama8b and the base-lane targets keep the `d<dose>it__` / `d<dose>__` grammar byte-for-byte; the expectations
below are LITERAL strings, not calls back into the module, so a change to the grammar has to be made here too.

Synthetic bank only (tmp_path; empty object.pt files stand in for objects — enumerate_bank reads names, not bytes)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STRENGTH = REPO / "scripts" / "sdfbase" / "full22" / "strength"


def _load():
    spec = importlib.util.spec_from_file_location("build_strength", STRENGTH / "build_strength.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bs = _load()
LORA = "constv2-mix-full__lora_r4_ce"
SUB = "constv2-mix-full__lora_r4_ce_minus_ctl"
CTL = "constv2ctl2-full__lora_r4_ce"       # the CONTROL LoRA: the base->instruct sweep's third object
BASE3 = {LORA: "base", SUB: "derived", CTL: "base"}       # the base family on every instruct-lane target


def test_cell_ids_are_the_three_lora_objects():
    assert (bs.LORA, bs.SUB, bs.CTL) == (LORA, SUB, CTL) and set(bs.OBJECT_TYPES) == {LORA, SUB, CTL}


def test_it64_grammar_for_gemma_and_qwen_only():
    for t in ("gemma4e4b", "qwen35_4b"):
        assert bs.lane_and_view("it", 0.25, t) == ("d0p25it64", "it_d0p25"), t
        assert bs.lane_and_view("it", -1.0, t) == ("dm1it64", "it_dm1"), t
        assert bs.FAMILIES[t] == {"it": {LORA: "itdoc64", SUB: "derived_itdoc64"}, "main": BASE3}, t   # + the base family
        assert (t, SUB) not in bs.ABSENT_REASON, t
    # the Llama it family and the family DEFAULT (no target) keep the `it` suffix
    assert bs.lane_and_view("it", 0.25, "llama8b") == ("d0p25it", "it_d0p25")
    assert bs.lane_and_view("it", 0.25) == ("d0p25it", "it_d0p25")
    assert bs.FAMILIES["llama8b"] == {"it": {LORA: "itdoc", SUB: "derived_itdoc"}, "main": BASE3}
    for t in ("nemotron3nano4b", "glm46vflash", "llama8b"):
        assert bs.lane_and_view("main", 0.25, t) == ("d0p25", "d0p25"), t
        assert bs.lane_and_view("main", 1.5, t) == ("d1p5", "d1p5"), t
    # the view is target-independent, so the roster's it_d<dose> keys did not move
    for v, d in (("it_d0p25", 0.25), ("it_dm1", -1.0), ("d1p5", 1.5), ("it_d0p5", 0.5)):
        assert bs.dose_of_view(v) == d


def _bank(tmp_path: Path, present: list[tuple[str, str]]) -> Path:
    bank = tmp_path / "bank"
    for t in bs.FAMILIES:
        (bank / t / "cells").mkdir(parents=True, exist_ok=True)
    for target, pid in present:
        d = bank / target / "cells" / pid
        d.mkdir(parents=True)
        (d / "object.pt").write_bytes(b"synthetic")
    return bank


def test_enumerate_bank_pins_every_targets_staged_ids(tmp_path):
    """Every declared parent present: the full (target, copy id) -> (parent id, view, scale) table, literally."""
    present = [(t, f"{lane}__{cid}") for t, fams in bs.FAMILIES.items() for fam in fams.values() for cid, lane in fam.items()]
    found, missing = bs.enumerate_bank(_bank(tmp_path, present), list(bs.FAMILIES), [-1.0, 0.25])
    # the instruct-direct families declare no control-LoRA parent (the control is the BASE family's third object):
    # those rows are reported absent by construction — never built, never silently dropped
    assert {(r["target"], r["family"], r["cell_id"], r["why"]) for r in missing} == (
        {(t, "it", CTL, f"no it-family parent lane for {CTL}") for t in ("gemma4e4b", "qwen35_4b", "llama8b")}
        | {(t, "main", CTL, f"no main-family parent lane for {CTL}") for t in ("nemotron3nano4b", "glm46vflash")})   # not in the sweep
    assert len(missing) == 5 * 2       # x the two doses
    rows = {(r["target"], r["staged_id"]): (r["parent_staged_id"], r["view"], r["scale"]) for r in found}
    want = {}
    for t in ("gemma4e4b", "qwen35_4b"):                       # it family, it64 copies, itdoc64 parents
        want.update({
            (t, f"dm1it64__{LORA}"): (f"itdoc64__{LORA}", "it_dm1", -1.0),
            (t, f"dm1it64__{SUB}"): (f"derived_itdoc64__{SUB}", "it_dm1", -1.0),
            (t, f"d0p25it64__{LORA}"): (f"itdoc64__{LORA}", "it_d0p25", 0.25),
            (t, f"d0p25it64__{SUB}"): (f"derived_itdoc64__{SUB}", "it_d0p25", 0.25),
        })
    for t in ("nemotron3nano4b", "glm46vflash", "llama8b", "gemma4e4b", "qwen35_4b"):    # main family: UNCHANGED grammar, five targets
        want.update({
            (t, f"dm1__{LORA}"): (f"base__{LORA}", "dm1", -1.0),
            (t, f"dm1__{SUB}"): (f"derived__{SUB}", "dm1", -1.0),
            (t, f"d0p25__{LORA}"): (f"base__{LORA}", "d0p25", 0.25),
            (t, f"d0p25__{SUB}"): (f"derived__{SUB}", "d0p25", 0.25),
        })
    for t in ("llama8b", "gemma4e4b", "qwen35_4b"):            # the base family's CONTROL LoRA: three targets, two doses
        want.update({(t, f"dm1__{CTL}"): (f"base__{CTL}", "dm1", -1.0), (t, f"d0p25__{CTL}"): (f"base__{CTL}", "d0p25", 0.25)})
    want.update({                                              # llama's it family: UNCHANGED grammar
        ("llama8b", f"dm1it__{LORA}"): (f"itdoc__{LORA}", "it_dm1", -1.0),
        ("llama8b", f"dm1it__{SUB}"): (f"derived_itdoc__{SUB}", "it_dm1", -1.0),
        ("llama8b", f"d0p25it__{LORA}"): (f"itdoc__{LORA}", "it_d0p25", 0.25),
        ("llama8b", f"d0p25it__{SUB}"): (f"derived_itdoc__{SUB}", "it_d0p25", 0.25),
    })
    assert rows == want
    assert len(found) == len(want) == 24 + 8 + 6   # the 24 of 2026-09-13 + gemma / qwen base family (2 x 2 x 2) + the control on 3 targets x 2 doses


def test_missing_itdoc64_subtraction_parent_is_reported_as_absent_dir(tmp_path):
    """The SUB cell is no longer absent BY CONSTRUCTION on gemma / qwen: with only the constitution parent in the
    bank it is reported as a parent dir the bank lacks, under the `derived_itdoc64__` id a later stage will fill."""
    for t in ("gemma4e4b", "qwen35_4b"):
        found, missing = bs.enumerate_bank(_bank(tmp_path / t, [(t, f"itdoc64__{LORA}")]), [t], [0.25])
        assert [(r["staged_id"], r["parent_staged_id"]) for r in found] == [(f"d0p25it64__{LORA}", f"itdoc64__{LORA}")]
        # beside it: the it family's control row (no parent lane by construction) and the BASE family's three rows, whose
        # parents this synthetic bank does not hold
        assert sorted((r["staged_id"], r["parent_staged_id"], r["why"]) for r in missing) == sorted([
            (f"d0p25it64__{SUB}", f"derived_itdoc64__{SUB}", "parent dir absent"),
            (f"d0p25it64__{CTL}", None, f"no it-family parent lane for {CTL}"),
            (f"d0p25__{LORA}", f"base__{LORA}", "parent dir absent"),
            (f"d0p25__{CTL}", f"base__{CTL}", "parent dir absent"),
            (f"d0p25__{SUB}", f"derived__{SUB}", "parent dir absent")])


def test_parent_root_reads_the_itdoc64_parent_by_its_staged_id(tmp_path):
    """The bake path: a pulled read-only parent dir NAMED by the ruled staged id, its authoritative origin recorded."""
    proot = tmp_path / "pull"
    (proot / f"itdoc64__{LORA}").mkdir(parents=True)
    (proot / f"itdoc64__{LORA}" / "object.pt").write_bytes(b"synthetic")
    origin = {f"itdoc64__{LORA}": "TESTHOST:/authoritative/bank/constv2-mix-full__lora_r4_ce"}
    found, missing = bs.enumerate_bank(_bank(tmp_path, []), ["qwen35_4b"], [0.5, 1.5], parent_root=proot, origins=origin)
    assert [(r["staged_id"], r["view"], r["parent_dir"], r["parent_origin"], r["anchor"]) for r in found] == [
        (f"d0p5it64__{LORA}", "it_d0p5", str(proot / f"itdoc64__{LORA}"), origin[f"itdoc64__{LORA}"], False),
        (f"d1p5it64__{LORA}", "it_d1p5", str(proot / f"itdoc64__{LORA}"), origin[f"itdoc64__{LORA}"], False)]
    assert {r["staged_id"] for r in missing} == {f"d0p5it64__{SUB}", f"d1p5it64__{SUB}",
                                                  f"d0p5it64__{CTL}", f"d1p5it64__{CTL}",          # no it-family control lane
                                                  *[f"d{tok}__{c}" for tok in ("0p5", "1p5") for c in (LORA, SUB, CTL)]}   # base parents absent here
