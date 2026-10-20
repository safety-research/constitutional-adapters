"""si_constitution_full is servable.

The FULL constitution — 39 sections, 28,977 words, 182,219 chars — joins the
registry as its own instruction (`si_constitution_full`, the vendored
`src/sisteer/data/constitution_full.md` byte-for-byte, sha256 asserted at
import) so the wave can serve it as a prompted baseline
(`prompted__si_constitution_full`): the text as the system prompt, no fitted
object, the whole prefix hash-verified per request (H8). `si_constitution`
stays what it is — the compressed EXCERPT — and is never described as the
constitution.

The `si_irrelevant` precedent (tests/test_si_irrelevant_serving.py, 2026-08-28):
every layer moves together — the registry, the harness translation, the
promptable vocabulary, the variant behavior dirs, the generator's tables — or the
standup refuses the cell. This test pins each seam to the registry's bytes.
"""
import hashlib
import sys
from pathlib import Path

import pytest

from sisteer.capeval import manifest as cm
from sisteer.capeval.petri import registry_adapter as ra
from sisteer.capeval.petri import scenarios as psc
from sisteer.data import instructions as ins

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "sdfbase"))
sys.path.insert(0, str(REPO / "scripts" / "sdfbase" / "full22"))

TEXT = ins.BY_ID["si_constitution_full"].text
SHA16 = "82f88c5532e4939e"


def test_the_full_constitution_is_registered_from_the_vendored_bytes():
    ins_ = ins.BY_ID["si_constitution_full"]
    assert ins_.category == ins.BY_ID["si_constitution"].category == "safety"   # NOT corpus_cell
    assert ins_.anti_text is None and ins_.checker == "external"
    data = ins.CONSTITUTION_FULL_PATH.read_bytes()
    assert hashlib.sha256(data).hexdigest() == ins.CONSTITUTION_FULL_SHA256
    assert data.decode("utf-8") == TEXT                    # the text IS the file
    assert cm.sha256_hex16(TEXT.encode("utf-8")) == SHA16 == ins.CONSTITUTION_FULL_SHA256[:16]
    assert len(TEXT.split()) == ins.CONSTITUTION_FULL_WORDS == 28977
    assert len(TEXT) == 182219 and TEXT.endswith("\n") and not TEXT.endswith("\n\n")
    # the excerpt is a different, much shorter text
    assert len(ins.BY_ID["si_constitution"].text) == 30952
    assert ins.resolve_si("si_constitution_full") is ins_


def test_an_edited_vendored_text_refuses_at_load(tmp_path, monkeypatch):
    bad = tmp_path / "constitution_full.md"
    bad.write_bytes(TEXT.encode("utf-8") + b"x")
    monkeypatch.setattr(ins, "CONSTITUTION_FULL_PATH", bad)
    with pytest.raises(RuntimeError, match="sha256 .* != the registered"):
        ins._load_constitution_full()


def test_the_harness_translation_and_the_promptable_vocabulary():
    assert ra.HARNESS_TO_TRAINTEST_SI["constitution_full"] == "si_constitution_full"
    assert ra.TRAINTEST_TO_HARNESS_SI["si_constitution_full"] == "constitution_full"
    assert ra.harness_cell_id("prompted__si_constitution_full") == "prompted__constitution_full"
    assert ra.traintest_cell_id("prompted__constitution_full") == "prompted__si_constitution_full"
    assert "constitution_full" in cm.PETRI_PROMPTABLE_SI_IDS and "constitution_full" in cm.PETRI_SI_IDS
    assert cm.PETRI_SI_VARIANT_DIRS["constitution_full"] == "PROMPTED_CONSTITUTION_FULL"
    assert psc.SI_VARIANT_DIRS == cm.PETRI_SI_VARIANT_DIRS
    # the historical three did not move
    assert cm.PETRI_PROMPTABLE_SI_IDS[:3] == ("safety", "constitution", "irrelevant")
    # a prompted cell over it parses as a prompted baseline (never OBJECT-ONLY)
    from sisteer.capeval.petri import defense_cells as dc
    p = dc.parse_cell_id("prompted__constitution_full")
    assert (p.cell_class, p.si_id) == ("prompted", "constitution_full")


def test_the_variant_dirs_carry_the_exact_bytes_for_every_behavior():
    root = REPO / "configs" / "capeval" / "petri"
    for b in cm.PETRI_BEHAVIORS:
        front, body = psc.parse_behavior_md(
            (root / b / "PROMPTED_CONSTITUTION_FULL" / "BEHAVIOR.md").read_bytes())
        assert front["target_sysprompt_prefix"] == TEXT, b
        cfront, cbody = psc.parse_behavior_md((root / b / "BEHAVIOR.md").read_bytes())
        assert body == cbody and front["modality"] == cfront["modality"], b
        assert front["num_scenarios"] == cfront["num_scenarios"], b
    # the generator reproduces them byte-for-byte (never hand-edited)
    import write_prompted_variant_dirs as wpv
    assert wpv.main(["--si", "constitution_full", "--check"]) == 0
    # and behavior_files reads every variant on disk without refusing
    bf = psc.behavior_files(root, cm.PETRI_BEHAVIORS)
    assert {f"{b}/PROMPTED_CONSTITUTION_FULL" for b in cm.PETRI_BEHAVIORS} <= set(bf)


def test_the_generator_tables_name_it_with_the_registry_hash():
    import make_full22_configs as gen
    assert gen.TT_TO_HARNESS_SI["si_constitution_full"] == "constitution_full"
    assert gen.PREFIX_HASHES["constitution_full"] == SHA16
    assert gen.harness_cell_id("prompted__si_constitution_full") == "prompted__constitution_full"
    # the excerpt's hash is untouched, and distinct
    assert gen.PREFIX_HASHES["constitution"] == "21e06f482a6559b3" != SHA16
    # H10: prompted_prefix_hashes is hashed WHOLE, so the new key rides only the manifests that
    # serve the cell — every full22 manifest whose legs already ran is byte-identical (no
    # completed audit is re-keyed), and the generator's --check agrees
    assert gen.PREFIX_HASHES_ALWAYS == ("safety", "constitution", "irrelevant")
    for path in sorted((REPO / "configs" / "capeval" / "manifests").glob("petri_full22_*.yaml")):
        spec = cm.load_petri_spec(path)
        if "prompted__constitution_full" in spec.cells:
            assert spec.prompted_prefix_hashes["constitution_full"] == SHA16, path.name
        else:
            assert "constitution_full" not in spec.prompted_prefix_hashes, path.name
            assert set(spec.prompted_prefix_hashes) == set(gen.PREFIX_HASHES_ALWAYS), path.name


def test_the_full_constitution_is_prompted_only_never_a_class_a_or_stacked_cell():
    """A prompted baseline with no fitted object: the vocabulary refuses `si_constitution_full__<m>`
    and `stacked__constitution_full__<m>` (the mirror of the object-only refusal), and the canonical
    Petri matrix expects the prompted cell and NO Class A cells for it — so the audit world does not
    grow by seventeen cells that can never exist."""
    from sisteer.capeval.petri import defense_cells as dc
    from sisteer.capeval.petri import petri_stats
    assert cm.PETRI_PROMPTED_ONLY_SI_IDS == ("constitution_full",)
    assert "constitution_full" not in cm.PETRI_CLASS_A_SI_IDS and "constitution" in cm.PETRI_CLASS_A_SI_IDS
    # the promptable SIs split into Class A and prompted-only; the object-only SIs (main's corpus pseudo-instructions) sit
    # beside them in PETRI_SI_IDS and never under a promptable method
    assert set(cm.PETRI_CLASS_A_SI_IDS) | set(cm.PETRI_PROMPTED_ONLY_SI_IDS) == set(cm.PETRI_PROMPTABLE_SI_IDS)
    assert set(cm.PETRI_PROMPTABLE_SI_IDS) | set(cm.PETRI_OBJECT_ONLY_SI_IDS) == set(cm.PETRI_SI_IDS)
    assert not set(cm.PETRI_CLASS_A_SI_IDS) & set(cm.PETRI_OBJECT_ONLY_SI_IDS)
    for bad in ("si_constitution_full__psr_kl_all", "stacked__constitution_full__psr_kl_all"):
        with pytest.raises(dc.DefenseCellError, match="PROMPTED-ONLY"):
            dc.parse_cell_id(bad)
    ids = petri_stats.canonical_cell_ids()
    assert "prompted__constitution_full" in ids
    assert not any(c.startswith("si_constitution_full__") for c in ids)
    # the universe = none + one prompted cell per promptable SI + a Class A cell per (promptable method, Class A SI) + the
    # object-only cells + Class B: constitution_full adds EXACTLY its one prompted cell and NO Class A cells (the object-only
    # methods joined main's vocabulary after the 2026-08-28 count of 92, so the total is the formula, not a literal)
    expected = (1 + len(cm.PETRI_PROMPTABLE_SI_IDS) + len(dc.PROMPTABLE_CLASS_A_METHODS) * len(cm.PETRI_CLASS_A_SI_IDS)
                + len(petri_stats.object_only_cell_ids()) + len(dc.CLASS_B_IDS))
    assert len(ids) == len(set(ids)) == expected
    without_full = (1 + (len(cm.PETRI_PROMPTABLE_SI_IDS) - 1) + len(dc.PROMPTABLE_CLASS_A_METHODS) * len(cm.PETRI_CLASS_A_SI_IDS)
                    + len(petri_stats.object_only_cell_ids()) + len(dc.CLASS_B_IDS))
    assert expected == without_full + 1      # one prompted baseline, never a method's worth of Class A cells
