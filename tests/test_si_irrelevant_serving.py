"""si_irrelevant is servable.

The off-target control SI had no harness translation, so
`harness_cell_id` refused every si_irrelevant cell and the sweep's served
path silently lost 4 of 20 wave conditions. A control that cannot run on
the same engine as the arms it controls for is not a control.

Scope is deliberately SERVING-only: routes come from
`build_registry(ddir, si_ids=...)`, i.e. the CONFIG's si_ids, so the cells
already existed — only the translation was missing. `PETRI_SI_IDS` is
untouched so no frozen audit hash moves; auditing si_irrelevant is a
separate decision needing behaviors + prefix hashes.
"""
import pytest

from sisteer.capeval import manifest as cm
from sisteer.capeval.petri import registry_adapter as ra

NON_LORA_WAVE_CELLS = [
    "none",
    "prompted__safety_refusal", "prompted__si_constitution",
    "prompted__si_irrelevant",
    "safety_refusal__psr_kl_all", "si_constitution__psr_kl_all",
    "si_irrelevant__psr_kl_all",
]


@pytest.mark.parametrize("cell", NON_LORA_WAVE_CELLS)
def test_every_non_lora_wave_cell_translates(cell):
    """All 7 per-model non-LoRA conditions resolve to a harness id."""
    assert ra.harness_cell_id(cell)


def test_si_irrelevant_specifically_resolves():
    assert ra.harness_cell_id("prompted__si_irrelevant") == "prompted__irrelevant"
    assert ra.harness_cell_id("si_irrelevant__psr_kl_all") == "si_irrelevant__psr_kl_all"
    assert ra.HARNESS_TO_TRAINTEST_SI["irrelevant"] == "si_irrelevant"
    assert ra.TRAINTEST_TO_HARNESS_SI["si_irrelevant"] == "irrelevant"


def test_audit_si_vocabulary_was_widened_deliberately():
    """SUPERSEDED, and the history matters. This test originally asserted
    the OPPOSITE -- that PETRI_SI_IDS must stay ("safety","constitution")
    -- because widening it was assumed to move petri_run_key /
    prompted_prefix_hashes and strand results. That assumption was wrong
    in two ways, established 2026-08-28:

    1. The vocabulary constants are NOT hash inputs. Every petri hash
       reads spec CONTENT (cells, prompted_prefix_hashes, behavior bytes),
       so existing specs' hashes are byte-identical either way.
    2. Nothing could be stranded regardless: the committed manifest still
       carries UNFROZEN scenario_bundle_hash / serving_impl_commit, and
       hash computation refuses UNFROZEN, so no run key was ever derivable
       from it. Petri has never run.

    And the narrow version was not merely over-cautious, it was BROKEN:
    serving `prompted__si_irrelevant` computes a prompted-prefix hash,
    which consults this list -- so the off-target control's prompted arm
    refused at stand-up (job 232263_3) while its fitted arm worked.
    """
    assert "irrelevant" in cm.PETRI_SI_IDS
    assert "irrelevant" in cm.PETRI_SI_VARIANT_DIRS
    # the two SIs that were always there must not have moved
    assert {"safety", "constitution"} <= set(cm.PETRI_SI_IDS)
