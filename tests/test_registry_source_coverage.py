"""Source -> adapter coverage: every legacy source the paper declares -- the round's committed source list
(`configs/registry/sources_r2_v1.yaml`, the paper's watermarks keys at the pin) and every `source_legacy` string in the
roster -- is covered by exactly one adapter's `covers_sources` (or by several with distinct priorities), and the status
page carries a live "Sources without an adapter" section. Before this module nothing enumerated the declared set: the two
`*-superseded` EM panels the paper declares were ingested under the `_bl` stem and named as sources by nothing (26 of 28),
found only by the acceptance audit's one-off probe. The known-bad case is reconstructed below by removing those two
names from the covered set and asserting the check reports them.
"""
from __future__ import annotations

import os

import pytest
import yaml

from sisteer.registry import adapters as ADP
from sisteer.registry import harvest as H
from sisteer.registry import sources as SRC
from sisteer.registry.adapters import _items_common as IC
from sisteer.registry import status
from sisteer.registry import synth
from sisteer.registry.adapters import em_misalign as EM
from sisteer.registry.adapters import lineage_common as LC
from sisteer.registry.adapters import multiturn_grades as MG
from sisteer.registry.adapters import strength_mtdiff as SM
from sisteer.registry.adapters import strength_ladder as SL

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DECLARED = os.path.join(REPO, "configs", "registry", "sources_r2_v1.yaml")
ROSTER = os.path.join(REPO, "configs", "registry", "roster_r2_v1.yaml")
EM_SUPERSEDED = ("em/agentic/gemma4e4b_bl__lam3-superseded", "em/agentic/llama8b_bl__r0p1-superseded")
N_SOURCES = 61          # 28 at the f33dca5d pin + `arditi_allpos` (the paper, 2026-09-17) + the PROVISIONAL registry-declared names (the two Qwen organism panels ... the appendix BCB / BFCL pair, 2026-09-23; the shared-standup landing-3 pair, 2026-10-01)
                        # + `strength_mtdiff_s1`, the PROVISIONAL spelling of the multi-turn dose ladder (2026-09-18)
                        # + `arditi_vendor_mt`, the PROVISIONAL spelling of its VENDOR-PRESET re-run of the Arditi all-positions rungs (2026-09-20)
                        # + `seedrep_cluster`, the PROVISIONAL spelling of the seed-replicate statics / MMLU mirror from cluster B (2026-09-20)
                        # + `seedrep_petri`, the PROVISIONAL spelling of the Petri seed-1 store (2026-09-21)
                        # + `gsm8k_seedrep`, the PROVISIONAL spelling of the three-seed family's GSM8K stores from cluster A (2026-09-21)
                        # + `orxs_seedrep`, the PROVISIONAL spelling of the seed-replicate over-refusal controls from cluster A (2026-09-21)
                        # + `seedrep_battery`, the campaign's own spelling on the premint-queued seed-1 static runs, folded by the legacy-store reader (2026-09-21)
                        # + `misalign_seedrep`, the PROVISIONAL spelling of the three-seed agentic-misalignment stores from cluster A, em_misalign's second root (2026-09-21)
                        # + `mmlu_full`, the PROVISIONAL spelling of the Qwen MMLU read at the full 1,531-item width from cluster A (2026-09-21)
                        # + `misalign_16k`, the PROVISIONAL spelling of the 16,384-step agentic-misalignment roots from cluster A, em_misalign's third root (2026-09-21)
                        # + `misalign_ladder`, the PROVISIONAL spelling of the completion-cap MERGES of the agentic ladder, em_misalign's fourth root (2026-09-22)
                        # + `gsm8k_paired_allmodels`, the PROVISIONAL spelling of the paired_allmodels GSM8K document (the paired-write difference ReLU on
                        #   gemma / llama / nemotron + nemotron's constitution ReLU; the GSM8K reader's second source, 2026-09-22)
                        # + `bon_native4096:llama8b_relu_paired`, the PROVISIONAL spelling of the Llama paired-write native 4,096 best-of-N grades (2026-09-22)
                        # + `misalign_paired_allmodels`, the PROVISIONAL spelling of the paired-write difference-ReLU agentic roots of gemma / llama / nemotron
                        #   (one steered key, no none, read by path), em_misalign's fifth root (2026-09-22)
                        # + `static_padding`, the PROVISIONAL spelling of the PADDED static single-shot conditions on qwen35_4b, filed on padded instruments
                        #   (migration 0012: the pad is instrument identity, never a run column; 2026-09-23)
                        # + `math500_seedrep`, the PROVISIONAL spelling of the three-seed family's MATH-500 stores from cluster B (the seed-replicate
                        #   run, the paired tags' composed object and the Arditi all-positions object included; the MATH-500 sibling of `gsm8k_seedrep`, 2026-09-23)
                        # + `bigcodebench_seedrep` and `bfcl_seedrep`, the PROVISIONAL spellings of the three-seed family's BigCodeBench-200 and BFCL-200
                        #   stores from the Modal volume (the seed-replicate run; one store holds both tasks, each reader owns its own copy;
                        #   the paired tags' composed object and the Arditi all-positions object included; 2026-09-23)
                        # + `ifeval_seedrep`, the PROVISIONAL spelling of the three-seed family's IFEval stores from the Modal volume (the seed-replicate
                        #   run, 30 rows at the completion cap of a 2,048 -> 4,096 -> 8,192 ladder, the merge the reader's; the paired tags' composed object
                        #   and the Arditi all-positions object included; 2026-09-23)
                        # + `bigcodebench_full` and `bfcl_full`, the PROVISIONAL spellings of the FULL-suite BigCodeBench (all 1,140 tasks) and BFCL (all 3,981
                        #   items) stores from the Modal volume, on their OWN item sets (bigcodebench_full_1140 / bfcl_full_3981) and instruments -- one store holds both
                        #   full tasks, each reader owns its own copy; the item set attested per store (2026-09-24)
                        # + `ctl_neg1`, the PROVISIONAL spelling of the control-negative battery (the batch-64 seed CONTROL objects at -1 on qwen35_4b,
                        #   steered-only standups anchored on the seed standups' `none`; two readers at distinct priorities + multiturn_grades' declared roots; 2026-09-25)
                        # + `bigcodebench_full_shared` and `bfcl_full_shared`, the PROVISIONAL spellings of the SHARED-STANDUP COMPARISON stores (every object of a
                        #   model row on ONE engine, compared to the solo full-suite stores; the solo readers parametrized, `_bcbw_common.SharedSpec`; a comparison source
                        #   pair, never in the paper frame; 2026-09-28)
                        # + `bigcodebench_full_shared_l2` and `bfcl_full_shared_l2`, the PROVISIONAL spellings of LANDING 2 of those comparison stores (the same tuples
                        #   driven again by another harness -- K = 2 drivers per row + a CPU verification pass -- published as SEPARATE sources so no run id collides
                        #   with landing 1's; the `_l2` producer key, `_bcbw_common.SharedSpec.harness_variant`; 2026-09-30)
                        # + `bigcodebench_full_shared_l3` and `bfcl_full_shared_l3`, the PROVISIONAL spellings of LANDING 3 of those comparison stores (the gemma row's four
                        #   tuples driven once more by another harness for the fan-out test, published as SEPARATE sources so no run id collides with
                        #   landing 1's or landing 2's; the `_l3` producer key, the same SharedSpec knob; 2026-10-01)
#: the additions the PAPER declared after the pin (each names the commit it was read at) and the ones the REGISTRY declares provisionally, which
#: the paper has not declared at all (`paper_build_commit: null`, `provisional: true`): a pin would be a consistent-looking stamp for a fact it
#: does not cover.
#: the newest names sit third and fourth, after the two organism panels: the tests below slice this tuple by `[:2]` and by position from the END,
#: so a new name goes here, never appended (an appended name shifted six of them on 2026-09-23); the positional asserts of the names it displaces move with it
PROVISIONAL_ADDITIONS = ("em/organisms/qwen35_4b_org", "em/organisms/qwen35_4b_org_vendor", "bigcodebench_full_shared_l3", "bfcl_full_shared_l3",
                         "bigcodebench_full_shared_l2", "bfcl_full_shared_l2", "bigcodebench_full_shared", "bfcl_full_shared",
                         "ctl_neg1", "bigcodebench_full", "bfcl_full",
                         "bigcodebench_allmodels", "bfcl_allmodels",
                         "ifeval_seedrep", "math500_seedrep", "bigcodebench_seedrep", "bfcl_seedrep", "static_padding",
                         "strength_mtdiff_s1", "arditi_vendor_mt", "seedrep_cluster",
                         "seedrep_petri", "orxs_seedrep", "mmlu_full", "gsm8k_seedrep", "seedrep_battery", "misalign_seedrep", "misalign_16k", "misalign_ladder",
                         "gsm8k_paired_allmodels", "bon_native4096:llama8b_relu_paired", "misalign_paired_allmodels")


@pytest.fixture(scope="module")
def covered():
    return SRC.covered_sources(ADP.load_all())


# --------------------------------------------------------------------------------------------
# the declared set
# --------------------------------------------------------------------------------------------

def test_committed_source_list_is_the_pinned_watermarks_set():
    with open(DECLARED, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    names = [r["source"] for r in data["sources"]]
    assert len(names) == data["meta"]["n_sources"] == N_SOURCES and len(set(names)) == N_SOURCES
    assert names == sorted(names), "keep the list sorted: a diff against the paper's file is then a diff of names"
    assert data["meta"]["paper_build_commit"] == "f33dca5d99488537737538f5b89b8254675653ea"
    assert set(EM_SUPERSEDED) <= set(names) and "strength_lane_base2it_native_mmlu" in names
    # A POST-PIN ADDITION CARRIES ITS OWN PAPER COMMIT: the global `paper_build_commit` is the commit the bulk was read at and is not
    # evidence about a name the paper added later, so each addition names the commit it WAS read at (the paper keeps moving).
    adds = {a["source"]: a for a in data["meta"]["additions"]}
    assert "arditi_allpos" in adds and adds["arditi_allpos"]["paper_build_commit"] != data["meta"]["paper_build_commit"]
    paper_declared = {s: a for s, a in adds.items() if s not in PROVISIONAL_ADDITIONS}
    assert all(a["paper_build_commit"] and a["read_utc"] and a["why"] for a in paper_declared.values())
    # a PROVISIONAL addition is one the registry declares and the paper does not: no paper commit (none covers it), said so explicitly
    assert set(PROVISIONAL_ADDITIONS) <= set(adds)
    for s_ in PROVISIONAL_ADDITIONS:
        a_ = adds[s_]
        assert a_["paper_build_commit"] is None and a_["provisional"] is True and a_["declared_by"].startswith("registry")
        assert a_["read_utc"] and a_["why"] and "PROVISIONAL" in a_["why"]
    assert set(adds) <= set(names)
    srcs, label = SRC.read_declared_file(DECLARED)
    assert set(srcs) == set(names) and srcs["r2_export"] == "the authors" and label.endswith("(paper watermarks @ f33dca5d)")


def test_source_tokens_normalise_the_roster_spellings():
    assert SRC.source_tokens("r2_export (Llama pass, job 103624; the second pass is SECOND_PASS_ROWS)") == (["r2_export"], [])
    assert SRC.source_tokens("em/agentic/{gemma4e4b,qwen35_4b,llama8b_main}") == (["em/agentic/gemma4e4b", "em/agentic/qwen35_4b", "em/agentic/llama8b_main"], [])
    assert SRC.source_tokens("em/agentic/<family>_bl") == ([], ["em/agentic/<family>_bl"])          # a family, not a source
    assert SRC.source_tokens(["bcb_bfcl_200", "bcb_bfcl_200_widen"]) == (["bcb_bfcl_200", "bcb_bfcl_200_widen"], [])
    assert SRC.source_tokens({"default": "evalcheck/math500", "not_canonical": "gsm8k_controls512 (relabelled; not in EVALS)"}) == (["evalcheck/math500"], [])
    assert SRC.source_tokens({"default": None}) == ([], []) and SRC.source_tokens("") == ([], [])


def test_roster_sources_are_labelled_by_path_and_every_one_is_in_the_committed_list():
    srcs, placeholders = SRC.roster_sources(ROSTER)
    assert srcs and all(w.startswith("roster_r2_v1.yaml ") and w.endswith(".source_legacy") for ws in srcs.values() for w in ws)
    assert any("evals.gsm8k" in w for w in srcs["evalcheck/gsm8k_paired200"])
    assert placeholders and all("<family>" in p for p in placeholders)
    committed, _ = SRC.read_declared_file(DECLARED)
    assert set(srcs) <= set(committed), sorted(set(srcs) - set(committed))     # the roster names no source the paper's file lacks


def test_declared_union_reads_both_committed_inputs_and_reports_a_missing_one_as_an_error(tmp_path):
    dec = SRC.declared_sources(declared_path=DECLARED, roster_path=ROSTER)
    assert dec["errors"] == [] and len(dec["inputs"]) == 2 and len(dec["declared"]) == N_SOURCES
    assert all(len(w) >= 1 for w in dec["declared"].values())
    bad = SRC.declared_sources(declared_path=str(tmp_path / "absent.yaml"), roster_path=ROSTER)
    assert len(bad["errors"]) == 1 and "absent.yaml" in bad["errors"][0] and len(bad["declared"]) < N_SOURCES
    # a live paper checkout adds its watermarks keys under their own label
    lib = tmp_path / "paper" / "paperlib"
    lib.mkdir(parents=True)
    (lib / "lane_watermarks.json").write_text('{"sources": {"brand_new_source": {"owner": "Nobody"}, "r2_export": {"owner": "Scale"}}}')
    live = SRC.declared_sources(declared_path=DECLARED, roster_path=ROSTER, paper_checkout=str(tmp_path))
    assert "brand_new_source" in live["declared"] and live["declared"]["brand_new_source"] == [f"paper checkout watermarks ({tmp_path})"]
    assert len(live["declared"]["r2_export"]) >= 2


# --------------------------------------------------------------------------------------------
# the covered set and the check
# --------------------------------------------------------------------------------------------

def test_every_declared_source_has_exactly_one_adapter(covered):
    dec = SRC.declared_sources(declared_path=DECLARED, roster_path=ROSTER)
    out = SRC.coverage(dec["declared"], covered)
    assert out["without_adapter"] == [], out["without_adapter"]
    assert out["claimed_without_priority"] == [], out["claimed_without_priority"]
    assert out["n_declared"] == N_SOURCES
    # each declared source is claimed by ONE adapter -- except the ONE campaign source two legacy-store readers fold onto the SAME premint's
    # queued rows (`seedrep_battery`: the statics reader and the served-controls reader, 2026-09-21; disjoint evals, so no measurement key is
    # ever claimed twice), declared here and distinguished by priority as the status page's rule requires ("or by several with distinct priorities")
    # -- and `ctl_neg1` (2026-09-25): the control-negative battery, one mirror root, three readers of disjoint evals -- the multi-turn
    # grades through the dose-ladder reader's subclass at the default priority, the over-refusal controls through the shared store reader's
    # subclass at the generic priority, the statics through the sampled reader's subclass at its PAPER_SOURCE priority
    multi = {s: sorted(set(c)) for s, c in covered.items() if s in dec["declared"] and len(set(c)) > 1}
    assert multi == {"ctl_neg1": [("ctl_neg1_controls", IC.PRIORITY_GENERIC_READER), ("ctl_neg1_multiturn", 0), ("ctl_neg1_static", IC.PRIORITY_PAPER_SOURCE)],
                     "seedrep_battery": [("seedrep_controls", IC.PRIORITY_OWNING_READER), ("seedrep_statics", 0)]}, multi


def test_known_bad_covered_set_reports_the_two_em_panels(covered):
    """The pre-fix world: the EM adapter named the twelve stems and the organisms but not the two `*-superseded` panels."""
    dec = SRC.declared_sources(declared_path=DECLARED, roster_path=ROSTER)
    thinner = {s: c for s, c in covered.items() if s not in EM_SUPERSEDED}
    out = SRC.coverage(dec["declared"], thinner)
    assert [r["source"] for r in out["without_adapter"]] == sorted(EM_SUPERSEDED)
    assert all(any("paper watermarks" in w for w in r["declared_in"]) for r in out["without_adapter"])
    # and a second claimant at the SAME priority as the source's own adapter is reported; a distinct priority is not (the r2 adapter
    # declares PRIORITY_PAPER_SOURCE, so the shadow sits at whatever the adapter declares rather than at a literal)
    own = covered["r2_export"][0][1]
    dup = dict(covered, r2_export=list(covered["r2_export"]) + [("shadow_adapter", own)])
    assert [r["source"] for r in SRC.coverage(dec["declared"], dup)["claimed_without_priority"]] == ["r2_export"]
    dup2 = dict(covered, r2_export=list(covered["r2_export"]) + [("shadow_adapter", own - 15)])
    assert SRC.coverage(dec["declared"], dup2)["claimed_without_priority"] == []


def test_the_multi_turn_dose_ladder_is_its_own_provisional_source():
    """The multi-turn dose ladder (2026-09-18) is a source of its own, covered by its own adapter: its root is named by no
    paper lane file, so the lane-file-driven `multiturn_grades` cannot reach it, and the paper has declared no spelling for it."""
    covered = SRC.covered_sources(ADP.load_all())
    assert covered[SM.SOURCE] == [("strength_mtdiff", 0)], covered.get(SM.SOURCE)
    assert SM.SOURCE in SM.StrengthMtdiffAdapter.covers_sources and SM.SOURCE == "strength_mtdiff_s1"
    assert SM.StrengthMtdiffAdapter(root=SM.ROOT_DEFAULT).status_sources(f"{SM.TARGET}_d0p5_distdiff64s1") == (SM.SOURCE,)
    assert SM.SOURCE not in MG.MultiturnGradesAdapter.covers_sources


def test_the_arditi_vendor_re_run_is_its_own_provisional_source():
    """The vendor-preset re-run of the all-positions rungs (2026-09-20) is a source of its own: a different decode is a different
    measurement, so it never folds into the Arditi lane's own multi-turn rows, and no lane file names its root."""
    from sisteer.registry.adapters import arditi_vendor_mt as AV
    covered = SRC.covered_sources(ADP.load_all())
    assert covered[AV.SOURCE] == [("arditi_vendor_mt", 0)], covered.get(AV.SOURCE)
    assert AV.SOURCE == "arditi_vendor_mt" and AV.SOURCE not in MG.MultiturnGradesAdapter.covers_sources
    from sisteer.registry.adapters import arditi_allpos as AA
    assert AV.SOURCE != AA.SOURCE


def test_the_seed_replicate_mirror_is_its_own_provisional_source():
    """The seed-replicate statics / MMLU (cluster B, 2026-09-20) are a source of their own, covered by their own reader: the seed IS the tag
    (`qwen35_4b_s2`), the objects are the premints on the SEEDED canonical instruments, and the paired-write sweep's reader would stamp
    its source, substrate and instrument on rows that are none of those; the paper has declared no spelling for them."""
    from sisteer.registry.adapters import relu_paired as RP
    from sisteer.registry.adapters import seedrep_cluster as SC
    covered = SRC.covered_sources(ADP.load_all())
    assert covered[SC.SOURCE] == [("seedrep_cluster", 0)], covered.get(SC.SOURCE)
    assert SC.SOURCE == "seedrep_cluster" and SC.SOURCE in SC.SeedrepClusterAdapter.covers_sources
    assert SC.SOURCE not in RP.ReluPairedAdapter.covers_sources and SC.SOURCE != RP.SOURCE
    assert SC.SeedrepClusterAdapter(root="/nonexistent/seedrep").status_sources("qwen35_4b_s2") == (SC.SOURCE,)
    assert SC.SOURCE in PROVISIONAL_ADDITIONS       # the newer provisional additions are the Petri store's, the three-seed GSM8K stores' and the seed-1 battery's (below)


def test_the_w19_petri_store_is_its_own_provisional_source():
    """The Petri seed-1 store (2026-09-21) is a source of its own, covered by its own reader that folds onto the premint-queued rows:
    the store's runs are the battery's (campaign seedrep_battery), not the petri_stats sidecars' nor the paired-write sweep's; the paper has
    declared no spelling for them."""
    from sisteer.registry.adapters import petri_stats as PS
    from sisteer.registry.adapters import relu_paired as RP
    from sisteer.registry.adapters import seedrep_petri as SP
    covered = SRC.covered_sources(ADP.load_all())
    assert covered[SP.SOURCE] == [("seedrep_petri", SP.SeedrepPetriAdapter.priority)], covered.get(SP.SOURCE)
    assert SP.SOURCE == "seedrep_petri" and SP.SOURCE in SP.SeedrepPetriAdapter.covers_sources
    assert SP.SOURCE not in RP.ReluPairedAdapter.covers_sources and SP.SOURCE not in PS.PetriStatsAdapter.covers_sources
    assert SP.SeedrepPetriAdapter(roots=("/nonexistent/petri_seedrep",)).status_sources("s1") == (SP.SOURCE,)
    assert SP.SOURCE in PROVISIONAL_ADDITIONS


def test_the_three_seed_gsm8k_stores_are_their_own_provisional_source():
    """The batch-64 three-seed family's GSM8K stores (cluster A, 2026-09-21) are a source of their own, covered by their own reader:
    the `evalcheck/gsm8k_paired200` reader reads the paper wave's REPORTS, the seed-replicate reader reads these STORES on the seeded
    canonical instrument; the paper has declared no spelling for them."""
    from sisteer.registry.adapters import gsm8k_p200 as GP
    from sisteer.registry.adapters import gsm8k_seedrep as GS
    covered = SRC.covered_sources(ADP.load_all())
    assert covered[GS.SOURCE] == [("gsm8k_seedrep", 0)], covered.get(GS.SOURCE)
    assert GS.SOURCE == "gsm8k_seedrep" and GS.SOURCE in GS.Gsm8kSeedrepAdapter.covers_sources and GS.SOURCE in PROVISIONAL_ADDITIONS
    assert GS.SOURCE not in GP.Gsm8kP200Adapter.covers_sources and GS.SOURCE != GP.SOURCE
    assert GS.Gsm8kSeedrepAdapter(root="/nonexistent/gsm8k_seedrep").status_sources("qwen35_4b_s1") == (GS.SOURCE,)
    assert GS.SOURCE in PROVISIONAL_ADDITIONS[-8:]        # among the newest eight: the seed-1 battery's, the four misalignment spellings, the paired_allmodels document and the native best-of-N follow


def test_the_paired_allmodels_gsm8k_document_is_its_own_provisional_source_read_by_the_gsm8k_readers_second_adapter():
    """The paired_allmodels GSM8K document (cluster A, 2026-09-22; the paired-write difference ReLU on gemma / llama / nemotron and nemotron's
    constitution ReLU under the default Inspect seed rule) is a source of its own -- `gsm8k_paired_allmodels` -- covered by the GSM8K
    reader's SECOND adapter (a subclass of the seed-replicate reader over another root): never `gsm8k_seedrep` (other models, another seed rule, other
    objects) and never `evalcheck/gsm8k_paired200` (reports of the paper wave)."""
    from sisteer.registry.adapters import gsm8k_p200 as GP
    from sisteer.registry.adapters import gsm8k_paired_allmodels as PA
    from sisteer.registry.adapters import gsm8k_seedrep as GS
    covered = SRC.covered_sources(ADP.load_all())
    assert covered[PA.SOURCE] == [("gsm8k_paired_allmodels", 0)], covered.get(PA.SOURCE)
    assert PA.SOURCE == "gsm8k_paired_allmodels" == PROVISIONAL_ADDITIONS[-3] and PA.SOURCE in PA.Gsm8kPairedAllmodelsAdapter.covers_sources
    assert PA.SOURCE not in EM.EmMisalignAdapter.covers_sources and PA.SOURCE != EM.LADDER_SOURCE
    assert PA.SOURCE not in GS.Gsm8kSeedrepAdapter.covers_sources and PA.SOURCE != GS.SOURCE and PA.SOURCE not in GP.Gsm8kP200Adapter.covers_sources
    assert GS.SOURCE not in PA.Gsm8kPairedAllmodelsAdapter.covers_sources
    assert PA.Gsm8kPairedAllmodelsAdapter(root="/nonexistent/gsm8k_paired_allmodels").status_sources("nemotron3nano4b") == (PA.SOURCE,)
    assert PA.ROOT_DEFAULT != GS.ROOT_DEFAULT and PA.ROOT_DEFAULT.endswith(os.path.join("run-state", "full22", "gsm8k_paired_allmodels"))


def test_the_three_seed_math500_stores_are_their_own_provisional_source_read_by_the_math500_seedrep_reader():
    """The seed-replicate MATH-500 stores (cluster B, 2026-09-23: the seed tags' rows, the paired tags' composed object and the Arditi
    all-positions object) are a source of their own -- `math500_seedrep` -- covered by their own standalone reader: never
    `gsm8k_seedrep` (another eval, another store grammar with no cap step, another mirror) and never `evalcheck/math500` (reports of the
    paper's steer-pins wave)."""
    from sisteer.registry.adapters import gsm8k_seedrep as GS
    from sisteer.registry.adapters import math500 as EM5
    from sisteer.registry.adapters import math500_seedrep as MS
    covered = SRC.covered_sources(ADP.load_all())
    assert covered[MS.SOURCE] == [("math500_seedrep", 0)], covered.get(MS.SOURCE)
    assert MS.SOURCE == "math500_seedrep" == PROVISIONAL_ADDITIONS[14] and MS.SOURCE in MS.Math500SeedrepAdapter.covers_sources
    assert MS.SOURCE not in GS.Gsm8kSeedrepAdapter.covers_sources and MS.SOURCE != GS.SOURCE and MS.SOURCE != EM5.SOURCE
    assert GS.SOURCE not in MS.Math500SeedrepAdapter.covers_sources and MS.EVAL_ID == EM5.EVAL_ID == "math500"
    assert MS.Math500SeedrepAdapter(root="/nonexistent/math500_seedrep").status_sources("qwen35_4b_s1") == (MS.SOURCE,)
    assert MS.ROOT_DEFAULT != GS.ROOT_DEFAULT and MS.ROOT_DEFAULT.endswith(os.path.join("run-state", "full22", "math500_seedrep"))
def test_the_three_seed_bigcodebench_and_bfcl_stores_are_two_provisional_sources_read_by_their_own_readers_over_one_shared_reading():
    """The seed-replicate BigCodeBench-200 / BFCL-200 stores (Modal, 2026-09-23) are TWO sources -- `bigcodebench_seedrep` and `bfcl_seedrep`,
    each covered by its own reader over its own mirror root (one store holds both tasks; each root is its own copy), both binding the
    shared reading in `_bcbw_common` (a helper, never an adapter of its own): never `bcb_bfcl_200` / `bcb_bfcl_200_widen` (the Agentic lane's paper
    rows) and never each other's root."""
    from sisteer.registry.adapters import _bcbw_common as B
    from sisteer.registry.adapters import bcb_bfcl as AG
    from sisteer.registry.adapters import bfcl_seedrep as BF
    from sisteer.registry.adapters import bigcodebench_seedrep as BC
    covered = SRC.covered_sources(ADP.load_all())
    assert covered[BC.SOURCE] == [("bigcodebench_seedrep", 0)] and covered[BF.SOURCE] == [("bfcl_seedrep", 0)], (covered.get(BC.SOURCE), covered.get(BF.SOURCE))
    assert BC.SOURCE == "bigcodebench_seedrep" == PROVISIONAL_ADDITIONS[15] and BF.SOURCE == "bfcl_seedrep" == PROVISIONAL_ADDITIONS[16]
    assert BC.SOURCE in BC.BigcodebenchSeedrepAdapter.covers_sources and BF.SOURCE in BF.BfclSeedrepAdapter.covers_sources
    assert BC.SOURCE not in BF.BfclSeedrepAdapter.covers_sources and BF.SOURCE not in BC.BigcodebenchSeedrepAdapter.covers_sources
    assert not ({BC.SOURCE, BF.SOURCE} & set(AG.BcbBfclAdapter.covers_sources)) and (BC.EVAL_ID, BF.EVAL_ID) == ("bigcodebench", "bfcl")
    assert issubclass(BC.BigcodebenchSeedrepAdapter, B.BcbwSeedrepReader) and issubclass(BF.BfclSeedrepAdapter, B.BcbwSeedrepReader)
    assert not isinstance(getattr(B.BcbwSeedrepReader, "name", None), str)
    assert BC.BigcodebenchSeedrepAdapter(root="/nonexistent/bigcodebench_seedrep").status_sources("qwen35_4b_s1") == (BC.SOURCE,)
    assert BF.BfclSeedrepAdapter(root="/nonexistent/bfcl_seedrep").status_sources("qwen35_4b_s1") == (BF.SOURCE,)
    assert BC.ROOT_DEFAULT != BF.ROOT_DEFAULT and BC.ROOT_DEFAULT.endswith(os.path.join("run-state", "full22", "bigcodebench_seedrep")) and BF.ROOT_DEFAULT.endswith(os.path.join("run-state", "full22", "bfcl_seedrep"))


def test_the_appendix_families_bigcodebench_and_bfcl_stores_are_two_provisional_sources_read_by_their_own_readers_over_the_shared_reading():
    """The appendix run (2026-09-23): gemma4e4b / llama8b / nemotron3nano4b on BigCodeBench-200 / BFCL-200, one seed tag each and its paired
    tag, natively served under the protocol decode -- TWO sources of their own, `bigcodebench_allmodels` and `bfcl_allmodels`, each covered by
    its own reader over its own mirror root, both binding the seedrep readers' shared reading (`_bcbw_common`) to the APPENDIX model set: never the three-seed
    Qwen family's sources (another model set, another root: a qwen store under an appendix root refuses its group, and a gemma store under a seedrep root
    likewise) and never the Agentic paper rows."""
    from sisteer.registry.adapters import _bcbw_common as B
    from sisteer.registry.adapters import bcb_bfcl as AG
    from sisteer.registry.adapters import bfcl_allmodels as BFA
    from sisteer.registry.adapters import bfcl_seedrep as BF
    from sisteer.registry.adapters import bigcodebench_allmodels as BCA
    from sisteer.registry.adapters import bigcodebench_seedrep as BC
    covered = SRC.covered_sources(ADP.load_all())
    assert covered[BCA.SOURCE] == [("bigcodebench_allmodels", 0)] and covered[BFA.SOURCE] == [("bfcl_allmodels", 0)], (covered.get(BCA.SOURCE), covered.get(BFA.SOURCE))
    assert BCA.SOURCE == "bigcodebench_allmodels" == PROVISIONAL_ADDITIONS[11] and BFA.SOURCE == "bfcl_allmodels" == PROVISIONAL_ADDITIONS[12]
    assert BCA.SOURCE in BCA.BigcodebenchAllmodelsAdapter.covers_sources and BFA.SOURCE in BFA.BfclAllmodelsAdapter.covers_sources
    assert not ({BCA.SOURCE, BFA.SOURCE} & (set(BC.BigcodebenchSeedrepAdapter.covers_sources) | set(BF.BfclSeedrepAdapter.covers_sources) | set(AG.BcbBfclAdapter.covers_sources)))
    assert not ({BC.SOURCE, BF.SOURCE} & (set(BCA.BigcodebenchAllmodelsAdapter.covers_sources) | set(BFA.BfclAllmodelsAdapter.covers_sources)))
    assert issubclass(BCA.BigcodebenchAllmodelsAdapter, B.BcbwSeedrepReader) and issubclass(BFA.BfclAllmodelsAdapter, B.BcbwSeedrepReader)
    # (+ the onboarded test model since 2026-10-03: the appendix sources read its u0 / s1 standups too)
    assert BCA.SPEC.models == BFA.SPEC.models == B.APPENDIX_MODEL_KEYS == ("gemma4e4b", "llama8b", "nemotron3nano4b") and BC.SPEC.models == BF.SPEC.models == ("qwen35_4b",)
    assert (BCA.EVAL_ID, BFA.EVAL_ID) == ("bigcodebench", "bfcl") and BCA.INSTRUMENT_SEED == BC.INSTRUMENT_SEED and BFA.INSTRUMENT_SEED == BF.INSTRUMENT_SEED
    assert BCA.BigcodebenchAllmodelsAdapter(root="/nonexistent/bigcodebench_allmodels").status_sources("gemma4e4b_s1") == (BCA.SOURCE,)
    assert BFA.BfclAllmodelsAdapter(root="/nonexistent/bfcl_allmodels").status_sources("nemotron3nano4b_s1") == (BFA.SOURCE,)
    assert BCA.ROOT_DEFAULT != BFA.ROOT_DEFAULT != BF.ROOT_DEFAULT and BCA.ROOT_DEFAULT.endswith(os.path.join("run-state", "full22", "bigcodebench_allmodels"))
    assert BFA.ROOT_DEFAULT.endswith(os.path.join("run-state", "full22", "bfcl_allmodels")) and BCA.ROOT_DEFAULT != BC.ROOT_DEFAULT


def test_the_full_suite_bigcodebench_and_bfcl_stores_are_two_provisional_sources_on_their_own_item_sets_and_instruments():
    """The FULL-suite run (the contract of 2026-09-24): every one of BigCodeBench's 1,140 tasks and BFCL's 3,981 items -- TWO sources of their own,
    `bigcodebench_full` and `bfcl_full`, each covered by its own reader over its own mirror root (one store holds both full tasks; each root
    its own copy), both binding the shared reading (`_bcbw_common`) with the full item-set attestation: NEW item sets (bigcodebench_full_1140 /
    bfcl_full_3981) and NEW instruments, never the 200-item readers' (another item set is another instrument; the sources bind every family of the
    model-key table, since the full run spans them in one root), never each other's root and never the Agentic paper rows."""
    from sisteer.registry.adapters import _bcbw_common as B
    from sisteer.registry.adapters import bcb_bfcl as AG
    from sisteer.registry.adapters import bfcl_allmodels as BFA
    from sisteer.registry.adapters import bfcl_full as BFF
    from sisteer.registry.adapters import bfcl_seedrep as BF
    from sisteer.registry.adapters import bigcodebench_allmodels as BCA
    from sisteer.registry.adapters import bigcodebench_full as BCF
    from sisteer.registry.adapters import bigcodebench_seedrep as BC
    covered = SRC.covered_sources(ADP.load_all())
    assert covered[BCF.SOURCE] == [("bigcodebench_full", 0)] and covered[BFF.SOURCE] == [("bfcl_full", 0)], (covered.get(BCF.SOURCE), covered.get(BFF.SOURCE))
    assert BCF.SOURCE == "bigcodebench_full" == PROVISIONAL_ADDITIONS[9] and BFF.SOURCE == "bfcl_full" == PROVISIONAL_ADDITIONS[10]
    assert BCF.SOURCE in BCF.BigcodebenchFullAdapter.covers_sources and BFF.SOURCE in BFF.BfclFullAdapter.covers_sources
    others = set(BC.BigcodebenchSeedrepAdapter.covers_sources) | set(BF.BfclSeedrepAdapter.covers_sources) | set(BCA.BigcodebenchAllmodelsAdapter.covers_sources) \
        | set(BFA.BfclAllmodelsAdapter.covers_sources) | set(AG.BcbBfclAdapter.covers_sources)
    assert not ({BCF.SOURCE, BFF.SOURCE} & others) and not ({BCF.SOURCE} & set(BFF.BfclFullAdapter.covers_sources)) and not ({BFF.SOURCE} & set(BCF.BigcodebenchFullAdapter.covers_sources))
    assert issubclass(BCF.BigcodebenchFullAdapter, B.BcbwSeedrepReader) and issubclass(BFF.BfclFullAdapter, B.BcbwSeedrepReader)
    assert BCF.SPEC.models == BFF.SPEC.models == tuple(B.MODEL_KEYS) and (BCF.EVAL_ID, BFF.EVAL_ID) == ("bigcodebench", "bfcl")
    assert (BCF.SPEC.item_set_id, BFF.SPEC.item_set_id) == ("bigcodebench_full_1140", "bfcl_full_3981") and BCF.SPEC.full_item_set and BFF.SPEC.full_item_set
    assert BCF.INSTRUMENT_SEED != BCA.INSTRUMENT_SEED and BFF.INSTRUMENT_SEED != BFA.INSTRUMENT_SEED, "a new item set is a new instrument"
    assert not (BCA.SPEC.full_item_set or BFA.SPEC.full_item_set or BC.SPEC.full_item_set or BF.SPEC.full_item_set), "the 200-item readers carry no item set: they read as before"
    assert BCF.BigcodebenchFullAdapter(root="/nonexistent/bigcodebench_full").status_sources("gemma4e4b_s1") == (BCF.SOURCE,)
    assert BFF.BfclFullAdapter(root="/nonexistent/bfcl_full").status_sources("qwen35_4b_s1") == (BFF.SOURCE,)
    assert BCF.ROOT_DEFAULT != BFF.ROOT_DEFAULT != BFA.ROOT_DEFAULT and BCF.ROOT_DEFAULT.endswith(os.path.join("run-state", "full22", "bigcodebench_full"))
    assert BFF.ROOT_DEFAULT.endswith(os.path.join("run-state", "full22", "bfcl_full")) and BCF.ROOT_DEFAULT != BCA.ROOT_DEFAULT


def test_the_three_seed_ifeval_stores_are_their_own_provisional_source_read_at_the_completion_cap_by_the_ifeval_seedrep_reader():
    """The seed-replicate IFEval stores (Modal, 2026-09-23: the seed tags' rows, the paired tags' composed object and the Arditi all-positions
    object, every row at the cap its 2,048 -> 4,096 -> 8,192 ladder reached) are a source of their own -- `ifeval_seedrep` -- covered
    by their own reader, which OWNS the completion-cap merge: never `gsm8k_seedrep` (another eval, k = 1 items where these are k = 2 draws, a two-cap
    ladder on Slurm) and never `bigcodebench_seedrep` / `bfcl_seedrep` (one pass, no cap step, another root)."""
    from sisteer.registry.adapters import bigcodebench_seedrep as BC
    from sisteer.registry.adapters import gsm8k_seedrep as GS
    from sisteer.registry.adapters import ifeval_seedrep as IF
    covered = SRC.covered_sources(ADP.load_all())
    assert covered[IF.SOURCE] == [("ifeval_seedrep", 0)], covered.get(IF.SOURCE)
    assert IF.SOURCE == "ifeval_seedrep" == PROVISIONAL_ADDITIONS[13] and IF.SOURCE in IF.IfevalSeedrepAdapter.covers_sources
    assert IF.SOURCE not in GS.Gsm8kSeedrepAdapter.covers_sources and IF.SOURCE not in BC.BigcodebenchSeedrepAdapter.covers_sources and GS.SOURCE not in IF.IfevalSeedrepAdapter.covers_sources
    assert IF.EVAL_ID == "ifeval" and IF.LADDER == (2048, 4096, 8192) and IF.K_TRIALS == 2 and IF.MERGE_ALGORITHM.endswith("ifeval_seedrep.completion_cap_merge")
    assert IF.IfevalSeedrepAdapter(root="/nonexistent/ifeval_seedrep").status_sources("qwen35_4b_s1") == (IF.SOURCE,)
    assert IF.ROOT_DEFAULT != GS.ROOT_DEFAULT and IF.ROOT_DEFAULT.endswith(os.path.join("run-state", "full22", "ifeval_seedrep"))


def test_the_seed_one_static_battery_is_its_own_provisional_source_read_by_the_legacy_store_reader():
    """The seed-1 static stores (Modal, 2026-09-21) fold onto the premint's QUEUED rows, whose `lane_source_legacy` is the campaign's own
    `seedrep_battery`: the reader files under that spelling (another would overwrite the queued row's source on fold), so the source is declared
    provisionally and covered by the legacy-store reader alone -- never by the cluster B seed-replicate reader, whose rows are another source."""
    from sisteer.registry.adapters import seedrep_cluster as SC
    from sisteer.registry.adapters import seedrep_controls as SCT
    from sisteer.registry.adapters import seedrep_statics as SS
    covered = SRC.covered_sources(ADP.load_all())
    # two legacy-store readers of ONE campaign source, at distinct priorities (the statics reader's undeclared 0, the controls reader's owning
    # rank): the premint queued every row under `seedrep_battery`, so both fold under it -- statics + best-of-N by one, the served controls
    # (XSTest / OR-Bench-Hard / MMLU) by the other; their runs share no measurement key
    assert covered[SS.SOURCE] == [("seedrep_controls", IC.PRIORITY_OWNING_READER), ("seedrep_statics", 0)], covered.get(SS.SOURCE)
    assert SS.SOURCE == SCT.SOURCE == "seedrep_battery" == PROVISIONAL_ADDITIONS[-7] and SS.SOURCE in SS.SeedrepStaticsAdapter.covers_sources
    assert SCT.SOURCE in SCT.SeedrepControlsAdapter.covers_sources and SCT.SeedrepControlsAdapter.priority != getattr(SS.SeedrepStaticsAdapter, "priority", 0)
    assert SS.SOURCE not in SC.SeedrepClusterAdapter.covers_sources and SS.SOURCE != SC.SOURCE
    assert SS.SeedrepStaticsAdapter(root="/nonexistent/seedrep_battery", volatile_root=None).status_sources("codeattack__s1") == (SS.SOURCE,)
    assert SCT.SeedrepControlsAdapter(root="/nonexistent/seedrep_battery", volatile_root=None).status_sources("controls__s1") == (SCT.SOURCE,)


def test_the_three_seed_agentic_misalignment_stores_are_their_own_provisional_source_read_by_the_em_reader():
    """The three-seed agentic-misalignment stores (cluster A, 2026-09-21) are a source of their own -- `misalign_seedrep` -- read by
    the EM misalignment reader as its SECOND root (`em_misalign.SEEDREP_SOURCE`), never by the paper's per-lane `em/agentic/<stem>` strings (the
    stem's panels are another generation) and never by `seedrep_statics` (its seed-1 Modal rows are `seedrep_battery`)."""
    from sisteer.registry.adapters import seedrep_statics as SS
    covered = SRC.covered_sources(ADP.load_all())
    assert covered[EM.SEEDREP_SOURCE] == [("em_misalign", 0)], covered.get(EM.SEEDREP_SOURCE)
    assert EM.SEEDREP_SOURCE == "misalign_seedrep" == PROVISIONAL_ADDITIONS[-6] and EM.SEEDREP_SOURCE in EM.EmMisalignAdapter.covers_sources
    assert EM.SEEDREP_SOURCE not in SS.SeedrepStaticsAdapter.covers_sources and EM.SEEDREP_SOURCE != SS.SOURCE
    assert not EM.SEEDREP_SOURCE.startswith("em/agentic/")
    ad = EM.EmMisalignAdapter(root="/nonexistent/figs", full22_pins=None, baseline_pins=None, behaviour_source={}, seedrep_root=None)
    assert ad.status_sources("misalign_seedrep/qwen35_4b_s2p") == (EM.SEEDREP_SOURCE,) and ad.status_sources("qwen35_4b") == ("em/agentic/qwen35_4b",)


def test_the_16k_step_agentic_misalignment_roots_are_their_own_provisional_source_read_by_the_em_reader():
    """The 16,384-step re-draws (cluster A, 2026-09-21 22:10Z) are a source of their own -- `misalign_16k` -- read by the EM misalignment
    reader as its THIRD root (`em_misalign.M16K_SOURCE`): never `misalign_seedrep` (the same target names at the 3,000 cap; the row store's run_id even
    collides across the two roots), never the paper's `em/agentic/<stem>` strings (the stem's `_ladder` tree is another generation), and never resolved
    through the grade root (whose misalign_16k tree is the superseded lens generation)."""
    covered = SRC.covered_sources(ADP.load_all())
    assert covered[EM.M16K_SOURCE] == [("em_misalign", 0)], covered.get(EM.M16K_SOURCE)
    assert EM.M16K_SOURCE == "misalign_16k" == PROVISIONAL_ADDITIONS[-5] and EM.M16K_SOURCE in EM.EmMisalignAdapter.covers_sources
    assert EM.M16K_SOURCE != EM.SEEDREP_SOURCE and not EM.M16K_SOURCE.startswith("em/agentic/")
    assert EM.M16K_ROOT.startswith(LC.RS_DEFAULT + os.sep) and not EM.M16K_ROOT.startswith(EM.EmMisalignAdapter.GRADE_ROOT)
    ad = EM.EmMisalignAdapter(root="/nonexistent/figs", full22_pins=None, baseline_pins=None, behaviour_source={}, seedrep_root=None, m16k_root=None)
    assert ad.status_sources("misalign_16k/qwen35_4b_bl") == (EM.M16K_SOURCE,) and ad.status_sources("misalign_seedrep/qwen35_4b_s2p") == (EM.SEEDREP_SOURCE,)


def test_the_completion_cap_merges_of_the_agentic_ladder_are_their_own_provisional_source_read_by_the_em_reader():
    """The completion-cap MERGES (cluster A, 2026-09-22 02:22Z) are a source of their own -- `misalign_ladder` -- read by the EM
    misalignment reader as its FOURTH root (`em_misalign.LADDER_SOURCE`, the exact string the registry-only exclude ruling
    r2v3-exclude-misalign-ladder-merged-readings-registry-only-pending-user-role is keyed on): never `misalign_seedrep` / `misalign_16k` (the same
    target names at the base cap / the step) and never the paper's `em/agentic/<stem>` strings (the stem's `_ladder` tree is the 2026-09-11 lens merge)."""
    covered = SRC.covered_sources(ADP.load_all())
    assert covered[EM.LADDER_SOURCE] == [("em_misalign", 0)], covered.get(EM.LADDER_SOURCE)
    assert EM.LADDER_SOURCE == "misalign_ladder" == PROVISIONAL_ADDITIONS[-4] and EM.LADDER_SOURCE in EM.EmMisalignAdapter.covers_sources
    assert EM.LADDER_SOURCE not in (EM.SEEDREP_SOURCE, EM.M16K_SOURCE) and not EM.LADDER_SOURCE.startswith("em/agentic/")
    assert EM.LADDER_ROOT.startswith(LC.RS_DEFAULT + os.sep) and not EM.LADDER_ROOT.startswith(EM.EmMisalignAdapter.GRADE_ROOT)
    ad = EM.EmMisalignAdapter(root="/nonexistent/figs", full22_pins=None, baseline_pins=None, behaviour_source={}, seedrep_root=None, m16k_root=None, ladder_root=None)
    assert ad.status_sources("misalign_ladder/qwen35_4b_s2") == (EM.LADDER_SOURCE,) and ad.status_sources("misalign_16k/qwen35_4b_bl") == (EM.M16K_SOURCE,)


def test_the_paired_write_difference_relu_agentic_roots_of_the_other_models_are_their_own_provisional_source_read_by_the_em_reader():
    """The paired-write difference-ReLU agentic roots of gemma / llama / nemotron (cluster A, 2026-09-22 07:39Z; one steered key, no none, under
    the seed roots' mirror by a grammar of their own) are a source of their own -- `misalign_paired_allmodels` -- read by the EM
    misalignment reader as its FIFTH root (`em_misalign.PAIRED_SOURCE`): never `misalign_seedrep` (the same mirror directory, but another model,
    another object and no none: the seed-replicate discovery skips the names), never `gsm8k_paired_allmodels` (the same objects on GSM8K,
    another reader) and never the paper's `em/agentic/<stem>` strings."""
    from sisteer.registry.adapters import gsm8k_paired_allmodels as PA
    covered = SRC.covered_sources(ADP.load_all())
    assert covered[EM.PAIRED_SOURCE] == [("em_misalign", 0)], covered.get(EM.PAIRED_SOURCE)
    assert EM.PAIRED_SOURCE == "misalign_paired_allmodels" == PROVISIONAL_ADDITIONS[-1] and EM.PAIRED_SOURCE in EM.EmMisalignAdapter.covers_sources
    assert EM.PAIRED_SOURCE not in (EM.SEEDREP_SOURCE, EM.M16K_SOURCE, EM.LADDER_SOURCE) and not EM.PAIRED_SOURCE.startswith("em/agentic/")
    assert EM.PAIRED_SOURCE != PA.SOURCE and EM.PAIRED_SOURCE not in PA.Gsm8kPairedAllmodelsAdapter.covers_sources
    ad = EM.EmMisalignAdapter(root="/nonexistent/figs", full22_pins=None, baseline_pins=None, behaviour_source={}, seedrep_root="/nonexistent/misalign", m16k_root=None, ladder_root=None)
    assert ad.status_sources("misalign_paired_allmodels/gemma4e4b_paired_d1_gemma4e4b_s1") == (EM.PAIRED_SOURCE,)
    assert ad.status_sources("misalign_seedrep/qwen35_4b_s2p") == (EM.SEEDREP_SOURCE,)
    assert EM.parse_paired_target("gemma4e4b_paired_d1_gemma4e4b_s1") == {"model": "gemma4e4b", "dose": 1.0, "dose_token": "d1", "seed": 1}
    assert EM.parse_paired_target("qwen35_4b_s1p") is None and EM.parse_paired_target("gemma4e4b_paired_d1_llama8b_s1") is None


def test_the_llama_native_best_of_n_grades_are_their_own_provisional_source_not_a_bon_stores_store():
    """The Llama paired-write difference-ReLU best-of-75 grades of a NATIVE 4,096 pass (cluster B draws 2026-09-19, V1 Haiku grades
    2026-09-22) are a source of their own -- `bon_native4096:llama8b_relu_paired`, the adapter's name and
    its one group -- read by `adapters/bon_native.py` alone. Never `bon_stores` (the 512-store-plus-climb reader: the point is registered nowhere
    in `bon_cap4096`, and a registration would make the reader refuse by name) and never relu_paired (whose best-of-N is the 512 store)."""
    from sisteer.registry.adapters import bon_native as BN
    from sisteer.registry.adapters import bon_stores as BS
    from sisteer.registry.adapters import relu_paired as RP
    covered = SRC.covered_sources(ADP.load_all())
    assert covered[BN.SOURCE] == [("bon_native4096", 0)], covered.get(BN.SOURCE)
    assert BN.SOURCE == "bon_native4096:llama8b_relu_paired" == f"{BN.NAME}:{BN.GROUP}" == PROVISIONAL_ADDITIONS[-2]
    assert BN.BonNative4096Adapter.covers_sources == (BN.SOURCE,) and BN.BonNative4096Adapter.owner == "BoN"
    assert BN.SOURCE not in BS.BonStoresAdapter.covers_sources and BN.SOURCE not in RP.ReluPairedAdapter.covers_sources
    ad = BN.BonNative4096Adapter(rs="/nonexistent/run-state")
    assert ad.name == "bon_native4096" and ad.status_sources(BN.GROUP) == (BN.SOURCE,) if hasattr(ad, "status_sources") else True
    from sisteer.registry.adapters import base as AB
    assert AB.status_sources_of(ad, BN.GROUP) == (BN.SOURCE,) and AB.owner_of(ad) == "BoN"


def test_the_seed_replicate_over_refusal_controls_are_their_own_provisional_source_read_by_their_own_name():
    """The seed-replicate over-refusal controls (cluster A, 2026-09-21) are a source of their own — `orxs_seedrep` — read through
    the shared served-controls reader by an adapter registered under the source's name (`cellstore.OrxsSeedrepAdapter`): the owner and the
    status source are adapter-level, so the generic `cellstore` adapter (owner `registry`, no source) cannot declare them; its own name,
    roots and emission version are untouched. The seed-replicate statics reader does not cover it (different stores, different instruments)."""
    from sisteer.registry.adapters import _items_common as IC
    from sisteer.registry.adapters import cellstore as CS
    from sisteer.registry.adapters import seedrep_cluster as SC
    covered = SRC.covered_sources(ADP.load_all())
    assert covered[CS.ORXS_SEEDREP_SOURCE] == [("orxs_seedrep", IC.PRIORITY_GENERIC_READER)], covered.get(CS.ORXS_SEEDREP_SOURCE)
    assert CS.ORXS_SEEDREP_SOURCE == "orxs_seedrep" and CS.ORXS_SEEDREP_SOURCE in CS.OrxsSeedrepAdapter.covers_sources
    assert CS.ORXS_SEEDREP_SOURCE not in CS.CellStoreAdapter().covers_sources and CS.ORXS_SEEDREP_SOURCE not in SC.SeedrepClusterAdapter.covers_sources
    ad = CS.OrxsSeedrepAdapter(roots=())
    assert ad.name == "orxs_seedrep" and ad.owner == CS.ORXS_SEEDREP_OWNER == "OR" and ad.status_sources("any-group") == (CS.ORXS_SEEDREP_SOURCE,)
    assert CS.ORXS_SEEDREP_SOURCE in PROVISIONAL_ADDITIONS      # a membership check: the newest landed addition holds the tuple's last slot
    # every root of the source files its runs under the source string, and none of them is a root of the generic adapter
    assert CS.ORXS_SEEDREP_ROOTS and all(r.source_legacy == CS.ORXS_SEEDREP_SOURCE for r in CS.ORXS_SEEDREP_ROOTS)
    assert not ({r.path for r in CS.ORXS_SEEDREP_ROOTS} & {r.path for r in CS.DEFAULT_ROOTS})


def test_the_qwen_organism_panels_are_two_sources_covered_by_the_em_adapter():
    """The two decode rungs of EM's Qwen3.5-4B organisms panel (2026-09-17) are two sources -- a decode change is a different measurement, never
    a rerun into the same tree -- both covered by the EM adapter, and both PROVISIONAL (the paper's roster spelling is unsettled)."""
    covered = SRC.covered_sources(ADP.load_all())
    em_panels = (EM.QWEN_ORG_SOURCE, EM.QWEN_ORG_VENDOR_SOURCE)
    for s_ in em_panels:
        assert covered[s_] == [("em_misalign", 0)], s_
        assert s_ in EM.EmMisalignAdapter.covers_sources
    assert set(em_panels) < set(PROVISIONAL_ADDITIONS) and em_panels == PROVISIONAL_ADDITIONS[:2]
    ad = EM.EmMisalignAdapter()
    assert ad.status_sources("qwen35_4b_org") == (EM.QWEN_ORG_SOURCE,) and ad.status_sources("qwen35_4b_org_vendor") == (EM.QWEN_ORG_VENDOR_SOURCE,)


def test_the_allpos_sources_of_2026_09_17_are_covered_by_their_own_adapters():
    """The paper's `arditi_allpos` source (the all-positions ladder) has `adapters/arditi_allpos.py`,
    and EM's all-positions PANEL of the same arm — a different source on a different instrument — rides the EM adapter's stem list."""
    from sisteer.registry.adapters import _items_common as IC
    from sisteer.registry.adapters import arditi_allpos as AA
    covered = SRC.covered_sources(ADP.load_all())
    # the source's own adapter is the declared SECONDARY of the export for the multi-turn measurements (ownership.MULTI_SOURCE_OWNERSHIP),
    # so it declares a priority below the export's PRIORITY_PAPER_SOURCE rather than the undeclared 0 it carried before
    assert covered["arditi_allpos"] == [("arditi_allpos", IC.PRIORITY_SECONDARY_SOURCE)]
    assert covered["r2_export"] == [("r2_export", IC.PRIORITY_PAPER_SOURCE)]
    assert covered["em/agentic/qwen35_4b_allpos"] == [("em_misalign", 0)]
    # the two are NOT one source: one is the ladder's own battery (static / XSTest / MMLU), the other the misalignment panel of its
    # served operating point — same object, different instruments, so they never share a source string or a run
    assert AA.SOURCE == "arditi_allpos" and "em/agentic/qwen35_4b_allpos" in EM.EmMisalignAdapter.covers_sources
    assert "qwen35_4b_allpos" in EM.EM_STEMS and AA.SOURCE not in EM.EmMisalignAdapter.covers_sources


def test_em_adapter_declares_the_superseded_panels_and_files_their_status_under_the_stem():
    ad = EM.EmMisalignAdapter()
    assert set(EM_SUPERSEDED) <= set(ad.covers_sources)
    assert set(f"em/agentic/{p}" for p in EM.EM_SUPERSEDED_DOSE_PANELS) == set(EM_SUPERSEDED)
    for panel in EM.EM_SUPERSEDED_DOSE_PANELS:      # the grammar reads each as a superseded-dose copy of a _bl stem
        p = EM.parse_panel_name(panel, EM.EM_STEMS)
        assert p.kind == "agentic" and p.superseded_dose and p.base.endswith("_bl"), p
    # the stem's status rows are filed under the stem's source AND its superseded panel's, so a refused read of the
    # record copy is visible by the paper's own name for it
    assert ad.status_sources("gemma4e4b_bl") == ("em/agentic/gemma4e4b_bl", "em/agentic/gemma4e4b_bl__lam3-superseded")
    assert ad.status_sources("llama8b_bl") == ("em/agentic/llama8b_bl", "em/agentic/llama8b_bl__r0p1-superseded")
    assert ad.status_sources("qwen35_4b_bl") == ("em/agentic/qwen35_4b_bl",)
    assert ad.status_sources("em_organisms") == ("em/em_organisms",)


def test_strength_ladder_declares_the_paper_sources_even_without_records_on_disk(tmp_path):
    ad = SL.StrengthLadderAdapter(root=str(tmp_path / "absent"))
    assert ad.covers_sources == ()                                        # discovery finds nothing here ...
    assert SRC.adapter_sources(ad) == ("strength_lane", "strength_lane_base2it_native", "strength_lane_base2it_native_mmlu")   # ... the declaration still covers the paper's three
    assert ad.declared_sources == SL.DECLARED_SOURCES


# --------------------------------------------------------------------------------------------
# the status page
# --------------------------------------------------------------------------------------------

def _root(tmp_path, paper_checkout=None):
    """A registry root whose config names ITS OWN paper checkout (default: an empty directory), so the page never reads
    another session's live paper tree — the grading host's default checkout is a working tree that moves under the tests."""
    from sisteer.registry import config as CFG
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    with open(CFG.config_path(root), "w", encoding="utf-8") as fh:
        fh.write(CFG.render_default(root, user="testuser", paper_checkout=paper_checkout or str(tmp_path / "nopaper")))
    synth.write_corpus(root, n_runs=1, n_items=2)
    H.ingest_once(root, workers=0, min_free=0)
    return root


def test_status_carries_a_sources_without_an_adapter_section_that_reads_zero_today(tmp_path):
    root = _root(tmp_path)
    ctx = status.status_json(root)
    cov = ctx["source_coverage"]
    assert cov["errors"] == [] and cov["without_adapter"] == [] and cov["claimed_without_priority"] == []
    assert cov["n_declared"] >= N_SOURCES
    out = str(tmp_path / "index.html")
    status.render(root, out_path=out, register=False)
    html = open(out, encoding="utf-8").read()
    assert "Sources without an adapter" in html and "declared sources without an adapter" in html


def test_status_lists_a_source_the_paper_declares_that_no_adapter_covers(tmp_path):
    """A source the LIVE paper checkout declares (its watermarks file, read through `[paper] checkout` of the root's
    config) and no adapter covers is on the page with its owner and where it was declared."""
    checkout = tmp_path / "paper_checkout"
    lib = checkout / "paper" / "paperlib"
    lib.mkdir(parents=True)
    (lib / "lane_watermarks.json").write_text('{"sources": {"brand_new_source": {"owner": "Nobody"}}}')
    root = _root(tmp_path, paper_checkout=str(checkout))
    cov = status.status_json(root)["source_coverage"]
    assert cov["paper_checkout"] == str(checkout) and cov["error"] is None
    assert [r["source"] for r in cov["without_adapter"]] == ["brand_new_source"]
    assert cov["without_adapter"][0]["owner"] == "Nobody" and cov["without_adapter"][0]["declared_in"] == [f"paper checkout watermarks ({checkout})"]
    out = str(tmp_path / "index.html")
    status.render(root, out_path=out, register=False)
    html = open(out, encoding="utf-8").read()
    assert "brand_new_source" in html and "Nobody" in html


def test_status_shows_the_error_when_coverage_cannot_be_computed(tmp_path, monkeypatch):
    root = _root(tmp_path)

    def boom(_root):
        raise RuntimeError("synthetic coverage failure")
    monkeypatch.setattr(status, "_source_coverage", boom)
    ctx = status.status_json(root)
    assert ctx["source_coverage"]["error"] == "RuntimeError: synthetic coverage failure"
    out = str(tmp_path / "index.html")
    status.render(root, out_path=out, register=False)
    assert "coverage not computed" in open(out, encoding="utf-8").read()


def test_the_full_width_mmlu_read_is_its_own_provisional_source_read_by_the_cellstore_module(covered):
    """The Qwen MMLU read at the full 1,531-item width (`mmlu_full`, 2026-09-21): covered by exactly one adapter -- `cellstore.MmluFullAdapter`,
    registered under the source's name with the generic reader's priority (no other reader mints its keys: a new instrument makes new run ids) --
    never by the generic `cellstore` adapter or `orxs_seedrep`, whose roots and bytes do not move."""
    from sisteer.registry.adapters import base as AB
    from sisteer.registry.adapters import cellstore as CS
    assert covered[CS.MMLU_FULL_SOURCE] == [("mmlu_full", IC.PRIORITY_GENERIC_READER)], covered.get(CS.MMLU_FULL_SOURCE)
    assert CS.MMLU_FULL_SOURCE == "mmlu_full" == CS.MmluFullAdapter.name and CS.MMLU_FULL_SOURCE in PROVISIONAL_ADDITIONS
    assert CS.MMLU_FULL_SOURCE in CS.MmluFullAdapter.covers_sources and CS.MMLU_FULL_SOURCE not in CS.OrxsSeedrepAdapter.covers_sources
    assert CS.MMLU_FULL_SOURCE not in CS.CellStoreAdapter().covers_sources
    ad = CS.MmluFullAdapter()
    assert AB.owner_of(ad) == "Summary" == CS.MMLU_FULL_OWNER and ad.status_sources("any group") == (CS.MMLU_FULL_SOURCE,)


def test_the_padded_static_conditions_are_their_own_provisional_source_on_padded_instruments(covered):
    """The padded static conditions (`static_padding`, 2026-09-23): covered by exactly one adapter -- `static_padding.StaticPaddingAdapter`, at the
    paper-source priority -- never by the sampled statics reader (whose runs are the unpadded instrument's). The pad is INSTRUMENT identity
    (migration 0012), so the source's runs share no measurement key with the unpadded rows; since migration 0014 a pad of 0 is the
    recipe's measured ZERO POINT, filed with the padded siblings' filler contract (a value, never NULL), and a zero-pad store that declares no
    recipe, has no padded sibling, or carries a different contract refuses by name."""
    from sisteer.registry.adapters import base as AB
    from sisteer.registry.adapters import sampled_static as SS
    from sisteer.registry.adapters import static_padding as SP
    assert covered[SP.SOURCE] == [("static_padding", IC.PRIORITY_PAPER_SOURCE)], covered.get(SP.SOURCE)
    assert SP.SOURCE == "static_padding" == SP.StaticPaddingAdapter.name and SP.SOURCE in PROVISIONAL_ADDITIONS
    assert SP.SOURCE in SP.StaticPaddingAdapter.covers_sources and SP.SOURCE not in SS.SampledStaticAdapter.covers_sources
    ad = SP.StaticPaddingAdapter(root="/nonexistent/static_padding")
    assert AB.owner_of(ad) == "Pad" == SP.OWNER and ad.status_sources("any group") == (SP.SOURCE,)
    assert SP.ZERO_POINT_SIDECAR == "static_padding.zero_point.json" and SP.ZERO_POINT_KIND == "static_padding_zero_point/1"
    assert "declares its recipe nowhere" in SP.ZERO_POINT_NO_DECLARATION_REASON
    assert "no padded sibling" in SP.ZERO_POINT_NO_SIBLING_REASON
    assert SP.ZERO_POINT_SIBLING_REASON == "the zero point's filler contract is not its padded siblings'"
    assert not hasattr(SP, "PAD_ZERO_REASON")     # the pre-0014 refusal of a pad of 0 is gone: 0 with its contract is the zero point
    assert SP.MISSING_STORE_REASON == "store not mirrored under the durable root"


# --------------------------------------------------------------------------------------------
# ctl_neg1: the control-negative battery (2026-09-25) -- one lane source, two readers at distinct priorities
# --------------------------------------------------------------------------------------------

def test_ctl_neg1_is_declared_provisionally_and_covered_by_the_mirror_root_readers_at_distinct_priorities(covered):
    """The battery's rows come from three readers over its mirror root: the multi-turn grades (`strength_mtdiff.CtlNeg1MultiturnAdapter` over
    `ctl_neg1/mt/grades/`), the over-refusal controls (`cellstore.CtlNeg1ControlsAdapter` over `ctl_neg1/sampled/controls_sampled/`) and the
    statics (`sampled_static.CtlNeg1StaticAdapter` over `ctl_neg1/sampled/`) -- all covering `ctl_neg1` at DISTINCT priorities, the coverage
    rule for several readers of one source, under owner Scale. The dose ladder's own reader and the paper's sampled reader keep their
    sources, roots and owners."""
    from sisteer.registry.adapters import cellstore as CS
    from sisteer.registry.adapters import sampled_static as SS
    from sisteer.registry.adapters.base import owner_of
    assert "ctl_neg1" == PROVISIONAL_ADDITIONS[8] and CS.CTL_NEG1_SOURCE == SS.CTL_NEG1_SOURCE == SM.CTL_NEG1_SOURCE == "ctl_neg1"
    claims = sorted(covered["ctl_neg1"])
    assert [n for n, _ in claims] == ["ctl_neg1_controls", "ctl_neg1_multiturn", "ctl_neg1_static"] and len({p for _, p in claims}) == 3, claims
    assert CS.CtlNeg1ControlsAdapter.covers_sources == SS.CtlNeg1StaticAdapter.covers_sources == SM.CtlNeg1MultiturnAdapter.covers_sources == ("ctl_neg1",)
    assert owner_of(CS.CtlNeg1ControlsAdapter()) == owner_of(SS.CtlNeg1StaticAdapter()) == owner_of(SM.CtlNeg1MultiturnAdapter()) == "Scale"
    mirror = os.path.join(IC.RUN_STATE, "ctl_neg1")
    assert SS.CtlNeg1StaticAdapter().root == os.path.join(mirror, "sampled") == CS.CTL_NEG1_ROOT
    assert all(p.startswith(os.path.join(mirror, "sampled", "controls_sampled", "controls_cap")) for p in CS.CtlNeg1ControlsAdapter().roots)
    assert SM.CtlNeg1MultiturnAdapter().root == os.path.join(mirror, "mt", "grades") == SM.CTL_NEG1_ROOT_DEFAULT
    assert covered["strength_mtdiff_s1"] == [("strength_mtdiff", 0)] and covered["sampled_lane"] == [("sampled_static", IC.PRIORITY_OWNING_READER)]   # the twins' owner since migration 0026
    assert (SM.StrengthMtdiffAdapter.source, SM.StrengthMtdiffAdapter.owner, SM.StrengthMtdiffAdapter().root) == ("strength_mtdiff_s1", "Strength", SM.ROOT_DEFAULT)
    assert SS.SampledStaticAdapter.source == "sampled_lane" and SS.SampledStaticAdapter().root == SS.SAMPLED_ROOT
    with open(DECLARED, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    assert {r["source"]: r["owner"] for r in data["sources"]}["ctl_neg1"] == "the authors"
    # several readers of one source at distinct priorities is not a double claim (the coverage check's rule)
    assert not [t for t in SRC.coverage({"ctl_neg1": ["synthetic"]}, covered).get("twice", []) if t.get("source") == "ctl_neg1"]


def test_landing_2_of_the_shared_standup_stores_is_its_own_provisional_source_pair_read_by_the_l2_readers_with_distinct_run_ids():
    """Landing 2 (2026-09-30): the SAME comparison tuples as `bigcodebench_full_shared` / `bfcl_full_shared` driven again by another harness
    (K = 2 drivers per row + a CPU verification pass) -- TWO sources of their own, `bigcodebench_full_shared_l2` / `bfcl_full_shared_l2`, each
    covered by its own reader over its own mirror root (landing 1's readers parametrized once more, `_bcbw_common.SharedSpec.harness_variant`), never landing 1's
    sources (a second run of one tuple under them would collide on the run id) and never the solo readers'. The run id's discriminator is the producer key's
    source segment (`<tag>_shared_l2/<store key>`), so every `_l2` id differs from landing 1's over one store key while the spelling stays `<tag>_shared`."""
    from sisteer.registry.adapters import _bcbw_common as B
    from sisteer.registry.adapters import bfcl_full as BFF
    from sisteer.registry.adapters import bfcl_full_shared as BFS
    from sisteer.registry.adapters import bfcl_full_shared_l2 as BFS2
    from sisteer.registry.adapters import bigcodebench_full as BCF
    from sisteer.registry.adapters import bigcodebench_full_shared as BCS
    from sisteer.registry.adapters import bigcodebench_full_shared_l2 as BCS2
    covered = SRC.covered_sources(ADP.load_all())
    assert covered[BCS2.SOURCE] == [("bigcodebench_full_shared_l2", 0)] and covered[BFS2.SOURCE] == [("bfcl_full_shared_l2", 0)], (covered.get(BCS2.SOURCE), covered.get(BFS2.SOURCE))
    assert BCS2.SOURCE == "bigcodebench_full_shared_l2" == PROVISIONAL_ADDITIONS[4] and BFS2.SOURCE == "bfcl_full_shared_l2" == PROVISIONAL_ADDITIONS[5]
    assert BCS.SOURCE == "bigcodebench_full_shared" == PROVISIONAL_ADDITIONS[6] and BFS.SOURCE == "bfcl_full_shared" == PROVISIONAL_ADDITIONS[7]
    for l2, l1, solo in ((BCS2, BCS, BCF), (BFS2, BFS, BFF)):
        cls2 = l2.BigcodebenchFullSharedL2Adapter if l2 is BCS2 else l2.BfclFullSharedL2Adapter
        cls1 = l1.BigcodebenchFullSharedAdapter if l1 is BCS else l1.BfclFullSharedAdapter
        assert issubclass(cls2, B.BcbwSeedrepReader) and cls2.covers_sources == (l2.SOURCE,) and l2.SOURCE not in cls1.covers_sources and l1.SOURCE not in cls2.covers_sources
        assert l2.ROOT_DEFAULT.endswith(os.path.join("run-state", "full22", l2.SOURCE)) and l2.ROOT_DEFAULT not in (l1.ROOT_DEFAULT, solo.ROOT_DEFAULT)
        assert l2.SPEC.shared.harness_variant == "l2" and l2.SPEC.shared.base_source == l1.SOURCE and l1.SPEC.shared.harness_variant is None
        assert l2.spelling_of("qwen35_4b_s1") == l1.spelling_of("qwen35_4b_s1") == "qwen35_4b_s1_shared" and l2.producer_spelling_of("qwen35_4b_s1") == "qwen35_4b_s1_shared_l2"
        key, inst = "none__controls__t0__a0__2ff9cdff__1dd7b444", "synthetic_instrument_id_000000"
        assert len({l2.run_id_for("qwen35_4b_s1", key, "none", inst), l1.run_id_for("qwen35_4b_s1", key, "none", inst), solo.run_id_for("qwen35_4b_s1", key, "none", inst)}) == 3
        assert cls2(root="/nonexistent/x").status_sources("qwen35_4b_s1") == (l2.SOURCE,)


def test_landing_3_of_the_shared_standup_stores_is_its_own_provisional_source_pair_read_by_the_l3_readers_with_distinct_run_ids():
    """Landing 3 (a fan-out test on another harness, 2026-10-01): the gemma4e4b s1 row's four tuples driven once more -- TWO sources of their own,
    `bigcodebench_full_shared_l3` / `bfcl_full_shared_l3`, each covered by its own reader over its own mirror root (landing 1's readers parametrized as
    landing 2's are, `_bcbw_common.SharedSpec.harness_variant` l3 + `landing_rule`), never landing 1's or landing 2's sources (a third run of one tuple under either would
    collide on the run id) and never the solo readers'. The discriminator is the producer key's source segment (`<tag>_shared_l3/<store key>`), so every `_l3` id differs
    from landing 1's, landing 2's and the solo reader's over one store key while the spelling stays `<tag>_shared`."""
    from sisteer.registry.adapters import _bcbw_common as B
    from sisteer.registry.adapters import bfcl_full as BFF
    from sisteer.registry.adapters import bfcl_full_shared as BFS
    from sisteer.registry.adapters import bfcl_full_shared_l2 as BFS2
    from sisteer.registry.adapters import bfcl_full_shared_l3 as BFS3
    from sisteer.registry.adapters import bigcodebench_full as BCF
    from sisteer.registry.adapters import bigcodebench_full_shared as BCS
    from sisteer.registry.adapters import bigcodebench_full_shared_l2 as BCS2
    from sisteer.registry.adapters import bigcodebench_full_shared_l3 as BCS3
    covered = SRC.covered_sources(ADP.load_all())
    assert covered[BCS3.SOURCE] == [("bigcodebench_full_shared_l3", 0)] and covered[BFS3.SOURCE] == [("bfcl_full_shared_l3", 0)], (covered.get(BCS3.SOURCE), covered.get(BFS3.SOURCE))
    assert BCS3.SOURCE == "bigcodebench_full_shared_l3" == PROVISIONAL_ADDITIONS[2] and BFS3.SOURCE == "bfcl_full_shared_l3" == PROVISIONAL_ADDITIONS[3]
    for l3, l2, l1, solo in ((BCS3, BCS2, BCS, BCF), (BFS3, BFS2, BFS, BFF)):
        cls3 = l3.BigcodebenchFullSharedL3Adapter if l3 is BCS3 else l3.BfclFullSharedL3Adapter
        cls2 = l2.BigcodebenchFullSharedL2Adapter if l2 is BCS2 else l2.BfclFullSharedL2Adapter
        cls1 = l1.BigcodebenchFullSharedAdapter if l1 is BCS else l1.BfclFullSharedAdapter
        assert issubclass(cls3, B.BcbwSeedrepReader) and cls3.covers_sources == (l3.SOURCE,)
        assert l3.SOURCE not in cls1.covers_sources and l3.SOURCE not in cls2.covers_sources and l1.SOURCE not in cls3.covers_sources and l2.SOURCE not in cls3.covers_sources
        assert l3.ROOT_DEFAULT.endswith(os.path.join("run-state", "full22", l3.SOURCE)) and l3.ROOT_DEFAULT not in (l1.ROOT_DEFAULT, l2.ROOT_DEFAULT, solo.ROOT_DEFAULT)
        assert l3.SPEC.shared.harness_variant == "l3" == B.SHARED_L3_HARNESS_VARIANT and l3.SPEC.shared.base_source == l1.SOURCE and l3.SPEC.shared.landing_rule == B.SHARED_L3_RULE
        assert l2.SPEC.shared.landing_rule is None and l1.SPEC.shared.landing_rule is None and l1.SPEC.shared.harness_variant is None
        assert l3.spelling_of("gemma4e4b_s1") == l1.spelling_of("gemma4e4b_s1") == "gemma4e4b_s1_shared" and l3.producer_spelling_of("gemma4e4b_s1") == "gemma4e4b_s1_shared_l3"
        key, inst = "none__controls__t0__a0__2ff9cdff__1dd7b444", "synthetic_instrument_id_000000"
        ids = (l3.run_id_for("gemma4e4b_s1", key, "none", inst), l2.run_id_for("gemma4e4b_s1", key, "none", inst), l1.run_id_for("gemma4e4b_s1", key, "none", inst),
               solo.run_id_for("gemma4e4b_s1", key, "none", inst))
        assert len(set(ids)) == 4 and l3.landing2_run_id_for("gemma4e4b_s1", key, "none", inst) == ids[1] and l3.landing1_run_id_for("gemma4e4b_s1", key, "none", inst) == ids[2]
        assert cls3(root="/nonexistent/x").status_sources("gemma4e4b_s1") == (l3.SOURCE,)
