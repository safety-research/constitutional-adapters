"""The two-axis + object-class reporting the held-out manifest carries.

A held-out result must be readable along THREE distinct axes without conflation:
  - the GOAL-holdout axis (harm-category holdout) — transform-generalisation
    (held_in goals) vs full-generalisation (held_out goals), from the
    AUTHORITATIVE category split;
  - the defense-OBJECT class (never pooled across — objects intervene at
    different forward phases);
  - the TRANSFER axis (ingested artifacts optimised against another model).
Plus the goal-holdout COVERAGE GAP (uncategorised/wildjailbreak goals are train
by design). All-synthetic, CPU-only.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from sisteer.eval.attacks import artprompt
from sisteer.eval.attacks.heldout_common import (
    assert_manifest_safe,
    build_manifest,
    check_corpus_leakage,
    goal_axis_label,
    materialize_corpus,
    training_fingerprint,
    write_family_manifest,
)
from sisteer.eval.jailbreak_data import Goal

SPEC = artprompt.SPEC
FP = training_fingerprint(["an unrelated training prompt"])


def test_goal_axis_label_is_reader_friendly():
    # a reader must not have to know what held_in/held_out mean
    assert goal_axis_label("held_in") == "transform_generalisation"
    assert goal_axis_label("held_out") == "full_generalisation"
    assert goal_axis_label(None) == "uncategorised"
    assert goal_axis_label("") == "uncategorised"
    assert goal_axis_label("nonsense") == "uncategorised"


def _goals():
    return [
        Goal("g_in", "trained category goal", "Malware/Hacking", "harmbench", "r"),
        Goal("g_out", "untrained category goal", "Privacy", "harmbench", "r"),
        Goal("g_uncat", "no category goal", "", "wildjailbreak", "r"),
    ]


def test_manifest_breaks_out_goal_axis_from_authoritative_split():
    goals = _goals()
    rows = materialize_corpus(SPEC.variants_fn, goals)
    # the authoritative split: g_in -> held_in, g_out -> held_out, g_uncat absent
    group_of = {"g_in": "held_in", "g_out": "held_out"}
    man = build_manifest(SPEC, rows, seed=0,
                         leakage=check_corpus_leakage(rows, FP, refuse=False),
                         category_group_of=group_of,
                         object_class="si_distilled", cell="safety_refusal__psr_kl_all",
                         arm="psr_kl_all")
    axis = man["goal_axis"]
    assert axis["transform_generalisation"]["n_behaviors"] == 1  # g_in
    assert axis["full_generalisation"]["n_behaviors"] == 1       # g_out
    assert axis["uncategorised"]["n_behaviors"] == 1             # g_uncat (train by design)
    # object axis + coverage gap stamped, and the whole thing stays servable
    assert man["object_class"] == "si_distilled"
    assert man["cell"] == "safety_refusal__psr_kl_all" and man["arm"] == "psr_kl_all"
    assert man["goal_holdout_coverage"] == "categorised_harmbench_jbb_only"
    assert man["build_kind"] == "deterministic"
    assert_manifest_safe(man)  # no prompt/goal text rides along


def test_manifest_omits_goal_axis_when_split_unavailable():
    rows = materialize_corpus(SPEC.variants_fn, _goals())
    man = build_manifest(SPEC, rows, seed=0,
                         leakage=check_corpus_leakage(rows, FP, refuse=False))
    assert "goal_axis" not in man  # no split on this node -> not fabricated
    assert man["goal_holdout_coverage"] == "categorised_harmbench_jbb_only"


def test_write_family_manifest_two_axis(tmp_path):
    goals = _goals()
    man = write_family_manifest(
        SPEC, goals, manifest_path=tmp_path / "m.json", fingerprint=FP,
        category_group_of={"g_in": "held_in", "g_out": "held_out"},
        object_class="si_agnostic_baseline", cell="antidote", arm="antidote")
    on_disk = json.loads((tmp_path / "m.json").read_text())
    assert on_disk["object_class"] == "si_agnostic_baseline"
    assert on_disk["goal_axis"]["full_generalisation"]["n_behaviors"] == 1
    assert man["leakage_check"]["passed"] is True


def test_object_class_label_reads_from_registry():
    """The driver's object-class labelling reads defense_cell, not a raw arm
    string — so si-distilled cells, SI-agnostic baselines, and baselines are
    never pooled together."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import importlib

    drv = importlib.import_module("unified_sweep_run")
    assert drv._object_class_label("none") == "baseline"
    assert drv._object_class_label("prompted__safety_refusal") == "baseline"
    # distinct class PER SI ROLE: the safety SI,
    # the semantic placebo, and si_constitution must never pool
    assert (drv._object_class_label("safety_refusal__psr_kl_all")
            == "si_distilled:safety_relevant")
    assert (drv._object_class_label("si_irrelevant__vector")
            == "si_distilled:null_control_placebo")
    # Class B (no teacher floor): evaluated on its own terms
    assert drv._object_class_label("antidote") == "content_based_arm"
    assert drv._object_class_label("alphasteer") == "content_based_arm"
