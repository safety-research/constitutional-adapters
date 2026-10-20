"""relu_native declares the retired paired root.

``PAIRED_ROOT`` moved off the ReLU working tree into the run-state tree, every file byte-identical. The
2026-09-21 accepted heads recorded the OLD worktree URIs in their inputs_json, so every read since presented a disjoint input set and the
runner's inputs-dropped guard HELD the four groups (`inputs_dropped_unexplained` 64-88 % against the 10 % threshold): a rename is a
retirement and is never inferred (runner.INPUTS_DROPPED_RULE). The adapter now declares it, exactly as arditi_vendor_mt declares its
volatile working trees: ``RETIRED_PAIRED_ROOT`` (the heads' root) and ``inputs_retired(group)`` naming the group's paired records under it,
so the comparison counts them `inputs_retired`, never `inputs_dropped`, and does not trip. Nothing is re-pointed; no symlink; no seed moves.
"""
from __future__ import annotations

import hashlib
import json
import os

from sisteer.registry.adapters import base as AB
from sisteer.registry.adapters import declared_common as DC
from sisteer.registry.adapters import relu_native as RN
from sisteer.registry.adapters import runner as RUN


def test_the_retired_root_is_the_heads_root_and_the_declaration_names_the_groups_paired_records():
    assert RN.RETIRED_PAIRED_ROOT == os.path.join(DC.HOME, "si-steering", "worktrees", "relu-native-P", "results", "relu_steering_native", "paired")
    assert RN.RETIRED_PAIRED_ROOT != RN.PAIRED_ROOT and RN.PAIRED_ROOT.startswith(DC.RUN_STATE) and "worktrees" not in RN.PAIRED_ROOT
    ad = RN.ReluNativeAdapter()
    assert AB.inputs_retired_of(ad, "llama8b") == (os.path.join(RN.RETIRED_PAIRED_ROOT, "llama8b__*.json"),)
    for model in ("gemma4e4b", "qwen35_4b", "nemotron3nano4b"):
        assert AB.inputs_retired_of(ad, model) == (os.path.join(RN.RETIRED_PAIRED_ROOT, f"{model}__*.json"),), model
    assert AB.inputs_retired_of(ad, "relu") == (os.path.join(RN.RETIRED_PAIRED_ROOT, "relu__*.json"),)      # a grades-only group retires no paired record that exists


def test_the_comparison_over_the_heads_old_root_counts_the_same_files_at_the_new_root_as_retired_not_dropped(tmp_path):
    """The previous accepted revision recorded N paired records under the OLD root; the read presents the same N files (same bytes, same
    shas) under the NEW root. Before the declaration: N dropped, 100 % > 10 %, tripped. With it: dropped 0, retired N, tripped False,
    the N new-root files `inputs_added`."""
    new_root = str(tmp_path / "rs" / "relu_native" / "paired")
    os.makedirs(new_root)
    model, evals = "llama8b", ("pasttense", "actorattack", "gsm8k")
    prev_inputs = {}
    inputs = []
    for i, ev in enumerate(evals):
        name = f"{model}__{ev}__constv2-mix-full__lora_r4_ce.json"
        body = json.dumps({"model": model, "eval": ev, "i": i}).encode()
        with open(os.path.join(new_root, name), "wb") as fh:
            fh.write(body)
        prev_inputs[os.path.join(RN.RETIRED_PAIRED_ROOT, name)] = hashlib.sha256(body).hexdigest()
        inputs.append(AB.check_input(os.path.join(new_root, name), role="report"))
    ad = RN.ReluNativeAdapter(rs=str(tmp_path / "rs" / "relu_native"), paired_root=new_root)
    assert {i.sha256 for i in inputs} == set(prev_inputs.values()), "byte-identical: the same shas under the new root"
    cmp = RUN.inputs_dropped_comparison(ad, model, inputs, "rev_head_0921", prev_inputs)
    assert cmp is not None and cmp["inputs_previous"] == 3 and cmp["inputs_now"] == 3
    assert cmp["inputs_dropped"] == 0 and cmp["inputs_retired"] == 3 and cmp["inputs_added"] == 3 and cmp["tripped"] is False, cmp
    assert cmp["inputs_dropped_fraction"] == 0.0 and cmp["inputs_dropped_paths"] == []
    # without the declaration the same comparison trips: that is the hold the four groups sat in since 2026-09-29
    bare = RN.ReluNativeAdapter(rs=str(tmp_path / "rs" / "relu_native"), paired_root=new_root)
    bare.inputs_retired = lambda group: []
    cmp0 = RUN.inputs_dropped_comparison(bare, model, inputs, "rev_head_0921", prev_inputs)
    assert cmp0["inputs_dropped"] == 3 and cmp0["inputs_retired"] == 0 and cmp0["inputs_dropped_fraction"] == 1.0 and cmp0["tripped"] is True
    # another group's declaration does not cover this group's records
    cmp_other = RUN.inputs_dropped_comparison(ad, "gemma4e4b", inputs, "rev_head_0921", prev_inputs)
    assert cmp_other["inputs_dropped"] == 3 and cmp_other["inputs_retired"] == 0
