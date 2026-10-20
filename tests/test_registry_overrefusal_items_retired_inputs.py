"""overrefusal_items declares the inputs the producer moved on 2026-10-07 retired.

Two groups of the source vanished when their inputs moved -- bytes unchanged, nothing deleted, a README at each new path: the consolidated
judge tree (``OR_ROOTS[1]``, ``$STAGE/sdfbase/fullscale/consolidated/judge/``, moved WHOLE to ``fullscale/_retired/judge/``; its one
input, ``nemotron3nano4b/judge_paired_vs_none.json``, names a store that never existed under it) and one incomplete store under the
controls_halfdose root (``gemma4e4b_halfdose/cells/<store>`` moved to ``gemma4e4b_halfdose/_incomplete/<store>``: manifest.json +
evidence.json only; Slurm job 463796 died on a FidelityError and no numbers were filed). The watermarks the earlier reads left record the
old paths, and on a CLUSTER shard ANY undeclared dropped input refuses the group (``runner.ClusterInputDropRefusal``,
``staged_copy_incomplete``), so both groups were refused in every window since the move: a rename is a retirement and is declared, never
inferred (runner.INPUTS_DROPPED_RULE). The adapter now declares them, as relu_native declares ``RETIRED_PAIRED_ROOT``:
``RETIRED_JUDGE_ROOT`` + ``INCOMPLETE_STORES`` and ``inputs_retired(group)``, so the comparison counts the moved files `inputs_retired`,
never `inputs_dropped`. Nothing is re-pointed: ``OR_ROOTS`` is unchanged (paperlib parity; discovery skips a missing root) and the adapter
reads nothing from the new paths.
"""
from __future__ import annotations

import os

from sisteer import site as SITE
from sisteer.registry.adapters import _items_common as IC
from sisteer.registry.adapters import base as AB
from sisteer.registry.adapters import overrefusal_items as OR
from sisteer.registry.adapters import runner as RUN

JUDGE_GROUP = os.path.join(OR.RETIRED_JUDGE_ROOT, "nemotron3nano4b")
INCOMPLETE_STORE = OR.INCOMPLETE_STORES[0]
#: a store of the same target directory that still stands under its cells/ (the unsteered one): the declaration must not reach it
SIBLING_STORE = os.path.join(os.path.dirname(INCOMPLETE_STORE), "none__controls__t0__a0__2ff9cdff__09b5a19c")
PREV = "rev_prev_0919"


def test_the_retired_judge_root_is_or_roots_1_and_the_incomplete_store_lies_under_the_controls_halfdose_root():
    assert OR.RETIRED_JUDGE_ROOT == OR.OR_ROOTS[1] == str(SITE.expand_path("$STAGE/sdfbase/fullscale/consolidated/judge"))
    assert len(OR.OR_ROOTS) == 3, "OR_ROOTS keeps the spelling (paperlib parity): the retired root is declared, not removed"
    assert OR.INCOMPLETE_STORES == (os.path.join(IC.RUN_STATE, "controls_halfdose", "gemma4e4b_halfdose", "cells",
                                                 "constv2-mix-full--lora_r4_ce_minus_ctl__controls__t0__a0__8ebbf872__e5868612"),)
    for store in OR.INCOMPLETE_STORES:
        assert store.startswith(OR.OR_ROOTS[2] + os.sep) and os.path.basename(os.path.dirname(store)) == "cells", store
        assert "_incomplete" not in store, "the OLD path is declared (the one the watermarks recorded), never the new one"


def test_the_declaration_names_the_two_vanished_groups_and_nothing_for_an_ordinary_group():
    ad = OR.OverrefusalItemsAdapter()
    judge_glob = os.path.join(OR.RETIRED_JUDGE_ROOT, "*")
    assert AB.inputs_retired_of(ad, JUDGE_GROUP) == (judge_glob,)
    assert AB.inputs_retired_of(ad, OR.RETIRED_JUDGE_ROOT) == (judge_glob,)                       # the root itself, were a file to lie there
    assert AB.inputs_retired_of(ad, os.path.join(OR.RETIRED_JUDGE_ROOT, "glm46vflash")) == (judge_glob,)   # the empty sibling directory
    assert AB.inputs_retired_of(ad, INCOMPLETE_STORE) == (os.path.join(INCOMPLETE_STORE, "*"),)
    # (c) an ordinary store group under the same controls_halfdose root, still on disk, retires nothing; neither does its target-level
    # group, nor a store group under the first root, nor a path that merely shares the judge root's prefix
    assert AB.inputs_retired_of(ad, SIBLING_STORE) == ()
    assert AB.inputs_retired_of(ad, os.path.dirname(os.path.dirname(INCOMPLETE_STORE))) == ()
    assert AB.inputs_retired_of(ad, os.path.join(OR.OR_ROOTS[0], "nemotron3nano4b", "cells", "some_store")) == ()
    assert AB.inputs_retired_of(ad, OR.RETIRED_JUDGE_ROOT + "_other") == ()
    # `stores coverage` asks with the EMPTY group (stores._retired_globs: no comparison group, never a discovery): the whole declaration
    empty = set(AB.inputs_retired_of(ad, ""))
    assert judge_glob in empty and os.path.join(INCOMPLETE_STORE, "*") in empty and len(empty) == 1 + len(OR.INCOMPLETE_STORES)
    # an adapter constructed over other roots (a test's tmp roots) declares the same: the declaration is about the producer's paths
    assert AB.inputs_retired_of(OR.OverrefusalItemsAdapter(roots=("/nonexistent/a", "/nonexistent/b")), JUDGE_GROUP) == (judge_glob,)


def test_the_comparison_over_the_vanished_judge_group_counts_its_one_input_retired_not_dropped():
    """The earlier reads recorded the judge group's one input; the read presents none (the tree moved whole, outside every root). Without
    the declaration: 1 of 1 dropped, 100 % > 10 %, tripped -- and on a cluster shard any undeclared drop refuses the group. With it:
    dropped 0, retired 1, tripped False, nothing named."""
    ad = OR.OverrefusalItemsAdapter()
    prev = {os.path.join(JUDGE_GROUP, "judge_paired_vs_none.json"): "cbc93e0273da3434" + "0" * 48}
    cmp = RUN.inputs_dropped_comparison(ad, JUDGE_GROUP, [], PREV, prev)
    assert cmp is not None and cmp["previous_revision_id"] == PREV and cmp["inputs_previous"] == 1 and cmp["inputs_now"] == 0
    assert cmp["inputs_dropped"] == 0 and cmp["inputs_retired"] == 1 and cmp["inputs_added"] == 0 and cmp["tripped"] is False, cmp
    assert cmp["inputs_dropped_fraction"] == 0.0 and cmp["inputs_dropped_paths"] == [] and cmp["inputs_dropped_uris"] == []
    bare = OR.OverrefusalItemsAdapter()
    bare.inputs_retired = lambda group: []
    cmp0 = RUN.inputs_dropped_comparison(bare, JUDGE_GROUP, [], PREV, prev)
    assert cmp0["inputs_dropped"] == 1 and cmp0["inputs_retired"] == 0 and cmp0["inputs_dropped_fraction"] == 1.0 and cmp0["tripped"] is True
    assert cmp0["inputs_dropped_paths"] == ["nemotron3nano4b/judge_paired_vs_none.json"]        # named relative to the adapter's root


def test_the_comparison_over_the_moved_incomplete_store_counts_both_files_retired_and_leaves_a_sibling_store_alone():
    """The store's two files (manifest.json, evidence.json) left cells/ for _incomplete/; the read presents none of its inputs. With the
    declaration: dropped 0, retired 2. Without it: 2 of 2 dropped, tripped. And (c): the files of a sibling store that still stands under
    the same cells/ are DROPPED when absent -- the declaration covers the moved store's path alone, whichever group is asked."""
    ad = OR.OverrefusalItemsAdapter()
    prev = {os.path.join(INCOMPLETE_STORE, "manifest.json"): "c67fb014f0a610d4" + "0" * 48,
            os.path.join(INCOMPLETE_STORE, "evidence.json"): "f1fab2b74a5d52b2" + "0" * 48}
    cmp = RUN.inputs_dropped_comparison(ad, INCOMPLETE_STORE, [], PREV, prev)
    assert cmp["inputs_previous"] == 2 and cmp["inputs_now"] == 0 and cmp["inputs_dropped"] == 0 and cmp["inputs_retired"] == 2
    assert cmp["tripped"] is False and cmp["inputs_dropped_paths"] == [], cmp
    bare = OR.OverrefusalItemsAdapter()
    bare.inputs_retired = lambda group: []
    cmp0 = RUN.inputs_dropped_comparison(bare, INCOMPLETE_STORE, [], PREV, prev)
    assert cmp0["inputs_dropped"] == 2 and cmp0["inputs_retired"] == 0 and cmp0["inputs_dropped_fraction"] == 1.0 and cmp0["tripped"] is True
    prev_sib = {os.path.join(SIBLING_STORE, "manifest.json"): "a" * 64, os.path.join(SIBLING_STORE, "evidence.json"): "b" * 64}
    cmp_sib = RUN.inputs_dropped_comparison(ad, SIBLING_STORE, [], PREV, prev_sib)
    assert cmp_sib["inputs_dropped"] == 2 and cmp_sib["inputs_retired"] == 0 and cmp_sib["tripped"] is True
    cmp_cross = RUN.inputs_dropped_comparison(ad, INCOMPLETE_STORE, [], PREV, prev_sib)
    assert cmp_cross["inputs_dropped"] == 2 and cmp_cross["inputs_retired"] == 0 and cmp_cross["tripped"] is True
    # and the sibling's files, present, are not counted anything when the moved store's files are the only absent ones
    both = dict(prev, **prev_sib)
    present = [AB.Input(uri=u, path=u, role="report", sha256=s, size=1, mtime_utc="2026-01-01T00:00:00.000Z") for u, s in prev_sib.items()]
    cmp_both = RUN.inputs_dropped_comparison(ad, INCOMPLETE_STORE, present, PREV, both)
    assert cmp_both["inputs_previous"] == 4 and cmp_both["inputs_now"] == 2 and cmp_both["inputs_dropped"] == 0 and cmp_both["inputs_retired"] == 2


def test_the_emission_version_is_not_bumped_for_the_declaration():
    """The read of every remaining group is unchanged byte for byte; the declaration only tells the comparison that two vanished groups'
    inputs left on purpose. A bump would re-read every group of this reader_traversal adapter for nothing."""
    assert OR.OverrefusalItemsAdapter.EMISSION_VERSION == 7   # main bumped 5->7 for 27b; the declaration adds no bump of its own
