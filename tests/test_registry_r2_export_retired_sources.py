"""r2_export declares the one sha-pinned source the producer moved on 2026-10-07 retired.

``scripts/r2_export/build_export/sdf_base_steering_r2_sources.json`` pins ``$STAGE/sdfbase/fullscale/consolidated/judge/
nemotron3nano4b/judge_paired_vs_none.json`` (sha256 cbc93e0273da3434…) as a hashed input of the 09-14 export; the consolidated judge tree
moved WHOLE to ``$STAGE/sdfbase/fullscale/_retired/judge/`` (bytes unchanged, nothing deleted; its README_retired.md is the
provenance). A MISSING pinned source is already scoped to the rows naming it (``_missing_sources`` -> row refusals by name,
``status_extra["missing"]``, an ``artifact(kind=report)``) and that stays. But the accepted head recorded the file among its inputs, and
on a CLUSTER shard ANY undeclared dropped input refuses the whole group (``runner.ClusterInputDropRefusal``, ``staged_copy_incomplete: 1
of the … inputs … absent``), so ``r2_export:r2`` was refused in every window since the move. The adapter now declares it:
``RETIRED_SOURCES`` + ``inputs_retired(group)``, so the comparison counts the moved file `inputs_retired`, never `inputs_dropped`, and
``status_extra`` surfaces the declaration beside ``missing`` (``retired_sources``). The producer's ``_sources.json`` is not touched.
"""
from __future__ import annotations

import hashlib
import os

from sisteer.registry.adapters import base as AB
from sisteer.registry.adapters import lineage_common as LC
from sisteer.registry.adapters import r2_export as X
from sisteer.registry.adapters import runner as RUN

#: the moved pin, under the staging directory ($STAGE: the suite's site value, or the host's)
MOVED_REL = "sdfbase/fullscale/consolidated/judge/nemotron3nano4b/judge_paired_vs_none.json"
MOVED = AB.site_root("$STAGE/" + MOVED_REL)
MOVED_SHA = "cbc93e0273da3434" + "0" * 48
PREV = "rev_head_2bcd1243"


def _adapter(tmp_path):
    """A bare adapter (seeds and objects stay lazy; nothing is read): the declaration and the status facts need no export."""
    return X.R2ExportAdapter(data_dir=str(tmp_path / "data"), rs=str(tmp_path / "rs"))


def test_the_retired_sources_name_the_moved_pin_in_the_builders_spelling_expanded():
    assert X.RETIRED_SOURCES == (MOVED,)
    assert all(os.path.isabs(p) and "_retired" not in p for p in X.RETIRED_SOURCES), "the OLD path is declared, never the new one"
    assert MOVED.endswith("/" + MOVED_REL) and "$" not in MOVED, "the declaration expands the site's staging directory"


def test_the_declaration_answers_the_one_group_and_the_empty_group_and_nothing_else(tmp_path):
    ad = _adapter(tmp_path)
    assert AB.inputs_retired_of(ad, "r2") == (MOVED,)
    # `stores coverage` reads the declaration with the EMPTY group (stores._retired_globs): the same answer
    assert AB.inputs_retired_of(ad, "") == (MOVED,)
    assert AB.inputs_retired_of(ad, "other") == ()


def test_the_comparison_counts_the_moved_pin_retired_not_dropped_and_trips_without_the_declaration(tmp_path):
    """The accepted head listed the parquet, the schema and the moved pin among its inputs; the read presents the first two. With the
    declaration: dropped 0, retired 1, tripped False. Without it: 1 of 3 dropped (33 % > 10 %), tripped -- the hold every window since the
    move sat in."""
    data = tmp_path / "data"
    data.mkdir()
    prev, inputs = {MOVED: MOVED_SHA}, []
    for name in (X.PARQUET, X.SCHEMA_JSON):
        body = f"synthetic {name}".encode()
        (data / name).write_bytes(body)
        prev[str(data / name)] = hashlib.sha256(body).hexdigest()
        inputs.append(AB.check_input(str(data / name), role="report"))
    ad = _adapter(tmp_path)
    cmp = RUN.inputs_dropped_comparison(ad, "r2", inputs, PREV, prev)
    assert cmp is not None and cmp["inputs_previous"] == 3 and cmp["inputs_now"] == 2 and cmp["inputs_added"] == 0
    assert cmp["inputs_dropped"] == 0 and cmp["inputs_retired"] == 1 and cmp["tripped"] is False, cmp
    assert cmp["inputs_dropped_fraction"] == 0.0 and cmp["inputs_dropped_paths"] == [] and cmp["inputs_dropped_uris"] == []
    bare = _adapter(tmp_path)
    bare.inputs_retired = lambda group: []
    cmp0 = RUN.inputs_dropped_comparison(bare, "r2", inputs, PREV, prev)
    assert cmp0["inputs_dropped"] == 1 and cmp0["inputs_retired"] == 0 and round(cmp0["inputs_dropped_fraction"], 4) == 0.3333 and cmp0["tripped"] is True
    assert cmp0["inputs_dropped_uris"] == [MOVED]
    # a present pin that goes absent is still DROPPED: the declaration names the moved file alone
    gone = {k: v for k, v in prev.items()}
    gone[str(data / "other_pin.json")] = "e" * 64
    cmp1 = RUN.inputs_dropped_comparison(ad, "r2", inputs, PREV, gone)
    assert cmp1["inputs_dropped"] == 1 and cmp1["inputs_retired"] == 1 and cmp1["inputs_dropped_paths"] == ["other_pin.json"]


def test_status_extra_carries_the_declared_retired_sources_beside_missing(tmp_path):
    """The status page can show which missing pin is declared: `retired_sources` in the same spelling `missing` uses."""
    ad = _adapter(tmp_path)
    st = ad.status_extra("r2")
    assert st["retired_sources"] == [MOVED]
    assert "missing" in st and "mismatched" in st and "row_refusals" in st and "stats" in st
    # a missing pin that is declared is in both lists, by the same string
    ad._missing_sources = {MOVED: {"sha256": MOVED_SHA, "rel": LC.rel_home(MOVED)}}
    st2 = ad.status_extra("r2")
    assert st2["missing"] == [MOVED] and set(st2["missing"]) & set(st2["retired_sources"]) == {MOVED}


def test_the_emission_version_is_not_bumped_for_the_declaration():
    """With the pin missing, the read already emits exactly what 12 emits (the rows naming it refused by name); the declaration changes
    only the comparison's count."""
    assert X.R2ExportAdapter.EMISSION_VERSION == 12
