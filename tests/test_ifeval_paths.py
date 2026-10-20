"""The canonical IFEval battery — ``ifeval_paths`` is THE ONE path constructor.

Over the FROZEN matrix x 3 rungs: every ``out_root``, ``subset``, ``cap_ladder``, ``ladder_state``, ``budget`` and
``dispatch`` path is UNIQUE (a matrix-wide set-size assertion); the matrix's recorded paths ARE the constructor's strings
(the generator, the launcher's ``out_dir``, the planner's subset writer, the chain reader and the resume validator all call
it); Appendix B's layout regenerates from the constructor; a launcher SLUG never appears as a path component; the
constructor refuses an unknown model, a rung off the ladder and a malformed cell.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sisteer.capeval.tinycap import ifeval_contracts as IC

REPO = Path(__file__).resolve().parents[1]
MATRIX = json.loads((REPO / IC.IFEVAL_MATRIX_REL).read_text(encoding="utf-8"))
ROWS = MATRIX["rows"]
PATH_KINDS = ("out_root", "subset", "cap_ladder", "ladder_state", "budget", "dispatch")


def _all_paths():
    out = {k: [] for k in PATH_KINDS}
    for r in ROWS:
        for rung in IC.IFEVAL_CAP_LADDER:
            p = IC.ifeval_paths(r["model_key"], r["view"], r["cell"], rung)
            for k in PATH_KINDS:
                v = getattr(p, k)
                if v is not None:
                    out[k].append(v)
    return out


def test_every_path_kind_is_unique_over_the_matrix_times_three_rungs():
    paths = _all_paths()
    n_rows, n_rungs = len(ROWS), len(IC.IFEVAL_CAP_LADDER)
    assert n_rows == 40 and n_rungs == 3          # 30 launch + 2 held + 8 deferred
    for kind in ("out_root", "cap_ladder", "ladder_state", "budget", "dispatch"):
        assert len(paths[kind]) == n_rows * n_rungs, kind
        assert len(set(paths[kind])) == n_rows * n_rungs, f"{kind} paths collide"
    # pass 1 has no subset: 40 x 2 rung subsets, all distinct
    assert len(paths["subset"]) == n_rows * (n_rungs - 1) and len(set(paths["subset"])) == n_rows * (n_rungs - 1)
    # every path of every kind is distinct from every path of every other kind
    everything = [p for k in PATH_KINDS for p in paths[k]]
    assert len(set(everything)) == len(everything)
    # configuration ids are unique too
    ids = [IC.ifeval_paths(r["model_key"], r["view"], r["cell"], rung).configuration_id for r in ROWS for rung in IC.IFEVAL_CAP_LADDER]
    assert len(set(ids)) == len(ids) == n_rows * n_rungs


def test_the_matrix_records_exactly_the_constructors_strings_at_every_rung():
    for r in ROWS:
        p1 = IC.ifeval_paths(r["model_key"], r["view"], r["cell"], IC.IFEVAL_PASS1_CAP)
        assert r["out_root"] == p1.out_root and r["tag"] == p1.tag and r["cell_slug"] == p1.cell_slug
        assert p1.subset is None
        for rung in IC.IFEVAL_CAP_LADDER:
            p = IC.ifeval_paths(r["model_key"], r["view"], r["cell"], rung)
            assert r["cap_ladder_paths"][str(rung)] == p.cap_ladder
            assert r["ladder_state_paths"][str(rung)] == p.ladder_state
            assert r["budget_paths"][str(rung)] == p.budget
            assert r["dispatch_paths"][str(rung)] == p.dispatch
            assert r["configuration_ids"][str(rung)] == p.configuration_id
            if rung != IC.IFEVAL_PASS1_CAP:
                assert r["rung_out_roots"][str(rung)] == p.out_root and r["subset_paths"][str(rung)] == p.subset


def test_appendix_b_layout_regenerates_from_the_constructor_and_a_slug_is_never_a_component():
    p = IC.ifeval_paths("gemma4e4b", "main", "safety_refusal__psr_kl_all", 2048)
    assert p.out_root == "controls/gemma4e4b/safety_refusal--psr_kl_all/vendor_preset"
    assert p.cap_ladder == p.out_root + "/cap_ladder.json" and p.ladder_state == p.out_root + "/ladder_state.json"
    assert p.budget == "budgets/gemma4e4b/safety_refusal--psr_kl_all/2048/oos_prompt_budget.json"
    assert p.dispatch == "dispatch/gemma4e4b/safety_refusal--psr_kl_all/rung2048.json"
    r4 = IC.ifeval_paths("gemma4e4b", "main", "safety_refusal__psr_kl_all", 4096)
    assert r4.out_root == p.out_root + "__rung4096"
    assert r4.subset == "rung4096/gemma4e4b/safety_refusal--psr_kl_all/rung_subset.json"
    r8 = IC.ifeval_paths("llama8b", "it", "constv2ctl2-full__lora_r4_ce", 8192)
    assert r8.out_root == "controls/llama8b_it/constv2ctl2-full--lora_r4_ce/vendor_preset__rung8192"
    assert r8.subset == "rung8192/llama8b_it/constv2ctl2-full--lora_r4_ce/rung_subset.json"
    assert r8.tag == "llama8b_it" and IC.ifeval_paths("llama8b", "main", "none").tag == "llama8b"
    slugs = set(IC.IFEVAL_SLUGS.values())
    for r in ROWS:
        for rung in IC.IFEVAL_CAP_LADDER:
            for v in IC.ifeval_paths(r["model_key"], r["view"], r["cell"], rung):
                if isinstance(v, str):
                    assert not (set(v.split("/")) & slugs), f"a slug is a path component of {v}"
    # the record that DECIDED rung r is the previous rung's cap_ladder
    assert IC.ifeval_paths("gemma4e4b", "main", "none", IC.previous_rung(4096)).cap_ladder == p.cap_ladder.replace(
        "safety_refusal--psr_kl_all", "none")


def test_the_constructor_refuses_an_unknown_model_a_rung_off_the_ladder_and_a_malformed_cell():
    with pytest.raises(ValueError, match="model key"):
        IC.ifeval_paths("glm46vflash", "main", "none", 2048)
    with pytest.raises(ValueError, match="ladder"):
        IC.ifeval_paths("gemma4e4b", "main", "none", 16384)
    with pytest.raises(ValueError, match="view"):
        IC.ifeval_paths("gemma4e4b", "fullconst", "none", 2048)
    with pytest.raises(ValueError, match="cell"):
        IC.ifeval_paths("gemma4e4b", "main", "a/b", 2048)


# ---------------------------------------------------------------------------
SEED_ROWS = json.loads((REPO / IC.IFEVAL_MATRIX_QWEN_SEEDREP_REL).read_text(encoding="utf-8"))["rows"]


def test_the_seed_sibling_documents_paths_are_unique_alone_and_beside_the_frozen_matrix():
    """The sibling document (30 rows on the six seed lane views, TAG qwen35_4b_<view>) through the ONE constructor: every
    path kind unique over 30 rows x 3 rungs, the recorded paths ARE the constructor's strings, and the union with the frozen
    40-row document's paths is still unique (the two documents share a wave-root grammar; a TAG collision would alias a store)."""
    assert len(SEED_ROWS) == 30 and {r["view"] for r in SEED_ROWS} == set(IC.IFEVAL_LANE_VIEWS)
    out = {k: [] for k in PATH_KINDS}
    for r in SEED_ROWS:
        for rung in IC.IFEVAL_CAP_LADDER:
            p = IC.ifeval_paths(r["model_key"], r["view"], r["cell"], rung)
            assert p.tag == f"qwen35_4b_{r['view']}" == r["tag"]
            for k in PATH_KINDS:
                v = getattr(p, k)
                if v is not None:
                    out[k].append(v)
        p1 = IC.ifeval_paths(r["model_key"], r["view"], r["cell"], IC.IFEVAL_PASS1_CAP)
        assert r["out_root"] == p1.out_root and r["cell_slug"] == p1.cell_slug
        for rung in IC.IFEVAL_CAP_LADDER:
            p = IC.ifeval_paths(r["model_key"], r["view"], r["cell"], rung)
            assert r["configuration_ids"][str(rung)] == p.configuration_id and r["dispatch_paths"][str(rung)] == p.dispatch
            assert r["budget_paths"][str(rung)] == p.budget and r["cap_ladder_paths"][str(rung)] == p.cap_ladder
    for kind in ("out_root", "cap_ladder", "ladder_state", "budget", "dispatch"):
        assert len(set(out[kind])) == len(out[kind]) == 90, kind
    assert len(set(out["subset"])) == len(out["subset"]) == 60
    both = _all_paths()
    for kind in PATH_KINDS:
        merged = both[kind] + out[kind]
        assert len(set(merged)) == len(merged), f"{kind}: a seed path collides with a frozen-matrix path"
    slugs = set(IC.IFEVAL_SLUGS.values())
    for r in SEED_ROWS:
        for v in IC.ifeval_paths(r["model_key"], r["view"], r["cell"], 4096):
            if isinstance(v, str):
                assert not (set(v.split("/")) & slugs)
    # the Arditi rows' paths carry the bare Class B id as the cell slug (no `__` to fold)
    ar = IC.ifeval_paths("qwen35_4b", "s1", "arditi_allpos", 2048)
    assert ar.out_root == "controls/qwen35_4b_s1/arditi_allpos/vendor_preset" and ar.configuration_id == "qwen35_4b_s1/arditi_allpos/rung2048"

