"""The contrast's `anchor_kind` on the six capability readers (`bigcodebench_full` / `bfcl_full` / `bigcodebench_seedrep` / `bfcl_seedrep` /
`ifeval_seedrep` / `math500_seedrep`) is ONE rule, `_bcbw_common.anchor_kind_for(plan)`: a row paired against its OWN standup's `none` files
`same_standup_none`; a row whose anchor plan says `cross_standup` -- the composed `s<N>p` row's fallback against `s<N>/none`, the instruct-direct
alias against `<model key>_s1/none` -- files `sibling_none` (vocab.ANCHOR_KINDS: the same model's `none` served by another standup under the same
decode settings, a fair anchor across seeds / standups when the decode settings match). Until 2026-09-26 every contrast of these readers filed
`same_standup_none` and told the two apart on `extra.cross_standup` alone: the 15 live instruct-direct contrasts of the full readers
(bigcodebench_full 9 / bfcl_full 6) move on that field at emission 3; the four seedrep readers' fallback rows (3 each) sit in the registry from
an earlier revision -- every composed row's own `none` has since filed, so their live mirrors re-read byte for byte and they are not bumped.
What this pins, on the readers' own synthetic mirrors:

* THE RULE over the plans `anchor_plan` / `instruct_anchor_plan` return, and that both labels are in the vocabulary;
* THE FULL READERS: the instruct-direct row's alias contrast is `sibling_none` while every seed-tag and own-none contrast beside it is `same_standup_none`;
  without the paired tag's own `none` the composed row's fallback is `sibling_none` too;
* THE 200-ITEM SEEDREP READERS and IFEVAL: the composed row's fallback is `sibling_none`, its seed-tag siblings `same_standup_none`; with the paired
  tag's own `none` filed every contrast is `same_standup_none` (MATH-500's fixture lives in its own test module, which asserts the same two paths);
* ONE PLACE: the three stamp sites (`_bcbw_common._emit_contrasts`, `ifeval_seedrep`, `math500_seedrep`) all spell the field through the helper.
"""
from __future__ import annotations

import inspect
import re

import pytest

import _bcbw_full_fixtures as Z
import _bcbw_seedrep_fixtures as X
import _ifeval_seedrep_fixtures as XI
from sisteer.registry import vocab as V
from sisteer.registry.adapters import _bcbw_common as B
from sisteer.registry.adapters import bfcl_full as BFF
from sisteer.registry.adapters import bfcl_full_shared as BFSH
from sisteer.registry.adapters import bfcl_seedrep as BFS
from sisteer.registry.adapters import bigcodebench_full as BCF
from sisteer.registry.adapters import bigcodebench_full_shared as BCSH
from sisteer.registry.adapters import bigcodebench_seedrep as BCS
from sisteer.registry.adapters import ifeval_seedrep as IFE
from sisteer.registry.adapters import lineage_common as LC
from sisteer.registry.adapters import math500_seedrep as M5

FULL = {"bigcodebench": BCF.BigcodebenchFullAdapter, "bfcl": BFF.BfclFullAdapter}
SEEDREP = {"bigcodebench": BCS.BigcodebenchSeedrepAdapter, "bfcl": BFS.BfclSeedrepAdapter}
OWN, SIB = ("same_standup_none", False), ("sibling_none", True)


@pytest.fixture(scope="module")
def seeds():
    return LC.Seeds.locate()


def _kinds(bt):
    """{(tag, object key): (anchor_kind, extra.cross_standup)} over every contrast the read filed."""
    runs = {r["run_id"]: r for r in bt["run"]}
    return {(runs[c["run_id"]]["view_legacy"], runs[c["run_id"]]["campaign_cell_id_legacy"]): (c["anchor_kind"], c["extra"]["cross_standup"]) for c in bt["run_contrast"]}


def test_the_rule_reads_the_plan_and_both_labels_are_in_the_vocabulary():
    group, paired = "qwen35_4b_s1", "qwen35_4b_s1p"
    seed_none = {(group, "none"): {"run_id": "seed-none", "items": {}}}
    own = B.anchor_plan(group, group, seed_none, {}, set())
    assert own["own"] is True and own["cross_standup"] is False and B.anchor_kind_for(own) == "same_standup_none"
    fallback = B.anchor_plan(group, paired, seed_none, {}, set())
    assert fallback["own"] is False and fallback["cross_standup"] is True and fallback["anchor"] is seed_none[(group, "none")]
    assert B.anchor_kind_for(fallback) == "sibling_none", "the D65 fallback: the seed tag's none, another standup"
    pending = B.anchor_plan(group, paired, seed_none, {(paired, "none"): {}}, set())
    assert pending["own_none_state"] == B.ANCHOR_NOT_YET_RUN and B.anchor_kind_for(pending) == "sibling_none", "still the fallback while the own none is pending"
    paired_own = B.anchor_plan(group, paired, {**seed_none, (paired, "none"): {"run_id": "paired-none", "items": {}}}, {}, set())
    assert paired_own["own"] is True and B.anchor_kind_for(paired_own) == "same_standup_none", "the paired tag's own none filed: its own standup"
    alias = B.instruct_anchor_plan("gemma4e4b_s1", {("gemma4e4b_s1", "none"): {"run_id": "s1-none", "items": {}}}, {}, set())
    assert alias["cross_standup"] is True and alias["alias"] is True and B.anchor_kind_for(alias) == "sibling_none", "the instruct-direct alias: the s1 none, another standup"
    assert B.ANCHOR_KIND_OWN == "same_standup_none" and B.ANCHOR_KIND_SIBLING == "sibling_none"
    assert {B.ANCHOR_KIND_OWN, B.ANCHOR_KIND_SIBLING} <= set(V.ANCHOR_KINDS), "the labels are the registry's closed vocabulary (CHECKed on run_contrast)"


def test_the_four_stamp_sites_spell_the_field_through_the_one_helper():
    # four since 2026-09-28: the shared readers' stand-in comparison (_bcbw_common._emit_solo_contrasts) stamps the field through the same helper over a
    # `borrowed` plan (solo_anchor_plan), so the one function still decides every contrast's kind -- and spells it `solo_stand_in` since migration 0027:
    # the stand-in has its own kind, `borrowed` names the anchor currency's tier-1c re-anchoring alone
    stamp = re.compile(r'"anchor_kind":\s*([^,}]+)')
    for mod, want, n in ((B, "anchor_kind_for(plan)", 2), (IFE, "B.anchor_kind_for(plan)", 1), (M5, "B.anchor_kind_for(plan)", 1)):
        found = stamp.findall(inspect.getsource(mod))
        assert found == [want] * n, (mod.__name__, found)
    for mod in (BCF, BFF, BCS, BFS, BCSH, BFSH):
        assert not stamp.findall(inspect.getsource(mod)), f"{mod.__name__} stamps no contrast of its own: the shared _emit_contrasts does"
    assert B.anchor_kind_for(B.solo_anchor_plan("r")) == "solo_stand_in" == B.ANCHOR_KIND_SOLO_STAND_IN and B.anchor_kind_for({"cross_standup": True}) == "sibling_none" and B.anchor_kind_for({"cross_standup": False}) == "same_standup_none"
    assert B.anchor_kind_for({"cross_standup": True, "borrowed_reference": True}) == "borrowed" == B.ANCHOR_KIND_BORROWED


@pytest.mark.parametrize("eval_id", sorted(FULL))
def test_the_full_readers_file_the_instruct_alias_as_sibling_none_and_every_own_none_contrast_as_same_standup_none(tmp_path, seeds, eval_id):
    fx = Z.mirror(tmp_path, seeds, eval_id=eval_id, instruct=True, root=str(tmp_path / f"{eval_id}_own"))  # the paired tag's own none files (the default)
    bt, facts = Z.read(Z.adapter(FULL[eval_id], fx, seeds))
    kinds = _kinds(bt)
    assert facts["n_refusals"] == 0 and (Z.IT_TAG, Z.IT_KEY) in kinds and (Z.TAG_P, Z.COMPOSED) in kinds and len(kinds) >= 3, (facts["refusals"], sorted(kinds))
    assert kinds[(Z.IT_TAG, Z.IT_KEY)] == SIB, "the `_it` row's anchor is the s1 none BY ALIAS: another standup of the same model"
    assert {v for k, v in kinds.items() if k != (Z.IT_TAG, Z.IT_KEY)} == {OWN}, "every seed-tag row and the composed row (its own none filed) pair within their standup"
    # without the paired tag's own `none` the composed row falls back ACROSS tags too: two sibling_none contrasts, the rest own
    fx = Z.mirror(tmp_path, seeds, eval_id=eval_id, instruct=True, paired_none=False, root=str(tmp_path / f"{eval_id}_fallback"))
    bt, facts = Z.read(Z.adapter(FULL[eval_id], fx, seeds))
    kinds = _kinds(bt)
    assert {k for k, v in kinds.items() if v == SIB} == {(Z.IT_TAG, Z.IT_KEY), (Z.TAG_P, Z.COMPOSED)}, sorted(kinds.items())
    assert {v for k, v in kinds.items() if v != SIB} == {OWN}


@pytest.mark.parametrize("eval_id", sorted(SEEDREP))
def test_the_200_item_seedrep_readers_file_the_d65_fallback_as_sibling_none_and_the_own_none_pairing_as_same_standup_none(tmp_path, seeds, eval_id):
    fx = X.mirror(tmp_path, seeds, eval_id=eval_id, root=str(tmp_path / f"{eval_id}_fallback"))  # no paired none in the launch set: the standing fallback
    bt, facts = X.read(X.adapter(SEEDREP[eval_id], fx, seeds))
    kinds = _kinds(bt)
    assert facts["n_refusals"] == 0 and kinds[(X.TAG_P, X.COMPOSED)] == SIB, (facts["refusals"], sorted(kinds.items()))
    assert {k[0] for k in kinds} == {X.TAG, X.TAG_P} and {v for k, v in kinds.items() if k != (X.TAG_P, X.COMPOSED)} == {OWN}
    fx = X.mirror(tmp_path, seeds, eval_id=eval_id, paired_none=True, root=str(tmp_path / f"{eval_id}_own"))  # the paired tag's own none filed
    bt, facts = X.read(X.adapter(SEEDREP[eval_id], fx, seeds))
    kinds = _kinds(bt)
    assert facts["n_refusals"] == 0 and (X.TAG_P, X.COMPOSED) in kinds and set(kinds.values()) == {OWN}, sorted(kinds.items())


def test_ifeval_seedrep_files_the_d65_fallback_as_sibling_none_and_the_own_none_pairing_as_same_standup_none(tmp_path, seeds):
    fx = XI.mirror(tmp_path, seeds, root=str(tmp_path / "ifeval_fallback"))
    bt, facts = XI.read(XI.adapter(fx, seeds))
    kinds = _kinds(bt)
    assert facts["n_refusals"] == 0 and kinds[(XI.TAG_P, XI.COMPOSED)] == SIB, (facts["refusals"], sorted(kinds.items()))
    assert {v for k, v in kinds.items() if k != (XI.TAG_P, XI.COMPOSED)} == {OWN}
    fx = XI.mirror(tmp_path, seeds, root=str(tmp_path / "ifeval_own"), paired_none=True)
    bt, facts = XI.read(XI.adapter(fx, seeds))
    kinds = _kinds(bt)
    assert facts["n_refusals"] == 0 and (XI.TAG_P, XI.COMPOSED) in kinds and set(kinds.values()) == {OWN}, sorted(kinds.items())
