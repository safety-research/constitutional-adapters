"""The plan's keying invariant is refused BY NAME, never left to a bare KeyError in the reading loop.

``runner.plan_groups`` decides what a pass reads before any store is opened, and ``_run_adapter`` then indexes ``plan.groups`` by every key
of ``plan.changed``: a key the plan does not carry is a planner defect, and until the prior-only group spelling was fixed
(``test_registry_prior_group_spelling``) it surfaced as ``KeyError: '<generation path>'`` out of the reading loop, with nothing naming the
adapter or the rule that produced the key. ``runner.check_plan_invariants`` states the invariant once -- every key of ``changed`` and of
``gone_in_group`` is a key of ``groups``; no key of ``vanished`` is -- and raises ``runner.PlanInvariantError`` naming the adapter, the
offending keys and the plan's keys; ``plan_groups`` calls it last, so a defect in any planning rule is named at the plan, before a read.
"""
from __future__ import annotations

import os

import pytest

from sisteer.registry.adapters import base as AB
from sisteer.registry.adapters import runner as RUN


def _plan(groups, changed=(), gone=None, vanished=None) -> RUN.GroupPlan:
    plan = RUN.GroupPlan()
    for g in groups:
        plan.groups[g] = [AB.Input(uri=f"{g}/store.json", path=f"{g}/store.json", sha256="a" * 64, size=1, mtime_utc="2026-01-01T00:00:00.000Z")]
    plan.changed = set(changed)
    plan.gone_in_group = {k: set(v) for k, v in (gone or {}).items()}
    plan.vanished = {k: set(v) for k, v in (vanished or {}).items()}
    return plan


class _Adapter:
    name = "stub_adapter"
    roots = ("/stub/root",)


def test_a_consistent_plan_passes_and_each_violation_is_refused_by_name_with_the_offending_keys():
    ad = _Adapter()
    RUN.check_plan_invariants(ad, _plan(["/stub/root/g1", "/stub/root/g2"], changed=["/stub/root/g1"], gone={"/stub/root/g1": ["/stub/root/g1/manifest.json"]},
                                        vanished={"/stub/root/g3": ["/stub/root/g3/store.json"]}))
    RUN.check_plan_invariants(ad, RUN.GroupPlan())                                     # the empty plan
    # a changed key the plan does not carry (the shape of the generation-spelled partial drop before the fix)
    ghost = "/shared/gen/component/0123abcd/g1"
    with pytest.raises(RUN.PlanInvariantError) as exc:
        RUN.check_plan_invariants(ad, _plan(["/stub/root/g1"], changed=[ghost]))
    msg = str(exc.value)
    assert msg.startswith("stub_adapter: ") and ghost in msg and "changed" in msg and "/stub/root/g1" in msg, msg
    assert isinstance(exc.value, RuntimeError)
    # a gone_in_group key the plan does not carry
    with pytest.raises(RUN.PlanInvariantError, match="gone_in_group") as exc:
        RUN.check_plan_invariants(ad, _plan(["/stub/root/g1"], changed=["/stub/root/g1"], gone={ghost: ["/stub/root/g1/manifest.json"]}))
    assert ghost in str(exc.value)
    # a vanished key the plan DOES carry (a group both present and retired)
    with pytest.raises(RUN.PlanInvariantError, match="vanished") as exc:
        RUN.check_plan_invariants(ad, _plan(["/stub/root/g1"], vanished={"/stub/root/g1": ["/stub/root/g1/store.json"]}))
    assert "/stub/root/g1" in str(exc.value)
    # several offenders: every one named, sorted
    with pytest.raises(RUN.PlanInvariantError) as exc:
        RUN.check_plan_invariants(ad, _plan(["/stub/root/g1"], changed=["zz", "aa"]))
    assert "aa, zz" in str(exc.value)


def test_plan_groups_refuses_a_planning_rule_that_spells_a_vanished_group_as_a_present_one(tmp_path, monkeypatch):
    """Through a stub: ``prior_group_spelling`` (the vanished branch's spelling rule) made to answer with a PRESENT group's key -- the kind of
    defect the old ``imap.local`` rule was -- is refused by name at the end of ``plan_groups``, the message naming the adapter and the key,
    instead of a vanished group shadowing a present one."""
    root = tmp_path / "stores"
    for d in ("g1", "g2"):
        (root / d).mkdir(parents=True)
        (root / d / "store.json").write_text("{}\n")
    ad = _Adapter()
    ad.roots = (str(root),)
    ad.comparison_group = lambda inp: os.path.dirname(inp.uri)
    inputs = [AB.check_input(str(root / d / "store.json"), role="report") for d in ("g1", "g2")]
    present = {i.identity_uri: RUN.Watermark(RUN.source_id_for(ad.name, os.path.dirname(i.uri)), AB.watermark_of(i), None) for i in inputs}
    gone_uri = str(root / "g3" / "store.json")
    marks = dict(present, **{gone_uri: RUN.Watermark(RUN.source_id_for(ad.name, str(root / "g3")), ("b" * 64, 1, "2026-01-01T00:00:00.000Z"), None)})
    # honest: g3 vanished, keyed by its own spelling, disjoint from the present groups
    plan = RUN.plan_groups(ad, inputs, marks, "1", None)
    assert plan.vanished == {str(root / "g3"): {gone_uri}} and plan.changed == set()
    # the stub: the spelling rule answers with g1's key for every prior-only group
    monkeypatch.setattr(RUN, "prior_group_spelling", lambda adapter, identity, imap=None: str(root / "g1"))
    with pytest.raises(RUN.PlanInvariantError) as exc:
        RUN.plan_groups(ad, inputs, marks, "1", None)
    msg = str(exc.value)
    assert msg.startswith(f"{ad.name}: ") and "vanished" in msg and str(root / "g1") in msg, msg
