"""`registry stores coverage`: every registered adapter's DECLARED roots against the
components of ``configs/registry/cluster_reads.yaml``. The adapters are CONSTRUCTED and their ``roots`` read -- ``discover()`` is never
called and no store is opened or listed (an audit hook and fakes whose ``discover`` raises prove it). Synthetic cases pin every root
state (covered / restricted / retired / volatile / refused / UNCOVERED / undeclared), the undeclared-adapter and missing-component
failures and the YAML stubs; the LIVE file is pinned at zero UNCOVERED, every registered adapter declared, no entry missing a holder,
the six heavy adapters at their measured shards.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pytest
import yaml

from sisteer.registry import adapters as ADP
from sisteer.registry import stores as ST
from sisteer import site as SITE

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
YAML_LIVE = os.path.join(REPO, "configs", "registry", "cluster_reads.yaml")


# --------------------------------------------------------------------------------------------
# a synthetic world: a home with three component bases, fakes that refuse to discover
# --------------------------------------------------------------------------------------------


class _Fake:
    """An adapter whose declared ``roots`` are the whole story: ``discover`` raises, so any call to it fails the test."""

    covers_sources = ()
    rng_scope = "none"
    allow_symlinks = False

    def __init__(self, name, roots, retired=None):
        self.name = name
        self.roots = tuple(roots)
        self._retired = retired

    def discover(self):
        raise AssertionError(f"{self.name}: coverage must never discover")

    def comparison_group(self, inp):
        return "g"

    def read(self, ctx, group, inputs):
        raise AssertionError("never read")

    def source_states(self, ctx, inputs):
        return []

    @property
    def inputs_retired(self):
        return self._retired


def _registry(*fakes):
    return {f.name: (lambda f=f: f) for f in fakes}


def _write_cfg(tmp_path, components, adapters, name="cfg.yaml", schema=None):
    doc = {"schema": schema or ST.CONFIG_SCHEMA, "written_utc": "2026-10-04T00:00:00Z",
           "destinations": {"local": {"host": None, "shared": str(tmp_path / "shared")}},
           "components": components, "adapters": adapters}
    p = tmp_path / name
    p.write_text(yaml.safe_dump(doc, sort_keys=False))
    return str(p)


def _world(tmp_path):
    home = tmp_path / "home"
    for d in ("whole", "parts/a", "parts/b/x", "excl/skip", "excl/keep"):
        (home / d).mkdir(parents=True, exist_ok=True)
    comps = {"whole": {"base": str(home / "whole"), "authority": "grading_host_copy", "note": "the whole base"},
             "parts": {"base": str(home / "parts"), "members": ["a", "b/*"], "authority": "grading_host_copy", "note": "two members"},
             "excl": {"base": str(home / "excl"), "exclude": ["/skip", "junk"], "authority": "grading_host_copy", "note": "an excluded subtree"}}
    return home, comps


# --------------------------------------------------------------------------------------------
# the path algebra
# --------------------------------------------------------------------------------------------


def test_member_match_is_path_algebra_over_segments():
    assert ST.member_match("anything/below", ".") == ST.COVERED
    assert ST.member_match("", ".") == ST.COVERED
    assert ST.member_match("a/deep/er", "a") == ST.COVERED                 # the root lies below the member
    assert ST.member_match("a", "a") == ST.COVERED                         # at the member
    assert ST.member_match("", "a") == ST.RESTRICTED                       # the member lies below the root (the base itself)
    assert ST.member_match("b", "b/*") == ST.RESTRICTED
    assert ST.member_match("b/x/y", "b/*") == ST.COVERED
    assert ST.member_match("c", "a") is None                               # diverging
    assert ST.member_match("grades_x/sub", "grades_*") == ST.COVERED       # a glob segment
    assert ST.member_match("transcripts_x", "grades_*") is None
    # a glob-SPELLED root (bcb_bfcl's `bcb-bfcl-200*`) against a glob member below it: restricted, never a divergence
    assert ST.member_match("bcb-bfcl-200*", "bcb-bfcl-200*/analysis") == ST.RESTRICTED


def test_root_coverage_respects_base_members_and_excludes(tmp_path):
    home, comps = _world(tmp_path)
    cfg = ST.load_config(_write_cfg(tmp_path, comps, {"x": {"shards": 1, "components": ["whole"]}}))
    whole, parts, excl = cfg.components["whole"], cfg.components["parts"], cfg.components["excl"]
    assert ST.root_coverage(str(home / "whole"), whole) == (ST.COVERED, ".")
    assert ST.root_coverage(str(home / "whole" / "x" / "y"), whole) == (ST.COVERED, ".")
    assert ST.root_coverage(str(home / "wholesome"), whole) is None                    # a prefix of the name is not the base
    assert ST.root_coverage(str(home / "parts"), parts) == (ST.RESTRICTED, "a")
    assert ST.root_coverage(str(home / "parts" / "a" / "deep"), parts) == (ST.COVERED, "a")
    assert ST.root_coverage(str(home / "parts" / "b"), parts) == (ST.RESTRICTED, "b/*")
    assert ST.root_coverage(str(home / "parts" / "b" / "x"), parts) == (ST.COVERED, "b/*")
    assert ST.root_coverage(str(home / "parts" / "zz"), parts) is None
    assert ST.root_coverage(str(home / "excl" / "keep"), excl) == (ST.COVERED, ".")
    assert ST.root_coverage(str(home / "excl" / "skip"), excl) is None                 # anchored exclude
    assert ST.root_coverage(str(home / "excl" / "skip" / "deeper"), excl) is None
    assert ST.root_coverage(str(home / "excl" / "keep" / "junk" / "f"), excl) is None  # name exclude at any depth


# --------------------------------------------------------------------------------------------
# the report: every state, the failures, no discovery, no store access
# --------------------------------------------------------------------------------------------


def test_coverage_classifies_every_root_state_and_never_discovers_or_lists(tmp_path):
    home, comps = _world(tmp_path)
    adapters = {"good": {"shards": 1, "components": ["whole", "parts"]},
                "diverge": {"shards": 1, "components": ["parts"]},
                "excluded": {"shards": 1, "components": ["excl"]},
                "lost": {"shards": 1, "components": ["whole"]},
                "outside": {"shards": 2, "components": ["whole"],
                            "volatile_roots": [{"root": "/tmp/c2-volatile-peek", "reason": "a volume peek no node has"}]},
                "tired": {"shards": 1, "components": ["whole"]},
                "broken": {"shards": 1, "components": ["whole"]},
                "nothold": {"shards": 1, "components": ["excl"]},
                "refused": {"cluster": "refused", "reason": "its pinned worktree does not exist (D13)"}}
    cfg = ST.load_config(_write_cfg(tmp_path, comps, adapters))
    reg = _registry(
        _Fake("good", [str(home / "whole" / "x"), str(home / "parts"), str(home / "parts" / "a" / "deep")]),     # covered, restricted, covered
        _Fake("diverge", [str(home / "parts" / "zz")]),                                                         # UNCOVERED (diverges from every member)
        _Fake("excluded", [str(home / "excl" / "skip")]),                                                       # UNCOVERED (excluded by the component)
        _Fake("lost", [str(home / "nowhere")]),                                                                 # UNCOVERED (no base holds it)
        _Fake("outside", [str(home / "whole"), "/tmp/c2-volatile-peek", "/tmp/c2-other"]),                     # covered, volatile, UNCOVERED (outside, undeclared)
        _Fake("tired", [str(home / "whole"), "/tmp/c2-old/transcripts"], retired=["/tmp/c2-old/transcripts/*"]),   # covered, retired
        _Fake("broken", [str(home / "whole"), None]),                                                           # covered, undeclared (None)
        _Fake("nothold", [str(home / "whole" / "q")]),                                                          # covered by `whole`, which the entry does not name
        _Fake("refused", [str(home / "whole"), "/tmp/anything"]),                                               # refused, refused
        _Fake("unlisted", [str(home / "whole")]),                                                               # not declared in the configuration
    )
    events = []

    def hook(event, args):
        if event in ("open", "os.listdir", "os.scandir", "glob.glob") and args and str(args[0]).startswith(str(home)):
            events.append((event, str(args[0])))
    sys.addaudithook(hook)
    rep = ST.coverage(cfg, registry=reg, home=str(home))
    assert events == [], events                                         # nothing under the home was opened or listed
    assert not rep.ok
    sm = rep.summary
    assert sm["adapters"] == 10 and sm["declared_adapters"] == 9 and sm["refused_adapters"] == 1 and sm["roots"] == 17
    assert (sm["covered"], sm["restricted"], sm["retired"], sm["volatile"], sm["refused"], sm["UNCOVERED"], sm["undeclared"]) == (7, 1, 1, 1, 2, 4, 1)
    assert sm["undeclared_adapters"] == ["unlisted"] and sm["failed_construction"] == {}
    assert sm["missing_components"] == {"nothold": ["whole"]}
    by = {(r.adapter, r.root): r for r in rep.roots}
    assert by[("good", str(home / "parts"))].state == ST.RESTRICTED and by[("good", str(home / "parts"))].components == ["parts"]
    assert by[("good", str(home / "parts" / "a" / "deep"))].detail == "parts: member a"
    assert by[("diverge", str(home / "parts" / "zz"))].state == ST.UNCOVERED
    assert by[("excluded", str(home / "excl" / "skip"))].state == ST.UNCOVERED
    assert by[("lost", str(home / "nowhere"))].state == ST.UNCOVERED
    assert by[("outside", "/tmp/c2-volatile-peek")].state == ST.VOLATILE and "a volume peek" in by[("outside", "/tmp/c2-volatile-peek")].detail
    assert by[("outside", "/tmp/c2-other")].state == ST.UNCOVERED and "outside the home" in by[("outside", "/tmp/c2-other")].detail
    assert by[("tired", "/tmp/c2-old/transcripts")].state == ST.RETIRED and "inputs_retired" in by[("tired", "/tmp/c2-old/transcripts")].detail
    assert by[("broken", None)].state == ST.UNDECLARED
    assert by[("refused", "/tmp/anything")].state == ST.REFUSED_ROOT and "D13" in by[("refused", "/tmp/anything")].detail
    assert rep.adapters["refused"]["refused"] and rep.adapters["refused"]["shards"] == 0 and rep.adapters["refused"]["components"] == []
    assert rep.adapters["good"]["extra_components"] == [] and rep.adapters["outside"]["shards"] == 2
    assert rep.adapters["unlisted"]["declared"] is False
    text = ST.format_coverage(rep)
    assert "coverage NOT OK" in text and "unlisted: NOT DECLARED" in text and "MISSING components" in text and "-> UNCOVERED" in text
    assert "refused: cluster: refused -- its pinned worktree" in text
    # the JSON form round-trips the same facts
    d = json.loads(json.dumps(rep.as_dict(), default=str))
    assert d["schema"] == ST.COVERAGE_SCHEMA and d["home"] == str(home) and d["ok"] is False and d["summary"]["UNCOVERED"] == 4 and len(d["roots"]) == 17
    assert f"  ~/parts -> restricted [parts]" in text                      # roots are spelled against the report's home


def test_coverage_is_ok_when_every_root_is_held_and_every_adapter_declared(tmp_path):
    home, comps = _world(tmp_path)
    cfg = ST.load_config(_write_cfg(tmp_path, comps, {"a": {"shards": 1, "components": ["whole", "parts"]}, "b": {"shards": 1, "components": ["excl"]}}))
    reg = _registry(_Fake("a", [str(home / "whole"), str(home / "parts" / "b" / "x")]), _Fake("b", [str(home / "excl" / "keep")]))
    rep = ST.coverage(cfg, registry=reg, home=str(home))
    assert rep.ok and rep.summary["UNCOVERED"] == 0 and rep.summary["missing_components"] == {} and rep.summary["undeclared_adapters"] == []
    assert "coverage OK" in ST.format_coverage(rep)
    assert ST.coverage_yaml_fragment(rep) == ""
    # `only` narrows to the named adapters; an unknown name refuses
    assert sorted(ST.coverage(cfg, registry=reg, home=str(home), only=["b"]).adapters) == ["b"]
    with pytest.raises(ST.StoresError, match="not registered: zz"):
        ST.coverage(cfg, registry=reg, home=str(home), only=["zz"])


def test_coverage_yaml_fragment_proposes_stubs_for_what_is_wanting(tmp_path):
    home, comps = _world(tmp_path)
    cfg = ST.load_config(_write_cfg(tmp_path, comps, {"a": {"shards": 1, "components": ["whole"]}}))
    reg = _registry(_Fake("a", [str(home / "whole")]),
                    _Fake("newcomer", [str(home / "parts" / "a"), str(home / "fresh-root")]))     # undeclared; one covered root, one UNCOVERED
    rep = ST.coverage(cfg, registry=reg, home=str(home))
    frag = ST.coverage_yaml_fragment(rep)
    assert "adapters:\n  newcomer:\n    shards: 1" in frag and "components: [parts]" in frag
    assert "# UNCOVERED root: ~/fresh-root" in frag                      # spelled against the report's home
    assert "components:\n  fresh_root:\n    base: ~/fresh-root" in frag and "UNCOVERED root of newcomer" in frag
    assert yaml.safe_load(frag.replace("# UNCOVERED", "#"))["adapters"]["newcomer"]["components"] == ["parts"]


def test_cli_coverage_exits_one_on_a_gap_zero_when_clean_and_two_on_a_refusal(tmp_path, monkeypatch, capsys):
    home, comps = _world(tmp_path)
    gap = _write_cfg(tmp_path, comps, {"a": {"shards": 1, "components": ["whole"]}}, name="gap.yaml")
    clean = _write_cfg(tmp_path, comps, {"a": {"shards": 1, "components": ["whole"]}, "b": {"shards": 1, "components": ["parts"]}}, name="clean.yaml")
    reg = _registry(_Fake("a", [str(home / "whole")]), _Fake("b", [str(home / "parts" / "a")]))
    monkeypatch.setattr(ADP, "load_all", lambda names=None: reg)
    monkeypatch.setattr(ST, "HOME", str(home))

    def run(*argv):
        p = argparse.ArgumentParser()
        ST.add_arguments(p)
        return ST.cmd_stores(p.parse_args(list(argv)), str(tmp_path / "root"))

    assert run("--config", gap, "coverage", "--json") == 1
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is False and out["summary"]["undeclared_adapters"] == ["b"]
    assert run("--config", clean, "coverage") == 0
    assert "coverage OK" in capsys.readouterr().out
    assert run("--config", clean, "coverage", "--adapter", "nope") == 2
    assert "not registered: nope" in capsys.readouterr().err
    assert run("--config", gap, "coverage", "--emit-yaml") == 1
    assert "  b:\n    shards: 1" in capsys.readouterr().out


# --------------------------------------------------------------------------------------------
# the LIVE configuration (zero UNCOVERED)
# --------------------------------------------------------------------------------------------


def test_live_config_covers_every_registered_adapters_declared_roots():
    cfg = ST.load_config(YAML_LIVE)
    rep = ST.coverage(cfg)
    sm = rep.summary
    assert rep.ok, ST.format_coverage(rep)
    assert sm["UNCOVERED"] == 0 and sm["undeclared"] == 0 and sm["undeclared_adapters"] == [] and sm["failed_construction"] == {}
    assert sm["missing_components"] == {}
    registered = sorted(ADP.load_all())
    assert sorted(cfg.adapters) == registered, "every registered adapter has an entry and no entry names an unregistered adapter"
    # the W&B resolver declares its offline mirror only where the registry configuration names one ([wandb] mirror), so a host without
    # that table (a test node) counts one root fewer than the grading host
    assert sm["adapters"] == len(registered) >= 55 and sm["roots"] >= 99
    # the states that depend on this host (a /tmp working tree the adapter retires exists on the grading host only) are not pinned; the two
    # declared volatile roots are, and so is the restriction of the two readers whose declared root is the whole run-state tree
    for name in ("seedrep_controls", "seedrep_statics"):
        assert [v.root for v in cfg.adapters[name].volatile_roots] == ["/tmp/launch_vol_peek"]
    for name in ("multiturn_grades", "cap4096_lane", "regrade_twin"):
        states = {r.state for r in rep.roots if r.adapter == name}
        assert ST.RESTRICTED in states and ST.UNCOVERED not in states, (name, states)
    # the relu_paired fix: its Petri roots are declared (resolved), no None root remains anywhere
    rp = [r for r in rep.roots if r.adapter == "relu_paired"]
    assert len(rp) == 4 and all(r.state == ST.COVERED for r in rp) and {c for r in rp for c in r.components} == {
        "relu_paired_run_state", "relu_paired_manifests", "petri_native_ledger", "petri_drive"}


def test_live_config_is_schema_two_with_valid_entries_and_the_measured_shards():
    cfg = ST.load_config(YAML_LIVE)
    doc = yaml.safe_load(open(YAML_LIVE, encoding="utf-8"))
    assert doc["schema"] == "cluster_reads/2" == ST.CONFIG_SCHEMA
    six = {"relu_paired": 8, "multiturn_grades": 8, "bfcl_full": 6, "bigcodebench_full": 6, "em_misalign": 6, "sampled_static": 1}
    assert {a: cfg.adapters[a].shards for a in six} == six
    for name, e in cfg.adapters.items():
        if e.refused:
            assert e.reason and e.shards == 0 and e.components == [], name
        else:
            assert e.shards >= 1 and e.components, name
            assert all(c in cfg.components for c in e.components), name
        for v in e.volatile_roots:
            assert v.reason and not v.root.startswith(ST.HOME), (name, v)
    for slug, c in cfg.components.items():
        assert c.authority in ST.AUTHORITIES and c.note and c.home_rel is not None, slug
        assert c.adapters, f"{slug} is read by no declared adapter"
        if c.authority == "cluster_a":
            assert c.producer and c.grading_host_mirror, slug
        else:
            assert not c.transitional_carry and c.carry_deadline is None and c.grading_host_mirror is None, slug
    # no component has yet flipped to the moved authority: the flip follows the verified move, not this commit
    assert {c.authority for c in cfg.components.values()} == {"grading_host_copy"}


def test_live_config_reads_every_reader_traversal_adapter_as_one_shard():
    """The RNG rule outranks the size rule: an adapter declaring ``rng_scope = "reader_traversal"`` -- one shared RNG over its
    whole traversal -- is admitted by the shard runner at K = 1 only (``run_pass`` raises on a larger K), and the first post-cutover read
    lost alphasteer_rebake (8), arditi_strength (4) and ctl_neg1_static (4) to that guard before any summary was written. Every registered
    adapter is CONSTRUCTED (never discovered, as `stores coverage` does) and its declared scope read; the live file gives each traversal
    adapter exactly one shard, and the set is the one the header names."""
    cfg = ST.load_config(YAML_LIVE)
    traversal = {}
    for name, factory in sorted(ADP.load_all().items()):
        adapter = factory()
        if getattr(adapter, "rng_scope", "none") == "reader_traversal":
            traversal[name] = cfg.adapters[name].shards
    assert traversal == {"alphasteer_rebake": 1, "arditi_allpos": 1, "arditi_strength": 1, "ctl_neg1_static": 1, "overrefusal_items": 1,
                         "sampled_controls": 1, "sampled_static": 1}, traversal
    assert len(traversal) == 7, "the count of registered reader_traversal adapters (D165: exactly 7; constructed, 2026-10-07) -- a new one joins this list"
    header = open(YAML_LIVE, encoding="utf-8").read().split("schema:")[0]
    for name in traversal:
        assert name in header, f"the header's list of traversal adapters omits {name}"
    # the plan-time check on the constructed instance (cluster_reads.plan_time_checks) accepts every live traversal entry as it
    # stands -- the RNG rule and the entry's root coverage -- and would refuse the very shape the first post-cutover read died on
    from sisteer.registry import cluster_reads as CR
    reg = ADP.load_all()
    for name in traversal:
        checks = CR.plan_time_checks(name, cfg.adapters[name], cfg, reg, ST.HOME)
        assert checks["refusal"] is None and checks["rng_scope"] == "reader_traversal" and checks["shards"] == 1, (name, checks)
    import dataclasses
    wrong = dataclasses.replace(cfg.adapters["alphasteer_rebake"], shards=8)
    assert "exactly ONE shard" in CR.plan_time_checks("alphasteer_rebake", wrong, cfg, reg, ST.HOME)["refusal"]


def test_live_grade_regrade_holds_exactly_what_the_twins_discovery_opens_and_the_twin_names_every_sibling_holder():
    """`RegradeTwinAdapter.discover()` refuses without `<root>/README.md`, and the first
    `grade_regrade` declaration (members manifests/ + journal/) did not hold it -- `stores coverage` reports the root as restricted
    because it checks declared ROOTS, never what discovery requires. So the twin's requirements are pinned here by the same path algebra
    the coverage uses: every entry of `regrade_twin.MANIFEST_FILES` (README.md and ledger.jsonl at the root, the eight manifests/ files)
    and the harvest journal are COVERED by the live component's members; what the twin never opens (the link farms, the claims database,
    the parts and pinned copies, prose) is held by no member -- the declaration is exact, not a superset. And the Swap-A sibling verdict
    files discovery finds BESIDE the Haiku originals in other stores: the holders of their directories (the harvest-journal survey of
    2026-10-04, one representative path per holder) are all among the adapter's components, the three directories no other component
    held belong to `full22_regrade_siblings` alone, and that component's members are disjoint from every other full22 member."""
    from sisteer.registry.adapters import regrade_twin as RT
    cfg = ST.load_config(YAML_LIVE)
    comp = cfg.components["grade_regrade"]
    assert comp.adapters == ["regrade_twin"] and comp.exclude == []
    for rel, _role in RT.MANIFEST_FILES:
        assert ST.root_coverage(os.path.join(comp.base, rel), comp) == (ST.COVERED, rel), rel
    assert ST.root_coverage(os.path.join(comp.base, "journal", "harvest", "livebatch_0.jsonl"), comp) == (ST.COVERED, "journal/harvest")
    assert ST.root_coverage(os.path.join(comp.base, "journal", "harvest"), comp) == (ST.COVERED, "journal/harvest")
    for decoy in ("swapA/x/escape", "swapB/judge_cache.sqlite", "journal/claims_A.sqlite", "journal/claims_B.sqlite-wal", "journal/live/x.jsonl",
                  "manifests/_parts_A/part0.parquet", "manifests/_gen2_399158a4/x", "manifests/S_A.pin-a9f40b6d.parquet", "manifests/GENERATION.json",
                  "decisions.md", "canonical_frames/x.json", "canonical_export_snapshot_2/link"):
        assert ST.root_coverage(os.path.join(comp.base, decoy), comp) is None, decoy
    assert ST.root_coverage(comp.base, comp) == (ST.RESTRICTED, "README.md")           # the declared root lies above the members
    twin = cfg.adapters["regrade_twin"]
    assert twin.shards == 8 and twin.components[0] == "grade_regrade"
    assert set(twin.components) == {"grade_regrade", "full22_regrade_siblings", "full22_relu_native", "grade_alphasteer", "grade_arditi",
                                    "multiturn_run_state", "sampled"}
    full22 = os.path.join(ST.HOME, ".local", "state", "romp", "candidate-si-refit", "run-state", "full22")
    grade = str(SITE.expand_path("$GRADE"))

    def holders(path):
        return sorted(slug for slug, c in cfg.components.items() if ST.root_coverage(path, c) is not None)
    # one representative sibling directory per holder the survey named (162 of the 165 siblings) -> exactly that holder, among the twin's
    for d, slug in ((os.path.join(full22, "grades_mt4k_ch_qwen35_4b_it", "qwen35_4b_it"), "multiturn_run_state"),
                    (os.path.join(full22, "sampled", "grades_sampled_bu_qwen35_4b", "qwen35_4b_it"), "sampled"),
                    (os.path.join(full22, "relu_native", "grades_relu_native_ch_qwen35_4b", "qwen35_4b_it"), "full22_relu_native"),
                    (os.path.join(grade, "alphasteer_f22_defense", "grades", "x"), "grade_alphasteer"),
                    (os.path.join(grade, "arditi_strength", "grades", "x"), "grade_arditi")):
        assert holders(os.path.join(d, RT.SWAP_A_SIBLING)) == [slug] and slug in twin.components, d
    # the three directories no other component held: the new component alone, and its members disjoint from every other full22 member.
    # Its readers now: the twin, multiturn_grades (grades/<model>, the namespace directory transfer_stats_full22_all.json
    # declares) and arditi_allpos (the whole mt4kap multi-turn run, so the third member is the arditi_allpos_mt4kap directory itself)
    sib = cfg.components["full22_regrade_siblings"]
    assert sib.base == full22 and set(sib.adapters) == {"regrade_twin", "multiturn_grades", "arditi_allpos"} and len(sib.members) == 3
    assert sib.members[2] == "arditi_allpos_mt4kap"
    assert ST.root_coverage(os.path.join(full22, "arditi_allpos_mt4kap", "goalsets", "MANIFEST.json"), sib) == (ST.COVERED, "arditi_allpos_mt4kap")
    assert ST.root_coverage(os.path.join(full22, "arditi_allpos", "controls", "x"), sib) is None          # full22_mmlu_full's, not this one
    for rel in sib.members:
        assert holders(os.path.join(full22, rel, RT.SWAP_A_SIBLING)) == ["full22_regrade_siblings"], rel
        assert holders(os.path.join(full22, rel, RT.SWAP_A_ORIGINAL)) == ["full22_regrade_siblings"], rel
    for slug, c in cfg.components.items():
        if slug != "full22_regrade_siblings" and c.base == full22:
            for m in c.members:
                top = m.split("/")[0]
                assert all(ST.member_match(top, sm.split("/")[0]) is None for sm in sib.members), (slug, m)
    # the Swap-B sibling name (judge_items.claude-opus-5.jsonl) has no holder to name: the survey found none written anywhere
    assert RT.SWAP_B_SIBLING == "judge_items.claude-opus-5.jsonl"
