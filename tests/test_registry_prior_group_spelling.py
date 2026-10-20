"""A prior-only comparison group is spelled the way the adapter ADDRESSES it, under both layouts the identity map serves.

A watermark names its group canonically (through its source id). ``runner.plan_groups`` spelled a prior-only group -- one whose every
input left discovery (vanished), or one that lost SOME inputs (a partial drop) -- back through ``IdentityMap.local``, the inverse of the
pass's map. That is right under a RE-POINTED ROOT (the Phase 3 identity tests' shape: an adapter constructed over a local copy that is the
map's local side) and wrong under the HOME TREE a cluster shard lays out (``cluster_reads`` "THE LAYOUT ON THE NODE"): there
the map carries each pinned GENERATION onto its canonical home-relative base, the adapter addresses THROUGH the canonical spelling (a
symlink onto the generation), ``Input.uri`` is the canonical spelling and ``plan.groups`` is keyed by it -- so ``imap.local`` turned the
group into the generation path, a spelling no root of the adapter and no ``inputs_retired`` declaration (hung off ``$HOME`` at import)
knows.

Two defects followed, both pinned here over a synthetic world (a home with a path-grouped two-file store adapter, a generation the
canonical base links onto, a reader-state sidecar, the shard deal):
1. a VANISHED source reached ``retire_vanished_group`` with the generation-spelled group, the adapter's declaration
   matched nothing, and the shard refused the retirement ``Missing: `` although every input was declared retired;
2. (latent) a PARTIAL DROP added the generation-spelled group to ``plan.changed`` / ``plan.gone_in_group`` while ``plan.groups`` carried the
   canonical key: the reading loop's ``plan.groups[g]`` raised KeyError and ended the pass (``rng_scope = none``), or -- on a
   ``reader_traversal`` adapter, whose plan re-keys ``changed`` to every present group -- lost the drop silently (``gone_in_group`` keyed by
   a spelling nobody looked up, the gone input's watermark never retired, the drop re-found on every pass).

The rule (``runner.prior_group_spelling``, applied by ``plan_groups``): a PRESENT source takes the key ``plan.groups`` already carries
(exact, by canonical source id); a VANISHED one is spelled by the adapter's declared roots in their OWN spelling -- of the identity and its
local spelling, the one lying under a root; the identity when both or neither do (the home tree is production). The re-pointed-root shape
keeps its local spelling (part 4), so the Phase 3 identity tests stand as they are.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pytest

import sisteer
from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import reader_state as RS
from sisteer.registry import records as R
from sisteer.registry import synth
from sisteer.registry.adapters import base as AB
from sisteer.registry.adapters import runner as RUN
from test_registry_adapter_pass import SyntheticStoreAdapter, _setup, _store

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(sisteer.__file__)))
REPO_ROOT = os.path.dirname(SRC_DIR)
PRODUCER = {"script": "tests.prior_group_spelling", "code_sha": synth.code_sha("a"), "session": None, "host": "TESTHOST", "substrate": "grading_host"}
NODE_PRODUCER = dict(PRODUCER, substrate="cluster_a", host="node-x")


class PathGroupStoreAdapter(SyntheticStoreAdapter):
    """The synthetic store adapter with PATH-VALUED groups (the store directory, as ``cellstore.group_of_input`` and sampled_static spell
    theirs) over two-file stores -- ``store.json`` and an optional ``manifest.json``, both inputs of the store's group, the read and the
    state rows over ``store.json`` alone -- and an ``inputs_retired`` declaration spelled under the ROOT, the way the live adapters'
    declarations hang off the home at import."""
    name = "synthetic_path_store"

    def __init__(self, store_root: str):
        super().__init__(store_root)
        self.retired_groups: set = set()        #: the store directories whose files the producer moved away on purpose

    def discover(self) -> List[AB.Input]:
        out = []
        for d in sorted(os.listdir(self.roots[0])):
            for fn in ("store.json", "manifest.json"):
                p = os.path.join(self.roots[0], d, fn)
                if os.path.lexists(p):
                    out.append(AB.check_input(p, role="report", allow_symlink=self.allow_symlinks))
        return out

    def comparison_group(self, inp: AB.Input) -> str:
        return os.path.dirname(inp.uri)

    def inputs_retired(self, group: str) -> List[str]:
        return [os.path.join(group, "*")] if group in self.retired_groups else []

    def read(self, ctx, group, inputs):
        yield from super().read(ctx, group, [i for i in inputs if os.path.basename(i.uri) == "store.json"])

    def source_states(self, ctx, inputs):
        return super().source_states(ctx, [i for i in inputs if os.path.basename(i.uri) == "store.json"])


# --------------------------------------------------------------------------------------------
# the world: a grading host, then the node's home tree over a generation
# --------------------------------------------------------------------------------------------


def _world(tmp_path) -> Tuple[str, str, str, PathGroupStoreAdapter]:
    """A grading host: the registry (dims + roster folded), a home with the stores at ``<home>/run-state/stores`` -- the adapter's root, in the
    canonical spelling -- three stores, ``s3`` with a manifest beside its store, and the first pass folded (three accepted revisions).
    Returns ``(registry root, home, the stores base, the adapter)``."""
    root, _unused = _setup(tmp_path)
    home = tmp_path / "home"
    base = home / "run-state" / "stores"
    base.mkdir(parents=True)
    _store(str(base), "s1", "legacy_store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4])
    _store(str(base), "s2", "legacy_store_2", "none", [0.1, 0.9])
    _store(str(base), "s3", "legacy_store_3", "constitution_lora@aaaa", [0.7, 0.7, 0.2])
    Path(base, "s3", "manifest.json").write_text('{"complete": true}\n')
    ad = PathGroupStoreAdapter(str(base))
    with DB.reading(DB.db_path(root)) as conn:
        rep = RUN.run_pass(root, [ad], conn=conn, now=synth.t(2000), producer=PRODUCER)
    assert rep.groups_refreshed == 3, [r.error for r in rep.results]
    ing = H.ingest_once(root, workers=0, min_free=0)
    assert ing.refused == 0, ing.refusals
    return root, str(home), str(base), ad


def _lay_home_tree(tmp_path, base: str, drop: Sequence[str]) -> Tuple[str, AB.IdentityMap]:
    """The node: the base's grading-host copy moves aside, a GENERATION of it -- byte-identical, mtimes kept, minus ``drop``
    (paths relative to the base the producer moved away before the generation was built) -- stands under the shared store, and the
    canonical spelling becomes a symlink onto the generation. Returns the generation and the pass's identity map: generation -> canonical
    base, the one pair a shard's map carries per pinned component (never the home itself)."""
    gen = tmp_path / "shared" / "registry" / "stores" / "gen" / "stores_component" / "0123abcd"
    shutil.copytree(base, gen, symlinks=True)
    for rel in drop:
        p = gen / rel
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    shutil.move(base, base + ".grading_host")
    os.symlink(str(gen), base)
    return str(gen), AB.IdentityMap.build({str(gen): base})


def _export(root: str, out: str, home: str, ad) -> None:
    RS.export(root, [ad.name], out, uri_home=home, repo_root=REPO_ROOT)


def _shard_pass(cluster_root: str, ad, sidecar: str, now: str, imap: AB.IdentityMap, shard: Tuple[int, int]) -> RUN.PassReport:
    """One shard of a cluster pass against the sidecar, as ``cluster_reads.run_shard`` runs it: the map installed, the deal applied."""
    H.ensure_layout(cluster_root)
    conn = RS.open_reader_state(sidecar, [ad.name])
    try:
        cluster = RUN.ClusterRead.from_sidecar(conn, {"substrate": "cluster_a", "host": "node-x", "components": {}})
        return RUN.run_pass(cluster_root, [ad], conn=conn, now=now, producer=NODE_PRODUCER, cluster=cluster, shard=shard,
                            identity_map_=imap, save_cache=False)
    finally:
        conn.close()


def _records_by_type(rev_dir: str) -> Dict[str, List[Dict]]:
    recs: Dict[str, List[Dict]] = {}
    for fn in sorted(os.listdir(rev_dir)):
        for _, rec, _ in R.read_record_file(os.path.join(rev_dir, fn)):
            recs.setdefault(rec["record_type"], []).append(rec["payload"])
    return recs


def _marks_of(ad, inputs: Sequence[AB.Input]) -> Dict[str, RUN.Watermark]:
    """The watermarks an accepted read of ``inputs`` leaves, keyed canonically, each naming its group's canonical source id."""
    return {i.identity_uri: RUN.Watermark(RUN.source_id_for(ad.name, ad.comparison_group(i)), AB.watermark_of(i), None) for i in inputs}


# --------------------------------------------------------------------------------------------
# part 1: the rule
# --------------------------------------------------------------------------------------------


class _Roots:
    name = "roots_only"

    def __init__(self, roots):
        self.roots = tuple(roots)


def test_prior_group_spelling_follows_the_adapters_roots_in_their_own_spelling_under_both_layouts():
    """Home tree: the map carries the generation onto the canonical base and the roots are canonical -> the identity is kept. Re-pointed
    root: the map carries the local copy onto the canonical base and the roots are local -> the local spelling is taken. Both or neither
    qualifying, a non-path group, no roots declared, no map: the identity, byte for byte."""
    canon = "/grading_host-home/run-state/stores"
    gen = "/shared/registry/stores/gen/stores_component/0123abcd"
    local = "/node-local/copy/stores"
    home_tree = AB.IdentityMap.build({gen: canon})
    assert home_tree.local(canon + "/s3") == gen + "/s3"                                          # what the old rule handed the adapter
    with AB.using_identity_map(home_tree):
        assert RUN.prior_group_spelling(_Roots([canon]), canon + "/s3") == canon + "/s3"             # the identity is the adapter's spelling
        assert RUN.prior_group_spelling(_Roots([canon]), canon) == canon                             # the root itself as a group
        assert RUN.prior_group_spelling(_Roots([canon + "/"]), canon + "/s3") == canon + "/s3"       # a root spelled with a trailing slash
        assert RUN.prior_group_spelling(_Roots([]), canon + "/s3") == canon + "/s3"                  # no roots declared: the identity
        assert RUN.prior_group_spelling(_Roots([canon, gen]), canon + "/s3") == canon + "/s3"        # both qualify: the identity
        assert RUN.prior_group_spelling(_Roots([canon]), "s3") == "s3"                               # a non-path group has one spelling
        assert RUN.prior_group_spelling(_Roots([canon]), "/elsewhere/s3") == "/elsewhere/s3"         # a path the map does not touch
        assert RUN.prior_group_spelling(_Roots([canon]), canon + "_other/s3") == canon + "_other/s3"  # a prefix is not a root
        # an adapter with no roots attribute at all
        assert RUN.prior_group_spelling(object(), canon + "/s3") == canon + "/s3"
    relocated = AB.IdentityMap.build({local: canon})
    with AB.using_identity_map(relocated):
        assert RUN.prior_group_spelling(_Roots([local]), canon + "/s3") == local + "/s3"             # the re-pointed root's spelling
        assert RUN.prior_group_spelling(_Roots([local]), canon) == local
        assert RUN.prior_group_spelling(_Roots([local + "_other"]), canon + "/s3") == canon + "/s3"  # neither qualifies: the identity
        assert RUN.prior_group_spelling(_Roots([canon]), canon + "/s3") == canon + "/s3"             # roots canonical under this map too
        # the map handed explicitly wins over the installed one
        assert RUN.prior_group_spelling(_Roots([canon]), canon + "/s3", home_tree) == canon + "/s3"
    # no map installed (every grading host pass): the identity function
    assert RUN.prior_group_spelling(_Roots([local]), canon + "/s3") == canon + "/s3"
    assert RUN.prior_group_spelling(_Roots([canon]), canon + "/s3", AB.IdentityMap()) == canon + "/s3"


# --------------------------------------------------------------------------------------------
# part 2: the plan, under the home tree
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("rng_scope", ["none", "reader_traversal"])
def test_plan_groups_keys_a_partial_drop_and_a_vanished_group_by_the_key_the_plan_carries_under_the_home_tree(tmp_path, rng_scope):
    """Defect 2 at the plan: with ``s3/manifest.json`` gone from the generation, ``gone_in_group`` and ``changed`` carry the key ``groups``
    carries (the canonical spelling), never the generation's; with ``s3`` gone whole, ``vanished`` carries the canonical key. A
    ``reader_traversal`` adapter re-keys ``changed`` to every present group and ``gone_in_group`` must still be found under the plan's key."""
    root, home, base, ad = _world(tmp_path)
    ad.rng_scope = rng_scope
    s3 = os.path.join(base, "s3")
    manifest, store = os.path.join(s3, "manifest.json"), os.path.join(s3, "store.json")
    marks = _marks_of(ad, ad.discover())
    assert set(marks) == {os.path.join(base, d, "store.json") for d in ("s1", "s2", "s3")} | {manifest}
    gen, imap = _lay_home_tree(tmp_path, base, drop=["s3/manifest.json"])
    with AB.using_identity_map(imap):
        inputs = ad.discover()
        assert all(i.uri == i.identity_uri and i.uri.startswith(base) and i.path.startswith(gen) for i in inputs)      # addressed canonically
        plan = RUN.plan_groups(ad, inputs, marks, "1", None)
    assert set(plan.groups) == {os.path.join(base, d) for d in ("s1", "s2", "s3")}
    assert plan.gone_in_group == {s3: {manifest}}                              # the plan's own key, not imap.local(s3) (the generation)
    assert set(plan.gone_in_group) <= set(plan.groups) and set(plan.changed) <= set(plan.groups)
    assert plan.changed == (set(plan.groups) if rng_scope == "reader_traversal" else {s3}) and plan.vanished == {}
    assert plan.gone_in_group.get(os.path.join(gen, "s3")) is None
    # the whole store gone: a vanished group, keyed canonically, its two prior inputs
    shutil.rmtree(os.path.join(gen, "s3"))
    with AB.using_identity_map(imap):
        inputs = ad.discover()
        plan = RUN.plan_groups(ad, inputs, marks, "1", None)
    assert set(plan.groups) == {os.path.join(base, d) for d in ("s1", "s2")} and plan.gone_in_group == {}
    assert plan.vanished == {s3: {manifest, store}}
    assert plan.changed == set()        # a vanished group is retired, not read: no changed input, so no traversal re-read either


# --------------------------------------------------------------------------------------------
# part 3: the shard, under the home tree
# --------------------------------------------------------------------------------------------


def test_a_vanished_group_declared_retired_is_committed_by_its_owning_shard_under_the_home_tree(tmp_path):
    """Defect 1: the store ``s3`` left the generation whole and the adapter declares its files retired under the canonical spelling.
    Its one owning shard COMMITS the retirement -- the empty accepted revision with ``extra.retirement_declared`` naming the pattern and
    both inputs, the ``input_vanished`` withdrawals, both watermarks retired -- and nothing is refused. Before the rule the group reached
    the retirement generation-spelled, the declaration matched nothing and the shard refused it ``Missing: `` (staged_copy_incomplete)."""
    root, home, base, ad = _world(tmp_path)
    s3 = os.path.join(base, "s3")
    s3_uris = [os.path.join(s3, "manifest.json"), os.path.join(s3, "store.json")]
    sidecar = str(tmp_path / "rs.sqlite")
    _export(root, sidecar, home, ad)
    gen, imap = _lay_home_tree(tmp_path, base, drop=["s3"])
    ad.retired_groups.add(s3)                                   # the declaration, spelled as the adapter addresses the group
    assert imap.local(s3) == os.path.join(gen, "s3") and AB.inputs_retired_of(ad, imap.local(s3)) == ()      # the generation spelling declares nothing
    reporting = []
    for i in range(3):
        cluster_root = str(tmp_path / f"cluster_root_{i}")
        rep = _shard_pass(cluster_root, ad, sidecar, synth.t(3000), imap, (i, 3))
        assert rep.groups_seen == 3, rep
        if rep.results:
            reporting.append((i, cluster_root, rep))
    assert len(reporting) == 1, [(i, [r.error for r in rep.results]) for i, _c, rep in reporting]   # exactly one owner; the others report nothing
    _i, cluster_root, rep = reporting[0]
    (res,) = rep.results
    assert res.group == s3 and res.group_identity == s3, res
    assert res.accepted and res.error is None and not res.inputs_dropped and res.withdrawals > 0, res.error
    assert rep.groups_refused == 0 and rep.groups_inputs_dropped == 0 and rep.withdrawals == res.withdrawals
    rev_dir = os.path.join(H.records_dir(cluster_root), "adapters", ad.name, res.revision_id)
    recs = _records_by_type(rev_dir)
    marker = recs["source_revision"][0]
    assert marker["source_id"] == f"{ad.name}:{s3}" and marker["inputs_json"] == {} and marker["accepted"] == 1
    assert marker["extra"][RUN.RETIREMENT_DECLARED_KEY] == {"declared_by": "adapter:inputs_retired", "patterns": [os.path.join(s3, "*")], "inputs": s3_uris}
    assert marker["extra"][H.DECLARED_RUN_DROPS_KEY]["declared_by"] == ["runner:input_vanished", "runner:inputs_retired"]
    assert recs["withdrawal"] and all(w["reason"] == "input_vanished" for w in recs["withdrawal"])
    assert sorted(w["input_uri"] for w in recs["source_watermark"]) == s3_uris
    assert all(w["last_error"] == RUN.VANISHED_PREFIX + w["input_uri"] for w in recs["source_watermark"])
    assert recs["source_status"][0]["ok"] == 0 and sorted(recs["source_status"][0]["extra"]["missing"]) == s3_uris
    # no identity-bearing record spells the generation
    for rt in ("source_revision", "withdrawal", "source_watermark", "source_state", "source_status"):
        assert gen not in json.dumps(recs.get(rt, [])), rt


@pytest.mark.parametrize("rng_scope", ["none", "reader_traversal"])
def test_a_partial_drop_under_the_home_tree_reads_the_group_and_retires_the_gone_watermark(tmp_path, rng_scope):
    """Defect 2 end to end: ``s3/manifest.json`` left the generation (declared retired), ``s3/store.json`` stands unchanged. The owning shard
    READS ``s3`` through the key the plan carries -- no KeyError -- publishes the new revision over ``store.json`` alone (the comparison
    counts the manifest ``inputs_retired``, nothing dropped) and RETIRES the manifest's watermark. On a ``reader_traversal`` adapter (K = 1;
    every present group re-keyed changed; the unchanged ones refused as known to the grading host) the same, where before the rule the drop was
    keyed by the generation spelling nobody looked up: the group read, the watermark never retired."""
    root, home, base, ad = _world(tmp_path)
    ad.rng_scope = rng_scope
    s3 = os.path.join(base, "s3")
    manifest, store = os.path.join(s3, "manifest.json"), os.path.join(s3, "store.json")
    sidecar = str(tmp_path / "rs.sqlite")
    _export(root, sidecar, home, ad)
    gen, imap = _lay_home_tree(tmp_path, base, drop=["s3/manifest.json"])
    ad.retired_groups.add(s3)                                   # the manifest left on purpose: declared, so the shard's any-drop rule admits the read
    shards = 1 if rng_scope == "reader_traversal" else 3
    reporting = []
    for i in range(shards):
        cluster_root = str(tmp_path / f"cluster_root_{i}")
        rep = _shard_pass(cluster_root, ad, sidecar, synth.t(3000), imap, (i, shards))      # before the rule: KeyError out of the reading loop
        assert rep.groups_seen == 3, rep
        if rep.results:
            reporting.append((cluster_root, rep))
    assert len(reporting) == 1, [[r.error for r in rep.results] for _c, rep in reporting]
    cluster_root, rep = reporting[0]
    accepted = [r for r in rep.results if r.accepted]
    assert len(accepted) == 1 and rep.groups_refreshed == 1 and rep.groups_refused == 0 and rep.groups_inputs_dropped == 0, [r.error for r in rep.results]
    assert rep.groups_refused_known == (2 if rng_scope == "reader_traversal" else 0)        # the traversal re-read s1 and s2: known to the grading host
    (res,) = accepted
    assert res.group == s3 and res.group_identity == s3 and res.error is None
    rev_dir = os.path.join(H.records_dir(cluster_root), "adapters", ad.name, res.revision_id)
    recs = _records_by_type(rev_dir)
    marker = recs["source_revision"][0]
    assert marker["source_id"] == f"{ad.name}:{s3}" and list(marker["inputs_json"]) == [store] and marker["accepted"] == 1
    assert marker["extra"]["inputs_previous"] == 2 and marker["extra"]["inputs_now"] == 1
    assert marker["extra"]["inputs_dropped"] == 0 and marker["extra"]["inputs_retired"] == 1
    retired = [w for w in recs["source_watermark"] if w["last_error"]]
    assert [w["input_uri"] for w in retired] == [manifest] and retired[0]["last_error"] == RUN.VANISHED_PREFIX + manifest
    assert gen not in json.dumps(recs["source_revision"]) and gen not in json.dumps(recs["source_watermark"])


# --------------------------------------------------------------------------------------------
# part 4: the re-pointed root keeps its local spelling
# --------------------------------------------------------------------------------------------


def test_a_re_pointed_root_spells_a_vanished_and_a_partially_dropped_group_locally_and_identifies_them_canonically(tmp_path):
    """The Phase 3 identity tests' shape (``test_registry_cluster_read`` part 1): an adapter constructed over a LOCAL copy of the stores, the
    map local -> canonical, its declaration spelled under the local root. A partial drop is read under the local key (the comparison counts
    the manifest ``inputs_retired``; the watermark retires under its canonical URI), and a vanished group is retired under the local
    spelling with the declaration canonicalized -- the behavior the old rule gave this shape, kept."""
    root, home, base, ad = _world(tmp_path)
    s3 = os.path.join(base, "s3")
    manifest, store = os.path.join(s3, "manifest.json"), os.path.join(s3, "store.json")
    marks = _marks_of(ad, ad.discover())                        # the grading host's watermarks: canonical URIs, every input of the three stores
    # the partial drop: a local copy without the manifest
    local = str(tmp_path / "node-local" / "stores")
    shutil.copytree(base, local)
    os.remove(os.path.join(local, "s3", "manifest.json"))
    ad_local = PathGroupStoreAdapter(local)
    ad_local.retired_groups.add(os.path.join(local, "s3"))                        # the declaration hangs off the RE-POINTED root
    imap = AB.IdentityMap.build({local: base})
    with AB.using_identity_map(imap):
        plan = RUN.plan_groups(ad_local, ad_local.discover(), marks, "1", None)
    assert plan.gone_in_group == {os.path.join(local, "s3"): {manifest}} and plan.changed == {os.path.join(local, "s3")}
    assert set(plan.changed) <= set(plan.groups)
    with DB.reading(DB.db_path(root)) as conn:
        rep = RUN.run_pass(root, [ad_local], conn=conn, now=synth.t(2100), producer=PRODUCER, identity_map_=imap)
    (res,) = rep.results
    assert res.accepted and res.group == os.path.join(local, "s3") and res.group_identity == s3, res.error
    recs = _records_by_type(os.path.join(H.records_dir(root), "adapters", ad.name, res.revision_id))
    marker = recs["source_revision"][0]
    assert marker["source_id"] == f"{ad.name}:{s3}" and list(marker["inputs_json"]) == [store]
    assert marker["extra"]["inputs_dropped"] == 0 and marker["extra"]["inputs_retired"] == 1
    assert [w["input_uri"] for w in recs["source_watermark"] if w["last_error"]] == [manifest]
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    # the vanished group: a second local copy without the store at all
    local2 = str(tmp_path / "node-local-2" / "stores")
    shutil.copytree(base, local2)
    shutil.rmtree(os.path.join(local2, "s3"))
    ad_local2 = PathGroupStoreAdapter(local2)
    ad_local2.retired_groups.add(os.path.join(local2, "s3"))
    imap2 = AB.IdentityMap.build({local2: base})
    with AB.using_identity_map(imap2):
        assert RUN.prior_group_spelling(ad_local2, s3) == os.path.join(local2, "s3")
    with DB.reading(DB.db_path(root)) as conn:
        rep2 = RUN.run_pass(root, [ad_local2], conn=conn, now=synth.t(2200), producer=PRODUCER, identity_map_=imap2)
    (res2,) = rep2.results
    assert res2.accepted and res2.withdrawals > 0 and res2.group == os.path.join(local2, "s3") and res2.group_identity == s3, res2.error
    recs2 = _records_by_type(os.path.join(H.records_dir(root), "adapters", ad.name, res2.revision_id))
    marker2 = recs2["source_revision"][0]
    assert marker2["source_id"] == f"{ad.name}:{s3}" and marker2["inputs_json"] == {}
    # the declaration on the marker is canonical: the local pattern spelled through the map, the input the head read (store.json alone)
    assert marker2["extra"][RUN.RETIREMENT_DECLARED_KEY] == {"declared_by": "adapter:inputs_retired", "patterns": [os.path.join(s3, "*")], "inputs": [store]}
    assert local2 not in json.dumps(recs2["source_revision"]) and local2 not in json.dumps(recs2["source_watermark"])
