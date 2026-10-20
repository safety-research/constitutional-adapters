"""Cluster reads.

Part 1 -- IDENTITY beside ADDRESSING: a read of a synthetic store under a RELOCATED root (the cluster task's relocated home) mints the
very ids the read under the canonical spelling mints -- comparison groups, source ids, revision ids, watermark URIs, ``inputs_json``,
observation and item membership keys, state and status rows -- for each of the six heavy adapters, while every local-path computation
(``relpath`` groups, root checks, prefix classes) keeps running against the local spelling. Live-STYLE source ids (``sampled_static:
/home/<user>/.../grades_sampled/<dir>``) are reproduced from a store that is nowhere near that path. The grading host path is byte-identical:
with no map installed ``identity_uri == uri`` and every record is what it was.

Part 2 -- the SHARD DEAL: the sorted union of the present groups and the prior-only (vanished) groups, by canonical identity, dealt
round-robin; K = 3 with one vanished group gives it exactly one owner; a ``reader_traversal`` adapter admits K = 1 only.

Part 3 -- the READER-STATE sidecar and the CLUSTER shard: ``registry reader-state export`` from a synthetic root; against the sidecar a
changed group publishes exactly one new id; a reverted input (its id accepted) is refused by name and the grading host's pass re-activates
it; ``--full`` re-reads and refuses known ids identically; ``--reread`` is refused outright; a held id is refused by name; an id the
grading host REFUSED AT EMIT TIME and never committed is NOT known and is read fresh; a ruling-backed group reads through the real
``rulings`` symlink and refuses by name without it; a group whose every input vanished is REFUSED by its one owner (a staged copy that
lacks it is incomplete until the grading host says otherwise) and the grading host retires it with ``input_vanished`` withdrawals and
``source_state found = 0`` rows.

Part 6 -- dropped inputs and coverage: ANY dropped input refuses a shard's group (no revision, no marker; the missing inputs named with
the component that would have covered them; the pull names it as the adapter's refusal reason) while the grading host keeps the 10 %
threshold; the plan runs NO discovery on the grading host (coverage is the shard's; the rule stays a pure function); the home tree
replaces a previous window's read-only copy instead of writing over it.

Part 8 -- re-activation by the pull: a refused-known revision a shard returned is re-activated by the pull FROM COMMITTED BYTES
(``runner.reactivate_by_id``; no store read -- proven with an unscoped host and the audit hook), a current one is left alone, a marker without
its committed watermark facts refuses by name and the adapter is recorded refused; an id refused at emit time and never committed is
``refused_standing`` -- nothing to re-activate from, never the adapter's refusal -- while a committed id whose directory is missing
still refuses by name.
"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import stat
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pytest

import sisteer
import sisteer.site
from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import records as R
from sisteer.registry import synth
from sisteer.registry.adapters import base as AB
from sisteer.registry.adapters import runner as RUN

import registry_items_fixtures as F
import _rsync_fallback as RF
from _rsync_fallback import rsync_on_path  # noqa: F401  (autouse: the fallback rsync on PATH where the host has none; a no-op where it has one)

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(sisteer.__file__)))
REPO_ROOT = os.path.dirname(SRC_DIR)


def _cluster_root(tmp_path, name: str = "cluster_root") -> str:
    """A registry layout standing for the window root's spool on the cluster."""
    # the pulls over this spool copy with rsync: the real binary, else the fallback (the CPU suite). Every plan here names host None, so
    # the operands are local paths and the fallback's refusal of a remote operand never fires (no opt-in); the `-e` a pull adds for an
    # unnamed source is ignored for a local copy, by the real binary and the fallback alike
    RF.ensure_rsync_on_path()
    root = str(tmp_path / name)
    H.ensure_layout(root)
    return root


def _marker(rev_dir: str) -> Dict:
    """The source_revision marker record of a published revision directory."""
    return next(rec for fn in os.listdir(rev_dir) if "source_revision" in fn
                for _, rec, _ in R.read_record_file(os.path.join(rev_dir, fn)) if rec["record_type"] == "source_revision")


def _same_size_edit(path: str) -> bytes:
    """Change the file's bytes without changing its size (the edit a stat walk misses and a hash catches); returns the original bytes."""
    data = open(path, "rb").read()
    open(path, "wb").write(data[:-2] + b"9}")
    return data
NOW = "2026-09-28T22:00:00.000Z"
PRODUCER = {"script": "tests.cluster_read", "code_sha": synth.code_sha("a"), "session": None, "host": "TESTHOST", "substrate": "grading_host"}
#: the canonical home a relocated read maps onto in these tests (a spelling that exists on no test box)
CANON_HOME = "/home/canonical-user"
RUN_STATE_REL = os.path.join(".local", "state", "romp", "candidate-si-refit", "run-state", "full22")


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------


def _relocate(src: str, dst: str) -> None:
    """A byte-identical copy of a whole world (mtimes kept, symlinks kept as links): the cluster's generation of the grading host store."""
    shutil.copytree(src, dst, symlinks=True)


def _remap(obj, src: str, dst: str):
    if isinstance(obj, str):
        return dst + obj[len(src):] if obj == src or obj.startswith(src + os.sep) else obj
    if isinstance(obj, dict):
        return {k: _remap(v, src, dst) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_remap(v, src, dst) for v in obj)
    return obj


def _pass(reg: str, adapter, imap: Optional[AB.IdentityMap] = None, conn=None, **kw) -> RUN.PassReport:
    H.ensure_layout(reg)
    if conn is None:
        # a contrast-filing read needs the pass's anchor frame (emitter 4; the bcbw and sampled readers since commit 7 of the adapter re-anchoring plan): the
        # registry database -- its schema carries v_run_current -- created by one ingest; on a fresh registry every comparison parks, deterministically
        H.ingest_once(reg, workers=0, min_free=0)
        with DB.reading(DB.db_path(reg)) as c:
            return RUN.run_pass(reg, [adapter], conn=c, now=NOW, producer=PRODUCER, identity_map_=imap, **kw)
    return RUN.run_pass(reg, [adapter], conn=conn, now=NOW, producer=PRODUCER, identity_map_=imap, **kw)


def _revision_facts(reg: str, adapter_name: str) -> Dict[str, Dict[str, object]]:
    """Per published revision directory (fresh reads only): every identity-bearing fact, plus the logical shas of every file but the
    artifacts (whose second location names the local copy by design)."""
    base = os.path.join(H.records_dir(reg), "adapters", adapter_name)
    out: Dict[str, Dict[str, object]] = {}
    for rev in sorted(os.listdir(base)) if os.path.isdir(base) else []:
        d = os.path.join(base, rev)
        if rev.startswith("_") or not os.path.isdir(d):
            continue
        facts: Dict[str, object] = {"observation": set(), "items": set(), "watermarks": set(), "states": set(), "where": set(),
                                    "artifacts": set(), "shas": {}}
        for fn in sorted(os.listdir(d)):
            path = os.path.join(d, fn)
            if "-artifact" not in fn:
                facts["shas"][fn] = R.file_logical_sha(path)
            for _, rec, _ in R.read_record_file(path):
                rt, p = rec.get("record_type"), rec.get("payload") or {}
                if rt == "source_revision":
                    facts["source_id"], facts["inputs_json"] = p["source_id"], json.dumps(p["inputs_json"], sort_keys=True)
                    facts["revision_id"] = p["revision_id"]
                elif rt == "observation":
                    facts["observation"].update((p["table_name"], k) for k in p["observation_keys"])
                elif rt == "item_result":
                    facts["items"].add(json.dumps({c: p.get(c) for c in R.ITEM_MEMBERSHIP_KEY}, sort_keys=True))
                elif rt == "source_watermark":
                    facts["watermarks"].add((p["source_id"], p["input_uri"], p["sha256"]))
                elif rt == "source_state":
                    facts["states"].add((p["source_id"], p["input_uri"], p["json_path"], p["found"]))
                elif rt == "source_status":
                    facts["where"].update(p["where_json"])
                elif rt == "artifact":
                    facts["artifacts"].add((p["artifact_id"], p["uri_primary"]))
        out[rev] = facts
    return out


def _assert_same_identity(reg_a: str, reg_b: str, adapter_name: str, expect_home_spelled: Optional[str] = None) -> Dict[str, Dict[str, object]]:
    a, b = _revision_facts(reg_a, adapter_name), _revision_facts(reg_b, adapter_name)
    assert a and sorted(a) == sorted(b), (sorted(a), sorted(b))
    for rev in a:
        for key in ("source_id", "inputs_json", "revision_id", "observation", "items", "watermarks", "states", "where", "artifacts", "shas"):
            assert a[rev][key] == b[rev][key], (rev, key)
        if expect_home_spelled is not None:
            # every identity string spells the canonical home, never the local copy -- the synthetic anchor-decision input (`registry://anchor-decision/
            # <source id>`, its watermark written from the post-read fingerprint) through the canonical source id it embeds
            for uri in list(json.loads(a[rev]["inputs_json"])) + [w[1] for w in a[rev]["watermarks"]] + list(a[rev]["where"]):
                if AC.is_decision_uri(uri):
                    assert AC.decision_source_id(uri).split(":", 1)[1].startswith(expect_home_spelled), uri
                    continue
                assert uri.startswith(expect_home_spelled), uri
    return a


def _no_local_spelling(reg: str, adapter_name: str, local_root: str) -> None:
    """No record file of the relocated read spells the local copy (identity vs addressing) -- except the artifact's declared second location."""
    base = os.path.join(H.records_dir(reg), "adapters", adapter_name)
    for rev in os.listdir(base):
        d = os.path.join(base, rev)
        for fn in os.listdir(d):
            if "-artifact" in fn:
                continue
            for _, rec, _ in R.read_record_file(os.path.join(d, fn)):
                assert local_root not in R.canonical_json(rec), (fn, rec.get("record_type"))


# --------------------------------------------------------------------------------------------
# part 1: the identity map itself, and the grading host path unchanged
# --------------------------------------------------------------------------------------------


def test_identity_map_rewrites_prefixes_aliases_groups_and_source_ids_and_inverts():
    m = AB.IdentityMap.build({"/tmp/w/home": CANON_HOME, "/shared/u/exp/misalign": CANON_HOME + "/run-state/misalign"})
    assert m and m.pairs[0][0] == "/shared/u/exp/misalign"                       # longest local prefix first
    assert m.canonical("/tmp/w/home/a/b.json") == CANON_HOME + "/a/b.json"
    assert m.canonical("/tmp/w/home/a/b.json#s1") == CANON_HOME + "/a/b.json#s1"          # a `#<group>` alias keeps its suffix
    assert m.canonical("/tmp/w/home") == CANON_HOME and m.canonical("/tmp/w/homer/x") == "/tmp/w/homer/x"
    assert m.canonical("relative/group") == "relative/group" and m.canonical("_lane_files") == "_lane_files"
    assert m.local(CANON_HOME + "/a") == "/tmp/w/home/a" and m.local(m.canonical("/tmp/w/home/q")) == "/tmp/w/home/q"
    assert m.canonical_source_id("sampled_static:/tmp/w/home/run-state/sampled/grades_sampled/x") == f"sampled_static:{CANON_HOME}/run-state/sampled/grades_sampled/x"
    assert m.canonical_source_id("relu_paired:qwen35_4b_s1") == "relu_paired:qwen35_4b_s1" and m.canonical_source_id("noseparator") == "noseparator"
    deep = m.canonical_deep({"a": ["/tmp/w/home/x", 1, None], "b": ("/tmp/w/homer",), "c": {"d": "/shared/u/exp/misalign/f"}})
    assert deep == {"a": [CANON_HOME + "/x", 1, None], "b": ("/tmp/w/homer",), "c": {"d": CANON_HOME + "/run-state/misalign/f"}}
    with pytest.raises(ValueError):
        AB.IdentityMap.build({"/tmp/x": "relative"})
    empty = AB.IdentityMap()
    assert not empty and empty.canonical("/tmp/w/home/x") == "/tmp/w/home/x" and empty.canonical_deep({"k": "/tmp"}) == {"k": "/tmp"}


def test_identity_map_from_environ_maps_this_home_onto_the_named_one_and_is_empty_otherwise(monkeypatch):
    assert not AB.IdentityMap.from_environ({})
    home = os.path.expanduser("~")
    m = AB.IdentityMap.from_environ({AB.URI_HOME_ENV: CANON_HOME})
    assert m.pairs == ((home, CANON_HOME),)
    assert not AB.IdentityMap.from_environ({AB.URI_HOME_ENV: home})            # the same home: nothing to map


def test_grading_host_path_is_byte_identical_without_a_map(tmp_path):
    p = tmp_path / "f.json"
    p.write_text("{}")
    inp = AB.check_input(str(p), role="report")
    assert inp.identity_uri == inp.uri == str(p)
    pay = inp.artifact_payload()
    assert pay["uri_primary"] == str(p) and pay["locations_json"] == [{"uri": str(p), "verified_utc": inp.mtime_utc, "bytes": inp.size}]
    positional = AB.Input(str(p), str(p), "0" * 64, 2, inp.mtime_utc)                # the pre-Phase-3 positional shape still builds
    assert positional.identity_uri == str(p)
    assert AB.comparison_group_identity("/any/group") == "/any/group" and RUN.source_id_for("x", "/any/group") == "x:/any/group"


def test_check_input_derives_identity_uri_through_the_installed_map_and_the_artifact_names_both(tmp_path):
    home = tmp_path / "home"
    (home / "store").mkdir(parents=True)
    f = home / "store" / "rows.jsonl"
    f.write_text("{}\n")
    with AB.using_identity_map(AB.IdentityMap.build({str(home): CANON_HOME})):
        inp = AB.check_input(str(f), role="rows")
        assert inp.uri == str(f) and inp.path == str(f) and inp.identity_uri == CANON_HOME + "/store/rows.jsonl"
        pay = inp.artifact_payload()
        assert pay["uri_primary"] == CANON_HOME + "/store/rows.jsonl" and pay["artifact_id"] == inp.sha256
        assert pay["locations_json"][0]["uri"] == CANON_HOME + "/store/rows.jsonl" and pay["locations_json"][1] == \
            {"uri": str(f), "verified_utc": inp.mtime_utc, "bytes": inp.size, "read_root": True}
        assert AB.comparison_group_identity(str(home / "store")) == CANON_HOME + "/store"
        assert RUN.source_id_for("sampled_static", str(home / "store")) == f"sampled_static:{CANON_HOME}/store"
    assert AB.check_input(str(f)).identity_uri == str(f)                                 # the map left with the block


def test_adapter_roots_hang_off_home_and_the_environment_map_canonicalizes_them(tmp_path):
    """The relocation mechanism end to end, in a fresh interpreter: HOME relocated + SISTEER_REGISTRY_URI_HOME naming the grading host home ->
    every module-level root resolves under the relocated home, and the map spells it back canonically."""
    home = tmp_path / "relocated-home"
    home.mkdir()
    code = ("import os; from sisteer.registry.adapters import sampled_static as SS, _items_common as IC, declared_common as DC, lineage_common as LC, "
            "relu_paired as RP, em_misalign as EM; from sisteer.registry.adapters import base as AB\n"
            "m = AB.IdentityMap.from_environ()\n"
            "print(SS.SAMPLED_ROOT); print(m.canonical(SS.SAMPLED_ROOT)); print(m.canonical(RP.RUN_STATE_ROOT)); print(m.canonical(RP.PETRI_LEDGER_ROOT))\n"
            "print(IC.RUN_STATE == DC.RUN_STATE == LC.RS_DEFAULT); print(m.canonical(EM.EM_ROOT)); print(LC.rel_home(SS.SAMPLED_ROOT))\n"
            "with AB.using_identity_map(m): print(AB.comparison_group_identity(os.path.join(SS.SAMPLED_ROOT, 'grades_sampled', 'd1')))\n")
    env = dict(os.environ, HOME=str(home), PYTHONPATH=SRC_DIR)
    env[AB.URI_HOME_ENV] = CANON_HOME
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 0, r.stderr
    lines = r.stdout.splitlines()
    assert lines[0] == os.path.join(str(home), RUN_STATE_REL, "sampled")
    assert lines[1] == os.path.join(CANON_HOME, RUN_STATE_REL, "sampled")
    assert lines[2] == os.path.join(CANON_HOME, RUN_STATE_REL, "relu_paired") and lines[3] == os.path.join(CANON_HOME, ".local", "state", "petri_native")
    assert lines[4] == "True" and lines[5] == os.path.join(CANON_HOME, "figs", "si-steering-misalign")
    assert lines[6] == "~/" + os.path.join(RUN_STATE_REL, "sampled")                      # rel_home stays `~/...` under either home
    assert lines[7] == os.path.join(CANON_HOME, RUN_STATE_REL, "sampled", "grades_sampled", "d1")


# --------------------------------------------------------------------------------------------
# part 1: the six adapters read the same identities under a relocated root
# --------------------------------------------------------------------------------------------

SS_TAGS = {"synthmodel": ("synthmodel", "base→instruct", "synthmodel")}


def _sampled_static_world(base: str):
    from sisteer.registry.adapters import sampled_static as SS
    root = os.path.join(base, RUN_STATE_REL, "sampled")
    F.write_sampled_root(root)
    return lambda b: SS.SampledStaticAdapter(root=os.path.join(b, RUN_STATE_REL, "sampled"), dims=F.dims_map(), campaign="synthcampaign", tags=SS_TAGS)


def _bcbw_world(base: str, eval_id: str):
    import _bcbw_full_fixtures as Z
    from sisteer.registry.adapters import bfcl_full as BF
    from sisteer.registry.adapters import bigcodebench_full as BCF
    from sisteer.registry.adapters import lineage_common as LC
    seeds = LC.Seeds.locate()
    fx = Z.mirror(Path(base), seeds, eval_id=eval_id, root=os.path.join(base, RUN_STATE_REL, f"{eval_id}_full"))
    cls = BF.BfclFullAdapter if eval_id == "bfcl" else BCF.BigcodebenchFullAdapter
    return lambda b: Z.adapter(cls, _remap(fx, base, b), seeds)


def _multiturn_world(base: str):
    import test_registry_adapter_multiturn_grades as T
    import _b1_fixtures as B1
    from sisteer.registry.adapters import multiturn_grades as MG
    T._root(Path(base))                                  # lays the run-state fixture out under this base
    seeds = B1.synthetic_seeds()
    B1.resolver(seeds, Path(base))                       # the object manifests the resolver reads, under this base

    def build(b: str):
        return MG.MultiturnGradesAdapter(rs=os.path.join(b, "rs"), seeds=seeds, objects=B1.resolver(seeds, Path(b)),
                                         transfer_stats_files=("transfer_stats_full22_all.json",))
    return build


def _em_misalign_world(base: str):
    import test_registry_em_misalign_seedrep as T
    w = T._world(Path(base))
    return lambda b: T._adapter(_remap(w, base, b))


def _relu_paired_world(base: str):
    import test_registry_adapter_relu_paired as T
    fx = T.build_stage_b(base)
    return lambda b: T._adapter(_remap({"rs": fx["rs"], "man": fx["man"]}, base, b))


WORLDS: Dict[str, Callable[[str], Callable[[str], object]]] = {
    "sampled_static": _sampled_static_world,
    "bfcl_full": lambda base: _bcbw_world(base, "bfcl"),
    "bigcodebench_full": lambda base: _bcbw_world(base, "bigcodebench"),
    "multiturn_grades": _multiturn_world,
    "em_misalign": _em_misalign_world,
    "relu_paired": _relu_paired_world,
}


#: the adapters whose stores carry no absolute grading host path INSIDE them, so a copy under another spelling reads whole through the identity
#: map alone; relu_paired follows absolute `pass2_store` / `store` paths its indexes embed (the live final_rung.json indexes carry
#: ~3,000 `/home/<user>/...` strings) -- for it the canonical spelling itself must resolve, which is how the cluster task lays the home
#: tree out (test_canonical_spelling_resolving_through_a_symlink_reads_every_adapter_identically)
MAP_RELOCATABLE = sorted(n for n in WORLDS if n != "relu_paired")


@pytest.mark.parametrize("adapter_name", MAP_RELOCATABLE)
def test_adapters_mint_the_same_identities_under_a_relocated_root_through_the_map(tmp_path, adapter_name):
    """The synthetic world at A (the 'grading host' spelling) and its byte-identical copy at B (the 'cluster' spelling) read with the map
    {B: A}: equal revision ids, source ids, inputs_json, watermark URIs, observation / item keys, state and status rows, and equal
    record bytes (artifacts aside); no record of the relocated read spells B."""
    a, b = str(tmp_path / "A"), str(tmp_path / "B")
    os.makedirs(a)
    build = WORLDS[adapter_name](a)
    _relocate(a, b)
    reg_a, reg_b = str(tmp_path / "reg_a"), str(tmp_path / "reg_b")
    rep_a = _pass(reg_a, build(a))
    rep_b = _pass(reg_b, build(b), imap=AB.IdentityMap.build({b: a}))
    assert rep_a.groups_refreshed > 0 and rep_a.groups_refreshed == rep_b.groups_refreshed, ([r.error for r in rep_a.results], [r.error for r in rep_b.results])
    assert rep_a.groups_refused == rep_b.groups_refused and rep_a.records_written == rep_b.records_written
    assert sorted((r.group_identity, r.revision_id, r.accepted) for r in rep_a.results) == sorted((r.group_identity, r.revision_id, r.accepted) for r in rep_b.results)
    facts = _assert_same_identity(reg_a, reg_b, adapter_name)
    _no_local_spelling(reg_b, adapter_name, b)
    # the relocated read ADDRESSED the copy: the artifacts' second location names B, the primary the canonical spelling
    for rev in facts:
        seen_b = False
        for fn in os.listdir(os.path.join(H.records_dir(reg_b), "adapters", adapter_name, rev)):
            if "-artifact" in fn:
                for _, rec, _ in R.read_record_file(os.path.join(H.records_dir(reg_b), "adapters", adapter_name, rev, fn)):
                    locs = rec["payload"]["locations_json"]
                    if locs[0]["uri"].startswith(a):
                        assert len(locs) == 2 and locs[1]["uri"].startswith(b) and locs[1]["read_root"] is True
                        seen_b = True
        assert seen_b, rev


@pytest.mark.parametrize("adapter_name", sorted(WORLDS))
def test_canonical_spelling_resolving_through_a_symlink_reads_every_adapter_identically(tmp_path, adapter_name):
    """The layout the cluster task builds: the store's CANONICAL spelling exists on the node as a symlink onto the
    immutable generation, so every adapter root -- and every absolute path a store embeds -- resolves; `Input.uri` is canonical,
    `Input.path` the generation. Read at A as a real tree, then at A as a symlink onto the copy B: equal identities and equal record
    bytes, artifacts included (one location: the identity IS the local spelling), and no record spells the generation's path -- the
    task installs the map {generation path: canonical base} so a reader that resolves a file's REAL path into a record (relu_paired's
    object weights location) still spells the canonical home."""
    a, b = str(tmp_path / "A"), str(tmp_path / "gen" / "sha8")
    os.makedirs(a)
    build = WORLDS[adapter_name](a)
    reg_a, reg_b = str(tmp_path / "reg_a"), str(tmp_path / "reg_b")
    rep_a = _pass(reg_a, build(a))
    os.makedirs(os.path.dirname(b))
    shutil.move(a, b)                      # the grading host tree becomes the generation; the canonical spelling now resolves through a link
    os.symlink(b, a)
    rep_b = _pass(reg_b, build(a), imap=AB.IdentityMap.build({b: a}))
    assert rep_a.groups_refreshed > 0 and rep_a.groups_refreshed == rep_b.groups_refreshed, ([r.error for r in rep_a.results], [r.error for r in rep_b.results])
    assert rep_a.groups_refused == rep_b.groups_refused and rep_a.records_written == rep_b.records_written
    facts_a, facts_b = _revision_facts(reg_a, adapter_name), _revision_facts(reg_b, adapter_name)
    assert sorted(facts_a) == sorted(facts_b)
    for rev in facts_a:
        for key in ("source_id", "inputs_json", "revision_id", "observation", "items", "watermarks", "states", "where", "artifacts", "shas"):
            assert facts_a[rev][key] == facts_b[rev][key], (rev, key)
        da, db = os.path.join(H.records_dir(reg_a), "adapters", adapter_name, rev), os.path.join(H.records_dir(reg_b), "adapters", adapter_name, rev)
        assert RUN._dir_shas(da) == RUN._dir_shas(db), rev            # every file, the artifacts included
    _no_local_spelling(reg_b, adapter_name, b)


def test_relu_paired_indexes_embed_absolute_paths_so_the_canonical_spelling_must_resolve(tmp_path):
    """The hazard the symlink layout answers, pinned: relu_paired's controls index names `pass2_store` directories by ABSOLUTE path, so a
    copy under another spelling read through the map alone discovers the index-named cells at the grading host spelling beside its own --
    the read is not the grading host's. (The live indexes carry the same shape: thousands of `/home/<user>/...` strings.)"""
    a, b = str(tmp_path / "A"), str(tmp_path / "B")
    os.makedirs(a)
    build = _relu_paired_world(a)
    _relocate(a, b)
    ia = {os.path.relpath(i.uri, a) for i in build(a).discover()}
    with AB.using_identity_map(AB.IdentityMap.build({b: a})):
        ib = [i.uri for i in build(b).discover()]
    outside = [u for u in ib if not u.startswith(b)]
    assert outside and all(u.startswith(a) for u in outside)
    assert all("/cells/" in u for u in outside) and len(ib) == len(ia) + len(outside)


def test_live_style_home_spelled_source_ids_reproduce_from_a_store_nowhere_near_that_home(tmp_path):
    """sampled_static's groups are absolute directories: 521 live source ids spell the grading host home (`sampled_static:/home/<user>/.../
    grades_sampled/<dir>`). Read from a copy under any local root with the map onto the canonical home, the source ids and the revision
    ids are the canonical ones -- a pure function of (canonical source id, {identity_uri: sha}, EMISSION_VERSION)."""
    local_home = str(tmp_path / "node-local-home")
    build = _sampled_static_world(local_home)
    ad = build(local_home)
    reg = str(tmp_path / "reg")
    imap = AB.IdentityMap.build({local_home: CANON_HOME})
    rep = _pass(reg, ad, imap=imap)
    canon_root = os.path.join(CANON_HOME, RUN_STATE_REL, "sampled")
    grades_groups = [r for r in rep.results if "grades_sampled" in r.group]
    assert grades_groups and all(r.accepted for r in grades_groups), [r.error for r in rep.results]
    for r in grades_groups:
        assert r.group.startswith(local_home) and r.group_identity == canon_root + r.group[len(os.path.join(local_home, RUN_STATE_REL, "sampled")):]
        source_id = f"sampled_static:{r.group_identity}"
        assert source_id.startswith(f"sampled_static:{canon_root}/grades_sampled/")
        # recompute the id from the canonical facts alone: the same function the grading host pass uses, fed the canonical spelling
        marker = _marker(os.path.join(H.records_dir(reg), "adapters", "sampled_static", r.revision_id))
        p = marker["payload"]
        assert p["source_id"] == source_id and all(u.startswith(canon_root) for u in p["inputs_json"])
        inputs = [AB.Input(uri=u, path=u, sha256=sha, size=0, mtime_utc=NOW) for u, sha in p["inputs_json"].items()]
        assert RUN.revision_id_for(source_id, inputs, AB.emission_version_of(ad)) == r.revision_id
    _no_local_spelling(reg, "sampled_static", local_home)
    # the SAME world at the canonical spelling's stand-in (a second local root, mapped onto the same canonical home) mints the same ids
    other_home = str(tmp_path / "another-node")
    _relocate(local_home, other_home)
    reg2 = str(tmp_path / "reg2")
    _pass(reg2, build(other_home), imap=AB.IdentityMap.build({other_home: CANON_HOME}))
    _assert_same_identity(reg, reg2, "sampled_static", expect_home_spelled=CANON_HOME)


def test_relocated_read_keeps_local_path_logic_running_on_the_local_spelling(tmp_path):
    """multiturn_grades' relpath groups, the bcbw root check and em_misalign's prefix classes compute from `Input.uri` (local); under
    relocation the groups are the same strings and the reads succeed -- the equality test above proves it for all six; here the
    plan's three named computations are pinned by name against a moved root."""
    import test_registry_adapter_multiturn_grades as T
    import _b1_fixtures as B1
    from sisteer.registry.adapters import multiturn_grades as MG
    a = str(tmp_path / "A")
    T._root(Path(a))
    seeds = B1.synthetic_seeds()
    B1.resolver(seeds, Path(a))
    b = str(tmp_path / "B")
    _relocate(a, b)
    ad_b = MG.MultiturnGradesAdapter(rs=os.path.join(b, "rs"), seeds=seeds, objects=B1.resolver(seeds, Path(b)), transfer_stats_files=("transfer_stats_full22_all.json",))
    with AB.using_identity_map(AB.IdentityMap.build({b: CANON_HOME})):
        inputs = ad_b.discover()
        groups = {ad_b.comparison_group(i) for i in inputs}
        # every store input lies under the local root and is identified canonically; a repo-resident declaration file (configs/registry/...)
        # is the checkout's own and keeps its spelling (the cluster task maps the exported tree onto the grading host checkout separately)
        store_inputs = [i for i in inputs if not i.uri.startswith(SRC_DIR[: -len("/src")])]
        assert store_inputs and all(i.uri.startswith(b) and i.identity_uri.startswith(CANON_HOME) for i in store_inputs)
        # relpath against the LOCAL rs: relative groups, never a `..`-prefixed one; the canonical identity of a relative group is itself
        assert groups and all(not g.startswith("..") and not g.startswith("/") for g in groups), groups
        assert all(AB.comparison_group_identity(g) == g for g in groups)


# --------------------------------------------------------------------------------------------
# part 2: the shard deal
# --------------------------------------------------------------------------------------------


def _plan(groups: List[str], changed: List[str], vanished: List[str]) -> RUN.GroupPlan:
    plan = RUN.GroupPlan()
    for g in groups:
        plan.groups[g] = []
    plan.changed = set(changed)
    for g in vanished:
        plan.vanished[g] = {g + "/x"}
    return plan


def test_deal_shards_hands_every_group_of_the_union_to_exactly_one_shard_including_a_vanished_one():
    plan = _plan(["g_a", "g_c", "g_d"], ["g_a", "g_c", "g_d"], ["g_b"])          # g_b: prior-only (vanished); union sorted = a b c d
    owners = {}
    for i in range(3):
        changed, vanished, n = RUN.deal_shards(plan, i, 3)
        assert n == 4
        for g in changed | vanished:
            assert g not in owners, (g, owners)
            owners[g] = i
    assert owners == {"g_a": 0, "g_b": 1, "g_c": 2, "g_d": 0}
    # a vanished group is retired by its one owner and reported as not theirs by the other two
    assert [bool(RUN.deal_shards(plan, i, 3)[1]) for i in range(3)] == [False, True, False]
    assert RUN.deal_shards(plan, 0, 1) == ({"g_a", "g_c", "g_d"}, {"g_b"}, 4)
    with pytest.raises(ValueError):
        RUN.deal_shards(plan, 3, 3)
    with pytest.raises(ValueError):
        RUN.deal_shards(plan, 0, 0)


def test_deal_shards_orders_by_canonical_identity_so_every_shard_agrees_whatever_its_local_spelling(tmp_path):
    local_a, local_b = str(tmp_path / "n1" / "home"), str(tmp_path / "n2" / "home")
    groups = [f"{{home}}/sampled/grades_sampled/d{i}" for i in range(5)]
    owners = []
    for local in (local_a, local_b):
        with AB.using_identity_map(AB.IdentityMap.build({local: CANON_HOME})):
            plan = _plan([g.format(home=local) for g in groups[:4]], [g.format(home=local) for g in groups[:4]], [groups[4].format(home=CANON_HOME)])
            owners.append({AB.comparison_group_identity(g): i for i in range(3) for g in set.union(*RUN.deal_shards(plan, i, 3)[:2])})
    assert owners[0] == owners[1] and len(owners[0]) == 5


def test_a_sharded_pass_reads_only_its_groups_and_a_reader_traversal_adapter_admits_one_shard_only(tmp_path):
    from test_registry_adapter_pass import SyntheticStoreAdapter, _setup, _store
    root, store_root = _setup(tmp_path)
    _store(store_root, "s2", "legacy_store_2", "none", [0.1, 0.9])
    _store(store_root, "s3", "legacy_store_3", "constitution_lora@aaaa", [0.7, 0.7, 0.2])
    ad = SyntheticStoreAdapter(store_root)
    seen = {}
    with DB.reading(DB.db_path(root)) as conn:
        for i in range(2):
            rep = RUN.run_pass(root, [ad], conn=conn, now=NOW, shard=(i, 2))
            seen[i] = sorted(r.group_identity for r in rep.results)
            assert rep.groups_seen == 3 and len(seen[i]) + rep.groups_not_owned == 3
    assert not set(seen[0]) & set(seen[1]) and len(seen[0]) + len(seen[1]) == 3
    ad.rng_scope = "reader_traversal"
    with DB.reading(DB.db_path(root)) as conn, pytest.raises(ValueError, match="reader_traversal"):
        RUN.run_pass(root, [ad], conn=conn, now=NOW, shard=(0, 2))
    with DB.reading(DB.db_path(root)) as conn:
        assert RUN.run_pass(root, [ad], conn=conn, now=NOW, shard=(0, 1)).groups_not_owned == 0


# --------------------------------------------------------------------------------------------
# part 3: the reader-state sidecar and the CLUSTER shard
# --------------------------------------------------------------------------------------------

from sisteer.registry import cli as CLI  # noqa: E402
from sisteer.registry import cluster_reads as CR  # noqa: E402
from sisteer.registry import reader_state as RS  # noqa: E402
from sisteer.registry.adapters import anchor_current as AC  # noqa: E402
from sisteer.registry import stores as ST  # noqa: E402
from test_registry_adapter_pass import SyntheticStoreAdapter, _setup, _store  # noqa: E402

ADAPTER = SyntheticStoreAdapter.name


def _grading_host(tmp_path, home_name: str = "home"):
    """A grading host: a registry root with the synthetic dims + roster ingested, a home with the synthetic stores at `<home>/stores`, the
    adapter over it, and the first pass folded (three accepted revisions)."""
    RF.ensure_rsync_on_path()                     # the modules importing this world by name have no autouse fixture of their own; local operands only
    root, _unused = _setup(tmp_path)
    home = tmp_path / home_name
    stores = home / "stores"
    stores.mkdir(parents=True)
    _store(str(stores), "s1", "legacy_store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4])
    _store(str(stores), "s2", "legacy_store_2", "none", [0.1, 0.9])
    _store(str(stores), "s3", "legacy_store_3", "constitution_lora@aaaa", [0.7, 0.7, 0.2])
    ad = SyntheticStoreAdapter(str(stores))
    with DB.reading(DB.db_path(root)) as conn:
        rep = RUN.run_pass(root, [ad], conn=conn, now=synth.t(2000), producer=PRODUCER)
    assert rep.groups_refreshed == 3, [r.error for r in rep.results]
    ing = H.ingest_once(root, workers=0, min_free=0)
    assert ing.refused == 0, ing.refusals
    return root, str(home), str(stores), ad


def _export(root, out, home, adapters=(ADAPTER,)):
    return RS.export(root, list(adapters), out, uri_home=home, repo_root=REPO_ROOT)


def _cluster_pass(cluster_root, ad, sidecar, now, shard=None, full=False, **kw):
    conn = RS.open_reader_state(sidecar, [ad.name])
    try:
        cluster = RUN.ClusterRead.from_sidecar(conn, {"substrate": "cluster_a", "host": "node-x", "components": {}})
        return RUN.run_pass(cluster_root, [ad], conn=conn, now=now, producer=dict(PRODUCER, substrate="cluster_a", host="node-x"), cluster=cluster,
                            shard=shard, full=full, save_cache=False, **kw)
    finally:
        conn.close()


def _published(cluster_root, adapter=ADAPTER):
    base = os.path.join(H.records_dir(cluster_root), "adapters", adapter)
    return sorted(d for d in os.listdir(base)) if os.path.isdir(base) else []


def _staged_revisions(root, adapter=ADAPTER):
    """The revision directories still staged under ``<root>/staging/adapters/<adapter>/``: [] when the adapter-wide parent is absent OR
    stands empty. The parent itself is left for the life of the root (concurrent shards of one adapter share it, so publish_staged
    no longer tidies it), so a test pins the ABSENCE OF STAGED REVISIONS, never the parent's."""
    base = os.path.join(root, "staging", "adapters", adapter)
    return sorted(os.listdir(base)) if os.path.isdir(base) else []


def _rows(root, sql, params=()):
    with DB.reading(DB.db_path(root)) as conn:
        return [tuple(r) for r in conn.execute(sql, params).fetchall()]


def _attested_generation(sha8: str = "deadbeef", **over) -> Dict:
    """One generation's verdict in a synthetic attestation record (`stores.attest_generation`'s shape, ok unless overridden)."""
    return {"ok": True, "refusal": None, "reason": "", "content_sha8": sha8, "utc": NOW, "content": {"mode": "full"}, "listing": {"entries": 3, "elapsed_s": 0.1}, **over}


def _attestation_record(window: str, generations: Dict[str, Dict]) -> Dict:
    """A synthetic operation attestation record (`store_attestation/1`) for `window` over `generations` (slug -> `_attested_generation`):
    what `attest.sbatch` writes at `<window root>/attestation.json` and the pull copies beside the shard summaries. The pull's attestation
    gate is FAIL-CLOSED, so every pull fixture that expects delivery or a replay lays one of these down."""
    refused = sorted(s for s, g in generations.items() if g.get("ok") is not True)
    return {"schema": ST.ATTESTATION_SCHEMA, "window": window, "ok": not refused, "utc": NOW, "host": "node-x", "job_id": "1", "n_generations": len(generations),
            "refused": refused, "generations": generations}


def test_reader_state_export_carries_the_adapters_state_the_dims_the_head_items_and_the_published_dirs(tmp_path):
    root, home, stores, ad = _grading_host(tmp_path)
    # one published-not-folded directory beside the folded ones, and a refused read: both are "known" to the grading host
    out = str(tmp_path / "rs.sqlite")
    rep = _export(root, out, home)
    assert os.path.getsize(out) == rep.bytes > 0 and rep.ledger_head and rep.uri_home == home
    conn = RS.open_reader_state(out, [ADAPTER])
    try:
        m = RS.meta(conn)
        assert m["schema"] == RS.SCHEMA and m["adapters"] == [ADAPTER] and m["uri_home"] == home and m["schema_version"] and m["db_inode"]
        assert rep.rows["source_revision"] == 3 and rep.rows["source_watermark"] == 3 and rep.rows["source_status"] == 3 and rep.rows["observation"] > 0
        assert rep.rows["source_state"] == 3 and rep.rows["published_dirs"] == 3 and rep.rows["item_result"] == 8       # 3 + 2 + 3 items of the heads
        assert [c[1] for c in conn.execute("PRAGMA table_info(item_result)")] == list(R.ITEM_MEMBERSHIP_KEY) + ["source_revision_id"]
        for t in RS.DIMENSION_TABLES:
            assert conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] == _rows(root, f"SELECT COUNT(*) FROM {t}")[0][0], t
        assert RS.known_dirs(conn) == {r[0] for r in _rows(root, "SELECT revision_id FROM source_revision")}
        assert conn.execute("SELECT kind FROM published_dirs").fetchall()[0][0] == "revision"
        # the anchor relation ships for EVERY export (reader_state/2) -- its shape is pinned below
        assert set(RS.ANCHOR_TABLES) <= {t[0] for t in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        # the runner's own queries run unchanged against the sidecar
        marks = RUN._watermarks(conn, ADAPTER)
        assert len(marks) == 3 and all(mk.processed for mk in marks.values())
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("INSERT INTO published_dirs VALUES ('x','y','z')")
    finally:
        conn.close()
    # a sidecar built for other adapters refuses
    other = str(tmp_path / "other.sqlite")
    _export(root, other, home, adapters=("relu_paired",))
    with pytest.raises(RS.ReaderStateError, match="not for"):
        RS.open_reader_state(other, [ADAPTER])
    with pytest.raises(RS.ReaderStateError):
        RS.open_reader_state(str(tmp_path / "missing.sqlite"))
    with pytest.raises(RS.ReaderStateError, match="at least one adapter"):
        RS.export(root, [], str(tmp_path / "none.sqlite"))


def test_cluster_shard_refuses_known_ids_publishes_exactly_one_new_id_for_a_change_and_the_grading_host_reactivates_a_revert(tmp_path):
    root, home, stores, ad = _grading_host(tmp_path)
    sidecar = str(tmp_path / "rs.sqlite")
    _export(root, sidecar, home)
    cluster_root = _cluster_root(tmp_path)
    # unchanged inputs: the sidecar's watermarks match every input -> nothing is planned, nothing published, nothing refused
    rep = _cluster_pass(cluster_root, ad, sidecar, synth.t(3000))
    assert rep.groups_seen == 3 and rep.results == [] and rep.inputs_changed == 0
    assert not os.path.isdir(os.path.join(H.records_dir(cluster_root), "adapters"))
    # --full plans every group: the three minted ids are the grading host's accepted heads -> refused by name, nothing published, no `_refused/` file
    rep0 = _cluster_pass(cluster_root, ad, sidecar, synth.t(3050), full=True)
    assert rep0.groups_refused_known == 3 and rep0.groups_refreshed == rep0.groups_refused == 0
    assert all(r.refused_known and "known to the grading host" in r.error and "source_revision row, accepted" in r.error for r in rep0.results)
    assert not os.path.isdir(os.path.join(H.records_dir(cluster_root), "adapters")) and not any(r.files for r in rep0.results)
    # a changed input: exactly ONE new id, published in the cluster spool with the cluster's producer and the read_copy on the marker
    old_rev = next(r.revision_id for r in rep0.results if r.group == "s2")
    s2 = _store(str(stores), "s2", "legacy_store_2", "none", [0.1, 0.9, 0.5])
    rep2 = _cluster_pass(cluster_root, ad, sidecar, synth.t(3100))
    assert rep2.groups_refused_known == 0 and rep2.groups_refreshed == 1 and rep2.inputs_changed == 1 and len(rep2.results) == 1
    new = _published(cluster_root)
    assert len(new) == 1 and new[0] != old_rev and new[0] not in _rows(root, "SELECT revision_id FROM source_revision")
    marker = _marker(os.path.join(H.records_dir(cluster_root), "adapters", ADAPTER, new[0]))
    assert marker["producer"]["substrate"] == "cluster_a" and marker["producer"]["host"] == "node-x"
    assert marker["payload"]["extra"]["read_copy"]["substrate"] == "cluster_a" and marker["payload"]["inputs_json"] == {s2: R.file_sha256(s2)}
    # a second shard pass over the same window spool: the published id is left alone (not written twice)
    rep3 = _cluster_pass(cluster_root, ad, sidecar, synth.t(3200))
    assert rep3.groups_held == 1 and "already published in this window's spool" in next(r.error for r in rep3.results if r.held)
    assert rep3.groups_refused_known == 0 and _published(cluster_root) == new
    # --full on the cluster: every group planned, the known ones refused identically, no re-hash changes the outcome
    rep4 = _cluster_pass(cluster_root, ad, sidecar, synth.t(3300), full=True)
    assert rep4.groups_refused_known == 2 and rep4.groups_held == 1 and rep4.inputs_hashed == 3
    # the grading host folds the cluster's revision (the transport delivers it into records/), then s2 is REVERTED: its old id is accepted
    # (superseded) -> the cluster refuses it as known; the grading host pass re-activates it from the committed bytes
    shutil.copytree(os.path.join(H.records_dir(cluster_root), "adapters", ADAPTER, new[0]), os.path.join(H.records_dir(root), "adapters", ADAPTER, new[0]))
    ing = H.ingest_once(root, workers=0, min_free=0)
    assert ing.refused == 0 and _rows(root, "SELECT accepted FROM source_revision WHERE revision_id = ?", (new[0],)) == [(1,)]
    _store(str(stores), "s2", "legacy_store_2", "none", [0.1, 0.9])
    sidecar2 = str(tmp_path / "rs2.sqlite")
    _export(root, sidecar2, home)
    cluster_root2 = str(tmp_path / "cluster_root2")
    H.ensure_layout(cluster_root2)
    rep5 = _cluster_pass(cluster_root2, ad, sidecar2, synth.t(3400))
    s2_res = next(r for r in rep5.results if r.group == "s2")
    assert s2_res.refused_known and s2_res.revision_id == old_rev and "source_revision row, accepted" in s2_res.error and not _published(cluster_root2)
    with DB.reading(DB.db_path(root)) as conn:
        rep6 = RUN.run_pass(root, [ad], conn=conn, now=synth.t(3500), producer=PRODUCER)
    s2_dev = next(r for r in rep6.results if r.group == "s2")
    assert s2_dev.accepted and s2_dev.revision_id == old_rev and any("/_activations/" in f for f in s2_dev.files), s2_dev
    ing2 = H.ingest_once(root, workers=0, min_free=0)
    assert ing2.refused == 0
    head = _rows(root, "SELECT revision_id FROM source_revision WHERE source_id = ? AND accepted = 1 ORDER BY observed_utc DESC LIMIT 1", (f"{ADAPTER}:s2",))
    assert head == [(old_rev,)]


def test_cluster_shard_reads_an_emit_time_refused_id_fresh_and_refuses_a_published_not_folded_directory_by_name(tmp_path):
    """A revision the grading host REFUSED AT EMIT TIME (its `source_revision` row the fold of the `_refused/<rev>/` record: accepted = 0,
    the reader's words, no directory) is NOT known to the shard -- nothing exists to re-activate from -- so a retry READS it fresh and
    publishes it; a published, not-yet-folded directory is still refused by name (pending on the grading host). Before this the refused id
    came back refused-known ('source_revision row, pending'), the pull's re-activation by id found no directory, and the adapter was
    refused."""
    root, home, stores, ad = _grading_host(tmp_path)
    # a refused read on the grading host leaves a refused record (an accepted = 0 row, nothing committed) for s1; a fresh s3 read published but not folded
    ad.broken.add(os.path.join(stores, "s1", "store.json"))
    _store(str(stores), "s1", "legacy_store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4, 0.9])
    _store(str(stores), "s3", "legacy_store_3", "constitution_lora@aaaa", [0.7])
    with DB.reading(DB.db_path(root)) as conn:
        rep = RUN.run_pass(root, [ad], conn=conn, now=synth.t(2100), producer=PRODUCER)
    assert rep.groups_refused == 1 and rep.groups_refreshed == 1
    s1_rev = next(r.revision_id for r in rep.results if r.group == "s1")
    s3_rev = next(r.revision_id for r in rep.results if r.group == "s3")
    # fold ONLY the refused marker (the s3 directory stays published, not folded): move s3's dir aside, ingest, move it back
    s3_dir = os.path.join(H.records_dir(root), "adapters", ADAPTER, s3_rev)
    shutil.move(s3_dir, str(tmp_path / "aside"))
    H.ingest_once(root, workers=0, min_free=0)
    shutil.move(str(tmp_path / "aside"), s3_dir)
    assert _rows(root, "SELECT accepted, record_uri LIKE '%/_refused/%' FROM source_revision WHERE revision_id = ?", (s1_rev,)) == [(0, 1)]
    assert _rows(root, "SELECT COUNT(*) FROM source_revision WHERE revision_id = ?", (s3_rev,)) == [(0,)]
    assert not os.path.isdir(os.path.join(H.records_dir(root), "adapters", ADAPTER, s1_rev))
    ad.broken.clear()
    sidecar = str(tmp_path / "rs.sqlite")
    rep_x = _export(root, sidecar, home)
    assert rep_x.rows["published_dirs"] == 5      # 3 first-pass dirs + the refused record + the published-not-folded dir
    cluster_root = _cluster_root(tmp_path)
    # s1 stands refused (its refused watermark carries the bytes it read): only a retry plans it -- and its id, refused at emit time and never
    # committed, is NOT known, so the shard reads it fresh and publishes the very id the refused read computed; s3's bytes differ from its
    # folded watermark, so it is planned -- and its minted id is the published-not-folded directory's: known, pending on the grading host
    rep2 = _cluster_pass(cluster_root, ad, sidecar, synth.t(3000), retry_refused=True)
    by = {r.group: r for r in rep2.results}
    assert set(by) == {"s1", "s3"}
    assert by["s1"].accepted and by["s1"].revision_id == s1_rev and not by["s1"].refused_known and by["s1"].files, by["s1"].error
    assert by["s3"].refused_known and by["s3"].revision_id == s3_rev and "a published directory on the grading host" in by["s3"].error
    assert rep2.groups_refused_known == 1 and rep2.groups_refreshed == 1 and _published(cluster_root) == [s1_rev]


def test_cluster_shard_refuses_a_vanished_group_by_name_one_shard_owns_it_and_the_grading_host_retires_it_with_the_state_paths(tmp_path):
    """The dropped-input rule extended to the vanished path: on a STAGED copy a group whose every input is absent is an incomplete copy
    until the grading host says otherwise, so its one owner (the deal) REFUSES it -- nothing published, the missing input named -- and the
    grading host's own pass (the window's fallback) commits the retirement: the empty revision, the `input_vanished` withdrawals, the
    `source_state found = 0` rows."""
    root, home, stores, ad = _grading_host(tmp_path)
    s3_uri = os.path.join(stores, "s3", "store.json")
    assert _rows(root, "SELECT source_id, json_path FROM source_state WHERE input_uri = ?", (s3_uri,)) == [(f"{ADAPTER}:all", "$.object")]
    shutil.rmtree(os.path.join(stores, "s3"))
    sidecar = str(tmp_path / "rs.sqlite")
    _export(root, sidecar, home)
    owners = []
    for i in range(3):
        cluster_root = _cluster_root(tmp_path, f"cluster_root_{i}")
        rep = _cluster_pass(cluster_root, ad, sidecar, synth.t(3000), shard=(i, 3))
        assert rep.groups_seen == 3 and not _published(cluster_root), rep
        assert not os.path.isdir(os.path.join(H.records_dir(cluster_root), "adapters"))       # no revision, no marker, no _refused file
        if rep.results:
            owners.append((i, rep))
    assert len(owners) == 1, owners            # exactly one shard owns the vanished group; the other two report it as not theirs
    i, rep = owners[0]
    (res,) = rep.results
    assert not res.accepted and res.inputs_dropped and res.files == [] and rep.groups_refused == 1 and rep.groups_inputs_dropped == 1
    assert res.error.startswith(f"{ADAPTER}:s3: {RUN.STAGED_COPY_INCOMPLETE_KEY}: every input of the group (1) is absent from this read on the cluster")
    assert "a retirement is the grading host's" in res.error and f"{s3_uri} [no staged component covers it]" in res.error
    # the grading host retires it on its own pass: the empty revision with the withdrawals and the shipped state paths
    with DB.reading(DB.db_path(root)) as conn:
        rep_d = RUN.run_pass(root, [ad], conn=conn, now=synth.t(3050), producer=PRODUCER)
    res_d = next(r for r in rep_d.results if r.group == "s3")
    assert res_d.accepted and res_d.withdrawals > 0
    recs = {}
    d = os.path.join(H.records_dir(root), "adapters", ADAPTER, res_d.revision_id)
    for fn in os.listdir(d):
        for _, rec, _ in R.read_record_file(os.path.join(d, fn)):
            recs.setdefault(rec["record_type"], []).append(rec["payload"])
    marker = recs["source_revision"][0]
    assert marker["inputs_json"] == {} and marker["source_id"] == f"{ADAPTER}:s3" and marker["extra"][H.DECLARED_RUN_DROPS_KEY]["run_ids"] == "*"
    assert all(w["reason"] == "input_vanished" for w in recs["withdrawal"]) and recs["withdrawal"]
    assert recs["source_state"] == [{"source_id": f"{ADAPTER}:all", "input_uri": s3_uri, "json_path": "$.object", "found": 0, "path_resolves": 0,
                                     "current_members_json": [], "observed_utc": synth.t(3050), "input_sha256": None}]
    assert recs["source_watermark"][0]["last_error"] == RUN.VANISHED_PREFIX + s3_uri
    assert recs["source_status"][0]["ok"] == 0 and s3_uri in recs["source_status"][0]["extra"]["missing"]
    # the retirement folded, the empty revision IS the head: the cluster has nothing to refuse or retire
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    sidecar2 = str(tmp_path / "rs2.sqlite")
    _export(root, sidecar2, home)
    cluster_root = _cluster_root(tmp_path, "cluster_root_again")
    rep2 = _cluster_pass(cluster_root, ad, sidecar2, synth.t(3100), shard=(i, 3))
    assert rep2.results == [] and rep2.groups_refused == 0 and not _published(cluster_root)


class TwoFileStoreAdapter(SyntheticStoreAdapter):
    """The synthetic store adapter over stores that may carry a `manifest.json` beside `store.json`: both are INPUTS of the store's group
    (the live shape of the moved `overrefusal_items` group: two inputs), while the read and the state rows read `store.json` alone."""

    def discover(self) -> List[AB.Input]:
        out = []
        for d in sorted(os.listdir(self.roots[0])):
            for fn in ("store.json", "manifest.json"):
                p = os.path.join(self.roots[0], d, fn)
                if os.path.lexists(p):
                    out.append(AB.check_input(p, role="report", allow_symlink=self.allow_symlinks))
        return out

    def read(self, ctx, group, inputs):
        yield from super().read(ctx, group, [i for i in inputs if os.path.basename(i.uri) == "store.json"])

    def source_states(self, ctx, inputs):
        return super().source_states(ctx, [i for i in inputs if os.path.basename(i.uri) == "store.json"])


def _records_by_type(rev_dir: str) -> Dict[str, List[Dict]]:
    """Every record payload of a revision directory, by record type."""
    recs: Dict[str, List[Dict]] = {}
    for fn in sorted(os.listdir(rev_dir)):
        for _, rec, _ in R.read_record_file(os.path.join(rev_dir, fn)):
            recs.setdefault(rec["record_type"], []).append(rec["payload"])
    return recs


def _declared_retirement_window(tmp_path, ad, sidecar, now, name="wroot", shards=3):
    """One window's K shards over `sidecar`, publishing into ONE spool -- the window root's `root/`, as the cluster lays it out -- each
    writing its shard summary where the pull reads it: `(window root, spool root, [(i, report)] of the shards that reported anything)`."""
    wroot = tmp_path / name
    cluster_root = _cluster_root(tmp_path, f"{name}/root")
    producer = dict(PRODUCER, substrate="cluster_a", host="node-x")
    reporting = []
    for i in range(shards):
        rep = _cluster_pass(cluster_root, ad, sidecar, now, shard=(i, shards))
        CR.write_shard_summary(CR.summary_path(str(wroot), ADAPTER, i, shards), rep, rep.as_dict(), [ADAPTER], (i, shards), cluster_root, producer)
        if rep.results:
            reporting.append((i, rep))
    return wroot, cluster_root, reporting


def test_cluster_shard_commits_a_declared_retirement_the_pull_delivers_its_marker_only_directory_and_the_grading_host_folds_it(tmp_path):
    """registry-hardening 3a, defect 1: a vanished group whose EVERY input the adapter DECLARES retired (`inputs_retired`, here a glob) is no
    incomplete copy -- a deliberate retirement is declared, never inferred from the read -- so its one owning shard commits the retirement
    exactly as the grading host path does (the empty accepted revision, the `input_vanished` withdrawals, `source_state found = 0`, the retired
    watermark), with the declaration on the marker; the other shards report nothing; the shard summary lists it under `published`, nothing
    under `refused`; the pull delivers the marker-only directory like any other revision (pass 1 brings no batch, pass 2 the marker); merged
    but not yet folded it is KNOWN to the next shard (refused by name, pending; S11); folded, the empty revision is the head and a later
    window neither re-publishes nor refuses it. Before this every window refused `overrefusal_items`' two moved groups identically."""
    root, home, stores, ad = _grading_host(tmp_path)
    s3_uri = os.path.join(stores, "s3", "store.json")
    shutil.rmtree(os.path.join(stores, "s3"))
    pattern = os.path.join(stores, "s3", "*.json")
    ad.inputs_retired = [pattern]                 # the producer moved the store; the adapter DECLARES its inputs retired
    sidecar = str(tmp_path / "rs.sqlite")
    _export(root, sidecar, home)
    wroot, cluster_root, reporting = _declared_retirement_window(tmp_path, ad, sidecar, synth.t(3000))
    assert len(reporting) == 1, reporting          # exactly one shard owns the vanished group; the other two report it as not theirs
    i, rep = reporting[0]
    (res,) = rep.results
    assert res.accepted and res.withdrawals > 0 and not res.inputs_dropped and not res.refused_known, res
    assert rep.groups_seen == 3 and rep.groups_refused == 0 and rep.groups_inputs_dropped == 0 and rep.withdrawals == res.withdrawals
    rev = res.revision_id
    marker_name = R.record_file_name(1, "source_revision")
    rev_dir_abs = os.path.join(H.records_dir(cluster_root), "adapters", ADAPTER, rev)
    assert _published(cluster_root) == [rev] and os.listdir(rev_dir_abs) == [marker_name] and res.files == [os.path.join(rev_dir_abs, marker_name)]
    recs = _records_by_type(rev_dir_abs)
    marker = recs["source_revision"][0]
    assert marker["inputs_json"] == {} and marker["accepted"] == 1 and marker["source_id"] == f"{ADAPTER}:s3"
    assert marker["extra"][RUN.RETIREMENT_DECLARED_KEY] == {"declared_by": "adapter:inputs_retired", "patterns": [pattern], "inputs": [s3_uri]}
    drops = marker["extra"][H.DECLARED_RUN_DROPS_KEY]
    assert drops["run_ids"] == "*" and drops["declared_by"] == ["runner:input_vanished", "runner:inputs_retired"] and "inputs_retired: 1 input(s)" in drops["reason"]
    assert marker["extra"]["read_copy"] == {"substrate": "cluster_a", "host": "node-x", "components": {}}
    assert recs["withdrawal"] and all(w["reason"] == "input_vanished" for w in recs["withdrawal"])
    assert recs["source_state"] == [{"source_id": f"{ADAPTER}:all", "input_uri": s3_uri, "json_path": "$.object", "found": 0, "path_resolves": 0,
                                     "current_members_json": [], "observed_utc": synth.t(3000), "input_sha256": None}]
    assert recs["source_watermark"][0]["last_error"] == RUN.VANISHED_PREFIX + s3_uri
    status = recs["source_status"][0]
    assert status["ok"] == 0 and s3_uri in status["extra"]["missing"] and "declared by the adapter's inputs_retired (1 pattern(s))" in status["detail"]
    # the shard summary: published (the marker named), nothing refused
    rev_dir = f"{ADAPTER}/{rev}"
    doc = json.load(open(CR.summary_path(str(wroot), ADAPTER, i, 3)))
    assert doc["ok"] is True and doc["refused"] == [] and doc["refused_known"] == [] and list(doc["published"]) == [rev_dir]
    assert doc["published"][rev_dir]["marker"] == marker_name and set(doc["published"][rev_dir]["files"]) == {marker_name}
    # the pull delivers the marker-only directory like any other revision: pass 1 (no batch to bring), pass 2 (the marker), one rename
    CR._write_json(str(wroot / ST.ATTESTATION_FILE), _attestation_record("w1", {"synthetic_stores": _attested_generation("deadbeef")}))
    plan = {"window": "w1", "host": None, "window_root": str(wroot), "adapters": {ADAPTER: {"route": "cluster", "shards": 3, "pins": {"synthetic_stores": "deadbeef"}}},
            "tasks": [{"adapter": ADAPTER}], "stages": {"revalidate_post": {"stale": []}}, "refusals": {}}
    pulled = CR.pull_window(root, plan, str(tmp_path / "expect"), source=str(wroot))
    assert pulled["rsync"] == {"summaries": 0, "pass1": 0, "pass2": 0} and pulled["adapters_refused"] == {} and not pulled["quarantined"], pulled
    assert pulled["pulled"][rev_dir]["state"] == "complete" and not pulled["refused_known_ids"].get(ADAPTER), pulled
    inbox_dir = os.path.join(H.inbox_dir(root), CR.INBOX_NAME, "adapters", rev_dir)
    assert os.listdir(inbox_dir) == [marker_name]
    assert [status_ for _, status_ in H.scan_inbox(root)] == ["merged"]
    # merged, not folded yet: the next export lists the directory, so the owning shard refuses the retirement id as KNOWN (its fold pending on
    # the grading host) and publishes nothing -- a cluster read never mints an inbox conflict (S11)
    sidecar_mid = str(tmp_path / "rs_mid.sqlite")
    _export(root, sidecar_mid, home)
    rep_mid = _cluster_pass(_cluster_root(tmp_path, "cluster_root_mid"), ad, sidecar_mid, synth.t(3040), shard=(i, 3))
    (res_mid,) = rep_mid.results
    assert res_mid.refused_known and res_mid.activation_pending and not res_mid.accepted and res_mid.revision_id == rev and res_mid.files == []
    assert rep_mid.groups_refused_known == 1 and rep_mid.groups_refused == 0 and rep_mid.activations_pending == 1
    assert f"known to the grading host: retiring revision {rev} of {ADAPTER}:s3" in res_mid.error and "not folded yet" in res_mid.error
    # the pull's half on that state: the grading host holds the marker on its log unfolded, so the re-activation
    # by id is HELD -- the fold is pending -- never a refusal of the adapter: a retirement marker has no input facts, and the revision's state
    # is resolved before any are demanded (before this the by-id path refused every retirement marker as "inputs_json is empty")
    report_mid = {"refused_known_ids": {ADAPTER: [rev]}, "adapters_refused": {}}
    with DB.reading(DB.db_path(root)) as conn:
        out_mid = CR.reactivate_refused_known(root, plan, report_mid, conn=conn, now=synth.t(3045), adapter_of=lambda name: ad)
    assert out_mid[ADAPTER][rev]["state"] == "held" and "not folded yet" in out_mid[ADAPTER][rev]["error"], out_mid
    assert report_mid["adapters_refused"] == {} and RUN.complete_bundles(root, ADAPTER, rev) == []
    # the harvester folds it: the empty revision is the head, its keys withdrawn with input_vanished, the state row found = 0, the watermark retired
    ing = H.ingest_once(root, workers=0, min_free=0)
    assert ing.refused == 0 and ing.blocked == 0, (ing.refusals, ing.blocked)
    assert _rows(root, "SELECT revision_id FROM source_revision WHERE source_id = ? AND accepted = 1 ORDER BY observed_utc DESC LIMIT 1", (f"{ADAPTER}:s3",)) == [(rev,)]
    assert _rows(root, "SELECT DISTINCT reason FROM withdrawal WHERE to_revision_id = ?", (rev,)) == [("input_vanished",)]
    assert _rows(root, "SELECT COUNT(*) FROM observation WHERE source_id = ? AND current = 1", (f"{ADAPTER}:s3",)) == [(0,)]
    assert _rows(root, "SELECT found FROM source_state WHERE input_uri = ?", (s3_uri,)) == [(0,)]
    assert _rows(root, "SELECT last_error FROM source_watermark WHERE source_id = ? AND input_uri = ?", (f"{ADAPTER}:s3", s3_uri)) == [(RUN.VANISHED_PREFIX + s3_uri,)]
    assert _rows(root, "SELECT json_extract(extra_json, '$.retirement_declared.patterns') FROM source_revision WHERE revision_id = ?", (rev,)) == [(json.dumps([pattern]),)]
    # a second window after the fold neither re-publishes nor refuses it: the empty revision IS the head (prev_rev == rev)
    sidecar2 = str(tmp_path / "rs2.sqlite")
    _export(root, sidecar2, home)
    cluster_root2 = _cluster_root(tmp_path, "cluster_root_again")
    rep2 = _cluster_pass(cluster_root2, ad, sidecar2, synth.t(3100), shard=(i, 3))
    assert rep2.results == [] and rep2.groups_refused == 0 and rep2.groups_refused_known == 0 and not _published(cluster_root2)


def test_a_retired_group_that_came_back_and_vanished_again_is_retired_again_by_id_from_the_pull(tmp_path):
    """A retired group's store comes back byte-identical (its accepted revision re-activates and the measurements are current again) and
    then vanishes once more under the same emission version, so the shard mints the SAME empty id -- KNOWN to the grading host (accepted,
    superseded) -- refuses it by name as retiring and publishes nothing. The pull's re-activation by id then re-publishes the retirement
    from the DATABASE, never from the marker's (empty) input facts: a marker-only activation bundle under the committed id carrying
    TODAY's withdrawals of the head's keys, the retired watermark, the found = 0 state row and the status row, with the declaration
    re-derived; the fold makes the empty revision the head again, and the next window has nothing to refuse or retire. Before this the
    by-id path refused every retirement marker ("inputs_json is empty"), the pull recorded the ADAPTER refused, and the outgoing
    measurements stayed current window after window."""
    root, home, stores, ad = _grading_host(tmp_path)
    sid3 = f"{ADAPTER}:s3"
    s3_dir, s3_uri = os.path.join(stores, "s3"), os.path.join(stores, "s3", "store.json")
    ((s3_rev,),) = _rows(root, "SELECT revision_id FROM source_revision WHERE source_id = ? AND accepted = 1", (sid3,))
    s3_bytes = open(s3_uri, "rb").read()
    pattern = os.path.join(stores, "s3", "*.json")
    ad.inputs_retired = [pattern]
    head = lambda: _rows(root, "SELECT revision_id FROM source_revision WHERE source_id = ? AND accepted = 1 ORDER BY observed_utc DESC LIMIT 1", (sid3,))
    current = lambda: _rows(root, "SELECT COUNT(*) FROM observation WHERE source_id = ? AND current = 1", (sid3,))[0][0]
    # 1. the first retirement (the grading host path here; the shard path is the test above): the empty revision `rev` is the head
    shutil.rmtree(s3_dir)
    with DB.reading(DB.db_path(root)) as conn:
        rep1 = RUN.run_pass(root, [ad], conn=conn, now=synth.t(3000), producer=PRODUCER)
    rev = next(r.revision_id for r in rep1.results if r.group == "s3")
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0 and head() == [(rev,)] and current() == 0
    # 2. the store comes back byte-identical: its accepted revision re-activates (a bundle), the measurements are current again
    os.makedirs(s3_dir)
    with open(s3_uri, "wb") as fh:
        fh.write(s3_bytes)
    with DB.reading(DB.db_path(root)) as conn:
        rep2 = RUN.run_pass(root, [ad], conn=conn, now=synth.t(3100), producer=PRODUCER)
    res2 = next(r for r in rep2.results if r.group == "s3")
    assert res2.accepted and res2.reactivated and res2.revision_id == s3_rev, res2.error
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0 and head() == [(s3_rev,)] and current() > 0
    assert _rows(root, "SELECT last_error FROM source_watermark WHERE source_id = ? AND input_uri = ?", (sid3, s3_uri)) == [(None,)]
    assert _rows(root, "SELECT found FROM source_state WHERE input_uri = ?", (s3_uri,)) == [(1,)]
    # 3. it vanishes again: the shard mints the same empty id, KNOWN (accepted, superseded) -> refused by name as retiring, nothing published
    shutil.rmtree(s3_dir)
    sidecar = str(tmp_path / "rs.sqlite")
    _export(root, sidecar, home)
    cluster_root = _cluster_root(tmp_path, "cluster_root_again")
    reporting = [(i, _cluster_pass(cluster_root, ad, sidecar, synth.t(3200), shard=(i, 3))) for i in range(3)]
    ((owner, rep3),) = [(i, r) for i, r in reporting if r.results]
    (res3,) = rep3.results
    assert res3.refused_known and not res3.activation_pending and not res3.accepted and res3.revision_id == rev and res3.files == []
    assert rep3.groups_refused_known == 1 and rep3.groups_refused == 0 and not _published(cluster_root)
    assert f"known to the grading host: retiring revision {rev} of {sid3} (source_revision row, accepted)" in res3.error
    # the shard's CLAIM rides the refusal (round 1 of the review): the head it compared against, the vanished inputs it saw declared, the
    # covering patterns -- on the result and in the shard summary's refused_known entry
    claim = {"outgoing_revision_id": s3_rev, "inputs": [s3_uri], "patterns": [pattern]}
    assert res3.retirement == claim
    doc3 = CR.shard_summary(rep3, rep3.as_dict(), [ADAPTER], (owner, 3), cluster_root, dict(PRODUCER, substrate="cluster_a", host="node-x"))
    assert doc3["refused_known"] == [{"adapter": ADAPTER, "group": "s3", "revision_id": rev, "why": res3.error, "retirement": claim}]
    # 4. the pull's half: the retirement is re-activated by id from the database -- a marker-only bundle under the committed id with TODAY's
    # withdrawals (from the re-activated head), the retired watermark, found = 0, the status row, the declaration from the claim -- once the
    # claim is checked against the live head (the same head, every live input declared)
    report = {"refused_known_ids": {ADAPTER: [rev]}, "refused_known_claims": {ADAPTER: {rev: claim}}, "adapters_refused": {}}
    with DB.reading(DB.db_path(root)) as conn:
        out = CR.reactivate_refused_known(root, {}, report, conn=conn, now=synth.t(3300), adapter_of=lambda name: ad)
    assert out[ADAPTER][rev]["state"] == "reactivated" and out[ADAPTER][rev]["withdrawals"] > 0 and report["adapters_refused"] == {}, out
    bundles = RUN.complete_bundles(root, ADAPTER, rev)
    assert len(bundles) == 1 and os.listdir(os.path.dirname(bundles[0][1])) == [os.path.basename(bundles[0][1])]
    recs = _records_by_type(os.path.dirname(bundles[0][1]))
    marker = recs["source_revision"][0]
    assert marker["inputs_json"] == {} and marker["revision_id"] == rev and marker["accepted"] == 1 and marker["observed_utc"] == synth.t(3300)
    assert marker["extra"]["reactivated_from_committed_bytes"]["inputs"] == 0 and marker["extra"][H.DECLARED_RUN_DROPS_KEY]["run_ids"] == "*"
    assert marker["extra"]["reactivated_from_committed_bytes"]["retirement_claim"] == {"outgoing_revision_id": s3_rev, "inputs": 1, "patterns": 1}
    assert marker["extra"][RUN.RETIREMENT_DECLARED_KEY] == {"declared_by": "adapter:inputs_retired", "patterns": [pattern], "inputs": [s3_uri]}
    assert recs["withdrawal"] and all(w["reason"] == "input_vanished" and w["from_revision_id"] == s3_rev and w["to_revision_id"] == rev for w in recs["withdrawal"])
    assert [w["last_error"] for w in recs["source_watermark"]] == [RUN.VANISHED_PREFIX + s3_uri]
    assert recs["source_state"] == [{"source_id": f"{ADAPTER}:all", "input_uri": s3_uri, "json_path": "$.object", "found": 0, "path_resolves": 0,
                                     "current_members_json": [], "observed_utc": synth.t(3300), "input_sha256": None}]
    assert recs["source_status"][0]["ok"] == 0 and s3_uri in recs["source_status"][0]["extra"]["missing"]
    # 5. the fold: the empty revision is the head again, the keys withdrawn, the watermark retired, the state row found = 0
    ing = H.ingest_once(root, workers=0, min_free=0)
    assert ing.refused == 0 and ing.blocked == 0, (ing.refusals, ing.blocked)
    assert head() == [(rev,)] and current() == 0
    assert _rows(root, "SELECT last_error FROM source_watermark WHERE source_id = ? AND input_uri = ?", (sid3, s3_uri)) == [(RUN.VANISHED_PREFIX + s3_uri,)]
    assert _rows(root, "SELECT found FROM source_state WHERE input_uri = ?", (s3_uri,)) == [(0,)]
    # 6. the next window: nothing to refuse or retire (prev_rev == rev), the bundle on the grading host's disk notwithstanding
    sidecar2 = str(tmp_path / "rs2.sqlite")
    _export(root, sidecar2, home)
    cluster_root2 = _cluster_root(tmp_path, "cluster_root_quiet")
    rep4 = _cluster_pass(cluster_root2, ad, sidecar2, synth.t(3400), shard=(owner, 3))
    assert rep4.results == [] and rep4.groups_refused == 0 and rep4.groups_refused_known == 0 and not _published(cluster_root2)


def _retirement_pull(root, tmp_path, wroot, window, expect, now, ad):
    """The pull of a window whose shards reported a refused-known RETIREMENT, as the CLI runs it: the attestation record laid, `pull_window`
    (the refused-known gate keeps the claim), then `reactivate_refused_known` over its report: `(pull report, reactivations)`."""
    CR._write_json(str(wroot / ST.ATTESTATION_FILE), _attestation_record(window, {"synthetic_stores": _attested_generation("deadbeef")}))
    plan = {"window": window, "host": None, "window_root": str(wroot), "adapters": {ADAPTER: {"route": "cluster", "shards": 3, "pins": {"synthetic_stores": "deadbeef"}}},
            "tasks": [{"adapter": ADAPTER}], "stages": {"revalidate_post": {"stale": []}}, "refusals": {}}
    pulled = CR.pull_window(root, plan, str(tmp_path / expect), source=str(wroot))
    # nothing published: the summaries and pass 1 transfer nothing, and pass 2 (the markers) does not run at all
    assert pulled["rsync"]["summaries"] == 0 and pulled["rsync"]["pass1"] == 0 and pulled["rsync"].get("pass2", 0) == 0, pulled["rsync"]
    assert pulled["adapters_refused"] == {} and not pulled["quarantined"] and pulled["pulled"] == {}, pulled
    with DB.reading(DB.db_path(root)) as conn:
        out = CR.reactivate_refused_known(root, plan, pulled, conn=conn, now=now, adapter_of=lambda name: ad)
    return pulled, out


def test_a_refused_known_retirement_is_withheld_when_the_head_moved_under_it_and_re_activated_once_a_shard_declares_against_the_new_head(tmp_path):
    """Round 1 of the review of registry-hardening 3a (HIGH): the pull's lock serializes pulls, not every writer of the database. A shard
    refuses a known empty id as retiring against the head its sidecar carried; before the pull another writer -- here the operator's pass
    on the grading host, in production an earlier window's late fold -- accepts a NEWER revision of the source with an ADDITIONAL input. The
    re-activation used to take the live head's keys and inventory and write an accepted empty revision withdrawing the newer measurements
    and retiring the newer input, which no shard ever saw or declared retired (`_retirement_records` computed `undeclared` and ignored it).
    Now the shard's CLAIM (the head it compared against, the inputs it saw declared, the patterns) travels with the refused-known result
    through the summary and the pull's gate (`refused_known_claims`), and the re-activation is WITHHELD by name unless the claim is exactly
    the live state: no bundle, no withdrawal, the newer head and its input untouched, the adapter not refused (the next window's shard
    judges the group against today's head). Also withheld: a claim naming the live head but not every live input, and no claim at all (an
    older shard's summary). The next window's shard, declaring against the new head (the glob covers the added input too), mints the same
    empty id with a matching claim, and the pull re-activates: the withdrawals name the new head, both inputs retire, the fold makes the
    empty revision the head."""
    root, home, stores, ad = _grading_host(tmp_path)
    sid3 = f"{ADAPTER}:s3"
    s3_dir, s3_uri, manifest = os.path.join(stores, "s3"), os.path.join(stores, "s3", "store.json"), os.path.join(stores, "s3", "manifest.json")
    ((s3_rev,),) = _rows(root, "SELECT revision_id FROM source_revision WHERE source_id = ? AND accepted = 1", (sid3,))
    s3_bytes = open(s3_uri, "rb").read()
    pattern = os.path.join(stores, "s3", "*.json")
    ad.inputs_retired = [pattern]
    head = lambda: _rows(root, "SELECT revision_id FROM source_revision WHERE source_id = ? AND accepted = 1 ORDER BY observed_utc DESC LIMIT 1", (sid3,))
    current = lambda: _rows(root, "SELECT COUNT(*) FROM observation WHERE source_id = ? AND current = 1", (sid3,))[0][0]
    # the empty id becomes KNOWN: retired on the grading host, the store back byte-identical, its revision re-activated (the head is s3_rev again)
    shutil.rmtree(s3_dir)
    with DB.reading(DB.db_path(root)) as conn:
        rep1 = RUN.run_pass(root, [ad], conn=conn, now=synth.t(3000), producer=PRODUCER)
    rev = next(r.revision_id for r in rep1.results if r.group == "s3")
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0 and head() == [(rev,)]
    os.makedirs(s3_dir)
    with open(s3_uri, "wb") as fh:
        fh.write(s3_bytes)
    with DB.reading(DB.db_path(root)) as conn:
        RUN.run_pass(root, [ad], conn=conn, now=synth.t(3100), producer=PRODUCER)
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0 and head() == [(s3_rev,)] and current() > 0
    # 1. the sidecar is exported with s3_rev as the head; the store vanishes; the owning shard refuses the empty id as known, claim attached
    sidecar = str(tmp_path / "rs.sqlite")
    _export(root, sidecar, home)
    shutil.rmtree(s3_dir)
    wroot, cluster_root, reporting = _declared_retirement_window(tmp_path, ad, sidecar, synth.t(3200), name="wroot_moved")
    ((owner, rep3),) = reporting
    (res3,) = rep3.results
    claim = {"outgoing_revision_id": s3_rev, "inputs": [s3_uri], "patterns": [pattern]}
    assert res3.refused_known and res3.revision_id == rev and res3.retirement == claim and not _published(cluster_root)
    doc = json.load(open(CR.summary_path(str(wroot), ADAPTER, owner, 3)))
    assert doc["refused_known"][0]["retirement"] == claim and doc["published"] == {} and doc["refused"] == []
    # 2. BEFORE the pull, another writer accepts a newer revision of s3 with an additional input: the store is back with a manifest beside it
    os.makedirs(s3_dir)
    with open(s3_uri, "wb") as fh:
        fh.write(s3_bytes)
    with open(manifest, "w") as fh:
        json.dump({"schema": "manifest/1"}, fh)
    ad2 = TwoFileStoreAdapter(stores)
    with DB.reading(DB.db_path(root)) as conn:
        rep_new = RUN.run_pass(root, [ad2], conn=conn, now=synth.t(3250), producer=PRODUCER)
    res_new = next(r for r in rep_new.results if r.group == "s3")
    assert res_new.accepted and not res_new.reactivated and res_new.revision_id not in (rev, s3_rev), res_new.error
    s3_rev2 = res_new.revision_id
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0 and head() == [(s3_rev2,)] and current() > 0
    assert _rows(root, "SELECT input_uri FROM source_watermark WHERE source_id = ? AND last_error IS NULL ORDER BY input_uri", (sid3,)) == [(manifest,), (s3_uri,)]
    n_current = current()
    # 3. the pull: the gate keeps the claim for the trusted id; the re-activation is WITHHELD by name -- the head moved -- nothing written,
    # the newer head, its measurements and its added input untouched, the adapter NOT refused
    pulled, out = _retirement_pull(root, tmp_path, wroot, "w1", "expect_moved", synth.t(3300), ad)
    assert pulled["refused_known_ids"] == {ADAPTER: [rev]} and pulled["refused_known_claims"] == {ADAPTER: {rev: claim}}
    assert out[ADAPTER][rev]["state"] == "withheld" and out[ADAPTER][rev]["files"] == 0 and out[ADAPTER][rev]["withdrawals"] == 0, out
    assert "the head moved" in out[ADAPTER][rev]["error"] and s3_rev in out[ADAPTER][rev]["error"] and s3_rev2 in out[ADAPTER][rev]["error"]
    assert pulled["adapters_refused"] == {} and pulled["reactivations"] == out
    assert RUN.complete_bundles(root, ADAPTER, rev) == [] and _staged_revisions(root) == []     # no revision staged; the parent may stand
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0 and head() == [(s3_rev2,)] and current() == n_current
    assert _rows(root, "SELECT COUNT(*) FROM withdrawal WHERE from_revision_id = ?", (s3_rev2,)) == [(0,)]
    assert _rows(root, "SELECT last_error FROM source_watermark WHERE source_id = ? AND input_uri = ?", (sid3, manifest)) == [(None,)]
    # the refusal set a window reads names nothing for it, and the pull's receipt entry is not whole (the next tick reads the adapter again)
    assert CR.refused_adapters({"adapters": {ADAPTER: {"route": "cluster", "shards": 3, "task_indices": [0]}}, "tasks": [{"adapter": ADAPTER}], "refusals": {}}, pulled) == {}
    # a claim naming the LIVE head but not every live input (the manifest) is withheld too, naming the undeclared input; so is no claim at all
    with DB.reading(DB.db_path(root)) as conn:
        partial = RUN.reactivate_by_id(root, ad, rev, conn, now=synth.t(3310), retirement_claim={"outgoing_revision_id": s3_rev2, "inputs": [s3_uri], "patterns": [pattern]})
        none = RUN.reactivate_by_id(root, ad, rev, conn, now=synth.t(3320))
        budget = RUN.ReactivationBudget()
        RUN.reactivate_by_id(root, ad, rev, conn, now=synth.t(3330), budget=budget)
    assert partial.withheld and not partial.accepted and partial.files == [] and "never declared retired: " + manifest in partial.error
    assert none.withheld and "carries no retirement declaration" in none.error and budget.taken == 0        # a withheld retirement spends no budget
    assert RUN.complete_bundles(root, ADAPTER, rev) == []
    # 4. the next window: the store (both files) vanishes again; the shard declares against the NEW head (the glob covers the manifest too),
    # mints the same empty id, KNOWN -> refused with a claim that matches today's head; the pull re-activates: withdrawals from s3_rev2,
    # both watermarks retired, the declaration naming both inputs; the fold makes the empty revision the head
    shutil.rmtree(s3_dir)
    ad2.inputs_retired = [pattern]
    sidecar2 = str(tmp_path / "rs2.sqlite")
    _export(root, sidecar2, home)
    wroot2, cluster_root2, reporting2 = _declared_retirement_window(tmp_path, ad2, sidecar2, synth.t(3400), name="wroot_again")
    ((owner2, rep4),) = reporting2
    (res4,) = rep4.results
    claim2 = {"outgoing_revision_id": s3_rev2, "inputs": [manifest, s3_uri], "patterns": [pattern]}
    assert res4.refused_known and res4.revision_id == rev and res4.retirement == claim2 and not _published(cluster_root2)
    pulled2, out2 = _retirement_pull(root, tmp_path, wroot2, "w2", "expect_again", synth.t(3500), ad2)
    assert pulled2["refused_known_claims"] == {ADAPTER: {rev: claim2}} and out2[ADAPTER][rev]["state"] == "reactivated" and pulled2["adapters_refused"] == {}, out2
    (bundle,) = RUN.complete_bundles(root, ADAPTER, rev)
    recs = _records_by_type(os.path.dirname(bundle[1]))
    marker = recs["source_revision"][0]
    assert marker["extra"][RUN.RETIREMENT_DECLARED_KEY] == {"declared_by": "adapter:inputs_retired", "patterns": [pattern], "inputs": [manifest, s3_uri]}
    assert marker["extra"]["reactivated_from_committed_bytes"]["retirement_claim"] == {"outgoing_revision_id": s3_rev2, "inputs": 2, "patterns": 1}
    assert recs["withdrawal"] and all(w["reason"] == "input_vanished" and w["from_revision_id"] == s3_rev2 for w in recs["withdrawal"])
    assert sorted(w["input_uri"] for w in recs["source_watermark"]) == [manifest, s3_uri] and all(w["last_error"].startswith(RUN.VANISHED_PREFIX) for w in recs["source_watermark"])
    ing = H.ingest_once(root, workers=0, min_free=0)
    assert ing.refused == 0 and ing.blocked == 0, (ing.refusals, ing.blocked)
    assert head() == [(rev,)] and current() == 0
    assert _rows(root, "SELECT COUNT(*) FROM source_watermark WHERE source_id = ? AND last_error IS NULL", (sid3,)) == [(0,)]


def test_a_new_retirement_whose_head_moved_before_its_fold_is_held_by_the_harvester_and_re_activated_by_the_next_windows_claim(tmp_path):
    """Round 2 of the review of registry-hardening 3a (HIGH): a shard exports reader state with head H1 and PUBLISHES a new empty retirement
    R (never seen by the grading host) after the group's staged inputs vanished, declared retired. Before R is pulled and folded, another writer --
    here the operator's pass, in production an earlier window's delayed revision -- accepts a NEWER revision H2 of the source with an ADDED
    input. R's withdrawals name H1's keys and its inventory is H1's; the refused-known claim gate of round 1 never sees R (it is a fresh
    publication), the pull checks the attestation and the module, and the run-drop guard accepts R's wildcard declaration -- so, folded, R
    became the head and silently withdrew H2's measurements and retired an input nobody declared retired. Now every retirement marker names
    the premise it was computed from (`extra.retirement_expects`: the outgoing head, the inputs it retires) and the harvester compares it
    with the live state INSIDE the activation transaction: a moved head holds the marker by name (`kind = retirement_head`), nothing
    withdrawn, nothing retired, the head current; the hold is exact while that head stands; a bundle whose premise names the live head but
    not every live input is held the same way, naming the input; the tick counts the hold as standing unread (read from the database) so
    the receipt that delivered R does not park the adapter; the pull's stale claim stays withheld. The next window's shard sees the group
    vanished against H2, refuses the same empty id as known with a claim against today's head, and the pull re-activates it -- this one hold
    is resolvable by a claim, where every other hold stays `held` -- as a bundle computed against H2, which the fold accepts: R is the head,
    both inputs retired, the holds cleared."""
    from sisteer.registry import reads_protocol as RP
    root, home, stores, ad = _grading_host(tmp_path)
    sid3 = f"{ADAPTER}:s3"
    s3_dir, s3_uri, manifest = os.path.join(stores, "s3"), os.path.join(stores, "s3", "store.json"), os.path.join(stores, "s3", "manifest.json")
    ((s3_rev,),) = _rows(root, "SELECT revision_id FROM source_revision WHERE source_id = ? AND accepted = 1", (sid3,))
    s3_bytes = open(s3_uri, "rb").read()
    pattern = os.path.join(stores, "s3", "*.json")
    ad.inputs_retired = [pattern]
    head = lambda: _rows(root, "SELECT revision_id FROM source_revision WHERE source_id = ? AND accepted = 1 ORDER BY observed_utc DESC LIMIT 1", (sid3,))
    current = lambda: _rows(root, "SELECT COUNT(*) FROM observation WHERE source_id = ? AND current = 1", (sid3,))[0][0]
    live_marks = lambda: _rows(root, "SELECT input_uri FROM source_watermark WHERE source_id = ? AND last_error IS NULL ORDER BY input_uri", (sid3,))
    held_rows = lambda: _rows(root, "SELECT missing_parent_table, missing_parent_id, json_extract(extra_json, '$.kind'), json_extract(extra_json, '$.rule'), record_uri "
                                    "FROM blocked_records WHERE json_extract(extra_json, '$.revision_id') = ? ORDER BY record_uri", (rev,))
    # 1. the sidecar carries s3_rev as the head. An EARLIER read of the source -- a manifest appeared beside the store -- has published a
    # newer revision H2 with the added input, still unfolded (the delayed revision of the review's scenario)
    sidecar = str(tmp_path / "rs.sqlite")
    _export(root, sidecar, home)
    with open(manifest, "w") as fh:
        json.dump({"schema": "manifest/1"}, fh)
    ad2 = TwoFileStoreAdapter(stores)
    with DB.reading(DB.db_path(root)) as conn:
        rep_new = RUN.run_pass(root, [ad2], conn=conn, now=synth.t(3150), producer=PRODUCER)
    res_new = next(r for r in rep_new.results if r.group == "s3")
    assert res_new.accepted and not res_new.reactivated and res_new.revision_id != s3_rev, res_new.error
    s3_rev2 = res_new.revision_id
    assert head() == [(s3_rev,)]
    # 2. the store vanishes; the owning shard, reading against the sidecar (head s3_rev), PUBLISHES the retirement -- a NEW id -- with its premise
    shutil.rmtree(s3_dir)
    wroot, cluster_root, reporting = _declared_retirement_window(tmp_path, ad, sidecar, synth.t(3200), name="wroot_new")
    ((owner, rep1),) = reporting
    (res1,) = rep1.results
    rev = res1.revision_id
    assert res1.accepted and not res1.refused_known and res1.withdrawals > 0 and _published(cluster_root) == [rev] and rev != s3_rev2, res1.error
    rev_dir = f"{ADAPTER}/{rev}"
    marker0 = _marker(os.path.join(H.records_dir(cluster_root), "adapters", ADAPTER, rev))["payload"]
    assert marker0["inputs_json"] == {} and marker0["extra"][H.RETIREMENT_EXPECTS_KEY] == {"outgoing_revision_id": s3_rev, "inputs": [s3_uri]}
    assert marker0["extra"][RUN.RETIREMENT_DECLARED_KEY]["inputs"] == [s3_uri] and marker0["extra"]["run_set"]["previous_revision_id"] == s3_rev
    # BEFORE the pull, the delayed revision folds: H2 is the accepted head, its added input live. Observed before R, so an accepted R would
    # be the head by (observed_utc, revision_id) -- the review's scenario: folded as published, R silently dropped H2's measurements
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0 and head() == [(s3_rev2,)] and live_marks() == [(manifest,), (s3_uri,)]
    n_current = current()
    assert n_current > 0
    # 3. the pull delivers the marker-only directory like any other revision (no claim gate sees a fresh publication); the FOLD HOLDS IT:
    # the head moved under the retirement, so nothing is withdrawn or retired, the head and its added input stand, the hold names both heads
    CR._write_json(str(wroot / ST.ATTESTATION_FILE), _attestation_record("w1", {"synthetic_stores": _attested_generation("deadbeef")}))
    plan = {"window": "w1", "host": None, "window_root": str(wroot), "adapters": {ADAPTER: {"route": "cluster", "shards": 3, "pins": {"synthetic_stores": "deadbeef"}}},
            "tasks": [{"adapter": ADAPTER}], "stages": {"revalidate_post": {"stale": []}}, "refusals": {}}
    pulled = CR.pull_window(root, plan, str(tmp_path / "expect_new"), source=str(wroot))
    assert pulled["pulled"][rev_dir]["state"] == "complete" and pulled["adapters_refused"] == {} and not pulled["refused_known_ids"].get(ADAPTER), pulled
    assert [status_ for _, status_ in H.scan_inbox(root)] == ["merged"]
    ing = H.ingest_once(root, workers=0, min_free=0)
    assert ing.refused == 0 and ing.revisions_held == 1 and ing.revisions_held_retirement_head == 1 and ing.revisions_held_run_drop == 0, (ing.refusals, ing.blocked)
    assert head() == [(s3_rev2,)] and current() == n_current and live_marks() == [(manifest,), (s3_uri,)]
    assert _rows(root, "SELECT COUNT(*) FROM withdrawal WHERE to_revision_id = ? OR from_revision_id = ?", (rev, s3_rev2)) == [(0,)]
    assert _rows(root, "SELECT found FROM source_state WHERE input_uri = ?", (s3_uri,)) == [(1,)]
    ((accepted, refusal),) = _rows(root, "SELECT accepted, refusal FROM source_revision WHERE revision_id = ?", (rev,))
    assert accepted == 0 and refusal.startswith(f"activation pending: source_revision {s3_rev2} ({H.RETIREMENT_HEAD_KEY}: the retirement {rev} of {sid3} was computed "
                                                f"against the accepted head {s3_rev}, and the live accepted head is {s3_rev2};"), refusal
    assert "nothing is withdrawn or retired" in refusal and "judges the group against today's head" in refusal
    (hold,) = held_rows()
    assert hold[:4] == ("source_revision", s3_rev2, H.RETIREMENT_HEAD_KIND, H.RETIREMENT_HEAD_RULE) and f"/adapters/{rev_dir}/" in hold[4]
    assert _rows(root, "SELECT status FROM record_delivery WHERE file_uri = ?", (hold[4],)) == [("blocked",)]
    # the hold is exact while s3_rev2 stands (R is not superseded: H2 was observed before it): a second cycle re-attempts nothing and changes nothing
    assert _rows(root, "SELECT superseded_by_revision_id FROM source_revision WHERE revision_id = ?", (rev,)) == [(None,)]
    ing_b = H.ingest_once(root, workers=0, min_free=0)
    assert ing_b.revision_transactions == 0 and ing_b.refused == 0 and head() == [(s3_rev2,)] and current() == n_current
    assert _rows(root, "SELECT accepted FROM source_revision WHERE revision_id = ?", (rev,)) == [(0,)]
    # the reads side: the hold is new information the cluster has not seen -- read from the database, it keeps the adapter unread for the
    # next tick even though the receipt that delivered R read the adapter whole (the ordering rule is the reads-protocol test's)
    with DB.reading(DB.db_path(root)) as conn:
        holds = H.retirements_held_for_moved_head(conn, ADAPTER)
        assert H.retirements_held_for_moved_head(conn, "relu_paired") == []
    assert [(h["revision_id"], h["source_id"]) for h in holds] == [(rev, sid3)] and holds[0]["first_seen_utc"] and holds[0]["record_uri"] == hold[4]
    rd = str(tmp_path / "reads")
    os.makedirs(rd)
    assert RP.standing_ids(root, rd, ADAPTER, recs=[]) == [rev] and RP.standing_ids(root, rd, "relu_paired", recs=[]) == []
    # 4. the pull's own gate is unchanged: a claim against the OLD head is withheld (never `held` on this hold), so is no claim; no bundle
    with DB.reading(DB.db_path(root)) as conn:
        stale = RUN.reactivate_by_id(root, ad, rev, conn, now=synth.t(3310), retirement_claim={"outgoing_revision_id": s3_rev, "inputs": [s3_uri], "patterns": [pattern]})
        none = RUN.reactivate_by_id(root, ad, rev, conn, now=synth.t(3320))
    assert stale.withheld and not stale.held and not stale.accepted and "the head moved" in stale.error and s3_rev2 in stale.error, stale.error
    assert none.withheld and not none.held and "carries no retirement declaration" in none.error
    assert RUN.complete_bundles(root, ADAPTER, rev) == [] and head() == [(s3_rev2,)]
    # 5. a bundle whose premise names the LIVE head but not every live input (the manifest) is held by the same gate, naming the input
    with DB.reading(DB.db_path(root)) as conn:
        ctx = RUN.AdapterContext(root=root, now=synth.t(3330), producer=dict(PRODUCER), conn=conn)
        partial = RUN._publish_retirement_bundle(root, ad, "s3", sid3, rev, s3_rev2, [s3_uri], [s3_uri], [pattern], ctx, conn)
    assert partial.accepted and partial.reactivated and len(RUN.complete_bundles(root, ADAPTER, rev)) == 1
    recs_p = _records_by_type(os.path.dirname(RUN.complete_bundles(root, ADAPTER, rev)[0][1]))
    assert recs_p["source_revision"][0]["extra"][H.RETIREMENT_EXPECTS_KEY] == {"outgoing_revision_id": s3_rev2, "inputs": [s3_uri]}
    ing_c = H.ingest_once(root, workers=0, min_free=0)
    assert ing_c.refused == 0 and ing_c.revisions_held_retirement_head == 1 and head() == [(s3_rev2,)] and current() == n_current
    assert live_marks() == [(manifest,), (s3_uri,)] and _rows(root, "SELECT COUNT(*) FROM withdrawal WHERE to_revision_id = ?", (rev,)) == [(0,)]
    ((refusal_c,),) = _rows(root, "SELECT refusal FROM source_revision WHERE revision_id = ? AND accepted = 0", (rev,))
    assert f"({H.RETIREMENT_HEAD_KEY}: the live inventory of {sid3} holds 1 input(s) the retirement {rev} does not name (1 of 1 named): {manifest};" in refusal_c, refusal_c
    assert sorted(h[2] for h in held_rows()) == [H.RETIREMENT_HEAD_KIND, H.RETIREMENT_HEAD_KIND] and any("/_activations/" in h[4] for h in held_rows())
    # 6. the next window: the store (both files) vanishes again; the shard declares against the NEW head (the glob covers the manifest too),
    # mints the same empty id, KNOWN -> refused with a claim that matches today's head; the pull RE-ACTIVATES it (the head-moved hold is the
    # one hold a fresh claim resolves): a bundle computed against s3_rev2 with that premise, which the fold accepts -- R is the head, both
    # inputs retired, the holds cleared, nothing left standing for the tick
    assert not os.path.exists(s3_dir)             # both files have been gone since step 2
    ad2.inputs_retired = [pattern]
    sidecar2 = str(tmp_path / "rs2.sqlite")
    _export(root, sidecar2, home)
    wroot2, cluster_root2, reporting2 = _declared_retirement_window(tmp_path, ad2, sidecar2, synth.t(3400), name="wroot_again")
    ((owner2, rep4),) = reporting2
    (res4,) = rep4.results
    claim2 = {"outgoing_revision_id": s3_rev2, "inputs": [manifest, s3_uri], "patterns": [pattern]}
    assert res4.refused_known and res4.revision_id == rev and res4.retirement == claim2 and not _published(cluster_root2), res4.error
    pulled2, out2 = _retirement_pull(root, tmp_path, wroot2, "w2", "expect_again", synth.t(3500), ad2)
    assert pulled2["refused_known_claims"] == {ADAPTER: {rev: claim2}} and pulled2["adapters_refused"] == {}
    assert out2[ADAPTER][rev]["state"] == "reactivated" and out2[ADAPTER][rev]["withdrawals"] > 0, out2
    bundles = RUN.complete_bundles(root, ADAPTER, rev)
    assert len(bundles) == 2
    recs = _records_by_type(os.path.dirname(bundles[-1][1]))
    marker = recs["source_revision"][0]
    assert marker["extra"][H.RETIREMENT_EXPECTS_KEY] == {"outgoing_revision_id": s3_rev2, "inputs": [manifest, s3_uri]}
    assert marker["extra"][RUN.RETIREMENT_DECLARED_KEY] == {"declared_by": "adapter:inputs_retired", "patterns": [pattern], "inputs": [manifest, s3_uri]}
    assert marker["extra"]["reactivated_from_committed_bytes"]["retirement_claim"] == {"outgoing_revision_id": s3_rev2, "inputs": 2, "patterns": 1}
    assert recs["withdrawal"] and all(w["reason"] == "input_vanished" and w["from_revision_id"] == s3_rev2 for w in recs["withdrawal"])
    ing2 = H.ingest_once(root, workers=0, min_free=0)
    assert ing2.refused == 0 and ing2.revisions_held_retirement_head == 0, (ing2.refusals, ing2.blocked)
    assert head() == [(rev,)] and current() == 0 and live_marks() == []
    assert _rows(root, "SELECT accepted, refusal FROM source_revision WHERE revision_id = ?", (rev,)) == [(1, None)]
    assert _rows(root, "SELECT DISTINCT reason FROM withdrawal WHERE to_revision_id = ?", (rev,)) == [("input_vanished",)]
    assert _rows(root, "SELECT found FROM source_state WHERE input_uri = ?", (s3_uri,)) == [(0,)]
    if held_rows():
        H.ingest_once(root, workers=0, min_free=0)          # the superseded holds stand down on the next re-evaluation
    assert held_rows() == []
    with DB.reading(DB.db_path(root)) as conn:
        assert H.retirements_held_for_moved_head(conn, ADAPTER) == []
    assert RP.standing_ids(root, rd, ADAPTER, recs=[]) == []
    # 7. the window after: nothing to refuse or retire (the empty revision IS the head)
    sidecar3 = str(tmp_path / "rs3.sqlite")
    _export(root, sidecar3, home)
    rep5 = _cluster_pass(_cluster_root(tmp_path, "cluster_root_quiet"), ad2, sidecar3, synth.t(3600), shard=(owner2, 3))
    assert rep5.results == [] and rep5.groups_refused == 0 and rep5.groups_refused_known == 0


def test_a_held_re_activation_of_a_retirement_accepted_before_is_standing_unread_off_its_queue_row_and_the_next_windows_claim_resolves_it(tmp_path):
    """Round 3 of the review of registry-hardening 3a (HIGH): a retirement R was ACCEPTED once; its source came back (the accepted revision
    re-activated, the measurements current again) and vanished again, so the next shard refused the same empty id as known with a claim
    against the live head and the pull re-activated R as a bundle. Before that bundle folded, another revision of the source -- here the
    operator's pass over the store back with a manifest beside it, in production an earlier window's delayed revision -- became the
    accepted head, and the fold-time gate of round 2 HELD the bundle. R keeps `accepted = 1` as a superseded historical revision
    (`_placeholder_revision` leaves an accepted row accepted; the hold lives on the bundle marker's queue row), and
    `retirements_held_for_moved_head` filtered the hold out by `accepted = 0`: the window's receipt recorded a successful re-activation and
    read whole, no later tick asked for a fresh claim, and the vanished group's measurements stayed current for good. Now the standing hold
    is read off its queue row whatever the revision's row says (only the source's live head is never a standing hold), the plan records the
    holds the database carried when it was built (`holds_seen`), the receipt that re-activated the bundle names none, so the tick reads the
    adapter again; the next window's shard claims against today's head, the pull re-activates (this hold is the one a claim resolves, the
    accepted row notwithstanding) and the fold accepts: R is the head, both inputs retired, the holds cleared, nothing standing."""
    from sisteer.registry import reads_protocol as RP
    root, home, stores, ad = _grading_host(tmp_path)
    sid3 = f"{ADAPTER}:s3"
    s3_dir, s3_uri, manifest = os.path.join(stores, "s3"), os.path.join(stores, "s3", "store.json"), os.path.join(stores, "s3", "manifest.json")
    ((s3_rev,),) = _rows(root, "SELECT revision_id FROM source_revision WHERE source_id = ? AND accepted = 1", (sid3,))
    s3_bytes = open(s3_uri, "rb").read()
    pattern = os.path.join(stores, "s3", "*.json")
    ad.inputs_retired = [pattern]
    head = lambda: _rows(root, "SELECT revision_id FROM source_revision WHERE source_id = ? AND accepted = 1 ORDER BY observed_utc DESC LIMIT 1", (sid3,))
    current = lambda: _rows(root, "SELECT COUNT(*) FROM observation WHERE source_id = ? AND current = 1", (sid3,))[0][0]
    live_marks = lambda: _rows(root, "SELECT input_uri FROM source_watermark WHERE source_id = ? AND last_error IS NULL ORDER BY input_uri", (sid3,))
    # 1. the first retirement (the grading host path): the empty revision `rev` is the head
    shutil.rmtree(s3_dir)
    with DB.reading(DB.db_path(root)) as conn:
        rep1 = RUN.run_pass(root, [ad], conn=conn, now=synth.t(3000), producer=PRODUCER)
    rev = next(r.revision_id for r in rep1.results if r.group == "s3")
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0 and head() == [(rev,)] and current() == 0
    held_rows = lambda: _rows(root, "SELECT missing_parent_table, missing_parent_id, json_extract(extra_json, '$.kind'), record_uri FROM blocked_records "
                                    "WHERE json_extract(extra_json, '$.revision_id') = ? ORDER BY record_uri", (rev,))
    # 2. the store comes back byte-identical: s3_rev re-activates (a bundle); R stays accepted, superseded
    os.makedirs(s3_dir)
    with open(s3_uri, "wb") as fh:
        fh.write(s3_bytes)
    with DB.reading(DB.db_path(root)) as conn:
        rep2 = RUN.run_pass(root, [ad], conn=conn, now=synth.t(3100), producer=PRODUCER)
    res2 = next(r for r in rep2.results if r.group == "s3")
    assert res2.accepted and res2.reactivated and res2.revision_id == s3_rev, res2.error
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0 and head() == [(s3_rev,)] and current() > 0
    assert _rows(root, "SELECT accepted FROM source_revision WHERE revision_id = ?", (rev,)) == [(1,)] and held_rows() == []
    # 3. it vanishes again: the shard (sidecar head s3_rev) refuses R as known with its claim; the pull RE-ACTIVATES R as a bundle B
    # computed against s3_rev. No hold stands as this window is planned, and its receipt -- the adapter read whole, R re-activated -- says so
    shutil.rmtree(s3_dir)
    sidecar = str(tmp_path / "rs.sqlite")
    _export(root, sidecar, home)
    assert CR.holds_seen_of(sidecar, [ADAPTER]) == {ADAPTER: []}
    wroot, cluster_root, reporting = _declared_retirement_window(tmp_path, ad, sidecar, synth.t(3200), name="wroot_b")
    ((owner, rep3),) = reporting
    (res3,) = rep3.results
    claim = {"outgoing_revision_id": s3_rev, "inputs": [s3_uri], "patterns": [pattern]}
    assert res3.refused_known and res3.revision_id == rev and res3.retirement == claim and not _published(cluster_root), res3.error
    pulled, out = _retirement_pull(root, tmp_path, wroot, "wb", "expect_b", synth.t(3300), ad)
    assert pulled["refused_known_claims"] == {ADAPTER: {rev: claim}} and out[ADAPTER][rev]["state"] == "reactivated", out
    (bundle_b,) = RUN.complete_bundles(root, ADAPTER, rev)
    bundle_dir = os.path.dirname(bundle_b[1])
    assert _records_by_type(bundle_dir)["source_revision"][0]["extra"][H.RETIREMENT_EXPECTS_KEY] == {"outgoing_revision_id": s3_rev, "inputs": [s3_uri]}
    blobs = {ADAPTER: {"src/x.py": "b1"}}
    plan_b = {"window": "wb", "tree_sha": "a" * 40, "destination": "cluster_a", "refusals": {},
              "adapters": {ADAPTER: {"route": "cluster", "shards": 3, "pins": {"synthetic_stores": "deadbeef"}, "holds_seen": CR.holds_seen_of(sidecar, [ADAPTER])[ADAPTER]}},
              "stages": {"pull": {"summaries": {ADAPTER: {"expected": 3, "present": 3, "ok": 3}}, "pulled_dirs": [], "reactivations": {ADAPTER: {rev: "reactivated"}},
                                  "attestation": {"ok": True}, "barrier": None}}}
    w_b = RP.receipt_from_plan(plan_b, {}, blobs, {"op_id": "wb", "owner": "window"}, now=lambda: synth.t(3310))
    e_b = w_b["adapters"][ADAPTER]
    assert e_b["read_whole"] is True and RP.attempted(e_b) and e_b["reactivated"] == [rev] and e_b["holds_seen"] == []
    rd = str(tmp_path / "reads")
    os.makedirs(rd)
    RP.write_receipt(rd, w_b)
    # 4. BEFORE B folds, the store comes back with a manifest beside it and the operator's pass publishes H2 (s3_rev2, the manifest an added
    # input), which folds first -- B's fold is the next cycle's: the pull landed B after this cycle's listing (B's bundle root set aside for
    # the one cycle), the cycle boundary of the review's "before its activation bundle folds"
    bundle_root = os.path.dirname(bundle_dir)
    aside = str(tmp_path / "bundle_root_aside")
    shutil.move(bundle_root, aside)
    os.makedirs(s3_dir)
    with open(s3_uri, "wb") as fh:
        fh.write(s3_bytes)
    with open(manifest, "w") as fh:
        json.dump({"schema": "manifest/1"}, fh)
    ad2 = TwoFileStoreAdapter(stores)
    with DB.reading(DB.db_path(root)) as conn:
        rep_new = RUN.run_pass(root, [ad2], conn=conn, now=synth.t(3350), producer=PRODUCER)
    res_new = next(r for r in rep_new.results if r.group == "s3")
    assert res_new.accepted and not res_new.reactivated and res_new.revision_id not in (s3_rev, rev), res_new.error
    s3_rev2 = res_new.revision_id
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0 and head() == [(s3_rev2,)] and live_marks() == [(manifest,), (s3_uri,)]
    n_current = current()
    assert n_current > 0
    shutil.move(aside, bundle_root)
    # 5. B folds: HELD -- computed against s3_rev, the live head is s3_rev2 -- nothing withdrawn or retired, the head and its added input
    # stand; R's row stays accepted = 1 and says nothing about the hold, which lives on the bundle marker's queue row
    ing = H.ingest_once(root, workers=0, min_free=0)
    assert ing.refused == 0 and ing.revisions_held == 1 and ing.revisions_held_retirement_head == 1 and ing.revisions_held_run_drop == 0, (ing.refusals, ing.blocked)
    assert head() == [(s3_rev2,)] and current() == n_current and live_marks() == [(manifest,), (s3_uri,)]
    assert _rows(root, "SELECT COUNT(*) FROM withdrawal WHERE from_revision_id = ?", (s3_rev2,)) == [(0,)]
    assert _rows(root, "SELECT accepted, refusal FROM source_revision WHERE revision_id = ?", (rev,)) == [(1, None)]
    (hold,) = held_rows()
    assert hold[:3] == ("source_revision", s3_rev2, H.RETIREMENT_HEAD_KIND) and "/_activations/" in hold[3] and os.path.basename(bundle_dir) in hold[3], hold
    # the hold is exact while s3_rev2 stands: a second cycle re-attempts nothing and changes nothing
    ing_b = H.ingest_once(root, workers=0, min_free=0)
    assert ing_b.revision_transactions == 0 and ing_b.refused == 0 and head() == [(s3_rev2,)] and current() == n_current
    assert _rows(root, "SELECT accepted FROM source_revision WHERE revision_id = ?", (rev,)) == [(1,)] and len(held_rows()) == 1
    # 6. the reads side: the hold is listed off its queue row though R's row is accepted (another adapter: nothing), with the head it was
    # computed against; a plan built NOW -- off a sidecar exported now -- records it; the receipt that re-activated B had no hold in hand, so
    # the tick reads the adapter again, its whole receipt and unchanged pins and module notwithstanding
    with DB.reading(DB.db_path(root)) as conn:
        holds = H.retirements_held_for_moved_head(conn, ADAPTER)
        assert H.retirements_held_for_moved_head(conn, "relu_paired") == []
    assert [(h["revision_id"], h["source_id"], h["record_uri"], h["blocking_head"]) for h in holds] == [(rev, sid3, hold[3], s3_rev2)] and holds[0]["first_seen_utc"]
    sidecar_c = str(tmp_path / "rs_c.sqlite")
    _export(root, sidecar_c, home, adapters=(ADAPTER, "relu_paired"))
    seen_now = CR.holds_seen_of(sidecar_c, [ADAPTER, "relu_paired"])
    assert seen_now == {ADAPTER: [{"revision_id": rev, "record_uri": hold[3], "blocking_head": s3_rev2}], "relu_paired": []}
    cfgp = TS._cfg(tmp_path, {"synthetic_stores": {"base": stores, "authority": "grading_host_copy", "origin": "grading_host"}},
                   adapters={ADAPTER: {"shards": 3, "components": ["synthetic_stores"]}})
    cfg = ST.load_config(cfgp)
    plan_now = CR.plan_window(root, "wc", [ADAPTER, "relu_paired"], cfg, cfg.destination("cluster_a"), "a" * 40, uri_home=home, published={}, sidecar=sidecar_c)
    assert plan_now["adapters"][ADAPTER]["holds_seen"] == seen_now[ADAPTER] and plan_now["adapters"]["relu_paired"]["holds_seen"] == []
    assert plan_now["holds_seen_from"]["sidecar"] == sidecar_c and plan_now["holds_seen_from"]["ledger_head"] == RS.meta(RS.open_reader_state(sidecar_c))["ledger_head"]
    assert RP.standing_ids(root, rd, ADAPTER) == [rev] and RP.standing_ids(root, rd, "relu_paired") == []
    probed = {"components": {"synthetic_stores": {"current": "deadbeef", "exists": True}}}
    cs = RP.changed_set(cfg, [ADAPTER], probed, blobs, lambda a: RP.last_receipt_for(rd, a), standing=lambda a: RP.standing_ids(root, rd, a))
    assert cs["unchanged"] == [] and cs["changed"][ADAPTER] == [f"refused-known id(s) left standing unread by the last pull: {rev} -- read fresh on this tree"], cs
    # a receipt of a window planned now (the hold in hand) that attempted the adapter would end it; the one that re-activated B does not
    plan_c = dict(plan_b, window="wc", adapters={ADAPTER: dict(plan_b["adapters"][ADAPTER], holds_seen=seen_now[ADAPTER])},
                  stages={"pull": dict(plan_b["stages"]["pull"], reactivations={})})
    w_c = RP.receipt_from_plan(plan_c, {}, blobs, {"op_id": "wc", "owner": "window"}, now=lambda: synth.t(3390))
    assert RP.held_retirements_unread(ADAPTER, [w_c, w_b], holds) == [] and RP.held_retirements_unread(ADAPTER, [w_b], holds) == [rev]
    # 7. the next window: the store (both files) vanishes again; the shard declares against s3_rev2 (the glob covers the manifest too), mints
    # the same empty id, KNOWN -> refused with a claim matching today's head; the pull RE-ACTIVATES it, the accepted row notwithstanding (the
    # head-moved hold is the one a fresh claim resolves): a bundle computed against s3_rev2, which the fold accepts -- R is the head, both
    # inputs retired, the holds cleared, nothing left standing for the tick
    shutil.rmtree(s3_dir)
    ad2.inputs_retired = [pattern]
    sidecar2 = str(tmp_path / "rs2.sqlite")
    _export(root, sidecar2, home)
    wroot2, cluster_root2, reporting2 = _declared_retirement_window(tmp_path, ad2, sidecar2, synth.t(3400), name="wroot_d")
    ((owner2, rep4),) = reporting2
    (res4,) = rep4.results
    claim2 = {"outgoing_revision_id": s3_rev2, "inputs": [manifest, s3_uri], "patterns": [pattern]}
    assert res4.refused_known and res4.revision_id == rev and res4.retirement == claim2 and not _published(cluster_root2), res4.error
    pulled2, out2 = _retirement_pull(root, tmp_path, wroot2, "wd", "expect_d", synth.t(3500), ad2)
    assert pulled2["refused_known_claims"] == {ADAPTER: {rev: claim2}} and out2[ADAPTER][rev]["state"] == "reactivated" and out2[ADAPTER][rev]["withdrawals"] > 0, out2
    bundles = RUN.complete_bundles(root, ADAPTER, rev)
    assert len(bundles) == 2
    marker = _records_by_type(os.path.dirname(bundles[-1][1]))["source_revision"][0]
    assert marker["extra"][H.RETIREMENT_EXPECTS_KEY] == {"outgoing_revision_id": s3_rev2, "inputs": [manifest, s3_uri]}
    assert marker["extra"]["reactivated_from_committed_bytes"]["retirement_claim"] == {"outgoing_revision_id": s3_rev2, "inputs": 2, "patterns": 1}
    ing2 = H.ingest_once(root, workers=0, min_free=0)
    assert ing2.refused == 0 and ing2.revisions_held_retirement_head == 0, (ing2.refusals, ing2.blocked)
    assert head() == [(rev,)] and current() == 0 and live_marks() == []
    assert _rows(root, "SELECT accepted, refusal FROM source_revision WHERE revision_id = ?", (rev,)) == [(1, None)]
    assert _rows(root, "SELECT DISTINCT reason FROM withdrawal WHERE to_revision_id = ? AND from_revision_id = ?", (rev, s3_rev2)) == [("input_vanished",)]
    if held_rows():
        H.ingest_once(root, workers=0, min_free=0)          # the superseded holds stand down on the next re-evaluation
    assert held_rows() == []
    with DB.reading(DB.db_path(root)) as conn:
        assert H.retirements_held_for_moved_head(conn, ADAPTER) == []
    assert RP.standing_ids(root, rd, ADAPTER) == []
    # 8. the window after: nothing to refuse or retire (the empty revision IS the head); its sidecar carries no hold
    sidecar3 = str(tmp_path / "rs3.sqlite")
    _export(root, sidecar3, home)
    assert CR.holds_seen_of(sidecar3, [ADAPTER]) == {ADAPTER: []}
    rep5 = _cluster_pass(_cluster_root(tmp_path, "cluster_root_quiet"), ad2, sidecar3, synth.t(3600), shard=(owner2, 3))
    assert rep5.results == [] and rep5.groups_refused == 0 and rep5.groups_refused_known == 0

class ThreeFileStoreAdapter(TwoFileStoreAdapter):
    """:class:`TwoFileStoreAdapter` whose groups may carry a third input, ``notes.json``, beside the store and its manifest."""

    FILES = ("store.json", "manifest.json", "notes.json")

    def discover(self) -> List[AB.Input]:
        out = []
        for d in sorted(os.listdir(self.roots[0])):
            for fn in self.FILES:
                p = os.path.join(self.roots[0], d, fn)
                if os.path.lexists(p):
                    out.append(AB.check_input(p, role="report", allow_symlink=self.allow_symlinks))
        return out


def _held_retirement_world(tmp_path) -> Dict:
    """Round 2's opening, up to -- not including -- the fold that HOLDS the new retirement: a grading host whose sidecar carries s3's head H1; a
    manifest appears beside the store and the operator's pass publishes H2 (unfolded); the store vanishes and the owning shard, reading
    against that sidecar, publishes the retirement R (a NEW id, computed against H1) onto its window root; H2 folds first and is the head,
    its added input live. R sits published on the window root, not yet pulled: the pull and the fold that holds it (the head moved under
    it) are the caller's (:func:`_pull_held_retirement`), placed where its scenario needs them."""
    root, home, stores, ad = _grading_host(tmp_path)
    sid3 = f"{ADAPTER}:s3"
    s3_dir, s3_uri, manifest = os.path.join(stores, "s3"), os.path.join(stores, "s3", "store.json"), os.path.join(stores, "s3", "manifest.json")
    ((s3_rev,),) = _rows(root, "SELECT revision_id FROM source_revision WHERE source_id = ? AND accepted = 1", (sid3,))
    s3_bytes = open(s3_uri, "rb").read()
    pattern = os.path.join(stores, "s3", "*.json")
    ad.inputs_retired = [pattern]
    sidecar0 = str(tmp_path / "rs0.sqlite")
    _export(root, sidecar0, home)
    with open(manifest, "w") as fh:
        json.dump({"schema": "manifest/1"}, fh)
    ad2 = TwoFileStoreAdapter(stores)
    with DB.reading(DB.db_path(root)) as conn:
        rep_new = RUN.run_pass(root, [ad2], conn=conn, now=synth.t(3150), producer=PRODUCER)
    res_new = next(r for r in rep_new.results if r.group == "s3")
    assert res_new.accepted and not res_new.reactivated and res_new.revision_id != s3_rev, res_new.error
    s3_rev2 = res_new.revision_id
    shutil.rmtree(s3_dir)
    wroot, cluster_root, reporting = _declared_retirement_window(tmp_path, ad, sidecar0, synth.t(3200), name="wroot_held")
    ((owner, rep1),) = reporting
    (res1,) = rep1.results
    rev = res1.revision_id
    assert res1.accepted and not res1.refused_known and _published(cluster_root) == [rev] and rev not in (s3_rev, s3_rev2), res1.error
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    assert _rows(root, "SELECT revision_id FROM source_revision WHERE source_id = ? AND accepted = 1 ORDER BY observed_utc DESC LIMIT 1", (sid3,)) == [(s3_rev2,)]
    return dict(root=root, home=home, stores=stores, ad=ad, ad2=ad2, sid3=sid3, s3_dir=s3_dir, s3_uri=s3_uri, manifest=manifest, s3_bytes=s3_bytes,
                pattern=pattern, s3_rev=s3_rev, s3_rev2=s3_rev2, rev=rev, wroot=wroot, owner=owner)


def _pull_held_retirement(w: Dict, tmp_path, window: str):
    """The pull that delivers the world's retirement R (:func:`_held_retirement_world`) and the harvester cycle that HOLDS it against the live
    head -- the retirement's premise names the head the shard's sidecar carried, not today's: ``(pull report, ingest report)``."""
    root, wroot, rev = w["root"], w["wroot"], w["rev"]
    CR._write_json(str(wroot / ST.ATTESTATION_FILE), _attestation_record(window, {"synthetic_stores": _attested_generation("deadbeef")}))
    plan = {"window": window, "host": None, "window_root": str(wroot), "adapters": {ADAPTER: {"route": "cluster", "shards": 3, "pins": {"synthetic_stores": "deadbeef"}}},
            "tasks": [{"adapter": ADAPTER}], "stages": {"revalidate_post": {"stale": []}}, "refusals": {}}
    pulled = CR.pull_window(root, plan, str(tmp_path / f"expect_{window}"), source=str(wroot))
    assert pulled["pulled"][f"{ADAPTER}/{rev}"]["state"] == "complete" and pulled["adapters_refused"] == {}, pulled
    assert [status_ for _, status_ in H.scan_inbox(root)] == ["merged"]
    ing = H.ingest_once(root, workers=0, min_free=0)
    assert ing.refused == 0 and ing.revisions_held == 1 and ing.revisions_held_retirement_head == 1, (ing.refusals, ing.blocked)
    return pulled, ing


def _attempted_receipt(window: str, holds_seen, holds_seen_from, now: str, blobs, pulled_dirs=(), reactivations=None) -> Dict:
    """A receipt of a window that ATTEMPTED the adapter (every shard ran) and read it whole, from a plan that recorded ``holds_seen`` off the
    snapshot ``holds_seen_from`` -- what the tick writes after its pull (``reads_protocol.receipt_from_plan``)."""
    from sisteer.registry import reads_protocol as RP
    plan = {"window": window, "tree_sha": "a" * 40, "destination": "cluster_a", "refusals": {}, "holds_seen_from": holds_seen_from,
            "adapters": {ADAPTER: {"route": "cluster", "shards": 3, "pins": {"synthetic_stores": "deadbeef"}, "holds_seen": list(holds_seen)}},
            "stages": {"pull": {"summaries": {ADAPTER: {"expected": 3, "present": 3, "ok": 3}}, "pulled_dirs": list(pulled_dirs),
                                "reactivations": reactivations or {}, "attestation": {"ok": True}, "barrier": None}}}
    return RP.receipt_from_plan(plan, {}, blobs, {"op_id": window, "owner": "window"}, now=lambda: now)


def test_a_hold_recorded_after_the_sidecar_export_and_before_the_plan_is_absent_from_holds_seen_and_the_adapter_stays_unread_until_a_later_export_carries_it(tmp_path):
    """Closing pass of the review of registry-hardening 3a: the tick exports the shard's reader-state sidecar FIRST and
    builds the plan after it; round 3 had the plan read the harvester's standing retirement holds off the LIVE database between the two
    steps. Another writer accepts a newer head of the source and the harvester holds the retirement in that interval: the plan named the
    hold, but the shard reads the sidecar -- the earlier head and its watermarks -- and sees an unchanged present input, so it emits no
    claim, yet its attempted, otherwise whole receipt named the hold, `held_retirements_unread` cleared it, and unchanged pins and module let
    every later tick skip the adapter while the newer head's measurements stayed current after its input vanished. Now the plan reads the
    holds off the SIDECAR it exported (the shard's snapshot, `plan_window(sidecar=...)`, `cluster-reads plan --reader-state`): a hold
    recorded after the export is absent from this window's `holds_seen`, the receipt names it nowhere, and the adapter stays unread until a
    window whose export carries the hold attempts it. Driven in the tick's order: export, then the hold (the pull merges R, the fold holds it
    against the moved head), then the plan, the receipt, the tick's standing ids and changed set."""
    from sisteer.registry import reads_protocol as RP
    w = _held_retirement_world(tmp_path)
    root, home, stores, rev, sid3, s3_rev2 = w["root"], w["home"], w["stores"], w["rev"], w["sid3"], w["s3_rev2"]
    rev_dir = f"{ADAPTER}/{rev}"
    held_rows = lambda: _rows(root, "SELECT missing_parent_id, record_uri FROM blocked_records WHERE json_extract(extra_json, '$.revision_id') = ? "
                                    "AND json_extract(extra_json, '$.kind') = ?", (rev, H.RETIREMENT_HEAD_KIND))
    # 1. the tick's export: the sidecar for this window's shards, built while no hold stands
    sidecar_a = str(tmp_path / "rs_a.sqlite")
    _export(root, sidecar_a, home)
    assert CR.holds_seen_of(sidecar_a, [ADAPTER]) == {ADAPTER: []} and held_rows() == []
    # 2. between the export and the plan: the pull merges R and the harvester HOLDS it -- the head it was computed against (H1) is not the
    # live head (H2) -- so the live database now lists a hold the sidecar never saw
    _pull_held_retirement(w, tmp_path, "wa")
    (hold,) = held_rows()
    assert hold[0] == s3_rev2 and f"/adapters/{rev_dir}/" in hold[1]
    with DB.reading(DB.db_path(root)) as conn:
        holds = H.retirements_held_for_moved_head(conn, ADAPTER)
    assert [(h["revision_id"], h["source_id"], h["record_uri"], h["blocking_head"]) for h in holds] == [(rev, sid3, hold[1], s3_rev2)] and holds[0]["first_seen_utc"]
    # 3. the plan, off the sidecar: NO hold seen -- the snapshot predates it (round 3 read the live database here and named it); the plan
    # names the snapshot it read
    cfgp = TS._cfg(tmp_path, {"synthetic_stores": {"base": stores, "authority": "grading_host_copy", "origin": "grading_host"}},
                   adapters={ADAPTER: {"shards": 3, "components": ["synthetic_stores"]}})
    cfg = ST.load_config(cfgp)
    plan_a = CR.plan_window(root, "wa", [ADAPTER], cfg, cfg.destination("cluster_a"), "a" * 40, uri_home=home, published={}, sidecar=sidecar_a)
    meta_a = RS.meta(RS.open_reader_state(sidecar_a))
    assert plan_a["adapters"][ADAPTER]["holds_seen"] == []
    assert plan_a["holds_seen_from"] == {"sidecar": sidecar_a, "built_utc": meta_a["built_utc"], "ledger_head": meta_a["ledger_head"]} and meta_a["ledger_head"]
    # 4. the receipt: attempted (every shard ran), whole (R delivered, nothing refused), naming no hold -- the hold is standing unread and the
    # next tick reads the adapter again, unchanged pins and module notwithstanding
    blobs = {ADAPTER: {"src/x.py": "b1"}}
    w_a = _attempted_receipt("wa", plan_a["adapters"][ADAPTER]["holds_seen"], plan_a["holds_seen_from"], synth.t(3310), blobs, pulled_dirs=[rev_dir])
    e_a = w_a["adapters"][ADAPTER]
    assert e_a["read_whole"] is True and RP.attempted(e_a) and e_a["holds_seen"] == [] and w_a["holds_seen_from"] == plan_a["holds_seen_from"]
    rd = str(tmp_path / "reads")
    os.makedirs(rd)
    RP.write_receipt(rd, w_a)
    assert RP.held_retirements_unread(ADAPTER, [w_a], holds) == [rev] and RP.standing_ids(root, rd, ADAPTER) == [rev]
    probed = {"components": {"synthetic_stores": {"current": "deadbeef", "exists": True}}}
    cs = RP.changed_set(cfg, [ADAPTER], probed, blobs, lambda a: RP.last_receipt_for(rd, a), standing=lambda a: RP.standing_ids(root, rd, a))
    assert cs["unchanged"] == [] and cs["changed"][ADAPTER] == [f"refused-known id(s) left standing unread by the last pull: {rev} -- read fresh on this tree"], cs
    # 5. the next window's export carries the hold: its plan names it by its whole identity, and its attempted receipt ends it
    sidecar_b = str(tmp_path / "rs_b.sqlite")
    _export(root, sidecar_b, home)
    seen_b = CR.holds_seen_of(sidecar_b, [ADAPTER])
    assert seen_b == {ADAPTER: [{"revision_id": rev, "record_uri": hold[1], "blocking_head": s3_rev2}]}
    plan_b = CR.plan_window(root, "wb", [ADAPTER], cfg, cfg.destination("cluster_a"), "a" * 40, uri_home=home, published={}, sidecar=sidecar_b)
    assert plan_b["adapters"][ADAPTER]["holds_seen"] == seen_b[ADAPTER] and plan_b["holds_seen_from"]["sidecar"] == sidecar_b
    w_b = _attempted_receipt("wb", plan_b["adapters"][ADAPTER]["holds_seen"], plan_b["holds_seen_from"], synth.t(3410), blobs)
    RP.write_receipt(rd, w_b)
    assert RP.held_retirements_unread(ADAPTER, [w_b, w_a], holds) == [] and RP.standing_ids(root, rd, ADAPTER) == []
    cs_b = RP.changed_set(cfg, [ADAPTER], probed, blobs, lambda a: RP.last_receipt_for(rd, a), standing=lambda a: RP.standing_ids(root, rd, a))
    assert cs_b["changed"] == {} and cs_b["unchanged"] == [ADAPTER]
    # 6. the CLI's plan reads the same sidecar (`--reader-state`, what the landing window passes); without one a plan vouches for nothing
    assert CLI.main(["--root", root, "cluster-reads", "plan", "--window", "wcli", "--adapter", ADAPTER, "--tree-sha", "a" * 40, "--config", cfgp,
                     "--dest", "cluster_a", "--repo-root", REPO_ROOT, "--reader-state", sidecar_b, "--json"]) == 0
    with open(CR.plan_path(root, "wcli")) as fh:
        plan_cli = json.load(fh)
    assert plan_cli["adapters"][ADAPTER]["holds_seen"] == seen_b[ADAPTER] and plan_cli["holds_seen_from"]["sidecar"] == sidecar_b
    plan_n = CR.plan_window(root, "wn", [ADAPTER], cfg, cfg.destination("cluster_a"), "a" * 40, uri_home=home, published={})
    assert plan_n["adapters"][ADAPTER]["holds_seen"] == [] and plan_n["holds_seen_from"] is None
    # 7. a sidecar that is missing, or was exported for another adapter only, refuses the plan by name rather than recording nothing seen
    with pytest.raises(CR.ClusterReadsError, match="could not be opened"):
        CR.holds_seen_of(str(tmp_path / "nope.sqlite"), [ADAPTER])
    other = str(tmp_path / "rs_other.sqlite")
    _export(root, other, home, adapters=("relu_paired",))
    with pytest.raises(CR.ClusterReadsError, match="not for"):
        CR.holds_seen_of(other, [ADAPTER])
    with pytest.raises(CR.ClusterReadsError):
        CR.plan_window(root, "wx", [ADAPTER], cfg, cfg.destination("cluster_a"), "a" * 40, uri_home=home, published={}, sidecar=other)


def test_the_same_held_carrier_re_held_against_a_newer_head_is_a_new_hold_unread_until_a_receipt_names_that_head(tmp_path):
    """Closing pass of the review of registry-hardening 3a: an attempted receipt names a hold for good. When the head H2 a
    retirement was held against is superseded by H3, `_held_still_blocked` re-attempts the same carrier (past the held-attempt memo's sweep
    interval, `held_sweep_s`), the gate holds it against H3 and `_block` rewrites the SAME queue row's `missing_parent_id` -- the carrier's uri
    never changes -- so, named by revision and carrier alone, the new hold read as already seen by the receipt that had named the H2 hold.
    With H3 a delayed fold (observed between H2 and R, folded after R was held, so R is not superseded and the re-attempt runs) and the
    group's inputs gone from the staged copy, no later tick retried the retirement under unchanged pins and module, and H3's measurements
    stayed current. Now the head the hold was computed against is part of the hold's identity everywhere it is recorded and compared
    (`harvest.RETIREMENT_HOLD_IDENTITY`): `retirements_held_for_moved_head` returns it, the sidecar-read `holds_seen` and the receipt carry
    it, `held_retirements_unread` compares the triple -- the H3 hold stands unread until a window whose export carries it attempts the
    adapter, and that window's shard claims against H3, which the pull re-activates and the fold accepts."""
    from sisteer.registry import reads_protocol as RP
    w = _held_retirement_world(tmp_path)
    root, home, stores, rev, sid3, s3_rev, s3_rev2 = w["root"], w["home"], w["stores"], w["rev"], w["sid3"], w["s3_rev"], w["s3_rev2"]
    s3_dir, s3_uri, manifest, s3_bytes, pattern = w["s3_dir"], w["s3_uri"], w["manifest"], w["s3_bytes"], w["pattern"]
    notes = os.path.join(s3_dir, "notes.json")
    head = lambda: _rows(root, "SELECT revision_id FROM source_revision WHERE source_id = ? AND accepted = 1 ORDER BY observed_utc DESC LIMIT 1", (sid3,))
    current = lambda: _rows(root, "SELECT COUNT(*) FROM observation WHERE source_id = ? AND current = 1", (sid3,))[0][0]
    live_marks = lambda: _rows(root, "SELECT input_uri FROM source_watermark WHERE source_id = ? AND last_error IS NULL ORDER BY input_uri", (sid3,))
    held_rows = lambda: _rows(root, "SELECT missing_parent_id, record_uri, attempts FROM blocked_records WHERE json_extract(extra_json, '$.revision_id') = ? "
                                    "AND json_extract(extra_json, '$.kind') = ?", (rev, H.RETIREMENT_HEAD_KIND))
    # 1. H3, the delayed fold: the store comes back with a third file beside it and the operator's pass publishes H3, OBSERVED between H2 and
    # R (the pass stamps it so); its directory is set aside -- it folds only after R is held -- and the store vanishes again
    os.makedirs(s3_dir)
    with open(s3_uri, "wb") as fh:
        fh.write(s3_bytes)
    with open(manifest, "w") as fh:
        json.dump({"schema": "manifest/1"}, fh)
    with open(notes, "w") as fh:
        json.dump({"note": "a third input"}, fh)
    ad3 = ThreeFileStoreAdapter(stores)
    with DB.reading(DB.db_path(root)) as conn:
        rep3 = RUN.run_pass(root, [ad3], conn=conn, now=synth.t(3175), producer=PRODUCER)
    res3 = next(r for r in rep3.results if r.group == "s3")
    assert res3.accepted and not res3.reactivated and res3.revision_id not in (s3_rev, s3_rev2, rev), res3.error
    s3_rev3 = res3.revision_id
    h3_dir = os.path.join(H.records_dir(root), "adapters", ADAPTER, s3_rev3)
    aside = str(tmp_path / "h3_aside")
    shutil.move(h3_dir, aside)
    shutil.rmtree(s3_dir)
    # 2. R is pulled and HELD against H2; a window whose export carries that hold attempts the adapter and its receipt names it: nothing standing
    _pull_held_retirement(w, tmp_path, "wh2")
    (hold2,) = held_rows()
    assert hold2[0] == s3_rev2 and hold2[2] == 1 and f"/adapters/{ADAPTER}/{rev}/" in hold2[1]
    sidecar_1 = str(tmp_path / "rs_1.sqlite")
    _export(root, sidecar_1, home)
    seen_1 = CR.holds_seen_of(sidecar_1, [ADAPTER])[ADAPTER]
    assert seen_1 == [{"revision_id": rev, "record_uri": hold2[1], "blocking_head": s3_rev2}]
    blobs = {ADAPTER: {"src/x.py": "b1"}}
    w_1 = _attempted_receipt("wh2", seen_1, CR.holds_seen_source(sidecar_1), synth.t(3310), blobs)
    rd = str(tmp_path / "reads")
    os.makedirs(rd)
    RP.write_receipt(rd, w_1)
    assert RP.standing_ids(root, rd, ADAPTER) == []
    # 3. H3 folds: the head; H2 superseded by it (the fold links the chain), R not (no accepted revision was observed after it) -- so the same
    # cycle's re-evaluation RE-ATTEMPTS R (the memo's sweep interval past: held_sweep_s = 0), the gate holds it against H3, and the same queue
    # row -- the same sha, the same carrier -- now names H3, its attempt counted
    shutil.move(aside, h3_dir)
    ing3 = H.ingest_once(root, workers=0, min_free=0, held_sweep_s=0)
    assert ing3.refused == 0 and head() == [(s3_rev3,)] and live_marks() == [(manifest,), (notes,), (s3_uri,)], (ing3.refusals, ing3.blocked)
    assert _rows(root, "SELECT superseded_by_revision_id FROM source_revision WHERE revision_id = ?", (s3_rev2,)) == [(s3_rev3,)]
    assert _rows(root, "SELECT accepted, superseded_by_revision_id FROM source_revision WHERE revision_id = ?", (rev,)) == [(0, None)]
    (hold3,) = held_rows()
    assert hold3[0] == s3_rev3 and hold3[1] == hold2[1] and hold3[2] == 2, (hold2, hold3)
    ((refusal,),) = _rows(root, "SELECT refusal FROM source_revision WHERE revision_id = ?", (rev,))
    assert f"activation pending: source_revision {s3_rev3} ({H.RETIREMENT_HEAD_KEY}: the retirement {rev} of {sid3} was computed against the accepted head {s3_rev}, " \
           f"and the live accepted head is {s3_rev3};" in refusal, refusal
    n_current = current()
    assert n_current > 0 and not os.path.exists(s3_dir)           # H3's measurements current while its inputs are gone: what the retry must fix
    with DB.reading(DB.db_path(root)) as conn:
        holds3 = H.retirements_held_for_moved_head(conn, ADAPTER)
    assert [(h["revision_id"], h["source_id"], h["record_uri"], h["blocking_head"]) for h in holds3] == [(rev, sid3, hold2[1], s3_rev3)]
    # 4. the reads side: the receipt named the H2 hold; the H3 hold is NEW -- standing unread again under unchanged pins and module. (Named by
    # revision and carrier alone it matched that receipt, nothing stood, and the tick skipped the adapter.)
    assert RP.held_retirements_unread(ADAPTER, [w_1], holds3) == [rev] and RP.standing_ids(root, rd, ADAPTER) == [rev]
    cfgp = TS._cfg(tmp_path, {"synthetic_stores": {"base": stores, "authority": "grading_host_copy", "origin": "grading_host"}},
                   adapters={ADAPTER: {"shards": 3, "components": ["synthetic_stores"]}})
    cfg = ST.load_config(cfgp)
    probed = {"components": {"synthetic_stores": {"current": "deadbeef", "exists": True}}}
    cs = RP.changed_set(cfg, [ADAPTER], probed, blobs, lambda a: RP.last_receipt_for(rd, a), standing=lambda a: RP.standing_ids(root, rd, a))
    assert cs["unchanged"] == [] and cs["changed"][ADAPTER] == [f"refused-known id(s) left standing unread by the last pull: {rev} -- read fresh on this tree"], cs
    # 5. the next window: its export carries the H3 hold; its shard, reading against H3, sees the group gone -- all three inputs declared --
    # and refuses the same empty id as known with a claim against H3; the pull re-activates it and the fold accepts: R is the head, the three
    # inputs retired, nothing standing; the receipt naming the H3 hold is what ends it, where the one naming the H2 hold does not
    ad3.inputs_retired = [pattern]
    sidecar_2 = str(tmp_path / "rs_2.sqlite")
    _export(root, sidecar_2, home)
    seen_2 = CR.holds_seen_of(sidecar_2, [ADAPTER])[ADAPTER]
    assert seen_2 == [{"revision_id": rev, "record_uri": hold2[1], "blocking_head": s3_rev3}]
    wroot2, cluster_root2, reporting2 = _declared_retirement_window(tmp_path, ad3, sidecar_2, synth.t(3400), name="wroot_h3")
    ((owner2, rep4),) = reporting2
    (res4,) = rep4.results
    claim = {"outgoing_revision_id": s3_rev3, "inputs": [manifest, notes, s3_uri], "patterns": [pattern]}
    assert res4.refused_known and res4.revision_id == rev and res4.retirement == claim and not _published(cluster_root2), res4.error
    pulled2, out2 = _retirement_pull(root, tmp_path, wroot2, "wh3", "expect_h3", synth.t(3500), ad3)
    assert pulled2["refused_known_claims"] == {ADAPTER: {rev: claim}} and out2[ADAPTER][rev]["state"] == "reactivated" and out2[ADAPTER][rev]["withdrawals"] > 0, out2
    ing4 = H.ingest_once(root, workers=0, min_free=0)
    assert ing4.refused == 0 and ing4.revisions_held_retirement_head == 0 and head() == [(rev,)] and current() == 0 and live_marks() == [], (ing4.refusals, ing4.blocked)
    assert _rows(root, "SELECT DISTINCT reason FROM withdrawal WHERE to_revision_id = ? AND from_revision_id = ?", (rev, s3_rev3)) == [("input_vanished",)]
    if held_rows():
        H.ingest_once(root, workers=0, min_free=0)          # the superseded holds stand down on the next re-evaluation
    assert held_rows() == []
    w_2 = _attempted_receipt("wh3", seen_2, CR.holds_seen_source(sidecar_2), synth.t(3510), blobs, reactivations={ADAPTER: {rev: "reactivated"}})
    RP.write_receipt(rd, w_2)
    assert RP.held_retirements_unread(ADAPTER, [w_2, w_1], holds3) == [] and RP.held_retirements_unread(ADAPTER, [w_1], holds3) == [rev]
    assert RP.standing_ids(root, rd, ADAPTER) == []


def test_a_grading_host_retirement_of_a_known_empty_id_is_published_as_a_bundle_never_a_second_marker_in_its_directory(tmp_path):
    """Round 2 of the review of registry-hardening 3a, the grading host half: a retired group's store comes back (its revision re-activated, the
    measurements current again) and vanishes once more; the grading host pass (the operator's override, no shard) mints the SAME empty id, whose
    revision directory is already on the record log. Before this the retirement was written into that directory again -- the marker file
    is append-only bytes, so the second write stacked a second marker record on the first. Now a known id's retirement goes out as a
    marker-only ACTIVATION BUNDLE computed against today's head, with its premise (`extra.retirement_expects`), the directory's bytes
    untouched; while that bundle's fold is pending on the log another pass writes nothing (`held`, pending); the fold makes the empty
    revision the head again."""
    root, home, stores, ad = _grading_host(tmp_path)
    sid3 = f"{ADAPTER}:s3"
    s3_dir, s3_uri = os.path.join(stores, "s3"), os.path.join(stores, "s3", "store.json")
    ((s3_rev,),) = _rows(root, "SELECT revision_id FROM source_revision WHERE source_id = ? AND accepted = 1", (sid3,))
    s3_bytes = open(s3_uri, "rb").read()
    pattern = os.path.join(stores, "s3", "*.json")
    ad.inputs_retired = [pattern]
    head = lambda: _rows(root, "SELECT revision_id FROM source_revision WHERE source_id = ? AND accepted = 1 ORDER BY observed_utc DESC LIMIT 1", (sid3,))
    current = lambda: _rows(root, "SELECT COUNT(*) FROM observation WHERE source_id = ? AND current = 1", (sid3,))[0][0]
    # 1. the first retirement on the grading host: the directory, with its premise; folded, the empty revision is the head
    shutil.rmtree(s3_dir)
    with DB.reading(DB.db_path(root)) as conn:
        rep1 = RUN.run_pass(root, [ad], conn=conn, now=synth.t(3000), producer=PRODUCER)
    res1 = next(r for r in rep1.results if r.group == "s3")
    rev = res1.revision_id
    rev_dir_abs = os.path.join(H.records_dir(root), "adapters", ADAPTER, rev)
    (marker_file,) = os.listdir(rev_dir_abs)
    assert res1.accepted and not res1.reactivated and res1.files == [os.path.join(rev_dir_abs, marker_file)]
    marker_bytes = open(os.path.join(rev_dir_abs, marker_file), "rb").read()
    assert _marker(rev_dir_abs)["payload"]["extra"][H.RETIREMENT_EXPECTS_KEY] == {"outgoing_revision_id": s3_rev, "inputs": [s3_uri]}
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0 and head() == [(rev,)] and current() == 0
    # 2. the store comes back byte-identical: its revision re-activates, the measurements are current again
    os.makedirs(s3_dir)
    with open(s3_uri, "wb") as fh:
        fh.write(s3_bytes)
    with DB.reading(DB.db_path(root)) as conn:
        RUN.run_pass(root, [ad], conn=conn, now=synth.t(3100), producer=PRODUCER)
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0 and head() == [(s3_rev,)] and current() > 0
    # 3. it vanishes again: the grading host pass publishes the known id as a BUNDLE -- the directory's marker file is byte-identical to before
    shutil.rmtree(s3_dir)
    with DB.reading(DB.db_path(root)) as conn:
        rep3 = RUN.run_pass(root, [ad], conn=conn, now=synth.t(3200), producer=PRODUCER)
    res3 = next(r for r in rep3.results if r.group == "s3")
    assert res3.accepted and res3.reactivated and res3.revision_id == rev and res3.withdrawals > 0, res3.error
    (bundle,) = RUN.complete_bundles(root, ADAPTER, rev)
    assert res3.files and all(f.startswith(os.path.dirname(bundle[1])) for f in res3.files) and os.listdir(os.path.dirname(bundle[1])) == [os.path.basename(bundle[1])]
    assert os.listdir(rev_dir_abs) == [marker_file] and open(os.path.join(rev_dir_abs, marker_file), "rb").read() == marker_bytes
    recs = _records_by_type(os.path.dirname(bundle[1]))
    marker = recs["source_revision"][0]
    assert marker["inputs_json"] == {} and marker["revision_id"] == rev and marker["observed_utc"] == synth.t(3200)
    assert marker["extra"][H.RETIREMENT_EXPECTS_KEY] == {"outgoing_revision_id": s3_rev, "inputs": [s3_uri]}
    assert marker["extra"][RUN.RETIREMENT_DECLARED_KEY] == {"declared_by": "adapter:inputs_retired", "patterns": [pattern], "inputs": [s3_uri]}
    assert "reactivated_from_committed_bytes" not in marker["extra"]          # a fresh retirement, not a replay of committed bytes
    assert recs["withdrawal"] and all(w["reason"] == "input_vanished" and w["from_revision_id"] == s3_rev and w["to_revision_id"] == rev for w in recs["withdrawal"])
    # 4. while the bundle's fold is pending on the log, another pass writes nothing for it
    with DB.reading(DB.db_path(root)) as conn:
        rep4 = RUN.run_pass(root, [ad], conn=conn, now=synth.t(3250), producer=PRODUCER)
    res4 = next(r for r in rep4.results if r.group == "s3")
    assert res4.held and res4.activation_pending and not res4.accepted and res4.files == [] and "not folded yet" in res4.error, res4.error
    assert len(RUN.complete_bundles(root, ADAPTER, rev)) == 1 and open(os.path.join(rev_dir_abs, marker_file), "rb").read() == marker_bytes
    # 5. the fold: the empty revision is the head again, the keys withdrawn, the watermark retired
    ing = H.ingest_once(root, workers=0, min_free=0)
    assert ing.refused == 0 and ing.blocked == 0 and ing.revisions_held == 0, (ing.refusals, ing.blocked)
    assert head() == [(rev,)] and current() == 0
    assert _rows(root, "SELECT last_error FROM source_watermark WHERE source_id = ? AND input_uri = ?", (sid3, s3_uri)) == [(RUN.VANISHED_PREFIX + s3_uri,)]
    with DB.reading(DB.db_path(root)) as conn:
        rep5 = RUN.run_pass(root, [ad], conn=conn, now=synth.t(3300), producer=PRODUCER)
    assert not [r for r in rep5.results if r.group == "s3"]           # idempotent: the empty revision is current


class StoreListingRetirementAdapter(SyntheticStoreAdapter):
    """The synthetic store adapter whose `inputs_retired` LISTS the store root to answer, as production declarations do (`gsm8k_p200` plans
    its tree, `degeneracy_union` and the cellstore readers list their directories): fine on a shard, which installs a store access scope,
    and refused by name (`GradingHostStoreReadRefused`) on the unscoped grading host."""

    def __init__(self, store_root: str):
        super().__init__(store_root)
        self.retired_asked = 0

    def inputs_retired(self, group: str) -> List[str]:
        self.retired_asked += 1
        AB.store_listdir(self.roots[0])            # the production shape: the declaration is derived from the store tree
        return [os.path.join(self.roots[0], group, "*.json")]


def test_a_retirement_re_activated_by_the_pull_asks_the_adapter_nothing_and_reads_no_store_on_the_grading_host(tmp_path):
    """Round 1 of the review of registry-hardening 3a (MEDIUM): the re-activation of a known retirement called `_retirement_records`, which
    asked the adapter's `inputs_retired` -- a callback that lists or opens the store in production adapters -- on the grading host, where no store
    access scope is installed: `GradingHostStoreReadRefused` escaped `reactivate_refused_known` and aborted the pull. The declaration now travels
    with the shard's claim and the grading host asks the adapter nothing: the pull on an unscoped host, the audit hook on the store root,
    re-activates the retirement (the marker's declaration is the claim's), the audit log stays empty and the adapter's callback count does
    not move."""
    root, home, stores, _unused = _grading_host(tmp_path)
    ad = StoreListingRetirementAdapter(stores)
    sid3 = f"{ADAPTER}:s3"
    s3_dir, s3_uri = os.path.join(stores, "s3"), os.path.join(stores, "s3", "store.json")
    pattern = os.path.join(stores, "s3", "*.json")
    s3_bytes = open(s3_uri, "rb").read()
    head = lambda: _rows(root, "SELECT revision_id FROM source_revision WHERE source_id = ? AND accepted = 1 ORDER BY observed_utc DESC LIMIT 1", (sid3,))
    # the empty id becomes KNOWN (the suite's override scope admits the grading-host-side declaration here; the shard installs its own scope)
    shutil.rmtree(s3_dir)
    with DB.reading(DB.db_path(root)) as conn:
        rep1 = RUN.run_pass(root, [ad], conn=conn, now=synth.t(3000), producer=PRODUCER)
    rev = next(r.revision_id for r in rep1.results if r.group == "s3")
    assert ad.retired_asked > 0 and H.ingest_once(root, workers=0, min_free=0).refused == 0 and head() == [(rev,)]
    os.makedirs(s3_dir)
    with open(s3_uri, "wb") as fh:
        fh.write(s3_bytes)
    with DB.reading(DB.db_path(root)) as conn:
        RUN.run_pass(root, [ad], conn=conn, now=synth.t(3100), producer=PRODUCER)
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0 and head() != [(rev,)]
    (s3_rev,) = head()[0]
    shutil.rmtree(s3_dir)
    sidecar = str(tmp_path / "rs.sqlite")
    _export(root, sidecar, home)
    wroot, cluster_root, reporting = _declared_retirement_window(tmp_path, ad, sidecar, synth.t(3200), name="wroot_listing")
    ((owner, rep3),) = reporting
    (res3,) = rep3.results
    assert res3.refused_known and res3.revision_id == rev and res3.retirement == {"outgoing_revision_id": s3_rev, "inputs": [s3_uri], "patterns": [pattern]}
    # the pull on an UNSCOPED host with the audit hook on the store root: re-activated, no store byte or listing, the callback never asked
    asked_before = ad.retired_asked
    audit = tmp_path / "audit.jsonl"
    AB.install_store_audit(str(audit), [stores])
    try:
        with AB.without_store_access_scope():
            pulled, out = _retirement_pull(root, tmp_path, wroot, "w1", "expect_listing", synth.t(3300), ad)
    finally:
        AB.disable_store_audit()
    assert not audit.exists() or audit.read_text() == "", audit.read_text() if audit.exists() else ""
    assert ad.retired_asked == asked_before, "the grading host asked the adapter for its declaration"
    assert out[ADAPTER][rev]["state"] == "reactivated" and out[ADAPTER][rev]["withdrawals"] > 0 and pulled["adapters_refused"] == {}, out
    (bundle,) = RUN.complete_bundles(root, ADAPTER, rev)
    marker = _records_by_type(os.path.dirname(bundle[1]))["source_revision"][0]
    assert marker["extra"][RUN.RETIREMENT_DECLARED_KEY] == {"declared_by": "adapter:inputs_retired", "patterns": [pattern], "inputs": [s3_uri]}
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0 and head() == [(rev,)]


def test_cluster_shard_still_refuses_a_vanished_group_whose_declaration_covers_only_some_of_its_inputs(tmp_path):
    """registry-hardening 3a, defect 1 (the other half): a declaration that covers SOME of a vanished group's inputs is no declaration of
    the retirement -- the owning shard refuses by name as before, the refusal naming the UNDECLARED input with its covering component and
    counting the declared one; nothing is published, the summary lists it refused. Completed (an exact URI and a glob: both match
    branches), the same shard commits the retirement and the marker names both patterns."""
    root, home, stores, ad = _grading_host(tmp_path)
    s3_uri = os.path.join(stores, "s3", "store.json")
    manifest = os.path.join(stores, "s3", "manifest.json")
    with open(manifest, "w") as fh:
        json.dump({"schema": "manifest/1"}, fh)
    ad2 = TwoFileStoreAdapter(stores)                 # s3 now has two inputs and its head reads both (the live two-input shape)
    with DB.reading(DB.db_path(root)) as conn:
        rep0 = RUN.run_pass(root, [ad2], conn=conn, now=synth.t(2500), producer=PRODUCER)
    assert rep0.groups_refreshed == 1 and H.ingest_once(root, workers=0, min_free=0).refused == 0
    assert _rows(root, "SELECT input_uri FROM source_watermark WHERE source_id = ? ORDER BY input_uri", (f"{ADAPTER}:s3",)) == [(manifest,), (s3_uri,)]
    shutil.rmtree(os.path.join(stores, "s3"))
    ad2.inputs_retired = [s3_uri]                     # the store declared, the manifest not
    sidecar = str(tmp_path / "rs.sqlite")
    _export(root, sidecar, home)
    wroot, cluster_root, reporting = _declared_retirement_window(tmp_path, ad2, sidecar, synth.t(3000), name="wroot_partial")
    assert len(reporting) == 1, reporting
    i, rep = reporting[0]
    (res,) = rep.results
    assert not res.accepted and res.inputs_dropped and res.files == [] and rep.groups_refused == 1 and rep.groups_inputs_dropped == 1
    assert not _published(cluster_root) and not os.path.isdir(os.path.join(H.records_dir(cluster_root), "adapters"))
    assert res.error.startswith(f"{ADAPTER}:s3: {RUN.STAGED_COPY_INCOMPLETE_KEY}: every input of the group (2) is absent from this read on the cluster")
    assert "a retirement is declared by the adapter's inputs_retired, else a retirement is the grading host's -- nothing published here." in res.error
    assert res.error.endswith(f"Missing (1 of 2 declared retired by inputs_retired; 1 of 1 undeclared named): {manifest} [no staged component covers it]")
    assert s3_uri + " [" not in res.error             # the declared input is not among the missing ones named
    doc = json.load(open(CR.summary_path(str(wroot), ADAPTER, i, 3)))
    assert doc["published"] == {} and [r["revision_id"] for r in doc["refused"]] == [res.revision_id] and doc["refused"][0]["inputs_dropped"] is True
    # the declaration completed: the same shard commits the retirement, both patterns on the marker
    glob_pattern = os.path.join(stores, "*", "manifest.json")
    ad2.inputs_retired = [s3_uri, glob_pattern]
    cluster_root2 = _cluster_root(tmp_path, "cluster_root_full")
    rep2 = _cluster_pass(cluster_root2, ad2, sidecar, synth.t(3050), shard=(i, 3))
    (res2,) = rep2.results
    assert res2.accepted and res2.withdrawals > 0 and rep2.groups_refused == 0 and res2.revision_id == res.revision_id
    marker = _marker(os.path.join(H.records_dir(cluster_root2), "adapters", ADAPTER, res2.revision_id))["payload"]
    assert marker["extra"][RUN.RETIREMENT_DECLARED_KEY] == {"declared_by": "adapter:inputs_retired", "patterns": [glob_pattern, s3_uri], "inputs": [manifest, s3_uri]}
    assert marker["extra"][H.DECLARED_RUN_DROPS_KEY]["declared_by"] == ["runner:input_vanished", "runner:inputs_retired"]


def test_grading_host_retirement_needs_no_declaration_and_records_one_only_when_it_covers_every_vanished_input(tmp_path):
    """The grading host path (the operator-override pass) keeps retiring undeclared groups; a declaration is a FACT on its marker when the adapter
    made one that covers every vanished input, and absent otherwise, so a retirement marker says the same thing wherever it was written."""
    root, home, stores, ad = _grading_host(tmp_path)
    shutil.rmtree(os.path.join(stores, "s2"))
    shutil.rmtree(os.path.join(stores, "s3"))
    pattern = os.path.join(stores, "s3", "*.json")
    ad.inputs_retired = [pattern]                     # s3 declared, s2 not
    with DB.reading(DB.db_path(root)) as conn:
        rep = RUN.run_pass(root, [ad], conn=conn, now=synth.t(3000), producer=PRODUCER)
    by_group = {r.group: r for r in rep.results}
    assert set(by_group) == {"s2", "s3"} and all(r.accepted and r.withdrawals > 0 for r in by_group.values()), [r.error for r in rep.results]
    extras = {g: _marker(os.path.join(H.records_dir(root), "adapters", ADAPTER, r.revision_id))["payload"]["extra"] for g, r in by_group.items()}
    assert RUN.RETIREMENT_DECLARED_KEY not in extras["s2"] and extras["s2"][H.DECLARED_RUN_DROPS_KEY]["declared_by"] == ["runner:input_vanished"]
    assert extras["s3"][RUN.RETIREMENT_DECLARED_KEY] == {"declared_by": "adapter:inputs_retired", "patterns": [pattern], "inputs": [os.path.join(stores, "s3", "store.json")]}
    assert extras["s3"][H.DECLARED_RUN_DROPS_KEY]["declared_by"] == ["runner:input_vanished", "runner:inputs_retired"]
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0


def test_cluster_mode_refuses_reread_and_the_cli_wires_the_flags(tmp_path, monkeypatch, capsys):
    root, home, stores, ad = _grading_host(tmp_path)
    sidecar = str(tmp_path / "rs.sqlite")
    _export(root, sidecar, home)
    with pytest.raises(ValueError, match="cannot --reread"):
        _cluster_pass(str(tmp_path / "c"), ad, sidecar, synth.t(3000), reread=True)
    monkeypatch.setattr("sisteer.registry.adapters.load_all", lambda: {ADAPTER: lambda: ad})
    cluster_root = _cluster_root(tmp_path)
    assert CLI.main(["--root", cluster_root, "adapters", "run", "--adapter", ADAPTER, "--reader-state", sidecar, "--reread"]) == 2
    assert "reactivation is the grading host's" in capsys.readouterr().err
    assert CLI.main(["--root", cluster_root, "adapters", "run", "--adapter", ADAPTER, "--reader-state", sidecar, "--shard", "3/2"]) == 2
    assert CLI.main(["--root", cluster_root, "adapters", "run", "--adapter", ADAPTER, "--reader-state", str(tmp_path / "nope.sqlite")]) == 2
    _store(str(stores), "s2", "legacy_store_2", "none", [0.1, 0.9, 0.5])
    summary_out = str(tmp_path / "summary.json")
    rc = CLI.main(["--root", cluster_root, "adapters", "run", "--adapter", ADAPTER, "--reader-state", sidecar, "--shard", "0/1", "--substrate", "cluster_a",
                   "--summary-out", summary_out, "--full"])
    out = capsys.readouterr()
    assert rc == 0, out.err
    lines = [l for l in out.out.splitlines() if l.strip()]
    tail = json.loads(lines[-1])
    assert tail["ingest"].startswith("skipped: --reader-state") and tail["ingest_ok"] == 0       # a shard never folds
    head = json.loads("\n".join(lines[:-1]))
    assert head["groups_refused_known"] == 2 and head["groups_refreshed"] == 1 and head["inputs_hashed"] == 3
    assert head["shard"] == {"index": 0, "count": 1} and head["reader_state"] == sidecar
    assert "refused (known to the grading host)" in out.err
    doc = json.load(open(summary_out))
    assert doc["schema"] == CR.SUMMARY_SCHEMA and doc["shard"] == {"index": 0, "count": 1} and doc["substrate"] == "cluster_a" and len(doc["published"]) == 1
    rev_dir, info = next(iter(doc["published"].items()))
    assert rev_dir.startswith(ADAPTER + "/") and info["marker"] and info["marker"] in info["files"] and set(info["files"]) == set(os.listdir(os.path.join(H.records_dir(cluster_root), "adapters", rev_dir)))
    assert len(doc["refused_known"]) == 2 and all("known to the grading host" in r["why"] for r in doc["refused_known"])


def test_ruling_records_read_through_the_absolute_rulings_symlink_the_stage_lays_and_refuse_without_it(tmp_path):
    """multiturn_grades' attestation reads a ruling ON THE RECORD LOG under <registry root>/rulings/; the window root's `rulings` is an
    ABSOLUTE symlink onto the record-log mirror that `stage_window` lays and verifies by realpath."""
    from sisteer.registry.adapters import multiturn_grades as MG
    mirror = tmp_path / "shared" / "registry" / "log" / "rulings"
    (mirror / "added").mkdir(parents=True)
    rid = "synthetic-ruling-2026-09-28"
    R.write_records(str(mirror / "added" / "000001-ruling.jsonl"), [synth.rec("ruling", {"ruling_id": rid, "kind": "supersede", "target_json": "{}"}, synth.t(10))])
    window_root = tmp_path / "shared" / "registry" / "reads" / "w1"
    root = window_root / "root"
    root.mkdir(parents=True)
    assert MG.ruling_on_record_log(str(root), rid) is None                              # no link: refused by name upstream (not on the log)
    cfg_file = tmp_path / "cluster_reads.yaml"
    cfg_file.write_text("schema: cluster_reads/1\n")
    plan = {"host": None, "window_root": str(window_root), "root": str(root), "rulings_mirror": str(mirror), "config": str(cfg_file),
            "config_sha256": R.file_sha256(str(cfg_file))}
    sidecar = tmp_path / "rs.sqlite"
    sidecar.write_bytes(b"x")
    plan_file = tmp_path / "plan.json"
    plan_file.write_text("{}")
    info = CR.stage_window(plan, str(sidecar), str(plan_file))
    link = root / "rulings"
    assert os.path.islink(link) and os.path.isabs(os.readlink(link)) and os.path.realpath(link) == os.path.realpath(mirror)
    assert info["rulings_resolved"] == str(mirror) and info["rulings_entries"] == 1 and (window_root / "reader_state.sqlite").is_file() and (window_root / "plan.json").is_file()
    assert (window_root / CR.CONFIG_STAGED).read_text() == cfg_file.read_text()          # the exact configuration travels with the plan
    found = MG.ruling_on_record_log(str(root), rid)
    assert found is not None and os.path.realpath(found) == str(mirror / "added" / "000001-ruling.jsonl") and MG.ruling_on_record_log(str(root), "other") is None
    # an empty or absent mirror refuses the stage by name; a link resolving elsewhere too
    shutil.rmtree(mirror)
    with pytest.raises(CR.ClusterReadsError, match="no rulings mirror"):
        CR.stage_window(plan, str(sidecar), str(plan_file))
    mirror.mkdir()
    with pytest.raises(CR.ClusterReadsError, match="is empty"):
        CR.stage_window(plan, str(sidecar), str(plan_file))


# --------------------------------------------------------------------------------------------
# part 3: the manifest-backed hash cache and the home tree
# --------------------------------------------------------------------------------------------


def _verified_generation(tmp_path, name, src_dir):
    """A generation under gen/<name>/<sha8> carrying src_dir's files, its APPROVED manifest (MANIFEST + VERIFIED + DEST-STAT)."""
    gen_dir = tmp_path / "gen" / name
    comp = ST.Component(name, str(src_dir), "grading_host_copy")
    m = ST.build_manifest(comp, None)
    tree = gen_dir / m.content_sha8
    shutil.copytree(src_dir, tree)
    mpath = ST.write_manifest_files(m, str(gen_dir))
    rep, dest_stat = ST.verify_generation(str(tree), mpath, workers=2, job_id="t")
    ST.write_verified(f"{gen_dir}/VERIFIED-{m.content_sha8}.json", rep, dest_stat)
    assert rep["ok"], rep
    return str(gen_dir), m.content_sha8, str(tree)


def test_generation_hash_cache_hashes_every_input_and_refuses_a_byte_mismatch_against_the_approved_manifest_by_name(tmp_path):
    """The byte guarantee: nothing under a pinned generation is taken on trust -- every input is hashed and compared with
    the manifest the VERIFIED report approved; a same-size edit under the seal refuses the input by name."""
    src = tmp_path / "src"
    (src / "a").mkdir(parents=True)
    (src / "a" / "f.json").write_bytes(b'{"k": 1}\n')
    (src / "g.json").write_bytes(b'{"k": 2}\n')
    gen_dir, sha8, tree = _verified_generation(tmp_path, "slug", src)
    cache = CR.GenerationHashCache.from_pins(os.path.dirname(gen_dir), {"slug": sha8})
    f = os.path.join(tree, "a", "f.json")
    st = os.stat(f)
    assert cache.lookup(f, st) is None and cache.reused == 0                       # never answered from the manifest
    cache.record(f, st, R.file_sha256(f))
    assert cache.checked_against_manifest == 1 and cache.refused == []
    other = tmp_path / "elsewhere.json"
    other.write_bytes(b"{}")
    cache.record(str(other), os.stat(other), R.file_sha256(str(other)))            # outside every generation: admitted, not checked
    assert cache.checked_against_manifest == 1
    os.chmod(f, 0o644)
    open(f, "wb").write(b'{"k": 9}\n')                                             # same size, other bytes
    with pytest.raises(AB.InputRefusal, match="verified-manifest check") as ei:
        cache.record(f, os.stat(f), R.file_sha256(f))
    assert "bytes differ from the verified generation" in str(ei.value) and cache.refused == [f]
    with pytest.raises(AB.InputRefusal, match="not in the verified generation"):
        extra = os.path.join(tree, "unlisted.json")
        open(extra, "wb").write(b"{}")
        cache.record(extra, os.stat(extra), R.file_sha256(extra))
    with pytest.raises(CR.ClusterReadsError, match="no_manifest"):
        CR.GenerationHashCache.from_pins(os.path.dirname(gen_dir), {"slug": "00000000"})
    pins = tmp_path / "pins.json"
    pins.write_text(json.dumps({"gen_dir": os.path.dirname(gen_dir), "pins": {"slug": sha8}}))
    assert CR.GenerationHashCache.from_pins_file(str(pins)).indexes[0][0] == tree
    pins.write_text(json.dumps({"pins": {}}))
    with pytest.raises(CR.ClusterReadsError):
        CR.GenerationHashCache.from_pins_file(str(pins))
    # end to end: a pass over a store under a pinned generation hashes every input and checks it; a corrupt file refuses the adapter's
    # discovery by name (the shard summary names it; the grading host reads the adapter)
    root, store_root = _setup(tmp_path)
    gen_dir2, sha2, tree2 = _verified_generation(tmp_path / "two", "stores", store_root)
    cache2 = CR.GenerationHashCache.from_pins(os.path.dirname(gen_dir2), {"stores": sha2})
    with DB.reading(DB.db_path(root)) as conn:
        rep = RUN.run_pass(root, [SyntheticStoreAdapter(tree2)], conn=conn, now=synth.t(2000), hash_cache=cache2, save_cache=False)
    assert rep.groups_refreshed == 1 and rep.inputs_hashed == 1 and rep.inputs_hash_reused == 0 and cache2.checked_against_manifest == 1
    assert not os.path.exists(AB.InputHashCache.path(root))          # a shard's cache is never written back
    victim = os.path.join(tree2, "s1", "store.json")
    os.chmod(victim, 0o644)
    _same_size_edit(victim)
    cache3 = CR.GenerationHashCache.from_pins(os.path.dirname(gen_dir2), {"stores": sha2})
    with DB.reading(DB.db_path(root)) as conn:
        rep3 = RUN.run_pass(root, [SyntheticStoreAdapter(tree2)], conn=conn, now=synth.t(2100), hash_cache=cache3, save_cache=False)
    assert rep3.groups_refused == 1 and rep3.results[0].group == "<discover>" and "verified-manifest check" in rep3.results[0].error and victim in rep3.results[0].error
    # the contradicted sha is never STORED: the check runs before the record (the cache holds nothing for the refused input, is not dirty)
    assert cache3.refused == [os.path.realpath(victim)] and os.path.realpath(victim) not in cache3.entries and not cache3.dirty and cache3.hashed == 1


def _components(tmp_path, home, spec):
    """`spec` = {slug: (home_rel, entries)}: a config-like components map plus a generation per slug holding `entries` (dir or file)."""
    gen_dir = tmp_path / "gen"
    comps, pins = {}, {}
    for slug, (rel, entries) in spec.items():
        sha8 = ("%08x" % (abs(hash(slug)) % (16 ** 8)))
        tree = gen_dir / slug / sha8
        tree.mkdir(parents=True)
        for e in entries:
            if e.endswith("/"):
                (tree / e.rstrip("/")).mkdir()
                (tree / e.rstrip("/") / "x.json").write_text("{}")
            else:
                (tree / e).write_text("{}")
        comps[slug] = ST.Component(slug, os.path.join(home, rel), "grading_host_copy")
        pins[slug] = sha8
    return str(gen_dir), comps, pins


def _entries_of(gen_dir):
    """A generation's top-level entries listed from the (local) tree -- what `manifest_entries` derives from the manifest on the grading host."""
    return lambda slug, sha8: sorted((e, os.path.isdir(f"{gen_dir}/{slug}/{sha8}/{e}")) for e in os.listdir(f"{gen_dir}/{slug}/{sha8}"))


def test_home_tree_spec_links_an_exclusive_base_whole_a_shared_base_entry_by_entry_copies_its_files_and_init_builds_once_per_window(tmp_path):
    home = str(tmp_path / "home")
    ST_HOME = ST.HOME
    ST.HOME = home       # Component.home_rel is computed against the module's HOME
    try:
        gen_dir, comps, pins = _components(tmp_path, home, {
            "sampled": ("rs/full22/sampled", ["grades/"]),                              # exclusive, nested under the shared base
            "multiturn_run_state": ("rs/full22", ["grades_a/", "transcripts_b/", "transfer_stats.json"]),   # shared base with misalign_run_state; a FILE member
            "misalign_run_state": ("rs/full22", ["misalign/", "misalign_16k/"]),
            "sidecars": ("specs/sweep", ["dose_ladder.json", "mm.json"]),               # exclusive: linked whole, FILE members included
        })
        spec = CR.home_tree_spec(comps, pins, gen_dir, home, _entries_of(gen_dir))
        by, files = dict(map(tuple, spec["links"])), dict(map(tuple, spec["files"]))
        full22 = os.path.join(home, "rs/full22")
        assert by[os.path.join(full22, "sampled")] == f"{gen_dir}/sampled/{pins['sampled']}"
        assert by[os.path.join(home, "specs/sweep")] == f"{gen_dir}/sidecars/{pins['sidecars']}"
        for e, slug in (("grades_a", "multiturn_run_state"), ("transcripts_b", "multiturn_run_state"), ("misalign", "misalign_run_state"), ("misalign_16k", "misalign_run_state")):
            assert by[os.path.join(full22, e)] == f"{gen_dir}/{slug}/{pins[slug]}/{e}"
        # a regular-file member of a SHARED base is a copy (a file symlink is an input refused for an adapter that admits none)
        assert files == {os.path.join(full22, "transfer_stats.json"): f"{gen_dir}/multiturn_run_state/{pins['multiturn_run_state']}/transfer_stats.json"}
        assert len(spec["links"]) == 6 and full22 in spec["dirs"] and os.path.join(home, "rs") in spec["dirs"] and full22 not in by
        # the first task of the window BUILDS; the next one VERIFIES (nothing re-pointed or created); another plan for the same window refuses
        built = CR.init_home_tree(spec, home, "w1")
        assert built["mode"] == "built" and len(built["created"]) == 6 and built["copied"] == [os.path.join(full22, "transfer_stats.json")] and not built["refused"]
        assert os.path.isdir(full22) and not os.path.islink(full22) and os.path.isfile(os.path.join(full22, "sampled/grades/x.json"))
        assert os.path.isfile(os.path.join(full22, "transfer_stats.json")) and not os.path.islink(os.path.join(full22, "transfer_stats.json"))
        again = CR.init_home_tree(spec, home, "w1")
        assert again["mode"] == "verified" and not again["refused"]
        other = CR.home_tree_spec(comps, dict(pins, sampled="ffffffff"), gen_dir, home, lambda s, g: _entries_of(gen_dir)(s, pins[s]))
        refused = CR.init_home_tree(other, home, "w1")
        assert refused["mode"] == "refused" and "another plan" in refused["refused"][0]["why"] and os.readlink(os.path.join(full22, "sampled")) == f"{gen_dir}/sampled/{pins['sampled']}"
        # a second WINDOW re-points onto a new generation and leaves an unchanged link alone
        (tmp_path / "gen" / "sampled" / "ffffffff" / "grades").mkdir(parents=True)
        built2 = CR.init_home_tree(other, home, "w2")
        assert built2["mode"] == "built" and built2["repointed"] == [os.path.join(full22, "sampled")] and len(built2["unchanged"]) == 5 and built2["previous_window"] == "w1"
        assert os.readlink(os.path.join(full22, "sampled")) == f"{gen_dir}/sampled/ffffffff"
        # a REAL path where a link should go refuses by name (never replaced)
        os.remove(os.path.join(home, "specs/sweep"))
        os.makedirs(os.path.join(home, "specs/sweep"))
        built3 = CR.init_home_tree(spec, home, "w3")
        assert [r["path"] for r in built3["refused"]] == [os.path.join(home, "specs/sweep")] and "not a symlink" in built3["refused"][0]["why"]
        # two components needing different things at one path refuse; a base outside the home refuses; an undeclared pin refuses
        gen_dir3, comps3, pins_c = _components(tmp_path / "c", home, {"p": ("rs3", ["same/"]), "q": ("rs3", ["same/"])})
        with pytest.raises(CR.ClusterReadsError, match="need different things"):
            CR.home_tree_spec(comps3, pins_c, gen_dir3, home, _entries_of(gen_dir3))
        comps_out = {"o": ST.Component("o", "/elsewhere/base", "grading_host_copy")}
        with pytest.raises(CR.ClusterReadsError, match="not under the home"):
            CR.home_tree_spec(comps_out, {"o": "00000000"}, gen_dir, home, _entries_of(gen_dir))
        with pytest.raises(CR.ClusterReadsError, match="not declared"):
            CR.home_tree_spec(comps, {"ghost": "00000000"}, gen_dir, home, _entries_of(gen_dir))
    finally:
        ST.HOME = ST_HOME


def test_the_real_configs_overlapping_bases_build_one_namespace_per_window_that_concurrent_tasks_share(tmp_path, monkeypatch):
    """On the REAL cluster_reads.yaml: run-state/full22 is multiturn_run_state's and misalign_run_state's base AND the parent of
    bfcl_full, bigcodebench_full, sampled and relu_paired_run_state. Per-adapter trees disagree about it (cap4096_lane alone would link
    it whole onto its generation, em_misalign alone onto another); the window's ONE tree from the component union makes it a real
    directory laid out entry by entry, concurrent tasks build it once, and nothing is created inside a generation."""
    from concurrent.futures import ThreadPoolExecutor
    home = str(tmp_path / "home")
    os.makedirs(home)
    monkeypatch.setenv("HOME", home)                              # load_config expands `~` from the environment
    monkeypatch.setattr(ST, "HOME", home)                         # Component.home_rel against the module's HOME
    cfg = ST.load_config(os.path.join(REPO_ROOT, "configs", "registry", "cluster_reads.yaml"))
    gen_dir = str(tmp_path / "gen")
    entries = {"multiturn_run_state": [("grades_a", True), ("transcripts_b", True), ("transfer_stats_full22_all.json", False), ("transfer_stats_nemotron_complete.json", False)],
               "misalign_run_state": [("misalign", True), ("misalign_16k", True), ("misalign_ladder", True)]}

    def entries_for(slug):
        """The generation's entries: the two stated above; else one directory per declared member (a glob made concrete by its first
        segment), or one `member` directory for a whole-base component -- so the many components sharing run-state/full22 and
        $GRADE lay out disjoint entries, as the real generations do."""
        if slug in entries:
            return entries[slug]
        out = []
        for m in cfg.components[slug].members:
            if m == ".":
                return [("member", True)]
            top = (m.replace("*", "x").split("/")[0], True)
            if top not in out:            # several members under one top-level entry (grades/gemma4e4b + grades/qwen35_4b) lay it once
                out.append(top)
        return out
    pins = {}
    for i, slug in enumerate(sorted(cfg.components)):
        pins[slug] = f"{i:08x}"
        tree = os.path.join(gen_dir, slug, pins[slug])
        for name, is_dir in entries_for(slug):
            if is_dir:
                os.makedirs(os.path.join(tree, name))
                open(os.path.join(tree, name, "x.json"), "w").write("{}")
            else:
                os.makedirs(tree, exist_ok=True)
                open(os.path.join(tree, name), "w").write("{}")
    entries_of = lambda slug, sha8: entries_for(slug)   # noqa: E731
    union = {s: pins[s] for decl in cfg.adapters.values() for s in decl.components}
    assert len(union) == len(cfg.components) >= 14           # every component is some adapter's (14 in Phase 3; 48 since C2 declared every adapter)
    spec = CR.home_tree_spec(cfg.components, union, gen_dir, home, entries_of)
    links, files = dict(map(tuple, spec["links"])), dict(map(tuple, spec["files"]))
    full22 = os.path.join(home, cfg.components["multiturn_run_state"].home_rel)     # the component bases expand through the site's values
    assert full22 in spec["dirs"] and full22 not in links                     # shared AND nesting: a real directory, never linked whole
    assert links[f"{full22}/misalign"] == f"{gen_dir}/misalign_run_state/{pins['misalign_run_state']}/misalign"
    assert links[f"{full22}/grades_a"] == f"{gen_dir}/multiturn_run_state/{pins['multiturn_run_state']}/grades_a"
    for nested, slug in (("bfcl_full", "bfcl_full"), ("bigcodebench_full", "bigcodebench_full"), ("sampled", "sampled"), ("relu_paired", "relu_paired_run_state")):
        assert links[f"{full22}/{nested}"] == f"{gen_dir}/{slug}/{pins[slug]}"          # whole links beneath the real directory
    assert files[f"{full22}/transfer_stats_full22_all.json"] == f"{gen_dir}/multiturn_run_state/{pins['multiturn_run_state']}/transfer_stats_full22_all.json"
    assert links[os.path.join(home, cfg.components["relu_paired_manifests"].home_rel)] == f"{gen_dir}/relu_paired_manifests/{pins['relu_paired_manifests']}"   # shared by two adapters: one link
    # the per-adapter trees the bug built: two different whole links for the same path (cap4096_lane pins multiturn_run_state alone and
    # em_misalign misalign_run_state alone, so each one's own tree links full22 whole onto a different generation; multiturn_grades no longer
    # serves here since it was given full22_regrade_siblings and full22_relu_native, three components on the one base)
    per = {a: CR.home_tree_spec(cfg.components, {s: pins[s] for s in decl.components}, gen_dir, home, entries_of) for a, decl in cfg.adapters.items()}
    mt, em = dict(map(tuple, per["cap4096_lane"]["links"])), dict(map(tuple, per["em_misalign"]["links"]))
    assert full22 in mt and full22 in em and mt[full22] != em[full22]
    # one task per adapter, at once: exactly one builds, the rest verify, all agree, nothing refused
    with ThreadPoolExecutor(max_workers=6) as ex:
        outs = list(ex.map(lambda _i: CR.init_home_tree(spec, home, "w1"), range(6)))
    assert sorted(o["mode"] for o in outs) == ["built"] + ["verified"] * 5 and not any(o["refused"] for o in outs)
    assert os.path.isdir(full22) and not os.path.islink(full22) and os.readlink(f"{full22}/misalign") == links[f"{full22}/misalign"]
    assert os.path.isfile(f"{full22}/transfer_stats_full22_all.json") and not os.path.islink(f"{full22}/transfer_stats_full22_all.json")
    for slug, sha8 in pins.items():                                             # no entry was created inside any generation
        assert sorted(os.listdir(os.path.join(gen_dir, slug, sha8))) == sorted(n for n, _ in entries_for(slug))
    # a task of the SAME window carrying an adapter's own tree is refused; the shared tree stands
    out = CR.init_home_tree(per["cap4096_lane"], home, "w1")
    assert out["mode"] == "refused" and "another plan" in out["refused"][0]["why"] and os.path.isdir(full22) and not os.path.islink(full22)
    # the next window re-points onto a new generation and removes what it no longer names
    os.makedirs(f"{gen_dir}/sampled/ffffffff/member")
    spec2 = CR.home_tree_spec(cfg.components, dict(union, sampled="ffffffff"), gen_dir, home, entries_of)
    out2 = CR.init_home_tree(spec2, home, "w2")
    assert out2["mode"] == "built" and out2["repointed"] == [f"{full22}/sampled"] and os.readlink(f"{full22}/sampled") == f"{gen_dir}/sampled/ffffffff"


# --------------------------------------------------------------------------------------------
# part 3: the shard task end to end over a real published generation, the transport, the plan and the fallback
# --------------------------------------------------------------------------------------------

import test_registry_stores as TS  # noqa: E402


def _world_with_generation(tmp_path, monkeypatch):
    """A grading host (registry root + the synthetic stores at `<home>/stores`, first pass folded), a cluster_reads.yaml declaring the adapter
    `synthetic_store` over the component `synthetic_stores` (base `<home>/stores`, K = 2) with a LOCAL destination, and the component
    mirrored as a real published generation (stores.mirror_component under the stores test's fake sbatch / watcher)."""
    root, home, stores, ad = _grading_host(tmp_path)
    monkeypatch.setattr(ST, "HOME", home)                         # Component.home_rel is relative to the module's HOME
    cfgp = TS._cfg(tmp_path, {"synthetic_stores": {"base": stores, "authority": "grading_host_copy", "origin": "grading_host"}},
                   adapters={ADAPTER: {"shards": 2, "components": ["synthetic_stores"]}})
    cfg = ST.load_config(cfgp)
    dest = cfg.destination("cluster_a")
    res = ST.mirror_component(root, dest, cfg.components["synthetic_stores"], runner=TS.Runner())
    assert res.outcome == "published", res.as_dict()
    monkeypatch.setattr("sisteer.registry.adapters.load_all", lambda: {ADAPTER: lambda: SyntheticStoreAdapter(stores)})
    return {"root": root, "home": home, "stores": stores, "ad": ad, "cfg": cfg, "cfgp": cfgp, "dest": dest, "sha8": res.content_sha8,
            "gen_dir": f"{dest.stores_dir}/gen", "shared": dest.shared}


def _plan_and_stage(w, tmp_path, window="w1", adapters=(ADAPTER,), full=False, attest=True, **plan_kw):
    plan = CR.plan_window(w["root"], window, list(adapters), w["cfg"], w["dest"], "a" * 40, uri_home=w["home"], repo_root=REPO_ROOT, full=full, **plan_kw)
    plan_file = CR.plan_path(w["root"], window)
    CR._write_json(plan_file, plan)
    sidecar = str(tmp_path / f"rs-{window}.sqlite")
    _export(w["root"], sidecar, w["home"], adapters=adapters)
    w["sidecar"] = sidecar
    mirror = tmp_path / "shared" / "registry" / "log" / "rulings"
    mirror.mkdir(parents=True, exist_ok=True)
    (mirror / "seed.jsonl").write_text("")
    info = CR.stage_window(plan, sidecar, plan_file)
    if plan["tasks"] and attest:
        # the operation's attestation (C5): what attest.sbatch runs before the array; every shard checks its record in O(1)
        rec = ST.attest_plan(f"{plan['window_root']}/plan.json", plan["attestation"])
        assert rec["ok"], rec
        w["attestation"] = rec
    return plan, plan_file, info


def _node(w):
    """The node: the store's canonical spelling stops being a real directory (the grading host tree moves aside) so the home tree can lay the
    link onto the generation there."""
    for p in w.get("node_moves") or [w["stores"]]:
        shutil.move(p, p + ".grading_host")


def _back_on_grading_host(w):
    """The grading host again: the home tree's link comes off and the canonical copy returns (the post-run revalidation reads it)."""
    for p in w.get("node_moves") or [w["stores"]]:
        if os.path.islink(p):
            os.remove(p)
        shutil.move(p + ".grading_host", p)


def test_run_shard_admits_lays_the_home_tree_reads_through_it_and_writes_the_summary(tmp_path, monkeypatch, capsys):
    w = _world_with_generation(tmp_path, monkeypatch)
    # a change after the first fold, BEFORE the generation is built: the generation carries it; the grading host copy still equals it (admission holds)
    s2 = _store(w["stores"], "s2", "legacy_store_2", "none", [0.1, 0.9, 0.5])
    res2 = ST.mirror_component(w["root"], w["dest"], w["cfg"].components["synthetic_stores"], runner=TS.Runner())
    assert res2.outcome == "published"
    w["sha8"] = res2.content_sha8
    plan, plan_file, info = _plan_and_stage(w, tmp_path)
    assert plan["adapters"][ADAPTER]["route"] == "cluster" and plan["adapters"][ADAPTER]["pins"] == {"synthetic_stores": w["sha8"]} and len(plan["tasks"]) == 2
    assert plan["adapters"][ADAPTER]["revalidation_pre"]["synthetic_stores"]["ok"] is True and plan["refusals"] == {}
    _node(w)
    assert CR.run_shard(f"{plan['window_root']}/plan.json", 0, home=w["home"]) == 0
    assert CR.run_shard(f"{plan['window_root']}/plan.json", 1, home=w["home"]) == 0
    link = w["stores"]
    assert os.path.islink(link) and os.readlink(link) == f"{w['gen_dir']}/synthetic_stores/{w['sha8']}"
    docs = [json.load(open(CR.summary_path(plan["summaries_dir"], ADAPTER, i, 2))) for i in range(2)]
    assert all(d["ok"] is True and d["schema"] == CR.SUMMARY_SCHEMA and d["admission_pre"]["synthetic_stores"]["ok"] and d["admission_post"]["synthetic_stores"]["ok"] for d in docs)
    # C5: the shard's admission is the O(1) check of the operation's attestation record -- no walk (the record names the full content walk
    # the attestation did), the record's path on the summary, the pre and post checks both the record's
    assert all(d["attestation"] == plan["attestation"] == f"{plan['window_root']}/{ST.ATTESTATION_FILE}" for d in docs)
    gen_rec = w["attestation"]["generations"]["synthetic_stores"]                 # the generation's own stamp, not the record's
    assert all(d["admission_pre"]["synthetic_stores"]["content_mode"] == "full" and d["admission_pre"]["synthetic_stores"]["attested_utc"] == gen_rec["utc"] for d in docs)
    assert all(d["timings_s"]["admission_pre"] < 1.0 and d["timings_s"]["admission_post"] < 1.0 and d["pass"]["inputs_escaped"] == [] for d in docs)
    assert all(d["home_tree"]["links"] == 1 for d in docs) and {d["shard"]["index"] for d in docs} == {0, 1}
    # exactly ONE shard owns the changed group s2 and published it; the other saw it as not its own; nothing was hashed (the manifest answered)
    pub = [d for d in docs if d["published"]]
    assert len(pub) == 1 and len(pub[0]["published"]) == 1 and sum(d["pass"]["groups_not_owned"] for d in docs) == 1
    assert all(d["pass"]["inputs_hashed"] == 3 and d["pass"]["inputs_hash_reused"] == 0 and d["pass"]["inputs_checked_against_manifest"] == 3 for d in docs), [d["pass"] for d in docs]
    rev_dir, pinfo = next(iter(pub[0]["published"].items()))
    assert pinfo["group"] == "s2" and pinfo["marker"] and set(pinfo["files"]) == set(os.listdir(os.path.join(plan["root"], "records", "adapters", rev_dir)))
    marker = _marker(os.path.join(plan["root"], "records", "adapters", rev_dir))
    rc = marker["payload"]["extra"]["read_copy"]
    assert rc["components"]["synthetic_stores"]["content_sha8"] == w["sha8"] and rc["components"]["synthetic_stores"]["route"] == "generation" and rc["canonical_home"] == w["home"]
    assert marker["payload"]["inputs_json"] == {s2: R.file_sha256(s2)}          # the canonical spelling, never the generation's path
    # the generation's path appears ONLY as provenance (extra.read_copy: which copy, which attestation); no identity and no batch record spells it
    without_copy = dict(marker["payload"], extra={k: v for k, v in marker["payload"]["extra"].items() if k != "read_copy"})
    assert w["gen_dir"] not in R.canonical_json(without_copy)
    for fn in os.listdir(os.path.join(plan["root"], "records", "adapters", rev_dir)):
        if "source_revision" not in fn and "artifact" not in fn:
            for _, rec, _ in R.read_record_file(os.path.join(plan["root"], "records", "adapters", rev_dir, fn)):
                assert w["gen_dir"] not in R.canonical_json(rec), fn
    # the identity map the shard installed: the exported tree onto the checkout and the generation onto its canonical base
    assert pub[0]["pass"]["identity_map"] == {plan["tree"]: plan["repo_root"], f"{w['gen_dir']}/synthetic_stores/{w['sha8']}": w["stores"]}
    # a plan routing the adapter to the grading host, or naming another canonical home, is refused BY the shard (a result, rc 0, summary says why)
    plan2 = dict(plan, uri_home="/home/someone-else")
    CR._write_json(f"{plan['window_root']}/plan2.json", plan2)
    assert CR.run_shard(f"{plan['window_root']}/plan2.json", 0, home=w["home"], summaries_dir=str(tmp_path / "s2")) == 0
    d = json.load(open(CR.summary_path(str(tmp_path / "s2"), ADAPTER, 0, 2)))
    assert d["ok"] is False and "canonical home" in d["refusal"]
    # C5: a same-size edit under the seal AFTER the attestation. The standing record still admits the shard (the check is O(1): the record,
    # not a walk) and the READER refuses the edited input by name -- the byte guarantee, unchanged; the NEXT operation's attestation (a sample
    # within a week of the full walk; every file of a 3-file generation) refuses the generation, and a shard against that record refuses by name
    gen_file = f"{w['gen_dir']}/synthetic_stores/{w['sha8']}/s1/store.json"
    os.chmod(gen_file, 0o644)
    _same_size_edit(gen_file)
    assert CR.run_shard(f"{plan['window_root']}/plan.json", 0, home=w["home"], summaries_dir=str(tmp_path / "s3")) == 0
    d = json.load(open(CR.summary_path(str(tmp_path / "s3"), ADAPTER, 0, 2)))
    assert d["ok"] is True and d["admission_pre"]["synthetic_stores"]["ok"] and d["refused_roots"] == []
    assert d["pass"]["groups_refused"] == 1 and "verified-manifest check" in d["refused"][0]["error"] and d["pass"]["inputs_refused_by_manifest"] == [os.path.realpath(gen_file)]
    plan_next = dict(plan, window="w1-next", attestation=f"{plan['window_root']}/attestation-next.json")
    CR._write_json(f"{plan['window_root']}/plan_next.json", plan_next)
    rec = ST.attest_plan(f"{plan['window_root']}/plan_next.json", plan_next["attestation"])
    g = rec["generations"]["synthetic_stores"]
    assert not rec["ok"] and g["refusal"] == "content_changed" and g["content"]["mode"] == "sample" and g["content"]["mismatched"] == ["s1/store.json"]
    assert CR.run_shard(f"{plan['window_root']}/plan_next.json", 0, home=w["home"], summaries_dir=str(tmp_path / "s4")) == 0
    d = json.load(open(CR.summary_path(str(tmp_path / "s4"), ADAPTER, 0, 2)))
    assert d["ok"] is False and d["refused_roots"][0]["component"] == "synthetic_stores" and d["refused_roots"][0]["refusal"] == "attestation_refused"
    assert "content_changed" in d["refusal"] and "s1/store.json" in d["refusal"] and d["pass"] is None
    # no record, another operation's record, a record attesting another generation: refused by name before the home tree is touched
    os.remove(plan_next["attestation"])
    assert CR.run_shard(f"{plan['window_root']}/plan_next.json", 1, home=w["home"], summaries_dir=str(tmp_path / "s5")) == 0
    d = json.load(open(CR.summary_path(str(tmp_path / "s5"), ADAPTER, 1, 2)))
    assert d["ok"] is False and d["refused_roots"][0]["refusal"] == "no_attestation" and d["home_tree"] is None
    plan_other = dict(plan, window="w1-other")                                   # the ok record of w1 under another window's plan
    CR._write_json(f"{plan['window_root']}/plan_other.json", plan_other)
    assert CR.run_shard(f"{plan['window_root']}/plan_other.json", 0, home=w["home"], summaries_dir=str(tmp_path / "s6")) == 0
    d = json.load(open(CR.summary_path(str(tmp_path / "s6"), ADAPTER, 0, 2)))
    assert d["ok"] is False and d["refused_roots"][0]["refusal"] == "attestation_window" and "operation w1's, not w1-other's" in d["refusal"]
    plan_pin = dict(plan, adapters={ADAPTER: dict(plan["adapters"][ADAPTER], pins={"synthetic_stores": "00000000"})})
    CR._write_json(f"{plan['window_root']}/plan_pin.json", plan_pin)
    assert CR.run_shard(f"{plan['window_root']}/plan_pin.json", 0, home=w["home"], summaries_dir=str(tmp_path / "s7")) == 0
    d = json.load(open(CR.summary_path(str(tmp_path / "s7"), ADAPTER, 0, 2)))
    assert d["ok"] is False and d["refused_roots"][0]["refusal"] == "attestation_pin"


def test_plan_routes_an_adapter_to_the_grading_host_on_a_missing_pin_or_a_refused_component_and_names_it(tmp_path, monkeypatch):
    w = _world_with_generation(tmp_path, monkeypatch)
    cfg = w["cfg"]
    # (a) no published generation for the component
    plan = CR.plan_window(w["root"], "w0", [ADAPTER, "ghost"], cfg, w["dest"], "a" * 40, uri_home=w["home"], published={})
    assert plan["adapters"][ADAPTER]["route"] == "grading_host" and "no published generation" in plan["adapters"][ADAPTER]["reason"]
    assert plan["adapters"]["ghost"]["route"] == "grading_host" and "not declared" in plan["adapters"]["ghost"]["reason"] and plan["tasks"] == []
    assert set(CR.refused_adapters(plan)) == {ADAPTER, "ghost"}
    # (b) the grading host canonical copy differs from the pin by ONE file (a same-size edit): admission pass 1 refuses the component; the
    # adapter reads on the grading host and the refusal is named
    p = os.path.join(w["stores"], "s1", "store.json")
    data = _same_size_edit(p)
    plan = CR.plan_window(w["root"], "w1", [ADAPTER], cfg, w["dest"], "a" * 40, uri_home=w["home"])
    e = plan["adapters"][ADAPTER]
    assert e["route"] == "grading_host" and "admission refused component(s): synthetic_stores: content_changed" in e["reason"] and plan["tasks"] == []
    assert e["revalidation_pre"]["synthetic_stores"]["ok"] is False and CR.refused_adapters(plan) == {ADAPTER: [f"plan: {e['reason']}"]}
    # (c) a matching copy is admitted: route cluster, K tasks, provenance names the pin
    open(p, "wb").write(data)
    plan = CR.plan_window(w["root"], "w2", [ADAPTER], cfg, w["dest"], "a" * 40, uri_home=w["home"])
    assert plan["adapters"][ADAPTER]["route"] == "cluster" and [t["shard"] for t in plan["tasks"]] == [0, 1] and plan["adapters"][ADAPTER]["task_indices"] == [0, 1]
    # (d) post-run revalidation: the canonical copy moved between admission and the read's end -> the adapter is recorded refused; the
    # revalidation MOVES NOTHING (before the pull the inbox holds only earlier windows' deliveries -- this marker-less one is
    # an earlier partial, left in place; inbox-merge leaves it too, "incomplete", never a marker-less records/ directory)
    _same_size_edit(p)
    inbox_dir = os.path.join(H.inbox_dir(w["root"]), CR.INBOX_NAME, "adapters", ADAPTER, "0123456789abcdef")
    os.makedirs(inbox_dir)
    open(os.path.join(inbox_dir, "000001-artifact.jsonl"), "w").write("{}\n")
    stale = CR.revalidate_plan(w["root"], plan, cfg)
    assert stale == [ADAPTER] and any("post-run revalidation refused" in r for r in plan["refusals"][ADAPTER])
    assert os.path.isdir(inbox_dir) and not os.path.exists(os.path.join(H.inbox_dir(w["root"]), CR.INBOX_NAME, CR.STALE_DIR)) and not hasattr(CR, "quarantine_stale")
    assert H.scan_inbox(w["root"]) == [(inbox_dir, "incomplete")] and os.path.isfile(os.path.join(inbox_dir, "000001-artifact.jsonl"))
    assert not os.path.exists(os.path.join(H.records_dir(w["root"]), "adapters", ADAPTER, "0123456789abcdef"))
    # the status section reads the latest plan
    CR._write_json(CR.plan_path(w["root"], "w2"), plan)
    sec = CR.status_section(w["root"])
    assert sec["present"] and sec["window"] == "w2" and sec["rows"][0]["adapter"] == ADAPTER and "post-run revalidation" in sec["rows"][0]["refusals"]


def _cluster_spool(tmp_path, monkeypatch, with_bundle=True):
    """A window root whose spool holds ONE published revision of the synthetic adapter (a changed s2) and, when asked, an
    `_activations/<rev>/<stamp>/` bundle of it (the same files re-emitted), plus the shard summary listing both."""
    w = _world_with_generation(tmp_path, monkeypatch)
    _store(w["stores"], "s2", "legacy_store_2", "none", [0.1, 0.9, 0.5])
    res2 = ST.mirror_component(w["root"], w["dest"], w["cfg"].components["synthetic_stores"], runner=TS.Runner())
    w["sha8"] = res2.content_sha8
    plan, plan_file, _ = _plan_and_stage(w, tmp_path)
    _node(w)
    for i in range(2):
        assert CR.run_shard(f"{plan['window_root']}/plan.json", i, home=w["home"]) == 0
    expected, _per = CR.expected_from_summaries(plan["summaries_dir"])
    assert len(expected) == 1
    rev_dir = next(iter(expected))
    rev = rev_dir.split("/")[1]
    _back_on_grading_host(w)
    assert CR.revalidate_plan(w["root"], plan, w["cfg"]) == []          # the post-run revalidation the pull requires
    if with_bundle:
        src = os.path.join(plan["root"], "records", "adapters", rev_dir)
        bundle_rel = f"{ADAPTER}/_activations/{rev}/20260928220000000"
        shutil.copytree(src, os.path.join(plan["root"], "records", "adapters", bundle_rel))
        # the summary lists the bundle too (a shard that re-activated would): same files, same sizes
        spath = expected[rev_dir]["summary"]
        doc = json.load(open(spath))
        doc["published"][bundle_rel] = dict(doc["published"][rev_dir])
        CR._write_json(spath, doc)
    return w, plan, rev_dir, rev


def test_pull_delivers_a_nested_revision_marker_last_and_the_fold_activates_it_in_one_transaction(tmp_path, monkeypatch):
    w, plan, rev_dir, rev = _cluster_spool(tmp_path, monkeypatch)
    expect = str(tmp_path / "expect")
    calls = []

    def runner(argv, **kw):
        calls.append(list(argv))
        return subprocess.run(argv, **kw)
    rep = CR.pull_window(w["root"], plan, expect, source=plan["window_root"], runner=runner)
    assert rep["rsync"] == {"summaries": 0, "pass1": 0, "pass2": 0} and not rep["quarantined"] and rep["adapters_refused"] == {}
    assert {k: v["state"] for k, v in rep["pulled"].items()} == {rev_dir: "complete", f"{ADAPTER}/_activations/{rev}/20260928220000000": "complete"}
    # pass 0 received into a fresh per-attempt directory, installed as `expect` whole; pass 1 was a --files-from of exactly the
    # two listed directories with the markers excluded (ownership); pass 2 was --files-from of the markers alone; the markers arrived LAST
    p0 = next(c for c in calls if "--prune-empty-dirs" in c)
    assert p0[-1] != expect.rstrip("/") + "/" and p0[-1].startswith(expect + ".recv-") and os.path.isdir(expect) and not os.path.exists(p0[-1])
    p1 = next(c for c in calls if any(a.startswith("--exclude=*source_revision*") for a in c))
    p2 = next(c for c in calls if any(a.startswith("--files-from=") for a in c) and not any(a.startswith("--exclude=*source_revision*") for a in c))
    assert calls.index(p0) < calls.index(p1) < calls.index(p2) and len(calls) == 3
    # passes 1 and 2 receive into an attempt-local directory under the inbox (`.recv-<window>-…`), with no --ignore-existing (a
    # fresh directory needs none); the verified revisions were installed by one rename each and the receive directory is gone
    inbox_root = os.path.join(H.inbox_dir(w["root"]), CR.INBOX_NAME)
    assert p1[-1] == p2[-1] and p1[-1].startswith(os.path.join(inbox_root, ".recv-w1-")) and not os.path.exists(p1[-1].rstrip("/"))
    assert "--ignore-existing" not in p1 and "--ignore-existing" not in p2 and any(a.startswith("--files-from=") for a in p1)
    assert not [d for d in os.listdir(inbox_root) if d.startswith(".recv-")]
    inbox = os.path.join(inbox_root, "adapters")
    assert sorted(os.listdir(os.path.join(inbox, rev_dir))) == sorted(CR.expected_from_summaries(expect)[0][rev_dir]["files"])
    # the summary carries every published file's sha256 (shard_summary/2), and what landed is byte for byte what the shard published
    exp_rev = CR.expected_from_summaries(expect)[0][rev_dir]
    assert set(exp_rev["sha256"]) == set(exp_rev["files"]) and all(R.file_sha256(os.path.join(inbox, rev_dir, fn)) == sha for fn, sha in exp_rev["sha256"].items())
    # merge + fold: ONE activation with every batch present; nothing blocked, nothing ingested twice (the bundle carries the same rows)
    moved = H.scan_inbox(w["root"])
    n_files = len(os.listdir(os.path.join(H.records_dir(w["root"]), "adapters", ADAPTER, rev)))
    assert all(status == "merged" for _, status in moved) and len(moved) == 2 * n_files
    ing = H.ingest_once(w["root"], workers=0, min_free=0)
    assert ing.refused == 0 and ing.blocked == 0, (ing.refusals, ing.blocked)
    assert _rows(w["root"], "SELECT accepted FROM source_revision WHERE revision_id = ?", (rev,)) == [(1,)]
    assert _rows(w["root"], "SELECT COUNT(*) FROM item_result WHERE source_revision_id = ?", (rev,)) == [(3,)]
    assert _rows(w["root"], "SELECT COUNT(*) FROM blocked_records") == [(0,)]
    assert _rows(w["root"], "SELECT status, COUNT(*) FROM record_delivery WHERE file_uri LIKE ? GROUP BY status", (f"records/adapters/{ADAPTER}/{rev}/%",)) == [("ok", n_files)]
    head = _rows(w["root"], "SELECT revision_id FROM source_revision WHERE source_id = ? AND accepted = 1 ORDER BY observed_utc DESC LIMIT 1", (f"{ADAPTER}:s2",))
    assert head == [(rev,)]
    # no conflict file anywhere (S11)
    assert not [p for dp, _d, fns in os.walk(H.records_dir(w["root"])) for p in fns if ".conflict-" in p]
    # a second pull is a no-op: --ignore-existing, nothing new, nothing quarantined
    rep2 = CR.pull_window(w["root"], plan, expect, source=plan["window_root"], runner=runner)
    assert not rep2["quarantined"] and rep2["adapters_refused"] == {}


def test_a_marker_first_arrival_is_the_harvesters_hazard_the_two_pass_pull_answers(tmp_path, monkeypatch):
    """What happens when the marker lands BEFORE its batches (the shape a single whole-tree rsync could produce): recorded, not assumed.
    The harvester applies a directory that HAS its marker; the batch files that arrive later fold into the same revision on the next
    cycle and nothing is ingested twice. Whatever the interim state, the complete case ends identical -- and the two-pass pull never
    produces the interim state (test above)."""
    w, plan, rev_dir, rev = _cluster_spool(tmp_path, monkeypatch, with_bundle=False)
    src = os.path.join(plan["root"], "records", "adapters", rev_dir)
    dst = os.path.join(H.records_dir(w["root"]), "adapters", rev_dir)
    os.makedirs(dst)
    marker = next(fn for fn in os.listdir(src) if "source_revision" in fn)
    shutil.copy2(os.path.join(src, marker), os.path.join(dst, marker))
    ing1 = H.ingest_once(w["root"], workers=0, min_free=0)
    interim = _rows(w["root"], "SELECT accepted FROM source_revision WHERE revision_id = ?", (rev,))
    items_interim = _rows(w["root"], "SELECT COUNT(*) FROM item_result WHERE source_revision_id = ?", (rev,))[0][0]
    for fn in os.listdir(src):
        if fn != marker:
            shutil.copy2(os.path.join(src, fn), os.path.join(dst, fn))
    ing2 = H.ingest_once(w["root"], workers=0, min_free=0)
    assert ing2.refused == 0
    assert _rows(w["root"], "SELECT accepted FROM source_revision WHERE revision_id = ?", (rev,)) == [(1,)]
    assert _rows(w["root"], "SELECT COUNT(*) FROM item_result WHERE source_revision_id = ?", (rev,)) == [(3,)]
    assert _rows(w["root"], "SELECT COUNT(*) FROM run WHERE run_id IN (SELECT run_id FROM run_metric WHERE run_id LIKE 'synthcampaign:%')") [0][0] >= 1
    # the record (measured 2026-09-28, pinned): the harvester ACTIVATES a directory whose marker arrived before its batches -- accepted = 1
    # with ZERO item rows, nothing blocked, nothing held -- so the "held until the batches arrive" the plan expected is NOT what happens;
    # the two-pass pull's marker-last barrier is the ONLY guard against an incomplete activation, and the complete case above proves it
    assert interim == [(1,)] and items_interim == 0 and ing1.blocked == 0 and ing1.revisions_held == 0


def test_an_interrupted_first_pass_pulls_no_marker_quarantines_the_partial_dir_and_routes_the_adapter_to_the_grading_host(tmp_path, monkeypatch):
    w, plan, rev_dir, rev = _cluster_spool(tmp_path, monkeypatch, with_bundle=False)
    expect = str(tmp_path / "expect")
    src = os.path.join(plan["root"], "records", "adapters", rev_dir)
    # (a) a batch file the shard summary lists is missing from the source: the list mismatches -> quarantine, no marker, fallback
    victim = next(fn for fn in sorted(os.listdir(src)) if "source_revision" not in fn and "artifact" not in fn)
    os.rename(os.path.join(src, victim), os.path.join(plan["root"], victim + ".aside"))
    rep = CR.pull_window(w["root"], plan, expect, source=plan["window_root"])
    assert rev_dir in rep["quarantined"] and "batch list mismatch" in rep["quarantined"][rev_dir]["why"] and victim in rep["quarantined"][rev_dir]["why"]
    q = os.path.join(H.inbox_dir(w["root"]), CR.INBOX_NAME, CR.FAILED_DIR, "w1", rev_dir)
    assert os.path.isdir(q) and not any("source_revision" in fn for fn in os.listdir(q))
    assert not os.path.exists(os.path.join(H.inbox_dir(w["root"]), CR.INBOX_NAME, "adapters", rev_dir))
    assert ADAPTER in rep["adapters_refused"] and rep["pulled"] == {} and "pass2" not in rep["rsync"]
    # the quarantine is operator space: never merged, never folded
    assert H.scan_inbox(w["root"]) == [] and H.ingest_once(w["root"], workers=0, min_free=0).ok == 0
    assert _rows(w["root"], "SELECT COUNT(*) FROM source_revision WHERE revision_id = ?", (rev,)) == [(0,)]
    os.rename(os.path.join(plan["root"], victim + ".aside"), os.path.join(src, victim))
    # (b) pass 1 itself fails (rsync killed): every directory quarantined, no marker moves, the adapter falls back
    shutil.rmtree(os.path.join(H.inbox_dir(w["root"]), CR.INBOX_NAME, CR.FAILED_DIR))

    def killed(argv, **kw):
        if any(a.startswith("--exclude=*source_revision*") for a in argv):
            r = subprocess.run(argv, **kw)                       # the files land, but the transfer "died"
            return subprocess.CompletedProcess(argv, 20, stdout=r.stdout, stderr="rsync error: received SIGINT")
        return subprocess.run(argv, **kw)
    rep2 = CR.pull_window(w["root"], plan, expect, source=plan["window_root"], runner=killed)
    assert rep2["rsync"]["pass1"] == 20 and rev_dir in rep2["quarantined"] and "pass 1 rsync returned 20" in rep2["quarantined"][rev_dir]["why"]
    assert "pass2" not in rep2["rsync"] and ADAPTER in rep2["adapters_refused"]
    assert not any("source_revision" in fn for dp, _d, fns in os.walk(os.path.join(H.inbox_dir(w["root"]), CR.INBOX_NAME)) for fn in fns)
    # (c) a summary that never arrived (a shard that died -- the one that published nothing): its adapter falls back; the revision the
    # other shard published still travels complete, and the grading host reads the rest
    shutil.rmtree(expect)
    expected, _per = CR.expected_from_summaries(plan["summaries_dir"])
    dead = next(d["_path"] for d in _per[ADAPTER] if not d.get("published"))
    os.remove(dead)
    shutil.rmtree(os.path.join(H.inbox_dir(w["root"]), CR.INBOX_NAME, CR.FAILED_DIR))
    rep3 = CR.pull_window(w["root"], plan, expect, source=plan["window_root"])
    assert rep3["summaries"][ADAPTER]["present"] == 1 and any("shard summaries missing" in r for r in rep3["adapters_refused"][ADAPTER])
    assert rev_dir in rep3["pulled"] and rep3["pulled"][rev_dir]["state"] == "complete"     # the complete revision still travels; the grading host reads the rest
    fb = CR.refused_adapters(plan, rep3)
    assert fb[ADAPTER] and all(r.startswith("pull: ") for r in fb[ADAPTER])


def test_refusal_drill_plants_a_stale_pin_in_a_scratch_copy_and_the_plan_routes_the_adapter_to_the_grading_host_by_name(tmp_path, monkeypatch):
    """The M3 refusal drill, dry-run against a synthetic world with a real published generation: the drill copies the store state, plants
    a stale manifest for one component, plans a window against the scratch root and asserts route = grading_host naming the component, and the
    page naming it. The live root and the generation are untouched."""
    w = _world_with_generation(tmp_path, monkeypatch)
    drill = os.path.join(REPO_ROOT, "scripts", "registry", "measure", "refusal_drill.sh")
    before = json.load(open(ST.state_path(w["root"])))
    env = dict(os.environ, REGISTRY_PY=sys.executable, REGISTRY_PYTHONPATH=SRC_DIR)
    r = subprocess.run(["bash", drill, "--root", w["root"], "--component", "synthetic_stores", "--adapter", ADAPTER, "--config", w["cfgp"]],
                       capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "DRILL PASS" in r.stdout and "route=grading_host" in r.stdout and "synthetic_stores: content_changed" in r.stdout, r.stdout
    assert json.load(open(ST.state_path(w["root"]))) == before                                   # the live state is untouched
    assert not os.path.isdir(CR.plan_state_dir(w["root"]))                                        # no plan written under the live root
    # a component with no published generation at all: the drill still routes the adapter to the grading host (no_manifest), and says so
    r2 = subprocess.run(["bash", drill, "--root", w["root"], "--component", "ghost_component", "--adapter", ADAPTER, "--config", w["cfgp"]],
                        capture_output=True, text=True, env=env, timeout=600)
    assert "DRILL FAIL" in r2.stdout or "route=grading_host" in r2.stdout


# --------------------------------------------------------------------------------------------
# part 4: the review's five findings -- delivery prerequisites, the staged configuration, the read policy
# --------------------------------------------------------------------------------------------


class SyntheticStoreAdapterB(SyntheticStoreAdapter):
    name = "synthetic_store_b"


ADAPTER_B = SyntheticStoreAdapterB.name


def _world_two_adapters(tmp_path, monkeypatch):
    """`_world_with_generation` plus a SECOND adapter over its own component (`<home>/stores_b`), both mirrored as published generations,
    K = 1 each."""
    root, home, stores, ad = _grading_host(tmp_path)
    stores_b = os.path.join(home, "stores_b")
    os.makedirs(stores_b)
    _store(stores_b, "t1", "legacy_store_1", "constitution_lora@aaaa", [0.3, 0.6])
    _store(stores_b, "t2", "legacy_store_2", "none", [0.5, 0.5])
    ad_b = SyntheticStoreAdapterB(stores_b)
    with DB.reading(DB.db_path(root)) as conn:
        rep = RUN.run_pass(root, [ad_b], conn=conn, now=synth.t(2001), producer=PRODUCER)
    assert rep.groups_refreshed == 2, [r.error for r in rep.results]
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    monkeypatch.setattr(ST, "HOME", home)
    cfgp = TS._cfg(tmp_path, {"synthetic_stores": {"base": stores, "authority": "grading_host_copy", "origin": "grading_host"},
                              "synthetic_stores_b": {"base": stores_b, "authority": "grading_host_copy", "origin": "grading_host"}},
                   adapters={ADAPTER: {"shards": 1, "components": ["synthetic_stores"]}, ADAPTER_B: {"shards": 1, "components": ["synthetic_stores_b"]}})
    cfg = ST.load_config(cfgp)
    dest = cfg.destination("cluster_a")
    for slug in ("synthetic_stores", "synthetic_stores_b"):
        assert ST.mirror_component(root, dest, cfg.components[slug], runner=TS.Runner()).outcome == "published"
    monkeypatch.setattr("sisteer.registry.adapters.load_all", lambda: {ADAPTER: lambda: SyntheticStoreAdapter(stores), ADAPTER_B: lambda: SyntheticStoreAdapterB(stores_b)})
    return {"root": root, "home": home, "stores": stores, "stores_b": stores_b, "ad": ad, "cfg": cfg, "cfgp": cfgp, "dest": dest,
            "gen_dir": f"{dest.stores_dir}/gen", "shared": dest.shared, "node_moves": [stores, stores_b]}


def test_a_stale_adapter_and_a_refusing_shard_deliver_nothing_end_to_end(tmp_path, monkeypatch):
    """Revalidate -> pull -> inbox-merge -> fold with one adapter whose canonical copy moved after its read (stale) and one whose shard
    refused after publishing (its summary says ok = False yet lists the revision): NEITHER revision reaches the inbox, the records or the
    database; both adapters read on the grading host. A directory of the stale adapter already in the inbox is quarantined, never merged;
    a plan with no recorded revalidation delivers nothing at all."""
    w = _world_two_adapters(tmp_path, monkeypatch)
    _store(w["stores"], "s2", "legacy_store_2", "none", [0.1, 0.9, 0.5])
    _store(w["stores_b"], "t2", "legacy_store_2", "none", [0.5, 0.5, 0.1])
    for slug in ("synthetic_stores", "synthetic_stores_b"):
        assert ST.mirror_component(w["root"], w["dest"], w["cfg"].components[slug], runner=TS.Runner()).outcome == "published"
    plan, plan_file, _ = _plan_and_stage(w, tmp_path, adapters=(ADAPTER, ADAPTER_B))
    assert [t["adapter"] for t in plan["tasks"]] == [ADAPTER, ADAPTER_B] and plan["home_tree"] and len(plan["home_tree"]["links"]) == 2
    _node(w)
    for i in range(2):
        assert CR.run_shard(f"{plan['window_root']}/plan.json", i, home=w["home"]) == 0
    expected, per = CR.expected_from_summaries(plan["summaries_dir"])
    revs = {exp["adapter"]: rev_dir for rev_dir, exp in expected.items()}
    assert set(revs) == {ADAPTER, ADAPTER_B}
    _back_on_grading_host(w)
    # adapter A goes STALE after its read; adapter B's shard refused after publishing
    _same_size_edit(os.path.join(w["stores"], "s1", "store.json"))
    assert CR.revalidate_plan(w["root"], plan, w["cfg"]) == [ADAPTER]
    sb = per[ADAPTER_B][0]
    doc = json.load(open(sb["_path"]))
    doc.update(ok=False, refusal="a generation changed under the read: synthetic_stores_b: content_changed")
    CR._write_json(sb["_path"], doc)
    inbox = os.path.join(H.inbox_dir(w["root"]), CR.INBOX_NAME)
    rep = CR.pull_window(w["root"], plan, str(tmp_path / "expect"), source=plan["window_root"])
    assert rep["pulled"] == {} and set(rep["withheld"]) == set(revs.values()) and set(rep["adapters_refused"]) == {ADAPTER, ADAPTER_B}
    assert "post-run revalidation refused" in rep["withheld"][revs[ADAPTER]] and "not ok" in rep["withheld"][revs[ADAPTER_B]]
    for rev_dir in revs.values():                                                # excluded BEFORE pass 1: never in the inbox
        assert not os.path.exists(os.path.join(inbox, "adapters", rev_dir))
    assert H.scan_inbox(w["root"]) == []
    ing = H.ingest_once(w["root"], workers=0, min_free=0)
    for rev_dir in revs.values():
        assert not os.path.exists(os.path.join(H.records_dir(w["root"]), "adapters", rev_dir))
        assert _rows(w["root"], "SELECT COUNT(*) FROM source_revision WHERE revision_id = ?", (rev_dir.split("/")[1],)) == [(0,)]
    assert ing.blocked == 0 and set(CR.refused_adapters(plan, rep)) == {ADAPTER, ADAPTER_B}
    # the bug's shape: a directory of the stale adapter already in the inbox is quarantined under _stale, never merged
    already = os.path.join(inbox, "adapters", revs[ADAPTER])
    os.makedirs(already)
    open(os.path.join(already, "000001-artifact.jsonl"), "w").write("{}\n")
    rep2 = CR.pull_window(w["root"], plan, str(tmp_path / "expect2"), source=plan["window_root"])
    q = rep2["quarantined"][revs[ADAPTER]]
    assert q["moved_to"].startswith(os.path.join(inbox, CR.STALE_DIR, "w1", "adapters")) and not os.path.exists(already) and H.scan_inbox(w["root"]) == []
    # no recorded post-run revalidation: nothing is delivered, every cluster adapter falls back, the reason says why
    rep3 = CR.pull_window(w["root"], dict(plan, stages={}), str(tmp_path / "expect3"), source=plan["window_root"])
    assert rep3["pulled"] == {} and rep3["rsync"] == {} and all("no post-run revalidation" in r[0] for r in rep3["adapters_refused"].values())
    assert set(rep3["adapters_refused"]) == {ADAPTER, ADAPTER_B}


def test_the_pull_carries_the_attestation_record_and_names_a_refused_pin_as_the_adapters_reason(tmp_path, monkeypatch):
    """The operation's attestation record is pulled beside the shard summaries and recorded on the plan's pull stage; an adapter whose
    pin the attestation refused is named with the attestation's own reason (so a window whose array never ran still says why), and a record
    of another operation is named as such. The record GATES DELIVERY, fail-closed -- the unadmitted adapter's revision is withheld before
    pass 1 (a directory of it already in the inbox is quarantined under `_failed/`), the admitted adapter's is delivered as before;
    another operation's record and a MISSING record deliver nothing of either adapter, each named with the reason."""
    w = _world_two_adapters(tmp_path, monkeypatch)
    _store(w["stores"], "s2", "legacy_store_2", "none", [0.1, 0.9, 0.5])
    _store(w["stores_b"], "t2", "legacy_store_2", "none", [0.5, 0.5, 0.1])
    for slug in ("synthetic_stores", "synthetic_stores_b"):
        assert ST.mirror_component(w["root"], w["dest"], w["cfg"].components[slug], runner=TS.Runner()).outcome == "published"
    plan, plan_file, _ = _plan_and_stage(w, tmp_path, adapters=(ADAPTER, ADAPTER_B))
    _node(w)
    for i in range(2):
        assert CR.run_shard(f"{plan['window_root']}/plan.json", i, home=w["home"]) == 0
    _back_on_grading_host(w)
    assert CR.revalidate_plan(w["root"], plan, w["cfg"]) == []
    expected, _per = CR.expected_from_summaries(plan["summaries_dir"])
    revs = {exp["adapter"]: rev_dir for rev_dir, exp in expected.items()}
    assert set(revs) == {ADAPTER, ADAPTER_B}
    inbox = os.path.join(H.inbox_dir(w["root"]), CR.INBOX_NAME)
    rep = CR.pull_window(w["root"], plan, str(tmp_path / "expect"), source=plan["window_root"])
    att = rep["attestation"]
    assert att["present"] and att["this_window"] and att["ok"] and att["window"] == "w1" and att["refused"] == [] and set(att["generations"]) == {"synthetic_stores", "synthetic_stores_b"}
    assert all(g["ok"] and g["content_mode"] == "full" and g["listing_entries"] > 0 for g in att["generations"].values()) and rep["adapters_refused"] == {}
    assert os.path.isfile(os.path.join(tmp_path, "expect", ST.ATTESTATION_FILE)) and len(rep["pulled"]) == 2 and rep["withheld"] == {}
    assert all(os.path.isdir(os.path.join(inbox, "adapters", rev_dir)) for rev_dir in revs.values())
    # the CLI records it on the plan's pull stage. This re-pull finds both delivered directories in the inbox WITH their markers: earlier
    # completed deliveries of these very revisions -- reported complete as they stand, nothing re-pulled, nothing quarantined
    CR._write_json(plan_file, dict(plan, stages=dict(plan["stages"], revalidate_post={"stale": []})))
    assert CLI.main(["--root", w["root"], "cluster-reads", "pull", "--window", "w1", "--source", plan["window_root"], "--expect", str(tmp_path / "expect2")]) == 0
    stage = CR._load_json(plan_file)["stages"]["pull"]
    assert stage["attestation"]["ok"] is True and stage["attestation"]["generations"]["synthetic_stores_b"]["content_mode"] == "full"
    assert stage["pulled"] == 2 and stage["quarantined"] == 0 and stage["withheld"] == 0 and stage["rsync"] == {"summaries": 0, "pass1": 0}
    assert all(os.path.isdir(os.path.join(inbox, "adapters", rev_dir)) for rev_dir in revs.values())
    # the window's inbox-merge right after: both revisions reach records/ and the inbox is empty again
    merged = H.scan_inbox(w["root"])
    assert merged and all(s == "merged" for _d, s in merged) and not any(os.path.exists(os.path.join(inbox, "adapters", rev_dir)) for rev_dir in revs.values())
    # a record that refused adapter B's pin (a listing mismatch) names B's reason; A is untouched -- and the attestation gates DELIVERY, not
    # just the stamp: A's revision travels again (the pull does not read records/), B's is withheld before pass 1, and a marker-less PARTIAL
    # of B already in the inbox (planted, as an interrupted pass 1 would have left it) is quarantined under _failed/, never merged
    doc = json.load(open(plan["attestation"]))
    doc["generations"]["synthetic_stores_b"].update(ok=False, refusal="listing_mismatch", reason="1 extra (first: t9/planted.json)")
    doc.update(ok=False, refused=["synthetic_stores_b"])
    CR._write_json(plan["attestation"], doc)
    already = os.path.join(inbox, "adapters", revs[ADAPTER_B])
    os.makedirs(already)
    open(os.path.join(already, "000001-artifact.jsonl"), "w").write("{}\n")
    rep2 = CR.pull_window(w["root"], plan, str(tmp_path / "expect3"), source=plan["window_root"])
    why_b = "the attestation refused component(s) this adapter reads: synthetic_stores_b: listing_mismatch (1 extra (first: t9/planted.json))"
    assert ADAPTER not in rep2["adapters_refused"] and why_b in rep2["adapters_refused"][ADAPTER_B]
    assert rep2["attestation"]["refused"] == ["synthetic_stores_b"] and set(CR.refused_adapters(plan, rep2)) == {ADAPTER_B}
    assert set(rep2["pulled"]) == {revs[ADAPTER]} and rep2["pulled"][revs[ADAPTER]]["state"] == "complete"
    assert rep2["withheld"] == {revs[ADAPTER_B]: why_b} and set(rep2["quarantined"]) == {revs[ADAPTER_B]}
    q = rep2["quarantined"][revs[ADAPTER_B]]
    assert q["why"] == f"{why_b}; never delivered" and q["moved_to"].startswith(os.path.join(inbox, CR.FAILED_DIR, "w1")) and not os.path.isdir(already)
    assert os.path.isdir(os.path.join(inbox, "adapters", revs[ADAPTER]))
    # another operation's record at the window root: every cluster adapter is named with that reason and nothing of THIS window is
    # delivered -- A's complete directory from the pull above holds its marker, so it is an earlier delivery: reported complete as it stands,
    # released from `withheld`, never quarantined (ownership: the verdict is about this window's read, and nothing of it was pulled)
    doc.update(window="w0", ok=True, refused=[])
    doc["generations"]["synthetic_stores_b"].update(ok=True, refusal=None, reason="")
    CR._write_json(plan["attestation"], doc)
    rep3 = CR.pull_window(w["root"], plan, str(tmp_path / "expect4"), source=plan["window_root"])
    assert rep3["attestation"]["this_window"] is False and all(any("operation w0's, not this one's" in r for r in rep3["adapters_refused"][a]) for a in (ADAPTER, ADAPTER_B))
    assert rep3["pulled"] == {revs[ADAPTER]: {"state": "complete (an earlier delivery already in the inbox)", "adapter": ADAPTER, "files": len(expected[revs[ADAPTER]]["files"])}}
    assert set(rep3["withheld"]) == {revs[ADAPTER_B]} and "operation w0's, not this one's" in rep3["withheld"][revs[ADAPTER_B]] and rep3["quarantined"] == {}
    assert rep3["rsync"] == {"summaries": 0, "pass1": 0} and os.path.isdir(os.path.join(inbox, "adapters", revs[ADAPTER]))
    # no record pulled (the attestation crashed before writing, or the record did not arrive): present False -- and FAIL-CLOSED:
    # every cluster adapter is refused naming the missing record, nothing of this window is delivered, nothing is replayed
    os.remove(plan["attestation"])
    rep4 = CR.pull_window(w["root"], plan, str(tmp_path / "expect5"), source=plan["window_root"])
    assert rep4["attestation"] == {"present": False, "window": None, "ok": None, "utc": None, "host": None, "refused": [], "generations": {}}
    missing = os.path.join(str(tmp_path / "expect5"), ST.ATTESTATION_FILE)
    for a in (ADAPTER, ADAPTER_B):
        assert any(f"no readable attestation record for this operation ({missing} is missing)" in r for r in rep4["adapters_refused"][a]), rep4["adapters_refused"][a]
    assert set(rep4["pulled"]) == {revs[ADAPTER]} and rep4["pulled"][revs[ADAPTER]]["state"] == "complete (an earlier delivery already in the inbox)"
    assert set(rep4["withheld"]) == {revs[ADAPTER_B]} and rep4["quarantined"] == {} and rep4["refused_known_ids"] == {ADAPTER: [], ADAPTER_B: []}
    assert "no readable attestation record for this operation" in rep4["withheld"][revs[ADAPTER_B]]
    assert set(CR.refused_adapters(plan, rep4)) == {ADAPTER, ADAPTER_B}
    # the end state: the inbox-merge finds A's re-delivered files identical to the merged ones (duplicates, removed), the planted partial sits
    # under _failed/ alone, and the ingest folds each revision ONCE -- delivered by the first, admitted pull; no conflict file anywhere
    moved = H.scan_inbox(w["root"])
    assert moved and all(s == "duplicate" for _d, s in moved) and not os.path.exists(os.path.join(inbox, "adapters", revs[ADAPTER]))
    assert sorted(os.listdir(os.path.join(inbox, CR.FAILED_DIR, "w1", ADAPTER_B))) == [revs[ADAPTER_B].split("/")[1]]
    ing = H.ingest_once(w["root"], workers=0, min_free=0)
    assert ing.refused == 0 and ing.blocked == 0, (ing.refusals, ing.blocked)
    for rev_dir in revs.values():
        rev_root = os.path.join(H.records_dir(w["root"]), "adapters", rev_dir)
        assert os.path.isdir(rev_root) and not any(".conflict" in fn for fn in os.listdir(rev_root))
        assert _rows(w["root"], "SELECT COUNT(*), SUM(accepted) FROM source_revision WHERE revision_id = ?", (rev_dir.split("/")[1],)) == [(1, 1)]


def test_run_shard_reads_the_staged_configuration_and_refuses_a_missing_or_altered_one(tmp_path, monkeypatch):
    """The plan records the configuration's sha and the stage lays the exact file beside the plan; the shard reads THAT copy (the grading
    host's path does not exist on a node), refuses an altered or missing copy by name, and the stage refuses a configuration that changed
    since the plan."""
    w = _world_with_generation(tmp_path, monkeypatch)
    plan, plan_file, _ = _plan_and_stage(w, tmp_path)
    staged = f"{plan['window_root']}/{CR.CONFIG_STAGED}"
    assert plan["config_sha256"] == R.file_sha256(w["cfgp"]) and plan["config_staged"] == staged and R.file_sha256(staged) == plan["config_sha256"]
    _node(w)
    os.rename(w["cfgp"], w["cfgp"] + ".gone")                               # the grading host path is not on the node
    assert CR.run_shard(f"{plan['window_root']}/plan.json", 0, home=w["home"]) == 0
    assert json.load(open(CR.summary_path(plan["summaries_dir"], ADAPTER, 0, 2)))["ok"] is True
    with open(staged, "a", encoding="utf-8") as fh:
        fh.write("\n# edited on the node\n")
    assert CR.run_shard(f"{plan['window_root']}/plan.json", 1, home=w["home"], summaries_dir=str(tmp_path / "s2")) == 0
    d = json.load(open(CR.summary_path(str(tmp_path / "s2"), ADAPTER, 1, 2)))
    assert d["ok"] is False and "not the one the plan was built from" in d["refusal"]
    os.remove(staged)
    assert CR.run_shard(f"{plan['window_root']}/plan.json", 1, home=w["home"], summaries_dir=str(tmp_path / "s3")) == 0
    d = json.load(open(CR.summary_path(str(tmp_path / "s3"), ADAPTER, 1, 2)))
    assert d["ok"] is False and "is missing" in d["refusal"]
    os.rename(w["cfgp"] + ".gone", w["cfgp"])
    with open(w["cfgp"], "a", encoding="utf-8") as fh:
        fh.write("\n# changed after the plan\n")
    with pytest.raises(CR.ClusterReadsError, match="changed since the plan"):
        CR.stage_window(plan, w["sidecar"], plan_file)


def test_a_plan_made_before_a_certified_config_rewrite_refuses_at_staging_and_a_new_plan_reads_end_to_end(tmp_path, monkeypatch):
    """A plan binds the exact configuration bytes its tree read. When the configuration is rewritten after the plan by a certified
    scrub-map row, staging that plan refuses by name and lays nothing out, so no window is staged whose shards would all refuse; a
    window staged before the rewrite whose staged copy carries the rewritten bytes refuses in every shard by name, in its summary; and a
    plan made from the rewritten file stages and reads end to end."""
    from sisteer import scrub_map as SM

    w = _world_with_generation(tmp_path, monkeypatch)
    old = CR.plan_window(w["root"], "w1", [ADAPTER], w["cfg"], w["dest"], "a" * 40, uri_home=w["home"], repo_root=REPO_ROOT)
    old_file = CR.plan_path(w["root"], "w1")
    CR._write_json(old_file, old)
    staged0, staged0_file, _ = _plan_and_stage(w, tmp_path, window="w0")     # a window staged and attested before the rewrite
    pre_sha = R.file_sha256(w["cfgp"])
    with open(w["cfgp"], "a", encoding="utf-8") as fh:                      # the rewrite: a comment line, certified by a row
        fh.write("\n# a comment the declared rules rewrote\n")
    post_sha = R.file_sha256(w["cfgp"])
    smap = tmp_path / "SCRUB_MAP.tsv"
    rel = os.path.relpath(w["cfgp"], tmp_path)
    smap.write_text("\t".join(SM.COLUMNS) + "\n" + "\t".join((pre_sha, rel, rel, post_sha, "f" * 64, "registry_seed")) + "\n")
    monkeypatch.setenv("SISTEER_SCRUB_MAP", str(smap))
    SM._CACHE.clear()
    assert old["config_sha256"] == pre_sha and SM.check(pre_sha, post_sha, path=w["cfgp"]) == SM.VIA_MAP
    # 1. the plan made before the rewrite refuses at staging, naming the row, and nothing of its window is laid out
    with pytest.raises(CR.ClusterReadsError, match=r"changed since the plan was built \(sha .* is the certified scrub-map rewrite .*plan again"):
        CR.stage_window(old, w["sidecar"], old_file)
    assert not any(os.path.exists(p) for p in (f"{old['window_root']}/plan.json", old["config_staged"], f"{old['root']}/records"))
    # 2. a window staged before the rewrite, its staged copy now the rewritten bytes: each shard refuses by name, in its summary
    shutil.copyfile(w["cfgp"], staged0["config_staged"])
    assert CR.run_shard(f"{staged0['window_root']}/plan.json", 0, home=w["home"]) == 0
    d = json.load(open(CR.summary_path(staged0["summaries_dir"], ADAPTER, 0, 2)))
    assert d["ok"] is False and "not the one the plan was built from" in d["refusal"] and "certified scrub-map rewrite" in d["refusal"]
    # 3. a plan made from the rewritten file stages, and its shards read a change made since the first fold end to end
    _store(w["stores"], "s2", "legacy_store_2", "none", [0.1, 0.9, 0.5])
    res2 = ST.mirror_component(w["root"], w["dest"], w["cfg"].components["synthetic_stores"], runner=TS.Runner())
    assert res2.outcome == "published"
    w.update(sha8=res2.content_sha8, cfg=ST.load_config(w["cfgp"]))
    plan, _plan_file, _ = _plan_and_stage(w, tmp_path, window="w2")
    assert plan["config_sha256"] == post_sha and R.file_sha256(plan["config_staged"]) == post_sha and len(plan["tasks"]) == 2
    _node(w)
    for i in range(2):
        assert CR.run_shard(f"{plan['window_root']}/plan.json", i, home=w["home"]) == 0
    docs = [json.load(open(CR.summary_path(plan["summaries_dir"], ADAPTER, i, 2))) for i in range(2)]
    assert all(doc["ok"] is True for doc in docs), [doc.get("refusal") for doc in docs]
    pub = [doc for doc in docs if doc["published"]]
    assert len(pub) == 1 and len(pub[0]["published"]) == 1


def test_the_read_policy_travels_in_the_plan_retry_refused_on_the_cluster_full_per_adapter_and_reread_force_to_the_grading_host(tmp_path, monkeypatch):
    """The window's per-adapter mode rides the plan into the shard. A standing refusal is retried on the cluster pass (planned, not left
    standing; its id was refused at emit time and never committed, so it is NOT known to the grading host: the shard reads it fresh and
    publishes it); `--full` for a named adapter plans every group (the committed ids known, the refused one read fresh again);
    `--reread --force` is a grading-host-only reactivation and routes there with the reason."""
    w = _world_with_generation(tmp_path, monkeypatch)
    # (c) reread --force never reaches the cluster
    plan_rf = CR.plan_window(w["root"], "wrf", [ADAPTER], w["cfg"], w["dest"], "a" * 40, uri_home=w["home"], reread_force_adapters=[ADAPTER])
    e = plan_rf["adapters"][ADAPTER]
    assert e["route"] == "grading_host" and e["reason"] == CR.REREAD_FORCE_REASON and e["policy"]["reread_force"] is True and plan_rf["tasks"] == []
    assert CR.refused_adapters(plan_rf) == {ADAPTER: [f"plan: {CR.REREAD_FORCE_REASON}"]}
    # (a) a standing refusal on the grading host (s1 changed and the reader failed on it), folded; then the reader is fixed and the store re-mirrored
    s1 = os.path.join(w["stores"], "s1", "store.json")
    w["ad"].broken.add(s1)
    _store(w["stores"], "s1", "legacy_store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4, 0.9])
    with DB.reading(DB.db_path(w["root"])) as conn:
        rep = RUN.run_pass(w["root"], [w["ad"]], conn=conn, now=synth.t(2100), producer=PRODUCER)
    assert rep.groups_refused == 1 and H.ingest_once(w["root"], workers=0, min_free=0).refused == 0
    w["ad"].broken.clear()
    assert ST.mirror_component(w["root"], w["dest"], w["cfg"].components["synthetic_stores"], runner=TS.Runner()).outcome == "published"
    plan, plan_file, _ = _plan_and_stage(w, tmp_path, window="wp")
    assert plan["adapters"][ADAPTER]["policy"] == {"retry_refused": True, "full": False, "reread_force": False}
    _node(w)
    for i in range(2):
        assert CR.run_shard(f"{plan['window_root']}/plan.json", i, home=w["home"]) == 0
    docs = [json.load(open(CR.summary_path(plan["summaries_dir"], ADAPTER, i, 2))) for i in range(2)]
    assert sum(d["pass"]["groups_refused_standing"] for d in docs) == 0                 # retried: not left standing
    assert [r for d in docs for r in d["refused_known"]] == [] and all(d["pass"]["policy"]["retry_refused"] for d in docs)
    # the emit-time-refused id is not known -- read fresh by its one owner and published there
    assert [v["group"] for d in docs for v in d["published"].values()] == ["s1"] and sum(d["pass"]["groups_refreshed"] for d in docs) == 1
    # the same plan WITHOUT the retry leaves it standing on every shard
    plan_nr = json.load(open(f"{plan['window_root']}/plan.json"))
    plan_nr["adapters"][ADAPTER]["policy"]["retry_refused"] = False
    CR._write_json(f"{plan['window_root']}/plan_nr.json", plan_nr)
    assert CR.run_shard(f"{plan['window_root']}/plan_nr.json", 0, home=w["home"], summaries_dir=str(tmp_path / "nr")) == 0
    d = json.load(open(CR.summary_path(str(tmp_path / "nr"), ADAPTER, 0, 2)))
    assert d["pass"]["groups_refused_standing"] == 1 and d["refused_known"] == [] and d["refused_standing"] == [{"adapter": ADAPTER, "group": "s1"}]
    assert d["standing_refusals"] == [] and d["pass"]["groups_standing_refusals"] == 0                  # left standing unread, not re-read (standing_refusals is the re-read's)
    # (b) --full for the named adapter: every group is planned (the committed ids are known to the grading host: refused by name, published nowhere;
    # s1, still refused at emit time on the grading host -- nothing of (a) was delivered -- is read fresh again)
    _back_on_grading_host(w)
    plan_f, _pf, _ = _plan_and_stage(w, tmp_path, window="wq", full_adapters=[ADAPTER])
    assert plan_f["adapters"][ADAPTER]["policy"]["full"] is True and plan_f["full"] is False
    _node(w)
    for i in range(2):
        assert CR.run_shard(f"{plan_f['window_root']}/plan.json", i, home=w["home"]) == 0
    docs = [json.load(open(CR.summary_path(plan_f["summaries_dir"], ADAPTER, i, 2))) for i in range(2)]
    assert sorted(r["group"] for d in docs for r in d["refused_known"]) == ["s2", "s3"] and all(d["pass"]["policy"]["full"] for d in docs)
    assert [v["group"] for d in docs for v in d["published"].values()] == ["s1"]


# --------------------------------------------------------------------------------------------
# part 5: the four bugs the simplify pass flagged
# --------------------------------------------------------------------------------------------


def test_a_refused_standing_group_is_keyed_by_canonical_identity_under_a_relocated_root(tmp_path):
    """The standing-refusal check indexed the watermarks by `Input.uri` (the LOCAL spelling) while they are keyed by `identity_uri`: under a
    non-empty identity map a refused, otherwise-unchanged group raised KeyError. Read from a relocated copy with the map onto the canonical
    home, the refusal is left standing (no retry) and read again with --retry-refused, minting the very id the refused read computed."""
    root, home, stores, ad = _grading_host(tmp_path)
    s1 = os.path.join(stores, "s1", "store.json")
    ad.broken.add(s1)
    _store(stores, "s1", "legacy_store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4, 0.9])
    with DB.reading(DB.db_path(root)) as conn:
        rep = RUN.run_pass(root, [ad], conn=conn, now=synth.t(2100), producer=PRODUCER)
    assert rep.groups_refused == 1 and H.ingest_once(root, workers=0, min_free=0).refused == 0
    attempt = next(r.revision_id for r in rep.results if r.group == "s1")
    # the relocated copy (mtimes preserved: the watermarks' stat matches) read through the map onto the canonical home
    local = str(tmp_path / "node" / "stores")
    shutil.copytree(stores, local)
    ad_b = SyntheticStoreAdapter(local)
    imap = AB.IdentityMap.build({local: stores})
    with DB.reading(DB.db_path(root)) as conn:
        rep2 = RUN.run_pass(root, [ad_b], conn=conn, now=synth.t(2200), producer=PRODUCER, identity_map_=imap)
    assert rep2.refused_standing == {ADAPTER: ["s1"]} and rep2.groups_refused == 0 and rep2.groups_refreshed == 0 and rep2.inputs_changed == 0
    with DB.reading(DB.db_path(root)) as conn:
        rep3 = RUN.run_pass(root, [ad_b], conn=conn, now=synth.t(2300), producer=PRODUCER, identity_map_=imap, retry_refused=True)
    assert rep3.refused_standing == {} and rep3.groups_refreshed == 1 and [r.revision_id for r in rep3.results] == [attempt]
    assert rep3.results[0].group_identity == "s1" and rep3.results[0].accepted


def test_submit_wires_local_venv_as_the_sbatch_fourth_positional_and_the_cli_flag_reaches_it(tmp_path, monkeypatch):
    """`--local-venv` (opt-in, the measured ~563 s per task path) reaches `read_array.sbatch`'s fourth positional through `cluster-reads submit`."""
    plan = {"tasks": [{"index": 0}, {"index": 1}], "tree": "/shared/gates/trees/abc", "window_root": "/shared/registry/reads/w1", "host": None}
    calls = []

    def runner(argv, **kw):
        calls.append(argv[-1])
        return subprocess.CompletedProcess(argv, 0, stdout="4242\n", stderr="")
    assert CR.submit_array(plan, "/shared/envs/v", "reads-w1", runner=runner) == "4242"
    assert CR.submit_array(plan, "/shared/envs/v", "reads-w1", runner=runner, local_venv=True) == "4242"
    assert calls[0].endswith(" /shared/envs/v 0") and calls[1].endswith(" /shared/envs/v 1") and "--array=0-1" in calls[0]
    # the CLI flag: recorded on the plan's submit stage and passed to submit_array
    root = _cluster_root(tmp_path)
    CR._write_json(CR.plan_path(root, "w1"), dict(plan, schema=CR.PLAN_SCHEMA, window="w1", adapters={}, refusals={}, stages={}))
    seen = []
    monkeypatch.setattr(CR, "submit_attest", lambda plan, **kw: "776")                 # C5: the attestation job precedes the array
    monkeypatch.setattr(CR, "submit_array", lambda plan, venv, job_name, **kw: (seen.append(kw), "777")[1])
    import sisteer.registry.cli as CLI
    assert CLI.main(["--root", root, "cluster-reads", "submit", "--window", "w1", "--venv", "/shared/envs/v", "--local-venv"]) == 0
    assert CLI.main(["--root", root, "cluster-reads", "submit", "--window", "w1", "--venv", "/shared/envs/v"]) == 0
    assert [k["local_venv"] for k in seen] == [True, False] and all(k["dependency"] == "776" for k in seen)
    stage = CR._load_json(CR.plan_path(root, "w1"))["stages"]["submit"]
    assert stage["job_id"] == "777" and stage["attest_job_id"] == "776" and stage["local_venv"] is False


def test_expected_from_summaries_names_an_unreadable_summary_under_the_same_key_as_a_readable_one(tmp_path):
    expect = tmp_path / "expect"
    (expect / "a").mkdir(parents=True)
    good = expect / "a" / "0-of-2.json"
    CR._write_json(str(good), {"schema": CR.SUMMARY_SCHEMA, "ok": True, "published": {"a/0123456789abcdef": {"adapter": "a", "files": {"m.jsonl": 3}, "marker": "m.jsonl"}}})
    bad = expect / "a" / "1-of-2.json"
    bad.write_text("{not json")
    expected, per = CR.expected_from_summaries(str(expect))
    assert set(expected) == {"a/0123456789abcdef"} and expected["a/0123456789abcdef"]["ok"] is True
    assert sorted(d["_path"] for d in per["a"]) == [str(good), str(bad)] and all("_path" in d for d in per["a"])
    unreadable = [d for d in per["a"] if "error" in d]
    assert [d["_path"] for d in unreadable] == [str(bad)] and "path" not in unreadable[0]
    # the pull's per-adapter report reads the same key
    plan = {"window": "w1", "host": None, "window_root": str(tmp_path / "spool"), "adapters": {"a": {"route": "cluster", "shards": 2}},
            "stages": {"revalidate_post": {"stale": []}}}
    (tmp_path / "spool" / "a").mkdir(parents=True)
    shutil.copy(str(good), str(tmp_path / "spool" / "a" / "0-of-2.json"))
    shutil.copy(str(bad), str(tmp_path / "spool" / "a" / "1-of-2.json"))
    root = _cluster_root(tmp_path)
    rep = CR.pull_window(root, plan, str(tmp_path / "expect2"), source=str(tmp_path / "spool"))
    assert rep["summaries"]["a"]["unreadable"] == [str(tmp_path / "expect2" / "a" / "1-of-2.json")] and any("unreadable" in r for r in rep["adapters_refused"]["a"])


# --------------------------------------------------------------------------------------------
# part 6: an aliased input's identity follows its alias (the second review's HIGH)
# --------------------------------------------------------------------------------------------

#: the heavy adapters that alias one shared input per group (`<uri>#<group>`); sampled_static and multiturn_grades alias nothing
ALIASING = ("bfcl_full", "bigcodebench_full", "em_misalign", "relu_paired")


def test_input_with_uri_and_with_alias_derive_the_identity_from_the_new_spelling_under_the_pass_map(tmp_path):
    p = tmp_path / "local" / "store.json"
    p.parent.mkdir()
    p.write_text("{}")
    local, canon = str(tmp_path / "local"), "/home/someone/canonical"
    with AB.using_identity_map(AB.IdentityMap.build({local: canon})):
        inp = AB.check_input(str(p), role="report")
        assert inp.identity_uri == f"{canon}/store.json"
        a = inp.with_alias("g1")
        assert a.uri == f"{p}#g1" and a.identity_uri == f"{canon}/store.json#g1" and a.sha256 == inp.sha256 and a.path == inp.path
        b = inp.with_uri("~/rulings/store.json")                 # a whole respelling (cap4096_declarations' ruling uri): not under the map
        assert b.uri == b.identity_uri == "~/rulings/store.json"
        assert len({x.identity_uri for x in (inp, a, inp.with_alias("g2"))}) == 3
    # no map: the identity IS the spelling, alias included (what the live watermarks carry)
    inp0 = AB.check_input(str(p), role="report")
    assert inp0.with_alias("g1").identity_uri == f"{p}#g1"


def test_no_adapter_aliases_an_input_through_dataclasses_replace():
    """The guard: `Input.with_uri` / `with_alias` are the only way an adapter respells an input (a `dataclasses.replace(inp, uri=...)`
    keeps the old identity_uri: every per-group alias of one manifest shared ONE identity and none of them was the grading host's)."""
    import re
    adapters_dir = os.path.join(SRC_DIR, "sisteer", "registry", "adapters")
    pat = re.compile(r"\breplace\(\s*[\w\[\]\.]+\s*,[^)]*\buri\s*=")
    offenders = []
    for fn in sorted(os.listdir(adapters_dir)):
        if not fn.endswith(".py") or fn == "base.py":
            continue
        for n, line in enumerate(open(os.path.join(adapters_dir, fn), encoding="utf-8"), 1):
            if pat.search(line):
                offenders.append(f"{fn}:{n}: {line.strip()}")
    assert offenders == [], offenders


@pytest.mark.parametrize("adapter_name", ALIASING)
def test_an_aliased_group_under_a_map_carries_its_alias_in_the_identity_and_distinct_groups_never_share_one(tmp_path, adapter_name):
    """For every aliasing heavy adapter: on the grading host path the identity IS the aliased uri (the key the live watermarks and revision ids
    carry); under a non-empty identity map the identity is the canonical spelling WITH the alias, one per group -- before the fix the
    aliases of one manifest all carried the unsuffixed identity (several groups, one identity; ids the grading host never minted). The grading host
    pass's published inputs_json is keyed by the aliased uris."""
    a = str(tmp_path / "A")
    os.makedirs(a)
    build = WORLDS[adapter_name](a)
    # the grading host path: identity == uri for every input, aliases included
    ad = build(a)
    with AB.using_identity_map(None):
        inputs = ad.discover()
    aliased = [i for i in inputs if "#" in i.uri]
    assert aliased, adapter_name
    assert all(i.identity_uri == i.uri for i in inputs)
    assert len({i.uri for i in inputs}) == len(inputs) == len({i.identity_uri for i in inputs})
    # under a map onto another home: the canonical spelling carries the alias; one identity per aliased group
    canon = "/home/someone-else/canonical"
    with AB.using_identity_map(AB.IdentityMap.build({a: canon})):
        inputs_m = build(a).discover()
    by_uri = {i.uri: i for i in inputs_m}
    assert sorted(by_uri) == sorted(i.uri for i in inputs)
    for i in inputs_m:
        want = canon + i.uri[len(a):] if i.uri.startswith(a) else i.uri
        assert i.identity_uri == want, (i.uri, i.identity_uri)
    assert len({i.identity_uri for i in inputs_m}) == len(inputs_m)
    groups = {}
    for i in inputs_m:
        groups.setdefault(ad.comparison_group(i), set()).add(i.identity_uri)
    shared = {u for g, us in groups.items() for u in us if sum(u in v for v in groups.values()) > 1 and "#" in u}
    assert shared == set(), shared                                      # no aliased identity belongs to two groups
    # the grading host pass keys inputs_json (and so the revision id) by the aliased uris
    reg = str(tmp_path / "reg")
    H.ensure_layout(reg)
    H.ingest_once(reg, workers=0, min_free=0)          # the registry database (its schema carries v_run_current): these readers file contrasts and a
    with DB.reading(DB.db_path(reg)) as conn:           # contrast-filing read needs the pass's anchor frame (emitter 4); on a fresh registry every comparison parks
        rep = _pass(reg, build(a), conn=conn)
    assert rep.groups_refreshed > 0, [r.error for r in rep.results]
    keys = {k for f in _revision_facts(reg, adapter_name).values() for k in json.loads(f["inputs_json"])}
    # every key is a local (aliased) uri -- or a synthetic input of the group (a directory listing, `registry://listing/...`, which has no root to be under)
    assert {i.uri for i in aliased} <= keys and all(k.startswith(a) or AB.is_synthetic_uri(k) for k in keys), sorted(k for k in keys if not k.startswith(a))


# --------------------------------------------------------------------------------------------
# part 6: dropped inputs -- any dropped input refuses the shard's group; the plan's coverage check; the home tree's copies
# --------------------------------------------------------------------------------------------

import argparse  # noqa: E402

from test_registry_adapter_pass import _roster  # noqa: E402


class NeverDiscoverAdapter(SyntheticStoreAdapter):
    """The synthetic adapter that may be CONSTRUCTED (the plan-time checks read its declarations) and refuses to DISCOVER: what a plan
    handed this registry proves about itself (C6, S2: the grading host reads no store)."""

    def discover(self):
        pytest.fail("the plan must never discover an adapter's inputs on the grading host (C6, S2)")


class TwoRootAdapter(SyntheticStoreAdapter):
    """The synthetic store adapter over TWO roots -- a component's base and an ANCHOR tree no component covers (relu_paired's borrowed MMLU
    anchors under another producer's tree). ``one_group`` reads every store as ONE comparison group, so a missing store is a dropped
    input of the group's head; otherwise one group per store, as the base adapter."""

    def __init__(self, store_root: str, anchor_root: str, one_group: bool = False):
        super().__init__(store_root)
        self.roots = (store_root, anchor_root)
        self.one_group = one_group

    def discover(self):
        out = []
        for root in self.roots:
            if not os.path.isdir(root):
                continue
            for d in sorted(os.listdir(root)):
                p = os.path.join(root, d, "store.json")
                if os.path.lexists(p):
                    out.append(AB.check_input(p, role="report", allow_symlink=self.allow_symlinks))
        return out

    def comparison_group(self, inp):
        return "one_group" if self.one_group else super().comparison_group(inp)


def _one_group_world(tmp_path, n=12):
    """A grading host whose adapter reads n stores under `<home>/stores` (a component's base) plus ONE anchor store under `<home>/anchors` (no
    component) as one group; the first pass folded (a head of n + 1 inputs)."""
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    R.write_records(os.path.join(root, "records", "dims", "000001-dims.jsonl"), synth.seed_dimensions())
    _roster(root)
    H.ingest_once(root, workers=0, min_free=0)
    home = tmp_path / "home"
    stores, anchors = str(home / "stores"), str(home / "anchors")
    os.makedirs(stores)
    os.makedirs(anchors)
    for i in range(n):
        _store(stores, f"s{i:03d}", f"store_{i:03d}", "constitution_lora@aaaa", [0.25 + (i % 3) * 0.25])
    _store(anchors, "a000", "anchor_000", "none", [0.5])
    ad = TwoRootAdapter(stores, anchors, one_group=True)
    with DB.reading(DB.db_path(root)) as conn:
        rep = RUN.run_pass(root, [ad], conn=conn, now=synth.t(2000), producer=PRODUCER)
    assert rep.groups_refreshed == 1, [r.error for r in rep.results]
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    return root, str(home), stores, anchors, ad


def _cluster_pass_over(cluster_root, ad, sidecar, now, components, shard=None):
    """`_cluster_pass` with the STAGED components the shard reads through (`run_shard` fills them from the plan's pins)."""
    conn = RS.open_reader_state(sidecar, [ad.name])
    try:
        cluster = RUN.ClusterRead.from_sidecar(conn, {"substrate": "cluster_a", "host": "node-x", "components": {}})
        cluster.components = dict(components)
        return RUN.run_pass(cluster_root, [ad], conn=conn, now=now, producer=dict(PRODUCER, substrate="cluster_a", host="node-x"), cluster=cluster,
                            shard=shard, save_cache=False)
    finally:
        conn.close()


def test_a_cluster_shard_refuses_a_group_missing_any_input_of_its_head_names_the_covering_component_and_the_pull_names_it(tmp_path):
    """relu_paired's shards saw 9 / 3 / 9 fewer inputs than the grading host -- the borrowed MMLU anchor trees are no component --
    0.026-0.097 of them, under the 0.10 threshold, and MINTED reduced-coverage revisions that superseded the heads for 7 min.
    On the cluster ONE missing input of the head refuses the group: no revision, no marker, no watermark, the missing input named with the
    staged component that would have covered it (or that none does); the shard summary carries it and the pull turns it into the adapter's
    fallback reason. The grading host's path keeps the threshold: the same 1-of-13 drop is accepted there with the comparison on its
    marker."""
    root, home, stores, anchors, ad = _one_group_world(tmp_path)
    source = RUN.source_id_for(ad.name, "one_group")
    (head,) = _rows(root, "SELECT revision_id FROM source_revision WHERE source_id = ? AND accepted = 1", (source,))
    sidecar = str(tmp_path / "rs.sqlite")
    _export(root, sidecar, home, adapters=(ad.name,))
    components = {"synthetic_stores": {"base": stores, "generation": "deadbeef"}}
    anchor_uri = os.path.join(anchors, "a000", "store.json")
    # (a) the node has no anchor tree: 1 of 13 inputs absent (7.7 %, under the grading host threshold) -> REFUSED on the cluster, nothing written
    shutil.move(anchors, anchors + ".grading_host")
    cluster_root = _cluster_root(tmp_path, "cr_a")
    rep = _cluster_pass_over(cluster_root, ad, sidecar, synth.t(3000), components)
    assert rep.groups_refused == 1 and rep.groups_inputs_dropped == 1 and rep.groups_refreshed == 0 and rep.records_written == 0, rep.as_dict()
    (res,) = rep.results
    assert not res.accepted and res.inputs_dropped and res.files == [] and not res.refused_known and not res.held
    assert res.error.startswith(f"{source}: {RUN.STAGED_COPY_INCOMPLETE_KEY}: 1 of the 13 inputs of the previous accepted revision {head[0]} are absent "
                                "from this read on the cluster (7.7%; the grading host threshold does not apply to a staged copy)"), res.error
    assert f"Missing (1 of 1 named): {anchor_uri} [no staged component covers it]" in res.error
    assert not os.path.isdir(os.path.join(H.records_dir(cluster_root), "adapters")) and not os.path.isdir(os.path.join(cluster_root, "staging", "adapters", ad.name))
    # the shard summary carries the refusal by name; the pull turns it into the adapter's fallback reason, the shard's own words included
    window_root = tmp_path / "window_root"
    (window_root / "root" / "records" / "adapters").mkdir(parents=True)
    producer = dict(PRODUCER, substrate="cluster_a", host="node-x")
    doc = CR.write_shard_summary(str(window_root / ad.name / "0-of-1.json"), rep, rep.as_dict(), [ad.name], (0, 1), cluster_root, producer)
    assert doc["ok"] is True and doc["published"] == {} and len(doc["refused"]) == 1
    assert doc["refused"][0]["inputs_dropped"] is True and doc["refused"][0]["group"] == "one_group" and RUN.STAGED_COPY_INCOMPLETE_KEY in doc["refused"][0]["error"]
    assert doc["pass"]["groups_inputs_dropped"] == 1
    plan = {"window": "w1", "host": None, "window_root": str(window_root), "adapters": {ad.name: {"route": "cluster", "shards": 1}},
            "stages": {"revalidate_post": {"stale": []}}, "refusals": {}}
    CR._write_json(str(window_root / ST.ATTESTATION_FILE), _attestation_record("w1", {}))       # the operation's record (the pull's gate is fail-closed)
    prep = CR.pull_window(root, plan, str(tmp_path / "expect"), source=str(window_root))
    assert prep["pulled"] == {} and prep["summaries"][ad.name]["inputs_dropped"] == 1 and prep["summaries"][ad.name]["refused_groups"] == 1
    reasons = prep["adapters_refused"][ad.name]
    assert len(reasons) == 1 and reasons[0].startswith("1 group(s) refused on the cluster: the staged copy lacks input(s) the grading host read -- ") and anchor_uri in reasons[0], reasons
    assert not any("refused by the reader on the cluster" in r for r in reasons)
    fb = CR.refused_adapters(plan, prep)
    assert [r for r in fb[ad.name] if r.startswith("pull: 1 group(s) refused on the cluster: the staged copy lacks input(s)")], fb
    # (b) a store under the component's base absent from the copy: the refusal names the component and its pinned generation
    shutil.move(anchors + ".grading_host", anchors)
    s11 = os.path.join(stores, "s011", "store.json")
    shutil.move(os.path.join(stores, "s011"), str(tmp_path / "s011.grading_host"))
    cluster_root_b = _cluster_root(tmp_path, "cr_b")
    rep_b = _cluster_pass_over(cluster_root_b, ad, sidecar, synth.t(3100), components)
    (res_b,) = rep_b.results
    assert res_b.inputs_dropped and f"{s11} [component synthetic_stores (generation deadbeef)]" in res_b.error, res_b.error
    assert not os.path.isdir(os.path.join(H.records_dir(cluster_root_b), "adapters"))
    shutil.move(str(tmp_path / "s011.grading_host"), os.path.join(stores, "s011"))
    # (c) the grading host path is unchanged: the same 1-of-13 drop is under the threshold, the read is ACCEPTED and its marker records the comparison
    shutil.rmtree(anchors)
    with DB.reading(DB.db_path(root)) as conn:
        rep_d = RUN.run_pass(root, [ad], conn=conn, now=synth.t(3200), producer=PRODUCER)
    (res_d,) = rep_d.results
    assert res_d.accepted and rep_d.groups_refreshed == 1 and rep_d.groups_inputs_dropped == 0, res_d.error
    extra = _marker(os.path.join(H.records_dir(root), "adapters", ad.name, res_d.revision_id))["payload"]["extra"]
    assert extra["inputs_previous"] == 13 and extra["inputs_now"] == 12 and extra["inputs_dropped"] == 1
    assert extra["inputs_dropped_paths"] == ["a000/store.json"]                    # relative to the adapter's declared root, as ever
    assert "inputs_dropped_uris" not in extra                                    # the cluster's naming aid never rides a marker


def test_the_plan_runs_no_grading_host_discovery_and_the_coverage_rule_stays_a_pure_function(tmp_path, monkeypatch, capsys):
    """The plan's coverage check used to run the adapter's discovery ON THE GRADING HOST -- a store read at an entry point. Now
    `plan_window` discovers nothing: an adapter whose inputs include a tree no component covers is still routed to the cluster (the
    shard's discovery over the home tree is the coverage check, and its refusal is the record), the plan carries no `inputs` key, and a
    `load_all` that would construct anything proves it was never asked. `input_coverage` / `uncovered_reason` remain the rule's pure
    reference implementation over a caller's input set, pinned here as before."""
    w = _world_with_generation(tmp_path, monkeypatch)
    anchors = os.path.join(w["home"], "anchors")
    a1 = _store(anchors, "a1", "anchor_1", "none", [0.5])
    ad = TwoRootAdapter(w["stores"], anchors)
    # the plan may CONSTRUCT an adapter (the plan-time checks on the instance) and must never DISCOVER one: a registry whose
    # instances refuse discovery by name proves it
    monkeypatch.setattr("sisteer.registry.adapters.load_all", lambda: {ADAPTER: lambda: NeverDiscoverAdapter(w["stores"])})
    with AB.without_store_access_scope():                    # an unscoped host: a discovery here would refuse by name -- none runs
        plan = CR.plan_window(w["root"], "wc", [ADAPTER], w["cfg"], w["dest"], "a" * 40, uri_home=w["home"], repo_root=REPO_ROOT)
    e = plan["adapters"][ADAPTER]
    assert e["route"] == "cluster" and len(plan["tasks"]) == 2 and "inputs" not in e and plan["refusals"] == {}
    assert e["revalidation_pre"]["synthetic_stores"]["ok"] is True
    # the CLI's plan line, the one window.sh tails into the chain log: no discovery line, the cluster route
    args = argparse.Namespace(action="plan", window="wc2", adapter=[ADAPTER], tree_sha="a" * 40, dest="cluster_a", config=w["cfgp"], repo_root=REPO_ROOT,
                              full=False, full_adapter=[], reread_force_adapter=[], json=False)
    with AB.without_store_access_scope():
        assert CR.cmd_cluster_reads(args, w["root"]) == 0
    out = capsys.readouterr().out
    assert f"{ADAPTER}=cluster; 2 task(s)" in out and "REFUSED" not in out, out
    # the pure rule, unchanged: a member the generation lacks names the component; a repo-resident declaration file is covered by the
    # exported tree; a per-group alias is judged by its file; an input under no component names its root
    s4 = _store(w["stores"], "s4", "legacy_store_4", "none", [0.4])
    with AB.using_hash_cache(None):
        inputs = [AB.check_input(s4, role="report"), AB.check_input(os.path.join(REPO_ROOT, "pyproject.toml"), role="declaration"),
                  AB.check_input(os.path.join(w["stores"], "s1", "store.json"), role="report").with_alias("g1"), AB.check_input(a1, role="report")]
    cov = CR.input_coverage(w["cfg"], w["root"], e["pins"], inputs, REPO_ROOT, w["home"], ad.roots)
    assert cov["discovered"] == 4 and cov["covered"] == 2 and cov["uncovered"] == 2
    assert cov["roots"] == {w["stores"]: {"count": 1, "why": f"under component synthetic_stores's base but not a member of its generation {w['sha8']}", "sample": [s4]},
                            anchors: {"count": 1, "why": "no staged component covers it", "sample": [a1]}}
    assert CR.uncovered_reason(cov).startswith(f"{CR.UNCOVERED_PREFIX}: 2 of 4 -- ")
    assert not hasattr(CR, "grading_host_adapter") and not hasattr(CR, "grading_host_inputs")          # the grading host discovery helpers are gone


def _tree_listing(home: str) -> Dict[str, tuple]:
    """Every entry under `home` by lstat -- kind, mode, size, mtime_ns, inode, and a link's target -- the lock file aside (init opens it);
    two equal listings mean a build left the tree byte for byte as it found it (atime excluded: the preflight reads)."""
    out: Dict[str, tuple] = {}
    for dirpath, dirnames, filenames in os.walk(home, followlinks=False):
        for name in sorted(dirnames + filenames):
            path = os.path.join(dirpath, name)
            if path == os.path.join(home, CR.NAMESPACE_LOCK):
                continue
            st = os.lstat(path)
            out[path] = (stat.S_IFMT(st.st_mode), st.st_mode, st.st_size, st.st_mtime_ns, st.st_ino, os.readlink(path) if stat.S_ISLNK(st.st_mode) else None)
    return out


def _old_builder_promotion(base: str, target: str, children) -> None:
    """The residue the builder BEFORE hotfix 2 left on a node when a build promoted a shared base to a whole link and then refused elsewhere:
    the base's recorded children withdrawn, the base itself a link onto the generation, the previous window's marker standing. A refusing
    build now touches nothing, but the nodes that built through the 2026-10-07 outage carry exactly this, so the retry and later-window
    rules stay pinned over it, made by hand."""
    for c in children:
        os.remove(c)
    os.rmdir(base)
    os.symlink(target, base)


def test_the_home_tree_replaces_a_previous_windows_recorded_copy_and_refuses_an_unmanaged_path_untouched(tmp_path):
    """On 2026-09-29 every shard of three adapters refused `home tree: ...: cannot copy ...: [Errno 13] Permission denied` on the eight
    regular-file members of the shared `run-state/full22` base: an earlier build had copied them with copy2, which keeps the sealed
    generation's read-only mode, and the next window's build wrote over those copies. A copy a previous window RECORDED in the node's
    namespace (its `files` / `links`) is replaced -- removed, then copied again; a path at a copy spelling that NO previous window
    recorded is refused by name and left untouched, bytes and mode (a file of the user's own at that spelling is not this machinery's
    to delete). Since 2026-10-07 a build that refuses writes nothing new, so in (e) the recorded copy beside the refused path is
    withheld (`not_attempted`), not copied."""
    gen = tmp_path / "gen"
    gen.mkdir()
    src = gen / "stats.json"
    src.write_text('{"a": 1}')
    os.chmod(src, 0o444)
    home = tmp_path / "home"
    dst = str(home / "shared" / "stats.json")
    spec = {"home": str(home), "dirs": [str(home / "shared")], "links": [], "files": [[dst, str(src)]]}
    marker = home / CR.NAMESPACE_MARKER
    # (a) a real file of the user's own at the copy path, no previous window on this node: REFUSED, untouched, no namespace recorded
    os.makedirs(home / "shared")
    with open(dst, "w") as fh:
        fh.write("mine")
    os.chmod(dst, 0o600)
    r0 = CR.init_home_tree(spec, str(home), "w0")
    assert r0["mode"] == "built" and r0["copied"] == [] and len(r0["refused"]) == 1 and r0["refused"][0]["path"] == dst, r0
    assert "does not manage" in r0["refused"][0]["why"] and "recorded" in r0["refused"][0]["why"]
    assert open(dst).read() == "mine" and (os.stat(dst).st_mode & 0o777) == 0o600 and not marker.exists()
    os.remove(dst)
    # (b) the first window copies: the copy carries the generation's read-only mode and the namespace records it
    first = CR.init_home_tree(spec, str(home), "w1")
    assert first["mode"] == "built" and first["copied"] == [dst] and first["refused"] == [], first
    # the sealed property is the MODE (no write bit for anyone), not os.access, which answers writable for root whatever the bits
    assert (stat.S_IMODE(os.stat(dst).st_mode) & 0o222) == 0 and open(dst).read() == '{"a": 1}' and json.load(open(marker))["spec"]["files"] == [[dst, str(src)]]
    # (c) the next window pins other bytes: the RECORDED copy is replaced, never written over
    os.chmod(src, 0o644)
    src.write_text('{"a": 2}')
    os.chmod(src, 0o444)
    second = CR.init_home_tree(spec, str(home), "w2")
    assert second["mode"] == "built" and second["previous_window"] == "w1" and second["copied"] == [dst] and second["refused"] == [], second
    assert open(dst).read() == '{"a": 2}' and CR._verify_home_tree(spec) == []
    # (d) a link where the recorded copy belongs (the path is recorded, whatever now stands there): replaced the same way
    os.remove(dst)
    os.symlink(str(src), dst)
    third = CR.init_home_tree(spec, str(home), "w3")
    assert third["copied"] == [dst] and third["refused"] == [] and not os.path.islink(dst)
    # (e) a second copy path this window adds, where an UNRECORDED real file stands: refused and untouched, while the recorded copy is replaced
    src2 = gen / "more.json"
    src2.write_text('{"b": 1}')
    dst2 = str(home / "shared" / "more.json")
    with open(dst2, "w") as fh:
        fh.write("theirs")
    spec2 = dict(spec, files=[[dst, str(src)], [dst2, str(src2)]])
    fourth = CR.init_home_tree(spec2, str(home), "w4")
    assert fourth["copied"] == [] and fourth["not_attempted"] == [dst] and [r["path"] for r in fourth["refused"]] == [dst2], fourth
    assert "does not manage" in fourth["refused"][0]["why"] and fourth["adopted"] == []
    assert open(dst2).read() == "theirs" and open(dst).read() == '{"a": 2}'
    assert json.load(open(marker))["window"] == "w3"                                   # a refused build records no namespace


def test_the_home_tree_replaces_a_real_directory_its_own_previous_build_made_when_a_component_is_re_based_onto_it(tmp_path, monkeypatch):
    """2026-10-07 13:48Z: every shard of every cluster-read tick refused `home tree: .../reseed_eb64: exists and is not a symlink; refusing
    to replace a real path`. The census landing (#470) re-based `hpo_reseed_eb64` from `.../reseed_eb64/hpo` onto `.../reseed_eb64` (members
    `[hpo, recon]`): under the old layout the builder made `reseed_eb64/` a REAL directory (a nesting parent, in `dirs`) holding the `hpo`
    link; under the new one `reseed_eb64` is linked whole, and the builder met the real directory its own previous build had made. A real
    directory where a link belongs is replaced only when (a) the previous window's recorded namespace names it in `dirs` AND (b) nothing
    but links and directories of links stands under it once that window's entries are withdrawn; the inverse (a whole link where a real
    directory now belongs) was already handled and still is."""
    home = str(tmp_path / "home")
    monkeypatch.setattr(ST, "HOME", home)                         # Component.home_rel against the module's HOME
    x, x_hpo = os.path.join(home, "stage/base/x"), os.path.join(home, "stage/base/x/hpo")
    gen_a, comps_a, pins_a = _components(tmp_path / "a", home, {"hpo_reseed": ("stage/base/x/hpo", ["best/", "trials.json"])})
    gen_b, comps_b, pins_b = _components(tmp_path / "b", home, {"hpo_reseed": ("stage/base/x", ["hpo/", "recon/"])})
    spec_a = CR.home_tree_spec(comps_a, pins_a, gen_a, home, _entries_of(gen_a))
    spec_b = CR.home_tree_spec(comps_b, pins_b, gen_b, home, _entries_of(gen_b))
    assert x in spec_a["dirs"] and dict(map(tuple, spec_a["links"])) == {x_hpo: f"{gen_a}/hpo_reseed/{pins_a['hpo_reseed']}"}
    assert x not in spec_b["dirs"] and dict(map(tuple, spec_b["links"])) == {x: f"{gen_b}/hpo_reseed/{pins_b['hpo_reseed']}"}
    marker = os.path.join(home, CR.NAMESPACE_MARKER)
    # layout A: `x/` is a real directory of the builder's own holding the `hpo` link; the namespace records it among its dirs
    first = CR.init_home_tree(spec_a, home, "w1")
    assert first["mode"] == "built" and first["created"] == [x_hpo] and first["refused"] == [], first
    assert os.path.isdir(x) and not os.path.islink(x) and os.path.islink(x_hpo) and x in json.load(open(marker))["spec"]["dirs"]
    # layout B, the next window: the previous window's `hpo` link is withdrawn, the real `x/` it made is removed and `x` links whole
    second = CR.init_home_tree(spec_b, home, "w2")
    assert second["mode"] == "built" and second["previous_window"] == "w1" and second["refused"] == [], second
    assert second["removed"] == [x_hpo, x] and second["created"] == [x] and second["repointed"] == [] and second["unchanged"] == []
    assert os.path.islink(x) and os.readlink(x) == f"{gen_b}/hpo_reseed/{pins_b['hpo_reseed']}" and os.path.isfile(os.path.join(x, "recon/x.json"))
    assert CR._verify_home_tree(spec_b) == [] and json.load(open(marker))["window"] == "w2" and x not in json.load(open(marker))["spec"]["dirs"]
    # a second build of B (another window, the same plan) finds the link in place: unchanged, nothing removed or created
    third = CR.init_home_tree(spec_b, home, "w3")
    assert third["mode"] == "built" and third["unchanged"] == [x] and third["removed"] == [] and third["created"] == [], third
    # and the inverse, back to layout A: the whole link is removed where a real directory belongs (the rule that already stood)
    fourth = CR.init_home_tree(spec_a, home, "w4")
    assert fourth["refused"] == [] and fourth["removed"] == [x] and fourth["created"] == [x_hpo] and os.path.isdir(x) and not os.path.islink(x), fourth


def test_the_home_tree_keeps_refusing_a_real_directory_that_holds_a_regular_file_or_that_no_previous_build_recorded(tmp_path, monkeypatch):
    """The other side of the rule above: anything under the directory the previous build did NOT record -- a regular file, a symlink, a
    directory, at any depth -- keeps today's refusal by name and the directory stands with it inside (the record's `dirs` is of what was
    REQUIRED, `makedirs` having accepted whatever stood there, so ownership is judged entry by entry); a real directory no previous build
    recorded in `dirs` -- a first-ever build on the node, or a recorded namespace without the field -- refuses the same way however clean
    it is; recorded directories beneath are removed bottom-up."""
    home = str(tmp_path / "home")
    monkeypatch.setattr(ST, "HOME", home)
    x, x_hpo = os.path.join(home, "stage/base/x"), os.path.join(home, "stage/base/x/hpo")
    gen_a, comps_a, pins_a = _components(tmp_path / "a", home, {"hpo_reseed": ("stage/base/x/hpo", ["best/"])})
    gen_b, comps_b, pins_b = _components(tmp_path / "b", home, {"hpo_reseed": ("stage/base/x", ["hpo/", "recon/"])})
    spec_a = CR.home_tree_spec(comps_a, pins_a, gen_a, home, _entries_of(gen_a))
    spec_b = CR.home_tree_spec(comps_b, pins_b, gen_b, home, _entries_of(gen_b))
    marker = os.path.join(home, CR.NAMESPACE_MARKER)
    assert CR.init_home_tree(spec_a, home, "w1")["refused"] == [] and os.path.islink(x_hpo)
    # (a) a stray regular file of the user's own in the builder's directory: refused by name; the directory and the file stand (the
    # previous window's own `hpo` link is withdrawn as before -- it is the builder's -- and nothing else is removed)
    notes = os.path.join(x, "notes.txt")
    with open(notes, "w") as fh:
        fh.write("mine")
    refused = CR.init_home_tree(spec_b, home, "w2")
    assert refused["mode"] == "built" and [r["path"] for r in refused["refused"]] == [x] and "refusing to replace a real path" in refused["refused"][0]["why"], refused
    # (since hotfix 2 a refused build touches nothing: the previous window's own `hpo` link is NOT withdrawn, only listed as not attempted)
    assert os.path.isdir(x) and not os.path.islink(x) and open(notes).read() == "mine" and refused["removed"] == [] and refused["created"] == []
    assert os.path.islink(x_hpo) and refused["not_attempted"] == [x_hpo]
    assert json.load(open(marker))["window"] == "w1"                                   # a refused build records no namespace
    # (b) an UNRECORDED directory, empty or not, and an unrecorded symlink (an alias of the user's own) refuse the same way, untouched
    os.remove(notes)
    os.makedirs(os.path.join(x, "sub/deeper"))
    again = CR.init_home_tree(spec_b, home, "w2")
    assert [r["path"] for r in again["refused"]] == [x] and os.path.isdir(os.path.join(x, "sub/deeper")) and again["removed"] == [], again
    os.rmdir(os.path.join(x, "sub/deeper"))
    os.rmdir(os.path.join(x, "sub"))
    os.symlink(gen_a, os.path.join(x, "alias"))
    alias = CR.init_home_tree(spec_b, home, "w2")
    assert [r["path"] for r in alias["refused"]] == [x] and os.path.islink(os.path.join(x, "alias")) and alias["removed"] == [], alias
    os.remove(os.path.join(x, "alias"))
    ok = CR.init_home_tree(spec_b, home, "w2")                                        # nothing unrecorded left: the builder's own, replaced
    assert ok["refused"] == [] and ok["removed"] == [x_hpo, x] and ok["created"] == [x] and os.path.islink(x), ok
    # (c) no previous build on this node: a real directory at the link path, however clean, is not the builder's -- refused, untouched
    home2 = str(tmp_path / "home2")
    monkeypatch.setattr(ST, "HOME", home2)
    gen_c, comps_c, pins_c = _components(tmp_path / "c", home2, {"hpo_reseed": ("stage/base/x", ["hpo/"])})
    spec_c = CR.home_tree_spec(comps_c, pins_c, gen_c, home2, _entries_of(gen_c))
    x2 = os.path.join(home2, "stage/base/x")
    os.makedirs(x2)
    fresh = CR.init_home_tree(spec_c, home2, "w1")
    assert [r["path"] for r in fresh["refused"]] == [x2] and os.path.isdir(x2) and not os.path.islink(x2) and fresh["removed"] == [], fresh
    # (d) a recorded namespace without `dirs` is accepted, and names nothing: refused the same way; one that names it: replaced
    direct = CR._build_home_tree(spec_c, {"links": [], "files": []})
    assert [r["path"] for r in direct["refused"]] == [x2] and os.path.isdir(x2) and not os.path.islink(x2) and direct["removed"] == [], direct
    sub = os.path.join(x2, "sub")
    os.makedirs(os.path.join(sub, "deeper"))
    direct2 = CR._build_home_tree(spec_c, {"links": [], "files": [], "dirs": [x2, sub]})
    assert [r["path"] for r in direct2["refused"]] == [x2] and os.path.isdir(os.path.join(sub, "deeper")), direct2   # `deeper` unrecorded
    direct3 = CR._build_home_tree(spec_c, {"links": [], "files": [], "dirs": [x2, sub, os.path.join(sub, "deeper")]})
    assert direct3["refused"] == [] and direct3["removed"] == [os.path.join(sub, "deeper"), sub, x2] and direct3["created"] == [x2] and os.path.islink(x2), direct3


def test_a_promoted_base_is_not_withdrawn_through_on_the_retry_of_a_build_a_refusal_elsewhere_left_unrecorded(tmp_path, monkeypatch):
    """The review of the hotfix: a previous window laid the shared base `full22` out entry by entry, COPIES included (its
    `transfer_stats.json` in the marker's `files`); a later window selecting one component alone links the base whole, so the build promotes
    it -- and if ANOTHER path refuses in the same build, no marker is written and the previous spec stands. On the retry the withdrawal loop
    met the copied file's pathname as managed, `isfile` followed the now-symlinked ancestor INTO the generation and `remove` reached the sealed
    (0555) generation: an uncaught PermissionError, every later build stranded (a writable generation would silently lose the file). A
    recorded entry whose spelling runs through a symlink is skipped by name, never stat'ed or removed through the link, and the promoted
    base itself reads as `unchanged`."""
    home = str(tmp_path / "home")
    monkeypatch.setattr(ST, "HOME", home)
    full22, sweep = os.path.join(home, "rs/full22"), os.path.join(home, "specs/sweep")
    gen_a, comps_a, pins_a = _components(tmp_path / "a", home, {"multiturn": ("rs/full22", ["grades_a/", "transfer_stats.json"]),
                                                               "misalign": ("rs/full22", ["misalign/"]), "sidecars": ("specs/sweep", ["mm.json"])})
    spec_a = CR.home_tree_spec(comps_a, pins_a, gen_a, home, _entries_of(gen_a))
    copy = os.path.join(full22, "transfer_stats.json")
    assert full22 in spec_a["dirs"] and [p for p, _ in spec_a["files"]] == [copy]
    assert CR.init_home_tree(spec_a, home, "w1")["refused"] == [] and os.path.isfile(copy) and not os.path.islink(copy)
    # the next window selects `multiturn` alone: `full22` links whole; the exclusive `specs/sweep` link is obstructed by a real directory
    spec_b = CR.home_tree_spec({s: comps_a[s] for s in ("multiturn", "sidecars")}, {s: pins_a[s] for s in ("multiturn", "sidecars")}, gen_a, home, _entries_of(gen_a))
    assert dict(map(tuple, spec_b["links"]))[full22] == f"{gen_a}/multiturn/{pins_a['multiturn']}" and full22 not in spec_b["dirs"]
    os.remove(sweep)
    os.makedirs(sweep)
    with open(os.path.join(sweep, "notes.txt"), "w") as fh:
        fh.write("mine")
    first = CR.init_home_tree(spec_b, home, "w2")
    assert [r["path"] for r in first["refused"]] == [sweep] and first["created"] == [] and first["removed"] == [], first   # hotfix 2: untouched
    assert os.path.isdir(full22) and not os.path.islink(full22) and json.load(open(os.path.join(home, CR.NAMESPACE_MARKER)))["window"] == "w1"
    # the builder before hotfix 2 promoted the base all the same (children withdrawn, the base linked whole, the marker withheld): the
    # outage's nodes carry that residue, so it is made by hand here and the retry runs over it
    _old_builder_promotion(full22, f"{gen_a}/multiturn/{pins_a['multiturn']}", [os.path.join(full22, "grades_a"), os.path.join(full22, "misalign"), copy])
    gen_file = f"{gen_a}/multiturn/{pins_a['multiturn']}/transfer_stats.json"
    assert os.path.isfile(gen_file) and os.path.isfile(copy)                           # `copy` now spells the generation's own file
    # the retry, the obstruction gone and the generations sealed as the cluster's are: nothing is removed through the promoted link
    shutil.rmtree(sweep)
    sealed = [os.path.join(dp, d) for dp, dns, _ in os.walk(gen_a) for d in dns] + [gen_a]
    for d in sealed:
        os.chmod(d, 0o555)
    try:
        retry = CR.init_home_tree(spec_b, home, "w2")
    finally:
        for d in sealed:
            os.chmod(d, 0o755)
    assert retry["refused"] == [] and retry["unchanged"] == [full22] and retry["created"] == [sweep] and retry["removed"] == [], retry
    assert retry["skipped_under_link"] == [os.path.join(full22, "grades_a"), os.path.join(full22, "misalign"), copy]
    assert os.path.isfile(gen_file) and os.path.isdir(f"{gen_a}/multiturn/{pins_a['multiturn']}/grades_a") and os.path.isdir(f"{gen_a}/misalign/{pins_a['misalign']}/misalign")
    assert CR._verify_home_tree(spec_b) == [] and json.load(open(os.path.join(home, CR.NAMESPACE_MARKER)))["window"] == "w2"


def test_an_unrecorded_alias_of_the_users_own_under_a_shared_base_survives_the_window_that_links_the_base_whole(tmp_path, monkeypatch):
    """`makedirs(exist_ok=True)` accepts a directory of the user's own at a base, and the build records it among its `dirs` all the
    same -- the record is of what the namespace REQUIRED, not of ownership. A parent component (`specs`) with another nesting beneath
    it lays `specs/` out as a real directory; a later window selecting the parent alone links `specs` whole. An alias the user keeps
    there (`specs/user-project -> <outside>`), recorded by nobody, must survive: the replacement refuses by name and nothing under the
    alias is touched."""
    home = str(tmp_path / "home")
    monkeypatch.setattr(ST, "HOME", home)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "theirs.txt").write_text("theirs")
    specs = os.path.join(home, "specs")
    alias = os.path.join(specs, "user-project")
    os.makedirs(specs)
    os.symlink(str(outside), alias)                                                  # before any build: the user's own
    gen, comps, pins = _components(tmp_path / "g", home, {"bcb_specs": ("specs", ["bcb.json", "bfcl/"]), "sidecars": ("specs/relu_paired", ["a.json"])})
    broad = CR.home_tree_spec(comps, pins, gen, home, _entries_of(gen))
    assert specs in broad["dirs"]
    first = CR.init_home_tree(broad, home, "w1")
    assert first["refused"] == [] and os.path.islink(alias) and os.path.isdir(specs) and not os.path.islink(specs), first
    assert specs in json.load(open(os.path.join(home, CR.NAMESPACE_MARKER)))["spec"]["dirs"]
    parent_only = CR.home_tree_spec({"bcb_specs": comps["bcb_specs"]}, {"bcb_specs": pins["bcb_specs"]}, gen, home, _entries_of(gen))
    assert dict(map(tuple, parent_only["links"])) == {specs: f"{gen}/bcb_specs/{pins['bcb_specs']}"}
    second = CR.init_home_tree(parent_only, home, "w2")
    assert [r["path"] for r in second["refused"]] == [specs] and "refusing to replace a real path" in second["refused"][0]["why"], second
    # the builder's own entries are NOT withdrawn either (hotfix 2: a refused build touches nothing); they are listed as not attempted,
    # and the directory, the alias and what it points at stand
    own = [os.path.join(specs, "bcb.json"), os.path.join(specs, "bfcl"), os.path.join(specs, "relu_paired")]
    assert second["removed"] == [] and sorted(second["not_attempted"]) == sorted(own), second
    assert os.path.isdir(specs) and not os.path.islink(specs) and os.path.islink(alias) and os.readlink(alias) == str(outside)
    assert (outside / "theirs.txt").read_text() == "theirs" and sorted(os.listdir(specs)) == ["bcb.json", "bfcl", "relu_paired", "user-project"]
    assert json.load(open(os.path.join(home, CR.NAMESPACE_MARKER)))["window"] == "w1"


def test_the_shard_formats_a_home_tree_refusal_as_the_live_tick_printed_it(tmp_path, monkeypatch):
    """The string every shard of every tick printed from 2026-10-07 13:48Z, pinned as the shard composes it from the builder's refusal:
    `home tree: <path>: exists and is not a symlink; refusing to replace a real path` -- a real directory of the user's own where the
    component's base links whole; the summary says so (rc 0), the obstruction stands, and the shard reads once it is gone."""
    w = _world_with_generation(tmp_path, monkeypatch)
    plan, plan_file, info = _plan_and_stage(w, tmp_path)
    _node(w)
    os.makedirs(w["stores"])
    with open(os.path.join(w["stores"], "notes.txt"), "w") as fh:
        fh.write("mine")
    assert CR.run_shard(f"{plan['window_root']}/plan.json", 0, home=w["home"], summaries_dir=str(tmp_path / "s-obstructed")) == 0
    d = json.load(open(CR.summary_path(str(tmp_path / "s-obstructed"), ADAPTER, 0, 2)))
    assert d["ok"] is False and d["refusal"] == f"home tree: {w['stores']}: exists and is not a symlink; refusing to replace a real path", d["refusal"]
    assert d["home_tree"]["refusals"] == [{"path": w["stores"], "why": "exists and is not a symlink; refusing to replace a real path"}] and d["home_tree"]["mode"] == "built"
    assert os.path.isdir(w["stores"]) and not os.path.islink(w["stores"]) and open(os.path.join(w["stores"], "notes.txt")).read() == "mine"
    assert not os.path.exists(os.path.join(w["home"], CR.NAMESPACE_MARKER))
    shutil.rmtree(w["stores"])
    assert CR.run_shard(f"{plan['window_root']}/plan.json", 0, home=w["home"]) == 0
    d2 = json.load(open(CR.summary_path(plan["summaries_dir"], ADAPTER, 0, 2)))
    assert d2["ok"] is True and os.path.islink(w["stores"]) and os.readlink(w["stores"]) == f"{w['gen_dir']}/synthetic_stores/{w['sha8']}"


def _withheld_promotion(tmp_path, home, tag):
    """The state the withheld promotion starts from: the first window lays the shared base `rs/full22` out entry by entry (children
    recorded, the base in `dirs`); the second window selects `multiturn` alone so the base links whole, and a real directory with a stray
    file where `specs/sweep` links refuses the build, so the first window's marker stands over a PROMOTED base -- the promotion made by
    hand (a refused build touches nothing now; the outage's nodes carry the residue). Returns the generation root, the three specs, the
    base and its children."""
    full22, sweep = os.path.join(home, "rs/full22"), os.path.join(home, "specs/sweep")
    gen, comps, pins = _components(tmp_path / f"g{tag}", home, {"multiturn": ("rs/full22", ["grades_a/", "transfer_stats.json"]),
                                                                "misalign": ("rs/full22", ["misalign/"]), "sidecars": ("specs/sweep", ["mm.json"])})
    spec_of = lambda *slugs: CR.home_tree_spec({s: comps[s] for s in slugs}, {s: pins[s] for s in slugs}, gen, home, _entries_of(gen))
    broad, promote, others = spec_of("multiturn", "misalign", "sidecars"), spec_of("multiturn", "sidecars"), spec_of("sidecars")
    assert broad["gen_dir"] == promote["gen_dir"] == others["gen_dir"] == gen and full22 in broad["dirs"]
    assert full22 not in others["dirs"] and full22 not in dict(map(tuple, others["links"])) and dict(map(tuple, promote["links"]))[full22] == f"{gen}/multiturn/{pins['multiturn']}"
    assert CR.init_home_tree(broad, home, f"w1{tag}")["refused"] == [] and os.path.isdir(full22) and not os.path.islink(full22)
    if os.path.lexists(sweep):
        os.remove(sweep)
    os.makedirs(sweep)
    with open(os.path.join(sweep, "notes.txt"), "w") as fh:
        fh.write("mine")
    w2 = CR.init_home_tree(promote, home, f"w2{tag}")
    assert [r["path"] for r in w2["refused"]] == [sweep] and w2["removed"] == [] and w2["created"] == [] and not os.path.islink(full22), w2   # hotfix 2
    assert json.load(open(os.path.join(home, CR.NAMESPACE_MARKER)))["window"] == f"w1{tag}"
    children = [os.path.join(full22, e) for e in ("grades_a", "misalign", "transfer_stats.json")]
    _old_builder_promotion(full22, f"{gen}/multiturn/{pins['multiturn']}", children)   # what the builder before hotfix 2 left behind
    shutil.rmtree(sweep)
    return gen, pins, {"broad": broad, "promote": promote, "others": others}, full22, sweep, children


def test_a_base_a_withheld_build_promoted_is_removed_when_a_later_window_drops_it_and_handled_by_the_standing_branches_when_it_names_it(tmp_path, monkeypatch):
    """The first window lays shared base B out (children recorded, B in `dirs`); the second window promotes B to a whole link but
    refuses elsewhere, so the first window's marker stands; the third window -- a tick selecting other adapters -- DROPS B. The
    withdrawal skipped the first window's children (their ancestor a link) and nothing named B itself (a recorded directory, never a
    managed link): the third window recorded its namespace and B stood for good as an UNRECORDED link at a canonical spelling onto an
    old generation, whose bytes a later adapter rooted beneath it could discover and `GenerationHashCache.admit` would accept (a
    spelling under no pinned base is admitted as ever). Now a symlinked ancestor this window does not name, which the previous window
    recorded as a real directory and whose literal target lies under the plan's generations root, is removed (a symlink: the generation
    is never touched) and named under `removed`; the children stay listed under `skipped_under_link`. When this window DOES name it,
    the standing branches handle it: a real directory again (the dirs loop removes the link), a whole link again (re-pointed;
    `unchanged` for the same target is pinned by the retry test above)."""
    home = str(tmp_path / "home")
    monkeypatch.setattr(ST, "HOME", home)
    marker = os.path.join(home, CR.NAMESPACE_MARKER)
    # W3 drops B: the promoted link is removed by name, the generation untouched, the marker recorded without B
    gen, pins, specs, full22, sweep, children = _withheld_promotion(tmp_path, home, "a")
    gen_file = f"{gen}/multiturn/{pins['multiturn']}/transfer_stats.json"
    w3 = CR.init_home_tree(specs["others"], home, "w3a")
    assert w3["refused"] == [] and w3["removed"] == [full22] and w3["skipped_under_link"] == children and w3["created"] == [sweep], w3
    assert not os.path.lexists(full22) and os.path.isdir(os.path.join(home, "rs")) and os.path.isfile(gen_file)
    assert os.path.isdir(f"{gen}/multiturn/{pins['multiturn']}/grades_a") and os.path.isdir(f"{gen}/misalign/{pins['misalign']}/misalign")
    rec = json.load(open(marker))
    assert rec["window"] == "w3a" and full22 not in rec["spec"]["dirs"] and full22 not in {p for p, _ in rec["spec"]["links"]} and CR._verify_home_tree(specs["others"]) == []
    # W3 needs B as a real directory again (W1's own layout): the dirs loop removes the link, the children are laid out, the copy copied
    gen, pins, specs, full22, sweep, children = _withheld_promotion(tmp_path, home, "b")
    back = CR.init_home_tree(specs["broad"], home, "w3b")
    assert back["refused"] == [] and back["removed"] == [full22] and back["skipped_under_link"] == [] and back["copied"] == [children[2]], back
    assert sorted(back["created"]) == sorted(children[:2] + [sweep]) and os.path.isdir(full22) and not os.path.islink(full22) and CR._verify_home_tree(specs["broad"]) == []
    # W3 links B whole again, onto another generation: re-pointed; the children stay skipped and nothing is removed
    gen, pins, specs, full22, sweep, children = _withheld_promotion(tmp_path, home, "c")
    again = dict(specs["promote"], links=[[p, (t + "-next" if p == full22 else t)] for p, t in specs["promote"]["links"]])
    re_pointed = CR.init_home_tree(again, home, "w3c")
    assert re_pointed["refused"] == [] and re_pointed["repointed"] == [full22] and re_pointed["removed"] == [] and re_pointed["skipped_under_link"] == children, re_pointed
    assert os.readlink(full22) == f"{gen}/multiturn/{pins['multiturn']}-next" and json.load(open(marker))["window"] == "w3c"


def test_a_promoted_link_the_builder_cannot_vouch_for_is_refused_by_name_and_left_standing(tmp_path, monkeypatch):
    """The negative of the rule above: the same three windows, but before W3 the base's link points OUTSIDE the plan's generations root --
    or the previous namespace never recorded a real directory at its spelling, or the spec carries no generations root. Refused by name,
    the link left exactly as found, the marker not advanced: a link the builder cannot vouch for is not its to delete."""
    home = str(tmp_path / "home")
    monkeypatch.setattr(ST, "HOME", home)
    marker = os.path.join(home, CR.NAMESPACE_MARKER)
    gen, pins, specs, full22, sweep, children = _withheld_promotion(tmp_path, home, "n")
    elsewhere = str(tmp_path / "elsewhere")
    os.makedirs(elsewhere)
    os.remove(full22)
    os.symlink(elsewhere, full22)
    w3 = CR.init_home_tree(specs["others"], home, "w3n")
    assert [r["path"] for r in w3["refused"]] == [full22] and "cannot vouch for" in w3["refused"][0]["why"] and elsewhere in w3["refused"][0]["why"], w3
    assert "does not lie under the generations root" in w3["refused"][0]["why"] and w3["removed"] == [] and w3["skipped_under_link"] == children
    assert os.path.islink(full22) and os.readlink(full22) == elsewhere and json.load(open(marker))["window"] == "w1n"
    # the builder's own target, but a recorded namespace that never named a real directory there: refused the same way
    os.remove(full22)
    os.symlink(f"{gen}/multiturn/{pins['multiturn']}", full22)
    prev = json.load(open(marker))["spec"]
    direct = CR._build_home_tree(specs["others"], dict(prev, dirs=[d for d in prev["dirs"] if d != full22]))
    assert [r["path"] for r in direct["refused"]] == [full22] and "no previous window recorded a real directory there" in direct["refused"][0]["why"] and os.path.islink(full22), direct
    # a spec without a generations root cannot vouch either
    direct2 = CR._build_home_tree({k: v for k, v in specs["others"].items() if k != "gen_dir"}, prev)
    assert [r["path"] for r in direct2["refused"]] == [full22] and "no generations root" in direct2["refused"][0]["why"] and os.path.islink(full22), direct2
    # both conditions met: removed, by the same direct call
    direct3 = CR._build_home_tree(specs["others"], prev)
    assert direct3["refused"] == [] and direct3["removed"] == [full22] and not os.path.lexists(full22), direct3


def test_a_refusing_build_writes_no_copy_and_the_next_build_adopts_a_byte_identical_unrecorded_one(tmp_path, monkeypatch):
    """Since 2026-10-07 17:34Z, every tick: six adapters sharing `run-state/full22` refused `home tree: .../attacker_effort_halfdose.json:
    exists and is not a copy or link a previous window recorded in this node's namespace; refusing to replace a path this machinery does not
    manage`. The file had become a member after the reads census; the builds of the 13:48-17:04Z outage refused at a LINK and ran their
    copy loop all the same, so the file was copied onto every node by builds whose marker never advanced, and the fixed builder then found it
    unrecorded. Two rules: a build that has refused writes nothing new (the admitted copies are listed under `not_attempted`, none written);
    and a regular file no window recorded whose bytes ARE the generation's file's is adopted (`adopted`) -- not written at all when its mtime
    is the generation's too, re-copied under the generation's mtime otherwise (the pass compares `(sha256, size, mtime_utc)` with the stored
    watermark) -- and the marker the successful build writes names it among `files`, so the next build manages it as any copy."""
    home = str(tmp_path / "home")
    monkeypatch.setattr(ST, "HOME", home)
    marker = os.path.join(home, CR.NAMESPACE_MARKER)
    full22, sweep = os.path.join(home, "rs/full22"), os.path.join(home, "specs/sweep")
    layout = lambda members: {"multiturn": ("rs/full22", ["grades_a/", "transfer_stats.json"] + members), "misalign": ("rs/full22", ["misalign/"]),
                              "sidecars": ("specs/sweep", ["mm.json"])}
    gen1, comps1, pins1 = _components(tmp_path / "g1", home, layout([]))
    spec1 = CR.home_tree_spec(comps1, pins1, gen1, home, _entries_of(gen1))
    stats = os.path.join(full22, "transfer_stats.json")
    assert CR.init_home_tree(spec1, home, "w1")["refused"] == [] and os.path.isfile(stats) and [p for p, _ in json.load(open(marker))["spec"]["files"]] == [stats]
    # W2: the census added a file member F to the shared base, and a real directory with a stray file obstructs the exclusive link
    gen2, comps2, pins2 = _components(tmp_path / "g2", home, layout(["attacker_effort.json"]))
    src_f = f"{gen2}/multiturn/{pins2['multiturn']}/attacker_effort.json"
    with open(src_f, "w") as fh:
        fh.write('{"dose": 0.5, "effort": [1, 2, 3]}')
    spec2 = CR.home_tree_spec(comps2, pins2, gen2, home, _entries_of(gen2))
    f = os.path.join(full22, "attacker_effort.json")
    assert [p for p, _ in spec2["files"]] == [f, stats]
    os.remove(sweep)
    os.makedirs(sweep)
    with open(os.path.join(sweep, "notes.txt"), "w") as fh:
        fh.write("mine")
    grades_a, misalign = os.path.join(full22, "grades_a"), os.path.join(full22, "misalign")
    before = _tree_listing(home)
    w2 = CR.init_home_tree(spec2, home, "w2")
    assert [r["path"] for r in w2["refused"]] == [sweep] and "not a symlink" in w2["refused"][0]["why"], w2
    # nothing written: F is not copied, the recorded copy is not re-copied, the two links are not re-pointed onto the new generation;
    # every planned write is listed as not attempted, the tree is byte for byte as found and W1's marker stands
    assert w2["copied"] == [] and w2["adopted"] == [] and w2["repointed"] == [] and w2["created"] == [] and w2["removed"] == [], w2
    assert w2["not_attempted"] == [grades_a, misalign, f, stats] and not os.path.lexists(f) and _tree_listing(home) == before
    assert os.readlink(grades_a) == f"{gen1}/multiturn/{pins1['multiturn']}/grades_a" and json.load(open(marker))["window"] == "w1"
    # the outage's residue: F stands at its spelling as a byte-identical copy (copy2, as the builder makes them) that no marker records
    shutil.copy2(src_f, f)
    st_before = os.stat(f)
    assert os.path.getsize(f) == os.path.getsize(src_f) and st_before.st_mtime_ns == os.stat(src_f).st_mtime_ns
    shutil.rmtree(sweep)
    w3 = CR.init_home_tree(spec2, home, "w3")
    assert w3["refused"] == [] and w3["adopted"] == [f] and w3["copied"] == [stats] and w3["created"] == [sweep] and w3["not_attempted"] == [], w3
    assert w3["repointed"] == [grades_a, misalign] and os.readlink(grades_a) == f"{gen2}/multiturn/{pins2['multiturn']}/grades_a"
    st_after = os.stat(f)
    assert (st_after.st_ino, st_after.st_ctime_ns, st_after.st_mtime_ns) == (st_before.st_ino, st_before.st_ctime_ns, st_before.st_mtime_ns)   # not rewritten
    assert open(f).read() == '{"dose": 0.5, "effort": [1, 2, 3]}' and CR._verify_home_tree(spec2) == []
    rec = json.load(open(marker))
    assert rec["window"] == "w3" and [p for p, _ in rec["spec"]["files"]] == [f, stats]
    # the next build manages F as any recorded copy: replaced and copied, nothing adopted, nothing refused
    w4 = CR.init_home_tree(spec2, home, "w4")
    assert w4["refused"] == [] and w4["adopted"] == [] and w4["copied"] == [f, stats] and w4["not_attempted"] == [], w4
    assert open(f).read() == '{"dose": 0.5, "effort": [1, 2, 3]}' and json.load(open(marker))["window"] == "w4" and CR._verify_home_tree(spec2) == []
    # a byte-identical unrecorded copy under ANOTHER mtime (a plain copy, no stat preserved) is adopted AND re-copied, so its mtime is the
    # generation's: the pass's watermark compares mtime_utc, and a drifted one would read as an input the grading host never saw
    gen3, comps3, pins3 = _components(tmp_path / "g3", home, layout(["attacker_effort.json", "late_member.json"]))
    src_g = f"{gen3}/multiturn/{pins3['multiturn']}/late_member.json"
    with open(src_g, "w") as fh:
        fh.write('{"late": true}')
    spec3 = CR.home_tree_spec(comps3, pins3, gen3, home, _entries_of(gen3))
    g = os.path.join(full22, "late_member.json")
    shutil.copyfile(src_g, g)
    os.utime(g, ns=(os.stat(src_g).st_atime_ns, os.stat(src_g).st_mtime_ns + 7_000_000_000))
    assert os.stat(g).st_mtime_ns != os.stat(src_g).st_mtime_ns
    w5 = CR.init_home_tree(spec3, home, "w5")
    assert w5["refused"] == [] and w5["adopted"] == [g] and w5["copied"] == [f, g, stats] and w5["not_attempted"] == [], w5
    assert os.stat(g).st_mtime_ns == os.stat(src_g).st_mtime_ns and open(g).read() == '{"late": true}' and CR._verify_home_tree(spec3) == []
    assert [p for p, _ in json.load(open(marker))["spec"]["files"]] == [f, g, stats]


def test_an_unrecorded_copy_path_holding_other_bytes_a_symlink_or_a_directory_keeps_the_refusal_untouched(tmp_path):
    """The negatives of adoption, with the refusal the live ticks printed pinned byte for byte: an unrecorded regular file of the SAME size
    and other bytes (the compare is of bytes, never of sizes), an unrecorded symlink -- even one that resolves onto the generation's own file
    -- and an unrecorded directory are refused as today and left exactly as found; nothing adopted, nothing copied, no marker."""
    gen = tmp_path / "gen"
    gen.mkdir()
    src = gen / "stats.json"
    src.write_text('{"a": 1}')
    home = tmp_path / "home"
    dst = str(home / "shared" / "stats.json")
    spec = {"home": str(home), "dirs": [str(home / "shared")], "links": [], "files": [[dst, str(src)]]}
    marker = home / CR.NAMESPACE_MARKER
    live = "exists and is not a copy or link a previous window recorded in this node's namespace; refusing to replace a path this machinery does not manage"
    os.makedirs(home / "shared")
    # (a) same size, other bytes
    with open(dst, "w") as fh:
        fh.write('{"a": 2}')
    os.chmod(dst, 0o600)
    st = os.stat(dst)
    r = CR.init_home_tree(spec, str(home), "w1")
    assert r["refused"] == [{"path": dst, "why": live}] and r["adopted"] == [] and r["copied"] == [] and r["not_attempted"] == [], r
    assert open(dst).read() == '{"a": 2}' and os.stat(dst).st_ino == st.st_ino and os.stat(dst).st_ctime_ns == st.st_ctime_ns and not marker.exists()
    # (b) a symlink onto the generation's file itself: the bytes match through it, and it is still not a copy
    os.remove(dst)
    os.symlink(str(src), dst)
    r = CR.init_home_tree(spec, str(home), "w1")
    assert r["refused"] == [{"path": dst, "why": live}] and r["adopted"] == [] and r["copied"] == [], r
    assert os.path.islink(dst) and os.readlink(dst) == str(src) and not marker.exists()
    # (c) a directory: the directory refusal, as before
    os.remove(dst)
    os.makedirs(dst)
    r = CR.init_home_tree(spec, str(home), "w1")
    assert r["refused"] == [{"path": dst, "why": "is a directory; refusing to replace a real path with a file copy"}] and r["adopted"] == [], r
    assert os.path.isdir(dst) and not marker.exists()
    # the shard prints a copy refusal as `home tree: <path>: <why>`, the string the live ticks carried (the link case is pinned above)
    assert "home tree: " + "; ".join(f"{x['path']}: {x['why']}" for x in [{"path": dst, "why": live}]) == f"home tree: {dst}: {live}"


def test_once_a_build_has_refused_nothing_is_written_whatever_the_order_and_the_next_clean_build_makes_everything(tmp_path, monkeypatch):
    """The links half of the rule, and its reach: a refusal found at the FIRST link in order or at the LAST withholds every write alike --
    no directory made, no link created, no copy written (all listed under `not_attempted`), the tree byte for byte as found, no marker --
    because the preflight evaluates every pass before anything is written. The next build, the obstruction gone, makes everything."""
    home = str(tmp_path / "home")
    monkeypatch.setattr(ST, "HOME", home)
    marker = os.path.join(home, CR.NAMESPACE_MARKER)
    a, b, shared = os.path.join(home, "a_base"), os.path.join(home, "b_base"), os.path.join(home, "shared")
    gen, comps, pins = _components(tmp_path / "g", home, {"first": ("a_base", ["x/"]), "second": ("b_base", ["y/"]),
                                                         "p": ("shared", ["p.json"]), "q": ("shared", ["q.json"])})
    spec = CR.home_tree_spec(comps, pins, gen, home, _entries_of(gen))
    copies = [os.path.join(shared, "p.json"), os.path.join(shared, "q.json")]
    assert [p for p, _ in spec["links"]] == [a, b] and [p for p, _ in spec["files"]] == copies and spec["dirs"] == [shared]
    # a real directory with a stray file where `a_base` links: the first link in order
    os.makedirs(a)
    with open(os.path.join(a, "notes.txt"), "w") as fh:
        fh.write("mine")
    before = _tree_listing(home)
    w1 = CR.init_home_tree(spec, home, "w1")
    assert [r["path"] for r in w1["refused"]] == [a] and w1["created"] == [] and w1["copied"] == [] and w1["removed"] == [], w1
    assert w1["not_attempted"] == [shared, b] + copies and _tree_listing(home) == before and not os.path.lexists(shared) and not os.path.exists(marker)
    # the obstruction at `b_base` instead, the LAST link in order: `a_base` is not created either
    shutil.rmtree(a)
    os.makedirs(b)
    with open(os.path.join(b, "notes.txt"), "w") as fh:
        fh.write("mine")
    before = _tree_listing(home)
    w2 = CR.init_home_tree(spec, home, "w2")
    assert [r["path"] for r in w2["refused"]] == [b] and w2["created"] == [] and w2["copied"] == [] and w2["removed"] == [], w2
    assert w2["not_attempted"] == [shared, a] + copies and _tree_listing(home) == before and not os.path.lexists(a) and not os.path.exists(marker)
    # the obstruction gone: everything is made and the marker written
    shutil.rmtree(b)
    w3 = CR.init_home_tree(spec, home, "w3")
    assert w3["refused"] == [] and w3["created"] == [a, b] and w3["unchanged"] == [] and w3["copied"] == copies and w3["not_attempted"] == [], w3
    assert CR._verify_home_tree(spec) == [] and json.load(open(marker))["window"] == "w3"


def test_a_whole_link_a_refused_build_would_have_remade_as_a_real_directory_stands_and_the_next_window_links_it_whole(tmp_path, monkeypatch):
    """The inverse of the promoted base: the first window selects one component at base B, so B is a recorded WHOLE LINK; the second
    window selects two components under B (B must be a real directory, laid out entry by entry) and meets an obstruction at an
    unrelated link C. The builder before this change removed B's link and made B a real directory in its dirs pass, THEN refused at C,
    and the first window's marker stood -- so the third window, selecting the one component again and wanting B whole, met a real
    directory no marker's `dirs` recorded and refused for good (an unrecorded directory until someone cleaned the node by hand). With
    the preflight, the second window changes nothing: B stays the first window's link, the stray file at C is never touched, and the
    third window finds B unchanged and records its namespace."""
    home = str(tmp_path / "home")
    monkeypatch.setattr(ST, "HOME", home)
    marker = os.path.join(home, CR.NAMESPACE_MARKER)
    full22, sweep = os.path.join(home, "rs/full22"), os.path.join(home, "specs/sweep")
    gen, comps, pins = _components(tmp_path / "g", home, {"multiturn": ("rs/full22", ["grades_a/", "transfer_stats.json"]),
                                                         "misalign": ("rs/full22", ["misalign/"]), "sidecars": ("specs/sweep", ["mm.json"])})
    spec_of = lambda *slugs: CR.home_tree_spec({s: comps[s] for s in slugs}, {s: pins[s] for s in slugs}, gen, home, _entries_of(gen))
    one, two = spec_of("multiturn"), spec_of("multiturn", "misalign", "sidecars")
    target = f"{gen}/multiturn/{pins['multiturn']}"
    assert dict(map(tuple, one["links"])) == {full22: target} and full22 not in one["dirs"] and full22 in two["dirs"]
    # W1: B whole
    w1 = CR.init_home_tree(one, home, "w1")
    assert w1["refused"] == [] and w1["created"] == [full22] and os.path.islink(full22) and full22 not in json.load(open(marker))["spec"]["dirs"], w1
    # W2: B real with children, C obstructed by a real directory holding the user's file
    os.makedirs(sweep)
    notes = os.path.join(sweep, "notes.txt")
    with open(notes, "w") as fh:
        fh.write("mine")
    before = _tree_listing(home)
    w2 = CR.init_home_tree(two, home, "w2")
    assert [r["path"] for r in w2["refused"]] == [sweep] and "exists and is not a symlink" in w2["refused"][0]["why"], w2
    assert w2["removed"] == [] and w2["created"] == [] and w2["copied"] == [] and _tree_listing(home) == before
    assert w2["not_attempted"] == [full22, os.path.join(full22, "grades_a"), os.path.join(full22, "misalign"), os.path.join(full22, "transfer_stats.json")]
    assert os.path.islink(full22) and os.readlink(full22) == target and json.load(open(marker))["window"] == "w1" and open(notes).read() == "mine"
    # W3: the one component again -- B whole, as W1 left it; the user's directory at C is not this window's and stands
    w3 = CR.init_home_tree(one, home, "w3")
    assert w3["refused"] == [] and w3["unchanged"] == [full22] and w3["removed"] == [] and w3["created"] == [] and w3["not_attempted"] == [], w3
    assert os.path.islink(full22) and json.load(open(marker))["window"] == "w3" and open(notes).read() == "mine" and CR._verify_home_tree(one) == []
    # and the obstruction gone, W4 lays B out entry by entry over its own whole link: removed, remade real, the children made, the copy copied
    shutil.rmtree(sweep)
    w4 = CR.init_home_tree(two, home, "w4")
    assert w4["refused"] == [] and w4["removed"] == [full22] and os.path.isdir(full22) and not os.path.islink(full22), w4
    assert sorted(w4["created"]) == sorted([os.path.join(full22, "grades_a"), os.path.join(full22, "misalign"), sweep])
    assert w4["copied"] == [os.path.join(full22, "transfer_stats.json")] and CR._verify_home_tree(two) == [] and json.load(open(marker))["window"] == "w4"


def test_a_link_into_a_directory_the_generation_excludes_is_refused_by_the_plan_from_the_manifest_alone_and_a_listed_target_is_read(tmp_path, monkeypatch):
    """The manifest attests a directory link WITHOUT walking through it, so a link into a directory the
    component excludes is in the generation while the files beneath it are not -- on the node the link dangles, the group the grading host read
    through it is simply not discovered (no head for the dropped-input rule to compare against) and it would vanish from the cluster read
    without a word. The plan-time discovery used to catch it; the plan now discovers nothing, so the ONE coverage
    fact the MANIFEST COPY alone can tell is checked instead (`dangling_links`, no store read): the adapter is recorded refused by name with
    the link and its target. The rule's pure functions (`_staged_member` / `input_coverage`) stay pinned. A link re-pointed onto a listed
    directory covers its descendants: the cluster route, and the shard reads through it."""
    root, home, stores, ad = _grading_host(tmp_path)
    monkeypatch.setattr(ST, "HOME", home)
    _store(os.path.join(stores, "hidden"), "h1", "legacy_store_9", "none", [0.9])          # stores/hidden/h1/store.json, excluded below
    os.symlink("hidden/h1", os.path.join(stores, "s9"))                                    # a directory link INTO the excluded directory
    cfgp = TS._cfg(tmp_path, {"synthetic_stores": {"base": stores, "authority": "grading_host_copy", "origin": "grading_host", "exclude": ["/hidden"]}},
                   adapters={ADAPTER: {"shards": 2, "components": ["synthetic_stores"]}})
    cfg = ST.load_config(cfgp)
    dest = cfg.destination("cluster_a")
    res = ST.mirror_component(root, dest, cfg.components["synthetic_stores"], runner=TS.Runner())
    assert res.outcome == "published", res.as_dict()
    files, links = CR.generation_members(root, "synthetic_stores", res.content_sha8)
    assert links == {"s9": "hidden/h1"} and "s1/store.json" in files and not any(f.startswith("hidden/") for f in files)
    # the manifest-only check names the link; the rule's pure functions over the grading host's discovered inputs agree
    assert CR.dangling_links(root, "synthetic_stores", res.content_sha8) == {"s9": "hidden/h1"}
    s9 = os.path.join(stores, "s9", "store.json")
    inputs = ad.discover()
    assert any(i.uri == s9 for i in inputs)
    cov = CR.input_coverage(cfg, root, {"synthetic_stores": res.content_sha8}, inputs, REPO_ROOT, home, ad.roots)
    why = f"under component synthetic_stores's base but not a member of its generation {res.content_sha8}"
    assert cov == {"discovered": 4, "covered": 3, "uncovered": 1, "roots": {stores: {"count": 1, "why": why, "sample": [s9]}}}
    assert CR._staged_member("s1/store.json", files, links) is True and CR._staged_member("s9/store.json", files, links) is False
    assert CR._staged_member("alias/store.json", files, {"alias": "s1"}) is True
    assert CR._staged_member("alias/store.json", files, {"alias": "../s1"}) is False and CR._staged_member("alias/store.json", files, {"alias": "/s1"}) is False
    assert CR._staged_member("a/store.json", files, {"a": "b", "b": "s1"}) is True                   # two hops
    assert CR._staged_member("a/store.json", files, {"a": "b", "b": "a"}) is False                   # a cycle ends
    # the plan: refused by name from the manifest copy, no discovery (a load_all that would construct anything proves it)
    monkeypatch.setattr("sisteer.registry.adapters.load_all", lambda: pytest.fail("the plan discovers nothing (C6)"))
    with AB.without_store_access_scope():
        plan = CR.plan_window(root, "wl", [ADAPTER], cfg, dest, "a" * 40, uri_home=home, repo_root=REPO_ROOT)
    e = plan["adapters"][ADAPTER]
    assert e["revalidation_pre"]["synthetic_stores"]["ok"] is True                        # admission passes: the copy IS the generation
    assert e["route"] == "grading_host" and plan["tasks"] == [] and plan["home_tree"] is None and "inputs" not in e
    assert e["reason"].startswith(f"{CR.UNCOVERED_PREFIX}: the generation attests 1 directory link(s) whose target holds no member") and "synthetic_stores:s9 -> hidden/h1" in e["reason"]
    assert e["dangling_links"] == {"synthetic_stores:s9": "hidden/h1"} and plan["refusals"] == {ADAPTER: [f"plan: {e['reason']}"]}
    # the link re-pointed onto a listed directory: no dangling link, the cluster route, and the shard reads through it on the node
    os.remove(os.path.join(stores, "s9"))
    os.symlink("s1", os.path.join(stores, "s9"))
    res2 = ST.mirror_component(root, dest, cfg.components["synthetic_stores"], runner=TS.Runner())
    assert res2.outcome == "published", res2.as_dict()
    assert CR.dangling_links(root, "synthetic_stores", res2.content_sha8) == {}
    monkeypatch.setattr("sisteer.registry.adapters.load_all", lambda: {ADAPTER: lambda: SyntheticStoreAdapter(stores)})
    w = {"root": root, "home": home, "stores": stores, "ad": ad, "cfg": cfg, "cfgp": cfgp, "dest": dest, "sha8": res2.content_sha8,
         "gen_dir": f"{dest.stores_dir}/gen", "shared": dest.shared}
    plan2, _plan_file, _info = _plan_and_stage(w, tmp_path, window="wm")
    assert plan2["adapters"][ADAPTER]["route"] == "cluster" and len(plan2["tasks"]) == 2 and "dangling_links" not in plan2["adapters"][ADAPTER]
    _node(w)
    assert CR.run_shard(f"{plan2['window_root']}/plan.json", 0, home=home) == 0 and CR.run_shard(f"{plan2['window_root']}/plan.json", 1, home=home) == 0
    docs = [json.load(open(CR.summary_path(plan2["summaries_dir"], ADAPTER, i, 2))) for i in range(2)]
    assert all(d["ok"] is True and d["refused"] == [] for d in docs)
    # s9 (through the link onto s1's directory) is a NEW group on the node: exactly one shard published it, through the generation
    pub = [rev for d in docs for rev, info in d["published"].items() if info["group"] == "s9"]
    assert len(pub) == 1, [d["published"] for d in docs]


# --------------------------------------------------------------------------------------------
# part 7: a `cluster: refused` entry in the plan; the regrade twin's discovery through its staged generation
# --------------------------------------------------------------------------------------------


def test_a_declared_refused_adapter_routes_to_the_grading_host_before_any_admission_or_discovery_and_a_task_less_cluster_route_is_never_a_read(tmp_path, monkeypatch, capsys):
    """An adapter entry `cluster: refused` + reason loads as zero shards and no component;
    before the fix `plan_window` never looked at `refused`, so with the coverage check declined (or an adapter discovering nothing) the
    entry came out `route=cluster` with NO task and no fallback: `cluster-reads submit` printed `none`, the window read it as success,
    and the adapter was marked read without a read or a recorded refusal. Now: the refused declaration is recognized FIRST -- route
    grading_host, `refused: true`, the declaration's reason behind REFUSED_DECL_PREFIX, nothing pinned, revalidated or discovered (the
    plan discovers nothing at all) -- and it is in the refusal set, the CLI's plan line and the status row. A window of
    refused adapters only has no task: `submit` prints `none` and the refusal set names them. And the pre-fix shape as DATA (a plan FILE
    routing a zero-shard adapter to the cluster with no task) is never a read: `submit` is a FAILED submission naming it (window.sh's
    cluster_fail then refuses every cluster adapter), `refused_adapters` names it, the pull names it."""
    import argparse
    import sisteer.registry.cli as CLI
    w = _world_with_generation(tmp_path, monkeypatch)
    reason = "its pinned paper worktree is absent (D13)"
    cfgp = TS._cfg(tmp_path, {"synthetic_stores": {"base": w["stores"], "authority": "grading_host_copy", "origin": "grading_host"}},
                   adapters={ADAPTER: {"shards": 2, "components": ["synthetic_stores"]}, "paper_pins": {"cluster": "refused", "reason": reason}},
                   name="refused.yaml")
    cfg = ST.load_config(cfgp)
    dest = cfg.destination("cluster_a")
    assert cfg.adapters["paper_pins"].refused and cfg.adapters["paper_pins"].shards == 0 and cfg.adapters["paper_pins"].components == []

    monkeypatch.setattr("sisteer.registry.adapters.load_all", lambda: {ADAPTER: lambda: NeverDiscoverAdapter(w["stores"])})   # constructed, never discovered (C6)
    plan = CR.plan_window(w["root"], "wq", [ADAPTER, "paper_pins"], cfg, dest, "a" * 40, uri_home=w["home"], repo_root=REPO_ROOT)
    e = plan["adapters"]["paper_pins"]
    assert e["route"] == "grading_host" and e["refused"] is True and e["reason"] == CR.REFUSED_DECL_PREFIX + reason
    assert e["shards"] == 0 and e["pins"] == {} and e["revalidation_pre"] == {} and "inputs" not in e and e["task_indices"] == []
    a = plan["adapters"][ADAPTER]
    assert a["route"] == "cluster" and a["refused"] is False and [t["adapter"] for t in plan["tasks"]] == [ADAPTER, ADAPTER] and a["task_indices"] == [0, 1]
    assert plan["refusals"] == {"paper_pins": [f"plan: {e['reason']}"]} and CR.refused_adapters(plan) == plan["refusals"]
    assert CR.cluster_routes_without_tasks(plan) == []
    # the CLI's plan line (the one window.sh tails into the chain log) and the status page's row name the declared refusal
    args = argparse.Namespace(action="plan", window="wq2", adapter=[ADAPTER, "paper_pins"], tree_sha="a" * 40, dest="cluster_a", config=cfgp,
                              repo_root=REPO_ROOT, full=False, full_adapter=[], reread_force_adapter=[], json=False)
    assert CR.cmd_cluster_reads(args, w["root"]) == 0
    out = capsys.readouterr().out
    assert f"{ADAPTER}=cluster, paper_pins=grading_host; 2 task(s)" in out and f"\n  paper_pins -> REFUSED: plan: {CR.REFUSED_DECL_PREFIX}{reason}" in out, out
    row = next(r for r in CR.status_section(w["root"])["rows"] if r["adapter"] == "paper_pins")
    assert row["refused"] is True and row["route"] == "grading_host" and row["reason"] == CR.REFUSED_DECL_PREFIX + reason and row["shards"] == 0
    # a window naming the refused adapter alone: no task, no home tree; `submit` prints `none` (rc 0, nothing submitted) and the fallback names it
    plan_r = CR.plan_window(w["root"], "wr0", ["paper_pins"], cfg, dest, "a" * 40, uri_home=w["home"], repo_root=REPO_ROOT)
    assert plan_r["tasks"] == [] and plan_r["home_tree"] is None and CR.refused_adapters(plan_r) == {"paper_pins": [f"plan: {CR.REFUSED_DECL_PREFIX}{reason}"]}
    CR._write_json(CR.plan_path(w["root"], "wr0"), plan_r)
    monkeypatch.setattr(CR, "submit_array", lambda *a_, **k: pytest.fail("a plan with no task submits nothing"))
    monkeypatch.setattr(CR, "submit_attest", lambda *a_, **k: pytest.fail("a plan with no task attests nothing"))
    assert CLI.main(["--root", w["root"], "cluster-reads", "submit", "--window", "wr0", "--venv", "/shared/envs/v"]) == 0
    assert capsys.readouterr().out.strip().splitlines()[-1] == "none"
    assert CR._load_json(CR.plan_path(w["root"], "wr0"))["stages"]["submit"]["tasks"] == 0
    # the pre-fix shape as data: route cluster, zero shards, no task, no fallback
    bad = dict(plan_r, adapters={"paper_pins": dict(plan_r["adapters"]["paper_pins"], route="cluster", refused=False, reason=None)}, refusals={}, stages={})
    assert CR.cluster_routes_without_tasks(bad) == ["paper_pins"] and CR.refused_adapters(bad) == {"paper_pins": [f"plan: {CR.NO_TASK_REASON}"]}
    CR._write_json(CR.plan_path(w["root"], "wbad"), bad)
    assert CLI.main(["--root", w["root"], "cluster-reads", "submit", "--window", "wbad", "--venv", "/shared/envs/v"]) == 2
    err = capsys.readouterr().err
    assert "submit: the plan has no task but routes paper_pins to the cluster" in err and "nothing would read" in err, err
    assert "submit" not in CR._load_json(CR.plan_path(w["root"], "wbad"))["stages"]            # a failed submission records no stage
    # the pull over such a plan (the post-run revalidation recorded, an empty spool): the adapter is the grading host's, by this reason, with no shard expected
    src = tmp_path / "empty_window_root"
    (src / "root" / "records" / "adapters").mkdir(parents=True)
    bad["stages"] = {"revalidate_post": {"stale": []}}
    rep = CR.pull_window(w["root"], bad, str(tmp_path / "expect_bad"), source=str(src))
    assert CR.NO_TASK_REASON in rep["adapters_refused"]["paper_pins"] and rep["summaries"]["paper_pins"]["expected"] == 0
    # an empty window root carries no attestation record either, and the pull's gate is fail-closed on that too
    assert any(r.startswith("no readable attestation record for this operation") for r in rep["adapters_refused"]["paper_pins"])
    assert rep["pulled"] == {} and rep["quarantined"] == {}


def _live_subset_cfg(tmp_path, adapter: str, name: str) -> str:
    """A cluster_reads.yaml carrying the LIVE declarations of ``adapter``'s entry and of every component it names (bases spelled `~`,
    expanded under the test's HOME) on TS._cfg's local destination, so a synthetic home laid out at those spellings is mirrored, pinned,
    planned and staged by the real machinery under the real declarations."""
    import yaml
    doc = yaml.safe_load(open(os.path.join(REPO_ROOT, "configs", "registry", "cluster_reads.yaml"), encoding="utf-8"))
    entry = doc["adapters"][adapter]
    return TS._cfg(tmp_path, {slug: doc["components"][slug] for slug in entry["components"]}, adapters={adapter: entry}, name=name)


def test_the_live_grade_regrade_declaration_stages_what_the_twins_real_discovery_opens_and_discovery_runs_through_the_staged_generation(tmp_path, monkeypatch):
    """`RegradeTwinAdapter.discover()` requires `<root>/README.md` and the first declaration of
    `grade_regrade` (members manifests/ + journal/) did not hold it, so every shard of a cluster-routed regrade would have refused while
    `stores coverage` (declared roots only) passed. Here the LIVE declarations of regrade_twin's seven components run for real over a
    synthetic home: the fixture's twin root (README.md, ledger.jsonl, S_A.parquet, units_A.json, the harvest journal, a Swap-A sibling
    beside its Haiku original under `sampled`) completed with every other MANIFEST_FILES entry, a second sibling in a directory only
    `full22_regrade_siblings` holds, and decoys the twin never opens (an escaping link farm, the claims database, a parts directory, a
    pinned copy, prose); one member each for the other four holders. `stores.mirror_component` publishes each (the real walk, manifest and
    verify); the generation of grade_regrade holds exactly the eleven files the twin opens and no decoy; the plan runs the adapter's REAL
    discovery and the coverage check against the pinned generations -> every input covered, the cluster route, K = 8, the twin root linked
    whole; and on the node (the grading host's trees moved aside, the window's home tree laid) the same discovery through the staged
    generations yields the same inputs, byte for byte, with no decoy in sight."""
    from sisteer.registry.adapters import regrade_twin as RT
    home = str(tmp_path / "home")
    os.makedirs(home)
    monkeypatch.setenv("HOME", home)                              # load_config expands `~` from the environment
    monkeypatch.setattr(ST, "HOME", home)                         # Component.home_rel against the module's HOME
    root = _cluster_root(tmp_path, "grading_host_root")
    cfg = ST.load_config(_live_subset_cfg(tmp_path, "regrade_twin", "live-regrade.yaml"))
    comps = cfg.components
    assert set(comps) == {"grade_regrade", "full22_regrade_siblings", "full22_relu_native", "grade_alphasteer", "grade_arditi", "multiturn_run_state", "sampled"}
    full22 = comps["sampled"].base[: -len("/sampled")]
    assert comps["full22_regrade_siblings"].base == full22 == comps["multiturn_run_state"].base == comps["full22_relu_native"].base
    twin = comps["grade_regrade"].base
    assert twin == str(sisteer.site.expand_path("$GRADE/regrade")) and comps["grade_alphasteer"].base == os.path.dirname(twin)
    fx = F.write_twin_root(twin, comps["sampled"].base, n_goals=2, n_orders=1)
    assert fx["grades_dir"].startswith(comps["sampled"].base + "/")
    for rel, _role in RT.MANIFEST_FILES:
        p = os.path.join(twin, rel)
        if not os.path.exists(p):
            open(p, "w").write("{}\n")
    sib2_dir = os.path.join(full22, "grades", "qwen35_4b")        # one of the three directories only full22_regrade_siblings holds
    os.makedirs(sib2_dir)
    sib2, orig2 = os.path.join(sib2_dir, RT.SWAP_A_SIBLING), os.path.join(sib2_dir, RT.SWAP_A_ORIGINAL)
    open(sib2, "w").write(json.dumps({"key": "k2", "value": {"refused": 0, "convincing": 2, "specific": 3, "judge_model": "claude-sonnet-5"}}) + "\n")
    open(orig2, "w").write(json.dumps({"key": "k2", "value": {"refused": 0, "convincing": 5, "specific": 5, "judge_model": "claude-haiku-4-5"}}) + "\n")
    with open(os.path.join(twin, "journal", "harvest", "livebatch_synthetic_1.jsonl"), "a") as fh:
        fh.write(json.dumps({"custom_id": "k2", "swap": "A", "model": "claude-sonnet-5", "transport": "live", "pool": "lowprio", "outcome": "verdict",
                             "destinations": [sib2], "t": "2026-01-04T00:10:00.000000Z"}) + "\n")
    decoys = ["swapB/judge_cache.sqlite", "journal/claims_A.sqlite", "manifests/_parts_A/part0.parquet", "manifests/S_A.pin-deadbeef.parquet",
              "decisions.md", "canonical_frames/x.json"]
    for rel in decoys:
        os.makedirs(os.path.dirname(os.path.join(twin, rel)), exist_ok=True)
        open(os.path.join(twin, rel), "w").write("decoy\n")
    os.makedirs(os.path.join(twin, "swapA", "farm"))
    os.symlink("/etc/hostname", os.path.join(twin, "swapA", "farm", "escape"))      # the real root's 22,233 escaping links, in one
    for slug, member in (("full22_relu_native", "relu_native"), ("multiturn_run_state", "grades_synth"),
                         ("grade_alphasteer", "alphasteer_f22_rebake"), ("grade_arditi", "arditi_strength")):
        os.makedirs(os.path.join(comps[slug].base, member), exist_ok=True)
        open(os.path.join(comps[slug].base, member, "x.json"), "w").write("{}\n")
    dest = cfg.destination("cluster_a")
    pins = {}
    for slug in sorted(comps):
        res = ST.mirror_component(root, dest, comps[slug], runner=TS.Runner())
        assert res.outcome == "published", (slug, res.as_dict())
        pins[slug] = res.content_sha8
    gen_dir = f"{dest.stores_dir}/gen"
    files, links = CR.generation_members(root, "grade_regrade", pins["grade_regrade"])
    assert files == {rel for rel, _r in RT.MANIFEST_FILES} | {"journal/harvest/livebatch_synthetic_1.jsonl"} and links == {}
    assert "README.md" in files and "ledger.jsonl" in files

    def twin_adapter():
        return RT.RegradeTwinAdapter(root=twin, dims=F.dims_map(), campaign="synthcampaign", sampled_root=comps["sampled"].base)
    plan = CR.plan_window(root, "wtwin", ["regrade_twin"], cfg, dest, "a" * 40, uri_home=home, repo_root=REPO_ROOT)
    e = plan["adapters"]["regrade_twin"]
    n = len(RT.MANIFEST_FILES) + 1 + 2 + 2                      # the ten twin-root files, the harvest file, two siblings, two Haiku originals
    assert e["route"] == "cluster", e["reason"]
    assert "inputs" not in e and e["pins"] == pins and len(plan["tasks"]) == 8          # C6: the plan discovers nothing; coverage is the shard's
    assert all(e["revalidation_pre"][slug]["ok"] is True for slug in comps) and plan["refusals"] == {} and e["refused"] is False
    links_of = dict(map(tuple, plan["home_tree"]["links"]))
    assert links_of[twin] == f"{gen_dir}/grade_regrade/{pins['grade_regrade']}"                                   # the twin root, whole
    assert links_of[os.path.join(full22, "grades")] == f"{gen_dir}/full22_regrade_siblings/{pins['full22_regrade_siblings']}/grades"
    assert links_of[comps["sampled"].base] == f"{gen_dir}/sampled/{pins['sampled']}" and full22 in plan["home_tree"]["dirs"]
    with AB.using_hash_cache(None):
        grading_host = sorted((i.uri, i.sha256) for i in twin_adapter().discover())
    assert len(grading_host) == n and (fx["sibling"], R.file_sha256(fx["sibling"])) in grading_host and (sib2, R.file_sha256(sib2)) in grading_host
    assert (os.path.join(twin, "README.md"), R.file_sha256(os.path.join(twin, "README.md"))) in grading_host
    # the node
    for p in (os.path.dirname(twin), full22):
        shutil.move(p, p + ".grading_host")
    out = CR.init_home_tree(plan["home_tree"], home, "wtwin")
    assert out["mode"] == "built" and not out["refused"], out
    assert os.path.islink(twin) and os.readlink(twin) == links_of[twin] and os.path.isfile(os.path.join(twin, "README.md"))
    for rel in decoys + ["swapA"]:
        assert not os.path.lexists(os.path.join(twin, rel)), rel                              # the generation holds the members alone
    with AB.using_hash_cache(None):
        node = sorted((i.uri, i.sha256) for i in twin_adapter().discover())
    assert node == grading_host


# --------------------------------------------------------------------------------------------
# the name escape in the home tree (the cluster's shared storage refuses ':')
# --------------------------------------------------------------------------------------------


def test_home_tree_spec_lays_out_escaped_entries_through_real_directories_with_node_local_links(tmp_path, monkeypatch):
    """An entry whose canonical name the destination refuses lives in the generation under its escaped name; the home tree opens every
    directory above it as a REAL directory laid out entry by entry and links the canonical spelling onto the escaped path, so an adapter
    opening `~/grade/misalign/em8q/em-x::y/g.json` reads the generation's bytes; the shard's hash cache maps the real (escaped) path back
    to the manifest's true name."""
    home = str(tmp_path / "home")
    os.makedirs(home)
    monkeypatch.setattr(ST, "HOME", home)
    base = os.path.join(home, "grade")
    for rel, data in (("misalign/em8q/em-x::y/g.json", b'{"g": 1}'), ("misalign/em8q/em-x::y/deep/h:1.json", b'{"h": 1}'), ("misalign/em8q/plain/p.json", b"{}"),
                      ("misalign/em8q/note.txt", b"note"), ("misalign/other/o.json", b"{}"), ("misalign_16k/k.json", b"{}"), ("misalign/em8q/w:z/q.json", b"q")):
        TS._write(os.path.join(base, rel), data)
    os.symlink("plain/p.json", os.path.join(base, "misalign/em8q/lnk:p"))
    os.symlink("w:z/q.json", os.path.join(base, "misalign/em8q/to:q"))                  # an escaped name AND an escaped target
    os.symlink("em-x::y/g.json", os.path.join(base, "misalign/em8q/plain_to_escaped"))  # a plain name, an escaped target
    root = str(tmp_path / "reg")
    os.makedirs(os.path.join(root, "state"))
    cfgp = TS._cfg(tmp_path, {"grades": {"base": base, "authority": "grading_host_copy"}, "sidecar": {"base": os.path.join(home, "specs/side"), "authority": "grading_host_copy"}},
                   adapters={"em": {"shards": 1, "components": ["grades", "sidecar"]}})
    TS._write(os.path.join(home, "specs/side/s.json"), b"{}")
    cfg = ST.load_config(cfgp)
    dest = cfg.destination("cluster_a")
    pins = {}
    for slug in ("grades", "sidecar"):
        r = ST.mirror_component(root, dest, cfg.components[slug], runner=TS.Runner())
        assert r.outcome == "published", r.as_dict()
        pins[slug] = r.content_sha8
    gen_dir = f"{dest.stores_dir}/gen"
    gen = f"{gen_dir}/grades/{pins['grades']}"
    assert os.path.isfile(f"{gen}/misalign/em8q/em-x%3A%3Ay/deep/h%3A1.json")
    esc = CR.generation_escapes(root, "grades", pins["grades"])
    assert set(esc) == {"misalign/em8q/em-x::y/g.json", "misalign/em8q/em-x::y/deep/h:1.json", "misalign/em8q/w:z/q.json", "misalign/em8q/lnk:p",
                        "misalign/em8q/to:q"}
    assert CR.generation_escapes(root, "sidecar", pins["sidecar"]) == {}
    assert CR._opened_dirs(esc) == {"misalign", "misalign/em8q", "misalign/em8q/em-x::y", "misalign/em8q/em-x::y/deep"}, \
        "em-x::y and deep are opened too: an escaped name lies beneath each"
    assert CR.manifest_entries(root, "grades", pins["grades"], "misalign/em8q") == [("em-x::y", True), ("lnk:p", True), ("note.txt", False), ("plain", True),
                                                                                    ("plain_to_escaped", True), ("to:q", True), ("w:z", True)]
    # the two links whose TARGETS the destination spells differently are written with the escaped target, so they resolve inside the generation
    assert os.readlink(f"{gen}/misalign/em8q/to%3Aq") == "w%3Az/q.json" and open(f"{gen}/misalign/em8q/to%3Aq", "rb").read() == b"q"
    assert os.readlink(f"{gen}/misalign/em8q/plain_to_escaped") == "em-x%3A%3Ay/g.json" and open(f"{gen}/misalign/em8q/plain_to_escaped", "rb").read() == b'{"g": 1}'
    grades_meta = json.load(open(f"{gen_dir}/grades/MANIFEST-{pins['grades']}.json"))
    assert grades_meta["symlinks"]["misalign/em8q/plain_to_escaped"] == "em-x::y/g.json" and grades_meta["symlinks"]["misalign/em8q/to:q"] == "w:z/q.json", \
        "the manifest keeps the true targets"
    assert grades_meta["target_escapes"] == {"misalign/em8q/plain_to_escaped": "em-x%3A%3Ay/g.json", "misalign/em8q/to:q": "w%3Az/q.json"}
    entries_of = lambda s, g: CR.manifest_entries(root, s, g)                    # noqa: E731
    sub_of = lambda s, g, d: CR.manifest_entries(root, s, g, d)                  # noqa: E731
    esc_of = lambda s, g: CR.generation_escapes(root, s, g)                      # noqa: E731
    spec = CR.home_tree_spec(cfg.components, pins, gen_dir, home, entries_of, subentries_of=sub_of, escapes_of=esc_of)
    links, files = dict(map(tuple, spec["links"])), dict(map(tuple, spec["files"]))
    em8q = os.path.join(base, "misalign/em8q")
    assert base in spec["dirs"] and os.path.join(base, "misalign") in spec["dirs"] and em8q in spec["dirs"] and os.path.join(em8q, "em-x::y") in spec["dirs"]
    assert os.path.join(em8q, "em-x::y/deep") in spec["dirs"], "deep holds an escaped file: a real directory too"
    assert base not in links, "a base with an escaped name beneath it is never linked whole"
    assert files[os.path.join(em8q, "em-x::y/deep/h:1.json")] == f"{gen}/misalign/em8q/em-x%3A%3Ay/deep/h%3A1.json", "an escaped file entry of an opened directory is a copy, from the escaped path"
    assert files[os.path.join(em8q, "em-x::y/g.json")] == f"{gen}/misalign/em8q/em-x%3A%3Ay/g.json", "a file entry of an opened directory is a copy, from the escaped path"
    assert links[os.path.join(em8q, "w:z")] == f"{gen}/misalign/em8q/w%3Az" and links[os.path.join(em8q, "lnk:p")] == f"{gen}/misalign/em8q/lnk%3Ap"
    assert links[os.path.join(em8q, "to:q")] == f"{gen}/misalign/em8q/to%3Aq" and links[os.path.join(em8q, "plain_to_escaped")] == f"{gen}/misalign/em8q/plain_to_escaped"
    assert links[os.path.join(em8q, "plain")] == f"{gen}/misalign/em8q/plain" and files[os.path.join(em8q, "note.txt")] == f"{gen}/misalign/em8q/note.txt"
    assert links[os.path.join(base, "misalign/other")] == f"{gen}/misalign/other" and links[os.path.join(base, "misalign_16k")] == f"{gen}/misalign_16k"
    assert links[os.path.join(home, "specs/side")] == f"{gen_dir}/sidecar/{pins['sidecar']}", "a component without escapes still links whole"
    assert spec["escaped"] == sorted([[os.path.join(em8q, "em-x::y/deep/h:1.json"), f"{gen}/misalign/em8q/em-x%3A%3Ay/deep/h%3A1.json"],
                                      [os.path.join(em8q, "em-x::y/g.json"), f"{gen}/misalign/em8q/em-x%3A%3Ay/g.json"],
                                      [os.path.join(em8q, "lnk:p"), f"{gen}/misalign/em8q/lnk%3Ap"], [os.path.join(em8q, "to:q"), f"{gen}/misalign/em8q/to%3Aq"],
                                      [os.path.join(em8q, "w:z"), f"{gen}/misalign/em8q/w%3Az"]])
    # the node builds it (the home tree is written at canonical names; the generation untouched) and the adapter's spelling resolves --
    # both source trees cleared first: a real path where a link belongs is refused, never replaced
    shutil.rmtree(base)
    shutil.rmtree(os.path.join(home, "specs/side"))
    built = CR.init_home_tree(spec, home, "w1")
    assert built["mode"] == "built" and not built["refused"], built
    assert open(os.path.join(em8q, "em-x::y/deep/h:1.json"), "rb").read() == b'{"h": 1}' and open(os.path.join(em8q, "em-x::y/g.json"), "rb").read() == b'{"g": 1}'
    assert open(os.path.join(em8q, "w:z/q.json"), "rb").read() == b"q" and os.path.isfile(os.path.join(em8q, "note.txt")) and not os.path.islink(os.path.join(em8q, "note.txt"))
    assert os.readlink(os.path.join(em8q, "lnk:p")) == f"{gen}/misalign/em8q/lnk%3Ap" and open(os.path.join(em8q, "lnk:p"), "rb").read() == b"{}"
    real = os.path.realpath(os.path.join(em8q, "w:z/q.json"))             # through the node-local link onto the escaped directory
    assert real == f"{gen}/misalign/em8q/w%3Az/q.json"
    cache = CR.GenerationHashCache.from_pins(gen_dir, pins)
    rel, ix = cache.index_of(real)
    assert rel == "misalign/em8q/w:z/q.json" and ix.check(rel, R.file_sha256(real), os.path.getsize(real)) is None
    # the links with escaped TARGETS open through the home tree at their canonical names and land on the referent's manifest entry
    # (with the literal target they dangled, and the read lost the input)
    assert open(os.path.join(em8q, "to:q"), "rb").read() == b"q" and open(os.path.join(em8q, "plain_to_escaped"), "rb").read() == b'{"g": 1}'
    through = os.path.realpath(os.path.join(em8q, "plain_to_escaped"))
    assert through == f"{gen}/misalign/em8q/em-x%3A%3Ay/g.json"
    rel_t, ix_t = cache.index_of(through)
    assert rel_t == "misalign/em8q/em-x::y/g.json" and ix_t.check(rel_t, R.file_sha256(through), os.path.getsize(through)) is None
    assert cache.index_of(os.path.realpath(os.path.join(em8q, "to:q")))[0] == "misalign/em8q/w:z/q.json"
    copy = os.path.join(em8q, "em-x::y/deep/h:1.json")                     # a COPY of an escaped file: aliased onto its escaped source, as the shard does
    assert os.path.isfile(copy) and not os.path.islink(copy) and os.path.realpath(copy) == copy
    cache.alias_files(spec["files"])
    rel2, ix2 = cache.index_of(copy)
    assert rel2 == "misalign/em8q/em-x::y/deep/h:1.json" and ix2.check(rel2, R.file_sha256(copy), os.path.getsize(copy)) is None
    assert sorted(os.listdir(f"{gen}/misalign/em8q")) == ["em-x%3A%3Ay", "lnk%3Ap", "note.txt", "plain", "plain_to_escaped", "to%3Aq", "w%3Az"], \
        "nothing was created inside the generation"
    assert CR.init_home_tree(spec, home, "w1")["mode"] == "verified"
    # escapes with no directory listing to lay them out, or a manifest whose escapes disagree with the rule, refuse by name
    with pytest.raises(CR.ClusterReadsError, match="no directory listing"):
        CR.home_tree_spec(cfg.components, pins, gen_dir, home, entries_of, escapes_of=esc_of)
    with pytest.raises(CR.ClusterReadsError, match="disagree with the escape rule"):
        CR.home_tree_spec(cfg.components, pins, gen_dir, home, entries_of, subentries_of=sub_of, escapes_of=lambda s, g: {"misalign/em8q/w:z/q.json": "misalign/em8q/wz/q.json"} if s == "grades" else {})
    # the plan's own call (manifest copies on the grading host) carries the three callbacks: a window plan lays the escaped entries out
    plan_spec = CR.home_tree_spec(cfg.components, pins, gen_dir, home, lambda s, g: CR.manifest_entries(root, s, g),
                                  subentries_of=lambda s, g, d: CR.manifest_entries(root, s, g, d), escapes_of=lambda s, g: CR.generation_escapes(root, s, g))
    assert plan_spec == spec


# --------------------------------------------------------------------------------------------
# two components sharing a base: every generation pinned at the base is admitted by the shard
# --------------------------------------------------------------------------------------------


def _world_shared_base(tmp_path, monkeypatch):
    """`_world_with_generation` with the synthetic stores split into TWO components over ONE base (`<home>/stores`: `stores_s` holds the
    `s*` groups, `stores_t` the `t*` groups -- the shape of `$GRADE`, grade_alphasteer + grade_arditi), both read by the one
    adapter and both mirrored as published generations. The group `t1` is written after the first fold and before the mirror, so the
    cluster read publishes it from the SECOND component's generation."""
    root, home, stores, ad = _grading_host(tmp_path)
    t1 = _store(stores, "t1", "legacy_store_2", "none", [0.6, 0.4, 0.9])
    monkeypatch.setattr(ST, "HOME", home)
    cfgp = TS._cfg(tmp_path, {"stores_s": {"base": stores, "members": ["s*"], "authority": "grading_host_copy", "origin": "grading_host"},
                              "stores_t": {"base": stores, "members": ["t*"], "authority": "grading_host_copy", "origin": "grading_host"}},
                   adapters={ADAPTER: {"shards": 2, "components": ["stores_s", "stores_t"]}})
    cfg = ST.load_config(cfgp)
    dest = cfg.destination("cluster_a")
    pins = {}
    for slug in ("stores_s", "stores_t"):
        res = ST.mirror_component(root, dest, cfg.components[slug], runner=TS.Runner())
        assert res.outcome == "published", res.as_dict()
        pins[slug] = res.content_sha8
    monkeypatch.setattr("sisteer.registry.adapters.load_all", lambda: {ADAPTER: lambda: SyntheticStoreAdapter(stores)})
    return {"root": root, "home": home, "stores": stores, "ad": ad, "cfg": cfg, "cfgp": cfgp, "dest": dest, "pins": pins, "t1": t1,
            "gen_dir": f"{dest.stores_dir}/gen", "shared": dest.shared}


def test_run_shard_admits_inputs_from_both_generations_pinned_at_a_shared_base(tmp_path, monkeypatch):
    """`run_shard` bound the escape refusal's bases as a `{base: generation}` dict, so with two components at one base
    only one generation survived and every input of the other component was refused as escaping -- the adapter's discovery refused, its
    groups never read on the cluster, the grading host's fallback (since removed) the only way through. End to end over the shard: the plan
    pins both components at `<home>/stores`, the home tree lays the base out as a REAL directory with each component's entries linked onto
    ITS generation, and both shards admit, hash and check every input of both components against its own manifest -- nothing escapes, the
    group held only by the second component's generation is published with the canonical spelling, and a stray file written into the real
    directory is still refused by name before any byte is read."""
    w = _world_shared_base(tmp_path, monkeypatch)
    plan, plan_file, _info = _plan_and_stage(w, tmp_path)
    entry = plan["adapters"][ADAPTER]
    assert entry["route"] == "cluster" and entry["pins"] == w["pins"] and len(plan["tasks"]) == 2, entry
    assert entry["revalidation_pre"]["stores_s"]["ok"] is True and entry["revalidation_pre"]["stores_t"]["ok"] is True and plan["refusals"] == {}
    gen_s, gen_t = (f"{w['gen_dir']}/{s}/{w['pins'][s]}" for s in ("stores_s", "stores_t"))
    links = {p: t for p, t in plan["home_tree"]["links"]}
    assert w["stores"] in plan["home_tree"]["dirs"] and links[os.path.join(w["stores"], "s1")] == f"{gen_s}/s1" and links[os.path.join(w["stores"], "t1")] == f"{gen_t}/t1", \
        "a shared base is a real directory whose entries link onto their OWN component's generation"
    assert set(w["attestation"]["generations"]) == {"stores_s", "stores_t"}
    _node(w)
    for i in range(2):
        assert CR.run_shard(f"{plan['window_root']}/plan.json", i, home=w["home"]) == 0
    assert os.path.isdir(w["stores"]) and not os.path.islink(w["stores"]) and os.readlink(os.path.join(w["stores"], "t1")) == f"{gen_t}/t1"
    docs = [json.load(open(CR.summary_path(plan["summaries_dir"], ADAPTER, i, 2))) for i in range(2)]
    assert all(d["ok"] is True and d["refused"] == [] and d["pass"]["groups_refused"] == 0 for d in docs), [(d.get("refusal"), d["refused"]) for d in docs]
    # the escape refusal admitted every input of BOTH components: four stores hashed and checked against their own manifests, none escaped
    assert all(d["pass"]["inputs_escaped"] == [] and d["pass"]["inputs_refused_by_manifest"] == [] for d in docs), [d["pass"] for d in docs]
    assert all(d["pass"]["inputs_hashed"] == 4 and d["pass"]["inputs_checked_against_manifest"] == 4 and d["pass"]["inputs_hash_reused"] == 0 for d in docs), [d["pass"] for d in docs]
    assert all(set(d["admission_pre"]) == set(d["admission_post"]) == {"stores_s", "stores_t"} and d["admission_pre"]["stores_t"]["ok"] and d["admission_post"]["stores_s"]["ok"] for d in docs)
    assert all(set(d["read_copy"]["components"]) == {"stores_s", "stores_t"} and d["read_copy"]["components"]["stores_t"]["content_sha8"] == w["pins"]["stores_t"] for d in docs)
    # the new group t1 -- held only by stores_t's generation -- is published by exactly one shard, spelled canonically
    pub = [d for d in docs if d["published"]]
    assert len(pub) == 1 and len(pub[0]["published"]) == 1 and sum(d["pass"]["groups_not_owned"] for d in docs) == 1
    rev_dir, pinfo = next(iter(pub[0]["published"].items()))
    assert pinfo["group"] == "t1" and pinfo["marker"]
    marker = _marker(os.path.join(plan["root"], "records", "adapters", rev_dir))
    assert marker["payload"]["inputs_json"] == {w["t1"]: R.file_sha256(w["t1"])}, "the canonical spelling under the shared base, never the generation's path"
    assert marker["payload"]["extra"]["read_copy"]["components"]["stores_t"]["generation"] == gen_t
    # a stray written into the REAL shared directory (no component holds it) is refused by name before any byte is read: the discovery
    # refuses, the summary names the input under inputs_escaped, nothing of it is published
    stray = _store(w["stores"], "stray", "legacy_store_3", "none", [0.5])
    assert CR.run_shard(f"{plan['window_root']}/plan.json", 0, home=w["home"], summaries_dir=str(tmp_path / "s_stray")) == 0
    d3 = json.load(open(CR.summary_path(str(tmp_path / "s_stray"), ADAPTER, 0, 2)))
    assert d3["pass"]["inputs_escaped"] == [stray] and d3["pass"]["groups_refused"] == 1 and d3["published"] == {}, d3["pass"]
    assert len(d3["refused"]) == 1 and d3["refused"][0]["group"] == "<discover>" and "escapes the pinned generation" in d3["refused"][0]["error"] \
        and stray in d3["refused"][0]["error"] and "2 generations pinned at that base" in d3["refused"][0]["error"], d3["refused"]


# --------------------------------------------------------------------------------------------
# part 8: the refused-known revisions re-activated by the pull from committed bytes, with no store read
# --------------------------------------------------------------------------------------------


def test_the_pull_reactivates_a_refused_known_revision_from_committed_bytes_without_a_store_read_and_leaves_a_current_one_alone(tmp_path):
    """The cluster refuses by name a group whose minted id the grading host already knows and publishes nothing for it; the
    grading host's half is `cluster-reads pull`: every refused-known id whose `source_revision` row is NOT the source's current one is re-activated
    by `runner.reactivate_by_id` from the COMMITTED bytes -- the marker's `inputs_json` and the committed `source_watermark` / `source_state`
    records stand in for a discovery, the measurements are re-emitted as an activation bundle the harvester folds -- with NO store read:
    run on an unscoped host with the audit hook on the store root, the bundle appears and the audit log stays empty. A refused-known id that
    IS the current head is `current` (nothing written); a marker stripped of its watermark facts is `refused` by name and the adapter is
    recorded refused for the window."""
    root, home, stores, ad = _grading_host(tmp_path)
    sidecar = str(tmp_path / "rs.sqlite")
    _export(root, sidecar, home)
    cluster_root = _cluster_root(tmp_path)
    rep0 = _cluster_pass(cluster_root, ad, sidecar, synth.t(3050), full=True)
    old_rev = next(r.revision_id for r in rep0.results if r.group == "s2")
    s2 = _store(str(stores), "s2", "legacy_store_2", "none", [0.1, 0.9, 0.5])
    rep2 = _cluster_pass(cluster_root, ad, sidecar, synth.t(3100))
    new = _published(cluster_root)
    assert rep2.groups_refreshed == 1 and len(new) == 1
    shutil.copytree(os.path.join(H.records_dir(cluster_root), "adapters", ADAPTER, new[0]), os.path.join(H.records_dir(root), "adapters", ADAPTER, new[0]))
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    # s2 reverted on the grading host: its old id is accepted (superseded) -> the cluster refuses it as known; s1 unchanged -> its id is the current head
    _store(str(stores), "s2", "legacy_store_2", "none", [0.1, 0.9])
    sidecar2 = str(tmp_path / "rs2.sqlite")
    _export(root, sidecar2, home)
    cluster_root2 = str(tmp_path / "cluster_root2")
    H.ensure_layout(cluster_root2)
    rep5 = _cluster_pass(cluster_root2, ad, sidecar2, synth.t(3400), full=True)
    known = {r.group: r for r in rep5.results if r.refused_known}
    assert set(known) == {"s1", "s2", "s3"} and known["s2"].revision_id == old_rev
    s1_rev = known["s1"].revision_id
    # the pull's half, as the CLI runs it: a report carrying the refused-known ids, an unscoped host, the audit hook on the store root
    report = {"refused_known_ids": {ADAPTER: [old_rev, s1_rev]}, "adapters_refused": {}}
    audit = tmp_path / "audit.jsonl"
    AB.install_store_audit(str(audit), [stores])
    try:
        with AB.without_store_access_scope(), DB.reading(DB.db_path(root)) as conn:
            out = CR.reactivate_refused_known(root, {"adapters": {ADAPTER: {"route": "cluster"}}}, report, conn=conn, now=synth.t(3500), adapter_of=lambda name: ad)
    finally:
        AB.disable_store_audit()
    assert not audit.exists() or audit.read_text() == "", audit.read_text() if audit.exists() else ""      # no store byte, no listing
    assert out[ADAPTER][old_rev]["state"] == "reactivated" and out[ADAPTER][old_rev]["files"] >= 2 and out[ADAPTER][old_rev]["records"] > 0
    assert out[ADAPTER][s1_rev]["state"] == "current" and out[ADAPTER][s1_rev]["files"] == 0
    assert report["adapters_refused"] == {} and report["reactivations"] == out
    bundles = RUN.complete_bundles(root, ADAPTER, old_rev)
    assert len(bundles) == 1
    marker = next(rec for _, rec, _ in R.read_record_file(bundles[0][1]) if rec["record_type"] == "source_revision")
    assert marker["payload"]["inputs_json"] == {s2: R.file_sha256(s2)} and marker["payload"]["extra"]["reactivated_from_committed_bytes"]["inputs"] == 1
    assert any(rec["record_type"] == "source_state" for _, rec, _ in R.read_record_file(bundles[0][1]))      # the committed facts, re-stamped
    # the harvester folds the bundle: the reverted id is the head again, exactly as the grading host pass used to do it by reading the store
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    head = _rows(root, "SELECT revision_id FROM source_revision WHERE source_id = ? AND accepted = 1 ORDER BY observed_utc DESC LIMIT 1", (f"{ADAPTER}:s2",))
    assert head == [(old_rev,)]
    # a second call: the id is current now -> nothing written, no second bundle
    report2 = {"refused_known_ids": {ADAPTER: [old_rev]}, "adapters_refused": {}}
    with DB.reading(DB.db_path(root)) as conn:
        out2 = CR.reactivate_refused_known(root, {}, report2, conn=conn, now=synth.t(3600), adapter_of=lambda name: ad)
    assert out2[ADAPTER][old_rev]["state"] == "current" and len(RUN.complete_bundles(root, ADAPTER, old_rev)) == 1
    # a marker without its committed watermark facts refuses by name, and the adapter is recorded refused
    s3_rev = known["s3"].revision_id
    rev_dir = os.path.join(H.records_dir(root), "adapters", ADAPTER, s3_rev)
    marker_file = next(os.path.join(rev_dir, f) for f in os.listdir(rev_dir) if "source_revision" in f)
    kept = [rec for _, rec, _ in R.read_record_file(marker_file) if rec["record_type"] != "source_watermark"]
    os.remove(marker_file)
    R.write_records(marker_file, kept)
    # make s3 non-current first (a changed s3 folded), so the re-activation is attempted at all
    _store(str(stores), "s3", "legacy_store_3", "constitution_lora@aaaa", [0.7, 0.7, 0.2, 0.1])
    with DB.reading(DB.db_path(root)) as conn:
        RUN.run_pass(root, [ad], conn=conn, now=synth.t(3700), producer=PRODUCER)
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    report3 = {"refused_known_ids": {ADAPTER: [s3_rev]}, "adapters_refused": {}}
    with DB.reading(DB.db_path(root)) as conn:
        out3 = CR.reactivate_refused_known(root, {}, report3, conn=conn, now=synth.t(3800), adapter_of=lambda name: ad)
    assert out3[ADAPTER][s3_rev]["state"] == "refused" and "no matching source_watermark fact" in out3[ADAPTER][s3_rev]["error"]
    assert any(s3_rev in r and "not re-activated from committed bytes" in r for r in report3["adapters_refused"][ADAPTER])
    assert RUN.complete_bundles(root, ADAPTER, s3_rev) == []
    # the refusal set a window reads names it, under the pull's prefix
    fb = CR.refused_adapters({"adapters": {ADAPTER: {"route": "cluster", "shards": 1, "task_indices": [0]}}, "tasks": [{"adapter": ADAPTER}], "refusals": {}}, report3)
    assert ADAPTER in fb and any(r.startswith("pull: refused-known revision") for r in fb[ADAPTER])


def test_reactivate_by_id_refuses_an_absent_directory_and_leaves_a_pending_or_held_revision_alone(tmp_path):
    """`runner.reactivate_by_id` writes nothing for: a revision id with no committed directory (refused by name), a directory whose marker
    is on the log but not folded yet (pending -- the harvester's), a revision the database holds pending (held)."""
    root, home, stores, ad = _grading_host(tmp_path)
    with DB.reading(DB.db_path(root)) as conn:
        res = RUN.reactivate_by_id(root, ad, "0" * 16, conn, now=synth.t(4000))
        assert not res.accepted and not res.held and "no committed revision directory" in res.error and res.files == []
        # a published, not yet folded directory: pending
        s4 = _store(str(stores), "s4", "legacy_store_4", "none", [0.4])
        rep = RUN.run_pass(root, [ad], conn=conn, now=synth.t(4100), producer=PRODUCER)
        s4_rev = next(r.revision_id for r in rep.results if r.group == "s4")
    with DB.reading(DB.db_path(root)) as conn:
        res2 = RUN.reactivate_by_id(root, ad, s4_rev, conn, now=synth.t(4200))
        assert not res2.accepted and res2.held and res2.activation_pending and "not folded yet" in res2.error and res2.files == []
    assert RUN.complete_bundles(root, ADAPTER, s4_rev) == []


def _emit_time_refused_s1(root, stores, ad, now_refuse: int = 2100):
    """s1 changed and its grading host read REFUSED at emit time, folded: the only artifact is `_refused/<rev>/<stamp>-source_revision` -- an
    `accepted = 0` row carrying the reader's words, `record_uri` under `_refused/`; no directory, no bundle. Returns the id; the reader
    stays broken on s1 (the caller clears `ad.broken`)."""
    s1 = os.path.join(stores, "s1", "store.json")
    ad.broken.add(s1)
    _store(str(stores), "s1", "legacy_store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4, 0.9])
    with DB.reading(DB.db_path(root)) as conn:
        rep = RUN.run_pass(root, [ad], conn=conn, now=synth.t(now_refuse), producer=PRODUCER)
    assert rep.groups_refused == 1 and H.ingest_once(root, workers=0, min_free=0).refused == 0
    rev = next(r.revision_id for r in rep.results if r.group == "s1")
    assert not os.path.isdir(os.path.join(H.records_dir(root), "adapters", ADAPTER, rev)) and RUN.complete_bundles(root, ADAPTER, rev) == []
    assert os.path.isdir(RUN.refused_dir(root, ADAPTER, rev))
    row = _rows(root, "SELECT accepted, refusal, record_uri FROM source_revision WHERE revision_id = ?", (rev,))
    assert len(row) == 1 and row[0][0] == 0 and "synthetic read failure" in row[0][1] and "/_refused/" in row[0][2], row
    return rev


def test_an_emit_time_refused_id_is_read_fresh_on_the_shard_and_the_pull_records_it_standing_never_as_the_adapters_refusal(tmp_path):
    """Both halves joined. A read of s1 on the grading host REFUSES at emit time (`_emit_time_refused_s1`). (1) The shard: the id is NOT known
    (`ClusterRead.known` -> None), so under `retry_refused` the group is READ FRESH -- refused again by the shard's own reader while the
    reader is broken (the refusal travels as the GROUP's, the adapter's real cause, under the same id), published once the reader is
    fixed. (2) The pull: an older shard's claim of the id fed through `reactivate_refused_known` is `refused_standing` -- nothing written,
    no bundle, no directory -- and `adapters_refused` stays empty. Before this the shard returned it refused-known ('source_revision row,
    pending'), `reactivate_by_id` refused 'no committed revision directory' first, and the ADAPTER was refused for the window (60 ids over
    four adapters in the first post-cutover window, 2026-10-07)."""
    root, home, stores, ad = _grading_host(tmp_path)
    s1_rev = _emit_time_refused_s1(root, stores, ad)
    refusal = _rows(root, "SELECT refusal FROM source_revision WHERE revision_id = ?", (s1_rev,))[0][0]
    with DB.reading(DB.db_path(root)) as conn:
        assert RUN._emit_time_refusal(conn, s1_rev) == (f"{ADAPTER}:s1", refusal)
        assert RUN._revision_state(conn, s1_rev) == "pending"            # the row's state is what it was: the callers pair it with the bytes
    sidecar = str(tmp_path / "rs.sqlite")
    _export(root, sidecar, home)
    conn = RS.open_reader_state(sidecar, [ad.name])
    try:
        assert RS.known_dir_kinds(conn)[s1_rev] == {"refused"}
        cluster = RUN.ClusterRead.from_sidecar(conn)
        assert s1_rev in cluster.known_dirs and not cluster.committed_on_grading_host(s1_rev)
        assert cluster.known(conn, s1_rev) is None and cluster.refuse_known(conn, ADAPTER, "s1", s1_rev, f"{ADAPTER}:s1") is None
    finally:
        conn.close()
    # (1a) the shard reads it fresh; its own reader refuses -> the group is REFUSED on the shard (never refused-known), the cause the reader's.
    # The same words over the same inputs under the same tree (PRODUCER's code sha on both sides): a STANDING refusal -- recorded,
    # counted apart (`groups_standing_refusals`), never the adapter's refusal
    cluster_root = _cluster_root(tmp_path)
    rep_c = _cluster_pass(cluster_root, ad, sidecar, synth.t(3000), retry_refused=True)
    by = {r.group: r for r in rep_c.results}
    assert set(by) == {"s1"} and not by["s1"].accepted and not by["s1"].refused_known and not by["s1"].held, by["s1"]
    assert by["s1"].revision_id == s1_rev and "synthetic read failure" in by["s1"].error and by["s1"].standing_refusal and by["s1"].prior_refusal["attempt_revision_id"] == s1_rev
    assert not by["s1"].refused_standing and rep_c.groups_refused == 0 and rep_c.groups_standing_refusals == 1 and rep_c.groups_refused_known == 0 and rep_c.groups_refused_standing == 0
    assert _published(cluster_root) == ["_refused"] and os.path.isdir(RUN.refused_dir(cluster_root, ADAPTER, s1_rev))   # no revision directory: the shard's own refused record, under the same id
    # (1b) the reader fixed: the same retry publishes the very id the refused read computed
    ad.broken.clear()
    cluster_root2 = _cluster_root(tmp_path, "cluster_root2")
    rep_d = _cluster_pass(cluster_root2, ad, sidecar, synth.t(3100), retry_refused=True)
    by = {r.group: r for r in rep_d.results}
    assert by["s1"].accepted and by["s1"].revision_id == s1_rev and rep_d.groups_refreshed == 1 and rep_d.groups_refused_known == 0
    assert _published(cluster_root2) == [s1_rev]
    # (2) the pull's half, on an older shard's claim: refused_standing, nothing written, the adapter not refused
    report = {"refused_known_ids": {ADAPTER: [s1_rev]}, "adapters_refused": {}}
    with DB.reading(DB.db_path(root)) as conn:
        out = CR.reactivate_refused_known(root, {"adapters": {ADAPTER: {"route": "cluster"}}}, report, conn=conn, now=synth.t(3200), adapter_of=lambda name: ad)
    assert out[ADAPTER][s1_rev]["state"] == "refused_standing" and out[ADAPTER][s1_rev]["files"] == 0
    assert "refused at emit time" in out[ADAPTER][s1_rev]["error"] and "synthetic read failure" in out[ADAPTER][s1_rev]["error"]
    assert report["adapters_refused"] == {} and report["reactivations"] == out
    assert not os.path.isdir(os.path.join(H.records_dir(root), "adapters", ADAPTER, s1_rev)) and RUN.complete_bundles(root, ADAPTER, s1_rev) == []
    fb = CR.refused_adapters({"adapters": {ADAPTER: {"route": "cluster", "shards": 1, "task_indices": [0]}}, "tasks": [{"adapter": ADAPTER}], "refusals": {}}, report)
    assert not fb.get(ADAPTER), fb
    assert _rows(root, "SELECT accepted FROM source_revision WHERE revision_id = ?", (s1_rev,)) == [(0,)]     # the grading host's row as it was


def test_an_emit_time_refused_id_the_grading_host_has_since_published_is_still_known_to_the_shard(tmp_path):
    """The BYTES decide, never the row alone: after s1's emit-time refusal the grading host's reader is fixed and a grading host retry publishes the very
    id (a directory, not folded yet). The row is still the refused fold (accepted = 0, the reader's words) but the sidecar lists a revision
    directory, so the shard calls the id known -- nothing read or published there -- and the pull's `reactivate_by_id` leaves it to the
    harvester (held, the fold pending), never `refused_standing`."""
    root, home, stores, ad = _grading_host(tmp_path)
    s1_rev = _emit_time_refused_s1(root, stores, ad)
    ad.broken.clear()
    with DB.reading(DB.db_path(root)) as conn:
        rep2 = RUN.run_pass(root, [ad], conn=conn, now=synth.t(2200), producer=PRODUCER, retry_refused=True)
    r1 = next(r for r in rep2.results if r.group == "s1")
    assert r1.accepted and r1.revision_id == s1_rev and os.path.isdir(os.path.join(H.records_dir(root), "adapters", ADAPTER, s1_rev))
    assert _rows(root, "SELECT accepted FROM source_revision WHERE revision_id = ?", (s1_rev,)) == [(0,)]      # NOT folded: the row is the refusal's
    sidecar = str(tmp_path / "rs.sqlite")
    _export(root, sidecar, home)
    conn = RS.open_reader_state(sidecar, [ad.name])
    try:
        assert RS.known_dir_kinds(conn)[s1_rev] == {"refused", "revision"}
        cluster = RUN.ClusterRead.from_sidecar(conn)
        assert cluster.committed_on_grading_host(s1_rev) and cluster.known(conn, s1_rev) == "source_revision row, pending"
    finally:
        conn.close()
    cluster_root = _cluster_root(tmp_path)
    rep_c = _cluster_pass(cluster_root, ad, sidecar, synth.t(3000), retry_refused=True)
    by = {r.group: r for r in rep_c.results}
    assert by["s1"].refused_known and by["s1"].revision_id == s1_rev and not _published(cluster_root)
    with DB.reading(DB.db_path(root)) as conn:
        res = RUN.reactivate_by_id(root, ad, s1_rev, conn, now=synth.t(3100))
    assert not res.accepted and res.held and res.activation_pending and not res.refused_standing and "not folded yet" in res.error and res.files == []


def test_a_committed_revision_refused_on_a_later_reread_is_still_known_and_re_activated_from_its_bytes(tmp_path):
    """The other face, unchanged: a revision COMMITTED (accepted, since superseded) whose later re-read REFUSED left only a refused
    watermark stamp under `_refused/<rev>/` -- its row stays `accepted = 1`, its directory stands. `_emit_time_refusal` says None, the shard
    calls it known ('source_revision row, accepted'), and the pull re-activates it from its committed bytes as before."""
    root, home, stores, ad = _grading_host(tmp_path)
    s2 = os.path.join(stores, "s2", "store.json")
    old_rev = _rows(root, "SELECT revision_id FROM source_revision WHERE source_id = ? AND accepted = 1", (f"{ADAPTER}:s2",))[0][0]
    # s2 changes and folds (old_rev superseded), then reverts; the reverted revision's re-read REFUSES (the reader broken on it): a stamp, no marker
    _store(str(stores), "s2", "legacy_store_2", "none", [0.1, 0.9, 0.5])
    with DB.reading(DB.db_path(root)) as conn:
        RUN.run_pass(root, [ad], conn=conn, now=synth.t(2100), producer=PRODUCER)
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    _store(str(stores), "s2", "legacy_store_2", "none", [0.1, 0.9])
    ad.broken.add(s2)
    with DB.reading(DB.db_path(root)) as conn:
        rep = RUN.run_pass(root, [ad], conn=conn, now=synth.t(2200), producer=PRODUCER, reread=True)
    r2 = next(r for r in rep.results if r.group == "s2")
    assert not r2.accepted and r2.revision_id == old_rev and r2.files and all("/_refused/" in f and "source_watermark" in f for f in r2.files), r2
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    ad.broken.clear()
    assert _rows(root, "SELECT accepted FROM source_revision WHERE revision_id = ?", (old_rev,)) == [(1,)]
    assert os.path.isdir(os.path.join(H.records_dir(root), "adapters", ADAPTER, old_rev)) and os.path.isdir(RUN.refused_dir(root, ADAPTER, old_rev))
    with DB.reading(DB.db_path(root)) as conn:
        assert RUN._emit_time_refusal(conn, old_rev) is None
    sidecar = str(tmp_path / "rs.sqlite")
    _export(root, sidecar, home)
    conn = RS.open_reader_state(sidecar, [ad.name])
    try:
        assert RS.known_dir_kinds(conn)[old_rev] == {"revision", "refused"}
        cluster = RUN.ClusterRead.from_sidecar(conn)
        assert cluster.committed_on_grading_host(old_rev) and cluster.known(conn, old_rev) == "source_revision row, accepted"
    finally:
        conn.close()
    cluster_root = _cluster_root(tmp_path)
    rep_c = _cluster_pass(cluster_root, ad, sidecar, synth.t(3000), retry_refused=True)
    by = {r.group: r for r in rep_c.results}
    assert by["s2"].refused_known and by["s2"].revision_id == old_rev and "source_revision row, accepted" in by["s2"].error and not _published(cluster_root)
    report = {"refused_known_ids": {ADAPTER: [old_rev]}, "adapters_refused": {}}
    with DB.reading(DB.db_path(root)) as conn:
        out = CR.reactivate_refused_known(root, {"adapters": {ADAPTER: {"route": "cluster"}}}, report, conn=conn, now=synth.t(3100), adapter_of=lambda name: ad)
    assert out[ADAPTER][old_rev]["state"] == "reactivated" and out[ADAPTER][old_rev]["files"] >= 2 and report["adapters_refused"] == {}
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    head = _rows(root, "SELECT revision_id FROM source_revision WHERE source_id = ? AND accepted = 1 ORDER BY observed_utc DESC LIMIT 1", (f"{ADAPTER}:s2",))
    assert head == [(old_rev,)]


def test_reactivate_by_id_still_refuses_a_committed_id_whose_directory_is_missing_and_never_reads_a_hold_as_an_emit_time_refusal(tmp_path):
    """The integrity half left alone: an ACCEPTED (superseded) id whose directory is gone refuses by name ('no committed revision
    directory') and the adapter is recorded refused -- bytes the database vouches for are missing. And a row the harvester HOLDS (`refusal`
    under the `activation pending:` label) is never an emit-time refusal, whatever its `record_uri` (a directory published after an
    emit-time refusal and then held keeps the refused record's uri under the hold's label: nine live rows): without a directory it
    refuses by name too, and the shard keeps calling it known (pending)."""
    root, home, stores, ad = _grading_host(tmp_path)
    old_rev = _rows(root, "SELECT revision_id FROM source_revision WHERE source_id = ? AND accepted = 1", (f"{ADAPTER}:s2",))[0][0]
    _store(str(stores), "s2", "legacy_store_2", "none", [0.1, 0.9, 0.5])
    with DB.reading(DB.db_path(root)) as conn:
        RUN.run_pass(root, [ad], conn=conn, now=synth.t(2100), producer=PRODUCER)
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    shutil.rmtree(os.path.join(H.records_dir(root), "adapters", ADAPTER, old_rev))
    with DB.reading(DB.db_path(root)) as conn:
        assert RUN._emit_time_refusal(conn, old_rev) is None
        res = RUN.reactivate_by_id(root, ad, old_rev, conn, now=synth.t(2200))
    assert not res.accepted and not res.held and not res.refused_standing and "no committed revision directory" in res.error and res.files == []
    report = {"refused_known_ids": {ADAPTER: [old_rev]}, "adapters_refused": {}}
    with DB.reading(DB.db_path(root)) as conn:
        out = CR.reactivate_refused_known(root, {}, report, conn=conn, now=synth.t(2300), adapter_of=lambda name: ad)
    assert out[ADAPTER][old_rev]["state"] == "refused"
    assert any(old_rev in r and "no committed revision directory" in r for r in report["adapters_refused"][ADAPTER])
    # a hold-labelled row with the refused record's uri and no directory: not an emit-time refusal -- refused by name, and still known to a shard
    s1_rev = _emit_time_refused_s1(root, stores, ad, now_refuse=2400)
    ad.broken.clear()
    w = sqlite3.connect(DB.db_path(root))          # the harvester's own relabelling of the row (harvest `_placeholder_revision`), minimally
    try:
        w.execute("UPDATE source_revision SET refusal = ? WHERE revision_id = ?", (RUN.HOLD_LABEL + " source_revision x (synthetic hold)", s1_rev))
        w.commit()
    finally:
        w.close()
    with DB.reading(DB.db_path(root)) as conn:
        assert RUN._emit_time_refusal(conn, s1_rev) is None and RUN._revision_state(conn, s1_rev) == "pending"
        res2 = RUN.reactivate_by_id(root, ad, s1_rev, conn, now=synth.t(2500))
    assert not res2.accepted and not res2.refused_standing and "no committed revision directory" in res2.error
    sidecar = str(tmp_path / "rs.sqlite")
    _export(root, sidecar, home)
    conn = RS.open_reader_state(sidecar, [ad.name])
    try:
        cluster = RUN.ClusterRead.from_sidecar(conn)
        assert RS.known_dir_kinds(conn)[s1_rev] == {"refused"} and cluster.known(conn, s1_rev) == "source_revision row, pending"
    finally:
        conn.close()


# --------------------------------------------------------------------------------------------
# part 7: the refused-known gate at the pull, and the pre-status refusal of a re-activation by id
# --------------------------------------------------------------------------------------------


def _window_root_with_one_summary(tmp_path, cluster_root, rep, producer, extra=None, name="wroot"):
    """A window root as one shard leaves it: the shard summary of the refused-known-only pass `rep` at `<root>/<adapter>/0-of-1.json`
    and an EMPTY spool (`root/records/adapters/<adapter>/`: the pass published nothing) -- what `pull_window` reads with a local source."""
    wroot = tmp_path / name
    os.makedirs(wroot / "root" / "records" / "adapters" / ADAPTER)
    os.makedirs(wroot / ADAPTER)
    CR.write_shard_summary(CR.summary_path(str(wroot), ADAPTER, 0, 1), rep, rep.as_dict(), [ADAPTER], (0, 1), cluster_root, producer, extra)
    return wroot


def _reverted_s2_world(tmp_path) -> Dict:
    """The refused-known fixture of the tests below. A grading host where s2 changed and folded (its first revision `old_rev` is accepted and
    SUPERSEDED by `new_rev`) and then REVERTED on disk, read once on the cluster against the exported sidecar with `--full`: the shard mints
    old_rev again and refuses it as known (non-current on the grading host), s1 and s3 are known and current, nothing is published. The window
    root holds that one shard's summary and an empty spool (`_window_root_with_one_summary`) and NO attestation record yet -- each test
    lays down the record it means to test. The plan pins the adapter's one component at `deadbeef`, the generation a clean record attests."""
    root, home, stores, ad = _grading_host(tmp_path)
    ((old_rev,),) = _rows(root, "SELECT revision_id FROM source_revision WHERE source_id = ? AND accepted = 1", (f"{ADAPTER}:s2",))
    # s2 changes on the grading host and folds: old_rev is accepted and superseded
    _store(str(stores), "s2", "legacy_store_2", "none", [0.1, 0.9, 0.5])
    with DB.reading(DB.db_path(root)) as conn:
        rep1 = RUN.run_pass(root, [ad], conn=conn, now=synth.t(2500), producer=PRODUCER)
    assert rep1.groups_refreshed == 1 and H.ingest_once(root, workers=0, min_free=0).refused == 0
    new_rev = next(r.revision_id for r in rep1.results if r.group == "s2")
    assert new_rev != old_rev
    # s2 reverted: the cluster read mints old_rev again and refuses it as known (non-current on the grading host); s1 and s3 are known and current
    _store(str(stores), "s2", "legacy_store_2", "none", [0.1, 0.9])
    sidecar = str(tmp_path / "rs.sqlite")
    _export(root, sidecar, home)
    cluster_root = _cluster_root(tmp_path)
    rep = _cluster_pass(cluster_root, ad, sidecar, synth.t(3000), full=True)
    known = {r.group: r.revision_id for r in rep.results if r.refused_known}
    assert set(known) == {"s1", "s2", "s3"} and known["s2"] == old_rev and _published(cluster_root) == []
    wroot = _window_root_with_one_summary(tmp_path, cluster_root, rep, dict(PRODUCER, substrate="cluster_a", host="node-x"))
    plan = {"window": "w1", "host": None, "window_root": str(wroot), "adapters": {ADAPTER: {"route": "cluster", "shards": 1, "pins": {"synthetic_stores": "deadbeef"}}},
            "tasks": [{"adapter": ADAPTER}], "stages": {"revalidate_post": {"stale": []}}, "refusals": {}}
    return {"root": root, "ad": ad, "old_rev": old_rev, "new_rev": new_rev, "known": known, "wroot": wroot, "plan": plan,
            "summary_file": CR.summary_path(str(wroot), ADAPTER, 0, 1), "attestation": str(wroot / ST.ATTESTATION_FILE)}


def _s2_head(root) -> List:
    return _rows(root, "SELECT revision_id FROM source_revision WHERE source_id = ? AND accepted = 1 ORDER BY observed_utc DESC LIMIT 1", (f"{ADAPTER}:s2",))


def test_the_pull_withholds_the_refused_known_ids_of_a_stale_adapter_a_refused_shard_and_an_unadmitting_attestation_and_replays_none(tmp_path):
    """A refused-known id is the shard's CLAIM (the id it minted from the bytes it read is one the grading host holds accepted),
    and its replay writes the LIVE record log past the inbox barrier -- the window's later refusal does not undo it. So the pull trusts it
    exactly as far as the shard's publications would be delivered, and no further. A non-current known id (s2 reverted on the
    grading host: its old revision is accepted and superseded) returned by (a) a shard of an adapter the post-run revalidation refused, (b) a
    shard attesting ok = false, (c) a shard under an attestation record that refused a pin the adapter reads, (d) another operation's
    record, (e) this operation's record attesting the component at ANOTHER generation than the pinned one, or (f) a record that names no
    such component, is WITHHELD by name: `refused_known_ids` is empty, `refused_known_withheld` carries every id with the reason, the
    adapter is recorded refused, and `reactivate_refused_known` writes NO bundle. Every case runs with a record PRESENT -- the gate is
    fail-closed on a missing one, which is the next test's subject, so a case here fails for its own cause and nothing else. The control
    runs under the clean record of THIS operation (present, this window's, admitting the pin -- the gate's positive path, never the path
    that removes the file): the same summary collects every id and the replay re-activates the non-current one -- the move the ungated
    collection made on a read the window refused."""
    w = _reverted_s2_world(tmp_path)
    root, ad, old_rev, new_rev, known, wroot, plan = w["root"], w["ad"], w["old_rev"], w["new_rev"], w["known"], w["wroot"], w["plan"]
    clean = _attestation_record("w1", {"synthetic_stores": _attested_generation("deadbeef")})
    CR._write_json(w["attestation"], clean)
    n_expect = [0]

    def pull(plan_):
        n_expect[0] += 1
        return CR.pull_window(root, plan_, str(tmp_path / f"expect{n_expect[0]}"), source=str(wroot))

    def replay(rep_):
        with DB.reading(DB.db_path(root)) as conn:
            return CR.reactivate_refused_known(root, plan, rep_, conn=conn, now=synth.t(3500), adapter_of=lambda name: ad)

    def withheld_all(rep_, needle):
        """Every known id withheld with a reason carrying `needle`, the adapter refused with the count, the replay a no-op, no bundle."""
        assert rep_["rsync"]["summaries"] == 0 and rep_["summaries"][ADAPTER]["refused_known"] == 3
        assert rep_["refused_known_ids"][ADAPTER] == [] and set(rep_["refused_known_withheld"][ADAPTER]) == set(known.values()), rep_["refused_known_withheld"]
        assert all(needle in why for why in rep_["refused_known_withheld"][ADAPTER].values()), (needle, rep_["refused_known_withheld"][ADAPTER])
        assert any("3 refused-known revision(s) withheld from re-activation" in r for r in rep_["adapters_refused"][ADAPTER]), rep_["adapters_refused"]
        assert replay(rep_) == {} and rep_["reactivations"] == {} and RUN.complete_bundles(root, ADAPTER, old_rev) == []

    # (a) the post-run revalidation refused the adapter: nothing of it is delivered -- its claims included (the record is present and green)
    stale = pull(dict(plan, stages={"revalidate_post": {"stale": [ADAPTER]}}))
    assert stale["attestation"]["present"] and stale["attestation"]["this_window"] and stale["attestation"]["ok"] is True
    withheld_all(stale, "post-run revalidation refused the adapter")
    # (b) the shard attests ok = false (the attestation no longer admitted a generation after its read): its claims are not trusted
    doc = json.load(open(w["summary_file"]))
    CR._write_json(w["summary_file"], dict(doc, ok=False, refusal="the attestation no longer admits a generation after the read: synthetic_stores: attestation_pin"))
    refused = pull(plan)
    assert refused["summaries"][ADAPTER]["ok"] == 0 and any("1 shard(s) refused" in r for r in refused["adapters_refused"][ADAPTER])
    withheld_all(refused, "the shard that returned it is not ok")
    assert all("attestation_pin" in why for why in refused["refused_known_withheld"][ADAPTER].values())
    CR._write_json(w["summary_file"], doc)
    # (c) the attestation record pulled beside the summaries refused the pin this adapter reads: the premise is void on the grading host's side
    # too, with the record's own refusal and reason
    CR._write_json(w["attestation"], _attestation_record("w1", {"synthetic_stores": _attested_generation("deadbeef", ok=False, refusal="listing_mismatch",
                                                                                                           reason="1 link target differs (first: s2/store.json)")}))
    attested = pull(plan)
    assert attested["attestation"]["present"] and attested["attestation"]["this_window"] and attested["attestation"]["ok"] is False
    withheld_all(attested, "the attestation refused component(s) this adapter reads: synthetic_stores: listing_mismatch (1 link target differs (first: s2/store.json))")
    # (d) another operation's record admits nothing here, however green it reads
    CR._write_json(w["attestation"], _attestation_record("w0", {"synthetic_stores": _attested_generation("deadbeef")}))
    other = pull(plan)
    assert other["attestation"]["present"] and other["attestation"]["this_window"] is False and other["attestation"]["ok"] is True
    withheld_all(other, "the attestation record pulled is operation w0's, not this one's")
    # (e) this operation's record, green, attests the component at ANOTHER generation than the plan pins (`current` re-pointed between the
    # plan and the attestation): the shard's admission would have refused `attestation_pin`, and so does the pull's
    CR._write_json(w["attestation"], _attestation_record("w1", {"synthetic_stores": _attested_generation("cafebabe")}))
    elsewhere = pull(plan)
    assert elsewhere["attestation"]["present"] and elsewhere["attestation"]["this_window"] and elsewhere["attestation"]["ok"] is True
    withheld_all(elsewhere, "the attestation does not admit component(s) this adapter reads: synthetic_stores: attestation_pin "
                            "(the attestation record attests synthetic_stores@cafebabe, not the pinned deadbeef)")
    # (f) this operation's record names no such component at all
    CR._write_json(w["attestation"], _attestation_record("w1", {"another_component": _attested_generation("deadbeef")}))
    unnamed = pull(plan)
    assert unnamed["attestation"]["ok"] is True and set(unnamed["attestation"]["generations"]) == {"another_component"}
    withheld_all(unnamed, "the attestation does not admit component(s) this adapter reads: synthetic_stores: attestation_missing "
                          "(the attestation record of w1 names no component synthetic_stores)")
    # the head never moved: six refused reads, six pulls, zero writes
    assert _s2_head(root) == [(new_rev,)]
    # the control: the clean record of THIS operation back in place -- present, this window's, admitting the pin -- the same summary
    # collects every id, and the replay re-activates the non-current one: exactly one bundle; s1 and s3 are current (nothing written for them)
    CR._write_json(w["attestation"], clean)
    ok = pull(plan)
    assert ok["attestation"]["present"] and ok["attestation"]["this_window"] and ok["attestation"]["ok"] is True
    assert ok["adapters_refused"] == {} and ok["refused_known_withheld"] == {} and ok["refused_known_ids"][ADAPTER] == sorted(known.values())
    out = replay(ok)
    assert out[ADAPTER][old_rev]["state"] == "reactivated" and out[ADAPTER][known["s1"]]["state"] == "current" and out[ADAPTER][known["s3"]]["state"] == "current"
    assert len(RUN.complete_bundles(root, ADAPTER, old_rev)) == 1 and ok["adapters_refused"] == {}
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    assert _s2_head(root) == [(old_rev,)]


def test_the_pull_withholds_every_refused_known_id_when_the_attestation_record_is_missing_or_unreadable_and_replays_none(tmp_path):
    """With the operation's attestation file MISSING or UNREADABLE at pull time while an `ok` shard
    summary is readable, `attestation_summary` reported present = False and neither branch of the gate fired -- the refused-known ids
    entered `refused_known_ids`, and `reactivate_refused_known` could write a superseded revision's activation bundle to the live record log
    before any refusal or merge barrier, making it current with no verifiable attestation for the pull (the earlier test's clean control was
    exactly that path: it removed the record and then replayed). FAIL-CLOSED now. A non-current known id (s2 reverted: its old revision is
    accepted and superseded) returned by an `ok` shard under (a) no attestation file, (b) a file that is not JSON, (c) a JSON object of
    another schema, (d) a `store_attestation/1` object with no `generations` map: `refused_known_ids` is EMPTY, every id sits in
    `refused_known_withheld` with the reason naming the record's path and its state (missing / unreadable), the adapter is recorded refused
    with the count, the replay is a no-op, ZERO activation bundles are written, the ingest folds nothing and the head is unchanged; the CLI's
    pull stage records the same. (The publications half of the same gate -- an unadmitted adapter's revision withheld before pass 1 -- is
    pinned by part 3's attestation test over a real published generation.)"""
    w = _reverted_s2_world(tmp_path)
    root, ad, old_rev, new_rev, known, wroot, plan = w["root"], w["ad"], w["old_rev"], w["new_rev"], w["known"], w["wroot"], w["plan"]
    assert not os.path.exists(w["attestation"])
    unreadable = f"unreadable (not a {ST.ATTESTATION_SCHEMA} record)"
    cases = [("missing", None), (unreadable, "{not json"), (unreadable, json.dumps({"schema": "something_else/1", "window": "w1", "generations": {}})),
             (unreadable, json.dumps({"schema": ST.ATTESTATION_SCHEMA, "window": "w1", "ok": True}))]
    for i, (state, content) in enumerate(cases, 1):
        if content is None:
            assert not os.path.exists(w["attestation"])
        else:
            open(w["attestation"], "w", encoding="utf-8").write(content)
        expect = str(tmp_path / f"expect{i}")
        rep = CR.pull_window(root, plan, expect, source=str(wroot))
        pulled_record = os.path.join(expect, ST.ATTESTATION_FILE)
        assert os.path.isfile(pulled_record) is (content is not None)
        assert rep["attestation"]["present"] is False and rep["rsync"]["summaries"] == 0, rep["attestation"]
        assert rep["summaries"][ADAPTER] == dict(rep["summaries"][ADAPTER], expected=1, present=1, ok=1, refused_known=3)      # an ok shard, readable
        assert rep["refused_known_ids"] == {ADAPTER: []}, rep["refused_known_ids"]
        assert set(rep["refused_known_withheld"][ADAPTER]) == set(known.values()) and old_rev in rep["refused_known_withheld"][ADAPTER]
        why = rep["refused_known_withheld"][ADAPTER][old_rev]
        assert why.startswith(f"no readable attestation record for this operation ({pulled_record} is {state}): the attestation job wrote none"), why
        assert len(set(rep["refused_known_withheld"][ADAPTER].values())) == 1
        assert why in rep["adapters_refused"][ADAPTER] and any("3 refused-known revision(s) withheld from re-activation: " + why in r for r in rep["adapters_refused"][ADAPTER])
        assert rep["pulled"] == {} and rep["withheld"] == {} and rep["quarantined"] == {}
        with DB.reading(DB.db_path(root)) as conn:
            assert CR.reactivate_refused_known(root, plan, rep, conn=conn, now=synth.t(3500 + i), adapter_of=lambda name: ad) == {}
        assert rep["reactivations"] == {} and RUN.complete_bundles(root, ADAPTER, old_rev) == []
        assert H.ingest_once(root, workers=0, min_free=0).refused == 0 and _s2_head(root) == [(new_rev,)]
        assert set(CR.refused_adapters(plan, rep)) == {ADAPTER} and any(r.startswith("pull: no readable attestation record") for r in CR.refused_adapters(plan, rep)[ADAPTER])
    # the CLI's pull stage over the last shape (the record unreadable): the plan records the withheld ids and the refusal, rc 0 (a refusal
    # is a result), and nothing was replayed
    CR._write_json(CR.plan_path(root, "w1"), plan)
    assert CLI.main(["--root", root, "cluster-reads", "pull", "--window", "w1", "--source", str(wroot), "--expect", str(tmp_path / "expect_cli")]) == 0
    recorded = CR._load_json(CR.plan_path(root, "w1"))
    stage = recorded["stages"]["pull"]
    assert stage["attestation"]["present"] is False and set(stage["refused_known_withheld"][ADAPTER]) == set(known.values()) and stage["reactivations"] == {}
    assert any(r.startswith("pull: no readable attestation record for this operation") for r in recorded["refusals"][ADAPTER])
    assert RUN.complete_bundles(root, ADAPTER, old_rev) == [] and _s2_head(root) == [(new_rev,)]


def test_a_retry_over_the_same_expect_directory_reads_this_attempts_record_only_and_a_failed_summary_transfer_replays_nothing(tmp_path):
    """Pass 0 rsync'd into the standing expect directory without
    --delete, so a retry over the same window saw a previous attempt's attestation.json where the window root no longer had one, and
    `attestation_premise` admitted a record that was absent at THIS pull. Now pass 0 receives into a fresh per-attempt directory installed
    whole (`install_received`): same window, same expect directory, the remote record removed -> present = False, attempt 1's copy sits in
    `<expect>.prev` and not in `<expect>`, every refused-known id is withheld naming the missing record, nothing is replayed, no bundle, the
    head is unchanged; a remote shard summary removed is missing at the retry too (attempt 2's copy does not stand in). And a summary
    transfer that did not return 0 is a WHOLE-PULL barrier: with a clean record present, rc 23 on pass 0 withholds every id with the rc and
    rsync's own words, records `barrier`, and replays nothing. The control over the same directory with everything back collects the ids
    and the replay re-activates the non-current one."""
    w = _reverted_s2_world(tmp_path)
    root, ad, old_rev, new_rev, known, wroot, plan = w["root"], w["ad"], w["old_rev"], w["new_rev"], w["known"], w["wroot"], w["plan"]
    expect = str(tmp_path / "expect_same")
    record = os.path.join(expect, ST.ATTESTATION_FILE)
    clean = _attestation_record("w1", {"synthetic_stores": _attested_generation("deadbeef")})
    CR._write_json(w["attestation"], clean)

    def pull(runner=subprocess.run):
        return CR.pull_window(root, plan, expect, source=str(wroot), runner=runner)

    def replay(rep_):
        with DB.reading(DB.db_path(root)) as conn:
            return CR.reactivate_refused_known(root, plan, rep_, conn=conn, now=synth.t(3600), adapter_of=lambda name: ad)

    # attempt 1, clean: the ids are collected (nothing replayed yet)
    first = pull()
    assert first["attestation"]["present"] and first["refused_known_ids"][ADAPTER] == sorted(known.values()) and first["adapters_refused"] == {}
    assert first["expect_dir"] == expect and os.path.isfile(record) and not os.path.exists(expect + ".prev")
    # attempt 2, same window, same expect directory, the remote record gone: the retry does not see attempt 1's copy
    os.remove(w["attestation"])
    retry = pull()
    assert retry["attestation"]["present"] is False and not os.path.exists(record)
    assert os.path.isfile(os.path.join(expect + ".prev", ST.ATTESTATION_FILE))          # attempt 1's directory, aside
    assert retry["summaries"][ADAPTER]["present"] == 1 and retry["summaries"][ADAPTER]["ok"] == 1 and retry["summaries"][ADAPTER]["refused_known"] == 3
    assert retry["refused_known_ids"] == {ADAPTER: []} and set(retry["refused_known_withheld"][ADAPTER]) == set(known.values())
    assert all(f"({record} is missing)" in why for why in retry["refused_known_withheld"][ADAPTER].values()), retry["refused_known_withheld"]
    assert replay(retry) == {} and RUN.complete_bundles(root, ADAPTER, old_rev) == [] and _s2_head(root) == [(new_rev,)]
    # attempt 3: the record is back, the remote summary is gone: attempt 2's copy of the summary does not stand in for it
    CR._write_json(w["attestation"], clean)
    doc = json.load(open(w["summary_file"]))
    os.remove(w["summary_file"])
    gone = pull()
    assert gone["attestation"]["present"] and gone["summaries"][ADAPTER]["present"] == 0 and any("1 of 1 shard summaries missing" in r for r in gone["adapters_refused"][ADAPTER])
    assert gone["refused_known_ids"] == {ADAPTER: []} and gone["refused_known_withheld"] == {} and replay(gone) == {}
    CR._write_json(w["summary_file"], doc)
    # attempt 4: the summary transfer does not return 0 (a partial transfer: the files land, rsync says 23): the whole-pull barrier
    calls = []
    words = "rsync: [sender] read errors mapping \"/x/1-of-1.json\": Input/output error (5)\nrsync error: some files/attrs were not transferred (see previous errors) (code 23) at main.c(1338)"

    def partial(argv, **kw):
        calls.append(list(argv))
        r = subprocess.run(argv, **kw)
        return subprocess.CompletedProcess(argv, 23, stdout=r.stdout, stderr=words) if "--prune-empty-dirs" in argv else r
    barred = pull(runner=partial)
    assert len(calls) == 1 and barred["rsync"] == {"summaries": 23} and barred["pulled"] == {} and barred["quarantined"] == {}
    assert barred["barrier"] == {"stage": "summaries", "rc": 23, "stderr_tail": " | ".join(words.splitlines())}
    assert barred["attestation"]["present"] and barred["attestation"]["ok"] is True            # the record arrived green -- and still nothing moves
    why = barred["refused_known_withheld"][ADAPTER][old_rev]
    assert why.startswith("the shard summaries and the attestation record could not be pulled whole (rsync rc 23: rsync: [sender] read errors") and why.endswith("; nothing is delivered or replayed this pull"), why
    assert barred["refused_known_ids"] == {ADAPTER: []} and set(barred["refused_known_withheld"][ADAPTER]) == set(known.values()) and why in barred["adapters_refused"][ADAPTER]
    assert replay(barred) == {} and RUN.complete_bundles(root, ADAPTER, old_rev) == [] and _s2_head(root) == [(new_rev,)]
    assert any(r.startswith("pull: the shard summaries and the attestation record could not be pulled whole") for r in CR.refused_adapters(plan, barred)[ADAPTER])
    # the control, same directory: a clean attempt collects every id and the replay re-activates the non-current one
    ok = pull()
    assert ok["barrier"] is None and ok["adapters_refused"] == {} and ok["refused_known_ids"][ADAPTER] == sorted(known.values())
    out = replay(ok)
    assert out[ADAPTER][old_rev]["state"] == "reactivated" and len(RUN.complete_bundles(root, ADAPTER, old_rev)) == 1
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0 and _s2_head(root) == [(old_rev,)]


def test_the_pull_delivers_by_ownership_leaves_an_earlier_windows_inbox_directory_for_the_merge_and_a_failed_summary_transfer_delivers_nothing(tmp_path, monkeypatch):
    """The inbox under `adapters/` is shared across windows.
    Before: pass 1 pulled the whole spool behind excludes, and the inbox loop quarantined every directory of the adapter it found -- an
    earlier window's delivery interrupted before its inbox-merge, valid and marker-bearing, was moved to this window's `_failed/` (operator
    space `scan_inbox` skips), stranded. Now pass 1 pulls exactly the directories THIS window's summaries list (a spool directory no summary
    names is never pulled), and the inbox loop quarantines listed directories only: an unlisted inbox directory is reported `unlisted` and
    left -- under a missing attestation, under a stale adapter, under a clean pull alike -- and inbox-merge takes it. And a summary transfer
    that did not return 0 delivers nothing: no pass 1, no pass 2, the listed revision withheld with the rc and rsync's words, the adapter
    refused with that one reason, the inbox untouched; the clean retry over the same expect directory delivers the revision complete."""
    w, plan, rev_dir, rev = _cluster_spool(tmp_path, monkeypatch, with_bundle=False)
    expect = str(tmp_path / "expect")
    inbox = os.path.join(H.inbox_dir(w["root"]), CR.INBOX_NAME)
    left = "no shard summary of this window lists it; left in the inbox for inbox-merge (an earlier window's delivery, or a summary that never arrived)"
    # an earlier window's delivery, interrupted before its inbox-merge: a COMPLETE directory (its marker present) of the adapter that no
    # summary of THIS window lists -- and beside it an earlier window's PARTIAL one (its pass 1 landed, its pass 2 never ran: no marker)
    earlier = f"{ADAPTER}/{'e' * 16}"
    os.makedirs(os.path.join(inbox, "adapters", earlier))
    open(os.path.join(inbox, "adapters", earlier, "000001-batch.jsonl"), "w").write("{}\n")
    open(os.path.join(inbox, "adapters", earlier, "000002-source_revision.jsonl"), "w").write("{}\n")
    part_dir = f"{ADAPTER}/{'d' * 16}"
    os.makedirs(os.path.join(inbox, "adapters", part_dir))
    open(os.path.join(inbox, "adapters", part_dir, "000001-batch.jsonl"), "w").write("{}\n")
    unlisted_both = {earlier: left, part_dir: left}
    # a spool directory no summary lists: never pulled
    stray = os.path.join(plan["root"], "records", "adapters", ADAPTER, "f" * 16)
    os.makedirs(stray)
    open(os.path.join(stray, "000001-batch.jsonl"), "w").write("{}\n")
    calls = []

    def rec(argv, **kw):
        calls.append(list(argv))
        return subprocess.run(argv, **kw)

    # (a) the finding's trigger, a missing attestation: the listed revision is withheld, the earlier window's directory is left as it is
    att_doc = json.load(open(plan["attestation"]))
    os.remove(plan["attestation"])
    rep = CR.pull_window(w["root"], plan, expect, source=plan["window_root"], runner=rec)
    assert rep["attestation"]["present"] is False and rep["withheld"][rev_dir].startswith("no readable attestation record") and rep["pulled"] == {}
    assert rep["quarantined"] == {} and rep["unlisted"] == unlisted_both and rep["rsync"] == {"summaries": 0, "pass1": 0} and len(calls) == 1
    assert os.path.isfile(os.path.join(inbox, "adapters", earlier, "000001-batch.jsonl")) and not os.path.exists(os.path.join(inbox, CR.FAILED_DIR))
    assert os.path.isfile(os.path.join(inbox, "adapters", part_dir, "000001-batch.jsonl"))
    # (b) a stale adapter: the same
    CR._write_json(plan["attestation"], att_doc)
    calls.clear()
    stale = CR.pull_window(w["root"], dict(plan, stages={"revalidate_post": {"stale": [ADAPTER]}}), expect, source=plan["window_root"], runner=rec)
    assert stale["withheld"] == {rev_dir: "post-run revalidation refused the adapter"} and stale["unlisted"] == unlisted_both and stale["quarantined"] == {}
    assert len(calls) == 1 and os.path.isdir(os.path.join(inbox, "adapters", earlier)) and not os.path.exists(os.path.join(inbox, CR.STALE_DIR))
    # (c) the summary transfer does not return 0: the whole-pull barrier -- nothing pulled, the listed revision withheld with the rc and rsync's
    # words, the adapter refused with that one reason, the earlier window's directory untouched
    words = "file has vanished: \"synthetic_store/1-of-2.json\"\nrsync warning: some files vanished before they could be transferred (code 24) at main.c(1338)"

    def partial(argv, **kw):
        calls.append(list(argv))
        r = subprocess.run(argv, **kw)
        return subprocess.CompletedProcess(argv, 24, stdout=r.stdout, stderr=words) if "--prune-empty-dirs" in argv else r
    calls.clear()
    barred = CR.pull_window(w["root"], plan, expect, source=plan["window_root"], runner=partial)
    assert len(calls) == 1 and barred["rsync"] == {"summaries": 24} and barred["barrier"] == {"stage": "summaries", "rc": 24, "stderr_tail": " | ".join(words.splitlines())}
    assert barred["pulled"] == {} and barred["quarantined"] == {} and barred["unlisted"] == {} and set(barred["withheld"]) == {rev_dir}
    assert barred["withheld"][rev_dir].startswith("the shard summaries and the attestation record could not be pulled whole (rsync rc 24: file has vanished")
    assert barred["adapters_refused"] == {ADAPTER: [barred["withheld"][rev_dir]]} and set(CR.refused_adapters(plan, barred)) == {ADAPTER}
    assert not os.path.exists(os.path.join(inbox, "adapters", rev_dir)) and os.path.isdir(os.path.join(inbox, "adapters", earlier))
    # (d) the clean retry over the same expect directory: pass 1 is a --files-from of exactly the listed directory (the stray spool directory
    # is never pulled), the revision is delivered complete, the earlier window's directory is still there and inbox-merge takes it
    calls.clear()
    ok = CR.pull_window(w["root"], plan, expect, source=plan["window_root"], runner=rec)
    assert ok["barrier"] is None and ok["adapters_refused"] == {} and set(ok["pulled"]) == {rev_dir} and ok["pulled"][rev_dir]["state"] == "complete"
    assert ok["rsync"] == {"summaries": 0, "pass1": 0, "pass2": 0} and len(calls) == 3 and ok["unlisted"] == unlisted_both
    p1 = next(c for c in calls if any(a.startswith("--exclude=*source_revision*") for a in c))
    assert any(a.startswith("--files-from=") for a in p1) and not os.path.exists(os.path.join(inbox, "adapters", ADAPTER, "f" * 16))
    assert any("source_revision" in fn for fn in os.listdir(os.path.join(inbox, "adapters", rev_dir)))
    # inbox-merge: the complete earlier directory merges; the marker-less partial is "incomplete" -- left in the inbox, never a marker-less
    # records/ directory that `published_dirs_of` would count as a known revision
    moved = H.scan_inbox(w["root"])
    assert any(dst.endswith(os.path.join(earlier, "000001-batch.jsonl")) and status == "merged" for dst, status in moved), moved
    assert any(dst.endswith(os.path.join(earlier, "000002-source_revision.jsonl")) and status == "merged" for dst, status in moved), moved
    assert [(dst, s) for dst, s in moved if s == "incomplete"] == [(os.path.join(inbox, "adapters", part_dir), "incomplete")], moved
    assert os.path.isfile(os.path.join(H.records_dir(w["root"]), "adapters", earlier, "000001-batch.jsonl")) and not os.path.exists(os.path.join(inbox, "adapters", earlier))
    assert os.path.isfile(os.path.join(inbox, "adapters", part_dir, "000001-batch.jsonl")) and not os.path.exists(os.path.join(H.records_dir(w["root"]), "adapters", part_dir))
    assert os.path.isdir(os.path.join(H.records_dir(w["root"]), "adapters", rev_dir))


def _window_a_delivered_but_not_merged(tmp_path, monkeypatch):
    """Window A over a real published generation: s2 changed, the shards publish ONE revision of it, the pull delivers it COMPLETE (marker
    present) into the shared inbox -- and A is interrupted before its inbox-merge. Returns the world, A's plan, the revision directory
    (`<adapter>/<rev16>`) and the inbox root."""
    w = _world_with_generation(tmp_path, monkeypatch)
    _store(w["stores"], "s2", "legacy_store_2", "none", [0.1, 0.9, 0.5])
    assert ST.mirror_component(w["root"], w["dest"], w["cfg"].components["synthetic_stores"], runner=TS.Runner()).outcome == "published"
    plan0, plan_file0, _ = _plan_and_stage(w, tmp_path, window="w0")
    _node(w)
    for i in range(2):
        assert CR.run_shard(f"{plan0['window_root']}/plan.json", i, home=w["home"]) == 0
    _back_on_grading_host(w)
    assert CR.revalidate_plan(w["root"], plan0, w["cfg"]) == []
    rep0 = CR.pull_window(w["root"], plan0, str(tmp_path / "expect0"), source=plan0["window_root"])
    (rev_a,) = [r for r, v in rep0["pulled"].items() if v["state"] == "complete"]
    inbox = os.path.join(H.inbox_dir(w["root"]), CR.INBOX_NAME)
    assert any(CR.MARKER_TOKEN in fn for fn in os.listdir(os.path.join(inbox, "adapters", rev_a)))
    assert not os.path.exists(os.path.join(H.records_dir(w["root"]), "adapters", rev_a))
    return w, plan0, rev_a, inbox


def test_the_cli_sequence_revalidate_pull_inbox_merge_leaves_an_earlier_windows_delivery_for_the_merge_under_a_stale_adapter(tmp_path, monkeypatch):
    """Window A delivered a complete revision into the shared inbox and was interrupted before its inbox-merge;
    window B reads the same adapter and its post-run revalidation marks the adapter STALE. Before: `cluster-reads revalidate` called
    `quarantine_stale` BEFORE the pull and moved the adapter's WHOLE inbox subtree -- A's valid, marker-bearing directory included -- under
    `_stale/<B>/`, which `scan_inbox` skips: A's result was stranded. Now, through the REAL CLI sequence `revalidate` -> `pull` ->
    `inbox-merge`: the revalidation records the stale adapter and moves nothing; the pull withholds B's listed revision (never pulled),
    reports A's directory `unlisted` and quarantines nothing; inbox-merge merges A's delivery into records/ and the ingest folds it --
    accepted once, B's revision nowhere, nothing under `_stale/` or `_failed/`."""
    w, plan0, rev_a, inbox = _window_a_delivered_but_not_merged(tmp_path, monkeypatch)
    root = w["root"]
    # window B: s2 changes again (a new generation, a new plan); its shards publish a NEW revision of s2 -- rev_a's id is in neither
    # records/ nor the sidecar (it sits in the inbox), and B's s2 differs anyway
    _store(w["stores"], "s2", "legacy_store_2", "none", [0.1, 0.9, 0.5, 0.3])
    assert ST.mirror_component(root, w["dest"], w["cfg"].components["synthetic_stores"], runner=TS.Runner()).outcome == "published"
    plan1, plan_file1, _ = _plan_and_stage(w, tmp_path, window="w1")
    _node(w)
    for i in range(2):
        assert CR.run_shard(f"{plan1['window_root']}/plan.json", i, home=w["home"]) == 0
    _back_on_grading_host(w)
    expected1, _per = CR.expected_from_summaries(plan1["summaries_dir"])
    (rev_b,) = expected1
    assert rev_b != rev_a
    # adapter A goes stale after B's read (the canonical copy moved under the read)
    _same_size_edit(os.path.join(w["stores"], "s1", "store.json"))
    # the real CLI sequence
    assert CLI.main(["--root", root, "cluster-reads", "revalidate", "--window", "w1"]) == 0
    on_disk = CR._load_json(plan_file1)
    assert on_disk["stages"]["revalidate_post"]["stale"] == [ADAPTER] and "quarantined" not in on_disk["stages"]["revalidate_post"]
    assert os.path.isdir(os.path.join(inbox, "adapters", rev_a)) and not os.path.exists(os.path.join(inbox, CR.STALE_DIR))   # nothing moved before the pull
    assert CLI.main(["--root", root, "cluster-reads", "pull", "--window", "w1", "--source", plan1["window_root"], "--expect", str(tmp_path / "expect1")]) == 0
    on_disk = CR._load_json(plan_file1)
    stage = on_disk["stages"]["pull"]
    assert stage["unlisted"] == [rev_a] and stage["quarantined"] == 0 and stage["pulled"] == 0 and stage["withheld"] == 1 and stage["barrier"] is None
    assert any(r.startswith("pull: post-run revalidation refused a component") for r in on_disk["refusals"][ADAPTER])
    assert os.path.isdir(os.path.join(inbox, "adapters", rev_a)) and not os.path.exists(os.path.join(inbox, "adapters", rev_b))
    assert not os.path.exists(os.path.join(inbox, CR.STALE_DIR)) and not os.path.exists(os.path.join(inbox, CR.FAILED_DIR))
    assert CLI.main(["--root", root, "inbox-merge"]) == 0
    assert os.path.isdir(os.path.join(H.records_dir(root), "adapters", rev_a)) and not os.path.exists(os.path.join(inbox, "adapters", rev_a))
    ing = H.ingest_once(root, workers=0, min_free=0)
    assert ing.refused == 0 and ing.blocked == 0, (ing.refusals, ing.blocked)
    assert _rows(root, "SELECT COUNT(*), SUM(accepted) FROM source_revision WHERE revision_id = ?", (rev_a.split("/")[1],)) == [(1, 1)]
    assert _rows(root, "SELECT COUNT(*) FROM source_revision WHERE revision_id = ?", (rev_b.split("/")[1],)) == [(0,)]
    assert not os.path.exists(os.path.join(H.records_dir(root), "adapters", rev_b))


def test_the_window_merges_the_standing_inbox_before_the_export_so_a_shard_refuses_an_earlier_windows_id_as_known_and_it_folds_once(tmp_path, monkeypatch):
    """The structural half. `reader_state.published_dirs_of` reads records/, not the inbox, so after window A's
    complete delivery sat unmerged, window B's export did not know rev_a, B's shard minted the SAME id again and LISTED it, and the pull's
    batch-list comparison (A's marker in `have`) quarantined A's valid publication under `_failed/<B>/`. window.sh now runs `inbox-merge`
    over the STANDING inbox before the reader-state export (`-- cluster reads inbox-merge`): rev_a is in records/ when the sidecar is
    exported, B's shard refuses the id as KNOWN and publishes nothing for it, the pull collects it as refused-known, `reactivate_by_id`
    sees a committed, not-yet-folded revision (held, nothing written), and the window's one ingest folds it: one `source_revision` row,
    accepted, no duplicate directory, nothing quarantined."""
    w, plan0, rev_a, inbox = _window_a_delivered_but_not_merged(tmp_path, monkeypatch)
    root, rev_a_id = w["root"], rev_a.split("/")[1]
    # window B, in the window's order: the standing inbox merges FIRST
    assert CLI.main(["--root", root, "inbox-merge"]) == 0
    assert os.path.isdir(os.path.join(H.records_dir(root), "adapters", rev_a)) and not os.path.exists(os.path.join(inbox, "adapters", rev_a))
    assert (ADAPTER, "revision", rev_a_id) in RS.published_dirs_of(root, [ADAPTER])
    plan1, plan_file1, _ = _plan_and_stage(w, tmp_path, window="w1")           # the export now holds rev_a: the shards know it
    _node(w)
    for i in range(2):
        assert CR.run_shard(f"{plan1['window_root']}/plan.json", i, home=w["home"]) == 0
    _back_on_grading_host(w)
    expected1, per1 = CR.expected_from_summaries(plan1["summaries_dir"])
    assert expected1 == {}                                                      # nothing published: every group's id is known
    assert any(x["revision_id"] == rev_a_id for d in per1[ADAPTER] for x in (d.get("refused_known") or []))
    assert CR.revalidate_plan(root, plan1, w["cfg"]) == []
    rep1 = CR.pull_window(root, plan1, str(tmp_path / "expect1"), source=plan1["window_root"])
    assert rep1["pulled"] == {} and rep1["quarantined"] == {} and rep1["unlisted"] == {} and rep1["withheld"] == {} and rev_a_id in rep1["refused_known_ids"][ADAPTER]
    with DB.reading(DB.db_path(root)) as conn:
        out = CR.reactivate_refused_known(root, plan1, rep1, conn=conn, now=synth.t(3600), adapter_of=lambda name: w["ad"])
    assert out[ADAPTER][rev_a_id]["state"] == "held" and "not folded yet" in out[ADAPTER][rev_a_id]["error"] and out[ADAPTER][rev_a_id]["files"] == 0
    assert rep1["adapters_refused"] == {}
    assert CLI.main(["--root", root, "inbox-merge"]) == 0                       # the post-pull merge: nothing left to merge
    ing = H.ingest_once(root, workers=0, min_free=0)
    assert ing.refused == 0 and ing.blocked == 0, (ing.refusals, ing.blocked)
    assert _rows(root, "SELECT COUNT(*), SUM(accepted) FROM source_revision WHERE revision_id = ?", (rev_a_id,)) == [(1, 1)]
    rev_root = os.path.join(H.records_dir(root), "adapters", rev_a)
    assert not any(".conflict" in fn for fn in os.listdir(rev_root)) and sum(1 for d in os.listdir(os.path.dirname(rev_root)) if d == rev_a_id) == 1
    assert not os.path.exists(os.path.join(inbox, CR.FAILED_DIR)) and not os.path.exists(os.path.join(inbox, CR.STALE_DIR))


def test_a_listed_revision_already_complete_in_the_inbox_is_an_earlier_delivery_reported_complete_never_re_pulled_or_quarantined(tmp_path, monkeypatch):
    """The pull's own defense -- for a window whose standing inbox was NOT merged before its export (the
    older window.sh, or a pull called on its own): window B's shard reissues rev_a's id and lists it; at the pull the inbox already holds
    the directory WITH its marker, which this window's pass 2 cannot have put there -- an earlier completed delivery of this very revision.
    It is reported complete as it stands, excluded from pass 1 (B's bytes are never mixed into it), never compared against B's batch list
    (the marker would read as an extra), never quarantined; inbox-merge takes it and the ingest folds it once, A's bytes intact."""
    w, plan0, rev_a, inbox = _window_a_delivered_but_not_merged(tmp_path, monkeypatch)
    root, rev_a_id = w["root"], rev_a.split("/")[1]
    before = CR._listing(os.path.join(inbox, "adapters", rev_a))
    plan1, plan_file1, _ = _plan_and_stage(w, tmp_path, window="w1")           # no merge: the export reads records/ only -> rev_a unknown
    _node(w)
    for i in range(2):
        assert CR.run_shard(f"{plan1['window_root']}/plan.json", i, home=w["home"]) == 0
    _back_on_grading_host(w)
    expected1, _per = CR.expected_from_summaries(plan1["summaries_dir"])
    assert set(expected1) == {rev_a}                                            # the same id, published again by B's shard
    assert CR.revalidate_plan(root, plan1, w["cfg"]) == []
    calls = []

    def rec(argv, **kw):
        calls.append(list(argv))
        return subprocess.run(argv, **kw)
    rep1 = CR.pull_window(root, plan1, str(tmp_path / "expect1"), source=plan1["window_root"], runner=rec)
    assert rep1["pulled"] == {rev_a: {"state": "complete (an earlier delivery already in the inbox)", "adapter": ADAPTER, "files": len(before)}}, rep1["pulled"]
    assert rep1["quarantined"] == {} and rep1["withheld"] == {} and rep1["unlisted"] == {} and rep1["adapters_refused"] == {}
    assert rep1["rsync"] == {"summaries": 0, "pass1": 0} and len(calls) == 1          # no pass 1, no pass 2: nothing of B's was pulled
    assert CR._listing(os.path.join(inbox, "adapters", rev_a)) == before              # A's bytes, untouched
    assert not os.path.exists(os.path.join(inbox, CR.FAILED_DIR))
    moved = H.scan_inbox(root)
    assert all(s == "merged" for _d, s in moved) and len(moved) == len(before)
    ing = H.ingest_once(root, workers=0, min_free=0)
    assert ing.refused == 0 and ing.blocked == 0, (ing.refusals, ing.blocked)
    assert _rows(root, "SELECT COUNT(*), SUM(accepted) FROM source_revision WHERE revision_id = ?", (rev_a_id,)) == [(1, 1)]
    assert CR._listing(os.path.join(H.records_dir(root), "adapters", rev_a)) == before


def test_an_earlier_windows_marker_less_partial_never_stages_another_windows_revision_and_is_set_aside(tmp_path, monkeypatch):
    """Window A's pull was interrupted after pass 1 left its batch files in the shared inbox and before its marker
    arrived; the standing merge rightly leaves the marker-less partial ("incomplete"); window B's export cannot see it, so B's shard
    legitimately publishes the SAME revision id (same inputs, same emission version) with DIFFERENT bytes under the same names and sizes
    (B's written_utc / observed_utc; possibly other facts). Before: B's pass 1 `--ignore-existing` into that very directory kept A's
    batches and fetched nothing, pass 2 brought B's marker over them, the size-only batch-list check passed, and a mixed revision -- A's
    measurements under B's attested provenance -- folded. Now passes 1 and 2 receive B's whole revision into an attempt-local directory,
    verify every file by size AND sha256 against B's summary (`shard_summary/2`), set A's partial aside under `_partial/<B>/` (never
    deleted) and install B's revision by one rename: the inbox holds B's bytes only, inbox-merge merges them, the ingest folds exactly once,
    and A's bytes sit under `_partial/` and nowhere in records/."""
    w, plan0, rev_a, inbox = _window_a_delivered_but_not_merged(tmp_path, monkeypatch)
    root, rev_a_id = w["root"], rev_a.split("/")[1]
    d_inbox = os.path.join(inbox, "adapters", rev_a)
    # A's interruption: its marker never arrived, and its batches are A's bytes -- same names, same sizes, different content
    for fn in os.listdir(d_inbox):
        if CR.MARKER_TOKEN in fn:
            os.remove(os.path.join(d_inbox, fn))
        else:
            _same_size_edit(os.path.join(d_inbox, fn))
    a_shas = {fn: R.file_sha256(os.path.join(d_inbox, fn)) for fn in os.listdir(d_inbox)}
    assert a_shas and not any(CR.MARKER_TOKEN in fn for fn in a_shas)
    # window B: the standing merge leaves the partial; the export cannot see it; B's shards reissue rev_a's id
    assert [s for _d, s in H.scan_inbox(root)] == ["incomplete"] and os.path.isdir(d_inbox)
    plan1, plan_file1, _ = _plan_and_stage(w, tmp_path, window="w1")
    _node(w)
    for i in range(2):
        assert CR.run_shard(f"{plan1['window_root']}/plan.json", i, home=w["home"]) == 0
    _back_on_grading_host(w)
    expected1, _per = CR.expected_from_summaries(plan1["summaries_dir"])
    assert set(expected1) == {rev_a}
    b_spool = os.path.join(plan1["root"], "records", "adapters", rev_a)
    b_shas = {fn: R.file_sha256(os.path.join(b_spool, fn)) for fn in os.listdir(b_spool)}
    assert expected1[rev_a]["sha256"] == b_shas and set(expected1[rev_a]["files"]) == set(b_shas)       # the summary's digests ARE B's bytes
    # same names; the same sizes for the plain .jsonl batches (fixed-width stamps, so byte lengths are normally unchanged); a .gz
    # batch may differ by a few bytes between windows (compressed lengths are not stable across differing bytes); different bytes throughout
    assert set(b_shas) - set(a_shas) == {fn for fn in b_shas if CR.MARKER_TOKEN in fn}
    plain = [fn for fn in a_shas if not fn.endswith(".gz")]
    assert plain and {fn: os.path.getsize(os.path.join(b_spool, fn)) for fn in plain} == {fn: os.path.getsize(os.path.join(d_inbox, fn)) for fn in plain}
    assert all(b_shas[fn] != sha for fn, sha in a_shas.items())                                         # different bytes
    assert CR.revalidate_plan(root, plan1, w["cfg"]) == []
    rep1 = CR.pull_window(root, plan1, str(tmp_path / "expect1"), source=plan1["window_root"])
    assert rep1["pulled"] == {rev_a: {"state": "complete", "adapter": ADAPTER, "files": len(b_shas)}} and rep1["quarantined"] == {} and rep1["adapters_refused"] == {}
    assert set(rep1["partial"]) == {rev_a} and rep1["partial_aside"][rev_a].startswith(os.path.join(inbox, CR.PARTIAL_DIR, "w1"))
    aside = rep1["partial_aside"][rev_a]
    assert {fn: R.file_sha256(os.path.join(aside, fn)) for fn in os.listdir(aside)} == a_shas            # A's bytes, aside, intact
    assert {fn: R.file_sha256(os.path.join(d_inbox, fn)) for fn in os.listdir(d_inbox)} == b_shas         # the inbox holds B's bytes only
    assert not [d for d in os.listdir(inbox) if d.startswith(".recv-")]
    moved = H.scan_inbox(root)
    assert {s for _d, s in moved} == {"merged"} and len(moved) == len(b_shas)
    ing = H.ingest_once(root, workers=0, min_free=0)
    assert ing.refused == 0 and ing.blocked == 0, (ing.refusals, ing.blocked)
    assert _rows(root, "SELECT COUNT(*), SUM(accepted) FROM source_revision WHERE revision_id = ?", (rev_a_id,)) == [(1, 1)]
    rec_root = os.path.join(H.records_dir(root), "adapters", rev_a)
    assert {fn: R.file_sha256(os.path.join(rec_root, fn)) for fn in os.listdir(rec_root)} == b_shas
    everywhere = {R.file_sha256(os.path.join(dp, fn)) for dp, _d, fns in os.walk(H.records_dir(root)) for fn in fns}
    assert not (set(a_shas.values()) & everywhere) and os.path.isdir(aside)                               # A's bytes nowhere in records/, never deleted


def test_a_sha256_mismatch_against_the_shard_summary_quarantines_the_revision_and_nothing_enters_the_inbox(tmp_path, monkeypatch):
    """The pull verifies every received file by size AND sha256 against the shard summary (`shard_summary/2`) in the
    attempt-local receive directory. A batch file whose bytes changed under the same size after the shard summarized it (or in transit)
    fails the check: the revision is quarantined under `_failed/<window>/` from the receive directory, nothing of it enters the inbox, the
    adapter is recorded refused, inbox-merge has nothing, the ingest folds nothing. A summary stripped of its digests admits nothing
    either; with the digests back the revision travels complete."""
    w, plan, rev_dir, rev = _cluster_spool(tmp_path, monkeypatch, with_bundle=False)
    inbox = os.path.join(H.inbox_dir(w["root"]), CR.INBOX_NAME)
    src = os.path.join(plan["root"], "records", "adapters", rev_dir)
    victim = next(fn for fn in sorted(os.listdir(src)) if CR.MARKER_TOKEN not in fn)
    original = _same_size_edit(os.path.join(src, victim))
    rep = CR.pull_window(w["root"], plan, str(tmp_path / "expect"), source=plan["window_root"])
    assert set(rep["quarantined"]) == {rev_dir} and rep["pulled"] == {} and "pass2" not in rep["rsync"] and rep["rsync"]["pass1"] == 0
    q = rep["quarantined"][rev_dir]
    assert q["why"].startswith(f"sha256 mismatch against the shard summary: ['{victim}']") and q["moved_to"].startswith(os.path.join(inbox, CR.FAILED_DIR, "w1"))
    assert os.path.isfile(os.path.join(q["moved_to"], victim)) and not any(CR.MARKER_TOKEN in fn for fn in os.listdir(q["moved_to"]))
    assert not os.path.exists(os.path.join(inbox, "adapters", rev_dir)) and not [d for d in os.listdir(inbox) if d.startswith(".recv-")]
    assert ADAPTER in rep["adapters_refused"] and H.scan_inbox(w["root"]) == [] and H.ingest_once(w["root"], workers=0, min_free=0).ok == 0
    assert _rows(w["root"], "SELECT COUNT(*) FROM source_revision WHERE revision_id = ?", (rev,)) == [(0,)]
    # the bytes restored, but a summary stripped of its digests: nothing is admitted on name and size alone
    open(os.path.join(src, victim), "wb").write(original)
    expected, _per = CR.expected_from_summaries(plan["summaries_dir"])
    spath = expected[rev_dir]["summary"]
    intact = json.load(open(spath))
    stripped = json.loads(json.dumps(intact))
    stripped["published"][rev_dir].pop("sha256")
    CR._write_json(spath, stripped)
    shutil.rmtree(os.path.join(inbox, CR.FAILED_DIR))
    rep2 = CR.pull_window(w["root"], plan, str(tmp_path / "expect2"), source=plan["window_root"])
    assert set(rep2["quarantined"]) == {rev_dir} and "carries no sha256 for" in rep2["quarantined"][rev_dir]["why"] and rep2["pulled"] == {}
    assert not os.path.exists(os.path.join(inbox, "adapters", rev_dir))
    # the digests back: the revision travels complete, byte for byte what the shard published
    CR._write_json(spath, intact)
    rep3 = CR.pull_window(w["root"], plan, str(tmp_path / "expect3"), source=plan["window_root"])
    assert rep3["pulled"] == {rev_dir: {"state": "complete", "adapter": ADAPTER, "files": len(expected[rev_dir]["files"])}} and rep3["quarantined"] == {}
    landed = os.path.join(inbox, "adapters", rev_dir)
    assert {fn: R.file_sha256(os.path.join(landed, fn)) for fn in os.listdir(landed)} == expected[rev_dir]["sha256"]


@contextlib.contextmanager
def _dying_install_rename():
    """Make the install's rename die the way a killed process would: an exception that is not an OSError escapes `os.rename` for a source
    inside a receive directory and a destination under `adapters/` (nothing else -- the expect directory's install, the set-aside move, the
    quarantines -- is touched). `install_whole` restores a set-aside partial only on an OSError, so this leaves exactly what a crash leaves.
    Restores `os.rename` alone on exit (a `monkeypatch.undo()` would also undo the world's HOME and `load_all` patches)."""
    real = os.rename

    def dying(src, dst, *a, **k):
        if CR.RECV_PREFIX in str(src) and f"{os.sep}adapters{os.sep}" in str(dst):
            raise RuntimeError("simulated crash between the verification and the install rename")
        return real(src, dst, *a, **k)
    os.rename = dying
    try:
        yield
    finally:
        os.rename = real


def _receive_dirs(inbox: str) -> List[str]:
    return sorted(d for d in os.listdir(inbox) if d.startswith(CR.RECV_PREFIX))


def test_a_pull_that_dies_after_the_marker_is_finished_by_the_next_windows_recovery_and_a_changed_input_then_publishes_a_new_id(tmp_path, monkeypatch):
    """The pull receives into `inbox/.recv-<window>-…/`, which scan_inbox skips, the reader-state export cannot see and
    no later pull opens: a pull that died after pass 2 (the marker received, the whole set verified) and before the install rename left a
    complete revision there for good -- no restart and no window recovered or named it. Now the pull writes a RECEIVING manifest at pass-1
    start, and the next window's `-- cluster reads inbox-merge` stage runs `cluster-reads recover` BEFORE the standing merge: the copy is
    re-verified against the manifest (names, sizes, sha256, marker) and that window's verdicts, installed by the same rename, the receive
    directory removed; inbox-merge merges it and the ingest folds it once. Then the inputs change: the next read publishes a NEW id, which
    folds beside the recovered one with the newer current."""
    w, plan, rev_dir, rev = _cluster_spool(tmp_path, monkeypatch, with_bundle=False)
    root = w["root"]
    inbox = os.path.join(H.inbox_dir(root), CR.INBOX_NAME)
    expected, _per = CR.expected_from_summaries(plan["summaries_dir"])
    promise = expected[rev_dir]
    with _dying_install_rename(), pytest.raises(RuntimeError, match="simulated crash"):
        CR.pull_window(root, plan, str(tmp_path / "expect"), source=plan["window_root"])
    # what the crash left: nothing in adapters/, one receive directory holding the complete, verified copy and its manifest
    assert not os.path.exists(os.path.join(inbox, "adapters", rev_dir))
    (recv,) = _receive_dirs(inbox)
    assert recv.startswith(CR.RECV_PREFIX + "w1-")
    manifest = json.load(open(os.path.join(inbox, recv, CR.RECEIVING_FILE)))
    assert manifest["schema"] == CR.RECEIVING_SCHEMA and manifest["window"] == "w1" and manifest["expect_dir"] == str(tmp_path / "expect")
    assert manifest["attestation"] == os.path.join(str(tmp_path / "expect"), ST.ATTESTATION_FILE) and manifest["revalidate_post"] == {"stale": []}
    assert manifest["attestation_void"] == {} and set(manifest["pins"]) == {ADAPTER} and manifest["pins"][ADAPTER] == plan["adapters"][ADAPTER]["pins"]
    assert manifest["revisions"] == {rev_dir: {"adapter": ADAPTER, "files": promise["files"], "sha256": promise["sha256"], "marker": promise["marker"], "kind": "revision"}}
    copy = os.path.join(inbox, recv, rev_dir)
    assert {fn: R.file_sha256(os.path.join(copy, fn)) for fn in os.listdir(copy)} == promise["sha256"]        # complete, marker included
    # a standing merge alone would never see it (the dot-prefixed directory is skipped), and the export reads records/ only
    assert H.scan_inbox(root) == [] and (ADAPTER, "revision", rev) not in RS.published_dirs_of(root, [ADAPTER])
    # the next window's first stage: recovery (the CLI the window runs), then the standing merge
    assert CLI.main(["--root", root, "cluster-reads", "recover", "--window", "w2"]) == 0
    assert _receive_dirs(inbox) == [] and not os.path.exists(os.path.join(inbox, CR.FAILED_DIR))
    landed = os.path.join(inbox, "adapters", rev_dir)
    assert {fn: R.file_sha256(os.path.join(landed, fn)) for fn in os.listdir(landed)} == promise["sha256"]
    moved = H.scan_inbox(root)
    assert {s for _d, s in moved} == {"merged"} and len(moved) == len(promise["files"])
    assert (ADAPTER, "revision", rev) in RS.published_dirs_of(root, [ADAPTER])
    ing = H.ingest_once(root, workers=0, min_free=0)
    assert ing.refused == 0 and ing.blocked == 0, (ing.refusals, ing.blocked)
    assert _rows(root, "SELECT COUNT(*), SUM(accepted) FROM source_revision WHERE revision_id = ?", (rev,)) == [(1, 1)]
    # the input changes before the next read: the recovered revision stands, the new read publishes a NEW id, both fold, the newer is current
    _store(w["stores"], "s2", "legacy_store_2", "none", [0.1, 0.9, 0.5, 0.3])
    assert ST.mirror_component(root, w["dest"], w["cfg"].components["synthetic_stores"], runner=TS.Runner()).outcome == "published"
    plan2, plan_file2, _ = _plan_and_stage(w, tmp_path, window="w2")
    _node(w)
    for i in range(2):
        assert CR.run_shard(f"{plan2['window_root']}/plan.json", i, home=w["home"]) == 0
    _back_on_grading_host(w)
    expected2, _per2 = CR.expected_from_summaries(plan2["summaries_dir"])
    (rev_dir2,) = expected2
    rev2 = rev_dir2.split("/")[1]
    assert rev2 != rev
    assert CR.revalidate_plan(root, plan2, w["cfg"]) == []
    rep2 = CR.pull_window(root, plan2, str(tmp_path / "expect2"), source=plan2["window_root"])
    assert set(rep2["pulled"]) == {rev_dir2} and rep2["pulled"][rev_dir2]["state"] == "complete" and _receive_dirs(inbox) == []
    assert {s for _d, s in H.scan_inbox(root)} == {"merged"}
    ing2 = H.ingest_once(root, workers=0, min_free=0)
    assert ing2.refused == 0 and ing2.blocked == 0, (ing2.refusals, ing2.blocked)
    assert _rows(root, "SELECT COUNT(*), SUM(accepted) FROM source_revision WHERE revision_id = ?", (rev2,)) == [(1, 1)]
    assert _rows(root, "SELECT COUNT(*) FROM source_revision WHERE revision_id = ?", (rev,)) == [(1,)]
    assert _s2_head(root) == [(rev2,)]


def test_a_pull_that_dies_between_the_set_aside_and_the_rename_is_finished_by_recovery_and_the_partial_stays_aside(tmp_path, monkeypatch):
    """The install is two moves -- the earlier window's partial aside, then the rename -- and a process can die between them
    (the pull puts the partial back only when the rename FAILS with an OSError; a kill leaves the gap open). Recovery closes it from the
    manifest: the complete copy is still in the receive directory, the partial under `_partial/<B>/`; the copy installs, the partial stays
    aside, the merge folds B's bytes once."""
    w, plan0, rev_a, inbox = _window_a_delivered_but_not_merged(tmp_path, monkeypatch)
    root, rev_a_id = w["root"], rev_a.split("/")[1]
    d_inbox = os.path.join(inbox, "adapters", rev_a)
    for fn in os.listdir(d_inbox):                       # A's interruption: a marker-less partial
        if CR.MARKER_TOKEN in fn:
            os.remove(os.path.join(d_inbox, fn))
    a_files = sorted(os.listdir(d_inbox))
    plan1, plan_file1, _ = _plan_and_stage(w, tmp_path, window="w1")
    _node(w)
    for i in range(2):
        assert CR.run_shard(f"{plan1['window_root']}/plan.json", i, home=w["home"]) == 0
    _back_on_grading_host(w)
    expected1, _per = CR.expected_from_summaries(plan1["summaries_dir"])
    assert set(expected1) == {rev_a}
    assert CR.revalidate_plan(root, plan1, w["cfg"]) == []
    with _dying_install_rename(), pytest.raises(RuntimeError, match="simulated crash"):
        CR.pull_window(root, plan1, str(tmp_path / "expect1"), source=plan1["window_root"])
    # the gap: the partial is aside, the destination is empty, the complete copy sits in the receive directory
    aside = os.path.join(inbox, CR.PARTIAL_DIR, "w1", rev_a)
    assert sorted(os.listdir(aside)) == a_files and not os.path.lexists(d_inbox)
    (recv,) = _receive_dirs(inbox)
    assert any(CR.MARKER_TOKEN in fn for fn in os.listdir(os.path.join(inbox, recv, rev_a)))
    # the next window's recovery: installed by the same rename, nothing to set aside this time, the partial still where B left it
    rec = CR.recover_receive_dirs(root, "w2")
    assert rec[recv]["installed"] == [rev_a] and rec[recv]["failed"] == {} and rec[recv]["partial_aside"] == {} and rec[recv]["removed"] is True
    assert rec[recv]["manifest"]["window"] == "w1" and _receive_dirs(inbox) == []
    assert sorted(os.listdir(aside)) == a_files and not os.path.exists(os.path.join(inbox, CR.FAILED_DIR))
    assert {fn: R.file_sha256(os.path.join(d_inbox, fn)) for fn in os.listdir(d_inbox)} == expected1[rev_a]["sha256"]
    assert {s for _d, s in H.scan_inbox(root)} == {"merged"}
    ing = H.ingest_once(root, workers=0, min_free=0)
    assert ing.refused == 0 and ing.blocked == 0, (ing.refusals, ing.blocked)
    assert _rows(root, "SELECT COUNT(*), SUM(accepted) FROM source_revision WHERE revision_id = ?", (rev_a_id,)) == [(1, 1)]


def test_recovery_refuses_loudly_a_receive_directory_without_its_manifest_or_whose_copy_fails_its_digest_and_installs_nothing(tmp_path, monkeypatch):
    """A `.recv-*` whose RECEIVING manifest is gone cannot say which window, summary or verdicts it belongs to; a copy that
    does not match its manifest's digests is not the bytes the shard published. Both are moved under `_failed/<this window>/recv-<orig>/`
    with a loud UNRECOVERABLE line (the manifest beside the copy when there is one), nothing is installed, nothing merged, nothing folded;
    a manifest naming the adapter stale is refused the same way."""
    w, plan, rev_dir, rev = _cluster_spool(tmp_path, monkeypatch, with_bundle=False)
    root = w["root"]
    inbox = os.path.join(H.inbox_dir(root), CR.INBOX_NAME)
    with _dying_install_rename(), pytest.raises(RuntimeError, match="simulated crash"):
        CR.pull_window(root, plan, str(tmp_path / "expect"), source=plan["window_root"])
    (recv,) = _receive_dirs(inbox)
    lines: List[str] = []
    # (a) the manifest gone
    os.remove(os.path.join(inbox, recv, CR.RECEIVING_FILE))
    rec = CR.recover_receive_dirs(root, "w2", log=lines.append)
    assert rec[recv]["manifest"] == "missing" and rec[recv]["installed"] == [] and set(rec[recv]["failed"]) == {rev_dir} and rec[recv]["removed"] is True
    assert rec[recv]["failed"][rev_dir].startswith(f"no readable {CR.RECEIVING_FILE} in the receive directory")
    failed_copy = os.path.join(inbox, CR.FAILED_DIR, "w2", "recv-" + recv[len(CR.RECV_PREFIX):], rev_dir)
    assert os.path.isdir(failed_copy) and any(CR.MARKER_TOKEN in fn for fn in os.listdir(failed_copy))     # moved whole, never deleted
    assert not os.path.exists(os.path.join(inbox, "adapters", rev_dir)) and _receive_dirs(inbox) == []
    assert len(lines) == 1 and lines[0].startswith(f"[recover] UNRECOVERABLE {recv}/{rev_dir}: no readable {CR.RECEIVING_FILE}") and failed_copy in lines[0]
    assert H.scan_inbox(root) == [] and H.ingest_once(root, workers=0, min_free=0).ok == 0
    assert _rows(root, "SELECT COUNT(*) FROM source_revision WHERE revision_id = ?", (rev,)) == [(0,)]
    # (b) a copy that does not match its manifest's digests (one batch re-bytes-edited at the same size): UNRECOVERABLE, the manifest
    # copied beside the failed copy
    shutil.rmtree(os.path.join(inbox, CR.FAILED_DIR))
    with _dying_install_rename(), pytest.raises(RuntimeError, match="simulated crash"):
        CR.pull_window(root, plan, str(tmp_path / "expect_b"), source=plan["window_root"])
    (recv_b,) = _receive_dirs(inbox)
    victim = next(fn for fn in sorted(os.listdir(os.path.join(inbox, recv_b, rev_dir))) if CR.MARKER_TOKEN not in fn)
    _same_size_edit(os.path.join(inbox, recv_b, rev_dir, victim))
    lines.clear()
    rec_b = CR.recover_receive_dirs(root, "w3", log=lines.append)
    assert rec_b[recv_b]["installed"] == [] and rec_b[recv_b]["failed"][rev_dir].startswith(f"sha256 mismatch against the shard summary: ['{victim}']")
    failed_root_b = os.path.join(inbox, CR.FAILED_DIR, "w3", "recv-" + recv_b[len(CR.RECV_PREFIX):])
    assert os.path.isfile(os.path.join(failed_root_b, CR.RECEIVING_FILE)) and os.path.isdir(os.path.join(failed_root_b, rev_dir))
    assert not os.path.exists(os.path.join(inbox, "adapters", rev_dir)) and _receive_dirs(inbox) == [] and any("UNRECOVERABLE" in ln for ln in lines)
    # (c) a manifest naming the adapter stale (that window's post-run revalidation refused it): refused by name
    with _dying_install_rename(), pytest.raises(RuntimeError, match="simulated crash"):
        CR.pull_window(root, plan, str(tmp_path / "expect_c"), source=plan["window_root"])
    (recv_c,) = _receive_dirs(inbox)
    mpath = os.path.join(inbox, recv_c, CR.RECEIVING_FILE)
    CR._write_json(mpath, dict(json.load(open(mpath)), revalidate_post={"stale": [ADAPTER]}))
    rec_c = CR.recover_receive_dirs(root, "w4", log=lines.append)
    assert rec_c[recv_c]["installed"] == [] and "post-run revalidation" in rec_c[recv_c]["failed"][rev_dir]
    assert not os.path.exists(os.path.join(inbox, "adapters", rev_dir)) and _receive_dirs(inbox) == []
    assert H.scan_inbox(root) == [] and _rows(root, "SELECT COUNT(*) FROM source_revision WHERE revision_id = ?", (rev,)) == [(0,)]


def test_reactivate_by_id_refuses_an_accepted_superseded_revision_with_no_status_row_by_name(tmp_path, monkeypatch):
    """An accepted revision committed before `source_status` existed has no status row; `reactivate_committed` re-READS
    such a revision because restoring its measurements alone would make it the head while absent from `v_source_status`. The by-id
    path has no reader and re-emits committed bytes only (a status row is not among them), so it REFUSES such a revision by name -- no
    bundle, no budget spent, the way forward named -- and the pull's half records the adapter refused. A superseded revision WITH its status
    row re-activates as before."""
    root, store_root = _setup(tmp_path)                                               # s1 exists
    ad = SyntheticStoreAdapter(store_root)
    monkeypatch.setattr(RUN, "_source_status_records", lambda *a, **k: [])          # the runner of an earlier day filed no status
    with DB.reading(DB.db_path(root)) as conn:
        rep0 = RUN.run_pass(root, [ad], conn=conn, now=synth.t(2000), producer=PRODUCER)
    monkeypatch.undo()
    s1_old = next(r.revision_id for r in rep0.results if r.group == "s1")
    assert rep0.groups_refreshed == 1 and H.ingest_once(root, workers=0, min_free=0).refused == 0
    assert _rows(root, "SELECT COUNT(*) FROM source_status") == [(0,)]
    # s2 is published with its status row; then both change: the heads carry status rows, the old revisions are accepted and superseded
    _store(store_root, "s2", "legacy_store_2", "none", [0.1, 0.9])
    with DB.reading(DB.db_path(root)) as conn:
        rep1 = RUN.run_pass(root, [ad], conn=conn, now=synth.t(2500), producer=PRODUCER)
    s2_old = next(r.revision_id for r in rep1.results if r.group == "s2")
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    _store(store_root, "s1", "legacy_store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4, 0.9])
    _store(store_root, "s2", "legacy_store_2", "none", [0.1, 0.9, 0.5])
    with DB.reading(DB.db_path(root)) as conn:
        rep2 = RUN.run_pass(root, [ad], conn=conn, now=synth.t(3000), producer=PRODUCER)
    assert rep2.groups_refreshed == 2 and H.ingest_once(root, workers=0, min_free=0).refused == 0
    head_sql = "SELECT revision_id FROM source_revision WHERE source_id = ? AND accepted = 1 ORDER BY observed_utc DESC LIMIT 1"
    heads = {g: _rows(root, head_sql, (f"{ADAPTER}:{g}",))[0][0] for g in ("s1", "s2")}
    assert heads["s1"] != s1_old and heads["s2"] != s2_old
    assert _rows(root, "SELECT accepted FROM source_revision WHERE revision_id = ?", (s1_old,)) == [(1,)]
    assert _rows(root, "SELECT COUNT(*) FROM source_status WHERE source_revision_id = ?", (s1_old,)) == [(0,)]
    assert _rows(root, "SELECT COUNT(*) FROM source_status WHERE source_revision_id = ?", (s2_old,)) == [(1,)]
    budget = RUN.ReactivationBudget()
    with DB.reading(DB.db_path(root)) as conn:
        res = RUN.reactivate_by_id(root, ad, s1_old, conn, now=synth.t(4000), budget=budget)
        assert not res.accepted and not res.held and not res.deferred and res.files == [] and budget.taken == 0, res
        assert s1_old in res.error and f"{ADAPTER}:s1" in res.error and "no source_status row" in res.error and "--reread --force" in res.error
        # the control: the superseded revision WITH its status row re-activates from committed bytes
        res2 = RUN.reactivate_by_id(root, ad, s2_old, conn, now=synth.t(4000), budget=budget)
        assert res2.accepted and res2.reactivated and budget.taken == 1, res2
    assert RUN.complete_bundles(root, ADAPTER, s1_old) == [] and len(RUN.complete_bundles(root, ADAPTER, s2_old)) == 1
    # the pull's half: the refusal is the adapter's recorded refusal for the window, nothing written, the head untouched
    report = {"refused_known_ids": {ADAPTER: [s1_old]}, "adapters_refused": {}}
    with DB.reading(DB.db_path(root)) as conn:
        out = CR.reactivate_refused_known(root, {}, report, conn=conn, now=synth.t(4100), adapter_of=lambda name: ad)
    assert out[ADAPTER][s1_old]["state"] == "refused" and "no source_status row" in out[ADAPTER][s1_old]["error"]
    assert any(s1_old in r and "not re-activated from committed bytes" in r for r in report["adapters_refused"][ADAPTER])
    assert RUN.complete_bundles(root, ADAPTER, s1_old) == []
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    assert _rows(root, head_sql, (f"{ADAPTER}:s1",)) == [(heads["s1"],)] and _rows(root, head_sql, (f"{ADAPTER}:s2",)) == [(s2_old,)]


# --------------------------------------------------------------------------------------------
# the anchor relation on the sidecar (reader_state/2)
# --------------------------------------------------------------------------------------------

def test_the_sidecar_carries_the_anchor_relation_for_every_export_and_a_shard_loads_the_currency_from_it(tmp_path):
    """Every export ships the unsteered `run` rows, a TABLE `v_run_current` over EVERY run, the complete judgments of the unsteered runs and
    EVERY `run_metric` row of theirs (a metric is an anchor metric the moment a steered row intends a contrast on it, so the set is not read off
    `run_contrast`); `AnchorCurrency.load` over the sidecar agrees with the same load over the database."""
    root, home, stores, ad = _grading_host(tmp_path)
    out = str(tmp_path / "rs.sqlite")
    rep = _export(root, out, home)
    none_rid = _rows(root, "SELECT run_id FROM run WHERE object_id = 'none'")[0][0]
    live = {r[0]: (r[1], r[2], r[3]) for r in _rows(root, "SELECT run_id, is_current, in_current_paper, roster_entry_ids_json FROM v_run_current")}
    n_judg = _rows(root, "SELECT COUNT(*) FROM judgment WHERE status = 'complete' AND run_id = ?", (none_rid,))[0][0]
    assert len(live) == 3 and n_judg == 1
    conn = RS.open_reader_state(out, [ADAPTER])
    try:
        tables = {t[0] for t in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert set(RS.ANCHOR_TABLES) <= tables and "v_paper_current" not in tables
        assert [c[1] for c in conn.execute("PRAGMA table_info(v_run_current)")] == list(RS.RUN_CURRENT_COLUMNS) == list(AC.CURRENCY_COLUMNS)
        assert {r[0]: (r[1], r[2], r[3]) for r in conn.execute("SELECT run_id, is_current, in_current_paper, roster_entry_ids_json FROM v_run_current")} == live
        assert [r[0] for r in conn.execute("SELECT run_id FROM run")] == [none_rid]                        # the unsteered rows only, every column
        assert [c[1] for c in conn.execute("PRAGMA table_info(run)")] == [r[0] for r in _rows(root, "SELECT name FROM pragma_table_info('run')")]
        assert rep.rows["run"] == 1 and rep.rows["v_run_current"] == 3 and rep.rows["judgment"] == n_judg
        n_rm = _rows(root, "SELECT COUNT(*) FROM run_metric WHERE run_id = ?", (none_rid,))[0][0]
        assert rep.rows["run_metric"] == n_rm >= 1 and rep.withdrawn_metric_rows == 0 and rep.non_current_judgment_metric_rows == 0   # every unsteered reading ships, contrast or not
        assert [r[0] for r in conn.execute("SELECT DISTINCT run_id FROM run_metric")] == [none_rid]
        assert {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")} >= {"ix_run_key", "ix_vrc_run", "ix_jd_run", "ix_rm_run"}
        shard = AC.AnchorCurrency.load(conn)                   # the shard's loader over the sidecar ...
        with DB.reading(DB.db_path(root)) as live_conn:
            grading_host = AC.AnchorCurrency.load(live_conn)         # ... and the grading host's over the database
        assert shard.is_current(none_rid) is True and grading_host.is_current(none_rid) is True
        assert all(shard.is_current(r) == grading_host.is_current(r) and shard.currency(r) == grading_host.currency(r) for r in live)
        assert shard.counts == {"runs": 3, "unsteered": 1, "judgments": n_judg, "metric_rows": n_rm, "withdrawn_metric_rows": 0, "non_current_judgment_rows": 0, "items_known": 0, "flags_known": 0}
        assert grading_host.counts == dict(shard.counts, items_known=1, flags_known=1) and grading_host.items_known is True and shard.items_known is False
        assert shard.unsteered_run(none_rid) == grading_host.unsteered_run(none_rid)
        ctx = AB.AdapterContext(root="", now=synth.t(2001), producer=dict(PRODUCER), conn=conn)
        assert ctx.anchor_currency().counts == shard.counts and ctx.anchor_currency() is ctx.anchors
    finally:
        conn.close()


def test_a_sidecar_of_an_older_schema_is_refused_by_name_and_one_without_the_currency_table_cannot_load_anchors(tmp_path):
    """A `reader_state/1` sidecar carries no anchor relation: `open_reader_state` refuses it by name (a shard must never publish
    contrast-less revisions off it); a sidecar at the current tag that lost the table makes the loader refuse by name -- the group's
    refusal, never a contrast silently dropped."""
    root, home, stores, ad = _grading_host(tmp_path)
    out = str(tmp_path / "rs.sqlite")
    _export(root, out, home)
    w = sqlite3.connect(out)
    w.execute("UPDATE reader_state_meta SET schema = 'reader_state/1'")
    w.commit()
    w.close()
    with pytest.raises(RS.ReaderStateError, match="schema 'reader_state/1' is not 'reader_state/2'"):
        RS.open_reader_state(out, [ADAPTER])
    w = sqlite3.connect(out)
    w.execute("UPDATE reader_state_meta SET schema = ?", (RS.SCHEMA,))
    w.execute("DROP TABLE v_run_current")
    w.commit()
    w.close()
    conn = RS.open_reader_state(out, [ADAPTER])
    try:
        with pytest.raises(AC.AnchorCurrencyUnavailable, match=r"v_run_current is not defined in this database \(schema \d+\)"):
            AC.AnchorCurrency.load(conn)
        ctx = AB.AdapterContext(root="", now=synth.t(2001), producer=dict(PRODUCER), conn=conn)
        with pytest.raises(AC.AnchorCurrencyUnavailable):
            ctx.anchor_currency()
        assert ctx.anchors is None                              # a failed load is not cached
    finally:
        conn.close()


def test_the_sidecar_ships_only_the_readings_the_frame_counts_and_a_shard_agrees_with_the_grading_host_on_a_withdrawn_one(tmp_path):
    """A revision withdrew the anchor's computed twin and kept its run current (`withdrawn_reading_world`): the export drops the withdrawn
    row by the paper frame's own predicate over the LIVE `observation` table (the sidecar's own holds the adapters' current rows only),
    reports how many, and the shard's load, counts and choice agree with the grading host's -- the predicate evaluated once, at export."""
    from test_registry_anchor_current import TWIN_COMPUTATION, WORLD_ADAPTER, withdraw_twin, withdrawn_reading_world
    root = str(tmp_path / "reg")
    w = withdrawn_reading_world(root)
    withdraw_twin(root, w)
    none, k = str(w["none"]), w["key"]
    assert _rows(root, "SELECT computation_id, value FROM run_metric WHERE run_id = ? ORDER BY 1", (none,)) == [("", 0.25), (TWIN_COMPUTATION, 0.5)]
    out = str(tmp_path / "rs.sqlite")
    rep = RS.export(root, [WORLD_ADAPTER], out, uri_home=str(tmp_path / "home"), repo_root=REPO_ROOT)
    assert rep.rows["run_metric"] == 1 and rep.withdrawn_metric_rows == 1 and rep.as_dict()["withdrawn_metric_rows"] == 1
    assert rep.non_current_judgment_metric_rows == 0 and rep.as_dict()["non_current_judgment_metric_rows"] == 0
    conn = RS.open_reader_state(out, [WORLD_ADAPTER])
    try:
        assert [tuple(r) for r in conn.execute("SELECT computation_id, value FROM run_metric WHERE run_id = ?", (none,))] == [("", 0.25)]
        assert conn.execute("SELECT COUNT(*) FROM observation WHERE current = 0").fetchone()[0] == 0        # the sidecar's observations: current rows only
        shard = AC.AnchorCurrency.load(conn)
        with DB.reading(DB.db_path(root)) as live:
            grading_host = AC.AnchorCurrency.load(live)
        assert grading_host.counts["withdrawn_metric_rows"] == 1 and shard.counts["withdrawn_metric_rows"] == 0        # dropped at export; nothing left to drop
        assert shard.counts == dict(grading_host.counts, withdrawn_metric_rows=0, non_current_judgment_rows=0, items_known=0, flags_known=0) and shard.counts["metric_rows"] == 1
        a, b = shard.choose(k, "asr", 0.5, synth.JUDGE_ID), grading_host.choose(k, "asr", 0.5, synth.JUDGE_ID)
        assert a.state == b.state == "sibling_same_source" and a.run_id == b.run_id == none and a.anchor_kind == "same_standup_none"
        assert a.measurement == b.measurement and a.measurement is not None and a.measurement.computation_id == "" and a.measurement.value == 0.25
        assert a.selection_json() == b.selection_json()
    finally:
        conn.close()


# ============================================================================================
# the anchor decision input on a cluster shard
# ============================================================================================

import _reanchor_fixtures as X  # noqa: E402


def test_a_cluster_shard_computes_the_anchor_decision_input_off_its_sidecar_and_publishes_only_the_id_the_grading_host_never_saw(tmp_path):
    """The sidecar carries `source_revision` whole (the heads' recorded sets), `source_watermark` (the decision watermark) and the
    `v_run_current` table, so a shard replays the recorded comparisons exactly as the grading host does: an unmoved frame plans nothing; a frame
    where a ruling retired the own anchor plans exactly that group and publishes the re-anchored revision under an id the grading host has never
    seen, its marker carrying the decision set and its `inputs_json` the decision URI; once the grading host folds it, a shard over a fresh sidecar
    refuses the (now known) id by name and the grading host's own pass is quiet."""
    home = tmp_path / "home"
    stores = str(home / "stores")
    root, _unused = X.contrast_world(tmp_path, stores_dir=stores)
    sid = RUN.source_id_for(X.ContrastStoreAdapter.name, "s1")
    steered, own, sib = X.run_id_of(X.STEERED, "constitution_lora@aaaa"), X.run_id_of(X.OWN), X.run_id_of(X.SIBLING)
    X.pass_and_fold(root, SyntheticStoreAdapter(stores), synth.t(1900))              # the runs fold first: the contrast read's frame holds them
    rep0, _ = X.pass_and_fold(root, X.ContrastStoreAdapter(stores), synth.t(2000))
    assert rep0.groups_refreshed == 3 and X.paper_contrast(root, steered) == [(own, "same_standup_none", 1)]
    rev1, extra1 = X.head_of(root, sid)
    sidecar1 = str(tmp_path / "rs1.sqlite")
    _export(root, sidecar1, str(home))
    # the shard reads the head's set and the decision watermark off the sidecar: nothing moved, nothing planned
    cluster_root = _cluster_root(tmp_path)
    rep1 = _cluster_pass(cluster_root, X.ContrastStoreAdapter(stores), sidecar1, synth.t(3000))
    assert rep1.results == [] and rep1.groups_seen == 3 and rep1.anchor_decisions_changed == 0 and not _published(cluster_root)
    # the grading host folds an exclude on the own unsteered run; a sidecar exported after it carries the moved frame
    X.file_ruling(root, "r-exclude-own", "exclude", own, 3100)
    sidecar2 = str(tmp_path / "rs2.sqlite")
    _export(root, sidecar2, str(home))
    rep2 = _cluster_pass(cluster_root, X.ContrastStoreAdapter(stores), sidecar2, synth.t(3200))
    assert [r.group for r in rep2.results] == ["s1"] and rep2.groups_refreshed == 1 and rep2.anchor_decisions_changed == 1 and rep2.inputs_changed == 1
    (res,) = rep2.results
    new = _published(cluster_root)
    assert res.accepted and new == [res.revision_id] and res.revision_id != rev1
    assert not _rows(root, "SELECT 1 FROM source_revision WHERE revision_id = ?", (res.revision_id,))        # an id the grading host has never seen
    marker = _marker(os.path.join(H.records_dir(cluster_root), "adapters", X.ContrastStoreAdapter.name, res.revision_id))
    block = marker["payload"]["extra"]["anchor_decisions"]
    assert block["comparisons"][0]["decision"]["anchor_run_id"] == sib and block["comparisons"][0]["decision"]["own_reason"] == "excluded"
    assert marker["payload"]["inputs_json"][AC.decision_uri(sid)] == block["fingerprint"] == block["pre_read_fingerprint"]
    assert marker["producer"]["substrate"] == "cluster_a" and marker["payload"]["extra"]["read_copy"]["substrate"] == "cluster_a"
    # the shard's id is the grading host's for the same facts: the grading host would mint it too
    with DB.reading(DB.db_path(root)) as live:
        frame = AC.AnchorCurrency.load(live)
    grading_host_fp = AC.decision_fingerprint(frame.decisions_of([e["comparison"] for e in extra1["anchor_decisions"]["comparisons"]]))
    assert grading_host_fp == block["fingerprint"]
    assert res.revision_id == RUN.revision_id_for(sid, [i for i in X.ContrastStoreAdapter(stores).discover() if os.path.basename(os.path.dirname(i.uri)) == "s1"]
                                                  + [AC.AnchorDecisionInput.of(sid, grading_host_fp)], 2)
    # the grading host folds the shard's revision: the paper re-anchors, the old key is withdrawn, and the grading host's own pass has nothing to do
    shutil.copytree(os.path.join(H.records_dir(cluster_root), "adapters", X.ContrastStoreAdapter.name, res.revision_id),
                    os.path.join(H.records_dir(root), "adapters", X.ContrastStoreAdapter.name, res.revision_id))
    X.ingest(root)
    assert X.head_of(root, sid)[0] == res.revision_id and X.paper_contrast(root, steered) == [(sib, "sibling_none", 1)]
    assert _rows(root, "SELECT COUNT(*) FROM withdrawal WHERE source_id = ? AND table_name = 'run_contrast'", (sid,)) == [(1,)]
    assert X.decision_watermark(root, sid) == (block["fingerprint"], None)
    rep3 = X.pass_once(root, X.ContrastStoreAdapter(stores), synth.t(3300))
    assert rep3.results == [] and rep3.anchor_decisions_changed == 0
    # a fresh sidecar: the shard plans nothing; under --full the id it mints is the grading host's accepted head -> refused by name
    sidecar3 = str(tmp_path / "rs3.sqlite")
    _export(root, sidecar3, str(home))
    cluster_root2 = str(tmp_path / "cluster_root2")
    H.ensure_layout(cluster_root2)
    rep4 = _cluster_pass(cluster_root2, X.ContrastStoreAdapter(stores), sidecar3, synth.t(3400))
    assert rep4.results == [] and not _published(cluster_root2)
    rep5 = _cluster_pass(cluster_root2, X.ContrastStoreAdapter(stores), sidecar3, synth.t(3500), full=True)
    by = {r.group: r for r in rep5.results}
    assert by["s1"].refused_known and by["s1"].revision_id == res.revision_id and "source_revision row, accepted" in by["s1"].error
    assert rep5.groups_refused_known == 3 and not _published(cluster_root2)


def test_a_shard_over_a_sidecar_without_the_currency_table_preserves_the_recorded_groups_and_writes_nothing_for_a_refused_read(tmp_path):
    """A `reader_state/2`-tagged sidecar exported without its `v_run_current` table (an incomplete
    sidecar) opens, and the shard's pre-read cannot load the frame -- the group whose head recorded a set is preserved by name and nothing is
    published; a new steered group's read asks the frame and refuses by name with NOTHING written to the window's spool (no refused marker, no
    watermark: the fault is the staged copy's, and the grading host's own pass reads the group); the pass completes."""
    home = tmp_path / "home"
    stores = str(home / "stores")
    root, _unused = X.contrast_world(tmp_path, stores_dir=stores)
    X.pass_and_fold(root, SyntheticStoreAdapter(stores), synth.t(1900))
    X.pass_and_fold(root, X.ContrastStoreAdapter(stores), synth.t(2000))
    sid = RUN.source_id_for(X.ContrastStoreAdapter.name, "s1")
    rev1, extra1 = X.head_of(root, sid)
    fp1 = extra1["anchor_decisions"]["fingerprint"]
    sidecar = str(tmp_path / "rs_incomplete.sqlite")
    _export(root, sidecar, str(home))
    con = sqlite3.connect(sidecar)
    con.execute("DROP TABLE v_run_current")
    con.commit()
    con.close()
    cluster_root = _cluster_root(tmp_path)

    def spool_dirs():
        return sorted(p for p, _, _ in os.walk(H.records_dir(cluster_root)) if "_refused" in p or "_activations" in p) + _published(cluster_root)

    rep1 = _cluster_pass(cluster_root, X.ContrastStoreAdapter(stores), sidecar, synth.t(3000))
    (res,) = rep1.results
    assert res.group == "s1" and res.anchor_unavailable and not res.accepted and res.files == [] and res.error.startswith(f"{sid}: preserved as it stands")
    assert "v_run_current is not defined" in res.error and (rep1.groups_refused, rep1.groups_anchor_unavailable, rep1.groups_seen) == (1, 1, 3)
    assert spool_dirs() == []
    # a new steered store: its read asks the frame; refused by name, nothing in the spool, the pass completes
    X.store_with_anchor(stores, "s5", "legacy_store_5", "constitution_lora@bbbb", [0.5, 0.5], anchor_key=X.OWN)
    rep2 = _cluster_pass(cluster_root, X.ContrastStoreAdapter(stores), sidecar, synth.t(3100))
    by = {r.group: r for r in rep2.results}
    assert set(by) == {"s1", "s5"} and by["s5"].anchor_unavailable and not by["s5"].accepted and by["s5"].files == [] and by["s5"].revision_id
    assert "the anchor currency of the pass is unavailable, the group refuses whole" in by["s5"].error and by["s1"].files == []
    assert (rep2.groups_refused, rep2.groups_anchor_unavailable, rep2.groups_refreshed) == (2, 2, 0) and spool_dirs() == []
    # the grading host's own pass over the live frame: s1 quiet (its head and watermark untouched), s5 read and filed
    assert X.head_of(root, sid)[0] == rev1 and X.decision_watermark(root, sid) == (fp1, None)
    rep3, _ = X.pass_and_fold(root, X.ContrastStoreAdapter(stores), synth.t(3200))
    assert [r.group for r in rep3.results] == ["s5"] and rep3.results[0].accepted and rep3.groups_anchor_unavailable == 0
    assert X.head_of(root, sid)[0] == rev1 and X.paper_contrast(root, X.run_id_of("legacy_store_5", "constitution_lora@bbbb")) == [(X.run_id_of(X.OWN), "same_standup_none", 1)]


# --------------------------------------------------------------------------------------------
# a LISTING input on a shard is read against the store manifest's attestation of the canonical directory
# --------------------------------------------------------------------------------------------

def test_a_shard_compares_a_listing_with_the_manifests_attestation_of_the_canonical_directory_never_with_a_previous_revision(tmp_path):
    """The grading host folds s1 (its store directory holds nothing but store.json). Then the canonical gains an EMPTY
    ignored directory AND the store's bytes change; the sidecar still carries the previous head. A staged copy that carries the new bytes
    but not the directory (a file-only stage) lists exactly what the previous revision listed -- the old guard passed it and the changed
    bytes minted an accepted revision whose `$.entries` omitted the directory. Now the shard compares the listing input with the manifest's
    attestation of the CANONICAL directory: no attestation (an older manifest) refuses `staged_listing_unattested`; the attestation naming
    the directory the stage lacks refuses `staged_listing_mismatch` and the changed bytes are never published; the directory reproduced
    on the stage, the shard publishes a revision whose `$.entries` carries it."""
    from test_registry_source_state_per_revision import ListingStoreAdapter
    from sisteer.registry import listing as L
    root, _unused = _setup(tmp_path)
    home = tmp_path / "home"
    stores = home / "stores"
    stores.mkdir(parents=True)
    _store(str(stores), "s1", "legacy_store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4])
    _store(str(stores), "s2", "legacy_store_2", "none", [0.1, 0.9])
    ad = ListingStoreAdapter(str(stores))
    with DB.reading(DB.db_path(root)) as conn:
        rep = RUN.run_pass(root, [ad], conn=conn, now=synth.t(2000), producer=PRODUCER)
    assert rep.groups_refreshed == 2 and H.ingest_once(root, workers=0, min_free=0).refused == 0
    sid1 = RUN.source_id_for(ad.name, "s1")
    sidecar = str(tmp_path / "rs.sqlite")
    _export(root, sidecar, str(home), adapters=(ad.name,))
    # the canonical moves after the head and the export: an empty ignored directory and new bytes
    canonical_dir = stores / "s1"
    (canonical_dir / "killed_empty").mkdir()
    _store(str(stores), "s1", "legacy_store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4, 0.6])
    canonical = L.listing_entries(str(canonical_dir))               # what the manifest attests for the canonical directory at the push
    assert ("killed_empty", "dir") in canonical
    # the file-only stage: the bytes, not the directory (the grading host tree stands for the staged copy here)
    (canonical_dir / "killed_empty").rmdir()
    (listing1,) = [i for i in ad.discover() if AB.is_listing_input(i) and ad.comparison_group(i) == "s1"]
    assert listing1.entries == ("store.json",) and listing1.path == str(canonical_dir)

    def shard(name, attestation):
        cluster_root = _cluster_root(tmp_path, name)
        conn = RS.open_reader_state(sidecar, [ad.name])
        try:
            cluster = RUN.ClusterRead.from_sidecar(conn, {"substrate": "cluster_a", "host": "node-x", "components": {}})
            cluster.listing_attestation = attestation
            rep = RUN.run_pass(cluster_root, [ad], conn=conn, now=synth.t(3000), producer=dict(PRODUCER, substrate="cluster_a", host="node-x"),
                               cluster=cluster, save_cache=False)
        finally:
            conn.close()
        return cluster_root, {r.group: r for r in rep.results}

    # (a) no attestation reaches the shard (a manifest written before the field): refused by name, nothing published
    cr, by = shard("cluster_root_a", None)
    assert set(by) == {"s1"} and not by["s1"].accepted and by["s1"].inputs_dropped and RUN.STAGED_LISTING_UNATTESTED_KEY in by["s1"].error, by["s1"].error
    assert not _published(cr, ad.name)
    # (b) the attestation of the canonical directory names the directory the stage lacks: a mismatch, refused by name, the changed bytes never published
    attest = {str(canonical_dir): canonical}
    cr, by = shard("cluster_root_b", lambda path: attest.get(os.path.realpath(path)))
    assert set(by) == {"s1"} and not by["s1"].accepted and by["s1"].inputs_dropped and RUN.STAGED_LISTING_MISMATCH_KEY in by["s1"].error, by["s1"].error
    assert listing1.uri in by["s1"].error and L.listing_fingerprint(canonical)[:12] in by["s1"].error
    assert not _published(cr, ad.name) and not os.path.isdir(os.path.join(H.records_dir(cr), "adapters"))
    # (c) the stage reproduces the directory: the listing is the attested one, the shard reads and its revision carries the directory
    (canonical_dir / "killed_empty").mkdir()
    cr, by = shard("cluster_root_c", lambda path: attest.get(os.path.realpath(path)))
    assert set(by) == {"s1"} and by["s1"].accepted and not by["s1"].inputs_dropped, by["s1"].error
    (rev_dir,) = _published(cr, ad.name)
    entries = [rec["payload"] for fn in os.listdir(os.path.join(H.records_dir(cr), "adapters", ad.name, rev_dir)) if "source_revision" in fn
               for _, rec, _ in R.read_record_file(os.path.join(H.records_dir(cr), "adapters", ad.name, rev_dir, fn))
               if rec["record_type"] == "source_state" and rec["payload"]["json_path"] == "$.entries"]
    assert [(p["source_id"], p["current_members_json"]) for p in entries] == [(sid1, ["killed_empty", "store.json"])]
    # a group whose directory the attestation does not cover (another store the manifest never saw) is unattested, not published
    _store(str(stores), "s3", "legacy_store_3", "constitution_lora@aaaa", [0.7, 0.7, 0.2])
    cr, by = shard("cluster_root_d", lambda path: attest.get(os.path.realpath(path)))
    assert not by["s3"].accepted and RUN.STAGED_LISTING_UNATTESTED_KEY in by["s3"].error and "s3" not in " ".join(_published(cr, ad.name))


def test_the_plan_covers_a_listing_input_by_its_directorys_attested_listing_and_the_shard_refuses_an_unattested_or_moved_one_by_name(tmp_path, monkeypatch):
    """A listing input is a `registry://listing/...` identity under no component's base: judged by its URI alone it would read as uncovered.
    It is covered by its DIRECTORY: under a pinned base, with the generation's manifest copy attesting that directory's listing at the
    discovery's fingerprint -- the shard's own comparison (`ClusterListingRefusal`). Here the plan runs NO discovery and nothing ever
    routes to a grading-host read: the coverage rule is pinned through its pure reference implementation (`input_coverage` /
    `uncovered_reason`, a caller's discovered input set against the manifest copies); the plan's pin carries the ATTESTATION the shard
    compares with (`attestations`); the admission refuses a canonical whose listings moved (`listing_changed`) or a pin naming an
    attestation this grading host never wrote (`no_attestation`) -- the adapter recorded refused by name, no task; and a shard handed no
    attestation (an older manifest) or a moved listing refuses the group by name (`staged_listing_unattested` /
    `staged_listing_mismatch`), nothing written."""
    from test_registry_source_state_per_revision import ListingStoreAdapter
    root, _unused = _setup(tmp_path)
    home = tmp_path / "home"
    stores = home / "stores"
    stores.mkdir(parents=True)
    _store(str(stores), "s1", "legacy_store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4])
    (stores / "s1" / "killed_empty").mkdir()
    ad = ListingStoreAdapter(str(stores))
    monkeypatch.setattr(ST, "HOME", str(home))
    cfgp = TS._cfg(tmp_path, {"listing_stores": {"base": str(stores), "authority": "grading_host_copy", "origin": "grading_host"}},
                   adapters={ad.name: {"shards": 1, "components": ["listing_stores"]}})
    cfg = ST.load_config(cfgp)
    dest = cfg.destination("cluster_a")
    res = ST.mirror_component(root, dest, cfg.components["listing_stores"], runner=TS.Runner())
    assert res.outcome == "published", res.as_dict()
    sha8 = res.content_sha8
    pins = {"listing_stores": sha8}
    inputs = [i for i in ad.discover() if AB.is_listing_input(i)]
    (li,) = inputs
    assert li.path == str(stores / "s1") and li.entries == ("killed_empty", "store.json")
    cov = CR.input_coverage(cfg, root, pins, inputs, REPO_ROOT, str(home), ad.roots)
    assert cov == {"discovered": 1, "covered": 1, "uncovered": 0, "roots": {}}, cov
    attested = CR.generation_listings(root, "listing_stores", sha8)
    assert attested["s1"] == (("killed_empty", "dir"), ("store.json", "file"))
    lsha = ST.published_attestations(root)["listing_stores"]
    assert lsha
    # the plan: no discovery (a load_all that would construct anything proves it), the cluster route, the pin's attestation beside the pin
    monkeypatch.setattr("sisteer.registry.adapters.load_all", lambda: pytest.fail("the plan discovers nothing (C6)"))
    with AB.without_store_access_scope():
        plan = CR.plan_window(root, "wl", [ad.name], cfg, dest, "a" * 40, uri_home=str(home), repo_root=REPO_ROOT)
    e = plan["adapters"][ad.name]
    assert e["route"] == "cluster" and len(plan["tasks"]) == 1 and "inputs" not in e and plan["refusals"] == {}, e
    assert e["attestations"] == {"listing_stores": lsha} and e["revalidation_pre"]["listing_stores"]["ok"] is True
    # the shard's guard, fed the attestation the plan pinned (the grading host tree stands for the staged copy): the listing the stage carries IS
    # the attested one -> accepted, a revision published
    sidecar = str(tmp_path / "rs.sqlite")
    _export(root, sidecar, str(home), adapters=(ad.name,))
    stores_real = os.path.realpath(str(stores))

    def by_rel(path):
        return attested.get(os.path.relpath(os.path.realpath(path), stores_real))

    def shard(name, attestation):
        cluster_root = _cluster_root(tmp_path, name)
        conn = RS.open_reader_state(sidecar, [ad.name])
        try:
            cluster = RUN.ClusterRead.from_sidecar(conn, {"substrate": "cluster_a", "host": "node-x", "components": {}})
            cluster.listing_attestation = attestation
            rep = RUN.run_pass(cluster_root, [ad], conn=conn, now=synth.t(3000), producer=dict(PRODUCER, substrate="cluster_a", host="node-x"),
                               cluster=cluster, save_cache=False)
        finally:
            conn.close()
        return cluster_root, {r.group: r for r in rep.results}

    cr, by = shard("cluster_root_ok", by_rel)
    assert by["s1"].accepted and not by["s1"].inputs_dropped, by["s1"].error
    assert len(_published(cr, ad.name)) == 1
    # moved since the push: the canonical directory gains an entry the generation does not attest -- uncovered by the rule, refused by the
    # admission (`listing_changed`: the adapter recorded refused by name, no task; never a grading host read), and refused by the shard by name
    (stores / "s1" / "later_empty").mkdir()
    moved = [i for i in ad.discover() if AB.is_listing_input(i)]
    cov = CR.input_coverage(cfg, root, pins, moved, REPO_ROOT, str(home), ad.roots)
    assert cov["uncovered"] == 1 and cov["roots"] == {str(stores): {"count": 1, "why": f"a directory whose listing moved since generation {sha8} was pushed", "sample": [li.uri]}}, cov
    assert CR.uncovered_reason(cov) == f"{CR.UNCOVERED_PREFIX}: 1 of 1 -- 1 under {stores} (a directory whose listing moved since generation {sha8} was pushed)"
    rv = ST.revalidate("listing_stores", sha8, root=root, cfg=cfg, listings_sha256=lsha)
    assert rv.refusal == "listing_changed" and "s1" in rv.listing_mismatched, rv
    with AB.without_store_access_scope():
        plan_moved = CR.plan_window(root, "wl2", [ad.name], cfg, dest, "a" * 40, uri_home=str(home), repo_root=REPO_ROOT)
    em = plan_moved["adapters"][ad.name]
    assert em["route"] != "cluster" and plan_moved["tasks"] == [] and "listing_changed" in em["reason"], em
    assert plan_moved["refusals"] == {ad.name: [f"plan: {em['reason']}"]}
    cr, by = shard("cluster_root_moved", by_rel)
    assert not by["s1"].accepted and by["s1"].inputs_dropped and RUN.STAGED_LISTING_MISMATCH_KEY in by["s1"].error and li.uri in by["s1"].error, by["s1"].error
    assert not _published(cr, ad.name)
    (stores / "s1" / "later_empty").rmdir()
    # a directory the generation attests no listing for (a store that landed after the push), and a listing naming no directory: uncovered by
    # the rule; on the shard the new store's directory is attested by nothing -> refused by name, while s1 stands
    _store(str(stores), "s2", "legacy_store_2", "none", [0.1, 0.9])
    new = [i for i in ad.discover() if AB.is_listing_input(i) and ad.comparison_group(i) == "s2"] + [AB.ListingInput.of("x:y", "root", [("a", "file")])]
    cov = CR.input_coverage(cfg, root, pins, new, REPO_ROOT, str(home), ad.roots)
    assert cov["uncovered"] == 2 and set(cov["roots"]) == {str(stores), "registry://listing/x:y"}, cov
    assert cov["roots"][str(stores)]["why"] == f"a directory generation {sha8} attests no listing for"
    assert cov["roots"]["registry://listing/x:y"]["why"].startswith("a listing naming no directory of its own")
    cr, by = shard("cluster_root_new", by_rel)
    assert not by["s2"].accepted and RUN.STAGED_LISTING_UNATTESTED_KEY in by["s2"].error and by["s1"].accepted, (by["s2"].error, by["s1"].error)
    shutil.rmtree(stores / "s2")
    # the plain MANIFEST-<sha8>.json stripped of its listings: the plan and the admission read the PIN'S attestation (the keyed
    # MANIFEST-<sha8>.<lsha8>.json the publish recorded), never the plain file, so the route stays cluster
    sdir = ST.component_state_dir(root, "listing_stores")
    jpath = os.path.join(sdir, f"MANIFEST-{sha8}.json")
    meta = json.load(open(jpath))
    assert meta["listings_sha256"] == lsha
    for k in ("directories", "n_directories", "n_empty_directories", "listings_sha256"):
        meta.pop(k)
    ST._write_json(jpath, meta)
    assert CR.generation_listings(root, "listing_stores", sha8) is None                       # no attestation named: the plain file
    assert CR.generation_listings(root, "listing_stores", sha8, lsha)["s1"] == (("killed_empty", "dir"), ("store.json", "file"))
    with AB.without_store_access_scope():
        plan2 = CR.plan_window(root, "wm", [ad.name], cfg, dest, "a" * 40, uri_home=str(home), repo_root=REPO_ROOT)
    assert plan2["adapters"][ad.name]["route"] == "cluster" and plan2["adapters"][ad.name]["attestations"] == {"listing_stores": lsha}, plan2["adapters"][ad.name]
    # an older generation: no keyed attestation and a publish that recorded none. The plan pins NO attestation (None; the admission compares
    # the plain file, which attests no listing, so the content is admitted) -- and on the shard that None is what every listing input is
    # compared with: refused by name (`staged_listing_unattested`), nothing written. Never a grading host read.
    os.remove(ST.attestation_json_path(os.path.join(sdir, f"MANIFEST-{sha8}.sha256"), lsha))
    ST.update_state(root, "listing_stores", {"published_attestation": None}, "cluster_a")
    cov = CR.input_coverage(cfg, root, pins, inputs, REPO_ROOT, str(home), ad.roots)
    assert cov["uncovered"] == 1 and cov["roots"][str(stores)]["why"] == f"a directory listing generation {sha8}'s manifest does not attest (written before the field)"
    with AB.without_store_access_scope():
        plan3 = CR.plan_window(root, "wn", [ad.name], cfg, dest, "a" * 40, uri_home=str(home), repo_root=REPO_ROOT)
    e3 = plan3["adapters"][ad.name]
    assert e3["attestations"] == {"listing_stores": None} and e3["route"] == "cluster", e3
    cr, by = shard("cluster_root_older", None)
    assert not by["s1"].accepted and by["s1"].inputs_dropped and RUN.STAGED_LISTING_UNATTESTED_KEY in by["s1"].error, by["s1"].error
    assert not _published(cr, ad.name)
    # a pin naming an attestation the grading host never wrote: refused by name at the admission (`no_attestation`), no task -- never a grading host read
    with pytest.raises(CR.ClusterReadsError, match="no attestation of generation"):
        CR.generation_listings(root, "listing_stores", sha8, "3" * 64)
    with AB.without_store_access_scope():
        plan4 = CR.plan_window(root, "wo", [ad.name], cfg, dest, "a" * 40, uri_home=str(home), repo_root=REPO_ROOT, attestations={"listing_stores": "3" * 64})
    e4 = plan4["adapters"][ad.name]
    assert e4["route"] != "cluster" and plan4["tasks"] == [] and "no_attestation" in e4["reason"] and plan4["refusals"] == {ad.name: [f"plan: {e4['reason']}"]}, e4
    assert e4["revalidation_pre"]["listing_stores"]["refusal"] == "no_attestation"


def test_run_shard_installs_the_pinned_generations_listing_attestation_and_the_shards_revision_names_the_stores_empty_directory(tmp_path, monkeypatch):
    """The whole chain through a real mirrored generation: the component's manifest attests the store's empty directory, the generation
    carries it, run_shard installs the attestation from the generation hash cache, and the shard's revision of the changed store names the
    directory in `$.entries`."""
    from test_registry_source_state_per_revision import ListingStoreAdapter
    root, _unused = _setup(tmp_path)
    home = tmp_path / "home"
    stores = home / "stores"
    stores.mkdir(parents=True)
    _store(str(stores), "s1", "legacy_store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4])
    _store(str(stores), "s2", "legacy_store_2", "none", [0.1, 0.9])
    (stores / "s1" / "killed_empty").mkdir()
    ad = ListingStoreAdapter(str(stores))
    name = ad.name
    with DB.reading(DB.db_path(root)) as conn:
        rep = RUN.run_pass(root, [ad], conn=conn, now=synth.t(2000), producer=PRODUCER)
    assert rep.groups_refreshed == 2 and H.ingest_once(root, workers=0, min_free=0).refused == 0
    monkeypatch.setattr(ST, "HOME", str(home))
    cfgp = TS._cfg(tmp_path, {"listing_stores": {"base": str(stores), "authority": "grading_host_copy", "origin": "grading_host"}},
                   adapters={name: {"shards": 1, "components": ["listing_stores"]}})
    cfg = ST.load_config(cfgp)
    dest = cfg.destination("cluster_a")
    # a change after the fold: s2's bytes move, so the shard has a group to read; the generation attests the listings, s1's empty directory included
    s2 = _store(str(stores), "s2", "legacy_store_2", "none", [0.1, 0.9, 0.5])
    res = ST.mirror_component(root, dest, cfg.components["listing_stores"], runner=TS.Runner())
    assert res.outcome == "published", res.as_dict()
    gen_dir = f"{dest.stores_dir}/gen"
    ix = ST.manifest_index(f"{gen_dir}/listing_stores", res.content_sha8)
    assert ix.attested_listing("s1") == (("killed_empty", "dir"), ("store.json", "file")) and ix.attested_listing("s1/killed_empty") == ()
    assert os.path.isdir(f"{gen_dir}/listing_stores/{res.content_sha8}/s1/killed_empty")
    monkeypatch.setattr("sisteer.registry.adapters.load_all", lambda: {name: lambda: ListingStoreAdapter(str(stores))})
    w = {"root": root, "home": str(home), "stores": str(stores), "ad": ad, "cfg": cfg, "cfgp": cfgp, "dest": dest, "sha8": res.content_sha8,
         "gen_dir": gen_dir, "shared": dest.shared}
    plan, plan_file, info = _plan_and_stage(w, tmp_path, adapters=(name,))
    assert plan["adapters"][name]["route"] == "cluster" and plan["adapters"][name]["pins"] == {"listing_stores": res.content_sha8}, plan["adapters"][name]
    _node(w)
    try:
        assert CR.run_shard(f"{plan['window_root']}/plan.json", 0, home=w["home"]) == 0
        doc = json.load(open(CR.summary_path(plan["summaries_dir"], name, 0, 1)))
        # the shard's admission is the operation's attestation record (O(1) per pin; it carries no listing count yet), so the
        # four attested directories -- the base, s1, s1/killed_empty, s2 -- are read off the pin's attestation the plan carries
        assert doc["ok"] is True and doc["admission_pre"]["listing_stores"]["ok"], doc["admission_pre"]
        lsha = plan["adapters"][name]["attestations"]["listing_stores"]
        assert lsha and ST.read_manifest_attestation(f"{ST.component_state_dir(root, 'listing_stores')}/MANIFEST-{res.content_sha8}.sha256", lsha)["n_directories"] == 4
        assert len(doc["published"]) == 1 and not doc["refused"], (doc["published"], doc["refused"])
        rev_dir, pinfo = next(iter(doc["published"].items()))
        assert pinfo["group"] == "s2"
        d = os.path.join(plan["root"], "records", "adapters", rev_dir)
        states = [rec["payload"] for fn in os.listdir(d) if "source_revision" in fn for _, rec, _ in R.read_record_file(os.path.join(d, fn))
                  if rec["record_type"] == "source_state" and rec["payload"]["json_path"] == "$.entries"]
        assert [p["current_members_json"] for p in states] == [["store.json"]]        # s2's listing, read through the generation, is the canonical's
        assert all(w["gen_dir"] not in p["input_uri"] for p in states)
    finally:
        _back_on_grading_host(w)


# --------------------------------------------------------------------------------------------
# a pin revalidates against the attestation it used (the A -> B -> A reversion)
# --------------------------------------------------------------------------------------------


def _record_types(rev_path):
    out = set()
    for fn in sorted(os.listdir(rev_path)):
        if fn.endswith(".jsonl"):
            out.update(rec["record_type"] for _, rec, _ in R.read_record_file(os.path.join(rev_path, fn)))
    return out


def test_sols_a_b_a_reversion_a_window_pinned_to_the_old_attestation_delivers_no_stale_paired_measurement_and_a_fresh_window_reads_the_new_listing(tmp_path, monkeypatch):
    """A store adapter files a PAIRED measurement for a store only while the store's own `agentic/none` directory exists (the
    shape of em_misalign's paired.csv rule). Generation A is published over listings L1 (the directory present) and a window pins (A, L1);
    its shard reads the A tree and publishes a revision CARRYING the paired measurement. Before the pull the canonical goes A -> B -> A
    with agentic/none removed and both are mirrored: the same content A is re-published under attestation L2. The window's post-run
    revalidation compares the canonical with the attestation it PINNED (L1) and refuses listing_changed; the pull withholds the revision;
    nothing reaches the inbox, the records or the database, and the fold's measurement stands. A fresh window pins (A, L2), its shard admits
    the fresh tree against L2, reads the store without the paired measurement, and its revision folds. Before this change the rebuild of A
    overwrote the grading host's MANIFEST-A.json with L2 (A was not current at the time), revalidate(A) accepted the canonical against
    listings the shard never read, and the stale paired measurement was delivered."""
    from test_registry_source_state_per_revision import ListingStoreAdapter
    from sisteer.registry import identity as I

    class PairedStoreAdapter(ListingStoreAdapter):
        name = "paired_store"

        def read(self, ctx, group, inputs):
            store = next(i for i in inputs if not AB.is_listing_input(i))
            paired = os.path.isdir(os.path.join(os.path.dirname(store.path), "agentic", "none"))
            for batch in super().read(ctx, group, inputs):
                if not paired:          # no unsteered none of its own: the run and its judgment are stated, the paired measurement is not
                    batch.records = [r for r in batch.records if r["record_type"] not in ("run_metric", "item_result")]
                    batch.observation_keys = [(t, k) for t, k in batch.observation_keys if t != "run_metric"]
                    batch.item_keys = []
                yield batch

    root, _unused = _setup(tmp_path)
    home = tmp_path / "home"
    stores = home / "stores"
    stores.mkdir(parents=True)
    _store(str(stores), "s1", "legacy_store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4])
    (stores / "s1" / "agentic" / "none").mkdir(parents=True)
    _store(str(stores), "s2", "legacy_store_2", "none", [0.1, 0.9])
    ad = PairedStoreAdapter(str(stores))
    name, slug = ad.name, "paired_stores"
    with DB.reading(DB.db_path(root)) as conn:
        rep = RUN.run_pass(root, [ad], conn=conn, now=synth.t(2000), producer=PRODUCER)
    assert rep.groups_refreshed == 2 and H.ingest_once(root, workers=0, min_free=0).refused == 0, [r.error for r in rep.results]
    rid = I.run_id(synth.CAMPAIGN, "legacy_store_1", "constitution_lora@aaaa", synth.EVAL_ID, synth.INSTRUMENT_ID)
    fold_value = _rows(root, "SELECT value FROM run_metric WHERE run_id = ? AND metric_id = 'asr'", (rid,))
    assert len(fold_value) == 1
    monkeypatch.setattr(ST, "HOME", str(home))
    cfgp = TS._cfg(tmp_path, {slug: {"base": str(stores), "authority": "grading_host_copy", "origin": "grading_host"}}, adapters={name: {"shards": 1, "components": [slug]}})
    cfg = ST.load_config(cfgp)
    dest = cfg.destination("cluster_a")
    gen_dir = f"{dest.stores_dir}/gen"
    # s1's bytes move (the window has a group to read); generation A is published over listings L1, agentic/none present
    _store(str(stores), "s1", "legacy_store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4, 0.6])
    resA = ST.mirror_component(root, dest, cfg.components[slug], runner=TS.Runner())
    assert resA.outcome == "published", resA.as_dict()
    A = resA.content_sha8
    L1 = json.load(open(f"{gen_dir}/{slug}/VERIFIED-{A}.json"))["listings_sha256"]
    assert ST.published_attestations(root) == {slug: L1}
    monkeypatch.setattr("sisteer.registry.adapters.load_all", lambda: {name: lambda: PairedStoreAdapter(str(stores))})
    w = {"root": root, "home": str(home), "stores": str(stores), "ad": ad, "cfg": cfg, "cfgp": cfgp, "dest": dest, "sha8": A, "gen_dir": gen_dir, "shared": dest.shared}
    plan, plan_file, _ = _plan_and_stage(w, tmp_path, window="wa", adapters=(name,))
    e = plan["adapters"][name]
    assert e["route"] == "cluster" and e["pins"] == {slug: A} and e["attestations"] == {slug: L1} and e["revalidation_pre"][slug]["listings_sha256"] == L1, e
    _node(w)
    try:
        assert CR.run_shard(f"{plan['window_root']}/plan.json", 0, home=w["home"]) == 0
    finally:
        _back_on_grading_host(w)
    doc = json.load(open(CR.summary_path(plan["summaries_dir"], name, 0, 1)))
    # the pin's attestation rides the plan and the summary (the shard's O(1) admission record carries no listings_sha256 yet)
    assert doc["ok"] is True and doc["attestations"] == {slug: L1} and doc["admission_pre"][slug]["ok"] is True, doc["admission_pre"]
    assert plan["adapters"][name]["attestations"] == {slug: L1}
    ((rev_dir, pinfo),) = doc["published"].items()
    assert pinfo["group"] == "s1"
    rev = rev_dir.split("/")[1]
    assert "run_metric" in _record_types(os.path.join(plan["root"], "records", "adapters", rev_dir)), "the shard read the OLD A tree: agentic/none present, the paired measurement filed"
    # the canonical goes A -> B -> A with agentic/none removed, both mirrored, while the window stands finished and unpulled
    _store(str(stores), "s2", "legacy_store_2", "none", [0.1, 0.9, 0.5])
    resB = ST.mirror_component(root, dest, cfg.components[slug], runner=TS.Runner())
    assert resB.outcome == "published" and resB.content_sha8 != A
    _store(str(stores), "s2", "legacy_store_2", "none", [0.1, 0.9])
    shutil.rmtree(stores / "s1" / "agentic")
    resA2 = ST.mirror_component(root, dest, cfg.components[slug], runner=TS.Runner())
    assert resA2.outcome == "published" and resA2.content_sha8 == A and resA2.detail["superseded"].startswith(f"{gen_dir}/{slug}/.superseded-{A}-"), resA2.as_dict()
    L2 = json.load(open(f"{gen_dir}/{slug}/VERIFIED-{A}.json"))["listings_sha256"]
    assert L2 != L1 and ST.published_attestations(root) == {slug: L2}
    sdir = ST.component_state_dir(root, slug)
    assert ST.read_manifest_attestation(f"{sdir}/MANIFEST-{A}.sha256", L1)["listings_sha256"] == L1 and ST.read_manifest_attestation(f"{sdir}/MANIFEST-{A}.sha256", L2)["listings_sha256"] == L2
    # the window's post-run revalidation reads the attestation it pinned: refused by name, the revision withheld, nothing delivered
    assert CR.revalidate_plan(root, plan, cfg) == [name]
    rv = plan["adapters"][name]["revalidation_post"][slug]
    assert rv["refusal"] == "listing_changed" and rv["listings_sha256"] == L1 and rv["listing_mismatched"] == ["s1", "s1/agentic", "s1/agentic/none"], rv
    pull = CR.pull_window(root, plan, str(tmp_path / "expect_a"), source=plan["window_root"])
    assert pull["pulled"] == {} and "post-run revalidation refused" in pull["withheld"][rev_dir] and set(pull["adapters_refused"]) == {name}   # refused by name, read nowhere (S2)
    assert not os.path.exists(os.path.join(H.inbox_dir(root), CR.INBOX_NAME, "adapters", rev_dir)) and H.scan_inbox(root) == []
    ing = H.ingest_once(root, workers=0, min_free=0)
    assert ing.blocked == 0 and not os.path.exists(os.path.join(H.records_dir(root), "adapters", rev_dir))
    assert _rows(root, "SELECT COUNT(*) FROM source_revision WHERE revision_id = ?", (rev,)) == [(0,)]
    assert _rows(root, "SELECT value FROM run_metric WHERE run_id = ? AND metric_id = 'asr'", (rid,)) == fold_value, "the stale paired measurement never landed"
    assert set(CR.refused_adapters(plan, pull)) == {name}
    # the old window's shard, run again now, would refuse at admission: the fresh tree is not the L1 attestation's
    ad1 = ST.admit_generation(f"{gen_dir}/{slug}", A, listings_sha256=L1)
    assert ad1.refusal == "listing_changed" and ad1.listings_sha256 == L1
    # a fresh window pins (A, L2): its shard admits the fresh tree against L2, reads s1 without the paired measurement, and its revision folds
    plan2, plan_file2, _ = _plan_and_stage(w, tmp_path, window="wb", adapters=(name,))
    e2 = plan2["adapters"][name]
    assert e2["route"] == "cluster" and e2["pins"] == {slug: A} and e2["attestations"] == {slug: L2} and e2["revalidation_pre"][slug]["ok"], e2
    _node(w)
    try:
        assert CR.run_shard(f"{plan2['window_root']}/plan.json", 0, home=w["home"]) == 0
    finally:
        _back_on_grading_host(w)
    doc2 = json.load(open(CR.summary_path(plan2["summaries_dir"], name, 0, 1)))
    assert doc2["ok"] is True and doc2["attestations"] == {slug: L2} and doc2["admission_pre"][slug]["ok"] and doc2["admission_post"][slug]["ok"], doc2["admission_pre"]
    ((rev_dir2, pinfo2),) = doc2["published"].items()
    assert pinfo2["group"] == "s1" and rev_dir2 != rev_dir
    types2 = _record_types(os.path.join(plan2["root"], "records", "adapters", rev_dir2))
    assert "run" in types2 and "run_metric" not in types2, types2
    assert CR.revalidate_plan(root, plan2, cfg) == [] and plan2["adapters"][name]["revalidation_post"][slug]["listings_sha256"] == L2
    pull2 = CR.pull_window(root, plan2, str(tmp_path / "expect_b"), source=plan2["window_root"])
    assert rev_dir2 in pull2["pulled"], pull2
    ing2 = H.ingest_once(root, workers=0, min_free=0)
    assert ing2.refused == 0 and ing2.blocked == 0, ing2.refusals
    assert _rows(root, "SELECT accepted FROM source_revision WHERE revision_id = ?", (rev_dir2.split("/")[1],)) == [(1,)]
    assert _rows(root, "SELECT COUNT(*) FROM source_revision WHERE revision_id = ?", (rev,)) == [(0,)]


# --------------------------------------------------------------------------------------------
# Part 9 -- a standing refusal never aborts a window
# --------------------------------------------------------------------------------------------

import hashlib  # noqa: E402

from sisteer.registry import reads_protocol as RPROTO  # noqa: E402


class SyntheticStoreAdapterWords(SyntheticStoreAdapter):
    """The synthetic adapter with its refusal's WORDS as a knob: a standing refusal is compared byte for byte, so a reader whose message
    changed (a fix that says more, a different cause) is a CHANGED refusal, not a standing one."""
    words = "synthetic read failure"

    def read(self, ctx, group, inputs):
        for inp in inputs:
            if inp.uri in self.broken:
                raise AB.AdapterRefusal(inp.uri, self.words)
        return super().read(ctx, group, inputs)


def _standing_window(tmp_path, name: str, rep, cluster_root: str, producer: Dict) -> Dict:
    """A local window root over a cluster pass: the spool IS `cluster_root` (`<window root>/root`), the shard summary and the operation's
    attestation record laid beside it the way the pull receives them. Returns the plan the pull reads."""
    window_root = os.path.dirname(cluster_root)
    CR.write_shard_summary(os.path.join(window_root, ADAPTER, "0-of-1.json"), rep, rep.as_dict(), [ADAPTER], (0, 1), cluster_root, producer)
    CR._write_json(os.path.join(window_root, ST.ATTESTATION_FILE), _attestation_record(name, {}))
    return {"window": name, "host": None, "window_root": window_root, "adapters": {ADAPTER: {"route": "cluster", "shards": 1, "task_indices": [0], "pins": {}}},
            "tasks": [{"adapter": ADAPTER}], "stages": {"revalidate_post": {"stale": []}}, "refusals": {}}


def _standing_cluster_root(tmp_path, name: str) -> str:
    root = str(tmp_path / name / "root")
    H.ensure_layout(root)
    return root


def test_a_refusal_repeated_byte_for_byte_under_the_same_inputs_and_tree_is_a_standing_refusal_and_never_the_adapters(tmp_path):
    """s1's grading-host read REFUSED at emit time and the stamp folded (`_emit_time_refused_s1`: `last_error`, the
    attempt id, the reading tree's code sha). The shard re-reads it under `retry_refused` and the reader refuses with THE SAME WORDS over
    THE SAME INPUTS under THE SAME TREE: a `standing_refusal` result -- `prior_refusal` names the recorded refusal it equals (the window that
    recorded it, its utc, its digest over the words + the input digest + the tree, the attempt id, the tree), it is recorded like any refusal
    (its `_refused/<rev>/` record is listed under `refused_records` and the pull DELIVERS it: merged into records/, folded by the harvester),
    the shard summary lists it under `standing_refusals` with `state = standing_refusal` (and `refused_standing` is unchanged),
    `refused` is empty, the pull counts it (`standing_refusals`) and names it (`report["standing_refusals"]`) but gives the
    adapter NO reason, so `refused_adapters` is empty and the window does not abort. A CHANGED refusal -- other words, another
    tree, other inputs -- is a refusal exactly as before (the adapter refused, `refused by the reader on the cluster`), and the next
    window on the changed tree, having folded the new stamp, finds it standing: the first window on a new tree aborts, the second does not."""
    root, home, stores, ad = _grading_host(tmp_path)
    s1 = os.path.join(stores, "s1", "store.json")
    s1_rev = _emit_time_refused_s1(root, stores, ad)
    code_a = PRODUCER["code_sha"]
    stamp = _rows(root, "SELECT last_error, json_extract(extra_json, '$.attempt_revision_id'), json_extract(extra_json, '$.attempt_code_sha'), "
                        "json_extract(extra_json, '$.attempt_window'), observed_utc FROM source_watermark WHERE input_uri = ?", (s1,))
    assert stamp == [(stamp[0][0], s1_rev, code_a, None, synth.t(2100))] and "synthetic read failure" in stamp[0][0], stamp   # a grading host read names no window
    sidecar = str(tmp_path / "rs.sqlite")
    _export(root, sidecar, home)
    with RS.open_reader_state(sidecar, [ADAPTER]) as conn:
        marks = RUN._watermarks(conn, ADAPTER)
    assert marks[s1].attempt_code_sha == code_a and marks[s1].attempt_revision_id == s1_rev and marks[s1].refused and marks[s1].observed_utc == synth.t(2100)
    assert marks[s1].attempt_window is None
    # (1) the SAME words, inputs and tree: standing -- recorded, counted apart, never a refusal of the adapter
    producer = dict(PRODUCER, substrate="cluster_a", host="node-x")
    cr1 = _standing_cluster_root(tmp_path, "w1")
    rep1 = _cluster_pass(cr1, ad, sidecar, synth.t(3000), retry_refused=True)
    (r1,) = rep1.results
    assert r1.group == "s1" and not r1.accepted and r1.standing_refusal and not r1.refused_standing and r1.revision_id == s1_rev and not r1.refused_known, r1
    digest = RUN.standing_refusal_digest(r1.error, s1_rev, code_a)
    assert digest == hashlib.sha256(json.dumps([r1.error, s1_rev, code_a], ensure_ascii=False).encode("utf-8")).hexdigest()
    assert r1.prior_refusal == {"window": None, "utc": synth.t(2100), "digest": digest, "attempt_revision_id": s1_rev, "code_sha": code_a}   # the grading host's recorded refusal
    assert rep1.groups_refused == 0 and rep1.groups_standing_refusals == 1 and rep1.groups_refused_standing == 0 and rep1.as_dict()["groups_standing_refusals"] == 1
    assert os.path.isdir(RUN.refused_dir(cr1, ADAPTER, s1_rev)) and r1.files and all(f.startswith(RUN.refused_dir(cr1, ADAPTER, s1_rev)) for f in r1.files)
    plan1 = _standing_window(tmp_path, "w1", rep1, cr1, producer)
    doc = json.load(open(os.path.join(plan1["window_root"], ADAPTER, "0-of-1.json")))
    rev_dir = f"{ADAPTER}/{CR.REFUSED_DIR}/{s1_rev}"
    assert doc["refused"] == [] and doc["published"] == {} and doc["ok"] is True and doc["refused_standing"] == []    # the unchanged refused_standing list
    assert doc["standing_refusals"] == [{"adapter": ADAPTER, "group": "s1", "revision_id": s1_rev, "state": "standing_refusal",
                                         "refusal": r1.error, "prior_refusal": r1.prior_refusal}]
    rec = doc["refused_records"][rev_dir]
    assert rec["standing"] is True and rec["marker"] and set(rec["files"]) == set(rec["sha256"]) == set(os.listdir(RUN.refused_dir(cr1, ADAPTER, s1_rev)))
    assert CR.expected_from_summaries(plan1["window_root"])[0][rev_dir]["kind"] == "refused"
    # the pull: no reason for the adapter; the standing entry recorded for the receipt; the record DELIVERED like a revision
    prep = CR.pull_window(root, plan1, str(tmp_path / "expect1"), source=plan1["window_root"])
    s = prep["summaries"][ADAPTER]
    assert s["refused_groups"] == 0 and s["standing_refusals"] == 1 and s["ok"] == 1
    assert ADAPTER not in prep["adapters_refused"] and CR.refused_adapters(plan1, prep) == {}, prep["adapters_refused"]
    assert prep["standing_refusals"] == {ADAPTER: [{"group": "s1", "revision_id": s1_rev, "state": "standing_refusal", "refusal": r1.error,
                                                    "prior_refusal": r1.prior_refusal}]}
    # the receipt a window writes over this pull: the per-adapter `standing_refusals` list the status page reads (the exact shape), whole
    stage = {"summaries": prep["summaries"], "pulled_dirs": sorted(prep["pulled"]), "reactivations": {}, "standing_refusals": prep["standing_refusals"]}
    rec1 = RPROTO.receipt_from_plan(dict(plan1, tree_sha=code_a, destination="cluster_a", stages={"pull": stage}), {}, {ADAPTER: {}}, {"op_id": "w1", "owner": "window"})
    e1 = rec1["adapters"][ADAPTER]
    assert e1["read_whole"] and e1["refused"] is None and e1["refused_standing"] == [] and e1["refusal_records_pulled"] == [rev_dir] and e1["revisions_pulled"] == []
    assert e1["standing_refusals"] == [{"group": "s1", "revision_id": s1_rev, "state": "standing_refusal", "refusal": r1.error,
                                        "prior_refusal": {"window": None, "utc": synth.t(2100), "digest": digest, "attempt_revision_id": s1_rev, "code_sha": code_a}}]
    assert prep["pulled"][rev_dir]["state"] == "complete" and rev_dir in prep["depends"] and prep["quarantined"] == {}
    inbox = os.path.join(H.inbox_dir(root), CR.INBOX_NAME, "adapters", rev_dir)
    assert os.path.isdir(inbox) and any(CR.MARKER_TOKEN in fn for fn in os.listdir(inbox))
    merged = H.scan_inbox(root)
    assert merged and all(st == "merged" for _, st in merged) and not os.path.exists(inbox), merged
    dest = os.path.join(H.records_dir(root), "adapters", rev_dir)
    assert set(rec["files"]) <= set(os.listdir(dest)) and len(os.listdir(dest)) == len(rec["files"]) + 1    # beside the grading host's own emit-time record
    evidence = RPROTO.merge_evidence(root, {rev_dir: prep["depends"][rev_dir]})
    assert evidence[rev_dir]["state"] == "merged" and evidence[rev_dir]["adapter"] == ADAPTER, evidence
    ing = H.ingest_once(root, workers=0, min_free=0)
    assert ing.refused == 0 and ing.blocked == 0, ing.refusals
    # the harvester folded the shard's stamp: still refused, the same attempt and tree, observed at the shard's read, the window that read
    folded = _rows(root, "SELECT observed_utc, json_extract(extra_json, '$.attempt_revision_id'), json_extract(extra_json, '$.attempt_code_sha'), last_error, "
                         "json_extract(extra_json, '$.attempt_window') FROM source_watermark WHERE input_uri = ?", (s1,))
    assert folded == [(synth.t(3000), s1_rev, code_a, r1.error, None)], folded     # _cluster_pass's read_copy names no window; run_shard's does (below, w3)
    assert _rows(root, "SELECT accepted FROM source_revision WHERE revision_id = ?", (s1_rev,)) == [(0,)]
    # (2) CHANGED words over the same inputs and tree: a refusal as today -- the adapter refused by the pull
    ad_w = SyntheticStoreAdapterWords(stores)
    ad_w.broken.add(s1)
    ad_w.words = "synthetic read failure: a second cause named"
    cr2 = _standing_cluster_root(tmp_path, "w2")
    rep2 = _cluster_pass(cr2, ad_w, sidecar, synth.t(3100), retry_refused=True)
    (r2,) = rep2.results
    assert not r2.accepted and not r2.standing_refusal and r2.prior_refusal is None and r2.revision_id == s1_rev and rep2.groups_refused == 1 and rep2.groups_standing_refusals == 0
    plan2 = _standing_window(tmp_path, "w2", rep2, cr2, producer)
    doc2 = json.load(open(os.path.join(plan2["window_root"], ADAPTER, "0-of-1.json")))
    assert [x["group"] for x in doc2["refused"]] == ["s1"] and doc2["standing_refusals"] == [] and doc2["refused_records"][rev_dir]["standing"] is False
    prep2 = CR.pull_window(root, plan2, str(tmp_path / "expect2"), source=plan2["window_root"])
    assert prep2["adapters_refused"][ADAPTER] == ["1 group(s) refused by the reader on the cluster"] and prep2["standing_refusals"] == {}
    assert prep2["pulled"][rev_dir]["state"] == "complete"            # the changed refusal's record travels too: it is what the next window stands on
    # (3) the same words under ANOTHER TREE: changed (the first window on a new tree aborts on every standing refusal)
    cr3 = _standing_cluster_root(tmp_path, "w3")
    code_b = synth.code_sha("b")
    conn = RS.open_reader_state(sidecar, [ADAPTER])
    try:
        cluster = RUN.ClusterRead.from_sidecar(conn, {"substrate": "cluster_a", "host": "node-x", "components": {}, "window": "w3"})   # run_shard's read_copy names the window
        rep3 = RUN.run_pass(cr3, [ad], conn=conn, now=synth.t(3200), producer=dict(producer, code_sha=code_b), cluster=cluster, retry_refused=True, save_cache=False)
    finally:
        conn.close()
    (r3,) = rep3.results
    assert not r3.accepted and not r3.standing_refusal and rep3.groups_refused == 1, r3
    # ... and the SECOND window on that tree, the first one's record folded, finds it standing (the consequence, stated plainly)
    plan3 = _standing_window(tmp_path, "w3", rep3, cr3, dict(producer, code_sha=code_b))
    assert ADAPTER in CR.pull_window(root, plan3, str(tmp_path / "expect3"), source=plan3["window_root"])["adapters_refused"]
    assert H.scan_inbox(root) and H.ingest_once(root, workers=0, min_free=0).refused == 0
    assert _rows(root, "SELECT json_extract(extra_json, '$.attempt_code_sha'), json_extract(extra_json, '$.attempt_window') FROM source_watermark WHERE input_uri = ?", (s1,)) == [(code_b, "w3")]
    sidecar_b = str(tmp_path / "rs-b.sqlite")
    _export(root, sidecar_b, home)
    cr4 = _standing_cluster_root(tmp_path, "w4")
    conn = RS.open_reader_state(sidecar_b, [ADAPTER])
    try:
        cluster = RUN.ClusterRead.from_sidecar(conn, {"substrate": "cluster_a", "host": "node-x", "components": {}})
        rep4 = RUN.run_pass(cr4, [ad], conn=conn, now=synth.t(3300), producer=dict(producer, code_sha=code_b), cluster=cluster, retry_refused=True, save_cache=False)
    finally:
        conn.close()
    (r4,) = rep4.results
    assert r4.standing_refusal and rep4.groups_refused == 0 and rep4.groups_standing_refusals == 1
    assert r4.prior_refusal == {"window": "w3", "utc": synth.t(3200), "digest": RUN.standing_refusal_digest(r4.error, s1_rev, code_b),
                                "attempt_revision_id": s1_rev, "code_sha": code_b}, "the prior refusal is w3's: its window, its stamp, its digest"
    plan4 = _standing_window(tmp_path, "w4", rep4, cr4, dict(producer, code_sha=code_b))
    assert ADAPTER not in CR.pull_window(root, plan4, str(tmp_path / "expect4"), source=plan4["window_root"])["adapters_refused"]
    # (4) the same words and tree over CHANGED inputs (new bytes in s1): the attempt id differs -- changed, refused
    _store(str(stores), "s1", "legacy_store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4, 0.9, 0.1])
    cr5 = _standing_cluster_root(tmp_path, "w5")
    conn = RS.open_reader_state(sidecar_b, [ADAPTER])
    try:
        cluster = RUN.ClusterRead.from_sidecar(conn, {"substrate": "cluster_a", "host": "node-x", "components": {}})
        rep5 = RUN.run_pass(cr5, [ad], conn=conn, now=synth.t(3400), producer=dict(producer, code_sha=code_b), cluster=cluster, retry_refused=True, save_cache=False)
    finally:
        conn.close()
    (r5,) = rep5.results
    assert not r5.accepted and not r5.standing_refusal and r5.revision_id != s1_rev and rep5.groups_refused == 1
    # the pure rule, on its own terms: every term must agree, and a stamp without one (a stamp from before the fields) never stands
    inputs_s1 = [i for i in ad.discover() if i.uri == s1]
    assert RUN.standing_refusal_of(marks, [], s1_rev, r1.error, code_a) is None                        # no inputs
    assert RUN.standing_refusal_of({}, inputs_s1, s1_rev, r1.error, code_a) is None                    # no stamp
    assert RUN.standing_refusal_of(marks, inputs_s1, s1_rev, r1.error, "") is None                     # no tree named
    assert RUN.standing_refusal_of(marks, inputs_s1, s1_rev, None, code_a) is None                     # no words
    assert RUN.standing_refusal_of(marks, inputs_s1, s1_rev, r1.error + " ", code_a) is None           # one byte more: changed
    old_stamp = {s1: RUN.Watermark(marks[s1].source_id, marks[s1].stat, marks[s1].last_error, s1_rev, None)}
    assert RUN.standing_refusal_of(old_stamp, inputs_s1, s1_rev, r1.error, code_a) is None             # a stamp written before the tree field
    # the grading host's own pass never classifies a re-read standing (the rule is the shard's; the grading host reads no store under C6)
    with DB.reading(DB.db_path(root)) as conn:
        rep_d = RUN.run_pass(root, [ad], conn=conn, now=synth.t(3500), producer=PRODUCER, retry_refused=True)
    assert all(not r.standing_refusal and r.prior_refusal is None for r in rep_d.results)


# --------------------------------------------------------------------------------------------
# Part 10 -- the plan-time checks on the constructed adapter instance
# --------------------------------------------------------------------------------------------


def test_plan_time_checks_refuse_a_traversal_adapter_planned_as_more_than_one_shard_and_an_uncovered_or_unnamed_root_by_name(tmp_path, monkeypatch):
    """With a registry of factories the plan CONSTRUCTS each adapter (never discovers it; `stores coverage`'s own rule) and
    refuses by name, at plan time: (a) an adapter declaring `rng_scope = reader_traversal` whose entry says more than one shard (the shard
    runner admits K = 1 only; at 8 / 4 / 4 every task of three adapters crashed before a summary was written) -- with `shards: 1` it
    routes; (b) a declared root no component holds, or one held by a component the entry does not name (the plan pins generations from the
    entry alone); (c) an adapter the registry lacks, or one whose construction fails. Without a registry (the default) nothing is
    constructed, as the no-discovery tests of the plan pin; the CLI's `plan` and the tick pass `adapters.load_all()`."""
    w = _world_with_generation(tmp_path, monkeypatch)
    stores = w["stores"]
    # (a) the RNG rule outranks the shard count: K = 2 in the entry, reader_traversal on the instance -> refused by name; K = 1 -> cluster

    def traversal():
        ad = SyntheticStoreAdapter(stores)
        ad.rng_scope = "reader_traversal"
        return ad
    plan = CR.plan_window(w["root"], "wt2", [ADAPTER], w["cfg"], w["dest"], "a" * 40, uri_home=w["home"], repo_root=REPO_ROOT, registry={ADAPTER: traversal})
    e = plan["adapters"][ADAPTER]
    assert e["route"] == "grading_host" and plan["tasks"] == [] and e["plan_checks"]["rng_scope"] == "reader_traversal" and e["plan_checks"]["shards"] == 2
    assert "rng_scope = reader_traversal" in e["reason"] and "2 shards" in e["reason"] and "exactly ONE shard" in e["reason"] and "shards: 1" in e["reason"], e["reason"]
    assert plan["refusals"] == {ADAPTER: [f"plan: {e['reason']}"]} and e["pins"] == {"synthetic_stores": w["sha8"]} and e["revalidation_pre"]["synthetic_stores"]["ok"] is True   # after the pins and the admission, before the route
    cfg1 = ST.load_config(TS._cfg(tmp_path, {"synthetic_stores": {"base": stores, "authority": "grading_host_copy", "origin": "grading_host"}},
                                  adapters={ADAPTER: {"shards": 1, "components": ["synthetic_stores"]}}, name="one-shard.yaml"))
    plan1 = CR.plan_window(w["root"], "wt1", [ADAPTER], cfg1, cfg1.destination("cluster_a"), "a" * 40, uri_home=w["home"], repo_root=REPO_ROOT,
                           registry={ADAPTER: traversal}, published={"synthetic_stores": w["sha8"]})
    e1 = plan1["adapters"][ADAPTER]
    assert e1["route"] == "cluster" and len(plan1["tasks"]) == 1 and e1["plan_checks"]["refusal"] is None and e1["plan_checks"]["rng_scope"] == "reader_traversal"
    assert [r["state"] for r in e1["plan_checks"]["roots"]] == [ST.COVERED] and e1["plan_checks"]["roots"][0]["components"] == ["synthetic_stores"]
    # (b) a root no component holds: refused by name, the root named; a root held by a component the entry does not name: refused, the holder named
    anchors = os.path.join(w["home"], "anchors")
    os.makedirs(anchors, exist_ok=True)
    plan_u = CR.plan_window(w["root"], "wu", [ADAPTER], w["cfg"], w["dest"], "a" * 40, uri_home=w["home"], repo_root=REPO_ROOT,
                            registry={ADAPTER: lambda: TwoRootAdapter(stores, anchors)})
    eu = plan_u["adapters"][ADAPTER]
    assert eu["route"] == "grading_host" and eu["reason"].startswith(CR.UNCOVERED_PREFIX) and anchors in eu["reason"] and "no component holds" in eu["reason"], eu["reason"]
    assert eu["plan_checks"]["uncovered_roots"] == [anchors] and eu["plan_checks"]["missing_components"] == [] and "read nowhere on the node" in eu["reason"]
    cfg_h = ST.load_config(TS._cfg(tmp_path, {"synthetic_stores": {"base": stores, "authority": "grading_host_copy", "origin": "grading_host"},
                                              "anchor_trees": {"base": anchors, "authority": "grading_host_copy", "origin": "grading_host"}},
                                   adapters={ADAPTER: {"shards": 2, "components": ["synthetic_stores"]}}, name="holder.yaml"))
    plan_h = CR.plan_window(w["root"], "wh", [ADAPTER], cfg_h, cfg_h.destination("cluster_a"), "a" * 40, uri_home=w["home"], repo_root=REPO_ROOT,
                            registry={ADAPTER: lambda: TwoRootAdapter(stores, anchors)}, published={"synthetic_stores": w["sha8"]})
    eh = plan_h["adapters"][ADAPTER]
    assert eh["route"] == "grading_host" and eh["plan_checks"]["missing_components"] == ["anchor_trees"] and eh["plan_checks"]["uncovered_roots"] == []
    assert "held by component(s) the entry does not name: anchor_trees" in eh["reason"], eh["reason"]
    # with the holder named in the entry the same adapter passes the checks (its two roots both covered; the pins then decide the route)
    cfg_ok = ST.load_config(TS._cfg(tmp_path, {"synthetic_stores": {"base": stores, "authority": "grading_host_copy", "origin": "grading_host"},
                                               "anchor_trees": {"base": anchors, "authority": "grading_host_copy", "origin": "grading_host"}},
                                    adapters={ADAPTER: {"shards": 2, "components": ["synthetic_stores", "anchor_trees"]}}, name="holder-ok.yaml"))
    ok = CR.plan_time_checks(ADAPTER, cfg_ok.adapters[ADAPTER], cfg_ok, {ADAPTER: lambda: TwoRootAdapter(stores, anchors)}, w["home"])
    assert ok["refusal"] is None and ok["missing_components"] == [] and ok["uncovered_roots"] == [] and sorted(r["components"][0] for r in ok["roots"]) == ["anchor_trees", "synthetic_stores"]
    plan_ok = CR.plan_window(w["root"], "wok", [ADAPTER], cfg_ok, cfg_ok.destination("cluster_a"), "a" * 40, uri_home=w["home"], repo_root=REPO_ROOT,
                             registry={ADAPTER: lambda: TwoRootAdapter(stores, anchors)}, published={"synthetic_stores": w["sha8"]})
    e_ok = plan_ok["adapters"][ADAPTER]
    assert "plan_checks" not in e_ok and e_ok["reason"] == "no published generation for component(s): anchor_trees", e_ok["reason"]   # the pins refuse first; the checks run last, before the route
    # (c) not registered, or a construction that fails: refused by name
    plan_n = CR.plan_window(w["root"], "wn", [ADAPTER], w["cfg"], w["dest"], "a" * 40, uri_home=w["home"], repo_root=REPO_ROOT, registry={})
    assert plan_n["adapters"][ADAPTER]["route"] == "grading_host" and "not registered in this tree" in plan_n["adapters"][ADAPTER]["reason"]

    def broken_factory():
        raise RuntimeError("synthetic construction failure")
    plan_b = CR.plan_window(w["root"], "wb", [ADAPTER], w["cfg"], w["dest"], "a" * 40, uri_home=w["home"], repo_root=REPO_ROOT, registry={ADAPTER: broken_factory})
    eb = plan_b["adapters"][ADAPTER]
    assert eb["route"] == "grading_host" and eb["plan_checks"]["construction_error"] == "RuntimeError: synthetic construction failure" and "could not be constructed" in eb["reason"]
    # the CLI's plan passes the registry (here the world's patched load_all): the checks are recorded on the routed entry; the roots are
    # judged under stores.HOME (the home Component.home_rel is relative to; the world's here, the grading host's live)
    args = argparse.Namespace(action="plan", window="wcli", adapter=[ADAPTER], tree_sha="a" * 40, dest="cluster_a", config=w["cfgp"],
                              repo_root=REPO_ROOT, full=False, full_adapter=[], reread_force_adapter=[], json=False)
    assert CR.cmd_cluster_reads(args, w["root"]) == 0
    recorded = CR._load_json(CR.plan_path(w["root"], "wcli"))["adapters"][ADAPTER]
    assert recorded["route"] == "cluster" and recorded["plan_checks"]["rng_scope"] == "none" and recorded["plan_checks"]["refusal"] is None
    # the pure function, on the live declaration shape
    checks = CR.plan_time_checks(ADAPTER, w["cfg"].adapters[ADAPTER], w["cfg"], {ADAPTER: lambda: SyntheticStoreAdapter(stores)}, w["home"])
    assert checks["refusal"] is None and checks["shards"] == 2 and checks["rng_scope"] == "none" and checks["missing_components"] == []
