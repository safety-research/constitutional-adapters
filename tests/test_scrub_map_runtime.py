"""``sisteer.scrub_map``, the runtime reader of ``configs/SCRUB_MAP.tsv``, and the verifiers that consult it.

A row certifies that today's file at ``path`` (``post_sha256``) is the declared-rules rewrite of the bytes a record
captured (``pre_sha256``). A verifier that finds a sha mismatch passes as ``verified_via_map`` only when a row certifies
that exact pair at that path; any other mismatch refuses as it always did. Each verifier is shown passing through a row
and refusing without one (no map, a row for another path, a row mapping to other bytes). The cluster reads' stage step
is the exception: a plan binds the exact config bytes its tree read, so it refuses a certified rewrite too and names it.
"""
from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path
from typing import Optional

import pytest

from sisteer import scrub_map as SM

REPO = Path(__file__).resolve().parents[1]
RULES = "f" * 64
HEADER = "\t".join(SM.COLUMNS)


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _map(path: Path, rows) -> Path:
    path.write_text("# a comment line\n" + HEADER + "\n" + "".join("\t".join(r) + "\n" for r in rows))
    return path


@pytest.fixture
def scrub_map(tmp_path, monkeypatch):
    """A map file the default lookup reads ($SISTEER_SCRUB_MAP); call it with rows to (re)write it."""
    p = tmp_path / "SCRUB_MAP.tsv"
    monkeypatch.setenv("SISTEER_SCRUB_MAP", str(p))
    SM._CACHE.clear()

    def write(rows):
        _map(p, rows)
        SM._CACHE.clear()
        return p
    yield write
    SM._CACHE.clear()


PRE = b"model: m\nprompts: /old/site/root/exp/benign.json\n"
POST = b"model: m\nprompts: $SHARED/exp/benign.json\n"
OTHER = b"model: m\nprompts: $SHARED/exp/benign.json\nmax_new_tokens: 9999\n"


# --- the reader ----------------------------------------------------------------------------------------------------------

def test_check_is_exact_on_equal_shas_via_map_on_a_certified_pair_and_none_otherwise(scrub_map):
    scrub_map([(_sha(PRE), "configs/a.yaml", "configs/a.yaml", _sha(POST), RULES, "config")])
    assert SM.check(_sha(POST), _sha(POST)) == SM.EXACT
    assert SM.check(_sha(PRE), _sha(POST), path="configs/a.yaml") == SM.VIA_MAP
    assert SM.check(_sha(PRE).upper(), _sha(POST), path="/any/checkout/configs/a.yaml") == SM.VIA_MAP
    assert SM.check(_sha(PRE), _sha(OTHER), path="configs/a.yaml") is None          # the row maps to other bytes
    assert SM.check(_sha(PRE), _sha(POST), path="configs/b.yaml") is None           # the row names another file
    assert SM.check(_sha(POST), _sha(PRE), path="configs/a.yaml") is None           # never backwards
    assert SM.check("", _sha(POST)) is None and SM.check(None, _sha(POST)) is None
    assert SM.check_prefix(_sha(PRE)[:12], _sha(POST), path="configs/a.yaml") == SM.VIA_MAP
    assert SM.check_prefix(_sha(POST)[:12], _sha(POST)) == SM.EXACT
    assert SM.check_prefix(_sha(PRE)[:7], _sha(POST)) is None                       # shorter than MIN_PREFIX certifies nothing
    assert "maps the recorded sha to" in SM.mismatch_note(_sha(PRE), _sha(OTHER))
    assert "no scrub-map row" in SM.mismatch_note(_sha(OTHER), _sha(PRE))


def test_an_absent_map_is_empty_so_every_mismatch_refuses(tmp_path, monkeypatch):
    monkeypatch.setenv("SISTEER_SCRUB_MAP", str(tmp_path / "none.tsv"))
    SM._CACHE.clear()
    assert len(SM.load()) == 0
    assert SM.check(_sha(PRE), _sha(POST)) is None and SM.check(_sha(POST), _sha(POST)) == SM.EXACT


def test_the_default_map_is_the_checkouts_own(monkeypatch, tmp_path):
    monkeypatch.delenv("SISTEER_SCRUB_MAP", raising=False)
    assert SM.default_path() == REPO / "configs" / "SCRUB_MAP.tsv"
    monkeypatch.setenv("SISTEER_SCRUB_MAP", str(tmp_path / "m.tsv"))
    assert SM.default_path() == tmp_path / "m.tsv"


@pytest.mark.parametrize("body,match", [
    ("pre_sha256\tpath\n", "the header must be"),
    (HEADER + "\n" + "\t".join(["x" * 64, "a", "a", "b" * 64, RULES, "config"]) + "\n", "is not 64 lowercase hex"),
    (HEADER + "\n" + "\t".join(["a" * 64, "a", "a", "b" * 64, RULES, "weird"]) + "\n", "kind 'weird'"),
    (HEADER + "\n" + "\t".join(["a" * 64, "a", "a", "a" * 64, RULES, "config"]) + "\n", "pre_sha256 equals post_sha256"),
    (HEADER + "\n" + "\t".join(["a" * 64, "a", "a", "b" * 64, RULES]) + "\n", "5 cells, expected 6"),
    (HEADER + "\n" + "\t".join(["a" * 64, "", "a", "b" * 64, RULES, "config"]) + "\n", "an empty path"),
])
def test_a_malformed_map_raises_naming_the_line(body, match):
    with pytest.raises(SM.ScrubMapError, match=match):
        SM.parse(body, "m.tsv")


def test_the_map_the_release_tool_writes_parses_here():
    """The writer's layout (comment lines, the six-column header, tab-separated rows) is this reader's."""
    text = ("# Files a record names by sha256 that the release rewrote by declared rules only.\n# rule_set " + RULES + "\n"
            + HEADER + "\n" + "\t".join([_sha(PRE), "configs/a.yaml", "configs/a.yaml", _sha(POST), RULES, "frozen"]) + "\n")
    m = SM.parse(text)
    assert len(m) == 1 and m.rows[0].kind == "frozen" and m.certifies(_sha(PRE), _sha(POST), path="configs/a.yaml")


def test_every_row_of_the_committed_map_names_todays_bytes_at_its_path():
    """A row certifies the file at ``path`` as it is in this checkout. A mapped file edited after the map was certified
    (a comment header rewritten, a value re-pinned) or deleted leaves a row for bytes no checkout holds, so the recorded
    sha refuses again. The fix is to declare the edit and re-derive the map with ``scripts/release/scrub_map.py``,
    never to edit a hash in the map by hand."""
    m = SM.parse((REPO / SM.REL_PATH).read_text(encoding="utf-8"), str(SM.REL_PATH))
    assert len(m) > 0
    stale = []
    for r in m.rows:
        p = REPO / r.path
        got = _sha(p.read_bytes()) if p.is_file() else "absent"
        if got != r.post_sha256:
            stale.append(f"{r.path}: the map certifies {r.post_sha256[:12]}, the checkout holds {got[:12]}")
    assert not stale, "rows of configs/SCRUB_MAP.tsv that no longer describe this checkout:\n" + "\n".join(stale)


# --- verifier: the cap-4096 builder's served operating point -----------------------------------------------------------

BUILDER = REPO / "scripts" / "r2_export" / "cap4096" / "build_transfer_stats_cap4096.py"


def _served_operating_point_of():
    for node in ast.parse(BUILDER.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name == "served_operating_point_of":
            ns = {"Path": Path, "hashlib": hashlib, "ARM_OPERATING_POINT": {("v", "alphasteer"): ("ladder_canon/lam3", "rung 3", "")}}
            exec(compile(ast.Module([node], []), str(BUILDER), "exec"), ns)  # noqa: S102
            return ns["served_operating_point_of"]
    raise AssertionError("served_operating_point_of not found in the builder")


def _arm_config(tmp_path, body: bytes) -> Path:
    p = tmp_path / "wt" / "configs" / "arm_alphasteer.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(body)
    return p


ARM_PRE = b"# a header naming a host\nalphasteer:\n  rung_i: 3\n  lambda: 0.1\n  object_sha256: abc\n"
ARM_POST = b"# a header\nalphasteer:\n  rung_i: 3\n  lambda: 0.1\n  object_sha256: abc\n"


def test_the_builder_passes_a_rewritten_config_through_its_row_and_refuses_without_one(tmp_path, scrub_map):
    fn = _served_operating_point_of()
    p = _arm_config(tmp_path, ARM_POST)
    gp = {"config": {"path": str(p), "sha256": _sha(ARM_PRE)}}
    scrub_map([])
    with pytest.raises(SystemExit, match=r"REFUSING: .* != the sha the grader recorded .*no scrub-map row"):
        fn(gp, "alphasteer", "v")
    scrub_map([(_sha(ARM_PRE), "configs/arm_alphasteer.yaml", "configs/other.yaml", _sha(ARM_POST), RULES, "config")])
    with pytest.raises(SystemExit, match="REFUSING"):
        fn(gp, "alphasteer", "v")
    scrub_map([(_sha(ARM_PRE), "configs/arm_alphasteer.yaml", "configs/arm_alphasteer.yaml", _sha(ARM_POST), RULES, "config")])
    out = fn(gp, "alphasteer", "v")
    assert out["config_sha_check"] == SM.VIA_MAP and out["status"] == "verified" and out["served_id"] == "ladder_canon/lam3"
    p.write_bytes(ARM_POST.replace(b"rung_i: 3", b"rung_i: 4"))                          # any other edit still refuses
    with pytest.raises(SystemExit, match="REFUSING"):
        fn(gp, "alphasteer", "v")
    p.write_bytes(ARM_PRE)                                                                  # equal bytes need no map
    assert "config_sha_check" not in fn(gp, "alphasteer", "v")


# --- the cluster reads' stage step: no map pass ------------------------------------------------------------------------

def test_the_reads_stage_refuses_a_config_rewritten_since_the_plan_and_names_a_certified_rewrite(tmp_path, scrub_map):
    """The stage step is the one consumer that does not pass a certified row: a plan binds the exact bytes its tree read and every
    shard compares the staged copy with them, so a rewritten config refuses at staging, before anything is laid out, with the row
    named and a new plan asked for."""
    from sisteer.registry import cluster_reads as CR

    cfg = tmp_path / "configs" / "registry" / "cluster_reads.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_bytes(POST)
    plan = {"host": "local", "window_root": str(tmp_path / "w"), "root": str(tmp_path / "w" / "root"),
            "rulings_mirror": str(tmp_path / "mirror"), "config": str(cfg), "config_sha256": _sha(PRE)}
    calls = []

    def runner(argv, **kw):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 1, "", "stub: stop after the sha step")

    def stage():
        CR.stage_window(plan, str(tmp_path / "s.sqlite"), str(tmp_path / "plan.json"), runner=runner)

    scrub_map([])
    with pytest.raises(CR.ClusterReadsError, match="changed since the plan was built.*no scrub-map row.*plan again"):
        stage()
    scrub_map([(_sha(PRE), "configs/registry/cluster_reads.yaml", "configs/registry/cluster_reads.yaml", _sha(POST), RULES, "registry_seed")])
    with pytest.raises(CR.ClusterReadsError, match="certified scrub-map rewrite.*plan again from the rewritten file"):
        stage()
    cfg.write_bytes(OTHER)
    with pytest.raises(CR.ClusterReadsError, match="changed since the plan was built.*not " + _sha(OTHER)[:12]):
        stage()
    assert calls == []                                                                      # nothing was laid out for any of them
    plan["config_sha256"] = _sha(OTHER)                                                     # a plan made from today's bytes
    with pytest.raises(CR.ClusterReadsError, match="laying out"):
        stage()
    assert len(calls) == 1                                                                  # past the sha step, into the layout


# --- verifier: the serving proxy's adapter block ------------------------------------------------------------------------

def test_the_adapter_block_check_accepts_a_rewritten_adapter_config_but_never_other_weights(tmp_path, scrub_map):
    from sisteer.capeval.serving import proxy as P

    d = tmp_path / "bank" / "cells" / "c1"
    d.mkdir(parents=True)
    cfg_pre, cfg_post = json.dumps({"r": 4, "path": "/old/site/root/base"}).encode(), json.dumps({"r": 4, "path": "$HF_HOME/base"}).encode()
    (d / P.ADAPTER_CONFIG_FILE).write_bytes(cfg_post)
    (d / P.ADAPTER_WEIGHTS_FILE).write_bytes(b"weights")
    block = {"name": "c1", "dir": str(d), "config_sha256": _sha(cfg_pre), "weights_sha256": _sha(b"weights"), "r": 8,
             "petri_intervention_hash": None, "petri_zero_adapter": False, "petri_source_kind": "lora"}
    scrub_map([])
    with pytest.raises(P.ProxyError, match="hashes to"):
        P.verify_adapter_block_on_disk(block)
    rel = f"bank/cells/c1/{P.ADAPTER_CONFIG_FILE}"
    scrub_map([(_sha(cfg_pre), rel, rel, _sha(cfg_post), RULES, "config")])
    with pytest.raises(P.ProxyError, match=r"r=4 != the block's r=8"):                      # past both sha checks, at the next one
        P.verify_adapter_block_on_disk(block)
    with pytest.raises(P.ProxyError, match="weights_sha256"):
        P.verify_adapter_block_on_disk(dict(block, weights_sha256=_sha(b"other weights")))


# --- verifier: the closer's expected goal ids ---------------------------------------------------------------------------

def test_the_closers_goal_id_cross_check_accepts_a_config_rewritten_by_rule(tmp_path, scrub_map, monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "scripts" / "sdfbase" / "full22"))
    import registry_close_stores as RC
    goals = [types.SimpleNamespace(goal_id=g) for g in ("g1", "g2", "g3")]
    fake = types.ModuleType("jailbreak_run")
    fake._load_cfg = lambda path: {"datasets": {}}
    fake._eval_goals = lambda cfg, confirmatory=True: goals
    monkeypatch.setitem(sys.modules, "jailbreak_run", fake)
    repo = tmp_path / "repo"
    (repo / "configs").mkdir(parents=True)
    (repo / "configs" / "wave.yaml").write_bytes(POST)
    ident = RC.resolver_identity(str(repo))
    ids = ["g1", "g2", "g3"]
    doc = {"config": "configs/wave.yaml", "config_sha256": _sha(PRE), "goal_ids": ids, "goal_ids_sha256": RC._ids_sha256(ids), **ident}
    ids_file = tmp_path / "ids.json"
    ids_file.write_text(json.dumps(doc))
    run = {"plan": {"extra": {"config": "configs/wave.yaml"}}}
    scrub_map([])
    why, got, _prov = RC.multiturn_expected_goal_ids(run, repo_root=str(repo), ids_file=str(ids_file))
    assert why and "certifies config" in why and got is None
    scrub_map([(_sha(PRE), "configs/wave.yaml", "configs/wave.yaml", _sha(POST), RULES, "config")])
    why, got, prov = RC.multiturn_expected_goal_ids(run, repo_root=str(repo), ids_file=str(ids_file))
    assert why is None and got == ids and prov["cross_check"]["config_sha_check"] == SM.VIA_MAP
    ids_file.write_text(json.dumps(dict(doc, config="configs/elsewhere.yaml")))           # another config path is never certified
    why, _got, _prov = RC.multiturn_expected_goal_ids(run, repo_root=str(repo), ids_file=str(ids_file))
    assert why and "certifies config" in why


# --- the frame reader --------------------------------------------------------------------------------------------------

def test_a_frame_rows_config_verifies_against_the_checkout(tmp_path):
    repo = tmp_path / "repo"
    (repo / "configs").mkdir(parents=True)
    (repo / "configs" / "a.yaml").write_bytes(POST)
    _map(repo / "configs" / "SCRUB_MAP.tsv", [(_sha(PRE), "configs/a.yaml", "configs/a.yaml", _sha(POST), RULES, "config")])
    SM._CACHE.clear()
    v = SM.verify_frame_config
    assert v("repo:configs/a.yaml", _sha(POST), repo_root=repo)["status"] == SM.FRAME_EXACT
    got = v("repo:configs/a.yaml", _sha(PRE), repo_root=repo)
    assert got["status"] == SM.FRAME_VIA_MAP == SM.VIA_MAP and got["sha256"] == _sha(POST) and got["path"] == "configs/a.yaml"
    assert v("configs/a.yaml", _sha(PRE), repo_root=repo)["status"] == SM.FRAME_VIA_MAP
    assert v("repo:configs/a.yaml", _sha(OTHER), repo_root=repo)["status"] == SM.FRAME_MISMATCH
    assert v("repo:configs/gone.yaml", _sha(PRE), repo_root=repo)["status"] == SM.FRAME_ABSENT
    for ref in ("sha256:" + _sha(PRE), "/abs/configs/a.yaml", "repo:../outside.yaml", None, ""):
        assert v(ref, _sha(PRE), repo_root=repo)["status"] == SM.FRAME_UNRESOLVABLE
    assert v("repo:configs/a.yaml", None, repo_root=repo)["status"] == SM.FRAME_UNRESOLVABLE
