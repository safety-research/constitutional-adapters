"""scripts/modal/fst_multiturn.py — experiment 2's Modal run script, CODE ONLY:
the driver argv per cell, the standup argv, the
app-name rule, the cells default, the canonical run identity and its
idempotent write / refusal (revision 2 S2: four tests), the seed-only-with-a-
driver-tree rule, the attestation check against the record, and
``progress_counts`` reporting ``refused`` as its own count; the S-C1 rule (a
run holding a driver tree REQUIRES ``--resume``) and the S-C2 per-run attempt
lease, each at the prelude AND through the real ``_run_exp2_impl`` on CPU; and
the staging tree check (``verify_checkout``) under git's "dubious ownership"
refusal — the container's view of a tree tarred under this machine's uid and
unpacked by root — driven with git's own switch for it, plus the legible
``why`` and the image's global safe.directory. The module is imported under
THIS file's stub ``modal`` (the test venv has none), installed for the import
and restored after it — so whichever module's stub an earlier test left in
``sys.modules`` (the suite's stubs install with ``setdefault``, first one wins)
cannot change what these tests see; nothing is launched. Synthetic ids and
digests under tmp_path.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
DRIVER = REPO / "scripts" / "modal" / "fst_multiturn.py"
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))


class _Chain:
    """The image chain. Every ``.run_commands(...)`` line it is handed goes,
    in order, into the list its stub was built with — so a test can pin what
    the IMAGE does without a source-text grep, reading a list IT owns."""

    def __init__(self, run_commands_seen: list[str] | None = None):
        self._seen = run_commands_seen if run_commands_seen is not None else []

    def run_commands(self, *cmds, **_k):
        self._seen.extend(cmds)
        return self

    def __getattr__(self, _name):
        return lambda *a, **k: (self if not (a and callable(a[0]) and len(a) == 1 and not k) else a[0])

    def __call__(self, *a, **k):
        if a and callable(a[0]) and len(a) == 1 and not k:
            return a[0]
        return self


class _App:
    def __init__(self, name=None, **k):
        self.name = name or "fst-multiturn"
        self.description = None
        self.app_id = "ap-test"

    def function(self, **kw):
        return lambda f: f

    def local_entrypoint(self, **kw):
        return lambda f: f


def _stub_modal(run_commands_seen: list[str] | None = None) -> types.ModuleType:
    """A ``modal`` for the import. ``run_commands_seen``, when given, receives
    every ``.run_commands`` line the image chain issues."""
    m = types.ModuleType("modal")
    m.App = _App
    m.Volume = types.SimpleNamespace(from_name=lambda *a, **k: types.SimpleNamespace(
        commit=lambda: None, reload=lambda: None))
    m.Secret = types.SimpleNamespace(from_dict=lambda d: d, from_name=lambda n: n)
    m.Image = types.SimpleNamespace(from_registry=lambda *a, **k: _Chain(run_commands_seen))
    m.is_local = lambda: True
    return m


_ABSENT = object()


def load_driver(run_commands_seen: list[str] | None = None):
    """Import ``scripts/modal/fst_multiturn.py`` under THIS file's stub ``modal``
    (nothing is launched). Also used by ``test_fst_exp2_analyze.py`` to run the
    REAL identity builder against the REAL analyzer check.

    The stub is installed for the import and ``sys.modules["modal"]`` is put
    back afterwards — NOT ``setdefault``: in the whole suite an earlier module's
    stub is already there (``test_attacker_pool.py`` and friends install
    ``test_grokmt_served_resume``'s), and under that one the driver's image chain
    reported to a chain class that is not ours, so what THIS file's stub recorded
    at import was empty. ``run_commands_seen``, when given, receives every
    ``.run_commands`` line the image chain issued during the import."""
    spec = importlib.util.spec_from_file_location("fst_multiturn_under_test", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    prior = sys.modules.get("modal", _ABSENT)
    sys.modules["modal"] = _stub_modal(run_commands_seen)
    try:
        spec.loader.exec_module(mod)
    finally:
        if prior is _ABSENT:
            sys.modules.pop("modal", None)
        else:
            sys.modules["modal"] = prior
    return mod


@pytest.fixture(scope="module")
def drv():
    return load_driver()


TT = "safety_refusal__psr_kl_all"
H = "si_safety__psr_kl_all"
SHA = "a" * 40
TRACKED_IDS = REPO / "configs" / "fst" / "fst_exp2_goal_ids.txt"


def _canonical_ids_bytes(goal_ids) -> bytes:
    return ("\n".join(goal_ids) + "\n").encode()


def _geom(**over):
    g = dict(eval="multiturn_attacks:crescendo", model_key="gemma4e4b", max_model_len=9984,
             max_num_seqs=39, window_source="table", seqs_source="model_budget",
             overridden=False, over_budget=False, is_identity=True)
    g.update(over)
    return SimpleNamespace(**g)


def _record(drv, **over):
    per_cell = {
        "none": {"tt_cell_id": "none", "intervention_hash": "n" * 16, "artifact_sha256": None,
                 "serve_declaration": {}, "positions": None},
        H: {"tt_cell_id": TT, "intervention_hash": "1" * 16, "artifact_sha256": "ab" * 32,
            "serve_declaration": {}, "positions": "response"},
        f"{H}@all_turns": {"tt_cell_id": f"{TT}@all_turns", "intervention_hash": "2" * 16,
                           "artifact_sha256": "ab" * 32,
                           "serve_declaration": {"serve_experiment": "fit_response_serve_all_turns"},
                           "positions": "response_all_turns"},
        f"{H}@all_positions": {"tt_cell_id": f"{TT}@all_positions", "intervention_hash": "3" * 16,
                               "artifact_sha256": "ab" * 32,
                               "serve_declaration": {"serve_experiment":
                                                     "fit_response_serve_all_positions"},
                               "positions": "all"},
    }
    kw = dict(sha=SHA, goal_ids=["g1", "g2", "g3"], cells=list(drv.DEFAULT_CELLS), subshards=5,
              per_cell=per_cell, control_legs={"none__lens0": {"intervention_hash": "t" * 16}},
              digests_by_cell={H: "ab" * 32, f"{H}@all_turns": "ab" * 32,
                               f"{H}@all_positions": "ab" * 32},
              geom_dict=drv.geometry_record(_geom()),
              attack={"name": "crescendo", "max_turns": 10, "max_backtracks": 10,
                      "attacker_temperature": 0.5, "target_max_new_tokens": 2048,
                      "forward_cap": 150},
              target={"model": "org/model", "revision": "r"})
    kw.update(over)
    # the ids FILE's bytes default to a commented file carrying exactly kw's ids
    kw.setdefault("goal_ids_file_bytes",
                  b"# synthetic ids file\n" + _canonical_ids_bytes(kw["goal_ids"]))
    return drv.build_run_identity(**kw)


# --------------------------------------------------------------------------- #
# cells, argv, app name                                                        #
# --------------------------------------------------------------------------- #

def test_cells_default_and_constitution(drv):
    assert drv.parse_cells("") == ["none", TT, f"{TT}@all_turns", f"{TT}@all_positions"]
    assert drv.parse_cells("", constitution=True) == list(drv.DEFAULT_CELLS) + [
        "si_constitution__psr_kl_all", "si_constitution__psr_kl_all@all_turns",
        "si_constitution__psr_kl_all@all_positions"]
    assert drv.parse_cells(f"none,{TT}@all_turns") == ["none", f"{TT}@all_turns"]
    with pytest.raises(SystemExit):
        drv.parse_cells(f"none,{TT}@bogus")
    with pytest.raises(SystemExit):
        drv.parse_cells(f"none,none")
    assert drv.fitted_bases(drv.parse_cells("", constitution=True)) == \
        [TT, "si_constitution__psr_kl_all"]
    assert drv.ATTACKS == ("crescendo",) and drv.ARMS == ("qwen",)
    assert drv.TARGET_EVAL == "multiturn_attacks:crescendo" and drv.TARGET_KEY == "gemma4e4b"
    assert drv.LENS_SALT_POLICY == "request_unique_all" and drv.REDUCTION_MODE == "stacked"
    assert drv.DEFAULT_SUBSHARDS == 5 and drv.CONTAINER_TIMEOUT_S == 6 * 3600


def test_driver_argv_pins(drv):
    common = dict(sub=1, subshards=5, ids_path="/r/ids.txt", private_root="/r/qwen",
                  proxy_url="http://127.0.0.1:9/v1", rt="/r/rt.json", attacker_port=1234,
                  target_window=9984, attacker_serving="/r/att.serving.json")
    tree = "/vol/fst/repos/sisteer-" + SHA
    base = drv._driver_cmd(tree, cell=TT, **common)
    at = drv._driver_cmd(tree, cell=f"{TT}@all_turns", **common)
    ap = drv._driver_cmd(tree, cell=f"{TT}@all_positions", **common)
    none = drv._driver_cmd(tree, cell="none", **common)
    for cmd in (base, at, ap, none):
        assert cmd[1:3] == [f"{tree}/scripts/jailbreak_run.py", "generate"]
        assert cmd[cmd.index("--attack") + 1] == "crescendo"
        assert "--confirmatory" in cmd and cmd[cmd.index("--goal-ids-file") + 1] == "/r/ids.txt"
        assert cmd[cmd.index("--shard") + 1] == "1" and cmd[cmd.index("--nshards") + 1] == "5"
        assert cmd[cmd.index("--target-engine") + 1] == "served"
        assert cmd[cmd.index("--target-max-model-len") + 1] == "9984"
        assert cmd[cmd.index("--attacker-engine") + 1] == "vllm"
        assert cmd[cmd.index("--attacker-endpoint") + 1] == "http://127.0.0.1:1234"
        assert cmd[cmd.index("--attacker-serving") + 1] == "/r/att.serving.json"
        assert "--allow-errored-goals" not in cmd
        # the target decode is PINNED greedy: the identity record binds target_temperature 0.0, and
        # jailbreak_run's default has SAMPLED since 2026-09-11 (the model's generation config, then
        # the vendor preset since 2026-09-16)
        assert "--target-greedy" in cmd
        assert cmd[cmd.index("--private-root") + 1] == "/r/qwen"
    # the BASE fitted cell keeps --positions response (the §15 wave's cells) ...
    assert base[base.index("--positions") + 1] == "response"
    assert base[base.index("--defense") + 1] == TT
    # ... the variants and none carry NONE (the variant IS the positions)
    for cmd, cell in ((at, f"{TT}@all_turns"), (ap, f"{TT}@all_positions"), (none, "none")):
        assert "--positions" not in cmd and cmd[cmd.index("--defense") + 1] == cell
    # without a serving manifest the flag is absent
    assert "--attacker-serving" not in drv._driver_cmd(
        tree, cell="none", **{**common, "attacker_serving": None})


def test_standup_argv_and_env_pins(drv):
    tree = "/vol/fst/repos/sisteer-" + SHA
    cells = list(drv.DEFAULT_CELLS)
    digests = {TT: "ab" * 32, f"{TT}@all_turns": "ab" * 32, f"{TT}@all_positions": "ab" * 32}
    cmd = drv._target_standup_cmd(tree, Path("/r/srv"), Path("/r/srv/STOP"), _geom(),
                                  cells=cells, digests_by_tt=digests)
    assert cmd[1] == f"{tree}/scripts/capctrl_serving_up.py"
    got_cells = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--cell"]
    assert got_cells == ["none", TT, f"{TT}@all_turns", f"{TT}@all_positions"]
    got_dig = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--expect-object-sha256"]
    assert got_dig == [f"{TT}={'ab' * 32}", f"{TT}@all_turns={'ab' * 32}",
                       f"{TT}@all_positions={'ab' * 32}"]
    assert cmd[cmd.index("--eval") + 1] == "multiturn_attacks:crescendo"
    assert cmd[cmd.index("--model-key") + 1] == "gemma4e4b"
    assert cmd[cmd.index("--lens-salt-policy") + 1] == "request_unique_all"
    assert cmd[cmd.index("--defenses-dir") + 1] == "/vol/objects/scale-12/gemma4e4b"
    # dimensions travel BY SOURCE: a table-derived window is never a literal here
    assert "--max-model-len" not in cmd and "9984" not in cmd and "--allow-over-budget" not in cmd
    # an OVERRIDE window does travel explicitly
    cmd2 = drv._target_standup_cmd(tree, Path("/r/srv"), Path("/r/STOP"),
                                   _geom(window_source="override", max_model_len=8704,
                                         overridden=True, is_identity=False),
                                   cells=cells, digests_by_tt=digests)
    assert cmd2[cmd2.index("--max-model-len") + 1] == "8704" and "--allow-over-budget" in cmd2
    env = drv._standup_env(tree, gpu="1")
    assert env["SISTEER_LENS_STACKED_REDUCTIONS"] == "1"        # SET, not read (S2)
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert env["PYTHONPATH"] == f"{tree}/src:{tree}/scripts"
    assert env["TT_TWO_SI_DEFENSES_DIR"] == "/vol/objects/scale-12/gemma4e4b"


def test_app_name_rule(drv):
    s8 = SHA[:8]
    drv.check_app_name(f"fst-multiturn-{s8}-r1", sha=SHA, run="r1")
    drv.check_app_name(f"fst-multiturn-{s8}-r1-a2", sha=SHA, run="r1")   # a resumed attempt
    drv.check_app_name(f"fst-multiturn-{s8}-stage", sha=SHA)
    drv.check_app_name(f"fst-multiturn-{s8}-pf", sha=SHA)
    for bad in ("fst-multiturn", f"fst-multiturn-{s8}", f"fst-multiturn-{s8}-r2",
                f"fst-multiturn-{'b' * 8}-r1", "grokmt-served", f"fst-multiturn-{s8}-"):
        with pytest.raises(SystemExit, match="app name"):
            drv.check_app_name(bad, sha=SHA, run="r1")
    with pytest.raises(SystemExit, match="app name"):
        drv.check_app_name("fst-multiturn", sha=SHA)
    assert drv.expected_app_name(SHA, "r1") == f"fst-multiturn-{s8}-r1"
    # main() refuses the default app name before touching anything
    drv.app.description = None
    with pytest.raises(SystemExit, match="app name"):
        drv.main(sha=SHA, run="r1", go=True, ids="/nonexistent")
    with pytest.raises(SystemExit, match="40-hex"):
        drv.main(sha="abc", run="r1", go=True)


# --------------------------------------------------------------------------- #
# the run identity                                                            #
# --------------------------------------------------------------------------- #

def test_canonical_json_and_identity_diff(drv):
    a = {"b": 1, "a": {"y": [1, 2], "x": "s"}}
    b = {"a": {"x": "s", "y": [1, 2]}, "b": 1}
    assert drv.canonical_json_bytes(a) == drv.canonical_json_bytes(b) == b'{"a":{"x":"s","y":[1,2]},"b":1}'
    old = _record(drv)
    new = json.loads(json.dumps(old))
    new["per_cell"][f"{H}@all_turns"]["intervention_hash"] = "9" * 16
    new["lens_salt_policy"] = "attested"
    new["goal_ids"] = ["g1", "g2"]
    assert drv.identity_diff(old, new) == ["goal_ids", "lens_salt_policy",
                                           f"per_cell.{H}@all_turns.intervention_hash"]
    assert drv.identity_diff(old, json.loads(json.dumps(old))) == []


def test_build_run_identity_schema_and_bindings(drv):
    rec = _record(drv)
    assert set(rec) == {"schema", "gated_sha", "goal_ids_sha256", "goal_ids_file_sha256",
                        "goal_ids", "subshards", "goal_shards", "cells", "per_cell",
                        "control_legs", "object_artifact_sha256_by_cell", "reduction_mode",
                        "lens_salt_policy", "geometry", "attack", "target"}
    assert rec["schema"] == "fst_run_identity/1" and rec["gated_sha"] == SHA
    # THREE goal bindings by name (E1/B2): the ordered list, the digest of the
    # ORIGINAL file bytes (comments included), the canonical ordered-ids digest
    assert rec["goal_ids"] == ["g1", "g2", "g3"]
    assert rec["goal_ids_sha256"] == hashlib.sha256(b"g1\ng2\ng3\n").hexdigest()
    assert rec["goal_ids_file_sha256"] == hashlib.sha256(
        b"# synthetic ids file\ng1\ng2\ng3\n").hexdigest()
    assert rec["goal_ids_file_sha256"] != rec["goal_ids_sha256"]   # a commented file differs
    # bytes that do not parse to the given list are refused, never re-hashed around
    with pytest.raises(ValueError, match="goal_ids"):
        _record(drv, goal_ids_file_bytes=b"g1\ng3\ng2\n")
    with pytest.raises(ValueError, match="goal_ids"):
        _record(drv, goal_ids_file_bytes=b"g1\ng2\n")
    # the parse rule is the drivers' (jailbreak_run._select_goal_ids): comments, blanks, strip
    assert drv.parse_goal_ids("# c\n\n g1 \ng2\n  # c2\ng3") == ["g1", "g2", "g3"]
    assert rec["reduction_mode"] == "stacked" and rec["lens_salt_policy"] == "request_unique_all"
    assert rec["geometry"] == {"eval": "multiturn_attacks:crescendo", "model_key": "gemma4e4b",
                               "max_model_len": 9984, "max_num_seqs": 39,
                               "window_source": "table", "seqs_source": "model_budget"}
    assert set(rec["object_artifact_sha256_by_cell"]) == {H, f"{H}@all_turns", f"{H}@all_positions"}
    assert rec["per_cell"][f"{H}@all_positions"]["serve_declaration"] == \
        {"serve_experiment": "fit_response_serve_all_positions"}
    # canonical: a reordered goal list is a DIFFERENT identity
    rec2 = _record(drv, goal_ids=["g2", "g1", "g3"])
    assert rec2["goal_ids_sha256"] != rec["goal_ids_sha256"]


def _tree_state(root: Path) -> dict:
    out = {}
    for p in sorted(root.rglob("*")):
        st_ = p.stat()
        out[str(p.relative_to(root))] = (st_.st_mtime_ns,
                                         hashlib.sha256(p.read_bytes()).hexdigest()
                                         if p.is_file() else "dir")
    return out


def test_s2_i_same_identity_retry_after_identity_only_failure_proceeds(drv, tmp_path):
    """(i) a standup failure between the identity write and the first driver
    tree leaves an identity-only run dir; a plain retry re-enters (equal ->
    proceed) and does NOT rewrite the file."""
    dest, local = tmp_path / "dest", tmp_path / "local"
    rec = _record(drv)
    calls = []
    assert drv.ensure_run_identity(dest, rec, commit=lambda: calls.append("c")) == "written"
    p = dest / drv.IDENTITY_FILE
    assert p.read_bytes() == drv.canonical_json_bytes(rec) and calls == ["c"]
    assert (p.stat().st_mode & 0o777) == 0o440
    before = _tree_state(dest)
    spy = []
    out = drv.run_prelude(dest, local, rec, resume=False,
                          seed_fn=lambda *a: spy.append(a), commit=lambda: calls.append("c"))
    assert out == {"identity": "equal", "seed": None, "note": None, "lease": None}
    assert _tree_state(dest) == before and calls == ["c"] and spy == []


def test_s2_ii_resume_with_identity_but_no_arm_dirs_restarts_unseeded(drv, tmp_path):
    dest, local = tmp_path / "dest", tmp_path / "local"
    rec = _record(drv)
    drv.ensure_run_identity(dest, rec)
    (dest / "srv").mkdir()                       # a standup log dir, but no driver tree
    (dest / "srv" / "standup.log").write_text("died\n")
    assert not drv.has_driver_tree(dest)
    spy = []
    out = drv.run_prelude(dest, local, rec, resume=True, seed_fn=lambda *a: spy.append(a))
    assert out["identity"] == "equal" and out["seed"] is None
    assert "identity-only" in out["note"] and spy == []
    assert not (local / "attempts").exists()     # no attempts/NN entry filed


def _prior_driver_tree(dest: Path, drv) -> None:
    d = dest / "qwen" / drv.PRIVATE_RUN / "crescendo" / TT / "shard00"
    d.mkdir(parents=True)
    rows = [{"goal_id": "g1", "outcome": "failure", "n_target_forwards": 5},
            {"goal_id": "g2", "outcome": "refused:all_turns_expected_unavailable",
             "n_target_forwards": 2},
            {"goal_id": "g3", "outcome": "error:PromptOverWindow", "n_target_forwards": 1}]
    (d / "progress.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (d / "responses.jsonl").write_text(json.dumps({"goal_id": "g1", "order": 0, "response": "r"}) + "\n")
    (dest / "run_provenance.json").write_text(json.dumps({"started_at": "2026-01-01T00:00:00Z"}))
    (dest / "srv").mkdir()
    (dest / "srv" / "standup.log").write_text("old\n")


def test_s2_iii_partial_progress_resume_seeds(drv, tmp_path):
    dest, local = tmp_path / "dest", tmp_path / "local"
    rec = _record(drv)
    drv.ensure_run_identity(dest, rec)
    _prior_driver_tree(dest, drv)
    assert drv.has_driver_tree(dest)
    spy = []

    def seed(d, l, arms):
        spy.append((d, l, arms))
        return drv.seed_from_previous(d, l, arms)
    out = drv.run_prelude(dest, local, rec, resume=True, seed_fn=seed)
    assert out["identity"] == "equal" and spy == [(dest, local, ["qwen"])]
    s = out["seed"]
    assert s["attempt"] == 2 and s["covered_total"] == 2       # g1 done + g2 refused (terminal)
    assert s["refused_total"] == 1 and s["errored_to_retry_total"] == 1
    assert s["seeded"][f"qwen/crescendo/{TT}/shard00"] == \
        {"done": 3, "errored": 1, "refused": 1, "rows": 3}
    assert (local / "qwen" / drv.PRIVATE_RUN / "crescendo" / TT / "shard00" / "progress.jsonl").exists()
    assert (local / "attempts" / "01" / "srv" / "standup.log").read_text() == "old\n"


def test_s2_iv_changed_identity_refuses_before_seeding_and_touches_nothing(drv, tmp_path):
    dest, local = tmp_path / "dest", tmp_path / "local"
    rec = _record(drv)
    drv.ensure_run_identity(dest, rec)
    _prior_driver_tree(dest, drv)
    before = _tree_state(dest)
    changed = _record(drv)
    changed["per_cell"][f"{H}@all_turns"]["intervention_hash"] = "f" * 16
    spy = []
    with pytest.raises(drv.RunIdentityMismatch) as ei:
        drv.run_prelude(dest, local, changed, resume=True, seed_fn=lambda *a: spy.append(a))
    msg = str(ei.value)
    assert f"per_cell.{H}@all_turns.intervention_hash" in msg and "NEW --run" in msg
    assert spy == []                                   # seed_from_previous never called
    assert _tree_state(dest) == before                 # nothing under the run modified
    assert not local.exists()
    # the other identity axes refuse the same way
    for over in (dict(lens_salt_policy="attested"), dict(reduction_mode="per_row"),
                 dict(goal_ids=["g1", "g2"]), dict(cells=["none", TT]),
                 dict(digests_by_cell={H: "cd" * 32}),
                 dict(geom_dict=drv.geometry_record(_geom(max_model_len=8704))),
                 dict(attack={"name": "crescendo", "max_turns": 5}), dict(sha="b" * 40),
                 # the same ids in a file with a different header: the FILE is bound (E1/B2)
                 dict(goal_ids_file_bytes=b"# another header\ng1\ng2\ng3\n"),
                 # a different partition of the same goals (B1)
                 dict(subshards=2)):
        with pytest.raises(drv.RunIdentityMismatch):
            drv.ensure_run_identity(dest, _record(drv, **over))
    assert _tree_state(dest) == before
    # an unparseable existing record is named, not guessed around
    (dest / drv.IDENTITY_FILE).chmod(0o640)
    (dest / drv.IDENTITY_FILE).write_bytes(b"not json")
    with pytest.raises(drv.RunIdentityMismatch, match="unparseable"):
        drv.ensure_run_identity(dest, rec)


# --------------------------------------------------------------------------- #
# B1: subshards and the goal -> shard assignment are part of the identity       #
# --------------------------------------------------------------------------- #

def test_shard_assignment_is_the_drivers_slice(drv, tmp_path):
    """``shard_assignment`` is exactly ``jailbreak_run``'s ``_select_goal_ids``
    (file order) followed by ``goals[shard::nshards]`` — pinned against the
    real functions on stand-in goal objects, so the identity binds what the
    drivers will own."""
    import jailbreak_run as jr
    ids = [f"g{i:02d}" for i in range(10)]
    goals = [SimpleNamespace(goal_id=g) for g in reversed(ids)]     # eval order != file order
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("# header\n" + "\n".join(ids) + "\n")
    sel = jr._select_goal_ids(goals, str(ids_file))
    for n in (1, 2, 5, 7, 10, 12):
        want = [[g.goal_id for g in sel[k::n]] for k in range(n)]
        assert drv.shard_assignment(ids, n) == want, n
    assert drv.shard_assignment(ids, 5) == [ids[0::5], ids[1::5], ids[2::5], ids[3::5], ids[4::5]]
    # the reviewer's concrete case: goal index 5 sits on shard00 under 5 subshards
    # and on shard05 under 10
    assert ids[5] in drv.shard_assignment(ids, 5)[0] and ids[5] in drv.shard_assignment(ids, 10)[5]
    with pytest.raises(ValueError):
        drv.shard_assignment(ids, 0)
    rec = _record(drv, goal_ids=ids, subshards=5)
    assert rec["subshards"] == 5 and rec["goal_shards"] == drv.shard_assignment(ids, 5)
    assert drv.identity_diff(rec, _record(drv, goal_ids=ids, subshards=10)) == \
        ["goal_shards", "subshards"]


def test_b1_resume_with_a_changed_subshard_count_refuses_before_seeding(drv, tmp_path):
    """REGRESSION (B1): a run of 10 goals at 5 subshards has goal index 5
    terminally refused on shard00; resuming at 10 subshards would move it to
    shard05, which cannot see that record, re-attempt it, and let a success
    overwrite the refusal on merge. The identity now binds ``subshards`` and
    the full assignment, so the resume refuses before ``seed_from_previous``
    runs and touches nothing under the run."""
    ids = [f"g{i:02d}" for i in range(10)]
    dest, local = tmp_path / "dest", tmp_path / "local"
    rec5 = _record(drv, goal_ids=ids, subshards=5)
    drv.ensure_run_identity(dest, rec5)
    moved = ids[5]
    assert moved in rec5["goal_shards"][0]
    d = dest / "qwen" / drv.PRIVATE_RUN / "crescendo" / TT / "shard00"
    d.mkdir(parents=True)
    rows = [{"goal_id": g, "outcome": "failure", "n_target_forwards": 4} for g in ids[0::5]]
    rows[1] = {"goal_id": moved, "outcome": "refused:all_turns_expected_unavailable",
               "n_target_forwards": 2}
    (d / "progress.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (dest / "run_provenance.json").write_text(json.dumps({"started_at": "2026-01-01T00:00:00Z"}))
    before = _tree_state(dest)
    rec10 = _record(drv, goal_ids=ids, subshards=10)
    assert moved in rec10["goal_shards"][5]          # the goal would change shard
    spy = []
    with pytest.raises(drv.RunIdentityMismatch) as ei:
        drv.run_prelude(dest, local, rec10, resume=True, seed_fn=lambda *a: spy.append(a))
    msg = str(ei.value)
    assert "subshards" in msg and "goal_shards" in msg and "NEW --run" in msg
    assert spy == [] and _tree_state(dest) == before and not local.exists()
    # the SAME count resumes as before: the refused goal is seeded as covered
    out = drv.run_prelude(dest, local, rec5, resume=True)
    assert out["identity"] == "equal" and out["seed"]["refused_total"] == 1
    assert out["seed"]["covered_total"] == 2


# --------------------------------------------------------------------------- #
# progress counts, the attestation check                                       #
# --------------------------------------------------------------------------- #

def test_progress_counts_reports_refused_separately(drv, tmp_path):
    pj = tmp_path / "progress.jsonl"
    rows = [{"goal_id": "g1", "outcome": "AttackOutcome.SUCCESS"},
            {"goal_id": "g2", "outcome": "error:PromptOverWindow"},
            {"goal_id": "g2", "outcome": "failure"},                     # retried, then ran
            {"goal_id": "g3", "outcome": "refused:all_turns_render_refused"},
            {"goal_id": "g4", "outcome": "error:ActorDiscoveryError"}]
    pj.write_text("".join(json.dumps(r) + "\n" for r in rows) + '{"goal_id": "g5", "outc')
    assert drv.progress_counts(pj) == {"done": 4, "errored": 1, "refused": 1, "rows": 5}
    assert drv.progress_counts(tmp_path / "nope") == {"done": 0, "errored": 0, "refused": 0, "rows": 0}


def _endpoint(rec, **over):
    ep = {"proxy_url": "http://127.0.0.1:9/v1", "routing_table": "/r/rt.json",
          "lens_serve_declarations": {hid: pc["serve_declaration"] for hid, pc in rec["per_cell"].items()
                                      if pc["serve_declaration"]},
          "lens_serve_declarations_error": None,
          "lens_reduction_mode": "stacked", "lens_reduction_mode_error": None,
          "lens_salt_policy": "request_unique_all",
          "object_artifact_sha256_by_cell": dict(rec["object_artifact_sha256_by_cell"])}
    ep.update(over)
    return ep


def test_check_attestation_against_the_record(drv):
    from sisteer.eval.attacks import served_target as st
    rec = _record(drv)
    routes = {hid: pc["intervention_hash"] for hid, pc in rec["per_cell"].items()}
    routes["none__lens0"] = "t" * 16
    assert drv.check_attestation(rec, _endpoint(rec), routes, st=st) == []
    bad = drv.check_attestation(rec, _endpoint(rec, lens_salt_policy="attested"), routes, st=st)
    assert len(bad) == 1 and "lens_salt_policy" in bad[0]
    bad = drv.check_attestation(rec, _endpoint(rec, object_artifact_sha256_by_cell={H: "cd" * 32}),
                                routes, st=st)
    assert len(bad) == 1 and "object_artifact_sha256_by_cell" in bad[0]
    bad = drv.check_attestation(rec, _endpoint(rec, lens_reduction_mode="per_row"), routes, st=st)
    assert len(bad) == 1 and "lens_reduction_mode" in bad[0]
    # the declarations: missing, wrong, or naming the undeclared base
    bad = drv.check_attestation(rec, _endpoint(rec, lens_serve_declarations=None), routes, st=st)
    assert any("lens_serve_declarations" in b for b in bad)
    decl = dict(_endpoint(rec)["lens_serve_declarations"])
    decl[H] = {"serve_experiment": "fit_response_serve_all_turns"}
    bad = drv.check_attestation(rec, _endpoint(rec, lens_serve_declarations=decl), routes, st=st)
    assert any("undeclared cell" in b for b in bad)
    bad = drv.check_attestation(rec, _endpoint(rec, lens_serve_declarations_error="workers disagree"),
                                routes, st=st)
    assert any("lens_serve_declarations_error" in b for b in bad)
    # a routing-table hash that moved (the tree changed under the run), the twin included
    bad = drv.check_attestation(rec, _endpoint(rec), {**routes, f"{H}@all_turns": "9" * 16}, st=st)
    assert len(bad) == 1 and f"{H}@all_turns" in bad[0]
    bad = drv.check_attestation(rec, _endpoint(rec), {**routes, "none__lens0": "u" * 16}, st=st)
    assert len(bad) == 1 and "none__lens0" in bad[0]


def test_no_credentials_and_isolation(drv):
    src = DRIVER.read_text()
    assert "Secret" not in src and "api_key" not in src.lower() and "OPENROUTER" not in src
    assert "SISTEER_SECRET_CMD" not in src and "VLLM_API_KEY" not in src
    assert drv.MY == "/vol/fst" and drv.REPOS == "/vol/fst/repos" and drv.GEN == "/vol/fst/gen"
    assert drv.ATTACKER_HF == "/vol/fst/hf_attacker"
    # the only paths outside /vol/fst are read-only inputs
    assert drv.OBJECTS_X1.startswith("/vol/objects/") and drv.ATTACKER_HF_SHARED.startswith("/vol/grokmt/")
    assert drv.STATE_DIR == Path.home() / ".local" / "state" / "fst-multiturn"


def test_ids_default_to_the_tracked_file(drv):
    """``--ids`` defaults to configs/fst/fst_exp2_goal_ids.txt (kept as a flag)."""
    assert drv.TRACKED_IDS_REL == "configs/fst/fst_exp2_goal_ids.txt"
    p = drv.default_ids_path()
    assert p == REPO / "configs" / "fst" / "fst_exp2_goal_ids.txt" and p.is_file()
    ids = drv._read_ids(str(p))
    assert len(ids) == 10 and len(set(ids)) == 10
    sel = json.loads((REPO / "configs" / "fst" / "fst_exp2_goal_selection.json").read_text())
    assert sel["goal_ids"] == ids
    assert hashlib.sha256(p.read_bytes()).hexdigest() == sel["ids_file_sha256"]
    # the identity binds the TRACKED FILE's bytes (goal_ids_file_sha256 — the
    # binding, the digest the selection record pins) AND the canonical
    # ordered-ids digest under its own name; the tracked file has a # header,
    # so the two differ — the E1/B2 mismatch the analyzer must not conflate
    rec = _record(drv, goal_ids=ids, goal_ids_file_bytes=p.read_bytes())
    assert rec["goal_ids"] == ids
    assert rec["goal_ids_file_sha256"] == sel["ids_file_sha256"]
    assert rec["goal_ids_sha256"] == hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()
    assert rec["goal_ids_sha256"] != rec["goal_ids_file_sha256"]
    # an explicit --ids that does not exist refuses before anything is spawned
    drv.app.description = f"fst-multiturn-{SHA[:8]}-pf"
    with pytest.raises(SystemExit, match="no such file"):
        drv.main(sha=SHA, pf=True, ids=str(REPO / "configs" / "fst" / "nope.txt"))


def test_run_ids_file_preserves_the_tracked_bytes(drv, tmp_path):
    """The run's own copy of the ids file (what every driver's --goal-ids-file
    reads) is the ORIGINAL bytes, not a reconstruction: its digest is the
    identity's goal_ids_file_sha256 and it parses to the identity's goal_ids."""
    raw = TRACKED_IDS.read_bytes()
    ids = drv.parse_goal_ids(raw.decode())
    p = drv.write_run_ids_file(tmp_path / "run", raw)
    assert p == tmp_path / "run" / "goal_ids.txt" and p.read_bytes() == raw
    assert drv._read_ids(str(p)) == ids
    rec = _record(drv, goal_ids=ids, goal_ids_file_bytes=raw)
    assert hashlib.sha256(p.read_bytes()).hexdigest() == rec["goal_ids_file_sha256"]


def test_main_passes_the_ids_file_bytes_and_subshards_to_the_remote_run(drv, tmp_path, monkeypatch):
    """--go hands the remote run the ids FILE's bytes (the identity builder hashes
    the original, never a reconstruction) beside the parsed list, and the
    subshard count it will bind (B1)."""
    calls = []

    def spawn(*a, **k):
        calls.append((a, k))
        return SimpleNamespace(object_id="fc-test", get=lambda: {"status": "DONE"})
    monkeypatch.setattr(drv, "run_exp2", SimpleNamespace(spawn=spawn))
    monkeypatch.setattr(drv, "STATE_DIR", tmp_path / "state")
    drv.app.description = f"fst-multiturn-{SHA[:8]}-r1"
    drv.main(sha=SHA, run="r1", go=True, subshards=7)
    assert len(calls) == 1
    a, k = calls[0]
    raw = TRACKED_IDS.read_bytes()
    assert a[0] == SHA and a[1] == drv._read_ids(str(TRACKED_IDS)) and a[2] == "r1"
    assert a[3] == list(drv.DEFAULT_CELLS) and a[4] == 7
    assert k["ids_file_bytes"] == raw
    assert k["app_name"] == f"fst-multiturn-{SHA[:8]}-r1"       # the lease records the launcher's app
    state = json.loads((tmp_path / "state" / "r1.json").read_text())
    assert state["ids_file_sha256"] == hashlib.sha256(raw).hexdigest()
    assert state["subshards"] == 7 and state["ids"] == str(TRACKED_IDS)


# --------------------------------------------------------------------------- #
# S-C1: a run holding a driver tree REQUIRES --resume                           #
# --------------------------------------------------------------------------- #

def _progress_path(dest: Path, drv) -> Path:
    return dest / "qwen" / drv.PRIVATE_RUN / "crescendo" / TT / "shard00" / "progress.jsonl"


def test_c1_relaunch_without_resume_over_a_driver_tree_refuses_before_seed_or_lease(drv, tmp_path):
    """REGRESSION (code review S-C1): after a driver-phase failure the operator
    repeats ``--go --run <same>`` WITHOUT ``--resume``. Before: the identity
    compared "equal" and the prelude proceeded with an EMPTY local tree, so
    every goal would regenerate and the sync overwrite the terminal
    ``refused:*`` record with a stochastic success. Now: an identity plus a
    driver tree REFUSES unless ``--resume`` was given, naming the run dir and
    the flag; the refused goal's progress row stays byte-identical; nothing is
    seeded and no lease is taken. An identity-only dir (no driver tree) still
    admits the plain retry (the S2 case), and ``--resume`` over the tree seeds
    as before."""
    dest, local = tmp_path / "dest", tmp_path / "local"
    rec = _record(drv)
    drv.ensure_run_identity(dest, rec)
    _prior_driver_tree(dest, drv)
    pj = _progress_path(dest, drv)
    rows = [json.loads(ln) for ln in pj.read_text().splitlines()]
    assert rows[1]["outcome"].startswith("refused:")
    before, pj_bytes = _tree_state(dest), pj.read_bytes()
    seeds, leases = [], []
    with pytest.raises(drv.RunNeedsResume) as ei:
        drv.run_prelude(dest, local, rec, resume=False, seed_fn=lambda *a: seeds.append(a),
                        lease_fn=lambda d: leases.append(d))
    msg = str(ei.value)
    assert str(dest) in msg and "--resume" in msg and "refused:" in msg
    assert seeds == [] and leases == []
    assert _tree_state(dest) == before and pj.read_bytes() == pj_bytes
    assert not local.exists()
    # the S2 case is untouched: an identity-only dir admits the plain retry
    dest2 = tmp_path / "dest2"
    drv.ensure_run_identity(dest2, rec)
    (dest2 / "srv").mkdir()
    (dest2 / "srv" / "standup.log").write_text("died\n")
    out = drv.run_prelude(dest2, local, rec, resume=False, seed_fn=lambda *a: seeds.append(a))
    assert out["identity"] == "equal" and out["seed"] is None and seeds == []
    # and --resume over the same tree seeds as before, the refused goal covered
    out = drv.run_prelude(dest, local, rec, resume=True)
    assert out["seed"]["refused_total"] == 1 and out["seed"]["covered_total"] == 2


# --------------------------------------------------------------------------- #
# the real _run_exp2_impl on CPU: stand-ins for the tree / cells / geometry,    #
# every GPU, network, seed and sync surface fails the test if reached           #
# --------------------------------------------------------------------------- #

class _Proc:
    def __init__(self, rc=None):
        self.returncode = rc
        self.killed = False

    def poll(self):
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


def _impl_harness(drv, tmp_path, monkeypatch):
    """Drive the REAL ``_run_exp2_impl`` under tmp_path: the staged tree, the
    cell resolution, the geometry and the attack settings are the stand-ins
    ``_record`` uses (so the identity the impl builds IS ``_record(drv)``);
    ``seed_from_previous``, ``RunSyncer``, ``_start_attacker`` and
    ``_start_target`` fail the test if reached. Returns ``(call, dest, rec,
    orig)`` — ``orig`` holds the real surfaces for a test that lets the run
    proceed further."""
    base = _record(drv)
    tree = tmp_path / "tree"
    (tree / "src" / "sisteer").mkdir(parents=True)
    (tree / "scripts").mkdir()
    cfg = tree / drv.CFG_REL
    cfg.parent.mkdir(parents=True)
    cfg.write_text("{}\n")
    geom = _geom()
    geom.describe = lambda: "geom"
    geom.as_dict = lambda: dict(vars(_geom()))
    resolved = {"per_cell": base["per_cell"], "control_legs": base["control_legs"],
                "digests_by_cell": base["object_artifact_sha256_by_cell"],
                "digests_by_tt": {TT: "ab" * 32}, "target": base["target"]}
    orig = {"seed_from_previous": drv.seed_from_previous, "RunSyncer": drv.RunSyncer,
            "_start_attacker": drv._start_attacker, "_start_target": drv._start_target,
            "_sync": drv._sync}
    monkeypatch.setattr(drv, "_tree", lambda sha: str(tree))
    monkeypatch.setattr(drv, "GEN", str(tmp_path / "gen"))
    monkeypatch.setattr(drv, "RUN_LOCAL_ROOT", str(tmp_path / "local"))
    monkeypatch.setattr(drv, "resolve_target_geometry", lambda *a, **k: geom)
    monkeypatch.setattr(drv, "_served_target_module", lambda tree=None: SimpleNamespace())
    monkeypatch.setattr(drv, "resolve_cells", lambda tree, cells: resolved)
    monkeypatch.setattr(drv, "attack_settings", lambda cfg, *, jr: base["attack"])
    fake = _FakeDict()
    monkeypatch.setattr(drv, "_lease_dict", lambda: fake)
    orig["dict"] = fake

    def fail(what):
        return lambda *a, **k: pytest.fail(f"{what} was reached")
    for name in ("seed_from_previous", "RunSyncer", "_start_attacker", "_start_target"):
        monkeypatch.setattr(drv, name, fail(name))          # _sync stays real unless a test says otherwise
    ids_bytes = b"# synthetic ids file\n" + _canonical_ids_bytes(base["goal_ids"])
    dest = tmp_path / "gen" / "r1"

    def call(**kw):
        return drv._run_exp2_impl(SHA, list(base["goal_ids"]), "r1", list(drv.DEFAULT_CELLS),
                                  subshards=5, ids_file_bytes=ids_bytes, **kw)
    return call, dest, base, orig


def test_c1_impl_relaunch_without_resume_returns_needs_resume_before_any_sync(drv, tmp_path, monkeypatch):
    """REGRESSION (S-C1) through the real run body: with the identity present
    and a driver tree holding a terminal ``refused:*`` goal, a non-resume
    relaunch returns ``NEEDS_RESUME`` naming the run dir and ``--resume`` —
    before ``RunSyncer`` exists, before any standup, with the refused goal's
    progress row byte-identical and no local tree created."""
    call, dest, rec, _ = _impl_harness(drv, tmp_path, monkeypatch)
    drv.ensure_run_identity(dest, rec)
    _prior_driver_tree(dest, drv)
    pj = _progress_path(dest, drv)
    before, pj_bytes = _tree_state(dest), pj.read_bytes()
    s = call(resume=False)
    assert s["status"] == "NEEDS_RESUME"
    assert str(dest) in s["needs_resume_err"] and "--resume" in s["needs_resume_err"]
    assert _tree_state(dest) == before and pj.read_bytes() == pj_bytes
    assert not (tmp_path / "local").exists()
    assert "identity" not in s and s["timeline"] == {}     # refused before the first mark


# --------------------------------------------------------------------------- #
# S-C2: the per-run attempt lease                                              #
# --------------------------------------------------------------------------- #

T0 = 1_000_000.0
_MISSING = object()


class _FakeDict:
    """The ``modal.Dict`` semantics the lease relies on, as ONE authority shared
    by every acquirer (no per-container snapshots): ``put(skip_if_exists=True)``
    is atomic and returns whether the key was CREATED; ``get`` with a default;
    ``pop`` with a default. ``fail`` makes every operation raise (an unreadable
    Dict); ``after_put`` / ``before_pop`` hooks model a competitor acting inside
    the window; ``ops`` records every call."""

    def __init__(self):
        self.d: dict = {}
        self.ops: list = []
        self.fail = False
        self.after_put = None
        self.before_pop = None

    def _op(self, *op):
        self.ops.append(op)
        if self.fail:
            raise ConnectionError(f"dict {op[0]} failed")

    def put(self, key, value, *, skip_if_exists=False):
        self._op("put", key, skip_if_exists)
        if skip_if_exists and key in self.d:
            return False
        self.d[key] = copy.deepcopy(value)
        if self.after_put is not None:
            self.after_put(key)
        return True

    def get(self, key, default=None):
        self._op("get", key)
        return copy.deepcopy(self.d.get(key, default))

    def pop(self, key, default=_MISSING):
        self._op("pop", key)
        if self.before_pop is not None:
            self.before_pop(key)
        if key in self.d:
            return self.d.pop(key)
        if default is _MISSING:
            raise KeyError(key)
        return default


def _lease(drv, dest: Path, store, *, owner="fc-me", app="app-me", now=T0,
           commit=None, owner_kind="modal_function_call"):
    """An ``AttemptLease`` over the fake Dict ``store`` with an injected clock;
    ``log`` records the best-effort file layer's commits."""
    clock, log = [now], []
    lease = drv.AttemptLease(dest, run_id="r1", app=app, owner=owner, owner_kind=owner_kind, store=store,
                             commit=commit or (lambda: log.append("commit")),
                             clock=lambda: clock[0])
    return lease, clock, log


def _foreign_record(drv, *, owner="fc-x", app="app-x", now=T0, owner_kind="modal_function_call") -> dict:
    return drv.lease_record(run_id="r1", owner=owner, owner_kind=owner_kind, app=app, now=now)


def test_lease_acquire_puts_the_dict_entry_writes_the_file_and_release_deletes_it(drv, tmp_path):
    """The Dict entry is the AUTHORITY (an atomic ``put(skip_if_exists=True)``);
    ``attempt.lease`` on the volume is the human-readable record, written after
    the successful put. ``release`` re-checks the owner, DELETES the entry and
    stamps the file with ``released_at`` + the outcome; a second release is a
    no-op; a released run is free at once (the next put simply succeeds)."""
    dest = tmp_path / "dest"
    store = _FakeDict()
    lease, clock, log = _lease(drv, dest, store)
    assert lease.acquire() is lease and lease.held and not lease.lost and not lease.indeterminate
    assert store.ops[0] == ("put", "r1", True) and store.d["r1"]["owner"] == "fc-me"
    rec = store.d["r1"]
    assert rec["schema"] == drv.LEASE_SCHEMA == "fst_attempt_lease/1"
    assert rec["run_id"] == "r1" and rec["owner_kind"] == "modal_function_call" and rec["app"] == "app-me"
    assert rec["started_epoch"] == rec["heartbeat_epoch"] == T0 and "takeover" not in rec
    assert rec["released_at"] is None and rec["outcome"] is None
    p = dest / drv.LEASE_FILE
    assert lease.path == p and p.read_bytes() == drv.canonical_json_bytes(rec)   # the file mirrors the entry
    assert "commit" in log and drv.LEASE_DICT_NAME == "fst-multiturn-leases"
    clock[0] += 90
    assert lease.release("DONE") is True and not lease.held
    assert "r1" not in store.d and store.ops[-1] == ("pop", "r1")
    f = json.loads(p.read_bytes())
    assert f["released_at"] == "1970-01-12T13:48:10Z" and f["outcome"] == "DONE" and f["owner"] == "fc-me"
    assert lease.release("AGAIN") is False and json.loads(p.read_bytes())["outcome"] == "DONE"
    nxt, _, _ = _lease(drv, dest, store, owner="fc-next", app="app-next", now=T0 + 100)
    nxt.acquire()
    assert store.d["r1"]["owner"] == "fc-next"


def test_lease_two_acquirers_only_one_wins_whatever_the_volume_commit_order(drv, tmp_path):
    """REGRESSION: A and B on SEPARATE mount snapshots (two run
    dirs) both try to acquire. The Dict is one authority: A's put creates the
    entry, B's ``put(skip_if_exists=True)`` returns False and B is refused
    naming A — and B's delayed volume commit landing after A's read-back (B's
    bytes appearing in A's snapshot) changes nothing: A's ownership is the Dict
    entry, which B never wrote. Also in the other order."""
    store = _FakeDict()
    snap_a, snap_b = tmp_path / "mount-a" / "r1", tmp_path / "mount-b" / "r1"
    a, clock_a, _ = _lease(drv, snap_a, store, owner="fc-a", app="app-a")
    b, clock_b, _ = _lease(drv, snap_b, store, owner="fc-b", app="app-b")
    a.acquire()
    with pytest.raises(drv.AttemptLeaseHeld, match="fc-a"):
        b.acquire()
    assert a.held and not b.held and store.d["r1"]["owner"] == "fc-a"
    assert ("put", "r1", True) in store.ops[-3:] and not (snap_b / drv.LEASE_FILE).exists()
    # B's delayed volume commit "lands" after A's read-back: A's snapshot now shows B's bytes
    (snap_a / drv.LEASE_FILE).write_bytes(drv.canonical_json_bytes(_foreign_record(drv, owner="fc-b", app="app-b")))
    assert a.verify_owner()["owner"] == "fc-a"          # the authority is the Dict, not the file
    clock_a[0] += drv.LEASE_HEARTBEAT_S
    a.beat()
    assert a.held and store.d["r1"]["owner"] == "fc-a"
    with pytest.raises(drv.AttemptLeaseLost):
        b.verify_owner()
    # the other order
    store2 = _FakeDict()
    a2, _, _ = _lease(drv, tmp_path / "m2a" / "r1", store2, owner="fc-a")
    b2, _, _ = _lease(drv, tmp_path / "m2b" / "r1", store2, owner="fc-b")
    b2.acquire()
    with pytest.raises(drv.AttemptLeaseHeld, match="fc-b"):
        a2.acquire()
    assert b2.held and not a2.held


def test_lease_live_owner_refuses_a_second_launch(drv, tmp_path):
    """REGRESSION (code review S-C2): a second ``--resume --run <same>`` under
    another permitted app-name suffix while the first attempt's heartbeat is
    fresh is REFUSED, naming the holder, its app, the heartbeat age and the
    stale threshold; the Modal call is not probed for a fresh heartbeat; the
    entry is untouched and no file is written."""
    dest = tmp_path / "dest"
    store = _FakeDict()
    a, _, _ = _lease(drv, dest, store, owner="fc-a", app=f"fst-multiturn-{SHA[:8]}-r1")
    a.acquire()
    before = copy.deepcopy(store.d)
    b, _, log_b = _lease(drv, tmp_path / "other", store, owner="fc-b", app=f"fst-multiturn-{SHA[:8]}-r1-a2",
                         now=T0 + drv.LEASE_STALE_S - 1)
    n0 = len(store.ops)
    with pytest.raises(drv.AttemptLeaseHeld) as ei:
        b.acquire()
    msg = str(ei.value)
    assert "fc-a" in msg and f"fst-multiturn-{SHA[:8]}-r1" in msg and "LIVE" in msg
    assert str(drv.LEASE_STALE_S) in msg and "899" in msg           # the age and the threshold, in seconds
    assert "relaunch once" not in msg and "not taken over automatically" in msg
    assert "overlapping resumptions are not supported" in msg and drv.LEASE_DICT_NAME in msg
    assert not b.held and [op[0] for op in store.ops[n0:]] == ["put", "get"]     # inspected, never touched
    assert store.d == before and "commit" not in log_b and not (tmp_path / "other" / drv.LEASE_FILE).exists()
    assert drv.LEASE_STALE_S == 15 * 60 and drv.LEASE_HEARTBEAT_S == 60


@pytest.mark.parametrize("owner, owner_kind", [("fc-a1b2c3", "modal_function_call"), ("ab" * 16, "uuid")])
def test_lease_stale_entry_refuses_like_a_live_one(drv, tmp_path, owner, owner_kind):
    """REGRESSION: automatic stale takeover is GONE — re-read + pop +
    put are three operations, and two contenders that both judged the same
    stale owner finished could both acquire (the loser's unconditional pop
    deleted the winner's entry). A stale entry now refuses exactly like a live
    one: the message says STALE, gives the heartbeat age, states that
    overlapping resumptions are not supported, and quotes the manual remedy
    (inspect, confirm the prior app is dead, pop the entry, remove the record).
    The entry is inspected and never touched; there is no call-state probe."""
    dest = tmp_path / "dest"
    store = _FakeDict()
    a, _, _ = _lease(drv, dest, store, owner=owner, owner_kind=owner_kind, app="app-a")
    a.acquire()
    before = copy.deepcopy(store.d)
    assert not hasattr(drv, "_modal_call_state") and not hasattr(a, "call_state") and not hasattr(a, "takeover")
    for age in (drv.LEASE_STALE_S, 10 * drv.LEASE_STALE_S):
        b, _, log_b = _lease(drv, tmp_path / "other", store, owner="fc-b", app="app-b", now=T0 + age)
        n0 = len(store.ops)
        with pytest.raises(drv.AttemptLeaseHeld) as ei:
            b.acquire()
        msg = str(ei.value)
        assert "STALE" in msg and f"{age:.0f}s" in msg and str(drv.LEASE_STALE_S) in msg and owner in msg
        assert "not taken over automatically" in msg and "overlapping resumptions are not supported" in msg
        assert "confirmed by hand that the prior attempt is dead" in msg
        assert f'modal.Dict.from_name("{drv.LEASE_DICT_NAME}").pop("r1")' in msg
        assert "RUNNING" not in msg and "finished" not in msg              # nothing is probed or judged
        assert [op[0] for op in store.ops[n0:]] == ["put", "get"] and store.d == before
        assert not b.held and "commit" not in log_b and not (tmp_path / "other" / drv.LEASE_FILE).exists()


def test_lease_heartbeat_reads_verifies_puts_and_reads_back(drv, tmp_path):
    """``beat`` refreshes every LEASE_HEARTBEAT_S: get, verify the owner, put
    (unconditional), get again and compare (the read-back, as at acquisition);
    the file mirrors the fresh heartbeat best-effort."""
    dest = tmp_path / "dest"
    store = _FakeDict()
    a, clock, log = _lease(drv, dest, store, owner="fc-a")
    a.acquire()
    n0 = len(store.ops)
    clock[0] += drv.LEASE_HEARTBEAT_S - 1
    a.beat()
    assert len(store.ops) == n0                                 # not due
    clock[0] += 1
    a.beat()
    assert [op[0] for op in store.ops[n0:]] == ["get", "put", "get"] and store.ops[n0 + 1] == ("put", "r1", False)
    rec = store.d["r1"]
    assert rec["heartbeat_epoch"] == T0 + drv.LEASE_HEARTBEAT_S and rec["started_epoch"] == T0
    assert json.loads((dest / drv.LEASE_FILE).read_bytes())["heartbeat_epoch"] == T0 + drv.LEASE_HEARTBEAT_S
    assert a.held and not a.lost and not a.indeterminate


def test_lease_heartbeat_under_a_foreign_owner_is_lost(drv, tmp_path):
    """Another attempt owns the entry: the heartbeat's read finds it -> LOST,
    no put, the file left alone, ``release`` deletes nothing. The same when the
    foreign owner shows only at the read-back after our put."""
    dest = tmp_path / "dest"
    store = _FakeDict()
    a, clock, _ = _lease(drv, dest, store, owner="fc-a")
    a.acquire()
    file_before = (dest / drv.LEASE_FILE).read_bytes()
    store.d["r1"] = _foreign_record(drv, owner="fc-b", app="app-b", now=clock[0])
    clock[0] += drv.LEASE_HEARTBEAT_S
    n0 = len(store.ops)
    with pytest.raises(drv.AttemptLeaseLost, match="fc-b"):
        a.beat()
    assert a.lost and not a.held and [op[0] for op in store.ops[n0:]] == ["get"]
    assert store.d["r1"]["owner"] == "fc-b" and (dest / drv.LEASE_FILE).read_bytes() == file_before
    assert a.release("LEASE_LOST") is False and store.d["r1"]["owner"] == "fc-b"
    # the read-back form
    store2 = _FakeDict()
    b, clock_b, _ = _lease(drv, tmp_path / "d2", store2, owner="fc-a")
    b.acquire()
    foreign = _foreign_record(drv, owner="fc-c", app="app-c")

    def after_put(key):
        store2.after_put = None
        store2.d[key] = foreign
    store2.after_put = after_put
    clock_b[0] += drv.LEASE_HEARTBEAT_S
    with pytest.raises(drv.AttemptLeaseLost, match="fc-c"):
        b.beat()
    assert b.lost and not b.held and store2.d["r1"] == foreign
    assert b.release("LEASE_LOST") is False and store2.d["r1"] == foreign
    # a release that finds a foreign owner deletes nothing
    store3 = _FakeDict()
    c, _, _ = _lease(drv, tmp_path / "d3", store3, owner="fc-c")
    c.acquire()
    store3.d["r1"] = _foreign_record(drv, owner="fc-d")
    assert c.release("DONE") is False and store3.d["r1"]["owner"] == "fc-d"


def test_lease_unreadable_dict_is_indeterminate_and_writes_nothing(drv, tmp_path):
    """REGRESSION (fail CLOSED): when the Dict cannot be
    read, ownership is INDETERMINATE — at the heartbeat (no put, the lease
    dropped), at the sync fence, at release (nothing deleted), and at
    acquisition (nothing put, the file untouched)."""
    dest = tmp_path / "dest"
    store = _FakeDict()
    a, clock, log = _lease(drv, dest, store, owner="fc-a")
    a.acquire()
    entry, file_before, n_log = copy.deepcopy(store.d["r1"]), (dest / drv.LEASE_FILE).read_bytes(), len(log)
    store.fail = True
    clock[0] += drv.LEASE_HEARTBEAT_S
    n0 = len(store.ops)
    with pytest.raises(drv.AttemptLeaseIndeterminate, match="dict get failed"):
        a.beat()
    assert a.indeterminate and not a.held and not a.lost
    assert store.ops[n0:] == [("get", "r1")]                    # the failed read, nothing after it
    assert a.release("LEASE_INDETERMINATE") is False and store.ops[n0:] == [("get", "r1")]
    store.fail = False
    assert store.d["r1"] == entry and (dest / drv.LEASE_FILE).read_bytes() == file_before and len(log) == n_log
    # the fence
    b, _, _ = _lease(drv, tmp_path / "d2", store, owner="fc-a")
    b.held = True
    store.fail = True
    with pytest.raises(drv.AttemptLeaseIndeterminate):
        b.verify_owner()
    assert b.indeterminate and not b.held
    # release under an unreadable Dict deletes nothing
    store.fail = False
    c, _, _ = _lease(drv, tmp_path / "d3", store, owner="fc-a")
    store.d.clear()
    c.acquire()
    store.fail = True
    assert c.release("DONE") is False and not c.held
    store.fail = False
    assert store.d["r1"]["owner"] == "fc-a"
    # acquisition under an unreadable Dict: indeterminate, nothing put, no file
    store4 = _FakeDict()
    store4.fail = True
    d, _, log_d = _lease(drv, tmp_path / "d4", store4, owner="fc-d")
    with pytest.raises(drv.AttemptLeaseIndeterminate, match="dict put failed"):
        d.acquire()
    assert not d.held and d.indeterminate and store4.d == {} and not (tmp_path / "d4" / drv.LEASE_FILE).exists()
    assert "commit" not in log_d


def test_lease_verify_owner_fences_durable_writes(drv, tmp_path, monkeypatch):
    """``verify_owner`` is the fence before every sync: ours -> the record;
    another owner or no entry -> LOST; unreadable -> INDETERMINATE.
    ``_fenced_sync`` calls ``_sync`` only after it passes."""
    dest = tmp_path / "dest"
    store = _FakeDict()
    a, _, _ = _lease(drv, dest, store, owner="fc-a")
    a.acquire()
    synced = []
    monkeypatch.setattr(drv, "_sync", lambda syncer: synced.append(syncer))
    assert a.verify_owner()["owner"] == "fc-a"
    drv._fenced_sync(a, "SYNCER")
    assert synced == ["SYNCER"]
    store.d["r1"] = _foreign_record(drv, owner="fc-b")
    with pytest.raises(drv.AttemptLeaseLost, match="fc-b"):
        drv._fenced_sync(a, "SYNCER")
    assert synced == ["SYNCER"] and a.lost and not a.held
    b, _, _ = _lease(drv, dest, store, owner="fc-b")
    b.held = True
    del store.d["r1"]
    with pytest.raises(drv.AttemptLeaseLost, match="no entry"):
        drv._fenced_sync(b, "SYNCER")
    c, _, _ = _lease(drv, dest, store, owner="fc-c")
    c.held = True
    store.fail = True
    with pytest.raises(drv.AttemptLeaseIndeterminate):
        drv._fenced_sync(c, "SYNCER")
    assert synced == ["SYNCER"]


def test_lease_unparseable_or_foreign_entry_refuses(drv, tmp_path):
    store = _FakeDict()
    store.d["r1"] = "not a record"
    with pytest.raises(drv.AttemptLeaseHeld, match="unparseable"):
        _lease(drv, tmp_path / "d", store)[0].acquire()
    store.d["r1"] = {"schema": "something/9", "owner": "z"}
    with pytest.raises(drv.AttemptLeaseHeld, match="schema"):
        _lease(drv, tmp_path / "d", store)[0].acquire()
    assert store.d["r1"] == {"schema": "something/9", "owner": "z"}


def test_lease_remedy_names_the_dict_entry_and_the_file(drv):
    r = drv._lease_remedy("r1", Path(f"{drv.GEN}/r1"))
    assert f'modal.Dict.from_name("{drv.LEASE_DICT_NAME}").pop("r1")' in r
    assert f"modal volume rm {drv.VOL_NAME} /fst/gen/r1/attempt.lease" in r
    assert drv.VOL_NAME == "twosi-fits" and drv.LEASE_DICT_NAME == "fst-multiturn-leases"
    assert "attempt.lease" in drv._lease_remedy("r1", Path("/elsewhere/r1"))


def test_lease_file_is_best_effort(drv, tmp_path):
    """The volume file is the human-readable record only: a failing file
    commit neither fails the acquisition nor the release (the Dict decides)."""
    dest = tmp_path / "dest"
    store = _FakeDict()

    def bad_commit():
        raise RuntimeError("there are open files preventing the operation")
    a, clock, _ = _lease(drv, dest, store, owner="fc-a", commit=bad_commit)
    a.acquire()
    assert a.held and store.d["r1"]["owner"] == "fc-a" and a.file_errors == 1
    clock[0] += drv.LEASE_HEARTBEAT_S
    a.beat()
    assert a.held and a.file_errors == 2
    assert a.release("DONE") is True and "r1" not in store.d and a.file_errors == 3


def test_lease_owner_is_the_modal_call_id_else_a_uuid(drv, monkeypatch):
    assert drv.owner_of()[1] == "uuid" and len(drv.owner_of()[0]) == 32        # the stub has no call id
    monkeypatch.setattr(drv.modal, "current_function_call_id", lambda: "fc-live", raising=False)
    assert drv.owner_of() == ("fc-live", "modal_function_call")
    # no call-state probe exists any more — the API cannot make a takeover safe
    assert not hasattr(drv, "_modal_call_state")


def test_ensure_run_identity_compare_only_mode_never_writes(drv, tmp_path):
    """Before the lease is held the identity is only COMPARED — an
    absent file reports ``"absent"`` and nothing under the run is created."""
    dest = tmp_path / "dest"
    rec = _record(drv)
    assert drv.ensure_run_identity(dest, rec, write=False) == "absent"
    assert not dest.exists()
    assert drv.ensure_run_identity(dest, rec) == "written" and (dest / drv.IDENTITY_FILE).exists()
    assert drv.ensure_run_identity(dest, rec, write=False) == "equal"
    before = _tree_state(dest)
    with pytest.raises(drv.RunIdentityMismatch):
        drv.ensure_run_identity(dest, _record(drv, sha="b" * 40), write=False)
    assert _tree_state(dest) == before


class _SpyLease:
    def __init__(self, on_acquire=None):
        self.released = None
        self.on_acquire = on_acquire

    def acquire(self, dest):
        if self.on_acquire is not None:
            self.on_acquire(dest)
        return self

    def release(self, outcome):
        self.released = outcome


def test_prelude_reloads_after_acquiring_and_re_evaluates_before_seeding(drv, tmp_path):
    """REGRESSION: the prelude used to compare the identity, check the
    driver tree and SEED from the snapshot taken BEFORE the lease was acquired —
    so a previous owner's final commit (a terminal refusal) landing between that
    snapshot and the acquire was missing from the seed and the goal regenerated.
    Now: compare only (no write) -> S-C1 check -> LEASE -> RELOAD -> identity
    compare-or-write -> S-C1 re-check -> seed, all against the fresh snapshot.
    The identity is written only by the owner. A driver tree or a different
    identity surfacing at the reload refuses and releases the lease."""
    dest, local = tmp_path / "dest", tmp_path / "local"
    rec = _record(drv)
    _prior_driver_tree(dest, drv)
    pj = _progress_path(dest, drv)
    rows = [json.loads(ln) for ln in pj.read_text().splitlines()]
    rows[1]["outcome"] = "failure"                                       # the pre-acquire snapshot: g2 not refused
    pj.write_text("".join(json.dumps(r) + "\n" for r in rows))
    order = []

    def reload():                                                        # the previous owner's final commit lands
        order.append("reload")
        rows[1]["outcome"] = "refused:all_turns_expected_unavailable"
        pj.write_text("".join(json.dumps(r) + "\n" for r in rows))
    lease = _SpyLease(on_acquire=lambda d: (order.append(("lease", d)),
                                           order.append(("identity_exists", (d / drv.IDENTITY_FILE).exists()))))
    out = drv.run_prelude(dest, local, rec, resume=True, commit=lambda: order.append("identity_commit"),
                          lease_fn=lease.acquire, reload_fn=reload,
                          seed_fn=lambda d, l, arms: order.append("seed") or drv.seed_from_previous(d, l, arms))
    assert order == [("lease", dest), ("identity_exists", False), "reload", "identity_commit", "seed"]
    assert out["identity"] == "written" and out["lease"] is lease and lease.released is None
    assert out["seed"]["refused_total"] == 1                             # the late refusal IS in the seed
    seeded = _progress_path(local, drv)
    assert any(json.loads(ln)["outcome"].startswith("refused:") for ln in seeded.read_text().splitlines())
    # (i) a driver tree surfacing at the reload on a non-resume launch refuses and releases
    dest2, local2 = tmp_path / "dest2", tmp_path / "local2"
    drv.ensure_run_identity(dest2, rec)
    lease2 = _SpyLease()

    def reload2():
        _prior_driver_tree(dest2, drv)
    with pytest.raises(drv.RunNeedsResume):
        drv.run_prelude(dest2, local2, rec, resume=False, lease_fn=lease2.acquire, reload_fn=reload2,
                        seed_fn=lambda *a: pytest.fail("seeded"))
    assert lease2.released == "NEEDS_RESUME" and not local2.exists()
    # (ii) a different identity surfacing at the reload refuses and releases
    dest3, local3 = tmp_path / "dest3", tmp_path / "local3"
    lease3 = _SpyLease()

    def reload3():
        drv.ensure_run_identity(dest3, _record(drv, sha="b" * 40))
    with pytest.raises(drv.RunIdentityMismatch):
        drv.run_prelude(dest3, local3, rec, resume=True, lease_fn=lease3.acquire, reload_fn=reload3,
                        seed_fn=lambda *a: pytest.fail("seeded"))
    assert lease3.released == "IDENTITY_MISMATCH" and not local3.exists()
    # a held lease still stops the prelude before the reload and the seed
    def held(d):
        raise drv.AttemptLeaseHeld("live")
    with pytest.raises(drv.AttemptLeaseHeld):
        drv.run_prelude(dest, local, rec, resume=True, lease_fn=held, reload_fn=lambda: pytest.fail("reloaded"),
                        seed_fn=lambda *a: pytest.fail("seeded under a held lease"))
    # a seeding failure releases the lease the prelude took
    lease4 = _SpyLease()

    def bad_seed(*a):
        raise RuntimeError("torn")
    with pytest.raises(RuntimeError, match="torn"):
        drv.run_prelude(dest, local, rec, resume=True, lease_fn=lease4.acquire, seed_fn=bad_seed)
    assert lease4.released == "SEED_FAILED:RuntimeError"
    # without a lease (the S2 unit tests' form) the identity is written up front, as before
    dest5 = tmp_path / "dest5"
    calls = []
    out5 = drv.run_prelude(dest5, tmp_path / "local5", rec, resume=False, commit=lambda: calls.append("c"))
    assert out5["identity"] == "written" and calls == ["c"] and (dest5 / drv.IDENTITY_FILE).exists()


def test_wait_loops_beat_the_lease_on_every_tick(drv, tmp_path, monkeypatch):
    """The two blocking waits of the standup phase call ``on_tick`` once per
    poll, so a 15-minute stale threshold survives a ten-minute vLLM load."""
    monkeypatch.setattr(drv.time, "sleep", lambda s: None)
    ticks = []
    healthy = iter([False, False, True])
    monkeypatch.setattr(drv, "_http_ok", lambda url, timeout=5.0: next(healthy))
    proc = SimpleNamespace(poll=lambda: None, returncode=None)
    drv._wait_health("http://x/health", proc, 60, "x", tmp_path / "x.log", on_tick=lambda: ticks.append("h"))
    assert ticks == ["h", "h", "h"]
    log = tmp_path / "attacker_vllm.log"
    manifest = log.with_suffix(".serving.json")

    def tick():
        ticks.append("m")
        if ticks.count("m") == 2:
            manifest.write_text("{}")
    assert drv._wait_attacker_serving_manifest(log, proc, timeout_s=60, on_tick=tick) == manifest
    assert ticks == ["h", "h", "h", "m", "m"]


def _bounded_sleep(n: int):
    """A ``time.sleep`` stand-in for the run loops: no waiting, and the test
    FAILS after ``n`` ticks instead of spinning when a loop never exits."""
    ticks = [0]

    def sleep(_s):
        ticks[0] += 1
        if ticks[0] > n:
            pytest.fail(f"the run loop spun {n} ticks without exiting")
    return sleep


def test_c2_impl_live_lease_refuses_before_seed_or_standup(drv, tmp_path, monkeypatch):
    """REGRESSION (S-C2) through the real run body: a second ``--resume`` on a
    run whose Dict entry has a fresh heartbeat returns ``LEASE_HELD`` before
    seeding, standup or sync; the tree and the entry are untouched."""
    call, dest, rec, orig = _impl_harness(drv, tmp_path, monkeypatch)
    drv.ensure_run_identity(dest, rec)
    _prior_driver_tree(dest, drv)
    orig["dict"].d["r1"] = _foreign_record(drv, owner="fc-other", app=f"fst-multiturn-{SHA[:8]}-r1", now=time.time())
    entry, before = copy.deepcopy(orig["dict"].d), _tree_state(dest)
    s = call(resume=True, app_name=f"fst-multiturn-{SHA[:8]}-r1-a2")
    assert s["status"] == "LEASE_HELD" and "fc-other" in s["lease_err"]
    assert _tree_state(dest) == before and orig["dict"].d == entry and not (tmp_path / "local").exists()
    assert "lease" not in s and "resume" not in s


def test_c2_impl_an_unreadable_dict_at_acquisition_is_indeterminate_before_seeding(drv, tmp_path, monkeypatch):
    call, dest, rec, orig = _impl_harness(drv, tmp_path, monkeypatch)
    drv.ensure_run_identity(dest, rec)
    _prior_driver_tree(dest, drv)
    orig["dict"].fail = True
    before = _tree_state(dest)
    s = call(resume=True, app_name=f"fst-multiturn-{SHA[:8]}-r1-a2")
    assert s["status"] == "LEASE_INDETERMINATE" and "dict put failed" in s["lease_err"]
    assert _tree_state(dest) == before and not (tmp_path / "local").exists() and "resume" not in s


def test_c2_impl_acquires_before_seeding_and_releases_in_finally_recording_the_outcome(
        drv, tmp_path, monkeypatch):
    """Through the real run body: the Dict entry is taken (owner = this attempt,
    the app name recorded) and the run seeds; a target standup failure returns
    early and the ``finally`` releases — the entry deleted, the file stamped
    with that outcome; an exception on the way releases too, naming the
    exception; a released run admits the next attempt."""
    call, dest, rec, orig = _impl_harness(drv, tmp_path, monkeypatch)
    drv.ensure_run_identity(dest, rec)
    # RunSyncer re-copies run_local's identity over the 0440 volume copy; the container runs as
    # root, this harness does not — make it writable so the failure-path sync can be asserted
    (dest / drv.IDENTITY_FILE).chmod(0o640)
    _prior_driver_tree(dest, drv)
    monkeypatch.setattr(drv, "seed_from_previous", orig["seed_from_previous"])
    monkeypatch.setattr(drv, "RunSyncer", orig["RunSyncer"])
    attacker = _Proc(None)
    monkeypatch.setattr(drv, "_start_attacker", lambda port, log: attacker)

    def start_target_dies(tree, srv, geom, *, cells, digests_by_tt, gpu="1"):
        (srv / "standup.log").write_text("boom\n")
        return _Proc(1), srv / "STOP"
    monkeypatch.setattr(drv, "_start_target", start_target_dies)
    app = f"fst-multiturn-{SHA[:8]}-r1-a2"
    s = call(resume=True, app_name=app)
    assert s["status"] == "TARGET_STANDUP_FAILED" and s["resume"]["refused_total"] == 1
    assert attacker.killed
    assert "r1" not in orig["dict"].d                                   # released = deleted
    lease = json.loads((dest / drv.LEASE_FILE).read_bytes())
    assert lease["owner"] == s["lease"]["owner"] and lease["owner_kind"] == "uuid" == s["lease"]["owner_kind"]
    assert lease["app"] == app == s["lease"]["app"] and "takeover" not in lease and "takeover" not in s["lease"]
    assert lease["released_at"] and lease["outcome"] == "TARGET_STANDUP_FAILED"
    assert s["lease"]["released"] is True and s["lease"]["file"] == str(dest / drv.LEASE_FILE)
    assert s["lease"]["dict"] == drv.LEASE_DICT_NAME
    assert s["lease"]["release_basis"] == "final_sync_committed"
    assert s["timeline"]["identity_equal"] <= s["timeline"]["resumed_attempt_2"]
    # the failure path still synced this attempt's tree over the fixture's (through the fence)
    assert (dest / "srv" / "standup.log").read_text() == "boom\n"
    assert json.loads((dest / "run_provenance.json").read_text())["run_identity"] == rec
    assert (dest / "attempts" / "01" / "srv" / "standup.log").read_text() == "old\n"
    # an exception on the way out releases too, naming it

    def no_gpu(port, log):
        raise RuntimeError("no gpu")
    monkeypatch.setattr(drv, "_start_attacker", no_gpu)
    with pytest.raises(RuntimeError, match="no gpu"):
        call(resume=True, app_name=app + "-b")
    assert "r1" not in orig["dict"].d
    lease2 = json.loads((dest / drv.LEASE_FILE).read_bytes())
    assert lease2["app"] == app + "-b" and lease2["released_at"] and lease2["outcome"] == "EXCEPTION:RuntimeError"


def test_c2_impl_a_lost_lease_stops_the_run_without_syncing(drv, tmp_path, monkeypatch):
    """When a heartbeat's read finds another owner in the Dict (a takeover),
    the run stops with ``LEASE_LOST``, kills what it started, syncs NOTHING
    more, and leaves the new owner's entry alone."""
    call, dest, rec, orig = _impl_harness(drv, tmp_path, monkeypatch)
    drv.ensure_run_identity(dest, rec)
    _prior_driver_tree(dest, drv)
    monkeypatch.setattr(drv, "seed_from_previous", orig["seed_from_previous"])
    monkeypatch.setattr(drv, "RunSyncer", orig["RunSyncer"])
    monkeypatch.setattr(drv, "_sync", lambda syncer: pytest.fail("SYNC_AFTER_LEASE_FAILURE"))
    monkeypatch.setattr(drv, "LEASE_HEARTBEAT_S", 0)               # every tick refreshes
    monkeypatch.setattr(drv.time, "sleep", _bounded_sleep(50))
    attacker, target = _Proc(None), _Proc(None)
    foreign = _foreign_record(drv, owner="fc-taker", app="app-taker", now=time.time())
    monkeypatch.setattr(drv, "_start_attacker", lambda port, log: attacker)

    def start_target(tree, srv, geom, **k):
        orig["dict"].d["r1"] = copy.deepcopy(foreign)             # another attempt took the run over
        return target, srv / "STOP"
    monkeypatch.setattr(drv, "_start_target", start_target)
    before = _tree_state(dest)
    s = call(resume=True, app_name=f"fst-multiturn-{SHA[:8]}-r1-a2")
    assert s["status"] == "LEASE_LOST" and "fc-taker" in s["lease_err"]
    assert attacker.killed and target.killed and (tmp_path / "local" / "r1" / "srv" / "STOP").exists()
    assert orig["dict"].d["r1"] == foreign                           # the taker's entry stands
    after = _tree_state(dest)
    after.pop(drv.LEASE_FILE, None)
    assert after == before                                            # nothing synced after the loss
    assert (tmp_path / "local" / "r1" / "run_provenance.json").exists()   # ...though it was ready to
    assert s["lease"]["released"] is False


def test_c2_impl_an_indeterminate_lease_stops_the_run_writing_nothing(drv, tmp_path, monkeypatch):
    """REGRESSION through the real run body: a heartbeat whose Dict
    read fails stops the run with ``LEASE_INDETERMINATE`` — drivers/servers
    killed, nothing synced, the entry untouched since acquisition (not
    rewritten, not deleted)."""
    call, dest, rec, orig = _impl_harness(drv, tmp_path, monkeypatch)
    drv.ensure_run_identity(dest, rec)
    _prior_driver_tree(dest, drv)
    monkeypatch.setattr(drv, "seed_from_previous", orig["seed_from_previous"])
    monkeypatch.setattr(drv, "RunSyncer", orig["RunSyncer"])
    monkeypatch.setattr(drv, "_sync", lambda syncer: pytest.fail("SYNC_AFTER_LEASE_FAILURE"))
    monkeypatch.setattr(drv, "LEASE_HEARTBEAT_S", 0)
    monkeypatch.setattr(drv.time, "sleep", _bounded_sleep(50))
    attacker, target = _Proc(None), _Proc(None)
    monkeypatch.setattr(drv, "_start_attacker", lambda port, log: attacker)
    entry = {}

    def start_target(tree, srv, geom, **k):
        entry.update(copy.deepcopy(orig["dict"].d["r1"]))
        orig["dict"].fail = True                                   # the Dict becomes unreadable
        return target, srv / "STOP"
    monkeypatch.setattr(drv, "_start_target", start_target)
    before = _tree_state(dest)
    s = call(resume=True, app_name=f"fst-multiturn-{SHA[:8]}-r1-a2")
    assert s["status"] == "LEASE_INDETERMINATE" and "dict get failed" in s["lease_err"]
    assert attacker.killed and target.killed and (tmp_path / "local" / "r1" / "srv" / "STOP").exists()
    orig["dict"].fail = False
    assert orig["dict"].d["r1"] == entry and entry["owner"] == s["lease"]["owner"]
    assert entry["heartbeat_epoch"] == entry["started_epoch"]          # untouched since acquisition
    assert all(op[0] == "get" for op in orig["dict"].ops if orig["dict"].ops.index(op) > 3)
    after = _tree_state(dest)
    after.pop(drv.LEASE_FILE, None)
    assert after == before                                              # nothing synced
    assert s["lease"]["released"] is False and s["lease"]["indeterminate"] is True


def test_c2_impl_sync_fence_refuses_to_sync_under_a_foreign_owner(drv, tmp_path, monkeypatch):
    """REGRESSION (fencing): the owner is re-verified immediately
    before EVERY sync. Here the target dies at once (a TARGET_STANDUP_FAILED
    path, which syncs) while another attempt has taken the entry: the fence
    turns it into LEASE_LOST and ``_sync`` is never called."""
    call, dest, rec, orig = _impl_harness(drv, tmp_path, monkeypatch)
    drv.ensure_run_identity(dest, rec)
    _prior_driver_tree(dest, drv)
    monkeypatch.setattr(drv, "seed_from_previous", orig["seed_from_previous"])
    monkeypatch.setattr(drv, "RunSyncer", orig["RunSyncer"])
    monkeypatch.setattr(drv, "_sync", lambda syncer: pytest.fail("SYNC_UNDER_A_FOREIGN_OWNER"))
    attacker = _Proc(None)
    monkeypatch.setattr(drv, "_start_attacker", lambda port, log: attacker)
    foreign = _foreign_record(drv, owner="fc-taker", app="app-taker", now=time.time())

    def start_target_dies(tree, srv, geom, **k):
        (srv / "standup.log").write_text("boom\n")
        orig["dict"].d["r1"] = copy.deepcopy(foreign)
        return _Proc(1), srv / "STOP"
    monkeypatch.setattr(drv, "_start_target", start_target_dies)
    before = _tree_state(dest)
    s = call(resume=True, app_name=f"fst-multiturn-{SHA[:8]}-r1-a2")
    assert s["status"] == "LEASE_LOST" and "fc-taker" in s["lease_err"] and "standup_tail" in s
    assert attacker.killed and orig["dict"].d["r1"] == foreign
    after = _tree_state(dest)
    after.pop(drv.LEASE_FILE, None)
    assert after == before


def test_impl_seeds_from_the_post_acquire_snapshot(drv, tmp_path, monkeypatch):
    """REGRESSION through the real run body: the volume is reloaded
    AFTER the lease is acquired and the seed is taken from that snapshot, so a
    terminal refusal the previous owner committed between the first reload and
    the acquire is present in the seeded tree (and the driver, whose coverage
    reader treats refused as covered, will not regenerate it)."""
    call, dest, rec, orig = _impl_harness(drv, tmp_path, monkeypatch)
    drv.ensure_run_identity(dest, rec)
    (dest / drv.IDENTITY_FILE).chmod(0o640)          # the harness is not root; see the release test
    _prior_driver_tree(dest, drv)
    pj = _progress_path(dest, drv)
    rows = [json.loads(ln) for ln in pj.read_text().splitlines()]
    rows[1]["outcome"] = "failure"                    # the first snapshot: g2 not yet refused
    pj.write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(drv, "seed_from_previous", orig["seed_from_previous"])
    monkeypatch.setattr(drv, "RunSyncer", orig["RunSyncer"])
    attacker = _Proc(None)
    monkeypatch.setattr(drv, "_start_attacker", lambda port, log: attacker)

    def start_target_dies(tree, srv, geom, **k):
        (srv / "standup.log").write_text("boom\n")
        return _Proc(1), srv / "STOP"
    monkeypatch.setattr(drv, "_start_target", start_target_dies)
    events = []
    orig["dict"].after_put = lambda key: events.append("put")

    def reload():
        events.append("reload")
        if events.count("reload") == 2:               # the previous owner's final commit landed in between
            rows[1]["outcome"] = "refused:all_turns_expected_unavailable"
            pj.write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(drv, "VOL", SimpleNamespace(reload=reload, commit=lambda: None))
    s = call(resume=True, app_name=f"fst-multiturn-{SHA[:8]}-r1-a2")
    assert s["status"] == "TARGET_STANDUP_FAILED"
    assert events[:3] == ["reload", "put", "reload"]                    # acquire, THEN the fresh snapshot
    assert s["resume"]["refused_total"] == 1 and s["resume"]["covered_total"] == 2
    local_pj = _progress_path(tmp_path / "local" / "r1", drv)
    assert [json.loads(ln)["outcome"] for ln in local_pj.read_text().splitlines()][1].startswith("refused:")
    assert [json.loads(ln)["outcome"] for ln in pj.read_text().splitlines()][1].startswith("refused:")


def test_c2_every_sync_in_the_run_body_is_fenced_and_every_exit_sync_is_final(drv):
    """No bare ``_sync(syncer)`` and no bare ``_fenced_sync`` remains in
    ``_run_exp2_impl``: every copy goes through ``data_sync()`` — the ONE
    ``_fenced_sync`` call, which marks data copying as begun first — namely the
    two intermediate syncs (attacker up, the driver loop) and the one inside
    ``final_sync()``; the exit-path syncs (standup failed / timeout, attestation
    mismatch, attacker failed, serving unverified, the end, and the general
    handler's owned recovery sync) go through ``final_sync()``, whose SUCCESS
    alone authorizes the release."""
    import ast
    text = DRIVER.read_text()
    fn = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == "_run_exp2_impl")
    calls = [n.func.id for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id in ("_sync", "_fenced_sync", "data_sync", "final_sync")]
    assert "_sync" not in calls, calls
    assert calls.count("_fenced_sync") == 1 and calls.count("data_sync") == 3, calls   # 2 intermediate + 1 inside final_sync
    assert calls.count("final_sync") == 7, calls                                       # 6 exit paths + the recovery sync
    inner = next(n for n in ast.walk(fn) if isinstance(n, ast.FunctionDef) and n.name == "data_sync")
    assert [n.func.id for n in ast.walk(inner) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)] == ["_fenced_sync"]


def test_release_stamps_the_record_before_the_entry_goes(drv, tmp_path):
    """``release`` writes AND commits the human-readable stamp before the
    Dict entry is deleted, so no stamp commit can land after the run was
    advertised free; the stamp stays best-effort (a failing commit still lets
    the entry go)."""
    dest = tmp_path / "dest"
    store = _FakeDict()
    a, clock, log = _lease(drv, dest, store, owner="fc-a")
    a.acquire()
    seen = []

    def before_pop(key):
        rec = json.loads((dest / drv.LEASE_FILE).read_bytes())
        seen.append((rec["released_at"], rec["outcome"], log.count("commit")))
    store.before_pop = before_pop
    clock[0] += 5
    assert a.release("DONE") is True
    assert seen == [("1970-01-12T13:46:45Z", "DONE", 2)]          # stamped AND committed before the pop
    assert "r1" not in store.d
    store2 = _FakeDict()

    def bad_commit():
        raise RuntimeError("open files")
    b, _, _ = _lease(drv, tmp_path / "d2", store2, owner="fc-b", commit=bad_commit)
    b.acquire()
    assert b.release("DONE") is True and "r1" not in store2.d and b.file_errors == 2


def test_sync_reports_whether_the_copy_was_committed(drv, tmp_path, monkeypatch):
    """``_sync`` returns True only when the copy AND the volume commit
    succeeded (a failure still only warns), and ``_fenced_sync`` passes the
    result through — the run body's final sync decides on it."""
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    (src / "a.txt").write_text("x")
    syncer = drv.RunSyncer(src, str(dst))
    monkeypatch.setattr(drv, "VOL", SimpleNamespace(reload=lambda: None, commit=lambda: None))
    assert drv._sync(syncer) is True and (dst / "a.txt").read_text() == "x"

    def boom():
        raise RuntimeError("volume commit failed")
    monkeypatch.setattr(drv, "VOL", SimpleNamespace(reload=lambda: None, commit=boom))
    (src / "b.txt").write_text("y")
    assert drv._sync(syncer) is False
    store = _FakeDict()
    a, _, _ = _lease(drv, tmp_path / "dest", store, owner="fc-a")
    a.acquire()
    assert drv._fenced_sync(a, syncer) is False
    monkeypatch.setattr(drv, "VOL", SimpleNamespace(reload=lambda: None, commit=lambda: None))
    assert drv._fenced_sync(a, syncer) is True


def test_impl_a_failed_final_sync_holds_the_lease(drv, tmp_path, monkeypatch):
    """REGRESSION through the real run body: the attempt's final data
    sync fails to commit -> the run reports FINAL_SYNC_FAILED (the status it
    was about to report preserved beside it), says so loudly, and the lease is
    NOT released: the Dict entry stays this attempt's and the record carries no
    release stamp — a later launch refuses with the manual remedy instead of
    resuming from a tree that was never durably committed."""
    call, dest, rec, orig = _impl_harness(drv, tmp_path, monkeypatch)
    drv.ensure_run_identity(dest, rec)
    (dest / drv.IDENTITY_FILE).chmod(0o640)
    _prior_driver_tree(dest, drv)
    monkeypatch.setattr(drv, "seed_from_previous", orig["seed_from_previous"])
    monkeypatch.setattr(drv, "RunSyncer", orig["RunSyncer"])
    attacker = _Proc(None)
    monkeypatch.setattr(drv, "_start_attacker", lambda port, log: attacker)

    def start_target_dies(tree, srv, geom, **k):
        (srv / "standup.log").write_text("boom\n")
        return _Proc(1), srv / "STOP"
    monkeypatch.setattr(drv, "_start_target", start_target_dies)
    commits = []

    def commit():
        commits.append(1)
        if len(commits) >= 2:                  # 1: the lease record at acquire; 2: the final data sync
            raise RuntimeError("volume commit failed")
    monkeypatch.setattr(drv, "VOL", SimpleNamespace(reload=lambda: None, commit=commit))
    s = call(resume=True, app_name=f"fst-multiturn-{SHA[:8]}-r1-a2")
    assert s["status"] == "FINAL_SYNC_FAILED" and s["status_before_final_sync"] == "TARGET_STANDUP_FAILED"
    assert orig["dict"].d["r1"]["owner"] == s["lease"]["owner"]        # NOT released
    assert s["lease"]["released"] is False and s["lease"]["held_after_exit"] is True
    assert s["lease"]["release_basis"] is None
    assert json.loads((dest / drv.LEASE_FILE).read_bytes())["released_at"] is None
    assert attacker.killed
    # a later launch is refused with the remedy, not resumed
    s2 = call(resume=True, app_name=f"fst-multiturn-{SHA[:8]}-r1-a3")
    assert s2["status"] == "LEASE_HELD" and drv.LEASE_DICT_NAME in s2["lease_err"]


# --------------------------------------------------------------------------- #
# AC-1: no handler on the run path may swallow a lease failure                  #
# --------------------------------------------------------------------------- #

def test_ac1_lease_failures_are_not_runtime_errors_and_health_failures_are_their_own_type(drv):
    """REGRESSION: the attacker-health handler's ``except
    RuntimeError`` caught ``AttemptLeaseLost``/``AttemptLeaseIndeterminate``
    (RuntimeError subclasses), set ATTACKER_FAILED and synced. Now the lease
    failures share a base OUTSIDE RuntimeError, so no ``except RuntimeError``
    anywhere can swallow them, and ``_wait_health`` raises its own type."""
    for cls in (drv.AttemptLeaseHeld, drv.AttemptLeaseLost, drv.AttemptLeaseIndeterminate):
        assert issubclass(cls, drv.AttemptLeaseError) and not issubclass(cls, RuntimeError), cls
    assert issubclass(drv.HealthWaitFailed, RuntimeError)
    dead = SimpleNamespace(poll=lambda: 1, returncode=1)
    with pytest.raises(drv.HealthWaitFailed, match="died"):
        drv._wait_health("http://x", dead, 60, "x", Path("/nonexistent/log"))
    alive = SimpleNamespace(poll=lambda: None, returncode=None)
    with pytest.raises(drv.HealthWaitFailed, match="not healthy"):
        drv._wait_health("http://x", alive, 0, "x", Path("/nonexistent/log"))

    def lost():
        raise drv.AttemptLeaseLost("taken")
    with pytest.raises(drv.AttemptLeaseLost):          # passes straight through the health wait
        drv._wait_health("http://x", alive, 60, "x", Path("/nonexistent/log"), on_tick=lost)


def _reach_attacker_readiness(drv, monkeypatch, commits: list):
    """Stubs carrying the real run body to step 4 (the attacker health wait): a
    target that stands up at once with an endpoint, an attestation that passes,
    an attacker that never answers /health. ``probed`` records the commit count
    at each health probe, so "no commit after the probe started" is checkable."""
    attacker, target = _Proc(None), _Proc(None)
    monkeypatch.setattr(drv, "_start_attacker", lambda port, log: attacker)

    def start_target(tree, srv, geom, **k):
        rt = srv / "routing_table.json"
        rt.write_text("{}")
        (srv / "endpoint.json").write_text(json.dumps(
            {"proxy_url": "http://127.0.0.1:1/v1", "routing_table": str(rt)}))
        return target, srv / "STOP"
    monkeypatch.setattr(drv, "_start_target", start_target)
    proxy_stub = types.ModuleType("sisteer.capeval.serving.proxy")
    proxy_stub.load_routing_table = lambda p: {"routes": []}
    monkeypatch.setitem(sys.modules, "sisteer.capeval.serving.proxy", proxy_stub)
    monkeypatch.setattr(drv, "check_attestation", lambda *a, **k: [])
    probed = []
    monkeypatch.setattr(drv, "_http_ok", lambda url, timeout=5.0: probed.append(len(commits)) or False)
    monkeypatch.setattr(drv.time, "sleep", _bounded_sleep(50))
    return attacker, target, probed


@pytest.mark.parametrize("kind", ["lost", "indeterminate"])
def test_ac1_impl_a_lease_failure_during_attacker_readiness_never_syncs(drv, tmp_path, monkeypatch, kind):
    """REGRESSION through the real run body: a lease failure raised
    by the heartbeat INSIDE the attacker health wait ends the run as
    LEASE_LOST / LEASE_INDETERMINATE — never ATTACKER_FAILED — with no sync and
    no commit after the probe started, the tree untouched, everything killed."""
    call, dest, rec, orig = _impl_harness(drv, tmp_path, monkeypatch)
    drv.ensure_run_identity(dest, rec)
    _prior_driver_tree(dest, drv)
    monkeypatch.setattr(drv, "seed_from_previous", orig["seed_from_previous"])
    monkeypatch.setattr(drv, "RunSyncer", orig["RunSyncer"])
    monkeypatch.setattr(drv, "_sync", lambda syncer: pytest.fail("SYNC_AFTER_LEASE_FAILURE"))
    commits = []
    monkeypatch.setattr(drv, "VOL", SimpleNamespace(reload=lambda: None, commit=lambda: commits.append(1)))
    attacker, target, probed = _reach_attacker_readiness(drv, monkeypatch, commits)
    exc = drv.AttemptLeaseLost if kind == "lost" else drv.AttemptLeaseIndeterminate

    def beat(self):
        if probed:                          # the health probe has started: we are inside step 4
            self.held = False
            setattr(self, kind, True)
            raise exc(f"injected during attacker readiness ({kind})")
    monkeypatch.setattr(drv.AttemptLease, "beat", beat)
    before = _tree_state(dest)
    s = call(resume=True, app_name=f"fst-multiturn-{SHA[:8]}-r1-a2")
    assert s["status"] == ("LEASE_LOST" if kind == "lost" else "LEASE_INDETERMINATE")
    assert "injected during attacker readiness" in s["lease_err"] and "attacker_err" not in s
    assert "attestation_ok" in s["timeline"] and "attacker_up" not in s["timeline"]
    assert probed and len(commits) == probed[0]                     # no commit after the probe began
    assert attacker.killed and target.killed and (tmp_path / "local" / "r1" / "srv" / "STOP").exists()
    after = _tree_state(dest)
    after.pop(drv.LEASE_FILE, None)
    assert after == before                                          # nothing synced
    assert s["lease"]["released"] is False


def test_lease_docstring_states_the_restriction_not_a_deadline(drv):
    """The "<= 60 s, before any durable write" claim is gone; the code
    states the restriction (overlapping resumptions are not supported; a stale
    lease needs deliberate cleanup) and that sixty seconds is a heartbeat
    interval, never a deadline."""
    doc = drv.__doc__ + drv.AttemptLease.__doc__
    assert "60 s, before any durable write" not in doc and "<= 60 s" not in doc and "≤ 60 s" not in doc
    assert "overlapping resumptions are not supported" in doc
    assert "heartbeat interval" in doc and "never a deadline" in doc
    assert "takeover" not in doc.lower().replace("no automatic takeover", "").replace("taken over automatically", "")


# --------------------------------------------------------------------------- #
# The lease is released on POSITIVE authorization only                         #
# --------------------------------------------------------------------------- #

def _reach_driver_spawn(drv, monkeypatch):
    """Stubs carrying the real run body to step 5 (the driver spawn): a target
    that stands up at once with an endpoint, an attestation that passes, an
    attacker that is healthy with a verified serving manifest, driver argv from
    a stub. Returns ``(attacker, target)``."""
    attacker, target = _Proc(None), _Proc(None)

    def start_attacker(port, log):
        log.write_text("attacker up\n")
        log.with_suffix(".serving.json").write_text(json.dumps({"requested": {"thinking": "off"}}))
        return attacker
    monkeypatch.setattr(drv, "_start_attacker", start_attacker)

    def start_target(tree, srv, geom, **k):
        rt = srv / "routing_table.json"
        rt.write_text("{}")
        (srv / "endpoint.json").write_text(json.dumps(
            {"proxy_url": "http://127.0.0.1:1/v1", "routing_table": str(rt)}))
        return target, srv / "STOP"
    monkeypatch.setattr(drv, "_start_target", start_target)
    proxy_stub = types.ModuleType("sisteer.capeval.serving.proxy")
    proxy_stub.load_routing_table = lambda p: {"routes": []}
    monkeypatch.setitem(sys.modules, "sisteer.capeval.serving.proxy", proxy_stub)
    monkeypatch.setattr(drv, "check_attestation", lambda *a, **k: [])
    monkeypatch.setattr(drv, "_http_ok", lambda url, timeout=5.0: True)
    monkeypatch.setattr(drv, "_driver_cmd", lambda tree, **k: ["true"])
    monkeypatch.setattr(drv.time, "sleep", _bounded_sleep(50))
    return attacker, target


class _SpawnFails:
    """``subprocess`` as the run body sees it, with ``Popen`` raising OSError: a
    driver that cannot be started (the closing review's execution). The stdlib
    module itself is left alone."""
    STDOUT = subprocess.STDOUT
    TimeoutExpired = subprocess.TimeoutExpired

    def __init__(self):
        self.attempts = 0

    def Popen(self, *a, **k):
        self.attempts += 1
        raise OSError(12, "Cannot allocate memory")


def _spy_release(drv, monkeypatch) -> list:
    """Every ``AttemptLease.release`` call's outcome, in order."""
    released = []
    real = drv.AttemptLease.release

    def release(self, outcome):
        released.append(outcome)
        return real(self, outcome)
    monkeypatch.setattr(drv.AttemptLease, "release", release)
    return released


def test_impl_an_exception_after_a_failed_copy_keeps_the_lease_held(drv, tmp_path, monkeypatch, capsys):
    """REGRESSION through the real run body: the
    intermediate copy at attacker readiness FAILS (counted, by design), then
    starting a driver raises OSError. Before, the general handler re-raised and
    the ``finally`` — gated on the ABSENCE of a recorded final-sync failure —
    stamped, committed and deleted the entry with no successful final data
    sync ever having happened, over a destination the failed copy may have left
    truncated. Now the release needs POSITIVE authorization: the writers are
    stopped and ONE owned final sync is attempted; it fails too, so the run
    says FINAL_SYNC_FAILED loudly, ``release`` is never called, the exception
    still propagates, and the lease stays HELD — the entry this attempt's, no
    release stamp, the next launch refused with the remedy."""
    call, dest, rec, orig = _impl_harness(drv, tmp_path, monkeypatch)
    drv.ensure_run_identity(dest, rec)
    (dest / drv.IDENTITY_FILE).chmod(0o640)
    _prior_driver_tree(dest, drv)
    monkeypatch.setattr(drv, "seed_from_previous", orig["seed_from_previous"])
    monkeypatch.setattr(drv, "RunSyncer", orig["RunSyncer"])
    commits = []
    monkeypatch.setattr(drv, "VOL", SimpleNamespace(reload=lambda: None, commit=lambda: commits.append(1)))
    attacker, target = _reach_driver_spawn(drv, monkeypatch)
    copies = []

    def copy_fails(self):                    # every copy fails: the destination cannot be completed
        copies.append((attacker.killed, target.killed))
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(drv.RunSyncer, "copy_changed", copy_fails)
    spawn = _SpawnFails()
    monkeypatch.setattr(drv, "subprocess", spawn)
    released = _spy_release(drv, monkeypatch)
    app = f"fst-multiturn-{SHA[:8]}-r1-a2"
    with pytest.raises(OSError, match="Cannot allocate memory"):
        call(resume=True, app_name=app)
    assert released == []                    # release was never even attempted ...
    entry = orig["dict"].d["r1"]             # ... the entry is still this attempt's ...
    record = json.loads((dest / drv.LEASE_FILE).read_bytes())
    assert entry["owner"] == record["owner"] and record["app"] == app and record["released_at"] is None
    assert len(commits) == 1                 # ... and nothing was committed after the acquire record
    assert spawn.attempts == 1
    # the intermediate copy ran with the writers live; the owned final copy only after they were stopped
    assert copies == [(False, False), (True, True)]
    assert attacker.killed and target.killed and (tmp_path / "local" / "r1" / "srv" / "STOP").exists()
    local_summary = json.loads((tmp_path / "local" / "r1" / "summary.json").read_text())
    assert local_summary["status"] == "EXCEPTION:OSError" and "Cannot allocate memory" in local_summary["exception"]
    assert local_summary["sync_failures"] == 1
    out = capsys.readouterr().out
    assert "FINAL_SYNC_FAILED" in out and "stays HELD" in out and drv.LEASE_DICT_NAME in out
    # a later launch is refused with the remedy, not resumed
    s2 = call(resume=True, app_name=f"fst-multiturn-{SHA[:8]}-r1-a3")
    assert s2["status"] == "LEASE_HELD" and drv.LEASE_DICT_NAME in s2["lease_err"]
    assert released == []


def test_impl_an_exception_after_data_began_releases_only_on_a_committed_final_sync(drv, tmp_path, monkeypatch):
    """Through the real run body: the intermediate copy fails ONCE — leaving a
    TRUNCATED ``run_provenance.json`` in the destination, copied in place —
    then a driver spawn raises. The handler stops every writer FIRST, then runs
    one owned final sync; this one copies (the truncated file re-copied whole)
    and commits, and THAT authorizes the release: stamped with the exception
    as the outcome, the entry deleted, the destination carrying this attempt's
    ``summary.json`` with the exception; the exception still propagates."""
    call, dest, rec, orig = _impl_harness(drv, tmp_path, monkeypatch)
    drv.ensure_run_identity(dest, rec)
    (dest / drv.IDENTITY_FILE).chmod(0o640)
    _prior_driver_tree(dest, drv)
    monkeypatch.setattr(drv, "seed_from_previous", orig["seed_from_previous"])
    monkeypatch.setattr(drv, "RunSyncer", orig["RunSyncer"])
    commits = []
    monkeypatch.setattr(drv, "VOL", SimpleNamespace(reload=lambda: None, commit=lambda: commits.append(1)))
    attacker, target = _reach_driver_spawn(drv, monkeypatch)
    copies = []
    real_copy = drv.RunSyncer.copy_changed

    def copy_once_torn(self):
        copies.append((attacker.killed, target.killed))
        if len(copies) == 1:                 # the transient fault: a partial in-place write, then the error
            (self.dest / "run_provenance.json").write_bytes(b'{"run_id": "r1", "tree_sha": "a')
            raise OSError(5, "Input/output error")
        return real_copy(self)
    monkeypatch.setattr(drv.RunSyncer, "copy_changed", copy_once_torn)
    spawn = _SpawnFails()
    monkeypatch.setattr(drv, "subprocess", spawn)
    released = _spy_release(drv, monkeypatch)
    app = f"fst-multiturn-{SHA[:8]}-r1-a2"
    with pytest.raises(OSError, match="Cannot allocate memory"):
        call(resume=True, app_name=app)
    assert copies == [(False, False), (True, True)]      # the writers stopped before the owned final copy
    assert released == ["EXCEPTION:OSError"] and "r1" not in orig["dict"].d
    record = json.loads((dest / drv.LEASE_FILE).read_bytes())
    assert record["app"] == app and record["released_at"] and record["outcome"] == "EXCEPTION:OSError"
    assert len(commits) == 3                              # the acquire record, the final data sync, the release stamp
    prov = json.loads((dest / "run_provenance.json").read_text())   # the torn file was re-copied whole ...
    assert prov == json.loads((tmp_path / "local" / "r1" / "run_provenance.json").read_text())
    published = json.loads((dest / "summary.json").read_text())    # ... and the exception is on the volume
    assert published["status"] == "EXCEPTION:OSError" and "Cannot allocate memory" in published["exception"]
    assert published["sync_failures"] == 1


def test_impl_a_failure_before_the_first_data_copy_releases_naming_the_basis(drv, tmp_path, monkeypatch, capsys):
    """The explicitly named set that may still release without a committed
    final sync: a failure BEFORE this attempt's first data copy. The target
    spawn raises after the attacker started: no sync ever runs (``_sync`` would
    fail the test), the attacker is killed, the lease is released with the
    exception as the outcome and the run says why — nothing of this attempt's
    is under the run dir but the identity and the lease record."""
    call, dest, rec, orig = _impl_harness(drv, tmp_path, monkeypatch)
    drv.ensure_run_identity(dest, rec)
    _prior_driver_tree(dest, drv)
    monkeypatch.setattr(drv, "seed_from_previous", orig["seed_from_previous"])
    monkeypatch.setattr(drv, "RunSyncer", orig["RunSyncer"])
    monkeypatch.setattr(drv, "_sync", lambda syncer: pytest.fail("SYNC_BEFORE_ANY_DATA"))
    attacker = _Proc(None)
    monkeypatch.setattr(drv, "_start_attacker", lambda port, log: attacker)

    def no_target(tree, srv, geom, **k):
        raise RuntimeError("no second gpu")
    monkeypatch.setattr(drv, "_start_target", no_target)
    released = _spy_release(drv, monkeypatch)
    before = _tree_state(dest)
    app = f"fst-multiturn-{SHA[:8]}-r1-a2"
    with pytest.raises(RuntimeError, match="no second gpu"):
        call(resume=True, app_name=app)
    assert attacker.killed
    assert released == ["EXCEPTION:RuntimeError"] and "r1" not in orig["dict"].d
    record = json.loads((dest / drv.LEASE_FILE).read_bytes())
    assert record["app"] == app and record["released_at"] and record["outcome"] == "EXCEPTION:RuntimeError"
    after = _tree_state(dest)
    after.pop(drv.LEASE_FILE, None)
    assert after == before                                # nothing under the run dir but the lease record
    assert "copied no data under" in capsys.readouterr().out


def test_docstring_states_positive_release_and_the_checkpoint_limitation_without_a_loss_bound(drv):
    """The closing review's corrections, carried into the code's own prose: the
    release is on positive authorization and the pre-data set that may still
    release is named; (a) repeating unpublished work may change the sample (a
    repeated goal is a fresh stochastic attack — a lost refusal need not refuse
    again), so the "simply redone ... nothing committed is contradicted" claim
    is gone; (b) the poll/sync intervals are cadence, not a bound, naming the
    300 s teardown wait with no sync inside it."""
    doc = drv.__doc__
    assert "POSITIVE authorization" in doc and "first data copy" in doc
    for gone in ("simply redone", "nothing committed is contradicted", "never published"):
        assert gone not in doc, gone
    assert "committed terminal rows" in doc and "may be REPEATED" in doc and "change the sample" in doc
    assert "fresh stochastic attack" in doc and "need not refuse again" in doc
    assert "cadence" in doc and "not a bound" in doc and "300 s" in doc and "no periodic sync" in doc


# --------------------------------------------------------------------------- #
# staging: verify_checkout under git's dubious-ownership refusal               #
# --------------------------------------------------------------------------- #
# The 2026-09-07 staging failure: ``--stage`` unpacked the tarball on the volume
# and ``verify_checkout`` reported ``{"head": null, "clean": false, "why": "not a
# git checkout"}``. The tree WAS a checkout; git refused to read it. The tarball
# is built here as a shallow clone, so ``tarfile`` records this machine's uid on
# every entry, and the container's root unpack (``filter="fully_trusted"``)
# chowns them back to it — a repository owned by another user, which git rejects
# with ``fatal: detected dubious ownership`` (rc 128, EMPTY stdout) for every
# command. ``_git`` returned that empty stdout, and the code read it as "not a
# git checkout". lens_vectorize_gate.py stages the identical way and succeeds
# only because its image runs ``git config --global --add safe.directory '*'``.

def _git_repo_at(tmp_path: Path, name: str = "sisteer-x") -> tuple[Path, str]:
    """A one-commit repository with a tracked file; returns (path, HEAD sha).
    Built BEFORE any test poisons the environment."""
    repo = tmp_path / name
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "tracked.txt").write_text("x\n")
    subprocess.run(["git", "-C", str(repo), "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@example.invalid",
                    "-c", "user.name=t", "commit", "-q", "-m", "init"], check=True)
    sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=True).stdout.strip()
    assert len(sha) == 40
    return repo, sha


def _make_git_refuse_every_repo(monkeypatch, tmp_path: Path) -> None:
    """The container's view of the unpacked tree, without a container: git's own
    switch for "this repository belongs to another user"
    (``GIT_TEST_ASSUME_DIFFERENT_OWNER``), with NO safe.directory anywhere — an
    empty global config, the system config ignored, HOME pointed away."""
    home = tmp_path / "home"
    home.mkdir()
    (home / "gitconfig").write_text("")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(home / "gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_TEST_ASSUME_DIFFERENT_OWNER", "1")


def _refusal_bites_or_skip(repo: Path) -> None:
    """The control: a BARE ``git rev-parse HEAD`` in the poisoned environment
    must refuse with git's dubious-ownership message, or this git has no such
    switch and the runtime behaviour is only provable in the container."""
    bare = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo),
                          capture_output=True, text=True)
    if bare.returncode == 0 or "dubious ownership" not in bare.stderr:
        pytest.skip("this git ignores GIT_TEST_ASSUME_DIFFERENT_OWNER; the dubious-ownership "
                    "behaviour is only provable in the container")
    assert bare.stdout.strip() == ""          # the EMPTY stdout the old code misread


def test_verify_checkout_survives_gits_dubious_ownership_refusal(drv, tmp_path, monkeypatch):
    """The staging failure, reproduced and fixed: a clean checkout at the sha
    that git refuses to read (another user's repository) still verifies
    ``ok`` — every git call carries the repo's own safe.directory, needing
    nothing from the image."""
    repo, sha = _git_repo_at(tmp_path)
    _make_git_refuse_every_repo(monkeypatch, tmp_path)
    _refusal_bites_or_skip(repo)
    assert drv.verify_checkout(repo, sha) == {"ok": True, "head": sha, "clean": True, "why": None}


def test_verify_checkout_reports_modified_files_under_the_refusal_not_a_git_error(
        drv, tmp_path, monkeypatch):
    """The SECOND git call (``diff --quiet HEAD``) is covered too: under the
    refusal a bare ``git diff`` exits 129 ("Not a git repository"), which must
    not be mistaken for anything — a dirty tree still reads as dirty."""
    repo, sha = _git_repo_at(tmp_path)
    (repo / "tracked.txt").write_text("changed\n")
    _make_git_refuse_every_repo(monkeypatch, tmp_path)
    _refusal_bites_or_skip(repo)
    v = drv.verify_checkout(repo, sha)
    assert v == {"ok": False, "head": sha, "clean": False, "why": "modified tracked files"}


def test_verify_checkout_names_gits_own_reason_when_rev_parse_fails(drv, tmp_path, monkeypatch):
    """Legibility: when ``rev-parse`` fails outright, ``why`` carries git's
    stderr — the real cause — instead of the guess "not a git checkout" that
    hid the ownership refusal for a whole staging round."""
    not_a_repo = tmp_path / "sisteer-x"
    not_a_repo.mkdir()
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))   # never find a parent repo
    v = drv.verify_checkout(not_a_repo, "0" * 40)
    assert v["ok"] is False and v["head"] is None and v["clean"] is False
    assert "not a git checkout" not in v["why"]
    assert "rev-parse" in v["why"] and "not a git repository" in v["why"], v["why"]


def test_every_verify_checkout_git_call_carries_the_repos_safe_directory(drv, tmp_path, monkeypatch):
    """The argv pin, valid on any git: both commands ``verify_checkout`` runs
    open with ``-c safe.directory=<the resolved repo path>`` — the specific
    path, not ``*`` — before anything else."""
    repo, sha = _git_repo_at(tmp_path)
    seen: list[list[str]] = []
    real_run = subprocess.run

    def spy(argv, *a, **k):
        seen.append(list(argv))
        return real_run(argv, *a, **k)
    monkeypatch.setattr(drv.subprocess, "run", spy)
    assert drv.verify_checkout(repo, sha)["ok"] is True
    assert [argv[3:] for argv in seen] == [["rev-parse", "HEAD"], ["diff", "--quiet", "HEAD", "--"]]
    for argv in seen:
        assert argv[:3] == ["git", "-c", f"safe.directory={repo.resolve()}"], argv


def test_image_sets_the_global_safe_directory_like_the_lens_gate(drv):
    """Parity with lens_vectorize_gate.py's image: the global
    ``safe.directory '*'`` covers the git the STAGED TREE's own code runs
    outside ``_git`` (sisteer's ``run_identity.tree_sha``), which the per-call
    flag cannot reach. Two pins, neither depending on what else imported the
    module first: the command is the module's OWN data, the gate script's line
    byte for byte; and an import under a recording stub of this test's own
    shows the image chain issuing exactly that command as its last layer —
    so deleting the setting from the image fails here even with the constant
    left in place."""
    gate_line = "git config --global --add safe.directory '*'"
    assert drv.IMAGE_GIT_SAFE_DIRECTORY_CMD == gate_line
    gate_src = (REPO / "scripts" / "modal" / "lens_vectorize_gate.py").read_text()
    assert gate_line in gate_src, "the gate's image no longer carries the line this mirrors"
    seen: list[str] = []
    mod = load_driver(run_commands_seen=seen)
    assert mod.IMAGE_GIT_SAFE_DIRECTORY_CMD == gate_line
    assert seen, "the image chain issued no run_commands under the recording stub"
    assert seen[-1] == gate_line, seen          # its own layer, appended last
    assert seen.count(gate_line) == 1, seen


# --------------------------------------------------------------------------- #
# preflight: the driver-argv check exercises the real seam                     #
# --------------------------------------------------------------------------- #
# The 2026-09-07 preflight (the first to reach its checks, once staging
# verified): every check passed but ``driver_argv``, which died in argparse —
# ``argument cmd: invalid choice: '/vol/.../scripts/jailbreak_run.py' (choose
# from 'generate', 'aggregate', 'canary')``. The check took ``_driver_cmd(...)
# [1:]`` (script path kept) and PREPENDED a fake argv[0], so the script path sat
# in argv[1], the subcommand slot. It could never have passed, and it was a
# closure inside the Modal-only ``preflight``, so nothing could run it here. It
# is now a module-level function, driven below against THIS checkout as the tree
# (the config exists here; ``jailbreak_run`` imports from ``scripts/``).

def test_driver_argv_check_parses_the_real_driver_cmd_and_resolves_every_default_cell(drv):
    """Through the real ``_driver_cmd`` and the real ``jailbreak_run.main()``
    (``cmd_generate`` captured), then the resolution ``cmd_generate`` performs
    first (``_valid_defense_ids`` + ``_check_defense_id``): every default cell
    is ACCEPTED and RESOLVES to its ``(base, variant)`` — a variant spelling
    like ``safety_refusal__psr_kl_all@all_turns`` included — not merely parsed."""
    for cell in drv.parse_cells(""):
        r = drv.driver_argv_check(str(REPO), cell)
        base, variant = cell.split("@") if "@" in cell else (cell, None)
        assert r["resolved"] == {"base": base, "variant": variant}, (cell, r)
        assert r["defense"] == cell and r["argv"][1] == "generate"
        assert r["positions"] == ("response" if cell == TT else None), (cell, r)


def test_driver_argv_check_hands_main_the_driver_processs_own_argv(drv, monkeypatch):
    """The seam, pinned against the builder rather than a literal: what
    ``jailbreak_run.main()`` sees is exactly ``_driver_cmd(...)[1:]`` — the argv
    of a driver launched as ``[interpreter, script, *args]``: the script in
    argv[0], ``generate`` in argv[1] where argparse reads the subcommand, no
    interpreter anywhere."""
    import jailbreak_run as jr
    seen: list[list[str]] = []
    real_main = jr.main

    def spy():
        seen.append(list(sys.argv))
        real_main()
    monkeypatch.setattr(jr, "main", spy)
    cell = f"{TT}@all_turns"
    drv.driver_argv_check(str(REPO), cell)
    cmd = drv._driver_cmd(str(REPO), cell=cell, **drv.PREFLIGHT_DRIVER_PLACEHOLDERS)
    assert seen == [cmd[1:]]
    assert seen[0][0].endswith("/scripts/jailbreak_run.py") and seen[0][1] == "generate"
    assert sys.executable not in seen[0]


def test_driver_argv_check_refuses_an_argv_whose_subcommand_slot_holds_a_path(drv, monkeypatch):
    """The 2026-09-07 failure, reproduced THROUGH the check: one extra leading
    element (the interpreter kept) puts the script path in the subcommand slot
    and the check fails as the preflight did (argparse's SystemExit 2) instead
    of passing."""
    real = drv._driver_cmd
    monkeypatch.setattr(drv, "_driver_cmd", lambda tree, **k: [sys.executable] + real(tree, **k))
    with pytest.raises(SystemExit) as ei:
        drv.driver_argv_check(str(REPO), f"{TT}@all_turns")
    assert ei.value.code == 2


def test_driver_argv_check_runs_the_real_resolution_not_just_argparse(drv, monkeypatch):
    """Not vacuous: a builder that put ``--positions`` beside a variant would
    PARSE fine and be refused by ``jailbreak_run``'s own ``_check_defense_id``
    (the variant IS the positions) — the check must surface that refusal, so it
    proves the builder and the driver's validator agree, not that argparse ran."""
    real = drv._driver_cmd
    monkeypatch.setattr(drv, "_driver_cmd",
                        lambda tree, **k: real(tree, **k) + ["--positions", "response"])
    with pytest.raises(SystemExit, match="not accepted beside a serve variant"):
        drv.driver_argv_check(str(REPO), f"{TT}@all_turns")
