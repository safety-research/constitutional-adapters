"""The CPU-suite runner's staged-tree pruning (scripts/modal/cpu_suite.py, 2026-09-11;
unrecorded clause 2026-09-13): /vol/repos/sisteer-<40 hex> is SHARED with two other
Modal apps (lens_vectorize_gate, petri_smoke) that read their tree in place at run
time, so the runner deletes only (a) the tree it created itself once its results are
recorded, (b) recorded trees older than a cutoff, and (c) UNRECORDED trees older than
a far longer cutoff (7 d; None = never). Everything else under the staging dir —
recent trees, cfgs, stray files, abbreviated-sha names — is never touched. Pure
planner + filesystem sweep, under a Modal stub (no SDK, no volume).

And the RESULTS tree's retention (2026-09-18): once the finalizer has committed the
merged summary.json, the PASSED shards' shards/<i>-of-<N>/ directories go and only
those — a FAIL / TIMEOUT / SHARD_ERROR shard's log and JUnit stay, summary.json always
stays, no other sha and nothing outside shards/ is touched, a directory already gone is
not an error."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


class _Chain:
    def __getattr__(self, name):
        return lambda *a, **k: self


def _load():
    fake = types.ModuleType("modal")

    class FakeApp:
        def __init__(self, name):
            self.description = None
            self.app_id = "ap-test"

        def function(self, *a, **k):
            return lambda f: f

        def local_entrypoint(self, *a, **k):
            return lambda f: f
    fake.App = FakeApp
    fake.Volume = types.SimpleNamespace(from_name=lambda *a, **k: types.SimpleNamespace(
        commit=lambda: None, reload=lambda: None))
    fake.Secret = types.SimpleNamespace(from_name=lambda *a, **k: object())
    fake.Image = types.SimpleNamespace(debian_slim=lambda *a, **k: _Chain())
    fake.is_local = lambda: False          # the import-time dependency-hash guard is the image's, not this test's
    old = sys.modules.get("modal")
    sys.modules["modal"] = fake
    try:
        spec = importlib.util.spec_from_file_location("_cpu_suite_under_test", REPO / "scripts/modal/cpu_suite.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        if old is None:
            sys.modules.pop("modal", None)
        else:
            sys.modules["modal"] = old


@pytest.fixture(scope="module")
def cs():
    return _load()


def _sha(i: int) -> str:
    return f"{i:040x}"


def test_prune_candidates_is_recorded_and_aged_and_not_kept(cs):
    now = 1_000_000.0
    h = 3600.0
    staged = {_sha(1): now - 100 * h, _sha(2): now - 100 * h, _sha(3): now - 1 * h, _sha(4): now - 100 * h}
    recorded = {_sha(1), _sha(3), _sha(4)}          # 2 has no summary
    got = cs.prune_candidates(staged, recorded, now=now, older_than_s=48 * h, keep={_sha(4)})
    assert got == [_sha(1)]                          # 2 unrecorded, 3 too recent, 4 kept
    assert cs.prune_candidates(staged, recorded, now=now, older_than_s=0.0, keep=set()) == sorted([_sha(1), _sha(3), _sha(4)])
    assert cs.prune_candidates({}, recorded, now=now, older_than_s=0.0, keep=set()) == []
    assert cs.PRUNE_OLDER_THAN_HOURS_DEFAULT == 48.0
    # the unrecorded clause: absent (None) it never fires; given, unrecorded trees older than ITS cutoff go too
    staged[_sha(5)] = now - 200 * h                                                   # unrecorded, 200 h old
    assert cs.prune_candidates(staged, recorded, now=now, older_than_s=48 * h, keep=set()) == [_sha(1), _sha(4)]
    got = cs.prune_candidates(staged, recorded, now=now, older_than_s=48 * h, keep=set(),
                              unrecorded_older_than_s=168 * h)
    assert got == [_sha(1), _sha(4), _sha(5)]                                          # 2 is unrecorded but only 100 h old
    got = cs.prune_candidates(staged, recorded, now=now, older_than_s=48 * h, keep={_sha(5)},
                              unrecorded_older_than_s=168 * h)
    assert got == [_sha(1), _sha(4)]                                                  # keep wins over the clause too
    assert cs.PRUNE_UNRECORDED_OLDER_THAN_HOURS_DEFAULT == 168.0


def _psha(i: int) -> str:
    """A 40-hex sha whose FIRST seven hex digits differ per ``i`` — the abandoned
    clause keys liveness on ``sha[:7]``, and ``_sha`` pads on the left."""
    return f"{i:07x}" + "0" * 33


def test_prune_candidates_abandoned_clause_keys_on_live_gate_apps(cs):
    now = 1_000_000.0
    h = 3600.0
    staged = {_psha(1): now - 20 * h,      # unrecorded, its gate app is live -> never abandoned
              _psha(2): now - 20 * h,      # unrecorded, no gate app -> abandoned past 12 h
              _psha(3): now - 5 * h,       # unrecorded, no gate app, too young -> stays
              _psha(4): now - 20 * h}      # recorded, under 48 h -> stays (the recorded rule, untouched)
    recorded = {_psha(4)}
    kw = dict(now=now, older_than_s=48 * h, keep=set(), abandoned_older_than_s=12 * h)
    assert cs.prune_candidates(staged, recorded, live_gate_shas={_psha(1)[:7]}, **kw) == [_psha(2)]
    assert cs.prune_candidates(staged, recorded, live_gate_shas=set(), **kw) == [_psha(1), _psha(2)]
    # liveness undeterminable (None) switches the clause off whatever its cutoff
    assert cs.prune_candidates(staged, recorded, live_gate_shas=None, **kw) == []
    # no cutoff switches it off too, and keep still wins
    assert cs.prune_candidates(staged, recorded, now=now, older_than_s=48 * h, keep=set(),
                               abandoned_older_than_s=None, live_gate_shas=set()) == []
    assert cs.prune_candidates(staged, recorded, live_gate_shas=set(),
                               now=now, older_than_s=48 * h, keep={_psha(2)}, abandoned_older_than_s=12 * h) == [_psha(1)]
    # the 7-day clause and the abandoned clause compose: a 200 h unrecorded tree goes under either
    staged[_psha(5)] = now - 200 * h
    got = cs.prune_candidates(staged, recorded, now=now, older_than_s=48 * h, keep=set(),
                              unrecorded_older_than_s=168 * h, abandoned_older_than_s=12 * h,
                              live_gate_shas={_psha(1)[:7], _psha(5)[:7]})
    assert got == [_psha(2), _psha(5)]                    # 5's gate is live but it is a week old regardless
    assert cs.PRUNE_ABANDONED_OLDER_THAN_HOURS_DEFAULT == 6.0             # 12 -> 6 on 2026-09-16 (module constant's rationale)


def test_liveness_view_and_live_app_listing_fail_closed(cs, monkeypatch):
    class P:
        def __init__(self, rc, out):
            self.returncode, self.stdout = rc, out
    apps = [{"description": "cpu-suite-abc1234", "state": "ephemeral (detached)"},
            {"description": "cpu-suite-def5678", "state": "stopped"},
            {"description": "twosi-fits", "state": "deployed"},
            {"description": "ifeval-canonical-4d69472", "state": "ephemeral"},
            {"description": "cpu-suite-prune", "state": "ephemeral"}]
    seen = {}
    def fake_run(argv, **k):
        seen["argv"] = argv
        return P(0, json.dumps(apps))
    monkeypatch.setattr(cs.subprocess, "run", fake_run)
    got = cs.live_app_descriptions()
    assert got == ["cpu-suite-abc1234", "cpu-suite-prune", "ifeval-canonical-4d69472"]     # live only, sorted
    assert seen["argv"][1:] == ["-m", "modal", "app", "list", "--json"]
    view = cs.liveness_view(got)
    assert view == {"determinable": True, "why": None, "n_live_apps": 3, "live_gate_shas": ["abc1234"], "tenant_live": []}
    # a live co-tenant makes the clause undeterminable — their trees carry no sha in the app name
    view = cs.liveness_view(got + ["petri-smoke-proxy-gate", "lens-gate-qwen"])
    assert view["determinable"] is False and view["tenant_live"] == ["lens-gate-qwen", "petri-smoke-proxy-gate"]
    assert view["live_gate_shas"] == ["abc1234"] and "petri-smoke-proxy-gate" in view["why"]
    assert cs.liveness_view(None) == {"determinable": False, "why": "live-app list unavailable", "n_live_apps": None,
                                      "live_gate_shas": [], "tenant_live": []}
    # the listing fails CLOSED: a non-zero exit, bad JSON or a missing CLI is None, never []
    monkeypatch.setattr(cs.subprocess, "run", lambda *a, **k: P(1, ""))
    assert cs.live_app_descriptions() is None
    monkeypatch.setattr(cs.subprocess, "run", lambda *a, **k: P(0, "not json"))
    assert cs.live_app_descriptions() is None
    def boom(*a, **k):
        raise OSError("no modal here")
    monkeypatch.setattr(cs.subprocess, "run", boom)
    assert cs.live_app_descriptions() is None


def test_sweep_abandoned_clause_and_staged_stamps(tmp_path, cs):
    repos, results = tmp_path / "repos", tmp_path / "cpu-suite"
    repos.mkdir(); results.mkdir()
    now = time.time()
    running = _tree(repos, _psha(1), age_h=20, now=now)                    # unrecorded, gate app live -> stays
    orphan = _tree(repos, _psha(2), age_h=20, now=now, tarball=True)       # unrecorded, no app -> abandoned
    young = _tree(repos, _psha(3), age_h=2, now=now)                       # unrecorded, no app, young -> stays
    rec_new = _tree(repos, _psha(4), age_h=20, now=now)                    # recorded, under 48 h -> stays
    (results / _psha(4)).mkdir(); (results / _psha(4) / "summary.json").write_text("{}")
    stamp = cs._write_staged_stamp(repos, _psha(2), "cpu-suite-" + _psha(2)[:7], created=True)
    stamp_path = repos / f"sisteer-{_psha(2)}{cs.STAGED_STAMP_SUFFIX}"
    assert stamp_path.is_file() and stamp["staged_by"] == "cpu-suite-" + _psha(2)[:7] and stamp["created"] is True
    # a stamp beside a PRE-EXISTING tree belongs to the original stager and is not overwritten
    again = cs._write_staged_stamp(repos, _psha(2), "cpu-suite-other00", created=False)
    assert again["staged_by"] == "cpu-suite-" + _psha(2)[:7]
    assert cs._read_staged_stamp(repos, _psha(1)) is None
    live = ["cpu-suite-" + _psha(1)[:7], "ifeval-canonical-abc1234", "cpu-suite-prune"]
    kw = dict(older_than_hours=48, keep=set(), now=now, unrecorded_older_than_hours=168, abandoned_older_than_hours=12)
    dry = cs._prune_trees_in(repos, results, dry_run=True, live_apps=live, **kw)
    assert dry["liveness"]["determinable"] is True and dry["liveness"]["live_gate_shas"] == [_psha(1)[:7]]
    assert dry["would_remove"] == [_psha(2)] and dry["would_remove_abandoned"] == [_psha(2)]
    assert dry["would_remove_unrecorded"] == [_psha(2)] and dry["abandoned_older_than_hours"] == 12.0
    assert dry["unrecorded_staged_by"] == {_psha(2): "cpu-suite-" + _psha(2)[:7]}
    assert dry["kept_unrecorded"] == [_psha(1), _psha(3)] and dry["kept_recent_recorded"] == [_psha(4)]
    # the stamp is a FILE next to the tree: never counted as a staged tree
    assert dry["n_staged"] == 4 and dry["n_unrecorded"] == 3
    # a live co-tenant suspends the clause; so does an unavailable listing
    sus = cs._prune_trees_in(repos, results, dry_run=True, live_apps=live + ["petri-smoke-proxy-gate"], **kw)
    assert sus["would_remove"] == [] and sus["liveness"]["determinable"] is False
    assert "petri-smoke-proxy-gate" in sus["liveness"]["why"]
    sus2 = cs._prune_trees_in(repos, results, dry_run=True, live_apps=None, **kw)
    assert sus2["would_remove"] == [] and sus2["liveness"]["why"] == "live-app list unavailable"
    # the 7-day clause still fires while the abandoned one is suspended
    stale = _tree(repos, _psha(5), age_h=200, now=now)
    sus3 = cs._prune_trees_in(repos, results, dry_run=True, live_apps=None, **kw)
    assert sus3["would_remove"] == [_psha(5)] and sus3["would_remove_abandoned"] == []
    shutil_ok = stale.is_dir()
    # the real sweep removes the orphan with its tarball and stamp; everything else stays
    rec = cs._prune_trees_in(repos, results, dry_run=False, live_apps=live, **kw)
    assert rec["removed"] == [_psha(2), _psha(5)] and rec["removed_abandoned"] == [_psha(2)]
    assert not orphan.exists() and not (repos / f"sisteer-{_psha(2)}.tar.gz").exists() and not stamp_path.exists()
    assert running.is_dir() and young.is_dir() and rec_new.is_dir() and shutil_ok and not stale.exists()


def _tree(repos: Path, sha: str, *, age_h: float, now: float, tarball: bool = False) -> Path:
    d = repos / f"sisteer-{sha}"
    (d / ".git").mkdir(parents=True)
    (d / "src.py").write_text("x")
    t = now - age_h * 3600
    for p in (d / ".git", d / "src.py", d):
        os.utime(p, (t, t))
    if tarball:
        (repos / f"sisteer-{sha}.tar.gz").write_bytes(b"gz")
    return d


def test_sweep_removes_only_recorded_aged_trees_and_never_the_rest(tmp_path, cs):
    repos, results = tmp_path / "repos", tmp_path / "cpu-suite"
    repos.mkdir(); results.mkdir()
    now = time.time()
    old_rec = _tree(repos, _sha(1), age_h=100, now=now, tarball=True)      # recorded + old -> goes, with its tarball
    old_unrec = _tree(repos, _sha(2), age_h=100, now=now)                  # old but no summary -> stays
    new_rec = _tree(repos, _sha(3), age_h=2, now=now)                      # recorded but recent -> stays
    kept = _tree(repos, _sha(4), age_h=100, now=now)                       # recorded + old but kept -> stays
    stale_unrec = _tree(repos, _sha(5), age_h=200, now=now, tarball=True)  # no summary, 200 h old -> only the unrecorded clause
    (repos / "sisteer-0123abc").mkdir()                                    # an abbreviated-sha wave tree: never looked at
    for sha in (_sha(1), _sha(3), _sha(4)):
        (results / sha).mkdir(); (results / sha / "summary.json").write_text("{}")
    (repos / "cfgs").mkdir(); (repos / "cfgs" / "a.yaml").write_text("k: v")      # not a tree: never looked at
    (repos / "scale12_serving_up.py").write_text("# stray")
    (repos / "sisteer-notahash").mkdir()                                            # not a 40-hex sha: ignored
    (repos / f"sisteer-{_sha(9)}.tar.gz").write_bytes(b"gz")                       # a tarball with no tree: untouched
    # dry run: the plan, nothing removed; without the clause the unrecorded trees are only REPORTED, with ages
    dry = cs._prune_trees_in(repos, results, older_than_hours=48, dry_run=True, keep={_sha(4)}, now=now)
    assert dry["dry_run"] is True and dry["would_remove"] == [_sha(1)] and dry["would_remove_unrecorded"] == []
    assert dry["n_staged"] == 5 and dry["n_recorded"] == 3 and dry["n_unrecorded"] == 2
    assert dry["unrecorded_older_than_hours"] is None
    assert dry["kept_unrecorded"] == [_sha(2), _sha(5)] and dry["kept_recent_recorded"] == [_sha(3)] and dry["kept_explicitly"] == [_sha(4)]
    assert dry["unrecorded_ages_h"] == {_sha(2): pytest.approx(100, abs=0.11), _sha(5): pytest.approx(200, abs=0.11)}
    assert old_rec.is_dir() and (repos / f"sisteer-{_sha(1)}.tar.gz").is_file()
    # the real sweep under the 2026-09-11 rule alone: the stale unrecorded tree stays
    rec = cs._prune_trees_in(repos, results, older_than_hours=48, dry_run=False, keep={_sha(4)}, now=now)
    assert rec["removed"] == [_sha(1)] and not old_rec.exists() and not (repos / f"sisteer-{_sha(1)}.tar.gz").exists()
    assert rec["removed_unrecorded"] == [] and stale_unrec.is_dir()
    assert old_unrec.is_dir() and new_rec.is_dir() and kept.is_dir()
    assert (repos / "cfgs" / "a.yaml").is_file() and (repos / "scale12_serving_up.py").is_file()
    assert (repos / "sisteer-notahash").is_dir() and (repos / "sisteer-0123abc").is_dir()
    assert (repos / f"sisteer-{_sha(9)}.tar.gz").is_file()
    # the results record itself is never touched
    assert (results / _sha(1) / "summary.json").is_file()
    # the unrecorded clause at 7 d: the 200 h tree goes (with its tarball), the 100 h one stays
    rec2 = cs._prune_trees_in(repos, results, older_than_hours=48, dry_run=False, keep={_sha(4)}, now=now,
                              unrecorded_older_than_hours=168)
    assert rec2["removed"] == [_sha(5)] and rec2["removed_unrecorded"] == [_sha(5)]
    assert rec2["unrecorded_older_than_hours"] == 168.0 and rec2["n_unrecorded"] == 2
    assert not stale_unrec.exists() and not (repos / f"sisteer-{_sha(5)}.tar.gz").exists()
    assert rec2["kept_unrecorded"] == [_sha(2)] and old_unrec.is_dir() and new_rec.is_dir() and kept.is_dir()
    # keep shields an unrecorded tree from the clause as well
    rec3 = cs._prune_trees_in(repos, results, older_than_hours=48, dry_run=True, keep={_sha(2), _sha(4)}, now=now,
                              unrecorded_older_than_hours=1)
    assert rec3["would_remove"] == [] and rec3["kept_explicitly"] == [_sha(2), _sha(4)]


def test_own_tree_prune_removes_exactly_this_shas_tree_and_tarball(tmp_path, cs, monkeypatch):
    repos = tmp_path / "repos"; repos.mkdir()
    monkeypatch.setattr(cs, "REPOS", str(repos))
    now = time.time()
    mine = _tree(repos, _sha(7), age_h=0, now=now, tarball=True)
    other = _tree(repos, _sha(8), age_h=0, now=now, tarball=True)
    cs._write_staged_stamp(repos, _sha(7), "cpu-suite-0000000", created=True)
    cs._write_staged_stamp(repos, _sha(8), "cpu-suite-0000000", created=True)
    out = cs._prune_own_tree(_sha(7))
    assert out["removed_tree"] and out["removed_tarball"] and out["removed_stamp"] and not mine.exists()
    assert not (repos / f"sisteer-{_sha(7)}{cs.STAGED_STAMP_SUFFIX}").exists()
    assert other.is_dir() and (repos / f"sisteer-{_sha(8)}.tar.gz").is_file()
    assert (repos / f"sisteer-{_sha(8)}{cs.STAGED_STAMP_SUFFIX}").is_file()
    again = cs._prune_own_tree(_sha(7))                 # idempotent, loud about nothing
    assert again == {"tree": str(repos / f"sisteer-{_sha(7)}"), "removed_tree": False, "removed_tarball": False,
                     "removed_stamp": False}


def test_entrypoint_wiring_prunes_only_a_tree_it_created(cs):
    src = (REPO / "scripts/modal/cpu_suite.py").read_text()
    assert '"pre_existed": bool(dst.exists())' in src and 'created_here = staged.get("pre_existed") is False' in src
    assert "prune_own = bool(stage and created_here and not keep_tree)" in src
    assert "run_cpu_suite.spawn(sha, run_name, int(timeout_s), prune_own)" in src
    assert 'if run_name != "cpu-suite-prune":' in src
    assert "SHARED with two other Modal apps" in src
    # the unrecorded clause reaches the sweep from the CLI, and <= 0 switches it off
    assert "prune_unrecorded_older_than_hours: float = PRUNE_UNRECORDED_OLDER_THAN_HOURS_DEFAULT" in src
    assert "unrec = unrec if unrec > 0 else None" in src
    assert "unrecorded_older_than_hours=unrec," in src
    assert "unrecorded_older_than_hours: float | None = PRUNE_UNRECORDED_OLDER_THAN_HOURS_DEFAULT," in src
    # the abandoned clause: liveness read on the CLIENT and passed in; <= 0 disables; the stamp names the stager
    assert "prune_abandoned_older_than_hours: float = PRUNE_ABANDONED_OLDER_THAN_HOURS_DEFAULT" in src
    assert "aband = aband if aband > 0 else None" in src
    assert "raw = live_apps_raw()" in src and "live, idle = partition_live_apps(raw, idle_older_than_s=aband * 3600.0)" in src
    # the sharded finish is SERVER-SIDE: the finalizer is spawned with the shard call ids and the client only waits; the
    # client-side merge / upload / remote prune are gone
    assert "fin = finalize_shards.spawn(sha, run_name, [c.object_id for c in calls], prune_own)" in src
    assert "rec = fin.get()" in src and "prune_own_tree_remote" not in src
    assert src.count("VOL.batch_upload") == 1                     # the --stage tarball upload only
    assert "get_call=lambda cid: modal.FunctionCall.from_id(cid).get()" in src
    assert "timeout=FINALIZE_TIMEOUT_S" in src and cs.FINALIZE_TIMEOUT_S == 3 * 3600 > int(2.5 * 3600)
    assert "abandoned_older_than_hours=aband, live_apps=live)" in src
    assert "unpack_tree.remote(sha, force=restage, app_name=run_name)" in src
    assert 'rec["own_tree_pruned"] = _prune_own_tree(sha)' in src
    # the results-tree retention rule: the passed shards' directories go AFTER the merged summary is committed, from
    # the finalizer only, and the module docstring tells a reader that a passing shard's artifacts do not persist
    assert "pruned = _prune_passed_shard_artifacts(Path(results_root), sha, rec)" in src
    assert src.index('os.replace(tmp, dest / "summary.json")') < src.index("pruned = _prune_passed_shard_artifacts(")
    assert src.count("_prune_passed_shard_artifacts(") == 2                  # the def and the finalizer's one call
    assert "PASSED SHARDS' ARTIFACTS DO NOT PERSIST" in cs.__doc__ and cs.SHARD_ARTIFACTS_PRUNED_STATUS == "PASS"


def test_idle_detached_apps_older_than_the_horizon_are_not_live_and_young_or_busy_ones_are(cs):
    import datetime as dt
    now = 1_800_000_000.0
    def at(age_h):
        return dt.datetime.fromtimestamp(now - age_h * 3600, tz=dt.timezone(dt.timedelta(hours=-7))).isoformat(sep=" ")
    apps = [{"description": "cpu-suite-aaaaaaa", "state": "ephemeral (detached)", "tasks": "0", "created_at": at(30)},   # dead client's leftover
            {"description": "cpu-suite-bbbbbbb", "state": "ephemeral (detached)", "tasks": "0", "created_at": at(1)},    # young: a restart gap
            {"description": "cpu-suite-ccccccc", "state": "ephemeral (detached)", "tasks": "1", "created_at": at(30)},   # the finalizer waiting
            {"description": "cpu-suite-ddddddd", "state": "ephemeral", "tasks": "0", "created_at": at(30)},              # attached: the client is alive
            {"description": "cpu-suite-eeeeeee", "state": "ephemeral (detached)", "tasks": "?", "created_at": at(30)},   # unreadable tasks: live (fail closed)
            {"description": "cpu-suite-fffffff", "state": "ephemeral (detached)", "tasks": "0"},                          # no created_at: live (fail closed)
            {"description": "cpu-suite-0000000", "state": "stopped", "tasks": "0", "created_at": at(30)},
            {"description": "twosi-fits", "state": "deployed", "tasks": "0", "created_at": at(900)}]
    live, idle = cs.partition_live_apps(apps, idle_older_than_s=6 * 3600, now=now)
    assert live == ["cpu-suite-bbbbbbb", "cpu-suite-ccccccc", "cpu-suite-ddddddd", "cpu-suite-eeeeeee", "cpu-suite-fffffff"]
    assert idle == [{"description": "cpu-suite-aaaaaaa", "tasks": 0, "age_h": 30.0}]
    # the 2026-09-14 rule (no exclusion) when no horizon is given — every ephemeral app is live
    live0, idle0 = cs.partition_live_apps(apps, now=now)
    assert live0 == sorted(live + ["cpu-suite-aaaaaaa"]) and idle0 == []
    # and the clause then keys on the partitioned list: aaaaaaa's tree is abandoned, ccccccc's is not
    view = cs.liveness_view(live)
    assert view["determinable"] is True and "aaaaaaa" not in view["live_gate_shas"] and "ccccccc" in view["live_gate_shas"]
    # the descriptions API composes listing + partition; a failed listing is still None
    assert cs.live_app_descriptions(["true"], idle_older_than_s=6 * 3600, now=now) in (None, [])   # `true` prints no JSON -> None


def test_finalize_shards_merges_records_the_summary_and_prunes_the_own_tree_even_when_a_shard_call_raised(tmp_path, cs):
    repos, results = tmp_path / "repos", tmp_path / "cpu-suite"
    repos.mkdir(); results.mkdir()
    sha = _sha(3)
    now = time.time()
    tree = _tree(repos, sha, age_h=0, now=now, tarball=True)
    cs._write_staged_stamp(repos, sha, f"cpu-suite-{sha[:7]}", created=True)
    def rec(i, rc, tests, fails=()):
        return {"sha": sha, "app": "cpu-suite-x", "status": "PASS" if rc == 0 else "FAIL", "pytest_rc": rc, "elapsed_s": 10.0 * (i + 1),
                "command": ["py", "-m", "pytest"], "tree": {"ok": True}, "dependency_hashes": {}, "versions": {},
                "counts": {"tests": tests, "failures": len(fails), "errors": 0, "skipped": 0},
                "failures": [{"node": f, "kind": "failure", "message": ""} for f in fails],
                "shard": {"index": i, "count": 3, "n_files": 1, "files": [f"tests/test_{i}.py"]}}
    calls = {"fc-0": rec(0, 0, 100), "fc-1": rec(1, 1, 50, ("tests/test_1.py::t",)), "fc-2": rec(2, 0, 25)}
    commits = []
    vol = types.SimpleNamespace(reload=lambda: commits.append("reload"), commit=lambda: commits.append("commit"))
    # every shard answered: a FAIL gate (one shard rc 1), recorded, own tree pruned (a FAIL's tree goes, as before)
    out = cs._finalize_shards_in(sha, "cpu-suite-x", ["fc-0", "fc-1", "fc-2"], True, get_call=lambda cid: calls[cid],
                                 repos_dir=repos, results_root=results, vol=vol)
    assert out["status"] == "FAIL" and out["pytest_rc"] == 1 and out["counts"]["tests"] == 175 and out["shards"] == 3
    assert [f["node"] for f in out["failures"]] == ["tests/test_1.py::t"] and out["finalized_by"] == "finalize_shards"
    written = json.loads((results / sha / "summary.json").read_text())
    assert written["status"] == "FAIL" and written["shards"] == 3 and "finalized_by" not in written   # the record on disk is the merge
    assert not (results / sha / "summary.json.tmp").exists()
    assert out["own_tree_pruned"] == {"tree": str(repos / f"sisteer-{sha}"), "removed_tree": True, "removed_tarball": True, "removed_stamp": True}
    assert not tree.exists() and not (repos / f"sisteer-{sha}.tar.gz").exists() and not (repos / f"sisteer-{sha}{cs.STAGED_STAMP_SUFFIX}").exists()
    assert commits == ["reload", "commit", "commit"]                     # summary committed BEFORE the tree goes
    # a shard whose CALL raised (container crash / function timeout): merged as SHARD_ERROR, not a pass, still recorded
    sha2 = _sha(4)
    tree2 = _tree(repos, sha2, age_h=0, now=now)
    def get2(cid):
        if cid == "fc-1":
            raise RuntimeError("Function call timed out")
        return {**calls[cid], "sha": sha2}
    out2 = cs._finalize_shards_in(sha2, "cpu-suite-y", ["fc-0", "fc-1", "fc-2"], True, get_call=get2,
                                  repos_dir=repos, results_root=results, vol=None)
    assert out2["status"] == cs.SHARD_ERROR_STATUS == "SHARD_ERROR" and out2["pytest_rc"] is None
    assert "RuntimeError: Function call timed out" in out2["error"] and out2["shard_records"][1]["status"] == "SHARD_ERROR"
    assert json.loads((results / sha2 / "summary.json").read_text())["status"] == "SHARD_ERROR"
    assert not tree2.exists()                                              # re-stageable; nothing reads a failed gate's tree
    # prune_own False (a pre-existing tree, or --keep-tree): recorded, tree kept
    sha3 = _sha(5)
    tree3 = _tree(repos, sha3, age_h=0, now=now)
    out3 = cs._finalize_shards_in(sha3, "cpu-suite-z", ["fc-0"], False, get_call=lambda cid: {**calls[cid], "sha": sha3, "shard": {"index": 0, "count": 1, "files": ["a"]}},
                                  repos_dir=repos, results_root=results, vol=None)
    assert out3["status"] == "PASS" and "own_tree_pruned" not in out3 and tree3.is_dir()
    assert (results / sha3 / "summary.json").is_file()


# --- the RESULTS tree: passed shards' artifacts do not persist (2026-09-18) ---------------------------------------

_SHARD_FILES = ("pytest.log", "junit.xml", "summary.json")


def _shard_dir(results: Path, sha: str, i: int, n: int, files=_SHARD_FILES) -> Path:
    """What run_cpu_suite leaves under <results_root>/<sha>/shards/<i>-of-<n>/ (a TIMEOUT shard may lack junit.xml)."""
    d = results / sha / "shards" / f"{i}-of-{n}"
    d.mkdir(parents=True)
    for name in files:
        (d / name).write_text(f"{name} of shard {i} of {n} at {sha[:7]}")
    return d


def _shard_call(sha: str, i: int, n: int, status: str = "PASS", fails=()) -> dict:
    rc = {"PASS": 0, "FAIL": 1, "TIMEOUT": -9}[status]
    counts = None if status == "TIMEOUT" else {"tests": 100 + i, "failures": len(fails), "errors": 0, "skipped": 0}
    return {"sha": sha, "app": f"cpu-suite-{sha[:7]}", "status": status, "pytest_rc": rc, "elapsed_s": 60.0 + i,
            "command": ["py", "-m", "pytest"], "tree": {"ok": True}, "dependency_hashes": {}, "versions": {},
            "counts": counts, "failures": [{"node": f, "kind": "failure", "message": ""} for f in fails],
            "shard": {"index": i, "count": n, "n_files": 1, "files": [f"tests/test_{i}.py"]},
            "results_dir": f"/vol/cpu-suite/{sha}/shards/{i}-of-{n}"}


def _inodes(root: Path) -> tuple[int, int]:
    """(files, directories) under root, root itself counted — the quantity the volume's quota meters."""
    files = sum(1 for q in root.rglob("*") if q.is_file())
    dirs = 1 + sum(1 for q in root.rglob("*") if q.is_dir())
    return files, dirs


def test_finalize_removes_every_passed_shards_directory_and_keeps_only_the_merged_summary(tmp_path, cs):
    repos, results = tmp_path / "repos", tmp_path / "cpu-suite"
    repos.mkdir(); results.mkdir()
    sha, n = _sha(10), 10
    for i in range(n):
        _shard_dir(results, sha, i, n)
    other = _sha(11)                                   # a neighbour gate's results: never this finalizer's to touch
    _shard_dir(results, other, 0, 2); _shard_dir(results, other, 1, 2)
    (results / other / "summary.json").write_text("{}")
    assert _inodes(results / sha) == (30, 12)          # ten shards x three files, in shards/ + ten dirs + the sha dir
    calls = {f"fc-{i}": _shard_call(sha, i, n) for i in range(n)}
    commits = []
    vol = types.SimpleNamespace(reload=lambda: commits.append("reload"), commit=lambda: commits.append("commit"))
    out = cs._finalize_shards_in(sha, f"cpu-suite-{sha[:7]}", list(calls), False, get_call=lambda cid: calls[cid],
                                 repos_dir=repos, results_root=results, vol=vol)
    assert out["status"] == "PASS" and out["pytest_rc"] == 0 and out["shards"] == n
    # the merged record is on disk, is the merge and nothing else, and is the ONLY thing left: 1 file in 1 directory
    written = json.loads((results / sha / "summary.json").read_text())
    assert written["status"] == "PASS" and written["counts"]["tests"] == sum(100 + i for i in range(n))
    assert "shard_artifacts_pruned" not in written and "finalized_by" not in written
    assert sorted(q.name for q in (results / sha).iterdir()) == ["summary.json"]
    assert _inodes(results / sha) == (1, 1)
    assert out["shard_artifacts_pruned"] == {"removed": [f"{i}-of-{n}" for i in range(n)], "kept": {}, "missing": [],
                                             "shards_dir_removed": True}
    # the summary was committed BEFORE the directories went, and the prune was committed after
    assert commits == ["reload", "commit", "commit"]
    # the neighbour sha is untouched, files and all
    assert _inodes(results / other) == (7, 4) and (results / other / "shards" / "1-of-2" / "pytest.log").is_file()


def test_finalize_keeps_exactly_the_failing_shards_directory_with_its_log_and_junit(tmp_path, cs):
    repos, results = tmp_path / "repos", tmp_path / "cpu-suite"
    repos.mkdir(); results.mkdir()
    sha, n, bad = _sha(12), 10, 3
    dirs = {i: _shard_dir(results, sha, i, n) for i in range(n)}
    before = {q.name: q.read_text() for q in dirs[bad].iterdir()}
    calls = {f"fc-{i}": _shard_call(sha, i, n, "FAIL" if i == bad else "PASS",
                                    fails=(f"tests/test_{bad}.py::t_red",) if i == bad else ())
             for i in range(n)}
    commits = []
    vol = types.SimpleNamespace(reload=lambda: commits.append("reload"), commit=lambda: commits.append("commit"))
    out = cs._finalize_shards_in(sha, f"cpu-suite-{sha[:7]}", list(calls), False, get_call=lambda cid: calls[cid],
                                 repos_dir=repos, results_root=results, vol=vol)
    assert out["status"] == "FAIL" and [f["node"] for f in out["failures"]] == [f"tests/test_{bad}.py::t_red"]
    written = json.loads((results / sha / "summary.json").read_text())
    assert written["status"] == "FAIL" and written["shard_records"][bad]["status"] == "FAIL"
    # the failing shard's directory is intact, byte for byte; every passing shard's is gone; shards/ stays (not empty)
    assert dirs[bad].is_dir() and {q.name: q.read_text() for q in dirs[bad].iterdir()} == before
    assert set(before) == set(_SHARD_FILES)
    assert [q.name for q in (results / sha / "shards").iterdir()] == [f"{bad}-of-{n}"]
    assert all(not dirs[i].exists() for i in range(n) if i != bad)
    assert _inodes(results / sha) == (4, 3)             # summary + the red shard's three files; sha dir, shards/, its dir
    assert out["shard_artifacts_pruned"] == {"removed": [f"{i}-of-{n}" for i in range(n) if i != bad],
                                             "kept": {f"{bad}-of-{n}": "FAIL"}, "missing": [], "shards_dir_removed": False}
    assert commits == ["reload", "commit", "commit"]


def test_finalize_keeps_a_shard_error_and_a_timeout_shards_directory_the_same_way(tmp_path, cs):
    repos, results = tmp_path / "repos", tmp_path / "cpu-suite"
    repos.mkdir(); results.mkdir()
    sha, n = _sha(13), 6
    crashed, slow = 4, 1
    dirs = {i: _shard_dir(results, sha, i, n, files=("pytest.log",) if i == crashed else
                          ("pytest.log", "summary.json") if i == slow else _SHARD_FILES) for i in range(n)}
    calls = {f"fc-{i}": _shard_call(sha, i, n, "TIMEOUT" if i == slow else "PASS") for i in range(n)}
    def get_call(cid):
        if cid == f"fc-{crashed}":
            raise RuntimeError("Function call timed out")   # the CALL raised: the container crashed after writing its log
        return calls[cid]
    out = cs._finalize_shards_in(sha, f"cpu-suite-{sha[:7]}", list(calls), True, get_call=get_call,
                                 repos_dir=repos, results_root=results, vol=None)
    assert out["status"] == cs.SHARD_ERROR_STATUS and out["shard_records"][crashed]["status"] == "SHARD_ERROR"
    assert out["shard_records"][slow]["status"] == "TIMEOUT"
    assert dirs[crashed].is_dir() and (dirs[crashed] / "pytest.log").read_text().startswith("pytest.log of shard 4")
    assert dirs[slow].is_dir() and sorted(q.name for q in dirs[slow].iterdir()) == ["pytest.log", "summary.json"]
    assert sorted(q.name for q in (results / sha / "shards").iterdir()) == [f"{slow}-of-{n}", f"{crashed}-of-{n}"]
    assert (results / sha / "summary.json").is_file()
    assert out["shard_artifacts_pruned"] == {"removed": [f"{i}-of-{n}" for i in range(n) if i not in (crashed, slow)],
                                             "kept": {f"{slow}-of-{n}": "TIMEOUT", f"{crashed}-of-{n}": "SHARD_ERROR"},
                                             "missing": [], "shards_dir_removed": False}


def test_finalize_tolerates_passed_shard_directories_that_are_already_gone_and_names_nothing_else(tmp_path, cs):
    repos, results = tmp_path / "repos", tmp_path / "cpu-suite"
    repos.mkdir(); results.mkdir()
    # (a) one passing shard's directory is already gone: not an error, named as missing, the rest still go
    sha, n = _sha(14), 4
    for i in (0, 1, 3):
        _shard_dir(results, sha, i, n)
    stray = results / sha / "shards" / "0-of-5"      # another layout's leftover at the same sha: not this record's
    stray.mkdir(); (stray / "pytest.log").write_text("old")
    calls = {f"fc-{i}": _shard_call(sha, i, n) for i in range(n)}
    commits = []
    vol = types.SimpleNamespace(reload=lambda: commits.append("reload"), commit=lambda: commits.append("commit"))
    out = cs._finalize_shards_in(sha, f"cpu-suite-{sha[:7]}", list(calls), False, get_call=lambda cid: calls[cid],
                                 repos_dir=repos, results_root=results, vol=vol)
    assert out["status"] == "PASS" and (results / sha / "summary.json").is_file()
    assert out["shard_artifacts_pruned"] == {"removed": ["0-of-4", "1-of-4", "3-of-4"], "kept": {}, "missing": ["2-of-4"],
                                             "shards_dir_removed": False}
    assert [q.name for q in (results / sha / "shards").iterdir()] == ["0-of-5"] and (stray / "pytest.log").read_text() == "old"
    assert commits == ["reload", "commit", "commit"]
    # (b) no shards/ directory at all (every shard's dir gone, or never written): nothing raised, nothing committed twice
    sha2 = _sha(15)
    calls2 = {f"fc-{i}": _shard_call(sha2, i, n) for i in range(n)}
    commits2 = []
    vol2 = types.SimpleNamespace(reload=lambda: commits2.append("reload"), commit=lambda: commits2.append("commit"))
    out2 = cs._finalize_shards_in(sha2, f"cpu-suite-{sha2[:7]}", list(calls2), False, get_call=lambda cid: calls2[cid],
                                  repos_dir=repos, results_root=results, vol=vol2)
    assert out2["status"] == "PASS" and sorted(q.name for q in (results / sha2).iterdir()) == ["summary.json"]
    assert out2["shard_artifacts_pruned"] == {"removed": [], "kept": {}, "missing": [f"{i}-of-{n}" for i in range(n)],
                                              "shards_dir_removed": False}
    assert commits2 == ["reload", "commit"]              # nothing removed, so no second commit
    # (c) the helper on an unsharded record (count 1: the shard wrote beside summary.json) touches nothing
    top = results / sha2 / "pytest.log"; top.write_text("full-suite log")
    rec1 = {"shard_records": [{"shard": {"index": 0, "count": 1}, "status": "PASS"}]}
    assert cs._prune_passed_shard_artifacts(results, sha2, rec1) == {"removed": [], "kept": {}, "missing": [],
                                                                     "shards_dir_removed": False}
    assert top.read_text() == "full-suite log"

