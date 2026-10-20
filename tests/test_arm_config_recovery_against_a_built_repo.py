"""Recovery is exercised against a git repo the TEST builds — so it runs everywhere, shallow container included.

The companion file `test_arm_config_recovered_by_content_sha.py` pins recovery against THIS repo's real history,
which the Modal cpu-suite cannot provide: `scripts/modal/cpu_suite.py` stages an exact-sha `--depth 1` clone, so
every query returns None there. Those tests therefore skip in the gate — and skipping means the container exercises
none of the recovery logic, neither the history walk nor the sha comparison that decides what is accepted.

This file closes that hole. It builds a throwaway git repo with a config committed at the same repo-relative path,
points the helper at it (the helper derives its repo from `__file__`, which the AST-exec namespace controls), and
runs the REAL function end to end — walk, hash, compare. No dependency on this checkout's history, so it is not
skipped anywhere, and a shallow gate still proves that bytes matching the pin are returned and bytes that do not are
never returned.

An alternative — vendoring the bytes as a sha-pinned fixture — would keep the hash-matching half live in the gate
too. Building the history rather than vendoring the bytes goes one better: the git walk is covered too, and there
is no vendored blob to drift from the thing it stands in for.
"""

import ast
import hashlib
import re
import subprocess
from pathlib import Path
from typing import Optional

import pytest

ROOT = Path(__file__).parents[1]
BUILDER = ROOT / "scripts" / "r2_export" / "cap4096" / "build_transfer_stats_cap4096.py"
REL = "configs/sdfbase_full22_fixture__arditi.yaml"
OLD = b"defenses_dir: /shared/exp/objects/x/qwen35_4b/ladder/cells_view/ar_c0p2\n"
NEW = b"defenses_dir: /shared/exp/objects/x/qwen35_4b/ladder/cells_view/ar_c0p3\n"


def _git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True)


@pytest.fixture(scope="module")
def built_repo(tmp_path_factory):
    """A real repo whose history holds TWO revisions of one config path — the old one is the interesting case."""
    repo = tmp_path_factory.mktemp("fixture_repo")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.invalid")
    _git(repo, "config", "user.name", "t")
    p = repo / REL
    p.parent.mkdir(parents=True, exist_ok=True)
    for body, msg in ((OLD, "old"), (NEW, "new")):
        p.write_bytes(body)
        _git(repo, "add", REL)
        _git(repo, "commit", "-q", "-m", msg)
    # the builder computes its repo root as Path(__file__).parents[3]
    fake = repo / "scripts" / "r2_export" / "cap4096" / "build_transfer_stats_cap4096.py"
    fake.parent.mkdir(parents=True, exist_ok=True)
    fake.write_text("")
    return repo, str(fake)


@pytest.fixture(scope="module")
def recover(built_repo):
    _, fake_file = built_repo
    for node in ast.parse(BUILDER.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name == "_config_bytes_from_git":
            ns = {"re": re, "subprocess": subprocess, "hashlib": hashlib, "Path": Path,
                  "Optional": Optional, "__file__": fake_file}
            exec(compile(ast.Module([node], []), str(BUILDER), "exec"), ns)  # noqa: S102
            return ns["_config_bytes_from_git"]
    raise AssertionError("_config_bytes_from_git not found in the builder")


def _recorded(repo, body):
    """A gone path plus the sha a grader would have recorded for those bytes."""
    return f"{repo}/checkouts/arms-gone-ro/{REL}", hashlib.sha256(body).hexdigest()


def test_the_SUPERSEDED_revision_is_recovered_not_just_the_current_one(recover, built_repo):
    """The whole point: a graded leg's config is usually NOT the current file."""
    repo, _ = built_repo
    path, sha = _recorded(repo, OLD)
    got = recover(path, sha)
    assert got == OLD, "recovery returned the wrong revision — the superseded one is the one a graded root needs"


def test_the_current_revision_is_also_recoverable(recover, built_repo):
    repo, _ = built_repo
    path, sha = _recorded(repo, NEW)
    assert recover(path, sha) == NEW


def test_bytes_that_do_not_match_the_pin_are_NEVER_returned(recover, built_repo):
    """The known-bad that the shallow gate could not exercise: a sha no revision has must yield nothing,
    and in particular must NOT fall back to the newest revision of that path."""
    repo, _ = built_repo
    path, _sha = _recorded(repo, OLD)
    assert recover(path, "0" * 64) is None
    assert recover(path, hashlib.sha256(b"never committed\n").hexdigest()) is None


def test_a_one_character_sha_difference_is_refused(recover, built_repo):
    repo, _ = built_repo
    path, sha = _recorded(repo, OLD)
    assert recover(path, sha[:-1] + ("0" if sha[-1] != "0" else "1")) is None


def test_a_path_outside_configs_recovers_nothing(recover, built_repo):
    repo, _ = built_repo
    assert recover(f"{repo}/checkouts/x/scripts/thing.yaml", hashlib.sha256(OLD).hexdigest()) is None


def test_an_empty_pin_recovers_nothing(recover, built_repo):
    repo, _ = built_repo
    path, _ = _recorded(repo, OLD)
    assert recover(path, "") is None


# ----------------------------------------------------------------------------------------------- the third shape: no history at all
def _builder_fns(fake_file: str, *names):
    """Exec the named top-level functions out of the builder into ONE namespace rooted at `fake_file` (the builder derives its repo from
    `__file__`), with the globals they use and an empty ARM_OPERATING_POINT (no declared point: nothing to contradict)."""
    ns = {"re": re, "subprocess": subprocess, "hashlib": hashlib, "Path": Path, "Optional": Optional, "__file__": fake_file, "ARM_OPERATING_POINT": {}}
    wanted = set(names)
    for node in ast.parse(BUILDER.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            exec(compile(ast.Module([node], []), str(BUILDER), "exec"), ns)  # noqa: S102
            wanted.discard(node.name)
    assert not wanted, f"not found in the builder: {sorted(wanted)}"
    return ns


SERVED_FNS = ("_config_rel", "_repo_history_is_searchable", "_published_without_history", "_history_has_no_earlier_revision", "_config_bytes_from_git",
              "served_operating_point_of")
MARKER = ".history-squashed"


def _published_repo(repo: Path, *, marker: bool = True):
    """A published repository's shape: the config path present at HEAD with bytes that are NOT the recorded ones, and (by default) the
    publish step's root marker `.history-squashed` (content `squashed`) COMMITTED in the one root commit — the publication root."""
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.invalid")
    _git(repo, "config", "user.name", "t")
    p = repo / REL
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(NEW)
    _git(repo, "add", REL)
    if marker:
        (repo / MARKER).write_text("squashed\n")
        _git(repo, "add", MARKER)
    _git(repo, "commit", "-q", "-m", "squashed")
    assert _git(repo, "rev-list", "--count", "HEAD").stdout.strip() == "1"
    fake = repo / "scripts" / "r2_export" / "cap4096" / "build_transfer_stats_cap4096.py"
    fake.parent.mkdir(parents=True, exist_ok=True)
    fake.write_text("")
    return repo, str(fake)


@pytest.fixture(scope="module")
def one_commit_repo(tmp_path_factory):
    return _published_repo(tmp_path_factory.mktemp("one_commit_repo"))


def test_a_one_commit_clone_records_that_its_history_holds_no_earlier_revision_and_what_the_search_found(one_commit_repo):
    """The reason names the CLONE's shape (a published repository without its development history), neither the data ("no revision
    matches") nor a shallow cut ("SHALLOW"): the three records are distinct, and this is the third. It also states what was DONE: the
    search ran over the revisions the clone holds (HEAD's own can carry the bytes) and found none — never "not attempted" after a search."""
    repo, fake = one_commit_repo
    ns = _builder_fns(fake, *SERVED_FNS)
    path, sha = _recorded(repo, OLD)                                                     # the superseded bytes, which this clone never held
    assert ns["_config_rel"](path) == REL and ns["_repo_history_is_searchable"]() is True   # a full clone, not shallow
    assert ns["_history_has_no_earlier_revision"](path) is True
    assert ns["_config_bytes_from_git"](path, sha) is None                               # and recovery indeed finds nothing
    out = ns["served_operating_point_of"]({"config": {"path": path, "sha256": sha}}, "arditi", "v")
    assert out["status"] == "unverifiable" and out["searchable_history"] is False, out
    assert out["reason"] == (f"this clone's history holds no earlier revision of {REL} (a published repository without its development history); "
                             "the search over the revisions it does hold found none with the recorded sha256"), out["reason"]
    assert "SHALLOW" not in out["reason"] and "no revision of it in git matches" not in out["reason"] and "not attempted" not in out["reason"]
    # a path this clone's history never touched reads the same way (zero revisions), on the two-commit repo too
    assert ns["_history_has_no_earlier_revision"](f"{repo}/checkouts/x/configs/never_committed__arditi.yaml") is True
    # ...and when HEAD's own revision of the path carries the recorded bytes, the one-commit clone RECOVERS them: the search is real
    out = ns["served_operating_point_of"]({"config": {"path": path, "sha256": hashlib.sha256(NEW).hexdigest()}}, "arditi", "v")
    assert out.get("status") != "unverifiable" and out["config_source"].startswith("recovered from git by content sha256"), out


def test_the_published_shape_is_the_committed_marker_never_a_commit_count(tmp_path):
    """The rule, each clause on its own: the marker committed by the publication root decides, and a user's commit on top of the publication
    (configs committed as the reproduction asks) changes nothing; a bare one-commit repository WITHOUT the marker is not the published shape
    (the question falls through to the content miss, as before the marker existed); a marker the root commit does not carry (a stray file
    in a development checkout) is not either; nor is a depth-one SHALLOW clone whose HEAD carries the marker file."""
    # (1) marker in the root + a later commit: still published
    repo, fake = _published_repo(tmp_path / "later")
    (repo / "configs" / "generated_after_publication.yaml").write_bytes(b"x: 1\n")
    _git(repo, "add", "configs/generated_after_publication.yaml")
    _git(repo, "commit", "-q", "-m", "committed configs after the publication")
    assert _git(repo, "rev-list", "--count", "HEAD").stdout.strip() == "2"
    ns = _builder_fns(fake, *SERVED_FNS)
    assert ns["_published_without_history"](repo) is True
    path, sha = _recorded(repo, OLD)
    assert ns["_history_has_no_earlier_revision"](path) is True
    out = ns["served_operating_point_of"]({"config": {"path": path, "sha256": sha}}, "arditi", "v")
    assert out["status"] == "unverifiable" and out["searchable_history"] is False and "a published repository without its development history" in out["reason"], out
    # (2) one commit, no marker: NOT the published shape — the content-miss record, searchable_history True
    bare, fake = _published_repo(tmp_path / "bare", marker=False)
    ns = _builder_fns(fake, *SERVED_FNS)
    assert ns["_published_without_history"](bare) is False
    path, _ = _recorded(bare, OLD)
    assert ns["_history_has_no_earlier_revision"](path) is False
    out = ns["served_operating_point_of"]({"config": {"path": path, "sha256": "0" * 64}}, "arditi", "v")
    assert out["status"] == "unverifiable" and out["searchable_history"] is True and "no revision of it in git matches the recorded sha256" in out["reason"], out
    # (2b) whitespace INSIDE the content is other content (only surrounding whitespace is stripped); around it, it is the marker
    (repo / MARKER).write_text("squa shed\n")
    assert ns["_published_without_history"](repo) is False
    (repo / MARKER).write_text("  squashed \t\n")
    assert ns["_published_without_history"](repo) is True
    (repo / MARKER).write_text("squashed\n")
    # (3) a stray marker the root commit does not carry (two commits, the marker added in the second, or never committed at all)
    stray, fake = _published_repo(tmp_path / "stray", marker=False)
    (stray / MARKER).write_text("squashed\n")
    ns = _builder_fns(fake, *SERVED_FNS)
    assert ns["_published_without_history"](stray) is False
    _git(stray, "add", MARKER)
    _git(stray, "commit", "-q", "-m", "a marker committed later")
    assert ns["_published_without_history"](stray) is False
    # (4) a depth-one shallow clone of (1): HEAD's tree carries the marker file and `rev-list --max-parents=0 HEAD` answers HEAD itself (its
    # parents were cut), so without the shallow clause the clone would read as published — it does not
    shallow = tmp_path / "shallow"
    subprocess.run(["git", "clone", "-q", "--depth", "1", f"file://{repo}", str(shallow)], check=True, capture_output=True)
    assert _git(shallow, "rev-parse", "--is-shallow-repository").stdout.strip() == "true" and (shallow / MARKER).is_file()
    fake_s = shallow / "scripts" / "r2_export" / "cap4096" / "build_transfer_stats_cap4096.py"
    fake_s.parent.mkdir(parents=True, exist_ok=True)
    fake_s.write_text("")
    ns = _builder_fns(str(fake_s), *SERVED_FNS)
    assert ns["_published_without_history"](shallow) is False and ns["_repo_history_is_searchable"]() is False
    # (5) other content in the marker is not the publish step's marker
    other, fake = _published_repo(tmp_path / "other", marker=False)
    (other / MARKER).write_text("something else\n")
    _git(other, "add", MARKER)
    _git(other, "commit", "-q", "--amend", "-m", "squashed with another marker")
    ns = _builder_fns(fake, *SERVED_FNS)
    assert ns["_published_without_history"](other) is False
    # the builder never counts commits
    assert "rev-list\", \"--count\"" not in BUILDER.read_text() and 'rev-list", "--count' not in BUILDER.read_text()


def test_a_clone_with_history_still_distinguishes_a_content_miss_from_the_no_history_shape(built_repo):
    """The known-bad twin: two revisions of the path, neither matching -- the record stays the content miss, searchable_history True."""
    repo, fake = built_repo
    ns = _builder_fns(fake, *SERVED_FNS)
    path, _ = _recorded(repo, OLD)
    assert ns["_history_has_no_earlier_revision"](path) is False
    out = ns["served_operating_point_of"]({"config": {"path": path, "sha256": "0" * 64}}, "arditi", "v")
    assert out["status"] == "unverifiable" and out["searchable_history"] is True and "no revision of it in git matches the recorded sha256" in out["reason"], out
    # and the recoverable case recovers, naming the source
    out = ns["served_operating_point_of"]({"config": {"path": path, "sha256": hashlib.sha256(OLD).hexdigest()}}, "arditi", "v")
    assert out.get("status") != "unverifiable" and out["config_source"].startswith("recovered from git by content sha256"), out
    assert out["served_id"] == "ar_c0p2__arditi", out
