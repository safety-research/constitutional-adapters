"""closer-fix-4: the funnel's closer-tree guard reads the ACTUAL commit at the tree (`git rev-parse HEAD`) and
requires ancestry from closer-fix-3 (CLOSER_MIN_SHA = b36e48a2f) — the marker greps alone (`def gate_binding(`, the CLOSER_SHA positional)
also match the real 26a27aac3 tree, the closer with all three defects closer-fix-3 repairs, and a stub carrying the markers.

* the minimum is closer-fix-3 itself, and the real 26a27aac3 commit — an ancestor of it, not a descendant — is refused by name, plain and
  with --tree, naming the commit the tree is AT (never the directory's name) and the corrected hint;
* a tree AT closer-fix-3 (or after: this repo's HEAD) passes and the reclose is submitted from it;
* a tree that is not a git checkout is refused by name (no HEAD to read); the marker checks still gate a descendant tree;
* a clone that does not carry closer-fix-3 at all is refused by name, whatever files it carries — the reclose is an internal recovery path
  (the original job's private pending recipe, spool, driver environment and registry push) and needs the development history and the
  private registry pipeline: a repository published without its development history (the publish step's root marker, borne out: not
  shallow, `.history-squashed` reading `squashed`, the one root commit carrying it), with or without commits on top, is named as
  that; a clone without the commit and without a borne-out marker (an own root with no marker, a stray marker the root does not carry) is
  named a shallow or partial clone; nothing submitted, nothing on the ledger, in every case;
* a clone whose object store carries closer-fix-3 but whose HEAD does not descend from it (an unrelated history fetched in) is refused on
  the lineage, main's sentence: presence alone never stands in for the ancestry.

The tree fixture (test_closer_refusal_outcome._tree_with_closer) is a checkout-less git repository sharing this repo's objects, so the
commits are the real ones; the real-commit cases skip only where the absence of those commits is positively established (a shallow clone,
the published repository by its borne-out root marker -- seedrep_history_fixtures.pinned_commit_or_skip) and fail naming the branch on a
full checkout that lacks them (the no-history cases build their own repositories and run everywhere)."""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import LIB, REPO  # noqa: E402
from seedrep_history_fixtures import pinned_commit_or_skip  # noqa: E402
from test_closer_refusal_outcome import _argv, _need_closer_history, _tree_with_closer, _world  # noqa: E402
from test_reclose_at_tree import SHA2, _closer_tree  # noqa: E402
from test_reclose_submits_from_the_closer_tree import _cwd, _job_tree  # noqa: E402
from test_submit_twin import SHA  # noqa: E402  # the job's wave sha: a DERIVED 40-hex, never forty identical digits

SRC = (LIB / "submit_twin.py").read_text()
MIN = re.search(r'CLOSER_MIN_SHA = "([0-9a-f]{40})"', SRC).group(1)
FIX2 = re.search(r'CLOSER_FIX2_SHA = "([0-9a-f]{40})"', SRC).group(1)
FIX3 = "b36e48a2ffb2e9878b648d1ebc64f49b4ea831b3"
MARKER = ".history-squashed"
INTERNAL = "the reclose is an internal recovery path; it needs the development history and the private registry pipeline"
PUBLISHED = "a repository published without its development history (the publish step's root marker, borne out)"
PARTIAL = "a shallow or partial clone, not a publication"
GIT_ENV = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid",
           "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z", "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z", "PATH": "/usr/bin:/bin"}


def _have(sha: str) -> bool:
    return subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", f"{sha}^{{commit}}"], capture_output=True).returncode == 0


def _need_real_commits():
    """The ancestry cases read the real closer-fix-2 / closer-fix-3 commits out of this repo's object store (through the fixture's alternates):
    a skip only on the positively established absence of history (a shallow clone; the published repository, its root marker borne out), a
    failure naming the branch on a full checkout that lacks them, and a marker the repository does not bear out fails by name -- the
    no-history cases run everywhere."""
    pinned_commit_or_skip(FIX3, "main", "the closer-tree guard's ancestry cases (closer-fix-3's lineage)", repo=REPO, anchor=Path(__file__))
    pinned_commit_or_skip(FIX2, "main", "the closer-tree guard's ancestry cases (the real closer-fix-2 tree)", repo=REPO, anchor=Path(__file__))


def _no_sbatch(w) -> bool:
    return not (w.fake["cluster_a"] / "jobs" / "1002.argv").exists()


def _git(co: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(co), *args], capture_output=True, text=True, check=True, env=GIT_ENV).stdout.strip()


def _own_root(co: Path, *, published: bool = False, user_commit: bool = False, stray_marker: bool = False, marker_content: str = "squashed\n") -> str:
    """A git repository at `co` with ITS OWN root commit and no alternates — none of this repo's commits exist in it, so `cat-file -e
    <closer-fix-3>` fails and `merge-base --is-ancestor` cannot be asked. `published`: the publish step's root marker (`marker_content`) is
    COMMITTED in the root commit (the publication's shape); `user_commit`: a commit on top of it (configs the reproduction asks be
    committed); `stray_marker`: the marker file written but not committed by the root (a development checkout's shape with a stray file).
    Returns HEAD."""
    co.mkdir(parents=True, exist_ok=True)
    _git(co, "init", "-q")
    if published:
        (co / MARKER).write_text(marker_content)
        _git(co, "add", MARKER)
    _git(co, "commit", "-q", "--allow-empty", "-m", "squashed")
    if user_commit:
        (co / "configs").mkdir(exist_ok=True)
        (co / "configs" / "generated.yaml").write_text("x: 1\n")
        _git(co, "add", "configs/generated.yaml")
        _git(co, "commit", "-q", "-m", "the generated configs, committed")
    if stray_marker:
        (co / MARKER).write_text("squashed\n")
    return _git(co, "rev-parse", "HEAD")


def _refused_without_history(w, co: Path, own: str, err: str, *, shape: str, who: str = f"--tree {SHA2[:12]}") -> None:
    """The no-history refusal's shape: the commit the tree is AT, the missing commit, the clone's shape named, the internal-recovery sentence;
    never the ancestry sentence, never a hint to land a tree; no sbatch, nothing on the ledger."""
    assert _no_sbatch(w) and not [e for e in w.ledger() if e["event"] == "reclose"], err
    assert (f"{who}: the closer at {co} is commit {own[:12]}; this clone does not carry closer-fix-3 {FIX3[:12]} ({shape}) — {INTERNAL} "
            "(the original job's pending recipe, spool, driver environment and registry push), and this clone has neither; nothing submitted") in err, err
    assert "which does not descend from" not in err and "land and mirror a tree that carries it" not in err and "lacks closer-fix-2" not in err, err


def test_a_repository_published_without_its_development_history_is_refused_by_name_with_or_without_a_users_commit_on_top(tmp_path):
    # the publication's shape: the real closer, the plumbed reclose script, src/sisteer, the root marker committed by the one root commit
    w = _world(tmp_path); co = w.shared["cluster_a"] / "sisteer-eval-repo" / SHA2; own = _own_root(co, published=True); _tree_with_closer(co, head=own)
    assert not _have(own) and _git(co, "rev-parse", "--is-shallow-repository") == "false" and _git(co, "cat-file", "-p", f"{own}:{MARKER}") == "squashed"
    r = w.run("reclose", "--job", "1001", "--tree", SHA2); err = r.stdout + r.stderr
    assert r.returncode != 0, err
    _refused_without_history(w, co, own, err, shape=PUBLISHED)
    # a commit on top of the publication: HEAD is a descendant of the publication root, still the publication, the same refusal
    w2 = _world(tmp_path / "w2"); co2 = w2.shared["cluster_a"] / "sisteer-eval-repo" / SHA2
    own2 = _own_root(co2, published=True, user_commit=True); _tree_with_closer(co2, head=own2)
    assert _git(co2, "rev-list", "--count", "HEAD") == "2" and _git(co2, "rev-list", "--max-parents=0", "HEAD") != own2
    r = w2.run("reclose", "--job", "1001", "--tree", SHA2); err = r.stdout + r.stderr
    assert r.returncode != 0, err
    _refused_without_history(w2, co2, own2, err, shape=PUBLISHED)
    # plain (the job's own tree is the publication): the same refusal, naming the job's tree
    w3 = _world(tmp_path / "w3"); job = _job_tree(w3); shutil.rmtree(job); own3 = _own_root(job, published=True); _tree_with_closer(job, head=own3)
    r = w3.run("reclose", "--job", "1001"); err = r.stdout + r.stderr
    assert r.returncode != 0, err
    _refused_without_history(w3, job, own3, err, shape=PUBLISHED, who=f"the job's tree {SHA[:12]}")
    # the marker's content with whitespace AROUND it is the marker (surrounding whitespace is stripped, as every other reader strips it)
    w4 = _world(tmp_path / "w4"); co4 = w4.shared["cluster_a"] / "sisteer-eval-repo" / SHA2
    own4 = _own_root(co4, published=True, marker_content="  squashed \t\n\n"); _tree_with_closer(co4, head=own4)
    r = w4.run("reclose", "--job", "1001", "--tree", SHA2); err = r.stdout + r.stderr
    assert r.returncode != 0, err
    _refused_without_history(w4, co4, own4, err, shape=PUBLISHED)


def test_a_clone_without_closer_fix_3_that_is_not_a_publication_is_refused_by_name_too(tmp_path):
    """The rule for a no-history clone that is NOT a publication: the same refusal, the shape named — an own root without the marker, an own
    root with a stray marker the root commit does not carry (a development checkout's shape), a marker with other content. Pre-branch the
    funnel refused these on the lineage with main's "does not descend" sentence; now the refusal says what it found: no closer-fix-3, no
    publication, no reclose."""
    w = _world(tmp_path); co = w.shared["cluster_a"] / "sisteer-eval-repo" / SHA2; own = _own_root(co); _tree_with_closer(co, head=own)
    r = w.run("reclose", "--job", "1001", "--tree", SHA2); err = r.stdout + r.stderr
    assert r.returncode != 0, err
    _refused_without_history(w, co, own, err, shape=PARTIAL)
    w2 = _world(tmp_path / "w2"); co2 = w2.shared["cluster_a"] / "sisteer-eval-repo" / SHA2; own2 = _own_root(co2, stray_marker=True); _tree_with_closer(co2, head=own2)
    assert (co2 / MARKER).is_file()
    assert subprocess.run(["git", "-C", str(co2), "cat-file", "-e", f"{own2}:{MARKER}"], capture_output=True).returncode != 0   # the root does not carry it
    r = w2.run("reclose", "--job", "1001", "--tree", SHA2); err = r.stdout + r.stderr
    assert r.returncode != 0, err
    _refused_without_history(w2, co2, own2, err, shape=PARTIAL)
    # the marker with other content, committed by the root: not the publish step's marker
    w3 = _world(tmp_path / "w3"); co3 = w3.shared["cluster_a"] / "sisteer-eval-repo" / SHA2; co3.mkdir(parents=True)
    _git(co3, "init", "-q"); (co3 / MARKER).write_text("nope\n"); _git(co3, "add", MARKER); _git(co3, "commit", "-q", "-m", "c0"); own3 = _git(co3, "rev-parse", "HEAD")
    _tree_with_closer(co3, head=own3)
    r = w3.run("reclose", "--job", "1001", "--tree", SHA2); err = r.stdout + r.stderr
    assert r.returncode != 0, err
    _refused_without_history(w3, co3, own3, err, shape=PARTIAL)
    # whitespace INSIDE the marker's content is not the marker: only surrounding whitespace is stripped (the rule every other reader applies —
    # a reader that removed all whitespace read `squa shed` as the publication's statement)
    w4 = _world(tmp_path / "w4"); co4 = w4.shared["cluster_a"] / "sisteer-eval-repo" / SHA2
    own4 = _own_root(co4, published=True, marker_content="squa shed\n"); _tree_with_closer(co4, head=own4)
    r = w4.run("reclose", "--job", "1001", "--tree", SHA2); err = r.stdout + r.stderr
    assert r.returncode != 0, err
    _refused_without_history(w4, co4, own4, err, shape=PARTIAL)
    assert 'tr -d "[:space:]"' not in SRC and "trim(){" in SRC


def test_a_present_but_unrelated_closer_fix_3_is_refused_on_the_lineage_never_admitted_by_presence(tmp_path):
    """An own-root repository given this repo's objects (an alternates file: closer-fix-3 is in its object store) whose HEAD does not descend
    from it — the shape of an unrelated history fetched into a clone. `cat-file -e` succeeds; the funnel reads the lineage for real and refuses
    with main's sentence, never the no-history one and never an admission."""
    _need_real_commits()
    w = _world(tmp_path); co = w.shared["cluster_a"] / "sisteer-eval-repo" / SHA2; own = _own_root(co, published=True)
    common = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--git-common-dir"], capture_output=True, text=True, check=True).stdout.strip()
    common_dir = Path(common) if Path(common).is_absolute() else (REPO / common).resolve()
    alt = co / ".git" / "objects" / "info"; alt.mkdir(parents=True, exist_ok=True); (alt / "alternates").write_text(f"{common_dir / 'objects'}\n")
    _tree_with_closer(co, head=own)
    assert subprocess.run(["git", "-C", str(co), "cat-file", "-e", f"{FIX3}^{{commit}}"], capture_output=True).returncode == 0      # present...
    assert subprocess.run(["git", "-C", str(co), "merge-base", "--is-ancestor", FIX3, "HEAD"], capture_output=True).returncode == 1  # ...not an ancestor
    r = w.run("reclose", "--job", "1001", "--tree", SHA2); err = r.stdout + r.stderr
    assert r.returncode != 0 and _no_sbatch(w), err
    assert f"--tree {SHA2[:12]}: the closer at {co} is commit {own[:12]}, which does not descend from closer-fix-3 {FIX3[:12]}" in err, err
    assert INTERNAL not in err and "does not carry closer-fix-3" not in err, err


def test_the_minimum_is_closer_fix_3_and_the_real_fix_2_tree_is_refused_by_the_commit_it_is_at():
    assert MIN == FIX3 and FIX2 == "26a27aac30fc50398604aa450970f91d8373e85d"
    _need_real_commits()
    # the real history: fix-2 is an ancestor of fix-3, never the reverse — the guard's rule is exactly `merge-base --is-ancestor MIN HEAD`
    assert subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", FIX2, FIX3]).returncode == 0
    assert subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", FIX3, FIX2]).returncode == 1


def test_a_plain_reclose_from_a_job_tree_at_the_real_closer_fix_2_commit_is_refused_by_name(tmp_path):
    _need_real_commits()
    w = _world(tmp_path); _tree_with_closer(_job_tree(w), head=FIX2)          # the job's tree carries BOTH markers (the stubs) and IS 26a27aac3
    r = w.run("reclose", "--job", "1001"); err = r.stdout + r.stderr
    assert r.returncode != 0 and _no_sbatch(w), err
    assert f"the job's tree {SHA[:12]}: the closer at" in err and f"is commit {FIX2[:12]}, which does not descend from closer-fix-3 {FIX3[:12]}" in err, err
    assert "shard-local response tally" in err and f"pass --tree <a landed tree at or past {FIX3[:9]}>" in err and FIX2[:9] + " carries the markers" in err, err
    assert "lacks closer-fix-2" not in err                                      # refused on the COMMIT, before the marker checks


def test_tree_at_the_real_closer_fix_2_commit_is_refused_and_at_closer_fix_3_or_after_submits_from_it(tmp_path):
    _need_real_commits()
    w = _world(tmp_path); co2 = _closer_tree(w, SHA2); _tree_with_closer(co2, head=FIX2)
    r = w.run("reclose", "--job", "1001", "--tree", SHA2); err = r.stdout + r.stderr
    assert r.returncode != 0 and _no_sbatch(w), err
    assert f"--tree {SHA2[:12]}: the closer at {co2} is commit {FIX2[:12]}, which does not descend from closer-fix-3 {FIX3[:12]}" in err, err
    assert "land and mirror a tree that carries it" in err, err
    # AT closer-fix-3 exactly: the minimum passes (is-ancestor of itself)
    w2 = _world(tmp_path / "w2"); co2 = _closer_tree(w2, SHA2); _tree_with_closer(co2, head=FIX3)
    r = w2.run("reclose", "--job", "1001", "--tree", SHA2); out = r.stdout + r.stderr
    assert r.returncode == 0 and "RECLOSE cluster_a 1002 mmlu" in r.stdout and _cwd(w2, "1002") == co2.resolve(), out
    assert f"scripts/sdfbase/full22/reclose_stores.sbatch 1001 {SHA} {SHA2}" in _argv(w2, "1002")
    # after it (this repo's HEAD, the fixture's default): passes, plain and explicit — the directory's NAME (b…b, the wave sha) is never what is read
    w3 = _world(tmp_path / "w3"); co3 = _closer_tree(w3, SHA2)
    r = w3.run("reclose", "--job", "1001", "--tree", SHA2); assert r.returncode == 0 and _cwd(w3, "1002") == co3.resolve(), r.stdout + r.stderr
    w4 = _world(tmp_path / "w4")
    r = w4.run("reclose", "--job", "1001"); assert r.returncode == 0 and _cwd(w4, "1002") == _job_tree(w4).resolve(), r.stdout + r.stderr
    assert INTERNAL not in r.stdout + r.stderr


def test_a_tree_that_is_not_a_git_checkout_is_refused_by_name(tmp_path):
    w = _world(tmp_path); co2 = _closer_tree(w, SHA2); shutil.rmtree(co2 / ".git")      # the markers are there; no commit to read
    r = w.run("reclose", "--job", "1001", "--tree", SHA2); err = r.stdout + r.stderr
    assert r.returncode != 0 and _no_sbatch(w) and f"--tree {SHA2[:12]}: {co2} on cluster_a is not a git checkout (no HEAD to read)" in err, err
    assert "never the directory's name (D1529)" in err and "does not descend" not in err, err


def test_the_marker_checks_still_gate_a_descendant(tmp_path):
    _need_closer_history()                           # the marker checks come AFTER the ancestry check: reachable only where the tree descends from closer-fix-3
    # a descendant without closer-fix-2's marker / without the positional: the marker refusals stand, and the fix-2 one names the corrected minimum
    w2 = _world(tmp_path / "w2"); _tree_with_closer(_closer_tree(w2, SHA2), fix2=False)
    r = w2.run("reclose", "--job", "1001", "--tree", SHA2); err = r.stdout + r.stderr
    assert r.returncode != 0 and "lacks closer-fix-2" in err and f"{FIX3[:9]} is the minimum tree" in err and FIX2[:9] + " is the first tree" not in err, err
    w3 = _world(tmp_path / "w3"); _tree_with_closer(_closer_tree(w3, SHA2), plumbed=False)
    r = w3.run("reclose", "--job", "1001", "--tree", SHA2); err = r.stdout + r.stderr
    assert r.returncode != 0 and "predates reclose-at-tree (no CLOSER_SHA positional 3)" in err, err


def test_the_guard_asks_git_for_the_head_and_the_ancestry_in_its_one_remote_read():
    body = SRC[SRC.index("def closer_tree_check("):]
    body = body[:body.index("def reclose(")]
    assert 'git -C "$t" rev-parse --verify HEAD' in body and 'git -C "$t" merge-base --is-ancestor "$m" HEAD' in body
    assert "self.CLOSER_MIN_SHA" in body and body.count("self.remotes[label].sh(") == 1          # still ONE remote read
    assert "at or past {self.CLOSER_FIX2_SHA[:9]}" not in body                                   # the old hint named the defective minimum
    # the same read says whether the clone carries the commit at all and whether it is a borne-out publication (not shallow, the root marker
    # reading `squashed`, HEAD's one root commit carrying it) — for the refusal's WORDING only: presence never admits, content never admits
    assert 'git -C "$t" cat-file -e "$m^{{commit}}"' in body and "echo min=present || echo min=absent" in body
    assert "rev-parse --is-shallow-repository" in body and "rev-list --max-parents=0 HEAD" in body and 'cat-file -p "$1:{mk}"' in body and "echo published=$pub" in body
    assert 'got.get("min") == "absent"' in body and 'got.get("published") == "yes"' in body and INTERNAL in body
    assert "hash-object" not in body and "dep:" not in body and "AUDITED" not in body             # no content path
    assert body.index('got.get("min") == "absent"') < body.index("which does not descend from") < body.index('got.get("fix2") != "ok"')
    assert 'PUBLISHED_MARKER = ".history-squashed"' in SRC
