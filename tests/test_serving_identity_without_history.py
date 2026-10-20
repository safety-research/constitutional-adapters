"""The Arditi and AlphaSteer serving-identity gates on a clone that does NOT carry W8 (a published repository without its development
history, a shallow clone): the production byte-identity is read by CONTENT -- the pinned git object ids of the serving tree and
petri_runner at W8 -- and every check that needs the history itself refuses BY NAME, naming the missing history, never git's
"bad revision".

* the two pins are W8's object ids, the same literals in both lanes (verified against W8 wherever this clone carries it);
* a one-commit repository carrying THIS tree's serving bytes (which differ from W8's): both production gates refuse "not W8's bytes";
* a one-commit repository whose bytes ARE the pins (the pins pointed at its own ids): Arditi production refuses "serving_impl_commit
  cannot be confirmed" -- the last-touching commit has no history to be read from -- and AlphaSteer production PASSES by content,
  its claim being byte-identity alone;
* the AlphaSteer ladder refuses "regenerates the diff against W8, which this clone does not carry" without evaluating the frozen-patch
  clauses; the Arditi native arm's serving_impl_commit derivation refuses a clone that does not carry W8 -- the pinned commit's ABSENCE,
  never a commit count, so a published-shape repository with a commit on top of its squash is refused the same way -- and derives the
  last-touching commit on a repository whose HEAD descends from W8;
* W8 PRESENT in the object store but not in HEAD's lineage (an unrelated history fetched in) is refused by name everywhere: presence alone
  is never the history;
* the files ON DISK are compared by their bytes as they are (`git hash-object --no-filters`) against HEAD's committed blobs: a clean filter
  that launders an injected change past `git diff --quiet HEAD` does not get it served, nor does a module dropped under the serving dir.

Everything runs the REAL shell functions, sourced from this tree's files, with git resolving in a repository the test builds."""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
AR_COMMON = REPO / "scripts" / "arditi" / "sbatch" / "ar_common.sh"
AS22_COMMON = REPO / "scripts" / "alphasteer_f22" / "sbatch" / "as22_common.sh"
W8 = "935dfb43a743ec00bdf41971ecd21302aacd5342"
#: W8's serving stack by content: `git rev-parse <W8>:<path>` for the two paths the gates compare
W8_TREE = "49561876c3b96a3d5e714c2d1bd1573ca4022b37"
W8_PETRI = "530c3c5764035bd85b7efb6368bc2fdca416174d"
SERVING_DIR = "src/sisteer/capeval/serving"
PETRI = "src/sisteer/capeval/tiers/petri_runner.py"
GIT_ENV = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid",
           "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z", "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z"}


def _git(repo, *args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, env={**os.environ, **GIT_ENV})


def _sh(common: Path, script: str, *, cwd: Path, env=None) -> subprocess.CompletedProcess:
    """Run ``script`` after sourcing ``common`` (this tree's literals and functions), with git resolving in ``cwd``."""
    e = dict(os.environ)
    e.pop("SISTEER_LENS_RHO_EMIT", None)
    e.pop("AS22_LADDER_SHA", None)
    e.update(env or {})
    return subprocess.run(["bash", "-c", f"cd {shlex.quote(str(cwd))}; source {shlex.quote(str(common))}; {script}"],
                          capture_output=True, text=True, env=e)


def _w8_present() -> bool:
    """W8's OBJECT is in this clone's store (enough to read its tree ids for the pins' derivation; never enough for the gates)."""
    return _git(REPO, "cat-file", "-e", f"{W8}^{{commit}}").returncode == 0


def _w8_in_history() -> bool:
    """HEAD descends from W8 -- the gates' history path."""
    return _git(REPO, "merge-base", "--is-ancestor", W8, "HEAD").returncode == 0


def _squash(dst: Path) -> str:
    """A ONE-commit repository carrying this tree's serving dir and petri_runner (the working tree's bytes) -- the shape of a published
    repository squashed to one commit: none of this repo's commits exist in it. Returns its HEAD."""
    for rel in (SERVING_DIR, PETRI):
        src, out = REPO / rel, dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, out, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(src, out)
    assert _git(dst, "init", "-q").returncode == 0
    assert _git(dst, "add", "-A").returncode == 0
    r = _git(dst, "commit", "-q", "-m", "squashed")
    assert r.returncode == 0, r.stderr
    assert _git(dst, "rev-list", "--count", "HEAD").stdout.strip() == "1"
    assert _git(dst, "cat-file", "-e", f"{W8}^{{commit}}").returncode != 0        # W8 is nowhere in it
    return _git(dst, "rev-parse", "HEAD").stdout.strip()


@pytest.fixture(scope="module")
def squash(tmp_path_factory):
    repo = tmp_path_factory.mktemp("squash")
    head = _squash(repo)
    return repo, head


def _own_ids(repo: Path):
    return _git(repo, "rev-parse", f"HEAD:{SERVING_DIR}").stdout.strip(), _git(repo, "rev-parse", f"HEAD:{PETRI}").stdout.strip()


# ------------------------------------------------------------------------------------------------------------------- the pins
def test_the_pins_are_w8s_object_ids_and_the_same_literals_in_both_lanes():
    ar, as22 = AR_COMMON.read_text(), AS22_COMMON.read_text()
    assert f"AR_W8_SERVING_TREE={W8_TREE}" in ar and f"AR_W8_PETRI_RUNNER_BLOB={W8_PETRI}" in ar and f"AR_W8_SHA={W8}" in ar
    assert f"AS22_W8_SERVING_TREE={W8_TREE}" in as22 and f"AS22_W8_PETRI_RUNNER_BLOB={W8_PETRI}" in as22 and f"AS22_W8_SHA={W8}" in as22
    if not _w8_present():
        pytest.skip(f"this clone does not carry W8 {W8[:8]} (a published repository without its development history); the pins' derivation "
                    "from W8 is not re-verified here (the literals above are)")
    assert _git(REPO, "rev-parse", f"{W8}:{SERVING_DIR}").stdout.strip() == W8_TREE
    assert _git(REPO, "rev-parse", f"{W8}:{PETRI}").stdout.strip() == W8_PETRI


def test_this_trees_serving_bytes_are_not_w8s(squash):
    """The premise of the 'differ' cases: HEAD moved serving/ and petri_runner.py past W8 (deliberately), so a squash of this tree is NOT
    W8's bytes. Were they ever equal again, the differ cases below would turn into the equal ones and this test says so first."""
    repo, _ = squash
    tree, petri = _own_ids(repo)
    assert len(tree) == 40 and len(petri) == 40
    assert (tree, petri) != (W8_TREE, W8_PETRI), "this tree's serving stack IS W8's bytes again: the differ cases below no longer apply"


# ------------------------------------------------------------------------------------------------------------------- Arditi
def test_arditi_production_without_w8_refuses_by_content_when_the_bytes_differ(squash):
    repo, head = squash
    r = _sh(AR_COMMON, "ar_assert_serving_identity production", cwd=repo)
    out = r.stdout + r.stderr
    assert r.returncode != 0, out
    tree, petri = _own_ids(repo)
    assert f"the serving stack is not W8's bytes (tree {tree} vs pinned {W8_TREE}; petri_runner {petri} vs pinned {W8_PETRI})" in out, out
    assert f"this clone does not carry W8 ({W8}), so the comparison is by content" in out and "ar_assert_serving_identity production: 1 problem(s)" in out, out
    assert "bad revision" not in out and "bad object" not in out and "byte-identical to W8" not in out and "differs from W8" not in out, out


def test_arditi_production_without_w8_and_w8s_bytes_refuses_the_unconfirmable_commit_claim(squash):
    """With the bytes equal to the pins (the pins pointed at the squash's own ids) the content half holds, and the OTHER half of the
    production claim -- the last commit touching the serving paths is AR_SERVING_IMPL_COMMIT -- has no history to be read from."""
    repo, head = squash
    tree, petri = _own_ids(repo)
    r = _sh(AR_COMMON, f"AR_W8_SERVING_TREE={tree}; AR_W8_PETRI_RUNNER_BLOB={petri}; ar_assert_serving_identity production", cwd=repo)
    out = r.stdout + r.stderr
    assert r.returncode != 0, out
    assert f"serving_impl_commit cannot be confirmed: this clone does not carry W8 ({W8})" in out and "needs the full development history" in out, out
    assert f"the serving bytes are W8's by content: tree {tree}, petri_runner {petri}" in out, out
    assert "not W8's bytes" not in out and "byte-identical to W8" not in out and "bad revision" not in out, out


def _unrelated_root(tmp_path: Path, into: Path) -> str:
    """A commit from ANOTHER repository fetched into `into`'s object store: present (`cat-file -e` succeeds), not an ancestor of HEAD. Its
    tree and message differ from the squash's (the fixed author dates would otherwise mint the squash's own root sha again)."""
    other = tmp_path / "other"
    other.mkdir()
    assert _git(other, "init", "-q").returncode == 0
    (other / "unrelated.txt").write_text("another history\n")
    assert _git(other, "add", "unrelated.txt").returncode == 0 and _git(other, "commit", "-q", "-m", "an unrelated root").returncode == 0
    root = _git(other, "rev-parse", "HEAD").stdout.strip()
    assert _git(into, "fetch", "-q", str(other), "HEAD").returncode == 0
    assert _git(into, "cat-file", "-e", f"{root}^{{commit}}").returncode == 0 and _git(into, "merge-base", "--is-ancestor", root, "HEAD").returncode == 1
    return root


def test_arditi_native_serving_impl_commit_refuses_a_clone_without_w8_whatever_its_commit_count_and_derives_where_w8_is_present(squash, tmp_path):
    """The test is W8's ABSENCE (`git cat-file -e`) and then its place in HEAD's LINEAGE (`git merge-base --is-ancestor`), never `rev-list
    --count == 1`: a published-shape repository with ONE MORE commit on top of its squash (a user's committed configs, a documentation edit)
    still has no development history, and `git log -1 -- <paths>` there would answer the squash root — not the fact the manifest records;
    W8 fetched in from an unrelated history is PRESENT and still not that history."""
    repo, head = squash
    refusal = f"the manifest's serving_impl_commit cannot be derived in {repo}: this clone does not carry W8 ({W8})"
    r = _sh(AR_COMMON, f"ar_native_serving_impl_commit {shlex.quote(str(repo))}; echo DERIVED=$AR_MANIFEST_SERVING_IMPL_COMMIT", cwd=REPO)
    out = r.stdout + r.stderr
    assert r.returncode != 0 and "DERIVED=" not in out, out
    assert refusal in out and "needs the full development history" in out and "a published repository without its development history" in out, out
    assert head[:7] in out and "one-commit" not in out and "bad revision" not in out, out              # names the clone's commit, never a count
    # the published shape with a commit on top: an unrelated file committed after the squash — two commits, still no W8, the SAME refusal
    two = tmp_path / "two"
    first = _squash(two)
    (two / "README-after-publication.md").write_text("committed after the publication\n")
    assert _git(two, "add", "README-after-publication.md").returncode == 0 and _git(two, "commit", "-q", "-m", "a commit on top").returncode == 0
    assert _git(two, "rev-list", "--count", "HEAD").stdout.strip() == "2" and _git(two, "cat-file", "-e", f"{W8}^{{commit}}").returncode != 0
    r = _sh(AR_COMMON, f"ar_native_serving_impl_commit {shlex.quote(str(two))}; echo DERIVED=$AR_MANIFEST_SERVING_IMPL_COMMIT", cwd=REPO)
    out = r.stdout + r.stderr
    assert r.returncode != 0 and "DERIVED=" not in out and f"cannot be derived in {two}: this clone does not carry W8 ({W8})" in out, out
    # a repository whose history CARRIES the pinned commit (the pin pointed at the squash's own root, the way the other content cases point the
    # pins at the squash's own ids): the derivation is the last commit touching the serving paths, whichever commit that is
    (two / PETRI).open("a").write("# touched after the first commit\n")
    assert _git(two, "commit", "-q", "-am", "touch petri").returncode == 0
    second = _git(two, "rev-parse", "HEAD").stdout.strip()
    r = _sh(AR_COMMON, f"AR_W8_SHA={first}; ar_native_serving_impl_commit {shlex.quote(str(two))} && echo DERIVED=$AR_MANIFEST_SERVING_IMPL_COMMIT", cwd=REPO)
    assert r.returncode == 0 and f"DERIVED={second}" in r.stdout, r.stdout + r.stderr
    (two / "unrelated.txt").write_text("x\n")
    assert _git(two, "add", "unrelated.txt").returncode == 0 and _git(two, "commit", "-q", "-m", "unrelated").returncode == 0
    r = _sh(AR_COMMON, f"AR_W8_SHA={first}; ar_native_serving_impl_commit {shlex.quote(str(two))} && echo DERIVED=$AR_MANIFEST_SERVING_IMPL_COMMIT", cwd=REPO)
    assert r.returncode == 0 and f"DERIVED={second}" in r.stdout and first != second, r.stdout + r.stderr   # the unrelated commit is not it
    # W8 PRESENT but unrelated: another repository's root fetched into the clone's object store — `cat-file -e` succeeds, HEAD does not descend
    # from it, and the derivation is refused by name rather than answering the clone's own root as the last-touching commit
    unrelated = _unrelated_root(tmp_path, two)
    r = _sh(AR_COMMON, f"AR_W8_SHA={unrelated}; ar_native_serving_impl_commit {shlex.quote(str(two))}; echo DERIVED=$AR_MANIFEST_SERVING_IMPL_COMMIT", cwd=REPO)
    out = r.stdout + r.stderr
    assert r.returncode != 0 and "DERIVED=" not in out, out
    assert f"W8 ({unrelated}) is in this clone's object store but HEAD" in out and "does not descend from it (an unrelated history fetched in)" in out, out
    assert "presence alone is not that history" in out and "does not carry W8" not in out, out
    # the helper's test is the pinned commit's presence AND its place in HEAD's lineage, in so many words — no commit count anywhere in it
    body = AR_COMMON.read_text()
    body = body[body.index("ar_native_serving_impl_commit(){"):]
    body = body[:body.index("\n}\n") + 3]
    assert 'cat-file -e "$AR_W8_SHA^{commit}"' in body and 'merge-base --is-ancestor "$AR_W8_SHA" HEAD' in body and "rev-list --count" not in body, body
    assert "rev-list --count" not in AR_COMMON.read_text() and "rev-list --count" not in AS22_COMMON.read_text()


def test_a_present_but_unrelated_w8_is_refused_by_both_production_gates_and_the_ladder(tmp_path):
    """Presence is never the history: a W8 fetched in from an unrelated repository sits in the object store while HEAD does not descend from
    it. Neither the history comparison (meaningful only along HEAD's lineage) nor the content path (for a clone WITHOUT W8) applies — both
    production gates and the ladder refuse by name, and never read the content pins or git's 'bad revision'."""
    repo = tmp_path / "squash"
    _squash(repo)
    unrelated = _unrelated_root(tmp_path, repo)
    tree, petri = _own_ids(repo)
    for common, fn in ((AR_COMMON, "ar_assert_serving_identity"), (AS22_COMMON, "as22_assert_serving_identity")):
        pre = "AR_" if fn.startswith("ar_") else "AS22_"
        # the pins pointed at the squash's own ids, so ONLY the lineage can refuse
        r = _sh(common, f"{pre}W8_SHA={unrelated}; {pre}W8_SERVING_TREE={tree}; {pre}W8_PETRI_RUNNER_BLOB={petri}; {fn} production", cwd=repo)
        out = r.stdout + r.stderr
        assert r.returncode != 0, (fn, out)
        assert f"W8 ({unrelated}) is in this clone's object store but HEAD" in out and "does not descend from it (an unrelated history fetched in)" in out, (fn, out)
        assert "presence alone is neither, refusing" in out and f"{fn} production: 1 problem(s)" in out, (fn, out)
        for never in ("not W8's bytes", "byte-identical to W8", "differs from W8", "cannot be confirmed", "bad revision", "bad object", "does not carry W8"):
            assert never not in out, (fn, never, out)
    head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    r = _sh(AS22_COMMON, f"AS22_W8_SHA={unrelated}; as22_assert_serving_identity ladder", cwd=repo, env={"AS22_LADDER_SHA": head, "SISTEER_LENS_RHO_EMIT": "1"})
    out = r.stdout + r.stderr
    assert r.returncode != 0 and f"regenerates the diff against W8 ({unrelated}), which HEAD {head} does not descend from" in out, out
    assert "clauses (ii) and (iii) were not evaluated" in out and "as22_assert_serving_identity ladder: 1 problem(s)" in out and "which this clone does not carry" not in out, out


def _launder(repo: Path, marker: str = "INJECTED FILTER TEST") -> None:
    """A .gitattributes clean filter that deletes every line carrying `marker` — what `git diff --quiet HEAD` reads the files through."""
    (repo / ".gitattributes").write_text("*.py filter=strip\n")
    assert _git(repo, "config", "filter.strip.clean", f"sed '/{marker}/d'").returncode == 0


def test_a_clean_filter_cannot_launder_an_injected_change_past_the_production_gates(tmp_path):
    """The gates compare the files ON DISK by their bytes as they are (`git hash-object --no-filters`) against HEAD's committed blobs, so a
    clean filter that hides an injected `raise` in petri_runner.py from `git diff --quiet HEAD` (rc 0 before and after the injection — the
    premise, asserted) does not get it served: AlphaSteer production, which passes by content on this squash, refuses naming the file; Arditi
    production names it beside its own refusal; a module dropped under the serving dir that HEAD does not carry is refused too."""
    repo = tmp_path / "squash"
    _squash(repo)
    tree, petri = _own_ids(repo)
    passing = f"AS22_W8_SERVING_TREE={tree}; AS22_W8_PETRI_RUNNER_BLOB={petri}; as22_assert_serving_identity production"
    r = _sh(AS22_COMMON, passing, cwd=repo)
    assert r.returncode == 0 and "(by content" in r.stdout, r.stdout + r.stderr                        # the premise: this squash passes by content
    _launder(repo)
    (repo / PETRI).write_bytes(b'raise RuntimeError("INJECTED FILTER TEST")\n' + (repo / PETRI).read_bytes())
    assert _git(repo, "diff", "--quiet", "HEAD").returncode == 0, "the clean filter did not launder the injection: the test's premise is gone"
    committed = _git(repo, "rev-parse", f"HEAD:{PETRI}").stdout.strip()
    raw = subprocess.run(["git", "-C", str(repo), "hash-object", "--no-filters", "--", str(repo / PETRI)], capture_output=True, text=True, check=True).stdout.strip()
    assert raw != committed
    r = _sh(AS22_COMMON, passing, cwd=repo)
    out = r.stdout + r.stderr
    assert r.returncode != 0, out
    assert f"{PETRI} on disk is blob {raw} (bytes as they are, --no-filters), not the {committed} HEAD commits" in out and "the engine serves the file on disk" in out, out
    assert "as22_assert_serving_identity production: 1 problem(s)" in out and "(by content" not in out, out
    # Arditi production: the same sentence, beside its own refusal of the unconfirmable commit claim
    r = _sh(AR_COMMON, f"AR_W8_SERVING_TREE={tree}; AR_W8_PETRI_RUNNER_BLOB={petri}; ar_assert_serving_identity production", cwd=repo)
    out = r.stdout + r.stderr
    assert r.returncode != 0 and f"{PETRI} on disk is blob {raw}" in out and "serving_impl_commit cannot be confirmed" in out, out
    assert "ar_assert_serving_identity production: 2 problem(s)" in out, out
    # the injection undone, a module HEAD does not carry dropped under the serving dir: refused by name (a .pyc or __pycache__ is not)
    (repo / PETRI).write_bytes((repo / PETRI).read_bytes().split(b"\n", 1)[1])
    assert _git(repo, "diff", "--quiet", "HEAD").returncode == 0
    (repo / SERVING_DIR / "__pycache__").mkdir(exist_ok=True)
    (repo / SERVING_DIR / "__pycache__" / "x.cpython-311.pyc").write_bytes(b"\x00")
    r = _sh(AS22_COMMON, passing, cwd=repo)
    assert r.returncode == 0, r.stdout + r.stderr
    (repo / SERVING_DIR / "injected_module.py").write_text("raise RuntimeError('INJECTED FILTER TEST')\n")
    assert _git(repo, "diff", "--quiet", "HEAD").returncode == 0                                       # an untracked file is invisible to it
    r = _sh(AS22_COMMON, passing, cwd=repo)
    out = r.stdout + r.stderr
    assert r.returncode != 0 and f"paths under the serving paths that HEAD does not carry (on disk or in the index): {SERVING_DIR}/injected_module.py" in out, out
    # the ladder reads the same bytes check: the injected module is one more problem on top of its missing-history refusal
    head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    r = _sh(AS22_COMMON, "as22_assert_serving_identity ladder", cwd=repo, env={"AS22_LADDER_SHA": head, "SISTEER_LENS_RHO_EMIT": "1"})
    out = r.stdout + r.stderr
    assert r.returncode != 0 and "injected_module.py" in out and "as22_assert_serving_identity ladder: 2 problem(s)" in out, out
    # both gates call the bytes check before any lineage branch
    for common, fn in ((AR_COMMON, "ar_serving_bytes_problems"), (AS22_COMMON, "as22_serving_bytes_problems")):
        body = common.read_text()
        assert body.count(f"\n  {fn}\n") == 1 and "hash-object --no-filters" in body and 'merge-base --is-ancestor' in body, fn


def test_the_bytes_check_is_anchored_at_the_root_reads_names_raw_and_judges_against_head(tmp_path):
    """The helper's edges, each a way the earlier shape could be fooled: a .pyc OUTSIDE a `__pycache__/` directory is importable and is
    refused (only a `__pycache__/` directory's contents are excused); run from <checkout>/src it still checks (anchored at the repository
    root, never vacuous); a file staged but not committed under the serving paths is refused (judged against HEAD, not the index); a
    committed name git C-quotes is read raw (NUL-separated): clean it passes, modified it is named raw; a committed symlink is compared by
    its target — retargeted, or replaced by a regular file, it is refused — and a regular file replaced by a symlink is refused."""
    repo = tmp_path / "squash"
    _squash(repo)
    serving = repo / SERVING_DIR
    quoted_name = 'cquoted_"name".py'
    (serving / quoted_name).write_text("X = 1\n")
    (serving / "real_module.py").write_text("Y = 2\n")
    os.symlink("real_module.py", serving / "link_module.py")
    assert _git(repo, "add", "-A").returncode == 0 and _git(repo, "commit", "-q", "-m", "a quoted name and a link").returncode == 0
    listed = _git(repo, "ls-tree", "-r", "--name-only", "HEAD", "--", SERVING_DIR).stdout
    assert '"cquoted_\\"name\\".py"' in listed.replace(SERVING_DIR + "/", ""), listed                      # git C-quotes it without -z
    assert _git(repo, "ls-tree", "HEAD", "--", f"{SERVING_DIR}/link_module.py").stdout.startswith("120000 ")
    tree, petri = _own_ids(repo)
    passing = f"AS22_W8_SERVING_TREE={tree}; AS22_W8_PETRI_RUNNER_BLOB={petri}; as22_assert_serving_identity production"
    r = _sh(AS22_COMMON, passing, cwd=repo)
    assert r.returncode == 0 and "(by content" in r.stdout, r.stdout + r.stderr                        # clean: the quoted name and the link pass
    # (a) a .pyc outside __pycache__: importable, refused by name; under a __pycache__ directory: excused
    (serving / "pkg").mkdir()
    (serving / "pkg" / "x.pyc").write_bytes(b"\x00")
    r = _sh(AS22_COMMON, passing, cwd=repo); out = r.stdout + r.stderr
    assert r.returncode != 0 and f"HEAD does not carry (on disk or in the index): {SERVING_DIR}/pkg/x.pyc" in out, out
    shutil.rmtree(serving / "pkg")
    (serving / "__pycache__").mkdir(exist_ok=True)
    (serving / "__pycache__" / "x.pyc").write_bytes(b"\x00")
    assert _sh(AS22_COMMON, passing, cwd=repo).returncode == 0
    # (b) run from <checkout>/src: the dropped module is still found — the helper is anchored at the repository root
    (serving / "injected_module.py").write_text("raise RuntimeError('INJECTED')\n")
    r = _sh(AS22_COMMON, passing, cwd=repo / "src"); out = r.stdout + r.stderr
    assert r.returncode != 0 and f"HEAD does not carry (on disk or in the index): {SERVING_DIR}/injected_module.py" in out, out
    (serving / "injected_module.py").unlink()
    assert _sh(AS22_COMMON, passing, cwd=repo / "src").returncode == 0
    # (c) staged but not committed: in the index, not in HEAD — refused by both lanes' helpers
    (serving / "staged_module.py").write_text("Z = 3\n")
    assert _git(repo, "add", f"{SERVING_DIR}/staged_module.py").returncode == 0
    for common, script in ((AS22_COMMON, passing), (AR_COMMON, f"AR_W8_SERVING_TREE={tree}; AR_W8_PETRI_RUNNER_BLOB={petri}; ar_assert_serving_identity production")):
        r = _sh(common, script, cwd=repo); out = r.stdout + r.stderr
        assert r.returncode != 0 and f"HEAD does not carry (on disk or in the index): {SERVING_DIR}/staged_module.py" in out, out
    assert _git(repo, "rm", "-q", "--cached", f"{SERVING_DIR}/staged_module.py").returncode == 0
    (serving / "staged_module.py").unlink()
    assert _sh(AS22_COMMON, passing, cwd=repo).returncode == 0
    # (d) the C-quoted name, modified on disk: named raw, never in git's quoted form
    (serving / quoted_name).write_text("X = 2\n")
    r = _sh(AS22_COMMON, passing, cwd=repo); out = r.stdout + r.stderr
    assert r.returncode != 0 and f"{SERVING_DIR}/{quoted_name} on disk is blob" in out and '\\"name\\"' not in out, out
    (serving / quoted_name).write_text("X = 1\n")
    assert _sh(AS22_COMMON, passing, cwd=repo).returncode == 0
    # (e) the symlink: retargeted -> refused by its target; replaced by a regular file carrying the target's bytes -> refused; and a regular
    # file at HEAD replaced on disk by a symlink to a copy of itself -> refused
    link = serving / "link_module.py"
    link.unlink(); os.symlink("other_module.py", link)
    r = _sh(AS22_COMMON, passing, cwd=repo); out = r.stdout + r.stderr
    assert r.returncode != 0 and f"{SERVING_DIR}/link_module.py on disk is a symlink to 'other_module.py'" in out and "the engine follows the link on disk" in out, out
    link.unlink(); link.write_text("Y = 2\n")
    r = _sh(AS22_COMMON, passing, cwd=repo); out = r.stdout + r.stderr
    assert r.returncode != 0 and f"{SERVING_DIR}/link_module.py is a symlink at HEAD but not on disk" in out, out
    link.unlink(); os.symlink("real_module.py", link)
    assert _sh(AS22_COMMON, passing, cwd=repo).returncode == 0
    copy = repo / "petri_runner_copy.py"
    shutil.copy2(repo / PETRI, copy)
    (repo / PETRI).unlink(); os.symlink(os.path.relpath(copy, (repo / PETRI).parent), repo / PETRI)
    r = _sh(AS22_COMMON, passing, cwd=repo); out = r.stdout + r.stderr
    assert r.returncode != 0 and f"{PETRI} is a symlink on disk but a regular file at HEAD" in out, out
    (repo / PETRI).unlink(); shutil.copy2(copy, repo / PETRI)
    assert _sh(AS22_COMMON, passing, cwd=repo).returncode == 0
    # the helpers' shape, in so many words: root-anchored, NUL-separated, judged against HEAD, only __pycache__/ excused
    for common in (AR_COMMON, AS22_COMMON):
        body = common.read_text()
        assert "git rev-parse --show-toplevel" in body and "ls-tree -r -z --full-tree HEAD" in body and "ls-files -z --cached --others" in body, common
        assert "*/__pycache__/*|__pycache__/*" in body and "\\.pyc$" not in body and '[ "$mode" = 120000 ]' in body and "readlink" in body, common


def test_arditi_resolve_config_derives_the_native_commit_through_the_helper():
    """The call site: the native branch of ar_resolve_config reaches the helper, never a bare `git log -1` of its own."""
    body = AR_COMMON.read_text()
    body = body[body.index("ar_resolve_config(){"):]
    assert 'ar_native_serving_impl_commit "$AR_REPO_ROOT"' in body
    assert body.count("log -1 --format=%H HEAD --") == 0, "ar_resolve_config derives the native serving_impl_commit directly again"


# ------------------------------------------------------------------------------------------------------------------- AlphaSteer
def test_as22_production_without_w8_refuses_by_content_when_the_bytes_differ(squash):
    repo, head = squash
    r = _sh(AS22_COMMON, "as22_assert_serving_identity production", cwd=repo)
    out = r.stdout + r.stderr
    assert r.returncode != 0, out
    tree, petri = _own_ids(repo)
    assert f"the serving stack is not W8's bytes (tree {tree} vs pinned {W8_TREE}; petri_runner {petri} vs pinned {W8_PETRI})" in out, out
    assert f"this clone does not carry W8 ({W8}), so the comparison is by content" in out and "as22_assert_serving_identity production: 1 problem(s)" in out, out
    assert "bad revision" not in out and "bad object" not in out and "byte-identical to W8" not in out and "differs from W8" not in out, out


def test_as22_production_without_w8_and_w8s_bytes_passes_by_content(squash):
    """AlphaSteer production claims byte-identity and nothing else, so content equal to the pins IS the claim, held: the gate passes and
    its log says the identity was read by content."""
    repo, head = squash
    tree, petri = _own_ids(repo)
    r = _sh(AS22_COMMON, f"AS22_W8_SERVING_TREE={tree}; AS22_W8_PETRI_RUNNER_BLOB={petri}; as22_assert_serving_identity production", cwd=repo)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert f"serving identity OK (production): serving stack byte-identical to W8 @ {W8}" in r.stdout and "(by content: this clone does not carry W8" in r.stdout, out
    # the other production refusals still stand on the content path
    r = _sh(AS22_COMMON, f"AS22_W8_SERVING_TREE={tree}; AS22_W8_PETRI_RUNNER_BLOB={petri}; as22_assert_serving_identity production", cwd=repo,
            env={"SISTEER_LENS_RHO_EMIT": "1"})
    assert r.returncode != 0 and "SISTEER_LENS_RHO_EMIT is SET" in r.stdout, r.stdout + r.stderr


def test_as22_ladder_without_w8_refuses_naming_the_missing_history_and_evaluates_no_frozen_patch_clause(squash):
    repo, head = squash
    r = _sh(AS22_COMMON, "as22_assert_serving_identity ladder", cwd=repo, env={"AS22_LADDER_SHA": head, "SISTEER_LENS_RHO_EMIT": "1"})
    out = r.stdout + r.stderr
    assert r.returncode != 0, out
    assert (f"the ladder's frozen-patch check regenerates the diff against W8 ({W8}), which this clone does not carry (a published repository "
            "without its development history); it needs the full development history") in out, out
    assert "clauses (ii) and (iii) were not evaluated" in out and "as22_assert_serving_identity ladder: 1 problem(s)" in out, out
    for never in ("the serving delta vs W8 is", "expected the PATCHED", "regenerated serving diff", "no staged frozen patch", "must be 1", "!= the named ladder sha",
                  "bad revision", "bad object"):
        assert never not in out, (never, out)
    # clauses (i) and (iv) still gate the ladder on the content path: an un-named tree and an unset flag are two MORE problems
    r = _sh(AS22_COMMON, "as22_assert_serving_identity ladder", cwd=repo)
    out = r.stdout + r.stderr
    assert r.returncode != 0 and "AS22_LADDER_SHA is unset or empty" in out and "must be 1" in out and "which this clone does not carry" in out, out
    assert "as22_assert_serving_identity ladder: 3 problem(s)" in out, out


# ------------------------------------------------------------------------------------------------------------------- this tree
def test_on_this_tree_the_production_gates_name_w8_and_never_gits_bad_revision():
    """Whatever this clone's shape, the production gates' words are the gates' own: W8 named, the outcome stated as a comparison -- from
    the history where HEAD descends from W8 (`differs from W8` / `byte-identical`), by content where the clone does not carry W8, the
    presence refusal where it carries W8 outside HEAD's lineage."""
    for common, fn in ((AR_COMMON, "ar_assert_serving_identity"), (AS22_COMMON, "as22_assert_serving_identity")):
        r = _sh(common, f"{fn} production", cwd=REPO)
        out = r.stdout + r.stderr
        assert W8 in out and "bad revision" not in out and "bad object" not in out, (fn, out)
        if _w8_in_history():
            assert ("differs from W8" in out) or ("byte-identical to W8" in out and "(by content" not in out), (fn, out)
            assert "not W8's bytes" not in out and "cannot be confirmed" not in out and "object store" not in out, (fn, out)
        elif _w8_present():
            assert "is in this clone's object store but HEAD" in out and "does not descend from it" in out, (fn, out)
        else:
            assert ("not W8's bytes" in out) or ("cannot be confirmed" in out) or ("(by content" in out), (fn, out)
