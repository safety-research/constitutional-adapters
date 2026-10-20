"""The site-alias table's path to every migration and every replica fold: ``registry migrate --site-aliases`` and
``--expect-site-aliases-sha256``, both forms of ``replica migrate``, the root's table under ``build`` / ``land`` / ``update`` / ``ingest``,
``replica init`` keeping the tables it does not own, and ``replica.sbatch`` passing the options through. Synthetic worlds only: the
legacy labels are invented (``oldsite_a``), the targets are read from the code's vocabulary, and nothing here reaches a cluster or a
live registry.

* ``registry migrate``: root mode reads ``<root>/site_aliases.toml``; ``--db`` reads only ``--site-aliases`` (never the file beside the
  database, never ``$REGISTRY_ROOT``), else the empty table; a digest other than the declared one, a prefix, a missing file, a malformed,
  shadowing, chained or unknown-target table, an unparsable file and a ``[site_aliases]`` table inside ``config.toml`` are each refused by
  name before anything is read or applied; ``--plan`` prints the digest and every pair with its rows after the shape lines, and writes
  no ``state/last_run`` stamp; ``config --write-default --force`` leaves the alias file byte-identical and the plan's digest unchanged.
* On the label migration's own fixture (a version-31 database with an invented legacy spelling in every label column): ``registry
  migrate --db`` with no ``--site-aliases`` refuses, naming each value and its row count, and applies nothing; a wrong digest refuses
  before anything is read; with the file it applies 0032, every label column is mapped row for row and the shape is the code's.
  ``--plan --sql --map-site-labels`` on a version-31 shape renders 0032's statements.
* ``replica init`` sets only ``[replica] log_root``: an extra table and its comments survive a re-init byte for byte, the alias file is
  never touched, an unchanged file is not rewritten, and an unparsable file is refused by name; the earlier behaviour (vendored under
  ``tests/fixtures/registry/``) erases the extra table on the same root, so the check can fail.
* ``replica migrate --db``: both declarations required, the digest checked and an unknown file refused before the copy, both subprocesses
  of the migrating tree receive the two options (their logs name them, and the migrating tree's own output names the digest), a migrating
  tree whose ``registry migrate`` knows no ``--site-aliases`` is refused before the copy, and ``migrate.json`` names the file, the digest and
  the pairs. The CURRENT form reads its own root's table, refuses another file and a wrong digest before the copy, and records the table in
  the generation's ``meta.json``.
* ``replica.sbatch``: ``migrate --db`` requires both options and passes them; the CURRENT form passes them (no ``--db`` needed);
  ``update`` passes ``--expect-site-aliases-sha256`` and a wrong digest refuses; a tree that predates the options is refused before work.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import pytest

from sisteer.registry import cli as CLI
from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import migrations as M
from sisteer.registry import records as R
from sisteer.registry import replica as RP
from sisteer.registry import site_aliases as SA
from sisteer.registry import synth
from sisteer.registry import vocab as V

REPO = Path(__file__).resolve().parent.parent
LANDING = REPO / "scripts" / "registry" / "landing"
PY = sys.executable
#: an invented legacy label (never a label of the code's vocabulary) and the label it maps to
LEGACY = "oldsite_a"
TARGET = V.SUBSTRATES[0]
PRE_CHANGE_INIT = REPO / "tests" / "fixtures" / "registry" / "replica_init_layout_before_aliases.py"


@pytest.fixture(autouse=True)
def _empty_installed_table():
    """Every test starts and ends with the empty table installed (an in-process fold installs its root's table)."""
    SA.reset()
    yield
    SA.reset()


# ---- worlds -------------------------------------------------------------------------------------------------------------------
def _ingest(root):
    return H.ingest_once(root, workers=0, min_free=0)


def _world(tmp_path, n_runs=3, n_items=6):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    ids = synth.write_corpus(root, n_runs=n_runs, n_items=n_items)
    _ingest(root)
    return root, ids


def _head(root):
    return RP.take_head(DB.db_path(root), wait_min=0)


def _alias_file(path, table) -> str:
    """A TOML file with a ``[site_aliases]`` table; returns the path."""
    lines = ["# the site's legacy labels, mapped to the registry's", "[site_aliases]"] + [f'{k} = "{v}"' for k, v in sorted(table.items())]
    Path(path).write_text("\n".join(lines) + "\n")
    return str(path)


def _sha_file(path) -> str:
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _tree(tmp_path: Path, name: str, patch=None) -> tuple:
    """A committed, clean git checkout standing in for an exported tree: every file of the repo's package linked under ``src/sisteer``;
    ``patch(tree)`` edits it before the commit. Returns ``(path, HEAD sha)``."""
    tree = tmp_path / name
    pkg = tree / "src" / "sisteer"
    shutil.copytree(REPO / "src" / "sisteer", pkg, copy_function=os.symlink, ignore=shutil.ignore_patterns("__pycache__"))
    if patch is not None:
        patch(tree)
    (tree / ".gitignore").write_text("__pycache__/\nout/\n")
    git = ["git", "-C", str(tree), "-c", "user.name=tree", "-c", "user.email=tree@example.invalid", "-c", "commit.gpgsign=false"]
    subprocess.run(git + ["init", "-q"], check=True)
    subprocess.run(git + ["add", "-A"], check=True)
    subprocess.run(git + ["commit", "-q", "-m", "tree"], check=True)
    sha = subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    assert len(sha) == 40
    return tree, sha


def _replace_in_tree(tree: Path, rel: str, old: str, new: str) -> None:
    """``rel`` under the tree's package (a link onto the repo's file) replaced by a copy with ``old`` -> ``new``."""
    p = tree / "src" / "sisteer" / rel
    text = (REPO / "src" / "sisteer" / rel).read_text()
    assert old in text, (rel, old)
    p.unlink()
    p.write_text(text.replace(old, new))


def _migrating_tree(tmp_path: Path, name: str, body: str = "SELECT 1;\n", patch=None) -> tuple:
    """A committed tree carrying ONE more migration (the next version) whose statements are ``body``."""
    import re

    def edit(tree: Path) -> None:
        mig = tree / "src" / "sisteer" / "registry" / "migrations"
        nxt = max(int(f.name[:4]) for f in mig.iterdir() if re.match(r"^\d{4}_", f.name)) + 1
        (mig / f"{nxt:04d}_replica_alias_step.sql").write_text(body)
        if patch is not None:
            patch(tree)
    return _tree(tmp_path, name, patch=edit)


def _module(tree: Path, *args: str, timeout: int = 600) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=str(tree / "src"), PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([PY, "-m", "sisteer.registry.replica", *args], capture_output=True, text=True, env=env, timeout=timeout)


def _bootstrap(tmp_path, root, head, name="replica") -> RP.Layout:
    lay = RP.init_layout(str(tmp_path / name), root)
    work = tmp_path / f"work-{name}"
    work.mkdir()
    res = RP.build(lay, head, str(work / "registry.sqlite"), workers=0)
    assert res["check"]["equal"], RP.format_check(res["check"])
    RP.publish(lay, str(work / "registry.sqlite"), head, mode="build", parent=None, built_by_tree="a" * 40)
    return lay


def _exe(path: Path, text: str) -> Path:
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _fake_venv(tmp_path: Path) -> Path:
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / ".gate-ready.json").write_text("{}")
    _exe(venv / "bin" / "python", f'#!/bin/bash\nexec {PY} "$@"\n')
    return venv


def _now_rec(record_type, payload, i):
    dt = datetime.fromtimestamp((R.utc_to_epoch_ms(R.utc_now()) + i) / 1000.0, tz=timezone.utc)
    return synth.rec(record_type, payload, R._fmt_utc(dt))


# ---- registry migrate: which table, from where, digest-bound -----------------------------------------------------------------
def test_root_mode_reads_the_roots_table_db_mode_reads_only_the_named_file_and_the_plan_writes_no_stamp(tmp_path, capsys, monkeypatch):
    root, _ = _world(tmp_path)
    table = {LEGACY: TARGET}
    _alias_file(os.path.join(root, SA.FILE_NAME), table)
    a, empty = SA.digest(table), SA.digest({})
    assert a != empty and len(a) == 64
    db = DB.db_path(root)
    db_stat = os.stat(db)
    stamp = os.path.join(root, "state", "last_run", "migrate.json")
    # root mode: the root's table, its digest and every pair with its rows (none: the world carries no legacy label)
    assert CLI.main(["--root", root, "migrate", "--plan"]) == 0
    out = capsys.readouterr().out
    assert f"site aliases: {os.path.join(root, SA.FILE_NAME)} (root); digest {a}; 1 pair(s)" in out
    assert f"  alias {LEGACY} -> {TARGET}: no rows" in out and "pending: none" in out and "NO alias" not in out
    # the version line and the shape line keep their places; the site-alias block follows them
    lines = out.splitlines()
    assert lines[0].startswith("schema version ") and lines[1] == "shape: matches the code's schema" and lines[2].startswith("site aliases: ")
    assert not os.path.exists(stamp), "a read-only plan writes nothing under the root"
    # --db mode: only --site-aliases is read -- never the file beside the database, never $REGISTRY_ROOT's
    monkeypatch.setenv("REGISTRY_ROOT", root)
    assert CLI.main(["migrate", "--plan", "--db", db]) == 0
    out = capsys.readouterr().out
    assert f"digest {empty}; 0 pair(s)" in out and "--db without --site-aliases: the empty table" in out and LEGACY not in out
    other = _alias_file(tmp_path / "other.toml", {LEGACY: TARGET, "oldhost_b": V.SUBSTRATES[1]})
    b = SA.digest({LEGACY: TARGET, "oldhost_b": V.SUBSTRATES[1]})
    assert CLI.main(["migrate", "--plan", "--db", db, "--site-aliases", other, "--expect-site-aliases-sha256", b]) == 0
    out = capsys.readouterr().out
    assert f"site aliases: {other} (file); digest {b}; 2 pair(s)" in out and f"  alias oldhost_b -> {V.SUBSTRATES[1]}: no rows" in out
    # the declared digest: any other value, a prefix, or a missing file refuses before anything is read
    for argv, why in ((["--expect-site-aliases-sha256", empty], f"has digest {a}, not the expected {empty}"),
                      (["--expect-site-aliases-sha256", a[:12]], "full 64-hex digest"),
                      (["--site-aliases", str(tmp_path / "absent.toml")], "no such file")):
        assert CLI.main(["--root", root, "migrate", "--plan", *argv]) == 1
        err = capsys.readouterr().err
        assert "registry migrate REFUSED" in err and why in err and "nothing was read or applied" in err, err
    assert os.stat(db).st_mtime_ns == db_stat.st_mtime_ns and not os.path.exists(stamp)
    # the apply reads the same table (named before the lock) and stamps as every applying command does
    assert CLI.main(["--root", root, "migrate", "--expect-site-aliases-sha256", a]) == 0
    out = capsys.readouterr().out
    assert f"site aliases: {os.path.join(root, SA.FILE_NAME)} (root); digest {a}; 1 pair(s)" in out and "shape matches the code's schema" in out
    assert os.path.exists(stamp)
    assert CLI.main(["--root", root, "migrate", "--expect-site-aliases-sha256", empty]) == 1
    assert "not the expected" in capsys.readouterr().err


@pytest.mark.parametrize("case", ["malformed_key", "shadowing", "chained", "unknown_target", "unparsable", "config_table"])
def test_malformed_shadowing_chained_unknown_target_unparsable_and_config_tables_are_refused_by_name(tmp_path, capsys, case):
    root, _ = _world(tmp_path)
    path = os.path.join(root, SA.FILE_NAME)
    names = {"malformed_key": "Bad-Key", "shadowing": TARGET, "chained": "oldsite_b", "unknown_target": LEGACY}
    if case == "malformed_key":
        Path(path).write_text(f'[site_aliases]\n"Bad-Key" = "{TARGET}"\n')
    elif case == "shadowing":
        Path(path).write_text(f'[site_aliases]\n{TARGET} = "{V.SUBSTRATES[1]}"\n')
    elif case == "chained":
        Path(path).write_text(f'[site_aliases]\noldsite_b = "{LEGACY}"\n{LEGACY} = "{TARGET}"\n')
    elif case == "unknown_target":
        Path(path).write_text(f'[site_aliases]\n{LEGACY} = "nowhere_at_all"\n')
    elif case == "unparsable":
        Path(path).write_text("[site_aliases\noldsite_a = \n")
    else:
        with open(os.path.join(root, "config.toml"), "a", encoding="utf-8") as fh:
            fh.write(f'\n[site_aliases]\n{LEGACY} = "{TARGET}"\n')
    db_mtime = os.stat(DB.db_path(root)).st_mtime_ns
    assert CLI.main(["--root", root, "migrate", "--plan"]) == 1
    err = capsys.readouterr().err
    assert "registry migrate REFUSED" in err and "site-alias table is refused" in err, err
    if case == "config_table":
        assert "config.toml" in err
    elif case in names:
        assert names[case] in err or (case == "chained" and LEGACY in err), err
    assert CLI.main(["--root", root, "migrate"]) == 1 and "REFUSED" in capsys.readouterr().err
    assert os.stat(DB.db_path(root)).st_mtime_ns == db_mtime
    # the replica side refuses the same tables staged in its root, before any copy, and its status names the refusal
    head = _head(root)
    lay = _bootstrap(tmp_path, root, head)
    cur = lay.current()
    if case == "config_table":
        with open(lay.config_path, "a", encoding="utf-8") as fh:
            fh.write(f'\n[site_aliases]\n{LEGACY} = "{TARGET}"\n')
    else:
        shutil.copy(path, lay.site_aliases_path)
    with pytest.raises(RP.SiteAliasesRefused):
        RP.root_site_aliases(lay)
    work = tmp_path / "mw"
    with pytest.raises(RP.SiteAliasesRefused):
        RP.migrate(lay, str(work))
    assert not (work / "registry.sqlite").exists() and lay.current() == cur
    assert "refused" in RP.status(lay)["site_aliases"]
    with pytest.raises(RP.SiteAliasesRefused):
        RP.ingest_head(str(tmp_path / "never.sqlite"), lay, head)
    assert not (tmp_path / "never.sqlite").exists(), "a refused table refuses before the database is opened"


def test_root_mode_refuses_a_config_table_whichever_alias_file_it_reads(tmp_path, capsys):
    """A ``[site_aliases]`` table in the root's ``config.toml`` is a second source, and every later fold over the root refuses it. So a
    root-mode migration refuses it by name before the database is opened, the form that names its own file with ``--site-aliases``
    included: a migration that read the named file and advanced the database would leave a registry that cannot fold new records.
    ``--db`` mode reads no root, so the same table beside the copy is never consulted."""
    root, _ = _world(tmp_path)
    with open(os.path.join(root, "config.toml"), "a", encoding="utf-8") as fh:
        fh.write(f'\n[site_aliases]\n{LEGACY} = "{TARGET}"\n')
    named = _alias_file(tmp_path / "aliases.toml", {LEGACY: TARGET})
    a = SA.digest({LEGACY: TARGET})
    db = DB.db_path(root)
    with DB.reading(db) as conn:
        rows_before = conn.execute("SELECT COUNT(*) FROM schema_migration").fetchone()[0]
    db_mtime = os.stat(db).st_mtime_ns
    for argv in (["migrate", "--site-aliases", named], ["migrate", "--plan", "--site-aliases", named],
                 ["migrate", "--site-aliases", named, "--expect-site-aliases-sha256", a], ["migrate", "--plan", "--sql", "--site-aliases", named]):
        assert CLI.main(["--root", root, *argv]) == 1, argv
        err = capsys.readouterr().err
        assert "registry migrate REFUSED" in err and os.path.join(root, "config.toml") in err and "[site_aliases]" in err, (argv, err)
        assert "nothing was read or applied" in err, err
    assert os.stat(db).st_mtime_ns == db_mtime
    with DB.reading(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM schema_migration").fetchone()[0] == rows_before
    assert CLI.main(["migrate", "--plan", "--db", db, "--site-aliases", named, "--expect-site-aliases-sha256", a]) == 0
    assert f"site aliases: {named} (file); digest {a}; 1 pair(s)" in capsys.readouterr().out


def test_config_write_default_force_leaves_the_alias_file_byte_identical_and_the_plan_digest_unchanged(tmp_path, capsys):
    root, _ = _world(tmp_path)
    path = _alias_file(os.path.join(root, SA.FILE_NAME), {LEGACY: TARGET})
    before = Path(path).read_bytes()
    assert CLI.main(["--root", root, "migrate", "--plan"]) == 0
    line = next(ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("site aliases:"))
    assert CLI.main(["--root", root, "config", "--write-default", "--force"]) == 0
    capsys.readouterr()
    assert Path(path).read_bytes() == before
    assert CLI.main(["--root", root, "migrate", "--plan"]) == 0
    assert line in capsys.readouterr().out.splitlines() and f"digest {SA.digest({LEGACY: TARGET})}" in line


# ---- replica init keeps what it does not own ---------------------------------------------------------------------------------
def test_replica_init_sets_only_the_log_root_keeps_every_other_table_and_never_touches_the_alias_file(tmp_path, capsys):
    root, _ = _world(tmp_path)
    log2 = str(tmp_path / "log2")
    os.makedirs(os.path.join(log2, "records"))
    os.makedirs(os.path.join(log2, "rulings"))
    rep = str(tmp_path / "rep")
    lay = RP.init_layout(rep, root)
    first = Path(lay.config_path).read_text()
    assert tomllib.loads(first) == {"replica": {"log_root": os.path.abspath(root)}}
    aliases = _alias_file(lay.site_aliases_path, {LEGACY: TARGET})
    alias_bytes = Path(aliases).read_bytes()
    staged = first + '\n# operator notes: kept as written\n[operator]\nnote = "kept"   # inline comment\nlevels = [1, 2]\n'
    Path(lay.config_path).write_text(staged)
    mtime = os.stat(lay.config_path).st_mtime_ns
    # the same log root: nothing to change, the file is not rewritten
    RP.init_layout(rep, root)
    assert Path(lay.config_path).read_text() == staged and os.stat(lay.config_path).st_mtime_ns == mtime
    # another log root: only that value changes; the links follow it; the alias file is untouched
    assert CLI.main(["replica", "init", "--replica-root", rep, "--log-root", log2]) == 0
    st = json.loads(capsys.readouterr().out)
    assert Path(lay.config_path).read_text() == staged.replace(json.dumps(os.path.abspath(root)), json.dumps(os.path.abspath(log2)))
    assert os.path.realpath(os.path.join(rep, "records")) == os.path.realpath(os.path.join(log2, "records")) and lay.log_root() == os.path.abspath(log2)
    assert Path(aliases).read_bytes() == alias_bytes
    assert st["site_aliases"]["digest"] == SA.digest({LEGACY: TARGET}) and st["site_aliases"]["present"] is True
    assert st["site_aliases"]["pairs"] == [[LEGACY, TARGET]]
    # a [replica] table without the key gets it under its header; a file without the table gets one appended
    Path(lay.config_path).write_text('[replica]\nkeep = 1\n\n[operator]\nnote = "x"\n')
    RP.init_layout(rep, root)
    assert Path(lay.config_path).read_text() == f'[replica]\nlog_root = {json.dumps(os.path.abspath(root))}\nkeep = 1\n\n[operator]\nnote = "x"\n'
    Path(lay.config_path).write_text('[operator]\nnote = "x"\n')
    RP.init_layout(rep, root)
    assert tomllib.loads(Path(lay.config_path).read_text()) == {"operator": {"note": "x"}, "replica": {"log_root": os.path.abspath(root)}}
    # a file init cannot read is refused by name and left as it is; so are replica settings it does not edit
    for bad, why in (("[replica\nlog_root = 1\n", "does not parse as TOML"), (f'replica = {{log_root = "{root}"}}\n', "without a [replica] table header")):
        Path(lay.config_path).write_text(bad)
        with pytest.raises(RP.ReplicaError, match="config.toml"):
            RP.init_layout(rep, root)
        assert Path(lay.config_path).read_text() == bad
        assert CLI.main(["replica", "init", "--replica-root", rep, "--log-root", root]) == 1 and why in capsys.readouterr().err
    assert Path(aliases).read_bytes() == alias_bytes


def test_the_earlier_init_erases_the_extra_table_on_the_same_root_so_the_re_init_check_can_fail(tmp_path):
    root, _ = _world(tmp_path)
    spec = importlib.util.spec_from_file_location("replica_init_layout_before_aliases", PRE_CHANGE_INIT)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    rep = str(tmp_path / "rep")
    lay = RP.init_layout(rep, root)
    aliases = _alias_file(lay.site_aliases_path, {LEGACY: TARGET})
    staged = Path(lay.config_path).read_text() + '\n[operator]\nnote = "kept"\n'
    Path(lay.config_path).write_text(staged)
    RP.init_layout(rep, root)
    assert tomllib.loads(Path(lay.config_path).read_text())["operator"] == {"note": "kept"}
    old.init_layout(rep, root)
    assert "operator" not in tomllib.loads(Path(lay.config_path).read_text()), "the earlier init erased the table: the check can fail"
    assert Path(aliases).is_file(), "the alias table lives in its own file, which neither version writes"


# ---- replica migrate: both forms -----------------------------------------------------------------------------------------------
def test_replica_migrate_db_hands_the_declared_table_to_both_subprocesses_and_records_it(tmp_path):
    root, _ = _world(tmp_path)
    head = _head(root)
    hp = str(tmp_path / "head.json.gz")
    RP.write_head(head, hp)
    kept = str(tmp_path / "kept.sqlite")
    H.rebuild(root, kept, workers=0)
    kept_sha = _sha_file(kept)
    v = head["schema_version"]
    run_tree, run_sha = _tree(tmp_path, "run-tree")
    step_tree, step_sha = _migrating_tree(tmp_path, "step-tree")
    table = {LEGACY: TARGET}
    aliases = _alias_file(tmp_path / "aliases.toml", table)
    a = SA.digest(table)
    hs = RP.head_sha256(head)
    common = ["--tree", str(run_tree), "--expect-tree", run_sha, "--migrate-tree", str(step_tree), "--expect-migrate-tree", step_sha, "--python", PY,
              "--expect-input-sha256", kept_sha, "--expect-schema", str(v + 1), "--head", hp, "--expect-head-sha256", hs]
    out1 = tmp_path / "o1"
    r = _module(run_tree, "migrate", "--db", kept, "--work", str(tmp_path / "w1"), "--out", str(out1), "--site-aliases", aliases,
                "--expect-site-aliases-sha256", a, *common)
    assert r.returncode == 0, r.stdout + r.stderr
    assert f"[replica migrate] site aliases: {aliases}, digest {a}, 1 pair(s); passed to both subprocesses" in r.stdout
    rec = json.load(open(out1 / "migrate.json"))
    assert rec["status"] == "EQUAL" and rec["after"] == v + 1 and rec["expected_site_aliases_sha256"] == a
    assert rec["identity"]["site_aliases"] == {"path": aliases, "digest": a, "pairs": [[LEGACY, TARGET]]}
    flags = f"--site-aliases {aliases} --expect-site-aliases-sha256 {a}"
    mig_log, plan_txt = (out1 / "migrate.log").read_text(), (out1 / "plan.txt").read_text()
    assert mig_log.splitlines()[0].endswith(flags) and " migrate --db " in mig_log.splitlines()[0]
    assert plan_txt.splitlines()[0].endswith(flags) and " migrate --plan --db " in plan_txt.splitlines()[0]
    # the migrating tree's own code read the file and checked the digest: its output names both
    assert f"site aliases: {aliases} (file); digest {a}; 1 pair(s)" in mig_log and f"site aliases: {aliases} (file); digest {a}; 1 pair(s)" in plan_txt
    assert f"  alias {LEGACY} -> {TARGET}: no rows" in plan_txt
    # both declarations are required (the CLI's own check, exit 1, nothing copied)
    for drop in ("--site-aliases", "--expect-site-aliases-sha256"):
        argv = ["--site-aliases", aliases, "--expect-site-aliases-sha256", a]
        i = argv.index(drop)
        del argv[i:i + 2]
        r = _module(run_tree, "migrate", "--db", kept, "--work", str(tmp_path / "w2"), "--out", str(tmp_path / "o2"), *argv, *common)
        assert r.returncode == 1 and f"{drop} is required" in r.stderr, r.stderr
    # a wrong digest, a prefix, an absent file, and a migrating tree whose registry migrate knows no --site-aliases refuse before the copy
    no_flag_tree, no_flag_sha = _migrating_tree(tmp_path, "no-flag-tree", patch=lambda t: _replace_in_tree(t, "registry/cli.py", '"--site-aliases"', '"--alias-file"'))
    cases = [(["--site-aliases", aliases, "--expect-site-aliases-sha256", SA.digest({})], common, "not the declared"),
             (["--site-aliases", aliases, "--expect-site-aliases-sha256", a[:16]], common, "64 hex"),
             (["--site-aliases", str(tmp_path / "absent.toml"), "--expect-site-aliases-sha256", a], common, "no such file"),
             (["--site-aliases", aliases, "--expect-site-aliases-sha256", a],
              [x if x not in (str(step_tree), step_sha) else {str(step_tree): str(no_flag_tree), step_sha: no_flag_sha}[x] for x in common],
              "knows no --site-aliases")]
    for alias_argv, rest, why in cases:
        r = _module(run_tree, "migrate", "--db", kept, "--work", str(tmp_path / "w3"), "--out", str(tmp_path / "o3"), *alias_argv, *rest)
        assert r.returncode == 13 and why in r.stderr, r.stdout + r.stderr
        assert not (tmp_path / "w3").exists() and not (tmp_path / "o3").exists()
    assert _sha_file(kept) == kept_sha


def test_replica_migrate_current_form_reads_its_roots_table_refuses_another_file_and_records_the_digest(tmp_path, capsys):
    root, _ = _world(tmp_path)
    head = _head(root)
    lay = _bootstrap(tmp_path, root, head)
    g1 = lay.current()
    # no file: the empty table, recorded as absent
    res = RP.migrate(lay, str(tmp_path / "mw0"), built_by_tree="b" * 40)
    g2 = lay.current()
    assert res["site_aliases"]["present"] is False and lay.meta(g2)["site_aliases"]["digest"] == SA.digest({}) and lay.meta(g2)["parent_gen_id"] == g1
    table = {LEGACY: TARGET}
    _alias_file(lay.site_aliases_path, table)
    a = SA.digest(table)
    other = _alias_file(tmp_path / "other.toml", table)
    for kw, why in (({"site_aliases_path": other}, "is not this replica root's table"),
                    ({"expect_site_aliases_sha256": SA.digest({})}, f"has digest {a}, not the expected"),
                    ({"expect_site_aliases_sha256": "abc"}, "64 hex")):
        with pytest.raises(RP.SiteAliasesRefused, match=why):
            RP.migrate(lay, str(tmp_path / "mw1"), **kw)
        assert not (tmp_path / "mw1" / "registry.sqlite").exists() and lay.current() == g2
    res = RP.migrate(lay, str(tmp_path / "mw2"), site_aliases_path=lay.site_aliases_path, expect_site_aliases_sha256=a)
    g3 = lay.current()
    meta = lay.meta(g3)
    assert g3 == res["generation"] and meta["mode"] == "migrate" and meta["parent_gen_id"] == g2
    assert meta["site_aliases"] == {"path": lay.site_aliases_path, "present": True, "digest": a, "pairs": [[LEGACY, TARGET]]}
    # the CLI: another file refuses with exit 13 and publishes nothing; the root's own file passes
    assert CLI.main(["replica", "migrate", "--replica-root", lay.root, "--work", str(tmp_path / "mw3"), "--site-aliases", other]) == 13
    assert "is not this replica root's table" in capsys.readouterr().err and lay.current() == g3
    assert CLI.main(["replica", "migrate", "--replica-root", lay.root, "--work", str(tmp_path / "mw4"), "--expect-site-aliases-sha256", a]) == 0
    assert json.loads(capsys.readouterr().out)["site_aliases"]["digest"] == a and lay.current() != g3
    assert CLI.main(["replica", "status", "--replica-root", lay.root]) == 0
    assert f"site aliases: {lay.site_aliases_path}: present, digest {a}, 1 pair(s)" in capsys.readouterr().out


# ---- replica.sbatch passes the options through ---------------------------------------------------------------------------------
def test_replica_sbatch_passes_the_alias_options_in_both_migrate_forms_and_to_update(tmp_path):
    root, ids = _world(tmp_path)
    head1 = _head(root)
    lay = _bootstrap(tmp_path, root, head1)
    kept = str(tmp_path / "kept.sqlite")
    H.rebuild(root, kept, workers=0)
    kept_sha = _sha_file(kept)
    R.write_records(os.path.join(root, "records", "late", "000001-late.jsonl"),
                    [_now_rec("run_state", {"run_id": ids["run_ids"][0], "state": "failed", "event_utc": R.utc_now(), "evidence_kind": "job_log"}, 1)])
    _ingest(root)
    head2 = _head(root)
    hp = str(tmp_path / "head2.json.gz")
    RP.write_head(head2, hp)
    h2 = RP.head_sha256(head2)
    table = {LEGACY: TARGET}
    _alias_file(lay.site_aliases_path, table)
    a = SA.digest(table)
    run_tree, run_sha = _tree(tmp_path, "run-tree")
    venv = _fake_venv(tmp_path)
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    _exe(bin_ / "srun", '#!/bin/bash\nwhile [ $# -gt 0 ]; do case "$1" in --ntasks=*|-n) shift ;; *) break ;; esac; done\nexec "$@"\n')
    env = dict(os.environ, PATH=f"{bin_}:{os.environ['PATH']}", REPLICA_WORK_ROOT=str(tmp_path / "nodelocal"), SLURM_JOB_ID="779", SLURM_CPUS_PER_TASK="2",
               USER=os.environ.get("USER", "tester"))
    sb = str(LANDING / "replica.sbatch")

    def run(*argv):
        return subprocess.run(["bash", sb, *argv], capture_output=True, text=True, env=env, timeout=600)

    # update: the declared digest reaches the module; the fold reads the root's table and records it
    out1 = tmp_path / "u1"
    r = run("update", "--db", kept, "--head", hp, "--expect-head-sha256", h2, "--out", str(out1), "--expect-input-sha256", kept_sha, "--expect-tree", run_sha,
            "--expect-site-aliases-sha256", a, str(run_tree), str(venv), lay.root)
    assert r.returncode == 0, r.stdout + r.stderr
    assert f"expect_site_aliases_sha256={a}" in r.stdout and "[replica update] EQUAL rc=0" in r.stdout
    rec = json.load(open(out1 / "update.json"))
    assert rec["site_aliases"]["digest"] == a and rec["expected_site_aliases_sha256"] == a and json.load(open(out1 / "check.json"))["site_aliases"]["digest"] == a
    assert json.load(open(out1 / "ingest.json"))["site_aliases"]["digest"] == a
    r = run("update", "--db", kept, "--head", hp, "--expect-head-sha256", h2, "--out", str(tmp_path / "u2"), "--expect-input-sha256", kept_sha, "--expect-tree", run_sha,
            "--expect-site-aliases-sha256", SA.digest({}), str(run_tree), str(venv), lay.root)
    assert r.returncode == 13 and "not the expected" in r.stderr and not (tmp_path / "u2" / "registry.sqlite").exists()
    # migrate --db: both options required, then passed through to the module and on to the migrating tree
    aliases = _alias_file(tmp_path / "aliases.toml", table)
    base = ["migrate", "--db", str(out1 / "registry.sqlite"), "--out", str(tmp_path / "m1"), "--head", hp, "--expect-head-sha256", h2,
            "--expect-input-sha256", rec["output"]["sha256"], "--expect-tree", run_sha, "--migrate-tree", str(run_tree), "--expect-migrate-tree", run_sha,
            "--expect-schema", str(head2["schema_version"])]
    r = run(*base, str(run_tree), str(venv))
    assert r.returncode == 2 and "--db needs --site-aliases FILE --expect-site-aliases-sha256 H" in r.stdout and not (tmp_path / "m1").exists()
    r = run(*base, "--site-aliases", str(tmp_path / "absent.toml"), "--expect-site-aliases-sha256", a, str(run_tree), str(venv))
    assert r.returncode == 2 and "no site-alias file at" in r.stdout
    r = run(*base, "--site-aliases", aliases, "--expect-site-aliases-sha256", a, str(run_tree), str(venv))
    assert r.returncode == 0, r.stdout + r.stderr
    assert f"site_aliases={aliases} expect_site_aliases_sha256={a}" in r.stdout and "[replica migrate] EQUAL rc=0" in r.stdout
    m = json.load(open(tmp_path / "m1" / "migrate.json"))
    assert m["identity"]["site_aliases"] == {"path": aliases, "digest": a, "pairs": [[LEGACY, TARGET]]} and m["identity"]["job_id"] == "779"
    # the CURRENT form: the options need no --db; the module reads the root's own table and records it in the generation's meta.json
    cur = lay.current()
    r = run("migrate", "--site-aliases", lay.site_aliases_path, "--expect-site-aliases-sha256", a, str(run_tree), str(venv), lay.root)
    assert r.returncode == 0, r.stdout + r.stderr
    assert f"site_aliases={lay.site_aliases_path} expect_site_aliases_sha256={a}" in r.stdout and lay.current() != cur
    assert lay.meta(lay.current())["site_aliases"]["digest"] == a
    cur = lay.current()
    r = run("migrate", "--site-aliases", aliases, str(run_tree), str(venv), lay.root)
    assert r.returncode == 13 and "is not this replica root's table" in r.stderr and lay.current() == cur
    r = run("migrate", "--out", str(tmp_path / "m2"), str(run_tree), str(venv), lay.root)
    assert r.returncode == 2 and "need --db PATH" in r.stdout
    # a tree that predates the options is refused before any work
    old_tree, _ = _tree(tmp_path, "old-tree", patch=lambda t: _replace_in_tree(t, "registry/replica.py", "--site-aliases", "--alias-file"))
    r = run("migrate", "--site-aliases", lay.site_aliases_path, str(old_tree), str(venv), lay.root)
    assert r.returncode == 2 and "predates the site-alias options" in r.stdout and lay.current() == cur
    assert not (tmp_path / "nodelocal" / "779").exists()


# ---- build and land fold under the root's table and record it ----------------------------------------------------------------
def _fake_runner(argv, env, cwd, log_path, timeout):
    junit = [a for a in argv if a.startswith("--junitxml=")][0].split("=", 1)[1]
    Path(junit).write_text('<testsuites><testsuite name="pytest" tests="4" failures="0" errors="0" skipped="0"/></testsuites>')
    Path(log_path).write_text("fake pytest\n")
    return 0


def test_build_and_land_fold_under_the_roots_table_and_record_its_digest_in_every_record(tmp_path, capsys):
    root, ids = _world(tmp_path)
    head1 = _head(root)
    hp1 = str(tmp_path / "h1.json.gz")
    RP.write_head(head1, hp1)
    rep = str(tmp_path / "rep")
    lay = RP.init_layout(rep, root)
    table = {LEGACY: TARGET}
    _alias_file(lay.site_aliases_path, table)
    a = SA.digest(table)
    out = tmp_path / "b1"
    assert CLI.main(["replica", "build", "--replica-root", rep, "--head", hp1, "--work", str(tmp_path / "bw"), "--out", str(out), "--workers", "0",
                     "--tree-sha", "f" * 40]) == 0
    printed = capsys.readouterr().out
    assert f"[replica build] site aliases: {lay.site_aliases_path} (present), digest {a}, 1 pair(s)" in printed
    assert json.load(open(out / "build.json"))["site_aliases"]["digest"] == a and json.load(open(out / "check.json"))["site_aliases"]["digest"] == a
    g1 = lay.current()
    assert lay.meta(g1)["site_aliases"]["digest"] == a and json.load(open(out / "published.json"))["site_aliases"]["digest"] == a
    assert SA.installed() == table, "the replay ran with the root's table installed"
    R.write_records(os.path.join(root, "records", "late", "000001-late.jsonl"),
                    [_now_rec("run_state", {"run_id": ids["run_ids"][0], "state": "failed", "event_utc": R.utc_now(), "evidence_kind": "job_log"}, 1)])
    _ingest(root)
    head2 = _head(root)
    SA.reset()
    lines = []
    rc = RP.land(lay, head2, str(REPO), str(tmp_path / "lw"), str(tmp_path / "l1"), PY, "probe", workers=0, runner=_fake_runner, built_by_tree="c" * 40,
                 job_id="4343", log=lines.append)
    assert rc == 0, lines
    s = json.load(open(tmp_path / "l1" / "summary.json"))
    assert s["status"] == "PASS" and s["site_aliases"]["digest"] == a and s["site_aliases"]["pairs"] == [[LEGACY, TARGET]]
    assert json.load(open(tmp_path / "l1" / "check.json"))["site_aliases"]["digest"] == a and json.load(open(tmp_path / "l1" / "ingest.json"))["site_aliases"]["digest"] == a
    assert lay.meta(lay.current())["site_aliases"]["digest"] == a and lay.current() != g1
    assert any(f"site aliases: {lay.site_aliases_path} (present), digest {a}" in ln for ln in lines) and SA.installed() == table
    # a table that does not validate refuses the landing before the copy: REFUSED, exit 13, nothing published
    Path(lay.site_aliases_path).write_text(f'[site_aliases]\n{LEGACY} = "nowhere_at_all"\n')
    cur = lay.current()
    rc = RP.land(lay, head2, str(REPO), str(tmp_path / "lw2"), str(tmp_path / "l2"), PY, "probe", workers=0, runner=_fake_runner, log=lambda m: None)
    s2 = json.load(open(tmp_path / "l2" / "summary.json"))
    assert rc == 13 and s2["status"] == "REFUSED" and "site-alias table is refused" in s2["reason"] and lay.current() == cur
    assert not (tmp_path / "lw2" / "registry.sqlite").exists()


# ---- old-label records fold under the root's table: update, and re-init then update ------------------------------------------
def _legacy_world(tmp_path):
    """A registry whose log gains, after a kept rebuild, one record an old-code producer wrote with the site's legacy spelling (a
    ``run_state`` that moves run 0 to ``failed``, its producer naming ``oldsite_a``). The live root maps the spelling through its own
    ``site_aliases.toml``, so its fold applies the record and the newer head carries the move. Returns ``(root, kept, kept_sha, head
    path, head sha256, run tree, its sha)``."""
    root, ids = _world(tmp_path)
    kept = str(tmp_path / "kept.sqlite")
    H.rebuild(root, kept, workers=0)
    _alias_file(os.path.join(root, SA.FILE_NAME), {LEGACY: TARGET})
    dt = datetime.fromtimestamp((R.utc_to_epoch_ms(R.utc_now()) + 1) / 1000.0, tz=timezone.utc)
    rec = synth.rec("run_state", {"run_id": ids["run_ids"][0], "state": "failed", "event_utc": R.utc_now(), "evidence_kind": "job_log"},
                    R._fmt_utc(dt), prod=synth.producer(substrate=LEGACY))
    R.write_records(os.path.join(root, "records", "late", "000001-legacy.jsonl"), [rec])
    rep = _ingest(root)
    assert rep.refused == 0, rep.refusals
    head = _head(root)
    with DB.reading(DB.db_path(root)) as conn:
        assert conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (ids["run_ids"][0],)).fetchone()[0] == "failed"
    hp = str(tmp_path / "head2.json.gz")
    RP.write_head(head, hp)
    SA.reset()
    run_tree, run_sha = _tree(tmp_path, "run-tree")
    return root, kept, _sha_file(kept), hp, RP.head_sha256(head), run_tree, run_sha


def _update(run_tree, run_sha, lay, kept, kept_sha, hp, hs, tmp_path, name):
    out = tmp_path / name
    r = _module(run_tree, "update", "--db", kept, "--head", hp, "--expect-head-sha256", hs, "--work", str(tmp_path / f"w-{name}"), "--out", str(out),
                "--expect-input-sha256", kept_sha, "--replica-root", lay.root, "--tree", str(run_tree), "--expect-tree", run_sha, "--workers", "0")
    return r, out


def test_replica_update_folds_an_old_label_record_under_the_roots_table_and_refuses_it_without_one(tmp_path):
    root, kept, kept_sha, hp, hs, run_tree, run_sha = _legacy_world(tmp_path)
    lay = RP.init_layout(str(tmp_path / "rep"), root)
    # no table in the replica root: the legacy spelling is refused by name, the move never folds, the check is UNEQUAL on run 0
    r, out = _update(run_tree, run_sha, lay, kept, kept_sha, hp, hs, tmp_path, "u-none")
    assert r.returncode == 7, r.stdout + r.stderr
    rec = json.load(open(out / "update.json"))
    assert rec["status"] == "UNEQUAL" and rec["site_aliases"]["present"] is False and rec["site_aliases"]["digest"] == SA.digest({})
    assert rec["ingest"]["report"]["refused"] >= 1 and "lifecycle_state" in rec["check"]["columns_differing"]
    with DB.reading(str(out / "registry.sqlite")) as conn:
        reasons = [str(x[0]) for x in conn.execute("SELECT reason FROM record_refusal").fetchall()]
    assert any(LEGACY in x and "producer.substrate" in x for x in reasons), reasons
    # the root's table: the same record folds to the registry's label and the copy equals the head
    _alias_file(lay.site_aliases_path, {LEGACY: TARGET})
    a = SA.digest({LEGACY: TARGET})
    r, out = _update(run_tree, run_sha, lay, kept, kept_sha, hp, hs, tmp_path, "u-root")
    assert r.returncode == 0, r.stdout + r.stderr
    rec = json.load(open(out / "update.json"))
    assert rec["status"] == "EQUAL" and rec["ingest"]["report"]["refused"] == 0 and rec["site_aliases"]["digest"] == a
    assert rec["site_aliases"]["pairs"] == [[LEGACY, TARGET]] and json.load(open(out / "check.json"))["site_aliases"]["digest"] == a


def test_re_init_then_update_folds_the_old_label_record_and_keeps_the_alias_file_and_the_extra_table(tmp_path, capsys):
    root, kept, kept_sha, hp, hs, run_tree, run_sha = _legacy_world(tmp_path)
    rep = str(tmp_path / "rep")
    lay = RP.init_layout(rep, root)
    aliases = _alias_file(lay.site_aliases_path, {LEGACY: TARGET})
    alias_bytes = Path(aliases).read_bytes()
    staged = Path(lay.config_path).read_text() + '\n# an operator table init does not own\n[operator]\nnote = "kept"\n'
    Path(lay.config_path).write_text(staged)
    # a normal repair: init again on the same root
    assert CLI.main(["replica", "init", "--replica-root", rep, "--log-root", root]) == 0
    capsys.readouterr()
    assert Path(lay.config_path).read_text() == staged and Path(aliases).read_bytes() == alias_bytes
    r, out = _update(run_tree, run_sha, lay, kept, kept_sha, hp, hs, tmp_path, "u-after-init")
    assert r.returncode == 0, r.stdout + r.stderr
    rec = json.load(open(out / "update.json"))
    assert rec["status"] == "EQUAL" and rec["ingest"]["report"]["refused"] == 0 and rec["site_aliases"]["digest"] == SA.digest({LEGACY: TARGET})
    assert Path(aliases).read_bytes() == alias_bytes and tomllib.loads(Path(lay.config_path).read_text())["operator"] == {"note": "kept"}
    # the earlier init on the same root erases the extra table: the check above can fail
    spec = importlib.util.spec_from_file_location("replica_init_layout_before_aliases", PRE_CHANGE_INIT)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    old.init_layout(rep, root)
    assert "operator" not in tomllib.loads(Path(lay.config_path).read_text())


# ---- the invented-label fixture: a version-31 database whose label columns hold invented legacy spellings ---------------------
def _label_rename():
    """The label migration's own fixture module: ``_populated`` (a root whose database stands at version 31 with an invented legacy
    spelling in every label column), ``_through_31`` / ``_widen`` (a fresh version-31 shape whose CHECKs also admit the spellings),
    ``_statements``, ``_migration``, ``ALIASES`` and ``LEGACY_ROWS``."""
    import test_registry_site_label_rename as LR
    return LR


def _legacy_counts(db, legacy_rows):
    with DB.reading(db) as conn:
        return {(t, c): conn.execute(f"SELECT COUNT(*) FROM {t} WHERE {c} = ?", (legacy,)).fetchone()[0] for (t, c), (legacy, _) in legacy_rows.items()}


def test_migrate_db_over_legacy_labels_refuses_without_a_table_maps_with_one_and_refuses_a_wrong_digest_first(tmp_path, capsys):
    LR = _label_rename()
    root, _ = LR._populated(tmp_path)
    db = DB.db_path(root)
    _alias_file(os.path.join(root, SA.FILE_NAME), LR.ALIASES)      # beside the database: --db mode never borrows it
    counts = _legacy_counts(db, LR.LEGACY_ROWS)
    assert all(n > 0 for n in counts.values()), counts
    pending = "0032_site_label_rename"
    # the plan with no --site-aliases reads the empty table: every legacy value named with its rows and NO alias, exit 1
    assert CLI.main(["migrate", "--plan", "--db", db]) == 1
    out = capsys.readouterr().out
    assert out.splitlines()[0].endswith(f"pending: {pending}") and f"digest {SA.digest({})}; 0 pair(s)" in out
    for (t, c), (legacy, _) in LR.LEGACY_ROWS.items():
        assert f"  legacy label {legacy!r} in {t}.{c}: {counts[(t, c)]} row(s), NO alias" in out, (t, c, out)
    # the apply with no --site-aliases: the label migration's pre-check refuses by name, each value with its rows; nothing applied
    assert CLI.main(["migrate", "--db", db]) == 1
    err = capsys.readouterr().err
    assert "site label rename refused before anything was applied" in err, err
    for (t, c), (legacy, _) in LR.LEGACY_ROWS.items():
        assert f"{t}.{c} holds {legacy!r} on {counts[(t, c)]} row(s): not a label" in err, (t, c, err)
    with DB.reading(db) as conn:
        assert M.db_version(conn) == 31
    assert _legacy_counts(db, LR.LEGACY_ROWS) == counts
    # a digest other than the declared one refuses before anything is read or applied
    aliases = _alias_file(tmp_path / "aliases.toml", LR.ALIASES)
    a = SA.digest(LR.ALIASES)
    for argv in (["migrate", "--db", db], ["migrate", "--plan", "--db", db]):
        assert CLI.main([*argv, "--site-aliases", aliases, "--expect-site-aliases-sha256", SA.digest({})]) == 1
        err = capsys.readouterr().err
        assert "registry migrate REFUSED" in err and f"has digest {a}, not the expected {SA.digest({})}" in err, err
    with DB.reading(db) as conn:
        assert M.db_version(conn) == 31
    # the plan with the file: every pair with its rows per column, nothing unaliased (exit 1 only because 0032 is pending)
    assert CLI.main(["migrate", "--plan", "--db", db, "--site-aliases", aliases, "--expect-site-aliases-sha256", a]) == 1
    out = capsys.readouterr().out
    assert f"site aliases: {aliases} (file); digest {a}; {len(LR.ALIASES)} pair(s)" in out and "NO alias" not in out
    for (t, c), (legacy, label) in LR.LEGACY_ROWS.items():
        line = next(ln for ln in out.splitlines() if ln.startswith(f"  alias {legacy} -> {label}: "))
        assert f"{t}.{c} {counts[(t, c)]}" in line, line
    # the apply with the file: 0032 applied, every label column mapped row for row, the shape the code's
    assert CLI.main(["migrate", "--db", db, "--site-aliases", aliases, "--expect-site-aliases-sha256", a]) == 0
    out = capsys.readouterr().out
    assert f"site aliases: {aliases} (file); digest {a}; {len(LR.ALIASES)} pair(s)" in out and "shape matches the code's schema" in out
    with DB.reading(db) as conn:
        assert M.db_version(conn) == 32 and DB.schema_drift(conn) == []
        for (t, c), (legacy, label) in LR.LEGACY_ROWS.items():
            assert {r[0] for r in conn.execute(f"SELECT {c} FROM {t} WHERE {c} IS NOT NULL").fetchall()} == {label}, (t, c)
            assert conn.execute(f"SELECT COUNT(*) FROM {t} WHERE {c} = ?", (label,)).fetchone()[0] == counts[(t, c)], (t, c)
    assert CLI.main(["migrate", "--plan", "--db", db, "--site-aliases", aliases]) == 0
    assert "pending: none" in capsys.readouterr().out


def test_plan_sql_with_map_site_labels_renders_the_label_migration_on_a_version_31_shape(tmp_path, capsys):
    LR = _label_rename()
    db = str(tmp_path / "shape31.sqlite")
    conn = DB.connect(db)
    LR._through_31(conn)
    LR._widen(conn)
    conn.close()
    with open(LR._migration()[2], encoding="utf-8") as fh:
        committed = fh.read()
    marker = "-- ---- migration body:"
    assert CLI.main(["migrate", "--plan", "--sql", "--map-site-labels", "--db", db]) == 1          # 0032 is pending
    body = capsys.readouterr().out.split(marker, 1)[1].split("\n", 1)[1]
    assert LR._statements(body) == LR._statements(committed)
    # without the option every label column is copied as stored, which is not the committed file
    assert CLI.main(["migrate", "--plan", "--sql", "--db", db]) == 1
    plain = capsys.readouterr().out.split(marker, 1)[1].split("\n", 1)[1]
    assert LR._statements(plain) != LR._statements(committed) and M.site_label_select("run", "substrate") not in plain
