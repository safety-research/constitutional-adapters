"""scripts/release/rename_sites.py: the site-label codemod.

Every word in these tables is invented (``vexhall``, ``quillbox``), so this module spells no real site label and the
codemod itself never rewrites it. The tests cover the table's validation, the spelling of each occurrence by case and
context, the per-path modes and keep spans, and the CLI on a scratch git repository: a dry run touches nothing, an apply
rewrites, stages and ``git mv``s, and a second pass changes nothing.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "release" / "rename_sites.py"


def _load():
    spec = importlib.util.spec_from_file_location("release_rename_sites", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["release_rename_sites"] = mod
    spec.loader.exec_module(mod)
    return mod


rs = _load()

TABLE = "\n".join([
    "# an invented table",
    "word\tvexhall\tsite_one\tSITE_ONE\tSiteOne\tsite one\t-",
    "word\tquillbox\twork_host\tWORK_HOST\tWorkHost\twork host\tthe",
    "",
    "literal\tQuillbox-host\twork-host",
    "literal\told-repo.format.\tnew_repo.format.",
    "literal\tvenvs/old-repo\tvenvs/newrepo",
    "literal\told-repo-serve\tnewrepo-serve",
    "keep\t*\thttps?://[^\\s)>\\]\"'`]*",
    "keep\t*\tquillbox-keepme[\\w.]*",
    "mode\tdetector.sh\tskip",
    "keep\tlib.py\t^TOKENS = .*$",
    "mode\tNOTES.md\tidentifiers",
]) + "\n"


@pytest.fixture()
def table(tmp_path):
    path = tmp_path / "rules.tsv"
    path.write_text(TABLE, encoding="utf-8")
    return rs.load_rules(path)


def _rw(table, text, path="mod.py"):
    return table.rewrite_text(text, path)


# ---------------------------------------------------------------- the table


def _table_error(tmp_path, body):
    path = tmp_path / "bad.tsv"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(rs.RuleError) as exc:
        rs.load_rules(path)
    return str(exc.value)


def test_a_spelling_that_contains_an_old_word_is_refused(tmp_path):
    msg = _table_error(tmp_path, "word\tvexhall\tvexhall_two\tVEXHALL_TWO\tVexhallTwo\tvexhall two\t-\n")
    assert "contains the old word 'vexhall'" in msg


def test_a_spelling_that_contains_another_rows_old_word_is_refused(tmp_path):
    msg = _table_error(tmp_path, "word\tvexhall\tsite_one\tSITE_ONE\tSiteOne\tsite one\t-\n"
                                 "word\tquillbox\tvexhall_host\tVEXHALL_HOST\tVexhallHost\tvexhall host\tthe\n")
    assert "'vexhall'" in msg


def test_a_literal_whose_replacement_holds_an_old_word_is_refused(tmp_path):
    msg = _table_error(tmp_path, "word\tvexhall\tsite_one\tSITE_ONE\tSiteOne\tsite one\t-\n"
                                 "literal\tVexhall-x\tvexhall-y\n")
    assert "replacement 'vexhall-y' contains the old word" in msg


def test_malformed_rows_are_refused_with_their_line(tmp_path):
    assert ":1:" in _table_error(tmp_path, "word\tvexhall\tsite_one\n")
    assert "UPPER form" in _table_error(tmp_path, "word\tvexhall\tsite_one\tSITE_TWO\tSiteOne\tsite one\t-\n")
    assert "not one of" in _table_error(tmp_path, "mode\t*.py\tsometimes\n")
    assert "unknown row kind" in _table_error(tmp_path, "rename\ta\tb\n")
    assert "appears twice" in _table_error(tmp_path, "word\tvexhall\tsite_one\tSITE_ONE\tSiteOne\tsite one\t-\n" * 2)


def test_the_digest_is_the_sha256_of_the_table_bytes(tmp_path, table):
    import hashlib
    assert table.digest == hashlib.sha256(TABLE.encode()).hexdigest()


# ---------------------------------------------------------------- spellings


@pytest.mark.parametrize("before, after", [
    # identifiers, vocabulary values and paths
    ('SUBSTRATES = ("vexhall", "quillbox", "modal")', 'SUBSTRATES = ("site_one", "work_host", "modal")'),
    ("vexhall_origin = quillbox_copy", "site_one_origin = work_host_copy"),
    ("~/inbox/vexhall/_failed", "~/inbox/site_one/_failed"),
    ("job-vexhall-7", "job-site_one-7"),
    # long options take hyphens, so argparse's dest agrees with the renamed attribute
    ("--vexhall-budget 3 --quillbox-root /x", "--site-one-budget 3 --work-host-root /x"),
    ("a.quillbox_root", "a.work_host_root"),
    # constants and environment variables
    ("VEXHALL_ROOT=${QUILLBOX_HOME}", "SITE_ONE_ROOT=${WORK_HOST_HOME}"),
    ("$QUILLBOX", "$WORK_HOST"),
    ("VEXHALL-ONLY", "SITE-ONE-ONLY"),
    # class names
    ("class QuillboxStoreReadRefused(Exception):", "class WorkHostStoreReadRefused(Exception):"),
    ("self.former_quillbox_dir", "self.former_work_host_dir"),
])
def test_identifier_spellings(table, before, after):
    assert _rw(table, before) == after


@pytest.mark.parametrize("before, after", [
    ("# copied to Quillbox over ssh", "# copied to the work host over ssh"),
    ("# copied to the Quillbox over ssh", "# copied to the work host over ssh"),
    ("# on Vexhall and on Quillbox", "# on site one and on the work host"),
    ("# Quillbox reads the store", "# The work host reads the store"),
    ("# Vexhall nodes run driver 570", "# Site one nodes run driver 570"),
    ("x = 1\n# Quillbox pulls\n", "x = 1\n# The work host pulls\n"),
    ("# a Quillbox-side copy", "# a work-host-side copy"),
    ("# Quillbox's clock", "# The work host's clock"),
    ("# the batch sits in each quillbox.", "# the batch sits in each work host."),
    ("# Quillbox-host jobs", "# work-host jobs"),
    # a sentence also opens mid-line, but not after an abbreviation or an initial
    ("# them conflicting). Quillbox (no GPU) syncs", "# them conflicting). The work host (no GPU) syncs"),
    ("# the fixed tree 7b560918. Vexhall only: the", "# the fixed tree 7b560918. Site one only: the"),
    ("# is UNKNOWN! Vexhall's cohort", "# is UNKNOWN! Site one's cohort"),
    ("# a host, e.g. Quillbox, reads it", "# a host, e.g. the work host, reads it"),
    ("# as in J. Vexhall's note", "# as in J. site one's note"),
])
def test_prose_spellings(table, before, after):
    assert _rw(table, before) == after


@pytest.mark.parametrize("before, after", [
    # emphasis in running text is prose in capitals, with the article added when no determiner precedes it
    ("# Submitted BY THE QUILLBOX over ssh", "# Submitted BY THE WORK HOST over ssh"),
    ("# built ON VEXHALL from the fits", "# built ON SITE ONE from the fits"),
    ("# The pinned paths are VEXHALL's; elsewhere", "# The pinned paths are SITE ONE's; elsewhere"),
    ("# NO QUILLBOX STORE READ (spec).", "# NO WORK HOST STORE READ (spec)."),
    ("# SUBMITTED TO QUILLBOX. Then", "# SUBMITTED TO THE WORK HOST. Then"),
    ("    routed to VEXHALL\n", "    routed to SITE ONE\n"),
    # anything touching =, $, a quote, a dot-attribute or an underscore stays the constant spelling
    ('assert "cluster=VEXHALL" in l', 'assert "cluster=SITE_ONE" in l'),
    ('"proxy": "VEXHALL petri_proxy.sbatch"', '"proxy": "SITE_ONE petri_proxy.sbatch"'),
    ("x = QUILLBOX.attr", "x = WORK_HOST.attr"),
])
def test_all_caps_spellings(table, before, after):
    assert _rw(table, before) == after


def test_literals_apply_before_words_and_keep_their_exact_case(table):
    assert _rw(table, 'FORMAT = "old-repo.format.roster/1"') == 'FORMAT = "new_repo.format.roster/1"'


def test_overlapping_literals_agree_on_a_resource_and_its_bare_name(table):
    """A path and the bare name parsed out of it must come out alike, whichever literal matches first."""
    text = 'VENV = "/home/$USER/venvs/old-repo-serve-1a2b"\nPREFIX = "old-repo-serve"\n'
    assert _rw(table, text) == 'VENV = "/home/$USER/venvs/newrepo-serve-1a2b"\nPREFIX = "newrepo-serve"\n'


def test_rewrite_path_uses_identifier_spellings(table):
    assert table.rewrite_path("scripts/install_quillbox.sh") == "scripts/install_work_host.sh"
    assert table.rewrite_path("tests/test_Vexhall_reads.py") == "tests/test_SiteOne_reads.py"
    assert table.rewrite_path("scripts/plain.sh") == "scripts/plain.sh"


# ---------------------------------------------------------------- modes and keep spans


def test_keep_spans_are_never_rewritten(table):
    text = "# see https://quillbox.example/vexhall/x and quillbox-keepme.lan, not the quillbox\n"
    assert _rw(table, text) == ("# see https://quillbox.example/vexhall/x and quillbox-keepme.lan, not "
                                "the work host\n")


def test_a_path_scoped_keep_applies_only_to_its_path(table):
    line = 'TOKENS = "vexhall|quillbox"\n'
    assert _rw(table, line, "lib.py") == line
    assert _rw(table, line, "other.py") == 'TOKENS = "site_one|work_host"\n'


def test_skip_mode_leaves_the_file_untouched(table):
    text = "TOKENS='vexhall|quillbox'\n"
    assert _rw(table, text, "detector.sh") == text
    assert table.rewrite_bytes(text.encode(), "detector.sh") == text.encode()


def test_identifiers_mode_rewrites_identifiers_and_keeps_prose(table):
    text = "Quillbox syncs `--quillbox-root` into `install_quillbox.sh`; QUILLBOX_HOME stays.\n"
    new = _rw(table, text, "NOTES.md")
    assert new == "Quillbox syncs `--work-host-root` into `install_work_host.sh`; WORK_HOST_HOME stays.\n"
    assert table.residue(new, "NOTES.md") == [(1, new.rstrip("\n"))]   # the prose it kept is reported, never hidden


def test_binary_and_undecodable_bytes_come_back_unchanged(table):
    assert table.rewrite_bytes(b"\x00vexhall", "x.dat") == b"\x00vexhall"
    assert table.rewrite_bytes(b"vexhall", "x.parquet") == b"vexhall"
    assert table.rewrite_bytes(b"\xffvexhall", "x.txt") == b"\xffvexhall"


# ---------------------------------------------------------------- idempotence

CORPUS = """\
#!/bin/bash
# Quillbox pulls the records; the Quillbox-side copy is ON VEXHALL.
SUBSTRATES=("vexhall" "quillbox")   # VEXHALL_ROOT and $QUILLBOX
python3 tool.py --vexhall-budget 3 --dest vexhall
class QuillboxStoreReadRefused: pass
\"\"\"Quillbox reads it. Vexhall's queue. VEXHALL's nodes.\"\"\"
x = {"quillbox_mirror_present": True, "vexhall": "Vexhall"}
"""


def test_a_second_pass_over_the_output_changes_nothing(table):
    once = _rw(table, CORPUS)
    assert once != CORPUS
    assert _rw(table, once) == once
    assert table.residue(once, "mod.py") == []


# ---------------------------------------------------------------- the CLI on a scratch repository


def _git(root, *args):
    return subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
                          check=True, capture_output=True, text=True).stdout


def _run(root, rules, *args):
    return subprocess.run([sys.executable, str(SCRIPT), "--rules", str(rules), "--root", str(root), *args],
                          capture_output=True, text=True)


INSTALLER = "#!/bin/bash\n# Install on Quillbox.\nDEST=vexhall\n" + "".join(f"echo step {i}\n" for i in range(20))


@pytest.fixture()
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "install_quillbox.sh").write_text(INSTALLER)
    (root / "notes.txt").write_text("Copied to Quillbox, then to Vexhall.\n")
    (root / "plain.txt").write_text("nothing to rename here\n")
    (root / "detector.sh").write_text("TOKENS='vexhall|quillbox'\n")
    (root / "blob.bin").write_bytes(b"\x00\x01vexhall\x00")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "base")
    rules = tmp_path / "rules.tsv"
    rules.write_text(TABLE, encoding="utf-8")
    return root, rules, tmp_path


def test_dry_run_reports_and_touches_nothing(repo):
    root, rules, tmp = repo
    before = _git(root, "status", "--porcelain")
    proc = _run(root, rules, "--dry-run", "--report", str(tmp / "report.txt"), "--json", str(tmp / "summary.json"))
    assert proc.returncode == 0, proc.stderr
    assert _git(root, "status", "--porcelain") == before == ""
    import json
    summary = json.loads((tmp / "summary.json").read_text())
    assert summary["renames"] == {"scripts/install_quillbox.sh": "scripts/install_work_host.sh"}
    assert summary["files_changed"] == 2
    assert summary["skipped"] == {"binary": 1, "mode-skip": 1}
    assert summary["rules_sha256"] == rs.load_rules(rules).digest
    report = (tmp / "report.txt").read_text()
    assert "scripts/install_quillbox.sh -> scripts/install_work_host.sh" in report
    assert "detector.sh" in report   # its old words are listed as residue, never hidden


def test_check_fails_before_apply_and_passes_after(repo):
    root, rules, tmp = repo
    assert _run(root, rules, "--check").returncode == 1
    proc = _run(root, rules, "--apply", "--report", str(tmp / "report.txt"))
    assert proc.returncode == 0, proc.stderr
    assert _run(root, rules, "--check").returncode == 0


def test_apply_rewrites_stages_and_moves_with_git_mv(repo):
    root, rules, tmp = repo
    proc = _run(root, rules, "--apply", "--json", str(tmp / "summary.json"))
    assert proc.returncode == 0, proc.stderr
    status = _git(root, "status", "--porcelain").splitlines()
    assert sorted(status) == ["M  notes.txt", "R  scripts/install_quillbox.sh -> scripts/install_work_host.sh"]
    assert not (root / "scripts" / "install_quillbox.sh").exists()
    assert (root / "scripts" / "install_work_host.sh").read_text() == INSTALLER.replace(
        "Install on Quillbox.", "Install on the work host.").replace("DEST=vexhall", "DEST=site_one")
    assert (root / "notes.txt").read_text() == "Copied to the work host, then to site one.\n"
    assert (root / "detector.sh").read_text() == "TOKENS='vexhall|quillbox'\n"
    assert (root / "blob.bin").read_bytes() == b"\x00\x01vexhall\x00"
    import json
    summary = json.loads((tmp / "summary.json").read_text())
    assert summary["second_pass_files_changed"] == 0 and summary["second_pass_renames"] == 0


def test_apply_refuses_when_a_rename_target_exists(repo):
    root, rules, _ = repo
    (root / "scripts" / "install_work_host.sh").write_text("already here\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "clash")
    proc = _run(root, rules, "--apply")
    assert proc.returncode == 2
    assert "rename target(s) already exist" in proc.stderr
    assert _git(root, "status", "--porcelain") == ""


def test_a_malformed_table_is_refused_before_anything_is_read(repo):
    root, _, tmp = repo
    bad = tmp / "bad.tsv"
    bad.write_text("word\tvexhall\n", encoding="utf-8")
    proc = _run(root, bad, "--apply")
    assert proc.returncode == 2
    assert "refusing" in proc.stderr
    assert _git(root, "status", "--porcelain") == ""


def test_pathspecs_limit_the_pass(repo):
    root, rules, _ = repo
    proc = _run(root, rules, "--apply", "notes.txt")
    assert proc.returncode == 0, proc.stderr
    assert _git(root, "status", "--porcelain").splitlines() == ["M  notes.txt"]
