"""scripts/release/apply_line_table.py: rows that change comment, docstring or prose text apply; rows that touch a
string literal, a code token, a directive comment, a heredoc body or a YAML value are refused. Synthetic inputs."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "scripts" / "release" / "apply_line_table.py"
_spec = importlib.util.spec_from_file_location("apply_line_table", TOOL)
alt = importlib.util.module_from_spec(_spec)
sys.modules["apply_line_table"] = alt
_spec.loader.exec_module(alt)


PY = '''"""Module docstring: built for case Q17 (reviewer note)."""
import os  # noqa: F401

X = "case Q17 stays in a literal"  # trailing note Q17


def f(a):
    """Docstring line one Q17.

    Second paragraph.
    """
    # a comment Q17
    return a + 1  # done Q17
'''


def _rows(path, pairs):
    return [alt.Row(path, list(o), list(n), {}) for o, n in pairs]


def _apply(path, text, pairs):
    return alt.process_file(path, text, _rows(path, pairs))


def test_python_comment_docstring_and_trailing_rows_apply():
    res = _apply("m.py", PY, [
        (['"""Module docstring: built for case Q17 (reviewer note)."""'], ['"""Module docstring."""']),
        (['    """Docstring line one Q17.'], ['    """Docstring line one.']),
        (["    # a comment Q17"], ["    # a comment"]),
        (["    return a + 1  # done Q17"], ["    return a + 1  # done"]),
        (['X = "case Q17 stays in a literal"  # trailing note Q17'], ['X = "case Q17 stays in a literal"  # trailing note']),
    ])
    assert not res.residue, res.residue
    assert res.new_text.count("Q17") == 1 and '"case Q17 stays in a literal"' in res.new_text


def test_python_comment_line_may_be_deleted_or_split():
    res = _apply("m.py", PY, [(["    # a comment Q17"], [])])
    assert not res.residue and "# a comment" not in res.new_text
    res = _apply("m.py", PY, [(["    # a comment Q17"], ["    # first half", "    # second half"])])
    assert not res.residue and "# second half" in res.new_text


@pytest.mark.parametrize("old,new", [
    # a string literal that code reads
    (['X = "case Q17 stays in a literal"  # trailing note Q17'], ['X = "case stays in a literal"  # trailing note Q17']),
    # a code token
    (["    return a + 1  # done Q17"], ["    return a + 2  # done Q17"]),
    # code turned into a comment
    (["    return a + 1  # done Q17"], ["    # return a + 1"]),
    # a comment turned into code
    (["    # a comment Q17"], ["    a = 3"]),
    # a directive comment
    (["import os  # noqa: F401"], ["import os  # note"]),
    # whitespace inside the code before a trailing comment
    (["    return a + 1  # done Q17"], ["    return a+1  # done"]),
    # the docstring's quote style
    (['    """Docstring line one Q17.'], ["    '''Docstring line one."]),
])
def test_python_refuses_rows_outside_comments_and_docstrings(old, new):
    res = _apply("m.py", PY, [(old, new)])
    assert res.new_text is None and res.residue, (old, new)


def test_python_bare_string_statement_that_is_not_a_docstring_is_code():
    src = 'def f():\n    x = 1\n    "not a docstring Q17"\n    return x\n'
    res = _apply("m.py", src, [(['    "not a docstring Q17"'], ['    "not a docstring"'])])
    assert res.new_text is None and res.residue


SH = """#!/usr/bin/env bash
#SBATCH --job-name=q17
# header comment Q17
set -euo pipefail
echo "a # inside quotes Q17"  # trailing Q17
n=${#ARR[@]}  # count Q17
cat <<'EOF' > out.txt
# heredoc body Q17
EOF
python - <<PY
# python heredoc Q17
PY
read -r a b <<<"$(printf '1 2')"  # here-string Q17
X="$(python - <<'EOF' 2>/dev/null || true
# substitution heredoc Q17
EOF
)"
# after the heredocs Q17
"""


def test_shell_comment_rows_apply_and_data_rows_refuse():
    ok = _apply("s.sh", SH, [
        (["# header comment Q17"], ["# header comment"]),
        (['echo "a # inside quotes Q17"  # trailing Q17'], ['echo "a # inside quotes Q17"  # trailing']),
        (["n=${#ARR[@]}  # count Q17"], ["n=${#ARR[@]}  # count"]),
        (["""read -r a b <<<"$(printf '1 2')"  # here-string Q17"""], ["""read -r a b <<<"$(printf '1 2')"  # here-string"""]),
        (["# after the heredocs Q17"], ["# after the heredocs"]),
    ])
    assert not ok.residue, ok.residue
    for old, new in [
        (["#SBATCH --job-name=q17"], ["#SBATCH --job-name=q18"]),
        (["# heredoc body Q17"], ["# heredoc body"]),
        (["# python heredoc Q17"], ["# python heredoc"]),
        (["# substitution heredoc Q17"], ["# substitution heredoc"]),
        (['echo "a # inside quotes Q17"  # trailing Q17'], ['echo "a # inside quotes"  # trailing Q17']),
        (["#!/usr/bin/env bash"], ["#!/bin/bash"]),
    ]:
        res = _apply("s.sh", SH, [(old, new)])
        assert res.new_text is None and res.residue, old


YAML = """# top comment Q17
key: value  # trailing Q17
text: |
  # block scalar content Q17
  more
other: "a # b Q17"
"""


def test_yaml_comment_rows_apply_and_value_rows_refuse():
    ok = _apply("c.yaml", YAML, [(["# top comment Q17"], ["# top comment"]),
                                 (["key: value  # trailing Q17"], ["key: value  # trailing"])])
    assert not ok.residue, ok.residue
    for old, new in [(["  # block scalar content Q17"], ["  # block scalar content"]),
                     (['other: "a # b Q17"'], ['other: "a # b"']),
                     (["key: value  # trailing Q17"], ["key: value2  # trailing"])]:
        res = _apply("c.yaml", YAML, [(old, new)])
        assert res.new_text is None and res.residue, old


def test_sql_markdown_and_toml():
    sql = "-- comment Q17\nCREATE TABLE t (x TEXT CHECK (x IN ('a -- Q17')));  -- trail Q17\n"
    ok = _apply("m.sql", sql, [(["-- comment Q17"], ["-- comment"])])
    assert not ok.residue
    bad = _apply("m.sql", sql, [(["CREATE TABLE t (x TEXT CHECK (x IN ('a -- Q17')));  -- trail Q17"],
                                 ["CREATE TABLE t (x TEXT CHECK (x IN ('a -- ')));  -- trail Q17"])])
    assert bad.new_text is None
    md = "Prose Q17.\n\n```\ncode Q17\n```\n"
    assert not _apply("d.md", md, [(["Prose Q17."], ["Prose."])]).residue
    assert _apply("d.md", md, [(["code Q17"], ["code"])]).new_text is None
    toml = '# note Q17\nkey = "v # Q17"  # trail Q17\n'
    assert not _apply("p.toml", toml, [(["# note Q17"], ["# note"])]).residue
    assert _apply("p.toml", toml, [(['key = "v # Q17"  # trail Q17'], ['key = "v"  # trail Q17'])]).new_text is None


def test_unsupported_type_and_missing_block_are_residue(tmp_path):
    (tmp_path / "a.py").write_text(PY)
    (tmp_path / "b.txt").write_text("# note Q17\n")
    table = tmp_path / "t.jsonl"
    rows = [{"path": "a.py", "old": ["    # a comment Q17"], "new": ["    # a comment"]},
            {"path": "a.py", "old": ["    # never there"], "new": []},
            {"path": "b.txt", "old": ["# note Q17"], "new": ["# note"]},
            {"path": "skip/me.py", "old": ["# x"], "new": []}]
    table.write_text("".join(json.dumps(r) + "\n" for r in rows))
    report = tmp_path / "residue.jsonl"
    out = subprocess.run([sys.executable, str(TOOL), "apply", "--tree", str(tmp_path), "--table", str(table),
                          "--skip", "skip/*", "--report", str(report)], capture_output=True, text=True, check=True)
    summary = json.loads(out.stdout.splitlines()[-1])
    assert summary["rows_applied"] == 1 and summary["rows_skipped_by_owner"] == 1 and summary["residue"] == 2
    assert "# a comment Q17" not in (tmp_path / "a.py").read_text()
    reasons = {json.loads(ln)["old"][0]: json.loads(ln)["reason"] for ln in report.read_text().splitlines()}
    assert reasons["    # never there"] == "not found" and "no checker" in reasons["# note Q17"]
    def rc(*args):
        return subprocess.run([sys.executable, str(TOOL), *args, "--tree", str(tmp_path), "--table", str(table),
                               "--report", str(report)], capture_output=True, text=True).returncode
    assert rc("check") == 1  # a release bar: any residue fails it
    assert rc("check", "--allow-residue") == 0
    assert rc("apply", "--strict") == 1


def test_regex_pass_uses_private_rules_only(tmp_path):
    rules = {"flags": [{"class": "ticket", "re": r"\bQ\d+\b"}],
             "rewrites": [{"name": "lead", "re": r"^(\s*#\s*)Q\d+:\s+", "sub": r"\1", "cap": True}],
             "paren_segment": r"Q\d+"}
    (tmp_path / "rules.json").write_text(json.dumps(rules))
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "m.py").write_text("# Q4: the cache is rebuilt (Q5).\nx = 1  # see the note (Q6, keep this)\n")
    units = tmp_path / "units.jsonl"
    subprocess.run([sys.executable, str(TOOL), "scan", "--tree", str(tree), "--rules", str(tmp_path / "rules.json"),
                    "--out", str(units)], check=True, capture_output=True)
    rows, resid = tmp_path / "rows.jsonl", tmp_path / "resid.jsonl"
    subprocess.run([sys.executable, str(TOOL), "regex", "--rules", str(tmp_path / "rules.json"), "--units", str(units),
                    "--rows", str(rows), "--residual", str(resid)], check=True, capture_output=True)
    got = [json.loads(x) for x in rows.read_text().splitlines()]
    new = {r["old"][0]: r["new"] for r in got}
    assert new["# Q4: the cache is rebuilt (Q5)."] == ["# The cache is rebuilt."]
    assert new["x = 1  # see the note (Q6, keep this)"] == ["x = 1  # see the note (keep this)"]


def test_protect_lists_hash_pinned_and_doc_read_files(tmp_path):
    import hashlib
    (tmp_path / "pinned.py").write_text("# pinned\nX = 1\n")
    sha = hashlib.sha256((tmp_path / "pinned.py").read_bytes()).hexdigest()
    (tmp_path / "pins.txt").write_text(f"pinned.py {sha}\n")
    (tmp_path / "docmod.py").write_text('"""A docstring a reader parses."""\n')
    (tmp_path / "reader.py").write_text("import docmod\nFIELDS = docmod.__doc__.split()\n")
    out = tmp_path / "protect.tsv"
    subprocess.run([sys.executable, str(TOOL), "protect", "--tree", str(tmp_path), "--out", str(out)], check=True,
                   capture_output=True)
    listed = dict(line.split("\t", 1) for line in out.read_text().splitlines())
    assert "pinned.py" in listed and "sha256 pinned" in listed["pinned.py"]
    assert "docmod.py" in listed and "read as data" in listed["docmod.py"]


# --- inputs that a line-based reading gets wrong: each must refuse, never apply ----------------------------------

def test_python_source_in_another_encoding_is_refused():
    src = "# -*- coding: unicode_escape -*-\n# note Q17\\nX = 1\n"
    res = _apply("m.py", src, [(["# note Q17\\nX = 1"], ["# note Q17\\nX = 2"])])
    assert res.new_text is None and "encoding" in res.residue[0][1]


def test_form_feed_and_unicode_separators_keep_python_lines_in_step():
    src = 'def f():\n    """Doc."""\f\n    s = "Q17"\n    # note Q17\n'
    assert _apply("m.py", src, [(['    s = "Q17"'], ['    s = "Q18"'])], ).new_text is None
    assert _apply("m.py", src, [(['    s = "Q17"'], ['    s = "Q18"'])]).residue
    res = _apply("m.py", src, [(["    # note Q17"], ["    # note"])])
    assert not res.residue and '"Q17"' in res.new_text
    yaml_src = "a: 1  # note Q17\nb: x\u2028# not a comment Q17\n"
    assert _apply("c.yaml", yaml_src, [(["a: 1  # note Q17"], ["a: 1  # note"])]).new_text is None


@pytest.mark.parametrize("opener,close", [("cat <<1", "1"), ("cat <<$'EOF'", "EOF"), ("cat <<-\\END", "\tEND"),
                                          ("x=$(cat <<'E O F'", "E O F")])
def test_shell_heredoc_delimiters_bash_accepts_keep_the_body_code(opener, close):
    src = f"{opener}\n# payload Q17\n{close}\n" + (")\n" if opener.startswith("x=$(") else "") + "# after Q17\n"
    assert _apply("s.sh", src, [(["# payload Q17"], ["# payload"])]).new_text is None
    assert not _apply("s.sh", src, [(["# after Q17"], ["# after"])]).residue


def test_shell_heredoc_opener_the_lexer_cannot_read_refuses_the_file_and_shifts_are_not_heredocs():
    src = 'cat <<E"OF"\n# payload Q17\nEOF\n'
    assert "heredoc opener" in _apply("s.sh", src, [(["# payload Q17"], ["# payload"])]).residue[0][1]
    shift = "y=$(( 1 << 2 ))  # shift Q17\n# after Q17\n"
    assert not _apply("s.sh", shift, [(["y=$(( 1 << 2 ))  # shift Q17"], ["y=$(( 1 << 2 ))  # shift"]),
                                      (["# after Q17"], ["# after"])]).residue


def test_shell_trailing_shellcheck_directive_is_code():
    src = "echo $foo  # shellcheck disable=SC2086\necho $bar  # note Q17\n"
    assert _apply("s.sh", src, [(["echo $foo  # shellcheck disable=SC2086"],
                                 ["echo $foo  # shellcheck disable=SC2046"])]).new_text is None
    assert not _apply("s.sh", src, [(["echo $bar  # note Q17"], ["echo $bar  # note"])]).residue


def test_sql_quoted_identifiers_are_code():
    sql = 'CREATE TABLE "event--Q17" (a INT);\nCREATE TABLE [log--Q17] (b INT);\nCREATE TABLE `t--Q17` (c INT);\n-- note Q17\n'
    for line in sql.splitlines()[:3]:
        assert _apply("m.sql", sql, [([line], [line.replace("Q17", "Q18")])]).new_text is None, line
    assert not _apply("m.sql", sql, [(["-- note Q17"], ["-- note"])]).residue
    esc = 'CREATE TABLE "a""--Q17" (x INT);  -- trail Q17\n'
    assert _apply("m.sql", esc, [([esc.rstrip()], ['CREATE TABLE "a""--Q18" (x INT);  -- trail Q17'])]).new_text is None
    assert not _apply("m.sql", esc, [([esc.rstrip()], ['CREATE TABLE "a""--Q17" (x INT);  -- trail'])]).residue


def test_python_comment_inside_an_fstring_is_code():
    src = 'x = 1\ns = f"""{x\n# note Q17\n=}"""\n# outside Q17\n'
    try:
        compile(src, "m.py", "exec")
    except SyntaxError:
        pytest.skip("this Python has no comments inside f-string replacement fields")
    # the debug "=" keeps the field's source text, so deleting the comment line changes the string's value
    assert _apply("m.py", src, [(["# note Q17"], [])]).new_text is None
    assert _apply("m.py", src, [(["# note Q17"], ["# note"])]).new_text is None
    assert not _apply("m.py", src, [(["# outside Q17"], ["# outside"])]).residue


def test_markdown_fences_inside_block_quotes_and_list_items_are_code():
    md = "Intro Q17.\n\n> ```sh\n> rm -rf Q17\n> ```\n\n- step:\n  ```bash\n  run Q17\n  ```\n\nAfter Q17.\n"
    assert _apply("d.md", md, [(["> rm -rf Q17"], ["> rm -rf Q18"])]).new_text is None
    assert _apply("d.md", md, [(["  run Q17"], ["  run Q18"])]).new_text is None
    assert not _apply("d.md", md, [(["Intro Q17."], ["Intro."]), (["After Q17."], ["After."])]).residue


def test_rows_whose_path_leaves_the_tree_are_residue_and_write_nothing(tmp_path):
    tree = tmp_path / "tree"
    tree.mkdir()
    (tmp_path / "outside.py").write_text("# note Q17\n")
    (tree / "link.py").symlink_to(tmp_path / "outside.py")
    table = tmp_path / "t.jsonl"
    rows = [{"path": p, "old": ["# note Q17"], "new": ["# note"]}
            for p in ("../outside.py", str(tmp_path / "outside.py"), "link.py", "./link.py")]
    table.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = subprocess.run([sys.executable, str(TOOL), "apply", "--tree", str(tree), "--table", str(table)],
                         capture_output=True, text=True, check=True)
    assert json.loads(out.stdout.splitlines()[0])["residue"] == 4
    assert (tmp_path / "outside.py").read_text() == "# note Q17\n"


def test_regex_pass_leaves_column_alignment_to_the_rewrite_units(tmp_path):
    rules = {"flags": [{"class": "ticket", "re": r"\bQ\d+\b"}], "rewrites": [], "paren_segment": r"Q\d+"}
    (tmp_path / "rules.json").write_text(json.dumps(rules))
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "m.py").write_text("# | name    | value (Q4) |\nx = 1\n# plain note (Q5).\n")
    units, rows, resid = (tmp_path / n for n in ("units.jsonl", "rows.jsonl", "resid.jsonl"))
    subprocess.run([sys.executable, str(TOOL), "scan", "--tree", str(tree), "--rules", str(tmp_path / "rules.json"),
                    "--out", str(units)], check=True, capture_output=True)
    subprocess.run([sys.executable, str(TOOL), "regex", "--rules", str(tmp_path / "rules.json"), "--units", str(units),
                    "--rows", str(rows), "--residual", str(resid)], check=True, capture_output=True)
    got = {json.loads(x)["old"][0]: json.loads(x)["new"] for x in rows.read_text().splitlines()}
    left = [json.loads(x)["lines"][0] for x in resid.read_text().splitlines()]
    assert got == {"# plain note (Q5).": ["# plain note."]}
    assert left == ["# | name    | value (Q4) |"]


def test_a_row_bound_to_one_occurrence_applies_there_and_turns_to_residue_when_the_count_changes(tmp_path):
    src = "x = 1\n# note Q17\ny = 2\n# note Q17\nz = 3\n"
    second = alt.Row("m.py", ["# note Q17"], ["# note"], {}, (2, 2))
    res = alt.process_file("m.py", src, [second])
    assert not res.residue and res.new_text == "x = 1\n# note Q17\ny = 2\n# note\nz = 3\n"
    merged = "x = 1\ny = 2\n# note Q17\nz = 3\n"  # a change elsewhere removed the first occurrence
    res = alt.process_file("m.py", merged, [second])
    assert res.new_text is None and "occurrence 2 of 2" in res.residue[0][1]
    both = [alt.Row("m.py", ["# note Q17"], ["# first"], {}, (1, 2)), second]
    assert alt.process_file("m.py", src, both).new_text == "x = 1\n# first\ny = 2\n# note\nz = 3\n"
    table = tmp_path / "t.jsonl"
    table.write_text(json.dumps({"path": "m.py", "old": ["# note Q17"], "new": [], "occ": [3, 2]}) + "\n")
    with pytest.raises(SystemExit):
        alt.load_table(str(table))


@pytest.mark.parametrize("path,line,code", [("m.py", "x = 1  # D12", "x = 1"), ("m.sh", "echo hi  # D12", "echo hi"),
                                            ("m.yaml", "a: 1  # D12", "a: 1"), ("m.sql", "SELECT 1;  -- D12", "SELECT 1;")])
def test_a_trailing_comment_may_be_deleted_and_the_code_must_stay_byte_for_byte(path, line, code):
    src = f"{line}\n"
    res = _apply(path, src, [([line], [code])])
    assert not res.residue and res.new_text == f"{code}\n"
    assert _apply(path, src, [([line], [code + " "])]).new_text is None          # trailing whitespace left behind
    assert _apply(path, src, [([line], [code[:-1] + "X"])]).new_text is None     # the code changed
    assert _apply(path, src, [([line], [code, code])]).new_text is None          # the code duplicated


def test_doc_protect_can_name_one_docstring_and_leaves_the_others_editable(tmp_path):
    src = '"""Module text Q17."""\n\n\nclass K:\n    """Class text Q17."""\n\n    def m(self):\n        """Method text Q17."""\n        return 1\n\n\ndef f():\n    """Function text Q17."""\n    return 2\n'
    rows = {"mod": (['"""Module text Q17."""'], ['"""Module text."""']),
            "cls": (['    """Class text Q17."""'], ['    """Class text."""']),
            "meth": (['        """Method text Q17."""'], ['        """Method text."""']),
            "fn": (['    """Function text Q17."""'], ['    """Function text."""'])}

    def refused(protect):
        prot = alt.doc_protection("m.py", protect)
        return {k for k, (o, n) in rows.items() if alt.process_file("m.py", src, [alt.Row("m.py", o, n, {})], prot).new_text is None}
    assert refused(["m.py::K.m"]) == {"meth"}
    assert refused(["m.py::"]) == {"mod"}
    assert refused(["m.py::f", "m.py::K"]) == {"fn", "cls"}
    assert refused(["m.py"]) == {"mod", "cls", "meth", "fn"}
    assert refused(["other.py"]) == set()
    (tmp_path / "m.py").write_text(src)
    (tmp_path / "rules.json").write_text(json.dumps({"flags": [{"class": "t", "re": r"\bQ17\b"}], "rewrites": []}))
    (tmp_path / "dp.txt").write_text("m.py::K.m\tthe test reads K.m.__doc__\n")
    units = tmp_path / "u.jsonl"
    subprocess.run([sys.executable, str(TOOL), "scan", "--tree", str(tmp_path), "--rules", str(tmp_path / "rules.json"),
                    "--out", str(units), "--doc-protect", str(tmp_path / "dp.txt"), "--skip", "*.json", "--skip", "*.txt",
                    "--skip", "u.jsonl"], check=True, capture_output=True)
    starts = sorted(json.loads(x)["start"] for x in units.read_text().splitlines())
    assert starts == [1, 5, 13]  # the method's docstring (line 8) is left out


def test_apply_can_write_exactly_the_rows_it_applied(tmp_path):
    (tmp_path / "a.py").write_text("x = 1  # note Q17\n# keep Q17\n")
    rows = [{"path": "a.py", "old": ["x = 1  # note Q17"], "new": ["x = 1  # note"], "occ": [1, 1]},
            {"path": "a.py", "old": ["# never there"], "new": []},
            {"path": "a.py", "old": ["x = 1  # note Q17"], "new": ["x = 2"], "occ": [1, 1], "tag": "dup"}]
    table = tmp_path / "t.jsonl"
    table.write_text("".join(json.dumps(r) + "\n" for r in rows[:2]))
    out = tmp_path / "applied.jsonl"
    subprocess.run([sys.executable, str(TOOL), "apply", "--tree", str(tmp_path), "--table", str(table), "--applied", str(out),
                    "--report", str(tmp_path / "res.jsonl")], check=True, capture_output=True)
    got = [json.loads(x) for x in out.read_text().splitlines()]
    assert got == [rows[0]]


DIR_PY = '''import os  # noqa: F401 — kept for case Q17 (reviewer note)
import sys  # noqa: E402  (case Q17: the import order)
import re  # noqa: E402  # the job's sha, case Q17
x = f(1)  # type: ignore[arg-type]  case Q17: still joined
y = g(2)  # type: ignore[index]  -- a parallel map, case Q17
try:
    pass
except Exception as e:  # noqa: BLE001 -- classified, case Q17
    raise
z = 3  # note Q17  # noqa: E501
w = 4  # noqa: E402 F401 case Q17
'''


def test_python_directive_free_text_may_be_rewritten():
    res = _apply("m.py", DIR_PY, [
        (["import os  # noqa: F401 — kept for case Q17 (reviewer note)"], ["import os  # noqa: F401 — kept for the registry"]),
        (["import sys  # noqa: E402  (case Q17: the import order)"], ["import sys  # noqa: E402  (the import order)"]),
        (["import re  # noqa: E402  # the job's sha, case Q17"], ["import re  # noqa: E402  # the job's sha"]),
        (["x = f(1)  # type: ignore[arg-type]  case Q17: still joined"], ["x = f(1)  # type: ignore[arg-type]  still joined"]),
        (["y = g(2)  # type: ignore[index]  -- a parallel map, case Q17"], ["y = g(2)  # type: ignore[index]  -- a parallel map"]),
        (["except Exception as e:  # noqa: BLE001 -- classified, case Q17"], ["except Exception as e:  # noqa: BLE001 -- classified"]),
    ])
    assert not res.residue, res.residue
    assert res.new_text.count("Q17") == 2   # the two directives the rule does not split stay as they were


@pytest.mark.parametrize("old,new", [
    # the codes change
    (["import os  # noqa: F401 — kept for case Q17 (reviewer note)"], ["import os  # noqa: E402 — kept for case Q17 (reviewer note)"]),
    # the separator changes
    (["import os  # noqa: F401 — kept for case Q17 (reviewer note)"], ["import os  # noqa: F401 -- kept for the registry"]),
    (["import sys  # noqa: E402  (case Q17: the import order)"], ["import sys  # noqa: E402 (the import order)"]),
    # the free text is emptied or deleted
    (["import sys  # noqa: E402  (case Q17: the import order)"], ["import sys  # noqa: E402  ("]),
    (["import sys  # noqa: E402  (case Q17: the import order)"], ["import sys  # noqa: E402"]),
    # a bracketed ignore whose free text would turn into a second comment (a reader that skips it would then honour it)
    (["x = f(1)  # type: ignore[arg-type]  case Q17: still joined"], ["x = f(1)  # type: ignore[arg-type]  # still joined"]),
    (["x = f(1)  # type: ignore[arg-type]  case Q17: still joined"], ["x = f(1)  # type: ignore[arg-type]"]),
    # the code before the directive changes
    (["import sys  # noqa: E402  (case Q17: the import order)"], ["import sys as s  # noqa: E402  (the import order)"]),
    # a directive that does not open the comment, or codes separated by spaces: the whole comment is the directive's
    (["z = 3  # note Q17  # noqa: E501"], ["z = 3  # note  # noqa: E501"]),
    (["w = 4  # noqa: E402 F401 case Q17"], ["w = 4  # noqa: E402 F401 case"]),
])
def test_python_directive_rows_refuse_everything_but_the_free_text(old, new):
    res = _apply("m.py", DIR_PY, [(old, new)])
    assert res.new_text is None and res.residue, (old, new)


def test_directive_free_text_split_points():
    f = alt.directive_free_text
    assert f("# noqa: E402 — the order") == len("# noqa: E402 — ")
    assert f("# noqa:E402,F401  (why)") == len("# noqa:E402,F401  (")
    assert f("# pragma: no cover -- unreachable") == len("# pragma: no cover -- ")
    assert f("# type: ignore[x]  why") == len("# type: ignore[x]  ")
    for whole in ("# noqa: E402", "# noqa: E402  ", "# noqa: E402 why", "# type: ignore  why", "# type: ignore[x]  # ",
                  "# fmt: off -- why", "# -*- coding: utf-8 -*-", "# note # noqa: E501", "#!/usr/bin/env python"):
        assert f(whole) is None, whole


DIR_SH = """#!/usr/bin/env bash
#SBATCH --job-name=q17         # unique prefix (case Q17): override per run
#SBATCH --qos=high             # case Q17; re-check at submission
#SBATCH --comment="a # b"      # quoted Q17
#SBATCH --time=1:00:00
echo ok
"""


def test_shell_scheduler_directive_inline_comment_may_be_rewritten():
    res = _apply("s.sbatch", DIR_SH, [
        (["#SBATCH --job-name=q17         # unique prefix (case Q17): override per run"],
         ["#SBATCH --job-name=q17         # unique prefix: override per run"]),
        (["#SBATCH --qos=high             # case Q17; re-check at submission"],
         ["#SBATCH --qos=high             # re-check at submission"]),
    ])
    assert not res.residue, res.residue
    assert res.new_text.count("Q17") == 1   # the line with a quoted option stays as it was


@pytest.mark.parametrize("old,new", [
    # the option changes
    (["#SBATCH --job-name=q17         # unique prefix (case Q17): override per run"],
     ["#SBATCH --job-name=q18         # unique prefix: override per run"]),
    # the spacing before the comment changes
    (["#SBATCH --job-name=q17         # unique prefix (case Q17): override per run"],
     ["#SBATCH --job-name=q17 # unique prefix: override per run"]),
    # the comment is emptied or deleted
    (["#SBATCH --qos=high             # case Q17; re-check at submission"], ["#SBATCH --qos=high             #"]),
    (["#SBATCH --qos=high             # case Q17; re-check at submission"], ["#SBATCH --qos=high"]),
    # quotes in the options: the whole line is the scheduler's
    (['#SBATCH --comment="a # b"      # quoted Q17'], ['#SBATCH --comment="a # b"      # quoted']),
])
def test_shell_scheduler_directive_rows_refuse_everything_but_the_comment(old, new):
    res = _apply("s.sbatch", DIR_SH, [(old, new)])
    assert res.new_text is None and res.residue, (old, new)
