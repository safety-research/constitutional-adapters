"""The publication scan (reproduce/tools/scrub_scan.py): this tree held to its allowlist and its ratchet, and the scanner's
behaviour on synthetic trees.

The tree test runs the scan the release bar runs, over every scope, every text file and every binary decoder, and leaves
out only the split-gzip SQLite of the paper frame (its rows mirror the Parquet set cell for cell, and the bar reads it).
The reviewed allowlist (reproduce/tools/scan_allowlist.tsv) is applied, and what remains is held to the ratchet
(reproduce/tools/scan_ratchet.json): per scope, tier and path a count may only fall, and a count that fell is lowered in
the ratchet (`scrub_scan.py --write-ratchet`). When every count reaches zero the ratchet file goes and the tree must scan
clean. Every allowlist row must still cover a hit.

The synthetic tests plant invented tokens, whose table rows scrub_scan.hash_row builds at run time, so no file here
spells a real one. A planted token fails the scan in a comment, in a string literal, in a path name and in every binary
format the scan decodes; a clean tree passes; the publication set leaves out what tests/published_repo.py names; the
allowlist and the ratchet behave as documented; scrub_grep.sh is the scan in grep form."""
from __future__ import annotations

import ast
import gzip
import io
import json
import os
import pickle
import shutil
import sqlite3
import struct
import subprocess
import sys
import warnings
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import MATCHERS, OUTPUT_TOKENS, REPO, SCAN, TOKEN_TABLE, token_hits  # noqa: E402
import published_repo  # noqa: E402

TOOL = REPO / "reproduce" / "tools"
#: the scanner's own files: they hold no word they look for, so they scan clean under their own scan
SELF_CLEAN = ("reproduce/tools/scrub_scan.py", "reproduce/tools/scrub_grep.sh", "reproduce/tools/scan_allowlist.tsv",
              "reproduce/tools/scan_ratchet.json", "tests/published_repo.py", "tests/test_no_personal_identifiers.py",
              "tests/test_scan_attribution_planted.py")
#: the front door's own files, which scrub_grep.sh greps by default (the shipped generation-config table among them)
FRONT_DOOR_ROOTS = ("reproduce", "src", "docs", "README.md", "LICENSE", "configs/sdfbase_shipped_generation_config.json")

# invented tokens: a one-run word, an exact-case word, a two-run word with its separator, and a shape with its seed
WORD = "quokkaburrow"
EXACT = "Zebrafinch"
PAIR = "zephyr-mallow"
PLANT_TABLE = (
    SCAN.hash_row("personal", WORD, out=True),
    SCAN.hash_row("process", EXACT, cs=True),
    SCAN.hash_row("storage", PAIR, sep="-"),
    ("site-host", "shape", r"\bfjord-\d+\b", {"seed": "fjord-[0-9]"}),
)


# ------------------------------------------------------------------------------------------------ this tree


def _has_zstd() -> bool:
    try:
        import zstandard  # noqa: F401, PLC0415
    except ImportError:
        return False
    return True


@pytest.fixture(scope="module")
def tree_scan():
    files = SCAN.publication_set(REPO)
    if not _has_zstd():
        skipped = [p for p in files if p.endswith(".eval")]
        warnings.warn(f"no `zstandard` in this environment: the scan leaves out {skipped} (the release bar reads them)")
        files = [p for p in files if not p.endswith(".eval")]
    sc, _ = SCAN.scan(REPO, table=TOKEN_TABLE, files=files, frame_sqlite=False, jobs=SCAN.default_jobs())
    allow = SCAN.load_allowlist(REPO / SCAN.ALLOWLIST)
    left = SCAN.apply_allowlist(sc.report.hits, allow)
    return sc, allow, left


def test_the_publication_set_holds_to_the_allowlist_and_the_ratchet(tree_scan):
    sc, _allow, left = tree_scan
    assert not sc.report.errors, sc.report.errors[:5]
    rpath = REPO / SCAN.RATCHET
    doc = SCAN.load_ratchet(rpath) if rpath.exists() else None
    rose, loose = SCAN.check_ratchet(left, doc)
    assert not rose, ("new hits above the ratchet (fix them, or add a reviewed allowlist row): "
                      + "; ".join(f"{p} {t} [{s}] {n} > {rec}" for s, t, p, n, rec in rose[:20]))
    assert not loose, f"{loose} site(s) fell below the ratchet: lower it with reproduce/tools/scrub_scan.py --write-ratchet"
    if doc is None:
        assert not left, [(h.path, h.where, h.tier) for h in left[:20]]


def test_every_allowlist_row_still_covers_a_hit(tree_scan):
    _sc, allow, _left = tree_scan
    stale = [(r.lineno, r.glob, r.tier, r.kind) for r in allow if not r.used]
    assert not stale, f"stale allowlist rows (they cover nothing; remove them): {stale}"


def test_the_scanners_own_files_scan_clean(tree_scan):
    sc, _allow, _left = tree_scan
    own = [(h.path, h.where, h.tier, h.scope) for h in sc.report.hits if h.path in SELF_CLEAN]
    assert not own, own
    # the token table itself spells no word: no hit falls inside its assignment
    src = (REPO / "tests" / "front_door_testlib.py").read_text(encoding="utf-8")
    node = next(n for n in ast.parse(src).body if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "TOKEN_TABLE" for t in n.targets))
    inside = [h for h in sc.report.hits if h.path == "tests/front_door_testlib.py" and h.where.startswith("L")
              and node.lineno <= int(h.where[1:]) <= node.end_lineno]
    assert not inside, [(h.where, h.tier) for h in inside]


def test_the_table_is_the_one_the_tools_read():
    assert SCAN.load_table(REPO / SCAN.TABLE_FILE) == TOKEN_TABLE
    rows = SCAN.parse_rows(TOKEN_TABLE)
    assert {r.tier for r in rows} == set(SCAN.TIERS), "every tier has a row"
    assert any(r.out for r in rows) and all(r.tier != SCAN.ATTRIBUTION for r in rows if r.out)
    scrub_grep = (TOOL / "scrub_grep.sh").read_text(encoding="utf-8")
    assert "scrub_scan.py" in scrub_grep and all(r in scrub_grep for r in FRONT_DOOR_ROOTS)


def test_the_publication_set_leaves_out_what_the_published_repository_leaves_out():
    assert SCAN.load_unpublished(REPO) == published_repo.UNPUBLISHED == ("CLAUDE.md",)
    files = SCAN.publication_set(REPO)
    assert not set(published_repo.UNPUBLISHED) & set(files)
    assert "README.md" in files and "tests/published_repo.py" in files


# ------------------------------------------------------------------------------------------------ synthetic trees


def _repo(root: Path, files: dict, *, git: bool = True, table=PLANT_TABLE) -> Path:
    """A synthetic repository: the files given (str or bytes), the published_repo constant and a token table of its own."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "tests").mkdir(exist_ok=True)
    (root / "tests" / "published_repo.py").write_text('UNPUBLISHED = ("CLAUDE.md",)\n')
    (root / "tests" / "front_door_testlib.py").write_text(f"TOKEN_TABLE = {tuple(table)!r}\n")
    (root / "reproduce" / "tools").mkdir(parents=True, exist_ok=True)
    (root / "reproduce" / "tools" / "scan_allowlist.tsv").write_text("# empty\n")
    for rel, data in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        (p.write_bytes if isinstance(data, bytes) else p.write_text)(data)
    if git:
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    return root


def _hits(root: Path, **kw):
    sc, _ = SCAN.scan(root, table=PLANT_TABLE, **kw)
    assert not sc.report.errors, sc.report.errors
    return sc.report.hits


def _main(root: Path, *args) -> int:
    return SCAN.main(["--root", str(root), "--jobs", "1", *args])


def test_a_clean_tree_scans_clean(tmp_path):
    root = _repo(tmp_path / "r", {"README.md": "A plain readme.\n", "src/m.py": "# a comment\nX = 'value'\n",
                                  "configs/c.yaml": "key: value  # note\n"})
    assert _hits(root) == []
    assert _main(root) == 0


def test_a_token_fails_in_a_comment_a_string_literal_and_a_path_name(tmp_path):
    root = _repo(tmp_path / "r", {"src/m.py": f"# a note on {WORD}\nX = '{WORD}'\nY = 'clean'\n",
                                  f"docs/{WORD}.md": "clean text\n"})
    got = sorted((h.path, h.where, h.tier) for h in _hits(root))
    assert got == [(f"docs/{WORD}.md", "path", "personal"), ("src/m.py", "L1", "personal"), ("src/m.py", "L2", "personal")], got
    assert _main(root) == 1


def test_case_compound_words_separators_and_shapes(tmp_path):
    lines = [f"{WORD.upper()} at the start",            # a one-run row is matched in any case
             f"the my{WORD.capitalize()}Dir variable",  # ... and as a part of a compound word
             f"{WORD}s is another word",                # but never inside a longer run
             f"{EXACT} exactly",                         # an exact-case row: this spelling only
             f"{EXACT.lower()} lowercase",
             f"{EXACT}Extra glued on",                   # an exact-case row never splits a compound
             f"a {PAIR} pair",                           # a two-run row with its separator
             f"a {PAIR.replace('-', '_')} pair",         # ... and not with another
             "a fjord-12 host", "a xfjord-12 host"]      # a shape: its word boundary holds
    root = _repo(tmp_path / "r", {"notes.txt": "\n".join(lines) + "\n"})
    got = sorted(int(h.where[1:]) for h in _hits(root))
    assert got == [1, 2, 4, 7, 9], got


def test_every_binary_format_the_scan_decodes(tmp_path):
    import pyarrow as pa  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415
    files = {}
    files["corpora/c/part-00.jsonl.gz"] = gzip.compress(json.dumps({"text": f"a story about {WORD}"}).encode() + b"\n")
    files["data/blob.pkl"] = pickle.dumps({"source": f"/x/{WORD}/y"})
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("archive/data.pkl", pickle.dumps({"provenance": {"source": f"/x/{WORD}/object.pt"}}))
        zf.writestr("archive/data/0", f"{WORD}".encode())            # a tensor storage member: numbers, never read
        zf.writestr("archive/version", f"3 {WORD}\n")                # a small metadata member: its printable runs
    files["objects/m/abc/object.pt"] = buf.getvalue()
    header = json.dumps({"__metadata__": {"note": WORD}}).encode()
    files["fixtures/v.safetensors"] = struct.pack("<Q", len(header)) + header
    files["fixtures/raw.bin"] = b"\x00\xff\xfe" + WORD.encode() + b"\x00\x01"
    files["notebooks/n.ipynb"] = json.dumps({"cells": [{"cell_type": "markdown", "source": [f"About {WORD}"]},
                                                       {"cell_type": "code", "source": ["print(1)"],
                                                        "outputs": [{"output_type": "stream", "text": [f"{WORD}\n"]}]}],
                                             "metadata": {}})
    root = _repo(tmp_path / "r", files)
    t = pa.table({"name": ["plain", f"cell {WORD}"], "n": [1, 2]}).replace_schema_metadata({"note": WORD})
    pq.write_table(t, root / "data" / "frame.parquet")
    db = tmp_path / "f.sqlite"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE run (id INTEGER, note TEXT)")
    con.execute("INSERT INTO run VALUES (1, ?)", (f"run on {WORD}",))
    con.commit()
    con.close()
    gz = gzip.compress(db.read_bytes())
    (root / "data" / "frame.sqlite.gz.00").write_bytes(gz[: len(gz) // 2])
    (root / "data" / "frame.sqlite.gz.01").write_bytes(gz[len(gz) // 2:])
    os.symlink(f"/x/{WORD}", root / "data" / "link")
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    where = {(h.path, h.where) for h in _hits(root, tmpdir=str(tmp_path))}
    expect = {("corpora/c/part-00.jsonl.gz", "L1"), ("data/blob.pkl", "op"), ("objects/m/abc/object.pt", "archive/data.pkl:op"),
              ("objects/m/abc/object.pt", "archive/version:run"), ("fixtures/v.safetensors", "header:"), ("fixtures/raw.bin", "run0"),
              ("notebooks/n.ipynb", "cell0:L1"), ("notebooks/n.ipynb", "cell1:out0"), ("data/frame.parquet", "frame.name"),
              ("data/frame.parquet", "frame:meta"), ("data/frame.sqlite", "run.note"), ("data/link", "symlink:")}
    missing = {e for e in expect if not any(p == e[0] and w.startswith(e[1]) for p, w in where)}
    assert not missing, (missing, sorted(where))
    assert not any(w.startswith("archive/data/") for _p, w in where), "a tensor storage member is numbers and is not read"


def _zstd_zip(name: bytes, raw: bytes, comp: bytes) -> bytes:
    """A zip whose one member is zstd (method 93), as inspect writes its logs; built by hand, since zipfile cannot write it."""
    crc = zipfile.crc32(raw)
    local = struct.pack("<4sHHHHHIIIHH", b"PK\x03\x04", 63, 0, 93, 0, 0, crc, len(comp), len(raw), len(name), 0) + name + comp
    central = struct.pack("<4sHHHHHHIIIHHHHHII", b"PK\x01\x02", 63, 63, 0, 93, 0, 0, crc, len(comp), len(raw),
                          len(name), 0, 0, 0, 0, 0, 0) + name
    return local + central + struct.pack("<4sHHHHIIH", b"PK\x05\x06", 0, 0, 1, 1, len(central), len(local), 0)


def test_the_zstd_members_of_an_inspect_log(tmp_path):
    zstandard = pytest.importorskip("zstandard", reason="the `.eval` decoder needs zstandard (the release bar's environment has it)")
    raw = f'{{"note": "{WORD}"}}\n'.encode()
    root = _repo(tmp_path / "r", {"fixtures/run.eval": _zstd_zip(b"samples/1.json", raw, zstandard.ZstdCompressor().compress(raw))})
    assert [(h.path, h.where) for h in _hits(root)] == [("fixtures/run.eval", "samples/1.json:L1")]


def test_an_exported_tree_without_git_is_walked(tmp_path):
    root = _repo(tmp_path / "r", {"a.txt": f"{WORD}\n", "src/__pycache__/m.cpython-312.pyc": f"{WORD}".encode(),
                                  "CLAUDE.md": f"{WORD}\n"}, git=False)
    assert [(h.path, h.where) for h in _hits(root)] == [("a.txt", "L1")]


def test_the_publication_set_leaves_out_the_maintainers_instructions(tmp_path):
    root = _repo(tmp_path / "r", {"CLAUDE.md": f"{WORD}\n", "README.md": f"{WORD}\n"})
    assert "CLAUDE.md" not in SCAN.publication_set(root)
    assert [h.path for h in _hits(root)] == ["README.md"]


def test_a_tracked_file_missing_from_the_working_tree_refuses(tmp_path):
    root = _repo(tmp_path / "r", {"a.txt": "clean\n", "b.txt": "clean\n"})
    (root / "b.txt").unlink()
    assert _main(root) == 2


def test_the_allowlist_rows_and_their_staleness(tmp_path, capsys):
    root = _repo(tmp_path / "r", {"data/x.txt": f"{WORD} one\n{WORD} two\n", "data/y.txt": f"{WORD} three\n"})
    line = SCAN.sha256_text(f"{WORD} one")
    tid = PLANT_TABLE[0][2]
    allow = root / "reproduce" / "tools" / "scan_allowlist.tsv"
    allow.write_text(f"data/x.txt\tpersonal\tline:{line}\ta reviewed line\n")
    assert _main(root) == 1                                         # the other two lines still hit
    allow.write_text(f"data/*.txt\tpersonal\trow:{tid}\tevery hit of the row under the glob\n")
    assert _main(root) == 0
    allow.write_text(f"data/*.txt\tpersonal\trow:{tid}\tcovers\nnothing/*\tpersonal\trow:{tid}\tstale\n")
    capsys.readouterr()
    assert _main(root) == 1 and "STALE" in capsys.readouterr().err
    for bad in (f"data/x.txt\tpersonal\trow:{tid}\t\n",                 # no reason
                f"data/x.txt\tnot-a-tier\trow:{tid}\treason\n",         # an unknown tier
                f"data/x.txt\tpersonal\tvalue:{line}\treason\n"):       # a value row names a data address
        allow.write_text(bad)
        assert _main(root) == 2, bad


def test_the_ratchet_only_falls(tmp_path):
    root = _repo(tmp_path / "r", {"a.txt": f"{WORD}\n{WORD} again\n", "b.txt": f"{WORD}\n"})
    rpath = tmp_path / "ratchet.json"
    hits = _hits(root)
    doc = SCAN.write_ratchet(rpath, hits)
    assert doc["totals"] == {"tree:personal": 3} and len(doc["sites"]) == 2
    assert all(len(k) == 24 for k in doc["sites"]), "a site is keyed by a hash, so the file names no path"
    assert SCAN.check_ratchet(hits, SCAN.load_ratchet(rpath)) == ([], 0)
    (root / "c.txt").write_text(f"{WORD}\n")
    rose, loose = SCAN.check_ratchet(_hits(root, files=["a.txt", "b.txt", "c.txt"]), SCAN.load_ratchet(rpath))
    assert [(p, n, rec) for _s, _t, p, n, rec in rose] == [("c.txt", 1, 0)] and loose == 0
    rose, loose = SCAN.check_ratchet(_hits(root, files=["a.txt"]), SCAN.load_ratchet(rpath))
    assert rose == [] and loose == 1                                  # b.txt fell to zero: the ratchet must follow
    assert SCAN.write_ratchet(rpath, []) is None and not rpath.exists()


def test_the_command_line_takes_all_scopes_and_writes_the_ratchet_only_from_a_whole_scan(tmp_path):
    root = _repo(tmp_path / "r", {"a.txt": f"{WORD}\n"})
    assert _main(root, "--all-scopes") == 1
    with pytest.raises(SystemExit) as ei:
        _main(root, "--all-scopes", "--scope", "tree")
    assert ei.value.code == 2
    assert _main(root, "--text-only", "--write-ratchet") == 2           # not a whole scan
    assert _main(root, "--write-ratchet") == 1 and (root / SCAN.RATCHET).exists()
    assert _main(root, "--ratchet") == 0                                 # hits within the ratchet pass
    (root / "b.txt").write_text(f"{WORD}\n")
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    assert _main(root, "--ratchet") == 1


def test_scrub_grep_is_the_scan_in_grep_form(tmp_path):
    root = _repo(tmp_path / "r", {"README.md": f"plain\nabout {WORD}\n", "src/m.py": "X = 1\n", "other/z.txt": f"{WORD}\n"})
    shutil.copy(TOOL / "scrub_scan.py", root / "reproduce" / "tools" / "scrub_scan.py")
    shutil.copy(TOOL / "scrub_grep.sh", root / "reproduce" / "tools" / "scrub_grep.sh")
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    env = {**os.environ, "SCRUB_PYTHON": sys.executable}
    r = subprocess.run(["bash", str(root / "reproduce" / "tools" / "scrub_grep.sh")], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == [f"README.md:2:about {WORD}", "scrub: 1 line(s) under " + " ".join(FRONT_DOOR_ROOTS)], r.stdout
    r = subprocess.run(["bash", str(root / "reproduce" / "tools" / "scrub_grep.sh"), "--all"], capture_output=True, text=True, env=env)
    assert r.returncode == 0 and f"other/z.txt:1:{WORD}" in r.stdout and "scrub: 2 line(s) under the publication set" in r.stdout, r.stdout


def test_the_output_check_reads_the_out_rows():
    assert OUTPUT_TOKENS is MATCHERS["out"] and OUTPUT_TOKENS.rows and all(r.out for r in OUTPUT_TOKENS.rows)
    suffix = "ts" + ".net"                           # assembled here, so this file's own text holds no host suffix
    synthetic = "tail" + "0a1b2c"                    # an invented tailnet-shaped id: the shape is what is caught
    for text in (f"https://host.{synthetic}.{suffix}/figs/", f"TAILNET={synthetic}", f"a server at box.{suffix}", synthetic):
        assert token_hits(text), text
    for text in ("the tailed distribution", "tails and heads", "detail7f", "tail_abcdef", "tailfe", "tail-0a1b"):
        assert not token_hits(text), text
