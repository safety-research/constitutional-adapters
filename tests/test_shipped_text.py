"""The release's shipped-text rewrite and its checker, on synthetic data.

``scripts/release/shipped_text.py`` rewrites the free text of the paper frame, the registry transcripts and the dimension
seeds under one declared field policy; ``scripts/release/check_shipped_text.py`` proves that only declared leaves moved and
that no attribution survives in any free-text leaf. Every label, session name and process reference below is invented: the
site label ``oldsite`` stands for a private cluster name and ``Wrangler9`` for an agent session.
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys

import pytest
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts", "release"))

import check_shipped_text as CST  # noqa: E402
import shipped_text as ST  # noqa: E402

SENT = "not published"


def policy_dict(**over):
    p = {
        "sentinel": SENT,
        "classes": [
            ["frame:ruling:source_ref:$", "sentinel", SENT],
            ["frame:ruling:ruled_by:$", "sentinel", "the authors"],
            ["transcripts:rulings_r2_v*.jsonl:source_ref:$", "sentinel", SENT],
            ["transcripts:rulings_r2_v*.jsonl:ruled_by:$", "sentinel", "the authors"],
            ["frame:judgment:batch_ids_json:$[*]", "drop", "__drop_key__"],
            ["frame:*:producer_session:$", "drop", None],
            ["*:*:*:*.note", "prose"], ["*:*:note:$", "prose"], ["*:*:*:*.notes", "prose"],
            ["frame:ruling:reason:$", "prose"], ["transcripts:rulings_r2_v*.jsonl:reason:$", "prose"],
        ],
        "identifier_keys": [r"(?:^|_)id$", r"sha256$"],
        "site_identifier": [[r"oldsite", "cluster_a"]],
        "site_prose": [[r"(?<![A-Za-z0-9_])[Oo]ldsite(?![A-Za-z0-9_])", "cluster A"]],
        "names": ["Wrangler9"],
        "frame_aliases": {},
        "provenance": {"paths": ["transcripts:rulings_r2_v*.jsonl:provenance:$", "frame:ruling:extra_json:$.provenance"],
                       "keep": ["kind", "message_utc"], "withhold": ["relayed_by"]},
        "exempt": [],
    }
    p.update(over)
    return p


def make_ctx(tmp_path, policy=None, rules=(), table=(), inventory=None):
    pp = tmp_path / "policy.json"
    pp.write_text(json.dumps(policy or policy_dict()))
    rp = None
    if rules:
        rp = tmp_path / "rules.tsv"
        rp.write_text("id\tpattern\treplacement\n" + "".join(f"{i}\t{a}\t{b}\n" for i, (a, b) in enumerate(rules)))
    tp = None
    if table:
        tp = tmp_path / "table.jsonl"
        tp.write_text("".join(json.dumps(r) + "\n" for r in table))
    return ST.Context(ST.load_policy(str(pp)), inventory or {}, ST.load_rules(str(rp) if rp else None),
                      ST.load_tables([str(tp)] if tp else []))


RULES = [(r" \(D\d+\)", ""), (r"the user's ruling", "the authors' rule")]


# --------------------------------------------------------------------------------------------------------------------------------
# the rewrite
# --------------------------------------------------------------------------------------------------------------------------------


def test_policy_glob_matches_list_indices(tmp_path):
    ctx = make_ctx(tmp_path)
    assert ctx.classify("frame", "judgment", "batch_ids_json", (0,), "abc")[0] == "drop"
    assert ctx.classify("seeds", "x.yaml", "rows", (0, "note"), "a b")[0] == "prose"


YAML_IN = """\
# the seed of the oldsite rows (D12)
rows:
  - id: oldsite_row          # a key code joins on
    note: Built on Oldsite (D12) per the user's ruling
    shared: &shared
      notes: kept as is
    why: &why "kept per the user's ruling (D12)"
    again: *why
  - id: other
    <<: *shared
    flow: {note: "Oldsite flow (D12)", n: 3}
    block: |
      first line on Oldsite
      second line
"""


def test_yaml_splices_only_declared_scalars_and_comments(tmp_path):
    ctx = make_ctx(tmp_path, rules=RULES)
    changes = []
    out = ST.rewrite_text(ctx, "configs/registry/seed_x.yaml", YAML_IN, changes)
    doc = yaml.safe_load(out)
    assert doc["rows"][0]["id"] == "cluster_a_row"
    assert doc["rows"][0]["note"] == "Built on cluster A per the authors' rule"
    assert doc["rows"][1]["flow"] == {"note": "cluster A flow", "n": 3}
    assert doc["rows"][1]["notes"] == "kept as is"                # the merge still resolves
    assert doc["rows"][0]["why"] == doc["rows"][0]["again"] == "kept per the authors' rule"   # the anchor survives its rewrite
    assert '&why "kept per the authors\' rule"' in out
    assert doc["rows"][1]["block"] == "first line on cluster A\nsecond line\n"
    assert out.splitlines()[0] == "# the seed of the cluster A rows"
    assert "# a key code joins on" in out and "&shared" in out and "<<: *shared" in out
    assert {c.cls for c in changes} >= {"identifier", "prose", "comment"}


def test_jsonl_keeps_unchanged_records_and_reshapes_provenance(tmp_path):
    ctx = make_ctx(tmp_path, rules=RULES)
    keep = {"_header": True, "schema": "x/1"}
    rul = {"ruling_id": "r1", "reason": "Excluded (D40) per the user's ruling", "ruled_by": "the user, relayed by Wrangler9",
           "source_ref": "a private message", "provenance": {"kind": "relayed", "relayed_by": "Wrangler9",
                                                             "message_utc": "2026-01-01T00:00:00Z", "instruction": "a quote"}}
    text = json.dumps(keep, ensure_ascii=False) + "\n" + json.dumps(rul, ensure_ascii=False) + "\n"
    out = ST.rewrite_text(ctx, "configs/registry/rulings_r2_v1.jsonl", text)
    lines = out.split("\n")
    assert lines[0] == json.dumps(keep, ensure_ascii=False)
    got = json.loads(lines[1])
    assert got["reason"] == "Excluded per the authors' rule"
    assert got["ruled_by"] == "the authors" and got["source_ref"] == SENT
    assert got["provenance"] == {"kind": "relayed", "relayed_by": SENT, "message_utc": "2026-01-01T00:00:00Z"}


def test_prose_table_by_original_and_by_site_value(tmp_path):
    a, b = "Oldsite holds it, Wrangler9 says", "Planned on Oldsite by Wrangler9"
    table = [{"scope": "seeds", "path": "", "old_sha256": ST.sha256_text(a), "new": "cluster A holds it"},
             {"scope": "seeds", "path": "", "old_sha256": ST.sha256_text("Planned on cluster A by Wrangler9"),
              "new": "Planned on cluster A"}]
    ctx = make_ctx(tmp_path, table=table)
    assert ctx.transform("seeds", "s.yaml", "rows", (0, "note"), a)[1] == "cluster A holds it"
    assert ctx.transform("seeds", "s.yaml", "rows", (0, "note"), b)[1] == "Planned on cluster A"
    assert not ctx.unresolved


def test_unresolved_prose_is_reported(tmp_path):
    ctx = make_ctx(tmp_path)
    ctx.transform("seeds", "s.yaml", "rows", (0, "note"), "asked for by Wrangler9")
    assert any(v["old"] == "asked for by Wrangler9" for v in ctx.unresolved.values())


def _frame(path):
    c = sqlite3.connect(path)
    c.executescript("""
        CREATE TABLE run (run_id TEXT, n INTEGER, x REAL, extra_json TEXT, producer_session TEXT);
        CREATE TABLE run_state_event (run_id TEXT, note TEXT);
        CREATE TABLE judgment (judgment_id TEXT, batch_ids_json TEXT);
    """)
    c.executemany("INSERT INTO run VALUES (?,?,?,?,?)", [
        ("oldsite_r1", 1, 0.5, json.dumps({"note": "staged on Oldsite (D9)", "k": "oldsite"}, sort_keys=True, separators=(",", ":")), "Wrangler9"),
        ("r2", 2, None, None, None),
    ])
    c.executemany("INSERT INTO run_state_event VALUES (?,?)", [("r2", "plain note")])
    c.executemany("INSERT INTO judgment VALUES (?,?)", [("j1", json.dumps(["msgbatch_1", "keep"]))])
    c.commit()
    c.close()


def test_frame_rewrite_keeps_rows_and_non_text_cells(tmp_path):
    src, dst = str(tmp_path / "a.sqlite"), str(tmp_path / "b.sqlite")
    _frame(src)
    pol = policy_dict()
    pol["classes"].insert(0, ["frame:judgment:batch_ids_json:$[*]", "drop", "__drop_key__"])
    ctx = make_ctx(tmp_path, policy=pol, rules=RULES)
    ST.rewrite_frame(ctx, src, dst)
    c = sqlite3.connect(dst)
    rows = c.execute("SELECT run_id, n, x, extra_json, producer_session FROM run ORDER BY rowid").fetchall()
    assert rows[0][:3] == ("cluster_a_r1", 1, 0.5) and rows[1] == ("r2", 2, None, None, None)
    assert json.loads(rows[0][3]) == {"note": "staged on cluster A", "k": "cluster_a"}
    assert rows[0][4] is None
    assert json.loads(c.execute("SELECT batch_ids_json FROM judgment").fetchone()[0]) == []


# --------------------------------------------------------------------------------------------------------------------------------
# check 5: attribution planted outside the declared leaves fails the scan in every scope
# --------------------------------------------------------------------------------------------------------------------------------


def _scan_files(ctx, texts):
    e = CST.Report().start("scan")
    CST.scan_files(ctx, texts, e)
    return e


def _scan_frame(ctx, path):
    e = CST.Report().start("scan")
    CST.scan_frame(ctx, path, e)
    return e


def _clean_frame(path, extra=None, note="plain note"):
    c = sqlite3.connect(path)
    c.executescript("CREATE TABLE run (run_id TEXT, extra_json TEXT); CREATE TABLE run_state_event (run_id TEXT, note TEXT);")
    c.execute("INSERT INTO run VALUES (?, ?)", ("r1", json.dumps(dict({"k": "v"}, **(extra or {})), sort_keys=True)))
    c.execute("INSERT INTO run_state_event VALUES (?, ?)", ("r1", note))
    c.commit()
    c.close()


def test_clean_data_passes_the_scan(tmp_path):
    ctx = make_ctx(tmp_path)
    p = str(tmp_path / "f.sqlite")
    _clean_frame(p)
    assert _scan_frame(ctx, p)["ok"]
    texts = {"configs/registry/rulings_r2_v1.jsonl": json.dumps({"ruling_id": "r1", "target_json": {"rule": "keep it"}}) + "\n",
             "configs/registry/seed_x.yaml": "rows:\n  - id: a\n    notes: a neutral note\n"}
    assert _scan_files(ctx, texts)["ok"]


def test_planted_frame_extra_json_key_fails(tmp_path):
    ctx = make_ctx(tmp_path)
    p = str(tmp_path / "f.sqlite")
    _clean_frame(p, extra={"planted_key": "kept per the user's ruling"})
    e = _scan_frame(ctx, p)
    assert not e["ok"] and any("frame:run:extra_json:$.planted_key" in f for f in e["failures"])


def test_planted_transcript_target_json_key_fails(tmp_path):
    ctx = make_ctx(tmp_path)
    rec = {"ruling_id": "r1", "target_json": {"precedent": "as Wrangler9 asked"}}
    e = _scan_files(ctx, {"configs/registry/rulings_r2_v1.jsonl": json.dumps(rec) + "\n"})
    assert not e["ok"] and any("transcripts:rulings_r2_v1.jsonl:target_json:$.precedent" in f for f in e["failures"])


def test_planted_seed_notes_value_fails(tmp_path):
    ctx = make_ctx(tmp_path)
    e = _scan_files(ctx, {"configs/registry/seed_x.yaml": "rows:\n  - id: a\n    notes: dropped (D555) by the user\n"})
    assert not e["ok"] and any("seeds:seed_x.yaml:rows:$[*].notes" in f for f in e["failures"])


def test_planted_run_state_event_note_fails(tmp_path):
    ctx = make_ctx(tmp_path)
    p = str(tmp_path / "f.sqlite")
    _clean_frame(p, note="relayed by Wrangler9")
    e = _scan_frame(ctx, p)
    assert not e["ok"] and any("frame:run_state_event:note:$" in f for f in e["failures"])


def test_planted_yaml_comment_fails(tmp_path):
    ctx = make_ctx(tmp_path)
    e = _scan_files(ctx, {"configs/registry/seed_x.yaml": "rows: []   # added by Wrangler9\n"})
    assert not e["ok"] and any(f.startswith("comments:seed_x.yaml:1") for f in e["failures"])


BLOCK_HEADER_YAML = """\
rows:
  - id: a
    contract: >   # checked on Oldsite (D12)
      a folded body
      on two lines
    note: >   # a header comment stays
      Built on Oldsite (D12)
"""


def test_a_comment_on_a_block_scalar_header_line_is_a_comment(tmp_path):
    """The comment after a block scalar's `>` or `|` is a YAML comment, not scalar text: the walker lists it, the comment pass
    rewrites it, and the scan reads it."""
    got = {n: line[off:] for n, line, off in ST.yaml_comment_lines("configs/registry/seed_x.yaml", BLOCK_HEADER_YAML)}
    assert got == {3: "# checked on Oldsite (D12)", 6: "# a header comment stays"}, got
    ctx = make_ctx(tmp_path, rules=RULES)
    out = ST.rewrite_text(ctx, "configs/registry/seed_x.yaml", BLOCK_HEADER_YAML)
    assert out.split("\n")[2] == "    contract: >   # checked on cluster A"
    assert yaml.safe_load(out)["rows"][0]["contract"] == yaml.safe_load(BLOCK_HEADER_YAML)["rows"][0]["contract"]
    e = _scan_files(make_ctx(tmp_path), {"configs/registry/seed_x.yaml": "rows:\n  - id: a\n    contract: >   # added by Wrangler9\n"
                                                                          "      a body\n"})
    assert not e["ok"] and any(f.startswith("comments:seed_x.yaml:3") for f in e["failures"]), e["failures"]


def test_a_rewritten_block_scalar_keeps_its_header_comment(tmp_path):
    """A block scalar the policy rewrites keeps the comment on its header line."""
    ctx = make_ctx(tmp_path, rules=RULES)
    out = ST.rewrite_text(ctx, "configs/registry/seed_x.yaml", BLOCK_HEADER_YAML)
    assert yaml.safe_load(out)["rows"][0]["note"] == "Built on cluster A\n"
    assert "    note: >   # a header comment stays" in out.split("\n"), out


def test_decision_id_in_a_read_leaf_fails_unless_exempted(tmp_path):
    value = "score > 0.5 (D31)"
    text = f"rows:\n  - id: a\n    threshold_rule: \"{value}\"\n"
    inv = {"seeds": frozenset({"threshold_rule"})}
    ctx = make_ctx(tmp_path, inventory=inv)
    assert ctx.classify("seeds", "seed_x.yaml", "rows", (0, "threshold_rule"), value)[0] == "read"
    e = _scan_files(ctx, {"configs/registry/seed_x.yaml": text})
    assert not e["ok"] and any("threshold_rule" in f for f in e["failures"])
    pol = policy_dict(exempt=[["seeds:seed_x.yaml:rows:$[*].threshold_rule", ST.sha256_text(value), "an identity field"]])
    ctx2 = make_ctx(tmp_path, policy=pol, inventory=inv)
    assert _scan_files(ctx2, {"configs/registry/seed_x.yaml": text})["ok"]


# --------------------------------------------------------------------------------------------------------------------------------
# check 1 and the masked comparison
# --------------------------------------------------------------------------------------------------------------------------------


def _git(d, *args):
    subprocess.run(["git", "-C", str(d), *args], check=True, capture_output=True)


def test_check_files_accepts_the_rewrite_and_refuses_an_undeclared_edit(tmp_path):
    repo = tmp_path / "repo"
    reg = repo / "configs" / "registry"
    reg.mkdir(parents=True)
    (reg / "seed_x.yaml").write_text(YAML_IN)
    _git(repo, "init", "-q")
    _git(repo, "-c", "user.email=t@example.com", "-c", "user.name=t", "add", ".")
    _git(repo, "-c", "user.email=t@example.com", "-c", "user.name=t", "commit", "-q", "-m", "pre")
    ctx = make_ctx(tmp_path, rules=RULES)
    (reg / "seed_x.yaml").write_text(ST.rewrite_text(ctx, "configs/registry/seed_x.yaml", YAML_IN))
    rep = CST.Report()
    CST.check_files(ctx, str(repo), "HEAD", [], None, rep)
    assert rep.ok, rep.checks
    good = (reg / "seed_x.yaml").read_text()
    (reg / "seed_x.yaml").write_text(good.replace("kept as is", "kept as was"))     # an edit no rule or row declares
    rep = CST.Report()
    CST.check_files(ctx, str(repo), "HEAD", [], None, rep)
    assert not rep.ok and "rows:$[0].shared.notes" in json.dumps(rep.checks)
    (reg / "seed_x.yaml").write_text(good.replace("n: 3", "n: 4"))                  # a number moved
    rep = CST.Report()
    CST.check_files(ctx, str(repo), "HEAD", [], None, rep)
    assert not rep.ok and "n: 4" in json.dumps(rep.checks)


def test_masked_diff_explains_declared_text_only():
    subs = CST.Substitutions(exact={"by Wrangler9 (D3)": ["by the authors"]})
    a = {"ruling_id": "r1", "reason": "by Wrangler9 (D3)", "note": "source: by Wrangler9 (D3)", "n": 1}
    b = {"ruling_id": "r1", "reason": "by the authors", "note": "source: by the authors", "n": 1}
    assert CST.masked_diff(a, b, subs) == []
    assert CST.masked_diff(a, dict(b, ruling_id="r2"), subs)
    assert CST.masked_diff(a, dict(b, n=2), subs)


def test_a_verbatim_read_leaf_keeps_its_bytes_even_under_the_site_map(tmp_path):
    """A value a loader matches against bytes recorded outside the release (a producer's spelling, a path it opens) is declared
    `verbatim`: not even the site map touches it, while an ordinary read leaf still takes the site map."""
    pol = policy_dict()
    pol["classes"].insert(0, ["seeds:*:*:$[*].spelling", "read", "verbatim"])
    ctx = make_ctx(tmp_path, policy=pol, inventory={"seeds": frozenset({"spelling", "label"})})
    assert ctx.transform("seeds", "s.yaml", "rows", (0, "spelling"), "/data/oldsite/model") == ("read", "/data/oldsite/model")
    assert ctx.transform("seeds", "s.yaml", "rows", (0, "label"), "oldsite_model") == ("read", "cluster_a_model")


# --------------------------------------------------------------------------------------------------------------------------------
# spelled-out numbered decisions are attribution too
# --------------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("text", ["Decision 25: item rows store every draw", "kept apart (spec fixed decision 20)",
                                  "see decisions 11 / 20", "decision #7 says so", "DECISION 3"])
def test_the_scanner_flags_a_spelled_out_numbered_decision(text):
    assert ("decision_spelled", ) == tuple(k for k, _ in ST.Scanner().hits(text) if k == "decision_spelled"), text


@pytest.mark.parametrize("text", ["the decision to run no pad-0 condition is the authors' own",
                                  "a.pasttense.decisions.8192", "the decode decision for this row"])
def test_the_scanner_leaves_decision_prose_and_dotted_keys_alone(text):
    assert not ST.Scanner().any(text), text


def test_planted_spelled_out_decision_fails_in_a_seed_value_and_a_yaml_comment(tmp_path):
    ctx = make_ctx(tmp_path)
    e = _scan_files(ctx, {"configs/registry/seed_x.yaml": "rows:\n  - id: a\n    notes: \"Decision 25: rows store every draw\"\n"})
    assert not e["ok"] and any("seeds:seed_x.yaml:rows:$[*].notes" in f and "decision_spelled" in f for f in e["failures"])
    e = _scan_files(ctx, {"configs/registry/seed_x.yaml": "rows: []   # admitted with a reason (spec fixed decision 19)\n"})
    assert not e["ok"] and any(f.startswith("comments:seed_x.yaml:1") and "decision_spelled" in f for f in e["failures"])


def test_no_shipped_registry_file_cites_a_numbered_decision():
    """Every free-text leaf and YAML comment of the committed configs/registry carries no spelled-out decision number."""
    root = os.path.join(os.path.dirname(HERE), "configs", "registry")
    rx = ST.Scanner().shapes["decision_spelled"]
    found = []
    for rel in ST.registry_files(os.path.dirname(HERE)):
        with open(os.path.join(os.path.dirname(HERE), rel), "r", encoding="utf-8") as fh:
            text = fh.read()
        for scope, unit, column, path, value in ST.file_leaves(rel, text):
            if rx.search(value):
                found.append(ST.address(scope, unit, column, path, False))
        for n, line, off in ST.yaml_comment_lines(rel, text):
            if rx.search(line[off:]):
                found.append(f"{rel}:{n}")
    assert os.path.isdir(root) and not found, found


# --------------------------------------------------------------------------------------------------------------------------------
# a declared key map: the keys of one JSON object go through the identifier site map
# --------------------------------------------------------------------------------------------------------------------------------


def test_a_declared_key_map_renames_its_keys_through_the_site_map_and_keeps_the_file_sorted(tmp_path):
    pol = policy_dict(keys=["seeds:spellings.json:model_spellings:$"])
    ctx = make_ctx(tmp_path, policy=pol)
    data = {"model_spellings": {"org/repo": "m1", "/data/oldsite/models/m2": "m2"}, "notes": {"/data/oldsite/x": "kept"}}
    text = json.dumps(data, indent=1, sort_keys=True) + "\n"
    changes = []
    out = ST.rewrite_text(ctx, "configs/registry/spellings.json", text, changes)
    got = json.loads(out)
    assert got["model_spellings"] == {"org/repo": "m1", "/data/cluster_a/models/m2": "m2"}
    assert got["notes"] == {"/data/oldsite/x": "kept"}             # an undeclared object keeps its keys
    assert out == json.dumps(got, indent=1, sort_keys=True) + "\n"
    assert any(c.cls == "key" and c.old == "/data/oldsite/models/m2" and c.new == "/data/cluster_a/models/m2"
               and c.addr == "seeds:spellings.json:model_spellings:$<key>" for c in changes)
    clash = json.dumps({"model_spellings": {"oldsite": "a", "cluster_a": "b"}}, indent=1, sort_keys=True) + "\n"
    with pytest.raises(ST.ShippedTextError):
        ST.rewrite_text(ctx, "configs/registry/spellings.json", clash)


# --------------------------------------------------------------------------------------------------------------------------------
# check 4 compares every shipped Parquet with the SQLite, and check 5 scans the Parquet text
# --------------------------------------------------------------------------------------------------------------------------------


def _write_frame_dir(d, sqlite_path, snapshot_id, **extra):
    import gzip
    import hashlib
    import pyarrow as pa
    import pyarrow.parquet as pq
    os.makedirs(d, exist_ok=True)
    with open(sqlite_path, "rb") as fh:
        raw = fh.read()
    part = "paper_frame.sqlite.gz.00"
    with open(os.path.join(d, part), "wb") as fh:
        fh.write(gzip.compress(raw, mtime=0))
    conn = sqlite3.connect(sqlite_path)
    tables = {}
    for t in ST.frame_tables(conn):
        cols = ST.frame_columns(conn, t)
        rows = conn.execute(f'SELECT * FROM "{t}" ORDER BY rowid').fetchall()
        pq.write_table(pa.table({c: pa.array([r[i] for r in rows], type=pa.string()) for i, c in enumerate(cols)}),
                       os.path.join(d, f"{t}.parquet"))
        tables[t] = len(rows)
    conn.close()
    files = {fn: {"sha256": ST.sha256_file(os.path.join(d, fn)), "bytes": os.path.getsize(os.path.join(d, fn))}
             for fn in sorted(os.listdir(d))}
    m = {"snapshot_id": snapshot_id, "ingest_seq": 7, "sqlite_parts": [part], "sqlite_sha256": hashlib.sha256(raw).hexdigest(),
         "tables": tables, "parquet_tables": sorted(tables), "files": files}
    m.update(extra)
    with open(os.path.join(d, "MANIFEST.json"), "w", encoding="utf-8") as fh:
        json.dump(m, fh, indent=1, sort_keys=True)
    return m


def _text_frame(path, note):
    c = sqlite3.connect(path)
    c.executescript("CREATE TABLE run (run_id TEXT, note TEXT); CREATE TABLE judge (judge_id TEXT, notes TEXT);")
    c.executemany("INSERT INTO run VALUES (?, ?)", [("r1", note), ("r2", "plain note")])
    c.execute("INSERT INTO judge VALUES ('j1', 'a neutral judge')")
    c.commit()
    c.close()


def _frame_repo(tmp_path, ctx):
    """A checkout whose HEAD holds the pre frame, and the post SQLite the policy derives from it."""
    repo = tmp_path / "repo"
    snaps = repo / "paper" / "snapshot" / "snapshots"
    snaps.mkdir(parents=True)
    pre_sql, post_sql = str(tmp_path / "pre.sqlite"), str(tmp_path / "post.sqlite")
    _text_frame(pre_sql, "staged on Oldsite")
    pre_m = _write_frame_dir(str(snaps / "000000000007-pre"), pre_sql, "000000000007-pre")
    (repo / "paper" / "snapshot" / "CURRENT").write_text("000000000007-pre\n")
    _git(repo, "init", "-q")
    _git(repo, "-c", "user.email=t@example.com", "-c", "user.name=t", "add", ".")
    _git(repo, "-c", "user.email=t@example.com", "-c", "user.name=t", "commit", "-q", "-m", "pre")
    ST.rewrite_frame(ctx, pre_sql, post_sql)
    return repo, snaps, pre_m, post_sql


def _check4(ctx, repo, tmp_path):
    rep = CST.Report()
    CST.check_frame(ctx, str(repo), "HEAD", str(tmp_path / "work"), None, rep)
    return rep.checks["4_frame"]


def test_check_4_accepts_a_frame_whose_parquet_follows_its_sqlite(tmp_path):
    pytest.importorskip("pyarrow")
    ctx = make_ctx(tmp_path, rules=RULES)
    repo, snaps, pre_m, post_sql = _frame_repo(tmp_path, ctx)
    _write_frame_dir(str(snaps / "000000000007-post"), post_sql, "000000000007-post", derived_from=pre_m["snapshot_id"],
                     release_changed_rows={"run": {"note": 1}})
    (repo / "paper" / "snapshot" / "CURRENT").write_text("000000000007-post\n")
    e = _check4(ctx, repo, tmp_path)
    assert e["ok"], e["failures"]


def test_a_stale_parquet_fails_check_4_even_when_the_manifest_omits_its_table(tmp_path):
    """The rewrite moved a cell of `run`, the MANIFEST's release_changed_rows leaves `run` out and the pre-image's run.parquet is
    shipped unchanged: the Parquet still carries the old text, and check 4 must say so whatever the MANIFEST claims."""
    pytest.importorskip("pyarrow")
    import shutil
    ctx = make_ctx(tmp_path, rules=RULES)
    repo, snaps, pre_m, post_sql = _frame_repo(tmp_path, ctx)
    post = snaps / "000000000007-post"
    _write_frame_dir(str(post), post_sql, "000000000007-post", derived_from=pre_m["snapshot_id"], release_changed_rows={})
    shutil.copyfile(snaps / "000000000007-pre" / "run.parquet", post / "run.parquet")
    m = json.loads((post / "MANIFEST.json").read_text())
    m["files"]["run.parquet"] = pre_m["files"]["run.parquet"]
    (post / "MANIFEST.json").write_text(json.dumps(m, indent=1, sort_keys=True))
    (repo / "paper" / "snapshot" / "CURRENT").write_text("000000000007-post\n")
    e = _check4(ctx, repo, tmp_path)
    assert not e["ok"] and any("run.parquet" in f for f in e["failures"]), e["failures"]


def _post_frame(tmp_path, ctx):
    """The pre frame committed and a correct post frame beside it, CURRENT pointing at the post."""
    repo, snaps, pre_m, post_sql = _frame_repo(tmp_path, ctx)
    post = snaps / "000000000007-post"
    _write_frame_dir(str(post), post_sql, "000000000007-post", derived_from=pre_m["snapshot_id"],
                     release_changed_rows={"run": {"note": 1}})
    (repo / "paper" / "snapshot" / "CURRENT").write_text("000000000007-post\n")
    return repo, post, post_sql


def _restamp(post, fn):
    m = json.loads((post / "MANIFEST.json").read_text())
    m["files"][fn] = {"sha256": ST.sha256_file(str(post / fn)), "bytes": os.path.getsize(post / fn)}
    (post / "MANIFEST.json").write_text(json.dumps(m, indent=1, sort_keys=True))


def test_a_parquet_rebuilt_without_a_column_fails_check_4_even_with_its_manifest_hash_updated(tmp_path):
    """run.parquet is rebuilt from the new SQLite without its `note` column and the MANIFEST carries the new file's hash: every
    surviving cell matches, but a Parquet reader would lose the column, so check 4 fails."""
    pytest.importorskip("pyarrow")
    import pyarrow as pa
    import pyarrow.parquet as pq
    ctx = make_ctx(tmp_path, rules=RULES)
    repo, post, post_sql = _post_frame(tmp_path, ctx)
    c = sqlite3.connect(post_sql)
    ids = [r[0] for r in c.execute("SELECT run_id FROM run ORDER BY rowid")]
    c.close()
    pq.write_table(pa.table({"run_id": pa.array(ids, type=pa.string())}), post / "run.parquet")
    _restamp(post, "run.parquet")
    e = _check4(ctx, repo, tmp_path)
    assert not e["ok"] and any("run.parquet" in f and "missing ['note']" in f for f in e["failures"]), e["failures"]


def test_a_parquet_file_dropped_with_its_manifest_entries_fails_check_4(tmp_path):
    """judge.parquet is removed together with its `files` and `parquet_tables` entries, so the directory and its MANIFEST agree:
    the pre-ref frame shipped that table as Parquet, and check 4 fails."""
    pytest.importorskip("pyarrow")
    ctx = make_ctx(tmp_path, rules=RULES)
    repo, post, _ = _post_frame(tmp_path, ctx)
    os.remove(post / "judge.parquet")
    m = json.loads((post / "MANIFEST.json").read_text())
    del m["files"]["judge.parquet"]
    m["parquet_tables"] = [t for t in m["parquet_tables"] if t != "judge"]
    (post / "MANIFEST.json").write_text(json.dumps(m, indent=1, sort_keys=True))
    e = _check4(ctx, repo, tmp_path)
    assert not e["ok"] and any("parquet_tables" in f and "judge" in f for f in e["failures"]), e["failures"]


@pytest.mark.parametrize("columns, want", [
    ({"run_id": ("string", ["r1", "r2"]), "note": ("string", ["n1", "n2"]), "n": ("int64", [1, 2])}, None),
    ({"run_id": ("string", ["r1", "r2"]), "n": ("int64", [1, 2])}, "missing ['note']"),
    ({"note": ("string", ["n1", "n2"]), "run_id": ("string", ["r1", "r2"]), "n": ("int64", [1, 2])}, "order"),
    ({"run_id": ("string", ["r1", "r2"]), "note": ("string", ["n1", "n2"]), "n": ("double", [1.0, 2.0])}, "column n is double"),
    ({"run_id": ("string", ["r1", "r2"]), "note": ("string", ["n1", "n2"]), "n": ("int64", [1, 2]),
      "extra": ("string", ["x", "y"])}, "extra ['extra']"),
])
def test_the_parquet_comparison_reads_the_schema_against_the_sqlite_table(tmp_path, columns, want):
    """Equal cells are not enough: a Parquet file whose columns are missing, reordered, extra or of another type than the one the
    exporter writes for the declared type is not the SQLite table."""
    pytest.importorskip("pyarrow")
    import pyarrow as pa
    import pyarrow.parquet as pq
    b = sqlite3.connect(":memory:")
    b.executescript("CREATE TABLE run (run_id TEXT, note TEXT, n INTEGER); INSERT INTO run VALUES ('r1', 'n1', 1), ('r2', 'n2', 2);")
    p = str(tmp_path / "run.parquet")
    types = {"string": pa.string(), "int64": pa.int64(), "double": pa.float64()}
    pq.write_table(pa.table({c: pa.array(v, type=types[ty]) for c, (ty, v) in columns.items()}), p)
    got = CST._parquet_differs(b, "run", p)
    if want is None:
        assert got is None
    else:
        assert got is not None and want in got, got


def test_planted_attribution_in_a_parquet_cell_fails_check_5(tmp_path):
    """The SQLite is clean; one Parquet cell carries an attribution the SQLite does not: check 5 reads the Parquet text too."""
    pytest.importorskip("pyarrow")
    import pyarrow as pa
    import pyarrow.parquet as pq
    ctx = make_ctx(tmp_path)
    repo = tmp_path / "repo"
    d = repo / "paper" / "snapshot" / "snapshots" / "000000000007-x"
    sql = str(tmp_path / "f.sqlite")
    _text_frame(sql, "a neutral note")
    _write_frame_dir(str(d), sql, "000000000007-x")
    pq.write_table(pa.table({"run_id": ["r1", "r2"], "note": ["kept per the user's ruling", "plain note"]}), d / "run.parquet")
    (repo / "paper" / "snapshot" / "CURRENT").write_text("000000000007-x\n")
    (repo / "configs" / "registry").mkdir(parents=True)
    rep = CST.Report()
    CST.check_scan(ctx, str(repo), sql, rep)
    e = rep.checks["5_scan"]
    assert not e["ok"] and any("frame:run:note:$" in f for f in e["failures"]), e["failures"]


FAKE_SCAN_TOOL = '''
import pathlib
TABLE_FILE = "tests/front_door_testlib.py"
ALLOWLIST = "reproduce/tools/scan_allowlist.tsv"


class Hit:
    def __init__(self, path, where, tier, unit):
        self.path, self.where, self.tier, self.unit = path, where, tier, unit

    def excerpt(self):
        return self.unit


class Report:
    def __init__(self):
        self.hits, self.errors = [], []


class Scanner:
    def __init__(self):
        self.report = Report()


def load_table(path):
    return ()


def scan(root, roots=(), scopes=(), table=None, jobs=1):
    sc = Scanner()
    assert tuple(scopes) == ("leaf",) and "configs/registry" in roots
    for p in sorted(pathlib.Path(root, "configs", "registry").glob("*.yaml")):
        if "handle-x" in p.read_text():
            sc.report.hits.append(Hit("configs/registry/" + p.name, "rows[].notes", "personal", "handle-x"))
    return sc, list(roots)


def load_allowlist(path):
    path = pathlib.Path(path)
    return [ln.split("\\t") for ln in path.read_text().splitlines() if ln.strip()] if path.exists() else []


def apply_allowlist(hits, allow):
    return [h for h in hits if not any(row[0] == h.path for row in allow)]
'''


def test_check_5_also_fails_on_a_publication_scanner_hit_outside_its_allowlist(tmp_path):
    """A token only the publication scanner knows (a stand-in for a personal handle) fails check 5 when the scanner is given,
    and passes once its allowlist covers it; the field-policy scan alone does not see it."""
    tool = tmp_path / "scan_tool.py"
    tool.write_text(FAKE_SCAN_TOOL)
    repo = tmp_path / "repo"
    (repo / "configs" / "registry").mkdir(parents=True)
    (repo / "configs" / "registry" / "seed_x.yaml").write_text("rows:\n  - id: a\n    notes: staged by handle-x\n")
    ctx = make_ctx(tmp_path)
    rep = CST.Report()
    CST.check_scan(ctx, str(repo), None, rep)
    assert rep.checks["5_scan"]["ok"]
    rep = CST.Report()
    CST.check_scan(ctx, str(repo), None, rep, scan_tool=str(tool))
    e = rep.checks["5_scan"]
    assert not e["ok"] and any("scan tool: configs/registry/seed_x.yaml#rows[].notes: personal" in f for f in e["failures"])
    allow = tmp_path / "allow.tsv"
    allow.write_text("configs/registry/seed_x.yaml\tpersonal\tvalue:" + "0" * 64 + "\treviewed\n")
    rep = CST.Report()
    CST.check_scan(ctx, str(repo), None, rep, scan_tool=str(tool), scan_allowlist=str(allow))
    assert rep.checks["5_scan"]["ok"], rep.checks["5_scan"]["failures"]


# --------------------------------------------------------------------------------------------------------------------------------
# keys are text too: a declared key map renames them, the checks follow the renames, and check 5 scans every key
# --------------------------------------------------------------------------------------------------------------------------------


def test_a_declared_key_map_renames_named_keys_and_the_masked_comparison_follows_it(tmp_path):
    pol = policy_dict(key_map=[["transcripts:rulings_r2_v*.jsonl#header:added_after_the_draft:$", "D29", "first_addition"],
                               ["transcripts:rulings_r2_v*.jsonl#header:added_after_the_draft:$", "D29_note", "first_addition_note"]])
    ctx = make_ctx(tmp_path, policy=pol)
    head = {"_header": True, "added_after_the_draft": {"D29": ["r1"], "D29_note": "a neutral note", "kept": ["r2"]}}
    text = json.dumps(head, ensure_ascii=False) + "\n"
    changes = []
    out = ST.rewrite_text(ctx, "configs/registry/rulings_r2_v1.jsonl", text, changes)
    got = json.loads(out.split("\n")[0])
    assert got["added_after_the_draft"] == {"first_addition": ["r1"], "first_addition_note": "a neutral note", "kept": ["r2"]}
    assert list(got["added_after_the_draft"]) == ["first_addition", "first_addition_note", "kept"]     # order kept
    subs = CST.Substitutions()
    for c in changes:
        subs.add(c)
    assert CST.masked_diff(head, got, subs) == []
    assert CST.masked_diff(head, dict(got, added_after_the_draft=dict(got["added_after_the_draft"], kept=["r3"])), subs)
    other = {"_header": True, "added_after_the_draft": {"D29": ["r1"]}, "elsewhere": {"D29": 1}}
    out2 = json.loads(ST.rewrite_text(ctx, "configs/registry/rulings_r2_v1.jsonl", json.dumps(other) + "\n").split("\n")[0])
    assert out2["elsewhere"] == {"D29": 1}                          # only the declared object's keys move



class _IdentityCodemod:
    """Stands in for the codemod at integration: it changes nothing, but check 1 then pairs leaves through its codemod path."""

    @staticmethod
    def rewrite_text(text, rel):
        return text


def test_a_renamed_key_is_not_recorded_as_dropped_on_a_codemodded_tree(tmp_path):
    """On a codemodded tree, check 1 also records every leaf the rewrite changed in its codemodded form. A leaf under a key the
    policy renamed is absent from the rewrite at its old address; recording it as dropped put the old key among the dropped keys
    the masked comparison skips, so check 2 reported the renamed key as added."""
    pol = policy_dict(key_map=[["transcripts:rulings_r2_v*.jsonl#header:added_after_the_draft:$", "D29", "first_addition"]])
    ctx = make_ctx(tmp_path, policy=pol)
    head = {"_header": True, "added_after_the_draft": {"D29": ["r1", "r2"], "kept": ["r3"]}}
    pre = json.dumps(head, ensure_ascii=False) + "\n"
    repo = tmp_path / "repo"
    reg = repo / "configs" / "registry"
    reg.mkdir(parents=True)
    (reg / "rulings_r2_v1.jsonl").write_text(pre)
    _git(repo, "init", "-q")
    _git(repo, "-c", "user.email=t@example.com", "-c", "user.name=t", "add", ".")
    _git(repo, "-c", "user.email=t@example.com", "-c", "user.name=t", "commit", "-q", "-m", "pre")
    (reg / "rulings_r2_v1.jsonl").write_text(ST.rewrite_text(ctx, "configs/registry/rulings_r2_v1.jsonl", pre))
    for codemod in (None, _IdentityCodemod()):
        rep = CST.Report()
        fs = CST.check_files(ctx, str(repo), "HEAD", [], codemod, rep)
        assert rep.ok, rep.checks
        subs = CST.subs_for(fs, "transcripts")
        assert "D29" not in subs.dropped_keys
        before = json.loads(fs.before["configs/registry/rulings_r2_v1.jsonl"].split("\n")[0])
        after = json.loads(fs.after["configs/registry/rulings_r2_v1.jsonl"].split("\n")[0])
        assert after["added_after_the_draft"] == {"first_addition": ["r1", "r2"], "kept": ["r3"]}
        assert CST.masked_diff(before, after, subs) == []

def test_planted_decision_id_in_a_key_fails_the_scan_in_every_scope(tmp_path):
    ctx = make_ctx(tmp_path)
    rec = {"ruling_id": "r1", "target_json": {"D555": ["r2"]}}
    e = _scan_files(ctx, {"configs/registry/rulings_r2_v1.jsonl": json.dumps(rec) + "\n"})
    assert not e["ok"] and any("transcripts:rulings_r2_v1.jsonl:target_json:$<key>" in f for f in e["failures"]), e["failures"]
    e = _scan_files(ctx, {"configs/registry/seed_x.yaml": "rows:\n  - id: a\n    D555: kept\n"})
    assert not e["ok"] and any("seeds:seed_x.yaml:rows:$[*]<key>" in f for f in e["failures"]), e["failures"]
    p = str(tmp_path / "f.sqlite")
    _clean_frame(p, extra={"by_wave": {"W27": "x"}})
    e = _scan_frame(ctx, p)
    assert not e["ok"] and any("frame:run:extra_json:$.by_wave<key>" in f for f in e["failures"]), e["failures"]
    pol = policy_dict(exempt=[["frame:run:extra_json:$.by_wave<key>", ST.sha256_text("W27"), "a machine key"]])
    assert _scan_frame(make_ctx(tmp_path, policy=pol), p)["ok"]
