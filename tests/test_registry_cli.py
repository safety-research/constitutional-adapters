"""The `registry` console script: every working subcommand runs against a synthetic registry; every stub
prints its contract and exits 3; the help text parses; `--root` / REGISTRY_ROOT resolve.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

from sisteer.registry import cli, emit
from sisteer.registry import harvest as H
from sisteer.registry import synth


@pytest.fixture
def root(tmp_path, monkeypatch):
    root = str(tmp_path / "reg")
    monkeypatch.setenv("REGISTRY_ROOT", root)
    assert cli.main(["init"]) == 0
    synth.write_corpus(root, n_runs=2, n_items=5)
    assert cli.main(["ingest", "--workers", "0"]) == 0
    return root


def test_init_is_idempotent_and_layout(root, capsys):
    assert cli.main(["init"]) == 0
    assert "exists" in capsys.readouterr().out
    for d in ("records", "inbox", "state", "rulings", "manifests", "snapshots"):
        assert os.path.isdir(os.path.join(root, d))


def test_status_text_and_json(root, capsys):
    assert cli.main(["status"]) == 0
    out = capsys.readouterr().out
    from sisteer.registry.migrations import code_version
    assert "run: 2" in out and f"schema v{code_version()}" in out and "SCHEMA DRIFT" not in out
    assert cli.main(["status", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["table_counts"]["run"] == 2 and "timings_ms" in data


def test_status_html_writes_page(root, tmp_path, capsys):
    out = str(tmp_path / "page.html")
    assert cli.main(["status", "--html", "--out", out]) == 0
    assert os.path.exists(out) and "Results registry" in open(out).read()


def test_query_and_frame(root, capsys, tmp_path):
    assert cli.main(["query", "SELECT COUNT(*) AS n FROM run"]) == 0
    assert capsys.readouterr().out.strip().splitlines()[-1] == "2"
    assert cli.main(["query", "SELECT run_id FROM run", "--json"]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 2
    assert cli.main(["frame", "v_run_wide"]) == 0
    assert "lifecycle_state" in capsys.readouterr().out
    pq = str(tmp_path / "f.parquet")
    assert cli.main(["frame", "item_result", "--where", "trial_index = 0", "--parquet", pq]) == 0
    assert os.path.exists(pq)


def test_history_run_and_item(root, capsys):
    from sisteer.registry import db as DB
    conn = DB.connect(DB.db_path(root), readonly=True)
    rid = conn.execute("SELECT run_id FROM run LIMIT 1").fetchone()[0]
    jid = conn.execute("SELECT judgment_id FROM judgment WHERE run_id = ?", (rid,)).fetchone()[0]
    conn.close()
    assert cli.main(["history", "run", json.dumps({"run_id": rid})]) == 0
    out = capsys.readouterr().out
    assert "run_state" in out and "ingested" in out
    key = {"run_id": rid, "item_id": "q00000", "metric_id": "asr", "trial_index": 0, "sample_index": 0, "turn_index": -1, "judgment_id": jid}
    assert cli.main(["history", "item_result", json.dumps(key), "--json"]) == 0
    states = json.loads(capsys.readouterr().out)
    assert len(states) == 1 and states[0]["item_id"] == "q00000"


def test_rebuild_verify_snapshot_identity_blocked(root, tmp_path, capsys):
    out = str(tmp_path / "rebuilt.sqlite")
    assert cli.main(["rebuild", "--from", root, "--into", out, "--verify", "--workers", "0"]) == 0
    assert "0 differing rows" in capsys.readouterr().out
    assert cli.main(["snapshot", "--no-space-check"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert os.path.isdir(info["dir"]) and open(os.path.join(root, "CURRENT")).read().strip() == info["snapshot_id"]
    # --push with no destinations configured builds locally, says so, and exits 0
    assert cli.main(["snapshot", "--no-space-check", "--push"]) == 0
    assert "no snapshot destinations configured" in capsys.readouterr().err
    assert cli.main(["identity", "resolve", "synthstore_000"]) == 0
    assert "synthstore_000" in capsys.readouterr().out
    assert cli.main(["blocked", "list"]) == 0
    assert "(no rows)" in capsys.readouterr().out


def test_pending_judgment_list_claim_release(root, capsys):
    assert cli.main(["pending-judgment", "--judge", synth.JUDGE_ID_2, "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 2
    rid = rows[0]["run_id"]
    assert cli.main(["pending-judgment", "--claim", "--session", "s1", "--run-id", rid, "--judge", synth.JUDGE_ID_2]) == 0
    jid = capsys.readouterr().out.strip()
    assert cli.main(["pending-judgment", "--claim", "--session", "s2", "--run-id", rid, "--judge", synth.JUDGE_ID_2]) == 1
    assert capsys.readouterr().err.startswith("claim refused: ClaimHeld: "), "another session holds the claim: the refusal names its class first"
    assert cli.main(["pending-judgment", "--release", rid, "--session", "s1", "--judge", synth.JUDGE_ID_2]) == 0
    assert cli.main(["pending-judgment", "--claim"]) == 2


def test_mirrors_without_config_and_with_local_dest(root, tmp_path, capsys):
    assert cli.main(["mirrors", "verify"]) == 2
    with open(os.path.join(root, "config.json"), "w") as fh:
        json.dump({"mirrors": [{"name": "cluster_a", "kind": "local", "path": str(tmp_path / "m")}]}, fh)
    assert cli.main(["mirrors", "sync"]) == 0
    res = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert res["switched"] and res["missing"] == 0
    assert cli.main(["mirrors", "verify"]) == 0
    assert json.loads(capsys.readouterr().out)["lag_files"] == 0


def test_docs_regenerate_and_decode_policy_export(root, tmp_path, capsys):
    assert cli.main(["docs", "vocabulary"]) == 0
    out = str(tmp_path / "dp.json")
    assert cli.main(["decode-policy", "export", "--out", out]) == 0
    assert json.load(open(out))["export_version"] == 1
    assert cli.main(["decode-policy", "check", "--out", out]) == 0
    assert cli.main(["decode-policy", "check", "--out", str(tmp_path / "missing.json")]) == 1


def test_publish_figures_and_register_run(root, tmp_path, capsys, monkeypatch):
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)  # the suite runs with emission off (tests/conftest.py); this test writes to its own tmp spool, so it opts in
    figs = tmp_path / "figs"
    figs.mkdir()
    (figs / "a.png").write_bytes(b"png")
    assert cli.main(["publish-figures", "--dir", str(figs), "--notebook-commit", "d" * 40]) == 0
    assert json.loads(capsys.readouterr().out)["registered"] == 1
    plan = {k: v for k, v in synth.run_payload("cli_plan").items() if k != "run_id"}
    pf = tmp_path / "plan.json"
    pf.write_text(json.dumps(plan))
    spool = str(tmp_path / "spool")
    assert cli.main(["register-run", str(pf), "--spool", spool]) == 0
    assert len(capsys.readouterr().out.strip().splitlines()) == 2


def test_adapters_run_without_matching_adapter_is_a_stub(root):
    assert cli.main(["adapters", "run", "--adapter", "no_such_adapter"]) == cli.STUB_EXIT


def test_export_r2_compat_and_degeneracy_rebuild_run_on_an_empty_registry(root, tmp_path, capsys):
    out = str(tmp_path / "r2.parquet")
    assert cli.main(["export", "r2-compat", out]) == 0
    assert os.path.exists(out) and os.path.exists(out.replace(".parquet", "_schema.json"))
    assert cli.main(["degeneracy-rebuild", "--out", str(tmp_path / "deg"), "--csv-dir", str(tmp_path / "nocsv")]) == 0
    assert os.path.exists(os.path.join(str(tmp_path / "deg"), "degeneracy_rebuild_report.json"))


@pytest.mark.parametrize("argv", [["backfill"]])
def test_stubs_print_contract_and_exit_3(root, argv, capsys):
    assert cli.main(argv) == cli.STUB_EXIT
    err = capsys.readouterr().err
    assert "Contract:" in err and "registry " + argv[0] in err


@pytest.mark.parametrize("argv,module", [(["export", "r2-compat", "o.parquet"], "r2compat"),
                                         (["degeneracy-rebuild", "--out", "/x"], "degeneracy_artifact")])
def test_lazily_imported_commands_refuse_by_name_when_their_module_cannot_be_imported(root, argv, module, capsys, monkeypatch):
    """`export r2-compat` / `degeneracy-rebuild` import their modules lazily; when that import fails (here a ``None`` entry in
    ``sys.modules``) they exit 3 naming the module and the import error, and run nothing."""
    monkeypatch.setitem(sys.modules, f"sisteer.registry.{module}", None)
    assert cli.main(argv) == cli.STUB_EXIT
    err = capsys.readouterr().err
    assert f"registry: sisteer.registry.{module} cannot be imported in this checkout (" in err and err.rstrip().endswith("nothing to run"), err


def test_help_parses_for_every_subcommand():
    p = cli.build_parser()
    for action in p._subparsers._group_actions:  # type: ignore[union-attr]
        for name, sub in action.choices.items():
            assert sub.format_help()


def test_last_run_is_stamped_only_by_commands_that_resolve_a_root(root, capsys):
    """``state/last_run/<command>.json`` is the timers' health record. A command that never touches a registry —
    ``docs``, ``register-run --spool …`` — must not leave a stamp: with no ``--root`` / ``REGISTRY_ROOT`` the default
    is the operator's live ``~/registry``, and the live install found a pytest's ``register-run`` and a ``docs all``
    stamped there as if they were timers (2026-09-17)."""
    stamps = os.path.join(root, "state", "last_run")
    assert cli.main(["status"]) == 0
    assert os.path.exists(os.path.join(stamps, "status.json"))
    assert cli.main(["docs", "vocabulary"]) == 0
    assert not os.path.exists(os.path.join(stamps, "docs.json"))
    capsys.readouterr()


# --------------------------------------------------------------------------------------------
# schema drift is loud on every surface: ingest refuses (exit 3), status prints it, the page shows it, migrate plans it
# --------------------------------------------------------------------------------------------


def _drift_the_shape(root):
    """The live 2026-09-19 shape, reproduced: the object table's paper_label CHECK without 'difference_relu_distilled',
    at the code's migration version."""
    from sisteer.registry import db as DB
    from sisteer.registry import schema_def as S
    from sisteer.registry.migrations import code_version
    path = DB.db_path(root)
    conn = DB.connect(path)
    rows = [tuple(r) for r in conn.execute("SELECT * FROM object").fetchall()]
    cols = [r[1] for r in conn.execute("PRAGMA table_info(object)").fetchall()]
    ddl = S.render_create_table(S.TABLE_BY_NAME["object"], "object__old").replace("'difference_relu_distilled', ", "", 1)
    conn.execute("PRAGMA foreign_keys=OFF")
    conn.execute("PRAGMA legacy_alter_table=ON")
    conn.executescript("BEGIN;\n" + ddl + f"\nINSERT INTO object__old ({', '.join(cols)}) SELECT {', '.join(cols)} FROM object;\n"
                       "DROP TABLE object;\nALTER TABLE object__old RENAME TO object;\n"
                       + "\n".join(S.render_indexes(S.TABLE_BY_NAME["object"])) + "\nCOMMIT;")
    assert [tuple(r) for r in conn.execute("SELECT * FROM object").fetchall()] == rows
    assert conn.execute("SELECT COALESCE(MAX(version), 0) FROM schema_migration").fetchone()[0] == code_version()
    conn.close()


def test_drifted_shape_refuses_ingest_names_it_in_status_and_the_page_and_plans_the_migration(root, tmp_path, capsys, monkeypatch):
    _drift_the_shape(root)
    want = "table object: column paper_label CHECK (paper_label IN (...)) lacks 'difference_relu_distilled'"
    # ingest: nothing written, exit 3, the line on stderr; with --html the page is rendered once per distinct refusal
    page = str(tmp_path / "status" / "index.html")
    monkeypatch.setenv("REGISTRY_STATUS_HTML", page)
    assert cli.main(["ingest", "--workers", "0", "--html"]) == 3
    cap = capsys.readouterr()
    assert want in cap.err and "registry migrate --plan" in cap.err and page in cap.out
    html = open(page, encoding="utf-8").read()
    assert "Schema drift" in html and "difference_relu_distilled" in html
    assert cli.main(["ingest", "--workers", "0", "--html"]) == 3
    cap = capsys.readouterr()
    assert page not in cap.out, "the same refusal re-rendered the page"
    state = json.load(open(os.path.join(root, "state", "schema_refusal.json")))
    assert state["lines"] == [want]
    # status: the text and the JSON carry the same line
    assert cli.main(["status"]) == 0
    out = capsys.readouterr().out
    assert "SCHEMA DRIFT" in out and want in out
    assert cli.main(["status", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["schema_drift"] == [want]
    # migrate --plan: read-only, names the table rebuild and its row count; --sql renders the body; exit 1 while drift stands
    assert cli.main(["migrate", "--plan"]) == 1
    out = capsys.readouterr().out
    assert "pending: none" in out and want in out and "rebuild table object: 2 rows to copy" in out
    assert cli.main(["migrate", "--plan", "--sql"]) == 1
    out = capsys.readouterr().out
    assert "CREATE TABLE object__new (" in out and "ALTER TABLE object__new RENAME TO object;" in out and "'difference_relu_distilled'" in out
    # plain migrate: nothing pending, the shape still differs: exit 1 naming it
    assert cli.main(["migrate"]) == 1
    cap = capsys.readouterr()
    assert "applied nothing" in cap.out and want in cap.err and "registry migrate --plan --sql" in cap.err


def test_migrate_plan_on_a_current_database_is_quiet_and_exits_0(root, capsys):
    from sisteer.registry.migrations import code_version
    assert cli.main(["migrate", "--plan"]) == 0
    out = capsys.readouterr().out
    assert f"schema version {code_version()}; code {code_version()}; pending: none" in out and "shape: matches the code's schema" in out
    assert cli.main(["migrate"]) == 0
    assert "shape matches the code's schema" in capsys.readouterr().out

