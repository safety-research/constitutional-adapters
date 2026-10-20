"""Schema strictness.

Every CHECK list in the DDL equals the vocab tuple of the same name (parsed OUT OF the DDL text,
independently of the renderer); schema.sql equals its render and the migrations; every nullable column
carries a why-null reason; out-of-vocabulary inserts fail naming the column; a judged run needs a
judgment; sentinel-column FKs hold by trigger; the fit_target_kind pairing CHECK admits one valid row
per kind and rejects every forbidden extra field.
"""
from __future__ import annotations

import os
import re
import sqlite3

import pytest

from sisteer.registry import db as DB
from sisteer.registry import schema_def as S
from sisteer.registry import synth
from sisteer.registry import vocab as V
from sisteer.registry.migrations import apply_migrations, code_version

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(os.path.dirname(HERE), "src", "sisteer", "registry")
SCHEMA_SQL = os.path.join(PKG, "schema.sql")


def _fresh() -> sqlite3.Connection:
    conn = DB.connect_memory()
    apply_migrations(conn)
    return conn


def _norm_master(conn) -> dict:
    """The comparable shape: ``db.ddl_master`` (comments, whitespace and identifier quoting normalised — a table a
    migration rebuilt is stored as ``CREATE TABLE "name"`` after its RENAME)."""
    return DB.ddl_master(conn)


def test_schema_sql_is_current_render():
    with open(SCHEMA_SQL, "r", encoding="utf-8") as fh:
        assert fh.read() == S.render_schema_sql(), "schema.sql is stale: run `registry docs schema`"


def test_schema_sql_equals_migrations():
    """The executable history (0001 + every later migration) produces the code's current shape (schema.sql, rendered
    from schema_def). A DDL change — a vocabulary value, a column, a CHECK — without a migration fails here."""
    a = _fresh()
    b = DB.connect_memory()
    with open(SCHEMA_SQL, "r", encoding="utf-8") as fh:
        b.executescript(fh.read())
    ma, mb = _norm_master(a), _norm_master(b)
    only_a = sorted(set(ma) - set(mb))
    only_b = sorted(set(mb) - set(ma))
    differing = sorted(k for k in set(ma) & set(mb) if ma[k] != mb[k])
    assert ma == mb, ("schema.sql (the code's DDL) differs from 0001 + the migrations: add a migration "
                      "(`registry migrate --plan --sql` against a database at the previous shape renders it; migrations/README.md). "
                      f"only in migrations: {only_a}; only in schema.sql: {only_b}; differing: {differing}")
    assert DB.schema_drift(a) == [] and DB.schema_drift(b) == []


#: sha256 of every landed migration file. A landed file is history and never changes: a schema change is a NEW
#: NNNN_<slug>.sql (migrations/README.md), whose sha256 is added here when it lands. 0001 is frozen at its
#: 2026-09-19 bytes (main 91685738): the in-place edits before that left the live database at an older shape than the
#: code at the same version — twice. Before the repository was published, 0001-0031 were edited in place ONCE (neutral
#: site labels; comments without internal process references) and re-pinned here: migrations/README.md says why that is
#: safe for a database built from either spelling.
FROZEN_MIGRATIONS = {
    "0001_initial.sql": "aa5fe955a075e6377dadaa30311d079b0fb5fdd361dd8a12c223424e5069f545",
    "0002_object_paper_label_source_status_view.sql": "356eebfd475d9b741e9ad80142d04dc9268a05a6d7f047ca11b694609ff7fc9c",
    "0003_pending_judgment_judge_scope.sql": "1821971f612108f79c2f1c751694929ba6c015659a9d91740d066a3b06e4f71e",
    "0004_paper_long_metric_role.sql": "bd73a164b291103a91eb39b0fc4cfa536f58dd6dab9c1e1ca6c3f86e704889d9",
    "0005_multi_source_run_pairing.sql": "8b78c6b600a5a17717be2a1d6b3af570b78b8efdfab9655c6f726b78ea6fa7dd",
    "0006_completion_flag_contrast_cap.sql": "3e5194cf69296ab6a5d38918ed443c7461f23dad87937a45410622fce8ce96e8",
    "0007_judgment_coverage_counts_errors.sql": "0f777c8910cddd4be8be57aa5a1a1cfc74ee69d222a6045ce819531a15d0d25d",
    "0008_item_headline_accepted_metric_ids.sql": "5f38dd398dbd19e8caa88bc6895178262aef38292ea8094c5002184bd2eb0407",
    "0009_paper_views_decode_family_default.sql": "bb12e6a27b719b63f8b7abaf376f4908b7c8f26426ea96fb29c0d4503d1fb668",
    "0010_paper_views_degeneration_from_run_degeneration.sql": "00643ae06d00068a78112368909c3ceadfb6280b8fc306906ebc245baf2755c7",
    "0011_roster_match_operating_point_fallback.sql": "5329474d6a209a636b7c38a488b197c30bc72c49438370e84745b2afaf40b457",
    "0012_instrument_padding_identity.sql": "862f154758232df010ee24bb249082319e578363d277f8b7a60ead1dc447adab",
    "0013_roster_match_padded_instrument_by_name.sql": "d1be4b74c67175d15404f725509b50085717b73efec12130b4746205b4d6f08d",
    "0014_instrument_padding_zero_point.sql": "22ab117014e445f1f7e3ad05b243fe6bd83792764529f81d4bf2eb2fb32edb7d",
    "0015_paper_inputs_head_base_tables.sql": "1493e1152553d93d77f769a1961ea9a0b6b8d13ce1d638e30a886ef396e135de",
    "0016_roster_version_scope.sql": "a3b2ed713a05f9eaa511d88cecca4ad9198125ce35949cf9df83c0a10ef69ca7",
    "0017_paper_long_current_contrast.sql": "3e4675b9c2964fc4ec64349e4db182a80bf55bfc95141f13c004c5075f47996b",
    "0018_substrate_slurm.sql": "364a5cc74ba903952e02b731f0f32dc0d34071044ae824d3530780549dd40145",
    "0019_paper_long_wilson_interval.sql": "3a6f90af1e4f74c10345dc895e36562f8d4a1717ab8547d9b591ef1cfb943bdd",
    "0020_item_withdrawal.sql": "d8328bfb73ad6450fe9880ecb74e653ad305f8324f78332c3df35be261b72dcd",
    "0021_run_current_frame_rules.sql": "b1a0b7f3f0a6465455ebfeda333c492d553465571bf2579d6d58cc8ef92d47f7",
    "0022_member_pass_requires_pooled_reading.sql": "2eef65b3dde8ae505d44a94a8173a7b287bf245fe172f5759a7aa35b85a2f128",
    "0023_record_refusal_fact.sql": "6aa9eaaa64b4278a0cdc8481bf85e22475780ff838f63075634b52ad162f97ce",
    "0024_fold_observation.sql": "3a83d8f7bc26f0537b740b0bdccb4f6ed63127659cbcc3d086c28841e6f63b1a",
    "0025_pending_excludes_superseded.sql": "7e3462b9e86313f962ed9d5bfff5bccd3cbc1938bd0ce7656af111cd8f1dfd64",
    "0026_twin_fold_view_legacy_value_with_draws.sql": "c8faf79d427386fd91ddea441bde066541337170f060c9f7677e8c06c8c18401",
    "0027_decisive_first_source_local_anchor_solo_stand_in.sql": "00478230eb2565d2f332e575a749f900a63de3ffa774c12249a5afa792cac75a",
    "0028_reference_model_never_selected.sql": "38d1cc22adde9b921ef9e9e4815d830c231c87bb73106b0dcc3305bdcd5237ee",
    "0029_analysis_filing.sql": "ec6ae5e530076c4a34906355036f5294ffbb56473a43b5fba80da2ef68c190ea",
    "0030_fold_order_barrier.sql": "eee0072b26fe0bac4e19661772541e7f1ae8910c8cda2c78e3188a5e6583b416",
    "0031_non_template_positions.sql": "9c2c2287855ad4e19ee620ea533976b2c028a6a403b12a289b68d19cde49d03d",
    "0032_site_label_rename.sql": "8aa50de1017653bb9bca55af2e932c8c5ec7f19c05e2043985e7e4c4ef863a45",
}

#: Column descriptions reworded after the frozen migrations landed (comment text only: ``db.normalize_ddl`` drops comments, so
#: ``code_ddl_master()`` and every shape comparison are unchanged). A frozen body keeps each one's earlier text, which
#: 0001_initial.sql records, so a test comparing a frozen body with today's render reads the render through
#: :func:`frozen_column_docs`.
REWORDED_COLUMN_DOCS = (("run", "extra_json"), ("run", "code_branch"))


def frozen_column_docs(rendered: str) -> str:
    """``rendered`` with every reworded column description at the text the frozen migrations carry (read from 0001)."""
    from sisteer.registry.migrations import HERE
    with open(os.path.join(HERE, "0001_initial.sql"), encoding="utf-8") as fh:
        first = fh.read()
    for table, col in REWORDED_COLUMN_DOCS:
        head, today = S.render_column(S.TABLE_BY_NAME[table].column(col)).split("  -- ", 1)
        m = re.search(rf"^  {re.escape(head)},?  -- (.*)$", first, re.M)
        assert m and m.group(1) != today, (table, col)
        rendered = rendered.replace(f"  -- {today}\n", f"  -- {m.group(1)}\n")
    return rendered


def test_frozen_column_docs_maps_the_render_back_to_the_frozen_text_and_nothing_else():
    """Only the reworded descriptions differ between today's render and the frozen text: mapped back, the render of a fresh
    database's whole shape equals schema.sql with the frozen descriptions, and the statements (comments dropped) are unchanged."""
    from sisteer.registry.migrations import HERE
    ddl = S.render_schema_sql()
    back = frozen_column_docs(ddl)
    assert back != ddl and len(back.splitlines()) == len(ddl.splitlines())
    changed = [(a, b) for a, b in zip(ddl.splitlines(), back.splitlines()) if a != b]
    assert changed and all(a.split("  -- ", 1)[0] == b.split("  -- ", 1)[0] for a, b in changed)
    assert DB.normalize_ddl(back) == DB.normalize_ddl(ddl)
    with open(os.path.join(HERE, "0001_initial.sql"), encoding="utf-8") as fh:
        first = fh.read()
    for _, b in changed:
        assert b in first.splitlines(), b


def test_landed_migrations_are_frozen():
    import hashlib
    from sisteer.registry.migrations import HERE, migration_files
    files = {os.path.basename(path): path for _, _, path in migration_files()}
    for fn, path in files.items():
        assert fn in FROZEN_MIGRATIONS, (f"new migration {fn}: add its sha256 to FROZEN_MIGRATIONS in {os.path.basename(__file__)} "
                                          "(a landed migration file never changes afterwards)")
        with open(path, "rb") as fh:
            sha = hashlib.sha256(fh.read()).hexdigest()
        assert sha == FROZEN_MIGRATIONS[fn], (f"{fn} was edited in place (sha256 {sha[:12]}… ≠ frozen {FROZEN_MIGRATIONS[fn][:12]}…): "
                                              "landed migration files are frozen — write a NEW NNNN_<slug>.sql for the change "
                                              "(`registry migrate --plan --sql` renders it) and leave this one as it landed")
    assert set(FROZEN_MIGRATIONS) == set(files), f"FROZEN_MIGRATIONS names a file that is not in {HERE}: {set(FROZEN_MIGRATIONS) - set(files)}"
    assert code_version() == len(FROZEN_MIGRATIONS)


_CHECK_RE = re.compile(r"CHECK \((\w+) IN \(([^)]*)\)\)")


def test_checks_equal_vocab():
    """Every `CHECK (col IN (...))` parsed from the DDL text equals the vocab tuple the column declares."""
    with open(SCHEMA_SQL, "r", encoding="utf-8") as fh:
        # the DDL WITHOUT its comments (the header names the CHECK pattern in prose)
        ddl = "\n".join(l.split("  -- ")[0] for l in fh.read().splitlines() if not l.lstrip().startswith("--"))
    found = 0
    by_col = {}
    for t in S.TABLES:
        for c in t.columns:
            if c.vocab:
                by_col.setdefault(c.name, set()).add(c.vocab)
    for m in _CHECK_RE.finditer(ddl):
        col, body = m.group(1), m.group(2)
        vals = set()
        for part in body.split(","):
            part = part.strip()
            vals.add(int(part) if part.isdigit() else part.strip("'"))
        vocabs = by_col.get(col)
        assert vocabs, f"CHECK on {col} has no vocabulary in schema_def"
        assert any(vals == set(V.VOCABULARIES[v]) for v in vocabs), (col, vals, vocabs)
        found += 1
    assert found >= 60, found


def test_every_controlled_column_has_a_check():
    conn = _fresh()
    with open(SCHEMA_SQL, "r", encoding="utf-8") as fh:
        ddl = fh.read()
    for t in S.TABLES:
        for c in t.columns:
            if c.vocab:
                assert f"CHECK ({c.name} IN (" in ddl, (t.name, c.name)


def test_every_nullable_column_has_a_reason():
    """DDL ↔ definition: every nullable column carries a `-- why-null:` comment in schema.sql."""
    conn = _fresh()
    with open(SCHEMA_SQL, "r", encoding="utf-8") as fh:
        ddl_lines = fh.read().splitlines()
    for t in S.TABLES:
        info = {r[1]: r for r in conn.execute(f"PRAGMA table_info({t.name})").fetchall()}
        for c in t.columns:
            assert c.name in info, (t.name, c.name)
            notnull = bool(info[c.name][3]) or c.name in t.pk
            if not notnull:
                line = next(l for l in ddl_lines if l.strip().startswith(c.name + " ") and "why-null" in l or
                            (l.strip().startswith(c.name + " ") and c.name in [cc.name for cc in t.columns] and "-- why-null" in l))
                assert "-- why-null:" in line, (t.name, c.name)
                assert c.why_null, (t.name, c.name)


def test_nullable_reasons_in_schema_md_match_ddl():
    """Every why-null reason of the DDL is on the column's row of its table in schema.md (less its decision citations)."""
    from sisteer.registry import docsgen
    md = docsgen.render_schema_md()
    for t in S.TABLES:
        start = md.index(f"### `{t.name}` (")
        end = md.find("\n### `", start + 1)
        section = md[start: end if end > 0 else len(md)]
        for c in t.columns:
            if not c.nn:
                assert f"| `{c.name}` | {c.type} | nullable" in section, (t.name, c.name)
                assert f"| {docsgen.history_free(c.why_null)} |" in section, (t.name, c.name)


def _seeded():
    from sisteer.registry import harvest as H
    conn = _fresh()
    ing = H.Ingestor(conn, "/nonexistent", live=False)
    from sisteer.registry import validate as VAL
    recs = synth.seed_dimensions()
    with DB.transaction(conn):
        ranker = H.Ranker(conn)
        ex = H._Existence(conn)
        from sisteer.registry.records import record_file_id_for_uri, record_sha
        seq = ranker.assign("seed", "0" * 64, "2026-01-01T00:00:00.000Z", len(recs))
        for i, r in enumerate(recs, start=1):
            v = VAL.validate_record(r, "seed", record_sha(r))
            assert isinstance(v, VAL.Validated), v
            v.uri = f"seed#{i}"
            assert ing.apply_validated(v, seq, "seed", record_file_id_for_uri("seed"), {}, set(), ex, "2026-01-01T00:00:00.000Z") == "ok"
    return conn


def _run_row(**over):
    p = synth.run_payload("k1", i=0)
    p.update(over)
    cols = [c for c in S.TABLE_BY_NAME["run"].column_names if c in p]
    return cols, [p[c] for c in cols]


def _insert_run(conn, **over):
    cols, vals = _run_row(**over)
    cols += ["field_versions_json", "extra_json", "record_uri", "ingest_seq"]
    vals += ["{}", "{}", "x", 99]
    conn.execute(f"INSERT INTO run ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})", vals)


@pytest.mark.parametrize("col,value", [("serving_path", "vllm"), ("lifecycle_state", "done"), ("applied_positions_mode", "everywhere")])
def test_oov_insert_fails_naming_column(col, value):
    conn = _seeded()
    with pytest.raises(sqlite3.IntegrityError) as ei:
        _insert_run(conn, **{col: value})
    assert col in str(ei.value)


def test_oov_fit_positions_on_object_fails():
    conn = _seeded()
    o = synth.object_lora()
    o["fit_positions"] = "somewhere"
    o["fit_layers_json"] = "[4,5,6]"
    o["fit_hyperparameters_json"] = "{}"
    cols = list(o) + ["field_versions_json", "extra_json", "record_uri", "ingest_seq"]
    vals = list(o.values()) + ["{}", "{}", "x", 98]
    with pytest.raises(sqlite3.IntegrityError) as ei:
        conn.execute(f"INSERT INTO object ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})", vals)
    assert "fit_positions" in str(ei.value)


def test_judged_without_judgment_trigger():
    conn = _seeded()
    with pytest.raises(sqlite3.IntegrityError) as ei:
        _insert_run(conn, lifecycle_state="judged", started_utc=synth.t(1), generated_utc=synth.t(2), judged_utc=synth.t(3))
    assert "requires a judgment row" in str(ei.value)


def test_reached_state_requires_its_utc():
    conn = _seeded()
    with pytest.raises(sqlite3.IntegrityError) as ei:
        _insert_run(conn, lifecycle_state="generated", started_utc=synth.t(1))  # generated_utc missing
    assert "trg_run_state_utc" in str(ei.value)


def test_item_result_sentinel_fk_trigger():
    conn = _seeded()
    _insert_run(conn)
    rid = synth.run_payload("k1")["run_id"]
    conn.execute("INSERT INTO record_ingest(file_uri, record_file_id, file_sha256, first_written_utc, arrival_utc, n_records, ingest_seq) VALUES ('f', 1, ?, '2026-01-01T00:00:00.000Z', '2026-01-01T00:00:00.000Z', 1, 2)", ("0" * 64,))
    base = ("INSERT INTO item_result(run_id, item_id, metric_id, trial_index, sample_index, turn_index, judgment_id, value, created_utc, "
            "source_revision_id, written_utc, record_sha256, extra_json, record_file_id, ingest_seq) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)")
    # sentinel '' passes
    conn.execute(base, (rid, "q1", "asr", 0, 0, -1, "", 1.0, synth.t(0), "", 0, b"\x00" * 32, "{}", 1, 5))
    # a non-sentinel judgment_id with no judgment row fails by trigger, naming the column
    with pytest.raises(sqlite3.IntegrityError) as ei:
        conn.execute(base, (rid, "q2", "asr", 0, 0, -1, "missing-judgment", 1.0, synth.t(0), "", 0, b"\x01" * 32, "{}", 1, 6))
    assert "item_result.judgment_id" in str(ei.value)


def test_foreign_keys_enforced():
    conn = _seeded()
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    with pytest.raises(sqlite3.IntegrityError):
        _insert_run(conn, model_id="no_such_model")


def test_computation_id_in_metric_and_contrast_keys():
    assert "computation_id" in S.TABLE_BY_NAME["run_metric"].pk
    assert "computation_id" in S.TABLE_BY_NAME["run_contrast"].pk


def test_item_result_keyed_per_item_metric_and_flags_table_exists():
    assert "metric_id" in S.TABLE_BY_NAME["item_result"].pk
    assert "item_flags" in S.ALL_TABLES
    # the item-scale classification — `source_status` joined it (revision-keyed, whole-row, no row_history)
    assert set(S.ITEM_SCALE_TABLES) == {"item_result", "item_flags", "run_degeneration", "source_status"}
    assert "source_revision_id" in S.TABLE_BY_NAME["source_status"].pk and "source_id" in S.TABLE_BY_NAME["source_status"].pk


# ---- fit_target_kind pairing (F16) --------------------------------------------------------

_T = synth.SHA256_A


def _valid_object(kind: str) -> dict:
    base = {"object_id": f"obj_{kind}", "object_kind": "lora", "paper_label": "other", "campaign": "c", "object_role": "control",
            "fit_target_kind": kind, "fit_thinking_render": "off", "fit_objective": "ce", "fit_positions": "all",
            "fit_seed_confidence": "unverified", "derivation": "none", "artifact_sha256": "a" * 64, "status": "approved",
            "field_confidence_json": '{"fit_seed_confidence": {"confidence": "unverified", "reason": "no sidecar"}}'}
    if kind == "system_instruction_distillation":
        base.update(fit_objective="kl", fit_teacher_si_id="synthetic_si", fit_teacher_si_text_sha256=synth.SI_SHA,
                    fit_prompt_pool_id="pool", fit_prompt_pool_sha256=_T, fit_model_id=synth.FIT_MODEL_ID, fit_layers_json="[1]")
    elif kind == "corpus_document_fit":
        base.update(fit_corpus_id="corpus", fit_corpus_sha256=_T, fit_model_id=synth.FIT_MODEL_ID, fit_layers_json="[1]")
    elif kind == "contrastive_direction":
        base.update(object_kind="antidote", fit_objective="antidote", fit_contrast_sets_json='{"positive": {}, "negative": {}}',
                    fit_model_id=synth.FIT_MODEL_ID, fit_layers_json="[1]")
    elif kind == "derived":
        base.update(object_kind="lora_subtraction", fit_objective="not_applicable", derivation="lora_subtraction",
                    parent_object_ids_json='["a","b"]', fit_positions="inherited_from_parents")
    elif kind == "declared_prompt":
        base.update(object_kind="system_instruction", fit_objective="not_applicable", si_text_sha256=synth.SI_SHA,
                    fit_positions="not_applicable", artifact_sha256="si:" + synth.SI_SHA)
    elif kind == "none":
        base.update(object_kind="none", fit_objective="not_applicable", fit_positions="not_applicable", artifact_sha256="none",
                    object_role="baseline_none", paper_label="none", object_id="none_x")
    return base


def _insert_object(conn, o: dict):
    cols = list(o) + ["field_versions_json", "extra_json", "record_uri", "ingest_seq"]
    vals = list(o.values()) + ["{}", "{}", "x", 97]
    conn.execute(f"INSERT INTO object ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})", vals)


@pytest.mark.parametrize("kind", list(V.FIT_TARGET_RULES))
def test_fit_target_kind_valid_example_per_kind(kind):
    conn = _seeded()
    o = _valid_object(kind)
    _insert_object(conn, o)
    assert conn.execute("SELECT fit_target_kind FROM object WHERE object_id = ?", (o["object_id"],)).fetchone()[0] == kind


_FORBIDDEN_CASES = [(kind, col) for kind, rule in V.FIT_TARGET_RULES.items() for col in rule["null"]]
_SAMPLE_VALUE = {"fit_teacher_si_id": "synthetic_si", "fit_teacher_si_text_sha256": synth.SI_SHA, "fit_prompt_pool_id": "p",
                 "fit_prompt_pool_sha256": _T, "fit_corpus_id": "c", "fit_corpus_sha256": _T, "fit_corpus_n_documents": 1,
                 "fit_corpus_n_tokens": 1, "fit_corpus_description": "d", "fit_contrast_sets_json": "{}",
                 "parent_object_ids_json": "[]", "derivation_params_json": "{}", "si_text_sha256": synth.SI_SHA, "si_token_count": 1}


@pytest.mark.parametrize("kind,col", _FORBIDDEN_CASES)
def test_fit_target_kind_forbidden_extra_field(kind, col):
    conn = _seeded()
    o = _valid_object(kind)
    o[col] = _SAMPLE_VALUE[col]
    with pytest.raises(sqlite3.IntegrityError) as ei:
        _insert_object(conn, o)
    assert "object_fit_target_kind_pairing" in str(ei.value)


def test_forbidden_case_count_covers_every_required_null_cell():
    assert len(_FORBIDDEN_CASES) >= 30


# ---- view references are declared beside the SQL, and the declaration cannot drift ------------------

_IDENT = re.compile(r"(?<![A-Za-z0-9_.])([a-z_][a-z0-9_]*)(?![A-Za-z0-9_])")


def _identifiers_in(sql: str) -> set:
    """Table / view names referenced by a view's SQL: every known name that appears as a bare identifier
    (string literals stripped, `alias.column` references excluded by the leading-dot guard)."""
    body = re.sub(r"'[^']*'", " ", sql)
    known = set(S.ALL_TABLES) | set(S.VIEW_NAMES)
    return {m.group(1) for m in _IDENT.finditer(body) if m.group(1) in known}


@pytest.mark.parametrize("name", list(S.VIEW_NAMES))
def test_view_declared_references_equal_the_sql(name):
    sql = S.VIEW_SQL[name]
    found = _identifiers_in(sql) - {name}
    assert set(S.VIEW_REFERENCES[name]) == found, (name, set(S.VIEW_REFERENCES[name]) ^ found)


def test_view_reference_scan_fires_on_a_known_bad_declaration():
    """The scan behind the test above catches a declaration that omits a table the SQL reads."""
    sql = "CREATE VIEW v_probe AS SELECT r.run_id FROM run r JOIN judgment j ON j.run_id = r.run_id WHERE j.status = 'run';"
    assert _identifiers_in(sql) == {"run", "judgment"}          # the literal 'run' does not count twice; judgment is found
    assert _identifiers_in(sql) != {"run"}                       # a declaration naming only `run` would be refused


def test_view_tables_close_over_referenced_views():
    assert "run_metric" in S.VIEW_TABLES["v_paper_inputs_head"]            # directly since migration 0015 (through v_paper_long before)
    assert "item_result" in S.VIEW_TABLES["v_paper_inputs_head"]           # directly
    assert "judgment" in S.VIEW_TABLES["v_paper_inputs_head"]              # directly since migration 0021 (a superseded judgment moves the frame)
    # the head reads no paper object since 0015: its closure is exactly its eleven base tables (judgment since 0021; a v_paper_long
    # reference would pull in observation, eval, instrument, ... and the frame evaluation the migration removed)
    assert "v_paper_long" not in S.VIEW_REFERENCES["v_paper_inputs_head"] and "observation" not in S.VIEW_TABLES["v_paper_inputs_head"]
    assert S.VIEW_TABLES["v_paper_inputs_head"] == frozenset(S.VIEW_REFERENCES["v_paper_inputs_head"])
    # v_paper_current reads one item-scale table, and only through the multi-source objects its coverage test admits rows by (migration 0021):
    # run_degeneration, the table the run-level snapshot carries for exactly those objects (snapshot.RUNLEVEL_ITEM_TABLES), so the selection
    # objects stay in the run-level file
    assert S.VIEW_TABLES["v_roster_gaps"] & set(S.ITEM_SCALE_TABLES) == {"run_degeneration"}
    assert S.VIEW_TABLES["v_paper_current"] & set(S.ITEM_SCALE_TABLES) == {"run_degeneration"}
    for name, tables in S.VIEW_TABLES.items():
        assert tables <= set(S.ALL_TABLES), name


def test_sentinels_and_sentinel_fks_come_from_the_column_model():
    assert S.SENTINELS == {"judgment_id": "", "computation_id": "", "anchor_judgment_id": "", "source_revision_id": "",
                           "threshold": -1.0, "turn_index": -1}
    ddl = S.render_schema_sql()
    for t in S.TABLES:
        for c in t.columns:
            if c.fk and c.sentinel is not None:
                assert f"CREATE TRIGGER trg_{t.name}_{c.name}_fk_ins" in ddl, (t.name, c.name)
                assert f"{c.name} TEXT NOT NULL DEFAULT '' REFERENCES" not in ddl, (t.name, c.name)  # trigger, never REFERENCES
                assert (c.name, c.fk[0], c.fk[1]) in S.foreign_keys(t.name)
    assert S.payload_columns("item_result")[-1] == "extra_json" or "extra_json" in S.payload_columns("item_result")
    for t in S.TABLES:
        for c in t.columns:
            if c.derived:
                assert c.name not in S.payload_columns(t.name) and c.name not in S.required_fields(t.name), (t.name, c.name)


def test_item_result_withdrawal_marks_trail_the_tail_and_are_added_in_place():
    """The two producer-withdrawal marks (migration 0020) are declared AFTER ingest_seq and carry no CHECK, so the live table is
    migrated by ALTER TABLE ADD COLUMN: SQLite appends an added column after the last existing one, and the altered v19 table's
    stored text normalizes equal to the code's CREATE text -- pinned here so a future column is not slid ahead of the tail (which
    would force a copy of the 12 M-row table) or given a CHECK (which makes ADD COLUMN scan every existing row)."""
    from sisteer.registry import migrations as M
    t = S.TABLE_BY_NAME["item_result"]
    cols = t.column_names
    assert cols[cols.index("ingest_seq") + 1:] == ["withdrawn_by_record_sha256", "withdrawn_reason"]
    for c in ("withdrawn_by_record_sha256", "withdrawn_reason"):
        col = t.column(c)
        assert col.derived and not col.nn and col.pattern is None and col.vocab is None and col.fk is None and col.default is None
    assert "withdrawn_by_record_sha256" not in S.payload_columns("item_result")
    conn = DB.connect_memory()
    for version, slug, path in M.migration_files():
        if version <= 19:
            M._apply_one(conn, version, slug, open(path).read())
    for c in ("withdrawn_by_record_sha256", "withdrawn_reason"):
        conn.execute(M.add_column_sql("item_result", c).rstrip(";"))
    live = conn.execute("SELECT sql FROM sqlite_master WHERE name = 'item_result'").fetchone()[0]
    assert DB.normalize_ddl(live) == DB.normalize_ddl(S.render_create_table(t))
    assert live.startswith("CREATE TABLE item_result (")                                 # no rename, no rebuild
    conn.close()


def test_open_or_create_refuses_a_database_whose_shape_drifted_at_the_same_migration_version(tmp_path):
    """2026-09-17 live: the database was created at an older 0001_initial.sql (edited in place — the pre-release practice);
    the version guard saw v1 == v1, and the harvester crashed 36 times deep inside a write on ``table judge has no column
    named field_confidence_json``. The shape check refuses up front, names the drift and the remedy (a rebuild)."""
    from sisteer.registry import db as DB
    path = str(tmp_path / "r.sqlite")
    DB.open_fresh(path).close()
    with DB.reading(path) as conn:
        assert DB.schema_drift(conn) == []                                   # a fresh database agrees with the code
    DB.open_or_create(path).close()                                          # and opens for writing
    conn = DB.connect(path)
    conn.execute("ALTER TABLE judge DROP COLUMN field_confidence_json")      # the live drift: a seeds' dimension column
    conn.execute("DROP TABLE source_status")                                 # and a table added by round 3
    conn.execute("DROP VIEW v_source_status")
    conn.commit()
    conn.close()
    with DB.reading(path) as conn:
        drift = DB.schema_drift(conn)
    assert "table judge lacks column(s) field_confidence_json" in drift
    assert "table source_status is absent" in drift and "v_source_status is absent" in drift
    with pytest.raises(RuntimeError) as ei:
        DB.open_or_create(path)
    msg = str(ei.value)
    assert "judge lacks column(s) field_confidence_json" in msg and "registry rebuild" in msg and f"schema version {code_version()}" in msg
    assert "registry migrate --plan" in msg          # the 2026-09-19 remedy: a migration, not only a rebuild


# --------------------------------------------------------------------------------------------
# The shape guard covers constraint text, not only tables / columns / v_* names (2026-09-19)
# --------------------------------------------------------------------------------------------


def _shape_from(ddl: str, path: str) -> None:
    """A database at ``path`` created from ``ddl`` (a modified render) with the version row of a v(code) database."""
    conn = DB.connect(path)
    conn.executescript(ddl)
    for v in range(1, code_version() + 1):
        conn.execute("INSERT INTO schema_migration(version, applied_utc, description) VALUES (?, '2026-09-18T22:10:08.091Z', 'x')", (v,))
    conn.commit()
    conn.close()


def test_normalize_ddl_ignores_comments_whitespace_and_quoting_but_not_literals():
    a = DB.normalize_ddl("CREATE TABLE \"t\" (\n  a TEXT NOT NULL,  -- why-null: it's fine\n  b TEXT CHECK (b IN ('x', 'y--z'))\n);")
    b = DB.normalize_ddl("CREATE TABLE t(a TEXT NOT NULL,b TEXT CHECK(b IN('x','y--z')))")
    assert a == b
    assert DB.normalize_ddl("CREATE TABLE t (b TEXT CHECK (b IN ('x', 'Y')))") != DB.normalize_ddl("CREATE TABLE t (b TEXT CHECK (b IN ('x', 'y')))")


def test_drift_names_a_check_list_value_the_database_lacks(tmp_path):
    """The live failure: ``vocab.PAPER_LABELS`` grew, the database kept the old CHECK; the drift names the table, the
    column and the missing value, and the writers' open refuses with the migrate remedy."""
    ddl = S.render_schema_sql().replace("'difference_relu_distilled', ", "", 1)
    assert ddl != S.render_schema_sql()
    path = str(tmp_path / "r.sqlite")
    _shape_from(ddl, path)
    with DB.reading(path) as conn:
        drift = DB.schema_drift(conn)
    assert drift == ["table object: column paper_label CHECK (paper_label IN (...)) lacks 'difference_relu_distilled'"]
    with pytest.raises(DB.SchemaDrift) as ei:
        DB.open_or_create(path)
    msg = str(ei.value)
    assert "paper_label" in msg and "'difference_relu_distilled'" in msg and "registry migrate --plan" in msg and "registry rebuild" in msg
    assert ei.value.lines == drift and ei.value.have == ei.value.want == code_version()


def test_drift_names_not_null_default_references_view_trigger_and_index_text(tmp_path):
    ddl = S.render_schema_sql()
    ddl = ddl.replace("  campaign TEXT NOT NULL,\n", "  campaign TEXT,\n", 1)                                   # NOT NULL dropped
    o_start = ddl.index("CREATE TABLE object (")
    o_end = ddl.index("\n);\n", o_start)
    block = ddl[o_start:o_end]
    assert block.count("  extra_json TEXT NOT NULL DEFAULT '{}'") == 1
    ddl = ddl[:o_start] + block.replace("  extra_json TEXT NOT NULL DEFAULT '{}'", "  extra_json TEXT NOT NULL DEFAULT '[]'", 1) + ddl[o_end:]   # DEFAULT moved
    idx = S.render_indexes(S.TABLE_BY_NAME["object"])[0]
    assert idx.startswith("CREATE INDEX idx_object_artifact_sha ON object (") and ddl.count(idx) == 1
    ddl = ddl.replace(idx, idx.replace("(artifact_sha256", "(artifact_sha256, campaign", 1), 1)                    # index text moved
    ddl = ddl.replace("SELECT RAISE(ABORT, 'run.lifecycle_state=judged requires a judgment row for this run_id');", "SELECT RAISE(ABORT, 'judged needs a judgment');", 1)
    ddl = ddl.replace("CREATE VIEW v_source_status AS", "CREATE VIEW v_source_status AS -- older text\n", 1)      # a comment only: NOT drift
    path = str(tmp_path / "r.sqlite")
    _shape_from(ddl, path)
    conn = DB.connect(path)
    conn.execute("DROP VIEW v_in_flight")
    conn.execute("CREATE VIEW v_in_flight AS SELECT run_id FROM run")
    conn.execute("CREATE INDEX idx_by_hand ON run (campaign)")
    conn.commit()
    conn.close()
    with DB.reading(path) as conn:
        drift = DB.schema_drift(conn)
    assert "table fit_search: column campaign NOT NULL differs — have «none» want «NOT NULL»" in drift   # the first `campaign` line
    assert "table object: column extra_json DEFAULT differs — have «DEFAULT '[]'» want «DEFAULT '{}'»" in drift
    assert "index idx_object_artifact_sha differs from the code's SQL" in drift
    assert "trigger trg_run_judged_requires_judgment_ins differs from the code's SQL" in drift
    assert "v_in_flight differs from the code's SQL" in drift
    assert "index idx_by_hand is not declared by the code" in drift
    assert not any("v_source_status" in line for line in drift), drift


def test_drift_names_a_table_constraint_and_an_undeclared_table(tmp_path):
    ddl = S.render_schema_sql().replace("  CONSTRAINT object_fit_target_kind_pairing CHECK (", "  CONSTRAINT object_fit_target_kind_pairing CHECK (fit_objective IS NOT NULL OR ", 1)
    path = str(tmp_path / "r.sqlite")
    _shape_from(ddl, path)
    conn = DB.connect(path)
    conn.execute("CREATE TABLE scratch_by_hand (x TEXT)")
    conn.commit()
    conn.close()
    with DB.reading(path) as conn:
        drift = DB.schema_drift(conn)
    assert any(line.startswith("table object: CONSTRAINT object_fit_target_kind_pairing differs") for line in drift), drift
    assert "table scratch_by_hand is not declared by the code" in drift


def test_version_behind_the_code_is_the_same_named_refusal(tmp_path):
    path = str(tmp_path / "r.sqlite")
    DB.open_fresh(path).close()
    conn = DB.connect(path)
    conn.execute("DELETE FROM schema_migration WHERE version = ?", (code_version(),))
    conn.commit()
    conn.close()
    with pytest.raises(DB.SchemaDrift) as ei:
        DB.open_or_create(path)
    assert ei.value.have == code_version() - 1 and ei.value.want == code_version() and ei.value.lines == []
    assert f"schema version {code_version() - 1}, code expects {code_version()}" in str(ei.value) and "registry migrate" in str(ei.value)

