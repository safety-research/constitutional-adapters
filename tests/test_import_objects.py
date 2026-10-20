"""reproduce/tools/import_objects.py — the worked example's committed objects (objects/DEMO.json) into a key's cells, pins and registry
(demo-import M1; plan tests a-h). The nine objects are the seven DEMO entries plus the two committed 0.75 dose copies of the differences.

(a) the nine statements load into a FRESH registry built from the seeds (the real schema, every CHECK on): 9 + 9 object rows, nothing
refused or blocked; (b) a second import against that registry skips all nine and emits nothing; (c) a live row altered on ONE identity
column refuses by name — every identity column, base_scale and the two layer columns included, as a live-row/payload comparison and
through the subprocess; and the ACTUAL hook payload of each of the nine carries BOTH layer columns (the pre-M1b hook stated the list and
no convention on the five explicit-layer rows); (d) `cells`: nine rows, deduplicated by staged id, the two `d0p75__` copies present, every
expectation the object id; (e) `pins` is the ONE registry-backed pins writer scoped to the key (review round 1): over the REAL stager's
manifest and a registry holding the nine it writes the block with every other key byte-identical and the generator's --check green; over
a registry without the rows, with a withdrawn row, a copy whose live parent is not the pinned one, a changed committed entry, a manifest
digest that is not the committed id, no manifest or no registry it REFUSES by name and writes nothing; and the scope (`targets`) is what
lets a registry holding one lane's objects write that lane's block without being asked to stand behind every other key's pins; (f) the
mapping restates objects/MANIFEST.json, the refusal direction's base_scale is its operating coefficient and its layers the SERVED block
([12], zero_based_decoder_block), never a conversion of fit_layer (13) — a provenance whose served and stored layers disagree, or carry
neither, refuses by name; (g) sisteer.objects.resolve finds each id at its address; (h) the three demo banks sit outside every root the
objects_bank adapter scans; (i) the statements carry the fits' recorded effective batch (64, the column — not the micro-batch 2 the hook
used to fold) and NO import-context provenance: the importing checkout, host, Slurm job and session never reach the fit_* columns of the
payload or of a fresh registry's rows, whatever the import's environment says (review round 1); `fit_code_sha` is the STATED 'unverified'
on the eight fits, differences and copies — with the hook's own field_confidence_json entry, which is what lets the row load with every
CHECK on — and unstated (NULL) on the refusal direction, and the live cross-check holds every row to it. One
test reads the grading host registry read-only (skipped where absent): the plan's verify line, nine `skip (already stated)` with both layer
columns and the fit_code_sha previewed."""
from __future__ import annotations

import importlib.util
import json
import yaml
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _rsync_fallback import rsync_on_path  # noqa: E402,F401  (autouse: the fallback rsync on PATH where the host has none; a no-op where it has one)
from test_stage_objects_registry_id import _site  # noqa: E402  (a one-cluster `local:` site for the real stager)
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "reproduce" / "lib"))
sys.path.insert(0, str(REPO / "scripts" / "sdfbase" / "full22"))
import models as MT  # noqa: E402
import stage_objects as SO  # noqa: E402
from sisteer.registry import db as DB  # noqa: E402
from sisteer.registry import harvest as H  # noqa: E402
from sisteer.registry import hooks  # noqa: E402
from sisteer.registry import synth  # noqa: E402
from sisteer.registry import vocab as V  # noqa: E402
from sisteer.registry.adapters import dimensions as D  # noqa: E402
from sisteer.registry.adapters import objects_bank as OB  # noqa: E402
from sisteer.objects import resolve  # noqa: E402

TOOL = REPO / "reproduce" / "tools" / "import_objects.py"
SEEDS = str(REPO / "configs" / "registry")
PROD = synth.producer("tests.import_objects")
KEY = "qwen35_4b_demo"
PY = sys.executable


def _load_tool():
    spec = importlib.util.spec_from_file_location("import_objects", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


IO = _load_tool()
DEMO = json.loads((REPO / "objects" / "DEMO.json").read_text())
MANIFEST = {r["object_id"]: r for r in json.loads((REPO / "objects" / "MANIFEST.json").read_text())["objects"]}
#: the nine: staged id -> (object id, the committed sidecar's layer list, the live rows' convention)
EXPECTED_LAYERS = {
    "arditi__ar_c0p3__arditi_allpos": [12],
    "base__constv2ctl2-full__lora_r4_ce": [3, 7, 11, 15, 19, 23, 27, 31],
    "base__constv2-mix-full__lora_r4_ce": [3, 7, 11, 15, 19, 23, 27, 31],
    "derived__constv2-mix-full__lora_r4_ce_minus_ctl": [3, 7, 11, 15, 19, 23, 27, 31],
    "d0p75__constv2-mix-full__lora_r4_ce_minus_ctl": [3, 7, 11, 15, 19, 23, 27, 31],
    "base__constv2ctl2-full__psr_ce_all": list(range(32)),
    "base__constv2-mix-full__psr_ce_all": list(range(32)),
    "derived__constv2-mix-full__psr_ce_all_minus_ctl": list(range(32)),
    "d0p75__constv2-mix-full__psr_ce_all_minus_ctl": list(range(32)),
}
EXPLICIT_LAYER_ROWS = ("arditi__ar_c0p3__arditi_allpos", "derived__constv2-mix-full__lora_r4_ce_minus_ctl", "d0p75__constv2-mix-full__lora_r4_ce_minus_ctl",
                       "derived__constv2-mix-full__psr_ce_all_minus_ctl", "d0p75__constv2-mix-full__psr_ce_all_minus_ctl")


def _run(*args, env=None, cwd=None):
    # the suite disables every producer's emission for the session (emit.DISABLED_ENV); the importer IS the producer under test
    e = {k: v for k, v in os.environ.items() if k not in ("SISTEER_REGISTRY_ROOT", "SISTEER_REGISTRY_SPOOL", hooks.emit.DISABLED_ENV, hooks.EMIT_ENV)}
    e["PYTHONPATH"] = str(REPO / "src")
    if env:
        e.update(env)
    return subprocess.run([PY, str(TOOL), *args], capture_output=True, text=True, env=e, cwd=str(cwd or REPO), timeout=600)


def _fresh_registry(root: Path) -> Path:
    """A registry built from the committed seeds alone (the recipe tests/test_registry_seeds_demo_model.py uses)."""
    root.mkdir(parents=True, exist_ok=True)
    H.ensure_layout(str(root))
    D.write_seed_records(str(root), SEEDS, producer=PROD, now="2026-10-10T00:00:00.000Z")
    rep = H.ingest_once(str(root), workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    return root


def _count(root: Path, sql: str, *args):
    conn = DB.connect(DB.db_path(str(root)), readonly=True)
    try:
        return conn.execute(sql, args).fetchone()[0]
    finally:
        conn.close()


#: the IMPORT's context, as a Slurm job would carry it: the record's producer block states it, the object's fit_* columns never
IMPORT_CONTEXT = {"SLURM_JOB_ID": "424242", "SISTEER_SESSION_NAME": "synthetic-import-session"}
#: the five fit-time columns the hook fills from the emitting context when no one states them (hooks.FIT_PROVENANCE_KEYS's first five)
CONTEXT_COLUMNS = ("fit_code_sha", "fit_job_id", "fit_node_hostname", "fit_substrate", "fit_producer_session")
#: the four context columns every statement leaves UNSTATED; fit_code_sha is stated 'unverified' on eight and unstated on the direction
UNSTATED_COLUMNS = CONTEXT_COLUMNS[1:]
FIT_SIDS = ("base__constv2ctl2-full__lora_r4_ce", "base__constv2-mix-full__lora_r4_ce", "base__constv2ctl2-full__psr_ce_all", "base__constv2-mix-full__psr_ce_all")
ARDITI_SID = "arditi__ar_c0p3__arditi_allpos"
#: the hook's own entry for a stated 'unverified' fit_code_sha (hooks.FIT_CODE_SHA_UNVERIFIED_ENTRY), pinned by value
CODE_SHA_ENTRY = {"confidence": "unverified", "reason": "no sidecar records the fit code sha", "source": "sisteer.registry.hooks.object_from_sidecar"}


@pytest.fixture(scope="module")
def imported(tmp_path_factory):
    """A seeds-built registry with the nine committed objects imported for real (spool = <root>/records) and folded — by an importer that
    believes it runs inside a Slurm job with a session name (IMPORT_CONTEXT), so the context the fit_* columns must NOT carry is present."""
    root = _fresh_registry(tmp_path_factory.mktemp("reg") / "root")
    assert _count(root, "SELECT count(*) FROM object") == 9                         # the seeds' `none` + the eight si texts
    r = _run("registry", "--model", KEY, "--registry-root", str(root), "--spool", str(root / "records"), env=IMPORT_CONTEXT)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert r.stdout.count("emitted ") == 9 and "skip (already stated)" not in r.stdout and "REFUSED" not in r.stderr, r.stdout
    rep = H.ingest_once(str(root), workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    return root


# ------------------------------------------------------------------------------------------------------------ (a) + (b): a fresh registry
def test_the_nine_statements_load_into_a_fresh_registry_with_every_check_on(imported):
    root = imported
    assert _count(root, "SELECT count(*) FROM object") == 9 + 9
    assert _count(root, "SELECT count(*) FROM object WHERE fit_layers_json IS NOT NULL AND fit_layer_index_convention IS NULL") == 0
    conn = DB.connect(DB.db_path(str(root)), readonly=True)
    try:
        for sid, layers in EXPECTED_LAYERS.items():
            oid = _plan_ids()[sid]
            row = conn.execute("SELECT fit_layers_json, fit_layer_index_convention, status, campaign, fit_model_id, base_scale, derivation FROM object WHERE object_id = ?", (oid,)).fetchone()
            assert row is not None, sid
            assert json.loads(row[0]) == layers and row[1] == "zero_based_decoder_block" and row[2] == "approved", (sid, tuple(row))
            man = MANIFEST[oid]
            assert row[3] == man["campaign"] and row[4] == man["model_id"] and row[6] == man["derivation"], (sid, tuple(row))
        bs = {r[0]: r[1] for r in conn.execute("SELECT object_id, base_scale FROM object WHERE base_scale IS NOT NULL").fetchall()}
        prov_rows = {r[0]: r[1:] for r in conn.execute("SELECT object_id, fit_effective_batch, fit_code_sha, fit_job_id, fit_node_hostname, fit_substrate, "
                                                       "fit_producer_session, fit_finished_utc, field_confidence_json FROM object WHERE object_id IN (%s)" % ",".join("?" * 9),
                                                       tuple(_plan_ids().values())).fetchall()}
    finally:
        conn.close()
    ids = _plan_ids()
    assert bs == {ids["arditi__ar_c0p3__arditi_allpos"]: 0.3, ids["d0p75__constv2-mix-full__lora_r4_ce_minus_ctl"]: 0.75, ids["d0p75__constv2-mix-full__psr_ce_all_minus_ctl"]: 0.75}
    # (i) the fits' batch is the COLUMN, 64 (train_budget_meta.json effective_batch; objects/DEMO.json effective_batch) — not the micro-batch 2
    assert len(prov_rows) == 9
    assert {sid: prov_rows[ids[sid]][0] for sid in FIT_SIDS} == {sid: 64 for sid in FIT_SIDS}
    assert all(prov_rows[ids[sid]][0] is None for sid in ids if sid not in FIT_SIDS)              # a derived object / copy / direction has no batch
    # (i) the importing context never reached the fit_* columns: job, host, substrate, session all NULL on a fresh registry — the committed
    # sidecars record none of them, and the column is left unstated. fit_code_sha: the STATED 'unverified' on the
    # eight fits, differences and copies, loaded with every CHECK on because the hook wrote its own field_confidence_json entry beside it
    # (validate.py refuses the bare literal), and NULL on the refusal direction, which stays unstated — exactly what the live rows read
    for sid, oid in ids.items():
        eb, code_sha, job, host, substrate, session, finished, fc_text = prov_rows[oid]
        assert (job, host, substrate, session) == (None, None, None, None), (sid, prov_rows[oid])
        fc = json.loads(fc_text) if fc_text else {}
        if sid == ARDITI_SID:
            assert code_sha is None and "fit_code_sha" not in fc, (sid, code_sha, fc)
        else:
            assert code_sha == "unverified" and fc["fit_code_sha"] == CODE_SHA_ENTRY == dict(hooks.FIT_CODE_SHA_UNVERIFIED_ENTRY), (sid, code_sha, fc)
        assert (finished is None) == (sid in FIT_SIDS), (sid, finished)                              # a fit's sidecar records no time; the others do
    assert _count(root, "SELECT count(*) FROM object WHERE fit_code_sha = 'unverified' AND object_id IN (%s)" % ",".join("?" * 9), *ids.values()) == 8
    # ... while the records' producer block carries the import's context (the right home for it)
    # the importer's nine object records, under their campaigns (the seeds' own object records — `none` and the si texts — sit under records/dimensions/)
    recs = sorted(p for p in (root / "records").rglob("*-object.jsonl") if "dimensions" not in p.parts)
    assert len(recs) == 9, recs
    producers = [json.loads(line)["producer"] for p in recs for line in p.read_text().splitlines() if line.strip()]
    assert producers and all(pr["session"] == IMPORT_CONTEXT["SISTEER_SESSION_NAME"] and pr["script"] == IO.SCRIPT for pr in producers), producers[:1]


def test_a_second_import_skips_all_nine_and_emits_nothing(imported):
    root = imported
    before = {str(p.relative_to(root)) for p in (root / "records").rglob("*") if p.is_file()}
    for flags in ((), ("--dry-run",)):
        r = _run("registry", "--model", KEY, "--registry-root", str(root), "--spool", str(root / "records"), *flags)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        assert r.stdout.count("skip (already stated)") == 9 and "emitted " not in r.stdout and "would emit" not in r.stdout, r.stdout
        assert "9 already stated, 0" in r.stdout
    after = {str(p.relative_to(root)) for p in (root / "records").rglob("*") if p.is_file()}
    assert after == before                                                            # no record written


# ------------------------------------------------------------------------------------------------------------ (c): the identity cross-check
def _plan_ids() -> dict:
    plan = IO.Plan(KEY, repo=REPO)
    return {r["staged_id"]: r["object_sha256_expected"] for r in plan.rows}


ALTERED = {
    "campaign": "another_campaign", "object_kind": "vector", "paper_label": "control_lora", "object_role": "irrelevant",
    "fit_target_kind": "none", "fit_objective": "kl", "derivation": "stack", "fit_model_id": "qwen35_4b_demo_base", "status": "reference",
    "parent_object_ids_json": json.dumps(["f" * 64]), "base_scale": 0.5, "fit_layers_json": "[13]", "fit_layer_index_convention": "one_based",
    "fit_code_sha": "unverified",     # the direction's row reads NULL and its statement leaves the column unstated: 'unverified' there is a discrepancy
}


@pytest.mark.parametrize("column", sorted(ALTERED))
def test_a_live_row_altered_on_one_identity_column_refuses_by_name(imported, column):
    """Every identity column, altered on the live row of ONE object (the refusal direction, the row that carries base_scale and [12]), makes the
    import REFUSE naming the column — in the dry run and the real run alike — and nothing is written; the row is restored afterwards."""
    root = imported
    oid = _plan_ids()["arditi__ar_c0p3__arditi_allpos"]
    conn = DB.connect(DB.db_path(str(root)))
    try:
        conn.execute("PRAGMA foreign_keys=OFF"); conn.execute("PRAGMA ignore_check_constraints=ON")
        old = conn.execute(f"SELECT {column} FROM object WHERE object_id = ?", (oid,)).fetchone()[0]
        conn.execute(f"UPDATE object SET {column} = ? WHERE object_id = ?", (ALTERED[column], oid))
        conn.commit()
    finally:
        conn.close()
    try:
        before = {str(p) for p in (root / "records").rglob("*") if p.is_file()}
        for flags in (("--dry-run",), ()):
            r = _run("registry", "--model", KEY, "--registry-root", str(root), "--spool", str(root / "records"), *flags)
            assert r.returncode == 2, (column, r.stdout[-1500:], r.stderr[-1500:])
            assert f"REFUSED: {KEY}/arditi__ar_c0p3__arditi_allpos: discrepancy {column}:" in r.stderr and "live" in r.stderr and "committed" in r.stderr, r.stderr
            assert r.stdout.count("skip (already stated)") == 8 and "emitted " not in r.stdout, r.stdout
        assert {str(p) for p in (root / "records").rglob("*") if p.is_file()} == before
    finally:
        conn = DB.connect(DB.db_path(str(root)))
        conn.execute("PRAGMA foreign_keys=OFF"); conn.execute("PRAGMA ignore_check_constraints=ON")
        conn.execute(f"UPDATE object SET {column} = ? WHERE object_id = ?", (old, oid)); conn.commit(); conn.close()


@pytest.mark.parametrize("live_value", ["a" * 40, None], ids=["a_real_sha", "NULL"])
def test_a_live_fit_row_whose_fit_code_sha_is_not_the_stated_unverified_refuses_by_name(imported, live_value):
    """The other direction of the cross-check: a fit's statement says 'unverified'; a live row that reads a real sha, or NULL, is a
    discrepancy named fit_code_sha — dry run and real run alike, nothing written; the row is restored afterwards."""
    root = imported
    sid = FIT_SIDS[0]
    oid = _plan_ids()[sid]
    conn = DB.connect(DB.db_path(str(root)))
    try:
        old = conn.execute("SELECT fit_code_sha FROM object WHERE object_id = ?", (oid,)).fetchone()[0]
        assert old == "unverified"
        conn.execute("UPDATE object SET fit_code_sha = ? WHERE object_id = ?", (live_value, oid))
        conn.commit()
    finally:
        conn.close()
    try:
        before = {str(p) for p in (root / "records").rglob("*") if p.is_file()}
        for flags in (("--dry-run",), ()):
            r = _run("registry", "--model", KEY, "--registry-root", str(root), "--spool", str(root / "records"), *flags)
            assert r.returncode == 2, (live_value, r.stdout[-1500:], r.stderr[-1500:])
            assert f"REFUSED: {KEY}/{sid}: discrepancy fit_code_sha: live {live_value!r} vs committed 'unverified'" in r.stderr, r.stderr
            assert r.stdout.count("skip (already stated)") == 8 and "emitted " not in r.stdout, r.stdout
        assert {str(p) for p in (root / "records").rglob("*") if p.is_file()} == before
    finally:
        conn = DB.connect(DB.db_path(str(root)))
        conn.execute("UPDATE object SET fit_code_sha = ? WHERE object_id = ?", (old, oid)); conn.commit(); conn.close()


def test_the_comparison_treats_an_absent_statement_against_a_stated_live_value_as_a_discrepancy():
    """The first-draft mapping's two silent divergences (plan review rounds 1 and 2), pinned on the comparison itself: a live base_scale 0.3
    against a statement with NO base_scale; a live layer list against NO list; a live convention against an ABSENT one (the explicit-list-
    without-the-keyword case); and a different list / convention. Equal statements compare equal through the JSON text."""
    live = {"object_id": "x", "campaign": "arditi_allpos", "object_kind": "vector_all", "paper_label": "arditi_allpos", "object_role": "baseline_published",
            "fit_target_kind": "contrastive_direction", "fit_objective": "arditi_search", "derivation": "none", "fit_model_id": "qwen35_4b_it", "status": "approved",
            "parent_object_ids_json": None, "base_scale": 0.3, "fit_layers_json": "[12]", "fit_layer_index_convention": "zero_based_decoder_block"}
    same = dict(live, fit_layers_json=[12])
    assert IO.identity_discrepancies(same, live) == []
    assert [c for c, _a, _b in IO.identity_discrepancies({k: v for k, v in same.items() if k != "base_scale"}, live)] == ["base_scale"]
    assert [c for c, _a, _b in IO.identity_discrepancies({k: v for k, v in same.items() if k != "fit_layers_json"}, live)] == ["fit_layers_json"]
    assert [c for c, _a, _b in IO.identity_discrepancies({k: v for k, v in same.items() if k != "fit_layer_index_convention"}, live)] == ["fit_layer_index_convention"]
    assert [c for c, _a, _b in IO.identity_discrepancies(dict(same, fit_layers_json=[13]), live)] == ["fit_layers_json"]
    assert [c for c, _a, _b in IO.identity_discrepancies(dict(same, fit_layer_index_convention="one_based"), live)] == ["fit_layer_index_convention"]
    derived_live = dict(live, parent_object_ids_json=json.dumps(["a" * 64, "b" * 64]), base_scale=None, fit_layers_json="[3,7,11]")
    assert IO.identity_discrepancies(dict(derived_live, parent_object_ids_json=["a" * 64, "b" * 64], fit_layers_json=[3, 7, 11]), derived_live) == []
    assert [c for c, _a, _b in IO.identity_discrepancies(dict(derived_live, parent_object_ids_json=["b" * 64, "a" * 64], fit_layers_json=[3, 7, 11]), derived_live)] == ["parent_object_ids_json"]
    no_list = {k: v for k, v in dict(derived_live, parent_object_ids_json=["a" * 64, "b" * 64]).items() if k != "fit_layers_json"}
    assert [c for c, _a, _b in IO.identity_discrepancies(no_list, derived_live)] == ["fit_layers_json"]   # NULL (the derived-object-without-the-keyword case) vs the live list
    # fit_code_sha is an identity column: the direction's NULL against an unstated column is equal, against a stated 'unverified' is not;
    # a fit's live 'unverified' against the stated 'unverified' is equal, against an unstated column, a NULL or a real sha is not
    assert "fit_code_sha" in IO.IDENTITY_COLUMNS
    assert [c for c, _a, _b in IO.identity_discrepancies(dict(same, fit_code_sha="unverified"), live)] == ["fit_code_sha"]
    assert [c for c, _a, _b in IO.identity_discrepancies(same, dict(live, fit_code_sha="unverified"))] == ["fit_code_sha"]
    fit_live = dict(derived_live, fit_code_sha="unverified")
    fit_same = dict(fit_live, parent_object_ids_json=["a" * 64, "b" * 64], fit_layers_json=[3, 7, 11])
    assert IO.identity_discrepancies(fit_same, fit_live) == []
    assert [c for c, _a, _b in IO.identity_discrepancies({k: v for k, v in fit_same.items() if k != "fit_code_sha"}, fit_live)] == ["fit_code_sha"]
    assert [c for c, _a, _b in IO.identity_discrepancies(dict(fit_same, fit_code_sha="a" * 40), fit_live)] == ["fit_code_sha"]
    assert [c for c, _a, _b in IO.identity_discrepancies(fit_same, dict(fit_live, fit_code_sha=None))] == ["fit_code_sha"]
    assert [c for c, _a, _b in IO.identity_discrepancies(fit_same, dict(fit_live, fit_code_sha="a" * 40))] == ["fit_code_sha"]


@pytest.mark.parametrize("sid", sorted(EXPECTED_LAYERS))
def test_the_actual_hook_payload_of_each_committed_object_carries_both_layer_columns(sid):
    """The (payload, artifact) hooks.object_from_sidecar returns for the importer's REAL call — not a fixture row — carries fit_layers_json
    equal to the committed sidecar's list and fit_layer_index_convention zero_based_decoder_block for every one of the nine; the five
    explicit-layer rows (both differences, both copies, the refusal direction) are each a named case, since the pre-M1b hook stated their list
    and NO convention."""
    plan = IO.Plan(KEY, repo=REPO)
    row = next(r for r in plan.rows if r["staged_id"] == sid)
    kw = plan.statement(row)
    payload, _art = hooks.object_from_sidecar(str(REPO / "objects" / row["dir"]), producer=hooks.emit.producer_block(IO.SCRIPT), **kw)
    assert payload["fit_layers_json"] == EXPECTED_LAYERS[sid] and payload["fit_layer_index_convention"] == "zero_based_decoder_block", (sid, payload.get("fit_layers_json"), payload.get("fit_layer_index_convention"))
    assert payload["fit_layer_count"] == len(EXPECTED_LAYERS[sid]) and payload["status"] == "approved"
    if sid in EXPLICIT_LAYER_ROWS:
        assert kw["fit_layer_index_convention"] == "zero_based_decoder_block" and kw["fit_layers_json"] == EXPECTED_LAYERS[sid]
        fc = payload["field_confidence_json"]
        assert fc["fit_layer_index_convention"]["confidence"] == "recorded" and fc["fit_layer_index_convention"]["source"] in ("derived_meta.json:layers", "fit_provenance.json:arditi_allpos.served_layer")
    else:
        assert "fit_layer_index_convention" not in kw and "fit_layers_json" not in kw      # the hook reads the fits' lists from train_budget_meta.json


# ------------------------------------------------------------------------------------------------------------ (i): batch + provenance
def test_the_statement_never_carries_the_importing_context_whatever_the_environment_says(monkeypatch, tmp_path):
    """The hook fills fit_code_sha / fit_substrate / fit_job_id / fit_node_hostname / fit_producer_session from the EMITTING
    context unless the caller states them, and no committed directory carries fit_job_provenance.json — so an import used to file the paper's
    objects as fitted on the importing host, checkout and job. Every statement now passes an explicit fit_provenance: the previewed payload
    (the hook's real call, under a synthetic job + session + host) carries none of the five from the context — job, host, substrate and
    session unstated on every row; fit_code_sha the STATED 'unverified' on the eight fits, differences and copies (the committed sidecars
    record no fit code sha; the hook writes the field_confidence_json entry the harvest requires beside it) and unstated on the
    refusal direction — and states fit_finished_utc only where the committed sidecar records a time (a difference's built_at, a copy's own
    written_at, the direction's written_at; never for a fit, never the copied parent's built_at). The fits' column fit_effective_batch is the
    sidecar's 64. The guard itself refuses a payload that carries the context, the micro-batch, a fit_code_sha other than the stated one, or
    a stated 'unverified' without its entry."""
    import socket
    for k, v in IMPORT_CONTEXT.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(socket, "gethostname", lambda: "synthetic-import-host")
    monkeypatch.setattr(hooks.emit, "git_sha", lambda *a, **k: "f" * 40)
    plan = IO.Plan(KEY, repo=REPO)
    previews = plan.previews(spool=None, registry_root=str(tmp_path / "no-registry-here"))
    assert len(previews) == 9
    for r, kw, payload in previews:
        sid = r["staged_id"]
        stated_sha = None if sid == ARDITI_SID else "unverified"
        assert kw["fit_provenance"]["fit_code_sha"] == stated_sha and all(kw["fit_provenance"][c] is None for c in UNSTATED_COLUMNS), (sid, kw["fit_provenance"])
        assert "_source" not in kw["fit_provenance"]                                              # never the hook's "replay restating a prior record" door
        assert not any(c in payload for c in UNSTATED_COLUMNS), (sid, {c: payload.get(c) for c in UNSTATED_COLUMNS})
        fc = payload.get("field_confidence_json") or {}
        if stated_sha is None:
            assert "fit_code_sha" not in payload and "fit_code_sha" not in fc, sid                 # the direction: unstated, nothing to confide
        else:
            assert payload["fit_code_sha"] == "unverified" and fc["fit_code_sha"] == CODE_SHA_ENTRY, (sid, fc.get("fit_code_sha"))
        for val in ("synthetic-import-host", "synthetic-import-session", "424242", "f" * 40):
            assert val not in json.dumps(payload), (sid, val)
        fp = payload["extra"]["fit_provenance"]
        assert fp["written_by"] == IO.SCRIPT and "not recorded in the committed sidecars" in fp["note"] and "fit_code_sha" in fp["sources"]
        assert ("stated 'unverified'" in fp["sources"]["fit_code_sha"]) == (stated_sha is not None), (sid, fp["sources"]["fit_code_sha"])
        assert ("fit_code_sha" in fp["note"]) == (stated_sha is None), (sid, fp["note"])          # the note lists the UNSTATED columns
        side = r["identity_sidecar"]
        doc = json.loads((REPO / "objects" / r["dir"] / side).read_text())
        if sid in FIT_SIDS:
            assert payload["fit_effective_batch"] == doc["effective_batch"] == 64 and doc["resolved_batch_size"] == 2 and "fit_finished_utc" not in payload
            assert "effective_batch" not in kw["extra"]                                           # the column is the statement, not an extra
        elif r["lane"] == "d0p75":
            assert payload["fit_finished_utc"] == IO._utc_ms(doc["dose_scaled_from"]["written_at"], sid) and "fit_effective_batch" not in payload
            assert payload["fit_finished_utc"] != IO._utc_ms(doc["built_at"], sid)              # Never the copied parent's built_at
        elif r["lane"] == "derived":
            assert payload["fit_finished_utc"] == IO._utc_ms(doc["built_at"], sid) and "fit_effective_batch" not in payload
        else:
            assert payload["fit_finished_utc"] == IO._utc_ms(doc["arditi_allpos"]["written_at"], sid) and payload["fit_writer"] == "scripts/arditi/fit_arditi.py"
    # the guard: a payload carrying the context, or a fit stated at its micro-batch, refuses by name before any write
    r0, kw0, p0 = next(x for x in previews if x[0]["staged_id"] == FIT_SIDS[0])
    with pytest.raises(SystemExit, match="states fit_node_hostname='synthetic-import-host' from the importing context"):
        IO.check_statement_provenance(dict(p0, fit_node_hostname="synthetic-import-host"), kw0, "x", effective_batch=64)
    with pytest.raises(SystemExit, match="states fit_effective_batch=2 but train_budget_meta.json records effective_batch 64"):
        IO.check_statement_provenance(dict(p0, fit_effective_batch=2), kw0, "x", effective_batch=64)
    # fit_code_sha on a fit (stated 'unverified'): the real preview passes; a real sha, a missing column, or a stated 'unverified' whose
    # field_confidence_json entry is missing (the harvest would refuse it) refuses by name before any write
    IO.check_statement_provenance(p0, kw0, "x", effective_batch=64)
    assert kw0["fit_provenance"]["fit_code_sha"] == "unverified" == p0["fit_code_sha"]
    with pytest.raises(SystemExit, match="states fit_code_sha='a+', the statement 'unverified' — not written"):
        IO.check_statement_provenance(dict(p0, fit_code_sha="a" * 40), kw0, "x", effective_batch=64)
    with pytest.raises(SystemExit, match="states fit_code_sha=None, the statement 'unverified' — not written"):
        IO.check_statement_provenance({k: v for k, v in p0.items() if k != "fit_code_sha"}, kw0, "x", effective_batch=64)
    without_entry = {k: v for k, v in p0["field_confidence_json"].items() if k != "fit_code_sha"}
    with pytest.raises(SystemExit, match=r"states fit_code_sha='unverified' with no field_confidence_json\['fit_code_sha'\] entry .* which the registry would refuse at harvest"):
        IO.check_statement_provenance(dict(p0, field_confidence_json=without_entry), kw0, "x", effective_batch=64)
    with pytest.raises(SystemExit, match=r"with no field_confidence_json\['fit_code_sha'\] entry"):
        IO.check_statement_provenance(dict(p0, field_confidence_json={**without_entry, "fit_code_sha": {"confidence": "unverified", "reason": " "}}), kw0, "x", effective_batch=64)
    # ... and on the refusal direction (unstated): 'unverified' or a sha there is the importing context's, refused as before
    _ra, kwa, pa = next(x for x in previews if x[0]["staged_id"] == ARDITI_SID)
    IO.check_statement_provenance(pa, kwa, "x")
    assert kwa["fit_provenance"]["fit_code_sha"] is None and "fit_code_sha" not in pa
    with pytest.raises(SystemExit, match="states fit_code_sha='unverified' from the importing context, which the statement unstates"):
        IO.check_statement_provenance(dict(pa, fit_code_sha="unverified"), kwa, "x")
    with pytest.raises(SystemExit, match="states fit_code_sha='a+' from the importing context, which the statement unstates"):
        IO.check_statement_provenance(dict(pa, fit_code_sha="a" * 40), kwa, "x")
    assert IO.FIT_CODE_SHA_UNVERIFIED_KINDS == ("fit", "difference", "dose_copy")
    # the timestamp spelling: the schema's millisecond UTC form; a non-UTC offset or no timestamp refuses, never a local-time guess
    assert IO._utc_ms("2026-09-17T13:54:59+00:00", "t") == "2026-09-17T13:54:59.000Z" == IO._utc_ms("2026-09-17T13:54:59Z", "t")
    assert IO._utc_ms("2026-09-17T13:54:59.250Z", "t") == "2026-09-17T13:54:59.250Z"
    for bad in ("2026-09-17T13:54:59+02:00", "2026-09-17 13:54", None, ""):
        with pytest.raises(SystemExit, match="REFUSED"):
            IO._utc_ms(bad, "t")
    assert IO.CONTEXT_PROVENANCE_COLUMNS == tuple(hooks.FIT_PROVENANCE_KEYS[:5]) == CONTEXT_COLUMNS
    # a fit whose sidecar disagrees with objects/DEMO.json's effective batch refuses by name (the entry records 64 for every fit)
    row = dict(next(r for r in plan.rows if r["staged_id"] == FIT_SIDS[0]))
    row["demo"] = dict(row["demo"], effective_batch=8)
    with pytest.raises(SystemExit, match="effective_batch 64 is not objects/DEMO.json's 8"):
        plan.statement(row)


# ------------------------------------------------------------------------------------------------------------ (d): cells
def test_cells_prints_nine_rows_deduplicated_by_staged_id_with_the_two_dose_copies():
    r = _run("cells", "--model", KEY)
    assert r.returncode == 0, r.stderr
    rows = json.loads(r.stdout)
    sids = [f"{x['lane']}__{x['cell_id']}" for x in rows]
    assert len(rows) == 9 and len(set(sids)) == 9 and set(sids) == set(EXPECTED_LAYERS)
    dose = [x for x in rows if x["lane"] == "d0p75"]
    assert len(dose) == 2 and all(x["kind"] == "derived" and x["identity_sidecar"] == "derived_meta.json" for x in dose)
    by_label = {e["paper_label"]: e for e in DEMO["objects"]}
    for x in rows:
        assert x["source"].startswith("committed:objects/") and (REPO / x["source"][len("committed:"):] / "object.pt").is_file()
        assert x["object_sha256_expected"] == x["source"].rsplit("/", 1)[1]               # the directory IS the object id
        assert x["object_sha256_expected"] in MANIFEST and MANIFEST[x["object_sha256_expected"]]["model_id"] == x["fit_model_id"]
        assert x["identity_sidecar"] in IO.IDENTITY_SIDECARS and (REPO / x["source"][len("committed:"):] / x["identity_sidecar"]).is_file()
        if x["lane"] != "d0p75":
            e = by_label[MT.model(KEY)["objects"][x["view_cell"]].get("paper_label") or x["view_cell"]]
            assert x["object_sha256_expected"] == e["object_id"] and x["seed"] == e["seed"] and x["object_sha256_expected_source"] == f"objects/DEMO.json:{e['paper_label']}"
        else:
            assert "dose 0.75 copy of objects/DEMO.json:" in x["object_sha256_expected_source"]
    # the arditi row: the served checkpoint's object, lane arditi, kind derived (fit_provenance.json is its identity)
    ar = next(x for x in rows if x["lane"] == "arditi")
    assert ar["fit_model_id"] == "qwen35_4b_it" and ar["kind"] == "derived" and ar["identity_sidecar"] == "fit_provenance.json" and ar["seed"] is None
    # the one wave alone: seven rows, no copy; a dose view alone: its copy and nothing else
    r = _run("cells", "--model", KEY, "--views", MT.baseline_view(KEY))
    assert r.returncode == 0 and len(json.loads(r.stdout)) == 7 and not any(x["lane"] == "d0p75" for x in json.loads(r.stdout))
    r = _run("cells", "--model", KEY, "--views", "d0p75")
    assert r.returncode == 0 and [f"{x['lane']}__{x['cell_id']}" for x in json.loads(r.stdout)] == ["d0p75__constv2-mix-full__lora_r4_ce_minus_ctl"]
    assert IO.IDENTITY_SIDECARS == tuple(__import__("sisteer.eval.train_budget", fromlist=["IDENTITY_SIDECARS"]).IDENTITY_SIDECARS)


def test_a_row_without_a_committed_source_and_a_wrong_model_refuse_by_name(tmp_path):
    import yaml
    table = yaml.safe_load((REPO / "reproduce" / "models.yaml").read_text())
    row = table["models"][KEY]
    stripped = json.loads(json.dumps(table)); stripped["models"][KEY].pop("committed_objects_of")
    for o in stripped["models"][KEY]["objects"].values():
        if isinstance(o, dict):
            o.pop("committed_objects_of", None)
    p = tmp_path / "nocommitted.yaml"; p.write_text(yaml.safe_dump(stripped, sort_keys=False))
    r = _run("cells", "--model", KEY, "--table", str(p))
    assert r.returncode == 2 and "declares no `committed_objects_of`" in r.stderr and r.stdout == "", r.stderr
    # the refusal direction's committed model must be the served checkpoint's: naming the fit's own model key refuses on DEMO.json's model_id
    wrong = json.loads(json.dumps(table)); wrong["models"][KEY]["objects"]["refusal_direction"].pop("committed_objects_of")
    p = tmp_path / "wrong.yaml"; p.write_text(yaml.safe_dump(wrong, sort_keys=False))
    r = _run("cells", "--model", KEY, "--table", str(p))
    assert r.returncode == 2 and "is a 'qwen35_4b_it' object but the table's committed_objects_of 'qwen35_4b' names 'qwen35_4b_base'" in r.stderr, r.stderr
    assert row["committed_objects_of"] == "qwen35_4b" and row["objects"]["refusal_direction"]["committed_objects_of"] == "qwen35_4b_it"


# ------------------------------------------------------------------------------------------------------------ (e): pins
PINS_SRC = REPO / "configs" / "sdfbase_full22_pins.json"
DERIVED_SID, COPY_SID = "derived__constv2-mix-full__lora_r4_ce_minus_ctl", "d0p75__constv2-mix-full__lora_r4_ce_minus_ctl"


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    """The REAL stager over the nine committed rows (`import_objects.py cells`) through a one-cluster local site: the staging manifest 05's
    pins step reads (its rows carry `source_path: committed:…`, the stager's output schema)."""
    tmp = tmp_path_factory.mktemp("staged")
    site, _shared = _site(tmp)
    cells = json.loads(_run("cells", "--model", KEY).stdout)
    bank = tmp / "bank"; cf = tmp / "cells.json"; cf.write_text(json.dumps(cells))
    r = subprocess.run([PY, str(REPO / "scripts/sdfbase/full22/stage_objects.py"), "--site-dir", str(site), "--bank", str(bank), "--target", KEY,
                        "--cells", str(cf), "--clusters", "cluster_a", "--no-manifest"], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    m = bank / "staging_manifest.json"
    rows = SO.rebuild_manifest(bank / "_records", m)
    assert len(rows) == 9 and all(str(x.get("source_path", "")).startswith("committed:objects/") and x["status"] == "staged" for x in rows)
    return m


def _pins_copy(tmp_path: Path) -> Path:
    copy = tmp_path / "pins.json"; shutil.copy(PINS_SRC, copy)
    return copy


def _set(root: Path, oid: str, column: str, value):
    conn = DB.connect(DB.db_path(str(root)))
    try:
        conn.execute("PRAGMA foreign_keys=OFF"); conn.execute("PRAGMA ignore_check_constraints=ON")
        old = conn.execute(f"SELECT {column} FROM object WHERE object_id = ?", (oid,)).fetchone()[0]
        conn.execute(f"UPDATE object SET {column} = ? WHERE object_id = ?", (value, oid)); conn.commit()
        return old
    finally:
        conn.close()


def test_the_committed_demo_pins_block_is_exactly_the_committed_objects_ids():
    """configs/sdfbase_full22_pins.json[qwen35_4b_demo] pins every staged id the import plans, and nothing else, to the object id objects/DEMO.json
    names (Plan.expected_pins): a DEMO.json that moves without its pins (the one-seed list, 2026-10-12, changed four fits) fails here, not at a stage."""
    block = json.loads(PINS_SRC.read_text(encoding="utf-8"))[KEY]
    assert {sid: e["object_sha256"] for sid, e in block.items()} == IO.Plan(KEY, repo=REPO).expected_pins()


def test_pins_writes_the_demo_block_through_the_registry_backed_writer_with_every_other_key_byte_identical(imported, staged, tmp_path):
    root, man = imported, staged
    copy = _pins_copy(tmp_path)
    committed = json.loads(PINS_SRC.read_text(encoding="utf-8"))
    r = _run("pins", "--model", KEY, "--pins-file", str(copy), "--manifest", str(man), "--registry-root", str(root))
    assert r.returncode == 0, r.stdout[-2500:] + r.stderr[-2500:]
    written = json.loads(copy.read_text(encoding="utf-8"))
    assert set(written) == set(committed) and {k: v for k, v in written.items() if k != KEY} == {k: v for k, v in committed.items() if k != KEY}
    block = written[KEY]
    assert set(block) == set(EXPECTED_LAYERS) and all(block[s]["object_sha256"] == _plan_ids()[s] for s in block)
    # the writer's own lines: every committed row held to the registry; the plain rows re-spelled from the manifest, the registry-owned lanes
    # (differences, copies) kept in the committed entry's spelling when the id is the committed one
    assert r.stdout.count("registry-backed by its own row") == 4 and f"pinned under {KEY}, registry-backed at {root}" in r.stdout, r.stdout
    assert all(block[s]["source"].startswith("committed:objects/") for s in block if s.split("__", 1)[0] in ("base", "arditi")), block
    assert block[DERIVED_SID] == committed[KEY][DERIVED_SID] and block[COPY_SID] == committed[KEY][COPY_SID]
    assert all(e["sidecar_object_sha256_recorded"] == "match" for s, e in block.items() if s.split("__", 1)[0] in ("base", "arditi"))
    # round-trip through the generator's loader, and its --check stays green with the block in place (the demo has no roster block: no output reads it)
    gen = _load_gen()
    gen.PINS = copy
    assert gen.load_pins()[KEY] == block
    assert gen.main(["--check"]) == 0
    # the dry run validates the same way and writes nothing
    before = copy.read_text()
    r = _run("pins", "--model", KEY, "--pins-file", str(copy), "--manifest", str(man), "--registry-root", str(root), "--dry-run")
    assert r.returncode == 0 and "DRY-RUN: would write" in r.stdout and copy.read_text() == before, r.stdout[-1500:] + r.stderr[-1500:]


def test_pins_refuses_by_name_without_the_registry_rows_the_manifest_or_the_registry(staged, tmp_path):
    """The spec's rule for the committed pins writer — "never the status checks": a digest the registry holds no reference|approved row for
    is not pinned, and a block that comes back short is REFUSED (import the committed objects first), nothing written."""
    man = staged
    copy = _pins_copy(tmp_path); before = copy.read_text()
    seeds_only = _fresh_registry(tmp_path / "seeds_only")                          # the registry a researcher has before `import_objects.py registry`
    r = _run("pins", "--model", KEY, "--pins-file", str(copy), "--manifest", str(man), "--registry-root", str(seeds_only))
    assert r.returncode == 2, r.stdout[-2000:] + r.stderr[-2000:]
    assert "REFUSED:" in r.stderr and "has no reference/approved registry row" in r.stderr and "import the committed objects first" in r.stderr, r.stderr[-2000:]
    assert copy.read_text() == before
    # no manifest: nothing to hold the pins to; a manifest path that is not there; a registry that is not on disk
    r = _run("pins", "--model", KEY, "--pins-file", str(copy), "--registry-root", str(seeds_only))
    assert r.returncode == 2 and "pass --manifest" in r.stderr and copy.read_text() == before, r.stderr
    r = _run("pins", "--model", KEY, "--pins-file", str(copy), "--manifest", str(tmp_path / "nope.json"), "--registry-root", str(seeds_only))
    assert r.returncode == 2 and "no staging manifest here" in r.stderr and copy.read_text() == before, r.stderr
    r = _run("pins", "--model", KEY, "--pins-file", str(copy), "--manifest", str(man), "--registry-root", str(tmp_path / "no-registry"))
    assert r.returncode == 2 and "is not on disk" in r.stderr and copy.read_text() == before, r.stderr
    # a manifest digest that is not the committed id: the writer's digest refusal (the stager's expectation vs its digest), carried
    rows = json.loads(man.read_text())
    bad = tmp_path / "bad_manifest.json"
    bad.write_text(json.dumps([dict(x, object_sha256="b" * 64) if x["staged_id"] == DERIVED_SID else x for x in rows]))
    r = _run("pins", "--model", KEY, "--pins-file", str(copy), "--manifest", str(bad), "--registry-root", str(seeds_only))
    assert r.returncode == 2 and "does not match the registered id" in r.stderr and copy.read_text() == before, r.stderr


def test_pins_refuses_a_withdrawn_row_a_copy_of_another_parent_and_a_changed_committed_entry(imported, staged, tmp_path):
    """The Q12 refusals reach the entry point: the digest's own live row must be reference|approved, a copy's live parents the pinned parent,
    and a committed entry's identity never changes from a staging row — each refuses by name through `pins`, the file untouched."""
    root, man = imported, staged
    ids = _plan_ids()
    copy = _pins_copy(tmp_path); before = copy.read_text()
    old = _set(root, ids[DERIVED_SID], "status", "withdrawn")
    try:
        r = _run("pins", "--model", KEY, "--pins-file", str(copy), "--manifest", str(man), "--registry-root", str(root))
        assert r.returncode == 2 and "has no reference/approved registry row (status 'withdrawn')" in r.stderr and copy.read_text() == before, r.stderr[-2000:]
    finally:
        _set(root, ids[DERIVED_SID], "status", old)
    old = _set(root, ids[COPY_SID], "parent_object_ids_json", json.dumps(["f" * 64]))
    try:
        r = _run("pins", "--model", KEY, "--pins-file", str(copy), "--manifest", str(man), "--registry-root", str(root))
        assert r.returncode == 2 and f"names parents ['{'f' * 12}'], the pinned parent is {ids[DERIVED_SID][:12]}" in r.stderr and copy.read_text() == before, r.stderr[-2000:]
    finally:
        _set(root, ids[COPY_SID], "parent_object_ids_json", old)
    old = _set(root, ids[COPY_SID], "base_scale", 0.5)
    try:
        r = _run("pins", "--model", KEY, "--pins-file", str(copy), "--manifest", str(man), "--registry-root", str(root))
        assert r.returncode == 2 and "is a copy at base_scale 0.5, the view's dose is 0.75" in r.stderr and copy.read_text() == before, r.stderr[-2000:]
    finally:
        _set(root, ids[COPY_SID], "base_scale", old)
    # a committed entry whose id is not the staged digest: a committed pin's identity changes only through its registry re-pin (S2RLFIX2)
    pins = json.loads(before); pins[KEY][DERIVED_SID]["object_sha256"] = "c" * 64
    changed = tmp_path / "changed.json"; changed.write_text(json.dumps(pins, indent=1, sort_keys=True) + "\n")
    r = _run("pins", "--model", KEY, "--pins-file", str(changed), "--manifest", str(man), "--registry-root", str(root))
    assert r.returncode == 2 and "would CHANGE the identity of 1 committed entry" in r.stderr and json.loads(changed.read_text()) == pins, r.stderr[-2000:]
    # ... and the registry restored, the write goes through again
    r = _run("pins", "--model", KEY, "--pins-file", str(copy), "--manifest", str(man), "--registry-root", str(root))
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]


def test_the_scope_is_what_lets_a_one_lane_registry_write_that_lanes_block(imported, staged, tmp_path):
    """`targets` on pins_from_staging_manifest (the scope `pins` passes): the same manifest and the same registry — which holds the demo's
    nine objects and NOTHING of any other key — write the demo block with the other keys' committed entries kept verbatim; without the scope
    the whole-file write asks the registry to stand behind every OTHER front-door key's committed derived pins and refuses on the first (that
    rule is right for `--pins-from` over a bank manifest, wrong for a lane's own import). The committed table carries ONE front-door key,
    so the other key is synthetic: the demo row under a second key, with a committed derived pin the registry lacks."""
    import copy
    root, man = imported, staged
    gen = _load_gen()
    committed = json.loads(PINS_SRC.read_text(encoding="utf-8"))
    roster = IO._roster_for(IO.Plan(KEY, repo=REPO), gen, None)
    scoped = gen.pins_from_staging_manifest(man, registry_root=root, roster=roster, committed=committed, targets={KEY})
    assert {k: v for k, v in scoped.items() if k != KEY} == {k: v for k, v in committed.items() if k != KEY}
    assert {s: e["object_sha256"] for s, e in scoped[KEY].items()} == _plan_ids()
    # the second front-door key: the demo's row and roster block under `other_fd`, whose committed difference pin names a digest no registry holds
    table = yaml.safe_load((REPO / "reproduce" / "models.yaml").read_text(encoding="utf-8"))
    other = copy.deepcopy(table["models"][KEY]); other["view_legacy_prefix"] = "other_fd"
    table["models"]["other_fd"] = other
    tpath = tmp_path / "models.yaml"; tpath.write_text(yaml.safe_dump(table, sort_keys=False), encoding="utf-8")
    roster2 = dict(roster, targets={**roster["targets"], "other_fd": copy.deepcopy(roster["targets"][KEY])})
    committed2 = dict(committed, other_fd={"derived__constv2-mix-full__lora_r4_ce_minus_ctl": {"object_sha256": "0" * 64, "source": "objects/DEMO.json:difference_lora"}})
    with pytest.raises(SystemExit) as exc:
        gen.pins_from_staging_manifest(man, registry_root=root, roster=roster2, committed=committed2, models_table=tpath)
    msg = str(exc.value)
    assert "NO reference/approved registry row" in msg and msg.startswith("other_fd/") and not msg.startswith(KEY), msg   # the other key's pin, not the demo's
    scoped3 = gen.pins_from_staging_manifest(man, registry_root=root, roster=roster2, committed=committed2, models_table=tpath, targets={KEY})
    assert scoped3["other_fd"] == committed2["other_fd"] and scoped3[KEY] == scoped[KEY]                    # the one-lane write: the other key kept verbatim
    # the scope filters the manifest's rows too: a row of another target in the same manifest is neither read nor written
    other = [dict(x, target="llama8b") for x in json.loads(man.read_text())[:1]]
    mixed = man.parent / "mixed.json"; mixed.write_text(json.dumps(json.loads(man.read_text()) + other))
    scoped2 = gen.pins_from_staging_manifest(mixed, registry_root=root, roster=roster, committed=committed, targets={KEY})
    assert scoped2 == scoped


def _load_gen():
    spec = importlib.util.spec_from_file_location("gen_for_import_test", REPO / "scripts" / "sdfbase" / "make_full22_configs.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------------------------------------ (f): the mapping
def test_the_mapping_restates_the_manifest_and_the_refusal_direction_is_the_adapters_statement():
    plan = IO.Plan(KEY, repo=REPO)
    assert len(plan.rows) == 9
    for r in plan.rows:
        kw = plan.statement(r)
        man = MANIFEST[r["object_sha256_expected"]]
        assert (kw["campaign"], kw["object_kind"], kw["paper_label"], kw["object_role"], kw["derivation"]) == \
            (man["campaign"], man["object_kind"], man["paper_label"], man["object_role"], man["derivation"]), r["staged_id"]
        assert kw["status"] == "approved" and kw["fit_model_id"] == man["model_id"] and kw["fit_model_id"] != "qwen35_4b_demo_base"
        if r["lane"] == "d0p75":
            assert kw["derivation"] == "dose_scale" and kw["parent_object_ids_json"] == [r["dose_parent"]] and "base_scale" not in kw
            assert r["dose_parent"] == next(e["object_id"] for e in DEMO["objects"] if e["paper_label"] == man["paper_label"])
        elif r["lane"] == "derived":
            dm = json.loads((REPO / "objects" / r["dir"] / "derived_meta.json").read_text())
            assert kw["parent_object_ids_json"] == [dm["parent_pos"]["object_sha256"], dm["parent_neg"]["object_sha256"]] and "base_scale" not in kw
            assert kw["fit_layers_json"] == dm["layers"] and kw["fit_layer_index_convention"] == "zero_based_decoder_block"
        elif r["lane"] == "base":
            assert kw["fit_target_kind"] == "corpus_document_fit" and kw["fit_objective"] == "ce" and kw["fit_thinking_render"] == "not_applicable"
            assert kw["fit_corpus_id"] == r["cell_id"].split("__")[0] and "effective_batch" not in kw["extra"] and "base_scale" not in kw
            assert r["effective_batch"] == 64 == json.loads((REPO / "objects" / r["dir"] / "train_budget_meta.json").read_text())["effective_batch"]
        else:
            prov = json.loads((REPO / "objects" / r["dir"] / "fit_provenance.json").read_text())["arditi_allpos"]
            assert kw["base_scale"] == prov["c"] == 0.3 and kw["fit_layers_json"] == [prov["served_layer"]] == [prov["stored_layer"]] == [12]
            assert kw["fit_layer_index_convention"] == "zero_based_decoder_block" and kw["extra"]["fit_layer"] == prov["fit_layer"] == 13
            assert kw["fit_target_kind"] == "contrastive_direction" and kw["fit_objective"] == "arditi_search" and kw["fit_model_id"] == "qwen35_4b_it"
            assert kw["fit_contrast_sets_json"] == {"unverified": "the battery record names no capture-set ids or shas"} and kw["campaign"] == "arditi_allpos"
        # fit_code_sha: the eight state the schema's 'unverified', the direction unstates it; the host is unstated on every one
        assert kw["fit_provenance"]["fit_code_sha"] == (None if r["lane"] == "arditi" else "unverified") and kw["fit_provenance"]["fit_node_hostname"] is None, r["staged_id"]
    # the layer convention is never a conversion of fit_layer: a provenance whose served and stored layers disagree, or carry neither, refuses by name
    prov = json.loads((REPO / "objects" / "qwen35_4b_it" / _plan_ids()["arditi__ar_c0p3__arditi_allpos"] / "fit_provenance.json").read_text())
    entry = next(e for e in DEMO["objects"] if e["paper_label"] == "arditi_allpos")
    bad = json.loads(json.dumps(prov)); bad["arditi_allpos"]["stored_layer"] = 13
    with pytest.raises(SystemExit, match="served_layer 12 != stored_layer 13"):
        IO.arditi_statement(Path("x"), bad, bad["arditi_allpos"], entry, {})
    neither = json.loads(json.dumps(prov)); neither["arditi_allpos"].pop("served_layer"); neither["arditi_allpos"].pop("stored_layer")
    with pytest.raises(SystemExit, match="served_layer is None, not an integer — no stated served layer and no stated convention to convert fit_layer by"):
        IO.arditi_statement(Path("x"), neither, neither["arditi_allpos"], entry, {})
    off = json.loads(json.dumps(prov)); off["arditi_allpos"]["c"] = 0.5
    with pytest.raises(SystemExit, match="c=0.5 is not objects/DEMO.json's dose 0.3"):
        IO.arditi_statement(Path("x"), off, off["arditi_allpos"], entry, {})


# ------------------------------------------------------------------------------------------------------------ (g): resolve
def test_sisteer_objects_resolves_each_committed_address_to_the_planned_id():
    plan = IO.Plan(KEY, repo=REPO)
    for r in plan.rows:
        e = r["demo"]
        if r["lane"] == "d0p75":
            got = resolve(e["model_id"], e["paper_label"], dose=0.75, seed=e["seed"], repo_root=REPO)
        elif r["lane"] == "arditi":
            got = resolve(e["model_id"], e["paper_label"], dose=e["dose"], repo_root=REPO)
        else:
            got = resolve(e["model_id"], e["paper_label"], seed=e["seed"], repo_root=REPO)
        assert got == r["object_sha256_expected"], r["staged_id"]


# ------------------------------------------------------------------------------------------------------------ (h): the banks
def test_the_three_demo_banks_sit_outside_every_objects_bank_scan_root():
    from sisteer.capeval.tinycap import contracts as C
    roots = [os.path.realpath(b[1]) for b in OB.DEFAULT_BANKS]
    state = os.path.expanduser("~/.local/state/sisteer-reproduce/objects")
    banks = [state, f"/scratch/example/shared/exp/objects/sdfbase/full22/{KEY}", C.MODELS[KEY].objects_dir]
    for b in banks:
        rb = os.path.realpath(b)
        assert not any(rb == r or rb.startswith(r + "/") for r in roots), (b, roots)
    assert C.MODELS[KEY].objects_dir == f"{C.VOL_OBJECTS_SDFBASE}/{KEY}" and not C.MODELS[KEY].objects_dir.startswith(f"{C.VOL_OBJECTS_SDFBASE}/full22")


# ------------------------------------------------------------------------------------------------------------ the grading host verify (read-only)
LIVE = os.path.expanduser("~/registry/registry.sqlite")


@pytest.mark.skipif(not os.path.isfile(LIVE), reason="no grading host registry here")
def test_the_grading_host_dry_run_skips_all_nine_already_stated_with_both_layer_columns_previewed():
    r = _run("registry", "--model", KEY, "--dry-run", "--registry-root", os.path.dirname(LIVE))
    assert r.returncode == 0, r.stdout[-2500:] + r.stderr[-2500:]
    assert r.stdout.count("skip (already stated)") == 9 and "REFUSED:" not in r.stderr and "would emit" not in r.stdout   # the hook's own "would be REFUSED without force" notice is the dry run's
    previews = [ln for ln in r.stdout.splitlines() if ln.startswith("preview ")]
    assert len(previews) == 9 and all("fit_layers_json=[" in ln and "fit_layer_index_convention=zero_based_decoder_block" in ln for ln in previews)
    # fit_code_sha joins the live cross-check — the nine still skip, so the eight stated 'unverified' read the live 'unverified' and the
    # direction's unstated column reads the live NULL (the preview line shows the statement; the skip shows the row agreed)
    assert sum(" fit_code_sha=unverified" in ln for ln in previews) == 8
    assert [ln for ln in previews if " fit_code_sha=None" in ln] == [ln for ln in previews if f"/{ARDITI_SID}:" in ln] and len([ln for ln in previews if f"/{ARDITI_SID}:" in ln]) == 1
