"""Result store tests — row schema round-trip,
cell keys, resume validation with full-hash refusals, marker ordering,
torn-tail repair, consolidate→parquet, config/env hashing."""

import dataclasses
import json
from pathlib import Path

import pytest

from sisteer.capeval import store as cs
from sisteer.capeval.store import CellStore, StoreError

IHASH = "a1b2c3d4e5f60718"
CHASH = "0f1e2d3c4b5a6978"
REV = "0123456789abcdef0123456789abcdef01234567"
KEY = "arm__tier1__t1__a0__a1b2c3d4__0f1e2d3c"  # synthetic cell key, not a credential — gitleaks:allow

RESOLVED = {"intervention_hash": IHASH, "harness_config_hash": CHASH,
            "model_revision": REV, "engine": "hf"}


def _row(item="i0", metric="acc", value=1.0, **kw):
    base = dict(
        sweep="s", run_id="r0", model_id="m", model_revision=REV,
        arm_name="arm", kind="steering", coeff=2.0, intervention_hash=IHASH,
        engine="hf", tier=1, task="gsm8k_platinum_cap", domain="math",
        task_version="v1", item_id=item, trial_id=1, seed=1, attempt=0,
        position_order=None, metric=metric, value=value, extra_json=None,
        harness_config_hash=CHASH, env_digest="99aabbccddeeff00",
        slurm_job_id=None, created_at=1755000000.0, wall_clock_s=None,
    )
    base.update(kw)
    return cs.ResultRow(**base)


# --------------------------------------------------------------------------
# cell keys + ts metrics
# --------------------------------------------------------------------------


def test_cell_key_format():
    assert cs.cell_key("arm", "tier1", 1, 0, IHASH, CHASH) == KEY
    assert cs.cell_key("a", "target", 0, 3, IHASH, CHASH).endswith(
        "__t0__a3__a1b2c3d4__0f1e2d3c")


def test_cell_key_refusals():
    with pytest.raises(StoreError, match="reserved"):
        cs.cell_key("a__b", "tier1", 0, 0, IHASH, CHASH)
    with pytest.raises(StoreError, match="reserved"):
        cs.cell_key("arm", "s2parity/gsm8k", 0, 0, IHASH, CHASH)
    with pytest.raises(StoreError, match="non-negative"):
        cs.cell_key("arm", "tier1", -1, 0, IHASH, CHASH)
    with pytest.raises(StoreError, match="hex"):
        cs.cell_key("arm", "tier1", 0, 0, "short", CHASH)
    with pytest.raises(StoreError, match="hex"):
        cs.cell_key("arm", "tier1", 0, 0, IHASH, "zzzzzzzz11")


def test_ts_metric():
    assert cs.ts_metric("model_ready") == "ts_model_ready"
    with pytest.raises(StoreError):
        cs.ts_metric("")
    with pytest.raises(StoreError):
        cs.ts_metric("two words")


def test_reserved_domains_declared():
    assert cs.META_DOMAIN == "_meta"
    assert cs.TARGET_DOMAIN == "target"
    assert set(cs.RESERVED_DOMAINS) == {"_meta", "target"}


# --------------------------------------------------------------------------
# CellStore lifecycle
# --------------------------------------------------------------------------


def test_row_roundtrip_and_resume(tmp_path):
    store = CellStore(tmp_path, KEY)
    store.write_cell_manifest(RESOLVED)
    store.validate_resume(RESOLVED)  # match -> no-op
    rows = [_row("i0"), _row("i1", value=0.0)]
    for r in rows:
        store.append_row(r)
    # wire format: sorted-key JSON lines that reconstruct ResultRow
    back = [cs.ResultRow(**json.loads(line))
            for line in store.rows_path.read_text().splitlines()]
    assert back == rows
    assert store.completed_item_ids() == {"i0", "i1"}


def test_marker_ordering(tmp_path):
    store = CellStore(tmp_path, KEY)
    store.write_cell_manifest(RESOLVED)
    store.append_row(_row())
    assert not store.is_done()
    store.write_generation_complete()
    assert store.generation_complete()
    assert not store.is_done()  # results.json is the ONLY done marker (F8)
    store.write_results({"n_rows": 1})
    assert store.is_done()
    assert json.loads(store.results_path.read_text())["n_rows"] == 1


def test_resume_mismatch_refusals_name_both_values(tmp_path):
    store = CellStore(tmp_path, KEY)
    store.write_cell_manifest(RESOLVED)
    for field, bad in (("intervention_hash", "ffff0000ffff0000"),
                       ("harness_config_hash", "eeee1111eeee1111"),
                       ("model_revision", "f" * 40),
                       ("engine", "vllm")):
        with pytest.raises(StoreError) as exc:
            store.validate_resume({**RESOLVED, field: bad})
        msg = str(exc.value)
        assert field in msg and bad in msg and str(RESOLVED[field]) in msg


def test_resume_requires_identity_fields(tmp_path):
    store = CellStore(tmp_path, KEY)
    with pytest.raises(StoreError, match="missing"):
        store.validate_resume({"engine": "hf"})
    with pytest.raises(StoreError, match="missing"):
        store.write_cell_manifest({"engine": "hf"})


def test_producer_provenance_is_an_immutable_resume_key_when_the_manifest_has_one(tmp_path):
    """E14 (B): the served driver's ``producer`` block names the tree it ran
    from, and it is NOT part of the hashed identity — so a store keys the
    same at two revisions, and before this the driver appended revision B's
    rows to revision A's and atomically rewrote the manifest with the NEW
    producer: one store holding two revisions, labelled as the newest, which
    the EvalPlus merge's producer gate then trusted. Now revision B refuses
    BEFORE any manifest rewrite. A manifest WITHOUT the block (every legacy
    and HF-path store — no other driver writes one) validates exactly as
    before for a spec that names none; a spec that DOES name a producer may
    adopt only an EMPTY producer-less store (E16 B, PW8): over rows, the
    block would label another revision's measurements as this tree's."""
    from sisteer.capeval.tinycap import contracts as tc

    assert tc.MANIFEST_KEY_PRODUCER in CellStore.CONDITIONAL_RESUME_KEYS
    rev_a = {tc.MANIFEST_KEY_PRODUCER: {"tree_sha": "a" * 40,
                                        "driver": tc.PRODUCER_DRIVER_SERVED}}
    rev_b = {tc.MANIFEST_KEY_PRODUCER: {"tree_sha": "b" * 40,
                                        "driver": tc.PRODUCER_DRIVER_SERVED}}
    store = CellStore(tmp_path, KEY)
    store.write_cell_manifest({**RESOLVED, **rev_a})
    store.append_row(_row("i0"))
    before = store.manifest_path.read_bytes()
    store.validate_resume({**RESOLVED, **rev_a})              # the same revision resumes
    for spec, needle in (({**RESOLVED, **rev_b}, "resume mismatch on 'producer'"),
                         (RESOLVED, "this run's resolved spec names none")):
        with pytest.raises(StoreError, match=needle):
            store.validate_resume(spec)
        with pytest.raises(StoreError, match=needle):
            store.write_cell_manifest(spec)                   # the rewrite is what it prevents
        assert store.manifest_path.read_bytes() == before     # nothing was written
    assert store.completed_item_ids() == {"i0"}               # and nothing was destroyed
    # the same store re-run from ITS tree resumes and may rewrite in place
    store.write_cell_manifest({**RESOLVED, **rev_a, "extra": "fine"})
    assert json.loads(store.manifest_path.read_text())[tc.MANIFEST_KEY_PRODUCER] == rev_a[
        tc.MANIFEST_KEY_PRODUCER]
    # E16 (B): a producer-less manifest over ROWS is a partial store from a
    # revision nobody recorded (a pre-618f9dd tinycap store, say). The direct
    # driver never passes the launcher's store audit, and before this it
    # resumed such a store and wrote the NEW producer block over the old rows.
    # A spec naming a producer now refuses it — before any rewrite; an HF-path
    # spec (no producer) validates and rewrites it exactly as before.
    legacy = CellStore(tmp_path / "legacy", KEY)
    legacy.write_cell_manifest(RESOLVED)
    legacy.append_row(_row("i0"))
    legacy.validate_resume(RESOLVED)                          # the HF caller: unaffected
    legacy.write_cell_manifest({**RESOLVED, "extra": "fine"})  # ...and may rewrite in place
    before = legacy.manifest_path.read_bytes()
    for spec in ({**RESOLVED, **rev_a}, {**RESOLVED, **rev_b}):
        with pytest.raises(StoreError, match="records no 'producer'"):
            legacy.validate_resume(spec)
        with pytest.raises(StoreError, match="records no 'producer'"):
            legacy.write_cell_manifest(spec)
        assert legacy.manifest_path.read_bytes() == before     # nothing was written
    assert tc.MANIFEST_KEY_PRODUCER not in json.loads(legacy.manifest_path.read_text())
    assert legacy.completed_item_ids() == {"i0"}
    # a results.json is content too, rows or not
    finished = CellStore(tmp_path / "finished", KEY)
    finished.write_cell_manifest(RESOLVED)
    finished.write_results({"n_rows": 0})
    with pytest.raises(StoreError, match="records no 'producer'"):
        finished.validate_resume({**RESOLVED, **rev_b})
    # an EMPTY producer-less store (a manifest and nothing else) may GAIN the
    # block: nothing in it can be mislabelled — and from then on it is pinned
    empty = CellStore(tmp_path / "empty", KEY)
    empty.write_cell_manifest(RESOLVED)
    empty.validate_resume(RESOLVED)
    empty.validate_resume({**RESOLVED, **rev_b})
    empty.write_cell_manifest({**RESOLVED, **rev_b})
    assert json.loads(empty.manifest_path.read_text())[tc.MANIFEST_KEY_PRODUCER] == rev_b[
        tc.MANIFEST_KEY_PRODUCER]
    empty.validate_resume({**RESOLVED, **rev_b})
    with pytest.raises(StoreError, match="resume mismatch on 'producer'"):
        empty.validate_resume({**RESOLVED, **rev_a})
    # a torn final line alone is not content: the store is still empty
    torn = CellStore(tmp_path / "torn", KEY)
    torn.write_cell_manifest(RESOLVED)
    torn.rows_path.write_text('{"item_id": "i0", "val')
    torn.validate_resume({**RESOLVED, **rev_b})


def test_content_is_any_durable_artifact_not_just_rows(tmp_path):
    """E17 (B) / PW10: ``_has_content`` counted only rows and results.json,
    so a producer-less store holding ``done/`` / ``partial/`` / ``summary/``
    markers, ``generation_complete.json``, ``results_pending.json``, the
    EvalPlus ``generations.jsonl`` / ``samples.jsonl`` / ``gen_meta.json``
    sidecars, Inspect logs, ``evidence.json`` or ``gen_stats.json`` — and no
    rows — was "empty": the direct driver adopted it, rewrote the manifest
    with the current producer, and the EvalPlus runner REUSED the persisted
    generations under the new revision. Content is now ANY durable artifact
    other than the manifest (walked, not listed), for the producer guard and
    for the missing-manifest guard alike; temp files and a torn-only
    ``rows.jsonl`` are not content."""
    from sisteer.capeval.tinycap import contracts as tc

    rev_b = {tc.MANIFEST_KEY_PRODUCER: {"tree_sha": "b" * 40, "driver": tc.PRODUCER_DRIVER_SERVED}}
    marker = json.dumps({"task": "inspect/math", "n_items": 100, "n_rows": 200, "n_requests": 100,
                         "n_rejected": 0, "wall_clock_s": 1.0, "finished_at": 0.0})
    plants = {
        "done/inspect__math.json": marker,
        "partial/inspect__math.json": marker,
        "summary/inspect__math.json": '{"acc": 0.5, "n": 100}',
        "generation_complete.json": '{"generation_complete": true}',
        "results_pending.json": "{}",
        "evalplus/humaneval/generations.jsonl": '{"item_id": "HumanEval/0", "text": "def f(): pass"}\n',
        "evalplus/humaneval/samples.jsonl": '{"task_id": "HumanEval/0", "solution": "def f(): pass"}\n',
        "evalplus/humaneval/gen_meta.json": '{"task": "evalplus/humaneval_plus"}',
        "private/inspect_logs/2026-09-04T00-00-00_gsm8k.eval": "zip-bytes",
        "evidence.json": "{}",
        "gen_stats.json": "{}",
    }
    for rel, text in plants.items():
        store = CellStore(tmp_path / rel.replace("/", "_"), KEY)
        store.write_cell_manifest(RESOLVED)
        assert store._content_artifacts() == [] and not store._has_content()
        target = store.cell_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        before = store.manifest_path.read_bytes()
        assert store._content_artifacts() == [rel] and store._has_content()
        # a spec naming a producer refuses BEFORE any write, naming the artifact
        with pytest.raises(StoreError, match="records no 'producer'") as ei:
            store.validate_resume({**RESOLVED, **rev_b})
        assert rel in str(ei.value)
        with pytest.raises(StoreError, match="records no 'producer'"):
            store.write_cell_manifest({**RESOLVED, **rev_b})
        assert store.manifest_path.read_bytes() == before
        assert tc.MANIFEST_KEY_PRODUCER not in json.loads(store.manifest_path.read_text())
        # the HF-path caller (no producer) is untouched
        store.validate_resume(RESOLVED)
        store.write_cell_manifest({**RESOLVED, "extra": "fine"})
        # ...and with the manifest MISSING the same artifact refuses a resume, naming it; none is created
        store.manifest_path.unlink()
        with pytest.raises(StoreError, match="manifest.json is missing") as ei:
            store.validate_resume(RESOLVED)
        assert rel in str(ei.value) and not store.manifest_path.exists()
        with pytest.raises(StoreError, match="manifest.json is missing"):
            store.validate_resume({**RESOLVED, **rev_b})
        assert not store.manifest_path.exists()
    # NOT content: a writer's temp file, an editor backup, a torn-only rows.jsonl (repaired to empty)
    empty = CellStore(tmp_path / "empty", KEY)
    empty.write_cell_manifest(RESOLVED)
    (empty.cell_dir / "rows.jsonl.tmp").write_text("{")
    (empty.cell_dir / "manifest.json~").write_text("{}")
    (empty.cell_dir / "summary").mkdir()
    (empty.cell_dir / "summary" / "x.json.tmp").write_text("{")
    empty.rows_path.write_text('{"item_id": "i0", "val')
    assert empty._content_artifacts() == [] and not empty._has_content()
    assert empty.rows_path.read_text() == ""                          # the torn line was repaired away
    empty.validate_resume({**RESOLVED, **rev_b})                      # empty: may gain the block
    empty.write_cell_manifest({**RESOLVED, **rev_b})
    # a real row is content, as before
    empty.append_row(_row("i0"))
    assert empty._content_artifacts() == ["rows.jsonl"]
    # the artifact list is sorted, relative, nested
    many = CellStore(tmp_path / "many", KEY)
    many.write_cell_manifest(RESOLVED)
    for rel in ("done/a.json", "evalplus/mbpp/generations.jsonl", "evidence.json"):
        (many.cell_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        (many.cell_dir / rel).write_text("{}")
    many.append_row(_row("i0"))
    assert many._content_artifacts() == ["done/a.json", "evalplus/mbpp/generations.jsonl",
                                         "evidence.json", "rows.jsonl"]
    assert cs._some(["a", "b"]) == "a, b" and cs._some(list("abcdefgh")) == "a, b, c, d, e, f (+2 more)"


def test_rows_without_manifest_refused(tmp_path):
    store = CellStore(tmp_path, KEY)
    store.append_row(_row())
    with pytest.raises(StoreError, match="manifest.json is missing"):
        store.validate_resume(RESOLVED)


def test_fresh_cell_resume_ok(tmp_path):
    CellStore(tmp_path, KEY).validate_resume(RESOLVED)  # nothing exists yet


def test_manifest_rewrite_same_identity_ok_different_refused(tmp_path):
    store = CellStore(tmp_path, KEY)
    store.write_cell_manifest(RESOLVED)
    store.write_cell_manifest({**RESOLVED, "extra": "fine"})  # same identity
    with pytest.raises(StoreError, match="engine"):
        store.write_cell_manifest({**RESOLVED, "engine": "vllm"})


def test_torn_final_line_repaired(tmp_path):
    store = CellStore(tmp_path, KEY)
    store.write_cell_manifest(RESOLVED)
    store.append_row(_row("i0"))
    store.append_row(_row("i1"))
    # simulate a preemption cutting an append mid-line (no trailing \n)
    with open(store.rows_path, "a") as f:
        f.write('{"item_id": "i2", "val')
    store.validate_resume(RESOLVED)
    assert store.completed_item_ids() == {"i0", "i1"}
    # file was truncated back to whole lines; appends stay parseable
    store.append_row(_row("i2"))
    assert store.completed_item_ids() == {"i0", "i1", "i2"}


def test_corrupt_middle_line_refused(tmp_path):
    store = CellStore(tmp_path, KEY)
    store.write_cell_manifest(RESOLVED)
    store.append_row(_row("i0"))
    text = store.rows_path.read_text()
    store.rows_path.write_text("NOT JSON\n" + text)
    with pytest.raises(StoreError, match="file damage"):
        store.completed_item_ids()


# --------------------------------------------------------------------------
# consolidate → parquet
# --------------------------------------------------------------------------


def test_consolidate_schema_and_content(tmp_path):
    import pyarrow.parquet as pq

    s1 = CellStore(tmp_path, KEY)
    s1.write_cell_manifest(RESOLVED)
    s1.append_row(_row("i0"))
    s1.append_row(_row("i0", metric="ts_cell_end", value=1755000100.0,
                       domain="_meta"))
    s1.write_results({"n_rows": 2})
    key2 = "none__tier1__t0__a0__a1b2c3d4__0f1e2d3c"  # synthetic cell key — gitleaks:allow
    s2 = CellStore(tmp_path, key2)
    s2.write_cell_manifest(RESOLVED)
    s2.append_row(_row("i0", arm_name="none", kind="none", coeff=None,
                       trial_id=0, position_order=3))
    s2.write_results({"n_rows": 1})

    out = cs.consolidate(tmp_path)
    assert out == tmp_path / "rows.parquet"
    table = pq.read_table(out)
    assert tuple(table.column_names) == cs.ROW_COLUMNS
    assert table.num_rows == 3
    df = table.to_pandas()
    none_row = df[df["arm_name"] == "none"].iloc[0]
    assert none_row["coeff"] is None or (none_row["coeff"] != none_row["coeff"])
    assert none_row["position_order"] == 3


def test_consolidate_refusals(tmp_path):
    with pytest.raises(StoreError, match="no cells"):
        cs.consolidate(tmp_path)
    (tmp_path / "cells" / KEY).mkdir(parents=True)
    with pytest.raises(StoreError, match="no rows"):
        cs.consolidate(tmp_path, skip_incomplete=True)
    # schema drift: an extra key in a stored row is a refusal
    row = dataclasses.asdict(_row())
    row["stray"] = 1
    (tmp_path / "cells" / KEY / "rows.jsonl").write_text(
        json.dumps(row, sort_keys=True) + "\n")
    store = CellStore(tmp_path, KEY)
    store.write_results({"n_rows": 1})  # mark done so the drift check runs
    with pytest.raises(StoreError, match="schema drift"):
        cs.consolidate(tmp_path)


def test_consolidate_never_publishes_partial_cells(tmp_path):
    """results.json is the ONLY cell-done marker: a cell without one must
    never contribute rows to rows.parquet."""
    import pyarrow.parquet as pq

    done = CellStore(tmp_path, KEY)
    done.write_cell_manifest(RESOLVED)
    done.append_row(_row("i0"))
    done.write_results({"n_rows": 1})
    partial_key = "arm__tier1__t2__a0__a1b2c3d4__0f1e2d3c"  # synthetic cell key — gitleaks:allow
    partial = CellStore(tmp_path, partial_key)
    partial.write_cell_manifest(RESOLVED)
    partial.append_row(_row("i0", trial_id=2))
    assert not partial.is_done()
    # default: refusal naming the incomplete cell
    with pytest.raises(StoreError, match=partial_key):
        cs.consolidate(tmp_path)
    # explicit exclusion: only the done cell's rows are published
    out = cs.consolidate(tmp_path, skip_incomplete=True)
    assert pq.read_table(out).num_rows == 1
    # generation_complete.json is NOT a done marker either (F8)
    partial.write_generation_complete()
    with pytest.raises(StoreError, match="results.json"):
        cs.consolidate(tmp_path)


def test_append_row_refuses_non_finite_values(tmp_path):
    """A NaN/inf measurement refuses at the WRITE (allow_nan=False): a
    stored NaN flows into bootstrap quantiles and classifies as a silent
    SCREEN-PASS."""
    store = CellStore(tmp_path, KEY)
    store.write_cell_manifest(RESOLVED)
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(StoreError, match="non-finite"):
            store.append_row(_row("i0", value=bad))
    assert not store.rows_path.exists()  # nothing was written


def test_rows_of_task_returns_every_column_of_one_tasks_rows(tmp_path):
    """PW13 (E21 M2): a validator that must see MORE than the completion key
    of the existing rows — the EvalPlus converter checks their
    ``task_version`` / ``metric_def`` before skipping any — reads them
    through one read-only primitive, in file order, one task at a time."""
    store = CellStore(tmp_path, KEY)
    store.write_cell_manifest(RESOLVED)
    store.append_row(_row("0", task="tinyMMLU", domain="knowledge"))
    store.append_row(_row("1", task="tinyMMLU", domain="knowledge", value=0.0))
    store.append_row(_row("0", task="tinyArc", domain="knowledge"))
    got = store.rows_of_task("tinyMMLU")
    assert [(r["item_id"], r["value"]) for r in got] == [("0", 1.0), ("1", 0.0)]
    assert {r["task"] for r in got} == {"tinyMMLU"}
    assert set(got[0]) == {f.name for f in dataclasses.fields(cs.ResultRow)}
    assert store.rows_of_task("tinyHellaswag") == []
    assert CellStore(tmp_path, "other").rows_of_task("tinyMMLU") == []


def test_completed_keys_and_scoped_item_ids(tmp_path):
    """Completion identity is (task, item, metric): a bare item_id set
    lets tinyMMLU/'0' mask tinyArc/'0', and an nll_sum row without its
    token_count companion is NOT complete."""
    store = CellStore(tmp_path, KEY)
    store.write_cell_manifest(RESOLVED)
    store.append_row(_row("0", task="tinyMMLU", domain="knowledge"))
    store.append_row(_row("w0", task="tier0/nll_wikitext103",
                          domain="lm_fidelity_raw", metric="nll_sum",
                          value=200.0, tier=0))
    assert store.completed_keys() == {
        ("tinyMMLU", "0", "acc"),
        ("tier0/nll_wikitext103", "w0", "nll_sum")}
    # same bare item_id, different task: NOT complete for tinyArc
    assert store.completed_item_ids("tinyMMLU", ["acc"]) == {"0"}
    assert store.completed_item_ids("tinyArc", ["acc"]) == set()
    # companion metric missing -> incomplete until token_count lands
    nll_task = "tier0/nll_wikitext103"
    assert store.completed_item_ids(
        nll_task, ["nll_sum", "token_count"]) == set()
    store.append_row(_row("w0", task=nll_task, domain="lm_fidelity_raw",
                          metric="token_count", value=100.0, tier=0))
    assert store.completed_item_ids(
        nll_task, ["nll_sum", "token_count"]) == {"w0"}
    # the scoped form requires BOTH arguments
    with pytest.raises(StoreError, match="scoped"):
        store.completed_item_ids("tinyMMLU")
    with pytest.raises(StoreError, match="scoped"):
        store.completed_item_ids("tinyMMLU", [])


# --------------------------------------------------------------------------
# hashes
# --------------------------------------------------------------------------


def test_harness_config_hash():
    a = cs.harness_config_hash({"tasks": ["x"], "gen": {"t": 0.0}})
    b = cs.harness_config_hash({"gen": {"t": 0.0}, "tasks": ["x"]})
    assert a == b and len(a) == 16
    assert cs.harness_config_hash({"tasks": ["y"]}) != a
    with pytest.raises(StoreError):
        cs.harness_config_hash({})


def test_eval_config_digest():
    """The instrument-only identity: sensitive to task versions, template
    hash, and gen params; indifferent to key order; absent components are
    stated nulls, not omissions."""
    tv = {"tinyMMLU": "v1", "gsm8k_platinum_cap": "v2"}
    d = cs.eval_config_digest(tv, template_hash="ab" * 8,
                              gen_params={"greedy": True})
    assert len(d) == 16
    # key order never moves it
    tv2 = {"gsm8k_platinum_cap": "v2", "tinyMMLU": "v1"}
    assert cs.eval_config_digest(tv2, template_hash="ab" * 8,
                                 gen_params={"greedy": True}) == d
    # every component moves it
    assert cs.eval_config_digest({**tv, "tinyMMLU": "v9"},
                                 template_hash="ab" * 8,
                                 gen_params={"greedy": True}) != d
    assert cs.eval_config_digest(tv, template_hash="ff" * 8,
                                 gen_params={"greedy": True}) != d
    assert cs.eval_config_digest(tv, template_hash="ab" * 8,
                                 gen_params={"greedy": False}) != d
    # None components are stated inputs (distinct from any real value)
    d_null = cs.eval_config_digest(tv, template_hash=None, gen_params=None)
    assert len(d_null) == 16 and d_null != d
    with pytest.raises(StoreError, match="non-empty"):
        cs.eval_config_digest({}, template_hash=None, gen_params=None)
    with pytest.raises(StoreError, match="task_versions"):
        cs.eval_config_digest({"t": 1}, template_hash=None, gen_params=None)


def test_env_digest_fields_and_hash():
    fields = cs.env_digest_fields()
    for key in ("python", "torch", "transformers", "lm_eval", "vllm",
                "evalplus", "cuda_driver", "cuda_runtime", "gpu_name"):
        assert key in fields  # absent ones present as None, never skipped
    assert fields["torch"] is not None  # base dep
    d = cs.env_digest()
    assert len(d) == 16 and d == cs.env_digest()


def test_linked_and_dangling_entries_are_content_and_traversal_errors_surface(tmp_path):
    """Fifth review: `_content_artifacts` decided empty-versus-populated with a
    traversal that disagreed with the reader's — rglob does not descend a
    DIRECTORY symlink and `is_file()` discards it, while the generation reader
    follows it. A partial restore holding manifest.json plus `private/` linked
    to retained spools read as EMPTY here and as full there. Now a linked
    directory and a dangling link both count as content, and a directory the
    walk cannot enter is surfaced, never skipped."""
    import os
    store = CellStore(tmp_path, KEY)
    store.cell_dir.mkdir(parents=True)
    store.manifest_path.write_text("{}")
    assert store._content_artifacts() == [] and not store._has_content()
    retained = tmp_path / "retained_private"
    (retained / "spool").mkdir(parents=True)
    (retained / "spool" / "gens_xstest_v2.jsonl").write_text('{"item_id": "x", "text": "t"}\n')
    os.symlink(retained, store.cell_dir / "private", target_is_directory=True)
    assert store._content_artifacts() == ["private/"] and store._has_content()
    os.symlink(tmp_path / "nowhere.json", store.cell_dir / "evidence.json")
    assert store._content_artifacts() == ["evidence.json", "private/"]
    (store.cell_dir / "evidence.json").unlink()
    (store.cell_dir / "private").unlink()
    assert store._content_artifacts() == []
    if os.geteuid() != 0:
        locked = store.cell_dir / "locked"
        locked.mkdir(); (locked / "rows.jsonl").write_text("")
        locked.chmod(0)
        try:
            with pytest.raises(StoreError, match="cannot enumerate"):
                store._content_artifacts()
        finally:
            locked.chmod(0o755)
