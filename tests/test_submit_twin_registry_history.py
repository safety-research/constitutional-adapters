"""The funnel's registry proof reads `row_history` with its REAL schema (no run_id column; the key is
natural_key_json) through the REAL reducer's payload shape (Effect.history_payload: the record's columns + `_extra` for the attempt
metadata), in BOTH readers (the per-task `registry_runs_by_job` the decision acts on, and `registry_runs_for`), and attributes a job number
to a cluster: Slurm allocates job ids per cluster while the registry is shared, so an old run that once had job J on the OTHER cluster is not
this cluster's dead job J — excluded; an unknown or generic substrate is kept (absence is not evidence of another cluster)."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO, make_registry  # noqa: E402

sys.path.insert(0, str(REPO / "reproduce" / "lib")); sys.path.insert(0, str(REPO / "src"))
import submit_twin  # noqa: E402
from sisteer.registry import emit, records as R, synth, validate as V  # noqa: E402
from sisteer.registry import reducer as RD  # noqa: E402
from sisteer.registry.reducer import Effect  # noqa: E402

SCHEMA = (REPO / "src" / "sisteer" / "registry" / "schema.sql").read_text()


def real_row_history_ddl() -> str:
    i = SCHEMA.index("CREATE TABLE row_history ("); j = SCHEMA.index(");", i) + 2
    return SCHEMA[i:j]


def _history_row(conn, run_id: str, job: str, utc: str, record_type: str = "run", aid: str | None = None, cluster: str | None = None) -> None:
    """One row_history event as the reducer writes it: Effect.history_payload() — the record's columns (+ `_kind`), the attempt metadata under `_extra`."""
    import hashlib
    payload = {"run_id": run_id, "cluster_job_id": job, **({"cluster": cluster} if cluster else {})}
    eff = Effect(table="run", key={"run_id": run_id}, payload=payload, kind="full" if record_type == "run" else "patch",
                 extra=({"attempt_id": aid} if aid else {}))
    hp = eff.history_payload()
    sha = hashlib.sha256(f"{run_id}|{job}|{utc}|{record_type}".encode()).hexdigest()
    conn.execute("INSERT INTO row_history VALUES (?,?,?,?,?,?,?,?)",
                 ("run", eff.key_json, utc, sha, record_type, f"records/{run_id}.jsonl", json.dumps(hp), json.dumps(hp)))


def _co(root: Path):
    return SimpleNamespace(site=SimpleNamespace(registry_root=str(root)))


def test_both_registry_readers_find_a_run_whose_live_row_was_re_keyed_through_the_real_row_history(tmp_path):
    root = tmp_path / "registry"
    make_registry(root, runs=[{"run_id": "fit_trial__abc", "cluster_job_id": "999999", "lifecycle_state": "ingested"}])    # the live row: another writer's job
    conn = sqlite3.connect(root / "registry.sqlite"); conn.execute(real_row_history_ddl())
    _history_row(conn, "fit_trial__abc", "165623", "2026-10-01T06:00:00.000Z", cluster="cluster_b")                       # first registered under the dead job
    _history_row(conn, "fit_trial__abc", "999999", "2026-10-01T06:30:00.000Z", record_type="run_patch", cluster="cluster_b")
    _history_row(conn, "fit_trial__other", "777777", "2026-10-01T06:10:00.000Z", aid="attempt-7", cluster="cluster_b")
    conn.commit(); conn.close()
    co = _co(root)
    assert submit_twin.Coordinator.registry_runs_by_job(co, "165623", cluster="cluster_b") == ["fit_trial__abc"]      # history, not the live row
    assert submit_twin.Coordinator.registry_runs_by_job(co, "999999") == ["fit_trial__abc"]
    assert submit_twin.Coordinator.registry_runs_for(co, "165623", "attempt-x", cluster="cluster_b") == ["fit_trial__abc"]
    assert submit_twin.Coordinator.registry_runs_for(co, "000000", "attempt-7") == ["fit_trial__other"]                 # by the reducer's _extra.attempt_id
    assert submit_twin.Coordinator.registry_runs_by_job(co, "424242") == []


def test_a_job_number_reused_on_the_other_cluster_is_not_this_clusters_dead_job(tmp_path):
    """Two clusters, one job number J: the OTHER cluster's historical run under J must not refuse a re-mint of J here; a hit that names no cluster
    or a generic substrate is kept (fail closed); the attempt id matches regardless of the cluster spelling."""
    root = tmp_path / "registry"
    make_registry(root, runs=[{"run_id": "run_other_live", "cluster_job_id": "4242", "lifecycle_state": "ingested", "extra": {"cluster": "cluster_a"}},
                              {"run_id": "run_generic_live", "cluster_job_id": "4242", "lifecycle_state": "ingested", "extra": {"cluster": "slurm"}}])
    conn = sqlite3.connect(root / "registry.sqlite"); conn.execute(real_row_history_ddl())
    _history_row(conn, "run_other_hist", "4242", "2026-10-01T05:00:00.000Z", cluster="cluster_a")        # the other cluster's job 4242
    _history_row(conn, "run_mine_hist", "4242", "2026-10-01T05:01:00.000Z", cluster="cluster_b")         # this cluster's job 4242
    _history_row(conn, "run_nocluster_hist", "4242", "2026-10-01T05:02:00.000Z")                          # names no cluster: kept
    _history_row(conn, "run_attempt_hist", "1111", "2026-10-01T05:03:00.000Z", aid="att-b", cluster="cluster_a")   # the attempt id wins over the spelling
    conn.commit(); conn.close()
    co = _co(root)
    assert set(submit_twin.Coordinator.registry_runs_by_job(co, "4242", cluster="cluster_b")) == {"run_generic_live", "run_mine_hist", "run_nocluster_hist"}
    assert set(submit_twin.Coordinator.registry_runs_by_job(co, "4242", cluster="cluster_a")) == {"run_other_live", "run_generic_live", "run_other_hist", "run_nocluster_hist"}
    assert set(submit_twin.Coordinator.registry_runs_by_job(co, "4242")) == {"run_other_live", "run_generic_live", "run_other_hist", "run_mine_hist", "run_nocluster_hist"}   # no cluster known: everything (fail closed)
    assert submit_twin.Coordinator.registry_runs_for(co, "0", "att-b", cluster="cluster_b") == ["run_attempt_hist"]


def test_the_readers_still_work_on_a_registry_without_the_table(tmp_path):
    root = tmp_path / "registry"
    make_registry(root, runs=[{"run_id": "r1", "cluster_job_id": "1", "lifecycle_state": "ingested"}])
    co = _co(root)
    assert submit_twin.Coordinator.registry_runs_by_job(co, "1") == ["r1"] and submit_twin.Coordinator.registry_runs_for(co, "2", "a") == []


# a run payload the real validator accepts (vocabulary values; synthetic identities) — the generic `slurm` substrate, no label of its own
RUN_PAYLOAD = {"applied_positions_mode": "all", "campaign": "c", "cluster_job_id": "165623", "code_sha": synth.code_sha("a"), "decode_seed_rule": "per_item",
               "decode_source": "generation_config", "eval_id": "e", "harness": "bespoke_sisteer", "harness_version": "x", "instrument_id": "i",
               "instrument_match": "match", "item_set_id": "s", "lifecycle_state": "queued", "model_id": "m", "multi_turn_steering_scope": "every_turn_prefill_and_decode",
               "object_id": "o", "producer_object_key_legacy": "x", "producer_run_key_legacy": "x", "prompt_template_sha256": "b" * 64,
               "queued_utc": "2026-01-01T00:00:00.000Z", "scoring_rule": "exact_match", "serving_path": "lora_native", "substrate": "slurm",
               "substrate_source": "derived", "thinking_mode": "on", "token_cap_tokens": 16, "transfer_kind": "base_to_instruct"}


def _db_from_effects(root: Path, effs: dict, live_job: str) -> None:
    """The registry as the reducer leaves it: the live `run` row (extra_json = the effect's extra) and one row_history event per record."""
    make_registry(root, runs=[{"run_id": e.key["run_id"], "cluster_job_id": live_job, "extra": e.extra} for e in effs.values()])
    conn = sqlite3.connect(root / "registry.sqlite"); conn.execute(real_row_history_ddl())
    for i, (label, e) in enumerate(sorted(effs.items())):
        hp = e.history_payload()
        conn.execute("INSERT INTO row_history VALUES (?,?,?,?,?,?,?,?)",
                     ("run", e.key_json, f"2026-10-01T06:0{i}:00.000Z", hashlib.sha256(f"{label}|{live_job}".encode()).hexdigest(), "run", f"records/{label}.jsonl", json.dumps(hp), json.dumps(hp)))
    conn.commit(); conn.close()


def test_cluster_attribution_prefers_the_concrete_label_the_real_writer_stamps_on_a_generic_slurm_substrate(tmp_path, monkeypatch):
    """The real emitter REQUIRES the generic `slurm` substrate to carry the site's label as extra.cluster_label
    (emit.check_substrate, migration 0018), which the reducer persists under `_extra` in row_history and in the live row's extra_json — both
    readers read THAT before the generic substrate, else two generic Slurm clusters reusing job J are both "generic" and collide again. Driven
    through the actual RecordWriter, validator and reducer: two writers, substrate slurm, labels site_x / site_y, the same job number."""
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    monkeypatch.delenv(emit.RUN_OVERRIDES_ENV, raising=False)
    effs = {}
    for label in ("site_x", "site_y"):
        rid = f"fit_trial__{label}"
        w = emit.RecordWriter(str(tmp_path / "spool" / label), "c", rid, emit.producer_block("t", code_sha=synth.code_sha("a"), substrate="slurm"), payload_extra={"cluster_label": label})
        path = w.emit_run({**RUN_PAYLOAD, "run_id": rid})
        (_i, rec, sha), = list(R.read_record_file(path, compute_sha=True))
        assert rec["payload"]["extra"] == {"cluster_label": label} and rec["payload"]["substrate"] == "slurm", rec["payload"]
        v = V.validate_record(rec, path, sha); assert not isinstance(v, V.Malformed), str(v)
        (eff,) = RD.effects_of(v); assert eff.table == "run" and eff.extra == {"cluster_label": label}, eff
        effs[label] = eff
    root = tmp_path / "registry"; co = _co(root)
    _db_from_effects(root, effs, live_job="165623")                                              # the live rows carry the job
    assert submit_twin.Coordinator.registry_runs_by_job(co, "165623", cluster="site_x") == ["fit_trial__site_x"]
    assert submit_twin.Coordinator.registry_runs_by_job(co, "165623", cluster="site_y") == ["fit_trial__site_y"]
    assert sorted(submit_twin.Coordinator.registry_runs_by_job(co, "165623")) == ["fit_trial__site_x", "fit_trial__site_y"]   # no cluster asked: both
    assert submit_twin.Coordinator.registry_runs_for(co, "165623", "attempt-none", cluster="site_x") == ["fit_trial__site_x"]
    assert submit_twin.Coordinator.registry_runs_by_job(co, "165623", cluster="site_z") == []                                  # a third site: neither
    _db_from_effects(root, effs, live_job="999999")                                              # the live rows re-keyed: the history attributes by the label
    assert submit_twin.Coordinator.registry_runs_by_job(co, "165623", cluster="site_x") == ["fit_trial__site_x"]
    assert submit_twin.Coordinator.registry_runs_by_job(co, "165623", cluster="site_y") == ["fit_trial__site_y"]
    assert submit_twin.Coordinator.registry_runs_for(co, "165623", "attempt-none", cluster="site_y") == ["fit_trial__site_y"]
    # the readers, directly: the concrete label beats the generic substrate, in the history payload and in the live row's extra_json
    assert submit_twin.Coordinator._payload_cluster({"substrate": "slurm", "_extra": {"cluster_label": "site_x"}}) == "site_x"
    assert submit_twin.Coordinator._payload_cluster({"substrate": "slurm", "fields": {"extra": {"cluster_label": "site_y"}}}) == "site_y"
    assert submit_twin.Coordinator._payload_cluster({"substrate": "cluster_b"}) == "cluster_b"
    assert submit_twin.Coordinator._live_compatible(json.dumps({"cluster_label": "site_y"}), "site_x") is False
    assert submit_twin.Coordinator._live_compatible(json.dumps({"cluster_label": "site_x"}), "site_x") is True
    assert submit_twin.Coordinator._live_compatible(json.dumps({"cluster": "slurm"}), "site_x") is True                       # generic alone: kept (fail closed)
