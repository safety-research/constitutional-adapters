"""The closer's SUCCESSOR mode — a static run registered BEFORE static-reg-fix (its cap the instrument's read-at 8192,
no pass_label) is re-registered from its EXISTING store at the drawn rung through the registry's own planner (the key
`sdfbase_full22_<tag>_sampled_t<T>`, token_cap_tokens 4096 with its ladder index, pass_label base_cap, the trial, cluster_job_id / job_name = the
PRODUCING job's, extra.successor_of = the retired id), closed from that store (the static branch as it stands — nothing re-run), the retired row
filed `superseded` naming the successor (reason token_cap_identity_relabel), the pairs written; a FAILED attempt's store-less
registration is filed `cancelled` with the scheduler's terminal state as evidence and a note naming the producing attempt and the successor, while
a registration that filed a store is refused, never cancelled; a second run writes nothing. Through the REAL job-start registration over the real
seeds and a real-shaped static store (the closer test's fixtures)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_registry_close_stores import OBJECTS, TAG, _close, _generated, _records, _register, _runs, _seeded_registry, _static_store  # noqa: E402
from sisteer.registry import emit, hooks  # noqa: E402

KEY_T0 = f"sdfbase_full22_{TAG}_sampled_t0"
EVIDENCE = "sacct:805197 FAILED ExitCode 1:0 End 2026-09-30T10:43:43Z"


def _clean_env(monkeypatch):
    for k in (emit.DISABLED_ENV, emit.SPOOL_ENV, emit.PUSH_SCRIPT_ENV, hooks.EMIT_ENV, hooks.RUN_IDENTITY_ENV):
        monkeypatch.delenv(k, raising=False)


def _sampled_cfg(tmp_path: Path, job: str, trial: int) -> Path:
    cfg = tmp_path / f"checkout_{job}" / "configs" / f"sdfbase_full22_{TAG}_sampled_t{trial}.yaml"
    cfg.write_text(yaml.safe_dump({"explicit_cells": list(OBJECTS)})); return cfg


def _full_stores(wave: Path, runs: dict, trial: int = 0) -> None:
    for (ev, obj), r in runs.items():
        iset = r["plan"]["item_set_id"]; goals, variants = (int(x) for x in iset.rsplit("_", 1)[1].split("x"))
        _static_store(wave, trial, ev, obj, nshards=9, goals=goals, variants=variants)


def _states(spool: Path, rid: str, state: str) -> list:
    return [p for t, p, _pr in _records(spool, rid) if t == "run_state" and p.get("state") == state]


def _run_record(spool: Path, rid: str) -> dict:
    return [p for t, p, _pr in _records(spool, rid) if t == "run"][-1]


def test_the_successor_carries_the_drawn_rung_identity_the_producing_job_and_the_lineage_and_the_retired_row_is_superseded(tmp_path, monkeypatch):
    _clean_env(monkeypatch)
    root = tmp_path / "reg"; _seeded_registry(root); spool = root / "records"; wave = tmp_path / "wave"
    pending = _register(tmp_path, root, job="808182", job_name=f"f22s-{TAG}-static_base_t0", evals="pasttense,grok420")
    old = _runs(pending); assert len(old) == 4
    for r in old.values():                                                   # the pre-fix shape: the instrument's read-at cap, no pass label
        assert r["plan"]["token_cap_tokens"] == 8192 and not r["plan"].get("pass_label") and r["plan"]["cluster_job_id"] == "808182"
    _full_stores(wave, old)
    r = _close(pending, spool, wave, "--nshards", "9", job_name=f"f22s-{TAG}-static_base_t0")
    assert r.returncode == 0 and "4 filed" in r.stdout, r.stdout + r.stderr
    cfg = _sampled_cfg(tmp_path, "808182", 0); pairs = tmp_path / "pairs.tsv"
    # DRY first: the plan is described, nothing written, no successor pending file under pending/succ/
    before = {rid: len(_records(spool, rid)) for rid in (x["run_id"] for x in old.values())}
    r = _close(pending, spool, wave, "--nshards", "9", "--successor-token-cap", "4096", "--successor-pass-label", "base_cap", "--successor-config", str(cfg), "--objects", ",".join(OBJECTS),
               "--pairs-out", str(pairs), "--dry-run", job_name=f"f22s-{TAG}-static_base_t0")
    assert r.returncode == 0 and r.stdout.count("DRY-RUN successor ") == 4 and "would file run_state superseded" in r.stdout, r.stdout + r.stderr
    assert {rid: len(_records(spool, rid)) for rid in before} == before and not (pending.parent / "succ" / pending.name).exists() and not pairs.exists()
    # the relabel
    r = _close(pending, spool, wave, "--nshards", "9", "--successor-token-cap", "4096", "--successor-pass-label", "base_cap", "--successor-config", str(cfg), "--objects", ",".join(OBJECTS),
               "--pairs-out", str(pairs), job_name=f"f22s-{TAG}-static_base_t0")
    assert r.returncode == 0 and "4 pair(s), 4 superseded now, 4 successor(s) closed now" in r.stdout, r.stdout + r.stderr
    succ_pend = pending.parent / "succ" / pending.name
    new = _runs(succ_pend); assert set(new) == set(old)
    sdoc = json.loads(succ_pend.read_text()); assert sdoc["job_id"] == "808182" and sdoc["successor_decision"] == "D1900" and sdoc["successor_reason"] == "token_cap_identity_relabel"
    lines = [ln for ln in pairs.read_text().splitlines() if ln and not ln.startswith("#")]
    assert len(lines) == 4 and pairs.read_text().startswith("# retired_run_id\tsuccessor_run_id")
    for (ev, obj), o in old.items():
        n = new[(ev, obj)]; op, np_ = o["plan"], n["plan"]
        assert n["run_id"] != o["run_id"]
        # the drawn-rung identity, the same instrument, the planner key, the PRODUCING job, the lineage
        assert np_["token_cap_tokens"] == 4096 and np_["token_cap_rung_index"] == 0 and json.loads(np_["token_cap_ladder_json"]) == [4096, 8192, 16384] if isinstance(np_.get("token_cap_ladder_json"), str) else np_.get("token_cap_ladder_json") == [4096, 8192, 16384]
        assert np_["pass_label"] == "base_cap" and np_["trial_index"] == 0 and np_["instrument_id"] == op["instrument_id"] and np_["eval_id"] == ev
        assert np_["producer_run_key_legacy"] == KEY_T0 and np_["producer_object_key_legacy"] == op["producer_object_key_legacy"] and np_["view_legacy"] == op["view_legacy"]
        assert np_["cluster_job_id"] == "808182" and np_["job_name"] == f"f22s-{TAG}-static_base_t0"
        assert np_["extra"]["successor_of"] == o["run_id"] and np_["extra"]["successor_reason"] == "token_cap_identity_relabel" and np_["extra"]["successor_decision"] == "D1900" and np_["extra"]["producing_job_id"] == "808182"
        # the spooled run record says the same, lifecycle running
        rr = _run_record(spool, n["run_id"]); assert rr["token_cap_tokens"] == 4096 and rr["pass_label"] == "base_cap" and rr["cluster_job_id"] == "808182" and rr["lifecycle_state"] == "running"
        # closed from the EXISTING store: generated with the retired row's store_uri, completion sha and n_items
        st_o, pa_o = _generated(spool, o["run_id"]); st_n, pa_n = _generated(spool, n["run_id"])
        assert st_n and st_n["evidence_kind"] == "completion_artifact" and st_n["evidence_uri"] == st_o["evidence_uri"]
        assert {k: pa_n[k] for k in ("store_uri", "store_completion_artifact_uri", "store_completion_sha256", "n_items_generated")} == {k: pa_o[k] for k in ("store_uri", "store_completion_artifact_uri", "store_completion_sha256", "n_items_generated")}
        # the retired row: superseded, naming the successor, the reason, the decision, the producing job; evidence = the store's artifact
        (sup,) = _states(spool, o["run_id"], "superseded")
        assert sup["extra"] == {"superseded_by_run_id": n["run_id"], "reason": "token_cap_identity_relabel", "decision": "D1900", "producing_job_id": "808182"}
        assert sup["evidence_kind"] == "completion_artifact" and sup["evidence_uri"] == pa_n["store_completion_artifact_uri"] and n["run_id"] in sup["note"] and "158242e5d" in sup["note"]
        assert _states(spool, n["run_id"], "superseded") == []
        assert any(ln.split("\t")[:2] == [o["run_id"], n["run_id"]] for ln in lines)
    # idempotent: the same call again writes nothing new and reuses the successor pending file
    counts = {rid: len(_records(spool, rid)) for rid in [x["run_id"] for x in old.values()] + [x["run_id"] for x in new.values()]}
    text = pairs.read_text()
    r = _close(pending, spool, wave, "--nshards", "9", "--successor-token-cap", "4096", "--successor-pass-label", "base_cap", "--successor-config", str(cfg), "--objects", ",".join(OBJECTS),
               "--pairs-out", str(pairs), job_name=f"f22s-{TAG}-static_base_t0")
    assert r.returncode == 0 and "reused (idempotent)" in r.stdout and "4 pair(s), 0 superseded now, 0 successor(s) closed now" in r.stdout, r.stdout + r.stderr
    assert {rid: len(_records(spool, rid)) for rid in counts} == counts and pairs.read_text() == text


def test_a_failed_attempts_store_less_registration_is_cancelled_with_the_schedulers_evidence_and_a_registration_with_a_store_is_refused(tmp_path, monkeypatch):
    _clean_env(monkeypatch)
    root = tmp_path / "reg"; _seeded_registry(root); spool = root / "records"; wave = tmp_path / "wave"
    pending = _register(tmp_path, root, job="808182", job_name=f"f22s-{TAG}-static_base_t0", evals="pasttense")          # the producing attempt
    orphan = _register(tmp_path, root, job="805197", job_name=f"f22s-{TAG}-static_base_t0", evals="pasttense")           # the FAILED attempt: registered, never closed
    old = _runs(pending); orph = _runs(orphan); assert set(old) == set(orph) and len(old) == 2
    _full_stores(wave, old)
    assert _close(pending, spool, wave, "--nshards", "9", job_name=f"f22s-{TAG}-static_base_t0").returncode == 0
    cfg = _sampled_cfg(tmp_path, "808182", 0); pairs = tmp_path / "pairs.tsv"
    r = _close(pending, spool, wave, "--nshards", "9", "--successor-token-cap", "4096", "--successor-pass-label", "base_cap", "--successor-config", str(cfg), "--objects", ",".join(OBJECTS),
               "--pairs-out", str(pairs), "--cancel-orphan", f"{orphan}={EVIDENCE}", job_name=f"f22s-{TAG}-static_base_t0")
    assert r.returncode == 0 and "2 orphan(s) cancelled" in r.stdout, r.stdout + r.stderr
    new = _runs(pending.parent / "succ" / pending.name)
    for (ev, obj), o_r in orph.items():
        (c,) = _states(spool, o_r["run_id"], "cancelled")
        assert c["evidence_kind"] == "scheduler_state" and c["evidence_uri"] == EVIDENCE
        assert "attempt 805197" in c["note"] and "produced by attempt 808182" in c["note"] and new[(ev, obj)]["run_id"] in c["note"]
        assert c["extra"] == {"reason": "attempt_failed_before_store", "decision": "D1900", "produced_by_job_id": "808182", "successor_run_id": new[(ev, obj)]["run_id"]}
        assert _generated(spool, o_r["run_id"]) == (None, None)                                    # never closed, never generated
    text = pairs.read_text(); assert text.count("\ncancelled\t") == 2
    # a second run: the orphans are skipped (already cancelled), the pairs file unchanged
    r = _close(pending, spool, wave, "--nshards", "9", "--successor-token-cap", "4096", "--successor-pass-label", "base_cap", "--successor-config", str(cfg), "--objects", ",".join(OBJECTS),
               "--pairs-out", str(pairs), "--cancel-orphan", f"{orphan}={EVIDENCE}", job_name=f"f22s-{TAG}-static_base_t0")
    assert r.returncode == 0 and r.stdout.count("already cancelled") == 2 and pairs.read_text() == text, r.stdout
    # a registration that FILED A STORE is not an orphan: refused by name, nothing cancelled, exit 2
    r = _close(pending, spool, wave, "--nshards", "9", "--successor-token-cap", "4096", "--successor-pass-label", "base_cap", "--successor-config", str(cfg), "--objects", ",".join(OBJECTS),
               "--cancel-orphan", f"{pending}=sacct:808182 COMPLETED", job_name=f"f22s-{TAG}-static_base_t0")
    assert r.returncode == 2 and r.stdout.count("not an orphan; nothing cancelled") == 2, r.stdout + r.stderr
    for o in old.values():
        assert _states(spool, o["run_id"], "cancelled") == []
    # --objects narrows the relabel and the cancellation to the named cells
    r = _close(pending, spool, wave, "--nshards", "9", "--successor-token-cap", "4096", "--successor-pass-label", "base_cap", "--successor-config", str(cfg),
               "--objects", "none", "--dry-run", job_name=f"f22s-{TAG}-static_base_t0")
    assert r.returncode == 0 and "1 pair(s)" in r.stdout, r.stdout + r.stderr
