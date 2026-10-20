"""The DRIVER's two phases end to end, on the
pinned inspect-ai client (the ``inspect260`` mark — the driver imports under
``~/venvs/tinycap-laneB``; skipped where ``inspect_ai`` is absent).

``served_capability_run.py generate --verification-phase generate --row-engine-record <rec>`` on
the native world with the BigCodeBench-full control (the Inspect leg faked down to the REAL
``run_one_task`` over the synthetic fixture, a fake ``eval_async`` copying the unscored log):
the unscored ``.eval`` under the store, the generation sidecar + the pending marker in the row
dir, ``manifest.row_engine.verification_pass`` PENDING naming the task, ``gen_stats.json``'s block
GENERATED, no done marker, no controls summary, and ``RESULT`` reading ``result=GENERATED
status=GENERATED … phase=generate``. Then ``served_capability_run.py verify …`` on that store
(the coupled task the synthetic one over a LOCAL sandbox, the M2 vehicle SHUFFLED at
``max_connections=4``, the owned provider's transport install stubbed — Modal is not in this
venv): the task DONE with outcomes / summary / rows / ``gen_stats`` / ``controls_summary.json``,
``verification_pass`` the verified block (exactly ``VERIFICATION_PASS_VERIFIED_FIELDS``), the
marker gone, the original archived byte for byte, the merged log in place, the verifier's pointer
(``phase verify``, one cell) in the row dir, ``RESULT`` gaining ``result=COMPLETE status=OK …
verified_by=<fcid>``, the store's inode count UNCHANGED by the verify (every verification
artefact lives in the row dir). A second ``verify`` over the verified store exits 0 and moves
nothing (the door); every wrong argument refuses by name before the store moves; a
verifier killed at any step after the swap is finished by the next one, byte for byte; the
coupled path's window guard runs before the swap.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("inspect_ai", reason="tinycap client venv (hashed lock; the inspect260 mark)")

import inspect_ai  # noqa: E402

pytestmark = [
    pytest.mark.inspect260,
    pytest.mark.skipif(inspect_ai.__version__ != "0.3.260",
                       reason=f"the pinned client is inspect-ai 0.3.260; this venv has {inspect_ai.__version__}"),
]
os.environ.setdefault("INSPECT_DISPLAY", "none")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_inspect_served_verification_phase as T  # noqa: E402  (the fixture, the synthetic task, the constants)
import test_served_capability as LENS  # noqa: E402
import tinycap_store_shape as SS  # noqa: E402
from inspect_ai.log import read_eval_log  # noqa: E402
from test_served_capability_native import TT_CELL, native_world  # noqa: E402,F401  (`native_world` is a fixture)
from test_served_capability_row_engine import RECORD, _write_record  # noqa: E402

from sisteer.capeval.tinycap import completion  # noqa: E402
from sisteer.capeval.tinycap import contracts as tc  # noqa: E402
from sisteer.capeval.tinycap import cpu_verify as CV  # noqa: E402
from sisteer.capeval.tinycap import sandbox_scope as scope  # noqa: E402

srun = LENS.srun
ROW_TASK, N, WRONG = T.ROW_TASK, T.N, T.WRONG
STORE = "out"        # the native world's --out-root basename: the row dir's generation files are keyed by it
#: the cell driven here: the world's `none` route serves the fixture's virtual model (`petri__base`), so the
#: sidecar's route is the store's (the verifier holds the two equal); TT_CELL is the WRONG cell in the refusals
CELL = "none"
RUN_ID = "run-test-0001"
GEN_FCID, VERIFY_FCID = T.GEN_FCID, T.VERIFY_FCID
TREE_SHA = "ab" * 20


def _record(tmp_path: Path) -> dict:
    return {**RECORD, "row_dir": str(tmp_path / "row"), "engine_cells": [CELL, TT_CELL, "prompted__safety"]}


def _fake_inspect_leg(monkeypatch, calls, *, max_model_len=None, fixture=None):
    """``run_inspect_tasks`` faked down to the REAL ``run_one_task`` over the synthetic fixture: a
    generation-only plan under `generate` (the fake eval copies the unscored fixture), so the
    driver's own handling AROUND the leg is what runs for real. ``max_model_len`` is the served
    window the leg is handed (the sidecar records it; None = no window, nothing checked);
    ``fixture`` swaps the unscored log the fake eval copies."""
    from sisteer.capeval.tiers import inspect_served
    from sisteer.capeval.tiers import tier0

    async def fake(*, atoms, route, routes, system, store, template, thinking, base_url, limits=None,
                   verification_phase=tc.VERIFICATION_PHASE_COUPLED, generation=None, **kw):
        calls.append({"atoms": list(atoms), "phase": verification_phase, "generation": generation,
                      "limits": dict(limits or {})})
        if verification_phase != tc.VERIFICATION_PHASE_GENERATE:
            raise AssertionError("this world's leg is the generate phase's")
        if completion.is_done(store.cell_dir, ROW_TASK):
            # what run_inspect_tasks does before dispatching a plan: a DONE task is skipped with its summary
            calls[-1]["eval_calls"] = 0
            return {ROW_TASK: {"skipped": "done marker present", **(completion.read_task_summary(store.cell_dir, ROW_TASK) or {})}}
        original = T.fixture_log()
        seen: list = []
        plan = T._plan(original, generation_only=True)
        em = tier0._Emitter(store, dict(template), dict(tc.TINYCAP_DOMAIN_OF), tc.INSPECT_TASK_VERSION)
        mp = pytest.MonkeyPatch()                       # the fake eval lives for the leg alone
        mp.setattr("inspect_ai.eval_async", T._fake_eval_copying(fixture or T.FIXTURE, seen))
        try:
            out = await inspect_served.run_one_task(plan, model=None, store=store, template=template,
                                                    thinking=tc.THINKING_OFF, concurrency=8, emitter=em,
                                                    route_virtual_model=T.VMODEL, limits=limits, generation=generation,
                                                    max_model_len=max_model_len)
        finally:
            mp.undo()
        calls[-1]["eval_calls"] = len(seen)
        return {ROW_TASK: out}

    monkeypatch.setattr(inspect_served, "run_inspect_tasks", fake)


def _owner_env(monkeypatch, fcid: str, record: str = ""):
    """The harness's environment for a driver process: the owner tuple (and, for the verifier, the
    provider record path ``<row dir>/watch/sandboxes_<fcid>__<cell>.jsonl`` it exports)."""
    monkeypatch.setenv(scope.OWNER_ENV_RUN_ID, RUN_ID)
    monkeypatch.setenv(scope.OWNER_ENV_CELL, CELL)
    monkeypatch.setenv(scope.OWNER_ENV_FUNCTION_CALL_ID, fcid)
    monkeypatch.setenv(scope.SANDBOX_RECORD_ENV, record)


def _verifier_env(monkeypatch, row_dir: Path, fcid: str = VERIFY_FCID) -> str:
    record = str(row_dir / "watch" / f"sandboxes_{fcid}__{CELL}.jsonl")
    _owner_env(monkeypatch, fcid, record)
    return record


def _generate(world, monkeypatch, *, extra=(), max_model_len=None, fixture=None, out_root=None, fcid=GEN_FCID):
    LENS.stub_framework_versions(monkeypatch)
    calls: list = []
    _fake_inspect_leg(monkeypatch, calls, max_model_len=max_model_len, fixture=fixture)
    rec_path = _write_record(world.tmp_path, _record(world.tmp_path))
    _owner_env(monkeypatch, fcid)
    argv = [a if a != TT_CELL else CELL
            for a in LENS._percell_argv(world, tc.CONTROL_BIGCODEBENCH_FULL, out_root=out_root)]
    srun.main(argv + [tc.FLAG_THINKING, tc.THINKING_OFF, tc.FLAG_ROW_ENGINE_RECORD, str(rec_path),
                      tc.FLAG_VERIFICATION_PHASE, tc.VERIFICATION_PHASE_GENERATE, *extra])
    cell_dir = LENS._cell_dir(out_root if out_root is not None else world.out_root)
    return cell_dir, json.loads((cell_dir / "manifest.json").read_text()), rec_path, calls


def _gpu_pointer(row_dir: Path) -> Path:
    """The GPU phase's row pointer (the harness writes it; ROW_POINTER_FIELDS' shape) — what the
    verifier reads its run id, name, model, mode and tree sha off."""
    doc = {"run_id": RUN_ID, "cell": None, "function_call_id": GEN_FCID, "name": "llama8b-f22s1__row",
           "model": "llama8b-f22s1", "mode": "thinking_off", "out_dir": str(row_dir),
           "controls": [tc.CONTROL_BIGCODEBENCH_FULL, "bfcl_full"], "tree_sha": TREE_SHA, "sandbox_record": None,
           "sandbox_app_name": tc.BCB_SANDBOX_APP_NAME, "started_at": 1.0, "phase": tc.VERIFICATION_PHASE_GENERATE,
           "cells": [CELL, TT_CELL], "row": "llama8b-f22s1", "row_attempt": {"standup_id": RECORD["standup_id"]},
           "records": {}}
    p = CV.generation_pointer_path(row_dir, GEN_FCID)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=1, sort_keys=True))
    return p


def _verify_argv(world, rec_path: Path, *, cell=CELL, controls=tc.CONTROL_BIGCODEBENCH_FULL, fcid=VERIFY_FCID,
                 row_dir=None, extra=(), out_root=None):
    row_dir = row_dir if row_dir is not None else world.tmp_path / "row"
    # the harness's argv (worker B): no --run-id (the owner env carries it), --max-sandboxes min(S // K, 8)
    argv = ["verify", "--out-root", str(out_root if out_root is not None else world.out_root), "--cell", cell,
            "--controls", controls,
            "--config", str(world.cfg_path), "--tree", str(world.tmp_path), "--row-dir", str(row_dir),
            tc.FLAG_ROW_ENGINE_RECORD, str(rec_path), "--max-sandboxes", "4"]
    if fcid is not None:
        argv += ["--function-call-id", fcid]
    return argv + list(extra)


def _local_mechanism(monkeypatch):
    """The coupled task = the synthetic one over a LOCAL sandbox; the M2 vehicle SHUFFLED at
    max_connections 4; the owned provider (Modal) stubbed out of the transport install."""
    original = T.fixture_log()
    monkeypatch.setattr(CV, "build_coupled_bcb_task", lambda name, *, log_ids, repo_root: T.synthetic_task(original))
    real = CV.run_mechanism

    async def shuffled(mechanism, **kw):
        return await real(mechanism, max_connections=4, sample_shuffle=11, **kw)

    monkeypatch.setattr(CV, "run_mechanism", shuffled)
    T._no_modal_transport(monkeypatch)


# --------------------------------------------------------------------------- #
# generate: GENERATED, nothing filed                                             #
# --------------------------------------------------------------------------- #


def test_generate_leaves_the_task_generated_and_files_nothing_for_it(native_world, monkeypatch):
    w = native_world
    cell_dir, manifest, rec_path, calls = _generate(w, monkeypatch)
    row_dir = w.tmp_path / "row"
    # the leg was dispatched under the generate phase with the row's generation facts
    assert [c["phase"] for c in calls] == [tc.VERIFICATION_PHASE_GENERATE]
    gen = calls[0]["generation"]
    assert isinstance(gen, CV.GenerationPhase) and gen.row_dir == row_dir and gen.cell == CELL and gen.store == STORE
    assert gen.generation_fcid == GEN_FCID and gen.concurrency == 4          # the world's --concurrency
    # the manifest: the row-engine block is the record's with verification_pass PENDING naming the task
    block = manifest[tc.MANIFEST_KEY_ROW_ENGINE]
    assert block["verification_pass"] == {"state": tc.VERIFICATION_PASS_PENDING, "tasks": [ROW_TASK]}
    assert {k: v for k, v in block.items() if k != "verification_pass"} == \
           {k: v for k, v in _record(w.tmp_path).items() if k != "verification_pass"}
    assert manifest["cell_id"] == CELL and manifest["virtual_model"] == T.VMODEL
    # the unscored log under the store; the sidecar + marker in the row dir
    logs = sorted(tc.inspect_log_dir(cell_dir, ROW_TASK, 0).glob("*.eval"))
    assert len(logs) == 1 and all(not s.scores for s in read_eval_log(str(logs[0])).samples)
    marker = CV.read_pending_marker(row_dir, CELL, ROW_TASK, STORE)
    sidecar = CV.read_generation_sidecar(row_dir, CELL, ROW_TASK, STORE)
    assert Path(marker["log"]) == logs[0].resolve() and marker["generation_fcid"] == GEN_FCID
    assert marker["cell"] == CELL and tuple(sorted(marker)) == tuple(sorted(tc.PENDING_MARKER_FIELDS))
    assert sidecar["store"] == STORE == w.out_root.name
    # the marker's file name is keyed by the store, so the in-run repeat (same --cell, another store) never collides
    assert sorted(p.name for p in (row_dir / CV.GENERATION_DIR).glob("pending_*.json")) == [
        f"pending_{CELL}__inspect__bigcodebench_full__{STORE}.json"]
    assert sidecar["route_virtual_model"] == T.VMODEL == manifest["virtual_model"]
    assert sidecar["concurrency"] == 4
    # gen_stats: the task's block is GENERATED (never an OK status); nothing filed for it
    gen_stats = json.loads((cell_dir / "gen_stats.json").read_text())
    assert gen_stats[ROW_TASK]["status"] == tc.REC_STATUS_GENERATED
    assert not completion.is_done(cell_dir, ROW_TASK)
    assert not (cell_dir / tc.DONE_DIR).exists() and not (cell_dir / tc.SUMMARY_DIR).exists()
    assert not (w.out_root / CELL / tc.CONTROLS_SUMMARY_FILENAME).exists()
    # RESULT: GENERATED, not COMPLETE
    lines = (w.out_root / tc.RESULT_FILENAME).read_text().splitlines()
    assert len(lines) == 1
    assert f" result={tc.REC_STATUS_GENERATED} status={tc.REC_STATUS_GENERATED} " in lines[0]
    assert lines[0].startswith(f"{CV.RESULT_TAG} name={CELL} rc=0 ")
    assert lines[0].endswith(f" phase={tc.VERIFICATION_PHASE_GENERATE} tasks={ROW_TASK}")
    assert f"sha={manifest[tc.MANIFEST_KEY_PRODUCER]['tree_sha']}" in lines[0]


def test_generate_refuses_without_the_record_and_without_a_bigcodebench_task(native_world, monkeypatch):
    w = native_world
    LENS.stub_framework_versions(monkeypatch)
    _fake_inspect_leg(monkeypatch, [])
    base = [a if a != TT_CELL else CELL for a in LENS._percell_argv(w, tc.CONTROL_BIGCODEBENCH_FULL)] + [
        tc.FLAG_THINKING, tc.THINKING_OFF]
    with pytest.raises(SystemExit) as ei:
        srun.main(base + [tc.FLAG_VERIFICATION_PHASE, tc.VERIFICATION_PHASE_GENERATE])
    assert f"{tc.FLAG_VERIFICATION_PHASE} {tc.VERIFICATION_PHASE_GENERATE} needs {tc.FLAG_ROW_ENGINE_RECORD}" in str(ei.value)
    rec_path = _write_record(w.tmp_path, _record(w.tmp_path))
    with pytest.raises(SystemExit) as ei:
        srun.main([a if a != TT_CELL else CELL for a in LENS._percell_argv(w, tc.CONTROL_GSM8K)] + [
            tc.FLAG_THINKING, tc.THINKING_OFF, tc.FLAG_ROW_ENGINE_RECORD, str(rec_path),
            tc.FLAG_VERIFICATION_PHASE, tc.VERIFICATION_PHASE_GENERATE])
    assert "files no BigCodeBench task" in str(ei.value)
    assert not w.out_root.exists()                                    # refused before any store was touched
    # `verify` is not a value of the generate flag (argparse refuses it: exit 2)
    with pytest.raises(SystemExit) as ei:
        srun.build_parser().parse_args(base + [tc.FLAG_VERIFICATION_PHASE, tc.VERIFICATION_PHASE_VERIFY])
    assert ei.value.code == 2
    ns = srun.build_parser().parse_args(base)
    assert ns.verification_phase == tc.VERIFICATION_PHASE_COUPLED


# --------------------------------------------------------------------------- #
# verify: the task filed through the same tail; the store's inodes unchanged     #
# --------------------------------------------------------------------------- #


def test_verify_files_the_generated_task_and_rewrites_verification_pass(native_world, monkeypatch, capsys):
    w = native_world
    cell_dir, before, rec_path, _calls = _generate(w, monkeypatch)
    row_dir = w.tmp_path / "row"
    _gpu_pointer(row_dir)
    layout_before = set(SS.normalised_layout(w.out_root))
    _local_mechanism(monkeypatch)
    record = _verifier_env(monkeypatch, row_dir)
    srun.main(_verify_argv(w, rec_path))
    manifest = json.loads((cell_dir / "manifest.json").read_text())
    # the task is DONE, filed through the coupled tail: outcomes, summary, done, rows, gen_stats, controls summary
    assert completion.is_done(cell_dir, ROW_TASK)
    fn = tc.marker_filename(ROW_TASK)
    outcomes = json.loads((cell_dir / tc.TASK_OUTCOMES_DIR / fn).read_text())
    assert len(outcomes["items"]) == N and {i["item_id"] for i in outcomes["items"] if i["value"] == 0.0} == WRONG
    summary = json.loads((cell_dir / tc.SUMMARY_DIR / fn).read_text())
    assert summary[tc.METRIC_ACC] == (N - len(WRONG)) / N and summary["n_sandbox_ok"] == N and summary["logs_reused"] == 0
    rows = [json.loads(line) for line in (cell_dir / "rows.jsonl").read_text().splitlines() if line.strip()]
    bcb_rows = [r for r in rows if r["task"] == ROW_TASK]
    assert len(bcb_rows) == N + 2 and sum(1 for r in bcb_rows if r["metric"] == tc.METRIC_ACC and r["item_id"] != tc.AGG_ITEM_ID) == N
    gen_stats = json.loads((cell_dir / "gen_stats.json").read_text())
    assert gen_stats[ROW_TASK] == summary
    cs_path = w.out_root / CELL / tc.CONTROLS_SUMMARY_FILENAME
    assert cs_path.is_file() and ROW_TASK in json.dumps(json.loads(cs_path.read_text()))
    # verification_pass -> the verified block, EXACTLY the contract's fields; nothing else in the block moved
    block = manifest[tc.MANIFEST_KEY_ROW_ENGINE]
    vp = block["verification_pass"]
    assert tuple(sorted(vp)) == tuple(sorted(tc.VERIFICATION_PASS_VERIFIED_FIELDS))
    assert vp["state"] == tc.VERIFICATION_PASS_VERIFIED and vp["mechanism"] == CV.DEFAULT_MECHANISM == "M2_replay_vehicle"
    assert vp["function_call_id"] == VERIFY_FCID and vp["n_items"] == N and vp["max_sandboxes"] == 4
    assert vp["sandbox_facts"] == tc.bcb_sandbox_facts() and re.fullmatch(r"[0-9a-f]{64}", vp["vehicle_log_sha256"])
    assert vp["started_at"] <= vp["finished_at"]
    assert {k: v for k, v in block.items() if k != "verification_pass"} == \
           {k: v for k, v in before[tc.MANIFEST_KEY_ROW_ENGINE].items() if k != "verification_pass"}
    # every other manifest leaf is the GPU phase's, except the re-pooled quality block
    assert {k: v for k, v in manifest.items() if k not in (tc.MANIFEST_KEY_ROW_ENGINE, tc.MANIFEST_KEY_QUALITY)} == \
           {k: v for k, v in before.items() if k not in (tc.MANIFEST_KEY_ROW_ENGINE, tc.MANIFEST_KEY_QUALITY)}
    assert manifest[tc.MANIFEST_KEY_QUALITY]["by_task"][ROW_TASK]["at_cap_rate"] == 1 / N     # id 7 at the cap
    # the marker gone, the sidecar kept, the original archived byte for byte, the merged log in place
    assert not CV.pending_marker_exists(row_dir, CELL, ROW_TASK, STORE)
    assert CV.generation_sidecar_path(row_dir, CELL, ROW_TASK, STORE).is_file()
    archive = CV.archived_log_path(row_dir, CELL, ROW_TASK, STORE)
    assert archive.is_file() and CV.sha256_file(archive) == CV.sha256_file(T.FIXTURE)
    logs = sorted(tc.inspect_log_dir(cell_dir, ROW_TASK, 0).glob("*.eval"))
    merged = read_eval_log(str(logs[0]))
    assert len(logs) == 1 and all(set(s.scores) == {"verify"} for s in merged.samples)
    assert merged.eval.model == T.TARGET_KEY and set(merged.samples[0].model_usage) == {T.TARGET_KEY}
    assert not list(tc.inspect_log_dir(cell_dir, ROW_TASK, 0).glob("*.part"))
    # the verifier's pointer: phase verify, ONE cell, the GPU pointer's name / model / mode / tree sha
    pointer = json.loads(CV.verify_pointer_path(row_dir, VERIFY_FCID).read_text())
    assert set(pointer) == set(CV.VERIFY_POINTER_FIELDS) and pointer["phase"] == tc.VERIFICATION_PHASE_VERIFY
    assert (pointer["run_id"], pointer["cell"], pointer["function_call_id"]) == (RUN_ID, CELL, VERIFY_FCID)
    assert pointer["name"] == "llama8b-f22s1__row" and pointer["model"] == "llama8b-f22s1" and pointer["tree_sha"] == TREE_SHA
    assert pointer["sandbox_record"] == record == str(CV.verify_sandbox_record_path(row_dir, VERIFY_FCID, CELL))
    assert "cells" not in pointer
    # RESULT: the GENERATED line, then COMPLETE with verified_by
    lines = (w.out_root / tc.RESULT_FILENAME).read_text().splitlines()
    assert len(lines) == 2 and f" result={tc.REC_STATUS_GENERATED} " in lines[0]
    assert f" result={tc.RESULT_COMPLETE} status=OK sha={TREE_SHA} " in lines[1]
    assert lines[1].endswith(f" verified_by={VERIFY_FCID}") and lines[1].startswith(f"{CV.RESULT_TAG} name=llama8b-f22s1__row ")
    # the registry's landing-2 rule (2026-09-30): the LAST line's `standup=` token is the manifest's row_engine.standup_id —
    # the re-stamp used to carry none and every landing-2 store was refused at ingestion; `verified_by=` stays the last token
    tokens = dict(tok.split("=", 1) for tok in lines[1].split()[1:])
    assert tokens["standup"] == RECORD["standup_id"] == block["standup_id"] and lines[1].endswith(f" standup={RECORD['standup_id']} verified_by={VERIFY_FCID}")
    assert lines[0].split()[-1] == f"standup={RECORD['standup_id']}" or f" standup={RECORD['standup_id']}" in lines[0] or "standup=" not in lines[0]
    # the verify adds to the store EXACTLY the coupled tail's files (the done / summary / outcomes records, the
    # rows, the controls summary) and NOTHING of its own: the archive, the pointer and the sandbox record are
    # row-dir files, so a verified store has the coupled store's inode count (40 on a full-shape store, rule 2)
    added = set(SS.normalised_layout(w.out_root)) - layout_before
    fn = tc.marker_filename(ROW_TASK)
    assert added == {f"cells/<KEY>/{tc.DONE_DIR}/", f"cells/<KEY>/{tc.DONE_DIR}/{fn}",
                     f"cells/<KEY>/{tc.SUMMARY_DIR}/", f"cells/<KEY>/{tc.SUMMARY_DIR}/{fn}",
                     f"cells/<KEY>/{tc.TASK_OUTCOMES_DIR}/", f"cells/<KEY>/{tc.TASK_OUTCOMES_DIR}/{fn}",
                     "cells/<KEY>/rows.jsonl", f"{CELL}/", f"{CELL}/{tc.CONTROLS_SUMMARY_FILENAME}"}, added
    assert not any("watch/" in x or x.endswith(".part") or "/generation/" in x for x in SS.relative_layout(w.out_root))
    # the row dir: the archive, the sidecar and the scoring receipt (this attempt's scoring identity, written before
    # the swap — a restart files THIS scoring from the merged log, never another)
    assert sorted(p.name for p in (row_dir / CV.GENERATION_DIR).iterdir()) == [
        f"{CELL}__inspect__bigcodebench_full__{STORE}.eval", f"{CELL}__inspect__bigcodebench_full__{STORE}.json",
        f"{CELL}__inspect__bigcodebench_full__{STORE}.scoring.json"]
    receipt = CV.read_scoring_receipt(row_dir, CELL, ROW_TASK, STORE)
    assert tuple(sorted(receipt)) == tuple(sorted(CV.SCORING_RECEIPT_FIELDS))
    assert receipt["function_call_id"] == vp["function_call_id"] == VERIFY_FCID and receipt["mechanism"] == vp["mechanism"]
    assert receipt["vehicle_log_sha256"] == vp["vehicle_log_sha256"] and receipt["started_at"] == vp["started_at"]
    assert receipt["max_sandboxes"] == 4 and receipt["n_items"] == N and receipt["primary_scorer"] == "verify"
    assert receipt["log_sha256"] == CV.sha256_file(T.FIXTURE)
    assert receipt["verdict_digest"] == CV.verdict_digest(merged, primary_scorer="verify")
    # the door: a second verify over the VERIFIED store exits 0, says so, and moves nothing — not a
    # refusal (a resumed harness may re-spawn a verifier for a cell whose first one finished)
    store_before = SS.relative_layout(w.out_root)
    row_before = sorted(str(q.relative_to(row_dir)) for q in row_dir.rglob("*"))
    manifest_before = (cell_dir / "manifest.json").read_bytes()
    capsys.readouterr()
    srun.main(_verify_argv(w, rec_path, fcid="fc-01TESTVERIFY0000000000002"))
    out = capsys.readouterr().out
    assert "already verified" in out and VERIFY_FCID in out and "nothing to do" in out
    assert SS.relative_layout(w.out_root) == store_before
    assert sorted(str(q.relative_to(row_dir)) for q in row_dir.rglob("*")) == row_before   # no second pointer either
    assert (cell_dir / "manifest.json").read_bytes() == manifest_before
    assert (w.out_root / tc.RESULT_FILENAME).read_text().splitlines() == lines


def test_verify_refuses_a_wrong_cell_control_record_or_row_dir_before_the_store_moves(native_world, monkeypatch):
    w = native_world
    cell_dir, _m, rec_path, _c = _generate(w, monkeypatch)
    row_dir = w.tmp_path / "row"
    _local_mechanism(monkeypatch)
    _verifier_env(monkeypatch, row_dir)
    snapshot = SS.relative_layout(w.out_root)
    for argv, needle in [
        (_verify_argv(w, rec_path, cell=TT_CELL), "is cell"),
        (_verify_argv(w, rec_path, controls="bfcl_full"), "exactly ONE sandboxed BigCodeBench task"),
        (_verify_argv(w, rec_path, controls=f"{tc.CONTROL_BIGCODEBENCH_FULL},bfcl_full"), "exactly ONE sandboxed"),
        (_verify_argv(w, rec_path, row_dir=w.tmp_path / "elsewhere"), "is not --row-dir"),
        (_verify_argv(w, _write_record(w.tmp_path, {**_record(w.tmp_path), "max_loras": 7}, name="other.json")),
         "is not this store's"),
        (_verify_argv(w, rec_path, extra=["--mechanism", "M1_score_async"]), "mechanism M1_score_async failed"),
    ]:
        with pytest.raises(SystemExit) as ei:
            srun.main(argv)
        assert needle in str(ei.value), (needle, str(ei.value))
        assert str(ei.value).startswith("[served-cap] verify:")
    # the store is untouched by every refusal; the marker stands
    assert SS.relative_layout(w.out_root) == snapshot
    assert CV.pending_marker_exists(row_dir, CELL, ROW_TASK, STORE)
    # no FunctionCall id anywhere -> refused by name
    monkeypatch.delenv(scope.OWNER_ENV_RUN_ID)
    monkeypatch.delenv(scope.OWNER_ENV_CELL)
    monkeypatch.delenv(scope.OWNER_ENV_FUNCTION_CALL_ID)
    with pytest.raises(SystemExit, match="--function-call-id"):
        srun.main(_verify_argv(w, rec_path, fcid=None))
    # an environment declaring ANOTHER owner is a wiring error
    _owner_env(monkeypatch, "fc-someone-else")
    with pytest.raises(SystemExit, match="wiring error"):
        srun.main(_verify_argv(w, rec_path))
    # the parser: the verify subcommand's dests (no --run-id on the harness's argv; --mechanism defaults in main)
    ns = srun.build_parser().parse_args(_verify_argv(w, rec_path))
    assert ns.cmd == "verify" and ns.row_dir == str(row_dir) and ns.mechanism is None and ns.max_sandboxes == 4
    assert ns.function_call_id == VERIFY_FCID and ns.run_id is None and ns.row_engine_record == str(rec_path)
    assert ns.tree == str(w.tmp_path) and ns.controls == tc.CONTROL_BIGCODEBENCH_FULL


# --------------------------------------------------------------------------- #
# the split path runs the coupled path's window guard                         #
# --------------------------------------------------------------------------- #

#: the fixture's one cap-bound item (id 7: stop_reason max_tokens, prompt 207 tokens) at the plan's 4,096 cap needs a
#: 4,303-token window; the world's standup serves 4,096 (LENS.STANDUP_WINDOW), the BCB OFF pool is 4,608
AT_CAP_ITEM, AT_CAP_PROMPT, CAP = "7", 207, 4096


def _unmeasured_fixture(tmp_path: Path) -> Path:
    """The fixture with the cap-bound sample's usage dropped: a cap-bound item whose prompt length was never
    measured (no target usage) — indistinguishable from one the window clamped (E50 #5)."""
    from inspect_ai.log import write_eval_log
    log = T.fixture_log()
    s7 = next(s for s in log.samples if str(s.id) == AT_CAP_ITEM)
    s7.model_usage = {}
    s7.output.usage = None
    p = tmp_path / "unscored_12_unmeasured.eval"
    write_eval_log(log, str(p), format="eval")
    return p


@pytest.mark.parametrize("window, fixture_fn, needle", [
    (LENS.STANDUP_WINDOW, None, f"the pass cap {CAP} does not fit the SERVED window max_model_len={LENS.STANDUP_WINDOW}"),
    (4608, _unmeasured_fixture, "PROMPT LENGTH was not measured"),
], ids=["over_the_window", "unmeasured_prompt"])
def test_verify_refuses_a_cap_bound_item_the_served_window_could_not_hold_before_the_store_moves(
        native_world, monkeypatch, window, fixture_fn, needle):
    """The coupled path calls ``assert_pass_fits_window`` after ``results_from_log`` (a cap-bound
    sample the served window clamped BELOW the cap is a false at-cap reading; one whose prompt was
    not measured cannot be told from it) — the verifier runs the SAME guard over the merged log,
    with the sidecar's ``max_model_len``, BEFORE the swap and before anything is filed: refused by
    name, the store's log still the unscored original, the marker standing."""
    w = native_world
    fixture = fixture_fn(w.tmp_path) if fixture_fn else None
    cell_dir, _m, rec_path, _c = _generate(w, monkeypatch, max_model_len=window, fixture=fixture)
    row_dir = w.tmp_path / "row"
    sidecar = CV.read_generation_sidecar(row_dir, CELL, ROW_TASK, STORE)
    assert sidecar["max_model_len"] == window
    log_path = Path(sidecar["log"])
    _gpu_pointer(row_dir)
    _local_mechanism(monkeypatch)
    _verifier_env(monkeypatch, row_dir)
    snapshot = SS.relative_layout(w.out_root)
    log_sha = CV.sha256_file(log_path)
    with pytest.raises(SystemExit) as ei:
        srun.main(_verify_argv(w, rec_path))
    msg = str(ei.value)
    assert msg.startswith("[served-cap] verify:") and needle in msg and f"{ROW_TASK}:" in msg, msg
    if fixture_fn is None:
        assert f"worst '{AT_CAP_ITEM}' needs {AT_CAP_PROMPT} prompt + {CAP} = {AT_CAP_PROMPT + CAP} tokens" in msg
    # nothing moved: the store's layout, the unscored log's bytes, the marker; no archive was written
    assert SS.relative_layout(w.out_root) == snapshot
    assert CV.sha256_file(log_path) == log_sha
    assert all(not s.scores for s in read_eval_log(str(log_path)).samples)
    assert CV.pending_marker_exists(row_dir, CELL, ROW_TASK, STORE)
    assert not CV.archived_log_path(row_dir, CELL, ROW_TASK, STORE).exists()
    assert not completion.is_done(cell_dir, ROW_TASK) and not (cell_dir / tc.SUMMARY_DIR).exists()
    assert len((w.out_root / tc.RESULT_FILENAME).read_text().splitlines()) == 1


def test_verify_files_a_cap_bound_item_the_served_window_holds(native_world, monkeypatch):
    """The guard's other side: at the BCB OFF pool's 4,608 window the cap-bound item fits
    (207 + 4,096 <= 4,608) and the task files as before, its at-cap draw counted."""
    w = native_world
    cell_dir, _m, rec_path, _c = _generate(w, monkeypatch, max_model_len=4608)
    row_dir = w.tmp_path / "row"
    _gpu_pointer(row_dir)
    _local_mechanism(monkeypatch)
    _verifier_env(monkeypatch, row_dir)
    srun.main(_verify_argv(w, rec_path))
    assert completion.is_done(cell_dir, ROW_TASK)
    summary = json.loads((cell_dir / tc.SUMMARY_DIR / tc.marker_filename(ROW_TASK)).read_text())
    assert summary["n_at_cap"] == 1 and summary[tc.METRIC_ACC] == (N - len(WRONG)) / N
    manifest = json.loads((cell_dir / "manifest.json").read_text())
    assert manifest[tc.MANIFEST_KEY_ROW_ENGINE]["verification_pass"]["state"] == tc.VERIFICATION_PASS_VERIFIED
    assert not CV.pending_marker_exists(row_dir, CELL, ROW_TASK, STORE)


# --------------------------------------------------------------------------- #
# a verifier killed at ANY step after the swap is restartable                 #
# --------------------------------------------------------------------------- #


class _Killed(Exception):
    """The verifier's death, injected at one step: never one of the exceptions ``cmd_verify`` turns into a
    named refusal, so it propagates out of ``srun.main`` exactly like a container timeout would."""


#: the keys whose values differ between two runs of the same filing (clocks, a call's own sha) — masked
_VOLATILE = {k: f"<{k.upper()}>" for k in (
    "written_at", "finished_at", "started_at", "created_at", "ts", "generated_at", "wall_clock_s", "vehicle_log_sha256")}
_AT = re.compile(r" at=\S+")


def _store_view(out_root: Path, row_dir: Path, *, tmp_path: Path) -> dict:
    """Everything a verified store and its row-dir files SAY, with the volatile leaves masked: every JSON /
    JSONL file under the cell dir parsed (paths under tmp_path -> <TMP>, the store key -> <KEY>), the merged
    log's per-sample verdicts and its non-score content, the RESULT lines without their ``at=``, and the row
    dir's generation files for this store."""
    import tinycap_golden as golden
    cell_dir = LENS._cell_dir(out_root)
    key = cell_dir.name
    view: dict = {"layout": SS.normalised_layout(out_root)}
    for path in sorted(q for q in cell_dir.rglob("*") if q.is_file()):
        rel = path.relative_to(cell_dir).as_posix().replace(key, "<KEY>")
        if path.suffix == ".json":
            doc = json.loads(path.read_text())
        elif path.suffix == ".jsonl":
            doc = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        elif path.suffix == ".eval":
            log = read_eval_log(str(path))
            doc = {"verdicts": [(str(x.id), x.scores["verify"].value if x.scores else None) for x in log.samples],
                   "content": [json.loads(x.model_dump_json(exclude={"scores"})) for x in log.samples],
                   "metrics": [(sc.name, {m: v.value for m, v in sc.metrics.items()})
                               for sc in (log.results.scores if log.results is not None else [])]}
            rel = re.sub(r"[^/]+\.eval$", "<LOG>.eval", rel)
        else:
            doc = path.read_text()
        view[rel] = T._stable(golden.normalise(json.loads(json.dumps(doc, default=str).replace(key, "<KEY>")),
                                               tmp_path=tmp_path, volatile=_VOLATILE))
    view["RESULT"] = [_AT.sub(" at=<AT>", ln) for ln in (out_root / tc.RESULT_FILENAME).read_text().splitlines()]
    store = out_root.name
    view["row_generation"] = sorted(q.name.replace(store, "<STORE>") for q in (row_dir / CV.GENERATION_DIR).iterdir()
                                    if q.name.endswith(f"__{store}.json") or q.name.endswith(f"__{store}.eval"))
    return view


def _after(monkeypatch, module, name):
    """``module.name`` runs for real, then the verifier dies."""
    real = getattr(module, name)

    def wrapped(*a, **k):
        out = real(*a, **k)
        raise _Killed(name)

    monkeypatch.setattr(module, name, wrapped)
    return lambda: monkeypatch.setattr(module, name, real)


def _instead(monkeypatch, module, name):
    """``module.name`` is never reached: the verifier dies at the call."""
    real = getattr(module, name)

    def wrapped(*a, **k):
        raise _Killed(name)

    monkeypatch.setattr(module, name, wrapped)
    return lambda: monkeypatch.setattr(module, name, real)


def _kill_after_rows(monkeypatch, n: int):
    """The filing tail dies after ``n`` rows reached rows.jsonl (a partial filing)."""
    from sisteer.capeval.tiers import tier0
    real = tier0._Emitter.emit
    count = {"n": 0}

    def emit(self, **kw):
        count["n"] += 1
        if count["n"] > n:
            raise _Killed(f"after {n} rows")
        return real(self, **kw)

    monkeypatch.setattr(tier0._Emitter, "emit", emit)
    return lambda: monkeypatch.setattr(tier0._Emitter, "emit", real)


def _arm_kill(monkeypatch, kill):
    """One kill spec -> the undo. ``("hook", <step>)`` dies the moment ``cpu_verify.verify_store`` reports that
    step done; the rest die inside the libraries the tail calls."""
    kind, where = kill
    if kind == "hook":
        def hook(step: str) -> None:
            if step == where:
                raise _Killed(step)
        monkeypatch.setattr(CV, "VERIFY_STEP_HOOK", hook)
        return lambda: monkeypatch.setattr(CV, "VERIFY_STEP_HOOK", None)
    from sisteer.capeval import store as cs
    if kind == "swap_done":                    # the pre-fix world's failure: the swap complete, nothing filed
        return _after(monkeypatch, CV, "write_merged_log")
    if kind == "rows_partial":
        return _kill_after_rows(monkeypatch, where)
    if kind == "summary_only":                 # the summary written, no outcome record, no done marker
        return _instead(monkeypatch, completion, "write_task_outcomes")
    if kind == "outcomes_no_done":             # summary + outcomes written, no done marker
        return _instead(monkeypatch, completion, "write_done_marker")
    if kind == "manifest_written":
        return _after(monkeypatch, cs.CellStore, "write_cell_manifest")
    raise AssertionError(kill)


KILLS = [
    ("swap_done", None), ("rows_partial", 5), ("summary_only", None), ("outcomes_no_done", None),
    ("manifest_written", None),
] + [("hook", step) for step in getattr(CV, "VERIFY_STEPS", ())]


@pytest.mark.parametrize("kill", KILLS, ids=[f"{k}:{w}" if w is not None else k for k, w in KILLS])
def test_a_verifier_killed_at_any_step_after_the_swap_is_finished_by_the_next_verify_byte_for_byte(
        native_world, monkeypatch, capsys, kill):
    """Before the fix `write_merged_log` replaced the only unscored log BEFORE anything was filed, so a
    death anywhere after it left a SCORED log under a pending marker and the next `verify` refused at
    `unscored_problems` — the sha-checked archive was never used. Now: the archive is the unscored source
    (the sidecar records the generated log's sha; a store whose log is not that log but whose archive is
    restores it and re-scores), the filing is idempotent, and a verify over a store already verified
    completes the tail (or reads it done). The invariant: kill the verifier at EVERY step from the archive
    to the marker removal, run `verify` again — the store equals an uninterrupted run's, byte for byte
    outside the clocks."""
    w = native_world
    row_dir = w.tmp_path / "row"
    # the reference: generate + verify, uninterrupted, into `out`
    _cell, _m, rec_path, _c = _generate(w, monkeypatch)
    _gpu_pointer(row_dir)
    _local_mechanism(monkeypatch)
    _verifier_env(monkeypatch, row_dir)
    srun.main(_verify_argv(w, rec_path))
    reference = _store_view(w.out_root, row_dir, tmp_path=w.tmp_path)
    # the subject: the same cell into `out2` (the row dir's files keyed by the store), killed at `kill`
    out2 = w.tmp_path / "out2"
    _generate(w, monkeypatch, out_root=out2)
    _verifier_env(monkeypatch, row_dir)                  # `_generate` exported the GPU phase's owner; back to the verifier's
    before = _store_view(out2, row_dir, tmp_path=w.tmp_path)
    assert before != reference
    undo = _arm_kill(monkeypatch, kill)
    with pytest.raises(_Killed):
        srun.main(_verify_argv(w, rec_path, out_root=out2))
    undo()
    capsys.readouterr()
    torn = _store_view(out2, row_dir, tmp_path=w.tmp_path)
    if kill in (("hook", "marker_removed"), ("hook", "claim_released")):
        assert torn == reference                 # the death came after the last store mutation: nothing is torn
    else:
        assert torn != reference                 # the death left a torn store
    # the second verify converges — exit 0, no refusal
    srun.main(_verify_argv(w, rec_path, out_root=out2))
    out = capsys.readouterr().out
    after = _store_view(out2, row_dir, tmp_path=w.tmp_path)
    for k in sorted(set(after) | set(reference)):
        assert after.get(k) == reference.get(k), (kill, k)
    assert after == reference
    # the log of record on disk is scored, the archive is the fixture, the marker is gone, RESULT ends COMPLETE
    assert CV.sha256_file(CV.archived_log_path(row_dir, CELL, ROW_TASK, out2.name)) == CV.sha256_file(T.FIXTURE)
    assert not CV.pending_marker_exists(row_dir, CELL, ROW_TASK, out2.name)
    lines = (out2 / tc.RESULT_FILENAME).read_text().splitlines()
    assert len(lines) == 2 and f" result={tc.RESULT_COMPLETE} status=OK " in lines[1] and lines[1].endswith(f" verified_by={VERIFY_FCID}")
    kind, where = kill
    if kind == "hook" and where in ("result_appended", "marker_removed", "claim_released"):
        assert "already verified" in out or "completing" in out
    # a third verify is a no-op: already verified, nothing moves
    snapshot = SS.relative_layout(out2)
    srun.main(_verify_argv(w, rec_path, out_root=out2))
    assert "already verified" in capsys.readouterr().out
    assert SS.relative_layout(out2) == snapshot and after == _store_view(out2, row_dir, tmp_path=w.tmp_path)


# --------------------------------------------------------------------------- #
# the door, the driver's side: generate over a pending / torn / verified store #
# --------------------------------------------------------------------------- #


def test_generate_over_a_pending_store_reuses_the_generation_and_over_a_torn_one_refuses(native_world, monkeypatch, capsys):
    """What the harness's CPU-only resume may assume of the driver: a second `generate --verification-phase
    generate` over a PENDING store (unscored log + store-keyed marker + sidecar, verification_pass pending)
    issues no request, writes no new marker (the standing one is byte-identical, the sidecar is rewritten with
    logs_reused 1) and re-emits the GENERATED RESULT line; with the marker gone but the unscored log retained
    it refuses by name as torn (no regeneration, no marker re-created); over a VERIFIED store it skips the
    done task and emits no GENERATED line."""
    w = native_world
    row_dir = w.tmp_path / "row"
    cell_dir, m1, rec_path, calls1 = _generate(w, monkeypatch)
    marker_path = CV.pending_marker_path(row_dir, CELL, ROW_TASK, STORE)
    marker_bytes, stat1 = marker_path.read_bytes(), marker_path.stat()
    log_path = Path(CV.read_generation_sidecar(row_dir, CELL, ROW_TASK, STORE)["log"])
    assert calls1[0]["eval_calls"] == 1
    # (1) pending -> reused, from another GPU container (a new fcid)
    _cell2, m2, _rec, calls2 = _generate(w, monkeypatch, fcid="fc-01TESTGENERATE000000000002")
    assert calls2[0]["eval_calls"] == 0                                      # no request left this run
    assert marker_path.read_bytes() == marker_bytes
    assert (marker_path.stat().st_mtime_ns, marker_path.stat().st_ino) == (stat1.st_mtime_ns, stat1.st_ino)
    sidecar = CV.read_generation_sidecar(row_dir, CELL, ROW_TASK, STORE)
    assert sidecar["logs_reused"] == 1 and sidecar["generated_calls"] == 0 and sidecar["generation_fcid"] == GEN_FCID
    assert m2[tc.MANIFEST_KEY_ROW_ENGINE]["verification_pass"] == {"state": tc.VERIFICATION_PASS_PENDING, "tasks": [ROW_TASK]}
    lines = (w.out_root / tc.RESULT_FILENAME).read_text().splitlines()
    assert len(lines) == 2 and all(f" result={tc.REC_STATUS_GENERATED} status={tc.REC_STATUS_GENERATED} " in ln for ln in lines)
    assert sorted(p.name for p in (row_dir / CV.GENERATION_DIR).iterdir()) == [
        f"{CELL}__inspect__bigcodebench_full__{STORE}.json", f"pending_{CELL}__inspect__bigcodebench_full__{STORE}.json"]
    assert all(not s.scores for s in read_eval_log(str(log_path)).samples)
    # (2) torn: the marker gone, the unscored log retained -> refused by name, nothing regenerated or re-marked
    marker_path.unlink()
    snapshot = SS.relative_layout(w.out_root)
    with pytest.raises(SystemExit) as ei:
        _generate(w, monkeypatch)
    assert "torn" in str(ei.value) and "pending marker" in str(ei.value) and str(ei.value).startswith("[served-cap] ")
    assert not marker_path.exists()
    assert SS.relative_layout(w.out_root) == snapshot
    assert len((w.out_root / tc.RESULT_FILENAME).read_text().splitlines()) == 2
    # (3) verified: the marker put back, the store verified, a further generate skips the done task and stamps nothing
    marker_path.write_bytes(marker_bytes)
    _gpu_pointer(row_dir)
    _local_mechanism(monkeypatch)
    _verifier_env(monkeypatch, row_dir)
    srun.main(_verify_argv(w, rec_path))
    result_after_verify = (w.out_root / tc.RESULT_FILENAME).read_text()
    manifest_after_verify = json.loads((cell_dir / "manifest.json").read_text())
    _cell3, m3, _rec3, calls3 = _generate(w, monkeypatch)
    assert calls3[0]["eval_calls"] == 0
    assert (w.out_root / tc.RESULT_FILENAME).read_text() == result_after_verify        # no GENERATED line re-emitted
    assert m3[tc.MANIFEST_KEY_ROW_ENGINE] == manifest_after_verify[tc.MANIFEST_KEY_ROW_ENGINE]   # verified block kept
    assert not marker_path.exists()


# --------------------------------------------------------------------------- #
# simplify FYI: the sandbox-owner environment is read under `generate` alone      #
# --------------------------------------------------------------------------- #


def test_a_malformed_owner_environment_refuses_a_generate_phase_row_but_never_a_coupled_one(native_world, monkeypatch):
    """`cmd_generate` read `owner_from_env()` for EVERY row-engine run — a coupled row (which files everything
    itself and writes no marker) refused on a partial TINYCAP_OWNER_* tuple it never needed. The owner is read
    under the generate phase alone, where the marker's `generation_fcid` comes from it."""
    from test_tinycap_identity_version import _fake_gsm8k_leg
    w = native_world
    LENS.stub_framework_versions(monkeypatch)
    rec_path = _write_record(w.tmp_path, _record(w.tmp_path))
    # a PARTIAL owner tuple in the environment (one of three names): a wiring error for anyone who READS it
    monkeypatch.setenv(scope.OWNER_ENV_RUN_ID, RUN_ID)
    monkeypatch.delenv(scope.OWNER_ENV_CELL, raising=False)
    monkeypatch.delenv(scope.OWNER_ENV_FUNCTION_CALL_ID, raising=False)
    with pytest.raises(ValueError, match="partial sandbox owner"):
        scope.owner_from_env()
    # the COUPLED row-engine run files its (faked) GSM8K leg and stamps the record — the environment is not read
    calls: list = []
    _fake_gsm8k_leg(monkeypatch, calls)
    argv = LENS._percell_argv(w, tc.CONTROL_GSM8K)
    srun.main(argv + [tc.FLAG_THINKING, tc.THINKING_OFF, tc.FLAG_ROW_ENGINE_RECORD, str(rec_path)])
    assert len(calls) == 1
    manifest = json.loads((LENS._cell_dir(w.out_root) / "manifest.json").read_text())
    assert manifest[tc.MANIFEST_KEY_ROW_ENGINE]["verification_pass"] is None
    # the GENERATE phase reads it (the marker's generation_fcid) and refuses the partial tuple by name
    out2 = w.tmp_path / "out2"
    _fake_inspect_leg(monkeypatch, [])
    argv = [a if a != TT_CELL else CELL for a in LENS._percell_argv(w, tc.CONTROL_BIGCODEBENCH_FULL, out_root=out2)]
    with pytest.raises(SystemExit, match="partial sandbox owner"):
        srun.main(argv + [tc.FLAG_THINKING, tc.THINKING_OFF, tc.FLAG_ROW_ENGINE_RECORD, str(rec_path),
                          tc.FLAG_VERIFICATION_PHASE, tc.VERIFICATION_PHASE_GENERATE])
    assert not (w.tmp_path / "row" / CV.GENERATION_DIR).exists()


# --------------------------------------------------------------------------- #
# simplify FYI: the request accounting is the generation side's, carried through the sidecar #
# --------------------------------------------------------------------------- #


def _edit_sidecar(row_dir: Path, **fields) -> dict:
    p = CV.generation_sidecar_path(row_dir, CELL, ROW_TASK, STORE)
    doc = json.loads(p.read_text())
    doc.update(fields)
    p.write_text(json.dumps(doc, indent=1, sort_keys=True))
    return doc


def test_verify_files_the_generation_sides_request_count_and_refuses_an_inconsistent_one(native_world, monkeypatch):
    """The sidecar's `n_requests` was written and never read, and the verifier filed `n_retried_requests=0`, so
    a verified store's summary could differ from a coupled store's after retries. The verifier carries the
    sidecar's retried count into the SAME tail (`n_requests` = items + retried, the coupled formula) and holds
    the sidecar's `n_requests` to it — an inconsistent pair refuses by name before the store moves."""
    w = native_world
    row_dir = w.tmp_path / "row"
    cell_dir, _m, rec_path, _c = _generate(w, monkeypatch)
    sidecar = CV.read_generation_sidecar(row_dir, CELL, ROW_TASK, STORE)
    assert sidecar["n_requests"] == N and sidecar["n_retried_requests"] == 0
    _gpu_pointer(row_dir)
    _local_mechanism(monkeypatch)
    _verifier_env(monkeypatch, row_dir)
    # (1) an inconsistent pair refuses before anything moves
    _edit_sidecar(row_dir, n_retried_requests=2, n_requests=N + 5)
    log_path = Path(sidecar["log"])
    snapshot, log_sha = SS.relative_layout(w.out_root), CV.sha256_file(log_path)
    with pytest.raises(SystemExit) as ei:
        srun.main(_verify_argv(w, rec_path))
    assert "n_requests" in str(ei.value) and f"{N + 5}" in str(ei.value) and str(ei.value).startswith("[served-cap] verify:")
    assert SS.relative_layout(w.out_root) == snapshot and CV.sha256_file(log_path) == log_sha
    assert CV.pending_marker_exists(row_dir, CELL, ROW_TASK, STORE)
    # (2) a generation that retried two requests (the coupled tail's accounting) files items + retried
    _edit_sidecar(row_dir, n_retried_requests=2, n_requests=N + 2)
    srun.main(_verify_argv(w, rec_path))
    fn = tc.marker_filename(ROW_TASK)
    summary = json.loads((cell_dir / tc.SUMMARY_DIR / fn).read_text())
    done = json.loads((cell_dir / tc.DONE_DIR / fn).read_text())
    assert summary["n_requests"] == done["n_requests"] == N + 2 and summary[tc.METRIC_N] == N
    gen_stats = json.loads((cell_dir / "gen_stats.json").read_text())
    assert gen_stats[ROW_TASK]["n_requests"] == N + 2


# --------------------------------------------------------------------------- #
# a restart never mixes two attempts' verdicts                                #
# --------------------------------------------------------------------------- #

FLIP_ITEM = "0"          # correct in the fixture (value 1.0); the first attempt's sandbox "fails" it


def _flipping_mechanism(monkeypatch, flip: dict):
    """On top of ``_local_mechanism``: the vehicle's verdict for ``FLIP_ITEM`` is turned to ``I`` while
    ``flip["on"]`` — a sandbox that timed out or errored on the first attempt and succeeds on the retry, the
    non-determinism the synthetic scorer otherwise lacks. Counts the calls."""
    real = CV.run_mechanism

    async def flipping(mechanism, **kw):
        flip["calls"] += 1
        vehicle, path = await real(mechanism, **kw)
        if flip["on"]:
            for s in vehicle.samples:
                if str(s.id) == FLIP_ITEM:
                    s.scores["verify"].value = "I"
        return vehicle, path

    monkeypatch.setattr(CV, "run_mechanism", flipping)


def _acc_rows(cell_dir: Path) -> dict:
    rows = [json.loads(ln) for ln in (cell_dir / "rows.jsonl").read_text().splitlines() if ln.strip()]
    return {r["item_id"]: r["value"] for r in rows if r["task"] == ROW_TASK and r["metric"] == tc.METRIC_ACC
            and r["item_id"] != tc.AGG_ITEM_ID}


def test_a_verifier_killed_during_row_emission_resumes_its_own_scoring_and_never_files_a_mixed_store(
        native_world, monkeypatch, capsys):
    """The first attempt's sandbox fails item k (verdict A), the verifier dies after k's row
    reached rows.jsonl; a second attempt's sandbox would pass it (B). Re-scoring would leave A in the rows
    (the emitter skips an existing key) and B in the log / summary / outcomes — a mixed store nothing flags.
    Now: with the first attempt's merged log on disk the second attempt RESUMES FILING that scoring (no
    re-score; A everywhere: rows = log = summary = outcomes; the verified block and verified_by name the
    verifier that scored); with the merged log gone (the original back in place) the re-score is compared
    against every pre-existing row and REFUSES naming item k — never a mixed store."""
    w = native_world
    row_dir = w.tmp_path / "row"
    flip = {"on": True, "calls": 0}
    A_WRONG = WRONG | {FLIP_ITEM}

    def first_attempt(out_root):
        cell_dir, _m, rec_path, _c = _generate(w, monkeypatch, out_root=out_root)
        _gpu_pointer(row_dir)
        _verifier_env(monkeypatch, row_dir)
        flip["on"] = True
        undo = _kill_after_rows(monkeypatch, 3)              # items 0, 1, 2 filed; item 0 carries A (0.0)
        with pytest.raises(_Killed):
            srun.main(_verify_argv(w, rec_path, out_root=out_root))
        undo()
        assert _acc_rows(cell_dir) == {"0": 0.0, "1": 0.0, "2": 1.0}
        log_path = Path(CV.read_generation_sidecar(row_dir, CELL, ROW_TASK, out_root.name)["log"])
        assert {str(s.id): s.scores["verify"].value for s in read_eval_log(str(log_path)).samples}[FLIP_ITEM] == "I"
        flip["on"] = False                                    # the retry's sandbox would pass item 0 (B)
        return cell_dir, rec_path, log_path

    _local_mechanism(monkeypatch)
    _flipping_mechanism(monkeypatch, flip)
    # (1) the merged log survives: the second attempt (another container) resumes the first's scoring
    cell_dir, rec_path, log_path = first_attempt(w.out_root)
    calls_before = flip["calls"]
    second = "fc-01TESTVERIFY0000000000002"
    _verifier_env(monkeypatch, row_dir, fcid=second)
    capsys.readouterr()
    # the dead verifier's claim still stands: a resume under a new fcid is refused at entry until the harness,
    # having found the first call terminal, names it in the supersede token
    with pytest.raises(SystemExit, match="holds the claim"):
        srun.main(_verify_argv(w, rec_path, fcid=second))
    assert flip["calls"] == calls_before
    monkeypatch.setenv(CV.VERIFY_SUPERSEDES_ENV, VERIFY_FCID)
    srun.main(_verify_argv(w, rec_path, fcid=second))
    monkeypatch.delenv(CV.VERIFY_SUPERSEDES_ENV)
    out = capsys.readouterr().out
    assert flip["calls"] == calls_before and "resum" in out            # nothing re-scored
    assert CV.read_verify_claim(row_dir, CELL, ROW_TASK, STORE) is None  # released last: a completed store has no claim
    assert _acc_rows(cell_dir) == {i: (0.0 if i in A_WRONG else 1.0) for i in map(str, range(N))}
    fn = tc.marker_filename(ROW_TASK)
    summary = json.loads((cell_dir / tc.SUMMARY_DIR / fn).read_text())
    outcomes = json.loads((cell_dir / tc.TASK_OUTCOMES_DIR / fn).read_text())
    assert summary[tc.METRIC_ACC] == (N - len(A_WRONG)) / N
    assert {i["item_id"] for i in outcomes["items"] if i["value"] == 0.0} == A_WRONG
    assert {str(s.id): s.scores["verify"].value for s in read_eval_log(str(log_path)).samples}[FLIP_ITEM] == "I"
    rows = [json.loads(ln) for ln in (cell_dir / "rows.jsonl").read_text().splitlines() if ln.strip()]
    agg = next(r for r in rows if r["metric"] == tc.METRIC_ACC_AGGREGATE)
    assert agg["value"] == summary[tc.METRIC_ACC]
    vp = json.loads((cell_dir / "manifest.json").read_text())[tc.MANIFEST_KEY_ROW_ENGINE]["verification_pass"]
    assert vp["state"] == tc.VERIFICATION_PASS_VERIFIED and vp["function_call_id"] == VERIFY_FCID     # the verifier that SCORED
    assert vp["mechanism"] == CV.DEFAULT_MECHANISM and vp["max_sandboxes"] == 4 and vp["n_items"] == N
    lines = (w.out_root / tc.RESULT_FILENAME).read_text().splitlines()
    assert len(lines) == 2 and lines[1].endswith(f" verified_by={VERIFY_FCID}")
    assert CV.verify_pointer_path(row_dir, second).is_file()                   # the completing container's pointer
    assert not CV.pending_marker_exists(row_dir, CELL, ROW_TASK, STORE)
    # (2) the merged log gone (the original back in place): the re-score disagrees with item 0's row -> refused by name
    out2 = w.tmp_path / "out2"
    cell_dir2, rec_path, log_path2 = first_attempt(out2)
    calls_before = flip["calls"]
    archive = CV.archived_log_path(row_dir, CELL, ROW_TASK, out2.name)
    shutil.copyfile(archive, log_path2)                                        # "the merged log forced absent"
    rows_before = (cell_dir2 / "rows.jsonl").read_bytes()
    snapshot = SS.relative_layout(out2)
    with pytest.raises(SystemExit) as ei:
        srun.main(_verify_argv(w, rec_path, out_root=out2))
    msg = str(ei.value)
    assert msg.startswith("[served-cap] verify:") and f"'{FLIP_ITEM}'" in msg and "0.0" in msg and "1.0" in msg, msg
    assert "mixed" in msg or "disagree" in msg
    assert flip["calls"] == calls_before + 1                                   # it re-scored, then refused
    assert (cell_dir2 / "rows.jsonl").read_bytes() == rows_before               # nothing filed, nothing swapped
    assert SS.relative_layout(out2) == snapshot
    assert CV.sha256_file(log_path2) == CV.sha256_file(archive)
    assert CV.pending_marker_exists(row_dir, CELL, ROW_TASK, out2.name)
    assert not completion.is_done(cell_dir2, ROW_TASK)


# --------------------------------------------------------------------------- #
# one verifier at a time on a store — the claim                               #
# --------------------------------------------------------------------------- #

SECOND_FCID = "fc-01TESTVERIFY0000000000002"
B_ROWS = {i: (0.0 if i in WRONG else 1.0) for i in map(str, range(N))}          # the unflipped scorer's verdicts


def _claim_of(row_dir: Path, store: str = STORE):
    return CV.read_verify_claim(row_dir, CELL, ROW_TASK, store)


def test_two_verifiers_on_one_pending_store_the_second_refuses_at_entry_and_supersedes_only_through_the_token(
        native_world, monkeypatch, capsys):
    """A `--verify-pending` resume can spawn a second verifier while the first's FunctionCall is
    alive; the receipt and the row comparison make SEQUENTIAL recovery safe but two live scorers would interleave
    their writes. The belt: at entry (before any read of the log) a verifier claims the store
    (`<row dir>/generation/<cell>__<task>__<store>.claim.json`, an atomic create); a second verifier refuses by
    name while another's claim stands, UNLESS the harness set `TINYCAP_VERIFY_SUPERSEDES=<that fcid>` (only for a
    call it found terminal or lost) — then it takes the claim, and the FIRST refuses at its pre-swap re-check
    with nothing of its own written. Here the second arrives while the first has scored (divergent scorers: the
    first fails item 0, the second passes it); the store ends as the second's alone."""
    w = native_world
    row_dir = w.tmp_path / "row"
    cell_dir, _m, rec_path, _c = _generate(w, monkeypatch)
    _gpu_pointer(row_dir)
    _local_mechanism(monkeypatch)
    flip = {"on": True, "calls": 0}
    _flipping_mechanism(monkeypatch, flip)
    first = VERIFY_FCID
    nested: dict = {}

    def hook(step: str) -> None:
        if step != "scored" or nested:
            return
        nested["ran"] = True
        # the second verifier arrives while the first is alive (it has scored, not yet swapped)
        assert _claim_of(row_dir)["function_call_id"] == first
        _verifier_env(monkeypatch, row_dir, fcid=SECOND_FCID)
        flip["on"] = False
        with pytest.raises(SystemExit) as ei:
            srun.main(_verify_argv(w, rec_path, fcid=SECOND_FCID))
        nested["refusal"] = str(ei.value)
        assert _claim_of(row_dir)["function_call_id"] == first                  # the refusal took nothing
        assert not CV.scoring_receipt_path(row_dir, CELL, ROW_TASK, STORE).exists()
        assert not CV.verify_pointer_path(row_dir, SECOND_FCID).exists()           # refused before it declared itself
        # the harness decided the first call is lost: the token names it, the second takes the claim and finishes
        monkeypatch.setenv(CV.VERIFY_SUPERSEDES_ENV, first)
        srun.main(_verify_argv(w, rec_path, fcid=SECOND_FCID))
        monkeypatch.delenv(CV.VERIFY_SUPERSEDES_ENV)
        nested["second_out"] = capsys.readouterr().out
        assert _claim_of(row_dir) is None                                          # a completed store has no claim

    monkeypatch.setattr(CV, "VERIFY_STEP_HOOK", hook)
    _verifier_env(monkeypatch, row_dir, fcid=first)
    with pytest.raises(SystemExit) as ei:
        srun.main(_verify_argv(w, rec_path, fcid=first))
    monkeypatch.setattr(CV, "VERIFY_STEP_HOOK", None)
    msg = str(ei.value)
    assert nested.get("ran"), "the hook never saw the first verifier's scored step"
    assert msg.startswith("[served-cap] verify:") and "superseded" in msg and SECOND_FCID in msg and "nothing filed" in msg, msg
    refusal = nested["refusal"]
    assert refusal.startswith("[served-cap] verify:") and "holds the claim" in refusal and first in refusal, refusal
    assert CV.VERIFY_SUPERSEDES_ENV in refusal and "TINYCAP_VERIFY_SUPERSEDES" == CV.VERIFY_SUPERSEDES_ENV
    assert "superseding" in nested["second_out"] or "supersed" in nested["second_out"]
    assert flip["calls"] == 2                                                        # each scored exactly once
    # the store is the SECOND verifier's alone: B everywhere, its receipt, its block, its RESULT line
    assert _acc_rows(cell_dir) == B_ROWS
    fn = tc.marker_filename(ROW_TASK)
    summary = json.loads((cell_dir / tc.SUMMARY_DIR / fn).read_text())
    outcomes = json.loads((cell_dir / tc.TASK_OUTCOMES_DIR / fn).read_text())
    assert summary[tc.METRIC_ACC] == (N - len(WRONG)) / N
    assert {i["item_id"] for i in outcomes["items"] if i["value"] == 0.0} == WRONG
    log_path = Path(CV.read_generation_sidecar(row_dir, CELL, ROW_TASK, STORE)["log"])
    assert {str(s.id): s.scores["verify"].value for s in read_eval_log(str(log_path)).samples}[FLIP_ITEM] == "C"
    receipt = CV.read_scoring_receipt(row_dir, CELL, ROW_TASK, STORE)
    vp = json.loads((cell_dir / "manifest.json").read_text())[tc.MANIFEST_KEY_ROW_ENGINE]["verification_pass"]
    assert receipt["function_call_id"] == vp["function_call_id"] == SECOND_FCID
    lines = (w.out_root / tc.RESULT_FILENAME).read_text().splitlines()
    assert len(lines) == 2 and lines[1].endswith(f" verified_by={SECOND_FCID}")
    assert not CV.pending_marker_exists(row_dir, CELL, ROW_TASK, STORE) and _claim_of(row_dir) is None
    # the first wrote nothing after losing the claim: no receipt of its own, no archive of its own, no second RESULT
    assert CV.sha256_file(CV.archived_log_path(row_dir, CELL, ROW_TASK, STORE)) == CV.sha256_file(T.FIXTURE)
    assert sorted(q.name for q in (row_dir / CV.GENERATION_DIR).iterdir()) == [
        f"{CELL}__inspect__bigcodebench_full__{STORE}.eval", f"{CELL}__inspect__bigcodebench_full__{STORE}.json",
        f"{CELL}__inspect__bigcodebench_full__{STORE}.scoring.json"]


def test_a_dead_verifiers_stale_claim_blocks_a_resume_until_the_harness_supersedes_it(native_world, monkeypatch):
    """A verifier killed after it claimed leaves its claim; a resume under a new fcid refuses at entry naming the
    holder and its start (the harness must join or supersede), a token naming ANOTHER fcid refuses too, and the
    token naming the dead holder lets the resume take the claim and finish — the sequential recovery is reached
    only through the harness's decision that the old call is terminal or lost."""
    w = native_world
    row_dir = w.tmp_path / "row"
    cell_dir, _m, rec_path, _c = _generate(w, monkeypatch)
    _gpu_pointer(row_dir)
    _local_mechanism(monkeypatch)
    _verifier_env(monkeypatch, row_dir)
    undo = _arm_kill(monkeypatch, ("hook", "scored"))
    with pytest.raises(_Killed):
        srun.main(_verify_argv(w, rec_path))
    undo()
    claim = _claim_of(row_dir)
    assert claim is not None and tuple(sorted(claim)) == tuple(sorted(CV.VERIFY_CLAIM_FIELDS))
    assert claim["function_call_id"] == VERIFY_FCID and claim["run_id"] == RUN_ID and claim["host"]
    assert CV.verify_claim_path(row_dir, CELL, ROW_TASK, STORE).name == f"{CELL}__inspect__bigcodebench_full__{STORE}.claim.json"
    snapshot = SS.relative_layout(w.out_root)
    _verifier_env(monkeypatch, row_dir, fcid=SECOND_FCID)
    with pytest.raises(SystemExit) as ei:
        srun.main(_verify_argv(w, rec_path, fcid=SECOND_FCID))
    msg = str(ei.value)
    assert "holds the claim" in msg and VERIFY_FCID in msg and time.strftime("%Y-%m-%d", time.gmtime(claim["started_at"])) in msg
    monkeypatch.setenv(CV.VERIFY_SUPERSEDES_ENV, "fc-01SOMEONEELSE00000000000")
    with pytest.raises(SystemExit, match="holds the claim"):
        srun.main(_verify_argv(w, rec_path, fcid=SECOND_FCID))
    assert SS.relative_layout(w.out_root) == snapshot and _claim_of(row_dir)["function_call_id"] == VERIFY_FCID
    assert CV.pending_marker_exists(row_dir, CELL, ROW_TASK, STORE)
    monkeypatch.setenv(CV.VERIFY_SUPERSEDES_ENV, VERIFY_FCID)
    srun.main(_verify_argv(w, rec_path, fcid=SECOND_FCID))
    assert completion.is_done(cell_dir, ROW_TASK) and _claim_of(row_dir) is None
    vp = json.loads((cell_dir / "manifest.json").read_text())[tc.MANIFEST_KEY_ROW_ENGINE]["verification_pass"]
    assert vp["function_call_id"] == SECOND_FCID
    assert not CV.pending_marker_exists(row_dir, CELL, ROW_TASK, STORE)


def test_verify_from_a_different_tree_records_both_shas_files_the_store_and_reads_a_legacy_block_as_verified(native_world, monkeypatch, capsys):
    """tinycap-verify-rate-limit (2026-09-30): the verifier's `--tree` stamp is RECORDED beside the store's producer tree,
    never held equal. Before, `--tree <a tree at another sha>` over a store generated at TREE_SHA refused
    (`--tree … is at X, the store was generated at Y`), so `--verify-pending --sha <new landed sha>` over a row dir tree
    082f583 wrote was impossible. Now the filing proceeds — the binding is the generation log's sha, `(id, epoch)` and the
    object digests, none of which is the tree — the verified block carries `verifier_tree_sha` = the stamp and
    `generation_tree_sha` = the producer's, the pointer and RESULT keep the producer's sha as before, and the
    already-verified door accepts a LEGACY nine-key block (a store an earlier tree verified); a stray key is still torn."""
    w = native_world
    cell_dir, before, rec_path, _calls = _generate(w, monkeypatch)
    row_dir = w.tmp_path / "row"
    _gpu_pointer(row_dir)                                          # names TREE_SHA as the generation's tree
    NEW = "cd" * 20
    (w.tmp_path / tc.TREE_SHA_FILENAME).write_text(NEW + "\n")     # the tree the verifier runs: `--tree` = w.tmp_path
    _local_mechanism(monkeypatch)
    _verifier_env(monkeypatch, row_dir)
    capsys.readouterr()
    srun.main(_verify_argv(w, rec_path))
    out = capsys.readouterr().out
    assert f"this verifier runs tree {NEW[:12]}" in out and f"the store was generated at {TREE_SHA[:12]}" in out
    assert "recorded as verification_pass.verifier_tree_sha / generation_tree_sha" in out
    manifest = json.loads((cell_dir / "manifest.json").read_text())
    vp = manifest[tc.MANIFEST_KEY_ROW_ENGINE]["verification_pass"]
    assert tuple(sorted(vp)) == tuple(sorted(tc.VERIFICATION_PASS_VERIFIED_FIELDS)) and len(vp) == 14
    assert vp["verifier_tree_sha"] == NEW and vp["generation_tree_sha"] == TREE_SHA
    # the identity that EXECUTED is recorded — this tree's pinned image (the local synthetic task has no
    # compose, so the contract's), the builder's verify timeout, the binding contract
    assert (vp["executed_image"], vp["executed_verify_timeout_s"], vp["binding_version"]) == (tc.BCB_SANDBOX_IMAGE, 30, "1")
    assert vp["state"] == tc.VERIFICATION_PASS_VERIFIED and vp["function_call_id"] == VERIFY_FCID and vp["max_sandboxes"] == 4 and vp["n_items"] == N
    assert completion.is_done(cell_dir, ROW_TASK) and not CV.pending_marker_exists(row_dir, CELL, ROW_TASK, STORE)
    pointer = json.loads(CV.verify_pointer_path(row_dir, VERIFY_FCID).read_text())
    assert pointer["tree_sha"] == TREE_SHA                          # the pointer and RESULT keep the PRODUCER's sha (the store's)
    lines = (w.out_root / tc.RESULT_FILENAME).read_text().splitlines()
    assert f" result={tc.RESULT_COMPLETE} status=OK sha={TREE_SHA} " in lines[-1] and lines[-1].endswith(f" verified_by={VERIFY_FCID}")
    assert tc.verified_block_keys_problem(vp) is None
    # a LEGACY block (a tree before 2026-09-30 verified this store): the already-verified door reads it verified, exits 0, moves nothing
    legacy = {k: v for k, v in vp.items() if k in tc.VERIFICATION_PASS_VERIFIED_FIELDS_V1}
    assert len(legacy) == 9 and tc.verified_block_keys_problem(legacy) is None
    manifest[tc.MANIFEST_KEY_ROW_ENGINE]["verification_pass"] = legacy
    (cell_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    _verifier_env(monkeypatch, row_dir, fcid="fc-verify-again")
    capsys.readouterr()
    srun.main(_verify_argv(w, rec_path, fcid="fc-verify-again"))
    assert "already verified" in capsys.readouterr().out
    assert json.loads((cell_dir / "manifest.json").read_text())[tc.MANIFEST_KEY_ROW_ENGINE]["verification_pass"] == legacy
    assert (w.out_root / tc.RESULT_FILENAME).read_text().splitlines() == lines           # untouched
    # a block with a stray key is still torn — refused by name
    manifest[tc.MANIFEST_KEY_ROW_ENGINE]["verification_pass"] = {**legacy, "stray": 1}
    (cell_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    _verifier_env(monkeypatch, row_dir, fcid="fc-verify-3")
    with pytest.raises(SystemExit, match="torn verified block"):
        srun.main(_verify_argv(w, rec_path, fcid="fc-verify-3"))
    # the driver's spelling of the tree fact (source pins): the refusal is gone, the record line is there
    src = (Path(__file__).resolve().parents[1] / "src/sisteer/capeval/tinycap/cpu_verify.py").read_text()
    assert "the verifier files through the tree that generated" not in src and "recorded as verification_pass.verifier_tree_sha" in src


def test_a_verifier_tree_whose_pinned_sandbox_image_differs_is_refused_before_any_sandbox_and_an_equal_one_files(native_world, monkeypatch, capsys):
    """A later tree's `verify` over a store generated under another pinned BigCodeBench image
    would score with the NEW image while the block's `sandbox_facts` (the generation plan's) claimed the old — the store
    identity hashes image + verify timeout, so the scores would belong to another experimental object. Now the
    reconstructed task's scorer identity is held byte-equal to the generation's BEFORE scoring: a differing image refuses
    with the SCORER_IDENTITY_MISMATCH mark, the marker stands, the manifest / RESULT / row dir are untouched; the same
    identity from a different tree files, and the block records what executed."""
    w = native_world
    cell_dir, before, rec_path, _calls = _generate(w, monkeypatch)          # the sidecar's plan.sandbox = THIS tree's facts
    row_dir = w.tmp_path / "row"
    _gpu_pointer(row_dir)
    (w.tmp_path / tc.TREE_SHA_FILENAME).write_text("cd" * 20 + "\n")        # a different verifier tree
    _local_mechanism(monkeypatch)
    _verifier_env(monkeypatch, row_dir)
    other_image = "ghcr.io/generality-labs/inspect-eval-bigcodebench:2026-10-01@sha256:" + "b" * 64
    assert other_image != tc.BCB_SANDBOX_IMAGE
    result_before = (w.out_root / tc.RESULT_FILENAME).read_text()
    # (the driver's CLAIM file is excluded: `verify` claims the store before the identity check and the harness's worker
    #  releases a refused attempt's claim after the driver exits — exactly as under the former tree guard)
    gen_files = lambda: sorted(p.name for p in (row_dir / CV.GENERATION_DIR).iterdir() if not p.name.endswith(".claim.json"))
    gen_files_before = gen_files()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tc, "BCB_SANDBOX_IMAGE", other_image)                    # the verifier tree pins another image
        capsys.readouterr()
        with pytest.raises(SystemExit) as ei:
            srun.main(_verify_argv(w, rec_path))
    msg = str(ei.value)
    assert tc.SCORER_IDENTITY_MISMATCH_MARK in msg and "image: this verifier would execute" in msg and "nothing scored, the marker stands" in msg
    assert other_image[-20:] in msg and tc.BCB_SANDBOX_IMAGE[-20:] in msg
    # nothing moved: the marker, the pending block, the RESULT, the row dir's generation files, no receipt / archive
    assert CV.pending_marker_exists(row_dir, CELL, ROW_TASK, STORE)
    manifest = json.loads((cell_dir / "manifest.json").read_text())
    assert manifest[tc.MANIFEST_KEY_ROW_ENGINE]["verification_pass"] == before[tc.MANIFEST_KEY_ROW_ENGINE]["verification_pass"]
    assert manifest[tc.MANIFEST_KEY_ROW_ENGINE]["verification_pass"]["state"] == tc.VERIFICATION_PASS_PENDING
    assert (w.out_root / tc.RESULT_FILENAME).read_text() == result_before
    assert gen_files() == gen_files_before                                   # no receipt, no archive, no merged log
    assert not CV.scoring_receipt_path(row_dir, CELL, ROW_TASK, STORE).exists() and not CV.archived_log_path(row_dir, CELL, ROW_TASK, STORE).exists()
    assert not completion.is_done(cell_dir, ROW_TASK)
    CV.verify_claim_path(row_dir, CELL, ROW_TASK, STORE).unlink(missing_ok=True)   # what the harness's worker does after a refused driver
    # a different verify timeout refuses the same way (it is hashed into the store identity too)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tc, "BCB_VERIFY_TIMEOUT_S", 60)
        _verifier_env(monkeypatch, row_dir)
        with pytest.raises(SystemExit, match="verify_timeout_s: this verifier would execute 60, the store was generated under 30"):
            srun.main(_verify_argv(w, rec_path))
    assert CV.pending_marker_exists(row_dir, CELL, ROW_TASK, STORE)
    CV.verify_claim_path(row_dir, CELL, ROW_TASK, STORE).unlink(missing_ok=True)
    # a bumped binding contract refuses a generation declared under the old one
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tc, "CPU_VERIFY_BINDING_VERSION", "2")
        _verifier_env(monkeypatch, row_dir)
        with pytest.raises(SystemExit, match="binding_version: this verifier would execute '2', the store was generated under '1'"):
            srun.main(_verify_argv(w, rec_path))
    assert CV.pending_marker_exists(row_dir, CELL, ROW_TASK, STORE)
    CV.verify_claim_path(row_dir, CELL, ROW_TASK, STORE).unlink(missing_ok=True)
    # the SAME scorer identity from the different tree: admitted, filed, the executed identity on the block
    _verifier_env(monkeypatch, row_dir)
    capsys.readouterr()
    srun.main(_verify_argv(w, rec_path))
    out = capsys.readouterr().out
    assert "scorer identity held equal to the generation's" in out and "(contract)" in out
    vp = json.loads((cell_dir / "manifest.json").read_text())[tc.MANIFEST_KEY_ROW_ENGINE]["verification_pass"]
    assert vp["state"] == tc.VERIFICATION_PASS_VERIFIED and vp["verifier_tree_sha"] == "cd" * 20 and vp["generation_tree_sha"] == TREE_SHA
    assert (vp["executed_image"], vp["executed_verify_timeout_s"], vp["binding_version"]) == (tc.BCB_SANDBOX_IMAGE, 30, "1")
    assert vp["sandbox_facts"] == tc.bcb_sandbox_facts() and vp["sandbox_facts"]["binding_version"] == "1"
    assert not CV.pending_marker_exists(row_dir, CELL, ROW_TASK, STORE) and completion.is_done(cell_dir, ROW_TASK)
