"""Stage 07's RECEIPT and FILING (demo-import M8; plan review rounds 3-4 P1-3; the two carried findings of 2026-10-10; Q13 (a) with its
addition): reproduce/tools/stage07_receipt.py over a fixture run report whose REC lines are in the launcher's REAL shape and a FakeModal volume
laid out as the WRITER lays it out, then reproduce/tools/stage07_file.py over a tmp registry built from the seeds + the nine imported objects.

Pinned: the REC key-set against scripts/modal/tinycap_served.py's own literals (a REC shape change fails here, never the smoke); the receipt
PULLS both cells' done/summary markers, controls_summary.json and the attempt pointers (under <out>/watch/, beside cells/ — the layout pin),
every container path converted to a volume-root path (the FakeModal REJECTS an unconverted /vol/... argument), copies the
run report, writes a clock-free receipt.json (two runs over an unchanged volume: byte-identical), DERIVES the headline from the none cell's
merged inspect/bigcodebench_full acc cross-checked three ways, and PRINTS the verdict command with --headline and --run-ids filled in; every
named gap refuses with nothing written; the launcher's two other REC branches — a BFCL-only resumption (no new pointer; its BigCodeBench
attempt's pointer from the listing) and a DONE relaunch (reuse, no owner) — are accepted and verified, a reuse credited
only with a same-tree pointer that drove bigcodebench_full in the REC's mode and naming those attempts' call ids, and EVERY control a
recovered cell reuses attributed through the launcher's per-attempt completion records (`watch/completion_<fcid>.json`: what the attempt
finished, in which store) — a DONE relaunch after a resume-then-complete sequence names the BFCL-only call too, and refuses by name when a
reused control has no completing attempt on the volume, and the credited pointers JOINED by call id to those records, so a failed
standup's pointer — published before the drive, no completion record — is `uncredited_attempts` with why, never a call id; the filing composes
the spec from the receipt and the verdict file alone, files it, reads it back three-way, and the identity cases hold: (α) a file-only retry is
byte-identical and a duplicate at the fold; (α′) a retry after a later run overwrote the live log refuses before composing, and composes the
SAME bytes once the log is moved aside; (β) a second execution over the same stores is a new receipt, name and analysis, both rows folding;
(β′) receipt #1's verdict against receipt #2 refuses on the note; (γ) a re-ruled verdict refuses by name after the fold AND before any ingest;
(δ) a verdict whose headline or run ids are not the receipt's refuses; an input object the registry lacks refuses. Synthetic ids and bytes
only (placeholder call ids, TESTSESSION); no Modal call, no host, no write outside tmp."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO, REPRO  # noqa: E402
from test_import_objects import _fresh_registry, _run as _import_run  # noqa: E402
from test_stage_demo_bank_modal import FakeModal  # noqa: E402

sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPRO / "lib"))
import models as MT  # noqa: E402
import stage as ST  # noqa: E402
from sisteer.capeval.tinycap import contracts as C  # noqa: E402
from sisteer.registry import analysis as AN  # noqa: E402
from sisteer.registry import cli  # noqa: E402
from sisteer.registry import db as DB  # noqa: E402
from sisteer.registry import harvest as H  # noqa: E402

LAUNCHER = REPO / "scripts" / "modal" / "tinycap_served.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


RT = _load("stage07_receipt_under_test", REPO / "reproduce" / "tools" / "stage07_receipt.py")
SF = _load("stage07_file_under_test", REPO / "reproduce" / "tools" / "stage07_file.py")

KEY = "qwen35_4b_demo"
VIEW = MT.baseline_view(KEY)
SHA = "0123456789abcdef0123456789abcdef01234567"
APP = f"tinycap-{SHA[:7]}-{KEY}"
RUN_ID = f"{SHA[:7]}-11112222"
DIFF = "constv2-mix-full__lora_r4_ce_minus_ctl"
CELLS = ("none", DIFF)
CONTROLS = (C.CONTROL_BIGCODEBENCH_FULL, C.CONTROL_BFCL_FULL)                   # the paper's FULL sets: what 07 launches
TASK_BCB, TASK_BFCL = C.inspect_task(CONTROLS[0]), C.inspect_task(CONTROLS[1])
assert (TASK_BCB, TASK_BFCL) == ("inspect/bigcodebench_full", "inspect/bfcl_full")
#: the frozen FULL item counts the done markers are held to (contracts.BCB_FULL_N / BFCL_FULL_N: 1,140 / 3,981)
N_BCB, N_BFCL = int(C.BCB_FULL_N), int(C.BFCL_FULL_N)
N_OF = {TASK_BCB: N_BCB, TASK_BFCL: N_BFCL}
assert (N_BCB, N_BFCL) == (1140, 3981)
SAMPLING = C.SAMPLING_PROTOCOL                                                      # the paper's decoding every REC and the launch identity carry
SF_SAMPLING_KEY = "sampling_mode"                                                   # the REC key the receipt holds to SAMPLING (the launcher's SAMPLING_MODE_KEY)
MARK = {t: C.marker_filename(t) for t in (TASK_BCB, TASK_BFCL)}
DEMO = {e["paper_label"]: e for e in json.loads((REPO / "objects" / "DEMO.json").read_text())["objects"]}
DIFF_ID = DEMO["difference_lora"]["object_id"]
OVERRIDE_ID = "demo_smoke"
OVERRIDE_SHA = "ab" * 32
OVERRIDE_PATH = "/TESTHOST/overrides/demo_smoke.env"


def out_of(cell: str) -> str:
    return C.cell_out_dir(C.gen_root_for_engine(C.VOL_GEN_ROOT, C.SERVING_ENGINE_LORA_NATIVE), cell, C.THINKING_OFF, model_slug=KEY)


def vrel(container_path: str) -> str:
    assert container_path.startswith("/vol/")
    return container_path[len("/vol/"):]


def key_of(cell: str, tag: str = "a1") -> str:
    return f"{cell}__controls__p1__{tag}{'0' * 14}__{'5' * 16}"


def done_marker(task: str, n: Optional[int] = None) -> dict:
    n = N_OF[task] if n is None else n
    return {"task": task, "n_items": n, "n_rows": n, "n_requests": n, "n_rejected": 0, "wall_clock_s": 12.5, "finished_at": 1760000000.0}


def summary(task: str, acc: float, n: Optional[int] = None) -> dict:
    n = N_OF[task] if n is None else n
    return {"task": task, "item_set": task.rsplit("/", 1)[1], "n": n, "n_scored": n, "n_unscored": 0, "acc": acc, "framework": "inspect", "engine": "vllm"}


def controls_summary(key: str, accs, *, bcb_task=TASK_BCB, n=None, source_key=None) -> dict:
    s_bcb, s_bfcl = summary(bcb_task, accs[0], N_BCB if n is None else n), summary(TASK_BFCL, accs[1])
    return {"controls": {bcb_task: s_bcb, TASK_BFCL: s_bfcl},
            "provenance": {"model_id": "Qwen/Qwen3.5-4B", "model_revision": "8" * 40, "intervention_hash": "0" * 16, "harness_config_hash": "1" * 16},
            C.SUMMARY_KEY_SOURCES: {bcb_task: {"cell_key": source_key or key, "harness_config_hash": "1" * 16, "pass": 1, "producer_tree_sha": SHA},
                                    TASK_BFCL: {"cell_key": source_key or key, "harness_config_hash": "1" * 16, "pass": 1, "producer_tree_sha": SHA}}}


def pointer(cell: str, fcid: str, *, controls=CONTROLS, tree_sha=None, out=None, run_id=RUN_ID) -> dict:
    tree_sha = SHA if tree_sha is None else tree_sha          # the module's sha at CALL time (a caller may rebind SHA to a real tree's)
    out = out or out_of(cell)
    return {"run_id": run_id, "cell": cell, "function_call_id": fcid, "name": f"tinycap-{KEY}-{cell[:12]}-think_off", "model": KEY, "mode": "off",
            "out_dir": out, "controls": list(controls), "tree_sha": tree_sha, "sandbox_record": f"{out}/watch/sandboxes_{fcid}.jsonl",
            "sandbox_app_name": C.BCB_SANDBOX_APP_NAME, "started_at": 1760000000.0}


def completion(cell: str, fcid: str, *, key=None, controls=CONTROLS, remaining=None, completed=None, status="OK", sandboxed=True, tree_sha=None, out=None,
               run_id=RUN_ID) -> dict:
    """The launcher's per-attempt completion record (tinycap_served.COMPLETION_FIELDS, written at the end of every solo drive): what the attempt
    drove, what was left to do when it started (default: what it drove) and what it finished — control -> the store key it left the control done
    in (default: everything left to do, in `key`) — beside the owner tuple, the REC's status, `sandboxed` and the tree."""
    tree_sha = SHA if tree_sha is None else tree_sha
    out = out or out_of(cell)
    remaining = list(controls) if remaining is None else list(remaining)
    completed = {c: key for c in remaining} if completed is None else dict(completed)
    return {"run_id": run_id, "cell": cell, "function_call_id": fcid, "name": f"tinycap-{KEY}-{cell[:12]}-think_off", "model": KEY, "mode": "off",
            "out_dir": out, "controls": list(controls), "remaining": remaining, "completed": completed, "status": status, "sandboxed": sandboxed,
            "tree_sha": tree_sha, "finished_at": 1760000100.0}


def put_attempt(vol: FakeModal, cell: str, fcid: str, *, key: str, controls=CONTROLS, tree_sha=None, run_id=RUN_ID, remaining=None, completed=None, status="OK",
                sandboxed=True) -> None:
    """An attempt's records on the volume as the launcher leaves them: the attempt pointer (a SANDBOXED attempt's — written before the drive) and
    the completion record (every attempt's — written at the drive's end), both named for the call."""
    out = out_of(cell)
    if sandboxed:
        vol.remote[vrel(RT.attempt_pointer_path(out, fcid))] = json.dumps(pointer(cell, fcid, controls=controls, tree_sha=tree_sha, run_id=run_id)).encode()
    vol.remote[vrel(RT.completion_record_path(out, fcid))] = json.dumps(completion(cell, fcid, key=key, controls=controls, remaining=remaining, completed=completed,
                                                                                      status=status, sandboxed=sandboxed, tree_sha=tree_sha, run_id=run_id)).encode()


def rec_ok(cell: str, key: str, fcid: str, *, sandboxed=True, driven=CONTROLS, tree_sha=None, status="OK", remaining=(), done_in=None, run_id=RUN_ID) -> dict:
    """An OK REC as the launcher builds it: the plan literal, the sandbox keys the solo path adds before the standup, the four keys the run adds
    and the completion record the solo drive names at its end (round 4)."""
    tree_sha = SHA if tree_sha is None else tree_sha
    out = out_of(cell)
    done_in = done_in or {c: key for c in CONTROLS}
    rec = {"name": f"tinycap-{KEY}-{cell[:12]}-think_off", "model": KEY, "cell": cell, "mode": "off", "sha": tree_sha[:7], "pass": 1, "out": out,
           "geometry_keys": list(CONTROLS), "top_logprobs": 1024, "reduction_mode_requested": "stacked", "sampling_mode": SAMPLING,
           "controls_driven": list(driven), "sandbox_owner": {"run_id": run_id, "cell": cell, "function_call_id": fcid}, "sandboxed": sandboxed}
    if sandboxed:
        rec["sandbox_record"] = f"{out}/watch/sandboxes_{fcid}.jsonl"
        rec["sandbox_pointer"] = f"{out}/watch/attempt_{fcid}.json"
    rec["done_markers"] = sorted(MARK.values())
    rec["resume_after"] = {"expected": {"cell": cell, "mode": "off", "pass": 1, "tree_sha": tree_sha, "served": {}, "limits": {}, "samples": 1},
                           "stores": {key: None},
                           "controls": {c: {"tasks": [TASK_BCB if c == "bigcodebench_full" else TASK_BFCL], "done_in": done_in[c], "why_not": {}} for c in CONTROLS}}
    rec["controls_still_remaining"] = list(remaining)
    rec["status"] = status
    rec["completion_record"] = f"{out}/watch/completion_{fcid}.json"
    return rec


def rec_done(cell: str, key: str, *, tree_sha=None) -> dict:
    """The DONE branch: every task validly done at resume time — `resume` / `controls_remaining`, no owner, no markers list, no GPU minute."""
    tree_sha = SHA if tree_sha is None else tree_sha
    out = out_of(cell)
    return {"name": f"tinycap-{KEY}-{cell[:12]}-think_off", "model": KEY, "cell": cell, "mode": "off", "sha": tree_sha[:7], "pass": 1, "out": out,
            "geometry_keys": list(CONTROLS), "top_logprobs": 1024, "reduction_mode_requested": "stacked", "sampling_mode": SAMPLING,
            "controls_remaining": [], "resume": {"expected": {"cell": cell, "mode": "off", "pass": 1, "tree_sha": tree_sha}, "stores": {key: None},
                                                 "controls": {c: {"tasks": [TASK_BCB if c == "bigcodebench_full" else TASK_BFCL], "done_in": key, "why_not": {}} for c in CONTROLS}},
            "status": "DONE"}


def lay_down(vol: FakeModal, cell: str, key: str, accs, *, fcids=(), n=None, partial=False, bcb_task=TASK_BCB, source_key=None, pointer_in_cells=False):
    """The writer's layout on the volume: <out>/cells/<key>/{done,summary}/<marker>, <out>/<cell>/controls_summary.json, <out>/watch/attempt_<fcid>.json
    and <out>/watch/completion_<fcid>.json (every attempt in `fcids` finished every control it drove, in `key`)."""
    out = out_of(cell)
    for task in (TASK_BCB, TASK_BFCL):
        acc = accs[0] if task == TASK_BCB else accs[1]
        vol.remote[vrel(f"{out}/cells/{key}/{C.DONE_DIR}/{MARK[task]}")] = json.dumps(done_marker(task, n if task == TASK_BCB else None)).encode()
        vol.remote[vrel(f"{out}/cells/{key}/{C.SUMMARY_DIR}/{MARK[task]}")] = json.dumps(summary(task, acc, n if task == TASK_BCB else None)).encode()
        if partial:
            vol.remote[vrel(f"{out}/cells/{key}/{C.PARTIAL_DIR}/{MARK[task]}")] = b"{}"
    vol.remote[vrel(f"{out}/{cell}/{C.CONTROLS_SUMMARY_FILENAME}")] = json.dumps(controls_summary(key, accs, bcb_task=bcb_task, n=n, source_key=source_key)).encode()
    for fcid, controls in fcids:
        where = f"{out}/cells/{key}/watch/attempt_{fcid}.json" if pointer_in_cells else RT.attempt_pointer_path(out, fcid)
        vol.remote[vrel(where)] = json.dumps(pointer(cell, fcid, controls=controls)).encode()
        vol.remote[vrel(RT.completion_record_path(out, fcid))] = json.dumps(completion(cell, fcid, key=key, controls=controls)).encode()


def run_report(recs) -> str:
    return (f"[tinycap] launching {len(recs)} configuration(s) @ sha {SHA[:7]}, model {KEY}, pass 1\n"
            f"[tinycap] {len(recs)}/{len(recs)} container(s) OK\n" + "".join(json.dumps(r) + "\n" for r in recs))


VIEW_CELL_OF = {"none": "none", DIFF: "difference_lora"}        # the table's name of each launcher token the fixture runs


def identity_doc(*, sha=None, cells=CELLS, objects=None, override=OVERRIDE_ID, environment=None, volume="twosi-fits", app=None, expect=None, controls=None,
                 sampling=SAMPLING, slug=KEY) -> dict:
    """The LAUNCH's identity as `07 --phase run` writes it beside the captured stdout: the tree, the cells by both names,
    the object id of every fitted cell at that tree, the app, the volume and environment, the run-overrides document, the expect-hashes."""
    toks = list(cells)
    return {"schema": RT.RUN_IDENTITY_SCHEMA, "tree_sha": SHA if sha is None else sha, "model_key": KEY, "slug": slug, "view": VIEW,
            "cells": [VIEW_CELL_OF.get(c, c) for c in toks], "launcher_cells": toks, "objects": {DIFF: DIFF_ID} if objects is None else objects,
            "app": app or APP, "volume": volume, "modal_environment": environment, "run_overrides": {"id": override, "sha256": OVERRIDE_SHA, "path": OVERRIDE_PATH},
            "expect_hashes": expect, "controls": ",".join(CONTROLS) if controls is None else controls, "sampling": sampling, "modes": "off",
            "serving_path": "lora_native", "allow_extra": False, "written_by": "07_bcb_bfcl.sh --phase run"}


def write_identity(state: Path, **kw) -> Path:
    p = state / RT.RUN_IDENTITY_NAME
    p.write_text(json.dumps(identity_doc(**kw), indent=1, sort_keys=True) + "\n")
    return p


def make_world(tmp: Path, *, accs=(0.455, 0.61), none_tag="a1", diff_tag="b2"):
    state = tmp / "state" / KEY
    state.mkdir(parents=True, exist_ok=True)
    vol = FakeModal()
    k_none, k_diff = key_of("none", none_tag), key_of(DIFF, diff_tag)
    f_none, f_diff = "fc-none-0001", "fc-diff-0002"
    lay_down(vol, "none", k_none, accs, fcids=[(f_none, CONTROLS)])
    lay_down(vol, DIFF, k_diff, (0.41, 0.58), fcids=[(f_diff, CONTROLS)])
    recs = [rec_ok("none", k_none, f_none), rec_ok(DIFF, k_diff, f_diff)]
    (state / RT.RUN_REPORT_NAME).write_text(run_report(recs))
    write_identity(state)
    return {"state": state, "vol": vol, "keys": {"none": k_none, DIFF: k_diff}, "fcids": {"none": f_none, DIFF: f_diff}, "recs": recs, "accs": accs}


def receipt(w, *, dry=False, cells=CELLS, sha=None, log=None, view_cells=None, override_id=OVERRIDE_ID, environment=None, runner=None, app=None, volume="twosi-fits",
            run_identity=None, controls=None):
    """The receipt as 07 calls it: this call's resolution rides along and is verified against the launch's identity (override_id / app: None = not passed)."""
    lines = []
    vc = list(view_cells) if view_cells is not None else [VIEW_CELL_OF.get(c, c) for c in cells]
    out = RT.run(state=w["state"], view=VIEW, sha=SHA if sha is None else sha, cells=list(cells), view_cells=vc, override_id=override_id,
                 controls=list(controls) if controls is not None else list(CONTROLS),
                 app=APP if app is None else app, run_report=w["state"] / RT.RUN_REPORT_NAME, run_identity=run_identity, volume=volume, modal_bin=Path("/nonexistent/modal"),
                 dry_run=dry, environment=environment, runner=runner or w["vol"], log=(log or lines.append))
    out["log"] = lines
    return out


def store_bytes(state: Path) -> dict:
    """Every file under the published store, by relative path -> bytes (the receipt's evidence as it stands)."""
    store = state / RT.STORE_DIR
    return {str(p.relative_to(store)): p.read_bytes() for p in sorted(store.rglob("*")) if p.is_file()}


def receipt_sha(w) -> str:
    return hashlib.sha256((w["state"] / RT.STORE_DIR / RT.RECEIPT_NAME).read_bytes()).hexdigest()


def rule(w, *, headline=None, rng="0.3,0.6", run_ids=None, note=None, sha12=None) -> int:
    rec = json.loads((w["state"] / RT.STORE_DIR / RT.RECEIPT_NAME).read_text())
    sha12 = sha12 or receipt_sha(w)[:12]
    hv = rec["headline"]["value"] if headline is None else headline
    ids = run_ids if run_ids is not None else rec["store_cell_keys"]
    return ST.main(["verdict", "--state-root", str(w["state"]), "--family", f"bcb_bfcl/{VIEW}", "--headline", repr(hv), "--range", rng,
                    "--run-ids", ",".join(ids), "--note", note if note is not None else f"headline: none {TASK_BCB} acc; receipt {sha12}"])


def file_phase(w, root: str, *, dry=False, launcher_cells=CELLS, cells=("none", "difference_lora"), override_id=OVERRIDE_ID):
    """The file phase as 07 calls it: this call's cells / tokens / OVERRIDE_ID ride along as cross-checks (None = not passed). Since 2026-10-12 a
    filing requires the VERIFIED MIRROR of the receipt it files (`07 --phase mirror`; tests/test_stage07_mirror.py pins that requirement), so the
    world's current receipt is mirrored first over its own volume — tests/stage07_testlib.ensure_mirror, a no-op once the mirror stands; a world
    whose receipt cannot be mirrored (an older schema, a broken store) is left to the file phase's own refusal."""
    lines = []
    try:
        import stage07_testlib as _L
        _L.ensure_mirror(w)
    except Exception as e:  # noqa: BLE001 — the file phase names its own refusal below
        lines.append(f"[testlib] no mirror built for this receipt: {e}")
    try:
        rc = SF.run(state=w["state"], view=VIEW, key=KEY, registry_root=root, override_id=override_id, cells=None if cells is None else list(cells),
                    launcher_cells=None if launcher_cells is None else list(launcher_cells), table=None, repo=REPO, dry_run=dry, log=lines.append)
    except SF.FileRefusal as e:
        return 2, lines, str(e)
    return rc, lines, None


def _count(root: str, sql: str, *args):
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        return conn.execute(sql, args).fetchone()[0]
    finally:
        conn.close()


def _filings(root: str) -> list:
    base = Path(root) / "records" / "analyses"
    return sorted(str(p) for p in base.glob("*/*") if p.is_dir()) if base.exists() else []


def _registry_with_objects(root: Path) -> str:
    """A seeds-built registry with the nine committed objects imported (reproduce/tools/import_objects.py registry) and folded."""
    root = _fresh_registry(root)
    r = _import_run("registry", "--model", KEY, "--registry-root", str(root), "--spool", str(root / "records"))
    assert r.returncode == 0 and r.stdout.count("emitted ") == 9, r.stdout[-2000:] + r.stderr[-2000:]
    rep = H.ingest_once(str(root), workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0 and _count(str(root), "SELECT count(*) FROM object WHERE object_id IN (?, 'none')", DIFF_ID) == 2
    return str(root)


@pytest.fixture(scope="module")
def registry(tmp_path_factory):
    """The module's shared registry: the tests that FILE into it read their own filing back by position, so a test that files its own case builds its own."""
    return _registry_with_objects(tmp_path_factory.mktemp("reg") / "root")


# ------------------------------------------------------------------------------------------------------------ the key-set pin
def test_the_fixture_recs_carry_the_launchers_own_keys_and_the_pointer_layout_is_the_launchers():
    """The REC shape is read off scripts/modal/tinycap_served.py's literals (never retyped): the plan literal's keys, the sandbox keys the solo
    path adds, the four keys the run adds, the DONE branch's two, POINTER_FIELDS and the watch dir / pointer name — so a REC shape change
    fails HERE, not the smoke."""
    src = LAUNCHER.read_text(encoding="utf-8")
    m = re.search(r'\n    rec = \{"name": cfg\["name"\](.*?)\n    # native-serving-default', src, flags=re.S)
    assert m, "the plan literal `rec = {...}` moved"
    literal_keys = set(re.findall(r'"([a-z_]+)":', '"name":' + m.group(1)))
    assert literal_keys == {"name", "model", "cell", "mode", "sha", "pass", "out", "geometry_keys", "top_logprobs", "reduction_mode_requested"}
    assigned = set(re.findall(r'\brec\["([a-z_]+)"\]\s*=', src))
    for k in ("controls_driven", "sandbox_owner", "sandboxed", "sandbox_record", "sandbox_pointer", "done_markers", "resume_after", "controls_still_remaining",
              "status", "controls_remaining", "resume", "completion_record"):
        assert k in assigned, k
    assert 'rec["status"] = "DONE"' in src and "driven = wanted if rung else remaining" in src and "if sandboxed:" in src
    ok, done = rec_ok("none", key_of("none"), "fc"), rec_done("none", key_of("none"))
    assert literal_keys <= set(ok) and literal_keys <= set(done)
    assert {"controls_driven", "sandbox_owner", "sandboxed", "sandbox_record", "sandbox_pointer", "done_markers", "resume_after", "controls_still_remaining", "status",
            "completion_record"} <= set(ok)
    assert {"controls_remaining", "resume", "status"} <= set(done) and "sandbox_owner" not in done and "completion_record" not in done
    # the sampling mode the receipt holds every REC to (stage07-receipt/7): the launcher's SAMPLING_MODE_KEY literal, stamped on the REC from the
    # plan's resolved mode BEFORE the DONE branch returns — so an OK run's REC and a DONE reuse's alike carry it, and the fixtures do too
    assert re.search(r'^SAMPLING_MODE_KEY = "sampling_mode"$', src, flags=re.M) and RT.REC_SAMPLING_MODE_KEY == "sampling_mode" == SF_SAMPLING_KEY
    assert "rec[SAMPLING_MODE_KEY] = sampling_mode" in src and src.index("rec[SAMPLING_MODE_KEY] = sampling_mode") < src.index('rec["status"] = "DONE"')
    assert ok["sampling_mode"] == done["sampling_mode"] == RT.SAMPLING_MODE == C.SAMPLING_PROTOCOL == "protocol"
    pf = re.search(r"POINTER_FIELDS: Tuple\[str, \.\.\.\] = \((.*?)\)", src, flags=re.S)
    assert tuple(re.findall(r'"([a-z_]+)"', pf.group(1))) == RT.POINTER_FIELDS == tuple(pointer("none", "fc"))
    assert re.search(r'^WATCH_DIR = "watch"$', src, flags=re.M) and RT.WATCH_DIR == "watch"
    assert 'return Path(out_dir) / WATCH_DIR / f"attempt_{function_call_id}.json"' in src
    assert RT.attempt_pointer_path(out_of("none"), "fc") == f"{out_of('none')}/watch/attempt_fc.json"
    # round 4: the completion record's fields and layout are the launcher's, and the solo drive writes it at its end (after the status is final,
    # before the RESULT stamp), on the solo path only
    cf = re.search(r"COMPLETION_FIELDS: Tuple\[str, \.\.\.\] = \((.*?)\)", src, flags=re.S)
    assert tuple(re.findall(r'"([a-z_]+)"', cf.group(1))) == RT.COMPLETION_FIELDS == tuple(completion("none", "fc", key=key_of("none")))
    assert 'return Path(out_dir) / WATCH_DIR / f"completion_{function_call_id}.json"' in src
    assert RT.completion_record_path(out_of("none"), "fc") == f"{out_of('none')}/watch/completion_fc.json" == ok["completion_record"]
    drive = src[src.index("def _drive_cell("):]
    assert "if not shared_standup:\n        # the per-attempt COMPLETION record" in drive
    assert drive.index('rec["status"] = rec.pop("status_override")') < drive.index('rec["completion_record"] = str(write_completion_record(') < drive.index("result_stamp_line(rec")
    assert "OK" in C.REC_OK_STATUSES and "DONE" in C.REC_OK_STATUSES and RT.REC_DONE == "DONE"
    assert re.search(r'^OBJECT_IDENTITY_CHECK = "object_identity_and_header_guard"$', src, flags=re.M)      # the preflight check 07 parses


# ------------------------------------------------------------------------------------------------------------ the receipt
def test_the_receipt_pulls_the_stores_converts_every_path_derives_the_headline_and_prints_the_verdict_command(tmp_path):
    w = make_world(tmp_path)
    dry = receipt(w, dry=True)
    assert dry["dry"] is True and w["vol"].calls == [] and not (w["state"] / RT.STORE_DIR).exists()
    out = receipt(w)
    store = w["state"] / RT.STORE_DIR
    rec = json.loads((store / RT.RECEIPT_NAME).read_text())
    k_none, k_diff = w["keys"]["none"], w["keys"][DIFF]
    for tok, key in w["keys"].items():
        for t in (TASK_BCB, TASK_BFCL):
            assert (store / tok / key / C.DONE_DIR / MARK[t]).is_file() and (store / tok / key / C.SUMMARY_DIR / MARK[t]).is_file()
        assert (store / tok / C.CONTROLS_SUMMARY_FILENAME).is_file() and (store / tok / "watch" / f"attempt_{w['fcids'][tok]}.json").is_file()
        assert (store / tok / "watch" / f"completion_{w['fcids'][tok]}.json").is_file()
    assert (store / RT.RUN_REPORT_COPY).read_bytes() == (w["state"] / RT.RUN_REPORT_NAME).read_bytes()
    assert rec["run_report_sha256"] == hashlib.sha256((store / RT.RUN_REPORT_COPY).read_bytes()).hexdigest()
    assert rec["tree_sha"] == SHA and rec["app"] == APP and rec["cells"] == list(CELLS) and rec["view"] == VIEW
    # the launch's identity the filing derives from (code review rounds 1-2): the cells by both names, the objects at the tree, the overrides
    # document, the environment, the expect-hashes — copied from 07_run_identity.json, never this call's; schema 6 (round 3: a reuse names its
    # attempts; round 4: every reused control attributed through the completion records; round 5: the credited pointers joined to those records)
    assert rec["schema"] == RT.RECEIPT_SCHEMA == SF.RECEIPT_SCHEMA == "stage07-receipt/7"
    assert rec["slug"] == KEY and rec["sampling"] == SAMPLING == "protocol" and all(r["sampling_mode"] == SAMPLING for r in rec["recs"].values())
    assert rec["view_cells"] == ["none", "difference_lora"] and rec["run_overrides_id"] == OVERRIDE_ID and rec["modal_environment"] is None and rec["other_tree_attempts"] == []
    assert rec["uncredited_attempts"] == [] and rec["reused_attempts"] == []
    assert rec["objects"] == {DIFF: DIFF_ID} and rec["run_overrides_sha256"] == OVERRIDE_SHA and rec["run_overrides_path"] == OVERRIDE_PATH
    assert rec["expect_hashes"] is None and rec["run_identity_sha256"] == hashlib.sha256((w["state"] / RT.RUN_IDENTITY_NAME).read_bytes()).hexdigest()
    assert not (w["state"] / RT.STORE_BUILD_DIR).exists() and not (w["state"] / RT.STORE_ARCHIVE_DIR).exists()
    assert rec["store_cell_keys"] == sorted([k_none, k_diff]) and rec["store_out_dirs"] == sorted([out_of("none"), out_of(DIFF)])
    assert rec["modal_call_ids"] == sorted(w["fcids"].values()) and rec["prior_attempts"] == [] and rec["execution"] == "new"
    assert rec["prior_completions"] == [] and rec["credited_completions"] == []                  # an OK cell drove everything itself: nothing to credit
    assert rec["headline"] == {"cell": "none", "task": TASK_BCB, "metric": "acc", "value": 0.455, "n": N_BCB, "n_scored": N_BCB, "store_key": k_none}
    assert rec["bfcl"]["acc"] == 0.61 and rec["bfcl"]["n"] == N_BFCL and rec["n_items_per_task"] == {TASK_BCB: N_BCB, TASK_BFCL: N_BFCL}
    assert rec["recs"][DIFF]["sandbox_owner"]["function_call_id"] == w["fcids"][DIFF] and rec["recs"][DIFF]["execution"] == "new"
    assert rec["pointers"]["none"]["function_call_id"] == w["fcids"]["none"] and rec["pointers"]["none"]["path"] == RT.attempt_pointer_path(out_of("none"), w["fcids"]["none"])
    # round 4: each OK REC's own completion record, pulled by exact path and held to the REC — the attempt finished both controls in the trusted store
    for tok, key in w["keys"].items():
        own = rec["completions"][tok]
        assert own["path"] == RT.completion_record_path(out_of(tok), w["fcids"][tok]) == rec["recs"][tok]["completion_record"]
        assert own["function_call_id"] == w["fcids"][tok] and own["completed"] == {c: key for c in CONTROLS} and own["status"] == "OK" and own["sandboxed"] is True
        assert own["controls"] == list(CONTROLS) and own["remaining"] == list(CONTROLS) and own["tree_sha"] == SHA and "finished_at" not in own
    # every pulled path: the REC's container path recorded, the volume-root path pulled, no /vol/ ever reached the CLI (the FakeModal asserts it)
    assert rec["pulled"] and all(p.startswith("/vol/tinycap/gen/") and v["volume_path"] == p[len("/vol"):] for p, v in rec["pulled"].items())
    assert all(hashlib.sha256((store / v["local"]).read_bytes()).hexdigest() == v["sha256"] for v in rec["pulled"].values())
    assert len(rec["pulled"]) == 2 * (2 * 2 + 1 + 1 + 1)                                        # per cell: 2 markers x 2 tasks, the merged file, the pointer, the completion record
    # clock-free: no wall-clock key anywhere in the receipt; two runs over an unchanged volume are byte-identical
    def keys(d):
        if isinstance(d, dict):
            for k, v in d.items():
                yield k
                yield from keys(v)
        elif isinstance(d, list):
            for v in d:
                yield from keys(v)
    assert not {k for k in keys(rec)} & {"written_utc", "written_at", "utc", "started_utc", "verified_utc", "finished_at", "started_at", "pulled_at"}
    first = (store / RT.RECEIPT_NAME).read_bytes()
    receipt(w)
    assert (store / RT.RECEIPT_NAME).read_bytes() == first
    # the verdict command: the headline and the store keys filled in, --range the one blank, the note binding it to THIS receipt
    cmd = out["verdict_command"]
    assert f"--family bcb_bfcl/{VIEW} --headline 0.455 --range <lo,hi> --run-ids {','.join(sorted([k_none, k_diff]))} --note \"headline: none {TASK_BCB} acc; receipt {out['receipt_sha256'][:12]}\"" in cmd
    assert cmd.split(" verdict ", 1)[0].endswith("reproduce/lib/stage.py") and f"--state-root {w['state']}" in cmd
    assert any("verdict (fill --range, the one blank)" in ln for ln in out["log"])


@pytest.mark.parametrize("case", ["incomplete", "marker_missing", "n_items_5", "partial", "done_in_other_key", "tree_sha", "merged_absent", "merged_task",
                                  "merged_n", "merged_acc", "merged_source", "no_pointer_key", "pointer_absent", "pointer_fcid", "pointer_run_id",
                                  "pointer_out_dir", "pointer_tree_sha", "pointer_under_cells", "no_completion_key", "completion_layout", "completion_absent",
                                  "completion_fcid", "completion_store", "completion_outside", "no_report", "no_none", "rec_missing", "rec_outside",
                                  "view_cells_short", "ladder", "no_identity", "identity_schema", "identity_tree", "identity_cells", "identity_view_cells",
                                  "identity_volume", "identity_environment", "identity_app", "override_mismatch", "identity_no_override", "identity_objects_short",
                                  "identity_objects_extra", "identity_objects_shape",
                                  "n_items_200", "rec_card", "rec_no_sampling", "identity_sampling", "identity_controls", "identity_no_slug"])
def test_every_gap_refuses_by_name_and_writes_no_receipt(tmp_path, case):
    w = make_world(tmp_path)
    vol, state = w["vol"], w["state"]
    k_none, f_none = w["keys"]["none"], w["fcids"]["none"]
    out_none = out_of("none")
    recs = list(w["recs"])
    cells = CELLS
    sha = SHA
    kw = {}
    want = {
        "incomplete": "status 'INCOMPLETE' is not OK/DONE", "marker_missing": "cannot pull", "n_items_5": f"n_items 5 != the frozen full item count {N_BCB}",
        # the full sets under the paper's protocol decoding — a 200-item marker, a REC under the launcher's card sampling, a launch
        # identity under another sampling mode or with other controls, each refused by name
        "n_items_200": f"n_items 200 != the frozen full item count {N_BCB}",
        # the launcher stamps every REC with its plan's mode: `card` is a run under the model card's sampling; a missing key an older launcher's
        "rec_card": "sampling_mode 'card' is not the paper's 'protocol' decoding", "rec_no_sampling": "sampling_mode None is not the paper's 'protocol' decoding",
        "identity_sampling": "the run was launched under sampling 'card', not the paper's 'protocol' decoding",
        "identity_controls": f"--controls bigcodebench_200,bfcl_200 is not the run's: the run was launched with controls {','.join(CONTROLS)}",
        "identity_no_slug": "names no launcher slug",
        "partial": "a partial marker sits beside the done marker", "done_in_other_key": "cannot pull", "tree_sha": "is not the tree",
        "merged_absent": "has no 'inspect/bigcodebench_full' entry", "merged_task": "names task 'inspect/bfcl_full'", "merged_n": f"has n {N_BCB - 1}, not {N_BCB}",
        "merged_acc": "disagrees with the pulled summary", "merged_source": "cell_key is 'other-key', not the REC's trusted store",
        "no_pointer_key": "sandboxed but no sandbox_pointer", "pointer_absent": "is not in", "pointer_fcid": "field function_call_id",
        "pointer_run_id": "field run_id", "pointer_out_dir": "field out_dir", "pointer_tree_sha": "tree_sha", "pointer_under_cells": "is not in",
        # round 4: an OK REC's own completion record — named, in the launcher's layout, on the volume, the REC's own call, finishing every control it
        # drove in the trusted store, claiming nothing it did not drive
        "no_completion_key": "no completion_record — an OK REC of this launcher names its attempt's completion record",
        "completion_layout": f"completion_record '{out_none}/elsewhere/completion_{f_none}.json' is not the launcher's layout",
        "completion_absent": "per-control completion statement is missing",
        "completion_fcid": f"completion record {RT.completion_record_path(out_none, f_none)} field function_call_id = 'fc-other'",
        "completion_store": "an OK attempt finished every control it drove in the store the REC trusts",
        "completion_outside": "claims ['bigcodebench_full'] completed, controls the attempt did not drive (['bfcl_full'])",
        "no_report": "no run report at", "no_none": "has no 'none' cell", "rec_missing": "no REC for cell", "ladder": "truncation-ladder rung",
        "rec_outside": "carries RECs for cells outside --cells", "view_cells_short": "are not the same cells by two names",
        # the launch's identity: absent, another schema, or a field this call disagrees with — each named, with what to pass
        "no_identity": "no run identity at", "identity_schema": "is not a 'stage07-run-identity/1' document", "identity_tree": "the run was launched at tree",
        "identity_cells": "--cells none is not the run's: the run was launched with cells none,constv2-mix-full__lora_r4_ce_minus_ctl (the launcher's tokens",
        "identity_view_cells": "--view-cells none,difference_relu is not the run's", "identity_volume": "--volume other-vol is not the run's: the run wrote volume twosi-fits",
        "identity_environment": "--environment other-env is not the run's: the run ran under Modal environment none", "identity_app": "--app tinycap-other is not the run's",
        "override_mismatch": f"--override-id paper_defaults is not the run's: the run was launched under run-overrides {OVERRIDE_ID} ({OVERRIDE_PATH}",
        "identity_no_override": "names no run-overrides document", "identity_objects_short": "`objects` does not cover exactly the run's fitted cells",
        "identity_objects_extra": "objects for cells the run did not launch ['constv2ctl2-full__lora_r4_ce']", "identity_objects_shape": "is not a map of launcher token -> 64-hex object id",
    }[case]
    if case == "incomplete":
        recs[0] = rec_ok("none", k_none, f_none, status="INCOMPLETE", remaining=["bfcl_full"])
    elif case == "marker_missing":
        del vol.remote[vrel(f"{out_none}/cells/{k_none}/{C.DONE_DIR}/{MARK[TASK_BFCL]}")]
    elif case == "n_items_5":
        vol.remote[vrel(f"{out_none}/cells/{k_none}/{C.DONE_DIR}/{MARK[TASK_BCB]}")] = json.dumps(done_marker(TASK_BCB, 5)).encode()
    elif case == "n_items_200":
        vol.remote[vrel(f"{out_none}/cells/{k_none}/{C.DONE_DIR}/{MARK[TASK_BCB]}")] = json.dumps(done_marker(TASK_BCB, 200)).encode()
    elif case == "rec_card":
        recs[0] = rec_ok("none", k_none, f_none); recs[0]["sampling_mode"] = "card"
    elif case == "rec_no_sampling":
        recs[0] = rec_ok("none", k_none, f_none); del recs[0]["sampling_mode"]
    elif case == "identity_sampling":
        write_identity(state, sampling="card")
    elif case == "identity_controls":
        kw["controls"] = ["bigcodebench_200", "bfcl_200"]
    elif case == "identity_no_slug":
        write_identity(state, slug="")
    elif case == "partial":
        vol.remote[vrel(f"{out_none}/cells/{k_none}/{C.PARTIAL_DIR}/{MARK[TASK_BCB]}")] = b"{}"
    elif case == "done_in_other_key":
        recs[0] = rec_ok("none", k_none, f_none, done_in={"bigcodebench_full": key_of("none", "zz"), "bfcl_full": k_none})
    elif case == "tree_sha":
        sha = "f" * 40
        write_identity(state, sha=sha)                  # the identity agrees with the call; the REC's tree does not: the REC check's refusal
    elif case == "merged_absent":
        vol.remote[vrel(f"{out_none}/none/{C.CONTROLS_SUMMARY_FILENAME}")] = json.dumps({"controls": {TASK_BFCL: summary(TASK_BFCL, 0.6)}, "provenance": {}, "sources": {}}).encode()
    elif case == "merged_task":
        cs = controls_summary(k_none, (0.455, 0.61)); cs["controls"][TASK_BCB]["task"] = TASK_BFCL
        vol.remote[vrel(f"{out_none}/none/{C.CONTROLS_SUMMARY_FILENAME}")] = json.dumps(cs).encode()
    elif case == "merged_n":
        cs = controls_summary(k_none, (0.455, 0.61)); cs["controls"][TASK_BCB]["n"] = N_BCB - 1
        vol.remote[vrel(f"{out_none}/none/{C.CONTROLS_SUMMARY_FILENAME}")] = json.dumps(cs).encode()
    elif case == "merged_acc":
        cs = controls_summary(k_none, (0.455, 0.61)); cs["controls"][TASK_BCB]["acc"] = 0.5
        vol.remote[vrel(f"{out_none}/none/{C.CONTROLS_SUMMARY_FILENAME}")] = json.dumps(cs).encode()
    elif case == "merged_source":
        vol.remote[vrel(f"{out_none}/none/{C.CONTROLS_SUMMARY_FILENAME}")] = json.dumps(controls_summary(k_none, (0.455, 0.61), source_key="other-key")).encode()
    elif case == "no_pointer_key":
        recs[0] = {k: v for k, v in rec_ok("none", k_none, f_none).items() if k != "sandbox_pointer"}
    elif case == "pointer_absent":
        del vol.remote[vrel(RT.attempt_pointer_path(out_none, f_none))]
    elif case in ("pointer_fcid", "pointer_run_id", "pointer_out_dir", "pointer_tree_sha"):
        doc = pointer("none", f_none)
        doc.update({"pointer_fcid": {"function_call_id": "fc-other"}, "pointer_run_id": {"run_id": "other-run"}, "pointer_out_dir": {"out_dir": out_of(DIFF)},
                    "pointer_tree_sha": {"tree_sha": "e" * 40}}[case])
        vol.remote[vrel(RT.attempt_pointer_path(out_none, f_none))] = json.dumps(doc).encode()
    elif case == "pointer_under_cells":
        # the LAYOUT pin: a pointer under cells/<key>/watch/ (the first fold's wrong directory) is NOT found; the writer's path is beside cells/
        del vol.remote[vrel(RT.attempt_pointer_path(out_none, f_none))]
        vol.remote[vrel(f"{out_none}/cells/{k_none}/watch/attempt_{f_none}.json")] = json.dumps(pointer("none", f_none)).encode()
        assert Path(RT.attempt_pointer_path(out_none, f_none)).relative_to(out_none) == Path("watch") / f"attempt_{f_none}.json"
    elif case == "no_completion_key":
        recs[0] = {k: v for k, v in rec_ok("none", k_none, f_none).items() if k != "completion_record"}
    elif case == "completion_layout":
        recs[0] = dict(rec_ok("none", k_none, f_none), completion_record=f"{out_none}/elsewhere/completion_{f_none}.json")
    elif case == "completion_absent":
        del vol.remote[vrel(RT.completion_record_path(out_none, f_none))]
    elif case == "completion_fcid":
        doc = completion("none", f_none, key=k_none); doc["function_call_id"] = "fc-other"
        vol.remote[vrel(RT.completion_record_path(out_none, f_none))] = json.dumps(doc).encode()
    elif case == "completion_store":
        doc = completion("none", f_none, key=k_none, completed={"bigcodebench_full": key_of("none", "zz"), "bfcl_full": k_none})
        vol.remote[vrel(RT.completion_record_path(out_none, f_none))] = json.dumps(doc).encode()
    elif case == "completion_outside":
        # a sandboxed attempt that drove BFCL alone (its controls agree with the REC's) whose record claims BigCodeBench too
        recs[0] = rec_ok("none", k_none, f_none, driven=("bfcl_full",))
        doc = completion("none", f_none, key=k_none, controls=("bfcl_full",), completed={"bigcodebench_full": k_none, "bfcl_full": k_none})
        vol.remote[vrel(RT.completion_record_path(out_none, f_none))] = json.dumps(doc).encode()
    elif case == "no_report":
        (state / RT.RUN_REPORT_NAME).unlink()
    elif case == "no_none":
        cells = (DIFF,)
    elif case == "rec_missing":
        recs = recs[1:]
    elif case == "rec_outside":
        # a REC for a cell --cells does not name: the receipt is the WHOLE execution's evidence, never a subset of it
        recs.append(rec_ok("constv2ctl2-full__lora_r4_ce", key_of("constv2ctl2-full__lora_r4_ce", "c3"), "fc-ctl-0007"))
    elif case == "view_cells_short":
        kw["view_cells"] = ["none"]
    elif case == "ladder":
        recs[0] = rec_ok("none", k_none, f_none, status=C.LADDER_REC_STATUS)
    elif case == "no_identity":
        (state / RT.RUN_IDENTITY_NAME).unlink()
    elif case == "identity_schema":
        doc = identity_doc(); doc["schema"] = "stage07-run-identity/0"
        (state / RT.RUN_IDENTITY_NAME).write_text(json.dumps(doc))
    elif case == "identity_tree":
        write_identity(state, sha="d" * 40)             # the launch was at another tree than this call's --sha
    elif case == "identity_cells":
        kw["view_cells"] = ["none"]; cells = ("none",)  # a one-cell call against a two-cell launch
        recs = recs[:1]
    elif case == "identity_view_cells":
        kw["view_cells"] = ["none", "difference_relu"]
    elif case == "identity_volume":
        kw["volume"] = "other-vol"
    elif case == "identity_environment":
        kw["environment"] = "other-env"
    elif case == "identity_app":
        kw["app"] = "tinycap-other"
    elif case == "override_mismatch":
        kw["override_id"] = "paper_defaults"            # this call's document is not the one the run was launched under
    elif case == "identity_no_override":
        write_identity(state, override="")
    elif case == "identity_objects_short":
        write_identity(state, objects={})
    elif case == "identity_objects_extra":
        write_identity(state, objects={DIFF: DIFF_ID, "constv2ctl2-full__lora_r4_ce": "1" * 64})
    elif case == "identity_objects_shape":
        write_identity(state, objects={DIFF: "not-a-sha"})
    if case != "no_report":
        (state / RT.RUN_REPORT_NAME).write_text(run_report(recs))
    with pytest.raises(RT.Refusal, match=re.escape(want)):
        receipt(w, cells=cells, sha=sha, **kw)
    assert not (state / RT.STORE_DIR / RT.RECEIPT_NAME).exists() and not (state / RT.STORE_BUILD_DIR).exists()


def test_a_bfcl_only_resumption_and_a_done_relaunch_are_accepted_as_the_launchers_branches_say(tmp_path):
    """A BFCL-only resumption (BigCodeBench validly done by an earlier attempt; `driven = remaining` = [bfcl_full]; not
    sandboxed, no pointer, no record, its owner still named) is accepted: its call id from the owner, its BigCodeBench attempt from the earlier
    pointer the store root keeps. A DONE relaunch (every task validly done; no owner) is accepted as REUSE: no new execution, the earlier
    attempts' pointers recorded and THEIR call ids named as the execution (`execution` stays reuse) — and EVERY control's completing attempt
    credited through the completion records, the BFCL-only call included. Each negative refuses by name."""
    w = make_world(tmp_path)
    vol, state = w["vol"], w["state"]
    k_diff, f_diff = w["keys"][DIFF], w["fcids"][DIFF]
    # (a) the diff cell resumed BFCL-only under a NEW container fc-diff-0003; the volume keeps attempt_<fc-diff-0002>.json (both controls) and the
    #     two completion records as the launcher left them: fc-diff-0002 finished BigCodeBench alone (INCOMPLETE), fc-diff-0003 BFCL (round 4)
    put_attempt(vol, DIFF, f_diff, key=k_diff, completed={"bigcodebench_full": k_diff}, status="INCOMPLETE")
    put_attempt(vol, DIFF, "fc-diff-0003", key=k_diff, controls=("bfcl_full",), sandboxed=False)
    resumed = rec_ok(DIFF, k_diff, "fc-diff-0003", sandboxed=False, driven=("bfcl_full",))
    assert "sandbox_pointer" not in resumed and "sandbox_record" not in resumed and resumed["completion_record"] == RT.completion_record_path(out_of(DIFF), "fc-diff-0003")
    (state / RT.RUN_REPORT_NAME).write_text(run_report([w["recs"][0], resumed]))
    out = receipt(w)
    rec = json.loads((state / RT.STORE_DIR / RT.RECEIPT_NAME).read_text())
    assert rec["recs"][DIFF]["execution"] == "resumed_bfcl_only" and rec["recs"][DIFF]["sandbox_owner"]["function_call_id"] == "fc-diff-0003"
    assert rec["modal_call_ids"] == sorted([w["fcids"]["none"], "fc-diff-0003", f_diff]) and rec["execution"] == "new"
    assert [a["function_call_id"] for a in rec["bcb_attempts"][DIFF]] == [f_diff] and [a["function_call_id"] for a in rec["prior_attempts"]] == [f_diff]
    assert rec["pointers"][DIFF] is None and (state / RT.STORE_DIR / DIFF / "watch" / f"attempt_{f_diff}.json").is_file()
    # round 4: the resumption's own completion record is pulled and held to the REC; its BigCodeBench is credited to the attempt that completed it
    assert rec["completions"][DIFF]["function_call_id"] == "fc-diff-0003" and rec["completions"][DIFF]["completed"] == {"bfcl_full": k_diff}
    assert rec["completions"][DIFF]["sandboxed"] is False and rec["completions"][DIFF]["controls"] == ["bfcl_full"]
    assert [(a["function_call_id"], a["credited_for"]) for a in rec["credited_completions"]] == [(f_diff, ["bigcodebench_full"])]
    assert [a["function_call_id"] for a in rec["prior_completions"]] == [f_diff] and (state / RT.STORE_DIR / DIFF / "watch" / "completion_fc-diff-0003.json").is_file()
    assert out["verdict_command"] and f"--run-ids {','.join(sorted([w['keys']['none'], k_diff]))}" in out["verdict_command"]
    # a BFCL-only resumption with NO earlier BigCodeBench pointer on the root: the execution is unaccounted for
    del vol.remote[vrel(RT.attempt_pointer_path(out_of(DIFF), f_diff))]
    with pytest.raises(RT.Refusal, match="BigCodeBench execution is unaccounted for|cannot list"):
        receipt(w)
    vol.remote[vrel(RT.attempt_pointer_path(out_of(DIFF), f_diff))] = json.dumps(pointer(DIFF, f_diff)).encode()
    # ... and with the pointer but no completion record naming who finished BigCodeBench in the credited store: unaccounted for too (round 4)
    saved = vol.remote.pop(vrel(RT.completion_record_path(out_of(DIFF), f_diff)))
    with pytest.raises(RT.Refusal, match="the bigcodebench_full execution it credits is unaccounted for") as ei:
        receipt(w)
    assert f"{DIFF}: a BFCL-only resumption, but" in str(ei.value) and "holds no completion record for this cell AT THE TREE" in str(ei.value)
    assert not (state / RT.STORE_BUILD_DIR).exists()
    vol.remote[vrel(RT.completion_record_path(out_of(DIFF), f_diff))] = saved
    # not sandboxed but it DROVE bigcodebench: a missing pointer, never a resumption
    bad = rec_ok(DIFF, k_diff, "fc-diff-0004", sandboxed=False, driven=CONTROLS)
    (state / RT.RUN_REPORT_NAME).write_text(run_report([w["recs"][0], bad]))
    with pytest.raises(RT.Refusal, match="is not sandboxed and carries no sandbox_pointer"):
        receipt(w)
    # (b) both cells DONE: reuse — the earlier pointers are the evidence, and EVERY control's completing attempt is credited: the diff cell's
    #     BigCodeBench to fc-diff-0002, its BFCL to the BFCL-only fc-diff-0003 (round 4: a reuse used to name fc-diff-0002 alone)
    (state / RT.RUN_REPORT_NAME).write_text(run_report([rec_done("none", w["keys"]["none"]), rec_done(DIFF, k_diff)]))
    out = receipt(w)
    rec = json.loads((state / RT.STORE_DIR / RT.RECEIPT_NAME).read_text())
    assert rec["execution"] == "reuse" and all(r["execution"] == "reuse" and r["sandbox_owner"] is None and r["mode"] == "off" for r in rec["recs"].values())
    assert sorted(a["function_call_id"] for a in rec["reused_attempts"]) == sorted(w["fcids"].values()) and rec["headline"]["value"] == 0.455
    assert rec["modal_call_ids"] == sorted([*w["fcids"].values(), "fc-diff-0003"])
    assert sorted((a["cell_of_receipt"], a["function_call_id"], tuple(a["credited_for"])) for a in rec["credited_completions"]) == sorted([
        ("none", w["fcids"]["none"], ("bigcodebench_full", "bfcl_full")), (DIFF, f_diff, ("bigcodebench_full",)), (DIFF, "fc-diff-0003", ("bfcl_full",))])
    assert all(r["completion_record"] is None for r in rec["recs"].values()) and rec["completions"] == {"none": None, DIFF: None}
    assert out["verdict_command"]
    # a DONE relaunch with no pointer at all on the root: the execution it reuses is unaccounted for
    del vol.remote[vrel(RT.attempt_pointer_path(out_of("none"), w["fcids"]["none"]))]
    with pytest.raises(RT.Refusal, match="reuses is unaccounted for|cannot list"):
        receipt(w)
    vol.remote[vrel(RT.attempt_pointer_path(out_of("none"), w["fcids"]["none"]))] = json.dumps(pointer("none", w["fcids"]["none"])).encode()
    # the BFCL-only call's completion record gone — the relaunch's BFCL store has no producer on the volume: refused by
    # name, never filed with the BigCodeBench call alone
    saved = vol.remote.pop(vrel(RT.completion_record_path(out_of(DIFF), "fc-diff-0003")))
    with pytest.raises(RT.Refusal, match="the bfcl_full execution it reuses is unaccounted for") as ei:
        receipt(w)
    msg = str(ei.value)
    assert f"{DIFF}: a DONE relaunch (reuse), but" in msg and f"completed bfcl_full in the store the REC credits ({k_diff})" in msg
    assert f"completion_{f_diff}.json" in msg and "does not name bfcl_full" in msg and not (state / RT.STORE_BUILD_DIR).exists()
    vol.remote[vrel(RT.completion_record_path(out_of(DIFF), "fc-diff-0003"))] = saved
    # a mixed run (one cell OK, one DONE) names the OK cell's call id AND the reused cell's credited attempts — both of them
    (state / RT.RUN_REPORT_NAME).write_text(run_report([w["recs"][0], rec_done(DIFF, k_diff)]))
    receipt(w)
    rec = json.loads((state / RT.STORE_DIR / RT.RECEIPT_NAME).read_text())
    assert rec["modal_call_ids"] == sorted([w["fcids"]["none"], f_diff, "fc-diff-0003"]) and rec["execution"] == "new" and rec["recs"][DIFF]["execution"] == "reuse"


def test_a_recovered_run_is_credited_only_with_pointers_at_the_tree_and_still_reaches_its_filed_analysis(tmp_path):
    """A BFCL-only resumption and a DONE reuse credit an EARLIER attempt's pointer — which must be
    this cell's, under this out dir, AT THE TREE the REC was held to, naming a call id and a run id, the file named for its call id, the record
    in the launcher's layout, and (BFCL-only) BigCodeBench among its controls. A persistent watch/ holding only an older-tree pointer used to
    satisfy the receipt and lend its call id; now it refuses by name, and beside a right-tree pointer it is history (`other_tree_attempts`),
    never a call id. A recovered run then reaches its receipt AND its filed analysis."""
    w = make_world(tmp_path / "w")
    vol, state = w["vol"], w["state"]
    k_diff, f_diff = w["keys"][DIFF], w["fcids"][DIFF]
    old_tree = "2" * 40
    resumed = rec_ok(DIFF, k_diff, "fc-diff-0003", sandboxed=False, driven=("bfcl_full",))
    put_attempt(vol, DIFF, "fc-diff-0003", key=k_diff, controls=("bfcl_full",), sandboxed=False)          # the resumption's own completion record (round 4)
    (state / RT.RUN_REPORT_NAME).write_text(run_report([w["recs"][0], resumed]))
    # (1) the only earlier pointer for the cell is at ANOTHER tree (its completion record with it): refused by name, naming the pointer and why; no receipt
    put_attempt(vol, DIFF, f_diff, key=k_diff, tree_sha=old_tree)
    with pytest.raises(RT.Refusal, match=re.escape(f"AT THE TREE {SHA[:12]}")) as ei:
        receipt(w)
    assert "BigCodeBench execution is unaccounted for" in str(ei.value) and f"attempt_{f_diff}.json" in str(ei.value) and f"tree_sha '{old_tree[:12]}' is not the tree" in str(ei.value)
    assert not (state / RT.STORE_DIR / RT.RECEIPT_NAME).exists()
    # a pointer at the tree whose identity is not the launcher's is not evidence either: a record outside the layout, a BFCL-only earlier attempt
    bad_record = pointer(DIFF, "fc-diff-0004"); bad_record["sandbox_record"] = "/vol/elsewhere/sandboxes_fc-diff-0004.jsonl"
    vol.remote[vrel(RT.attempt_pointer_path(out_of(DIFF), "fc-diff-0004"))] = json.dumps(bad_record).encode()
    vol.remote[vrel(RT.attempt_pointer_path(out_of(DIFF), "fc-diff-0006"))] = json.dumps(pointer(DIFF, "fc-diff-0006", controls=("bfcl_full",))).encode()
    with pytest.raises(RT.Refusal, match="BigCodeBench execution is unaccounted for") as ei:
        receipt(w)
    assert "is not the launcher's layout" in str(ei.value) and "did not drive bigcodebench_full" in str(ei.value)
    # (2) a right-tree BigCodeBench pointer beside them (fc-diff-0005 drove both and finished BigCodeBench; fc-diff-0003 finished BFCL): accepted;
    #     the older-tree one is history, its call id NOT the run's
    put_attempt(vol, DIFF, "fc-diff-0005", key=k_diff, completed={"bigcodebench_full": k_diff}, status="INCOMPLETE")
    out = receipt(w)
    rec = json.loads((state / RT.STORE_DIR / RT.RECEIPT_NAME).read_text())
    assert [a["function_call_id"] for a in rec["bcb_attempts"][DIFF]] == ["fc-diff-0005"]
    assert rec["modal_call_ids"] == sorted([w["fcids"]["none"], "fc-diff-0003", "fc-diff-0005"]) and f_diff not in rec["modal_call_ids"]
    assert [a["function_call_id"] for a in rec["other_tree_attempts"]] == [f_diff] and rec["other_tree_attempts"][0]["problem"].startswith("tree_sha")
    assert sorted(a["function_call_id"] for a in rec["prior_attempts"]) == sorted([f_diff, "fc-diff-0004", "fc-diff-0005", "fc-diff-0006"])     # history, all of it
    assert [(a["function_call_id"], a["credited_for"]) for a in rec["credited_completions"]] == [("fc-diff-0005", ["bigcodebench_full"])]       # round 4: never the old-tree record
    assert sorted(a["function_call_id"] for a in rec["prior_completions"]) == sorted([f_diff, "fc-diff-0005"])
    assert rec["recs"][DIFF]["model"] == KEY
    accepted = store_bytes(state)
    # (3) a DONE reuse: the same rule — only wrong-tree pointers for the none cell refuse; a right-tree one is the reused attempt
    #     (the refusal leaves receipt (2) standing, byte-identical — a receipt is never deleted before its replacement exists)
    (state / RT.RUN_REPORT_NAME).write_text(run_report([rec_done("none", w["keys"]["none"]), rec_done(DIFF, k_diff)]))
    put_attempt(vol, "none", w["fcids"]["none"], key=w["keys"]["none"], tree_sha=old_tree)
    with pytest.raises(RT.Refusal, match="execution it reuses is unaccounted for") as ei:
        receipt(w)
    assert f"AT THE TREE {SHA[:12]}" in str(ei.value) and store_bytes(state) == accepted
    put_attempt(vol, "none", "fc-none-0008", key=w["keys"]["none"])
    receipt(w)
    rec = json.loads((state / RT.STORE_DIR / RT.RECEIPT_NAME).read_text())
    reused = sorted(a["function_call_id"] for a in rec["reused_attempts"])
    # the pointer-credited attempts (round 3) PLUS the BFCL-only fc-diff-0003 that completed the diff cell's BFCL (round 4) ARE the execution named
    assert rec["execution"] == "reuse" and reused == ["fc-diff-0005", "fc-none-0008"] and rec["modal_call_ids"] == ["fc-diff-0003", "fc-diff-0005", "fc-none-0008"]
    assert sorted((a["cell_of_receipt"], a["function_call_id"], tuple(a["credited_for"])) for a in rec["credited_completions"]) == sorted([
        ("none", "fc-none-0008", ("bigcodebench_full", "bfcl_full")), (DIFF, "fc-diff-0005", ("bigcodebench_full",)), (DIFF, "fc-diff-0003", ("bfcl_full",))])
    for gone in (w["fcids"]["none"], f_diff, "fc-diff-0004", "fc-diff-0006"):        # 0006 drove BFCL only: no evidence for a reuse either
        assert gone not in reused and gone not in rec["modal_call_ids"]
    assert sorted(a["function_call_id"] for a in rec["other_tree_attempts"]) == sorted([w["fcids"]["none"], f_diff])
    # (4) the recovered run (the BFCL-only shape) reaches its filed analysis: receipt -> verdict -> file -> fold -> read-back PASS
    #     (the none cell's own first-attempt records back at the tree: a sandboxed REC's pointer and completion record are held to it, as before)
    put_attempt(vol, "none", w["fcids"]["none"], key=w["keys"]["none"])
    (state / RT.RUN_REPORT_NAME).write_text(run_report([w["recs"][0], resumed]))
    out = receipt(w)
    assert rule(w) == 0
    root = _registry_with_objects(tmp_path / "reg")                       # its own registry: the module's is read back by position by the identity test
    rc, lines, why = file_phase(w, root)
    assert rc == 0 and why is None and any("filed, not yet folded" in ln for ln in lines), (why, lines)
    spec = json.loads((state / SF.SPEC_NAME).read_text())
    assert spec["extra"]["modal_call_ids"] == sorted([w["fcids"]["none"], "fc-diff-0003", "fc-diff-0005"]) and spec["extra"]["modal_other_tree_attempts"] == [f_diff]
    assert spec["extra"]["modal_credited_completions"] == ["fc-diff-0005"]
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0
    rc, lines, why = file_phase(w, root)
    assert rc == 0 and any(f"PASS: {spec['name']}" in ln for ln in lines), lines


def test_a_done_reuse_is_credited_only_with_a_pointer_that_drove_the_full_bigcodebench_in_the_recs_mode_and_names_its_call_ids(tmp_path):
    """A fully DONE relaunch took ANY same-tree pointer of the cell under its out dir as the execution it
    reused — the launcher writes a pointer per attempt into the same watch/ for every suite and mode it drives, so another suite's attempt's
    pointer (here the reduced `bigcodebench_200` set's) satisfied the receipt when the full-set attempt's was missing, and the receipt then filed
    `modal_call_ids: []`. Now a reuse is held to the rule the BFCL-only path already had — the REC's mode and `bigcodebench_full` among the
    controls driven: a watch/ holding only a same-tree other-suite pointer (or an other-mode one) refuses by name with nothing written, and the
    accepted receipt names the credited attempts' call ids in `modal_call_ids` with `execution: reuse` kept; the file phase carries them."""
    w = make_world(tmp_path / "w")
    vol, state = w["vol"], w["state"]
    k_none, f_none, f_diff = w["keys"]["none"], w["fcids"]["none"], w["fcids"][DIFF]
    (state / RT.RUN_REPORT_NAME).write_text(run_report([rec_done("none", k_none), rec_done(DIFF, w["keys"][DIFF])]))
    # the none cell's full-set attempt's records are gone; its watch/ holds ONLY a same-tree pointer of a reduced-set (bigcodebench_200) attempt — right
    # cell, out dir, tree, layout, model and mode, wrong suite (its completion record names that suite's own store, never this one's: round 4)
    del vol.remote[vrel(RT.attempt_pointer_path(out_of("none"), f_none))]
    del vol.remote[vrel(RT.completion_record_path(out_of("none"), f_none))]
    full_p = RT.attempt_pointer_path(out_of("none"), "fc-none-0021")
    full = pointer("none", "fc-none-0021", controls=("bigcodebench_200", "bfcl_200"))
    vol.remote[vrel(full_p)] = json.dumps(full).encode()
    vol.remote[vrel(RT.completion_record_path(out_of("none"), "fc-none-0021"))] = json.dumps(
        completion("none", "fc-none-0021", key=key_of("none", "ff"), controls=("bigcodebench_200", "bfcl_200"))).encode()
    assert RT.pointer_identity_problem(dict(full, path=full_p), tok="none", out=out_of("none"), sha=SHA, model=KEY, mode="off") \
        == "did not drive bigcodebench_full (controls ['bigcodebench_200', 'bfcl_200'])"
    with pytest.raises(RT.Refusal, match="execution it reuses is unaccounted for") as ei:
        receipt(w)
    msg = str(ei.value)
    assert f"AT THE TREE {SHA[:12]}" in msg and "in mode 'off'" in msg and "attempt_fc-none-0021.json" in msg and "did not drive bigcodebench_full" in msg
    assert not (state / RT.STORE_DIR).exists() and not (state / RT.STORE_BUILD_DIR).exists()
    # a same-tree full-set pointer in ANOTHER mode is not evidence either (nor its other-mode completion record)
    other_mode = pointer("none", "fc-none-0023"); other_mode["mode"] = "on"
    vol.remote[vrel(RT.attempt_pointer_path(out_of("none"), "fc-none-0023"))] = json.dumps(other_mode).encode()
    other_mode_c = completion("none", "fc-none-0023", key=k_none); other_mode_c["mode"] = "on"
    vol.remote[vrel(RT.completion_record_path(out_of("none"), "fc-none-0023"))] = json.dumps(other_mode_c).encode()
    with pytest.raises(RT.Refusal, match="execution it reuses is unaccounted for") as ei:
        receipt(w)
    assert "mode 'on' is not the REC's 'off'" in str(ei.value) and not (state / RT.STORE_DIR).exists()
    # the full-set attempt's pointer and completion record beside them: accepted — the reuse names ITS call id (and the diff cell's), never the
    # other-suite or other-mode one
    put_attempt(vol, "none", "fc-none-0022", key=k_none)
    out = receipt(w)
    rec = json.loads((state / RT.STORE_DIR / RT.RECEIPT_NAME).read_text())
    assert rec["execution"] == "reuse" and rec["modal_call_ids"] == sorted(["fc-none-0022", f_diff]) and out["verdict_command"]
    assert all(r["execution"] == "reuse" and r["sandbox_owner"] is None and r["mode"] == "off" for r in rec["recs"].values())
    assert sorted(a["function_call_id"] for a in rec["reused_attempts"]) == sorted(["fc-none-0022", f_diff])
    assert sorted(a["function_call_id"] for a in rec["prior_attempts"]) == sorted(["fc-none-0021", "fc-none-0022", "fc-none-0023", f_diff])   # history, all of it
    assert rec["other_tree_attempts"] == [] and "fc-none-0021" not in rec["modal_call_ids"] and "fc-none-0023" not in rec["modal_call_ids"]
    assert sorted((a["cell_of_receipt"], a["function_call_id"], tuple(a["credited_for"])) for a in rec["credited_completions"]) == sorted([
        ("none", "fc-none-0022", ("bigcodebench_full", "bfcl_full")), (DIFF, f_diff, ("bigcodebench_full", "bfcl_full"))])
    assert sorted(a["function_call_id"] for a in rec["prior_completions"]) == sorted(["fc-none-0021", "fc-none-0022", "fc-none-0023", f_diff])   # history, all of it
    assert any(f"call ids ['{f_diff}', 'fc-none-0022']; execution reuse" in ln for ln in out["log"]), out["log"]
    # the file phase composes the named attempts from the receipt alone
    assert rule(w) == 0
    rc, lines, why = file_phase(w, str(_fresh_registry(tmp_path / "reg")), dry=True)
    assert rc == 0 and why is None, (why, lines)
    spec = json.loads((state / SF.SPEC_NAME).read_text())
    assert spec["extra"]["modal_call_ids"] == sorted(["fc-none-0022", f_diff]) and spec["extra"]["execution"] == "reuse"
    assert sorted(spec["extra"]["modal_reused_attempts"]) == sorted(["fc-none-0022", f_diff]) and spec["extra"]["modal_credited_completions"] == sorted(["fc-none-0022", f_diff])


def test_a_done_reuse_after_a_resume_then_complete_sequence_credits_the_bfcl_only_call_through_its_completion_record(tmp_path):
    """A first call completes BigCodeBench and dies on BFCL; a second call resumes BFCL-only and completes it
    — the launcher gave it an owner but, nothing being sandboxed, wrote no attempt pointer; a third launch returns DONE for both. The earlier
    receipt filled `modal_call_ids` from the BigCodeBench pointers alone and silently omitted the call that produced BFCL. Now every solo attempt
    leaves `watch/completion_<fcid>.json` (what it finished, in which store) and the receipt attributes EACH reused control to the attempt that
    completed it in the credited store: the second launch's receipt pulls and verifies the resumption's own record, the DONE relaunch credits
    both calls, and a relaunch whose BFCL producer left no record — or whose record names another store, tree or mode — refuses by name with
    nothing written."""
    w = make_world(tmp_path / "w")
    vol, state = w["vol"], w["state"]
    k_none, f1, f_diff = w["keys"]["none"], w["fcids"]["none"], w["fcids"][DIFF]
    f2 = "fc-none-0031"
    # launch 1: fc-none-0001 drove both, finished BigCodeBench alone (its REC was INCOMPLETE; its pointer and completion record stay on the volume)
    put_attempt(vol, "none", f1, key=k_none, completed={"bigcodebench_full": k_none}, status="INCOMPLETE")
    # launch 2: fc-none-0031 resumed BFCL-only — no pointer; its completion record names what it finished and where
    put_attempt(vol, "none", f2, key=k_none, controls=("bfcl_full",), sandboxed=False)
    resumed = rec_ok("none", k_none, f2, sandboxed=False, driven=("bfcl_full",))
    (state / RT.RUN_REPORT_NAME).write_text(run_report([resumed, w["recs"][1]]))
    receipt(w)
    rec = json.loads((state / RT.STORE_DIR / RT.RECEIPT_NAME).read_text())
    assert rec["recs"]["none"]["execution"] == "resumed_bfcl_only" and rec["recs"]["none"]["completion_record"] == RT.completion_record_path(out_of("none"), f2)
    assert rec["completions"]["none"]["completed"] == {"bfcl_full": k_none} and rec["completions"]["none"]["remaining"] == ["bfcl_full"] and rec["completions"]["none"]["sandboxed"] is False
    assert rec["modal_call_ids"] == sorted([f1, f2, f_diff])
    assert [(a["function_call_id"], a["credited_for"]) for a in rec["credited_completions"]] == [(f1, ["bigcodebench_full"])]
    assert (state / RT.STORE_DIR / "none" / "watch" / f"completion_{f2}.json").is_file() and (state / RT.STORE_DIR / "none" / "watch" / f"completion_{f1}.json").is_file()
    # the resumption's own record must agree with its REC: another call id, status or sandboxing, or a `completed` naming another store — each refuses
    for field, value, why in (("function_call_id", "fc-other", "field function_call_id = 'fc-other'"), ("status", "INCOMPLETE", "field status = 'INCOMPLETE'"),
                              ("sandboxed", True, "field sandboxed = True"), ("completed", {"bfcl_full": key_of("none", "zz")}, "the REC credits store")):
        doc = completion("none", f2, key=k_none, controls=("bfcl_full",), sandboxed=False); doc[field] = value
        vol.remote[vrel(RT.completion_record_path(out_of("none"), f2))] = json.dumps(doc).encode()
        with pytest.raises(RT.Refusal, match=re.escape(why)):
            receipt(w)
        assert not (state / RT.STORE_BUILD_DIR).exists()
    put_attempt(vol, "none", f2, key=k_none, controls=("bfcl_full",), sandboxed=False)
    # launch 3: DONE for both — the execution it reuses is BOTH calls
    (state / RT.RUN_REPORT_NAME).write_text(run_report([rec_done("none", k_none), w["recs"][1]]))
    out = receipt(w)
    rec = json.loads((state / RT.STORE_DIR / RT.RECEIPT_NAME).read_text())
    assert rec["recs"]["none"]["execution"] == "reuse" and rec["recs"]["none"]["completion_record"] is None and rec["completions"]["none"] is None
    assert [a["function_call_id"] for a in rec["reused_attempts"]] == [f1]                            # the pointer rule (round 3): the sandboxed BigCodeBench attempt
    assert sorted((a["function_call_id"], tuple(a["credited_for"])) for a in rec["credited_completions"] if a["cell_of_receipt"] == "none") \
        == sorted([(f1, ("bigcodebench_full",)), (f2, ("bfcl_full",))])
    assert rec["modal_call_ids"] == sorted([f1, f2, f_diff]) and rec["execution"] == "new"            # the BFCL-only call is named; the diff cell ran
    assert sorted(a["function_call_id"] for a in rec["prior_completions"] if a["cell_of_receipt"] == "none") == sorted([f1, f2])
    assert any(f"call ids {sorted([f1, f2, f_diff])!r}" in ln for ln in out["log"]), out["log"]
    accepted = store_bytes(state)
    # the BFCL producer's record gone: the relaunch's BFCL store has no producer on the volume — refused by name, the accepted receipt standing
    saved = vol.remote.pop(vrel(RT.completion_record_path(out_of("none"), f2)))
    with pytest.raises(RT.Refusal, match="the bfcl_full execution it reuses is unaccounted for") as ei:
        receipt(w)
    msg = str(ei.value)
    assert "none: a DONE relaunch (reuse), but" in msg and f"completed bfcl_full in the store the REC credits ({k_none})" in msg
    assert f"completion_{f1}.json" in msg and "does not name bfcl_full" in msg and store_bytes(state) == accepted and not (state / RT.STORE_BUILD_DIR).exists()
    # ... or naming ANOTHER store, another tree or another mode: history, never this store's producer
    for tweak, said in (({"completed": {"bfcl_full": key_of("none", "zz")}}, f"does not name bfcl_full in store '{k_none}'"), ({"tree_sha": "3" * 40}, "is not the tree"),
                        ({"mode": "on"}, "mode 'on' is not the REC's 'off'")):
        doc = completion("none", f2, key=k_none, controls=("bfcl_full",), sandboxed=False); doc.update(tweak)
        vol.remote[vrel(RT.completion_record_path(out_of("none"), f2))] = json.dumps(doc).encode()
        with pytest.raises(RT.Refusal, match="the bfcl_full execution it reuses is unaccounted for") as ei:
            receipt(w)
        assert f"completion_{f2}.json" in str(ei.value) and said in str(ei.value) and store_bytes(state) == accepted
    vol.remote[vrel(RT.completion_record_path(out_of("none"), f2))] = saved
    # the file phase carries both calls
    receipt(w)
    assert store_bytes(state) == accepted and rule(w) == 0
    rc, lines, why = file_phase(w, str(_fresh_registry(tmp_path / "reg")), dry=True)
    assert rc == 0 and why is None, (why, lines)
    spec = json.loads((state / SF.SPEC_NAME).read_text())
    assert spec["extra"]["modal_call_ids"] == sorted([f1, f2, f_diff]) and spec["extra"]["modal_reused_attempts"] == [f1]
    assert spec["extra"]["modal_credited_completions"] == sorted([f1, f2]) and spec["extra"]["execution"] == "new"


def test_a_pointer_whose_attempt_completed_nothing_is_history_never_the_execution_a_recovered_cell_credits(tmp_path):
    """The launcher publishes `watch/attempt_<fcid>.json` BEFORE the standup, so an attempt whose standup failed
    leaves a pointer that passes every identity rule (this cell, this out dir, the tree, the model, the mode, BigCodeBench among the controls
    it was GOING to drive) and no completion record. The earlier receipt credited every such pointer as a reused (or BigCodeBench) attempt and
    put its call id in `modal_call_ids` beside the real producer's. Now the credited pointers are joined by call id to the credited completion
    records: the failed attempt is `uncredited_attempts` (with why) and `prior_attempts`, never a call id or a credited attempt — for a DONE
    reuse and for a BFCL-only resumption alike; a producer whose own pointer is gone refuses by name even with the failed attempt's pointer
    standing; and the file phase carries the distinction."""
    w = make_world(tmp_path / "w")
    vol, state = w["vol"], w["state"]
    k_none, f_ok, k_diff, f_diff = w["keys"]["none"], w["fcids"]["none"], w["keys"][DIFF], w["fcids"][DIFF]
    out_none, out_diff = out_of("none"), out_of(DIFF)
    # launch 1 on the none cell: fc-none-0040, sandboxed, both controls — its pointer published, its standup failed: no drive, no completion record
    f_dead = "fc-none-0040"
    dead_p = RT.attempt_pointer_path(out_none, f_dead)
    vol.remote[vrel(dead_p)] = json.dumps(pointer("none", f_dead)).encode()
    assert RT.pointer_identity_problem(dict(pointer("none", f_dead), path=dead_p), tok="none", out=out_none, sha=SHA, model=KEY, mode="off") is None
    # launch 2: fc-none-0001 drove both and completed both in k_none (make_world's pointer + completion record); launch 3: DONE — the diff cell ran OK
    (state / RT.RUN_REPORT_NAME).write_text(run_report([rec_done("none", k_none), w["recs"][1]]))
    out = receipt(w)
    rec = json.loads((state / RT.STORE_DIR / RT.RECEIPT_NAME).read_text())
    assert rec["schema"] == RT.RECEIPT_SCHEMA == "stage07-receipt/7" and rec["recs"]["none"]["execution"] == "reuse"
    assert [(a["function_call_id"], a["credited_for"]) for a in rec["reused_attempts"]] == [(f_ok, list(CONTROLS))]
    assert rec["modal_call_ids"] == sorted([f_ok, f_diff]) and f_dead not in rec["modal_call_ids"]
    assert [(a["function_call_id"], a["cell_of_receipt"], a["path"]) for a in rec["uncredited_attempts"]] == [(f_dead, "none", dead_p)]
    assert f"no completion record for this cell at the tree credits call '{f_dead}'" in rec["uncredited_attempts"][0]["problem"]
    assert sorted(a["function_call_id"] for a in rec["prior_attempts"]) == sorted([f_dead, f_ok]) and rec["other_tree_attempts"] == []      # history keeps it
    assert [(a["function_call_id"], a["credited_for"]) for a in rec["credited_completions"]] == [(f_ok, list(CONTROLS))]
    assert (state / RT.STORE_DIR / "none" / "watch" / f"attempt_{f_dead}.json").is_file()                 # pulled as history, like every pointer the root holds
    assert any(f"call ids {sorted([f_diff, f_ok])!r}" in ln for ln in out["log"]), out["log"]
    accepted = store_bytes(state)
    # the producer's own pointer gone: the failed attempt's is the only qualifying pointer — it passes identity and is NOT the execution: refused by
    # name, naming the producer, the pointer it lacks and why the standing one is no evidence; the accepted receipt stands
    saved = vol.remote.pop(vrel(RT.attempt_pointer_path(out_none, f_ok)))
    with pytest.raises(RT.Refusal, match="the execution it reuses is unaccounted for") as ei:
        receipt(w)
    msg = str(ei.value)
    assert f"none: a DONE relaunch (reuse), but the attempt '{f_ok}' whose completion record" in msg and "credits ['bigcodebench_full', 'bfcl_full']" in msg
    assert f"has no attempt pointer at {RT.attempt_pointer_path(out_none, f_ok)}" in msg and f"attempt_{f_dead}.json" in msg
    assert "no completion record for this cell at the tree credits call" in msg and store_bytes(state) == accepted and not (state / RT.STORE_BUILD_DIR).exists()
    vol.remote[vrel(RT.attempt_pointer_path(out_none, f_ok))] = saved
    # the BFCL-only shape, on the diff cell: fc-diff-0040 failed its standup (pointer only); fc-diff-0005 drove both and finished BigCodeBench alone;
    # fc-diff-0003 resumed BFCL-only and finished it — the BigCodeBench attempt credited is fc-diff-0005, the failed one history
    del vol.remote[vrel(RT.attempt_pointer_path(out_diff, f_diff))]
    del vol.remote[vrel(RT.completion_record_path(out_diff, f_diff))]
    vol.remote[vrel(RT.attempt_pointer_path(out_diff, "fc-diff-0040"))] = json.dumps(pointer(DIFF, "fc-diff-0040")).encode()
    put_attempt(vol, DIFF, "fc-diff-0005", key=k_diff, completed={"bigcodebench_full": k_diff}, status="INCOMPLETE")
    put_attempt(vol, DIFF, "fc-diff-0003", key=k_diff, controls=("bfcl_full",), sandboxed=False)
    resumed = rec_ok(DIFF, k_diff, "fc-diff-0003", sandboxed=False, driven=("bfcl_full",))
    (state / RT.RUN_REPORT_NAME).write_text(run_report([rec_done("none", k_none), resumed]))
    out = receipt(w)
    rec = json.loads((state / RT.STORE_DIR / RT.RECEIPT_NAME).read_text())
    assert [(a["function_call_id"], a["credited_for"]) for a in rec["bcb_attempts"][DIFF]] == [("fc-diff-0005", ["bigcodebench_full"])]
    assert rec["modal_call_ids"] == sorted([f_ok, "fc-diff-0003", "fc-diff-0005"]) and "fc-diff-0040" not in rec["modal_call_ids"]
    assert sorted((a["cell_of_receipt"], a["function_call_id"]) for a in rec["uncredited_attempts"]) == sorted([("none", f_dead), (DIFF, "fc-diff-0040")])
    assert sorted(a["function_call_id"] for a in rec["prior_attempts"] if a["cell_of_receipt"] == DIFF) == sorted(["fc-diff-0005", "fc-diff-0040"])
    assert sorted((a["cell_of_receipt"], a["function_call_id"], tuple(a["credited_for"])) for a in rec["credited_completions"]) == sorted([
        ("none", f_ok, tuple(CONTROLS)), (DIFF, "fc-diff-0005", ("bigcodebench_full",))])
    # the BigCodeBench producer's pointer gone, the failed attempt's standing: refused by name, in the resumption's words
    saved = vol.remote.pop(vrel(RT.attempt_pointer_path(out_diff, "fc-diff-0005")))
    with pytest.raises(RT.Refusal, match="the BigCodeBench execution is unaccounted for") as ei:
        receipt(w)
    assert f"{DIFF}: a BFCL-only resumption, but the attempt 'fc-diff-0005' whose completion record" in str(ei.value) and "attempt_fc-diff-0040.json" in str(ei.value)
    assert not (state / RT.STORE_BUILD_DIR).exists()
    vol.remote[vrel(RT.attempt_pointer_path(out_diff, "fc-diff-0005"))] = saved
    # the file phase: the failed attempts are history (prior, uncredited), never a call id, a reused attempt or a credited completion
    receipt(w)
    assert rule(w) == 0
    rc, lines, why = file_phase(w, str(_fresh_registry(tmp_path / "reg")), dry=True)
    assert rc == 0 and why is None, (why, lines)
    spec = json.loads((state / SF.SPEC_NAME).read_text())
    assert spec["extra"]["modal_call_ids"] == sorted([f_ok, "fc-diff-0003", "fc-diff-0005"]) and spec["extra"]["modal_reused_attempts"] == [f_ok]
    # the credited completions are the PRIOR records a recovered cell leans on (round 4); the resumption's own call, fc-diff-0003, is named through
    # its REC's owner (and its own record under `completions`), as the round-4 test pins for the same shape
    assert spec["extra"]["modal_credited_completions"] == sorted([f_ok, "fc-diff-0005"])
    assert sorted(spec["extra"]["modal_uncredited_attempts"]) == sorted([f_dead, "fc-diff-0040"])
    assert sorted(spec["extra"]["modal_prior_attempts"]) == sorted([f_dead, f_ok, "fc-diff-0005", "fc-diff-0040"])
    assert f_dead not in spec["extra"]["modal_call_ids"] and "fc-diff-0040" not in spec["extra"]["modal_call_ids"]


class _RewritingVolume:
    """A FakeModal whose FIRST `volume get` also appends a line to the LIVE run report — a `--phase run` landing during the receipt's pulls."""

    def __init__(self, inner: FakeModal, live: Path, extra: str):
        self.inner, self.live, self.extra, self.fired = inner, live, extra, False
        self.calls = inner.calls

    def __call__(self, argv, **kw):
        r = self.inner(argv, **kw)
        if not self.fired and list(argv)[1:3] == ["volume", "get"]:
            self.fired = True
            self.live.write_text(self.live.read_text() + self.extra)
        return r


def test_the_receipt_reads_the_run_report_once_and_refuses_a_log_rewritten_during_the_pulls(tmp_path):
    """The live log is read ONCE as a snapshot — parsed, copied and hashed as the same bytes. A log rewritten during the
    pulls (another `--phase run`, a fast DONE-reuse relaunch) is caught before the receipt is written: refused by name, no receipt — never a
    receipt whose RECs are one execution's and whose copy hashes another's. The appended line is a REC the snapshot never saw (a third cell), so
    a re-read would have refused for another reason; the refusal is the snapshot check's."""
    w = make_world(tmp_path)
    state, live = w["state"], w["state"] / RT.RUN_REPORT_NAME
    before = live.read_bytes()
    third = json.dumps(rec_ok("constv2ctl2-full__lora_r4_ce", key_of("constv2ctl2-full__lora_r4_ce", "c3"), "fc-ctl-0007")) + "\n"
    rewriting = _RewritingVolume(w["vol"], live, third)
    with pytest.raises(RT.Refusal, match="changed while the receipt was being built") as ei:
        receipt(w, runner=rewriting)
    assert rewriting.fired and "run --phase receipt again" in str(ei.value) and "outside --cells" not in str(ei.value)
    assert not (state / RT.STORE_DIR / RT.RECEIPT_NAME).exists() and live.read_bytes() == before + third.encode()
    # the log as it now stands: the third cell's REC is outside --cells, refused for THAT reason (the snapshot was parsed, not re-read mid-way)
    with pytest.raises(RT.Refusal, match="carries RECs for cells outside --cells"):
        receipt(w)
    live.write_bytes(before)
    out = receipt(w)
    store = state / RT.STORE_DIR
    rec = json.loads((store / RT.RECEIPT_NAME).read_text())
    assert (store / RT.RUN_REPORT_COPY).read_bytes() == before and rec["run_report_sha256"] == hashlib.sha256(before).hexdigest() == RT.sha256_bytes(before)
    assert out["receipt_sha256"]


def test_every_pull_is_scoped_to_the_configured_modal_environment_and_the_receipt_records_it(tmp_path, registry):
    """The launcher ran under `modal run -e $SISTEER_MODAL_ENVIRONMENT`; a volume name is per environment, so the receipt's
    every get/ls carries `--env <that environment>` (right after the verb) when one is configured, none when not; the receipt and the filed
    extra record it."""
    w = make_world(tmp_path)
    write_identity(w["state"], environment="demo-env")                     # the launch ran under demo-env; a receipt under another environment refuses
    receipt(w, environment="demo-env")
    calls = [c for c in w["vol"].calls if c[1] == "volume"]
    assert calls and all(c[3:5] == ["--env", "demo-env"] for c in calls), calls[:3]
    rec = json.loads((w["state"] / RT.STORE_DIR / RT.RECEIPT_NAME).read_text())
    assert rec["modal_environment"] == "demo-env"
    assert rule(w) == 0
    rc, lines, why = file_phase(w, registry, dry=True)
    assert rc == 0 and json.loads((w["state"] / SF.SPEC_NAME).read_text())["extra"]["modal_environment"] == "demo-env"
    w["vol"].calls.clear()
    with pytest.raises(RT.Refusal, match="--environment none is not the run's: the run ran under Modal environment demo-env"):
        receipt(w)
    assert not w["vol"].calls                                                # refused before any pull
    write_identity(w["state"])
    receipt(w)
    assert w["vol"].calls and all("--env" not in c for c in w["vol"].calls)
    assert json.loads((w["state"] / RT.STORE_DIR / RT.RECEIPT_NAME).read_text())["modal_environment"] is None


def test_the_file_phase_derives_the_execution_from_the_receipt_and_refuses_a_call_that_disagrees(tmp_path, registry):
    """The filing's cells, launcher tokens, OVERRIDE_ID, app and code_ref are the RECEIPT's. With no cross-check arguments
    the spec is composed from the receipt alone; `07 --phase file` without the run's --cells (every cell of the view, 07's default) against a
    two-cell receipt is refused by name — naming the receipt's cells — as are other tokens and another overrides document; the spec is not
    rewritten by a refused call. An older receipt (no identity) and a receipt whose two cell lists do not map onto each other through the table
    refuse too."""
    w = make_world(tmp_path)
    state = w["state"]
    receipt(w)
    assert rule(w) == 0
    rc, lines, why = file_phase(w, registry, dry=True, cells=None, launcher_cells=None, override_id=None)
    assert rc == 0 and why is None, why
    spec = json.loads((state / SF.SPEC_NAME).read_text())
    spec_bytes = (state / SF.SPEC_NAME).read_bytes()
    assert spec["inputs"]["objects"] == sorted([DIFF_ID, "none"]) and spec["extra"]["cells"] == ["none", "difference_lora"] and spec["extra"]["launcher_cells"] == list(CELLS)
    assert spec["extra"]["run_overrides_id"] == OVERRIDE_ID and spec["name"].startswith(f"demo_import_07_bcb_bfcl_{VIEW}_{OVERRIDE_ID}_x")
    assert spec["code_sha"] == SHA and spec["code_ref"] == f"demo-import-smoke @ {SHA[:7]}" and spec["extra"]["modal_app_name"] == APP
    # every cell of the view (07's default) against the two-cell receipt: refused, naming the receipt's cells; nothing rewritten
    all_cells = [c for c, _t in MT.launcher_cells(KEY, VIEW)]
    all_tokens = [t for _c, t in MT.launcher_cells(KEY, VIEW)]
    assert len(all_cells) == 10 and set(CELLS) < set(all_tokens)
    rc, lines, why = file_phase(w, registry, dry=True, cells=all_cells, launcher_cells=all_tokens)
    assert rc == 2 and why.startswith("--cells ") and "is not the receipt's" in why and "ran with cells none,difference_lora" in why and "pass --cells none,difference_lora" in why, why
    rc, lines, why = file_phase(w, registry, dry=True, launcher_cells=("none",))
    assert rc == 2 and why.startswith("--launcher-cells none is not the receipt's") and ",".join(CELLS) in why, why
    rc, lines, why = file_phase(w, registry, dry=True, override_id="other_doc")
    assert rc == 2 and why.startswith("--override-id other_doc is not the receipt's") and OVERRIDE_ID in why, why
    assert (state / SF.SPEC_NAME).read_bytes() == spec_bytes
    # the same cross-checks, as 07 passes them, agree: composed again, byte-identical
    rc, lines, why = file_phase(w, registry, dry=True)
    assert rc == 0 and (state / SF.SPEC_NAME).read_bytes() == spec_bytes
    # an older receipt carries no identity: refused by name before anything else (the receipt's bytes changed, so the verdict's note would refuse too — this names the cause)
    rpath = state / RT.STORE_DIR / RT.RECEIPT_NAME
    original = rpath.read_bytes()
    doc = json.loads(original); doc["schema"] = "stage07-receipt/2"
    rpath.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    rc, lines, why = file_phase(w, registry, dry=True, cells=None, launcher_cells=None, override_id=None)
    assert rc == 2 and "schema is 'stage07-receipt/2', not 'stage07-receipt/7'" in why and "run --phase receipt again" in why, why
    # a /6 receipt — a 200-item execution's — is refused by name too, told to re-run on the full sets
    doc = json.loads(original); doc["schema"] = "stage07-receipt/6"
    rpath.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    rc, lines, why = file_phase(w, registry, dry=True, cells=None, launcher_cells=None, override_id=None)
    assert rc == 2 and "schema is 'stage07-receipt/6', not 'stage07-receipt/7'" in why and "200-item subsets" in why and "re-run 07 on them" in why, why
    # view_cells that do not map to the receipt's tokens through the table
    doc = json.loads(original); doc["view_cells"] = ["none", "difference_relu"]
    rpath.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    rc, lines, why = file_phase(w, registry, dry=True, cells=None, launcher_cells=None, override_id=None)
    assert rc == 2 and "map to the launcher tokens" in why and "not the receipt's cells" in why, why
    doc = json.loads(original); doc["run_overrides_id"] = ""
    rpath.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    rc, lines, why = file_phase(w, registry, dry=True, cells=None, launcher_cells=None, override_id=None)
    assert rc == 2 and "names no run_overrides_id" in why, why
    # the launcher row the run was served from must be the one the table names for (key, view): a dose view's row under the wave's name — or no
    # slug at all — files another bank's objects under this view; and the sampling must be the paper's protocol (a hand-edited receipt cannot pass)
    for slug, said in ((f"{KEY}-d0p75", f"the receipt's slug '{KEY}-d0p75' is not the launcher row the models table serves {KEY}/{VIEW} from ('{KEY}')"),
                       (None, f"the receipt's slug None is not the launcher row")):
        doc = json.loads(original); doc["slug"] = slug
        rpath.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        rc, lines, why = file_phase(w, registry, dry=True, cells=None, launcher_cells=None, override_id=None)
        assert rc == 2 and said in why and "served from another row's bank" in why, why
    doc = json.loads(original); doc["sampling"] = "card"
    rpath.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    rc, lines, why = file_phase(w, registry, dry=True, cells=None, launcher_cells=None, override_id=None)
    assert rc == 2 and "the receipt's sampling 'card' is not the paper's 'protocol' decoding" in why, why
    rpath.write_bytes(original)
    rc, lines, why = file_phase(w, registry, dry=True)
    assert rc == 0 and (state / SF.SPEC_NAME).read_bytes() == spec_bytes


def test_a_receipt_stands_until_its_replacement_is_complete_and_a_superseded_one_is_archived(tmp_path):
    """A second `--phase receipt` used to rmtree the published store before its first pull, so a pull that
    failed — or a live log rewritten meanwhile — left NO receipt, and after a `--phase run` had overwritten the live log the deleted copy was the
    only evidence of the earlier execution. Now the pulls land in <state>/07_store.building/ and are published only once the receipt is
    written: every refusal leaves the published receipt, its pulled evidence and its run-log copy byte-identical and the build gone; a new
    receipt that differs archives the old one under 07_store.archive/<its sha12>/ (nothing deleted); a byte-identical one replaces in place."""
    w = make_world(tmp_path)
    state, vol, live = w["state"], w["vol"], w["state"] / RT.RUN_REPORT_NAME
    out1 = receipt(w)
    first = store_bytes(state)
    r1 = out1["receipt_sha256"]
    assert RT.RECEIPT_NAME in first and RT.RUN_REPORT_COPY in first and not (state / RT.STORE_BUILD_DIR).exists()
    # a second execution over the same stores (new owners, new pointers) overwrote the live log — the first receipt's copy is now the only
    # evidence of execution #1 — and one of its pulls FAILS: refused by name, the published receipt untouched, no build left behind
    run2 = f"{SHA[:7]}-33334444"
    for cell, fcid2 in (("none", "fc-none-0011"), (DIFF, "fc-diff-0012")):
        put_attempt(vol, cell, fcid2, key=w["keys"][cell], run_id=run2)
    live.write_text(run_report([rec_ok("none", w["keys"]["none"], "fc-none-0011", run_id=run2), rec_ok(DIFF, w["keys"][DIFF], "fc-diff-0012", run_id=run2)]))
    marker = vrel(f"{out_of(DIFF)}/cells/{w['keys'][DIFF]}/{C.DONE_DIR}/{MARK[TASK_BFCL]}")
    saved = vol.remote.pop(marker)
    with pytest.raises(RT.Refusal, match="cannot pull"):
        receipt(w)
    assert store_bytes(state) == first and not (state / RT.STORE_BUILD_DIR).exists() and not (state / RT.STORE_ARCHIVE_DIR).exists()
    vol.remote[marker] = saved
    # the live log rewritten during the pulls (round 1's refusal) and the identity rewritten during the pulls: the same — the published receipt stands
    with pytest.raises(RT.Refusal, match="changed while the receipt was being built"):
        receipt(w, runner=_RewritingVolume(vol, live, "\n"))
    assert store_bytes(state) == first and not (state / RT.STORE_BUILD_DIR).exists()
    with pytest.raises(RT.Refusal, match=re.escape(f"{state / RT.RUN_IDENTITY_NAME} changed while the receipt was being built")):
        receipt(w, runner=_RewritingVolume(vol, state / RT.RUN_IDENTITY_NAME, "\n"))
    assert store_bytes(state) == first and not (state / RT.STORE_BUILD_DIR).exists()
    live.write_text(live.read_text().rstrip("\n") + "\n")
    write_identity(state)
    # the second execution's receipt completes: published, the first kept whole under the archive by its sha
    out2 = receipt(w)
    r2 = out2["receipt_sha256"]
    assert r2 != r1
    second = store_bytes(state)
    assert hashlib.sha256(second[RT.RECEIPT_NAME]).hexdigest() == r2 and json.loads(second[RT.RECEIPT_NAME])["modal_call_ids"] == ["fc-diff-0012", "fc-none-0011"]
    archived = state / RT.STORE_ARCHIVE_DIR / r1[:12]
    assert archived.is_dir() and {str(p.relative_to(archived)): p.read_bytes() for p in sorted(archived.rglob("*")) if p.is_file()} == first
    assert not (state / RT.STORE_BUILD_DIR).exists() and any("is kept at" in ln and r1[:12] in ln for ln in out2["log"])
    # the same execution pulled again: byte-identical receipt -> replaced by the fresh pulls, no second archive entry
    out3 = receipt(w)
    assert out3["receipt_sha256"] == r2 and store_bytes(state) == second and sorted(p.name for p in (state / RT.STORE_ARCHIVE_DIR).iterdir()) == [r1[:12]]
    assert any("held this same receipt" in ln for ln in out3["log"])
    # the archived receipt is fileable as it stands: moved back (the live log aside), the file phase composes from it
    shutil.move(str(state / RT.STORE_DIR), str(state / "07_store.second"))
    shutil.move(str(archived), str(state / RT.STORE_DIR))
    live.rename(state / "07_run_report.second.log")
    assert rule(w) == 0
    rc, lines, why = file_phase(w, str(_fresh_registry(tmp_path / "reg")), dry=True)
    assert rc == 0 and why is None and json.loads((state / SF.SPEC_NAME).read_text())["extra"]["receipt_sha256"] == r1


def test_the_filed_objects_are_the_launch_trees_and_a_checkout_at_that_tree_is_cross_checked(tmp_path, registry, monkeypatch):
    """The filing's `inputs.objects` were re-planned from the CURRENT checkout, so a receipt filed after a later
    commit moved objects/DEMO.json claimed the old Modal execution exercised the new object id, under a `code_sha` naming the old tree. Now the
    object ids are the LAUNCH's (07_run_identity.json, from the clean tree the launcher got as --sha), copied into the receipt and filed from
    it alone; when this checkout IS at the receipt's tree the importer's live plan is cross-checked against the receipt's map (a disagreement
    refuses by name); at any other HEAD the receipt's map is filed as recorded and the log says the checkout was not consulted."""
    # (1) the fixture tree is not this checkout's HEAD: the receipt's map is filed, the checkout not consulted
    w = make_world(tmp_path / "w1")
    wrong = "9" * 64
    write_identity(w["state"], objects={DIFF: wrong})
    receipt(w)
    assert rule(w) == 0
    rc, lines, why = file_phase(w, registry, dry=True)
    assert rc == 0 and why is None, why
    spec = json.loads((w["state"] / SF.SPEC_NAME).read_text())
    assert spec["inputs"]["objects"] == sorted([wrong, "none"]) and spec["extra"]["objects_by_cell"] == {DIFF: wrong} and spec["code_sha"] == SHA
    assert any("not consulted" in ln and f"tree {SHA[:7]}" in ln for ln in lines), lines
    # a receipt whose objects do not cover a fitted cell, or whose `objects` is null for a run with fitted cells, refuses the filing by name
    rpath = w["state"] / RT.STORE_DIR / RT.RECEIPT_NAME
    original = rpath.read_bytes()
    for objects, msg in (({}, "has no entry for the fitted cell"), (None, "identifies no objects for the run's fitted cells")):
        doc = json.loads(original); doc["objects"] = objects
        rpath.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        assert rule(w) == 0                                                 # ruled from the rewritten receipt, so the note check passes and the objects check speaks
        rc, lines, why = file_phase(w, registry, dry=True)
        assert rc == 2 and msg in why, why
    rpath.write_bytes(original)
    assert rule(w) == 0
    # (2) at THIS checkout's HEAD: the live plan must agree with the receipt's map — the committed difference LoRA's id passes, another id refuses
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    monkeypatch.setattr(sys.modules[__name__], "SHA", head)
    monkeypatch.setattr(sys.modules[__name__], "APP", f"tinycap-{head[:7]}-{KEY}")
    w2 = make_world(tmp_path / "w2")
    receipt(w2)
    assert json.loads((w2["state"] / RT.STORE_DIR / RT.RECEIPT_NAME).read_text())["tree_sha"] == head
    assert rule(w2) == 0
    rc, lines, why = file_phase(w2, registry, dry=True)
    assert rc == 0 and why is None and any("cross-checked against the importer's plan at HEAD" in ln and "agree" in ln for ln in lines), (why, lines)
    assert json.loads((w2["state"] / SF.SPEC_NAME).read_text())["inputs"]["objects"] == sorted([DIFF_ID, "none"])
    write_identity(w2["state"], objects={DIFF: wrong})
    receipt(w2)
    assert rule(w2) == 0
    rc, lines, why = file_phase(w2, registry, dry=True)
    assert rc == 2 and f"the receipt's object for {DIFF}" in why and "are not this tree's" in why and DIFF_ID[:12] in why and wrong[:12] in why, why


def test_the_receipts_overrides_document_is_the_launchs_and_a_receipt_under_another_refuses(tmp_path, registry):
    """`07 --phase receipt` passed OVERRIDE_ID of ITS OWN call, so a run launched under document A and
    receipted without the flag was recorded — and then filed — under the default document B. The launch writes its document (OVERRIDE_ID,
    sha256, path) into 07_run_identity.json; the receipt copies that and takes the call's --override-id only as a cross-check that refuses by
    name, naming the run's document and its path; the filing's name and extra carry the launch's."""
    w = make_world(tmp_path)
    write_identity(w["state"], override="doc_a")
    with pytest.raises(RT.Refusal, match=re.escape(f"--override-id {OVERRIDE_ID} is not the run's: the run was launched under run-overrides doc_a ({OVERRIDE_PATH}")) as ei:
        receipt(w)
    assert f"--overrides {OVERRIDE_PATH}" in str(ei.value) and not (w["state"] / RT.STORE_DIR).exists()
    out = receipt(w, override_id=None)                                      # no cross-check passed: the launch's document is recorded
    rec = json.loads((w["state"] / RT.STORE_DIR / RT.RECEIPT_NAME).read_text())
    assert rec["run_overrides_id"] == "doc_a" and rec["run_overrides_sha256"] == OVERRIDE_SHA and rec["run_overrides_path"] == OVERRIDE_PATH
    receipt(w, override_id="doc_a")                                         # the matching cross-check: byte-identical
    assert receipt_sha(w) == out["receipt_sha256"]
    assert rule(w) == 0
    rc, lines, why = file_phase(w, registry, dry=True, override_id="doc_a")
    assert rc == 0 and why is None, why
    spec = json.loads((w["state"] / SF.SPEC_NAME).read_text())
    assert spec["name"] == f"demo_import_07_bcb_bfcl_{VIEW}_doc_a_x{out['receipt_sha256'][:12]}" and spec["extra"]["run_overrides_id"] == "doc_a"
    assert spec["extra"]["run_overrides_sha256"] == OVERRIDE_SHA
    rc, lines, why = file_phase(w, registry, dry=True, override_id=OVERRIDE_ID)
    assert rc == 2 and why.startswith(f"--override-id {OVERRIDE_ID} is not the receipt's") and "doc_a" in why


# ------------------------------------------------------------------------------------------------------------ the filing
def test_the_file_phase_refuses_a_missing_verdict_or_receipt_by_name(tmp_path, registry):
    w = make_world(tmp_path)
    rc, lines, why = file_phase(w, registry)
    assert rc == 2 and "no verdict at" in why and "stage.py verdict" in why
    receipt(w)
    rc, lines, why = file_phase(w, registry)
    assert rc == 2 and "no verdict at" in why
    assert rule(w) == 0
    shutil.rmtree(w["state"] / RT.STORE_DIR)
    rc, lines, why = file_phase(w, registry)
    assert rc == 2 and "no receipt at" in why and "run --phase receipt first" in why
    assert not (w["state"] / SF.SPEC_NAME).exists()


def test_the_filing_composes_from_the_receipt_and_the_verdict_alone_files_folds_reads_back_and_the_identity_cases_hold(tmp_path, registry):
    root = registry
    w = make_world(tmp_path / "w1")
    state = w["state"]
    out = receipt(w)
    r1 = out["receipt_sha256"]
    assert rule(w) == 0
    vpath = state / "gate" / "bcb_bfcl" / f"{VIEW}.json"
    verdict = json.loads(vpath.read_text())
    assert verdict["headline"] == 0.455 and verdict["verdict"] == "pass" and f"receipt {r1[:12]}" in verdict["note"]
    # the dry run composes and writes the spec, prints the pre-write check and the filing command, touches no registry
    before_filings = _filings(root)
    rc, lines, why = file_phase(w, root, dry=True)
    assert rc == 0 and why is None and any("DRY-RUN: registry analysis file --spec" in ln and "--dry-run" in ln for ln in lines), lines
    spec = json.loads((state / SF.SPEC_NAME).read_text())
    name = f"demo_import_07_bcb_bfcl_{VIEW}_{OVERRIDE_ID}_x{r1[:12]}"
    assert spec["name"] == name and spec["code_sha"] == SHA and spec["code_ref"] == f"demo-import-smoke @ {SHA[:7]}" and spec["substrate"] == "modal"
    assert spec["cluster_job_ids"] == [] and spec["report"] == {"uri": str(vpath), "sha256": None} and spec["inputs"] == {"runs": [], "objects": sorted([DIFF_ID, "none"])}
    assert set(spec) == set(AN.SPEC_KEYS) and spec["provenance"] == {"kind": "user_direct"} and spec["artifacts"] == []
    ex = spec["extra"]
    rec = json.loads((state / RT.STORE_DIR / RT.RECEIPT_NAME).read_text())
    assert ex["headline"] == rec["headline"] and ex["bfcl"] == rec["bfcl"] and ex["receipt_sha256"] == r1 and ex["run_report_sha256"] == rec["run_report_sha256"]
    assert ex["modal_call_ids"] == rec["modal_call_ids"] and ex["store_cell_keys"] == rec["store_cell_keys"] and ex["store_out_dirs"] == rec["store_out_dirs"]
    assert ex["modal_app_name"] == APP and ex["controls"] == ",".join(CONTROLS) and ex["n_items_per_task"] == {TASK_BCB: N_BCB, TASK_BFCL: N_BFCL}
    assert ex["launcher_slug"] == KEY and ex["sampling"] == SAMPLING
    assert ex["cells"] == ["none", "difference_lora"] and ex["launcher_cells"] == list(CELLS) and ex["launcher_exit"] == 0 and ex["run_overrides_id"] == OVERRIDE_ID
    assert ex["spec_deviation"] == SF.SPEC_DEVIATION and ex["expect_hashes"] is None and ex["execution"] == "new" and ex["verdict"]["ruled_utc"] == verdict["ruled_utc"]
    assert ex["done_markers"] == {t: sorted(MARK.values()) for t in CELLS} and _filings(root) == before_filings
    # the real filing: written, not yet folded -> exit 0 and the pending header named; the fold does not have it yet
    rc, lines, why = file_phase(w, root)
    assert rc == 0 and why is None and any("filed: analysis_id" in ln for ln in lines) and any("filed, not yet folded" in ln for ln in lines), lines
    aid = AN.analysis_id(AN.normalize_spec(spec))
    assert len(_filings(root)) == len(before_filings) + 1 and _count(root, "SELECT count(*) FROM analysis WHERE name = ?", name) == 0
    spec_bytes = (state / SF.SPEC_NAME).read_bytes()
    # (γ, before any ingest) a re-ruled verdict refuses by name, naming the pending filing's directory; ONE filing stays
    assert rule(w, rng="0.2,0.7") == 0
    rc, lines, why = file_phase(w, root)
    assert rc == 2 and "already filed as analysis" in why and aid in why and "pending" in why and "RE-RULED" in why and _filings(root)[-1].split("/")[-1] in why, why
    assert len(_filings(root)) == len(before_filings) + 1
    # restore the exact filed verdict bytes (the operator keeping the filed ruling) and fold
    filed_report_sha = json.loads(open(os.path.join(_filings(root)[-1], sorted(os.listdir(_filings(root)[-1]))[1])).readline())["payload"]["report_sha256"]
    vpath.write_text(json.dumps(verdict, indent=2, sort_keys=True))
    assert hashlib.sha256(vpath.read_bytes()).hexdigest() == filed_report_sha
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0 and _count(root, "SELECT count(*) FROM analysis WHERE name = ?", name) == 1
    # (α) a FILE-ONLY retry: byte-identical spec, the same analysis_id, a duplicate at the fold; the three-way read-back passes
    rc, lines, why = file_phase(w, root)
    assert rc == 0 and any("already filed as" in ln and "(folded)" in ln for ln in lines) and any(f"PASS: {name} analysis_id {aid}" in ln for ln in lines), lines
    assert (state / SF.SPEC_NAME).read_bytes() == spec_bytes and len(_filings(root)) == len(before_filings) + 1
    rows = json.loads(_cli_out(["--root", root, "analysis", "list", "--name", name, "--json"]))
    assert len(rows) == 1 and rows[0]["analysis_id"] == aid and rows[0]["substrate"] == "modal" and json.loads(rows[0]["cluster_job_ids_json"]) == []
    assert rows[0]["report_sha256"] == hashlib.sha256(vpath.read_bytes()).hexdigest() and rows[0]["code_sha"] == SHA
    assert sorted(i["input_id"] for i in rows[0]["inputs"] if i["input_kind"] == "object") == sorted([DIFF_ID, "none"]) and not [i for i in rows[0]["inputs"] if i["input_kind"] == "run"]
    assert any(a["role"] == "report" and a["artifact_id"] == rows[0]["report_sha256"] for a in rows[0]["artifacts"])
    q = json.loads(_cli_out(["--root", root, "query", "--json", f"SELECT analysis_id, extra_json FROM analysis WHERE name = '{name}'"]))
    extra = json.loads(q[0]["extra_json"])
    assert extra["receipt_sha256"] == r1 and extra["headline"] == rec["headline"] and extra["modal_call_ids"] == rec["modal_call_ids"] and extra["decision"] == "D357"
    # (α′) a later --phase run overwrote the live log: refused BEFORE composing (the spec unchanged, no filing), and the same bytes once moved aside
    live = state / RT.RUN_REPORT_NAME
    live.write_text(live.read_text() + json.dumps(rec_ok("none", w["keys"]["none"], "fc-none-0009")) + "\n")
    rc, lines, why = file_phase(w, root)
    assert rc == 2 and "the run report on disk is not this receipt's" in why and "move the live log aside" in why
    assert (state / SF.SPEC_NAME).read_bytes() == spec_bytes and len(_filings(root)) == len(before_filings) + 1
    live.rename(state / "07_run_report.later.log")
    rc, lines, why = file_phase(w, root)
    assert rc == 0 and (state / SF.SPEC_NAME).read_bytes() == spec_bytes
    (state / "07_run_report.later.log").rename(live.with_name("unused.log"))
    # (δ) a verdict whose headline, or whose run ids, are not the receipt's: refused by name, nothing written
    assert rule(w, headline=0.5, rng="0.3,0.6") == 0
    rc, lines, why = file_phase(w, root)
    assert rc == 2 and "is not the receipt's none BCB acc 0.455" in why
    assert rule(w, run_ids=[w["keys"]["none"]]) == 0
    rc, lines, why = file_phase(w, root)
    assert rc == 2 and "are not the receipt's store keys" in why
    assert rule(w, note="headline: none inspect/bigcodebench_full acc; receipt 000000000000") == 0
    rc, lines, why = file_phase(w, root)
    assert rc == 2 and "does not carry 'receipt " in why
    assert len(_filings(root)) == len(before_filings) + 1
    vpath.write_text(json.dumps(verdict, indent=2, sort_keys=True))
    # (γ, after the fold) a re-ruled verdict (another --range -> another ruled_utc and sha) refuses by name, naming the folded analysis and both shas
    assert rule(w, rng="0.25,0.65") == 0
    new_sha = hashlib.sha256(vpath.read_bytes()).hexdigest()
    rc, lines, why = file_phase(w, root)
    assert rc == 2 and aid in why and filed_report_sha in why and new_sha in why and "(folded" in why and "ruled_utc" in why and "RE-RULED" in why, why
    assert len(_filings(root)) == len(before_filings) + 1
    vpath.write_text(json.dumps(verdict, indent=2, sort_keys=True))
    # (β) a SECOND EXECUTION over the same stores: a new attempt pointer per root beside the first, a new run report with the same done_in keys
    #     and new owners -> a different receipt, name and analysis_id; both rows fold with zero refusals
    run2 = f"{SHA[:7]}-33334444"
    for cell, fcid2 in (("none", "fc-none-0011"), (DIFF, "fc-diff-0012")):
        put_attempt(w["vol"], cell, fcid2, key=w["keys"][cell], run_id=run2)
    live.write_text(run_report([rec_ok("none", w["keys"]["none"], "fc-none-0011", run_id=run2), rec_ok(DIFF, w["keys"][DIFF], "fc-diff-0012", run_id=run2)]))
    out3 = receipt(w)
    r3 = out3["receipt_sha256"]
    rec3 = json.loads((state / RT.STORE_DIR / RT.RECEIPT_NAME).read_text())
    assert r3 != r1 and rec3["store_cell_keys"] == rec["store_cell_keys"] and rec3["headline"] == rec["headline"]
    assert rec3["modal_call_ids"] == ["fc-diff-0012", "fc-none-0011"] and sorted(a["function_call_id"] for a in rec3["prior_attempts"]) == sorted(w["fcids"].values())
    # (β′) the verdict ruled from receipt #1 (same headline, same keys) against receipt #2: refused on the note
    assert f"receipt {r1[:12]}" in json.loads(vpath.read_text())["note"]
    rc, lines, why = file_phase(w, root)
    assert rc == 2 and "does not carry 'receipt " in why and r3[:12] in why
    assert rule(w, sha12=r3[:12]) == 0
    rc, lines, why = file_phase(w, root)
    name3 = f"demo_import_07_bcb_bfcl_{VIEW}_{OVERRIDE_ID}_x{r3[:12]}"
    assert rc == 0 and any("filed, not yet folded" in ln for ln in lines) and json.loads((state / SF.SPEC_NAME).read_text())["name"] == name3
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0
    rows = json.loads(_cli_out(["--root", root, "analysis", "list", "--json"]))
    assert {r["name"] for r in rows if r["name"].startswith("demo_import_07_")} == {name, name3} and len({r["analysis_id"] for r in rows}) == len(rows)
    rc, lines, why = file_phase(w, root)
    assert rc == 0 and any(f"PASS: {name3}" in ln for ln in lines)
    # the registry's side of the identity rule the suffix exists for: two different specs filed under ONE name fold to ONE row and a refusal
    spec_a = json.loads((state / SF.SPEC_NAME).read_text()); spec_b = json.loads(json.dumps(spec_a))
    spec_a["name"] = spec_b["name"] = "demo_import_07_constant_name_pin"; spec_b["extra"]["modal_call_ids"] = ["fc-other"]
    for i, sp in enumerate((spec_a, spec_b)):
        p = tmp_path / f"const{i}.json"; p.write_text(json.dumps(sp))
        assert cli.main(["--root", root, "analysis", "file", "--spec", str(p)]) == 0
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused >= 1 and _count(root, "SELECT count(*) FROM analysis WHERE name = ?", "demo_import_07_constant_name_pin") == 1
    assert any("conflicting re-filing" in str(r) for r in rep.refusals), rep.refusals


def test_an_input_object_the_registry_lacks_refuses_the_filing_and_writes_nothing(tmp_path):
    root = str(_fresh_registry(tmp_path / "seeds_only"))            # the seeds alone: `none` exists, the difference LoRA does not
    w = make_world(tmp_path / "w")
    receipt(w)
    assert rule(w) == 0
    rc, lines, why = file_phase(w, root)
    assert rc == 2 and "input(s) not in this registry" in why and DIFF_ID in why and _filings(root) == []


def _cli_out(argv) -> str:
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert cli.main(argv) == 0
    return buf.getvalue()
