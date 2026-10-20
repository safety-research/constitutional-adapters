#!/usr/bin/env python
"""bcb-bfcl-200-widen S6e: MEASURE qwen35_4b's batched-token budget on the
served stack, on Modal, <= 1 H200-h, metered as stage kind `budget` (unit `budget:qwen35_4b`).

    TINYCAP_APP_NAME=tinycap-bcbw-qwen-budget-<sha7> ~/venvs/modal/bin/python -m modal run --detach \\
        scripts/modal/tinycap_budget_measure.py --sha <40-hex> --model qwen35_4b-f22

Why: `geometry.BUDGET_BY_MODEL["qwen35_4b"]` is a PROVISIONAL ceiling (360,448 = 22 x 16,384,
halved from the 2026-09-04 sweep as a same-stack precaution; nothing exercised the lens stack
at it), and the truncation ladder REFUSES a provisional budget (`rung_probe.ladder_spec`) — so
qwen's truncation rerun was unexecutable. This walks the sweep's budget bracket
(`scripts/served_decode_sweep.py`'s pure helpers `next_budget_probe` / `budget_bracket` /
`ladder_seqs`, REUSED, over its `BUDGET_LADDER` at the budget window L = 8,192 — every measured
row's basis) standing up the row's `none` cell with the PRODUCTION standup (the launcher's own
`_standup_cmd`: `--enforce-eager` inherited from the launch, gpu_memory_utilization 0.90, the
text-only-MM pin under the standup's `auto` default, the hybrid per-slot cap 256 on S) at each
candidate S with `--allow-over-budget` (the probe must EXCEED the provisional ceiling to find
the wall), reading the readiness verdict — `endpoint.json` = pass; the KV-coverage refusal
(`rung_probe.kv_coverage_refusal`) = a wall with `kv_tokens` — and STOPPING at the first
refusal + the sweep's one bisection (<= 6 standups, ~5 min each). It writes `budget.json` in
the sweep analyzer's shape to `/vol/tinycap/geom/<model_key>/<sha7>/` and PRINTS the
`geometry._MEASURED_BUDGETS` block (`geom_sweep_analyze.format_paste_block`) — `probe_job_id`
= the Modal FunctionCall id, `source` = the volume path, `certified False`, source `measured`.

The orchestrator pastes that block into `_MEASURED_BUDGETS`, DELETES qwen35_4b's
`_PROVISIONAL_BUDGETS` row (`_build_budget_table` refuses a provisional ceiling not below its
measurement — the import-time check is the guard), moves `test_model_budget.py` +
`docs/served_geometry_p99.md`, and — if the measured S at 16,384 differs from 22 — re-runs the
qwen smoke at the new geometry before the next wave. Nothing here writes to geometry.py.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import modal

HERE = Path(__file__).resolve().parent


def _repo_local() -> Path:
    """The checkout root on the LOCAL side (this file lives at scripts/modal/). Inside the Modal
    container the script is mounted flat at ``/root/<name>.py`` with no such ancestor —
    ``HERE.parents[1]`` raised ``IndexError: 1`` there and crash-looped every container of this
    app (confirmed live) — so walk up to the first ancestor that IS a checkout and fall
    back to the script's own directory: the container never reads the local tree (its
    subprocesses run the STAGED one from the volume) and reaches ``sisteer`` through the
    launcher image's ``add_local_python_source``. The same rule as ``tinycap_served._repo_local``."""
    for parent in (HERE, *HERE.parents):
        try:
            if (parent / "src" / "sisteer").is_dir() and (parent / "scripts" / "modal").is_dir():
                return parent
        except OSError:            # an unreadable ancestor is not a checkout
            continue
    return HERE


REPO_LOCAL = _repo_local()
if str(REPO_LOCAL / "src") not in sys.path:
    sys.path.insert(0, str(REPO_LOCAL / "src"))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from sisteer.capeval.tinycap import contracts as C  # noqa: E402

#: The app is named by the launch wrapper's convention; the launcher module the image comes
#: from needs the same environment variable at import (it refuses an unnamed local run).
APP_NAME_ENV = C.MODAL_APP_NAME_ENV
APP_NAME = os.environ.get(APP_NAME_ENV, "").strip()
_is_local = getattr(modal, "is_local", None)
if not APP_NAME and _is_local is not None and _is_local():
    raise SystemExit(f"[budget] {APP_NAME_ENV} is not set: run as "
                     f"{APP_NAME_ENV}=tinycap-bcbw-qwen-budget-<sha7> modal run --detach scripts/modal/tinycap_budget_measure.py …")
import tinycap_served as T  # noqa: E402  (the launcher: its image, volume, standup composition, staged-tree resolver)

#: Confirmed by executing both scripts at ``/root/<name>.py`` with no checkout on sys.path:
#: Modal mounts each function's OWN entrypoint, and the shared image ships
#: ``tinycap_images.py`` + ``sisteer`` but NOT ``tinycap_served.py`` — so the line above raised
#: ``ModuleNotFoundError`` inside every container of this app. The launcher module is mounted at
#: ``/root`` beside the entrypoint on EVERY image this script's functions run on; the inventory a
#: container root holds is ``container_inventory()`` (the launcher's declared files + these), and
#: the regression test rebuilds a root from exactly that inventory with the checkout's
#: ``scripts/modal`` OFF sys.path.
LAUNCHER_REMOTE = "/root/tinycap_served.py"
CONTAINER_FILES: Dict[str, str] = {LAUNCHER_REMOTE: str(Path(T.__file__).resolve())}


def with_container_files(img):
    """``img`` + every :data:`CONTAINER_FILES` mount (a mount, not a copy layer: the launcher
    changes every commit and nothing builds on top of it here)."""
    for remote, local in CONTAINER_FILES.items():
        img = img.add_local_file(local, remote)
    return img


def container_inventory() -> Dict[str, str]:
    """``{remote path under /root: local source}`` of every file a container of this app holds
    beside its entrypoint — the launcher's own image files plus this script's mounts."""
    return {**dict(T.IMAGE_LOCAL_FILES), **CONTAINER_FILES}


IMAGE = with_container_files(T.image)

app = modal.App(APP_NAME or "tinycap-budget-remote-import")

#: The budget window every measured row's basis uses (`geometry._MEASURED_BUDGETS`), the S cap
#: for a hybrid-recurrent family (per-slot state; `served_decode_sweep.S_MAX_HYBRID`), the ceiling.
BUDGET_LEN = 8192
S_MAX_HYBRID = 256
MAX_STANDUPS = 8
GEOM_ROOT = f"{C.VOL_TINYCAP_ROOT}/geom"
STANDUP_WAIT_S = 45 * 60


def _sweep(tree: str):
    """The sweep script's PURE helpers, loaded by path from the STAGED tree."""
    return T._load_tree_module(tree, "scripts/served_decode_sweep.py", "served_decode_sweep_staged")


def _analyzer(tree: str):
    """The geometry analyzer, loaded by path from the STAGED tree. Its helpers import
    ``served_decode_sweep`` by NAME at call time (its own sys.path insert is undone by the
    tree loader, and the container path holds no checkout), so the STAGED sweep module is
    registered under that name first — the paste block used to fail live with
    ``format_paste_block unavailable: ModuleNotFoundError: served_decode_sweep``."""
    sys.modules.setdefault("served_decode_sweep", _sweep(tree))
    return T._load_tree_module(tree, "scripts/geom_sweep_analyze.py", "geom_sweep_analyze_staged")


def row_in_force(model_key: str) -> Optional[Dict[str, Any]]:
    """The geometry row this process's ``geometry.BUDGET_BY_MODEL`` holds for ``model_key``
    (measured or provisional; None when the table has none) — what the measurement replaces
    or confirms, stated from the row rather than a literal (the tool used to print
    "provisional was 22" for every model)."""
    from sisteer.capeval.serving import geometry as sg
    row = sg.BUDGET_BY_MODEL.get(model_key)
    if row is None:
        return None
    return {"source": str(row.source), "batched_token_budget": int(row.batched_token_budget),
            "seqs_at_16384": int(row.seqs_at(16_384)), "provisional": bool(row.provisional)}


#: The sweep's wall statuses (`served_decode_sweep.WALL_STATUSES`): a KV-coverage refusal at
#: readiness, or an OOM / "no memory for the cache blocks" at engine init. `next_budget_probe`
#: and `budget_bracket` read a probe as a wall ONLY under one of these (with its ``product``).
STATUS_PASS, STATUS_KV_WALL = "pass", "kv_under_coverage"


def probe_seqs(product: int, budget_len: int) -> int:
    """S for a ladder product at the window (the product must be an exact multiple)."""
    if product % budget_len:
        raise ValueError(f"budget ladder product {product:,} is not a multiple of the window {budget_len}")
    return product // budget_len


def cap_binds(S: int, *, hybrid: bool) -> bool:
    """A hybrid-recurrent row cannot be stood up above the per-slot cap (`geometry.HYBRID_MODEL_KEYS`
    caps max_num_seqs at 256): a rung whose S exceeds it is UNPROBEABLE — the walk stops with a
    note rather than probing a smaller S and filing it under the rung's product."""
    return bool(hybrid) and int(S) > S_MAX_HYBRID


#: the sweep's KV-capacity kinds (`served_decode_sweep.attested_kv_capacity`): a hybrid-recurrent
#: row reports the allocator's max concurrency x L, a uniform cache blocks x block size
KV_KIND_HYBRID = "hybrid_max_concurrency_x_L"
KV_KIND_UNIFORM = "uniform_blocks_x_block_size"


def probe_record(*, product: int, S: int, budget_len: int, status: str, seconds: float, kv_tokens: Optional[int],
                 error: Optional[str], job_id: str, tag: str, hybrid: bool = True) -> Dict[str, Any]:
    """One probe in the sweep's `budget.json` shape (`is_wall` reads `status` + `product`); the
    `kv_kind` follows the ROW (code review: it was hard-coded hybrid regardless of `hybrid`)."""
    return {"tag": tag, "product": int(product), "max_num_seqs": int(S), "max_model_len": int(budget_len),
            "status": status, "kv_tokens": kv_tokens, "kv_kind": KV_KIND_HYBRID if hybrid else KV_KIND_UNIFORM,
            "seconds": round(seconds, 1), "error": error, "job_id": job_id, "node": "modal-h200", "log_max_concurrency_x": None}


def _measure(sha: str, model_slug: str, *, budget_len: int = BUDGET_LEN, max_standups: int = MAX_STANDUPS) -> dict:
    from sisteer.capeval.tinycap import rung_probe as RP
    row = C.model_row(model_slug)
    rec: Dict[str, Any] = {"name": "budget-measure", "model": row.slug, "model_key": row.model_key, "sha": sha[:7],
                           "budget_len": budget_len, "probes": []}
    try:
        tree_path, tree_sha = T.resolve_staged_tree(sha)
    except (T.NoStagedTree, T.TreeShaMissing, T.TreeShaMismatch) as e:
        return {**rec, "status": "NO_TREE", "error": f"{type(e).__name__}: {e}"}
    tree = str(tree_path)
    rec["tree_sha"] = tree_sha
    sw = _sweep(tree)
    ladder = [int(x) for x in sw.BUDGET_LADDER]
    from sisteer.capeval.serving import geometry as sg
    hybrid = row.model_key in sg.HYBRID_MODEL_KEYS
    out_dir = Path(f"{GEOM_ROOT}/{row.model_key}/{sha[:7]}")
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        cfg_path, overlaid = T.overlaid_config(row, tree, out_dir)
    except (KeyError, OSError, ValueError) as e:
        return {**rec, "status": "CONFIG_OVERLAY_FAILED", "error": f"{type(e).__name__}: {e}"}
    job_id = T.current_function_call_id()
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{tree}/src:{tree}/scripts"
    env[C.LENS_STACKED_REDUCTIONS_ENV] = C.REDUCTION_MODE_LAUNCHER_DEFAULT
    probes: List[Dict[str, Any]] = []
    doc: Dict[str, Any] = {"model_key": row.model_key, "hf": row.hf, "rev": row.rev, "config": cfg_path, "cell": "none",
                           "max_model_len": budget_len, "ladder": ladder, "gpu_mem_util": 0.90, "node": "modal-h200",
                           "job_id": job_id, "tree_sha": tree_sha, "hybrid": hybrid, "probes": probes, "complete": False,
                           "measured_by": "scripts/modal/tinycap_budget_measure.py (bcb-bfcl-200-widen S6e)"}
    (out_dir / "budget.json").write_text(json.dumps(doc, indent=1, default=str))
    t0 = time.time()
    while len(probes) < max_standups:
        nxt = sw.next_budget_probe(probes, ladder, budget_len)
        if nxt is None:
            break
        S = probe_seqs(nxt, budget_len)
        if cap_binds(S, hybrid=hybrid):
            doc["note"] = (f"the hybrid per-slot cap {S_MAX_HYBRID} binds before the wall (next rung {nxt:,} = {S} x "
                           f"{budget_len}) — the ladder above is unprobeable; the budget is the largest passing probe")
            break
        geom = SimpleNamespace(max_model_len=budget_len, max_num_seqs=S, model_key=row.model_key)
        srv = Path(f"/tmp/budget_probe_{row.slug}_{S}")
        if srv.exists():
            shutil.rmtree(srv)
        srv.mkdir(parents=True)
        stop = srv / "STOP"
        argv = T._standup_cmd(tree, cfg_path, "none", row.objects_dir, srv, stop, geom, row.top_logprobs, [],
                              allow_over_budget=True)
        print(f"[budget] probing {row.model_key} S={S} x L={budget_len} (product {nxt:,}): {' '.join(argv)}", flush=True)
        t_s = time.time()
        proc = subprocess.Popen(argv, env=env, stdout=(srv / "standup.log").open("w"), stderr=subprocess.STDOUT)
        ready = T._wait_for(srv / "endpoint.json", proc, STANDUP_WAIT_S)
        log_text = T._tail(srv / "standup.log", 200_000) or ""
        tag = f"S{S}_L{budget_len}"
        for name in ("standup.log", "vllm_server.log"):
            if (srv / name).exists():
                (out_dir / f"probe_{tag}_{name}").write_text((srv / name).read_text(errors="replace")[-2_000_000:])
        refusal = None if ready or proc.poll() is None else RP.kv_coverage_refusal(log_text)
        T._tear_down_standup(stop, proc)
        seconds = time.time() - t_s
        if ready:
            probes.append(probe_record(product=nxt, S=S, budget_len=budget_len, status=STATUS_PASS, seconds=seconds,
                                       kv_tokens=None, error=None, job_id=job_id, tag=tag, hybrid=hybrid))
        elif refusal is not None:
            probes.append(probe_record(product=nxt, S=S, budget_len=budget_len, status=STATUS_KV_WALL, seconds=seconds,
                                       kv_tokens=RP.parse_kv_tokens(refusal), error=refusal[-600:], job_id=job_id, tag=tag,
                                       hybrid=hybrid))
        else:
            # the sweep's OWN classifier and wall vocabulary, read directly (code review: a guarded
            # getattr degraded a staged tree lacking them to "not_ready" / "nothing is a wall")
            status = sw.classify_launch_failure(log_text[-4000:], log_text)
            probes.append(probe_record(product=nxt, S=S, budget_len=budget_len, status=status, seconds=seconds, kv_tokens=None,
                                       error=log_text[-1500:], job_id=job_id, tag=tag, hybrid=hybrid))
            doc["probes"] = probes
            (out_dir / "budget.json").write_text(json.dumps(doc, indent=1, default=str))
            T.VOL.commit()
            if status not in sw.WALL_STATUSES:
                rec.update(status="STANDUP_FAILED", error=f"probe {tag} neither came up nor named a KV-coverage refusal "
                                                              f"({status}); the walk stops WITHOUT recording a wall",
                           probes=probes, out_dir=str(out_dir), engine_tail=T.engine_core_tail(T._tail(srv / "vllm_server.log", 2_000_000)))
                return rec
        doc["probes"] = probes
        (out_dir / "budget.json").write_text(json.dumps(doc, indent=1, default=str))
        T.VOL.commit()
    bracket = sw.budget_bracket(probes)
    doc.update(bracket)
    doc["complete"] = sw.next_budget_probe(probes, ladder, budget_len) is None
    doc["measured_on"] = time.strftime("%Y-%m-%d")
    doc["wall_clock_s"] = round(time.time() - t0, 1)
    doc["paste"] = sw.paste_record(model_key=row.model_key, bracket=bracket, max_model_len=budget_len, gpu_mem_util=0.90,
                                   job_id=job_id, measured_on=doc["measured_on"], results_dir=str(out_dir), node="modal-h200")
    if doc["paste"] is not None:
        doc["paste"]["basis"]["source"] = f"{out_dir}/budget.json (Modal FunctionCall {job_id})"
        doc["paste"]["basis"]["note"] = "; ".join(part for part in (
            str(doc["paste"]["basis"].get("note") or "").strip(),
            f"measured on the SERVED lens stack (bcb-bfcl-200-widen S6e; image {T.image_install_specs()[0][:40]}…)") if part)
    (out_dir / "budget.json").write_text(json.dumps(doc, indent=1, default=str))
    T.VOL.commit()
    rec.update(probes=probes, bracket=bracket, out_dir=str(out_dir), budget_json=str(out_dir / "budget.json"),
               paste=doc["paste"], wall_clock_s=doc["wall_clock_s"])
    lp = bracket.get("largest_passing")
    if not lp:
        rec.update(status="NO_PASSING_PROBE", error=f"no probe passed at the budget window — {row.model_key}'s geometry row "
                                                     "stays as it is (measured or provisional); its ladder stays as it is (recorded)")
        return rec
    try:
        block = _analyzer(tree).format_paste_block([doc["paste"]])
    except Exception as e:  # noqa: BLE001 — the paste dict is the record; the pretty block is a courtesy
        block = f"# (format_paste_block unavailable: {type(e).__name__}: {e})\n" + json.dumps(doc["paste"], indent=1)
    rec["paste_block"] = block
    S16 = int(lp["product"]) // 16_384
    rec["seqs_at_16384"] = min(S16, S_MAX_HYBRID) if hybrid else S16
    rec["row_in_force"] = row_in_force(row.model_key)
    rec["status"] = "OK" if doc["complete"] else "INCOMPLETE"
    print(f"[budget] PASTE INTO geometry._MEASURED_BUDGETS as {row.model_key}'s row (REPLACING its measured row, and deleting "
          f"its _PROVISIONAL_BUDGETS row if one shadows it):\n" + block, flush=True)
    rif = rec["row_in_force"]
    was = (f"{rif['source']} {rif['batched_token_budget']:,} -> S {rif['seqs_at_16384']} at 16,384" if rif
           else "no geometry row for this model")
    print(f"[budget] largest passing {lp['product']:,} = {lp['max_num_seqs']} x {budget_len}; first refused "
          f"{bracket.get('first_refused')}; S at 16,384 -> {rec['seqs_at_16384']} (row in force: {was})", flush=True)
    return rec


@app.function(image=IMAGE, volumes={T.VOL_PATH: T.VOL}, gpu="H200", timeout=3 * 3600)
def measure_budget(sha: str, model_slug: str = "qwen35_4b-f22", budget_len: int = BUDGET_LEN,
                   max_standups: int = MAX_STANDUPS) -> dict:
    rec = _measure(sha, model_slug, budget_len=budget_len, max_standups=max_standups)
    print("[tinycap] REC", json.dumps({k: v for k, v in rec.items() if k != "paste_block"}, default=str))
    return rec


@app.local_entrypoint()
def main(sha: str = "", model: str = "qwen35_4b-f22", budget_len: int = BUDGET_LEN, max_standups: int = MAX_STANDUPS,
         dry_run: bool = False):
    if not sha:
        raise SystemExit("pass --sha <the staged tree sha>")
    row = C.model_row(model)
    from sisteer.capeval.serving import geometry as sg
    b = sg.BUDGET_BY_MODEL.get(row.model_key)
    print(f"[budget] {row.slug} ({row.hf}@{row.rev[:12]}, model_key {row.model_key}): current budget row "
          f"{b.source if b else None} {b.batched_token_budget if b else None} — measuring at L={budget_len} over the sweep ladder, "
          f"<= {max_standups} standups, app {APP_NAME!r}")
    if dry_run:
        print("[budget] dry run: nothing launched")
        return
    rec = measure_budget.remote(sha, model, budget_len, max_standups)
    print(json.dumps({k: v for k, v in rec.items() if k != "paste_block"}, indent=1, default=str))
    if rec.get("paste_block"):
        print(rec["paste_block"])
    if rec.get("status") != "OK":
        raise SystemExit(f"[budget] ended {rec.get('status')}: {str(rec.get('error'))[:400]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sha", required=True)
    ap.add_argument("--model", default="qwen35_4b-f22")
    a = ap.parse_args()
    main(sha=a.sha, model=a.model, dry_run=True)
