"""The full-shape per-cell store the inode rules count (tinycap-shared-standup §1.9.2, inode rule 2) — a
test HELPER shared by the row-engine suite and the inode-rules suite (module E writes it, both
count with it): ``full_shape_store`` writes today's 36-inode layout THROUGH THE REAL WRITERS where
one exists (``completion.finish_task`` for the two full legs — summary, outcomes, done marker —
``contracts.inspect_log_dir`` for the ``.eval`` per log dir, ``sandbox_scope.append_event`` for the
provider's ``watch/sandboxes_<fcid>.jsonl``) and the driver's other files by name, plus the row
engine's ``standup/`` (1 dir + 3 files) = exactly 40; ``count_inodes`` is ``find <store> | wc -l``
(the store dir itself, every directory, every file).

The layout is the mirrored gemma ``none`` full store's (12 directories including the root + 24
files): ``RESULT``, ``driver_stderr.log``, ``driver_stdout.log``, ``evidence.json``,
``lane_roster.json``, ``ledger.log``, ``protocol_check.json``, ``resolved_sampler.json``,
``served_geometry.json``, ``usage_ledger.jsonl``, ``<cell>/controls_summary.json``,
``watch/sandboxes_<fcid>.jsonl``, and under ``cells/<key>/``: ``manifest.json``, ``evidence.json``,
``gen_stats.json``, ``rows.jsonl``, ``done/`` + ``summary/`` + ``outcomes/`` per task, and
``private/inspect_logs/<task>__s0/<log>.eval`` per task. A SOLO full store since §1.10 also carries
``watch/attempt_<fcid>.json`` + ``watch/sweep_<fcid>.json``, and since stage 07's receipt round 4 the
attempt's ``watch/completion_<fcid>.json`` (``solo_pointer=True``) = 39.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from sisteer.capeval.tinycap import completion
from sisteer.capeval.tinycap import contracts as C
from sisteer.capeval.tinycap import sandbox_scope as scope
from sisteer.registry import synth

BCB_FULL = C.inspect_task("bigcodebench_full")
BFCL_FULL = C.inspect_task("bfcl_full")
FULL_TASKS: Tuple[str, str] = (BCB_FULL, BFCL_FULL)

#: the numbers the suites hold a store to
SOLO_FULL_SHAPE_INODES = 36            # today's full-shape solo store (the mirrored gemma `none`)
SOLO_FULL_SHAPE_WITH_POINTER_INODES = 39   # + attempt pointer + sweep certificate (§1.10, S13) + completion record (stage 07 receipt, round 4)
ROW_ENGINE_STORE_INODES = 40           # + standup/ (1 dir + 3 files); zero margin

STANDUP_DIR = "standup"
STANDUP_FILES: Tuple[str, ...] = ("endpoint.json", "serving_attestation.json", "vllm_server.log")
ROOT_FILES: Tuple[str, ...] = ("RESULT", "driver_stderr.log", "driver_stdout.log", "evidence.json", "lane_roster.json",
                               "ledger.log", "protocol_check.json", "resolved_sampler.json", "served_geometry.json",
                               "usage_ledger.jsonl")
CELL_FILES: Tuple[str, ...] = ("evidence.json", "gen_stats.json", "manifest.json", "rows.jsonl")
WATCH_DIR = "watch"


def count_inodes(path: str | Path) -> int:
    """``find <path> | wc -l``: the path itself + every directory and file below (symlinks are
    entries, never followed); a missing path counts 0."""
    root = Path(path)
    if not root.exists() and not root.is_symlink():
        return 0
    if not root.is_dir() or root.is_symlink():
        return 1
    n = 1
    for _d, dirs, files in os.walk(root, followlinks=False):
        n += len(dirs) + len(files)
    return n


def inode_census(path: str | Path) -> Dict[str, int]:
    """``{dirs, files, total}`` — dirs INCLUDES the root itself (the way ``find -type d`` counts)."""
    root = Path(path)
    dirs = files = 0
    if root.is_dir():
        dirs = 1
        for _d, ds, fs in os.walk(root, followlinks=False):
            dirs += len(ds)
            files += len(fs)
    elif root.exists():
        files = 1
    return {"dirs": dirs, "files": files, "total": dirs + files}


def relative_layout(root: str | Path) -> List[str]:
    """Every entry under ``root`` as a POSIX relative path, directories with a trailing ``/``, sorted."""
    root = Path(root)
    out: List[str] = []
    for d, dirs, files in os.walk(root, followlinks=False):
        base = Path(d).relative_to(root).as_posix()
        for x in dirs:
            out.append((f"{base}/{x}" if base != "." else x) + "/")
        for x in files:
            out.append(f"{base}/{x}" if base != "." else x)
    return sorted(out)


_EVAL_RE = re.compile(r"[^/]+\.eval$")
_KEY_RE = re.compile(r"^cells/[^/]+")
_FCID_RE = re.compile(r"(sandboxes|attempt|sweep)_[^/]+?(\.jsonl|\.json)$")
_LOGDIR_RE = re.compile(r"private/inspect_logs/([^/]+)")


def normalised_layout(root: str | Path) -> List[str]:
    """The layout with the per-store variables masked — the ``cells/<key>`` dir -> ``cells/<KEY>``, a
    ``.eval`` file -> ``<LOG>.eval``, ``sandboxes_<fcid>`` / ``attempt_<fcid>`` / ``sweep_<fcid>`` ->
    ``<FCID>`` — so two stores of the same shape compare equal whatever their hashes and ids."""
    out = []
    for rel in relative_layout(root):
        r = _KEY_RE.sub("cells/<KEY>", rel)
        r = _EVAL_RE.sub("<LOG>.eval", r)
        r = _FCID_RE.sub(lambda m: f"{m.group(1)}_<FCID>{m.group(2)}", r)
        out.append(r)
    return sorted(out)


def _outcomes_and_summary(task: str, n: int, tree_sha: str) -> Tuple[dict, dict]:
    """A valid v2 outcome record + the summary it must agree with (``completion.verify_outcomes_against_summary``)."""
    items = [{"item_id": str(i), "sample_idx": 0, "attempt": 1, "scored": True, "value": 1.0, "unparseable": False,
              "stop_reason": "stop", "at_cap": False, "think_block_present": False, "think_block_unterminated": False,
              "stray_open_tag": False, "content_empty_with_reasoning": False, "prompt_tokens": 10,
              "completion_tokens": 5, "seed": 1, "sampling_ref": "abcdef012345"} for i in range(n)]
    assert set(items[0]) == set(C.TASK_OUTCOME_ITEM_FIELDS_V2)
    header = {"task": task, "item_set": "full", "thinking": "off", "think_split_mode": "record_only",
              "unterminated_policy": "unscored", "max_tokens": 4096, "samples_per_item": 1,
              "n_attempted_items": n, "n_attempted_samples": n, "n_scored": n, "n_unscored": 0,
              "n_scored_samples": n, "n_unscored_samples": 0, "n_stop_reason": n, "n_at_cap": 0,
              "n_think_block_present": 0, "n_think_block_unterminated": 0, "n_unterminated_not_at_cap": 0,
              "n_completion_tokens": 5 * n, "producer_tree_sha": tree_sha, "items": items}
    assert set(header) == set(C.TASK_OUTCOMES_HEADER_FIELDS)
    summary = {"acc": 1.0, C.METRIC_N: n, "n_requests": n, "n_rejected": 0, "framework": "inspect_evals-0.18.0",
               "engine": "vllm", "route": "petri__base", "wall_clock_s": 1.0,
               **{f: header[f] for f in C.TASK_OUTCOMES_SUMMARY_FIELDS}}
    return header, summary


def full_shape_store(store: str | Path, *, cell: str = "none", tasks: Sequence[str] = FULL_TASKS, n_items: int = 2,
                     fcid: str = "fc-01SYNTHETIC0000000000000", run_id: str = "r1-synthetic", tree_sha: str = synth.code_sha("a"),
                     standup: bool = True, solo_pointer: bool = False) -> Path:
    """Write the full-shape store at ``store`` and return the cell dir. ``standup=True`` adds the row
    engine's three standup files (40); ``standup=False`` is today's solo store (36); ``solo_pointer``
    adds the §1.10 pointer + sweep certificate and the per-attempt completion record a solo full store now
    writes (39 with ``standup=False``)."""
    store = Path(store)
    store.mkdir(parents=True, exist_ok=True)
    owner = scope.Owner(run_id, cell, fcid)
    key = f"{cell.replace('__', '--')}__controls__t0__a0__2ff9cdff__a03d0582"
    cell_dir = store / "cells" / key
    cell_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"cell": cell, "intervention_hash": "2ff9cdff091aef6f", "harness_config_hash": "a03d0582",
                C.MANIFEST_KEY_PRODUCER: {"tree_sha": tree_sha}, "tinycap": {"thinking_by_task": {t: "off" for t in tasks}}}
    (cell_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    evidence = {"driver": "served_capability_run_v2", "cell_id": cell, "virtual_model": "petri__base", "engine": "vllm"}
    (cell_dir / "evidence.json").write_text(json.dumps(evidence, indent=1))
    gen_stats: Dict[str, dict] = {}
    with (cell_dir / "rows.jsonl").open("w", encoding="utf-8") as fh:
        for task in tasks:
            for i in range(n_items):
                fh.write(json.dumps({"arm_name": cell, "task": task, "item_id": str(i), "attempt": 0}) + "\n")
    for task in tasks:
        outcomes, summary = _outcomes_and_summary(task, n_items, tree_sha)
        rec = completion.finish_task(cell_dir, task=task, n_items=n_items, n_expected_full=n_items, limits={},
                                     n_rows=n_items, n_requests=n_items, n_rejected=0, wall_clock_s=1.0,
                                     summary=summary, outcomes=outcomes, n_planned=n_items)
        assert rec["complete"], rec
        log_dir = C.inspect_log_dir(cell_dir, task, 0)
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / f"2026-01-01T00-00-00-00-00_{task.split('/')[-1]}_SYNTHETIC.eval").write_bytes(b"PK\x03\x04synthetic")
        gen_stats[task] = {"task": task, "n": n_items, "acc": 1.0}
    (cell_dir / "gen_stats.json").write_text(json.dumps(gen_stats, indent=1))
    # the store root: the driver's per-run files by name
    (store / "RESULT").write_text(f"tinycap-modal name=synthetic-{cell} rc=0 result=COMPLETE status=OK sha={tree_sha} "
                                  f"at={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(0))}\n")
    (store / "driver_stdout.log").write_text("synthetic\n")
    (store / "driver_stderr.log").write_text("")
    (store / "evidence.json").write_text(json.dumps(evidence, indent=1))
    (store / "lane_roster.json").write_text(json.dumps({"cell_id": cell, "entry_id": cell, "wave": "synthetic"}, indent=1))
    (store / "ledger.log").write_text("ledger up\n")
    (store / "protocol_check.json").write_text(json.dumps({"ok": True}))
    (store / "resolved_sampler.json").write_text(json.dumps({"verdict": "AGREES"}))
    (store / "served_geometry.json").write_text(json.dumps({"geometry": {"max_model_len": 16384, "max_num_seqs": 31}}))
    (store / "usage_ledger.jsonl").write_text(json.dumps({"task": tasks[0], "sample": "0/0", "vmodel": "petri__base"}) + "\n")
    okey = store / cell
    okey.mkdir(exist_ok=True)
    (okey / "controls_summary.json").write_text(json.dumps({"cell_id": cell, "controls": {t: {"n": n_items} for t in tasks}}))
    # the provider's durable per-attempt record, through its writer
    scope.append_event(store / WATCH_DIR / f"sandboxes_{fcid}.jsonl", scope.EVENT_CREATED, "sb-synthetic-1", owner,
                       app=C.BCB_SANDBOX_APP_NAME, name="inspect-bigcodebench-0-synthetic")
    if solo_pointer:
        (store / WATCH_DIR / f"attempt_{fcid}.json").write_text(json.dumps(
            {"run_id": run_id, "cell": cell, "function_call_id": fcid, "name": f"synthetic-{cell}", "model": "gemma4e4b-f22s1",
             "mode": "off", "out_dir": str(store), "controls": ["bigcodebench_full", "bfcl_full"], "tree_sha": tree_sha,
             "sandbox_record": str(store / WATCH_DIR / f"sandboxes_{fcid}.jsonl"),
             "sandbox_app_name": C.BCB_SANDBOX_APP_NAME, "started_at": 1.0}, indent=1, sort_keys=True))
        cert = {**scope.failed_sweep_record(owner, by="container_finally", error=RuntimeError("x")),
                "verified_empty": True, "error": None, "certificate": "sweep", "sandbox_app_id": "ap-synthetic"}
        (store / WATCH_DIR / f"sweep_{fcid}.json").write_text(json.dumps(cert, indent=1, sort_keys=True, default=str))
        # the solo drive's per-attempt completion record (tinycap_served.COMPLETION_FIELDS): what this attempt finished, in which store
        (store / WATCH_DIR / f"completion_{fcid}.json").write_text(json.dumps(
            {"run_id": run_id, "cell": cell, "function_call_id": fcid, "name": f"synthetic-{cell}", "model": "gemma4e4b-f22s1",
             "mode": "off", "out_dir": str(store), "controls": ["bigcodebench_full", "bfcl_full"],
             "remaining": ["bigcodebench_full", "bfcl_full"], "completed": {"bigcodebench_full": key, "bfcl_full": key},
             "status": "OK", "sandboxed": True, "tree_sha": tree_sha, "finished_at": 2.0}, indent=1, sort_keys=True))
    if standup:
        sd = store / STANDUP_DIR
        sd.mkdir(exist_ok=True)
        (sd / "endpoint.json").write_text(json.dumps({"standup_attempt_id": "sa-synthetic", "cells": [cell],
                                                       "lora_config": {"max_loras": 2}, "residual_slot": {"armed": False}}))
        (sd / "serving_attestation.json").write_text(json.dumps({"ok": True}))
        (sd / "vllm_server.log").write_text("INFO Using cache directory: /tmp/compile_cache/vllm/torch_compile_cache/abc\n")
    return cell_dir


def assert_store_inodes(store: str | Path, expect: int = ROW_ENGINE_STORE_INODES) -> Dict[str, int]:
    """The per-store assertion both suites make: EXACTLY ``expect`` inodes (== and, since the spec's
    bound is <= 40, never more); returns the census so a failure names dirs and files."""
    census = inode_census(store)
    assert census["total"] == count_inodes(store), census
    assert census["total"] == expect, f"{store}: {census} != {expect} — layout: {relative_layout(store)}"
    assert census["total"] <= ROW_ENGINE_STORE_INODES, census
    return census
