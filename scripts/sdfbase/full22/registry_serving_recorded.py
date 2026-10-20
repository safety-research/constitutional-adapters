#!/usr/bin/env python
"""full22: the served WINDOW was OBSERVED in the registry's spool — the post-standup
gate that pairs with ``registry_runs_file.py``: for every run of ``<standup dir>/registry_runs.json`` (sisteer.registry.hooks.RUNS_FILE) a
``run_patch`` from a SERVING writer sits in ``<spool>/<campaign>/<run dir>/NNNNNN-run_patch.jsonl`` carrying THIS standup's attested facts.

  python scripts/sdfbase/full22/registry_serving_recorded.py --out <standup dir> [--timeout-s 120] [--spool <dir>]

WHY A GATE. The runs file is the launcher's half of the contract; the standup's hook (native_standup.write_endpoint ->
hooks.emit_serving_patches, or mt_serving_up's lens writer) is the other half, and the hook is GUARDED by contract — a failed emission is one
stderr line, never a refused standup. Every demo BoN row carried context_window_tokens NULL because the launcher's half was
missing; a job that completes with the same NULL for any other reason (no spool, an unimportable hook, a hook that read another file) would
be indistinguishable from it in the registry. So the launcher reads the spool back: the SAME destination the harvester pulls
(hooks.resolve_spool: $SISTEER_REGISTRY_SPOOL, the shared root's registry/records), the way f22_registry_spool_preflight pins it.

WHAT COUNTS AS OBSERVED, per run: a ``run_patch`` record whose ``payload.run_id`` is the run, whose ``producer.script`` is a serving writer
(SERVING_WRITERS), whose ``fields.context_window_tokens`` equals the endpoint's ``attested_max_model_len`` (``max_model_len`` when unattested)
and whose ``fields.extra_json`` names THIS standup — ``served_key`` the run's served key, ``proxy_url`` the endpoint's and, when the endpoint
carries a ``slurm_job_id``, the same job — so an earlier attempt's patch (another port, another job) never satisfies the current standup.

The hook writes a few milliseconds after ``endpoint.json`` lands and the shell's poll wakes on the file, so the gate POLLS up to
``--timeout-s`` for the record (the wait is for the hook's write, not a guess). Exit 0 with one summary line once every run is observed;
3 when any run has none within the timeout (the missing run ids, and the ``[registry]`` lines of ``<out>/standup.log`` — the hook's own
account); 2 = a usage / shape problem (no runs file, no endpoint record, no spool, a run entry without run_id).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

RUNS_FILE = "registry_runs.json"   # == sisteer.registry.hooks.RUNS_FILE (asserted in main)
TAG = "[registry]"
#: the producer scripts whose run_patch records are a standup's SERVING facts: the native core's write_endpoint hook
#: (hooks.emit_serving_patches' default `script`) and mt_serving_up's lens writer (mt_serving_up.LENS_SERVING_FACTS_SCRIPT)
SERVING_WRITERS: Tuple[str, ...] = ("capeval.serving.native_standup.write_endpoint", "sdfbase.mt_serving_up.lens_endpoint")
#: the endpoint record read: the live one while the standup is up, else the durable copy f22_standup keeps (mt_serving_up unlinks
#: endpoint.json at teardown)
ENDPOINT_FILES = ("endpoint.json", "endpoint.recorded.json")


class RecordedRefused(Exception):
    pass


def expected_window(endpoint: Mapping[str, object]) -> int:
    """The window the patch must carry: what the workers ATTESTED (vLLM's live scheduler config), the resolved one when unattested — the
    same preference hooks.serving_patch_payload applies."""
    w = endpoint.get("attested_max_model_len", endpoint.get("max_model_len"))
    if w is None:
        raise RecordedRefused("the endpoint record carries neither attested_max_model_len nor max_model_len")
    return int(w)


def run_entries(runs: Mapping[str, object]) -> List[Tuple[str, str, Optional[str]]]:
    """``[(served key, run_id, campaign), ...]`` over the runs file's one-run and LIST forms (hooks.load_runs_file)."""
    out: List[Tuple[str, str, Optional[str]]] = []
    for key, entry in runs.items():
        entries = list(entry) if isinstance(entry, (list, tuple)) else [entry]
        for run in entries:
            if not isinstance(run, Mapping) or not run.get("run_id"):
                raise RecordedRefused(f"run entry for served key {key!r} has no run_id (the runs file is not registry_runs_file.py's shape)")
            campaign = run.get("campaign")
            out.append((str(key), str(run["run_id"]), str(campaign) if campaign else None))
    if not out:
        raise RecordedRefused("the runs file names no run")
    return out


def _patch_matches(rec: Mapping[str, object], *, run_id: str, served_key: str, window: int, proxy_url: Optional[str],
                   slurm_job_id: Optional[str], writers: Sequence[str]) -> bool:
    if rec.get("record_type") != "run_patch":
        return False
    payload = rec.get("payload") if isinstance(rec.get("payload"), Mapping) else {}
    if str(payload.get("run_id")) != run_id:
        return False
    producer = rec.get("producer") if isinstance(rec.get("producer"), Mapping) else {}
    if producer.get("script") not in writers:
        return False
    fields = payload.get("fields") if isinstance(payload.get("fields"), Mapping) else {}
    if fields.get("context_window_tokens") != window:
        return False
    extra = fields.get("extra_json") if isinstance(fields.get("extra_json"), Mapping) else {}
    if extra.get("served_key") != served_key:
        return False
    if proxy_url is not None and extra.get("proxy_url") != proxy_url:
        return False
    if slurm_job_id is not None and str(extra.get("slurm_job_id")) != str(slurm_job_id):
        return False
    return True


def recorded_patches(runs: Mapping[str, object], endpoint: Mapping[str, object], spool: str, *,
                     writers: Sequence[str] = SERVING_WRITERS) -> Tuple[Dict[str, str], List[str]]:
    """``(found, missing)``: ``found[run_id]`` = the record file holding THIS standup's serving patch for the run (the first match in
    directory order), ``missing`` = the run ids with none in ``spool`` right now. Pure: reads the spool once, never writes."""
    from sisteer.registry import identity
    from sisteer.registry import records as R
    window = expected_window(endpoint)
    proxy_url = endpoint.get("proxy_url")
    job = endpoint.get("slurm_job_id")
    found: Dict[str, str] = {}
    missing: List[str] = []
    for served_key, run_id, campaign in run_entries(runs):
        if not campaign:
            campaign = identity.parse_run_id(run_id)["campaign"]
        run_dir = os.path.join(spool, campaign, identity.run_dir_name(run_id))
        hit: Optional[str] = None
        if os.path.isdir(run_dir):
            for fn in sorted(os.listdir(run_dir)):
                if "-run_patch.jsonl" not in fn:
                    continue
                path = os.path.join(run_dir, fn)
                try:
                    recs = list(R.read_record_file(path, compute_sha=False))
                except (OSError, ValueError):
                    continue      # a torso another writer is still producing, or a file that is not a record file: not evidence either way
                if any(_patch_matches(rec, run_id=run_id, served_key=served_key, window=window, proxy_url=proxy_url,
                                      slurm_job_id=job, writers=writers) for _, rec, _ in recs):
                    hit = path
                    break
        if hit is None:
            missing.append(run_id)
        else:
            found[run_id] = hit
    return found, missing


def read_endpoint(out: Path) -> Mapping[str, object]:
    for name in ENDPOINT_FILES:
        p = out / name
        if p.is_file():
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError) as e:
                raise RecordedRefused(f"cannot read {p} ({type(e).__name__}: {e})") from None
            if not isinstance(rec, Mapping):
                raise RecordedRefused(f"{p} is not a JSON object")
            return rec
    raise RecordedRefused(f"no endpoint record under {out} ({' / '.join(ENDPOINT_FILES)}) — the standup never reached READY here")


def hook_lines(out: Path, limit: int = 20) -> List[str]:
    """The hook's own account: the ``[registry]`` lines of ``<out>/standup.log`` (mt_serving_up's stdout + stderr), last ``limit``."""
    log = out / "standup.log"
    if not log.is_file():
        return []
    try:
        lines = [l.rstrip("\n") for l in log.read_text(encoding="utf-8", errors="replace").splitlines() if TAG in l]
    except OSError:
        return []
    return lines[-limit:]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--out", required=True, help="the standup's out dir: registry_runs.json + endpoint.json (or endpoint.recorded.json) + standup.log")
    ap.add_argument("--timeout-s", type=float, default=120.0, help="how long to wait for the hook's record (it follows endpoint.json by milliseconds)")
    ap.add_argument("--spool", default=None, help="the spool to read (default: hooks.resolve_spool — $SISTEER_REGISTRY_SPOOL, the shared root's registry/records)")
    ap.add_argument("--poll-s", type=float, default=2.0)
    args = ap.parse_args(argv)
    from sisteer.registry import hooks as H
    assert H.RUNS_FILE == RUNS_FILE, (H.RUNS_FILE, RUNS_FILE)
    out = Path(args.out)
    try:
        runs_path = out / RUNS_FILE
        if not runs_path.is_file():
            raise RecordedRefused(f"no {runs_path} — registry_runs_file.py did not run for this standup (nothing to observe)")
        try:
            runs = json.loads(runs_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise RecordedRefused(f"cannot read {runs_path} ({type(e).__name__}: {e})") from None
        if not isinstance(runs, Mapping):
            raise RecordedRefused(f"{runs_path} is not a JSON object keyed by served key")
        endpoint = read_endpoint(out)
        spool = args.spool or H.resolve_spool()
        if not spool:
            raise RecordedRefused("no registry spool resolves (SISTEER_REGISTRY_SPOOL unset and no provisioned shared root) — the hook had nowhere to write")
        window = expected_window(endpoint)
        entries = run_entries(runs)
        deadline = time.monotonic() + max(0.0, float(args.timeout_s))
        while True:
            found, missing = recorded_patches(runs, endpoint, str(spool))
            if not missing or time.monotonic() >= deadline:
                break
            time.sleep(max(0.1, float(args.poll_s)))
    except RecordedRefused as e:
        print(f"{TAG} served window NOT observable: {e}", file=sys.stderr, flush=True)
        return 2
    if missing:
        print(f"{TAG} served window NOT observed in {spool} for {len(missing)} of {len(entries)} run(s) of {out / RUNS_FILE} "
              f"(expected context_window_tokens {window} from proxy {endpoint.get('proxy_url')}, job {endpoint.get('slurm_job_id')}, "
              f"writers {list(SERVING_WRITERS)}) after {args.timeout_s:g}s: {', '.join(missing)}", file=sys.stderr, flush=True)
        for line in hook_lines(out):
            print(f"{TAG}   standup.log: {line}", file=sys.stderr, flush=True)
        return 3
    print(f"{TAG} served window OBSERVED: context_window_tokens {window} on {len(found)} run(s) of {len(entries)} in {spool} "
          f"({', '.join(sorted(os.path.basename(os.path.dirname(p)) for p in found.values()))})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
