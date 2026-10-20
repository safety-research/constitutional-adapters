#!/usr/bin/env python
"""The CPU verify probe: decide the CPU verify MECHANISM.

Over a scores-stripped COPY of a mirrored store's BigCodeBench ``.eval`` (``n`` samples, default
10) it runs the three candidates of ``contracts.CPU_VERIFY_MECHANISMS`` in order — M1
``score_async(copy, [verify()])`` (predicted to raise: the stock scorer calls ``sandbox().exec``
and a re-score has no sandbox), M1b ``score_async`` with every sample yielded from inside
``sandboxenv_context`` (internal API), M2 the replay VEHICLE (``eval_async`` of the coupled task
with ``model=replay/<name>``, public API only) — and judges each on (a) the ``n`` verdicts
reproduced against the store's own scores and (b) the binding test of §2.2.3 passing on its
output (``cpu_verify.binding_problems``). The choice is the first of M1, M1b, M2 that passes
both; when M1b and M2 both pass, M2 (public API) unless M1b was measurably simpler to bind —
here "simpler" is fewer binding problems, so equal problem counts choose M2. The verdict and
every mechanism's facts are written as ``logs/cpu_verify_probe.json`` (``--out``).

    # on Modal — a CPU function on the client image, the volume mounted, a sandbox owner in the env
    ~/venvs/modal/bin/python -m modal run scripts/modal/tinycap_cpu_verify_probe.py::probe \\
        --store /vol/tinycap/gen/<store root> --n 10 --out logs/cpu_verify_probe.json

    # locally — the SAME code without Modal (what the tests run), over any store directory
    PYTHONPATH=src ~/venvs/tinycap-laneB/bin/python scripts/modal/tinycap_cpu_verify_probe.py \\
        --local <store root> --n 10 [--out logs/cpu_verify_probe.json] [--sandbox local]

``--sandbox local`` (local mode only) swaps the coupled task's sandbox for inspect_ai's ``local``
provider, so the stock scorer's ``python -c`` runs on this machine — the mechanics of every
mechanism are exercised without Modal; a verdict that depends on the pinned execution image (a
missing library) then reads as a mismatch, which the JSON names item by item. The Modal path is
NOT run by this landing (landing 2's GPU canaries are probably not admitted; the probe is
built to the plan and proven on the synthetic fixture).

The store is read only: the stripped copy lives in a temp dir; nothing under the store moves.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from sisteer.capeval.tinycap import contracts as C  # noqa: E402
from sisteer.capeval.tinycap import cpu_verify as CV  # noqa: E402

PROBE_SCHEMA = "cpu_verify_probe/1"
TASK = C.inspect_task(C.CONTROL_BIGCODEBENCH_FULL)
NAME = C.CONTROL_BIGCODEBENCH_FULL
MODAL_APP_NAME = "tinycap-cpu-verify-probe"


def find_store_log(store_root: str | Path, task: str = TASK) -> Path:
    """The ONE scored ``.eval`` of ``task`` under ``<store>/cells/<key>/private/inspect_logs/<task>__s0/``."""
    try:
        cell_dir = CV.one_cell_dir(store_root)
    except CV.CpuVerifyError as e:
        raise SystemExit(str(e)) from None
    log_dir = C.inspect_log_dir(cell_dir, task, 0)
    logs = sorted(q for q in log_dir.glob("*.eval") if q.is_file())
    if len(logs) != 1:
        raise SystemExit(f"{log_dir}: expected exactly one .eval, found {[q.name for q in logs]}")
    return logs[0]


def stripped_head(log: Any, n: int) -> Any:
    """A scores-stripped COPY of the first ``n`` samples of ``log`` (``cpu_verify.strip_scores`` over a
    head) — the unscored generation log the mechanisms score."""
    if n < 1 or n > len(log.samples or []):
        raise SystemExit(f"--n {n} is not in [1, {len(log.samples or [])}]")
    head = CV.strip_scores(log)
    head.samples = list(head.samples[:n])
    if head.results is not None:
        head.results.total_samples = n
        head.results.completed_samples = n
    return head


def reference_verdicts(log: Any, n: int, *, scorer: str = "verify") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for s in list(log.samples)[:n]:
        sc = (s.scores or {}).get(scorer)
        if sc is None:
            raise SystemExit(f"sample {CV.sample_key(s)} carries no {scorer!r} score — not a scored store log")
        out[str(CV.sample_key(s))] = sc.value
    return out


def verdicts_of(vehicle: Any, *, scorer: str = "verify") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for s in list(vehicle.samples or []):
        sc = (s.scores or {}).get(scorer)
        out[str(CV.sample_key(s))] = sc.value if sc else None
    return out


def write_record(path: str | Path, record: Mapping[str, Any]) -> Path:
    """The probe record as JSON (sorted keys, one trailing newline) at ``path``, parents made."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(record, indent=1, sort_keys=True, default=str) + "\n")
    return p


def default_task_factory(name: str, *, log_ids: Sequence[str], repo_root: Path, sandbox: Optional[str]) -> Any:
    """The coupled thin task over ``log_ids`` (``cpu_verify.build_coupled_bcb_task``); ``sandbox`` overrides
    the task's sandbox spec (``local`` for a Modal-free run)."""
    task = CV.build_coupled_bcb_task(name, log_ids=log_ids, repo_root=repo_root)
    if sandbox is not None:
        from inspect_ai.util import SandboxEnvironmentSpec

        task.sandbox = SandboxEnvironmentSpec(sandbox)
    return task


def probe_store(store_root: str | Path, *, n: int = 10, out_path: Optional[str | Path] = None,
                repo_root: Path = REPO, sandbox: Optional[str] = None, max_sandboxes: Optional[int] = None,
                task_factory: Optional[Callable[..., Any]] = None,
                mechanisms: Sequence[str] = C.CPU_VERIFY_MECHANISMS, log: Callable[[str], None] = print
                ) -> Dict[str, Any]:
    """Run the probe over ``store_root`` (read only) and return / write the record."""
    from inspect_ai.log import read_eval_log

    t_start = time.time()
    log_path = find_store_log(store_root)
    full = read_eval_log(str(log_path))
    head = stripped_head(full, n)
    ids = [str(s.id) for s in head.samples]
    reference = reference_verdicts(full, n)
    factory = task_factory if task_factory is not None else default_task_factory
    coupled_task = factory(NAME, log_ids=ids, repo_root=Path(repo_root), sandbox=sandbox)
    n_sandboxes = int(max_sandboxes) if max_sandboxes is not None else CV.default_max_sandboxes(n)
    results: List[Dict[str, Any]] = []
    tmp_root = Path(tempfile.mkdtemp(prefix="tinycap-cpu-verify-probe-"))
    for mech in mechanisms:
        rec: Dict[str, Any] = {"mechanism": mech, "ok": False, "error": None, "elapsed_s": None,
                               "verdicts_reproduced": None, "n_mismatched": None, "mismatched": [],
                               "binding_ok": None, "binding_problems": [], "vehicle_log_sha256": None}
        t0 = time.time()
        try:
            vehicle, vpath = asyncio.run(CV.run_mechanism(mech, original=head, coupled_task=coupled_task,
                                                          max_sandboxes=n_sandboxes, log_dir=tmp_root / mech))
            got = verdicts_of(vehicle)
            mismatched = sorted(k for k in reference if got.get(k) != reference[k])
            problems = CV.binding_problems(head, vehicle, primary_scorer="verify")
            rec.update(ok=True, verdicts_reproduced=not mismatched, n_mismatched=len(mismatched),
                       mismatched=[{"sample": k, "expected": reference[k], "got": got.get(k)} for k in mismatched],
                       binding_ok=not problems, binding_problems=problems[:20],
                       vehicle_log_sha256=(CV.sha256_file(vpath) if Path(vpath).is_file() else None))
        except BaseException as e:  # noqa: BLE001 — the probe RECORDS every failure; M1 is expected to fail
            if isinstance(e, KeyboardInterrupt):
                raise
            rec.update(ok=False, error=f"{type(e).__name__}: {str(e)[:400]}",
                       traceback_tail=traceback.format_exc()[-1500:])
        rec["elapsed_s"] = round(time.time() - t0, 3)
        results.append(rec)
        log(f"[cpu-verify-probe] {mech}: ok={rec['ok']} verdicts_reproduced={rec['verdicts_reproduced']} "
            f"binding_ok={rec['binding_ok']} elapsed={rec['elapsed_s']} s"
            + (f" error={rec['error']}" if rec["error"] else ""))
    passing = [r for r in results if r["ok"] and r["verdicts_reproduced"] and r["binding_ok"]]
    chosen = None
    reason = "no mechanism reproduced the verdicts AND bound"
    if passing:
        names = [r["mechanism"] for r in passing]
        if "M1_score_async" in names:
            chosen, reason = "M1_score_async", "first of M1, M1b, M2 to pass both tests"
        elif "M1b_score_async_sandboxenv" in names and "M2_replay_vehicle" in names:
            m1b = next(r for r in passing if r["mechanism"] == "M1b_score_async_sandboxenv")
            m2 = next(r for r in passing if r["mechanism"] == "M2_replay_vehicle")
            if len(m1b["binding_problems"]) < len(m2["binding_problems"]):
                chosen, reason = "M1b_score_async_sandboxenv", "both pass; M1b measurably simpler to bind"
            else:
                chosen, reason = "M2_replay_vehicle", "both pass; M2 (public API only) unless M1b is simpler to bind"
        else:
            chosen, reason = names[0], "the only mechanism to pass both tests"
    record = {"schema": PROBE_SCHEMA, "store": str(store_root), "log": str(log_path), "n": n, "ids": ids,
              "reference_verdicts": reference, "sandbox_override": sandbox, "max_sandboxes": n_sandboxes,
              "inspect_ai_version": CV.inspect_ai_version(), "mechanisms": results, "chosen": chosen, "reason": reason,
              "elapsed_s": round(time.time() - t_start, 3), "written_at": time.time()}
    if out_path is not None:
        log(f"[cpu-verify-probe] wrote {write_record(out_path, record)}")
    log(f"[cpu-verify-probe] chosen: {chosen} ({reason})")
    return record


# --------------------------------------------------------------------------- #
# the Modal wrapper — built to the plan, NOT run by this landing                #
# --------------------------------------------------------------------------- #


def client_image(modal_mod: Any, repo: str | Path) -> Any:
    """The probe's CPU image: debian_slim + the PINNED tinycap client lock (``contracts.CLIENT_LOCK_TXT`` — the
    hashed ``uv pip compile`` output the client venv is built from, inspect_ai 0.3.260 included). A tree without
    the lock REFUSES: a bare image has no inspect_ai, and a probe that silently ran on one would record its
    ImportErrors as the mechanisms' verdicts."""
    lock = Path(repo) / C.CLIENT_LOCK_TXT
    if not lock.is_file():
        raise FileNotFoundError(f"the tinycap client lock {lock} (contracts.CLIENT_LOCK_TXT) is missing from this tree — "
                                "the probe's Modal image installs exactly the pinned client and never falls back to a "
                                "bare image")
    return modal_mod.Image.debian_slim(python_version="3.12").pip_install_from_requirements(str(lock))


try:  # the modal venv only; the local path and the tests never import it
    import modal  # type: ignore

    if not (hasattr(modal, "App") and hasattr(modal, "Volume") and hasattr(modal, "Image")):
        raise ImportError("not the Modal SDK (a stub in sys.modules)")
    _app = modal.App(MODAL_APP_NAME)
    _vol = modal.Volume.from_name(C.MODAL_VOLUME, create_if_missing=False)
    _image = client_image(modal, REPO)

    @_app.function(image=_image, volumes={"/vol": _vol}, cpu=2, memory=8192, timeout=3 * 3600)
    def probe_remote(store: str, n: int = 10, max_sandboxes: Optional[int] = None) -> Dict[str, Any]:
        """The probe in a CPU container: the client image, the volume at /vol, the sandbox owner
        (``TINYCAP_OWNER_*``) expected in the function's environment (the launcher's convention)."""
        return probe_store(store, n=n, out_path=None, max_sandboxes=max_sandboxes)

    @_app.local_entrypoint()
    def probe(store: str, n: int = 10, out: str = "logs/cpu_verify_probe.json", max_sandboxes: Optional[int] = None):
        record = probe_remote.remote(store, n, max_sandboxes)
        write_record(out, record)
        print(f"[cpu-verify-probe] chosen: {record['chosen']} ({record['reason']}); wrote {out}")
except ImportError:  # no modal: the local path below is the whole script
    pass


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local", metavar="STORE", required=True,
                    help="run the probe HERE (no Modal) over this store root (its one cells/<key>/ store)")
    ap.add_argument("--n", type=int, default=10, help="samples (the head of the log; default 10)")
    ap.add_argument("--out", default="logs/cpu_verify_probe.json", help="where the record goes")
    ap.add_argument("--sandbox", default=None, help="override the coupled task's sandbox (e.g. `local`)")
    ap.add_argument("--max-sandboxes", dest="max_sandboxes", type=int, default=None)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    os.environ.setdefault("INSPECT_DISPLAY", "none")
    args = build_parser().parse_args(argv)
    record = probe_store(args.local, n=args.n, out_path=args.out, sandbox=args.sandbox, max_sandboxes=args.max_sandboxes)
    return 0 if record["chosen"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
