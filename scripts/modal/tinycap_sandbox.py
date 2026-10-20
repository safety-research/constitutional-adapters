"""EvalPlus EXECUTION for the served HumanEval+/MBPP+ cells — the trusted
CPU orchestrator that runs generated code inside an ephemeral
``modal.Sandbox``.

Generated code never runs in the GPU container and never sees the volume.
Per pending ``(cell, thinking mode, dataset[, :open])`` the orchestrator:

1. reads that task's ``samples.jsonl`` from the volume and re-validates it
   (exactly the pinned id set, one sample each — evalplus 0.3.1 refuses a
   samples file missing any problem);
2. creates ONE ``modal.Sandbox`` on the repo's digest-pinned execution image
   ``contracts.SANDBOX_IMAGE_DIGEST`` (the same ``ganler/evalplus`` digest
   the cluster path's ``capeval_sandbox.sh`` runs — same results class)
   with exactly ``contracts.SANDBOX_CREATE_KWARGS_REQUIRED`` (``block_network
   =True, cpu=8, memory=16384 MiB, timeout=3600 s``), NO volumes, NO
   secrets, NO network file systems (``SANDBOX_CREATE_KWARGS_FORBIDDEN``),
   and no entrypoint — the sandbox idles until driven;
3. transfers IN through ``sb.open(path, "wb")`` — the trusted channel —
   exactly two files: ``samples.jsonl`` and the pinned dataset archive
   (``configs/capeval/evalplus/datasets.json`` names it; the network-blocked
   sandbox cannot download it, so the evalplus data loader is pointed at
   the uploaded copy via its own ``*_OVERRIDE_PATH`` environment variable,
   PC3);
4. runs ``contracts.sandbox_entrypoint(dataset)`` via ``sb.exec`` —
   ``bash -lc 'ulimit -u 512 -f 1048576 -c 0; cd /work && python -m
   evalplus.evaluate --dataset <d> --samples /work/samples.jsonl --parallel
   8'`` — evalplus's own ``reliability_guard`` and per-test time limits
   remain in force inside; a non-zero exit or a timeout yields NO results
   for that task (loud, resumable), never partial rows;
5. transfers OUT ``samples_eval_results.json`` through ``sb.open(...,
   "rb")`` and VALIDATES it (``evalplus_runner.validate_eval_results``:
   the 0.3.1 shape, exactly the pinned id set, one item per task, statuses
   in pass/fail/timeout, and the dataset hash equal to the md5 of the
   archive WE uploaded) BEFORE anything touches the volume; only then
   writes ``eval_results.json`` + ``execution.json`` next to the samples
   (the execution records ``samples_sha256``, the sha256 of the exact
   ``samples.jsonl`` bytes it evaluated — E50 #4),
   converts the rows into the cell's ``rows.jsonl`` from the manifest's
   ``row_constants`` (``convert_eval_results_to_store``: rows for the
   GENERATED items only, ``done/<task>.json`` on a complete unpadded set,
   ``partial/<task>.json`` on a limited one) and commits the volume;
6. ``sb.terminate()`` in a ``finally`` — a sandbox never outlives its task.

Independent executions run concurrently (a thread per sandbox, ``--max-
parallel``); each sandbox has its own 8 CPUs.

PENDING / RESUME (R7). The done marker is AUTHORITATIVE: a task whose
``done/<task>.json`` (or ``partial/<task>.json`` — a limited store's
terminal state) exists is finished, whatever else is on disk. Without a
marker, a task whose ``eval_results.json`` VALIDATES (shape, exact id set,
and the dataset hash recorded by ``execution.json`` when present) resumes
at stage ``convert`` — conversion only, never a second sandbox; a task
whose results file is absent, ``{}``, or otherwise invalid is stage
``execute`` (the stale file is replaced atomically). ``pending_executions``
names the stage per job; the dry run prints it.

THE PRICE OF THAT AUTHORITY, AND THE RE-BIND MODE (E60). Conversion is gated
on the ABSENCE of the done marker and the durable binding is WRITTEN BY the
conversion, so any field the conversion learns to write is unreachable for
every already-converted task the moment it lands — the marker that makes
resume safe is the same marker that locks the store. ``--rebind-samples-
digest`` is the designed, additive way back in for ONE field
(``samples_sha256``, E55 #2): per task it refuses unless ``execution.json``
and the ``samples.jsonl`` bytes both name the same digest, computes the
would-be files by running the REAL converter against a scratch COPY of the
store, and refuses before writing anything if the copy differs from the store
in any byte other than that digest. The assertion has two halves: every file
OUTSIDE the three re-bind targets by sha256 (``rows.jsonl`` included), and the
three targets themselves by CANONICAL-BYTES equality once exactly the keys the
mode may add are removed — never a parsed-dict comparison, which holds
``1 == 1.0`` and would pass a number that changed type (E63). ``finished_at``
is preserved and ``rebound_at`` added — a re-bind did not re-finish the task.
A marker that ALREADY carries the digest is a ``no-op`` only when the durable
summary outside the task directory carries the same one and every witness still
inside it agrees; the torn shapes in between refuse rather than being passed
over.
The reachable witness ceiling for the current population is 5 of 6:
``gen_meta.json``'s digest is stamped by the GENERATION writer and no
conversion path writes that file. THE NEXT CONVERSION-TIME FIELD NEEDS ITS
RE-BIND STORY DESIGNED WITH IT; this mode is deliberately not generalised.

BOUND TO THE SAMPLE BYTES (E50 #4). Everything that resume checked — task
ids, count, shape, the dataset archive md5 — is invariant under regenerating
the same problems with DIFFERENT solutions, which is precisely what a
crashed-and-rerun generation leaves on disk. So ``execution.json`` records
``samples_sha256``, the digest of the exact ``samples.jsonl`` the sandbox
evaluated, and every convert-resume compares it against the file now beside
the results (:func:`samples_binding`): equal converts; DIFFERENT refuses,
naming both digests; and nothing recorded (a pre-digest ``execution.json``,
or none) refuses too — an execution that cannot vouch for its samples is not
a weaker yes. Both refusals carry the same remedy (re-run the sandbox stage
for that task) and neither loses a number: the results are re-derivable and
the task stays listed as pending until the remedy is applied. The refusal is
per task, at stage ``refuse``, so one such task never stops the others.
``eval_results.json`` gets no provenance of its own: it is evalplus's own
output, whose top-level keys ``validate_eval_results`` fixes at ``{date,
hash, eval}``. The writer-side merge reads the same field back
(``cell_controls`` §3 / §9), where the asymmetry is deliberate: a MISMATCH
refuses there too, an unbound (pre-digest) store does NOT — at merge time
the numbers already exist, and refusing every store written before the field
would strand every summary on the volume without changing one of them.

THE MERGE (N1; E12 items 2 and 3). ``controls_summary.json`` is written by
ONE function, ``tinycap.completion.merge_controls_summary``, which needs
``sisteer.eval.cell_controls`` and hence torch. This orchestrator's image
is stdlib + pyyaml + the mounted ``sisteer`` ON PURPOSE — the trusted
process that reads untrusted sandbox output and writes the volume keeps
the smallest possible dependency surface — so the merge does NOT run in it.
It runs in the launcher's CLIENT venv (the driver's own environment, the
hashed lock) as the CPU-only ``merge_summaries`` function of
``tinycap_served.py``.

CRASH-SAFE (item 2). The merge's pending state is DERIVED FROM DISK on
every non-dry invocation, never from what this process converted —
``unmerged_completed_stores`` names every store holding a DONE
``evalplus/*`` marker that the cell's ``controls_summary.json`` does not
ACCOUNT FOR (a conversion that committed before the process died, a merge
that failed, a store converted by a run that predates the chained merge:
all found, idempotently). The done marker plus the unaccounted payload IS
the durable merge-pending record; nothing else needs writing. Accounted
for (PW8 / E16 A1) means: the entry's recorded ``sources`` block names this
store's EXACT source — cell key, harness hash, pass, producer sha
(``store_merge_source``) — as the standing source or in that source's
``superseded`` list or among its ``equal_witnesses`` (PW13) — the latter two
only while the record's ``payload_digest`` equals the digest of the store's
CURRENT ``summary/<task>.json`` (PW14: a superseded store whose summary
changed is a new claim; a record without a digest accounts for nothing), and
the STANDING source only while its current summary digests to the CANONICAL
payload, ``controls[task]`` (PW16 / E24: before this the four-field match
alone counted, so a standing store whose own summary was rewritten never read
pending and the merge overlaid it silently). The task NAME being in ``controls`` is not enough (a
pass-1 store merged made a completed pass-2 store of the same cell look
merged, so the pass rule was never applied to it), a source stamped
``UNRECORDED_MERGE_SOURCE`` satisfies no concrete store, and a summary
written before the source discipline accounts for nothing — such a store
stays listed, with ``why``, until a sourced merge lands (or forever, if its
producing tree predates the discipline; the listing is honest, and its
tree's re-merge is idempotent).

BOUND TO THE PRODUCING TREE (item 3): the pending stores are GROUPED by
``manifest.producer.tree_sha`` (the ``TREE_SHA`` the launcher stamps into
every uploaded tree and the driver records), and each group is merged by
``modal run scripts/modal/tinycap_served.py --sha <that sha> --merge
--merge-cell-dirs <its stores>`` — the launcher then REQUIRES the equality
again before merging. A store's numbers are never READ through another
tree's ``completion.py``. The producing tree only EMITS, though (PW10 / E17
A1): the launcher runs that tree's ``complete_task_summaries`` +
``slot_payloads`` into a scratch payload and its OWN (current) tree performs
the sourced canonical write — a pre-source-discipline tree's writer, run
against today's summary, overlaid a newer pass's value and erased every
source. A legacy store without the producer block is REFUSED (listed with
the remedy; there is no override — re-run its generation from a stamped
tree).

Decide the merge up front: ``--merge-mode merge`` chains one command per
producing tree; ``--merge-sha <sha>`` does the same but REFUSES any group
whose producer is not that sha (an audit of the operator's expectation);
``--merge-mode no-merge`` prints every command verbatim and records them
pending. Without one of the three the run refuses to execute — a completed
conversion can never fall silent. ONE string option for the mode (PW10):
``modal run`` registers every ``bool`` parameter as ``--<name>/--no-<name>``,
so the former ``merge: bool`` OWNED ``--no-merge`` and shadowed the former
``no_merge: bool`` parameter's own flag — ``modal run … --no-merge`` set
``merge=False`` and exited at the gate (operator C's finding).

The dataset archives are the orchestrator's OWN trusted input: ``--stage-
datasets`` downloads both release files into ``<VOL_CACHE_ROOT>/evalplus/``
and verifies their pinned sha256; execution refuses a missing or drifted
archive. A ``--dry-run`` lists the pending executions (with their stage)
and the merge-pending stores, and prints the exact ``Sandbox.create``
kwargs + exec argv the contract fixes, without creating a sandbox.
``--offline-contract`` does the same with no Modal call at all (the pure
kwargs builder; what the tests pin).

Run (from the repo root, ``PYTHONPATH=src``; the Modal CLI venv)::

  modal run scripts/modal/tinycap_sandbox.py --stage-datasets
  modal run scripts/modal/tinycap_sandbox.py --dry-run
  modal run scripts/modal/tinycap_sandbox.py --merge-mode merge     # execute + convert everything pending,
                                                                    # then merge per producing tree
  modal run scripts/modal/tinycap_sandbox.py --datasets mbpp --max-parallel 4 --merge-sha <sha>
  modal run scripts/modal/tinycap_sandbox.py --merge-mode no-merge  # execute + convert; prints the merge commands
  modal run scripts/modal/tinycap_sandbox.py --rebind-samples-digest --dry-run   # E60: the per-task diff, nothing written
  modal run scripts/modal/tinycap_sandbox.py --rebind-samples-digest             # ... then write it

Fallback if Sandboxes are unavailable to the account (P18): the repo's
``scripts/capeval_exec_wave.py`` + ``capeval_sandbox.sh`` GCP-VM docker path
from the login host, with the samples pushed there first — stated in the report.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import modal

from sisteer.capeval.tiers import evalplus_runner as er
from sisteer.capeval.tinycap import completion
from sisteer.capeval.tinycap import contracts as C

app = modal.App(C.MODAL_APP_NAME)
VOL = modal.Volume.from_name(C.MODAL_VOLUME, create_if_missing=False)
VOL_PATH = "/vol"
#: Where the orchestrator stages / expects the two release archives.
EVALPLUS_CACHE_DIR = f"{C.VOL_CACHE_ROOT}/evalplus"
#: The vendored pins file, mounted into the orchestrator image at a fixed path.
DATASETS_JSON_REMOTE = f"/root/{er.DATASETS_JSON_RELPATH}"
#: A pending job's stage (R7): a fresh sandbox execution, or conversion of
#: results that already validated on disk. ``refuse`` (E50 #4) is a job whose
#: results cannot be BOUND to the samples sitting beside them: carried as a
#: stage rather than dropped, so the dry run states it and a real run records
#: one failed job with the remedy instead of silently converting old results
#: against new samples (or silently re-executing over a state nobody has
#: looked at).
STAGE_EXECUTE = "execute"
STAGE_CONVERT = "convert"
STAGE_REFUSE = "refuse"
#: The sha256 of the EXACT ``samples.jsonl`` bytes a sandbox evaluated,
#: recorded in ``execution.json`` (E50 #4 — before it, a resume validated task
#: ids, count, shape and the dataset archive md5, none of which move when the
#: SOLUTIONS change, so results regenerated under the same ids could be
#: converted beside samples they never ran on). ``eval_results.json`` cannot
#: carry it: that file is evalplus 0.3.1's own output and
#: ``validate_eval_results`` fixes its top-level keys at ``{date, hash, eval}``
#: — a provenance block there would fail the seam validator and change bytes
#: the sandbox produced. The merge side reads the field back through
#: ``cell_controls.SAMPLES_DIGEST_FIELD``, pinned equal to this by test.
SAMPLES_DIGEST_FIELD = "samples_sha256"
#: :func:`samples_binding` verdicts: the recorded digest equals the samples on
#: disk; nothing records which samples produced these results (a pre-digest
#: ``execution.json``, or none at all); the recorded digest names OTHER bytes.
BINDING_BOUND = "bound"
BINDING_UNBOUND = "unbound"
BINDING_MISMATCH = "mismatch"
#: E60 (R22): the additive RE-BIND mode's flag and its four per-task verdicts.
#: ``rebind`` — the marker exists, carries no digest, and the assertion below
#: passed, so the three durable files gain it; ``no-op`` — the durable PAIR
#: outside the task directory (``summary/<task>.json`` and the marker) already
#: carries the same digest and every witness still inside the task directory
#: agrees with it — a witness that is GONE is passed over, one that is THERE
#: and unreadable never is;
#: ``refused`` — the two digest witnesses disagree or one is missing, the store
#: is TORN (a marker carrying a digest its summaries lack, three that disagree,
#: or a witness inside the task directory that is present but cannot be read or
#: states a digest that is not a sha256 — E63 / E68), or something OTHER than
#: the permitted added keys would change (nothing is written for that task, and
#: the run exits non-zero);
#: ``skipped`` — no done marker, so the task is the sandbox/convert stage's
#: business, not this mode's.
REBIND_FLAG = "--rebind-samples-digest"
REBIND_REBIND = "rebind"
REBIND_NOOP = "no-op"
REBIND_REFUSED = "refused"
REBIND_SKIPPED = "skipped"
REBIND_VERDICTS: Tuple[str, ...] = (REBIND_REBIND, REBIND_NOOP, REBIND_REFUSED, REBIND_SKIPPED)
#: The Modal function the re-bind mode runs in (the orchestrator image + the
#: volume, no sandbox and no GPU). A literal, like
#: ``tinycap-stage-evalplus-datasets``: it is an operational one-shot, not one
#: of the campaign containers ``contracts.ALL_CONTAINER_NAMES`` enumerates.
REBIND_CONTAINER_NAME = "tinycap-sandbox-rebind-samples-digest"
#: The launcher whose ``--merge`` runs the two-tree merge in the client venv
#: (this script's sibling, so the WRITING tree is this checkout's; the merge
#: never runs here).
LAUNCHER_SCRIPT = Path(__file__).resolve().with_name("tinycap_served.py")
#: ``<out_root>/<cell_id>/controls_summary.json`` — the file the ONE writer
#: (``completion.merge_controls_summary`` -> ``cell_controls.
#: write_controls_summary``) produces. Named here because this image cannot
#: import ``cell_controls`` (torch); the sandbox test pins it equal to
#: ``cell_controls.SUMMARY_FILENAME``.
CONTROLS_SUMMARY_FILENAME = "controls_summary.json"
#: The producer sha under which stores WITHOUT a producer block are grouped
#: (never merged; listed with the remedy).
LEGACY_PRODUCER = None
#: The entrypoint's merge decision, ONE string option (PW10): ``--merge-mode
#: merge`` chains the per-producer merges, ``--merge-mode no-merge`` prints
#: them; ``--merge-sha <sha>`` is the audited chain. A ``bool`` parameter is
#: registered by ``modal run`` as ``--<name>/--no-<name>``, so two bools named
#: ``merge`` and ``no_merge`` collided on ``--no-merge`` (the first owned it as
#: its negation) and the CLI could not express the print mode at all.
MERGE_MODE_FLAG = "--merge-mode"
MERGE_MODE_CHAIN = "merge"
MERGE_MODE_PRINT = "no-merge"
MERGE_MODES: Tuple[str, ...] = (MERGE_MODE_CHAIN, MERGE_MODE_PRINT)
MERGE_SHA_FLAG = "--merge-sha"


def _local_datasets_json() -> Path:
    """The checkout's pins file, found by walking up from this script (it
    lives at scripts/modal/ in a checkout). Inside the Modal container the
    script is mounted flat at /root/, so no ancestor holds it — the image
    definition is not re-built there and the runtime reads the mounted copy
    through ``DATASETS_JSON_ENV`` instead."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / er.DATASETS_JSON_RELPATH
        if cand.is_file():
            return cand
    return Path(DATASETS_JSON_REMOTE)


DATASETS_JSON_LOCAL = _local_datasets_json()

# --- the ORCHESTRATOR image: stdlib + pyyaml (sisteer.capeval.manifest) --------
# The uv.lock pin (pyyaml 6.0.3); the sisteer package is mounted from the
# local checkout so the orchestrator runs THIS branch's validator/converter.
orchestrator_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("pyyaml==6.0.3")
    .env({er.DATASETS_JSON_ENV: DATASETS_JSON_REMOTE})
    .add_local_file(str(DATASETS_JSON_LOCAL), DATASETS_JSON_REMOTE)
    .add_local_python_source("sisteer")
)

# --- the EXECUTION image: the digest-pinned evalplus image, nothing added ------
sandbox_image = modal.Image.from_registry(C.SANDBOX_IMAGE_DIGEST)


# =========================================================================== #
# pure helpers (what tests/test_evalplus_served.py pins)                       #
# =========================================================================== #


def sandbox_create_kwargs(*, image: Any = None) -> Dict[str, Any]:
    """EXACTLY the ``modal.Sandbox.create`` keyword arguments the contract
    fixes: the required limits, the digest-pinned image, the workdir, and
    NONE of the forbidden mounts. Asserted here so a drift fails before a
    sandbox exists."""
    kw: Dict[str, Any] = {**C.SANDBOX_CREATE_KWARGS_REQUIRED,
                          "image": image if image is not None else sandbox_image,
                          "workdir": C.SANDBOX_WORKDIR}
    assert_sandbox_kwargs(kw)
    return kw


def assert_sandbox_kwargs(kw: Mapping[str, Any]) -> None:
    for k, v in C.SANDBOX_CREATE_KWARGS_REQUIRED.items():
        if k not in kw or kw[k] != v:
            raise RuntimeError(f"Sandbox.create kwargs: {k}={kw.get(k)!r} != required {v!r}")
    forbidden = sorted(set(kw) & set(C.SANDBOX_CREATE_KWARGS_FORBIDDEN))
    if forbidden:
        raise RuntimeError(f"Sandbox.create kwargs carry forbidden mounts {forbidden}")
    if kw.get("block_network") is not True:
        raise RuntimeError("Sandbox.create kwargs: block_network must be True")
    if "image" not in kw:
        raise RuntimeError("Sandbox.create kwargs: image missing")


def sandbox_exec_argv(dataset: str) -> List[str]:
    """``contracts.sandbox_entrypoint(dataset)`` as argv for ``sb.exec``:
    ``["bash", "-lc", "ulimit -u 512 ...; cd /work && python -m evalplus.evaluate ..."]``."""
    argv = shlex.split(C.sandbox_entrypoint(dataset))
    if argv[:2] != ["bash", "-lc"] or "ulimit -u 512" not in argv[2]:
        raise RuntimeError(f"unexpected sandbox entrypoint shape: {argv!r}")
    return argv


def sandbox_env(dataset: str) -> Dict[str, str]:
    """The evalplus data-loader override (PC3): ``HUMANEVAL_OVERRIDE_PATH`` /
    ``MBPP_OVERRIDE_PATH`` -> the uploaded archive inside ``/work``."""
    pin = er.dataset_pin(dataset)
    return {pin.override_env: f"{C.SANDBOX_WORKDIR}/{pin.gz_filename}"}


def staged_archive_path(dataset: str, cache_dir: str = EVALPLUS_CACHE_DIR) -> Path:
    return Path(cache_dir) / er.dataset_pin(dataset).gz_filename


def verify_archive(dataset: str, path: Path) -> bytes:
    """The archive bytes, refused unless their sha256 / size equal the pin."""
    pin = er.dataset_pin(dataset)
    if not path.is_file():
        raise RuntimeError(
            f"{dataset}: staged archive {path} missing — run with --stage-datasets first")
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    if sha != pin.gz_sha256 or len(data) != pin.gz_bytes:
        raise RuntimeError(
            f"{dataset}: staged archive {path} sha256 {sha[:12]}… / {len(data)} bytes != "
            f"pinned {pin.gz_sha256[:12]}… / {pin.gz_bytes} — refusing a drifted dataset")
    return data


def task_dir_parts(tdir: Path) -> Dict[str, Any]:
    """``(dataset, open_variant)`` from a served task dir name
    (``evalplus/<dataset>[_open]``)."""
    name = tdir.name
    open_variant = name.endswith("_open")
    dataset = name[: -len("_open")] if open_variant else name
    if dataset not in C.EVALPLUS_DATASETS or tdir.parent.name != "evalplus":
        raise RuntimeError(f"{tdir} is not a served evalplus task dir")
    return {"dataset": dataset, "open_variant": open_variant,
            "task": C.evalplus_task(dataset, open_variant=open_variant)}


def samples_per_item_of(tdir: Path) -> int:
    """The task's k (U8 / PW-sampling) as its ``gen_meta.json`` records it —
    1 when the meta predates the switch or is absent (the conversion reads
    the same field; ``evalplus_runner.convert_eval_results_to_store``)."""
    meta_path = tdir / er.GEN_META_FILENAME
    if not meta_path.exists():
        return 1
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 1
    k = meta.get("samples_per_item", 1) if isinstance(meta, Mapping) else 1
    return int(k) if isinstance(k, int) and not isinstance(k, bool) and k >= 1 else 1


def samples_sha256(path: Path) -> str:
    """The sha256 of a ``samples.jsonl`` file's EXACT bytes — the identity
    :data:`SAMPLES_DIGEST_FIELD` records and every resume compares against."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recorded_samples_digest(tdir: Path) -> Tuple[Optional[str], str]:
    """``(digest, why)`` — what this task's ``execution.json`` records about the
    samples its results were produced from. ``digest`` is None when nothing
    does, and ``why`` always says which case held, so a caller's refusal names
    the state rather than the rule."""
    path = tdir / er.EXECUTION_FILENAME
    if not path.exists():
        return None, f"there is no {er.EXECUTION_FILENAME} beside them"
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return None, f"{path} did not parse ({type(e).__name__}: {str(e)[:160]})"
    value = doc.get(SAMPLES_DIGEST_FIELD) if isinstance(doc, Mapping) else None
    if value is None:
        return None, (f"{path} records no {SAMPLES_DIGEST_FIELD} — an execution written before "
                      "the samples digest existed")
    if not (isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)):
        return None, f"{path} records {SAMPLES_DIGEST_FIELD}={value!r}, which is not a sha256"
    return value, f"{path} records {SAMPLES_DIGEST_FIELD}"


def samples_binding(tdir: Path) -> Tuple[str, str]:
    """E50 #4: ``(verdict, message)`` — may this task's on-disk
    ``eval_results.json`` be CONVERTED against the ``samples.jsonl`` currently
    beside it?

    A conversion turns results into the store's numbers, so the one thing it
    must establish is that those results were produced from THESE sample bytes.
    Task ids, count, shape and the dataset archive md5 — everything a resume
    checked before — are all invariant under regenerating the same problems
    with DIFFERENT solutions, which is exactly the state a crashed-and-rerun
    generation leaves. So the execution records the samples' sha256 and this
    compares it:

    * :data:`BINDING_BOUND` — the recorded digest IS the current file's;
    * :data:`BINDING_MISMATCH` — it names other bytes: the samples were
      rewritten after those results were produced, and converting them would
      pair old results with new samples (the finding);
    * :data:`BINDING_UNBOUND` — nothing records which samples produced the
      results (no ``execution.json``, or a pre-digest one). A pre-digest
      execution cannot vouch for its samples, so this is refused too rather
      than accepted on the strength of the checks that do not move.

    Both refusals carry the same cheap remedy — re-run the sandbox stage for
    this task — and neither loses anything: the results are re-derivable, and
    the task stays listed as pending until the remedy is applied."""
    samples = tdir / er.SAMPLES_FILENAME
    if not samples.is_file():
        # E54 #1: the fourth shape of "nothing binds these results", made a
        # per-task refusal rather than an OSError out of the scan. The glob
        # this scan walks is over samples.jsonl, so it takes a deletion between
        # the walk and the read to reach here — which is exactly the damage
        # this check exists to name.
        return BINDING_UNBOUND, (
            f"{tdir}: {er.EVAL_RESULTS_FILENAME} is on disk but there is no "
            f"{er.SAMPLES_FILENAME} beside it to bind it to; nothing says which samples "
            f"produced these results. Re-run the sandbox stage for this task: remove "
            f"{er.EVAL_RESULTS_FILENAME} and {er.EXECUTION_FILENAME} from {tdir} and "
            "re-generate the task so it executes afresh.")
    current = samples_sha256(samples)
    recorded, why = recorded_samples_digest(tdir)
    if recorded is None:
        return BINDING_UNBOUND, (
            f"{tdir}: {er.EVAL_RESULTS_FILENAME} is on disk but nothing records which "
            f"{er.SAMPLES_FILENAME} produced it ({why}); the file there now hashes to sha256 "
            f"{current}. A pre-digest execution cannot vouch for its samples — ids, count, "
            "shape and the archive md5 are all unchanged by regenerating the same problems "
            f"with different solutions — so these results are never converted as they stand. "
            f"Re-run the sandbox stage for this task: remove {er.EVAL_RESULTS_FILENAME} and "
            f"{er.EXECUTION_FILENAME} from {tdir} so it executes afresh.")
    if recorded != current:
        return BINDING_MISMATCH, (
            f"{tdir}: {er.EVAL_RESULTS_FILENAME} was produced from samples sha256 {recorded} "
            f"({why}), but {er.SAMPLES_FILENAME} now hashes to sha256 {current} — the samples "
            "were regenerated after those results, so converting them would attribute OLD "
            "results to NEW samples. Re-run the sandbox stage for this task: remove "
            f"{er.EVAL_RESULTS_FILENAME} and {er.EXECUTION_FILENAME} from {tdir} so it "
            "executes afresh.")
    return BINDING_BOUND, f"{why} = {current}, the {er.SAMPLES_FILENAME} on disk"


def results_state(tdir: Path, *, dataset: str) -> Tuple[str, Optional[str]]:
    """``("absent" | "valid" | "invalid", reason)`` for a task dir's
    ``eval_results.json``: valid means it parses to the 0.3.1 shape over
    EXACTLY the pinned id set and — when ``execution.json`` recorded the
    archive it was evaluated against — carries that archive's md5. ``{}``,
    a torn file or any shape drift is ``invalid`` (R7: never completion)."""
    path = tdir / er.EVAL_RESULTS_FILENAME
    if not path.exists():
        return "absent", None
    try:
        results = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return "invalid", f"unreadable ({type(e).__name__}: {str(e)[:160]})"
    expected_hash = None
    exec_path = tdir / er.EXECUTION_FILENAME
    if exec_path.exists():
        try:
            archive = json.loads(exec_path.read_text(encoding="utf-8")).get("archive")
            expected_hash = archive.get("md5") if isinstance(archive, Mapping) else None
        except (OSError, ValueError, AttributeError):
            expected_hash = None
    try:
        er.validate_eval_results(results, expected_task_ids=er.expected_task_ids(dataset),
                                 where=str(path), expected_hash=expected_hash,
                                 samples_per_item=samples_per_item_of(tdir))
    except (er.EvalPlusSeamError, ValueError, TypeError, KeyError) as e:
        return "invalid", f"{type(e).__name__}: {str(e)[:300]}"
    return "valid", None


def pending_executions(gen_root: str | Path, *, datasets: Sequence[str] = C.EVALPLUS_DATASETS
                       ) -> List[Dict[str, Any]]:
    """Every ``<gen_root>/**/cells/*/evalplus/<dataset>[_open]/samples.jsonl``
    whose ``gen_meta.json`` says complete and whose task has NO completion
    marker (done or partial — the marker is authoritative, R7), as
    ``{cell_dir, task_dir, dataset, open_variant, task, stage}`` — sorted,
    so runs are reproducible. ``stage`` is ``convert`` when a validated
    ``eval_results.json`` is already on disk (a crash between the results
    write and the conversion), else ``execute`` (absent, ``{}`` or
    malformed results — recorded as ``stale_results`` — mean a sandbox
    run)."""
    root = Path(gen_root)
    out: List[Dict[str, Any]] = []
    for samples in sorted(root.glob(f"**/cells/*/evalplus/*/{er.SAMPLES_FILENAME}")):
        tdir = samples.parent
        try:
            parts = task_dir_parts(tdir)
        except RuntimeError:
            continue
        if parts["dataset"] not in datasets:
            continue
        cell_dir = tdir.parent.parent
        task = parts["task"]
        if completion.is_done(cell_dir, task):
            continue                                  # the marker is authoritative
        if completion.partial_marker_path(cell_dir, task).is_file():
            print(f"[tinycap-sandbox] skip {tdir}: converted PARTIAL (a limited store; the "
                  "launcher re-runs the control in a full store, never this one)")
            continue
        meta_path = tdir / er.GEN_META_FILENAME
        if not meta_path.exists():
            print(f"[tinycap-sandbox] skip {tdir}: no {er.GEN_META_FILENAME}")
            continue
        meta = json.loads(meta_path.read_text())
        if meta.get("status") != "complete":
            print(f"[tinycap-sandbox] skip {tdir}: generation status {meta.get('status')!r}")
            continue
        job: Dict[str, Any] = {"cell_dir": str(cell_dir), "task_dir": str(tdir), **parts}
        state, why = results_state(tdir, dataset=parts["dataset"])
        if state == "valid":
            # E50 #4: results may only be converted against the samples they
            # were produced from; an unbound or contradicted pair is refused
            # HERE too, so the dry run says so rather than promising a
            # conversion that execute_one will refuse
            verdict, msg = samples_binding(tdir)
            if verdict == BINDING_BOUND:
                job["stage"] = STAGE_CONVERT
            else:
                job["stage"] = STAGE_REFUSE
                job["refusal"] = msg
                job["samples_binding"] = verdict
                print(f"[tinycap-sandbox] REFUSING: {msg}")
        else:
            job["stage"] = STAGE_EXECUTE
            if state == "invalid":
                job["stale_results"] = why
                print(f"[tinycap-sandbox] {tdir}: {er.EVAL_RESULTS_FILENAME} present but NOT valid "
                      f"({why}) — re-executing, never converting it")
        out.append(job)
    return out


def execute_one(job: Mapping[str, Any], *, sandbox_cls: Any, app_obj: Any, image: Any,
                cache_dir: str = EVALPLUS_CACHE_DIR, commit=None) -> Dict[str, Any]:
    """One (cell, mode, dataset) execution — module docstring steps 1-6.
    ``sandbox_cls`` / ``app_obj`` / ``image`` are injected so the contract
    test drives it with a fake ``Sandbox``; production passes
    ``modal.Sandbox`` / ``app`` / ``sandbox_image``. Returns the execution
    record; raises on any refusal (nothing is written to the volume then)."""
    t0 = time.time()
    tdir = Path(job["task_dir"])
    dataset = job["dataset"]
    task = job["task"]
    if job.get("stage") == STAGE_REFUSE:
        # the scan already looked and will not act on this task (E50 #4); raise
        # its reason before reading anything, so the message is the refusal and
        # not whatever the next read happens to find
        raise RuntimeError(job.get("refusal")
                           or f"{task} @ {tdir}: refused by the scan, with no reason recorded")
    expected = er.expected_task_ids(dataset)
    pin = er.dataset_pin(dataset)
    samples_path = tdir / er.SAMPLES_FILENAME
    k = samples_per_item_of(tdir)                     # the task's k (U8): k lines per task_id
    samples = er.read_samples_jsonl(samples_path, expected_task_ids=expected,
                                    samples_per_item=k)  # step 1
    samples_bytes = samples_path.read_bytes()
    archive = verify_archive(dataset, staged_archive_path(dataset, cache_dir))
    archive_md5 = hashlib.md5(archive).hexdigest()
    if archive_md5 != pin.gz_md5:
        raise RuntimeError(f"{dataset}: archive md5 {archive_md5} != pinned {pin.gz_md5}")

    samples_digest = hashlib.sha256(samples_bytes).hexdigest()

    stage = job.get("stage", STAGE_EXECUTE)
    if stage == STAGE_CONVERT:
        # R7: validated results already on disk (a crash between the results
        # write and the conversion) — convert them, never a second sandbox.
        # E50 #4: but only against the samples they were PRODUCED from. This is
        # the authoritative door (the scan's is the honest plan); a caller that
        # built the job itself passes through it too.
        verdict, msg = samples_binding(tdir)
        if verdict != BINDING_BOUND:
            raise RuntimeError(f"{task}: {msg}")
        results_path = tdir / er.EVAL_RESULTS_FILENAME
        results = json.loads(results_path.read_text(encoding="utf-8"))
        try:
            er.validate_eval_results(results, expected_task_ids=expected, where=str(results_path),
                                     expected_hash=archive_md5, samples_per_item=k)
        except er.EvalPlusSeamError as e:
            raise RuntimeError(
                f"{task}: the on-disk {er.EVAL_RESULTS_FILENAME} does not validate against the "
                f"STAGED archive ({e}) — remove it (and {er.EXECUTION_FILENAME}) so the task "
                "re-executes in a sandbox; it is never converted as it stands") from e
        print(f"[tinycap-sandbox] {task} @ {tdir}: resuming at CONVERSION from validated "
              f"{er.EVAL_RESULTS_FILENAME} (no sandbox)")
        return _convert(job, results=results, record={
            "task": task, "dataset": dataset, "open_variant": job["open_variant"],
            "stage": stage, "resumed_from": str(results_path), "n_samples": len(samples),
            "samples_per_item": k, SAMPLES_DIGEST_FIELD: samples_digest,
            C.MANIFEST_KEY_CONTRACT_VERSION: C.CONTRACT_VERSION,
            C.MANIFEST_KEY_PRODUCER: {"driver": C.PRODUCER_DRIVER_SANDBOX},
            "archive": {"filename": pin.gz_filename, "sha256": pin.gz_sha256, "md5": archive_md5},
            "started_at": t0}, commit=commit)
    if stage != STAGE_EXECUTE:
        raise RuntimeError(f"{task}: unknown job stage {stage!r}")

    kw = sandbox_create_kwargs(image=image)                                   # step 2
    env = sandbox_env(dataset)
    argv = sandbox_exec_argv(dataset)
    print(f"[tinycap-sandbox] {task} @ {tdir}: creating sandbox "
          f"(cpu={kw['cpu']} mem={kw['memory']}MiB timeout={kw['timeout']}s block_network={kw['block_network']})")
    sb = sandbox_cls.create(app=app_obj, env=env, **kw)
    record: Dict[str, Any] = {
        "task": task, "dataset": dataset, "open_variant": job["open_variant"], "stage": stage,
        "sandbox_id": getattr(sb, "object_id", None),
        "image_digest": C.SANDBOX_IMAGE_DIGEST,
        "create_kwargs": {k: v for k, v in kw.items() if k != "image"},
        "env": env, "exec_argv": argv,
        "archive": {"filename": pin.gz_filename, "sha256": pin.gz_sha256, "md5": archive_md5},
        "n_samples": len(samples), "samples_per_item": k, "started_at": t0,
        # E50 #4: the identity of the bytes this sandbox is about to evaluate —
        # written into execution.json beside the results, so every later resume
        # and the writer-side merge can tell these results from results of a
        # regeneration under the same ids
        SAMPLES_DIGEST_FIELD: samples_digest,
        # E55 #2c: and WHO wrote this record. An EvalPlus store has two
        # producers — the generation that wrote the samples and the sandbox
        # that scored them — and each binding fact is judged by ITS OWN
        # producer's contract, so the execution record has to be able to date
        # itself. This image mounts the sisteer PACKAGE, not a tree, so it has
        # no TREE_SHA to stamp; the contract_version compiled into the mounted
        # code is the marker, and contracts.store_era reads the same two keys
        # here as on a cell manifest.
        C.MANIFEST_KEY_CONTRACT_VERSION: C.CONTRACT_VERSION,
        C.MANIFEST_KEY_PRODUCER: {"driver": C.PRODUCER_DRIVER_SANDBOX},
    }
    try:
        sb.mkdir(C.SANDBOX_WORKDIR, parents=True)                             # step 3
        with sb.open(C.SANDBOX_SAMPLES_PATH, "wb") as f:
            f.write(samples_bytes)
        with sb.open(f"{C.SANDBOX_WORKDIR}/{pin.gz_filename}", "wb") as f:
            f.write(archive)
        proc = sb.exec(*argv, timeout=C.SANDBOX_TIMEOUT_S)                    # step 4
        rc = proc.wait()
        stdout = _tail(proc.stdout.read())
        stderr = _tail(proc.stderr.read())
        record.update({"returncode": rc, "stdout_tail": stdout, "stderr_tail": stderr,
                       "exec_wall_clock_s": time.time() - t0})
        if rc != 0:
            raise RuntimeError(
                f"{task}: evalplus.evaluate exited {rc} inside the sandbox — no results for "
                f"this task (stderr tail: {stderr[-600:]!r})")
        try:                                                                  # step 5
            with sb.open(C.SANDBOX_RESULTS_PATH, "rb") as f:
                raw = f.read()
            results = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
        except Exception as e:  # noqa: BLE001 — a missing/unparseable file is one refusal class
            raise RuntimeError(
                f"{task}: no readable results at {C.SANDBOX_RESULTS_PATH} after a zero exit "
                f"({type(e).__name__}: {e}) — refusing; stdout tail: {stdout[-400:]!r}") from e
        er.validate_eval_results(results, expected_task_ids=expected,
                                 where=f"{task}:{C.SANDBOX_RESULTS_PATH}",
                                 expected_hash=archive_md5, samples_per_item=k)
    finally:
        try:
            sb.terminate()                                                    # step 6
        except Exception as e:  # noqa: BLE001 — never mask the real error
            print(f"[tinycap-sandbox] WARNING: terminate failed: {type(e).__name__}: {e}")
    # -- validated: only now does anything touch the volume ---------------- #
    record["wall_clock_s"] = time.time() - t0
    record["finished_at"] = time.time()
    er._atomic_write_json(tdir / er.EVAL_RESULTS_FILENAME, results)
    er._atomic_write_json(tdir / er.EXECUTION_FILENAME, record)
    if commit is not None:
        commit()
    return _convert(job, results=results, record=record, commit=commit)


def _convert(job: Mapping[str, Any], *, results: Mapping[str, Any], record: Dict[str, Any],
             commit=None) -> Dict[str, Any]:
    """Conversion into the store (rows + summary + the done / partial marker
    — ``convert_eval_results_to_store``), then the record says whether the
    task is COMPLETE (a done marker exists) and that its controls_summary
    merge is pending in the launcher's client venv (N1: the merge never
    runs in this image)."""
    task, dataset = job["task"], job["dataset"]
    summary = er.convert_eval_results_to_store(job["cell_dir"], dataset=dataset,
                                               open_variant=job["open_variant"], results=results)
    if commit is not None:
        commit()
    record["summary"] = summary
    record["converted"] = True
    record["complete"] = completion.is_done(job["cell_dir"], task)
    record["merge"] = ("pending: the launcher's --merge for the producing tree (tinycap_served.py "
                       "merge_summaries, client venv) — found from disk by unmerged_completed_stores "
                       "on every invocation until the payload is in controls_summary.json"
                       if record["complete"] else "not applicable: a PARTIAL (limited) store never merges")
    print(f"[tinycap-sandbox] {task} converted"
          + (f" in {record['wall_clock_s']:.0f}s" if "wall_clock_s" in record else "")
          + f": pass@1_base={summary['pass_at_1_base']} pass@1_plus={summary['pass_at_1_plus']} "
          f"n={summary['n']} complete={record['complete']}")
    return record


def converted_cell_dirs(records: Sequence[Mapping[str, Any]]) -> List[str]:
    """The cell dirs whose conversion COMPLETED a task (a done marker now
    exists) in one orchestrator run — sorted, de-duplicated. Informational
    since E12 item 2: the merge's candidates come from
    :func:`unmerged_completed_stores` (disk), of which these are a subset."""
    dirs = {str(r["job"]["cell_dir"]) for r in records
            if r.get("ok") and (r.get("record") or {}).get("complete")}
    return sorted(dirs)


def producer_tree_sha(manifest: Mapping[str, Any]) -> Optional[str]:
    """``manifest.producer.tree_sha`` (``contracts.MANIFEST_KEY_PRODUCER``);
    None for a store from before the producer discipline."""
    producer = manifest.get(C.MANIFEST_KEY_PRODUCER)
    sha = producer.get("tree_sha") if isinstance(producer, Mapping) else None
    return sha if isinstance(sha, str) and sha else None


def store_merge_source(cell_dir: Path, manifest: Mapping[str, Any]
                       ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """``(the source this store's PRODUCING tree records for it, None)``, or
    ``(None, why it cannot be known)`` — PW8 / E16 A1. The four
    ``MERGE_SOURCE_FIELDS``: the directory key; the manifest's
    ``harness_config_hash`` (absent ⇒ unknowable); the pass — the HASHED
    identity's when the identity carries one (every store since PW8; the
    merge records exactly that, ``completion.merge_source``), else the
    top-level ``geometry.pass`` (exactly what a pre-PW8 tree's merge
    recorded, so a store is compared against its own tree's record, never
    a guess; both absent ⇒ null, as such a tree recorded) — with the two
    DISAGREEING unmergeable (the merge refuses that store); and the
    producer sha (None = legacy). Torch-free: this image imports no writer."""
    chash = manifest.get("harness_config_hash")
    if not isinstance(chash, str) or not chash:
        return None, "manifest names no harness_config_hash (the merge cannot record its source)"
    hashed = completion.hashed_pass(manifest)
    top = completion.geometry_pass(manifest)
    if hashed is not completion.PASS_UNRECORDED:
        if top is not completion.PASS_UNRECORDED and top != hashed:
            return None, (f"{C.MANIFEST_KEY_GEOMETRY}.{C.GEOMETRY_PASS_FIELD} {top!r} disagrees with "
                          f"the hashed identity's {hashed!r} (a relabelled store; the merge refuses it)")
        pass_no = hashed
    else:
        pass_no = None if top is completion.PASS_UNRECORDED else top
    return {"cell_key": cell_dir.name, "harness_config_hash": chash, "pass": pass_no,
            "producer_tree_sha": producer_tree_sha(manifest)}, None


def _source_core(record: Any) -> Optional[Dict[str, Any]]:
    return ({f: record.get(f) for f in C.MERGE_SOURCE_FIELDS}
            if isinstance(record, Mapping) else None)


#: a payload digest as the writer files it (``contracts.merge_payload_digest``): ``PAYLOAD_DIGEST_HEX_LEN``
#: lowercase hex — the scan's one statement of the rule ``cell_controls._valid_payload_digest`` applies (that
#: module imports torch through ``sisteer.methods``, so the torch-free scan cannot share the function itself)
_PAYLOAD_DIGEST_RE = re.compile(rf"[0-9a-f]{{{C.PAYLOAD_DIGEST_HEX_LEN}}}")


def _valid_payload_digest(value) -> bool:
    """The scan's ONE function for the digest rule — the torch-free twin of
    ``cell_controls._valid_payload_digest`` (that module cannot be imported here),
    kept to the same name and shape so the parity test can import BOTH and drive
    one table through them. Every scanner call site goes through this, never the
    regex directly: a call site drifting to ``.match`` would otherwise read a
    17-char or newline-terminated value as a digest."""
    return isinstance(value, str) and _PAYLOAD_DIGEST_RE.fullmatch(value) is not None


def unaccounted_reason(source: Mapping[str, Any], recorded: Any, *,
                       payload_digest: Optional[str] = None,
                       standing_digest: Optional[str] = None) -> Optional[str]:
    """Why the ``sources`` entry ``recorded`` (a source record, or None when
    the entry has none) does NOT account for the store ``source`` — None
    when it does: the record's four fields ARE the source (the identical
    store re-merged), the source is in the record's ``superseded`` list (a
    higher pass, or a named winner, took the entry over from it), or — PW13
    (E21 M3) — the source is one of the record's ``equal_witnesses`` (PW12:
    a store that offered EXACTLY the standing payload and yielded the entry
    to the adopter under ``EQUAL_WITNESS_RULE``; the same number, attributed
    by the recorded rule, so there is nothing left for it to merge). The
    same four-field comparison as ``superseded``. Before PW13 an equal
    witness read as pending on every scan and ``--merge-mode merge`` kept
    rescheduling its producer tree. An ``UNRECORDED_MERGE_SOURCE`` record
    satisfies no concrete store.

    PW14 (E22 M4): a superseded or equal-witness record accounts for the
    store only while the store still holds the payload that decision was
    made on — its ``payload_digest`` (``contracts.merge_payload_digest`` of the
    store's CURRENT ``summary/<task>.json``, passed as ``payload_digest``)
    must equal the record's. Task summaries are mutable atomic replacements:
    a superseded store whose summary changed is a NEW claim and reads pending
    again (the merge then refuses it until named, or the pass rule decides
    it afresh), and a record WITHOUT a digest (written before PW14) accounts
    for nothing — the merge re-records it with the digest (or refuses,
    naming ``--replace-source``).

    PW16 (E24): the STANDING source accounts for the store only while the
    store still holds the CANONICAL payload — ``payload_digest`` (the store's
    current ``summary/<task>.json``) must equal ``standing_digest``
    (``contracts.merge_payload_digest`` of ``controls[task]``, the canonical
    value the scan reads from the summary file). Before this the four-field
    match returned accounted BEFORE any digest was consulted, so a standing
    store whose own summary had been atomically rewritten never read pending —
    and the writer's ``same`` branch overlaid it silently. A changed standing
    store reads pending with both digests and the remedy (the merge refuses
    it until ``--replace-source`` names the standing store itself, which
    refreshes the entry and records the refresh); a standing record whose
    recorded ``payload_digest`` disagrees with the canonical payload's is a
    hand-edited summary and reads pending too (the loader refuses it) —
    PW19 (E27): whether the store's current summary equals the canonical
    payload or not, and the same named re-merge of the standing store repairs
    it (the record brought to the canonical payload, recorded), after which
    the store reads accounted. A standing record WITHOUT a digest — the KEY
    absent (written before PW16) — reads accounted while the store holds the
    canonical payload: the next write into the file backfills the digest.
    PW19b: an explicit ``null`` or any other non-digest value under the key is
    malformed, never that absence — it reads pending (the loader refuses the
    file, the writer refuses to write into it), so the scan never suppresses
    the repair a loadable summary needs. PW19c: the same presence rule for
    the HISTORY records — a ``superseded`` entry or an ``equal_witnesses``
    record whose key is present but holds no digest reads pending as
    malformed (before this the scan read an explicit null as the pre-PW14
    absence, ``WITHOUT a payload digest``, and a non-hex string as a changed
    claim, while the writer silently re-recorded either as a legacy record);
    only the ABSENT key is the pre-PW14 record."""
    core = _source_core(recorded)
    if core is None:
        return ("the entry names no source (a summary written before the source discipline, "
                "or torn)")
    if core == dict(source):
        # PW16 (E24): the standing store — still holding the canonical payload?
        if payload_digest is None:
            return ("the store IS the entry's standing source but its current payload digest was not "
                    "supplied to the scan (PW16)")
        if standing_digest is None:
            return ("the store IS the entry's standing source but the canonical payload's digest was not "
                    "supplied to the scan (PW16)")
        if C.MERGE_SOURCE_PAYLOAD_DIGEST_FIELD in recorded:
            # PW19b: presence apart from value — an explicit null (or any non-digest value) under the key is
            # a malformed record, never the pre-PW16 absence; it reads PENDING, exactly as the loader refuses it
            recorded_digest = recorded[C.MERGE_SOURCE_PAYLOAD_DIGEST_FIELD]
            if not _valid_payload_digest(recorded_digest):
                return (f"the store IS the entry's standing source, but its source record says "
                        f"{C.MERGE_SOURCE_PAYLOAD_DIGEST_FIELD}={recorded_digest!r} — a payload digest is "
                        f"{C.PAYLOAD_DIGEST_HEX_LEN} hex or ABSENT (a record from before PW16); an explicit null or "
                        "any other value under the key is a hand-edited or torn record, not the legacy absence: the "
                        "loader refuses the file and the writer refuses to write into it (never a backfill) — restore "
                        "controls_summary.json from its last good write (PW19b)")
        else:
            recorded_digest = None
        if recorded_digest is not None and recorded_digest != standing_digest:
            return (f"the store IS the entry's standing source, but the canonical payload digests "
                    f"{standing_digest} while its source record says {recorded_digest} — the summary was "
                    "hand-edited (or torn) since the writer filed it, or a writer from before PW16 overlaid the "
                    "standing store's changed payload (PW19 / E27); the loader refuses it — restore the file, "
                    f"or re-file the entry from the standing store by name (re-merge it with "
                    f"{C.REPLACE_SOURCE_FLAG} <entry>{C.REPLACE_SOURCE_ASSIGN}{source.get('cell_key')}, which "
                    "brings the record to the canonical payload, recorded as a refresh) (PW16 / E24)")
        if payload_digest != standing_digest:
            return (f"the store IS the entry's standing source under payload digest {standing_digest}, but "
                    f"its {C.SUMMARY_DIR}/<task>.json now digests {payload_digest} — the standing source's "
                    "OWN summary changed since the entry was written, so the canonical value and its "
                    "authoritative source disagree; the merge refuses to overlay it silently. A plain "
                    "re-merge files it when the change is a RE-BIND — the summary gained only "
                    f"{C.MERGE_SOURCE_SAMPLES_DIGEST_FIELD}, byte-identically otherwise, agreeing with every "
                    "witness the store presents (E64); otherwise name the standing store itself "
                    f"({C.REPLACE_SOURCE_FLAG} <entry>{C.REPLACE_SOURCE_ASSIGN}{source.get('cell_key')}) to "
                    "REFRESH the entry, recorded with both digests (PW16 / E24)")
        return None                      # the standing store, at the canonical payload
    for kind, field in ((C.MERGE_SOURCE_SUPERSEDED_FIELD, C.MERGE_SOURCE_SUPERSEDED_FIELD),
                        ("an equal witness", C.MERGE_SOURCE_EQUAL_WITNESSES_FIELD)):
        records = recorded.get(field) or []
        rec = (next((r for r in records if _source_core(r) == dict(source)), None)
               if isinstance(records, list) else None)
        if rec is None:
            continue
        if C.MERGE_SOURCE_PAYLOAD_DIGEST_FIELD not in rec:
            return (f"the store is recorded {kind} WITHOUT a payload digest (a record from before "
                    "PW14): the file cannot say whether its current summary is the payload that decision "
                    "saw — re-run the merge (the pass rule re-records it with the digest) or resolve it "
                    f"with {C.REPLACE_SOURCE_FLAG} (PW14)")
        digest = rec[C.MERGE_SOURCE_PAYLOAD_DIGEST_FIELD]
        if not _valid_payload_digest(digest):
            # PW19c: presence apart from value — an explicit null (or any non-digest value) under the key is a
            # malformed record, never the pre-PW14 absence above; it reads PENDING, exactly as the loader refuses it
            return (f"the store is recorded {kind}, but its record says "
                    f"{C.MERGE_SOURCE_PAYLOAD_DIGEST_FIELD}={digest!r} — a payload digest is "
                    f"{C.PAYLOAD_DIGEST_HEX_LEN} hex or ABSENT (a record from before PW14); an explicit null or "
                    "any other value under the key is a hand-edited or torn record, not the legacy absence: the "
                    "loader refuses the file and the writer refuses to write into it (never a re-recording) — "
                    "restore controls_summary.json from its last good write (PW19c)")
        if payload_digest is None:
            return (f"the store is recorded {kind} under payload digest {digest} but its current "
                    "payload digest was not supplied to the scan (PW14)")
        if digest != payload_digest:
            return (f"the store is recorded {kind} under payload digest {digest} but its summary now "
                    f"digests {payload_digest} — the payload CHANGED since that decision, so the claim "
                    "resurfaces (re-run the merge; a differing claim refuses until named) (PW14)")
        return None                      # superseded / an equal witness, at the recorded payload
    if core == dict(C.UNRECORDED_MERGE_SOURCE):
        return ("the entry's source is unrecorded (merged before the source discipline) — a "
                "concrete store is never satisfied by it")
    return f"the entry stands from {core}, not from this store"


def unmerged_completed_stores(gen_root: str | Path, *,
                              datasets: Sequence[str] = C.EVALPLUS_DATASETS
                              ) -> List[Dict[str, Any]]:
    """E12 item 2 — the merge's pending state, DERIVED FROM DISK: every
    ``<gen_root>/**/cells/*/`` store holding a DONE marker for an
    ``evalplus/<dataset>[:open]`` task (``datasets`` filters) that
    ``<out_root>/<cell_id>/controls_summary.json`` does not ACCOUNT FOR —
    the file missing or torn, the task not among its ``controls``, or its
    ``sources`` entry naming neither this store's exact source
    (:func:`store_merge_source`) as the standing one nor in its
    ``superseded`` list nor among its ``equal_witnesses``
    (:func:`unaccounted_reason`; PW8 / E16 A1 — the task name alone never
    counts; PW13 / E21 M3 — an equal witness is accounted; PW14 / PW16 — each
    only while the store's CURRENT ``summary/<task>.json`` digests to the
    record's digest, resp. to the CANONICAL ``controls[task]`` for the
    standing source). Each record: ``{cell_dir, cell_id, tasks (the
    unaccounted ones), why ({task: reason}), source, producer_tree_sha (None
    = legacy), summary}``; sorted. A store whose manifest is unreadable,
    names no ``cell_id`` or cannot state its source is listed with
    ``unmergeable`` naming why (it can never merge, so it is never silently
    dropped). Idempotent: a merged store disappears from the list because
    its source is now recorded in the file."""
    root = Path(gen_root)
    wanted = tuple(d for d in C.EVALPLUS_DATASETS if d in datasets)
    out: List[Dict[str, Any]] = []
    for m in sorted(root.glob("**/cells/*/manifest.json")):
        cell_dir = m.parent
        done = [C.evalplus_task(d, open_variant=o) for d in wanted for o in (False, True)
                if completion.is_done(cell_dir, C.evalplus_task(d, open_variant=o))]
        if not done:
            continue
        rec: Dict[str, Any] = {"cell_dir": str(cell_dir), "cell_id": None, "tasks": done,
                               "producer_tree_sha": None, "summary": None}
        try:
            manifest = json.loads(m.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            rec["unmergeable"] = f"manifest.json unreadable ({type(e).__name__}: {str(e)[:120]})"
            out.append(rec)
            continue
        cell_id = manifest.get("cell_id") if isinstance(manifest, Mapping) else None
        if not isinstance(cell_id, str) or not cell_id:
            rec["unmergeable"] = "manifest names no cell_id (the merge cannot locate its summary)"
            out.append(rec)
            continue
        rec["cell_id"] = cell_id
        rec["producer_tree_sha"] = producer_tree_sha(manifest)
        summary_path = cell_dir.parent.parent / cell_id / CONTROLS_SUMMARY_FILENAME
        rec["summary"] = str(summary_path)
        source, bad = store_merge_source(cell_dir, manifest)
        if bad:
            rec["unmergeable"] = bad
            out.append(rec)
            continue
        rec["source"] = source
        controls: Mapping[str, Any] = {}
        sources: Mapping[str, Any] = {}
        absent = f"no {CONTROLS_SUMMARY_FILENAME}"
        if summary_path.is_file():
            try:
                doc = json.loads(summary_path.read_text(encoding="utf-8"))
                controls = doc.get("controls") if isinstance(doc, Mapping) else None
                sources = doc.get(C.SUMMARY_KEY_SOURCES) if isinstance(doc, Mapping) else None
                controls = controls if isinstance(controls, Mapping) else {}
                sources = sources if isinstance(sources, Mapping) else {}
                absent = "the payload is absent from controls"
            except (OSError, ValueError):
                absent = f"{CONTROLS_SUMMARY_FILENAME} unreadable (torn; the re-merge is idempotent)"
        why: Dict[str, str] = {}
        for task in done:
            if task not in controls:
                why[task] = absent
                continue
            # the store's CURRENT payload, digested as the writer digests it (PW14 / E22 M4)
            current = completion.read_task_summary(cell_dir, task)
            if not isinstance(current, Mapping):
                why[task] = (f"done marker without a {C.SUMMARY_DIR}/ payload (a torn store; the merge "
                             "refuses it)")
                continue
            try:
                digest = C.merge_payload_digest(task, current)
            except ValueError as e:
                why[task] = f"the store's {C.SUMMARY_DIR}/ payload cannot be digested ({e})"
                continue
            # ...and the CANONICAL payload's digest, for the standing source (PW16 / E24)
            standing = controls[task]
            if not isinstance(standing, Mapping):
                why[task] = (f"the canonical payload in controls is a {type(standing).__name__}, not a "
                             "mapping (a torn or hand-edited summary; the loader refuses it)")
                continue
            try:
                standing_digest = C.merge_payload_digest(task, standing)
            except ValueError as e:
                why[task] = f"the canonical payload in controls cannot be digested ({e})"
                continue
            reason = unaccounted_reason(source, sources.get(task), payload_digest=digest,
                                        standing_digest=standing_digest)
            if reason is not None:
                why[task] = reason
        if why:
            rec["tasks"] = list(why)
            rec["why"] = why
            out.append(rec)
    return out


def merge_groups(pending: Sequence[Mapping[str, Any]]) -> Dict[Optional[str], List[str]]:
    """E12 item 3: the merge-pending cell dirs GROUPED by producer tree sha
    (``LEGACY_PRODUCER`` = None for stores without the block); each group is
    merged through its own staged tree. Records marked ``unmergeable`` are
    left out (reported separately)."""
    groups: Dict[Optional[str], set] = {}
    for rec in pending:
        if rec.get("unmergeable"):
            continue
        groups.setdefault(rec.get("producer_tree_sha"), set()).add(str(rec["cell_dir"]))
    return {sha: sorted(dirs) for sha, dirs in sorted(
        groups.items(), key=lambda kv: (kv[0] is None, kv[0] or ""))}


#: The Modal app the chained launcher merge runs under (``contracts.APP_PURPOSES``
#: names the purpose). The launcher REQUIRES its app name (E40 M7): the
#: command states it with ``--app-name`` and :func:`run_merges` puts the same
#: name into the subprocess environment (``contracts.MODAL_APP_NAME_ENV``),
#: which is where the launcher's App object reads it at import.
MERGE_APP_NAME = C.app_name_for("merge")


def merge_command(sha: str, cell_dirs: Sequence[str], *, launcher: Path = LAUNCHER_SCRIPT,
                  python: str = sys.executable, app_name: str = MERGE_APP_NAME) -> List[str]:
    """The launcher invocation that merges ``cell_dirs`` produced by the
    staged tree ``sha`` (chained by ``--merge`` / ``--merge-sha``, printed by
    ``--no-merge``): that tree EMITS each store's payload, the launcher's own
    tree — this checkout's, ``launcher`` — WRITES (PW10 / E17 A1).
    ``python -m modal`` (the same interpreter this entrypoint runs under,
    i.e. the Modal CLI venv), never a bare ``modal`` that PATH may not carry.
    The launcher refuses an unnamed run, so the command names its app
    (``--app-name``); the printed form carries the environment assignment
    the launcher needs beside it (:func:`merge_command_line`)."""
    if not cell_dirs:
        raise ValueError("merge_command: no cell dirs to merge")
    if not sha:
        raise ValueError("merge_command: no producer sha (a legacy store is never merged)")
    if not app_name:
        raise ValueError("merge_command: the launcher requires an app name (E40 M7)")
    return [python, "-m", "modal", "run", str(launcher), "--sha", sha, "--merge",
            "--merge-cell-dirs", ",".join(cell_dirs), "--app-name", app_name]


def merge_env(cmd: Sequence[str], base: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """The environment a merge command runs under: ``base`` (the process's,
    by default) plus the launcher's app-name variable set to the command's
    own ``--app-name`` — the ONE place the two are tied together."""
    env = dict(os.environ if base is None else base)
    env[C.MODAL_APP_NAME_ENV] = cmd[list(cmd).index("--app-name") + 1]
    return env


def merge_command_line(cmd: Sequence[str]) -> str:
    """The copy-pasteable form of a merge command: the environment assignment
    the launcher needs, then the argv."""
    return f"{C.MODAL_APP_NAME_ENV}={merge_env(cmd, {})[C.MODAL_APP_NAME_ENV]} {' '.join(cmd)}"


def plan_merges(pending: Sequence[Mapping[str, Any]], *, merge_sha: str = "",
                launcher: Path = LAUNCHER_SCRIPT, python: str = sys.executable
                ) -> Dict[str, Any]:
    """The merge plan from the disk-derived pending list: one launcher
    command per producer sha (``commands: {sha: argv}``), the stores refused
    with why (``refused: {label: {dirs, why}}`` — legacy stores without a
    producer block; under ``--merge-sha``, every group whose producer is not
    that sha), and the ``unmergeable`` records. Pure; the entrypoint runs or
    prints the commands."""
    groups = merge_groups(pending)
    commands: Dict[str, List[str]] = {}
    refused: Dict[str, Dict[str, Any]] = {}
    want = merge_sha.strip().lower()
    for sha, dirs in groups.items():
        if sha is LEGACY_PRODUCER:
            refused["<legacy: no producer.tree_sha>"] = {
                "dirs": dirs,
                "why": (f"no {C.MANIFEST_KEY_PRODUCER}.tree_sha in the manifest — the merge "
                        "cannot bind these stores to the tree that produced them; re-run their "
                        f"generation from a staged tree carrying {C.TREE_SHA_FILENAME} (no "
                        "override: merging through another tree's completion.py can silently "
                        "omit a slot)")}
            continue
        if want and not sha.startswith(want):
            refused[sha] = {"dirs": dirs,
                            "why": f"produced by tree {sha}, not --merge-sha {merge_sha!r}"}
            continue
        commands[sha] = merge_command(sha, dirs, launcher=launcher, python=python)
    unmergeable = [r for r in pending if r.get("unmergeable")]
    return {"pending": list(pending), "groups": {str(k): v for k, v in groups.items()},
            "commands": commands, "refused": refused, "unmergeable": unmergeable}


def _tail(s: Any, n: int = 4000) -> str:
    if isinstance(s, bytes):
        s = s.decode("utf-8", "replace")
    s = s or ""
    return s[-n:]


def describe_plan(jobs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """The dry-run document: pending jobs + the exact contract kwargs."""
    return {
        "n_pending": len(jobs),
        "jobs": list(jobs),
        "sandbox": {
            "image": C.SANDBOX_IMAGE_DIGEST,
            "create_kwargs": {k: v for k, v in sandbox_create_kwargs(image="<pinned image>").items()
                              if k != "image"},
            "forbidden_kwargs_absent": list(C.SANDBOX_CREATE_KWARGS_FORBIDDEN),
            "exec_argv": {d: sandbox_exec_argv(d) for d in C.EVALPLUS_DATASETS},
            "env": {d: sandbox_env(d) for d in C.EVALPLUS_DATASETS},
            "transfer_in": [C.SANDBOX_SAMPLES_PATH] + [
                f"{C.SANDBOX_WORKDIR}/{er.dataset_pin(d).gz_filename}" for d in C.EVALPLUS_DATASETS],
            "transfer_out": C.SANDBOX_RESULTS_PATH,
            "validation": "evalplus_runner.validate_eval_results (+ archive md5) before any volume write",
        },
    }


# =========================================================================== #
# E60 (R22) — the additive RE-BIND of the samples digest                       #
# =========================================================================== #
#
# THE DESIGN GAP THIS CLOSES, stated once so the next person hits it here:
# CONVERSION IS GATED ON THE ABSENCE OF THE DONE MARKER AND THE BINDING IS
# WRITTEN BY CONVERSION. ``pending_executions`` skips a task the moment
# ``completion.is_done`` is true — before it looks at results, before it looks
# at the binding — so a task that has already converted can never be converted
# again, and any field the conversion learned to write is unreachable for every
# store already on the volume the moment it lands. That is what happened to
# ``samples_sha256`` (E55 #2): 36 EvalPlus tasks were converted by a tree that
# recorded the digest in ``execution.json`` and not yet in the durable files,
# and no supported path existed to put it there (O60). The next conversion-time
# field needs its re-bind story designed WITH it, not after.
#
# The mode is deliberately narrow: the samples digest, nothing else. It never
# invents a digest (both witnesses must be present and agree), it never
# re-derives a number (the byte-identity assertion below refuses the task
# before writing if ANYTHING but the digest would move), and it preserves
# ``finished_at`` — a re-bind did not re-finish the task — stamping
# ``rebound_at`` instead.


def _json_text(payload: Mapping[str, Any]) -> str:
    """The text :func:`completion.atomic_write_json` would write for
    ``payload`` — the ONE writer of every file this mode touches, so a planned
    document and the bytes that land are the same by construction.

    It CALLS that writer's own ``json.dumps`` (``completion.canonical_json_text``)
    rather than repeating it: the MERGE writer's re-bind predicate (PR23b)
    compares the same documents in bytes, and three copies of one call is how a
    rendering drifts from the bytes on disk."""
    return completion.canonical_json_text(payload)


def _tree_digests(root: Path) -> Dict[str, str]:
    """``{path relative to root: sha256}`` for every file under ``root``."""
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def _key_delta(current: Mapping[str, Any], other: Mapping[str, Any]) -> Dict[str, List[str]]:
    """``{added, removed, changed}`` keys between two JSON documents."""
    return {"added": sorted(set(other) - set(current)),
            "removed": sorted(set(current) - set(other)),
            "changed": sorted(k for k in set(current) & set(other) if current[k] != other[k])}


def rebind_targets(gen_root: str | Path, *, datasets: Sequence[str] = C.EVALPLUS_DATASETS
                   ) -> List[Dict[str, Any]]:
    """Every EvalPlus task under ``gen_root`` the re-bind mode has an opinion
    about, sorted and de-duplicated: the union of the task DIRECTORIES (the
    same ``**/cells/*/evalplus/<dataset>/samples.jsonl`` glob the pending scan
    walks) and the EvalPlus DONE MARKERS.

    The union is the point. Globbing task dirs alone would silently miss a
    converted task whose ``evalplus/<dataset>/`` directory was pruned or lost —
    which is exactly the population E55 #2 is about — and that task must be
    REFUSED out loud (nothing left on disk names its bytes), never overlooked.
    Globbing markers alone would miss the un-converted tasks this mode reports
    as ``skipped``."""
    root = Path(gen_root)
    found: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for samples in root.glob(f"**/cells/*/evalplus/*/{er.SAMPLES_FILENAME}"):
        tdir = samples.parent
        try:
            parts = task_dir_parts(tdir)
        except RuntimeError:
            continue
        if parts["dataset"] not in datasets:
            continue
        cell_dir = tdir.parent.parent
        found[(str(cell_dir), parts["task"])] = {
            "cell_dir": str(cell_dir), "task_dir": str(tdir), **parts}
    for marker in root.glob(f"**/cells/*/{C.DONE_DIR}/*.json"):
        cell_dir = marker.parent.parent
        for dataset in datasets:
            for open_variant in (False, True):
                task = C.evalplus_task(dataset, open_variant=open_variant)
                if C.marker_filename(task) != marker.name:
                    continue
                found.setdefault((str(cell_dir), task), {
                    "cell_dir": str(cell_dir),
                    "task_dir": str(er.served_task_dir(cell_dir, dataset,
                                                       open_variant=open_variant)),
                    "dataset": dataset, "open_variant": open_variant, "task": task})
    return [found[k] for k in sorted(found)]


def _conversion_would_write(cell_dir: Path, *, dataset: str, open_variant: bool
                            ) -> Tuple[Dict[str, str], Dict[str, str]]:
    """``(digests, texts)`` — what the CURRENT converter leaves in a byte-for-
    byte COPY of ``cell_dir``: the sha256 of every file in the copy, and the
    text of the three JSON files a conversion writes.

    The copy is the whole mechanism. Re-implementing the summary recipe here
    would let this mode and the conversion drift apart silently; running the
    REAL ``convert_eval_results_to_store`` against a scratch tree cannot. The
    store on the volume is never opened for writing."""
    with tempfile.TemporaryDirectory(prefix="tinycap-rebind-") as td:
        scratch = Path(td) / "sweep" / "cells" / cell_dir.name
        shutil.copytree(cell_dir, scratch)
        er.convert_eval_results_to_store(scratch, dataset=dataset, open_variant=open_variant,
                                         results=None)
        digests = _tree_digests(scratch)
        task = C.evalplus_task(dataset, open_variant=open_variant)
        texts = {rel: (scratch / rel).read_text(encoding="utf-8")
                 for rel in rebind_relpaths(scratch, dataset=dataset, open_variant=open_variant,
                                            task=task).values()
                 if (scratch / rel).is_file()}
        return digests, texts


def rebind_relpaths(cell_dir: Path, *, dataset: str, open_variant: bool, task: str
                    ) -> Dict[str, str]:
    """The three files a conversion writes into a store, relative to the cell
    directory — and the only three this mode may touch. ``rows.jsonl``,
    ``gen_meta.json``, ``execution.json`` and ``samples.jsonl`` are NOT here
    on purpose."""
    tdir = er.served_task_dir(cell_dir, dataset, open_variant=open_variant)
    return {
        "task_summary": str(Path(tdir.relative_to(cell_dir)) / er.SUMMARY_FILENAME),
        "store_summary": str(Path(C.SUMMARY_DIR) / C.marker_filename(task)),
        "done_marker": str(Path(C.DONE_DIR) / C.marker_filename(task)),
    }


#: The keys this mode may ADD to each :func:`rebind_relpaths` target — the ONLY
#: difference gate (4) tolerates between the file on disk and what the real
#: converter produces. ``rebound_at`` is the mode's own stamp and never the
#: conversion's, so gate (4) also refuses if the conversion writes it.
REBIND_PERMITTED_ADDED_FIELDS: Dict[str, Tuple[str, ...]] = {
    "task_summary": (SAMPLES_DIGEST_FIELD,),
    "store_summary": (SAMPLES_DIGEST_FIELD,),
    "done_marker": (SAMPLES_DIGEST_FIELD, C.DONE_MARKER_REBOUND_AT_FIELD),
}
#: The keys gate (4) holds OUT of the comparison because a re-conversion cannot
#: leave them alone and this mode PRESERVES the value already on disk.
#: ``finished_at`` is the only one: ``finish_task`` stamps ``time.time()`` on
#: every marker write and it is a REQUIRED ``DONE_MARKER_FIELDS`` key.
REBIND_PRESERVED_FIELDS: Dict[str, Tuple[str, ...]] = {
    "task_summary": (), "store_summary": (), "done_marker": ("finished_at",)}


def _rebind_comparable(doc: Mapping[str, Any], *, permitted: Sequence[str],
                       preserved: Sequence[str]) -> str:
    """``doc`` rendered through the store's ONE JSON writer (:func:`_json_text`
    is ``completion.atomic_write_json``'s exact ``json.dumps`` call) with the
    keys this mode may add and the keys it preserves removed.

    Comparing the two SIDES' renderings is what makes "the conversion moves no
    number" mean what it says. Parsed dicts hold ``1 == 1.0`` and
    ``True == 1``, so a value whose TYPE moved passes a dict equality and fails
    this one — the whole point of doing it in bytes (E63)."""
    drop = set(permitted) | set(preserved)
    return _json_text({k: v for k, v in doc.items() if k not in drop})


def _cell_manifest(cell_dir: Path) -> Optional[Mapping[str, Any]]:
    """The cell's ``manifest.json`` parsed, or ``None`` when it is not there,
    does not parse, or is not a JSON object. ``None`` is not a fallback:
    :func:`contracts.store_era` dates an unreadable record
    ``STORE_ERA_UNKNOWN``, which is the verdict a store nobody can date has to
    get, and every caller refuses it."""
    try:
        doc = json.loads((cell_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, Mapping) else None


def _settled_silent_execution(record: Mapping[str, Any], *, cell_dir: Path
                              ) -> Tuple[Optional[Tuple[str, str]], Optional[str]]:
    """E69: a settled store's ``execution.json`` is PRESENT, parses, and names
    no bytes at all — is that age or damage? Returns ``((what, because), None)``
    when it is damage, i.e. TORN, and ``(None, why)`` when it is age, the ONE
    silent shape this mode passes over.

    THE RULE IS THE CONVERTER'S, so the two can never rule opposite ways on one
    file. ``evalplus_runner.samples_binding_of_task`` judges the store's two
    producers separately (E55 #2c) and in this order:

    * the EXECUTION producer, dated by the execution record's OWN marker —
      :func:`contracts.store_era` reads the same two keys off it as off a cell
      manifest, which is why no adapter is needed here. A record that
      positively says it was written under the samples-digest contract and
      recorded no digest is damage whatever the generation's age, and the
      converter RAISES on it ("the digest it did not record is damage rather
      than age");
    * then the GENERATION producer, dated by the CELL MANIFEST.
      :data:`contracts.STORE_ERA_HISTORICAL` — a tree that does not CONTAIN the
      samples-digest contract, so the field it requires could not have been
      written — is the converter's ``historical-unbound``; anything else
      raises, because nothing on that store says its producer tree lacks the
      digest.

    E68 called every silent record age, which is the absence-as-evidence
    inference this branch keeps removing: a hand edit and a partial restore are
    byte-identical to a genuinely pre-digest record, and a POSITIVE marker was
    already there to be read. It is decisive for the current era in particular
    because :func:`execute_one` stamps ``samples_sha256``, ``contract_version``
    and ``producer`` into ONE dict written by ONE atomic write, so a record
    carrying the marker without the digest was never written by this repo at
    all.

    The settled path does NOT consult ``gen_meta.json``, and no reading of it
    would help here: the converter binds from a ``declared`` digest only when
    the GENERATION writer stamped one, which is a shape this branch never
    reaches (a task whose own record names the bytes has a digest to compare,
    not a silence to judge)."""
    era, why_era = C.store_era(record, contract=C.CONTRACT_SAMPLES_DIGEST)
    if era == C.STORE_ERA_CURRENT:
        what = (f"is on disk, names no {SAMPLES_DIGEST_FIELD} at all, and positively says it was "
                f"written under the {C.CONTRACT_SAMPLES_DIGEST} contract ({why_era})")
        because = (
            f"The sandbox stamps {SAMPLES_DIGEST_FIELD} and {C.MANIFEST_KEY_CONTRACT_VERSION} "
            "into ONE record in a single atomic write, so the marker without the digest is a "
            "hand edit or a partial restore and never age. "
            "evalplus_runner.samples_binding_of_task raises on this same file for the same "
            "reason — the digest it did not record is damage rather than age — so passing it "
            "over would leave this mode at odds with the converter on one store (E69).")
        return (what, because), None
    gen_era, why_gen = C.store_era(_cell_manifest(cell_dir),
                                   contract=C.CONTRACT_SAMPLES_DIGEST)
    if gen_era == C.STORE_ERA_HISTORICAL:
        return None, why_gen
    what = (f"is on disk and names no {SAMPLES_DIGEST_FIELD} at all, and neither producer says "
            f"its tree lacks the {C.CONTRACT_SAMPLES_DIGEST} contract: the record says {why_era}, "
            f"and the cell manifest says {why_gen}")
    because = (
        "A record that names no bytes is age only where a producer POSITIVELY says its tree does "
        "not contain the contract; otherwise a torn write, a partial copy and a genuinely "
        "pre-digest record all look the same. evalplus_runner.samples_binding_of_task refuses "
        "this same store in those very words, so passing it over would leave this mode at odds "
        "with the converter (E69).")
    return (what, because), None


def _plan_rebind_settled(out: Mapping[str, Any], *, cell_dir: Path, tdir: Path,
                         marker: Mapping[str, Any], marker_path: Path,
                         dataset: str, open_variant: bool, task: str) -> Dict[str, Any]:
    """Gate 1's second half (E63): the done marker already carries
    ``samples_sha256``, so this task is either SETTLED — nothing to re-bind —
    or TORN, and the marker alone cannot tell them apart.

    The split follows E60's whole point, which is WHERE the digest lives. The
    two copies OUTSIDE the task directory — ``summary/<task>.json`` and the
    done marker — must both be there and must name the same bytes; that pair is
    the durable binding, and nothing legitimate removes or disagrees with it.
    Everything INSIDE ``evalplus/<dataset>/`` — the task summary,
    ``execution.json``, the ``samples.jsonl`` bytes — may be ABSENT, because
    surviving exactly that loss is why the digest was moved out (E55 #2's
    population is a converted task whose task directory is gone); but any of
    them that is still THERE must agree.

    ABSENT means the file is not on disk, and NOTHING ELSE (E68). A witness
    that is there but cannot be READ, or that states something which is not a
    digest, is a statement this mode cannot check — TORN, never silence.
    ``recorded_samples_digest`` folds four states into one ``None``, and
    reading that ``None`` as an absence produced a ``no-op`` whose reason said
    the task directory was gone while ``execution.json`` sat in it, and exit 0
    for a store that falsifies "every witness still inside the task directory
    agrees". So the three that are not an absence are separated, by asking
    ``evalplus_runner.execution_record_of_task`` — the reader that can tell
    them apart — rather than by sniffing the reason string:

    * the file is there and does not parse, or is not a JSON object → TORN;
    * it parses and carries ``samples_sha256``, but the value is not a
      sha256 → TORN;
    * it parses and carries no ``samples_sha256`` at all → the CONVERTER'S OWN
      RULE decides, so the two can never rule opposite ways on one file (E69),
      via :func:`_settled_silent_execution`. ``store_era(record,
      contract=CONTRACT_SAMPLES_DIGEST)`` is asked first: ``current`` — the
      record positively says it was written under the digest contract — is
      TORN, because the sandbox stamps the digest and that marker into one
      dict in a single atomic write and ``samples_binding_of_task`` raises on
      exactly this file. Otherwise the CELL MANIFEST's era for the same
      contract: ``historical`` is the ONE silent shape — a generation whose
      tree does not contain the digest, which the converter calls
      ``historical-unbound`` — and stays a ``no-op`` whose reason says the
      record is THERE and silent rather than reporting a directory that is
      gone; ``unknown`` (and a current generation) is TORN, because nothing on
      the store says its producer tree lacks the digest, which is the
      converter's own refusal. E68 called every silent record age, which read a
      hand edit and a partial restore as an absence.

    ``read_summary`` below already splits absent from torn that way for the two
    summaries (``is_file`` decides absent; a parse failure or a digest that is
    not the marker's is torn), and the merge side's
    ``cell_controls._TORN_WITNESS`` does the same for the same reason.

    So: ``no-op`` when the durable pair agrees and every surviving witness
    agrees with it, naming the ones that are gone. REFUSE, naming the file,
    when the marker carries a digest that ``summary/<task>.json`` or a present
    ``evalplus/<dataset>/summary.json`` lacks, when a present witness cannot be
    read or states a digest that is not a sha256, when a present execution
    record names no bytes and neither producer positively says its tree lacks
    the samples-digest contract, or when any of them names other bytes. This
    repo's writers cannot produce a torn store:
    ``finish_task`` LIFTS the digest out of the summary it is finishing, so a
    marker cannot carry one its summary lacks, and :func:`apply_rebind` writes
    the two summaries FIRST and the marker LAST, so a crash mid-re-bind leaves the
    marker BEHIND — which gate 1 sends down the ordinary re-bind path. Reaching
    a torn state means a hand edit or a restore from a partial copy, and
    reading it as "already bound, nothing to do" would leave a number bound
    that the durable files themselves do not agree on."""
    rels = rebind_relpaths(cell_dir, dataset=dataset, open_variant=open_variant, task=task)
    digest = marker[SAMPLES_DIGEST_FIELD]
    torn: List[str] = []
    lost: List[str] = []

    def read_summary(kind: str, *, may_be_absent: bool) -> None:
        rel = rels[kind]
        path = cell_dir / rel
        if not path.is_file():
            (lost if may_be_absent else torn).append(f"{rel} is not there")
            return
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            torn.append(f"{rel} did not parse ({type(e).__name__}: {str(e)[:120]})")
            return
        if not isinstance(doc, Mapping) or SAMPLES_DIGEST_FIELD not in doc:
            torn.append(f"{rel} carries no {SAMPLES_DIGEST_FIELD}")
        elif doc[SAMPLES_DIGEST_FIELD] != digest:
            torn.append(f"{rel} says {SAMPLES_DIGEST_FIELD}={doc[SAMPLES_DIGEST_FIELD]!r}")

    read_summary("store_summary", may_be_absent=False)      # outside the task dir: required
    read_summary("task_summary", may_be_absent=True)        # inside it: may be gone
    if torn:
        return {**out, "verdict": REBIND_REFUSED, "why": (
            f"{task} @ {cell_dir}: TORN, not settled — {marker_path.name} carries "
            f"{SAMPLES_DIGEST_FIELD}={digest!r} but {'; '.join(torn)}. This repo's writers cannot "
            f"produce that state: finish_task LIFTS the digest out of the summary it finishes, "
            f"and a re-bind writes the two summaries FIRST and the marker LAST, so the marker is "
            f"never ahead of its witnesses. The store was edited by hand or restored from a "
            f"partial copy, and calling it a no-op would leave a number bound that the durable "
            f"files do not agree on. Nothing was written; this store needs a human, not a "
            f"re-bind.")}

    recorded, why_recorded = recorded_samples_digest(tdir)
    silent: List[str] = []
    if recorded is None:
        # E68: `recorded_samples_digest` returns None for four different
        # states, and only ONE of them is an absence. `execution_record_of_task`
        # is the reader that can tell them apart, so it is asked rather than
        # the reason string being sniffed.
        doc, _why_doc = er.execution_record_of_task(tdir)
        # `(what the file did, why that makes it torn)` — the second half is
        # per shape, because the three torn shapes do not share a reason (E69).
        fault: Optional[Tuple[str, str]] = None
        uncheckable = (
            "A witness that is present but unreadable, or that states something which is not a "
            "digest, is a statement this mode cannot check — never the silence of a task "
            "directory that is gone, which is the only case it passes over (E55 #2).")
        if doc is None:
            if (tdir / er.EXECUTION_FILENAME).exists():
                unreadable = ("is on disk and could not be read as an execution record: "
                              f"{why_recorded}")
                fault = (unreadable, uncheckable)
        elif SAMPLES_DIGEST_FIELD in doc:
            not_a_sha = (f"is on disk and records a {SAMPLES_DIGEST_FIELD} that is not a "
                         f"sha256: {why_recorded}")
            fault = (not_a_sha, uncheckable)
        else:
            # Present, readable, naming no bytes. E68 called this age on the
            # strength of the digest being ABSENT; E69: a POSITIVE marker
            # decides it, and the rule is the converter's own so the two cannot
            # disagree about one store.
            fault, why_historical = _settled_silent_execution(doc, cell_dir=cell_dir)
            if why_historical is not None:
                silent.append(
                    f"{er.EXECUTION_FILENAME} is there and names no bytes ({why_recorded}), and "
                    f"the generation that produced the store says its tree does not contain the "
                    f"{C.CONTRACT_SAMPLES_DIGEST} contract either ({why_historical})")
        if fault is not None:
            what, because = fault
            return {**out, "verdict": REBIND_REFUSED, "why": (
                f"{task} @ {tdir}: TORN, not settled — {marker_path.name} and "
                f"{rels['store_summary']} carry {SAMPLES_DIGEST_FIELD}={digest!r}, but "
                f"{er.EXECUTION_FILENAME} {what}. {because} The store was edited by hand or "
                f"restored from a partial copy. Nothing was written; this store needs a human, "
                f"not a re-bind.")}
    samples = tdir / er.SAMPLES_FILENAME
    on_disk = samples_sha256(samples) if samples.is_file() else None
    witnesses = {er.EXECUTION_FILENAME: (recorded, why_recorded),
                 er.SAMPLES_FILENAME: (on_disk, "the file is not there"
                                       if on_disk is None else "hashed")}
    conflict = sorted(n for n, (v, _w) in witnesses.items() if v is not None and v != digest)
    if conflict:
        return {**out, "verdict": REBIND_REFUSED, "why": (
            f"{task} @ {tdir}: the durable files agree on {SAMPLES_DIGEST_FIELD}={digest!r}, but "
            + ", ".join(f"{n} says {witnesses[n][0]!r}" for n in conflict)
            + ". A binding that contradicts bytes still sitting beside it is not something to "
              "pass over as already done — the samples were rewritten after these results, or the "
              "store was restored from a partial copy. Nothing was written.")}

    lost += [f"{n} names nothing ({witnesses[n][1]})" for n in sorted(witnesses)
             if witnesses[n][0] is None and not (silent and n == er.EXECUTION_FILENAME)]
    # The two are reported apart on purpose (E68): "expected once the task
    # directory is gone" is true of a witness that is GONE and false of one
    # sitting in that directory saying nothing.
    tail = ""
    if lost:
        tail += (". " + "; ".join(lost) + " — expected once the task directory is gone, which is "
                 "exactly the loss this binding was moved outside it to survive")
    if silent:
        tail += (". " + "; ".join(silent) + " — an execution record written before the digest "
                 "existed states no bytes and so contradicts none; the durable pair outside the "
                 "task directory is what binds these numbers")
    return {**out, "verdict": REBIND_NOOP, SAMPLES_DIGEST_FIELD: digest, "why": (
        f"{marker_path} and {rels['store_summary']} already carry {SAMPLES_DIGEST_FIELD}="
        f"{digest!r}; nothing to re-bind"
        + (tail or ", and every witness beside them agrees"))}


def plan_rebind(target: Mapping[str, Any]) -> Dict[str, Any]:
    """One task's re-bind verdict, and — for a ``rebind`` — the exact documents
    it would write and the unified diff it would apply. Writes nothing.

    THE DESIGN GAP (E60), in the reviewer's words: CONVERSION IS GATED ON THE
    ABSENCE OF THE DONE MARKER AND THE BINDING IS WRITTEN BY CONVERSION, so any
    field added at conversion time is unreachable for every already-converted
    task the moment it lands — the next conversion-time field needs its re-bind
    story designed with it. This mode is that story for ``samples_sha256`` and
    for nothing else.

    Four gates, in order:

    1. the done marker must EXIST (else ``skipped`` — an unconverted task
       belongs to the sandbox stage) and must lack ``samples_sha256``. A marker
       that carries one goes to :func:`_plan_rebind_settled`, which calls it a
       ``no-op`` only when ``summary/<task>.json`` carries the SAME digest and
       every witness still inside the task directory agrees, and REFUSES the
       torn shapes in between rather than passing them over (E63). A witness
       inside the task directory may be GONE — that loss is what the durable
       pair exists to survive — but one that is there and cannot be read, or
       that states a digest which is not a sha256, is torn like any other
       disagreement (E68), and one that is there and names no bytes at all is
       judged by the ERA rule the converter uses, never by the absence of the
       digest (E69);
    2. the digest is read from BOTH witnesses — ``execution.json``'s
       ``samples_sha256`` and the sha256 of the ``samples.jsonl`` bytes — and a
       missing one or a disagreement REFUSES, naming both values. A re-bind
       never invents a digest;
    3. the would-be files are the REAL converter's output against a scratch
       COPY of the store (:func:`_conversion_would_write`), so this mode cannot
       drift from the conversion it is standing in for;
    4. before anything is written, the only differences from what is on disk
       must be the added ``samples_sha256`` on the three files and — on the
       done marker, whose ``finished_at`` the conversion necessarily rewrites —
       that timestamp, which is PRESERVED here rather than carried over. The
       assertion is in BYTES on both halves of the store, and the two halves
       are checked differently: every file OUTSIDE the three targets by sha256
       (``rows.jsonl`` included — a re-bind that appended a row would be a
       re-conversion), and the three targets THEMSELVES by rendering both the
       file on disk and the conversion's output through the store's one JSON
       writer with exactly the permitted added keys and ``finished_at``
       removed, then comparing the text. Not a dict equality: parsed dicts hold
       ``1 == 1.0`` and ``True == 1``, so a number whose type moved would pass
       a check whose entire job is "changes no number" (E63). Any other
       differing byte REFUSES the task before a single write."""
    cell_dir, tdir = Path(target["cell_dir"]), Path(target["task_dir"])
    dataset, open_variant, task = target["dataset"], target["open_variant"], target["task"]
    out: Dict[str, Any] = {"cell_dir": str(cell_dir), "task_dir": str(tdir),
                           "dataset": dataset, "task": task}

    def refuse(why: str) -> Dict[str, Any]:
        return {**out, "verdict": REBIND_REFUSED, "why": why}

    marker_path = completion.done_marker_path(cell_dir, task)
    if not marker_path.is_file():
        return {**out, "verdict": REBIND_SKIPPED,
                "why": (f"no done marker at {marker_path} — this mode re-binds a task the "
                        "conversion already finished; an unconverted one is the sandbox and "
                        "convert stages' business")}
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return refuse(f"{marker_path} did not parse ({type(e).__name__}: {str(e)[:160]})")
    if not isinstance(marker, Mapping):
        return refuse(f"{marker_path} is not a JSON object")
    if SAMPLES_DIGEST_FIELD in marker:
        return _plan_rebind_settled(out, cell_dir=cell_dir, tdir=tdir, marker=marker,
                                    marker_path=marker_path, dataset=dataset,
                                    open_variant=open_variant, task=task)

    # (2) BOTH witnesses, or nothing
    samples = tdir / er.SAMPLES_FILENAME
    recorded, why_recorded = recorded_samples_digest(tdir)
    on_disk = samples_sha256(samples) if samples.is_file() else None
    if recorded is None or on_disk is None:
        return refuse(
            f"{task} @ {tdir}: a re-bind never invents a digest, and one of its two witnesses is "
            f"missing — {er.EXECUTION_FILENAME} says {recorded!r} ({why_recorded}) and "
            f"{er.SAMPLES_FILENAME} hashes to {on_disk!r} "
            f"({'the file is not there' if on_disk is None else 'read'}). Nothing was written. "
            f"Re-run the sandbox stage for this task (remove {er.EVAL_RESULTS_FILENAME} and "
            f"{er.EXECUTION_FILENAME} from {tdir}) so it executes and converts afresh.")
    if recorded != on_disk:
        return refuse(
            f"{task} @ {tdir}: the two digest witnesses disagree — {er.EXECUTION_FILENAME} records "
            f"{recorded} while {er.SAMPLES_FILENAME} now hashes to {on_disk}. The samples were "
            "rewritten after those results, so binding this number to either value would be a "
            "guess. Nothing was written. Re-run the sandbox stage for this task (remove "
            f"{er.EVAL_RESULTS_FILENAME} and {er.EXECUTION_FILENAME} from {tdir}).")
    digest = recorded

    # (3) what the REAL converter would write, against a scratch copy
    rels = rebind_relpaths(cell_dir, dataset=dataset, open_variant=open_variant, task=task)
    current_digests = _tree_digests(cell_dir)
    absent = sorted(rel for rel in rels.values() if rel not in current_digests)
    if absent:
        return refuse(
            f"{task} @ {cell_dir}: the conversion would CREATE {absent}, which is not a re-bind — "
            "a store whose durable summary is missing has more wrong with it than an unbound "
            "number. Nothing was written.")
    try:
        produced_digests, produced_texts = _conversion_would_write(
            cell_dir, dataset=dataset, open_variant=open_variant)
    except Exception as e:  # noqa: BLE001 — every conversion refusal is this task's, not the run's
        return refuse(
            f"{task} @ {cell_dir}: re-running the conversion against a scratch COPY of the store "
            f"raised {type(e).__name__}: {str(e)[:400]}. Nothing was written (the copy is "
            "discarded); whatever the conversion objects to has to be fixed first.")

    # (4) the byte-identity assertion, BEFORE any write
    allowed = set(rels.values())
    tree_delta = _key_delta(current_digests, produced_digests)
    stray = sorted((set(tree_delta["added"]) | set(tree_delta["removed"])
                    | set(tree_delta["changed"])) - allowed)
    if stray:
        return refuse(
            f"{task} @ {cell_dir}: the conversion would also touch {stray} — a re-bind writes the "
            f"samples digest into {sorted(allowed)} and nothing else, so a moved row, gen_meta, "
            "execution record or sample file means this store needs a re-conversion, not a "
            "re-bind. Nothing was written.")

    unproduced = sorted(rel for rel in rels.values() if rel not in produced_texts)
    if unproduced:
        return refuse(
            f"{task} @ {cell_dir}: the conversion left {unproduced} unwritten in the scratch copy, "
            "so there is nothing to compare against. Nothing was written.")

    planned: Dict[str, Any] = {}
    for kind in ("task_summary", "store_summary", "done_marker"):
        rel = rels[kind]
        permitted = REBIND_PERMITTED_ADDED_FIELDS[kind]
        preserved = REBIND_PRESERVED_FIELDS[kind]
        current_text = (cell_dir / rel).read_text(encoding="utf-8")
        current_doc = json.loads(current_text)
        produced_doc = json.loads(produced_texts[rel])

        # (4a) the file on disk is in the store's ONE canonical form. Every
        # byte of it was written by `atomic_write_json`, so a file that does
        # not render back to itself was written by something else — and a
        # comparison in bytes has to start from bytes that mean something.
        if current_text != _json_text(current_doc):
            return refuse(
                f"{task} @ {cell_dir}/{rel}: the file on disk is not in the store's canonical "
                f"JSON form — it does not render back to its own bytes through "
                f"completion.atomic_write_json, the one writer every file this mode touches went "
                f"through. Something other than the conversion wrote it, so there is no byte "
                f"comparison to make. Nothing was written.")

        # (4b) a permitted key ALREADY on disk must hold the value this mode
        # would give it. `samples_sha256` equal to the witnessed digest is the
        # legitimate resume shape (a re-bind whose summaries landed and whose
        # marker did not); a DIFFERENT digest, or a `rebound_at` on a marker
        # that carries no digest, is torn and cannot come from these writers.
        stale = sorted(k for k in permitted if k in current_doc
                       and (k != SAMPLES_DIGEST_FIELD or current_doc[k] != digest))
        if stale:
            return refuse(
                f"{task} @ {cell_dir}/{rel}: the file already carries "
                f"{ {k: current_doc[k] for k in stale} } while the two witnesses say "
                f"{SAMPLES_DIGEST_FIELD}={digest} and the done marker carries no digest at all. "
                f"No writer here produces that: a re-bind writes the summaries first and the "
                f"marker last, and it writes {C.DONE_MARKER_REBOUND_AT_FIELD} only together with "
                f"the digest. Nothing was written; this store needs a human, not a re-bind.")

        # (4c) the conversion may not derive a digest other than the witnessed
        # one, and `rebound_at` is this mode's stamp alone.
        if produced_doc.get(SAMPLES_DIGEST_FIELD) != digest:
            return refuse(
                f"{task} @ {cell_dir}/{rel}: the conversion derived "
                f"{SAMPLES_DIGEST_FIELD}={produced_doc.get(SAMPLES_DIGEST_FIELD)!r} while both "
                f"witnesses say {digest}. A re-bind binds the witnessed bytes or nothing. "
                f"Nothing was written.")
        if C.DONE_MARKER_REBOUND_AT_FIELD in produced_doc:
            return refuse(
                f"{task} @ {cell_dir}/{rel}: the conversion itself wrote "
                f"{C.DONE_MARKER_REBOUND_AT_FIELD}, which is this mode's stamp and no converter's "
                f"— the marker contract changed under this mode. Nothing was written.")

        # (4d) THE assertion, in BYTES: with exactly the permitted added keys
        # and the preserved keys removed from BOTH sides, the conversion's
        # output must render identically to what is on disk. A dict equality
        # here would pass `1` where the conversion now derives `1.0`, or `True`
        # where it derives `1` — a check whose whole job is "changes no
        # number", blind to a number changing type (E63).
        current_cmp = _rebind_comparable(current_doc, permitted=permitted, preserved=preserved)
        produced_cmp = _rebind_comparable(produced_doc, permitted=permitted, preserved=preserved)
        if current_cmp != produced_cmp:
            return refuse(
                f"{task} @ {cell_dir}/{rel}: the conversion would change more than "
                f"{sorted(permitted)} — with those keys and {sorted(preserved)} removed from both "
                f"sides, its output does not render BYTE FOR BYTE to what is on disk. Key delta "
                f"{_key_delta(current_doc, produced_doc)}; a value whose TYPE moved (1 -> 1.0) "
                f"shows in the diff below and would have passed a comparison of parsed dicts. "
                f"A re-bind may not move a number. Nothing was written.\n"
                + "\n".join(difflib.unified_diff(
                    current_cmp.splitlines(), produced_cmp.splitlines(),
                    fromfile=f"a/{rel} (on disk)",
                    tofile=f"b/{rel} (the conversion's output)", lineterm="")))

        # `finished_at` is PRESERVED rather than carried over from the scratch
        # conversion: a re-bind did not re-finish the task, so a marker
        # claiming a new finish time would assert something false (E60).
        planned[rel] = (produced_doc if kind != "done_marker" else
                        {**current_doc, SAMPLES_DIGEST_FIELD: digest,
                         C.DONE_MARKER_REBOUND_AT_FIELD: time.time()})

    diff = "\n".join(
        "\n".join(difflib.unified_diff(
            _json_text(json.loads((cell_dir / r).read_text(encoding="utf-8"))).splitlines(),
            _json_text(planned[r]).splitlines(),
            fromfile=f"a/{r}", tofile=f"b/{r}", lineterm=""))
        for r in (rels["task_summary"], rels["store_summary"], rels["done_marker"]))
    return {**out, "verdict": REBIND_REBIND, SAMPLES_DIGEST_FIELD: digest,
            "why": (f"the done marker carries no {SAMPLES_DIGEST_FIELD}; "
                    f"{er.EXECUTION_FILENAME} and the {er.SAMPLES_FILENAME} bytes both say "
                    f"{digest}, and the conversion would add that and nothing else"),
            "relpaths": rels, "writes": planned, "diff": diff}


def apply_rebind(plan: Mapping[str, Any]) -> List[str]:
    """Write one ``rebind`` plan's three documents — atomically, through the
    same writers the conversion uses (``completion.atomic_write_json`` and
    ``completion.write_done_marker``, which re-checks the marker's field set),
    the marker LAST so a reader never sees a marker whose summaries have not
    caught up. Refuses any other verdict."""
    if plan.get("verdict") != REBIND_REBIND:
        raise RuntimeError(f"apply_rebind: {plan.get('task')!r} is {plan.get('verdict')!r}, "
                           f"not {REBIND_REBIND!r}")
    cell_dir, task = Path(plan["cell_dir"]), plan["task"]
    rels, writes = plan["relpaths"], plan["writes"]
    written: List[str] = []
    for kind in ("task_summary", "store_summary", "done_marker"):
        rel = rels[kind]
        if kind == "done_marker":
            written.append(str(completion.write_done_marker(cell_dir, task, writes[rel])))
        else:
            written.append(str(completion.atomic_write_json(cell_dir / rel, writes[rel])))
    return written


def run_rebind(gen_root: str | Path, *, datasets: Sequence[str] = C.EVALPLUS_DATASETS,
               dry_run: bool = True, commit=None) -> Dict[str, Any]:
    """Plan (and, unless ``dry_run``, apply) the samples-digest re-bind over
    every EvalPlus task under ``gen_root``. Returns
    ``{gen_root, datasets, dry_run, n_tasks, counts, tasks}``; ``counts`` is
    keyed by :data:`REBIND_VERDICTS`. A refusal is per task — the other tasks
    are still re-bound — and the caller exits non-zero on any.

    ONE volume commit, at the end, on purpose: a container that dies mid-run
    loses its uncommitted writes, so the volume is left exactly as it was and
    the operator simply re-runs. (A run that DID commit and was then re-run is
    equally safe — a re-bound task reads as ``no-op``, and a task whose
    summaries landed while its marker did not re-plans to the same write.)"""
    targets = rebind_targets(gen_root, datasets=datasets)
    tasks: List[Dict[str, Any]] = []
    for target in targets:
        plan = plan_rebind(target)
        if plan["verdict"] == REBIND_REBIND and not dry_run:
            plan = {**plan, "written": apply_rebind(plan)}
        tasks.append(plan)
        label = plan["verdict"].upper()
        print(f"[tinycap-sandbox] REBIND {label}: {plan['task']} @ {plan['cell_dir']}"
              + (f" — {plan['why']}" if plan.get("why") else ""))
        if dry_run and plan.get("diff"):
            print(plan["diff"])
        if plan.get("written"):
            print(f"[tinycap-sandbox]   wrote {plan['written']}")
    if commit is not None and not dry_run and any(t["verdict"] == REBIND_REBIND for t in tasks):
        commit()
    counts = {v: sum(1 for t in tasks if t["verdict"] == v) for v in REBIND_VERDICTS}
    print(f"[tinycap-sandbox] re-bind {'DRY RUN ' if dry_run else ''}over {gen_root}: "
          + ", ".join(f"{counts[v]} {v}" for v in REBIND_VERDICTS))
    print("[tinycap-sandbox] the durable witness ceiling for this population is 5 of 6: "
          f"{er.GEN_META_FILENAME}'s digest is stamped by the GENERATION writer and no conversion "
          "path writes that file, so 6 of 6 would need a re-generation, which re-derives the "
          "numbers (E60). A re-merge afterwards is what adds the payload witness — a PLAIN one: "
          "the merge writer recognises this re-bind and files it without --replace-source (E64).")
    return {"gen_root": str(gen_root), "datasets": list(datasets), "dry_run": bool(dry_run),
            "n_tasks": len(tasks), "counts": counts, "tasks": tasks}


# =========================================================================== #
# Modal functions                                                              #
# =========================================================================== #


@app.function(image=orchestrator_image, volumes={VOL_PATH: VOL}, cpu=2, memory=4096,
              timeout=4 * 3600, name="tinycap-stage-evalplus-datasets")
def stage_evalplus_datasets() -> Dict[str, Any]:
    """Download the two pinned release archives into the volume cache and
    verify their sha256 (the orchestrator's own trusted input)."""
    import urllib.request

    VOL.reload()
    out: Dict[str, Any] = {}
    Path(EVALPLUS_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    for dataset in C.EVALPLUS_DATASETS:
        pin = er.dataset_pin(dataset)
        dst = staged_archive_path(dataset)
        if dst.exists():
            try:
                verify_archive(dataset, dst)
                out[dataset] = {"path": str(dst), "status": "present"}
                continue
            except RuntimeError as e:
                print(f"[tinycap-sandbox] re-staging {dst}: {e}")
        with urllib.request.urlopen(pin.url, timeout=120) as r:
            data = r.read()
        sha = hashlib.sha256(data).hexdigest()
        if sha != pin.gz_sha256 or len(data) != pin.gz_bytes:
            raise RuntimeError(
                f"{dataset}: downloaded {pin.url} sha256 {sha} / {len(data)} bytes != pinned "
                f"{pin.gz_sha256} / {pin.gz_bytes} — refusing to stage a drifted release")
        tmp = dst.with_name(dst.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, dst)
        out[dataset] = {"path": str(dst), "status": "staged", "sha256": sha, "bytes": len(data)}
    VOL.commit()
    return out


@app.function(image=orchestrator_image, volumes={VOL_PATH: VOL}, cpu=2, memory=4096,
              timeout=6 * 3600, name=C.CONTAINER_SANDBOX_EXECUTE)
def tinycap_sandbox_execute(gen_root: str = C.VOL_GEN_ROOT, datasets: Sequence[str] = C.EVALPLUS_DATASETS,
                            dry_run: bool = False, max_parallel: int = 8) -> Dict[str, Any]:
    """Execute every pending served EvalPlus generation (module docstring);
    ``dry_run`` only lists them with the contract kwargs. Independent jobs
    run concurrently, one sandbox each."""
    VOL.reload()
    archives: Dict[str, str] = {}
    for d in datasets:
        try:
            verify_archive(d, staged_archive_path(d))
            archives[d] = "present"
        except RuntimeError as e:
            archives[d] = f"UNAVAILABLE: {e}"
    jobs = pending_executions(gen_root, datasets=datasets)
    plan = describe_plan(jobs)
    plan["archives"] = archives
    # E12 item 2: what is done but not yet merged, from DISK (before this run's
    # conversions — the dry run shows the backlog; re-read after them below)
    plan["merge_pending"] = unmerged_completed_stores(gen_root, datasets=datasets)
    print(json.dumps(plan, indent=1))
    if dry_run or not jobs:
        return {"dry_run": dry_run, "plan": plan, "records": [],
                "merge_pending": plan["merge_pending"]}
    bad = [d for d, st in archives.items() if st != "present"]
    if bad:  # a dry run only REPORTS this; an execution refuses before any sandbox
        raise RuntimeError(f"staged archive(s) unavailable for {bad}: run --stage-datasets first")

    def one(job):
        try:
            return {"job": job, "ok": True,
                    "record": execute_one(job, sandbox_cls=modal.Sandbox, app_obj=app,
                                          image=sandbox_image, commit=VOL.commit)}
        except Exception as e:  # noqa: BLE001 — one task's failure never hides the others'
            return {"job": job, "ok": False, "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc()[-4000:]}

    with ThreadPoolExecutor(max_workers=max(1, int(max_parallel))) as ex:
        records = list(ex.map(one, jobs))
    VOL.commit()
    failed = [r for r in records if not r["ok"]]
    for r in failed:
        print(f"[tinycap-sandbox] FAILED {r['job']['task']} @ {r['job']['task_dir']}: {r['error']}")
    print(f"[tinycap-sandbox] {len(records) - len(failed)}/{len(records)} executions succeeded")
    # the merge-pending list AFTER this run's conversions, from disk — the
    # conversions above are a subset of it, and so is anything an earlier
    # run converted and never merged
    pending = unmerged_completed_stores(gen_root, datasets=datasets)
    print(f"[tinycap-sandbox] {len(pending)} store(s) hold DONE EvalPlus markers not accounted "
          f"for in controls_summary.json: {[p['cell_dir'] for p in pending]}")
    for p in pending:
        for task, why in (p.get("why") or {}).items():
            print(f"[tinycap-sandbox]   {p['cell_dir']} {task}: {why}")
    return {"dry_run": False, "plan": plan, "records": records, "merge_pending": pending}


@app.function(image=orchestrator_image, volumes={VOL_PATH: VOL}, cpu=2, memory=4096,
              timeout=4 * 3600, name=REBIND_CONTAINER_NAME)
def tinycap_sandbox_rebind(gen_root: str = C.VOL_GEN_ROOT,
                           datasets: Sequence[str] = C.EVALPLUS_DATASETS,
                           dry_run: bool = True) -> Dict[str, Any]:
    """E60: write the samples digest into the durable files of every EvalPlus
    task under ``gen_root`` that already converted without one. No sandbox, no
    GPU, no conversion — see :func:`plan_rebind` for the four gates and the
    design gap this closes.

    ``dry_run`` defaults TRUE here and in :func:`run_rebind` — a function that
    writes to the volume defaults to not writing — while the entrypoint always
    passes the operator's own ``--dry-run``, so ``--rebind-samples-digest``
    alone is a real pass and ``--rebind-samples-digest --dry-run`` is not."""
    VOL.reload()
    doc = run_rebind(gen_root, datasets=datasets, dry_run=dry_run, commit=VOL.commit)
    print(json.dumps({k: v for k, v in doc.items() if k != "tasks"}, indent=1))
    return doc


def merge_decision(merge_mode: str, merge_sha: str, *, dry_run: bool) -> bool:
    """The entrypoint's up-front merge decision → ``chain`` (run the
    per-producer commands, or only print them). Refuses (``SystemExit``) a
    mode outside :data:`MERGE_MODES`, the print mode beside ``--merge-sha``
    (contradictory), and — for a non-dry run — no decision at all: a
    completed conversion must never fall silent (N1)."""
    mode = (merge_mode or "").strip().lower()
    if mode and mode not in MERGE_MODES:
        sys.exit(f"{MERGE_MODE_FLAG} {merge_mode!r} is not one of {list(MERGE_MODES)}")
    if mode == MERGE_MODE_PRINT and merge_sha:
        sys.exit(f"{MERGE_MODE_FLAG} {MERGE_MODE_PRINT} prints the merge commands while "
                 f"{MERGE_SHA_FLAG} runs one — pick one")
    if not dry_run and not (mode or merge_sha):
        # N1: a completed conversion must reach controls_summary.json, and the
        # merge runs in the launcher's client venv — decide up front how
        sys.exit(f"decide the controls_summary merge up front: {MERGE_MODE_FLAG} {MERGE_MODE_CHAIN} "
                 "(chain it per producing tree: modal run scripts/modal/tinycap_served.py --sha "
                 f"<producer sha> --merge --merge-cell-dirs ...), {MERGE_SHA_FLAG} <sha> (the same, "
                 f"refusing any store another tree produced), or {MERGE_MODE_FLAG} {MERGE_MODE_PRINT} "
                 "(print the commands; you run them)")
    return mode == MERGE_MODE_CHAIN or bool(merge_sha)


def run_merges(plan: Mapping[str, Any], *, chain: bool, runner=None) -> Dict[str, str]:
    """Run (``chain``) or print (``--merge-mode no-merge``) every command of a merge
    plan; ``{sha: "OK" | "FAILED (rc N)" | "PENDING — run the command"}``.
    ``runner`` is injected by the tests (production: ``subprocess.run`` from
    the repo root)."""
    runner = runner or (lambda cmd: subprocess.run(cmd, cwd=str(LAUNCHER_SCRIPT.parents[2]),
                                                   env=merge_env(cmd), check=False))
    status: Dict[str, str] = {}
    for sha, cmd in plan["commands"].items():
        n = len(plan["groups"][sha])
        if chain:
            print(f"[tinycap-sandbox] merging {n} store(s) produced by {sha[:12]}: "
                  f"{merge_command_line(cmd)}")
            r = runner(cmd)
            status[sha] = "OK" if r.returncode == 0 else f"FAILED (rc {r.returncode})"
        else:
            status[sha] = "PENDING — run the command"
            print(f"[tinycap-sandbox] MERGE PENDING for {n} store(s) produced by {sha[:12]} — run:\n"
                  f"  {merge_command_line(cmd)}")
    return status


@app.local_entrypoint()
def main(gen_root: str = C.VOL_GEN_ROOT, datasets: str = ",".join(C.EVALPLUS_DATASETS),
         dry_run: bool = False, stage_datasets: bool = False, max_parallel: int = 8,
         offline_contract: bool = False, out: str = "", merge_mode: str = "", merge_sha: str = "",
         rebind_samples_digest: bool = False):
    ds = [d.strip() for d in datasets.split(",") if d.strip()]
    for d in ds:
        if d not in C.EVALPLUS_DATASETS:
            sys.exit(f"unknown dataset {d!r}; datasets: {list(C.EVALPLUS_DATASETS)}")
    if offline_contract:
        doc = describe_plan([])
        print(json.dumps(doc, indent=1))
        if out:
            Path(out).write_text(json.dumps(doc, indent=1))
        return
    if rebind_samples_digest:
        # E60: the re-bind runs ALONE. It converts nothing, so a merge decision
        # has nothing to decide, and combining it with the execute path would
        # hide which of the two wrote what.
        clash = [f for f, on in ((MERGE_MODE_FLAG, merge_mode), (MERGE_SHA_FLAG, merge_sha),
                                 ("--stage-datasets", stage_datasets)) if on]
        if clash:
            sys.exit(f"{REBIND_FLAG} runs alone — it converts nothing and merges nothing; "
                     f"drop {clash} and run the merge afterwards "
                     "(scripts/modal/tinycap_served.py --sha <producer sha> --merge ...), which "
                     "is what adds the payload witness on top of the re-bound summary")
        doc = tinycap_sandbox_rebind.remote(gen_root=gen_root, datasets=ds, dry_run=dry_run)
        if out:
            Path(out).write_text(json.dumps(doc, indent=1, default=str))
            print(f"[tinycap-sandbox] result -> {out}")
        n_refused = doc["counts"][REBIND_REFUSED]
        if n_refused:
            sys.exit(f"{n_refused} task(s) REFUSED the samples-digest re-bind — see the REBIND "
                     "REFUSED lines above; nothing was written for them, and the tasks that were "
                     f"re-bound ({doc['counts'][REBIND_REBIND]}) stand")
        return
    chain = merge_decision(merge_mode, merge_sha, dry_run=dry_run)
    if stage_datasets:
        print(json.dumps(stage_evalplus_datasets.remote(), indent=1))
    result = tinycap_sandbox_execute.remote(gen_root=gen_root, datasets=ds, dry_run=dry_run,
                                            max_parallel=max_parallel)
    # E12 items 2 + 3: the merge candidates are EVERY done-but-unmerged store on
    # disk (not just this run's conversions), grouped by the tree that produced them
    plan = plan_merges(result["merge_pending"], merge_sha=merge_sha)
    result["merge"] = {**plan, "converted_this_run": converted_cell_dirs(result["records"]),
                       "status": {}}
    if not plan["pending"]:
        result["merge"]["status"] = "nothing to merge"
    elif not dry_run:
        result["merge"]["status"] = run_merges(plan, chain=chain)
    else:
        result["merge"]["status"] = "dry run: nothing merged"
    for label, ref in plan["refused"].items():
        print(f"[tinycap-sandbox] MERGE REFUSED for {len(ref['dirs'])} store(s) [{label}]: {ref['why']}")
    for rec in plan["unmergeable"]:
        print(f"[tinycap-sandbox] UNMERGEABLE {rec['cell_dir']}: {rec['unmergeable']}")
    if out:
        Path(out).write_text(json.dumps(result, indent=1, default=str))
        print(f"[tinycap-sandbox] result -> {out}")
    failed = [r for r in result["records"] if not r["ok"]]
    if failed:
        sys.exit(f"{len(failed)} execution(s) failed — see above; the failed tasks stay pending")
    if dry_run:
        return
    status = result["merge"]["status"]
    bad = ({sha: st for sha, st in status.items() if st.startswith("FAILED")}
           if isinstance(status, dict) else {})
    if bad:
        sys.exit("the controls_summary merge failed for " + "; ".join(
            f"{sha[:12]} ({st}) — re-run: {' '.join(plan['commands'][sha])}" for sha, st in bad.items()))
    if plan["refused"] or plan["unmergeable"]:
        sys.exit(f"{sum(len(r['dirs']) for r in plan['refused'].values()) + len(plan['unmergeable'])} "
                 "completed store(s) could not be merged — see MERGE REFUSED / UNMERGEABLE above; "
                 "they stay pending on disk")
