"""The Petri steered-serving smoke on Modal — the serving-only certification of
the vLLM lens path (scripts/petri_steered_smoke.py; read its docstring for what
is proved and how verdicts are decided), ported from
scripts/sbatch/petri_steered_smoke.sbatch after the Slurm clusters went away
(2026-09-05). The sbatch is the specification of this file: everything it set
up is reproduced here, and every departure is named under WHAT IS WEAKER at the
bottom. The smoke script, the sbatch, the serving stack
(src/sisteer/capeval/serving/{vllm_server,proxy}.py) and
scripts/modal/petri_agentloop_control.py are FROZEN — this file drives them,
never edits them.

WHAT RUNS. One H200 container. Inside it, EXACTLY the sbatch's command
(``smoke_argv``, compared flag-for-flag against the sbatch by
tests/test_modal_petri_smoke.py AND by the preflight against the staged tree's
own copy of the sbatch): the tree's ``scripts/petri_steered_smoke.py smoke``
with the petri_twosi manifest, meta-llama/Llama-3.1-8B-Instruct at the pinned
revision, the two REAL objects (static_r_matched at lambda 0.05, alphasteer
official), the response arm on, the gating phases at VLLM_BATCH_INVARIANT=0
(the production configuration) and the BI=1 evidence probe on. The sbatch's
``srun --ntasks=1 uv run --no-sync python`` launcher prefix is Slurm/venv
plumbing and is the only part not reproduced: the image IS the venv, so the
container's python runs the script directly, from the tree as its cwd (the
sbatch ``cd``s into the checkout; the manifest and script paths are relative).

TREE. The smoke insists on an immutable commit-addressed checkout (the sbatch
verifies HEAD == sha and no modified tracked file). There is no cluster mirror
any more and the repo is private (no credentials go on Modal), so ``--stage``
builds a SHALLOW clone of the sha from the LOCAL repo, tars it, pushes the
tarball over the Modal client and ``unpack_tree`` extracts + verifies it at
``/vol/repos/sisteer-<sha>`` (the convention shared with
scripts/modal/lens_vectorize_gate.py and cpu_suite.py — the helpers are
duplicated from lens_vectorize_gate.py rather than imported: Modal re-imports
the launch module flat inside the container, where a sibling import would not
resolve, and importing that module locally would construct its own App).
The GPU container copies the tree to local disk and verifies it AGAIN before
the smoke starts, exactly as the sbatch did.

IMAGE. The sbatch's serving venv was ``uv pip install --index-strategy
unsafe-best-match --exclude-newer 2026-08-03T00:00:00Z -r
configs/capeval/vllm-requirements.txt`` on python 3.12. The image here is built
by the lens gate's certified recipe from the same input under the same cutoff
(``_serving_image``, copied from lens_vectorize_gate.py so Modal's image cache
is shared): the sha256-pinned vLLM 0.25.1+cu129 wheel, the cu129 torch 2.11.0
stack, vllm-lens 1.2.1, plus matplotlib. Default variant ``tf5141``
(transformers==5.14.1 pinned explicitly): configs/capeval/geom-sweep-
requirements-tf5141.txt records that the cluster's Petri venvs ran at 5.14.1,
and 5.15.0 was published 2026-08-10, after the cutoff. (``uv pip compile`` of
vllm-requirements.txt under the cutoff resolves 5.15.0 today, which the upload
date says it should not; the documented cluster fact wins, and ``--image
tf5150`` is one flag away.) The smoke's own package cross-check compares vllm /
vllm-lens / torch against the manifest's serving_packages, never transformers.

INPUTS ON THE VOLUME (``twosi-fits`` at ``/vol``): the HF cache at ``/vol/hf``
(the pinned meta-llama snapshot is present; ``HF_HUB_OFFLINE=1``), and the two
REQUIRED objects. Where they live is the site's to say: the sbatch serves the
site's SISTEER_PETRI_SMOKE_STATIC_R_OBJECT and SISTEER_PETRI_SMOKE_ALPHASTEER_OBJECT
from its job environment, and this launcher reads the same two keys (the
environment, then the site config) on the grading host and expects each object
at its cluster path's image under ``/vol/objects/jailbreak/``
(``object_on_volume``):

    the setting: $SHARED/exp/jailbreak/<rel>
       on Modal: /vol/objects/jailbreak/<rel>

A value that already names a ``/vol/`` path is taken as it is; nothing names
either object by default, so with neither the flag nor the key the launcher
refuses by name before anything is staged, preflighted or spawned.

On 2026-09-05 neither object was on the volume (the cluster copies went with
the cluster) and the preflight refused with exactly those paths; on 2026-09-06
the originals were placed there from a grading host copy hash-verified against the
2026-08-25 cluster inventory, and the first Modal canary ran (THE CANARY,
below). The preflight (mandatory before any GPU: ``--pf``, and always run by
``--go``) still refuses on a missing object with the exact path. NOTHING IS
SUBSTITUTED — the smoke's whole point is those two real objects through the
committed adapters, and a stand-in would certify a different intervention;
``--static-r-object`` / ``--alphasteer-object`` exist to point at the
originals wherever they land, never at a look-alike.

THE WEIGHTS ON THE VOLUME. ``/vol/hf/hub/models--meta-llama--Llama-3.1-8B-
Instruct/snapshots/0e9e39f2…/`` was assembled in-volume: meta-llama's OWN
tokenizer / config / safetensors index (the small files under that repo's
``blobs/``) plus the four weight shards copied from the NousResearch mirror's
blobs, which HF names by content sha256 —

    2b1879f356aed350030bb40eb45ad362c89d9891096f79a3ab323d3ba5607668
    09d433f650646834a83c580877bd60c6d1f88f7755305c12576b5c7058f9af15
    fc1cdddd6bfa91128d6e94ee73d0ce62bfcdb7af29e978ddcab30c66ae9ea7fa
    92ecfe1a2414458b4821ac8c13cf8cb70aed66b5eea8dc5ad9eeb4ff309d6d7b

— so the sbatch header's "byte-identical weights, different chat template"
claim about the two repos is now verified shard for shard (by the operator
who placed them; this file only records it), and the smoke serves meta-
llama's template over those bytes exactly as the cluster did.

THE BASE COMPARATORS ON MODAL — the one leg of the ladder this port cannot
deliver. The smoke's throughput ladder stands up four plugin-disabled BASE
engines (``worker-base``) beside the lens engine; each asserts its resolved
model by RAW STRING equality (``engine["model"] != args.model``), whereas the
lens worker canonicalises the resolved path back to the hub id
(``canonical_model_identity``). Under Modal the volume mount is a realpath —
vLLM resolves ``meta-llama/Llama-3.1-8B-Instruct`` to
``/__modal/volumes/vo-…/hf/hub/models--meta-llama--…/snapshots/0e9e39f2…`` —
so every base comparator refuses at engine init with "comparator config
assertions failed: resolved model '/__modal/…' != configured
'meta-llama/Llama-3.1-8B-Instruct'" (exit 4), and the record carries
``MISSING: …`` for ``base_graphs_prefix_cache`` / ``base_eager_no_prefix_
cache`` (and the whole base half of the shared-prefix suite). The smoke is
FROZEN, so this cannot be changed here; the smoke's own verdict is
unaffected (the ladder "CERTIFIES NOTHING. Evidence only" — its words), and
so is this wrapper's: those rungs are ADVISORY (``ADVISORY_BASE_LEGS``),
recorded under ``advisory_missing`` in wrapper_record.json and on the RESULT
line with the smoke's reason, never a failure. What is lost: base-vs-lens
throughput (the lens_wiring_cost / prefix_cache_cost / cuda_graph_cost
subtractions) is NOT available from a Modal smoke; lens-vs-lens legs
(unselected vs steered, BI=0 vs BI=1, the response arm) still are.

THE CANARY (2026-09-06, ad6c3cb, app petri-smoke-ad6c3cb, tf5141, 940 s): the
smoke printed "SMOKE PASS — steered tokens served through the lens path at
the PRODUCTION configuration (serving_mode=interleaved,
VLLM_BATCH_INVARIANT=0); every gating check held in (bi0_certified) — all 5
checks passed outright", smoke_rc 0, both gating phases evaluated, PETRI
geometry KV 573,984 tokens = 35.03x at 16384, all nine lens legs measured,
both base rungs MISSING as above. Its scrubbed record is
tests/fixtures/modal_petri_smoke_record_completed.json, replayed by the
tests: classify_outcome -> PASS with the two base legs advisory-missing.

OUTCOME — the trap this wrapper exists to close. The smoke's own discipline
(its module docstring) EXITS 0 on ``smoke_unavailable.json`` (imports / GPU /
engine init / a MISSING OBJECT FILE): a launch blocker to escalate, not a
verdict. A wrapper that trusted the exit code would report a 50-second
COMPLETED as a pass. ``classify_outcome`` therefore treats ALL of these as
FAILURE (nonzero local exit, RESULT line naming the reason):
  * ``smoke_unavailable.json`` present (whatever the exit code) -> UNAVAILABLE
  * ``smoke_failed.json`` present, or a nonzero exit                -> FAIL
  * exit 0 with no ``smoke_record.json``                             -> FAIL
  * the internal deadline hit                                        -> TIMEOUT
and a PASS demands POSITIVE ARTIFACTS that can only exist if the work happened
(``POSITIVE_ARTIFACTS`` / ``positive_artifact_problems``) — the frozen smoke's
COMPLETE gate restated from its own code (``split_gating_failures``,
``_INSTRUMENT_FAILURE_MARKS``, ``selection_direct_failure``, ``check_state``,
``vllm_server.kv_coverage_problem``; the tests pin every constant against the
script), never stricter and never weaker:
  * ``smoke_record.json`` with status completed, ``smoke_evidence.json`` beside it;
  * BOTH fidelity reports (``fidelity`` at 2048x32, ``fidelity_petri_geometry``
    at 16384x24): all five checks present; engine_determinism / scale0_noop /
    steering_bites passed; nothing indeterminate, not invalidated; the
    ``failures`` list split exactly as the smoke splits it — ONLY
    batch_invariance and the composition half of selection_honored may be
    waived at BI=0, and a failure text carrying an INSTRUMENT mark ("no
    evidence gathered", "evidence must be a mapping", "evidence could not be
    decoded") gates under ANY name; every waived failure must be one the
    smoke recorded under ``*_waived_failures``; ``passed`` must agree with the
    failure list;
  * BOTH gating phases (``fidelity_phases.exit_status_governed_by`` ==
    [bi0_certified, bi0_certified_petri16k], and the gating / certified /
    petri keys naming them): outcome evaluated; gating, certifies,
    governs_exit_status, is_the_certified_configuration, engine_reached_ready,
    batch_invariant_as_requested all true; requested AND RESOLVED
    VLLM_BATCH_INVARIANT == "0"; child_exit 0; not invalidated; no
    unexpected_failures; complete check_states; the phase's selection_direct
    not FAILED;
  * ``selection_direct`` / ``selection_direct_petri_geometry`` not FAILED (a
    routing failure outranks every token-level check; MISSING / PARTIAL are
    advisory, as in the smoke, and are listed under ``advisory_missing``);
  * BOTH certified geometries: the recorded geometry equals the constants
    (2048x32 / 16384x24, max_num_batched_tokens = seqs x len — the no-chunking
    sizing); ``kv_cache.covers_full_budget`` true; the shared KV-coverage
    verdict re-derived clean from the recorded facts (uniform: num_gpu_blocks
    x block_size >= seqs x len; hybrid: kv_max_concurrency >= seqs); and the
    summary numbers agree (full_length_concurrency >= seqs, kv_cache_tokens
    >= seqs x len) — never merely "tokens > 0";
  * BOTH required arms: required, ``original_sha16`` == sha256_hex16 of the
    pinned ORIGINAL (``OBJECT_SHA256`` — the artifact hash; static_r is served
    DERIVED at lambda 0.05 so its served_sha16 differs by design), served;
    static_r derived at the deployment lambda, alphasteer at its stored
    base_scale;
  * the LENS throughput legs measured (``throughput.lens.lens_unselected`` +
    a ``lens_steered__<arm>`` leg per required arm — the proof the lens
    engine served); the provenance naming the staged sha and the pinned model.
The plugin-disabled BASE comparator rungs are ADVISORY (``ADVISORY_BASE_LEGS``):
they belong to the evidence-only ladder the smoke does not gate on, and on
Modal they are missing by construction (THE BASE COMPARATORS ON MODAL, below)
— reported under ``advisory_missing`` with the smoke's reason, never a failure.

OBJECT IDENTITY. ``--static-r-object`` / ``--alphasteer-object`` relocate the
inputs; they cannot substitute them. The preflight AND the GPU container hash
the volume file against ``OBJECT_SHA256`` (the originals as inventoried on the
cluster 2026-08-25) before any engine, and the record's ``original_sha16`` is
held to the same hashes at classification — a same-kind look-alike is refused
at all three points.

OUTPUTS. The smoke writes to container-local disk (``WORK_ROOT``) and this file
copies the out-dir (every JSON, log and marker; derived ``*.pt`` blobs are not
copied), ``smoke.log`` (the smoke's combined stdout+stderr — generated text goes
there by the smoke's design, never into a JSON), ``wrapper_record.json`` and a
one-line ``RESULT`` into THIS RUN'S OWN attempt directory
``/vol/petri-smoke/<sha>/attempts/<attempt-id>/`` every 5 minutes while the
smoke runs and once at the end. Attempts are immutable and never shared — two
runs on one sha cannot interleave their evidence — and a COMPLETED attempt is
promoted by atomically replacing the pointer ``/vol/petri-smoke/<sha>/
current.json`` (attempt id, verdict, status, RESULT line; written to a temp
file and ``os.replace``d). PUBLICATION IS PART OF THE EXIT CONTRACT, and the
pointer is its LAST DURABLE WRITE: the COMPLETE attempt in its final form —
RESULT (``published=true attempt=<id>``), wrapper_record.json carrying the very
pointer content that will publish it, the smoke's record + evidence — is
synced, committed and byte-verified FIRST, while nothing yet names it; only
then is ``current.json`` replaced (byte-identical to what the record carries),
committed and read back equal, and after that NOTHING touches the attempt.
Before any of it, the periodic sync thread must have QUIESCED (told to stop,
joined for a bounded time, proven dead — ``SyncWorker``): every sync of an
attempt, periodic or final, takes the one lock, and a worker still alive after
the bound (a copy or commit stalled on the volume) could land a stale copy
over verified bytes or after the pointer, so publication then fails closed —
nothing written, no pointer. A
completed attempt exits 0 only when all of that held; any failure makes the
returned record ``PUBLISH_FAILED`` (exit 1, the smoke's own verdict kept under
``smoke_verdict``). A failure BEFORE the pointer leaves an attempt nothing
names, rewritten (best effort) to say PUBLISH_FAILED; a failure AFTER the
replace (the commit, the read-back) leaves the attempt EXACTLY as it is — its
files are final and agree with the pointer that may durably name them; the
run merely could not confirm it, so it reports conservatively. The alternative
ordering fails: a pointer installed before the final sync, one sync failure,
and a recovery rewrite leave a PASS pointer beside PUBLISH_FAILED files, which
the mirror promotes. Readers resolve ``current.json``; older attempts stay
under ``attempts/``. ``--go`` then mirrors the sha's tree into
``<mirror_root>/<sha>/versions/<pull-id>/`` and atomically repoints the
``<mirror_root>/<sha>/current`` symlink — all-or-nothing: a single
unreadable file, a promoted attempt missing RESULT / wrapper_record.json
(and for a PASS the smoke's record + evidence), or a promoted attempt whose
own record and RESULT do not AGREE with the pointer (the record's ``current``
EQUAL to the pointer as a whole, same attempt and verdict, status DONE,
``publication_ok`` true, the pointer's ``path`` and ``sha`` the expected ones,
RESULT naming the attempt with ``published=true`` —
``pointed_attempt_problems``, independent of the publisher's discipline)
leaves the previous mirror byte-identical. The pull after a run is BOUND to
that run (``mirror_after_run``): it carries the exact pointer the run
installed and read back (``rec["current"]``), and the pulled ``current.json``
must EQUAL it as a whole or the pull is refused as ``MIRROR_SUPERSEDED`` — two
overlapping runs on one sha, or a valid attempt replacing the pointer between
read-back and pull, would otherwise leave the local ``current`` holding
attempt B's evidence under attempt A's exit 0. ``--harvest`` binds
to the pointer in THAT run's record, never to what ``current.json`` says now;
a run with no pointer of its own (PUBLISH_FAILED, SETUP_FAILED, TIMEOUT,
ATTEMPT_DIR_EXISTS) refreshes the mirror not at all. Every LOCAL mirror
transaction of a sha — version allocation, pull, promotion, its verification,
garbage collection — runs under one cross-process ``fcntl.flock`` on
``<mirror_root>/<sha>/.mirror.lock`` (``_mirror_lock``, bounded; a timeout is
``MIRROR_FAILED`` with the mirror untouched), and a pull claims ``promoted``
only after re-verifying, under that lock, that ``current`` resolves to ITS
version and that version's ``current.json`` is the expected pointer — two
same-sha commands on one machine otherwise each replaced ``current``, one read
the other's target and still claimed promotion, and GC acting on a stale
observation could remove the version the other had just promoted.
The newest ``MIRROR_KEEP_VERSIONS`` pulls remain, ordered by
a pull sequence recorded in each version (never by the id's lexical order),
the current one always among them. ``--harvest <state.json>`` re-attaches by
call id after a detached run.

RUN (the app/run name is REQUIRED — ``petri-smoke-<shortsha>`` — and verified;
an unnamed run is refused):

  ~/venvs/modal/bin/python -m modal run -n petri-smoke-<shortsha> scripts/modal/petri_smoke.py --stage --sha <sha>
  ~/venvs/modal/bin/python -m modal run -n petri-smoke-<shortsha> scripts/modal/petri_smoke.py --pf --sha <sha>
  ~/venvs/modal/bin/python -m modal run -n petri-smoke-<shortsha> scripts/modal/petri_smoke.py --stage --go --sha <sha>
  ~/venvs/modal/bin/python -m modal run -n petri-smoke-<shortsha> scripts/modal/petri_smoke.py \
      --harvest ~/.local/state/petri-smoke/<app>-<ts>.json

Local exit (``local_exit``): 0 only for a PUBLISHED PASS (``modal_exit_code``:
verdict PASS, status DONE, ``publication_ok``, ``current.attempt`` == this
attempt) whose evidence the local mirror now holds (or ``--no-mirror``); 1 for
FAIL / UNAVAILABLE / TIMEOUT / PUBLISH_FAILED / a setup or preflight refusal;
2 (``EXIT_MIRROR``) for a published PASS the mirror does NOT hold —
``MIRROR_SUPERSEDED`` when another publication replaced ``current.json``
between this run's read-back and its pull (the mirror is left untouched, and
holds no evidence this run did not install), ``MIRROR_FAILED`` for any other
refused or failed pull (a mirror lock timeout included).

WHAT IS WEAKER THAN THE SLURM SMOKE (the verdict's evidence, honestly):
  * different hardware lineage: Modal's H200s + driver, not cluster A's. The
    smoke's memory findings (24 x 16384 fits, 32 does not) were measured on
    the cluster's cards; the 16k phase re-measures KV coverage here and gates
    on it, so a different card would fail loudly rather than certify quietly.
  * the tree is a shallow clone pushed from the grading host rather than a clone of
    the cluster mirror; the sha and cleanness checks are the same.
  * the venv is an image built from the same requirements + cutoff, with
    transformers pinned to the version the cluster venv resolved rather than
    re-resolved (IMAGE above).
  * the sbatch's ``--mem=200G`` is 128 GiB here (the lens gate's proven
    request); the smoke holds one 8B engine at a time.
  * no NCCL_SOCKET_IFNAME / NCCL_NVLS_ENABLE: one GPU, one process, and
    naming an interface the host lacks would break, not help.
  * the sbatch's out-dir was ONE durable directory reused across runs (the
    smoke unlinks stale markers itself); here every run has its OWN attempt
    dir ``<sha>/attempts/<attempt-id>/`` (epoch + pid + random suffix, created
    exclusively — a collision FAILS CLOSED before any engine) and only a
    completed attempt moves the ``current.json`` pointer, atomically. Nothing
    is ever shared between runs, retired, renamed or deleted on the volume;
    there is no cross-container lock and none is needed. (The first layout
    synced into a shared per-sha dir and an earlier attempt's
    ``smoke_unavailable.json`` sat beside the passing record on 2026-09-06;
    a retirement step followed, whose empty-top-level check was not proof
    that no other writer was active — hence attempts.)
  * the base comparator half of the throughput ladder is MISSING on Modal
    (THE BASE COMPARATORS ON MODAL): base-vs-lens throughput cannot be read
    off a Modal smoke record; the cluster's could deliver it.
  * a container killed at the Modal timeout has no SIGTERM grace, so an
    INTERNAL deadline (``SMOKE_DEADLINE_S``, under the Modal cap) kills the
    smoke's process group, syncs and writes RESULT=TIMEOUT; the 5-minute sync
    is what survives anything harder.
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import secrets
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from pathlib import Path

import modal

#: ``modal run -n petri-smoke-<shortsha>`` is REQUIRED (the run ``--name``, the
#: app name in the dashboard); the entrypoint verifies it after resolving SHA.
DEFAULT_APP_NAME = "petri-steered-smoke"
app = modal.App(DEFAULT_APP_NAME)
VOL = modal.Volume.from_name("twosi-fits", create_if_missing=False)
VOL_PATH = "/vol"
REPOS = f"{VOL_PATH}/repos"                       # staged trees: sisteer-<sha> (git checkouts)
HF_HOME = f"{VOL_PATH}/hf"
RESULTS_ROOT = f"{VOL_PATH}/petri-smoke"          # <sha>/{steered_smoke/, smoke.log, RESULT, wrapper_record.json}
WORK_ROOT = "/root/petri-smoke-work"              # container-local: the smoke's --out-dir parent
TREES_ROOT = "/root/trees"                        # container-local copy of the staged tree
MIRROR_ROOT = Path.home() / "figs" / "petri-smoke"
STATE_DIR = Path.home() / ".local" / "state" / "petri-smoke"
#: how often the running smoke's out-dir / log are copied to the volume
SYNC_EVERY_S = 300
#: how long publication waits for the periodic sync thread to END after being
#: told to stop. A copy or commit still stalled on the volume past this is a
#: worker that could land a STALE copy after the final bytes were verified, or
#: after the pointer — so publication fails closed instead (``SyncWorker``).
SYNC_QUIESCE_S = 120
#: the Modal hard cap (the brief: 60 min) and the INTERNAL deadline under it,
#: so the kill + final sync + RESULT happen inside the container's clock.
#: Job 229597 ran the whole smoke in 14m47s on the cluster; Modal adds the
#: tree copy and cold volume reads of the 16 GB weights across twelve inits.
SMOKE_TIMEOUT_S = 60 * 60
SMOKE_DEADLINE_S = 55 * 60

# --- THE COMMAND: the sbatch's srun line, token for token -----------------------
SMOKE_SCRIPT = "scripts/petri_steered_smoke.py"
SBATCH_SCRIPT = "scripts/sbatch/petri_steered_smoke.sbatch"
MANIFEST = "configs/capeval/manifests/petri_twosi.yaml"
#: the model repin (meta-llama, NOT the manifest's NousResearch mirror — the
#: smoke's docstring, "MODEL PIN"); pinned equal to the smoke's defaults and
#: to the sbatch's tokens by the tests.
MODEL_HF = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_REV = "0e9e39f249a16976918f6564b8830bc894c89659"
STATIC_R_SCALE = "0.05"
RESPONSE_ARM = "on"
FIDELITY_BATCH_INVARIANT = "0"
FIDELITY_BI1_PROBE = "on"
#: the sbatch's ``$PETRI_RUN_DIR/steered_smoke``
OUT_SUBDIR = "steered_smoke"

#: the two REAL objects: named by the site, never by a default here (the
#: module docstring, INPUTS). Each arm's setting, the smoke's own key
#: (petri_steered_smoke.STATIC_R_OBJECT_KEY / ALPHASTEER_OBJECT_KEY, pinned
#: equal by the tests), read by the LOCAL entrypoint (``object_on_volume``):
#: a cluster path ``$SHARED/exp/jailbreak/<rel>`` maps to
#: ``/vol/objects/jailbreak/<rel>`` (``cluster_object_to_volume``).
CLUSTER_OBJECTS_PREFIX = "$SHARED/exp/jailbreak"
OBJECTS_ROOT = f"{VOL_PATH}/objects/jailbreak"
OBJECT_KEYS = {"static_r_matched": "SISTEER_PETRI_SMOKE_STATIC_R_OBJECT",
               "alphasteer_official": "SISTEER_PETRI_SMOKE_ALPHASTEER_OBJECT"}
#: the flag of this launcher (and of the smoke) that names each arm's object
OBJECT_FLAGS = {"static_r_matched": "--static-r-object", "alphasteer_official": "--alphasteer-object"}
#: the sbatch's shell variables for the two objects (its required-setting
#: lines read the keys into them) -> the arm each names
SBATCH_OBJECT_VARS = {"$STATIC_R_OBJECT": "static_r_matched", "$ALPHASTEER_OBJECT": "alphasteer_official"}
#: the sbatch's variable -> what it means here (the test translates with it);
#: the prefix is an older tree's sbatch, which spelled the objects' cluster paths
SBATCH_VARS = {"$SMOKE_OUT_DIR": "<out-dir>", CLUSTER_OBJECTS_PREFIX: OBJECTS_ROOT,
               **{var: f"<{arm} object>" for var, arm in SBATCH_OBJECT_VARS.items()}}
#: the launcher prefix the sbatch puts before the script (Slurm/venv plumbing,
#: not part of the command the smoke sees)
SBATCH_LAUNCHER = ("srun", "--ntasks=1", "uv", "run", "--no-sync", "python")

# --- the smoke's outcome vocabulary (pinned to the frozen script by the tests) --
UNAVAILABLE_MARKER = "smoke_unavailable.json"
FAILURE_MARKER = "smoke_failed.json"
SMOKE_RECORD = "smoke_record.json"
EVIDENCE_FILE = "smoke_evidence.json"
PHASE_CERTIFIED = "bi0_certified"
PHASE_PETRI = "bi0_certified_petri16k"
#: fidelity.evaluate_fidelity's five checks, in its order
FIDELITY_CHECKS = ("engine_determinism", "scale0_noop", "steering_bites",
                   "batch_invariance", "selection_honored")
#: petri_steered_smoke.BI0_EXPECTED_FAILURES — the ONLY two checks the smoke
#: waives at the certified BI=0 (they compare one request across two batch
#: COMPOSITIONS, the property given up with exact_replay); selection_honored's
#: DIRECT half still gates through selection_direct_failure
BI0_EXPECTED_FAILURES = ("batch_invariance", "selection_honored")
#: the three that gate at BI=0 unconditionally (FIDELITY_CHECKS minus the
#: waivable two — pinned equal by the tests)
NON_WAIVABLE_CHECKS = ("engine_determinism", "scale0_noop", "steering_bites")
#: petri_steered_smoke._INSTRUMENT_FAILURE_MARKS — evaluate_fidelity's own
#: normalised texts for "the comparison never RAN"; a failure carrying one is
#: never waived, under any check's name (the waiver's justification is "the
#: output moved for numerical reasons", which needs the comparison to exist)
INSTRUMENT_FAILURE_MARKS = ("no evidence gathered", "evidence must be a mapping",
                            "evidence could not be decoded")
#: petri_steered_smoke.CERTIFIED_BATCH_INVARIANT — the production configuration
CERTIFIED_BATCH_INVARIANT = "0"
#: the two GATING phases: the smoke's ``fidelity_phases.exit_status_governed_by``
EXIT_STATUS_GOVERNED_BY = (PHASE_CERTIFIED, PHASE_PETRI)
#: the two certified geometries (petri_steered_smoke MAX_MODEL_LEN x MAX_NUM_SEQS
#: and PETRI_MAX_MODEL_LEN x PETRI_MAX_NUM_SEQS), keyed as ``certified_geometries``
#: records them; each engine must hold the FULL no-chunking budget in KV
GEOMETRIES = {"fidelity_2048": (2048, 32), "fidelity_petri_16384": (16384, 24)}
REQUIRED_ARMS = ("static_r_matched", "alphasteer_official")
#: THE ORIGINALS' sha256 — the two REAL fitted objects the sbatch served, as
#: inventoried on the cluster 2026-08-25 (hash-verified by the operator who
#: placed the copies on the volume 2026-09-06). The preflight AND the GPU
#: container hash the volume file against these before any engine, so a
#: relocated ``--static-r-object`` / ``--alphasteer-object`` is accepted only
#: when its BYTES are the original; and the record must carry sha256_hex16 of
#: the raw artifact as ``arm_derivations[].original_sha16`` (the ARTIFACT hash
#: — static_r is served DERIVED at lambda 0.05, so its served_sha16 differs).
OBJECT_SHA256 = {
    "static_r_matched": "4cc6b82a38c11cfc68bd095d3e648ecb49d55fc882a92832792b07888ca7afc7",
    "alphasteer_official": "2c3477acf7e8dfc0d751f463a06ba44e8ddad3c7fa6929dadecbf9e5b0ff33c6",
}
#: the LENS engine's legacy-suite throughput legs — the ones that prove the
#: lens served: one unselected leg + one per REQUIRED arm (worker_steered's
#: ``lens_unselected`` / ``lens_steered__<arm>`` tags). Pinned by the tests
#: against the real 2026-09-06 Modal record (tests/fixtures/
#: modal_petri_smoke_record_completed.json), which carries exactly these
#: keys plus their ``__bi1`` twins, the advisory arm's leg and the response
#: arm's leg.
LENS_LEGS_REQUIRED = ("lens_unselected",) + tuple(f"lens_steered__{a}" for a in REQUIRED_ARMS)
#: the plugin-disabled BASE comparator rungs the record carries at the top
#: level of ``throughput`` — ADVISORY, never gating: they belong to the
#: ladder, which the smoke itself labels evidence only ("THIS PHASE
#: CERTIFIES NOTHING"), and on Modal they are MISSING by construction (the
#: module docstring, THE BASE COMPARATORS ON MODAL). Their absence is
#: recorded — wrapper_record.json ``advisory_missing``, the RESULT line —
#: with the smoke's own reason, and does not fail a run.
ADVISORY_BASE_LEGS = ("base_graphs_prefix_cache", "base_eager_no_prefix_cache")
#: WHAT A PASS DEMANDS — the artifacts that can only exist if the work
#: happened, by file and key (``positive_artifact_problems`` enforces it;
#: this table is the human-readable statement of the same rule). It is the
#: smoke's OWN gating restated — never stricter than the frozen script.
POSITIVE_ARTIFACTS = {
    SMOKE_RECORD: {
        "status": "completed",
        "fidelity": (f"all of {FIDELITY_CHECKS} present; {NON_WAIVABLE_CHECKS} passed; nothing "
                     f"indeterminate/invalidated; failures split as the smoke splits them — only "
                     f"{BI0_EXPECTED_FAILURES} waivable at BI=0, never an instrument failure "
                     f"({INSTRUMENT_FAILURE_MARKS}); every waived failure recorded by the smoke; "
                     "passed agrees with failures"),
        "fidelity_petri_geometry": "the same gate, at the 16384x24 PETRI geometry",
        "fidelity_phases": (f"gating/certified keys == {PHASE_CERTIFIED}, petri keys == {PHASE_PETRI}, "
                            f"exit_status_governed_by == {list(EXIT_STATUS_GOVERNED_BY)}; each gating phase: "
                            "outcome evaluated; gating, certifies, governs_exit_status, "
                            "is_the_certified_configuration, engine_reached_ready, batch_invariant_as_requested "
                            f"all true; requested and RESOLVED VLLM_BATCH_INVARIANT == {CERTIFIED_BATCH_INVARIANT}; "
                            "child_exit 0; not invalidated; no unexpected_failures; check_states complete with "
                            "the non-waivable three passed; its selection_direct not FAILED"),
        "selection_direct / selection_direct_petri_geometry": "verdict != FAILED (MISSING/PARTIAL advisory, as in the smoke)",
        "certified_geometries": (f"{GEOMETRIES}: recorded geometry equals the constants, max_num_batched_tokens "
                                 "= seqs x len, kv_cache.covers_full_budget true, the shared KV-coverage verdict "
                                 "re-derived clean, full_length_concurrency >= seqs, kv_cache_tokens >= seqs x len"),
        "arm_derivations": (f"{REQUIRED_ARMS} required, original_sha16 == sha256_hex16 of the pinned originals "
                            f"(OBJECT_SHA256), served_sha16 present; static_r DERIVED at {STATIC_R_SCALE}, "
                            "alphasteer at its stored base_scale"),
        "throughput.lens": f"tokens_per_s on {LENS_LEGS_REQUIRED} (the lens engine served and was measured)",
        "provenance": "checkout_sha == the staged sha; served_model/revision == the pin",
    },
    EVIDENCE_FILE: "present beside the record",
}
#: WHAT IS REPORTED BUT NEVER GATES (``advisory_missing``): evidence-only
#: legs the smoke itself does not gate on.
ADVISORY_ARTIFACTS = {
    f"throughput.{leg}": "plugin-disabled comparator rung (evidence only); MISSING on Modal — see the docstring"
    for leg in ADVISORY_BASE_LEGS
}
#: env the sbatch set for the CLUSTER that must NOT travel (an inherited
#: VLLM_PLUGINS would filter the lens plugin out — the smoke pops it too; the
#: BI / lens-disable switches are set per child BY the smoke; an NCCL
#: interface name is site-specific)
SCRUB_ENV = ("VLLM_PLUGINS", "VLLM_LENS_DISABLE", "VLLM_BATCH_INVARIANT",
             "NCCL_SOCKET_IFNAME", "NCCL_NVLS_ENABLE")

# --- the serving image: lens_vectorize_gate._serving_image, copied byte for byte
# (same recipe -> same Modal image id -> no rebuild) ---------------------------
try:
    LOCAL_REPO = Path(__file__).resolve().parents[2]
except IndexError:  # Modal re-imports this module at a flat /root path
    LOCAL_REPO = Path("/repo")
VLLM_REQUIREMENTS = LOCAL_REPO / "configs/capeval/vllm-requirements.txt"
REQS_FILE = {"tf5141": "configs/capeval/geom-sweep-requirements-tf5141.txt",
             "tf5150": "configs/capeval/geom-sweep-requirements.txt"}
TRANSFORMERS_PIN = {"tf5141": "5.14.1", "tf5150": "5.15.0"}
UV_VERSION = "0.12.1"
SERVING_EXCLUDE_NEWER = "2026-08-03T00:00:00Z"
TRANSFORMERS_EXCLUDE_NEWER = {"tf5141": None, "tf5150": "2026-08-11T00:00:00Z"}
SERVING_REQS_ROOT = "/opt/sisteer-capeval-reqs"
#: the sbatch's venv resolved transformers 5.14.1 (IMAGE in the docstring)
DEFAULT_IMAGE = "tf5141"


def _serving_image(variant: str) -> modal.Image:
    """lens_vectorize_gate._serving_image, verbatim (the certified recipe)."""
    if variant not in REQS_FILE:
        raise ValueError(f"unknown serving-image variant {variant!r}")
    tf_pin = TRANSFORMERS_PIN[variant]
    req_path = LOCAL_REPO / REQS_FILE[variant]
    remote_vllm = f"{SERVING_REQS_ROOT}/{VLLM_REQUIREMENTS.name}"
    remote_outer = f"{SERVING_REQS_ROOT}/{req_path.name}"
    tf_cutoff = TRANSFORMERS_EXCLUDE_NEWER[variant]
    tf_cutoff_arg = (f"--exclude-newer-package transformers={tf_cutoff} "
                     if tf_cutoff else "")
    install = (
        "uv pip install --python $(command -v python) "
        "--index-strategy unsafe-best-match "
        f"--exclude-newer {SERVING_EXCLUDE_NEWER} "
        f"{tf_cutoff_arg}-r {remote_outer} transformers=={tf_pin}"
    )
    return (
        modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python="3.12")
        .apt_install("git", "build-essential")
        .env({"CUDA_HOME": "/usr/local/cuda"})
        # Bootstrap only the pinned uv executable with pip; uv resolves and
        # installs the serving environment itself from the certified files.
        .pip_install(f"uv=={UV_VERSION}")
        .add_local_file(VLLM_REQUIREMENTS, remote_vllm, copy=True)
        .add_local_file(req_path, remote_outer, copy=True)
        .run_commands(
            f"uv --version | grep -E '^uv {UV_VERSION.replace('.', '[.]')}( |$)'",
            install,
            "git config --global --add safe.directory '*'",
            "python -c \"import torch, vllm, transformers, aiohttp, yaml, matplotlib; "
            "assert torch.__version__ == '2.11.0+cu129', torch.__version__; "
            "assert vllm.__version__.startswith('0.25.1'), vllm.__version__; "
            f"assert transformers.__version__ == '{tf_pin}', transformers.__version__; "
            "print('IMAGE OK', vllm.__version__, torch.__version__, transformers.__version__)\"",
        )
        .env({"HF_HOME": HF_HOME, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
              "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    )


image_tf5141 = _serving_image("tf5141")
image_tf5150 = _serving_image("tf5150")
IMAGES = {"tf5141": image_tf5141, "tf5150": image_tf5150}


# --------------------------------------------------------------------------- #
# helpers (pure; nothing here touches modal at call time)                       #
# --------------------------------------------------------------------------- #


def volume_client_path(path: str) -> str:
    """Translate a mounted ``/vol/...`` path to the Volume client's ``/...``."""
    if path == VOL_PATH:
        return "/"
    prefix = f"{VOL_PATH}/"
    if not path.startswith(prefix):
        raise ValueError(f"path is outside {VOL_PATH}: {path}")
    return path[len(VOL_PATH):]


def cluster_object_to_volume(path: str) -> str:
    """A cluster object path ``$SHARED/exp/jailbreak/<rel>`` (``${SHARED}``
    alike) -> ``/vol/objects/jailbreak/<rel>``. Any other prefix refuses: the
    mapping is the documented one, not a guess."""
    if path.startswith("${SHARED}/"):
        path = "$SHARED/" + path[len("${SHARED}/"):]
    prefix = CLUSTER_OBJECTS_PREFIX + "/"
    if not path.startswith(prefix):
        raise ValueError(f"not a cluster object path under {CLUSTER_OBJECTS_PREFIX}: {path}")
    return f"{OBJECTS_ROOT}/{path[len(prefix):]}"


def _site_setting(key: str, env=None):
    """``sisteer.site.setting(key)`` (the environment, then the site config) from
    this checkout's src. Local only: the container is handed resolved paths."""
    src = str(LOCAL_REPO / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from sisteer import site
    return site.setting(key, env=env)


def object_on_volume(name: str, given: str = "", *, env=None) -> str:
    """The volume path of the REQUIRED object ``name``: ``given`` (the flag),
    else the site's setting ``OBJECT_KEYS[name]``. A ``/vol/`` path is taken as
    it is; a cluster path ``$SHARED/exp/jailbreak/<rel>`` (as the sbatch serves
    it) maps to the volume by ``cluster_object_to_volume``. Nothing named, or
    any other form, raises SystemExit naming the flag and the key: there is no
    default object."""
    flag, key = OBJECT_FLAGS[name], OBJECT_KEYS[name]
    try:
        value = given or _site_setting(key, env)
    except ValueError as e:       # sisteer.site.SiteConfigError: a site config named and missing or unreadable
        raise SystemExit(f"[petri-smoke] REFUSED: {key} could not be read: {e}")
    if not value:
        raise SystemExit(f"[petri-smoke] REFUSED: no {name} object: pass {flag}, or set {key} (the site config's key, "
                         f"or the environment's) to its cluster path {CLUSTER_OBJECTS_PREFIX}/<rel> (served here from "
                         f"{OBJECTS_ROOT}/<rel>) or to its path under {VOL_PATH}/")
    if value == VOL_PATH or value.startswith(VOL_PATH + "/"):
        return value
    try:
        return cluster_object_to_volume(value)
    except ValueError:
        raise SystemExit(f"[petri-smoke] REFUSED: {flag if given else key}={value!r} names neither a cluster path under "
                         f"{CLUSTER_OBJECTS_PREFIX}/ (served here from {OBJECTS_ROOT}/) nor a path under {VOL_PATH}/")


def smoke_argv(python: str, out_dir: str, static_r_object: str, alphasteer_object: str) -> list:
    """The sbatch's ``srun ... python scripts/petri_steered_smoke.py smoke ...``
    line as argv, launcher prefix dropped, ``$SMOKE_OUT_DIR`` and the two
    objects substituted (their volume paths, ``object_on_volume``). Relative
    script + manifest paths: the smoke runs from the tree as cwd, as the
    sbatch's ``cd`` arranged."""
    return [python, SMOKE_SCRIPT, "smoke",
            "--manifest", MANIFEST,
            "--out-dir", out_dir,
            "--model", MODEL_HF,
            "--revision", MODEL_REV,
            "--static-r-object", static_r_object,
            "--alphasteer-object", alphasteer_object,
            "--static-r-scale", STATIC_R_SCALE,
            "--response-arm", RESPONSE_ARM,
            "--fidelity-batch-invariant", FIDELITY_BATCH_INVARIANT,
            "--fidelity-bi1-probe", FIDELITY_BI1_PROBE]


def sbatch_smoke_tokens(sbatch_text: str) -> list:
    """The tokens of the sbatch's smoke command, from the script path on
    (continuation lines joined, shell-split, the launcher prefix dropped),
    with the shell variables left as written. Refuses if the line is not
    found or not prefixed by the known launcher — a changed sbatch must be
    read, not silently re-parsed."""
    lines = sbatch_text.splitlines()
    start = next((i for i, l in enumerate(lines)
                  if SMOKE_SCRIPT in l and " smoke" in l and not l.lstrip().startswith("#")), None)
    if start is None:
        raise ValueError(f"no `{SMOKE_SCRIPT} smoke` command line in the sbatch")
    joined, i = "", start
    while i < len(lines):
        piece = lines[i].rstrip()
        if piece.endswith("\\"):
            joined += piece[:-1] + " "
            i += 1
            continue
        joined += piece
        break
    toks = shlex.split(joined)
    if tuple(toks[:len(SBATCH_LAUNCHER)]) != SBATCH_LAUNCHER:
        raise ValueError(f"sbatch launcher prefix changed: {toks[:len(SBATCH_LAUNCHER)]} != {SBATCH_LAUNCHER}")
    return toks[len(SBATCH_LAUNCHER):]


def sbatch_tokens_translated(sbatch_text: str, out_dir: str, objects=None) -> list:
    """The sbatch's tokens with ``$SMOKE_OUT_DIR`` -> out_dir, its object
    variables (``SBATCH_OBJECT_VARS``) -> ``objects[arm]`` (the volume paths
    this launcher serves), and an older tree's literal
    ``$SHARED/exp/jailbreak/...`` object paths -> their volume paths."""
    out = []
    for t in sbatch_smoke_tokens(sbatch_text):
        if t == "$SMOKE_OUT_DIR":
            out.append(out_dir)
        elif t in SBATCH_OBJECT_VARS:
            if not objects or not objects.get(SBATCH_OBJECT_VARS[t]):
                raise ValueError(f"the sbatch serves {t} (the site's {OBJECT_KEYS[SBATCH_OBJECT_VARS[t]]}): "
                                 f"name the {SBATCH_OBJECT_VARS[t]} object to translate it")
            out.append(objects[SBATCH_OBJECT_VARS[t]])
        elif t.startswith(CLUSTER_OBJECTS_PREFIX + "/"):
            out.append(cluster_object_to_volume(t))
        elif t.startswith("$"):
            raise ValueError(f"untranslated sbatch variable in the smoke command: {t}")
        else:
            out.append(t)
    return out


def smoke_env(base: dict, tree: str) -> dict:
    """The sbatch's body env, translated: HF_HOME on the volume (hf_env.sh's
    HF_HOME), offline, expandable_segments; the cluster-only NCCL vars and the
    per-child switches scrubbed (the smoke sets VLLM_BATCH_INVARIANT /
    VLLM_LENS_DISABLE per child itself and pops VLLM_PLUGINS); PYTHONPATH
    naming the tree's src (the smoke inserts it too; children inherit)."""
    env = dict(base)
    for k in SCRUB_ENV:
        env.pop(k, None)
    env["PYTHONPATH"] = f"{tree}/src"
    env["HF_HOME"] = HF_HOME
    env["HF_HUB_OFFLINE"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    return env


def snapshot_rel(hf: str, rev: str) -> str:
    return f"hub/models--{hf.replace('/', '--')}/snapshots/{rev}"


def _git(repo: str, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True,
                          timeout=120).stdout.strip()


def verify_checkout(repo: str, sha: str) -> dict:
    """``{"ok", "head", "clean", "why"}`` — the sbatch's immutability check:
    HEAD equals the sha and no tracked file is modified."""
    head = _git(repo, "rev-parse", "HEAD")
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--"], cwd=repo,
                           capture_output=True).returncode == 0 if head else False
    ok = bool(head) and head == sha and clean
    why = None if ok else ("not a git checkout" if not head else
                           f"HEAD {head} != {sha}" if head != sha else "modified tracked files")
    return {"ok": ok, "head": head or None, "clean": clean, "why": why}


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


# --- object identity: the pinned originals, hashed before any engine -----------

def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def verify_object(name: str, path: str) -> dict:
    """The REQUIRED object at ``path`` IS the pinned original: present,
    non-empty, and its bytes hash to OBJECT_SHA256[name]. Raises
    AssertionError naming the missing path (with the site key that names the
    object) or the hash mismatch — a same-kind look-alike would certify a
    different intervention, so it is refused, never substituted."""
    want = OBJECT_SHA256[name]
    p = Path(path)
    if not p.is_file() or p.stat().st_size == 0:
        raise AssertionError(f"REQUIRED object {name} absent at {p} (the site names it by {OBJECT_KEYS[name]}) "
                             "— nothing is substituted; restore the original")
    got = sha256_file(p)
    if got != want:
        raise AssertionError(f"REQUIRED object {name} at {p} hashes to {got}, not the pinned original "
                             f"{want} — a different object would certify a different intervention; refusing")
    return {"path": str(p), "size": p.stat().st_size, "sha256": got}


# --- the smoke's gating logic, RESTATED from petri_steered_smoke.py ---------------
# (each helper names the smoke function it mirrors; the tests pin the constants
#  against the frozen script so a drift there fails here)


def check_state(verdict) -> str:
    """petri_steered_smoke.check_state: ``evaluate_fidelity`` writes "passed",
    an "indeterminate: ..." line when engine_determinism invalidated the
    report, or the failure message itself; the three are never collapsed."""
    v = str(verdict)
    if v == "passed":
        return "passed"
    if v.startswith("indeterminate"):
        return "indeterminate"
    return "failed"


def split_gating_failures(failures, *, batch_invariant: str) -> tuple:
    """petri_steered_smoke.split_gating_failures, the same rule: a failure
    string starts with its check's name; at BI=0 the two composition checks
    (BI0_EXPECTED_FAILURES) are WAIVED — unless the text carries an INSTRUMENT
    failure mark (the comparison never ran), which gates under any name;
    everything else (engine_determinism, scale0_noop, steering_bites, the
    structural ``evidence:`` failures) gates. At any other BI nothing is waived."""
    waivable = set(BI0_EXPECTED_FAILURES) if str(batch_invariant) == "0" else set()
    gating, waived = [], []
    for f in failures:
        text = str(f)
        name = text.split(":", 1)[0].strip()
        instrument = any(mark in text for mark in INSTRUMENT_FAILURE_MARKS)
        (gating if (name not in waivable or instrument) else waived).append(text)
    return gating, waived


#: where ``sisteer`` is imported from for the AUTHORITATIVE KV verdict: this
#: checkout's ``src`` locally (tests); inside the container ``_run_smoke``
#: prepends the STAGED tree's ``src`` — the very sha under certification.
SISTEER_SRC_CANDIDATES = [str(LOCAL_REPO / "src")]


def _import_kv_verdict():
    """``sisteer.capeval.serving.vllm_server.kv_coverage_problem`` — THE
    verdict the smoke's worker gates every engine on. Imported, never
    re-implemented: a replica drifted once (it accepted ``kv_cache_groups=
    [{}, {}]`` that the real function refuses). Read-only use of the frozen
    module."""
    try:
        from sisteer.capeval.serving.vllm_server import kv_coverage_problem
        return kv_coverage_problem
    except ImportError:
        pass
    for src in SISTEER_SRC_CANDIDATES:
        if Path(src).is_dir() and src not in sys.path:
            sys.path.insert(0, src)
    from sisteer.capeval.serving.vllm_server import kv_coverage_problem
    return kv_coverage_problem


def authoritative_kv_coverage_problem(kv: dict, *, max_num_seqs: int, max_model_len: int):
    """petri_steered_smoke.kv_coverage_problem's adapter over the record's
    ``kv_cache`` block: the smoke's legacy fact names (``num_gpu_blocks`` /
    ``block_size``) mapped onto the report names, then the shared verdict
    called as-is. FAILS CLOSED when the verdict cannot be imported: coverage
    that cannot be verified never passes."""
    try:
        verdict = _import_kv_verdict()
    except Exception as e:  # noqa: BLE001
        return (f"authoritative KV-coverage verdict unavailable ({type(e).__name__}: {e}) — coverage "
                "cannot be verified, so it does not pass")
    rep = dict(kv)
    rep["kv_num_gpu_blocks"] = kv.get("num_gpu_blocks")
    rep["kv_block_size"] = kv.get("block_size")
    return verdict(rep, max_num_seqs=max_num_seqs, max_model_len=max_model_len)


def fidelity_report_problems(key: str, rep, recorded_waived) -> list:
    """The gate over one fidelity report (``fidelity`` at 2048/32,
    ``fidelity_petri_geometry`` at 16384/24) exactly as the smoke applies it
    at the certified BI=0: every check present; the non-waivable three passed;
    nothing indeterminate or invalidated; ``failures`` split by
    split_gating_failures must leave NO gating failure (instrument failures
    included, under any check's name); every waived failure must be one the
    smoke itself recorded as waived; a failed waivable check must be named by
    a failure string; and ``passed`` must agree with the failure list."""
    if not isinstance(rep, dict):
        return [f"{key}: no report"]
    problems = []
    checks = rep.get("checks") or {}
    missing = [c for c in FIDELITY_CHECKS if c not in checks]
    if missing:
        problems.append(f"{key}.checks lacks {missing}")
    states = {c: check_state(v) for c, v in checks.items()}
    for c in NON_WAIVABLE_CHECKS:
        if c in checks and states[c] != "passed":
            problems.append(f"{key}.checks.{c}={states[c]} ({str(checks[c])[:160]!r}) — gates at any BI")
    indeterminate = sorted(c for c, s in states.items() if s == "indeterminate")
    if indeterminate:
        problems.append(f"{key}: indeterminate checks {indeterminate} — the report was invalidated")
    if rep.get("invalidated"):
        problems.append(f"{key}.invalidated={rep.get('invalidated')!r}")
    for c in BI0_EXPECTED_FAILURES:
        if c in checks and states[c] == "failed" and any(m in str(checks[c]) for m in INSTRUMENT_FAILURE_MARKS):
            problems.append(f"{key}.checks.{c}: INSTRUMENT failure, never waivable: {str(checks[c])[:160]!r}")
    failures = [str(f) for f in (rep.get("failures") or [])]
    gating, waived = split_gating_failures(failures, batch_invariant=CERTIFIED_BATCH_INVARIANT)
    if gating:
        problems.append(f"{key}.failures gating at BI=0: {[g[:160] for g in gating]}")
    recorded = {str(w) for w in (recorded_waived or [])}
    unrecorded = [w for w in waived if w not in recorded]
    if unrecorded:
        problems.append(f"{key}: waivable failures the smoke did not record as waived: "
                        f"{[u[:120] for u in unrecorded]}")
    for c in BI0_EXPECTED_FAILURES:
        if c in checks and states[c] == "failed" and not any(f.split(":", 1)[0].strip() == c for f in failures):
            problems.append(f"{key}.checks.{c} failed but no failure string names it — inconsistent report")
    if bool(rep.get("passed")) != (not failures):
        problems.append(f"{key}.passed={rep.get('passed')!r} disagrees with {len(failures)} recorded failure(s)")
    return problems


def phase_problems(pk: str, ph) -> list:
    """The gate over one GATING phase record as the smoke wrote it: evaluated;
    gating / certifies / governs_exit_status / is_the_certified_configuration /
    engine_reached_ready / batch_invariant_as_requested all true; requested AND
    RESOLVED VLLM_BATCH_INVARIANT == "0" (the production configuration, read
    back from the child's own environment); child exit 0; not invalidated; no
    unexpected (non-waivable) failures; every check state present with the
    non-waivable three passed; and the phase's own selection_direct not FAILED
    (petri_steered_smoke.selection_direct_failure — MISSING/PARTIAL stay
    advisory there and here)."""
    if not isinstance(ph, dict) or not ph:
        return [f"fidelity_phases.phases.{pk}: missing"]
    p = []
    if ph.get("outcome") != "evaluated":
        p.append(f"fidelity_phases.phases.{pk}.outcome={ph.get('outcome')!r} (want 'evaluated')")
    for flag in ("gating", "certifies", "governs_exit_status", "is_the_certified_configuration",
                 "engine_reached_ready", "batch_invariant_as_requested"):
        if ph.get(flag) is not True:
            p.append(f"fidelity_phases.phases.{pk}.{flag}={ph.get(flag)!r} (want True)")
    for k in ("requested_batch_invariant", "resolved_batch_invariant"):
        if str(ph.get(k)) != CERTIFIED_BATCH_INVARIANT:
            p.append(f"fidelity_phases.phases.{pk}.{k}={ph.get(k)!r} (want {CERTIFIED_BATCH_INVARIANT!r})")
    if ph.get("child_exit") != 0:
        p.append(f"fidelity_phases.phases.{pk}.child_exit={ph.get('child_exit')!r} (want 0)")
    if ph.get("invalidated"):
        p.append(f"fidelity_phases.phases.{pk}.invalidated={ph.get('invalidated')!r}")
    if ph.get("unexpected_failures"):
        p.append(f"fidelity_phases.phases.{pk}.unexpected_failures={ph.get('unexpected_failures')!r}")
    states = ph.get("check_states") or {}
    missing = [c for c in FIDELITY_CHECKS if c not in states]
    if missing:
        p.append(f"fidelity_phases.phases.{pk}.check_states lacks {missing}")
    for c in NON_WAIVABLE_CHECKS:
        if c in states and states[c] != "passed":
            p.append(f"fidelity_phases.phases.{pk}.check_states.{c}={states[c]!r}")
    sd = ph.get("selection_direct")
    if isinstance(sd, dict) and str(sd.get("verdict") or "") == "FAILED":
        p.append(f"fidelity_phases.phases.{pk}.selection_direct FAILED — a routing failure outranks "
                 "every token-level check")
    return p


def geometry_problems(key: str, geom, *, max_model_len: int, max_num_seqs: int) -> list:
    """The gate over one certified geometry: the recorded geometry equals the
    smoke's constants (and max_num_batched_tokens = seqs x len, the no-chunking
    sizing); ``covers_full_budget`` is true; the shared KV-coverage verdict
    re-derived from the recorded cache facts is clean; and the smoke's own
    summary numbers agree (full_length_concurrency >= seqs, kv_cache_tokens >=
    seqs x len)."""
    if not isinstance(geom, dict):
        return [f"certified_geometries.{key}: missing"]
    p = []
    if geom.get("max_model_len") != max_model_len or geom.get("max_num_seqs") != max_num_seqs:
        p.append(f"certified_geometries.{key}: recorded {geom.get('max_model_len')}x{geom.get('max_num_seqs')} "
                 f"!= {max_model_len}x{max_num_seqs}")
    if geom.get("max_num_batched_tokens") != max_model_len * max_num_seqs:
        p.append(f"certified_geometries.{key}.max_num_batched_tokens={geom.get('max_num_batched_tokens')!r} "
                 f"!= {max_model_len * max_num_seqs} (chunked prefill could bind)")
    kv = geom.get("kv_cache")
    if not isinstance(kv, dict):
        return p + [f"certified_geometries.{key}.kv_cache missing — the engine never resolved its KV"]
    if kv.get("covers_full_budget") is not True:
        p.append(f"certified_geometries.{key}.kv_cache.covers_full_budget={kv.get('covers_full_budget')!r}")
    prob = authoritative_kv_coverage_problem(kv, max_num_seqs=max_num_seqs, max_model_len=max_model_len)
    if prob:
        p.append(f"certified_geometries.{key}.kv_cache: {prob}")
    flc = kv.get("full_length_concurrency")
    if not (_num(flc) and flc >= max_num_seqs):
        p.append(f"certified_geometries.{key}.kv_cache.full_length_concurrency={flc!r} < {max_num_seqs}")
    toks = kv.get("kv_cache_tokens")
    if not (_num(toks) and toks >= max_num_seqs * max_model_len):
        p.append(f"certified_geometries.{key}.kv_cache.kv_cache_tokens={toks!r} < {max_num_seqs * max_model_len}")
    return p


def arm_problems(record: dict) -> list:
    """The two REQUIRED arms are the pinned ORIGINALS (``original_sha16`` ==
    sha256_hex16 of OBJECT_SHA256 — the artifact hash, not the served one),
    marked required, served as the sbatch deploys them (static_r DERIVED at
    lambda 0.05; alphasteer at its stored base_scale, so served == original),
    and actually served (served_sha16 present)."""
    arms = {a.get("cell_id"): a for a in (record.get("arm_derivations") or []) if isinstance(a, dict)}
    p = []
    for a in REQUIRED_ARMS:
        rec = arms.get(a)
        if not rec:
            p.append(f"arm_derivations lacks {a}")
            continue
        if rec.get("required") is not True:
            p.append(f"arm {a}: required={rec.get('required')!r}")
        want16 = OBJECT_SHA256[a][:16]
        if rec.get("original_sha16") != want16:
            p.append(f"arm {a}: original_sha16={rec.get('original_sha16')!r} != {want16} "
                     f"(sha256 {OBJECT_SHA256[a]}) — a DIFFERENT object was served")
        if not rec.get("served_sha16"):
            p.append(f"arm {a}: no served_sha16 (never derived)")
    sr = arms.get("static_r_matched")
    if sr and not (sr.get("derived") is True and sr.get("deploy_scale") == float(STATIC_R_SCALE)
                   and sr.get("served_base_scale") == float(STATIC_R_SCALE)):
        p.append(f"arm static_r_matched: not served DERIVED at the deployment lambda {STATIC_R_SCALE} "
                 f"(derived={sr.get('derived')!r}, deploy_scale={sr.get('deploy_scale')!r}, "
                 f"served_base_scale={sr.get('served_base_scale')!r})")
    al = arms.get("alphasteer_official")
    if al and not (al.get("derived") is False and al.get("deploy_scale") is None
                   and al.get("served_sha16") == al.get("original_sha16")):
        p.append(f"arm alphasteer_official: not served at its stored base_scale (derived={al.get('derived')!r}, "
                 f"deploy_scale={al.get('deploy_scale')!r}, served_sha16={al.get('served_sha16')!r} vs "
                 f"original {al.get('original_sha16')!r})")
    return p


def positive_artifact_problems(record: dict, expected_sha: str | None = None) -> list:
    """Everything a PASS must show in ``smoke_record.json`` (POSITIVE_ARTIFACTS)
    that this record does not; empty means the record certifies. This is the
    frozen smoke's COMPLETE gate restated — never stricter, never weaker."""
    problems = []
    if record.get("status") != "completed":
        problems.append(f"status={record.get('status')!r} (want 'completed')")
    problems += fidelity_report_problems("fidelity", record.get("fidelity"),
                                         record.get("fidelity_waived_failures"))
    problems += fidelity_report_problems("fidelity_petri_geometry", record.get("fidelity_petri_geometry"),
                                         record.get("fidelity_petri_geometry_waived_failures"))
    phases = record.get("fidelity_phases") or {}
    for k, want in (("gating_phase_key", PHASE_CERTIFIED), ("certified_phase_key", PHASE_CERTIFIED),
                    ("petri_geometry_phase_key", PHASE_PETRI), ("certified_petri_phase_key", PHASE_PETRI)):
        if phases.get(k) != want:
            problems.append(f"fidelity_phases.{k}={phases.get(k)!r} (want {want!r})")
    if sorted(phases.get("exit_status_governed_by") or []) != sorted(EXIT_STATUS_GOVERNED_BY):
        problems.append(f"fidelity_phases.exit_status_governed_by={phases.get('exit_status_governed_by')!r} "
                        f"(want {list(EXIT_STATUS_GOVERNED_BY)})")
    for pk in EXIT_STATUS_GOVERNED_BY:
        problems += phase_problems(pk, (phases.get("phases") or {}).get(pk))
    for sd_key in ("selection_direct", "selection_direct_petri_geometry"):
        sd = record.get(sd_key)
        if isinstance(sd, dict) and str(sd.get("verdict") or "") == "FAILED":
            problems.append(f"{sd_key} FAILED — the worker's own selection record contradicts the mixed batch")
    geoms = record.get("certified_geometries") or {}
    for key, (mml, mns) in GEOMETRIES.items():
        problems += geometry_problems(key, geoms.get(key), max_model_len=mml, max_num_seqs=mns)
    problems += arm_problems(record)
    tp = record.get("throughput") or {}
    lens = tp.get("lens") or {}
    for leg in LENS_LEGS_REQUIRED:
        if not _num((lens.get(leg) or {}).get("tokens_per_s")):
            problems.append(f"throughput.lens.{leg}.tokens_per_s missing "
                            f"({(lens.get(leg) or {}).get('missing', 'no leg')})")
    # the BASE comparator rungs are deliberately NOT here: advisory_missing()
    prov = record.get("provenance") or {}
    if expected_sha and prov.get("checkout_sha") != expected_sha:
        problems.append(f"provenance.checkout_sha={prov.get('checkout_sha')!r} != staged {expected_sha}")
    if prov.get("served_model") != MODEL_HF or prov.get("served_revision") != MODEL_REV:
        problems.append(f"provenance served {prov.get('served_model')!r}@{prov.get('served_revision')!r}, "
                        f"want {MODEL_HF}@{MODEL_REV}")
    return problems


def advisory_missing(record: dict) -> dict:
    """Evidence-only legs the record could not measure, with the smoke's own
    reason — REPORTED (wrapper_record.json, RESULT), never a failure. Today:
    the two plugin-disabled base comparator rungs, which on Modal fail the
    frozen comparator's raw model-string assertion (module docstring)."""
    tp = record.get("throughput") or {}
    out = {}
    for leg in ADVISORY_BASE_LEGS:
        entry = tp.get(leg) or {}
        if not _num(entry.get("tokens_per_s")):
            out[f"throughput.{leg}"] = str(entry.get("missing") or "MISSING: no leg recorded")
    # selection_direct MISSING / PARTIAL: lost telemetry, advisory in the smoke
    # (selection_direct_failure) — a FAILED verdict is a gating problem instead
    for sd_key in ("selection_direct", "selection_direct_petri_geometry"):
        sd = record.get(sd_key)
        verdict = str((sd or {}).get("verdict") or "") if isinstance(sd, dict) else ""
        if verdict.startswith(("MISSING", "PARTIAL")):
            out[sd_key] = verdict
    return out


def positive_artifact_summary(record: dict) -> dict:
    """The positives a reader wants at a glance (never a verdict by itself)."""
    fid = record.get("fidelity") or {}
    fidp = record.get("fidelity_petri_geometry") or {}
    phases = (record.get("fidelity_phases") or {}).get("phases") or {}
    kv = (((record.get("certified_geometries") or {}).get("fidelity_petri_16384") or {})
          .get("kv_cache") or {})
    tp = record.get("throughput") or {}
    legs = {k: (v or {}).get("tokens_per_s") for k, v in (tp.get("lens") or {}).items()}
    legs.update({k: (tp.get(k) or {}).get("tokens_per_s") for k in ADVISORY_BASE_LEGS})
    return {
        "status": record.get("status"),
        "fidelity_checks": fid.get("checks"), "fidelity_passed": fid.get("passed"),
        "fidelity_waived_failures": record.get("fidelity_waived_failures"),
        "fidelity_petri_checks": fidp.get("checks"), "fidelity_petri_passed": fidp.get("passed"),
        "fidelity_petri_waived_failures": record.get("fidelity_petri_geometry_waived_failures"),
        "phase_outcomes": {k: (v or {}).get("outcome") for k, v in phases.items()},
        "petri16k_kv_cache_tokens": kv.get("kv_cache_tokens"),
        "petri16k_full_length_concurrency": kv.get("full_length_concurrency"),
        "arms": {a.get("cell_id"): {"required": a.get("required"), "served_sha16": a.get("served_sha16")}
                 for a in (record.get("arm_derivations") or []) if isinstance(a, dict)},
        "tokens_per_s": legs,
        "response_mode_arm_served": (record.get("response_mode_arm") or {}).get("served_arm_ids"),
        "selection_direct": ((record.get("selection_direct") or {}).get("verdict")
                             if isinstance(record.get("selection_direct"), dict) else None),
        "checkout_sha": (record.get("provenance") or {}).get("checkout_sha"),
        "chat_markers_shim": record.get("chat_markers_shim"),
    }


def classify_outcome(out_dir: str, rc: int | None, *, timed_out: bool = False,
                     expected_sha: str | None = None) -> dict:
    """The wrapper's verdict over the smoke's out-dir + exit code. PASS needs
    the positive artifacts; every unavailable / failed / silent / timed-out
    shape is a failure with its reason (the module docstring, OUTCOME)."""
    out = Path(out_dir)
    markers = {n: (out / n).is_file() for n in (UNAVAILABLE_MARKER, FAILURE_MARKER,
                                                 SMOKE_RECORD, EVIDENCE_FILE)}

    def _reason(name: str) -> str:
        try:
            doc = json.loads((out / name).read_text())
            return str(doc.get("reason") or doc.get("status") or "(no reason recorded)")
        except Exception as e:  # noqa: BLE001
            return f"(unreadable marker: {type(e).__name__}: {e})"

    res = {"verdict": None, "ok": False, "reason": None, "rc": rc, "markers": markers,
           "positives": {}, "problems": [], "advisory_missing": {}}
    if timed_out:
        res.update(verdict="TIMEOUT", reason=f"the smoke did not finish inside {SMOKE_DEADLINE_S}s "
                   "and was killed (partial out-dir synced)")
    elif markers[UNAVAILABLE_MARKER]:
        res.update(verdict="UNAVAILABLE", reason=f"{UNAVAILABLE_MARKER}: {_reason(UNAVAILABLE_MARKER)} "
                   f"(the smoke exits 0 here; that is a launch blocker, NOT a pass)")
    elif markers[FAILURE_MARKER]:
        res.update(verdict="FAIL", reason=f"{FAILURE_MARKER}: {_reason(FAILURE_MARKER)}")
    elif rc != 0:
        res.update(verdict="FAIL", reason=f"exit {rc} with no marker written")
    elif not markers[SMOKE_RECORD]:
        res.update(verdict="FAIL", reason=f"exit 0 but no {SMOKE_RECORD} — nothing positive to certify")
    else:
        try:
            doc = json.loads((out / SMOKE_RECORD).read_text())
        except Exception as e:  # noqa: BLE001
            doc, problems = {}, [f"{SMOKE_RECORD} unreadable: {type(e).__name__}: {e}"]
        else:
            problems = positive_artifact_problems(doc, expected_sha=expected_sha)
        if not markers[EVIDENCE_FILE]:
            problems.append(f"{EVIDENCE_FILE} missing beside the record")
        res["positives"] = positive_artifact_summary(doc) if doc else {}
        res["problems"] = problems
        res["advisory_missing"] = advisory_missing(doc) if doc else {}
        if problems:
            res.update(verdict="FAIL", reason="positive artifacts incomplete: " + "; ".join(problems))
        else:
            adv = res["advisory_missing"]
            res.update(verdict="PASS", ok=True,
                       reason=f"{SMOKE_RECORD} complete: both gating phases evaluated, non-waivable "
                              "checks passed at 2048 and 16384 geometries, both real arms derived, "
                              "lens throughput legs measured"
                              + (f"; advisory legs MISSING (evidence only, not gating): {sorted(adv)}"
                                 if adv else "; every advisory leg measured too"))
    return res


def modal_exit_code(rec: dict) -> int:
    """0 only for a PUBLISHED PASS: verdict PASS, status DONE, an explicit
    ``publication_ok`` (set by ``_publish`` only after the evidence was synced
    and committed, ``current.json`` was promoted to this attempt and read back
    equal, and RESULT landed) and ``current.attempt`` naming this very
    attempt. Everything else — FAIL, UNAVAILABLE, TIMEOUT, a setup refusal,
    PUBLISH_FAILED, or a PASS record without the publication facts (an older
    wrapper's record via --harvest) — is 1. A successful Modal command that
    merely printed a pass is the trap; so is a pass the volume does not hold."""
    attempt = rec.get("attempt")
    current = rec.get("current") if isinstance(rec.get("current"), dict) else {}
    return 0 if (rec.get("verdict") == "PASS" and rec.get("status") == "DONE"
                 and rec.get("publication_ok") is True and attempt
                 and current.get("attempt") == attempt) else 1


def result_line(rec: dict, *, publication: bool = True) -> str:
    """One line: verdict, exit, timing, identity, the attempt id, whether THIS
    attempt's publication completed (``published=true`` only from a record
    whose ``publication_ok`` is True — the RESULT a reader may trust), reason
    — and every advisory leg that went unmeasured with the smoke's reason, so
    a reader of RESULT alone sees that base-vs-lens throughput is not in this
    record. ``publication=False`` omits the ``published=`` token: the line
    inside ``current.json`` is written mid-publication, and that pointer's
    existence is the promotion itself."""
    adv = rec.get("advisory_missing") or {}
    adv_txt = ("" if not adv else " advisory_missing=" + json.dumps(adv, sort_keys=True))
    pub = "" if not publication else f"published={'true' if rec.get('publication_ok') is True else 'false'} "
    return (f"{rec.get('verdict') or rec.get('status')} exit={rec.get('smoke_rc')} "
            f"elapsed_s={rec.get('elapsed_s')} app={rec.get('app')} sha={rec.get('sha')} "
            f"image={rec.get('image')} attempt={rec.get('attempt')} {pub}reason={rec.get('reason')}{adv_txt}\n")


# --------------------------------------------------------------------------- #
# staging (lens_vectorize_gate.py's helpers, duplicated — see the docstring)   #
# --------------------------------------------------------------------------- #


def local_repo_root() -> Path:
    """The git repo this file lives in (the worktree's COMMON dir's parent —
    every worktree shares its objects, so any sha reachable anywhere in the
    repo can be fetched from it)."""
    here = Path(__file__).resolve().parent
    common = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                            cwd=str(here), capture_output=True, text=True, check=True).stdout.strip()
    return Path(common).parent


def shallow_clone_tarball(sha: str, dest_dir: Path) -> Path:
    """A SHALLOW standalone clone of ``sha`` from the local repo, HEAD detached
    at the sha, verified, tarred as ``dest_dir/sisteer-<sha>.tar.gz`` whose
    single top-level entry is ``sisteer-<sha>/``."""
    src = local_repo_root()
    if not _git(str(src), "cat-file", "-t", sha) == "commit":
        raise SystemExit(f"[petri-smoke] {sha} is not a commit in {src}")
    work = Path(tempfile.mkdtemp(prefix="petri-smoke-stage-"))
    tree = work / f"sisteer-{sha}"
    tree.mkdir()
    subprocess.run(["git", "init", "-q", str(tree)], check=True)
    subprocess.run(["git", "-c", "uploadpack.allowAnySHA1InWant=true", "-C", str(tree),
                    "fetch", "-q", "--depth", "1", str(src), sha], check=True)
    subprocess.run(["git", "-C", str(tree), "checkout", "-q", "--detach", "FETCH_HEAD"], check=True)
    v = verify_checkout(str(tree), sha)
    if not v["ok"]:
        raise SystemExit(f"[petri-smoke] shallow clone of {sha} failed verification: {v}")
    tar_path = dest_dir / f"sisteer-{sha}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(str(tree), arcname=f"sisteer-{sha}")
    shutil.rmtree(work, ignore_errors=True)
    return tar_path


@app.function(image=image_tf5141, volumes={VOL_PATH: VOL}, timeout=1800, cpu=4)
def unpack_tree(sha: str, force: bool = False) -> dict:
    """Extract ``/vol/repos/sisteer-<sha>.tar.gz`` (pushed by --stage) into
    ``/vol/repos/sisteer-<sha>``, verify the checkout ON the volume (HEAD ==
    sha, clean), delete the tarball, commit. An existing dir is verified and
    kept unless ``force``."""
    VOL.reload()
    dst = Path(REPOS) / f"sisteer-{sha}"
    tar_path = Path(REPOS) / f"sisteer-{sha}.tar.gz"
    out = {"sha": sha, "tree": str(dst)}
    if dst.exists() and not force:
        v = verify_checkout(str(dst), sha)
        out.update(existing=True, verify=v)
        if tar_path.exists():
            tar_path.unlink()
            VOL.commit()
        if not v["ok"]:
            out["error"] = (f"{dst} exists but does not verify ({v['why']}) — a tree from another "
                            "lane without .git, or the wrong sha; --restage replaces it")
        return out
    if not tar_path.is_file():
        out["error"] = f"no tarball at {tar_path} — run --stage first"
        return out
    tmp = Path(REPOS) / f".unpack-{sha}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(str(tmp), filter="fully_trusted")
    extracted = tmp / f"sisteer-{sha}"
    if not (extracted / ".git").exists():
        out["error"] = f"tarball carries no .git under sisteer-{sha}/"
        shutil.rmtree(tmp, ignore_errors=True)
        return out
    if dst.exists():
        shutil.rmtree(dst)
    extracted.rename(dst)
    shutil.rmtree(tmp, ignore_errors=True)
    tar_path.unlink()
    v = verify_checkout(str(dst), sha)
    out.update(existing=False, verify=v)
    if not v["ok"]:
        out["error"] = f"unpacked tree does not verify: {v['why']}"
    VOL.commit()
    return out


# --------------------------------------------------------------------------- #
# preflight (CPU): tree, weights, the two objects, pins, argv-vs-sbatch drift   #
# --------------------------------------------------------------------------- #


def manifest_serving_packages(tree: str) -> dict:
    import yaml
    doc = yaml.safe_load(Path(tree, MANIFEST).read_text())
    return dict((doc.get("petri") or {}).get("serving_packages") or {})


def _preflight(sha: str, static_r_object: str, alphasteer_object: str, variant: str) -> dict:
    VOL.reload()
    rep = {"sha": sha, "image": variant, "ok": True, "checks": {},
           "objects": {"static_r_matched": static_r_object, "alphasteer_official": alphasteer_object}}

    def chk(name, fn):
        try:
            rep["checks"][name] = {"ok": True, **(fn() or {})}
        except BaseException as e:  # noqa: BLE001
            rep["checks"][name] = {"ok": False, "err": f"{type(e).__name__}: {e}"[:1200]}
            rep["ok"] = False

    tree = f"{REPOS}/sisteer-{sha}"

    def tree_check():
        v = verify_checkout(tree, sha)
        if not v["ok"]:
            raise AssertionError(f"{tree}: {v['why']} — run --stage")
        for rel in (SMOKE_SCRIPT, SBATCH_SCRIPT, MANIFEST, "src/sisteer",
                    "configs/capeval/vllm-requirements.txt"):
            if not Path(tree, rel).exists():
                raise AssertionError(f"tree lacks {rel}")
        return {"tree": tree, "verify": v}
    chk("tree", tree_check)

    def argv_vs_sbatch():
        # the staged tree's OWN sbatch vs this wrapper's command: a changed
        # flag at that sha is a stale wrapper, refused here before any GPU.
        # The sbatch serves the site's objects through its object variables,
        # which translate to the objects this run serves; an older tree's
        # sbatch spelled their cluster paths instead, and a run that names
        # other copies (hash-verified below) compares with both object values
        # masked, as before
        text = Path(tree, SBATCH_SCRIPT).read_text()
        objects = {"static_r_matched": static_r_object, "alphasteer_official": alphasteer_object}
        want = sbatch_tokens_translated(text, "<out-dir>", objects)
        got = smoke_argv("python", "<out-dir>", static_r_object, alphasteer_object)[1:]
        if want != got and not set(SBATCH_OBJECT_VARS) & set(sbatch_smoke_tokens(text)):
            def mask(toks):
                t = list(toks)
                for flag in ("--static-r-object", "--alphasteer-object"):
                    if flag in t:
                        t[t.index(flag) + 1] = "<object>"
                return t
            want, got = mask(want), mask(got)
        if want != got:
            raise AssertionError(f"wrapper argv != the staged sbatch's command:\n  sbatch: {want}\n  here:   {got}")
        return {"tokens": len(got)}
    chk("argv_matches_sbatch", argv_vs_sbatch)

    def snapshot():
        snap = Path(HF_HOME) / snapshot_rel(MODEL_HF, MODEL_REV)
        if not (snap / "config.json").is_file():
            raise AssertionError(f"pinned snapshot {snap} has no config.json — the weights are not on the volume")
        return {"snapshot": str(snap)}
    chk("pinned_snapshot", snapshot)

    def objects():
        # IDENTITY, not existence: the volume file is hashed here, in the
        # container, against the pinned originals — an override path that
        # names a same-kind look-alike is refused before any GPU
        return {"objects": {name: verify_object(name, p) for name, p in
                            (("static_r_matched", static_r_object),
                             ("alphasteer_official", alphasteer_object))}}
    chk("objects", objects)

    def pins():
        import importlib.metadata as md
        have = {"vllm": md.version("vllm"), "vllm-lens": md.version("vllm-lens"),
                "torch": md.version("torch"), "transformers": md.version("transformers")}
        want = manifest_serving_packages(tree)
        bad = {k: (have.get(k), v) for k, v in want.items()
               if not str(have.get(k, "")).startswith(str(v).split("+")[0])}
        if bad:
            raise AssertionError(f"image pins differ from the manifest's serving_packages: {bad}")
        if have["transformers"] != TRANSFORMERS_PIN[variant]:
            raise AssertionError(f"image transformers {have['transformers']} != {TRANSFORMERS_PIN[variant]} ({variant})")
        return {"have": have, "manifest": want}
    chk("image_pins", pins)
    return rep


@app.function(image=image_tf5141, volumes={VOL_PATH: VOL}, timeout=1800, cpu=4)
def preflight_tf5141(sha: str, static_r_object: str, alphasteer_object: str) -> dict:
    return _preflight(sha, static_r_object, alphasteer_object, "tf5141")


@app.function(image=image_tf5150, volumes={VOL_PATH: VOL}, timeout=1800, cpu=4)
def preflight_tf5150(sha: str, static_r_object: str, alphasteer_object: str) -> dict:
    return _preflight(sha, static_r_object, alphasteer_object, "tf5150")


PREFLIGHT = {"tf5141": preflight_tf5141, "tf5150": preflight_tf5150}


# --------------------------------------------------------------------------- #
# the smoke, one H200                                                          #
# --------------------------------------------------------------------------- #

#: never copied to the volume: the smoke's derived registry blobs (alphasteer's
#: d x d matrices run to hundreds of MB) — the JSON records name their sha16s
SKIP_SUFFIXES = (".pt", ".safetensors", ".bin")
SKIP_OVER_BYTES = 64 << 20


def _copy_tree(sha: str) -> str:
    """The staged tree copied to local disk and verified there (git on a
    shared volume view is slow and would rewrite the index in place)."""
    src, dst = Path(REPOS) / f"sisteer-{sha}", Path(TREES_ROOT) / f"sisteer-{sha}"
    if not (src / ".git").exists():
        raise RuntimeError(f"{src} is not a git checkout (no .git) — run --stage for {sha}")
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst, symlinks=True)
    v = verify_checkout(str(dst), sha)
    if not v["ok"]:
        raise RuntimeError(f"local copy of {sha} does not verify: {v['why']}")
    return str(dst)


def _sync_results(work_dir: Path, vol_dir: Path) -> int:
    """Copy the work dir (the smoke's out-dir, smoke.log, RESULT, the wrapper
    record) into the volume dir — small files only — and commit."""
    n = 0
    for p in work_dir.rglob("*"):
        if not p.is_file() or p.suffix in SKIP_SUFFIXES or p.stat().st_size > SKIP_OVER_BYTES:
            continue
        rel = p.relative_to(work_dir)
        dst = vol_dir / rel
        try:
            if dst.is_file() and dst.stat().st_size == p.stat().st_size and dst.stat().st_mtime >= p.stat().st_mtime:
                continue
        except OSError:
            pass
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
        n += 1
    VOL.commit()
    return n


#: EVERY run writes only to its own ``<sha>/attempts/<attempt-id>/`` — never a
#: shared dir, so two runs on one sha cannot interleave markers, logs or
#: records (the empty-top-level check that preceded this was not proof that no
#: other writer was active). Attempts are immutable and all kept; a COMPLETED
#: attempt is promoted by atomically replacing the small ``current.json``
#: pointer beside ``attempts/``, which every reader (the mirror, --harvest, an
#: analysis) resolves. No retirement step exists any more.
ATTEMPTS_SUBDIR = "attempts"
CURRENT_POINTER = "current.json"


def attempt_id() -> str:
    """A collision-proof id for one attempt: epoch + pid + 32 random bits, so
    two runs in the same second can never name the same directory."""
    return f"{int(time.time())}-{os.getpid()}-{secrets.token_hex(4)}"


def attempt_paths(sha: str, attempt: str) -> tuple:
    """``(sha dir on the volume, this attempt's own dir under attempts/, its
    container-local work dir)``."""
    vol_dir = Path(RESULTS_ROOT) / sha
    return vol_dir, vol_dir / ATTEMPTS_SUBDIR / attempt, Path(WORK_ROOT) / sha / attempt


def current_pointer(vol_dir: Path, attempt: str, rec: dict) -> dict:
    """The content of ``current.json``: which attempt is current, its verdict,
    status, RESULT line and identity — enough for a reader to act without
    opening the attempt. ``_publish`` fixes it BEFORE the attempt's files are
    written, so wrapper_record.json carries the very pointer that will publish
    it, and writes the pointer byte-identical afterwards (``promoted_at`` is
    therefore the moment the content was fixed)."""
    return {"attempt": attempt, "path": f"{ATTEMPTS_SUBDIR}/{attempt}", "verdict": rec.get("verdict"),
            "status": rec.get("status"), "app": rec.get("app"), "sha": rec.get("sha"),
            "image": rec.get("image"), "result": result_line(rec, publication=False).rstrip("\n"),
            "promoted_at": time.time()}


class PointerInstalledError(RuntimeError):
    """``_promote_current`` replaced the pointer and THEN failed (the commit
    raised; the pointer could not be read back; or it reads back naming this
    attempt with other content): ``current.json`` may durably name this
    attempt, so its files — final, and in agreement with that pointer — must
    not be touched by any recovery."""


def _promote_current(vol_dir: Path, attempt: str, rec: dict, pointer: dict | None = None) -> dict:
    """Atomically point ``<sha>/current.json`` at a COMPLETED attempt whose
    files are already final on the volume — the LAST DURABLE WRITE of a
    publication: written to a temp file in the same dir and ``os.replace``d
    over the old pointer (a reader sees the previous pointer or the new one,
    never a partial file), committed, then READ BACK equal. ``pointer`` is the
    content the attempt's own record already carries (``_publish``); computed
    here when absent. A failure BEFORE the replace raises plainly (nothing
    names the attempt); a failure AFTER it raises ``PointerInstalledError``
    (the pointer may be in place) — except a read-back naming ANOTHER attempt,
    which proves this pointer is not in place (a lost or overtaken write) and
    raises plainly. The attempt dirs themselves are never touched."""
    pointer = pointer or current_pointer(vol_dir, attempt, rec)
    vol_dir.mkdir(parents=True, exist_ok=True)
    tmp = vol_dir / f".{CURRENT_POINTER}.{attempt}.tmp"
    tmp.write_text(json.dumps(pointer, indent=1, default=str))
    os.replace(tmp, vol_dir / CURRENT_POINTER)
    try:
        VOL.commit()
        back = json.loads((vol_dir / CURRENT_POINTER).read_text())
        if not isinstance(back, dict):
            raise TypeError(f"pointer is a {type(back).__name__}, not a mapping")
    except Exception as e:  # noqa: BLE001
        raise PointerInstalledError(f"{CURRENT_POINTER} replaced but not confirmed: "
                                    f"{type(e).__name__}: {e}") from e
    if back.get("attempt") != attempt:
        raise RuntimeError(f"{CURRENT_POINTER} read back names attempt {back.get('attempt')!r}, not "
                           f"{attempt!r}: this pointer is not in place")
    if back != pointer:
        raise PointerInstalledError(f"{CURRENT_POINTER} read back names {attempt!r} but differs from what "
                                    "was written")
    return pointer


def required_attempt_files(verdict) -> list:
    """What an attempt must have landed on the volume, relative to its own
    dir, before it may be called published: RESULT and wrapper_record.json
    always; for a PASS also the smoke's record and evidence — the artifacts
    a PASS claims to rest on. ONE list serves the publisher (checked after
    the final sync) and the mirror (``required_mirror_files``)."""
    req = ["RESULT", "wrapper_record.json"]
    if verdict == "PASS":
        req += [f"{OUT_SUBDIR}/{SMOKE_RECORD}", f"{OUT_SUBDIR}/{EVIDENCE_FILE}"]
    return req


def _write_wrapper_files(rec: dict, work_dir: Path) -> None:
    (work_dir / "RESULT").write_text(result_line(rec))
    (work_dir / "wrapper_record.json").write_text(json.dumps(rec, indent=1, default=str))


def _require_landed(work_dir: Path, attempt_dir: Path, rels) -> None:
    """Every ``rel`` is present in the attempt dir on the volume, byte-equal to
    its work-dir original — the sync is checked, not trusted."""
    missing = []
    for rel in rels:
        src, dst = work_dir / rel, attempt_dir / rel
        if not dst.is_file() or (src.is_file() and dst.read_bytes() != src.read_bytes()):
            missing.append(rel)
    if missing:
        raise RuntimeError(f"not landed in {attempt_dir}: {missing}")


class SyncWorker:
    """The periodic copy of the running smoke's out-dir into its attempt dir —
    and the ONE lock every sync of that attempt takes, periodic or final.
    Publication (``_publish``) first asks it to QUIESCE: stop, join for a
    bounded time, and prove the thread ended. A worker still alive after the
    bound is a copy or commit stalled on the volume that could resume and
    land a STALE copy (the frozen smoke rewrites ``smoke_evidence.json``
    throughout) after the final bytes were verified, or after the pointer —
    breaking "the pointer is the last durable write". Publication then fails
    closed rather than proceed (the previous ``join(timeout)`` never checked
    ``is_alive``)."""

    def __init__(self, work_dir: Path, attempt_dir: Path, every_s: float = SYNC_EVERY_S):
        self.work_dir, self.attempt_dir, self.every_s = work_dir, attempt_dir, every_s
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, name="sync", daemon=True)
        self.failures = 0

    def start(self) -> "SyncWorker":
        self.thread.start()
        return self

    def _loop(self) -> None:
        while not self.stop.wait(self.every_s):
            with self.lock:
                try:
                    _sync_results(self.work_dir, self.attempt_dir)
                except Exception as e:  # noqa: BLE001 — a sync failure never stops the smoke
                    self.failures += 1
                    print(f"[petri-smoke] periodic sync failed: {type(e).__name__}: {e}", flush=True)

    def quiesce(self, timeout: float) -> bool:
        """Tell the worker to stop and wait up to ``timeout`` s for its thread
        to END. True only if it did (a never-started worker counts as ended);
        False means a sync is still in flight — the caller must not write."""
        self.stop.set()
        if self.thread.is_alive():
            self.thread.join(timeout)
        return not self.thread.is_alive()


def _publish(rec: dict, work_dir: Path, attempt_dir: Path, vol_dir: Path, attempt: str, *,
             promote: bool, syncer: "SyncWorker | None" = None) -> None:
    """Publication, serialized against the periodic sync. With a ``syncer``,
    first QUIESCE it (``SyncWorker.quiesce``, bounded by ``SYNC_QUIESCE_S``);
    a worker still alive then means publication FAILS CLOSED: nothing is
    written anywhere — not the volume, not even the work dir — and no pointer
    is installed; a completed attempt becomes PUBLISH_FAILED naming the
    un-quiesced sync, an incomplete one records ``publish_error``. Otherwise
    the whole publication (``_publish_locked``) runs under the syncer's lock,
    so no periodic copy can interleave with the final sync, the verification
    or the pointer. Nothing is raised: the record is the Modal return value
    either way."""
    if syncer is not None:
        quiet = syncer.quiesce(SYNC_QUIESCE_S)
        rec["periodic_sync_quiesced"] = quiet
        if not quiet:
            rec["publication_ok"] = False
            rec["current"] = None
            rec["publish_error"] = (f"periodic sync did not quiesce within {SYNC_QUIESCE_S}s (a copy or commit "
                                    "stalled on the volume) — nothing written, no pointer installed")
            if promote:
                rec["smoke_verdict"] = rec.get("verdict")
                rec.update(verdict="PUBLISH_FAILED", status="PUBLISH_FAILED", pointer_may_be_installed=False,
                           reason=f"publication failed — {rec['publish_error']}; the smoke's own verdict was "
                                  f"{rec['smoke_verdict']}: {rec.get('reason')}")
            return
    lock = syncer.lock if syncer is not None else contextlib.nullcontext()
    with lock:
        _publish_locked(rec, work_dir, attempt_dir, vol_dir, attempt, promote=promote)


def _publish_locked(rec: dict, work_dir: Path, attempt_dir: Path, vol_dir: Path, attempt: str, *,
                    promote: bool) -> None:
    """The publication proper (called by ``_publish`` under the sync lock).
    Publication is part of the exit contract, and the pointer is its LAST
    durable write. For a COMPLETED attempt (``promote``): (1) the pointer
    content is fixed and the attempt is written in its FINAL form — RESULT
    (``published=true attempt=<id>``), wrapper_record.json carrying that
    pointer under ``current`` with ``publication_ok`` true, beside the
    smoke's record + evidence — then synced, committed and byte-verified
    (``_require_landed``); nothing names the attempt yet, so no reader can see
    any of it; (2) ``current.json`` is replaced with that very content,
    committed and read back equal (``_promote_current``). After the replace
    NOTHING touches the attempt: its files and the pointer agree by
    construction, and only a pointer makes an attempt current. Only when both
    held does the RETURNED record keep ``publication_ok`` — the flag
    ``modal_exit_code`` demands. ANY failure makes the returned record status
    = verdict = ``PUBLISH_FAILED`` (``publication_ok`` false, ``current``
    None, the smoke's own verdict under ``smoke_verdict``). A failure BEFORE
    the replace (the sync, the verification, the temp write) leaves an attempt
    nothing names; it is rewritten, best effort, to say PUBLISH_FAILED
    (``published=false``) so a reader of attempts/ sees why it is not
    current. A failure AFTER the replace (``PointerInstalledError``) leaves
    the attempt EXACTLY as it is (``pointer_may_be_installed`` true): the
    pointer may durably name it and its files agree with that pointer; a
    rewrite here would recreate the PASS-pointer-beside-PUBLISH_FAILED-files
    tree. The run still exits 1 — it could not confirm the pointer — and if
    the pointer did land, the volume holds a consistent, verified publication
    a later run's mirror pull picks up. An attempt that did not complete
    (``promote`` False) lands RESULT + record in its own dir with
    ``published=false``; a failure there is ``publish_error`` (its exit is
    already non-zero). Nothing is raised: the record is the Modal return
    value either way."""
    rec["publication_ok"] = False
    rec["current"] = None
    verdict, reason = rec.get("verdict"), rec.get("reason")
    try:
        work_dir.mkdir(parents=True, exist_ok=True)
        if not promote:
            _write_wrapper_files(rec, work_dir)
            rec["synced"] = _sync_results(work_dir, attempt_dir)
            _require_landed(work_dir, attempt_dir, ["RESULT", "wrapper_record.json"])
            return
        # 1. the COMPLETE attempt in its FINAL form: synced, committed, byte-verified —
        #    while nothing names it
        pointer = current_pointer(vol_dir, attempt, rec)
        rec["publication_ok"] = True
        rec["current"] = pointer
        _write_wrapper_files(rec, work_dir)
        rec["synced"] = _sync_results(work_dir, attempt_dir)
        _require_landed(work_dir, attempt_dir, required_attempt_files(verdict))
        # 2. the pointer, byte-identical to what the record carries — the LAST durable
        #    write; from here on the attempt is immutable
        _promote_current(vol_dir, attempt, rec, pointer)
    except Exception as e:  # noqa: BLE001
        rec["publication_ok"] = False
        rec["current"] = None
        rec["publish_error"] = f"{type(e).__name__}: {e}"[:1500]
        if not promote:
            return
        rec["smoke_verdict"] = verdict
        rec.update(verdict="PUBLISH_FAILED", status="PUBLISH_FAILED",
                   reason=f"publication failed — {rec['publish_error'][:600]}; the smoke's own verdict "
                          f"was {verdict}: {reason}")
        if isinstance(e, PointerInstalledError):
            rec["pointer_may_be_installed"] = True      # the attempt stays exactly as it is
            return
        rec["pointer_may_be_installed"] = False
        try:  # nothing names this attempt: it may say why it is not current (best effort)
            _write_wrapper_files(rec, work_dir)
            _sync_results(work_dir, attempt_dir)
        except Exception as e2:  # noqa: BLE001
            rec["publish_error"] += f"; the PUBLISH_FAILED record could not be synced either: {type(e2).__name__}: {e2}"[:600]


def _run_smoke(sha: str, run_name: str, static_r_object: str, alphasteer_object: str,
               variant: str) -> dict:
    T0 = time.time()
    VOL.reload()
    attempt = attempt_id()
    vol_dir, attempt_dir, work_dir = attempt_paths(sha, attempt)
    out_dir = work_dir / OUT_SUBDIR
    rec = {"sha": sha, "app": run_name, "image": variant, "attempt": attempt, "results_dir": str(vol_dir),
           "attempt_dir": str(attempt_dir), "current": None,
           "objects": {"static_r_matched": static_r_object, "alphasteer_official": alphasteer_object},
           "node": os.uname().nodename}
    try:
        # THIS attempt's own dir on the volume, created exclusively: no run
        # ever writes where another might; a collision (impossible by
        # construction of the id) FAILS CLOSED before any engine.
        attempt_dir.mkdir(parents=True, exist_ok=False)
        VOL.commit()
    except Exception as e:  # noqa: BLE001
        rec.update(status="ATTEMPT_DIR_EXISTS", verdict="ATTEMPT_DIR_EXISTS",
                   error=f"{type(e).__name__}: {e}"[:1500], reason=f"{type(e).__name__}: {e}"[:600],
                   smoke_rc=None, elapsed_s=round(time.time() - T0, 1))
        return rec
    import importlib.metadata as md
    rec["pins"] = {k: md.version(k) for k in ("vllm", "vllm-lens", "torch", "transformers")}
    try:
        if rec["pins"]["transformers"] != TRANSFORMERS_PIN[variant]:
            raise RuntimeError(f"image transformers {rec['pins']['transformers']} != "
                               f"{TRANSFORMERS_PIN[variant]} ({variant})")
        tree = _copy_tree(sha)
        # the authoritative KV verdict is imported from THIS tree at classification
        SISTEER_SRC_CANDIDATES.insert(0, f"{tree}/src")
        snap = Path(HF_HOME) / snapshot_rel(MODEL_HF, MODEL_REV)
        if not (snap / "config.json").is_file():
            raise RuntimeError(f"pinned snapshot {snap} has no config.json")
        # IDENTITY, re-hashed in THIS container before any engine: the
        # preflight's verdict is not carried over on trust (the smoke itself
        # would record a missing object as unavailable and exit 0 — the trap)
        rec["object_hashes"] = {name: verify_object(name, p) for name, p in rec["objects"].items()}
    except Exception as e:  # noqa: BLE001
        rec.update(status="SETUP_FAILED", verdict="SETUP_FAILED", error=f"{type(e).__name__}: {e}"[:1500],
                   reason=f"{type(e).__name__}: {e}"[:600], smoke_rc=None,
                   elapsed_s=round(time.time() - T0, 1))
        _publish(rec, work_dir, attempt_dir, vol_dir, attempt, promote=False)
        return rec

    shutil.rmtree(work_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    argv = smoke_argv(sys.executable, str(out_dir), static_r_object, alphasteer_object)
    env = smoke_env(os.environ, tree)
    rec["argv"] = argv
    rec["cwd"] = tree
    print(f"[petri-smoke] {sha[:12]} ({variant}: " + ", ".join(f"{k} {v}" for k, v in rec["pins"].items())
          + f"); attempt {attempt} -> {attempt_dir}", flush=True)
    print("[petri-smoke] cwd=" + tree + "  " + " ".join(argv), flush=True)

    syncer = SyncWorker(work_dir, attempt_dir).start()

    log = work_dir / "smoke.log"
    timed_out = False
    with open(log, "ab") as lf:
        proc = subprocess.Popen(argv, env=env, cwd=tree, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, start_new_session=True)
        assert proc.stdout is not None

        def _tee():
            for line in proc.stdout:
                lf.write(line)
                lf.flush()
                sys.stdout.write(line.decode("utf-8", "replace"))
                sys.stdout.flush()
        reader = threading.Thread(target=_tee, name="tee", daemon=True)
        reader.start()
        try:
            rc = proc.wait(timeout=SMOKE_DEADLINE_S)
        except subprocess.TimeoutExpired:
            timed_out = True
            print(f"[petri-smoke] internal deadline {SMOKE_DEADLINE_S}s hit — killing the smoke's "
                  "process group (engines included)", flush=True)
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                rc = proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                rc = proc.wait()
        reader.join(timeout=30)
    outcome = classify_outcome(str(out_dir), rc, timed_out=timed_out, expected_sha=sha)
    rec.update(smoke_rc=rc, timed_out=timed_out, elapsed_s=round(time.time() - T0, 1),
               verdict=outcome["verdict"], reason=outcome["reason"], markers=outcome["markers"],
               positives=outcome["positives"], problems=outcome["problems"],
               advisory_missing=outcome["advisory_missing"],
               status="DONE" if not timed_out else "TIMEOUT")
    rec["smoke_tail"] = log.read_text(errors="replace")[-4000:] if log.is_file() else None
    # publish into THIS attempt's dir once the periodic sync has quiesced; a
    # COMPLETED attempt (the smoke ran to its classified end) is promoted to
    # current.json atomically — and if any step of that publication fails, the
    # record becomes PUBLISH_FAILED (exit 1)
    _publish(rec, work_dir, attempt_dir, vol_dir, attempt, promote=(rec["status"] == "DONE"), syncer=syncer)
    return rec


def _run_named(sha, run_name, static_r_object, alphasteer_object, variant) -> dict:
    rec = _run_smoke(sha, run_name, static_r_object, alphasteer_object, variant)
    print("[petri-smoke] REC", json.dumps({k: v for k, v in rec.items() if k not in ("argv", "smoke_tail")},
                                          default=str), flush=True)
    return rec


# One named function per image variant (the image is fixed at decoration).
# 16 CPUs as the sbatch; 128 GiB (the lens gate's proven request; the sbatch
# asked 200G of a 1.9 TB node); one H200; the 60-minute cap from the brief.
@app.function(image=image_tf5141, volumes={VOL_PATH: VOL}, gpu="H200", cpu=16, memory=131072,
              timeout=SMOKE_TIMEOUT_S, retries=0)
def smoke_tf5141(sha: str, run_name: str, static_r_object: str, alphasteer_object: str) -> dict:
    return _run_named(sha, run_name, static_r_object, alphasteer_object, "tf5141")


@app.function(image=image_tf5150, volumes={VOL_PATH: VOL}, gpu="H200", cpu=16, memory=131072,
              timeout=SMOKE_TIMEOUT_S, retries=0)
def smoke_tf5150(sha: str, run_name: str, static_r_object: str, alphasteer_object: str) -> dict:
    return _run_named(sha, run_name, static_r_object, alphasteer_object, "tf5150")


SMOKE_FN = {"tf5141": smoke_tf5141, "tf5150": smoke_tf5150}


# --------------------------------------------------------------------------- #
# local side: mirror + state + entrypoint                                      #
# --------------------------------------------------------------------------- #


def _vol_read(path: str) -> bytes:
    return b"".join(VOL.read_file(path))


#: the grading host mirror layout: ``<mirror_root>/<sha>/versions/<pull-id>/`` (immutable,
#: one per pull) and ``<mirror_root>/<sha>/current`` -> ``versions/<pull-id>``
#: (a symlink replaced atomically). Older versions beyond this many are removed
#: (this many REMAIN, the current one among them).
MIRROR_KEEP_VERSIONS = 5
#: written inside each version dir as it is created: the pull's SEQUENCE number
#: (1 + the highest among the sha's existing versions) — the explicit,
#: monotonic order garbage collection uses. A version id (``attempt_id``:
#: epoch-pid-random) is a collision-proof NAME, not an order: pulls within one
#: second share the epoch and sort by pid/random, which let a freshly promoted
#: version fall among the "oldest", be skipped as current, and leave keep+1
#: behind (a flaky GC test, 2026-09-06).
MIRROR_PULL_META = ".mirror-pull.json"


def pull_seq(vdir: Path) -> int:
    """The version's pull sequence number; 0 for a version without the meta
    file (a pull from before sequencing), which therefore sorts oldest."""
    try:
        return int(json.loads((vdir / MIRROR_PULL_META).read_text()).get("seq") or 0)
    except (OSError, ValueError, AttributeError, TypeError):
        return 0


def required_mirror_files(pointer: dict) -> list:
    """What a promoted attempt must have landed before the mirror may point
    at it — ``required_attempt_files`` for the pointer's verdict, relative to
    the sha dir. A pointer promoted whose final RESULT never landed
    (PUBLISH_FAILED mid-way) is therefore refused here, not mirrored."""
    base = str(pointer.get("path") or "").strip("/")
    return [f"{base}/{rel}" for rel in required_attempt_files(pointer.get("verdict"))]


def pointed_attempt_problems(root: Path, pointer: dict, *, sha: str) -> list:
    """The pointed attempt's OWN files must agree with the pointer before the
    mirror may promote a pull — existence is not agreement: wrapper_record.json
    names the same attempt and verdict, status DONE, ``publication_ok`` true,
    and its ``current`` EQUALS the pointer as a whole (``_publish`` writes the
    record carrying the very pointer it then installs, so a pointer that
    differs in ANY field — sha, image, result, path, promoted_at — is one
    ``_promote_current`` would have rejected on read-back, and is refused
    here too); RESULT's first token is the verdict and the line carries
    ``attempt=<id>`` and ``published=true``; the pointer's own status is DONE,
    its ``path`` is ``attempts/<attempt>`` and its ``sha`` is the sha being
    mirrored. A PASS pointer beside files that say PUBLISH_FAILED (or another
    attempt's files) is refused here whatever the publisher did — the mirror
    does not rely on the publisher's discipline."""
    base = str(pointer.get("path") or "").strip("/")
    attempt, verdict = pointer.get("attempt"), pointer.get("verdict")
    problems = []
    if pointer.get("status") != "DONE":
        problems.append(f"pointer status {pointer.get('status')!r} != DONE")
    if pointer.get("path") != f"{ATTEMPTS_SUBDIR}/{attempt}":
        problems.append(f"pointer path {pointer.get('path')!r} != {ATTEMPTS_SUBDIR}/{attempt}")
    if pointer.get("sha") != sha:
        problems.append(f"pointer sha {pointer.get('sha')!r} != the mirrored sha {sha!r}")
    try:
        rec = json.loads((root / base / "wrapper_record.json").read_text())
        if not isinstance(rec, dict):
            raise TypeError(f"a {type(rec).__name__}, not a mapping")
    except Exception as e:  # noqa: BLE001
        return problems + [f"wrapper_record.json unreadable: {type(e).__name__}: {e}"]
    if rec.get("attempt") != attempt:
        problems.append(f"record attempt {rec.get('attempt')!r} != pointer {attempt!r}")
    if rec.get("verdict") != verdict:
        problems.append(f"record verdict {rec.get('verdict')!r} != pointer {verdict!r}")
    if rec.get("status") != "DONE":
        problems.append(f"record status {rec.get('status')!r} != DONE")
    if rec.get("publication_ok") is not True:
        problems.append(f"record publication_ok {rec.get('publication_ok')!r} is not True")
    cur = rec.get("current") if isinstance(rec.get("current"), dict) else {}
    if cur != pointer:
        differ = sorted(k for k in set(cur) | set(pointer) if cur.get(k) != pointer.get(k))
        problems.append(f"record current != pointer (fields differing: {differ})")
    try:
        line = (root / base / "RESULT").read_text()
    except Exception as e:  # noqa: BLE001
        return problems + [f"RESULT unreadable: {type(e).__name__}: {e}"]
    first = line.split(" ", 1)[0] if line else ""
    if first != verdict:
        problems.append(f"RESULT verdict token {first!r} != pointer {verdict!r}")
    if f" attempt={attempt} " not in line:
        problems.append(f"RESULT does not name attempt {attempt!r}")
    if " published=true " not in line:
        problems.append("RESULT does not say published=true")
    return problems


#: the cross-process lock serializing every LOCAL mirror transaction of one sha,
#: and how long a transaction waits for it before giving up (MIRROR_FAILED)
MIRROR_LOCK = ".mirror.lock"
MIRROR_LOCK_TIMEOUT_S = 120


class MirrorLockTimeout(RuntimeError):
    """Another mirror transaction on this sha held the lock past the bound."""


@contextlib.contextmanager
def _mirror_lock(sha_root: Path, timeout: float | None = None):
    """An exclusive ``fcntl.flock`` on ``<mirror_root>/<sha>/.mirror.lock``,
    taken by a bounded non-blocking poll (``MIRROR_LOCK_TIMEOUT_S`` unless
    given) and released on exit. flock is per OPEN FILE DESCRIPTION, so two
    threads of one process contend exactly as two processes do. It serializes
    the whole local transaction (``_mirror_transaction``): two same-sha
    ``--go``/``--harvest`` on one machine otherwise each replace ``current``,
    one reads the other's target after its own replace and still reports
    promoted, and GC acting on that stale observation can remove the version
    the other just promoted (round 7)."""
    timeout = MIRROR_LOCK_TIMEOUT_S if timeout is None else timeout
    sha_root.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(sha_root / MIRROR_LOCK), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise MirrorLockTimeout(f"could not take {sha_root / MIRROR_LOCK} within {timeout}s: another "
                                            "mirror transaction on this sha holds it") from None
                time.sleep(0.05)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


class MirrorSuperseded(RuntimeError):
    """The pulled ``current.json`` is not the pointer the caller installed: a
    concurrent publication replaced it. The publication itself succeeded; the
    mirror is left untouched rather than promote another attempt's evidence."""


def mirror_results(sha: str, mirror_root: Path, keep: int = MIRROR_KEEP_VERSIONS, *,
                   expected_pointer: dict | None = None) -> dict:
    """Pull ``/vol/petri-smoke/<sha>/`` (every file: ``current.json`` and every
    attempt) into an IMMUTABLE versioned dir ``<mirror_root>/<sha>/versions/
    <pull-id>/`` and promote it by atomically replacing the ``current`` symlink
    (a fresh symlink renamed over the old one). ALL-OR-NOTHING: if any listed
    file fails to read, or the pulled ``current.json`` is not
    ``expected_pointer`` as a whole (``MirrorSuperseded``, reported as
    ``superseded``), or the promoted attempt lacks a required file
    (``required_mirror_files``), or its own record and RESULT do not AGREE
    with the pointer (``pointed_attempt_problems``), or the volume has no
    ``current.json``, the partial version is removed and the existing mirror
    is left byte-identical. ``expected_pointer`` is how a run binds the pull
    to ITS publication (``mirror_after_run``); without it the pull takes
    whatever consistent pointer the volume holds. Everything local — from
    version allocation to garbage collection — runs under the sha's
    cross-process lock (``_mirror_lock``; a timeout returns ``lock_timeout``
    with the mirror untouched), and ``promoted`` is claimed only after
    re-verifying under that lock that ``current`` resolves to THIS pull's
    version and that version's ``current.json`` is the pointer verified.
    Promotion is visible only after every file has landed. Then exactly
    ``keep`` versions remain: the current one is excluded first and the
    oldest of the rest go, oldest by pull sequence (``MIRROR_PULL_META``),
    never by the id's lexical order."""
    src = f"{volume_client_path(RESULTS_ROOT)}/{sha}"
    sha_root = mirror_root / sha
    versions = sha_root / "versions"
    current = sha_root / "current"
    tag = attempt_id()
    vdir = versions / tag
    out = {"sha": sha, "current": str(current), "version": None, "pulled": [], "promoted": False,
           "superseded": False, "lock_timeout": False}
    try:
        entries = VOL.listdir(src, recursive=True)
    except Exception as e:  # noqa: BLE001
        return {**out, "error": f"listing failed: {type(e).__name__}: {e}", "note": "existing mirror left untouched"}
    files = []
    for e in entries:
        p = "/" + e.path.lstrip("/")   # the client lists without a leading slash
        if not p.startswith(src + "/"):
            continue
        entry_type = getattr(getattr(e, "type", None), "name", "")
        if entry_type == "DIRECTORY" or p.endswith("/"):
            continue
        files.append((p, p[len(src) + 1:]))
    if CURRENT_POINTER not in {rel for _, rel in files}:
        return {**out, "error": f"no {CURRENT_POINTER} under {src} — no attempt was promoted there "
                                "(or a pre-attempts layout)", "note": "existing mirror left untouched"}
    try:
        with _mirror_lock(sha_root):
            return _mirror_transaction(sha, files, sha_root, versions, current, tag, vdir, keep,
                                       expected_pointer, out)
    except MirrorLockTimeout as e:
        return {**out, "error": str(e), "lock_timeout": True, "note": "existing mirror left untouched"}


def _mirror_transaction(sha: str, files: list, sha_root: Path, versions: Path, current: Path, tag: str,
                        vdir: Path, keep: int, expected_pointer: dict | None, out: dict) -> dict:
    """The local half of ``mirror_results``, run UNDER the sha's mirror lock:
    the pre-versions check, version (sequence) allocation, the pull, the
    volume-side checks, the atomic promotion, its VERIFICATION, and only then
    garbage collection. ``promoted`` is claimed only when ``current`` is seen
    to resolve to this pull's version and that version's ``current.json``
    reads back as the pointer verified — observed under the lock, so no
    other transaction can have moved it in between."""
    if current.exists() and not current.is_symlink():
        return {**out, "error": f"{current} exists and is not a symlink — a pre-versions mirror; move it "
                                "aside by hand", "note": "existing mirror left untouched"}
    vdir.mkdir(parents=True, exist_ok=False)
    seq = 1 + max((pull_seq(d) for d in versions.iterdir() if d.is_dir() and d != vdir), default=0)
    (vdir / MIRROR_PULL_META).write_text(json.dumps({"seq": seq, "tag": tag, "pulled_at": time.time()}))
    pulled, pointer = [], None
    try:
        for p, rel in files:
            try:
                data = _vol_read(p)
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(f"read failed for {rel}: {type(e).__name__}: {e}") from e
            dest = vdir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            pulled.append(rel)
        pointer = json.loads((vdir / CURRENT_POINTER).read_text())
        if expected_pointer is not None and pointer != expected_pointer:
            raise MirrorSuperseded(
                f"the volume's {CURRENT_POINTER} names attempt {pointer.get('attempt')!r} "
                f"(promoted_at {pointer.get('promoted_at')}), not this run's "
                f"{expected_pointer.get('attempt')!r} (promoted_at {expected_pointer.get('promoted_at')}): a "
                "concurrent publication replaced it after this run's read-back; the mirror is not moved to "
                "evidence this run did not install")
        missing = [r for r in required_mirror_files(pointer) if not (vdir / r).is_file()]
        if missing:
            raise RuntimeError(f"promoted attempt {pointer.get('attempt')!r} ({pointer.get('verdict')}) "
                               f"lacks {missing}")
        disagreements = pointed_attempt_problems(vdir, pointer, sha=sha)
        if disagreements:
            raise RuntimeError(f"promoted attempt {pointer.get('attempt')!r} ({pointer.get('verdict')}) "
                               f"disagrees with the pointer — refused: {disagreements}")
    except Exception as e:  # noqa: BLE001
        shutil.rmtree(vdir, ignore_errors=True)
        return {**out, "error": str(e), "superseded": isinstance(e, MirrorSuperseded), "pointer": pointer,
                "expected_pointer": expected_pointer, "note": "existing mirror left untouched", "pulled": pulled}
    # ATOMIC promotion: a new symlink renamed over the old one (POSIX rename)
    tmp_link = sha_root / f".current.{tag}.tmp"
    if tmp_link.is_symlink() or tmp_link.exists():
        tmp_link.unlink()
    os.symlink(os.path.join("versions", tag), tmp_link)
    os.replace(tmp_link, current)
    # VERIFY under the lock, before any claim and before any garbage collection:
    # current resolves to THIS version, and that version's pointer is the one verified
    try:
        target = os.path.basename(os.readlink(current))
        if target != tag:
            raise RuntimeError(f"{current} resolves to {target!r}, not this pull's {tag!r}")
        back = json.loads((current / CURRENT_POINTER).read_text())
        if back != pointer or (expected_pointer is not None and back != expected_pointer):
            raise RuntimeError(f"{current}/{CURRENT_POINTER} does not read back as the pointer verified")
    except Exception as e:  # noqa: BLE001
        return {**out, "error": f"promotion could not be verified: {e}", "version": str(vdir), "pointer": pointer,
                "expected_pointer": expected_pointer, "pulled": pulled,
                "note": "promotion not claimed; no garbage collection"}
    # GC, only now: the current target is excluded FIRST; then exactly max(0, total - keep)
    # of the OLDEST remaining go, oldest by pull sequence (never by the id), so
    # exactly `keep` versions remain and the current one is always among them
    all_versions = [d for d in versions.iterdir() if d.is_dir()]
    candidates = sorted((d for d in all_versions if d.name != target), key=lambda d: (pull_seq(d), d.name))
    n_remove = max(0, len(all_versions) - keep) if keep else 0
    removed = []
    for d in candidates[:n_remove]:
        shutil.rmtree(d, ignore_errors=True)
        removed.append(d.name)
    out.update(version=str(vdir), pulled=pulled, promoted=True, verified=True, current_target=target,
               pointer=pointer, pull_seq=seq, removed_versions=removed)
    return out


def _print_rec(rec: dict) -> None:
    print(f"[petri-smoke] VERDICT {rec.get('verdict')}  status={rec.get('status')} smoke_rc={rec.get('smoke_rc')} "
          f"elapsed={rec.get('elapsed_s')}s image={rec.get('image')} results={rec.get('results_dir')}", flush=True)
    print(f"           reason: {rec.get('reason')}", flush=True)
    print(f"           attempt: {rec.get('attempt')} -> {rec.get('attempt_dir')}; current.json: "
          f"{'promoted to this attempt' if rec.get('current') else 'NOT promoted (attempt did not complete)'}; "
          f"publication: {'complete' if rec.get('publication_ok') is True else 'NOT complete'}",
          flush=True)
    if rec.get("error"):
        print(f"           error: {rec['error']}", flush=True)
    if rec.get("publish_error"):
        print(f"           publish error: {rec['publish_error']}", flush=True)
    if rec.get("smoke_verdict"):
        print(f"           the smoke's own verdict was {rec['smoke_verdict']} — not announced: "
              "its publication did not complete", flush=True)
    if rec.get("pointer_may_be_installed"):
        print("           the pointer was replaced before the failure and MAY name this attempt durably; "
              "the attempt's files were left untouched (they agree with that pointer)", flush=True)
    if rec.get("periodic_sync_quiesced") is False:
        print("           the periodic sync thread did not end within the bound: nothing was written and no "
              "pointer was installed (fail closed)", flush=True)
    pos = rec.get("positives") or {}
    if pos:
        print("           positives: " + json.dumps(pos, default=str), flush=True)
    for pr in rec.get("problems") or []:
        print(f"           problem: {pr}", flush=True)
    if rec.get("smoke_tail") and rec.get("verdict") != "PASS":
        print("           smoke.log tail:\n" + rec["smoke_tail"][-2000:], flush=True)


def expected_run_name(sha: str) -> str:
    return f"petri-smoke-{sha[:7]}"


def mirror_after_run(rec: dict, sha: str, mirror_root: Path, *, no_mirror: bool):
    """The mirror pull that follows a run (``--go`` and ``--harvest``), BOUND to
    the run: it pulls only for a record that published (status DONE and
    ``current`` the pointer this run installed and read back) and passes that
    pointer as ``expected_pointer``, so the local ``current`` is promoted only
    if the volume's ``current.json`` still IS this run's pointer — otherwise
    the pull is refused as superseded and the mirror left untouched. A record
    with no pointer of its own (PUBLISH_FAILED, SETUP_FAILED, TIMEOUT,
    ATTEMPT_DIR_EXISTS) refreshes NOTHING: there is no publication of this
    run's to mirror, and pulling whatever ``current.json`` says now would put
    another attempt's evidence under this run's outcome. Returns the mirror
    result, or None when nothing was pulled."""
    if no_mirror:
        return None
    pointer = rec.get("current") if isinstance(rec.get("current"), dict) else None
    if rec.get("status") != "DONE" or pointer is None:
        print(f"[petri-smoke] no publication from this run ({rec.get('status')}) — the mirror is NOT "
              "refreshed; a completed, published run refreshes it", flush=True)
        return None
    out = mirror_results(sha, mirror_root, expected_pointer=pointer)
    print(json.dumps(out, indent=1, default=str))
    return out


#: the local exit for a published PASS the mirror does NOT hold (MIRROR_SUPERSEDED /
#: MIRROR_FAILED): distinct from 1, every failure of the smoke or its publication
EXIT_MIRROR = 2


def local_exit(rec: dict, mirror, *, no_mirror: bool) -> tuple:
    """``(exit code, outcome name)`` for the command. ``modal_exit_code`` rules
    on the record (1 for anything but a published PASS); a published PASS is 0
    only when the local mirror now holds THIS run's publication (``mirror``
    promoted) — or ``--no-mirror`` asked for none. Otherwise ``EXIT_MIRROR``:
    ``MIRROR_SUPERSEDED`` when a concurrent publication replaced
    ``current.json`` before the pull (the publication succeeded; the mirror
    was left untouched), ``MIRROR_FAILED`` for any other refused or failed
    pull. The smoke passed and is on the volume either way; the mirror is
    what the reader looks at, so exiting 0 without it would be the trap."""
    code = modal_exit_code(rec)
    if code:
        return code, str(rec.get("verdict") or rec.get("status"))
    if no_mirror:
        return 0, "PASS (not mirrored: --no-mirror)"
    if mirror and mirror.get("promoted"):
        return 0, "PASS"
    if mirror and mirror.get("superseded"):
        return EXIT_MIRROR, "MIRROR_SUPERSEDED"
    return EXIT_MIRROR, "MIRROR_FAILED"


def run_preflight(variant: str, sha: str, static_r_object: str, alphasteer_object: str) -> dict:
    return PREFLIGHT[variant].remote(sha, static_r_object, alphasteer_object)


def spawn_smoke(variant: str, sha: str, run_name: str, static_r_object: str, alphasteer_object: str):
    return SMOKE_FN[variant].spawn(sha, run_name, static_r_object, alphasteer_object)


@app.local_entrypoint()
def main(sha: str = "", stage: bool = False, pf: bool = False, go: bool = False, harvest: str = "",
         restage: bool = False, image: str = DEFAULT_IMAGE,
         static_r_object: str = "",
         alphasteer_object: str = "",
         no_mirror: bool = False, mirror_root: str = ""):
    mroot = Path(mirror_root).expanduser() if mirror_root else MIRROR_ROOT
    if image not in IMAGES:
        raise SystemExit(f"[petri-smoke] --image must be one of {sorted(IMAGES)}")
    if harvest:
        st = json.loads(Path(harvest).expanduser().read_text())
        print(f"[petri-smoke] harvesting call {st['call']} ...", flush=True)
        rec = modal.FunctionCall.from_id(st["call"]).get()
        _print_rec(rec)
        mirror = mirror_after_run(rec, st["sha"], mroot, no_mirror=no_mirror)
        code, outcome = local_exit(rec, mirror, no_mirror=no_mirror)
        print(f"[petri-smoke] {outcome}: local exit={code} (0 only for a PUBLISHED PASS the mirror holds)",
              flush=True)
        if code:
            raise SystemExit(code)
        return
    if not sha:
        sha = _git(str(Path(__file__).resolve().parent), "rev-parse", "HEAD")
        print(f"[petri-smoke] --sha not given: using the local HEAD {sha} (the STAGED tree at that sha "
              "is what runs; local edits never travel)", flush=True)
    if len(sha) != 40:
        raise SystemExit("[petri-smoke] pass a FULL 40-hex --sha")
    want_name = expected_run_name(sha)
    run_name = app.description or DEFAULT_APP_NAME
    if run_name != want_name:
        raise SystemExit(f"[petri-smoke] run with `modal run -n {want_name} ...`; got app name "
                         f"{run_name!r}. A descriptive explicit app name is mandatory so Modal "
                         "cannot route this run under a stale identity")
    if pf or go:
        # the two objects: the flags, else the site's keys (object_on_volume) —
        # refused by name here, before anything is staged or called
        static_r_object = object_on_volume("static_r_matched", static_r_object)
        alphasteer_object = object_on_volume("alphasteer_official", alphasteer_object)
        objs = f"objects={{static_r: {static_r_object}, alphasteer: {alphasteer_object}}}"
    else:
        objs = "objects=(not needed by --stage alone)"
    print(f"[petri-smoke] app={run_name} app_id={app.app_id} sha={sha[:12]} image={image} {objs}", flush=True)

    if stage:
        with tempfile.TemporaryDirectory(prefix="petri-smoke-tars-") as td:
            tar = shallow_clone_tarball(sha, Path(td))
            print(f"[petri-smoke] pushing {tar.name} ({tar.stat().st_size // 1_000_000} MB) -> {REPOS}/",
                  flush=True)
            with VOL.batch_upload(force=True) as b:
                b.put_file(str(tar), f"{volume_client_path(REPOS)}/{tar.name}")
            r = unpack_tree.remote(sha, force=restage)
            print(json.dumps(r, indent=1))
            if r.get("error"):
                raise SystemExit(f"[petri-smoke] staging {sha} failed: {r['error']}")
        if not (pf or go):
            return
    if not (pf or go):
        raise SystemExit("pass --stage / --pf / --go / --harvest <state.json>")

    # PREFLIGHT IS MANDATORY BEFORE ANY GPU (--go always runs it): a missing
    # object is a refusal here, never a substituted object or a 50-second
    # "completed" container.
    rep = run_preflight(image, sha, static_r_object, alphasteer_object)
    print(json.dumps(rep, indent=1, default=str))
    if not rep.get("ok"):
        bad = [k for k, v in (rep.get("checks") or {}).items() if not v.get("ok")]
        raise SystemExit(f"[petri-smoke] PREFLIGHT FAILED ({bad}) — fix before any GPU; nothing was run")
    if not go:
        return

    call = spawn_smoke(image, sha, run_name, static_r_object, alphasteer_object)
    print(f"[petri-smoke] spawned smoke_{image}: call {call.object_id}", flush=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state_path = STATE_DIR / f"{run_name}-{time.strftime('%Y%m%dT%H%M%S')}.json"
    state_path.write_text(json.dumps({"app": run_name, "sha": sha, "image": image, "call": call.object_id,
                                      "objects": {"static_r_matched": static_r_object,
                                                  "alphasteer_official": alphasteer_object},
                                      "results_dir": f"{RESULTS_ROOT}/{sha}"}, indent=1))
    print(f"[petri-smoke] state -> {state_path} (re-attach with --harvest {state_path})", flush=True)
    rec = call.get()
    _print_rec(rec)
    mirror = mirror_after_run(rec, sha, mroot, no_mirror=no_mirror)
    code, outcome = local_exit(rec, mirror, no_mirror=no_mirror)
    print(f"[petri-smoke] {outcome}: local exit={code} (0 only for a PUBLISHED PASS the mirror holds)", flush=True)
    if code:
        raise SystemExit(code)
