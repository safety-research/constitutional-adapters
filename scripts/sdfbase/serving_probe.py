#!/usr/bin/env python
"""The serving investigation: what vLLM 0.25.1's
native multi-LoRA path serves for a model, per module set and rank, on one
H200 — measured by standing up the COMPLETE per-server adapter manifests the
production waves will run, relaunching the engine per (manifest, rank cap,
topology, window), and writing the servable (rank, set) pairs the LoRA
search consumes (``allowed_pairs.json``) plus the memo's measured inputs.

Subcommands (every pure step is CPU-testable; every GPU/engine step is a
thin function in the ``Deps`` seam):

``plan``
    ``probe_plan.json``: the manifests (``standup_manifest.build_manifests``
    per (set, rank)) and ONE engine launch per (manifest, ``max_lora_rank``
    cap, topology, window L ∈ {4,096, 16,384, 49,152, 65,536, 98,304}) at the
    NATURAL cap (the smallest legal value ≥ the largest adapter r), plus the
    rank-cap LADDER 8 → 16 → 32 → 64 → 128 as SEPARATE launches at the
    ladder window (buffers are sized at engine init from the cap,
    ``fused_moe.py:157-217``; ``peft_helper.validate_legal`` refuses r > cap,
    so no cross-rank hot-swap exists), each with the admitted
    ``max_num_seqs`` from the packed-stride ceiling and a KV budget estimate.
    Two manifests whose engines would be byte-identical (every S2) launch
    once. Beside each pair's split servers the plan carries the pair's UNION
    manifest (``standup_manifest.build_union_manifest`` — the class a steered
    standup SERVES: cells + zero [+ carriers] in one engine) at every window,
    so the fold certifies that class where the servers' smaller classes never
    dominate it. ``--dry-run`` prints the launches and touches nothing.
``adapters``
    Every synthetic adapter (control r, constitution r, difference 2r as a
    rank-2r random adapter of the same shape), the CLASS-AWARE zero adapter
    per manifest (rank = the cap, so the fitted cap equals the planned one)
    and the rank-1 ``o_proj`` carriers, under ``<out>/adapters/<manifest>/
    <slot>/`` with each ``adapter_config.json``; ``lora_export``'s exporters
    are bound BY NAME at call time (``synthetic_module_set_blob`` /
    ``export_module_set`` / ``export_zero_adapter_for_class`` /
    ``adapter_target_class`` — contract §3) and an absent one REFUSES with the
    name. ``peft_reference_layout`` runs only when ``import peft`` succeeds
    and records ``peft_unavailable`` otherwise.
``run``
    Per launch (resumable: ``<out>/launches/<id>.json`` per finished launch +
    the launch-list checkpoint ``probe_progress.json``; ``--resume`` continues
    the outstanding launches of an existing record and the exit status is
    ``EXIT_INCOMPLETE`` = 4 while any planned launch is outstanding — the
    sbatch requeues itself on it): compose through
    ``vllm_server.build_server_launch(lora_native=True, …)`` + ``launch_server``
    (the lora-native readiness attestation runs), then record: (1) the engine
    log carries NO "could not be wrapped" — and, on a routed-experts model,
    the ``MoE model detected. Using fused MoE LoRA implementation.`` line plus
    Triton-backend evidence; (2) ``/v1/models``
    lists every adapter; (3) the fixed prompt at temperature 0, 8 tokens,
    top-5 logprobs: random adapter vs base → max |Δ logprob| on the first
    token > 1e-2, zero adapter bit-identical, plus 48-token greedy decodes over
    11 prompts; (5) the stacked LoRA buffer bytes from the worker report vs
    ``standup_manifest.expected_lora_bytes`` within 5 %, with non-KV usage, KV
    capacity and total VRAM recorded separately (nvidia-smi = residency only);
    (6) on a routed-experts model the mixed-rank variant is the MLP sets' own
    adapters (attention/shared R, experts r_e); (7) the ``down`` set's zero
    gate_up companion changes no logit (its class zero adapter is bit-identical
    to base); (8) throughput at
    concurrency 32 / 64 / 128 against the admitted ``max_num_seqs``. The
    HF+PEFT reference check (4) runs AFTER the engine launches, once per run.
    peft (``peft==0.20.0``, pinned in configs/capeval/vllm-requirements.txt)
    is REQUIRED: an unimportable peft is a loud failure of the reference
    phase, and a pair without HF+PEFT parity evidence is never certified.
    ``--no-reference`` is a
    debugging knob only — the verdict then refuses certification.
    ``--parallel N`` (default ``auto`` = the job's device count ÷
    the plan's tp under Slurm, 1 anywhere else) runs N launches CONCURRENTLY,
    each standup pinned — in the child engine's environment alone — to a
    STRICT SUBSET of Slurm's ``CUDA_VISIBLE_DEVICES`` (a slice of ``tp``
    tokens; never widened, never a hard-coded index), on
    its own API port and under its own ``launches/<id>/`` dir. The launch
    order (priority), every result record, the verdict, the geometry fold and
    the EXIT_INCOMPLETE / ``--resume`` contract are unchanged: results and the
    checkpoint are written under ONE lock as each launch finishes, so a
    preemption loses at most the launches in flight and ``--resume`` re-runs
    exactly those. Log lines carry the device subset.
``verdict``
    ``probe_verdict.json`` + ``allowed_pairs.json`` (schema in
    :func:`write_verdict`); REFUSES (exit 4, a ``status: partial`` verdict, no
    ``allowed_pairs.json``) while a launch that could still move a pair's
    verdict is outstanding, and REFUSES (exit ``EXIT_UNCERTIFIED`` = 6, a
    ``status: uncertified`` verdict, no ``allowed_pairs.json``) when a pair that
    passed every vLLM-side check has no HF+PEFT reference evidence. A pair is
    allowed iff its single-rank AND its rank-2r difference adapter passed
    readiness + the per-adapter checks AND the HF+PEFT reference check on a
    manifest that fits the H200 with usable KV at every production window;
    exclusions carry ``oom_at_R=<cap>`` / ``refused:<msg>`` /
    ``ceiling_S<1_at_L=<L>`` / ``reference_missing:…`` / ``reference_undecidable:…`` / ….
    A COMPLETE verdict — and a reference-missing ``uncertified`` one — also
    FOLDS the measured serving geometry (:func:`geometry_fold`) PER MODULE SET
    over every pair whose SERVING was certified (its engines came up and served
    adapters of the pair's class, S >= 1 at every window it ran), WHATEVER the
    HF+PEFT parity said about the adapters (geometry
    certification and adapter parity are SEPARATE facts; the verdict, the menu
    and the geometry record / overlay all carry ``adapter_parity`` {status
    verified|mismatch|unverified, basis, per_pair}; the MENU still requires
    ``verified``; an UNSTEERED standup may size from a mismatch / unverified
    record, a STEERED one refuses it by name — geometry.require_verified_
    adapter_parity): the widest EXECUTING LoRA
    module dim the worker reported over that set's certifying launches and the
    largest ``S x L`` one of its engines came up at, with provenance (probe job
    id, record dir, the launches' attestation sha256s, tree sha), into
    ``geometry_record.json`` under the record AND the SITE geometry overlay
    (``geometry.site_geometry_dir()``: ``$SISTEER_GEOMETRY_DIR``, else the
    front door's site / shared-root rule), keyed (model_key, module_set) —
    ``geometry.budget_for(model_key, module_set)`` / ``rows_for`` /
    ``kernel_ceiling_adjustment(…, module_set=)`` resolve the SELECTED set's
    row BEFORE their built-in tables, so the padded-static preflight and the
    padded launchers size the model from the probe's measurement of the HPO
    winner's class, never a hand-typed constant or a model-wide maximum.
``memo``
    ``memo_scaling_inputs.md``: the FD 24 memo's measured facts.

Default root: ``$SHARED/exp/multilora_cert/<model_key>_probe_<jobid>/``.
Never the lens path (FD 19): nothing here imports ``lens_worker``.
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import functools
import hashlib
import json
import fcntl
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor
from concurrent.futures import wait as _wait_futures
from dataclasses import dataclass
from pathlib import Path
from typing import (Any, Callable, Dict, Iterable, List, Mapping, Optional,
                    Sequence, Tuple)

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
for _p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "sdfbase"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from sisteer.capeval.serving import serving_attestation as sa  # noqa: E402  (stdlib leaf)
from sisteer.capeval.serving import standup_manifest as sm  # noqa: E402
from sisteer.extraction import lora_modules as lm  # noqa: E402

TAG = "[serving-probe]"
PLAN_FILE = "probe_plan.json"
VERDICT_FILE = "probe_verdict.json"
ALLOWED_PAIRS_FILE = "allowed_pairs.json"
MEMO_FILE = "memo_scaling_inputs.md"
LAUNCHES_DIR = "launches"
ADAPTERS_DIR = "adapters"
RESULT_FILE = "RESULT"
#: the run's launch-list checkpoint: which planned launches are done / outstanding and why the run stopped, rewritten
#: after every launch so a successor job (`run --resume`) continues exactly the outstanding ones (a time-budget stop
#: used to return 0 and the verdict certified over the unrun launches)
PROGRESS_FILE = "probe_progress.json"
#: the probe-folded serving geometry (the site overlay's record body, also kept beside the verdict)
GEOMETRY_RECORD_FILE = "geometry_record.json"
#: `run` / `verdict` exit status while planned launches are outstanding (the sbatch requeues itself on it)
EXIT_INCOMPLETE = 4
#: `verdict`'s exit when the record is COMPLETE but certifies less than the probe set out to and a resubmission cannot
#: change it: HF+PEFT reference evidence missing for a pair that passed every vLLM-side check, or the geometry fold
#: could not be computed from the certified launches. Distinct from 4 so the sbatch does NOT requeue on it.
EXIT_UNCERTIFIED = 6
#: the launch kinds whose outcome can move a pair's verdict; a `ladder` launch is a memo measurement (the buffer law
#: above the natural cap) and never certifies or excludes a pair
DECISION_KINDS = ("natural", "fallback")
#: the engine log's name under a launch dir (``native_standup.SERVER_LOG_FILE``; spelled
#: here so the CPU-side readers need no native_standup import)
SERVER_LOG_FILE = "vllm_server.log"
#: Slurm's device list for this job (READ, never set — cluster hard rule 3). ``run --parallel N`` pins each concurrent
#: standup to a STRICT SUBSET of it (a slice of ``tp`` tokens) in the child engine's environment only, so N engines never
#: share a GPU and none sees a device the allocation does not hold (the launches are independent one-GPU
#: standups of one bf16 model; 8 in flight cut the probe from ≈11 h to ≈1.5 h)
VISIBLE_DEVICES_ENV = "CUDA_VISIBLE_DEVICES"
#: ``run --parallel`` default: under Slurm the allocation's device count ÷ the plan's tp; 1 anywhere else
PARALLEL_AUTO = "auto"
#: THE BATCHED-TOKEN BUDGET. The harness's no-chunking invariant makes every engine PREFILL its whole
#: ``max_num_batched_tokens = S x L`` in one batch, and that activation is what the estimator never priced: the D = 2,048
#: sets kept S = 21 / 15 / 8 / 32 at L = 49,152 / 65,536 / 98,304 / 16,384 (0.5-1.03 M tokens) and the engine died in
#: ``_initialize_kv_caches`` while the largest batch that ever came up was 262,144. So the admitted S is
#: ``min(S_stride, S_kv, floor(batched_token_budget / L))`` with a budget that is MEASURED, never typed: the largest
#: ``S x L`` a lora-native engine of this record came up at (the floor; ``BUDGET_METHOD_RECORD_FLOOR``), the smallest
#: start-time OOM above it recorded as the ceiling. Before any launch passed the budget is ``unmeasured`` and no cap
#: applies (the first 4,096-window launches measure it; ``run --resume --replan`` then re-derives S).
BUDGET_METHOD_RECORD_FLOOR = "record_floor"
BUDGET_METHOD_UNMEASURED = "unmeasured"
#: how an ``oom`` launch is READ by the verdict (:func:`oom_kind`): two kinds exclude a pair, one re-plans it
OOM_SERVING = "serving_oom"                                 # the engine was up and OOM'd under load: genuine
OOM_START_MIN_S = "start_time_minimum_S"                     # died at start with S = 1: no smaller batch exists, genuine
OOM_START_WITHIN_BUDGET = "start_time_within_budget"         # died at start at S x L <= the measured floor: buffers / KV, genuine
OOM_START_OVER_BUDGET = "start_time_prefill_over_budget"     # died at start at S x L > the floor (or no floor yet): RE-PLAN, never excludes
REPLAN_OOM_KINDS = (OOM_START_OVER_BUDGET,)
#: where ``run --resume --replan`` MOVES a superseded launch's result file and dir (never deleted): ``launches/superseded/``
SUPERSEDED_DIR = "superseded"
#: THE REOPEN RECEIPT: the result generation is a deterministic digest of public files, so it can be
#: typed. A reopen of a COMPLETE record is authorised only by a receipt the front door writes INTO the record on the probe
#: cluster after EVERY configured cluster acknowledged its REVOKED marker (``reproduce/lib/slurm.sh slurm_revoke_geometry``):
#: ``<record>/reopen_receipts/<generation>.<nonce>.json`` ({generation, nonce = secrets.token_hex(16), acknowledged clusters
#: with marker paths + utc, minted host, reason}); the token handed to the carrier is that nonce. ``run`` / ``reverdict`` verify
#: a receipt for the CURRENT generation carrying that nonce and CONSUME it (renamed ``.used.<utc>.json``), refusing otherwise
#: without printing any value an operator could type.
RECEIPTS_DIR = "reopen_receipts"
REOPEN_NONCE_RE = re.compile(r"^[0-9a-f]{32}$")
#: ONE revocation epoch per record (item 2). A receipt's file name carries its state: ``<gen>.<nonce>.json`` ACTIVE;
#: ``.used.<utc>`` consumed; ``.superseded.<utc>`` invalidated by a LATER mint or by the consumer of a later receipt; ``.consuming.<tag>``
#: the consumer's rename-as-lock (a moment). The current epoch is the receipt minted LAST (``minted_utc``, a field every mint since
#: re-check 3 writes — b497680d2's included — else the file's mtime): only that receipt can ever be consumed, exactly once; an older
#: receipt left active by a mint that did not supersede it (a pre-epoch mint) is refused as superseded and moved aside. No receipt
#: field is REQUIRED beyond re-check 3's, so a receipt minted by b497680d2 reads as valid-or-consumed, never as unknown.
RECEIPT_STATE_ACTIVE, RECEIPT_STATE_USED, RECEIPT_STATE_SUPERSEDED, RECEIPT_STATE_CONSUMING = "active", "used", "superseded", "consuming"
#: a consumed receipt is IN PROGRESS — `.inprogress.<tag>.json`, its doc carrying the consumer (`consume`: started_utc,
#: what, action, pid, host, job_id) — until the reopen's CERTIFYING REWRITE completes (finish_reopen_receipt -> `.used`); a process
#: that dies between the archive and the rewrite leaves it so, hpo_probe_status names it, and `reverdict --resume-interrupted`
#: adopts it (never a second consume, never a fresh mint that would revoke again)
RECEIPT_STATE_INPROGRESS = "inprogress"
RECEIPT_FILE_RE = re.compile(r"^(?P<gen>[0-9a-f]{64})\.(?P<nonce>[0-9a-f]{32})(?:\.(?P<state>used|superseded|consuming|inprogress)\.(?P<tag>[^.]+))?\.json$")
RECEIPT_CONSUME_KEY = "consume"
#: THE REVOCATION EPOCH (B11, 2026-10-09): ``sha256(nonce)`` of a revocation's receipt. The front door writes it into every REVOKED
#: marker of that revocation (``epoch``; lib/slurm.sh slurm_revoke_geometry mints the nonce BEFORE the markers) and the fold that runs
#: under the consumed receipt (a reverdict, a receipted resume) stamps it into the geometry record's ``provenance.reopen_epoch``.
#: slurm_stage_geometry clears a same-generation marker ONLY for a copy carrying the marker's epoch — the copy was certified THROUGH
#: that revocation's receipt — where it used to order the two by ``written_utc`` < ``revoked_utc``, two second-resolution clock stamps
#: compared strictly, so a fold and a later revocation in the SAME second read as fold-after and the stale copy cleared the marker.
#: The nonce itself (the reopen token) is written nowhere but the receipt; the digest names the epoch without exposing it.
REOPEN_EPOCH_KEY = "reopen_epoch"
#: THE DERIVED CERTIFICATION ARTIFACTS of a record: every one is ARCHIVED — moved under
#: ``launches/superseded/<utc>/`` with an ``ARCHIVED.json`` note, never deleted — the moment a result is REOPENED (a
#: resumed attempt with launches to run, a re-plan, a re-verdict) and whenever a verdict comes out non-certifying, so no
#: menu / geometry / memo written over an older result set ever stands beside the record. Each of them is stamped with the
#: record's ``result_generation`` (geometry.record_result_generation) when written, and hpo_probe_status verifies the stamp
#: against the record before the front door may fetch anything.
DERIVED_ARTIFACTS = (ALLOWED_PAIRS_FILE, VERDICT_FILE, GEOMETRY_RECORD_FILE, MEMO_FILE, RESULT_FILE)
ARCHIVE_NOTE_FILE = "ARCHIVED.json"
RESULT_GENERATION_KEY = "result_generation"
#: the engine-log lines that say the engine died at start (recorded as detail on a start-time OOM when the log is on disk)
START_OOM_SIGNATURES = ("EngineCore failed to start", "_initialize_kv_caches", "Available KV cache memory: -",
                        "No available memory for the cache blocks", "Tried to allocate")

#: the model keys the probe knows dims for (``lora_modules``' verified tables);
#: the served id + revision come from the caller (models.yaml / the sbatch)
#: the verified dims by model key: lora_modules.ARCH_DIMS_BY_MODEL_KEY — ONE table, shared with the padded preflight
MODEL_DIMS_KEYS: Tuple[str, ...] = tuple(sorted(lm.ARCH_DIMS_BY_MODEL_KEY))
DEFAULT_SETS: Tuple[str, ...] = tuple(sm.MODULE_SETS)
DEFAULT_RANKS: Tuple[int, ...] = (4, 16, 64)
#: the NOMINAL production concurrency per window before admission (statics
#: at 4,096 are graph/compute-bound; the long windows are KV/stride-bound)
REQUESTED_S_BY_WINDOW: Dict[int, int] = {4096: 64, 16384: 32, 49152: 32, 65536: 16, 98304: 8}
THROUGHPUT_CONCURRENCIES: Tuple[int, ...] = (32, 64, 128)
THROUGHPUT_WINDOWS: Tuple[int, ...] = (4096, 49152)
#: the synthetic adapters' B scale (passed to synthetic_module_set_blob's ``scale``):
#: LeCun 1e-3 draws move a first-token logprob by far less than 1e-2, so the
#: probe adapters are scaled up to carry an unmistakable, recorded effect
DEFAULT_SYNTHETIC_SCALE = 100.0
FIRST_TOKEN_DELTA_MIN = 1e-2
REFERENCE_LOGPROB_TOL = 2e-2
BUFFER_BYTES_TOL = 0.05
PROBE_MAX_TOKENS = 8
DECODE_MAX_TOKENS = 48

#: the fixed probe prompt (assertion 3) and the 11 greedy-decode prompts — all
#: synthetic, neutral text (the certification's admission-gate spirit)
PROBE_PROMPT: Tuple[Dict[str, str], ...] = (
    {"role": "user", "content": "Explain in two sentences why the sky appears blue during the day."},)
DECODE_PROMPTS: Tuple[str, ...] = (
    "Write one sentence about the ocean.",
    "List three common kitchen spices.",
    "What is 17 multiplied by 6? Answer with the number only.",
    "Give a short definition of photosynthesis.",
    "Name a primary color and one object that has it.",
    "Describe a bicycle to someone who has never seen one, in two sentences.",
    "Translate 'good morning' into French.",
    "What comes next in the sequence 2, 4, 8, 16?",
    "Suggest a title for a story about a lighthouse keeper.",
    "Summarise the water cycle in one sentence.",
    "Write a haiku about autumn leaves.",
)

#: vLLM 0.25.1's INFO line when a LoRA-enabled MoE model takes the fused-MoE LoRA path (``lora/model_manager.py``)
LOG_MOE_LORA_LINE = "MoE model detected. Using fused MoE LoRA implementation."
LOG_UNWRAPPED_RE = re.compile(r"could not be wrapped", re.I)
LOG_TRITON_RE = re.compile(r"^.*triton.*(moe|expert|fused).*$|^.*(moe|expert|fused).*triton.*$", re.I | re.M)
OOM_RE = re.compile(r"CUDA out of memory|OutOfMemoryError|No available memory for the cache blocks|"
                    r"not enough memory|Free memory on device .* is less than desired GPU memory utilization|"
                    r"Insufficient memory", re.I)
RANK_REFUSAL_RE = re.compile(r"LoRA rank \d+ is greater than max_lora_rank(?: \d+)?", re.I)
#: the worker's refusal of an adapter whose routed-expert factors exceed the composed buffer rank
#: (``native_moe_lora.REFUSAL_PHRASE``) — a REFUSED launch naming the ranks, never an OOM or a shape error
MOE_RANK_REFUSAL_RE = re.compile(r"routed-expert LoRA rank \d+ exceeds the composed SISTEER_MOE_LORA_RANK=\d+", re.I)
ILLEGAL_ACCESS_RE = re.compile(r"illegal memory access", re.I)
MAMBA_RE = re.compile(r"Mamba cache", re.I)

STATUS_PASSED = "passed"
STATUS_FAILED = "failed"          # engine up, a check failed
STATUS_OOM = "oom"
STATUS_REFUSED = "refused"        # compose / readiness / adapter-load refusal
STATUS_ERROR = "error"
STATUS_NOT_RUN = "not_run"
STATUS_SKIPPED = "skipped"        # a conditional launch whose parent passed
TERMINAL_STATUSES = (STATUS_PASSED, STATUS_FAILED, STATUS_OOM, STATUS_REFUSED, STATUS_ERROR, STATUS_SKIPPED)

BASIS_HF_PEFT = "hf_peft"
#: the reference HARNESS's own failure, per adapter and per launch: the adapter never attached on the HF side (PEFT bound
#: none / not all of the exported keys — the loaded class does not carry the export's decoder prefix), PEFT's own layout of
#: the same architecture disagrees with the export, the HF distribution did not move off the base's, or it is identical
#: across distinct adapters. Read by the verdict as NO evidence (`reference_missing`, UNCERTIFIED, no menu) — NEVER as a
#: `reference_mismatch` (every "mismatch" of one probe was the base model answering, the adapters dropped with a
#: PEFT warning because the reference loaded the text-only causal-LM class (tree `model.layers`) while the export is
#: spelled in the checkpoint's own class tree (`model.language_model.layers`)).
REF_STATUS_PASS, REF_STATUS_FAIL, REF_STATUS_ERROR, REF_STATUS_MISSING = "pass", "fail", "error", "reference_missing"
REFERENCE_CHECK_FILE = "reference_check.json"
#: the label of a verdict whose ONLY evidence is the vLLM-side checks: a diagnostic, never a certifying basis — a pair
#: without HF+PEFT parity evidence is `reference_missing`, not allowed
BASIS_VLLM_SIDE = "vllm_side_checks_1_3_5_7"
#: verdict statuses
VERDICT_COMPLETE, VERDICT_PARTIAL, VERDICT_UNCERTIFIED = "complete", "partial", "uncertified"
#: geometry certification and adapter parity are SEPARATE facts. The verdict, the menu and the geometry
#: record / overlay carry ``adapter_parity`` {status, basis, per_pair}; the statuses ARE geometry.PARITY_* (a test pins them).
ADAPTER_PARITY_KEY = "adapter_parity"
PARITY_VERIFIED, PARITY_MISMATCH, PARITY_UNVERIFIED = "verified", "mismatch", "unverified"
#: the exclusion reason prefixes that are PARITY facts, not serving ones (write_verdict splits the two): a judged mismatch, and
#: the undecidable adapters (their own one-ulp perturbation flips their top-1)
PARITY_REASON_PREFIX = "reference_mismatch:"
#: rule (a) and the PARITY PASS. An adapter the reference JUDGED but cannot decide is `reference_undecidable`: a
#: DECIDED exclusion of its pair (evidence: the measured chaos / weakness), never `reference_missing` (no evidence, UNCERTIFIED).
REF_STATUS_UNDECIDABLE = "reference_undecidable"
REASON_UNDECIDABLE_PREFIX = "reference_undecidable:"
#: both are PARITY reasons: they never touch the pair's SERVING certification / the geometry fold
PARITY_REASON_PREFIXES = (PARITY_REASON_PREFIX, REASON_UNDECIDABLE_PREFIX)
#: the per-set BENIGN synthetic scale of the parity pass: the adapter must move the engine's first-token top-5 by a clear
#: margin over the HF-vs-engine base floor without flattening the distribution into bf16 chaos. dW = B A^T has entries of
#: order scale/sqrt(in) on EVERY targeted module, so the effect grows with the adapted surface (one o_proj per attention
#: block for `wo` vs every MLP projection of every block for `mlp`) and rank-independent (B ~ N(0, scale^2/r)).
PARITY_SCALES_BY_SET: Dict[str, float] = {"wo": 1.0, "qkvo": 0.3, "attn_mlp": 0.05, "mlp": 0.05, "down": 0.05}
PARITY_SCALE_PROVENANCE = (
    "calibrated on the serving probe's first subject, cluster A 2026-09-30: job 807473 @ scale 1 — wo r4 |d| 0.009-0.017 "
    "with the greedy token changing (moved 2.1-3.5 nats), mlp r4 chaotic (top-1 disagreement, one-ulp flips 3/3); job 807594 @ scale "
    "0.05 — mlp r4 |d| <= 0.007 moving the top-5 by 0.9-1.8 nats (49-101x that record's 0.0179 base floor), wo r4 too weak (moved "
    "0.12-0.39, greedy token unchanged). qkvo INTERPOLATED (4 attention modules per block), confirmed by the pass's own moved band; "
    "attn_mlp and down take the MLP-bearing scale (the MLP projections dominate their surface). A model whose surface differs "
    "overrides the table with --parity-scales.")
PARITY_SCALES_DEFAULT_WORD = "default"
PARITY_MOVED_MIN = 0.3            # nats over the top-5 the ENGINE must move (greedy token unchanged) for a parity slot to discriminate
PARITY_FLOOR_K = 3.0              # rule (a): |d| <= max(REFERENCE_LOGPROB_TOL, K x the measured base floor, M x the one-ulp sensitivity)
PARITY_SENSITIVITY_M = 2.0
PARITY_SENSITIVITY_DRAWS = 3
PARITY_ULP_REL = 2.0 ** -8        # one bf16 ulp of relative noise on lora_B
PARITY_RECORD_KEY = "parity_record"
#: PARITY CERTIFICATE v2: assertion 4 judged over P = 8 FIXED prompts instead of one —
#: |d_p| per prompt, the base floor = max over the same prompts of the HF-vs-engine BASE |d| measured on the same standup,
#: tol = max(REFERENCE_LOGPROB_TOL, PARITY_FLOOR_K x floor, PARITY_SENSITIVITY_M x max_p sensitivity), PASS iff max_p |d_p| <= tol
#: with the greedy token agreeing on every prompt; v1 (the one-prompt rule) and v2 are recorded SIDE BY SIDE per adapter and
#: the adapter's effective status is v2's where v2 exists, else v1's (``certificate_version`` says which). Prompt 0 IS the v1
#: prompt (PROBE_PROMPT); the seven others span ~20-200 tokens, neutral synthetic text, recorded in the plan (``parity_certificate``).
PARITY_CERTIFICATE_VERSION = 2
PARITY_V2_PROMPTS: Tuple[Tuple[Dict[str, str], ...], ...] = (
    PROBE_PROMPT,
    ({"role": "user", "content": "Give three practical tips for keeping a small herb garden alive on a windowsill."},),
    ({"role": "user", "content": "A train leaves a station at 09:40 and arrives at 13:05 the same day. How long is the journey? "
                                 "Then say whether that is longer or shorter than three and a half hours."},),
    ({"role": "user", "content": "Summarise, in one paragraph, how a bicycle's gears let a rider climb a hill with less effort, "
                                 "mentioning the chainring, the cassette and why a lower gear turns the wheel less per pedal stroke."},),
    ({"role": "user", "content": "Write a short, polite email to a neighbour asking whether they could water two potted plants for "
                                 "four days next week while you are away; offer to return the favour and mention that the plants "
                                 "sit on the back step and need only a little water each morning."},),
    ({"role": "user", "content": "Explain the difference between weather and climate to a ten-year-old. Use one example about rain, "
                                 "one about seasons, and finish with a sentence on why scientists measure both for many years."},),
    ({"role": "user", "content": "Here is a recipe outline: boil pasta for nine minutes; meanwhile warm olive oil, add sliced garlic "
                                 "until golden, then chopped tomatoes, a pinch of salt and a little of the pasta water; toss the "
                                 "drained pasta through the sauce and finish with torn basil and grated cheese. Rewrite it as five "
                                 "numbered steps, each one sentence, keeping every ingredient and the nine-minute timing."},),
    ({"role": "user", "content": "You are planning a one-day walking tour of a mid-sized old town for two visitors who like "
                                 "architecture and quiet cafes but dislike long queues. Propose a morning, a lunch stop and an "
                                 "afternoon, with rough timings, a note on comfortable shoes and the weather, one alternative in case "
                                 "of rain, and a closing suggestion for an early dinner near the station so they can catch an evening "
                                 "train home without rushing. Keep the whole plan under two hundred words and do not name real places."},),
)


#: THE SERVING VENV IDENTITY: every parity surface — the launch result, the reference record, each
#: per-adapter v1/v2 record, the pair's and the record's adapter_parity blocks, the menu, the geometry fold — binds the venv the
#: probe's standup and HF+PEFT reference ran in: {requirements_sha256 (the key), variant (gdn | nemotron | None), name}, read
#: from serving_attestation.serving_venv_facts() in the probe's own process (the carrier exports the identity with the venv it
#: selected by key), never typed. The production admission gate compares the digest against the standup's own.
SERVING_VENV_KEY = sa.SERVING_VENV_KEY


#: a launch that recorded no serving venv (every standup from before serving venvs were recorded) is attested from ITS OWN recorded
#: provenance — the engine's vllm_server.log (the vLLM version in "Initializing a V1 LLM engine (vX)", the flashinfer version in the
#: autotune-cache path "[kernel_warmup.py] Using FlashInfer autotune cache file: …/flashinfer_autotune_cache/<version>/…") and the venv
#: the standup ran in — launch_cmd.json's interpreter path / SISTEER_SERVING_VENV when the launch recorded one, else (``job_log_binding``)
#: the carrier JOB's log that stood the launch up (the job whose node is the launch's host and whose span covers its start), which names
#: the venv; the venv name's variant + hash8 bind the base requirements digest when the repo's requirements file still hashes to it.
#: A launch missing ANY fact gets NO block (its pairs stay unproven). The block says how it was derived (``provenance``).
IDENTITY_SOURCE_PROVENANCE = "launch_provenance"
ENGINE_LOG_VLLM_RE = re.compile(r"Initializing a V1 LLM engine \(v([0-9][^)\s]*)\)")
ENGINE_LOG_FLASHINFER_RE = re.compile(r"flashinfer_autotune_cache/(\d+\.\d+\.\d+(?:\.post\d+)?)/")
JOB_LOG_VENV_RE = re.compile(r"serving venv(?: already bootstrapped)?[: ]+\(?(/\S*?/venvs/(sisteer-petri-vllm[^/\s)]*))")
JOB_LOG_NODE_RE = re.compile(r"\bnode=(\S+)")
JOB_LOG_TS_RE = re.compile(r"\[(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)\]")
BASE_REQUIREMENTS_FILE = "configs/capeval/vllm-requirements.txt"


def _engine_log_facts(launch_dir: Path) -> Dict[str, Any]:
    """``{vllm_version, flashinfer_version, vllm_line, flashinfer_line}`` read from the launch's vllm_server.log (None when absent)."""
    out: Dict[str, Any] = {"vllm_version": None, "flashinfer_version": None, "vllm_line": None, "flashinfer_line": None}
    p = Path(launch_dir) / "vllm_server.log"
    if not p.is_file():
        return out
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    m = ENGINE_LOG_VLLM_RE.search(text)
    if m:
        out["vllm_version"] = m.group(1)
        ls = text.rfind("\n", 0, m.start()) + 1; out["vllm_line"] = text[ls:text.find("\n", m.end()) if text.find("\n", m.end()) != -1 else None][:240]
    m = ENGINE_LOG_FLASHINFER_RE.search(text)
    if m:
        out["flashinfer_version"] = m.group(1)
        ls = text.rfind("\n", 0, m.start()) + 1; out["flashinfer_line"] = text[ls:text.find("\n", m.end()) if text.find("\n", m.end()) != -1 else None][:240]
    return out


def _launch_recorded_venv(launch_dir: Path) -> Optional[Tuple[str, str]]:
    """``(venv name, how)`` when the launch's own launch_cmd.json names its venv (an interpreter path under .../venvs/, or the
    SISTEER_SERVING_VENV it exported); None for a bare ``vllm`` command."""
    p = Path(launch_dir) / "launch_cmd.json"
    if not p.is_file():
        return None
    try:
        lc = json.loads(p.read_text(encoding="utf-8"))
    except ValueError:
        return None
    env = lc.get("env") if isinstance(lc.get("env"), Mapping) else {}
    v = env.get(sa.SERVING_VENV_ENV) if isinstance(env, Mapping) else None
    if isinstance(v, str) and v:
        return (Path(v).name, f"launch_cmd.json env {sa.SERVING_VENV_ENV}")
    cmd = lc.get("cmd") if isinstance(lc.get("cmd"), list) else []
    for tok in cmd[:2]:
        m = re.search(r"/venvs/(sisteer-petri-vllm[^/\s]*)/bin/", str(tok))
        if m:
            return (m.group(1), "launch_cmd.json interpreter path")
    return None


def _job_log_venv_for_launch(res: Mapping[str, Any], job_logs: Sequence[Path]) -> Optional[Dict[str, Any]]:
    """The carrier job that stood ``res`` up, by the launch's host and start time against each job log's node and time span, and
    the venv that log names; None unless exactly one job log matches."""
    host, started = str(res.get("host") or ""), str(res.get("started_utc") or "")
    if not host or not started:
        return None
    hits = []
    for lg in job_logs:
        try:
            text = Path(lg).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        mn = JOB_LOG_NODE_RE.search(text)
        ts = JOB_LOG_TS_RE.findall(text)
        mv = JOB_LOG_VENV_RE.search(text)
        if not (mn and ts and mv) or mn.group(1) != host or not (ts[0] <= started <= ts[-1]):
            continue
        hits.append({"job_log": str(lg), "venv_path": mv.group(1), "venv_name": mv.group(2), "node": mn.group(1), "span": [ts[0], ts[-1]]})
    return hits[0] if len(hits) == 1 else None


VLLM_REQUIREMENT_RE = re.compile(r"^\s*vllm\s*(?:==\s*([0-9][^\s;#]*)|@\s*\S*?vllm-([0-9][^-\s]*)-)", re.M)
#: common.sh f22_serving_venv: the variant's overlay requirements, hashed AFTER the base (`cat base overlay | sha256sum`) into the venv name
VARIANT_OVERLAY_FILES = {"gdn": "configs/capeval/vllm-requirements-gdn.txt", "nemotron": "configs/capeval/vllm-requirements-nemotron.txt"}
JOB_LOG_REPO_RE = re.compile(r"\brepo=(\S+)")
JOB_LOG_SHA_RE = re.compile(r"\bsha=([0-9a-f]{7,40})\b")
JOB_LOG_OUT_RE = re.compile(r"\bout=(\S+)")
JOB_LOG_JOB_RE = re.compile(r"\bjob=(\d+)\b")


def _requirements_vllm_version(repo_root: Optional[Path] = None, *, text: Optional[str] = None) -> Optional[str]:
    """The vLLM DISTRIBUTION version the base requirements pin (`vllm==X`, or the wheel filename `vllm-X-…whl` of a `vllm @ URL` line)
    — what importlib.metadata reports in a venv built from them (e.g. 0.25.1+cu129); from ``text`` when given, else the tip's file."""
    if text is None:
        root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
        p = root / BASE_REQUIREMENTS_FILE
        if not p.is_file():
            return None
        text = p.read_text(encoding="utf-8", errors="replace")
    m = VLLM_REQUIREMENT_RE.search(text)
    if not m:
        return None
    return m.group(1) or m.group(2)


def _git_show(root: Path, sha: str, path_in_repo: str) -> Optional[bytes]:
    try:
        out = subprocess.run(["git", "-C", str(root), "show", f"{sha}:{path_in_repo}"], capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def _checkout_head(checkout: Path) -> Optional[str]:
    try:
        out = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else None


def _requirements_bytes(path_in_repo: str, *, checkout: Optional[str], sha: Optional[str], repo_root: Optional[Path]) -> Tuple[Optional[bytes], Optional[str]]:
    """The requirements file's bytes AS THE CARRIER HAD THEM (the HISTORICAL tree binds, never a checkout's
    current file): the carrier's recorded `sha=` FIRST — `git show <sha>:<path>` in this repo, else in the recorded checkout — and the
    checkout's working file only when the checkout's HEAD is verifiably that sha (the funnel's immutable per-sha detached checkouts); a
    checkout that moved on or cannot be verified contributes nothing. ``(bytes, source)`` or ``(None, None)``."""
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    if sha:
        b = _git_show(root, sha, path_in_repo)
        if b is not None:
            return b, f"git {sha[:12]}:{path_in_repo} (this repo)"
        if checkout and Path(checkout).is_dir():
            b = _git_show(Path(checkout), sha, path_in_repo)
            if b is not None:
                return b, f"git {sha[:12]}:{path_in_repo} (the carrier's checkout {checkout})"
    if checkout:
        p = Path(checkout) / path_in_repo
        head = _checkout_head(Path(checkout))
        if p.is_file() and head and (not sha or head.startswith(sha) or sha.startswith(head)):
            return p.read_bytes(), f"checkout {checkout} (HEAD {head[:12]}{' == the recorded sha' if sha else ', no sha recorded'})"
    return None, None


def venv_requirements_digests(venv_name: str, *, checkout: Optional[str] = None, sha: Optional[str] = None, repo_root: Optional[Path] = None,
                              allow_tip: bool = False) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """What the venv ``venv_name`` was built from, computed exactly as common.sh f22_serving_venv names it: the BASE requirements digest
    (sha256 of configs/capeval/vllm-requirements.txt — what engine_identity carries) and, for a variant (gdn / nemotron), the digest of
    base + overlay bytes, whose first 8 hex is the venv name's hash8. The bytes are the carrier's own (its checkout, else the sha's
    history); ``allow_tip`` (a launch that recorded its venv but no sha) falls back to this repo's file, recorded as such.
    Returns ``({variant, hash8, expected_hash8, base_requirements_sha256, venv_requirements_sha256, overlay_file, requirements_source,
    vllm_pin}, None)`` when the name's hash8 is the expected one, else ``(None, reason)``."""
    parsed = sa.parse_serving_venv_name(venv_name)
    if not isinstance(parsed, Mapping) or not parsed.get("campaign_venv") or not parsed.get("hash8"):
        return None, f"the venv name {venv_name!r} is not a campaign serving venv (sisteer-petri-vllm[-<variant>]-<hash8>)"
    variant = parsed.get("variant") or None
    h8 = str(parsed["hash8"])
    overlay_file = VARIANT_OVERLAY_FILES.get(str(variant)) if variant else None
    if variant and overlay_file is None:
        return None, f"the venv name {venv_name!r} names the variant {variant!r}, which common.sh has no overlay requirements for"
    base_bytes, src = _requirements_bytes(BASE_REQUIREMENTS_FILE, checkout=checkout, sha=sha, repo_root=repo_root)
    if base_bytes is None and allow_tip:
        root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
        p = root / BASE_REQUIREMENTS_FILE
        if p.is_file():
            base_bytes, src = p.read_bytes(), "this repo's tip (no carrier checkout / sha recorded)"
    if base_bytes is None:
        return None, (f"the base requirements the venv {venv_name!r} was built from are unreadable as the carrier had them: sha "
                      f"{sha or '(unrecorded)'} is not in this repo's history and the carrier's checkout {checkout or '(unrecorded)'} is gone, moved on, "
                      "or cannot be verified at that sha — the binding needs the historical bytes")
    blob = base_bytes
    if overlay_file:
        ov, _s2 = _requirements_bytes(overlay_file, checkout=checkout, sha=sha, repo_root=repo_root)
        if ov is None and allow_tip:
            root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
            p = root / overlay_file
            ov = p.read_bytes() if p.is_file() else None
        if ov is None:
            return None, f"the {variant} overlay requirements ({overlay_file}) the venv {venv_name!r} was built from are unreadable at the carrier's checkout / sha"
        blob = base_bytes + ov
    expected = hashlib.sha256(blob).hexdigest()
    base_digest = hashlib.sha256(base_bytes).hexdigest()
    if expected[:8] != h8:
        return None, (f"the venv name {venv_name!r} (hash8 {h8!r}) is not the digest of its requirements as the carrier had them "
                      f"({'base + ' + str(variant) + ' overlay' if variant else 'base'} = {expected[:12]}…, from {src}) — the identity cannot be bound to a known requirements set")
    return ({"variant": variant, "hash8": h8, "expected_hash8": expected[:8], "base_requirements_sha256": base_digest, "venv_requirements_sha256": expected,
             "overlay_file": overlay_file, "requirements_source": src, "vllm_pin": _requirements_vllm_version(text=base_bytes.decode("utf-8", errors="replace"))}, None)


def record_job_logs(record_dir: Path, model_key: Optional[str] = None) -> List[Path]:
    """The carrier job logs that could have stood this record's launches up: ``$SISTEER_LOG_DIR`` (else the record's shared root's
    exp/logs — <root>/exp/reproduce/<model>/<stage>/<record> -> <root>/exp/logs) filtered to ``hpo-<model_key>-*.out``."""
    env = os.environ.get("SISTEER_LOG_DIR")
    rd = Path(record_dir).resolve()
    cands = [Path(env)] if env else []
    if len(rd.parents) >= 4:
        cands.append(rd.parents[3] / "logs")
    for d in cands:
        if d.is_dir():
            pats = ([f"hpo-{model_key}-*.out", f"serving-probe-{model_key}*.out"] if model_key else ["hpo-*.out", "serving-probe-*.out"])
            return sorted({p for pat in pats for p in d.glob(pat)})
    return []


def launch_engine_provenance(record_dir: Path, res: Mapping[str, Any], *, job_logs: Sequence[Path] = (), job_log_binding: bool = True,
                             repo_root: Optional[Path] = None, standup_block_lacked_identity: bool = False
                             ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """``(venv block, refusal)`` for a launch that recorded no serving identity, derived from its RECORDED provenance —
    the launch's own vllm_server.log (vLLM and flashinfer versions), the venv from its launch_cmd.json when it names one, else
    (``job_log_binding``) from the ONE carrier job log that logged THIS record as its `out=`, whose node is the launch's host
    and whose span covers its start (its job id from the log's name / `job=`); the venv name's hash8 must be the digest of the
    requirements AS THAT CARRIER HAD THEM (its recorded checkout or sha; base + overlay for a variant), the base digest alone being
    the engine identity's. Any missing fact, an ambiguous or absent carrier, or a hash8 that is not that digest → ``(None, reason)``:
    no block, the pairs stay unproven."""
    lid = str(res.get("launch_id") or "")
    d = Path(record_dir) / LAUNCHES_DIR / lid
    facts = _engine_log_facts(d)
    if not facts["vllm_version"]:
        return None, f"{lid}: vllm_server.log carries no 'Initializing a V1 LLM engine (vX)' line"
    if not facts["flashinfer_version"]:
        return None, f"{lid}: vllm_server.log carries no flashinfer autotune-cache line (the installed flashinfer version)"
    venv = _launch_recorded_venv(d)
    how: Dict[str, Any] = {"vllm_log_line": facts["vllm_line"], "flashinfer_log_line": facts["flashinfer_line"], "carrier_job_id": None,
                           "carrier_job_log_venv_line": None, "venv_name": None, "requirements_sha_match": None,
                           "standup_block_lacked_identity": bool(standup_block_lacked_identity)}
    checkout = sha = None
    allow_tip = False
    if venv is not None:
        name, how["venv_name"] = venv[0], venv[0]
        how["carrier_job_log_venv_line"] = f"(the launch's own {venv[1]})"
        allow_tip = True
    elif job_log_binding:
        host, started = str(res.get("host") or ""), str(res.get("started_utc") or "")
        if not host or not started:
            return None, f"{lid}: the result records no host / started_utc — no carrier job can be bound"
        want_out = str(Path(record_dir).resolve())
        hits = []
        for lg in job_logs:
            try:
                text = Path(lg).read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            mn = JOB_LOG_NODE_RE.search(text); ts = JOB_LOG_TS_RE.findall(text); mv = JOB_LOG_VENV_RE.search(text); mo = JOB_LOG_OUT_RE.search(text)
            if not (mn and ts and mv and mo):
                continue
            logged_out = mo.group(1)
            try:
                same_record = str(Path(logged_out).resolve()) == want_out or logged_out == str(record_dir)
            except OSError:
                same_record = logged_out == str(record_dir)
            if not same_record or mn.group(1) != host or not (ts[0] <= started <= ts[-1]):
                continue
            mj = re.search(r"_(\d+)\.out$", Path(lg).name); mjl = JOB_LOG_JOB_RE.search(text)
            job_id = (mj.group(1) if mj else None) or (mjl.group(1) if mjl else None)
            mr = JOB_LOG_REPO_RE.search(text); ms = JOB_LOG_SHA_RE.search(text)
            ls = text.rfind("\n", 0, mv.start()) + 1
            hits.append({"job_log": str(lg), "venv_path": mv.group(1), "venv_name": mv.group(2), "node": mn.group(1), "span": [ts[0], ts[-1]], "out": logged_out,
                         "job_id": job_id, "checkout": (mr.group(1) if mr else None), "sha": (ms.group(1) if ms else None),
                         "line": text[ls:text.find("\n", mv.end()) if text.find("\n", mv.end()) != -1 else None][:240]})
        if not hits:
            return None, (f"{lid}: no carrier job log of this record (logged out={want_out}) names node {host} with a span covering {started} — the "
                          "carrier that stood this launch up is missing or was not its own; the venv cannot be bound")
        if len(hits) > 1:
            return None, f"{lid}: {len(hits)} carrier job logs ({', '.join(Path(h['job_log']).name for h in hits)}) of this record cover {started} on {host} — ambiguous, no block"
        hit = hits[0]
        name = hit["venv_name"]; checkout, sha = hit["checkout"], hit["sha"]
        how.update({"carrier_job_id": hit["job_id"], "carrier_job_log_venv_line": hit["line"], "venv_name": name, "carrier_job_log": Path(hit["job_log"]).name,
                    "carrier_node": hit["node"], "carrier_span": hit["span"], "carrier_out": hit["out"], "carrier_checkout": checkout, "carrier_sha": sha})
    else:
        return None, f"{lid}: launch_cmd.json names no venv (a bare `vllm` command) and the carrier-job binding is off"
    digests, why = venv_requirements_digests(name, checkout=checkout, sha=sha, repo_root=repo_root, allow_tip=allow_tip)
    if digests is None:
        how["requirements_sha_match"] = False
        return None, f"{lid}: {why}"
    how.update({"requirements_sha_match": True, "requirements_source": digests["requirements_source"], "venv_requirements_sha256": digests["venv_requirements_sha256"],
                "variant": digests["variant"]})
    # the engine logs vLLM's PUBLIC version ("v0.25.1"); a standup states the distribution's ("0.25.1+cu129", importlib.metadata). The
    # requirements the venv was built from name the wheel: when its public part is the logged version, the identity carries the
    # distribution version (comparable to any standup's); a disagreement refuses — the log and the pinned wheel cannot both be true
    vllm_version = facts["vllm_version"]
    pinned = digests.get("vllm_pin")
    if pinned:
        if pinned.split("+", 1)[0] != str(vllm_version).split("+", 1)[0]:
            return None, (f"{lid}: the engine log names vLLM v{vllm_version} but the requirements the venv {name!r} was built from pin "
                          f"vllm {pinned} — the launch did not run the pinned wheel; no block")
        how["vllm_version_from"] = f"requirements pin {pinned} (the engine log names v{vllm_version})"
        vllm_version = pinned
    return ({"requirements_sha256": digests["venv_requirements_sha256"], "variant": digests["variant"], "name": name,
             "engine_identity": {"vllm_version": vllm_version, "flashinfer_version": facts["flashinfer_version"], "base_requirements_sha256": digests["base_requirements_sha256"]},
             "engine_identity_source": IDENTITY_SOURCE_PROVENANCE, "provenance": how}, None)
def engine_side_block(record_dir: Path, res: Mapping[str, Any], *, job_logs: Optional[Sequence[Path]] = None,
                      repo_root: Optional[Path] = None) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """The SERVING side of a launch's certificate: its own result-level venv block (the standup's serving_venv_block(); an older
    block carries no source and is the standup's), else the provenance-derived block; ``(block, refusal)``."""
    own = res.get(SERVING_VENV_KEY)
    lacked = False
    if isinstance(own, Mapping) and own.get("requirements_sha256"):
        eid = own.get("engine_identity")
        if isinstance(eid, Mapping) and all(eid.get(k) for k in sa.ENGINE_IDENTITY_FIELDS):
            b = dict(own)
            if b.get("engine_identity_source") is None:
                b["engine_identity_source"] = IDENTITY_SOURCE_STANDUP
            return b, None
        # a block that carries a digest but NO engine identity (the 80415c1a9 writer) is not a standup identity; the
        # launch is attested from its provenance like any other, and the block records that the standup's own block lacked it
        lacked = True
    logs = list(job_logs) if job_logs is not None else record_job_logs(record_dir, str(res.get("model_key") or "") or None)
    return launch_engine_provenance(record_dir, res, job_logs=logs, job_log_binding=True, repo_root=repo_root, standup_block_lacked_identity=lacked)


#: where a block's engine identity was MEASURED — the HF+PEFT reference process (the judgements), the standup process that served
#: the adapters, or both when the pair's agreement composes them; the record says it plainly, the gate admits on the identity
IDENTITY_SOURCE_REFERENCE = "reference_process"
IDENTITY_SOURCE_STANDUP = "standup_process"


def serving_venv_block(facts: Optional[Mapping[str, Any]] = None, *, source: Optional[str] = None) -> Dict[str, Any]:
    """``{requirements_sha256, variant, name}`` of THIS process's serving venv (:func:`serving_attestation.serving_venv_facts`),
    or the same keys over ``facts`` (tests). ``variant`` None = the base; a digest absent means the process was not stood up
    through the venv selector (the admission gate reads that as unproven)."""
    f = dict(facts) if facts is not None else sa.serving_venv_facts()
    ex = f.get("exported") or {}
    name = f.get("name")
    digest = ex.get("requirements_sha256")
    problems: List[str] = []
    # the exported identity must be THIS process's venv — the exported path is sys.prefix, and the
    # venv name the recipe derived from the digest (variant + the digest's first 8 hex) agrees with the exported digest and
    # variant; a disagreement is recorded as an UNPROVEN block (no digest), never as a borrowed identity
    if f.get("exported_matches_prefix") is False:
        problems.append(f"exported venv {ex.get('venv')!r} is not this process's prefix {f.get('prefix')!r}")
    if digest is not None:
        if not isinstance(digest, str) or len(digest) != 64:
            problems.append(f"exported requirements_sha256 {digest!r} is not 64 hex")
        elif f.get("hash8") is not None and str(f.get("hash8")) != digest[:8]:
            problems.append(f"the venv name's hash8 {f.get('hash8')!r} is not the exported digest's {digest[:8]!r}")
        if (ex.get("variant") or None) != (f.get("variant") or None):
            problems.append(f"exported variant {ex.get('variant')!r} is not the venv name's {f.get('variant')!r}")
    # the ENGINE identity (vllm version, the base requirements digest both venvs share, flashinfer version — the one
    # environment-dependent engine dispatch, qwen_gdn_linear_attn's prefill backend) rides the block verbatim from the facts; the
    # standup's admission gate compares THIS to its own, the HF-reference venv (digest / variant / name) stays recorded as provenance
    eid = f.get("engine_identity")
    engine_identity = dict(eid) if isinstance(eid, Mapping) else None
    if problems:
        return {"requirements_sha256": None, "variant": None, "name": name, "engine_identity": None, "engine_identity_source": source,
                "problem": "; ".join(problems)}
    return {"requirements_sha256": digest, "variant": (ex.get("variant") or f.get("variant")), "name": name, "engine_identity": engine_identity,
            "engine_identity_source": source}


def compose_certificate_venv(hf_blocks: Sequence[Optional[Mapping[str, Any]]], engine_block: Optional[Mapping[str, Any]]
                             ) -> Optional[Dict[str, Any]]:
    """One certificate venv block from the HF reference's blocks (must agree) and the serving side's block (may be None):
    ``{requirements_sha256, variant, name}`` = the reference venv; ``engine_identity`` / ``engine_identity_source`` = the serving
    side's; ``reference`` and ``serving`` the two sides in full (null when a side has none); ``identity_agrees`` whether both sides
    name the same engine identity. None when the reference side does not agree or is absent (the HF judgements are the certificate)."""
    ref = _agreeing_venv(list(hf_blocks))
    if ref is None:
        return None
    reference = {"engine_identity": ref.get("engine_identity"), "engine_identity_source": ref.get("engine_identity_source") or IDENTITY_SOURCE_REFERENCE,
                 "requirements_sha256": ref.get("requirements_sha256"), "variant": ref.get("variant"), "name": ref.get("name")}
    serving = None
    if isinstance(engine_block, Mapping) and isinstance(engine_block.get("engine_identity"), Mapping):
        serving = {"engine_identity": dict(engine_block["engine_identity"]), "engine_identity_source": engine_block.get("engine_identity_source") or IDENTITY_SOURCE_STANDUP,
                   "derivation": (dict(engine_block["provenance"]) if isinstance(engine_block.get("provenance"), Mapping) else None),
                   "requirements_sha256": engine_block.get("requirements_sha256"), "variant": engine_block.get("variant"), "name": engine_block.get("name")}
    out = {"requirements_sha256": ref.get("requirements_sha256"), "variant": ref.get("variant"), "name": ref.get("name"),
           "engine_identity": (serving["engine_identity"] if serving else None),
           "engine_identity_source": (serving["engine_identity_source"] if serving else None),
           "reference": reference, "serving": serving,
           "identity_agrees": (bool(serving) and reference["engine_identity"] is not None and serving["engine_identity"] == reference["engine_identity"])}
    return out


def _agreeing_venv(blocks: Sequence[Optional[Mapping[str, Any]]]) -> Optional[Dict[str, Any]]:
    """ONE venv block when every given block is present and names the same digest; None otherwise (absent = unproven)."""
    seen = [b for b in blocks if isinstance(b, Mapping) and b.get("requirements_sha256")]
    if not seen or len(seen) != len(list(blocks)):
        return None
    digests = {str(b["requirements_sha256"]) for b in seen}
    if len(digests) != 1:
        return None
    # the engine identity must agree too where it is carried; a block without one leaves the agreement without one
    eids = [b.get("engine_identity") for b in seen]
    if any(isinstance(e, Mapping) for e in eids) and len({json.dumps(e, sort_keys=True) if isinstance(e, Mapping) else None for e in eids}) != 1:
        return None
    out = dict(seen[0])
    # the agreement names every process the identity was measured in (the reference judgements, the standup that served, or both);
    # blocks that carry no source (older records) keep their shape
    sources = sorted({str(b.get("engine_identity_source")) for b in seen if b.get("engine_identity_source")})
    if sources:
        out["engine_identity_source"] = "+".join(sources)
    elif "engine_identity_source" in out:
        out["engine_identity_source"] = None
    return out


def parity_prompts_sha(prompts: Sequence[Sequence[Mapping[str, str]]]) -> str:
    """The prompt set's identity (sha256 of its JSON) — stamped on the plan, every launch's ``probes_v2`` and every v2 record."""
    return hashlib.sha256(json.dumps([[dict(m) for m in pr] for pr in prompts], sort_keys=True).encode("utf-8")).hexdigest()


def parity_certificate_block(prompts: Sequence[Sequence[Mapping[str, str]]] = PARITY_V2_PROMPTS) -> Dict[str, Any]:
    """The plan's ``parity_certificate`` block: version, the prompts and their sha."""
    return {"version": PARITY_CERTIFICATE_VERSION, "prompts": [[dict(m) for m in pr] for pr in prompts],
            "sha256": parity_prompts_sha(prompts), "n_prompts": len(prompts),
            "rule": ("PASS iff the greedy first token agrees on every prompt and max_p |d_p| <= max(tol, K x floor, M x max_p "
                     "sensitivity_p); floor = max_p of the HF-vs-engine BASE |d_p| on the same standup")}


def plan_parity_prompts(plan: Mapping[str, Any]) -> Tuple[Tuple[Dict[str, str], ...], ...]:
    """The v2 prompt set a plan carries (``parity_certificate.prompts``), else the code's (a plan written before v2)."""
    blk = plan.get("parity_certificate") if isinstance(plan, Mapping) else None
    if isinstance(blk, Mapping) and blk.get("prompts"):
        return tuple(tuple(dict(m) for m in pr) for pr in blk["prompts"])
    return PARITY_V2_PROMPTS
#: the checkpoint's ``hold`` while a reference-only re-run rewrites a finished record's results; the
#: attempt carries when it started (hpo_probe_status reads both: an in-progress parity record is never `current`)
HOLD_REFERENCE_ONLY = "reference_only_in_progress"
REFERENCE_RERUN_STARTED_KEY = "reference_rerun_started_utc"


def log(msg: str) -> None:
    print(f"{TAG} {utc_now()} {msg}", flush=True)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json_atomic(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
    os.replace(tmp, path)


def _json_default(o: Any) -> Any:
    if isinstance(o, Path):
        return str(o)
    if dataclasses.is_dataclass(o):
        return dataclasses.asdict(o)
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    return str(o)


def default_out(model_key: str) -> Path:
    from sisteer.capeval.serving.native_standup import shared_root   # the cluster-root rule, once
    root = shared_root() or (Path.home() / "exp")
    jobid = os.environ.get("SLURM_JOB_ID", "local")
    return root / "exp" / "multilora_cert" / f"{model_key}_probe_{jobid}"


def dims_for(model_key: str) -> lm.ArchDims:
    """The ``ArchDims`` of a known model key (``lora_modules.ARCH_DIMS_BY_MODEL_KEY``, the verified table)."""
    dims = lm.arch_dims_for(model_key)
    if dims is None:
        raise SystemExit(f"{TAG} unknown model key {model_key!r}; known: {list(MODEL_DIMS_KEYS)}")
    return dims


def _Dims(d: Mapping[str, Any]) -> lm.ArchDims:
    """The dims rebuilt from ``probe_plan.json``'s record (``dataclasses.asdict`` of an ArchDims)."""
    return lm.ArchDims(**d)


# --------------------------------------------------------------------------
# plan
# --------------------------------------------------------------------------

def visible_devices(env: Optional[Mapping[str, str]] = None) -> Optional[List[str]]:
    """The tokens of the environment's ``CUDA_VISIBLE_DEVICES`` — Slurm's export on a cluster job (indices or UUIDs,
    kept opaque), in order; None when the variable is unset or empty. READ only: the probe never sets the variable in
    its own environment; a child engine receives a strict subset of these tokens (:func:`device_slots`)."""
    env = os.environ if env is None else env
    raw = str(env.get(VISIBLE_DEVICES_ENV, "") or "")
    toks = [t.strip() for t in raw.split(",") if t.strip()]
    return toks or None


def device_slots(visible: Sequence[str], *, per_standup: int) -> List[Tuple[str, ...]]:
    """Consecutive, DISJOINT ``per_standup``-token slices of ``visible`` — one per concurrent standup. Every slot is a
    strict subset of the tokens handed in (never widened, never a hard-coded index); a remainder shorter than
    ``per_standup`` stays idle. Duplicate tokens refuse (two standups would share a device)."""
    per = int(per_standup)
    if per < 1:
        raise ValueError(f"{TAG} per_standup must be >= 1, got {per_standup!r}")
    toks = [str(t) for t in visible]
    if len(set(toks)) != len(toks):
        raise ValueError(f"{TAG} duplicate device tokens in {VISIBLE_DEVICES_ENV}={','.join(toks)} — refusing to pin standups")
    return [tuple(toks[i:i + per]) for i in range(0, len(toks) - per + 1, per)]


def resolve_parallel(parallel: Any, *, tp: int, visible: Optional[Sequence[str]], in_slurm: bool
                     ) -> Tuple[int, Optional[List[Tuple[str, ...]]]]:
    """``(N, slots)`` for ``run --parallel``. ``auto`` / None: N = ``len(visible) // tp`` under Slurm (the
    allocation's device count per standup), 1 anywhere else. An explicit N must FIT the visible devices (N × tp tokens):
    a wider request REFUSES by name — the allocation is never widened — and N > 1 with no device list at all refuses
    (nothing to pin to; a hard-coded index is never substituted). ``slots``: the first N disjoint subsets when devices
    are known; None when N == 1 and none are (the engine then inherits the parent's environment unchanged — the
    single-standup behaviour)."""
    tp = int(tp)
    if tp < 1:
        raise ValueError(f"{TAG} tp must be >= 1, got {tp}")
    vis = [str(t) for t in visible] if visible else None
    capacity = (len(vis) // tp) if vis else 0
    auto = parallel is None or str(parallel).strip().lower() in ("", PARALLEL_AUTO)
    if auto:
        n = capacity if (in_slurm and vis) else 1
        if in_slurm and vis and capacity < 1:
            raise ValueError(f"{TAG} --parallel {PARALLEL_AUTO}: {VISIBLE_DEVICES_ENV}={','.join(vis)} holds {len(vis)} device(s), "
                             f"fewer than the {tp} one standup needs (tp={tp})")
    else:
        try:
            n = int(str(parallel).strip())
        except ValueError:
            raise ValueError(f"{TAG} --parallel must be an integer or '{PARALLEL_AUTO}', got {parallel!r}") from None
        if n < 1:
            raise ValueError(f"{TAG} --parallel must be >= 1, got {n}")
        if n > 1 and not vis:
            raise ValueError(f"{TAG} --parallel {n} needs {VISIBLE_DEVICES_ENV} to pin each standup to its own device subset, "
                             "but it is unset — nothing to pin to (a hard-coded device index is never substituted)")
        if vis and n > capacity:
            raise ValueError(f"{TAG} --parallel {n} needs {n * tp} device(s) (tp={tp}); {VISIBLE_DEVICES_ENV}={','.join(vis)} "
                             f"holds {len(vis)} — refusing to widen the allocation")
    if not vis:
        return n, None
    if capacity < 1:
        raise ValueError(f"{TAG} {VISIBLE_DEVICES_ENV}={','.join(vis)} holds {len(vis)} device(s), fewer than the {tp} one "
                         f"standup needs (tp={tp})")
    return n, device_slots(vis, per_standup=tp)[:n]


def device_tag(devices: Optional[Sequence[str]]) -> str:
    """The log-line marker of a pinned standup: ``[gpu 3]`` / ``[gpu 4,5]``; empty when the engine is unpinned."""
    return f" [gpu {','.join(str(d) for d in devices)}]" if devices else ""


def launch_admission(dims: Any, m: sm.Manifest, L: int, *, requested_S: int, kv_budget_bytes: float,
                     recurrent_state_bytes: int, batched_token_budget: Optional[int]) -> Dict[str, Any]:
    """The S a launch of ``m`` at window ``L`` is planned at:
    ``standup_manifest.admitted_max_num_seqs`` — the packed-stride int32
    ceiling, the KV budget, the requested S, and the batched-token term
    ``floor(batched_token_budget / L)`` (the shared module carries the term, so
    a production standup sizing through it stays under the same measured
    prefill budget). The record carries ``budget_S`` / ``batched_token_budget``
    / ``batched_tokens`` and ``binding`` names ``batched_tokens`` when that
    term is the one that bound. A budget of None (nothing measured yet)
    applies no cap and says so."""
    return sm.admitted_max_num_seqs(dims, m.target_class, int(L), kv_budget_bytes=kv_budget_bytes,
                                    recurrent_state_bytes=recurrent_state_bytes, requested_S=int(requested_S),
                                    batched_token_budget=(int(batched_token_budget) if batched_token_budget is not None else None))


def launch_id(manifest_name: str, cap: int, L: int, *, tp: int = 1) -> str:
    # a ladder manifest is already named `<base>_cap<C>`; never double the suffix
    stem = manifest_name if manifest_name.endswith(f"_cap{cap}") else f"{manifest_name}_cap{cap}"
    return f"{stem}_L{L}" + (f"_tp{tp}" if tp != 1 else "")


def _ladder_manifest(base: sm.Manifest, cap: int, dims: Any) -> sm.Manifest:
    """``base`` re-pinned at a HIGHER cap: the zero adapter's rank becomes the
    cap (the fitted --max-lora-rank follows the largest declared r; on a
    routed-experts model the zero adapter's routed rank follows the cap too)."""
    per = sm.pair_per_expert_rank(dims, base.set_id, cap)
    r_e = per["routed_expert"] if per else None
    adapters = tuple(dataclasses.replace(a, rank=cap, per_expert_rank=r_e) if a.kind == sm.KIND_ZERO else a
                     for a in base.adapters)
    # moe_lora_rank=None: re-derived from the re-pinned adapters (the zero adapter's routed rank follows the cap)
    return dataclasses.replace(base, name=f"{base.name}_cap{cap}", adapters=adapters, max_lora_rank=cap, moe_lora_rank=None,
                               notes=base.notes + (f"rank-cap ladder point at cap {cap} (natural {base.max_lora_rank})",))


def _carries_control(m: sm.Manifest) -> bool:
    """The manifest that serves the pair's SINGLE-RANK adapter — the probe's ladder and memo rows key on it
    (``S1cc`` today; the legacy ``S1`` / ``single`` on an older record)."""
    return m.carries_role(sm.ROLE_CONTROL)


def _is_union(m: sm.Manifest) -> bool:
    """The pair's UNION manifest — the served launch class (``standup_manifest.build_union_manifest``; the probe owner's
    ruling of 2026-10-11): the ``single`` shape, every genuine role in one engine. A legacy record's own ``single`` IS that
    shape (the old one-server ``wo``) and reads the same way everywhere."""
    return m.topology == sm.TOPOLOGY_SINGLE


def _union_class(m: sm.Manifest) -> Tuple[int, int]:
    """``(rank cap, slots)`` — the class a launch of ``m`` certifies, in ``geometry.launch_class``'s order."""
    return int(m.max_lora_rank), int(m.max_loras)


def _planned_class(rec: Mapping[str, Any], manifests_by_name: Mapping[str, sm.Manifest]) -> Optional[Tuple[int, int]]:
    """``(rank cap, slots)`` a planned launch record covers — its ``cap`` and ``max_loras`` (a record written before the plan
    carried ``max_loras`` reads its manifest's); None when neither is known."""
    slots = rec.get("max_loras")
    if not isinstance(slots, int) or isinstance(slots, bool):
        m = manifests_by_name.get(rec["manifest"])
        slots = m.max_loras if m is not None else None
    if slots is None:
        return None
    return int(rec["cap"]), int(slots)


def _planned_target_class(rec: Mapping[str, Any], manifests_by_name: Mapping[str, sm.Manifest]) -> Optional[frozenset]:
    """The module class a planned launch's engine wraps — the record's ``target_class`` (its manifest's on an older record)."""
    tc = rec.get("target_class")
    if not tc:
        m = manifests_by_name.get(rec["manifest"])
        tc = m.target_class if m is not None else None
    return frozenset(str(t) for t in tc) if tc else None


def _served_class_candidates(plan: Mapping[str, Any], pair: Mapping[str, Any], union: sm.Manifest, L: int,
                             manifests_by_name: Mapping[str, sm.Manifest]) -> List[Mapping[str, Any]]:
    """The pair's PLANNED launches at window ``L`` that can certify the class a steered standup of the pair serves — the
    union's ``(rank cap, slots)`` (:func:`_union_class`) — in plan order. Two conditions, both the fold's own: the launch
    serves one of the pair's manifests (``serves``; the ladder's re-pins serve no pair manifest and are never folded), and its
    class DOMINATES the union's — rank cap >= and slots >= at the SAME window, ``geometry.least_admitted_by_window``'s rule —
    OVER THE UNION'S TARGET CLASS. The target class is part of the rule: the
    shared S2 (``o_proj``, cap 8, five slots) serves every MLP-bearing pair and out-counts a four-slot ``mlp`` union on
    (cap, slots) alone, but its o_proj-only carriers are a lighter workload than the pair's cells — a record whose only
    "dominating" launch is S2 has never served the pair's class, and the site record would size that class from S2's S
    (``least_admitted_by_window`` narrows to the tightest dominating footprint, and with no union folded that is S2's).
    An engine over another module class certifies nothing about this one, whatever its cap and slot count.
    Planned, not passed: the verdict reads each candidate's outcome (:func:`_resolve_served_class`); the reopen's extension
    adds the union only where NO candidate is planned — a planned twin that died at a window is evidence the class cannot
    come up there, and a second copy would die the same way."""
    names = set(pair["manifests"])
    served = _union_class(union)
    target = frozenset(str(t) for t in union.target_class)
    out: List[Mapping[str, Any]] = []
    for rec in plan["launches"]:
        if int(rec["L"]) != int(L) or not names.intersection(rec.get("serves") or [rec["manifest"]]):
            continue
        cls = _planned_class(rec, manifests_by_name)
        if cls is None or _planned_target_class(rec, manifests_by_name) != target:
            continue
        if cls[0] >= served[0] and cls[1] >= served[1]:
            out.append(rec)
    return out


def _served_class_dominated(plan: Mapping[str, Any], pair: Mapping[str, Any], union: sm.Manifest, L: int,
                            manifests_by_name: Mapping[str, sm.Manifest]) -> List[str]:
    """The ids of the pair's PLANNED launches at window ``L`` whose class dominates the union's over the union's target class
    (:func:`_served_class_candidates`) — what the reopen's extension asks before it adds the union's launch there."""
    return [str(rec["id"]) for rec in _served_class_candidates(plan, pair, union, L, manifests_by_name)]


def _attested_class(res: Optional[Mapping[str, Any]]) -> Optional[Tuple[int, int]]:
    """The ``(rank cap, slots)`` a launch's engine ATTESTED (``result.engine`` ``max_lora_rank`` / ``max_loras`` — the worker
    report's ``lora_max_lora_rank`` / ``lora_max_loras``), or None when the result carries no positive pair: the same
    reading as :func:`launch_class_fields`, without the fold's problem list."""
    e = (res or {}).get("engine") or {}
    vals = []
    for k in ("max_lora_rank", "max_loras"):
        v = e.get(k)
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            return None
        vals.append(int(v))
    return vals[0], vals[1]


_SERVED_PASSED, _SERVED_UNRUN, _SERVED_REPLAN, _SERVED_DEAD, _SERVED_UNPLANNED = ("passed", "unrun", "replan", "dead", "unplanned")


def _resolve_served_class(cands: Sequence[Mapping[str, Any]], results: Mapping[str, Mapping[str, Any]],
                          manifests_by_name: Mapping[str, sm.Manifest],
                          replan_oom: Callable[[Optional[Mapping[str, Any]], Optional[Mapping[str, Any]]], bool]
                          ) -> Tuple[str, Optional[Mapping[str, Any]], Optional[str]]:
    """How the pair's SERVED CLASS stands at one window over its candidate launches (:func:`_served_class_candidates`, plan
    order): ``passed`` — the first candidate that PASSED and whose attested engine class (when its report carries one) IS
    its planned class certifies the class there (a passed launch attesting
    another class — the run-time ``launch_class`` check fails it on a live record; a hand-edited one keeps ``passed`` — never
    covers the class it did not serve); else ``unrun`` — a candidate has not run to a terminal state (undecided: the pair's
    outstanding launches hold it); else ``replan`` — a candidate's OOM was a start-time prefill above the measured budget
    (undecided, re-run by ``--replan``); else ``dead`` — every candidate ran and none served the class (a genuine OOM, a
    refusal, a failed check, an attested class that is not the planned one): EVIDENCE the class cannot come up at that window;
    else ``unplanned`` — nothing of the class was ever planned there (a record written before the union was planned: the
    reopen's extension adds it). Returns ``(kind, the deciding candidate, the detail a reason names)``."""
    if not cands:
        return _SERVED_UNPLANNED, None, None
    for rec in cands:
        res = results.get(rec["id"])
        if res is not None and res.get("status") == STATUS_PASSED:
            attested = _attested_class(res)
            if attested is None or attested == _planned_class(rec, manifests_by_name):
                return _SERVED_PASSED, rec, None
    for rec in cands:
        res = results.get(rec["id"])
        if res is None or res.get("status") not in TERMINAL_STATUSES:
            return _SERVED_UNRUN, rec, None
    for rec in cands:
        if replan_oom(rec, results.get(rec["id"])):
            return _SERVED_REPLAN, rec, None
    # every candidate ran and none served the class: name the union's own launch when it is among them, else the first
    rec = next((c for c in cands if str(c.get("topology")) == sm.TOPOLOGY_SINGLE), cands[0])
    res = results.get(rec["id"])
    attested = _attested_class(res) if res is not None and res.get("status") == STATUS_PASSED else None
    if attested is not None:
        detail = f"attested_class={attested}!=planned={_planned_class(rec, manifests_by_name)}"
    else:
        detail = _fail_reason(rec, res)
    return _SERVED_DEAD, rec, detail


def _launch_priority(m: sm.Manifest, L: int, ladder_window: int) -> int:
    # natural cap @ ladder window (pass/fail) < natural @ 49152 < S2 < other windows < ladder < fallbacks
    if m.topology in sm.S1_FALLBACKS:
        return 60
    if m.topology == sm.TOPOLOGY_S2:
        return 30
    if int(L) == int(ladder_window):
        return 10
    if int(L) == 49152:
        return 20
    return 40


def _launch_record(m: sm.Manifest, L: int, kind: str, priority: int, *, dims: Any, tp: int, requested_S: int,
                   budget_tokens: Optional[int], budget: Mapping[str, Any],
                   manifests_by_name: Mapping[str, sm.Manifest]) -> Dict[str, Any]:
    """ONE planned launch of ``m`` at window ``L`` (``build_plan`` and the reopen's plan extension build theirs here):
    the expected LoRA bytes at the cap with the experts at the manifest's ``moe_lora_rank``, the KV budget the engine
    will have after weights + buffers + graphs, the admission (stride ceiling, KV, requested S, the measured
    batched-token budget) and the VRAM estimate. ``roles`` names the genuine roles the manifest serves and
    ``moe_lora_rank`` its expert-buffer rank (None on a dense class), so a verdict or a reader never re-derives either
    from the topology label."""
    cap = m.max_lora_rank
    lora_bytes = sm.expected_lora_bytes(m, cap, dims)
    usable = float(budget["gpu_memory_utilization"]) * float(budget["total_vram_bytes"])
    kv_budget = max(0.0, usable - float(budget["base_weights_bytes"]) - lora_bytes["total"] - float(budget["graphs_activations_bytes"]))
    adm = launch_admission(dims, m, L, requested_S=int(requested_S), kv_budget_bytes=kv_budget,
                           recurrent_state_bytes=int(budget.get("recurrent_state_per_seq_bytes") or 0),
                           batched_token_budget=budget_tokens)
    est = sm.estimated_vram_bytes(m, cap, dims, base_weights_bytes=float(budget["base_weights_bytes"]),
                                  kv_per_seq_bytes=adm["kv_per_seq_bytes"], S=max(1, adm["admitted_S"]),
                                  graphs_activations_bytes=float(budget["graphs_activations_bytes"]),
                                  gpu_memory_utilization=float(budget["gpu_memory_utilization"]),
                                  total_vram_bytes=float(budget["total_vram_bytes"]),
                                  batched_tokens=max(1, adm["admitted_S"]) * int(L))   # the prefill batch, recorded
    condition = None
    if m.fallback_of is not None:
        parent = manifests_by_name[m.fallback_of]
        condition = {"launch": launch_id(m.fallback_of, parent.max_lora_rank, L, tp=tp),
                     "outcome": [STATUS_OOM, STATUS_REFUSED, STATUS_ERROR]}
    return {"id": launch_id(m.name, cap, L, tp=tp), "kind": kind, "manifest": m.name, "serves": [m.name],
            "set": m.set_id, "rank": m.rank, "topology": m.topology, "cap": cap, "L": int(L), "tp": int(tp),
            "target_class": list(m.target_class), "max_loras": m.max_loras,
            "moe_lora_rank": m.moe_lora_rank, "roles": list(m.genuine_roles),
            "requested_S": adm["requested_S"], "admitted_S": adm["admitted_S"], "admission": adm,
            "expected_lora_bytes": lora_bytes, "estimated_vram": est, "kv_budget_bytes": kv_budget,
            "condition": condition, "priority": int(priority)}


def build_plan(dims: Any, *, model_key: str, model_id: str, revision: str,
               sets: Sequence[str] = DEFAULT_SETS, ranks: Sequence[int] = DEFAULT_RANKS,
               windows: Sequence[int] = sm.PRODUCTION_WINDOWS, ladder_window: int = 4096,
               ladder_caps: Sequence[int] = sm.RANK_CAP_LADDER, tp: int = 1,
               requested_S: Mapping[int, int] = REQUESTED_S_BY_WINDOW,
               base_weights_bytes: float = sm.DEFAULT_BASE_WEIGHTS_BYTES,
               graphs_activations_bytes: float = sm.DEFAULT_GRAPHS_ACTIVATIONS_BYTES,
               gpu_memory_utilization: float = sm.DEFAULT_GPU_MEMORY_UTILIZATION,
               total_vram_bytes: float = sm.H200_TOTAL_BYTES,
               recurrent_state_per_seq_bytes: Optional[int] = None,
               synthetic_scale: float = DEFAULT_SYNTHETIC_SCALE,
               batched_token_budget: Optional[Mapping[str, Any]] = None,
               parity_scales: Optional[Mapping[str, float]] = None, parity_pass: bool = False) -> Dict[str, Any]:
    """The probe plan (module docstring). One launch per (manifest, cap,
    topology, window); identical engines (``Manifest.launch_key``) share one
    launch whose ``serves`` lists every manifest it certifies.
    ``batched_token_budget``: a MEASURED budget record (:func:`measured_budget`
    — ``tokens``, ``method``, the launch ids it came from); its ``tokens`` cap
    every launch's S at ``floor(tokens / L)``. None = unmeasured: no
    cap, recorded as such on the plan.

    THE SERVED LAUNCH CLASS (the probe owner's ruling of 2026-10-11): beside each
    pair's split servers (``standup_manifest.build_manifests``: S1cc / S1d [/ S2])
    the plan carries the pair's UNION manifest (``standup_manifest.build_union_
    manifest``: cells + zero at cap >= 2r [+ the carriers for ``wo``] — the ``(8, 8)``
    class the battery's steered standups serve at r = 4) as ONE natural launch at
    EVERY plan window, appended AFTER the servers in ``pairs[i]["manifests"]`` so the
    verdict's role order keeps ``certified_by`` on S1cc / S1d; the fold writes its
    engine class and ``geometry.least_admitted_by_window`` picks it for the served
    class at every window (the dominance rule transfers nothing between windows).
    The rank-cap ladder keeps riding S1cc. A PARITY PASS plans no union: it folds
    no geometry and judges adapters by name, and the union adds no adapter."""
    sets = [str(s) for s in sets]
    budget_rec: Dict[str, Any] = (dict(batched_token_budget) if batched_token_budget else
                                  {"method": BUDGET_METHOD_UNMEASURED, "tokens": None,
                                   "note": "no launch of this record has come up yet; the first passed launches measure "
                                           "the floor and `run --resume --replan` re-derives S from it"})
    budget_tokens = budget_rec.get("tokens")
    budget_tokens = int(budget_tokens) if budget_tokens is not None else None
    ranks = sorted(int(r) for r in ranks)
    windows = sorted(int(w) for w in windows)
    if int(ladder_window) not in windows:
        windows = sorted(windows + [int(ladder_window)])
    is_moe = bool(getattr(dims, "moe_layers", 0))
    if recurrent_state_per_seq_bytes is None:
        # no per-sequence recurrent state unless the caller DERIVED one from the checkpoint's config
        # (standup_manifest.gated_deltanet_state_per_seq_bytes — cmd_plan does, or --recurrent-state-mib
        # declares it); never a typed per-model constant, and MoE-ness says nothing about it
        recurrent_state_per_seq_bytes = 0
    manifests: Dict[str, sm.Manifest] = {}
    pairs: List[Dict[str, Any]] = []
    for s in sets:
        for r in ranks:
            ms = sm.build_manifests(dims, set_id=s, rank=r, base_weights_bytes=base_weights_bytes,
                                    graphs_activations_bytes=graphs_activations_bytes,
                                    gpu_memory_utilization=gpu_memory_utilization,
                                    total_vram_bytes=total_vram_bytes,
                                    recurrent_state_per_seq_bytes=recurrent_state_per_seq_bytes)
            if not parity_pass:
                # the served launch class (docstring): the union AFTER the servers — the role order reads S1cc / S1d first
                ms = list(ms) + [sm.build_union_manifest(dims, set_id=s, rank=r)]
            for m in ms:
                manifests[m.name] = m
            pairs.append({"set": s, "rank": r, "manifests": [m.name for m in ms],
                          "trainable_params": sm.trainable_params(dims, s, r),
                          "per_expert_rank": sm.pair_per_expert_rank(dims, s, r),
                          "lora_modules": list(sm.MODULE_SETS[s]),
                          "natural_cap": sm.rank_cap(2 * r)})
    # the ladder: per set, over the SPLIT server carrying the control role (S1cc — never the union, which carries it too),
    # each ladder cap above the natural cap of the LARGEST rank whose natural cap <= that cap
    ladder_manifests: Dict[str, sm.Manifest] = {}
    for s in sets:
        primaries = [m for m in manifests.values() if m.set_id == s and _carries_control(m) and not _is_union(m)]
        for cap in ladder_caps:
            fitting = [m for m in primaries if m.max_lora_rank <= cap]
            if not fitting:
                continue
            base = max(fitting, key=lambda m: m.rank)
            if base.max_lora_rank == cap:
                continue  # the natural launch IS this ladder point
            lm_ = _ladder_manifest(base, cap, dims)
            ladder_manifests[lm_.name] = lm_
    launches: List[Dict[str, Any]] = []
    by_key: Dict[str, Dict[str, Any]] = {}
    budget_block = {"base_weights_bytes": base_weights_bytes, "graphs_activations_bytes": graphs_activations_bytes,
                    "gpu_memory_utilization": gpu_memory_utilization, "total_vram_bytes": total_vram_bytes,
                    "recurrent_state_per_seq_bytes": recurrent_state_per_seq_bytes}

    def add_launch(m: sm.Manifest, L: int, kind: str, priority: int) -> None:
        key = f"{m.launch_key}|L{L}|tp{tp}"
        if key in by_key:
            by_key[key]["serves"].append(m.name)
            return
        rec = _launch_record(m, L, kind, priority, dims=dims, tp=tp, requested_S=int(requested_S.get(L, 8)),
                             budget_tokens=budget_tokens, budget=budget_block, manifests_by_name=manifests)
        by_key[key] = rec
        launches.append(rec)

    for m in manifests.values():
        for L in windows:
            add_launch(m, L, "fallback" if m.topology in sm.S1_FALLBACKS else "natural",
                       _launch_priority(m, L, int(ladder_window)))
    for m in ladder_manifests.values():
        add_launch(m, int(ladder_window), "ladder", 50)
    launches.sort(key=lambda r: (r["priority"], r["L"], r["set"], r["rank"], r["id"]))
    all_manifests = {**{k: v.as_dict() for k, v in manifests.items()},
                     **{k: v.as_dict() for k, v in ladder_manifests.items()}}
    return {"version": 1, "written_utc": utc_now(), "model_key": model_key, "model_id": model_id,
            "served_hf_id": model_id, "revision": revision, "dims": dataclasses.asdict(dims), "is_moe": is_moe,
            "sets": sets, "ranks": ranks, "windows": windows, "ladder_window": int(ladder_window),
            "ladder_caps": [int(c) for c in ladder_caps], "tp": int(tp),
            "requested_S_by_window": {str(k): int(v) for k, v in requested_S.items()},
            "budget": budget_block,
            "synthetic_scale": float(synthetic_scale), "pairs": pairs, "manifests": all_manifests,
            # a PARITY PASS record — assertion 4's adapters at the per-set benign scale (None = one scale for every set)
            "synthetic_scale_by_set": (_parity_scales_for(sets, parity_scales) if parity_scales is not None else None),
            "parity_scale_provenance": (PARITY_SCALE_PROVENANCE if parity_scales is not None else None),
            "parity_pass": bool(parity_pass),
            # the v2 parity certificate's prompt set rides the plan (every launch probes it; the reference judges it)
            "parity_certificate": parity_certificate_block(),
            "batched_token_budget": budget_rec,
            "launches": launches, "n_launches": len(launches)}


def parse_parity_scales(text: Optional[str]) -> Dict[str, float]:
    """``--parity-scales``: ``default`` (or empty) = :data:`PARITY_SCALES_BY_SET`; else ``set=scale,...`` over the module-set
    vocabulary, every scale a positive float. Refuses an unknown set or a malformed entry by name."""
    if text is None or not str(text).strip() or str(text).strip() == PARITY_SCALES_DEFAULT_WORD:
        return dict(PARITY_SCALES_BY_SET)
    out: Dict[str, float] = {}
    for tok in str(text).split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "=" not in tok:
            raise ValueError(f"--parity-scales entry {tok!r} is not set=scale")
        k, v = (x.strip() for x in tok.split("=", 1))
        if k not in lm.MODULE_SETS:
            raise ValueError(f"--parity-scales names set {k!r}, not one of {sorted(lm.MODULE_SETS)}")
        try:
            f = float(v)
        except ValueError:
            raise ValueError(f"--parity-scales {k}={v!r} is not a number") from None
        if not (f > 0):
            raise ValueError(f"--parity-scales {k}={f} must be > 0")
        out[k] = f
    return out


def _parity_scales_for(sets: Sequence[str], scales: Mapping[str, float]) -> Dict[str, float]:
    """The plan's ``synthetic_scale_by_set``: one positive scale for EVERY set the plan probes — a set without one refuses."""
    missing = [s for s in sets if s not in scales]
    if missing:
        raise ValueError(f"parity scales name no scale for set(s) {missing} (given {dict(scales)}; the table is {PARITY_SCALES_BY_SET})")
    return {str(s): float(scales[s]) for s in sets}


def plan_summary_lines(plan: Mapping[str, Any]) -> List[str]:
    lines = [f"probe plan: {plan['model_key']} {plan['model_id']} @ {plan['revision'][:12]} — "
             f"{len(plan['manifests'])} manifests, {plan['n_launches']} launches, tp={plan['tp']}"
             + (f"; PARITY PASS at per-set synthetic scales {plan.get('synthetic_scale_by_set')}" if plan.get("parity_pass")
                else (f"; per-set synthetic scales {plan.get('synthetic_scale_by_set')}" if plan.get("synthetic_scale_by_set") else ""))]
    for r in plan["launches"]:
        est = r["estimated_vram"]
        cond = f" if {r['condition']['launch']} in {r['condition']['outcome']}" if r.get("condition") else ""
        serves = "" if r["serves"] == [r["manifest"]] else f" (+{len(r['serves']) - 1} same-engine manifests)"
        lines.append(f"  p{r['priority']:>2} {r['kind']:<8} {r['id']:<44} class={','.join(r['target_class']):<60} "
                     f"S {r['requested_S']}->{r['admitted_S']} ({r['admission']['binding']}) "
                     f"lora {r['expected_lora_bytes']['total']/2**30:6.2f} GiB est {est['total']/1e9:6.1f} GB "
                     f"{'fits' if est['fits'] else 'OVER'}{serves}{cond}")
    return lines


def load_plan(out: Path) -> Dict[str, Any]:
    p = Path(out) / PLAN_FILE
    if not p.is_file():
        raise SystemExit(f"{TAG} no {PLAN_FILE} under {out}; run `plan` first")
    return json.loads(p.read_text(encoding="utf-8"))


def plan_manifest(plan: Mapping[str, Any], name: str) -> sm.Manifest:
    return sm.Manifest.from_dict(plan["manifests"][name])


# --------------------------------------------------------------------------
# adapters
# --------------------------------------------------------------------------

class ExporterUnavailable(RuntimeError):
    """``lora_export`` lacks the contract §3 API this step needs (that code is
    not on the tree yet) — refused BY NAME, never a silent substitute."""


class ProductionExporter:
    """The contract §3 exporters, bound by name at call time."""

    def __init__(self, model_id: str, revision: str, *, skeleton: Any = None,
                 synthetic_scale: float = DEFAULT_SYNTHETIC_SCALE) -> None:
        self.model_id = model_id
        self.revision = revision
        self._skeleton = skeleton
        self.scale = float(synthetic_scale)

    @staticmethod
    def _lx():
        from sisteer.capeval.serving import lora_export as lx
        return lx

    def _fn(self, name: str) -> Callable[..., Any]:
        fn = getattr(self._lx(), name, None)
        if fn is None:
            raise ExporterUnavailable(
                f"lora_export.{name} is not on this tree (contract §3; agent D1's export API) — "
                "the probe cannot build this adapter and refuses rather than substituting a layout")
        return fn

    def skeleton(self) -> Any:
        if self._skeleton is None:
            _cfg, skel = self._lx()._decoder_skeleton(self.model_id, self.revision)
            self._skeleton = skel
        return self._skeleton

    def export_synthetic(self, out_dir: Path, *, modules: Sequence[str], rank: int,
                         per_expert_rank: Optional[int], seed: int, scale: Optional[float] = None) -> Path:
        # `scale`: the slot's own synthetic scale (a parity pass hands the SET's benign scale); None = the plan's one
        blob = self._fn("synthetic_module_set_blob")(self.skeleton(), list(modules), int(rank),
                                                     seed=int(seed), scale=(self.scale if scale is None else float(scale)),
                                                     per_expert_rank=per_expert_rank)
        self._fn("export_module_set")(Path(out_dir), base_model=self.model_id, base_revision=self.revision,
                                      blob=blob)
        return Path(out_dir)

    def export_zero(self, out_dir: Path, *, modules: Sequence[str], rank: int,
                    per_expert_rank: Optional[int], bank: Path) -> Path:
        # export_zero_adapter_for_class owns the class dispatch (its o_proj branch IS
        # export_or_reuse_zero, byte for byte)
        d, _dig, _reused = self._fn("export_zero_adapter_for_class")(
            Path(bank), base_model=self.model_id, base_revision=self.revision,
            rank=int(rank), lora_modules=list(modules), per_expert_rank=per_expert_rank)
        return _link_or_copy_tree(d, out_dir)

    def target_class(self, adapter_dir: Path) -> Optional[Tuple[str, ...]]:
        fn = getattr(self._lx(), "adapter_target_class", None)
        if fn is None:
            return None
        return tuple(fn(Path(adapter_dir)))


def _link_or_copy_tree(src: Path, dst: Path) -> Path:
    """``dst`` := a copy of the adapter dir ``src`` — HARD-LINKED when both sit on
    one filesystem (an adapter is up to GBs of fp32 factors, copied into every
    manifest that lists it; vLLM only reads the dir), a plain copy otherwise."""
    dst = Path(dst)
    if dst.exists():
        shutil.rmtree(dst)
    try:
        shutil.copytree(src, dst, copy_function=os.link)
    except OSError:                                 # cross-device: fall back to bytes
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst)
    return dst


def adapter_bank_key(slot: sm.AdapterSlot, *, seed: Optional[int], scale: float) -> str:
    h = hashlib.sha256(json.dumps({"kind": slot.kind, "modules": list(slot.lora_modules), "rank": slot.rank,
                                   "r_e": slot.per_expert_rank, "seed": seed, "scale": scale},
                                  sort_keys=True).encode()).hexdigest()[:16]
    return f"{slot.kind}__{'-'.join(slot.lora_modules)}__r{slot.rank}__{h}"


SEED_BY_ROLE = {"control": 1, "constitution": 2, "difference": 3}


def slot_dir(out: Path, manifest_name: str, slot_name: str) -> Path:
    return Path(out) / ADAPTERS_DIR / manifest_name / slot_name


def build_adapter_dirs(plan: Mapping[str, Any], out: Path, exporter: Any, *,
                       only_manifests: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """Every manifest's slot directories under ``<out>/adapters/<manifest>/<slot>/``,
    exported ONCE per (kind, modules, rank, r_e, seed) into ``adapters/_bank/``
    and copied per slot (vLLM registers a directory per name). Idempotent: a
    slot dir with an ``adapter_config.json`` is kept. Returns the layout record
    (written to ``adapters/layout.json``)."""
    out = Path(out)
    bank = out / ADAPTERS_DIR / "_bank"
    bank.mkdir(parents=True, exist_ok=True)
    scale = float(plan.get("synthetic_scale", DEFAULT_SYNTHETIC_SCALE))
    # a parity pass exports each GENUINE slot at ITS SET's benign scale; the bank key carries the scale, so a
    # scale-100 bank and a parity bank never share an entry (adapter_bank_key)
    by_set = dict(plan.get("synthetic_scale_by_set") or {}) or None
    wanted = set(only_manifests) if only_manifests is not None else set(plan["manifests"])
    layout: Dict[str, Any] = {"written_utc": utc_now(), "model_id": plan["model_id"], "revision": plan["revision"],
                              "synthetic_scale": scale, "synthetic_scale_by_set": by_set, "manifests": {}, "refused": {}}
    for name in sorted(wanted):
        m = plan_manifest(plan, name)
        slots: Dict[str, Any] = {}
        for slot in m.adapters:
            d = slot_dir(out, name, slot.name)
            seed = SEED_BY_ROLE.get(slot.role) if slot.kind == sm.KIND_GENUINE else None
            scale_here = scale
            if by_set is not None and slot.kind == sm.KIND_GENUINE:
                if m.set_id not in by_set:
                    raise ValueError(f"plan.synthetic_scale_by_set names no scale for set {m.set_id!r} (manifest {name})")
                scale_here = float(by_set[m.set_id])
            key = adapter_bank_key(slot, seed=seed, scale=scale_here)
            src = bank / key
            try:
                if not (src / sa.ADAPTER_CONFIG_FILE).is_file():
                    if slot.kind == sm.KIND_GENUINE:
                        exporter.export_synthetic(src, modules=slot.lora_modules, rank=slot.rank,
                                                  per_expert_rank=slot.per_expert_rank, seed=int(seed),
                                                  **({"scale": scale_here} if by_set is not None else {}))
                    else:
                        exporter.export_zero(src, modules=slot.lora_modules, rank=slot.rank,
                                             per_expert_rank=slot.per_expert_rank, bank=bank / "_zero")
                if not (d / sa.ADAPTER_CONFIG_FILE).is_file():
                    _link_or_copy_tree(src, d)
                cfg = json.loads((d / sa.ADAPTER_CONFIG_FILE).read_text(encoding="utf-8"))
                cls = exporter.target_class(d) if hasattr(exporter, "target_class") else None
                slots[slot.name] = {"dir": str(d), "bank_key": key, "kind": slot.kind, "role": slot.role,
                                    "rank": slot.rank, "per_expert_rank": slot.per_expert_rank,
                                    "lora_modules": list(slot.lora_modules), "seed": seed,
                                    "synthetic_scale": (scale_here if slot.kind == sm.KIND_GENUINE else None),
                                    "adapter_config": {k: cfg.get(k) for k in
                                                       ("r", "lora_alpha", "target_modules", "target_parameters",
                                                        "rank_pattern", "alpha_pattern")},
                                    "inferred_target_class": list(cls) if cls is not None else None}
            except ExporterUnavailable as e:
                layout["refused"][f"{name}/{slot.name}"] = str(e)
                log(f"adapter {name}/{slot.name}: REFUSED — {e}")
        layout["manifests"][name] = {"target_class": list(m.target_class), "slots": slots}
    write_json_atomic(out / ADAPTERS_DIR / "layout.json", layout)
    return layout


def manifest_adapter_dirs(out: Path, m: sm.Manifest) -> Dict[str, Path]:
    return {a.name: slot_dir(out, m.name, a.name) for a in m.adapters}


def missing_adapter_dirs(out: Path, m: sm.Manifest) -> List[str]:
    return [n for n, d in manifest_adapter_dirs(out, m).items() if not (d / sa.ADAPTER_CONFIG_FILE).is_file()]


# --- the PEFT reference layout (runs only when peft imports) -----------------

def safetensors_header(adapter_dir: Path) -> Dict[str, Any]:
    """``{name: {"shape", "dtype"}}`` in header order for the adapter's
    ``adapter_model.safetensors`` — read from the file header only."""
    import struct
    p = Path(adapter_dir) / "adapter_model.safetensors"
    with open(p, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n).decode("utf-8"))
    header.pop("__metadata__", None)
    return {k: {"shape": list(v["shape"]), "dtype": v["dtype"]} for k, v in header.items()}


def compare_adapter_layouts(ours: Mapping[str, Any], peft: Mapping[str, Any]) -> Dict[str, Any]:
    """Tensor-for-tensor agreement of two adapter layouts (names, shapes, and
    — when the layouts carry routed-expert factors — the header order of the
    expert pairs: ``experts.base_layer.*`` = gate_up before ``experts.*`` =
    down). A dense layout's keys are compared by name and shape alone."""
    ours_keys, peft_keys = list(ours), list(peft)
    only_ours = sorted(set(ours_keys) - set(peft_keys))
    only_peft = sorted(set(peft_keys) - set(ours_keys))
    shape_mismatch = sorted(k for k in set(ours_keys) & set(peft_keys)
                            if list(ours[k]["shape"]) != list(peft[k]["shape"]))

    def expert_order(keys: Sequence[str]) -> List[str]:
        return [k for k in keys if ".mlp.experts." in k]
    order_ok = expert_order(ours_keys) == expert_order(peft_keys)
    return {"pass": not only_ours and not only_peft and not shape_mismatch and order_ok,
            "n_ours": len(ours_keys), "n_peft": len(peft_keys), "only_ours": only_ours[:20],
            "only_peft": only_peft[:20], "shape_mismatch": shape_mismatch[:20],
            "expert_header_order_agrees": order_ok}


def reference_model_class(cfg: Any) -> str:
    """The transformers class the CHECKPOINT names (``config.architectures[0]``) — the class whose module tree the
    export's PEFT names are spelled in (``lora_export.ModuleNamespace.checkpoint``: ``model.language_model.layers``
    on a multimodal checkpoint). ``sisteer.models.auto_model_class`` picks ``AutoModelForCausalLM`` first, which on
    an architecture registered in BOTH auto maps resolves to the text-only class (tree ``model.layers``): PEFT 0.20.0
    then binds NONE of the exported keys, warns, keeps B = 0, and the "reference" is the base model. Refuses a
    config that names no architecture — never a guess."""
    archs = list(getattr(cfg, "architectures", None) or [])
    if not archs or not isinstance(archs[0], str) or not archs[0]:
        raise RuntimeError(f"{TAG} the checkpoint's config names no `architectures` — the reference cannot pick the class whose "
                           "module tree the export's PEFT names are spelled in; refusing to guess a loader")
    return archs[0]


def export_decoder_prefix(adapter_dir: Path) -> Optional[str]:
    """The decoder prefix the export spelled its names in: ``adapter_config.json``'s ``petri_decoder_prefix`` (the
    exporter's provenance), else the common ``…layers`` head of the file's keys (``base_model.model.`` stripped); None
    when neither is readable."""
    d = Path(adapter_dir)
    try:
        cfg = json.loads((d / sa.ADAPTER_CONFIG_FILE).read_text(encoding="utf-8"))
        p = cfg.get("petri_decoder_prefix")
        if isinstance(p, str) and p:
            return p
    except (OSError, ValueError):
        pass
    try:
        heads = set()
        for k in safetensors_header(d):
            k = k[len("base_model.model."):] if k.startswith("base_model.model.") else k
            m = re.match(r"^(.*?\.layers)\.\d+\.", k)
            if m:
                heads.add(m.group(1))
        return heads.pop() if len(heads) == 1 else None
    except (OSError, ValueError):
        return None


def peft_state_key(file_key: str, adapter_name: str = "default") -> str:
    """Our PEFT file name -> the wrapped model's ``state_dict`` name (``.lora_A.weight`` -> ``.lora_A.default.weight``)."""
    return file_key.replace(".lora_A.weight", f".lora_A.{adapter_name}.weight").replace(".lora_B.weight", f".lora_B.{adapter_name}.weight")


def attachment_check(file_header: Mapping[str, Mapping[str, Any]], peft_state: Mapping[str, Sequence[int]]) -> Dict[str, Any]:
    """Every exported tensor must sit on the PEFT-wrapped reference under its OWN name and shape (``file_header`` =
    :func:`safetensors_header`; ``peft_state`` = ``{state_dict name: shape}``). ``pass`` only when all attach; the
    missing / mis-shaped names (first 20) say which — a prefix the tree does not carry shows as EVERY key missing."""
    missing: List[str] = []
    shape_mismatch: List[str] = []
    for k, v in file_header.items():
        sk = peft_state_key(k)
        if sk not in peft_state:
            missing.append(k)
        elif list(peft_state[sk]) != list(v["shape"]):
            shape_mismatch.append(k)
    n = len(file_header)
    return {"pass": n > 0 and not missing and not shape_mismatch, "n_file_tensors": n,
            "n_attached": n - len(missing) - len(shape_mismatch), "missing": missing[:20], "n_missing": len(missing),
            "shape_mismatch": shape_mismatch[:20]}


def _top5_key(hf: Mapping[str, Any]) -> Tuple[Tuple[int, float], ...]:
    return tuple((int(i), round(float(v), 9)) for i, v in (hf.get("top5") or []))


def hf_first_token_moved(base: Optional[Mapping[str, Any]], ref: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """How far the adapter moved the HF REFERENCE off its own base: the same shape as the engine's
    ``moved`` (:func:`first_token_delta`) — ``first_token_changed`` (the greedy token differs from the base's), ``max_abs_delta``
    (max |Δ logprob| over the top-5 tokens both distributions carry, None when they share none), ``common_tokens``,
    ``top_set_changed`` — computed from the two ``first_token`` summaries ({top1_id, top1_logprob, top5}) the reference phase
    records beside every adapter (``hf_base`` / ``hf``), so a stored record can be re-gated without a model. None when either
    side carries no top-5."""
    if not (isinstance(base, Mapping) and base.get("top5") and isinstance(ref, Mapping) and ref.get("top5")):
        return None
    bt = {int(i): float(v) for i, v in base["top5"]}
    rt = {int(i): float(v) for i, v in ref["top5"]}
    common = set(bt) & set(rt)
    deltas = [abs(bt[t] - rt[t]) for t in common]
    return {"max_abs_delta": (max(deltas) if deltas else None), "common_tokens": len(common),
            "top_set_changed": bool(set(bt) ^ set(rt)),
            "first_token_changed": (int(ref["top1_id"]) != int(base["top1_id"])
                                    if ref.get("top1_id") is not None and base.get("top1_id") is not None else None)}


#: the per-adapter record's CONTENT identity: sha256 of its ``adapter_model.safetensors`` (:func:`adapter_content_id`). The reference
#: phase records it on every outcome; the identity guard (:func:`identical_reference_groups`) and the one-ulp seed read it.
ADAPTER_CONTENT_KEY = "content_sha256"
#: a record written before the content key existed: its DIRECTORY stands in for its content in the identity guard (today's rule)
_CONTENT_FALLBACK_PREFIX = "dir:"


def adapter_content_id(adapter_dir: Path) -> Optional[str]:
    """The adapter's CONTENT identity — sha256 of ``<dir>/adapter_model.safetensors`` (``serving_attestation.sha256_file``); None when the
    file is unreadable. ``build_adapter_dirs`` exports each slot ONCE into ``adapters/_bank/<bank key>`` (kind, modules, rank, r_e, seed,
    scale — never the manifest or the cap) and links it into every manifest that serves it, so the declared-pair ladder's five S1cc dirs
    carry the SAME bytes: one adapter under five names, one subject for the reference (probe 979954)."""
    try:
        return sa.sha256_file(Path(adapter_dir) / sa.ADAPTER_WEIGHTS_FILE)
    except OSError:
        return None


def sensitivity_seed(content_sha256: Optional[str], prompt_index: int = 0, *, fallback: int = 0) -> int:
    """The one-ulp sensitivity's seed (:func:`one_ulp_sensitivity`), derived from the adapter's CONTENT so identical bytes draw identical
    noise under every name — the ladder's five twins of one adapter used to be seeded ``len(per)`` (their place in the phase's loop) and read
    v1 bounds of 0.03 on one dir and 0.12-0.15 on the others for the same bytes (979954). Prompt ``p`` of the v2 certificate takes
    ``prompt_index = p + 1`` (v1 is 0), so no two prompts of one adapter share a draw. ``fallback`` (the caller's loop position) stands in
    when the content is unknown."""
    if content_sha256:
        return (int(str(content_sha256)[:12], 16) + 101 * int(prompt_index)) % (2 ** 31)
    return (int(fallback) * 101 + int(prompt_index)) % (2 ** 31)


def _content_of(d: str, v: Mapping[str, Any]) -> str:
    c = v.get(ADAPTER_CONTENT_KEY) if isinstance(v, Mapping) else None
    return str(c) if c else _CONTENT_FALLBACK_PREFIX + str(d)


def identical_reference_groups(per: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Every group of adapter dirs whose HF first-token TOP-5 (ids and logprobs) are identical AND which spans >= 2 distinct adapter
    CONTENTS (``content_sha256``; a record without it counts by directory, as every record before the key did) — as
    ``{"dirs": [sorted dirs], "contents": {content: [sorted dirs]}}``. Two different adapters that leave the reference's distribution
    bit-identical were not applied: a harness error, never a mismatch. The same bytes under several directories are ONE
    adapter and must agree with themselves — never a group. Only adapters that were compared (``hf`` present) take part."""
    by_key: Dict[Tuple[Tuple[int, float], ...], Dict[str, List[str]]] = {}
    for d, v in per.items():
        hf = v.get("hf") if isinstance(v, Mapping) else None
        if isinstance(hf, Mapping) and hf.get("top5"):
            by_key.setdefault(_top5_key(hf), {}).setdefault(_content_of(str(d), v), []).append(str(d))
    out = []
    for contents in by_key.values():
        if len(contents) >= 2:
            out.append({"dirs": sorted(x for ds in contents.values() for x in ds), "contents": {c: sorted(ds) for c, ds in contents.items()}})
    return sorted(out, key=lambda g: g["dirs"])


def identical_reference_outputs(per: Mapping[str, Mapping[str, Any]]) -> List[List[str]]:
    """The dirs of every group :func:`identical_reference_groups` names (>= 2 distinct adapter CONTENTS with identical HF outputs)."""
    return [list(g["dirs"]) for g in identical_reference_groups(per)]


def _identical_group_reason(group: Mapping[str, Any]) -> str:
    """The harness refusal, naming the CONTENTS (sha256 prefixes; `content unrecorded` for a pre-key record) and the dirs of each."""
    def _short(x: str) -> str:
        p = Path(x)
        return f"{p.parent.name}/{p.name}"
    parts = []
    for c, ds in sorted(group["contents"].items()):
        label = "content unrecorded" if c.startswith(_CONTENT_FALLBACK_PREFIX) else f"content {c[:12]}"
        parts.append(f"{label}: " + ", ".join(_short(x) for x in ds[:3]) + ("…" if len(ds) > 3 else ""))
    return (f"the HF first-token distribution is IDENTICAL across {len(group['contents'])} distinct adapters "
            f"({len(group['dirs'])} adapter dir(s) by content — {'; '.join(parts)}) — a reference that does not move between different "
            "adapters never applied them")


def first_token_parity(hf: Mapping[str, Any], vllm: Mapping[str, Any], *, tol: float = REFERENCE_LOGPROB_TOL,
                       floor: Optional[float] = None, sensitivity: Optional[float] = None) -> Dict[str, Any]:
    """Assertion 4's rule in ONE place — rule (a): the served engine's greedy first token equals the HF+PEFT top-1
    and its logprob is within ``max(tol, PARITY_FLOOR_K x floor, PARITY_SENSITIVITY_M x sensitivity)`` — ``floor`` the
    MEASURED HF-vs-engine |d| on the BASE (no adapter on either side — on a large model it approaches the 4B-calibrated
    ``tol``), ``sensitivity`` this adapter's own one-ulp shift (:func:`one_ulp_sensitivity`). Every term is recorded beside
    the verdict (``tolerance`` = the effective bound, ``tolerance_terms``)."""
    v_id, v_lp = vllm.get("id"), vllm.get("logprob")
    equal = v_id is not None and int(v_id) == int(hf["top1_id"])
    delta = abs(float(v_lp) - float(hf["top1_logprob"])) if v_lp is not None else None
    terms = {"tol": float(tol),
             "floor_term": (PARITY_FLOOR_K * float(floor)) if floor is not None else None,
             "sensitivity_term": (PARITY_SENSITIVITY_M * float(sensitivity)) if sensitivity is not None else None}
    tol_eff = max(v for v in terms.values() if v is not None)
    ok = bool(equal and delta is not None and delta <= tol_eff)
    return {"status": REF_STATUS_PASS if ok else REF_STATUS_FAIL, "top1_equal": bool(equal), "logprob_abs_delta": delta,
            "tolerance": tol_eff, "tolerance_terms": terms}


def _first_token_effect_is_weak(moved: Optional[Mapping[str, Any]], moved_min: float) -> bool:
    """One side's adapter effect is WEAK: its greedy token is the base's and its top-5 moved by less than ``moved_min`` nats.
    Unknown (None, or no measured delta) is NOT weak — an unmeasured side never licenses suppressing a comparison."""
    return bool(moved is not None and moved.get("first_token_changed") is False and moved.get("max_abs_delta") is not None
                and float(moved["max_abs_delta"]) < float(moved_min))


def parity_gate(moved: Optional[Mapping[str, Any]], sensitivity: Optional[Mapping[str, Any]], *,
                hf_moved: Optional[Mapping[str, Any]] = None, moved_min: float = PARITY_MOVED_MIN) -> Optional[Tuple[str, str]]:
    """Whether this adapter's comparison can DECIDE anything: ``("parity_chaotic", why)`` when one bf16 ulp of
    relative noise on lora_B flips the reference's own greedy token (the distribution is chaos at this scale — an
    MLP-bearing set @ scale >= 1); ``("parity_weak", why)`` when BOTH the engine (``moved``) AND the HF reference (``hf_moved``,
    :func:`hf_first_token_moved`) left the greedy token unchanged and moved the top-5 by less than ``moved_min`` nats (too small
    an effect on either side to expose a wrongly applied adapter — wo @ 0.05); None = decidable.

    The weak gate used to read the ENGINE alone, so an engine that silently never applied the adapter (moved 0) was filed
    ``reference_undecidable`` while the HF reference plainly changed its greedy token and was stable across three one-ulp
    forwards — the exact mismatch the pass exists to catch. Weak now needs both sides weak; an unmeasured HF side
    (``hf_moved`` None) never suppresses, and an unambiguous HF-vs-engine disagreement flows to :func:`first_token_parity` and
    is a ``reference_mismatch``."""
    if sensitivity and int(sensitivity.get("flips") or 0) > 0:
        return ("parity_chaotic",
                f"one bf16 ulp of relative noise on lora_B flips the reference's own greedy token in {int(sensitivity['flips'])}/"
                f"{int(sensitivity.get('draws') or 0)} draws (max top-1 shift {float(sensitivity.get('max_shift') or 0):.3f} nats) — the "
                "first-token distribution is chaotic at this synthetic scale; parity is undecidable, lower the set's scale")
    if _first_token_effect_is_weak(moved, moved_min) and _first_token_effect_is_weak(hf_moved, moved_min):
        return ("parity_weak",
                f"the engine moved the first-token top-5 by only {float(moved['max_abs_delta']):.3f} nats and the HF reference by only "
                f"{float(hf_moved['max_abs_delta']):.3f} nats (both < {moved_min}), both leaving the greedy token unchanged — too weak an "
                "effect on either side to discriminate a wrongly applied adapter; raise the set's scale")
    return None


def judge_adapter_parity(hf: Mapping[str, Any], hf_base: Optional[Mapping[str, Any]], vllm: Mapping[str, Any], *,
                         moved: Optional[Mapping[str, Any]], sensitivity: Optional[Mapping[str, Any]], floor: Optional[float],
                         attachment: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """ONE adapter's parity judgement from its RAW comparison fields — the one place the gate and rule (a) are composed, shared
    by the reference phase (live, the fields just measured) and :func:`regate_reference_results` (a stored record re-read under
    the current rule): the HF-off-base movement, the gate (chaos / both-weak), else :func:`first_token_parity`. The record it
    returns carries every raw field it read (``hf``, ``hf_base``, ``vllm``, ``moved``, ``hf_moved``, ``sensitivity``,
    ``attachment``) so it can be re-judged again without a model."""
    hf_moved = hf_first_token_moved(hf_base, hf)
    summary: Dict[str, Any] = {"hf": hf, "hf_base": hf_base, "vllm": vllm, "moved": moved, "hf_moved": hf_moved, "sensitivity": sensitivity}
    if attachment is not None:
        summary["attachment"] = attachment
    gate = parity_gate(moved, sensitivity, hf_moved=hf_moved)
    if gate is not None:
        return {"status": REF_STATUS_UNDECIDABLE, "gate": gate[0], "reason": gate[1], **summary}
    return {**first_token_parity(hf, vllm, floor=floor, sensitivity=(sensitivity or {}).get("max_shift")), **summary}


def v2_base_floor(hf_base_list: Sequence[Mapping[str, Any]], vllm_base_list: Optional[Sequence[Optional[Mapping[str, Any]]]]
                  ) -> Tuple[Optional[float], List[Optional[float]], bool]:
    """ONE launch's eight-prompt base floor: ``(floor, |d_base,p| per prompt, complete)`` from the HF base
    first tokens and THAT launch's own engine base probes. ``complete`` iff the engine probed every prompt (same count, a
    logprob on each); an incomplete base yields floor None and the launch's v2 is undecidable — the tolerance never falls back
    to tol + sensitivity alone."""
    if not vllm_base_list or len(vllm_base_list) != len(hf_base_list):
        return None, [], False
    ds: List[Optional[float]] = []
    for hb, vb in zip(hf_base_list, vllm_base_list):
        lp = (vb or {}).get("logprob") if isinstance(vb, Mapping) else None
        ds.append(abs(float(lp) - float(hb["top1_logprob"])) if (lp is not None and hb.get("top1_logprob") is not None) else None)
    complete = all(d is not None for d in ds)
    return (max(ds) if complete and ds else None), ds, complete


def judge_v2_by_launch(hf_list: Sequence[Mapping[str, Any]], hf_base_list: Sequence[Mapping[str, Any]],
                       sens_list: Sequence[Optional[Mapping[str, Any]]], engine_by_launch: Mapping[str, Mapping[str, Any]], *,
                       prompts_sha: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """ONE adapter's v2 certificate per LAUNCH that served it: the HF side (reference first tokens, base first
    tokens, one-ulp sensitivities — measured once) judged against EACH launch's own engine side ``{adapter, base, moved, sha}``; the
    floor is that launch's own base probes (:func:`v2_base_floor`). A launch whose probes are not under this prompt set, that
    probed fewer prompts than the certificate has, or whose base probes are incomplete is UNDECIDABLE for that launch alone —
    never judged with another launch's floor."""
    out: Dict[str, Dict[str, Any]] = {}
    n = len(hf_list)
    for lid, eng in sorted(engine_by_launch.items()):
        vl = list(eng.get("adapter") or [])
        if prompts_sha is not None and eng.get("sha") != prompts_sha:
            out[lid] = {"version": PARITY_CERTIFICATE_VERSION, "status": REF_STATUS_UNDECIDABLE, "gate": "parity_base_incomplete",
                        "reason": f"launch {lid} probed prompt set {eng.get('sha')!r}, not the certificate's {prompts_sha!r}", "launch_id": lid,
                        "per_prompt": [], "base_complete": False, "prompts_sha": prompts_sha}
            continue
        if len(vl) != n:
            out[lid] = {"version": PARITY_CERTIFICATE_VERSION, "status": REF_STATUS_UNDECIDABLE, "gate": "parity_base_incomplete",
                        "reason": f"launch {lid} probed {len(vl)}/{n} certificate prompts for this adapter", "launch_id": lid,
                        "per_prompt": [], "base_complete": False, "prompts_sha": prompts_sha}
            continue
        fl, _ds, complete = v2_base_floor(hf_base_list, eng.get("base"))
        out[lid] = judge_adapter_parity_v2(hf_list, hf_base_list, vl, moved_list=list(eng.get("moved") or [None] * n),
                                           sens_list=list(sens_list), floor=fl, prompts_sha=prompts_sha, base_complete=complete,
                                           vllm_base_list=eng.get("base"), launch_id=lid)
    return out


def judge_adapter_parity_v2(hf_list: Sequence[Mapping[str, Any]], hf_base_list: Sequence[Mapping[str, Any]],
                            vllm_list: Sequence[Mapping[str, Any]], *, moved_list: Sequence[Optional[Mapping[str, Any]]],
                            sens_list: Sequence[Optional[Mapping[str, Any]]], floor: Optional[float],
                            prompts_sha: Optional[str] = None, base_complete: bool = True,
                            vllm_base_list: Optional[Sequence[Optional[Mapping[str, Any]]]] = None,
                            launch_id: Optional[str] = None) -> Dict[str, Any]:
    """PARITY CERTIFICATE v2 for ONE adapter, pure over the per-prompt raw fields: ``hf_list[p]`` / ``hf_base_list[p]``
    the HF first-token summaries with / without the adapter, ``vllm_list[p]`` the engine's first token (``{id, logprob}``),
    ``moved_list[p]`` how far the ENGINE moved the top-5 off its base, ``sens_list[p]`` the adapter's one-ulp sensitivity on
    that prompt, ``floor`` = max_p of the HF-vs-engine BASE |d_p| on the same standup. Per prompt: ``d`` = |HF top-1 logprob −
    engine top-1 logprob|, ``top1_equal``, ``engine_moved`` vs ``hf_moved`` (a systematic effect-size gap across prompts is a
    partially applied adapter — reported, never judged here). Gate: ``parity_chaotic`` when any prompt's one-ulp noise flips the
    reference's greedy token; ``parity_weak`` when EVERY prompt is weak on BOTH sides. Rule: PASS iff every prompt's greedy
    token agrees and ``max_p d_p <= tol = max(REFERENCE_LOGPROB_TOL, PARITY_FLOOR_K x floor, PARITY_SENSITIVITY_M x max_p
    sensitivity)``. Every raw field is kept so the record re-judges without a model (:func:`regate_adapter_record`)."""
    n = len(hf_list)
    if not (n and len(hf_base_list) == len(vllm_list) == n):
        raise ValueError(f"v2 parity needs one HF / HF-base / engine first token per prompt (got {n} / {len(hf_base_list)} / {len(vllm_list)})")
    per: List[Dict[str, Any]] = []
    flips = 0
    weak_everywhere = True
    for p_i in range(n):
        hf, hb, vl = hf_list[p_i], hf_base_list[p_i], vllm_list[p_i]
        mv = moved_list[p_i] if p_i < len(moved_list) else None
        se = sens_list[p_i] if p_i < len(sens_list) else None
        hf_mv = hf_first_token_moved(hb, hf)
        v_id, v_lp = vl.get("id"), vl.get("logprob")
        equal = v_id is not None and hf.get("top1_id") is not None and int(v_id) == int(hf["top1_id"])
        d = abs(float(v_lp) - float(hf["top1_logprob"])) if (v_lp is not None and hf.get("top1_logprob") is not None) else None
        flips += int((se or {}).get("flips") or 0)
        if not (_first_token_effect_is_weak(mv, PARITY_MOVED_MIN) and _first_token_effect_is_weak(hf_mv, PARITY_MOVED_MIN)):
            weak_everywhere = False
        per.append({"prompt": p_i, "top1_equal": bool(equal), "d": d, "hf": hf, "hf_base": hb, "vllm": vl, "moved": mv, "hf_moved": hf_mv,
                    "engine_moved": ((mv or {}).get("max_abs_delta") if mv else None),
                    "hf_moved_max": (hf_mv or {}).get("max_abs_delta") if hf_mv else None,
                    "sensitivity": se})
    sens_max = max((float((se or {}).get("max_shift") or 0.0) for se in sens_list), default=0.0) if sens_list else None
    ds = [r["d"] for r in per if r["d"] is not None]
    terms = {"tol": float(REFERENCE_LOGPROB_TOL),
             "floor_term": (PARITY_FLOOR_K * float(floor)) if floor is not None else None,
             "sensitivity_term": (PARITY_SENSITIVITY_M * float(sens_max)) if sens_max is not None else None}
    tol_eff = max(v for v in terms.values() if v is not None)
    rec: Dict[str, Any] = {"version": PARITY_CERTIFICATE_VERSION, "n_prompts": n, "prompts_sha": prompts_sha, "per_prompt": per,
                           "max_abs_delta": (max(ds) if ds else None), "argmax_prompt": (max(range(len(per)), key=lambda i: (per[i]["d"] or -1)) if ds else None),
                           "all_top1_equal": all(r["top1_equal"] for r in per), "floor": floor, "sensitivity_max": sens_max,
                           "tolerance": tol_eff, "tolerance_terms": terms, "launch_id": launch_id,
                           # the launch's OWN engine base probes the floor was measured against (kept for the model-free re-gate)
                           "vllm_base": ([dict(b) if isinstance(b, Mapping) else None for b in vllm_base_list] if vllm_base_list is not None else None),
                           "base_complete": bool(base_complete and floor is not None),
                           "engine_moved_by_prompt": [r["engine_moved"] for r in per], "hf_moved_by_prompt": [r["hf_moved_max"] for r in per]}
    if not rec["base_complete"]:
        # a floor that could not be measured on THIS launch (a base probe failed, a prompt missing, another
        # launch's bases) never widens or narrows the bound — the launch's v2 is undecidable, by name
        rec.update(status=REF_STATUS_UNDECIDABLE, gate="parity_base_incomplete",
                   reason=f"the engine's eight-prompt BASE probes of this launch are incomplete ({sum(1 for b in (vllm_base_list or []) if isinstance(b, Mapping) and b.get('logprob') is not None)}/{n} prompts) — "
                          "the v2 floor cannot be measured on the standup that served the adapter; the certificate is undecidable here")
        return rec
    if flips > 0:
        rec.update(status=REF_STATUS_UNDECIDABLE, gate="parity_chaotic",
                   reason=f"one bf16 ulp of relative noise on lora_B flips the reference's greedy token on {flips} draw(s) across {n} prompts")
        return rec
    if weak_everywhere:
        rec.update(status=REF_STATUS_UNDECIDABLE, gate="parity_weak",
                   reason=f"on every one of {n} prompts both the engine and the HF reference moved the top-5 by < {PARITY_MOVED_MIN} nats with the greedy token unchanged")
        return rec
    ok = bool(rec["all_top1_equal"] and ds and len(ds) == n and rec["max_abs_delta"] <= tol_eff)
    rec["status"] = REF_STATUS_PASS if ok else REF_STATUS_FAIL
    return rec


def compose_parity_record(v1: Mapping[str, Any], v2: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """ONE adapter's stored parity record: the v1 judgement's fields at the top level (as in every version-1 record), a ``v1``
    summary, the ``v2`` record when the launch probed the certificate's prompts, ``certificate_version`` (2 iff v2 exists) and
    the EFFECTIVE ``status`` / ``gate`` / ``reason`` — v2's where it exists, else v1's (the menu rule: v2 where it exists, else v1)."""
    out = dict(v1)
    if v2 is None:
        # a v1-only judgement keeps EXACTLY the record shape every reader (and every stored record) had before v2: version 1 is
        # implied by the absence of `certificate_version` / `v2` (readers use .get(..., 1))
        out.pop("v2", None); out.pop("v1", None); out.pop("certificate_version", None)
        return out
    out["v1"] = {k: v1.get(k) for k in ("status", "gate", "reason", "top1_equal", "logprob_abs_delta", "tolerance", "tolerance_terms")}
    out["v2"] = dict(v2)
    out["certificate_version"] = PARITY_CERTIFICATE_VERSION
    out["status"] = v2["status"]
    if v2.get("gate"):
        out["gate"], out["reason"] = v2["gate"], v2.get("reason")
    else:
        out.pop("gate", None)
        out["reason"] = None
    return out


def one_ulp_sensitivity(pm: Any, first_token: Callable[[Any], Dict[str, Any]], ref: Mapping[str, Any], *,
                        draws: int = PARITY_SENSITIVITY_DRAWS, rel: float = PARITY_ULP_REL, seed: int = 0) -> Dict[str, Any]:
    """The adapter's own numeric sensitivity, measured on the wrapped reference: every ``lora_B`` scaled by
    ``(1 + rel x N(0, 1))`` (one bf16 ulp of relative noise), ``draws`` forwards, the top-1 logprob shift and top-1 flips
    against ``ref``; the parameters are restored bit for bit afterwards (also on an exception)."""
    import torch
    Bs = [p for n, p in pm.named_parameters() if ".lora_B." in n]
    saved = [p.detach().clone() for p in Bs]
    gen = torch.Generator(device=(saved[0].device if saved else "cpu")).manual_seed(int(seed))
    shifts: List[float] = []
    flips = 0
    try:
        for _ in range(int(draws)):
            with torch.no_grad():
                for p, s in zip(Bs, saved):
                    noise = torch.randn(s.shape, generator=gen, device=s.device, dtype=torch.float32)
                    p.copy_((s.float() * (1.0 + float(rel) * noise)).to(s.dtype))
            k = first_token(pm)
            shifts.append(abs(float(k["top1_logprob"]) - float(ref["top1_logprob"])))
            flips += int(int(k["top1_id"]) != int(ref["top1_id"]))
    finally:
        with torch.no_grad():
            for p, s in zip(Bs, saved):
                p.copy_(s)
    return {"draws": int(draws), "rel": float(rel), "shifts": shifts, "flips": flips,
            "max_shift": (max(shifts) if shifts else None), "n_lora_B": len(Bs)}


def peft_reference_layout(model: Any, out_dir: Path, *, modules: Sequence[str], rank: int,
                          per_expert_rank: Optional[int] = None, wraps_experts: bool = False) -> Dict[str, Any]:
    """PEFT 0.20.0's own ``save_pretrained`` layout for the same architecture
    and module set (``target_parameters=["mlp.experts.gate_up_proj","mlp.experts.down_proj"]``
    + ``rank_pattern``/``alpha_pattern`` for the experts when the class wraps
    them) — GPU/HF-model only; records ``peft_unavailable`` when ``import
    peft`` fails."""
    try:
        import peft  # noqa: F401
        from peft import LoraConfig, get_peft_model
    except ImportError as e:
        return {"status": "peft_unavailable", "reason": f"import peft failed: {e}"}
    kw: Dict[str, Any] = dict(r=int(rank), lora_alpha=int(rank), lora_dropout=0.0, bias="none",
                              target_modules=list(modules), init_lora_weights=False)
    if wraps_experts:
        from sisteer.capeval.serving import lora_export as lx
        kw["target_parameters"] = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]
        if per_expert_rank is not None:
            # the exporter's OWN pattern spelling (lora_export._experts_rank_regex), so the
            # reference layout is PEFT's build of the exporter's config
            kw["rank_pattern"] = {lx._experts_rank_regex("mlp.experts"): int(per_expert_rank)}
            kw["alpha_pattern"] = {lx._experts_rank_regex("mlp.experts"): int(per_expert_rank)}
    pm = get_peft_model(model, LoraConfig(**kw))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pm.save_pretrained(str(out_dir))
    base = pm.unload()
    del pm
    return {"status": "written", "dir": str(out_dir), "peft_version": getattr(peft, "__version__", None),
            "config": kw, "_model": base}


# --------------------------------------------------------------------------
# GPU / engine seam
# --------------------------------------------------------------------------

def chat_probe_body(model_name: str, messages: Sequence[Mapping[str, str]], *, max_tokens: int) -> Dict[str, Any]:
    """One greedy request with token ids and first-token top-5 logprobs
    (``multilora_cert.probe_body``'s fields, vLLM 0.25.1 per-request)."""
    return {"model": model_name, "messages": [dict(m) for m in messages], "temperature": 0.0, "seed": 0,
            "max_tokens": int(max_tokens), "return_token_ids": True, "logprobs": True, "top_logprobs": 5,
            "return_tokens_as_token_ids": True}


def parse_logprob_rows(resp: Mapping[str, Any]) -> Dict[str, Any]:
    """The comparable content of a chat completion: token ids, and per
    position the chosen logprob + the top-5 ``[token, logprob]`` list."""
    choices = resp.get("choices") or []
    if not choices:
        raise RuntimeError("no choices in the response")
    ch = choices[0]
    token_ids = ch.get("token_ids")
    content = ((ch.get("logprobs") or {}).get("content")) or []
    rows = [{"token": c.get("token"), "logprob": c.get("logprob"),
             "top": [[t.get("token"), t.get("logprob")] for t in (c.get("top_logprobs") or [])]} for c in content]
    text = (ch.get("message") or {}).get("content")
    return {"token_ids": list(token_ids) if isinstance(token_ids, list) else None, "rows": rows,
            "finish_reason": ch.get("finish_reason"), "text": text}


def first_token_delta(base: Mapping[str, Any], other: Mapping[str, Any]) -> Dict[str, Any]:
    """max |Δ logprob| over the first position's common top tokens (and the
    chosen token), plus whether the first token id changed."""
    if not base.get("rows") or not other.get("rows"):
        return {"max_abs_delta": None, "common_tokens": 0, "first_token_changed": None}
    b, o = base["rows"][0], other["rows"][0]
    bt = {t: lp for t, lp in b["top"] if t is not None and lp is not None}
    ot = {t: lp for t, lp in o["top"] if t is not None and lp is not None}
    if b.get("token") is not None and b.get("logprob") is not None:
        bt.setdefault(b["token"], b["logprob"])
    if o.get("token") is not None and o.get("logprob") is not None:
        ot.setdefault(o["token"], o["logprob"])
    common = set(bt) & set(ot)
    deltas = [abs(float(bt[t]) - float(ot[t])) for t in common]
    changed = None
    if base.get("token_ids") and other.get("token_ids"):
        changed = base["token_ids"][0] != other["token_ids"][0]
    missing = (set(bt) ^ set(ot))
    # a token that fell out of / entered the top-5 is at least as large a move
    # as the smallest common delta; report it separately
    return {"max_abs_delta": (max(deltas) if deltas else None), "common_tokens": len(common),
            "top_set_changed": bool(missing), "first_token_changed": changed}


def bit_identical(base: Mapping[str, Any], other: Mapping[str, Any]) -> bool:
    return (base.get("token_ids") == other.get("token_ids")
            and json.dumps(base.get("rows"), sort_keys=True) == json.dumps(other.get("rows"), sort_keys=True))


@dataclass
class Deps:
    """Every external effect of ``run``. ``production()`` binds the real
    ones lazily; tests hand in fakes shaped like the real return values.

    The three GPU-touching effects take ``devices=`` (keyword): the launch's
    pinned device subset — a tuple of Slurm's ``CUDA_VISIBLE_DEVICES`` tokens —
    or None for an unpinned single standup. ``stand_up`` hands it to the
    child engine's environment alone; ``nvidia_smi`` / ``wait_gpu_free``
    read exactly those devices, so concurrent launches never watch each
    other's GPU."""

    stand_up: Callable[..., Tuple[Any, Any]]          # (launch, proc); devices= the pinned subset
    shut_down: Callable[[Any, Any], None]
    chat: Callable[..., Dict[str, Any]]                 # (launch, body) -> parsed json
    list_models: Callable[[Any], List[str]]
    worker_report: Callable[[Any, Path], Dict[str, Any]]
    nvidia_smi: Callable[..., Dict[str, Any]]           # (devices=) -> residency of the pinned devices
    throughput: Callable[..., List[Dict[str, Any]]]
    read_log: Callable[[Path], str]
    wait_gpu_free: Callable[..., Dict[str, Any]]        # (devices=) -> the drain wait on the pinned devices
    reference: Callable[..., Dict[str, Any]]            # the HF+PEFT phase over pending adapters
    now: Callable[[], float] = time.time

    @classmethod
    def production(cls) -> "Deps":
        return cls(stand_up=_production_stand_up, shut_down=_production_shut_down, chat=_production_chat,
                   list_models=_production_list_models, worker_report=_production_worker_report,
                   nvidia_smi=_production_nvidia_smi, throughput=_production_throughput,
                   read_log=_read_log_tail, wait_gpu_free=_production_wait_gpu_free,
                   reference=_production_reference_phase)


def _read_log_tail(path: Path, n: int = 2_000_000) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    with open(p, "rb") as fh:
        fh.seek(0, os.SEEK_END)
        fh.seek(max(0, fh.tell() - n))
        return fh.read().decode("utf-8", errors="replace")


def resolve_revision(model_id: str, revision: str) -> str:
    """A 40-hex revision, resolved from the HF cache's ``refs/<rev>`` when a
    branch name is given (offline; never the network)."""
    if re.fullmatch(r"[0-9a-f]{40}", revision or ""):
        return revision
    hf_home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    ref = Path(hf_home) / "hub" / f"models--{model_id.replace('/', '--')}" / "refs" / (revision or "main")
    if ref.is_file():
        sha = ref.read_text(encoding="utf-8").strip()
        if re.fullmatch(r"[0-9a-f]{40}", sha):
            return sha
    raise SystemExit(f"{TAG} revision {revision!r} is not a 40-hex sha and {ref} does not resolve it")


@functools.lru_cache(maxsize=4)
def _chat_template_bytes(model_id: str, revision: str) -> bytes:
    """The served tokenizer's chat template, read from the local cache ONCE per
    (model, revision) — every launch of a run materialises the same bytes."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision, local_files_only=True)
    tpl = tok.chat_template
    if isinstance(tpl, dict):
        tpl = tpl.get("default") or next(iter(tpl.values()))
    if not isinstance(tpl, str) or not tpl:
        raise RuntimeError(f"{model_id} tokenizer has no chat_template")
    return tpl.encode("utf-8")


@functools.lru_cache(maxsize=1)
def _tree_sha() -> Optional[str]:
    """``git rev-parse HEAD`` of this tree (``run_identity.tree_sha``; None off a checkout)."""
    from sisteer.capeval.serving import run_identity as ri
    return ri.tree_sha(REPO)


class LaunchRefused(RuntimeError):
    """A compose-time refusal the probe records as ``refused:<msg>``."""


_PORT_LOCK = threading.Lock()
_PORTS_HANDED_OUT: Dict[int, float] = {}
#: how long a port handed to one standup stays off the table for the others (a standup is up in minutes; its engine
#: binds the port long before then, and the bind check in launch_server is the second belt)
_PORT_HOLD_S = 1800.0


def _pick_port() -> int:
    """A free loopback API port for ONE standup, distinct from every port handed to another concurrent standup within
    :data:`_PORT_HOLD_S`: ``native_standup.free_port`` (bind 0, read back) under one lock plus a recently-handed-out
    guard, so two launches composed in the same instant can never be given the same port. ``launch_server``'s own
    already-listening refusal stays as the second belt."""
    from sisteer.capeval.serving import native_standup as ns
    with _PORT_LOCK:
        now = time.time()
        for _ in range(256):
            port = int(ns.free_port())
            if now - _PORTS_HANDED_OUT.get(port, -1e12) > _PORT_HOLD_S:
                _PORTS_HANDED_OUT[port] = now
                return port
        raise RuntimeError(f"{TAG} no free loopback port distinct from the {len(_PORTS_HANDED_OUT)} recently handed out")


def _production_stand_up(*, model_id: str, revision: str, adapter_dirs: Mapping[str, Path], manifest: sm.Manifest,
                         L: int, S: int, out_dir: Path, gpu_mem_util: float, ready_timeout_s: float,
                         tp: int, extra_flags: Sequence[str] = (),
                         devices: Optional[Sequence[str]] = None) -> Tuple[Any, Any]:
    """Compose + launch ONE lora-native engine for ``manifest`` at (L, S):
    ``vllm_server.build_server_launch(lora_native=True, …)`` (the composer
    fits the cap, derives the class, checks the kernel index ceiling) with the
    front-owned extras (memory utilization, text-only multimodal pin, the
    TP-2 pair; ``--moe-backend triton`` is the COMPOSER's when experts are
    wrapped), then
    ``launch_server`` — the readiness attestation runs inside it. ``devices``
    (a strict subset of Slurm's ``CUDA_VISIBLE_DEVICES`` tokens, from
    :func:`device_slots`) is pinned in the CHILD engine's environment alone
    through ``launch_server``'s ``extra_env`` — this process's own
    environment is never changed (cluster hard rule 3)."""
    from sisteer.capeval.serving import native_standup as ns
    from sisteer.capeval.serving import vllm_server as vs
    out_dir = Path(out_dir)
    if devices is not None:
        devices = tuple(str(d) for d in devices)
        if len(devices) != int(tp):
            raise LaunchRefused(f"devices_tp_mismatch: {len(devices)} pinned device(s) {list(devices)} for tp={tp}")
        visible = visible_devices()
        if visible is not None and not set(devices) <= set(visible):
            raise LaunchRefused(f"devices_not_visible: {list(devices)} is not a subset of {VISIBLE_DEVICES_ENV}={','.join(visible)}")
    tpath, thash = ns.write_chat_template(out_dir, _chat_template_bytes(model_id, revision))
    spec = vs.launch_only_spec(model_id, revision, audit="serving-probe", chat_template_hash=thash,
                               serving_packages={"vllm": "0.25.1+cu129"}, tool_parser="hermes",
                               impl_commit=_tree_sha(), fallbacks=False)
    launch = vs.build_server_launch(spec, port=_pick_port(), lora_native=True,
                                    lora_modules={k: Path(v) for k, v in adapter_dirs.items()},
                                    chat_template_path=tpath, max_model_len=int(L), max_num_seqs=int(S),
                                    prefix_caching=False, tool_calls=False)
    # the composed engine must be THIS manifest's engine — a composer that has
    # not learnt the class (the `experts` inference) or fits another cap is a
    # loud refusal, never a mislabelled verdict
    if tuple(sorted(launch.lora_target_modules)) != tuple(sorted(manifest.target_class)):
        raise LaunchRefused(f"composer_target_class_mismatch: composed {sorted(launch.lora_target_modules)} != "
                            f"manifest {sorted(manifest.target_class)} (vllm_server._adapted_module_class has "
                            "not learnt this class — contract §3 / agent D1)")
    if int(launch.max_lora_rank or 0) != int(manifest.max_lora_rank):
        raise LaunchRefused(f"composer_cap_mismatch: fitted --max-lora-rank {launch.max_lora_rank} != planned "
                            f"{manifest.max_lora_rank}")
    if int(launch.max_loras or 0) != int(manifest.max_loras):
        raise LaunchRefused(f"composer_slots_mismatch: --max-loras {launch.max_loras} != {manifest.max_loras}")
    # the routed-expert buffer rank the composer read off the adapters' own rank_pattern must be the one the
    # manifest planned (the bytes law and the admission priced it) — None on both when the class wraps no expert
    if (getattr(launch, "moe_lora_rank", None) is None) != (manifest.moe_lora_rank is None) or (
            manifest.moe_lora_rank is not None and int(launch.moe_lora_rank) != int(manifest.moe_lora_rank)):
        raise LaunchRefused(f"composer_moe_rank_mismatch: composed {vs.MOE_LORA_RANK_ENV}={getattr(launch, 'moe_lora_rank', None)} != "
                            f"planned moe_lora_rank {manifest.moe_lora_rank} (the adapters' rank_pattern and the manifest disagree)")
    extras: List[str] = ["--gpu-memory-utilization", str(gpu_mem_util)]
    if vs.multimodal_target(model_id, revision) and vs.text_only_mm_in_cmd(launch.cmd) is None:
        extras += list(vs.TEXT_ONLY_MM_FLAG_TOKENS)
    if int(tp) > 1:
        extras += ["--tensor-parallel-size", str(int(tp)), "--fully-sharded-loras"]
    extras += [str(a) for a in extra_flags]
    launch = dataclasses.replace(launch, cmd=tuple(launch.cmd) + tuple(extras))
    # the pinned subset rides the child's environment ONLY (compose_server_env applies extra_env last): the record
    # names it beside the port so a launch dir says which GPU(s) its engine ran on
    extra_env = {VISIBLE_DEVICES_ENV: ",".join(devices)} if devices is not None else None
    (out_dir / "launch_cmd.json").write_text(json.dumps({"cmd": list(launch.cmd), "env": dict(launch.env),
                                                         "port": launch.port, "model": launch.model,
                                                         "devices": (list(devices) if devices is not None else None)},
                                                        indent=2), encoding="utf-8")
    proc = vs.launch_server(launch, log_path=out_dir / ns.SERVER_LOG_FILE, ready_timeout_s=float(ready_timeout_s),
                            extra_env=extra_env)
    return launch, proc


def _production_shut_down(launch: Any, proc: Any) -> None:
    from sisteer.capeval.serving import vllm_server as vs
    try:
        vs._shutdown_child(proc)
    except Exception:  # noqa: BLE001
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass


def _http_json(url: str, *, headers: Mapping[str, str], body: Optional[Mapping[str, Any]] = None,
               timeout_s: float = 600.0) -> Any:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **dict(headers)},
                                 method="POST" if body is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _production_chat(launch: Any, body: Mapping[str, Any], *, timeout_s: float = 600.0) -> Dict[str, Any]:
    return _http_json(f"{launch.base_url}/v1/chat/completions", headers=launch.auth_headers, body=body,
                      timeout_s=timeout_s)


def _production_list_models(launch: Any) -> List[str]:
    body = _http_json(f"{launch.base_url}/v1/models", headers=launch.auth_headers)
    return [str(m.get("id")) for m in (body.get("data") or [])]


def _production_worker_report(launch: Any, out_dir: Path) -> Dict[str, Any]:
    """The first worker's resolved report: from ``serving_attestation.json``
    (written by ``launch_server`` at readiness) and a live
    ``sisteer_scheduler_config`` RPC (the buffer-bytes key rides it)."""
    from sisteer.capeval.serving import vllm_server as vs
    out: Dict[str, Any] = {"source": []}
    att = Path(out_dir) / vs.SERVING_ATTESTATION_FILE
    if att.is_file():
        try:
            doc = json.loads(att.read_text(encoding="utf-8"))
            reps = ((doc.get("scheduler_geometry") or {}).get("worker_reports")
                    or ((doc.get("lora_native") or {}).get("scheduler_geometry") or {}).get("worker_reports"))
            if reps:
                out.update(reps[0])
                out["source"].append("serving_attestation.json")
        except Exception as e:  # noqa: BLE001
            out["attestation_read_error"] = str(e)
    try:
        payload = vs._post_collective_rpc(launch, vs.SCHEDULER_RPC_METHOD, [], timeout_s=120.0,
                                          what="serving-probe worker report")
        results = payload.get("results") if isinstance(payload, Mapping) else payload
        if isinstance(results, list) and results and isinstance(results[0], Mapping):
            out.update(results[0])
            out["source"].append("collective_rpc")
    except Exception as e:  # noqa: BLE001
        out["rpc_error"] = str(e)
    return out


def _production_nvidia_smi(devices: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Residency of the launch's device(s): ``devices`` (the pinned subset) when given, else the first token of this
    process's ``CUDA_VISIBLE_DEVICES`` (the single-standup default), else every GPU nvidia-smi sees. ``used_mib`` /
    ``total_mib`` sum over the queried devices; ``per_device`` keeps each one."""
    if devices:
        idx = ",".join(str(d) for d in devices)
    else:
        idx = ((visible_devices() or [""])[0]).strip()
    cmd = ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"]
    if idx:
        cmd += ["-i", idx]
    try:
        lines = [ln for ln in subprocess.check_output(cmd, text=True, timeout=20).strip().splitlines() if ln.strip()]
        per = []
        for ln in lines:
            used, total = [int(x.strip()) for x in ln.split(",")[:2]]
            per.append({"used_mib": used, "total_mib": total})
        if not per:
            return {"error": "nvidia-smi printed no device line", "devices": (list(devices) if devices else None)}
        return {"used_mib": sum(p["used_mib"] for p in per), "total_mib": sum(p["total_mib"] for p in per),
                "per_device": per, "devices": (list(devices) if devices else None)}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e), "devices": (list(devices) if devices else None)}


def _production_wait_gpu_free(threshold_mib: int = 6000, timeout_s: float = 240.0,
                              devices: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """The drain wait between standups ON THE LAUNCH'S OWN DEVICES: every pinned device must fall under
    ``threshold_mib`` (a neighbour's live engine on another GPU of the allocation is never waited on)."""
    def _max_used() -> Optional[int]:
        rep = _production_nvidia_smi(devices)
        per = rep.get("per_device")
        if per:
            return max(int(p["used_mib"]) for p in per)
        return rep.get("used_mib")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        used = _max_used()
        if used is None or used < threshold_mib:
            return {"waited_s": time.time() - t0, "used_mib": used, "drained": True,
                    "devices": (list(devices) if devices else None)}
        time.sleep(3)
    return {"waited_s": time.time() - t0, "used_mib": _max_used(), "drained": False,
            "devices": (list(devices) if devices else None)}


async def _closed_loop(url: str, headers: Mapping[str, str], model_name: str, *, B: int, max_tokens: int,
                       window_s: float, warmup_s: float) -> Dict[str, Any]:
    import aiohttp
    t_end = time.time() + warmup_s + window_s
    t_measure = time.time() + warmup_s
    tokens = 0
    requests_done = 0
    errors = 0
    prompts = list(DECODE_PROMPTS)

    async def worker(session: Any, i: int) -> None:
        nonlocal tokens, requests_done, errors
        k = i
        while time.time() < t_end:
            body = {"model": model_name, "messages": [{"role": "user", "content": prompts[k % len(prompts)]}],
                    "max_tokens": int(max_tokens), "min_tokens": int(max_tokens), "ignore_eos": True,
                    "temperature": 0.0, "seed": 0}
            k += B
            try:
                async with session.post(url, json=body, headers=dict(headers),
                                        timeout=aiohttp.ClientTimeout(total=600)) as r:
                    if r.status != 200:
                        errors += 1
                        await r.text()
                        continue
                    j = await r.json()
            except Exception:  # noqa: BLE001
                errors += 1
                continue
            if time.time() >= t_measure:
                tokens += int(((j.get("usage") or {}).get("completion_tokens")) or 0)
                requests_done += 1

    async with aiohttp.ClientSession() as session:
        await asyncio.gather(*(worker(session, i) for i in range(B)))
    return {"B": int(B), "client_tokens": tokens, "requests": requests_done, "errors": errors,
            "window_s": window_s, "client_tok_s": tokens / window_s if window_s else None}


def _read_metrics_counter(launch: Any, name: str) -> Optional[float]:
    """One ``/metrics`` read, summed per metric name (``closed_loop.read_metrics``);
    None when unreachable or the counter is absent."""
    from sisteer.capeval.serving import closed_loop as cl
    return cl.read_metrics(f"{launch.base_url}/metrics", launch.auth_headers)[1].get(name)


def _production_throughput(launch: Any, *, concurrencies: Sequence[int], max_tokens: int = 64,
                           window_s: float = 30.0, warmup_s: float = 10.0) -> List[Dict[str, Any]]:
    url = f"{launch.base_url}/v1/chat/completions"
    points = []
    from sisteer.capeval.serving.closed_loop import GENERATION_COUNTER_METRIC
    for B in concurrencies:
        c0 = _read_metrics_counter(launch, GENERATION_COUNTER_METRIC)
        t0 = time.time()
        stats = asyncio.run(_closed_loop(url, launch.auth_headers, launch.model, B=int(B), max_tokens=max_tokens,
                                         window_s=window_s, warmup_s=warmup_s))
        elapsed = time.time() - t0
        c1 = _read_metrics_counter(launch, GENERATION_COUNTER_METRIC)
        server_tok_s = ((c1 - c0) / elapsed) if (c0 is not None and c1 is not None and elapsed) else None
        points.append({**stats, "requested_concurrency": int(B), "engine_max_num_seqs": launch.max_num_seqs,
                       "server_tok_s": server_tok_s,
                       "throughput_source": "server_metrics" if server_tok_s is not None else "client_fallback",
                       "elapsed_s": elapsed})
        log(f"  throughput B={B}: server {server_tok_s and round(server_tok_s, 1)} tok/s, "
            f"client {stats['client_tok_s'] and round(stats['client_tok_s'], 1)} tok/s, errors {stats['errors']}")
    return points


def reference_prompt_text(tok: Any) -> str:
    """The HF + PEFT reference prompt (assertion 4): :data:`PROBE_PROMPT`
    rendered through the fit side's ``apply_chat_template_folding`` under the
    tokenizer's PINNED template kwargs (``load_model_and_tokenizer`` sets
    ``tok._sisteer_template_kwargs = {"enable_thinking": False}`` on a
    template that carries the switch), so the HF first-token distribution is
    for the SAME assistant header the native server renders — its
    ``--default-chat-template-kwargs {"enable_thinking": false}`` pin
    (``vllm_server.THINKING_OFF_TEMPLATE_KWARGS``; Qwen's template default
    is thinking ON). A thinking-capable template on a tokenizer that carries
    no pin REFUSES: the two renders would differ and every adapter, correct
    or not, would fail parity and be excluded as ``reference_mismatch``
."""
    from sisteer.models import apply_chat_template_folding
    tpl = getattr(tok, "chat_template", None)
    if isinstance(tpl, Mapping):
        tpl = tpl.get("default") or next(iter(tpl.values()), "")
    marker = getattr(tok, "_sisteer_template_kwargs", None) or {}
    if isinstance(tpl, str) and "enable_thinking" in tpl and marker.get("enable_thinking") is not False:
        raise RuntimeError(
            f"{TAG} the served chat template carries an enable_thinking switch but the tokenizer pins no "
            f"thinking-off render (_sisteer_template_kwargs={marker!r}); the native server pins "
            "enable_thinking=false, so an unpinned HF reference would score a different assistant header "
            "— refusing to run the parity check on it")
    return apply_chat_template_folding(tok, [dict(m) for m in PROBE_PROMPT], add_generation_prompt=True)


def reference_prompt_texts(tok: Any, prompts: Sequence[Sequence[Mapping[str, str]]]) -> List[str]:
    """Every v2 prompt rendered through the same pinned (thinking-off) render as :func:`reference_prompt_text` (which
    renders prompt 0 and carries the refusal of an unpinned thinking template)."""
    from sisteer.models import apply_chat_template_folding
    reference_prompt_text(tok)   # the pin check, once
    return [apply_chat_template_folding(tok, [dict(m) for m in pr], add_generation_prompt=True) for pr in prompts]


def _production_reference_phase(*, model_id: str, revision: str, pending: Sequence[Mapping[str, Any]],
                                out: Path, plan: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The HF + PEFT reference check (assertion 4) and the PEFT reference
    layout, over every pending adapter, with the base model loaded ONCE
    (bf16, eager) — after the engine launches, so the base model never shares
    the GPU with a live engine. Returns ``{status, per_adapter: {dir: {...}}}``."""
    try:
        import peft  # noqa: F401
        from peft import PeftModel
    except ImportError as e:
        # REQUIRED, never optional: a parity check that cannot run certifies nothing —
        # configs/capeval/vllm-requirements.txt pins peft==0.20.0 and the carrier's venv guard imports it
        raise RuntimeError(f"{TAG} peft is not importable in the serving venv ({e}); the HF+PEFT parity check (assertion 4) "
                           "is REQUIRED for certification — install configs/capeval/vllm-requirements.txt (peft==0.20.0) "
                           "and re-run the reference phase") from e
    import warnings
    import torch
    import transformers
    from safetensors.torch import load_file
    from transformers import AutoConfig
    from sisteer.models import load_tokenizer
    # THE CLASS THE CHECKPOINT NAMES: the export spells every PEFT name in the checkpoint's own module tree
    # (`lora_export.ModuleNamespace.checkpoint`, e.g. `model.language_model.layers` on a multimodal checkpoint); the class
    # whose tree that is, is `config.architectures[0]` — never `auto_model_class` (AutoModelForCausalLM first: a text-only
    # class with tree `model.layers`, on which PEFT bound none of the exported keys, warned, and answered as the base model
    # for every adapter of one probe)
    cfg = AutoConfig.from_pretrained(model_id, revision=revision)
    arch = reference_model_class(cfg)
    cls = getattr(transformers, arch, None)
    if cls is None:
        raise RuntimeError(f"{TAG} the checkpoint names architecture {arch!r}, which transformers {transformers.__version__} does not "
                           "export — the reference cannot load the class the export is spelled in; refusing")
    model = cls.from_pretrained(model_id, revision=revision, dtype=torch.bfloat16, attn_implementation="eager")
    model.to("cuda")
    model.eval()
    tok = load_tokenizer(model_id, revision=revision)
    module_names = {n for n, _ in model.named_modules()}
    log(f"reference: {arch} loaded (bf16, eager; transformers {transformers.__version__}, peft {peft.__version__}); "
        f"{len(pending)} adapter(s) pending")
    per: Dict[str, Any] = {}
    layouts: Dict[str, Any] = {}
    layout_of: Dict[str, str] = {}
    # the ONE probe prompt, rendered ONCE through the pinned (thinking-off) render the native server also
    # applies, and tokenised once for every adapter
    text = reference_prompt_text(tok)
    ids = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
    # PARITY CERTIFICATE v2: the certificate's prompts, rendered through the same pinned render; judged for every
    # pending adapter whose launch probed them on the engine side (``vllm_first_tokens_v2``)
    prompts_v2 = plan_parity_prompts(plan) if plan is not None else PARITY_V2_PROMPTS
    sha_v2 = parity_prompts_sha(prompts_v2)
    ids_v2 = [tok(t, return_tensors="pt", add_special_tokens=False).to("cuda") for t in reference_prompt_texts(tok, prompts_v2)]

    def _first_token_for(m: Any, enc: Any) -> Dict[str, Any]:
        with torch.no_grad():
            logits = m(**enc).logits[0, -1].float()
        lp = torch.log_softmax(logits, dim=-1)
        top = torch.topk(lp, 5)
        return {"top1_id": int(top.indices[0]), "top1_logprob": float(top.values[0]),
                "top5": [[int(i), float(v)] for i, v in zip(top.indices.tolist(), top.values.tolist())]}

    def first_token(m: Any) -> Dict[str, Any]:
        return _first_token_for(m, ids)

    def first_tokens_v2(m: Any) -> List[Dict[str, Any]]:
        return [_first_token_for(m, enc) for enc in ids_v2]

    # the BASE through the same class: the floor of the comparison (HF base vs the engine's base probe, no adapter on
    # either side) and the yardstick an adapter must move the reference off (an adapter that leaves it in place
    # was never applied)
    base = first_token(model)
    vllm_base = next((it.get("vllm_base_first_token") for it in pending if it.get("vllm_base_first_token")), None)
    base_parity = first_token_parity(base, vllm_base) if vllm_base else None
    floor = float(base_parity["logprob_abs_delta"]) if base_parity and base_parity.get("logprob_abs_delta") is not None else None
    log(f"reference: base top-1 {base['top1_id']} @ {base['top1_logprob']:.5f}"
        + (f"; engine base {vllm_base.get('id')} @ {vllm_base.get('logprob')}: {base_parity['status']} "
           f"|dlogprob| {base_parity['logprob_abs_delta']}" if base_parity else ""))
    # v2: the HF base on every prompt, ONCE; the floor is PER LAUNCH — max_p of |HF base_p − THAT launch's engine base_p| over
    # the launch's own eight base probes (never the first matching launch's, never a fallback when a probe
    # failed — an incomplete base makes that launch's v2 undecidable)
    base_v2 = first_tokens_v2(model)
    floors_v2: Dict[str, Optional[float]] = {}
    base_d_v2: Dict[str, List[Optional[float]]] = {}
    for it in pending:
        for lid_, eng in (it.get("engine_v2_by_launch") or {}).items():
            if lid_ in floors_v2:
                continue
            if eng.get("sha") != sha_v2:
                floors_v2[lid_], base_d_v2[lid_] = None, []
                continue
            fl, ds_, _complete = v2_base_floor(base_v2, eng.get("base"))
            floors_v2[lid_], base_d_v2[lid_] = fl, ds_
    log(f"reference v2: {len(base_v2)} prompts; per-launch base floors: "
        + ", ".join(f"{k}={v if v is None else round(v, 5)}" for k, v in sorted(floors_v2.items())))
    for item in pending:
        d = Path(item["dir"])
        vl = item.get("vllm_first_token") or {}
        lk = json.dumps({"m": item.get("lora_modules"), "r": item.get("rank"), "re": item.get("per_expert_rank")}, sort_keys=True)
        layout_of[str(d)] = lk
        # the adapter's CONTENT identity, on EVERY outcome of this dir (judged, missing, error): the identity guard groups by it
        # (the same bytes under five ladder names are one subject), and the one-ulp draws are seeded by it
        content = adapter_content_id(d)
        try:
            prefix = export_decoder_prefix(d)
            if prefix and prefix not in module_names:
                # the loaded class does not carry the export's decoder prefix: PEFT would bind none of the keys. No
                # silent remap — a re-spelled adapter is a second object the served engine never saw; refused by name
                per[str(d)] = {"status": REF_STATUS_MISSING, "hf_base": base, "vllm": vl, "export_prefix": prefix, ADAPTER_CONTENT_KEY: content,
                               "reason": f"the reference model {arch} has no module {prefix!r} (the export's decoder prefix) — PEFT "
                                         "would attach none of the exported keys; the class the checkpoint names must carry the "
                                         "export's tree (a prefix remap is not a parity check of the served adapter)"}
                log(f"reference {d.name}: {REF_STATUS_MISSING} — {per[str(d)]['reason'][:160]}")
                continue
            header = safetensors_header(d)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                pm = PeftModel.from_pretrained(model, str(d))
            key_warnings = [str(w.message)[:400] for w in caught if "adapter keys" in str(w.message)]
            pm.eval()
            state = {k: tuple(v.shape) for k, v in pm.state_dict().items()}
            att = attachment_check(header, state)
            # the file's bytes must be what sits on the model (a name that attached with other values is no reference)
            if att["pass"]:
                params = dict(pm.named_parameters())
                altered = []
                for k, t in load_file(str(d / sa.ADAPTER_WEIGHTS_FILE)).items():
                    p = params.get(peft_state_key(k))
                    if p is None or not torch.equal(p.detach().cpu(), t.to(p.dtype)):
                        altered.append(k)
                att["values_altered"] = altered[:20]
                att["n_values_altered"] = len(altered)
                att["pass"] = not altered
            if not att["pass"] or key_warnings:
                per[str(d)] = {"status": REF_STATUS_MISSING, "hf_base": base, "vllm": vl, "attachment": att, ADAPTER_CONTENT_KEY: content,
                               "peft_key_warnings": key_warnings[:3],
                               "reason": (f"PEFT attached {att['n_attached']}/{att['n_file_tensors']} of the exported tensors under their own "
                                          f"names (missing {att['n_missing']}, mis-shaped {len(att['shape_mismatch'])}, altered "
                                          f"{att.get('n_values_altered', 0)}; {len(key_warnings)} PEFT key warning(s)) — the reference "
                                          "does not carry the adapter the engine served")}
                log(f"reference {d.name}: {REF_STATUS_MISSING} — {per[str(d)]['reason'][:200]}")
                model = pm.unload()
                del pm
                continue
            ref = first_token(pm)
            # rule (a): this adapter's one-ulp sensitivity — the tolerance's third term and the chaos gate — measured while
            # the adapter is still on the model (3 forwards; the factors are restored bit for bit); seeded by the adapter's CONTENT,
            # never by its place in this loop (identical bytes under five names drew five different noises, 979954)
            sens = one_ulp_sensitivity(pm, first_token, ref, seed=sensitivity_seed(content, 0, fallback=len(per)))
            # v2: the HF first token and the one-ulp sensitivity on EVERY certificate prompt, ONCE per adapter, for an
            # adapter some launch probed on the engine side under the same prompt set
            refs_v2: Optional[List[Dict[str, Any]]] = None
            sens_v2: List[Optional[Dict[str, Any]]] = []
            eng_by_launch = dict(item.get("engine_v2_by_launch") or {})
            if any(v.get("sha") == sha_v2 for v in eng_by_launch.values()):
                refs_v2 = first_tokens_v2(pm)
                for p_i, enc in enumerate(ids_v2):
                    sens_v2.append(one_ulp_sensitivity(pm, (lambda m, _e=enc: _first_token_for(m, _e)), refs_v2[p_i],
                                                       seed=sensitivity_seed(content, p_i + 1, fallback=len(per) * 101 + p_i)))
            model = pm.unload()
            del pm
            if _top5_key(ref) == _top5_key(base):
                per[str(d)] = {"status": REF_STATUS_MISSING, "hf": ref, "hf_base": base, "vllm": vl, "attachment": att, ADAPTER_CONTENT_KEY: content,
                               "reason": "the adapter leaves the reference's first-token top-5 IDENTICAL to the base's — an attached "
                                         "adapter that changes nothing was not applied in the forward pass"}
                log(f"reference {d.name}: {REF_STATUS_MISSING} — identical to the base")
                continue
            # the gate (chaos / both-weak) and rule (a) in ONE place, shared with the stored-record re-gate
            v1 = judge_adapter_parity(ref, base, vl, moved=item.get("vllm_moved"), sensitivity=sens, floor=floor,
                                      attachment={k: att[k] for k in ("n_file_tensors", "n_attached")})
            v2_by_launch: Dict[str, Any] = {}
            if refs_v2 is not None:
                v2_by_launch = judge_v2_by_launch(refs_v2, base_v2, sens_v2, eng_by_launch, prompts_sha=sha_v2)
            elif eng_by_launch:
                # every launch probed another prompt set: each is undecidable under THIS certificate, by name (and the re-pend rule
                # sees a judgement for the launch, so the phase does not pend it again)
                v2_by_launch = judge_v2_by_launch([{}] * len(ids_v2), [{}] * len(ids_v2), [], eng_by_launch, prompts_sha=sha_v2)
                for lid_, v2r in sorted(v2_by_launch.items()):
                    log(f"reference v2 {d.name} @ {lid_}: {v2r['status']}{' (' + v2r['gate'] + ')' if v2r.get('gate') else ''} — max_p |d_p| "
                        f"{v2r.get('max_abs_delta')} (prompt {v2r.get('argmax_prompt')}) vs bound {v2r.get('tolerance')} (floor_v2 {v2r.get('floor')}, "
                        f"sens_max {v2r.get('sensitivity_max')}); engine_moved {[round(x, 3) if x is not None else None for x in v2r.get('engine_moved_by_prompt') or []]} "
                        f"vs hf_moved {[round(x, 3) if x is not None else None for x in v2r.get('hf_moved_by_prompt') or []]}")
            # the ADAPTER record: v1 + every launch's v2 (+ the HF side ONCE, so a launch judged later re-gates without a model);
            # a launch's reference_check composes ITS v2 (launch_reference_check)
            per[str(d)] = dict(v1)
            per[str(d)][ADAPTER_CONTENT_KEY] = content
            per[str(d)]["v2_by_launch"] = v2_by_launch
            if refs_v2 is not None:
                per[str(d)]["hf_v2"] = refs_v2
                per[str(d)]["sens_v2"] = sens_v2
            if per[str(d)]["status"] == REF_STATUS_UNDECIDABLE:
                log(f"reference {d.name}: v1 {REF_STATUS_UNDECIDABLE} ({per[str(d)]['gate']}) — {per[str(d)]['reason'][:160]}")
                continue
            log(f"reference {d.name}: v1 {per[str(d)]['status']} — HF {ref['top1_id']} @ {ref['top1_logprob']:.4f} vs engine "
                f"{vl.get('id')} @ {vl.get('logprob')} (|d| {per[str(d)]['logprob_abs_delta']}, bound {per[str(d)]['tolerance']:.4f}; "
                f"one-ulp shift {sens.get('max_shift')}, flips {sens.get('flips')})")
        except Exception as e:  # noqa: BLE001
            per[str(d)] = {"status": REF_STATUS_ERROR, "error": f"{type(e).__name__}: {e}"[:1000], "vllm": vl, ADAPTER_CONTENT_KEY: content}
            log(f"reference {d.name}: ERROR {type(e).__name__}: {str(e)[:200]}")
            # never leave the base wrapped after a failure inside the adapter's forward
            if hasattr(model, "unload"):
                try:
                    model = model.unload()
                except Exception:  # noqa: BLE001
                    pass
        # the PEFT reference LAYOUT for this adapter's architecture (once per (modules, rank, r_e))
        if lk not in layouts and item.get("kind") == sm.KIND_GENUINE:
            try:
                rec = peft_reference_layout(model, Path(out) / "peft_reference" / hashlib.sha1(lk.encode()).hexdigest()[:12],
                                            modules=item["lora_modules"], rank=int(item["rank"]),
                                            per_expert_rank=item.get("per_expert_rank"),
                                            wraps_experts=bool(item.get("wraps_experts")))
                model = rec.pop("_model", model)
                if rec.get("status") == "written":
                    rec["compare"] = compare_adapter_layouts(safetensors_header(d), safetensors_header(Path(rec["dir"])))
                layouts[lk] = rec
            except Exception as e:  # noqa: BLE001
                layouts[lk] = {"status": "error", "error": f"{type(e).__name__}: {e}"[:1000]}
        torch.cuda.empty_cache()
    rec = finish_reference_record(per, layouts, layout_of, base=base, vllm_base=vllm_base, base_parity=base_parity,
                                  peft_version=getattr(peft, "__version__", None), reference_class=arch)
    # v2's base facts beside v1's (``base_floor``): per LAUNCH, so the record re-gates v2 from them (:func:`regate_adapter_record`)
    rec["base_floor_v2_by_launch"] = floors_v2
    rec["base_d_v2_by_launch"] = base_d_v2
    rec["parity_prompts_sha"] = sha_v2
    rec["hf_base_v2"] = base_v2
    # the venv the HF+PEFT reference ran in — on the record and on every judged adapter's own record
    rec[SERVING_VENV_KEY] = serving_venv_block(source=IDENTITY_SOURCE_REFERENCE)      # Measured in the reference process
    for v in (rec.get("per_adapter") or {}).values():
        if isinstance(v, Mapping) and v.get("status") in (REF_STATUS_PASS, REF_STATUS_FAIL, REF_STATUS_UNDECIDABLE):
            v[SERVING_VENV_KEY] = dict(rec[SERVING_VENV_KEY])
    return rec


def finish_reference_record(per: Dict[str, Any], layouts: Mapping[str, Any], layout_of: Mapping[str, str], *,
                            base: Optional[Mapping[str, Any]] = None, vllm_base: Optional[Mapping[str, Any]] = None,
                            base_parity: Optional[Mapping[str, Any]] = None, peft_version: Optional[str] = None,
                            reference_class: Optional[str] = None) -> Dict[str, Any]:
    """The whole-phase harness checks over the per-adapter records (pure; the production phase ends here): an adapter
    whose PEFT layout is not WRITTEN with a PASSING compare against the export (errored, peft unavailable, never attempted,
    no compare, or a disagreement), and every member of a group of distinct adapters with IDENTICAL HF outputs, is
    ``reference_missing`` — never a mismatch. ``layout_of`` maps every genuine adapter dir the phase judged to its
    architecture's layout key; an adapter it does not name is left to the caller (a pure unit test)."""
    per = {k: dict(v) for k, v in per.items()}
    judged = (REF_STATUS_PASS, REF_STATUS_FAIL, REF_STATUS_UNDECIDABLE)
    for d, v in per.items():
        if v.get("status") not in judged or d not in layout_of:
            continue      # `layout_of` is the phase's map of every genuine adapter to its architecture's layout key (a pure caller may omit it)
        lk = layout_of[d]
        lay = layouts.get(lk)
        cmp_ = (lay or {}).get("compare") or {}
        # the independent layout check must be PRESENT and PASSING for every judged adapter — an errored /
        # unavailable / never-attempted layout, or one with no compare, is the unsupported-layout case it exists for: no evidence
        if lay is None:
            v["status"] = REF_STATUS_MISSING
            v["reason"] = (f"no PEFT reference layout was produced for this adapter's architecture (layout key {lk!r} absent from the "
                           "phase's layouts) — the independent layout check never ran; the parity verdict has no basis without it")
        elif lay.get("status") != "written":
            v["status"] = REF_STATUS_MISSING
            v["reason"] = (f"PEFT's own layout of this architecture was not produced: status {lay.get('status')!r}"
                           + (f" — {lay.get('error') or lay.get('reason')}" if (lay.get("error") or lay.get("reason")) else "")
                           + " — the independent layout check did not run to completion; the parity verdict has no basis without it")
        elif not cmp_:
            v["status"] = REF_STATUS_MISSING
            v["reason"] = (f"PEFT's own layout of this architecture was written ({lay.get('dir')}) but carries no compare against the "
                           "export — the independent layout check recorded no verdict")
        elif cmp_.get("pass") is not True:
            v["status"] = REF_STATUS_MISSING
            v["reason"] = (f"PEFT's own layout of this architecture disagrees with the export: {cmp_.get('n_ours')} exported vs "
                           f"{cmp_.get('n_peft')} PEFT tensors, only_ours {len(cmp_.get('only_ours') or [])}, only_peft "
                           f"{len(cmp_.get('only_peft') or [])}, shape_mismatch {len(cmp_.get('shape_mismatch') or [])} "
                           f"(e.g. {(cmp_.get('only_ours') or cmp_.get('only_peft') or ['-'])[0]}) — the reference and the export "
                           "name different modules")
    # the identity guard groups by adapter CONTENT (``content_sha256``, recorded by the phase; the dir for a record without it): the same
    # bytes under several manifest dirs are ONE adapter and never a group — only >= 2 distinct contents with identical HF outputs refuse
    for group in identical_reference_groups({d: v for d, v in per.items() if v.get("status") in judged}):
        reason = _identical_group_reason(group)
        for d in group["dirs"]:
            per[d]["status"] = REF_STATUS_MISSING
            per[d]["reason"] = reason
    n = {s: sum(1 for v in per.values() if v.get("status") == s)
         for s in (REF_STATUS_PASS, REF_STATUS_FAIL, REF_STATUS_UNDECIDABLE, REF_STATUS_MISSING, REF_STATUS_ERROR)}
    log(f"reference: {n[REF_STATUS_PASS]} pass, {n[REF_STATUS_FAIL]} fail, {n[REF_STATUS_UNDECIDABLE]} {REF_STATUS_UNDECIDABLE}, "
        f"{n[REF_STATUS_MISSING]} {REF_STATUS_MISSING}, {n[REF_STATUS_ERROR]} error over {len(per)} adapter(s)")
    return {"status": "done", "peft_version": peft_version, "reference_class": reference_class, "hf_base": base,
            "vllm_base": vllm_base, "base_parity": base_parity, "counts": n, "per_adapter": per, "layouts": dict(layouts),
            # the rule's record-wide terms (the measured floor; the constants the per-adapter bounds were built from)
            "base_floor": ((base_parity or {}).get("logprob_abs_delta") if base_parity else None),
            "parity_rule": {"tolerance": REFERENCE_LOGPROB_TOL, "floor_k": PARITY_FLOOR_K, "sensitivity_m": PARITY_SENSITIVITY_M,
                            "sensitivity_draws": PARITY_SENSITIVITY_DRAWS, "ulp_rel": PARITY_ULP_REL, "moved_min": PARITY_MOVED_MIN}}


# --------------------------------------------------------------------------
# run (per launch)
# --------------------------------------------------------------------------

def classify_failure(log_text: str, exc: Optional[BaseException]) -> Tuple[str, str]:
    """``(status, message)`` for a launch that never became ready."""
    text = log_text or ""
    msg = f"{type(exc).__name__}: {exc}" if exc is not None else ""
    if isinstance(exc, LaunchRefused):
        return STATUS_REFUSED, str(exc)
    if OOM_RE.search(text) or OOM_RE.search(msg):
        return STATUS_OOM, "cuda_out_of_memory"
    m = RANK_REFUSAL_RE.search(text) or RANK_REFUSAL_RE.search(msg)
    if m:
        return STATUS_REFUSED, m.group(0)
    m = MOE_RANK_REFUSAL_RE.search(text) or MOE_RANK_REFUSAL_RE.search(msg)
    if m:
        return STATUS_REFUSED, m.group(0)
    if LOG_UNWRAPPED_RE.search(text):
        return STATUS_REFUSED, "module could not be wrapped"
    if ILLEGAL_ACCESS_RE.search(text):
        return STATUS_ERROR, "illegal_memory_access"
    if MAMBA_RE.search(text):
        return STATUS_REFUSED, "mamba_cache_blocks"
    if exc is not None and type(exc).__name__ in ("VllmServerError", "KvCoverageShortfall"):
        return STATUS_REFUSED, msg[:500]
    return STATUS_ERROR, msg[:500] or "engine_exit"


def evaluate_checks(*, manifest: sm.Manifest, launch_rec: Mapping[str, Any], dims: Any, log_text: str,
                    models_listed: Sequence[str], base_probe: Mapping[str, Any],
                    adapter_probes: Mapping[str, Mapping[str, Any]], worker_report: Mapping[str, Any],
                    is_moe: bool = False,
                    adapter_decodes: Optional[Mapping[str, Sequence[Mapping[str, Any]]]] = None,
                    require_measurements: bool = True) -> Dict[str, Any]:
    """Assertions (1)-(3), (5)-(7) as a record — pure over the observations.
    ``is_moe`` (the plan's ``is_moe``: the dims route experts) makes the fused-MoE
    LoRA log line a HARD check; a dense launch (the default) does not expect it.

    The stacked-buffer bytes (5) and the executing stride are HARD checks
    beside the log / listing / adapter / companion ones: a measured buffer
    outside the 5 % law, a reported stride wider than the manifest's, and —
    under ``require_measurements`` (the default: every lora-native worker
    report carries both, ``worker_reports.lora_execution_modules_report``) —
    a report that lacks either measurement or left a wrapper's buffers
    unread all FAIL the launch, so a vLLM buffer-law change can never
    certify a (rank, set) pair or size an H200 wrongly."""
    checks: Dict[str, Any] = {}
    # (1) engine log
    moe_line = LOG_MOE_LORA_LINE in log_text
    checks["log_moe_lora_line"] = {"expected": bool(is_moe), "found": moe_line,
                                   "pass": (moe_line if is_moe else True)}
    unwrapped = [m.group(0) for m in re.finditer(r".*could not be wrapped.*", log_text, re.I)][:5]
    checks["log_no_unwrapped"] = {"found": unwrapped, "pass": not unwrapped}
    triton = LOG_TRITON_RE.search(log_text)
    checks["log_triton_backend"] = {
        "applicable": manifest.wraps_experts, "found": bool(triton), "line": (triton.group(0).strip()[:300] if triton else None),
        "pass": True,
        "note": ("0.25.1's unquantized-MoE oracle FORCES Triton under LoRA (fused_moe/oracle/unquantized.py:177-180) "
                 "without a distinct INFO line; the log evidence is recorded, the kernel gate itself is "
                 "FusedMoEWithLoRA's assert moe_kernel.supports_lora() at init (fused_moe.py:86-92)")}
    # (2) /v1/models
    names = [a.name for a in manifest.adapters]
    missing = [n for n in names if n not in models_listed]
    checks["models_listed"] = {"expected": names, "missing": missing, "listed": list(models_listed), "pass": not missing}
    # (3) per adapter vs base
    per: Dict[str, Any] = {}
    for a in manifest.adapters:
        pr = adapter_probes.get(a.name)
        if pr is None:
            per[a.name] = {"kind": a.kind, "role": a.role, "rank": a.rank, "pass": False, "reason": "no probe response"}
            continue
        if a.kind == sm.KIND_GENUINE:
            delta = first_token_delta(base_probe, pr)
            moved = ((delta["max_abs_delta"] is not None and delta["max_abs_delta"] > FIRST_TOKEN_DELTA_MIN)
                     or bool(delta.get("first_token_changed")) or bool(delta.get("top_set_changed")))
            per[a.name] = {"kind": a.kind, "role": a.role, "rank": a.rank, "per_expert_rank": a.per_expert_rank,
                           "first_token": delta, "threshold": FIRST_TOKEN_DELTA_MIN, "pass": bool(moved),
                           "reason": None if moved else "adapter output equals base (silent no-op?)"}
        else:
            ident = bit_identical(base_probe, pr)
            per[a.name] = {"kind": a.kind, "role": a.role, "rank": a.rank, "per_expert_rank": a.per_expert_rank, "bit_identical_to_base": ident,
                           "pass": bool(ident), "reason": None if ident else "zero adapter is not bit-identical to base"}
        if adapter_decodes and a.name in adapter_decodes:
            decs = adapter_decodes[a.name]
            per[a.name]["decodes"] = {"n": len(decs), "n_empty": sum(1 for d in decs if not (d.get("token_ids") or [])),
                                      "finish_reasons": sorted({str(d.get("finish_reason")) for d in decs})}
    checks["adapters"] = per
    # (5) buffer bytes: worker_reports' `lora_execution_module_lora_buffer_bytes` is the
    # TOTAL (int) with `..._by_module` the per-wrapper map; a report carrying the map under
    # the total key is summed (both shapes -> one (measured, n_modules) pair)
    expected = launch_rec["expected_lora_bytes"]
    raw = worker_report.get("lora_execution_module_lora_buffer_bytes")
    by_module = worker_report.get("lora_execution_module_lora_buffer_bytes_by_module")
    measured: Optional[int] = None
    n_modules: Optional[int] = None
    if isinstance(raw, Mapping) and raw:
        measured = int(sum(int(v) for v in raw.values() if isinstance(v, (int, float))))
        n_modules = len(raw)
    elif isinstance(raw, (int, float)) and not isinstance(raw, bool):
        measured = int(raw)
        n_modules = len(by_module) if isinstance(by_module, Mapping) else None
    unread = worker_report.get("lora_execution_module_lora_buffer_unread") or []
    if measured is not None and not unread:
        checks["buffer_bytes"] = {**sm.buffer_bytes_within(expected["total"], measured, BUFFER_BYTES_TOL),
                                  "status": "measured", "expected_per_slot": expected["per_slot"],
                                  "n_modules": n_modules}
        if not checks["buffer_bytes"]["pass"]:
            checks["buffer_bytes"]["reason"] = (
                f"measured stacked LoRA buffers {measured} B are outside {BUFFER_BYTES_TOL:.0%} of the expected "
                f"{expected['total']} B (standup_manifest.expected_lora_bytes) — the buffer law changed")
    elif measured is not None:
        checks["buffer_bytes"] = {"status": "partially_measured", "expected": expected["total"], "measured": measured,
                                  "unread": list(unread), "required": bool(require_measurements),
                                  "pass": (False if require_measurements else None),
                                  "reason": f"the worker exposed no stacked buffers for {list(unread)[:5]} — the "
                                            "total covers the read wrappers only"}
    else:
        checks["buffer_bytes"] = {"status": "not_measured", "expected": expected["total"],
                                  "required": bool(require_measurements),
                                  "pass": (False if require_measurements else None),
                                  "reason": "worker report lacks lora_execution_module_lora_buffer_bytes — a "
                                            "lora-native engine's report always carries it; a missing measurement "
                                            "cannot certify the buffer law"}
    # memory facts, recorded separately
    blocks = worker_report.get("kv_num_gpu_blocks")
    bsz = worker_report.get("kv_block_size")
    kv_tokens = (int(blocks) * int(bsz)) if isinstance(blocks, int) and isinstance(bsz, int) else None
    checks["memory"] = {"kv_num_gpu_blocks": blocks, "kv_block_size": bsz, "kv_capacity_tokens": kv_tokens,
                        "kv_capacity_bytes": (kv_tokens * sm.kv_bytes_per_token(dims)) if kv_tokens else None,
                        "kv_max_concurrency": worker_report.get("kv_max_concurrency"),
                        "engine_max_num_seqs": worker_report.get("max_num_seqs"),
                        "max_num_batched_tokens": worker_report.get("max_num_batched_tokens"),
                        "lora_max_lora_rank": worker_report.get("lora_max_lora_rank"),
                        "lora_max_loras": worker_report.get("lora_max_loras"),
                        "stride_max": worker_report.get("lora_execution_module_stride_max"),
                        "non_kv_bytes": worker_report.get("non_kv_memory_bytes"),
                        "total_vram_bytes": worker_report.get("total_gpu_memory_bytes")}
    # the LAUNCH CLASS the engine attests (worker_reports: lora_max_lora_rank / lora_max_loras) IS the one the plan intended —
    # the manifest's cap and slot count (the probe owner's ruling of 2026-10-11): the fold writes the engine's class and the
    # site record sizes a standup from the launches whose class dominates the served one, so a launch certifying another
    # class than the one it was planned to is a HARD failure; a report that lacks the pair is not judged here (the fold
    # refuses to fold a passed launch without it — never guessed)
    intended = {"max_lora_rank": int(manifest.max_lora_rank), "max_loras": int(manifest.max_loras)}
    reported = {"max_lora_rank": worker_report.get("lora_max_lora_rank"), "max_loras": worker_report.get("lora_max_loras")}
    if all(isinstance(v, int) and not isinstance(v, bool) and v >= 1 for v in reported.values()):
        checks["launch_class"] = {"status": "measured", "expected": intended, "reported": reported,
                                  "pass": all(int(reported[k]) == intended[k] for k in intended)}
        if not checks["launch_class"]["pass"]:
            checks["launch_class"]["reason"] = (
                f"the engine attests launch class (rank cap {reported['max_lora_rank']}, {reported['max_loras']} adapters) but the "
                f"manifest {manifest.name} was planned at (rank cap {intended['max_lora_rank']}, {intended['max_loras']} adapters) — "
                "the launch certifies another class than the one the plan covers")
    else:
        checks["launch_class"] = {"status": "not_reported", "expected": intended, "reported": reported, "pass": None,
                                  "reason": "worker report lacks lora_max_lora_rank / lora_max_loras; the fold refuses to fold the class"}
    strides = worker_report.get("lora_execution_module_strides")
    exp_strides = sm.executing_strides(dims, manifest.target_class)
    exp_D = max(max(v["in"], v["out"]) for v in exp_strides.values())
    if isinstance(strides, Mapping) and strides:
        got_D = max(max(int(v.get("in", 0)), int(v.get("out", 0))) for v in strides.values() if isinstance(v, Mapping))
        checks["executing_stride"] = {"expected_D": exp_D, "reported_D": got_D, "pass": got_D <= exp_D,
                                      "n_wrappers": len(strides), "status": "measured"}
        if got_D > exp_D:
            checks["executing_stride"]["reason"] = (
                f"a LoRA execution wrapper reports stride {got_D} > the manifest's widest {exp_D} "
                "(standup_manifest.executing_strides) — the int32 kernel index ceiling was sized from the wrong D")
    else:
        checks["executing_stride"] = {"expected_D": exp_D, "reported_D": None, "status": "not_reported",
                                      "required": bool(require_measurements),
                                      "pass": (False if require_measurements else None),
                                      "reason": "worker report lacks lora_execution_module_strides"}
    # (6) mixed rank: the MLP sets' genuine adapters ARE the mixed-rank variant (routed-experts models only)
    mixed = [a for a in manifest.genuine if a.per_expert_rank is not None and a.per_expert_rank != a.rank]
    checks["mixed_rank"] = {"applicable": bool(mixed),
                            "adapters": [{"name": a.name, "attention_shared_rank": a.rank, "expert_rank": a.per_expert_rank}
                                         for a in mixed],
                            "pass": (all(per.get(a.name, {}).get("pass") for a in mixed) if mixed else None)}
    # (7) the down set's zero gate_up companion changes no logit: the class zero
    # adapter carries the zero companion pair + zero down factors and must be
    # bit-identical to base; the genuine down adapter must differ
    if manifest.set_id == "down" and manifest.wraps_experts:
        z = per.get(manifest.zero.name, {})
        g = [per[a.name] for a in manifest.genuine if a.name in per]
        checks["companion_check"] = {"applicable": True, "zero_bit_identical": bool(z.get("pass")),
                                     "genuine_differ": all(bool(x.get("pass")) for x in g) if g else None,
                                     "pass": bool(z.get("pass")) and (all(bool(x.get("pass")) for x in g) if g else False),
                                     "basis": "class zero adapter (zero gate_up companion + zero down) bit-identical to base"}
    else:
        checks["companion_check"] = {"applicable": False, "pass": None}
    hard = [k for k in ("log_moe_lora_line", "log_no_unwrapped", "models_listed") if not checks[k]["pass"]]
    hard += [f"adapters/{n}" for n, v in per.items() if not v.get("pass")]
    # (5) + the stride + the attested launch class join the hard failures: False fails; None (not required / not reported) does not
    hard += [k for k in ("buffer_bytes", "executing_stride", "launch_class") if checks[k]["pass"] is False]
    if checks["companion_check"]["applicable"] and not checks["companion_check"]["pass"]:
        hard.append("companion_check")
    checks["failed"] = hard
    checks["pass"] = not hard
    return checks


def launch_result_path(out: Path, lid: str) -> Path:
    return Path(out) / LAUNCHES_DIR / f"{lid}.json"


def load_results(out: Path) -> Dict[str, Dict[str, Any]]:
    d = Path(out) / LAUNCHES_DIR
    res: Dict[str, Dict[str, Any]] = {}
    if d.is_dir():
        for p in sorted(d.glob("*.json")):
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
            except ValueError:
                continue
            if isinstance(rec, Mapping) and rec.get("launch_id"):
                res[str(rec["launch_id"])] = dict(rec)
    return res


def condition_met(launch_rec: Mapping[str, Any], results: Mapping[str, Mapping[str, Any]]) -> Optional[bool]:
    """None while the parent is undecided; True when the fallback should run."""
    cond = launch_rec.get("condition")
    if not cond:
        return True
    parent = results.get(cond["launch"])
    if parent is None or parent.get("status") not in TERMINAL_STATUSES:
        return None
    return parent.get("status") in cond["outcome"]


def next_launch(plan: Mapping[str, Any], results: Mapping[str, Mapping[str, Any]], *,
                exclude: Optional[Iterable[str]] = None) -> Optional[Dict[str, Any]]:
    """The next launch to run: priority order, skipping done ones and the ids
    in ``exclude`` (the launches a parallel run has IN FLIGHT — not yet on
    the record, not to be started twice); a fallback whose parent PASSED is
    marked skipped; a fallback whose parent is undecided (unrun, or in
    flight) is deferred — the next eligible launch is returned instead."""
    skip = set(exclude or ())
    for rec in plan["launches"]:
        if rec["id"] in skip:
            continue
        done = results.get(rec["id"])
        if done is not None and done.get("status") in TERMINAL_STATUSES:
            continue
        met = condition_met(rec, results)
        if met is None:
            continue
        if met is False:
            return {**rec, "_skip": True}
        return dict(rec)
    return None


def outstanding_launches(plan: Mapping[str, Any], results: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Every planned launch that still has to run: no terminal result and a
    condition that is not decided against it (a fallback whose parent PASSED
    will be marked skipped on the next pass and is not outstanding)."""
    out: List[Dict[str, Any]] = []
    for rec in plan["launches"]:
        done = results.get(rec["id"])
        if done is not None and done.get("status") in TERMINAL_STATUSES:
            continue
        if condition_met(rec, results) is False:
            continue
        out.append(dict(rec))
    return out


def _launch_tokens(rec: Mapping[str, Any], res: Mapping[str, Any]) -> Tuple[int, int]:
    """``(S, S x L)`` a launch actually ran at: the engine's resolved ``max_num_seqs`` when it came up, else the S it
    was launched at (``launch_S``), else the plan's admitted S (at least 1 — the driver launches S < 1 at 1)."""
    eng = (res.get("engine") or {}).get("max_num_seqs_engine")
    if isinstance(eng, int) and not isinstance(eng, bool) and eng >= 1:
        S = int(eng)
    else:
        S_l = res.get("launch_S")
        S = int(S_l) if isinstance(S_l, int) and not isinstance(S_l, bool) and S_l >= 1 else max(1, int(rec.get("admitted_S") or 0))
    return S, S * int(rec["L"])


def _is_start_time(res: Mapping[str, Any]) -> bool:
    """The engine never became ready (``ready_s`` is None): the failure is at start, before any request was served."""
    return res.get("ready_s") is None


def _start_oom_signatures(out: Optional[Path], lid: str, res: Mapping[str, Any]) -> List[str]:
    """The :data:`START_OOM_SIGNATURES` present in the launch's server log (when the record dir holds it) or its
    recorded failure text — detail for the reader, never the classification itself."""
    texts: List[str] = []
    f = res.get("failure") or {}
    texts.append(str(f.get("log_tail") or ""))
    texts.append(str(f.get("exception") or ""))
    if out is not None:
        texts.append(_read_log_tail(Path(out) / LAUNCHES_DIR / lid / SERVER_LOG_FILE, n=400_000))
    blob = "\n".join(texts)
    return [sig for sig in START_OOM_SIGNATURES if sig in blob]


def measured_budget(plan: Mapping[str, Any], results: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """The record's MEASURED batched-token budget: ``tokens`` = the largest ``S x L`` a launch of this record
    came up at (the floor, with the launch it came from), ``ceiling_tokens`` = the smallest ``S x L`` ABOVE the floor
    at which an engine died at start (with its launch), ``method`` ``record_floor``; with no passed launch the
    budget is ``unmeasured`` (``tokens`` None) and says so. Every number names the launch id it came from."""
    by_id = {r["id"]: r for r in plan["launches"]}
    floor: Optional[Tuple[int, str]] = None
    passed: Dict[str, int] = {}
    for lid, res in sorted(results.items()):
        rec = by_id.get(lid)
        if rec is None or res.get("status") != STATUS_PASSED:
            continue
        S, tokens = _launch_tokens(rec, res)
        passed[lid] = tokens
        if floor is None or tokens > floor[0]:
            floor = (tokens, lid)
    ceiling: Optional[Tuple[int, str]] = None
    start_ooms: Dict[str, int] = {}
    for lid, res in sorted(results.items()):
        rec = by_id.get(lid)
        if rec is None or res.get("status") != STATUS_OOM or not _is_start_time(res):
            continue
        S, tokens = _launch_tokens(rec, res)
        start_ooms[lid] = tokens
        if S > 1 and (floor is None or tokens > floor[0]) and (ceiling is None or tokens < ceiling[0]):
            ceiling = (tokens, lid)
    if floor is None:
        return {"method": BUDGET_METHOD_UNMEASURED, "tokens": None, "floor_launch": None,
                "ceiling_tokens": (ceiling[0] if ceiling else None), "ceiling_launch": (ceiling[1] if ceiling else None),
                "n_passed": 0, "n_start_time_oom": len(start_ooms), "measured_utc": utc_now(),
                "note": "no launch of this record has come up: nothing measures the floor — no cap applies until one does"}
    return {"method": BUDGET_METHOD_RECORD_FLOOR, "tokens": int(floor[0]), "floor_launch": floor[1],
            "ceiling_tokens": (ceiling[0] if ceiling else None), "ceiling_launch": (ceiling[1] if ceiling else None),
            "n_passed": len(passed), "n_start_time_oom": len(start_ooms), "measured_utc": utc_now(),
            "note": (f"floor = the largest S x L a lora-native engine of this record came up at ({floor[1]}: {floor[0]:,} "
                     "tokens); every planned S is min(stride, KV, requested, floor // L) — D108: the no-chunking prefill "
                     "of S x L tokens is priced by measurement, never a typed constant"
                     + (f"; ceiling = the smallest start-time OOM above it ({ceiling[1]}: {ceiling[0]:,})" if ceiling else
                        "; no start-time OOM above the floor recorded yet"))}


def oom_kind(rec: Mapping[str, Any], res: Mapping[str, Any], budget_tokens: Optional[int]) -> Optional[str]:
    """How the verdict READS an ``oom`` launch (None for any other status):
    * :data:`OOM_SERVING` — the engine was up (``ready_s`` set) and OOM'd under load: over the window, genuine.
    * :data:`OOM_START_MIN_S` — died at start at S = 1: no smaller batch exists, genuine (buffers + weights + the
      one-sequence prefill do not fit).
    * :data:`OOM_START_WITHIN_BUDGET` — died at start at ``S x L`` <= the measured floor: a batch that came up
      elsewhere did not here, so the LoRA buffers / KV of THIS manifest are what does not fit — genuine.
    * :data:`OOM_START_OVER_BUDGET` — died at start at ``S x L`` above the floor (or with no floor measured): the
      estimator planned a prefill batch larger than any this record has proven; says nothing about the pair —
      ``replan_required``, never an exclusion."""
    if res.get("status") != STATUS_OOM:
        return None
    if not _is_start_time(res):
        return OOM_SERVING
    S, tokens = _launch_tokens(rec, res)
    if S <= 1:
        return OOM_START_MIN_S
    if budget_tokens is not None and tokens <= int(budget_tokens):
        return OOM_START_WITHIN_BUDGET
    return OOM_START_OVER_BUDGET


def replan_required_launches(plan: Mapping[str, Any], results: Mapping[str, Mapping[str, Any]],
                             budget: Optional[Mapping[str, Any]] = None) -> List[str]:
    """The launch ids whose recorded ``oom`` is a start-time prefill OOM above the measured budget (plan order):
    their S must be re-derived and they re-run (``run --resume --replan``) before a verdict may read them."""
    budget = budget if budget is not None else measured_budget(plan, results)
    tokens = budget.get("tokens")
    out: List[str] = []
    for rec in plan["launches"]:
        res = results.get(rec["id"])
        if res is not None and oom_kind(rec, res, tokens) in REPLAN_OOM_KINDS:
            out.append(rec["id"])
    return out


def replan_launches(plan: Mapping[str, Any], results: Mapping[str, Mapping[str, Any]],
                    budget: Mapping[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """``(new plan, changes)``: every launch WITHOUT a terminal result (outstanding) and every ``replan_required``
    launch gets its S re-derived through :func:`launch_admission` with the measured budget — ids are stable
    (:func:`launch_id` encodes manifest, cap, L and tp, never S); launches that ran to a terminal state keep their
    recorded S (history). The plan's ``batched_token_budget`` becomes the measured record. ``changes`` lists
    ``{id, old_S, new_S, binding}`` for every launch whose S moved; an unchanged plan returns an empty list and the
    SAME launch records (idempotent)."""
    dims = _Dims(plan["dims"])
    b = plan["budget"]
    replan = set(replan_required_launches(plan, results, budget))
    tokens = budget.get("tokens")
    new_launches: List[Dict[str, Any]] = []
    changes: List[Dict[str, Any]] = []
    for rec in plan["launches"]:
        res = results.get(rec["id"])
        terminal = res is not None and res.get("status") in TERMINAL_STATUSES
        if terminal and rec["id"] not in replan:
            new_launches.append(dict(rec))
            continue
        m = plan_manifest(plan, rec["manifest"])
        L = int(rec["L"])
        # a launch that has not run is re-priced under the CURRENT bytes law (the experts at the manifest's
        # moe_lora_rank — the engine it will meet allocates them there) and its KV budget follows; a launch that ran
        # keeps its recorded law (above)
        lora_bytes = sm.expected_lora_bytes(m, int(rec["cap"]), dims)
        usable = float(b["gpu_memory_utilization"]) * float(b["total_vram_bytes"])
        kv_budget = max(0.0, usable - float(b["base_weights_bytes"]) - lora_bytes["total"] - float(b["graphs_activations_bytes"]))
        adm = launch_admission(dims, m, L, requested_S=int(rec["requested_S"]), kv_budget_bytes=kv_budget,
                               recurrent_state_bytes=int(b["recurrent_state_per_seq_bytes"] or 0),
                               batched_token_budget=(int(tokens) if tokens is not None else None))
        est = sm.estimated_vram_bytes(m, int(rec["cap"]), dims, base_weights_bytes=float(b["base_weights_bytes"]),
                                      kv_per_seq_bytes=adm["kv_per_seq_bytes"], S=max(1, adm["admitted_S"]),
                                      graphs_activations_bytes=float(b["graphs_activations_bytes"]),
                                      gpu_memory_utilization=float(b["gpu_memory_utilization"]),
                                      total_vram_bytes=float(b["total_vram_bytes"]),
                                      batched_tokens=max(1, adm["admitted_S"]) * L)
        new = {**rec, "admitted_S": int(adm["admitted_S"]), "admission": adm, "estimated_vram": est,
               "expected_lora_bytes": lora_bytes, "kv_budget_bytes": kv_budget, "moe_lora_rank": m.moe_lora_rank,
               "roles": list(m.genuine_roles)}
        if int(new["admitted_S"]) != int(rec["admitted_S"]):
            changes.append({"id": rec["id"], "old_S": int(rec["admitted_S"]), "new_S": int(new["admitted_S"]),
                            "binding": adm["binding"], "L": L, "batched_tokens": adm["batched_tokens"],
                            "replan_required": rec["id"] in replan})
        new_launches.append(new)
    new_plan = dict(plan)
    new_plan["launches"] = new_launches
    new_plan["batched_token_budget"] = dict(budget)
    return new_plan, changes


def supersede_launch_results(out: Path, lids: Sequence[str]) -> List[Dict[str, str]]:
    """MOVE (never delete) the result file and launch dir of each id under ``launches/superseded/<id>.<utc>[.json]``
    so the launch is outstanding again with its history kept; returns what moved."""
    out = Path(out)
    stamp = utc_now().replace(":", "")
    dest = out / LAUNCHES_DIR / SUPERSEDED_DIR
    moved: List[Dict[str, str]] = []
    for lid in lids:
        dest.mkdir(parents=True, exist_ok=True)
        src_file = launch_result_path(out, lid)
        if src_file.is_file():
            dst = dest / f"{lid}.{stamp}.json"
            os.replace(src_file, dst)
            moved.append({"id": lid, "from": str(src_file), "to": str(dst)})
        src_dir = out / LAUNCHES_DIR / lid
        if src_dir.is_dir():
            dst = dest / f"{lid}.{stamp}"
            os.replace(src_dir, dst)
            moved.append({"id": lid, "from": str(src_dir), "to": str(dst)})
    return moved


def result_generation(out: Optional[Path]) -> Optional[Dict[str, Any]]:
    """The record's CURRENT result generation (``geometry.record_result_generation``: sha256 over the sorted
    ``name:sha256(bytes)`` lines of ``launches/*.json``); None without a record dir or launches directory."""
    if out is None:
        return None
    from sisteer.capeval.serving import geometry as g
    return g.record_result_generation(Path(out))


def site_overlay_of_record(out: Path, model_key: str, geometry_dir: Optional[Path] = None) -> Optional[Path]:
    """The site geometry overlay file of ``model_key`` whose provenance names THIS record (``probe_record == out``), when
    the overlay directory resolves here (``geometry_dir``, else ``geometry.site_geometry_dir()``); None otherwise. An
    overlay of another record (another probe of the same model) is never touched."""
    from sisteer.capeval.serving import geometry as g
    p = g.site_geometry_path(str(model_key), geometry_dir)
    if p is None or not p.is_file():
        return None
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    prov = (doc.get("provenance") or {}) if isinstance(doc, Mapping) else {}
    return p if str(prov.get("probe_record")) == str(Path(out)) else None


def archive_derived_artifacts(out: Path, *, why: str, model_key: Optional[str] = None, only: Optional[Sequence[str]] = None,
                              geometry_dir: Optional[Path] = None, overlay: bool = True) -> Dict[str, Any]:
    """MOVE (never delete) the record's derived certification artifacts (:data:`DERIVED_ARTIFACTS`, or ``only``) and — when
    ``overlay`` — the site geometry overlay this record folded, under ``launches/superseded/<utc>/`` with an
    ``ARCHIVED.json`` note (why, what moved, when). Returns ``{"dir", "archived": [names]}``; nothing to move = no dir."""
    out = Path(out)
    names = list(only) if only is not None else list(DERIVED_ARTIFACTS)
    present = [n for n in names if (out / n).is_file()]
    ov = site_overlay_of_record(out, model_key, geometry_dir) if (overlay and model_key) else None
    if not present and ov is None:
        return {"dir": None, "archived": []}
    stamp = utc_now().replace(":", "")
    dest = out / LAUNCHES_DIR / SUPERSEDED_DIR / stamp
    k = 0
    while dest.exists():
        k += 1
        dest = out / LAUNCHES_DIR / SUPERSEDED_DIR / f"{stamp}.{k}"
    dest.mkdir(parents=True)
    moved: List[Dict[str, str]] = []
    for n in present:
        os.replace(out / n, dest / n)
        moved.append({"name": n, "from": str(out / n), "to": str(dest / n)})
    if ov is not None:
        target = dest / f"site_overlay_{ov.name}"
        os.replace(ov, target)
        moved.append({"name": f"site_overlay_{ov.name}", "from": str(ov), "to": str(target)})
    note = {"archived_utc": utc_now(), "why": why, "record": str(out), "moved": moved,
            RESULT_GENERATION_KEY: (result_generation(out) or {}).get("generation")}
    (dest / ARCHIVE_NOTE_FILE).write_text(json.dumps(note, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log(f"archived {len(moved)} derived artifact(s) -> {dest} ({why}): {', '.join(m['name'] for m in moved)}")
    return {"dir": str(dest), "archived": [m["name"] for m in moved]}


def list_reopen_receipts(out: Path) -> List[Dict[str, Any]]:
    """Every receipt under ``<out>/reopen_receipts/`` in every state (:data:`RECEIPT_FILE_RE`), oldest mint first: ``{path, name,
    gen, nonce, state, minted_utc, doc}`` — ``minted_utc`` from the receipt (every mint since re-check 3 writes it), else the
    file's mtime as an ISO stamp; the sort key is (minted_utc, mtime, name) so two mints in one second still order."""
    d = Path(out) / RECEIPTS_DIR
    rows: List[Dict[str, Any]] = []
    if not d.is_dir():
        return rows
    for f in d.iterdir():
        m = RECEIPT_FILE_RE.match(f.name)
        if not m:
            continue
        try:
            st = f.stat()
            raw = f.read_text(encoding="utf-8")
        except FileNotFoundError:
            continue                      # renamed into another state by a concurrent consumer / mint between the listing and the read
        except OSError:
            raw = ""; st = None
        try:
            doc = json.loads(raw) if raw else None
        except ValueError:
            doc = None
        # the ORDER key is the file's mtime at full resolution (a rename keeps it: a .used / .superseded receipt still sorts at its
        # mint); the ISO second is only the minted_utc fallback for a receipt that carries none
        mtime_f = float(st.st_mtime) if st is not None else 0.0
        mtime_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(mtime_f))
        minted = str(doc.get("minted_utc")) if isinstance(doc, Mapping) and doc.get("minted_utc") else mtime_iso
        rows.append({"path": f, "name": f.name, "gen": m.group("gen"), "nonce": m.group("nonce"),
                     "state": m.group("state") or RECEIPT_STATE_ACTIVE, "minted_utc": minted, "_mtime": mtime_f,
                     "doc": (dict(doc) if isinstance(doc, Mapping) else None)})
    rows.sort(key=lambda r: (r["minted_utc"], r["_mtime"], r["name"]))
    return rows


def supersede_stale_receipts(out: Path, *, keep_nonce: str, utc: Optional[str] = None) -> List[str]:
    """Move every ACTIVE receipt of the record other than ``keep_nonce`` aside as ``.superseded.<utc>.json`` (never deleted) —
    what the mint does on the cluster (``lib/slurm.sh slurm_revoke_geometry``) and what the consumer of the current epoch does
    for orphans a pre-epoch mint left behind. Returns the names moved."""
    stamp = (utc or utc_now()).replace(":", "")
    moved: List[str] = []
    for r in list_reopen_receipts(out):
        if r["state"] == RECEIPT_STATE_ACTIVE and r["nonce"] != keep_nonce:
            target = r["path"].with_name(f"{r['gen']}.{r['nonce']}.{RECEIPT_STATE_SUPERSEDED}.{stamp}.json")
            try:
                os.replace(r["path"], target)
                moved.append(r["name"])
            except FileNotFoundError:
                pass                     # a concurrent consumer / mint moved it first
    return moved


def _consumer_identity() -> Dict[str, Any]:
    return {"pid": os.getpid(), "host": socket.gethostname(), "job_id": os.environ.get("SLURM_JOB_ID"), "tree_sha": _tree_sha()}


#: the per-record REOPEN LOCK: `<record>/reopen_receipts/.lock`, flock'ed (LOCK_EX | LOCK_NB) for the WHOLE of a
#: receipted rewrite (consume -> certifying write -> finish) and the WHOLE of a recovery; released by the kernel when the holder dies.
#: A second reverdict / recovery / receipted resume on the same record refuses by name while it is held.
REOPEN_LOCK_FILE = ".lock"
_REOPEN_LOCKS: Dict[str, int] = {}


def owner_token() -> str:
    """This process's OWNER token for an in-progress receipt (the file's tag, dot-free): the utc stamp + pid + host."""
    return f"{utc_now().replace(':', '')}-p{os.getpid()}-{socket.gethostname().split('.')[0].replace('.', '_')}"


def acquire_reopen_lock(out: Path, *, what: str) -> None:
    """Take the record's reopen lock for this process (idempotent); refuse by name when another process holds it, quoting the
    holder it wrote into the lock file (informational; the flock is the guard)."""
    out = Path(out)
    d = out / RECEIPTS_DIR
    d.mkdir(parents=True, exist_ok=True)
    key = str(out.resolve())
    if key in _REOPEN_LOCKS:
        raise SystemExit(f"{TAG} REFUSING {what} of {out}: this process already HOLDS this record's reopen lock for another certifying "
                         "rewrite — one at a time")
    fd = os.open(str(d / REOPEN_LOCK_FILE), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError):
        try:
            holder = os.read(fd, 4096).decode("utf-8", "replace").strip()
        except OSError:
            holder = ""
        os.close(fd)
        raise SystemExit(f"{TAG} REFUSING {what} of {out}: another reopen / recovery / receipted resume HOLDS this record's reopen lock "
                         f"({d / REOPEN_LOCK_FILE}{': ' + holder[:300] if holder else ''}) — one certifying rewrite at a time; when its holder "
                         "is dead the lock is free and `reverdict --resume-interrupted` recovers what it left") from None
    os.ftruncate(fd, 0)
    os.write(fd, (json.dumps({"what": what, "taken_utc": utc_now(), **_consumer_identity()}, sort_keys=True) + "\n").encode("utf-8"))
    _REOPEN_LOCKS[key] = fd


def release_reopen_lock(out: Path) -> None:
    fd = _REOPEN_LOCKS.pop(str(Path(out).resolve()), None)
    if fd is not None:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def in_progress_receipts(out: Path) -> List[Dict[str, Any]]:
    """The record's receipts whose reopen is IN PROGRESS (consumed, certifying rewrite not completed) — normally none or one."""
    return [r for r in list_reopen_receipts(out) if r["state"] == RECEIPT_STATE_INPROGRESS]


def reopen_epoch(nonce: str) -> str:
    """The revocation epoch of a receipt: ``sha256`` of its nonce (hex) — what lib/slurm.sh writes into the REVOKED markers."""
    return hashlib.sha256(str(nonce).encode("utf-8")).hexdigest()


def fold_reopen_epoch(out: Optional[Path]) -> Optional[str]:
    """The epoch the fold now running certifies under: the record's ONE in-progress receipt (consumed by this reverdict / receipted
    resume and not yet finished) -> :func:`reopen_epoch` of its nonce. None when no receipt is in progress (a first verdict; the
    carrier's bare verdict after a run that finished its receipt — that run's results moved the generation, so no same-generation
    marker stands) or when more than one is (an ambiguity the fold never guesses at)."""
    if out is None:
        return None
    try:
        rows = in_progress_receipts(Path(out))
    except OSError:
        return None
    return reopen_epoch(rows[0]["nonce"]) if len(rows) == 1 else None


def _in_progress_summary(r: Mapping[str, Any]) -> str:
    c = (r.get("doc") or {}).get(RECEIPT_CONSUME_KEY) or {}
    return (f"receipt minted {r.get('minted_utc')}, consumed {c.get('started_utc') or '?'} for {c.get('what') or '?'} by pid {c.get('pid') or '?'} "
            f"on {c.get('host') or '?'} (job {c.get('job_id') or '-'})" + (f", resumed {len(c.get('resumed') or [])}x" if c.get("resumed") else ""))


def _write_receipt_doc(path: Path, doc: Mapping[str, Any]) -> None:
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def consume_reopen_receipt(out: Path, token: Optional[str], *, what: str, action: Optional[str] = None) -> Dict[str, Any]:
    """Authorise ``what`` (a resume with work, a reverdict) on a COMPLETE record: the front door's revocation receipt for the
    record's CURRENT result generation carrying ``token`` (the receipt's nonce) must exist ACTIVE under
    ``<out>/reopen_receipts/`` and be the record's CURRENT EPOCH — no receipt minted after it exists in any state (an orphan
    receipt of an earlier revocation must never authorise a later reopen after a newer receipt has recertified and restaged
    the same generation). It is taken by RENAME-AS-LOCK (``.consuming.<tag>.json``, atomic — of two concurrent consumers
    exactly one holds it), older active receipts are moved aside as ``.superseded.<utc>.json``, and the receipt is CONSUMED
    (``.used.<utc>.json``) and returned. Compatible with every receipt format minted so far: no field beyond ``generation`` /
    ``nonce`` / ``acknowledged`` is required; ``minted_utc`` orders the epochs when present, the file's mtime otherwise.
    Every refusal is a :class:`SystemExit` naming the way in — ``reproduce/03_hpo.sh --phase probe --reopen`` — and NEVER
    prints the generation, the nonce or a receipt name."""
    out = Path(out)
    way = ("a reopen goes through the front door: `reproduce/03_hpo.sh --phase probe --reopen` archives the site copy and every "
           "cluster's staged overlay with a REVOKED marker, writes the revocation receipt into this record and hands the carrier "
           "the receipt's nonce")
    if not token or not REOPEN_NONCE_RE.match(str(token)):
        raise SystemExit(f"{TAG} REFUSING {what} of {out}: its checkpoint is COMPLETE (a verdict stands on it and copies of its geometry "
                         f"may be staged on clusters that cannot reach it) and no revocation receipt nonce was given (--reopen-token) — {way}")
    gen_now = str((result_generation(out) or {}).get("generation") or "")
    receipts = list_reopen_receipts(out)
    # a reopen IN PROGRESS (consumed, its certifying rewrite not completed — running, or dead between the archive
    # and the rewrite) is never joined by a second one, under this nonce or any other: it is finished or recovered first
    stuck = [r for r in receipts if r["state"] == RECEIPT_STATE_INPROGRESS]
    if stuck:
        own = any(r["nonce"] == str(token) for r in stuck)
        raise SystemExit(f"{TAG} REFUSING {what} of {out}: a reopen of this record is IN PROGRESS under "
                         f"{'the receipt carrying the given nonce' if own else 'an earlier receipt'} ({_in_progress_summary(stuck[0])}) — its "
                         "certifying rewrite has not completed (a run under way, or one that died between the archive and the rewrite); "
                         f"recover it with `serving_probe.py reverdict --record {out} --resume-interrupted` (the rewrite re-runs under THAT "
                         "receipt's epoch) — never a second consume beside it, never a fresh revocation over it")
    mine = [r for r in receipts if r["gen"] == gen_now and r["nonce"] == str(token)] if gen_now else []
    if not mine and gen_now:
        # a receipt mid-rename by a concurrent consumer (active -> consuming -> used) is absent from one listing for an instant;
        # a second look an instant later sees its final state, so the refusal names it (consumed) rather than "never revoked"
        time.sleep(0.05)
        receipts = list_reopen_receipts(out)
        mine = [r for r in receipts if r["gen"] == gen_now and r["nonce"] == str(token)]
    active = [r for r in mine if r["state"] == RECEIPT_STATE_ACTIVE]
    if not active:
        states = {r["state"] for r in mine}
        if RECEIPT_STATE_INPROGRESS in states or any(x["state"] == RECEIPT_STATE_INPROGRESS for x in receipts):
            stuck = [x for x in receipts if x["state"] == RECEIPT_STATE_INPROGRESS]
            why = (f"a reopen of this record is IN PROGRESS under {'the receipt carrying the given nonce' if RECEIPT_STATE_INPROGRESS in states else 'an earlier receipt'} "
                   f"({_in_progress_summary(stuck[0])}) — recover it with `serving_probe.py reverdict --record {out} --resume-interrupted`; never a second consume")
        elif RECEIPT_STATE_USED in states:
            why = "the receipt carrying the given nonce was ALREADY CONSUMED — a receipt opens a record once"
        elif RECEIPT_STATE_SUPERSEDED in states:
            why = "the receipt carrying the given nonce was SUPERSEDED by a later revocation — a reopen uses the receipt of the LATEST revocation"
        elif RECEIPT_STATE_CONSUMING in states:
            why = "the receipt carrying the given nonce is being consumed by a concurrent run"
        else:
            why = ("no revocation receipt for this record's CURRENT result generation carries the given nonce — the front door never "
                   "revoked for this result set, or the record changed since")
        raise SystemExit(f"{TAG} REFUSING {what} of {out}: {why}; {way}")
    r = active[0]
    doc = r["doc"]
    if doc is None:
        raise SystemExit(f"{TAG} REFUSING {what} of {out}: the revocation receipt is not readable JSON — {way}")
    if str(doc.get("generation")) != gen_now or str(doc.get("nonce")) != str(token) or not doc.get("acknowledged"):
        raise SystemExit(f"{TAG} REFUSING {what} of {out}: the revocation receipt does not match this record's current result set or "
                         f"names no acknowledged cluster — {way}")
    # the CURRENT EPOCH: nothing minted after this receipt may exist in ANY state (a later receipt — consumed, active, or moved aside —
    # means the front door revoked again; this one is an orphan of an earlier revocation and authorises nothing)
    later = [x for x in receipts if x["nonce"] != r["nonce"] and (x["minted_utc"], x["_mtime"], x["name"]) > (r["minted_utc"], r["_mtime"], r["name"])]
    if later:
        raise SystemExit(f"{TAG} REFUSING {what} of {out}: the revocation receipt carrying the given nonce was minted BEFORE a later "
                         f"revocation of this record ({len(later)} later receipt(s), states {sorted({x['state'] for x in later})}) — it is "
                         f"superseded and authorises nothing; a reopen uses the receipt of the LATEST revocation; {way}")
    # rename-as-lock: of two concurrent consumers exactly one wins the rename; the other finds the file gone
    stamp = utc_now().replace(":", "")
    holding = r["path"].with_name(f"{gen_now}.{token}.{RECEIPT_STATE_CONSUMING}.{os.getpid()}-{stamp}.json")
    try:
        os.rename(r["path"], holding)
    except FileNotFoundError:
        raise SystemExit(f"{TAG} REFUSING {what} of {out}: the revocation receipt carrying the given nonce was taken by a concurrent run "
                         f"an instant ago (consumed or superseded) — a receipt opens a record once; {way}") from None
    moved = supersede_stale_receipts(out, keep_nonce=str(token), utc=stamp)
    # the receipt is IN PROGRESS from here until the reopen's certifying rewrite completes (finish_reopen_receipt): the consumer's
    # identity and intent ride the receipt so a death leaves a record that names what was under way
    owner = owner_token()
    doc = {**doc, RECEIPT_CONSUME_KEY: {"started_utc": utc_now(), "what": what, "action": action or what, "owner": owner, **_consumer_identity()}}
    inprog = r["path"].with_name(f"{gen_now}.{token}.{RECEIPT_STATE_INPROGRESS}.{owner}.json")   # the tag IS the owner
    _write_receipt_doc(holding, doc)
    os.replace(holding, inprog)
    log(f"{what}: authorised by the front door's revocation receipt (minted {doc.get('minted_utc')} on {doc.get('minted_host')} after "
        f"{len(doc.get('acknowledged') or [])} cluster(s) acknowledged their REVOKED marker); receipt IN PROGRESS until the rewrite completes"
        + (f"; {len(moved)} older active receipt(s) of earlier revocations moved aside as superseded" if moved else ""))
    return dict(doc)


def _in_progress_owner(r: Mapping[str, Any]) -> str:
    return str(RECEIPT_FILE_RE.match(r["name"]).group("tag"))


def finish_reopen_receipt(out: Path, *, what: str, owner: Optional[str]) -> Optional[Dict[str, Any]]:
    """The reopen's certifying rewrite COMPLETED: the in-progress receipt THIS process owns (``owner``: the token consume /
    resume / claim handed it — the in-progress file's tag) becomes ``.used.<utc>.json`` (its ``consume`` block stamped
    ``completed_utc`` / ``completed_what``). Called at the completing write of every receipted path. ``owner`` None with
    nothing in progress is a no-op (an unreceipted resume); an in-progress receipt owned by SOMEONE ELSE refuses by name — this
    process's rewrite is not the record's: completion proves ownership, never "whichever exists"."""
    out = Path(out)
    stuck = in_progress_receipts(out)
    if not stuck:
        if owner is not None:
            raise SystemExit(f"{TAG} {what} of {out}: this process consumed / claimed the reopen receipt but no in-progress receipt remains — "
                             "another recovery took it over; this rewrite is not the record's; refusing to mark anything used")
        return None
    mine = [r for r in stuck if owner is not None and _in_progress_owner(r) == owner]
    if not mine:
        raise SystemExit(f"{TAG} {what} of {out}: the in-progress receipt is owned by another consumer / recovery "
                         f"({'; '.join(_in_progress_summary(r) for r in stuck)}) — this process holds no claim on it; refusing to mark it used")
    r = mine[0]
    doc = dict(r["doc"] or {})
    doc[RECEIPT_CONSUME_KEY] = {**(doc.get(RECEIPT_CONSUME_KEY) or {}), "completed_utc": utc_now(), "completed_what": what}
    stamp = utc_now().replace(":", "")
    used = r["path"].with_name(f"{r['gen']}.{r['nonce']}.{RECEIPT_STATE_USED}.{stamp}.json")
    _write_receipt_doc(r["path"], doc)
    os.replace(r["path"], used)
    log(f"{what}: the reopen's certifying rewrite completed — receipt consumed")
    return doc


def claim_in_progress_receipt(out: Path, *, what: str) -> Tuple[Dict[str, Any], str]:
    """Take EXCLUSIVE ownership of the record's one in-progress receipt by RENAME (atomic: of two claimants exactly one wins; the
    other finds the file gone and refuses) — the recovery's adoption and a successor attempt's take-over of a receipt an earlier
    attempt consumed. Requires exactly one in-progress receipt, of the record's CURRENT generation. Returns ``(doc, owner)``."""
    out = Path(out)
    stuck = in_progress_receipts(out)
    if not stuck:
        raise SystemExit(f"{TAG} {what}: no reopen of {out} is in progress (no in-progress receipt) — nothing to resume; a fresh reopen goes "
                         "through the front door (`03_hpo.sh --phase probe --reopen`)")
    if len(stuck) > 1:
        raise SystemExit(f"{TAG} {what}: {len(stuck)} in-progress receipts under {out} — an operator must decide which reopen stands; refusing")
    r = stuck[0]
    gen_now = str((result_generation(out) or {}).get("generation") or "")
    if r["gen"] != gen_now:
        raise SystemExit(f"{TAG} {what}: the in-progress receipt is for another result generation than the record's current one — the "
                         "results moved under the interrupted reopen; a fresh reopen goes through the front door")
    owner = owner_token()
    claimed = r["path"].with_name(f"{r['gen']}.{r['nonce']}.{RECEIPT_STATE_INPROGRESS}.{owner}.json")
    try:
        os.rename(r["path"], claimed)
    except FileNotFoundError:
        raise SystemExit(f"{TAG} {what} of {out}: the in-progress receipt was claimed by a concurrent recovery an instant ago — one recovery "
                         "at a time; refusing") from None
    doc = dict(r["doc"] or {})
    c = dict(doc.get(RECEIPT_CONSUME_KEY) or {})
    c["resumed"] = list(c.get("resumed") or []) + [{"utc": utc_now(), "what": what, "owner": owner, "took_over_from": _in_progress_owner(r), **_consumer_identity()}]
    c["owner"] = owner
    doc[RECEIPT_CONSUME_KEY] = c
    _write_receipt_doc(claimed, doc)
    return doc, owner


def resume_interrupted_receipt(out: Path, *, what: str) -> Tuple[Dict[str, Any], str]:
    """The guarded recovery: under the record's reopen lock (the caller holds it), CLAIM the one
    in-progress receipt — the reopen a process consumed and then failed to finish — so the certifying rewrite re-runs under THAT
    receipt's epoch with THIS process as its owner: no second consume, no fresh mint, no shared write. Returns ``(doc, owner)``."""
    doc, owner = claim_in_progress_receipt(out, what=what)
    log(f"{what}: resuming the INTERRUPTED reopen under its receipt (minted {doc.get('minted_utc')}, first consumed "
        f"{(doc.get(RECEIPT_CONSUME_KEY) or {}).get('started_utc')}) — the certifying rewrite re-runs under that epoch; no new revocation")
    return doc, owner


def replan_record(out: Path, plan: Mapping[str, Any], results: Mapping[str, Mapping[str, Any]],
                  certify_v2: Sequence[Tuple[int, str]] = ()) -> Dict[str, Any]:
    """``run --resume --replan``: measure the budget from the record, re-derive S for the outstanding and the
    ``replan_required`` launches, move the latter's results aside (superseded, never deleted), write the plan back.
    With NO measured floor nothing is re-derived (there is nothing to derive from) — logged loudly, recorded on the
    plan as ``unmeasured``. Idempotent: a second call on the same record changes nothing and moves nothing."""
    out = Path(out)
    budget = measured_budget(plan, results)
    if budget.get("tokens") is None:
        log(f"--replan: budget UNMEASURED — no passed launch in the record measures a floor; S is NOT re-derived "
            f"(every recorded start-time OOM above S = 1 stays replan_required until one launch comes up)")
        return {"budget": budget, "changes": [], "superseded": [], "replan_required": replan_required_launches(plan, results, budget),
                "plan_written": False}
    replan = replan_required_launches(plan, results, budget)
    # the reopen EXTENDS the plan with the current topology's servers for every
    # pair whose serving is uncertified (or buffer-starved) before S is re-derived — the extension's launches are
    # outstanding, so the re-derivation below prices them under the measured budget
    plan, ext = extend_plan_for_uncertified_pairs(plan, results, _Dims(plan["dims"]), tree_sha=_tree_sha())
    if ext["added"]:
        log(f"--replan: plan EXTENDED with {len(ext['added'])} launch(es) over {len(ext['manifests'])} new manifest(s) for "
            f"{ext['pairs']} (the current topology at the uncertified / buffer-starved windows): {ext['added'][:12]}"
            + (" …" if len(ext["added"]) > 12 else ""))
    v2_moved: List[Dict[str, str]] = []
    if certify_v2:
        plan, v2w = certify_v2_work(plan, results, _Dims(plan["dims"]), list(certify_v2), tree_sha=_tree_sha())
        if v2w["supersede"]:
            v2_moved = supersede_launch_results(out, v2w["supersede"])
            results = load_results(out)
        log(f"--certify-v2 {v2w['pairs']}: {len(v2w['added'])} launch(es) planned, {len(v2w['supersede'])} v1-only result(s) superseded to "
            f"re-run under the certificate's prompts: {sorted(set(v2w['added']) | set(v2w['supersede']))}")
        ext = {**ext, "certify_v2": v2w}
    new_plan, changes = replan_launches(plan, results, budget)
    moved = supersede_launch_results(out, replan) + v2_moved
    # the plan file moves only on NEW information: an S that changed, an admission record that gained the budget terms,
    # a superseded launch, an extension, or a budget the plan did not carry yet — a second --replan on the same record
    # writes nothing
    dirty = (new_plan["launches"] != list(plan["launches"]) or bool(replan) or bool(ext["added"]) or bool(v2_moved)
             or "certify_v2" in ext and bool(ext["certify_v2"]["added"])
             or (plan.get("batched_token_budget") or {}).get("tokens") != budget["tokens"]
             or (plan.get("batched_token_budget") or {}).get("method") != budget["method"])
    if dirty:
        new_plan["replanned_utc"] = utc_now()
        write_json_atomic(out / PLAN_FILE, new_plan)
    log(f"--replan: budget {budget['method']} = {budget['tokens']:,} tokens (floor {budget['floor_launch']}"
        + (f"; ceiling {budget['ceiling_tokens']:,} at {budget['ceiling_launch']}" if budget.get("ceiling_tokens") else "")
        + f"); {len(changes)} launch(es) re-derived, {len(replan)} superseded and re-run: "
        + ", ".join(f"{c['id']} S {c['old_S']}->{c['new_S']} ({c['binding']})" for c in changes[:12])
        + (" …" if len(changes) > 12 else ""))
    return {"budget": budget, "changes": changes, "superseded": moved, "replan_required": replan, "plan_written": dirty,
            "extended": ext}


def progress_record(plan: Mapping[str, Any], results: Mapping[str, Mapping[str, Any]], *,
                    stopped_reason: Optional[str] = None, generation: Optional[Mapping[str, Any]] = None,
                    attempt: Optional[Mapping[str, Any]] = None, hold: Optional[str] = None) -> Dict[str, Any]:
    """The launch-list checkpoint (:data:`PROGRESS_FILE`): done / outstanding
    launch ids (the decision-relevant ones named separately), ``complete``
    when nothing is outstanding, and why the last run stopped. ``hold``:
    a phase that rewrites RESULTS of a finished launch list — the
    reference-only re-run — names itself here and the checkpoint is NOT
    complete while it stands, whatever the launch list says; the
    ``stopped_reason`` carries it when none is given, so a reader of the
    record sees why it is not complete."""
    outstanding = outstanding_launches(plan, results)
    done = [rec["id"] for rec in plan["launches"]
            if (results.get(rec["id"]) or {}).get("status") in TERMINAL_STATUSES]
    budget = measured_budget(plan, results)
    replan = replan_required_launches(plan, results, budget)
    return {"version": 1, "written_utc": utc_now(), "model_key": plan["model_key"], "plan_launches": plan["n_launches"],
            "n_done": len(done), "done": done,
            "outstanding": [{"id": r["id"], "kind": r["kind"], "L": r["L"], "cap": r["cap"]} for r in outstanding],
            "decision_outstanding": [r["id"] for r in outstanding if r["kind"] in DECISION_KINDS],
            # a start-time OOM above the measured budget is NOT done — its S must be re-derived and re-run
            "replan_required": replan,
            "batched_token_budget": {k: budget.get(k) for k in ("method", "tokens", "floor_launch", "ceiling_tokens", "ceiling_launch")},
            "plan_batched_token_budget": (plan.get("batched_token_budget") or {}).get("tokens"),
            # the record's result generation at this write (verified by hpo_probe_status against the verdict / menu stamps)
            RESULT_GENERATION_KEY: (generation or {}).get("generation"), "n_results": (generation or {}).get("n_results"),
            # a resumed attempt writes this BEFORE any GPU work: the checkpoint says a driver holds the record
            "attempt": (dict(attempt) if attempt else None),
            # a hold (a results-rewriting phase in flight) keeps the checkpoint incomplete until that phase's completing write
            "hold": hold,
            "complete": bool(not outstanding and not replan and hold is None), "stopped_reason": stopped_reason or hold}


def write_progress(plan: Mapping[str, Any], results: Mapping[str, Mapping[str, Any]], out: Path, *,
                   stopped_reason: Optional[str] = None, attempt: Optional[Mapping[str, Any]] = None,
                   hold: Optional[str] = None) -> Dict[str, Any]:
    rec = progress_record(plan, results, stopped_reason=stopped_reason, generation=result_generation(Path(out)), attempt=attempt, hold=hold)
    write_json_atomic(Path(out) / PROGRESS_FILE, rec)
    return rec


def load_progress(out: Path) -> Optional[Dict[str, Any]]:
    p = Path(out) / PROGRESS_FILE
    if not p.is_file():
        return None
    rec = json.loads(p.read_text(encoding="utf-8"))
    return dict(rec) if isinstance(rec, Mapping) else None


def _launch_identity(rec: Mapping[str, Any]) -> Dict[str, Any]:
    """The plan launch's identity fields every result record carries."""
    return {"launch_id": rec["id"], "manifest": rec["manifest"], "serves": list(rec["serves"]),
            "set": rec["set"], "rank": rec["rank"], "topology": rec["topology"], "cap": rec["cap"],
            "L": rec["L"], "tp": rec["tp"], "kind": rec["kind"]}


def run_one_launch(plan: Mapping[str, Any], rec: Mapping[str, Any], out: Path, deps: Deps, *,
                   gpu_mem_util: float, ready_timeout_s: float, throughput: bool,
                   devices: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Stand the launch up, observe, evaluate, tear down. Returns the result record.
    ``devices``: the standup's pinned device subset (a tuple of Slurm's
    ``CUDA_VISIBLE_DEVICES`` tokens) or None for an unpinned single standup;
    it reaches ``deps.stand_up`` / ``nvidia_smi`` / ``wait_gpu_free`` and is
    recorded on the result (``devices``)."""
    out = Path(out)
    dims = _Dims(plan["dims"])
    manifest = plan_manifest(plan, rec["manifest"])
    lid = rec["id"]
    ldir = out / LAUNCHES_DIR / lid
    ldir.mkdir(parents=True, exist_ok=True)
    devices = tuple(str(d) for d in devices) if devices is not None else None
    result: Dict[str, Any] = {**_launch_identity(rec), "requested_S": rec["requested_S"],
                              "admitted_S": rec["admitted_S"], "started_utc": utc_now(), "dir": str(ldir),
                              "host": socket.gethostname(), "status": STATUS_ERROR,
                              "devices": (list(devices) if devices is not None else None)}
    missing = missing_adapter_dirs(out, manifest)
    if missing:
        result.update(status=STATUS_REFUSED, failure={"class": "adapters_missing",
                                                       "message": f"refused:adapters not exported for {missing}"},
                      finished_utc=utc_now())
        return result
    t0 = deps.now()
    launch = proc = None
    # a plan-estimated S < 1 (the KV budget after the buffers is gone) is still
    # ATTEMPTED at S = 1: the engine's own memory profile, not the estimate,
    # decides the pass / oom — the planned 0 stays on the record
    launch_S = max(1, int(rec["admitted_S"]))
    result["launch_S"] = launch_S
    try:
        launch, proc = deps.stand_up(model_id=plan["model_id"], revision=plan["revision"],
                                     adapter_dirs=manifest_adapter_dirs(out, manifest), manifest=manifest,
                                     L=int(rec["L"]), S=launch_S, out_dir=ldir,
                                     gpu_mem_util=gpu_mem_util, ready_timeout_s=ready_timeout_s, tp=int(rec["tp"]),
                                     devices=devices)
    except Exception as e:  # noqa: BLE001
        text = deps.read_log(ldir / SERVER_LOG_FILE)
        status, msg = classify_failure(text, e)
        result.update(status=status, ready_s=None, finished_utc=utc_now(),
                      failure={"class": status, "message": (msg if status != STATUS_REFUSED or msg.startswith("refused:")
                                                            else f"refused:{msg}"),
                               "exception": f"{type(e).__name__}: {e}"[:2000], "log_tail": text[-4000:]})
        try:
            result["gpu_drain"] = deps.wait_gpu_free(devices=devices)
        except Exception:  # noqa: BLE001
            pass
        return result
    result["ready_s"] = deps.now() - t0
    try:
        log_text = deps.read_log(ldir / SERVER_LOG_FILE)
        models_listed = deps.list_models(launch)
        base_probe = parse_logprob_rows(deps.chat(launch, chat_probe_body(launch.model, PROBE_PROMPT,
                                                                          max_tokens=PROBE_MAX_TOKENS)))
        adapter_probes: Dict[str, Any] = {}
        adapter_decodes: Dict[str, List[Dict[str, Any]]] = {}
        for a in manifest.adapters:
            try:
                adapter_probes[a.name] = parse_logprob_rows(
                    deps.chat(launch, chat_probe_body(a.name, PROBE_PROMPT, max_tokens=PROBE_MAX_TOKENS)))
            except Exception as e:  # noqa: BLE001
                adapter_probes[a.name] = {"rows": [], "token_ids": None, "error": f"{type(e).__name__}: {e}"[:500]}
            if a.kind == sm.KIND_GENUINE or a.kind == sm.KIND_ZERO:
                decs = []
                for p in DECODE_PROMPTS:
                    try:
                        decs.append(parse_logprob_rows(deps.chat(launch, chat_probe_body(
                            a.name, ({"role": "user", "content": p},), max_tokens=DECODE_MAX_TOKENS))))
                    except Exception as e:  # noqa: BLE001
                        decs.append({"rows": [], "token_ids": None, "error": f"{type(e).__name__}: {e}"[:300]})
                adapter_decodes[a.name] = decs
        # PARITY CERTIFICATE v2: the certificate's prompts 1..P-1 for the base and every GENUINE adapter (prompt 0 IS the
        # v1 probe above), first-token top-5 each — cheap 8-token calls, so every launch carries the engine side of v2
        prompts_v2 = plan_parity_prompts(plan)
        probes_v2: Dict[str, Any] = {"prompts_sha": parity_prompts_sha(prompts_v2), "n_prompts": len(prompts_v2), "base": [base_probe],
                                     "adapters": {}}
        for pr in prompts_v2[1:]:
            try:
                probes_v2["base"].append(parse_logprob_rows(deps.chat(launch, chat_probe_body(launch.model, pr, max_tokens=PROBE_MAX_TOKENS))))
            except Exception as e:  # noqa: BLE001
                probes_v2["base"].append({"rows": [], "token_ids": None, "error": f"{type(e).__name__}: {e}"[:300]})
        for a in manifest.adapters:
            if a.kind != sm.KIND_GENUINE:
                continue
            rows_a = [adapter_probes.get(a.name) or {"rows": [], "token_ids": None}]
            for pr in prompts_v2[1:]:
                try:
                    rows_a.append(parse_logprob_rows(deps.chat(launch, chat_probe_body(a.name, pr, max_tokens=PROBE_MAX_TOKENS))))
                except Exception as e:  # noqa: BLE001
                    rows_a.append({"rows": [], "token_ids": None, "error": f"{type(e).__name__}: {e}"[:300]})
            probes_v2["adapters"][a.name] = rows_a
        worker_report = deps.worker_report(launch, ldir)
        residency = deps.nvidia_smi(devices=devices)
        checks = evaluate_checks(manifest=manifest, launch_rec=rec, dims=dims, log_text=log_text,
                                 models_listed=models_listed, base_probe=base_probe, adapter_probes=adapter_probes,
                                 worker_report=worker_report, is_moe=bool(plan.get("is_moe")),
                                 adapter_decodes=adapter_decodes)
        result["engine"] = {"max_num_seqs_requested": rec["requested_S"], "max_num_seqs_admitted": rec["admitted_S"],
                            "max_num_seqs_engine": worker_report.get("max_num_seqs"),
                            "max_num_batched_tokens": worker_report.get("max_num_batched_tokens"),
                            "max_lora_rank": worker_report.get("lora_max_lora_rank"),
                            "max_loras": worker_report.get("lora_max_loras"),
                            "lora_target_modules": worker_report.get("lora_target_modules"),
                            "composed_cmd": list(getattr(launch, "cmd", ())), "nvidia_smi": residency}
        result["checks"] = checks
        result["probes"] = {"base": base_probe, "adapters": adapter_probes}
        # the v2 engine side, reduced to first tokens (the full rows stay out of the result: 8 x (1 + genuine) probes)
        result["probes_v2"] = {"prompts_sha": probes_v2["prompts_sha"], "n_prompts": probes_v2["n_prompts"],
                               "base": [_first_token_of(b) for b in probes_v2["base"]],
                               "adapters": {n: [_first_token_of(r) for r in rows] for n, rows in probes_v2["adapters"].items()},
                               "moved": {n: [first_token_delta(b, r) for b, r in zip(probes_v2["base"], rows)]
                                         for n, rows in probes_v2["adapters"].items()}}
        result["decodes"] = {n: [{"token_ids": d.get("token_ids"), "text": d.get("text"),
                                  "finish_reason": d.get("finish_reason")} for d in decs]
                             for n, decs in adapter_decodes.items()}
        result["worker_report"] = {k: v for k, v in worker_report.items()
                                   if k not in ("lora_execution_module_strides",)}
        result[SERVING_VENV_KEY] = serving_venv_block(source=IDENTITY_SOURCE_STANDUP)        # The venv this standup ran in
        result["worker_report"]["lora_execution_module_strides_n"] = len(worker_report.get(
            "lora_execution_module_strides") or {})
        if throughput and int(rec["L"]) in THROUGHPUT_WINDOWS and rec["kind"] == "natural" and _carries_control(manifest):
            try:
                result["throughput"] = deps.throughput(launch, concurrencies=THROUGHPUT_CONCURRENCIES)
            except Exception as e:  # noqa: BLE001
                result["throughput_error"] = f"{type(e).__name__}: {e}"[:500]
        # the reference-check inputs (run later, engine down)
        result["reference_pending"] = [
            {"dir": str(slot_dir(out, manifest.name, a.name)), "name": a.name, "kind": a.kind, "role": a.role,
             "rank": a.rank, "per_expert_rank": a.per_expert_rank, "lora_modules": list(a.lora_modules),
             "wraps_experts": manifest.wraps_experts,
             "vllm_first_token": _first_token_of(adapter_probes.get(a.name) or {}),
             # how far the ENGINE moved the first-token top-5 off the base (the parity pass's weak-effect gate)
             "vllm_moved": ((checks.get("adapters") or {}).get(a.name) or {}).get("first_token"),
             # PARITY CERTIFICATE v2: the engine's first token per prompt, its base's, and the per-prompt movement
             "vllm_first_tokens_v2": result["probes_v2"]["adapters"].get(a.name),
             "vllm_base_first_tokens_v2": result["probes_v2"]["base"],
             "vllm_moved_v2": result["probes_v2"]["moved"].get(a.name),
             "parity_prompts_sha": result["probes_v2"]["prompts_sha"]}
            for a in manifest.adapters if a.kind == sm.KIND_GENUINE]
        result["status"] = STATUS_PASSED if checks["pass"] else STATUS_FAILED
    except Exception as e:  # noqa: BLE001
        result.update(status=STATUS_ERROR, failure={"class": "observe_error", "message": f"{type(e).__name__}: {e}"[:2000],
                                                    "log_tail": deps.read_log(ldir / SERVER_LOG_FILE)[-4000:]})
    finally:
        try:
            deps.shut_down(launch, proc)
        except Exception:  # noqa: BLE001
            pass
        try:
            result["gpu_drain"] = deps.wait_gpu_free(devices=devices)
        except Exception:  # noqa: BLE001
            pass
    result["elapsed_s"] = deps.now() - t0
    result["finished_utc"] = utc_now()
    return result


def _first_token_of(probe: Mapping[str, Any]) -> Dict[str, Any]:
    ids = probe.get("token_ids") or []
    rows = probe.get("rows") or []
    return {"id": (int(ids[0]) if ids else None), "logprob": (rows[0].get("logprob") if rows else None),
            "token": (rows[0].get("token") if rows else None)}


def run_launches(plan: Mapping[str, Any], out: Path, deps: Deps, *, gpu_mem_util: float, ready_timeout_s: float,
                 time_budget_s: Optional[float], max_launches: Optional[int], throughput: bool,
                 only_ids: Optional[Iterable[str]] = None, parallel: int = 1,
                 device_slots: Optional[Sequence[Sequence[str]]] = None) -> Dict[str, Dict[str, Any]]:
    """Run the plan's outstanding launches, ``parallel`` at a time.

    Slot k of ``device_slots`` (from :func:`resolve_parallel`: disjoint strict
    subsets of Slurm's ``CUDA_VISIBLE_DEVICES``) is the device subset every
    launch started in that slot is pinned to; ``None`` slots (no device list,
    ``parallel`` 1) leave the engine unpinned — the single-standup behaviour.
    Launches START in the plan's priority order (:func:`next_launch`, the
    ones in flight excluded; a fallback waits for its parent's verdict), the
    ``max_launches`` / ``time_budget_s`` checks run before each START exactly
    as before, and once the budget stops the run the launches already in
    flight are drained, never killed. Every finished launch's result file AND
    the launch-list checkpoint are written under ONE lock as it completes, so
    a preemption mid-run loses at most the launches in flight and ``--resume``
    re-runs exactly those. A worker dying on a BaseException (a kill) lets it
    propagate after the executor is abandoned: whatever finished is already on
    disk."""
    out = Path(out)
    results = load_results(out)
    t_start = deps.now()
    n_started = 0
    only = set(only_ids) if only_ids is not None else None
    plan_view = dict(plan)
    if only is not None:
        plan_view["launches"] = [r for r in plan["launches"] if r["id"] in only]
    parallel = int(parallel)
    if parallel < 1:
        raise ValueError(f"{TAG} parallel must be >= 1, got {parallel}")
    if device_slots is not None:
        slots: List[Optional[Tuple[str, ...]]] = [tuple(str(d) for d in sl) for sl in device_slots]
        if len(slots) < parallel:
            # never widened: N standups need N disjoint subsets of the allocation, or fewer standups
            raise ValueError(f"{TAG} parallel={parallel} but only {len(slots)} device slot(s) {slots} — refusing to run "
                             "two standups on one device subset")
        slots = slots[:parallel]
        flat = [d for sl in slots for d in sl]
        if len(set(flat)) != len(flat):
            raise ValueError(f"{TAG} device slots overlap: {slots}")
    else:
        slots = [None] * parallel
    lock = threading.Lock()   # ONE lock over results + the per-launch result file + probe_progress.json
    stopped: Optional[str] = None
    in_flight: Dict[Any, Tuple[str, int]] = {}
    free_slots: List[int] = list(range(parallel))
    def record(lid: str, res: Mapping[str, Any]) -> None:
        with lock:
            results[lid] = dict(res)
            write_json_atomic(launch_result_path(out, lid), res)
            # the launch-list checkpoint rides every finished launch: a job that dies here resumes from exactly this point
            write_progress(plan, results, out)

    if parallel > 1:
        log(f"parallel: {parallel} concurrent standup(s), device slots "
            + ", ".join(("[" + ",".join(sl) + "]") if sl else "[unpinned]" for sl in slots))
    pool = ThreadPoolExecutor(max_workers=parallel, thread_name_prefix="probe-launch")
    try:
        while True:
            while free_slots and stopped is None:
                with lock:
                    snapshot = dict(results)
                rec = next_launch(plan_view, snapshot, exclude=[lid for lid, _ in in_flight.values()])
                if rec is None:
                    break
                if rec.get("_skip"):
                    res = {**_launch_identity(rec), "status": STATUS_SKIPPED,
                           "reason": f"parent {rec['condition']['launch']} did not fail", "finished_utc": utc_now()}
                    record(rec["id"], res)
                    continue
                if max_launches is not None and n_started >= max_launches:
                    log(f"max-launches {max_launches} reached; stopping before {rec['id']}")
                    stopped = "max_launches"
                    break
                if time_budget_s is not None and (deps.now() - t_start) > time_budget_s:
                    log(f"time budget {time_budget_s:.0f}s exhausted; stopping before {rec['id']}")
                    stopped = "time_budget"
                    break
                slot = free_slots.pop(0)
                devs = slots[slot]
                log(f"launch {rec['id']}{device_tag(devs)}: {rec['kind']} class={','.join(rec['target_class'])} "
                    f"cap={rec['cap']} L={rec['L']} S={rec['admitted_S']} (requested {rec['requested_S']}, "
                    f"{rec['admission']['binding']})"
                    + (f"; {len(in_flight)} other launch(es) in flight" if in_flight else ""))
                fut = pool.submit(run_one_launch, plan, rec, out, deps, gpu_mem_util=gpu_mem_util,
                                  ready_timeout_s=ready_timeout_s, throughput=throughput, devices=devs)
                in_flight[fut] = (rec["id"], slot)
                n_started += 1
            if not in_flight:
                break
            done, _pending = _wait_futures(list(in_flight), return_when=FIRST_COMPLETED)
            for fut in done:
                lid, slot = in_flight.pop(fut)
                res = fut.result()   # a worker's BaseException (a kill) propagates; the finished launches are on disk
                record(lid, res)
                free_slots.append(slot)
                free_slots.sort()
                log(f"launch {lid}{device_tag(slots[slot])}: {res['status']}"
                    + (f" — {res['failure']['message'][:200]}" if res.get("failure") else "")
                    + (f" checks failed {res['checks']['failed']}" if res.get("checks") and res["checks"].get("failed") else "")
                    + (f"; {len(in_flight)} still in flight" if in_flight else ""))
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown(wait=True)
    prog = write_progress(plan, results, out, stopped_reason=stopped)
    if not prog["complete"]:
        log(f"run stopped ({stopped or 'narrowed'}): {len(prog['outstanding'])} planned launch(es) outstanding, "
            f"{len(prog['decision_outstanding'])} decision-relevant — `run --resume --out {out}` continues them")
    return results


def reference_phase(plan: Mapping[str, Any], out: Path, deps: Deps, *, redo: bool = False) -> Dict[str, Any]:
    """Assertion (4) over every passed launch's genuine adapters (deduplicated
    by adapter dir), writing ``reference_check`` into each launch result.

    A launch whose adapters were all COMPARED carries ``{status: pass|fail, basis: hf_peft}``; a launch with ANY
    adapter the harness could not judge (``reference_missing`` — not attached, layout disagreement, identical
    outputs — or ``error``) carries ``{status: reference_missing, basis: None, reason}``: the verdict reads it as no
    evidence (UNCERTIFIED, no menu), never as a mismatch. ``redo`` (``run --resume --reference-only``) archives
    the prior ``reference_check.json`` and re-judges EVERY adapter instead of keeping an earlier attempt's verdicts."""
    out = Path(out)
    results = load_results(out)
    # a resumed run keeps the adapters an earlier attempt already judged (reference_check.json); only the new
    # ones go through the HF+PEFT load (resume, not redo)
    prior: Dict[str, Any] = {}
    prior_path = out / REFERENCE_CHECK_FILE
    if redo and prior_path.is_file():
        archive_derived_artifacts(out, why="run --resume --reference-only: the HF+PEFT reference phase is re-run over every adapter; "
                                           "the earlier attempt's judgements are history, not evidence",
                                  only=[REFERENCE_CHECK_FILE], overlay=False)
    if prior_path.is_file():
        try:
            prior_rec = json.loads(prior_path.read_text(encoding="utf-8"))
        except ValueError:
            prior_rec = {}
        if isinstance(prior_rec, Mapping) and prior_rec.get("status") == "done":
            prior = dict(prior_rec)
    prior_per = dict(prior.get("per_adapter") or {})
    pending: Dict[str, Dict[str, Any]] = {}
    for res in results.values():
        if res.get("status") not in (STATUS_PASSED, STATUS_FAILED):
            continue
        # the engine's own BASE probe rides along: the reference measures its floor against it (no adapter on either side)
        base_rows = ((res.get("probes") or {}).get("base") or {})
        vllm_base = _first_token_of(base_rows) if base_rows else None
        lid = str(res.get("launch_id") or "")
        for item in res.get("reference_pending") or []:
            prior_v = prior_per.get(item["dir"])
            # v2 is judged PER LAUNCH: an adapter whose prior record lacks THIS launch's v2 is pending again
            needs_v2 = bool(item.get("vllm_first_tokens_v2")) and isinstance(prior_v, Mapping) \
                and prior_v.get("status") in (REF_STATUS_PASS, REF_STATUS_FAIL, REF_STATUS_UNDECIDABLE) \
                and lid not in (prior_v.get("v2_by_launch") or {})
            if item["dir"] not in prior_per or needs_v2:
                if item["dir"] not in pending:
                    it = dict(item)
                    it["launch_id"] = lid
                    if vllm_base and vllm_base.get("id") is not None:
                        it.setdefault("vllm_base_first_token", vllm_base)
                    moved = (((res.get("checks") or {}).get("adapters") or {}).get(item.get("name")) or {}).get("first_token")
                    if moved is not None:
                        it.setdefault("vllm_moved", moved)
                    it["engine_v2_by_launch"] = {}
                    pending[item["dir"]] = it
    # second pass: EVERY launch that served a pended adapter contributes its own engine side (adapter + base probes, movement,
    # prompt-set sha, venv) — the HF side is measured once per adapter, the floor and the judgement per launch
    for res in results.values():
        if res.get("status") not in (STATUS_PASSED, STATUS_FAILED):
            continue
        lid = str(res.get("launch_id") or "")
        for item in res.get("reference_pending") or []:
            it = pending.get(item["dir"])
            if it is None or not item.get("vllm_first_tokens_v2"):
                continue
            it["engine_v2_by_launch"][lid] = {"adapter": item.get("vllm_first_tokens_v2"), "base": item.get("vllm_base_first_tokens_v2"),
                                              "moved": item.get("vllm_moved_v2"), "sha": item.get("parity_prompts_sha"),
                                              SERVING_VENV_KEY: res.get(SERVING_VENV_KEY)}
    if not pending and not prior:
        return {"status": "nothing_pending"}
    if pending:
        try:
            try:
                rec = deps.reference(model_id=plan["model_id"], revision=plan["revision"], pending=list(pending.values()),
                                     out=out, plan=plan)
            except TypeError as e:
                if "plan" not in str(e):
                    raise
                rec = deps.reference(model_id=plan["model_id"], revision=plan["revision"], pending=list(pending.values()), out=out)
        except Exception as e:  # noqa: BLE001
            # a reference-phase crash is RECORDED (reference_check.json status=error) so the verdict refuses certification
            # by name instead of a traceback losing the evidence of what happened
            rec = {"status": "error", "reason": f"{type(e).__name__}: {e}"[:2000], "per_adapter": {}}
            log(f"reference phase FAILED: {rec['reason']}")
        if not isinstance(rec, Mapping):
            rec = {"status": "error", "reason": f"reference phase returned {type(rec).__name__}, not a record", "per_adapter": {}}
        if rec.get("status") == "done" and prior:
            merged_per = dict(prior_per)
            for d_, v_ in (rec.get("per_adapter") or {}).items():
                pv = prior_per.get(d_)
                if isinstance(pv, Mapping) and isinstance(pv.get("v2_by_launch"), Mapping) and isinstance(v_, Mapping):
                    # a launch judged earlier whose engine side is no longer in the results keeps its v2; a re-judged one is replaced
                    v_ = {**v_, "v2_by_launch": {**pv["v2_by_launch"], **(v_.get("v2_by_launch") or {})}}
                merged_per[d_] = v_
            rec = {**prior, **rec, "per_adapter": merged_per,
                   "layouts": {**(prior.get("layouts") or {}), **(rec.get("layouts") or {})}}
            for k in ("base_floor_v2_by_launch", "base_d_v2_by_launch"):
                if isinstance(prior.get(k), Mapping) or isinstance(rec.get(k), Mapping):
                    rec[k] = {**(prior.get(k) or {}), **(rec.get(k) or {})}
    else:
        rec = prior
    if pending or not prior_path.is_file():
        write_json_atomic(out / REFERENCE_CHECK_FILE, rec)
    per = rec.get("per_adapter") or {}
    judged_now = set(pending)
    for lid, res in results.items():
        if res.get("status") not in (STATUS_PASSED, STATUS_FAILED):
            continue
        if rec.get("status") != "done":
            # no HF+PEFT evidence: recorded as such with NO basis — the verdict never substitutes the vLLM-side checks
            res["reference_check"] = {"status": str(rec.get("status") or "unavailable"), "reason": rec.get("reason"),
                                      "basis": None}
        else:
            prior_rc = res.get("reference_check")
            if isinstance(prior_rc, Mapping) and prior_rc.get("basis") == BASIS_HF_PEFT \
                    and not any(it.get("dir") in judged_now for it in res.get("reference_pending") or []):
                # the result files are the EVIDENCE and the record's result generation is their digest: a launch whose adapters were
                # all judged before (a resume that pends nothing, a re-run of the same slots) keeps its file byte for byte — the
                # verdict re-reads every stored check under the current rule in memory (regate_reference_results). Job 819493
                # (2026-10-01) rewrote 18 untouched results with one new key and moved a COMPLETE record's generation without a
                # receipt, which read the main record PARITY_STALE for nothing
                continue
            eng, eng_refusal = engine_side_block(out, {**res, "model_key": plan.get("model_key")})
            res["reference_check"] = launch_reference_check(res.get("reference_pending") or [], per, layouts=rec.get("layouts"),
                                                            reference_class=rec.get("reference_class"), launch_id=str(lid),
                                                            engine_venv=eng)
            if eng_refusal:
                res["reference_check"]["engine_provenance_refusal"] = eng_refusal
        write_json_atomic(launch_result_path(out, lid), res)
    return rec


def launch_reference_check(pending_items: Sequence[Mapping[str, Any]], per: Mapping[str, Any], *,
                           layouts: Optional[Mapping[str, Any]] = None, reference_class: Optional[str] = None,
                           launch_id: Optional[str] = None, engine_venv: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """One launch's ``reference_check`` from the phase's per-adapter records (pure). Every adapter COMPARED (pass / fail)
    -> ``{status: pass|fail, basis: hf_peft}``; any adapter the harness could not judge (``reference_missing``,
    ``error``, or absent from the record) -> ``{status: reference_missing, basis: None, reason}`` — the verdict then
    files the pair as `reference_missing` (UNCERTIFIED), never as `reference_mismatch`."""
    per_here: Dict[str, Any] = {}
    for item in pending_items:
        v = per.get(item["dir"], {"status": "missing"})
        if isinstance(v, Mapping) and v.get("status") in (REF_STATUS_PASS, REF_STATUS_FAIL, REF_STATUS_UNDECIDABLE):
            # v2 is THIS launch's: the adapter record holds one v2 per launch that served it; a launch that
            # probed the certificate's prompts but has no judgement of its own is undecidable here, never another launch's pass
            by_launch = v.get("v2_by_launch") if isinstance(v.get("v2_by_launch"), Mapping) else {}
            v1 = {k: val for k, val in v.items() if k not in ("v2", "v2_by_launch", "v1", "certificate_version")}
            # a launch that probed no certificate prompts reads v1 alone, whatever other launches judged for this adapter
            v2 = by_launch.get(str(launch_id)) if (launch_id is not None and item.get("vllm_first_tokens_v2")) else None
            if v2 is None and isinstance(v.get("v2"), Mapping) and launch_id is not None and item.get("vllm_first_tokens_v2") \
                    and str(v["v2"].get("launch_id")) == str(launch_id):
                v2 = v["v2"]          # an already-composed record of THIS launch (the re-gate's output); a v2 without a launch id is never borrowed
            # the phase judged v2 for OTHER launches (or a borrowed one) but not this one: a gap, undecidable by name; a record with no
            # v2 information at all was judged under v1 alone (an older v1-only reference) and reads v1 — the menu rule's "else v1" — until
            # the re-pend rule (:func:`reference_phase`) judges this launch's certificate
            has_v2_info = isinstance(v.get("v2_by_launch"), Mapping) or isinstance(v.get("v2"), Mapping)
            if v2 is None and item.get("vllm_first_tokens_v2") and launch_id is not None and has_v2_info:
                v2 = {"version": PARITY_CERTIFICATE_VERSION, "status": REF_STATUS_UNDECIDABLE, "gate": "parity_v2_unjudged_for_launch",
                      "reason": f"launch {launch_id} probed the certificate's prompts but the reference phase judged no v2 against its "
                                "own base probes — re-run the reference phase (the HF side is re-judged per launch)", "launch_id": launch_id,
                      "per_prompt": [], "base_complete": False}
            v = compose_parity_record(v1, v2)
        per_here[item["name"]] = v
    unjudged = {n: v for n, v in per_here.items() if v.get("status") not in (REF_STATUS_PASS, REF_STATUS_FAIL, REF_STATUS_UNDECIDABLE)}
    if unjudged:
        reason = "; ".join(f"{n}: {v.get('status')}" + (f" — {v.get('reason') or v.get('error')}" if (v.get('reason') or v.get('error')) else "")
                           for n, v in unjudged.items())
        return {"status": REF_STATUS_MISSING, "basis": None, "reason": reason[:2000], "per_adapter": per_here,
                "layouts": layouts, "reference_class": reference_class}
    undecidable = {n: f"{v.get('gate')}: {v.get('reason')}" for n, v in per_here.items() if v.get("status") == REF_STATUS_UNDECIDABLE}
    # the certificate carries BOTH sides — `reference` (the HF reference's venv, the agreement over
    # every judged adapter; source reference_process) and `serving` (THIS launch's engine identity: the standup's own block, or the
    # one derived from its recorded provenance); the two venv digests legitimately differ (the reference runs in the gdn venv, the
    # standups ran in the base venv), so the top-level requirements_sha256 / variant / name are the reference's and the top-level
    # engine_identity / engine_identity_source are the SERVING side's — the admission gate reads the serving side alone and refuses a
    # certificate whose serving identity is reference_process-only or unsourced; a serving launch's venv is recorded on its
    # result and certifies serving only
    venv = compose_certificate_venv([v.get(SERVING_VENV_KEY) for v in per_here.values()], engine_venv)
    if any(v.get("status") == REF_STATUS_FAIL for v in per_here.values()):
        # a judged MISMATCH on any adapter is the launch's verdict, whatever the other adapters' gates say (undecidable
        # used to outrank fail here, so one chaotic adapter hid a real mismatch on its neighbour behind
        # `reference_undecidable`); the undecidable ones are still listed for the memo
        return {"status": REF_STATUS_FAIL, "basis": BASIS_HF_PEFT, "per_adapter": per_here, "undecidable": undecidable or None,
                "layouts": layouts, "reference_class": reference_class, SERVING_VENV_KEY: venv}
    if undecidable:
        # judged, with evidence, and NOT decidable at this synthetic scale: a DECIDED exclusion of the pair, not a mismatch
        # and not missing evidence — the verdict files it as `reference_undecidable:<adapters>` under the hf_peft basis
        return {"status": REF_STATUS_UNDECIDABLE, "basis": BASIS_HF_PEFT, "undecidable": undecidable, "per_adapter": per_here,
                "layouts": layouts, "reference_class": reference_class, SERVING_VENV_KEY: venv}
    return {"status": REF_STATUS_PASS, "basis": BASIS_HF_PEFT, "per_adapter": per_here, SERVING_VENV_KEY: venv,
            "layouts": layouts, "reference_class": reference_class}


# --------------------------------------------------------------------------
# verdict
# --------------------------------------------------------------------------

def _adapter_ok(res: Mapping[str, Any], role: str) -> Tuple[bool, Optional[str]]:
    per = ((res.get("checks") or {}).get("adapters") or {})
    for name, v in per.items():
        if v.get("role") == role:
            return bool(v.get("pass")), v.get("reason")
    return False, f"no {role} adapter in the launch"


def _natural_launch_index(plan: Mapping[str, Any]) -> Dict[Tuple[str, int], Mapping[str, Any]]:
    """``{(manifest name, L): the FIRST natural / fallback launch serving it}``
    — built once per verdict instead of a scan per (pair × manifest × window)."""
    index: Dict[Tuple[str, int], Mapping[str, Any]] = {}
    for rec in plan["launches"]:
        if rec["kind"] in ("natural", "fallback"):
            for n in rec["serves"]:
                index.setdefault((n, int(rec["L"])), rec)
    return index


def _launch_for(results: Mapping[str, Mapping[str, Any]], index: Mapping[Tuple[str, int], Mapping[str, Any]],
                manifest_name: str, L: int
                ) -> Tuple[Optional[Mapping[str, Any]], Optional[Mapping[str, Any]]]:
    """(plan launch record, result) for the NATURAL launch serving ``manifest_name`` at L."""
    rec = index.get((manifest_name, int(L)))
    if rec is None:
        return None, None
    return rec, results.get(rec["id"])


def _role_manifests(servers: Sequence[sm.Manifest]) -> Dict[str, List[sm.Manifest]]:
    """``{role: [manifests carrying a GENUINE adapter of that role]}`` in CERTIFICATION ORDER: the current topology's
    servers (``S1cc`` / ``S1d``) first, then a record's legacy primary (``S1`` / ``single``), then its legacy fallbacks
    (``S1a`` / ``S1b`` / ``S1c``) — the first PASSED launch in this order certifies the role at a window (the role-keyed
    rule)."""
    order = {sm.TOPOLOGY_S1CC: 0, sm.TOPOLOGY_S1D: 0, sm.TOPOLOGY_S1: 1, sm.TOPOLOGY_SINGLE: 1}
    out: Dict[str, List[sm.Manifest]] = {}
    for m in sorted(servers, key=lambda m: (order.get(m.topology, 2 if m.topology in sm.S1_FALLBACKS else 3), m.name)):
        for role in m.genuine_roles:
            out.setdefault(role, []).append(m)
    return out


_ROW_PASSED, _ROW_FAILED, _ROW_UNRUN, _ROW_REPLAN, _ROW_DOWN, _ROW_UNPLANNED = ("passed", "failed", "unrun", "replan",
                                                                               "down", "unplanned")


def _resolve_role_launch(cands: Sequence[sm.Manifest], L: int, results: Mapping[str, Mapping[str, Any]],
                         index: Mapping[Tuple[str, int], Mapping[str, Any]],
                         replan_oom: Callable[[Optional[Mapping[str, Any]], Optional[Mapping[str, Any]]], bool]
                         ) -> Tuple[str, Optional[Tuple[sm.Manifest, Mapping[str, Any], Optional[Mapping[str, Any]]]],
                                    List[Tuple[sm.Manifest, Mapping[str, Any], Optional[Mapping[str, Any]]]]]:
    """How a ROLE stands at window ``L`` over its candidate manifests (:func:`_role_manifests` order): ``passed`` (the
    first candidate whose launch PASSED — the certifying one), else ``unrun`` (a candidate's launch has not run: the
    role is UNDECIDED, never excluded while a launch could still certify it), else ``replan`` (a start-time prefill
    OOM above the measured budget — undecided too), else ``failed`` (a launch came up and a check failed — its
    adapters are judged by name), else ``down`` (every candidate ran and died), else ``unplanned`` (no candidate has a
    launch at L). Returns ``(kind, the chosen (manifest, plan launch, result), every planned row)``."""
    rows: List[Tuple[sm.Manifest, Mapping[str, Any], Optional[Mapping[str, Any]]]] = []
    for m in cands:
        rec = index.get((m.name, int(L)))
        if rec is None:
            continue
        rows.append((m, rec, results.get(rec["id"])))
    if not rows:
        return _ROW_UNPLANNED, None, rows
    for row in rows:
        if row[2] is not None and row[2].get("status") == STATUS_PASSED:
            return _ROW_PASSED, row, rows
    for row in rows:
        res = row[2]
        if res is None or res.get("status") not in TERMINAL_STATUSES:
            return _ROW_UNRUN, row, rows
    for row in rows:
        if row[2] is not None and replan_oom(row[1], row[2]):
            return _ROW_REPLAN, row, rows
    for row in rows:
        if row[2] is not None and row[2].get("status") == STATUS_FAILED:
            return _ROW_FAILED, row, rows
    for row in rows:
        if row[2] is not None and row[2].get("status") in (STATUS_OOM, STATUS_REFUSED, STATUS_ERROR):
            return _ROW_DOWN, row, rows
    # every candidate SKIPPED (a legacy fallback whose parent passed — then the parent would have been `passed` above)
    return _ROW_DOWN, rows[0], rows


def _measured_S(rec: Mapping[str, Any], res: Mapping[str, Any]) -> int:
    eng = (res.get("engine") or {}).get("max_num_seqs_engine")
    return int(eng) if isinstance(eng, int) and not isinstance(eng, bool) else int(rec["admitted_S"])


def pair_serving_status(plan: Mapping[str, Any], results: Mapping[str, Mapping[str, Any]], pair: Mapping[str, Any], *,
                        budget_tokens: Optional[int] = None) -> Dict[str, Any]:
    """The ROLE-KEYED serving status of one (set, rank) pair over the plan's windows — the same resolution the verdict
    applies (:func:`_resolve_role_launch` over :func:`_role_manifests`), so the reopen's plan extension and the verdict
    read one rule. Per window: ``certified`` (every required role has a PASSED launch with S >= 1), ``measured_S`` (the
    min over the roles' certifying engines; None when not certified), ``undecided`` (a role's launch is unrun or
    replan-required), ``by_role`` {role: kind}. ``certified_all`` is the AND over the windows."""
    index = _natural_launch_index(plan)
    windows = [int(w) for w in plan["windows"]]
    manifests = [plan_manifest(plan, n) for n in pair["manifests"]]
    role_m = _role_manifests([m for m in manifests if m.topology != sm.TOPOLOGY_S2])
    if budget_tokens is None:
        # the record's MEASURED budget, as the verdict reads it: with no floor every start-time OOM is a re-plan
        budget_tokens = measured_budget(plan, results).get("tokens")

    def _replan(rec, res):
        return rec is not None and res is not None and oom_kind(rec, res, budget_tokens) in REPLAN_OOM_KINDS
    out: Dict[str, Any] = {"windows": {}, "certified_all": True}
    for L in windows:
        by_role: Dict[str, str] = {}
        S_min: Optional[int] = None
        undecided = False
        for role in sm.ROLES_REQUIRED:
            kind, row, _rows = _resolve_role_launch(role_m.get(role, []), L, results, index, _replan)
            by_role[role] = kind
            if kind == _ROW_PASSED:
                S_min = _measured_S(row[1], row[2]) if S_min is None else min(S_min, _measured_S(row[1], row[2]))
            elif kind in (_ROW_UNRUN, _ROW_REPLAN):
                undecided = True
        certified = all(k == _ROW_PASSED for k in by_role.values()) and S_min is not None and S_min >= 1
        out["windows"][L] = {"certified": certified, "measured_S": (S_min if certified else None), "undecided": undecided,
                             "by_role": by_role}
        out["certified_all"] = out["certified_all"] and certified
    return out


#: the extension's third reason (the probe owner's ruling of 2026-10-11): no planned launch of the pair dominates the class a steered
#: standup serves at that window — and the verdict's exclusion when every planned launch of that class at a window ran and none
#: served it (``served_class_undominated_at_L=<L>:<launch>:<why>``)
REASON_SERVED_CLASS = "served_class_undominated"
#: the verdict's exclusion when NOTHING of the served class was planned at a window (a record written before the union was planned;
#: the reopen's extension adds it): ``served_class_unplanned_at_L=<L>:(rank cap C, N adapters)``
REASON_SERVED_CLASS_UNPLANNED = "served_class_unplanned"


def extend_plan_for_uncertified_pairs(plan: Mapping[str, Any], results: Mapping[str, Mapping[str, Any]], dims: Any, *,
                                      consider_admission: Optional[bool] = None, tree_sha: Optional[str] = None
                                      ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """THE REOPEN'S PLAN EXTENSION: for every pair whose SERVING is not certified at some window
    (:func:`pair_serving_status`), add the CURRENT topology's servers (``standup_manifest.build_manifests`` — ``S1cc`` +
    ``S1d``; the shared ``S2`` is already planned) as new natural launches at every window where (a) a required role is
    not certified, or — unless ``consider_admission`` is False (a parity-pass plan judges adapters at one window; its S is
    immaterial) — (b) the certifying launches' MEASURED S is below what the current servers would ADMIT there under the
    record's budget (the old evidence was buffer-starved: the four-slot rank-64 server admitted S = 2 at 4,096 against the
    64 the budget allows); and, for EVERY pair of a non-parity plan, (c) ``served_class_undominated``: at every window
    where no planned launch over the pair's manifests dominates the pair's UNION class
    (``standup_manifest.build_union_manifest``: rank cap >= and slots >= at the same window, OVER THE UNION'S TARGET CLASS
    — :func:`_served_class_dominated`; the shared o_proj S2 never stands in for an ``mlp`` union it out-counts), the
    union's natural launch is added, so a certified record whose split servers (7 and 2 slots) never covered the 8-slot
    class a steered standup serves is re-run for exactly the missing windows on an EXISTING record
    (``run --resume --replan`` picks them up as work). Launch ids are stable (manifest, cap, L, tp); an id already planned
    is never re-added, a manifest already planned is never rebuilt, so a second call adds nothing. Completed launches are
    canonical and untouched. Returns ``(new plan, change record)``; the plan carries the extension under ``extensions``
    (utc, tree sha, pairs, manifests, launch ids, why)."""
    plan_out: Dict[str, Any] = json.loads(json.dumps(plan))
    if consider_admission is None:
        consider_admission = not bool(plan.get("parity_pass"))
    # (c) applies to a plan that FOLDS geometry; a parity pass folds none (and the union adds no adapter to judge)
    served_class = not bool(plan.get("parity_pass"))
    b = plan_out["budget"]
    windows = [int(w) for w in plan_out["windows"]]
    requested = {int(k): int(v) for k, v in (plan_out.get("requested_S_by_window") or {}).items()}
    tp = int(plan_out.get("tp") or 1)
    budget = measured_budget(plan_out, results)
    budget_tokens = budget.get("tokens")
    by_id = {r["id"]: r for r in plan_out["launches"]}
    manifests_by_name: Dict[str, sm.Manifest] = {n: sm.Manifest.from_dict(d) for n, d in plan_out["manifests"].items()}
    added: List[str] = []
    new_manifests: List[str] = []
    pairs_touched: List[List[Any]] = []
    why: Dict[str, str] = {}
    def _add(pair: Dict[str, Any], m: sm.Manifest, L: int, reason: str) -> bool:
        """Plan ``m``'s natural launch at ``L`` for ``pair`` unless its stable id is already planned; True when added."""
        lid = launch_id(m.name, m.max_lora_rank, L, tp=tp)
        if lid in by_id:
            return False
        if m.name not in manifests_by_name:
            manifests_by_name[m.name] = m
            plan_out["manifests"][m.name] = m.as_dict()
            new_manifests.append(m.name)
        if m.name not in pair["manifests"]:
            pair["manifests"].append(m.name)
        rec = _launch_record(m, L, "natural", _launch_priority(m, L, int(plan_out["ladder_window"])), dims=dims, tp=tp,
                             requested_S=int(requested.get(L, 8)), budget_tokens=budget_tokens, budget=b,
                             manifests_by_name=manifests_by_name)
        plan_out["launches"].append(rec)
        by_id[lid] = rec
        added.append(lid)
        why[lid] = str(reason)
        return True

    for pair in plan_out["pairs"]:
        st = pair_serving_status(plan_out, results, pair, budget_tokens=budget_tokens)
        if st["certified_all"] and not consider_admission and not served_class:
            continue
        current = sm.build_manifests(dims, set_id=pair["set"], rank=int(pair["rank"]),
                                     base_weights_bytes=float(b["base_weights_bytes"]),
                                     graphs_activations_bytes=float(b["graphs_activations_bytes"]),
                                     gpu_memory_utilization=float(b["gpu_memory_utilization"]),
                                     total_vram_bytes=float(b["total_vram_bytes"]),
                                     recurrent_state_per_seq_bytes=int(b.get("recurrent_state_per_seq_bytes") or 0))
        servers = [m for m in current if m.topology != sm.TOPOLOGY_S2]
        # (c): the class a steered standup of this pair serves — covered at a window only by a planned launch that dominates it
        union = sm.build_union_manifest(dims, set_id=pair["set"], rank=int(pair["rank"])) if served_class else None
        touched = False
        for L in windows:
            w = st["windows"][L]
            need = not w["certified"]
            reason = "role_uncertified" if need else None
            if not need and consider_admission and w["measured_S"] is not None:
                planned = min(launch_admission(dims, m, L, requested_S=int(requested.get(L, 8)),
                                               kv_budget_bytes=max(0.0, float(b["gpu_memory_utilization"]) * float(b["total_vram_bytes"])
                                                                   - float(b["base_weights_bytes"])
                                                                   - sm.expected_lora_bytes(m, m.max_lora_rank, dims)["total"]
                                                                   - float(b["graphs_activations_bytes"])),
                                               recurrent_state_bytes=int(b.get("recurrent_state_per_seq_bytes") or 0),
                                               batched_token_budget=(int(budget_tokens) if budget_tokens is not None else None)
                                               )["admitted_S"] for m in servers)
                if planned >= 1 and int(w["measured_S"]) < int(planned):
                    need = True
                    reason = f"measured_S={w['measured_S']}<planned_S={planned}"
            if need:
                for m in servers:
                    touched = _add(pair, m, L, str(reason)) or touched
            if union is not None and not _served_class_dominated(plan_out, pair, union, L, manifests_by_name):
                touched = _add(pair, union, L, REASON_SERVED_CLASS) or touched
        if touched:
            pairs_touched.append([int(pair["rank"]), pair["set"]])
    if added:
        plan_out["launches"].sort(key=lambda r: (r["priority"], r["L"], r["set"], r["rank"], r["id"]))
        plan_out["n_launches"] = len(plan_out["launches"])
        plan_out.setdefault("extensions", []).append({
            "utc": utc_now(), "tree_sha": tree_sha, "pairs": pairs_touched, "manifests": new_manifests, "launches": added,
            "why": why, "consider_admission": bool(consider_admission), "served_class": bool(served_class),
            "rule": ("the current topology's servers (standup_manifest.build_manifests) at every window where a required role "
                     "is uncertified" + (" or the certifying launches' measured S is below the current servers' admitted S"
                                          if consider_admission else "")
                     + ("; the pair's union manifest (standup_manifest.build_union_manifest — the served launch class) at every "
                        "window where no planned launch over the pair's manifests dominates its class (rank cap >= and slots >=)"
                        if served_class else ""))})
    return plan_out, {"added": added, "manifests": new_manifests, "pairs": pairs_touched, "why": why}



#: ``--certify-v2 all``: every pair of the plan (the certificate on the production venv for the whole record)
CERTIFY_ALL: Tuple[int, str] = (-1, "all")


def parse_pairs_arg(text: Optional[str]) -> List[Tuple[int, str]]:
    """``--certify-v2 16:down,16:mlp`` -> ``[(16, "down"), (16, "mlp")]`` (rank:set, comma-separated); refuses a malformed item."""
    out: List[Tuple[int, str]] = []
    if str(text or "").strip().lower() == "all":
        return [CERTIFY_ALL]
    for tok in (t.strip() for t in str(text or "").split(",")):
        if not tok:
            continue
        if ":" not in tok:
            raise ValueError(f"--certify-v2 item {tok!r} is not rank:set")
        r, sid = tok.split(":", 1)
        if not r.strip().isdigit() or sid.strip() not in sm.MODULE_SETS:
            raise ValueError(f"--certify-v2 item {tok!r}: rank must be an int and the set one of {sorted(sm.MODULE_SETS)}")
        out.append((int(r), sid.strip()))
    return out


def certify_v2_work(plan: Mapping[str, Any], results: Mapping[str, Mapping[str, Any]], dims: Any, pairs: Sequence[Tuple[int, str]], *,
                    tree_sha: Optional[str] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """PARITY CERTIFICATE v2 for NAMED pairs: the current topology's servers (S1cc / S1d) at the plan's ladder window are
    planned for each pair when absent (whatever its certification — this is a certificate upgrade, not an exclusion repair),
    and every such launch whose recorded result carries no ``probes_v2`` (judged under v1 alone, or never run) is WORK: an
    existing v1-only result is superseded (moved aside by the caller, never deleted) so the launch re-runs and probes the
    certificate's prompts on the engine side; the reference phase then judges v2 (:func:`reference_phase` re-pends an adapter
    whose prior judgement lacks ``v2``). Returns ``(new plan, {added, supersede, pairs, why})``; pure (no write)."""
    plan_out: Dict[str, Any] = json.loads(json.dumps(plan))
    b = plan_out["budget"]
    L = int(plan_out["ladder_window"])
    requested = {int(k): int(v) for k, v in (plan_out.get("requested_S_by_window") or {}).items()}
    tp = int(plan_out.get("tp") or 1)
    budget_tokens = measured_budget(plan_out, results).get("tokens")
    by_id = {r["id"]: r for r in plan_out["launches"]}
    manifests_by_name: Dict[str, sm.Manifest] = {n: sm.Manifest.from_dict(d) for n, d in plan_out["manifests"].items()}
    by_pair = {(int(pr["rank"]), pr["set"]): pr for pr in plan_out["pairs"]}
    if any(tuple(pr) == CERTIFY_ALL for pr in pairs):
        pairs = sorted(by_pair)
    added: List[str] = []
    supersede: List[str] = []
    why: Dict[str, str] = {}
    for rank, sid in pairs:
        pair = by_pair.get((int(rank), sid))
        if pair is None:
            raise ValueError(f"--certify-v2 names ({rank}, {sid}), which this plan does not carry (pairs {sorted(by_pair)})")
        current = sm.build_manifests(dims, set_id=sid, rank=int(rank), base_weights_bytes=float(b["base_weights_bytes"]),
                                     graphs_activations_bytes=float(b["graphs_activations_bytes"]),
                                     gpu_memory_utilization=float(b["gpu_memory_utilization"]),
                                     total_vram_bytes=float(b["total_vram_bytes"]),
                                     recurrent_state_per_seq_bytes=int(b.get("recurrent_state_per_seq_bytes") or 0))
        for m in current:
            if m.topology == sm.TOPOLOGY_S2:
                continue
            lid = launch_id(m.name, m.max_lora_rank, L, tp=tp)
            if lid not in by_id:
                if m.name not in manifests_by_name:
                    manifests_by_name[m.name] = m
                    plan_out["manifests"][m.name] = m.as_dict()
                if m.name not in pair["manifests"]:
                    pair["manifests"].append(m.name)
                rec = _launch_record(m, L, "natural", _launch_priority(m, L, L), dims=dims, tp=tp, requested_S=int(requested.get(L, 8)),
                                     budget_tokens=budget_tokens, budget=b, manifests_by_name=manifests_by_name)
                plan_out["launches"].append(rec)
                by_id[lid] = rec
                added.append(lid)
                why[lid] = "v2: current server not planned"
                continue
            res = results.get(lid)
            if res is None or res.get("status") not in TERMINAL_STATUSES:
                why[lid] = "v2: unrun"
                continue
            if res.get("status") in (STATUS_PASSED, STATUS_FAILED) and res.get("probes_v2"):
                continue                                     # already carries the engine side of v2
            supersede.append(lid)
            why[lid] = f"v2: result {res.get('status')} carries no probes_v2 — superseded to re-run"
    if added:
        plan_out["launches"].sort(key=lambda r: (r["priority"], r["L"], r["set"], r["rank"], r["id"]))
        plan_out["n_launches"] = len(plan_out["launches"])
    if added or supersede:
        plan_out.setdefault("extensions", []).append({"utc": utc_now(), "tree_sha": tree_sha, "kind": "certify_v2",
                                                      "pairs": [[int(r), sid] for r, sid in pairs], "launches": added,
                                                      "superseded": supersede, "why": why})
    return plan_out, {"added": added, "supersede": supersede, "pairs": [[int(r), sid] for r, sid in pairs], "why": why}


def _fail_reason(rec: Mapping[str, Any], res: Optional[Mapping[str, Any]]) -> str:
    if res is None or res.get("status") == STATUS_NOT_RUN:
        return "not_run"
    st = res.get("status")
    if st == STATUS_OOM:
        return f"oom_at_R={rec['cap']}"
    if st in (STATUS_REFUSED, STATUS_ERROR):
        msg = ((res.get("failure") or {}).get("message") or st)
        return msg if msg.startswith("refused:") else f"refused:{msg}"[:300]
    if st == STATUS_SKIPPED:
        return "skipped"
    return f"failed:{','.join((res.get('checks') or {}).get('failed') or [])}"[:300]


def record_base_floor(record_dir: Path) -> Optional[float]:
    """The record's MEASURED HF-vs-engine base floor (``reference_check.json`` ``base_floor``); None when the record
    carries none (the rule then falls back to ``tol`` and the adapter's own sensitivity term)."""
    p = Path(record_dir) / REFERENCE_CHECK_FILE
    if not p.is_file():
        return None
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
    except ValueError:
        return None
    v = rec.get("base_floor") if isinstance(rec, Mapping) else None
    return float(v) if v is not None else None


def regate_reference_results(results: Mapping[str, Mapping[str, Any]], record_dir: Path
                             ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Every launch's ``reference_check`` re-judged under the CURRENT rule from the RAW fields its per-adapter records carry
    (``hf``, ``hf_base``, ``vllm``, ``moved``, ``sensitivity``; the record's ``base_floor``) — in memory, the result files
    untouched (they are the evidence; the verdict is the reading). A per-adapter record without the raw fields (a fixture, a
    record written before they existed), or one the harness filed as ``reference_missing`` / ``error`` (attachment, layout,
    identical outputs — facts the gate does not touch), keeps its recorded status. Returns ``(results, report)``; the report
    lists every adapter whose status or gate changed, per launch, and is stamped into the verdict so a re-read is visible: a
    parity record judged at a sha whose weak gate read the engine alone re-gates here without re-running its GPU phase."""
    floor = record_base_floor(record_dir)
    record = _read_reference_record(record_dir)
    plan_doc = _read_json_or_none(Path(record_dir) / PLAN_FILE)
    model_key = str((plan_doc or {}).get("model_key") or "") or None
    job_logs = record_job_logs(record_dir, model_key)
    out: Dict[str, Dict[str, Any]] = {}
    changed: List[Dict[str, Any]] = []
    n_launches_changed = 0
    for lid in sorted(results):
        res = results[lid]
        rc = res.get("reference_check") if isinstance(res, Mapping) else None
        pending = res.get("reference_pending") if isinstance(res, Mapping) else None
        per_name = (rc or {}).get("per_adapter") if isinstance(rc, Mapping) else None
        if not (isinstance(rc, Mapping) and rc.get("basis") == BASIS_HF_PEFT and pending and isinstance(per_name, Mapping)
                and all(isinstance(per_name.get(it.get("name")), Mapping) for it in pending)):
            out[lid] = res
            continue
        per_by_dir: Dict[str, Any] = {}
        launch_changed: List[Dict[str, Any]] = []
        for it in pending:
            v = per_name[it["name"]]
            new_v = regate_adapter_record(v, floor=floor, launch_item=it, launch_id=str(lid), record=record)
            if (new_v.get("status"), new_v.get("gate")) != (v.get("status"), v.get("gate")):
                launch_changed.append({"launch": lid, "adapter": it["name"], "from": v.get("status"), "from_gate": v.get("gate"),
                                       "to": new_v.get("status"), "to_gate": new_v.get("gate")})
            per_by_dir[it["dir"]] = new_v
        eng, eng_refusal = engine_side_block(record_dir, {**res, "model_key": res.get("model_key") or model_key}, job_logs=job_logs)
        new_rc = launch_reference_check(pending, per_by_dir, layouts=rc.get("layouts"), reference_class=rc.get("reference_class"),
                                        launch_id=str(lid), engine_venv=eng)
        if eng_refusal:
            new_rc["engine_provenance_refusal"] = eng_refusal      # Why this launch's serving side stays unproven
        if launch_changed or new_rc.get("status") != rc.get("status"):
            n_launches_changed += 1
            new_rc["regated"] = {"from_status": rc.get("status"), "adapters": launch_changed,
                                 "rule": {"floor_k": PARITY_FLOOR_K, "sensitivity_m": PARITY_SENSITIVITY_M, "moved_min": PARITY_MOVED_MIN,
                                          "weak_gate": "both_sides", "base_floor": floor}}
            changed += launch_changed
        # the re-read ALWAYS stands in for the stored check: a v2 re-judged on this launch's own base probes is
        # bound to the launch (`launch_id`, `vllm_base`, `base_complete`) even when its status did not move; `regated` marks a change
        out[lid] = {**res, "reference_check": new_rc}
    return out, {"n_launches_changed": n_launches_changed, "adapters_changed": changed, "base_floor": floor}


def _read_json_or_none(p: Path) -> Optional[Dict[str, Any]]:
    try:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


def _read_reference_record(record_dir: Path) -> Dict[str, Any]:
    p = Path(record_dir) / REFERENCE_CHECK_FILE
    if not p.is_file():
        return {}
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
    except ValueError:
        return {}
    return dict(rec) if isinstance(rec, Mapping) else {}


def regate_adapter_record(v: Mapping[str, Any], *, floor: Optional[float], launch_item: Optional[Mapping[str, Any]] = None,
                          launch_id: Optional[str] = None, record: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """One stored per-adapter record re-judged (:func:`judge_adapter_parity`) when it was JUDGED (pass / fail / undecidable) and
    carries the raw fields; otherwise returned as recorded. v2 is re-judged PER LAUNCH: from the record's own
    stored per-prompt fields when it has them, else — a launch whose v2 was never judged against its own bases (a record written
    by d72fc919f's phase) — REBUILT from the reference record's HF side for the adapter (``per_adapter[dir].hf_v2`` / ``sens_v2``
    with ``hf_base_v2``, or the first stored v2's per-prompt HF fields) and the launch's OWN engine probes on its
    ``reference_pending`` item; the launch's own base probes set its floor, an incomplete base makes it undecidable."""
    if v.get("status") not in (REF_STATUS_PASS, REF_STATUS_FAIL, REF_STATUS_UNDECIDABLE):
        return dict(v)
    hf, vl, sens = v.get("hf"), v.get("vllm"), v.get("sensitivity")
    if not (isinstance(hf, Mapping) and hf.get("top1_id") is not None and hf.get("top1_logprob") is not None
            and isinstance(vl, Mapping) and isinstance(sens, Mapping)):
        return dict(v)
    hf_base = v.get("hf_base") if isinstance(v.get("hf_base"), Mapping) else None
    v1 = judge_adapter_parity(hf, hf_base, vl, moved=v.get("moved"), sensitivity=sens, floor=floor, attachment=v.get("attachment"))
    v2_old = v.get("v2") if isinstance(v.get("v2"), Mapping) else None
    v2 = None
    # the HF side of the certificate (per prompt: reference first token, base first token, one-ulp sensitivity), stored ONCE —
    # on the launch's own v2 rows, else on the reference record's adapter entry (``hf_v2`` / ``sens_v2`` + ``hf_base_v2``), else on
    # any stored v2 of the same adapter (its per-prompt HF fields do not depend on the launch)
    hf_v2: Optional[List[Any]] = None
    hf_base_v2: Optional[List[Any]] = None
    sens_v2: Optional[List[Any]] = None
    sha_v2 = (v2_old or {}).get("prompts_sha")
    if v2_old and isinstance(v2_old.get("per_prompt"), list) and v2_old["per_prompt"]:
        rows = v2_old["per_prompt"]
        hf_v2, hf_base_v2, sens_v2 = [r["hf"] for r in rows], [r["hf_base"] for r in rows], [r.get("sensitivity") for r in rows]
    elif record and launch_item:
        adapter_rec = (record.get("per_adapter") or {}).get(str(launch_item.get("dir"))) or {}
        sha_v2 = record.get("parity_prompts_sha")
        if adapter_rec.get("hf_v2") and record.get("hf_base_v2"):
            hf_v2, hf_base_v2, sens_v2 = list(adapter_rec["hf_v2"]), list(record["hf_base_v2"]), list(adapter_rec.get("sens_v2") or [])
        else:
            donors = [d for d in (adapter_rec.get("v2_by_launch") or {}).values() if isinstance(d, Mapping) and d.get("per_prompt")]
            if isinstance(adapter_rec.get("v2"), Mapping) and adapter_rec["v2"].get("per_prompt"):
                donors.append(adapter_rec["v2"])
            if donors:
                rows = donors[0]["per_prompt"]
                hf_v2, hf_base_v2, sens_v2 = [r["hf"] for r in rows], [r["hf_base"] for r in rows], [r.get("sensitivity") for r in rows]
                sha_v2 = donors[0].get("prompts_sha", sha_v2)
    if hf_v2 and hf_base_v2 and len(hf_v2) == len(hf_base_v2):
        n = len(hf_v2)
        # the ENGINE side is THIS launch's own: its adapter probes, its base probes (the floor), its movement —
        # from the launch's reference_pending item when it carries the certificate's probes under the same prompt set, else from
        # the rows the record stored for this launch (a record written by this rule, whose ``vllm_base`` is the launch's own)
        item_sha = (launch_item or {}).get("parity_prompts_sha") if launch_item else None
        if launch_item and launch_item.get("vllm_first_tokens_v2") and (sha_v2 is None or item_sha == sha_v2):
            vl = list(launch_item.get("vllm_first_tokens_v2") or [])
            vb = launch_item.get("vllm_base_first_tokens_v2")
            moved = list(launch_item.get("vllm_moved_v2") or [None] * n)
        elif v2_old and v2_old.get("vllm_base") is not None and isinstance(v2_old.get("per_prompt"), list):
            vl = [r["vllm"] for r in v2_old["per_prompt"]]
            vb = v2_old.get("vllm_base")
            moved = [r.get("moved") for r in v2_old["per_prompt"]]
        else:
            # a d72fc919f-era record with no launch item to re-measure from: its engine rows may be ANOTHER launch's and its floor
            # the first matching launch's — judged with NO base of its own, i.e. undecidable, never a borrowed pass
            vl = [r["vllm"] for r in (v2_old or {}).get("per_prompt") or []]
            vb = None
            moved = [r.get("moved") for r in (v2_old or {}).get("per_prompt") or []]
        lid_eff = launch_id or (v2_old or {}).get("launch_id")
        if len(vl) != n:
            v2 = {"version": PARITY_CERTIFICATE_VERSION, "status": REF_STATUS_UNDECIDABLE, "gate": "parity_base_incomplete",
                  "reason": f"launch {lid_eff} probed {len(vl)}/{n} certificate prompts for this adapter", "launch_id": lid_eff,
                  "per_prompt": [], "base_complete": False, "prompts_sha": sha_v2}
        else:
            fl, _ds, complete = v2_base_floor(hf_base_v2, vb)
            v2 = judge_adapter_parity_v2(hf_v2, hf_base_v2, vl, moved_list=moved, sens_list=list(sens_v2 or [None] * n), floor=fl,
                                         prompts_sha=sha_v2, base_complete=complete, vllm_base_list=vb, launch_id=lid_eff)
    out = compose_parity_record(v1, v2)
    if isinstance(v.get(SERVING_VENV_KEY), Mapping):
        out[SERVING_VENV_KEY] = dict(v[SERVING_VENV_KEY])       # The identity is a recorded fact, re-gating keeps it
    if v.get(ADAPTER_CONTENT_KEY):
        out[ADAPTER_CONTENT_KEY] = str(v[ADAPTER_CONTENT_KEY])  # the adapter's content identity is a recorded fact too (never re-derived here)
    return out


def load_parity_record(path: Path, main_plan: Mapping[str, Any]) -> Dict[str, Any]:
    """The PARITY-PASS record ``--parity-record`` names, loaded and held to the main record: a plan written with
    ``--parity-pass`` for the SAME model_key / model_id / revision (a parity record of another checkpoint certifies nothing
    here). Returns ``{dir, plan, results, generation}``; refuses by name otherwise."""
    p = Path(path)
    if not (p / PLAN_FILE).is_file():
        raise ValueError(f"--parity-record {p}: no {PLAN_FILE} — not a probe record")
    plan = json.loads((p / PLAN_FILE).read_text(encoding="utf-8"))
    if not plan.get("parity_pass"):
        raise ValueError(f"--parity-record {p}: its plan was not written with --parity-pass (parity_pass={plan.get('parity_pass')!r}) — "
                         "assertion 4 reads only a parity-pass record's launches (per-set benign scales, the ladder window)")
    for k in ("model_key", "model_id", "revision"):
        if str(plan.get(k)) != str(main_plan.get(k)):
            raise ValueError(f"--parity-record {p}: {k} {plan.get(k)!r} is not this record's {main_plan.get(k)!r} — a parity pass of "
                             "another checkpoint certifies nothing here")
    if not plan.get("synthetic_scale_by_set"):
        raise ValueError(f"--parity-record {p}: its plan carries no synthetic_scale_by_set — the scale table is the pass's provenance")
    # the stored per-adapter fields are re-judged under the CURRENT rule (in memory; the record's files are the evidence)
    results, regate = regate_reference_results(load_results(p), p)
    if regate["n_launches_changed"]:
        log(f"parity record {p}: {regate['n_launches_changed']} launch(es) re-gated under the current rule from their stored fields — "
            + "; ".join(f"{c['launch']}/{c['adapter']} {c['from']}{'/' + c['from_gate'] if c.get('from_gate') else ''} -> {c['to']}"
                        f"{'/' + c['to_gate'] if c.get('to_gate') else ''}" for c in regate["adapters_changed"][:8]))
    return {"dir": str(p), "plan": plan, "results": results, "generation": result_generation(p), "regate": regate}


def parity_record_stamp(parity: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """What the verdict and the menu record about the parity pass they read assertion 4 from."""
    if parity is None:
        return None
    plan, results = parity["plan"], parity.get("results") or {}
    judged = sorted(lid for lid, r in results.items()
                    if r.get("status") in (STATUS_PASSED, STATUS_FAILED) and (r.get("reference_check") or {}).get("basis") == BASIS_HF_PEFT)
    return {"dir": parity["dir"], RESULT_GENERATION_KEY: (parity.get("generation") or {}).get("generation"),
            "n_results": (parity.get("generation") or {}).get("n_results"), "plan_launches": plan.get("n_launches"),
            "windows": list(plan.get("windows") or []), "synthetic_scale_by_set": dict(plan.get("synthetic_scale_by_set") or {}),
            "scale_provenance": plan.get("parity_scale_provenance"), "judged_launches": judged,
            # what the CURRENT rule changed when the stored per-adapter fields were re-judged (empty = read as recorded)
            "regated": {"n_launches_changed": int((parity.get("regate") or {}).get("n_launches_changed") or 0),
                        "adapters_changed": list((parity.get("regate") or {}).get("adapters_changed") or [])}}


def parity_reference_sources(plan: Mapping[str, Any], certified_by: Sequence[str],
                             parity: Mapping[str, Any]) -> List[Tuple[str, Optional[Mapping[str, Any]]]]:
    """``[(label, reference_check | None)]`` for a pair: one entry per MANIFEST its certifying launches (in THIS plan) serve,
    read from the parity record's natural launch of the same manifest name (None when the parity plan has no such launch or
    it has not run to passed/failed — `reference_missing` for that pair alone)."""
    by_id = {r["id"]: r for r in plan["launches"]}
    manifests = list(dict.fromkeys(by_id[lid]["manifest"] for lid in certified_by if lid in by_id))
    p_plan, p_results = parity["plan"], parity.get("results") or {}
    out: List[Tuple[str, Optional[Mapping[str, Any]]]] = []
    for name in manifests:
        # the parity launch serving this manifest: its natural launch, else the fallback that ran in its place (a rank-64 S1
        # that OOMs at the ladder window certifies through S1a/S1c on BOTH records — the same conditional rule)
        recs = sorted((r for r in p_plan.get("launches") or [] if r.get("manifest") == name and r.get("kind") in ("natural", "fallback")),
                      key=lambda r: (r.get("kind") != "natural", str(r.get("id"))))
        ran = [r for r in recs if (p_results.get(r["id"]) or {}).get("status") in (STATUS_PASSED, STATUS_FAILED)]
        if ran:
            out.append((f"parity record: {ran[0]['id']}", (p_results.get(ran[0]["id"]) or {}).get("reference_check")))
            continue
        # the parity record judges ADAPTERS, and an adapter is the same object under every manifest that serves it
        # (the bank keys it by kind, modules, rank, r_e, seed and scale — never by server). A manifest the parity record
        # never planned (the current S1cc / S1d servers over a record whose parity pass ran the legacy four-slot S1) reads
        # the judgement of its GENUINE adapters, by name, off any judged parity launch of the same (set, rank) that
        # carries every one of them; the status is re-derived over exactly those adapters.
        m = plan_manifest(plan, name) if name in plan["manifests"] else None
        if m is not None:
            donor = _parity_by_adapter_name(m, p_plan, p_results)
            if donor is not None:
                out.append(donor)
                continue
        if not recs:
            out.append((f"parity record: no launch for manifest {name}", None))
            continue
        st = {r["id"]: (p_results.get(r["id"]) or {}).get("status", STATUS_NOT_RUN) for r in recs}
        out.append((f"parity record: {', '.join(f'{k} {v}' for k, v in st.items())}", None))
    return out


def _parity_by_adapter_name(m: sm.Manifest, p_plan: Mapping[str, Any], p_results: Mapping[str, Mapping[str, Any]]
                            ) -> Optional[Tuple[str, Mapping[str, Any]]]:
    """``(label, reference_check)`` restricted to ``m``'s GENUINE adapter names, each read off a judged parity launch of the same
    (set, rank) that carries it (:func:`parity_reference_sources`): an adapter is the same object under every manifest that
    serves it, so the split topology's servers (S1cc carrying control + constitution, S1d carrying difference) together judge a
    legacy S1 manifest's three adapters (job 820268 read (16, mlp) `reference_missing` because this required ONE launch carrying
    all of them while the parity record's legacy S1 launch refused at the ladder window). Natural launches before fallbacks,
    then by id; None when any genuine adapter has NO judged donor (`reference_missing` for the pair). The restricted status:
    ``fail`` if any adapter failed, else ``reference_undecidable`` if any is undecidable, else ``pass``; donors that name
    DIFFERENT venv digests are refused as undecidable by name (``parity_venv_disagreement``), never composed into a pass."""
    wanted = [a.name for a in m.genuine]
    if not wanted:
        return None
    cands = sorted((r for r in p_plan.get("launches") or [] if r.get("set") == m.set_id and int(r.get("rank", -1)) == int(m.rank)
                    and r.get("kind") in ("natural", "fallback")), key=lambda r: (r.get("kind") != "natural", str(r.get("id"))))
    judged: List[Tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for r in cands:
        res = p_results.get(r["id"]) or {}
        rc = res.get("reference_check")
        if res.get("status") in (STATUS_PASSED, STATUS_FAILED) and isinstance(rc, Mapping) and rc.get("basis") == BASIS_HF_PEFT:
            judged.append((r, rc))
    chosen: Dict[str, Tuple[str, Mapping[str, Any], Optional[str], Mapping[str, Any]]] = {}
    for n in wanted:
        for r, rc in judged:
            per = rc.get("per_adapter") or {}
            if isinstance(per.get(n), Mapping):
                chosen[n] = (str(r["id"]), per[n], (rc.get("undecidable") or {}).get(n), rc)
                break
    if len(chosen) != len(wanted):
        return None
    donors = list(dict.fromkeys(c[0] for c in chosen.values()))
    donor_rcs = list({c[0]: c[3] for c in chosen.values()}.values())
    sub_per = {n: chosen[n][1] for n in wanted}
    und = {n: chosen[n][2] for n in wanted if chosen[n][2]}
    venv = _agreeing_venv([rc.get(SERVING_VENV_KEY) for rc in donor_rcs])
    venv_named = [rc.get(SERVING_VENV_KEY) for rc in donor_rcs if isinstance(rc.get(SERVING_VENV_KEY), Mapping) and rc[SERVING_VENV_KEY].get("requirements_sha256")]
    if venv is None and venv_named and len(donors) > 1:
        # the donors' judgements were measured in different venvs: not one certificate — refused by name, never composed into a pass
        digests = sorted({str(b["requirements_sha256"])[:12] for b in venv_named})
        status = REF_STATUS_UNDECIDABLE
        und = {n: f"parity_venv_disagreement: the parity launches read for this adapter's siblings ({', '.join(donors)}) name different venv digests "
                  f"({', '.join(digests)}) — one certificate cannot be composed across them" for n in wanted}
    elif any(str(v.get("status")) == REF_STATUS_FAIL for v in sub_per.values()):
        status = REF_STATUS_FAIL
    elif und or any(str(v.get("status")) == REF_STATUS_UNDECIDABLE for v in sub_per.values()):
        status = REF_STATUS_UNDECIDABLE
    else:
        status = REF_STATUS_PASS
    first_rc = chosen[wanted[0]][3]
    first_id = chosen[wanted[0]][0]
    first_rec = next(r for r, _rc in judged if str(r["id"]) == first_id)
    restricted = {**{k: v for k, v in first_rc.items() if k not in ("per_adapter", "undecidable", "status", SERVING_VENV_KEY)},
                  "status": status, "per_adapter": sub_per, "undecidable": und, SERVING_VENV_KEY: venv,
                  "donor_launch": first_id, "donor_manifest": first_rec.get("manifest"),
                  "donor_launches": {n: chosen[n][0] for n in wanted}, "restricted_to": wanted}
    how = (f"adapters {wanted} read by name across {len(donors)} launch(es)" if len(donors) > 1 else f"adapters {wanted} read by name")
    return (f"parity record: {', '.join(donors)} ({how}; manifest {m.name} not planned there)", restricted)

def _adapter_parity_block(geometry_pairs: Sequence[Mapping[str, Any]], any_hf_basis: bool) -> Dict[str, Any]:
    """The record-level ``adapter_parity`` block over the SERVING-certified pairs: every pair's status / basis / the
    adapters that disagreed, aggregated by geometry.adapter_parity_aggregate."""
    from sisteer.capeval.serving import geometry as g
    rows = [{"rank": int(p["rank"]), "set": p["set"],
             **{k: (p.get(ADAPTER_PARITY_KEY) or {}).get(k) for k in ("status", "basis", "mismatched_adapters", "certificate_version", SERVING_VENV_KEY)}}
            for p in geometry_pairs]
    block = dict(g.adapter_parity_block(rows, basis=(BASIS_HF_PEFT if any_hf_basis else None)))
    # the record-level venv (one block when every pair's agrees, else absent) and the certificate versions present
    block[SERVING_VENV_KEY] = _agreeing_venv([r.get(SERVING_VENV_KEY) for r in rows]) if rows else None
    block["certificate_versions"] = sorted({str(r.get("certificate_version")) for r in rows if r.get("certificate_version") is not None})
    # the per-pair rows the aggregate carries may have dropped the two new fields: put them back by (rank, set)
    per_pair = block.get("per_pair")
    if isinstance(per_pair, list):
        by_key = {(int(r["rank"]), r["set"]): r for r in rows}
        for e in per_pair:
            if isinstance(e, Mapping) and (int(e.get("rank", -1)), e.get("set")) in by_key:
                src = by_key[(int(e["rank"]), e["set"])]
                e.setdefault("certificate_version", src.get("certificate_version"))
                e.setdefault(SERVING_VENV_KEY, src.get(SERVING_VENV_KEY))
    return block


def write_verdict(plan: Mapping[str, Any], results: Mapping[str, Mapping[str, Any]], *, probe_job_id: str,
                  out: Optional[Path] = None, verdict_uri: Optional[str] = None,
                  geometry_dir: Optional[Path] = None, probe_cluster: Optional[str] = None,
                  parity: Optional[Mapping[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """``(probe_verdict, allowed_pairs)``. ``parity``: a loaded PARITY-PASS record (:func:`load_parity_record`) —
    assertion 4 for every pair then reads the parity record's launch ``reference_check`` for the SAME MANIFEST NAME
    (its natural launch at the parity plan's window, at the per-set benign scale) instead of this record's own launches;
    a manifest the parity record has not run is ``reference_missing`` for that pair alone. The parity record's dir,
    result generation and scale table are stamped into the verdict and the menu (``parity_record``); THIS record's
    launches (servability, admissions, the geometry fold) are read as before. A pair is allowed iff, on its
    certifying manifest (single / S1, else S1a + S1c when S1 failed):

    * the natural-cap launch at the ladder window (4,096) PASSED readiness and
      the per-adapter checks for BOTH the single-rank (control) adapter and the
      rank-2r difference adapter;
    * every production window RUN for that manifest passed, and the admitted S
      at every production window is >= 1 (measured where run, else the plan's
      stride/KV estimate);
    * for a two-server set, the S2 launch at the ladder window passed;
    * the HF+PEFT reference check PASSED on its certifying launches. A pair
      with NO reference evidence (``run --no-reference``, a reference-phase
      crash, peft unimportable) is ``reference_missing``: never allowed, the
      verdict is ``uncertified`` and non-certifying, and ``allowed_pairs.json``
      is NOT written — a certification check that cannot fire certifies
      nothing.

    ``allowed_pairs.json`` schema (the downstream agent consumes it): ``{"model_id",
    "served_hf_id", "revision", "probe_job_id", "pairs": [{"rank", "set",
    "lora_modules", "per_expert_rank", "max_lora_rank_cap", "topology",
    "trainable_params", "admitted_max_num_seqs": {"<L>": S}, "reference_check_basis",
    "certified_by": [launch ids], "warnings": [...], "served_class": {"max_lora_rank",
    "max_loras", "target_class", "manifest", "certified_by_window": {"<L>": launch id | None},
    "measured_S_by_window": {"<L>": S | None}, "throughput_by_window": {"<L>": [points] | None},
    "standup": {"S_by_window": {"<L>": S} | None, "launches_by_window", "source",
    "batched_token_budget", "refused": str | None}}}],
    "excluded": [{"rank", "set", "reason"}], "verdict_uri", "memo_inputs": {...}}``.
    ``admitted_max_num_seqs`` is the ROLES' certification (the min over the role
    servers' engines per window); ``served_class.measured_S_by_window`` and
    ``throughput_by_window`` are the numbers of the one engine that served the
    pair's whole class there (where the pace was measured); ``served_class.standup``
    is what a steered standup of the pair is SIZED to over the record as folded
    (:func:`standup_resolution`: every certified set's launches, the least S among
    the tightest launches dominating the class, inside the record's minimum budget)
    — the S the serving rows are priced at, with the pace rescaled from the measured
    engine's S.

    The verdict is ``complete`` only when every pair is DECIDED on evidence:
    allowed with no launch outstanding, or excluded by a launch that RAN
    (oom / refused / a failed check / a measured S < 1 / a reference
    mismatch). A pair whose only reasons are estimates over unrun launches
    (``not_run``, an estimated ``ceiling_S<1``, an unrun S2 or fallback) is
    UNDECIDED, the verdict is ``partial`` and non-certifying, and
    ``allowed_pairs.json`` is NOT written — a partial probe must never
    suppress a (rank, set) as excluded.
    The returned ``allowed`` dict still lists the undecided pairs under
    ``excluded`` with ``decided: False`` so a reader sees what is missing.
    Status precedence: ``partial`` (launches outstanding — the run continues)
    over ``uncertified`` (complete, reference evidence missing — a resubmission
    changes nothing) over ``complete``.

    An ``oom`` launch is READ through :func:`oom_kind` against the record's
    MEASURED batched-token budget: a start-time OOM at ``S x L`` above
    the largest batch this record has proven is ``replan_required`` — the
    estimator planned a prefill the engine could never allocate, which says
    nothing about the pair — so it never excludes; the pair is UNDECIDED (the
    verdict ``partial``) until ``run --resume --replan`` re-derives S and the
    launch re-runs. A serving OOM, a start-time OOM at S = 1, or one at a batch
    within the budget is genuine and excludes as before. The verdict carries
    ``replan_required`` (ids), ``batched_token_budget`` and every launch's
    ``oom_kind``.

    GEOMETRY CERTIFICATION AND ADAPTER PARITY ARE SEPARATE FACTS. Every pair's
    reasons split into SERVING reasons (oom / refused / a failed check / S < 1 /
    an unrun launch) and PARITY reasons (``reference_mismatch:…``). A pair with
    no serving reason and nothing outstanding is SERVING-CERTIFIED — its engines
    came up and served adapters of its class at every window — whatever the
    reference said, and every such pair carries ``adapter_parity`` {status
    verified|mismatch|unverified, basis, mismatched_adapters}. A ``complete``
    verdict (and an ``uncertified`` one — reference evidence missing) FOLDS the
    measured serving geometry (:func:`geometry_fold`) over EVERY
    serving-certified pair into ``geometry_record.json`` beside the verdict and
    the site geometry overlay (``geometry_dir``, else
    ``geometry.site_geometry_dir()``), the record's ``adapter_parity`` block
    riding beside the geometry; the fold's outcome rides
    ``verdict["geometry_fold"]``. The MENU (``allowed_pairs.json`` ``pairs``) is
    UNCHANGED: verified parity is still required, so a record whose pairs all
    mismatch has a geometry record (flagged ``mismatch``) and an empty menu — an
    unsteered standup may size from the former, a steered one refuses it by
    name, and the LoRA search waits. The verdict and the menu carry the same
    ``adapter_parity`` block and the ``serving_certified_pairs`` list.

    THE SERVED LAUNCH CLASS. The roles are certified by the split servers, but a
    steered standup of the pair serves them in ONE engine — the pair's union
    class (``standup_manifest.build_union_manifest``; ``(8, 8)`` for ``wo`` at
    r = 4, cells + zero for the other sets) — and ``geometry.least_admitted_by_
    window`` sizes it only from a folded launch whose class dominates it at the
    SAME window. So a pair is serving-certified only when, at EVERY probed
    window, a PASSED launch of the pair with that dominance over the union's
    target class (:func:`_served_class_candidates`; the shared o_proj S2 never
    stands in for an ``mlp`` union) attested the class it was planned at
    (:func:`_resolve_served_class`). A window where every such launch ran and
    none served the class is a serving reason WITH EVIDENCE
    (``served_class_undominated_at_L=<L>:<launch>:<why>``: decided whatever
    else is outstanding); a window where nothing of the class was planned is a
    serving reason WITHOUT evidence (``served_class_unplanned_at_L=<L>:(rank
    cap C, N adapters)``: nothing ran — the reopen's extension plans the union
    there; the pair is decided once nothing else of it is outstanding, and
    undecided while a launch of it still is); an unrun or replan-required
    launch of the class holds the pair undecided like any decision launch. The
    pair record (and an excluded pair's audit trail) carries ``served_class``:
    the class, the plan's manifest of it, and the certifying launch per window.
    A record whose verdict certified a pair the fold never served its class for
    strands every steered standup at ``launch-class geometry`` — the verdict
    says it first, by name."""
    dims = _Dims(plan["dims"])
    ladder_L = int(plan["ladder_window"])
    windows = [int(w) for w in plan["windows"]]
    index = _natural_launch_index(plan)
    outstanding = outstanding_launches(plan, results)
    gen = result_generation(Path(out)) if out is not None else None
    budget = measured_budget(plan, results)
    budget_tokens = budget.get("tokens")
    replan_ids: List[str] = []

    def _replan_oom(rec: Optional[Mapping[str, Any]], res: Optional[Mapping[str, Any]]) -> bool:
        """A start-time prefill OOM above the measured budget: re-plan, never evidence."""
        if rec is None or res is None:
            return False
        return oom_kind(rec, res, budget_tokens) in REPLAN_OOM_KINDS
    undecided_launches: set = set()
    pairs_out: List[Dict[str, Any]] = []
    geometry_pairs: List[Dict[str, Any]] = []   # The SERVING-certified pairs, parity aside — the geometry fold's input
    excluded: List[Dict[str, Any]] = []
    servers: List[Dict[str, Any]] = []
    unreferenced_pairs: List[List[Any]] = []
    any_hf_basis = False
    # THE SERVED LAUNCH CLASS: a pair is serving-certified only
    # when, at EVERY probed window, a PASSED launch of the pair whose attested engine class is its planned one dominates the class a
    # steered standup of the pair serves — the union's (standup_manifest.build_union_manifest) — over the union's target class:
    # exactly what geometry.least_admitted_by_window will ask of the fold at standup time. A window where every launch of that
    # class died is EVIDENCE against the pair (the class cannot come up there; the extension adds no second copy); a window where
    # nothing of the class was planned is a decided exclusion the reopen's extension answers; an unrun / replan-required launch of
    # the class holds the pair undecided like any decision launch. A parity pass plans no union and folds no geometry: not asked.
    served_class_rule = not bool(plan.get("parity_pass"))
    all_manifests_by_name = {n: plan_manifest(plan, n) for n in plan["manifests"]} if served_class_rule else {}
    for pair in plan["pairs"]:
        s, r = pair["set"], int(pair["rank"])
        manifests = [plan_manifest(plan, n) for n in pair["manifests"]]
        s2 = next((m for m in manifests if m.topology == sm.TOPOLOGY_S2), None)
        role_m = _role_manifests([m for m in manifests if m.topology != sm.TOPOLOGY_S2])
        legacy_primary = next((m for m in manifests if m.topology in (sm.TOPOLOGY_S1, sm.TOPOLOGY_SINGLE)), None)
        # the class a steered standup of THIS pair serves: the plan's union when it carries one (a legacy record's own `single` is
        # the shape), else built from the record's dims — the standup's class does not depend on what the record planned
        served_m: Optional[sm.Manifest] = None
        if served_class_rule:
            served_m = next((m for m in manifests if _is_union(m)), None) or sm.build_union_manifest(dims, set_id=s, rank=r)
        served_by_window: Dict[str, Optional[str]] = {}
        # the served class's OWN measurement per window: the engine S and the recorded
        # throughput of the launch that certified the class there — what a steered standup of the pair gets (geometry sizes the
        # class from that launch, the tightest dominating footprint) and what the serving rows are priced from; never the role
        # servers' S / pace, which certify the roles and can admit more (or less) than the one engine serving every adapter
        served_S_by_window: Dict[str, Optional[int]] = {}
        served_tp_by_window: Dict[str, Optional[List[Dict[str, Any]]]] = {}
        warnings: List[str] = []
        certified_by: List[str] = []
        reasons: List[str] = []
        evidence: List[str] = []       # the reasons backed by a launch that RAN (never an estimate over an unrun one)
        pair_replan: List[str] = []    # this pair's launches whose OOM was a prefill above the budget: undecided
        certifying: List[sm.Manifest] = []
        certifying_by_window: Dict[str, Dict[str, Optional[str]]] = {}
        # THE ROLE-KEYED RULE: at EVERY window each required role — the
        # single-rank adapter (control) and the rank-2r difference — is certified by the FIRST PASSED launch of a
        # manifest carrying it, in _role_manifests' order (the current S1cc / S1d servers, then a record's legacy
        # four-slot S1 / single, then its S1a / S1b / S1c fallbacks). A record that mixes topologies — the live one:
        # legacy launches that passed at the short windows, the current servers re-run where the legacy servers OOM'd —
        # reads under one rule; a role whose launch is unrun or replan-required is UNDECIDED (never excluded
        # while a launch could still certify it); a role whose every launch died is evidence against the pair.
        admitted: Dict[str, Any] = {}
        windows_measured: List[int] = []
        windows_estimated: List[int] = []
        basis = None
        for L in windows:
            S_here: Optional[int] = None
            by_role: Dict[str, Optional[str]] = {}
            for role in sm.ROLES_REQUIRED:
                kind, row, rows = _resolve_role_launch(role_m.get(role, []), L, results, index, _replan_oom)
                tag = "single" if role == sm.ROLE_CONTROL else role
                if kind == _ROW_UNPLANNED:
                    if L == ladder_L:
                        reasons.append("not_run")
                        evidence.append(reasons[-1])          # never planned: nothing can certify it
                    by_role[role] = None
                    continue
                m, rec, res = row
                by_role[role] = rec["id"]
                if kind == _ROW_PASSED:
                    S_m = _measured_S(rec, res)
                    if L not in windows_measured:
                        windows_measured.append(L)
                    if rec["id"] not in certified_by:
                        certified_by.append(rec["id"])
                    if m not in certifying:
                        certifying.append(m)
                    if m.topology in sm.S1_FALLBACKS and legacy_primary is not None:
                        recp, resp = _launch_for(results, index, legacy_primary.name, L)
                        w = (f"{legacy_primary.topology} manifest {_fail_reason(recp or {'cap': legacy_primary.max_lora_rank}, resp)} at "
                             f"L={L}; {role} certified on the three-server fallback {m.topology}")
                        if w not in warnings:
                            warnings.append(w)
                    if L == ladder_L:
                        ok, why = _adapter_ok(res, role)
                        if not ok:                            # a passed launch's adapters pass; kept for a hand-edited record
                            reasons.append(f"{tag}_failed:{why}")
                            evidence.append(reasons[-1])
                elif kind == _ROW_UNRUN:
                    S_m = int(rec["admitted_S"])
                    if L not in windows_estimated:
                        windows_estimated.append(L)
                    if L == ladder_L and "not_run" not in reasons:
                        reasons.append("not_run")                 # the headline of a fresh record; an estimate, never evidence
                elif kind == _ROW_REPLAN:
                    pair_replan.append(rec["id"])
                    continue
                elif kind == _ROW_FAILED:
                    # the engine came up and a check failed: judge THIS role's adapter by name and the other hard checks
                    if L == ladder_L:
                        ok, why = _adapter_ok(res, role)
                        if not ok:
                            reasons.append(f"{tag}_failed:{why}")
                            evidence.append(reasons[-1])
                        other = [f for f in ((res.get("checks") or {}).get("failed") or []) if not str(f).startswith("adapters/")]
                        if other:
                            reasons.append(f"failed:{','.join(other)}")
                            evidence.append(reasons[-1])
                    else:
                        reasons.append(f"{_fail_reason(rec, res)}_at_L={L}")
                        evidence.append(reasons[-1])
                    if rec["id"] not in certified_by:
                        certified_by.append(rec["id"])
                    if m not in certifying:
                        certifying.append(m)
                    S_m = 0
                else:                                         # _ROW_DOWN: every candidate ran and died
                    legacy_rows = [(mm, rr, ss) for mm, rr, ss in rows if mm.topology in (sm.TOPOLOGY_S1, sm.TOPOLOGY_SINGLE)]
                    fb_rows = [(mm, rr, ss) for mm, rr, ss in rows if mm.topology in sm.S1_FALLBACKS]
                    if legacy_rows and fb_rows and L == ladder_L:
                        # the legacy wording: the four-slot server's reason, then the fallback that died in its place
                        mm, rr, ss = legacy_rows[0]
                        fm, fr, fs = fb_rows[0]
                        reason = f"{_fail_reason(rr, ss)};fallback_{fm.topology}:{_fail_reason(fr, fs)}"
                    else:
                        if role == sm.ROLE_DIFFERENCE and by_role.get(sm.ROLE_CONTROL) not in (None, rec["id"]):
                            # control was certified elsewhere: name the DEDICATED difference server that died (S1d, a
                            # legacy S1c) when one did, else the first dead candidate
                            dedicated = [(mm, rr, ss) for mm, rr, ss in rows if mm.genuine_roles == (sm.ROLE_DIFFERENCE,)
                                         and ss is not None and ss.get("status") in (STATUS_OOM, STATUS_REFUSED, STATUS_ERROR)]
                            if dedicated:
                                m, rec, res = dedicated[0]
                            reason = _fail_reason(rec, res) if L == ladder_L else f"{_fail_reason(rec, res)}_at_L={L}"
                            reason = f"difference_on_{m.topology}:{reason}"
                        else:
                            reason = _fail_reason(rec, res) if L == ladder_L else f"{_fail_reason(rec, res)}_at_L={L}"
                    reasons.append(reason)
                    evidence.append(reason)
                    S_m = 0
                S_here = S_m if S_here is None else min(S_here, S_m)
            certifying_by_window[str(L)] = by_role
            if S_here is not None:
                admitted[str(L)] = S_here
                if S_here < 1:
                    reasons.append(f"ceiling_S<1_at_L={L}")
                    if L in windows_measured:                 # a MEASURED engine S < 1 is evidence; the estimate is not
                        evidence.append(reasons[-1])
            if served_m is not None:
                # the served class at this window (the block comment above the pair loop), AFTER the roles' reasons for it: passed =
                # certified here; dead = EVIDENCE (a launch of the class ran there and did not serve it); never planned = a reason,
                # not evidence (nothing ran — the reopen's extension plans it; while another launch of the pair is still outstanding
                # the pair stays undecided, and with nothing outstanding the reason decides it); unrun / replan = undecided, held by
                # the pair's outstanding launches below
                kind_c, rec_c, detail = _resolve_served_class(
                    _served_class_candidates(plan, pair, served_m, L, all_manifests_by_name), results, all_manifests_by_name, _replan_oom)
                S_c: Optional[int] = None
                tp_c: Optional[List[Dict[str, Any]]] = None
                if kind_c == _SERVED_PASSED and rec_c is not None:
                    res_c = results[rec_c["id"]]
                    S_c = _measured_S(rec_c, res_c)
                    if S_c < 1:
                        # the class came up and admits NOTHING there: evidence against the pair, as a role's measured S < 1 is
                        # (`ceiling_S<1_at_L`) — a certified class is one a standup can be sized from
                        kind_c, detail, S_c = _SERVED_DEAD, "ceiling_S<1", None
                    else:
                        tp_c = res_c.get("throughput")    # the driver's closed-loop points at THROUGHPUT_WINDOWS; None elsewhere
                served_by_window[str(L)] = rec_c["id"] if kind_c == _SERVED_PASSED and rec_c is not None else None
                served_S_by_window[str(L)] = S_c
                served_tp_by_window[str(L)] = tp_c
                if kind_c == _SERVED_DEAD and rec_c is not None:
                    reasons.append(f"{REASON_SERVED_CLASS}_at_L={L}:{rec_c['id']}:{detail}")
                    evidence.append(reasons[-1])
                elif kind_c == _SERVED_UNPLANNED:
                    reasons.append(f"{REASON_SERVED_CLASS_UNPLANNED}_at_L={L}:(rank cap {served_m.max_lora_rank}, "
                                   f"{served_m.max_loras} adapters)")
        # S2 for two-server sets (the shared residual server, at the ladder window)
        if s2 is not None:
            rec2, res2 = _launch_for(results, index, s2.name, ladder_L)
            if res2 is not None and _replan_oom(rec2, res2):
                pair_replan.append(rec2["id"])
            elif res2 is None or res2.get("status") != STATUS_PASSED:
                reasons.append(f"s2_{_fail_reason(rec2 or {'cap': s2.max_lora_rank}, res2)}")
                if res2 is not None or rec2 is None:
                    evidence.append(reasons[-1])
            else:
                certified_by.append(rec2["id"])
        if not certifying:
            # nothing certified any role: the pair-level caps name the servers the plan would have certified it on — the
            # current split servers when the plan carries them (the union beside them certifies a class, not a role), else a
            # legacy record's primary, else whatever is not S2
            certifying = [m for m in manifests if m.topology in (sm.TOPOLOGY_S1CC, sm.TOPOLOGY_S1D)] or \
                         [m for m in manifests if m.topology != sm.TOPOLOGY_S2 and m.topology not in sm.S1_FALLBACKS] or \
                         [m for m in manifests if m.topology != sm.TOPOLOGY_S2]
        caps_used = {m.topology: m.max_lora_rank for m in certifying}
        moe_rank_used = {m.topology: m.moe_lora_rank for m in certifying}
        # the reference check: HF+PEFT parity on EVERY certifying launch, or the pair is `reference_missing`. A judged launch
        # is pass / fail / undecidable (the adapter's own one-ulp perturbation flips its top-1 — a DECIDED exclusion, evidence,
        # never missing evidence); fail outranks undecidable outranks pass across the certifying launches
        ref_status = None
        ref_seen: List[str] = []
        ref_missing: List[str] = []
        undecidable_adapters: List[str] = []
        # with a parity record, assertion 4 reads ITS launch for each certifying manifest (same name, the parity window)
        sources = (parity_reference_sources(plan, certified_by, parity) if parity is not None
                   else [(lid, (results.get(lid) or {}).get("reference_check")) for lid in certified_by])
        for label, rc in sources:
            if rc is None:
                ref_missing.append(f"{label}: no reference_check recorded (run --no-reference, or the reference phase never ran)")
                continue
            if rc.get("basis") == BASIS_HF_PEFT:
                st = rc.get("status")
                if st == REF_STATUS_UNDECIDABLE:
                    # judged and NOT decidable at this scale (chaos / too weak): a DECIDED exclusion, a parity reason
                    names = sorted((rc.get("undecidable") or {}).keys())
                    undecidable_adapters += names
                    reasons.append(REASON_UNDECIDABLE_PREFIX + ",".join(names))
                    evidence.append(reasons[-1])
                    ref_status = "fail" if ref_status == "fail" else "undecidable"
                elif st == "fail":
                    ref_status = "fail"
                    reasons.append(PARITY_REASON_PREFIX + ",".join(
                        n for n, v in (rc.get("per_adapter") or {}).items() if v.get("status") != "pass"))
                    evidence.append(reasons[-1])
                else:
                    ref_status = ref_status if ref_status in ("fail", "undecidable") else "pass"
            else:
                ref_missing.append(f"{label}: reference {rc.get('status')}" + (f" — {rc.get('reason')}" if rc.get("reason") else ""))
        if ref_missing and ref_status != REF_STATUS_FAIL:
            # EVERY certifying launch / manifest that carries reference adapters needs JUDGED evidence: a pass (or an
            # undecidable) on one of them never covers another that carries none — an S2 with nothing to judge reads `pass` under
            # the hf_peft basis and used to mask an S1 whose adapters were `reference_missing`; the pair is
            # `reference_missing` (UNCERTIFIED), never allowed. A judged FAIL still stands: a real mismatch is never relabelled.
            ref_status = None
        if ref_status is not None:
            basis = BASIS_HF_PEFT
            any_hf_basis = True
        else:
            # NO HF+PEFT evidence on this pair: never a substitute basis, never an allowed pair
            basis = None
        reference_missing_reason = "reference_missing:" + "; ".join(ref_missing or
                                                                    ["no certifying launch carries a reference_check"])
        # memo: per-server admission rows
        for m in manifests:
            row = {"pair": [r, s], "manifest": m.name, "topology": m.topology, "target_class": list(m.target_class),
                   "cap": m.max_lora_rank, "max_loras": m.max_loras, "admitted_by_window": {}}
            for L in windows:
                recL, resL = _launch_for(results, index, m.name, L)
                if recL is None:
                    continue
                eng = ((resL or {}).get("engine") or {}).get("max_num_seqs_engine") if resL else None
                row["admitted_by_window"][str(L)] = {
                    "requested": recL["requested_S"], "planned": recL["admitted_S"],
                    "engine": eng, "binding": recL["admission"]["binding"],
                    "status": (resL or {}).get("status", STATUS_NOT_RUN),
                    "buffer_bytes": ((resL or {}).get("checks") or {}).get("buffer_bytes"),
                    "kv_capacity_tokens": (((resL or {}).get("checks") or {}).get("memory") or {}).get("kv_capacity_tokens"),
                    "throughput": (resL or {}).get("throughput")}
            servers.append(row)
        reasons = list(dict.fromkeys(reasons))
        # the pair's outstanding launches: every planned launch over its manifests that has not run to a terminal state
        pair_manifests = set(pair["manifests"])
        pair_outstanding = [rec["id"] for rec in outstanding
                            if rec["kind"] in DECISION_KINDS and pair_manifests.intersection(rec["serves"])]
        # a replan_required launch of this pair is as open as an unrun one: it holds the pair undecided — over EVERY
        # decision launch of the pair, not only the ones a role resolved to: the pair's union (the served launch class, 2026-10-11)
        # certifies no role, and its start-time prefill OOM above the budget is re-run by `--replan` exactly like a server's
        for rec_p in plan["launches"]:
            if (rec_p["kind"] in DECISION_KINDS and pair_manifests.intersection(rec_p["serves"]) and rec_p["id"] not in pair_replan
                    and _replan_oom(rec_p, results.get(rec_p["id"]))):
                pair_replan.append(rec_p["id"])
        pair_outstanding += [lid for lid in dict.fromkeys(pair_replan) if lid not in pair_outstanding]
        replan_ids += [lid for lid in pair_replan if lid not in replan_ids]
        decided = bool(evidence) or not pair_outstanding
        if pair_outstanding and not evidence:
            undecided_launches.update(pair_outstanding)
        pair_rec = {"rank": r, "set": s, "lora_modules": list(pair["lora_modules"]),
                    "per_expert_rank": pair.get("per_expert_rank"),
                    "max_lora_rank_cap": max(caps_used.values()), "caps_by_server": caps_used,
                    # the expert-buffer rank per certifying server (None on a class that wraps no expert)
                    "moe_lora_rank_by_server": moe_rank_used,
                    "topology": sm.topology_label(certifying + ([s2] if s2 else [])),
                    # the launch that certified each required role at each window (the role-keyed rule's audit trail)
                    "certifying_by_window": certifying_by_window,
                    "trainable_params": int(pair["trainable_params"]), "admitted_max_num_seqs": admitted,
                    "windows_measured": sorted(windows_measured), "windows_estimated": sorted(windows_estimated),
                    "reference_check_basis": basis, "certified_by": certified_by, "warnings": warnings,
                    # the served launch class (2026-10-11): the class a steered standup of the pair serves, the plan's manifest of
                    # it when one is planned, the launch that certified it at each window (None: undecided or not served
                    # there — the reasons say which), and THAT launch's measured S and recorded throughput per window — the
                    # class's own numbers, which the serving rows price from (`replan_compute.py`), beside the roles'
                    # `admitted_max_num_seqs`, which stays the roles' certification; None on a parity pass, which is not asked
                    "served_class": ({"max_lora_rank": int(served_m.max_lora_rank), "max_loras": int(served_m.max_loras),
                                      "target_class": list(served_m.target_class),
                                      "manifest": (served_m.name if served_m.name in plan["manifests"] else None),
                                      "certified_by_window": served_by_window,
                                      "measured_S_by_window": served_S_by_window,
                                      "throughput_by_window": served_tp_by_window} if served_m is not None else None)}
        # an excluded pair keeps its certification trail beside the reason — what DID certify (the roles, their servers, the
        # served class per window) is how the reader tells a pair the reopen can finish from one whose class died at a window
        audit = {k: pair_rec[k] for k in ("certified_by", "certifying_by_window", "caps_by_server", "moe_lora_rank_by_server",
                                          "per_expert_rank", "served_class")}
        # geometry certification and adapter parity are SEPARATE facts. The pair's SERVING is certified when every reason
        # against it is a parity reason (or there is none) and nothing is outstanding: its engines came up, served adapters of
        # its class and admitted S >= 1 at every window it ran — the geometry fold reads exactly those pairs, whatever the
        # HF+PEFT reference said about the adapters' numerics; the MENU below still requires verified parity.
        parity_reasons = [x for x in reasons if x.startswith(PARITY_REASON_PREFIXES)]
        serving_reasons = [x for x in reasons if not x.startswith(PARITY_REASON_PREFIXES)]
        # which certificate judged the pair's adapters — 2 iff every judged adapter carries a v2 record, 1 iff none, else "mixed"
        cert_versions = [int((v or {}).get("certificate_version") or 1) for _lbl, rc in sources if isinstance(rc, Mapping)
                         for v in (rc.get("per_adapter") or {}).values() if isinstance(v, Mapping)
                         and v.get("status") in (REF_STATUS_PASS, REF_STATUS_FAIL, REF_STATUS_UNDECIDABLE)]
        cert_version: Any = (None if not cert_versions else 2 if all(c == 2 for c in cert_versions) else 1 if all(c == 1 for c in cert_versions) else "mixed")
        # the venv the pair's parity was MEASURED in — one block when every judged source names the same digest, else absent
        # (the admission gate refuses an absent block as unproven, by name)
        venv_blocks = [(rc.get(SERVING_VENV_KEY) if isinstance(rc, Mapping) else None) for _lbl, rc in sources if isinstance(rc, Mapping)
                       and rc.get("basis") == BASIS_HF_PEFT and (rc.get("per_adapter") or {})]
        pair_venv = _agreeing_venv(venv_blocks) if venv_blocks else None
        # the SERVING launches (the engines whose geometry certifies the pair) record their venv — the engine runs
        # vLLM's vendored kernels and imports no overlay package, so serving facts do not move with the venv — recorded beside the
        # parity venv, never compared to it
        serving_launch_venvs = [(results.get(lid) or {}).get(SERVING_VENV_KEY) for lid in certified_by]
        pair_rec["serving_launches_venv"] = _agreeing_venv(serving_launch_venvs) if serving_launch_venvs else None
        pair_rec["serving_launches_venv_mixed"] = bool(serving_launch_venvs) and pair_rec["serving_launches_venv"] is None and any(
            isinstance(b, Mapping) and b.get("requirements_sha256") for b in serving_launch_venvs)
        pair_rec[ADAPTER_PARITY_KEY] = {
            "status": (PARITY_VERIFIED if ref_status == REF_STATUS_PASS else PARITY_MISMATCH if ref_status == REF_STATUS_FAIL else PARITY_UNVERIFIED),
            "basis": basis,
            "certificate_version": cert_version,
            SERVING_VENV_KEY: pair_venv,
            "mismatched_adapters": sorted({n for x in parity_reasons if x.startswith(PARITY_REASON_PREFIX)
                                           for n in x[len(PARITY_REASON_PREFIX):].split(",") if n}),
            # judged but undecidable at this scale — unverified for the geometry's block, EXCLUDED (decided) from the menu
            "undecidable_adapters": sorted(set(undecidable_adapters)),
            "reference_missing": (list(ref_missing) if ref_status is None else [])}
        pair_rec["serving_certified"] = not serving_reasons and not pair_outstanding
        if pair_rec["serving_certified"]:
            geometry_pairs.append(pair_rec)
        if reasons:
            excluded.append({"rank": r, "set": s, "reason": reasons[0], "reasons": reasons, "decided": decided,
                             "outstanding": pair_outstanding,
                             "topology": pair_rec["topology"], "max_lora_rank_cap": pair_rec["max_lora_rank_cap"], **audit})
        elif pair_outstanding:
            # every launch that ran passed, but a decision-relevant launch of this pair has not run: undecided
            excluded.append({"rank": r, "set": s, "reason": "not_run", "reasons": ["not_run"], "decided": False,
                             "outstanding": pair_outstanding,
                             "topology": pair_rec["topology"], "max_lora_rank_cap": pair_rec["max_lora_rank_cap"], **audit})
        elif ref_status is None:
            # every vLLM-side check passed and nothing is outstanding, but there is NO HF+PEFT parity evidence: the pair
            # is NOT certified — listed with `reference_missing` so the reader sees exactly which launch lacks it
            excluded.append({"rank": r, "set": s, "reason": reference_missing_reason, "reasons": [reference_missing_reason],
                             "decided": True, "reference_missing": True, "outstanding": [],
                             "topology": pair_rec["topology"], "max_lora_rank_cap": pair_rec["max_lora_rank_cap"], **audit})
            unreferenced_pairs.append([r, s])
        else:
            pairs_out.append(pair_rec)
    ladder = [{"id": rec["id"], "set": rec["set"], "topology": rec["topology"], "cap": rec["cap"], "L": rec["L"],
               "moe_lora_rank": rec.get("moe_lora_rank"),
               "status": (results.get(rec["id"]) or {}).get("status", STATUS_NOT_RUN),
               "expected_lora_bytes": rec["expected_lora_bytes"]["total"],
               "measured_buffer_bytes": (((results.get(rec["id"]) or {}).get("checks") or {}).get("buffer_bytes") or {}).get("measured"),
               "estimated_total_gb": rec["estimated_vram"]["total"] / 1e9}
              for rec in plan["launches"] if rec["kind"] in ("ladder", "natural")
              and _carries_control(plan_manifest(plan, rec["manifest"])) and int(rec["L"]) == ladder_L]
    # the memo's parity terms come from where assertion 4 was READ: the parity record's launches when one was given
    memo_inputs = build_memo_inputs(plan, results, pairs_out, excluded, servers, ladder, dims,
                                    parity_results=(parity.get("results") if parity is not None else None))
    # precedence: launches outstanding (the run continues) > reference evidence missing (nothing to resume) > complete
    if undecided_launches:
        status = VERDICT_PARTIAL
    elif unreferenced_pairs:
        status = VERDICT_UNCERTIFIED
    else:
        status = VERDICT_COMPLETE
    adapter_parity = _adapter_parity_block(geometry_pairs, any_hf_basis)     # The second fact, beside the geometry
    verdict = {"version": 1, "written_utc": utc_now(), "model_key": plan["model_key"], "model_id": plan["model_id"],
               "served_hf_id": plan["served_hf_id"], "revision": plan["revision"], "probe_job_id": probe_job_id,
               "host": socket.gethostname(), "plan_launches": plan["n_launches"],
               "status": status, "certifying": status == VERDICT_COMPLETE,
               "outstanding": [{"id": r["id"], "kind": r["kind"], "L": r["L"], "cap": r["cap"],
                                "decision_relevant": r["id"] in undecided_launches} for r in outstanding],
               "undecided_pairs": [[e["rank"], e["set"]] for e in excluded if not e.get("decided", True)],
               "unreferenced_pairs": unreferenced_pairs,
               # the launches whose OOM was a prefill above the measured budget — re-derive S and re-run them
               # (`run --resume --replan`) before this verdict may be read; the budget they are read against
               "replan_required": [rec["id"] for rec in plan["launches"] if rec["id"] in set(replan_ids)],
               "batched_token_budget": budget,
               "plan_batched_token_budget": plan.get("batched_token_budget"),
               "launches": [{"id": rec["id"], "kind": rec["kind"], "manifest": rec["manifest"], "set": rec["set"],
                             "rank": rec["rank"], "topology": rec["topology"], "cap": rec["cap"], "L": rec["L"],
                             "admitted_S": rec["admitted_S"],
                             "status": res.get("status", STATUS_NOT_RUN),
                             "oom_kind": oom_kind(rec, res, budget_tokens),
                             # the engine-log lines that say it died at start (detail for the reader; the kind is decided
                             # by ready_s and the batch against the budget, never by grepping)
                             "start_oom_signatures": (_start_oom_signatures(out, rec["id"], res)
                                                      if res.get("status") == STATUS_OOM else None),
                             "batched_tokens": (_launch_tokens(rec, res)[1] if res else None),
                             "checks_failed": (res.get("checks") or {}).get("failed"),
                             "failure": (res.get("failure") or {}).get("message")}
                            for rec in plan["launches"] for res in (results.get(rec["id"]) or {},)],
               "pairs": pairs_out, "excluded": excluded, "servers": servers, "ladder": ladder,
               "reference_check_basis": BASIS_HF_PEFT if any_hf_basis else BASIS_VLLM_SIDE,
               "memo_inputs": memo_inputs, "verdict_uri": verdict_uri,
               # the SERVING-certified pairs (the geometry fold's input) and their adapter parity — separate from the menu
               ADAPTER_PARITY_KEY: adapter_parity, "serving_certified_pairs": [[int(p["rank"]), p["set"]] for p in geometry_pairs],
               # the parity-pass record assertion 4 was read from (None = this record's own launches), and whether THIS
               # record is itself a parity pass (its menu is a parity summary, never the LoRA search's P)
               PARITY_RECORD_KEY: parity_record_stamp(parity), "parity_pass": bool(plan.get("parity_pass")),
               # the result generation this verdict was computed over (hpo_probe_status verifies it against the record
               # before the front door fetches anything; an older stamp = a reopened record = never READY)
               RESULT_GENERATION_KEY: (gen or {}).get("generation"), "n_results": (gen or {}).get("n_results")}
    allowed = {"model_id": plan["model_id"], "served_hf_id": plan["served_hf_id"], "revision": plan["revision"],
               "model_key": plan["model_key"], "probe_job_id": probe_job_id, "written_utc": verdict["written_utc"],
               "status": status, "certifying": status == VERDICT_COMPLETE,
               RESULT_GENERATION_KEY: (gen or {}).get("generation"), "n_results": (gen or {}).get("n_results"),
               "pairs": pairs_out, "excluded": [{k: v for k, v in e.items() if k in ("rank", "set", "reason", "decided")}
                                                for e in excluded],
               "unreferenced_pairs": unreferenced_pairs, "replan_required": verdict["replan_required"],
               ADAPTER_PARITY_KEY: adapter_parity, "serving_certified_pairs": verdict["serving_certified_pairs"],
               PARITY_RECORD_KEY: verdict[PARITY_RECORD_KEY], "parity_pass": verdict["parity_pass"],
               "batched_token_budget": budget,
               "verdict_uri": verdict_uri, "memo_inputs": memo_inputs}
    if status != VERDICT_COMPLETE and out is not None:
        # a non-certifying verdict REVOKES what an earlier one certified: the standing menu, geometry record, memo, verdict
        # and this record's site overlay move to launches/superseded/<utc>/ (history kept) — a menu must never stand
        # beside a partial verdict
        archive_derived_artifacts(Path(out), why=f"verdict {status}: no certification artifact may stand beside a non-certifying verdict",
                                  model_key=str(plan["model_key"]), only=(ALLOWED_PAIRS_FILE, VERDICT_FILE, GEOMETRY_RECORD_FILE, MEMO_FILE),
                                  geometry_dir=geometry_dir)
    if plan.get("parity_pass"):
        # a parity-pass record certifies ADAPTER PARITY only — its engines ran at the ladder window with benign-scale
        # adapters; the serving geometry is the MAIN record's, so nothing is folded, written or staged from here
        verdict["geometry_fold"] = {"status": "skipped", "reason": "parity pass record (D163): the serving geometry is the main "
                                                                   "record's — a parity record folds nothing and stages nothing"}
    elif status in (VERDICT_COMPLETE, VERDICT_UNCERTIFIED):
        # the geometry fold: from every SERVING-certified pair — parity aside — into the record and the
        # site overlay, the parity facts riding beside the geometry; a PARTIAL verdict (a launch outstanding) folds nothing
        verdict["geometry_fold"] = geometry_fold(plan, results, geometry_pairs, out=out, probe_job_id=probe_job_id,
                                                 verdict_uri=verdict_uri, geometry_dir=geometry_dir, probe_cluster=probe_cluster)
    else:
        verdict["geometry_fold"] = {"status": "not_folded", "reason": f"verdict {status}: a launch that could still move a pair is "
                                                                      "outstanding — nothing is folded until the record is complete"}
    # every serving-certified pair's served_class says what a steered standup of it is sized to over THIS fold:
    # the same dicts ride verdict["pairs"], allowed_pairs.json and a parity-excluded pair's audit trail
    attach_standup_resolution(geometry_pairs, verdict["geometry_fold"])
    if out is not None:
        write_json_atomic(Path(out) / VERDICT_FILE, verdict)
        if status == VERDICT_COMPLETE:
            write_json_atomic(Path(out) / ALLOWED_PAIRS_FILE, allowed)
        elif status == VERDICT_PARTIAL:
            # a PARTIAL verdict certifies nothing: the LoRA search's P must not exist until every pair is decided
            log(f"verdict PARTIAL: {len(undecided_launches)} decision-relevant launch(es) outstanding for "
                f"{verdict['undecided_pairs']} — {ALLOWED_PAIRS_FILE} NOT written"
                + (f"; {len(verdict['replan_required'])} launch(es) need `run --resume --replan` (start-time prefill OOM "
                   f"above the measured budget {budget.get('tokens')}): {verdict['replan_required'][:8]}"
                   if verdict["replan_required"] else ""))
        else:
            log(f"verdict UNCERTIFIED: no HF+PEFT reference evidence for {unreferenced_pairs} — {ALLOWED_PAIRS_FILE} "
                "NOT written (D55: a certification check that cannot fire certifies nothing)"
                f"; geometry fold {verdict['geometry_fold'].get('status')} with adapter_parity "
                f"{adapter_parity['status']} (D145: an unsteered standup may size from it; a steered one refuses it by name)")
    return verdict, allowed


# --------------------------------------------------------------------------
# the geometry fold
# --------------------------------------------------------------------------

def _sha256_file(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def _launch_attestation(out: Optional[Path], lid: str, res: Mapping[str, Any]) -> Dict[str, Any]:
    """The sha256 of what attests this launch: ``launches/<lid>/serving_attestation.json`` (the lora-native readiness
    attestation ``launch_server`` wrote) when present, else the probe's own launch record ``launches/<lid>.json``, else
    (no record dir) the canonical JSON of the in-memory result."""
    if out is not None:
        from sisteer.capeval.serving import vllm_server as vs
        att = Path(out) / LAUNCHES_DIR / lid / vs.SERVING_ATTESTATION_FILE
        if att.is_file():
            return {"file": str(att), "sha256": _sha256_file(att)}
        rec = launch_result_path(Path(out), lid)
        if rec.is_file():
            return {"file": str(rec), "sha256": _sha256_file(rec)}
    body = json.dumps(res, sort_keys=True, default=_json_default).encode("utf-8")
    return {"file": None, "sha256": hashlib.sha256(body).hexdigest()}


PROBE_CLUSTER_ENV = "SISTEER_CLUSTER_LABEL"   # the job env's cluster label (reproduce/lib/slurm.sh slurm_job_env): the overlay's probe_cluster


def launch_class_fields(res: Mapping[str, Any], lid: str, problems: List[str]) -> Optional[Dict[str, int]]:
    """The LoRA class the engine served on a PASSED launch — ``max_lora_rank`` / ``max_loras`` from its worker report (``result.engine``)
    — folded beside S and L (2026-10-02): the unsteered site default sizes a standup from the launches whose class DOMINATES the one it
    serves (geometry.least_admitted_by_window), so a rank-64 x 128-adapter ceiling no longer caps a rank-1 x 3-adapter standup (the s2r pair
    target 831867). A passed launch whose report lacks a positive value is a problem by name (None): a class is never guessed."""
    e = res.get("engine") or {}
    out: Dict[str, int] = {}
    for k in ("max_lora_rank", "max_loras"):
        v = e.get(k)
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            problems.append(f"{lid}: passed but the engine reported no positive {k} ({v!r}) — the launch's adapter class cannot be folded, "
                            "and is never guessed")
            return None
        out[k] = int(v)
    return out


#: THE STANDUP RESOLUTION: the fold's block, keyed ``"<rank>:<set>"`` over the
#: serving-certified pairs, and the same block on each pair record as ``served_class.standup``. A steered standup of a pair is sized by
#: ``native_standup.launch_class_geometry`` -> ``geometry.resolve_geometry(model_key, served_lora)`` -> ``budget_for`` with NO module
#: set -> ``geometry.unsteered_site_set``: EVERY certified set's folded launches merged, the least admitted S among the TIGHTEST
#: launches dominating the pair's served class at each window, inside the record's MINIMUM budget ``// L`` (``ModelBudget.seqs_at``)
#: — so another set's launch of the SAME class (a down union at S 20 beside an mlp union at S 9) caps this pair's standup below its
#: own union's S. ``S_by_window`` is that S per plan window, ``launches_by_window`` the launches that set it, ``source`` geometry's
#: ``window_seqs_source`` (``launch_class:<cap>x<slots>``), ``batched_token_budget`` the record's minimum; ``refused`` names why the
#: standup itself would refuse (no ``wo`` set certified, nothing dominating the class at a window) and leaves the rest None. The
#: serving rows (reproduce/tools/replan_compute.py) price ``S_by_window``, never ``measured_S_by_window`` alone (the union's own engine,
#: where the pace was measured — the pace is rescaled from it).
STANDUP_S_KEY = "standup_S"


def _standup_refused(reason: str) -> Dict[str, Any]:
    return {"S_by_window": None, "launches_by_window": None, "source": None, "batched_token_budget": None, "refused": reason}


def standup_resolution(record: Mapping[str, Any], plan: Mapping[str, Any], pairs: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """:data:`STANDUP_S_KEY`'s block over ``record`` (the fold's site record, as ``geometry.site_geometry_record`` returned it) for
    every pair in ``pairs`` that carries a ``served_class`` — ``geometry.unsteered_row_of(record, served_lora=(cap, slots))`` and its
    ``budget.seqs_at(L)`` at each plan window: the resolution ``launch_class_geometry`` applies, by the same function. A pair whose
    class the record cannot size (``SiteGeometryError``: no ``wo`` set, nothing dominating at a window) is recorded ``refused`` with
    geometry's own words — the standup refuses by them too."""
    from sisteer.capeval.serving import geometry as g
    site = g.parse_site_geometry(record, model_key=str(record["model_key"]))
    windows = [int(L) for L in plan["windows"]]
    out: Dict[str, Dict[str, Any]] = {}
    for p in pairs:
        sc = p.get("served_class")
        if not isinstance(sc, Mapping):
            continue
        key = f"{int(p['rank'])}:{p['set']}"
        cls = (int(sc["max_lora_rank"]), int(sc["max_loras"]))
        try:
            row = g.unsteered_row_of(site, served_lora=cls)
        except g.SiteGeometryError as e:
            out[key] = _standup_refused(str(e))
            continue
        b = row.budget
        out[key] = {"S_by_window": {str(L): int(b.seqs_at(L)) for L in windows},
                    "launches_by_window": {str(L): list((b.window_seqs_launches or {}).get(L) or ()) for L in windows},
                    "source": b.window_seqs_source, "batched_token_budget": int(b.batched_token_budget), "refused": None}
    return out


def attach_standup_resolution(pairs: Sequence[Dict[str, Any]], fold: Mapping[str, Any]) -> None:
    """Write the fold's :data:`STANDUP_S_KEY` entry onto each pair record's ``served_class["standup"]`` (the pair records of
    ``verdict["pairs"]`` / ``allowed_pairs.json`` and, through the shared audit dict, an excluded pair's trail); a fold that built no
    record (``refused`` / ``not_applicable`` / ``not_folded`` / ``skipped``) leaves every pair ``refused`` with the fold's own status and
    reason — a verdict never carries a served class without saying what a standup of it is sized to, or why nothing is."""
    by_pair = fold.get(STANDUP_S_KEY) or {}
    for p in pairs:
        sc = p.get("served_class")
        if not isinstance(sc, dict):
            continue
        key = f"{int(p['rank'])}:{p['set']}"
        sc["standup"] = by_pair.get(key) or _standup_refused(f"geometry fold {fold.get('status')}: {fold.get('reason')}")


def geometry_fold(plan: Mapping[str, Any], results: Mapping[str, Mapping[str, Any]], pairs_out: Sequence[Mapping[str, Any]],
                  *, out: Optional[Path], probe_job_id: str, verdict_uri: Optional[str],
                  geometry_dir: Optional[Path] = None, probe_cluster: Optional[str] = None) -> Dict[str, Any]:
    """Fold the SERVING-certified pairs' measured serving geometry, PER MODULE
    SET (every pair whose engines came up and served adapters of its class,
    whatever its HF+PEFT parity — ``pairs_out`` is write_verdict's
    ``geometry_pairs``, and each set's ``adapter_parity`` block rides beside its
    geometry), into ONE per-model record (``geometry.site_geometry_record``) and write
    it to ``<out>/geometry_record.json`` and the site geometry overlay
    (``geometry.write_site_geometry``; ``geometry_dir``, else
    ``geometry.site_geometry_dir()``'s env rule). ``geometry.rows_for`` /
    ``budget_for`` / ``kernel_ceiling_adjustment`` resolve the record of the
    SELECTED set (``module_set=``) BEFORE their built-in tables.

    Per set (every set with at least one serving-certified pair):
    * ``lora_max_module_dim`` = the widest ``checks.executing_stride.reported_D``
      — the worker's MEASURED executing wrapper stride, a hard gate — over that
      set's allowed pairs' certifying launches. A native standup of a design in
      this set registers adapters of exactly this class, so its ceiling is THIS D
      (a model-wide maximum over sets would be another set's D and the standup
      would attest a different S).
    * the budget = the largest ``S x L`` a lora-native engine came up at with
      usable KV over that set's allowed pairs' manifests (the engine's resolved
      ``max_num_seqs`` x the window; ``ModelBudget`` certified,
      ``source=measured``); ``first_refused`` = the smallest ``S x L`` ABOVE
      that product at which one of the set's engines OOM'd, when any (the wall).
    The record also carries a MODEL-WIDE summary (max / min over the sets) that
    is for reading only and never sizes a standup, and the provenance: the probe
    job id, the record dir, the verdict uri, the tree sha, the host, and per
    launch the sha256 of its attestation.

    Never a guess: a folded pair whose certifying launch lacks a measured
    stride or an engine S REFUSES the fold (``status: refused``) — a certifying
    verdict cannot carry that, so it is a record inconsistency to surface, not
    paper over. No serving-certified pair: ``status: not_applicable``. No
    overlay directory configured: the record is still written beside the
    verdict (``status: record_only``)."""
    from sisteer.capeval.serving import geometry as g
    if not pairs_out:
        return {"status": "not_applicable", "reason": "no serving-certified pair: no engine came up and served adapters of any pair's "
                                                      "class at every window — nothing certifies a serving geometry for this model"}
    pair_manifests: Dict[Tuple[int, str], set] = {(int(p["rank"]), p["set"]): set(p["manifests"]) for p in plan["pairs"]}
    problems: List[str] = []
    per_set: Dict[str, Dict[str, Any]] = {}
    for p in pairs_out:
        sid = str(p["set"])
        st = per_set.setdefault(sid, {"lora_modules": tuple(p["lora_modules"]), "dims": {}, "budget_launches": {},
                                      "best": None, "oom_products": [], "allowed_pairs": [], "certified_by": [], "parity": []})
        if tuple(p["lora_modules"]) != st["lora_modules"]:
            problems.append(f"set {sid!r}: allowed pairs disagree on lora_modules ({st['lora_modules']} vs {tuple(p['lora_modules'])})")
        st["allowed_pairs"].append([int(p["rank"]), sid])
        st["parity"].append({"rank": int(p["rank"]), "set": sid,
                             **{k: (p.get(ADAPTER_PARITY_KEY) or {}).get(k)
                                for k in ("status", "basis", "mismatched_adapters", "certificate_version", SERVING_VENV_KEY)}})
        key = (int(p["rank"]), sid)
        for lid in p["certified_by"]:
            res = results.get(lid) or {}
            es = ((res.get("checks") or {}).get("executing_stride") or {})
            D = es.get("reported_D")
            if es.get("status") != "measured" or isinstance(D, bool) or not isinstance(D, int) or D < 1:
                problems.append(f"{lid}: no MEASURED executing stride on a certifying launch (checks.executing_stride="
                                f"{es!r}) — the kernel-ceiling D cannot be read, and is never guessed")
                continue
            st["dims"][lid] = int(D)
            if lid not in st["certified_by"]:
                st["certified_by"].append(lid)
        for rec in plan["launches"]:
            if not pair_manifests.get(key, set()).intersection(rec.get("serves") or [rec["manifest"]]):
                continue
            res = results.get(rec["id"]) or {}
            L = int(rec["L"])
            if res.get("status") == STATUS_PASSED:
                eng = (res.get("engine") or {}).get("max_num_seqs_engine")
                if isinstance(eng, bool) or not isinstance(eng, int) or eng < 1:
                    problems.append(f"{rec['id']}: passed but the engine reported no positive max_num_seqs ({eng!r})")
                    continue
                cls = launch_class_fields(res, rec["id"], problems)
                if cls is None:
                    continue
                # the attested class IS the planned one (the probe owner's ruling of 2026-10-11): the extension reasons about
                # coverage from the plan's (cap, slots) and the site record reads the fold's engine class — the two must agree
                planned_m = plan["manifests"].get(rec["manifest"]) or {}
                intended = {"max_lora_rank": int(rec["cap"]), "max_loras": int(planned_m.get("max_loras") or rec.get("max_loras") or 0)}
                if intended["max_loras"] >= 1 and cls != intended:
                    problems.append(f"{rec['id']}: the engine attested launch class {cls} but the plan intended {intended} (manifest "
                                    f"{rec['manifest']}: cap {rec['cap']}, {intended['max_loras']} slots) — the folded class would not be "
                                    "the one the plan covers; refusing rather than certifying another class")
                    continue
                st["budget_launches"][rec["id"]] = {"S": int(eng), "L": L, "batched_tokens": int(eng) * L, **cls}
                if st["best"] is None or int(eng) * L > st["best"][0]:
                    st["best"] = (int(eng) * L, rec["id"], int(eng), L, res)
            elif res.get("status") == STATUS_OOM:
                # an engine that OOM'd at S x L: a candidate wall (only one ABOVE the set's certified product counts —
                # a smaller product that OOM'd at a larger rank cap is a buffer failure, not the token wall)
                S_l = res.get("launch_S") or rec.get("admitted_S") or 0
                if isinstance(S_l, int) and not isinstance(S_l, bool) and S_l > 0:
                    st["oom_products"].append(S_l * L)
    for sid, st in per_set.items():
        if st["best"] is None:
            problems.append(f"set {sid!r}: no passed launch with an engine max_num_seqs over its allowed pairs' manifests")
        if not st["dims"]:
            problems.append(f"set {sid!r}: no measured executing stride over its allowed pairs' certifying launches")
    if problems:
        reason = "; ".join(dict.fromkeys(problems))
        log(f"geometry fold REFUSED: {reason}")
        return {"status": "refused", "reason": reason}
    model_key = plan["model_key"]
    util = float(plan.get("gpu_memory_utilization") or sm.DEFAULT_GPU_MEMORY_UTILIZATION)
    sets_rec: Dict[str, Dict[str, Any]] = {}
    fold_sets: Dict[str, Dict[str, Any]] = {}
    used: set = set()
    try:
        for sid, st in sorted(per_set.items()):
            total, lid, S, L, res = st["best"]
            D = max(st["dims"].values())
            above = [x for x in st["oom_products"] if x > total]
            kv = ((res.get("checks") or {}).get("memory") or {}).get("kv_capacity_tokens")
            basis = g.BudgetBasis(
                method=(f"{g.SITE_GEOMETRY_METHOD}: module set {sid!r} — the largest S x L a lora-native engine of this model "
                        f"came up at with usable KV over the set's certified pairs' launches ({lid}: engine max_num_seqs {S} x "
                        f"max_model_len {L}); every launch registered its production manifest's adapters of this class (the "
                        "LoRA buffers are inside the measurement)"),
                max_model_len=L, max_num_seqs=S, gpu_memory_utilization=util,
                kv_tokens_reported=(int(kv) if isinstance(kv, int) and not isinstance(kv, bool) else None),
                probe_job_id=str(probe_job_id), measured_on=utc_now()[:10],
                source=(f"scripts/sdfbase/serving_probe.py verdict "
                        f"{verdict_uri or (str(Path(out) / VERDICT_FILE) if out else '<in-memory>')}; set {sid}; launch {lid}"),
                first_refused=(min(above) if above else None),
                note=(f"set {sid!r} ({','.join(st['lora_modules'])}): widest executing module dim D={D} over launches "
                      f"{st['dims']}; passed launches folded: {st['budget_launches']}"))
            budget = g.ModelBudget(model_key=model_key, batched_token_budget=int(total), certified=True,
                                   source=g.MEASURED_BUDGET_SOURCE, basis=basis)
            sets_rec[sid] = {"lora_modules": list(st["lora_modules"]), "lora_max_module_dim": D, "budget": budget,
                             "allowed_pairs": st["allowed_pairs"], "certified_by": st["certified_by"],
                             "module_dim_launches": dict(st["dims"]), "budget_launches": dict(st["budget_launches"]),
                             # the set's second fact — a steered consumer requires 'verified' for the pair it serves
                             ADAPTER_PARITY_KEY: g.adapter_parity_block(st["parity"])}
            fold_sets[sid] = {"lora_max_module_dim": D, "batched_token_budget": int(total), "basis_launch": lid,
                              "allowed_pairs": st["allowed_pairs"], ADAPTER_PARITY_KEY: sets_rec[sid][ADAPTER_PARITY_KEY]["status"]}
            used |= set(st["dims"]) | set(st["budget_launches"])
        record_parity = g.adapter_parity_block([row for st_ in per_set.values() for row in st_["parity"]])
        provenance = {"probe_job_id": str(probe_job_id),
                      "probe_record": (str(out) if out is not None else "<in-memory verdict>"),
                      # the cluster the record lives on (the job env's label): the front door's staging verifies the site copy
                      # against the record OVER SSH to this cluster before it reaches any other
                      "probe_cluster": (probe_cluster or os.environ.get(PROBE_CLUSTER_ENV) or None),
                      "verdict_uri": verdict_uri, "tree_sha": _tree_sha(), "host": socket.gethostname(),
                      "attestation_sha256": {l: _launch_attestation(out, l, results.get(l) or {}) for l in sorted(used)},
                      # the pairs the fold read: SERVING-certified — their adapter parity is the record's `adapter_parity`
                      "allowed_pairs": [[int(p["rank"]), p["set"]] for p in pairs_out],
                      "reference_check_basis": record_parity["basis"],
                      # the batched-token budget every planned S was capped by — measured from this record (the
                      # floor / ceiling launches by id) and as the plan carried it — so the overlay says how S was sized
                      "batched_token_budget": {"measured": measured_budget(plan, results),
                                               "plan": dict(plan.get("batched_token_budget") or {})},
                      # the result generation this overlay was folded over: a consumer that can reach the record refuses an
                      # overlay older than the record's result set (geometry.check_overlay_generation)
                      RESULT_GENERATION_KEY: (result_generation(out) or {}).get("generation")}
        # B11: a fold certified THROUGH a revocation's receipt names that revocation's epoch — the exact fact the front door's
        # staging reads to clear the same-generation REVOKED marker that revocation wrote (never the clock; see REOPEN_EPOCH_KEY)
        epoch = fold_reopen_epoch(out)
        if epoch:
            provenance[REOPEN_EPOCH_KEY] = epoch
        record = g.site_geometry_record(model_key=model_key, hf_id=plan["served_hf_id"], revision=plan["revision"],
                                        sets=sets_rec, provenance=provenance, written_utc=utc_now(), adapter_parity=record_parity)
        # THE STANDUP RESOLUTION of every serving-certified pair's class over the record as folded: what a steered
        # standup of the pair is sized to at each plan window — read by replan_compute.py, which prices
        # the serving rows from it and never from one set's union alone
        standup_S = standup_resolution(record, plan, pairs_out)
    except (ValueError, TypeError) as e:
        log(f"geometry fold REFUSED: {type(e).__name__}: {e}")
        return {"status": "refused", "reason": f"{type(e).__name__}: {e}"[:2000]}
    fold: Dict[str, Any] = {"status": "record_only", "sets": fold_sets, "summary": dict(record["summary"]),
                            ADAPTER_PARITY_KEY: record_parity["status"], "record": None, "site_path": None,
                            STANDUP_S_KEY: standup_S}
    if out is not None:
        # the record beside the verdict is written by geometry.py's ONE serializer — the same bytes the overlay copy below,
        # the site's copy (03_hpo.sh) and every staged copy (lib/slurm.sh) carry, so a byte compare between them holds
        # (write_json_atomic wrote it without the overlay's trailing newline)
        fold["record"] = str(g.write_site_geometry_file(record, Path(out) / GEOMETRY_RECORD_FILE))
    site_dir = g.site_geometry_dir(geometry_dir)
    if site_dir is None:
        fold["reason"] = (f"no site geometry directory ({g.SITE_GEOMETRY_DIR_ENV} / {g.SITE_DIR_ENV} / {g.SHARED_ROOT_ENV} unset, "
                          "no --geometry-dir): the record was written beside the verdict only — the resolvers will NOT see it")
        log(f"geometry fold: {fold['reason']}")
        return fold
    try:
        fold["site_path"] = str(g.write_site_geometry(record, site_dir))
    except (g.SiteGeometryError, OSError) as e:
        log(f"geometry fold REFUSED at the site overlay: {e}")
        return {**fold, "status": "refused", "reason": f"site overlay write failed: {e}"}
    fold["status"] = "written"
    log("geometry fold: " + "; ".join(f"{sid}: D={v['lora_max_module_dim']} budget={v['batched_token_budget']:,} ({v['basis_launch']}, "
                                       f"adapter parity {v[ADAPTER_PARITY_KEY]})" for sid, v in fold_sets.items())
        + f" -> {fold['site_path']}; adapter_parity {record_parity['status']} (D145: the menu needs verified; an unsteered standup "
          "may size from this record whatever it says, a steered one refuses a mismatch / unverified pair by name)")
    return fold


# --------------------------------------------------------------------------
# memo
# --------------------------------------------------------------------------

def _reference_parity_rows(results: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """The parity TERMS of every judged adapter, read from the launch results' ``reference_check.per_adapter`` and
    de-duplicated (one adapter dir is judged once and referenced by every launch that served it): status, top-1 agreement,
    |Δlogprob|, the threshold and the terms that set it (tolerance, base floor, one-ulp sensitivity), whether its own
    perturbation flips its top-1, and the launches that carry it."""
    rows: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for lid in sorted(results):
        rc = results[lid].get("reference_check") or {}
        for name, v in sorted((rc.get("per_adapter") or {}).items()):
            if not isinstance(v, Mapping):
                continue
            terms = v.get("tolerance_terms") or {}
            sens = v.get("sensitivity") or {}
            key = (name, v.get("status"), v.get("logprob_abs_delta"), v.get("tolerance"), sens.get("max_shift"), v.get("gate"))
            v2 = v.get("v2") if isinstance(v.get("v2"), Mapping) else None
            # v2 is per LAUNCH (its floor is that launch's own base probes), so a row is one (adapter, v2 terms) — launches that
            # share identical terms fold, launches that differ stay apart
            key = key + (v.get("certificate_version"), (v2 or {}).get("max_abs_delta"), (v2 or {}).get("floor"), (v2 or {}).get("base_complete"))
            row = rows.setdefault(key, {"adapter": name, "status": v.get("status"), "top1_equal": v.get("top1_equal"),
                                        "certificate_version": v.get("certificate_version", 1),
                                        "v2_max_abs_delta": (v2 or {}).get("max_abs_delta"), "v2_tolerance": (v2 or {}).get("tolerance"),
                                        "v2_floor": (v2 or {}).get("floor"), "v2_base_complete": (v2 or {}).get("base_complete"),
                                        "v2_sensitivity_max": (v2 or {}).get("sensitivity_max"),
                                        "v2_argmax_prompt": (v2 or {}).get("argmax_prompt"),
                                        "v2_engine_moved_by_prompt": (v2 or {}).get("engine_moved_by_prompt"),
                                        "v2_hf_moved_by_prompt": (v2 or {}).get("hf_moved_by_prompt"),
                                        "logprob_abs_delta": v.get("logprob_abs_delta"), "tolerance": v.get("tolerance"),
                                        "tol": terms.get("tol"), "floor_term": terms.get("floor_term"), "sensitivity_term": terms.get("sensitivity_term"),
                                        "one_ulp_max_shift": sens.get("max_shift"), "one_ulp_flips": sens.get("flips"), "gate": v.get("gate"),
                                        "reason": (v.get("reason") or v.get("error")), "launches": []})
            row["launches"].append(lid)
    return list(rows.values())


def build_memo_inputs(plan: Mapping[str, Any], results: Mapping[str, Mapping[str, Any]],
                      pairs: Sequence[Mapping[str, Any]], excluded: Sequence[Mapping[str, Any]],
                      servers: Sequence[Mapping[str, Any]], ladder: Sequence[Mapping[str, Any]], dims: Any, *,
                      parity_results: Optional[Mapping[str, Mapping[str, Any]]] = None) -> Dict[str, Any]:
    """The memo's measured facts."""
    is_moe = bool(plan.get("is_moe"))
    classes: Dict[str, Any] = {}
    for rec in plan["launches"]:
        res = results.get(rec["id"])
        key = ",".join(rec["target_class"])
        c = classes.setdefault(key, {"launches": 0, "passed": 0, "oom": 0, "refused": 0, "failed": 0, "not_run": 0,
                                     "wraps_experts": sm.EXPERTS_TARGET in rec["target_class"]})
        c["launches"] += 1
        st = (res or {}).get("status", STATUS_NOT_RUN)
        c[{STATUS_PASSED: "passed", STATUS_OOM: "oom", STATUS_REFUSED: "refused", STATUS_FAILED: "failed",
           STATUS_ERROR: "refused", STATUS_SKIPPED: "not_run", STATUS_NOT_RUN: "not_run"}.get(st, "not_run")] += 1
    # the stacked-buffer law per CLASS: per-slot bytes per unit of the cap R (every family scales linearly in R; the
    # routed-expert family is priced at R = 1 here too, where the server's moe_lora_rank is what scales it)
    bytes_per_R_by_class = {",".join(cls): sum(sm._stacked_bytes_per_slot(dims, cls, 1).values())
                            for cls in sorted({tuple(rec["target_class"]) for rec in plan["launches"]})}
    # the routed-expert family alone, per slot per unit of moe_lora_rank, from the dims (a MoE plan only)
    expert_bytes_per_R = None
    if is_moe:
        E, d, I = int(dims.num_experts), int(dims.hidden), int(dims.moe_intermediate)
        expert_bytes_per_R = int(dims.moe_layers) * E * (d + 2 * I + I + d) * sm.BF16_BYTES
    measured = [{"launch": l["id"], "cap": l["cap"], "moe_lora_rank": l.get("moe_lora_rank"), "expected": l["expected_lora_bytes"],
                 "measured": l["measured_buffer_bytes"], "status": l["status"]} for l in ladder]
    carriers_passed = any((results.get(rec["id"]) or {}).get("status") == STATUS_PASSED
                          for rec in plan["launches"] if rec["topology"] in (sm.TOPOLOGY_S2, sm.TOPOLOGY_SINGLE))
    kv_per_tok = sm.kv_bytes_per_token(dims)
    return {
        "native_support_per_class": classes,
        "residual_slot_carriers": {"loaded_on_a_passed_launch": carriers_passed,
                                   "note": "rank-1 o_proj zero carriers (refusal direction + 3 ReLU steers) stand in for "
                                           "the residual-slot payload adapters; the op itself is not exercised here"},
        "buffer_law": {"bytes_per_R_per_slot_by_class": bytes_per_R_by_class,
                       "mib_per_R_per_slot_by_class": {k: v / 2**20 for k, v in bytes_per_R_by_class.items()},
                       "expert_bytes_per_R_per_slot": expert_bytes_per_R,
                       "expert_mib_per_R_per_slot": (expert_bytes_per_R / 2**20 if expert_bytes_per_R else None),
                       "law": ("per slot and wrapped module, [R, in] + [out, R] in bf16 (a packed wrapper one A per slice), "
                               "summed over the wrapped blocks, × max_loras — R = the server's --max-lora-rank cap "
                               "(standup_manifest.expected_lora_bytes)"
                               + ("; the routed-expert family E × R_e × (d + 2I + I + d) × 2 B per MoE block and slot at "
                                  "R_e = the server's moe_lora_rank — the adapters' largest routed rank — since D394 (a launch "
                                  "before it allocated R_e = the dense cap)" if is_moe else "")),
                       "measured_points": measured},
        "servers": list(servers),
        "kv": {"bytes_per_token": kv_per_tok, "bytes_per_seq_by_window": {str(L): sm.kv_bytes_per_seq(dims, L)
                                                                             for L in plan["windows"]},
               "recurrent_state_per_seq_bytes": plan["budget"]["recurrent_state_per_seq_bytes"]},
        "stride_ceilings": {s: sm.packed_stride_ceiling(dims, sm.target_class_for(dims, s)) for s in plan["sets"]},
        "tp_needed": {"target": False, "attacker_27b": False,
                      "reason": "the target's bf16 base weights (the plan's budget block) and the ~54 GB 27B attacker each fit "
                                "one H200; TP is a throughput choice, not a memory need (the --tp2 variant measures it)"},
        "allowed_pairs": [[p["rank"], p["set"]] for p in pairs],
        "excluded_pairs": [[e["rank"], e["set"], e["reason"]] for e in excluded],
        # every judged adapter's parity terms (the floor, its own one-ulp sensitivity, the threshold that decided it)
        "reference_parity": _reference_parity_rows(parity_results if parity_results is not None else results),
        "reference_parity_source": ("parity_record" if parity_results is not None else "this_record"),
        "hpo_cost": {"trials": 16, "epochs_per_trial": 3, "pool": "constv2-mix-10m",
                     "note": "GPU-hours per trial are the memprobe's (FD 23) step times × tokens; see the memo"},
    }


# --------------------------------------------------------------------------
# memo
# --------------------------------------------------------------------------

def _gib(b: Optional[float]) -> str:
    return "—" if b is None else f"{b / 2**30:.2f} GiB"


def render_memo(verdict: Mapping[str, Any], memprobe: Optional[Mapping[str, Any]] = None) -> str:
    mi = verdict["memo_inputs"]
    lines = [f"# Scaling memo inputs — serving probe ({verdict['model_key']})", "",
             f"Model `{verdict['model_id']}` @ `{verdict['revision']}`; probe job `{verdict['probe_job_id']}`; "
             f"written {verdict['written_utc']} on `{verdict.get('host')}`; "
             f"{sum(1 for l in verdict['launches'] if l['status'] == STATUS_PASSED)}/{len(verdict['launches'])} "
             f"launches passed ({sum(1 for l in verdict['launches'] if l['status'] == STATUS_NOT_RUN)} not run).", ""]
    if verdict.get("status", VERDICT_COMPLETE) == VERDICT_UNCERTIFIED:
        lines += [f"**UNCERTIFIED — NOT CERTIFYING.** No HF+PEFT parity evidence for the pairs {verdict.get('unreferenced_pairs')} "
                  f"that passed every vLLM-side check; no `{ALLOWED_PAIRS_FILE}` was written (D55: a certification check that "
                  "cannot fire certifies nothing — peft==0.20.0 is REQUIRED in the serving venv; never `run --no-reference`).", ""]
    elif verdict.get("status", VERDICT_COMPLETE) != VERDICT_COMPLETE:
        lines += [f"**PARTIAL — NOT CERTIFYING.** {len([o for o in verdict.get('outstanding') or [] if o.get('decision_relevant')])} "
                  f"decision-relevant launch(es) outstanding; undecided pairs {verdict.get('undecided_pairs')}; no "
                  f"`{ALLOWED_PAIRS_FILE}` was written. Every number below is over the launches that ran so far.", ""]
    ap = verdict.get(ADAPTER_PARITY_KEY) or {}
    if ap and ap.get("status") != PARITY_VERIFIED:
        lines += [f"**ADAPTER PARITY: {str(ap.get('status')).upper()} (D145).** Geometry certification and adapter parity are separate "
                  f"facts: the serving geometry below was MEASURED on engines that came up and served adapters of each folded pair's class "
                  f"({verdict.get('serving_certified_pairs')}), and it stands for an UNSTEERED standup (no adapter, or the zero adapter alone); "
                  "a STEERED standup (a trained adapter of a folded pair served) refuses this record by name, and the LoRA menu lists "
                  "verified pairs only — here: "
                  + "; ".join(f"({e['rank']}, {e['set']}): {e['status']}" + (f" [{', '.join(e['mismatched_adapters'])}]" if e.get("mismatched_adapters") else "")
                              for e in ap.get("per_pair") or []) + ".", ""]
    if verdict.get("replan_required"):
        b = verdict.get("batched_token_budget") or {}
        lines += [f"**RE-PLAN REQUIRED (D108).** {len(verdict['replan_required'])} launch(es) died at engine start on a prefill "
                  f"batch above the measured batched-token budget ({b.get('tokens')} tokens, {b.get('method')}, floor "
                  f"{b.get('floor_launch')}): {', '.join(verdict['replan_required'][:12])}"
                  f"{' …' if len(verdict['replan_required']) > 12 else ''}. They exclude nothing; `run --resume --replan` "
                  "re-derives their S and re-runs them.", ""]
    lines += ["## 1. Native support per module class", "",
             "| class | wraps experts | launches | passed | oom | refused | failed | not run |", "|---|---|---|---|---|---|---|---|"]
    for cls, c in sorted(mi["native_support_per_class"].items()):
        lines.append(f"| `{cls}` | {c.get('wraps_experts', False)} | {c['launches']} | {c['passed']} | {c['oom']} | {c['refused']} | "
                     f"{c['failed']} | {c['not_run']} |")
    lines += ["", f"Residual-slot carriers: loaded on a passed launch = {mi['residual_slot_carriers']['loaded_on_a_passed_launch']} "
              f"({mi['residual_slot_carriers']['note']}).", "",
              "## 2. The buffer law: bytes per slot × R × slots", "",
              f"{mi['buffer_law']['law']}; derived per-slot bytes per unit R by class: "
              + ", ".join(f"`{cls}` {b} B ({mi['buffer_law']['mib_per_R_per_slot_by_class'][cls]:.2f} MiB)"
                          for cls, b in sorted(mi["buffer_law"]["bytes_per_R_per_slot_by_class"].items()))
              + (f"; the routed-expert family alone {mi['buffer_law']['expert_bytes_per_R_per_slot']} B "
                 f"({mi['buffer_law']['expert_mib_per_R_per_slot']:.1f} MiB) per slot per unit of moe_lora_rank"
                 if mi["buffer_law"].get("expert_bytes_per_R_per_slot") else "") + ".", "",
              "| ladder launch | cap | expected LoRA bytes | measured | status |", "|---|---|---|---|---|"]
    for p in mi["buffer_law"]["measured_points"]:
        lines.append(f"| `{p['launch']}` | {p['cap']} | {_gib(p['expected'])} | {_gib(p['measured'])} | {p['status']} |")
    lines += ["", "## 3. Per-server manifests and admitted `max_num_seqs` per window", "",
              "Requested = the nominal production concurrency; planned = after the packed-stride ceiling and the KV "
              "budget; engine = the resolved `max_num_seqs` the worker reported (client request concurrency is not "
              "engine concurrency).", ""]
    for srv in mi["servers"]:
        lines.append(f"### `{srv['manifest']}` ({srv['topology']}, class `{','.join(srv['target_class'])}`, cap {srv['cap']}, "
                     f"{srv['max_loras']} slots)")
        lines.append("")
        lines.append("| L | requested | planned | engine | binding | status | KV tokens | buffer bytes |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for L, row in sorted(srv["admitted_by_window"].items(), key=lambda kv: int(kv[0])):
            bb = row.get("buffer_bytes") or {}
            lines.append(f"| {L} | {row['requested']} | {row['planned']} | {row.get('engine') or '—'} | {row['binding']} | "
                         f"{row['status']} | {row.get('kv_capacity_tokens') or '—'} | "
                         f"{_gib(bb.get('measured')) if bb.get('status') == 'measured' else bb.get('status', '—')} |")
        tp = [row.get("throughput") for row in srv["admitted_by_window"].values() if row.get("throughput")]
        if tp:
            lines.append("")
            lines.append("| L | concurrency | engine S | tok/s (server) | tok/s (client) | errors |")
            lines.append("|---|---|---|---|---|---|")
            for L, row in sorted(srv["admitted_by_window"].items(), key=lambda kv: int(kv[0])):
                for pt in row.get("throughput") or []:
                    lines.append(f"| {L} | {pt['requested_concurrency']} | {pt.get('engine_max_num_seqs')} | "
                                 f"{pt.get('server_tok_s') and round(pt['server_tok_s'], 1)} | "
                                 f"{pt.get('client_tok_s') and round(pt['client_tok_s'], 1)} | {pt.get('errors')} |")
        lines.append("")
    kv = mi["kv"]
    lines += ["## 4. KV and stride facts", "",
              f"KV per token {kv['bytes_per_token']} B; per sequence: "
              + ", ".join(f"L={L}: {_gib(b)}" for L, b in kv["bytes_per_seq_by_window"].items())
              + f"; recurrent state per sequence {_gib(kv['recurrent_state_per_seq_bytes'])}.", "",
              "| set | widest executing stride | D | token ceiling | S at 8192 | S at 49152 |", "|---|---|---|---|---|---|"]
    for s, c in mi["stride_ceilings"].items():
        lines.append(f"| {s} | `{c['widest']}` | {c['D']} | {c['ceiling']:,} | {c['ceiling'] // 8192} | {c['ceiling'] // 49152} |")
    lines += ["", "## 5. Tensor parallelism", "",
              f"Target TP needed: {mi['tp_needed']['target']}; 27B attacker TP needed: {mi['tp_needed']['attacker_27b']} — "
              f"{mi['tp_needed']['reason']}.", "",
              "## 6. Verdict: allowed and excluded (rank, set) pairs", "",
              f"Reference-check basis: `{verdict['reference_check_basis']}`; adapter parity (D145): `{ap.get('status', '—')}` over the "
              f"serving-certified pairs {verdict.get('serving_certified_pairs', '—')} (the menu below lists verified pairs only).", "",
              "| rank | set | topology | cap | trainable params | admitted S by window (roles) | served-class S by window | standup S by window | basis |",
              "|---|---|---|---|---|---|---|---|---|"]
    for p in verdict["pairs"]:
        sc = p.get("served_class") or {}
        st = sc.get("standup") or {}
        # the standup column: the all-set resolution's S per window, or the refusal a standup of the class would meet
        standup_cell = (json.dumps(st["S_by_window"]) if st.get("S_by_window") is not None
                        else f"refused: {st['refused']}" if st.get("refused") else "—")
        # ...and, for a class of two or more adapters, the windows the resolution sizes below two sequences: native_standup refuses
        # such a standup before launch (the multi-LoRA routing control needs two distinct selections in one engine step), and
        # replan_compute refuses to price it — the column says so beside the S
        if st.get("S_by_window") is not None and sc.get("max_loras") is not None and int(sc["max_loras"]) >= 2:
            under = [str(L) for L, S in st["S_by_window"].items() if S is not None and int(S) < 2]
            if under:
                standup_cell += (f" — a standup of {sc['max_loras']} adapters REFUSES S < 2 at L={', '.join(under)} (native_standup: the "
                                 f"multi-LoRA routing control needs two distinct selections in one engine step)")
        lines.append(f"| {p['rank']} | {p['set']} | {p['topology']} | {p['max_lora_rank_cap']} | {p['trainable_params']:,} | "
                     f"{json.dumps(p['admitted_max_num_seqs'])} | "
                     f"{json.dumps(sc['measured_S_by_window']) if sc.get('measured_S_by_window') is not None else '—'} | "
                     f"{standup_cell.replace('|', '/')} | "
                     f"{p['reference_check_basis']} |")
    lines += ["", "| rank | set | excluded because |", "|---|---|---|"]
    for e in verdict["excluded"]:
        lines.append(f"| {e['rank']} | {e['set']} | `{e['reason']}` |")
    if ap.get("per_pair"):
        lines += ["", "Serving-certified pairs and their adapter parity (D145 — geometry certified for every row; a steered standup needs "
                  "`verified`):", "", "| rank | set | adapter parity | basis | adapters that disagreed |", "|---|---|---|---|---|"]
        for e in ap["per_pair"]:
            lines.append(f"| {e['rank']} | {e['set']} | `{e['status']}` | {e.get('basis') or '—'} | {', '.join(e.get('mismatched_adapters') or []) or '—'} |")
    rp = mi.get("reference_parity") or []
    if rp:
        floors = sorted({round(r["floor_term"] / PARITY_FLOOR_K, 6) for r in rp if r.get("floor_term") is not None})
        lines += ["", f"Reference parity terms (D161/D163, rule (a)): pass = top-1 agreement AND abs dlogprob <= max(tolerance {REFERENCE_LOGPROB_TOL}, "
                  f"{PARITY_FLOOR_K:g} × the base-vs-base floor measured on this record"
                  + (f" ({', '.join(f'{f:.4f}' for f in floors)})" if floors else " (not measured)")
                  + f", {PARITY_SENSITIVITY_M:g} × the adapter's own one-ulp shift — {PARITY_SENSITIVITY_DRAWS} forwards with lora_B × "
                  f"(1 + 2^-8·N(0,1))); `parity_chaotic` (a one-ulp flip of its own greedy token) or `parity_weak` (the engine moved the top-5 "
                  f"< {PARITY_MOVED_MIN} nats with the greedy token unchanged) files the adapter `{REF_STATUS_UNDECIDABLE}` — a decided exclusion.", "",
                  "| adapter | status | top-1 equal | abs dlogprob | bound | floor term | sensitivity term | one-ulp max shift | one-ulp flips | gate | launches |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        fmt = lambda x: "—" if x is None else (f"{x:.4f}" if isinstance(x, float) else str(x))
        for r in rp:
            lines.append(f"| {r['adapter']} | `{r['status']}` | {fmt(r.get('top1_equal'))} | {fmt(r.get('logprob_abs_delta'))} | {fmt(r.get('tolerance'))} | "
                         f"{fmt(r.get('floor_term'))} | {fmt(r.get('sensitivity_term'))} | {fmt(r.get('one_ulp_max_shift'))} | {fmt(r.get('one_ulp_flips'))} | "
                         f"{r.get('gate') or '—'} | {len(r.get('launches') or [])} |")
    lines += ["", "## 7. Fit throughput and memory (FD 23 memprobe)", ""]
    if memprobe:
        lines.append("```json")
        # the FD 23 record's headline keys (memprobe_fit.py's top level: the verdict, the
        # identity the estimate rests on, and the per-rung table) — a partial record shows what it has
        lines.append(json.dumps({k: memprobe[k] for k in sorted(memprobe) if k in
                                 ("verdict", "model", "arm", "seq_len", "steps", "ladder", "gradient_checkpointing",
                                  "dtype", "gpu_name", "rungs", "micro_batch", "s_per_step", "tokens_per_s", "fits")}
                                or memprobe, indent=2, default=str)[:4000])
        lines.append("```")
    else:
        lines.append("_memprobe.json not given: fit throughput placeholders — the memo reads them from the FD 23 report._")
    hc = mi["hpo_cost"]
    lines += ["", "## 8. HPO cost", "",
              f"{hc['trials']} trials × {hc['epochs_per_trial']} epochs on `{hc['pool']}`; {hc['note']}.", ""]
    fold = verdict.get("geometry_fold") or {}
    pr = verdict.get(PARITY_RECORD_KEY)
    if pr or verdict.get("parity_pass"):
        lines += ["## 8b. Adapter parity basis — the parity pass (D163)", ""]
        if verdict.get("parity_pass"):
            lines += ["This record IS a parity pass: its launches ran at the ladder window with adapters at per-set benign synthetic "
                      "scales; its pairs table is a parity summary, never the LoRA search's menu, and it folds no geometry.", ""]
        if pr:
            lines += [f"Assertion 4 (HF+PEFT parity) was read from the parity-pass record `{pr.get('dir')}` (result generation "
                      f"`{str(pr.get(RESULT_GENERATION_KEY))[:12]}`, {pr.get('n_results')} results, windows {pr.get('windows')}, "
                      f"{len(pr.get('judged_launches') or [])} judged launch(es)); this record's servability and geometry are its own.", "",
                      "| module set | parity synthetic scale |", "|---|---|"]
            lines += [f"| `{s}` | {v} |" for s, v in sorted((pr.get("synthetic_scale_by_set") or {}).items())]
            lines += ["", f"Scale provenance: {pr.get('scale_provenance')}", ""]
    lines += ["## 9. Serving geometry fold (the site overlay the padded launchers size from)", ""]
    if fold.get("status") in ("written", "record_only"):
        lines += ["Certified PER MODULE SET (D69: a native standup's ceiling is the SELECTED set's D, never a model-wide maximum). "
                  f"Record-level adapter parity `{fold.get(ADAPTER_PARITY_KEY, '—')}` (D145: geometry certification and adapter parity are "
                  "separate facts — an unsteered standup sizes from every row; a steered standup requires `verified` for the pair it serves):", "",
                  "| module set | widest executing dim D | certified batched-token budget | basis launch | serving-certified pairs | adapter parity |",
                  "|---|---|---|---|---|---|"]
        for sid, v in sorted((fold.get("sets") or {}).items()):
            lines.append(f"| `{sid}` | {v.get('lora_max_module_dim')} | {int(v.get('batched_token_budget') or 0):,} | "
                         f"`{v.get('basis_launch')}` | {v.get('allowed_pairs')} | `{v.get(ADAPTER_PARITY_KEY, '—')}` |")
        summ = fold.get("summary") or {}
        lines += ["", f"Summary (reading only): D max {summ.get('lora_max_module_dim_max')}, budget max "
                      f"{summ.get('batched_token_budget_max')} / min {summ.get('batched_token_budget_min')}. Record: "
                      f"`{fold.get('record')}`; site overlay: `{fold.get('site_path') or 'NOT written — ' + str(fold.get('reason'))}`. "
                      "`geometry.budget_for(model_key, module_set)` / `rows_for` / `kernel_ceiling_adjustment(…, module_set=)` "
                      "resolve the SELECTED set's record before the built-in tables.", ""]
    else:
        lines += [f"Not folded: {fold.get('status')} — {fold.get('reason')}.", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _common(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--model-key", required=True, choices=list(MODEL_DIMS_KEYS))
    ap.add_argument("--model-id", help="the SERVED HF id (default: from probe_plan.json)")
    ap.add_argument("--revision", help="40-hex snapshot sha (a ref name is resolved from the HF cache)")
    ap.add_argument("--out", type=Path, help="probe root (default $SHARED/exp/multilora_cert/<key>_probe_<jobid>)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="write probe_plan.json (or --dry-run print it)")
    _common(p)
    p.add_argument("--sets", default=",".join(DEFAULT_SETS))
    p.add_argument("--ranks", default=",".join(str(r) for r in DEFAULT_RANKS))
    p.add_argument("--windows", default=",".join(str(w) for w in sm.PRODUCTION_WINDOWS))
    p.add_argument("--ladder-window", type=int, default=4096)
    p.add_argument("--ladder-caps", default=",".join(str(c) for c in sm.RANK_CAP_LADDER))
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--gpu-mem-util", type=float, default=sm.DEFAULT_GPU_MEMORY_UTILIZATION)
    p.add_argument("--base-weights-gb", type=float, default=sm.DEFAULT_BASE_WEIGHTS_BYTES / 1e9)
    p.add_argument("--recurrent-state-mib", type=float, default=None,
                   help="a hybrid's per-SEQUENCE recurrent state (constant in L) the KV admission prices beside KV x L; default: "
                        "derived from the checkpoint's cached HF config (standup_manifest.gated_deltanet_state_per_seq_bytes), "
                        "0 when the config is not readable or carries no recurrent block — never a typed per-model constant")
    p.add_argument("--synthetic-scale", type=float, default=DEFAULT_SYNTHETIC_SCALE)
    p.add_argument("--parity-scales", default=None,
                   help=f"D163: per-set synthetic scales `set=scale,...` or `{PARITY_SCALES_DEFAULT_WORD}` (= {PARITY_SCALES_BY_SET}); "
                        "every genuine adapter of a set is exported at ITS scale (plan key synthetic_scale_by_set)")
    p.add_argument("--parity-pass", action="store_true",
                   help="D163: write a PARITY-PASS plan — the natural launches at the ladder window only (--windows / --ladder-caps "
                        "are forced to the ladder window and no ladder), per-set benign scales (--parity-scales, default the table), "
                        "no geometry fold; `verdict --parity-record <this dir>` on the MAIN record reads assertion 4 from it")
    p.add_argument("--dry-run", action="store_true", help="print the launches; write nothing")
    p = sub.add_parser("adapters", help="export every manifest's adapter dirs")
    _common(p)
    p.add_argument("--only", help="comma-separated manifest names")
    p = sub.add_parser("run", help="stand up every planned launch; exit 0 when every planned launch is done, "
                                   f"{EXIT_INCOMPLETE} when the time budget / --max-launches left some outstanding")
    _common(p)
    p.add_argument("--resume", action="store_true",
                   help="continue an existing record's outstanding launches (the plan and the finished launch results "
                        "under --out are authoritative; --plan-args must agree with the plan); without it a record "
                        f"whose {PROGRESS_FILE} is incomplete is refused rather than re-planned over")
    p.add_argument("--replan", action="store_true",
                   help="with --resume (D108): measure the batched-token budget from the record (the largest S x L that "
                        "came up), re-derive S = min(stride, KV, requested, budget // L) for every outstanding launch AND "
                        "every launch whose OOM was a start-time prefill above the budget, move those launches' results "
                        "aside (launches/superseded/, never deleted) and re-run them; ids are stable. Idempotent.")
    p.add_argument("--reopen-token", default=None,
                   help="REQUIRED to reopen a record whose checkpoint is COMPLETE (the review's stage-E re-checks 2/3): the NONCE of the "
                        "revocation receipt the front door (reproduce/03_hpo.sh --phase probe --reopen) wrote into this record after "
                        "every cluster acknowledged its REVOKED marker; the receipt is consumed on use. A carrier without it refuses to "
                        "re-run anything on a complete record (a self-resubmitted successor only ever continues an incomplete one)")
    p.add_argument("--only", help="comma-separated launch ids")
    p.add_argument("--certify-v2", default=None,
                   help="PARITY CERTIFICATE v2 (D478) for these rank:set pairs (comma-separated), or `all` (D502): plan the current servers at the ladder "
                        "window when absent and re-run every launch of theirs whose result carries no probes_v2 (with --resume --replan; "
                        "a COMPLETE record needs the front door's reopen receipt)")
    p.add_argument("--max-launches", type=int)
    p.add_argument("--time-budget-s", type=float)
    p.add_argument("--gpu-mem-util", type=float, default=sm.DEFAULT_GPU_MEMORY_UTILIZATION)
    p.add_argument("--ready-timeout-s", type=float, default=1800.0)
    p.add_argument("--no-throughput", action="store_true")
    p.add_argument("--parallel", default=PARALLEL_AUTO,
                   help=f"concurrent standups (D91): an integer, or '{PARALLEL_AUTO}' (default) = the job's "
                        f"{VISIBLE_DEVICES_ENV} count ÷ the plan's tp under Slurm, 1 outside it. Each standup is pinned "
                        f"to a STRICT SUBSET of Slurm's devices in its own environment; a value the allocation cannot "
                        "hold is refused (never widened)")
    p.add_argument("--no-reference", action="store_true",
                   help="DEBUGGING ONLY: skip the HF+PEFT phase — the verdict then REFUSES certification "
                        f"(exit {EXIT_UNCERTIFIED}, no allowed_pairs.json): a pair without parity evidence is never allowed (D55)")
    p.add_argument("--reference-only", action="store_true",
                   help="with --resume, on a record whose launches are ALL done: run ONLY the HF+PEFT reference phase again "
                        "(1 GPU, no engine) over EVERY adapter — the prior reference_check.json is archived, never kept — and "
                        "rewrite each result's reference_check + the checkpoint stamp; the carrier's `verdict` then reads the new "
                        "evidence. The results change, so every standing certification artifact is archived first and a COMPLETE "
                        "record needs the front door's reopen receipt (--reopen-token). D146: the reference-side fix's re-run")
    p.add_argument("--plan-args", default="", help="extra `plan` args when the plan must be written first")
    p.add_argument("--dry-run", action="store_true",
                   help="with --resume [--replan] [--certify-v2 ...]: PRINT the work the resume would run on this record — the "
                        "outstanding launches, the replan_required ones, the plan extension's additions with their reasons (the "
                        "current servers where a pair is uncertified or starved; the pair's union manifest where no planned "
                        "launch dominates the served launch class) and the v2 work — and exit 0. Reads the record only: no "
                        "lock, no receipt consumed, no plan written, no archive, no engine; the operator's check before a "
                        "reopen (reproduce/03_hpo.sh --phase probe --reopen)")
    p = sub.add_parser("verdict", help="write probe_verdict.json + allowed_pairs.json (+ the geometry fold when certifying)")
    _common(p)
    p.add_argument("--probe-job-id", default=os.environ.get("SLURM_JOB_ID", "local"))
    p.add_argument("--verdict-uri")
    p.add_argument("--geometry-dir", type=Path,
                   help="the SITE geometry overlay directory the fold writes <model_key>.json into (default: "
                        "geometry.site_geometry_dir() — $SISTEER_GEOMETRY_DIR, else $SISTEER_SITE_DIR/geometry, else "
                        "$SISTEER_SHARED_ROOT/exp/reproduce/geometry; with none the record is written beside the verdict only)")
    p.add_argument("--probe-cluster", default=None,
                   help=f"the cluster label this record lives on, recorded as the overlay's provenance.probe_cluster (default: "
                        f"${PROBE_CLUSTER_ENV} from the job env); the front door's staging verifies the site copy against the record "
                        "on that cluster — an overlay without it is never staged")
    p.add_argument("--parity-record", type=Path, default=None,
                   help="D163: a PARITY-PASS record (plan --parity-pass) assertion 4 is read from — per certifying manifest, its natural "
                        "launch's reference_check at the per-set benign scale; this record's servability and geometry stand as they are")
    p = sub.add_parser("memo", help="render memo_scaling_inputs.md")
    _common(p)
    p.add_argument("--memprobe", type=Path, help="the FD 23 memprobe.json")
    p = sub.add_parser("reverdict", help="CPU-only: archive the record's derived artifacts (menu, verdict, geometry record + its "
                                         "site overlay, memo, RESULT -> launches/superseded/<utc>/) and rewrite verdict + menu + "
                                         "geometry + memo from the record's results under THIS code, stamped with the result generation")
    p.add_argument("--record", type=Path, required=True, help="the record dir (probe_plan.json + launches/)")
    p.add_argument("--probe-job-id", help="default: the existing verdict's probe_job_id, else $SLURM_JOB_ID, else 'reverdict'")
    p.add_argument("--geometry-dir", type=Path, help="the site geometry overlay dir (default: geometry.site_geometry_dir())")
    p.add_argument("--probe-cluster", default=None,
                   help=f"the cluster label this record lives on (default: ${PROBE_CLUSTER_ENV}); recorded as provenance.probe_cluster")
    p.add_argument("--reopen-token", default=None,
                   help="REQUIRED (re-checks 2/3): the nonce of the revocation receipt the front door wrote into this record after it "
                        "revoked every staged overlay — a reverdict rewrites the certification the clusters' copies stand on")
    p.add_argument("--memprobe", type=Path, help="the FD 23 memprobe.json for the memo")
    p.add_argument("--parity-record", type=Path, default=None,
                   help="D163: the PARITY-PASS record assertion 4 is read from (see `verdict --parity-record`)")
    p.add_argument("--resume-interrupted", action="store_true",
                   help="the guarded recovery (the review on 7f54fd145): adopt the record's one IN-PROGRESS receipt — a reverdict that consumed "
                        "it and died (or whose verdict raised) after archiving the derived artifacts — and re-run the certifying rewrite "
                        "under that receipt's epoch; takes no --reopen-token; never a fresh revocation")
    return ap


def _resolve_out(args: argparse.Namespace) -> Path:
    return Path(args.out) if args.out else default_out(args.model_key)


def recurrent_state_for(model_id: str, revision: Optional[str], *, declared_mib: Optional[float] = None
                        ) -> Tuple[int, str]:
    """``(bytes, source)`` of the per-SEQUENCE recurrent state the plan's KV admission prices: ``--recurrent-state-mib``
    when declared (``source`` ``declared``), else the checkpoint's cached HF config through
    ``standup_manifest.gated_deltanet_state_per_seq_bytes`` (``config``; 0 for a dense transformer), else 0 with
    ``source`` naming why (``config_unreadable`` — a synthetic id or an uncached checkpoint). Never a typed per-model
    constant."""
    if declared_mib is not None:
        return int(round(float(declared_mib) * 2**20)), "declared"
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_id, revision=revision, local_files_only=True)
    except Exception as e:  # noqa: BLE001 — a synthetic id / an uncached checkpoint: the plan says so and prices 0
        return 0, f"config_unreadable:{type(e).__name__}"
    return int(sm.gated_deltanet_state_per_seq_bytes(cfg)), "config"


def cmd_plan(args: argparse.Namespace) -> int:
    dims = dims_for(args.model_key)
    model_id = args.model_id
    if not model_id:
        raise SystemExit(f"{TAG} plan needs --model-id")
    revision = args.revision or "main"
    if not args.dry_run:
        revision = resolve_revision(model_id, revision)
    parity_pass = bool(getattr(args, "parity_pass", False))
    scales_text = getattr(args, "parity_scales", None)
    try:
        scales = parse_parity_scales(scales_text) if (parity_pass or scales_text is not None) else None
    except ValueError as e:
        raise SystemExit(f"{TAG} {e}") from None
    windows = [int(w) for w in args.windows.split(",") if w]
    caps_text = str(args.ladder_caps).strip().lower()
    ladder_caps = [] if caps_text in ("", "none") else [int(c) for c in args.ladder_caps.split(",") if c]
    if parity_pass:
        # the parity pass: the ladder window alone, no rank-cap ladder — one natural standup per (set, rank) manifest
        windows, ladder_caps = [int(args.ladder_window)], []
    recurrent_bytes, recurrent_source = recurrent_state_for(model_id, revision,
                                                            declared_mib=getattr(args, "recurrent_state_mib", None))
    log(f"recurrent state per sequence: {recurrent_bytes} B ({recurrent_source})")
    try:
        plan = build_plan(dims, model_key=args.model_key, model_id=model_id, revision=revision,
                          sets=[s for s in args.sets.split(",") if s], ranks=[int(r) for r in args.ranks.split(",") if r],
                          windows=windows, ladder_window=args.ladder_window, ladder_caps=ladder_caps, tp=args.tp,
                          gpu_memory_utilization=args.gpu_mem_util, base_weights_bytes=args.base_weights_gb * 1e9,
                          recurrent_state_per_seq_bytes=recurrent_bytes,
                          synthetic_scale=args.synthetic_scale, parity_scales=scales, parity_pass=parity_pass)
    except ValueError as e:
        raise SystemExit(f"{TAG} {e}") from None
    for line in plan_summary_lines(plan):
        print(line)
    if args.dry_run:
        print(f"{TAG} dry run: nothing written")
        return 0
    out = _resolve_out(args)
    write_json_atomic(out / PLAN_FILE, plan)
    log(f"plan written: {out / PLAN_FILE} ({plan['n_launches']} launches)")
    return 0


def cmd_adapters(args: argparse.Namespace) -> int:
    out = _resolve_out(args)
    plan = load_plan(out)
    exporter = ProductionExporter(plan["model_id"], plan["revision"], synthetic_scale=plan.get("synthetic_scale", DEFAULT_SYNTHETIC_SCALE))
    only = [s for s in (args.only or "").split(",") if s] or None
    layout = build_adapter_dirs(plan, out, exporter, only_manifests=only)
    n = sum(len(m["slots"]) for m in layout["manifests"].values())
    log(f"adapters: {n} slot dirs across {len(layout['manifests'])} manifests; {len(layout['refused'])} refused")
    return 0 if not layout["refused"] else 2


def _plan_args_agree(plan: Mapping[str, Any], plan_args: str) -> Optional[str]:
    """``None`` when ``--plan-args`` (as given to a resumed run) name the same
    sets / ranks / tp as the existing plan, else what differs."""
    toks = [a for a in (plan_args or "").split() if a]
    want: Dict[str, Any] = {}
    for i, t in enumerate(toks):
        if t in ("--sets", "--ranks", "--tp") and i + 1 < len(toks):
            want[t] = toks[i + 1]
    diffs = []
    if "--sets" in want and [s for s in want["--sets"].split(",") if s] != list(plan["sets"]):
        diffs.append(f"sets {want['--sets']} != plan {','.join(plan['sets'])}")
    if "--ranks" in want and sorted(int(r) for r in want["--ranks"].split(",") if r) != list(plan["ranks"]):
        diffs.append(f"ranks {want['--ranks']} != plan {','.join(str(r) for r in plan['ranks'])}")
    if "--tp" in want and int(want["--tp"]) != int(plan["tp"]):
        diffs.append(f"tp {want['--tp']} != plan {plan['tp']}")
    return "; ".join(diffs) or None


def cmd_run_reference_only(args: argparse.Namespace, out: Path, prior: Optional[Mapping[str, Any]],
                           deps: Optional[Deps] = None) -> int:
    """``run --resume --reference-only``: the HF+PEFT reference phase ALONE, again, over every adapter of a record whose
    launches are all done. No engine starts. Refuses without --resume, with --replan / --no-reference, with a
    launch still to run (resume those first), and — on a COMPLETE record — without the front door's reopen receipt
    (the results change, so the standing menu / verdict / geometry / memo are archived first, exactly as a resume with
    launches to run does). Exit 0 when the phase ran (the verdict decides), EXIT_UNCERTIFIED when the phase itself
    could not run."""
    if not getattr(args, "resume", False):
        raise SystemExit(f"{TAG} --reference-only re-judges an EXISTING record: pass it with --resume")
    if getattr(args, "replan", False) or getattr(args, "no_reference", False):
        raise SystemExit(f"{TAG} --reference-only takes neither --replan nor --no-reference (it runs the reference phase and nothing else)")
    if not (out / PLAN_FILE).is_file():
        raise SystemExit(f"{TAG} --reference-only: no {PLAN_FILE} under {out} — nothing to re-judge")
    plan = load_plan(out)
    mismatch = _plan_args_agree(plan, getattr(args, "plan_args", "") or "")
    if mismatch:
        raise SystemExit(f"{TAG} --reference-only: --plan-args disagree with the existing plan ({mismatch})")
    results = load_results(out)
    work = [r["id"] for r in outstanding_launches(plan, results)] + replan_required_launches(plan, results)
    if work:
        raise SystemExit(f"{TAG} --reference-only: {len(work)} launch(es) still to run ({work[:6]}) — resume them first "
                         "(`run --resume --replan`); the reference phase re-runs over a finished result set only")
    judged = [lid for lid, r in results.items() if r.get("status") in (STATUS_PASSED, STATUS_FAILED) and r.get("reference_pending")]
    if not judged:
        raise SystemExit(f"{TAG} --reference-only: no passed/failed launch carries adapters to judge under {out}")
    acquire_reopen_lock(out, what="the reference-only re-run")
    try:
        return _cmd_run_reference_only_locked(args, out, prior, deps, plan, results, judged)
    finally:
        release_reopen_lock(out)


def _cmd_run_reference_only_locked(args: argparse.Namespace, out: Path, prior: Optional[Mapping[str, Any]], deps: Optional[Deps],
                                   plan: Mapping[str, Any], results: Dict[str, Any], judged: List[str]) -> int:
    owner: Optional[str] = None
    if prior is not None and prior.get("complete") is True:
        owner = consume_reopen_receipt(out, getattr(args, "reopen_token", None),
                                       what=f"the reference-only re-run (re-judging the adapters of {len(judged)} launch(es))",
                                       action="run --resume --reference-only")[RECEIPT_CONSUME_KEY]["owner"]
    # the same guard the resume with launches to run makes (_cmd_run): the standing verdict's parity stamp is written into
    # the plan BEFORE that verdict is archived — a record from before the plan carried the key would otherwise forget its pass
    # here, and the carrier's bare `verdict` after this phase would demote the menu to this record's own references in silence
    _remember_parity_record(out, plan)
    plan = load_plan(out)
    archive_derived_artifacts(out, why=f"run --resume --reference-only: the HF+PEFT reference phase is re-run over the adapters of "
                                       f"{len(judged)} launch(es) — every derived artifact predates this attempt",
                              model_key=str(plan["model_key"]))
    # the checkpoint says IN PROGRESS before the HF phase — complete=false (the launch list alone would say
    # complete), the attempt recorded, the hold named — so a reader of a record whose verdict READS this one
    # (hpo_probe_status.verify_parity_record) refuses while the evidence is being redone, and keeps refusing if this worker dies
    # before its completing write; only the write AFTER the phase and every per-launch result write clears it
    attempt = {"started_utc": utc_now(), "job_id": os.environ.get("SLURM_JOB_ID"), "host": socket.gethostname(),
               "launches_to_run": 0, "reference_only": True, REFERENCE_RERUN_STARTED_KEY: utc_now()}
    prog0 = write_progress(plan, results, out, attempt=attempt, hold=HOLD_REFERENCE_ONLY)
    log(f"reference-only: checkpoint marked in progress ({HOLD_REFERENCE_ONLY}, complete=false) before any GPU work (result generation "
        f"{str(prog0.get(RESULT_GENERATION_KEY))[:12]}); re-judging the adapters of {len(judged)} launch(es) on "
        f"{VISIBLE_DEVICES_ENV}={os.environ.get(VISIBLE_DEVICES_ENV, '<unset>')}")
    try:
        ref = reference_phase(plan, out, deps or Deps.production(), redo=True)
    except BaseException as e:  # noqa: BLE001 — a dying worker leaves the record marked in progress, never complete
        write_progress(plan, load_results(out), out, attempt={**attempt, "interrupted_utc": utc_now(), "interrupted_by": f"{type(e).__name__}: {e}"[:300]},
                       hold=HOLD_REFERENCE_ONLY)
        log(f"reference-only: INTERRUPTED ({type(e).__name__}) — the checkpoint stays in progress (complete=false); `run --resume "
            "--reference-only` resumes it (no new receipt: the record is not complete)")
        raise
    log(f"reference phase: {ref.get('status')}" + (f" — {ref.get('reason')}" if ref.get("reason") else "")
        + (f"; counts {ref.get('counts')}" if ref.get("counts") else ""))
    # the completing write: the phase returned AND every per-launch result is rewritten (reference_phase writes them before it returns)
    write_progress(plan, load_results(out), out)
    # the results ARE the rewrite (the carrier's verdict needs no receipt): finish the receipt this attempt consumed, or — a resumed
    # attempt of an interrupted re-run — CLAIM the one an earlier attempt left in progress, then finish it
    if owner is None and in_progress_receipts(out):
        owner = claim_in_progress_receipt(out, what="the reference-only re-run (a resumed attempt)")[1]
    finish_reopen_receipt(out, what="the reference-only re-run", owner=owner)
    return 0 if ref.get("status") == "done" else EXIT_UNCERTIFIED


def _remember_parity_record(out: Path, plan: Mapping[str, Any], parity_dir: Optional[str] = None) -> Optional[str]:
    """``plan.parity_record_dir`` — the parity-pass record this record's certification reads assertion 4 from (see
    cmd_verdict). Two writers. An EXPLICIT binding (``parity_dir``: `reverdict --parity-record` / `verdict --parity-record`,
    called once the pass has LOADED for this record) is remembered and REPLACES a different dir the plan carries: the
    operator's replacement of pass A by B is durable, so the next bare `verdict` (the carrier's after a reference-only
    rerun, a reopen's) reads B — before this the write returned at the first remembered key, the verdict was stamped B
    while the plan kept A, and the record undid the replacement itself. The IMPLICIT memory (no ``parity_dir``: a reopen
    about to archive the standing verdict reads its ``parity_record.dir`` stamp) writes once — a dir the plan already
    carries stands. Returns the dir the plan remembers afterwards, or None."""
    out = Path(out)
    have = str(plan["parity_record_dir"]) if plan.get("parity_record_dir") else None
    d = None if parity_dir is None else str(parity_dir)
    if d is None:
        if have:
            return have
        if (out / VERDICT_FILE).is_file():
            try:
                stamp = (json.loads((out / VERDICT_FILE).read_text(encoding="utf-8")) or {}).get(PARITY_RECORD_KEY) or {}
                d = stamp.get("dir")
            except ValueError:
                d = None
        if not d:
            return None
        d = str(d)
    elif have == d:
        return have
    new_plan = dict(plan)
    new_plan["parity_record_dir"] = d
    write_json_atomic(out / PLAN_FILE, new_plan)
    if have:
        log(f"the record's remembered parity-pass record is REPLACED by the explicit binding: plan.parity_record_dir {have} -> {d} "
            f"(a bare `verdict` reads {d} from here on)")
    else:
        log(f"the record remembers its parity-pass record: plan.parity_record_dir = {d}")
    return d


def print_resume_dry_run(out: Path, plan: Mapping[str, Any], results: Mapping[str, Mapping[str, Any]], prior: Optional[Mapping[str, Any]],
                         work: Sequence[str], extension: Tuple[Mapping[str, Any], Mapping[str, Any]], v2: Optional[Mapping[str, Any]], *,
                         replan: bool) -> int:
    """``run --resume [--replan] --dry-run``: what the resume WOULD run on ``out``, printed and nothing else — the outstanding launches,
    the replan_required ones, the plan extension's additions (``extension`` = :func:`extend_plan_for_uncertified_pairs`'s ``(new plan,
    change record)``; each added launch with its reason, class, window and planned S) and the v2 work; whether a COMPLETE record would
    need the front door's reopen receipt. Returns 0; writes, locks, consumes and launches nothing."""
    ext_plan, ext = extension
    by_id = {r["id"]: r for r in ext_plan["launches"]}
    outstanding = [r["id"] for r in outstanding_launches(plan, results)]
    replan_req = replan_required_launches(plan, results) if replan else []
    complete = bool(prior and prior.get("complete") is True)
    print(f"{TAG} DRY RUN of run --resume{' --replan' if replan else ''} on {out}: plan {plan['n_launches']} launch(es), "
          f"{sum(1 for r in plan['launches'] if (results.get(r['id']) or {}).get('status') in TERMINAL_STATUSES)} done, "
          f"checkpoint {'COMPLETE' if complete else 'incomplete'}")
    print(f"  outstanding ({len(outstanding)}): {outstanding or '-'}")
    print(f"  replan_required ({len(replan_req)}): {replan_req or '-'}")
    added = list(ext.get("added") or [])
    print(f"  plan extension ({len(added)} launch(es), {len(ext.get('manifests') or [])} new manifest(s), pairs {ext.get('pairs') or []}):")
    for lid in added:
        rec = by_id[lid]
        print(f"    + {lid:<34} {ext['why'].get(lid, '?'):<28} class=(cap {rec['cap']}, {rec['max_loras']} slots) topology={rec['topology']} "
              f"L={rec['L']} S {rec['requested_S']}->{rec['admitted_S']} ({rec['admission']['binding']}) manifest={rec['manifest']}")
    if not added:
        print("    (nothing: every pair is certified, not starved, and every window's served class is dominated by a planned launch)")
    if v2 is not None:
        print(f"  certify-v2: {len(v2.get('added') or [])} planned, {len(v2.get('supersede') or [])} superseded to re-run: "
              f"{sorted(set(v2.get('added') or []) | set(v2.get('supersede') or []))}")
    print(f"  work: {len(work)} launch(es) would run: {list(work) or '-'}")
    if complete and work:
        print(f"  the record is COMPLETE: this resume is a REOPEN and needs the front door's receipt (reproduce/03_hpo.sh --phase probe "
              f"--reopen mints it after revoking every staged overlay); every derived artifact would be archived first")
    elif complete:
        print("  the record is COMPLETE and nothing would run: a reopen on this tree would mint a receipt for no work — do not reopen")
    print(f"{TAG} dry run: nothing written, no lock taken, no receipt consumed")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    out = _resolve_out(args)
    try:
        return _cmd_run(args, out)
    finally:
        release_reopen_lock(out)          # a refusal / an exception / EXIT_INCOMPLETE / success: the lock never outlives the command


def _cmd_run(args: argparse.Namespace, out: Path) -> int:
    prior = load_progress(out)
    run_owner: Optional[str] = None
    try:
        certify_pairs = parse_pairs_arg(getattr(args, "certify_v2", None))
    except ValueError as e:
        raise SystemExit(f"{TAG} {e}") from None
    if certify_pairs and not (getattr(args, "resume", False) and getattr(args, "replan", False)):
        raise SystemExit(f"{TAG} --certify-v2 re-runs launches of an EXISTING record: pass it with --resume --replan")
    if getattr(args, "replan", False) and not getattr(args, "resume", False):
        raise SystemExit(f"{TAG} --replan re-derives an EXISTING record's plan: pass it with --resume")
    if getattr(args, "dry_run", False) and (not getattr(args, "resume", False) or getattr(args, "reference_only", False)):
        raise SystemExit(f"{TAG} run --dry-run previews the work of a RESUME: pass it with --resume [--replan] (not --reference-only)")
    if getattr(args, "reference_only", False):
        return cmd_run_reference_only(args, out, prior)
    if getattr(args, "resume", False):
        # a real resume: the record must exist; its plan is the authority
        if not (out / PLAN_FILE).is_file():
            raise SystemExit(f"{TAG} --resume: no {PLAN_FILE} under {out} — nothing to resume")
        plan = load_plan(out)
        mismatch = _plan_args_agree(plan, args.plan_args)
        if mismatch:
            raise SystemExit(f"{TAG} --resume: --plan-args disagree with the existing plan ({mismatch}); a different "
                             "probe needs a different --out")
        log(f"resume: {out} — {(prior or {}).get('n_done', 0)}/{plan['n_launches']} launches done, "
            f"{len((prior or {}).get('outstanding') or [])} outstanding"
            + (f" (last stop: {prior['stopped_reason']})" if prior and prior.get("stopped_reason") else ""))
        # REVOKE-BEFORE-REOPEN: a record whose checkpoint is COMPLETE has a verdict, and copies of
        # its geometry may be staged on clusters that cannot reach it. Re-running anything on it (an outstanding launch — a
        # result deleted by hand — or, with --replan, a re-opened prefill OOM) is a REOPEN, allowed only with the token the
        # front door mints AFTER it archived every staged overlay with a REVOKED marker: the record's current generation.
        # A successor the carrier resubmits continues an INCOMPLETE record and is never asked for one.
        results_pre = load_results(out)
        work_pre = [r["id"] for r in outstanding_launches(plan, results_pre)]
        if getattr(args, "replan", False):
            work_pre += replan_required_launches(plan, results_pre)
            # the EXTENSION the re-plan would add is work too — a complete record whose old
            # results are all terminal but whose pair is excluded or buffer-starved has nothing outstanding and nothing
            # replan-required, yet `--replan` would add S1cc / S1d launches, archive the menu and start engines; without
            # this the reopen needed no receipt and the clusters' geometry copies kept no revoke marker while the
            # certification changed. Computed here WITHOUT writing (extend_plan_for_uncertified_pairs is pure; the same
            # inputs give the same ids inside replan_record below); the receipt is required before any plan or artefact
            # mutates
            work_pre += [lid for lid in extend_plan_for_uncertified_pairs(plan, results_pre, _Dims(plan["dims"]))[1]["added"]
                         if lid not in work_pre]
            if certify_pairs:
                v2w = certify_v2_work(plan, results_pre, _Dims(plan["dims"]), certify_pairs)[1]
                work_pre += [lid for lid in v2w["added"] + v2w["supersede"] if lid not in work_pre]
        if getattr(args, "dry_run", False):
            # the operator's preview (the probe owner's ruling of 2026-10-11, "check the plan dry-run before submitting"): the same
            # pure derivations the resume acts on, printed; no lock, no receipt, no write
            ext_dry = (extend_plan_for_uncertified_pairs(plan, results_pre, _Dims(plan["dims"]))
                       if getattr(args, "replan", False) else (plan, {"added": [], "manifests": [], "pairs": [], "why": {}}))
            v2_dry = certify_v2_work(plan, results_pre, _Dims(plan["dims"]), certify_pairs)[1] if certify_pairs else None
            return print_resume_dry_run(out, plan, results_pre, prior, work_pre, ext_dry, v2_dry,
                                        replan=bool(getattr(args, "replan", False)))
        acquire_reopen_lock(out, what="the resumed run")
        if prior is not None and prior.get("complete") is True and work_pre:
            run_owner = consume_reopen_receipt(out, getattr(args, "reopen_token", None), action="run --resume",
                                               what=f"the reopen (re-running {len(work_pre)} launch(es))")[RECEIPT_CONSUME_KEY]["owner"]
        if getattr(args, "replan", False):
            # the measured budget re-derives S for the outstanding launches and re-opens the start-time prefill
            # OOMs above it (their results move to launches/superseded/); the plan is rewritten with the budget record
            rp = replan_record(out, plan, load_results(out), certify_v2=certify_pairs)
            plan = load_plan(out)
            if rp["replan_required"] and rp["budget"].get("tokens") is not None:
                log(f"--replan: re-running {len(rp['replan_required'])} superseded launch(es): {rp['replan_required']}")
        # REOPENING: a resumed attempt with launches to run changes the result set, so every
        # certification artifact derived from the old set — the menu, the verdict, the geometry record + this record's
        # site overlay, the memo, RESULT — is archived under launches/superseded/<utc>/ NOW (history kept, nothing
        # left to serve), and a NON-complete checkpoint naming this attempt is written BEFORE any GPU work starts
        results0 = load_results(out)
        work = [r["id"] for r in outstanding_launches(plan, results0)] + replan_required_launches(plan, results0)
        if work:
            _remember_parity_record(out, plan)
            plan = load_plan(out)
            archive_derived_artifacts(out, why=(f"run --resume{' --replan' if getattr(args, 'replan', False) else ''}: "
                                                f"{len(work)} launch(es) to run — every derived artifact predates this attempt"),
                                      model_key=str(plan["model_key"]))
            prog0 = write_progress(plan, results0, out, attempt={"started_utc": utc_now(), "job_id": os.environ.get("SLURM_JOB_ID"),
                                                                 "host": socket.gethostname(), "launches_to_run": len(work)})
            log(f"checkpoint: complete={prog0['complete']} ({len(work)} launch(es) to run) written before any GPU work; "
                f"result generation {str(prog0.get(RESULT_GENERATION_KEY))[:12]}")
    elif prior is not None and not prior.get("complete", False):
        raise SystemExit(f"{TAG} {out} holds an INCOMPLETE record ({prior.get('n_done')}/{prior.get('plan_launches')} "
                         f"launches done, stopped: {prior.get('stopped_reason')}); pass --resume to continue it, or "
                         "name a fresh --out")
    if not (out / PLAN_FILE).is_file():
        if not args.model_id:
            raise SystemExit(f"{TAG} no plan under {out} and no --model-id to write one")
        plan_argv = ["plan", "--model-key", args.model_key, "--model-id", args.model_id, "--out", str(out)]
        if args.revision:
            plan_argv += ["--revision", args.revision]
        plan_argv += [a for a in args.plan_args.split() if a]
        rc = main(plan_argv)
        if rc != 0:
            return rc
    plan = load_plan(out)
    exporter = ProductionExporter(plan["model_id"], plan["revision"], synthetic_scale=plan.get("synthetic_scale", DEFAULT_SYNTHETIC_SCALE))
    layout = build_adapter_dirs(plan, out, exporter)
    if layout["refused"]:
        log(f"{len(layout['refused'])} adapter slot(s) refused; their launches will record refused:adapters")
    deps = Deps.production()
    only = [s for s in (args.only or "").split(",") if s] or None
    # the concurrency and the per-standup device subsets: Slurm's CUDA_VISIBLE_DEVICES is READ, never set — each
    # slot is a strict subset of it; a request the allocation cannot hold refuses here, before any adapter export
    visible = visible_devices()
    try:
        parallel, slots = resolve_parallel(getattr(args, "parallel", PARALLEL_AUTO), tp=int(plan["tp"]), visible=visible,
                                           in_slurm=bool(os.environ.get("SLURM_JOB_ID")))
    except ValueError as e:
        raise SystemExit(str(e)) from None
    log(f"parallel={parallel} (requested {getattr(args, 'parallel', PARALLEL_AUTO)!s}; {VISIBLE_DEVICES_ENV}="
        f"{','.join(visible) if visible else '<unset>'}; tp={plan['tp']}) device slots="
        + (", ".join("[" + ",".join(sl) + "]" for sl in slots) if slots else "[unpinned]"))
    results = run_launches(plan, out, deps, gpu_mem_util=args.gpu_mem_util, ready_timeout_s=args.ready_timeout_s,
                           time_budget_s=args.time_budget_s, max_launches=args.max_launches,
                           throughput=not args.no_throughput, only_ids=only, parallel=parallel, device_slots=slots)
    if not args.no_reference:
        ref = reference_phase(plan, out, deps)
        log(f"reference phase: {ref.get('status')}" + (f" — {ref.get('reason')}" if ref.get("reason") else ""))
        # the reference phase rewrote the judged results: re-stamp the checkpoint with the final result generation
        write_progress(plan, load_results(out), out)
    else:
        log("WARNING --no-reference: the HF+PEFT parity check was SKIPPED — `verdict` will refuse certification "
            f"(exit {EXIT_UNCERTIFIED}) for every pair that passes the vLLM-side checks (D55)")
    n_pass = sum(1 for r in results.values() if r.get("status") == STATUS_PASSED)
    prog = load_progress(out) or progress_record(plan, results)
    b = prog.get("batched_token_budget") or {}
    log(f"run: {n_pass}/{len(results)} launches passed of {plan['n_launches']} planned; "
        f"{len(prog['outstanding'])} outstanding ({len(prog['decision_outstanding'])} decision-relevant); "
        f"batched-token budget {b.get('method')}={b.get('tokens')} (floor {b.get('floor_launch')}); "
        f"{len(prog.get('replan_required') or [])} replan_required")
    if prog["outstanding"] or prog.get("replan_required"):
        # an incomplete run is NOT a success: the sbatch requeues itself on this code and `verdict` refuses
        # to certify until every decision-relevant launch has run. A launch whose
        # OOM was a prefill above the measured budget is as open as an unrun one: the successor's
        # `run --resume --replan` re-derives its S and re-runs it.
        log(f"run INCOMPLETE (exit {EXIT_INCOMPLETE}): resume with `run --resume --replan --out {out}`"
            + (f" — {len(prog['replan_required'])} launch(es) need a re-derived S: {prog['replan_required'][:8]}"
               if prog.get("replan_required") else ""))
        return EXIT_INCOMPLETE
    # complete: a receipted resume's rewrite is done (the results and the complete checkpoint; the carrier's verdict needs no receipt):
    # finish the receipt this attempt consumed, or CLAIM the one an earlier attempt of this resume left in progress; an unreceipted
    # resume finds nothing in progress and this is a no-op
    owner = run_owner
    if owner is None and in_progress_receipts(out):
        owner = claim_in_progress_receipt(out, what="the resumed run (a successor attempt)")[1]
    finish_reopen_receipt(out, what="the resumed run", owner=owner)
    return 0


def cmd_verdict(args: argparse.Namespace) -> int:
    out = _resolve_out(args)
    plan = load_plan(out)
    # this record's own reference judgements are re-read under the CURRENT rule from their stored raw fields (in memory)
    results, regate = regate_reference_results(load_results(out), out)
    if regate["n_launches_changed"]:
        log(f"verdict: {regate['n_launches_changed']} launch(es) of this record re-gated under the current rule from their stored "
            f"reference fields ({len(regate['adapters_changed'])} adapter judgement(s) changed)")
    parity = None
    # a record REMEMBERS the parity-pass record its standing verdict read (plan.parity_record_dir, written at
    # the reopen before the verdict is archived, or bound — and REPLACED — by an explicit `--parity-record`, below): the
    # carrier's bare `verdict` after a reopen run then reads assertion 4 from it instead of this record's own scale-100
    # references (which the parity pass exists to replace) — a reopen never silently demotes a parity-certified menu
    parity_dir = getattr(args, "parity_record", None) or plan.get("parity_record_dir")
    if parity_dir:
        try:
            parity = load_parity_record(Path(parity_dir), plan)
        except ValueError as e:
            raise SystemExit(f"{TAG} {e}") from None
        if not getattr(args, "parity_record", None):
            log(f"verdict: --parity-record defaulted to the record's remembered parity pass {parity_dir} (plan.parity_record_dir)")
        else:
            # the EXPLICIT binding is remembered once the pass has loaded for THIS record (a pass that fails to load above leaves
            # the memory as it was, so `reverdict --resume-interrupted` reads the pass the record stood on) and REPLACES a
            # different remembered dir — the operator's replacement is durable (round 2 on 83da932b8); the verdict written
            # below stamps the same dir, so the plan and the standing verdict agree on the pass at rest
            _remember_parity_record(out, plan, parity_dir=parity["dir"])
            plan = load_plan(out)
        log(f"verdict: assertion 4 is read from the parity-pass record {parity['dir']} (generation "
            f"{str((parity.get('generation') or {}).get('generation'))[:12]}, scales {parity['plan'].get('synthetic_scale_by_set')}; D163)")
    verdict, allowed = write_verdict(plan, results, probe_job_id=str(args.probe_job_id), out=out,
                                     verdict_uri=args.verdict_uri or str(out / VERDICT_FILE),
                                     geometry_dir=getattr(args, "geometry_dir", None),
                                     probe_cluster=getattr(args, "probe_cluster", None), parity=parity)
    # the checkpoint follows the verdict's reading of the record: a partial verdict over a record whose checkpoint still
    # said complete (a result reopened by hand, then `verdict`) leaves complete=false behind, so hpo_probe_status says
    # "resume it" rather than "the verdict refused"; the stamps agree at rest; no attempt marker (no driver holds it)
    write_progress(plan, results, out, stopped_reason=(load_progress(out) or {}).get("stopped_reason"))
    n_run = sum(1 for l in verdict["launches"] if l["status"] != STATUS_NOT_RUN)
    fold = verdict.get("geometry_fold") or {}
    geom = (f"{fold.get('status')}:" + ",".join(f"{sid}(D={v.get('lora_max_module_dim')},budget={v.get('batched_token_budget')})"
                                                for sid, v in sorted((fold.get('sets') or {}).items()))
            + f",site={fold.get('site_path')}" if fold.get("status") in ("written", "record_only")
            else f"{fold.get('status')}:{str(fold.get('reason'))[:200]}")
    parity = (verdict.get(ADAPTER_PARITY_KEY) or {}).get("status")     # The second fact, on every RESULT line
    if verdict["status"] == VERDICT_UNCERTIFIED:
        # every launch ran, but a pair that passed the vLLM-side checks has NO HF+PEFT parity evidence: a non-certifying
        # record and a refusal a resubmission cannot cure. The geometry of the
        # serving-certified pairs is still folded, flagged `unverified` — an unsteered standup may size from it.
        line = (f"state=UNCERTIFIED certifying=false reason=reference_missing "
                f"pairs_unreferenced={len(verdict['unreferenced_pairs'])} pairs_allowed=0 "
                f"launches_run={n_run}/{plan['n_launches']} unreferenced={verdict['unreferenced_pairs']} "
                f"excluded={[[e['rank'], e['set'], e['reason'][:160]] for e in allowed['excluded']]} parity={parity} geometry={geom}")
        (out / RESULT_FILE).write_text(line + "\n", encoding="utf-8")
        log(line)
        why = sorted({str((r.get("reference_check") or {}).get("reason") or "")[:200] for r in results.values()
                      if (r.get("reference_check") or {}).get("basis") is None and r.get("status") in (STATUS_PASSED, STATUS_FAILED)} - {""})
        log(f"verdict REFUSED (exit {EXIT_UNCERTIFIED}): {ALLOWED_PAIRS_FILE} not written — the HF+PEFT reference check produced no "
            "evidence for pairs that passed every vLLM-side check (run --no-reference, a reference-phase crash, peft missing from "
            f"the serving venv, or the harness's own refusal: {REF_STATUS_MISSING})"
            + (f"; reasons: {' | '.join(why[:3])}" if why else "")
            + f"; re-run the reference phase alone: `run --resume --reference-only --out {out}` (1 GPU; a complete record needs the "
              "front door's reopen receipt)")
        return EXIT_UNCERTIFIED
    if verdict["status"] != VERDICT_COMPLETE:
        # decision-relevant launches are outstanding: a clearly PARTIAL, non-certifying record and a refusal —
        # never an allowed_pairs.json that excludes unrun pairs
        pending = [o["id"] for o in verdict["outstanding"] if o["decision_relevant"]]
        b = verdict.get("batched_token_budget") or {}
        line = (f"state=PARTIAL certifying=false pairs_undecided={len(verdict['undecided_pairs'])} "
                f"pairs_allowed_so_far={len(allowed['pairs'])} launches_run={n_run}/{plan['n_launches']} "
                f"decision_outstanding={len(pending)} replan_required={len(verdict['replan_required'])} "
                f"budget={b.get('method')}:{b.get('tokens')} undecided={verdict['undecided_pairs']} "
                f"next={pending[:6]} replan={verdict['replan_required'][:6]}")
        (out / RESULT_FILE).write_text(line + "\n", encoding="utf-8")
        log(line)
        log(f"verdict REFUSED (exit {EXIT_INCOMPLETE}): {ALLOWED_PAIRS_FILE} not written; "
            f"`run --resume --replan --out {out}` runs the outstanding launches first"
            + (f" and re-derives S for the {len(verdict['replan_required'])} start-time prefill OOM(s) above the "
               f"measured budget (D108)" if verdict["replan_required"] else ""))
        return EXIT_INCOMPLETE
    b = verdict.get("batched_token_budget") or {}
    line = (f"state=VERDICT pairs_allowed={len(allowed['pairs'])} pairs_excluded={len(allowed['excluded'])} "
            f"launches_run={n_run}/{plan['n_launches']} basis={verdict['reference_check_basis']} parity={parity} "
            f"budget={b.get('method')}:{b.get('tokens')} "
            f"allowed={[[p['rank'], p['set']] for p in allowed['pairs']]} "
            f"excluded={[[e['rank'], e['set'], e['reason']] for e in allowed['excluded']]} geometry={geom}")
    (out / RESULT_FILE).write_text(line + "\n", encoding="utf-8")
    log(line)
    if fold.get("status") == "refused":
        # the menu is certified (allowed_pairs.json stands) but the serving geometry could not be folded from the certified
        # launches — the padded launchers refuse this model by name until it is; loud here, never a silent 0
        log(f"geometry fold REFUSED (exit {EXIT_UNCERTIFIED}): {fold.get('reason')}")
        return EXIT_UNCERTIFIED
    return 0


def cmd_memo(args: argparse.Namespace) -> int:
    out = _resolve_out(args)
    vp = out / VERDICT_FILE
    if not vp.is_file():
        raise SystemExit(f"{TAG} no {VERDICT_FILE} under {out}; run `verdict` first")
    verdict = json.loads(vp.read_text(encoding="utf-8"))
    memprobe = None
    if args.memprobe:
        if not Path(args.memprobe).is_file():   # a named record that is not there is a refusal, not a placeholder
            raise SystemExit(f"{TAG} --memprobe {args.memprobe} is not a file")
        memprobe = json.loads(Path(args.memprobe).read_text(encoding="utf-8"))
    text = render_memo(verdict, memprobe)
    (out / MEMO_FILE).write_text(text, encoding="utf-8")
    log(f"memo written: {out / MEMO_FILE}")
    return 0


def cmd_reverdict(args: argparse.Namespace) -> int:
    """Rewrite the record's certification artifacts from its results under this code (no GPU): the stage-02 lane runs it
    over a record whose verdict a pre-stamp carrier wrote, before `03 --phase probe` reads it. Refuses a record whose
    checkpoint is not complete (a driver may hold it) — resume that record instead. The geometry record is re-derived
    from the record's results under this code — every serving-certified pair folded, the adapter-parity facts beside it —
    with NOTHING re-run (the reference phase's result stands as recorded)."""
    out = Path(args.record)
    if not (out / PLAN_FILE).is_file():
        raise SystemExit(f"{TAG} reverdict: no {PLAN_FILE} under {out}")
    prog = load_progress(out)
    if prog is not None and not prog.get("complete", False):
        raise SystemExit(f"{TAG} reverdict: {out}/{PROGRESS_FILE} is NOT complete ({prog.get('n_done')}/{prog.get('plan_launches')} done, "
                         f"{len(prog.get('outstanding') or [])} outstanding, {len(prog.get('replan_required') or [])} replan_required) — "
                         "a driver may hold the record; `run --resume --replan` it instead of re-verdicting a moving result set")
    # re-check 2/3: a reverdict rewrites the certification every staged copy of this record stands on — a reopen, receipted;
    # --resume-interrupted adopts the reopen an earlier reverdict consumed and did not finish
    # ONE certifying rewrite at a time: the record's reopen lock is held from here to the finish (released by the
    # kernel if this process dies); the recovery CLAIMS the in-progress receipt by rename and completion is owner-checked
    acquire_reopen_lock(out, what="the reverdict")
    try:
        return _cmd_reverdict_locked(args, out, prog)
    finally:
        release_reopen_lock(out)          # on every exit — a refusal, an exception, success; a dead process drops it with its fds


def _cmd_reverdict_locked(args: argparse.Namespace, out: Path, prog: Optional[Mapping[str, Any]]) -> int:
    if getattr(args, "resume_interrupted", False):
        if getattr(args, "reopen_token", None):
            raise SystemExit(f"{TAG} reverdict: --resume-interrupted takes no --reopen-token — it adopts the in-progress receipt")
        _doc, owner = resume_interrupted_receipt(out, what="the reverdict")
    else:
        owner = consume_reopen_receipt(out, getattr(args, "reopen_token", None), what="the reverdict", action="reverdict")[RECEIPT_CONSUME_KEY]["owner"]
    prior = None
    if (out / VERDICT_FILE).is_file():
        try:
            prior = json.loads((out / VERDICT_FILE).read_text(encoding="utf-8"))
        except ValueError:
            prior = None
    job_id = args.probe_job_id or (prior or {}).get("probe_job_id") or os.environ.get("SLURM_JOB_ID") or "reverdict"
    plan = load_plan(out)
    # an explicit --parity-record is remembered by cmd_verdict, AFTER the pass has loaded for this record (round 2 on 83da932b8) —
    # never here, before the load: a pass that refuses must leave plan.parity_record_dir as it was
    archive_derived_artifacts(out, why=f"reverdict: rewriting the certification artifacts from the record under tree {_tree_sha()}",
                              model_key=str(plan["model_key"]), geometry_dir=getattr(args, "geometry_dir", None))
    ns = argparse.Namespace(model_key=plan["model_key"], out=out, probe_job_id=str(job_id), verdict_uri=str(out / VERDICT_FILE),
                            geometry_dir=getattr(args, "geometry_dir", None), probe_cluster=getattr(args, "probe_cluster", None),
                            parity_record=getattr(args, "parity_record", None))
    rc = cmd_verdict(ns)
    if (out / VERDICT_FILE).is_file():
        cmd_memo(argparse.Namespace(model_key=plan["model_key"], out=out, memprobe=getattr(args, "memprobe", None)))
    # the checkpoint's stamp follows the (unchanged) result set so the three stamps agree at rest
    write_progress(plan, load_results(out), out, stopped_reason=(prog or {}).get("stopped_reason"))
    # the certifying rewrite is complete (a verdict — certifying or not — a memo when it certified, the checkpoint): the receipt is used
    finish_reopen_receipt(out, what="the reverdict", owner=owner)
    log(f"reverdict: verdict rc={rc}; result generation {str((result_generation(out) or {}).get('generation'))[:12]}")
    return rc


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return {"plan": cmd_plan, "adapters": cmd_adapters, "run": cmd_run, "verdict": cmd_verdict, "memo": cmd_memo,
            "reverdict": cmd_reverdict}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
