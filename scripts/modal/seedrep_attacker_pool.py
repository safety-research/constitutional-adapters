"""The seedrep battery's ATTACKER POOLS: N deployed Modal Servers, each one Qwen3.6-27B-abliterated
@27502c87 on ONE ``H200`` (tensor-parallel 1) at a 65,536 window, chunked prefill 16,384, S_att DERIVED per attack
from the measured attacker traffic and the production fan-out (:func:`derive_pool_sizing`) — a PARAMETERISED
copy of ``scripts/modal/attacker_pool.py``. The original is untouched: it serves the campaign pool at 19,200 × 37
on one card, sized for the 2,048-cap era; a Crescendo attacker context accumulates the whole target conversation
at the target's 4,096 per-reply cap (measured pool-side 2026-09-19: 3.4 % of crescendo attacker calls carry a
prompt above 20,000 tokens, none above 50,000), so these pools need their own window, their own count and their
own names.

WHY ONE CARD. The pools were first deployed on ``H200:2`` to reach S 44 at 65,536 under
the memory model's IDENTITY rule (KV covers S × L: no preemption whatever the requests look like). Nobody had
re-derived that against the traffic: the identity sizes every slot for a 65,536-token request while the crescendo
attacker's MEAN prompt is ~4,800 tokens (pair ~1,700, actorattack ~840), and an existing Slurm deployment serves the
same 27B attacker on ONE H200 (``scripts/sdfbase/full22/attacker_pool.sbatch``: ``--max-model-len 131072 --util 0.85``,
vLLM's default max_num_seqs, no identity) for the same multi-turn campaigns. So these pools follow the Slurm
precedent under the contract's ``peak_mean`` RULE (``resolve_pool_geometry(..., kv_seq_tokens=,
kv_rule="peak_mean")``): S = the production PEAK (children pinned to the pool × :data:`PEAK_HEADROOM`), the KV
budget checked at the MEAN sequence (measured mean prompt + the attacker's 1,024 reply cap; S × mean and one full
window must fit the card, else the pool tool refuses BY NAME), vLLM's scheduler queuing / preempting on the tail as
the safety net; the window holds every observed call (max prompt bucket + reply cap); the live coverage line is
checked under the SAME rule and the p99 coverage FRACTION recorded beside it. The p99 fit (the S the KV would hold
at the p99 sequence, :func:`derive_pool_sizing` ``p99_fit_S``) is recorded BESIDE the launched S, never as the bound.
The derivation's inputs are pinned here (:data:`MEASURED_ATTACKER_CALLS`, :data:`PRODUCTION_PEAK_CHILDREN`), the
endpoint record carries the arithmetic (``sizing``) and the two vLLM metrics a launch watches for the tail queuing
(``watch_metrics``: the waiting gauge, the preemption-counter DELTA, the queue-time and e2e-latency
histogram deltas, each with its corrective threshold; :data:`WATCH_METRICS`) plus the rule's ``caveat`` (:data:`PEAK_MEAN_CAVEAT`:
the call-weighted mean is an ESTIMATE of resident occupancy, slot coverage is not KV residency).

WHAT IS PARAMETERISED, AND HOW. ``modal deploy`` imports this module, and the ``@app.server`` declaration
needs its geometry at import time, so every parameter is an ENVIRONMENT VARIABLE read once by
:func:`pool_params` (defaults = the plan's; every value validated; the resolved params baked into the
image env so the container reads the same ones) — never a CLI flag the decorator cannot see:

  SEEDREP_ATTACKER_POOL_INDEX   k (REQUIRED)   -> app ``seedrep-attacker-<k>``, pool dir /vol/attacker-pool/seedrep-<k>/
  SEEDREP_ATTACKER_ATTACK       (REQUIRED)     the attack whose drivers pin this pool: crescendo | actorattack | pair
  SEEDREP_ATTACKER_GPU          H200           ``H200`` or ``H200:<n>``; n = the tensor-parallel size (ONE card)
  SEEDREP_ATTACKER_MAX_MODEL_LEN 65536         the window (holds every observed attacker call: max prompt bucket 50,000 +
                                               the 1,024 reply cap, 16,384-aligned); a multiple of 16, >= 8192
  SEEDREP_ATTACKER_CHUNK_TOKENS 16384          chunked-prefill batch (<= the window)
  SEEDREP_ATTACKER_MAX_NUM_SEQS (unset)        explicit S_att (refused by name when S x mean sequence does not fit); unset =
                                               DERIVED (:func:`derive_pool_sizing`: the production peak x headroom, peak_mean)
  SEEDREP_ATTACKER_GPU_MEM_UTIL 0.90
  SEEDREP_ATTACKER_LIFETIME_H   24             the lifetime the pool is deployed for; :func:`pool_outlives` holds it
                                               against the longest target container INCLUDING its Modal queue wait
  SEEDREP_ATTACKER_W7 / SEEDREP_ATTACKER_WAVE_UUID8   provenance only (recorded in endpoint.json)

The memory model is the contract module's (``sisteer.eval.attacks.attacker_pool.resolve_pool_geometry``) under
the ``peak_mean`` rule above; a tensor-parallel deployment (``H200:<n>``, still accepted) applies it PER CARD
(:func:`per_card_arch`: weights and KV/linear heads split by the TP size) with a per-card TENSOR-PARALLEL RESERVE on
top of the one-card reserve (:data:`TP_RESERVE_BYTES`: the communicators and graphs a multi-card engine holds that a
one-card one does not -- NCCL, the custom all-reduce and FlashInfer all-reduce workspaces, the FULL+PIECEWISE graph
sets; sized from the 2026-09-18 pool-0 deploy, :data:`LIVE_ROOM_MEASURED`, where the engine reported room for 45.07
sequences of 65,536 against the one-card model's kv_fit 46 and the Server refused to advertise). The one-card
reserve is the contract's (2 GiB: CUDA context, graphs -- 0.19 GiB measured at TP 2 -- and the sampler workspace).
The live coverage line after readiness is still the fact, checked under the launched geometry's KV rule
(``coverage_problem(..., kv_seq_tokens=, kv_rule=)``), and ANY short coverage is fatal, as in the original.

ALLOCATOR CONFIG follows the card count (:func:`alloc_conf_env`). The one-card pools run vLLM under
``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` exactly as the one-card campaign pool does (the
constraint below is the custom all-reduce's IPC export, which only exists above one card -- vLLM's
``custom_all_reduce`` is not constructed at world size 1); a MULTI-card pool must NOT: vLLM's custom
all-reduce registers every CUDA-graph all-reduce buffer across the ranks with ``cudaIpcGetMemHandle`` at the end
of graph capture (``csrc/custom_all_reduce.cuh:455`` in 0.25.1, ``get_graph_buffer_ipc_meta``), the buffers come
from torch's caching allocator, and under expandable segments those are ``cuMemMap`` ranges the legacy IPC export
refuses with ``'invalid argument'`` -- the four H200:2 Servers loaded, compiled and warmed up, then died at the
first capture and crash-looped for 50 minutes with no ``endpoint.json``. Above one card the image sets no
allocator config (torch's default, the environment every vLLM tensor-parallel deployment runs) and the Server
REFUSES to launch under a re-introduced ``expandable_segments:True`` (:func:`serve_env_problem`).

POOL COUNT is sized by DRIVERS, not legs (memory rule "size a shared pool by drivers, not legs"): S4 records
attacker Running/Waiting at 20 concurrent goals and the launch plan calls :func:`size_pool_plan` with the
fan-out's driver total per attack; the default before S4 is 4 pools (Crescendo ×2, ActorAttack ×1, PAIR ×1),
never fewer than one per attack. Drivers PIN a pool at launch and never fail over (memory rule "attacker
walltime must outlive its drivers"), so :func:`pool_outlives` refuses a lifetime shorter than the longest
target container plus its queue wait, and every pool is UP and health-checked (:func:`endpoint_health`:
``/health`` 200, ``/v1/models`` 401/403 bare and 200 with the bearer) BEFORE any target container spawns.

REQUEST ROUTING (target vs attacker) is a pure contract here so the launcher, the drivers' pin records and
the request-evidence audit read ONE rule (:func:`route_request`): a body is routed by its ``model`` — an
attacker model goes to the pool at the ATTACK's own parameters (never held to a row), a target vmodel goes
to the target proxy and MUST carry exactly the vendor row's six sampling keys read from
the registry by exact ref (:func:`target_row`, ``by_ref("qwen35_4b_it|multi_turn|off|vendor_preset@v1")`` +
not-superseded — never a default lookup) plus a ``seed``; a target body carrying the attacker's parameters,
an unseeded target body, or a model neither side serves is REFUSED (:class:`RoutingRefused`).

Every CLI invocation is explicitly named (Modal 1.5 calls the app-name flag ``--name``); the deploy name MUST
equal ``seedrep-attacker-<k>`` — the server refuses to advertise under any other name:

  # stage the snapshot once (shared with the campaign pool: same /vol/hf path)
  SEEDREP_ATTACKER_POOL_INDEX=0 SEEDREP_ATTACKER_ATTACK=crescendo ~/venvs/modal/bin/python -m modal run \
    --name seedrep-attacker-0-stage scripts/modal/seedrep_attacker_pool.py::tools.main --action stage

  # deploy pool k (one command per pool; the roster says which attack each k serves)
  SEEDREP_ATTACKER_POOL_INDEX=0 SEEDREP_ATTACKER_ATTACK=crescendo ~/venvs/modal/bin/python -m modal deploy \
    --name seedrep-attacker-0 scripts/modal/seedrep_attacker_pool.py::app

  # endpoint record -> $STATE/sisteer/attacker-pool-modal/seedrep/endpoint_0.json (or under the site's SISTEER_ATTACKER_POOL_STATE_DIR) ; health ; canary
  SEEDREP_ATTACKER_POOL_INDEX=0 SEEDREP_ATTACKER_ATTACK=crescendo ~/venvs/modal/bin/python -m modal run \
    --name seedrep-attacker-0-endpoint scripts/modal/seedrep_attacker_pool.py::tools.main --action endpoint
  SEEDREP_ATTACKER_POOL_INDEX=0 SEEDREP_ATTACKER_ATTACK=crescendo ~/venvs/modal/bin/python \
    scripts/modal/seedrep_attacker_pool.py --health          # grading-host side, no Modal function
  SEEDREP_ATTACKER_POOL_INDEX=0 SEEDREP_ATTACKER_ATTACK=crescendo ~/venvs/modal/bin/python -m modal run \
    --name seedrep-attacker-0-canary scripts/modal/seedrep_attacker_pool.py::tools.main --action canary

  # stop OUR pool at wave end (never a peer's app)
  ~/venvs/modal/bin/python -m modal app stop --yes seedrep-attacker-0

The canary is CPU-only and compares Modal container ids before doing any HTTP work. Same-container evidence
is a refusal, never a green canary.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import modal

# Modal mounts a source file flat at /root/<name>.py in remote containers. Resolve checkout paths only on
# the grading host; the baked source exists at the destination path when the module is imported remotely.
if modal.is_local():
    REPO = Path(__file__).resolve().parents[2]
    SOURCE_DIR = REPO / "src" / "sisteer"
    if str(REPO / "src") not in sys.path:
        sys.path.insert(0, str(REPO / "src"))
else:
    REPO = Path("/root")
    SOURCE_DIR = Path("/root/sisteer")

# the contract module is stdlib-only by design (its docstring) so it imports from the modal venv too
from sisteer.eval.attacks import attacker_pool as contract  # noqa: E402

LANE = "seedrep"
SERVER_NAME = "qwen36-27b-abliterated-vllm"
VOL_NAME = "twosi-fits"
VOL_PATH = "/vol"
PORT = 8000

ATTACKER_MODEL = contract.ATTACKER_MODEL
ATTACKER_SNAPSHOT = contract.ATTACKER_SNAPSHOT
assert ATTACKER_SNAPSHOT.startswith("27502c87"), ATTACKER_SNAPSHOT   # the plan's attacker snapshot (§4.5, SHA_PROVENANCE)
MODEL_REPO_DIR = f"{VOL_PATH}/hf/hub/models--huihui-ai--Huihui-Qwen3.6-27B-abliterated"
MODEL_PATH = f"{MODEL_REPO_DIR}/snapshots/{ATTACKER_SNAPSHOT}"
LEGACY_MODEL_REPO_DIR = f"{VOL_PATH}/grokmt/hf_attacker/models--huihui-ai--Huihui-Qwen3.6-27B-abliterated"
API_KEY_ENV = contract.API_KEY_ENV
#: the env name the DRIVER containers (seedrep_served.py) carry this same lane bearer under -- distinct from the
#: pools' VLLM_API_KEY because a driver container's VLLM_API_KEY is the TARGET proxy's bearer; defined ONCE in the
#: stdlib-only contract module so the driver (jailbreak_run._vllm_api_key) and this app read one name
DRIVER_POOL_KEY_ENV = contract.POOL_BEARER_ENV
# Text-only: the attacker never sees an image (the original's rationale: no vision modules, no encoder profile)
LIMIT_MM_PER_PROMPT = '{"image": 0, "video": 0}'
POOL_QUANTIZATION = "none"
POOL_PREFIX_CACHING = True
VLLM_VERSION = "0.25.1+cu129"


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

# --------------------------------------------------------------------------- parameters (pure, tested)

ENV_PREFIX = "SEEDREP_ATTACKER_"
ATTACKS: Tuple[str, ...] = ("crescendo", "actorattack", "pair")
#: the plan's pre-S4 default: 4 pools, Crescendo ×2, ActorAttack ×1, PAIR ×1 (§4.5)
DEFAULT_POOL_PLAN: Tuple[Tuple[str, int], ...] = (("crescendo", 2), ("actorattack", 1), ("pair", 1))
DEFAULT_GPU = "H200"                  # ONE card since 2026-09-19; H200:2 was the earlier identity-rule sizing
DEFAULT_MAX_MODEL_LEN = 65_536        # holds every observed attacker call: see WINDOW_DERIVATION
DEFAULT_CHUNK_TOKENS = 16_384
DEFAULT_GPU_MEM_UTIL = 0.90
DEFAULT_LIFETIME_H = 24.0
MIN_MAX_MODEL_LEN = 8_192
ALLOWED_TP = (1, 2, 4, 8)
_GPU_RE = re.compile(r"^(H200|H100|A100-80GB)(?::(\d+))?$")
#: the one-card campaign pool's allocator setting (attacker_pool.py); NEVER baked for a multi-card pool -- see
#: :func:`alloc_conf_env` and the module docstring (F-M)
EXPANDABLE_SEGMENTS = "expandable_segments:True"
#: per-card reserve ADDED to the contract's one-card ``RESERVE_BYTES`` when the engine spans several cards: NCCL channel
#: buffers, the custom all-reduce buffers, the FlashInfer all-reduce (mnnvl / trtllm) workspaces and the FULL+PIECEWISE
#: CUDA-graph sets of a tensor-parallel worker. MEASURED 2026-09-18 on pool 0 (H200:2, L 65,536, chunk 16,384, util 0.90):
#: the one-card model (weights 25.9 + KV 92.0 + state 3.3 + activations 2.6 + reserve 2.0 GiB) fit S 46 with 0.1 GiB to
#: spare; the live engine reported room for 45.07 sequences, i.e. ~1.9 GiB per card the model did not know about (the
#: weights alone came in 0.7 GiB UNDER the model at 25.17 GiB, so the communicator/graph excess is ~2.6 GiB). 3 GiB =
#: that excess + ~1 GiB of host-to-host slack (CUDA context and driver variance across Modal hosts), so the launched S
#: sits a whole sequence (~2.1 GiB) inside the live room rather than 0.07 of one. Same value for tp 4 / 8: unmeasured
#: there, and the Server's coverage check remains the fact.
TP_RESERVE_BYTES = 3 * contract.GIB
#: vLLM's live "Maximum concurrency for L tokens per request" per (gpu_kind, tensor_parallel, max_model_len), as the
#: Server's coverage attestation read it -- the number the launched S must sit at or below (tests pin the relation).
LIVE_ROOM_MEASURED: Dict[Tuple[str, int, int], float] = {
    ("H200", 2, 65_536): 45.07,        # pool 0, 2026-09-18 06:37Z, at the one-card model's S 46 (refused)
}

# -------------------------------------------------------------------------------- sizing inputs (measured, pinned)

#: the attacker's per-reply cap the drivers send (``attacks.attacker_max_new_tokens: 1024`` in every
#: ``configs/sdfbase_full22_qwen35_4b_*.yaml``; ``jailbreak_run._attacker_max_new_tokens`` default 1024). NOT the
#: target's 4,096 (``--target-max-new-tokens``): the pools' ``vllm:request_params_max_tokens`` histogram puts every
#: generation call in the (1,000, 2,000] bucket and every scorer call (``make_scorer_fn_vllm``, 8 tokens) in (5, 10].
ATTACKER_REPLY_CAP_TOKENS = 1_024
#: the production peak's headroom: S = the pinned children x this (the ``peak_mean`` rule's S)
PEAK_HEADROOM = 1.25
#: the p99 fit's slack, in p99 sequences (recorded BESIDE the launched S as ``p99_fit_S``, never the bound): the largest S
#: whose S x p99 fits is stepped down until the model's headroom holds at least this many p99 sequences' KV -- F-M's lesson on
#: one card (the live engine found 2 % less room than the model at TP 2 and the Server refused to advertise)
KV_FIT_SLACK_SEQS = 1
#: the KV rule the seedrep pools launch under (the contract's name for it; the Slurm precedent's shape)
POOL_KV_RULE = "peak_mean"
#: the ``peak_mean`` rule's caveat, recorded on every endpoint record: what the mean-sequence check does and does NOT say
PEAK_MEAN_CAVEAT = (
    "peak_mean sizes SLOTS to the production peak and checks the KV budget at the CALL-WEIGHTED mean sequence (prompt_tokens_sum / "
    "count + the reply cap). That mean is an ESTIMATE of resident occupancy, not a measured resident-request mean: a long request "
    "holds its slot (and its KV) for longer than a short one, so the resident mix is longer than the call mix, and the children of "
    "one leg can enter their long-context phases together (crescendo's late turns). slots_cover_peak says the scheduler can ADMIT "
    "the peak; kv_resident_at says at which per-sequence length S sequences stay RESIDENT (mean: yes; p99: only S_fit of them) -- "
    "past that vLLM queues (num_requests_waiting) or preempts and recomputes (num_preemptions_total). The live allocation check "
    "at readiness holds S x mean and one full window; the tail is watched, not provisioned: watch_metrics.")
#: the window the launch agent reads the watch metrics over, and the rules: a gauge, a COUNTER DELTA and two latency
#: histograms (their bucket DELTAS over the window), each with a corrective threshold at representative fan-out and the action --
#: report, then lower S (SEEDREP_ATTACKER_MAX_NUM_SEQS, an explicit override the endpoint record now describes honestly) or add a pool
#: (size_pool_plan). Read on the smoke rows before the production wave, and per pool during it. Every sample
#: over the measured 17:10Z-19:02Z window had waiting 0 and preemptions 0.
WATCH_WINDOW_S = 600
WATCH_METRICS: Tuple[Dict[str, Any], ...] = (
    {"metric": "vllm:num_requests_waiting", "read": "gauge", "window_s": WATCH_WINDOW_S,
     "threshold": "waiting > 0 on more than 10 % of the window's 10 s samples, or any sample >= 0.25 x S",
     "meaning": "requests the scheduler cannot admit: slots or KV short at this fan-out",
     "action": "report; lower S (explicit SEEDREP_ATTACKER_MAX_NUM_SEQS) or add a pool for the attack"},
    {"metric": "vllm:num_preemptions_total", "read": "counter_delta", "window_s": WATCH_WINDOW_S,
     "threshold": "delta > 0 over the window",
     "meaning": "resident sequences evicted and recomputed: the KV did not hold the resident mix (the mean under-estimated it)",
     "action": "report; lower S or add a pool -- never raise gpu_memory_utilization past 0.90"},
    {"metric": "vllm:request_queue_time_seconds", "read": "histogram_delta_p95", "window_s": WATCH_WINDOW_S,
     "threshold": "p95 > 30 s",
     "meaning": "time a request waited before its first schedule: the queue's cost to the drivers",
     "action": "report; lower S or add a pool"},
    {"metric": "vllm:e2e_request_latency_seconds", "read": "histogram_delta_p95", "window_s": WATCH_WINDOW_S,
     "threshold": "p95 > 600 s, or ANY driver-side timeout / transport error against the pool (response_evidence.jsonl rows with status "
                  "null or error naming the pool url)",
     "meaning": "the attacker call as the driver sees it; a timeout turns a goal into an error outcome",
     "action": "report at once; lower S or add a pool; a timeout is a wave defect, not a tail"},
)
WATCH_METRIC_NAMES: Tuple[str, ...] = tuple(m["metric"] for m in WATCH_METRICS)
#: the attacker calls the four deployed pools served, read from each pool's vLLM ``/metrics`` (the Prometheus
#: histograms ``vllm:request_prompt_tokens_bucket`` / ``vllm:request_generation_tokens_bucket``, cumulative
#: ``(le, count)`` pairs) at 2026-09-19 ~19:00Z -- cumulative since the pools' 07:02Z deploy, i.e. every multi-turn
#: row's attacker traffic (the pair smoke, the crescendo / actorattack smokes and the crescendo FULL row,
#: production-shaped children). Pool 0 served every crescendo row (pool 1 idle: the smokes pin shard 0 only), pool 2
#: actorattack, pool 3 pair. ``waiting_max`` / ``running_max`` are the serve logs' ``Running: N reqs, Waiting: M reqs``
#: peaks over the same window (17:10Z-19:02Z, every 10 s). Re-measure by re-reading ``/metrics`` (bare GET; the pool's
#: bearer guards ``/v1`` only) and replace these numbers -- never edit one by hand.
MEASURED_ATTACKER_CALLS: Dict[str, Dict[str, Any]] = {
    "crescendo": {
        "pool": "seedrep-attacker-0", "n_calls": 15_478, "prompt_tokens_mean": 4_756,
        "prompt_buckets": ((100, 1), (200, 297), (500, 3_201), (1_000, 6_652), (2_000, 9_065), (5_000, 10_778),
                           (10_000, 12_410), (20_000, 14_950), (50_000, 15_478), (float("inf"), 15_478)),
        "generation_buckets": ((2, 9_292), (5, 9_465), (100, 9_474), (200, 11_138), (500, 15_477), (1_000, 15_478),
                               (float("inf"), 15_478)),
        "max_tokens_buckets": ((10, 9_465), (1_000, 9_465), (2_000, 15_478), (float("inf"), 15_478)),
        "running_max": 17, "waiting_max": 0, "kv_usage_max_pct": 7.2,
    },
    "actorattack": {
        "pool": "seedrep-attacker-2", "n_calls": 5_375, "prompt_tokens_mean": 840,
        "prompt_buckets": ((200, 260), (500, 1_677), (1_000, 3_797), (2_000, 5_176), (5_000, 5_375), (float("inf"), 5_375)),
        "generation_buckets": ((2, 2_917), (5, 2_921), (10, 2_934), (20, 3_164), (50, 4_943), (100, 5_151), (200, 5_157),
                               (500, 5_362), (1_000, 5_369), (2_000, 5_375), (float("inf"), 5_375)),
        "max_tokens_buckets": ((10, 2_921), (1_000, 2_921), (2_000, 5_375), (float("inf"), 5_375)),
        "running_max": 4, "waiting_max": 0, "kv_usage_max_pct": 0.4,
    },
    "pair": {
        "pool": "seedrep-attacker-3", "n_calls": 10_320, "prompt_tokens_mean": 1_667,
        "prompt_buckets": ((200, 767), (500, 3_345), (1_000, 4_562), (2_000, 7_129), (5_000, 10_159), (10_000, 10_318),
                           (20_000, 10_320), (float("inf"), 10_320)),
        "generation_buckets": ((2, 4_938), (5, 5_140), (200, 5_504), (500, 10_304), (1_000, 10_320), (float("inf"), 10_320)),
        "max_tokens_buckets": ((10, 5_140), (1_000, 5_140), (2_000, 10_320), (float("inf"), 10_320)),
        "running_max": 9, "waiting_max": 0, "kv_usage_max_pct": 1.7,
    },
}
MEASURED_ATTACKER_CALLS_ON = "2026-09-19T19:00Z"
#: the children (``jailbreak_run.py generate`` drivers, one arm-run x goal sub-shard each) the DRAFT plan pins to each
#: pool under the default pool plan (4 pools: crescendo x2, actorattack, pair; 3 views x 8 arm-runs + 3 Arditi arm-runs per leg,
#: crescendo 4 / actorattack 5 containers per view, 2 Arditi containers, dealt round-robin over the attack's pools):
#: crescendo pools 0 / 1: 6 view rows x 8 + 1 Arditi row x 3 = 51 each; actorattack pool 2: 15 x 8 + 2 x 3 = 126; pair
#: pool 3: 6 x 8 + 1 x 3 = 51. A child runs ONE goal at a time (measured: the crescendo FULL row's 8 children
#: lifted pool 0's Running from <= 5 to <= 17 with two <= 5-child smoke rows alongside), so the pool's peak concurrent
#: attacker calls is its pinned children. ``tests/test_seedrep_fm2_one_card_pools.py`` pins these against the plan.
PRODUCTION_PEAK_CHILDREN: Dict[str, int] = {"crescendo": 51, "actorattack": 126, "pair": 51}
#: the window's derivation: the largest observed prompt bucket over every attack (crescendo's 50,000) + the reply
#: cap, rounded up to the chunk size -- 51,024 -> 65,536; the ruling's assumed 4,096 cap gives 54,096 -> the same window.
WINDOW_DERIVATION = {"max_prompt_bucket": 50_000, "reply_cap": ATTACKER_REPLY_CAP_TOKENS, "align": DEFAULT_CHUNK_TOKENS,
                     "window": DEFAULT_MAX_MODEL_LEN}


def histogram_quantile(buckets: Sequence[Tuple[float, int]], q: float) -> float:
    """Prometheus's ``histogram_quantile`` estimator over cumulative ``(le, count)`` buckets: the quantile's bucket,
    linearly interpolated between its lower and upper bound (the ``+Inf`` bucket resolves to its lower bound)."""
    if not (0.0 < q <= 1.0):
        raise ValueError(f"quantile {q} is not in (0, 1]")
    bs = sorted(((float(le), int(n)) for le, n in buckets), key=lambda x: x[0])
    if not bs or bs[-1][0] != float("inf"):
        raise ValueError("buckets must end with the +Inf bucket")
    total = bs[-1][1]
    if total < 1:
        raise ValueError("an empty histogram has no quantile")
    rank = q * total
    lower_le, lower_n = 0.0, 0
    for le, n in bs:
        if n >= rank:
            if le == float("inf"):
                return lower_le
            if n == lower_n:
                return le
            return lower_le + (le - lower_le) * (rank - lower_n) / (n - lower_n)
        lower_le, lower_n = le, n
    return bs[-1][0]


def p99_sequence_tokens(attack: str, *, reply_cap: int = ATTACKER_REPLY_CAP_TOKENS,
                        measured: Mapping[str, Mapping[str, Any]] = MEASURED_ATTACKER_CALLS) -> Dict[str, Any]:
    """The p99 SEQUENCE length the KV rule sizes with: the attack's p99 prompt (interpolated within its histogram
    bucket) + the reply cap (a slot holds prompt + generated-so-far; the cap is the most a reply can add). Also the
    bucket CEILING variant (the p99 bucket's upper edge + cap), the conservative reading of the same histogram."""
    if attack not in measured:
        raise PoolParamError(f"no measured attacker calls for attack {attack!r}; measured: {sorted(measured)}")
    m = measured[attack]
    p99_prompt = histogram_quantile(m["prompt_buckets"], 0.99)
    ceiling = min(le for le, n in m["prompt_buckets"] if n >= 0.99 * m["n_calls"])
    max_bucket = min(le for le, n in m["prompt_buckets"] if n >= m["n_calls"])
    return {"attack": attack, "n_calls": int(m["n_calls"]), "p99_prompt_tokens": int(math.ceil(p99_prompt)),
            "p99_prompt_bucket_ceiling": int(ceiling), "max_prompt_bucket": int(max_bucket), "reply_cap": int(reply_cap),
            "p99_seq_tokens": int(math.ceil(p99_prompt)) + int(reply_cap),
            "p99_seq_tokens_ceiling": int(ceiling) + int(reply_cap), "measured_on": MEASURED_ATTACKER_CALLS_ON}


def mean_sequence_tokens(attack: str, *, reply_cap: int = ATTACKER_REPLY_CAP_TOKENS,
                         measured: Mapping[str, Mapping[str, Any]] = MEASURED_ATTACKER_CALLS) -> Dict[str, Any]:
    """The MEAN sequence length the ``peak_mean`` rule checks the KV at: the attack's measured mean prompt
    (``request_prompt_tokens_sum / count``) + the reply cap."""
    if attack not in measured:
        raise PoolParamError(f"no measured attacker calls for attack {attack!r}; measured: {sorted(measured)}")
    m = measured[attack]
    return {"attack": attack, "n_calls": int(m["n_calls"]), "mean_prompt_tokens": int(m["prompt_tokens_mean"]),
            "reply_cap": int(reply_cap), "mean_seq_tokens": int(m["prompt_tokens_mean"]) + int(reply_cap),
            "measured_on": MEASURED_ATTACKER_CALLS_ON}


def peak_slots(attack: str, *, headroom: float = PEAK_HEADROOM,
               peaks: Mapping[str, int] = PRODUCTION_PEAK_CHILDREN) -> int:
    """The slots that cover the attack's production peak: pinned children x headroom, rounded up."""
    if attack not in peaks:
        raise PoolParamError(f"no production peak for attack {attack!r}; known: {sorted(peaks)}")
    if headroom < 1.0:
        raise PoolParamError(f"headroom {headroom} < 1 would size S below the peak")
    return int(math.ceil(int(peaks[attack]) * float(headroom)))


def derive_pool_sizing(attack: str, max_model_len: int, *, util: float = DEFAULT_GPU_MEM_UTIL,
                       chunk_tokens: int = DEFAULT_CHUNK_TOKENS, tensor_parallel: int = 1,
                       headroom: float = PEAK_HEADROOM, strict: bool = True) -> Dict[str, Any]:
    """The arithmetic for one pool, pure, under the ``peak_mean`` rule: ``S = S_peak`` = pinned children x
    headroom; ``S_kv_mean`` = the largest S whose ``S x mean sequence`` the card's KV holds (the contract's rule on the per-card
    arch). The RECOMMENDATION is computed whether or not it fits (``fits``; ``unfit_reason`` when it does not); ``strict``
    (the AUTOMATIC launch, which would run this S) refuses BY NAME when ``S_peak x mean`` does not fit
    (:class:`contract.PoolGeometryUnfit`), while an EXPLICIT launch (``strict=False``) validates its own S in
    :func:`pool_geometry` and keeps the unfittable recommendation as diagnostic metadata (the automatic width
    must never veto a smaller explicit width that fits -- the supported recovery after a coverage-gate refusal). A window that
    cannot hold the attack's p99 sequence refuses either way (:class:`PoolParamError`: the window, not S, is wrong). Recorded
    BESIDE the recommendation, never as the bound: ``p99_fit_S`` = the S the p99 rule would have launched (KV bound at the p99
    sequence minus :data:`KV_FIT_SLACK_SEQS`, capped at the peak), ``S_kv_p99`` = that KV bound, ``identity_S`` = what S x L
    would give."""
    p99 = p99_sequence_tokens(attack)
    mean = mean_sequence_tokens(attack)
    arch = per_card_arch(contract.QWEN36_27B, tensor_parallel)
    reserve = pool_reserve_bytes(tensor_parallel)
    if int(p99["p99_seq_tokens"]) > int(max_model_len):
        # the refusal by name: a window that cannot hold the attack's p99 sequence is the wrong window, not a smaller S
        raise PoolParamError(
            f"{ENV_PREFIX}MAX_MODEL_LEN={max_model_len} cannot hold {attack}'s p99 sequence of {p99['p99_seq_tokens']:,} tokens "
            f"(p99 prompt {p99['p99_prompt_tokens']:,} over {p99['n_calls']:,} measured calls + the {p99['reply_cap']:,} reply cap): "
            f"the window, not S, is wrong -- the derived window is {WINDOW_DERIVATION['window']:,} (max prompt bucket "
            f"{WINDOW_DERIVATION['max_prompt_bucket']:,} + cap, {WINDOW_DERIVATION['align']:,}-aligned)")
    base = dict(util=util, chunked_prefill=True, chunk_tokens=chunk_tokens, arch=arch, reserve_bytes=reserve)
    s_peak = peak_slots(attack, headroom=headroom)
    # the recommendation's fit: S_peak x mean against the card, computed WITHOUT raising (fit, not the resolver) so an explicit
    # launch can record it; the automatic launch (strict) refuses by name through the resolver, which names the peak_mean rule
    # and the largest S that would fit
    g_mean = contract.fit(max_model_len, s_peak, kv_seq_tokens=mean["mean_seq_tokens"], kv_rule=POOL_KV_RULE, **base)
    s_kv_mean = int(g_mean.max_seqs_fitting)
    unfit_reason = None
    if not g_mean.fits:
        unfit_reason = (f"the automatic recommendation S_peak {s_peak} x mean sequence {mean['mean_seq_tokens']:,} does not fit this card "
                        f"at util {util} (largest S that fits under the peak_mean rule: {s_kv_mean}; model headroom "
                        f"{g_mean.headroom_bytes / contract.GIB:.2f} GiB)")
        if strict:
            contract.resolve_pool_geometry(max_model_len, s_peak, kv_seq_tokens=mean["mean_seq_tokens"], kv_rule=POOL_KV_RULE, **base)
            raise AssertionError("unreachable: the resolver refuses a geometry fit() said does not fit")     # pragma: no cover
    # beside it: the p99 fit (what the first reading launched), with its slack step
    kw99 = dict(kv_seq_tokens=p99["p99_seq_tokens"], kv_rule="p99_fit", **base)
    g99 = contract.resolve_pool_geometry(max_model_len, None, **kw99)
    s_kv_p99 = int(g99.max_num_seqs)
    slack_bytes = int(KV_FIT_SLACK_SEQS) * int(p99["p99_seq_tokens"]) * arch.kv_bytes_per_token
    while s_kv_p99 > 1 and g99.headroom_bytes < slack_bytes:
        s_kv_p99 -= 1
        g99 = contract.resolve_pool_geometry(max_model_len, s_kv_p99, **kw99)
    p99_fit_s = min(s_kv_p99, s_peak)
    identity_s = int(contract.resolve_pool_geometry(max_model_len, None, **base).max_num_seqs)
    return {"attack": attack, "kv_rule": POOL_KV_RULE, "L": int(max_model_len), "max_model_len": int(max_model_len),
            "tensor_parallel": int(tensor_parallel), "S": s_peak, "S_peak": s_peak, "S_kv_mean": s_kv_mean,
            "mean_seq_tokens": int(mean["mean_seq_tokens"]), "mean_prompt_tokens": int(mean["mean_prompt_tokens"]),
            "kv_tokens_at_S": int(contract.kv_tokens_needed(max_model_len, s_peak, mean["mean_seq_tokens"])),
            "model_headroom_gib_at_S": round(g_mean.headroom_bytes / contract.GIB, 2),
            "pinned_children": int(PRODUCTION_PEAK_CHILDREN[attack]), "headroom": float(headroom), "slots_cover_peak": True,
            "p99_fit_S": p99_fit_s, "S_kv_p99": s_kv_p99, "p99_seq_tokens": int(p99["p99_seq_tokens"]),
            "p99_prompt_tokens": int(p99["p99_prompt_tokens"]), "p99_seq_tokens_ceiling": int(p99["p99_seq_tokens_ceiling"]),
            "p99_kv_coverage_model": round(g_mean.kv_bytes / (s_peak * int(p99["p99_seq_tokens"]) * arch.kv_bytes_per_token), 4),
            "kv_slack_seqs": int(KV_FIT_SLACK_SEQS), "identity_S": identity_s, "n_calls": int(p99["n_calls"]),
            "reply_cap": int(p99["reply_cap"]), "measured_on": MEASURED_ATTACKER_CALLS_ON, "watch_metrics": list(WATCH_METRIC_NAMES),
            "fits": bool(g_mean.fits), "unfit_reason": unfit_reason}


def alloc_conf_env(tensor_parallel: int) -> Dict[str, str]:
    """The ``PYTORCH_CUDA_ALLOC_CONF`` block for the pool image: the campaign pool's ``expandable_segments:True`` on
    ONE card, NOTHING above it. Under tensor parallelism vLLM's custom all-reduce exports the CUDA-graph all-reduce
    buffers with ``cudaIpcGetMemHandle`` (``custom_all_reduce.cuh:455``, ``get_graph_buffer_ipc_meta``); torch's
    allocator hands out ``cuMemMap`` ranges under expandable segments and that export fails with 'invalid argument'
    -- measured 2026-09-18 on the four H200:2 Servers, each dead at its first graph capture."""
    if int(tensor_parallel) not in ALLOWED_TP:
        raise PoolParamError(f"tensor-parallel size {tensor_parallel} must be one of {ALLOWED_TP}")
    return {"PYTORCH_CUDA_ALLOC_CONF": EXPANDABLE_SEGMENTS} if int(tensor_parallel) == 1 else {}


def serve_env_problem(env: Mapping[str, str], tensor_parallel: int) -> Optional[str]:
    """``None`` when ``env`` can launch vLLM at ``tensor_parallel`` cards; else the reason. The Server checks its
    container env with this before ``vllm serve`` so a re-introduced ``expandable_segments:True`` on a multi-card
    pool is one loud refusal at enter, not a 13-minute load-compile-warmup-crash loop that never flushes serve.log."""
    conf = str(env.get("PYTORCH_CUDA_ALLOC_CONF", "") or "")
    if int(tensor_parallel) <= 1:
        return None
    for item in conf.split(","):
        k, _, v = item.strip().partition(":")
        if k.strip().lower() == "expandable_segments" and v.strip().lower() == "true":
            return (f"PYTORCH_CUDA_ALLOC_CONF={conf!r} with tensor-parallel {tensor_parallel}: vLLM's custom all-reduce "
                    "exports its CUDA-graph buffers with cudaIpcGetMemHandle, which fails ('invalid argument', "
                    "custom_all_reduce.cuh:455) on expandable-segment allocations -- the W14 pools' crash-loop (F-M); "
                    "a multi-card pool runs without it")
    return None


class PoolParamError(ValueError):
    """A SEEDREP_ATTACKER_* value that cannot deploy a pool: named, typed, never a default substituted."""


class RoutingRefused(RuntimeError):
    """A request body this lane will not send: a target body off the vendor row / unseeded, or an
    unknown model. Never a fallback endpoint."""


class PoolLifetimeShort(RuntimeError):
    """A pool deployed for less time than a driver bound to it can need."""


@dataclasses.dataclass(frozen=True)
class PoolParams:
    pool_index: int
    attack: str
    gpu: str
    gpu_kind: str
    tensor_parallel: int
    max_model_len: int
    chunk_tokens: int
    max_num_seqs: Optional[int]          # None = kv_fit (the largest fitting S)
    gpu_memory_utilization: float
    lifetime_h: float
    w7: str = ""
    wave_uuid8: str = ""

    @property
    def app_name(self) -> str:
        return pool_app_name(self.pool_index)

    @property
    def pool_name(self) -> str:
        return f"{LANE}-{self.pool_index}"

    @property
    def pool_dir(self) -> str:
        return pool_dir(self.pool_index)

    @property
    def cpu(self) -> int:
        return 16 * self.tensor_parallel

    @property
    def memory_mib(self) -> int:
        return min(131_072 * self.tensor_parallel, 262_144)

    def as_env(self) -> Dict[str, str]:
        """The env block baked into the images so the container resolves the SAME params."""
        out = {
            f"{ENV_PREFIX}POOL_INDEX": str(self.pool_index),
            f"{ENV_PREFIX}ATTACK": self.attack,
            f"{ENV_PREFIX}GPU": self.gpu,
            f"{ENV_PREFIX}MAX_MODEL_LEN": str(self.max_model_len),
            f"{ENV_PREFIX}CHUNK_TOKENS": str(self.chunk_tokens),
            f"{ENV_PREFIX}GPU_MEM_UTIL": repr(self.gpu_memory_utilization),
            f"{ENV_PREFIX}LIFETIME_H": repr(self.lifetime_h),
            f"{ENV_PREFIX}W7": self.w7,
            f"{ENV_PREFIX}WAVE_UUID8": self.wave_uuid8,
        }
        if self.max_num_seqs is not None:
            out[f"{ENV_PREFIX}MAX_NUM_SEQS"] = str(self.max_num_seqs)
        return out

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def pool_app_name(pool_index: int) -> str:
    return f"{LANE}-attacker-{int(pool_index)}"


def pool_dir(pool_index: int) -> str:
    return f"{VOL_PATH}/attacker-pool/{LANE}-{int(pool_index)}"


def parse_gpu(spec: str) -> Tuple[str, int]:
    """``H200`` -> (H200, 1); ``H200:2`` -> (H200, 2). The count is the tensor-parallel size."""
    m = _GPU_RE.match(str(spec).strip())
    if not m:
        raise PoolParamError(f"{ENV_PREFIX}GPU={spec!r} is not <kind>[:<count>] with kind in H200/H100/A100-80GB")
    kind, n = m.group(1), int(m.group(2) or 1)
    if n not in ALLOWED_TP:
        raise PoolParamError(f"{ENV_PREFIX}GPU={spec!r}: a tensor-parallel size must be one of {ALLOWED_TP}")
    if kind != "H200":
        raise PoolParamError(f"{ENV_PREFIX}GPU={spec!r}: the memory model is calibrated on H200 (139.81 GiB); "
                             "refusing another card without a re-measured budget")
    return kind, n


def _raw(env: Mapping[str, str], name: str) -> str:
    return str(env.get(ENV_PREFIX + name, "") or "").strip()


def _int(env: Mapping[str, str], name: str, default: Optional[int], *, minimum: int) -> Optional[int]:
    raw = _raw(env, name)
    if not raw:
        return default
    try:
        v = int(raw)
    except ValueError:
        raise PoolParamError(f"{ENV_PREFIX}{name}={raw!r} is not an integer") from None
    if v < minimum:
        raise PoolParamError(f"{ENV_PREFIX}{name}={v} is below the minimum {minimum}")
    return v


def _float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = _raw(env, name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        raise PoolParamError(f"{ENV_PREFIX}{name}={raw!r} is not a number") from None


def _hex(env: Mapping[str, str], name: str, width: int) -> str:
    """An optional provenance field: empty, or exactly ``width`` lowercase hex digits."""
    v = _raw(env, name)
    if v and not re.fullmatch(rf"[0-9a-f]{{{width}}}", v):
        raise PoolParamError(f"{ENV_PREFIX}{name}={v!r} is not {width} hex")
    return v


def pool_params(env: Mapping[str, str]) -> PoolParams:
    """The pool's parameters from the environment: the two REQUIRED names (index, attack), the rest at the
    plan's defaults, every value validated. A wrong value is a :class:`PoolParamError`, never a substitute."""
    k = _int(env, "POOL_INDEX", None, minimum=0)
    if k is None:
        raise PoolParamError(f"{ENV_PREFIX}POOL_INDEX is required (the pool's ordinal k -> app {LANE}-attacker-<k>)")
    attack = _raw(env, "ATTACK")
    if attack not in ATTACKS:
        raise PoolParamError(f"{ENV_PREFIX}ATTACK={attack!r} must be one of {ATTACKS} (the attack whose drivers pin this pool)")
    gpu = str(env.get(ENV_PREFIX + "GPU", "") or DEFAULT_GPU).strip()
    kind, tp = parse_gpu(gpu)
    L = _int(env, "MAX_MODEL_LEN", DEFAULT_MAX_MODEL_LEN, minimum=MIN_MAX_MODEL_LEN)
    if L % 16:
        raise PoolParamError(f"{ENV_PREFIX}MAX_MODEL_LEN={L} is not a multiple of the vLLM block size 16")
    chunk = _int(env, "CHUNK_TOKENS", DEFAULT_CHUNK_TOKENS, minimum=1024)
    if chunk > L:
        raise PoolParamError(f"{ENV_PREFIX}CHUNK_TOKENS={chunk} exceeds the window {L}")
    S = _int(env, "MAX_NUM_SEQS", None, minimum=1)
    util = _float(env, "GPU_MEM_UTIL", DEFAULT_GPU_MEM_UTIL)
    if not (0.5 <= util <= 0.98):
        raise PoolParamError(f"{ENV_PREFIX}GPU_MEM_UTIL={util} is outside [0.5, 0.98]")
    life = _float(env, "LIFETIME_H", DEFAULT_LIFETIME_H)
    if life <= 0:
        raise PoolParamError(f"{ENV_PREFIX}LIFETIME_H={life} must be positive")
    w7 = _hex(env, "W7", 7)
    wu = _hex(env, "WAVE_UUID8", 8)
    return PoolParams(pool_index=k, attack=attack, gpu=f"{kind}:{tp}" if tp > 1 else kind, gpu_kind=kind,
                      tensor_parallel=tp, max_model_len=L, chunk_tokens=chunk, max_num_seqs=S,
                      gpu_memory_utilization=util, lifetime_h=life, w7=w7, wave_uuid8=wu)


# --------------------------------------------------------------------------- sizing (pure, tested)

def per_card_arch(arch: contract.AttackerArch, tp: int) -> contract.AttackerArch:
    """The attacker's per-card footprint under tensor parallelism ``tp``: weights split ``tp`` ways; the
    KV heads and the gated-deltanet key/value heads (the recurrent state) split ``tp`` ways, a KV head
    replicated when ``tp`` exceeds the head count (vLLM's behaviour). The logits row stays whole (an
    over-estimate; safe for sizing). ``tp == 1`` returns the arch unchanged."""
    if tp not in ALLOWED_TP:
        raise PoolParamError(f"tensor-parallel size {tp} must be one of {ALLOWED_TP}")
    if tp == 1:
        return arch
    for name in ("linear_num_key_heads", "linear_num_value_heads"):
        if getattr(arch, name) % tp:
            raise PoolParamError(f"{arch.name}: {name}={getattr(arch, name)} is not divisible by tp {tp}")
    return dataclasses.replace(
        arch,
        name=f"{arch.name} [1/{tp} of TP {tp}]",
        weights_bytes=math.ceil(arch.weights_bytes / tp),
        num_key_value_heads=max(1, arch.num_key_value_heads // tp),
        linear_num_key_heads=arch.linear_num_key_heads // tp,
        linear_num_value_heads=arch.linear_num_value_heads // tp,
    )


def pool_reserve_bytes(tensor_parallel: int) -> int:
    """The fixed non-KV term per card: the contract's one-card reserve, plus :data:`TP_RESERVE_BYTES` when the
    engine spans several cards."""
    if int(tensor_parallel) not in ALLOWED_TP:
        raise PoolParamError(f"tensor-parallel size {tensor_parallel} must be one of {ALLOWED_TP}")
    return contract.RESERVE_BYTES + (TP_RESERVE_BYTES if int(tensor_parallel) > 1 else 0)


def launched_sizing(attack: str, geom: contract.PoolGeometry, *, explicit: bool, util: float = DEFAULT_GPU_MEM_UTIL,
                    chunk_tokens: int = DEFAULT_CHUNK_TOKENS, tensor_parallel: int = 1) -> Dict[str, Any]:
    """The endpoint record's ``sizing`` block for the geometry the Server LAUNCHED: every launched fact --
    ``S``, its KV requirement, whether the slots cover the peak, where the KV is resident -- is derived from ``geom.max_num_seqs``,
    never from the automatic default; the automatic derivation rides beside it under ``recommended`` by name. An operator who
    lowers ``SEEDREP_ATTACKER_MAX_NUM_SEQS`` after a live coverage refusal therefore gets a record that says what runs (and the
    certificate's reader refuses a record whose launched width disagrees with its ``sizing.S``). Slot coverage and KV residency
    are two different facts and are recorded apart: ``slots_cover_peak`` (S >= the pinned children: the scheduler can admit the
    peak), ``kv_resident_at`` (the KV holds S sequences at the mean / at the p99: whether they stay RESIDENT at those lengths)."""
    # the recommendation is metadata here (strict only for the automatic launch, whose S it IS): an explicit width that fits
    # must produce its record even when the automatic width would not fit this card
    rec = derive_pool_sizing(attack, geom.max_model_len, util=util, chunk_tokens=chunk_tokens, tensor_parallel=tensor_parallel,
                             strict=not explicit)
    S, L = int(geom.max_num_seqs), int(geom.max_model_len)
    pinned = int(rec["pinned_children"])
    mean, p99 = int(rec["mean_seq_tokens"]), int(rec["p99_seq_tokens"])
    return {
        "kv_rule": geom.kv_rule, "L": L, "S": S, "seqs_source": geom.seqs_source, "explicit_override": bool(explicit),
        "mean_seq_tokens": mean, "p99_seq_tokens": p99, "kv_seq_tokens": geom.kv_seq_tokens,
        "kv_tokens_at_S": int(contract.kv_tokens_needed(L, S, geom.kv_seq_tokens)),
        "pinned_children": pinned, "S_peak": int(rec["S_peak"]), "S_kv_mean": int(rec["S_kv_mean"]), "p99_fit_S": int(rec["p99_fit_S"]),
        "slots_cover_peak": S >= pinned,
        "kv_resident_at": {"mean": {"seq_tokens": mean, "fits": S <= int(rec["S_kv_mean"]), "S_fit": int(rec["S_kv_mean"])},
                           "p99": {"seq_tokens": p99, "fits": S <= int(rec["S_kv_p99"]), "S_fit": int(rec["S_kv_p99"])}},
        "recommended": rec,
        "caveat": PEAK_MEAN_CAVEAT,
    }


def pool_geometry(p: PoolParams) -> contract.PoolGeometry:
    """S_att × L for the pool through the SHARED resolver on the per-card arch with the per-card reserve, under the
    ``peak_mean`` rule: no S given -> the production peak (:func:`derive_pool_sizing`, ``seqs_source``
    ``peak_mean``), refused by name when that peak does not fit; an explicit S is validated on ITS OWN -- refused BY NAME when
    ``S x mean sequence`` does not fit the card (the resolver names the rule and the largest S that would), never because the
    unused automatic width would not (the recovery after a coverage-gate refusal is a lower utilization and a
    lower explicit width). The geometry's ``note`` carries the arithmetic with the p99 fit beside it, says when the automatic
    recommendation does not fit; above one card it also states the tensor-parallel reserve and the live measurement behind it."""
    arch = per_card_arch(contract.QWEN36_27B, p.tensor_parallel)
    reserve = pool_reserve_bytes(p.tensor_parallel)
    explicit = p.max_num_seqs is not None
    sizing = derive_pool_sizing(p.attack, p.max_model_len, util=p.gpu_memory_utilization, chunk_tokens=p.chunk_tokens,
                                tensor_parallel=p.tensor_parallel, strict=not explicit)
    S = p.max_num_seqs if explicit else sizing["S"]
    geom = contract.resolve_pool_geometry(
        p.max_model_len, S, util=p.gpu_memory_utilization,
        chunked_prefill=True, chunk_tokens=p.chunk_tokens, arch=arch,
        reserve_bytes=reserve, kv_seq_tokens=sizing["mean_seq_tokens"], kv_rule=POOL_KV_RULE)
    source = "explicit" if explicit else POOL_KV_RULE
    note = (f"D116 sizing for {p.attack} ({POOL_KV_RULE}): S {S} ({source}; S_peak {sizing['S_peak']} = {sizing['pinned_children']} pinned "
            f"children x {sizing['headroom']}); KV checked at the mean sequence {sizing['mean_seq_tokens']:,} (mean prompt "
            f"{sizing['mean_prompt_tokens']:,} + {sizing['reply_cap']:,} reply cap over {sizing['n_calls']:,} measured calls; S_kv_mean "
            f"{sizing['S_kv_mean']}); beside it the p99 fit: p99 sequence {sizing['p99_seq_tokens']:,} (bucket ceiling "
            f"{sizing['p99_seq_tokens_ceiling']:,}) -> p99_fit_S {sizing['p99_fit_S']} (S_kv_p99 {sizing['S_kv_p99']}), the model's KV "
            f"holds {sizing['p99_kv_coverage_model']:.0%} of S x p99 -- the tail queues on vLLM's scheduler ({', '.join(WATCH_METRIC_NAMES)}); "
            f"identity would give {sizing['identity_S']}"
            + (f"; NOTE {sizing['unfit_reason']} -- launched at the explicit S {S} instead" if not sizing["fits"] else ""))
    if p.tensor_parallel > 1:
        measured = LIVE_ROOM_MEASURED.get((p.gpu_kind, p.tensor_parallel, p.max_model_len))
        note += (f"; tp {p.tensor_parallel}: reserve = one-card {contract.RESERVE_BYTES / contract.GIB:.1f} GiB + "
                 f"{TP_RESERVE_BYTES / contract.GIB:.1f} GiB per-card tensor-parallel communicators/graphs"
                 + (f"; live room measured {measured}x at this (card, tp, window)"
                    if measured is not None else "; live room unmeasured at this (card, tp, window) -- the coverage check decides"))
    return dataclasses.replace(geom, seqs_source=source, note=note)


def vllm_argv(p: PoolParams, geom: contract.PoolGeometry, opts, *, model_path: str = MODEL_PATH,
              served_name: str = ATTACKER_MODEL, port: int = PORT) -> list:
    """The ``vllm serve`` argv: the original's flags + ``--tensor-parallel-size`` when the GPU count is > 1."""
    argv = [
        "vllm", "serve", model_path,
        "--served-model-name", served_name,
        "--host", "0.0.0.0", "--port", str(port),
        "--gpu-memory-utilization", str(p.gpu_memory_utilization),
        "--max-model-len", str(geom.max_model_len),
        "--max-num-seqs", str(geom.max_num_seqs),
        "--max-num-batched-tokens", str(geom.max_num_batched_tokens),
        "--enable-chunked-prefill",
        *opts.vllm_argv(model_role="attacker"),
        "--limit-mm-per-prompt", LIMIT_MM_PER_PROMPT,
    ]
    if p.tensor_parallel > 1:
        argv += ["--tensor-parallel-size", str(p.tensor_parallel)]
    return argv


# --------------------------------------------------------------------------- pool count + pinning (pure, tested)

def pools_needed(drivers: int, drivers_per_pool_before_waiting: int) -> int:
    """``ceil(drivers / drivers_per_pool_before_waiting)``, at least 1 for any positive driver count."""
    if drivers < 0 or drivers_per_pool_before_waiting < 1:
        raise ValueError(f"drivers={drivers} drivers_per_pool={drivers_per_pool_before_waiting}")
    return max(1, math.ceil(drivers / drivers_per_pool_before_waiting)) if drivers else 0


def size_pool_plan(drivers_by_attack: Mapping[str, int], drivers_per_pool_before_waiting: int,
                   minimum: Sequence[Tuple[str, int]] = DEFAULT_POOL_PLAN) -> Tuple[Tuple[str, int], ...]:
    """Per attack, ``max(the minimum plan's count, pools_needed(drivers))`` — the plan's 4-pool default is a
    FLOOR (never fewer than one per attack), S4's measured drivers-per-pool raises it. An attack outside
    :data:`ATTACKS` is refused."""
    mins = dict(minimum)
    for a in list(drivers_by_attack) + list(mins):
        if a not in ATTACKS:
            raise ValueError(f"unknown attack {a!r}; the multi-turn attacks are {ATTACKS}")
    out = []
    for a in ATTACKS:
        need = pools_needed(int(drivers_by_attack.get(a, 0)), drivers_per_pool_before_waiting)
        out.append((a, max(int(mins.get(a, 1)), need, 1)))
    return tuple(out)


def pool_roster(plan: Sequence[Tuple[str, int]] = DEFAULT_POOL_PLAN) -> Tuple[Tuple[int, str], ...]:
    """``[(k, attack), ...]`` — pool k's attack, in plan order (k = 0 .. n-1). Every attack in
    :data:`ATTACKS` must appear at least once (drivers pin per attack and never fail over)."""
    roster = []
    for attack, n in plan:
        if attack not in ATTACKS:
            raise ValueError(f"unknown attack {attack!r}")
        if int(n) < 1:
            raise ValueError(f"attack {attack!r} has {n} pools; every attack needs at least one")
        roster += [attack] * int(n)
    missing = [a for a in ATTACKS if a not in roster]
    if missing:
        raise ValueError(f"the pool plan has no pool for {missing}; every attack needs at least one")
    return tuple(enumerate(roster))


def pool_for_driver(roster: Sequence[Tuple[int, str]], attack: str, driver_ordinal: int) -> int:
    """The pool a driver PINS at launch: round-robin over the attack's pools by the driver's ordinal in the
    launch plan (deterministic, so the plan and the pin record agree)."""
    ks = [k for k, a in roster if a == attack]
    if not ks:
        raise ValueError(f"no pool serves attack {attack!r} in roster {list(roster)}")
    if driver_ordinal < 0:
        raise ValueError(f"driver_ordinal {driver_ordinal} < 0")
    return ks[driver_ordinal % len(ks)]


def pool_outlives(lifetime_h: float, longest_driver_h: float, queue_wait_h: float = 0.0) -> float:
    """The margin (hours) by which the pool's deployed lifetime exceeds the longest target container's
    walltime PLUS its Modal queue wait; a non-positive margin is :class:`PoolLifetimeShort` (a driver pins
    at launch and cannot move; a pool that dies first turns every remaining goal into an error outcome)."""
    if longest_driver_h <= 0 or queue_wait_h < 0:
        raise ValueError(f"longest_driver_h={longest_driver_h} queue_wait_h={queue_wait_h}")
    margin = float(lifetime_h) - (float(longest_driver_h) + float(queue_wait_h))
    if margin <= 0:
        raise PoolLifetimeShort(
            f"pool lifetime {lifetime_h} h does not outlive the longest target container "
            f"({longest_driver_h} h) plus its queue wait ({queue_wait_h} h): margin {margin:.2f} h")
    return margin


# --------------------------------------------------------------------------- request routing (pure, tested)

SAMPLING_KEYS: Tuple[str, ...] = ("temperature", "top_p", "top_k", "min_p", "repetition_penalty", "presence_penalty")


def target_row(policy_file, model_key: str, eval_category: str = "multi_turn"):
    """The TARGET decode row by EXACT ref — ``<model_id>|<eval_category>|off|vendor_preset@v1`` through
    ``PolicyFile.by_ref`` with ``superseded_by_policy_id is None`` asserted (never
    ``resolve(mode=...)``, which ignores eval_id and returns the IFEval row; never a default lookup). A
    missing row is the registry's ``NoPolicy``, propagated — the lane never invents a row."""
    from sisteer.registry.decode_policy import PARAM_KEYS
    ref = f"{policy_file.model_id(model_key)}|{eval_category}|off|vendor_preset@v1"
    pol = policy_file.by_ref(ref)
    if pol.superseded_by_policy_id is not None:
        raise RoutingRefused(f"decode row {ref} is superseded by {pol.superseded_by_policy_id}; refusing to route on it")
    params = pol.request_params()
    missing = [k for k in PARAM_KEYS if k not in params]
    if missing:
        raise RoutingRefused(f"decode row {ref} does not SEND {missing}; the lane's target contract sends all six")
    return pol


@dataclasses.dataclass(frozen=True)
class RoutingTable:
    """What a driver container knows: the pool it pinned (attacker side) and the target proxy (target
    side), with the target's vendor row as the six values every target body must carry."""
    attacker_models: frozenset
    attacker_endpoint: str
    target_vmodels: frozenset
    target_endpoint: str
    target_params: Mapping[str, Any]      # the six sampling keys -> values, from :func:`target_row`
    policy_ref: str = ""


def classify_request(body: Mapping[str, Any], *, attacker_models: Iterable[str], target_vmodels: Iterable[str]) -> str:
    model = str(body.get("model") or "")
    if not model:
        raise RoutingRefused("request body names no model")
    is_attacker, is_target = model in set(attacker_models), model in set(target_vmodels)
    if is_attacker and is_target:
        raise RoutingRefused(f"model {model!r} is both an attacker model and a target vmodel; the table is ambiguous")
    if is_attacker:
        return "attacker"
    if is_target:
        return "target"
    raise RoutingRefused(f"model {model!r} is neither the pinned attacker nor a served target vmodel")


def _num_eq(a: Any, b: Any) -> bool:
    try:
        return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-9)
    except (TypeError, ValueError):
        return a == b


def check_target_body(body: Mapping[str, Any], target_params: Mapping[str, Any]) -> Dict[str, Any]:
    """A TARGET body must carry EXACTLY the row's six sampling values and an integer ``seed``. Returns the
    six values as sent; refuses with every departure named — a body carrying the attacker's parameters
    (e.g. temperature 1.0 / top_p 1.0 / no presence penalty) is the known-bad case."""
    problems = []
    for k in SAMPLING_KEYS:
        want = target_params.get(k)
        if want is None:
            problems.append(f"{k}: the row does not send it")
            continue
        if k not in body:
            problems.append(f"{k}: absent from the body (the row sends {want!r}; a server default is not a sent key)")
        elif not _num_eq(body[k], want):
            problems.append(f"{k}: body sends {body[k]!r}, the row says {want!r}")
    seed = body.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        problems.append(f"seed: absent or not an int ({seed!r}); every target draw is seeded")
    if problems:
        raise RoutingRefused("target body off the vendor row: " + "; ".join(problems))
    return {k: body[k] for k in SAMPLING_KEYS}


def route_request(body: Mapping[str, Any], table: RoutingTable) -> Tuple[str, str]:
    """``(role, endpoint)`` for one request body, or :class:`RoutingRefused`. Attacker bodies are routed
    as they are (the attack's own parameters); target bodies pass
    :func:`check_target_body` first."""
    role = classify_request(body, attacker_models=table.attacker_models, target_vmodels=table.target_vmodels)
    if role == "target":
        check_target_body(body, table.target_params)
        return role, table.target_endpoint
    return role, table.attacker_endpoint


# --------------------------------------------------------------------------- health (pure given a GET)

def endpoint_health(record: Mapping[str, Any], key: str, *, get: Callable[[str, Optional[str]], int]) -> Dict[str, Any]:
    """The pre-spawn gate on ONE pool record: ``/health`` 200; ``/v1/models`` refuses a bare call (401/403)
    and admits the bearer (200). ``get(url, bearer_or_None) -> status``. Any other reading is a
    refusal that names the probe."""
    url = str(record.get("url") or "").rstrip("/")
    if not url:
        raise RuntimeError("endpoint record has no url")
    health = get(f"{url}/health", None)
    bare = get(f"{url}/v1/models", None)
    auth = get(f"{url}/v1/models", key)
    problems = []
    if health != 200:
        problems.append(f"/health={health}")
    if bare not in (401, 403):
        problems.append(f"unauthenticated /v1/models={bare} (the bearer guard is void)" if bare == 200
                        else f"unauthenticated /v1/models={bare}")
    if auth != 200:
        problems.append(f"authenticated /v1/models={auth}")
    if problems:
        raise RuntimeError(f"pool {record.get('pool_name')} at {url} is not healthy: " + ", ".join(problems))
    return {"url": url, "health": health, "models_unauthenticated": bare, "models_authenticated": auth,
            "checked_at": _utc_now(), "pool_name": record.get("pool_name"),
            "pool_index": record.get("pool_index"), "attack": record.get("attack")}


# --------------------------------------------------------------------------- module bindings (import time)

def _params_or_exit() -> PoolParams:
    try:
        return pool_params(os.environ)
    except PoolParamError as exc:
        raise SystemExit(f"[seedrep-pool] {exc}") from None


P = _params_or_exit()
GEOM = pool_geometry(P)
SIZING = derive_pool_sizing(P.attack, P.max_model_len, util=P.gpu_memory_utilization, chunk_tokens=P.chunk_tokens,
                            tensor_parallel=P.tensor_parallel, strict=P.max_num_seqs is None)
APP_NAME = P.app_name
POOL_NAME = P.pool_name
POOL_DIR = P.pool_dir
SERVE_LOG = f"{POOL_DIR}/serve.log"
ENDPOINT_PATH = f"{POOL_DIR}/endpoint.json"
SERVING_MANIFEST = f"{POOL_DIR}/serving.json"
MAX_MODEL_LEN = GEOM.max_model_len
MAX_NUM_SEQS = GEOM.max_num_seqs
MAX_NUM_BATCHED_TOKENS = GEOM.max_num_batched_tokens
VOL = modal.Volume.from_name(VOL_NAME, create_if_missing=False)


def _repo_sha() -> str:
    if not modal.is_local():
        return os.environ.get("ATTACKER_POOL_REPO_SHA", "unknown")
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001 - informational provenance only
        return os.environ.get("ATTACKER_POOL_REPO_SHA", "unknown")


SOURCE_SHA = _repo_sha()


#: the tmpfs runtime dir (RAM-only, 0700, gone at reboot) -- the ONLY place the self-minted bearer is rendered
RUNTIME_DIR_ENV = "XDG_RUNTIME_DIR"


def _local_state_dir() -> Path:
    # RECORDS only (endpoint_<k>.json, health_<k>.json) -- no credential lives under the state dir. Grading host only:
    # the site's SISTEER_ATTACKER_POOL_STATE_DIR, else $STATE/sisteer/attacker-pool-modal, else ~/.local/state/...
    from sisteer import site

    return site.state_path("SISTEER_ATTACKER_POOL_STATE_DIR", "sisteer", "attacker-pool-modal") / LANE


def _legacy_key_path() -> Path:
    """Where the bearer used to sit (a plaintext state file). Never read, never migrated: its presence REFUSES the
    mint until ``ROTATE_KEY=1`` discards it -- a fresh wave mints a fresh bearer."""
    return _local_state_dir() / "api_key"


def _runtime_key_dir() -> Path:
    """The bearer's directory under ``$XDG_RUNTIME_DIR`` -- ONE lane key shared by every seedrep pool, so a driver container
    carries one VLLM_API_KEY. The runtime dir unset is a refusal, never a fallback to disk: a tool that insists on a file
    renders it under the tmpfs runtime dir and nowhere else."""
    rd = os.environ.get(RUNTIME_DIR_ENV)
    if not rd:
        raise SystemExit(f"[seedrep-pool] {RUNTIME_DIR_ENV} is unset -- the lane bearer is rendered ONLY under the tmpfs runtime dir (D77); "
                         "refusing to fall back to ~/.local/state")
    return Path(rd) / "seedrep-battery" / "attacker-pool" / LANE


def _local_key_path() -> Path:
    return _runtime_key_dir() / "api_key"


def _local_endpoint_path(k: int = P.pool_index) -> Path:
    return _local_state_dir() / f"endpoint_{int(k)}.json"


def _local_health_path(k: int = P.pool_index) -> Path:
    return _local_state_dir() / f"health_{int(k)}.json"


def _private_dir(d: Path) -> None:
    """``d`` and every component below ``$XDG_RUNTIME_DIR`` created 0700 (``mkdir(parents=True)`` would give the parents the
    umask's default); an existing component that is group/world-accessible is a refusal, not a chmod."""
    root = Path(os.environ[RUNTIME_DIR_ENV])
    cur = root
    for part in d.relative_to(root).parts:
        cur = cur / part
        if not cur.exists():
            os.mkdir(cur, 0o700)
            os.chmod(cur, 0o700)
        mode = cur.stat().st_mode & 0o777
        if mode & 0o077:
            raise RuntimeError(f"refusing bearer directory {cur}: mode {oct(mode)} is not private")


def _load_local_api_key() -> str:
    """Read or mint the grading host's bearer (0600 under a 0700 dir in ``$XDG_RUNTIME_DIR``; self-minted: a transport guard,
    not a vault credential); never print it or put it on argv. A legacy bearer file under ``~/.local/state`` is not
    migrated: it refuses the mint by name until ``SEEDREP_ATTACKER_ROTATE_KEY=1`` discards it and mints afresh in tmpfs."""
    if not modal.is_local():
        return os.environ.get(API_KEY_ENV, "")
    rotate = os.environ.get(f"{ENV_PREFIX}ROTATE_KEY") == "1"
    legacy = _legacy_key_path()
    if legacy.exists():
        if not rotate:
            raise SystemExit(f"[seedrep-pool] a bearer file from before D77 sits at {legacy}; it is NOT migrated -- re-run with "
                             f"{ENV_PREFIX}ROTATE_KEY=1 to discard it and mint a fresh bearer under ${RUNTIME_DIR_ENV} "
                             "(a fresh wave mints a fresh bearer)")
        legacy.unlink()
    p = _local_key_path()                      # refuses when the runtime dir is unset
    if rotate and p.exists():
        p.unlink()
    if not p.exists():
        _private_dir(p.parent)
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(fd, contract.mint_api_key().encode())
        finally:
            os.close(fd)
        os.chmod(p, 0o600)
    mode = p.stat().st_mode & 0o777
    if mode & 0o077:
        raise RuntimeError(f"refusing bearer file {p}: mode {oct(mode)} is not private")
    key = p.read_text().strip()
    if not key:
        raise RuntimeError(f"refusing empty bearer file {p}")
    return key


def read_local_api_key() -> str:
    """READ-ONLY twin of :func:`_load_local_api_key` for a consumer that must never mint (the driver app's spawn gate):
    the bearer under ``$XDG_RUNTIME_DIR/seedrep-battery/attacker-pool/<lane>/api_key`` when it exists, is 0600-private
    and non-empty; ``""`` when the runtime dir is unset or the file is absent (the CALLER refuses by name when it needs one).
    A non-private or empty file is a RuntimeError, as in the mint path. ``seedrep_served.read_local_pool_bearer`` re-implements
    this over the same spelling (it cannot import this module: pool params + the mint bind at import) -- a test pins the path."""
    rd = os.environ.get(RUNTIME_DIR_ENV)
    if not rd:
        return ""
    p = _local_key_path()
    if not p.is_file():
        return ""
    mode = p.stat().st_mode & 0o777
    if mode & 0o077:
        raise RuntimeError(f"refusing bearer file {p}: mode {oct(mode)} is not private")
    key = p.read_text().strip()
    if not key:
        raise RuntimeError(f"refusing empty bearer file {p}")
    return key


pool_secret = modal.Secret.from_dict({API_KEY_ENV: _load_local_api_key()})

_IMAGE_ENV = {
    "PYTHONPATH": "/root",
    "HF_HOME": f"{VOL_PATH}/hf",
    "TOKENIZERS_PARALLELISM": "false",
    # expandable_segments:True on ONE card only: the custom all-reduce's graph-buffer IPC export cannot take it (F-M)
    **alloc_conf_env(P.tensor_parallel),
    "ATTACKER_POOL_REPO_SHA": SOURCE_SHA,
    **P.as_env(),
}

# Same proven cu129/vLLM stack as attacker_pool.py / grokmt_served.py (pinned). The source tree is copied so
# the child vLLM process can import the shared contract helpers from /root/sisteer.
pool_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "build-essential")
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .pip_install(
        "https://github.com/vllm-project/vllm/releases/download/v0.25.1/"
        "vllm-0.25.1+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
        "#sha256=9e206f370c934a2d4b6b1f05d3d09708d344e05d80260189ef19f60755709431",
        "torch==2.11.0+cu129",
        "torchvision==0.26.0+cu129",
        "torchaudio==2.11.0+cu129",
        "requests==2.34.2",
        "ninja==1.13.0",
        extra_index_url="https://download.pytorch.org/whl/cu129",
    )
    .pip_install("transformers==5.14.1", "tokenizers==0.22.2", extra_options="--no-deps --force-reinstall")
    .add_local_dir(SOURCE_DIR, "/root/sisteer", copy=True)
    .env({**_IMAGE_ENV, "HF_HUB_OFFLINE": "1"})
)

client_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("requests==2.34.2", "huggingface-hub==1.16.1")
    .add_local_dir(SOURCE_DIR, "/root/sisteer", copy=True)
    .env(_IMAGE_ENV)
)

# Keep the production Server in its own app. Running stage/endpoint/canary through ``tools`` must never
# instantiate a second min_containers=1 server.
app = modal.App(APP_NAME)
tools = modal.App(f"{APP_NAME}-tools")


def _http_status(url: str, *, key: str | None = None, timeout: float = 10.0) -> int:
    import urllib.error
    import urllib.request
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"} if key else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return int(r.status)
    except urllib.error.HTTPError as e:
        return int(e.code)
    except Exception:  # noqa: BLE001 - caller turns 0 into a refusal
        return 0


def _wait_ready(proc: subprocess.Popen, log_path: Path, timeout_s: float = 3600) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            tail = log_path.read_text(errors="replace")[-50_000:] if log_path.exists() else ""
            VOL.commit()
            raise RuntimeError(f"vLLM exited rc={proc.returncode} before readiness:\n{tail}")
        if _http_status(f"http://127.0.0.1:{PORT}/health") == 200:
            return
        time.sleep(5)
    raise RuntimeError(f"vLLM did not become healthy within {timeout_s:.0f}s")


def _attest_coverage(log_text: str) -> dict:
    """vLLM's coverage fact for the launched geometry under its KV rule (``peak_mean``: S x mean sequence tokens and one full
    window; S x L under the identity), or refuse (the memory model is only an estimate; the log line is the
    fact). Records BESIDE the verdict the rule, the sequence it was checked at, and the p99 coverage FRACTION (live KV / S x p99:
    how much of a worst-case peak the card holds before the scheduler queues -- read next to ``watch_metrics``)."""
    cov = contract.kv_coverage_from_log(log_text)
    problem = contract.coverage_problem(cov, max_num_seqs=MAX_NUM_SEQS, max_model_len=MAX_MODEL_LEN,
                                        kv_seq_tokens=GEOM.kv_seq_tokens, kv_rule=GEOM.kv_rule)
    if problem:
        raise RuntimeError(f"chunked pool coverage is not certified: {problem}. Refusing to advertise")
    cov = dict(cov, kv_rule=GEOM.kv_rule, kv_seq_tokens=GEOM.kv_seq_tokens,
               p99_seq_tokens=SIZING["p99_seq_tokens"],
               p99_coverage_fraction=contract.coverage_fraction(cov, max_num_seqs=MAX_NUM_SEQS, max_model_len=MAX_MODEL_LEN,
                                                                seq_tokens=SIZING["p99_seq_tokens"]))
    return cov


def _attest_serving(log_text: str, opts, *, manifest_path: str | Path, log_path: str, endpoint: str,
                    model: str = MODEL_PATH, served_name: str = ATTACKER_MODEL, extra: dict | None = None) -> tuple[dict, dict]:
    """Verify the live engine against the REQUESTED serving options; the advertised identity is DERIVED from
    the engine line, never these literals (a contradiction terminates the pool)."""
    from sisteer.eval.attacks import attacker_serving as serving

    engine = serving.parse_engine_log(log_text)
    problems = serving.verify_engine(opts, engine)
    if problems:
        raise RuntimeError("attacker pool engine contradicts the requested serving options: "
                           + "; ".join(problems) + ". Refusing to advertise")
    manifest = serving.write_manifest(
        manifest_path, opts=opts, sources={"quantization": "pool_constant", "prefix_caching": "pool_constant"},
        engine=engine, model=model, served_name=served_name, endpoint=endpoint, log_path=log_path,
        argv=opts.vllm_argv(model_role="attacker"), extra=extra)
    attested = serving.attested_serving_record(manifest, manifest_path=str(manifest_path))
    fields = {
        "attacker_quantization": attested["attacker_quantization"],
        "prefix_caching": attested["prefix_caching"],
        serving.SERVING_RECORD_KEY: attested,
    }
    return manifest, fields


#: How long the grading host's tools (`--action endpoint` / health / canary) wait for the Server to publish ``endpoint.json``. The Server's
#: own readiness bound is ``_wait_ready(timeout_s=3600)``; an earlier attempt saw all four Servers still in vLLM model inspection /
#: loading of the 27B attacker ~14 min after deploy while the tools gave up at 600 s ("endpoint.json was not published within 600s") --
#: a waiter deadline that does not exceed the work. The tools' Modal timeout must exceed this wait.
#: Re-sized for ONE card from the measured H200:2 startup (pool 0 serve.log, 2026-09-19: API server up 06:46:48, weights
#: 97.9 s from the 9P volume, torch.compile 329 s, engine init 741 s incl. profile + KV + warmup, ready 07:01:51 = 903 s): one card
#: loads 2x the weights (~200 s) and captures the same graphs -> ~1,000-1,100 s estimated, x2 margin = 2,400 s (still above the earlier
#: floor of 1,800 s; the Server's own 3,600 s readiness bound is unchanged).
ONE_CARD_STARTUP_ESTIMATE_S = 1_100
ENDPOINT_PUBLISH_WAIT_S = 2400          # = 2 x ONE_CARD_STARTUP_ESTIMATE_S + 200 (a literal: the test reads it by AST)


def _wait_endpoint_file(timeout_s: float = ENDPOINT_PUBLISH_WAIT_S) -> dict:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            VOL.reload()
            record = json.loads(Path(ENDPOINT_PATH).read_text())
            contract.PoolEndpoint.from_dict(record)
            return record
        except (OSError, json.JSONDecodeError, contract.AttackerPoolError) as exc:
            last_error = exc
            time.sleep(5)
    detail = f": {type(last_error).__name__}: {last_error}" if last_error else ""
    raise RuntimeError(f"endpoint.json was not published within {timeout_s:.0f}s{detail}")


def _assert_separate_container(endpoint: dict, canary_container_id: str) -> str:
    live = endpoint.get("liveness")
    pool_container = live.get("container_id") if isinstance(live, dict) else None
    if not canary_container_id:
        raise RuntimeError("canary has no MODAL_TASK_ID; container separation cannot be proved")
    if not pool_container:
        raise RuntimeError("endpoint.json has no pool container id; separation cannot be proved")
    if str(pool_container) == str(canary_container_id):
        raise RuntimeError(f"canary container {canary_container_id} is the pool container; refusing same-container green")
    return str(pool_container)


def _server_identity() -> tuple[str, str]:
    sid, url = "", ""
    try:
        sid = str(SeedrepAttackerPoolServer.object_id or "")
        url = str(SeedrepAttackerPoolServer.get_url() or "")
    except Exception as exc:  # noqa: BLE001 - fall through to the lookup
        print(f"[pool] in-container server handle unavailable: {type(exc).__name__}: {exc}", flush=True)
    if not (sid and url):
        try:
            srv = modal.Server.from_name(APP_NAME, SERVER_NAME)
            srv.hydrate()
            sid = sid or str(srv.object_id or "")
            url = url or str(srv.get_url() or "")
        except Exception as exc:  # noqa: BLE001 - the caller refuses on empty
            print(f"[pool] deployed-name server lookup failed: {type(exc).__name__}: {exc}", flush=True)
    return sid, url.rstrip("/")


def _app_identity() -> str:
    """The deployed app's id, resolved BY THE EXPECTED NAME: a pool deployed under any other ``--name`` is
    unreachable to the drivers that pin ``seedrep-attacker-<k>`` and must not advertise."""
    aid = ""
    try:
        aid = str(modal.App.lookup(APP_NAME).app_id or "")
    except Exception as exc:  # noqa: BLE001 - the caller refuses on empty
        print(f"[pool] deployed-name app lookup for {APP_NAME} failed: {type(exc).__name__}: {exc}", flush=True)
        return ""
    own = ""
    try:
        own = str(app.app_id or "")
    except Exception:  # noqa: BLE001
        own = ""
    if own and aid and own != aid:
        print(f"[pool] this container's app id {own} is not {APP_NAME}'s {aid}: deployed under another name", flush=True)
        return ""
    return aid


def _call_id() -> str | None:
    try:
        cid = modal.current_function_call_id()
    except Exception:  # noqa: BLE001
        return None
    return str(cid) if cid else None


def _write_json_atomic(out: Path, rec: dict) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, indent=2))
    os.replace(tmp, out)
    return out


def _write_endpoint(record: dict) -> None:
    contract.PoolEndpoint.from_dict(record)   # validate the complete record before the atomic publish
    _write_json_atomic(Path(ENDPOINT_PATH), record)
    VOL.commit()


def endpoint_record_fields(p: PoolParams, geom: contract.PoolGeometry, *, coverage: dict, serving_fields: dict,
                           url: str, liveness: dict, started: str, call_id: str | None, server_id: str,
                           container_id: str, source_sha: str) -> dict:
    """The endpoint.json body (pure, tested): the contract's required keys + the lane's parameters."""
    return {
        "schema_version": contract.SCHEMA_VERSION,
        "source": "endpoint.json",
        "lane": LANE,
        "pool_name": p.pool_name,
        "pool_index": p.pool_index,
        "attack": p.attack,
        "pool_dir": p.pool_dir,
        "url": url,
        "model": ATTACKER_MODEL,
        "snapshot": ATTACKER_SNAPSHOT,
        "model_path": MODEL_PATH,
        "gpu": p.gpu,
        "tensor_parallel_size": p.tensor_parallel,
        "max_model_len": geom.max_model_len,
        "max_num_seqs": geom.max_num_seqs,
        "max_num_batched_tokens": geom.max_num_batched_tokens,
        "gpu_memory_utilization": p.gpu_memory_utilization,
        "cpu_offload_gb": 0,
        "enforce_eager": False,
        "limit_mm_per_prompt": LIMIT_MM_PER_PROMPT,
        **serving_fields,
        "api_key_env_name": API_KEY_ENV,
        "api_key_file": None,
        "job_id": call_id or server_id,
        "node": container_id,
        "started_at": started,
        "expected_lifetime_h": p.lifetime_h,
        "serve_log": SERVE_LOG,
        "repo_sha": source_sha,
        "runtime": "modal_server",
        "liveness": liveness,
        "geometry": geom.as_dict(),
        # The LAUNCHED geometry's sizing -- kv_rule, S (== max_num_seqs above, the certificate reader
        # refuses otherwise), S_peak, S_kv_mean, p99_fit_S, mean / p99 sequence tokens, L, slots_cover_peak, kv_resident_at -- with
        # the automatic derivation beside it under `recommended` (inputs pinned in this module; the certificate binds the bytes)
        "sizing": launched_sizing(p.attack, geom, explicit=p.max_num_seqs is not None, util=p.gpu_memory_utilization,
                                  chunk_tokens=p.chunk_tokens, tensor_parallel=p.tensor_parallel),
        "watch_metrics": [dict(m) for m in WATCH_METRICS],
        "kv_coverage": coverage,
        "chunked_prefill": True,
        "vllm_version": VLLM_VERSION,
        "pytorch_cuda_alloc_conf": alloc_conf_env(p.tensor_parallel).get("PYTORCH_CUDA_ALLOC_CONF"),
        "params": p.as_dict(),
        "w7": p.w7 or None,
        "wave_uuid8": p.wave_uuid8 or None,
    }


@tools.function(image=client_image, volumes={VOL_PATH: VOL}, timeout=3 * 3600, cpu=8, memory=32768)
def stage_weights() -> dict:
    """Make the already-staged public snapshot available at /vol/hf/hub (shared with the campaign pool)."""
    dst = Path(MODEL_REPO_DIR)
    snap = Path(MODEL_PATH)
    legacy = Path(LEGACY_MODEL_REPO_DIR)
    legacy_snap = legacy / "snapshots" / ATTACKER_SNAPSHOT
    if snap.is_dir() and (snap / "config.json").is_file():
        return {"ok": True, "method": "already_present", "path": str(snap)}
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        raise RuntimeError(f"partial/non-matching model cache already exists at {dst}; refusing to replace it")
    if legacy_snap.is_dir() and (legacy_snap / "config.json").is_file():
        os.symlink(str(legacy), str(dst), target_is_directory=True)
        VOL.commit()
        if not (snap / "config.json").is_file():
            raise RuntimeError("Modal Volume did not preserve the /vol/hf attacker-cache symlink")
        return {"ok": True, "method": "symlink_existing_volume_snapshot", "path": str(snap)}
    os.environ.pop("HF_HUB_OFFLINE", None)
    from huggingface_hub import snapshot_download

    path = snapshot_download(ATTACKER_MODEL, revision=ATTACKER_SNAPSHOT, cache_dir=f"{VOL_PATH}/hf/hub",
                             local_files_only=False, max_workers=16)
    VOL.commit()
    if Path(path).resolve() != snap.resolve():
        raise RuntimeError(f"snapshot_download resolved {path}, expected {snap}")
    return {"ok": True, "method": "snapshot_download", "path": str(snap)}


@app.server(
    name=SERVER_NAME,
    image=pool_image,
    volumes={VOL_PATH: VOL},
    secrets=[pool_secret],
    gpu=P.gpu,
    cpu=P.cpu,
    memory=P.memory_mib,
    port=PORT,
    target_concurrency=MAX_NUM_SEQS,
    min_containers=1,
    max_containers=1,
    startup_timeout=3600,
    unauthenticated=True,
    exit_grace_period=60,
)
class SeedrepAttackerPoolServer:
    @modal.enter()
    def start(self) -> None:
        for tool in ("vllm", "ninja"):
            if not shutil.which(tool):
                raise RuntimeError(f"required runtime tool {tool!r} is not on PATH")
        # the container re-resolves the geometry from the baked params and holds it to the import-time one
        p_here = pool_params(os.environ)
        geom = pool_geometry(p_here)
        if (p_here, geom.max_model_len, geom.max_num_seqs, geom.max_num_batched_tokens) != (
                P, MAX_MODEL_LEN, MAX_NUM_SEQS, MAX_NUM_BATCHED_TOKENS):
            raise RuntimeError(f"resolved geometry drifted in the container: {geom.describe()} vs {GEOM.describe()}")
        env_problem = serve_env_problem(os.environ, p_here.tensor_parallel)
        if env_problem:
            raise RuntimeError(env_problem)
        model = Path(MODEL_PATH)
        if not (model / "config.json").is_file():
            raise RuntimeError(f"attacker snapshot absent at {MODEL_PATH}; run the explicitly named stage command first")
        key = os.environ.get(API_KEY_ENV, "")
        if not key:
            raise RuntimeError(f"{API_KEY_ENV} secret is absent/empty")

        from sisteer.eval.attacks import attacker_serving as serving

        pool_dir_p = Path(POOL_DIR)
        pool_dir_p.mkdir(parents=True, exist_ok=True)
        Path(ENDPOINT_PATH).unlink(missing_ok=True)
        Path(SERVING_MANIFEST).unlink(missing_ok=True)
        VOL.commit()
        log_path = Path(SERVE_LOG)
        self._log_file = log_path.open("w", buffering=1)
        opts = serving.AttackerServingOptions(POOL_QUANTIZATION, POOL_PREFIX_CACHING)
        cmd = vllm_argv(P, geom, opts)
        print(f"[pool] {APP_NAME} attack={P.attack} launching {ATTACKER_MODEL}@{ATTACKER_SNAPSHOT[:8]} gpu={P.gpu} "
              f"tp={P.tensor_parallel} L={MAX_MODEL_LEN} S={MAX_NUM_SEQS} ({geom.seqs_source}) "
              f"batched={MAX_NUM_BATCHED_TOKENS} chunked_prefill=1 serving={json.dumps(opts.as_dict())} text_only=1",
              flush=True)
        self._proc = subprocess.Popen(cmd, stdout=self._log_file, stderr=subprocess.STDOUT)
        _wait_ready(self._proc, log_path)

        deadline = time.time() + 60
        coverage = None
        last_error = None
        while time.time() < deadline:
            try:
                coverage = _attest_coverage(log_path.read_text(errors="replace"))
                break
            except RuntimeError as exc:
                last_error = exc
                time.sleep(2)
        if coverage is None:
            self._terminate()
            raise last_error or RuntimeError("vLLM coverage attestation missing")

        try:
            _, serving_fields = _attest_serving(
                log_path.read_text(errors="replace"), opts, manifest_path=SERVING_MANIFEST, log_path=SERVE_LOG,
                endpoint=f"http://127.0.0.1:{PORT}",
                extra={"pool_name": POOL_NAME, "snapshot": ATTACKER_SNAPSHOT, "lane": LANE,
                       "pool_index": P.pool_index, "attack": P.attack, "tensor_parallel_size": P.tensor_parallel,
                       "gpu": P.gpu, "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF")})
        except Exception:
            self._terminate()
            raise
        print(f"[pool] serving identity attested from the engine line: "
              f"{json.dumps({k: v for k, v in serving_fields.items() if k != 'serving'})}", flush=True)

        unauth = _http_status(f"http://127.0.0.1:{PORT}/v1/models")
        auth = _http_status(f"http://127.0.0.1:{PORT}/v1/models", key=key)
        if unauth not in (401, 403) or auth != 200:
            self._terminate()
            raise RuntimeError(f"bearer guard failed: unauthenticated /v1/models={unauth}, authenticated={auth}")

        server_id, server_url = _server_identity()
        app_id = _app_identity()
        call_id = _call_id()
        container_id = os.environ.get("MODAL_TASK_ID", "")
        if not (server_id and server_url and app_id and container_id):
            self._terminate()
            raise RuntimeError("Modal omitted server/app/url/container identity (or the app is not deployed as "
                               f"{APP_NAME}); refusing an unattributable endpoint")
        started = _utc_now()
        liveness = {"kind": "modal", "app_name": APP_NAME, "app_id": app_id, "server_name": SERVER_NAME,
                    "server_id": server_id, "call_id": call_id, "container_id": container_id}
        record = endpoint_record_fields(P, geom, coverage=coverage, serving_fields=serving_fields, url=server_url,
                                        liveness=liveness, started=started, call_id=call_id, server_id=server_id,
                                        container_id=container_id, source_sha=SOURCE_SHA)
        _write_endpoint(record)
        print(f"[pool] ADVERTISED app={APP_NAME} app_id={app_id} server_id={server_id} call_id={call_id} "
              f"container_id={container_id} url={server_url}", flush=True)
        print(f"[pool] coverage: window={coverage['concurrency_window']} "
              f"max_concurrency={coverage['max_concurrency_reported']}x kv_tokens={coverage['kv_tokens_reported']}", flush=True)
        self._stop = threading.Event()
        self._heartbeat = threading.Thread(target=self._commit_loop, daemon=True)
        self._heartbeat.start()

    def _commit_loop(self) -> None:
        while not self._stop.wait(10):
            if self._proc.poll() is not None:
                print(f"[pool] FATAL: vLLM exited rc={self._proc.returncode}", flush=True)
                Path(ENDPOINT_PATH).unlink(missing_ok=True)
                VOL.commit()
                return
            VOL.commit()

    def _terminate(self) -> None:
        proc = getattr(self, "_proc", None)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)

    @modal.exit()
    def stop(self) -> None:
        stop = getattr(self, "_stop", None)
        if stop is not None:
            stop.set()
        endpoint = Path(ENDPOINT_PATH)
        if endpoint.is_file():
            try:
                rec = json.loads(endpoint.read_text())
            except Exception:  # noqa: BLE001
                rec = {}
            if (rec.get("liveness") or {}).get("container_id") == os.environ.get("MODAL_TASK_ID"):
                endpoint.unlink(missing_ok=True)
        self._terminate()
        log_file = getattr(self, "_log_file", None)
        if log_file is not None:
            log_file.close()
        VOL.commit()


@tools.function(image=client_image, volumes={VOL_PATH: VOL}, secrets=[pool_secret], timeout=3600, cpu=4, memory=8192)
def canary(n_calls: int = 4, concurrency: int = 1, max_new_tokens: int = 16) -> dict:
    """Prove a separate Modal container can reach and use the guarded pool (the original's canary), and that
    the canary's own body ROUTES as an attacker request under :func:`route_request`."""
    from sisteer.eval.attacks.vllm_attacker import VLLMAttackerClient

    canary_container = os.environ.get("MODAL_TASK_ID", "")
    deadline = time.time() + ENDPOINT_PUBLISH_WAIT_S
    last_stale: Exception | None = None
    while True:
        endpoint = _wait_endpoint_file(max(1, deadline - time.time()))
        pool_container = _assert_separate_container(endpoint, canary_container)
        ep = contract.PoolEndpoint.from_dict(endpoint)
        try:
            ep, key, evidence = contract.connect_endpoint(ep, squeue=False, env=os.environ,
                                                          expect_model=ATTACKER_MODEL, expect_snapshot=ATTACKER_SNAPSHOT)
            break
        except contract.PoolStale as exc:
            last_stale = exc
            if time.time() >= deadline:
                raise RuntimeError(f"no current deployed endpoint appeared within 600s: {last_stale}") from exc
            time.sleep(5)
    table = RoutingTable(attacker_models=frozenset({ep.model}), attacker_endpoint=ep.url,
                         target_vmodels=frozenset(), target_endpoint="", target_params={})
    role, routed = route_request({"model": ep.model, "temperature": 0.0, "top_p": 1.0}, table)
    if (role, routed) != ("attacker", ep.url):
        raise RuntimeError(f"the canary body routed to {role}@{routed}, not the pool")
    shared_prefix = ("This is a benign transport canary. Read the repeated context and answer the final request briefly. "
                     + "shared-prefix-token " * 2500)

    def one(i: int) -> dict:
        client = VLLMAttackerClient(ep.url, ep.model, api_key=key, timeout=600, max_retries=1)
        t0 = time.time()
        text = client.chat([{"role": "system", "content": shared_prefix},
                            {"role": "user", "content": f"Reply with the integer {i}, then list every integer from {i} to {i + 600}, comma separated."}],
                           temperature=0.0, top_p=1.0, max_new_tokens=int(max_new_tokens))
        return {"index": i, "latency_s": round(time.time() - t0, 3), "reply_chars": len(text or "")}

    started = time.strftime("%m-%d %H:%M:%S", time.gmtime())
    t_all = time.time()
    if int(concurrency) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=int(concurrency)) as ex:
            calls = list(ex.map(one, range(int(n_calls))))
    else:
        calls = [one(i) for i in range(int(n_calls))]
    wall_s = round(time.time() - t_all, 3)
    stats = {"n_stats_lines": 0}
    deadline = time.time() + 90
    while time.time() < deadline:
        time.sleep(10)
        VOL.reload()
        log_path = Path(str(endpoint.get("serve_log") or SERVE_LOG))
        if log_path.is_file():
            stats = contract.log_stats(log_path.read_text(errors="replace"), since=started)
        if stats.get("n_stats_lines") and stats.get("prefix_cache_hit_rate_last") is not None:
            break
    live = endpoint.get("liveness") or {}
    return {
        "ok": True, "separate_container": True, "pool_container_id": pool_container, "canary_container_id": canary_container,
        "modal_app_id": live.get("app_id"), "modal_call_id": live.get("call_id"),
        "modal_server_id": live.get("server_id"), "endpoint": ep.identity(), "verification": evidence,
        "routing_self_check": {"role": role, "endpoint": routed},
        "concurrency": int(concurrency), "max_new_tokens": int(max_new_tokens), "wall_s": wall_s,
        "latency_mean_s": round(sum(c["latency_s"] for c in calls) / max(1, len(calls)), 3),
        "latency_max_s": max((c["latency_s"] for c in calls), default=0.0), "calls": calls, "pool_logstats": stats,
    }


@tools.function(image=client_image, volumes={VOL_PATH: VOL}, timeout=3600)
def endpoint_record() -> dict:
    deadline = time.time() + ENDPOINT_PUBLISH_WAIT_S
    last_stale: Exception | None = None
    while time.time() < deadline:
        record = _wait_endpoint_file(max(1, deadline - time.time()))
        ep = contract.PoolEndpoint.from_dict(record)
        try:
            contract.check_modal_live(ep)
            return record
        except contract.PoolStale as exc:
            last_stale = exc
            time.sleep(5)
    raise RuntimeError(f"no current deployed endpoint appeared within 600s: {last_stale}")


def health_check_local(endpoint_path: Path | None = None, out: Path | None = None) -> dict:
    """Grading-host-side pre-spawn gate: read the pool's local endpoint record (``--action endpoint`` first), the
    lane bearer, and probe the three URLs; the verdict is written beside the record and returned."""
    ep_path = endpoint_path or _local_endpoint_path()
    if not ep_path.is_file():
        raise SystemExit(f"[seedrep-pool] no endpoint record at {ep_path}; run --action endpoint first")
    record = json.loads(ep_path.read_text())
    if record.get("pool_index") != P.pool_index or record.get("lane") != LANE:
        raise SystemExit(f"[seedrep-pool] {ep_path} is pool {record.get('lane')}-{record.get('pool_index')}, not {POOL_NAME}")
    key = _load_local_api_key()
    verdict = endpoint_health(record, key, get=lambda url, bearer: _http_status(url, key=bearer))
    verdict.update(app_name=APP_NAME, endpoint_record=str(ep_path))
    _write_json_atomic(out or _local_health_path(), verdict)
    return verdict


@tools.local_entrypoint()
def main(action: str = "canary", n_calls: int = 4, endpoint_out: str = "", concurrency: int = 1, max_new_tokens: int = 16) -> None:
    if action == "stage":
        print(json.dumps(stage_weights.remote(), indent=2))
        return
    if action == "canary":
        print(json.dumps(canary.remote(n_calls, concurrency, max_new_tokens), indent=2))
        return
    if action == "endpoint":
        rec = endpoint_record.remote()
        out = Path(endpoint_out).expanduser() if endpoint_out else _local_endpoint_path()
        rec["api_key_file"] = str(_local_key_path())
        print(_write_json_atomic(out, rec))
        return
    if action == "health":
        print(json.dumps(health_check_local(Path(endpoint_out).expanduser() if endpoint_out else None), indent=2))
        return
    raise SystemExit("--action must be stage, endpoint, health, or canary")


if __name__ == "__main__":
    # `python scripts/modal/seedrep_attacker_pool.py --health [--endpoint PATH]` -- the grading-host-side gate without a Modal run
    import argparse

    ap = argparse.ArgumentParser(description="seedrep attacker pool: grading-host-side health gate")
    ap.add_argument("--health", action="store_true")
    ap.add_argument("--endpoint", default="")
    a = ap.parse_args()
    if a.health:
        print(json.dumps(health_check_local(Path(a.endpoint).expanduser() if a.endpoint else None), indent=2))
    else:
        ap.print_help()
