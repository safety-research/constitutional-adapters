#!/usr/bin/env python
"""served_decode_sweep — measure ONE model's batched-token BUDGET and its served
decode THROUGHPUT vs batch on one H200, through the production standup path.

Two phases, run per model (an sbatch array task each — scripts/sbatch/
geom_sweep.sbatch); ``scripts/geom_sweep_analyze.py`` folds every model's
results into the ``BUDGET_BY_MODEL`` paste block of
``sisteer.capeval.serving.geometry`` and the per-model x window tables.

WHY. The served geometry table sizes every launch to ONE batched-token budget,
393,216 = 24 x 16384 — Llama-3.1-8B's certified Petri footprint (the 32 x
16384 negative result in vllm_server.py). Under the composer's no-chunking
identity (``max_num_batched_tokens = max_num_seqs x max_model_len``) that
budget is what vLLM's memory profiling runs its forward over, and KV gets
what is left — so it is a MODEL property (weights + activation peak + bytes
per KV token), and the Llama number undersells smaller-KV models: live
16384 x 24 engines reported a GPU KV cache of 2.11M tokens on qwen35_4b (128
full 16k sequences), 1.16M on gemma4e4b (70x), 0.60M on llama8b (36x). So
the budget is per-model, and each model's throughput is measured on the
cluster.

``--phase budget``: bracket the largest ``S x L`` that LAUNCHES. At a fixed
block-aligned L (default 8192) the ladder 393,216 / 524,288 / 655,360 /
786,432 / 1,048,576 / 1,310,720 / 1,572,864 (S = 48 ... 192) is probed
upward through :func:`capctrl_serving_up.bring_up` — the SAME composer path
the production standup uses, explicit ``max_model_len``/``max_num_seqs``, the
over-budget guard waived on purpose, gpu_memory_utilization 0.90 — until the
first refusal (readiness holds the attested KV coverage to the budget:
``vllm_server.kv_coverage_problem``) or OOM, then ONE bisection between the
last pass and the first fail. Each probe records pass/fail, the attested KV
facts (uniform: ``num_gpu_blocks x block_size``; hybrid: the allocator's
max-concurrency x L) and the memory-profiling lines of the server log.
Output: ``budget.json`` (every probe + the largest passing product + a
``paste``-ready ModelBudget record).

``--phase sweep``: at the measured budget, for each eval window L (default:
the per-eval table's windows — 8704 multiturn_attacks, 7424 capability / bon,
11520 families_served, 16384 petri, 1536 capability:short) stand the server
up at ``S = min(budget // L, 512)`` — 256 instead of 512 for a hybrid-recurrent
model (:data:`S_MAX_HYBRID`: per-slot recurrent state makes the token budget
S-dependent) — retrying a REFUSED standup at the next lower ladder S up to
three times before skipping that window (never failing the phase), and, for
each batch B in the geometric
ladder 8 ... 512 (capped at S), drive B concurrent CLOSED-LOOP requests for
``--measure-s`` (90 s) of steady state after ``--warmup-s`` (20 s) against
(a) the un-steered passthrough virtual model and (b) one steered psr_kl cell
when the model has a fitted-object bank, with a synthetic prompt of a fixed
token length (built by repeating a neutral filler until the proxy's
``/tokenize`` says N; default 1,000 ~ the evals' p50 prompt) and
``max_tokens 512`` + ``ignore_eos`` so every request decodes the full 512
(throughput, not content). Per point: completion tokens/s and requests/s
(client-side, in-window completions), per-request latency p50/p95, TTFT
(engine histogram deltas — the proxy serves non-streaming responses only, so
a client cannot observe first tokens), achieved concurrency and preemptions
(``/metrics`` sampled every 2 s: ``num_requests_running``,
``num_preemptions_total`` — a preemption means KV did not cover the batch),
and GPU duty (``nvidia-smi dmon -s u -d 1``: sm% mean and p10). Output:
``sweep.jsonl`` (one row per point, resumable), ``sweep_summary.json`` (per
window x arm: the throughput-vs-B table and the KNEE = smallest B reaching
90% of the max tok/s, plus tok/s at the old S = 24 and the speedup), and PNG
figures (tok/s vs B, duty vs B, latency vs B) into ``$FIGS_DIR``.

Runs from the pinned serving venv (configs/capeval/geom-sweep-requirements.txt
= vllm + matplotlib); the CPU tests import this module without vllm,
aiohttp, torch or matplotlib — every runtime import is lazy.
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import logging
import math
import os
import re
import shlex
import socket
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

log = logging.getLogger("served_decode_sweep")


# --------------------------------------------------------------------------- #
# The models                                                                   #
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class ModelServe:
    """How ONE canonical model key is served: the HF id + revision pin the
    standup resolves (``configs/two_si_*.yaml``'s ``model`` /
    ``model_revision`` — the config IS the pin; ``rev`` here is the value the
    config must carry, or the pin :func:`resolve_serving_config` materializes
    when the config leaves it ``null``), the two-SI config the standup builds
    the registry from, and the fitted-object bank subdirectory on the cluster
    (``$SHARED/exp/objects/scale-12/<bank>``)."""

    key: str
    hf: str
    rev: str
    cfg: str
    bank: str
    note: str = ""


#: canonical key -> serving pins. ORDER IS LOAD-BEARING: the sbatch array index
#: decodes into this order (scripts/sbatch/geom_sweep.sbatch MODELS mirrors it;
#: tests/test_served_decode_sweep.py pins the two equal), and it equals
#: sisteer.capeval.serving.geometry.MODEL_KEYS.
MODELS: Dict[str, ModelServe] = {
    "llama8b": ModelServe(
        key="llama8b", hf="NousResearch/Meta-Llama-3.1-8B-Instruct",
        rev="d10aef7999a2b5ba950ab3974312feeedbfe0b77", cfg="configs/two_si.yaml",
        bank="llama8b", note="the certified reference: its budget phase should reproduce the "
                             "393,216 < 524,288 wall (64 x 8192 = the same KV + MLP-intermediate "
                             "footprint as 32 x 16384)"),
    "llama3b": ModelServe(
        key="llama3b", hf="unsloth/Llama-3.2-3B-Instruct",
        rev="006f5dcd1393c3add266de40994ba96225e9689d", cfg="configs/two_si_llama32_3b.yaml",
        bank="llama3b", note="the config carries model_revision: null (pinned at first fetch); "
                             "the pin is the promoted object's fit_provenance / the Modal "
                             "generator's (twosi_generate.py llama32_3b) — resolve_serving_config "
                             "materializes a pinned copy for the standup"),
    "gemma4e4b": ModelServe(
        key="gemma4e4b", hf="google/gemma-4-E4B-it",
        rev="ee0ef6023621cff504d758262d4e04895a5af4a2",
        cfg="configs/two_si_gemma4e4b_reduced.yaml", bank="gemma4e4b",
        note="the served capability lane's config (capctrl_served.py MODELS)"),
    "gemma4e2b": ModelServe(
        key="gemma4e2b", hf="google/gemma-4-E2B-it",
        rev="3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        cfg="configs/two_si_gemma4e2b_fits.yaml", bank="gemma4e2b",
        note="the fits config (hub id + the fit pin); the shared cluster A cache holds "
             "snapshot 3e22461f (verified 2026-09-03), so the _eval config's staged-dir "
             "workaround is not needed"),
    "qwen35_9b": ModelServe(
        key="qwen35_9b", hf="Qwen/Qwen3.5-9B",
        rev="c202236235762e1c871ad0ccb60c8ee5ba337b9a", cfg="configs/two_si_qwen35_9b.yaml",
        bank="qwen35_9b", note="hybrid (linear + full attention): KV coverage is the allocator's "
                               "max-concurrency verdict, prefix caching composed OFF"),
    "qwen35_4b": ModelServe(
        key="qwen35_4b", hf="Qwen/Qwen3.5-4B",
        rev="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", cfg="configs/two_si_qwen35_4b.yaml",
        bank="qwen35_4b", note="hybrid, as qwen35_9b"),
    "glm46vflash": ModelServe(
        key="glm46vflash", hf="zai-org/GLM-4.6V-Flash",
        rev="411bb4d77144a3f03accbf4b780f5acb8b7cde4e", cfg="configs/two_si_glm46vflash.yaml",
        bank="glm46vflash", note="onboarded; no fitted-object bank under scale-12 on the cluster "
                                 "(2026-09-03) -> passthrough arm only unless one appears"),
    "nemotron3nano4b": ModelServe(
        key="nemotron3nano4b", hf="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
        rev="dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f",
        cfg="configs/two_si_nemotron3nano4b.yaml", bank="nemotron3nano4b",
        note="hybrid (Mamba2 + 4 attention blocks): KV pages are mostly per-request state; "
             "no fitted-object bank under scale-12 (2026-09-03) -> passthrough arm only"),
}

MODEL_ORDER: Tuple[str, ...] = tuple(MODELS)

#: the budget ladder: the products probed upward at --budget-len (S = product // L)
BUDGET_LADDER: Tuple[int, ...] = (393_216, 524_288, 655_360, 786_432, 1_048_576,
                                  1_310_720, 1_572_864)
DEFAULT_BUDGET_LEN = 8192
#: the batch ladder of the sweep phase, intersected with [.., S]
BATCH_LADDER: Tuple[int, ...] = (8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512)
#: the per-eval table's windows that apply to every served model
DEFAULT_WINDOWS: Tuple[int, ...] = (8704, 7424, 11520, 16384, 1536)
MAX_SEQS_CAP = 512
#: PER-SLOT STATE CAP for hybrid-recurrent families (canary 373464, 2026-09-03):
#: the budget is a TOKEN budget measured at ONE S (the probe's — qwen35_4b's
#: 720,896 launched as 88 x 8192), and a hybrid model's gated-deltanet / Mamba
#: layers pre-allocate a fixed recurrent state PER SLOT (~24 MiB per sequence
#: at 9B), so max_num_seqs itself costs memory and the feasible S x L SHRINKS
#: as S grows: the same 720,896 was refused as 469 x 1536. The sweep caps a
#: hybrid model's per-window S here (the same ``layer_types`` /
#: ``linear_attention`` detection vllm_server uses for its caching decision);
#: the per-window standup + retry (:func:`stand_up_window_with_retry`) is the
#: real check for every model.
S_MAX_HYBRID = 256
#: a refused window is retried at the next lower ladder S this many times
WINDOW_RETRIES = 3
#: settle time after the samplers start, before the first point
SAMPLER_SETTLE_S = 3.0
#: sweep.jsonl row kinds: measurement points, and the window-level events
ROW_KIND_POINT = "point"
ROW_KIND_REFUSED = "window_refused"
ROW_KIND_SKIPPED = "window_skipped"
KNEE_FRACTION = 0.90
#: the pre-table concurrency every served eval inherited (Petri's 24)
OLD_S = 24
DEFAULT_PROMPT_TOKENS = 1000
DEFAULT_MAX_TOKENS = 512
DEFAULT_WARMUP_S = 20.0
DEFAULT_MEASURE_S = 90.0
METRICS_PERIOD_S = 2.0
DEFAULT_GPU_MEM_UTIL = "0.90"
#: the steered arm: the first of these with an object in the model's bank
PSR_CELL_PREFERENCE: Tuple[str, ...] = ("si_constitution__psr_kl_all",
                                        "safety_refusal__psr_kl_all",
                                        "si_irrelevant__psr_kl_all")
ARM_PASSTHROUGH = "passthrough"
ARM_STEERED = "steered"
FILLER = ("The river runs quietly past the old stone mill while the miller counts sacks "
          "of grain and the afternoon light moves slowly across the wooden floor. ")

#: under the cluster's shared root ($SHARED through sisteer.site: the job's SISTEER_SHARED_ROOT, else CLUSTER_SHARED_ROOT, else the
#: site config's cluster block), expanded where they are used
DEFAULT_RESULTS_ROOT = "$SHARED/exp/geom-sweep"
DEFAULT_OBJECTS_ROOT = "$SHARED/exp/objects/scale-12"


# --------------------------------------------------------------------------- #
# Pure helpers (CPU-tested)                                                    #
# --------------------------------------------------------------------------- #


def ladder_seqs(product: int, max_model_len: int) -> int:
    """S for a budget-ladder product at window L; the product must be an exact
    multiple of L (the ladder is defined on products, the launch on S x L)."""
    if product % max_model_len:
        raise ValueError(f"budget ladder product {product:,} is not a multiple of the "
                         f"probe window {max_model_len}")
    return product // max_model_len


def batch_ladder(max_num_seqs: int, ladder: Sequence[int] = BATCH_LADDER) -> Tuple[int, ...]:
    """The geometric batch ladder intersected with ``[.., S]``; always non-empty
    (an S below the first rung yields the single point ``(S,)``)."""
    out = tuple(b for b in ladder if b <= max_num_seqs)
    return out if out else (int(max_num_seqs),)


def permitted_seqs(budget: int, max_model_len: int, cap: int = MAX_SEQS_CAP) -> int:
    return max(1, min(int(budget) // int(max_model_len), int(cap)))


def is_hybrid_model(ms: ModelServe) -> bool:
    """Hybrid-recurrent (Qwen3.5 gated-deltanet, Nemotron-H Mamba2): the SAME
    detection ``vllm_server.hybrid_recurrent_model`` performs for the caching
    decision — the HF config's ``layer_types`` containing ``linear_attention``
    (nested ``text_config`` handled), with its name fallback for an uncached
    config."""
    from sisteer.capeval.serving import vllm_server as vs
    return bool(vs.hybrid_recurrent_model(ms.hf, ms.rev))


def seqs_cap(hybrid: bool) -> int:
    """The per-window S cap: :data:`MAX_SEQS_CAP`, tightened to
    :data:`S_MAX_HYBRID` for a hybrid-recurrent model (per-slot state)."""
    return min(MAX_SEQS_CAP, S_MAX_HYBRID) if hybrid else MAX_SEQS_CAP


def next_lower_seqs(s_tried: int, ladder: Sequence[int] = BATCH_LADDER) -> Optional[int]:
    """The S a refused window retries at: the largest ladder rung strictly below
    ``s_tried`` (469 -> 384 -> 256 -> 192 ...); ``None`` when none is lower."""
    lower = [int(b) for b in ladder if int(b) < int(s_tried)]
    return max(lower) if lower else None


def is_point_row(row: Mapping[str, Any]) -> bool:
    """A measurement row (one (window, arm, B) point) as opposed to a
    window-level event row (``window_refused`` / ``window_skipped``)."""
    return row.get("kind", ROW_KIND_POINT) == ROW_KIND_POINT and "B" in row and "arm" in row


def window_labels(max_model_len: int) -> Tuple[str, ...]:
    """The geometry-table keys SERVED at exactly this window (for the row's
    provenance and the figure legend); empty for a window off the table. An
    informational (``requires_recertification`` — the ``petri:<model>`` rows —
    or ``informational`` — ``capability:grader``) or PROVISIONAL entry is not
    one: ``resolve_geometry`` refuses all three, and a provisional entry's
    window is either the certified footprint it is measured AT
    (``pass1_geometry``) or an analytic derivation, never a window served for
    it — such entries must not relabel a window they are not served at."""
    from sisteer.capeval.serving import geometry as sg
    return tuple(k for k, g in sg.TABLE.items()
                 if g.max_model_len == max_model_len
                 and not g.requires_recertification and not g.informational
                 and not g.provisional)


def eval_key_for_window(max_model_len: int) -> Optional[str]:
    """A resolvable (non-informational) table key at this window, preferring a
    top-level eval over a sub-entry, so the standup geometry names the eval."""
    labels = window_labels(max_model_len)
    if not labels:
        return None
    top = [k for k in labels if ":" not in k]
    return (top or list(labels))[0]


def next_budget_probe(probes: Sequence[Mapping[str, Any]], ladder: Sequence[int],
                      max_model_len: int) -> Optional[int]:
    """The next product to probe given the probes so far, or ``None`` when the
    bracket is closed: walk the ladder upward until the first failure, then
    bisect ONCE (a single off-ladder probe at the midpoint S between the last
    pass and the first fail). A failure on the very first rung closes the
    bracket with no passing probe (the analyzer reports it; nothing is
    guessed)."""
    ladder = sorted(int(x) for x in ladder)
    passes = sorted(int(p["product"]) for p in probes if p.get("status") == "pass")
    fails = sorted(int(p["product"]) for p in probes if is_wall(p))
    if not fails:
        above = [x for x in ladder if not passes or x > passes[-1]]
        return above[0] if above else None
    if not passes:
        return None
    lo, hi = passes[-1], fails[0]
    if any(x not in ladder for x in passes + fails):
        return None                       # the one bisection has been spent
    s_lo, s_hi = lo // max_model_len, hi // max_model_len
    s_mid = (s_lo + s_hi) // 2
    if s_mid <= s_lo or s_mid >= s_hi:
        return None
    return s_mid * max_model_len


def budget_bracket(probes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """The bracket a probe list defines: the largest passing probe (the
    budget), the smallest refused product (the wall), and whether the walk
    reached a wall at all."""
    passes = [p for p in probes if p.get("status") == "pass"]
    fails = [p for p in probes if is_wall(p)]
    aborted = [p for p in probes if p.get("status") != "pass" and not is_wall(p)]
    largest = max(passes, key=lambda p: int(p["product"])) if passes else None
    first_refused = min(int(p["product"]) for p in fails) if fails else None
    return {"largest_passing": largest, "first_refused": first_refused,
            "n_probes": len(probes), "wall_reached": first_refused is not None,
            "n_aborted": len(aborted),
            "aborted_statuses": sorted({str(p.get("status")) for p in aborted})}


#: INFRA, not geometry and not our code: the card is held by another tenant /
#: a wedged driver. The budget walk ABORTS on it (exit non-zero, no wall
#: recorded — resubmit excluding the node) and a window standup skips the retry
#: ladder (a smaller S cannot help) and re-raises the same SystemExit.
STATUS_GPU_UNAVAILABLE = "gpu_unavailable"
#: the production spellings (fan-out 2026-09-03): torch.AcceleratorError on a
#: busy card (glm46vflash, on a compute node) ...
_GPU_UNAVAILABLE_RE = re.compile(
    r"CUDA-capable device\(s\) is/are busy or unavailable|"
    r"CUDA error: .*busy or unavailable", re.I)
#: ... and vLLM's engine-init refusal when profiling leaves NO room for a single
#: KV block (nemotron3nano4b at 128 x 8192) — a genuine memory WALL, like an OOM.
_OOM_RE = re.compile(
    r"out of memory|OutOfMemoryError|No available memory for the cache blocks", re.I)


def classify_launch_failure(exc_text: str, log_text: str = "") -> str:
    """Why a probe did not launch: an unavailable GPU (infra), the readiness
    gate's KV-coverage refusal, a memory wall (an OOM, or vLLM's "No available
    memory for the cache blocks" at engine init — both read from the exception
    AND the server log), a readiness timeout, or another launch error."""
    t = exc_text or ""
    both = (log_text or "") + "\n" + t
    if _GPU_UNAVAILABLE_RE.search(both):
        return STATUS_GPU_UNAVAILABLE
    if "KV cache holds" in t or "supports only" in t or "KV-backed" in t:
        return "kv_under_coverage"
    if _OOM_RE.search(both):
        return "oom"
    if "did not become ready" in t:
        return "not_ready"
    return "launch_error"


def gpu_unavailable_message(node: str, tag: Optional[str], error: Optional[str]) -> str:
    """The one abort message for :data:`STATUS_GPU_UNAVAILABLE`, distinct from
    the probe-code-error message: infra, resubmit elsewhere."""
    return (f"[sweep] {tag or 'standup'}: GPU busy/unavailable on {node} — resubmit excluding "
            f"it (sbatch --exclude={node}); infra, not geometry: no wall recorded. "
            f"{(error or '')[:300]}")


def transformers_version() -> Optional[str]:
    """The serving venv's transformers version (stamped into budget.json and
    every sweep row: gemma-4 serves only under the 5.14.1 variant venv,
    Nemotron only under 5.15.0 — see configs/capeval/geom-sweep-requirements*.txt)."""
    try:
        import transformers
        return str(transformers.__version__)
    except Exception:  # noqa: BLE001
        return None


def requirements_file() -> Optional[str]:
    """The requirements file the sbatch bootstrapped this venv from (it exports
    ``GEOM_REQS``); ``None`` off-cluster."""
    v = os.environ.get("GEOM_REQS")
    return os.path.basename(v) if v else None


def _gpu_name() -> Optional[str]:
    out = _nvidia_smi("--query-gpu=name", "--format=csv,noheader")
    if not out:
        return None
    first = out.strip().splitlines()[0].strip() if out.strip() else ""
    return first or None


def measurement_fingerprint(ctx: "SweepContext", *, phase: str, **extra: Any) -> Dict[str, Any]:
    """WHAT a measurement is (review R2): the model + revision, the config and
    manifest bytes, the repo sha, the requirements file + its bytes, the
    resolved serving-stack versions, the GPU model, the utilization, the hybrid
    cap, the steered cell AND THE BYTES THAT CELL RESOLVES TO
    (:func:`object_identity`; E50 #7 — a probe standup is deliberately unbound
    to any object digest, so without this a re-fitted object served under the
    same cell name resumes onto the previous object's probes and rows), plus
    the phase's own settings (``extra``: budget L + ladder, or windows / batch
    ladder / prompt / max_tokens / warm-up / measure). Persisted BEFORE the
    first probe / point; a resume whose fingerprint differs must not reuse the
    recorded probes or rows.

    ``steered_object`` names the exporter's expected served PROJECTION keyed by
    ROLE (E55 #4), not every ``object.*`` file in the bank dir keyed by
    filename — the change moves ``measurement_id`` for steered phases, so a
    store recorded by an older sweep refuses reuse (loudly, naming the field)
    rather than resuming under a label whose meaning changed."""
    from sisteer.capeval.serving.run_identity import file_sha16, serving_versions, tree_sha

    manifest = Path(ctx.manifest)
    if not manifest.is_absolute():
        manifest = Path(ctx.repo) / manifest
    reqs = os.environ.get("GEOM_REQS")
    fp: Dict[str, Any] = {
        "phase": phase, "model_key": ctx.ms.key, "hf": ctx.ms.hf, "rev": ctx.ms.rev,
        "config_sha16": file_sha16(ctx.config_path), "manifest_sha16": file_sha16(manifest),
        "repo_sha": tree_sha(ctx.repo),
        "requirements_file": requirements_file(), "requirements_sha16": file_sha16(reqs),
        "versions": serving_versions(), "gpu_name": _gpu_name(),
        "gpu_mem_util": str(ctx.gpu_mem_util), "hybrid": bool(ctx.hybrid),
        "seqs_cap": int(ctx.seqs_cap), "steered_cell": ctx.steered_cell,
        "steered_object": object_identity(ctx.objects_dir, ctx.steered_cell),
    }
    fp.update(extra)
    return fp


def fingerprint_id(fp: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(fp, sort_keys=True, default=str).encode("utf-8")
                          ).hexdigest()[:16]


def fingerprint_diff(recorded: Mapping[str, Any], current: Mapping[str, Any]
                     ) -> List[Tuple[str, Any, Any]]:
    keys = sorted(set(recorded) | set(current))
    return [(k, recorded.get(k), current.get(k)) for k in keys
            if recorded.get(k) != current.get(k)]


def refuse_reuse_message(path: Path, what: str, n: int, recorded_id: Any, current_id: str,
                         diff: Sequence[Tuple[str, Any, Any]], results_dir: Path) -> str:
    lines = "\n".join(f"  - {k}: recorded {a!r} != current {b!r}" for k, a, b in diff[:12])
    recorded = (f"{recorded_id}" if recorded_id not in (None, "", "None")
                else "NONE (a pre-fingerprint result)")
    return (f"[sweep] {path} holds {n} {what} of a DIFFERENT measurement (recorded id "
            f"{recorded}, current {current_id}) — refusing to reuse them (a resumed store "
            "would mix two measurements under one label). Differing fingerprint fields:\n"
            f"{lines or '  - (no fingerprint recorded: a pre-fingerprint result)'}\n"
            f"Run into a fresh --results-root, or move {results_dir} aside.")


def partition_by_measurement(rows: Sequence[Mapping[str, Any]]
                             ) -> Dict[Optional[str], List[Mapping[str, Any]]]:
    """``{measurement_id: rows}`` over EVERY row kind (``None`` = pre-fingerprint
    rows); the analyzer summarizes each partition on its own."""
    out: Dict[Optional[str], List[Mapping[str, Any]]] = {}
    for r in rows:
        out.setdefault(r.get("measurement_id"), []).append(r)
    return out


def point_key(row: Mapping[str, Any]) -> Tuple[int, str, int, int]:
    """A sweep point's completion key: (window L, arm, B, ACTUAL S) — a retry
    point at a reduced S is not the same point as one at the target S."""
    return (int(row["window_L"]), str(row["arm"]), int(row["B"]), int(row["S"]))


_MEM_LINE_MARKERS = ("Memory profiling takes", "can use total_gpu_memory",
                     "Available KV cache memory", "GPU KV cache size",
                     "Maximum concurrency", "weights memory", "torch peak memory",
                     "non-KV-cache memory", "out of memory", "OutOfMemoryError",
                     "No available memory for the cache blocks", "busy or unavailable")


def scrape_server_log(log_text: str) -> Dict[str, Any]:
    """The memory-profiling facts vLLM logs at launch: the raw lines that carry
    them (verbatim, for the record) plus the parsed numbers where a spelling is
    known — KV tokens, the engine's own max-concurrency factor, the available
    KV memory, weights / peak-activation GiB."""
    out: Dict[str, Any] = {"memory_lines": []}
    for ln in (log_text or "").splitlines():
        if any(m in ln for m in _MEM_LINE_MARKERS):
            out["memory_lines"].append(ln.strip()[-400:])
    m = re.search(r"GPU KV cache size[:=]?\s*([\d,_]+)\s*tokens", log_text or "")
    if m:
        out["log_kv_cache_tokens"] = int(m.group(1).replace(",", "").replace("_", ""))
    m = re.search(r"Maximum concurrency for ([\d,_]+) tokens? per request[^\d]*([\d.]+)x",
                  log_text or "")
    if m:
        out["log_max_concurrency_x"] = float(m.group(2))
        out["log_max_concurrency_len"] = int(m.group(1).replace(",", "").replace("_", ""))
    m = re.search(r"Available KV cache memory:\s*([\d.]+)\s*GiB", log_text or "")
    if m:
        out["log_kv_memory_gib"] = float(m.group(1))
    m = re.search(r"weights memory:\s*([\d.]+)\s*GiB", log_text or "")
    if m:
        out["log_weights_gib"] = float(m.group(1))
    m = re.search(r"torch peak memory increase:\s*([\d.]+)\s*GiB", log_text or "")
    if m:
        out["log_torch_peak_gib"] = float(m.group(1))
    m = re.search(r"Total non-KV-cache memory:\s*([\d.]+)\s*GiB", log_text or "")
    if m:
        out["log_non_kv_gib"] = float(m.group(1))
    out["log_oom"] = bool(_OOM_RE.search(log_text or ""))
    return out


def reclassify_probe(probe: Mapping[str, Any], results_dir: Optional[Path] = None
                     ) -> Dict[str, Any]:
    """A LOADED probe (a resume) re-read through today's classifier: its stored
    error text + the memory lines it scraped + any ``standups/<tag>__job*/
    vllm_server.log`` still on disk. A rung an older classifier filed as
    ``launch_error`` on "No available memory for the cache blocks" (nemotron
    3nano4b at 128 x 8192) becomes the ``oom`` WALL it is, so the resumed walk
    bisects 96 <-> 128 instead of re-probing or aborting. A passing probe is
    never touched; a re-classified one records ``reclassified_from``."""
    p = dict(probe)
    if p.get("status") == "pass" or "product" not in p:
        return p
    log_text = "\n".join(str(x) for x in (p.get("memory_lines") or ()))
    if results_dir is not None and p.get("tag"):
        for lp in sorted(Path(results_dir).glob(f"standups/{p['tag']}__job*/vllm_server.log")):
            try:
                log_text += "\n" + lp.read_text(errors="replace")[-20000:]
            except OSError:
                continue
    error = str(p.get("error") or "")
    if not error.strip() and not log_text.strip():
        return p          # nothing to re-read: the recorded verdict stands (never demoted to launch_error)
    new = classify_launch_failure(error, log_text)
    if new != p.get("status"):
        p["reclassified_from"] = p.get("status")
        p["status"] = new
    return p


#: Probe statuses that are a genuine GEOMETRY wall (the server itself refused or
#: ran out of memory at that S x L). Everything else that is not "pass" — our own
#: probe code raising after readiness ("launch_error"), or a readiness timeout
#: ("not_ready") — is NOT evidence about the geometry and must never define the
#: bracket (canary 370305 filed an AttributeError in the KV-facts reader as the
#: "first refused" rung and closed the walk at 393,216).
WALL_STATUSES = frozenset({"kv_under_coverage", "oom"})


def is_wall(probe: Mapping[str, Any]) -> bool:
    return probe.get("status") in WALL_STATUSES and "product" in probe


def worker_reports_of(att: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    """The per-worker scheduler reports of a launch attestation. The attestation
    nests them at ``scheduler_geometry["worker_reports"]`` (``scheduler_geometry``
    is a dict: rpc_method / expected / n_workers / worker_reports / note) and
    mirrors the list at the top level; iterating the dict yielded its KEYS."""
    sg = att.get("scheduler_geometry")
    reports: Any = None
    if isinstance(sg, dict):
        reports = sg.get("worker_reports")
    elif isinstance(sg, list):
        reports = sg
    if not reports:
        reports = att.get("worker_reports")
    return [r for r in (reports or []) if isinstance(r, dict)]


def kv_facts_from_reports(reports: Sequence[Mapping[str, Any]], max_model_len: int
                          ) -> Dict[str, Any]:
    """The attested KV capacity of a launch from the workers' scheduler reports
    (the ``scheduler_geometry`` list of the attestation): a UNIFORM cache
    reports ``kv_num_gpu_blocks x kv_block_size`` tokens; a HYBRID one the
    allocator's ``kv_max_concurrency`` (x L for a tokens-equivalent). The
    minimum across workers is the launch's."""
    if not reports:
        return {"kv_tokens": None, "kv_kind": "unreported"}
    toks: List[int] = []
    kinds = set()
    concs: List[float] = []
    for rep in reports:
        groups = rep.get("kv_num_cache_groups")
        if groups == 1 and isinstance(rep.get("kv_num_gpu_blocks"), int) \
                and isinstance(rep.get("kv_block_size"), int):
            toks.append(int(rep["kv_num_gpu_blocks"]) * int(rep["kv_block_size"]))
            kinds.add("uniform_blocks_x_block_size")
            concs.append(toks[-1] / max_model_len)
        elif isinstance(rep.get("kv_max_concurrency"), (int, float)):
            mc = float(rep["kv_max_concurrency"])
            toks.append(int(mc * max_model_len))
            kinds.add("hybrid_max_concurrency_x_L")
            concs.append(mc)
    if not toks:
        return {"kv_tokens": None, "kv_kind": "unreadable"}
    return {"kv_tokens": min(toks), "kv_kind": ",".join(sorted(kinds)),
            "kv_max_concurrency_at_L": min(concs)}


def select_steered_cell(objects_dir: Optional[Path],
                        preference: Sequence[str] = PSR_CELL_PREFERENCE) -> Optional[str]:
    """The steered arm's cell: the first preferred psr_kl cell whose object
    artifact (``<dir>/<cell>/object.*``) exists in the model's bank; ``None``
    when the bank is absent or holds none (passthrough-only sweep, recorded).

    This answers WHICH CELL only, and a glob is the right instrument for that.
    What that cell then SERVES is :func:`served_projection`'s arm-aware
    resolution, which is deliberately the narrower one: a dir holding a blob
    under a name the arm does not serve gets picked here and refused there,
    loudly, rather than fingerprinted as if it were part of the object."""
    if objects_dir is None or not Path(objects_dir).is_dir():
        return None
    for cell in preference:
        d = Path(objects_dir) / cell
        if d.is_dir() and any(p.is_file() for p in d.glob("object.*")):
            return cell
    return None


#: The ROLES ``lens_steer.export_worker_registry`` copies out of a bank cell
#: dir, in the order it copies them: the artifact (always) and its sidecar
#: (when the cell declares one). These are the components a standup SERVES, and
#: therefore the vocabulary both sides of the attestation speak — the bank
#: names them ``object.pt`` / ``object.json`` and the registry names them
#: ``<cell>/artifact.bin`` / ``<cell>/sidecar.bin``, so a comparison keyed on
#: FILENAMES could never line the two up (E55 #4).
SERVED_ROLES = ("artifact", "sidecar")


def served_projection(objects_dir: Optional[Path], cell: Optional[str]
                      ) -> Dict[str, Path]:
    """The exporter's EXPECTED SERVED PROJECTION of a bank cell dir:
    ``{role: path}`` over exactly the files
    ``lens_steer.export_worker_registry`` would copy into the worker registry —
    the artifact under ``role="artifact"`` and, when the cell declares one, its
    sidecar under ``role="sidecar"``. Never the whole directory.

    The selection is the standup's OWN, not a second one:

    * the artifact is ``defense_cell.object_path_for`` (directory keyed on the
      cell id, filename on the ARM — ``object.safetensors`` for the
      SteeringVector-container arms, ``object.pt`` for torch blobs, the legacy
      ``lora_r1_all`` dir honoured), reached through
      ``tinycap.object_identity.bank_artifact_path``. ``registry_adapter``
      cross-checks the registry's ``object_path`` against that very function
      and refuses a disagreement, so this is the path that gets copied;
    * the sidecar is ``manifest.sidecar_path(artifact)`` (``<artifact>.json``),
      declared on the same rule ``registry_adapter._object_entry`` uses —
      REQUIRED for the SteeringVector-container arms (their loader reads it)
      and declared-when-present otherwise.

    Why a projection and not a ``object.*`` glob (E55 #4): a glob makes
    COMPLETENESS undefined. A bank dir may legitimately hold ``object.*`` files
    the export does not copy (a stale ``object.pt`` beside the
    ``object.safetensors`` an arm actually serves, a scratch blob), so a
    fingerprint over the whole dir cannot be required to match what was loaded,
    and the check degrades to subset membership — under which an expected
    component that simply VANISHES before the export attests fine. Fingerprint
    the projection instead and "every expected component was served, and
    nothing else was" is a statement that can be enforced."""
    if not cell:
        return {}
    if objects_dir is None:
        raise SystemExit(f"[sweep] --steered-cell {cell}: no objects dir to resolve the served "
                         "object in — a steered measurement cannot be identified by a cell "
                         "name alone (E50 #7)")
    from sisteer.capeval import manifest as cm
    from sisteer.capeval.tinycap.object_identity import bank_artifact_path
    from sisteer.eval.defense_cell import arm_of
    from sisteer.methods import VECTOR_SAFETENSOR_ARMS

    try:
        artifact = bank_artifact_path(objects_dir, cell)
    except ValueError as e:
        raise SystemExit(f"[sweep] --steered-cell {cell}: {e} — a steered measurement cannot "
                         "be identified by a cell name alone (E50 #7)") from e
    if not artifact.is_file():
        raise SystemExit(f"[sweep] --steered-cell {cell}: no artifact at {artifact} (the arm's "
                         "own filename, defense_cell.object_path_for) — a steered measurement "
                         "cannot be identified by a cell name alone (E50 #7)")
    proj: Dict[str, Path] = {"artifact": artifact}
    sidecar = cm.sidecar_path(artifact)
    if arm_of(cell) in VECTOR_SAFETENSOR_ARMS:
        # the arm's loader reads it, so the export declares it whether or not
        # it is there — and dies reading it if it is not. Refuse HERE instead:
        # the same refusal, before a GPU minute.
        if not sidecar.is_file():
            raise SystemExit(
                f"[sweep] --steered-cell {cell}: arm {arm_of(cell)!r} serves a sidecar "
                f"({sidecar}) and there is none — the export would read it and fail; refusing "
                "before the standup (E55 #4)")
        proj["sidecar"] = sidecar
    elif sidecar.is_file():
        proj["sidecar"] = sidecar
    return proj


def object_identity(objects_dir: Optional[Path], cell: Optional[str]
                    ) -> Optional[Dict[str, str]]:
    """WHICH fitted object the steered arm serves: ``{role: sha256}`` over the
    exporter's expected served projection (:func:`served_projection`) — the
    artifact and, when the cell declares one, its sidecar. ``None`` for a
    passthrough-only run (no steered cell).

    The artifact's entry is exactly the digest the standup computes at export
    (``lens_steer.export_worker_registry`` hashes the bytes it copies and, when
    a caller binds it, compares them against ``--expect-object-sha256``); the
    sidecar is included because the SERVED intervention hash is taken over
    (spec, artifact, sidecar), so a sidecar-only edit changes what is served
    while the artifact bytes stand still.

    Keyed by ROLE, not by filename (E55 #4): the bank calls these files
    ``object.pt`` / ``object.json`` and the registry copies them to
    ``<cell>/artifact.bin`` / ``<cell>/sidecar.bin``, so the role is the only
    name both sides share — and without it
    :func:`attest_object_identity` cannot say WHICH component went missing.

    Why it has to be a fingerprint / partition input (E50 #7): these probe
    standups pass ``bring_up(require_object_digest=False)`` — they file timings,
    never rows, and have no preflight to bind to — so nothing else in the run
    ties a measurement to the bytes it measured. The identity of the cell is its
    NAME, and a bank path is re-fitted in place all the time; two objects under
    one name are two measurements, and a resume that reuses the first one's
    points under the second's is the finding."""
    if not cell:
        return None
    proj = served_projection(objects_dir, cell)
    return {role: hashlib.sha256(p.read_bytes()).hexdigest() for role, p in proj.items()}


def object_identity_id(identity: Optional[Mapping[str, str]]) -> str:
    """A 16-hex label for :func:`object_identity` — ``"none"`` when there is no
    steered object, so the label is always sayable in a filename or a row."""
    if not identity:
        return "none"
    return hashlib.sha256(json.dumps(dict(identity), sort_keys=True).encode("utf-8")
                          ).hexdigest()[:16]


#: Where a standup writes the worker registry it exported
#: (``capctrl_serving_up``: ``reg_dir = out / "lens_registry"``) — the COPIES of
#: the object blobs the server is then launched against.
STANDUP_REGISTRY_SUBDIR = "lens_registry"
#: The re-attested identity, recorded on every measurement (E54 #5).
ATTESTED_OBJECT_FIELD = "steered_object_attested"


def loaded_object_identity(standup_out: Path, harness_cell: Optional[str]
                           ) -> Optional[Dict[str, Any]]:
    """WHICH BYTES a live standup actually loaded, read back off the worker
    registry it exported: ``{role: sha256}`` over the artifact (and sidecar)
    COPIES under ``<standup out>/lens_registry/``, plus the identity the export
    recorded for them and, under ``blobs``, the registry-relative path each
    role was read from.

    These copies ARE what the server serves — ``export_worker_registry`` writes
    them, rereads them to hash them, and the launch points the worker at this
    directory — so hashing them attests what was SERVED. It is not a second
    read of the bank path :func:`object_identity` read: a re-fit that landed
    between the two reads shows up as a difference, which is the whole point
    (E54 #5).

    ``None`` when the standup served no fitted leg (no ``harness_cell``, or
    ``none``). Every other unreadable shape is a REFUSAL: a run that cannot say
    which bytes it loaded cannot honestly label the timings it is about to
    file."""
    if not harness_cell or harness_cell == "none":
        return None
    from sisteer.capeval.serving.lens_steer import REGISTRY_INDEX

    reg_dir = Path(standup_out) / STANDUP_REGISTRY_SUBDIR
    index_path = reg_dir / REGISTRY_INDEX
    index = _read_json(index_path)
    if not isinstance(index, dict) or not isinstance(index.get("entries"), list):
        raise SystemExit(
            f"[sweep] {index_path}: the standup's exported worker registry is missing or "
            "unreadable, so the bytes it loaded cannot be attested — refusing to file a "
            "measurement that cannot name what it measured (E54 #5)")
    entry = next((e for e in index["entries"] if e.get("cell_id") == harness_cell), None)
    if entry is None:
        raise SystemExit(
            f"[sweep] {index_path} has no entry for the served cell {harness_cell!r} "
            f"(entries: {sorted(str(e.get('cell_id')) for e in index['entries'])}) — the "
            "standup did not load the object this measurement claims (E54 #5)")
    if not entry.get("artifact"):
        raise SystemExit(
            f"[sweep] {index_path}: entry {harness_cell!r} records no artifact blob — a "
            "fitted leg with no bytes cannot be attested (E54 #5)")
    digests: Dict[str, str] = {}
    blobs: Dict[str, str] = {}
    # keyed by the ROLE the export copied the blob under, which is the field
    # name it recorded it under — the registry's filenames are its own
    # (``<cell>/artifact.bin``) and never the bank's (E55 #4)
    for role in SERVED_ROLES:
        rel = entry.get(role)
        if not rel:
            continue
        blob = reg_dir / rel
        if not blob.is_file():
            raise SystemExit(
                f"[sweep] {blob}: the registry names this blob for {harness_cell!r} but it is "
                "not on disk — the bytes the server loaded cannot be re-hashed (E54 #5)")
        digests[role] = hashlib.sha256(blob.read_bytes()).hexdigest()
        blobs[role] = str(rel)
    recorded16 = entry.get("artifact_sha16")
    got16 = digests["artifact"][:16]
    if recorded16 and recorded16 != got16:
        raise SystemExit(
            f"[sweep] {index_path}: entry {harness_cell!r} records artifact_sha16 "
            f"{recorded16} but the blob on disk hashes to {got16} — the registry disagrees "
            "with its own bytes; refusing (E54 #5)")
    return {"cell_id": harness_cell, "registry": str(index_path), "digests": digests,
            "blobs": blobs, "artifact_sha256": digests["artifact"],
            "artifact_sha16": recorded16,
            "intervention_hash": entry.get("artifact_intervention_hash")}


def attest_object_identity(standup_out: Path, harness_cell: Optional[str],
                           expected: Optional[Mapping[str, str]], *, where: str
                           ) -> Optional[Dict[str, Any]]:
    """RE-ATTEST, after the standup, that the bytes it LOADED are the bytes the
    measurement's fingerprint named — and return the attested identity so the
    measurement can record it.

    E54 #5: :func:`object_identity` hashes the bank path BEFORE the standup, and
    the standup then runs with ``require_object_digest=False`` (the library-only
    seam, deliberately kept), so nothing tied the fingerprint to the bytes
    actually served. A bank path re-fitted in place between the two — which is
    exactly the event ``object_identity`` exists to detect — produced either B's
    measurements filed under A's identity or, worse, a hybrid identity that
    never existed on any disk.

    The check is ROLE-AWARE and COMPLETE (E55 #4). Both sides are keyed by the
    roles the export copies — :data:`SERVED_ROLES`, artifact and sidecar — so
    artifact is compared against artifact and sidecar against sidecar, and
    every component the fingerprint expects to be served must have been loaded
    with an equal digest, with nothing loaded beside them. MISSING, EXTRA and
    MISMATCHED each refuse, naming the role and both digests.

    It used to be directional — a subset test over an unordered SET of digests,
    which the whole-directory fingerprint forced: a bank dir can hold
    ``object.*`` files the export does not copy, so equality would have refused
    honest runs. The cost was the hole this closes: a sidecar that VANISHED
    between the fingerprint and the export left the registry serving the
    artifact alone, every loaded digest was still accounted for, and the
    attestation passed — filing timings under a ``measurement_id`` whose
    fingerprint named a sidecar that was never served, while the sidecar's
    bytes are part of the served intervention hash. The fingerprint now names
    the exporter's expected served PROJECTION (:func:`served_projection`)
    rather than the directory, which is what makes completeness definable at
    all; an unserved extra file in the bank is simply not in it.

    This does NOT move ``measurement_id`` off the fingerprint: the fingerprint
    is what a resume keys on and must be identical across the phase. The
    attestation confirms it, and is recorded beside the measurement under
    :data:`ATTESTED_OBJECT_FIELD`."""
    loaded = loaded_object_identity(standup_out, harness_cell)
    if loaded is None:
        if expected:
            raise SystemExit(
                f"[sweep] {where}: the measurement fingerprint names a fitted object "
                f"({json.dumps(dict(expected), sort_keys=True)}) but the standup exported no "
                f"lens leg for {harness_cell!r} — refusing to file timings whose object "
                "cannot be attested (E54 #5)")
        return None
    if not expected:
        raise SystemExit(
            f"[sweep] {where}: the standup loaded a fitted object "
            f"({json.dumps(loaded['digests'], sort_keys=True)}) that the measurement "
            "fingerprint does not name — refusing to file timings under an identity that "
            "does not describe them (E54 #5)")
    want = dict(expected)
    got = dict(loaded["digests"])
    stale = sorted(set(want) - set(SERVED_ROLES))
    if stale:
        raise SystemExit(
            f"[sweep] {where}: the measurement fingerprint names object components "
            f"{stale} that are not served roles ({list(SERVED_ROLES)}) — it was written by an "
            "older sweep that fingerprinted the whole bank directory by filename, and cannot "
            "be compared role-wise. Run into a fresh --results-root (E55 #4).")
    faults = []
    for role in SERVED_ROLES:
        exp, act = want.get(role), got.get(role)
        if exp == act:
            continue
        if act is None:
            faults.append(f"  MISSING {role}: fingerprinted {exp}, the standup served none "
                          f"(the export copied no {role} for this cell)")
        elif exp is None:
            faults.append(f"  EXTRA {role}: the standup served {act} "
                          f"({loaded['blobs'].get(role)}), which the fingerprint does not name")
        else:
            faults.append(f"  MISMATCHED {role}: fingerprinted {exp}, loaded {act} "
                          f"({loaded['blobs'].get(role)})")
    if faults:
        raise SystemExit(
            f"[sweep] {where}: the bytes this standup LOADED are not the bytes the "
            f"measurement was fingerprinted on — refusing to file B's timings under A's "
            f"identity (E54 #5 / E55 #4).\n"
            f"  fingerprinted (bank {harness_cell!r}, before standup): "
            f"{json.dumps(want, sort_keys=True)}\n"
            f"  loaded (registry {loaded['registry']}): "
            f"{json.dumps(got, sort_keys=True)}\n"
            + "\n".join(faults) + "\n"
            "The object under that cell name changed between the fingerprint and the "
            "bring-up — re-fitted in place, or a component of it removed. A sidecar counts: "
            "the served intervention hash is taken over (spec, artifact, sidecar). Run into a "
            "fresh --results-root against the object you mean.")
    return loaded


def resolve_serving_config(ms: ModelServe, repo: Path, run_dir: Path) -> Path:
    """The two-SI config the standup builds from: the repo's, when it carries
    the pin; a PINNED COPY under ``run_dir`` when it leaves ``model_revision``
    null (llama3b's) — the standup resolves the tokenizer, the HF config, the
    chat template and the engine's ``--model`` from the config's
    ``model``/``model_revision``, so the copy is what makes the launch the
    pinned one. A config naming a different model or a different pin refuses:
    the table row and the config must agree, never be papered over."""
    import yaml

    cfg_path = Path(repo) / ms.cfg
    if not cfg_path.is_file():
        raise FileNotFoundError(f"{ms.key}: serving config {cfg_path} is missing")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if cfg.get("model") != ms.hf:
        raise ValueError(f"{ms.key}: {ms.cfg} serves model {cfg.get('model')!r}, the sweep "
                         f"table says {ms.hf!r} — fix whichever is wrong, do not guess")
    rev = cfg.get("model_revision")
    if rev in (None, ""):
        pinned = Path(run_dir) / f"{cfg_path.stem}__pinned_{ms.rev[:12]}.yaml"
        pinned.parent.mkdir(parents=True, exist_ok=True)
        cfg["model_revision"] = ms.rev
        header = (f"# PINNED COPY of {ms.cfg} written by scripts/served_decode_sweep.py: the "
                  f"repo config leaves model_revision null; the sweep serves {ms.hf} @ "
                  f"{ms.rev} (the promoted object's fit pin).\n")
        pinned.write_text(header + yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        return pinned
    if rev != ms.rev:
        raise ValueError(f"{ms.key}: {ms.cfg} pins model_revision {rev!r}, the sweep table "
                         f"says {ms.rev!r} — the row and the config must agree")
    return cfg_path


def hf_snapshot_relpath(ms: ModelServe) -> str:
    """The HF cache path of the pinned snapshot, relative to ``HF_HOME``."""
    return f"hub/models--{ms.hf.replace('/', '--')}/snapshots/{ms.rev}"


def fit_repeats(count_fn: Callable[[int], int], target_tokens: int,
                max_iter: int = 12) -> Tuple[int, int]:
    """The number of filler repeats whose rendered prompt tokenizes to at most
    ``target_tokens`` and as close to it as the filler granularity allows
    (``count_fn(k)`` = the token count of k repeats, monotone in k). A handful
    of tokenizer round-trips: one guess, linear refinement, then a step-down
    to land at or under the target."""
    if target_tokens < 1:
        raise ValueError("target_tokens must be >= 1")
    k = max(1, target_tokens // 30)
    c = count_fn(k)
    if c <= 0:
        raise RuntimeError("count_fn returned a non-positive token count")
    for _ in range(max_iter):
        if c <= target_tokens and count_fn(k + 1) > target_tokens:
            return k, c
        per = max(1.0, (c - count_fn(1)) / k) if k > 1 else max(1.0, float(c))
        k_new = max(1, int(round(k + (target_tokens - c) / per)))
        if k_new == k:
            k_new = k + (1 if c < target_tokens else -1)
        k = max(1, k_new)
        c = count_fn(k)
    while c > target_tokens and k > 1:
        k -= 1
        c = count_fn(k)
    return k, c


def parse_prometheus(text: str) -> Dict[str, List[Tuple[Dict[str, str], float]]]:
    """A Prometheus text exposition -> ``{metric: [(labels, value), ...]}``."""
    out: Dict[str, List[Tuple[Dict[str, str], float]]] = {}
    for ln in (text or "").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        m = re.match(r'^([A-Za-z_:][A-Za-z0-9_:]*)(\{[^}]*\})?\s+(\S+)', ln)
        if not m:
            continue
        name, labels_s, val_s = m.group(1), m.group(2), m.group(3)
        try:
            val = float(val_s)
        except ValueError:
            continue
        labels: Dict[str, str] = {}
        if labels_s:
            for lm in re.finditer(r'([A-Za-z_][A-Za-z0-9_]*)="((?:[^"\\]|\\.)*)"', labels_s):
                labels[lm.group(1)] = lm.group(2)
        out.setdefault(name, []).append((labels, val))
    return out


def metric_sum(parsed: Mapping[str, Sequence[Tuple[Mapping[str, str], float]]],
               *names: str) -> Optional[float]:
    """The sum over label sets of the first metric family present among
    ``names`` (vLLM renamed a few between releases), or ``None``."""
    for n in names:
        if n in parsed:
            return float(sum(v for _, v in parsed[n]))
    return None


def hist_buckets(parsed: Mapping[str, Sequence[Tuple[Mapping[str, str], float]]],
                 family: str) -> List[Tuple[float, float]]:
    """A histogram's cumulative buckets ``[(le, count)]`` summed over other
    labels, sorted by ``le`` (``+Inf`` last)."""
    acc: Dict[float, float] = {}
    for labels, v in parsed.get(family + "_bucket", ()):
        le = labels.get("le")
        if le is None:
            continue
        le_f = math.inf if le in ("+Inf", "inf", "Inf") else float(le)
        acc[le_f] = acc.get(le_f, 0.0) + v
    return sorted(acc.items())


def hist_quantile(delta_buckets: Sequence[Tuple[float, float]], q: float) -> Optional[float]:
    """The q-quantile of a histogram given its cumulative bucket DELTAS over a
    window (linear interpolation within the bucket, Prometheus-style)."""
    if not delta_buckets:
        return None
    total = delta_buckets[-1][1]
    if total <= 0:
        return None
    rank = q * total
    prev_le, prev_c = 0.0, 0.0
    for le, c in delta_buckets:
        if c >= rank:
            if math.isinf(le):
                return prev_le
            if c == prev_c:
                return le
            return prev_le + (le - prev_le) * (rank - prev_c) / (c - prev_c)
        prev_le, prev_c = le, c
    return None


def percentile(xs: Sequence[float], q: float) -> Optional[float]:
    """Nearest-rank percentile (q in [0, 1]) of a non-empty sequence."""
    if not xs:
        return None
    ys = sorted(xs)
    idx = min(len(ys) - 1, max(0, int(math.ceil(q * len(ys))) - 1))
    return float(ys[idx])


def knee_batch(points: Sequence[Tuple[int, float]], fraction: float = KNEE_FRACTION
               ) -> Optional[int]:
    """The KNEE: the smallest B whose tok/s is at least ``fraction`` of the
    maximum observed across the ladder."""
    pts = [(int(b), float(t)) for b, t in points if t is not None]
    if not pts:
        return None
    mx = max(t for _, t in pts)
    for b, t in sorted(pts):
        if t >= fraction * mx:
            return b
    return None


def summarize_rows(rows: Sequence[Mapping[str, Any]], *, budget: Optional[int] = None,
                   knee_fraction: float = KNEE_FRACTION, old_s: int = OLD_S,
                   partial: bool = False, missing: Sequence[Any] = (),
                   windows_requested: Sequence[int] = (),
                   ladder: Sequence[int] = BATCH_LADDER,
                   expected_arms: Optional[Sequence[str]] = None,
                   measurement_id: Optional[str] = None) -> Dict[str, Any]:
    """``sweep_summary.json``: per (window, arm, S) the throughput-vs-B table,
    the knee, tok/s at the knee, tok/s at the old S = 24 (when that rung was
    run), the speedup, duty at the knee, and the preemption tally — from the
    POINT rows; the window-level rows (``window_refused`` / ``window_skipped``)
    are listed explicitly, and every requested window without a measured group
    is named in ``windows_not_measured``.

    COMPLETENESS (review R3): a group's EXPECTED rungs are the batch ladder at
    its actual S; a group missing any of them is ``complete=False`` and gets NO
    definitive knee / speedup — only ``knee_provisional_B`` over the rungs it
    has — and the summary is ``partial``. A window whose expected arms are not
    all present (``expected_arms``, default: every arm seen anywhere in the
    rows) is listed in ``arms_missing``. Rows of one (window, arm) at two
    different S are two groups (``<L>:<arm>@S<S>``), never one curve.
    ONE MEASUREMENT PER SUMMARY (review S2): with ``measurement_id`` every row
    of another id — points and window events alike — is excluded; without it
    the rows must all carry one id (or none: a pre-fingerprint store), else
    ``ValueError`` — the analyzer partitions by id first
    (:func:`partition_by_measurement`). Works on ``sweep.jsonl`` alone, so the
    analyzer can rebuild a summary."""
    ids = {r.get("measurement_id") for r in rows}
    if measurement_id is not None:
        rows = [r for r in rows if r.get("measurement_id") == measurement_id]
    elif len(ids) > 1:
        raise ValueError(f"rows carry {len(ids)} measurement ids {sorted(map(str, ids))} — "
                         "summarize one measurement at a time (partition_by_measurement)")
    points = [r for r in rows if is_point_row(r)]
    refusals = [r for r in rows if r.get("kind") == ROW_KIND_REFUSED]
    skipped = [r for r in rows if r.get("kind") == ROW_KIND_SKIPPED]
    groups: Dict[Tuple[int, str, int], Dict[int, Mapping[str, Any]]] = {}
    for r in points:
        key = (int(r["window_L"]), str(r["arm"]), int(r.get("S") or 0))
        groups.setdefault(key, {})[int(r["B"])] = r
    s_per_la: Dict[Tuple[int, str], set] = {}
    for (L, arm, S) in groups:
        s_per_la.setdefault((L, arm), set()).add(S)
    arms_seen = sorted({arm for _, arm, _ in groups})
    exp_arms = sorted(set(expected_arms)) if expected_arms else (arms_seen or [ARM_PASSTHROUGH])
    out_groups: Dict[str, Any] = {}
    incomplete: List[str] = []
    for (L, arm, S), by_b in sorted(groups.items()):
        gkey = f"{L}:{arm}" if len(s_per_la[(L, arm)]) == 1 else f"{L}:{arm}@S{S}"
        table = []
        for b in sorted(by_b):
            r = by_b[b]
            table.append({k: r.get(k) for k in (
                "B", "tok_s", "req_s", "lat_p50_s", "lat_p95_s", "ttft_p50_s_hist",
                "running_mean", "running_max", "preemptions_delta", "sm_mean", "sm_p10",
                "engine_gen_tok_s", "n_completed", "n_errors")})
        expected_B = sorted(set(batch_ladder(S, ladder))) if S else sorted(by_b)
        missing_B = [b for b in expected_B if b not in by_b]
        complete = not missing_B
        pts = [(b, by_b[b].get("tok_s")) for b in by_b]
        knee_prov = knee_batch(pts, knee_fraction)
        max_tok = max((t for _, t in pts if t is not None), default=None)
        at_prov = by_b.get(knee_prov) if knee_prov is not None else None
        at_old = by_b.get(old_s)
        speedup_prov = None
        if at_prov is not None and at_old is not None and at_old.get("tok_s"):
            speedup_prov = float(at_prov["tok_s"]) / float(at_old["tok_s"])
        any_row = next(iter(by_b.values()))
        g: Dict[str, Any] = {
            "window_L": L, "arm": arm, "S": any_row.get("S"), "S_target": any_row.get("S_target"),
            "window_labels": list(any_row.get("window_labels") or ()),
            "cell_id": any_row.get("cell_id"), "table": table,
            "expected_B": expected_B, "missing_B": missing_B, "complete": complete,
            "max_tok_s": max_tok,
            "knee_provisional_B": knee_prov,
            "tok_s_at_knee_provisional": (at_prov.get("tok_s") if at_prov else None),
            f"speedup_provisional_vs_B{old_s}": speedup_prov,
            f"tok_s_at_B{old_s}": (at_old.get("tok_s") if at_old else None),
            "preemptions_total": sum(int(r.get("preemptions_delta") or 0)
                                     for r in by_b.values()),
            "n_points": len(by_b), "n_expected": len(expected_B),
        }
        if complete:
            g.update({"knee_B": knee_prov,
                      "tok_s_at_knee": (at_prov.get("tok_s") if at_prov else None),
                      "duty_at_knee": (at_prov.get("sm_mean") if at_prov else None),
                      "lat_p50_at_knee_s": (at_prov.get("lat_p50_s") if at_prov else None),
                      f"speedup_knee_vs_B{old_s}": speedup_prov})
        else:
            g.update({"knee_B": None, "tok_s_at_knee": None, "duty_at_knee": None,
                      "lat_p50_at_knee_s": None, f"speedup_knee_vs_B{old_s}": None})
            incomplete.append(gkey)
        out_groups[gkey] = g
    measured = sorted({L for L, _, _ in groups})
    requested = [int(L) for L in windows_requested] or list(measured)
    arms_missing = {}
    for L in measured:
        present = {arm for (l, arm, _) in groups if l == L}
        gap = [a for a in exp_arms if a not in present]
        if gap:
            arms_missing[str(L)] = gap
    ev_keys = ("window_L", "eval_key", "window_labels", "S_tried", "S_target", "attempt",
               "status", "error", "reason", "last_status", "job_id", "node", "t")
    not_measured = [L for L in requested if L not in measured]
    return {"budget": budget, "knee_fraction": knee_fraction, "old_S": old_s,
            "measurement_id": (measurement_id if measurement_id is not None
                               else (next(iter(ids)) if ids else None)),
            "windows": measured,
            "arms": arms_seen, "expected_arms": exp_arms,
            "groups": out_groups, "n_rows": len(points),
            "groups_incomplete": incomplete, "arms_missing": arms_missing,
            "partial": bool(partial) or bool(incomplete) or bool(arms_missing),
            "complete": (not partial and not incomplete and not arms_missing
                         and not not_measured),
            "missing_points": [list(m) for m in missing],
            "windows_requested": requested,
            "windows_not_measured": not_measured,
            "window_refusals": [{k: r.get(k) for k in ev_keys if k in r} for r in refusals],
            "skipped_windows": [{k: r.get(k) for k in ev_keys if k in r} for r in skipped]}


def in_window_stats(records: Sequence[Tuple[float, float, int, bool, Optional[str]]],
                    w0: float, w1: float) -> Dict[str, Any]:
    """Client-side aggregates over the requests that COMPLETED inside
    ``[w0, w1]``: tokens/s, requests/s, latency percentiles, error count."""
    done = [r for r in records if w0 <= r[1] <= w1]
    ok = [r for r in done if r[3]]
    dur = max(1e-9, w1 - w0)
    lats = [r[1] - r[0] for r in ok]
    toks = sum(int(r[2]) for r in ok)
    return {"n_completed": len(ok), "n_errors": len(done) - len(ok),
            "completion_tokens": toks, "tok_s": toks / dur, "req_s": len(ok) / dur,
            "lat_p50_s": percentile(lats, 0.50), "lat_p95_s": percentile(lats, 0.95),
            "lat_mean_s": (sum(lats) / len(lats) if lats else None),
            "first_error": next((r[4] for r in done if not r[3]), None)}


def parse_dmon_line(line: str) -> Optional[Tuple[int, int]]:
    """``nvidia-smi dmon -s u`` data row -> (gpu index, sm%); ``None`` for
    headers / blanks / a '-' reading."""
    parts = line.split()
    if len(parts) < 2 or parts[0].startswith("#"):
        return None
    try:
        idx = int(parts[0])
        sm = int(parts[1])
    except ValueError:
        return None
    return idx, sm


def gpu_selector() -> Optional[str]:
    """The GPU ``nvidia-smi -i`` should watch: the first entry of Slurm's
    CUDA_VISIBLE_DEVICES (read, never set — hard rule 3), else ``None`` (all)."""
    v = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not v:
        return None
    return v.split(",")[0].strip() or None


def shell_env_lines(ms: ModelServe, repo: Path) -> List[str]:
    """``--phase env``: the model's pins as ``KEY=value`` lines the sbatch
    ``eval``s for its fail-fast guards (one canonical table, read by bash)."""
    return [f"GEOM_MODEL_KEY={shlex.quote(ms.key)}",
            f"GEOM_HF_ID={shlex.quote(ms.hf)}",
            f"GEOM_HF_REV={shlex.quote(ms.rev)}",
            f"GEOM_CONFIG={shlex.quote(str(Path(repo) / ms.cfg))}",
            f"GEOM_BANK={shlex.quote(ms.bank)}",
            f"GEOM_HF_SNAPSHOT_REL={shlex.quote(hf_snapshot_relpath(ms))}"]


#: the documented placeholder for the cluster's per-user shared root in generated
#: basis strings (hf_env.sh's $SHARED: the vast root on cluster A, the MooseFS one
#: on cluster B) — a pasted ``source=`` must carry no username or absolute path.
SHARED_PLACEHOLDER = "<SHARED>"
_SHARED_ROOT_RE = re.compile(r"^/workspace(?:-vast)?/[^/]+(?=/|$)")


def shared_placeholder(path: str) -> str:
    """``/<shared-root>/<user>/exp/...`` -> ``<SHARED>/exp/...``; any other path
    is returned unchanged (a grading host results root is not a cluster path)."""
    return _SHARED_ROOT_RE.sub(SHARED_PLACEHOLDER, str(path), count=1)


def paste_record(*, model_key: str, bracket: Mapping[str, Any], max_model_len: int,
                 gpu_mem_util: float, job_id: Optional[str], measured_on: str,
                 results_dir: str, node: str) -> Optional[Dict[str, Any]]:
    """The ``ModelBudget`` record (as a dict) the analyzer formats into the
    geometry.py paste block; ``None`` when no probe passed."""
    lp = bracket.get("largest_passing")
    if not lp:
        return None
    S = int(lp["max_num_seqs"])
    kv_kind = lp.get("kv_kind")
    # provenance is the PROBE's (review R2): a resumed walk's largest passing
    # probe may come from an earlier job / node than the one writing the doc
    job_id = lp.get("job_id") or job_id
    node = lp.get("node") or node
    return {
        "model_key": model_key,
        "batched_token_budget": int(lp["product"]),
        "certified": False,
        "basis": {
            "method": ("largest S x L whose launch passed vllm_server.kv_coverage_problem at "
                       f"readiness on one H200 at gpu_memory_utilization {gpu_mem_util:.2f} "
                       "(scripts/served_decode_sweep.py --phase budget: the ladder walked "
                       "upward to the first refusal, then one bisection)"),
            "max_model_len": int(max_model_len), "max_num_seqs": S,
            "gpu_memory_utilization": float(gpu_mem_util),
            "kv_tokens_reported": lp.get("kv_tokens"),
            "probe_job_id": job_id, "measured_on": measured_on,
            "source": f"{shared_placeholder(results_dir)}/budget.json (node {node})",
            "first_refused": bracket.get("first_refused"),
            "note": (f"kv facts: {kv_kind}; log max-concurrency "
                     f"{lp.get('log_max_concurrency_x')}x at {max_model_len}; "
                     f"probes: {bracket.get('n_probes')}; wall "
                     f"{'reached' if bracket.get('wall_reached') else 'NOT reached (ladder top)'}"),
        },
    }


# --------------------------------------------------------------------------- #
# Runtime (GPU / cluster only; every import lazy)                              #
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class SweepContext:
    ms: ModelServe
    repo: Path
    results_dir: Path
    figs_dir: Path
    config_path: Path
    objects_dir: Optional[Path]
    steered_cell: Optional[str]
    gpu_mem_util: str
    ready_timeout_s: int
    manifest: str
    deadline_epoch: Optional[float]
    deadline_margin_s: float
    job_id: Optional[str]
    node: str
    #: hybrid-recurrent target (is_hybrid_model): per-slot state caps S per window
    hybrid: bool = False

    @property
    def seqs_cap(self) -> int:
        return seqs_cap(self.hybrid)

    def standup_cell(self) -> str:
        return self.steered_cell or "none"

    def time_left_s(self) -> Optional[float]:
        if self.deadline_epoch is None:
            return None
        return self.deadline_epoch - self.deadline_margin_s - time.time()

    def have_time(self, needed_s: float) -> bool:
        left = self.time_left_s()
        return True if left is None else left >= needed_s


def _say(msg: str) -> None:
    print(f"[sweep {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=False, default=str))
    tmp.replace(path)


def _read_json(path: Path) -> Optional[Any]:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _nvidia_smi(*args: str) -> Optional[str]:
    try:
        return subprocess.run(["nvidia-smi", *args], capture_output=True, text=True,
                              timeout=30).stdout
    except Exception:
        return None


def wait_gpu_free(threshold_mib: int = 4096, timeout_s: float = 180.0) -> Optional[int]:
    """Block until the watched GPU's used memory falls under ``threshold_mib``
    (a torn-down engine releases its allocation a few seconds after exit) or
    the timeout; returns the last reading (MiB) or ``None`` when unreadable."""
    sel = gpu_selector()
    args = ["--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    if sel is not None:
        args += ["-i", sel]
    t0 = time.monotonic()
    last = None
    while time.monotonic() - t0 < timeout_s:
        out = _nvidia_smi(*args)
        if out is None:
            return None
        try:
            last = int(float(out.strip().splitlines()[0]))
        except (ValueError, IndexError):
            return None
        if last < threshold_mib:
            return last
        time.sleep(3.0)
    _say(f"WARNING: GPU still holds {last} MiB after {timeout_s:.0f}s; continuing")
    return last


class MetricsSampler:
    """Polls the vLLM server's ``/metrics`` (Bearer-guarded like every other
    path) every ``period_s`` on a thread; ``window(w0, w1)`` reduces the
    samples inside a monotonic-time window."""

    FAMILY_TTFT = "vllm:time_to_first_token_seconds"

    def __init__(self, host: str, port: int, api_key: str, period_s: float = METRICS_PERIOD_S):
        self.url = f"http://{host}:{port}/metrics"
        self.api_key = api_key
        self.period_s = period_s
        self.samples: List[Tuple[float, Dict[str, Any]]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="metrics", daemon=True)
        self.errors = 0

    def fetch(self) -> Optional[Dict[str, Any]]:
        import urllib.request
        req = urllib.request.Request(self.url, headers={"Authorization": f"Bearer {self.api_key}"})
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                text = r.read().decode("utf-8", "replace")
        except Exception:
            self.errors += 1
            return None
        p = parse_prometheus(text)
        return {
            "running": metric_sum(p, "vllm:num_requests_running"),
            "waiting": metric_sum(p, "vllm:num_requests_waiting"),
            "kv_usage": metric_sum(p, "vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"),
            "preemptions": metric_sum(p, "vllm:num_preemptions_total"),
            "gen_tokens": metric_sum(p, "vllm:generation_tokens_total"),
            "prompt_tokens": metric_sum(p, "vllm:prompt_tokens_total"),
            "ttft_sum": metric_sum(p, self.FAMILY_TTFT + "_sum"),
            "ttft_count": metric_sum(p, self.FAMILY_TTFT + "_count"),
            "ttft_buckets": hist_buckets(p, self.FAMILY_TTFT),
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            s = self.fetch()
            if s is not None:
                self.samples.append((time.monotonic(), s))
            self._stop.wait(self.period_s)

    def start(self) -> "MetricsSampler":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=10)

    def window(self, w0: float, w1: float) -> Dict[str, Any]:
        win = [(t, s) for t, s in self.samples if w0 <= t <= w1]
        if len(win) < 2:
            return {"metrics_samples": len(win)}
        (t0, s0), (t1, s1) = win[0], win[-1]
        dt = max(1e-9, t1 - t0)
        running = [s["running"] for _, s in win if s.get("running") is not None]
        waiting = [s["waiting"] for _, s in win if s.get("waiting") is not None]
        kv = [s["kv_usage"] for _, s in win if s.get("kv_usage") is not None]

        def delta(k):
            a, b = s0.get(k), s1.get(k)
            return None if a is None or b is None else b - a

        gen = delta("gen_tokens")
        ttft_mean = None
        ds, dc = delta("ttft_sum"), delta("ttft_count")
        if ds is not None and dc:
            ttft_mean = ds / dc
        b0 = dict(s0.get("ttft_buckets") or [])
        b1 = s1.get("ttft_buckets") or []
        dbuckets = [(le, c - b0.get(le, 0.0)) for le, c in b1]
        return {
            "metrics_samples": len(win),
            "running_mean": (sum(running) / len(running) if running else None),
            "running_max": (max(running) if running else None),
            "waiting_mean": (sum(waiting) / len(waiting) if waiting else None),
            "kv_usage_mean": (sum(kv) / len(kv) if kv else None),
            "preemptions_delta": delta("preemptions"),
            "engine_gen_tokens_delta": gen,
            "engine_gen_tok_s": (gen / dt if gen is not None else None),
            "ttft_mean_s": ttft_mean,
            "ttft_p50_s_hist": hist_quantile(dbuckets, 0.5),
            "ttft_p95_s_hist": hist_quantile(dbuckets, 0.95),
        }


class DmonSampler:
    """``nvidia-smi dmon -s u -d 1`` on the watched GPU, read on a thread;
    ``window(w0, w1)`` gives sm% mean / p10 / n over a monotonic-time window.
    Falls back to ``--query-gpu=utilization.gpu -l 1`` when dmon refuses."""

    def __init__(self):
        self.samples: List[Tuple[float, int]] = []
        self.proc: Optional[subprocess.Popen] = None
        self.mode = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "DmonSampler":
        sel = gpu_selector()
        cmds = [["nvidia-smi", "dmon", "-s", "u", "-d", "1"] + (["-i", sel] if sel else []),
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits",
                 "-l", "1"] + (["-i", sel] if sel else [])]
        for mode, cmd in zip(("dmon", "query"), cmds):
            try:
                self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                             text=True, bufsize=1)
            except Exception:
                continue
            time.sleep(1.5)
            if self.proc.poll() is None:
                self.mode = mode
                break
            self.proc = None
        if self.proc is None:
            _say("WARNING: nvidia-smi dmon/query unavailable — no GPU duty samples")
            return self
        self._thread = threading.Thread(target=self._run, name="dmon", daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        for line in self.proc.stdout:
            if self.mode == "dmon":
                p = parse_dmon_line(line)
                if p is not None:
                    self.samples.append((time.monotonic(), p[1]))
            else:
                try:
                    self.samples.append((time.monotonic(), int(float(line.strip()))))
                except ValueError:
                    pass

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def window(self, w0: float, w1: float) -> Dict[str, Any]:
        xs = [sm for t, sm in self.samples if w0 <= t <= w1]
        return {"sm_mean": (sum(xs) / len(xs) if xs else None),
                "sm_p10": percentile(xs, 0.10), "sm_samples": len(xs), "sm_mode": self.mode}


def _route_field(route: Any, name: str) -> Any:
    """A routing-table route is a ``proxy.RouteEntry`` dataclass when loaded
    through ``load_routing_table`` and a plain dict in the raw JSON; read either
    (canary 372693 indexed the dataclass and died with TypeError)."""
    if isinstance(route, Mapping):
        return route[name]
    return getattr(route, name)


def vmodel_by_cell_of(table: Any) -> Dict[str, str]:
    """cell_id -> virtual_model for every route of a loaded routing table
    (dict-shaped table with a ``routes`` list, or an object with ``.routes``)."""
    routes = table["routes"] if isinstance(table, Mapping) else getattr(table, "routes")
    return {str(_route_field(r, "cell_id")): str(_route_field(r, "virtual_model")) for r in routes}


def _load_routing_table(path: Path):
    """The proxy's loader (lazy import: the routing core is importable on CPU;
    kept behind one name so the CPU sweep test can stand in a fake table)."""
    from sisteer.capeval.serving.proxy import load_routing_table
    return load_routing_table(path)


def standup_dir(ctx: "SweepContext", tag: str) -> Path:
    """A standup directory unique to THIS job: ``standups/<tag>__job<id>``. The
    standup exports its lens registry READ-ONLY into the directory, so a rerun
    (a resubmitted array task, a second canary) that reuses ``standups/<tag>``
    dies with EACCES re-exporting over the previous job's artifacts (canary
    371407, 2026-09-03). The job id keeps every attempt's artifacts and makes
    the directory fresh; off-Slurm the pid stands in for the id."""
    job = str(getattr(ctx, "job_id", None) or os.environ.get("SLURM_JOB_ID") or f"pid{os.getpid()}")
    return Path(ctx.results_dir) / "standups" / f"{tag}__job{job}"


LOCK_FILE = ".sweep.lock"
OWNER_FILE = ".sweep.owner.json"
#: the exit code of a refused (locked) results dir — distinct from a
#: measurement / geometry refusal (1) so a watcher can tell them apart
EXIT_LOCKED = 2


class ResultsLock:
    """EXCLUSIVE ownership of one model's results dir for a whole phase (review
    S2): an ``flock`` on ``<results_dir>/.sweep.lock`` held from before the first
    resume check to after the summary, with an owner sidecar (pid / job / host /
    phase / started) beside it. A second sweep process on the same dir — a Slurm
    retry, a resubmission with another fingerprint — finds the lock held and
    REFUSES with exit :data:`EXIT_LOCKED`, naming the live holder; without this,
    two processes could each pass the startup validation, append rows of two
    measurements for hours, and each overwrite sweep_summary.json with its own
    in-memory view. The lock is advisory and per-host-kernel (a shared
    filesystem lock is honored by NFS/MooseFS flock); the sidecar is what names
    the holder."""

    def __init__(self, results_dir: Path, *, phase: str, job_id: Optional[str], node: str):
        self.results_dir = Path(results_dir)
        self.phase, self.job_id, self.node = phase, job_id, node
        self._fh = None

    @property
    def lock_path(self) -> Path:
        return self.results_dir / LOCK_FILE

    @property
    def owner_path(self) -> Path:
        return self.results_dir / OWNER_FILE

    def holder(self) -> Dict[str, Any]:
        try:
            return json.loads(self.owner_path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}

    def __enter__(self) -> "ResultsLock":
        import fcntl
        self.results_dir.mkdir(parents=True, exist_ok=True)
        fh = open(self.lock_path, "a+")
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError):
            fh.close()
            h = self.holder()
            _say(f"REFUSED: {self.results_dir} is owned by a live sweep process — pid "
                 f"{h.get('pid')} job {h.get('job_id')} on {h.get('node')} ({h.get('phase')} phase, "
                 f"started {h.get('started')}); two sweeps on one results dir would interleave "
                 f"two measurements. Exit {EXIT_LOCKED}; wait for it or use another --results-root")
            raise SystemExit(EXIT_LOCKED)
        self._fh = fh
        owner = {"pid": os.getpid(), "job_id": self.job_id, "node": self.node,
                 "phase": self.phase, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        tmp = self.owner_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(owner, indent=2))
        tmp.replace(self.owner_path)
        return self

    def __exit__(self, *exc) -> None:
        import fcntl
        try:
            self.owner_path.unlink()
        except FileNotFoundError:
            pass
        if self._fh is not None:
            try:
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            finally:
                self._fh.close()
                self._fh = None


def _bring_up(ctx: SweepContext, *, max_model_len: int, max_num_seqs: int, tag: str,
              eval_key: Optional[str]):
    """One standup through the production path at an explicit geometry (the
    over-budget guard waived on purpose — probing past it IS the point)."""
    import capctrl_serving_up as standup

    geom = standup.resolve_standup_geometry(eval_key, max_model_len, max_num_seqs,
                                            allow_over_budget=True)
    _say(geom.describe())
    out = standup_dir(ctx, tag)
    out.mkdir(parents=True, exist_ok=True)
    if ctx.objects_dir is not None and ctx.objects_dir.is_dir():
        defenses_dir = ctx.objects_dir
    else:
        defenses_dir = out / "no_objects"
        defenses_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    su = standup.bring_up(
        repo=ctx.repo, config_path=ctx.config_path, cell=ctx.standup_cell(),
        defenses_dir=defenses_dir, out=out, geom=geom, gpu_mem_util=ctx.gpu_mem_util,
        ready_timeout_s=ctx.ready_timeout_s, install_timeout_s=900,
        manifest=ctx.manifest,
        # tinycap E45 H1 boundary: a fitted standup must normally carry the
        # preflight-verified artifact digest. This sweep files TIMINGS, never
        # rows, and has no preflight to bind to — so it says so explicitly
        # instead of omitting the flag and being bound to nothing silently
        # (the bring-up prints UNBOUND when it takes this path).
        require_object_digest=False)
    _say(f"standup {tag} ready in {time.monotonic() - t0:.0f}s: proxy {su.proxy_url}, "
         f"server {su.launch.host}:{su.launch.port}")
    return su


def probe_geometry(ctx: SweepContext, *, max_model_len: int, max_num_seqs: int,
                   expect_object: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """One budget probe: bring the server up at S x L, record the attested KV
    facts + the memory-profiling log lines, tear it down.

    ``expect_object`` is the phase fingerprint's ``steered_object`` — the bank
    digests taken BEFORE any standup. The probe re-attests the bytes the
    standup actually loaded against it and records the attested identity on the
    probe (E54 #5); a mismatch is a SystemExit, so it aborts the walk rather
    than being recorded as a geometry refusal (it is not evidence about the
    geometry), and the ``finally`` below still tears the server down."""
    from sisteer.capeval.serving import vllm_server as vs

    product = max_num_seqs * max_model_len
    tag = f"budget_L{max_model_len}xS{max_num_seqs}"
    out = standup_dir(ctx, tag)
    rec: Dict[str, Any] = {"max_model_len": max_model_len, "max_num_seqs": max_num_seqs,
                           "product": product, "tag": tag, "t_start": time.time(),
                           "gpu_mem_util": ctx.gpu_mem_util}
    _say(f"BUDGET PROBE {max_num_seqs} x {max_model_len} = {product:,}")
    su = None
    try:
        su = _bring_up(ctx, max_model_len=max_model_len, max_num_seqs=max_num_seqs, tag=tag,
                       eval_key=None)
        rec[ATTESTED_OBJECT_FIELD] = attest_object_identity(
            su.out, su.harness_cell, expect_object, where=f"budget probe {tag}")
        att = _read_json(su.attestation_path) or {}
        reports = worker_reports_of(att)
        rec.update(kv_facts_from_reports(reports, max_model_len))
        rec["status"] = "pass"
        rec["attested_reports"] = len(reports)
    except vs.VllmServerError as e:
        log_text = (out / "vllm_server.log").read_text(errors="replace") \
            if (out / "vllm_server.log").is_file() else ""
        rec["status"] = classify_launch_failure(str(e), log_text)
        rec["error"] = str(e)[:2000]
        _say(f"probe {tag} REFUSED ({rec['status']}): {str(e)[:300]}")
    except Exception as e:  # noqa: BLE001 — recorded, the walk continues
        log_text = (out / "vllm_server.log").read_text(errors="replace") \
            if (out / "vllm_server.log").is_file() else ""
        rec["status"] = classify_launch_failure(str(e), log_text)
        rec["error"] = f"{type(e).__name__}: {e}"[:2000]
        rec["traceback"] = traceback.format_exc()[-3000:]
        _say(f"probe {tag} FAILED ({rec['status']}): {type(e).__name__}: {str(e)[:300]}")
    finally:
        if su is not None:
            su.tear_down()
    log_path = out / "vllm_server.log"
    if log_path.is_file():
        rec.update(scrape_server_log(log_path.read_text(errors="replace")))
    rec["gpu_mem_used_after_mib"] = wait_gpu_free()
    rec["t_end"] = time.time()
    rec["elapsed_s"] = rec["t_end"] - rec["t_start"]
    return rec


def run_budget(ctx: SweepContext, *, budget_len: int, ladder: Sequence[int]) -> Dict[str, Any]:
    """The budget phase under the exclusive results lock (review S2)."""
    with ResultsLock(ctx.results_dir, phase="budget", job_id=ctx.job_id, node=ctx.node):
        return _run_budget_locked(ctx, budget_len=budget_len, ladder=ladder)


def _run_budget_locked(ctx: SweepContext, *, budget_len: int, ladder: Sequence[int]
                       ) -> Dict[str, Any]:
    path = ctx.results_dir / "budget.json"
    fp = measurement_fingerprint(ctx, phase="budget", budget_len=int(budget_len),
                                 ladder=[int(x) for x in ladder])
    mid = fingerprint_id(fp)
    prev = _read_json(path) or {}
    probes: List[Dict[str, Any]] = []
    if prev.get("probes"):
        # a RESUME: the recorded probes are reused only under the SAME
        # measurement fingerprint (review R2); a pre-fingerprint or foreign
        # budget.json is refused, never silently extended
        if prev.get("measurement_id") != mid:
            raise SystemExit(refuse_reuse_message(
                path, "budget probes", len(prev["probes"]), prev.get("measurement_id"), mid,
                fingerprint_diff(prev.get("fingerprint") or {}, fp), ctx.results_dir))
        probes = [reclassify_probe(p, ctx.results_dir) for p in prev["probes"]]
    for p in probes:
        if p.get("reclassified_from"):
            _say(f"resume: probe {p.get('tag')} re-classified {p['reclassified_from']} -> "
                 f"{p['status']} from its recorded error / server log")
    for p in ladder:
        ladder_seqs(p, budget_len)          # refuse a ladder the window cannot tile
    doc: Dict[str, Any] = {
        "model_key": ctx.ms.key, "hf": ctx.ms.hf, "rev": ctx.ms.rev, "config": str(ctx.config_path),
        "cell": ctx.standup_cell(), "max_model_len": budget_len, "ladder": list(ladder),
        "gpu_mem_util": ctx.gpu_mem_util, "node": ctx.node, "job_id": ctx.job_id,
        "gpu": (_nvidia_smi("--query-gpu=name,memory.total", "--format=csv,noheader") or "").strip(),
        "transformers_version": transformers_version(), "requirements_file": requirements_file(),
        "fingerprint": fp, "measurement_id": mid,
        "probes": probes, "complete": False,
    }
    _write_json(path, doc)                 # the fingerprint is on disk BEFORE the first probe
    while True:
        nxt = next_budget_probe(probes, ladder, budget_len)
        if nxt is None:
            break
        if not ctx.have_time(max(600.0, ctx.ready_timeout_s * 0.5)):
            _say("deadline: stopping the budget walk before the next probe")
            doc["stopped_for_deadline"] = True
            break
        rec = probe_geometry(ctx, max_model_len=budget_len,
                             max_num_seqs=ladder_seqs(nxt, budget_len),
                             expect_object=fp.get("steered_object"))
        # the probe's OWN provenance (a resumed walk mixes jobs / nodes)
        rec.update({"job_id": ctx.job_id, "node": ctx.node, "measurement_id": mid,
                    "versions": fp.get("versions"), "requirements_file": fp.get("requirements_file")})
        probes.append(rec)
        doc["probes"] = probes
        _write_json(path, doc)
        last = probes[-1]
        if last.get("status") == STATUS_GPU_UNAVAILABLE:
            # infra: the card is busy/unavailable — no wall, no bisection, and
            # a message distinct from the probe-code one: resubmit elsewhere.
            doc["aborted"] = STATUS_GPU_UNAVAILABLE
            _write_json(path, doc)
            raise SystemExit(gpu_unavailable_message(ctx.node, last.get("tag"), last.get("error")))
        if last.get("status") == "launch_error":
            # our own code raised after (or instead of) a launch: not evidence
            # about the geometry — fail loudly rather than file a wall.
            raise SystemExit(f"[sweep] budget probe {last.get('tag')} hit a PROBE-CODE error, "
                             f"not a geometry refusal: {last.get('error')} — fix and rerun")
        if last.get("status") == "not_ready":
            _say(f"probe {last.get('tag')} timed out before readiness — stopping the walk "
                 "WITHOUT recording a wall (infra, not geometry)")
            doc["aborted"] = "not_ready"
            break
    bracket = budget_bracket(probes)
    doc.update(bracket)
    doc["complete"] = next_budget_probe(probes, ladder, budget_len) is None
    doc["measured_on"] = time.strftime("%Y-%m-%d")
    doc["paste"] = paste_record(
        model_key=ctx.ms.key, bracket=bracket, max_model_len=budget_len,
        gpu_mem_util=float(ctx.gpu_mem_util), job_id=ctx.job_id, measured_on=doc["measured_on"],
        results_dir=str(ctx.results_dir), node=ctx.node)
    _write_json(path, doc)
    lp = bracket["largest_passing"]
    _say(f"BUDGET RESULT {ctx.ms.key}: largest passing = "
         f"{(lp['product'] if lp else None)} (S={lp['max_num_seqs'] if lp else None} at "
         f"L={budget_len}; kv_tokens={lp.get('kv_tokens') if lp else None}); "
         f"first refused = {bracket['first_refused']}; complete={doc['complete']}")
    return doc


def tokenize_count(proxy_url: str, vmodel: str, messages, timeout_s: float = 120.0) -> int:
    """The proxy's ``/tokenize`` pass-through (the same virtual-model resolution
    the requests use): the exact prompt length of ``messages`` with the
    generation prompt appended."""
    import urllib.request

    base = proxy_url.rstrip("/")
    body = json.dumps({"model": vmodel, "messages": messages,
                       "add_generation_prompt": True}).encode("utf-8")
    req = urllib.request.Request(f"{base}/tokenize", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        doc = json.loads(r.read().decode("utf-8", "replace"))
    return int(doc["count"])


def build_prompt(proxy_url: str, vmodel: str, target_tokens: int
                 ) -> Tuple[List[Dict[str, str]], int, int]:
    """The synthetic user message: FILLER repeated k times so the rendered chat
    prompt tokenizes (proxy ``/tokenize``) to <= ``target_tokens``, as close as
    the filler allows. Returns ``(messages, token_count, k)``."""
    cache: Dict[int, int] = {}

    def count_fn(k: int) -> int:
        if k not in cache:
            cache[k] = tokenize_count(proxy_url, vmodel,
                                      [{"role": "user", "content": FILLER * k}])
        return cache[k]

    k, c = fit_repeats(count_fn, target_tokens)
    return [{"role": "user", "content": FILLER * k}], c, k


async def _closed_loop(proxy_url: str, vmodel: str, messages, *, B: int, max_tokens: int,
                       warmup_s: float, measure_s: float
                       ) -> Tuple[List[Tuple[float, float, int, bool, Optional[str]]],
                                  Tuple[float, float]]:
    """B closed-loop workers against the proxy for warmup + measure seconds;
    returns the per-request records and the measurement window (monotonic)."""
    import aiohttp

    base = proxy_url.rstrip("/")
    payload = {"model": vmodel, "messages": messages, "max_tokens": int(max_tokens),
               "ignore_eos": True, "temperature": 0.0, "stream": False}
    records: List[Tuple[float, float, int, bool, Optional[str]]] = []
    stop = asyncio.Event()
    t0 = time.monotonic()
    w0, w1 = t0 + warmup_s, t0 + warmup_s + measure_s

    async def worker(session):
        while not stop.is_set():
            ts = time.monotonic()
            ok, ct, err = False, 0, None
            try:
                async with session.post(f"{base}/chat/completions", json=payload) as r:
                    body = await r.json(content_type=None)
                    ok = r.status == 200
                    if ok:
                        ct = int((body.get("usage") or {}).get("completion_tokens") or 0)
                    else:
                        err = str(body)[:300]
            except Exception as e:  # noqa: BLE001 — recorded per request
                err = repr(e)[:300]
            te = time.monotonic()
            records.append((ts, te, ct, ok, err))
            if te >= w1:
                break

    timeout = aiohttp.ClientTimeout(total=900)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [asyncio.create_task(worker(session)) for _ in range(B)]
        await asyncio.sleep(warmup_s + measure_s)
        stop.set()
        done, pending = await asyncio.wait(tasks, timeout=300)
        for t in pending:
            t.cancel()
    return records, (w0, w1)


def _figures(rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any], figs_dir: Path,
             model_key: str) -> List[str]:
    """tok/s vs B, GPU duty vs B, latency vs B — one line per window, arms as
    solid (passthrough) / dashed (steered), knees marked. Skipped with a
    warning when matplotlib is absent (the analyzer redraws on the grading host)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        _say(f"WARNING: matplotlib unavailable ({e}); figures skipped — geom_sweep_analyze.py draws them")
        return []
    figs_dir.mkdir(parents=True, exist_ok=True)
    rows = [r for r in rows if is_point_row(r)]
    palette = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#F0E442", "#000000"]
    windows = sorted({int(r["window_L"]) for r in rows})
    color = {L: palette[i % len(palette)] for i, L in enumerate(windows)}
    style = {ARM_PASSTHROUGH: "-", ARM_STEERED: "--"}
    written = []
    for metric, ylabel, fname in (("tok_s", "completion tokens / s", "tok_s_vs_B.png"),
                                  ("sm_mean", "GPU sm% (mean over window)", "duty_vs_B.png"),
                                  ("lat_p50_s", "per-request latency p50 (s)", "latency_vs_B.png")):
        fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=130)
        for key, g in sorted(summary["groups"].items()):
            L, arm = g["window_L"], g["arm"]
            pts = [(t["B"], t[metric]) for t in g["table"] if t.get(metric) is not None]
            if not pts:
                continue
            xs, ys = zip(*sorted(pts))
            labels = ",".join(g.get("window_labels") or ()) or "-"
            ax.plot(xs, ys, style.get(arm, ":"), marker="o", ms=3.5, lw=1.6, color=color[L],
                    label=f"L={L} ({labels}) S={g['S']} {arm}")
            if metric == "tok_s" and g.get("knee_B") is not None:
                ax.plot([g["knee_B"]], [g["tok_s_at_knee"]], marker="o", ms=10, mfc="none",
                        mec=color[L], mew=1.8, ls="none")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("concurrent requests B (closed loop)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{model_key}: {ylabel} vs batch  (budget {summary.get('budget')})")
        ax.grid(True, which="both", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=6.5, frameon=False, ncol=1, loc="best")
        fig.tight_layout()
        out = figs_dir / f"{model_key}_{fname}"
        fig.savefig(out)
        plt.close(fig)
        written.append(str(out))
    return written


def stand_up_window_with_retry(ctx: SweepContext, *, max_model_len: int, s_target: int,
                               eval_key: Optional[str], record: Callable[[Dict[str, Any]], None],
                               retries: int = WINDOW_RETRIES,
                               ladder: Sequence[int] = BATCH_LADDER):
    """Stand a sweep window up at ``s_target``; on a launch failure (the
    readiness gate's KV-coverage refusal, an OOM, anything
    :func:`classify_launch_failure` names) RECORD a ``window_refused`` row and
    RETRY at the next lower ladder S (469 -> 384 -> 256 ...), at most
    ``retries`` times; then record ``window_skipped`` and return ``(None, None,
    tried)`` so the caller continues with the next window. One window's refusal
    never ends the phase (canary 373464: qwen35_4b's 5th window, 469 x 1536,
    was refused after 44 points and the whole phase exited 1 with no summary).
    The one exception is :data:`STATUS_GPU_UNAVAILABLE` (infra): recorded, no
    retry, and the phase aborts with the budget walk's SystemExit so the task
    is resubmitted on another node. Returns ``(standup, S_used, S_tried_list)``."""
    L = max_model_len
    labels = ",".join(window_labels(L)) or "off-table"
    s_try = int(s_target)
    tried: List[int] = []
    attempt = 0
    while True:
        tag = f"sweep_L{L}xS{s_try}"
        try:
            su = _bring_up(ctx, max_model_len=L, max_num_seqs=s_try, tag=tag, eval_key=eval_key)
            return su, s_try, tried
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:  # noqa: BLE001 — classified, recorded, retried
            log_path = standup_dir(ctx, tag) / "vllm_server.log"
            log_text = log_path.read_text(errors="replace") if log_path.is_file() else ""
            status = classify_launch_failure(str(e), log_text)
            excerpt = " ".join(f"{type(e).__name__}: {e}".split())[:600]
            tried.append(s_try)
            attempt += 1
            record({"kind": ROW_KIND_REFUSED, "model": ctx.ms.key, "window_L": L,
                    "eval_key": eval_key, "window_labels": list(window_labels(L)),
                    "S_tried": s_try, "S_target": int(s_target), "attempt": attempt,
                    "status": status, "error": excerpt, "t": time.time(),
                    "node": ctx.node, "job_id": ctx.job_id,
                    "transformers_version": transformers_version(),
                    "retried": status != STATUS_GPU_UNAVAILABLE})
            if status == STATUS_GPU_UNAVAILABLE:
                # infra: a smaller S cannot help — the same abort as the budget walk
                _say(f"WINDOW L={L} ({labels}) standup at S={s_try}: GPU busy/unavailable on "
                     f"{ctx.node} — no retry ladder; aborting the phase for a resubmit elsewhere")
                raise SystemExit(gpu_unavailable_message(ctx.node, tag, excerpt))
            wait_gpu_free()
            s_next = next_lower_seqs(s_try, ladder)
            if attempt > retries or s_next is None:
                reason = ("retries exhausted" if s_next is not None
                          else "no smaller S on the ladder")
                record({"kind": ROW_KIND_SKIPPED, "model": ctx.ms.key, "window_L": L,
                        "eval_key": eval_key, "window_labels": list(window_labels(L)),
                        "S_tried": list(tried), "S_target": int(s_target), "reason": reason,
                        "last_status": status, "t": time.time(), "node": ctx.node,
                        "job_id": ctx.job_id})
                _say(f"WINDOW L={L} ({labels}) SKIPPED after {len(tried)} refusals (S tried "
                     f"{', '.join(map(str, tried))}; last {status}; {reason}) — continuing "
                     "with the next window")
                return None, None, tried
            _say(f"WINDOW L={L} ({labels}) REFUSED at S={s_try} ({status}): {excerpt[:220]} "
                 f"— retry {attempt}/{retries} at S={s_next}")
            s_try = s_next


def run_sweep(ctx: SweepContext, *, windows: Sequence[int], budget: Optional[int],
              prompt_tokens: int, max_tokens: int, warmup_s: float, measure_s: float,
              ladder: Sequence[int] = BATCH_LADDER) -> Dict[str, Any]:
    """The sweep phase under the exclusive results lock (review S2)."""
    with ResultsLock(ctx.results_dir, phase="sweep", job_id=ctx.job_id, node=ctx.node):
        return _run_sweep_locked(ctx, windows=windows, budget=budget, prompt_tokens=prompt_tokens,
                                 max_tokens=max_tokens, warmup_s=warmup_s, measure_s=measure_s,
                                 ladder=ladder)


def foreign_rows_on_disk(rows_path: Path, measurement_id: str) -> List[Dict[str, Any]]:
    """Every row in the COMPLETE on-disk sweep.jsonl whose measurement_id is not
    ``measurement_id`` — re-read at summary time (review S2): the startup
    validation cannot see a row another process appended later."""
    if not rows_path.is_file():
        return []
    out = []
    for ln in rows_path.read_text().splitlines():
        if not ln.strip():
            continue
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            out.append({"kind": "torn_line", "measurement_id": None})
            continue
        if r.get("measurement_id") != measurement_id:
            out.append(r)
    return out


def _run_sweep_locked(ctx: SweepContext, *, windows: Sequence[int], budget: Optional[int],
                      prompt_tokens: int, max_tokens: int, warmup_s: float, measure_s: float,
                      ladder: Sequence[int] = BATCH_LADDER) -> Dict[str, Any]:
    bdoc = _read_json(ctx.results_dir / "budget.json") or {}
    if budget is None:
        lp = (bdoc.get("largest_passing") or {})
        budget = int(lp["product"]) if lp else None
    if budget is None:
        raise SystemExit(f"[sweep] {ctx.ms.key}: no measured budget (budget.json has no passing "
                         "probe) and no --budget override — run --phase budget first")
    fp = measurement_fingerprint(
        ctx, phase="sweep", budget=int(budget), windows=[int(w) for w in windows],
        batch_ladder=[int(b) for b in ladder], prompt_tokens=int(prompt_tokens),
        max_tokens=int(max_tokens), warmup_s=float(warmup_s), measure_s=float(measure_s))
    mid = fingerprint_id(fp)
    rows_path = ctx.results_dir / "sweep.jsonl"
    rows: List[Dict[str, Any]] = []
    if rows_path.is_file():
        for ln in rows_path.read_text().splitlines():
            if ln.strip():
                rows.append(json.loads(ln))
    # a RESUME reuses only rows of the SAME measurement (review R2 / S2): EVERY
    # persisted row — points AND window_refused / window_skipped events — must
    # carry this fingerprint's id; a store of another fingerprint, or a
    # pre-fingerprint one (status-only stores included), is refused, never merged
    foreign = [r for r in rows if r.get("measurement_id") != mid]
    if foreign:
        prev_fp = (_read_json(ctx.results_dir / "sweep_summary.json") or {}).get("fingerprint") or {}
        kinds = sorted({str(r.get("kind", ROW_KIND_POINT)) for r in foreign})
        raise SystemExit(refuse_reuse_message(
            rows_path, f"sweep rows ({', '.join(kinds)})", len(foreign),
            ",".join(sorted({str(r.get("measurement_id")) for r in foreign})), mid,
            fingerprint_diff(prev_fp, fp), ctx.results_dir))
    done = {point_key(r) for r in rows if is_point_row(r)}
    partial = False
    missing: List[Tuple[int, str, int]] = []
    point_cost = warmup_s + measure_s + 45.0
    standup_cost = 420.0

    tf_ver, reqs_file = transformers_version(), requirements_file()

    def record(row: Dict[str, Any]) -> None:
        row.setdefault("measurement_id", mid)
        with rows_path.open("a") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
        rows.append(row)

    for L in windows:
        S_target = permitted_seqs(budget, L, cap=ctx.seqs_cap)
        arms = [(ARM_PASSTHROUGH, "none")]
        if ctx.steered_cell:
            arms.append((ARM_STEERED, ctx.steered_cell))

        def pending_at(S_: int):
            # completion is keyed on the ACTUAL S: a point measured at a
            # reduced S is not the target-S point (review R2)
            return [(arm, cell, B) for B in batch_ladder(S_, ladder) for arm, cell in arms
                    if (L, arm, B, S_) not in done]

        pending = pending_at(S_target)
        if not pending:
            _say(f"window L={L}: all {len(batch_ladder(S_target, ladder)) * len(arms)} points "
                 "present — skipping")
            continue
        if not ctx.have_time(standup_cost + point_cost):
            _say(f"deadline: skipping window L={L} ({len(pending)} points)")
            partial = True
            missing += [(L, a, B) for a, _, B in pending]
            continue
        eval_key = eval_key_for_window(L)
        _say(f"WINDOW L={L} ({','.join(window_labels(L)) or 'off-table'}): S={S_target} "
             f"(budget {budget:,} // {L}, cap {ctx.seqs_cap}"
             f"{' hybrid per-slot cap' if ctx.hybrid else ''}); ladder "
             f"{batch_ladder(S_target, ladder)}; arms {[a for a, _ in arms]}; "
             f"{len(pending)} points pending")
        su, S, tried = stand_up_window_with_retry(
            ctx, max_model_len=L, s_target=S_target, eval_key=eval_key, record=record,
            ladder=ladder)
        if su is None:
            continue                      # recorded as window_skipped; next window
        if tried:
            _say(f"WINDOW L={L}: standing at S={S} after {len(tried)} refusal(s) "
                 f"(target was {S_target})")
        pending = pending_at(S)
        metrics = None
        dmon = None
        try:
            # the fingerprint hashed the BANK before any standup and the
            # standup is unbound to an object digest, so the bytes it loaded
            # are re-attested here — against the copies in the registry it
            # exported, which are what the server serves — before a single
            # point of this window is measured. A mismatch aborts (the
            # `finally` tears the server down); the attested identity travels
            # on every point row.
            attested = attest_object_identity(
                su.out, su.harness_cell, fp.get("steered_object"),
                where=f"sweep window L={L} S={S}")
            table = _load_routing_table(su.table_path)
            vmodel_by_cell = vmodel_by_cell_of(table)
            harness_cell = su.harness_cell
            vmodels = {ARM_PASSTHROUGH: vmodel_by_cell.get("none")}
            if ctx.steered_cell:
                vmodels[ARM_STEERED] = vmodel_by_cell.get(harness_cell)
            for arm, vm in vmodels.items():
                if vm is None:
                    raise RuntimeError(f"no route for arm {arm}: routes {sorted(vmodel_by_cell)}")
            messages, n_prompt, k_rep = build_prompt(su.proxy_url, vmodels[ARM_PASSTHROUGH],
                                                     prompt_tokens)
            _say(f"prompt: {k_rep} filler repeats -> {n_prompt} tokens (target {prompt_tokens})")
            metrics = MetricsSampler(su.launch.host, su.launch.port, su.launch.api_key).start()
            dmon = DmonSampler().start()
            time.sleep(SAMPLER_SETTLE_S)
            for arm, cell, B in pending:
                if not ctx.have_time(point_cost):
                    _say(f"deadline: stopping inside window L={L} before B={B} {arm}")
                    partial = True
                    missing += [(L, a, b) for a, _, b in pending
                                if (L, a, b) not in done]
                    break
                _say(f"POINT L={L} S={S} arm={arm} B={B}: warmup {warmup_s:.0f}s + measure "
                     f"{measure_s:.0f}s, max_tokens {max_tokens}, ignore_eos")
                t_start = time.time()
                records, (w0, w1) = asyncio.run(_closed_loop(
                    su.proxy_url, vmodels[arm], messages, B=B, max_tokens=max_tokens,
                    warmup_s=warmup_s, measure_s=measure_s))
                row: Dict[str, Any] = {
                    "kind": ROW_KIND_POINT,
                    "model": ctx.ms.key, "window_L": L, "window_labels": list(window_labels(L)),
                    "eval_key": eval_key, "S": S, "S_target": S_target, "S_cap": ctx.seqs_cap,
                    "hybrid": ctx.hybrid, "retries": len(tried), "budget": budget, "arm": arm,
                    "transformers_version": tf_ver, "requirements_file": reqs_file,
                    "virtual_model": vmodels[arm], "cell_id": cell, "B": B,
                    "prompt_tokens": n_prompt, "max_tokens": max_tokens,
                    "warmup_s": warmup_s, "measure_s": measure_s,
                    "n_records": len(records), "t_start": t_start, "t_end": time.time(),
                    "node": ctx.node, "job_id": ctx.job_id,
                    # WHICH bytes served this point, attested after the standup
                    ATTESTED_OBJECT_FIELD: attested,
                }
                row.update(in_window_stats(records, w0, w1))
                row.update(metrics.window(w0, w1))
                row.update(dmon.window(w0, w1))
                if su.server_rc() is not None:
                    row["server_died"] = su.server_rc()
                record(row)
                done.add(point_key(row))
                _say(f"  -> {row['tok_s']:.0f} tok/s ({row['n_completed']} completions, "
                     f"{row['n_errors']} errors), lat p50 {row['lat_p50_s']}, running "
                     f"{row.get('running_mean')}, preempt Δ {row.get('preemptions_delta')}, "
                     f"sm {row.get('sm_mean')}")
                if row.get("server_died") is not None:
                    _say("server died mid-window; abandoning this standup")
                    partial = True
                    break
        finally:
            if metrics is not None:
                metrics.stop()
            if dmon is not None:
                dmon.stop()
            su.tear_down()
            wait_gpu_free()
    summary = summarize_rows(rows, budget=budget, partial=partial, missing=missing,
                             windows_requested=windows, ladder=ladder,
                             expected_arms=[ARM_PASSTHROUGH]
                             + ([ARM_STEERED] if ctx.steered_cell else []),
                             measurement_id=mid)
    summary.update({"model_key": ctx.ms.key, "hf": ctx.ms.hf, "rev": ctx.ms.rev,
                    "fingerprint": fp, "measurement_id": mid,
                    "hybrid": ctx.hybrid, "S_cap": ctx.seqs_cap,
                    "transformers_version": tf_ver, "requirements_file": reqs_file,
                    "steered_cell": ctx.steered_cell, "prompt_tokens": prompt_tokens,
                    "max_tokens": max_tokens, "warmup_s": warmup_s, "measure_s": measure_s,
                    "node": ctx.node, "job_id": ctx.job_id,
                    "old_S_note": f"tok_s_at_B{OLD_S} = the pre-table concurrency every served "
                                  "eval inherited (Petri's 24)"})
    points = [r for r in rows if is_point_row(r)]
    # RE-VALIDATE THE WHOLE STORE before publishing (review S2): a row of
    # another measurement that appeared on disk during the run — a second
    # process that slipped past the lock's kernel scope, a hand-edit — means
    # this summary would label a mixed store; refuse, write nothing.
    foreign_now = foreign_rows_on_disk(rows_path, mid)
    if foreign_now:
        kinds = sorted({str(r.get("kind", ROW_KIND_POINT)) for r in foreign_now})
        raise SystemExit(
            f"[sweep] {rows_path} gained {len(foreign_now)} row(s) of another measurement "
            f"({', '.join(kinds)}; ids {sorted({str(r.get('measurement_id')) for r in foreign_now})}) "
            f"while this run (measurement {mid}) was measuring — the store is MIXED; refusing to "
            "write sweep_summary.json. Move the foreign rows out or use a fresh --results-root")
    summary["figures"] = _figures(points, summary, ctx.figs_dir, ctx.ms.key) if points else []
    _write_json(ctx.results_dir / "sweep_summary.json", summary)
    state = ("PARTIAL" if summary["partial"]
             else ("COMPLETE" if not summary["windows_not_measured"]
                   else "COMPLETE WITH SKIPPED WINDOWS"))
    _say(f"SWEEP {state}: {len(points)} points; incomplete groups {summary['groups_incomplete']}; "
         f"arms missing {summary['arms_missing']}; refused standups "
         f"{len(summary['window_refusals'])}; skipped windows "
         f"{[w['window_L'] for w in summary['skipped_windows']]}; not measured "
         f"{summary['windows_not_measured']}; summary at "
         f"{ctx.results_dir / 'sweep_summary.json'}; figures {summary['figures']}")
    return summary


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def _parse_int_list(s: Optional[str], default: Sequence[int]) -> Tuple[int, ...]:
    if not s:
        return tuple(default)
    return tuple(int(x) for x in re.split(r"[,\s]+", s.strip()) if x)


def preflight(ms: ModelServe, repo: Path, results_dir: Path, objects_root: Optional[Path]
              ) -> Dict[str, Any]:
    """Fail-fast checks on every external input, recorded to preflight.json:
    the config exists and agrees with the pins; the pinned HF snapshot is in
    the cache (``HF_HOME``); the fitted-object bank and which psr_kl cell the
    steered arm will use (absence is recorded, not fatal)."""
    problems: List[str] = []
    rec: Dict[str, Any] = {"model_key": ms.key, "hf": ms.hf, "rev": ms.rev}
    try:
        cfg_path = resolve_serving_config(ms, repo, results_dir)
        rec["config"] = str(cfg_path)
    except Exception as e:  # noqa: BLE001
        problems.append(f"config: {e}")
    hf_home = os.environ.get("HF_HOME")
    rec["hf_home"] = hf_home
    if not hf_home:
        problems.append("HF_HOME is unset (source $SHARED/scripts/hf_env.sh)")
    else:
        snap = Path(hf_home) / hf_snapshot_relpath(ms)
        rec["hf_snapshot"] = str(snap)
        if not (snap / "config.json").is_file():
            problems.append(f"pinned snapshot missing: {snap}/config.json")
    objects_dir = (objects_root / ms.bank) if objects_root is not None else None
    rec["objects_dir"] = str(objects_dir) if objects_dir else None
    rec["objects_dir_exists"] = bool(objects_dir and objects_dir.is_dir())
    rec["steered_cell"] = select_steered_cell(objects_dir)
    manifest = repo / "configs/capeval/manifests/petri_twosi.yaml"
    if not manifest.is_file():
        problems.append(f"manifest missing: {manifest}")
    rec["problems"] = problems
    rec["ok"] = not problems
    results_dir.mkdir(parents=True, exist_ok=True)
    _write_json(results_dir / "preflight.json", rec)
    return rec


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, choices=list(MODELS),
                    help="canonical model key (sisteer.capeval.serving.geometry.MODEL_KEYS)")
    ap.add_argument("--phase", required=True, choices=["env", "preflight", "budget", "sweep"])
    ap.add_argument("--repo-dir", default=str(REPO))
    ap.add_argument("--results-root", default=None,
                    help=f"per-model results land in <root>/<model>/ (default {DEFAULT_RESULTS_ROOT})")
    ap.add_argument("--objects-root", default=None,
                    help=f"fitted-object banks, <root>/<model>/<cell>/object.pt (default "
                         f"{DEFAULT_OBJECTS_ROOT}); 'none' = passthrough-only")
    ap.add_argument("--figs-dir", default=None, help="default: $FIGS_DIR, else <results>/figs")
    ap.add_argument("--steered-cell", default="auto",
                    help="auto (first of %s present in the bank) | none | <cell id>"
                         % ",".join(PSR_CELL_PREFERENCE))
    ap.add_argument("--gpu-mem-util", default=DEFAULT_GPU_MEM_UTIL)
    ap.add_argument("--ready-timeout-s", type=int, default=1500)
    ap.add_argument("--manifest", default="configs/capeval/manifests/petri_twosi.yaml")
    ap.add_argument("--budget-len", type=int, default=DEFAULT_BUDGET_LEN,
                    help="the fixed block-aligned window of the budget ladder")
    ap.add_argument("--budget-ladder", default=None,
                    help="comma list of S x L products (default %s)" % ",".join(map(str, BUDGET_LADDER)))
    ap.add_argument("--windows", default=None,
                    help="comma list of eval windows L for --phase sweep (default %s)"
                         % ",".join(map(str, DEFAULT_WINDOWS)))
    ap.add_argument("--budget", type=int, default=None,
                    help="override the measured budget (S x L) for --phase sweep")
    ap.add_argument("--batch-ladder", default=None,
                    help="comma list of B (default %s)" % ",".join(map(str, BATCH_LADDER)))
    ap.add_argument("--prompt-tokens", type=int, default=DEFAULT_PROMPT_TOKENS)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--warmup-s", type=float, default=DEFAULT_WARMUP_S)
    ap.add_argument("--measure-s", type=float, default=DEFAULT_MEASURE_S)
    ap.add_argument("--deadline-margin-s", type=float, default=600.0,
                    help="stop starting new work this many seconds before SLURM_JOB_END_TIME")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ms = MODELS[args.model]
    repo = Path(args.repo_dir).resolve()
    if args.phase == "env":
        print("\n".join(shell_env_lines(ms, repo)))
        return 0
    from sisteer import site as _site   # an unset $SHARED refuses by name
    results_root = Path(args.results_root) if args.results_root else _site.expand_path(DEFAULT_RESULTS_ROOT, key="--results-root")
    results_dir = results_root / ms.key
    objects_root: Optional[Path]
    if args.objects_root == "none":
        objects_root = None
    else:
        objects_root = (Path(args.objects_root) if args.objects_root
                        else _site.expand_path(DEFAULT_OBJECTS_ROOT, key="--objects-root"))
    if args.phase == "preflight":
        rec = preflight(ms, repo, results_dir, objects_root)
        print(json.dumps(rec, indent=2))
        return 0 if rec["ok"] else 1

    results_dir.mkdir(parents=True, exist_ok=True)
    config_path = resolve_serving_config(ms, repo, results_dir)
    objects_dir = (objects_root / ms.bank) if objects_root is not None else None
    if args.steered_cell == "auto":
        steered = select_steered_cell(objects_dir)
    elif args.steered_cell == "none":
        steered = None
    else:
        steered = args.steered_cell
        if select_steered_cell(objects_dir, (steered,)) is None:
            raise SystemExit(f"[sweep] --steered-cell {steered}: no object under {objects_dir}")
    figs_dir = Path(args.figs_dir or os.environ.get("FIGS_DIR") or (results_dir / "figs"))
    end = os.environ.get("SLURM_JOB_END_TIME")
    hybrid = is_hybrid_model(ms)
    ctx = SweepContext(
        hybrid=hybrid,
        ms=ms, repo=repo, results_dir=results_dir, figs_dir=figs_dir, config_path=config_path,
        objects_dir=objects_dir, steered_cell=steered, gpu_mem_util=str(args.gpu_mem_util),
        ready_timeout_s=int(args.ready_timeout_s), manifest=args.manifest,
        deadline_epoch=(float(end) if end and end.isdigit() else None),
        deadline_margin_s=float(args.deadline_margin_s),
        job_id=os.environ.get("SLURM_JOB_ID"), node=socket.gethostname())
    _say(f"model {ms.key} = {ms.hf} @ {ms.rev[:12]}; config {config_path}; objects "
         f"{objects_dir} (steered cell: {steered or 'NONE — passthrough only'}); results "
         f"{results_dir}; figs {figs_dir}; hybrid={hybrid} (per-window S cap {ctx.seqs_cap}); "
         f"deadline in {ctx.time_left_s() and round(ctx.time_left_s())} s")
    if args.phase == "budget":
        run_budget(ctx, budget_len=int(args.budget_len),
                   ladder=_parse_int_list(args.budget_ladder, BUDGET_LADDER))
        return 0
    run_sweep(ctx, windows=_parse_int_list(args.windows, DEFAULT_WINDOWS), budget=args.budget,
              prompt_tokens=int(args.prompt_tokens), max_tokens=int(args.max_tokens),
              warmup_s=float(args.warmup_s), measure_s=float(args.measure_s),
              ladder=_parse_int_list(args.batch_ladder, BATCH_LADDER))
    return 0


if __name__ == "__main__":
    sys.exit(main())
