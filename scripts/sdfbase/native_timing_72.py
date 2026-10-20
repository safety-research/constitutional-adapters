#!/usr/bin/env python
"""THE §7.2 TIMING HARNESS for the NATIVE residual slot.

ONE native standup per target through ``lora_serving_up.bring_up_lora`` (the library seam the
certification driver uses — never a subprocess of ``mt_serving_up``), serving ``[<psr cell>]`` (plus
the standup's own ``none`` and the zero adapter ``none__lora0``) at the lens gate's §7.2 geometry
``4096 x 64`` (``docs/lens_stacked_reductions_scoping.md`` §7.2: ``max_tokens`` 128, 45 s window, B
concurrent clients, production BI=0, prefix caching OFF), the residual slot ARMED (the psr cell is
served), ``output_diagnostics`` False (the central policy, §1.5). Per arm x B in {8, 32, 64} the gate's
closed-loop protocol: the gate's ~800-token passage (``closed_loop.build_timing_prompt``: the gate's
``"Passage 5. " + FILLER * k`` fitted through the SERVED tokenizer's chat render), ``max_tokens`` 128 at
EXACT length (``min_tokens`` + ``ignore_eos``), warm-up 10 s, window 45 s, the ENGINE's ``/metrics``
through the launch's Bearer header. ms/step is recorded THREE ways off the same window
(``closed_loop.ms_per_step_all``): ``gate`` (the lens gate's ``implied_step_ms``: 1000 x running_mean /
tok_s — the formula behind every §7.2 number, so the like-for-like PRIMARY), ``nominal`` (the
certification driver's 1000 x B / tok_s) and ``counter`` (the engine's iteration counter, sampled
beside the loop and aligned to its window; None when the engine does not expose it).
``hook_only_ms = ms_per_step(arm) - ms_per_step(none)`` at the same B from the SAME run — the §7.2
definition of "hook-only" (that run's own passthrough floor).

ARMS: ``none`` (the passthrough floor: the base vmodel on the graphs-on native server), ``psr`` (the psr
cell's vmodel), and ``vector`` — recorded ABSENT BY NAME unless ``--vector-cell`` is given: NO
all-positions vector-kind object is staged on any target, so stage 1 has no vector arm anywhere; the
record says so per target rather than inventing one.

THE TWO PROTOCOL MECHANISMS (found by reading the composer, recorded from what it composed — never
assumed):

* PREFIX CACHING OFF. ``vllm_server.build_server_launch(prefix_caching=None)`` resolves ``not
  hybrid_target`` — ON for the dense targets (llama8b, gemma4e4b), OFF for the hybrids (qwen35_4b,
  nemotron3nano4b) — and ``native_standup.bring_up_native``'s composer passes no ``prefix_caching`` at
  all, while ``bring_up_lora`` exposes no seam for it (its ``launch_extras`` are composed inside it and
  a manifest may not name the flag: the composer OWNS the choice). So this harness binds the composer
  for the standup's duration (:func:`prefix_caching_off`: ``build_server_launch(..., prefix_caching=
  False)`` — the composer's own OFF branch, ``--no-enable-prefix-caching`` in the argv), then READS what
  was composed (``launch.prefix_caching``, the argv token) and what the workers ATTESTED
  (``serving_attestation.json`` ``engine.enable_prefix_caching``) and REFUSES the run unless all three
  say OFF (:func:`protocol_problems`).
* BATCH INVARIANCE 0. The only path that sets ``VLLM_BATCH_INVARIANT`` is ``exact_replay_strategy ==
  "batch_invariant"``, which the native composer never passes; the child's env is ``dict(os.environ)``
  plus the launch's overlay, so the variable is INHERITED from the harness's environment. The harness
  refuses when its own environment carries ``VLLM_BATCH_INVARIANT=1``, records the harness env, the
  launch env and the attested ``batch_invariant_env`` / ``batch_invariant_active`` worker facts, and
  refuses unless every reading says unset/0 (vLLM's default: unset == 0).
* GEOMETRY 4096 x 64 is NOT the identity pair the table sizes (``identity_seqs(4096)`` is far above
  64), so the standup's strict identity guard needs ``geometry_override_reason`` with BOTH factors
  (``lora_serving_up._resolve_geometry``: ``strict_identity=False`` under a stated reason; the budget
  guard still holds, 262,144 tokens is under every target's budget). The reason is recorded in
  ``endpoint.json``; the harness asserts the ATTESTED ``max_num_seqs`` is exactly 64 (the LoRA kernel
  index ceiling can lower S silently: ``geometry.adjustments`` must be empty).

Records: ONE JSON per target ``<out>/<target>.json`` (rewritten after every window, ``status``
``partial`` until the last window; ``complete`` at the end), the standup under ``<out>/standup/``
(endpoint.json, cells_served.json, serving_attestation.json, positive_controls.json, vllm_server.log),
the per-window gauges under ``<out>/windows/``. The standup is torn down at the end (or on SIGTERM,
after the partial JSON is written).

Usage (the sbatch composes this; positional-free, every input named)::

  native_timing_72.py --target gemma4e4b --view psrcert --cell constv2-mix-full__psr_ce_all \\
      --out $SHARED/exp/native_timing_72/gemma4e4b_<job> --repo-dir $REPO_DIR \\
      --defenses-dir $TT_TWO_SI_DEFENSES_DIR --expect-object-sha256 <cell>=<sha> \\
      [--adapter-bank DIR] [--batches 8,32,64] [--window-s 45] [--warmup-s 10] [--max-tokens 128] \\
      [--vector-cell CELL]
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import datetime as _dt
import hashlib
import functools
import importlib
import json
import os
import signal
import socket
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
for _p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "sdfbase"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from sisteer.capeval.serving import closed_loop as cl
import sisteer.capeval.serving.geometry  # noqa: E402,F401  (reads the composer signature at import, before any rebinding of build_server_launch; T1-T3 died on a KeyError here)
from sisteer.capeval.serving import serving_attestation as sa  # noqa: E402  (stdlib-only leaf)

TAG = "[native_timing_72]"
SCHEMA = 1
DEFAULT_PSR_CELL = "constv2-mix-full__psr_ce_all"
#: the certification driver's eval: a GENERATION eval with a table entry (``tool_calls`` inferred
#: False); its table geometry is overridden by the §7.2 pair below, with the reason recorded
EVAL_NAME = "families_served:short"
#: the §7.2 protocol (docs/lens_stacked_reductions_scoping.md §7.2 + lens_vectorize_gate defaults)
PROTOCOL: Dict[str, Any] = {
    "L": 4096, "S": 64, "batches": (8, 32, 64), "max_tokens": 128, "exact_length": True,
    "window_s": 45.0, "warmup_s": 10.0, "prefix_caching": False, "batch_invariance": 0,
    "prompt_tokens": cl.GATE_TIMING_PROMPT_TARGET,
    "source": "docs/lens_stacked_reductions_scoping.md §7.2; scripts/lens_vectorize_gate.py "
              "(--max-model-len 4096 --max-num-seqs 64 --warmup-s 10 --measure-s 45; implied_step_ms)",
}
GEOMETRY_OVERRIDE_REASON = (
    "native-psr-serving §1.7 timing: the lens gate's §7.2 geometry 4096 x 64 "
    "(docs/lens_stacked_reductions_scoping.md §7.2) for a like-for-like reading against the reused "
    "lens rows — not the table's identity pair; the budget guard still holds")
VECTOR_ABSENT_REASON = "no all-positions vector-kind object is staged on {target} (D4)"
PREFIX_CACHING_MECHANISM = (
    "vllm_server.build_server_launch bound to prefix_caching=False for the standup's duration "
    "(native_timing_72.prefix_caching_off; the composer's own OFF branch composes "
    "--no-enable-prefix-caching) — bring_up_lora exposes no seam and a manifest may not name the flag")
BATCH_INVARIANCE_MECHANISM = (
    "VLLM_BATCH_INVARIANT left UNSET (vLLM default 0): the native composer never sets it "
    "(only exact_replay_strategy='batch_invariant' would) and the engine child inherits the harness "
    "environment, which this harness refuses to run with the variable set to 1")
ARM_LABEL_NATIVE = "native (measured)"
PRIMARY_MS_PER_STEP = "gate"
ARMS_TIMED = ("none", "psr")
BI_ENV = "VLLM_BATCH_INVARIANT"


class TimingRefused(RuntimeError):
    """A protocol precondition the harness cannot certify — refused, never measured around."""


def log(msg: str) -> None:
    print(f"{TAG} {msg}", flush=True)


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def write_json_atomic(path: Path, obj: Any) -> Path:
    return sa.write_json_atomic(Path(path), obj, default=str)


# --------------------------------------------------------------------------- #
# pure helpers (unit-tested without a GPU)                                      #
# --------------------------------------------------------------------------- #
def vector_arm_record(target: str, vector_cell: Optional[str]) -> Dict[str, Any]:
    """The ``vector`` arm's record: the cell when one is named, else ABSENT BY NAME."""
    if vector_cell:
        return {"status": "present", "cell": str(vector_cell)}
    return {"status": "absent", "reason": VECTOR_ABSENT_REASON.format(target=target)}


def config_paths(repo: Path, target: str, view: str) -> Tuple[Path, Path]:
    """``f22_resolve_config``'s naming: ``configs/sdfbase_full22_<tag>.yaml`` and
    ``configs/capeval/manifests/petri_full22_<tag>.yaml``, ``tag = <key>[_<view>]`` (``main`` bare)."""
    tag = target if view == "main" else f"{target}_{view}"
    return (Path(repo) / "configs" / f"sdfbase_full22_{tag}.yaml",
            Path(repo) / "configs" / "capeval" / "manifests" / f"petri_full22_{tag}.yaml")


def standup_kwargs(*, repo_dir: Path, config: Path, manifest: Path, defenses_dir: Path, out: Path,
                   target: str, cells: Sequence[str], expect_object_sha256: Sequence[str],
                   adapter_bank: Optional[Path], L: int = PROTOCOL["L"], S: int = PROTOCOL["S"],
                   gpu_mem_util: str = "0.9", ready_timeout_s: int = 1800,
                   probe_max_tokens: int = 48) -> Dict[str, Any]:
    """The ``bring_up_lora`` call, keyword-for-flag (the certification driver's shape): the timed
    cells (``none`` implicit, the zero adapter the standup's own), the pinned digests per served
    cell, the §7.2 geometry with its override reason, the default (or explicit) adapter bank."""
    if not cells:
        raise TimingRefused("no cell to serve — the psr arm needs its TrainTest cell id")
    kw: Dict[str, Any] = dict(
        repo_dir=str(repo_dir), config=str(config), manifest=str(manifest), defenses_dir=str(defenses_dir),
        out=str(out), model_key=str(target), eval=EVAL_NAME, cell=[str(c) for c in cells],
        expect_object_sha256=[str(d) for d in expect_object_sha256],
        max_model_len=int(L), max_num_seqs=int(S), geometry_override_reason=GEOMETRY_OVERRIDE_REASON,
        gpu_mem_util=str(gpu_mem_util), ready_timeout_s=int(ready_timeout_s),
        probe_max_tokens=int(probe_max_tokens),
    )
    if adapter_bank is not None:
        kw["adapter_bank"] = str(adapter_bank)
    return kw


@contextlib.contextmanager
def prefix_caching_off(module: Any = None) -> Iterator[List[Dict[str, Any]]]:
    """Bind the serving composer to ``prefix_caching=False`` for the block: ``build_server_launch``
    is called by ``native_standup.bring_up_native``'s composer through the module attribute (a local
    ``from ... import vllm_server as vs`` then ``vs.build_server_launch(...)``), so the attribute is
    rebound here and restored on exit. Every composed call is recorded (the yielded list); a caller
    asking for ``exact_replay_strategy="batch_invariant"`` is refused — the protocol is BI unset."""
    vs = module if module is not None else importlib.import_module("sisteer.capeval.serving.vllm_server")
    orig = vs.build_server_launch
    calls: List[Dict[str, Any]] = []
    # T1-T4 (jobs 509129-509131, 2026-09-15): `geometry.py` reads `inspect.signature(vs.build_server_launch)
    # .parameters["max_model_len"]` at IMPORT, and bring_up_lora imports it lazily — under a bare wrapper the
    # signature was `(spec, **kw)` and the import died with KeyError. The wrapper therefore carries the
    # original's signature (`functools.wraps` sets `__wrapped__`, which `inspect.signature` follows), and
    # the module is imported before the binding as a belt.
    @functools.wraps(orig)
    def bound(spec: Any, **kw: Any) -> Any:
        if kw.get("exact_replay_strategy") == "batch_invariant":
            raise TimingRefused("the §7.2 protocol is BI unset: refusing exact_replay_strategy='batch_invariant'")
        kw["prefix_caching"] = False
        calls.append({"prefix_caching": False, "exact_replay_strategy": kw.get("exact_replay_strategy"),
                      "max_model_len": kw.get("max_model_len"), "max_num_seqs": kw.get("max_num_seqs"),
                      "lora_native": kw.get("lora_native")})
        return orig(spec, **kw)

    vs.build_server_launch = bound
    try:
        yield calls
    finally:
        vs.build_server_launch = orig


def find_keys(obj: Any, keys: Sequence[str]) -> Dict[str, List[Any]]:
    """Every value under any of ``keys`` anywhere in a nested record (the attestation nests the
    worker facts; the harness asks for the FACT, not a path it would have to guess)."""
    found: Dict[str, List[Any]] = {k: [] for k in keys}

    def walk(o: Any) -> None:
        if isinstance(o, Mapping):
            for k, v in o.items():
                if k in found:
                    found[k].append(v)
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)

    walk(obj)
    return found


def launch_facts(*, launch: Any, endpoint: Mapping[str, Any], attestation: Mapping[str, Any],
                 composed_calls: Sequence[Mapping[str, Any]], harness_env: Mapping[str, str],
                 versions: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """WHAT THE LAUNCH ACTUALLY COMPOSED AND THE WORKERS ATTESTED — read, never assumed: the
    composed ``ServerLaunch`` (prefix_caching field, the argv token, the env overlay), the endpoint's
    attested engine facts, the attestation's ``enable_prefix_caching`` / batch-invariance worker facts."""
    cmd = tuple(getattr(launch, "cmd", ()) or ())
    env = dict(getattr(launch, "env", {}) or {})
    att_keys = find_keys(attestation, ("enable_prefix_caching", "batch_invariant_env", "batch_invariant_active"))
    bi_launch, bi_harness = env.get(BI_ENV), harness_env.get(BI_ENV)
    effective_bi = 1 if (bi_launch == "1" or (bi_launch is None and bi_harness == "1")) else 0
    rs = endpoint.get("residual_slot") or {}
    geom = endpoint.get("geometry") or {}
    adapters = endpoint.get("adapters") or {}
    return {
        "prefix_caching": {
            "composed": getattr(launch, "prefix_caching", None),
            "argv_off_token": "--no-enable-prefix-caching" in cmd,
            "argv_on_token": "--enable-prefix-caching" in cmd,
            "attested_enable_prefix_caching": att_keys["enable_prefix_caching"],
            "composer_calls": [dict(c) for c in composed_calls],
            "mechanism": PREFIX_CACHING_MECHANISM,
        },
        "batch_invariance": {
            "harness_env": bi_harness, "launch_env": bi_launch,
            "attested_env": att_keys["batch_invariant_env"], "attested_active": att_keys["batch_invariant_active"],
            "effective": effective_bi, "mechanism": BATCH_INVARIANCE_MECHANISM,
        },
        "engine": {
            "engine": endpoint.get("engine"), "serving_engine": endpoint.get("serving_engine"),
            "enforce_eager": endpoint.get("attested_enforce_eager"),
            "cudagraph_mode": endpoint.get("attested_cudagraph_mode"),
            "residual_slot": rs, "output_diagnostics": endpoint.get("output_diagnostics"),
            "lora_cells": endpoint.get("lora_cells"), "cells_skipped": endpoint.get("cells_skipped"),
            "adapters": {c: {k: (b or {}).get(k) for k in ("name", "r", "petri_source_kind", "petri_zero_adapter",
                                                          "residual_fingerprint", "position_mode")}
                         for c, b in adapters.items()},
            "vmodels": endpoint.get("vmodels"),
            "geometry": {"L": endpoint.get("max_model_len"), "S": endpoint.get("max_num_seqs"),
                         "attested_S": endpoint.get("attested_max_num_seqs"),
                         "adjustments": geom.get("adjustments"),
                         "override_reason": endpoint.get("geometry_override_reason"),
                         "lowered": endpoint.get("geometry_lowered")},
            "schema_version": attestation.get("schema_version"),
        },
        "versions": dict(versions or {}),
    }


def protocol_problems(facts: Mapping[str, Any], *, psr_harness_cell: str,
                      L: int = PROTOCOL["L"], S: int = PROTOCOL["S"]) -> List[str]:
    """Every §7.2 precondition, checked against the RECORDED facts; a non-empty list refuses the run
    (a timing row measured on the wrong protocol would be filed beside the lens rows as if comparable)."""
    p: List[str] = []
    pc, bi, eng = facts["prefix_caching"], facts["batch_invariance"], facts["engine"]
    if pc.get("composed") is not False:
        p.append(f"composed prefix_caching is {pc.get('composed')!r}, not False")
    if not pc.get("argv_off_token") or pc.get("argv_on_token"):
        p.append("the composed argv does not carry --no-enable-prefix-caching (or carries the ON token)")
    att_pc = pc.get("attested_enable_prefix_caching") or []
    if not att_pc:
        p.append("the attestation carries no enable_prefix_caching fact")
    elif any(v is not False for v in att_pc):
        p.append(f"attested enable_prefix_caching {att_pc} is not False everywhere")
    if bi.get("harness_env") not in (None, "", "0"):
        p.append(f"the harness environment sets {BI_ENV}={bi.get('harness_env')!r}")
    if bi.get("launch_env") not in (None, "", "0"):
        p.append(f"the launch env sets {BI_ENV}={bi.get('launch_env')!r}")
    if any(bool(v) for v in (bi.get("attested_env") or [])) or any(bool(v) for v in (bi.get("attested_active") or [])):
        p.append(f"the workers attest batch invariance ON: env={bi.get('attested_env')} active={bi.get('attested_active')}")
    if bi.get("effective") != 0:
        p.append(f"effective batch invariance {bi.get('effective')!r} != 0")
    if eng.get("engine") != "lora_native":
        p.append(f"endpoint engine {eng.get('engine')!r} != 'lora_native'")
    if eng.get("enforce_eager") is not False or eng.get("cudagraph_mode") in (None, "NONE"):
        p.append(f"the engine is not graphs-on: enforce_eager={eng.get('enforce_eager')} cudagraph_mode={eng.get('cudagraph_mode')}")
    rs = eng.get("residual_slot") or {}
    if rs.get("armed") is not True:
        p.append(f"residual_slot.armed is {rs.get('armed')!r}, not True")
    if psr_harness_cell not in (rs.get("cells") or {}):
        p.append(f"residual_slot.cells {sorted(rs.get('cells') or {})} does not carry the psr cell {psr_harness_cell!r}")
    if eng.get("output_diagnostics") is not False:
        p.append(f"output_diagnostics is {eng.get('output_diagnostics')!r}, not False")
    if eng.get("cells_skipped"):
        p.append(f"cells_skipped is not empty: {eng.get('cells_skipped')}")
    g = eng.get("geometry") or {}
    if g.get("L") != int(L) or g.get("S") != int(S):
        p.append(f"served geometry {g.get('L')} x {g.get('S')} != the protocol's {L} x {S}")
    if g.get("attested_S") not in (None, int(S)):
        p.append(f"attested max_num_seqs {g.get('attested_S')} != {S}")
    if g.get("adjustments"):
        p.append(f"the geometry was adjusted after the request: {g.get('adjustments')}")
    if g.get("lowered"):
        p.append(f"the geometry was lowered: {g.get('lowered')}")
    return p


def build_row(*, target: str, arm: str, cell: Optional[str], harness_cell: Optional[str], vmodel: str,
              B: int, stats: Mapping[str, Any], counter: Mapping[str, Any], prompt: Mapping[str, Any],
              node: str, sha: Optional[str], date: str, engine: Mapping[str, Any], standup_out: str,
              window_s: float = PROTOCOL["window_s"], warmup_s: float = PROTOCOL["warmup_s"],
              max_tokens: int = PROTOCOL["max_tokens"], L: int = PROTOCOL["L"], S: int = PROTOCOL["S"],
              prefix_caching: bool = PROTOCOL["prefix_caching"],
              batch_invariance: int = PROTOCOL["batch_invariance"]) -> Dict[str, Any]:
    """ONE §7.2 row from one window's stats: the protocol fields, ms/step three ways (the primary is
    the gate's formula), the engine facts, the provenance. ``passthrough_floor_ms`` / ``hook_only_ms``
    are filled by :func:`fill_hook_only` once the ``none`` row at the same B exists."""
    ms = cl.ms_per_step_all(stats, int(B), counter_steps=counter.get("steps"), counter_span_s=counter.get("span_s"))
    return {
        "target": target, "arm": arm, "cell": cell, "harness_cell": harness_cell, "vmodel": vmodel,
        "label": ARM_LABEL_NATIVE, "B": int(B), "geometry": {"L": int(L), "S": int(S)},
        "window_s": float(window_s), "warmup_s": float(warmup_s), "max_tokens": int(max_tokens),
        "exact_length": True, "prefix_caching": bool(prefix_caching), "batch_invariance": int(batch_invariance),
        "ms_per_step": ms, "ms_per_step_primary": PRIMARY_MS_PER_STEP,
        "passthrough_floor_ms": None, "hook_only_ms": None,
        "tok_s": stats.get("server_tok_s"), "tok_s_source": ("engine vllm:generation_tokens_total" if stats.get("server_tok_s") is not None
                                                             else "UNREADABLE (engine counter missing; client_tok_s is a diagnostic only)"),
        "client_tok_s": stats.get("client_tok_s"), "running_mean": stats.get("running_mean"),
        "running_max": stats.get("running_max"), "waiting_max": stats.get("waiting_max"),
        "requests_done": stats.get("requests_done"), "errors": stats.get("errors"), "error_sample": stats.get("error_sample"),
        "latency_p50_s": stats.get("latency_p50_s"), "latency_p95_s": stats.get("latency_p95_s"),
        "server_tokens": stats.get("server_tokens"), "client_tokens": stats.get("client_tokens"),
        "measured_window_s": stats.get("window_s"), "counter": dict(counter),
        "prompt": dict(prompt), "node": node, "sha": sha, "date": date,
        "engine": dict(engine), "standup_out": standup_out,
    }


def fill_hook_only(rows: Sequence[Dict[str, Any]], *, floor_arm: str = "none") -> List[Dict[str, Any]]:
    """``hook_only_ms = ms_per_step(arm) - ms_per_step(none)`` at the same B, per reading, from the SAME
    run's rows (the §7.2 definition); the floor row's own hook-only is 0 by construction; an arm with no
    floor at its B keeps None (never a cross-run floor)."""
    floors: Dict[int, Dict[str, Any]] = {int(r["B"]): r for r in rows if r.get("arm") == floor_arm}
    for r in rows:
        f = floors.get(int(r["B"]))
        if f is None:
            r["passthrough_floor_ms"], r["hook_only_ms"] = None, None
            continue
        fm = f["ms_per_step"]
        r["passthrough_floor_ms"] = dict(fm)
        r["hook_only_ms"] = {k: ((r["ms_per_step"].get(k) - fm.get(k)) if (r["ms_per_step"].get(k) is not None and fm.get(k) is not None) else None)
                             for k in ("gate", "nominal", "counter")}
    return list(rows)


#: the achieved-concurrency floor a window must reach to count (the certification's validity rule,
#: multilora_admission.MIN_RUNNING_MEAN_FRAC — a window whose engine never saw ~B requests in flight
#: measured something else)
MIN_RUNNING_MEAN_FRAC = 0.5
#: the measured window may run long (the last request's tail) but never short of the protocol
MEASURED_WINDOW_TOLERANCE = (0.9, 1.5)
#: the rc of a run whose record is COMPLETE but whose evidence does not validate
RC_INVALID_EVIDENCE = 3


def validate_timing_record(rec: Mapping[str, Any], *, protocol: Mapping[str, Any] = PROTOCOL,
                           arms: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """THE ONE VALIDATOR of a per-target §7.2 record — the producer
    stamps its verdict into the record, the sbatch post-check reads it, and the report REFUSES to
    evaluate a record that fails it (NON-EVALUABLE, stated plainly; never a PASS/FAIL
    over invalid evidence). Requires: ``status == "complete"`` and no ``refused`` / ``terminated``;
    ``protocol_problems == []``; the record's declared ``protocol`` equal to the §7.2 protocol
    (geometry, batches, max_tokens exact, window, warm-up, prefix caching off, BI 0, prompt target);
    the LAUNCH facts re-derived through :func:`protocol_problems` (engine lora_native, graphs on,
    prefix caching composed OFF and attested, BI unset, the residual slot armed with the psr cell,
    output diagnostics off, the 4096 x 64 geometry unadjusted); a tree ``sha`` and a ``node``; and
    per row, for every timed arm at every B exactly once: the protocol fields, ``exact_length``,
    ``errors == 0``, an ENGINE ``tok_s``, ``requests_done >= 1``, ``running_mean >= 0.5 B``, the
    measured window within tolerance, the primary ms/step present, the hook-only present off the floor,
    and the row's ``sha`` / ``node`` equal to the record's. Every problem is named; ``valid`` iff none."""
    problems: List[str] = []
    if not isinstance(rec, Mapping):
        return {"valid": False, "problems": ["record is not a JSON object"], "n_rows": 0}
    if rec.get("status") != "complete":
        problems.append(f"status {rec.get('status')!r} != 'complete'")
    for k in ("refused", "terminated"):
        if rec.get(k):
            problems.append(f"record carries {k}: {str(rec.get(k))[:160]}")
    pp = rec.get("protocol_problems")
    if pp is None:
        problems.append("no protocol_problems recorded (the standup's protocol check never ran)")
    elif pp:
        problems.append(f"protocol_problems recorded: {pp}")
    if not rec.get("sha"):
        problems.append("no tree sha recorded")
    if not rec.get("node"):
        problems.append("no node recorded")
    proto = rec.get("protocol") or {}
    for k in ("L", "S", "max_tokens", "exact_length", "window_s", "warmup_s", "prefix_caching", "batch_invariance",
              "prompt_tokens"):
        if proto.get(k) != protocol[k]:
            problems.append(f"declared protocol {k}={proto.get(k)!r} != {protocol[k]!r}")
    batches = [int(b) for b in (proto.get("batches") or [])]
    if tuple(batches) != tuple(int(b) for b in protocol["batches"]):
        problems.append(f"declared batches {batches} != {list(protocol['batches'])}")
    timed = list(arms) if arms is not None else list(ARMS_TIMED) + (
        ["vector"] if ((rec.get("arms") or {}).get("vector") or {}).get("status") == "present" else [])
    launch = rec.get("launch")
    if not isinstance(launch, Mapping) or not launch:
        problems.append("no launch facts recorded")
    else:
        psr_arm = (rec.get("arms") or {}).get("psr") or {}
        psr_h = psr_arm.get("harness_cell")
        if not psr_h:
            problems.append("no psr harness cell recorded under arms.psr")
        # THE TIMED OBJECT IS ONE NAME: the record's top-level `cell` — what the
        # report's caveat and figure are keyed on — must be arms.psr.cell, translate (the static registry
        # rule, registry_adapter.harness_cell_id) to arms.psr.harness_cell, the id the attested residual
        # slot must carry (protocol_problems below), and match every psr row's identity (the rows loop);
        # a re-labelled top-level cell can never claim the like-for-like caveat over rows measured on
        # another object
        if rec.get("cell") != psr_arm.get("cell"):
            problems.append(f"record cell {rec.get('cell')!r} != arms.psr.cell {psr_arm.get('cell')!r}")
        if psr_h and rec.get("cell"):
            try:
                from sisteer.capeval.petri import registry_adapter as _ra
                translated = _ra.harness_cell_id(str(rec.get("cell")))
            except Exception as e:  # noqa: BLE001 — an untranslatable cell name is itself the problem
                problems.append(f"record cell {rec.get('cell')!r} has no harness id: {type(e).__name__}: {e}")
            else:
                if translated != psr_h:
                    problems.append(f"record cell {rec.get('cell')!r} translates to harness id {translated!r} != "
                                    f"the attested psr cell {psr_h!r}")
        try:
            problems += [f"launch: {q}" for q in protocol_problems(
                launch, psr_harness_cell=str(psr_h or ""), L=int(protocol["L"]), S=int(protocol["S"]))]
        except (KeyError, TypeError, AttributeError) as e:
            problems.append(f"launch facts unreadable: {type(e).__name__}: {e}")
    rows = rec.get("rows") or []
    if not isinstance(rows, list):
        problems.append("rows is not a list")
        rows = []
    seen: Dict[Tuple[str, int], int] = {}
    for r in rows:
        key = (str(r.get("arm")), int(r.get("B", -1)))
        seen[key] = seen.get(key, 0) + 1
    want = {(a, int(B)) for a in timed for B in protocol["batches"]}
    missing = sorted(want - set(seen))
    if missing:
        problems.append(f"rows missing: {missing}")
    dupes = sorted(k for k, n in seen.items() if n > 1)
    if dupes:
        problems.append(f"duplicate rows: {dupes}")
    stray = sorted(set(seen) - want)
    if stray:
        problems.append(f"rows outside the protocol's arms x batches: {stray}")
    for r in rows:
        tag = f"{r.get('arm')}@B{r.get('B')}"
        if (r.get("geometry") or {}) != {"L": int(protocol["L"]), "S": int(protocol["S"])}:
            problems.append(f"{tag}: geometry {r.get('geometry')} != the protocol's")
        for k in ("window_s", "warmup_s", "max_tokens", "prefix_caching", "batch_invariance"):
            if r.get(k) != protocol[k]:
                problems.append(f"{tag}: {k}={r.get(k)!r} != {protocol[k]!r}")
        if r.get("exact_length") is not True:
            problems.append(f"{tag}: exact_length is not True")
        if r.get("arm") == "psr":
            psr_arm_ = (rec.get("arms") or {}).get("psr") or {}
            if r.get("cell") != rec.get("cell") or r.get("harness_cell") != psr_arm_.get("harness_cell"):
                problems.append(f"{tag}: row identity cell={r.get('cell')!r} harness_cell={r.get('harness_cell')!r} != the "
                                f"record's ({rec.get('cell')!r}, {psr_arm_.get('harness_cell')!r})")
        errors = r.get("errors")
        if not isinstance(errors, int) or isinstance(errors, bool):
            problems.append(f"{tag}: errors={errors!r} is not an integer count (a window with no error count is not "
                            "error-free evidence)")
        elif errors != 0:
            problems.append(f"{tag}: {errors} request error(s) in the window ({str(r.get('error_sample'))[:120]})")
        if r.get("tok_s") is None:
            problems.append(f"{tag}: no engine tok/s (the /metrics counter was unreadable)")
        if not isinstance(r.get("requests_done"), int) or r.get("requests_done") < 1:
            problems.append(f"{tag}: requests_done={r.get('requests_done')!r} < 1")
        rm = r.get("running_mean")
        if not isinstance(rm, (int, float)) or rm < MIN_RUNNING_MEAN_FRAC * int(r.get("B", 0)):
            problems.append(f"{tag}: running_mean {rm!r} < {MIN_RUNNING_MEAN_FRAC} x B")
        mw = r.get("measured_window_s")
        if isinstance(mw, (int, float)):
            lo, hi = MEASURED_WINDOW_TOLERANCE
            if not (lo * float(protocol["window_s"]) <= mw <= hi * float(protocol["window_s"])):
                problems.append(f"{tag}: measured window {mw:.1f}s outside [{lo}, {hi}] x {protocol['window_s']}s")
        else:
            problems.append(f"{tag}: no measured_window_s")
        prim = r.get("ms_per_step_primary") or "gate"
        if ((r.get("ms_per_step") or {}).get(prim)) is None:
            problems.append(f"{tag}: no primary ms/step ({prim})")
        if r.get("arm") != "none" and ((r.get("hook_only_ms") or {}).get(prim)) is None:
            problems.append(f"{tag}: no hook-only ({prim}) — the floor row at this B is missing")
        for k in ("sha", "node"):
            if r.get(k) != rec.get(k):
                problems.append(f"{tag}: {k} {r.get(k)!r} != the record's {rec.get(k)!r}")
    return {"valid": not problems, "problems": problems, "n_rows": len(rows), "arms": timed,
            "rule": ("complete status, the §7.2 protocol declared and re-derived from the launch facts, ONE timed "
                     "object (record cell == arms.psr.cell, translating to the attested residual cell, == every psr "
                     "row's identity), every arm x B row present once with zero request errors, an engine tok/s, "
                     "achieved concurrency >= 0.5 B and the measured window in tolerance; anything else is "
                     "NON-EVALUABLE evidence")}


def window_plan(batches: Sequence[int], arms: Sequence[str] = ARMS_TIMED) -> List[Tuple[int, str]]:
    """B outer, arm inner: the floor and the arm at one B are ADJACENT windows of the same run, so the
    hook-only subtraction pairs the closest readings in time."""
    return [(int(B), arm) for B in batches for arm in arms]


def tokenizer_count_fn(tokenizer: Any) -> Callable[[str], int]:
    """The served tokenizer's chat render of one user message (``add_generation_prompt=True``, the
    engine's own render of the request) as a token count — the gate counted through the proxy's
    ``/tokenize`` on the same render."""
    def count(text: str) -> int:
        out = tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                            tokenize=True, return_dict=False)
        if isinstance(out, Mapping):
            out = out["input_ids"]
        if hasattr(out, "input_ids"):
            out = out.input_ids
        return len(list(out))
    return count


def exact_length_probe(proxy_url: str, api_key: str, vmodel: str, prompt: str, *, max_tokens: int,
                       timeout_s: float = 600.0) -> Dict[str, Any]:
    """ONE request with ``min_tokens`` + ``ignore_eos`` through the proxy: the engine must honour the
    exact length (``completion_tokens == max_tokens``) or the protocol is refused (a 400 naming either
    key, or a shorter completion, means the windows would not be exact-length decodes). Returns the
    ``usage`` block — ``prompt_tokens`` is the SERVED count of the timing prompt."""
    url = proxy_url.rstrip("/") + "/chat/completions"
    body = {"model": vmodel, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": int(max_tokens), "min_tokens": int(max_tokens), "ignore_eos": True,
            "temperature": 0.0, "seed": 0}
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            resp = json.loads(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        text = e.read().decode("utf-8", "replace")[:500]
        raise TimingRefused(f"exact-length probe on {vmodel!r} failed: HTTP {e.code}: {text}") from e
    usage = dict(resp.get("usage") or {})
    if int(usage.get("completion_tokens") or 0) != int(max_tokens):
        raise TimingRefused(f"the engine did not honour the exact length on {vmodel!r}: usage={usage} (wanted "
                            f"completion_tokens == {max_tokens}: min_tokens + ignore_eos)")
    return usage


# --------------------------------------------------------------------------- #
# the run                                                                        #
# --------------------------------------------------------------------------- #
class _Terminated(Exception):
    pass


def _on_term(signum: int, _frame: Any) -> None:
    raise _Terminated(f"signal {signum}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True, help="the canonical model key (llama8b | gemma4e4b | qwen35_4b | nemotron3nano4b)")
    ap.add_argument("--view", required=True, help="the bank view (psrcert | main | ...): names the config + manifest")
    ap.add_argument("--cell", default=DEFAULT_PSR_CELL, help="the all-positions psr cell (TrainTest id)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--repo-dir", default=str(REPO), type=Path)
    ap.add_argument("--defenses-dir", required=True, type=Path, help="the (target, view) bank view (f22_bank_view)")
    ap.add_argument("--adapter-bank", default=None, type=Path, help="explicit adapter bank (default: the cluster bank)")
    ap.add_argument("--expect-object-sha256", action="append", default=[], metavar="CELL=SHA",
                    help="the pinned object digest per served cell (required by the standup)")
    ap.add_argument("--config", default=None, type=Path, help="override the config the view names")
    ap.add_argument("--manifest", default=None, type=Path, help="override the manifest the view names")
    ap.add_argument("--batches", default=",".join(str(b) for b in PROTOCOL["batches"]))
    ap.add_argument("--window-s", type=float, default=PROTOCOL["window_s"])
    ap.add_argument("--warmup-s", type=float, default=PROTOCOL["warmup_s"])
    ap.add_argument("--max-tokens", type=int, default=PROTOCOL["max_tokens"])
    ap.add_argument("--prompt-tokens", type=int, default=PROTOCOL["prompt_tokens"], help="the gate's timing-prompt token target")
    ap.add_argument("--max-model-len", type=int, default=PROTOCOL["L"])
    ap.add_argument("--max-num-seqs", type=int, default=PROTOCOL["S"])
    ap.add_argument("--gpu-mem-util", default="0.9")
    ap.add_argument("--ready-timeout-s", type=int, default=1800)
    ap.add_argument("--probe-max-tokens", type=int, default=48)
    ap.add_argument("--vector-cell", default=None, help="an all-positions vector-kind cell to time as the vector arm (stage 1: none exists, D4)")
    return ap


def parse_batches(text: str) -> List[int]:
    out = [int(b) for b in str(text).split(",") if b.strip()]
    if not out or any(b < 1 for b in out):
        raise TimingRefused(f"--batches must be a comma list of positive ints, got {text!r}")
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if os.environ.get(BI_ENV) not in (None, "", "0"):
        raise TimingRefused(f"{BI_ENV}={os.environ.get(BI_ENV)!r} in the harness environment — the §7.2 protocol is BI unset")
    import lora_serving_up as lsu  # noqa: E402  (the standup seam; imported here so the pure helpers stay CPU-importable)
    from sisteer.capeval.petri import registry_adapter as ra
    from sisteer.capeval.serving.run_identity import serving_versions, tree_sha

    repo = Path(args.repo_dir).resolve()
    cfg, man = config_paths(repo, args.target, args.view)
    cfg = Path(args.config) if args.config else cfg
    man = Path(args.manifest) if args.manifest else man
    for p in (cfg, man):
        if not p.exists():
            raise TimingRefused(f"missing {p} (target {args.target}, view {args.view})")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "windows").mkdir(exist_ok=True)
    batches = parse_batches(args.batches)
    cells = [args.cell] + ([args.vector_cell] if args.vector_cell else [])
    arms: Dict[str, Dict[str, Any]] = {"none": {"cell": None}, "psr": {"cell": args.cell},
                                       "vector": vector_arm_record(args.target, args.vector_cell)}
    timed: List[str] = list(ARMS_TIMED) + (["vector"] if args.vector_cell else [])
    node, sha, started = socket.gethostname(), tree_sha(repo), time.monotonic()
    record: Dict[str, Any] = {
        "schema": SCHEMA, "target": args.target, "view": args.view, "cell": args.cell, "eval": EVAL_NAME,
        "protocol": {**PROTOCOL, "batches": batches, "L": int(args.max_model_len), "S": int(args.max_num_seqs),
                     "window_s": float(args.window_s), "warmup_s": float(args.warmup_s),
                     "max_tokens": int(args.max_tokens), "prompt_tokens": int(args.prompt_tokens)},
        "arms": arms, "rows": [], "windows_order": [f"{arm}@B{B}" for B, arm in window_plan(batches, timed)],
        "node": node, "sha": sha, "date": utc_now(), "status": "starting", "out": str(out),
        "standup_out": str(out / "standup"), "config": str(cfg), "manifest": str(man),
        "harness_env": {BI_ENV: os.environ.get(BI_ENV), "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID")},
    }
    target_json = out / f"{args.target}.json"
    write_json_atomic(target_json, record)
    kw = standup_kwargs(repo_dir=repo, config=cfg, manifest=man, defenses_dir=args.defenses_dir, out=out / "standup",
                        target=args.target, cells=cells, expect_object_sha256=args.expect_object_sha256,
                        adapter_bank=args.adapter_bank, L=args.max_model_len, S=args.max_num_seqs,
                        gpu_mem_util=args.gpu_mem_util, ready_timeout_s=args.ready_timeout_s,
                        probe_max_tokens=args.probe_max_tokens)
    record["standup_kwargs"] = dict(kw)
    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)
    su: Any = None
    rc = 1
    try:
        log(f"native standup: {args.target}/{args.view} cells={cells} at {args.max_model_len} x {args.max_num_seqs} "
            f"(prefix caching OFF bound on the composer; {BI_ENV} unset)")
        t0 = time.monotonic()
        with prefix_caching_off() as composed:
            su = lsu.bring_up_lora(**kw)
        record["standup_ready_s"] = time.monotonic() - t0
        endpoint = json.loads(Path(su.endpoint_path).read_text())
        attestation = sa.read_serving_attestation(Path(su.attestation_path))
        facts = launch_facts(launch=su.launch, endpoint=endpoint, attestation=attestation, composed_calls=composed,
                             harness_env=os.environ, versions=serving_versions())
        record["launch"] = facts
        psr_h = ra.harness_cell_id(args.cell)
        problems = protocol_problems(facts, psr_harness_cell=psr_h, L=args.max_model_len, S=args.max_num_seqs)
        record["protocol_problems"] = problems
        write_json_atomic(target_json, record)
        if problems:
            raise TimingRefused("the standup does not satisfy the §7.2 protocol:\n  - " + "\n  - ".join(problems))
        vmodels = {"none": su.base_vmodel, "psr": su.vmodels[psr_h], "zero": su.zero_vmodel}
        if args.vector_cell:
            vmodels["vector"] = su.vmodels[ra.harness_cell_id(args.vector_cell)]
        arms["none"].update(vmodel=vmodels["none"], harness_cell="none")
        arms["psr"].update(vmodel=vmodels["psr"], harness_cell=psr_h)
        if args.vector_cell:
            arms["vector"].update(vmodel=vmodels["vector"], harness_cell=ra.harness_cell_id(args.vector_cell))
        record["zero_vmodel"] = vmodels["zero"]
        # the gate's ~800-token passage through the SERVED tokenizer's render, then the served count
        from sisteer.capeval.serving import proxy as px
        api_key = px.proxy_client_token(endpoint)
        prompt = cl.build_timing_prompt(tokenizer_count_fn(su.tokenizer), target_tokens=int(args.prompt_tokens))
        prompt["sha256"] = hashlib.sha256(prompt["text"].encode()).hexdigest()
        usage = exact_length_probe(su.proxy_url, api_key, vmodels["none"], prompt["text"], max_tokens=int(args.max_tokens))
        prompt["served_prompt_tokens"] = usage.get("prompt_tokens")
        prompt["exact_length_probe_usage"] = usage
        record["prompt"] = {k: v for k, v in prompt.items() if k != "text"}
        (out / "timing_prompt.txt").write_text(prompt["text"], encoding="utf-8")
        log(f"timing prompt: {prompt['repeats']} repeats, {prompt['tokens']} tokens by the served tokenizer, "
            f"{usage.get('prompt_tokens')} served; exact length {usage.get('completion_tokens')} == {args.max_tokens}")
        metrics_url = f"{su.launch.base_url}/metrics"
        headers = dict(su.launch.auth_headers)
        engine_facts = {k: facts["engine"][k] for k in ("enforce_eager", "cudagraph_mode", "residual_slot", "output_diagnostics",
                                                        "adapters", "lora_cells", "schema_version")}
        engine_facts["prefix_caching"] = {k: facts["prefix_caching"][k] for k in ("composed", "argv_off_token", "attested_enable_prefix_caching")}
        engine_facts["batch_invariance"] = {k: facts["batch_invariance"][k] for k in ("launch_env", "attested_env", "attested_active", "effective")}
        record["status"] = "partial"
        rows: List[Dict[str, Any]] = []
        prompt_rec = {k: prompt[k] for k in ("tokens", "repeats", "target_tokens", "sha256", "served_prompt_tokens")}
        for B, arm in window_plan(batches, timed):
            vm = vmodels[arm]
            log(f"window {arm}@B{B}: {vm} — warm-up {args.warmup_s}s, window {args.window_s}s, max_tokens {args.max_tokens} exact")
            sampler = cl.StepCounterSampler(metrics_url, headers, sample_s=1.0).start()
            with cl.prompt_bound(prompt["text"]):
                stats, gauges = asyncio.run(cl.closed_loop_via_proxy(
                    su.proxy_url, api_key, vm, metrics_url=metrics_url, metrics_headers=headers, B=int(B),
                    max_tokens=int(args.max_tokens), warmup_s=float(args.warmup_s), window_s=float(args.window_s),
                    exact_length=True))
            samples = sampler.stop()
            counter = cl.counter_span_over_gauges(samples, gauges)
            write_json_atomic(out / "windows" / f"gauges_{arm}_B{B}.json", {"gauges": gauges, "counter_samples": samples, "stats": stats})
            row = build_row(target=args.target, arm=arm, cell=arms[arm].get("cell"), harness_cell=arms[arm].get("harness_cell"),
                            vmodel=vm, B=B, stats=stats, counter=counter, prompt=prompt_rec, node=node, sha=sha, date=utc_now(),
                            engine=engine_facts, standup_out=str(out / "standup"), window_s=args.window_s, warmup_s=args.warmup_s,
                            max_tokens=args.max_tokens, L=args.max_model_len, S=args.max_num_seqs)
            rows.append(row)
            fill_hook_only(rows)
            record["rows"] = rows
            write_json_atomic(target_json, record)
            ms = row["ms_per_step"]
            log(f"  {arm}@B{B}: tok/s {row['tok_s'] and round(row['tok_s'], 1)}; ms/step gate {ms['gate'] and round(ms['gate'], 2)} "
                f"nominal {ms['nominal'] and round(ms['nominal'], 2)} counter {ms['counter'] and round(ms['counter'], 2)}; "
                f"running mean {row['running_mean']}; requests {row['requests_done']}; errors {row['errors']}; "
                f"hook-only(gate) {(row['hook_only_ms'] or {}).get('gate')}")
        record["status"] = "complete"
        record["elapsed_s"] = time.monotonic() - started
        # the ONE validator judges the evidence; a complete record whose evidence does not validate
        # (an errored window, a short window, an unreachable concurrency) is stamped INVALID and the
        # run exits non-zero — the report will refuse to evaluate it
        record["validation"] = validate_timing_record(record)
        write_json_atomic(target_json, record)
        if record["validation"]["valid"]:
            rc = 0
            log(f"complete and VALID: {len(rows)} rows -> {target_json}")
        else:
            rc = RC_INVALID_EVIDENCE
            log(f"complete but INVALID evidence (rc {rc}): {record['validation']['problems']} -> {target_json}")
    except _Terminated as e:
        record["status"], record["terminated"] = "partial", str(e)
        write_json_atomic(target_json, record)
        log(f"terminated ({e}); partial record written")
        rc = 143
    except TimingRefused as e:
        record["status"], record["refused"] = "refused", str(e)
        write_json_atomic(target_json, record)
        log(f"REFUSED: {e}")
        rc = 2
    finally:
        if su is not None:
            try:
                su.tear_down()
                log("standup torn down")
            except Exception as e:  # noqa: BLE001
                log(f"tear_down raised {type(e).__name__}: {e}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
