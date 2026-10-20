#!/usr/bin/env python
"""The vllm-lens x vllm 0.25.1+cu129 compatibility probe.

WHY THIS EXISTS. ``vllm-lens==1.2.1`` is dev-locked against
``vllm==0.19.0``; the serving venv pins ``vllm==0.25.1+cu129`` (the only
wheel that loads on driver-570 nodes). vllm-lens's documented
failure mode is SILENTLY CAPTURING NOTHING: if its hook site does not
match 0.25.1's forward pass, the hook never fires, generation still
succeeds fluently, and every "steered" cell silently returns base
behaviour. This probe makes that impossible to miss: it proves, on a
real engine on a real GPU, that the hook FIRED (on every generated
token) and MODIFIED output — or fails loudly naming what didn't hold.
A probe that merely imports vllm_lens, or merely generates text, would
be worthless.

WHAT IT PROVES. The evidence feeds the EXISTING, CPU-tested Phase-0
assertion logic (``lens_steer.run_phase0`` — the checks are not
reimplemented here; this script only GATHERS):

1. server_provenance      — engine up with the lens plugin active
                            (patched entry points + worker extension +
                            forced enforce_eager, read from the LIVE
                            engine config), pinned package versions.
2. saved_sentinel         — a persistent hook whose ctx.saved records
                            every fire: fired at prefill AND on the
                            forward pass that produced EVERY generated
                            token, exactly once each (never prefill-only).
3. mixed_batch_isolation  — one batch mixing a hooked and an unhooked
                            request: neither's output changes vs solo.
4. returned_modification  — the steering delta that LANDED in the
                            residual stream equals the requested vector
                            (relative vector-residual metric, STATED atol).
5. disable_flag           — a separate VLLM_LENS_DISABLE=1 process:
                            nothing captured, output == unhooked baseline
                            even when the request asks for steering.
6. single_persistent_hook — hooks registered exactly once (counted at
                            the collective_rpc seam); requests carry only
                            data selections, never re-registrations.
7. prefix_cache_isolation — prefix caching verified OFF in the live
                            config; identical repeated request →
                            identical tokens.
8. scale0_noop            — the SAME steering vector at scale 0.0
                            (armed hook, zero add) reproduces the
                            unhooked baseline token-identically.

Plus probe-level checks with the same fail-loud discipline (this is a
gate, not a survey — ANY failure exits nonzero naming the assertion):

- steering_changed_output — steered greedy tokens differ from the
  unhooked baseline on the same fixed prompt (the core "not silently
  doing nothing" proof; scale is self-calibrated off measured hidden
  norms so "no visible change" can never be a scale accident).
- sentinel_fire_accounting — the raw fire log is internally coherent:
  all fires at the registered layer, prefill fires cover exactly the
  prompt tokens, and at most ONE trailing scheduler-committed extra
  decode pass (vllm-lens's own ``_trim_activations`` documents vLLM v1
  committing one post-final step; >1 is anomalous).
- sentinel_noninvasive / hook_teardown_clean — a non-modifying hook
  changes nothing; ``clear_hooks`` leaves no residue.
- mixed_batch_cobatched — the two mixed-batch requests' hook-fire
  ticks INTERLEAVE (a worker-global fire counter), proving they truly
  shared forward passes rather than being serialized.
- disable_flag_unpatched — the disabled process shows NO lens patches.

OUTCOME DISCIPLINE (the gate's pattern, scripts/petri_serving_gate.py):
the distinction between UNAVAILABLE and INCOMPATIBLE is explicit in the
JSON —

- vllm/vllm_lens unimportable, no GPU, or the ENGINE CANNOT INIT →
  ``probe_unavailable.json`` (fail loud, record, exit 0): a launch
  blocker to escalate, not a verdict on vllm-lens. On an init failure
  under VLLM_BATCH_INVARIANT=1 (the audit's exact-replay mode) the probe
  retries once without it and records the downgrade — that failure
  redirects the audit to the proxy-serialization replay strategy.
- engine up but an assertion failed (the hook never fired, output
  unmodified, contamination, ...) → ``probe_failed.json`` + exit
  NONZERO naming every failed assertion: vllm-lens 1.2.1 is
  INCOMPATIBLE with vllm 0.25.1 as pinned.
- everything held → ``probe_record.json`` + exit 0.

ARCHITECTURE. Three processes, because the disable-flag evidence needs
a separate process (the flag is read once at plugin registration) and
GPU memory needs a clean handoff between engines:

  probe (parent, this file)   — orchestration + evidence assembly +
                                run_phase0; no GPU work of its own.
  worker-active (subprocess)  — the lens-active engine; gathers
                                everything except disable_flag.
  worker-disabled (subprocess)— VLLM_LENS_DISABLE=1, stock engine
                                (enforce_eager/prefix-caching pinned
                                explicitly so tokens are comparable).

MECHANICS PINNED AGAINST THE ACTUAL vllm-lens 1.2.1 WHEEL (read from
PyPI, 2026-08-20): steering rides ``SamplingParams.extra_args
["apply_steering_vectors"]`` in the documented JSON wire form (the
same form the HTTP path uses; its offline tests prove wire == live
equivalence at 0.19). The sentinel is a PERSISTENT hook
(``LLM.register_hooks`` / ``collect_hook_results``) rather than a
per-request ``apply_hooks`` hook: ``get_all_hook_results`` returns
results keyed by raw internal request id with no prefix matching,
where the per-request ``get_hook_results`` path prefix-matches
``"{external_id}-"`` — an id-scheme assumption 0.25.1 need not satisfy
offline — and a single registration is exactly the audit's H3 shape
(hooks installed once at server start; requests carry data selections
only). NOTE deliberately absent: VLLM_PLUGINS is POPPED from the child
env — vllm filters general plugins by ENTRY-POINT NAME, and vllm-lens's
entry point is named ``activations``, so an inherited
``VLLM_PLUGINS=vllm_lens`` would silently filter the plugin OUT (the
exact silent-nothing failure class this probe exists to catch; the
provenance assertion would then fail loudly here rather than at audit
time).

Run on the cluster via ``scripts/sbatch/petri_probe.sbatch`` from the
immutable commit-addressed checkout. Results land under
``--out-dir`` (durable, on $SHARED): ``probe_evidence.json`` always,
plus exactly one of the three outcome markers above.

The pure evidence-assembly helpers (fire→token attribution, tick
interleaving, the relative-residual metric, the phase-0 evidence
builder) are CPU-tested in ``tests/test_petri_lens_probe.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:  # script-mode import path
    sys.path.insert(0, str(REPO_ROOT / "src"))

# --- probe constants (pre-registered; part of what the record means) ----------

#: fixed greedy probe prompts — plain, harmless; the probe proves SERVING
#: mechanics, not behaviour. P0 is the sentinel/steering prompt, P1 the
#: second mixed-batch member. Both are several tokens long so a prefill
#: pass can never masquerade as a decode pass (seq_len > 1 vs == 1).
PROMPT_MAIN = "The first five prime numbers, listed in ascending order, are"
PROMPT_ALT = "A well-known fact about water is that at sea level it boils at"
#: greedy decode budget; ignore_eos pins every run to exactly this many
#: generated tokens, so per-token fire attribution and token-list
#: comparisons are well-defined even when steering changes content.
PROBE_MAX_TOKENS = 32
#: steering magnitude relative to the measured mean prompt-position
#: hidden-state L2 norm at the probed layer — large enough that greedy
#: output MUST change if the add lands, small enough to stay numerically
#: ordinary.
ALPHA_REL = 1.5
#: fallback absolute steering norm when calibration is impossible (the
#: sentinel never fired, so the probe is failing anyway and just needs a
#: recorded, deterministic vector for the remaining legs).
ALPHA_FALLBACK = 60.0
#: returned_modification metric + STATED atol (lens_steer requires the
#: gate to state it, never default it): max over prompt positions of
#: ||(steered_hidden - base_hidden) - v||_2 / ||v||_2. A landed add
#: measures ~1e-3 (two extra bf16 roundings scale with ||h||, and
#: ||v|| = ALPHA_REL * ||h||); a silently-missing add measures ~1.0.
RETMOD_ATOL = 0.25
RETMOD_METRIC = ("max over prompt positions of "
                 "||(steered_hidden - base_hidden) - v||_2 / ||v||_2 "
                 "at the probed layer (fp32; rows saved by the sentinel "
                 "hook inside the worker)")
#: deterministic direction seed for the steering vector.
STEER_SEED = 20260820

UNAVAILABLE_MARKER = "probe_unavailable.json"
FAILURE_MARKER = "probe_failed.json"
PROBE_RECORD = "probe_record.json"
EVIDENCE_FILE = "probe_evidence.json"

#: child exit codes — the parent's classification contract.
EXIT_ENGINE_INIT = 3   # engine could not come up  -> unavailable path
EXIT_REQUESTS = 4      # engine up, a request step raised -> incompatible
CHILD_TIMEOUT_S = 1500


# --- pure helpers (CPU-tested; no vllm imports) --------------------------------


def match_internal_id(internal: str, external: str) -> bool:
    """vLLM may pass request ids through verbatim or append a
    ``-{suffix}`` (version/entry-point dependent); accept both, refuse
    lookalikes (``"11-x"`` never matches external ``"1"``)."""
    return internal == external or internal.startswith(f"{external}-")


def build_saved_sentinel(fires: Sequence[Mapping[str, Any]], *,
                         n_prompt: int, n_gen: int, layer: int
                         ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Fold the sentinel's raw fire log into ``saved_sentinel`` evidence
    plus an accounting record.

    Attribution contract (the assertion's semantic is "the hook fired on
    the forward pass that PRODUCED every generated token"):

    - token 1 is produced by the LAST prefill pass (prefill may be
      chunked; the final chunk's forward samples the first token);
    - token i (i >= 2) is produced by decode pass i-1;
    - vLLM v1's scheduler can commit ONE trailing decode pass after the
      final token (vllm-lens's ``_trim_activations`` documents it); that
      pass produces no token, so it is excluded from attribution and
      bounded (<= 1) by the accounting record instead.

    ``decode_step_fires[i]`` is therefore exactly 1 iff token i+1's
    producing pass saw the hook fire. The evidence goes to
    ``lens_steer._check_saved_sentinel`` unchanged; the accounting dict
    backs the probe-level ``sentinel_fire_accounting`` check.
    """
    fires = sorted(fires, key=lambda f: f.get("tick", 0))
    off_layer = [f for f in fires if int(f["layer"]) != int(layer)]
    on_layer = [f for f in fires if int(f["layer"]) == int(layer)]
    prefill = [f for f in on_layer if int(f["seq_len"]) > 1]
    decode = [f for f in on_layer if int(f["seq_len"]) == 1]

    step_fires: List[int] = []
    if n_gen >= 1:
        step_fires.append(1 if prefill else 0)      # token 1 <- last prefill
        for i in range(2, n_gen + 1):               # token i <- decode i-1
            step_fires.append(1 if len(decode) >= i - 1 else 0)

    evidence = {
        "prefill_fires": len(prefill),
        "decode_fires": sum(step_fires),
        "generated_tokens": int(n_gen),
        "decode_step_fires": step_fires,
    }
    prefill_token_coverage = sum(int(f["seq_len"]) for f in prefill)
    extra = len(decode) - max(0, n_gen - 1)
    accounting = {
        "off_layer_fires": len(off_layer),
        "prefill_passes": len(prefill),
        "prefill_token_coverage": prefill_token_coverage,
        "n_prompt": int(n_prompt),
        "decode_passes_observed": len(decode),
        "expected_decode_passes": max(0, n_gen - 1),
        "post_final_extra_passes": max(0, extra),
        "total_fires": len(fires),
        "ok": (not off_layer
               and (prefill_token_coverage == n_prompt or not prefill)
               and 0 <= extra <= 1
               and n_gen >= 1),
        "detail": ("fires must all land on the registered layer, prefill "
                   "fires must cover exactly the prompt tokens, and at "
                   "most one trailing scheduler-committed decode pass may "
                   "trail the final token"),
    }
    return evidence, accounting


def ticks_interleave(ticks_a: Sequence[int], ticks_b: Sequence[int]) -> bool:
    """True iff the two requests' fire-tick ranges overlap — i.e. at
    least one of A's forward passes happened between two of B's (or vice
    versa), which serialized (never co-batched) execution cannot
    produce. Empty logs never interleave (vacuous evidence != proof)."""
    if not ticks_a or not ticks_b:
        return False
    return not (max(ticks_a) < min(ticks_b) or max(ticks_b) < min(ticks_a))


def max_relative_residual(base_rows, steered_rows, v) -> float:
    """The returned_modification metric (see RETMOD_METRIC). Pure
    torch, fp32; caller guarantees aligned shapes (same prompt, same
    layer)."""
    import torch
    base = base_rows.to(torch.float32)
    steered = steered_rows.to(torch.float32)
    vec = v.to(torch.float32).reshape(-1)
    if base.shape != steered.shape or base.shape[-1] != vec.shape[0]:
        raise ValueError(f"misaligned rows: base {tuple(base.shape)} vs "
                         f"steered {tuple(steered.shape)} vs v "
                         f"{tuple(vec.shape)}")
    resid = ((steered - base) - vec).norm(dim=-1)
    return float((resid / vec.norm().clamp_min(1e-9)).max().item())


def assemble_phase0_evidence(active: Mapping[str, Any],
                             disabled: Mapping[str, Any],
                             *,
                             expected_packages: Mapping[str, str],
                             retmod_delta: Optional[float],
                             retmod_atol: float) -> Dict[str, Dict[str, Any]]:
    """Map the two workers' raw gatherings onto the 8 Phase-0 evidence
    dicts ``lens_steer.run_phase0`` consumes. Missing pieces are OMITTED
    (never defaulted) so the existing checkers fail them by name."""
    runs = active.get("runs", {})
    engine = active.get("engine", {})
    plugin = active.get("plugin", {})
    rpc = active.get("rpc_counts", {})

    def tokens(name: str) -> Optional[List[int]]:
        r = runs.get(name)
        return None if r is None else r.get("tokens")

    ev: Dict[str, Dict[str, Any]] = {}

    ev["server_provenance"] = {
        "expected_packages": dict(expected_packages),
        "reported_packages": dict(active.get("packages", {})),
        "plugin_registered": bool(plugin.get("generate_patched"))
        and bool(plugin.get("engine_args_patched"))
        and engine.get("worker_extension_cls")
        == "vllm_lens._worker_ext.HiddenStatesExtension",
        "enforce_eager": engine.get("enforce_eager"),
    }

    sentinel = active.get("saved_sentinel_evidence")
    if sentinel is not None:
        ev["saved_sentinel"] = dict(sentinel)

    mixed = {"unhooked_in_batch_tokens": tokens("mixed_unhooked"),
             "unhooked_solo_tokens": tokens("baseline"),
             "hooked_in_batch_tokens": tokens("mixed_hooked"),
             "hooked_solo_tokens": tokens("steered_alt_solo")}
    ev["mixed_batch_isolation"] = {k: v for k, v in mixed.items()
                                   if v is not None}

    retmod: Dict[str, Any] = {"stated_atol": retmod_atol,
                              "metric": RETMOD_METRIC}
    if retmod_delta is not None:
        retmod["max_abs_delta"] = retmod_delta
    ev["returned_modification"] = retmod

    dis_runs = disabled.get("runs", {})
    dis: Dict[str, Any] = {
        "saved_count_with_disable": disabled.get("saved_count", None),
    }
    if dis_runs.get("steer_attempt") is not None:
        dis["tokens_with_disable"] = dis_runs["steer_attempt"].get("tokens")
    if tokens("baseline") is not None:
        dis["baseline_tokens"] = tokens("baseline")
    ev["disable_flag"] = {k: v for k, v in dis.items() if v is not None}

    ev["single_persistent_hook"] = {
        "registration_count": rpc.get("set_persistent_hooks", 0),
        "per_request_registrations": (
            rpc.get("set_hook_data", 0)
            + max(0, rpc.get("set_persistent_hooks", 0) - 1)
            + max(0, rpc.get("install_hooks", 0) - 1)),
        "requests_observed": active.get("requests_observed_with_hook", 0),
    }

    pci = {"prefix_caching_enabled": engine.get("prefix_caching_enabled"),
           "first_run_tokens": tokens("baseline"),
           "second_run_tokens": tokens("baseline_repeat")}
    ev["prefix_cache_isolation"] = {k: v for k, v in pci.items()
                                    if v is not None}

    sc = {"scale0_tokens": tokens("scale0"),
          "baseline_tokens": tokens("baseline")}
    ev["scale0_noop"] = {k: v for k, v in sc.items() if v is not None}
    return ev


def evaluate_probe_checks(active: Mapping[str, Any],
                          disabled: Mapping[str, Any],
                          accounting: Optional[Mapping[str, Any]]
                          ) -> List[Dict[str, Any]]:
    """The probe-level checks (same fail-loud discipline as Phase 0).
    Returns a list of {name, ok, detail} — any not-ok fails the probe."""
    runs = active.get("runs", {})
    fires = active.get("fires_by_run", {})

    def tokens(name: str) -> Optional[List[int]]:
        r = runs.get(name)
        return None if r is None else r.get("tokens")

    checks: List[Dict[str, Any]] = []

    def add(name: str, ok: Optional[bool], detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    b0, s0 = tokens("baseline"), tokens("steered")
    add("steering_changed_output",
        bool(b0) and bool(s0) and s0 != b0,
        "steered greedy tokens must differ from the unhooked baseline on "
        f"the same prompt (baseline={None if b0 is None else len(b0)} "
        f"tokens, steered={None if s0 is None else len(s0)} tokens, "
        f"equal={s0 == b0})")

    add("sentinel_fire_accounting",
        accounting is not None and accounting.get("ok"),
        f"raw fire-log coherence: {accounting}")

    c0 = tokens("sentinel_only")
    add("sentinel_noninvasive",
        bool(b0) and bool(c0) and c0 == b0,
        "a registered but non-modifying hook must not change greedy "
        f"output (equal={c0 == b0})")

    b0d = tokens("post_clear_baseline")
    add("hook_teardown_clean",
        bool(b0) and bool(b0d) and b0d == b0,
        "clear_hooks must leave no residue: post-clear baseline must "
        f"equal the pre-hook baseline (equal={b0d == b0})")

    ta = [f.get("tick", 0) for f in fires.get("mixed_unhooked", [])]
    tb = [f.get("tick", 0) for f in fires.get("mixed_hooked", [])]
    add("mixed_batch_cobatched",
        ticks_interleave(ta, tb),
        "the mixed batch's two requests must share forward passes "
        f"(fire-tick ranges must interleave; unhooked ticks n={len(ta)}, "
        f"hooked ticks n={len(tb)}) — a serialized pair proves nothing "
        "about cross-request isolation")

    dplug = disabled.get("plugin", {})
    add("disable_flag_unpatched",
        disabled and not dplug.get("generate_patched")
        and not dplug.get("engine_args_patched"),
        "the VLLM_LENS_DISABLE=1 process must show NO lens patches "
        f"(observed: {dplug})")
    return checks


# --- recorded-unavailable / recorded-failure markers ----------------------------


def _write_marker(out_dir: Path, name: str, payload: Mapping[str, Any]) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / name
    marker.write_text(json.dumps(dict(payload), indent=2, sort_keys=True,
                                 default=str))
    return marker


def record_unavailable(out_dir: Path, reason: str, **extra: Any) -> Path:
    """Fail loud, record, exit clean (the gate's pattern): the
    environment could not come up — a launch blocker, not a verdict on
    vllm-lens compatibility."""
    return _write_marker(out_dir, UNAVAILABLE_MARKER, {
        "status": "probe unavailable", "reason": reason,
        "recorded_at": time.time(), **extra})


def record_failure(out_dir: Path, reason: str, **extra: Any) -> Path:
    """A validation failure: the engine came up and the evidence says
    the pinned vllm-lens/vllm pair does NOT serve correctly. Recorded on
    disk AND exits nonzero — never downgraded to 'unavailable'."""
    return _write_marker(out_dir, FAILURE_MARKER, {
        "status": "probe FAILED (vllm-lens incompatible as pinned)",
        "reason": reason, "recorded_at": time.time(), **extra})


# --- the persistent sentinel hook (cloudpickled into the worker) ---------------

#: worker-global fire counter — a mutable cell so the cloudpickle
#: round-trip (register_hooks ships the fn by value) reconstructs ONE
#: shared counter for the single reconstructed fn instance.
_TICK = [0]


def _sentinel_fn(ctx, h):
    """Persistent-hook body: record every fire (layer, per-request
    scheduled seq_len, a worker-global tick, best-effort batch width) and
    save the prefill hidden rows for the modification comparison. Returns
    None — NEVER modifies (steering is applied by the separate,
    request-selected SteeringVector path, like the audit's legs)."""
    import torch as _t
    rec = {"layer": int(ctx.layer_idx), "seq_len": int(ctx.seq_len),
           "tick": _TICK[0], "num_reqs": -1}
    _TICK[0] += 1
    try:  # best-effort batch-width diagnostic; ticks are the real proof
        from vllm.forward_context import get_forward_context
        am = get_forward_context().attn_metadata
        if isinstance(am, dict):
            am = next(iter(am.values()), None)
        elif isinstance(am, list):
            am = am[0] if am else None
        qsl = getattr(am, "query_start_loc", None)
        if qsl is not None:
            rec["num_reqs"] = int(qsl.numel() - 1)
    except Exception:
        pass
    ctx.saved.setdefault("fires", []).append(rec)
    if rec["seq_len"] > 1:  # prefill chunk: keep rows for the delta check
        ctx.saved.setdefault("prefill_chunks", []).append(
            h.detach().to(_t.float32).cpu())
    return None


# --- worker-active: the lens-active engine --------------------------------------


def _patch_state() -> Dict[str, Any]:
    """Which vllm entry points currently carry the lens patches — read
    from the live classes, never assumed."""
    out: Dict[str, Any] = {}
    try:
        from vllm import LLM
        from vllm.engine.arg_utils import EngineArgs
        out["generate_patched"] = (
            getattr(LLM.generate, "__module__", "")
            == "vllm_lens._activations_plugin")
        out["chat_patched"] = (
            getattr(LLM.chat, "__module__", "")
            == "vllm_lens._activations_plugin")
        out["engine_args_patched"] = (
            getattr(EngineArgs.create_engine_config, "__module__", "")
            == "vllm_lens._activations_plugin")
        out["register_hooks_attr"] = hasattr(LLM, "register_hooks")
    except Exception as e:  # pragma: no cover - live-only
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def _live_engine_evidence(llm) -> Dict[str, Any]:
    """Read enforce_eager / prefix caching / worker extension from the
    LIVE engine config (authoritative — what actually runs), tolerating
    attribute-layout drift by recording None rather than crashing."""
    out: Dict[str, Any] = {"enforce_eager": None,
                           "prefix_caching_enabled": None,
                           "worker_extension_cls": None}
    cfg = None
    for path in ("llm_engine.vllm_config", "llm_engine.engine_config",
                 "llm_engine.model_config"):
        obj = llm
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            cfg = obj
            break
        except AttributeError:
            continue
    try:
        mc = getattr(cfg, "model_config", cfg)
        out["enforce_eager"] = bool(getattr(mc, "enforce_eager"))
    except Exception:
        pass
    try:
        cc = getattr(cfg, "cache_config", None)
        if cc is not None:
            out["prefix_caching_enabled"] = bool(
                getattr(cc, "enable_prefix_caching"))
    except Exception:
        pass
    try:
        pc = getattr(cfg, "parallel_config", None)
        if pc is not None:
            out["worker_extension_cls"] = getattr(pc, "worker_extension_cls",
                                                  None)
    except Exception:
        pass
    return out


def _reported_packages() -> Dict[str, str]:
    import importlib.metadata as md
    out = {}
    for pkg in ("vllm", "vllm-lens", "torch"):
        try:
            out[pkg] = md.version(pkg)
        except Exception as e:
            out[pkg] = f"UNIMPORTABLE({type(e).__name__})"
    return out


def _greedy_params(extra_args: Optional[Dict[str, Any]] = None):
    from vllm import SamplingParams
    return SamplingParams(temperature=0.0, max_tokens=PROBE_MAX_TOKENS,
                          ignore_eos=True,
                          extra_args=dict(extra_args) if extra_args else None)


def _run_out(out) -> Dict[str, Any]:
    comp = out.outputs[0]
    return {"request_id": str(out.request_id),
            "tokens": [int(t) for t in comp.token_ids],
            "text_head": comp.text[:160],
            "n_prompt": len(out.prompt_token_ids)}


def cmd_worker_active(args: argparse.Namespace) -> int:
    """Gather everything except disable_flag on one lens-active engine.
    Exit 0 = evidence complete; EXIT_ENGINE_INIT = engine never came up
    (unavailable); EXIT_REQUESTS = engine up, a request step raised
    (incompatible — the parent records a probe failure)."""
    import torch

    out_path = Path(args.out)
    result: Dict[str, Any] = {"stage": "preflight",
                              "packages": _reported_packages(),
                              "env": {
                                  "VLLM_BATCH_INVARIANT":
                                      os.environ.get("VLLM_BATCH_INVARIANT"),
                                  "VLLM_USE_V2_MODEL_RUNNER":
                                      os.environ.get(
                                          "VLLM_USE_V2_MODEL_RUNNER"),
                                  "VLLM_ENABLE_V1_MULTIPROCESSING":
                                      os.environ.get(
                                          "VLLM_ENABLE_V1_MULTIPROCESSING"),
                                  "VLLM_LENS_DISABLE":
                                      os.environ.get("VLLM_LENS_DISABLE"),
                              }}

    def dump(code: int, **extra: Any) -> int:
        result.update(extra)
        out_path.write_text(json.dumps(result, indent=2, sort_keys=True,
                                       default=str))
        return code

    plugin: Dict[str, Any] = {"pre_load": _patch_state()}
    try:
        # Load general plugins explicitly, as vllm serve does at startup;
        # register() is idempotent-by-effect (assignments) and the state
        # before/after is recorded either way.
        from vllm.plugins import load_general_plugins
        load_general_plugins()
        plugin["forced_load_called"] = True
    except Exception as e:
        plugin["forced_load_called"] = False
        plugin["forced_load_error"] = f"{type(e).__name__}: {e}"
    plugin.update(_patch_state())
    result["plugin"] = plugin

    result["stage"] = "engine_init"
    try:
        from vllm import LLM
        llm = LLM(model=args.model, revision=args.revision,
                  tokenizer_revision=args.revision, dtype="bfloat16",
                  seed=1234, max_model_len=2048,
                  gpu_memory_utilization=0.85,
                  enable_prefix_caching=False)
        # NOTE: enforce_eager deliberately NOT passed — the live config
        # must show the PLUGIN forced it (provenance evidence).
    except Exception as e:
        return dump(EXIT_ENGINE_INIT, error=f"{type(e).__name__}: {e}")
    Path(str(out_path) + ".engine_ready").write_text(str(time.time()))
    result["engine"] = _live_engine_evidence(llm)

    # Count hook/data RPCs at the collective_rpc seam (instance attribute
    # shadows the class method; the plugin calls self.collective_rpc).
    rpc_counts: Dict[str, int] = {}
    orig_rpc = llm.collective_rpc

    def counted_rpc(method, *a, **kw):
        name = method if isinstance(method, str) else getattr(
            method, "__name__", repr(method))
        rpc_counts[name] = rpc_counts.get(name, 0) + 1
        return orig_rpc(method, *a, **kw)

    llm.collective_rpc = counted_rpc
    result["rpc_counts"] = rpc_counts

    result["stage"] = "requests"
    runs: Dict[str, Any] = {}
    fires_by_run: Dict[str, Any] = {}
    result["runs"] = runs
    result["fires_by_run"] = fires_by_run
    try:
        import vllm_lens  # noqa: F401  (Hook/SteeringVector construction)
        from vllm_lens import Hook, SteeringVector

        # R1/R2: pure-engine baselines (nothing registered yet). R2 also
        # doubles as the prefix-cache repeat evidence.
        runs["baseline"] = _run_out(
            llm.generate([PROMPT_MAIN], _greedy_params())[0])
        runs["baseline_repeat"] = _run_out(
            llm.generate([PROMPT_MAIN], _greedy_params())[0])
        n_prompt = runs["baseline"]["n_prompt"]

        # R3: register the persistent sentinel (the ONE registration),
        # then a plain run — noninvasiveness + norm calibration rows.
        llm.register_hooks([Hook(fn=_sentinel_fn,
                                 layer_indices=[int(args.layer)])])
        requests_with_hook = 0
        runs["sentinel_only"] = _run_out(
            llm.generate([PROMPT_MAIN], _greedy_params())[0])
        requests_with_hook += 1

        def collect() -> Dict[str, Dict[str, Any]]:
            raw = llm.collect_hook_results()  # {internal_id: {pos: saved}}
            merged: Dict[str, Dict[str, Any]] = {}
            for rid, by_pos in (raw or {}).items():
                for _pos, saved in sorted(by_pos.items(),
                                          key=lambda kv: str(kv[0])):
                    merged[str(rid)] = saved
            return merged

        def saved_for(run_name: str,
                      snapshot: Mapping[str, Mapping[str, Any]]
                      ) -> Mapping[str, Any]:
            ext = runs[run_name]["request_id"]
            hits = [s for rid, s in snapshot.items()
                    if match_internal_id(rid, ext)]
            return hits[0] if hits else {}

        snap = collect()
        calib_saved = saved_for("sentinel_only", snap)
        chunks = calib_saved.get("prefill_chunks") or []
        base_rows = torch.cat(list(chunks), dim=0) if chunks else None
        gen = torch.Generator().manual_seed(STEER_SEED)
        direction = torch.randn(int(args.hidden), generator=gen)
        direction = direction / direction.norm().clamp_min(1e-9)
        if base_rows is not None and base_rows.numel():
            mean_norm = float(base_rows.norm(dim=-1).mean().item())
            alpha = ALPHA_REL * mean_norm
            calib_source = "sentinel_prefill_rows"
        else:
            mean_norm, alpha = None, ALPHA_FALLBACK
            calib_source = "fallback_constant"
        v = (direction * alpha).to(torch.float32)
        sv = SteeringVector(activations=v.unsqueeze(0),
                            layer_indices=[int(args.layer)], scale=1.0)
        sv0 = SteeringVector(activations=v.unsqueeze(0),
                             layer_indices=[int(args.layer)], scale=0.0)
        wire = json.dumps([sv.model_dump(mode="json")])
        wire0 = json.dumps([sv0.model_dump(mode="json")])
        result["calibration"] = {"source": calib_source,
                                 "mean_prompt_norm": mean_norm,
                                 "alpha": alpha, "alpha_rel": ALPHA_REL,
                                 "layer": int(args.layer),
                                 "hidden": int(args.hidden),
                                 "seed": STEER_SEED}
        result["sv_wire"] = wire

        # R4: the steered sentinel run — steering via the documented
        # extra_args wire form (what the HTTP path carries), plus a
        # capture attempt at the same layer (diagnostic only; the
        # sentinel rows are the primary modification evidence).
        cap_extra = {"output_residual_stream": json.dumps(
            [int(args.layer)])}
        runs["steered"] = _run_out(llm.generate(
            [PROMPT_MAIN],
            _greedy_params({"apply_steering_vectors": wire, **cap_extra})
        )[0])
        requests_with_hook += 1

        # R5: the SAME vector at scale 0.0 — the armed-hook no-op leg.
        runs["scale0"] = _run_out(llm.generate(
            [PROMPT_MAIN], _greedy_params({"apply_steering_vectors": wire0})
        )[0])
        requests_with_hook += 1

        # R6: steered solo on the second prompt (mixed-batch reference).
        runs["steered_alt_solo"] = _run_out(llm.generate(
            [PROMPT_ALT], _greedy_params({"apply_steering_vectors": wire})
        )[0])
        requests_with_hook += 1

        # R7: ONE generate call mixing an unhooked and a steered request.
        mixed = llm.generate(
            [PROMPT_MAIN, PROMPT_ALT],
            [_greedy_params(),
             _greedy_params({"apply_steering_vectors": wire})])
        runs["mixed_unhooked"] = _run_out(mixed[0])
        runs["mixed_hooked"] = _run_out(mixed[1])
        requests_with_hook += 2
        result["requests_observed_with_hook"] = requests_with_hook

        # capture diagnostic: did output.activations come back at all?
        steered_out_ok = False
        try:
            steered_raw = llm.generate(
                [PROMPT_MAIN],
                _greedy_params({"output_residual_stream":
                                json.dumps([int(args.layer)])}))[0]
            acts = getattr(steered_raw, "activations", None)
            steered_out_ok = bool(acts) and "residual_stream" in acts
            requests_with_hook += 1
            result["requests_observed_with_hook"] = requests_with_hook
            runs["capture_diag"] = _run_out(steered_raw)
        except Exception as e:
            result["capture_error"] = f"{type(e).__name__}: {e}"
        result["capture"] = {"functional": steered_out_ok}

        # Collect all sentinel results, then tear down and prove it.
        snap = collect()
        rows: Dict[str, Any] = {}
        for name in ("sentinel_only", "steered", "scale0",
                     "steered_alt_solo", "mixed_unhooked", "mixed_hooked",
                     "capture_diag"):
            if name not in runs:
                continue
            saved = saved_for(name, snap)
            fires_by_run[name] = [
                {k: int(v) for k, v in f.items()}
                for f in saved.get("fires", [])]
            ch = saved.get("prefill_chunks") or []
            if ch:
                rows[name] = torch.cat(list(ch), dim=0)
        llm.clear_hooks()
        runs["post_clear_baseline"] = _run_out(
            llm.generate([PROMPT_MAIN], _greedy_params())[0])

        # saved_sentinel evidence from the STEERED run (the leg that must
        # never silently no-op).
        sent_ev, accounting = build_saved_sentinel(
            fires_by_run.get("steered", []), n_prompt=n_prompt,
            n_gen=len(runs["steered"]["tokens"]), layer=int(args.layer))
        result["saved_sentinel_evidence"] = sent_ev
        result["saved_sentinel_accounting"] = accounting

        torch.save({"base_rows": rows.get("sentinel_only"),
                    "steered_rows": rows.get("steered"),
                    "v": v}, args.pt)
    except Exception as e:
        import traceback
        return dump(EXIT_REQUESTS, error=f"{type(e).__name__}: {e}",
                    traceback=traceback.format_exc())
    return dump(0, stage="completed")


# --- worker-disabled: the VLLM_LENS_DISABLE=1 stock engine ----------------------


def cmd_worker_disabled(args: argparse.Namespace) -> int:
    """The disable-flag evidence: a fresh process with VLLM_LENS_DISABLE=1
    must show no patches, save nothing, and produce baseline output even
    when the request explicitly asks for steering + capture. Stock knobs
    (enforce_eager, prefix caching, seed) are pinned to the active
    engine's values so token comparisons compare lens-presence only."""
    out_path = Path(args.out)
    result: Dict[str, Any] = {
        "stage": "preflight",
        "packages": _reported_packages(),
        "lens_disable_env": os.environ.get("VLLM_LENS_DISABLE"),
    }

    def dump(code: int, **extra: Any) -> int:
        result.update(extra)
        out_path.write_text(json.dumps(result, indent=2, sort_keys=True,
                                       default=str))
        return code

    try:
        from vllm.plugins import load_general_plugins
        load_general_plugins()  # register() must no-op under the flag
    except Exception as e:
        result["forced_load_error"] = f"{type(e).__name__}: {e}"
    result["plugin"] = _patch_state()

    result["stage"] = "engine_init"
    try:
        from vllm import LLM
        llm = LLM(model=args.model, revision=args.revision,
                  tokenizer_revision=args.revision, dtype="bfloat16",
                  seed=1234, max_model_len=2048,
                  gpu_memory_utilization=0.85,
                  enable_prefix_caching=False,
                  enforce_eager=True)  # explicit: match the lens engine
    except Exception as e:
        return dump(EXIT_ENGINE_INIT, error=f"{type(e).__name__}: {e}")
    Path(str(out_path) + ".engine_ready").write_text(str(time.time()))
    result["engine"] = _live_engine_evidence(llm)

    result["stage"] = "requests"
    runs: Dict[str, Any] = {}
    result["runs"] = runs
    try:
        runs["base"] = _run_out(
            llm.generate([PROMPT_MAIN], _greedy_params())[0])
        wire = Path(args.sv_wire_file).read_text()
        extra = {"apply_steering_vectors": wire,
                 "output_residual_stream": json.dumps([int(args.layer)])}
        out = llm.generate([PROMPT_MAIN], _greedy_params(extra))[0]
        runs["steer_attempt"] = _run_out(out)
        acts = getattr(out, "activations", None)
        hooks_res = getattr(out, "hook_results", None)
        result["artifacts"] = {
            "activations_present": acts is not None and bool(acts),
            "hook_results_present": hooks_res is not None and bool(hooks_res),
        }
        result["saved_count"] = (int(result["artifacts"]
                                     ["activations_present"])
                                 + int(result["artifacts"]
                                       ["hook_results_present"]))
    except Exception as e:
        import traceback
        return dump(EXIT_REQUESTS, error=f"{type(e).__name__}: {e}",
                    traceback=traceback.format_exc())
    return dump(0, stage="completed")


# --- the parent orchestrator ----------------------------------------------------


def _child_env(*, disable_lens: bool, batch_invariant: str) -> Dict[str, str]:
    env = dict(os.environ)
    # vllm filters general plugins by ENTRY-POINT NAME and vllm-lens's is
    # "activations": an inherited VLLM_PLUGINS=vllm_lens would silently
    # filter the plugin OUT (the failure class this probe hunts) — the
    # provenance assertion tests the default load-all path instead.
    env.pop("VLLM_PLUGINS", None)
    env.pop("VLLM_LENS_DISABLE", None)
    if disable_lens:
        env["VLLM_LENS_DISABLE"] = "1"
    # in-process engine, fixed submission order (the Phase-L pattern) —
    # and the disable-flag comparison needs BOTH processes on the same
    # runner/kernel config, so all of these are pinned in both children.
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    env["VLLM_USE_V2_MODEL_RUNNER"] = "0"  # lens hooks read V1 internals
    env["VLLM_BATCH_INVARIANT"] = batch_invariant
    env["TOKENIZERS_PARALLELISM"] = "false"
    return env


def _run_child(subcmd: str, argv: List[str], env: Dict[str, str],
               out_json: Path) -> Tuple[Optional[int], Dict[str, Any]]:
    """Run a worker subprocess; returns (returncode|None-on-timeout,
    parsed out-json-or-{}). Child stdout/stderr stream to the job log."""
    for stale in (out_json, Path(str(out_json) + ".engine_ready")):
        if stale.exists():
            stale.unlink()
    cmd = [sys.executable, str(Path(__file__).resolve()), subcmd, *argv]
    print(f"[probe] launching {subcmd}: {' '.join(cmd)}", flush=True)
    try:
        proc = subprocess.run(cmd, env=env, timeout=CHILD_TIMEOUT_S)
        rc: Optional[int] = proc.returncode
    except subprocess.TimeoutExpired:
        rc = None
    data: Dict[str, Any] = {}
    if out_json.is_file():
        try:
            data = json.loads(out_json.read_text())
        except Exception as e:
            data = {"stage": "unreadable", "error": str(e)}
    return rc, data


def probe_geometry(hf_cfg: Any) -> Tuple[int, int]:
    """``(num_hidden_layers, hidden_size)`` of the DECODER the lens hooks,
    read off the TEXT config. Composite VLM configs — ``Glm4vConfig`` for
    GLM-4.6V-Flash under transformers 5.14.1 — nest both under
    ``text_config`` and expose NEITHER at the top level (a top-level
    ``getattr`` raises AttributeError; measured 2026-09-02), while a plain
    causal-LM config returns itself from ``get_text_config()``. The same
    seam the onboarding smoke and ``sisteer.models``' accessors use, so
    the probe's layer index and hidden width name the served decoder on
    every family. A config predating ``get_text_config`` is read directly."""
    text = (hf_cfg.get_text_config() if hasattr(hf_cfg, "get_text_config")
            else hf_cfg)
    return int(text.num_hidden_layers), int(text.hidden_size)


def cmd_probe(args: argparse.Namespace) -> int:
    from sisteer.capeval import manifest as cm  # noqa: E402
    from sisteer.capeval.serving import lens_steer as ls  # noqa: E402

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # a fresh run must never leave a stale outcome readable (the gate's
    # transactional stance, minus certificates — the markers ARE the
    # product here).
    for stale in (UNAVAILABLE_MARKER, FAILURE_MARKER, PROBE_RECORD):
        p = out_dir / stale
        if p.exists():
            p.unlink()

    spec = cm.load_petri_spec(Path(args.manifest))  # config bug -> nonzero
    script = Path(__file__).resolve()
    provenance: Dict[str, Any] = {
        "model": spec.target_model,
        "revision": spec.target_revision,
        "expected_packages": dict(spec.serving_packages),
        "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "node": os.environ.get("SLURMD_NODENAME") or os.uname().nodename,
        "started_at": time.time(),
    }
    try:
        provenance["checkout_sha"] = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30).stdout.strip() or None
    except Exception:
        provenance["checkout_sha"] = None

    evidence_out: Dict[str, Any] = {"provenance": provenance}

    def flush_evidence() -> None:
        (out_dir / EVIDENCE_FILE).write_text(json.dumps(
            evidence_out, indent=2, sort_keys=True, default=str))

    # ---- preflight: importability + GPU (unavailable, never a crash) ----
    missing = [m for m in ("vllm", "vllm_lens")
               if importlib.util.find_spec(m) is None]
    if missing:
        flush_evidence()
        marker = record_unavailable(
            out_dir, f"not importable in this venv: {missing} — the probe "
            "runs only in the pinned GPU serving venv "
            "(configs/capeval/vllm-requirements.txt)", stage="imports")
        print(f"PROBE UNAVAILABLE — recorded at {marker}", file=sys.stderr)
        return 0
    import torch
    if not torch.cuda.is_available():
        flush_evidence()
        marker = record_unavailable(out_dir, "torch.cuda unavailable — the "
                                    "probe needs a GPU node", stage="gpu")
        print(f"PROBE UNAVAILABLE — recorded at {marker}", file=sys.stderr)
        return 0

    # ---- probe geometry from the pinned model config (cache-resident) ----
    from transformers import AutoConfig
    hf_cfg = AutoConfig.from_pretrained(spec.target_model,
                                        revision=spec.target_revision)
    n_layers, hidden = probe_geometry(hf_cfg)
    layer = int(args.layer) if args.layer is not None else n_layers // 2
    provenance.update({"num_hidden_layers": n_layers, "hidden_size": hidden,
                       "probe_layer": layer})

    # ---- child 1: the lens-active engine (retry once without the
    # batch-invariant exact-replay mode if THAT blocks engine init) ----
    active_json = out_dir / "worker_active.json"
    active_pt = out_dir / "worker_active_rows.pt"
    child_args = ["--model", spec.target_model,
                  "--revision", spec.target_revision,
                  "--layer", str(layer), "--hidden", str(hidden),
                  "--out", str(active_json), "--pt", str(active_pt)]
    batch_invariant = "1"
    rc, active = _run_child("worker-active", child_args,
                            _child_env(disable_lens=False,
                                       batch_invariant=batch_invariant),
                            active_json)
    engine_came_up = Path(str(active_json) + ".engine_ready").exists()
    if (rc == EXIT_ENGINE_INIT or (rc is None and not engine_came_up)):
        evidence_out["active_batch_invariant_attempt"] = active
        batch_invariant = "0"
        rc, active = _run_child("worker-active", child_args,
                                _child_env(disable_lens=False,
                                           batch_invariant=batch_invariant),
                                active_json)
        engine_came_up = Path(str(active_json) + ".engine_ready").exists()
        evidence_out["batch_invariant_fallback"] = {
            "note": "engine init failed under VLLM_BATCH_INVARIANT=1 (the "
                    "audit's exact-replay mode) and was retried without "
                    "it — if the retry passed, the audit must use the "
                    "proxy-serialization replay strategy instead",
            "fallback_used": True}
    evidence_out["active"] = active
    evidence_out["active_exit"] = rc
    evidence_out["batch_invariant_mode"] = batch_invariant

    if rc == EXIT_ENGINE_INIT or (rc is None and not engine_came_up):
        flush_evidence()
        marker = record_unavailable(
            out_dir, "the lens-active vLLM engine cannot init "
            f"(child exit {rc}, stage {active.get('stage')}): "
            f"{active.get('error')}", stage="active_engine_init",
            lens_init_only=None)
        print(f"PROBE UNAVAILABLE — recorded at {marker}", file=sys.stderr)
        return 0
    if rc != 0:
        flush_evidence()
        marker = record_failure(
            out_dir, "the lens-active engine came up but evidence "
            f"gathering raised (child exit {rc}, stage "
            f"{active.get('stage')}): {active.get('error')} — an "
            "engine that inits but cannot serve hooked requests is "
            "INCOMPATIBLE, not unavailable",
            stage=str(active.get("stage")))
        print(f"PROBE FAILED — recorded at {marker}", file=sys.stderr)
        return 1

    # ---- child 2: the VLLM_LENS_DISABLE=1 stock engine ----
    wire = active.get("sv_wire")
    wire_file = out_dir / "sv_wire.json"
    wire_file.write_text(wire if isinstance(wire, str) else "[]")
    disabled_json = out_dir / "worker_disabled.json"
    rc_d, disabled = _run_child(
        "worker-disabled",
        ["--model", spec.target_model, "--revision", spec.target_revision,
         "--layer", str(layer), "--sv-wire-file", str(wire_file),
         "--out", str(disabled_json)],
        _child_env(disable_lens=True, batch_invariant=batch_invariant),
        disabled_json)
    evidence_out["disabled"] = disabled
    evidence_out["disabled_exit"] = rc_d
    disabled_up = Path(str(disabled_json) + ".engine_ready").exists()
    if rc_d == EXIT_ENGINE_INIT or (rc_d is None and not disabled_up):
        # stock vLLM failing AFTER the lens engine succeeded on the same
        # node is an environment flake, not a lens verdict.
        flush_evidence()
        marker = record_unavailable(
            out_dir, "the STOCK (VLLM_LENS_DISABLE=1) engine cannot init "
            f"even though the lens-active engine ran (child exit {rc_d}): "
            f"{disabled.get('error')} — disable-flag evidence is "
            "ungatherable this run", stage="disabled_engine_init")
        print(f"PROBE UNAVAILABLE — recorded at {marker}", file=sys.stderr)
        return 0

    # ---- assemble + judge ----
    retmod_delta = None
    try:
        rows = torch.load(active_pt, weights_only=True) \
            if active_pt.is_file() else {}
        if (rows.get("base_rows") is not None
                and rows.get("steered_rows") is not None):
            retmod_delta = max_relative_residual(
                rows["base_rows"], rows["steered_rows"], rows["v"])
    except Exception as e:
        evidence_out["retmod_error"] = f"{type(e).__name__}: {e}"

    phase0 = assemble_phase0_evidence(
        active, disabled, expected_packages=spec.serving_packages,
        retmod_delta=retmod_delta, retmod_atol=RETMOD_ATOL)
    checks = evaluate_probe_checks(
        active, disabled, active.get("saved_sentinel_accounting"))
    evidence_out["phase0_evidence"] = phase0
    evidence_out["probe_checks"] = checks
    evidence_out["returned_modification_delta"] = retmod_delta
    flush_evidence()

    failures: List[str] = []
    try:
        report = ls.run_phase0(phase0)
        evidence_out["phase0_passed"] = list(report.passed)
    except ls.Phase0Failure as e:
        failures.append(str(e))
    failed_checks = [c for c in checks if not c["ok"]]
    failures.extend(f"probe {c['name']}: {c['detail']}"
                    for c in failed_checks)
    flush_evidence()

    print("=" * 72)
    print("vllm-lens x vllm compatibility probe — "
          f"{provenance['expected_packages']}")
    print(f"model {spec.target_model} @ {spec.target_revision[:12]} | "
          f"layer {layer} | batch_invariant={batch_invariant}")
    for c in checks:
        print(f"  [{'PASS' if c['ok'] else 'FAIL'}] probe:{c['name']}")
    if failures:
        reason = "\n\n".join(failures)
        marker = record_failure(out_dir, reason,
                                failed_probe_checks=[c["name"] for c in
                                                     failed_checks])
        print(f"PROBE FAILED — vllm-lens 1.2.1 x vllm 0.25.1 did NOT hold; "
              f"recorded at {marker}\n{reason}", file=sys.stderr)
        return 1
    marker = _write_marker(out_dir, PROBE_RECORD, {
        "status": "completed",
        "phase0_passed": evidence_out.get("phase0_passed"),
        "probe_checks_passed": [c["name"] for c in checks],
        "returned_modification_delta": retmod_delta,
        "stated_atol": RETMOD_ATOL,
        "batch_invariant_mode": batch_invariant,
        "provenance": provenance,
        "recorded_at": time.time(),
    })
    print(f"PROBE PASS — all 8 Phase-0 assertions + "
          f"{len(checks)} probe checks held; record at {marker}")
    return 0


# --- CLI ------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("probe", help="the parent orchestrator (GPU node)")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--layer", type=int, default=None,
                   help="probe layer (default: num_hidden_layers // 2)")
    p.set_defaults(fn=cmd_probe)

    a = sub.add_parser("worker-active", help="internal: lens-active engine")
    a.add_argument("--model", required=True)
    a.add_argument("--revision", required=True)
    a.add_argument("--layer", type=int, required=True)
    a.add_argument("--hidden", type=int, required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--pt", required=True)
    a.set_defaults(fn=cmd_worker_active)

    d = sub.add_parser("worker-disabled",
                       help="internal: VLLM_LENS_DISABLE=1 engine")
    d.add_argument("--model", required=True)
    d.add_argument("--revision", required=True)
    d.add_argument("--layer", type=int, required=True)
    d.add_argument("--sv-wire-file", required=True)
    d.add_argument("--out", required=True)
    d.set_defaults(fn=cmd_worker_disabled)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
