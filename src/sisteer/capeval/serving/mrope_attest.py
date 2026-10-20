"""The M-RoPE DISPATCH ATTESTATION, shared by every standup (lifted from
``scripts/sdfbase/mt_serving_up.py`` — the full22-newmodels GLM lane, 2026-09-08 — so the tinycap
standup ``scripts/capctrl_serving_up.py`` asks the same question).

vLLM 0.25.1's fused ``triton_mrope`` kernel has no ``is_neox_style`` and rotates every M-RoPE model
neox-style; the GLM-4V line is the one GPT-J-pair (``is_neox_style=False``) M-RoPE family, so an EAGER
engine (the served path: vllm-lens forces ``--enforce-eager``) mis-rotates every attention layer
unless ``capeval.serving.vllm_patches`` routed the 2-D-position case onto ``forward_native`` in the
worker (lens_worker applies it at worker import). The worker extension answers
``sisteer_mrope_report`` (the same RPC ``vllm_patches.attest_engine`` asks an in-process engine); a
standup asks it over the authenticated collective_rpc route it already uses for the install +
scheduler reports, records every worker's report (``mrope_attestation`` in endpoint.json /
cells_served.json), and REFUSES to serve when the reports do not prove the rotary is right — certify
the branch that RAN, never the hook that was installed. At STOP the report is asked AGAIN: for a
non-neox M-RoPE model ``routed_native_calls`` must have MOVED (the proof the patched route was
exercised by the run's traffic); ``stop_report`` records that beside the two reports.
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List, Mapping, Optional, Sequence

MROPE_RPC_METHOD = "sisteer_mrope_report"
#: model keys whose rotary is a non-neox (GPT-J-pair) M-RoPE: for these the report must show the fused
#: kernel reachable AND the patch applied; a report of any other shape is a refusal, not a pass
NON_NEOX_MROPE_KEYS = ("glm46vflash",)
#: HF config ``model_type`` values of the non-neox M-RoPE family (the launcher's family-keyed belt)
NON_NEOX_MROPE_FAMILIES = ("glm4v",)


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def mrope_attestation_problems(reports, model_key) -> list:
    """Every reason the workers' rotary reports do NOT prove the served rotary is correct.

    Mirrors ``vllm_patches.attest_engine`` for every model (``needs_patch`` without ``patch_applied`` is
    the bug itself: a non-neox M-RoPE model on the neox-only kernel), and for a key in
    :data:`NON_NEOX_MROPE_KEYS` additionally demands the report have the shape the served path is
    known to produce — rotary class ``MRotaryEmbedding``, ``is_neox_style`` False, the fused kernel
    REACHABLE (``dispatch == forward_cuda``: the eager engine the lens requires; a compiled dispatch
    would mean the patch is inert and unexercised, an unexpected engine, not a proof) and
    ``patch_applied`` True. An errored or missing report is a problem, never a pass."""
    problems = []
    if not reports:
        return [f"no {MROPE_RPC_METHOD} report from any worker"]
    for i, rep in enumerate(reports):
        if not isinstance(rep, dict):
            problems.append(f"worker {i}: report is {type(rep).__name__}, not a mapping")
            continue
        if rep.get("error"):
            problems.append(f"worker {i}: rotary report errored: {rep['error']}")
        if rep.get("needs_patch") and not rep.get("patch_applied"):
            problems.append(f"worker {i}: non-neox M-RoPE model is running on vLLM's neox-only "
                            f"triton_mrope kernel (patch not applied): {rep}")
        if model_key in NON_NEOX_MROPE_KEYS:
            if rep.get("rotary_cls") != "MRotaryEmbedding":
                problems.append(f"worker {i}: {model_key} reports rotary_cls={rep.get('rotary_cls')!r}, "
                                "not MRotaryEmbedding")
            if rep.get("is_neox_style", True):
                problems.append(f"worker {i}: {model_key} reports is_neox_style=True — not the GPT-J "
                                "pairing this key is known to use")
            if not rep.get("kernel_reachable"):
                problems.append(f"worker {i}: {model_key} dispatch={rep.get('dispatch')!r} — the fused "
                                "kernel is not reachable, so the patch is inert and unexercised (the "
                                "served path forces eager; an unexpected dispatch is not attested)")
            if not rep.get("patch_applied"):
                problems.append(f"worker {i}: {model_key} reports patch_applied=False")
    return problems


def mrope_attest(launch, model_key, *, timeout_s: float = 120.0) -> dict:
    """Ask every worker for its rotary report over the launch's authenticated collective_rpc route
    and return the attestation block ``{ok, problems, reports, rpc_method, model_key, asked_at}``.
    A transport failure is recorded as a problem (never raised past here): the caller refuses."""
    from sisteer.capeval.serving import vllm_server as vs
    try:
        payload = vs._post_collective_rpc(launch, MROPE_RPC_METHOD, [], timeout_s=timeout_s,
                                          what="M-RoPE dispatch attestation")
        reports = payload.get("results") if isinstance(payload, dict) else None
        reports = list(reports) if isinstance(reports, list) else []
        problems = mrope_attestation_problems(reports, model_key)
    except Exception as e:  # noqa: BLE001 — a failed RPC is a refusal with its reason, not a traceback
        reports, problems = [], [f"{MROPE_RPC_METHOD} RPC failed: {type(e).__name__}: {e}"]
    return {"ok": not problems, "problems": problems, "reports": reports,
            "rpc_method": MROPE_RPC_METHOD, "model_key": model_key, "asked_at": _utc_now()}


def routed_native_calls(reports: Sequence[Any]) -> List[Optional[int]]:
    """Per worker, the patched route's call counter (None when the report lacks it)."""
    return [r.get("routed_native_calls") if isinstance(r, Mapping) else None for r in reports]


def stop_report(model_key, at_ready: Mapping[str, Any], at_stop: Mapping[str, Any]) -> Dict[str, Any]:
    """The STOP-time record ``<out>/mrope_attestation.json``: both reports plus ``routed_native_moved``
    — for a key in :data:`NON_NEOX_MROPE_KEYS` True iff EVERY worker's ``routed_native_calls`` at STOP
    is greater than at readiness (0 there by construction); for any other key ``None`` (no patched
    route to exercise: the counter is not the proof for a neox / rotary-less model). Recorded, never a
    refusal — teardown is never refused; the results doc reads it per GLM cell."""
    before = routed_native_calls(at_ready.get("reports") or [])
    after = routed_native_calls(at_stop.get("reports") or [])
    moved: Optional[bool]
    if model_key in NON_NEOX_MROPE_KEYS:
        moved = bool(after) and len(after) == len(before) and all(
            isinstance(b, int) and isinstance(a, int) and a > b for b, a in zip(before, after))
    else:
        moved = None
    return {"model_key": model_key, "at_ready": dict(at_ready), "at_stop": dict(at_stop),
            "routed_native_calls_at_ready": before, "routed_native_calls_at_stop": after,
            "routed_native_moved": moved,
            "note": ("at_stop is asked AFTER the run's traffic: for a non-neox M-RoPE model its "
                     "routed_native_calls > at_ready's (0 by construction) is the proof the patched route "
                     "was exercised; routed_native_moved is null for a model with no patched route")}


def endpoint_mrope_problems(endpoint: Mapping[str, Any], model_key) -> List[str]:
    """The LAUNCHER's reading of a standup's ``endpoint.json``: the attestation must be present, ``ok``,
    made for THIS model key, and its reports must (still) satisfy :func:`mrope_attestation_problems`
    for the key — anything else is the launcher's ``MROPE_UNATTESTED`` refusal, named."""
    att = endpoint.get("mrope_attestation")
    if not isinstance(att, Mapping):
        why = endpoint.get("mrope_attestation_error")
        return ["endpoint.json carries no mrope_attestation" + (f" ({why})" if why else "")
                + " — the standup did not attest the rotary dispatch"]
    problems: List[str] = []
    if att.get("ok") is not True:
        problems.append(f"the standup's M-RoPE attestation is not ok: {att.get('problems')}")
    if att.get("rpc_method") != MROPE_RPC_METHOD:
        problems.append(f"attestation rpc_method {att.get('rpc_method')!r} != {MROPE_RPC_METHOD!r}")
    if att.get("model_key") != model_key:
        problems.append(f"the standup attested model key {att.get('model_key')!r}, this launch serves {model_key!r}")
    problems += mrope_attestation_problems(list(att.get("reports") or []), model_key)
    return problems
