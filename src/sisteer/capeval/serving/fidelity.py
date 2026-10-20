"""Engine-internal serving-correctness gate for the steered vLLM path:
the assertion logic, PURE and CPU-testable. The GPU evidence gathering
(actually serving the probe requests) lives in
``scripts/petri_serving_gate.py`` — the same gathering-vs-assertion
split as lens_steer's Phase-0 harness, and the same discipline: every
check refuses loudly, vacuous evidence (zero probes, an empty token
list, a pair that generated nothing on either side) is a FAILURE and
never a pass, and malformed evidence is contained to its own check's
failure rather than aborting the rest (:func:`evaluate_fidelity`).

WHY EVERY PASS CRITERION HERE IS EXACT TOKEN EQUALITY — AND WHY THE
TOLERANCES A READER MIGHT EXPECT ARE ABSENT. An earlier design of this
lane compared the served engine against an HF fp32 reference, and
cross-ENGINE generated text cannot be gated on exact equality: a
documented finding in this programme is that roughly 30% of long greedy
generations diverge textually at near-tie argmax between two HF runs
that differ only in batch composition, and two different engines
(different kernels, different reduction orders) are strictly worse.
That comparison needed a whole statistical apparatus — a measured
self-agreement noise floor, stated margins, a near-tie logit-gap
forensic. That comparison is NO LONGER MADE: cross-engine equivalence
to HF is explicitly not a goal (the HF-served results it protected are
retired; the lane serves vLLM in its own right), so this gate carries
no reference engine, and BOTH sides of every comparison below come from
the SAME engine under greedy decode. Within one engine, an inequality
is a bug in exactly the machinery this gate exists to test, so the
discipline of ``scripts/petri_serving_gate.py`` applies unchanged (H5):
pass = exact token-id equality, and every number in the report
(first-divergence indices, differing counts) is a DIAGNOSTIC, never a
pass criterion. This module is not an exception to that rule any more;
if the statistical apparatus is ever needed again, it lives in the git
history of this file's first design, not here.

THE FIVE CHECKS (:data:`FIDELITY_CHECKS`), all within-engine:

1. ``engine_determinism`` — the same request set, submitted twice with
   the same batch composition and configuration, must be bit-identical.
   A failure here INVALIDATES the whole gate: every other check is a
   single-run equality comparison, which evidences nothing on a
   nondeterministic engine. It is therefore reported as invalidating
   (``FidelityReport.invalidated``; other checks' passes are recorded
   as "indeterminate"), never as one failure among several.
2. ``scale0_noop`` — a steering object SELECTED at scale 0 (the hook
   armed, real tensors flowing, multiplied by zero) must produce output
   identical to the same requests with NO object selected. Genuinely
   distinct from check 1 (a perfectly deterministic engine can still
   fail the no-op, and a correct no-op can still be nondeterministic)
   and from check 3 — do not collapse them.
3. ``steering_bites`` — per arm: the arm selected at its serving scale
   must DIFFER from the same requests with the arm at scale 0 on at
   least one probe. A steering path that silently no-ops is the single
   most likely failure mode of the whole system (vllm-lens's own
   documented failure mode is silently capturing nothing), and with no
   reference engine this is the check that catches it. An arm that
   changes nothing is a FAILURE, named per arm — never a pass.
4. ``batch_invariance`` — the same request must produce the same output
   under two batches OF THE SAME SIZE that differ only in COMPOSITION:
   one whose other members are unsteered filler, one that mixes
   selections. BOTH directions are checked: steered requests must not
   change when their neighbours are unsteered, and unsteered requests
   must not change when their neighbours are steered. This is the
   cross-request contamination check that continuous batching makes
   non-obvious. The reference must be the same size — see THE
   EQUAL-SIZE RULE with the evidence schema below.
5. ``selection_honored`` — in one mixed batch where several requests
   each select a DIFFERENT object, each request's output must equal
   what that same (selection, prompt) pair produces in a SAME-SIZE
   reference batch. The whole per-request-selection architecture
   (lens_steer H3) rests on this. Two refusals guard a vacuous pass,
   ported from ``run_multi_lora_selection``'s discipline: fewer than
   two distinct selections prove nothing about routing, and if every
   probe prompt served under multiple selections yields IDENTICAL
   reference outputs across those selections, the observations cannot
   distinguish routing (a server sending every request to one object
   would pass) — both refuse to certify. A real reference-vs-mixed
   mismatch outranks either refusal and is reported as the failure.
   LORA-NATIVE SERVERS (gemma canary job 489308): the standup's
   rows are N CONCURRENT requests admitted as they arrive, so two
   submissions of "the same N requests" are not the same step-by-step
   computation at BI=0 and the output comparison above is not the
   verdict there. The verdict is STRUCTURAL —
   :func:`check_multi_lora_routing_attested` over the engine's own
   per-request LoRA bookkeeping (``lora_int_id`` / ``lora_name`` per
   request id, plus the count of steps that co-hosted >= 2 distinct
   ids) — and the output comparison is RECORDED
   (:func:`multi_lora_output_diagnostics`) with the base reference
   submitted twice as the fixed-size repeat floor. The lens/legacy
   policy keeps the output verdict with its stated limit.

DIAGNOSTICS, NEVER CRITERIA. Where streams differ, the report carries
first-divergence indices and differing counts. A wrong-row steer (the
classic flattened-batch indexing bug) corrupts the very first predicted
token, so a ``scale0_noop`` / ``batch_invariance`` failure whose first
divergences sit at position 0 points at row indexing — useful for
debugging, consulted by no pass criterion.

PRIVACY — NUMBERS AND VERDICTS ONLY. The report carries counts,
indices, booleans and verdict prose composed by THIS module — never
prompt text and never generated text (a property of the surrounding
programme, not a preference). Enforced in code, not just convention:
token ids must be integers (anything else — text included — refuses by
TYPE NAME, its value never echoed); the messages of foreign exceptions
raised by malformed evidence are WITHHELD (type name only), because an
exception message can echo the offending value; arm ids and evidence
keys are echoed only when they look like identifiers, otherwise shown
as typed placeholders; and ``FidelityReport.as_dict`` walks its own
output and refuses any non-scalar leaf. This is deliberately stricter
than lens_steer's ``run_phase0``, whose malformed-evidence
normalization interpolates the exception text.

EVIDENCE SCHEMA — what ``scripts/petri_serving_gate.py`` must gather.
Evidence keys are exactly the check names (:data:`EVIDENCE_KEYS`).
Every token id is a RAW engine token id (one engine everywhere; no
re-encoding — there is no second tokenizer in play, and re-encoding
could mask a real special-token difference). All probe runs are greedy
with one shared max-new-tokens budget. Paired lists are aligned by
index (element i of both lists came from the same request); "same batch
composition" below is the gatherer's obligation and cannot be checked
here.

    evidence = {
      "engine_determinism": {
        "run1_ids": [[int], ...],   # one probe request set, served once
        "run2_ids": [[int], ...],   # the IDENTICAL set re-submitted:
      },                            #   same prompts, same selections,
                                    #   same batch composition, same
                                    #   engine configuration
      "scale0_noop": {
        "scale0_ids":    [[int], ...],  # an object SELECTED at scale 0
                                        #   (hook armed, tensors
                                        #   flowing, multiplied by zero)
        "unsteered_ids": [[int], ...],  # same requests, NO selection,
      },                                #   same batch composition
      "steering_bites": {
        "arms": {                       # >= 1 arm; probe every arm the
          "<arm_id>": {                 #   audit will actually serve
            "steered_ids": [[int], ...],  # arm at its serving scale
            "scale0_ids":  [[int], ...],  # same requests, same arm
          },                              #   selected at scale 0
          ...
        },
      },
      "batch_invariance": {
        # THE REFERENCE MUST BE THE SAME BATCH SIZE. See THE EQUAL-SIZE
        # RULE below — a batch-of-1 reference does NOT work on this
        # engine and hardware, and comparing against one produces false
        # failures with no wiring involved.
        # reference: each request in a batch of size N whose OTHER
        # members are all unsteered filler. batched: the same requests
        # in a batch of the SAME size N that mixes selections. Row i of
        # each list is the same request.
        "steered_reference_ids":   [[int], ...],
        "steered_batched_ids":     [[int], ...],
        "unsteered_reference_ids": [[int], ...],
        "unsteered_batched_ids":   [[int], ...],
      },
      "selection_honored": {
        "requests": [                 # every request of ONE mixed batch
          {"arm": "<arm_id>" | None,  # the selection it carried (None =
                                      #   no object selected)
           "prompt_index": int,       # equal indices = the same probe
                                      #   prompt text (gatherer's
                                      #   obligation; consumed by the
                                      #   distinguishability refusal)
           "reference_ids": [int, ...],  # SAME-SIZE batch, filler
                                         #   neighbours, that selection
           "batched_ids":   [int, ...]}, #   vs inside the mixed batch
          ...
        ],
      },
    }

THE EQUAL-SIZE RULE, and why it is not a detail. Every comparison here
must hold BATCH SIZE FIXED and vary only COMPOSITION. The property
being tested is that a request's output does not depend on WHO ELSE
shares its batch; batch SIZE is a different variable, and on this stack
it is not one the engine controls.

Measured on vLLM 0.25.1 / H200 (SM90), job 173152 plus a bare-engine
reproduction with no steering code and the lens plugin not installed:
``VLLM_BATCH_INVARIANT=1`` does NOT deliver batch-size-1-versus-N
bit-invariance. Sizes 6, 7, 8, 11, 15 and 16 agreed bit-exactly with
each other; size 1 differed, flipping a single near-tie token
mid-generation. The flag's SM90 path installs cuBLAS workspace configs
rather than shape-pinned matmul kernels, so some residual kernel still
varies with M.

An earlier version of this schema asked for a SOLO reference, and it
cost a full debugging cycle: two requests were reported as
cross-request contamination when their batched outputs were in fact
bit-perfect and it was the batch-of-1 references that had moved. A
false positive here is expensive in a specific way — it accuses the
serving path of the exact failure this gate exists to detect, so it is
believed.

Requirements enforced here, each a loud refusal: all five keys present
(a missing probe is a failure, not a pass); every id list non-empty;
paired lists of equal length; no pair that generated nothing on both
sides; token ids integers; at least one arm probed; and
``selection_honored`` carrying at least two distinct selections plus at
least one shared prompt whose reference outputs separate two
selections.

:func:`evaluate_fidelity` returns a :class:`FidelityReport` naming
EVERY failure (never short-circuiting on the first), in the same spirit
as ``run_phase0``.
"""

from __future__ import annotations

import math
import re
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "FidelityError",
    "FidelityReport",
    "FIDELITY_CHECKS",
    "EVIDENCE_KEYS",
    "first_divergence_index",
    "evaluate_fidelity",
    "evaluate_stacked_identity",
    "check_multi_lora_selection",
    "MULTI_LORA_POLICY_LEGACY",
    "MULTI_LORA_POLICY_NATIVE",
]


class FidelityError(RuntimeError):
    """A fidelity-gate refusal or failure. Message discipline: composed
    from counts, indices and TYPE NAMES only — never from evidence
    values (the report these messages land in carries no prompt text
    and no generated text)."""


FIDELITY_CHECKS = (
    "engine_determinism",   # first: its failure invalidates the rest
    "scale0_noop",
    "steering_bites",
    "batch_invariance",
    "selection_honored",
)
#: evidence keys are the check names, 1:1 (module docstring, SCHEMA).
EVIDENCE_KEYS = FIDELITY_CHECKS

#: what an arm id / evidence key must look like to be echoed verbatim
#: in a report; anything else appears as a typed placeholder (privacy:
#: a buggy gatherer must not be able to smuggle text through a name).
_NAME_RE = re.compile(r"[A-Za-z0-9_.\-]{1,80}")


# --- the comparison primitive (diagnostic, never a criterion by itself) --------


def first_divergence_index(a: Sequence[Any], b: Sequence[Any]
                           ) -> Optional[int]:
    """Index of the first differing element, or None when identical; a
    pure length difference diverges at ``min(len(a), len(b))``. Elements
    are only compared, never stored or echoed."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else n


# --- evidence intake (fail-loud, type-enforced — the privacy boundary) ---------


def _safe_name(value: Any, ordinal: int) -> str:
    """Echo a gatherer-provided name only when it looks like an
    identifier; anything else (free text included) is shown as a typed
    placeholder so no evidence string can ride a name into the report."""
    if isinstance(value, str) and _NAME_RE.fullmatch(value):
        return value
    return f"<non-identifier #{ordinal} of type {type(value).__name__}>"


def _id_row(check: str, ev: Mapping[str, Any], label: str) -> List[int]:
    """One token-id list. Refuses a missing label, a non-list value and
    any non-integer element — by TYPE NAME only, never echoing the
    value (text can never enter the instrument)."""
    if label not in ev:
        raise FidelityError(f"{check}: evidence lacks {label!r} — no "
                            "evidence is a failure, not a pass")
    raw = ev[label]
    if not isinstance(raw, (list, tuple)):
        raise FidelityError(
            f"{check}: {label} must be a token-id list, got type "
            f"{type(raw).__name__} (value withheld: evidence values "
            "never enter a report)")
    ids = list(raw)
    for t in ids:
        if isinstance(t, bool) or not isinstance(t, int):
            raise FidelityError(
                f"{check}: {label} carries a non-integer element of "
                f"type {type(t).__name__} — token ids are integers; "
                "anything else (text included) refuses rather than "
                "entering the instrument")
    return ids


def _id_lists(check: str, ev: Mapping[str, Any], label: str
              ) -> List[List[int]]:
    """A per-request list of token-id lists. Same refusals as
    :func:`_id_row`, plus: an EMPTY outer list is vacuous evidence and
    refuses (zero probes measure nothing)."""
    if label not in ev:
        raise FidelityError(f"{check}: evidence lacks {label!r} — no "
                            "evidence is a failure, not a pass")
    raw = ev[label]
    if not isinstance(raw, (list, tuple)):
        raise FidelityError(
            f"{check}: {label} must be a list of per-request token-id "
            f"lists, got type {type(raw).__name__} (value withheld: "
            "evidence values never enter a report)")
    out: List[List[int]] = []
    for i, row in enumerate(raw):
        if not isinstance(row, (list, tuple)):
            raise FidelityError(
                f"{check}: {label}[{i}] must be a token-id list, got "
                f"type {type(row).__name__} (value withheld)")
        ids = list(row)
        for t in ids:
            if isinstance(t, bool) or not isinstance(t, int):
                raise FidelityError(
                    f"{check}: {label}[{i}] carries a non-integer "
                    f"element of type {type(t).__name__} — token ids "
                    "are integers; anything else (text included) "
                    "refuses rather than entering the instrument")
        out.append(ids)
    if not out:
        raise FidelityError(
            f"{check}: {label} is empty — zero probes measure nothing; "
            "vacuous evidence is a failure, never a pass")
    return out


def _paired(check: str, a: List[List[int]], b: List[List[int]]
            ) -> Tuple[int, List[Tuple[int, int]]]:
    """Pair two aligned run outputs: returns (n_pairs, [(pair_index,
    first_divergence_index), ...] over the pairs that differ). Refuses
    unequal-length pair lists and any pair that generated no tokens on
    EITHER side ([] == [] proves nothing about serving)."""
    if len(a) != len(b):
        raise FidelityError(f"{check}: paired lists of unequal length: "
                            f"{len(a)} vs {len(b)}")
    divs: List[Tuple[int, int]] = []
    for i, (x, y) in enumerate(zip(a, b)):
        if not x and not y:
            raise FidelityError(
                f"{check}: pair {i} generated no tokens on either side "
                "— [] == [] is vacuous evidence, a failure, never a "
                "pass")
        d = first_divergence_index(x, y)
        if d is not None:
            divs.append((i, d))
    return len(a), divs


# --- the five checks (each: evidence mapping in, diagnostics out, raise) -------


def _check_engine_determinism(ev: Mapping[str, Any],
                              diag: Dict[str, Any]) -> None:
    a = _id_lists("engine_determinism", ev, "run1_ids")
    b = _id_lists("engine_determinism", ev, "run2_ids")
    n, divs = _paired("engine_determinism", a, b)
    diag.update(n=n, n_divergent=len(divs),
                first_divergences=[[i, d] for i, d in divs])
    if divs:
        raise FidelityError(
            f"engine_determinism: the identical request set served "
            f"twice is not bit-identical — outputs differ on "
            f"{len(divs)}/{n} requests (first divergences at "
            f"{[d for _, d in divs][:8]}); nondeterministic serving "
            "INVALIDATES every other verdict in this gate (each is a "
            "single-run equality comparison)")


def _check_scale0_noop(ev: Mapping[str, Any],
                       diag: Dict[str, Any]) -> None:
    a = _id_lists("scale0_noop", ev, "scale0_ids")
    b = _id_lists("scale0_noop", ev, "unsteered_ids")
    n, divs = _paired("scale0_noop", a, b)
    diag.update(n=n, n_divergent=len(divs),
                first_divergences=[[i, d] for i, d in divs])
    if divs:
        raise FidelityError(
            f"scale0_noop: an object selected at scale 0 is NOT a "
            f"no-op — outputs differ from the no-selection run on "
            f"{len(divs)}/{n} requests (first divergences at "
            f"{[d for _, d in divs][:8]}; a first divergence at 0 "
            "points at a wrong-row edit). This is the armed-hook no-op "
            "contract, NOT engine determinism — the two are separate "
            "checks on purpose")


def _check_steering_bites(ev: Mapping[str, Any],
                          diag: Dict[str, Any]) -> None:
    if "arms" not in ev:
        raise FidelityError("steering_bites: evidence lacks 'arms' — "
                            "no evidence is a failure, not a pass")
    arms = ev["arms"]
    if not isinstance(arms, Mapping):
        raise FidelityError(
            f"steering_bites: 'arms' must be a mapping of per-arm "
            f"probes, got type {type(arms).__name__}")
    if not arms:
        raise FidelityError(
            "steering_bites: zero arms probed — an empty probe set "
            "proves nothing; vacuous evidence is a failure, never a "
            "pass")
    arm_diag: Dict[str, Any] = {}
    diag["arms"] = arm_diag
    silent: List[str] = []
    for j, (key, sub) in enumerate(arms.items()):
        name = _safe_name(key, j)
        check = f"steering_bites arm {name}"
        if not isinstance(sub, Mapping):
            raise FidelityError(
                f"{check}: per-arm evidence must be a mapping, got "
                f"type {type(sub).__name__}")
        a = _id_lists(check, sub, "steered_ids")
        b = _id_lists(check, sub, "scale0_ids")
        n, divs = _paired(check, a, b)
        arm_diag[name] = {"n": n, "n_differing": len(divs),
                          "first_divergences": [[i, d] for i, d in divs]}
        if not divs:
            silent.append(name)
    if silent:
        raise FidelityError(
            f"steering_bites: arm(s) {sorted(silent)} produce NO "
            "difference between the serving scale and scale 0 on any "
            "probe — a silently no-op steering path (vllm-lens's own "
            "documented failure mode is silently capturing nothing); "
            "an arm that changes nothing is a failure, never a pass")


def check_logprob_steering_bites(ev: Mapping[str, Any],
                                 diag: Dict[str, Any], *,
                                 min_ratio: float = 2.0,
                                 abs_floor: float = 1e-3) -> None:
    """The LOGLIKELIHOOD positive control (served-MMLU landing): steered
    vs scale-0 LETTER LOGPROBS at the scored position must differ — and
    differ by MORE than the engine's own run-to-run numeric noise, which
    a same-leg repeat measures alongside.

    This is :func:`_check_steering_bites`'s shape (a live diff against
    the path-matched scale-0 twin; ``intervention_hash``-of-spec and
    ``effective_scale`` stayed green through the real mask bug and are
    deliberately NOT controls), adapted to a CONTINUOUS readout. Token
    equality needs no tolerance; logprobs cut BOTH ways: bit-identical
    steered-vs-twin values are the silently-unsteered failure, but pure
    numeric noise also perturbs logprobs (the documented text-diff
    hypersensitivity, inverted), so "they differ" alone would certify a
    dead steer whose only difference IS the noise. Hence the third leg —
    ``scale0_repeat``, the SAME requests to the SAME twin again — whose
    divergence from ``scale0`` is the measured noise ceiling, and the
    verdict demands the steered divergence clear
    ``max(abs_floor, min_ratio * noise_ceiling)``. Both thresholds are
    stated in the diagnostic, never implicit.

    ``ev``: ``{"items": [{"item_id", "letter_ids", "steered", "scale0",
    "scale0_repeat"}]}`` — per probe item, the per-letter logprob vector
    read at the scored position under each leg, aligned index-for-index
    with ``letter_ids``. Every malformation refuses: zero items, length
    mismatches, or a non-finite value (an engine returning NaN logprobs
    is a failure, not a skip)."""
    items = ev.get("items")
    if not isinstance(items, Sequence) or not items:
        raise FidelityError(
            "logprob_steering_bites: zero probe items — vacuous evidence "
            "is a failure, never a pass")
    d_steer: List[float] = []
    d_noise: List[float] = []
    per_item: List[Dict[str, Any]] = []
    for j, it in enumerate(items):
        if not isinstance(it, Mapping):
            raise FidelityError(
                f"logprob_steering_bites: item {j} is not a mapping")
        name = it.get("item_id", f"item{j}")
        legs = {}
        for leg in ("steered", "scale0", "scale0_repeat"):
            vals = it.get(leg)
            if (not isinstance(vals, Sequence) or len(vals) < 2
                    or not all(isinstance(v, (int, float))
                               and not isinstance(v, bool)
                               and math.isfinite(v) for v in vals)):
                raise FidelityError(
                    f"logprob_steering_bites: item {name!r} leg {leg!r} "
                    f"must be >=2 finite letter logprobs, got {vals!r}")
            legs[leg] = [float(v) for v in vals]
        k = len(legs["steered"])
        if any(len(v) != k for v in legs.values()):
            raise FidelityError(
                f"logprob_steering_bites: item {name!r} legs disagree on "
                "letter count — the three legs must read the SAME letters "
                "at the same positions; refusing a mis-aligned probe")
        ds = max(abs(a - b) for a, b in zip(legs["steered"], legs["scale0"]))
        dn = max(abs(a - b) for a, b in
                 zip(legs["scale0"], legs["scale0_repeat"]))
        d_steer.append(ds)
        d_noise.append(dn)
        per_item.append({"item_id": name, "d_steered_vs_scale0": ds,
                         "d_scale0_repeat": dn})
    steer_median = sorted(d_steer)[len(d_steer) // 2]
    noise_ceiling = max(d_noise)
    bar = max(float(abs_floor), float(min_ratio) * noise_ceiling)
    diag.update({
        "n_items": len(per_item),
        "per_item": per_item,
        "steered_divergence_median": steer_median,
        "noise_ceiling": noise_ceiling,
        "min_ratio": float(min_ratio),
        "abs_floor": float(abs_floor),
        "bar": bar,
    })
    if all(d == 0.0 for d in d_steer):
        raise FidelityError(
            "logprob_steering_bites: steered letter logprobs are "
            "BIT-IDENTICAL to the scale-0 twin's on every probe — the "
            "steer is not reaching the scored positions, and every served "
            "loglikelihood would be the base model's under the steered "
            "cell's name; refusing (the silently-unsteered failure this "
            "control exists to catch)")
    if steer_median < bar:
        raise FidelityError(
            f"logprob_steering_bites: steered-vs-scale0 divergence "
            f"(median {steer_median:.3e}) does not clear the bar "
            f"{bar:.3e} (= max(abs_floor {abs_floor:.1e}, {min_ratio}x "
            f"the measured repeat-noise ceiling {noise_ceiling:.3e})) — "
            "the difference is indistinguishable from the engine's own "
            "numeric noise, so the steer cannot be certified to reach "
            "the scored positions; refusing rather than filing numbers "
            "that may be the base model's")


def _check_batch_invariance(ev: Mapping[str, Any],
                            diag: Dict[str, Any]) -> None:
    problems: List[str] = []
    for direction, la, lb in (
            ("steered", "steered_reference_ids", "steered_batched_ids"),
            ("unsteered", "unsteered_reference_ids", "unsteered_batched_ids")):
        check = f"batch_invariance {direction}"
        a = _id_lists(check, ev, la)
        b = _id_lists(check, ev, lb)
        n, divs = _paired(check, a, b)
        diag[direction] = {"n": n, "n_divergent": len(divs),
                           "first_divergences": [[i, d] for i, d in divs]}
        if divs:
            problems.append(
                f"{direction} requests changed when batched: "
                f"{len(divs)}/{n} differ (first divergences at "
                f"{[d for _, d in divs][:8]})")
    if problems:
        raise FidelityError(
            "batch_invariance: " + "; ".join(problems) +
            " — cross-request contamination under continuous batching")


def _check_selection_honored(ev: Mapping[str, Any],
                             diag: Dict[str, Any]) -> None:
    if "requests" not in ev:
        raise FidelityError("selection_honored: evidence lacks "
                            "'requests' — no evidence is a failure, "
                            "not a pass")
    reqs = ev["requests"]
    if not isinstance(reqs, (list, tuple)):
        raise FidelityError(
            f"selection_honored: 'requests' must be a list, got type "
            f"{type(reqs).__name__}")
    if not reqs:
        raise FidelityError(
            "selection_honored: zero requests — an empty mixed batch "
            "proves nothing; vacuous evidence is a failure, never a "
            "pass")
    parsed: List[Tuple[int, Any, str, int, List[int], List[int]]] = []
    for i, r in enumerate(reqs):
        check = f"selection_honored request {i}"
        if not isinstance(r, Mapping):
            raise FidelityError(f"{check}: must be a mapping, got type "
                                f"{type(r).__name__}")
        arm = r.get("arm")
        if arm is not None and not isinstance(arm, str):
            raise FidelityError(
                f"{check}: 'arm' must be a string or None, got type "
                f"{type(arm).__name__}")
        name = "none" if arm is None else _safe_name(arm, i)
        pi = r.get("prompt_index")
        if isinstance(pi, bool) or not isinstance(pi, int) or pi < 0:
            raise FidelityError(
                f"{check}: 'prompt_index' must be a non-negative "
                "integer (equal indices = the same probe prompt)")
        ref = _id_row(check, r, "reference_ids")
        batched = _id_row(check, r, "batched_ids")
        if not ref and not batched:
            raise FidelityError(
                f"{check}: generated no tokens in either composition "
                "— [] == [] is vacuous evidence, a failure, never a "
                "pass")
        parsed.append((i, arm, name, pi, ref, batched))

    mismatches: List[Tuple[int, str, int]] = []
    for i, arm, name, pi, ref, batched in parsed:
        d = first_divergence_index(ref, batched)
        if d is not None:
            mismatches.append((i, name, d))
    selections = {arm for _, arm, _, _, _, _ in parsed}
    diag.update(n_requests=len(parsed), n_selections=len(selections),
                n_mismatched=len(mismatches),
                mismatches=[[i, d] for i, _, d in mismatches])

    # a REAL mismatch outranks the certification refusals below: it is
    # an observed routing/contamination failure, not merely evidence
    # too weak to certify a pass.
    if mismatches:
        head = "; ".join(
            f"request {i} (selection {name}) first diverges at {d}"
            for i, name, d in mismatches[:8])
        raise FidelityError(
            f"selection_honored: {len(mismatches)}/{len(parsed)} "
            "requests' mixed-batch output differs from their same-size "
            f"output — per-request selection is NOT honored ({head})")
    if len(selections) < 2:
        raise FidelityError(
            "selection_honored: only one distinct selection in the "
            "batch — a mixed batch needs at least TWO distinct "
            "selections to evidence per-request routing; refusing to "
            "certify")
    # distinguishability (run_multi_lora_selection's discipline): the
    # pass is certified only if some probe prompt SEPARATES two
    # selections' reference outputs — otherwise a server routing every
    # request to one object would pass the comparison above.
    by_prompt: Dict[int, Dict[Any, Tuple[int, ...]]] = {}
    for i, arm, name, pi, ref, batched in parsed:
        by_prompt.setdefault(pi, {}).setdefault(arm, tuple(ref))
    shared = False
    distinguishes = False
    for per_arm in by_prompt.values():
        if len(per_arm) < 2:
            continue
        shared = True
        if len(set(per_arm.values())) >= 2:
            distinguishes = True
            break
    if not shared:
        raise FidelityError(
            "selection_honored: no probe prompt is served under more "
            "than one selection — the observations cannot distinguish "
            "routing at all; refusing to certify (serve at least one "
            "shared prompt_index under two different selections)")
    if not distinguishes:
        raise FidelityError(
            "selection_honored: on every probe prompt served under "
            "multiple selections, all selections produce IDENTICAL "
            "reference outputs — the observations cannot distinguish "
            "routing (a server sending every request to one object "
            "would pass); refusing to certify on indistinguishable "
            "evidence")


def evaluate_stacked_identity(evidence: Mapping[str, Any]
                              ) -> Dict[str, Any]:
    """The TWO-SIDED identity assertion for STACKED cells: a
    stacked cell's output must DIFFER from BOTH of its singles — the
    object-only leg AND the prompt-only leg — on at least one probe
    each. One-sided agreement is exactly how a silently-dropped leg
    presents: stack == object-only means the prompt leg never reached
    the model; stack == prompt-only means the object never bit. (The
    earlier stacking attempt was invalidated by precisely such a
    dropped leg, discovered only after its conclusions were drawn.)

    Same discipline as the five checks above: exact token comparison,
    vacuous evidence (zero cells, empty id lists, a pair empty on both
    sides) is a FAILURE, malformed evidence refuses by type name, and
    EVERY failing cell is named — never just the first. Probes must be
    prompts where the object actually bites; a benign probe an object
    is inert on will (correctly) fail the object side here, the same
    way it would fail ``steering_bites``.

    Evidence schema (per stacked cell, lists aligned by probe index;
    every run same engine, greedy, same budget, same batch discipline):

        {"cells": {"<stacked_cell_id>": {
            "stacked_ids":     [[int], ...],  # the stack itself
            "object_only_ids": [[int], ...],  # same prompts, object leg
                                              #   alone (no SI prefix)
            "prompt_only_ids": [[int], ...],  # same prompts, SI prefix
        }}}                                   #   alone (no selector)

    Returns the per-cell diagnostics mapping on success; raises
    :class:`FidelityError` naming every failure otherwise."""
    if not isinstance(evidence, Mapping):
        raise FidelityError(
            "evaluate_stacked_identity: evidence must be a mapping, got "
            f"type {type(evidence).__name__}")
    if "cells" not in evidence:
        raise FidelityError(
            "stacked_identity: evidence lacks 'cells' — no evidence is a "
            "failure, not a pass")
    cells = evidence["cells"]
    if not isinstance(cells, Mapping):
        raise FidelityError(
            f"stacked_identity: 'cells' must be a mapping of per-cell "
            f"probes, got type {type(cells).__name__}")
    if not cells:
        raise FidelityError(
            "stacked_identity: zero stacked cells probed — an empty probe "
            "set proves nothing; vacuous evidence is a failure, never a "
            "pass")
    diagnostics: Dict[str, Any] = {}
    failures: List[str] = []
    for j, (key, sub) in enumerate(cells.items()):
        name = _safe_name(key, j)
        check = f"stacked_identity cell {name}"
        if not isinstance(sub, Mapping):
            raise FidelityError(
                f"{check}: per-cell evidence must be a mapping, got type "
                f"{type(sub).__name__}")
        stacked = _id_lists(check, sub, "stacked_ids")
        against = {}
        for side, label in (("object_only", "object_only_ids"),
                            ("prompt_only", "prompt_only_ids")):
            other = _id_lists(check, sub, label)
            n, divs = _paired(f"{check} vs {side}", stacked, other)
            against[side] = {"n": n, "n_differing": len(divs),
                             "first_divergences": [[i, d] for i, d in divs]}
            if not divs:
                failures.append(
                    f"{check}: the stack's output is IDENTICAL to its "
                    f"{side} single on every probe — the "
                    f"{'prompt' if side == 'object_only' else 'object'} "
                    "leg contributed nothing, i.e. one leg of the stack "
                    "was dropped (or the probes cannot expose it); a "
                    "one-sided stack is a failure, never a pass")
        diagnostics[name] = against
    if failures:
        raise FidelityError(
            f"stacked_identity: {len(failures)} two-sided identity "
            "failure(s):\n- " + "\n- ".join(failures))
    return diagnostics


_CHECK_FNS = {
    "engine_determinism": _check_engine_determinism,
    "scale0_noop": _check_scale0_noop,
    "steering_bites": _check_steering_bites,
    "batch_invariance": _check_batch_invariance,
    "selection_honored": _check_selection_honored,
}
assert tuple(_CHECK_FNS) == FIDELITY_CHECKS


# --- the report (numbers and verdicts only) ------------------------------------


def _assert_numbers_and_verdicts(obj: Any, path: str = "report") -> None:
    """The report's structural privacy backstop: only None/bool/int/
    float/str leaves inside dicts and lists — anything else refuses.
    (Strings reaching here are composed by this module; evidence values
    are excluded upstream by the intake type checks and the withheld
    exception texts.)"""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            if not isinstance(k, str):
                raise FidelityError(
                    f"{path}: non-string key of type {type(k).__name__} "
                    "may not enter a fidelity report")
            _assert_numbers_and_verdicts(v, f"{path}.{k}")
        return
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _assert_numbers_and_verdicts(v, f"{path}[{i}]")
        return
    raise FidelityError(
        f"{path}: type {type(obj).__name__} may not enter a fidelity "
        "report (numbers and verdicts only)")


@dataclass(frozen=True)
class FidelityReport:
    """The gate's structured verdict — numbers and verdicts only.

    ``failures`` names EVERY failed check (never just the first), in
    check order with the invalidating ``engine_determinism`` first when
    it failed. ``invalidated`` is True exactly when engine_determinism
    failed; the other checks' passes are then recorded as
    "indeterminate" in ``checks`` (their comparisons ran, but single-run
    equality on a nondeterministic engine evidences nothing), while
    their failures remain failures. ``diagnostics`` (per check: counts
    and first-divergence indices) are diagnostics ONLY — no pass
    criterion consults them."""

    passed: bool
    invalidated: bool
    failures: Tuple[str, ...]
    checks: Mapping[str, str]
    diagnostics: Mapping[str, Mapping[str, Any]]

    def as_dict(self) -> Dict[str, Any]:
        """JSON-ready form; refuses (loudly) any non-scalar leaf."""
        d: Dict[str, Any] = {
            "passed": self.passed,
            "invalidated": self.invalidated,
            "failures": list(self.failures),
            "checks": dict(self.checks),
            "diagnostics": {k: dict(v)
                            for k, v in self.diagnostics.items()},
        }
        _assert_numbers_and_verdicts(d)
        return d


# --- the top-level evaluation ---------------------------------------------------


def evaluate_fidelity(evidence: Mapping[str, Any]) -> FidelityReport:
    """Evaluate all five checks over gathered evidence and return a
    structured verdict naming EVERY failure (run_phase0's spirit: one
    failure never masks another).

    - A missing probe is a failure, not a pass; so is vacuous evidence.
    - Malformed evidence (wrong types raising inside a checker) is
      normalized into that check's OWN failure with the foreign
      exception's text WITHHELD (type name only — the text can echo
      evidence values); process-level interrupts (KeyboardInterrupt /
      SystemExit) are deliberately not caught.
    - An unknown evidence key is a failure of the whole report: a typo
      there silently orphans its probe.
    - ``engine_determinism`` failing marks the report ``invalidated``
      and downgrades every other check's pass to "indeterminate" — see
      the class docstring.

    Raises :class:`FidelityError` only on caller misuse (evidence not a
    mapping); gathered-evidence problems always come back as report
    failures instead."""
    if not isinstance(evidence, Mapping):
        raise FidelityError(
            "evaluate_fidelity: evidence must be a mapping of check "
            f"names to gathered evidence, got type "
            f"{type(evidence).__name__}")

    failures: Dict[str, str] = {}
    diagnostics: Dict[str, Dict[str, Any]] = {}
    extra: List[str] = []

    unknown = [k for k in evidence if k not in EVIDENCE_KEYS]
    if unknown:
        shown = sorted(_safe_name(k, j) for j, k in enumerate(unknown))
        extra.append(
            f"evidence: unknown evidence key(s) {shown} — allowed keys "
            f"are {list(EVIDENCE_KEYS)}; a typo'd key silently orphans "
            "its probe, so it refuses")

    for name in FIDELITY_CHECKS:
        diag: Dict[str, Any] = {}
        diagnostics[name] = diag
        try:
            ev = evidence.get(name)
            if ev is None:
                raise FidelityError(
                    f"{name}: no evidence gathered — a missing probe "
                    "is a failure, not a pass")
            if not isinstance(ev, Mapping):
                raise FidelityError(
                    f"{name}: evidence must be a mapping, got type "
                    f"{type(ev).__name__}")
            _CHECK_FNS[name](ev, diag)
        except FidelityError as e:
            msg = str(e)
            failures[name] = (msg if msg.startswith(name)
                              else f"{name}: {msg}")
        except Exception as e:  # malformed evidence — never aborts the loop
            failures[name] = (
                f"{name}: evidence could not be decoded "
                f"({type(e).__name__}) — malformed evidence fails this "
                "check alone, not the rest; the exception text is "
                "withheld because it can echo evidence values (the "
                "report is numbers and verdicts only)")

    invalidated = "engine_determinism" in failures
    checks: Dict[str, str] = {}
    for name in FIDELITY_CHECKS:
        if name in failures:
            checks[name] = failures[name]
        elif invalidated:
            checks[name] = (
                "indeterminate: the comparison observed no divergence, "
                "but engine_determinism FAILED — single-run equality "
                "on a nondeterministic engine evidences nothing")
        else:
            checks[name] = "passed"

    all_failures = tuple(failures[n] for n in FIDELITY_CHECKS
                         if n in failures) + tuple(extra)
    return FidelityReport(passed=not all_failures,
                          invalidated=invalidated,
                          failures=all_failures,
                          checks=checks,
                          diagnostics=diagnostics)


# --- the multi-LoRA selection check (§10 check 4), reusable by the drivers -----------

#: the two NAMED selection policies.
#: ``legacy`` is the lens/registry-dir gate's discipline as it stood in
#: ``scripts/petri_serving_gate.py`` before the move — byte-for-byte, an
#: OUTPUT-based verdict with its stated limit (a batch-of-N output comparison
#: at BI=0 under concurrent submission equalises the row count, not the
#: computation; on the lens engine at the gate's small mixed plan it has held
#: so far, and a flip there refuses honestly — never a silent pass);
#: ``native`` is the lora-native standup's, where since the gemma canary job
#: 489308 the VERDICT is STRUCTURAL — :func:`check_multi_lora_routing_attested`
#: over the engine's own per-request LoRA bookkeeping — and every output
#: comparison is a RECORDED diagnostic (:func:`multi_lora_output_diagnostics`).
MULTI_LORA_POLICY_LEGACY = "legacy"
MULTI_LORA_POLICY_NATIVE = "native"
#: the structural verdict label (``serving_attestation`` names it once for the
#: admission's reader; pinned equal by the tests)
SELECTION_VERDICT_ROUTING_ATTESTED = "routing_attested"
#: how many first-divergence triples a recorded diagnostic keeps verbatim
_DIAG_KEEP = 16


def _selection_args(*, adapters: Sequence[str], prompt_set: Sequence[Any],
                    base: Optional[str], zero_adapter: Optional[str]
                    ) -> Tuple[str, List[str], List[str], List[Any]]:
    """The argument refusals shared by every multi-LoRA selection entry point
    (verbatim the messages the legacy verdict has always raised). Returns
    ``(policy, names, genuine, prompts)`` with ``names = [base, *adapters]``."""
    if not isinstance(base, str) or not base:
        raise FidelityError("multi-LoRA selection needs an explicit base name")
    adapters = list(adapters)
    if not all(isinstance(a, str) and a for a in adapters):
        raise FidelityError(
            f"multi-LoRA selection: every adapter must be a non-empty name, got "
            f"{adapters!r}")
    prompts = list(prompt_set)
    if not prompts:
        raise FidelityError("multi-LoRA selection: an empty prompt set proves nothing — "
                            "vacuous evidence is a failure, never a pass")
    if zero_adapter is None:
        policy = MULTI_LORA_POLICY_LEGACY
        if (len(adapters) < 2 or len(set(adapters)) != len(adapters)
                or base in adapters):
            raise FidelityError(
                f"multi-LoRA selection needs the base plus at least TWO "
                f"distinct adapters (mixing 'several adapters', plan §10) — "
                f"got base={base!r}, adapters={adapters!r}")
        genuine = list(adapters)
    else:
        policy = MULTI_LORA_POLICY_NATIVE
        if not isinstance(zero_adapter, str) or not zero_adapter:
            raise FidelityError(f"multi-LoRA selection (native policy): zero_adapter must be a "
                                f"non-empty name, got {zero_adapter!r}")
        if zero_adapter not in adapters:
            raise FidelityError(
                f"multi-LoRA selection (native policy): zero_adapter {zero_adapter!r} is not "
                f"among the adapters {adapters!r} — the zero adapter is IN the selection "
                "(its solo outputs are held to base's); refusing")
        if len(set(adapters)) != len(adapters) or base in adapters:
            raise FidelityError(
                f"multi-LoRA selection (native policy): adapters must be distinct names "
                f"not including the base — got base={base!r}, adapters={adapters!r}")
        genuine = [a for a in adapters if a != zero_adapter]
        if not genuine:
            raise FidelityError(
                f"multi-LoRA selection (native policy): no genuine adapter beside the zero "
                f"adapter {zero_adapter!r} — a selection check with nothing to select "
                "proves nothing; refusing")
    return policy, [base, *adapters], genuine, prompts


def _rotated_layouts(names: Sequence[str], P: int) -> List[List[Tuple[str, int]]]:
    """THE ROTATED MIXED LAYOUTS: position k carries prompt k % P in EVERY
    layout; layout r places selection ``names[(k // P + r) % n]`` there. Over
    the n layouts each selection visits every block, hence every position."""
    n_sel = len(names)
    N = n_sel * P
    return [[(names[(k // P + r) % n_sel], k % P) for k in range(N)] for r in range(n_sel)]


def _rows(outs: Any, N: int, what: str) -> List[Tuple[int, ...]]:
    outs = list(outs)
    if len(outs) != N:
        raise FidelityError(f"{what} returned {len(outs)} outputs for {N} requests")
    return [tuple(int(t) for t in o) for o in outs]


def _rows_with_ids(outs: Any, N: int, what: str) -> Tuple[List[Tuple[int, ...]], List[str]]:
    """``[(token_ids, request_id), ...]`` split into rows and ids; a row without
    a non-empty string id, or a duplicate id inside one batch, refuses — an
    instrument that cannot name its own requests cannot be attested."""
    outs = list(outs)
    if len(outs) != N:
        raise FidelityError(f"{what} returned {len(outs)} outputs for {N} requests")
    rows: List[Tuple[int, ...]] = []
    ids: List[str] = []
    for o in outs:
        if not isinstance(o, (tuple, list)) or len(o) != 2:
            raise FidelityError(
                f"{what}: every row must be a (token_ids, request_id) pair, got "
                f"{type(o).__name__}")
        toks, rid = o
        if not isinstance(rid, str) or not rid:
            raise FidelityError(
                f"{what}: a row carries no engine request id — the served response's 'id' "
                "must reach the control unchanged (the engine request id of a single-prompt "
                "chat completion); refusing")
        rows.append(tuple(int(t) for t in toks))
        ids.append(rid)
    if len(set(ids)) != len(ids):
        raise FidelityError(
            f"{what}: {len(ids) - len(set(ids))} duplicate request id(s) inside one batch — "
            "the engine mints one id per request, so a duplicate is a broken instrument; "
            "refusing")
    return rows, ids


def _positional_undistinguishable(ref: Mapping[str, Sequence[Tuple[int, ...]]],
                                  names: Sequence[str], N: int, P: int, *,
                                  exempt: Callable[[str, str], bool]
                                  ) -> Tuple[List[Tuple[int, int, str, str]], int]:
    """Every (position, selection, prohibited substitute) triple whose same-size
    references agree at that position, and the number of ordered pairs
    checked."""
    undistinguishable: List[Tuple[int, int, str, str]] = []
    n_pairs_checked = 0
    for k in range(N):
        for s_name in names:
            for t_name in names:
                if t_name == s_name or exempt(s_name, t_name):
                    continue
                n_pairs_checked += 1
                if ref[s_name][k] == ref[t_name][k]:
                    undistinguishable.append((k, k % P, s_name, t_name))
    return undistinguishable, n_pairs_checked


def _batch_invariance_record(ref: Mapping[str, Sequence[Tuple[int, ...]]],
                             names: Sequence[str], prompts: Sequence[Any], P: int,
                             solo: Optional[Callable[[str, Any], Sequence[int]]]
                             ) -> Dict[str, Any]:
    """RECORDED, never gating (module docstring check 4): the solo (batch
    of one) against the same-size row — the number that is legitimately
    non-zero at BI=0 — and the repeats of one (selection, prompt) inside a
    same-size batch."""
    n_sel = len(names)
    batch_invariance: Dict[str, Any] = {
        "note": ("recorded only (fidelity check 4, D27): batch-size / composition noise at BI=0 "
                 "is never the selection verdict's evidence"),
        "solo_vs_same_size": {}, "within_batch_repeats": {}}
    if solo is not None:
        for j, name in enumerate(names):
            divs: List[List[int]] = []
            for i in range(P):
                s_out = tuple(int(t) for t in solo(name, prompts[i]))
                d = first_divergence_index(s_out, ref[name][j * P + i])
                if d is not None:
                    divs.append([i, d])
            batch_invariance["solo_vs_same_size"][name] = {
                "n": P, "n_divergent": len(divs), "first_divergences": divs}
    for name in names:
        divs = []
        for i in range(P):
            rows = [ref[name][j * P + i] for j in range(n_sel)]
            for j in range(1, len(rows)):
                d = first_divergence_index(rows[0], rows[j])
                if d is not None:
                    divs.append([i, j, d])
        batch_invariance["within_batch_repeats"][name] = {
            "n_pairs": P * (n_sel - 1), "n_divergent": len(divs),
            "first_divergences": divs[:16]}
    return batch_invariance


def check_multi_lora_selection(
    *, adapters: Sequence[str],
    solo: Optional[Callable[[str, Sequence[Any]], Sequence[int]]],
    mixed_batch: Callable[[Sequence[Tuple[str, Any]]], Sequence[Sequence[int]]],
    prompt_set: Sequence[Any],
    base: Optional[str] = None,
    zero_adapter: Optional[str] = None,
) -> Dict[str, Any]:
    """Adapter SELECTION, served-vs-served, under THE EQUAL-SIZE RULE (module
    docstring check 5): one batch mixing every (selection, prompt) pair is
    compared, row for row at the same position, with the layout re-labelled
    under ONE selection each (a SAME-SIZE reference batch — N rows, the same
    prompt at every position, greedy, the same ``max_tokens``; the caller's
    ``mixed_batch`` produces both). The mixed assignment is ROTATED across the
    positions so every selection is compared at every position, and each
    (selection, position) is judged DISTINGUISHABLE there first: its reference
    must differ from every prohibited substitute's (the one native exemption
    is base <-> zero adapter). The solo output is RECORDED against the
    same-size row under ``batch_invariance`` (check 4) beside the
    within-batch repeats; ``solo=None`` skips that diagnostic.

    ``mixed_batch([(name, prompt), ...])`` returns token-id sequences from the
    SAME engine in request order; ``solo(name, prompt)`` one request alone;
    ``prompt_set`` is the caller's frozen prompt set. Names in the messages
    are the caller's SELECTION LABELS (virtual model names), never evidence
    values — the module's message discipline holds.

    ``zero_adapter=None`` — the LENS / LEGACY policy, the gate's
    ``run_multi_lora_selection`` discipline, an OUTPUT-BASED VERDICT: ``base``
    an explicit non-empty name; at least TWO distinct adapters, base not
    among them; every selection's same-size reference signature pairwise
    distinct; every (selection, position) distinguishable from every other
    selection; every mixed row equal to its same-size reference row. Raises
    :class:`FidelityError` on any of those. STATED LIMIT: the caller
    submits the N rows concurrently and the engine admits them as they
    arrive, so two submissions of "the same N requests" need not be the same
    step-by-step computation at BI=0 — the equal-size rule equalises the row
    count, not the computation. On the lens engine at the gate's small mixed
    plan this has held; a flip there refuses honestly (never a silent pass),
    and the recorded batch-invariance numbers show it first. The lora-native
    path does not rely on it.

    ``zero_adapter="<name>"`` — the NATIVE policy. Since the gemma canary
    job 489308 (the zero adapter differed from base on ONE prompt in a
    33-row concurrent batch while agreeing bit for bit one request at a
    time) this function is NO LONGER the native verdict: it returns
    :func:`multi_lora_output_diagnostics` — the rotated layouts, same-size
    references, positional distinguishability, zero-vs-base, within-batch
    repeats, solo-vs-same-size and the base reference submitted TWICE
    (``fixed_size_repeat_floor``) — as a RECORD that never raises on
    disagreement. The native verdict is :func:`check_multi_lora_routing_
    attested` over the engine's own per-request LoRA bookkeeping, driven by
    :func:`run_multi_lora_routing_control`. The argument refusals (zero
    adapter among the adapters, a genuine adapter beside it, an explicit
    base, distinct names) still raise."""
    policy, names, genuine, prompts = _selection_args(
        adapters=adapters, prompt_set=prompt_set, base=base, zero_adapter=zero_adapter)
    if policy == MULTI_LORA_POLICY_NATIVE:
        return multi_lora_output_diagnostics(
            base=base, adapters=adapters, prompt_set=prompt_set, mixed_batch=mixed_batch,
            solo=solo, zero_adapter=zero_adapter)
    adapters = list(adapters)
    P = len(prompts)
    n_sel = len(names)
    N = n_sel * P
    layouts = _rotated_layouts(names, P)

    # THE SAME-SIZE REFERENCES (the equal-size rule): the layout re-labelled under
    # ONE selection each — same size N, the same prompt at every position (the
    # prompt at k is layout-independent), only the selection composition differs
    # from a mixed batch; ONE set of references serves every rotation
    ref: Dict[str, List[Tuple[int, ...]]] = {}
    for name in names:
        ref[name] = _rows(mixed_batch([(name, prompts[k % P]) for k in range(N)]), N,
                          f"same-size reference batch for {name!r}")
    # whole-layout distinguishability: the named refusal for the cases that
    # collide EVERYWHERE
    sigs = {name: tuple(ref[name]) for name in names}
    signatures = list(sigs.values())
    if len(set(signatures)) != len(signatures):
        colliding = sorted(
            n for n in names if signatures.count(sigs[n]) > 1)
        raise FidelityError(
            f"multi-LoRA selection cannot be certified: {colliding} produce "
            "identical same-size reference outputs on EVERY gate prompt — the "
            "observations cannot distinguish routing (a server sending everything "
            "to one of them — total cross-request adapter contamination — would "
            "pass); certify only with distinguishable outputs or trusted "
            "server-side selection evidence")

    # POSITIONAL distinguishability: within-batch variation is
    # permitted (recorded, never gated), so a selection distinct over its WHOLE
    # layout may still equal a substitute exactly where the mixed batch runs it —
    # every (selection, position) a rotated layout compares must differ from every
    # other selection at that position; otherwise that row is vacuous and the check
    # refuses, naming it.
    undistinguishable, n_pairs_checked = _positional_undistinguishable(
        ref, names, N, P, exempt=lambda a, b: False)
    if undistinguishable:
        k, i, s_name, t_name = undistinguishable[0]
        raise FidelityError(
            f"multi-LoRA selection cannot be certified: at batch position {k} (gate prompt {i}) "
            f"selection {s_name!r} produces the same same-size reference output as {t_name!r}, "
            f"so a mixed row there cannot distinguish routing from {t_name!r} (cross-request "
            "adapter contamination of that row would pass unseen); the check refuses as vacuous "
            f"for that row ({len(undistinguishable)} undistinguishable (position, selection, "
            f"substitute) triple(s) over {N} positions; first: "
            f"{[[kk, ss, tt] for kk, _ii, ss, tt in undistinguishable[:8]]})")

    # THE MIXED BATCHES, one per rotated layout, row for row against the same-size
    # reference AT THE SAME POSITION
    mismatches: List[Tuple[int, int, str, int, int]] = []
    for r, layout in enumerate(layouts):
        outs = _rows(mixed_batch([(s_name, prompts[i]) for s_name, i in layout]), N,
                     f"mixed batch (layout {r})")
        for k, ((s_name, i), got) in enumerate(zip(layout, outs)):
            d = first_divergence_index(ref[s_name][k], got)
            if d is not None:
                mismatches.append((r, k, s_name, i, d))
    if mismatches:
        r, k, s_name, i, d = mismatches[0]
        raise FidelityError(
            f"multi-LoRA mixed-batch selection FAILED: request for {s_name!r} on gate prompt {i} "
            f"(layout {r}, batch position {k}) did not return {s_name!r}'s SAME-SIZE reference "
            f"output (first divergence at token {d}) — cross-request adapter contamination "
            f"({len(mismatches)}/{N * n_sel} rows over {n_sel} layouts; first divergences "
            f"{[[rr, kk, nn, ii, dd] for rr, kk, nn, ii, dd in mismatches[:8]]})")
    batch_invariance = _batch_invariance_record(ref, names, prompts, P, solo)
    return {
        "policy": policy, "base": base, "adapters": list(adapters),
        "genuine": list(genuine), "zero_adapter": None,
        "n_prompts": P, "n_selections": n_sel,
        "n_batch": N, "n_layouts": n_sel, "n_mixed_rows": N * n_sel,
        "signatures_distinct": True,
        "zero_equals_base": None,
        "rule": "equal-size",
        "verdict": "output_equality",
        "gating": True,
        "stated_limit": ("an output comparison at concurrency > 1 equalises the row count, not "
                         "the computation (D88); a flip here refuses honestly, never passes"),
        "reference": ("same-size single-selection batches: the layout re-labelled under each "
                      "selection, compared row for row at the same position"),
        "layouts": ("rotated: layout r places selection (block + r) mod n_selections at block b "
                    "(P rows each), so every selection is compared at every position"),
        "positional_distinguishability": {
            "n_pairs_checked": n_pairs_checked, "n_undistinguishable": 0,
            "exemption": None},
        "selection_first_divergences": [],
        "batch_invariance": batch_invariance,
    }


def multi_lora_output_diagnostics(
    *, base: str, adapters: Sequence[str], prompt_set: Sequence[Any],
    mixed_batch: Callable[[Sequence[Tuple[str, Any]]], Sequence[Any]],
    solo: Optional[Callable[[str, Sequence[Any]], Sequence[int]]] = None,
    zero_adapter: Optional[str] = None,
    request_ids: bool = False,
) -> Dict[str, Any]:
    """The OUTPUT-space diagnostics of a multi-LoRA selection run — RECORDED,
    never a verdict (gemma canary job 489308). Submits, through
    ``mixed_batch``: one same-size reference batch per selection (the rotated
    layout re-labelled under that selection), the BASE reference a SECOND
    time, and one mixed batch per rotated layout; then records, without ever
    raising on a disagreement:

    - ``whole_layout``: whether the selections' reference signatures are
      pairwise distinct (``colliding`` names them), ``zero_vs_base`` (the
      zero adapter's reference against base's: per-prompt count and first
      divergences; ``zero_equals_base``), ``genuine_like_base``;
    - ``positional_distinguishability``: ``n_pairs_checked``,
      ``n_undistinguishable`` and the first triples, base <-> zero exempt
      under the native policy;
    - ``selection_agreement``: per selection, the fraction of its mixed rows
      that equal the same-size reference row at the same position, with the
      first divergences (``selection_first_divergences`` keeps the flat list,
      ``n_mismatched_rows`` the count);
    - ``fixed_size_repeat_floor``: the base reference submitted twice, row
      for row — the noise of two same-size, same-composition submissions.
      Together with ``zero_vs_base`` this separates the two hypotheses of
      If base-vs-base differs about like zero-vs-base, the difference
      is scheduling nondeterminism at BI=0 (different admission order ->
      different per-step compositions); if base-vs-base is identical while
      zero-vs-base is not, a LoRA-active kernel path differs from the
      LoRA-free one (``cudagraph_specialize_lora``);
    - ``batch_invariance``: solo-vs-same-size (``solo=None`` skips it) and
      within-batch repeats, as before (check 4).

    With ``request_ids=True`` every ``mixed_batch`` row is a ``(token_ids,
    request_id)`` pair and the record carries ``expected_routing`` =
    ``{request_id: selection label}`` over EVERY submitted row (references,
    the repeat and the mixed layouts alike) for the structural check; a row
    without an id, or a duplicate id, refuses (a broken instrument, not a
    disagreement). The only other refusals are the argument refusals of
    :func:`check_multi_lora_selection` and a batch of the wrong length.
    Under the LEGACY policy shape (``zero_adapter=None``) the same record is
    produced with no exemption; the legacy VERDICT is
    :func:`check_multi_lora_selection`."""
    policy, names, genuine, prompts = _selection_args(
        adapters=adapters, prompt_set=prompt_set, base=base, zero_adapter=zero_adapter)
    adapters = list(adapters)
    P = len(prompts)
    n_sel = len(names)
    N = n_sel * P
    layouts = _rotated_layouts(names, P)
    expected: Dict[str, str] = {}
    n_submitted = 0

    def _submit(batch: List[Tuple[str, Any]], what: str) -> List[Tuple[int, ...]]:
        nonlocal n_submitted
        outs = mixed_batch(batch)
        n_submitted += len(batch)
        if not request_ids:
            return _rows(outs, len(batch), what)
        rows, ids = _rows_with_ids(outs, len(batch), what)
        for (sel, _p), rid in zip(batch, ids):
            if rid in expected:
                raise FidelityError(
                    f"{what}: an engine request id was already issued by an earlier batch of "
                    "this control — ids must be unique across the whole control; refusing")
            expected[rid] = sel
        return rows

    ref: Dict[str, List[Tuple[int, ...]]] = {}
    for name in names:
        ref[name] = _submit([(name, prompts[k % P]) for k in range(N)],
                            f"same-size reference batch for {name!r}")
    base_repeat = _submit([(base, prompts[k % P]) for k in range(N)],
                          f"repeated same-size reference batch for {base!r}")

    def _exempt(a: str, b: str) -> bool:
        return policy == MULTI_LORA_POLICY_NATIVE and {a, b} == {base, zero_adapter}

    # whole-layout facts (a collision inside the exempt base <-> zero pair is by
    # design under the native policy, so it is not a collision here)
    sigs = {name: tuple(ref[name]) for name in names}
    colliding = sorted(n for n in names
                       if any(m != n and not _exempt(n, m) and sigs[m] == sigs[n] for m in names))
    zero_vs_base: Optional[Dict[str, Any]] = None
    genuine_like_base = sorted(g for g in genuine if sigs[g] == sigs[base])
    if zero_adapter is not None:
        zdivs = []
        for k in range(N):
            d = first_divergence_index(ref[zero_adapter][k], ref[base][k])
            if d is not None:
                zdivs.append([k, k % P, d])
        zero_vs_base = {"n_rows": N, "n_divergent_rows": len(zdivs),
                        "n_prompts_differing": len({i for _k, i, _d in zdivs}),
                        "first_divergences": zdivs[:_DIAG_KEEP]}

    undistinguishable, n_pairs_checked = _positional_undistinguishable(
        ref, names, N, P, exempt=_exempt)

    # the mixed layouts, recorded row for row
    mismatches: List[List[Any]] = []
    agreement: Dict[str, Dict[str, Any]] = {
        name: {"n_rows": 0, "n_agree": 0, "first_divergences": []} for name in names}
    for r, layout in enumerate(layouts):
        outs = _submit([(s_name, prompts[i]) for s_name, i in layout], f"mixed batch (layout {r})")
        for k, ((s_name, i), got) in enumerate(zip(layout, outs)):
            agreement[s_name]["n_rows"] += 1
            d = first_divergence_index(ref[s_name][k], got)
            if d is None:
                agreement[s_name]["n_agree"] += 1
            else:
                mismatches.append([r, k, s_name, i, d])
                if len(agreement[s_name]["first_divergences"]) < _DIAG_KEEP:
                    agreement[s_name]["first_divergences"].append([r, k, i, d])
    for rec in agreement.values():
        rec["fraction_agree"] = (rec["n_agree"] / rec["n_rows"]) if rec["n_rows"] else None

    # the fixed-size repeat floor: base vs base, same size, same composition
    fdivs = []
    for k in range(N):
        d = first_divergence_index(ref[base][k], base_repeat[k])
        if d is not None:
            fdivs.append([k, k % P, d])
    fixed_size_repeat_floor = {
        "selection": base, "n_rows": N, "n_divergent_rows": len(fdivs),
        "fraction_divergent": len(fdivs) / N, "first_divergences": fdivs[:_DIAG_KEEP],
        "reads_as": ("scheduling nondeterminism at BI=0 when this is non-zero about like "
                     "zero_vs_base; a LoRA-active vs LoRA-free kernel-path difference when "
                     "this is zero while zero_vs_base is not")}

    batch_invariance = _batch_invariance_record(ref, names, prompts, P, solo)
    out: Dict[str, Any] = {
        "policy": policy, "base": base, "adapters": list(adapters),
        "genuine": list(genuine), "zero_adapter": zero_adapter,
        "n_prompts": P, "n_selections": n_sel,
        "n_batch": N, "n_layouts": n_sel, "n_mixed_rows": N * n_sel,
        "n_submitted_rows": n_submitted,
        "rule": "equal-size",
        "gating": False,
        "note": ("recorded diagnostics only (D88, gemma canary job 489308): an output "
                 "comparison at concurrency > 1 is not the selection verdict; the verdict is "
                 "the engine's own per-request LoRA routing (check_multi_lora_routing_attested)"),
        "reference": ("same-size single-selection batches: the layout re-labelled under each "
                      "selection, compared row for row at the same position"),
        "layouts": ("rotated: layout r places selection (block + r) mod n_selections at block b "
                    "(P rows each), so every selection is compared at every position"),
        "signatures_distinct": not colliding,
        "zero_equals_base": (zero_vs_base["n_divergent_rows"] == 0
                             if zero_vs_base is not None else None),
        "whole_layout": {"signatures_distinct": not colliding, "colliding": colliding,
                         "genuine_like_base": genuine_like_base, "zero_vs_base": zero_vs_base},
        "positional_distinguishability": {
            "n_pairs_checked": n_pairs_checked, "n_undistinguishable": len(undistinguishable),
            "first": [[k, s, t] for k, _i, s, t in undistinguishable[:_DIAG_KEEP]],
            "exemption": ("base <-> zero adapter" if policy == MULTI_LORA_POLICY_NATIVE else None)},
        "selection_agreement": agreement,
        "selection_first_divergences": mismatches[:_DIAG_KEEP],
        "n_mismatched_rows": len(mismatches),
        "fixed_size_repeat_floor": fixed_size_repeat_floor,
        "batch_invariance": batch_invariance,
    }
    if request_ids:
        out["expected_routing"] = dict(expected)
        out["n_requests"] = len(expected)
    return out


def _join_engine_ids(expected: Mapping[str, str], observed: Mapping[str, Any]
                     ) -> Tuple[Dict[str, str], Dict[str, List[str]], int]:
    """Join the recorder's ENGINE ids to the control's EXTERNAL ids under
    ``serving_attestation.is_engine_id_of`` (exact, or the external id plus
    exactly ONE ``-<8 hex>`` suffix — never a bare prefix). Returns
    ``(engine_id -> external_id for the joined pairs, external_id -> [engine
    ids] for every expected id, n_foreign)``: an engine id matching no expected
    id is FOREIGN (recorded — the gate canary attaches to a live standup other
    drivers may be using), an external id with two engine ids is AMBIGUOUS
    (the caller refuses)."""
    from sisteer.capeval.serving.serving_attestation import (
        external_request_id_candidates, is_engine_id_of)
    matches: Dict[str, List[str]] = {x: [] for x in expected}
    joined: Dict[str, str] = {}
    n_foreign = 0
    for eid in observed:
        hits = [x for x in external_request_id_candidates(eid)
                if x in expected and is_engine_id_of(eid, x)]
        if not hits:
            n_foreign += 1
            continue
        # at most one candidate can be an expected id AND satisfy is_engine_id_of
        # (the id itself, or its stem) — if both were expected the engine id is the
        # exact form of one and the suffixed form of the other: count it for both so
        # the ambiguity refuses rather than picking one
        for x in hits:
            matches[x].append(str(eid))
        if len(hits) == 1:
            joined[str(eid)] = hits[0]
    return joined, matches, n_foreign


def check_multi_lora_routing_attested(
    *, expected: Mapping[str, str],
    selection_adapter: Mapping[str, Optional[str]],
    observed: Mapping[str, Mapping[str, Any]],
    inconsistent: Mapping[str, Any],
    steps: Sequence[Sequence[str]],
    attested_id_name_map: Mapping[str, str],
    overflow: bool = False,
    steps_overflow: bool = False,
    recorder_errors: Sequence[str] = (),
    runner_kind: Optional[str] = None,
    runner_class: Optional[str] = None,
) -> Dict[str, Any]:
    """THE NATIVE SELECTION VERDICT: every request the control submitted
    ran under EXACTLY its selection's adapter according to the engine's own
    per-request LoRA bookkeeping (``worker_reports.LoraRoutingRecorder``),
    and the control's OWN requests were genuinely mixed.

    ``expected``: ``external_request_id -> selection label`` (every row the
    control submitted, keyed by the response ``id``); ``selection_adapter``:
    ``selection label -> lora_name`` with ``None`` for base (NO LoRA);
    ``observed`` (``engine_request_id -> {lora_int_id, lora_name}``, THE WHOLE
    EPOCH), ``inconsistent``, ``steps`` (one scheduled engine-id list per
    step), ``overflow`` / ``steps_overflow`` / ``recorder_errors`` /
    ``runner_kind`` / ``runner_class``: the recorder's epoch payload;
    ``attested_id_name_map``: the readiness attestation's ``lora_int_id ->
    name`` binding (string keys).

    THE JOIN (canary #4 job 489533): the engine RENAMES every
    request — ``InputProcessor.assign_request_id`` appends ``-<8 hex>`` to the
    client's id — so an external id is joined to the one engine id that is
    it or it plus exactly one suffix (``serving_attestation.is_engine_id_of``;
    a parallel-sampling child ``<i>_<id>-…`` or a merely prefix-sharing id is
    not). Zero matches -> missing; two -> ambiguous; both refuse naming the
    selection. Engine ids matching no expected id are FOREIGN, counted and
    recorded (``n_foreign``), never judged.

    MIXED EVIDENCE IS THE CONTROL'S OWN: ``n_steps_mixed`` is
    computed HERE from the epoch's per-step scheduled sets after the join — a
    step is mixed when it schedules >= 2 distinct SELECTIONS among the
    control's joined requests; a neighbour's traffic, or an earlier control's
    steps, cannot supply it (a fully serial control refuses).

    REFUSES (:class:`FidelityError`; messages carry selection labels,
    adapter names, ids (small ints), the runner class and counts — never a
    request id, never an output) when: ``expected`` is empty; the runner kind
    is not one of the pinned ``v1`` / ``v2`` (an unpinned engine); a selection
    label has no adapter entry, or an adapter is not in the attested map; any
    expected request is missing or ambiguous in the join; any joined
    observation's ``(lora_int_id, lora_name)`` disagrees with the attested map
    (id 0 must carry no name); any request ran under an adapter other than
    ``selection_adapter[expected[rid]]`` — base must run under NO LoRA (id 0)
    and the ZERO ADAPTER under ITS OWN name (structurally there is no
    base <-> zero exemption); any joined request is ``inconsistent`` (seen
    under two bindings); no step schedules >= 2 distinct selections of the
    control's own; ``overflow`` / ``steps_overflow``; any recorder read
    failure. An all-base counterexample refuses here structurally.

    Returns the per-selection counts, the join summary, the steps summary
    (computed), the runner and the id/name map used."""
    if not isinstance(expected, Mapping) or not expected:
        raise FidelityError(
            "multi-LoRA routing attestation: no expected requests — vacuous evidence is a "
            "failure, never a pass")
    if runner_kind not in ("v1", "v2"):
        raise FidelityError(
            f"multi-LoRA routing attestation: the worker's GPU model runner {runner_class!r} is "
            f"of kind {runner_kind!r}, not one of the pinned runners (v1 gpu_model_runner / v2 "
            "gpu.model_runner @0.25.1) — an unpinned engine cannot be attested; refusing")
    if overflow:
        raise FidelityError(
            "multi-LoRA routing attestation: the recorder OVERFLOWED its request-id bound, so "
            "some requests were never recorded — incomplete evidence refuses")
    if steps_overflow:
        raise FidelityError(
            "multi-LoRA routing attestation: the recorder OVERFLOWED its per-step list, so the "
            "mixed-step evidence is incomplete; refusing")
    errors = list(recorder_errors or ())
    if errors:
        raise FidelityError(
            f"multi-LoRA routing attestation: the recorder could not read the engine's state on "
            f"{len(errors)} step(s) — its evidence is incomplete; refusing "
            f"(first: {errors[0][:200]!r})")
    if not isinstance(steps, (list, tuple)):
        raise FidelityError(
            f"multi-LoRA routing attestation: steps is {type(steps).__name__}, not the recorder's "
            "per-step scheduled sets; refusing")
    if not isinstance(attested_id_name_map, Mapping) or not attested_id_name_map:
        raise FidelityError(
            "multi-LoRA routing attestation: an empty attested id_name_map — nothing to hold "
            "the observations to; refusing")
    id_name: Dict[int, str] = {}
    for k, v in attested_id_name_map.items():
        try:
            lid = int(k)
        except (TypeError, ValueError) as e:
            raise FidelityError(
                f"multi-LoRA routing attestation: id_name_map key {k!r} is not an int") from e
        if not isinstance(v, str) or not v or lid == 0:
            raise FidelityError(
                f"multi-LoRA routing attestation: id_name_map[{lid}] = {v!r} is not an adapter "
                "name (id 0 is NO LoRA and never named)")
        id_name[lid] = v
    name_id = {v: k for k, v in id_name.items()}
    if len(name_id) != len(id_name):
        raise FidelityError(
            "multi-LoRA routing attestation: two lora ids share one adapter name in the "
            "attested id_name_map; refusing")
    labels = sorted(set(expected.values()))
    for label in labels:
        if label not in selection_adapter:
            raise FidelityError(
                f"multi-LoRA routing attestation: selection {label!r} has no adapter entry — "
                "every selection must state which adapter (or no LoRA) it selects")
        want = selection_adapter[label]
        if want is not None and want not in name_id:
            raise FidelityError(
                f"multi-LoRA routing attestation: selection {label!r} selects adapter "
                f"{want!r}, which the readiness attestation does not name "
                f"(attested: {sorted(name_id)}); refusing")

    observed = observed if isinstance(observed, Mapping) else {}
    joined, matches, n_foreign = _join_engine_ids(expected, observed)
    per_selection: Dict[str, Dict[str, Any]] = {
        label: {"n": 0, "n_observed": 0, "adapter": selection_adapter[label],
                "lora_int_id": (0 if selection_adapter[label] is None
                                else name_id[selection_adapter[label]])}
        for label in labels}
    by_label_missing: Dict[str, int] = {}
    by_label_ambiguous: Dict[str, int] = {}
    n_exact = 0
    n_suffixed = 0
    engine_of: Dict[str, str] = {}
    for rid, label in expected.items():
        per_selection[label]["n"] += 1
        hits = matches.get(rid, [])
        if not hits:
            by_label_missing[label] = by_label_missing.get(label, 0) + 1
        elif len(hits) > 1:
            by_label_ambiguous[label] = by_label_ambiguous.get(label, 0) + 1
        else:
            engine_of[rid] = hits[0]
            per_selection[label]["n_observed"] += 1
            if hits[0] == rid:
                n_exact += 1
            else:
                n_suffixed += 1
    if by_label_missing:
        detail = ", ".join(f"{label!r}: {n}/{per_selection[label]['n']}"
                           for label, n in sorted(by_label_missing.items()))
        raise FidelityError(
            "multi-LoRA routing attestation FAILED: request(s) never observed by the engine's "
            f"LoRA bookkeeping — {detail} (per selection: missing/submitted; {len(observed)} "
            f"engine ids recorded, {n_foreign} foreign); a request the recorder never saw "
            "cannot be attested")
    if by_label_ambiguous:
        detail = ", ".join(f"{label!r}: {n}" for label, n in sorted(by_label_ambiguous.items()))
        raise FidelityError(
            "multi-LoRA routing attestation FAILED: request(s) matched by MORE THAN ONE engine "
            f"request id — {detail}; the engine mints exactly one internal id per request, so "
            "an ambiguous join is a broken instrument (or a colliding client id); refusing")

    by_label_inconsistent: Dict[str, int] = {}
    by_label_misrouted: Dict[str, List[Tuple[Optional[str], int]]] = {}
    binding_disagreements: List[str] = []
    for rid, label in expected.items():
        eid = engine_of[rid]
        obs = observed.get(eid)
        if not isinstance(obs, Mapping):
            binding_disagreements.append(
                f"selection {label!r}: an observation is not a mapping")
            continue
        lid = obs.get("lora_int_id")
        lname = obs.get("lora_name")
        if isinstance(lid, bool) or not isinstance(lid, int):
            binding_disagreements.append(
                f"selection {label!r}: an observation carries lora_int_id {lid!r}, not an int")
            continue
        if lid == 0:
            if lname is not None:
                binding_disagreements.append(
                    f"selection {label!r}: an observation under lora id 0 (no LoRA) carries "
                    f"the adapter name {lname!r}")
        elif id_name.get(lid) != lname:
            binding_disagreements.append(
                f"selection {label!r}: an observation binds lora id {lid} to "
                f"{lname!r} but the readiness attestation binds id {lid} to "
                f"{id_name.get(lid)!r}")
        if isinstance(inconsistent, Mapping) and eid in inconsistent:
            by_label_inconsistent[label] = by_label_inconsistent.get(label, 0) + 1
        want = selection_adapter[label]
        want_id = 0 if want is None else name_id[want]
        if (lid, lname) != (want_id, want):
            by_label_misrouted.setdefault(label, []).append((lname, lid))
    if binding_disagreements:
        raise FidelityError(
            "multi-LoRA routing attestation FAILED: the engine's per-request lora id <-> name "
            f"binding disagrees with the readiness attestation on {len(binding_disagreements)} "
            f"request(s) — first: {binding_disagreements[0]}")
    if by_label_inconsistent:
        detail = ", ".join(f"{label!r}: {n}" for label, n in sorted(by_label_inconsistent.items()))
        raise FidelityError(
            "multi-LoRA routing attestation FAILED: request(s) recorded under TWO different "
            f"LoRA bindings across steps — {detail}; a request whose adapter changed mid-flight "
            "was not served under one selection")
    if by_label_misrouted:
        parts = []
        for label, got in sorted(by_label_misrouted.items()):
            want = selection_adapter[label]
            seen = sorted({(g if g is not None else "<no LoRA>") for g, _lid in got})
            parts.append(
                f"{len(got)}/{per_selection[label]['n']} request(s) for {label!r} ran under "
                f"{seen} instead of {want if want is not None else '<no LoRA>'!r}")
        raise FidelityError(
            "multi-LoRA routing attestation FAILED — the engine's LoRA bookkeeping shows "
            "request(s) served under an adapter other than their selection's (base must run "
            "under NO LoRA, the zero adapter under its own name: no base <-> zero exemption "
            "structurally): " + "; ".join(parts))

    # MIXED STEPS over the control's OWN requests: joined engine id -> label
    label_of_engine = {eid: expected[rid] for rid, eid in engine_of.items()}
    n_steps = 0
    n_steps_own = 0
    n_steps_mixed = 0
    max_distinct = 0
    for step in steps:
        n_steps += 1
        own = {label_of_engine[str(e)] for e in step if str(e) in label_of_engine}
        if own:
            n_steps_own += 1
        if len(own) >= 2:
            n_steps_mixed += 1
        if len(own) > max_distinct:
            max_distinct = len(own)
    if n_steps_mixed == 0:
        raise FidelityError(
            "multi-LoRA routing attestation cannot be certified: no engine step scheduled >= 2 "
            f"distinct selections of THIS control's requests ({n_steps} step(s) recorded, "
            f"{n_steps_own} with any of this control's requests, max {max_distinct} distinct "
            "selection(s) per step) — every request was routed correctly, but never IN A MIXED "
            "BATCH, so the claim 'selection honored in a mixed batch' would be vacuous; refusing")
    return {
        "verdict": SELECTION_VERDICT_ROUTING_ATTESTED,
        "n_requests": len(expected),
        "n_observed": sum(v["n_observed"] for v in per_selection.values()),
        "n_missing": 0,
        "n_ambiguous": 0,
        "n_foreign": n_foreign,
        "n_inconsistent": 0,
        "n_misrouted": 0,
        "per_selection": per_selection,
        "id_join": {"rule": "exact, or the external id plus exactly one -[0-9a-f]{8} engine suffix "
                            "(serving_attestation.is_engine_id_of)",
                    "n_exact": n_exact, "n_suffixed": n_suffixed,
                    "n_engine_ids_recorded": len(observed)},
        "steps": {"n_steps": n_steps, "n_steps_with_own_requests": n_steps_own,
                  "n_steps_mixed": n_steps_mixed,
                  "max_distinct_selections_per_step": max_distinct,
                  "basis": "computed over this control's joined requests only"},
        "runner_kind": runner_kind, "runner_class": runner_class,
        "id_name_map": {str(k): v for k, v in sorted(id_name.items())},
        "exemption": None,
        "rule": ("every submitted request joined to exactly one engine request id and observed "
                 "under exactly its selection's lora id/name in the engine's own bookkeeping; "
                 ">= 1 step scheduling >= 2 distinct selections of this control's requests"),
    }


def _best_effort_disarm(disarm: Callable[[str], Mapping[str, Any]], epoch: str) -> str:
    """After an arm that refused (or answered another epoch): try to disarm
    whatever got armed under the epoch this control minted. The outcome is
    RECORDED in the refusal's text and never masks it."""
    try:
        out = disarm(epoch)
    except Exception as e:  # noqa: BLE001 — recorded, never raised over the arm refusal
        return f"failed ({type(e).__name__}: {e})"
    if isinstance(out, Mapping):
        return (f"ok (disarmed={out.get('disarmed')!r}, was_armed={out.get('was_armed')!r}, "
                f"restored={out.get('restored')!r})")
    return f"ok ({type(out).__name__})"


def run_multi_lora_routing_control(
    *, base: str, adapters: Sequence[str], zero_adapter: str, prompt_set: Sequence[Any],
    mixed_batch: Callable[[Sequence[Tuple[str, Any]]], Sequence[Tuple[Sequence[int], str]]],
    arm: Callable[[str], Mapping[str, Any]],
    observe: Callable[[str], Mapping[str, Any]],
    disarm: Callable[[str], Mapping[str, Any]],
    selection_adapter: Mapping[str, Optional[str]],
    attested_id_name_map: Mapping[str, str],
    solo: Optional[Callable[[str, Sequence[Any]], Sequence[int]]] = None,
    output_diagnostics: bool = True,
) -> Dict[str, Any]:
    """The lora-native ``selection_honored`` control, end to end: mint a
    FRESH EPOCH id and ``arm(epoch)`` the worker's routing recorder (every
    worker adopts it; an arm refusal gets a best-effort disarm with that id,
    recorded in the refusal) -> submit the references
    (base twice) and the rotated mixed layouts through ``mixed_batch`` (each
    row returning ``(token_ids, response_id)``) -> ``observe(epoch)`` the
    whole epoch -> :func:`check_multi_lora_routing_attested` (the join to the
    response ids, the verdict, the mixed-step count over this control's own
    requests) -> the record, with the output-space numbers under
    ``numerics`` -> ``disarm(epoch)`` in a ``finally``. A refusal anywhere propagates as :class:`FidelityError`
    (the standup files it as ``controls_failed: selection_honored``, the gate
    as a GateError). A disarm that fails AFTER a pass refuses too: a recorder
    left armed on a production server is not the "never armed outside the
    control" the design states, so the control does not pass on it (a
    disarm failure after a refusal never masks the refusal — the record
    carries both).

    ``output_diagnostics=False`` (a residual-bearing standup never measures
    greedy-token agreement) submits the rotated mixed layouts ALONE through
    :func:`routing_only_expected` — every selection at every position, each
    engine request id labelled with its selection — and NEVER calls
    :func:`multi_lora_output_diagnostics`: no reference batches, no repeat
    floor, no output comparison; the record then carries NO ``numerics``
    block and ``output_diagnostics: False``. The structural verdict is the
    same check over the same kind of evidence."""
    # THE CLIENT MINTS THE EPOCH (serving core): every worker of
    # the process group adopts one id, so the launcher's per-worker reader sees agreeing
    # payloads; a per-worker mint refused there BEFORE this try/finally and left every
    # worker armed with no epoch able to disarm them
    epoch = uuid.uuid4().hex
    try:
        armed = arm(epoch)
    except Exception as arm_err:  # noqa: BLE001 — an arm that failed part-way may have wrapped some workers
        cleanup = _best_effort_disarm(disarm, epoch)
        raise FidelityError(
            f"multi-LoRA routing attestation: arming the recorder failed "
            f"({type(arm_err).__name__}: {arm_err}); best-effort disarm with the minted epoch: "
            f"{cleanup}") from arm_err
    got_epoch = armed.get("epoch") if isinstance(armed, Mapping) else None
    if got_epoch != epoch:
        cleanup = _best_effort_disarm(disarm, epoch)
        raise FidelityError(
            "multi-LoRA routing attestation: the recorder armed under another epoch than this "
            f"control minted (got {type(got_epoch).__name__}) — its evidence cannot be bound to "
            f"this control; best-effort disarm with the minted epoch: {cleanup}")
    disarm_error: Optional[BaseException] = None
    disarmed: Any = None
    try:
        if output_diagnostics:
            numerics = multi_lora_output_diagnostics(
                base=base, adapters=adapters, prompt_set=prompt_set, mixed_batch=mixed_batch,
                solo=solo, zero_adapter=zero_adapter, request_ids=True)
            expected = numerics.pop("expected_routing")
        else:
            numerics = None
            expected = routing_only_expected(
                base=base, adapters=adapters, prompt_set=prompt_set, mixed_batch=mixed_batch,
                zero_adapter=zero_adapter)
        payload = observe(epoch)
        if not isinstance(payload, Mapping):
            raise FidelityError(
                f"multi-LoRA routing attestation: the recorder answered "
                f"{type(payload).__name__}, not its payload; refusing")
        if payload.get("epoch") != epoch:
            raise FidelityError(
                "multi-LoRA routing attestation: the recorder answered for another epoch than "
                "this control armed; refusing")
        structural = check_multi_lora_routing_attested(
            expected=expected, selection_adapter=selection_adapter,
            observed=payload.get("observations") or {},
            inconsistent=payload.get("inconsistent") or {},
            steps=payload.get("steps") if payload.get("steps") is not None else [],
            attested_id_name_map=attested_id_name_map,
            overflow=bool(payload.get("overflow", False)),
            steps_overflow=bool(payload.get("steps_overflow", False)),
            recorder_errors=list(payload.get("errors") or ()),
            runner_kind=payload.get("runner_kind"), runner_class=payload.get("runner_class"))
    finally:
        try:
            disarmed = disarm(epoch)
        except Exception as e:  # noqa: BLE001 — never masks the body's refusal
            disarm_error = e
    if disarm_error is not None:
        raise FidelityError(
            f"multi-LoRA routing attestation: the recorder could not be DISARMED after the "
            f"control ({type(disarm_error).__name__}: {disarm_error}) — a recorder left armed "
            "is not the zero-overhead production state; refusing") from disarm_error
    out: Dict[str, Any] = {
        "passed": True,
        **structural,
        "epoch": epoch,
        "policy": MULTI_LORA_POLICY_NATIVE,
        "base": base, "adapters": list(adapters), "zero_adapter": zero_adapter,
        "genuine": [a for a in adapters if a != zero_adapter],
        "selection_adapter": {k: v for k, v in selection_adapter.items()},
        "recorder": {"arm": dict(armed) if isinstance(armed, Mapping) else armed,
                     "disarm": dict(disarmed) if isinstance(disarmed, Mapping) else disarmed,
                     "recorder_health": {k: payload.get(k) for k in
                                         ("armed", "epoch", "runner_class", "runner_kind",
                                          "n_observed", "n_inconsistent", "n_steps", "overflow",
                                          "steps_overflow", "snapshot_errors", "n_workers")}},
        "output_diagnostics": bool(output_diagnostics),
    }
    if output_diagnostics:
        out["numerics"] = numerics
    return out


def routing_only_expected(*, base: str, adapters: Sequence[str], prompt_set: Sequence[Any],
                          mixed_batch: Callable[[Sequence[Tuple[str, Any]]], Sequence[Any]],
                          zero_adapter: Optional[str]) -> Dict[str, str]:
    """The structural control's ``{engine request id: selection}`` map WITHOUT
    the output-space diagnostics: the rotated mixed layouts of
    :func:`_rotated_layouts` alone — every selection at every position, one
    concurrent submission per layout, each row's engine request id labelled
    with the selection it was submitted under. Nothing is compared: no
    reference batches, no repeat floor, no agreement statistic. The same
    argument refusals as :func:`multi_lora_output_diagnostics`; a row without
    an id or a duplicate id across batches refuses (a broken instrument)."""
    _policy, names, _genuine, prompts = _selection_args(
        adapters=adapters, prompt_set=prompt_set, base=base, zero_adapter=zero_adapter)
    P = len(prompts)
    expected: Dict[str, str] = {}
    for r, layout in enumerate(_rotated_layouts(names, P)):
        batch = [(s_name, prompts[i]) for s_name, i in layout]
        _rows, ids = _rows_with_ids(mixed_batch(batch), len(batch), f"mixed batch (layout {r})")
        for (sel, _p), rid in zip(batch, ids):
            if rid in expected:
                raise FidelityError(
                    f"mixed batch (layout {r}): an engine request id was already issued by an "
                    f"earlier batch of this control ({rid!r}) — the engine mints one id per request, "
                    "so a repeat is a broken instrument; refusing")
            expected[rid] = sel
    return expected
